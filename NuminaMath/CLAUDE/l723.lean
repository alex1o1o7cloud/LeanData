import Mathlib

namespace odd_symmetric_function_sum_l723_72385

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is symmetric about x=2 if f(2+x) = f(2-x) for all x -/
def IsSymmetricAbout2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem odd_symmetric_function_sum (f : ℝ → ℝ) 
    (h_odd : IsOdd f) 
    (h_sym : IsSymmetricAbout2 f) 
    (h_f2 : f 2 = 2018) : 
  f 2018 + f 2016 = 2018 := by
  sorry

end odd_symmetric_function_sum_l723_72385


namespace similar_triangles_leg_sum_l723_72361

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a^2 + b^2 = 64 →
  (1/2) * a * b = 10 →
  c^2 + d^2 = (5*8)^2 →
  (1/2) * c * d = 250 →
  c + d = 51 := by
sorry

end similar_triangles_leg_sum_l723_72361


namespace line_equation_correct_l723_72345

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def point_on_line (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a vector is parallel to a line --/
def vector_parallel_to_line (v : Vector2D) (l : Line2D) : Prop :=
  l.a * v.x + l.b * v.y = 0

/-- The line we're considering --/
def line_l : Line2D :=
  { a := 1, b := 2, c := -1 }

/-- The point A --/
def point_A : Point2D :=
  { x := 1, y := 0 }

/-- The direction vector of line l --/
def direction_vector : Vector2D :=
  { x := 2, y := -1 }

theorem line_equation_correct :
  point_on_line line_l point_A ∧
  vector_parallel_to_line direction_vector line_l :=
sorry

end line_equation_correct_l723_72345


namespace negation_equivalence_l723_72367

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) :=
by sorry

end negation_equivalence_l723_72367


namespace modulo_23_equivalence_l723_72377

theorem modulo_23_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 58294 ≡ n [ZMOD 23] ∧ n = 12 := by
  sorry

end modulo_23_equivalence_l723_72377


namespace optimal_rental_income_l723_72303

/-- Represents a travel agency's room rental scenario -/
structure RentalScenario where
  initialRooms : ℕ
  initialRate : ℕ
  rateIncrement : ℕ
  occupancyDecrease : ℕ

/-- Calculates the total daily rental income for a given rate increase -/
def totalIncome (scenario : RentalScenario) (rateIncrease : ℕ) : ℕ :=
  let newRate := scenario.initialRate + rateIncrease
  let newOccupancy := scenario.initialRooms - (rateIncrease / scenario.rateIncrement) * scenario.occupancyDecrease
  newRate * newOccupancy

/-- Finds the optimal rate increase to maximize total daily rental income -/
def optimalRateIncrease (scenario : RentalScenario) : ℕ :=
  sorry

/-- Calculates the increase in total daily rental income -/
def incomeIncrease (scenario : RentalScenario) : ℕ :=
  totalIncome scenario (optimalRateIncrease scenario) - totalIncome scenario 0

/-- Theorem stating the optimal rate increase and income increase for the given scenario -/
theorem optimal_rental_income (scenario : RentalScenario) 
  (h1 : scenario.initialRooms = 120)
  (h2 : scenario.initialRate = 50)
  (h3 : scenario.rateIncrement = 5)
  (h4 : scenario.occupancyDecrease = 6) :
  optimalRateIncrease scenario = 25 ∧ incomeIncrease scenario = 750 := by
  sorry

end optimal_rental_income_l723_72303


namespace quadratic_equation_single_solution_sum_l723_72352

theorem quadratic_equation_single_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + b₁ * x + 12 * x + 11 = 0 → (∀ y : ℝ, 3 * y^2 + b₁ * y + 12 * y + 11 = 0 → x = y)) ∧
  (∀ x : ℝ, 3 * x^2 + b₂ * x + 12 * x + 11 = 0 → (∀ y : ℝ, 3 * y^2 + b₂ * y + 12 * y + 11 = 0 → x = y)) ∧
  (∃ x : ℝ, 3 * x^2 + b₁ * x + 12 * x + 11 = 0) ∧
  (∃ x : ℝ, 3 * x^2 + b₂ * x + 12 * x + 11 = 0) ∧
  (b₁ ≠ b₂) →
  b₁ + b₂ = -24 := by
sorry

end quadratic_equation_single_solution_sum_l723_72352


namespace parabola_equation_l723_72330

/-- Prove that for a parabola y^2 = 2px with p > 0, if there exists a point M(3, y) on the parabola
    such that the distance from M to the focus F(p/2, 0) is 5, then p = 4. -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) : 
  (∃ y : ℝ, y^2 = 2*p*3 ∧ (3 - p/2)^2 + y^2 = 5^2) → p = 4 := by
  sorry

end parabola_equation_l723_72330


namespace paper_folding_l723_72323

theorem paper_folding (paper_area : Real) (folded_point_distance : Real) : 
  paper_area = 18 →
  folded_point_distance = 2 * Real.sqrt 6 →
  ∃ (side_length : Real) (folded_leg : Real),
    side_length ^ 2 = paper_area ∧
    folded_leg ^ 2 = 12 ∧
    folded_point_distance ^ 2 = 2 * folded_leg ^ 2 := by
  sorry

end paper_folding_l723_72323


namespace angle_sum_around_point_l723_72370

theorem angle_sum_around_point (y : ℝ) (h : y > 0) : 
  6 * y + 3 * y + 4 * y + 2 * y + y + 5 * y = 360 → y = 120 / 7 := by
  sorry

end angle_sum_around_point_l723_72370


namespace unknown_blanket_rate_l723_72348

theorem unknown_blanket_rate (blanket_count_1 blanket_count_2 blanket_count_3 : ℕ)
  (price_1 price_2 average_price : ℚ) (unknown_rate : ℚ) :
  blanket_count_1 = 4 →
  blanket_count_2 = 5 →
  blanket_count_3 = 2 →
  price_1 = 100 →
  price_2 = 150 →
  average_price = 150 →
  (blanket_count_1 * price_1 + blanket_count_2 * price_2 + blanket_count_3 * unknown_rate) / 
    (blanket_count_1 + blanket_count_2 + blanket_count_3) = average_price →
  unknown_rate = 250 := by
sorry


end unknown_blanket_rate_l723_72348


namespace original_number_proof_l723_72397

theorem original_number_proof (N : ℤ) : (N + 1) % 25 = 0 → N = 24 := by
  sorry

end original_number_proof_l723_72397


namespace min_moves_to_monochrome_l723_72340

/-- A move on a checkerboard that inverts colors in a rectangle -/
structure Move where
  top_left : Nat × Nat
  bottom_right : Nat × Nat

/-- A checkerboard with m rows and n columns -/
structure Checkerboard (m n : Nat) where
  board : Matrix (Fin m) (Fin n) Bool

/-- The result of applying a move to a checkerboard -/
def apply_move (board : Checkerboard m n) (move : Move) : Checkerboard m n :=
  sorry

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Check if a checkerboard is monochrome -/
def is_monochrome (board : Checkerboard m n) : Prop :=
  sorry

/-- The theorem stating the minimum number of moves required -/
theorem min_moves_to_monochrome (m n : Nat) :
  ∃ (moves : MoveSequence),
    (∀ (board : Checkerboard m n),
      is_monochrome (moves.foldl apply_move board)) ∧
    moves.length = Nat.floor (n / 2) + Nat.floor (m / 2) ∧
    (∀ (other_moves : MoveSequence),
      (∀ (board : Checkerboard m n),
        is_monochrome (other_moves.foldl apply_move board)) →
      other_moves.length ≥ moves.length) :=
  sorry

end min_moves_to_monochrome_l723_72340


namespace prime_pairs_dividing_sum_of_powers_l723_72358

theorem prime_pairs_dividing_sum_of_powers (p q : Nat) : 
  Nat.Prime p ∧ Nat.Prime q → (p * q ∣ (5^p + 5^q)) ↔ 
    ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ 
     (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) ∨ 
     (p = 5 ∧ q = 5) ∨ (p = 5 ∧ q = 313) ∨ 
     (p = 313 ∧ q = 5)) := by
  sorry

end prime_pairs_dividing_sum_of_powers_l723_72358


namespace tim_website_earnings_l723_72363

/-- Calculates Tim's earnings from his website for a week -/
def website_earnings (
  daily_visitors : ℕ)  -- Number of visitors per day for the first 6 days
  (days : ℕ)            -- Number of days with constant visitors
  (last_day_multiplier : ℕ)  -- Multiplier for visitors on the last day
  (earnings_per_visit : ℚ)  -- Earnings per visit in dollars
  : ℚ :=
  let first_days_visitors := daily_visitors * days
  let last_day_visitors := first_days_visitors * last_day_multiplier
  let total_visitors := first_days_visitors + last_day_visitors
  (total_visitors : ℚ) * earnings_per_visit

/-- Theorem stating Tim's earnings for the week -/
theorem tim_website_earnings :
  website_earnings 100 6 2 (1/100) = 18 := by
  sorry

end tim_website_earnings_l723_72363


namespace evaluate_expression_l723_72312

theorem evaluate_expression : (2 * 4 * 6) * (1/2 + 1/4 + 1/6) = 44 := by
  sorry

end evaluate_expression_l723_72312


namespace pens_cost_gained_l723_72356

/-- Represents the number of pens sold -/
def pens_sold : ℕ := 95

/-- Represents the gain percentage as a fraction -/
def gain_percentage : ℚ := 20 / 100

/-- Calculates the selling price given the cost price and gain percentage -/
def selling_price (cost : ℚ) : ℚ := cost * (1 + gain_percentage)

/-- Theorem stating that the number of pens' cost gained is 19 -/
theorem pens_cost_gained : 
  ∃ (cost : ℚ), cost > 0 ∧ 
  (pens_sold * (selling_price cost - cost) = 19 * cost) := by
  sorry

end pens_cost_gained_l723_72356


namespace overlapping_squares_theorem_l723_72374

/-- Represents a rectangle with numbers placed inside it -/
structure NumberedRectangle where
  width : ℕ
  height : ℕ
  numbers : List ℕ

/-- Represents the result of rotating a NumberedRectangle by 180° -/
def rotate180 (nr : NumberedRectangle) : NumberedRectangle :=
  { width := nr.width,
    height := nr.height,
    numbers := [6, 1, 2, 1] }

/-- Calculates the number of overlapping shaded squares when a NumberedRectangle is overlaid with its 180° rotation -/
def overlappingSquares (nr : NumberedRectangle) : ℕ :=
  nr.width * nr.height - 10

/-- The main theorem to be proved -/
theorem overlapping_squares_theorem (nr : NumberedRectangle) :
  nr.width = 8 ∧ nr.height = 5 ∧ nr.numbers = [1, 2, 1, 9] →
  rotate180 nr = { width := 8, height := 5, numbers := [6, 1, 2, 1] } →
  overlappingSquares nr = 30 := by
  sorry

#check overlapping_squares_theorem

end overlapping_squares_theorem_l723_72374


namespace quadratic_one_zero_l723_72383

def f (x : ℝ) := x^2 - 4*x + 4

theorem quadratic_one_zero :
  ∃! x, f x = 0 :=
by sorry

end quadratic_one_zero_l723_72383


namespace pure_imaginary_ratio_l723_72342

theorem pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = y * Complex.I) : 
  p / q = -4 / 3 := by
sorry

end pure_imaginary_ratio_l723_72342


namespace last_score_entered_l723_72393

def scores : List ℕ := [60, 65, 70, 75, 80, 85, 95]

theorem last_score_entered (s : ℕ) :
  s ∈ scores →
  (s = 80 ↔ (List.sum scores - s) % 6 = 0 ∧
    ∀ t ∈ scores, t ≠ s → (List.sum scores - t) % 6 ≠ 0) :=
by sorry

end last_score_entered_l723_72393


namespace wire_attachment_point_existence_l723_72349

theorem wire_attachment_point_existence :
  ∃! x : ℝ, 0 < x ∧ x < 5 ∧ Real.sqrt (x^2 + 3.6^2) + Real.sqrt ((x + 5)^2 + 3.6^2) = 13 := by
  sorry

end wire_attachment_point_existence_l723_72349


namespace candy_distribution_l723_72341

theorem candy_distribution (total_candy : Nat) (num_people : Nat) : 
  total_candy = 30 → num_people = 5 → 
  (∃ (pieces_per_person : Nat), total_candy = pieces_per_person * num_people) → 
  0 = total_candy - (total_candy / num_people) * num_people :=
by sorry

end candy_distribution_l723_72341


namespace archimedes_segment_theorem_l723_72381

/-- Archimedes' Theorem applied to segments -/
theorem archimedes_segment_theorem 
  (b c : ℝ) 
  (CT AK CK AT AB AC : ℝ) 
  (h1 : CT = AK) 
  (h2 : CK = AK + AB) 
  (h3 : AT = CK) 
  (h4 : AC = b) : 
  AT = (b + c) / 2 ∧ CT = (b - c) / 2 := by
  sorry

#check archimedes_segment_theorem

end archimedes_segment_theorem_l723_72381


namespace tims_cards_l723_72347

theorem tims_cards (ben_initial : ℕ) (ben_bought : ℕ) (tim : ℕ) : 
  ben_initial = 37 →
  ben_bought = 3 →
  ben_initial + ben_bought = 2 * tim →
  tim = 20 := by
sorry

end tims_cards_l723_72347


namespace profit_without_discount_is_fifty_percent_l723_72351

/-- Represents the profit percentage and discount percentage as ratios -/
structure ProfitDiscount where
  profit : ℚ
  discount : ℚ

/-- Calculates the profit percentage without discount given the profit percentage with discount -/
def profit_without_discount (pd : ProfitDiscount) : ℚ :=
  (1 + pd.profit) / (1 - pd.discount) - 1

/-- Theorem stating that a 42.5% profit with a 5% discount results in a 50% profit without discount -/
theorem profit_without_discount_is_fifty_percent :
  let pd : ProfitDiscount := { profit := 425/1000, discount := 5/100 }
  profit_without_discount pd = 1/2 := by
sorry

end profit_without_discount_is_fifty_percent_l723_72351


namespace concatenated_numbers_divisibility_l723_72331

def concatenate_numbers (n : ℕ) : ℕ :=
  sorry

theorem concatenated_numbers_divisibility (n : ℕ) :
  ¬(3 ∣ concatenate_numbers n) ↔ n % 3 = 1 := by
  sorry

end concatenated_numbers_divisibility_l723_72331


namespace tangent_slope_implies_tan_value_l723_72302

open Real

noncomputable def f (x : ℝ) : ℝ := (1/2) * x - (1/4) * sin x - (Real.sqrt 3 / 4) * cos x

theorem tangent_slope_implies_tan_value (x₀ : ℝ) :
  (deriv f x₀ = 1) → tan x₀ = -Real.sqrt 3 := by
  sorry

end tangent_slope_implies_tan_value_l723_72302


namespace expression_value_l723_72388

theorem expression_value (p q : ℚ) (h : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := by
  sorry

end expression_value_l723_72388


namespace monotonicity_condition_l723_72329

/-- A function f is monotonically increasing on an interval [a, +∞) if for all x, y in the interval with x < y, f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → f x < f y

/-- The function f(x) = kx^2 + (3k-2)x - 5 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (3*k - 2) * x - 5

theorem monotonicity_condition (k : ℝ) :
  (MonotonicallyIncreasing (f k) 1) ↔ k ≥ 2/5 := by sorry

end monotonicity_condition_l723_72329


namespace cloth_selling_price_l723_72382

/-- Given a cloth with the following properties:
  * Total length: 60 meters
  * Cost price per meter: 128 Rs
  * Profit per meter: 12 Rs
  Prove that the total selling price is 8400 Rs. -/
theorem cloth_selling_price 
  (total_length : ℕ) 
  (cost_price_per_meter : ℕ) 
  (profit_per_meter : ℕ) 
  (h1 : total_length = 60)
  (h2 : cost_price_per_meter = 128)
  (h3 : profit_per_meter = 12) :
  (cost_price_per_meter + profit_per_meter) * total_length = 8400 := by
  sorry

end cloth_selling_price_l723_72382


namespace student_rabbit_difference_l723_72364

/-- Proves that in 5 classrooms, where each classroom has 24 students and 3 rabbits,
    the difference between the total number of students and the total number of rabbits is 105. -/
theorem student_rabbit_difference :
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 24
  let rabbits_per_classroom : ℕ := 3
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  total_students - total_rabbits = 105 := by
sorry


end student_rabbit_difference_l723_72364


namespace dwarf_truth_count_l723_72376

/-- Represents the number of dwarfs who tell the truth -/
def truthful_dwarfs : ℕ := sorry

/-- Represents the number of dwarfs who lie -/
def lying_dwarfs : ℕ := sorry

/-- The total number of dwarfs -/
def total_dwarfs : ℕ := 10

/-- The number of dwarfs who raised their hands for vanilla ice cream -/
def vanilla_hands : ℕ := 10

/-- The number of dwarfs who raised their hands for chocolate ice cream -/
def chocolate_hands : ℕ := 5

/-- The number of dwarfs who raised their hands for fruit ice cream -/
def fruit_hands : ℕ := 1

theorem dwarf_truth_count :
  truthful_dwarfs + lying_dwarfs = total_dwarfs ∧
  truthful_dwarfs + 2 * lying_dwarfs = vanilla_hands + chocolate_hands + fruit_hands ∧
  truthful_dwarfs = 4 := by sorry

end dwarf_truth_count_l723_72376


namespace program_arrangements_l723_72310

theorem program_arrangements (n : ℕ) (k : ℕ) : 
  n = 4 → k = 2 → (n + 1) * (n + 2) = 30 := by
  sorry

end program_arrangements_l723_72310


namespace at_least_half_eligible_l723_72355

/-- Represents a team of sailors --/
structure Team where
  heights : List ℝ
  nonempty : heights ≠ []

/-- The median of a list of real numbers --/
def median (l : List ℝ) : ℝ := sorry

/-- The count of elements in a list satisfying a predicate --/
def count_if (l : List ℝ) (p : ℝ → Bool) : ℕ := sorry

theorem at_least_half_eligible (t : Team) (h_median : median t.heights = 167) :
  2 * (count_if t.heights (λ x => x ≤ 168)) ≥ t.heights.length := by sorry

end at_least_half_eligible_l723_72355


namespace square_prism_properties_l723_72300

/-- A right prism with a square base -/
structure SquarePrism where
  base_side : ℝ
  height : ℝ

/-- The lateral surface area of a square prism -/
def lateral_surface_area (p : SquarePrism) : ℝ := 4 * p.base_side * p.height

/-- The total surface area of a square prism -/
def surface_area (p : SquarePrism) : ℝ := 2 * p.base_side^2 + lateral_surface_area p

/-- The volume of a square prism -/
def volume (p : SquarePrism) : ℝ := p.base_side^2 * p.height

/-- Theorem about the surface area and volume of a specific square prism -/
theorem square_prism_properties :
  ∃ (p : SquarePrism), 
    lateral_surface_area p = 6^2 ∧ 
    surface_area p = 40.5 ∧ 
    volume p = 3.375 := by
  sorry


end square_prism_properties_l723_72300


namespace set_relations_imply_a_and_m_ranges_l723_72390

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 1 = 0}

-- State the theorem
theorem set_relations_imply_a_and_m_ranges :
  ∀ a m : ℝ,
  (A ∪ B a = A) →
  (A ∩ C m = C m) →
  ((a = 2 ∨ a = 3) ∧ (-2 < m ∧ m ≤ 2)) :=
by sorry

end set_relations_imply_a_and_m_ranges_l723_72390


namespace intersection_of_intervals_l723_72394

open Set

theorem intersection_of_intervals (A B : Set ℝ) :
  A = {x | -1 < x ∧ x < 2} →
  B = {x | 1 < x ∧ x < 3} →
  A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_intervals_l723_72394


namespace cubic_sum_theorem_l723_72365

theorem cubic_sum_theorem (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := by
  sorry

end cubic_sum_theorem_l723_72365


namespace disease_test_probability_l723_72318

theorem disease_test_probability (incidence_rate : ℝ) 
  (true_positive_rate : ℝ) (false_positive_rate : ℝ) :
  incidence_rate = 0.01 →
  true_positive_rate = 0.99 →
  false_positive_rate = 0.01 →
  let total_positive_rate := true_positive_rate * incidence_rate + 
    false_positive_rate * (1 - incidence_rate)
  (true_positive_rate * incidence_rate) / total_positive_rate = 0.5 := by
sorry

end disease_test_probability_l723_72318


namespace square_roots_problem_l723_72301

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : (x + 2)^2 = a) (h3 : (2*x - 5)^2 = a) : a = 9 := by
  sorry

end square_roots_problem_l723_72301


namespace thirty_switch_network_connections_l723_72353

/-- A network of switches with direct connections between them. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ

/-- The total number of connections in a switch network. -/
def total_connections (network : SwitchNetwork) : ℕ :=
  network.num_switches * network.connections_per_switch / 2

/-- Theorem: In a network of 30 switches, where each switch is directly
    connected to exactly 4 other switches, the total number of connections is 60. -/
theorem thirty_switch_network_connections :
  let network : SwitchNetwork := { num_switches := 30, connections_per_switch := 4 }
  total_connections network = 60 := by
  sorry


end thirty_switch_network_connections_l723_72353


namespace min_value_sum_l723_72320

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a + b ≥ 9 / (2 * a) + 2 / b) : 
  a + b ≥ 5 * Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y ≥ 9 / (2 * x) + 2 / y → x + y ≥ a + b :=
sorry

end min_value_sum_l723_72320


namespace power_of_two_equation_l723_72359

theorem power_of_two_equation (m : ℤ) : 
  2^2000 - 3 * 2^1999 + 2^1998 - 2^1997 + 2^1996 = m * 2^1996 → m = -5 := by
  sorry

end power_of_two_equation_l723_72359


namespace sprained_vs_normal_time_difference_l723_72313

/-- The time it takes Ann to frost a cake normally, in minutes -/
def normal_time : ℕ := 5

/-- The time it takes Ann to frost a cake with a sprained wrist, in minutes -/
def sprained_time : ℕ := 8

/-- The number of cakes Ann needs to frost -/
def num_cakes : ℕ := 10

/-- Theorem stating the difference in time to frost 10 cakes between sprained and normal conditions -/
theorem sprained_vs_normal_time_difference : 
  sprained_time * num_cakes - normal_time * num_cakes = 30 := by
  sorry

end sprained_vs_normal_time_difference_l723_72313


namespace exists_a_with_two_common_tangents_l723_72392

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Definition of circle C₂ -/
def C₂ (x y a : ℝ) : Prop := (x - 4)^2 + (y + a)^2 = 64

/-- Two circles have exactly two common tangents -/
def have_two_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : Prop := sorry

/-- Main theorem -/
theorem exists_a_with_two_common_tangents :
  ∃ a : ℕ, a > 0 ∧ have_two_common_tangents C₁ (C₂ · · a) :=
sorry

end exists_a_with_two_common_tangents_l723_72392


namespace train_length_train_length_proof_l723_72360

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_s

/-- Proof that a train's length is approximately 129.96 meters -/
theorem train_length_proof (speed_kmh : ℝ) (time_s : ℝ)
  (h1 : speed_kmh = 52)
  (h2 : time_s = 9) :
  ∃ ε > 0, |train_length speed_kmh time_s - 129.96| < ε :=
sorry

end train_length_train_length_proof_l723_72360


namespace remaining_sessions_proof_l723_72344

theorem remaining_sessions_proof (total_patients : Nat) (total_sessions : Nat) 
  (patient1_sessions : Nat) (extra_sessions : Nat) :
  total_patients = 4 →
  total_sessions = 25 →
  patient1_sessions = 6 →
  extra_sessions = 5 →
  total_sessions - (patient1_sessions + (patient1_sessions + extra_sessions)) = 8 :=
by
  sorry

end remaining_sessions_proof_l723_72344


namespace inequality_proof_l723_72322

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end inequality_proof_l723_72322


namespace circle_center_polar_coordinates_l723_72336

theorem circle_center_polar_coordinates :
  let ρ : ℝ → ℝ → ℝ := fun θ r ↦ r
  let circle_equation : ℝ → ℝ → Prop := fun θ r ↦ ρ θ r = Real.sqrt 2 * (Real.cos θ + Real.sin θ)
  let is_center : ℝ → ℝ → Prop := fun r θ ↦ ∀ θ' r', circle_equation θ' r' → 
    (r * Real.cos θ - r' * Real.cos θ')^2 + (r * Real.sin θ - r' * Real.sin θ')^2 = r^2
  is_center 1 (Real.pi / 4) := by sorry

end circle_center_polar_coordinates_l723_72336


namespace inequality_equivalence_l723_72379

theorem inequality_equivalence (x y : ℝ) : 
  (y - x < Real.sqrt (x^2 + 1)) ↔ (y < x + Real.sqrt (x^2 + 1)) := by
  sorry

end inequality_equivalence_l723_72379


namespace purple_position_correct_l723_72384

/-- The position of "PURPLE" in the alphabetized list of all its distinguishable rearrangements -/
def purple_position : ℕ := 226

/-- The word to be rearranged -/
def word : String := "PURPLE"

/-- The theorem stating that the position of "PURPLE" in the alphabetized list of all its distinguishable rearrangements is 226 -/
theorem purple_position_correct : 
  purple_position = 226 ∧ 
  word = "PURPLE" ∧
  purple_position = (List.filter (· ≤ word) (List.map String.mk (List.permutations word.data))).length :=
by sorry

end purple_position_correct_l723_72384


namespace shifted_cosine_to_sine_l723_72362

/-- Given a cosine function shifted to create an odd function, 
    prove the value at a specific point. -/
theorem shifted_cosine_to_sine (f g : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.cos (x / 2 - π / 3)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x, g x = f (x - φ)) →
  (∀ x, g x + g (-x) = 0) →
  g (2 * φ + π / 6) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end shifted_cosine_to_sine_l723_72362


namespace cosine_amplitude_l723_72371

/-- Given a cosine function y = a cos(bx) where a > 0 and b > 0,
    prove that a equals the maximum y-value of the graph. -/
theorem cosine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, ∃ y, y = a * Real.cos (b * x)) →
  (∃ M, M > 0 ∧ ∀ x, a * Real.cos (b * x) ≤ M) →
  (∀ ε > 0, ∃ x, a * Real.cos (b * x) > M - ε) →
  a = 3 :=
sorry

end cosine_amplitude_l723_72371


namespace equation_one_solutions_equation_two_solutions_equation_three_solutions_l723_72311

-- Equation 1: x^2 - 2x = 0
theorem equation_one_solutions (x : ℝ) : 
  (x = 0 ∨ x = 2) ↔ x^2 - 2*x = 0 := by sorry

-- Equation 2: (2x-1)^2 = (3-x)^2
theorem equation_two_solutions (x : ℝ) : 
  (x = -2 ∨ x = 4/3) ↔ (2*x - 1)^2 = (3 - x)^2 := by sorry

-- Equation 3: 3x(x-2) = x-2
theorem equation_three_solutions (x : ℝ) : 
  (x = 2 ∨ x = 1/3) ↔ 3*x*(x - 2) = x - 2 := by sorry

end equation_one_solutions_equation_two_solutions_equation_three_solutions_l723_72311


namespace transformed_triangle_area_l723_72387

-- Define the function f on the domain {x₁, x₂, x₃}
variable (f : ℝ → ℝ)
variable (x₁ x₂ x₃ : ℝ)

-- Define the area of a triangle given three points
def triangleArea (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem transformed_triangle_area 
  (h1 : triangleArea (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) = 27) :
  triangleArea (x₁/2, 3 * f x₁) (x₂/2, 3 * f x₂) (x₃/2, 3 * f x₃) = 40.5 :=
sorry

end transformed_triangle_area_l723_72387


namespace negation_of_implication_or_l723_72339

theorem negation_of_implication_or (p q r : Prop) :
  ¬(r → p ∨ q) ↔ (¬r → ¬p ∧ ¬q) := by sorry

end negation_of_implication_or_l723_72339


namespace only_valid_numbers_l723_72380

/-- A six-digit number starting with 523 that is divisible by 7, 8, and 9 -/
def validNumber (n : ℕ) : Prop :=
  523000 ≤ n ∧ n < 524000 ∧ 7 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n

/-- The theorem stating that 523656 and 523152 are the only valid numbers -/
theorem only_valid_numbers :
  ∀ n : ℕ, validNumber n ↔ n = 523656 ∨ n = 523152 :=
by sorry

end only_valid_numbers_l723_72380


namespace rectangle_width_from_square_l723_72375

theorem rectangle_width_from_square (square_side : ℝ) (rect_length : ℝ) :
  square_side = 12 →
  rect_length = 18 →
  4 * square_side = 2 * (rect_length + (4 * square_side - 2 * rect_length) / 2) →
  (4 * square_side - 2 * rect_length) / 2 = 6 := by
  sorry

end rectangle_width_from_square_l723_72375


namespace car_speed_second_hour_l723_72366

/-- Given a car traveling for two hours with an average speed of 75 km/h
    and a speed of 90 km/h in the first hour, prove that the speed in
    the second hour must be 60 km/h. -/
theorem car_speed_second_hour
  (average_speed : ℝ)
  (first_hour_speed : ℝ)
  (h_average : average_speed = 75)
  (h_first : first_hour_speed = 90)
  : (2 * average_speed - first_hour_speed) = 60 := by
  sorry

end car_speed_second_hour_l723_72366


namespace average_speed_of_trip_l723_72316

/-- Proves that the average speed of a 100-mile trip is 40 mph, given specific conditions -/
theorem average_speed_of_trip (total_distance : ℝ) (first_part_distance : ℝ) (second_part_distance : ℝ)
  (first_part_speed : ℝ) (second_part_speed : ℝ) (h1 : total_distance = 100)
  (h2 : first_part_distance = 30) (h3 : second_part_distance = 70)
  (h4 : first_part_speed = 60) (h5 : second_part_speed = 35)
  (h6 : total_distance = first_part_distance + second_part_distance) :
  (total_distance) / ((first_part_distance / first_part_speed) + (second_part_distance / second_part_speed)) = 40 := by
  sorry

end average_speed_of_trip_l723_72316


namespace coefficient_x9_eq_240_l723_72395

/-- The coefficient of x^9 in the expansion of (1+3x-2x^2)^5 -/
def coefficient_x9 : ℤ :=
  -- Define the coefficient here
  sorry

/-- Theorem stating that the coefficient of x^9 in (1+3x-2x^2)^5 is 240 -/
theorem coefficient_x9_eq_240 : coefficient_x9 = 240 := by
  sorry

end coefficient_x9_eq_240_l723_72395


namespace arithmetic_sequence_ninth_term_l723_72304

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_ninth_term :
  ∀ n : ℕ, arithmeticSequence 1 (-2) n = -15 → n = 9 := by
  sorry

end arithmetic_sequence_ninth_term_l723_72304


namespace billys_age_l723_72307

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 2 * joe) 
  (h2 : billy + joe = 45) : 
  billy = 30 := by
sorry

end billys_age_l723_72307


namespace min_value_fraction_l723_72328

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x - y + 2*z = 0) : 
  ∃ (m : ℝ), m = 8 ∧ ∀ k, k = y^2/(x*z) → k ≥ m :=
by
  sorry

end min_value_fraction_l723_72328


namespace all_lines_pass_through_point_common_point_is_neg_two_two_l723_72308

/-- A line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a, b, c form an arithmetic progression with common difference 3d -/
def is_ap (l : Line) (d : ℝ) : Prop :=
  l.b = l.a + 3 * d ∧ l.c = l.a + 6 * d

/-- Check if a point (x, y) lies on a line -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Theorem stating that all lines satisfying the arithmetic progression condition pass through (-2, 2) -/
theorem all_lines_pass_through_point (l : Line) (d : ℝ) :
  is_ap l d → point_on_line l (-2) 2 := by
  sorry

/-- Main theorem proving the common point is (-2, 2) -/
theorem common_point_is_neg_two_two :
  ∃ (x y : ℝ), ∀ (l : Line) (d : ℝ), is_ap l d → point_on_line l x y ∧ x = -2 ∧ y = 2 := by
  sorry

end all_lines_pass_through_point_common_point_is_neg_two_two_l723_72308


namespace integer_divisibility_equivalence_l723_72334

theorem integer_divisibility_equivalence (n : ℤ) : 
  (∃ a b : ℤ, 3 * n - 2 = 5 * a ∧ 2 * n + 1 = 7 * b) ↔ 
  (∃ k : ℤ, n = 35 * k + 24) := by
sorry

end integer_divisibility_equivalence_l723_72334


namespace circumscribed_circle_diameter_l723_72338

/-- The diameter of a triangle's circumscribed circle, given one side and its opposite angle -/
theorem circumscribed_circle_diameter 
  (side : ℝ) 
  (angle : ℝ) 
  (h1 : side = 15) 
  (h2 : angle = π / 4) : 
  side / Real.sin angle = 15 * Real.sqrt 2 := by
sorry

end circumscribed_circle_diameter_l723_72338


namespace expansion_coefficient_sum_l723_72305

theorem expansion_coefficient_sum (a : ℤ) (n : ℕ) : 
  (2^n = 64) → 
  ((1 + a)^6 = 729) → 
  (a = -4 ∨ a = 2) := by
  sorry

end expansion_coefficient_sum_l723_72305


namespace sammy_janine_bottle_cap_difference_l723_72332

/-- Proof that Sammy has 2 more bottle caps than Janine -/
theorem sammy_janine_bottle_cap_difference :
  ∀ (sammy janine billie : ℕ),
    sammy > janine →
    janine = 3 * billie →
    billie = 2 →
    sammy = 8 →
    sammy - janine = 2 := by
  sorry

end sammy_janine_bottle_cap_difference_l723_72332


namespace fred_savings_period_l723_72389

/-- The number of weeks Fred needs to save to buy the mountain bike -/
def weeks_to_save (bike_cost : ℕ) (birthday_money : ℕ) (weekly_earnings : ℕ) : ℕ :=
  (bike_cost - birthday_money) / weekly_earnings

theorem fred_savings_period :
  weeks_to_save 600 150 18 = 25 := by
  sorry

end fred_savings_period_l723_72389


namespace max_shaded_area_achievable_max_area_l723_72398

/-- Represents a rectangular picture frame made of eight identical trapezoids -/
structure PictureFrame where
  length : ℕ+
  width : ℕ+
  trapezoidArea : ℕ
  isPrime : Nat.Prime trapezoidArea

/-- Calculates the area of the shaded region in the picture frame -/
def shadedArea (frame : PictureFrame) : ℕ :=
  (frame.trapezoidArea - 1) * (3 * frame.trapezoidArea - 1)

/-- Theorem stating the maximum possible area of the shaded region -/
theorem max_shaded_area (frame : PictureFrame) :
  shadedArea frame < 2000 → shadedArea frame ≤ 1496 :=
by
  sorry

/-- Theorem proving that 1496 is achievable -/
theorem achievable_max_area :
  ∃ frame : PictureFrame, shadedArea frame = 1496 ∧ shadedArea frame < 2000 :=
by
  sorry

end max_shaded_area_achievable_max_area_l723_72398


namespace population_difference_l723_72378

/-- The population difference between thrice Willowdale and Roseville -/
theorem population_difference (willowdale roseville suncity : ℕ) : 
  willowdale = 2000 →
  suncity = 12000 →
  suncity = 2 * roseville + 1000 →
  roseville < 3 * willowdale →
  3 * willowdale - roseville = 500 := by
  sorry

end population_difference_l723_72378


namespace arc_length_calculation_l723_72396

theorem arc_length_calculation (r : ℝ) (θ_central : ℝ) (θ_peripheral : ℝ) :
  r = 5 →
  θ_central = (2/3) * θ_peripheral →
  θ_peripheral = 2 * π →
  r * θ_central = (20 * π) / 3 :=
by sorry

end arc_length_calculation_l723_72396


namespace select_male_and_female_prob_l723_72319

/-- The probability of selecting one male and one female from a group of 2 females and 4 males -/
theorem select_male_and_female_prob (num_female : ℕ) (num_male : ℕ) : 
  num_female = 2 → num_male = 4 → 
  (num_male.choose 1 * num_female.choose 1 : ℚ) / ((num_male + num_female).choose 2) = 8 / 15 := by
  sorry

#check select_male_and_female_prob

end select_male_and_female_prob_l723_72319


namespace intersection_A_B_union_A_complement_B_l723_72309

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x + 1 < 5}
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for A ∪ (ℝ \ B)
theorem union_A_complement_B : A ∪ (Set.univ \ B) = Set.univ := by sorry

end intersection_A_B_union_A_complement_B_l723_72309


namespace perpendicular_line_theorem_l723_72373

/-- A figure in a plane -/
inductive PlaneFigure
  | Triangle
  | Trapezoid
  | CircleDiameters
  | HexagonSides

/-- Represents whether two lines in a figure are guaranteed to intersect -/
def guaranteed_intersection (figure : PlaneFigure) : Prop :=
  match figure with
  | PlaneFigure.Triangle => true
  | PlaneFigure.Trapezoid => false
  | PlaneFigure.CircleDiameters => true
  | PlaneFigure.HexagonSides => false

/-- A line perpendicular to two sides of a figure is perpendicular to the plane -/
def perpendicular_to_plane (figure : PlaneFigure) : Prop :=
  guaranteed_intersection figure

theorem perpendicular_line_theorem (figure : PlaneFigure) :
  perpendicular_to_plane figure ↔ (figure = PlaneFigure.Triangle ∨ figure = PlaneFigure.CircleDiameters) :=
by sorry

end perpendicular_line_theorem_l723_72373


namespace graduating_class_size_l723_72399

/-- The number of boys in the graduating class -/
def num_boys : ℕ := 208

/-- The difference between the number of girls and boys -/
def girl_boy_difference : ℕ := 69

/-- The total number of students in the graduating class -/
def total_students : ℕ := num_boys + (num_boys + girl_boy_difference)

theorem graduating_class_size :
  total_students = 485 :=
by sorry

end graduating_class_size_l723_72399


namespace total_distance_QY_l723_72368

/-- Proves that the total distance between Q and Y is 45 km --/
theorem total_distance_QY (matthew_speed johnny_speed : ℝ)
  (johnny_distance : ℝ) (time_difference : ℝ) :
  matthew_speed = 3 →
  johnny_speed = 4 →
  johnny_distance = 24 →
  time_difference = 1 →
  ∃ (total_distance : ℝ), total_distance = 45 :=
by
  sorry


end total_distance_QY_l723_72368


namespace upper_limit_n_l723_72357

def is_integer (x : ℚ) : Prop := ∃ k : ℤ, x = k

def has_exactly_three_prime_factors (n : ℕ) : Prop :=
  ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  n = p * q * r

theorem upper_limit_n :
  ∀ n : ℕ, n > 0 →
  is_integer (14 * n / 60) →
  has_exactly_three_prime_factors n →
  n ≤ 210 :=
sorry

end upper_limit_n_l723_72357


namespace modulus_of_z_l723_72343

def i : ℂ := Complex.I

theorem modulus_of_z (z : ℂ) (h : z / (1 + i) = 1 - 2*i) : Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_of_z_l723_72343


namespace even_function_negative_domain_l723_72325

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_negative_domain
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_pos : ∀ x ≥ 0, f x = x^3 + x) :
  ∀ x < 0, f x = -x^3 - x :=
sorry

end even_function_negative_domain_l723_72325


namespace distance_IP_equals_half_R_minus_r_l723_72321

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the special points of the triangle
variable (I O G H P : EuclideanSpace ℝ (Fin 2))

-- Define the radii
variable (r R : ℝ)

-- Assumptions
variable (h_incenter : is_incenter I A B C)
variable (h_circumcenter : is_circumcenter O A B C)
variable (h_centroid : is_centroid G A B C)
variable (h_orthocenter : is_orthocenter H A B C)
variable (h_nine_point : is_nine_point_center P A B C)
variable (h_inradius : is_inradius r A B C)
variable (h_circumradius : is_circumradius R A B C)

-- Theorem statement
theorem distance_IP_equals_half_R_minus_r :
  dist I P = R / 2 - r :=
sorry

end distance_IP_equals_half_R_minus_r_l723_72321


namespace expression_simplification_l723_72369

theorem expression_simplification (m n : ℝ) (hm : m ≠ 0) :
  (m^(4/3) - 27 * m^(1/3) * n) / (m^(2/3) + 3 * (m*n)^(1/3) + 9 * n^(2/3)) / (1 - 3 * (n/m)^(1/3)) - m^(2/3) = 0 :=
by sorry

end expression_simplification_l723_72369


namespace number_of_divisors_of_36_l723_72333

theorem number_of_divisors_of_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end number_of_divisors_of_36_l723_72333


namespace binomial_10_3_l723_72354

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l723_72354


namespace flower_shop_problem_l723_72386

/-- Given information about flower purchases and sales, prove the cost price of the first batch and minimum selling price of the second batch -/
theorem flower_shop_problem (first_batch_cost second_batch_cost : ℝ) 
  (quantity_ratio : ℝ) (price_difference : ℝ) (min_total_profit : ℝ) 
  (first_batch_selling_price : ℝ) :
  first_batch_cost = 1000 →
  second_batch_cost = 2500 →
  quantity_ratio = 2 →
  price_difference = 0.5 →
  min_total_profit = 1500 →
  first_batch_selling_price = 3 →
  ∃ (first_batch_cost_price second_batch_min_selling_price : ℝ),
    first_batch_cost_price = 2 ∧
    second_batch_min_selling_price = 3.5 ∧
    (first_batch_cost / first_batch_cost_price) * quantity_ratio = 
      second_batch_cost / (first_batch_cost_price + price_difference) ∧
    (first_batch_cost / first_batch_cost_price) * 
      (first_batch_selling_price - first_batch_cost_price) +
    (second_batch_cost / (first_batch_cost_price + price_difference)) * 
      (second_batch_min_selling_price - (first_batch_cost_price + price_difference)) ≥ 
    min_total_profit := by
  sorry

end flower_shop_problem_l723_72386


namespace binary_101101_equals_base5_140_l723_72350

/-- Converts a binary number to decimal --/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to base 5 --/
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_101101_equals_base5_140 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true, false, true]) = [1, 4, 0] :=
sorry

end binary_101101_equals_base5_140_l723_72350


namespace regular_polygon_exterior_24_degrees_l723_72337

/-- Theorem: For a regular polygon with exterior angles measuring 24 degrees each,
    the number of sides is 15 and the sum of interior angles is 2340 degrees. -/
theorem regular_polygon_exterior_24_degrees :
  ∀ (n : ℕ) (exterior_angle : ℝ),
  exterior_angle = 24 →
  n * exterior_angle = 360 →
  n = 15 ∧ (n - 2) * 180 = 2340 := by
  sorry

end regular_polygon_exterior_24_degrees_l723_72337


namespace marble_distribution_solution_l723_72317

/-- Represents the distribution of marbles among three boys -/
structure MarbleDistribution where
  ben : ℕ
  adam : ℕ
  chris : ℕ

/-- Checks if a given marble distribution satisfies the problem conditions -/
def is_valid_distribution (d : MarbleDistribution) : Prop :=
  d.adam = 2 * d.ben ∧
  d.chris = d.ben + 5 ∧
  d.ben + d.adam + d.chris = 73

/-- The theorem stating the correct distribution of marbles -/
theorem marble_distribution_solution :
  ∃ (d : MarbleDistribution), is_valid_distribution d ∧
    d.ben = 17 ∧ d.adam = 34 ∧ d.chris = 22 := by
  sorry

end marble_distribution_solution_l723_72317


namespace total_money_calculation_l723_72324

/-- Proves that the total amount of money is Rs 3000 given the specified conditions -/
theorem total_money_calculation (part1 part2 total interest_rate1 interest_rate2 total_interest : ℝ) :
  part1 = 300 →
  interest_rate1 = 0.03 →
  interest_rate2 = 0.05 →
  total_interest = 144 →
  total = part1 + part2 →
  part1 * interest_rate1 + part2 * interest_rate2 = total_interest →
  total = 3000 := by
  sorry

end total_money_calculation_l723_72324


namespace equation_solution_l723_72335

theorem equation_solution (a b : ℝ) : 
  a^2 + b^2 + 2*a - 4*b + 5 = 0 → 2*a^2 + 4*b - 3 = 7 := by
  sorry

end equation_solution_l723_72335


namespace distance_to_reflection_distance_D_to_D_l723_72346

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection (x y : ℝ) : 
  let D : ℝ × ℝ := (x, y)
  let D' : ℝ × ℝ := (x, -y)
  Real.sqrt ((D.1 - D'.1)^2 + (D.2 - D'.2)^2) = 2 * abs y := by
  sorry

/-- The specific case for point D(2, 4) --/
theorem distance_D_to_D'_reflection : 
  let D : ℝ × ℝ := (2, 4)
  let D' : ℝ × ℝ := (2, -4)
  Real.sqrt ((D.1 - D'.1)^2 + (D.2 - D'.2)^2) = 8 := by
  sorry

end distance_to_reflection_distance_D_to_D_l723_72346


namespace longest_tennis_match_duration_l723_72306

theorem longest_tennis_match_duration (hours : ℕ) (minutes : ℕ) : 
  hours = 11 ∧ minutes = 5 → hours * 60 + minutes = 665 := by sorry

end longest_tennis_match_duration_l723_72306


namespace rectangle_area_stage_8_l723_72391

/-- The area of a rectangle formed by n squares, each measuring s by s units -/
def rectangleArea (n : ℕ) (s : ℝ) : ℝ := n * (s * s)

/-- Theorem: The area of a rectangle formed by 8 squares, each measuring 4 inches by 4 inches, is 128 square inches -/
theorem rectangle_area_stage_8 : rectangleArea 8 4 = 128 := by
  sorry

end rectangle_area_stage_8_l723_72391


namespace adiabatic_compression_work_l723_72326

theorem adiabatic_compression_work
  (k : ℝ) (p₁ V₁ V₂ : ℝ) (h_k : k > 1) (h_V : V₂ > 0) :
  let W := (p₁ * V₁) / (k - 1) * (1 - (V₁ / V₂) ^ (k - 1))
  let c := p₁ * V₁^k
  ∀ (p v : ℝ), p * v^k = c →
  W = -(∫ (x : ℝ) in V₁..V₂, c / x^k) :=
sorry

end adiabatic_compression_work_l723_72326


namespace tangent_line_polar_equation_l723_72314

/-- The polar coordinate equation of the tangent line to the circle ρ = 4sin θ
    that passes through the point (2√2, π/4) is ρ cos θ = 2. -/
theorem tangent_line_polar_equation (ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) →  -- Circle equation
  (∃ (ρ₀ θ₀ : ℝ), ρ₀ = 2 * Real.sqrt 2 ∧ θ₀ = π / 4 ∧ 
    ρ₀ * Real.cos θ₀ = 2 ∧ ρ₀ * Real.sin θ₀ = 2) →  -- Point (2√2, π/4)
  (ρ * Real.cos θ = 2) -- Tangent line equation
:= by sorry

end tangent_line_polar_equation_l723_72314


namespace inequality_equivalence_l723_72327

theorem inequality_equivalence (x : ℝ) : -9 < 2*x - 1 ∧ 2*x - 1 ≤ 6 → -4 < x ∧ x ≤ 3.5 := by
  sorry

end inequality_equivalence_l723_72327


namespace train_meeting_distance_l723_72372

theorem train_meeting_distance (route_length : ℝ) (time_x time_y : ℝ) 
  (h1 : route_length = 160)
  (h2 : time_x = 5)
  (h3 : time_y = 3)
  : let speed_x := route_length / time_x
    let speed_y := route_length / time_y
    let meeting_time := route_length / (speed_x + speed_y)
    speed_x * meeting_time = 60 := by
  sorry

end train_meeting_distance_l723_72372


namespace malfunctioning_odometer_theorem_l723_72315

/-- Converts a digit in the malfunctioning odometer system to its actual value -/
def convert_digit (d : Nat) : Nat :=
  if d < 4 then d else d + 2

/-- Converts an odometer reading to actual miles -/
def odometer_to_miles (reading : List Nat) : Nat :=
  reading.foldr (fun d acc => convert_digit d + 8 * acc) 0

/-- Theorem: The malfunctioning odometer reading 000306 corresponds to 134 miles -/
theorem malfunctioning_odometer_theorem :
  odometer_to_miles [0, 0, 0, 3, 0, 6] = 134 := by
  sorry

#eval odometer_to_miles [0, 0, 0, 3, 0, 6]

end malfunctioning_odometer_theorem_l723_72315
