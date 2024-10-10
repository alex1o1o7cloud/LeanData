import Mathlib

namespace company_female_managers_l3043_304300

/-- Represents the number of female managers in a company -/
def female_managers (total_employees : ℕ) (female_employees : ℕ) (male_employees : ℕ) : ℕ :=
  (2 * female_employees) / 5

theorem company_female_managers :
  let total_employees := female_employees + male_employees
  let female_employees := 625
  let total_managers := (2 * total_employees) / 5
  let male_managers := (2 * male_employees) / 5
  female_managers total_employees female_employees male_employees = 250 :=
by
  sorry

#check company_female_managers

end company_female_managers_l3043_304300


namespace particle_probability_theorem_l3043_304396

/-- Probability of hitting (0,0) first when starting from (x,y) -/
noncomputable def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

theorem particle_probability_theorem :
  ∃ (p : ℕ), p > 0 ∧ ¬(3 ∣ p) ∧ P 3 5 = p / 3^7 := by sorry

end particle_probability_theorem_l3043_304396


namespace fountain_length_is_105_l3043_304363

/-- Represents the water fountain construction scenario -/
structure FountainConstruction where
  initialMen : ℕ := 20
  initialLength : ℕ := 56
  initialDays : ℕ := 7
  wallDays : ℕ := 3
  newMen : ℕ := 35
  totalDays : ℕ := 9
  wallEfficiencyFactor : ℚ := 1/2

/-- Calculates the length of the fountain that can be built given the construction parameters -/
def calculateFountainLength (fc : FountainConstruction) : ℚ :=
  let workRatePerMan : ℚ := fc.initialLength / (fc.initialMen * fc.initialDays)
  let newWallDays : ℚ := fc.wallDays * fc.wallEfficiencyFactor
  let availableDaysForFountain : ℚ := fc.totalDays - newWallDays
  workRatePerMan * fc.newMen * availableDaysForFountain

theorem fountain_length_is_105 (fc : FountainConstruction) :
  calculateFountainLength fc = 105 := by
  sorry

end fountain_length_is_105_l3043_304363


namespace inverse_proportion_difference_positive_l3043_304369

theorem inverse_proportion_difference_positive 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₁ - y₂ > 0 := by
sorry

end inverse_proportion_difference_positive_l3043_304369


namespace vince_savings_l3043_304328

/-- Calculates Vince's savings given his earnings per customer, number of customers,
    fixed expenses, and percentage allocated for recreation. -/
def calculate_savings (earnings_per_customer : ℚ) (num_customers : ℕ) 
                      (fixed_expenses : ℚ) (recreation_percent : ℚ) : ℚ :=
  let total_earnings := earnings_per_customer * num_customers
  let recreation_amount := recreation_percent * total_earnings
  let total_expenses := fixed_expenses + recreation_amount
  total_earnings - total_expenses

/-- Proves that Vince's savings are $872 given the problem conditions. -/
theorem vince_savings : 
  calculate_savings 18 80 280 (20/100) = 872 := by
  sorry

end vince_savings_l3043_304328


namespace least_positive_integer_congruence_l3043_304370

theorem least_positive_integer_congruence : ∃! x : ℕ+, 
  (x : ℤ) + 3701 ≡ 1580 [ZMOD 15] ∧ 
  (x : ℤ) ≡ 7 [ZMOD 9] ∧
  ∀ y : ℕ+, ((y : ℤ) + 3701 ≡ 1580 [ZMOD 15] ∧ (y : ℤ) ≡ 7 [ZMOD 9]) → x ≤ y :=
by
  sorry

end least_positive_integer_congruence_l3043_304370


namespace problem_solution_l3043_304307

theorem problem_solution (x : ℝ) 
  (h1 : 2 * Real.sin x * Real.tan x = 3)
  (h2 : -Real.pi < x ∧ x < 0) : 
  x = -Real.pi/3 := by
  sorry

end problem_solution_l3043_304307


namespace wire_cutting_l3043_304329

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) :
  total_length = 50 →
  ratio = 2 / 5 →
  shorter_piece + shorter_piece / ratio = total_length →
  shorter_piece = 100 / 7 := by
sorry

end wire_cutting_l3043_304329


namespace investment_expected_profit_l3043_304310

/-- The number of investment projects -/
def num_projects : ℕ := 3

/-- The probability of success for each project -/
def prob_success : ℚ := 1/2

/-- The profit for a successful project -/
def profit_success : ℚ := 200000

/-- The loss for a failed project -/
def loss_failure : ℚ := 50000

/-- The expected profit for the investment projects -/
def expected_profit : ℚ := 225000

/-- Theorem stating that the expected profit for the investment projects is 225000 yuan -/
theorem investment_expected_profit :
  (num_projects : ℚ) * prob_success * (profit_success + loss_failure) - num_projects * loss_failure = expected_profit :=
by sorry

end investment_expected_profit_l3043_304310


namespace beef_weight_after_processing_l3043_304354

def initial_weight : ℝ := 1500
def weight_loss_percentage : ℝ := 50

theorem beef_weight_after_processing :
  let final_weight := initial_weight * (1 - weight_loss_percentage / 100)
  final_weight = 750 := by
sorry

end beef_weight_after_processing_l3043_304354


namespace urn_theorem_l3043_304387

/-- Represents the state of the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Represents a marble replacement rule -/
inductive Rule
  | rule1
  | rule2
  | rule3
  | rule4
  | rule5

/-- Applies a rule to the current urn state -/
def applyRule (state : UrnState) (rule : Rule) : UrnState :=
  match rule with
  | Rule.rule1 => UrnState.mk (state.white - 4) (state.black + 2)
  | Rule.rule2 => UrnState.mk (state.white - 1) (state.black)
  | Rule.rule3 => UrnState.mk (state.white - 1) (state.black)
  | Rule.rule4 => UrnState.mk (state.white + 1) (state.black - 3)
  | Rule.rule5 => UrnState.mk (state.white) (state.black - 1)

/-- Checks if the total number of marbles is even -/
def isEvenTotal (state : UrnState) : Prop :=
  Even (state.white + state.black)

/-- Checks if a given state is reachable from the initial state -/
def isReachable (initial : UrnState) (final : UrnState) : Prop :=
  ∃ (rules : List Rule), final = rules.foldl applyRule initial ∧ 
    (∀ (intermediate : UrnState), intermediate ∈ rules.scanl applyRule initial → isEvenTotal intermediate)

/-- The main theorem to prove -/
theorem urn_theorem (initial : UrnState) : 
  initial.white = 150 ∧ initial.black = 50 →
  (isReachable initial (UrnState.mk 78 72) ∨ isReachable initial (UrnState.mk 126 24)) :=
sorry


end urn_theorem_l3043_304387


namespace infinite_pairs_l3043_304389

/-- The set of prime divisors of a natural number -/
def primeDivisors (n : ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ p ∣ n}

/-- The condition for part (a) -/
def conditionA (a b : ℕ) : Prop :=
  primeDivisors a = primeDivisors b ∧
  primeDivisors (a + 1) = primeDivisors (b + 1) ∧
  a ≠ b

/-- The condition for part (b) -/
def conditionB (a b : ℕ) : Prop :=
  primeDivisors a = primeDivisors (b + 1) ∧
  primeDivisors (a + 1) = primeDivisors b

/-- The main theorem -/
theorem infinite_pairs :
  (∃ f : ℕ → ℕ × ℕ, Function.Injective f ∧ (∀ n, conditionA (f n).1 (f n).2)) ∧
  (∃ g : ℕ → ℕ × ℕ, Function.Injective g ∧ (∀ n, conditionB (g n).1 (g n).2)) :=
sorry

end infinite_pairs_l3043_304389


namespace craigs_age_l3043_304391

/-- Craig's age problem -/
theorem craigs_age (craig_age mother_age : ℕ) : 
  craig_age = mother_age - 24 →
  craig_age + mother_age = 56 →
  craig_age = 16 := by
  sorry

end craigs_age_l3043_304391


namespace point_p_coordinates_l3043_304315

-- Define a 2D point
structure Point2D where
  x : ℚ
  y : ℚ

-- Define a 2D vector
structure Vector2D where
  x : ℚ
  y : ℚ

-- Define vector between two points
def vectorBetween (p1 p2 : Point2D) : Vector2D :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

-- Define scalar multiplication for vectors
def scalarMult (k : ℚ) (v : Vector2D) : Vector2D :=
  { x := k * v.x, y := k * v.y }

theorem point_p_coordinates 
  (m n p : Point2D)
  (h1 : m = { x := 3, y := -2 })
  (h2 : n = { x := -5, y := -1 })
  (h3 : vectorBetween m p = scalarMult (1/3) (vectorBetween m n)) :
  p = { x := 1/3, y := -5/3 } := by
  sorry


end point_p_coordinates_l3043_304315


namespace constant_molecular_weight_l3043_304323

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 1280

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 8

/-- The total weight of the compound in grams -/
def total_weight : ℝ := molecular_weight * number_of_moles

theorem constant_molecular_weight :
  ∀ n : ℝ, n > 0 → molecular_weight = (molecular_weight * n) / n :=
by sorry

end constant_molecular_weight_l3043_304323


namespace intersection_condition_l3043_304377

/-- Set A defined by the given conditions -/
def set_A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m/2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

/-- Set B defined by the given conditions -/
def set_B (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2*m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2*m + 1}

/-- The main theorem stating the condition for non-empty intersection of A and B -/
theorem intersection_condition (m : ℝ) :
  (set_A m ∩ set_B m).Nonempty ↔ 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 := by
  sorry

end intersection_condition_l3043_304377


namespace lower_bound_of_a_l3043_304314

open Real

theorem lower_bound_of_a (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, x > 0 → f x = x * log x) →
  (∀ x, g x = x^3 + a*x^2 - x + 2) →
  (∀ x, x > 0 → 2 * f x ≤ (deriv g) x + 2) →
  a ≥ -2 := by
  sorry

end lower_bound_of_a_l3043_304314


namespace product_mod_five_l3043_304386

theorem product_mod_five : (1234 * 5678) % 5 = 2 := by
  sorry

end product_mod_five_l3043_304386


namespace parallel_iff_intersects_both_parallel_transitive_l3043_304384

-- Define the basic structures
structure Plane :=
(p : Type)

structure Line :=
(l : Type)

-- Define the relation for a line intersecting a plane at a single point
def intersects_at_single_point (l : Line) (α : Plane) : Prop :=
  ∃ (p : α.p), ∀ (q : α.p), l.l → (p = q)

-- Define the parallelism relation between planes
def parallel (α β : Plane) : Prop :=
  ∀ (l : Line), intersects_at_single_point l α → intersects_at_single_point l β

-- State the theorem
theorem parallel_iff_intersects_both (α β : Plane) :
  parallel α β ↔ ∀ (l : Line), intersects_at_single_point l α → intersects_at_single_point l β :=
sorry

-- State the transitivity of parallelism
theorem parallel_transitive (α β γ : Plane) :
  parallel α β → parallel β γ → parallel α γ :=
sorry

end parallel_iff_intersects_both_parallel_transitive_l3043_304384


namespace theater_attendance_l3043_304351

theorem theater_attendance 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (total_attendance : ℕ) 
  (total_revenue : ℚ) 
  (h1 : adult_price = 60 / 100)
  (h2 : child_price = 25 / 100)
  (h3 : total_attendance = 280)
  (h4 : total_revenue = 140) :
  ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adult_price * adults + child_price * children = total_revenue ∧
    children = 80 := by
sorry

end theater_attendance_l3043_304351


namespace xyz_squared_sum_l3043_304340

theorem xyz_squared_sum (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end xyz_squared_sum_l3043_304340


namespace fraction_simplification_l3043_304312

theorem fraction_simplification :
  (3 / 7 + 5 / 8) / (5 / 12 + 1 / 4) = 177 / 112 := by
  sorry

end fraction_simplification_l3043_304312


namespace rectangle_hyperbola_eccentricity_l3043_304342

/-- Rectangle with sides of length 4 and 3 -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_pos : length > 0)
  (width_pos : width > 0)
  (length_gt_width : length > width)

/-- Hyperbola passing through the vertices of the rectangle -/
structure Hyperbola (rect : Rectangle) :=
  (passes_through_vertices : Bool)

/-- Eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola rect) : ℝ := sorry

theorem rectangle_hyperbola_eccentricity (rect : Rectangle) 
  (h : Hyperbola rect) (h_passes : h.passes_through_vertices = true) :
  rect.length = 4 → rect.width = 3 → eccentricity h = 2 := by
  sorry

end rectangle_hyperbola_eccentricity_l3043_304342


namespace quadrilateral_area_bounds_sum_l3043_304350

/-- A convex quadrilateral with given side lengths -/
structure ConvexQuadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  convex : ab > 0 ∧ bc > 0 ∧ cd > 0 ∧ da > 0

/-- The area of a convex quadrilateral -/
def area (q : ConvexQuadrilateral) : ℝ := sorry

/-- The lower bound of the area of a convex quadrilateral -/
def lowerBound (q : ConvexQuadrilateral) : ℝ := sorry

/-- The upper bound of the area of a convex quadrilateral -/
def upperBound (q : ConvexQuadrilateral) : ℝ := sorry

theorem quadrilateral_area_bounds_sum :
  ∀ q : ConvexQuadrilateral,
  q.ab = 7 ∧ q.bc = 4 ∧ q.cd = 5 ∧ q.da = 6 →
  lowerBound q + upperBound q = 2 * Real.sqrt 210 := by
  sorry

end quadrilateral_area_bounds_sum_l3043_304350


namespace chip_drawing_probability_l3043_304339

def total_chips : ℕ := 11
def red_chips : ℕ := 5
def blue_chips : ℕ := 2
def green_chips : ℕ := 3
def yellow_chips : ℕ := 1

def consecutive_color_blocks : ℕ := 3
def yellow_positions : ℕ := total_chips + 1

theorem chip_drawing_probability : 
  (consecutive_color_blocks.factorial * red_chips.factorial * blue_chips.factorial * 
   green_chips.factorial * yellow_positions) / total_chips.factorial = 1 / 385 := by
  sorry

end chip_drawing_probability_l3043_304339


namespace only_clock_hands_rotate_l3043_304357

-- Define the concept of rotation
def is_rotation (motion : String) : Prop := 
  motion = "movement around a fixed point"

-- Define the given examples
def clock_hands : String := "movement of the hands of a clock"
def car_on_road : String := "car driving on a straight road"
def bottles_on_belt : String := "bottled beverages moving on a conveyor belt"
def soccer_ball : String := "soccer ball flying into the goal"

-- Theorem to prove
theorem only_clock_hands_rotate :
  is_rotation clock_hands ∧
  ¬is_rotation car_on_road ∧
  ¬is_rotation bottles_on_belt ∧
  ¬is_rotation soccer_ball :=
by sorry


end only_clock_hands_rotate_l3043_304357


namespace divisibility_implication_l3043_304376

theorem divisibility_implication (a b : ℤ) : 
  (31 ∣ (6 * a + 11 * b)) → (31 ∣ (a + 7 * b)) := by
sorry

end divisibility_implication_l3043_304376


namespace no_real_solution_l3043_304305

theorem no_real_solution : ¬∃ (a b c d : ℝ), 
  a^3 + c^3 = 2 ∧ 
  a^2*b + c^2*d = 0 ∧ 
  b^3 + d^3 = 1 ∧ 
  a*b^2 + c*d^2 = -6 := by
sorry

end no_real_solution_l3043_304305


namespace lunch_break_duration_l3043_304347

structure PaintingScenario where
  paula_rate : ℝ
  assistants_rate : ℝ
  lunch_break : ℝ

def monday_work (s : PaintingScenario) : ℝ :=
  (9 - s.lunch_break) * (s.paula_rate + s.assistants_rate)

def tuesday_work (s : PaintingScenario) : ℝ :=
  (7 - s.lunch_break) * s.assistants_rate

def wednesday_work (s : PaintingScenario) : ℝ :=
  (10 - s.lunch_break) * s.paula_rate

theorem lunch_break_duration (s : PaintingScenario) :
  monday_work s = 0.6 →
  tuesday_work s = 0.3 →
  wednesday_work s = 0.1 →
  s.lunch_break = 2/3 :=
by sorry

end lunch_break_duration_l3043_304347


namespace twenty_percent_of_twentyfive_percent_is_five_percent_l3043_304304

theorem twenty_percent_of_twentyfive_percent_is_five_percent :
  (20 / 100) * (25 / 100) = 5 / 100 := by
  sorry

end twenty_percent_of_twentyfive_percent_is_five_percent_l3043_304304


namespace coin_loading_impossibility_l3043_304365

theorem coin_loading_impossibility 
  (p q : ℝ) 
  (h1 : 0 < p ∧ p < 1) 
  (h2 : 0 < q ∧ q < 1) 
  (h3 : p ≠ 1 - p) 
  (h4 : q ≠ 1 - q) 
  (h5 : p * q = (1 : ℝ) / 4) 
  (h6 : p * (1 - q) = (1 : ℝ) / 4) 
  (h7 : (1 - p) * q = (1 : ℝ) / 4) 
  (h8 : (1 - p) * (1 - q) = (1 : ℝ) / 4) :
  False :=
sorry

end coin_loading_impossibility_l3043_304365


namespace line_properties_l3043_304352

/-- Given a line with equation 2x + y + 3 = 0, prove its slope and y-intercept -/
theorem line_properties :
  let line := {(x, y) : ℝ × ℝ | 2 * x + y + 3 = 0}
  ∃ (m b : ℝ), m = -2 ∧ b = -3 ∧ ∀ (x y : ℝ), (x, y) ∈ line ↔ y = m * x + b :=
by sorry

end line_properties_l3043_304352


namespace geometric_sequence_product_l3043_304368

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + a 9 = 10 →
  a 1 * a 9 = 16 →
  a 2 * a 5 * a 8 = 64 := by
  sorry

end geometric_sequence_product_l3043_304368


namespace initial_count_of_numbers_l3043_304366

/-- Given a set of numbers with average 27, prove that if removing 35 results in average 25, then there were initially 5 numbers -/
theorem initial_count_of_numbers (n : ℕ) (S : ℝ) : 
  S / n = 27 →
  (S - 35) / (n - 1) = 25 →
  n = 5 := by
sorry

end initial_count_of_numbers_l3043_304366


namespace point_2_4_in_first_quadrant_l3043_304356

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def is_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The theorem stating that the point (2,4) is in the first quadrant -/
theorem point_2_4_in_first_quadrant :
  let p : Point := ⟨2, 4⟩
  is_first_quadrant p := by
  sorry


end point_2_4_in_first_quadrant_l3043_304356


namespace book_pages_digits_unique_book_pages_l3043_304395

/-- Given a natural number n, calculate the total number of digits used to number pages from 1 to n. -/
def totalDigits (n : ℕ) : ℕ :=
  let singleDigits := min n 9
  let doubleDigits := max (min n 99 - 9) 0
  let tripleDigits := max (n - 99) 0
  singleDigits + 2 * doubleDigits + 3 * tripleDigits

/-- Theorem stating that a book with 266 pages requires exactly 690 digits to number all its pages. -/
theorem book_pages_digits : totalDigits 266 = 690 := by
  sorry

/-- Theorem stating that 266 is the unique number of pages that requires exactly 690 digits. -/
theorem unique_book_pages (n : ℕ) : totalDigits n = 690 → n = 266 := by
  sorry

end book_pages_digits_unique_book_pages_l3043_304395


namespace night_day_crew_loading_ratio_l3043_304318

theorem night_day_crew_loading_ratio 
  (day_crew : ℕ) 
  (night_crew : ℕ) 
  (total_boxes : ℝ) 
  (h1 : night_crew = (3 / 4 : ℝ) * day_crew)
  (h2 : (0.64 : ℝ) * total_boxes = day_crew * (total_boxes / day_crew)) :
  (total_boxes - 0.64 * total_boxes) / night_crew = 
  (3 / 4 : ℝ) * (0.64 * total_boxes / day_crew) := by
  sorry

end night_day_crew_loading_ratio_l3043_304318


namespace prob_at_least_two_fruits_l3043_304336

/-- The probability of choosing a specific fruit at a meal -/
def prob_single_fruit : ℚ := 1 / 3

/-- The number of meals in a day -/
def num_meals : ℕ := 4

/-- The probability of choosing the same fruit for all meals -/
def prob_same_fruit : ℚ := prob_single_fruit ^ num_meals

/-- The number of fruit types -/
def num_fruit_types : ℕ := 3

theorem prob_at_least_two_fruits : 
  1 - (num_fruit_types : ℚ) * prob_same_fruit = 26 / 27 := by sorry

end prob_at_least_two_fruits_l3043_304336


namespace sister_amount_calculation_l3043_304303

-- Define the amounts received from each source
def aunt_amount : ℝ := 9
def uncle_amount : ℝ := 9
def friends_amounts : List ℝ := [22, 23, 22, 22]

-- Define the mean of all amounts
def total_mean : ℝ := 16.3

-- Define the number of sources (including sister)
def num_sources : ℕ := 7

-- Theorem to prove
theorem sister_amount_calculation :
  let total_known := aunt_amount + uncle_amount + friends_amounts.sum
  let sister_amount := total_mean * num_sources - total_known
  sister_amount = 7.1 := by sorry

end sister_amount_calculation_l3043_304303


namespace square_equation_solution_l3043_304345

theorem square_equation_solution : ∃! x : ℝ, 97 + x * (19 + 91 / x) = 321 := by
  sorry

end square_equation_solution_l3043_304345


namespace min_sum_squares_l3043_304319

def S : Finset Int := {-6, -4, -1, 0, 3, 5, 7, 12}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 128 :=
by sorry

end min_sum_squares_l3043_304319


namespace final_round_probability_final_round_probability_value_l3043_304393

def GuessTheCard : Type := Unit

def tournament (n : ℕ) (rounds : ℕ) (win_prob : ℚ) : Type := Unit

theorem final_round_probability
  (n : ℕ)
  (rounds : ℕ)
  (win_prob : ℚ)
  (h1 : n = 16)
  (h2 : rounds = 4)
  (h3 : win_prob = 1 / 2)
  (h4 : ∀ (game : GuessTheCard), ∃! (winner : Unit), true)
  : ℚ :=
by
  sorry

#check final_round_probability

theorem final_round_probability_value
  (n : ℕ)
  (rounds : ℕ)
  (win_prob : ℚ)
  (h1 : n = 16)
  (h2 : rounds = 4)
  (h3 : win_prob = 1 / 2)
  (h4 : ∀ (game : GuessTheCard), ∃! (winner : Unit), true)
  : final_round_probability n rounds win_prob h1 h2 h3 h4 = 1 / 64 :=
by
  sorry

end final_round_probability_final_round_probability_value_l3043_304393


namespace john_gets_55_messages_l3043_304374

/-- The number of text messages John used to get per day -/
def old_messages_per_day : ℕ := 20

/-- The number of unintended text messages John gets per week -/
def unintended_messages_per_week : ℕ := 245

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Calculates the total number of text messages John gets per day now -/
def total_messages_per_day : ℕ :=
  old_messages_per_day + unintended_messages_per_week / days_per_week

/-- Theorem stating that John now gets 55 text messages per day -/
theorem john_gets_55_messages : total_messages_per_day = 55 := by
  sorry

end john_gets_55_messages_l3043_304374


namespace x_fourth_minus_six_x_l3043_304334

theorem x_fourth_minus_six_x (x : ℝ) : x = 3 → x^4 - 6*x = 63 := by
  sorry

end x_fourth_minus_six_x_l3043_304334


namespace triangle_side_length_l3043_304382

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (c^2 - a^2 = 5*b) → 
  (3 * Real.sin A * Real.cos C = Real.cos A * Real.sin C) → 
  b = 10 := by
  sorry

end triangle_side_length_l3043_304382


namespace train_speed_calculation_l3043_304341

/-- Proves that given the specified conditions, the speed of the first train is approximately 120.016 km/hr -/
theorem train_speed_calculation (length_train1 : ℝ) (length_train2 : ℝ) (speed_train2 : ℝ) (crossing_time : ℝ) :
  length_train1 = 250 →
  length_train2 = 250.04 →
  speed_train2 = 80 →
  crossing_time = 9 →
  ∃ (speed_train1 : ℝ), abs (speed_train1 - 120.016) < 0.001 :=
by
  sorry


end train_speed_calculation_l3043_304341


namespace mickey_mounts_98_horses_per_week_l3043_304359

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of horses Minnie mounts per day -/
def minnie_horses_per_day : ℕ := days_in_week + 3

/-- The number of horses Mickey mounts per day -/
def mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6

/-- The number of horses Mickey mounts per week -/
def mickey_horses_per_week : ℕ := mickey_horses_per_day * days_in_week

theorem mickey_mounts_98_horses_per_week :
  mickey_horses_per_week = 98 := by
  sorry

end mickey_mounts_98_horses_per_week_l3043_304359


namespace parabola_vertex_l3043_304325

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = (x - 2)^2 + 5

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 5)

/-- Theorem: The vertex of the parabola y = (x-2)^2 + 5 is at the point (2,5) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end parabola_vertex_l3043_304325


namespace perfect_squares_dividing_specific_l3043_304346

/-- The number of perfect squares dividing 2^3 * 3^5 * 5^7 * 7^9 -/
def perfectSquaresDividing (a b c d : ℕ) : ℕ :=
  (a/2 + 1) * (b/2 + 1) * (c/2 + 1) * (d/2 + 1)

/-- Theorem stating that the number of perfect squares dividing 2^3 * 3^5 * 5^7 * 7^9 is 120 -/
theorem perfect_squares_dividing_specific : perfectSquaresDividing 3 5 7 9 = 120 := by
  sorry

end perfect_squares_dividing_specific_l3043_304346


namespace line_through_points_l3043_304333

/-- Proves that for a line passing through (-3,1) and (1,3), m + b = 3 --/
theorem line_through_points (m b : ℚ) : 
  (1 = m * (-3) + b) ∧ (3 = m * 1 + b) → m + b = 3 := by
  sorry

end line_through_points_l3043_304333


namespace quiz_answer_key_combinations_l3043_304332

def num_true_false_questions : ℕ := 10
def num_multiple_choice_questions : ℕ := 6
def num_multiple_choice_options : ℕ := 6

theorem quiz_answer_key_combinations : 
  (Nat.choose num_true_false_questions (num_true_false_questions / 2)) * 
  (Nat.factorial num_multiple_choice_questions) = 181440 := by
  sorry

end quiz_answer_key_combinations_l3043_304332


namespace infinitely_many_non_sum_of_three_cubes_l3043_304360

theorem infinitely_many_non_sum_of_three_cubes :
  ∀ n : ℤ, (n % 9 = 4 ∨ n % 9 = 5) → ¬∃ a b c : ℤ, n = a^3 + b^3 + c^3 :=
by sorry

end infinitely_many_non_sum_of_three_cubes_l3043_304360


namespace smallest_divisor_for_perfect_square_l3043_304348

/-- A positive integer n is a perfect square if there exists an integer m such that n = m^2 -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- The smallest positive integer n such that 2880/n is a perfect square is 10 -/
theorem smallest_divisor_for_perfect_square : 
  (∀ k : ℕ, k > 0 ∧ k < 10 → ¬ IsPerfectSquare (2880 / k)) ∧ 
  IsPerfectSquare (2880 / 10) := by
  sorry

end smallest_divisor_for_perfect_square_l3043_304348


namespace sum_of_roots_equation_l3043_304301

theorem sum_of_roots_equation (x : ℝ) : 
  (7 = (x^3 - 2*x^2 - 8*x) / (x + 2)) → 
  (∃ y z : ℝ, x = y ∨ x = z ∧ y + z = 4) :=
by sorry

end sum_of_roots_equation_l3043_304301


namespace intersection_point_l3043_304372

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := (x^2 - 8*x + 12) / (2*x - 6)
def g (x b c d e : ℝ) : ℝ := (b*x^2 + c*x + d) / (x - e)

-- State the theorem
theorem intersection_point (b c d e : ℝ) :
  -- Conditions
  (∀ x, (2*x - 6 = 0 ↔ x - e = 0)) →  -- Same vertical asymptote
  (∃ k, ∀ x, g x b c d e = -2*x - 4 + k / (x - e)) →  -- Oblique asymptote of g
  (f (-3) = g (-3) b c d e) →  -- Intersection at x = -3
  -- Conclusion
  (∃ x y, x ≠ -3 ∧ f x = g x b c d e ∧ x = 14 ∧ y = 3) :=
by sorry

end intersection_point_l3043_304372


namespace triangle_problem_l3043_304321

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  b = Real.sqrt 13 ∧
  Real.sin A = (3 * Real.sqrt 13) / 13 ∧
  Real.sin (2 * A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end triangle_problem_l3043_304321


namespace price_reduction_theorem_l3043_304380

/-- Calculates the final price percentage after three consecutive price reductions -/
theorem price_reduction_theorem (initial_price : ℝ) 
  (reduction1 reduction2 reduction3 : ℝ) 
  (h1 : reduction1 = 0.09)
  (h2 : reduction2 = 0.10)
  (h3 : reduction3 = 0.15) : 
  (initial_price * (1 - reduction1) * (1 - reduction2) * (1 - reduction3)) / initial_price = 0.69615 := by
  sorry

#check price_reduction_theorem

end price_reduction_theorem_l3043_304380


namespace cars_between_15k_and_20k_l3043_304316

/-- Given a car dealership with the following properties:
  * There are 3000 cars in total
  * 15% of cars cost less than $15000
  * 40% of cars cost more than $20000
  Prove that the number of cars costing between $15000 and $20000 is 1350 -/
theorem cars_between_15k_and_20k (total_cars : ℕ) (percent_less_15k : ℚ) (percent_more_20k : ℚ) 
  (h_total : total_cars = 3000)
  (h_less_15k : percent_less_15k = 15 / 100)
  (h_more_20k : percent_more_20k = 40 / 100) :
  total_cars - (total_cars * percent_less_15k).floor - (total_cars * percent_more_20k).floor = 1350 := by
  sorry

end cars_between_15k_and_20k_l3043_304316


namespace sequence_general_term_l3043_304311

theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ∀ n, S n = 2 * n - a n) :
  ∀ n, a n = (2^n - 1) / 2^(n-1) := by
sorry

end sequence_general_term_l3043_304311


namespace responder_is_liar_responder_is_trulalala_l3043_304367

-- Define the brothers
inductive Brother
| Tweedledee
| Trulalala

-- Define the possible responses
inductive Response
| Circle
| Square

-- Define the property of being truthful or a liar
def isTruthful (b : Brother) : Prop :=
  match b with
  | Brother.Tweedledee => true
  | Brother.Trulalala => false

-- Define the question asked
def questionAsked (actual : Response) : Prop :=
  actual = Response.Square

-- Define the response given
def responseGiven : Response := Response.Circle

-- Theorem to prove
theorem responder_is_liar :
  ∀ (responder asker : Brother),
  responder ≠ asker →
  (isTruthful responder ↔ ¬isTruthful asker) →
  responseGiven = Response.Circle →
  ¬isTruthful responder := by
  sorry

-- Corollary: The responder is Trulalala
theorem responder_is_trulalala :
  ∀ (responder asker : Brother),
  responder ≠ asker →
  (isTruthful responder ↔ ¬isTruthful asker) →
  responseGiven = Response.Circle →
  responder = Brother.Trulalala := by
  sorry

end responder_is_liar_responder_is_trulalala_l3043_304367


namespace test_problems_l3043_304379

theorem test_problems (total_problems : ℕ) (comp_points : ℕ) (word_points : ℕ) (total_points : ℕ) :
  total_problems = 30 →
  comp_points = 3 →
  word_points = 5 →
  total_points = 110 →
  ∃ (comp_count : ℕ) (word_count : ℕ),
    comp_count + word_count = total_problems ∧
    comp_count * comp_points + word_count * word_points = total_points ∧
    comp_count = 20 := by
  sorry

end test_problems_l3043_304379


namespace solve_star_equation_l3043_304327

-- Define the star operation
def star (a b : ℝ) : ℝ := 4 * a + 2 * b

-- State the theorem
theorem solve_star_equation : 
  ∃ y : ℝ, star 3 (star 4 y) = 8 ∧ y = -9 := by
  sorry

end solve_star_equation_l3043_304327


namespace jim_card_distribution_l3043_304317

theorem jim_card_distribution (initial_cards : ℕ) (brother_sets sister_sets : ℕ) 
  (total_given : ℕ) (cards_per_set : ℕ) : 
  initial_cards = 365 →
  brother_sets = 8 →
  sister_sets = 5 →
  total_given = 195 →
  cards_per_set = 13 →
  (brother_sets + sister_sets + (total_given - (brother_sets + sister_sets) * cards_per_set) / cards_per_set : ℕ) = 
    brother_sets + sister_sets + 2 := by
  sorry

#check jim_card_distribution

end jim_card_distribution_l3043_304317


namespace max_rectangles_in_3x4_grid_l3043_304394

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a grid with width and height -/
structure Grid where
  width : Nat
  height : Nat

/-- Checks if a rectangle can fit in a grid -/
def fits (r : Rectangle) (g : Grid) : Prop :=
  r.width ≤ g.width ∧ r.height ≤ g.height

/-- Represents the maximum number of non-overlapping rectangles that can fit in a grid -/
def maxRectangles (r : Rectangle) (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that the maximum number of 1x2 rectangles in a 3x4 grid is 5 -/
theorem max_rectangles_in_3x4_grid :
  let r : Rectangle := ⟨1, 2⟩
  let g : Grid := ⟨3, 4⟩
  fits r g → maxRectangles r g = 5 := by
  sorry

end max_rectangles_in_3x4_grid_l3043_304394


namespace david_shells_l3043_304344

theorem david_shells (david mia ava alice : ℕ) : 
  mia = 4 * david →
  ava = mia + 20 →
  alice = ava / 2 →
  david + mia + ava + alice = 195 →
  david = 15 := by
sorry

end david_shells_l3043_304344


namespace cubic_equation_solution_l3043_304313

theorem cubic_equation_solution :
  let x : ℝ := Real.rpow (19/2) (1/3) - 2
  2 * x^3 + 24 * x = 3 - 12 * x^2 := by
    sorry

end cubic_equation_solution_l3043_304313


namespace shooting_match_sequences_l3043_304306

/-- The number of permutations of a multiset with the given multiplicities -/
def multiset_permutations (n : ℕ) (multiplicities : List ℕ) : ℕ :=
  n.factorial / (multiplicities.map Nat.factorial).prod

/-- The number of different sequences for breaking targets in the shooting match -/
theorem shooting_match_sequences : 
  multiset_permutations 10 [3, 3, 2, 2] = 25200 := by
  sorry

end shooting_match_sequences_l3043_304306


namespace smallest_factorial_with_1987_zeros_l3043_304353

/-- Count the number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- The smallest natural number n such that n! ends with exactly 1987 zeros -/
def smallestFactorialWith1987Zeros : ℕ := 7960

theorem smallest_factorial_with_1987_zeros :
  (∀ m : ℕ, m < smallestFactorialWith1987Zeros → trailingZeros m < 1987) ∧
  trailingZeros smallestFactorialWith1987Zeros = 1987 := by
  sorry

end smallest_factorial_with_1987_zeros_l3043_304353


namespace quadratic_root_sum_bound_l3043_304398

theorem quadratic_root_sum_bound (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  let roots := {x : ℝ | f x = 0}
  (∃ m n : ℝ, m ∈ roots ∧ n ∈ roots ∧ abs m + abs n ≤ 1) →
  -1/4 ≤ b ∧ b < 1/4 := by
sorry

end quadratic_root_sum_bound_l3043_304398


namespace tank_insulation_cost_l3043_304320

/-- Calculates the surface area of a rectangular prism -/
def surfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Calculates the total cost of insulation for a rectangular tank -/
def insulationCost (length width height costPerSquareFoot : ℝ) : ℝ :=
  surfaceArea length width height * costPerSquareFoot

/-- Theorem: The cost to insulate a 4x5x2 feet tank at $20 per square foot is $1520 -/
theorem tank_insulation_cost :
  insulationCost 4 5 2 20 = 1520 := by
  sorry

end tank_insulation_cost_l3043_304320


namespace average_price_per_pair_l3043_304349

/-- Given the total sales and number of pairs sold, prove the average price per pair -/
theorem average_price_per_pair (total_sales : ℝ) (pairs_sold : ℕ) (h1 : total_sales = 686) (h2 : pairs_sold = 70) :
  total_sales / pairs_sold = 9.80 := by
  sorry

end average_price_per_pair_l3043_304349


namespace integer_triangle_area_rational_l3043_304322

/-- A triangle with integer coordinates where two points form a line parallel to the x-axis -/
structure IntegerTriangle where
  x₁ : ℤ
  y₁ : ℤ
  x₂ : ℤ
  y₂ : ℤ
  x₃ : ℤ
  y₃ : ℤ
  parallel_to_x : y₁ = y₂

/-- The area of an IntegerTriangle is rational -/
theorem integer_triangle_area_rational (t : IntegerTriangle) : ∃ (q : ℚ), q = |((t.x₂ - t.x₁) * t.y₃) / 2| := by
  sorry

end integer_triangle_area_rational_l3043_304322


namespace min_value_of_sum_min_value_reached_min_value_is_27_l3043_304358

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 27) :
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 27 → a + 3 * b + 9 * c ≤ x + 3 * y + 9 * z :=
by
  sorry

theorem min_value_reached (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 27) :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 27 ∧ a + 3 * b + 9 * c = x + 3 * y + 9 * z :=
by
  sorry

theorem min_value_is_27 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 :=
by
  sorry

end min_value_of_sum_min_value_reached_min_value_is_27_l3043_304358


namespace root_conditions_l3043_304330

theorem root_conditions (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 4*x^4 + a*x^2 = b*x + 4*c ↔ x = 2 ∨ x = -2) ↔ 
  (a = -48 ∧ b = 16 ∧ c = -32) :=
sorry

end root_conditions_l3043_304330


namespace algebra_test_average_l3043_304308

theorem algebra_test_average (male_count : ℕ) (female_count : ℕ) 
  (male_avg : ℝ) (female_avg : ℝ) :
  male_count = 8 →
  female_count = 12 →
  male_avg = 87 →
  female_avg = 92 →
  let total_count := male_count + female_count
  let total_sum := male_count * male_avg + female_count * female_avg
  total_sum / total_count = 90 := by
sorry

end algebra_test_average_l3043_304308


namespace stability_comparison_l3043_304392

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when one athlete's performance is more stable than another's -/
def more_stable (a b : Athlete) : Prop :=
  a.average_score = b.average_score ∧ a.variance < b.variance

theorem stability_comparison (a b : Athlete) 
  (h : a.variance < b.variance) (h_avg : a.average_score = b.average_score) : 
  more_stable a b := by
  sorry

#check stability_comparison

end stability_comparison_l3043_304392


namespace vertical_shift_of_linear_function_l3043_304373

theorem vertical_shift_of_linear_function (x : ℝ) :
  (-3/4 * x) - (-3/4 * x - 4) = 4 := by
  sorry

end vertical_shift_of_linear_function_l3043_304373


namespace real_roots_of_polynomial_l3043_304364

theorem real_roots_of_polynomial (x₁ x₂ : ℝ) :
  x₁^5 - 55*x₁ + 21 = 0 →
  x₂^5 - 55*x₂ + 21 = 0 →
  x₁ * x₂ = 1 →
  ((x₁ = (3 + Real.sqrt 5) / 2 ∧ x₂ = (3 - Real.sqrt 5) / 2) ∨
   (x₁ = (3 - Real.sqrt 5) / 2 ∧ x₂ = (3 + Real.sqrt 5) / 2)) :=
by sorry

end real_roots_of_polynomial_l3043_304364


namespace min_distance_to_line_l3043_304371

/-- The minimum value of (x - 2)^2 + (y - 2)^2 when (x, y) lies on the line x - y - 1 = 0 -/
theorem min_distance_to_line : 
  ∃ (min : ℝ), min = (1/2 : ℝ) ∧ 
  ∀ (x y : ℝ), x - y - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≥ min :=
by sorry

end min_distance_to_line_l3043_304371


namespace sqrt_meaningful_range_l3043_304355

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 5 - x ∧ y ≥ 0) ↔ x ≤ 5 := by sorry

end sqrt_meaningful_range_l3043_304355


namespace place_value_ratio_l3043_304385

theorem place_value_ratio : 
  let number : ℚ := 56842.7093
  let digit_8_place_value : ℚ := 1000
  let digit_7_place_value : ℚ := 0.1
  digit_8_place_value / digit_7_place_value = 10000 := by
  sorry

end place_value_ratio_l3043_304385


namespace intersection_M_N_l3043_304362

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | (x + 1) * (x - 2) < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l3043_304362


namespace big_sale_commission_l3043_304331

theorem big_sale_commission 
  (sales_before : ℕ)
  (total_sales : ℕ)
  (avg_increase : ℚ)
  (new_avg : ℚ) :
  sales_before = 5 →
  total_sales = 6 →
  avg_increase = 150 →
  new_avg = 250 →
  (total_sales * new_avg - sales_before * (new_avg - avg_increase)) = 1000 := by
  sorry

end big_sale_commission_l3043_304331


namespace y_completion_time_l3043_304388

/-- A worker's rate is defined as the fraction of work they can complete in one day -/
def worker_rate (days_to_complete : ℚ) : ℚ := 1 / days_to_complete

/-- The time taken to complete a given fraction of work at a given rate -/
def time_for_fraction (fraction : ℚ) (rate : ℚ) : ℚ := fraction / rate

theorem y_completion_time (x_total_days : ℚ) (x_worked_days : ℚ) (y_completion_days : ℚ) : 
  x_total_days = 40 → x_worked_days = 8 → y_completion_days = 28 →
  time_for_fraction 1 (worker_rate (time_for_fraction 1 
    (worker_rate y_completion_days / (1 - x_worked_days * worker_rate x_total_days)))) = 35 := by
  sorry

end y_completion_time_l3043_304388


namespace polynomial_division_theorem_l3043_304375

theorem polynomial_division_theorem (x : ℝ) :
  ∃ R : ℝ, 5 * x^3 + 4 * x^2 - 6 * x - 9 = (x - 1) * (5 * x^2 + 9 * x + 3) + R :=
by sorry

end polynomial_division_theorem_l3043_304375


namespace two_books_total_cost_l3043_304381

/-- Proves that the total cost of two books is 420 given the specified conditions -/
theorem two_books_total_cost :
  ∀ (cost_loss cost_gain selling_price : ℝ),
  cost_loss = 245 →
  selling_price = cost_loss * 0.85 →
  selling_price = cost_gain * 1.19 →
  cost_loss + cost_gain = 420 :=
by
  sorry

end two_books_total_cost_l3043_304381


namespace diagonal_intersection_coincides_l3043_304343

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Function to check if a quadrilateral is inscribed in a circle
def isInscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Function to check if a point is on a circle
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Function to check if a line is tangent to a circle at a point
def isTangent (p1 p2 : Point) (c : Circle) (p : Point) : Prop := sorry

-- Function to find the intersection point of two lines
def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

-- Main theorem
theorem diagonal_intersection_coincides 
  (c : Circle) 
  (ABCD : Quadrilateral) 
  (E F G K : Point) :
  isInscribed ABCD c →
  isOnCircle E c ∧ isOnCircle F c ∧ isOnCircle G c ∧ isOnCircle K c →
  isTangent ABCD.A ABCD.B c E →
  isTangent ABCD.B ABCD.C c F →
  isTangent ABCD.C ABCD.D c G →
  isTangent ABCD.D ABCD.A c K →
  intersectionPoint ABCD.A ABCD.C ABCD.B ABCD.D = intersectionPoint E G F K := by
  sorry

end diagonal_intersection_coincides_l3043_304343


namespace unique_products_count_l3043_304378

def set_a : Finset ℕ := {2, 3, 5, 7, 11}
def set_b : Finset ℕ := {2, 4, 6, 19}

theorem unique_products_count : 
  Finset.card ((set_a.product set_b).image (λ (x : ℕ × ℕ) => x.1 * x.2)) = 19 := by
  sorry

end unique_products_count_l3043_304378


namespace negative_one_three_in_M_l3043_304383

def M : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, y = 2 * x ∧ p = (x - y, x + y)}

theorem negative_one_three_in_M : ((-1 : ℝ), (3 : ℝ)) ∈ M := by
  sorry

end negative_one_three_in_M_l3043_304383


namespace no_x_squared_term_l3043_304361

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 5*a*x + a) = x^3 + (-4*a)*x + a) → a = 1/5 := by
sorry

end no_x_squared_term_l3043_304361


namespace min_value_on_line_l3043_304338

/-- Given a point C(a,b) on the line passing through A(1,1) and B(-2,4),
    the minimum value of 1/a + 4/b is 9/2 -/
theorem min_value_on_line (a b : ℝ) (h : a + b = 2) :
  (∀ x y : ℝ, x + y = 2 → 1/a + 4/b ≤ 1/x + 4/y) ∧ (∃ x y : ℝ, x + y = 2 ∧ 1/x + 4/y = 9/2) := by
  sorry

end min_value_on_line_l3043_304338


namespace x_eq_two_sufficient_not_necessary_l3043_304335

def M (x : ℝ) : Set ℝ := {1, x}
def N : Set ℝ := {1, 2, 3}

theorem x_eq_two_sufficient_not_necessary :
  ∀ x : ℝ, 
  (x = 2 → M x ⊆ N) ∧ 
  ¬(M x ⊆ N → x = 2) :=
by sorry

end x_eq_two_sufficient_not_necessary_l3043_304335


namespace prime_of_square_minus_one_l3043_304337

theorem prime_of_square_minus_one (a : ℕ) (h : a ≥ 2) :
  Nat.Prime (a^2 - 1) ↔ a = 2 := by
  sorry

end prime_of_square_minus_one_l3043_304337


namespace pandas_weekly_bamboo_consumption_l3043_304399

/-- The amount of bamboo eaten by pandas in a week -/
def bamboo_eaten_in_week (adult_daily : ℕ) (baby_daily : ℕ) : ℕ :=
  (adult_daily + baby_daily) * 7

/-- Theorem: Pandas eat 1316 pounds of bamboo in a week -/
theorem pandas_weekly_bamboo_consumption :
  bamboo_eaten_in_week 138 50 = 1316 := by
  sorry

end pandas_weekly_bamboo_consumption_l3043_304399


namespace arnold_gas_expenditure_l3043_304309

def monthly_gas_expenditure (car1_mpg car2_mpg car3_mpg : ℚ) 
  (total_mileage : ℚ) (gas_price : ℚ) : ℚ :=
  let mileage_per_car := total_mileage / 3
  let gallons_car1 := mileage_per_car / car1_mpg
  let gallons_car2 := mileage_per_car / car2_mpg
  let gallons_car3 := mileage_per_car / car3_mpg
  let total_gallons := gallons_car1 + gallons_car2 + gallons_car3
  total_gallons * gas_price

theorem arnold_gas_expenditure :
  monthly_gas_expenditure 50 10 15 450 2 = 56 := by
  sorry

end arnold_gas_expenditure_l3043_304309


namespace transaction_gain_l3043_304302

/-- Calculates the simple interest for a given principal, rate, and time --/
def simpleInterest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  (principal : ℚ) * rate * (time : ℚ) / 100

/-- Calculates the annual gain from borrowing and lending money --/
def annualGain (principal : ℕ) (borrowRate : ℚ) (lendRate : ℚ) (time : ℕ) : ℚ :=
  (simpleInterest principal lendRate time - simpleInterest principal borrowRate time) / (time : ℚ)

theorem transaction_gain (principal : ℕ) (borrowRate lendRate : ℚ) (time : ℕ) :
  principal = 8000 →
  borrowRate = 4 →
  lendRate = 6 →
  time = 2 →
  annualGain principal borrowRate lendRate time = 800 := by
  sorry

end transaction_gain_l3043_304302


namespace bayberry_sales_theorem_l3043_304326

/-- Represents the bayberry selling scenario -/
structure BayberrySales where
  initial_price : ℝ
  initial_volume : ℝ
  cost_price : ℝ
  volume_increase_rate : ℝ

/-- Calculates the daily revenue given a price decrease -/
def daily_revenue (s : BayberrySales) (price_decrease : ℝ) : ℝ :=
  (s.initial_price - price_decrease) * (s.initial_volume + s.volume_increase_rate * price_decrease)

/-- Calculates the daily profit given a selling price -/
def daily_profit (s : BayberrySales) (selling_price : ℝ) : ℝ :=
  (selling_price - s.cost_price) * (s.initial_volume + s.volume_increase_rate * (s.initial_price - selling_price))

/-- The main theorem about bayberry sales -/
theorem bayberry_sales_theorem (s : BayberrySales) 
  (h1 : s.initial_price = 20)
  (h2 : s.initial_volume = 100)
  (h3 : s.cost_price = 8)
  (h4 : s.volume_increase_rate = 20) :
  (∃ x y, x ≠ y ∧ daily_revenue s x = 3000 ∧ daily_revenue s y = 3000 ∧ x = 5 ∧ y = 10) ∧
  (∃ max_price max_profit, 
    (∀ p, daily_profit s p ≤ max_profit) ∧
    daily_profit s max_price = max_profit ∧
    max_price = 16.5 ∧ max_profit = 1445) := by
  sorry


end bayberry_sales_theorem_l3043_304326


namespace vector_propositions_correctness_l3043_304397

variable {V : Type*} [AddCommGroup V]

-- Define vectors as differences between points
def vec (A B : V) : V := B - A

-- State the theorem
theorem vector_propositions_correctness :
  ∃ (A B C : V),
    (vec A B + vec B A = 0) ∧
    (vec A B + vec B C = vec A C) ∧
    ¬(vec A B - vec A C = vec B C) ∧
    ¬(0 • vec A B = 0) :=
by sorry

end vector_propositions_correctness_l3043_304397


namespace machine_B_efficiency_l3043_304324

/-- The number of sprockets produced by each machine -/
def total_sprockets : ℕ := 440

/-- The rate at which Machine A produces sprockets (sprockets per hour) -/
def rate_A : ℚ := 4

/-- The time difference between Machine A and Machine B to produce the total sprockets -/
def time_difference : ℕ := 10

/-- Calculates the percentage increase of rate B compared to rate A -/
def percentage_increase (rate_A rate_B : ℚ) : ℚ :=
  (rate_B - rate_A) / rate_A * 100

theorem machine_B_efficiency :
  let time_A := total_sprockets / rate_A
  let time_B := time_A - time_difference
  let rate_B := total_sprockets / time_B
  percentage_increase rate_A rate_B = 10 := by sorry

end machine_B_efficiency_l3043_304324


namespace solution_implies_a_value_l3043_304390

theorem solution_implies_a_value (a : ℝ) : 
  (3 * a + 4 = 1) → a = -1 := by
  sorry

end solution_implies_a_value_l3043_304390
