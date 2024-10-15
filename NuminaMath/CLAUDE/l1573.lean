import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l1573_157397

/-- Triangle ABC with inscribed square PQRS -/
structure InscribedSquareTriangle where
  /-- Side length of AB -/
  ab : ℝ
  /-- Side length of BC -/
  bc : ℝ
  /-- Side length of CA -/
  ca : ℝ
  /-- Point P lies on BC -/
  p_on_bc : Bool
  /-- Point R lies on BC -/
  r_on_bc : Bool
  /-- Point Q lies on CA -/
  q_on_ca : Bool
  /-- Point S lies on AB -/
  s_on_ab : Bool

/-- The side length of the inscribed square PQRS -/
def squareSideLength (t : InscribedSquareTriangle) : ℝ := sorry

/-- Theorem: The side length of the inscribed square is 42 -/
theorem inscribed_square_side_length 
  (t : InscribedSquareTriangle) 
  (h1 : t.ab = 13) 
  (h2 : t.bc = 14) 
  (h3 : t.ca = 15) 
  (h4 : t.p_on_bc = true) 
  (h5 : t.r_on_bc = true) 
  (h6 : t.q_on_ca = true) 
  (h7 : t.s_on_ab = true) : 
  squareSideLength t = 42 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l1573_157397


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1573_157363

theorem complex_fraction_equality : Complex.I ^ 2 + Complex.I ^ 3 + Complex.I ^ 4 = (1 / 2 - Complex.I / 2) * (1 - Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1573_157363


namespace NUMINAMATH_CALUDE_rational_absolute_value_and_negative_numbers_l1573_157305

theorem rational_absolute_value_and_negative_numbers :
  (∀ x : ℚ, |x| ≥ 0 ∧ (|x| = 0 ↔ x = 0)) ∧
  (∀ x : ℝ, -x > x → x < 0) := by
  sorry

end NUMINAMATH_CALUDE_rational_absolute_value_and_negative_numbers_l1573_157305


namespace NUMINAMATH_CALUDE_students_enjoying_both_sports_l1573_157312

theorem students_enjoying_both_sports 
  (total : ℕ) 
  (running : ℕ) 
  (basketball : ℕ) 
  (neither : ℕ) 
  (h1 : total = 38) 
  (h2 : running = 21) 
  (h3 : basketball = 15) 
  (h4 : neither = 10) :
  running + basketball - (total - neither) = 8 :=
by sorry

end NUMINAMATH_CALUDE_students_enjoying_both_sports_l1573_157312


namespace NUMINAMATH_CALUDE_rajs_house_area_l1573_157331

/-- The total area of Raj's house given the specified room dimensions and counts -/
theorem rajs_house_area : 
  let bedroom_count : ℕ := 4
  let bedroom_side : ℕ := 11
  let bathroom_count : ℕ := 2
  let bathroom_length : ℕ := 8
  let bathroom_width : ℕ := 6
  let kitchen_area : ℕ := 265
  
  bedroom_count * (bedroom_side * bedroom_side) +
  bathroom_count * (bathroom_length * bathroom_width) +
  kitchen_area +
  kitchen_area = 1110 := by
sorry

end NUMINAMATH_CALUDE_rajs_house_area_l1573_157331


namespace NUMINAMATH_CALUDE_quarter_probability_is_3_28_l1573_157365

/-- Represents the types of coins in the jar -/
inductive Coin
| Quarter
| Nickel
| Penny
| Dime

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
| Coin.Quarter => 25
| Coin.Nickel => 5
| Coin.Penny => 1
| Coin.Dime => 10

/-- The total value of each coin type in cents -/
def total_value : Coin → ℕ
| Coin.Quarter => 1200
| Coin.Nickel => 500
| Coin.Penny => 200
| Coin.Dime => 1000

/-- The number of coins of each type -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Quarter + coin_count Coin.Nickel + 
                       coin_count Coin.Penny + coin_count Coin.Dime

/-- The probability of choosing a quarter -/
def quarter_probability : ℚ := coin_count Coin.Quarter / total_coins

theorem quarter_probability_is_3_28 : quarter_probability = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_quarter_probability_is_3_28_l1573_157365


namespace NUMINAMATH_CALUDE_unique_number_property_l1573_157368

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def remove_digit (n : ℕ) (pos : Fin 5) : ℕ :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  let removed := digits.removeNth pos
  removed.foldl (fun acc d => acc * 10 + d) 0

theorem unique_number_property :
  ∃! n : ℕ, is_five_digit n ∧
    ∃ pos : Fin 5, n + remove_digit n pos = 54321 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l1573_157368


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_line_l1573_157316

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def areParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem to prove -/
theorem line_through_point_parallel_to_given_line :
  let A : Point2D := ⟨2, 3⟩
  let givenLine : Line2D := ⟨2, 4, -3⟩
  let resultLine : Line2D := ⟨1, 2, -8⟩
  pointOnLine A resultLine ∧ areParallel resultLine givenLine := by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_line_l1573_157316


namespace NUMINAMATH_CALUDE_expression_evaluation_l1573_157381

theorem expression_evaluation : 
  (1.2 : ℝ)^3 - (0.9 : ℝ)^3 / (1.2 : ℝ)^2 + 1.08 + (0.9 : ℝ)^2 = 3.11175 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1573_157381


namespace NUMINAMATH_CALUDE_larger_number_twice_smaller_l1573_157334

theorem larger_number_twice_smaller (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a - b/2 = 3 * (b - b/2)) : a = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_larger_number_twice_smaller_l1573_157334


namespace NUMINAMATH_CALUDE_basketball_tournament_equation_l1573_157319

/-- The number of games played in a basketball tournament -/
def num_games : ℕ := 28

/-- Theorem: In a basketball tournament with x teams, where each pair of teams plays exactly one game,
    and a total of 28 games are played, the equation ½x(x-1) = 28 holds true. -/
theorem basketball_tournament_equation (x : ℕ) (h : x > 1) :
  (x * (x - 1)) / 2 = num_games :=
sorry

end NUMINAMATH_CALUDE_basketball_tournament_equation_l1573_157319


namespace NUMINAMATH_CALUDE_unique_a_in_A_l1573_157345

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_in_A : ∃! a : ℝ, 1 ∈ A a := by sorry

end NUMINAMATH_CALUDE_unique_a_in_A_l1573_157345


namespace NUMINAMATH_CALUDE_teddy_bear_shelves_l1573_157371

theorem teddy_bear_shelves (total_bears : ℕ) (shelf_capacity : ℕ) (filled_shelves : ℕ) : 
  total_bears = 98 → 
  shelf_capacity = 7 → 
  filled_shelves = total_bears / shelf_capacity →
  filled_shelves = 14 := by
sorry

end NUMINAMATH_CALUDE_teddy_bear_shelves_l1573_157371


namespace NUMINAMATH_CALUDE_image_square_characterization_l1573_157339

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ := (x^2 - y^2, x*y)

-- Define the unit square
def unit_square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the image of the unit square
def image_square : Set (ℝ × ℝ) := {p | ∃ q ∈ unit_square, transform q.1 q.2 = p}

-- Define the boundary curves
def curve_OC : Set (ℝ × ℝ) := {p | ∃ y ∈ Set.Icc 0 1, p = (-y^2, 0)}
def curve_OA : Set (ℝ × ℝ) := {p | ∃ x ∈ Set.Icc 0 1, p = (x^2, 0)}
def curve_AB : Set (ℝ × ℝ) := {p | ∃ y ∈ Set.Icc 0 1, p = (1 - y^2, y)}
def curve_BC : Set (ℝ × ℝ) := {p | ∃ x ∈ Set.Icc 0 1, p = (x^2 - 1, x)}

-- Define the boundary of the image
def image_boundary : Set (ℝ × ℝ) := curve_OC ∪ curve_OA ∪ curve_AB ∪ curve_BC

-- Theorem statement
theorem image_square_characterization :
  image_square = {p | p ∈ image_boundary ∨ (∃ q ∈ image_boundary, p.1 < q.1 ∧ p.2 < q.2)} := by
  sorry

end NUMINAMATH_CALUDE_image_square_characterization_l1573_157339


namespace NUMINAMATH_CALUDE_function_fixed_point_l1573_157314

def iterateF (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterateF f n)

theorem function_fixed_point
  (f : ℝ → ℝ)
  (hf : Continuous f)
  (h : ∀ x : ℝ, ∃ n : ℕ, iterateF f n x = 1) :
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_fixed_point_l1573_157314


namespace NUMINAMATH_CALUDE_relationship_abcd_l1573_157384

theorem relationship_abcd (a b c d : ℝ) :
  (a + 2*b) / (2*b + c) = (c + 2*d) / (2*d + a) →
  (a = c ∨ a + c + 2*(b + d) = 0) :=
by sorry

end NUMINAMATH_CALUDE_relationship_abcd_l1573_157384


namespace NUMINAMATH_CALUDE_equidistant_line_from_three_parallel_lines_l1573_157342

/-- Given three parallel lines in the form Ax + By = Cᵢ, 
    this theorem states that the line Ax + By = (C₁ + 2C₂ + C₃) / 4 
    is equidistant from all three lines. -/
theorem equidistant_line_from_three_parallel_lines 
  (A B C₁ C₂ C₃ : ℝ) 
  (h_distinct₁ : C₁ ≠ C₂) 
  (h_distinct₂ : C₂ ≠ C₃) 
  (h_distinct₃ : C₁ ≠ C₃) :
  let d₁₂ := |C₂ - C₁| / Real.sqrt (A^2 + B^2)
  let d₂₃ := |C₃ - C₂| / Real.sqrt (A^2 + B^2)
  let d₁₃ := |C₃ - C₁| / Real.sqrt (A^2 + B^2)
  let M := (C₁ + 2*C₂ + C₃) / 4
  ∀ x y, A*x + B*y = M → 
    |A*x + B*y - C₁| / Real.sqrt (A^2 + B^2) = 
    |A*x + B*y - C₂| / Real.sqrt (A^2 + B^2) ∧
    |A*x + B*y - C₂| / Real.sqrt (A^2 + B^2) = 
    |A*x + B*y - C₃| / Real.sqrt (A^2 + B^2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_line_from_three_parallel_lines_l1573_157342


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1573_157300

theorem completing_square_equivalence (x : ℝ) :
  x^2 - 6*x + 1 = 0 ↔ (x - 3)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1573_157300


namespace NUMINAMATH_CALUDE_fraction_simplification_l1573_157317

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1573_157317


namespace NUMINAMATH_CALUDE_m_shaped_area_l1573_157351

/-- The area of the M-shaped region formed by folding a 12 × 18 rectangle along its diagonal -/
theorem m_shaped_area (width : ℝ) (height : ℝ) (diagonal : ℝ) (m_area : ℝ) : 
  width = 12 → 
  height = 18 → 
  diagonal = (width^2 + height^2).sqrt →
  m_area = 138 → 
  m_area = (width * height / 2) + 2 * (width * height / 2 - (13 / 36) * (width * height / 2)) :=
by sorry

end NUMINAMATH_CALUDE_m_shaped_area_l1573_157351


namespace NUMINAMATH_CALUDE_divisibility_criterion_l1573_157373

theorem divisibility_criterion (a b c : ℤ) (d : ℤ) (h1 : d = 10*c + 1) (h2 : ∃ k, a - b*c = d*k) : 
  ∃ m, 10*a + b = d*m :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l1573_157373


namespace NUMINAMATH_CALUDE_log_eight_negative_seven_fourths_l1573_157396

theorem log_eight_negative_seven_fourths (x : ℝ) : 
  Real.log x / Real.log 8 = -1.75 → x = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_negative_seven_fourths_l1573_157396


namespace NUMINAMATH_CALUDE_store_prices_existence_l1573_157313

theorem store_prices_existence (S : ℕ) (h : S ≥ 100) :
  ∃ (T C B P : ℕ), T > C ∧ C > B ∧ T + C + B = S ∧ T * C * B = P ∧
  ∃ (T' C' B' : ℕ), (T', C', B') ≠ (T, C, B) ∧
    T' > C' ∧ C' > B' ∧ T' + C' + B' = S ∧ T' * C' * B' = P :=
by sorry

end NUMINAMATH_CALUDE_store_prices_existence_l1573_157313


namespace NUMINAMATH_CALUDE_movie_theater_receipts_l1573_157336

/-- 
Given a movie theater with the following conditions:
- Child ticket price is $4.50
- Adult ticket price is $6.75
- There are 20 more children than adults
- There are 48 children at the matinee

Prove that the total receipts for today's matinee is $405.
-/
theorem movie_theater_receipts : 
  let child_price : ℚ := 4.5
  let adult_price : ℚ := 6.75
  let child_count : ℕ := 48
  let adult_count : ℕ := child_count - 20
  let total_receipts : ℚ := child_price * child_count + adult_price * adult_count
  total_receipts = 405 := by sorry

end NUMINAMATH_CALUDE_movie_theater_receipts_l1573_157336


namespace NUMINAMATH_CALUDE_yellow_apples_probability_l1573_157304

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of an event -/
def probability (favorable_outcomes total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

theorem yellow_apples_probability :
  let total_apples : ℕ := 10
  let yellow_apples : ℕ := 5
  let selected_apples : ℕ := 3
  probability (choose yellow_apples selected_apples) (choose total_apples selected_apples) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_apples_probability_l1573_157304


namespace NUMINAMATH_CALUDE_probability_consecutive_days_l1573_157347

-- Define the number of days
def total_days : ℕ := 10

-- Define the number of days to be selected
def selected_days : ℕ := 3

-- Define the number of ways to select 3 consecutive days
def consecutive_selections : ℕ := total_days - selected_days + 1

-- Define the total number of ways to select 3 days from 10 days
def total_selections : ℕ := Nat.choose total_days selected_days

-- Theorem statement
theorem probability_consecutive_days :
  (consecutive_selections : ℚ) / total_selections = 1 / 15 :=
sorry

end NUMINAMATH_CALUDE_probability_consecutive_days_l1573_157347


namespace NUMINAMATH_CALUDE_simplify_absolute_value_expression_l1573_157391

theorem simplify_absolute_value_expression 
  (a b c : ℝ) 
  (ha : |a| + a = 0) 
  (hab : |a * b| = a * b) 
  (hc : |c| - c = 0) : 
  |b| - |a + b| - |c - b| + |a - c| = b := by
  sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_expression_l1573_157391


namespace NUMINAMATH_CALUDE_thief_speed_l1573_157370

/-- Proves that given the initial conditions, the speed of the thief is 8 km/hr -/
theorem thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ)
  (h1 : initial_distance = 175 / 1000) -- Convert 175 meters to kilometers
  (h2 : policeman_speed = 10)
  (h3 : thief_distance = 700 / 1000) -- Convert 700 meters to kilometers
  : ∃ (thief_speed : ℝ), thief_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_thief_speed_l1573_157370


namespace NUMINAMATH_CALUDE_max_elevation_l1573_157341

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 200 * t - 20 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ t : ℝ, ∀ u : ℝ, s u ≤ s t ∧ s t = 500 := by
  sorry

end NUMINAMATH_CALUDE_max_elevation_l1573_157341


namespace NUMINAMATH_CALUDE_instrument_probability_l1573_157355

theorem instrument_probability (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 → 
  at_least_one = 3/5 → 
  two_or_more = 96 → 
  (((at_least_one * total) - two_or_more) : ℚ) / total = 48/100 := by
sorry

end NUMINAMATH_CALUDE_instrument_probability_l1573_157355


namespace NUMINAMATH_CALUDE_fraction_equality_l1573_157357

theorem fraction_equality (a b : ℚ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1573_157357


namespace NUMINAMATH_CALUDE_inverse_function_relation_l1573_157322

/-- Given a function h and its inverse f⁻¹, prove the relation between a and b --/
theorem inverse_function_relation (a b : ℝ) :
  (∀ x, 3 * x - 6 = (Function.invFun (fun x => a * x + b)) x - 2) →
  3 * a + 4 * b = 19 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_relation_l1573_157322


namespace NUMINAMATH_CALUDE_total_budget_allocation_l1573_157377

def budget_groceries : ℝ := 0.6
def budget_eating_out : ℝ := 0.2
def budget_transportation : ℝ := 0.1
def budget_rent : ℝ := 0.05
def budget_utilities : ℝ := 0.05

theorem total_budget_allocation :
  budget_groceries + budget_eating_out + budget_transportation + budget_rent + budget_utilities = 1 := by
  sorry

end NUMINAMATH_CALUDE_total_budget_allocation_l1573_157377


namespace NUMINAMATH_CALUDE_count_numbers_with_three_700_l1573_157367

def contains_three (n : Nat) : Bool :=
  n.repr.any (· = '3')

def count_numbers_with_three (upper_bound : Nat) : Nat :=
  (List.range upper_bound).filter contains_three |>.length

theorem count_numbers_with_three_700 :
  count_numbers_with_three 700 = 214 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_three_700_l1573_157367


namespace NUMINAMATH_CALUDE_exponent_calculation_l1573_157375

theorem exponent_calculation (a : ℝ) : (-a)^10 / (-a)^3 = -a^7 := by sorry

end NUMINAMATH_CALUDE_exponent_calculation_l1573_157375


namespace NUMINAMATH_CALUDE_unattainable_y_value_l1573_157378

theorem unattainable_y_value (x : ℝ) :
  x ≠ -2/3 → (x - 3) / (3 * x + 2) ≠ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_unattainable_y_value_l1573_157378


namespace NUMINAMATH_CALUDE_zeta_sum_eight_l1573_157364

theorem zeta_sum_eight (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 8)
  (h3 : ζ₁^4 + ζ₂^4 + ζ₃^4 = 26) :
  ζ₁^8 + ζ₂^8 + ζ₃^8 = 219 := by sorry

end NUMINAMATH_CALUDE_zeta_sum_eight_l1573_157364


namespace NUMINAMATH_CALUDE_car_speed_problem_l1573_157395

theorem car_speed_problem (train_speed_ratio : ℝ) (distance : ℝ) (train_stop_time : ℝ) :
  train_speed_ratio = 1.5 →
  distance = 75 →
  train_stop_time = 12.5 / 60 →
  ∃ (car_speed : ℝ),
    car_speed = 80 ∧
    distance = car_speed * (distance / car_speed) ∧
    distance = (train_speed_ratio * car_speed) * (distance / car_speed - train_stop_time) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1573_157395


namespace NUMINAMATH_CALUDE_point_on_y_axis_m_zero_l1573_157324

/-- A point P with coordinates (x, y) lies on the y-axis if and only if x = 0 -/
def lies_on_y_axis (P : ℝ × ℝ) : Prop := P.1 = 0

/-- The theorem states that if a point P(m,2) lies on the y-axis, then m = 0 -/
theorem point_on_y_axis_m_zero (m : ℝ) :
  lies_on_y_axis (m, 2) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_m_zero_l1573_157324


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l1573_157335

theorem integer_solutions_of_equation :
  ∀ m n : ℤ, m^2 + 2*m = n^4 + 20*n^3 + 104*n^2 + 40*n + 2003 →
  ((m = 128 ∧ n = 7) ∨ (m = 128 ∧ n = -17)) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l1573_157335


namespace NUMINAMATH_CALUDE_function_value_at_e_l1573_157348

open Real

theorem function_value_at_e (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, f x = 2 * (deriv f 1) * log x + x) →
  f (exp 1) = -2 + exp 1 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_e_l1573_157348


namespace NUMINAMATH_CALUDE_polynomial_equality_l1573_157337

theorem polynomial_equality (a b c : ℝ) : 
  ((a - b) - c = a - b - c) ∧ 
  (a - (b + c) = a - b - c) ∧ 
  (-(b + c - a) = a - b - c) ∧ 
  (a - (b - c) ≠ a - b - c) :=
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1573_157337


namespace NUMINAMATH_CALUDE_system_consistency_l1573_157318

/-- The system of equations is consistent if and only if a is 0, -2, or 54 -/
theorem system_consistency (x a : ℝ) : 
  (∃ x, (10 * x^2 + x - a - 11 = 0) ∧ (4 * x^2 + (a + 4) * x - 3 * a - 8 = 0)) ↔ 
  (a = 0 ∨ a = -2 ∨ a = 54) :=
by sorry

end NUMINAMATH_CALUDE_system_consistency_l1573_157318


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1573_157328

theorem sum_of_two_numbers (x : ℤ) : 
  x + 35 = 62 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1573_157328


namespace NUMINAMATH_CALUDE_sqrt_floor_equality_l1573_157372

theorem sqrt_floor_equality (n : ℕ) :
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ :=
by sorry

end NUMINAMATH_CALUDE_sqrt_floor_equality_l1573_157372


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1573_157308

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 3) :
  (2 - x) / (x - 3) + 1 / (3 - x) = 1 ↔ x = 2 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1573_157308


namespace NUMINAMATH_CALUDE_watch_cost_price_proof_l1573_157320

/-- The cost price of a watch satisfying certain selling conditions -/
def watch_cost_price : ℝ := 1400

/-- The selling price at a 10% loss -/
def selling_price_loss (cost : ℝ) : ℝ := cost * 0.9

/-- The selling price at a 4% gain -/
def selling_price_gain (cost : ℝ) : ℝ := cost * 1.04

theorem watch_cost_price_proof :
  (selling_price_gain watch_cost_price - selling_price_loss watch_cost_price = 196) ∧
  (watch_cost_price = 1400) := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_proof_l1573_157320


namespace NUMINAMATH_CALUDE_constant_b_value_l1573_157393

theorem constant_b_value (x y : ℝ) (b : ℝ) 
  (h1 : (7 * x + b * y) / (x - 2 * y) = 13)
  (h2 : x / (2 * y) = 5 / 2) :
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_b_value_l1573_157393


namespace NUMINAMATH_CALUDE_cosine_difference_l1573_157325

theorem cosine_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_difference_l1573_157325


namespace NUMINAMATH_CALUDE_silver_dollar_difference_l1573_157354

/-- The number of silver dollars owned by Mr. Chiu -/
def chiu_dollars : ℕ := 56

/-- The number of silver dollars owned by Mr. Phung -/
def phung_dollars : ℕ := chiu_dollars + 16

/-- The number of silver dollars owned by Mr. Ha -/
def ha_dollars : ℕ := 205 - phung_dollars - chiu_dollars

theorem silver_dollar_difference : ha_dollars - phung_dollars = 5 := by
  sorry

end NUMINAMATH_CALUDE_silver_dollar_difference_l1573_157354


namespace NUMINAMATH_CALUDE_ofelias_to_rileys_mistakes_ratio_l1573_157344

theorem ofelias_to_rileys_mistakes_ratio 
  (total_questions : ℕ) 
  (rileys_mistakes : ℕ) 
  (team_incorrect : ℕ) 
  (h1 : total_questions = 35)
  (h2 : rileys_mistakes = 3)
  (h3 : team_incorrect = 17) :
  (team_incorrect - rileys_mistakes) / rileys_mistakes = 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_ofelias_to_rileys_mistakes_ratio_l1573_157344


namespace NUMINAMATH_CALUDE_circle_symmetry_theorem_l1573_157323

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 4*y = 0

-- Define the line of symmetry
def symmetry_line (k b : ℝ) (x y : ℝ) : Prop := y = k*x + b

-- Define the symmetric circle centered at the origin
def symmetric_circle (x y : ℝ) : Prop := x^2 + y^2 = 20

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  symmetric_circle A.1 A.2 ∧ symmetric_circle B.1 B.2

-- Define the angle ACB
def angle_ACB (A B : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem circle_symmetry_theorem :
  ∃ (k b : ℝ) (A B : ℝ × ℝ),
    (∀ x y : ℝ, circle_C x y ↔ symmetric_circle (2*x - k*y + b) (2*y + k*x - k*b)) →
    k = 2 ∧ b = 5 ∧
    intersection_points A B ∧
    angle_ACB A B = 120 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_theorem_l1573_157323


namespace NUMINAMATH_CALUDE_derivative_of_x_minus_sin_l1573_157301

open Real

theorem derivative_of_x_minus_sin (x : ℝ) : 
  deriv (fun x => x - sin x) x = 1 - cos x := by
sorry

end NUMINAMATH_CALUDE_derivative_of_x_minus_sin_l1573_157301


namespace NUMINAMATH_CALUDE_difference_of_squares_403_397_l1573_157382

theorem difference_of_squares_403_397 : 403^2 - 397^2 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_403_397_l1573_157382


namespace NUMINAMATH_CALUDE_rope_length_theorem_l1573_157388

/-- Represents a rope that can be folded in a specific manner. -/
structure Rope where
  /-- The distance between points (2) and (3) in the final folding. -/
  distance_2_3 : ℝ
  /-- Assertion that the distance between points (2) and (3) is positive. -/
  distance_positive : distance_2_3 > 0

/-- Calculates the total length of the rope based on its properties. -/
def total_length (rope : Rope) : ℝ :=
  6 * rope.distance_2_3

/-- Theorem stating that for a rope with distance between points (2) and (3) equal to 20,
    the total length is 120. -/
theorem rope_length_theorem (rope : Rope) (h : rope.distance_2_3 = 20) :
  total_length rope = 120 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_theorem_l1573_157388


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1573_157310

/-- In a right-angled triangle ABC, the sum of arctan(b/(a+c)) and arctan(c/(a+b)) is equal to π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let triangle_abc : (ℝ × ℝ × ℝ) := (a, b, c)
  b^2 + c^2 = a^2 →
  Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1573_157310


namespace NUMINAMATH_CALUDE_group_sum_difference_l1573_157389

/-- S_n represents the sum of the n-th group in a sequence where
    the n-th group contains n consecutive natural numbers starting from
    n(n-1)/2 + 1 -/
def S (n : ℕ) : ℕ := n * (n^2 + 1) / 2

/-- The theorem states that S_16 - S_4 - S_1 = 2021 -/
theorem group_sum_difference : S 16 - S 4 - S 1 = 2021 := by
  sorry

end NUMINAMATH_CALUDE_group_sum_difference_l1573_157389


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1573_157385

-- Define the repeating decimals
def repeating_2 : ℚ := 2/9
def repeating_03 : ℚ := 1/33

-- Theorem statement
theorem sum_of_repeating_decimals : 
  repeating_2 + repeating_03 = 25/99 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1573_157385


namespace NUMINAMATH_CALUDE_range_of_a_l1573_157326

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - 1 / Real.exp x

theorem range_of_a (a : ℝ) :
  (f (a - 1) + f (2 * a^2) ≤ 0) → (-1 ≤ a ∧ a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1573_157326


namespace NUMINAMATH_CALUDE_min_value_of_trig_function_l1573_157387

theorem min_value_of_trig_function :
  ∃ (min : ℝ), min = -Real.sqrt 2 - 1 ∧
  ∀ (x : ℝ), 2 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_trig_function_l1573_157387


namespace NUMINAMATH_CALUDE_store_discount_percentage_l1573_157311

/-- Represents the pricing strategy and profit of a store selling turtleneck sweaters -/
theorem store_discount_percentage (C : ℝ) (D : ℝ) : 
  C > 0 → -- Cost price is positive
  (1.20 * C) * 1.25 * (1 - D / 100) = 1.35 * C → -- February selling price equals 35% profit
  D = 10 := by
  sorry

end NUMINAMATH_CALUDE_store_discount_percentage_l1573_157311


namespace NUMINAMATH_CALUDE_penny_draw_probability_l1573_157338

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 5

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 3

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of needing more than five draws to get the fourth shiny penny -/
def probability : ℚ := 31 / 56

theorem penny_draw_probability :
  probability = (Nat.choose 5 3 * Nat.choose 3 1 + Nat.choose 5 0 * Nat.choose 3 3) / Nat.choose total_pennies shiny_pennies ∧
  probability.num + probability.den = 87 := by sorry

end NUMINAMATH_CALUDE_penny_draw_probability_l1573_157338


namespace NUMINAMATH_CALUDE_girls_equal_barefoot_children_l1573_157303

theorem girls_equal_barefoot_children (B_b G_b G_s : ℕ) :
  B_b = G_s →
  B_b + G_b = G_b + G_s :=
by sorry

end NUMINAMATH_CALUDE_girls_equal_barefoot_children_l1573_157303


namespace NUMINAMATH_CALUDE_distance_point_to_line_l1573_157332

/-- The distance from a point to a vertical line -/
def distance_point_to_vertical_line (point : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |point.1 - line_x|

/-- Theorem: The distance from point (1, 2) to the line x = -2 is 3 -/
theorem distance_point_to_line : distance_point_to_vertical_line (1, 2) (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l1573_157332


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1573_157390

theorem rectangle_dimensions (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (harea : x * y = 36) (hperim : 2 * x + 2 * y = 30) :
  (x = 12 ∧ y = 3) ∨ (x = 3 ∧ y = 12) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1573_157390


namespace NUMINAMATH_CALUDE_money_problem_l1573_157346

theorem money_problem (c d : ℝ) 
  (h1 : 3 * c - 2 * d < 30)
  (h2 : 4 * c + d = 60) :
  c < 150 / 11 ∧ d > 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l1573_157346


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1573_157349

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1573_157349


namespace NUMINAMATH_CALUDE_existence_of_non_one_start_l1573_157352

def begins_with_same_digit (x : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧
  ∀ n : ℕ, n ≤ 2015 →
    ∃ k : ℕ, x^n = d * 10^k + (x^n - d * 10^k) ∧
             d * 10^k ≤ x^n ∧
             x^n < (d + 1) * 10^k

theorem existence_of_non_one_start :
  ∃ x : ℕ, begins_with_same_digit x ∧
    ∃ d : ℕ, d ≠ 1 ∧ d < 10 ∧
    ∀ n : ℕ, n ≤ 2015 →
      ∃ k : ℕ, x^n = d * 10^k + (x^n - d * 10^k) ∧
               d * 10^k ≤ x^n ∧
               x^n < (d + 1) * 10^k :=
by sorry

end NUMINAMATH_CALUDE_existence_of_non_one_start_l1573_157352


namespace NUMINAMATH_CALUDE_rational_sqrt_one_minus_ab_l1573_157362

theorem rational_sqrt_one_minus_ab (a b : ℚ) 
  (h : a^3 * b + a * b^3 + 2 * a^2 * b^2 + 2 * a + 2 * b + 1 = 0) : 
  ∃ q : ℚ, q^2 = 1 - a * b := by
  sorry

end NUMINAMATH_CALUDE_rational_sqrt_one_minus_ab_l1573_157362


namespace NUMINAMATH_CALUDE_division_simplification_l1573_157340

theorem division_simplification (a : ℝ) (h : a ≠ 0) : 6 * a^2 / (a / 2) = 12 * a := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1573_157340


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1573_157366

theorem complex_equation_solution (i : ℂ) (m : ℝ) : 
  i * i = -1 → (1 - m * i) / (i^3) = 1 + i → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1573_157366


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1573_157376

theorem weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 66 →
  avg_increase = 2.5 →
  ∃ (new_weight : ℝ), new_weight = replaced_weight + initial_count * avg_increase :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1573_157376


namespace NUMINAMATH_CALUDE_mp3_price_reduction_l1573_157394

/-- Given an item with a sale price of 112 after a 20% reduction,
    prove that its price after a 30% reduction would be 98. -/
theorem mp3_price_reduction (original_price : ℝ) : 
  (original_price * 0.8 = 112) → (original_price * 0.7 = 98) := by
  sorry

end NUMINAMATH_CALUDE_mp3_price_reduction_l1573_157394


namespace NUMINAMATH_CALUDE_point_on_y_axis_l1573_157302

/-- A point P with coordinates (m+2, m+1) lies on the y-axis if and only if its coordinates are (0, -1) -/
theorem point_on_y_axis (m : ℝ) : 
  (m + 2 = 0 ∧ ∃ y, (0, y) = (m + 2, m + 1)) ↔ (0, -1) = (m + 2, m + 1) :=
by sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l1573_157302


namespace NUMINAMATH_CALUDE_smallest_n_is_five_l1573_157321

/-- A triple of positive integers (x, y, z) such that x + y = 3z -/
structure SpecialTriple where
  x : ℕ+
  y : ℕ+
  z : ℕ+
  sum_condition : x + y = 3 * z

/-- The property that a positive integer n satisfies the condition -/
def SatisfiesCondition (n : ℕ+) : Prop :=
  ∃ (triples : Fin n → SpecialTriple),
    (∀ i j, i ≠ j → (triples i).x ≠ (triples j).x ∧ (triples i).y ≠ (triples j).y ∧ (triples i).z ≠ (triples j).z) ∧
    (∀ k : ℕ+, k ≤ 3*n → ∃ i, (triples i).x = k ∨ (triples i).y = k ∨ (triples i).z = k)

theorem smallest_n_is_five :
  SatisfiesCondition 5 ∧ ∀ m : ℕ+, m < 5 → ¬SatisfiesCondition m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_is_five_l1573_157321


namespace NUMINAMATH_CALUDE_eulers_formula_l1573_157333

/-- A convex polyhedron is represented by its number of vertices, edges, and faces. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for convex polyhedra -/
theorem eulers_formula (p : ConvexPolyhedron) : p.vertices + p.faces = p.edges + 2 := by
  sorry

end NUMINAMATH_CALUDE_eulers_formula_l1573_157333


namespace NUMINAMATH_CALUDE_product_no_x3_x2_terms_l1573_157309

theorem product_no_x3_x2_terms (p q : ℝ) : 
  (∀ x : ℝ, (x^2 + p*x + 8) * (x^2 - 3*x + q) = x^4 + (p*q - 24)*x + 8*q) → 
  p = 3 ∧ q = 1 := by
sorry

end NUMINAMATH_CALUDE_product_no_x3_x2_terms_l1573_157309


namespace NUMINAMATH_CALUDE_money_division_l1573_157369

theorem money_division (total : ℝ) (a b c : ℝ) 
  (h_total : total = 392)
  (h_a : a = b / 2)
  (h_b : b = c / 2)
  (h_sum : a + b + c = total) : c = 224 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l1573_157369


namespace NUMINAMATH_CALUDE_distance_walked_l1573_157358

theorem distance_walked (x t : ℝ) 
  (h1 : (x + 1) * (3/4 * t) = x * t) 
  (h2 : (x - 1) * (t + 3) = x * t) : 
  x * t = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l1573_157358


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l1573_157306

theorem shirt_price_reduction (original_price : ℝ) (first_reduction_percent : ℝ) (second_reduction_percent : ℝ) : 
  original_price = 20 →
  first_reduction_percent = 20 →
  second_reduction_percent = 40 →
  (1 - second_reduction_percent / 100) * ((1 - first_reduction_percent / 100) * original_price) = 9.60 := by
sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l1573_157306


namespace NUMINAMATH_CALUDE_problem_statement_l1573_157392

theorem problem_statement (a b x y : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h1 : a = 2 * b)
  (h2 : x = 3 * y)
  (h3 : a + b = x * y)
  (h4 : b = 4)
  (h5 : y = 2) :
  x * a = 48 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1573_157392


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1573_157374

theorem complex_magnitude_equation (t : ℝ) : 
  (t > 0 ∧ Complex.abs (t + 3 * Complex.I * Real.sqrt 2) * Complex.abs (8 - 3 * Complex.I) = 40) ↔ 
  t = Real.sqrt (286 / 73) := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1573_157374


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1573_157386

theorem coin_flip_probability : 
  let n : ℕ := 12  -- total number of flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 55/1024 := by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1573_157386


namespace NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l1573_157359

theorem five_integers_sum_20_product_420 : 
  ∃ (a b c d e : ℕ+), 
    (a.val + b.val + c.val + d.val + e.val = 20) ∧ 
    (a.val * b.val * c.val * d.val * e.val = 420) := by
  sorry

end NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l1573_157359


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l1573_157350

theorem chocolate_milk_probability :
  let n : ℕ := 5  -- number of days
  let k : ℕ := 4  -- number of successful days
  let p : ℚ := 2/3  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 80/243 := by
sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l1573_157350


namespace NUMINAMATH_CALUDE_triangle_side_length_l1573_157360

theorem triangle_side_length 
  (AB : ℝ) 
  (time_AB time_BC_CA : ℝ) 
  (h1 : AB = 1992)
  (h2 : time_AB = 24)
  (h3 : time_BC_CA = 166)
  : ∃ (BC : ℝ), BC = 6745 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1573_157360


namespace NUMINAMATH_CALUDE_cannot_make_24_l1573_157329

/-- Represents the four basic arithmetic operations -/
inductive Operation
| Add
| Sub
| Mul
| Div

/-- Applies an operation to two rational numbers -/
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => if b ≠ 0 then a / b else 0

/-- Checks if it's possible to get 24 using the given numbers and operations -/
def canMake24 (a b c d : ℚ) : Prop :=
  ∃ (op1 op2 op3 : Operation),
    (applyOp op3 (applyOp op2 (applyOp op1 a b) c) d = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a b) d) c = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a c) b) d = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a c) d) b = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a d) b) c = 24) ∨
    (applyOp op3 (applyOp op2 (applyOp op1 a d) c) b = 24)

theorem cannot_make_24 : ¬ canMake24 1 6 8 7 := by
  sorry

end NUMINAMATH_CALUDE_cannot_make_24_l1573_157329


namespace NUMINAMATH_CALUDE_factors_of_expression_factorization_of_expression_factorization_of_cube_difference_l1573_157398

variable (a b c x y z : ℝ)

-- Statement 1
theorem factors_of_expression :
  (a - b) ∣ (a^2*(b - c) + b^2*(c - a) + c^2*(a - b)) ∧
  (b - c) ∣ (a^2*(b - c) + b^2*(c - a) + c^2*(a - b)) ∧
  (c - a) ∣ (a^2*(b - c) + b^2*(c - a) + c^2*(a - b)) :=
by sorry

-- Statement 2
theorem factorization_of_expression :
  a^2*(b - c) + b^2*(c - a) + c^2*(a - b) = -(a - b)*(b - c)*(c - a) :=
by sorry

-- Statement 3
theorem factorization_of_cube_difference :
  (x + y + z)^3 - x^3 - y^3 - z^3 = 3*(x + y)*(y + z)*(z + x) :=
by sorry

end NUMINAMATH_CALUDE_factors_of_expression_factorization_of_expression_factorization_of_cube_difference_l1573_157398


namespace NUMINAMATH_CALUDE_solve_equations_l1573_157380

/-- Solutions to the quadratic equation x^2 - 6x + 3 = 0 -/
def solutions_eq1 : Set ℝ := {3 + Real.sqrt 6, 3 - Real.sqrt 6}

/-- Solutions to the equation x(x-2) = x-2 -/
def solutions_eq2 : Set ℝ := {2, 1}

theorem solve_equations :
  (∀ x ∈ solutions_eq1, x^2 - 6*x + 3 = 0) ∧
  (∀ x ∈ solutions_eq2, x*(x-2) = x-2) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l1573_157380


namespace NUMINAMATH_CALUDE_symmetric_point_exists_l1573_157343

def S : Set ℤ := {n : ℤ | ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ n = 19*a + 85*b}

theorem symmetric_point_exists : 
  ∃ (A : ℝ), ∀ (x y : ℤ), (x + y : ℝ) / 2 = A → 
    (x ∈ S ↔ y ∉ S) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_exists_l1573_157343


namespace NUMINAMATH_CALUDE_water_removal_proof_l1573_157383

/-- Represents the fraction of water remaining after n steps -/
def remainingWater (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of steps after which one eighth of the water remains -/
def stepsToOneEighth : ℕ := 14

theorem water_removal_proof :
  remainingWater stepsToOneEighth = 1/8 :=
sorry

end NUMINAMATH_CALUDE_water_removal_proof_l1573_157383


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1573_157307

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 65 / 6 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 1 / b₀ = 65 / 6 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1573_157307


namespace NUMINAMATH_CALUDE_sqrt_49284_squared_times_3_l1573_157315

theorem sqrt_49284_squared_times_3 : (Real.sqrt 49284)^2 * 3 = 147852 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49284_squared_times_3_l1573_157315


namespace NUMINAMATH_CALUDE_cooking_probability_l1573_157356

-- Define the set of courses
def Courses := Finset.range 4

-- Define the probability of selecting a specific course
def prob_select (course : Courses) : ℚ :=
  1 / Courses.card

-- State the theorem
theorem cooking_probability :
  ∃ (cooking : Courses), prob_select cooking = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_cooking_probability_l1573_157356


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l1573_157327

/-- The line x = k intersects the parabola x = -3y^2 + 2y + 7 at exactly one point if and only if k = 22/3 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 + 2 * y + 7) ↔ k = 22/3 := by
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l1573_157327


namespace NUMINAMATH_CALUDE_cannot_form_70_cents_l1573_157330

/-- Represents the types of coins available in the piggy bank -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a combination of coins -/
def CoinCombination := List Coin

/-- Calculates the total value of a coin combination in cents -/
def totalValue (comb : CoinCombination) : Nat :=
  comb.map coinValue |>.sum

/-- Predicate to check if a coin combination has exactly six coins -/
def hasSixCoins (comb : CoinCombination) : Prop :=
  comb.length = 6

theorem cannot_form_70_cents :
  ¬∃ (comb : CoinCombination), hasSixCoins comb ∧ totalValue comb = 70 :=
sorry

end NUMINAMATH_CALUDE_cannot_form_70_cents_l1573_157330


namespace NUMINAMATH_CALUDE_f_one_half_equals_two_l1573_157379

-- Define the function f
noncomputable def f (y : ℝ) : ℝ := (4 : ℝ) ^ y

-- State the theorem
theorem f_one_half_equals_two :
  f (1/2) = 2 :=
sorry

end NUMINAMATH_CALUDE_f_one_half_equals_two_l1573_157379


namespace NUMINAMATH_CALUDE_no_solution_exists_l1573_157361

theorem no_solution_exists : ¬∃ (x y z : ℕ+), x^(x.val) + y^(y.val) = 9^(z.val) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1573_157361


namespace NUMINAMATH_CALUDE_dot_product_bound_l1573_157353

theorem dot_product_bound (a b c m n p : ℝ) 
  (sum_abc : a + b + c = 1) 
  (sum_mnp : m + n + p = 1) : 
  -1 ≤ a*m + b*n + c*p ∧ a*m + b*n + c*p ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_dot_product_bound_l1573_157353


namespace NUMINAMATH_CALUDE_intersection_when_m_neg_one_intersection_empty_iff_m_nonnegative_l1573_157399

def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x | 2*m < x ∧ x < 1-m}

theorem intersection_when_m_neg_one :
  B (-1) ∩ A = {x | 1 < x ∧ x < 2} := by sorry

theorem intersection_empty_iff_m_nonnegative (m : ℝ) :
  A ∩ B m = ∅ ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_neg_one_intersection_empty_iff_m_nonnegative_l1573_157399
