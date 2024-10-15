import Mathlib

namespace NUMINAMATH_CALUDE_negation_equivalence_l3552_355217

theorem negation_equivalence (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → x ≠ 0 ∨ y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3552_355217


namespace NUMINAMATH_CALUDE_lucy_fish_count_l3552_355244

-- Define the given quantities
def current_fish : ℕ := 212
def additional_fish : ℕ := 68

-- Define the total fish Lucy wants to have
def total_fish : ℕ := current_fish + additional_fish

-- Theorem statement
theorem lucy_fish_count : total_fish = 280 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l3552_355244


namespace NUMINAMATH_CALUDE_black_squares_count_l3552_355242

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard :=
  (size : Nat)
  (has_black_corners : Bool)
  (alternating : Bool)

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard) : Nat :=
  sorry

/-- Theorem: A 32x32 checkerboard with black corners and alternating colors has 512 black squares -/
theorem black_squares_count (board : Checkerboard) :
  board.size = 32 ∧ board.has_black_corners ∧ board.alternating →
  count_black_squares board = 512 :=
sorry

end NUMINAMATH_CALUDE_black_squares_count_l3552_355242


namespace NUMINAMATH_CALUDE_interest_rate_calculation_interest_rate_proof_l3552_355223

theorem interest_rate_calculation (initial_investment : ℝ) 
  (first_rate : ℝ) (first_duration : ℝ) (second_duration : ℝ) 
  (final_value : ℝ) : ℝ :=
  let first_growth := initial_investment * (1 + first_rate * first_duration / 12)
  let second_rate := ((final_value / first_growth - 1) * 12 / second_duration) * 100
  second_rate

theorem interest_rate_proof (initial_investment : ℝ) 
  (first_rate : ℝ) (first_duration : ℝ) (second_duration : ℝ) 
  (final_value : ℝ) :
  initial_investment = 12000 ∧ 
  first_rate = 0.08 ∧ 
  first_duration = 3 ∧ 
  second_duration = 3 ∧ 
  final_value = 12980 →
  interest_rate_calculation initial_investment first_rate first_duration second_duration final_value = 24 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_interest_rate_proof_l3552_355223


namespace NUMINAMATH_CALUDE_valid_a_values_l3552_355224

def A : Set ℝ := {-1, 1/2, 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 = 1 ∧ a ≥ 0}

def full_eating (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def partial_eating (X Y : Set ℝ) : Prop :=
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

def valid_relationship (a : ℝ) : Prop :=
  full_eating A (B a) ∨ partial_eating A (B a)

theorem valid_a_values : {a : ℝ | valid_relationship a} = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_valid_a_values_l3552_355224


namespace NUMINAMATH_CALUDE_probability_of_sum_five_l3552_355227

def number_of_faces : ℕ := 6

def total_outcomes (n : ℕ) : ℕ := n * n

def favorable_outcomes : List (ℕ × ℕ) := [(1, 4), (2, 3), (3, 2), (4, 1)]

def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

theorem probability_of_sum_five :
  probability (favorable_outcomes.length) (total_outcomes number_of_faces) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_five_l3552_355227


namespace NUMINAMATH_CALUDE_total_fuel_needed_l3552_355238

def fuel_consumption : ℝ := 5
def trip1_distance : ℝ := 30
def trip2_distance : ℝ := 20

theorem total_fuel_needed : 
  fuel_consumption * (trip1_distance + trip2_distance) = 250 := by
sorry

end NUMINAMATH_CALUDE_total_fuel_needed_l3552_355238


namespace NUMINAMATH_CALUDE_abs_neg_two_l3552_355297

theorem abs_neg_two : |(-2 : ℤ)| = 2 := by sorry

end NUMINAMATH_CALUDE_abs_neg_two_l3552_355297


namespace NUMINAMATH_CALUDE_inequality_theorem_l3552_355204

theorem inequality_theorem (a : ℚ) (x : ℝ) :
  ((a > 1 ∨ a < 0) ∧ x > 0 ∧ x ≠ 1 → x^(a : ℝ) - a * x + a - 1 > 0) ∧
  (0 < a ∧ a < 1 ∧ x > 0 ∧ x ≠ 1 → x^(a : ℝ) - a * x + a - 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3552_355204


namespace NUMINAMATH_CALUDE_rectangle_diagonals_equal_diagonals_equal_not_always_rectangle_not_rectangle_diagonals_not_equal_not_always_diagonals_not_equal_not_rectangle_l3552_355275

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define what it means for a quadrilateral to be a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  sorry

-- Define what it means for diagonals to be equal
def diagonals_equal (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statements
theorem rectangle_diagonals_equal (q : Quadrilateral) :
  is_rectangle q → diagonals_equal q :=
sorry

theorem diagonals_equal_not_always_rectangle :
  ∃ q : Quadrilateral, diagonals_equal q ∧ ¬is_rectangle q :=
sorry

theorem not_rectangle_diagonals_not_equal_not_always :
  ∃ q : Quadrilateral, ¬is_rectangle q ∧ diagonals_equal q :=
sorry

theorem diagonals_not_equal_not_rectangle (q : Quadrilateral) :
  ¬diagonals_equal q → ¬is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_rectangle_diagonals_equal_diagonals_equal_not_always_rectangle_not_rectangle_diagonals_not_equal_not_always_diagonals_not_equal_not_rectangle_l3552_355275


namespace NUMINAMATH_CALUDE_probability_one_of_each_interpreter_l3552_355235

def team_size : ℕ := 5
def english_interpreters : ℕ := 3
def russian_interpreters : ℕ := 2

theorem probability_one_of_each_interpreter :
  let total_combinations := Nat.choose team_size 2
  let favorable_combinations := Nat.choose english_interpreters 1 * Nat.choose russian_interpreters 1
  (favorable_combinations : ℚ) / total_combinations = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_interpreter_l3552_355235


namespace NUMINAMATH_CALUDE_cube_root_simplification_l3552_355268

theorem cube_root_simplification :
  (8 + 27) ^ (1/3) * (8 + 27^(1/3)) ^ (1/3) = 385 ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l3552_355268


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3552_355252

theorem remainder_divisibility (x : ℤ) : x % 72 = 19 → x % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3552_355252


namespace NUMINAMATH_CALUDE_vacation_cost_l3552_355270

theorem vacation_cost (C : ℝ) : 
  (C / 3 - C / 4 = 60) → C = 720 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_l3552_355270


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l3552_355249

theorem magnitude_of_complex_fourth_power : 
  Complex.abs ((4 : ℂ) + (3 * Complex.I * Real.sqrt 3))^4 = 1849 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fourth_power_l3552_355249


namespace NUMINAMATH_CALUDE_candy_distribution_l3552_355203

theorem candy_distribution (C n : ℕ) 
  (h1 : C = 8 * n + 4)
  (h2 : C = 11 * (n - 1)) : 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3552_355203


namespace NUMINAMATH_CALUDE_milton_zoology_books_l3552_355261

theorem milton_zoology_books : 
  ∀ (z b : ℕ), 
    z + b = 80 → 
    b = 4 * z → 
    z = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_milton_zoology_books_l3552_355261


namespace NUMINAMATH_CALUDE_village_households_l3552_355200

/-- The number of households in a village where:
    - Each household uses 150 litres of water per month
    - 6000 litres of water lasts for 4 months for all households
-/
def number_of_households : ℕ := 10

/-- Water usage per household per month in litres -/
def water_per_household_per_month : ℕ := 150

/-- Total water available in litres -/
def total_water : ℕ := 6000

/-- Number of months the water lasts -/
def months : ℕ := 4

theorem village_households : 
  number_of_households * water_per_household_per_month * months = total_water :=
sorry

end NUMINAMATH_CALUDE_village_households_l3552_355200


namespace NUMINAMATH_CALUDE_binary_sum_equals_11101101_l3552_355237

/-- The sum of specific binary numbers equals 11101101₂ -/
theorem binary_sum_equals_11101101 :
  (0b10101 : Nat) + (0b11 : Nat) + (0b1010 : Nat) + (0b11100 : Nat) + (0b1101 : Nat) = (0b11101101 : Nat) := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_11101101_l3552_355237


namespace NUMINAMATH_CALUDE_point_on_line_l3552_355286

/-- Given a line with equation x + 2y + 5 = 0, if (m, n) and (m + 2, n + k) are two points on this line,
    then k = -1 -/
theorem point_on_line (m n k : ℝ) : 
  (m + 2*n + 5 = 0) → ((m + 2) + 2*(n + k) + 5 = 0) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3552_355286


namespace NUMINAMATH_CALUDE_frank_work_hours_l3552_355272

/-- The number of hours Frank worked per day -/
def hours_per_day : ℕ := 8

/-- The number of days Frank worked -/
def days_worked : ℕ := 4

/-- The total number of hours Frank worked -/
def total_hours : ℕ := hours_per_day * days_worked

theorem frank_work_hours : total_hours = 32 := by
  sorry

end NUMINAMATH_CALUDE_frank_work_hours_l3552_355272


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3552_355266

-- Define the propositions p and q
def p (x : ℝ) : Prop := x > 4
def q (x : ℝ) : Prop := x^2 - 5*x + 4 ≥ 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3552_355266


namespace NUMINAMATH_CALUDE_problem_statement_l3552_355234

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  (2/(a-1) + 1/(b-2) ≥ 2) ∧ (2*a + b ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3552_355234


namespace NUMINAMATH_CALUDE_finite_rule2_applications_l3552_355291

/-- Represents the state of the blackboard -/
def Blackboard := List ℤ

/-- Rule 1: If there's a pair of equal numbers, add a to one and b to the other -/
def applyRule1 (board : Blackboard) (a b : ℕ) : Blackboard :=
  sorry

/-- Rule 2: If there's no pair of equal numbers, write two zeros -/
def applyRule2 : Blackboard := [0, 0]

/-- Applies either Rule 1 or Rule 2 based on the current board state -/
def applyRule (board : Blackboard) (a b : ℕ) : Blackboard :=
  sorry

/-- Represents a sequence of rule applications -/
def RuleSequence := List (Blackboard → Blackboard)

/-- Counts the number of times Rule 2 is applied in a sequence -/
def countRule2Applications (seq : RuleSequence) : ℕ :=
  sorry

/-- The main theorem: Rule 2 is applied only finitely many times -/
theorem finite_rule2_applications (a b : ℕ) (h : a ≠ b) :
  ∃ N : ℕ, ∀ seq : RuleSequence, countRule2Applications seq ≤ N :=
  sorry

end NUMINAMATH_CALUDE_finite_rule2_applications_l3552_355291


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_l3552_355294

theorem tangent_line_perpendicular (a : ℝ) : 
  let f (x : ℝ) := Real.exp (2 * a * x)
  let f' (x : ℝ) := 2 * a * Real.exp (2 * a * x)
  let tangent_slope := f' 0
  let perpendicular_line_slope := -1 / 2
  (tangent_slope = perpendicular_line_slope) → a = -1/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_l3552_355294


namespace NUMINAMATH_CALUDE_sum_of_digits_in_period_l3552_355255

def period_length (n : ℕ) : ℕ := sorry

def decimal_expansion (n : ℕ) : List ℕ := sorry

theorem sum_of_digits_in_period (n : ℕ) (h : n = 98^2) :
  let m := period_length n
  let digits := decimal_expansion n
  List.sum (List.take m digits) = 900 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_period_l3552_355255


namespace NUMINAMATH_CALUDE_dining_room_tiles_l3552_355211

/-- Calculates the total number of tiles needed for a rectangular room with a border --/
def total_tiles (room_length room_width border_width : ℕ) : ℕ :=
  let border_tiles := 2 * (room_length + room_width - 4 * border_width)
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  let inner_tiles := (inner_area + 3) / 4  -- Ceiling division by 4
  border_tiles + inner_tiles

/-- Theorem stating that a 15ft by 18ft room with a 2ft border requires 139 tiles --/
theorem dining_room_tiles : total_tiles 18 15 2 = 139 := by
  sorry

end NUMINAMATH_CALUDE_dining_room_tiles_l3552_355211


namespace NUMINAMATH_CALUDE_journal_writing_sessions_per_week_l3552_355215

theorem journal_writing_sessions_per_week 
  (pages_per_session : ℕ) 
  (total_pages : ℕ) 
  (total_weeks : ℕ) 
  (h1 : pages_per_session = 4) 
  (h2 : total_pages = 72) 
  (h3 : total_weeks = 6) : 
  (total_pages / pages_per_session) / total_weeks = 3 := by
sorry

end NUMINAMATH_CALUDE_journal_writing_sessions_per_week_l3552_355215


namespace NUMINAMATH_CALUDE_nuts_in_third_box_l3552_355245

-- Define the weights of nuts in each box
def box1 (x y z : ℝ) : ℝ := y + z - 6
def box2 (x y z : ℝ) : ℝ := x + z - 10

-- Theorem statement
theorem nuts_in_third_box (x y z : ℝ) 
  (h1 : x = box1 x y z) 
  (h2 : y = box2 x y z) : 
  z = 16 := by
sorry

end NUMINAMATH_CALUDE_nuts_in_third_box_l3552_355245


namespace NUMINAMATH_CALUDE_chase_travel_time_l3552_355231

/-- Represents the travel time between Granville and Salisbury -/
def travel_time (speed : ℝ) : ℝ := sorry

theorem chase_travel_time :
  let chase_speed : ℝ := 1
  let cameron_speed : ℝ := 2 * chase_speed
  let danielle_speed : ℝ := 3 * cameron_speed
  let danielle_time : ℝ := 30

  travel_time chase_speed = 180 := by sorry

end NUMINAMATH_CALUDE_chase_travel_time_l3552_355231


namespace NUMINAMATH_CALUDE_angle_bisector_points_sum_l3552_355201

theorem angle_bisector_points_sum (a b : ℝ) : 
  ((-4 : ℝ) = -4 ∧ a = -4) → 
  ((-2 : ℝ) = -2 ∧ b = -2) → 
  a + b + a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_points_sum_l3552_355201


namespace NUMINAMATH_CALUDE_students_not_enrolled_l3552_355282

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 79)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l3552_355282


namespace NUMINAMATH_CALUDE_cosine_power_sum_l3552_355246

theorem cosine_power_sum (α : ℝ) (n : ℤ) (x : ℝ) (hx : x ≠ 0) :
  x + 1/x = 2 * Real.cos α →
  x^n + 1/x^n = 2 * Real.cos (n * α) := by
sorry

end NUMINAMATH_CALUDE_cosine_power_sum_l3552_355246


namespace NUMINAMATH_CALUDE_quadratic_properties_l3552_355212

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : quadratic a b c 0 = 2)
  (h2 : quadratic a b c 1 = 2)
  (h3 : quadratic a b c (3/2) < 0)
  (h4 : ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0)
  (h5 : ∃ x : ℝ, -1/2 < x ∧ x < 0 ∧ quadratic a b c x = 0) :
  (∀ x ≤ 0, ∀ y ≤ x, quadratic a b c y ≤ quadratic a b c x) ∧
  (3 * quadratic a b c (-1) - quadratic a b c 2 < -20/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3552_355212


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_l3552_355262

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 + 16 * x + 1

-- State the theorem
theorem range_of_f_on_interval :
  let a := 1
  let b := 2
  (∀ x ≤ -2, ∀ y ∈ Set.Ioo x (-2), f x ≥ f y) →
  (∀ x ≥ -2, ∀ y ∈ Set.Ioo (-2) x, f x ≥ f y) →
  Set.range (fun x ↦ f x) ∩ Set.Icc a b = Set.Icc (f a) (f b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_l3552_355262


namespace NUMINAMATH_CALUDE_actual_distance_walked_l3552_355202

/-- 
Given a person who walks at two different speeds for the same duration:
- At 5 km/hr, they cover a distance D
- At 15 km/hr, they would cover a distance D + 20 km
This theorem proves that the actual distance D is 10 km.
-/
theorem actual_distance_walked (D : ℝ) : 
  (D / 5 = (D + 20) / 15) → D = 10 := by sorry

end NUMINAMATH_CALUDE_actual_distance_walked_l3552_355202


namespace NUMINAMATH_CALUDE_power_calculation_l3552_355264

theorem power_calculation : (-1/2 : ℚ)^2023 * 2^2022 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3552_355264


namespace NUMINAMATH_CALUDE_prism_path_lengths_l3552_355269

/-- Regular triangular prism with given properties -/
structure RegularTriangularPrism where
  -- Base edge length
  ab : ℝ
  -- Height
  aa1 : ℝ
  -- Point on base edge BC
  p : ℝ × ℝ × ℝ
  -- Shortest path length from P to M
  shortest_path : ℝ

/-- Theorem stating the lengths of PC and NC in the given prism -/
theorem prism_path_lengths (prism : RegularTriangularPrism)
  (h_ab : prism.ab = 3)
  (h_aa1 : prism.aa1 = 4)
  (h_path : prism.shortest_path = Real.sqrt 29) :
  ∃ (pc nc : ℝ), pc = 2 ∧ nc = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_prism_path_lengths_l3552_355269


namespace NUMINAMATH_CALUDE_binomial_seven_two_minus_three_l3552_355251

theorem binomial_seven_two_minus_three : Nat.choose 7 2 - 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_binomial_seven_two_minus_three_l3552_355251


namespace NUMINAMATH_CALUDE_greatest_prime_factor_eleven_l3552_355210

def f (m : ℕ) : ℕ := Finset.prod (Finset.range (m/2)) (fun i => 2*(i+1))

theorem greatest_prime_factor_eleven (m : ℕ) (h1 : m > 0) (h2 : Even m) :
  (∀ p : ℕ, Prime p → p ∣ f m → p ≤ 11) ∧
  (11 ∣ f m) →
  m = 22 := by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_eleven_l3552_355210


namespace NUMINAMATH_CALUDE_safe_dish_fraction_is_one_ninth_l3552_355214

/-- Represents a restaurant menu with vegan and nut-containing dishes -/
structure Menu where
  total_dishes : ℕ
  vegan_dishes : ℕ
  vegan_with_nuts : ℕ
  vegan_fraction : Rat
  h_vegan_fraction : vegan_fraction = 1 / 3
  h_vegan_dishes : vegan_dishes = 6
  h_vegan_with_nuts : vegan_with_nuts = 4

/-- The fraction of dishes that are both vegan and nut-free -/
def safe_dish_fraction (m : Menu) : Rat :=
  (m.vegan_dishes - m.vegan_with_nuts) / m.total_dishes

/-- Theorem stating that the fraction of safe dishes is 1/9 -/
theorem safe_dish_fraction_is_one_ninth (m : Menu) : safe_dish_fraction m = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_safe_dish_fraction_is_one_ninth_l3552_355214


namespace NUMINAMATH_CALUDE_units_digit_G_100_l3552_355259

/-- Modified Fermat number -/
def G (n : ℕ) : ℕ := 3^(2^n) + 1

/-- Units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_100 : units_digit (G 100) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_100_l3552_355259


namespace NUMINAMATH_CALUDE_abc_product_magnitude_l3552_355289

theorem abc_product_magnitude (a b c : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  a ≠ b → b ≠ c → c ≠ a →
  (a + 1 / b^2 = b + 1 / c^2) →
  (b + 1 / c^2 = c + 1 / a^2) →
  |a * b * c| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_magnitude_l3552_355289


namespace NUMINAMATH_CALUDE_probability_three_suits_standard_deck_l3552_355290

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- The probability of drawing three cards from a standard deck
    and getting one each from ♠, ♥, and ♦ suits (in any order) -/
def probability_three_suits (d : Deck) : ℚ :=
  let total_outcomes := d.cards * (d.cards - 1) * (d.cards - 2)
  let favorable_outcomes := d.ranks * d.ranks * d.ranks * 6
  favorable_outcomes / total_outcomes

/-- Theorem stating the probability for a standard 52-card deck -/
theorem probability_three_suits_standard_deck :
  probability_three_suits ⟨52, 13, 4⟩ = 2197 / 22100 := by
  sorry

#eval probability_three_suits ⟨52, 13, 4⟩

end NUMINAMATH_CALUDE_probability_three_suits_standard_deck_l3552_355290


namespace NUMINAMATH_CALUDE_calculate_total_income_person_total_income_l3552_355219

/-- Calculates a person's total income based on given distributions --/
theorem calculate_total_income (children_share : Real) (wife_share : Real) 
  (orphan_donation_rate : Real) (final_amount : Real) : Real :=
  let total_distributed := children_share + wife_share
  let remaining_before_donation := 1 - total_distributed
  let orphan_donation := orphan_donation_rate * remaining_before_donation
  let final_share := remaining_before_donation - orphan_donation
  final_amount / final_share

/-- Proves that the person's total income is $150,000 --/
theorem person_total_income : 
  calculate_total_income 0.25 0.35 0.1 45000 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_income_person_total_income_l3552_355219


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l3552_355205

theorem cubic_function_derivative (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 + 2
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l3552_355205


namespace NUMINAMATH_CALUDE_KHSO4_moles_formed_l3552_355250

/-- Represents a chemical substance -/
inductive Substance
  | KOH
  | H2SO4
  | KHSO4
  | H2O

/-- Represents the balanced chemical equation -/
def balancedEquation : List (Nat × Substance) → List (Nat × Substance) → Prop :=
  fun reactants products =>
    reactants = [(1, Substance.KOH), (1, Substance.H2SO4)] ∧
    products = [(1, Substance.KHSO4), (1, Substance.H2O)]

/-- Theorem: The number of moles of KHSO4 formed is 2 -/
theorem KHSO4_moles_formed
  (koh_moles : Nat)
  (h2so4_moles : Nat)
  (h_koh : koh_moles = 2)
  (h_h2so4 : h2so4_moles = 2)
  (h_equation : balancedEquation [(1, Substance.KOH), (1, Substance.H2SO4)] [(1, Substance.KHSO4), (1, Substance.H2O)]) :
  (min koh_moles h2so4_moles) = 2 := by
  sorry

end NUMINAMATH_CALUDE_KHSO4_moles_formed_l3552_355250


namespace NUMINAMATH_CALUDE_distance_is_15_miles_l3552_355260

/-- Represents the walking scenario with distance, speed, and time. -/
structure WalkScenario where
  distance : ℝ
  speed : ℝ
  time : ℝ

/-- The original walking scenario. -/
def original : WalkScenario := sorry

/-- The scenario with increased speed. -/
def increased_speed : WalkScenario := sorry

/-- The scenario with decreased speed. -/
def decreased_speed : WalkScenario := sorry

theorem distance_is_15_miles :
  (∀ s : WalkScenario, s.distance = s.speed * s.time) →
  (increased_speed.speed = original.speed + 0.5) →
  (increased_speed.time = 4/5 * original.time) →
  (decreased_speed.speed = original.speed - 0.5) →
  (decreased_speed.time = original.time + 2.5) →
  (original.distance = increased_speed.distance) →
  (original.distance = decreased_speed.distance) →
  original.distance = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_15_miles_l3552_355260


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_l3552_355263

def P (x : ℕ) : ℕ := x^6 + x^5 + x^3 + 1

theorem sum_of_prime_factors (h1 : 23 ∣ 67208001) (h2 : P 20 = 67208001) :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (67208001 + 1))) id) = 781 :=
sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_l3552_355263


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l3552_355226

/-- A parabola is defined by the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k + 1/(4a)) where (h, k) is the vertex -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola -/
def our_parabola : Parabola :=
  { a := 4
    b := 8
    c := -1 }

theorem focus_of_our_parabola :
  focus our_parabola = (-1, -79/16) := by sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l3552_355226


namespace NUMINAMATH_CALUDE_clock_hands_at_30_degrees_48_times_daily_l3552_355240

/-- Represents a clock with an hour hand and a minute hand -/
structure Clock where
  hour_hand : ℝ
  minute_hand : ℝ

/-- The speed of the minute hand relative to the hour hand -/
def minute_hand_speed : ℝ := 12

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The angle between clock hands we're interested in -/
def target_angle : ℝ := 30

/-- Function to count the number of times the clock hands form the target angle in a day -/
def count_target_angle_occurrences (c : Clock) : ℕ :=
  sorry

theorem clock_hands_at_30_degrees_48_times_daily :
  ∀ c : Clock, count_target_angle_occurrences c = 48 :=
sorry

end NUMINAMATH_CALUDE_clock_hands_at_30_degrees_48_times_daily_l3552_355240


namespace NUMINAMATH_CALUDE_initial_distance_problem_l3552_355281

theorem initial_distance_problem (speed_A speed_B : ℝ) (start_time end_time : ℝ) :
  speed_A = 5 →
  speed_B = 7 →
  start_time = 1 →
  end_time = 3 →
  let time_walked := end_time - start_time
  let distance_A := speed_A * time_walked
  let distance_B := speed_B * time_walked
  let initial_distance := distance_A + distance_B
  initial_distance = 24 := by
  sorry

end NUMINAMATH_CALUDE_initial_distance_problem_l3552_355281


namespace NUMINAMATH_CALUDE_race_result_l3552_355299

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ → ℝ

/-- The race scenario -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner
  runner_c : Runner

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  r.distance = 100 ∧
  r.runner_a.speed > r.runner_b.speed ∧
  r.runner_b.speed > r.runner_c.speed ∧
  (∀ t, r.runner_a.position t = r.runner_a.speed * t) ∧
  (∀ t, r.runner_b.position t = r.runner_b.speed * t) ∧
  (∀ t, r.runner_c.position t = r.runner_c.speed * t) ∧
  (∃ t_a, r.runner_a.position t_a = r.distance ∧ r.runner_b.position t_a = r.distance - 10) ∧
  (∃ t_b, r.runner_b.position t_b = r.distance ∧ r.runner_c.position t_b = r.distance - 10)

/-- The theorem to be proved -/
theorem race_result (r : Race) (h : race_conditions r) :
  ∃ t, r.runner_a.position t = r.distance ∧ r.runner_c.position t = r.distance - 19 := by
  sorry

end NUMINAMATH_CALUDE_race_result_l3552_355299


namespace NUMINAMATH_CALUDE_class_average_age_l3552_355253

theorem class_average_age (initial_students : ℕ) (leaving_student_age : ℕ) (teacher_age : ℕ) (new_average : ℝ) :
  initial_students = 30 →
  leaving_student_age = 11 →
  teacher_age = 41 →
  new_average = 11 →
  (initial_students * (initial_average : ℝ) - leaving_student_age + teacher_age) / initial_students = new_average →
  initial_average = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_class_average_age_l3552_355253


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_bound_l3552_355276

theorem sum_of_fourth_powers_bound (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 ≤ 1) :
  (a+b)^4 + (a+c)^4 + (a+d)^4 + (b+c)^4 + (b+d)^4 + (c+d)^4 ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_bound_l3552_355276


namespace NUMINAMATH_CALUDE_consecutive_integers_base_sum_l3552_355243

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- Checks if a number is a valid base -/
def isValidBase (b : Nat) : Prop := b ≥ 2

theorem consecutive_integers_base_sum (C D : Nat) : 
  C.succ = D →
  isValidBase C →
  isValidBase D →
  isValidBase (C + D) →
  toBase10 [1, 4, 5] C + toBase10 [5, 6] D = toBase10 [9, 2] (C + D) →
  C + D = 11 := by
  sorry

#check consecutive_integers_base_sum

end NUMINAMATH_CALUDE_consecutive_integers_base_sum_l3552_355243


namespace NUMINAMATH_CALUDE_remove_six_for_average_l3552_355220

def original_list : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def removed_number : ℕ := 6

def remaining_list : List ℕ := original_list.filter (· ≠ removed_number)

def average (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

theorem remove_six_for_average :
  average remaining_list = 71/10 :=
sorry

end NUMINAMATH_CALUDE_remove_six_for_average_l3552_355220


namespace NUMINAMATH_CALUDE_units_digit_17_31_l3552_355280

theorem units_digit_17_31 : (17^31) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_31_l3552_355280


namespace NUMINAMATH_CALUDE_lous_shoes_monthly_goal_l3552_355208

/-- The number of shoes Lou's Shoes must sell each month -/
def monthly_goal (last_week : ℕ) (this_week : ℕ) (remaining : ℕ) : ℕ :=
  last_week + this_week + remaining

/-- Theorem stating the total number of shoes Lou's Shoes must sell each month -/
theorem lous_shoes_monthly_goal :
  monthly_goal 27 12 41 = 80 := by
  sorry

end NUMINAMATH_CALUDE_lous_shoes_monthly_goal_l3552_355208


namespace NUMINAMATH_CALUDE_seashell_points_sum_l3552_355228

/-- The total points earned for seashells collected by Joan, Jessica, and Jeremy -/
def total_points (joan_shells : ℕ) (joan_points : ℕ) (jessica_shells : ℕ) (jessica_points : ℕ) (jeremy_shells : ℕ) (jeremy_points : ℕ) : ℕ :=
  joan_shells * joan_points + jessica_shells * jessica_points + jeremy_shells * jeremy_points

/-- Theorem stating that the total points earned is 48 -/
theorem seashell_points_sum :
  total_points 6 2 8 3 12 1 = 48 := by
  sorry

end NUMINAMATH_CALUDE_seashell_points_sum_l3552_355228


namespace NUMINAMATH_CALUDE_multiple_of_x_l3552_355232

theorem multiple_of_x (x y z k : ℕ+) : 
  (k * x = 5 * y) ∧ (5 * y = 8 * z) ∧ (x + y + z = 33) → k = 40 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_x_l3552_355232


namespace NUMINAMATH_CALUDE_function_and_triangle_properties_l3552_355283

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - Real.cos (ω * x) ^ 2 - 1/2

theorem function_and_triangle_properties 
  (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_distance : ∀ x₁ x₂, f ω x₁ = f ω x₂ → x₂ - x₁ = π / ω ∨ x₂ - x₁ = -π / ω) 
  (A B C : ℝ) 
  (h_c : Real.sqrt 7 = 2 * Real.sin (A/2) * Real.sin (B/2))
  (h_fC : f ω C = 0) 
  (h_sinB : Real.sin B = 3 * Real.sin A) :
  (∀ k : ℤ, ∀ x ∈ Set.Icc (-π/6 + k*π) (k*π + π/3), 
    ∀ y ∈ Set.Icc (-π/6 + k*π) (k*π + π/3), 
    x ≤ y → f ω x ≤ f ω y) ∧
  2 * Real.sin (A/2) = 1 ∧ 
  2 * Real.sin (B/2) = 3 := by
sorry

end NUMINAMATH_CALUDE_function_and_triangle_properties_l3552_355283


namespace NUMINAMATH_CALUDE_profit_percentage_l3552_355273

/-- Given that the cost price of 150 articles equals the selling price of 120 articles,
    prove that the percent profit is 25%. -/
theorem profit_percentage (cost selling : ℝ) (h : 150 * cost = 120 * selling) :
  (selling - cost) / cost * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3552_355273


namespace NUMINAMATH_CALUDE_sqrt_5_is_simplest_l3552_355221

/-- A quadratic radical is considered simplest if it cannot be simplified further
    and does not have denominators under the square root. -/
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, x = Real.sqrt y → (∀ z : ℝ, y ≠ z^2) ∧ (∀ n : ℕ, n > 1 → y ≠ Real.sqrt n)

/-- The theorem states that √5 is the simplest quadratic radical among the given options. -/
theorem sqrt_5_is_simplest :
  is_simplest_quadratic_radical (Real.sqrt 5) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 8) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (a^2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (0.2 * b)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_5_is_simplest_l3552_355221


namespace NUMINAMATH_CALUDE_angle_C_is_30_degrees_l3552_355287

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = Real.pi

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  3 * Real.sin t.A + 4 * Real.cos t.B = 6

def condition2 (t : Triangle) : Prop :=
  4 * Real.sin t.B + 3 * Real.cos t.A = 1

-- Theorem statement
theorem angle_C_is_30_degrees (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.C = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_30_degrees_l3552_355287


namespace NUMINAMATH_CALUDE_parabola_f_value_l3552_355241

/-- Represents a parabola of the form x = dy² + ey + f -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.d * y^2 + p.e * y + p.f

theorem parabola_f_value (p : Parabola) :
  (p.x_coord 1 = -3) →  -- vertex at (-3, 1)
  (p.x_coord 3 = -1) →  -- passes through (-1, 3)
  (p.x_coord 0 = -2.5) →  -- passes through (-2.5, 0)
  p.f = -2.5 := by
  sorry

#check parabola_f_value

end NUMINAMATH_CALUDE_parabola_f_value_l3552_355241


namespace NUMINAMATH_CALUDE_min_value_expression_l3552_355285

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (3 * x) / (3 * x + 2 * y) + y / (2 * x + y) ≥ 4 * Real.sqrt 3 - 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3552_355285


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3552_355229

theorem two_numbers_problem (a b : ℕ) :
  a + b = 667 →
  Nat.lcm a b / Nat.gcd a b = 120 →
  ((a = 115 ∧ b = 552) ∨ (a = 552 ∧ b = 115)) ∨
  ((a = 232 ∧ b = 435) ∨ (a = 435 ∧ b = 232)) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3552_355229


namespace NUMINAMATH_CALUDE_x_range_l3552_355248

theorem x_range (x : ℝ) 
  (h1 : 1 / x < 3) 
  (h2 : 1 / x > -4) 
  (h3 : x^2 - 1 > 0) : 
  x > 1 ∨ x < -1 := by
sorry

end NUMINAMATH_CALUDE_x_range_l3552_355248


namespace NUMINAMATH_CALUDE_base_nine_addition_l3552_355207

/-- Represents a number in base 9 --/
def BaseNine : Type := List (Fin 9)

/-- Converts a base 9 number to a natural number --/
def to_nat (b : BaseNine) : ℕ :=
  b.foldr (λ d acc => 9 * acc + d.val) 0

/-- Adds two base 9 numbers --/
def add_base_nine (a b : BaseNine) : BaseNine :=
  sorry

theorem base_nine_addition :
  let a : BaseNine := [2, 5, 6]
  let b : BaseNine := [8, 5]
  let c : BaseNine := [1, 5, 5]
  let result : BaseNine := [5, 1, 7, 6]
  add_base_nine (add_base_nine a b) c = result := by
  sorry

end NUMINAMATH_CALUDE_base_nine_addition_l3552_355207


namespace NUMINAMATH_CALUDE_sequence_has_repeating_pair_l3552_355222

def is_valid_sequence (a : Fin 99 → Fin 10) : Prop :=
  ∀ n : Fin 98, (a n = 1 → a (n + 1) ≠ 2) ∧ (a n = 3 → a (n + 1) ≠ 4)

theorem sequence_has_repeating_pair (a : Fin 99 → Fin 10) (h : is_valid_sequence a) :
  ∃ k l : Fin 98, k ≠ l ∧ a k = a l ∧ a (k + 1) = a (l + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_has_repeating_pair_l3552_355222


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l3552_355254

theorem sum_of_three_squares (n : ℕ+) (h : ∃ m : ℕ, 3 * n + 1 = m^2) :
  ∃ a b c : ℕ, n + 1 = a^2 + b^2 + c^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l3552_355254


namespace NUMINAMATH_CALUDE_total_cost_of_seeds_bottles_not_enough_l3552_355230

-- Define the given values
def seed_price : ℝ := 9.48
def seed_amount : ℝ := 3.3
def bottle_capacity : ℝ := 0.35
def num_bottles : ℕ := 9

-- Theorem for the total cost of grass seeds
theorem total_cost_of_seeds : seed_price * seed_amount = 31.284 := by sorry

-- Theorem for the insufficiency of 9 bottles
theorem bottles_not_enough : seed_amount > (bottle_capacity * num_bottles) := by sorry

end NUMINAMATH_CALUDE_total_cost_of_seeds_bottles_not_enough_l3552_355230


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l3552_355278

theorem midpoint_distance_theorem (t : ℝ) : 
  let A : ℝ × ℝ := (2*t - 3, t)
  let B : ℝ × ℝ := (t - 1, 2*t + 4)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((midpoint.1 - A.1)^2 + (midpoint.2 - A.2)^2) = (t^2 + t) / 2 →
  t = -10 := by
sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l3552_355278


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3552_355233

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3552_355233


namespace NUMINAMATH_CALUDE_tennis_tournament_rounds_l3552_355274

theorem tennis_tournament_rounds :
  ∀ (rounds : ℕ)
    (games_per_round : List ℕ)
    (cans_per_game : ℕ)
    (balls_per_can : ℕ)
    (total_balls : ℕ),
  games_per_round = [8, 4, 2, 1] →
  cans_per_game = 5 →
  balls_per_can = 3 →
  total_balls = 225 →
  (List.sum games_per_round * cans_per_game * balls_per_can = total_balls) →
  rounds = 4 := by
sorry

end NUMINAMATH_CALUDE_tennis_tournament_rounds_l3552_355274


namespace NUMINAMATH_CALUDE_polynomial_monotonicity_l3552_355288

-- Define a polynomial function
variable (P : ℝ → ℝ)

-- Define strict monotonicity
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem polynomial_monotonicity
  (h1 : StrictlyMonotonic (λ x => P (P x)))
  (h2 : StrictlyMonotonic (λ x => P (P (P x))))
  : StrictlyMonotonic P := by
  sorry

end NUMINAMATH_CALUDE_polynomial_monotonicity_l3552_355288


namespace NUMINAMATH_CALUDE_rays_fish_market_rays_fish_market_specific_l3552_355267

/-- The number of customers who will not receive fish in Mr. Ray's fish market scenario -/
theorem rays_fish_market (total_customers : ℕ) (num_tuna : ℕ) (tuna_weight : ℕ) (customer_request : ℕ) : ℕ :=
  let total_fish := num_tuna * tuna_weight
  let served_customers := total_fish / customer_request
  total_customers - served_customers

/-- Proof of the specific scenario in Mr. Ray's fish market -/
theorem rays_fish_market_specific : rays_fish_market 100 10 200 25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rays_fish_market_rays_fish_market_specific_l3552_355267


namespace NUMINAMATH_CALUDE_valid_numbering_exists_l3552_355295

/-- Represents a numbering system for 7 contacts and 7 holes -/
def Numbering := Fin 7 → Fin 7

/-- Checks if a numbering system satisfies the alignment condition for all rotations -/
def isValidNumbering (n : Numbering) : Prop :=
  ∀ k : Fin 7, ∃ i : Fin 7, n i = (i + k : Fin 7)

/-- The main theorem stating that a valid numbering system exists -/
theorem valid_numbering_exists : ∃ n : Numbering, isValidNumbering n := by
  sorry


end NUMINAMATH_CALUDE_valid_numbering_exists_l3552_355295


namespace NUMINAMATH_CALUDE_gcd_f_x_l3552_355257

def f (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(12*x+7)*(3*x+11)

theorem gcd_f_x (x : ℤ) (h : ∃ k : ℤ, x = 18720 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 462 := by
  sorry

end NUMINAMATH_CALUDE_gcd_f_x_l3552_355257


namespace NUMINAMATH_CALUDE_christophers_age_l3552_355258

theorem christophers_age (christopher george ford : ℕ) : 
  george = christopher + 8 →
  ford = christopher - 2 →
  christopher + george + ford = 60 →
  christopher = 18 := by
sorry

end NUMINAMATH_CALUDE_christophers_age_l3552_355258


namespace NUMINAMATH_CALUDE_cone_height_increase_l3552_355271

theorem cone_height_increase (h r : ℝ) (h' : ℝ) : 
  h > 0 → r > 0 → 
  ((1/3) * Real.pi * r^2 * h') = 2.9 * ((1/3) * Real.pi * r^2 * h) → 
  (h' - h) / h = 1.9 := by
sorry

end NUMINAMATH_CALUDE_cone_height_increase_l3552_355271


namespace NUMINAMATH_CALUDE_planes_count_theorem_l3552_355239

/-- A straight line in 3D space -/
structure Line3D where
  -- Define necessary properties for a line

/-- A point in 3D space -/
structure Point3D where
  -- Define necessary properties for a point

/-- A plane in 3D space -/
structure Plane3D where
  -- Define necessary properties for a plane

/-- Predicate to check if a point is outside a line -/
def is_outside (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Predicate to check if three points are collinear -/
def are_collinear (p1 p2 p3 : Point3D) : Prop :=
  sorry

/-- Function to count the number of unique planes determined by a line and three points -/
def count_planes (l : Line3D) (p1 p2 p3 : Point3D) : Nat :=
  sorry

/-- Theorem stating the possible number of planes -/
theorem planes_count_theorem (l : Line3D) (A B C : Point3D) 
  (h1 : is_outside A l)
  (h2 : is_outside B l)
  (h3 : is_outside C l) :
  (count_planes l A B C = 1) ∨ (count_planes l A B C = 3) ∨ (count_planes l A B C = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_planes_count_theorem_l3552_355239


namespace NUMINAMATH_CALUDE_snow_probability_in_ten_days_l3552_355277

/-- Probability of snow on a given day -/
def snow_prob (day : ℕ) : ℚ :=
  if day ≤ 5 then 1/5 else 1/3

/-- Probability of temperature dropping below 0°C -/
def cold_prob : ℚ := 1/2

/-- Increase in snow probability when temperature drops below 0°C -/
def snow_prob_increase : ℚ := 1/10

/-- Adjusted probability of no snow on a given day -/
def adj_no_snow_prob (day : ℕ) : ℚ :=
  cold_prob * (1 - snow_prob day) + (1 - cold_prob) * (1 - snow_prob day - snow_prob_increase)

/-- Probability of no snow for the entire period -/
def no_snow_prob : ℚ :=
  (adj_no_snow_prob 1)^5 * (adj_no_snow_prob 6)^5

theorem snow_probability_in_ten_days :
  1 - no_snow_prob = 58806/59049 :=
sorry

end NUMINAMATH_CALUDE_snow_probability_in_ten_days_l3552_355277


namespace NUMINAMATH_CALUDE_product_and_reciprocal_relation_l3552_355206

theorem product_and_reciprocal_relation (x y : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_product : x * y = 16) 
  (h_reciprocal : 1 / x = 3 * (1 / y)) :
  2 * y - x = 24 - (4 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_relation_l3552_355206


namespace NUMINAMATH_CALUDE_smallest_number_remainder_l3552_355256

theorem smallest_number_remainder (n : ℕ) : 
  (n = 210) → 
  (n % 13 = 3) → 
  (∀ m : ℕ, m < n → m % 13 ≠ 3 ∨ m % 17 ≠ n % 17) → 
  n % 17 = 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_remainder_l3552_355256


namespace NUMINAMATH_CALUDE_a_arithmetic_l3552_355265

noncomputable def a (n : ℕ) : ℝ := sorry

noncomputable def S (n : ℕ) : ℝ := sorry

def q : ℝ := sorry

axiom q_neq_zero_one : q * (q - 1) ≠ 0

axiom sum_relation (n : ℕ) : (1 - q) * S n + q * a n = 1

axiom S_arithmetic : S 3 - S 9 = S 9 - S 6

theorem a_arithmetic : a 2 - a 8 = a 8 - a 5 := by sorry

end NUMINAMATH_CALUDE_a_arithmetic_l3552_355265


namespace NUMINAMATH_CALUDE_range_of_a_l3552_355279

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) 
  (h2 : ∃ x : ℝ, x^2 + 4*x + a = 0) : 
  Real.exp 1 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3552_355279


namespace NUMINAMATH_CALUDE_complement_of_B_in_A_l3552_355213

def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

theorem complement_of_B_in_A :
  (A \ B) = {0, 2, 6, 10} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_in_A_l3552_355213


namespace NUMINAMATH_CALUDE_total_smoothie_time_l3552_355284

/-- The time it takes to freeze ice cubes (in minutes) -/
def freezing_time : ℕ := 40

/-- The time it takes to make one smoothie (in minutes) -/
def smoothie_time : ℕ := 3

/-- The number of smoothies to be made -/
def num_smoothies : ℕ := 5

/-- Theorem stating the total time to make the smoothies -/
theorem total_smoothie_time : 
  freezing_time + num_smoothies * smoothie_time = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_smoothie_time_l3552_355284


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l3552_355236

theorem matrix_equation_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![43/7, -54/7; -33/14, 24/7]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l3552_355236


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3552_355218

theorem complex_equation_solution :
  ∃ z : ℂ, z^2 - 4*z + 21 = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3552_355218


namespace NUMINAMATH_CALUDE_consecutive_points_length_l3552_355216

/-- Given five consecutive points on a straight line, prove the length of the entire segment --/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (∃ (x : ℝ), b - a = 5 ∧ c - b = 2 * x ∧ d - c = x ∧ e - d = 4 ∧ c - a = 11) →
  e - a = 18 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_points_length_l3552_355216


namespace NUMINAMATH_CALUDE_election_votes_calculation_l3552_355293

theorem election_votes_calculation (total_votes : ℕ) : 
  (total_votes : ℝ) * 0.55 = (total_votes : ℝ) * 0.35 + 400 →
  total_votes = 2000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l3552_355293


namespace NUMINAMATH_CALUDE_determinant_k_value_l3552_355296

def determinant (a b c d e f g h i : ℝ) : ℝ :=
  a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h

def algebraic_cofactor_1_2 (a b c d e f g h i : ℝ) : ℝ :=
  -(b * i - c * h)

theorem determinant_k_value (k : ℝ) :
  algebraic_cofactor_1_2 4 2 k (-3) 5 4 (-1) 1 (-2) = -10 →
  k = -14 := by
  sorry

end NUMINAMATH_CALUDE_determinant_k_value_l3552_355296


namespace NUMINAMATH_CALUDE_shooting_probabilities_l3552_355209

/-- Probability of a person hitting a target -/
def prob_hit (p : ℝ) : ℝ := p

/-- Probability of missing at least once in n shots -/
def prob_miss_at_least_once (p : ℝ) (n : ℕ) : ℝ := 1 - p^n

/-- Probability of stopping exactly after n shots, given stopping after two consecutive misses -/
def prob_stop_after_n_shots (p : ℝ) (n : ℕ) : ℝ :=
  if n < 2 then 0
  else if n = 2 then (1 - p)^2
  else p * (prob_stop_after_n_shots p (n - 1)) + (1 - p) * p * (1 - p)^2

theorem shooting_probabilities :
  let pA := prob_hit (2/3)
  let pB := prob_hit (3/4)
  (prob_miss_at_least_once pA 4 = 65/81) ∧
  (prob_stop_after_n_shots pB 5 = 45/1024) :=
by sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l3552_355209


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l3552_355247

/-- The hyperbola equation -/
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x + 3*y - 1 = 0

/-- The condition for intersection based on slope comparison -/
def intersection_condition (b : ℝ) : Prop := b > 2/3

/-- The theorem stating that b > 1 is sufficient but not necessary for intersection -/
theorem hyperbola_line_intersection (b : ℝ) (h : b > 0) :
  (∀ x y, hyperbola x y b → line x y → intersection_condition b) ∧
  ¬(∀ b, intersection_condition b → b > 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l3552_355247


namespace NUMINAMATH_CALUDE_wildlife_sanctuary_count_l3552_355225

theorem wildlife_sanctuary_count (total_heads total_legs : ℕ) 
  (h1 : total_heads = 300) 
  (h2 : total_legs = 780) : ∃ (birds insects : ℕ),
  birds + insects = total_heads ∧
  2 * birds + 6 * insects = total_legs ∧
  birds = 255 := by
sorry

end NUMINAMATH_CALUDE_wildlife_sanctuary_count_l3552_355225


namespace NUMINAMATH_CALUDE_solution_set_is_ray_iff_l3552_355292

/-- The polynomial function representing the left side of the inequality -/
def f (a x : ℝ) : ℝ := x^3 - (a^2 + a + 1)*x^2 + (a^3 + a^2 + a)*x - a^3

/-- The set of solutions to the inequality for a given a -/
def SolutionSet (a : ℝ) : Set ℝ := {x : ℝ | f a x ≥ 0}

/-- A set is a ray if it's of the form [c, ∞) or (-∞, c] for some c ∈ ℝ -/
def IsRay (S : Set ℝ) : Prop :=
  ∃ c : ℝ, S = {x : ℝ | x ≥ c} ∨ S = {x : ℝ | x ≤ c}

/-- The main theorem: The solution set is a ray iff a = 1 or a = -1 -/
theorem solution_set_is_ray_iff (a : ℝ) :
  IsRay (SolutionSet a) ↔ a = 1 ∨ a = -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_ray_iff_l3552_355292


namespace NUMINAMATH_CALUDE_rational_root_condition_l3552_355298

theorem rational_root_condition (n : ℕ+) :
  (∃ (x : ℚ), x^(n : ℕ) + (2 + x)^(n : ℕ) + (2 - x)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_root_condition_l3552_355298
