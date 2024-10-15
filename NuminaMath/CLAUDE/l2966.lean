import Mathlib

namespace NUMINAMATH_CALUDE_malfunctioning_odometer_theorem_l2966_296685

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

end NUMINAMATH_CALUDE_malfunctioning_odometer_theorem_l2966_296685


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_m_for_solvable_inequality_l2966_296607

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for the solution set of f(x) < 8
theorem solution_set_f_less_than_8 :
  {x : ℝ | f x < 8} = {x : ℝ | -5/2 < x ∧ x < 3/2} := by sorry

-- Theorem for the range of m
theorem range_of_m_for_solvable_inequality :
  {m : ℝ | ∃ x, f x ≤ |3*m + 1|} = 
    {m : ℝ | m ≤ -5/3 ∨ m ≥ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_m_for_solvable_inequality_l2966_296607


namespace NUMINAMATH_CALUDE_baker_remaining_pastries_l2966_296667

/-- The number of pastries Baker made -/
def pastries_made : ℕ := 148

/-- The number of pastries Baker sold -/
def pastries_sold : ℕ := 103

/-- The number of pastries Baker still has -/
def pastries_remaining : ℕ := pastries_made - pastries_sold

theorem baker_remaining_pastries : pastries_remaining = 45 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_pastries_l2966_296667


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2966_296618

theorem point_in_fourth_quadrant (a : ℝ) :
  let A : ℝ × ℝ := (Real.sqrt a + 1, -3)
  A.1 > 0 ∧ A.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2966_296618


namespace NUMINAMATH_CALUDE_fence_area_inequality_l2966_296630

theorem fence_area_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

end NUMINAMATH_CALUDE_fence_area_inequality_l2966_296630


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2966_296621

/-- A line y = x - b intersects a circle (x-2)^2 + y^2 = 1 at two distinct points
    if and only if b is in the open interval (2 - √2, 2 + √2) -/
theorem line_circle_intersection (b : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = x₁ - b ∧ y₂ = x₂ - b ∧
    (x₁ - 2)^2 + y₁^2 = 1 ∧
    (x₂ - 2)^2 + y₂^2 = 1) ↔ 
  (2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2966_296621


namespace NUMINAMATH_CALUDE_sum_reciprocals_bounds_l2966_296687

theorem sum_reciprocals_bounds (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  (1 / a + 1 / b ≥ 4 / 3) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 3 ∧ 1 / x + 1 / y = 4 / 3) ∧
  (∀ M : ℝ, ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 3 ∧ 1 / x + 1 / y > M) :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bounds_l2966_296687


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l2966_296648

/-- The fixed point of the line mx - y + 2m + 1 = 0 for all real m is (-2, 1) -/
theorem fixed_point_of_line (m : ℝ) : 
  (∀ x y : ℝ, m * x - y + 2 * m + 1 = 0 → (x = -2 ∧ y = 1)) ∧ 
  (m * (-2) - 1 + 2 * m + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l2966_296648


namespace NUMINAMATH_CALUDE_probability_of_selecting_girl_l2966_296631

theorem probability_of_selecting_girl (num_boys num_girls : ℕ) 
  (h_boys : num_boys = 3) 
  (h_girls : num_girls = 2) : 
  (num_girls : ℚ) / ((num_boys + num_girls) : ℚ) = 2 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_selecting_girl_l2966_296631


namespace NUMINAMATH_CALUDE_larger_number_problem_l2966_296695

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2966_296695


namespace NUMINAMATH_CALUDE_first_month_sale_l2966_296642

/-- Given the sales data for a grocer over 6 months, prove that the first month's sale was 5420 --/
theorem first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) 
  (h1 : sale2 = 5660)
  (h2 : sale3 = 6200)
  (h3 : sale4 = 6350)
  (h4 : sale5 = 6500)
  (h5 : sale6 = 6470)
  (h6 : average = 6100) :
  let total := 6 * average
  let known_sales := sale2 + sale3 + sale4 + sale5 + sale6
  total - known_sales = 5420 := by
sorry

end NUMINAMATH_CALUDE_first_month_sale_l2966_296642


namespace NUMINAMATH_CALUDE_cos_zero_degrees_l2966_296649

theorem cos_zero_degrees : Real.cos (0 * π / 180) = 1 := by sorry

end NUMINAMATH_CALUDE_cos_zero_degrees_l2966_296649


namespace NUMINAMATH_CALUDE_pythagorean_sum_number_with_conditions_l2966_296670

def is_pythagorean_sum_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  c^2 + d^2 = 10 * a + b

def G (n : ℕ) : ℚ :=
  let c := (n / 10) % 10
  let d := n % 10
  (c + d : ℚ) / 9

def P (n : ℕ) : ℚ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (10 * a - 2 * c * d + b : ℚ) / 3

theorem pythagorean_sum_number_with_conditions :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧
    is_pythagorean_sum_number n ∧
    (∃ k : ℤ, G n = k) ∧
    P n = 3 →
  n = 3772 ∨ n = 3727 :=
sorry

end NUMINAMATH_CALUDE_pythagorean_sum_number_with_conditions_l2966_296670


namespace NUMINAMATH_CALUDE_jimin_english_score_l2966_296636

def jimin_scores (science social_studies english : ℕ) : Prop :=
  social_studies = science + 6 ∧
  science = 87 ∧
  (science + social_studies + english) / 3 = 92

theorem jimin_english_score :
  ∀ science social_studies english : ℕ,
  jimin_scores science social_studies english →
  english = 96 := by sorry

end NUMINAMATH_CALUDE_jimin_english_score_l2966_296636


namespace NUMINAMATH_CALUDE_average_fish_is_75_l2966_296643

/-- The number of fish in Boast Pool -/
def boast_pool : ℕ := 75

/-- The number of fish in Onum Lake -/
def onum_lake : ℕ := boast_pool + 25

/-- The number of fish in Riddle Pond -/
def riddle_pond : ℕ := onum_lake / 2

/-- The total number of fish in all three bodies of water -/
def total_fish : ℕ := boast_pool + onum_lake + riddle_pond

/-- The number of bodies of water -/
def num_bodies : ℕ := 3

/-- Theorem stating that the average number of fish in all three bodies of water is 75 -/
theorem average_fish_is_75 : total_fish / num_bodies = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_fish_is_75_l2966_296643


namespace NUMINAMATH_CALUDE_billy_tickets_l2966_296608

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def tickets_per_ride : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := (ferris_rides + bumper_rides) * tickets_per_ride

theorem billy_tickets : total_tickets = 50 := by
  sorry

end NUMINAMATH_CALUDE_billy_tickets_l2966_296608


namespace NUMINAMATH_CALUDE_sum_ac_l2966_296654

theorem sum_ac (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 42)
  (h2 : b + d = 5) : 
  a + c = 42 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_ac_l2966_296654


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2966_296609

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 1 * a 5 = 16)
  (h_fourth : a 4 = 8) :
  a 5 = 16 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2966_296609


namespace NUMINAMATH_CALUDE_f_minus_one_lt_f_one_l2966_296651

theorem f_minus_one_lt_f_one
  (f : ℝ → ℝ)
  (h_diff : Differentiable ℝ f)
  (h_eq : ∀ x, f x = x^2 + 2 * x * (deriv f 2)) :
  f (-1) < f 1 := by
sorry

end NUMINAMATH_CALUDE_f_minus_one_lt_f_one_l2966_296651


namespace NUMINAMATH_CALUDE_min_value_fraction_l2966_296678

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x - y + 2*z = 0) : 
  ∃ (m : ℝ), m = 8 ∧ ∀ k, k = y^2/(x*z) → k ≥ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2966_296678


namespace NUMINAMATH_CALUDE_mikes_net_salary_calculation_l2966_296605

-- Define the initial conditions
def freds_initial_salary : ℝ := 1000
def freds_bonus : ℝ := 500
def freds_investment_return : ℝ := 0.20
def mikes_salary_multiplier : ℝ := 10
def mikes_bonus_percentage : ℝ := 0.10
def mikes_investment_return : ℝ := 0.25
def mikes_salary_increase : ℝ := 0.40
def mikes_tax_rate : ℝ := 0.15

-- Define the theorem
theorem mikes_net_salary_calculation :
  let mikes_initial_salary := freds_initial_salary * mikes_salary_multiplier
  let mikes_initial_total := mikes_initial_salary * (1 + mikes_bonus_percentage)
  let mikes_investment_result := mikes_initial_total * (1 + mikes_investment_return)
  let mikes_new_salary := mikes_initial_salary * (1 + mikes_salary_increase)
  let mikes_tax := mikes_new_salary * mikes_tax_rate
  mikes_new_salary - mikes_tax = 11900 :=
by sorry

end NUMINAMATH_CALUDE_mikes_net_salary_calculation_l2966_296605


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_factorial_l2966_296624

theorem smallest_multiple_of_seven_factorial : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (k : ℕ), k < 7 → ¬(m ∣ Nat.factorial k)) ∧ 
  (m ∣ Nat.factorial 7) ∧
  (∀ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), k < 7 → ¬(n ∣ Nat.factorial k)) ∧ (n ∣ Nat.factorial 7) → m ≤ n) :=
by
  use 5040
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_factorial_l2966_296624


namespace NUMINAMATH_CALUDE_area_after_folding_l2966_296699

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (D : Point)
  (R : Point)
  (Q : Point)
  (C : Point)

/-- Calculates the area of a quadrilateral -/
def area_quadrilateral (quad : Quadrilateral) : ℝ := sorry

/-- Creates a rectangle with given dimensions -/
def create_rectangle (width : ℝ) (height : ℝ) : Rectangle := sorry

/-- Performs the folding operation on the rectangle -/
def fold_rectangle (rect : Rectangle) : Quadrilateral := sorry

theorem area_after_folding (width height : ℝ) :
  width = 5 →
  height = 8 →
  let rect := create_rectangle width height
  let folded := fold_rectangle rect
  area_quadrilateral folded = 11.5 := by sorry

end NUMINAMATH_CALUDE_area_after_folding_l2966_296699


namespace NUMINAMATH_CALUDE_seven_pow_minus_three_times_two_pow_eq_one_l2966_296645

theorem seven_pow_minus_three_times_two_pow_eq_one
  (m n : ℕ+) : 7^(m:ℕ) - 3 * 2^(n:ℕ) = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_seven_pow_minus_three_times_two_pow_eq_one_l2966_296645


namespace NUMINAMATH_CALUDE_parabola_vertex_vertex_coordinates_l2966_296693

/-- The vertex of a parabola y = a(x - h)^2 + k is the point (h, k) --/
theorem parabola_vertex (a h k : ℝ) :
  let f : ℝ → ℝ := fun x ↦ a * (x - h)^2 + k
  (h, k) = (h, f h) ∧ ∀ x, f x ≥ f h := by sorry

/-- The coordinates of the vertex of the parabola y = 2(x-1)^2 + 8 are (1, 8) --/
theorem vertex_coordinates :
  let f : ℝ → ℝ := fun x ↦ 2 * (x - 1)^2 + 8
  (1, 8) = (1, f 1) ∧ ∀ x, f x ≥ f 1 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_vertex_coordinates_l2966_296693


namespace NUMINAMATH_CALUDE_no_valid_pair_l2966_296612

def s : Finset ℤ := {2, 3, 4, 5, 9, 12, 18}
def b : Finset ℤ := {4, 5, 6, 7, 8, 11, 14, 19}

theorem no_valid_pair : ¬∃ (x y : ℤ), 
  x ∈ s ∧ y ∈ b ∧ 
  x % 3 = 2 ∧ y % 4 = 1 ∧ 
  (x % 2 = 0 ∧ y % 2 = 1 ∨ x % 2 = 1 ∧ y % 2 = 0) ∧ 
  x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_no_valid_pair_l2966_296612


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l2966_296640

/-- The total number of blocks Arthur walked -/
def total_blocks : ℕ := 8 + 16

/-- The number of blocks that are one-third of a mile each -/
def first_blocks : ℕ := 10

/-- The length of each of the first blocks in miles -/
def first_block_length : ℚ := 1 / 3

/-- The length of each additional block in miles -/
def additional_block_length : ℚ := 1 / 4

/-- The total distance Arthur walked in miles -/
def total_distance : ℚ :=
  first_blocks * first_block_length + 
  (total_blocks - first_blocks) * additional_block_length

theorem arthur_walk_distance : total_distance = 41 / 6 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l2966_296640


namespace NUMINAMATH_CALUDE_rent_calculation_l2966_296615

theorem rent_calculation (monthly_earnings : ℝ) 
  (h1 : monthly_earnings * 0.07 + monthly_earnings * 0.5 + 817 = monthly_earnings) : 
  monthly_earnings * 0.07 = 133 := by
  sorry

end NUMINAMATH_CALUDE_rent_calculation_l2966_296615


namespace NUMINAMATH_CALUDE_equation_result_l2966_296613

theorem equation_result (x : ℝ) : 
  14 * x + 5 - 21 * x^2 = -2 → 6 * x^2 - 4 * x + 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_l2966_296613


namespace NUMINAMATH_CALUDE_expression_evaluation_l2966_296619

theorem expression_evaluation :
  let x : ℝ := 3
  let numerator := 4 + x^2 - x*(2+x) - 2^2
  let denominator := x^2 - 2*x + 3
  numerator / denominator = -1 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2966_296619


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l2966_296611

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, 3^x + x < 0) ↔ (∀ x : ℝ, 3^x + x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l2966_296611


namespace NUMINAMATH_CALUDE_complex_modulus_l2966_296626

theorem complex_modulus (z : ℂ) : z = (1 + 2*Complex.I)/Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2966_296626


namespace NUMINAMATH_CALUDE_final_state_is_green_l2966_296688

/-- Represents the colors of chameleons -/
inductive Color
  | Yellow
  | Red
  | Green

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons -/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons -/
def totalChameleons : Nat := 34

/-- Simulates the color change when two different colored chameleons meet -/
def colorChange (state : ChameleonState) : ChameleonState :=
  sorry

/-- Checks if all chameleons are the same color -/
def allSameColor (state : ChameleonState) : Bool :=
  sorry

/-- Theorem: The only possible final state where all chameleons are the same color is green -/
theorem final_state_is_green (state : ChameleonState) :
  (state.yellow + state.red + state.green = totalChameleons) →
  (allSameColor state = true) →
  (state.green = totalChameleons ∧ state.yellow = 0 ∧ state.red = 0) :=
by sorry

end NUMINAMATH_CALUDE_final_state_is_green_l2966_296688


namespace NUMINAMATH_CALUDE_fixed_salary_is_400_l2966_296694

/-- Represents the fixed salary in the new commission scheme -/
def fixed_salary : ℕ := sorry

/-- Represents the total sales amount -/
def total_sales : ℕ := 12000

/-- Represents the threshold for commission in the new scheme -/
def commission_threshold : ℕ := 4000

/-- Calculates the commission under the old scheme -/
def old_commission (sales : ℕ) : ℕ :=
  (sales * 5) / 100

/-- Calculates the commission under the new scheme -/
def new_commission (sales : ℕ) : ℕ :=
  ((sales - commission_threshold) * 25) / 1000

/-- States that the new scheme pays 600 more than the old scheme -/
axiom new_scheme_difference : 
  fixed_salary + new_commission total_sales = old_commission total_sales + 600

theorem fixed_salary_is_400 : fixed_salary = 400 := by
  sorry

end NUMINAMATH_CALUDE_fixed_salary_is_400_l2966_296694


namespace NUMINAMATH_CALUDE_artist_paint_usage_l2966_296644

/-- The amount of paint used for all paintings --/
def total_paint_used (large_paint small_paint large_count small_count : ℕ) : ℕ :=
  large_paint * large_count + small_paint * small_count

/-- Proof that the artist used 17 ounces of paint --/
theorem artist_paint_usage : total_paint_used 3 2 3 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_artist_paint_usage_l2966_296644


namespace NUMINAMATH_CALUDE_golden_ratio_between_consecutive_integers_l2966_296652

theorem golden_ratio_between_consecutive_integers :
  ∃ (a b : ℤ), (a + 1 = b) ∧ (a < (Real.sqrt 5 + 1) / 2) ∧ ((Real.sqrt 5 + 1) / 2 < b) → a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_golden_ratio_between_consecutive_integers_l2966_296652


namespace NUMINAMATH_CALUDE_triangle_properties_l2966_296660

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if b * sin(A) = (√3/2) * a, a = 2c, and b = 2√6,
    then the measure of angle B is π/3 and the area of the triangle is 4√3. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- acute triangle condition
  b * Real.sin A = (Real.sqrt 3 / 2) * a →  -- given condition
  a = 2 * c →  -- given condition
  b = 2 * Real.sqrt 6 →  -- given condition
  B = π / 3 ∧ (1 / 2) * a * c * Real.sin B = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2966_296660


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l2966_296604

theorem opposite_of_negative_three : -((-3) : ℤ) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l2966_296604


namespace NUMINAMATH_CALUDE_largest_a_value_l2966_296650

theorem largest_a_value : ∃ (a_max : ℚ), 
  (∀ a : ℚ, (3 * a + 4) * (a - 2) = 7 * a → a ≤ a_max) ∧ 
  ((3 * a_max + 4) * (a_max - 2) = 7 * a_max) ∧
  a_max = 4 := by
sorry

end NUMINAMATH_CALUDE_largest_a_value_l2966_296650


namespace NUMINAMATH_CALUDE_triangle_area_angle_l2966_296655

theorem triangle_area_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (a^2 + b^2 - c^2) / (4 * Real.sqrt 3)
  ∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧
    S = 1/2 * a * b * Real.sin C ∧
    C = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_angle_l2966_296655


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2966_296690

/-- Given a point P(-2,3) in a plane rectangular coordinate system,
    its coordinates with respect to the origin are (2,-3). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-2, 3)
  let origin_symmetric (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  origin_symmetric P = (2, -3) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l2966_296690


namespace NUMINAMATH_CALUDE_thomas_worked_four_weeks_l2966_296661

/-- The number of whole weeks Thomas worked given his weekly rate and total amount paid -/
def weeks_worked (weekly_rate : ℕ) (total_amount : ℕ) : ℕ :=
  (total_amount / weekly_rate : ℕ)

/-- Theorem stating that Thomas worked for 4 weeks -/
theorem thomas_worked_four_weeks :
  weeks_worked 4550 19500 = 4 := by
  sorry

end NUMINAMATH_CALUDE_thomas_worked_four_weeks_l2966_296661


namespace NUMINAMATH_CALUDE_program_arrangements_l2966_296697

theorem program_arrangements (n : ℕ) (k : ℕ) : 
  n = 4 → k = 2 → (n + 1) * (n + 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_program_arrangements_l2966_296697


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2966_296623

theorem consecutive_even_integers_sum (x : ℕ) (h : x > 0) :
  (x - 2) * x * (x + 2) = 20 * ((x - 2) + x + (x + 2)) →
  (x - 2) + x + (x + 2) = 24 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2966_296623


namespace NUMINAMATH_CALUDE_sock_pair_count_l2966_296696

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_pairs (white brown blue green : ℕ) : ℕ :=
  white * brown + white * blue + white * green +
  brown * blue + brown * green +
  blue * green

/-- Theorem: There are 81 ways to choose a pair of socks of different colors
    from a drawer containing 5 white, 5 brown, 3 blue, and 2 green socks -/
theorem sock_pair_count :
  different_color_pairs 5 5 3 2 = 81 := by sorry

end NUMINAMATH_CALUDE_sock_pair_count_l2966_296696


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2966_296646

-- Define the variables and conditions
theorem sqrt_inequality (C : ℝ) (hC : C > 1) :
  Real.sqrt (C + 1) - Real.sqrt C < Real.sqrt C - Real.sqrt (C - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2966_296646


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2966_296616

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem: Given the specified conditions, the man's speed against the current is 10 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 15 2.5 = 10 := by
  sorry

#eval speed_against_current 15 2.5

end NUMINAMATH_CALUDE_mans_speed_against_current_l2966_296616


namespace NUMINAMATH_CALUDE_unique_intercept_line_l2966_296602

/-- A line passing through a point with equal absolute horizontal and vertical intercepts -/
structure InterceptLine where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  point_condition : 4 = m * 1 + b  -- line passes through (1, 4)
  intercept_condition : |m| = |b|  -- equal absolute intercepts

/-- There exists a unique line passing through (1, 4) with equal absolute horizontal and vertical intercepts -/
theorem unique_intercept_line : ∃! l : InterceptLine, True :=
  sorry

end NUMINAMATH_CALUDE_unique_intercept_line_l2966_296602


namespace NUMINAMATH_CALUDE_purely_imaginary_and_circle_l2966_296632

-- Define the complex number z
def z (a : ℝ) : ℂ := a * (1 + Complex.I) - 2 * Complex.I

-- State the theorem
theorem purely_imaginary_and_circle (a : ℝ) :
  (∃ b : ℝ, z a = Complex.I * b) →
  (a = 2 ∧ ∀ w : ℂ, Complex.abs w = 3 ↔ w.re ^ 2 + w.im ^ 2 = 3 ^ 2) :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_and_circle_l2966_296632


namespace NUMINAMATH_CALUDE_candy_distribution_l2966_296681

theorem candy_distribution (total_candy : Nat) (num_people : Nat) : 
  total_candy = 30 → num_people = 5 → 
  (∃ (pieces_per_person : Nat), total_candy = pieces_per_person * num_people) → 
  0 = total_candy - (total_candy / num_people) * num_people :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2966_296681


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_l2966_296698

-- Equation 1: x^2 - 2x = 0
theorem equation_one_solutions (x : ℝ) : 
  (x = 0 ∨ x = 2) ↔ x^2 - 2*x = 0 := by sorry

-- Equation 2: (2x-1)^2 = (3-x)^2
theorem equation_two_solutions (x : ℝ) : 
  (x = -2 ∨ x = 4/3) ↔ (2*x - 1)^2 = (3 - x)^2 := by sorry

-- Equation 3: 3x(x-2) = x-2
theorem equation_three_solutions (x : ℝ) : 
  (x = 2 ∨ x = 1/3) ↔ 3*x*(x - 2) = x - 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_l2966_296698


namespace NUMINAMATH_CALUDE_negation_of_statement_l2966_296629

theorem negation_of_statement : 
  (¬(∀ a : ℝ, a ≠ 0 → a^2 > 0)) ↔ (∃ a : ℝ, a = 0 ∧ a^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_statement_l2966_296629


namespace NUMINAMATH_CALUDE_tangent_line_polar_equation_l2966_296689

/-- The polar coordinate equation of the tangent line to the circle ρ = 4sin θ
    that passes through the point (2√2, π/4) is ρ cos θ = 2. -/
theorem tangent_line_polar_equation (ρ θ : ℝ) :
  (ρ = 4 * Real.sin θ) →  -- Circle equation
  (∃ (ρ₀ θ₀ : ℝ), ρ₀ = 2 * Real.sqrt 2 ∧ θ₀ = π / 4 ∧ 
    ρ₀ * Real.cos θ₀ = 2 ∧ ρ₀ * Real.sin θ₀ = 2) →  -- Point (2√2, π/4)
  (ρ * Real.cos θ = 2) -- Tangent line equation
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_polar_equation_l2966_296689


namespace NUMINAMATH_CALUDE_largest_initial_number_l2966_296639

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 + a + b + c + d + e = 200 ∧
    189 % a ≠ 0 ∧ 189 % b ≠ 0 ∧ 189 % c ≠ 0 ∧ 189 % d ≠ 0 ∧ 189 % e ≠ 0 ∧
    ∀ (n : ℕ), n > 189 →
      ¬∃ (x y z w v : ℕ),
        n + x + y + z + w + v = 200 ∧
        n % x ≠ 0 ∧ n % y ≠ 0 ∧ n % z ≠ 0 ∧ n % w ≠ 0 ∧ n % v ≠ 0 :=
by sorry


end NUMINAMATH_CALUDE_largest_initial_number_l2966_296639


namespace NUMINAMATH_CALUDE_sabrina_can_finish_series_l2966_296638

theorem sabrina_can_finish_series 
  (total_books : Nat) 
  (pages_per_book : Nat) 
  (books_read_first_month : Nat) 
  (reading_speed : Nat) 
  (total_days : Nat) 
  (h1 : total_books = 14)
  (h2 : pages_per_book = 200)
  (h3 : books_read_first_month = 4)
  (h4 : reading_speed = 40)
  (h5 : total_days = 60) :
  ∃ (pages_read : Nat), pages_read ≥ total_books * pages_per_book := by
  sorry

#check sabrina_can_finish_series

end NUMINAMATH_CALUDE_sabrina_can_finish_series_l2966_296638


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l2966_296682

theorem pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = y * Complex.I) : 
  p / q = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l2966_296682


namespace NUMINAMATH_CALUDE_select_male_and_female_prob_l2966_296666

/-- The probability of selecting one male and one female from a group of 2 females and 4 males -/
theorem select_male_and_female_prob (num_female : ℕ) (num_male : ℕ) : 
  num_female = 2 → num_male = 4 → 
  (num_male.choose 1 * num_female.choose 1 : ℚ) / ((num_male + num_female).choose 2) = 8 / 15 := by
  sorry

#check select_male_and_female_prob

end NUMINAMATH_CALUDE_select_male_and_female_prob_l2966_296666


namespace NUMINAMATH_CALUDE_weight_of_b_l2966_296617

/-- Given three weights a, b, and c, prove that b = 31 when:
    1. The average of a, b, and c is 45
    2. The average of a and b is 40
    3. The average of b and c is 43 -/
theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43) :
  b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l2966_296617


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2966_296672

theorem inequality_equivalence (x : ℝ) : 
  (6 * x - 2 < (x + 1)^2 ∧ (x + 1)^2 < 8 * x - 4) ↔ (3 < x ∧ x < 5) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2966_296672


namespace NUMINAMATH_CALUDE_balloons_in_park_l2966_296663

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 1

/-- The total number of balloons brought to the park -/
def total_balloons : ℕ := allan_balloons + jake_balloons

theorem balloons_in_park : total_balloons = 3 := by
  sorry

end NUMINAMATH_CALUDE_balloons_in_park_l2966_296663


namespace NUMINAMATH_CALUDE_inequality_solution_l2966_296662

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x y, 0 < x → x < y → f y < f x
axiom f_at_neg_three : f (-3) = 1

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ x > 3}

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f x < 1} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2966_296662


namespace NUMINAMATH_CALUDE_eight_power_problem_l2966_296600

theorem eight_power_problem (x : ℝ) (h : (8 : ℝ) ^ (3 * x) = 64) : (8 : ℝ) ^ (-x) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_eight_power_problem_l2966_296600


namespace NUMINAMATH_CALUDE_circumscribed_circle_diameter_l2966_296674

/-- The diameter of a triangle's circumscribed circle, given one side and its opposite angle -/
theorem circumscribed_circle_diameter 
  (side : ℝ) 
  (angle : ℝ) 
  (h1 : side = 15) 
  (h2 : angle = π / 4) : 
  side / Real.sin angle = 15 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_circumscribed_circle_diameter_l2966_296674


namespace NUMINAMATH_CALUDE_negation_of_implication_or_l2966_296675

theorem negation_of_implication_or (p q r : Prop) :
  ¬(r → p ∨ q) ↔ (¬r → ¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_or_l2966_296675


namespace NUMINAMATH_CALUDE_cheolsu_weight_l2966_296669

/-- Proves that Cheolsu's weight is 36 kg given the problem conditions -/
theorem cheolsu_weight (c m f : ℝ) 
  (h1 : (c + m + f) / 3 = m)  -- average equals mother's weight
  (h2 : c = (2/3) * m)        -- Cheolsu's weight is 2/3 of mother's
  (h3 : f = 72)               -- Father's weight is 72 kg
  : c = 36 := by
  sorry

#check cheolsu_weight

end NUMINAMATH_CALUDE_cheolsu_weight_l2966_296669


namespace NUMINAMATH_CALUDE_linear_equation_condition_l2966_296658

theorem linear_equation_condition (m : ℤ) : (|m| - 2 = 1 ∧ m - 3 ≠ 0) ↔ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l2966_296658


namespace NUMINAMATH_CALUDE_disease_test_probability_l2966_296665

theorem disease_test_probability (incidence_rate : ℝ) 
  (true_positive_rate : ℝ) (false_positive_rate : ℝ) :
  incidence_rate = 0.01 →
  true_positive_rate = 0.99 →
  false_positive_rate = 0.01 →
  let total_positive_rate := true_positive_rate * incidence_rate + 
    false_positive_rate * (1 - incidence_rate)
  (true_positive_rate * incidence_rate) / total_positive_rate = 0.5 := by
sorry

end NUMINAMATH_CALUDE_disease_test_probability_l2966_296665


namespace NUMINAMATH_CALUDE_fourth_level_open_spots_l2966_296684

-- Define the structure of the parking garage
structure ParkingGarage where
  total_levels : Nat
  spots_per_level : Nat
  open_spots_first : Nat
  open_spots_second : Nat
  open_spots_third : Nat
  full_spots_total : Nat

-- Define the problem instance
def parking_problem : ParkingGarage :=
  { total_levels := 4
  , spots_per_level := 100
  , open_spots_first := 58
  , open_spots_second := 60  -- 58 + 2
  , open_spots_third := 65   -- 60 + 5
  , full_spots_total := 186 }

-- Theorem statement
theorem fourth_level_open_spots :
  let p := parking_problem
  let total_spots := p.total_levels * p.spots_per_level
  let open_spots_first_three := p.open_spots_first + p.open_spots_second + p.open_spots_third
  let total_open_spots := total_spots - p.full_spots_total
  total_open_spots - open_spots_first_three = 31 := by
  sorry

end NUMINAMATH_CALUDE_fourth_level_open_spots_l2966_296684


namespace NUMINAMATH_CALUDE_routes_between_plains_cities_l2966_296627

theorem routes_between_plains_cities 
  (total_cities : Nat) 
  (mountainous_cities : Nat) 
  (plains_cities : Nat) 
  (total_routes : Nat) 
  (mountainous_routes : Nat) : 
  total_cities = 100 → 
  mountainous_cities = 30 → 
  plains_cities = 70 → 
  total_routes = 150 → 
  mountainous_routes = 21 → 
  ∃ (plains_routes : Nat), plains_routes = 81 ∧ 
    plains_routes + mountainous_routes + (total_routes - plains_routes - mountainous_routes) = total_routes := by
  sorry

end NUMINAMATH_CALUDE_routes_between_plains_cities_l2966_296627


namespace NUMINAMATH_CALUDE_log_823_bounds_sum_l2966_296659

theorem log_823_bounds_sum : ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 823 / Real.log 10 ∧ Real.log 823 / Real.log 10 < (d : ℝ) ∧ c + d = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_823_bounds_sum_l2966_296659


namespace NUMINAMATH_CALUDE_concatenated_numbers_divisibility_l2966_296680

def concatenate_numbers (n : ℕ) : ℕ :=
  sorry

theorem concatenated_numbers_divisibility (n : ℕ) :
  ¬(3 ∣ concatenate_numbers n) ↔ n % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_concatenated_numbers_divisibility_l2966_296680


namespace NUMINAMATH_CALUDE_one_equals_a_l2966_296677

theorem one_equals_a (x y z a : ℝ) 
  (sum_eq : x + y + z = a) 
  (inv_sum_eq : 1/x + 1/y + 1/z = 1/a) : 
  x = a ∨ y = a ∨ z = a := by
sorry

end NUMINAMATH_CALUDE_one_equals_a_l2966_296677


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2966_296671

/-- Three points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of collinearity for three points -/
def collinear (p q r : Point3D) : Prop :=
  ∃ t s : ℝ, q.x - p.x = t * (r.x - p.x) ∧
             q.y - p.y = t * (r.y - p.y) ∧
             q.z - p.z = t * (r.z - p.z) ∧
             q.x - p.x = s * (r.x - q.x) ∧
             q.y - p.y = s * (r.y - q.y) ∧
             q.z - p.z = s * (r.z - q.z)

theorem collinear_points_sum (m n : ℝ) :
  let M : Point3D := ⟨1, 0, 1⟩
  let N : Point3D := ⟨2, m, 3⟩
  let P : Point3D := ⟨2, 2, n + 1⟩
  collinear M N P → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2966_296671


namespace NUMINAMATH_CALUDE_children_after_addition_l2966_296628

-- Define the event parameters
def total_guests : Nat := 80
def num_men : Nat := 40
def num_women : Nat := num_men / 2
def added_children : Nat := 10

-- Theorem statement
theorem children_after_addition : 
  total_guests - (num_men + num_women) + added_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_children_after_addition_l2966_296628


namespace NUMINAMATH_CALUDE_ten_possible_values_for_d_l2966_296679

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct_digits (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Converts a five-digit number represented by individual digits to a natural number -/
def to_nat (a b c d e : Digit) : ℕ :=
  10000 * a.val + 1000 * b.val + 100 * c.val + 10 * d.val + e.val

/-- The main theorem stating that there are 10 possible values for D -/
theorem ten_possible_values_for_d :
  ∃ (possible_d_values : Finset Digit),
    possible_d_values.card = 10 ∧
    ∀ (a b c d : Digit),
      distinct_digits a b c d →
      (to_nat a b c b c) + (to_nat c b a d b) = (to_nat d b d d d) →
      d ∈ possible_d_values :=
sorry

end NUMINAMATH_CALUDE_ten_possible_values_for_d_l2966_296679


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2966_296614

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | |4*x - 3| < a ∧ a > 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 < 0}

-- Theorem statement
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x, x ∈ B → x ∈ A a) ∧ (∃ x, x ∈ A a ∧ x ∉ B) → 0 < a ∧ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2966_296614


namespace NUMINAMATH_CALUDE_sin_cos_identity_l2966_296641

theorem sin_cos_identity : Real.sin (15 * π / 180) * Real.sin (105 * π / 180) - 
  Real.cos (15 * π / 180) * Real.cos (105 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l2966_296641


namespace NUMINAMATH_CALUDE_amalie_remaining_coins_l2966_296676

/-- Given the ratio of Elsa's coins to Amalie's coins and their total coins,
    calculate how many coins Amalie remains with after spending 3/4 of her coins. -/
theorem amalie_remaining_coins
  (ratio_elsa : ℚ)
  (ratio_amalie : ℚ)
  (total_coins : ℕ)
  (h_ratio : ratio_elsa / ratio_amalie = 10 / 45)
  (h_total : ratio_elsa + ratio_amalie = 1)
  (h_coins : (ratio_amalie * total_coins : ℚ).num * (1 / (ratio_amalie * total_coins : ℚ).den : ℚ) * total_coins = 360) :
  (1 / 4 : ℚ) * ((ratio_amalie * total_coins : ℚ).num * (1 / (ratio_amalie * total_coins : ℚ).den : ℚ) * total_coins) = 90 :=
sorry

end NUMINAMATH_CALUDE_amalie_remaining_coins_l2966_296676


namespace NUMINAMATH_CALUDE_same_group_probability_correct_l2966_296620

def card_count : ℕ := 20
def people_count : ℕ := 4
def drawn_card1 : ℕ := 5
def drawn_card2 : ℕ := 14

def same_group_probability : ℚ := 7/51

theorem same_group_probability_correct :
  let remaining_cards := card_count - 2
  let smaller_group_cases := (card_count - drawn_card2) * (card_count - drawn_card2 - 1) / 2
  let larger_group_cases := (drawn_card1 - 1) * (drawn_card1 - 2) / 2
  let favorable_outcomes := smaller_group_cases + larger_group_cases
  let total_outcomes := remaining_cards * (remaining_cards - 1) / 2
  (favorable_outcomes : ℚ) / total_outcomes = same_group_probability := by
  sorry

end NUMINAMATH_CALUDE_same_group_probability_correct_l2966_296620


namespace NUMINAMATH_CALUDE_sequence_inequality_existence_l2966_296692

theorem sequence_inequality_existence (a b : ℕ → ℕ) :
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_existence_l2966_296692


namespace NUMINAMATH_CALUDE_modulus_of_z_l2966_296683

def i : ℂ := Complex.I

theorem modulus_of_z (z : ℂ) (h : z / (1 + i) = 1 - 2*i) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2966_296683


namespace NUMINAMATH_CALUDE_mrs_hilt_pizzas_l2966_296647

theorem mrs_hilt_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 16) :
  total_slices / slices_per_pizza = 2 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_pizzas_l2966_296647


namespace NUMINAMATH_CALUDE_average_speed_of_trip_l2966_296686

/-- Proves that the average speed of a 100-mile trip is 40 mph, given specific conditions -/
theorem average_speed_of_trip (total_distance : ℝ) (first_part_distance : ℝ) (second_part_distance : ℝ)
  (first_part_speed : ℝ) (second_part_speed : ℝ) (h1 : total_distance = 100)
  (h2 : first_part_distance = 30) (h3 : second_part_distance = 70)
  (h4 : first_part_speed = 60) (h5 : second_part_speed = 35)
  (h6 : total_distance = first_part_distance + second_part_distance) :
  (total_distance) / ((first_part_distance / first_part_speed) + (second_part_distance / second_part_speed)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_of_trip_l2966_296686


namespace NUMINAMATH_CALUDE_rob_has_three_dimes_l2966_296637

/-- Represents the number of coins of each type Rob has -/
structure RobsCoins where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value of Rob's coins in cents -/
def totalValue (coins : RobsCoins) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- Theorem stating that given Rob's coin counts and total value, he must have 3 dimes -/
theorem rob_has_three_dimes :
  ∀ (coins : RobsCoins),
    coins.quarters = 7 →
    coins.nickels = 5 →
    coins.pennies = 12 →
    totalValue coins = 242 →
    coins.dimes = 3 := by
  sorry


end NUMINAMATH_CALUDE_rob_has_three_dimes_l2966_296637


namespace NUMINAMATH_CALUDE_correct_quotient_calculation_l2966_296606

theorem correct_quotient_calculation (A B : ℕ) (dividend : ℕ) : 
  A > 0 → 
  A * 100 + B * 10 > 0 →
  dividend / (A * 10 + B) = 210 → 
  dividend / (A * 100 + B * 10) = 21 := by
sorry

end NUMINAMATH_CALUDE_correct_quotient_calculation_l2966_296606


namespace NUMINAMATH_CALUDE_measuring_cup_size_l2966_296633

/-- Given an 8 cup bag of flour, if removing 8 scoops leaves 6 cups,
    then the size of each scoop is 1/4 cup. -/
theorem measuring_cup_size (total_flour : ℚ) (scoops : ℕ) (remaining_flour : ℚ) 
    (scoop_size : ℚ) : 
    total_flour = 8 → 
    scoops = 8 → 
    remaining_flour = 6 → 
    total_flour - scoops * scoop_size = remaining_flour → 
    scoop_size = 1/4 := by
  sorry

#check measuring_cup_size

end NUMINAMATH_CALUDE_measuring_cup_size_l2966_296633


namespace NUMINAMATH_CALUDE_min_modulus_m_l2966_296691

/-- Given a complex number m such that the equation x^2 + mx + 1 + 2i = 0 has real roots,
    the minimum modulus of m is √(2 + 2√5). -/
theorem min_modulus_m (m : ℂ) : 
  (∃ x : ℝ, x^2 + m*x + 1 + 2*Complex.I = 0) → 
  Complex.abs m ≥ Real.sqrt (2 + 2 * Real.sqrt 5) ∧ 
  ∃ m₀ : ℂ, (∃ x : ℝ, x^2 + m₀*x + 1 + 2*Complex.I = 0) ∧ 
            Complex.abs m₀ = Real.sqrt (2 + 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_min_modulus_m_l2966_296691


namespace NUMINAMATH_CALUDE_ellipse_satisfies_equation_l2966_296635

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  -- Line passing through f2 perpendicular to x-axis
  line : Set (ℝ × ℝ)
  -- Intersection points of the line with the ellipse
  a : ℝ × ℝ
  b : ℝ × ℝ
  -- Distance between intersection points
  ab_distance : ℝ
  -- Properties
  f1_def : f1 = (-1, 0)
  f2_def : f2 = (1, 0)
  line_def : line = {p : ℝ × ℝ | p.1 = 1}
  ab_on_line : a ∈ line ∧ b ∈ line
  ab_distance_def : ab_distance = 3
  
/-- The equation of the ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Theorem stating that the given ellipse satisfies the equation -/
theorem ellipse_satisfies_equation (e : Ellipse) :
  ∀ x y, (x, y) ∈ {p : ℝ × ℝ | ellipse_equation e p.1 p.2} ↔ 
    (∃ t, 0 ≤ t ∧ t ≤ 1 ∧ 
      (x - e.f1.1)^2 + (y - e.f1.2)^2 + 
      (x - e.f2.1)^2 + (y - e.f2.2)^2 = 
      (2 * Real.sqrt ((x - e.f1.1)^2 + (y - e.f1.2)^2 + (x - e.f2.1)^2 + (y - e.f2.2)^2))^2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_satisfies_equation_l2966_296635


namespace NUMINAMATH_CALUDE_equation_solution_l2966_296673

theorem equation_solution : ∃! x : ℝ, (3 : ℝ) / (x - 3) = (4 : ℝ) / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2966_296673


namespace NUMINAMATH_CALUDE_food_lasts_five_more_days_l2966_296653

/-- Calculates the number of additional days food lasts after more men join -/
def additional_days_food_lasts (initial_men : ℕ) (initial_days : ℕ) (days_before_joining : ℕ) (additional_men : ℕ) : ℕ :=
  let total_food := initial_men * initial_days
  let remaining_food := total_food - (initial_men * days_before_joining)
  let total_men := initial_men + additional_men
  remaining_food / total_men

/-- Proves that given the initial conditions, the food lasts for 5 additional days -/
theorem food_lasts_five_more_days :
  additional_days_food_lasts 760 22 2 2280 = 5 := by
  sorry

#eval additional_days_food_lasts 760 22 2 2280

end NUMINAMATH_CALUDE_food_lasts_five_more_days_l2966_296653


namespace NUMINAMATH_CALUDE_fraction_problem_l2966_296657

theorem fraction_problem (x : ℚ) : (x * 48 + 15 = 27) → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2966_296657


namespace NUMINAMATH_CALUDE_unique_valid_number_l2966_296625

def is_valid_number (n : ℕ) : Prop :=
  (∀ k : ℕ, k ∈ Finset.range 10 → (n / 10^(9-k)) % k = 0) ∧
  (∀ d : ℕ, d ∈ Finset.range 10 → (∃! i : ℕ, i ∈ Finset.range 9 ∧ (n / 10^i) % 10 = d))

theorem unique_valid_number :
  ∃! n : ℕ, n = 381654729 ∧ is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2966_296625


namespace NUMINAMATH_CALUDE_five_students_two_teachers_arrangement_l2966_296656

/-- The number of ways two teachers can join a fixed line of students -/
def teacher_line_arrangements (num_students : ℕ) (num_teachers : ℕ) : ℕ :=
  (num_students + 1) * (num_students + 2)

/-- Theorem: With 5 students in fixed order and 2 teachers, there are 42 ways to arrange the line -/
theorem five_students_two_teachers_arrangement :
  teacher_line_arrangements 5 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_five_students_two_teachers_arrangement_l2966_296656


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l2966_296622

theorem triangle_angle_inequality (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) : 
  π * (1/A + 1/B + 1/C) ≥ (Real.sin A + Real.sin B + Real.sin C) * 
    (1/Real.sin A + 1/Real.sin B + 1/Real.sin C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l2966_296622


namespace NUMINAMATH_CALUDE_marble_distribution_solution_l2966_296664

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

end NUMINAMATH_CALUDE_marble_distribution_solution_l2966_296664


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l2966_296603

/-- A color type with exactly three colors -/
inductive Color
| Red
| Green
| Blue

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A coloring of the plane -/
def Coloring := Point → Color

/-- A rectangle in the plane -/
structure Rectangle where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Predicate to check if four points form a rectangle -/
def IsRectangle (r : Rectangle) : Prop := sorry

/-- Predicate to check if all vertices of a rectangle have the same color -/
def SameColorVertices (r : Rectangle) (c : Coloring) : Prop :=
  c r.p1 = c r.p2 ∧ c r.p1 = c r.p3 ∧ c r.p1 = c r.p4

/-- Theorem: In a plane colored with 3 colors, there exists a rectangle whose vertices are all the same color -/
theorem exists_same_color_rectangle (c : Coloring) :
  ∃ (r : Rectangle), IsRectangle r ∧ SameColorVertices r c := by sorry

end NUMINAMATH_CALUDE_exists_same_color_rectangle_l2966_296603


namespace NUMINAMATH_CALUDE_one_third_coloring_ways_l2966_296610

/-- The number of ways to choose k items from a set of n items -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The number of triangles in the square -/
def total_triangles : ℕ := 18

/-- The number of triangles to be colored -/
def colored_triangles : ℕ := 6

/-- Theorem stating that the number of ways to color one-third of the square is 18564 -/
theorem one_third_coloring_ways :
  binomial total_triangles colored_triangles = 18564 := by
  sorry

end NUMINAMATH_CALUDE_one_third_coloring_ways_l2966_296610


namespace NUMINAMATH_CALUDE_min_additional_matches_for_square_grid_l2966_296668

/-- Calculates the number of matches needed for a rectangular grid -/
def matches_for_grid (rows : ℕ) (cols : ℕ) : ℕ :=
  (rows + 1) * cols + (cols + 1) * rows

/-- Represents the problem of finding the minimum additional matches needed -/
theorem min_additional_matches_for_square_grid :
  let initial_matches := matches_for_grid 3 7
  let min_square_size := (initial_matches / 4 : ℕ).sqrt.succ
  let square_matches := matches_for_grid min_square_size min_square_size
  square_matches - initial_matches = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_matches_for_square_grid_l2966_296668


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2966_296601

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 5 - (1/2) * a 7 = (1/2) * a 7 - a 6 →
  (a 1 + a 2 + a 3) / (a 2 + a 3 + a 4) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2966_296601


namespace NUMINAMATH_CALUDE_point_on_y_axis_l2966_296634

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of a point being on the y-axis -/
def on_y_axis (p : CartesianPoint) : Prop := p.x = 0

/-- Theorem: A point with x-coordinate 0 lies on the y-axis -/
theorem point_on_y_axis (p : CartesianPoint) (h : p.x = 0) : on_y_axis p := by
  sorry

#check point_on_y_axis

end NUMINAMATH_CALUDE_point_on_y_axis_l2966_296634
