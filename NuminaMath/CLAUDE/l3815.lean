import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3815_381569

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 18 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 100 = 0 ∧ (n + 3) % 21 = 0

theorem smallest_number_divisible_by_all : 
  is_divisible_by_all 6297 ∧ ∀ m : ℕ, m < 6297 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3815_381569


namespace NUMINAMATH_CALUDE_parabola_c_value_l3815_381528

/-- A parabola with equation x = ay^2 + by + c, vertex at (-3, -1), and passing through (-1, 1) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := -3
  vertex_y : ℝ := -1
  point_x : ℝ := -1
  point_y : ℝ := 1
  eq_vertex : -3 = a * (-1)^2 + b * (-1) + c
  eq_point : -1 = a * 1^2 + b * 1 + c

/-- The value of c for the given parabola is -2.5 -/
theorem parabola_c_value (p : Parabola) : p.c = -2.5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3815_381528


namespace NUMINAMATH_CALUDE_factorial_ratio_l3815_381507

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3815_381507


namespace NUMINAMATH_CALUDE_fraction_value_l3815_381563

theorem fraction_value : (3020 - 2931)^2 / 121 = 64 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3815_381563


namespace NUMINAMATH_CALUDE_triangle_area_l3815_381576

/-- Given a triangle ABC with sides AC = 8 and BC = 10, and the condition that 32 cos(A - B) = 31,
    prove that the area of the triangle is 15√7. -/
theorem triangle_area (A B C : ℝ) (AC BC : ℝ) (h1 : AC = 8) (h2 : BC = 10) 
    (h3 : 32 * Real.cos (A - B) = 31) : 
    (1/2 : ℝ) * AC * BC * Real.sin (A + B - π) = 15 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3815_381576


namespace NUMINAMATH_CALUDE_simplify_expression_l3815_381543

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 9) - (x + 6)*(3*x + 2) = x - 66 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3815_381543


namespace NUMINAMATH_CALUDE_root_difference_quadratic_equation_l3815_381597

theorem root_difference_quadratic_equation : ∃ (x y : ℝ), 
  (x^2 + 40*x + 300 = -48) ∧ 
  (y^2 + 40*y + 300 = -48) ∧ 
  x ≠ y ∧ 
  |x - y| = 16 := by
sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_equation_l3815_381597


namespace NUMINAMATH_CALUDE_factors_of_8_cube_5_fifth_7_square_l3815_381544

def number_of_factors (n : ℕ) : ℕ := sorry

theorem factors_of_8_cube_5_fifth_7_square :
  number_of_factors (8^3 * 5^5 * 7^2) = 180 := by sorry

end NUMINAMATH_CALUDE_factors_of_8_cube_5_fifth_7_square_l3815_381544


namespace NUMINAMATH_CALUDE_hyperbola_center_l3815_381582

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  center = (7, 2) ↔ f1 = (3, -2) ∧ f2 = (11, 6) := by
  sorry

#check hyperbola_center

end NUMINAMATH_CALUDE_hyperbola_center_l3815_381582


namespace NUMINAMATH_CALUDE_vertex_determines_parameters_l3815_381525

def quadratic_function (h k : ℝ) (x : ℝ) : ℝ := -3 * (x - h)^2 + k

theorem vertex_determines_parameters (h k : ℝ) :
  (∀ x, quadratic_function h k x = quadratic_function 1 (-2) x) →
  h = 1 ∧ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_determines_parameters_l3815_381525


namespace NUMINAMATH_CALUDE_min_value_sqrt_inverse_equality_condition_l3815_381510

theorem min_value_sqrt_inverse (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x^2 ≥ 4 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 4 / x^2 = 4 * Real.sqrt 2 ↔ x = 2^(4/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_inverse_equality_condition_l3815_381510


namespace NUMINAMATH_CALUDE_problem_solution_l3815_381589

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.sin y = 2008)
  (h2 : x + 2008 * Real.cos y = 2007)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3815_381589


namespace NUMINAMATH_CALUDE_expand_binomials_l3815_381549

theorem expand_binomials (x : ℝ) : (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l3815_381549


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3815_381585

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (x - 1 : ℚ) / 2 ≥ (x - 2 : ℚ) / 3 ∧ (2 * x - 5 : ℤ) < -3 * x}
  S = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l3815_381585


namespace NUMINAMATH_CALUDE_cubic_factorization_l3815_381524

theorem cubic_factorization (t : ℝ) : t^3 - 144 = (t - 12) * (t^2 + 12*t + 144) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3815_381524


namespace NUMINAMATH_CALUDE_production_rates_l3815_381558

/-- The production rates of two workers --/
theorem production_rates (total_rate : ℝ) (a_parts b_parts : ℕ) 
  (h1 : total_rate = 35)
  (h2 : (a_parts : ℝ) / x = (b_parts : ℝ) / (total_rate - x))
  (h3 : a_parts = 90)
  (h4 : b_parts = 120) :
  ∃ (x y : ℝ), x + y = total_rate ∧ x = 15 ∧ y = 20 :=
sorry

end NUMINAMATH_CALUDE_production_rates_l3815_381558


namespace NUMINAMATH_CALUDE_max_min_difference_z_l3815_381565

theorem max_min_difference_z (x y z : ℝ) 
  (sum_condition : x + y + z = 3)
  (sum_squares_condition : x^2 + y^2 + z^2 = 18) :
  ∃ (z_max z_min : ℝ),
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≤ z_max) ∧
    (∀ z' : ℝ, (∃ x' y' : ℝ, x' + y' + z' = 3 ∧ x'^2 + y'^2 + z'^2 = 18) → z' ≥ z_min) ∧
    z_max - z_min = 6.5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l3815_381565


namespace NUMINAMATH_CALUDE_smallest_sum_B_d_l3815_381505

theorem smallest_sum_B_d : 
  ∃ (B d : ℕ), 
    B < 5 ∧ 
    d > 6 ∧ 
    125 * B + 25 * B + B = 4 * d + 4 ∧
    (∀ (B' d' : ℕ), 
      B' < 5 → 
      d' > 6 → 
      125 * B' + 25 * B' + B' = 4 * d' + 4 → 
      B + d ≤ B' + d') ∧
    B + d = 77 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_B_d_l3815_381505


namespace NUMINAMATH_CALUDE_existence_of_integers_l3815_381533

theorem existence_of_integers (a₁ a₂ a₃ : ℕ) (h₁ : 0 < a₁) (h₂ : a₁ < a₂) (h₃ : a₂ < a₃) :
  ∃ x₁ x₂ x₃ : ℤ,
    (abs x₁ + abs x₂ + abs x₃ > 0) ∧
    (a₁ * x₁ + a₂ * x₂ + a₃ * x₃ = 0) ∧
    (max (abs x₁) (max (abs x₂) (abs x₃)) < (2 / Real.sqrt 3) * Real.sqrt a₃ + 1) :=
sorry

end NUMINAMATH_CALUDE_existence_of_integers_l3815_381533


namespace NUMINAMATH_CALUDE_problem_statement_l3815_381592

theorem problem_statement : 2 * ((40 / 8) + (34 / 12)) = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3815_381592


namespace NUMINAMATH_CALUDE_average_of_rst_l3815_381545

theorem average_of_rst (r s t : ℝ) (h : (4 / 3) * (r + s + t) = 12) : 
  (r + s + t) / 3 = 3 := by sorry

end NUMINAMATH_CALUDE_average_of_rst_l3815_381545


namespace NUMINAMATH_CALUDE_emmys_journey_length_l3815_381574

-- Define the journey structure
structure Journey where
  total_length : ℚ
  first_part : ℚ
  second_part : ℚ
  third_part : ℚ

-- Define Emmy's journey
def emmys_journey : Journey where
  total_length := 360 / 7
  first_part := (360 / 7) / 4
  second_part := 30
  third_part := (360 / 7) / 6

-- Theorem statement
theorem emmys_journey_length :
  ∀ (j : Journey),
    j.first_part = j.total_length / 4 →
    j.third_part = j.total_length / 6 →
    j.second_part = 30 →
    j.total_length = 360 / 7 := by
  sorry

#check emmys_journey_length

end NUMINAMATH_CALUDE_emmys_journey_length_l3815_381574


namespace NUMINAMATH_CALUDE_stratified_sampling_third_group_size_l3815_381579

/-- Proves that in a stratified sampling scenario, given specific conditions, 
    the size of the third group is 1040. -/
theorem stratified_sampling_third_group_size 
  (total_sample : ℕ) 
  (grade11_sample : ℕ) 
  (grade10_pop : ℕ) 
  (grade11_pop : ℕ) 
  (h1 : total_sample = 81)
  (h2 : grade11_sample = 30)
  (h3 : grade10_pop = 1000)
  (h4 : grade11_pop = 1200) :
  ∃ n : ℕ, 
    (grade11_sample : ℚ) / total_sample = 
    grade11_pop / (grade10_pop + grade11_pop + n) ∧ 
    n = 1040 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_group_size_l3815_381579


namespace NUMINAMATH_CALUDE_stone_density_l3815_381577

/-- Given a cylindrical container with water and a stone, this theorem relates
    the density of the stone to the water level changes under different conditions. -/
theorem stone_density (S : ℝ) (ρ h₁ h₂ : ℝ) (hS : S > 0) (hρ : ρ > 0) (hh₁ : h₁ > 0) (hh₂ : h₂ > 0) :
  let ρ_s := (ρ * h₁) / h₂
  ρ_s = (ρ * S * h₁) / (S * h₂) :=
by sorry

end NUMINAMATH_CALUDE_stone_density_l3815_381577


namespace NUMINAMATH_CALUDE_p_range_q_range_p_or_q_false_range_l3815_381581

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (1 - 2*m) + y^2 / (m + 3) = 1 ∧ (1 - 2*m) * (m + 3) < 0

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 3 - 2*m = 0

-- Theorem for the range of m where p is true
theorem p_range (m : ℝ) : p m ↔ m < -3 ∨ m > 1/2 := by sorry

-- Theorem for the range of m where q is true
theorem q_range (m : ℝ) : q m ↔ m ≤ -3 ∨ m ≥ 1 := by sorry

-- Theorem for the range of m where "p ∨ q" is false
theorem p_or_q_false_range (m : ℝ) : ¬(p m ∨ q m) ↔ -3 < m ∧ m ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_p_range_q_range_p_or_q_false_range_l3815_381581


namespace NUMINAMATH_CALUDE_black_balls_probability_l3815_381598

theorem black_balls_probability 
  (m₁ m₂ k₁ k₂ : ℕ) 
  (h_total : m₁ + m₂ = 25)
  (h_white_prob : (k₁ : ℝ) / m₁ * (k₂ : ℝ) / m₂ = 0.54)
  : ((m₁ - k₁ : ℝ) / m₁) * ((m₂ - k₂ : ℝ) / m₂) = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_black_balls_probability_l3815_381598


namespace NUMINAMATH_CALUDE_rachel_picked_apples_l3815_381553

/-- Represents the number of apples Rachel picked from her tree -/
def apples_picked : ℕ := 2

/-- The initial number of apples on Rachel's tree -/
def initial_apples : ℕ := 4

/-- The number of new apples that grew on the tree -/
def new_apples : ℕ := 3

/-- The final number of apples on the tree -/
def final_apples : ℕ := 5

/-- Theorem stating that the number of apples Rachel picked is correct -/
theorem rachel_picked_apples :
  initial_apples - apples_picked + new_apples = final_apples :=
by sorry

end NUMINAMATH_CALUDE_rachel_picked_apples_l3815_381553


namespace NUMINAMATH_CALUDE_earl_floor_problem_l3815_381562

theorem earl_floor_problem (total_floors : ℕ) (start_floor : ℕ) 
  (up_first : ℕ) (down : ℕ) (up_second : ℕ) :
  total_floors = 20 →
  start_floor = 1 →
  up_first = 5 →
  down = 2 →
  up_second = 7 →
  total_floors - (start_floor + up_first - down + up_second) = 9 :=
by sorry

end NUMINAMATH_CALUDE_earl_floor_problem_l3815_381562


namespace NUMINAMATH_CALUDE_problem_statement_l3815_381532

theorem problem_statement (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) :
  (40/100) * N = 120 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3815_381532


namespace NUMINAMATH_CALUDE_division_equality_l3815_381593

theorem division_equality (h : 29.94 / 1.45 = 17.3) : 2994 / 14.5 = 173 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l3815_381593


namespace NUMINAMATH_CALUDE_paint_fraction_first_week_l3815_381523

/-- Proves that the fraction of paint used in the first week is 1/9 -/
theorem paint_fraction_first_week (total_paint : ℝ) (paint_used : ℝ) 
  (h1 : total_paint = 360)
  (h2 : paint_used = 104)
  (h3 : ∀ f : ℝ, paint_used = f * total_paint + 1/5 * (total_paint - f * total_paint)) :
  ∃ f : ℝ, f = 1/9 ∧ paint_used = f * total_paint + 1/5 * (total_paint - f * total_paint) :=
by sorry

end NUMINAMATH_CALUDE_paint_fraction_first_week_l3815_381523


namespace NUMINAMATH_CALUDE_pipe_fill_time_l3815_381571

/-- Represents the time (in hours) it takes for a pipe to fill a tank without a leak -/
def fill_time : ℝ := 6

/-- Represents the time (in hours) it takes for the pipe to fill the tank with the leak -/
def fill_time_with_leak : ℝ := 8

/-- Represents the time (in hours) it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 24

/-- Proves that the time it takes for the pipe to fill the tank without the leak is 6 hours -/
theorem pipe_fill_time : 
  (1 / fill_time - 1 / leak_empty_time) * fill_time_with_leak = 1 :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l3815_381571


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3815_381536

theorem sum_of_numbers (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.5)
  (ga : a > 0.1) (gb : b > 0.1) (gc : c > 0.1) : a + b + c = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3815_381536


namespace NUMINAMATH_CALUDE_book_pages_book_pages_proof_l3815_381517

theorem book_pages : ℝ → Prop :=
  fun x => 
    let day1_read := x / 4 + 17
    let day1_remain := x - day1_read
    let day2_read := day1_remain / 3 + 20
    let day2_remain := day1_remain - day2_read
    let day3_read := day2_remain / 2 + 23
    let day3_remain := day2_remain - day3_read
    day3_remain = 70 → x = 394

-- The proof goes here
theorem book_pages_proof : ∃ x : ℝ, book_pages x := by
  sorry

end NUMINAMATH_CALUDE_book_pages_book_pages_proof_l3815_381517


namespace NUMINAMATH_CALUDE_function_equation_1_bijective_function_equation_2_neither_function_equation_3_neither_function_equation_4_neither_l3815_381568

-- 1. f(x+f(y))=2f(x)+y is bijective
theorem function_equation_1_bijective (f : ℝ → ℝ) :
  (∀ x y, f (x + f y) = 2 * f x + y) → Function.Bijective f :=
sorry

-- 2. f(f(x))=0 is neither injective nor surjective
theorem function_equation_2_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = 0) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 3. f(f(x))=sin(x) is neither injective nor surjective
theorem function_equation_3_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = Real.sin x) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 4. f(x+y)=f(x)f(y) is neither injective nor surjective
theorem function_equation_4_neither (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x * f y) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

end NUMINAMATH_CALUDE_function_equation_1_bijective_function_equation_2_neither_function_equation_3_neither_function_equation_4_neither_l3815_381568


namespace NUMINAMATH_CALUDE_sum_of_park_areas_l3815_381502

theorem sum_of_park_areas :
  let park1_side : ℝ := 11
  let park2_side : ℝ := 5
  let park1_area := park1_side * park1_side
  let park2_area := park2_side * park2_side
  park1_area + park2_area = 146 := by
sorry

end NUMINAMATH_CALUDE_sum_of_park_areas_l3815_381502


namespace NUMINAMATH_CALUDE_closest_point_on_line_l3815_381548

/-- The line equation y = 2x - 4 -/
def line_equation (x : ℝ) : ℝ := 2 * x - 4

/-- The point we're finding the closest point to -/
def given_point : ℝ × ℝ := (3, 1)

/-- The claimed closest point on the line -/
def closest_point : ℝ × ℝ := (2.6, 1.2)

/-- Theorem stating that the closest_point is on the line and is the closest to given_point -/
theorem closest_point_on_line :
  (line_equation closest_point.1 = closest_point.2) ∧
  ∀ (p : ℝ × ℝ), (line_equation p.1 = p.2) →
    (closest_point.1 - given_point.1)^2 + (closest_point.2 - given_point.2)^2 ≤
    (p.1 - given_point.1)^2 + (p.2 - given_point.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l3815_381548


namespace NUMINAMATH_CALUDE_x_value_when_derivative_is_three_l3815_381515

def f (x : ℝ) := x^3

theorem x_value_when_derivative_is_three (x : ℝ) (h1 : x > 0) (h2 : (deriv f) x = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_derivative_is_three_l3815_381515


namespace NUMINAMATH_CALUDE_inequality_implies_x_greater_than_one_l3815_381588

theorem inequality_implies_x_greater_than_one (x : ℝ) :
  x * (x^2 + 1) > (x + 1) * (x^2 - x + 1) → x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_x_greater_than_one_l3815_381588


namespace NUMINAMATH_CALUDE_rectangle_area_l3815_381596

/-- The area of a rectangle with length 15 cm and width 0.9 times its length is 202.5 cm². -/
theorem rectangle_area : 
  let length : ℝ := 15
  let width : ℝ := 0.9 * length
  length * width = 202.5 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3815_381596


namespace NUMINAMATH_CALUDE_grasshopper_jumps_l3815_381586

/-- Given a grasshopper's initial position and first jump endpoint, calculate its final position after a second identical jump -/
theorem grasshopper_jumps (initial_pos : ℝ) (first_jump_end : ℝ) : 
  initial_pos = 8 → first_jump_end = 17.5 → 
  let jump_length := first_jump_end - initial_pos
  first_jump_end + jump_length = 27 := by
sorry

end NUMINAMATH_CALUDE_grasshopper_jumps_l3815_381586


namespace NUMINAMATH_CALUDE_square_root_problem_l3815_381527

-- Define the variables
variable (a b : ℝ)

-- State the theorem
theorem square_root_problem (h1 : a = 9) (h2 : b = 4/9) :
  (∃ (x : ℝ), x^2 = a ∧ (x = 3 ∨ x = -3)) ∧
  (Real.sqrt (a * b) = 2) →
  (a = 9 ∧ b = 4/9) ∧
  (∃ (y : ℝ), y^2 = a + 2*b ∧ (y = Real.sqrt 89 / 3 ∨ y = -Real.sqrt 89 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_square_root_problem_l3815_381527


namespace NUMINAMATH_CALUDE_house_selling_price_l3815_381580

theorem house_selling_price 
  (original_price : ℝ)
  (profit_percentage : ℝ)
  (commission_percentage : ℝ)
  (h1 : original_price = 80000)
  (h2 : profit_percentage = 20)
  (h3 : commission_percentage = 5)
  : original_price + (profit_percentage / 100) * original_price + (commission_percentage / 100) * original_price = 100000 := by
  sorry

end NUMINAMATH_CALUDE_house_selling_price_l3815_381580


namespace NUMINAMATH_CALUDE_selling_price_l3815_381594

/-- Represents the labelled price of a refrigerator -/
def R : ℝ := sorry

/-- Represents the labelled price of a washing machine -/
def W : ℝ := sorry

/-- The condition that the total discounted price is 35000 -/
axiom purchase_price : 0.80 * R + 0.85 * W = 35000

/-- The theorem stating the selling price formula -/
theorem selling_price : 
  0.80 * R + 0.85 * W = 35000 → 
  (1.10 * R + 1.12 * W) = (1.10 * R + 1.12 * W) :=
by sorry

end NUMINAMATH_CALUDE_selling_price_l3815_381594


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3815_381557

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 2 + a 5 = 18
  product_property : a 3 * a 4 = 32

/-- The theorem stating that for the given arithmetic sequence, a_n = 128 implies n = 8 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  ∃ n : ℕ, seq.a n = 128 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3815_381557


namespace NUMINAMATH_CALUDE_unknown_number_problem_l3815_381561

theorem unknown_number_problem (x : ℝ) : (x + 12) / 8 = 9 → 35 - (x / 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_problem_l3815_381561


namespace NUMINAMATH_CALUDE_paper_I_max_mark_l3815_381555

/-- The maximum mark for Paper I -/
def max_mark : ℕ := 262

/-- The passing percentage for Paper I -/
def passing_percentage : ℚ := 65 / 100

/-- The marks scored by the candidate -/
def scored_marks : ℕ := 112

/-- The marks by which the candidate failed -/
def failed_by : ℕ := 58

/-- Theorem stating that the maximum mark for Paper I is 262 -/
theorem paper_I_max_mark :
  (↑max_mark * passing_percentage).floor = scored_marks + failed_by :=
sorry

end NUMINAMATH_CALUDE_paper_I_max_mark_l3815_381555


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l3815_381599

/-- Triangle with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 7)
  C : ℝ × ℝ := (0, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The altitude from A to BC -/
def altitude (t : Triangle) : LineEquation :=
  { a := 2, b := 7, c := -21 }

/-- The median from BC -/
def median (t : Triangle) : LineEquation :=
  { a := 5, b := 1, c := -20 }

theorem triangle_altitude_and_median (t : Triangle) :
  (altitude t = { a := 2, b := 7, c := -21 }) ∧
  (median t = { a := 5, b := 1, c := -20 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l3815_381599


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3815_381550

/-- Given a hyperbola C passing through the point (1,1) with asymptotes 2x+y=0 and 2x-y=0,
    its standard equation is 4x²/3 - y²/3 = 1. -/
theorem hyperbola_standard_equation (C : Set (ℝ × ℝ)) :
  (∀ x y, (x, y) ∈ C ↔ 4 * x^2 / 3 - y^2 / 3 = 1) ↔
  ((1, 1) ∈ C ∧
   (∀ x y, 2*x + y = 0 → (x, y) ∈ frontier C) ∧
   (∀ x y, 2*x - y = 0 → (x, y) ∈ frontier C)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3815_381550


namespace NUMINAMATH_CALUDE_billboard_problem_l3815_381504

/-- The number of billboards to be erected -/
def num_billboards : ℕ := 200

/-- The length of the road in meters -/
def road_length : ℕ := 1100

/-- The spacing between billboards in the first scenario (in meters) -/
def spacing1 : ℚ := 5

/-- The spacing between billboards in the second scenario (in meters) -/
def spacing2 : ℚ := 11/2

/-- The number of missing billboards in the first scenario -/
def missing1 : ℕ := 21

/-- The number of missing billboards in the second scenario -/
def missing2 : ℕ := 1

theorem billboard_problem :
  (spacing1 * (num_billboards + missing1 - 1 : ℚ) = road_length) ∧
  (spacing2 * (num_billboards + missing2 - 1 : ℚ) = road_length) := by
  sorry

end NUMINAMATH_CALUDE_billboard_problem_l3815_381504


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3815_381518

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1/4) :
  ∃ q : ℝ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3815_381518


namespace NUMINAMATH_CALUDE_periodic_function_theorem_l3815_381552

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem periodic_function_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f :=
sorry

end NUMINAMATH_CALUDE_periodic_function_theorem_l3815_381552


namespace NUMINAMATH_CALUDE_apples_remaining_l3815_381595

def initial_apples : ℕ := 128
def sale_percentage : ℚ := 25 / 100

theorem apples_remaining (initial : ℕ) (sale_percent : ℚ) : 
  initial - ⌊initial * sale_percent⌋ - ⌊(initial - ⌊initial * sale_percent⌋) * sale_percent⌋ - 1 = 71 :=
by sorry

end NUMINAMATH_CALUDE_apples_remaining_l3815_381595


namespace NUMINAMATH_CALUDE_complex_magnitude_theorem_l3815_381519

theorem complex_magnitude_theorem (z₁ z₂ z₃ : ℂ) (a b c : ℝ) 
  (h₁ : (z₁ / z₂ + z₂ / z₃ + z₃ / z₁).im = 0)
  (h₂ : Complex.abs z₁ = 1)
  (h₃ : Complex.abs z₂ = 1)
  (h₄ : Complex.abs z₃ = 1) :
  ∃ (x : ℝ), x = Complex.abs (a * z₁ + b * z₂ + c * z₃) ∧
    (x = Real.sqrt ((a + b)^2 + c^2) ∨
     x = Real.sqrt ((a + c)^2 + b^2) ∨
     x = Real.sqrt ((b + c)^2 + a^2)) :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_theorem_l3815_381519


namespace NUMINAMATH_CALUDE_octal_subtraction_l3815_381572

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Subtraction operation in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Conversion from decimal to octal --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Conversion from octal to decimal --/
def from_octal (n : OctalNumber) : ℕ :=
  sorry

theorem octal_subtraction :
  octal_sub (to_octal 43) (to_octal 22) = to_octal 21 :=
by sorry

end NUMINAMATH_CALUDE_octal_subtraction_l3815_381572


namespace NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l3815_381541

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls needed to guarantee at least n balls of a single color -/
def minBallsForGuarantee (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in our problem -/
def problemCounts : BallCounts :=
  { red := 35, green := 30, yellow := 25, blue := 15, white := 12, black := 10 }

theorem min_balls_for_twenty_of_one_color :
  minBallsForGuarantee problemCounts 20 = 95 := by sorry

end NUMINAMATH_CALUDE_min_balls_for_twenty_of_one_color_l3815_381541


namespace NUMINAMATH_CALUDE_range_of_f_l3815_381556

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 18) / Real.log (1/3)

theorem range_of_f :
  Set.range f = Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3815_381556


namespace NUMINAMATH_CALUDE_divisibility_by_three_l3815_381591

theorem divisibility_by_three (x y : ℤ) : 
  (3 ∣ x^2 + y^2) → (3 ∣ x) ∧ (3 ∣ y) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l3815_381591


namespace NUMINAMATH_CALUDE_passenger_ticket_probability_l3815_381590

/-- The probability of a passenger getting a ticket at three counters -/
theorem passenger_ticket_probability
  (p₁ p₂ p₃ p₄ p₅ p₆ : ℝ)
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1)
  (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1)
  (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1)
  (h₄ : 0 ≤ p₄ ∧ p₄ ≤ 1)
  (h₅ : 0 ≤ p₅ ∧ p₅ ≤ 1)
  (h₆ : 0 ≤ p₆ ∧ p₆ ≤ 1)
  (h_sum : p₁ + p₂ + p₃ = 1) :
  let prob_get_ticket := p₁ * (1 - p₄) + p₂ * (1 - p₅) + p₃ * (1 - p₆)
  0 ≤ prob_get_ticket ∧ prob_get_ticket ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_passenger_ticket_probability_l3815_381590


namespace NUMINAMATH_CALUDE_max_visible_sum_l3815_381538

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers on each cube --/
def cube_numbers : Finset ℕ := {1, 3, 6, 12, 24, 48}

/-- A stack of three cubes --/
structure CubeStack :=
  (bottom : Cube)
  (middle : Cube)
  (top : Cube)

/-- The sum of visible numbers in a cube stack --/
def visible_sum (stack : CubeStack) : ℕ := sorry

/-- Theorem stating the maximum sum of visible numbers --/
theorem max_visible_sum :
  ∃ (stack : CubeStack),
    (∀ (c : Cube) (i : Fin 6), c.faces i ∈ cube_numbers) →
    (∀ (stack' : CubeStack), visible_sum stack' ≤ visible_sum stack) →
    visible_sum stack = 267 :=
sorry

end NUMINAMATH_CALUDE_max_visible_sum_l3815_381538


namespace NUMINAMATH_CALUDE_skew_lines_projection_not_two_points_l3815_381531

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D

-- Define a type for points in 2D space (the projection plane)
structure Point2D where
  -- Add necessary fields to represent a point in 2D

-- Define a projection function from 3D to 2D
def project (l : Line3D) : Point2D :=
  sorry

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem skew_lines_projection_not_two_points 
  (l1 l2 : Line3D) (h : are_skew l1 l2) : 
  ¬(∃ (p1 p2 : Point2D), project l1 = p1 ∧ project l2 = p2 ∧ p1 ≠ p2) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_projection_not_two_points_l3815_381531


namespace NUMINAMATH_CALUDE_square_difference_equality_l3815_381584

theorem square_difference_equality : 1005^2 - 995^2 - 1002^2 + 996^2 = 8012 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3815_381584


namespace NUMINAMATH_CALUDE_smaller_number_theorem_l3815_381530

theorem smaller_number_theorem (x y : ℝ) : 
  x + y = 15 → x * y = 36 → min x y = 3 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_theorem_l3815_381530


namespace NUMINAMATH_CALUDE_quadratic_root_difference_sum_l3815_381587

def quadratic_equation (x : ℝ) : Prop := 5 * x^2 - 13 * x - 6 = 0

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p^2 ∣ n) → p = 1

theorem quadratic_root_difference_sum (p q : ℕ) (hp : is_square_free p) :
  (∃ x₁ x₂ : ℝ, quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ 
    |x₁ - x₂| = (Real.sqrt (p : ℝ)) / (q : ℝ)) →
  p + q = 294 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_sum_l3815_381587


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l3815_381511

theorem polynomial_value_theorem (x : ℂ) (h : x^2 + x + 1 = 0) :
  x^2000 + x^1999 + x^1998 + 1000*x^1000 + 1000*x^999 + 1000*x^998 + 2000*x^3 + 2000*x^2 + 2000*x + 3000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l3815_381511


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l3815_381534

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 6) 
  (h2 : sum_first_two = 8/3) :
  ∃ (a : ℝ), (a = 6 + 2 * Real.sqrt 5 ∨ a = 6 - 2 * Real.sqrt 5) ∧ 
  (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a * (1 + r)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l3815_381534


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l3815_381501

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2014 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l3815_381501


namespace NUMINAMATH_CALUDE_converse_proposition_l3815_381509

theorem converse_proposition :
  (∀ x y : ℝ, (x ≤ 2 ∨ y ≤ 2) → x + y ≤ 4) ↔
  (¬∀ x y : ℝ, (x > 2 ∧ y > 2) → x + y > 4) :=
by sorry

end NUMINAMATH_CALUDE_converse_proposition_l3815_381509


namespace NUMINAMATH_CALUDE_stamp_arrangement_count_l3815_381567

/-- Represents the number of stamps of each denomination --/
def stamp_counts : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- Represents the value of each stamp denomination --/
def stamp_values : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

/-- A function to calculate the number of unique arrangements --/
def count_arrangements (counts : List Nat) (values : List Nat) (target : Nat) : Nat :=
  sorry

theorem stamp_arrangement_count :
  count_arrangements stamp_counts stamp_values 20 = 76 :=
by sorry

end NUMINAMATH_CALUDE_stamp_arrangement_count_l3815_381567


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3815_381551

theorem imaginary_part_of_z (z : ℂ) (h : (1 : ℂ) / z = 1 / (1 + 2*I) + 1 / (1 - I)) : 
  z.im = -(1 / 5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3815_381551


namespace NUMINAMATH_CALUDE_find_a_l3815_381520

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 < a^2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the intersection of A and B
def A_intersect_B (a : ℝ) : Set ℝ := A a ∩ B

-- State the theorem
theorem find_a : ∀ a : ℝ, A_intersect_B a = {x : ℝ | 1 < x ∧ x < 2} → a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l3815_381520


namespace NUMINAMATH_CALUDE_part_one_part_two_l3815_381535

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x - 3) / (2 - x) ≥ 0

-- Part 1
theorem part_one (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) 
  (h_not_necessary : ∃ x, q x ∧ p x a) : 1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3815_381535


namespace NUMINAMATH_CALUDE_greatest_difference_of_valid_units_digits_l3815_381521

/-- A function that checks if a number is divisible by 4 -/
def isDivisibleBy4 (n : ℕ) : Prop := n % 4 = 0

/-- The set of all possible three-digit numbers starting with 47 -/
def threeDigitNumbers : Set ℕ := {n : ℕ | 470 ≤ n ∧ n ≤ 479}

/-- The set of all three-digit numbers starting with 47 that are divisible by 4 -/
def divisibleNumbers : Set ℕ := {n ∈ threeDigitNumbers | isDivisibleBy4 n}

/-- The set of units digits of numbers in divisibleNumbers -/
def validUnitsDigits : Set ℕ := {x : ℕ | ∃ n ∈ divisibleNumbers, n % 10 = x}

theorem greatest_difference_of_valid_units_digits :
  ∃ (a b : ℕ), a ∈ validUnitsDigits ∧ b ∈ validUnitsDigits ∧ 
  ∀ (x y : ℕ), x ∈ validUnitsDigits → y ∈ validUnitsDigits → 
  (max a b - min a b : ℤ) ≥ (max x y - min x y) ∧
  (max a b - min a b : ℤ) = 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_of_valid_units_digits_l3815_381521


namespace NUMINAMATH_CALUDE_smallest_k_for_p_cubed_minus_k_div_24_l3815_381583

-- Define p as the largest prime number with 1007 digits
def p : Nat := sorry

-- Define the property that p is prime
axiom p_is_prime : Nat.Prime p

-- Define the property that p has 1007 digits
axiom p_has_1007_digits : 10^1006 ≤ p ∧ p < 10^1007

-- Define the property that p is the largest such prime
axiom p_is_largest : ∀ q : Nat, Nat.Prime q → 10^1006 ≤ q ∧ q < 10^1007 → q ≤ p

-- Theorem statement
theorem smallest_k_for_p_cubed_minus_k_div_24 :
  (∃ k : Nat, k > 0 ∧ (p^3 - k) % 24 = 0) ∧
  (∀ k : Nat, k > 0 ∧ (p^3 - k) % 24 = 0 → k ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_p_cubed_minus_k_div_24_l3815_381583


namespace NUMINAMATH_CALUDE_cos_double_angle_with_tan_l3815_381546

theorem cos_double_angle_with_tan (θ : ℝ) (h : Real.tan θ = 3) : Real.cos (2 * θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_with_tan_l3815_381546


namespace NUMINAMATH_CALUDE_square_floor_tiles_l3815_381554

/-- Represents a square floor tiled with congruent square tiles. -/
structure SquareFloor where
  side_length : ℕ
  is_valid : side_length > 0

/-- Calculates the number of black tiles on the boundary of the floor. -/
def black_tiles (floor : SquareFloor) : ℕ :=
  4 * floor.side_length - 4

/-- Calculates the total number of tiles on the floor. -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length ^ 2

/-- Theorem stating that a square floor with 100 black boundary tiles has 676 total tiles. -/
theorem square_floor_tiles (floor : SquareFloor) :
  black_tiles floor = 100 → total_tiles floor = 676 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l3815_381554


namespace NUMINAMATH_CALUDE_average_timing_error_l3815_381542

def total_watches : ℕ := 10

def timing_errors : List ℕ := [0, 1, 2, 3]
def error_frequencies : List ℕ := [3, 4, 2, 1]

def average_error : ℚ := 1.1

theorem average_timing_error :
  (List.sum (List.zipWith (· * ·) timing_errors error_frequencies) : ℚ) / total_watches = average_error :=
sorry

end NUMINAMATH_CALUDE_average_timing_error_l3815_381542


namespace NUMINAMATH_CALUDE_divisibility_by_1897_l3815_381526

theorem divisibility_by_1897 (n : ℕ) : 
  (1897 : ℤ) ∣ (2903^n - 803^n - 464^n + 261^n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1897_l3815_381526


namespace NUMINAMATH_CALUDE_point_A_in_third_quadrant_l3815_381516

/-- The point A with coordinates (sin 2018°, tan 117°) is in the third quadrant -/
theorem point_A_in_third_quadrant :
  let x : ℝ := Real.sin (2018 * π / 180)
  let y : ℝ := Real.tan (117 * π / 180)
  x < 0 ∧ y < 0 := by sorry

end NUMINAMATH_CALUDE_point_A_in_third_quadrant_l3815_381516


namespace NUMINAMATH_CALUDE_library_books_loaned_l3815_381540

theorem library_books_loaned (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) : 
  initial_books = 75 → 
  return_rate = 7/10 → 
  final_books = 63 → 
  ∃ (loaned_books : ℕ), loaned_books = 40 ∧ 
    initial_books - final_books = (1 - return_rate) * loaned_books := by
  sorry

end NUMINAMATH_CALUDE_library_books_loaned_l3815_381540


namespace NUMINAMATH_CALUDE_range_of_x_when_m_is_4_range_of_m_l3815_381559

/-- Definition of proposition p -/
def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

/-- Definition of proposition q -/
def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

/-- Theorem for part (1) -/
theorem range_of_x_when_m_is_4 (x : ℝ) :
  (∃ m : ℝ, m > 0 ∧ m = 4 ∧ p x ∧ q x m) → 4 < x ∧ x < 5 := by sorry

/-- Theorem for part (2) -/
theorem range_of_m (m : ℝ) :
  (m > 0 ∧ (∀ x : ℝ, ¬(q x m) → ¬(p x)) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m)) →
  (5/3 ≤ m ∧ m ≤ 2) := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_m_is_4_range_of_m_l3815_381559


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3815_381529

theorem algebraic_expression_value (x y : ℝ) (h : x + 2*y - 1 = 0) :
  (2*x + 4*y) / (x^2 + 4*x*y + 4*y^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3815_381529


namespace NUMINAMATH_CALUDE_emilys_small_gardens_l3815_381566

/-- Given Emily's gardening scenario, prove the number of small gardens. -/
theorem emilys_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : 
  total_seeds = 41 →
  big_garden_seeds = 29 →
  seeds_per_small_garden = 4 →
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 3 := by
  sorry

end NUMINAMATH_CALUDE_emilys_small_gardens_l3815_381566


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l3815_381560

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framedDimensions (fp : FramedPainting) : (ℝ × ℝ) :=
  (fp.painting_width + 2 * fp.side_frame_width, fp.painting_height + 4 * fp.side_frame_width)

/-- Calculates the area of the framed painting -/
def framedArea (fp : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions fp
  w * h

/-- Calculates the area of the original painting -/
def paintingArea (fp : FramedPainting) : ℝ :=
  fp.painting_width * fp.painting_height

/-- Theorem stating the ratio of dimensions for the specific framed painting -/
theorem framed_painting_ratio :
  ∀ (fp : FramedPainting),
  fp.painting_width = 20 →
  fp.painting_height = 30 →
  framedArea fp = 2 * paintingArea fp →
  let (w, h) := framedDimensions fp
  w / h = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l3815_381560


namespace NUMINAMATH_CALUDE_quadratic_properties_l3815_381539

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x - 1)^2 + 2

-- Theorem stating the properties of the quadratic function
theorem quadratic_properties :
  (∃ (a : ℝ), ∀ (x : ℝ), f x = f (2 - x)) ∧  -- Axis of symmetry at x = 1
  (∀ (x : ℝ), f x ≤ 2) ∧                    -- Maximum value is 2
  (∃ (x : ℝ), f x = 2)                      -- The maximum value is attained
:= by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3815_381539


namespace NUMINAMATH_CALUDE_linda_expenditure_l3815_381564

def notebook_price : ℝ := 1.20
def notebook_quantity : ℕ := 3
def pencil_box_price : ℝ := 1.50
def pen_box_price : ℝ := 1.70
def marker_pack_price : ℝ := 2.80
def calculator_price : ℝ := 12.50
def item_discount_rate : ℝ := 0.15
def coupon_discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

def total_expenditure : ℝ := 19.52

theorem linda_expenditure :
  let discountable_items_total := notebook_price * notebook_quantity + pencil_box_price + pen_box_price + marker_pack_price
  let discounted_items_total := discountable_items_total * (1 - item_discount_rate)
  let total_after_item_discount := discounted_items_total + calculator_price
  let total_after_coupon := total_after_item_discount * (1 - coupon_discount_rate)
  let final_total := total_after_coupon * (1 + sales_tax_rate)
  final_total = total_expenditure := by
sorry

end NUMINAMATH_CALUDE_linda_expenditure_l3815_381564


namespace NUMINAMATH_CALUDE_expression_evaluation_l3815_381575

theorem expression_evaluation : 
  ((15^15 / 15^14)^3 * 3^5) / 9^2 = 10120 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3815_381575


namespace NUMINAMATH_CALUDE_ben_old_car_sale_amount_l3815_381522

def old_car_cost : ℕ := 1900
def remaining_debt : ℕ := 2000

def new_car_cost : ℕ := 2 * old_car_cost

def amount_paid_off : ℕ := new_car_cost - remaining_debt

theorem ben_old_car_sale_amount : amount_paid_off = 1800 := by
  sorry

end NUMINAMATH_CALUDE_ben_old_car_sale_amount_l3815_381522


namespace NUMINAMATH_CALUDE_modified_sum_theorem_l3815_381503

theorem modified_sum_theorem (S a b : ℝ) (h : a + b = S) :
  (3 * a + 4) + (2 * b + 5) = 3 * S + 9 := by
  sorry

end NUMINAMATH_CALUDE_modified_sum_theorem_l3815_381503


namespace NUMINAMATH_CALUDE_equation_solution_l3815_381570

theorem equation_solution : ∃ x : ℝ, 2 * (x + 3) = 5 * x ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3815_381570


namespace NUMINAMATH_CALUDE_pony_jeans_discount_rate_l3815_381578

theorem pony_jeans_discount_rate :
  let fox_price : ℚ := 15
  let pony_price : ℚ := 18
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let total_savings : ℚ := 9
  let total_discount_rate : ℚ := 25

  ∀ (fox_discount pony_discount : ℚ),
    fox_discount + pony_discount = total_discount_rate →
    fox_quantity * fox_price * (fox_discount / 100) + 
    pony_quantity * pony_price * (pony_discount / 100) = total_savings →
    pony_discount = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_pony_jeans_discount_rate_l3815_381578


namespace NUMINAMATH_CALUDE_match_probabilities_and_expectation_l3815_381500

/-- Represents the outcome of a single game -/
inductive GameOutcome
| A_wins
| B_wins

/-- Represents the state of the match after the first two games -/
structure MatchState :=
  (A_wins : Nat)
  (B_wins : Nat)

/-- The probability of A winning a single game -/
def p_A_win : ℝ := 0.6

/-- The probability of B winning a single game -/
def p_B_win : ℝ := 0.4

/-- The initial state of the match after two games -/
def initial_state : MatchState := ⟨1, 1⟩

/-- The number of wins required to win the match -/
def wins_required : Nat := 3

/-- Calculates the probability of A winning the match given the current state -/
def prob_A_wins_match (state : MatchState) : ℝ :=
  sorry

/-- Calculates the expected number of additional games played -/
def expected_additional_games (state : MatchState) : ℝ :=
  sorry

theorem match_probabilities_and_expectation :
  prob_A_wins_match initial_state = 0.648 ∧
  expected_additional_games initial_state = 2.48 := by
  sorry

end NUMINAMATH_CALUDE_match_probabilities_and_expectation_l3815_381500


namespace NUMINAMATH_CALUDE_a_10_value_l3815_381512

def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value (a : ℕ → ℤ) 
  (h_seq : arithmeticSequence a) 
  (h_a7 : a 7 = 4) 
  (h_a8 : a 8 = 1) : 
  a 10 = -5 := by
sorry

end NUMINAMATH_CALUDE_a_10_value_l3815_381512


namespace NUMINAMATH_CALUDE_yanni_money_problem_l3815_381537

/-- The amount of money Yanni's mother gave him -/
def mothers_gift : ℚ := 0.40

theorem yanni_money_problem :
  let initial_money : ℚ := 0.85
  let found_money : ℚ := 0.50
  let toy_cost : ℚ := 1.60
  let final_balance : ℚ := 0.15
  initial_money + mothers_gift + found_money - toy_cost = final_balance :=
by sorry

end NUMINAMATH_CALUDE_yanni_money_problem_l3815_381537


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l3815_381514

theorem inequality_not_always_true (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  ∃ a b, 0 < b ∧ b < a ∧ ¬((1 / (a - b)) > (1 / b)) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l3815_381514


namespace NUMINAMATH_CALUDE_square_sum_difference_l3815_381547

theorem square_sum_difference (n : ℕ) : 
  (2*n+1)^2 - (2*n-1)^2 + (2*n-3)^2 - (2*n-5)^2 + (2*n-7)^2 - (2*n-9)^2 + 
  (2*n-11)^2 - (2*n-13)^2 + (2*n-15)^2 - (2*n-17)^2 + (2*n-19)^2 - 
  (2*n-21)^2 + (2*n-23)^2 - (2*n-25)^2 + (2*n-27)^2 = 389 :=
by sorry

end NUMINAMATH_CALUDE_square_sum_difference_l3815_381547


namespace NUMINAMATH_CALUDE_infinite_triples_with_coprime_c_l3815_381506

theorem infinite_triples_with_coprime_c : ∃ (a b c : ℕ → ℕ+), 
  (∀ n, (a n)^2 + (b n)^2 = (c n)^4) ∧ 
  (∀ n, Nat.gcd (c n) (c (n + 1)) = 1) := by
  sorry

end NUMINAMATH_CALUDE_infinite_triples_with_coprime_c_l3815_381506


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3815_381573

-- Define the concept of opposite for integers
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_negative_three :
  opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3815_381573


namespace NUMINAMATH_CALUDE_pizza_slices_with_both_toppings_l3815_381508

-- Define the total number of slices
def total_slices : ℕ := 24

-- Define the number of slices with pepperoni
def pepperoni_slices : ℕ := 12

-- Define the number of slices with mushrooms
def mushroom_slices : ℕ := 14

-- Define the number of vegetarian slices
def vegetarian_slices : ℕ := 4

-- Theorem to prove
theorem pizza_slices_with_both_toppings :
  ∃ n : ℕ, 
    -- Every slice has at least one condition met
    (n + (pepperoni_slices - n) + (mushroom_slices - n) + vegetarian_slices = total_slices) ∧
    -- n is the number of slices with both pepperoni and mushrooms
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_with_both_toppings_l3815_381508


namespace NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_square_root_9_l3815_381513

theorem cube_root_27_times_fourth_root_81_times_square_root_9 :
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by sorry

end NUMINAMATH_CALUDE_cube_root_27_times_fourth_root_81_times_square_root_9_l3815_381513
