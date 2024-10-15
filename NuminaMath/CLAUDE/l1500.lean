import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_area_l1500_150088

/-- The area of a parallelogram with given dimensions -/
theorem parallelogram_area (base slant_height : ℝ) (angle : ℝ) : 
  base = 20 → 
  slant_height = 10 → 
  angle = 30 * π / 180 → 
  base * (slant_height * Real.sin angle) = 100 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1500_150088


namespace NUMINAMATH_CALUDE_min_dot_product_plane_vectors_l1500_150047

theorem min_dot_product_plane_vectors (a b : ℝ × ℝ) :
  ‖(2 • a) - b‖ ≤ 3 →
  ∀ (c d : ℝ × ℝ), ‖(2 • c) - d‖ ≤ 3 →
  a • b ≥ -9/8 ∧ a • b ≤ c • d :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_plane_vectors_l1500_150047


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l1500_150019

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a-b)(sin A + sin B) = (c-b)sin C and a = √3, then 5 < b² + c² ≤ 6. -/
theorem triangle_side_sum_range (a b c A B C : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Acute triangle
  A + B + C = π ∧ -- Sum of angles in a triangle
  (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C ∧ -- Given condition
  a = Real.sqrt 3 → -- Given condition
  5 < b^2 + c^2 ∧ b^2 + c^2 ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l1500_150019


namespace NUMINAMATH_CALUDE_solution_is_i_div_3_l1500_150039

/-- The imaginary unit i, where i^2 = -1 -/
noncomputable def i : ℂ := Complex.I

/-- The equation to be solved -/
def equation (x : ℂ) : Prop := 3 + i * x = 5 - 2 * i * x

/-- The theorem stating that i/3 is the solution to the equation -/
theorem solution_is_i_div_3 : equation (i / 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_is_i_div_3_l1500_150039


namespace NUMINAMATH_CALUDE_yard_length_with_32_trees_l1500_150028

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℕ) : ℕ :=
  (numTrees - 1) * distanceBetweenTrees

/-- Theorem: The length of a yard with 32 equally spaced trees and 14 meters between consecutive trees is 434 meters -/
theorem yard_length_with_32_trees : yardLength 32 14 = 434 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_with_32_trees_l1500_150028


namespace NUMINAMATH_CALUDE_max_k_value_l1500_150004

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, (1/m + 2/(1-2*m) ≥ k) → k ≤ 8) ∧
  (∃ k : ℝ, k = 8 ∧ 1/m + 2/(1-2*m) ≥ k) :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l1500_150004


namespace NUMINAMATH_CALUDE_xy_max_value_l1500_150035

theorem xy_max_value (x y : ℝ) (h : x^2 + 2*y^2 - 2*x*y = 4) :
  x*y ≤ 2*Real.sqrt 2 + 2 := by
sorry

end NUMINAMATH_CALUDE_xy_max_value_l1500_150035


namespace NUMINAMATH_CALUDE_parabola_equation_l1500_150074

-- Define a parabola
def Parabola (a b c : ℝ) := {(x, y) : ℝ × ℝ | y = a * x^2 + b * x + c}

-- Define the properties of our specific parabola
def ParabolaProperties (p : Set (ℝ × ℝ)) :=
  ∃ a : ℝ, a ≠ 0 ∧ 
  p = Parabola 0 0 0 ∧  -- vertex at origin
  (∀ x y : ℝ, (x, y) ∈ p → (x, y) ∈ p) ∧  -- y-axis symmetry
  (-4, -2) ∈ p  -- passes through (-4, -2)

-- Theorem statement
theorem parabola_equation :
  ∃ p : Set (ℝ × ℝ), ParabolaProperties p ∧ p = {(x, y) : ℝ × ℝ | x^2 = -8*y} :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1500_150074


namespace NUMINAMATH_CALUDE_salazar_oranges_l1500_150082

theorem salazar_oranges (initial_oranges : ℕ) (sold_fraction : ℚ) 
  (rotten_oranges : ℕ) (remaining_oranges : ℕ) :
  initial_oranges = 7 * 12 →
  sold_fraction = 3 / 7 →
  rotten_oranges = 4 →
  remaining_oranges = 32 →
  ∃ (f : ℚ), 
    0 ≤ f ∧ f ≤ 1 ∧
    (1 - f) * initial_oranges - sold_fraction * ((1 - f) * initial_oranges) - rotten_oranges = remaining_oranges ∧
    f = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_salazar_oranges_l1500_150082


namespace NUMINAMATH_CALUDE_positive_sum_and_product_iff_both_positive_l1500_150095

theorem positive_sum_and_product_iff_both_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_and_product_iff_both_positive_l1500_150095


namespace NUMINAMATH_CALUDE_max_value_DEABC_l1500_150043

/-- Represents a single-digit number -/
def SingleDigit := {n : ℕ // n < 10}

/-- Converts a three-digit number represented by its digits to a natural number -/
def threeDigitToNat (a b c : SingleDigit) : ℕ := 100 * a.val + 10 * b.val + c.val

/-- Converts a two-digit number represented by its digits to a natural number -/
def twoDigitToNat (d e : SingleDigit) : ℕ := 10 * d.val + e.val

/-- Converts a five-digit number represented by its digits to a natural number -/
def fiveDigitToNat (d e a b c : SingleDigit) : ℕ := 
  10000 * d.val + 1000 * e.val + 100 * a.val + 10 * b.val + c.val

theorem max_value_DEABC 
  (A B C D E : SingleDigit)
  (h1 : twoDigitToNat D E = A.val + B.val + C.val)
  (h2 : threeDigitToNat A B C + threeDigitToNat B C A + threeDigitToNat C A B + twoDigitToNat D E = 2016) :
  (∀ A' B' C' D' E', 
    twoDigitToNat D' E' = A'.val + B'.val + C'.val →
    threeDigitToNat A' B' C' + threeDigitToNat B' C' A' + threeDigitToNat C' A' B' + twoDigitToNat D' E' = 2016 →
    fiveDigitToNat D' E' A' B' C' ≤ fiveDigitToNat D E A B C) →
  fiveDigitToNat D E A B C = 18783 :=
sorry

end NUMINAMATH_CALUDE_max_value_DEABC_l1500_150043


namespace NUMINAMATH_CALUDE_examination_student_count_l1500_150040

/-- The total number of students who appeared for the examination -/
def total_students : ℕ := 740

/-- The number of students who failed the examination -/
def failed_students : ℕ := 481

/-- The proportion of students who passed the examination -/
def pass_rate : ℚ := 35 / 100

theorem examination_student_count : 
  total_students = failed_students / (1 - pass_rate) := by
  sorry

end NUMINAMATH_CALUDE_examination_student_count_l1500_150040


namespace NUMINAMATH_CALUDE_painters_work_days_theorem_l1500_150051

/-- The number of work-days required for a given number of painters to complete a job,
    assuming the product of painters and work-days is constant. -/
def work_days (painters : ℕ) (total_work : ℚ) : ℚ :=
  total_work / painters

theorem painters_work_days_theorem (total_work : ℚ) :
  let five_painters_days : ℚ := 3/2
  let four_painters_days : ℚ := work_days 4 (5 * five_painters_days)
  four_painters_days = 15/8 := by sorry

end NUMINAMATH_CALUDE_painters_work_days_theorem_l1500_150051


namespace NUMINAMATH_CALUDE_trailer_homes_count_l1500_150064

/-- Represents the number of new trailer homes added -/
def new_homes : ℕ := 17

/-- The initial number of trailer homes -/
def initial_homes : ℕ := 25

/-- The initial average age of trailer homes (in years) -/
def initial_avg_age : ℕ := 15

/-- The time elapsed since the initial state (in years) -/
def time_elapsed : ℕ := 3

/-- The current average age of all trailer homes (in years) -/
def current_avg_age : ℕ := 12

theorem trailer_homes_count :
  (initial_homes * (initial_avg_age + time_elapsed) + new_homes * time_elapsed) / 
  (initial_homes + new_homes) = current_avg_age := by sorry

end NUMINAMATH_CALUDE_trailer_homes_count_l1500_150064


namespace NUMINAMATH_CALUDE_difference_of_squares_l1500_150055

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1500_150055


namespace NUMINAMATH_CALUDE_initial_student_count_l1500_150067

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) :
  initial_avg = 15 →
  new_avg = 14.9 →
  new_student_weight = 13 →
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_student_count_l1500_150067


namespace NUMINAMATH_CALUDE_min_value_of_f_l1500_150090

theorem min_value_of_f (x : ℝ) (h : x ≥ 5/2) : (x^2 - 4*x + 5) / (2*x - 4) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1500_150090


namespace NUMINAMATH_CALUDE_cubic_function_extrema_l1500_150086

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 27

def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_extrema (a b : ℝ) :
  f_derivative a b (-1) = 0 ∧ f_derivative a b 3 = 0 → a = -3 ∧ b = -9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_l1500_150086


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1500_150014

theorem unique_positive_solution :
  ∃! x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1500_150014


namespace NUMINAMATH_CALUDE_range_of_m_l1500_150009

-- Define the conditions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, p x → q x m) ∧  -- p is sufficient for q
  (∃ x, q x m ∧ ¬p x) ∧ -- p is not necessary for q
  (m > 0) →             -- given condition
  m ≥ 9 :=               -- conclusion
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1500_150009


namespace NUMINAMATH_CALUDE_triangle_area_change_l1500_150096

theorem triangle_area_change (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let a' := 2 * a
  let b' := 1.5 * b
  let c' := c
  let s' := (a' + b' + c') / 2
  let area' := Real.sqrt (s' * (s' - a') * (s' - b') * (s' - c'))
  2 * area < area' ∧ area' < 3 * area :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_change_l1500_150096


namespace NUMINAMATH_CALUDE_quadratic_value_at_zero_l1500_150026

-- Define the quadratic function
def f (h : ℝ) (x : ℝ) : ℝ := -(x + h)^2

-- Define the axis of symmetry
def axis_of_symmetry (h : ℝ) : ℝ := -3

-- Theorem statement
theorem quadratic_value_at_zero (h : ℝ) : 
  axis_of_symmetry h = -3 → f h 0 = -9 := by
  sorry

#check quadratic_value_at_zero

end NUMINAMATH_CALUDE_quadratic_value_at_zero_l1500_150026


namespace NUMINAMATH_CALUDE_difference_in_combined_area_l1500_150078

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem difference_in_combined_area : 
  let sheet1_length : ℝ := 11
  let sheet1_width : ℝ := 13
  let sheet2_length : ℝ := 6.5
  let sheet2_width : ℝ := 11
  let combined_area (l w : ℝ) := 2 * l * w
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 143
  := by sorry

end NUMINAMATH_CALUDE_difference_in_combined_area_l1500_150078


namespace NUMINAMATH_CALUDE_product_abcde_l1500_150080

theorem product_abcde (a b c d e : ℚ) : 
  3 * a + 4 * b + 6 * c + 8 * d + 10 * e = 55 →
  4 * (d + c + e) = b →
  4 * b + 2 * c = a →
  c - 2 = d →
  d + 1 = e →
  a * b * c * d * e = -1912397372 / 78364164096 := by
sorry

end NUMINAMATH_CALUDE_product_abcde_l1500_150080


namespace NUMINAMATH_CALUDE_cone_ratio_l1500_150069

theorem cone_ratio (circumference : ℝ) (volume : ℝ) :
  circumference = 28 * Real.pi →
  volume = 441 * Real.pi →
  ∃ (radius height : ℝ),
    circumference = 2 * Real.pi * radius ∧
    volume = (1/3) * Real.pi * radius^2 * height ∧
    radius / height = 14 / 9 :=
by sorry

end NUMINAMATH_CALUDE_cone_ratio_l1500_150069


namespace NUMINAMATH_CALUDE_line_l_standard_equation_l1500_150033

/-- A line in 2D space defined by parametric equations. -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric line. -/
def line_l : ParametricLine where
  x := fun t => 1 + t
  y := fun t => -1 + t

/-- The standard form of a line equation: ax + by + c = 0 -/
structure StandardLineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The proposed standard equation of the line. -/
def proposed_equation : StandardLineEquation where
  a := 1
  b := -1
  c := -2

theorem line_l_standard_equation :
  ∀ t : ℝ, proposed_equation.a * (line_l.x t) + proposed_equation.b * (line_l.y t) + proposed_equation.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_l_standard_equation_l1500_150033


namespace NUMINAMATH_CALUDE_lg_sum_equals_two_l1500_150097

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_equals_two : lg 4 + lg 25 = 2 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_two_l1500_150097


namespace NUMINAMATH_CALUDE_sachins_age_l1500_150032

theorem sachins_age (sachin_age rahul_age : ℕ) : 
  rahul_age = sachin_age + 14 →
  sachin_age * 9 = rahul_age * 7 →
  sachin_age = 49 := by
sorry

end NUMINAMATH_CALUDE_sachins_age_l1500_150032


namespace NUMINAMATH_CALUDE_cinema_renovation_unique_solution_l1500_150050

theorem cinema_renovation_unique_solution :
  ∃! (x y : ℕ), 
    x > 0 ∧ 
    y > 20 ∧ 
    y * (2 * x + y - 1) = 4008 := by
  sorry

end NUMINAMATH_CALUDE_cinema_renovation_unique_solution_l1500_150050


namespace NUMINAMATH_CALUDE_square_difference_value_l1500_150003

theorem square_difference_value (a b : ℝ) 
  (h1 : 3 * (a + b) = 18) 
  (h2 : a - b = 4) : 
  a^2 - b^2 = 24 := by
sorry

end NUMINAMATH_CALUDE_square_difference_value_l1500_150003


namespace NUMINAMATH_CALUDE_curve_intersection_arithmetic_sequence_l1500_150042

/-- Given a curve C: y = 1/x (x > 0) and points A₁(x₁, 0) and A₂(x₂, 0) where x₂ > x₁ > 0,
    perpendicular lines to the x-axis from A₁ and A₂ intersect C at B₁ and B₂.
    The line B₁B₂ intersects the x-axis at A₃(x₃, 0).
    This theorem proves that x₁, x₃/2, x₂ form an arithmetic sequence. -/
theorem curve_intersection_arithmetic_sequence
  (x₁ x₂ : ℝ)
  (h₁ : 0 < x₁)
  (h₂ : x₁ < x₂)
  (x₃ : ℝ)
  (h₃ : x₃ = x₁ + x₂) :
  x₂ - x₃/2 = x₃/2 - x₁ :=
by sorry

end NUMINAMATH_CALUDE_curve_intersection_arithmetic_sequence_l1500_150042


namespace NUMINAMATH_CALUDE_function_symmetry_l1500_150046

-- Define the function f and constant T
variable (f : ℝ → ℝ) (T : ℝ)

-- Define the conditions
def periodic : Prop := ∀ x, f (x + 2 * T) = f x
def symmetry_1 : Prop := ∀ x, T / 2 ≤ x → x ≤ T → f x = f (T - x)
def antisymmetry : Prop := ∀ x, T ≤ x → x ≤ 3 * T / 2 → f x = -f (x - T)
def symmetry_2 : Prop := ∀ x, 3 * T / 2 ≤ x → x ≤ 2 * T → f x = -f (2 * T - x)

-- State the theorem
theorem function_symmetry 
  (h1 : periodic f T) 
  (h2 : symmetry_1 f T) 
  (h3 : antisymmetry f T) 
  (h4 : symmetry_2 f T) : 
  ∀ x, f x = f (T - x) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l1500_150046


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l1500_150041

theorem sum_remainder_zero : (((7283 + 7284 + 7285 + 7286 + 7287) * 2) % 9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l1500_150041


namespace NUMINAMATH_CALUDE_ordering_abc_l1500_150079

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem ordering_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l1500_150079


namespace NUMINAMATH_CALUDE_gcd_of_sums_of_squares_l1500_150053

theorem gcd_of_sums_of_squares : 
  Nat.gcd (118^2 + 227^2 + 341^2) (119^2 + 226^2 + 340^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_sums_of_squares_l1500_150053


namespace NUMINAMATH_CALUDE_output_for_15_l1500_150005

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 2
  if step1 > 25 then step1 - 7 else step1 + 10

theorem output_for_15 : function_machine 15 = 23 := by
  sorry

end NUMINAMATH_CALUDE_output_for_15_l1500_150005


namespace NUMINAMATH_CALUDE_fuel_station_problem_l1500_150065

/-- Fuel station problem -/
theorem fuel_station_problem 
  (service_cost : ℝ) 
  (fuel_cost_per_liter : ℝ) 
  (total_cost : ℝ) 
  (minivan_tank : ℝ) 
  (truck_tank_ratio : ℝ) 
  (num_trucks : ℕ) 
  (h1 : service_cost = 2.30)
  (h2 : fuel_cost_per_liter = 0.70)
  (h3 : total_cost = 396)
  (h4 : minivan_tank = 65)
  (h5 : truck_tank_ratio = 2.20)
  (h6 : num_trucks = 2) :
  ∃ (num_minivans : ℕ), 
    (num_minivans : ℝ) * (service_cost + minivan_tank * fuel_cost_per_liter) + 
    (num_trucks : ℝ) * (service_cost + truck_tank_ratio * minivan_tank * fuel_cost_per_liter) = 
    total_cost ∧ num_minivans = 4 := by
  sorry

end NUMINAMATH_CALUDE_fuel_station_problem_l1500_150065


namespace NUMINAMATH_CALUDE_problem_statement_l1500_150060

theorem problem_statement (a b : ℝ) (h : a + b - 3 = 0) :
  2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1500_150060


namespace NUMINAMATH_CALUDE_remaining_savings_l1500_150017

def initial_savings : ℕ := 80
def earrings_cost : ℕ := 23
def necklace_cost : ℕ := 48

theorem remaining_savings : 
  initial_savings - (earrings_cost + necklace_cost) = 9 := by sorry

end NUMINAMATH_CALUDE_remaining_savings_l1500_150017


namespace NUMINAMATH_CALUDE_teacher_arrangement_count_l1500_150063

/-- The number of female teachers -/
def num_female : ℕ := 2

/-- The number of male teachers -/
def num_male : ℕ := 4

/-- The number of female teachers per group -/
def female_per_group : ℕ := 1

/-- The number of male teachers per group -/
def male_per_group : ℕ := 2

/-- The total number of groups -/
def num_groups : ℕ := 2

theorem teacher_arrangement_count :
  (num_female.choose female_per_group) * (num_male.choose male_per_group) = 12 := by
  sorry

end NUMINAMATH_CALUDE_teacher_arrangement_count_l1500_150063


namespace NUMINAMATH_CALUDE_expression_equality_l1500_150015

theorem expression_equality : 4 + (-8) / (-4) - (-1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1500_150015


namespace NUMINAMATH_CALUDE_tan_neg_alpha_problem_l1500_150027

theorem tan_neg_alpha_problem (α : Real) (h : Real.tan (-α) = -2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 ∧ Real.sin (2 * α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_alpha_problem_l1500_150027


namespace NUMINAMATH_CALUDE_integral_proof_l1500_150056

open Real

noncomputable def f (x : ℝ) : ℝ := 
  (1/2) * log (abs (x^2 + x + 1)) + 
  (1/sqrt 3) * arctan ((2*x + 1)/sqrt 3) + 
  (1/2) * log (abs (x^2 + 1))

theorem integral_proof (x : ℝ) : 
  deriv f x = (2*x^3 + 2*x^2 + 2*x + 1) / ((x^2 + x + 1) * (x^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l1500_150056


namespace NUMINAMATH_CALUDE_negation_equivalence_l1500_150013

theorem negation_equivalence :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x₀ : ℝ, |x₀ - 2| + |x₀ - 4| ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1500_150013


namespace NUMINAMATH_CALUDE_plan_y_cheaper_at_min_usage_l1500_150021

/-- Cost of Plan X in cents for z MB of data usage -/
def cost_plan_x (z : ℕ) : ℕ := 15 * z

/-- Cost of Plan Y in cents for z MB of data usage, without discount -/
def cost_plan_y_no_discount (z : ℕ) : ℕ := 3000 + 7 * z

/-- Cost of Plan Y in cents for z MB of data usage, with discount -/
def cost_plan_y_with_discount (z : ℕ) : ℕ := 
  if z > 500 then cost_plan_y_no_discount z - 1000 else cost_plan_y_no_discount z

/-- The minimum usage in MB where Plan Y becomes cheaper than Plan X -/
def min_usage : ℕ := 501

theorem plan_y_cheaper_at_min_usage : 
  cost_plan_y_with_discount min_usage < cost_plan_x min_usage ∧
  ∀ z : ℕ, z < min_usage → cost_plan_x z ≤ cost_plan_y_with_discount z :=
by sorry


end NUMINAMATH_CALUDE_plan_y_cheaper_at_min_usage_l1500_150021


namespace NUMINAMATH_CALUDE_sheila_attend_probability_l1500_150076

/-- The probability of rain -/
def prob_rain : ℝ := 0.3

/-- The probability of cloudy weather -/
def prob_cloudy : ℝ := 0.4

/-- The probability of sunshine -/
def prob_sunny : ℝ := 0.3

/-- The probability Sheila attends if it rains -/
def prob_attend_rain : ℝ := 0.25

/-- The probability Sheila attends if it's cloudy -/
def prob_attend_cloudy : ℝ := 0.5

/-- The probability Sheila attends if it's sunny -/
def prob_attend_sunny : ℝ := 0.75

/-- The theorem stating the probability of Sheila attending the picnic -/
theorem sheila_attend_probability : 
  prob_rain * prob_attend_rain + prob_cloudy * prob_attend_cloudy + prob_sunny * prob_attend_sunny = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_sheila_attend_probability_l1500_150076


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l1500_150010

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x + 5

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3*x - y + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l1500_150010


namespace NUMINAMATH_CALUDE_complex_number_location_l1500_150084

theorem complex_number_location :
  let z : ℂ := Complex.I * (1 + Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l1500_150084


namespace NUMINAMATH_CALUDE_gcd_2352_1560_l1500_150075

theorem gcd_2352_1560 : Nat.gcd 2352 1560 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2352_1560_l1500_150075


namespace NUMINAMATH_CALUDE_barbie_earrings_problem_l1500_150068

theorem barbie_earrings_problem (barbie_earrings : ℕ) 
  (h1 : barbie_earrings % 2 = 0)  -- Ensures barbie_earrings is even
  (h2 : ∃ (alissa_given : ℕ), alissa_given = barbie_earrings / 2)
  (h3 : ∃ (alissa_total : ℕ), alissa_total = 3 * (barbie_earrings / 2))
  (h4 : 3 * (barbie_earrings / 2) = 36) :
  barbie_earrings / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_barbie_earrings_problem_l1500_150068


namespace NUMINAMATH_CALUDE_triangle_max_area_l1500_150092

/-- Given a triangle ABC with sides a, b, c and area S, 
    if S = a² - (b-c)² and b + c = 8, 
    then the maximum possible value of S is 64/17 -/
theorem triangle_max_area (a b c S : ℝ) : 
  S = a^2 - (b-c)^2 → b + c = 8 → (∀ S' : ℝ, S' = a'^2 - (b'-c')^2 ∧ b' + c' = 8 → S' ≤ S) → S = 64/17 :=
by sorry


end NUMINAMATH_CALUDE_triangle_max_area_l1500_150092


namespace NUMINAMATH_CALUDE_westward_movement_negative_l1500_150094

/-- Represents the direction of movement --/
inductive Direction
| East
| West

/-- Represents a movement with magnitude and direction --/
structure Movement where
  magnitude : ℝ
  direction : Direction

/-- Converts a movement to a signed real number --/
def Movement.toSignedReal (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.magnitude
  | Direction.West => -m.magnitude

theorem westward_movement_negative 
  (east_convention : Movement.toSignedReal { magnitude := 2, direction := Direction.East } = 2) :
  Movement.toSignedReal { magnitude := 3, direction := Direction.West } = -3 := by
  sorry

end NUMINAMATH_CALUDE_westward_movement_negative_l1500_150094


namespace NUMINAMATH_CALUDE_integral_reciprocal_plus_one_l1500_150001

theorem integral_reciprocal_plus_one (u : ℝ) : 
  ∫ x in (0:ℝ)..(1:ℝ), 1 / (x + 1) = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_plus_one_l1500_150001


namespace NUMINAMATH_CALUDE_min_sum_squares_l1500_150059

def S : Finset Int := {-11, -8, -6, -1, 1, 5, 7, 12}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (∀ a b c d e f g h : Int, 
    a ∈ S → b ∈ S → c ∈ S → d ∈ S → e ∈ S → f ∈ S → g ∈ S → h ∈ S →
    a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g → a ≠ h →
    b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g → b ≠ h →
    c ≠ d → c ≠ e → c ≠ f → c ≠ g → c ≠ h →
    d ≠ e → d ≠ f → d ≠ g → d ≠ h →
    e ≠ f → e ≠ g → e ≠ h →
    f ≠ g → f ≠ h →
    g ≠ h →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 1) ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1500_150059


namespace NUMINAMATH_CALUDE_sqrt_neg_x_squared_meaningful_l1500_150073

theorem sqrt_neg_x_squared_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = -x^2) ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_x_squared_meaningful_l1500_150073


namespace NUMINAMATH_CALUDE_renovation_profit_threshold_l1500_150066

/-- Annual profit without renovation (in millions of yuan) -/
def a (n : ℕ) : ℚ := 500 - 20 * n

/-- Annual profit with renovation (in millions of yuan) -/
def b (n : ℕ) : ℚ := 1000 - 1000 / (2^n)

/-- Cumulative profit without renovation (in millions of yuan) -/
def A (n : ℕ) : ℚ := 500 * n - 10 * n * (n + 1)

/-- Cumulative profit with renovation (in millions of yuan) -/
def B (n : ℕ) : ℚ := 1000 * n - 2600 + 2000 / (2^n)

/-- The minimum number of years for cumulative profit with renovation to exceed that without renovation -/
theorem renovation_profit_threshold : 
  ∀ n : ℕ, n ≥ 5 ↔ B n > A n :=
by sorry

end NUMINAMATH_CALUDE_renovation_profit_threshold_l1500_150066


namespace NUMINAMATH_CALUDE_sand_remaining_proof_l1500_150030

/-- The amount of sand remaining on a truck after transit -/
def sand_remaining (initial : ℝ) (lost : ℝ) : ℝ :=
  initial - lost

/-- Theorem: The amount of sand remaining on the truck is 1.7 pounds -/
theorem sand_remaining_proof (initial : ℝ) (lost : ℝ) 
    (h1 : initial = 4.1)
    (h2 : lost = 2.4) : 
  sand_remaining initial lost = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_sand_remaining_proof_l1500_150030


namespace NUMINAMATH_CALUDE_middle_group_frequency_l1500_150036

/-- Represents a frequency distribution histogram -/
structure FrequencyHistogram where
  num_rectangles : ℕ
  sample_size : ℕ
  middle_area : ℝ
  other_areas : ℝ

/-- Theorem: The frequency of the middle group in a specific histogram -/
theorem middle_group_frequency (h : FrequencyHistogram) 
  (h_num_rectangles : h.num_rectangles = 11)
  (h_area_equality : h.middle_area = h.other_areas)
  (h_sample_size : h.sample_size = 160) :
  (h.middle_area / (h.middle_area + h.other_areas)) * h.sample_size = 80 := by
  sorry

#check middle_group_frequency

end NUMINAMATH_CALUDE_middle_group_frequency_l1500_150036


namespace NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l1500_150002

/-- Calculates the profit percentage given cost price, marked price, and discount rate. -/
def profit_percentage (cost_price marked_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let selling_price := marked_price * (1 - discount_rate)
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percentage is 25% for the given conditions. -/
theorem profit_percentage_is_25_percent :
  profit_percentage (47.50 : ℚ) (62.5 : ℚ) (0.05 : ℚ) = 25 :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_is_25_percent_l1500_150002


namespace NUMINAMATH_CALUDE_evaluate_polynomial_l1500_150025

theorem evaluate_polynomial (a b : ℤ) (h : b = a + 2) :
  b^3 - a*b^2 - a^2*b + a^3 = 8*(a + 1) := by
  sorry

end NUMINAMATH_CALUDE_evaluate_polynomial_l1500_150025


namespace NUMINAMATH_CALUDE_unique_divisible_by_20_l1500_150083

def is_divisible_by_20 (n : ℕ) : Prop := ∃ k : ℕ, n = 20 * k

def four_digit_number (x : ℕ) : ℕ := 1000 * x + 480 + x

theorem unique_divisible_by_20 :
  ∃! x : ℕ, x < 10 ∧ is_divisible_by_20 (four_digit_number x) :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_20_l1500_150083


namespace NUMINAMATH_CALUDE_payment_calculation_l1500_150011

/-- Represents the pricing and discount options for suits and ties -/
structure StorePolicy where
  suit_price : ℕ
  tie_price : ℕ
  option1_free_ties : ℕ
  option2_discount : ℚ

/-- Calculates the payment for Option 1 -/
def option1_payment (policy : StorePolicy) (suits : ℕ) (ties : ℕ) : ℕ :=
  policy.suit_price * suits + policy.tie_price * (ties - suits)

/-- Calculates the payment for Option 2 -/
def option2_payment (policy : StorePolicy) (suits : ℕ) (ties : ℕ) : ℚ :=
  (1 - policy.option2_discount) * (policy.suit_price * suits + policy.tie_price * ties)

/-- Theorem statement for the payment calculations -/
theorem payment_calculation (x : ℕ) (h : x > 10) :
  let policy : StorePolicy := {
    suit_price := 1000,
    tie_price := 200,
    option1_free_ties := 1,
    option2_discount := 1/10
  }
  option1_payment policy 10 x = 200 * x + 8000 ∧
  option2_payment policy 10 x = 180 * x + 9000 := by
  sorry

end NUMINAMATH_CALUDE_payment_calculation_l1500_150011


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1500_150057

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry -- Additional conditions to ensure the octagon is regular

/-- The octagon formed by joining the midpoints of a regular octagon's sides -/
def midpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of a regular octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (midpointOctagon o) = (1 : ℝ) / 4 * area o := by
  sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l1500_150057


namespace NUMINAMATH_CALUDE_target_probability_l1500_150099

/-- The probability of hitting a target in one shot. -/
def p : ℝ := 0.6

/-- The number of shots taken. -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in three shots. -/
def prob_at_least_two : ℝ := 3 * p^2 * (1 - p) + p^3

theorem target_probability :
  prob_at_least_two = 0.648 := by
  sorry

end NUMINAMATH_CALUDE_target_probability_l1500_150099


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l1500_150093

theorem condition_neither_sufficient_nor_necessary :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b ≥ ((a + b) / 2)^2) ∧
  (∃ a b : ℝ, a * b < ((a + b) / 2)^2 ∧ (a ≤ 0 ∨ b ≤ 0)) := by sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l1500_150093


namespace NUMINAMATH_CALUDE_expression_simplification_l1500_150018

theorem expression_simplification (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m^2 - 1/n^2)^m * (n + 1/m)^(n-m) / ((n^2 - 1/m^2)^n * (m - 1/n)^(m-n)) = (m/n)^(m+n) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1500_150018


namespace NUMINAMATH_CALUDE_max_t_value_l1500_150077

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 - 2*x + Real.log (x + 1)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - Real.log (x + 1) + x^3

theorem max_t_value (m : ℝ) (t : ℝ) :
  m ∈ Set.Icc (-4) (-1) →
  (∀ x ∈ Set.Icc 1 t, g m x ≤ g m 1) →
  t ≤ (1 + Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_l1500_150077


namespace NUMINAMATH_CALUDE_tens_digit_of_7_pow_2005_l1500_150045

/-- The last two digits of 7^n follow a cycle of length 4 -/
def last_two_digits_cycle : List (Fin 100) := [7, 49, 43, 1]

/-- The tens digit of a number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_7_pow_2005 :
  tens_digit (7^2005 % 100) = 0 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_7_pow_2005_l1500_150045


namespace NUMINAMATH_CALUDE_work_time_problem_l1500_150058

/-- The work time problem for Mr. Willson -/
theorem work_time_problem (total_time tuesday wednesday thursday friday : ℚ) :
  total_time = 4 ∧
  tuesday = 1/2 ∧
  wednesday = 2/3 ∧
  thursday = 5/6 ∧
  friday = 75/60 →
  total_time - (tuesday + wednesday + thursday + friday) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_work_time_problem_l1500_150058


namespace NUMINAMATH_CALUDE_exactly_one_divisible_l1500_150049

theorem exactly_one_divisible (p a b c d : ℕ) : 
  Prime p → 
  p % 2 = 1 →
  0 < a → a < p →
  0 < b → b < p →
  0 < c → c < p →
  0 < d → d < p →
  p ∣ (a^2 + b^2) →
  p ∣ (c^2 + d^2) →
  (p ∣ (a*c + b*d) ∧ ¬(p ∣ (a*d + b*c))) ∨ (¬(p ∣ (a*c + b*d)) ∧ p ∣ (a*d + b*c)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_divisible_l1500_150049


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1500_150020

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (hyp : a^2 + b^2 = c^2) -- Pythagorean theorem
  (hyp_length : c = 5) -- Hypotenuse length
  (side_length : a = 3) -- Known side length
  : b = 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1500_150020


namespace NUMINAMATH_CALUDE_bisecting_line_sum_l1500_150048

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(9, 0) -/
structure Triangle where
  P : ℝ × ℝ := (0, 10)
  Q : ℝ × ℝ := (3, 0)
  R : ℝ × ℝ := (9, 0)

/-- A line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The line through Q that bisects the area of the triangle -/
def bisectingLine (t : Triangle) : Line :=
  sorry

theorem bisecting_line_sum (t : Triangle) :
  let l := bisectingLine t
  l.slope + l.yIntercept = -20/3 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l1500_150048


namespace NUMINAMATH_CALUDE_median_divided_triangle_area_l1500_150022

/-- Given a triangle with sides 13, 14, and 15 cm, the area of each smaller triangle
    formed by its medians is 14 cm². -/
theorem median_divided_triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) :
  let s := (a + b + c) / 2
  let total_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  total_area / 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_median_divided_triangle_area_l1500_150022


namespace NUMINAMATH_CALUDE_S_at_one_l1500_150081

def R (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

def S (x : ℝ) : ℝ := (3 + 2) * x^3 + (-5 + 2) * x + (4 + 2)

theorem S_at_one : S 1 = 8 := by sorry

end NUMINAMATH_CALUDE_S_at_one_l1500_150081


namespace NUMINAMATH_CALUDE_a3_greater_b3_l1500_150061

/-- Two sequences satisfying the given conditions -/
def sequences (a b : ℕ+ → ℝ) : Prop :=
  (∀ n, a n + b n = 700) ∧
  (∀ n, a (n + 1) = (7/10) * a n + (2/5) * b n) ∧
  (a 6 = 400)

/-- Theorem stating that a_3 > b_3 for sequences satisfying the given conditions -/
theorem a3_greater_b3 (a b : ℕ+ → ℝ) (h : sequences a b) : a 3 > b 3 := by
  sorry

end NUMINAMATH_CALUDE_a3_greater_b3_l1500_150061


namespace NUMINAMATH_CALUDE_cookies_remaining_l1500_150024

theorem cookies_remaining (total_taken : ℕ) (h1 : total_taken = 11) 
  (h2 : total_taken * 2 = total_taken + total_taken) : 
  total_taken = total_taken * 2 - total_taken := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_l1500_150024


namespace NUMINAMATH_CALUDE_prob_odd_score_is_35_72_l1500_150098

/-- Represents the dartboard with given dimensions and point values -/
structure Dartboard :=
  (outer_radius : ℝ)
  (inner_radius : ℝ)
  (inner_points : Fin 3 → ℕ)
  (outer_points : Fin 3 → ℕ)

/-- Calculates the probability of scoring an odd sum with two darts -/
def prob_odd_score (db : Dartboard) : ℚ :=
  sorry

/-- The specific dartboard described in the problem -/
def problem_dartboard : Dartboard :=
  { outer_radius := 8
  , inner_radius := 4
  , inner_points := ![3, 4, 4]
  , outer_points := ![4, 3, 3] }

theorem prob_odd_score_is_35_72 :
  prob_odd_score problem_dartboard = 35 / 72 :=
sorry

end NUMINAMATH_CALUDE_prob_odd_score_is_35_72_l1500_150098


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1500_150007

-- Define the hyperbola C
def hyperbola_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the circle F
def circle_F (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 3 = 0

-- Define the right focus F of hyperbola C
def right_focus (c : ℝ) : Prop :=
  c = 2

-- Define the distance from F to asymptote
def distance_to_asymptote (b : ℝ) : Prop :=
  b = 1

-- Theorem statement
theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ),
  (∃ x y, hyperbola_C x y a b) →
  (∃ x y, circle_F x y) →
  right_focus c →
  distance_to_asymptote b →
  c^2 = a^2 + b^2 →
  c / a = 2 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1500_150007


namespace NUMINAMATH_CALUDE_sally_bought_three_frames_l1500_150070

/-- The number of photograph frames Sally bought -/
def frames_bought (frame_cost change_received total_paid : ℕ) : ℕ :=
  (total_paid - change_received) / frame_cost

/-- Theorem stating that Sally bought 3 photograph frames -/
theorem sally_bought_three_frames :
  frames_bought 3 11 20 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sally_bought_three_frames_l1500_150070


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l1500_150029

theorem largest_integer_with_remainder (n : ℕ) : 
  n < 100 ∧ n % 6 = 4 → n ≤ 94 :=
by
  sorry

theorem ninety_four_satisfies_conditions : 
  94 < 100 ∧ 94 % 6 = 4 :=
by
  sorry

theorem ninety_four_is_largest : 
  ∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ 94 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_four_satisfies_conditions_ninety_four_is_largest_l1500_150029


namespace NUMINAMATH_CALUDE_mary_book_count_l1500_150062

def book_count (initial : ℕ) (book_club : ℕ) (lent_jane : ℕ) (returned_alice : ℕ)
  (bought_5th_month : ℕ) (bought_yard_sales : ℕ) (birthday_daughter : ℕ)
  (birthday_mother : ℕ) (from_sister : ℕ) (buy_one_get_one : ℕ)
  (donated_charity : ℕ) (borrowed_neighbor : ℕ) (sold_used : ℕ) : ℕ :=
  initial + book_club - lent_jane + returned_alice + bought_5th_month +
  bought_yard_sales + birthday_daughter + birthday_mother + from_sister +
  buy_one_get_one - donated_charity - borrowed_neighbor - sold_used

theorem mary_book_count :
  book_count 200 (2 * 12) 10 5 15 8 1 8 6 4 30 5 7 = 219 := by
  sorry

end NUMINAMATH_CALUDE_mary_book_count_l1500_150062


namespace NUMINAMATH_CALUDE_total_points_theorem_l1500_150038

/-- The total number of participating teams -/
def num_teams : ℕ := 70

/-- The total number of points earned on question 33 -/
def points_q33 : ℕ := 3

/-- The total number of points earned on question 34 -/
def points_q34 : ℕ := 6

/-- The total number of points earned on question 35 -/
def points_q35 : ℕ := 4

/-- The total number of points A earned over all participating teams on questions 33, 34, and 35 -/
def A : ℕ := points_q33 + points_q34 + points_q35

theorem total_points_theorem : A = 13 := by sorry

end NUMINAMATH_CALUDE_total_points_theorem_l1500_150038


namespace NUMINAMATH_CALUDE_altitude_segment_length_l1500_150052

/-- An acute triangle with two altitudes dividing the sides -/
structure AcuteTriangleWithAltitudes where
  /-- The triangle is acute -/
  is_acute : Bool
  /-- Lengths of segments created by altitudes -/
  segment1 : ℝ
  segment2 : ℝ
  segment3 : ℝ
  segment4 : ℝ
  /-- Conditions on segment lengths -/
  h1 : segment1 = 6
  h2 : segment2 = 4
  h3 : segment3 = 3

/-- The theorem stating that the fourth segment length is 9/7 -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) : t.segment4 = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_length_l1500_150052


namespace NUMINAMATH_CALUDE_carlos_has_largest_result_l1500_150012

def starting_number : ℕ := 12

def alice_result : ℕ := ((starting_number - 2) * 3) + 3

def ben_result : ℕ := ((starting_number * 3) - 2) + 3

def carlos_result : ℕ := (starting_number - 2 + 3) * 3

theorem carlos_has_largest_result :
  carlos_result > alice_result ∧ carlos_result > ben_result :=
sorry

end NUMINAMATH_CALUDE_carlos_has_largest_result_l1500_150012


namespace NUMINAMATH_CALUDE_shane_photos_l1500_150071

theorem shane_photos (total_photos : ℕ) (jan_photos_per_day : ℕ) (jan_days : ℕ) (feb_weeks : ℕ)
  (h1 : total_photos = 146)
  (h2 : jan_photos_per_day = 2)
  (h3 : jan_days = 31)
  (h4 : feb_weeks = 4) :
  (total_photos - jan_photos_per_day * jan_days) / feb_weeks = 21 := by
  sorry

end NUMINAMATH_CALUDE_shane_photos_l1500_150071


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l1500_150037

theorem remainder_sum_mod_seven : (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l1500_150037


namespace NUMINAMATH_CALUDE_fundraising_contribution_l1500_150016

theorem fundraising_contribution (total_amount : ℕ) (num_participants : ℕ) 
  (h1 : total_amount = 2400) (h2 : num_participants = 9) : 
  (total_amount + num_participants - 1) / num_participants = 267 :=
by
  sorry

#check fundraising_contribution

end NUMINAMATH_CALUDE_fundraising_contribution_l1500_150016


namespace NUMINAMATH_CALUDE_no_special_polyhedron_l1500_150089

-- Define a polyhedron structure
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangle_faces : ℕ
  pentagon_faces : ℕ
  even_degree_vertices : ℕ

-- Define the conditions for our specific polyhedron
def SpecialPolyhedron (p : Polyhedron) : Prop :=
  p.faces = p.triangle_faces + p.pentagon_faces ∧
  p.pentagon_faces = 1 ∧
  p.vertices = p.even_degree_vertices ∧
  p.vertices - p.edges + p.faces = 2 ∧  -- Euler's formula
  3 * p.triangle_faces + 5 * p.pentagon_faces = 2 * p.edges

-- Theorem stating that such a polyhedron does not exist
theorem no_special_polyhedron :
  ¬ ∃ (p : Polyhedron), SpecialPolyhedron p :=
sorry

end NUMINAMATH_CALUDE_no_special_polyhedron_l1500_150089


namespace NUMINAMATH_CALUDE_equation_solution_l1500_150044

theorem equation_solution :
  ∃! x : ℝ, x ≠ 1 ∧ (3 * x) / (x - 1) = 2 + 1 / (x - 1) ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1500_150044


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1500_150008

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 * a 5 = 1 →
  a 8 * a 9 = 16 →
  a 6 * a 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1500_150008


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1500_150023

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ (∀ y : ℝ, y^2 - 2*y + k = 0 → y = x)) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1500_150023


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1500_150054

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 2) (h2 : m * n > 0) :
  1 / m + 1 / n ≥ 2 ∧ (1 / m + 1 / n = 2 ↔ m = 1 ∧ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1500_150054


namespace NUMINAMATH_CALUDE_cube_cut_surface_area_l1500_150085

/-- Calculates the total surface area of small blocks after cutting a cube -/
def total_surface_area (edge_length : ℝ) (horizontal_cuts : ℕ) (vertical_cuts : ℕ) : ℝ :=
  let original_surface_area := 6 * edge_length^2
  let horizontal_new_area := 2 * edge_length^2 * (2 * horizontal_cuts)
  let vertical_new_area := 2 * edge_length^2 * (2 * vertical_cuts)
  original_surface_area + horizontal_new_area + vertical_new_area

/-- Theorem: The total surface area of all small blocks after cutting a cube with edge length 2,
    4 horizontal cuts, and 5 vertical cuts, is equal to 96 square units -/
theorem cube_cut_surface_area :
  total_surface_area 2 4 5 = 96 := by sorry

end NUMINAMATH_CALUDE_cube_cut_surface_area_l1500_150085


namespace NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1500_150031

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) :
  speed1 > 0 →
  speed2 > 0 →
  (speed1 + speed2) / 2 = (speed1 * 1 + speed2 * 1) / (1 + 1) := by
  sorry

/-- The average speed of a car traveling 90 km/h for the first hour and 40 km/h for the second hour is 65 km/h -/
theorem car_average_speed :
  let speed1 : ℝ := 90
  let speed2 : ℝ := 40
  (speed1 + speed2) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_car_average_speed_l1500_150031


namespace NUMINAMATH_CALUDE_x1_range_l1500_150072

/-- The function f as defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (x + 1) - Real.exp x + x^2 + 2 * m * (x - 1)

/-- The theorem stating the range of x1 -/
theorem x1_range (m : ℝ) (hm : m > 0) :
  {x1 : ℝ | ∀ x2, x1 + x2 = 1 → f m x1 ≥ f m x2} = Set.Ici (1/2) :=
sorry

end NUMINAMATH_CALUDE_x1_range_l1500_150072


namespace NUMINAMATH_CALUDE_average_marks_abcd_l1500_150034

theorem average_marks_abcd (a b c d e : ℝ) : 
  ((a + b + c) / 3 = 48) →
  ((b + c + d + e) / 4 = 48) →
  (e = d + 3) →
  (a = 43) →
  ((a + b + c + d) / 4 = 47) :=
by sorry

end NUMINAMATH_CALUDE_average_marks_abcd_l1500_150034


namespace NUMINAMATH_CALUDE_savanna_animal_count_l1500_150006

/-- The number of animals in Savanna National Park -/
def savanna_total (safari_lions : ℕ) : ℕ :=
  let safari_snakes := safari_lions / 2
  let safari_giraffes := safari_snakes - 10
  let savanna_lions := safari_lions * 2
  let savanna_snakes := safari_snakes * 3
  let savanna_giraffes := safari_giraffes + 20
  savanna_lions + savanna_snakes + savanna_giraffes

/-- Theorem stating the total number of animals in Savanna National Park -/
theorem savanna_animal_count : savanna_total 100 = 410 := by
  sorry

end NUMINAMATH_CALUDE_savanna_animal_count_l1500_150006


namespace NUMINAMATH_CALUDE_complex_square_l1500_150091

-- Define the complex number i
axiom i : ℂ
axiom i_squared : i * i = -1

-- State the theorem
theorem complex_square : (1 + i) * (1 + i) = 2 * i := by sorry

end NUMINAMATH_CALUDE_complex_square_l1500_150091


namespace NUMINAMATH_CALUDE_non_monotonic_function_parameter_range_l1500_150087

/-- The function f(x) = (1/3)x^3 - x^2 + ax - 5 is not monotonic in the interval [-1, 2] -/
theorem non_monotonic_function_parameter_range (a : ℝ) : 
  (∃ x y, x ∈ Set.Icc (-1 : ℝ) 2 ∧ y ∈ Set.Icc (-1 : ℝ) 2 ∧ x < y ∧ 
    ((1/3)*x^3 - x^2 + a*x) > ((1/3)*y^3 - y^2 + a*y)) ↔ 
  a ∈ Set.Ioo (-3 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_non_monotonic_function_parameter_range_l1500_150087


namespace NUMINAMATH_CALUDE_workshop_workers_l1500_150000

/-- The total number of workers in a workshop with specific salary conditions -/
theorem workshop_workers (average_salary : ℕ) (technician_count : ℕ) (technician_salary : ℕ) (other_salary : ℕ) :
  average_salary = 8000 →
  technician_count = 7 →
  technician_salary = 20000 →
  other_salary = 6000 →
  ∃ (total_workers : ℕ),
    total_workers = technician_count + (technician_count * technician_salary + (total_workers - technician_count) * other_salary) / average_salary ∧
    total_workers = 49 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l1500_150000
