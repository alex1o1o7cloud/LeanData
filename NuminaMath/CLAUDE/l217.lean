import Mathlib

namespace NUMINAMATH_CALUDE_inequality_system_solution_l217_21756

theorem inequality_system_solution :
  let S := {x : ℝ | (2*x - 6 < 3*x) ∧ (x - 2 + (x-1)/3 ≤ 1)}
  S = {x : ℝ | -6 < x ∧ x ≤ 5/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l217_21756


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l217_21731

/-- An isosceles triangle with sides of length 3 and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 6 → b = 6 → c = 3 →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l217_21731


namespace NUMINAMATH_CALUDE_sum_greater_than_twice_a_l217_21794

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := x^2 + 2 * Real.cos x

def g (x : ℝ) : ℝ := (deriv f) x - 5 * x + 5 * a * Real.log x

theorem sum_greater_than_twice_a (h₁ : x₁ ≠ x₂) (h₂ : g a x₁ = g a x₂) : 
  x₁ + x₂ > 2 * a := by
  sorry

end

end NUMINAMATH_CALUDE_sum_greater_than_twice_a_l217_21794


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l217_21787

theorem arithmetic_square_root_of_nine : ∃! x : ℝ, x ≥ 0 ∧ x^2 = 9 :=
  by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_nine_l217_21787


namespace NUMINAMATH_CALUDE_regular_soda_count_l217_21704

/-- The number of regular soda bottles in a grocery store -/
def regular_soda : ℕ := sorry

/-- The number of diet soda bottles in a grocery store -/
def diet_soda : ℕ := 40

/-- The total number of regular and diet soda bottles in a grocery store -/
def total_regular_and_diet : ℕ := 89

/-- Theorem stating that the number of regular soda bottles is 49 -/
theorem regular_soda_count : regular_soda = 49 := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_count_l217_21704


namespace NUMINAMATH_CALUDE_expression_factorization_l217_21721

theorem expression_factorization (a b c : ℝ) (h : c ≠ 0) :
  3 * a^3 * (b^2 - c^2) - 2 * b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2) =
  (a - b) * (b - c) * (c - a) * (3 * a^2 - 2 * b^2 - 3 * a^3 / c + c) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l217_21721


namespace NUMINAMATH_CALUDE_complex_equality_l217_21789

-- Define the complex numbers
def z1 (x y : ℝ) : ℂ := x - 1 + y * Complex.I
def z2 (x : ℝ) : ℂ := Complex.I - 3 * x

-- Theorem statement
theorem complex_equality (x y : ℝ) :
  z1 x y = z2 x → x = 1/4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l217_21789


namespace NUMINAMATH_CALUDE_first_day_distance_l217_21777

theorem first_day_distance (total_distance : ℝ) (days : ℕ) (ratio : ℝ) 
  (h1 : total_distance = 378)
  (h2 : days = 6)
  (h3 : ratio = 1/2) :
  let first_day := total_distance * (1 - ratio) / (1 - ratio^days)
  first_day = 192 := by sorry

end NUMINAMATH_CALUDE_first_day_distance_l217_21777


namespace NUMINAMATH_CALUDE_greatest_k_value_l217_21738

theorem greatest_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 7 = 0 ∧ 
    x₂^2 + k*x₂ + 7 = 0 ∧ 
    |x₁ - x₂| = Real.sqrt 85) →
  k ≤ Real.sqrt 113 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l217_21738


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l217_21790

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 8 < 4*x - 1 ∧ (1/2)*x ≥ 4 - (3/2)*x}
  S = {x : ℝ | x > 3} :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l217_21790


namespace NUMINAMATH_CALUDE_expression_evaluation_l217_21736

theorem expression_evaluation (x y : ℝ) (hx : x = 1) (hy : y = -2) :
  ((x - y)^2 - x*(3*x - 2*y) + (x + y)*(x - y)) / (2*x) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l217_21736


namespace NUMINAMATH_CALUDE_rotation_result_l217_21786

-- Define the shapes
inductive Shape
  | Rectangle
  | SmallCircle
  | Pentagon

-- Define the positions
inductive Position
  | Top
  | LeftBottom
  | RightBottom

-- Define the circular plane
structure CircularPlane where
  shapes : List Shape
  positions : List Position
  arrangement : Shape → Position

-- Define the rotation
def rotate150 (plane : CircularPlane) : CircularPlane := sorry

-- Theorem statement
theorem rotation_result (plane : CircularPlane) :
  plane.arrangement Shape.Rectangle = Position.Top →
  plane.arrangement Shape.SmallCircle = Position.LeftBottom →
  plane.arrangement Shape.Pentagon = Position.RightBottom →
  (rotate150 plane).arrangement Shape.SmallCircle = Position.Top := by
  sorry

end NUMINAMATH_CALUDE_rotation_result_l217_21786


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l217_21781

theorem complex_number_coordinates (a : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (a - 2 : ℂ) + (a + 1) * i = b * i) (h3 : b ≠ 0) :
  (a - 3 * i) / (2 - i) = 7/5 - 4/5 * i :=
sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l217_21781


namespace NUMINAMATH_CALUDE_value_of_B_l217_21751

theorem value_of_B : ∃ B : ℝ, (3 * B + 2 = 20) ∧ (B = 6) := by
  sorry

end NUMINAMATH_CALUDE_value_of_B_l217_21751


namespace NUMINAMATH_CALUDE_f_equals_three_implies_x_is_sqrt_three_l217_21750

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_equals_three_implies_x_is_sqrt_three :
  ∀ x : ℝ, f x = 3 → x = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_f_equals_three_implies_x_is_sqrt_three_l217_21750


namespace NUMINAMATH_CALUDE_bruno_books_l217_21718

theorem bruno_books (initial_books : ℝ) : 
  initial_books - 4.5 + 10.25 = 39.75 → initial_books = 34 := by
sorry

end NUMINAMATH_CALUDE_bruno_books_l217_21718


namespace NUMINAMATH_CALUDE_negative_square_inequality_l217_21785

theorem negative_square_inequality (a b : ℝ) : a < b → b < 0 → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_inequality_l217_21785


namespace NUMINAMATH_CALUDE_function_relationship_l217_21709

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x ≥ -4 → y ≥ -4 → x < y → f x < f y)
variable (h2 : ∀ x, f (x - 4) = f (-x - 4))

-- State the theorem
theorem function_relationship :
  f (-4) < f (-6) ∧ f (-6) < f 0 :=
sorry

end NUMINAMATH_CALUDE_function_relationship_l217_21709


namespace NUMINAMATH_CALUDE_function_increasing_in_interval_l217_21702

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * sin (ω * x) + cos (ω * x)

theorem function_increasing_in_interval 
  (h_symmetry : ∀ (x : ℝ), f ω (π/6 - x) = f ω (π/6 + x))
  (h_smallest_ω : ∀ (ω' : ℝ), ω' > 0 → ω' ≥ ω)
  : StrictMonoOn f (Set.Ioo 0 (π/6)) := by sorry

end NUMINAMATH_CALUDE_function_increasing_in_interval_l217_21702


namespace NUMINAMATH_CALUDE_apples_per_box_is_10_l217_21708

/-- The number of apples in each box. -/
def apples_per_box : ℕ := 10

/-- The number of boxes Merry had on Saturday. -/
def saturday_boxes : ℕ := 50

/-- The number of boxes Merry had on Sunday. -/
def sunday_boxes : ℕ := 25

/-- The total number of apples Merry sold. -/
def sold_apples : ℕ := 720

/-- The number of boxes Merry has left. -/
def remaining_boxes : ℕ := 3

/-- Theorem stating that the number of apples in each box is 10. -/
theorem apples_per_box_is_10 :
  apples_per_box * (saturday_boxes + sunday_boxes) - sold_apples = apples_per_box * remaining_boxes :=
by sorry

end NUMINAMATH_CALUDE_apples_per_box_is_10_l217_21708


namespace NUMINAMATH_CALUDE_number_set_properties_l217_21714

/-- A set of natural numbers excluding 1 -/
def NumberSet : Set ℕ :=
  {n : ℕ | n > 1}

/-- Predicate for a number being prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Predicate for a number being composite -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

theorem number_set_properties (S : Set ℕ) (h : S = NumberSet) :
  (¬∀ n ∈ S, isComposite n) →
  (∃ n ∈ S, isPrime n) ∧
  (∀ n ∈ S, ¬isComposite n) ∧
  (∀ n ∈ S, isPrime n) ∧
  (∃ n ∈ S, isComposite n ∧ ∃ m ∈ S, isPrime m) ∧
  (∃ n ∈ S, isPrime n ∧ ∃ m ∈ S, isComposite m) :=
by sorry

end NUMINAMATH_CALUDE_number_set_properties_l217_21714


namespace NUMINAMATH_CALUDE_f_inverse_composition_l217_21775

def f : Fin 6 → Fin 6
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 5
| 5 => 3
| 6 => 1

theorem f_inverse_composition (h : Function.Bijective f) :
  (Function.invFun f (Function.invFun f (Function.invFun f 6))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_inverse_composition_l217_21775


namespace NUMINAMATH_CALUDE_cuboctahedron_volume_side_length_one_l217_21745

/-- A cuboctahedron is a polyhedron with 8 triangular faces and 6 square faces. -/
structure Cuboctahedron where
  side_length : ℝ

/-- The volume of a cuboctahedron. -/
noncomputable def volume (c : Cuboctahedron) : ℝ :=
  (5 * Real.sqrt 2) / 3

/-- Theorem: The volume of a cuboctahedron with side length 1 is (5 * √2) / 3. -/
theorem cuboctahedron_volume_side_length_one :
  volume { side_length := 1 } = (5 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cuboctahedron_volume_side_length_one_l217_21745


namespace NUMINAMATH_CALUDE_marbles_given_to_brother_l217_21722

theorem marbles_given_to_brother 
  (total_marbles : ℕ) 
  (mario_ratio : ℕ) 
  (manny_ratio : ℕ) 
  (manny_current : ℕ) 
  (h1 : total_marbles = 36)
  (h2 : mario_ratio = 4)
  (h3 : manny_ratio = 5)
  (h4 : manny_current = 18) :
  (manny_ratio * total_marbles) / (mario_ratio + manny_ratio) - manny_current = 2 :=
sorry

end NUMINAMATH_CALUDE_marbles_given_to_brother_l217_21722


namespace NUMINAMATH_CALUDE_continuity_at_two_delta_epsilon_relation_l217_21798

def f (x : ℝ) : ℝ := -5 * x^2 - 8

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by
  sorry

theorem delta_epsilon_relation :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 25 ∧
    ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuity_at_two_delta_epsilon_relation_l217_21798


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l217_21755

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 5 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 5 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l217_21755


namespace NUMINAMATH_CALUDE_r_fourth_plus_reciprocal_l217_21754

theorem r_fourth_plus_reciprocal (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_reciprocal_l217_21754


namespace NUMINAMATH_CALUDE_gcf_lcm_problem_l217_21783

theorem gcf_lcm_problem : Nat.gcd (Nat.lcm 18 30) (Nat.lcm 21 28) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_problem_l217_21783


namespace NUMINAMATH_CALUDE_min_units_B_required_twenty_units_B_not_sufficient_l217_21726

/-- Profit from selling one unit of model A (in thousand yuan) -/
def profit_A : ℝ := 3

/-- Profit from selling one unit of model B (in thousand yuan) -/
def profit_B : ℝ := 5

/-- Total number of units to be purchased -/
def total_units : ℕ := 30

/-- Minimum desired profit (in thousand yuan) -/
def min_profit : ℝ := 131

/-- Function to calculate the profit based on the number of model B units -/
def calculate_profit (units_B : ℕ) : ℝ :=
  profit_B * units_B + profit_A * (total_units - units_B)

/-- Theorem stating the minimum number of model B units required -/
theorem min_units_B_required :
  ∀ k : ℕ, k ≥ 21 → calculate_profit k ≥ min_profit :=
by sorry

/-- Theorem stating that 20 units of model B is not sufficient -/
theorem twenty_units_B_not_sufficient :
  calculate_profit 20 < min_profit :=
by sorry

end NUMINAMATH_CALUDE_min_units_B_required_twenty_units_B_not_sufficient_l217_21726


namespace NUMINAMATH_CALUDE_polynomial_identity_l217_21723

theorem polynomial_identity (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x, (1 - 2*x)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  (a + a₂ + a₄)^2 - (a₁ + a₃)^2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l217_21723


namespace NUMINAMATH_CALUDE_triangle_operation_result_l217_21715

-- Define the triangle operation
def triangle (P Q : ℚ) : ℚ := (P + Q) / 3

-- State the theorem
theorem triangle_operation_result :
  triangle 3 (triangle 6 9) = 8 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_operation_result_l217_21715


namespace NUMINAMATH_CALUDE_rotation_of_point_l217_21727

def rotate90ClockwiseAboutOrigin (x y : ℝ) : ℝ × ℝ := (y, -x)

theorem rotation_of_point :
  let D : ℝ × ℝ := (-3, 2)
  rotate90ClockwiseAboutOrigin D.1 D.2 = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_rotation_of_point_l217_21727


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l217_21768

-- Define a structure for shapes with diagonals
structure ShapeWithDiagonals :=
  (diagonal1 : ℝ)
  (diagonal2 : ℝ)

-- Define rectangle and square
class Rectangle extends ShapeWithDiagonals

class Square extends Rectangle

-- State the theorem about rectangle diagonals
axiom rectangle_diagonals_equal (r : Rectangle) : r.diagonal1 = r.diagonal2

-- State that a square is a rectangle
axiom square_is_rectangle (s : Square) : Rectangle

-- Theorem to prove
theorem square_diagonals_equal (s : Square) : 
  (square_is_rectangle s).diagonal1 = (square_is_rectangle s).diagonal2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonals_equal_l217_21768


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l217_21746

theorem diophantine_equation_solutions :
  ∃! (solutions : Set (ℤ × ℤ)),
    solutions = {(4, 9), (4, -9), (-4, 9), (-4, -9)} ∧
    ∀ (x y : ℤ), (x, y) ∈ solutions ↔ 3 * x^2 + 5 * y^2 = 453 :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l217_21746


namespace NUMINAMATH_CALUDE_incircle_radius_eq_area_div_semiperimeter_l217_21799

/-- Triangle DEF with given angles and side length --/
structure TriangleDEF where
  D : ℝ
  E : ℝ
  F : ℝ
  DE : ℝ
  angle_D_eq : D = 75
  angle_E_eq : E = 45
  angle_F_eq : F = 60
  side_DE_eq : DE = 20

/-- The radius of the incircle of triangle DEF --/
def incircle_radius (t : TriangleDEF) : ℝ := sorry

/-- The semi-perimeter of triangle DEF --/
def semi_perimeter (t : TriangleDEF) : ℝ := sorry

/-- The area of triangle DEF --/
def triangle_area (t : TriangleDEF) : ℝ := sorry

/-- Theorem: The radius of the incircle is equal to the area divided by the semi-perimeter --/
theorem incircle_radius_eq_area_div_semiperimeter (t : TriangleDEF) :
  incircle_radius t = triangle_area t / semi_perimeter t := by sorry

end NUMINAMATH_CALUDE_incircle_radius_eq_area_div_semiperimeter_l217_21799


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l217_21735

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h_divisible : 37800 ∣ n^3) : 
  ∃ q : ℕ, q > 0 ∧ q ∣ n ∧ ∀ m : ℕ, m > 0 → m ∣ n → m ≤ q ∧ q = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l217_21735


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l217_21733

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 1 > 0} = {x : ℝ | x < -1/2 ∨ x > 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l217_21733


namespace NUMINAMATH_CALUDE_log_comparison_l217_21778

theorem log_comparison : 
  Real.log 6 / Real.log 3 > Real.log 10 / Real.log 5 ∧ 
  Real.log 10 / Real.log 5 > Real.log 14 / Real.log 7 := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l217_21778


namespace NUMINAMATH_CALUDE_tan_condition_l217_21761

open Real

theorem tan_condition (k : ℤ) (x : ℝ) : 
  (∃ k, x = 2 * k * π + π/4) → tan x = 1 ∧ 
  ∃ x, tan x = 1 ∧ ∀ k, x ≠ 2 * k * π + π/4 :=
by sorry

end NUMINAMATH_CALUDE_tan_condition_l217_21761


namespace NUMINAMATH_CALUDE_problem_statement_l217_21788

/-- Given two expressions A and B in terms of a and b, prove that A + 2B has a specific form
    and that when it's independent of b, a has a specific value. -/
theorem problem_statement (a b : ℝ) : 
  let A := 2*a^2 + 3*a*b - 2*b - 1
  let B := -a^2 - a*b + 1
  (A + 2*B = a*b - 2*b + 1) ∧ 
  (∀ b, A + 2*B = a*b - 2*b + 1 → a = 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l217_21788


namespace NUMINAMATH_CALUDE_monotonicity_when_a_is_neg_one_monotonicity_condition_on_interval_l217_21716

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Statement for part 1
theorem monotonicity_when_a_is_neg_one :
  let f₁ := f (-1)
  ∀ x y, x < y →
    (x ≤ 1/2 → y ≤ 1/2 → f₁ y ≤ f₁ x) ∧
    (1/2 ≤ x → 1/2 ≤ y → f₁ x ≤ f₁ y) :=
sorry

-- Statement for part 2
theorem monotonicity_condition_on_interval :
  ∀ a : ℝ, (∀ x y, -5 ≤ x → x < y → y ≤ 5 → 
    (f a x < f a y ∨ f a y < f a x)) ↔ 
    (a < -10 ∨ a > 10) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_when_a_is_neg_one_monotonicity_condition_on_interval_l217_21716


namespace NUMINAMATH_CALUDE_pond_filling_time_l217_21795

/-- Represents the time needed to fill a pond under specific conditions. -/
def time_to_fill_pond (initial_flow_ratio : ℚ) (initial_fill_ratio : ℚ) (initial_days : ℚ) : ℚ :=
  let total_volume := 18 * initial_flow_ratio * initial_days / initial_fill_ratio
  let remaining_volume := total_volume * (1 - initial_fill_ratio)
  remaining_volume / 1

theorem pond_filling_time :
  let initial_flow_ratio : ℚ := 3/4
  let initial_fill_ratio : ℚ := 2/3
  let initial_days : ℚ := 16
  time_to_fill_pond initial_flow_ratio initial_fill_ratio initial_days = 6 := by
  sorry

#eval time_to_fill_pond (3/4) (2/3) 16

end NUMINAMATH_CALUDE_pond_filling_time_l217_21795


namespace NUMINAMATH_CALUDE_unique_pair_for_n_l217_21762

theorem unique_pair_for_n (n : ℕ+) :
  ∃! (a b : ℕ+), n = (1/2) * ((a + b - 1) * (a + b - 2) : ℕ) + a := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_for_n_l217_21762


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l217_21749

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l217_21749


namespace NUMINAMATH_CALUDE_coplanar_iff_k_eq_neg_two_l217_21797

/-- Two lines in 3D space -/
structure Line3D where
  parameterization : ℝ → ℝ × ℝ × ℝ

/-- Checks if two lines are coplanar -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧
    ∀ (t s : ℝ),
      let (x1, y1, z1) := l1.parameterization s
      let (x2, y2, z2) := l2.parameterization t
      a * x1 + b * y1 + c * z1 + d =
      a * x2 + b * y2 + c * z2 + d

theorem coplanar_iff_k_eq_neg_two :
  ∀ (k : ℝ),
    let l1 : Line3D := ⟨λ s => (-1 + s, 3 - k*s, 1 + k*s)⟩
    let l2 : Line3D := ⟨λ t => (t/2, 1 + t, 2 - t)⟩
    are_coplanar l1 l2 ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_iff_k_eq_neg_two_l217_21797


namespace NUMINAMATH_CALUDE_triangle_area_l217_21784

/-- A triangle with sides of length 9, 40, and 41 has an area of 180. -/
theorem triangle_area (a b c : ℝ) (ha : a = 9) (hb : b = 40) (hc : c = 41) : 
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l217_21784


namespace NUMINAMATH_CALUDE_number_of_divisors_of_36_l217_21780

theorem number_of_divisors_of_36 : Finset.card (Nat.divisors 36) = 9 := by sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_36_l217_21780


namespace NUMINAMATH_CALUDE_grants_test_score_l217_21796

theorem grants_test_score (hunter_score john_score grant_score : ℕ) :
  hunter_score = 45 →
  john_score = 2 * hunter_score →
  grant_score = john_score + 10 →
  grant_score = 100 := by
sorry

end NUMINAMATH_CALUDE_grants_test_score_l217_21796


namespace NUMINAMATH_CALUDE_inequality_range_l217_21711

theorem inequality_range (a : ℝ) : 
  (∀ x y : ℝ, x^2 + 2*x + a ≥ -y^2 - 2*y) → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l217_21711


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l217_21758

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_and_inequality (h : Real.exp 4 > 54) :
  (∃ m : ℝ, ∀ x : ℝ, x > 0 → (2 * x + m = f x → m = -Real.exp 1)) ∧
  (∀ x : ℝ, x > 0 → -1 / Real.exp 1 ≤ f x ∧ f x < Real.exp x / (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l217_21758


namespace NUMINAMATH_CALUDE_main_age_is_46_l217_21782

/-- Represents the ages of four siblings -/
structure Ages where
  main : ℕ
  brother : ℕ
  sister : ℕ
  youngest : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def satisfiesConditions (ages : Ages) : Prop :=
  let futureAges := Ages.mk (ages.main + 10) (ages.brother + 10) (ages.sister + 10) (ages.youngest + 10)
  futureAges.main + futureAges.brother + futureAges.sister + futureAges.youngest = 88 ∧
  futureAges.main = 2 * futureAges.brother ∧
  futureAges.main = 3 * futureAges.sister ∧
  futureAges.main = 4 * futureAges.youngest ∧
  ages.brother = ages.sister + 3 ∧
  ages.sister = 2 * ages.youngest ∧
  ages.youngest = 4

theorem main_age_is_46 :
  ∃ (ages : Ages), satisfiesConditions ages ∧ ages.main = 46 := by
  sorry

end NUMINAMATH_CALUDE_main_age_is_46_l217_21782


namespace NUMINAMATH_CALUDE_intersection_dot_product_l217_21773

/-- Given a line and a parabola that intersect at points A and B, and a point M,
    prove that if the dot product of MA and MB is zero, then the y-coordinate of M is √2/2. -/
theorem intersection_dot_product (A B M : ℝ × ℝ) : 
  (∃ x y, y = 2 * Real.sqrt 2 * (x - 1) ∧ y^2 = 4 * x ∧ A = (x, y)) →  -- Line and parabola intersection for A
  (∃ x y, y = 2 * Real.sqrt 2 * (x - 1) ∧ y^2 = 4 * x ∧ B = (x, y)) →  -- Line and parabola intersection for B
  M.1 = -1 →  -- x-coordinate of M is -1
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0 →  -- Dot product of MA and MB is zero
  M.2 = Real.sqrt 2 / 2 := by  -- y-coordinate of M is √2/2
sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l217_21773


namespace NUMINAMATH_CALUDE_edda_magni_winning_strategy_l217_21710

/-- Represents the hexagonal board game with n tiles on each side. -/
structure HexGame where
  n : ℕ
  n_gt_two : n > 2

/-- Represents a winning strategy for Edda and Magni. -/
def winning_strategy (game : HexGame) : Prop :=
  ∃ k : ℕ, k > 0 ∧ game.n = 3 * k + 1

/-- Theorem stating the condition for Edda and Magni to have a winning strategy. -/
theorem edda_magni_winning_strategy (game : HexGame) :
  winning_strategy game ↔ ∃ k : ℕ, k > 0 ∧ game.n = 3 * k + 1 :=
by sorry


end NUMINAMATH_CALUDE_edda_magni_winning_strategy_l217_21710


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l217_21737

theorem smallest_solution_of_equation :
  let f (x : ℝ) := 1 / (x - 3) + 1 / (x - 5) - 3 / (x - 4)
  ∃ (s : ℝ), s = 4 - Real.sqrt 3 ∧
    f s = 0 ∧
    ∀ (x : ℝ), f x = 0 → x ≥ s :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l217_21737


namespace NUMINAMATH_CALUDE_triangle_properties_l217_21765

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 13 ∧ 
  t.a + t.c = 4 ∧
  Real.cos t.B / Real.cos t.C = -t.b / (2 * t.a + t.c)

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = (3/4) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l217_21765


namespace NUMINAMATH_CALUDE_brendan_remaining_money_l217_21772

/-- Brendan's remaining money calculation -/
theorem brendan_remaining_money 
  (june_earnings : ℕ) 
  (car_cost : ℕ) 
  (h1 : june_earnings = 5000)
  (h2 : car_cost = 1500) :
  (june_earnings / 2) - car_cost = 1000 :=
by sorry

end NUMINAMATH_CALUDE_brendan_remaining_money_l217_21772


namespace NUMINAMATH_CALUDE_median_on_hypotenuse_l217_21703

/-- Represents a right triangle with legs a and b, and median m on the hypotenuse -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  m : ℝ

/-- The median on the hypotenuse of a right triangle with legs 6 and 8 is 5 -/
theorem median_on_hypotenuse (t : RightTriangle) (h1 : t.a = 6) (h2 : t.b = 8) : t.m = 5 := by
  sorry

end NUMINAMATH_CALUDE_median_on_hypotenuse_l217_21703


namespace NUMINAMATH_CALUDE_quadrilateral_area_l217_21740

/-- The area of a quadrilateral with given diagonal and offsets -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 offset2 : ℝ) :
  diagonal = 10 → offset1 = 7 → offset2 = 3 →
  (diagonal * offset1 / 2) + (diagonal * offset2 / 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l217_21740


namespace NUMINAMATH_CALUDE_car_sale_profit_ratio_l217_21792

theorem car_sale_profit_ratio (c₁ c₂ : ℝ) (h : c₁ > 0 ∧ c₂ > 0) :
  (1.1 * c₁ + 0.9 * c₂ - (c₁ + c₂)) / (c₁ + c₂) = 0.01 →
  c₂ = (9 / 11) * c₁ := by
  sorry

end NUMINAMATH_CALUDE_car_sale_profit_ratio_l217_21792


namespace NUMINAMATH_CALUDE_union_covers_reals_iff_a_leq_neg_two_l217_21720

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < -2 ∨ x ≥ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- State the theorem
theorem union_covers_reals_iff_a_leq_neg_two (a : ℝ) :
  A ∪ B a = Set.univ ↔ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_iff_a_leq_neg_two_l217_21720


namespace NUMINAMATH_CALUDE_maurice_current_age_l217_21748

/-- Given Ron's current age and the relation between Ron and Maurice's ages after 5 years,
    prove Maurice's current age. -/
theorem maurice_current_age :
  ∀ (ron_current_age : ℕ) (maurice_current_age : ℕ),
    ron_current_age = 43 →
    ron_current_age + 5 = 4 * (maurice_current_age + 5) →
    maurice_current_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_maurice_current_age_l217_21748


namespace NUMINAMATH_CALUDE_average_speed_round_trip_budapest_debrecen_average_speed_l217_21712

/-- The average speed of a round trip between two cities, given the speeds for each direction. -/
theorem average_speed_round_trip (s : ℝ) (v1 v2 : ℝ) (h1 : v1 > 0) (h2 : v2 > 0) :
  let t1 := s / v1
  let t2 := s / v2
  let total_time := t1 + t2
  let total_distance := 2 * s
  total_distance / total_time = 2 * v1 * v2 / (v1 + v2) :=
by sorry

/-- The average speed of a car traveling between Budapest and Debrecen. -/
theorem budapest_debrecen_average_speed :
  let v1 := 56 -- km/h
  let v2 := 72 -- km/h
  let avg_speed := 2 * v1 * v2 / (v1 + v2)
  avg_speed = 63 :=
by sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_budapest_debrecen_average_speed_l217_21712


namespace NUMINAMATH_CALUDE_fraction_simplification_l217_21771

theorem fraction_simplification (a b m n : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hm : m ≠ 0) (hn : n ≠ 0) :
  (a^2 * b) / (m * n^2) / ((a * b) / (3 * m * n)) = 3 * a / n :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l217_21771


namespace NUMINAMATH_CALUDE_batsman_average_increase_l217_21764

def average_increase (total_innings : ℕ) (final_average : ℚ) (last_score : ℕ) : ℚ :=
  final_average - (total_innings * final_average - last_score) / (total_innings - 1)

theorem batsman_average_increase :
  average_increase 17 39 87 = 3 := by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l217_21764


namespace NUMINAMATH_CALUDE_max_rearrangeable_guests_correct_l217_21729

/-- Represents a hotel with rooms numbered from 101 to 200 --/
structure Hotel :=
  (rooms : Finset Nat)
  (room_capacity : Nat → Nat)
  (room_range : ∀ r ∈ rooms, 101 ≤ r ∧ r ≤ 200)
  (capacity_matches_number : ∀ r ∈ rooms, room_capacity r = r)

/-- The maximum number of guests that can always be rearranged --/
def max_rearrangeable_guests (h : Hotel) : Nat :=
  8824

/-- Theorem stating that max_rearrangeable_guests is correct --/
theorem max_rearrangeable_guests_correct (h : Hotel) :
  ∀ n : Nat, n ≤ max_rearrangeable_guests h →
  (∀ vacated : h.rooms, ∃ destination : h.rooms,
    vacated ≠ destination ∧
    h.room_capacity destination ≥ h.room_capacity vacated) :=
sorry

#check max_rearrangeable_guests_correct

end NUMINAMATH_CALUDE_max_rearrangeable_guests_correct_l217_21729


namespace NUMINAMATH_CALUDE_no_valid_x_for_mean_12_l217_21730

theorem no_valid_x_for_mean_12 : 
  ¬ ∃ x : ℝ, (3 + 117 + 915 + 138 + 1917 + 2114 + x) / 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_x_for_mean_12_l217_21730


namespace NUMINAMATH_CALUDE_rhombus_area_l217_21791

theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁^2 - 14*d₁ + 48 = 0 → 
  d₂^2 - 14*d₂ + 48 = 0 → 
  d₁ ≠ d₂ →
  (d₁ * d₂) / 2 = 24 := by
sorry

end NUMINAMATH_CALUDE_rhombus_area_l217_21791


namespace NUMINAMATH_CALUDE_unique_k_for_quadratic_equation_l217_21719

theorem unique_k_for_quadratic_equation : ∃! k : ℝ, k ≠ 0 ∧
  (∃! a : ℝ, a ≠ 0 ∧
    (∃! x : ℝ, x^2 - (a^3 + 1/a^3) * x + k = 0)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_k_for_quadratic_equation_l217_21719


namespace NUMINAMATH_CALUDE_volume_of_large_cube_l217_21779

/-- Given a cube with surface area 96 cm², prove that 8 such cubes form a larger cube with volume 512 cm³ -/
theorem volume_of_large_cube (small_cube : Real → Real → Real → Real) 
  (h1 : ∀ x, small_cube x x x = 96) -- surface area of small cube is 96
  (h2 : ∀ x y z, small_cube x y z = 6 * x * y) -- definition of surface area for a cube
  (large_cube : Real → Real → Real → Real)
  (h3 : ∀ x, large_cube x x x = 8 * small_cube (x/2) (x/2) (x/2)) -- large cube is made of 8 small cubes
  : ∃ x, large_cube x x x = 512 :=
sorry

end NUMINAMATH_CALUDE_volume_of_large_cube_l217_21779


namespace NUMINAMATH_CALUDE_henry_walking_distance_l217_21757

/-- Given a constant walking rate and duration, calculate the distance walked. -/
def distance_walked (rate : ℝ) (time : ℝ) : ℝ :=
  rate * time

/-- Theorem: Henry walks 8 miles in 2 hours at a rate of 4 miles per hour. -/
theorem henry_walking_distance :
  let rate : ℝ := 4  -- miles per hour
  let time : ℝ := 2  -- hours
  distance_walked rate time = 8 := by
  sorry

end NUMINAMATH_CALUDE_henry_walking_distance_l217_21757


namespace NUMINAMATH_CALUDE_inequality_proof_l217_21769

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l217_21769


namespace NUMINAMATH_CALUDE_coin_count_theorem_l217_21724

theorem coin_count_theorem (quarters_piles : Nat) (quarters_per_pile : Nat)
                           (dimes_piles : Nat) (dimes_per_pile : Nat)
                           (nickels_piles : Nat) (nickels_per_pile : Nat)
                           (pennies_piles : Nat) (pennies_per_pile : Nat) :
  quarters_piles = 7 →
  quarters_per_pile = 4 →
  dimes_piles = 4 →
  dimes_per_pile = 2 →
  nickels_piles = 6 →
  nickels_per_pile = 5 →
  pennies_piles = 3 →
  pennies_per_pile = 8 →
  quarters_piles * quarters_per_pile +
  dimes_piles * dimes_per_pile +
  nickels_piles * nickels_per_pile +
  pennies_piles * pennies_per_pile = 90 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_theorem_l217_21724


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l217_21793

/-- The time taken for a train to pass a platform -/
theorem train_platform_passing_time 
  (train_length : ℝ) 
  (time_to_cross_point : ℝ) 
  (platform_length : ℝ) 
  (train_length_positive : 0 < train_length)
  (time_to_cross_point_positive : 0 < time_to_cross_point)
  (platform_length_positive : 0 < platform_length)
  (h1 : train_length = 1200)
  (h2 : time_to_cross_point = 120)
  (h3 : platform_length = 1200) : 
  (train_length + platform_length) / (train_length / time_to_cross_point) = 240 := by
sorry


end NUMINAMATH_CALUDE_train_platform_passing_time_l217_21793


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l217_21725

/-- In a cyclic quadrilateral ABCD, if angle BAC = d°, angle BCD = 43°, angle ACD = 59°, and angle BAD = 36°, then d = 42°. -/
theorem cyclic_quadrilateral_angle (d : ℝ) : 
  d + 43 + 59 + 36 = 180 → d = 42 := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l217_21725


namespace NUMINAMATH_CALUDE_inequality_proof_l217_21706

theorem inequality_proof (k m a b : ℝ) 
  (h : ∀ x ∈ Set.Icc a b, |x^2 - k*x - m| ≤ 1) : 
  b - a ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l217_21706


namespace NUMINAMATH_CALUDE_cinnamon_blend_probability_l217_21767

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p ^ k * (1 - p) ^ (n - k)

/-- The probability of exactly 5 successes in 7 trials with 3/4 probability of success in each trial is 5103/16384. -/
theorem cinnamon_blend_probability : 
  binomial_probability 7 5 (3/4) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_blend_probability_l217_21767


namespace NUMINAMATH_CALUDE_harold_adrienne_speed_difference_l217_21728

/-- Prove that Harold walks 1 mile per hour faster than Adrienne --/
theorem harold_adrienne_speed_difference :
  ∀ (total_distance : ℝ) (adrienne_speed : ℝ) (harold_catch_up_distance : ℝ),
    total_distance = 60 →
    adrienne_speed = 3 →
    harold_catch_up_distance = 12 →
    ∃ (harold_speed : ℝ),
      harold_speed > adrienne_speed ∧
      harold_speed - adrienne_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_harold_adrienne_speed_difference_l217_21728


namespace NUMINAMATH_CALUDE_integral_proof_l217_21752

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log (abs (x - 2)) - 1 / (2 * (x - 1)^2)

theorem integral_proof (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  deriv f x = (2 * x^3 - 6 * x^2 + 7 * x - 4) / ((x - 2) * (x - 1)^3) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l217_21752


namespace NUMINAMATH_CALUDE_bob_weight_l217_21741

theorem bob_weight (j b : ℝ) 
  (h1 : j + b = 210)
  (h2 : b - j = b / 3)
  : b = 126 := by
  sorry

end NUMINAMATH_CALUDE_bob_weight_l217_21741


namespace NUMINAMATH_CALUDE_complement_intersection_MN_l217_21766

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3}

theorem complement_intersection_MN :
  (M ∩ N)ᶜ = {1, 4} :=
sorry

end NUMINAMATH_CALUDE_complement_intersection_MN_l217_21766


namespace NUMINAMATH_CALUDE_fib_gcd_consecutive_fib_gcd_identity_fib_sum_identity_l217_21713

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_gcd_consecutive (n : ℕ) : Nat.gcd (fib n) (fib (n + 1)) = 1 := by sorry

theorem fib_gcd_identity (m n : ℕ) : 
  fib (Nat.gcd m n) = Nat.gcd (fib m) (fib n) := by sorry

theorem fib_sum_identity (m n : ℕ) :
  fib (n + m) = fib m * fib (n + 1) + fib (m - 1) * fib n := by sorry

end NUMINAMATH_CALUDE_fib_gcd_consecutive_fib_gcd_identity_fib_sum_identity_l217_21713


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_specific_geometric_series_sum_l217_21774

/-- The sum of an infinite geometric series with first term a and common ratio r is a / (1 - r),
    given that |r| < 1 -/
theorem infinite_geometric_series_sum (a r : ℚ) (h : |r| < 1) :
  (∑' n, a * r^n) = a / (1 - r) :=
sorry

/-- The sum of the specific infinite geometric series is 10/9 -/
theorem specific_geometric_series_sum :
  let a : ℚ := 5/3
  let r : ℚ := -1/2
  (∑' n, a * r^n) = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_specific_geometric_series_sum_l217_21774


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l217_21753

theorem expression_simplification_and_evaluation :
  let m : ℚ := 2
  let expr := (2 / (m - 3) + 1) / ((2 * m - 2) / (m^2 - 6 * m + 9))
  expr = -1/2 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l217_21753


namespace NUMINAMATH_CALUDE_johns_money_left_l217_21705

/-- Calculates the money John has left after walking his neighbor's dog, buying books, and giving money to his sister. -/
theorem johns_money_left (days_in_april : ℕ) (sundays_in_april : ℕ) (daily_pay : ℕ) (book_cost : ℕ) (sister_money : ℕ) : 
  days_in_april = 30 →
  sundays_in_april = 4 →
  daily_pay = 10 →
  book_cost = 50 →
  sister_money = 50 →
  (days_in_april - sundays_in_april) * daily_pay - (book_cost + sister_money) = 160 :=
by sorry

end NUMINAMATH_CALUDE_johns_money_left_l217_21705


namespace NUMINAMATH_CALUDE_checkerboard_probability_l217_21763

/-- The size of one side of the checkerboard -/
def board_size : ℕ := 8

/-- The total number of squares on the checkerboard -/
def total_squares : ℕ := board_size * board_size

/-- The number of squares on the perimeter of the checkerboard -/
def perimeter_squares : ℕ := 4 * (board_size - 1)

/-- The number of squares not touching the outer edge -/
def inner_squares : ℕ := total_squares - perimeter_squares

/-- The probability of choosing a square not touching the outer edge -/
def inner_square_probability : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  inner_square_probability = 9 / 16 := by sorry

end NUMINAMATH_CALUDE_checkerboard_probability_l217_21763


namespace NUMINAMATH_CALUDE_expression_value_l217_21759

theorem expression_value :
  let a : ℤ := 10
  let b : ℤ := 15
  let c : ℤ := 3
  let d : ℤ := 2
  (a * (b - c)) - ((a - b) * c) + d = 137 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l217_21759


namespace NUMINAMATH_CALUDE_min_acute_triangles_in_square_l217_21734

/-- A triangulation of a square. -/
structure SquareTriangulation where
  /-- The number of triangles in the triangulation. -/
  num_triangles : ℕ
  /-- All triangles in the triangulation are acute-angled. -/
  all_acute : Bool
  /-- The triangulation is valid (covers the entire square without overlaps). -/
  valid : Bool

/-- The minimum number of triangles in a valid acute-angled triangulation of a square. -/
def min_acute_triangulation : ℕ := 8

/-- Theorem: The minimum number of acute-angled triangles that a square can be divided into is 8. -/
theorem min_acute_triangles_in_square :
  ∀ t : SquareTriangulation, t.valid ∧ t.all_acute → t.num_triangles ≥ min_acute_triangulation :=
by sorry

end NUMINAMATH_CALUDE_min_acute_triangles_in_square_l217_21734


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_l217_21760

theorem sum_of_x_solutions (x₁ x₂ : ℝ) (y : ℝ) (h1 : y = 5) (h2 : x₁^2 + y^2 = 169) (h3 : x₂^2 + y^2 = 169) (h4 : x₁ ≠ x₂) : x₁ + x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_l217_21760


namespace NUMINAMATH_CALUDE_city_population_problem_l217_21742

theorem city_population_problem (p : ℝ) : 
  (0.85 * (p + 1500) = p - 45) → p = 8800 := by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l217_21742


namespace NUMINAMATH_CALUDE_manhattan_to_bronx_travel_time_l217_21747

/-- The total travel time from Manhattan to the Bronx -/
def total_travel_time (subway_time train_time bike_time : ℕ) : ℕ :=
  subway_time + train_time + bike_time

/-- Theorem stating that the total travel time is 38 hours -/
theorem manhattan_to_bronx_travel_time :
  ∃ (subway_time train_time bike_time : ℕ),
    subway_time = 10 ∧
    train_time = 2 * subway_time ∧
    bike_time = 8 ∧
    total_travel_time subway_time train_time bike_time = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_manhattan_to_bronx_travel_time_l217_21747


namespace NUMINAMATH_CALUDE_parrot_phrases_l217_21717

def phrases_learned (days : ℕ) (phrases_per_week : ℕ) (initial_phrases : ℕ) : ℕ :=
  initial_phrases + (days / 7) * phrases_per_week

theorem parrot_phrases :
  phrases_learned 49 2 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_parrot_phrases_l217_21717


namespace NUMINAMATH_CALUDE_geometric_progression_and_sum_l217_21700

theorem geometric_progression_and_sum : ∃ x : ℝ,
  let a₁ := 10 + x
  let a₂ := 30 + x
  let a₃ := 60 + x
  (a₂ / a₁ = a₃ / a₂) ∧ (a₁ + a₂ + a₃ = 190) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_and_sum_l217_21700


namespace NUMINAMATH_CALUDE_medal_winners_combinations_l217_21732

theorem medal_winners_combinations (semifinalists : ℕ) (advance : ℕ) (finalists : ℕ) (medals : ℕ) :
  semifinalists = 8 →
  advance = semifinalists - 2 →
  finalists = advance →
  medals = 3 →
  Nat.choose finalists medals = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_medal_winners_combinations_l217_21732


namespace NUMINAMATH_CALUDE_lineup_combinations_l217_21739

/-- The number of ways to choose a starting lineup -/
def choose_lineup (total_players : ℕ) (offensive_linemen : ℕ) (kickers : ℕ) : ℕ :=
  offensive_linemen * kickers * (total_players - 2) * (total_players - 3) * (total_players - 4)

/-- Theorem stating the number of ways to choose the lineup -/
theorem lineup_combinations :
  choose_lineup 12 4 2 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l217_21739


namespace NUMINAMATH_CALUDE_not_all_odd_divisible_by_3_l217_21701

theorem not_all_odd_divisible_by_3 : ¬ (∀ n : ℕ, Odd n → 3 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_not_all_odd_divisible_by_3_l217_21701


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l217_21770

theorem rectangle_area_theorem (m : ℕ) (hm : m > 12) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧
  (x * y > m) ∧
  ((x - 1) * y < m) ∧
  (x * (y - 1) < m) ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → a * b ≠ m) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l217_21770


namespace NUMINAMATH_CALUDE_division_multiplication_chain_l217_21744

theorem division_multiplication_chain : (180 / 6) * 3 / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_chain_l217_21744


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l217_21743

theorem arithmetic_calculations : 
  (26 - 7 + (-6) + 17 = 30) ∧ 
  (-81 / (9/4) * (-4/9) / (-16) = -1) ∧ 
  ((2/3 - 3/4 + 1/6) * (-36) = -3) ∧ 
  (-1^4 + 12 / (-2)^2 + 1/4 * (-8) = 0) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l217_21743


namespace NUMINAMATH_CALUDE_ratio_problem_l217_21707

theorem ratio_problem (x y : ℕ) (h1 : x + y = 420) (h2 : x = 180) :
  ∃ (a b : ℕ), a = 3 ∧ b = 4 ∧ x * b = y * a :=
sorry

end NUMINAMATH_CALUDE_ratio_problem_l217_21707


namespace NUMINAMATH_CALUDE_not_all_greater_than_quarter_l217_21776

theorem not_all_greater_than_quarter (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬(((1 - a) * b > 1/4) ∧ ((1 - b) * c > 1/4) ∧ ((1 - c) * a > 1/4)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_quarter_l217_21776
