import Mathlib

namespace NUMINAMATH_CALUDE_supermarket_queue_clearing_time_l3352_335276

/-- The average number of people lining up to pay per hour -/
def average_customers_per_hour : ℝ := 60

/-- The number of people a single cashier can handle per hour -/
def cashier_capacity_per_hour : ℝ := 80

/-- The number of hours it takes for one cashier to clear the line -/
def hours_for_one_cashier : ℝ := 4

/-- The number of cashiers working in the second scenario -/
def num_cashiers : ℕ := 2

/-- The time it takes for two cashiers to clear the line -/
def time_for_two_cashiers : ℝ := 0.8

theorem supermarket_queue_clearing_time :
  2 * cashier_capacity_per_hour * time_for_two_cashiers = 
  average_customers_per_hour * time_for_two_cashiers + 
  (cashier_capacity_per_hour * hours_for_one_cashier - average_customers_per_hour * hours_for_one_cashier) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_queue_clearing_time_l3352_335276


namespace NUMINAMATH_CALUDE_number_relationship_l3352_335247

theorem number_relationship (A B C : ℕ) : 
  A + B + C = 660 → A = 2 * B → B = 180 → C = A - 240 := by sorry

end NUMINAMATH_CALUDE_number_relationship_l3352_335247


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3352_335262

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -3 < x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3352_335262


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3352_335271

theorem arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) :
  a₁ = 3 → d = 3 → n = 10 →
  (n : ℝ) / 2 * (a₁ + (a₁ + (n - 1) * d)) = 165 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3352_335271


namespace NUMINAMATH_CALUDE_sum_of_specific_S_l3352_335268

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -n else n + 1

theorem sum_of_specific_S : S 18 + S 34 + S 51 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_l3352_335268


namespace NUMINAMATH_CALUDE_equation_solution_l3352_335239

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (15 - x))) = 55 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3352_335239


namespace NUMINAMATH_CALUDE_unique_number_with_remainders_and_quotient_condition_l3352_335241

theorem unique_number_with_remainders_and_quotient_condition :
  ∃! (n : ℕ),
    n > 0 ∧
    n % 7 = 2 ∧
    n % 8 = 4 ∧
    (n - 2) / 7 = (n - 4) / 8 + 7 ∧
    n = 380 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_remainders_and_quotient_condition_l3352_335241


namespace NUMINAMATH_CALUDE_min_value_expression_l3352_335257

theorem min_value_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧
  ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3352_335257


namespace NUMINAMATH_CALUDE_inscribed_circle_path_length_l3352_335232

theorem inscribed_circle_path_length (a b c : ℝ) (h_triangle : a = 10 ∧ b = 8 ∧ c = 12) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := area / s
  (a + b + c) - 2 * r = 15 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_path_length_l3352_335232


namespace NUMINAMATH_CALUDE_relationship_abc_l3352_335235

open Real

theorem relationship_abc (a b c : ℝ) (ha : a = 2^(log 2)) (hb : b = 2 + 2*log 2) (hc : c = (log 2)^2) :
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l3352_335235


namespace NUMINAMATH_CALUDE_min_of_quadratic_l3352_335201

/-- The quadratic function f(x) = x^2 - 2px + 4q -/
def f (p q x : ℝ) : ℝ := x^2 - 2*p*x + 4*q

/-- Theorem stating that the minimum of f occurs at x = p -/
theorem min_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f p q x_min ≤ f p q x ∧ x_min = p :=
sorry

end NUMINAMATH_CALUDE_min_of_quadratic_l3352_335201


namespace NUMINAMATH_CALUDE_alternate_interior_angles_relationship_l3352_335251

-- Define a structure for a line
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for an angle
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to create alternate interior angles
def alternate_interior_angles (l1 l2 l3 : Line) : (Angle × Angle) :=
  sorry

-- Theorem statement
theorem alternate_interior_angles_relationship 
  (l1 l2 l3 : Line) : 
  ¬ (∀ (a1 a2 : Angle), 
    (a1, a2) = alternate_interior_angles l1 l2 l3 → 
    a1.measure = a2.measure ∨ 
    a1.measure ≠ a2.measure) :=
sorry

end NUMINAMATH_CALUDE_alternate_interior_angles_relationship_l3352_335251


namespace NUMINAMATH_CALUDE_equation_transformation_l3352_335207

theorem equation_transformation (x y : ℝ) (hx : x ≠ 0) :
  y = x + 1/x →
  (x^4 + x^3 - 5*x^2 + x + 1 = 0) ↔ (x^2*(y^2 + y - 7) = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3352_335207


namespace NUMINAMATH_CALUDE_lcm_4_8_9_10_l3352_335202

theorem lcm_4_8_9_10 : Nat.lcm 4 (Nat.lcm 8 (Nat.lcm 9 10)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_4_8_9_10_l3352_335202


namespace NUMINAMATH_CALUDE_marbles_lost_calculation_l3352_335245

def initial_marbles : ℕ := 15
def marbles_found : ℕ := 9
def extra_marbles_lost : ℕ := 14

theorem marbles_lost_calculation :
  marbles_found + extra_marbles_lost = 23 :=
by sorry

end NUMINAMATH_CALUDE_marbles_lost_calculation_l3352_335245


namespace NUMINAMATH_CALUDE_triangle_side_length_l3352_335299

theorem triangle_side_length (a c area : ℝ) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
    (h_a : a = 4) (h_c : c = 6) (h_area : area = 6 * Real.sqrt 3) : 
    ∃ (b : ℝ), b^2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3352_335299


namespace NUMINAMATH_CALUDE_largest_integer_a_l3352_335244

theorem largest_integer_a : ∃ (a : ℤ), 
  (∀ x : ℝ, -π/2 < x ∧ x < π/2 → 
    a^2 - 15*a - (Real.tan x - 1)*(Real.tan x + 2)*(Real.tan x + 5)*(Real.tan x + 8) < 35) ∧ 
  (∀ b : ℤ, b > a → 
    ∃ x : ℝ, -π/2 < x ∧ x < π/2 ∧ 
      b^2 - 15*b - (Real.tan x - 1)*(Real.tan x + 2)*(Real.tan x + 5)*(Real.tan x + 8) ≥ 35) ∧
  a = 10 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_a_l3352_335244


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2145_l3352_335242

theorem smallest_prime_factor_of_2145 : Nat.minFac 2145 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2145_l3352_335242


namespace NUMINAMATH_CALUDE_nancy_insurance_percentage_l3352_335267

/-- Given a monthly insurance cost and an annual payment, 
    calculate the percentage of the total cost being paid. -/
def insurance_percentage (monthly_cost : ℚ) (annual_payment : ℚ) : ℚ :=
  (annual_payment / (monthly_cost * 12)) * 100

/-- Theorem stating that for a monthly cost of $80 and an annual payment of $384,
    the percentage paid is 40% of the total cost. -/
theorem nancy_insurance_percentage :
  insurance_percentage 80 384 = 40 := by
  sorry

end NUMINAMATH_CALUDE_nancy_insurance_percentage_l3352_335267


namespace NUMINAMATH_CALUDE_equation_equality_l3352_335203

theorem equation_equality (x : ℝ) : -x^3 + 7*x^2 + 2*x - 8 = -(x - 2)*(x - 4)*(x - 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l3352_335203


namespace NUMINAMATH_CALUDE_ellipse_condition_l3352_335243

def is_ellipse_equation (m : ℝ) : Prop :=
  (m - 2 > 0) ∧ (6 - m > 0) ∧ (m - 2 ≠ 6 - m)

theorem ellipse_condition (m : ℝ) :
  (is_ellipse_equation m → m ∈ Set.Ioo 2 6) ∧
  (∃ m ∈ Set.Ioo 2 6, ¬is_ellipse_equation m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l3352_335243


namespace NUMINAMATH_CALUDE_largest_square_advertisement_l3352_335272

theorem largest_square_advertisement (rectangle_width rectangle_length min_border : Real)
  (h1 : rectangle_width = 9)
  (h2 : rectangle_length = 16)
  (h3 : min_border = 1.5)
  (h4 : rectangle_width ≤ rectangle_length) :
  let max_side := min (rectangle_width - 2 * min_border) (rectangle_length - 2 * min_border)
  (max_side * max_side) = 36 := by
  sorry

#check largest_square_advertisement

end NUMINAMATH_CALUDE_largest_square_advertisement_l3352_335272


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3352_335296

def A : Set ℝ := {x | (x - 2) / (x + 3) ≤ 0}
def B : Set ℝ := {x | x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | -3 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3352_335296


namespace NUMINAMATH_CALUDE_right_triangle_condition_l3352_335270

/-- If in a triangle ABC, angle A equals the sum of angles B and C, then angle A is a right angle -/
theorem right_triangle_condition (A B C : Real) (h1 : A + B + C = Real.pi) (h2 : A = B + C) : A = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l3352_335270


namespace NUMINAMATH_CALUDE_megan_folders_l3352_335205

/-- Calculates the number of full folders given the initial number of files, 
    number of deleted files, and number of files per folder. -/
def fullFolders (initialFiles : ℕ) (deletedFiles : ℕ) (filesPerFolder : ℕ) : ℕ :=
  ((initialFiles - deletedFiles) / filesPerFolder : ℕ)

/-- Proves that Megan ends up with 15 full folders given the initial conditions. -/
theorem megan_folders : fullFolders 256 67 12 = 15 := by
  sorry

#eval fullFolders 256 67 12

end NUMINAMATH_CALUDE_megan_folders_l3352_335205


namespace NUMINAMATH_CALUDE_equation_solutions_l3352_335274

theorem equation_solutions : 
  {x : ℝ | (2 + x)^(2/3) + 3 * (2 - x)^(2/3) = 4 * (4 - x^2)^(1/3)} = {0, 13/7} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3352_335274


namespace NUMINAMATH_CALUDE_function_extrema_implies_a_range_l3352_335249

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

theorem function_extrema_implies_a_range (a : ℝ) :
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_implies_a_range_l3352_335249


namespace NUMINAMATH_CALUDE_uncool_parents_in_two_classes_l3352_335269

/-- Represents a math class with information about cool parents -/
structure MathClass where
  total_students : ℕ
  cool_dads : ℕ
  cool_moms : ℕ
  both_cool : ℕ

/-- Calculates the number of students with uncool parents in a class -/
def uncool_parents (c : MathClass) : ℕ :=
  c.total_students - (c.cool_dads + c.cool_moms - c.both_cool)

/-- The problem statement -/
theorem uncool_parents_in_two_classes 
  (class1 : MathClass)
  (class2 : MathClass)
  (h1 : class1.total_students = 45)
  (h2 : class1.cool_dads = 22)
  (h3 : class1.cool_moms = 25)
  (h4 : class1.both_cool = 11)
  (h5 : class2.total_students = 35)
  (h6 : class2.cool_dads = 15)
  (h7 : class2.cool_moms = 18)
  (h8 : class2.both_cool = 7) :
  uncool_parents class1 + uncool_parents class2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_uncool_parents_in_two_classes_l3352_335269


namespace NUMINAMATH_CALUDE_total_hangers_count_l3352_335206

def pink_hangers : ℕ := 7
def green_hangers : ℕ := 4
def blue_hangers : ℕ := green_hangers - 1
def yellow_hangers : ℕ := blue_hangers - 1
def orange_hangers : ℕ := 2 * pink_hangers
def purple_hangers : ℕ := yellow_hangers + 3
def red_hangers : ℕ := purple_hangers / 2

theorem total_hangers_count :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers +
  orange_hangers + purple_hangers + red_hangers = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_hangers_count_l3352_335206


namespace NUMINAMATH_CALUDE_inverse_sum_mod_31_l3352_335217

theorem inverse_sum_mod_31 :
  ∃ (a b : ℤ), a ≡ 25 [ZMOD 31] ∧ b ≡ 5 [ZMOD 31] ∧ (a + b) ≡ 30 [ZMOD 31] := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_31_l3352_335217


namespace NUMINAMATH_CALUDE_max_value_theorem_l3352_335265

theorem max_value_theorem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_constraint : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 243/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3352_335265


namespace NUMINAMATH_CALUDE_min_abs_phi_l3352_335250

/-- Given a function y = 3cos(2x + φ) with its graph symmetric about (2π/3, 0),
    the minimum value of |φ| is π/6 -/
theorem min_abs_phi (φ : ℝ) : 
  (∀ x, 3 * Real.cos (2 * x + φ) = 3 * Real.cos (2 * (4 * π / 3 - x) + φ)) →
  (∃ k : ℤ, φ = k * π - 5 * π / 6) →
  π / 6 ≤ |φ| ∧ (∃ φ₀, |φ₀| = π / 6 ∧ 
    (∀ x, 3 * Real.cos (2 * x + φ₀) = 3 * Real.cos (2 * (4 * π / 3 - x) + φ₀))) :=
by sorry

end NUMINAMATH_CALUDE_min_abs_phi_l3352_335250


namespace NUMINAMATH_CALUDE_closest_integer_to_ratio_l3352_335223

/-- Given two positive real numbers a and b where a > b, and their arithmetic mean
    is equal to twice their geometric mean, prove that the integer closest to a/b is 14. -/
theorem closest_integer_to_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : (a + b) / 2 = 2 * Real.sqrt (a * b)) : 
    ∃ (n : ℤ), n = 14 ∧ ∀ (m : ℤ), |a / b - ↑n| ≤ |a / b - ↑m| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_ratio_l3352_335223


namespace NUMINAMATH_CALUDE_work_ratio_proof_l3352_335288

/-- Represents the work rate of a single cat -/
def single_cat_rate : ℝ := 1

/-- Represents the total work to be done -/
def total_work : ℝ := 10

/-- Represents the number of days the initial cats work -/
def initial_days : ℕ := 5

/-- Represents the total number of days to complete the work -/
def total_days : ℕ := 7

/-- Represents the initial number of cats -/
def initial_cats : ℕ := 2

/-- Represents the final number of cats -/
def final_cats : ℕ := 5

theorem work_ratio_proof :
  let initial_work := (initial_cats : ℝ) * single_cat_rate * initial_days
  let remaining_days := total_days - initial_days
  let remaining_work := (final_cats : ℝ) * single_cat_rate * remaining_days
  initial_work / (initial_work + remaining_work) = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_work_ratio_proof_l3352_335288


namespace NUMINAMATH_CALUDE_tomato_planting_theorem_l3352_335261

def tomato_planting (total_seedlings : ℕ) (remi_day1 : ℕ) : Prop :=
  let remi_day2 := 2 * remi_day1
  let father_day3 := 3 * remi_day2
  let father_day4 := 4 * remi_day2
  let sister_day5 := remi_day1
  let sister_day6 := 5 * remi_day1
  let remi_total := remi_day1 + remi_day2
  let sister_total := sister_day5 + sister_day6
  let father_total := total_seedlings - remi_total - sister_total
  (remi_total = 600) ∧
  (sister_total = 1200) ∧
  (father_total = 6400) ∧
  (remi_total + sister_total + father_total = total_seedlings)

theorem tomato_planting_theorem :
  tomato_planting 8200 200 :=
by
  sorry

end NUMINAMATH_CALUDE_tomato_planting_theorem_l3352_335261


namespace NUMINAMATH_CALUDE_max_d_value_l3352_335284

def a (n : ℕ+) : ℕ := 120 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ (n : ℕ+), d n = 481) ∧ (∀ (n : ℕ+), d n ≤ 481) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l3352_335284


namespace NUMINAMATH_CALUDE_area_inside_circle_outside_square_l3352_335234

/-- The area inside a circle of radius √3/3 but outside a square of side length 1, 
    when they share the same center. -/
theorem area_inside_circle_outside_square : 
  let square_side : ℝ := 1
  let circle_radius : ℝ := Real.sqrt 3 / 3
  let circle_area : ℝ := π * circle_radius^2
  let square_area : ℝ := square_side^2
  let area_difference : ℝ := circle_area - square_area
  area_difference = 2 * π / 9 - Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_inside_circle_outside_square_l3352_335234


namespace NUMINAMATH_CALUDE_triangle_inequality_and_equality_l3352_335236

/-- Triangle ABC with side lengths a, b, c opposite to vertices A, B, C respectively,
    and h being the height from vertex C onto side AB -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_h : 0 < h
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Main theorem about the inequality and equality condition -/
theorem triangle_inequality_and_equality (t : Triangle) :
  t.a + t.b ≥ Real.sqrt (t.c^2 + 4*t.h^2) ∧
  (t.a + t.b = Real.sqrt (t.c^2 + 4*t.h^2) ↔ t.a = t.b ∧ t.a^2 + t.b^2 = t.c^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_equality_l3352_335236


namespace NUMINAMATH_CALUDE_ratio_b_to_c_l3352_335222

theorem ratio_b_to_c (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : c / d = 1 / 2)
  (h3 : d / e = 3)
  (h4 : e / f = 1 / 10)
  (h5 : a * b * c / (d * e * f) = 0.15) :
  b / c = 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_b_to_c_l3352_335222


namespace NUMINAMATH_CALUDE_function_composition_identity_l3352_335258

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x + b
  else if x < 3 then 2 * x - 1
  else 10 - 4 * x

theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_identity_l3352_335258


namespace NUMINAMATH_CALUDE_calories_per_bar_is_48_l3352_335211

-- Define the total number of calories
def total_calories : ℕ := 2016

-- Define the number of candy bars
def num_candy_bars : ℕ := 42

-- Define the function to calculate calories per candy bar
def calories_per_bar : ℚ := total_calories / num_candy_bars

-- Theorem to prove
theorem calories_per_bar_is_48 : calories_per_bar = 48 := by
  sorry

end NUMINAMATH_CALUDE_calories_per_bar_is_48_l3352_335211


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3352_335293

theorem greatest_divisor_with_remainders : Nat.gcd (28572 - 142) (39758 - 84) = 2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3352_335293


namespace NUMINAMATH_CALUDE_ordering_of_expressions_l3352_335213

theorem ordering_of_expressions : e^(0.11 : ℝ) > (1.1 : ℝ)^(1.1 : ℝ) ∧ (1.1 : ℝ)^(1.1 : ℝ) > 1.11 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_expressions_l3352_335213


namespace NUMINAMATH_CALUDE_rabbit_hit_probability_l3352_335248

/-- The probability that at least one hunter hits the rabbit. -/
def probability_hit (p1 p2 p3 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

/-- Theorem: Given three hunters with hit probabilities 0.6, 0.5, and 0.4,
    the probability that the rabbit is hit is 0.88. -/
theorem rabbit_hit_probability :
  probability_hit 0.6 0.5 0.4 = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_hit_probability_l3352_335248


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l3352_335214

theorem largest_inscribed_triangle_area (r : ℝ) (h : r = 10) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_height := r
  let triangle_area := (1/2) * diameter * max_height
  triangle_area = 100 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l3352_335214


namespace NUMINAMATH_CALUDE_final_basketball_count_l3352_335289

def initial_count : ℕ := 100

def transactions : List ℤ := [38, -42, 27, -33, -40]

theorem final_basketball_count : 
  initial_count + transactions.sum = 50 := by sorry

end NUMINAMATH_CALUDE_final_basketball_count_l3352_335289


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l3352_335277

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + a*x + (b - 2)

-- State the theorem
theorem range_of_a_minus_b (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < -1 ∧ -1 < x₂ ∧ x₂ < 0 ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) →
  ∀ y : ℝ, y > -1 → ∃ a' b' : ℝ, a' - b' = y ∧
    ∃ x₁ x₂ : ℝ, x₁ < -1 ∧ -1 < x₂ ∧ x₂ < 0 ∧ f a' b' x₁ = 0 ∧ f a' b' x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l3352_335277


namespace NUMINAMATH_CALUDE_age_difference_l3352_335281

/-- The problem of finding the age difference between A and B -/
theorem age_difference (a b : ℕ) : b = 36 → a + 10 = 2 * (b - 10) → a - b = 6 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3352_335281


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l3352_335285

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (l w h : ℝ) (cube_sa : ℝ) : 
  l = 12 → w = 3 → h = 24 → 
  cube_sa = 6 * (l * w * h) ^ (2/3) →
  cube_sa = 545.02 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l3352_335285


namespace NUMINAMATH_CALUDE_cricket_team_size_l3352_335204

/-- Represents the number of players on a cricket team -/
def total_players : ℕ := 55

/-- Represents the number of throwers on the team -/
def throwers : ℕ := 37

/-- Represents the number of right-handed players on the team -/
def right_handed : ℕ := 49

/-- Theorem stating the total number of players on the cricket team -/
theorem cricket_team_size :
  total_players = throwers + (right_handed - throwers) * 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_size_l3352_335204


namespace NUMINAMATH_CALUDE_factorization_analysis_l3352_335246

theorem factorization_analysis (x y a b : ℝ) :
  (x^4 - y^4 = (x^2 + y^2) * (x + y) * (x - y)) ∧
  (x^3*y - 2*x^2*y^2 + x*y^3 = x*y*(x - y)^2) ∧
  (4*x^2 - 4*x + 1 = (2*x - 1)^2) ∧
  (4*(a - b)^2 + 1 + 4*(a - b) = (2*a - 2*b + 1)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_analysis_l3352_335246


namespace NUMINAMATH_CALUDE_trapezoid_area_l3352_335252

theorem trapezoid_area (outer_area inner_area : ℝ) (h1 : outer_area = 36) (h2 : inner_area = 4) :
  (outer_area - inner_area) / 3 = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3352_335252


namespace NUMINAMATH_CALUDE_intersection_A_B_l3352_335240

-- Define set A
def A : Set ℝ := {x | x - 1 < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 0 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3352_335240


namespace NUMINAMATH_CALUDE_unique_number_l3352_335298

theorem unique_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n % 13 = 0 ∧ 
  n > 26 ∧ 
  n % 7 = 0 ∧ 
  n % 10 ≠ 6 ∧ 
  n % 10 ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_l3352_335298


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3352_335230

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set Nat := {1, 3, 5, 7}
def N : Set Nat := {5, 6, 7}

theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {2, 4, 8} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3352_335230


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3352_335237

theorem simplify_and_rationalize (x : ℝ) (h : x = 1 / (1 + 1 / (Real.sqrt 2 + 2))) :
  x = (4 + Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3352_335237


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3352_335224

theorem sum_of_squares_of_roots : ∃ (a b c : ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -3 ∧ x ≠ -6 →
    (1 / x + 2 / (x + 3) + 3 / (x + 6) = 1) ↔ (x = a ∨ x = b ∨ x = c)) ∧
  a^2 + b^2 + c^2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3352_335224


namespace NUMINAMATH_CALUDE_chess_club_committees_l3352_335260

/-- The number of teams in the chess club -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 6

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 3

/-- The total number of members in the organizing committee -/
def committee_size : ℕ := 16

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem chess_club_committees :
  (num_teams * choose team_size host_selection * (choose team_size non_host_selection) ^ (num_teams - 1)) = 12000000 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_committees_l3352_335260


namespace NUMINAMATH_CALUDE_jason_initial_cards_l3352_335297

theorem jason_initial_cards (initial_cards final_cards bought_cards : ℕ) 
  (h1 : bought_cards = 224)
  (h2 : final_cards = 900)
  (h3 : final_cards = initial_cards + bought_cards) :
  initial_cards = 676 := by
  sorry

end NUMINAMATH_CALUDE_jason_initial_cards_l3352_335297


namespace NUMINAMATH_CALUDE_circle_equation_example_l3352_335279

/-- The standard equation of a circle with center (h,k) and radius r is (x-h)^2 + (y-k)^2 = r^2 -/
def standard_circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Given a circle with center (3,1) and radius 5, its standard equation is (x-3)^2+(y-1)^2=25 -/
theorem circle_equation_example :
  ∀ x y : ℝ, standard_circle_equation 3 1 5 x y ↔ (x - 3)^2 + (y - 1)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_example_l3352_335279


namespace NUMINAMATH_CALUDE_bathroom_width_l3352_335287

/-- The width of a rectangular bathroom with length 4 feet and area 8 square feet is 2 feet. -/
theorem bathroom_width (length : ℝ) (area : ℝ) (width : ℝ) 
    (h1 : length = 4)
    (h2 : area = 8)
    (h3 : area = length * width) : width = 2 := by
  sorry

end NUMINAMATH_CALUDE_bathroom_width_l3352_335287


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3352_335209

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ p q : ℝ, 
    (3 * p^2 - 5 * p - 7 = 0) ∧ 
    (3 * q^2 - 5 * q - 7 = 0) ∧ 
    ((p + 2)^2 + b * (p + 2) + c = 0) ∧
    ((q + 2)^2 + b * (q + 2) + c = 0)) →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3352_335209


namespace NUMINAMATH_CALUDE_male_cousins_count_l3352_335219

/-- Represents the Martin family structure -/
structure MartinFamily where
  michael_sisters : ℕ
  michael_brothers : ℕ
  total_cousins : ℕ

/-- The number of male cousins counted by each female cousin in the Martin family -/
def male_cousins_per_female (family : MartinFamily) : ℕ :=
  family.michael_brothers + 1

/-- Theorem stating the number of male cousins counted by each female cousin -/
theorem male_cousins_count (family : MartinFamily) 
  (h1 : family.michael_sisters = 4)
  (h2 : family.michael_brothers = 6)
  (h3 : family.total_cousins = family.michael_sisters + family.michael_brothers + 2) 
  (h4 : ∃ n : ℕ, 2 * n = family.total_cousins) :
  male_cousins_per_female family = 8 := by
  sorry

#eval male_cousins_per_female { michael_sisters := 4, michael_brothers := 6, total_cousins := 14 }

end NUMINAMATH_CALUDE_male_cousins_count_l3352_335219


namespace NUMINAMATH_CALUDE_three_topping_pizzas_l3352_335263

theorem three_topping_pizzas (n : ℕ) (k : ℕ) : n = 7 → k = 3 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_three_topping_pizzas_l3352_335263


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3352_335283

/-- The line y - 1 = k(x + 2) passes through the point (-2, 1) for all values of k. -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (1 : ℝ) - 1 = k * ((-2 : ℝ) + 2) := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3352_335283


namespace NUMINAMATH_CALUDE_no_snuggly_integers_l3352_335228

/-- A two-digit positive integer is snuggly if it equals the sum of its nonzero tens digit and the cube of its units digit. -/
def is_snuggly (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n = (n / 10) + (n % 10)^3

/-- There are no snuggly two-digit positive integers. -/
theorem no_snuggly_integers : ¬∃ n : ℕ, is_snuggly n := by
  sorry

end NUMINAMATH_CALUDE_no_snuggly_integers_l3352_335228


namespace NUMINAMATH_CALUDE_symmetric_points_fourth_quadrant_l3352_335294

/-- Given points A(a, 3) and B(2, b) are symmetric with respect to the x-axis,
    prove that point M(a, b) is in the fourth quadrant. -/
theorem symmetric_points_fourth_quadrant (a b : ℝ) :
  (a = 2 ∧ b = -3) →  -- Symmetry conditions
  a > 0 ∧ b < 0       -- Fourth quadrant conditions
  := by sorry

end NUMINAMATH_CALUDE_symmetric_points_fourth_quadrant_l3352_335294


namespace NUMINAMATH_CALUDE_log_pieces_after_ten_cuts_l3352_335266

/-- The number of pieces obtained after cutting a log -/
def numPieces (cuts : ℕ) : ℕ := cuts + 1

/-- Theorem: The number of pieces obtained after 10 cuts on a log is 11 -/
theorem log_pieces_after_ten_cuts : numPieces 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_pieces_after_ten_cuts_l3352_335266


namespace NUMINAMATH_CALUDE_altitude_difference_l3352_335255

theorem altitude_difference (a b c : ℤ) (ha : a = -112) (hb : b = -80) (hc : c = -25) :
  (max a (max b c) - min a (min b c) : ℤ) = 87 := by
  sorry

end NUMINAMATH_CALUDE_altitude_difference_l3352_335255


namespace NUMINAMATH_CALUDE_pencil_length_after_sharpening_l3352_335233

def initial_length : ℕ := 100
def monday_sharpening : ℕ := 3
def tuesday_sharpening : ℕ := 5
def wednesday_sharpening : ℕ := 7
def thursday_sharpening : ℕ := 11
def friday_sharpening : ℕ := 13

theorem pencil_length_after_sharpening : 
  initial_length - (monday_sharpening + tuesday_sharpening + wednesday_sharpening + thursday_sharpening + friday_sharpening) = 61 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_after_sharpening_l3352_335233


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l3352_335291

def total_balls : ℕ := 11
def white_balls : ℕ := 4
def black_balls : ℕ := 7
def drawn_balls : ℕ := 3

theorem probability_three_white_balls : 
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 4 / 165 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l3352_335291


namespace NUMINAMATH_CALUDE_mike_bought_33_books_l3352_335227

/-- The number of books Mike bought at a yard sale -/
def books_bought (initial_books final_books books_given_away : ℕ) : ℕ :=
  final_books - (initial_books - books_given_away)

/-- Theorem stating that Mike bought 33 books at the yard sale -/
theorem mike_bought_33_books :
  books_bought 35 56 12 = 33 := by
  sorry

end NUMINAMATH_CALUDE_mike_bought_33_books_l3352_335227


namespace NUMINAMATH_CALUDE_max_value_theorem_l3352_335226

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hsum : a + b + c = 3) :
  (a * b / (a + b)) + (b * c / (b + c)) + (c * a / (c + a)) ≤ 3 / 2 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 3 ∧
    (a' * b' / (a' + b')) + (b' * c' / (b' + c')) + (c' * a' / (c' + a')) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3352_335226


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3352_335208

theorem quadratic_inequality_solution_set 
  (a : ℝ) (ha : a < 0) :
  {x : ℝ | 42 * x^2 + a * x - a^2 < 0} = {x : ℝ | a / 7 < x ∧ x < -a / 6} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3352_335208


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3352_335200

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem unique_number_satisfying_conditions : 
  ∃! n : ℕ, is_two_digit n ∧ 
    ((ends_in n 6 ∨ n % 7 = 0) ∧ ¬(ends_in n 6 ∧ n % 7 = 0)) ∧
    ((n > 26 ∨ ends_in n 8) ∧ ¬(n > 26 ∧ ends_in n 8)) ∧
    ((n % 13 = 0 ∨ n < 27) ∧ ¬(n % 13 = 0 ∧ n < 27)) ∧
    n = 91 := by
  sorry

#check unique_number_satisfying_conditions

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3352_335200


namespace NUMINAMATH_CALUDE_oranges_picked_sum_l3352_335231

/-- Given the number of oranges picked by Joan and Sara, prove that their sum
    equals the total number of oranges picked. -/
theorem oranges_picked_sum (joan_oranges sara_oranges total_oranges : ℕ)
  (h1 : joan_oranges = 37)
  (h2 : sara_oranges = 10)
  (h3 : total_oranges = 47) :
  joan_oranges + sara_oranges = total_oranges :=
by sorry

end NUMINAMATH_CALUDE_oranges_picked_sum_l3352_335231


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3352_335280

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3352_335280


namespace NUMINAMATH_CALUDE_math_competition_probabilities_l3352_335286

theorem math_competition_probabilities :
  let total_students : ℕ := 6
  let boys : ℕ := 3
  let girls : ℕ := 3
  let selected : ℕ := 2

  let prob_exactly_one_boy : ℚ := 3/5
  let prob_at_least_one_boy : ℚ := 4/5
  let prob_at_most_one_boy : ℚ := 4/5

  (total_students = boys + girls) →
  (prob_exactly_one_boy = 0.6) ∧
  (prob_at_least_one_boy = 0.8) ∧
  (prob_at_most_one_boy = 0.8) :=
by
  sorry

end NUMINAMATH_CALUDE_math_competition_probabilities_l3352_335286


namespace NUMINAMATH_CALUDE_existence_of_product_one_derivatives_l3352_335212

theorem existence_of_product_one_derivatives 
  (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_diff : DifferentiableOn ℝ f (Set.Ioo 0 1))
  (h_range : Set.range f ⊆ Set.Icc 0 1)
  (h_zero : f 0 = 0)
  (h_one : f 1 = 1) :
  ∃ a b : ℝ, a ∈ Set.Ioo 0 1 ∧ b ∈ Set.Ioo 0 1 ∧ a ≠ b ∧ 
    deriv f a * deriv f b = 1 :=
sorry

end NUMINAMATH_CALUDE_existence_of_product_one_derivatives_l3352_335212


namespace NUMINAMATH_CALUDE_cakes_sold_is_six_l3352_335254

/-- The number of cakes sold during dinner today, given the number of cakes
    baked today, yesterday, and the number of cakes left. -/
def cakes_sold_during_dinner (cakes_baked_today cakes_baked_yesterday cakes_left : ℕ) : ℕ :=
  cakes_baked_today + cakes_baked_yesterday - cakes_left

/-- Theorem stating that the number of cakes sold during dinner today is 6,
    given the specific conditions of the problem. -/
theorem cakes_sold_is_six :
  cakes_sold_during_dinner 5 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cakes_sold_is_six_l3352_335254


namespace NUMINAMATH_CALUDE_andrew_sandwiches_l3352_335282

/-- Given a total number of sandwiches and number of friends, 
    calculate the number of sandwiches per friend -/
def sandwiches_per_friend (total_sandwiches : ℕ) (num_friends : ℕ) : ℕ :=
  total_sandwiches / num_friends

/-- Theorem: Given 12 sandwiches and 4 friends, 
    the number of sandwiches per friend is 3 -/
theorem andrew_sandwiches : 
  sandwiches_per_friend 12 4 = 3 := by
  sorry


end NUMINAMATH_CALUDE_andrew_sandwiches_l3352_335282


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3352_335275

theorem quadratic_two_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) ↔ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3352_335275


namespace NUMINAMATH_CALUDE_football_game_attendance_l3352_335216

theorem football_game_attendance (saturday_attendance : ℕ) 
  (expected_total : ℕ) : saturday_attendance = 80 →
  expected_total = 350 →
  (saturday_attendance + 
   (saturday_attendance - 20) + 
   (saturday_attendance - 20 + 50) + 
   (saturday_attendance + (saturday_attendance - 20))) - 
  expected_total = 40 := by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_l3352_335216


namespace NUMINAMATH_CALUDE_allan_final_score_l3352_335225

/-- Calculates the final score on a test with the given parameters. -/
def final_score (total_questions : ℕ) (correct_answers : ℕ) (points_per_correct : ℚ) (points_per_incorrect : ℚ) : ℚ :=
  let incorrect_answers := total_questions - correct_answers
  (correct_answers : ℚ) * points_per_correct - (incorrect_answers : ℚ) * points_per_incorrect

/-- Theorem stating that Allan's final score is 100 given the test conditions. -/
theorem allan_final_score :
  let total_questions : ℕ := 120
  let correct_answers : ℕ := 104
  let points_per_correct : ℚ := 1
  let points_per_incorrect : ℚ := 1/4
  final_score total_questions correct_answers points_per_correct points_per_incorrect = 100 := by
  sorry

end NUMINAMATH_CALUDE_allan_final_score_l3352_335225


namespace NUMINAMATH_CALUDE_temperature_conversion_deviation_l3352_335278

theorem temperature_conversion_deviation (C : ℝ) : 
  let F_approx := 2 * C + 30
  let F_exact := (9 / 5) * C + 32
  let deviation := (F_approx - F_exact) / F_exact
  (40 / 29 ≤ C ∧ C ≤ 360 / 11) ↔ (abs deviation ≤ 0.05) :=
by sorry

end NUMINAMATH_CALUDE_temperature_conversion_deviation_l3352_335278


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3352_335215

theorem polynomial_remainder (x : ℝ) : 
  (x^5 + 2*x^2 + 3) % (x - 2) = 43 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3352_335215


namespace NUMINAMATH_CALUDE_dress_design_count_l3352_335295

/-- The number of available fabric colors -/
def num_colors : ℕ := 5

/-- The number of available patterns -/
def num_patterns : ℕ := 4

/-- The number of available sleeve styles -/
def num_sleeve_styles : ℕ := 3

/-- Each dress design requires exactly one color, one pattern, and one sleeve style -/
axiom dress_design_composition : True

/-- The total number of possible dress designs -/
def total_dress_designs : ℕ := num_colors * num_patterns * num_sleeve_styles

theorem dress_design_count : total_dress_designs = 60 := by
  sorry

end NUMINAMATH_CALUDE_dress_design_count_l3352_335295


namespace NUMINAMATH_CALUDE_even_function_quadratic_l3352_335264

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem even_function_quadratic 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_domain : Set.Icc (-1 - a) (2 * a) ⊆ Set.range (f a b)) :
  f a b (2 * a - b) = 5 := by
sorry

end NUMINAMATH_CALUDE_even_function_quadratic_l3352_335264


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_cubic_vs_quadratic_l3352_335259

-- Part 1
theorem sqrt_sum_comparison : Real.sqrt 7 + Real.sqrt 10 > Real.sqrt 3 + Real.sqrt 14 := by
  sorry

-- Part 2
theorem cubic_vs_quadratic (x : ℝ) (h : x > 1) : x^3 > x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_cubic_vs_quadratic_l3352_335259


namespace NUMINAMATH_CALUDE_max_triangle_area_l3352_335218

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The line l passing through F₂(1,0) -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 1

/-- The area of triangle F₁AB -/
def triangle_area (y₁ y₂ : ℝ) : ℝ :=
  |y₁ - y₂|

/-- The theorem stating the maximum area of triangle F₁AB -/
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = 3 ∧
  ∀ (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    trajectory x₁ y₁ →
    trajectory x₂ y₂ →
    line_l m x₁ y₁ →
    line_l m x₂ y₂ →
    x₁ ≠ x₂ →
    triangle_area y₁ y₂ ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3352_335218


namespace NUMINAMATH_CALUDE_problem_solution_l3352_335238

theorem problem_solution (a b c : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : c > 0) :
  (a * c < b * c) ∧ (a + b + c < b + c) ∧ (c / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3352_335238


namespace NUMINAMATH_CALUDE_cylinder_volume_l3352_335290

/-- The volume of a cylinder whose lateral surface unfolds into a square with side length 4 -/
theorem cylinder_volume (h : ℝ) (r : ℝ) : 
  h = 4 → 2 * Real.pi * r = 4 → Real.pi * r^2 * h = 16 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l3352_335290


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3352_335292

/-- Given a hyperbola C: mx^2 + ny^2 = 1 (m > 0, n < 0) with one of its asymptotes
    tangent to the circle x^2 + y^2 - 6x - 2y + 9 = 0, 
    the eccentricity of C is 5/4. -/
theorem hyperbola_eccentricity (m n : ℝ) (hm : m > 0) (hn : n < 0) :
  let C := {(x, y) : ℝ × ℝ | m * x^2 + n * y^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*x - 2*y + 9 = 0}
  let asymptote := {(x, y) : ℝ × ℝ | Real.sqrt m * x - Real.sqrt (-n) * y = 0}
  (∃ (p : ℝ × ℝ), p ∈ asymptote ∧ p ∈ circle) →
  let a := 1 / Real.sqrt m
  let b := 1 / Real.sqrt (-n)
  let e := Real.sqrt (1 + (b/a)^2)
  e = 5/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3352_335292


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l3352_335253

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coefficient_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 250000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation := {
  coefficient := 2.5,
  exponent := 5,
  coefficient_range := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem correct_scientific_notation :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l3352_335253


namespace NUMINAMATH_CALUDE_adult_tickets_bought_l3352_335229

/-- Proves the number of adult tickets bought given ticket prices and total information -/
theorem adult_tickets_bought (adult_price child_price : ℚ) (total_tickets : ℕ) (total_cost : ℚ) 
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_tickets : ℕ), 
    adult_tickets * adult_price + (total_tickets - adult_tickets) * child_price = total_cost ∧ 
    adult_tickets = 5 := by
  sorry

end NUMINAMATH_CALUDE_adult_tickets_bought_l3352_335229


namespace NUMINAMATH_CALUDE_maria_eggs_l3352_335256

/-- The number of eggs Maria has -/
def total_eggs (num_boxes : ℕ) (eggs_per_box : ℕ) : ℕ :=
  num_boxes * eggs_per_box

/-- Theorem: Maria has 21 eggs in total -/
theorem maria_eggs : total_eggs 3 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_maria_eggs_l3352_335256


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l3352_335221

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 2*x*y = 8) :
  ∃ (m : ℝ), m = 4 ∧ ∀ z, z = x + 2*y → z ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l3352_335221


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3352_335273

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 30)
  (h2 : Complex.abs (z + 3 * w) = 6)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = Real.sqrt 132.525 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3352_335273


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3352_335220

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 158 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3352_335220


namespace NUMINAMATH_CALUDE_smallest_valid_staircase_sum_of_digits_90_l3352_335210

def is_valid_staircase (n : ℕ) : Prop :=
  ⌈(n : ℚ) / 2⌉ - ⌈(n : ℚ) / 3⌉ = 15

theorem smallest_valid_staircase :
  ∀ m : ℕ, m < 90 → ¬(is_valid_staircase m) ∧ is_valid_staircase 90 :=
by sorry

theorem sum_of_digits_90 : (9 : ℕ) = (9 : ℕ) + (0 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_staircase_sum_of_digits_90_l3352_335210
