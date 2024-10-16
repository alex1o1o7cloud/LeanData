import Mathlib

namespace NUMINAMATH_CALUDE_rational_equation_solution_l1378_137828

theorem rational_equation_solution (x : ℝ) : 
  -5 < x ∧ x < 3 → ((x^2 - 4*x + 5) / (2*x - 2) = 2 ↔ x = 4 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1378_137828


namespace NUMINAMATH_CALUDE_crunch_difference_l1378_137829

/-- Given that Zachary did 17 crunches and David did 4 crunches,
    prove that David did 13 less crunches than Zachary. -/
theorem crunch_difference (zachary_crunches : ℕ) (david_crunches : ℕ)
  (h1 : zachary_crunches = 17)
  (h2 : david_crunches = 4) :
  zachary_crunches - david_crunches = 13 := by
  sorry

end NUMINAMATH_CALUDE_crunch_difference_l1378_137829


namespace NUMINAMATH_CALUDE_repeating_decimal_problem_l1378_137866

/-- Represents a repeating decimal with a single digit followed by 25 -/
def RepeatingDecimal (d : Nat) : ℚ :=
  (d * 100 + 25 : ℚ) / 999

/-- The main theorem -/
theorem repeating_decimal_problem (n : ℕ) (d : Nat) 
    (h_n_pos : n > 0)
    (h_d_digit : d < 10)
    (h_eq : (n : ℚ) / 810 = RepeatingDecimal d) :
    n = 750 ∧ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_problem_l1378_137866


namespace NUMINAMATH_CALUDE_expected_difference_coffee_tea_l1378_137803

-- Define the die sides
def dieSides : Nat := 8

-- Define perfect squares and primes up to 8
def perfectSquares : List Nat := [1, 4]
def primes : List Nat := [2, 3, 5, 7]

-- Define probabilities
def probCoffee : ℚ := 1 / 4
def probTea : ℚ := 1 / 2

-- Define number of days in a non-leap year
def daysInYear : Nat := 365

-- State the theorem
theorem expected_difference_coffee_tea :
  (probCoffee * daysInYear : ℚ) - (probTea * daysInYear : ℚ) = -91.25 := by
  sorry

end NUMINAMATH_CALUDE_expected_difference_coffee_tea_l1378_137803


namespace NUMINAMATH_CALUDE_plane_through_points_l1378_137820

/-- The plane equation coefficients -/
def A : ℤ := 1
def B : ℤ := 2
def C : ℤ := -2
def D : ℤ := -10

/-- The three points on the plane -/
def p : ℝ × ℝ × ℝ := (-2, 3, -3)
def q : ℝ × ℝ × ℝ := (2, 3, -1)
def r : ℝ × ℝ × ℝ := (4, 1, -2)

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_through_points :
  plane_equation p.1 p.2.1 p.2.2 ∧
  plane_equation q.1 q.2.1 q.2.2 ∧
  plane_equation r.1 r.2.1 r.2.2 ∧
  A > 0 ∧
  Nat.gcd (Int.natAbs A) (Int.natAbs B) = 1 ∧
  Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C) = 1 ∧
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Int.natAbs C)) (Int.natAbs D) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_through_points_l1378_137820


namespace NUMINAMATH_CALUDE_intersection_condition_l1378_137810

-- Define the curves
def curve1 (b x y : ℝ) : Prop := x^2 + y^2 = 2 * b^2
def curve2 (b x y : ℝ) : Prop := y = x^2 - b

-- Define the intersection condition
def intersect_at_four_points (b : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
    (curve1 b x1 y1 ∧ curve2 b x1 y1) ∧
    (curve1 b x2 y2 ∧ curve2 b x2 y2) ∧
    (curve1 b x3 y3 ∧ curve2 b x3 y3) ∧
    (curve1 b x4 y4 ∧ curve2 b x4 y4) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x1 ≠ x4 ∨ y1 ≠ y4) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3) ∧
    (x2 ≠ x4 ∨ y2 ≠ y4) ∧
    (x3 ≠ x4 ∨ y3 ≠ y4) ∧
    ∀ (x y : ℝ), (curve1 b x y ∧ curve2 b x y) →
      ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4))

-- State the theorem
theorem intersection_condition (b : ℝ) :
  intersect_at_four_points b ↔ b > 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_condition_l1378_137810


namespace NUMINAMATH_CALUDE_marie_age_proof_l1378_137843

/-- Marie's age in years -/
def marie_age : ℚ := 8/3

/-- Liam's age in years -/
def liam_age : ℚ := 4 * marie_age

/-- Oliver's age in years -/
def oliver_age : ℚ := marie_age + 8

theorem marie_age_proof :
  (liam_age = 4 * marie_age) ∧
  (oliver_age = marie_age + 8) ∧
  (liam_age = oliver_age) →
  marie_age = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_marie_age_proof_l1378_137843


namespace NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_5_mod_17_l1378_137879

theorem smallest_five_digit_negative_congruent_to_5_mod_17 : 
  ∀ n : ℤ, -99999 ≤ n ∧ n < -9999 ∧ n ≡ 5 [ZMOD 17] → n ≥ -10013 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_5_mod_17_l1378_137879


namespace NUMINAMATH_CALUDE_four_distinct_solutions_range_l1378_137808

-- Define the equation
def f (x m : ℝ) : ℝ := x^2 - 4 * |x| + 5 - m

-- State the theorem
theorem four_distinct_solutions_range (m : ℝ) :
  (∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a m = 0 ∧ f b m = 0 ∧ f c m = 0 ∧ f d m = 0) →
  m ∈ Set.Ioo 1 5 :=
by sorry

end NUMINAMATH_CALUDE_four_distinct_solutions_range_l1378_137808


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1378_137889

theorem sum_of_solutions (x : ℝ) : 
  (18 * x^2 - 45 * x - 70 = 0) → 
  (∃ y : ℝ, 18 * y^2 - 45 * y - 70 = 0 ∧ x + y = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1378_137889


namespace NUMINAMATH_CALUDE_pierre_cake_consumption_l1378_137801

theorem pierre_cake_consumption (total_weight : ℝ) (parts : ℕ) 
  (h1 : total_weight = 546)
  (h2 : parts = 12)
  (h3 : parts > 0) :
  let nathalie_portion := total_weight / parts
  let pierre_portion := 2.5 * nathalie_portion
  pierre_portion = 113.75 := by sorry

end NUMINAMATH_CALUDE_pierre_cake_consumption_l1378_137801


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_l1378_137814

theorem sin_cos_sixth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_l1378_137814


namespace NUMINAMATH_CALUDE_christen_peeled_twenty_l1378_137818

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  christen_join_time : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christen_peeled (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 20 potatoes in the given scenario -/
theorem christen_peeled_twenty (scenario : PotatoPeeling) 
  (h1 : scenario.total_potatoes = 44)
  (h2 : scenario.homer_rate = 3)
  (h3 : scenario.christen_rate = 5)
  (h4 : scenario.christen_join_time = 4) :
  christen_peeled scenario = 20 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_twenty_l1378_137818


namespace NUMINAMATH_CALUDE_minimum_bottles_needed_l1378_137822

def small_bottle_capacity : ℕ := 45
def large_bottle_capacity : ℕ := 600
def already_filled : ℕ := 90

theorem minimum_bottles_needed : 
  ∃ (n : ℕ), n * small_bottle_capacity + already_filled ≥ large_bottle_capacity ∧ 
  ∀ (m : ℕ), m * small_bottle_capacity + already_filled ≥ large_bottle_capacity → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_bottles_needed_l1378_137822


namespace NUMINAMATH_CALUDE_unique_solution_quartic_equation_l1378_137839

theorem unique_solution_quartic_equation :
  ∃! x : ℝ, x^4 + (2 - x)^4 + 2*x = 34 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quartic_equation_l1378_137839


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l1378_137864

/-- The rate per kg of grapes that Bruce purchased -/
def grape_rate : ℝ := 70

/-- The amount of grapes Bruce purchased in kg -/
def grape_amount : ℝ := 8

/-- The rate per kg of mangoes that Bruce purchased -/
def mango_rate : ℝ := 55

/-- The amount of mangoes Bruce purchased in kg -/
def mango_amount : ℝ := 11

/-- The total amount Bruce paid to the shopkeeper -/
def total_paid : ℝ := 1165

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l1378_137864


namespace NUMINAMATH_CALUDE_box_dimensions_sum_l1378_137846

-- Define the dimensions of the box
variable (P Q R : ℝ)

-- Define the conditions
def condition1 : Prop := P * Q = 30
def condition2 : Prop := P * R = 50
def condition3 : Prop := Q * R = 90

-- Theorem statement
theorem box_dimensions_sum 
  (h1 : condition1 P Q)
  (h2 : condition2 P R)
  (h3 : condition3 Q R) :
  P + Q + R = 18 * Real.sqrt 1.5 := by
  sorry

end NUMINAMATH_CALUDE_box_dimensions_sum_l1378_137846


namespace NUMINAMATH_CALUDE_five_objects_two_groups_l1378_137825

/-- The number of ways to partition n indistinguishable objects into k indistinguishable groups -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 3 ways to partition 5 indistinguishable objects into 2 indistinguishable groups -/
theorem five_objects_two_groups : partition_count 5 2 = 3 := by sorry

end NUMINAMATH_CALUDE_five_objects_two_groups_l1378_137825


namespace NUMINAMATH_CALUDE_vector_angle_problem_l1378_137806

/-- The angle between two 2D vectors -/
def angle_between (v w : ℝ × ℝ) : ℝ := sorry

/-- Converts degrees to radians -/
def deg_to_rad (deg : ℝ) : ℝ := sorry

theorem vector_angle_problem (a b : ℝ × ℝ) 
  (sum_eq : a.1 + b.1 = 2 ∧ a.2 + b.2 = -1)
  (a_eq : a = (1, 2)) :
  angle_between a b = deg_to_rad 135 := by sorry

end NUMINAMATH_CALUDE_vector_angle_problem_l1378_137806


namespace NUMINAMATH_CALUDE_right_triangle_legs_l1378_137878

theorem right_triangle_legs (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive lengths
  c = 25 →                 -- Hypotenuse is 25 cm
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  b / a = 4 / 3 →          -- Ratio of legs is 4:3
  a = 15 ∧ b = 20 :=       -- Legs are 15 cm and 20 cm
by sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l1378_137878


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l1378_137823

/-- The length of the path traversed by a vertex of an equilateral triangle rotating inside a square -/
theorem triangle_rotation_path_length 
  (triangle_side : ℝ) 
  (square_side : ℝ) 
  (rotations_per_corner : ℕ) 
  (num_corners : ℕ) 
  (h1 : triangle_side = 3) 
  (h2 : square_side = 6) 
  (h3 : rotations_per_corner = 2) 
  (h4 : num_corners = 4) : 
  (rotations_per_corner * num_corners * triangle_side * (2 * Real.pi / 3)) = 16 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l1378_137823


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1378_137817

/-- The equation (3x+8)(x-6) = -52 + kx has exactly one real solution if and only if k = 4√3 - 10 or k = -4√3 - 10 -/
theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x+8)*(x-6) = -52 + k*x) ↔ (k = 4*Real.sqrt 3 - 10 ∨ k = -4*Real.sqrt 3 - 10) := by
sorry


end NUMINAMATH_CALUDE_unique_solution_condition_l1378_137817


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1378_137854

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x = -6) ∧ (3 * 3^2 + k * 3 = -6) → 
  (∃ r : ℝ, r ≠ 3 ∧ 3 * r^2 + k * r = -6 ∧ r = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1378_137854


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1378_137834

theorem quadratic_equation_equivalence (x : ℝ) :
  let k : ℝ := 0.32653061224489793
  (2 * k * x^2 + 7 * k * x + 2 = 0) ↔ 
  (0.65306122448979586 * x^2 + 2.2857142857142865 * x + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1378_137834


namespace NUMINAMATH_CALUDE_largest_n_for_positive_sum_l1378_137842

/-- Given an arithmetic sequence {a_n} where a_1 = 9 and a_5 = 1,
    the largest natural number n for which the sum of the first n terms (S_n) is positive is 9. -/
theorem largest_n_for_positive_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 9 →
  a 5 = 1 →
  (∀ n, S n = n * (a 1 + a n) / 2) →  -- sum formula for arithmetic sequence
  (∀ m : ℕ, m > 9 → S m ≤ 0) ∧ S 9 > 0 := by
sorry


end NUMINAMATH_CALUDE_largest_n_for_positive_sum_l1378_137842


namespace NUMINAMATH_CALUDE_lentil_dishes_count_l1378_137873

/-- Represents the menu of a vegan restaurant -/
structure VeganMenu :=
  (total_dishes : ℕ)
  (beans_lentils : ℕ)
  (beans_seitan : ℕ)
  (tempeh_lentils : ℕ)
  (only_beans : ℕ)
  (only_seitan : ℕ)
  (only_lentils : ℕ)
  (only_tempeh : ℕ)

/-- The conditions of the vegan restaurant menu -/
def menu_conditions (m : VeganMenu) : Prop :=
  m.total_dishes = 20 ∧
  m.beans_lentils = 3 ∧
  m.beans_seitan = 4 ∧
  m.tempeh_lentils = 2 ∧
  m.only_beans = 2 * m.only_tempeh ∧
  m.only_seitan = 3 * m.only_tempeh ∧
  m.total_dishes = m.beans_lentils + m.beans_seitan + m.tempeh_lentils +
                   m.only_beans + m.only_seitan + m.only_lentils + m.only_tempeh

/-- The theorem stating that the number of dishes with lentils is 10 -/
theorem lentil_dishes_count (m : VeganMenu) :
  menu_conditions m → m.only_lentils + m.beans_lentils + m.tempeh_lentils = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_lentil_dishes_count_l1378_137873


namespace NUMINAMATH_CALUDE_sum_of_x_values_l1378_137838

theorem sum_of_x_values (x : ℝ) : 
  (|x - 25| = 50) → (∃ y : ℝ, |y - 25| = 50 ∧ x + y = 50) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l1378_137838


namespace NUMINAMATH_CALUDE_table_sticks_prove_table_sticks_l1378_137885

/-- The number of sticks of wood a chair makes -/
def chair_sticks : ℕ := 6

/-- The number of sticks of wood a stool makes -/
def stool_sticks : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chopped -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chopped -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chopped -/
def stools_chopped : ℕ := 4

/-- The number of hours Mary can keep warm -/
def hours_warm : ℕ := 34

/-- The theorem stating that a table makes 9 sticks of wood -/
theorem table_sticks : ℕ :=
  let total_sticks := hours_warm * sticks_per_hour
  let chair_total := chairs_chopped * chair_sticks
  let stool_total := stools_chopped * stool_sticks
  let table_total := total_sticks - chair_total - stool_total
  table_total / tables_chopped

/-- Proof of the theorem -/
theorem prove_table_sticks : table_sticks = 9 := by
  sorry


end NUMINAMATH_CALUDE_table_sticks_prove_table_sticks_l1378_137885


namespace NUMINAMATH_CALUDE_f_even_k_value_g_f_common_point_a_range_l1378_137802

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The logarithm base 4 -/
noncomputable def log4 (x : ℝ) : ℝ := (Real.log x) / (Real.log 4)

/-- The function f(x) -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := log4 (4^x + 1) + k * x

/-- The function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log4 (a * 2^x - 4/3 * a)

/-- The number of common points between f and g -/
def CommonPoints (f g : ℝ → ℝ) : Prop := ∃! x, f x = g x

theorem f_even_k_value :
  IsEven (f k) → k = -1/2 :=
sorry

theorem g_f_common_point_a_range :
  CommonPoints (f (-1/2)) (g a) → (a > 1 ∨ a = -3) :=
sorry

end NUMINAMATH_CALUDE_f_even_k_value_g_f_common_point_a_range_l1378_137802


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1378_137874

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

def line_equation (x y : ℝ) : Prop := x + y + 1 = 0

theorem min_distance_to_line (m n : ℝ) 
  (h : (a.1 - m) * (b.1 - m) + (a.2 - n) * (b.2 - n) = 0) : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), line_equation x y → 
    d ≤ Real.sqrt ((x - m)^2 + (y - n)^2) :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1378_137874


namespace NUMINAMATH_CALUDE_reflection_line_sum_l1378_137800

/-- Given that (2,3) is reflected across y = mx + b to (10,7), prove m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = (2 + 10) / 2 ∧ y = (3 + 7) / 2 ∧ y = m * x + b) →
  (m = -(10 - 2) / (7 - 3)) →
  m + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l1378_137800


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1378_137809

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1378_137809


namespace NUMINAMATH_CALUDE_perpendicular_and_parallel_lines_planes_l1378_137890

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def perp (l1 l2 : Line) : Prop := sorry
def para (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_and_parallel_lines_planes 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular m α) 
  (h2 : contained_in n β) :
  (parallel α β → perp m n) ∧ (para m n → perpendicular α β) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_and_parallel_lines_planes_l1378_137890


namespace NUMINAMATH_CALUDE_line_slope_l1378_137881

/-- Given a line with equation y = 2x + 1, its slope is 2. -/
theorem line_slope (x y : ℝ) : y = 2 * x + 1 → (∃ m : ℝ, m = 2 ∧ y = m * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1378_137881


namespace NUMINAMATH_CALUDE_max_sum_given_constraint_l1378_137898

theorem max_sum_given_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 + (x+y)^3 + 36*x*y = 3456) : 
  x + y ≤ 12 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^3 + y₀^3 + (x₀+y₀)^3 + 36*x₀*y₀ = 3456 ∧ x₀ + y₀ = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_constraint_l1378_137898


namespace NUMINAMATH_CALUDE_total_spent_is_135_l1378_137877

/-- The amount Jen spent on pastries -/
def jen_spent : ℝ := sorry

/-- The amount Lisa spent on pastries -/
def lisa_spent : ℝ := sorry

/-- For every dollar Jen spent, Lisa spent 20 cents less -/
axiom lisa_spent_relation : lisa_spent = 0.8 * jen_spent

/-- Jen spent $15 more than Lisa -/
axiom jen_spent_more : jen_spent = lisa_spent + 15

/-- The total amount spent on pastries -/
def total_spent : ℝ := jen_spent + lisa_spent

/-- Theorem stating that the total amount spent is $135 -/
theorem total_spent_is_135 : total_spent = 135 := by sorry

end NUMINAMATH_CALUDE_total_spent_is_135_l1378_137877


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l1378_137861

/-- A rectangle with perimeter 60 meters and area 224 square meters has a longer side of 16 meters. -/
theorem rectangle_longer_side (x y : ℝ) (h_perimeter : x + y = 30) (h_area : x * y = 224) 
  (h_x_longer : x ≥ y) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l1378_137861


namespace NUMINAMATH_CALUDE_total_candies_l1378_137856

/-- Represents the number of candies Lillian has initially -/
def initial_candies : ℕ := 88

/-- Represents the number of candies Lillian receives from her father -/
def additional_candies : ℕ := 5

/-- Theorem stating the total number of candies Lillian has after receiving more -/
theorem total_candies : initial_candies + additional_candies = 93 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l1378_137856


namespace NUMINAMATH_CALUDE_clock_correction_time_l1378_137848

/-- The number of minutes in 12 hours -/
def minutes_in_12_hours : ℕ := 12 * 60

/-- The number of minutes the clock gains per day -/
def minutes_gained_per_day : ℕ := 3

/-- The minimum number of days for the clock to show the correct time again -/
def min_days_to_correct_time : ℕ := minutes_in_12_hours / minutes_gained_per_day

theorem clock_correction_time :
  min_days_to_correct_time = 240 :=
sorry

end NUMINAMATH_CALUDE_clock_correction_time_l1378_137848


namespace NUMINAMATH_CALUDE_difference_of_sums_l1378_137840

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_5 (x : ℕ) : ℕ :=
  5 * ((x + 2) / 5)

def sum_rounded_to_5 (n : ℕ) : ℕ :=
  (n / 5) * (0 + 5 + 5 + 5 + 10)

theorem difference_of_sums (n : ℕ) (h : n = 200) : 
  (sum_to_n n) - (sum_rounded_to_5 n) = 19100 := by
  sorry

#check difference_of_sums

end NUMINAMATH_CALUDE_difference_of_sums_l1378_137840


namespace NUMINAMATH_CALUDE_find_divisor_l1378_137805

theorem find_divisor (divisor : ℕ) : 
  (144 / divisor = 13) ∧ (144 % divisor = 1) → divisor = 11 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l1378_137805


namespace NUMINAMATH_CALUDE_only_B_and_C_valid_l1378_137849

-- Define the set of individuals
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person
  | D : Person

-- Define a type for the selection of individuals
def Selection := Person → Prop

-- Define the conditions
def condition1 (s : Selection) : Prop := s Person.A → s Person.B
def condition2 (s : Selection) : Prop := ¬(s Person.C) → ¬(s Person.B)
def condition3 (s : Selection) : Prop := s Person.C → ¬(s Person.D)

-- Define that exactly two individuals are selected
def exactlyTwo (s : Selection) : Prop :=
  (∃ (p1 p2 : Person), p1 ≠ p2 ∧ s p1 ∧ s p2 ∧ ∀ (p : Person), s p → (p = p1 ∨ p = p2))

-- State the theorem
theorem only_B_and_C_valid :
  ∀ (s : Selection),
    condition1 s →
    condition2 s →
    condition3 s →
    exactlyTwo s →
    s Person.B ∧ s Person.C ∧ ¬(s Person.A) ∧ ¬(s Person.D) :=
by
  sorry


end NUMINAMATH_CALUDE_only_B_and_C_valid_l1378_137849


namespace NUMINAMATH_CALUDE_school_distance_proof_l1378_137892

/-- Represents the time taken to drive to school -/
structure DriveTime where
  rushHour : ℝ  -- Time in hours during rush hour
  holiday : ℝ   -- Time in hours during holiday

/-- Represents the speed of driving to school -/
structure DriveSpeed where
  rushHour : ℝ  -- Speed in miles per hour during rush hour
  holiday : ℝ   -- Speed in miles per hour during holiday

/-- The distance to school in miles -/
def distanceToSchool : ℝ := 10

theorem school_distance_proof (t : DriveTime) (s : DriveSpeed) : distanceToSchool = 10 :=
  by
  have h1 : t.rushHour = 1/2 := by sorry
  have h2 : t.holiday = 1/4 := by sorry
  have h3 : s.holiday = s.rushHour + 20 := by sorry
  have h4 : distanceToSchool = s.rushHour * t.rushHour := by sorry
  have h5 : distanceToSchool = s.holiday * t.holiday := by sorry
  sorry

#check school_distance_proof

end NUMINAMATH_CALUDE_school_distance_proof_l1378_137892


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1378_137865

theorem sqrt_x_minus_one_real (x : ℝ) : x ≥ 1 ↔ ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1378_137865


namespace NUMINAMATH_CALUDE_crossroads_four_roads_routes_l1378_137859

/-- Represents a crossroads with a given number of roads -/
structure Crossroads :=
  (num_roads : ℕ)

/-- Calculates the number of possible driving routes at a crossroads -/
def driving_routes (c : Crossroads) : ℕ :=
  c.num_roads * (c.num_roads - 1)

/-- Theorem: At a crossroads with 4 roads, where vehicles are not allowed to turn back,
    the total number of possible driving routes is 12 -/
theorem crossroads_four_roads_routes :
  ∃ (c : Crossroads), c.num_roads = 4 ∧ driving_routes c = 12 :=
sorry

end NUMINAMATH_CALUDE_crossroads_four_roads_routes_l1378_137859


namespace NUMINAMATH_CALUDE_sum_of_integers_l1378_137868

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 5)
  (eq4 : d - a + b = 4) :
  a + b + c + d = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1378_137868


namespace NUMINAMATH_CALUDE_interest_problem_l1378_137837

theorem interest_problem (P : ℝ) : 
  (P * 0.04 + P * 0.06 + P * 0.08 = 2700) → P = 15000 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l1378_137837


namespace NUMINAMATH_CALUDE_tan_two_beta_l1378_137897

theorem tan_two_beta (α β : Real) 
  (h1 : Real.tan (α + β) = 1) 
  (h2 : Real.tan (α - β) = 7) : 
  Real.tan (2 * β) = -3/4 := by sorry

end NUMINAMATH_CALUDE_tan_two_beta_l1378_137897


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1378_137891

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp_plane_line : Plane → Line → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (m n : Line)
  (h1 : perp_plane_line α m)
  (h2 : perp_plane_line β n)
  (h3 : perp_plane α β) :
  perp_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1378_137891


namespace NUMINAMATH_CALUDE_incorrect_operation_l1378_137844

theorem incorrect_operation : (4 + 5)^2 ≠ 4^2 + 5^2 := by sorry

end NUMINAMATH_CALUDE_incorrect_operation_l1378_137844


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equidistant_sum_l1378_137833

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_equidistant_sum
  (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 4 + a 8 = 16 → a 2 + a 10 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equidistant_sum_l1378_137833


namespace NUMINAMATH_CALUDE_floor_x_length_l1378_137867

/-- Represents the dimensions of a rectangular floor -/
structure Floor where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular floor -/
def area (f : Floor) : ℝ := f.width * f.length

theorem floor_x_length
  (x y : Floor)
  (h1 : area x = area y)
  (h2 : x.width = 10)
  (h3 : y.width = 9)
  (h4 : y.length = 20) :
  x.length = 18 := by
  sorry

end NUMINAMATH_CALUDE_floor_x_length_l1378_137867


namespace NUMINAMATH_CALUDE_percent_to_decimal_four_percent_to_decimal_l1378_137876

theorem percent_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem four_percent_to_decimal : (4 : ℚ) / 100 = 0.04 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_four_percent_to_decimal_l1378_137876


namespace NUMINAMATH_CALUDE_max_d_value_l1378_137853

def is_valid_number (d f : ℕ) : Prop :=
  d < 10 ∧ f < 10 ∧ (636330 + 100000 * d + f) % 33 = 0

theorem max_d_value :
  (∃ d f : ℕ, is_valid_number d f) →
  (∀ d f : ℕ, is_valid_number d f → d ≤ 9) ∧
  (∃ f : ℕ, is_valid_number 9 f) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1378_137853


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1378_137830

theorem triangle_perimeter (a : ℕ) (h1 : 2 < a) (h2 : a < 8) (h3 : Even a) :
  2 + 6 + a = 14 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1378_137830


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l1378_137807

theorem hyperbola_minimum_value (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) 
  (h_eccentricity : (a^2 + b^2) / a^2 = 4) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → 
  (∀ a' b', a' ≥ 1 → b' ≥ 1 → (a'^2 + b'^2) / a'^2 = 4 → 
    (b^2 + 1) / (Real.sqrt 3 * a) ≤ (b'^2 + 1) / (Real.sqrt 3 * a')) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l1378_137807


namespace NUMINAMATH_CALUDE_polygon_sides_l1378_137831

theorem polygon_sides (interior_angle : ℝ) (h : interior_angle = 140) :
  (360 : ℝ) / (180 - interior_angle) = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1378_137831


namespace NUMINAMATH_CALUDE_power_of_power_l1378_137871

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1378_137871


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l1378_137860

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_transitivity
  (α β γ : Plane) (m n : Line)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_diff_lines : m ≠ n)
  (h1 : perp n α)
  (h2 : perp n β)
  (h3 : perp m α)
  : perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l1378_137860


namespace NUMINAMATH_CALUDE_table_sum_zero_l1378_137847

structure Table :=
  (a b c d : ℝ)

def distinct (t : Table) : Prop :=
  t.a ≠ t.b ∧ t.a ≠ t.c ∧ t.a ≠ t.d ∧ t.b ≠ t.c ∧ t.b ≠ t.d ∧ t.c ≠ t.d

def row_sum_equal (t : Table) : Prop :=
  t.a + t.b = t.c + t.d

def column_product_equal (t : Table) : Prop :=
  t.a * t.c = t.b * t.d

theorem table_sum_zero (t : Table) 
  (h1 : distinct t) 
  (h2 : row_sum_equal t) 
  (h3 : column_product_equal t) : 
  t.a + t.b + t.c + t.d = 0 := by
  sorry

end NUMINAMATH_CALUDE_table_sum_zero_l1378_137847


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l1378_137826

-- Define ω as a complex number
variable (ω : ℂ)

-- Define the conditions
def omega_condition := ω^5 = 1 ∧ ω ≠ 1

-- Define α and β
def α := ω + ω^2
def β := ω^3 + ω^4

-- Define the theorem
theorem quadratic_coefficients (h : omega_condition ω) : 
  ∃ (p : ℝ × ℝ), p.1 = 0 ∧ p.2 = 2 ∧ 
  (α ω)^2 + p.1 * (α ω) + p.2 = 0 ∧ 
  (β ω)^2 + p.1 * (β ω) + p.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l1378_137826


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l1378_137836

theorem missing_fraction_sum (sum : ℚ) (f1 f2 f3 f4 f5 f6 : ℚ) :
  sum = 45/100 →
  f1 = 1/3 →
  f2 = 1/2 →
  f3 = -5/6 →
  f4 = 1/5 →
  f5 = -9/20 →
  f6 = -9/20 →
  ∃ x : ℚ, x = 23/20 ∧ sum = f1 + f2 + f3 + f4 + f5 + f6 + x :=
by sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l1378_137836


namespace NUMINAMATH_CALUDE_chemistry_books_count_l1378_137882

/-- The number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chemistry_books_count :
  ∃ (c : ℕ),
    c > 0 ∧
    (choose2 10) * (choose2 c) = 1260 ∧
    ∀ (x : ℕ), x > 0 → (choose2 10) * (choose2 x) = 1260 → x = c :=
by sorry

end NUMINAMATH_CALUDE_chemistry_books_count_l1378_137882


namespace NUMINAMATH_CALUDE_min_t_value_fixed_point_BD_l1378_137819

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the triangle area function
def triangle_area (t angle_AOB : ℝ) : Prop := t * Real.tan angle_AOB > 0

-- Theorem for minimum value of t
theorem min_t_value (a : ℝ) : 
  ∃ (t : ℝ), triangle_area t (Real.arctan ((4*a)/(a^2 - 4))) ∧ 
  t ≥ -2 ∧ 
  (t = -2 ↔ a = 2) := 
sorry

-- Theorem for fixed point of line BD when a = -1
theorem fixed_point_BD (x y : ℝ) : 
  parabola_C x y → 
  ∃ (x' y' : ℝ), parabola_C x' (-y') ∧ 
  (y - y' = (4 / (y' + y)) * (x - x'^2/4)) → 
  x = 1 ∧ y = 0 := 
sorry

end NUMINAMATH_CALUDE_min_t_value_fixed_point_BD_l1378_137819


namespace NUMINAMATH_CALUDE_characterization_of_n_l1378_137870

def invalid_n : Set ℕ := {2, 3, 5, 6, 7, 8, 13, 14, 15, 17, 19, 21, 23, 26, 27, 30, 47, 51, 53, 55, 61}

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (m : ℕ) (a : Fin (m-1) → ℕ), 
    (∀ i : Fin (m-1), 1 ≤ a i ∧ a i ≤ m - 1) ∧
    (∀ i j : Fin (m-1), i ≠ j → a i ≠ a j) ∧
    n = (Finset.univ.sum fun i => a i * (m - a i))

theorem characterization_of_n (n : ℕ) :
  n > 0 → (satisfies_condition n ↔ n ∉ invalid_n) := by sorry

end NUMINAMATH_CALUDE_characterization_of_n_l1378_137870


namespace NUMINAMATH_CALUDE_ad_space_width_l1378_137872

def ad_problem (num_spaces : ℕ) (length : ℝ) (cost_per_sqft : ℝ) (total_cost : ℝ) : Prop :=
  ∃ w : ℝ,
    w > 0 ∧
    num_spaces * length * w * cost_per_sqft = total_cost ∧
    w = 5

theorem ad_space_width :
  ad_problem 30 12 60 108000 :=
sorry

end NUMINAMATH_CALUDE_ad_space_width_l1378_137872


namespace NUMINAMATH_CALUDE_odd_power_sum_divisibility_l1378_137858

theorem odd_power_sum_divisibility (k : ℕ) (x y : ℤ) (h_odd : Odd k) (h_pos : k > 0) :
  (x^k + y^k) % (x + y) = 0 → (x^(k+2) + y^(k+2)) % (x + y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisibility_l1378_137858


namespace NUMINAMATH_CALUDE_tasty_pair_iff_isogonal_conjugate_exists_tasty_pair_for_both_triangles_l1378_137850

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the properties of the triangle
def isAcute (t : Triangle) : Prop := sorry

def isScalene (t : Triangle) : Prop := sorry

-- Define the tasty pair property
def isTastyPair (t : Triangle) (P Q : Point) : Prop := sorry

-- Define isogonal conjugates
def isIsogonalConjugate (t : Triangle) (P Q : Point) : Prop := sorry

-- Define the reflection of a triangle about its circumcenter
def reflectTriangle (t : Triangle) : Triangle := sorry

-- Main theorem
theorem tasty_pair_iff_isogonal_conjugate (t : Triangle) (h1 : isAcute t) (h2 : isScalene t) :
  ∀ P Q : Point, isTastyPair t P Q ↔ isIsogonalConjugate t P Q :=
sorry

-- Additional theorem
theorem exists_tasty_pair_for_both_triangles (t : Triangle) (h1 : isAcute t) (h2 : isScalene t) :
  ∃ P Q : Point, isTastyPair t P Q ∧ isTastyPair (reflectTriangle t) P Q :=
sorry

end NUMINAMATH_CALUDE_tasty_pair_iff_isogonal_conjugate_exists_tasty_pair_for_both_triangles_l1378_137850


namespace NUMINAMATH_CALUDE_smallest_integer_a_l1378_137824

theorem smallest_integer_a : ∃ (a : ℕ), (∀ (x y : ℝ), x > 0 → y > 0 → x + Real.sqrt (3 * x * y) ≤ a * (x + y)) ∧ 
  (∀ (b : ℕ), b < a → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + Real.sqrt (3 * x * y) > b * (x + y)) ∧ 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_a_l1378_137824


namespace NUMINAMATH_CALUDE_wades_tips_per_customer_l1378_137857

/-- Wade's tips per customer calculation -/
theorem wades_tips_per_customer :
  ∀ (tips_per_customer : ℚ),
  (28 : ℚ) * tips_per_customer +  -- Friday tips
  (3 * 28 : ℚ) * tips_per_customer +  -- Saturday tips (3 times Friday)
  (36 : ℚ) * tips_per_customer =  -- Sunday tips
  (296 : ℚ) →  -- Total tips
  tips_per_customer = 2 := by
sorry

end NUMINAMATH_CALUDE_wades_tips_per_customer_l1378_137857


namespace NUMINAMATH_CALUDE_system_solution_l1378_137851

/-- Given a system of equations x * y = a and x^5 + y^5 = b^5, this theorem states the solutions
    for different cases of a and b. -/
theorem system_solution (a b : ℝ) :
  (∀ x y : ℝ, x * y = a ∧ x^5 + y^5 = b^5 →
    (a = 0 ∧ b = 0 ∧ ∃ t : ℝ, x = t ∧ y = -t) ∨
    ((16 * b^5 ≤ a^5 ∧ a^5 < 0) ∨ (0 < a^5 ∧ a^5 ≤ 16 * b^5) ∧
      ((x = a/2 + Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4) ∧
        y = a/2 - Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4)) ∨
       (x = a/2 - Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4) ∧
        y = a/2 + Real.sqrt (Real.sqrt ((a^5 + 4*b^5)/(5*a))/2 - a^2/4))))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1378_137851


namespace NUMINAMATH_CALUDE_nigels_money_l1378_137812

theorem nigels_money (initial_amount : ℕ) (given_away : ℕ) (final_amount : ℕ) : 
  initial_amount = 45 →
  given_away = 25 →
  final_amount = 2 * initial_amount + 10 →
  final_amount - (initial_amount - given_away) = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_nigels_money_l1378_137812


namespace NUMINAMATH_CALUDE_solution_unique_l1378_137875

theorem solution_unique : 
  ∃! (x y : ℚ), (15 * x + 24 * y = 18) ∧ (24 * x + 15 * y = 63) ∧ (x = 46/13) ∧ (y = -19/13) := by
sorry

end NUMINAMATH_CALUDE_solution_unique_l1378_137875


namespace NUMINAMATH_CALUDE_f_2004_equals_2003_l1378_137845

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function g: ℝ → ℝ is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem f_2004_equals_2003 
  (f g : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_odd : IsOdd g)
  (h_relation : ∀ x, g x = f (x - 1))
  (h_g1 : g 1 = 2003) :
  f 2004 = 2003 := by
  sorry

end NUMINAMATH_CALUDE_f_2004_equals_2003_l1378_137845


namespace NUMINAMATH_CALUDE_complex_sum_equality_l1378_137835

theorem complex_sum_equality : 
  let A : ℂ := 3 + 2*I
  let O : ℂ := -3 + I
  let P : ℂ := 1 - 2*I
  let S : ℂ := 4 + 5*I
  let T : ℂ := -1
  A - O + P + S + T = 10 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l1378_137835


namespace NUMINAMATH_CALUDE_sqrt_three_minus_fraction_bound_l1378_137827

theorem sqrt_three_minus_fraction_bound (n m : ℕ) (h : Real.sqrt 3 - (m : ℝ) / n > 0) :
  Real.sqrt 3 - (m : ℝ) / n > 1 / (2 * (m : ℝ) * n) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_fraction_bound_l1378_137827


namespace NUMINAMATH_CALUDE_lanas_bouquets_l1378_137804

theorem lanas_bouquets (tulips roses extra : ℕ) : 
  tulips = 36 → roses = 37 → extra = 3 → 
  tulips + roses + extra = 76 := by
sorry

end NUMINAMATH_CALUDE_lanas_bouquets_l1378_137804


namespace NUMINAMATH_CALUDE_carl_josh_wage_ratio_l1378_137816

/-- Represents the hourly wage ratio between Carl and Josh -/
def wage_ratio : ℚ := 1 / 2

theorem carl_josh_wage_ratio : 
  let josh_hours_per_day : ℕ := 8
  let josh_days_per_week : ℕ := 5
  let josh_weeks_per_month : ℕ := 4
  let carl_hours_less_per_day : ℕ := 2
  let josh_hourly_wage : ℚ := 9
  let total_monthly_pay : ℚ := 1980
  
  let josh_monthly_hours : ℕ := josh_hours_per_day * josh_days_per_week * josh_weeks_per_month
  let carl_monthly_hours : ℕ := (josh_hours_per_day - carl_hours_less_per_day) * josh_days_per_week * josh_weeks_per_month
  let josh_monthly_pay : ℚ := josh_hourly_wage * josh_monthly_hours
  let carl_monthly_pay : ℚ := total_monthly_pay - josh_monthly_pay
  let carl_hourly_wage : ℚ := carl_monthly_pay / carl_monthly_hours

  carl_hourly_wage / josh_hourly_wage = wage_ratio :=
by
  sorry

#check carl_josh_wage_ratio

end NUMINAMATH_CALUDE_carl_josh_wage_ratio_l1378_137816


namespace NUMINAMATH_CALUDE_unique_zero_of_f_l1378_137815

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else Real.log x / Real.log a

-- Theorem statement
theorem unique_zero_of_f (a : ℝ) (h : a > 0) :
  ∃! x, f a x = a :=
sorry

end NUMINAMATH_CALUDE_unique_zero_of_f_l1378_137815


namespace NUMINAMATH_CALUDE_sqrt_product_and_difference_of_squares_l1378_137895

theorem sqrt_product_and_difference_of_squares :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y) ∧
  (∀ a b : ℝ, (a + b) * (a - b) = a^2 - b^2) ∧
  (Real.sqrt 3 * Real.sqrt 27 = 9) ∧
  ((Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) - (Real.sqrt 3 - 2)^2 = 4 * Real.sqrt 3 - 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_and_difference_of_squares_l1378_137895


namespace NUMINAMATH_CALUDE_focal_chord_circle_tangent_to_directrix_l1378_137880

-- Define a parabola
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ := (0, p)
  vertex : ℝ × ℝ := (0, 0)
  directrix : ℝ → ℝ := fun x ↦ -p

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the focal chord circle
def focal_chord_circle (parab : Parabola) : Circle :=
  { center := parab.focus
  , radius := parab.p }

-- Theorem statement
theorem focal_chord_circle_tangent_to_directrix (parab : Parabola) :
  let circle := focal_chord_circle parab
  let lowest_point := (circle.center.1, circle.center.2 - circle.radius)
  lowest_point.2 = 0 ∧ parab.directrix lowest_point.1 = -parab.p :=
sorry

end NUMINAMATH_CALUDE_focal_chord_circle_tangent_to_directrix_l1378_137880


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1378_137811

/-- Given a geometric sequence a, prove that if a₃ = -9 and a₇ = -1, then a₅ = -3 -/
theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_3 : a 3 = -9) 
  (h_7 : a 7 = -1) : 
  a 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1378_137811


namespace NUMINAMATH_CALUDE_triangle_sin_b_l1378_137863

theorem triangle_sin_b (A B C : Real) (AC BC : Real) (h1 : AC = 2) (h2 : BC = 3) (h3 : Real.cos A = 3/5) :
  Real.sin B = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_b_l1378_137863


namespace NUMINAMATH_CALUDE_at_least_one_multiple_of_three_l1378_137855

theorem at_least_one_multiple_of_three (a b : ℤ) : 
  (a + b) % 3 = 0 ∨ (a * b) % 3 = 0 ∨ (a - b) % 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_multiple_of_three_l1378_137855


namespace NUMINAMATH_CALUDE_point_coordinates_l1378_137821

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Determines if a point is in the second quadrant -/
def isSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The distance of a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance of a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates :
  ∀ (p : Point),
    isSecondQuadrant p →
    distanceToXAxis p = 2 →
    distanceToYAxis p = 3 →
    p.x = -3 ∧ p.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1378_137821


namespace NUMINAMATH_CALUDE_watch_cost_price_l1378_137884

theorem watch_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (additional_amount : ℝ) :
  loss_percentage = 10 →
  gain_percentage = 4 →
  additional_amount = 168 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_percentage / 100) + additional_amount = cost_price * (1 + gain_percentage / 100) ∧
    cost_price = 1200 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1378_137884


namespace NUMINAMATH_CALUDE_shale_mix_cost_per_pound_l1378_137869

/-- Prove that the cost of the shale mix per pound is $5 -/
theorem shale_mix_cost_per_pound
  (limestone_cost : ℝ)
  (total_weight : ℝ)
  (total_cost_per_pound : ℝ)
  (limestone_weight : ℝ)
  (h1 : limestone_cost = 3)
  (h2 : total_weight = 100)
  (h3 : total_cost_per_pound = 4.25)
  (h4 : limestone_weight = 37.5) :
  let shale_weight := total_weight - limestone_weight
  let total_cost := total_weight * total_cost_per_pound
  let limestone_total_cost := limestone_weight * limestone_cost
  let shale_total_cost := total_cost - limestone_total_cost
  shale_total_cost / shale_weight = 5 := by
sorry

end NUMINAMATH_CALUDE_shale_mix_cost_per_pound_l1378_137869


namespace NUMINAMATH_CALUDE_triple_composition_even_l1378_137899

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem triple_composition_even (g : ℝ → ℝ) (h : IsEven g) : IsEven (fun x ↦ g (g (g x))) := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_even_l1378_137899


namespace NUMINAMATH_CALUDE_negation_of_forall_proposition_l1378_137888

theorem negation_of_forall_proposition :
  (¬ ∀ x : ℝ, x^2 < 1 → x < 1) ↔ (∃ x : ℝ, x^2 < 1 ∧ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_proposition_l1378_137888


namespace NUMINAMATH_CALUDE_f_min_at_inv_e_l1378_137862

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem f_min_at_inv_e :
  ∀ x > 0, f (1 / Real.exp 1) ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_f_min_at_inv_e_l1378_137862


namespace NUMINAMATH_CALUDE_power_function_inequality_l1378_137886

-- Define a power function that passes through (2,8)
def f (x : ℝ) : ℝ := x^3

-- Theorem statement
theorem power_function_inequality (a : ℝ) :
  f (a - 3) > f (1 - a) ↔ a > 2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_inequality_l1378_137886


namespace NUMINAMATH_CALUDE_min_boxes_for_cube_l1378_137841

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the number of boxes needed to form a cube -/
def boxesNeededForCube (box : BoxDimensions) : ℕ :=
  let lcm := Nat.lcm (Nat.lcm box.width box.length) box.height
  (lcm / box.width) * (lcm / box.length) * (lcm / box.height)

/-- The main theorem stating that 24 boxes are needed to form a cube -/
theorem min_boxes_for_cube :
  let box : BoxDimensions := ⟨18, 12, 9⟩
  boxesNeededForCube box = 24 := by
  sorry

#eval boxesNeededForCube ⟨18, 12, 9⟩

end NUMINAMATH_CALUDE_min_boxes_for_cube_l1378_137841


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l1378_137883

def bacteria_growth (initial_count : ℕ) (final_count : ℕ) (tripling_time : ℕ) : ℕ → Prop :=
  fun hours => initial_count * (3 ^ (hours / tripling_time)) = final_count

theorem bacteria_growth_time : 
  bacteria_growth 200 16200 6 24 := by sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l1378_137883


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1378_137887

/-- Given a rectangular field with one side of 80 feet and three sides fenced with 97 feet of fencing,
    prove that the area of the field is 680 square feet. -/
theorem rectangular_field_area (L W : ℝ) : 
  L = 80 → 
  2 * W + L = 97 → 
  L * W = 680 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1378_137887


namespace NUMINAMATH_CALUDE_monotonicity_of_F_intersection_property_l1378_137894

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) : ℝ := x * (Real.log x + 1)

def f' (x : ℝ) : ℝ := Real.log x + 2

def F (x : ℝ) (a : ℝ) : ℝ := a * x^2 + f' x

theorem monotonicity_of_F (x : ℝ) (a : ℝ) (h : x > 0) :
  (a ≥ 0 → StrictMono (F · a)) ∧
  (a < 0 → StrictMonoOn (F · a) (Set.Ioo 0 (Real.sqrt (-1 / (2 * a)))) ∧
           StrictAntiOn (F · a) (Set.Ioi (Real.sqrt (-1 / (2 * a))))) :=
sorry

theorem intersection_property (x₁ x₂ k : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ < x₂) :
  k = (f' x₂ - f' x₁) / (x₂ - x₁) → x₁ < 1 / k ∧ 1 / k < x₂ :=
sorry

end NUMINAMATH_CALUDE_monotonicity_of_F_intersection_property_l1378_137894


namespace NUMINAMATH_CALUDE_tennis_racket_packaging_l1378_137896

/-- Given information about tennis racket packaging, prove the number of rackets in the other carton type. -/
theorem tennis_racket_packaging (total_cartons : ℕ) (total_rackets : ℕ) (three_racket_cartons : ℕ) 
  (h1 : total_cartons = 38)
  (h2 : total_rackets = 100)
  (h3 : three_racket_cartons = 24)
  : ∃ (other_carton_size : ℕ), 
    other_carton_size * (total_cartons - three_racket_cartons) + 3 * three_racket_cartons = total_rackets ∧ 
    other_carton_size = 2 :=
by sorry

end NUMINAMATH_CALUDE_tennis_racket_packaging_l1378_137896


namespace NUMINAMATH_CALUDE_congruent_sufficient_not_necessary_for_equal_area_l1378_137852

-- Define a triangle type
structure Triangle where
  -- You might define a triangle using its vertices or side lengths
  -- For simplicity, we'll just assume such a type exists
  mk :: (area : ℝ)

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop :=
  -- In reality, this would involve comparing all sides and angles
  -- For our purposes, we'll leave it as an abstract property
  sorry

-- Theorem statement
theorem congruent_sufficient_not_necessary_for_equal_area :
  (∀ t1 t2 : Triangle, congruent t1 t2 → t1.area = t2.area) ∧
  (∃ t1 t2 : Triangle, t1.area = t2.area ∧ ¬congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_congruent_sufficient_not_necessary_for_equal_area_l1378_137852


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt2_over_2_l1378_137893

theorem complex_modulus_sqrt2_over_2 (z : ℂ) (h : z * Complex.I / (z - Complex.I) = 1) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt2_over_2_l1378_137893


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l1378_137832

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the rug with three colored regions -/
structure Rug where
  inner : Rectangle
  middle : Rectangle
  outer : Rectangle

/-- Checks if three real numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem rug_inner_length_is_four (rug : Rug) : 
  rug.inner.width = 2 ∧ 
  rug.middle.length = rug.inner.length + 4 ∧ 
  rug.middle.width = rug.inner.width + 4 ∧
  rug.outer.length = rug.middle.length + 4 ∧
  rug.outer.width = rug.middle.width + 4 ∧
  isArithmeticProgression (area rug.inner) (area rug.middle - area rug.inner) (area rug.outer - area rug.middle) →
  rug.inner.length = 4 := by
sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l1378_137832


namespace NUMINAMATH_CALUDE_shaded_area_is_five_l1378_137813

/-- Given a parallelogram with regions labeled by areas, prove that the shaded region α has area 5 -/
theorem shaded_area_is_five (x y α : ℝ) 
  (h1 : 3 + α + y = 4 + α + x)
  (h2 : 1 + x + 3 + 3 + α + y + 4 + 1 = 2 * (4 + α + x)) : 
  α = 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_five_l1378_137813
