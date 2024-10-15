import Mathlib

namespace NUMINAMATH_CALUDE_walking_problem_solution_l1747_174718

/-- Two people walking on a line between points A and B -/
def WalkingProblem (distance_AB : ℝ) : Prop :=
  ∃ (first_meeting second_meeting : ℝ),
    first_meeting = 5 ∧
    second_meeting = distance_AB - 4 ∧
    2 * distance_AB = first_meeting + second_meeting

theorem walking_problem_solution :
  WalkingProblem 11 := by sorry

end NUMINAMATH_CALUDE_walking_problem_solution_l1747_174718


namespace NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l1747_174789

theorem no_real_solutions_for_sqrt_equation :
  ∀ z : ℝ, ¬(Real.sqrt (5 - 4*z) = 7) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_sqrt_equation_l1747_174789


namespace NUMINAMATH_CALUDE_square_difference_l1747_174730

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1747_174730


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l1747_174746

-- Define the edge lengths
def cube1_edge_inches : ℝ := 4
def cube2_edge_feet : ℝ := 2

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem statement
theorem cube_volume_ratio :
  (cube1_edge_inches ^ 3) / ((cube2_edge_feet * feet_to_inches) ^ 3) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l1747_174746


namespace NUMINAMATH_CALUDE_travel_time_ratio_l1747_174700

def time_NY_to_SF : ℝ := 24
def layover_time : ℝ := 16
def total_time : ℝ := 58

def time_NO_to_NY : ℝ := total_time - layover_time - time_NY_to_SF

theorem travel_time_ratio : time_NO_to_NY / time_NY_to_SF = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_ratio_l1747_174700


namespace NUMINAMATH_CALUDE_bisection_method_approximation_l1747_174729

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem bisection_method_approximation 
  (h_continuous : Continuous f)
  (h1 : f 0.64 < 0)
  (h2 : f 0.72 > 0)
  (h3 : f 0.68 < 0) :
  ∃ x : ℝ, x ∈ (Set.Ioo 0.68 0.72) ∧ f x = 0 ∧ |x - 0.7| < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_bisection_method_approximation_l1747_174729


namespace NUMINAMATH_CALUDE_student_rank_l1747_174796

theorem student_rank (total : Nat) (rank_right : Nat) (rank_left : Nat) : 
  total = 21 → rank_right = 17 → rank_left = total - rank_right + 1 → rank_left = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_l1747_174796


namespace NUMINAMATH_CALUDE_power_division_equality_l1747_174741

theorem power_division_equality : 3^12 / 27^2 = 729 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l1747_174741


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_equation_l1747_174722

theorem unique_integer_satisfying_equation :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 20200 ∧
  1 + ⌊(200 * n : ℚ) / 201⌋ = ⌈(198 * n : ℚ) / 200⌉ := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_equation_l1747_174722


namespace NUMINAMATH_CALUDE_inequality_proof_l1747_174762

theorem inequality_proof (a b x : ℝ) (h1 : a * b > 0) (h2 : 0 < x) (h3 : x < π / 2) :
  (1 + a^2 / Real.sin x) * (1 + b^2 / Real.cos x) ≥ ((1 + Real.sqrt 2 * a * b)^2 * Real.sin (2 * x)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1747_174762


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_5_l1747_174786

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + a

theorem tangent_parallel_implies_a_equals_5 (a : ℝ) :
  f' a 1 = 7 → a = 5 := by
  sorry

#check tangent_parallel_implies_a_equals_5

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_5_l1747_174786


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l1747_174750

theorem imaginary_part_of_z_plus_reciprocal (z : ℂ) (h : z = 1 - I) :
  (z + z⁻¹).im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l1747_174750


namespace NUMINAMATH_CALUDE_problem_solution_l1747_174705

theorem problem_solution (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) :
  (x + 2 * y) / (x - 2 * y) = -4 / Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1747_174705


namespace NUMINAMATH_CALUDE_sqrt_144000_simplification_l1747_174791

theorem sqrt_144000_simplification : Real.sqrt 144000 = 120 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144000_simplification_l1747_174791


namespace NUMINAMATH_CALUDE_minimum_loads_is_nineteen_l1747_174707

/-- Represents the capacity of the washing machine -/
structure MachineCapacity where
  shirts : ℕ
  sweaters : ℕ
  socks : ℕ

/-- Represents the number of clothes to be washed -/
structure ClothesCount where
  white_shirts : ℕ
  colored_shirts : ℕ
  white_sweaters : ℕ
  colored_sweaters : ℕ
  white_socks : ℕ
  colored_socks : ℕ

/-- Calculates the number of loads required for a given type of clothing -/
def loadsForClothingType (clothes : ℕ) (capacity : ℕ) : ℕ :=
  (clothes + capacity - 1) / capacity

/-- Calculates the total number of loads required -/
def totalLoads (capacity : MachineCapacity) (clothes : ClothesCount) : ℕ :=
  let white_loads := max (loadsForClothingType clothes.white_shirts capacity.shirts)
                         (max (loadsForClothingType clothes.white_sweaters capacity.sweaters)
                              (loadsForClothingType clothes.white_socks capacity.socks))
  let colored_loads := max (loadsForClothingType clothes.colored_shirts capacity.shirts)
                           (max (loadsForClothingType clothes.colored_sweaters capacity.sweaters)
                                (loadsForClothingType clothes.colored_socks capacity.socks))
  white_loads + colored_loads

/-- Theorem: The minimum number of loads required is 19 -/
theorem minimum_loads_is_nineteen (capacity : MachineCapacity) (clothes : ClothesCount) :
  capacity.shirts = 3 ∧ capacity.sweaters = 2 ∧ capacity.socks = 4 ∧
  clothes.white_shirts = 9 ∧ clothes.colored_shirts = 12 ∧
  clothes.white_sweaters = 18 ∧ clothes.colored_sweaters = 20 ∧
  clothes.white_socks = 16 ∧ clothes.colored_socks = 24 →
  totalLoads capacity clothes = 19 := by
  sorry


end NUMINAMATH_CALUDE_minimum_loads_is_nineteen_l1747_174707


namespace NUMINAMATH_CALUDE_three_f_value_l1747_174735

axiom f : ℝ → ℝ
axiom f_def : ∀ x > 0, f (3 * x) = 3 / (3 + x)

theorem three_f_value : ∀ x > 0, 3 * f x = 27 / (9 + x) := by sorry

end NUMINAMATH_CALUDE_three_f_value_l1747_174735


namespace NUMINAMATH_CALUDE_bucket_capacity_l1747_174708

theorem bucket_capacity : ∀ (x : ℚ), 
  (13 * x = 91 * 6) → x = 42 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_l1747_174708


namespace NUMINAMATH_CALUDE_integer_points_on_line_l1747_174780

theorem integer_points_on_line (n : ℕ) (initial_sum final_sum shift : ℤ) 
  (h1 : initial_sum = 25)
  (h2 : final_sum = -35)
  (h3 : shift = 5)
  (h4 : final_sum = initial_sum - n * shift) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_integer_points_on_line_l1747_174780


namespace NUMINAMATH_CALUDE_no_zero_roots_l1747_174766

theorem no_zero_roots : 
  (∀ x : ℝ, 5 * x^2 - 3 = 50 → x ≠ 0) ∧
  (∀ x : ℝ, (3*x - 1)^2 = (x - 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, x^2 - 9 ≥ 0 → 2*x - 2 ≥ 0 → x^2 - 9 = 2*x - 2 → x ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_no_zero_roots_l1747_174766


namespace NUMINAMATH_CALUDE_cookies_per_batch_is_three_l1747_174776

/-- Given the total number of chocolate chips, number of batches, and chips per cookie,
    calculate the number of cookies in a batch. -/
def cookiesPerBatch (totalChips : ℕ) (numBatches : ℕ) (chipsPerCookie : ℕ) : ℕ :=
  (totalChips / numBatches) / chipsPerCookie

/-- Prove that the number of cookies in a batch is 3 given the problem conditions. -/
theorem cookies_per_batch_is_three :
  cookiesPerBatch 81 3 9 = 3 := by
  sorry

#eval cookiesPerBatch 81 3 9

end NUMINAMATH_CALUDE_cookies_per_batch_is_three_l1747_174776


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l1747_174798

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 - y^2/m = 1

-- Define the focus point
def focus : ℝ × ℝ := (-3, 0)

-- Theorem statement
theorem hyperbola_m_value :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), hyperbola_equation x y m → True) ∧ 
    (focus.1^2 = 1 + m) →
    m = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l1747_174798


namespace NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_l1747_174782

def A (m : ℝ) : Set ℝ := {0, m^2}
def B : Set ℝ := {1, 2}

theorem m_equals_one_sufficient_not_necessary :
  (∃ m : ℝ, A m ∩ B = {1} ∧ m ≠ 1) ∧
  (∀ m : ℝ, m = 1 → A m ∩ B = {1}) :=
sorry

end NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_l1747_174782


namespace NUMINAMATH_CALUDE_sum_faces_edges_vertices_l1747_174754

/-- A rectangular prism is a three-dimensional geometric shape. -/
structure RectangularPrism where

/-- The number of faces in a rectangular prism. -/
def faces (rp : RectangularPrism) : ℕ := 6

/-- The number of edges in a rectangular prism. -/
def edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism. -/
def vertices (rp : RectangularPrism) : ℕ := 8

/-- The sum of faces, edges, and vertices in a rectangular prism is 26. -/
theorem sum_faces_edges_vertices (rp : RectangularPrism) :
  faces rp + edges rp + vertices rp = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_faces_edges_vertices_l1747_174754


namespace NUMINAMATH_CALUDE_max_sum_cubes_l1747_174723

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  ∃ (max : ℝ), max = 5 * Real.sqrt 5 ∧ 
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ max ∧
  ∃ (a' b' c' d' e' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 5 ∧
                           a'^3 + b'^3 + c'^3 + d'^3 + e'^3 = max :=
by sorry

end NUMINAMATH_CALUDE_max_sum_cubes_l1747_174723


namespace NUMINAMATH_CALUDE_correct_calculation_l1747_174765

theorem correct_calculation (x : ℝ) : (x + 20) * 5 = 225 → (x + 20) / 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1747_174765


namespace NUMINAMATH_CALUDE_rational_cube_sum_representation_l1747_174781

theorem rational_cube_sum_representation (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_representation_l1747_174781


namespace NUMINAMATH_CALUDE_zoe_money_made_l1747_174772

/-- Calculates the money made from selling chocolate bars -/
def money_made (cost_per_bar : ℕ) (total_bars : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * cost_per_bar

/-- Theorem: Zoe made $42 from selling chocolate bars -/
theorem zoe_money_made :
  let cost_per_bar : ℕ := 6
  let total_bars : ℕ := 13
  let unsold_bars : ℕ := 6
  money_made cost_per_bar total_bars unsold_bars = 42 := by
sorry

end NUMINAMATH_CALUDE_zoe_money_made_l1747_174772


namespace NUMINAMATH_CALUDE_intersection_A_B_l1747_174788

def A : Set ℤ := {x | |x| < 3}
def B : Set ℤ := {x | |x| > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1747_174788


namespace NUMINAMATH_CALUDE_equation_solution_l1747_174714

theorem equation_solution : 
  ∃ x : ℝ, (45 * x) + (625 / 25) - (300 * 4) = 2950 + 1500 / (75 * 2) ∧ x = 4135 / 45 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1747_174714


namespace NUMINAMATH_CALUDE_triangle_angle_and_max_area_l1747_174795

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangleCondition (t : Triangle) : Prop :=
  cos t.B / cos t.C = -t.b / (2 * t.a + t.c)

theorem triangle_angle_and_max_area (t : Triangle) 
  (h : triangleCondition t) : 
  t.B = 2 * π / 3 ∧ 
  (t.b = 3 → ∃ (maxArea : ℝ), maxArea = 3 * sqrt 3 / 4 ∧ 
    ∀ (area : ℝ), area ≤ maxArea) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_and_max_area_l1747_174795


namespace NUMINAMATH_CALUDE_time_ratio_l1747_174777

def minutes_to_seconds (m : ℕ) : ℕ := m * 60

def hours_to_seconds (h : ℕ) : ℕ := h * 3600

def time_period_1 : ℕ := minutes_to_seconds 37 + 48

def time_period_2 : ℕ := hours_to_seconds 2 + minutes_to_seconds 13 + 15

theorem time_ratio : 
  time_period_1 * 7995 = time_period_2 * 2268 := by sorry

end NUMINAMATH_CALUDE_time_ratio_l1747_174777


namespace NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1747_174773

-- Part 1: Equation
theorem equation_solution :
  ∃! x : ℝ, (1 / (x - 3) = 3 + x / (3 - x)) ∧ x = 5 := by sorry

-- Part 2: System of Inequalities
theorem inequalities_solution :
  ∀ x : ℝ, ((x - 1) / 2 < (x + 1) / 3 ∧ x - 3 * (x - 2) ≤ 4) ↔ (1 ≤ x ∧ x < 5) := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1747_174773


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1747_174738

/-- Given a circle and a point on its tangent, prove the x-coordinate of the point. -/
theorem tangent_point_x_coordinate 
  (a : ℝ) -- x-coordinate of point P
  (h1 : (a + 2)^2 + 16 = ((2 : ℝ) * Real.sqrt 3)^2 + 4) -- P is on the tangent and tangent length is 2√3
  : a = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1747_174738


namespace NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l1747_174721

/-- The sum of the infinite series ∑(n=1 to ∞) [3^n / (1 + 3^n + 3^(n+1) + 3^(2n+1))] is equal to 1/4 -/
theorem series_sum_equals_one_fourth :
  let a : ℕ → ℝ := λ n => (3 : ℝ)^n / (1 + (3 : ℝ)^n + (3 : ℝ)^(n+1) + (3 : ℝ)^(2*n+1))
  ∑' n, a n = 1/4 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_fourth_l1747_174721


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1747_174758

theorem diophantine_equation_solutions :
  ∀ m n k : ℕ, 2 * m + 3 * n = k^2 ↔
    (m = 0 ∧ n = 1 ∧ k = 2) ∨
    (m = 3 ∧ n = 0 ∧ k = 3) ∨
    (m = 4 ∧ n = 2 ∧ k = 5) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1747_174758


namespace NUMINAMATH_CALUDE_smallest_n_for_divisibility_l1747_174760

theorem smallest_n_for_divisibility (x y z : ℕ+) 
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3) :
  (∀ n : ℕ, n < 13 → ¬(x * y * z ∣ (x + y + z)^n)) ∧ 
  (x * y * z ∣ (x + y + z)^13) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisibility_l1747_174760


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l1747_174726

theorem roots_of_quadratic (x₁ x₂ : ℝ) : 
  (∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₁ + x₂ = 6 ∧ x₁ * x₂ = -7 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l1747_174726


namespace NUMINAMATH_CALUDE_fraction_division_simplification_l1747_174747

theorem fraction_division_simplification :
  (3 / 4) / (5 / 8) = 6 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_division_simplification_l1747_174747


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1747_174764

theorem quadratic_factorization (a b c : ℤ) : 
  (∀ x, x^2 + 7*x - 18 = (x + a) * (x + b)) →
  (∀ x, x^2 + 11*x + 24 = (x + b) * (x + c)) →
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1747_174764


namespace NUMINAMATH_CALUDE_point_M_coordinates_l1747_174731

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define point M on the left half of x-axis
def point_M (a : ℝ) : Prop := a < 0

-- Define the tangent line from M to circle O
def tangent_line (a x y : ℝ) : Prop := 
  ∃ (t : ℝ), x = a * (1 - t^2) / (1 + t^2) ∧ y = 2 * a * t / (1 + t^2)

-- Define points A, B, and C
def point_A (a x y : ℝ) : Prop := circle_O x y ∧ tangent_line a x y
def point_B (a x y : ℝ) : Prop := circle_O1 x y ∧ tangent_line a x y
def point_C (a x y : ℝ) : Prop := circle_O1 x y ∧ tangent_line a x y ∧ ¬(point_B a x y)

-- Define the condition AB = BC
def equal_segments (a : ℝ) : Prop := 
  ∀ (xa ya xb yb xc yc : ℝ), 
    point_A a xa ya → point_B a xb yb → point_C a xc yc →
    (xa - xb)^2 + (ya - yb)^2 = (xb - xc)^2 + (yb - yc)^2

-- Theorem statement
theorem point_M_coordinates : 
  ∀ (a : ℝ), point_M a → equal_segments a → a = -4 :=
sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l1747_174731


namespace NUMINAMATH_CALUDE_soccer_balls_count_l1747_174799

/-- The number of soccer balls in the gym. -/
def soccer_balls : ℕ := 20

/-- The number of baseballs in the gym. -/
def baseballs : ℕ := 5 * soccer_balls

/-- The number of volleyballs in the gym. -/
def volleyballs : ℕ := 3 * soccer_balls

/-- Theorem stating that the number of soccer balls is 20, given the conditions of the problem. -/
theorem soccer_balls_count :
  soccer_balls = 20 ∧
  baseballs = 5 * soccer_balls ∧
  volleyballs = 3 * soccer_balls ∧
  baseballs + volleyballs = 160 :=
by sorry

end NUMINAMATH_CALUDE_soccer_balls_count_l1747_174799


namespace NUMINAMATH_CALUDE_binomial_prob_two_to_four_out_of_five_l1747_174717

/-- The probability of getting 2, 3, or 4 successes in 5 trials with probability 0.5 each -/
theorem binomial_prob_two_to_four_out_of_five (n : Nat) (p : Real) (X : Nat → Real) :
  n = 5 →
  p = 0.5 →
  (∀ k, X k = Nat.choose n k * p^k * (1 - p)^(n - k)) →
  X 2 + X 3 + X 4 = 25/32 :=
by sorry

end NUMINAMATH_CALUDE_binomial_prob_two_to_four_out_of_five_l1747_174717


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1747_174767

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 1 * a 99 = 16 →
  a 1 + a 99 = 10 →
  a 20 * a 50 * a 80 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1747_174767


namespace NUMINAMATH_CALUDE_junghyeon_stickers_l1747_174761

/-- Given a total of 25 stickers shared between Junghyeon and Yejin, 
    where Junghyeon has 1 more sticker than twice Yejin's, 
    prove that Junghyeon will have 17 stickers. -/
theorem junghyeon_stickers : 
  ∀ (junghyeon_stickers yejin_stickers : ℕ),
  junghyeon_stickers + yejin_stickers = 25 →
  junghyeon_stickers = 2 * yejin_stickers + 1 →
  junghyeon_stickers = 17 := by
sorry

end NUMINAMATH_CALUDE_junghyeon_stickers_l1747_174761


namespace NUMINAMATH_CALUDE_cleaning_payment_l1747_174771

theorem cleaning_payment (payment_per_room : ℚ) (rooms_cleaned : ℚ) (discount_rate : ℚ) :
  payment_per_room = 13/3 →
  rooms_cleaned = 5/2 →
  discount_rate = 1/10 →
  (payment_per_room * rooms_cleaned) * (1 - discount_rate) = 39/4 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_payment_l1747_174771


namespace NUMINAMATH_CALUDE_multiple_of_power_minus_one_l1747_174774

theorem multiple_of_power_minus_one (a b c : ℕ) :
  (∃ k : ℤ, 2^a + 2^b + 1 = k * (2^c - 1)) ↔
  ((a = 0 ∧ b = 0 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3)) :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_power_minus_one_l1747_174774


namespace NUMINAMATH_CALUDE_sqrt_one_third_same_type_as_2sqrt3_l1747_174715

-- Define a function to check if a number is of the same type as 2√3
def isSameTypeAs2Sqrt3 (x : ℝ) : Prop :=
  ∃ (a : ℝ), x = a * Real.sqrt 3

-- Theorem statement
theorem sqrt_one_third_same_type_as_2sqrt3 :
  isSameTypeAs2Sqrt3 (Real.sqrt (1/3)) ∧
  ¬isSameTypeAs2Sqrt3 (Real.sqrt 8) ∧
  ¬isSameTypeAs2Sqrt3 (Real.sqrt 18) ∧
  ¬isSameTypeAs2Sqrt3 (Real.sqrt 9) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_one_third_same_type_as_2sqrt3_l1747_174715


namespace NUMINAMATH_CALUDE_toucans_joined_l1747_174706

theorem toucans_joined (initial final joined : ℕ) : 
  initial = 2 → final = 3 → joined = final - initial :=
by sorry

end NUMINAMATH_CALUDE_toucans_joined_l1747_174706


namespace NUMINAMATH_CALUDE_birthday_ratio_l1747_174709

def peters_candles : ℕ := 10
def ruperts_candles : ℕ := 35

def age_ratio (x y : ℕ) : ℚ := (x : ℚ) / (y : ℚ)

theorem birthday_ratio : 
  age_ratio ruperts_candles peters_candles = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_birthday_ratio_l1747_174709


namespace NUMINAMATH_CALUDE_base_three_to_decimal_l1747_174794

/-- Converts a digit in base 3 to its decimal value -/
def toDecimal (d : Nat) : Nat :=
  if d < 3 then d else 0

/-- Calculates the value of a base-3 number given its digits -/
def baseThreeToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + toDecimal d * 3^i) 0

/-- The decimal representation of 10212 in base 3 -/
def baseThreeNumber : Nat :=
  baseThreeToDecimal [2, 1, 2, 0, 1]

theorem base_three_to_decimal :
  baseThreeNumber = 104 := by sorry

end NUMINAMATH_CALUDE_base_three_to_decimal_l1747_174794


namespace NUMINAMATH_CALUDE_shaded_square_area_l1747_174779

/- Define the structure of the lawn -/
structure Lawn :=
  (total_area : ℝ)
  (rectangle_area : ℝ)
  (is_square : Bool)
  (has_four_rectangles : Bool)
  (has_square_in_rectangle : Bool)

/- Define the properties of the lawn -/
def lawn_properties (l : Lawn) : Prop :=
  l.is_square ∧ 
  l.has_four_rectangles ∧ 
  l.rectangle_area = 40 ∧
  l.has_square_in_rectangle

/- Theorem statement -/
theorem shaded_square_area (l : Lawn) :
  lawn_properties l →
  ∃ (square_area : ℝ), square_area = 2500 / 441 :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_l1747_174779


namespace NUMINAMATH_CALUDE_levis_brother_additional_scores_l1747_174734

/-- Proves that Levi's brother scored 3 more times given the initial conditions and Levi's goal -/
theorem levis_brother_additional_scores :
  ∀ (levi_initial : ℕ) (brother_initial : ℕ) (levi_additional : ℕ) (goal_difference : ℕ),
    levi_initial = 8 →
    brother_initial = 12 →
    levi_additional = 12 →
    goal_difference = 5 →
    ∃ (brother_additional : ℕ),
      levi_initial + levi_additional = brother_initial + brother_additional + goal_difference ∧
      brother_additional = 3 :=
by sorry

end NUMINAMATH_CALUDE_levis_brother_additional_scores_l1747_174734


namespace NUMINAMATH_CALUDE_power_of_64_two_thirds_l1747_174769

theorem power_of_64_two_thirds : (64 : ℝ) ^ (2/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_two_thirds_l1747_174769


namespace NUMINAMATH_CALUDE_prob_red_then_black_custom_deck_l1747_174783

/-- A deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The probability of drawing a red card first and then a black card from a shuffled deck -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) * (d.black_cards : ℚ) / ((d.total_cards : ℚ) * (d.total_cards - 1 : ℚ))

/-- The theorem stating the probability for the given deck -/
theorem prob_red_then_black_custom_deck :
  let d : Deck := ⟨60, 30, 30⟩
  prob_red_then_black d = 15 / 59 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_custom_deck_l1747_174783


namespace NUMINAMATH_CALUDE_factor_expression_l1747_174701

theorem factor_expression (b : ℝ) : 180 * b^2 + 36 * b = 36 * b * (5 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1747_174701


namespace NUMINAMATH_CALUDE_fish_tank_leakage_rate_l1747_174711

/-- Proves that the rate of leakage is 1.5 ounces per hour given the problem conditions -/
theorem fish_tank_leakage_rate 
  (bucket_capacity : ℝ) 
  (leakage_duration : ℝ) 
  (h1 : bucket_capacity = 36) 
  (h2 : leakage_duration = 12) 
  (h3 : bucket_capacity = 2 * (leakage_duration * leakage_rate)) : 
  leakage_rate = 1.5 := by
  sorry

#check fish_tank_leakage_rate

end NUMINAMATH_CALUDE_fish_tank_leakage_rate_l1747_174711


namespace NUMINAMATH_CALUDE_paint_contribution_is_360_l1747_174759

/-- Calculates the contribution of each person for the paint cost --/
def calculate_contribution (paint_cost_per_gallon : ℚ) (coverage_per_gallon : ℚ)
  (jason_wall_area : ℚ) (jason_coats : ℕ) (jeremy_wall_area : ℚ) (jeremy_coats : ℕ) : ℚ :=
  let total_area := jason_wall_area * jason_coats + jeremy_wall_area * jeremy_coats
  let gallons_needed := (total_area / coverage_per_gallon).ceil
  let total_cost := gallons_needed * paint_cost_per_gallon
  total_cost / 2

/-- Theorem stating that each person's contribution is $360 --/
theorem paint_contribution_is_360 :
  calculate_contribution 45 400 1025 3 1575 2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_paint_contribution_is_360_l1747_174759


namespace NUMINAMATH_CALUDE_thomas_work_hours_l1747_174756

theorem thomas_work_hours 
  (total_hours : ℕ)
  (rebecca_hours : ℕ)
  (h1 : total_hours = 157)
  (h2 : rebecca_hours = 56) :
  ∃ (thomas_hours : ℕ),
    thomas_hours = 37 ∧
    ∃ (toby_hours : ℕ),
      toby_hours = 2 * thomas_hours - 10 ∧
      rebecca_hours = toby_hours - 8 ∧
      total_hours = thomas_hours + toby_hours + rebecca_hours :=
by sorry

end NUMINAMATH_CALUDE_thomas_work_hours_l1747_174756


namespace NUMINAMATH_CALUDE_cleaning_event_calculation_l1747_174770

def total_members : ℕ := 2000
def adult_men_percentage : ℚ := 30 / 100
def senior_percentage : ℚ := 5 / 100
def child_teen_ratio : ℚ := 3 / 2
def child_collection_rate : ℚ := 3 / 2
def teen_collection_rate : ℕ := 3
def senior_collection_rate : ℕ := 1

theorem cleaning_event_calculation :
  let adult_men := (adult_men_percentage * total_members).floor
  let adult_women := 2 * adult_men
  let seniors := (senior_percentage * total_members).floor
  let children_and_teens := total_members - (adult_men + adult_women + seniors)
  let children := ((child_teen_ratio * children_and_teens) / (1 + child_teen_ratio)).floor
  let teenagers := children_and_teens - children
  ∃ (children teenagers : ℕ) (recyclable mixed various : ℚ),
    children = 60 ∧
    teenagers = 40 ∧
    recyclable = child_collection_rate * children ∧
    mixed = teen_collection_rate * teenagers ∧
    various = senior_collection_rate * seniors ∧
    recyclable = 90 ∧
    mixed = 120 ∧
    various = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_cleaning_event_calculation_l1747_174770


namespace NUMINAMATH_CALUDE_find_constant_k_l1747_174784

theorem find_constant_k (c : ℝ) (k : ℝ) :
  c = 2 →
  (∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - c)*(x - 4)) →
  k = -16 := by
  sorry

end NUMINAMATH_CALUDE_find_constant_k_l1747_174784


namespace NUMINAMATH_CALUDE_johnny_distance_when_met_l1747_174713

/-- The distance between Q and Y in km -/
def total_distance : ℝ := 45

/-- Matthew's walking rate in km/hour -/
def matthew_rate : ℝ := 3

/-- Johnny's walking rate in km/hour -/
def johnny_rate : ℝ := 4

/-- The time difference between Matthew's and Johnny's start in hours -/
def time_difference : ℝ := 1

/-- The distance Johnny walked when they met -/
def johnny_distance : ℝ := 24

theorem johnny_distance_when_met :
  let t := (total_distance - matthew_rate * time_difference) / (matthew_rate + johnny_rate)
  johnny_distance = johnny_rate * t :=
by sorry

end NUMINAMATH_CALUDE_johnny_distance_when_met_l1747_174713


namespace NUMINAMATH_CALUDE_fraction_sum_zero_implies_one_zero_l1747_174778

theorem fraction_sum_zero_implies_one_zero (a b c : ℝ) :
  (a - b) / (1 + a * b) + (b - c) / (1 + b * c) + (c - a) / (1 + c * a) = 0 →
  (a - b) / (1 + a * b) = 0 ∨ (b - c) / (1 + b * c) = 0 ∨ (c - a) / (1 + c * a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_implies_one_zero_l1747_174778


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1747_174724

theorem solution_set_of_inequality (x : ℝ) :
  (x * (x + 2) / (x - 3) < 0) ↔ (x < -2 ∨ (0 < x ∧ x < 3)) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1747_174724


namespace NUMINAMATH_CALUDE_crayon_factory_output_l1747_174744

/-- Calculates the number of boxes filled per hour in a crayon factory --/
def boxes_per_hour (num_colors : ℕ) (crayons_per_color_per_box : ℕ) (total_crayons_in_4_hours : ℕ) : ℕ :=
  let crayons_per_hour := total_crayons_in_4_hours / 4
  let crayons_per_box := num_colors * crayons_per_color_per_box
  crayons_per_hour / crayons_per_box

/-- Theorem stating that under given conditions, the factory fills 5 boxes per hour --/
theorem crayon_factory_output : 
  boxes_per_hour 4 2 160 = 5 := by
  sorry

end NUMINAMATH_CALUDE_crayon_factory_output_l1747_174744


namespace NUMINAMATH_CALUDE_rals_age_l1747_174733

/-- Given that Ral's age is twice Suri's age and Suri's age plus 3 years equals 16 years,
    prove that Ral's current age is 26 years. -/
theorem rals_age (suri_age : ℕ) (ral_age : ℕ) : 
  ral_age = 2 * suri_age → 
  suri_age + 3 = 16 → 
  ral_age = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_rals_age_l1747_174733


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1747_174739

theorem other_root_of_quadratic (m : ℚ) :
  (3 : ℚ) ∈ {x : ℚ | 3 * x^2 + m * x = 5} →
  (-5/9 : ℚ) ∈ {x : ℚ | 3 * x^2 + m * x = 5} :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1747_174739


namespace NUMINAMATH_CALUDE_f_2017_equals_neg_2_l1747_174792

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_2017_equals_neg_2 (f : ℝ → ℝ) 
  (h1 : is_odd_function f)
  (h2 : is_even_function (fun x ↦ f (x + 1)))
  (h3 : f (-1) = 2) : 
  f 2017 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2017_equals_neg_2_l1747_174792


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l1747_174753

theorem complex_number_imaginary_part (a : ℝ) :
  let z : ℂ := (1 - a * Complex.I) / (1 - Complex.I)
  Complex.im z = 4 → a = -7 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l1747_174753


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1747_174768

theorem fraction_equivalence (k : ℝ) (h : k ≠ -5) :
  (k + 3) / (k + 5) = 3 / 5 ↔ k = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1747_174768


namespace NUMINAMATH_CALUDE_algorithm_output_l1747_174728

theorem algorithm_output (x : ℤ) (y z : ℕ) : 
  x = -3 → 
  y = Int.natAbs x → 
  z = 2^y - y → 
  z = 5 := by sorry

end NUMINAMATH_CALUDE_algorithm_output_l1747_174728


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l1747_174742

theorem systematic_sampling_removal (total_students sample_size : ℕ) 
  (h1 : total_students = 1252)
  (h2 : sample_size = 50) :
  ∃ (removed : ℕ), 
    removed = 2 ∧ 
    (total_students - removed) % sample_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l1747_174742


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l1747_174712

theorem solution_set_of_equation (x : ℝ) : 
  (16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x) ↔ (x = 1/4 ∨ x = -1/4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l1747_174712


namespace NUMINAMATH_CALUDE_probability_two_girls_l1747_174727

/-- The probability of choosing two girls from a class with given composition -/
theorem probability_two_girls (total : ℕ) (girls : ℕ) (boys : ℕ) 
  (h1 : total = girls + boys) 
  (h2 : total = 8) 
  (h3 : girls = 5) 
  (h4 : boys = 3) : 
  (Nat.choose girls 2 : ℚ) / (Nat.choose total 2) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l1747_174727


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1747_174704

/-- A point in the second quadrant has a negative x-coordinate and positive y-coordinate -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The solution set of the inequality (2-m)x + 2 > m with respect to x -/
def solution_set (m : ℝ) : Set ℝ := {x | (2 - m) * x + 2 > m}

theorem inequality_solution_set (m : ℝ) :
  second_quadrant (3 - m) 1 → solution_set m = {x | x < -1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1747_174704


namespace NUMINAMATH_CALUDE_water_mixture_adjustment_l1747_174710

theorem water_mixture_adjustment (initial_volume : ℝ) (initial_water_percentage : ℝ) 
  (initial_acid_percentage : ℝ) (water_to_add : ℝ) (final_water_percentage : ℝ) 
  (final_acid_percentage : ℝ) : 
  initial_volume = 300 →
  initial_water_percentage = 0.60 →
  initial_acid_percentage = 0.40 →
  water_to_add = 100 →
  final_water_percentage = 0.70 →
  final_acid_percentage = 0.30 →
  (initial_volume * initial_water_percentage + water_to_add) / (initial_volume + water_to_add) = final_water_percentage ∧
  (initial_volume * initial_acid_percentage) / (initial_volume + water_to_add) = final_acid_percentage :=
by sorry

end NUMINAMATH_CALUDE_water_mixture_adjustment_l1747_174710


namespace NUMINAMATH_CALUDE_choose_five_representatives_choose_five_with_specific_girl_choose_five_with_at_least_two_boys_divide_into_three_groups_l1747_174748

def num_boys : ℕ := 4
def num_girls : ℕ := 5
def total_people : ℕ := num_boys + num_girls

-- Question 1
theorem choose_five_representatives : Nat.choose total_people 5 = 126 := by sorry

-- Question 2
theorem choose_five_with_specific_girl :
  (Nat.choose num_boys 2) * (Nat.choose (num_girls - 1) 2) = 36 := by sorry

-- Question 3
theorem choose_five_with_at_least_two_boys :
  (Nat.choose num_boys 2) * (Nat.choose num_girls 3) +
  (Nat.choose num_boys 3) * (Nat.choose num_girls 2) +
  (Nat.choose num_boys 4) * (Nat.choose num_girls 1) = 105 := by sorry

-- Question 4
theorem divide_into_three_groups :
  (Nat.choose total_people 4) * (Nat.choose (total_people - 4) 3) = 1260 := by sorry

end NUMINAMATH_CALUDE_choose_five_representatives_choose_five_with_specific_girl_choose_five_with_at_least_two_boys_divide_into_three_groups_l1747_174748


namespace NUMINAMATH_CALUDE_darcie_father_age_l1747_174785

def darcie_age : ℕ := 4

theorem darcie_father_age (mother_age father_age : ℕ) 
  (h1 : darcie_age = mother_age / 6)
  (h2 : mother_age * 5 = father_age * 4) : 
  father_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_darcie_father_age_l1747_174785


namespace NUMINAMATH_CALUDE_segment_AB_length_l1747_174757

-- Define the points on the number line
def point_A : ℝ := -5
def point_B : ℝ := 2

-- Define the length of the segment
def segment_length (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem segment_AB_length :
  segment_length point_A point_B = 7 := by
  sorry

end NUMINAMATH_CALUDE_segment_AB_length_l1747_174757


namespace NUMINAMATH_CALUDE_smallest_k_satisfying_inequality_l1747_174787

theorem smallest_k_satisfying_inequality (n m : ℕ) (hn : n > 0) (hm : 0 < m ∧ m ≤ 5) :
  (∀ k : ℕ, k % 3 = 0 → (64^k + 32^m > 4^(16 + n^2) → k ≥ 6)) ∧
  (64^6 + 32^m > 4^(16 + n^2)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_satisfying_inequality_l1747_174787


namespace NUMINAMATH_CALUDE_divisibility_implies_fraction_simplification_l1747_174736

theorem divisibility_implies_fraction_simplification (a b c : ℕ) :
  (100 * a + 10 * b + c) % 7 = 0 →
  ((10 * b + c + 16 * a) % 7 = 0 ∧ (10 * b + c - 61 * a) % 7 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_fraction_simplification_l1747_174736


namespace NUMINAMATH_CALUDE_high_precision_census_suitability_l1747_174755

/-- Represents different types of surveys --/
inductive SurveyType
  | DestructiveTesting
  | WideScopePopulation
  | HighPrecisionRequired
  | LargeAudienceSampling

/-- Represents different survey methods --/
inductive SurveyMethod
  | Census
  | Sampling

/-- Defines the characteristics of a survey --/
structure Survey where
  type : SurveyType
  method : SurveyMethod

/-- Defines the suitability of a survey method for a given survey type --/
def is_suitable (s : Survey) : Prop :=
  match s.type, s.method with
  | SurveyType.HighPrecisionRequired, SurveyMethod.Census => true
  | SurveyType.DestructiveTesting, SurveyMethod.Sampling => true
  | SurveyType.WideScopePopulation, SurveyMethod.Sampling => true
  | SurveyType.LargeAudienceSampling, SurveyMethod.Sampling => true
  | _, _ => false

/-- Theorem: A survey requiring high precision is most suitable for a census method --/
theorem high_precision_census_suitability :
  ∀ (s : Survey), s.type = SurveyType.HighPrecisionRequired → 
  is_suitable { type := s.type, method := SurveyMethod.Census } = true :=
by
  sorry


end NUMINAMATH_CALUDE_high_precision_census_suitability_l1747_174755


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1747_174743

theorem complex_number_modulus (a : ℝ) (h1 : a < 0) :
  let z : ℂ := (3 * a * Complex.I) / (1 - 2 * Complex.I)
  Complex.abs z = Real.sqrt 5 → a = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1747_174743


namespace NUMINAMATH_CALUDE_stratified_sampling_total_l1747_174716

/-- Calculates the total number of students sampled using stratified sampling -/
def totalSampleSize (firstGradeTotal : ℕ) (secondGradeTotal : ℕ) (thirdGradeTotal : ℕ) (firstGradeSample : ℕ) : ℕ :=
  let totalStudents := firstGradeTotal + secondGradeTotal + thirdGradeTotal
  (firstGradeSample * totalStudents) / firstGradeTotal

theorem stratified_sampling_total (firstGradeTotal : ℕ) (secondGradeTotal : ℕ) (thirdGradeTotal : ℕ) (firstGradeSample : ℕ)
    (h1 : firstGradeTotal = 600)
    (h2 : secondGradeTotal = 500)
    (h3 : thirdGradeTotal = 400)
    (h4 : firstGradeSample = 30) :
    totalSampleSize firstGradeTotal secondGradeTotal thirdGradeTotal firstGradeSample = 75 := by
  sorry

#eval totalSampleSize 600 500 400 30

end NUMINAMATH_CALUDE_stratified_sampling_total_l1747_174716


namespace NUMINAMATH_CALUDE_erased_number_proof_l1747_174737

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  n > 2 →
  (↑n * (↑n + 1) / 2 - 3) - x = (454 / 9 : ℚ) * (↑n - 1) →
  x = 107 :=
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l1747_174737


namespace NUMINAMATH_CALUDE_second_derivative_parametric_function_l1747_174751

noncomputable def x (t : ℝ) : ℝ := Real.cosh t

noncomputable def y (t : ℝ) : ℝ := (Real.sinh t) ^ (2/3)

theorem second_derivative_parametric_function (t : ℝ) :
  let x_t' := Real.sinh t
  let y_t' := (2 * Real.cosh t) / (3 * (Real.sinh t)^(1/3))
  let y_x' := y_t' / x_t'
  let y_x'_t' := -2 * (3 + Real.cosh t ^ 2) / (9 * Real.sinh t ^ 3)
  (y_x'_t' / x_t') = -2 * (3 + Real.cosh t ^ 2) / (9 * Real.sinh t ^ 4) :=
by sorry

end NUMINAMATH_CALUDE_second_derivative_parametric_function_l1747_174751


namespace NUMINAMATH_CALUDE_savings_proof_l1747_174725

/-- Calculates savings given income and expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that savings are 4000 given the conditions -/
theorem savings_proof (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (h1 : income = 20000)
  (h2 : income_ratio = 5)
  (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 4000 := by
  sorry

#eval calculate_savings 20000 5 4

end NUMINAMATH_CALUDE_savings_proof_l1747_174725


namespace NUMINAMATH_CALUDE_vector_sum_l1747_174749

def vector_a : ℝ × ℝ := (2, 0)
def vector_b : ℝ × ℝ := (-1, -2)

theorem vector_sum : vector_a + vector_b = (1, -2) := by sorry

end NUMINAMATH_CALUDE_vector_sum_l1747_174749


namespace NUMINAMATH_CALUDE_min_magnitude_linear_combination_l1747_174745

/-- Given vectors a and b in ℝ², prove that the minimum magnitude of their linear combination c = xa + yb is √3, under specific conditions. -/
theorem min_magnitude_linear_combination (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1) (h3 : a • b = 1/2) :
  ∃ (min : ℝ), min = Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 2 → 
  ‖x • a + y • b‖ ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_magnitude_linear_combination_l1747_174745


namespace NUMINAMATH_CALUDE_basketball_scores_second_half_total_l1747_174719

/-- Represents the score of a team in a quarter -/
structure QuarterScore :=
  (score : ℕ)

/-- Represents the scores of a team for all four quarters -/
structure GameScore :=
  (q1 : QuarterScore)
  (q2 : QuarterScore)
  (q3 : QuarterScore)
  (q4 : QuarterScore)

/-- Checks if a sequence of four numbers forms a geometric sequence -/
def isGeometricSequence (a b c d : ℕ) : Prop :=
  ∃ (r : ℚ), b = a * r ∧ c = b * r ∧ d = c * r

/-- Checks if a sequence of four numbers forms an arithmetic sequence -/
def isArithmeticSequence (a b c d : ℕ) : Prop :=
  ∃ (diff : ℤ), b = a + diff ∧ c = b + diff ∧ d = c + diff

/-- The main theorem statement -/
theorem basketball_scores_second_half_total
  (eagles : GameScore)
  (lions : GameScore)
  (h1 : eagles.q1.score = lions.q1.score)
  (h2 : eagles.q1.score + eagles.q2.score = lions.q1.score + lions.q2.score)
  (h3 : isGeometricSequence eagles.q1.score eagles.q2.score eagles.q3.score eagles.q4.score)
  (h4 : isArithmeticSequence lions.q1.score lions.q2.score lions.q3.score lions.q4.score)
  (h5 : eagles.q1.score + eagles.q2.score + eagles.q3.score + eagles.q4.score = 
        lions.q1.score + lions.q2.score + lions.q3.score + lions.q4.score + 1)
  (h6 : eagles.q1.score + eagles.q2.score + eagles.q3.score + eagles.q4.score ≤ 100)
  (h7 : lions.q1.score + lions.q2.score + lions.q3.score + lions.q4.score ≤ 100) :
  eagles.q3.score + eagles.q4.score + lions.q3.score + lions.q4.score = 109 :=
sorry

end NUMINAMATH_CALUDE_basketball_scores_second_half_total_l1747_174719


namespace NUMINAMATH_CALUDE_salary_calculation_l1747_174775

/-- Prove that if a salary is first increased by 10% and then decreased by 5%,
    resulting in Rs. 4180, the original salary was Rs. 4000. -/
theorem salary_calculation (original : ℝ) : 
  (original * 1.1 * 0.95 = 4180) → original = 4000 := by
  sorry

end NUMINAMATH_CALUDE_salary_calculation_l1747_174775


namespace NUMINAMATH_CALUDE_product_13_factor_l1747_174720

theorem product_13_factor (w : ℕ+) (h1 : w ≥ 468) 
  (h2 : ∃ (k : ℕ), 2^4 * 3^3 * k = 1452 * w) : 
  (∃ (m : ℕ), 13^1 * m = 1452 * w) ∧ 
  (∀ (n : ℕ), n > 1 → ¬(∃ (m : ℕ), 13^n * m = 1452 * w)) :=
sorry

end NUMINAMATH_CALUDE_product_13_factor_l1747_174720


namespace NUMINAMATH_CALUDE_inequality_comparison_l1747_174703

theorem inequality_comparison : 
  (¬ (0 < -1/2)) ∧ 
  (¬ (4/5 < -6/7)) ∧ 
  (9/8 > 8/9) ∧ 
  (¬ (-4 > -3)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_comparison_l1747_174703


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1747_174763

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 7*p - 1 = 0 ∧ 
  q^3 - 6*q^2 + 7*q - 1 = 0 ∧ 
  r^3 - 6*r^2 + 7*r - 1 = 0 →
  p / (q*r + 1) + q / (p*r + 1) + r / (p*q + 1) = 61/15 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1747_174763


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l1747_174752

theorem successive_discounts_equivalence (original_price : ℝ) 
  (first_discount second_discount : ℝ) (equivalent_discount : ℝ) : 
  original_price = 50 ∧ 
  first_discount = 0.15 ∧ 
  second_discount = 0.10 ∧ 
  equivalent_discount = 0.235 →
  original_price * (1 - first_discount) * (1 - second_discount) = 
  original_price * (1 - equivalent_discount) := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l1747_174752


namespace NUMINAMATH_CALUDE_no_cube_root_exists_l1747_174702

theorem no_cube_root_exists (n : ℤ) : ¬ ∃ k : ℤ, k^3 = 3*n^2 + 3*n + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_cube_root_exists_l1747_174702


namespace NUMINAMATH_CALUDE_dollar_three_neg_one_l1747_174732

def dollar (x y : Int) : Int :=
  x * (y + 2) + x * y - 5

theorem dollar_three_neg_one : dollar 3 (-1) = -5 := by
  sorry

end NUMINAMATH_CALUDE_dollar_three_neg_one_l1747_174732


namespace NUMINAMATH_CALUDE_power_inequality_l1747_174790

theorem power_inequality : 81^31 > 27^41 ∧ 27^41 > 9^61 := by sorry

end NUMINAMATH_CALUDE_power_inequality_l1747_174790


namespace NUMINAMATH_CALUDE_sin_m_theta_bound_l1747_174793

theorem sin_m_theta_bound (θ : ℝ) (m : ℕ) : 
  |Real.sin (m * θ)| ≤ m * |Real.sin θ| := by
  sorry

end NUMINAMATH_CALUDE_sin_m_theta_bound_l1747_174793


namespace NUMINAMATH_CALUDE_range_of_half_difference_l1747_174797

theorem range_of_half_difference (α β : ℝ) 
  (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) :
  ∀ x, x ∈ Set.Icc (-π/2) 0 ↔ ∃ α β, -π/2 ≤ α ∧ α < β ∧ β ≤ π/2 ∧ x = (α - β)/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_half_difference_l1747_174797


namespace NUMINAMATH_CALUDE_jason_total_cards_l1747_174740

/-- The number of Pokemon cards Jason has after receiving new ones from Alyssa -/
def total_cards (initial_cards new_cards : ℕ) : ℕ :=
  initial_cards + new_cards

/-- Theorem stating that Jason's total cards is 900 given the initial and new card counts -/
theorem jason_total_cards :
  total_cards 676 224 = 900 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_cards_l1747_174740
