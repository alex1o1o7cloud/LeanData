import Mathlib

namespace NUMINAMATH_CALUDE_best_route_is_D_l908_90894

-- Define the structure for a route
structure Route where
  name : String
  baseTime : ℕ
  numLights : ℕ
  redLightTime : ℕ
  trafficDensity : String
  weatherCondition : String
  roadCondition : String

-- Define the routes
def routeA : Route := {
  name := "A",
  baseTime := 10,
  numLights := 3,
  redLightTime := 3,
  trafficDensity := "moderate",
  weatherCondition := "light rain",
  roadCondition := "good"
}

def routeB : Route := {
  name := "B",
  baseTime := 12,
  numLights := 4,
  redLightTime := 2,
  trafficDensity := "high",
  weatherCondition := "clear",
  roadCondition := "pothole"
}

def routeC : Route := {
  name := "C",
  baseTime := 11,
  numLights := 2,
  redLightTime := 4,
  trafficDensity := "low",
  weatherCondition := "clear",
  roadCondition := "construction"
}

def routeD : Route := {
  name := "D",
  baseTime := 14,
  numLights := 0,
  redLightTime := 0,
  trafficDensity := "medium",
  weatherCondition := "potential fog",
  roadCondition := "unknown"
}

-- Define the list of all routes
def allRoutes : List Route := [routeA, routeB, routeC, routeD]

-- Calculate the worst-case travel time for a route
def worstCaseTime (r : Route) : ℕ := r.baseTime + r.numLights * r.redLightTime

-- Define the theorem
theorem best_route_is_D :
  ∀ r ∈ allRoutes, worstCaseTime routeD ≤ worstCaseTime r :=
sorry

end NUMINAMATH_CALUDE_best_route_is_D_l908_90894


namespace NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l908_90804

theorem angle_measure_in_acute_triangle (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ A < Real.pi/2 →
  0 < B ∧ B < Real.pi/2 →
  0 < C ∧ C < Real.pi/2 →
  A + B + C = Real.pi →
  a = Real.sin B * (Real.sin C / Real.sin A) →
  b = Real.sin C * (Real.sin A / Real.sin B) →
  2 * a * Real.sin B = Real.sqrt 3 * b →
  A = Real.pi/3 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_acute_triangle_l908_90804


namespace NUMINAMATH_CALUDE_M_divisible_by_49_l908_90818

/-- M is the concatenated number formed by writing integers from 1 to 48 in order -/
def M : ℕ := sorry

/-- Theorem stating that M is divisible by 49 -/
theorem M_divisible_by_49 : 49 ∣ M := by sorry

end NUMINAMATH_CALUDE_M_divisible_by_49_l908_90818


namespace NUMINAMATH_CALUDE_cube_volume_increase_l908_90813

theorem cube_volume_increase (s : ℝ) (h : s > 0) : 
  let new_edge := 1.6 * s
  let original_volume := s^3
  let new_volume := new_edge^3
  (new_volume - original_volume) / original_volume * 100 = 309.6 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_increase_l908_90813


namespace NUMINAMATH_CALUDE_proposition_equivalences_l908_90802

theorem proposition_equivalences (x y : ℝ) : 
  (((Real.sqrt (x - 2) + (y + 1)^2 = 0) → (x = 2 ∧ y = -1)) ↔
   ((x = 2 ∧ y = -1) → (Real.sqrt (x - 2) + (y + 1)^2 = 0))) ∧
  (((Real.sqrt (x - 2) + (y + 1)^2 ≠ 0) → (x ≠ 2 ∨ y ≠ -1)) ↔
   ((x ≠ 2 ∨ y ≠ -1) → (Real.sqrt (x - 2) + (y + 1)^2 ≠ 0))) :=
by sorry

#check proposition_equivalences

end NUMINAMATH_CALUDE_proposition_equivalences_l908_90802


namespace NUMINAMATH_CALUDE_largest_angle_of_special_quadrilateral_l908_90811

/-- A convex quadrilateral is rude if there exists a convex quadrilateral inside or on its sides
    with a larger sum of diagonals. -/
def IsRude (Q : Set (ℝ × ℝ)) : Prop := sorry

/-- The largest angle of a quadrilateral -/
def LargestAngle (Q : Set (ℝ × ℝ)) : ℝ := sorry

/-- A convex quadrilateral -/
def ConvexQuadrilateral (Q : Set (ℝ × ℝ)) : Prop := sorry

theorem largest_angle_of_special_quadrilateral 
  (A B C D : ℝ × ℝ) 
  (r : ℝ)
  (h_convex : ConvexQuadrilateral {A, B, C, D})
  (h_not_rude : ¬IsRude {A, B, C, D})
  (h_r_positive : r > 0)
  (h_nearby_rude : ∀ A', A' ≠ A → dist A' A ≤ r → IsRude {A', B, C, D}) :
  LargestAngle {A, B, C, D} = 150 * π / 180 := by sorry

end NUMINAMATH_CALUDE_largest_angle_of_special_quadrilateral_l908_90811


namespace NUMINAMATH_CALUDE_exponents_gp_iff_n_3_6_10_l908_90830

/-- A function that returns the sequence of exponents in the prime factorization of n! --/
def exponents_of_factorial (n : ℕ) : List ℕ :=
  sorry

/-- Check if a list of natural numbers forms a geometric progression --/
def is_geometric_progression (l : List ℕ) : Prop :=
  sorry

/-- The main theorem stating that the exponents in the prime factorization of n!
    form a geometric progression if and only if n is 3, 6, or 10 --/
theorem exponents_gp_iff_n_3_6_10 (n : ℕ) :
  n ≥ 3 → (is_geometric_progression (exponents_of_factorial n) ↔ n = 3 ∨ n = 6 ∨ n = 10) :=
sorry

end NUMINAMATH_CALUDE_exponents_gp_iff_n_3_6_10_l908_90830


namespace NUMINAMATH_CALUDE_correct_statements_count_l908_90879

/-- Represents a statement about sampling methods -/
inductive SamplingStatement
| SimpleRandomSmallPopulation
| SystematicSamplingMethod
| LotteryDrawingLots
| SystematicSamplingEqualProbability

/-- Checks if a given sampling statement is correct -/
def is_correct (s : SamplingStatement) : Bool :=
  match s with
  | SamplingStatement.SimpleRandomSmallPopulation => true
  | SamplingStatement.SystematicSamplingMethod => false
  | SamplingStatement.LotteryDrawingLots => true
  | SamplingStatement.SystematicSamplingEqualProbability => true

/-- The list of all sampling statements -/
def all_statements : List SamplingStatement :=
  [SamplingStatement.SimpleRandomSmallPopulation,
   SamplingStatement.SystematicSamplingMethod,
   SamplingStatement.LotteryDrawingLots,
   SamplingStatement.SystematicSamplingEqualProbability]

/-- Theorem stating that the number of correct sampling statements is 3 -/
theorem correct_statements_count :
  (all_statements.filter is_correct).length = 3 := by sorry

end NUMINAMATH_CALUDE_correct_statements_count_l908_90879


namespace NUMINAMATH_CALUDE_function_minimum_and_integer_bound_l908_90812

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a + Real.log x)

theorem function_minimum_and_integer_bound :
  (∃ a : ℝ, ∀ x > 0, f a x ≥ -Real.exp (-2) ∧ ∃ x₀ > 0, f a x₀ = -Real.exp (-2)) →
  (∃ a : ℝ, a = 1 ∧
    ∀ k : ℤ, (∀ x > 1, ↑k < (f a x) / (x - 1)) →
      k ≤ 3 ∧ (∃ x > 1, 3 < (f a x) / (x - 1))) :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_and_integer_bound_l908_90812


namespace NUMINAMATH_CALUDE_sum_complex_exp_argument_l908_90856

/-- The argument of the sum of five complex exponentials -/
theorem sum_complex_exp_argument :
  let z : ℂ := Complex.exp (11 * Real.pi * Complex.I / 100) +
               Complex.exp (31 * Real.pi * Complex.I / 100) +
               Complex.exp (51 * Real.pi * Complex.I / 100) +
               Complex.exp (71 * Real.pi * Complex.I / 100) +
               Complex.exp (91 * Real.pi * Complex.I / 100)
  0 ≤ Complex.arg z ∧ Complex.arg z < 2 * Real.pi →
  Complex.arg z = 51 * Real.pi / 100 :=
by sorry

end NUMINAMATH_CALUDE_sum_complex_exp_argument_l908_90856


namespace NUMINAMATH_CALUDE_fifteen_factorial_representation_l908_90828

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem fifteen_factorial_representation (X Y Z : ℕ) :
  X < 10 ∧ Y < 10 ∧ Z < 10 →
  factorial 15 = 1307674300000000 + X * 100000000 + Y * 10000 + Z * 100 →
  X + Y + Z = 0 := by
sorry

end NUMINAMATH_CALUDE_fifteen_factorial_representation_l908_90828


namespace NUMINAMATH_CALUDE_candy_has_nine_pencils_l908_90826

/-- The number of pencils each person has -/
structure PencilCounts where
  calen : ℕ
  caleb : ℕ
  candy : ℕ
  darlene : ℕ

/-- The conditions of the pencil problem -/
def PencilProblem (p : PencilCounts) : Prop :=
  p.calen = p.caleb + 5 ∧
  p.caleb = 2 * p.candy - 3 ∧
  p.darlene = p.calen + p.caleb + p.candy + 4 ∧
  p.calen - 10 = 10

/-- The theorem stating that under the given conditions, Candy has 9 pencils -/
theorem candy_has_nine_pencils (p : PencilCounts) (h : PencilProblem p) : p.candy = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_has_nine_pencils_l908_90826


namespace NUMINAMATH_CALUDE_average_xyz_in_terms_of_k_l908_90824

theorem average_xyz_in_terms_of_k (x y z k : ℝ) 
  (eq1 : 2 * x + y - z = 26)
  (eq2 : x + 2 * y + z = 10)
  (eq3 : x - y + z = k) :
  (x + y + z) / 3 = (36 + k) / 6 := by
  sorry

end NUMINAMATH_CALUDE_average_xyz_in_terms_of_k_l908_90824


namespace NUMINAMATH_CALUDE_substitution_result_l908_90816

theorem substitution_result (x y : ℝ) :
  y = x - 1 ∧ x + 2*y = 7 → x + 2*x - 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_substitution_result_l908_90816


namespace NUMINAMATH_CALUDE_pat_to_mark_ratio_project_hours_ratio_l908_90849

/-- Represents the hours charged by each person --/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Defines the conditions of the problem --/
def satisfiesConditions (hours : ProjectHours) : Prop :=
  hours.pat + hours.kate + hours.mark = 117 ∧
  hours.pat = 2 * hours.kate ∧
  hours.mark = hours.kate + 65

/-- Theorem stating the ratio of Pat's hours to Mark's hours --/
theorem pat_to_mark_ratio (hours : ProjectHours) 
  (h : satisfiesConditions hours) : 
  hours.pat * 3 = hours.mark * 1 := by
  sorry

/-- Main theorem proving the ratio is 1:3 --/
theorem project_hours_ratio : 
  ∃ hours : ProjectHours, satisfiesConditions hours ∧ hours.pat * 3 = hours.mark * 1 := by
  sorry

end NUMINAMATH_CALUDE_pat_to_mark_ratio_project_hours_ratio_l908_90849


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l908_90853

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l908_90853


namespace NUMINAMATH_CALUDE_intersects_both_branches_iff_l908_90810

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a line with slope k passing through a point -/
structure Line where
  k : ℝ

/-- Predicate indicating if a line intersects both branches of a hyperbola -/
def intersects_both_branches (h : Hyperbola) (l : Line) : Prop := sorry

/-- The necessary and sufficient condition for a line to intersect both branches of a hyperbola -/
theorem intersects_both_branches_iff (h : Hyperbola) (l : Line) :
  intersects_both_branches h l ↔ -h.b / h.a < l.k ∧ l.k < h.b / h.a := by sorry

end NUMINAMATH_CALUDE_intersects_both_branches_iff_l908_90810


namespace NUMINAMATH_CALUDE_f_period_l908_90876

open Real

noncomputable def f (x : ℝ) : ℝ := 
  (sin (2 * x) + sin (2 * x + π / 3)) / (cos (2 * x) + cos (2 * x + π / 3))

theorem f_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ 
  T = π / 2 :=
sorry

end NUMINAMATH_CALUDE_f_period_l908_90876


namespace NUMINAMATH_CALUDE_horner_method_proof_l908_90874

def f (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem horner_method_proof : f 5 = 7548 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l908_90874


namespace NUMINAMATH_CALUDE_erased_length_cm_l908_90851

-- Define the original length in meters
def original_length_m : ℝ := 1

-- Define the final length in centimeters
def final_length_cm : ℝ := 76

-- Define the conversion factor from meters to centimeters
def m_to_cm : ℝ := 100

-- Theorem to prove
theorem erased_length_cm : 
  (original_length_m * m_to_cm - final_length_cm) = 24 := by
  sorry

end NUMINAMATH_CALUDE_erased_length_cm_l908_90851


namespace NUMINAMATH_CALUDE_zero_not_equivalent_to_intersection_l908_90881

-- Define a real-valued function
def RealFunction := ℝ → ℝ

-- Define the zero of a function
def is_zero_of_function (f : RealFunction) (x : ℝ) : Prop := f x = 0

-- Define the intersection point of a function's graph and the x-axis
def is_intersection_with_x_axis (f : RealFunction) (x : ℝ) : Prop :=
  f x = 0 ∧ ∀ y : ℝ, y ≠ 0 → f x ≠ y

-- Theorem stating that these concepts are not equivalent
theorem zero_not_equivalent_to_intersection :
  ¬ (∀ (f : RealFunction) (x : ℝ), is_zero_of_function f x ↔ is_intersection_with_x_axis f x) :=
sorry

end NUMINAMATH_CALUDE_zero_not_equivalent_to_intersection_l908_90881


namespace NUMINAMATH_CALUDE_max_rectangles_in_5x5_grid_seven_rectangles_fit_5x5_grid_l908_90884

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the square grid -/
def Grid := ℕ × ℕ

/-- Check if a list of rectangles fits in the grid without overlap and covers it completely -/
def fits_grid (grid : Grid) (rectangles : List Rectangle) : Prop :=
  sorry

/-- The theorem stating that 7 is the maximum number of rectangles that can fit in a 5x5 grid -/
theorem max_rectangles_in_5x5_grid :
  ∀ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, (r.width = 1 ∧ r.height = 4) ∨ (r.width = 1 ∧ r.height = 3)) →
    fits_grid (5, 5) rectangles →
    rectangles.length ≤ 7 :=
  sorry

/-- The theorem stating that 7 rectangles can indeed fit in a 5x5 grid -/
theorem seven_rectangles_fit_5x5_grid :
  ∃ (rectangles : List Rectangle),
    (∀ r ∈ rectangles, (r.width = 1 ∧ r.height = 4) ∨ (r.width = 1 ∧ r.height = 3)) ∧
    fits_grid (5, 5) rectangles ∧
    rectangles.length = 7 :=
  sorry

end NUMINAMATH_CALUDE_max_rectangles_in_5x5_grid_seven_rectangles_fit_5x5_grid_l908_90884


namespace NUMINAMATH_CALUDE_brownie_pan_dimensions_l908_90859

theorem brownie_pan_dimensions :
  ∀ m n : ℕ,
    m * n = 48 →
    (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4) →
    ((m = 4 ∧ n = 12) ∨ (m = 12 ∧ n = 4) ∨ (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6)) :=
by sorry

end NUMINAMATH_CALUDE_brownie_pan_dimensions_l908_90859


namespace NUMINAMATH_CALUDE_equilateral_triangle_dot_product_l908_90896

/-- Equilateral triangle ABC with side length 2 -/
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ‖A - B‖ = 2 ∧ ‖B - C‖ = 2 ∧ ‖C - A‖ = 2

/-- Vector from point P to point Q -/
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := Q - P

theorem equilateral_triangle_dot_product 
  (A B C D E : ℝ × ℝ) 
  (h_triangle : Triangle A B C)
  (h_BC : vec B C = 2 • vec B D)
  (h_CA : vec C A = 3 • vec C E)
  (a b : ℝ × ℝ)
  (h_a : a = vec A B)
  (h_b : b = vec A C)
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 2)
  (h_dot : a • b = 2) :
  (1/2 • (a + b)) • (2/3 • b - a) = -1 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_dot_product_l908_90896


namespace NUMINAMATH_CALUDE_cube_edge_length_is_twelve_l908_90887

/-- Represents a cube with integer edge length -/
structure Cube where
  edge_length : ℕ

/-- Calculates the number of small cubes with three painted faces -/
def three_painted_faces (c : Cube) : ℕ := 8

/-- Calculates the number of small cubes with two painted faces -/
def two_painted_faces (c : Cube) : ℕ := 12 * (c.edge_length - 2)

/-- Theorem stating that when the number of small cubes with two painted faces
    is 15 times the number of small cubes with three painted faces,
    the edge length of the cube must be 12 -/
theorem cube_edge_length_is_twelve (c : Cube) :
  two_painted_faces c = 15 * three_painted_faces c → c.edge_length = 12 := by
  sorry


end NUMINAMATH_CALUDE_cube_edge_length_is_twelve_l908_90887


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l908_90814

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 + x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Set ℝ := {x | 1 < x ∧ x < 3}

-- Define the theorem
theorem quadratic_inequality_theorem (a c : ℝ) 
  (h : ∀ x, f a c x > 0 ↔ x ∈ solution_set a c) :
  a = -1/4 ∧ c = -3/4 ∧ 
  ∀ m : ℝ, (∀ x, -1/4 * x^2 + 2*x - 3 > 0 → x + m > 0) → m ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l908_90814


namespace NUMINAMATH_CALUDE_probability_one_boy_one_girl_l908_90808

def num_boys : ℕ := 3
def num_girls : ℕ := 2
def num_participants : ℕ := 2

def total_combinations : ℕ := (num_boys + num_girls).choose num_participants

def favorable_outcomes : ℕ := num_boys.choose 1 * num_girls.choose 1

theorem probability_one_boy_one_girl :
  (favorable_outcomes : ℚ) / total_combinations = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_one_boy_one_girl_l908_90808


namespace NUMINAMATH_CALUDE_total_points_after_perfect_games_l908_90863

/-- The number of points in a perfect score -/
def perfect_score : ℕ := 21

/-- The number of perfect games played -/
def games_played : ℕ := 11

/-- Theorem: The total points scored after playing 11 perfect games,
    where a perfect score is 21 points, is equal to 231 points. -/
theorem total_points_after_perfect_games :
  perfect_score * games_played = 231 := by
  sorry

end NUMINAMATH_CALUDE_total_points_after_perfect_games_l908_90863


namespace NUMINAMATH_CALUDE_second_exponent_base_l908_90806

theorem second_exponent_base (x b : ℕ) (h1 : b > 0) (h2 : (18^6) * (x^17) = (2^6) * (3^b)) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_exponent_base_l908_90806


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l908_90846

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- State the theorem
theorem union_of_M_and_N :
  M ∪ N = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l908_90846


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l908_90868

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∀ x, x^2 + m*x + n = 0 ↔ ∃ y, y^2 + p*y + m = 0 ∧ x = 3*y) →
  n / p = 27 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l908_90868


namespace NUMINAMATH_CALUDE_remaining_cooking_time_l908_90840

def total_potatoes : ℕ := 16
def cooked_potatoes : ℕ := 7
def cooking_time_per_potato : ℕ := 5

theorem remaining_cooking_time : 
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cooking_time_l908_90840


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_minus_circles_l908_90892

/-- The shaded area in a rectangle after subtracting two circles -/
theorem shaded_area_rectangle_minus_circles 
  (rectangle_length : ℝ) 
  (rectangle_width : ℝ)
  (circle1_radius : ℝ)
  (circle2_radius : ℝ)
  (h1 : rectangle_length = 16)
  (h2 : rectangle_width = 8)
  (h3 : circle1_radius = 4)
  (h4 : circle2_radius = 2) :
  rectangle_length * rectangle_width - π * (circle1_radius^2 + circle2_radius^2) = 128 - 20 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_minus_circles_l908_90892


namespace NUMINAMATH_CALUDE_square_area_from_corners_l908_90897

/-- The area of a square with adjacent corners at (4, -1) and (-1, 3) on a Cartesian coordinate plane is 41. -/
theorem square_area_from_corners : 
  let p1 : ℝ × ℝ := (4, -1)
  let p2 : ℝ × ℝ := (-1, 3)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  side_length^2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_corners_l908_90897


namespace NUMINAMATH_CALUDE_range_of_a_l908_90866

theorem range_of_a (p : ∀ x ∈ Set.Icc 1 4, x^2 ≥ a) 
                   (q : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a = 1 ∨ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l908_90866


namespace NUMINAMATH_CALUDE_haris_capital_contribution_l908_90809

/-- Represents the capital contribution of a business partner -/
structure Capital where
  amount : ℕ
  months : ℕ

/-- Calculates the effective capital based on the amount and months invested -/
def effectiveCapital (c : Capital) : ℕ := c.amount * c.months

/-- Represents the profit-sharing ratio between two partners -/
structure ProfitRatio where
  first : ℕ
  second : ℕ

theorem haris_capital_contribution 
  (praveens_capital : Capital)
  (haris_join_month : ℕ)
  (total_months : ℕ)
  (profit_ratio : ProfitRatio)
  (h1 : praveens_capital.amount = 3360)
  (h2 : praveens_capital.months = total_months)
  (h3 : haris_join_month = 5)
  (h4 : total_months = 12)
  (h5 : profit_ratio.first = 2)
  (h6 : profit_ratio.second = 3)
  : ∃ (haris_capital : Capital), 
    haris_capital.amount = 8640 ∧ 
    haris_capital.months = total_months - haris_join_month ∧
    effectiveCapital praveens_capital * profit_ratio.second = 
    effectiveCapital haris_capital * profit_ratio.first :=
sorry

end NUMINAMATH_CALUDE_haris_capital_contribution_l908_90809


namespace NUMINAMATH_CALUDE_probability_to_reach_target_l908_90829

-- Define the robot's position as a pair of integers
def Position := ℤ × ℤ

-- Define the possible directions
inductive Direction
| Left
| Right
| Up
| Down

-- Define a step as a movement in a direction
def step (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Left  => (pos.1 - 1, pos.2)
  | Direction.Right => (pos.1 + 1, pos.2)
  | Direction.Up    => (pos.1, pos.2 + 1)
  | Direction.Down  => (pos.1, pos.2 - 1)

-- Define the probability of each direction
def directionProbability : ℚ := 1 / 4

-- Define the maximum number of steps
def maxSteps : ℕ := 6

-- Define the target position
def target : Position := (3, 1)

-- Define the function to calculate the probability of reaching the target
noncomputable def probabilityToReachTarget : ℚ := sorry

-- State the theorem
theorem probability_to_reach_target :
  probabilityToReachTarget = 37 / 512 := by sorry

end NUMINAMATH_CALUDE_probability_to_reach_target_l908_90829


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l908_90835

def max_num : ℕ := 2020
def multiples_of_5 : ℕ := 404

-- Function to calculate the number of trailing zeros
def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

-- Theorem statement
theorem product_trailing_zeros :
  trailing_zeros max_num = 503 :=
sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l908_90835


namespace NUMINAMATH_CALUDE_billy_ice_cubes_l908_90843

/-- The number of ice cubes in each tray -/
def cubes_per_tray : ℕ := 25

/-- The number of trays Billy has -/
def number_of_trays : ℕ := 15

/-- The total number of ice cubes Billy can make -/
def total_ice_cubes : ℕ := cubes_per_tray * number_of_trays

theorem billy_ice_cubes : total_ice_cubes = 375 := by sorry

end NUMINAMATH_CALUDE_billy_ice_cubes_l908_90843


namespace NUMINAMATH_CALUDE_unique_function_divisibility_l908_90836

theorem unique_function_divisibility (k : ℕ) :
  ∃! f : ℕ → ℕ, ∀ m n : ℕ, (f m + f n) ∣ (m + n)^k :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_divisibility_l908_90836


namespace NUMINAMATH_CALUDE_steak_knife_set_cost_is_80_l908_90893

/-- Represents the cost of a steak knife set -/
def steak_knife_set_cost (knives_per_set : ℕ) (single_knife_cost : ℕ) : ℕ :=
  knives_per_set * single_knife_cost

/-- Proves that the cost of a steak knife set with 4 knives at $20 each is $80 -/
theorem steak_knife_set_cost_is_80 :
  steak_knife_set_cost 4 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_steak_knife_set_cost_is_80_l908_90893


namespace NUMINAMATH_CALUDE_cistern_width_l908_90878

/-- Calculates the width of a rectangular cistern given its length, depth, and total wet surface area. -/
theorem cistern_width (length depth area : ℝ) (h1 : length = 5) (h2 : depth = 1.25) (h3 : area = 42.5) :
  ∃ width : ℝ, width = 4 ∧ 
  area = length * width + 2 * (depth * length) + 2 * (depth * width) :=
by sorry

end NUMINAMATH_CALUDE_cistern_width_l908_90878


namespace NUMINAMATH_CALUDE_point_on_line_product_of_y_coordinates_l908_90823

theorem point_on_line_product_of_y_coordinates :
  ∀ y₁ y₂ : ℝ,
  ((-3 - 3)^2 + (-1 - y₁)^2 = 13^2) →
  ((-3 - 3)^2 + (-1 - y₂)^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -132 := by sorry

end NUMINAMATH_CALUDE_point_on_line_product_of_y_coordinates_l908_90823


namespace NUMINAMATH_CALUDE_gcd_1994_powers_and_product_l908_90864

theorem gcd_1994_powers_and_product : 
  Nat.gcd (1994^1994 + 1994^1995) (1994 * 1995) = 1994 * 1995 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1994_powers_and_product_l908_90864


namespace NUMINAMATH_CALUDE_set_equality_l908_90821

def positive_integers : Set ℕ := {n : ℕ | n > 0}

def set_a : Set ℕ := {x ∈ positive_integers | x - 3 < 2}
def set_b : Set ℕ := {1, 2, 3, 4}

theorem set_equality : set_a = set_b := by sorry

end NUMINAMATH_CALUDE_set_equality_l908_90821


namespace NUMINAMATH_CALUDE_valid_selections_count_l908_90831

/-- The number of male intern teachers --/
def male_teachers : ℕ := 5

/-- The number of female intern teachers --/
def female_teachers : ℕ := 4

/-- The total number of intern teachers --/
def total_teachers : ℕ := male_teachers + female_teachers

/-- The number of teachers to be selected --/
def selected_teachers : ℕ := 3

/-- The number of ways to select 3 teachers from the total pool --/
def total_selections : ℕ := Nat.descFactorial total_teachers selected_teachers

/-- The number of ways to select 3 male teachers --/
def all_male_selections : ℕ := Nat.descFactorial male_teachers selected_teachers

/-- The number of ways to select 3 female teachers --/
def all_female_selections : ℕ := Nat.descFactorial female_teachers selected_teachers

/-- The number of valid selection schemes --/
def valid_selections : ℕ := total_selections - (all_male_selections + all_female_selections)

theorem valid_selections_count : valid_selections = 420 := by
  sorry

end NUMINAMATH_CALUDE_valid_selections_count_l908_90831


namespace NUMINAMATH_CALUDE_power_of_two_plus_one_l908_90819

theorem power_of_two_plus_one (b m n : ℕ) 
  (hb : b > 1) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m ≠ n) 
  (h_prime_divisors : ∀ p : ℕ, Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_plus_one_l908_90819


namespace NUMINAMATH_CALUDE_intersecting_rectangles_area_l908_90822

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.length

/-- The total shaded area of two intersecting rectangles -/
def totalShadedArea (r1 r2 overlap : Rectangle) : ℝ :=
  r1.area + r2.area - overlap.area

theorem intersecting_rectangles_area :
  let r1 : Rectangle := ⟨4, 12⟩
  let r2 : Rectangle := ⟨5, 10⟩
  let overlap : Rectangle := ⟨4, 5⟩
  totalShadedArea r1 r2 overlap = 78 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_rectangles_area_l908_90822


namespace NUMINAMATH_CALUDE_three_numbers_sum_l908_90803

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →
  y = 10 →
  (x + y + z) / 3 = x + 20 →
  (x + y + z) / 3 = z - 25 →
  x + y + z = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l908_90803


namespace NUMINAMATH_CALUDE_smallest_n_cubic_minus_n_divisibility_l908_90800

theorem smallest_n_cubic_minus_n_divisibility : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 0 < m ∧ m < n → 
    ∀ (k : ℕ), 1 ≤ k ∧ k ≤ m + 2 → (m^3 - m) % k = 0) ∧
  (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n + 2 ∧ (n^3 - n) % k ≠ 0) ∧
  n = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_cubic_minus_n_divisibility_l908_90800


namespace NUMINAMATH_CALUDE_auntie_em_parking_probability_l908_90883

def total_spaces : ℕ := 18
def parked_cars : ℕ := 12
def suv_spaces : ℕ := 2

theorem auntie_em_parking_probability :
  let total_configurations := Nat.choose total_spaces parked_cars
  let unfavorable_configurations := Nat.choose (parked_cars + 1) parked_cars
  (total_configurations - unfavorable_configurations : ℚ) / total_configurations = 1403 / 1546 :=
by sorry

end NUMINAMATH_CALUDE_auntie_em_parking_probability_l908_90883


namespace NUMINAMATH_CALUDE_factor_expression_l908_90841

theorem factor_expression (x : ℝ) : x * (x + 3) - 2 * (x + 3) = (x + 3) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l908_90841


namespace NUMINAMATH_CALUDE_danes_daughters_flowers_l908_90867

def flowers_per_basket (people : ℕ) (flowers_per_person : ℕ) (additional_growth : ℕ) (died : ℕ) (baskets : ℕ) : ℕ :=
  ((people * flowers_per_person + additional_growth - died) / baskets)

theorem danes_daughters_flowers :
  flowers_per_basket 2 5 20 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_danes_daughters_flowers_l908_90867


namespace NUMINAMATH_CALUDE_percent_of_y_l908_90899

theorem percent_of_y (y : ℝ) (h : y > 0) : ((6 * y) / 20 + (3 * y) / 10) / y = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l908_90899


namespace NUMINAMATH_CALUDE_smallest_number_with_all_factors_l908_90847

def alice_number : ℕ := 24

-- Function to check if a number has all prime factors of another number
def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n) → (p ∣ m)

theorem smallest_number_with_all_factors :
  ∃ (bob_number : ℕ), 
    bob_number > 0 ∧ 
    has_all_prime_factors alice_number bob_number ∧
    (∀ k : ℕ, k > 0 → has_all_prime_factors alice_number k → bob_number ≤ k) ∧
    bob_number = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_all_factors_l908_90847


namespace NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l908_90889

theorem tan_fifteen_ratio_equals_sqrt_three :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l908_90889


namespace NUMINAMATH_CALUDE_equal_roots_C_value_l908_90833

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4 * eq.a * eq.c

/-- Checks if a quadratic equation has equal roots -/
def hasEqualRoots (eq : QuadraticEquation) : Prop :=
  discriminant eq = 0

/-- The specific quadratic equation from the problem -/
def problemEquation (k C : ℝ) : QuadraticEquation where
  a := 2 * k
  b := 6 * k
  c := C

/-- The theorem to be proved -/
theorem equal_roots_C_value :
  ∃ C : ℝ, hasEqualRoots (problemEquation 0.4444444444444444 C) ∧ C = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_C_value_l908_90833


namespace NUMINAMATH_CALUDE_inequality_solution_set_l908_90850

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a - 1) * x + 1 > 0) ↔ -1 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l908_90850


namespace NUMINAMATH_CALUDE_cost_per_side_of_square_l908_90827

/-- The cost of fencing each side of a square, given the total cost --/
theorem cost_per_side_of_square (total_cost : ℝ) (h : total_cost = 276) : 
  ∃ (side_cost : ℝ), side_cost * 4 = total_cost ∧ side_cost = 69 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_side_of_square_l908_90827


namespace NUMINAMATH_CALUDE_one_in_linked_triple_l908_90860

def is_linked (m n : ℕ+) : Prop :=
  (m.val ∣ 3 * n.val + 1) ∧ (n.val ∣ 3 * m.val + 1)

theorem one_in_linked_triple (a b c : ℕ+) :
  a ≠ b → b ≠ c → a ≠ c →
  is_linked a b → is_linked b c →
  1 ∈ ({a.val, b.val, c.val} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_one_in_linked_triple_l908_90860


namespace NUMINAMATH_CALUDE_sum_difference_is_450_l908_90805

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_down_to_10 (n : ℕ) : ℕ := (n / 10) * 10

def kate_sum (n : ℕ) : ℕ := 
  (List.range n).map round_down_to_10 |> List.sum

theorem sum_difference_is_450 (n : ℕ) (h : n = 100) : 
  (sum_first_n n) - (kate_sum n) = 450 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_is_450_l908_90805


namespace NUMINAMATH_CALUDE_max_square_plots_l908_90857

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def availableFence : ℕ := 2400

/-- Calculates the number of square plots along the field's width -/
def numPlotsWidth (field : FieldDimensions) : ℕ :=
  20

/-- Calculates the number of square plots along the field's length -/
def numPlotsLength (field : FieldDimensions) : ℕ :=
  30

/-- Calculates the total number of square plots -/
def totalPlots (field : FieldDimensions) : ℕ :=
  numPlotsWidth field * numPlotsLength field

/-- Calculates the length of internal fencing used -/
def usedFence (field : FieldDimensions) : ℕ :=
  field.width * (numPlotsLength field - 1) + field.length * (numPlotsWidth field - 1)

/-- Theorem stating that 600 is the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
    (h1 : field.width = 40) 
    (h2 : field.length = 60) : 
    totalPlots field = 600 ∧ 
    usedFence field ≤ availableFence ∧ 
    ∀ n m : ℕ, n * m > 600 → 
      field.width * (m - 1) + field.length * (n - 1) > availableFence :=
  sorry

#check max_square_plots

end NUMINAMATH_CALUDE_max_square_plots_l908_90857


namespace NUMINAMATH_CALUDE_rebate_calculation_l908_90870

theorem rebate_calculation (polo_price necklace_price game_price : ℕ)
  (polo_count necklace_count : ℕ) (total_after_rebate : ℕ) :
  polo_price = 26 →
  necklace_price = 83 →
  game_price = 90 →
  polo_count = 3 →
  necklace_count = 2 →
  total_after_rebate = 322 →
  (polo_price * polo_count + necklace_price * necklace_count + game_price) - total_after_rebate = 12 := by
  sorry

end NUMINAMATH_CALUDE_rebate_calculation_l908_90870


namespace NUMINAMATH_CALUDE_max_value_xyz_expression_l908_90890

theorem max_value_xyz_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  xyz * (x + y + z) / ((x + y)^2 * (x + z)^2) ≤ (1 : ℝ) / 4 ∧
  (xyz * (x + y + z) / ((x + y)^2 * (x + z)^2) = (1 : ℝ) / 4 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_max_value_xyz_expression_l908_90890


namespace NUMINAMATH_CALUDE_james_restaurant_revenue_l908_90832

theorem james_restaurant_revenue :
  -- Define the constants
  let beef_amount : ℝ := 20
  let pork_amount : ℝ := beef_amount / 2
  let meat_per_meal : ℝ := 1.5
  let price_per_meal : ℝ := 20

  -- Calculate total meat
  let total_meat : ℝ := beef_amount + pork_amount

  -- Calculate number of meals
  let number_of_meals : ℝ := total_meat / meat_per_meal

  -- Calculate total revenue
  let total_revenue : ℝ := number_of_meals * price_per_meal

  -- Prove that the total revenue is $400
  total_revenue = 400 := by sorry

end NUMINAMATH_CALUDE_james_restaurant_revenue_l908_90832


namespace NUMINAMATH_CALUDE_yellow_balloons_count_l908_90854

/-- The number of yellow balloons -/
def yellow_balloons : ℕ := 3414

/-- The number of black balloons -/
def black_balloons : ℕ := yellow_balloons + 1762

/-- The total number of balloons -/
def total_balloons : ℕ := yellow_balloons + black_balloons

theorem yellow_balloons_count : yellow_balloons = 3414 :=
  by
  have h1 : black_balloons = yellow_balloons + 1762 := rfl
  have h2 : total_balloons / 10 = 859 := by sorry
  sorry


end NUMINAMATH_CALUDE_yellow_balloons_count_l908_90854


namespace NUMINAMATH_CALUDE_min_boxes_to_eliminate_l908_90886

/-- The total number of boxes -/
def total_boxes : ℕ := 30

/-- The number of boxes containing at least $200,000 -/
def high_value_boxes : ℕ := 6

/-- The minimum number of boxes that must be eliminated -/
def boxes_to_eliminate : ℕ := 18

/-- Theorem stating that eliminating 18 boxes is the minimum required for a 50% chance of a high-value box -/
theorem min_boxes_to_eliminate :
  boxes_to_eliminate = total_boxes - 2 * high_value_boxes :=
sorry

end NUMINAMATH_CALUDE_min_boxes_to_eliminate_l908_90886


namespace NUMINAMATH_CALUDE_black_to_grey_ratio_in_square_with_circles_l908_90807

/-- The ratio of black to grey areas in a square with four inscribed circles -/
theorem black_to_grey_ratio_in_square_with_circles (s : ℝ) (h : s > 0) :
  let r := s / 4
  let circle_area := π * r^2
  let total_square_area := s^2
  let remaining_area := total_square_area - 4 * circle_area
  let black_area := remaining_area / 4
  let grey_area := 3 * black_area
  black_area / grey_area = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_black_to_grey_ratio_in_square_with_circles_l908_90807


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l908_90838

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (∀ x, -x^2 + 2*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l908_90838


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l908_90872

/-- The number of y-intercepts for the parabola x = 3y^2 - 4y + 5 -/
theorem parabola_y_intercepts :
  let f (y : ℝ) := 3 * y^2 - 4 * y + 5
  ∃! x : ℝ, (∀ y : ℝ, f y = x) ∧ (¬ ∃ y : ℝ, f y = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l908_90872


namespace NUMINAMATH_CALUDE_oil_barrels_problem_l908_90875

/-- The minimum number of barrels needed to contain a given amount of oil -/
def min_barrels (total_oil : ℕ) (barrel_capacity : ℕ) : ℕ :=
  (total_oil + barrel_capacity - 1) / barrel_capacity

/-- Proof that at least 7 barrels are needed for 250 kg of oil with 40 kg capacity barrels -/
theorem oil_barrels_problem :
  min_barrels 250 40 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oil_barrels_problem_l908_90875


namespace NUMINAMATH_CALUDE_distance_A_to_y_axis_l908_90869

def point_A : ℝ × ℝ := (-2, 1)

def distance_to_y_axis (p : ℝ × ℝ) : ℝ := |p.1|

theorem distance_A_to_y_axis :
  distance_to_y_axis point_A = 2 := by sorry

end NUMINAMATH_CALUDE_distance_A_to_y_axis_l908_90869


namespace NUMINAMATH_CALUDE_inequality_iff_in_interval_l908_90815

/-- The roots of the quadratic equation x^2 - (16/5)x - 8 = 0 --/
def a : ℝ := sorry
def b : ℝ := sorry

axiom a_lt_b : a < b
axiom b_lt_zero : b < 0
axiom roots_property : ∀ x : ℝ, x^2 - (16/5) * x - 8 = 0 ↔ (x = a ∨ x = b)

/-- The main theorem stating the equivalence between the inequality and the solution interval --/
theorem inequality_iff_in_interval (x : ℝ) : 
  1 / (x^2 + 2) + 1 / 2 > 5 / x + 21 / 10 ↔ (x < a ∨ (b < x ∧ x < 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_iff_in_interval_l908_90815


namespace NUMINAMATH_CALUDE_root_product_plus_one_l908_90844

theorem root_product_plus_one (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  (1 + a) * (1 + b) * (1 + c) = 51 := by
sorry

end NUMINAMATH_CALUDE_root_product_plus_one_l908_90844


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l908_90855

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∀ x : ℝ, x^2 + p*x + q = 0 → 
    ∃ y : ℝ, y^2 + p*y + q = 0 ∧ |x - y| = 2) →
  p = 2 * Real.sqrt (q + 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l908_90855


namespace NUMINAMATH_CALUDE_no_x4_term_implies_a_zero_l908_90845

theorem no_x4_term_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, ∃ b c d : ℝ, -5 * x^3 * (x^2 + a * x + 5) = b * x^5 + c * x^3 + d) →
  a = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_x4_term_implies_a_zero_l908_90845


namespace NUMINAMATH_CALUDE_inequality_proof_l908_90842

theorem inequality_proof (a b : ℝ) (h1 : a ≥ b) (h2 : b ≥ 0) :
  Real.sqrt (a^2 + b^2) + (a^3 + b^3)^(1/3) + (a^4 + b^4)^(1/4) ≤ 3*a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l908_90842


namespace NUMINAMATH_CALUDE_boat_speed_upstream_l908_90877

/-- The speed of a boat upstream given its speed in still water and the speed of the current. -/
def speed_upstream (speed_still_water : ℝ) (speed_current : ℝ) : ℝ :=
  speed_still_water - speed_current

/-- Theorem: The speed of a boat upstream is 30 kmph when its speed in still water is 50 kmph and the current speed is 20 kmph. -/
theorem boat_speed_upstream :
  speed_upstream 50 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_upstream_l908_90877


namespace NUMINAMATH_CALUDE_m_range_l908_90882

def f (x : ℝ) := x^3 + x

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → f (m * Real.sin θ) + f (1 - m) > 0) →
  m < 1 :=
sorry

end NUMINAMATH_CALUDE_m_range_l908_90882


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l908_90873

theorem product_of_repeating_decimal_and_eight :
  ∃ (x : ℚ), (∃ (n : ℕ), x = (456 : ℚ) / (10^n - 1)) ∧ 8 * x = 1216 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l908_90873


namespace NUMINAMATH_CALUDE_reading_growth_rate_l908_90885

theorem reading_growth_rate (initial_amount final_amount : ℝ) (growth_period : ℕ) (x : ℝ) :
  initial_amount = 1 →
  final_amount = 1.21 →
  growth_period = 2 →
  final_amount = initial_amount * (1 + x)^growth_period →
  100 * (1 + x)^2 = 121 :=
by sorry

end NUMINAMATH_CALUDE_reading_growth_rate_l908_90885


namespace NUMINAMATH_CALUDE_cat_leash_max_distance_l908_90898

theorem cat_leash_max_distance :
  let center : ℝ × ℝ := (6, 2)
  let radius : ℝ := 15
  let origin : ℝ × ℝ := (0, 0)
  let max_distance := radius + Real.sqrt ((center.1 - origin.1)^2 + (center.2 - origin.2)^2)
  max_distance = 15 + 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_cat_leash_max_distance_l908_90898


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l908_90861

/-- Given the number of total worksheets, graded worksheets, and problems per worksheet,
    calculate the number of problems left to grade. -/
theorem problems_left_to_grade 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (h1 : total_worksheets = 15)
  (h2 : graded_worksheets = 7)
  (h3 : problems_per_worksheet = 3)
  (h4 : graded_worksheets ≤ total_worksheets) :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 24 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l908_90861


namespace NUMINAMATH_CALUDE_integer_fraction_implication_l908_90820

theorem integer_fraction_implication (m n p q : ℕ) (h1 : m ≠ p) 
  (h2 : ∃ k : ℤ, k = (m * n + p * q) / (m - p)) : 
  ∃ l : ℤ, l = (m * q + n * p) / (m - p) := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_implication_l908_90820


namespace NUMINAMATH_CALUDE_simplify_expression_l908_90891

theorem simplify_expression (y : ℝ) :
  4 * y - 6 * y^2 + 8 - (3 + 5 * y - 9 * y^2) = 3 * y^2 - y + 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l908_90891


namespace NUMINAMATH_CALUDE_oh_squared_value_l908_90834

/-- Given a triangle ABC with circumcenter O, orthocenter H, side lengths a, b, c, and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- The squared distance between the circumcenter and orthocenter -/
def OH_squared (t : Triangle) : ℝ := 9 * t.R^2 - (t.a^2 + t.b^2 + t.c^2)

theorem oh_squared_value (t : Triangle) 
  (h1 : t.R = 5) 
  (h2 : t.a^2 + t.b^2 + t.c^2 = 50) : 
  OH_squared t = 175 := by
  sorry

end NUMINAMATH_CALUDE_oh_squared_value_l908_90834


namespace NUMINAMATH_CALUDE_simplify_expression_l908_90858

theorem simplify_expression (x y : ℚ) (hx : x = 10) (hy : y = -1/25) :
  ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l908_90858


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l908_90817

theorem negative_fractions_comparison : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l908_90817


namespace NUMINAMATH_CALUDE_vote_count_proof_l908_90888

theorem vote_count_proof (T : ℕ) (F : ℕ) (A : ℕ) : 
  F = A + 68 →  -- 68 more votes in favor than against
  A = (40 * T) / 100 →  -- 40% of total votes were against
  T = F + A →  -- total votes is sum of for and against
  T = 340 :=
by sorry

end NUMINAMATH_CALUDE_vote_count_proof_l908_90888


namespace NUMINAMATH_CALUDE_cereal_serving_size_l908_90825

def cereal_box_problem (total_cups : ℕ) (total_servings : ℕ) : Prop :=
  total_cups ≠ 0 ∧ total_servings ≠ 0 → total_cups / total_servings = 2

theorem cereal_serving_size : cereal_box_problem 18 9 := by
  sorry

end NUMINAMATH_CALUDE_cereal_serving_size_l908_90825


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_m_l908_90862

-- Define the functions f and g
def f (x : ℝ) := x^2 - 2*x - 8
def g (x : ℝ) := 2*x^2 - 4*x - 16

-- Theorem for the solution set of g(x) < 0
theorem solution_set_g (x : ℝ) : g x < 0 ↔ -2 < x ∧ x < 4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∀ x > 1, f x ≥ (m + 2)*x - m - 15) → m ≤ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_m_l908_90862


namespace NUMINAMATH_CALUDE_fraction_calculation_l908_90848

theorem fraction_calculation : (3/8) / (4/9) + 1/6 = 97/96 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l908_90848


namespace NUMINAMATH_CALUDE_percentage_of_fresh_peaches_l908_90895

def total_peaches : ℕ := 250
def thrown_away : ℕ := 15
def peaches_left : ℕ := 135

def fresh_peaches : ℕ := total_peaches - (thrown_away + (total_peaches - peaches_left))

theorem percentage_of_fresh_peaches :
  (fresh_peaches : ℚ) / total_peaches * 100 = 48 := by sorry

end NUMINAMATH_CALUDE_percentage_of_fresh_peaches_l908_90895


namespace NUMINAMATH_CALUDE_constant_function_invariant_l908_90871

-- Define the function g
def g : ℝ → ℝ := λ x => -3

-- State the theorem
theorem constant_function_invariant (x : ℝ) : g (3 * x - 1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_invariant_l908_90871


namespace NUMINAMATH_CALUDE_problem_solution_l908_90880

theorem problem_solution (m n x y : ℝ) 
  (h1 : m - n = 8) 
  (h2 : x + y = 1) : 
  (n + x) - (m - y) = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l908_90880


namespace NUMINAMATH_CALUDE_road_travel_rate_l908_90865

/-- The rate per square meter for traveling roads on a rectangular lawn -/
theorem road_travel_rate (lawn_length lawn_width road_width total_cost : ℕ) :
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  total_cost = 5200 →
  (total_cost : ℚ) / ((lawn_length * road_width + lawn_width * road_width - road_width * road_width) : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_road_travel_rate_l908_90865


namespace NUMINAMATH_CALUDE_second_number_is_72_l908_90852

theorem second_number_is_72 
  (sum : ℝ) 
  (first : ℝ) 
  (second : ℝ) 
  (third : ℝ) 
  (h1 : sum = 264) 
  (h2 : first = 2 * second) 
  (h3 : third = (1/3) * first) 
  (h4 : first + second + third = sum) : second = 72 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_72_l908_90852


namespace NUMINAMATH_CALUDE_sufficient_condition_absolute_value_necessary_condition_inequality_l908_90801

-- Statement ③
theorem sufficient_condition_absolute_value (a b : ℝ) :
  a^2 ≠ b^2 → |a| = |b| :=
sorry

-- Statement ④
theorem necessary_condition_inequality (a b c : ℝ) :
  a * c^2 < b * c^2 → a < b :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_absolute_value_necessary_condition_inequality_l908_90801


namespace NUMINAMATH_CALUDE_sqrt_8_times_sqrt_18_l908_90839

theorem sqrt_8_times_sqrt_18 : Real.sqrt 8 * Real.sqrt 18 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_8_times_sqrt_18_l908_90839


namespace NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l908_90837

/-- Represents the fishing pattern in the coastal village -/
structure FishingPattern :=
  (daily : ℕ)
  (everyOtherDay : ℕ)
  (everyThreeDay : ℕ)
  (yesterday : ℕ)
  (today : ℕ)

/-- Calculates the number of people fishing tomorrow given the fishing pattern -/
def fishersTomorrow (pattern : FishingPattern) : ℕ :=
  pattern.daily + pattern.everyThreeDay + (pattern.everyOtherDay - (pattern.yesterday - pattern.daily))

/-- Theorem stating that given the specific fishing pattern, 15 people will fish tomorrow -/
theorem fifteen_fishers_tomorrow : 
  ∀ (pattern : FishingPattern), 
  pattern.daily = 7 ∧ 
  pattern.everyOtherDay = 8 ∧ 
  pattern.everyThreeDay = 3 ∧
  pattern.yesterday = 12 ∧
  pattern.today = 10 →
  fishersTomorrow pattern = 15 := by
  sorry


end NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l908_90837
