import Mathlib

namespace NUMINAMATH_CALUDE_root_square_transformation_l3604_360479

/-- The original polynomial f(x) -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

/-- The resulting polynomial g(x) -/
def g (x : ℝ) : ℝ := x^3 - 4*x^2 - 7*x + 16

theorem root_square_transformation (r : ℝ) : 
  f r = 0 → ∃ s, g s = 0 ∧ s = r^2 := by sorry

end NUMINAMATH_CALUDE_root_square_transformation_l3604_360479


namespace NUMINAMATH_CALUDE_variance_of_literary_works_l3604_360465

def literary_works : List ℕ := [6, 9, 5, 8, 10, 4]

def mean (data : List ℕ) : ℚ :=
  (data.sum : ℚ) / data.length

def variance (data : List ℕ) : ℚ :=
  let μ := mean data
  (data.map (fun x => ((x : ℚ) - μ) ^ 2)).sum / data.length

theorem variance_of_literary_works : variance literary_works = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_literary_works_l3604_360465


namespace NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l3604_360483

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a tunnel drilled through it -/
structure CubeWithTunnel where
  /-- Length of the cube's edge -/
  edgeLength : ℝ
  /-- Point P, a corner of the cube -/
  p : Point3D
  /-- Point L on PQ -/
  l : Point3D
  /-- Point M on PR -/
  m : Point3D
  /-- Point N on PC -/
  n : Point3D

/-- The surface area of the cube with tunnel can be expressed as x + y√z -/
def surfaceAreaExpression (c : CubeWithTunnel) : ℕ × ℕ × ℕ :=
  sorry

theorem cube_with_tunnel_surface_area 
  (c : CubeWithTunnel)
  (h1 : c.edgeLength = 10)
  (h2 : c.p.x = 10 ∧ c.p.y = 10 ∧ c.p.z = 10)
  (h3 : c.l.x = 7.5 ∧ c.l.y = 10 ∧ c.l.z = 10)
  (h4 : c.m.x = 10 ∧ c.m.y = 7.5 ∧ c.m.z = 10)
  (h5 : c.n.x = 10 ∧ c.n.y = 10 ∧ c.n.z = 7.5) :
  let (x, y, z) := surfaceAreaExpression c
  x + y + z = 639 ∧ 
  (∀ p : ℕ, Prime p → ¬(p^2 ∣ z)) :=
sorry

end NUMINAMATH_CALUDE_cube_with_tunnel_surface_area_l3604_360483


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l3604_360429

/-- Given a line passing through points (1, 3) and (3, 7), prove that m + b = 3 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) → (7 = m * 3 + b) → m + b = 3 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l3604_360429


namespace NUMINAMATH_CALUDE_b_10_value_l3604_360490

theorem b_10_value (a b : ℕ → ℝ) 
  (h1 : ∀ n, (a n) * (a (n + 1)) = 2^n)
  (h2 : ∀ n, (a n) + (a (n + 1)) = b n)
  (h3 : a 1 = 1) :
  b 10 = 64 := by
sorry

end NUMINAMATH_CALUDE_b_10_value_l3604_360490


namespace NUMINAMATH_CALUDE_exists_multiple_of_E_l3604_360437

def E (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => 2 * (i + 1))

def D (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun i => 2 * i + 1)

theorem exists_multiple_of_E (n : ℕ) : ∃ m : ℕ, ∃ k : ℕ, D n * 2^m = k * E n := by
  sorry

end NUMINAMATH_CALUDE_exists_multiple_of_E_l3604_360437


namespace NUMINAMATH_CALUDE_no_solution_equation_one_solutions_equation_two_l3604_360463

-- Problem 1
theorem no_solution_equation_one : 
  ¬ ∃ x : ℝ, (1 / (x - 2) + 2 = (1 - x) / (2 - x)) ∧ (x ≠ 2) :=
sorry

-- Problem 2
theorem solutions_equation_two :
  ∀ x : ℝ, (x - 4)^2 = 4*(2*x + 1)^2 ↔ x = 2/5 ∨ x = -2 :=
sorry

end NUMINAMATH_CALUDE_no_solution_equation_one_solutions_equation_two_l3604_360463


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3604_360453

/-- Given two quadratic equations, where the roots of the second are each three less than
    the roots of the first, this theorem proves that the constant term of the second
    equation is 3.5. -/
theorem quadratic_root_relation (d : ℝ) :
  (∃ r s : ℝ, r + s = 2 ∧ r * s = 1/2 ∧ 
   ∀ x : ℝ, 4 * x^2 - 8 * x + 2 = 0 ↔ (x = r ∨ x = s)) →
  (∃ e : ℝ, ∀ x : ℝ, x^2 + d * x + e = 0 ↔ (x = r - 3 ∨ x = s - 3)) →
  ∃ e : ℝ, e = 3.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3604_360453


namespace NUMINAMATH_CALUDE_min_chinese_score_l3604_360415

/-- Represents the scores of a student in three subjects -/
structure Scores where
  chinese : ℝ
  mathematics : ℝ
  english : ℝ

/-- The average score of the three subjects is 92 -/
def average_score (s : Scores) : Prop :=
  (s.chinese + s.mathematics + s.english) / 3 = 92

/-- Each subject has a maximum score of 100 points -/
def max_score (s : Scores) : Prop :=
  s.chinese ≤ 100 ∧ s.mathematics ≤ 100 ∧ s.english ≤ 100

/-- The Mathematics score is 4 points higher than the Chinese score -/
def math_chinese_relation (s : Scores) : Prop :=
  s.mathematics = s.chinese + 4

/-- The minimum possible score for Chinese is 86 points -/
theorem min_chinese_score (s : Scores) 
  (h1 : average_score s) 
  (h2 : max_score s) 
  (h3 : math_chinese_relation s) : 
  s.chinese ≥ 86 := by
  sorry

end NUMINAMATH_CALUDE_min_chinese_score_l3604_360415


namespace NUMINAMATH_CALUDE_min_distance_for_ten_trees_l3604_360441

/-- Calculates the minimum distance to water trees in a row -/
def min_watering_distance (num_trees : ℕ) (tree_distance : ℕ) : ℕ :=
  let well_to_tree := tree_distance
  let tree_to_well := tree_distance
  let full_trips := (num_trees - 1) / 2
  let full_trip_distance := full_trips * (well_to_tree + tree_to_well)
  let remaining_trees := (num_trees - 1) % 2
  let last_trip_distance := remaining_trees * (well_to_tree + tree_to_well)
  full_trip_distance + last_trip_distance + (num_trees - 1) * tree_distance

theorem min_distance_for_ten_trees :
  min_watering_distance 10 10 = 410 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_for_ten_trees_l3604_360441


namespace NUMINAMATH_CALUDE_fraction_increase_condition_l3604_360481

theorem fraction_increase_condition (m n : ℤ) (h1 : n ≠ 0) (h2 : n ≠ -1) :
  (m : ℚ) / n < (m + 1 : ℚ) / (n + 1) ↔ (n > 0 ∧ m < n) ∨ (n < -1 ∧ m > n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_increase_condition_l3604_360481


namespace NUMINAMATH_CALUDE_building_stories_l3604_360461

theorem building_stories (apartments_per_floor : ℕ) (people_per_apartment : ℕ) (total_people : ℕ) :
  apartments_per_floor = 4 →
  people_per_apartment = 2 →
  total_people = 200 →
  total_people / (apartments_per_floor * people_per_apartment) = 25 :=
by sorry

end NUMINAMATH_CALUDE_building_stories_l3604_360461


namespace NUMINAMATH_CALUDE_a_share_of_profit_l3604_360439

/-- Calculates the share of profit for a partner in a business partnership --/
def calculateShareOfProfit (investmentA investmentB investmentC totalProfit : ℕ) : ℕ :=
  let totalInvestment := investmentA + investmentB + investmentC
  (investmentA * totalProfit) / totalInvestment

/-- Theorem stating that A's share of the profit is 4260 --/
theorem a_share_of_profit :
  calculateShareOfProfit 6300 4200 10500 14200 = 4260 := by
  sorry

#eval calculateShareOfProfit 6300 4200 10500 14200

end NUMINAMATH_CALUDE_a_share_of_profit_l3604_360439


namespace NUMINAMATH_CALUDE_reflected_light_ray_equation_l3604_360480

/-- Given an incident light ray along y = 2x + 1 reflected by the line y = x,
    the equation of the reflected light ray is x - 2y - 1 = 0 -/
theorem reflected_light_ray_equation (x y : ℝ) : 
  (y = 2*x + 1) → -- incident light ray equation
  (y = x) →       -- reflecting line equation
  (x - 2*y - 1 = 0) -- reflected light ray equation
  := by sorry

end NUMINAMATH_CALUDE_reflected_light_ray_equation_l3604_360480


namespace NUMINAMATH_CALUDE_triangle_configuration_l3604_360459

/-- Represents a triangle with side lengths x, y, and z -/
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  hx : x > 0
  hy : y > 0
  hz : z > 0
  hxy : x < y + z
  hyz : y < x + z
  hzx : z < x + y

/-- Theorem about a specific triangle configuration -/
theorem triangle_configuration (a : ℝ) : 
  ∃ (t : Triangle), 
    t.x + t.y = 3 * t.z ∧ 
    t.z + t.y = t.x + a ∧ 
    t.x + t.z = 60 → 
    (0 < a ∧ a < 60) ∧
    (a = 30 → t.x = 42 ∧ t.y = 48 ∧ t.z = 30) := by
  sorry

#check triangle_configuration

end NUMINAMATH_CALUDE_triangle_configuration_l3604_360459


namespace NUMINAMATH_CALUDE_bee_hive_population_l3604_360496

/-- The population growth function for bees in a hive -/
def bee_population (initial : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial * growth_factor ^ days

/-- Theorem stating the population of bees after 20 days -/
theorem bee_hive_population :
  bee_population 1 5 20 = 5^20 := by
  sorry

end NUMINAMATH_CALUDE_bee_hive_population_l3604_360496


namespace NUMINAMATH_CALUDE_remove_five_for_target_average_l3604_360428

def original_list : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def target_average : Rat := 41/5  -- 8.2 as a rational number

theorem remove_five_for_target_average :
  let remaining_list := original_list.filter (· ≠ 5)
  (remaining_list.sum : Rat) / remaining_list.length = target_average := by
  sorry

end NUMINAMATH_CALUDE_remove_five_for_target_average_l3604_360428


namespace NUMINAMATH_CALUDE_solution_set_l3604_360452

theorem solution_set (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3 →
  x ≥ 18 := by
sorry

end NUMINAMATH_CALUDE_solution_set_l3604_360452


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3604_360469

-- Define the quadratic function
def f (m x : ℝ) : ℝ := (m + 1) * x^2 + (m^2 - 2*m - 3) * x - m + 3

-- State the theorem
theorem quadratic_inequality_range (m : ℝ) :
  (∀ x, f m x > 0) ↔ (m ∈ Set.Icc (-1) 1 ∪ Set.Ioo 1 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3604_360469


namespace NUMINAMATH_CALUDE_rectangle_area_properties_l3604_360458

-- Define the rectangle's dimensions and measurement errors
def expected_length : Real := 2
def expected_width : Real := 1
def length_std_dev : Real := 0.003
def width_std_dev : Real := 0.002

-- Define the theorem
theorem rectangle_area_properties :
  let expected_area := expected_length * expected_width
  let area_variance := (expected_length^2 * width_std_dev^2) + (expected_width^2 * length_std_dev^2) + (length_std_dev^2 * width_std_dev^2)
  let area_std_dev := Real.sqrt area_variance
  (expected_area = 2) ∧ (area_std_dev * 100 = 5) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_properties_l3604_360458


namespace NUMINAMATH_CALUDE_duck_ratio_l3604_360478

theorem duck_ratio (total_birds : ℕ) (chicken_feed_cost : ℚ) (total_chicken_feed_cost : ℚ) :
  total_birds = 15 →
  chicken_feed_cost = 2 →
  total_chicken_feed_cost = 20 →
  (total_birds - (total_chicken_feed_cost / chicken_feed_cost)) / total_birds = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_duck_ratio_l3604_360478


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3604_360420

def A : Set Char := {'a', 'b', 'c', 'd'}
def B : Set Char := {'b', 'c', 'd', 'e'}

theorem intersection_of_A_and_B :
  A ∩ B = {'b', 'c', 'd'} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3604_360420


namespace NUMINAMATH_CALUDE_triangle_properties_l3604_360403

-- Define the triangle
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b)
  (h2 : Real.cos t.B = 1/4)
  (h3 : t.b = 2) :
  Real.sin t.C / Real.sin t.A = 2 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3604_360403


namespace NUMINAMATH_CALUDE_unique_prime_with_remainder_l3604_360482

theorem unique_prime_with_remainder : ∃! p : ℕ, 
  Prime p ∧ 
  20 < p ∧ p < 35 ∧ 
  p % 11 = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_remainder_l3604_360482


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3604_360445

theorem rectangle_dimensions (x : ℝ) : 
  (x - 3) * (3*x + 4) = 9*x - 19 → x = (7 + 2*Real.sqrt 7) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3604_360445


namespace NUMINAMATH_CALUDE_abs_sum_inequality_range_l3604_360485

theorem abs_sum_inequality_range :
  {x : ℝ | |x + 1| + |x| < 2} = Set.Ioo (-3/2 : ℝ) (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_range_l3604_360485


namespace NUMINAMATH_CALUDE_bruce_purchase_l3604_360457

/-- Calculates the total amount Bruce paid for grapes and mangoes -/
def totalAmountPaid (grapeQuantity : ℕ) (grapeRate : ℕ) (mangoQuantity : ℕ) (mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Proves that Bruce paid 985 for his purchase of grapes and mangoes -/
theorem bruce_purchase : totalAmountPaid 7 70 9 55 = 985 := by
  sorry

end NUMINAMATH_CALUDE_bruce_purchase_l3604_360457


namespace NUMINAMATH_CALUDE_sin_cos_derivative_l3604_360460

theorem sin_cos_derivative (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos x ^ 2 - Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_derivative_l3604_360460


namespace NUMINAMATH_CALUDE_range_of_a_for_two_zeros_l3604_360440

noncomputable def f (a x : ℝ) : ℝ := 
  if x ≥ a then x else x^3 - 3*x

noncomputable def g (a x : ℝ) : ℝ := 2 * f a x - a * x

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ g a x = 0 ∧ g a y = 0) ∧
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → ¬(g a x = 0 ∧ g a y = 0 ∧ g a z = 0)) →
  a > -3/2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_zeros_l3604_360440


namespace NUMINAMATH_CALUDE_correct_operation_l3604_360413

theorem correct_operation (a b : ℝ) : (a - b) * (2 * a + 2 * b) = 2 * a^2 - 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3604_360413


namespace NUMINAMATH_CALUDE_percentage_defective_meters_l3604_360493

theorem percentage_defective_meters 
  (total_meters : ℕ) 
  (rejected_meters : ℕ) 
  (h1 : total_meters = 2500) 
  (h2 : rejected_meters = 2) : 
  (rejected_meters : ℝ) / total_meters * 100 = 0.08 := by
sorry

end NUMINAMATH_CALUDE_percentage_defective_meters_l3604_360493


namespace NUMINAMATH_CALUDE_intersection_points_count_l3604_360405

/-- The number of intersection points between y = Bx^2 and y^2 + 4y - 2 = x^2 + 5y -/
theorem intersection_points_count (B : ℝ) (hB : B > 0) : 
  ∃ (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ),
    (y1 = B * x1^2 ∧ y1^2 + 4*y1 - 2 = x1^2 + 5*y1) ∧
    (y2 = B * x2^2 ∧ y2^2 + 4*y2 - 2 = x2^2 + 5*y2) ∧
    (y3 = B * x3^2 ∧ y3^2 + 4*y3 - 2 = x3^2 + 5*y3) ∧
    (y4 = B * x4^2 ∧ y4^2 + 4*y4 - 2 = x4^2 + 5*y4) ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x1 ≠ x4 ∨ y1 ≠ y4) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3) ∧
    (x2 ≠ x4 ∨ y2 ≠ y4) ∧
    (x3 ≠ x4 ∨ y3 ≠ y4) ∧
    ∀ (x y : ℝ), (y = B * x^2 ∧ y^2 + 4*y - 2 = x^2 + 5*y) →
      ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l3604_360405


namespace NUMINAMATH_CALUDE_correct_calculation_l3604_360470

theorem correct_calculation : (-0.5)^2010 * 2^2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3604_360470


namespace NUMINAMATH_CALUDE_train_length_proof_l3604_360421

/-- The length of two trains that pass each other under specific conditions -/
theorem train_length_proof (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 36) : 
  let L := (v_fast - v_slow) * t / (2 * 3600)
  L * 1000 = 50 := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l3604_360421


namespace NUMINAMATH_CALUDE_geometric_transformations_l3604_360426

-- Define the basic geometric entities
structure Point

structure Line

structure Surface

structure Body

-- Define the movement operation
def moves (a : Type) (b : Type) : Prop :=
  ∃ (x : a), ∃ (y : b), true

-- Theorem statement
theorem geometric_transformations :
  (moves Point Line) ∧
  (moves Line Surface) ∧
  (moves Surface Body) := by
  sorry

end NUMINAMATH_CALUDE_geometric_transformations_l3604_360426


namespace NUMINAMATH_CALUDE_curve_transformation_l3604_360438

/-- The matrix A --/
def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; 0, 1]

/-- The original curve C --/
def C (x y : ℝ) : Prop := (x - y)^2 + y^2 = 1

/-- The transformed curve C' --/
def C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Theorem stating that C' is the result of transforming C under A --/
theorem curve_transformation (x y : ℝ) : 
  C' x y ↔ ∃ x₀ y₀ : ℝ, C x₀ y₀ ∧ A.mulVec ![x₀, y₀] = ![x, y] :=
sorry

end NUMINAMATH_CALUDE_curve_transformation_l3604_360438


namespace NUMINAMATH_CALUDE_vector_ab_coordinates_l3604_360422

/-- Given a vector AB in 2D space, prove statements about the coordinates of points A and B when one is at the origin. -/
theorem vector_ab_coordinates (ab : Fin 2 → ℝ) (h : ab = ![(-2), 4]) :
  (∀ (b : Fin 2 → ℝ), b = ![0, 0] → ∃ (a : Fin 2 → ℝ), a = ![2, -4] ∧ ab = a - b) ∧
  (∀ (a : Fin 2 → ℝ), a = ![0, 0] → ∃ (b : Fin 2 → ℝ), b = ![(-2), 4] ∧ ab = b - a) :=
by sorry

end NUMINAMATH_CALUDE_vector_ab_coordinates_l3604_360422


namespace NUMINAMATH_CALUDE_unique_quadratic_polynomial_l3604_360435

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  b : ℝ
  c : ℝ

/-- The roots of a quadratic polynomial -/
def roots (p : QuadraticPolynomial) : Set ℝ :=
  {x : ℝ | x^2 + p.b * x + p.c = 0}

/-- The set of coefficients of a quadratic polynomial -/
def coefficients (p : QuadraticPolynomial) : Set ℝ :=
  {1, p.b, p.c}

/-- The theorem stating that there exists exactly one quadratic polynomial
    satisfying the given conditions -/
theorem unique_quadratic_polynomial :
  ∃! p : QuadraticPolynomial, roots p = coefficients p :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_polynomial_l3604_360435


namespace NUMINAMATH_CALUDE_sine_ratio_zero_l3604_360491

theorem sine_ratio_zero (c : Real) (h : c = π / 12) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c)) /
  (Real.sin (2 * c) * Real.sin (4 * c) * Real.sin (6 * c)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_ratio_zero_l3604_360491


namespace NUMINAMATH_CALUDE_rose_count_l3604_360404

theorem rose_count : ∃ (n : ℕ), 
  300 ≤ n ∧ n ≤ 400 ∧ 
  ∃ (x y : ℕ), n = 21 * x + 13 ∧ n = 15 * y - 8 ∧
  n = 307 := by
  sorry

end NUMINAMATH_CALUDE_rose_count_l3604_360404


namespace NUMINAMATH_CALUDE_david_math_homework_time_l3604_360471

/-- Given David's homework times, prove he spent 15 minutes on math. -/
theorem david_math_homework_time :
  ∀ (total_time spelling_time reading_time math_time : ℕ),
    total_time = 60 →
    spelling_time = 18 →
    reading_time = 27 →
    math_time = total_time - spelling_time - reading_time →
    math_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_david_math_homework_time_l3604_360471


namespace NUMINAMATH_CALUDE_skew_diagonals_properties_l3604_360414

/-- A cube with edge length 1 -/
structure UnitCube where
  edge_length : ℝ
  edge_length_eq_one : edge_length = 1

/-- Skew diagonals of two adjacent faces of a unit cube -/
structure SkewDiagonals (cube : UnitCube) where
  angle : ℝ
  distance : ℝ

/-- Theorem about the properties of skew diagonals in a unit cube -/
theorem skew_diagonals_properties (cube : UnitCube) :
  ∃ (sd : SkewDiagonals cube),
    sd.angle = Real.pi / 3 ∧ sd.distance = 1 / Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_skew_diagonals_properties_l3604_360414


namespace NUMINAMATH_CALUDE_derivative_cube_of_linear_l3604_360423

theorem derivative_cube_of_linear (x : ℝ) :
  deriv (λ x => (1 + 5*x)^3) x = 15 * (1 + 5*x)^2 := by sorry

end NUMINAMATH_CALUDE_derivative_cube_of_linear_l3604_360423


namespace NUMINAMATH_CALUDE_ellipse_m_values_l3604_360494

def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / 12 + y^2 / m = 1

def eccentricity (e : ℝ) : Prop :=
  e = 1/2

theorem ellipse_m_values (m : ℝ) :
  (∃ x y, ellipse_equation x y m) ∧ (∃ e, eccentricity e) →
  m = 9 ∨ m = 16 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_values_l3604_360494


namespace NUMINAMATH_CALUDE_olympic_inequality_l3604_360475

theorem olympic_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) :
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 > 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end NUMINAMATH_CALUDE_olympic_inequality_l3604_360475


namespace NUMINAMATH_CALUDE_simplify_expression_l3604_360401

theorem simplify_expression : Real.sqrt ((Real.pi - 4) ^ 2) + (Real.pi - 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3604_360401


namespace NUMINAMATH_CALUDE_square_sum_value_l3604_360473

theorem square_sum_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y + x + y = 71) (h2 : x^2 * y + x * y^2 = 880) :
  x^2 + y^2 = 146 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l3604_360473


namespace NUMINAMATH_CALUDE_inequality_problem_l3604_360467

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a * c < b * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l3604_360467


namespace NUMINAMATH_CALUDE_zoo_field_trip_l3604_360486

theorem zoo_field_trip (students : ℕ) (adults : ℕ) (vans : ℕ) :
  students = 40 →
  adults = 14 →
  vans = 6 →
  (students + adults) / vans = 9 :=
by sorry

end NUMINAMATH_CALUDE_zoo_field_trip_l3604_360486


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l3604_360432

theorem quadratic_equation_integer_solutions (k : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ (k^2 - 1) * x^2 - 6 * (3 * k - 1) * x + 72 = 0) ↔ 
  k = 1 ∨ k = 2 ∨ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l3604_360432


namespace NUMINAMATH_CALUDE_negation_empty_subset_any_set_l3604_360456

theorem negation_empty_subset_any_set :
  (¬ ∀ A : Set α, ∅ ⊆ A) ↔ (∃ A : Set α, ¬(∅ ⊆ A)) :=
by sorry

end NUMINAMATH_CALUDE_negation_empty_subset_any_set_l3604_360456


namespace NUMINAMATH_CALUDE_coordinates_of_P_l3604_360466

-- Define points M and N
def M : ℝ × ℝ := (3, 2)
def N : ℝ × ℝ := (-5, -5)

-- Define vector from M to N
def MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define point P
def P : ℝ × ℝ := (x, y) where
  x : ℝ := sorry
  y : ℝ := sorry

-- Define vector from M to P
def MP : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)

-- Theorem statement
theorem coordinates_of_P :
  MP = (1/2 : ℝ) • MN → P = (-1, -3/2) := by sorry

end NUMINAMATH_CALUDE_coordinates_of_P_l3604_360466


namespace NUMINAMATH_CALUDE_expression_evaluation_l3604_360424

theorem expression_evaluation : -30 + 5 * (9 / (3 + 3)) = -22.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3604_360424


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l3604_360498

-- Problem 1
theorem simplify_fraction_1 (a : ℝ) (h : a ≠ 1) :
  (a^2 / (a - 1)) - (a / (a - 1)) = a :=
sorry

-- Problem 2
theorem simplify_fraction_2 (x : ℝ) (h : x ≠ -1) :
  (x^2 / (x + 1)) - x + 1 = 1 / (x + 1) :=
sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l3604_360498


namespace NUMINAMATH_CALUDE_point_on_line_l3604_360468

/-- Given a line L with equation Ax + By + C = 0 that can be rewritten as A(x - x₀) + B(y - y₀) = 0,
    prove that the point (x₀, y₀) lies on the line L. -/
theorem point_on_line (A B C x₀ y₀ : ℝ) :
  (∀ x y, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0) →
  A * x₀ + B * y₀ + C = 0 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l3604_360468


namespace NUMINAMATH_CALUDE_gcd_8164_2937_l3604_360410

theorem gcd_8164_2937 : Nat.gcd 8164 2937 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8164_2937_l3604_360410


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3604_360449

theorem complex_equation_solution :
  ∀ z : ℂ, z + Complex.abs z = 2 + Complex.I → z = (3/4 : ℝ) + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3604_360449


namespace NUMINAMATH_CALUDE_pigsy_fruits_l3604_360425

def process (n : ℕ) : ℕ := 
  (n / 2 + 2) / 2

theorem pigsy_fruits : ∃ x : ℕ, process (process (process (process x))) = 5 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_pigsy_fruits_l3604_360425


namespace NUMINAMATH_CALUDE_fiftiethTermIs346_l3604_360430

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
def fiftiethTerm : ℤ := arithmeticSequenceTerm 3 7 50

theorem fiftiethTermIs346 : fiftiethTerm = 346 := by
  sorry

end NUMINAMATH_CALUDE_fiftiethTermIs346_l3604_360430


namespace NUMINAMATH_CALUDE_ad_agency_client_distribution_l3604_360418

/-- Given an advertising agency with 180 clients, where:
    - 115 use television
    - 110 use radio
    - 130 use magazines
    - 85 use television and magazines
    - 75 use television and radio
    - 80 use all three
    This theorem proves that 95 clients use radio and magazines. -/
theorem ad_agency_client_distribution (total : ℕ) (T R M TM TR TRM : ℕ) 
  (h_total : total = 180)
  (h_T : T = 115)
  (h_R : R = 110)
  (h_M : M = 130)
  (h_TM : TM = 85)
  (h_TR : TR = 75)
  (h_TRM : TRM = 80)
  : total = T + R + M - TR - TM - (T + R + M - TR - TM - total + TRM) + TRM := by
  sorry

end NUMINAMATH_CALUDE_ad_agency_client_distribution_l3604_360418


namespace NUMINAMATH_CALUDE_jogger_speed_l3604_360455

/-- The speed of a jogger on a path with specific conditions -/
theorem jogger_speed (inner_perimeter outer_perimeter : ℝ) 
  (h1 : outer_perimeter - inner_perimeter = 16 * Real.pi)
  (time_diff : ℝ) (h2 : time_diff = 60) :
  ∃ (speed : ℝ), speed = (4 * Real.pi) / 15 ∧ 
    outer_perimeter / speed = inner_perimeter / speed + time_diff :=
sorry

end NUMINAMATH_CALUDE_jogger_speed_l3604_360455


namespace NUMINAMATH_CALUDE_g_value_at_9_l3604_360476

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- Define the properties of g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^3 + b*x^2 + c*x + d) ∧  -- g is a cubic polynomial
  (g 0 = -1) ∧  -- g(0) = -1
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g s = 0 ∧ s = r^2)  -- roots of g are squares of roots of f

-- State the theorem
theorem g_value_at_9 (g : ℝ → ℝ) (hg : g_properties g) : g 9 = 899 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_9_l3604_360476


namespace NUMINAMATH_CALUDE_total_study_time_is_three_l3604_360427

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The time Sam spends studying Science in minutes -/
def science_time : ℕ := 60

/-- The time Sam spends studying Math in minutes -/
def math_time : ℕ := 80

/-- The time Sam spends studying Literature in minutes -/
def literature_time : ℕ := 40

/-- The total time Sam spends studying in hours -/
def total_study_time : ℚ :=
  (science_time + math_time + literature_time) / minutes_per_hour

theorem total_study_time_is_three : total_study_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_total_study_time_is_three_l3604_360427


namespace NUMINAMATH_CALUDE_complex_power_sum_l3604_360447

theorem complex_power_sum (z : ℂ) (h : z^2 + z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 1 + z := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3604_360447


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3604_360436

/-- Calculate the interest rate per annum given the principal, amount, and time period. -/
theorem interest_rate_calculation (principal amount : ℕ) (time : ℚ) :
  principal = 1100 →
  amount = 1232 →
  time = 12 / 5 →
  (amount - principal) * 100 / (principal * time) = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3604_360436


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l3604_360417

theorem systematic_sampling_proof (N n : ℕ) (hN : N = 92) (hn : n = 30) :
  let k := N / n
  (k = 3) ∧ (k - 1 = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l3604_360417


namespace NUMINAMATH_CALUDE_poster_difference_l3604_360454

/-- The number of posters Mario made -/
def mario_posters : ℕ := 18

/-- The total number of posters made by Mario and Samantha -/
def total_posters : ℕ := 51

/-- The number of posters Samantha made -/
def samantha_posters : ℕ := total_posters - mario_posters

/-- Samantha made more posters than Mario -/
axiom samantha_made_more : samantha_posters > mario_posters

theorem poster_difference : samantha_posters - mario_posters = 15 := by
  sorry

end NUMINAMATH_CALUDE_poster_difference_l3604_360454


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3604_360434

theorem max_x_minus_y (x y : Real) (h1 : 0 < y) (h2 : y ≤ x) (h3 : x < π/2) (h4 : Real.tan x = 3 * Real.tan y) :
  ∃ (max_val : Real), max_val = π/6 ∧ x - y ≤ max_val ∧ ∃ (x' y' : Real), 0 < y' ∧ y' ≤ x' ∧ x' < π/2 ∧ Real.tan x' = 3 * Real.tan y' ∧ x' - y' = max_val :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3604_360434


namespace NUMINAMATH_CALUDE_candy_distribution_l3604_360487

/-- Calculates the number of candy pieces each student receives -/
def candy_per_student (total : ℕ) (reserved : ℕ) (students : ℕ) : ℕ :=
  (total - reserved) / students

/-- Proves that each student receives 6 pieces of candy -/
theorem candy_distribution (total : ℕ) (reserved : ℕ) (students : ℕ) 
  (h1 : total = 344) 
  (h2 : reserved = 56) 
  (h3 : students = 43) : 
  candy_per_student total reserved students = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3604_360487


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3604_360409

theorem min_value_quadratic (x y : ℝ) : 5 * x^2 - 4 * x * y + y^2 + 6 * x + 25 ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3604_360409


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l3604_360488

theorem divisibility_by_seven (n : ℕ) : 
  7 ∣ n ↔ 7 ∣ ((n / 10) - 2 * (n % 10)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l3604_360488


namespace NUMINAMATH_CALUDE_line_equation_through_intersection_and_parallel_l3604_360412

/-- Given two lines in the plane and a third line parallel to one of them,
    this theorem proves the equation of the third line. -/
theorem line_equation_through_intersection_and_parallel
  (l₁ l₂ l₃ l : Real → Real → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ 3 * x + 5 * y - 4 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 6 * x - y + 3 = 0)
  (h₃ : ∀ x y, l₃ x y ↔ 2 * x + 3 * y + 5 = 0)
  (h_intersect : ∃ x y, l₁ x y ∧ l₂ x y ∧ l x y)
  (h_parallel : ∃ k ≠ 0, ∀ x y, l x y ↔ 2 * k * x + 3 * k * y + (k * 5 + c) = 0) :
  ∀ x y, l x y ↔ 6 * x + 9 * y - 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_line_equation_through_intersection_and_parallel_l3604_360412


namespace NUMINAMATH_CALUDE_y1_gt_y2_l3604_360499

/-- A linear function that does not pass through the third quadrant -/
structure LinearFunctionNotInThirdQuadrant where
  k : ℝ
  b : ℝ
  not_in_third_quadrant : k < 0

/-- The function corresponding to the LinearFunctionNotInThirdQuadrant -/
def f (l : LinearFunctionNotInThirdQuadrant) (x : ℝ) : ℝ :=
  l.k * x + l.b

/-- Theorem stating that y₁ > y₂ for the given conditions -/
theorem y1_gt_y2 (l : LinearFunctionNotInThirdQuadrant) (y₁ y₂ : ℝ)
    (h1 : f l (-1) = y₁)
    (h2 : f l 1 = y₂) :
    y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_gt_y2_l3604_360499


namespace NUMINAMATH_CALUDE_bagel_cut_theorem_l3604_360411

/-- The number of pieces resulting from cutting a bagel -/
def bagel_pieces (n : ℕ) : ℕ := n + 1

/-- Theorem: Cutting a bagel with 10 cuts results in 11 pieces -/
theorem bagel_cut_theorem :
  bagel_pieces 10 = 11 :=
by sorry

end NUMINAMATH_CALUDE_bagel_cut_theorem_l3604_360411


namespace NUMINAMATH_CALUDE_seventh_observation_value_l3604_360400

theorem seventh_observation_value (initial_count : Nat) (initial_average : ℝ) (new_average : ℝ) :
  initial_count = 6 →
  initial_average = 14 →
  new_average = 13 →
  (initial_count * initial_average + 7) / (initial_count + 1) = new_average →
  7 = (initial_count + 1) * new_average - initial_count * initial_average :=
by sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l3604_360400


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3604_360477

theorem solution_set_inequality (x : ℝ) : 
  (Set.Ioo (-2 : ℝ) 0).Nonempty ∧ 
  (∀ y ∈ Set.Ioo (-2 : ℝ) 0, |1 + y + y^2/2| < 1) ∧
  (∀ z : ℝ, z ∉ Set.Ioo (-2 : ℝ) 0 → |1 + z + z^2/2| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3604_360477


namespace NUMINAMATH_CALUDE_function_derivative_at_midpoint_negative_l3604_360407

open Real

theorem function_derivative_at_midpoint_negative 
  (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : 0 < x₁ ∧ x₁ < x₂) 
  (hf : ∀ x, f x = log x - a * x + 1) 
  (hz : f x₁ = 0 ∧ f x₂ = 0) : 
  deriv f ((x₁ + x₂) / 2) < 0 := by
  sorry


end NUMINAMATH_CALUDE_function_derivative_at_midpoint_negative_l3604_360407


namespace NUMINAMATH_CALUDE_intersection_distance_to_side_l3604_360431

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  sideLength : ℝ
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

def intersectionPoint (c1 c2 : Circle) : Point := sorry

/-- Calculates the distance between a point and a line defined by y = k -/
def distanceToHorizontalLine (p : Point) (k : ℝ) : ℝ := sorry

theorem intersection_distance_to_side (s : Square) 
  (c1 c2 : Circle) (h1 : s.sideLength = 10) 
  (h2 : c1.center = s.A) (h3 : c2.center = s.B) 
  (h4 : c1.radius = 5) (h5 : c2.radius = 5) :
  let X := intersectionPoint c1 c2
  distanceToHorizontalLine X s.sideLength = 10 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_to_side_l3604_360431


namespace NUMINAMATH_CALUDE_inequality_proof_l3604_360495

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c)/(a + 2*b + c) + 4*b/(a + b + 2*c) - 8*c/(a + b + 3*c) ≥ -17 + 12*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3604_360495


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3604_360464

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ 2374) ∧ (1275890 + x) % 2375 = 0 ∧ 
  ∀ y : ℕ, y < x → (1275890 + y) % 2375 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3604_360464


namespace NUMINAMATH_CALUDE_function_properties_l3604_360484

def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b*x + 3

theorem function_properties (b : ℝ) :
  f b 0 = f b 4 →
  (b = 4 ∧
   (∀ x, f b x = 0 ↔ x = 1 ∨ x = 3) ∧
   (∀ x, f b x < 0 ↔ 1 < x ∧ x < 3) ∧
   (∀ x ∈ Set.Icc 0 3, f b x ≥ -1) ∧
   (∃ x ∈ Set.Icc 0 3, f b x = -1) ∧
   (∀ x ∈ Set.Icc 0 3, f b x ≤ 3) ∧
   (∃ x ∈ Set.Icc 0 3, f b x = 3)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3604_360484


namespace NUMINAMATH_CALUDE_apple_count_l3604_360416

theorem apple_count (apples oranges : ℕ) : 
  oranges = 20 → 
  (apples : ℚ) / (apples + (oranges - 14 : ℚ)) = 7/10 → 
  apples = 14 := by
sorry

end NUMINAMATH_CALUDE_apple_count_l3604_360416


namespace NUMINAMATH_CALUDE_negative_two_minus_six_l3604_360472

theorem negative_two_minus_six : -2 - 6 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_minus_six_l3604_360472


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3604_360406

/-- Represents a geometric sequence. -/
structure GeometricSequence where
  a : ℕ → ℝ
  r : ℝ
  h1 : ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence {a_n} with a_1 = 2 and a_3a_5 = 4a_6^2, prove that a_3 = 1. -/
theorem geometric_sequence_problem (seq : GeometricSequence)
  (h2 : seq.a 1 = 2)
  (h3 : seq.a 3 * seq.a 5 = 4 * (seq.a 6)^2) :
  seq.a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3604_360406


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3604_360450

theorem trig_expression_simplification :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 = 
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3604_360450


namespace NUMINAMATH_CALUDE_expression_equals_one_l3604_360474

theorem expression_equals_one :
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3604_360474


namespace NUMINAMATH_CALUDE_g_range_l3604_360497

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 - 2*Real.pi * Real.arcsin (x/3) + (Real.arcsin (x/3))^2 + 
  (Real.pi^2/8) * (x^2 - 4*x + 12)

theorem g_range :
  ∀ x ∈ Set.Icc (-3 : ℝ) 3,
  g x ∈ Set.Icc (Real.pi^2/4 + 9*Real.pi^2/8) (Real.pi^2/4 + 33*Real.pi^2/8) :=
by sorry

end NUMINAMATH_CALUDE_g_range_l3604_360497


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3604_360443

def kilowatt_hours : ℝ := 448000

theorem scientific_notation_equivalence : 
  kilowatt_hours = 4.48 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3604_360443


namespace NUMINAMATH_CALUDE_kids_wearing_socks_and_shoes_l3604_360402

/-- Given a classroom with kids wearing socks, shoes, or barefoot, 
    prove the number of kids wearing both socks and shoes. -/
theorem kids_wearing_socks_and_shoes 
  (total : ℕ) 
  (socks : ℕ) 
  (shoes : ℕ) 
  (barefoot : ℕ) 
  (h1 : total = 22) 
  (h2 : socks = 12) 
  (h3 : shoes = 8) 
  (h4 : barefoot = 8) 
  (h5 : total = socks + barefoot) 
  (h6 : total = shoes + barefoot) :
  shoes = socks + shoes - total := by
sorry

end NUMINAMATH_CALUDE_kids_wearing_socks_and_shoes_l3604_360402


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3604_360451

theorem sufficient_not_necessary :
  (∀ a b : ℝ, a > 2 ∧ b > 1 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 2 ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3604_360451


namespace NUMINAMATH_CALUDE_initial_peanuts_l3604_360442

theorem initial_peanuts (added : ℕ) (final : ℕ) (h1 : added = 6) (h2 : final = 10) :
  final - added = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_peanuts_l3604_360442


namespace NUMINAMATH_CALUDE_odd_function_value_l3604_360408

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x < 0, f x = 2^x) →      -- f(x) = 2^x for x < 0
  f (Real.log 9 / Real.log 4) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_odd_function_value_l3604_360408


namespace NUMINAMATH_CALUDE_sock_pair_count_l3604_360433

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_pairs (white brown blue red : ℕ) : ℕ :=
  white * brown + white * blue + white * red +
  brown * blue + brown * red +
  blue * red

/-- Theorem: Given 5 white socks, 5 brown socks, 2 blue socks, and 1 red sock,
    the number of ways to choose a pair of socks with different colors is 57 -/
theorem sock_pair_count :
  different_color_pairs 5 5 2 1 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l3604_360433


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l3604_360419

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 4*x + 5

theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3*x - y + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l3604_360419


namespace NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_b_value_l3604_360446

/-- A cubic polynomial with coefficient b -/
def cubic (x b : ℂ) : ℂ := x^3 - 9*x^2 + 33*x + b

/-- Predicate to check if three complex numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℂ) : Prop := b - a = c - b

/-- Theorem stating that if the roots of the cubic form an arithmetic progression
    and at least one root is non-real, then b = -15 -/
theorem cubic_roots_arithmetic_progression_b_value (b : ℝ) :
  (∃ (r₁ r₂ r₃ : ℂ), 
    (∀ x, cubic x b = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ 
    isArithmeticProgression r₁ r₂ r₃ ∧
    (r₁.im ≠ 0 ∨ r₂.im ≠ 0 ∨ r₃.im ≠ 0)) →
  b = -15 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_arithmetic_progression_b_value_l3604_360446


namespace NUMINAMATH_CALUDE_family_admission_price_l3604_360444

/-- Calculates the total admission price for a family visiting an amusement park. -/
theorem family_admission_price 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (h1 : adult_price = 22)
  (h2 : child_price = 7)
  (h3 : num_adults = 2)
  (h4 : num_children = 2) :
  adult_price * num_adults + child_price * num_children = 58 := by
  sorry

#check family_admission_price

end NUMINAMATH_CALUDE_family_admission_price_l3604_360444


namespace NUMINAMATH_CALUDE_min_side_length_is_correct_l3604_360489

/-- The sequence of side lengths of squares to be packed -/
def a (n : ℕ+) : ℚ := 1 / n

/-- The minimum side length of the square that can contain all smaller squares -/
def min_side_length : ℚ := 3 / 2

/-- Theorem stating that min_side_length is the minimum side length of a square
    that can contain all squares with side lengths a(n) without overlapping -/
theorem min_side_length_is_correct :
  ∀ (s : ℚ), (∀ (arrangement : ℕ+ → ℚ × ℚ),
    (∀ (m n : ℕ+), m ≠ n →
      (abs (arrangement m).1 - (arrangement n).1 ≥ min (a m) (a n) ∨
       abs (arrangement m).2 - (arrangement n).2 ≥ min (a m) (a n))) →
    (∀ (n : ℕ+), (arrangement n).1 + a n ≤ s ∧ (arrangement n).2 + a n ≤ s)) →
  s ≥ min_side_length :=
sorry

end NUMINAMATH_CALUDE_min_side_length_is_correct_l3604_360489


namespace NUMINAMATH_CALUDE_unit_prices_min_type_A_boxes_l3604_360448

-- Define the types of gift boxes
inductive GiftBox
| A
| B

-- Define the unit prices as variables
variable (price_A price_B : ℕ)

-- Define the conditions of the problem
axiom first_purchase : 10 * price_A + 15 * price_B = 2800
axiom second_purchase : 6 * price_A + 5 * price_B = 1200

-- Define the total number of boxes and maximum cost
def total_boxes : ℕ := 40
def max_cost : ℕ := 4500

-- Theorem for the unit prices
theorem unit_prices : price_A = 100 ∧ price_B = 120 := by sorry

-- Function to calculate the total cost
def total_cost (num_A : ℕ) : ℕ :=
  num_A * price_A + (total_boxes - num_A) * price_B

-- Theorem for the minimum number of type A boxes
theorem min_type_A_boxes : 
  ∀ num_A : ℕ, num_A ≥ 15 → total_cost num_A ≤ max_cost := by sorry

end NUMINAMATH_CALUDE_unit_prices_min_type_A_boxes_l3604_360448


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3604_360462

open Set
open Real

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 + x + c

-- Define the modified quadratic function
def g (a c x : ℝ) := a * x^2 + 2*x + 4*c

-- Define the linear function
def h (m x : ℝ) := x + m

theorem quadratic_inequality_solution (a c : ℝ) :
  (∀ x, x ∈ solution_set ↔ f a c x > 0) →
  (∀ x, g a c x > 0 → h m x > 0) →
  (∃ x, h m x > 0 ∧ g a c x ≤ 0) →
  (a = -1/4 ∧ c = -3/4) ∧ (∀ m', m' ≥ -2 ↔ m' ≥ m) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3604_360462


namespace NUMINAMATH_CALUDE_carson_gold_stars_l3604_360492

/-- 
Given:
- Carson earned 6 gold stars yesterday
- Carson earned 9 gold stars today

Prove: The total number of gold stars Carson earned is 15
-/
theorem carson_gold_stars (yesterday_stars today_stars : ℕ) 
  (h1 : yesterday_stars = 6) 
  (h2 : today_stars = 9) : 
  yesterday_stars + today_stars = 15 := by
  sorry

end NUMINAMATH_CALUDE_carson_gold_stars_l3604_360492
