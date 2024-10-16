import Mathlib

namespace NUMINAMATH_CALUDE_justin_lost_flowers_l3693_369358

/-- Calculates the number of lost flowers given the gathering time, average time per flower,
    number of classmates, and additional time needed. -/
def lostFlowers (gatheringTime minutes : ℕ) (avgTimePerFlower : ℕ) (classmates : ℕ) (additionalTime : ℕ) : ℕ :=
  let flowersFilled := gatheringTime / avgTimePerFlower
  let additionalFlowers := additionalTime / avgTimePerFlower
  flowersFilled + additionalFlowers - classmates

/-- Theorem stating that Justin has lost 3 flowers. -/
theorem justin_lost_flowers : 
  lostFlowers 120 10 30 210 = 3 := by
  sorry

end NUMINAMATH_CALUDE_justin_lost_flowers_l3693_369358


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3693_369330

theorem complex_fraction_equality (a : ℝ) : (a + Complex.I) / (2 - Complex.I) = 1 + Complex.I → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3693_369330


namespace NUMINAMATH_CALUDE_shared_rest_days_count_l3693_369389

/-- Chris's work cycle in days -/
def chris_cycle : ℕ := 7

/-- Dana's work cycle in days -/
def dana_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1200

/-- Number of rest days Chris has in a cycle -/
def chris_rest_days : ℕ := 2

/-- Number of rest days Dana has in a cycle -/
def dana_rest_days : ℕ := 1

/-- The day in the cycle when both Chris and Dana rest -/
def common_rest_day : ℕ := 7

/-- The number of times Chris and Dana share a rest day in the given period -/
def shared_rest_days : ℕ := total_days / chris_cycle

theorem shared_rest_days_count :
  shared_rest_days = 171 :=
sorry

end NUMINAMATH_CALUDE_shared_rest_days_count_l3693_369389


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3693_369379

theorem arithmetic_expression_equality : 10 - 9^2 + 8 * 7 + 6^2 - 5 * 4 + 3 - 2^3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3693_369379


namespace NUMINAMATH_CALUDE_school_pizza_profit_l3693_369370

theorem school_pizza_profit :
  let num_pizzas : ℕ := 55
  let pizza_cost : ℚ := 685 / 100
  let slices_per_pizza : ℕ := 8
  let slice_price : ℚ := 1
  let total_revenue : ℚ := num_pizzas * slices_per_pizza * slice_price
  let total_cost : ℚ := num_pizzas * pizza_cost
  let profit : ℚ := total_revenue - total_cost
  profit = 6325 / 100 := by
  sorry

end NUMINAMATH_CALUDE_school_pizza_profit_l3693_369370


namespace NUMINAMATH_CALUDE_club_ranking_l3693_369306

def Chess : ℚ := 9/28
def Drama : ℚ := 11/28
def Art : ℚ := 1/7
def Science : ℚ := 5/14

theorem club_ranking :
  Drama > Science ∧ Science > Chess ∧ Chess > Art := by
  sorry

end NUMINAMATH_CALUDE_club_ranking_l3693_369306


namespace NUMINAMATH_CALUDE_lawn_length_is_70_l3693_369327

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  roadWidth : ℝ
  gravelCostPerSquareMeter : ℝ
  totalGravelCost : ℝ

/-- Calculates the total area of the roads -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.length * l.roadWidth + l.width * l.roadWidth - l.roadWidth * l.roadWidth

/-- Theorem stating that given the conditions, the length of the lawn must be 70 m -/
theorem lawn_length_is_70 (l : LawnWithRoads)
    (h1 : l.width = 30)
    (h2 : l.roadWidth = 5)
    (h3 : l.gravelCostPerSquareMeter = 4)
    (h4 : l.totalGravelCost = 1900)
    (h5 : l.totalGravelCost = l.gravelCostPerSquareMeter * roadArea l) :
    l.length = 70 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_is_70_l3693_369327


namespace NUMINAMATH_CALUDE_cans_per_bag_l3693_369377

/-- Given that Paul filled 6 bags on Saturday, 3 bags on Sunday, and collected a total of 72 cans,
    prove that the number of cans in each bag is 8. -/
theorem cans_per_bag (saturday_bags : Nat) (sunday_bags : Nat) (total_cans : Nat) :
  saturday_bags = 6 →
  sunday_bags = 3 →
  total_cans = 72 →
  total_cans / (saturday_bags + sunday_bags) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_bag_l3693_369377


namespace NUMINAMATH_CALUDE_odd_function_inequality_l3693_369392

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_inequality (f : ℝ → ℝ) 
  (h_odd : OddFunction f)
  (h_ineq : ∀ x₁ x₂, x₁ < 0 → x₂ < 0 → x₁ ≠ x₂ → 
    (x₂ * f x₁ - x₁ * f x₂) / (x₁ - x₂) > 0) :
  3 * f (1/3) > -5/2 * f (-2/5) ∧ -5/2 * f (-2/5) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l3693_369392


namespace NUMINAMATH_CALUDE_abs_one_minus_i_equals_sqrt_two_l3693_369308

theorem abs_one_minus_i_equals_sqrt_two :
  Complex.abs (1 - Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_one_minus_i_equals_sqrt_two_l3693_369308


namespace NUMINAMATH_CALUDE_mixture_volume_l3693_369313

/-- Given a mixture of liquids p and q with an initial ratio and a change in ratio after adding more of q, 
    calculate the initial volume of the mixture. -/
theorem mixture_volume (initial_p initial_q added_q : ℝ) 
  (h1 : initial_p / initial_q = 4 / 3) 
  (h2 : initial_p / (initial_q + added_q) = 5 / 7)
  (h3 : added_q = 13) : 
  initial_p + initial_q = 35 := by
  sorry

end NUMINAMATH_CALUDE_mixture_volume_l3693_369313


namespace NUMINAMATH_CALUDE_intersection_area_is_zero_l3693_369336

/-- The first curve: x^2 + y^2 = 16 -/
def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- The second curve: (x-3)^2 + y^2 = 9 -/
def curve2 (x y : ℝ) : Prop := (x-3)^2 + y^2 = 9

/-- The set of intersection points of the two curves -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | curve1 p.1 p.2 ∧ curve2 p.1 p.2}

/-- The polygon formed by the intersection points -/
def intersection_polygon : Set (ℝ × ℝ) :=
  intersection_points

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of the intersection polygon is 0 -/
theorem intersection_area_is_zero :
  area intersection_polygon = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_zero_l3693_369336


namespace NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l3693_369345

/-- Given a right-angled triangle, increasing all sides by the same amount results in an acute-angled triangle -/
theorem right_triangle_increase_sides_acute (a b c k : ℝ) 
  (h_right : a^2 + b^2 = c^2) -- Original triangle is right-angled
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0) -- Sides and increase are positive
  : (a + k)^2 + (b + k)^2 > (c + k)^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l3693_369345


namespace NUMINAMATH_CALUDE_expand_product_l3693_369382

theorem expand_product (x : ℝ) : (x + 4) * (x - 5) * (x + 6) = x^3 + 5*x^2 - 26*x - 120 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3693_369382


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_ten_l3693_369331

theorem sqrt_sum_equals_two_sqrt_ten : 
  Real.sqrt (20 - 8 * Real.sqrt 5) + Real.sqrt (20 + 8 * Real.sqrt 5) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_two_sqrt_ten_l3693_369331


namespace NUMINAMATH_CALUDE_equation_solution_l3693_369305

theorem equation_solution (x : ℝ) : 
  (3 * x + 6 = |(-20 + 2 * x - 3)|) ↔ (x = -29 ∨ x = 17/5) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3693_369305


namespace NUMINAMATH_CALUDE_smallest_eulerian_polyhedron_sum_l3693_369341

/-- A polyhedron is Eulerian if it has an Eulerian path -/
def IsEulerianPolyhedron (V E F : ℕ) : Prop :=
  ∃ (oddDegreeVertices : ℕ), oddDegreeVertices = 2 ∧ 
  V ≥ 4 ∧ E ≥ 6 ∧ F ≥ 4 ∧ V - E + F = 2

/-- The sum of vertices, edges, and faces for a polyhedron -/
def PolyhedronSum (V E F : ℕ) : ℕ := V + E + F

theorem smallest_eulerian_polyhedron_sum :
  ∀ V E F : ℕ, IsEulerianPolyhedron V E F →
  PolyhedronSum V E F ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_eulerian_polyhedron_sum_l3693_369341


namespace NUMINAMATH_CALUDE_arithmetic_equation_l3693_369323

theorem arithmetic_equation : 12.1212 + 17.0005 - 9.1103 = 20.0114 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l3693_369323


namespace NUMINAMATH_CALUDE_group_size_after_new_member_l3693_369335

theorem group_size_after_new_member (n : ℕ) : 
  (n * 14 = n * 14) →  -- Initial average age is 14
  (n * 14 + 32 = (n + 1) * 15) →  -- New average age is 15 after adding a 32-year-old
  n = 17 := by
sorry

end NUMINAMATH_CALUDE_group_size_after_new_member_l3693_369335


namespace NUMINAMATH_CALUDE_range_of_sum_l3693_369360

theorem range_of_sum (a b : ℝ) :
  (∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) ≥ -1) →
  -1 ≤ a + b ∧ a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l3693_369360


namespace NUMINAMATH_CALUDE_continuity_point_sum_l3693_369374

theorem continuity_point_sum (g : ℝ → ℝ) : 
  (∃ m₁ m₂ : ℝ, 
    (∀ x < m₁, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₁, g x = 3*x + 6) ∧
    (∀ x < m₂, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₂, g x = 3*x + 6) ∧
    (m₁^2 + 4 = 3*m₁ + 6) ∧
    (m₂^2 + 4 = 3*m₂ + 6) ∧
    (m₁ ≠ m₂)) →
  (∃ m₁ m₂ : ℝ, m₁ + m₂ = 3 ∧ 
    (∀ x < m₁, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₁, g x = 3*x + 6) ∧
    (∀ x < m₂, g x = x^2 + 4) ∧ 
    (∀ x ≥ m₂, g x = 3*x + 6) ∧
    (m₁^2 + 4 = 3*m₁ + 6) ∧
    (m₂^2 + 4 = 3*m₂ + 6) ∧
    (m₁ ≠ m₂)) :=
by sorry

end NUMINAMATH_CALUDE_continuity_point_sum_l3693_369374


namespace NUMINAMATH_CALUDE_log_expression_equality_l3693_369329

theorem log_expression_equality : 2 * Real.log 2 - Real.log (1 / 25) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l3693_369329


namespace NUMINAMATH_CALUDE_unique_magnitude_for_complex_roots_l3693_369372

theorem unique_magnitude_for_complex_roots (x : ℂ) : 
  x^2 - 4*x + 29 = 0 → ∃! m : ℝ, ∃ y : ℂ, y^2 - 4*y + 29 = 0 ∧ Complex.abs y = m :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_for_complex_roots_l3693_369372


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l3693_369317

theorem cubic_roots_problem (a b : ℝ) (r s : ℝ) : 
  (r^3 + a*r + b = 0) →
  (s^3 + a*s + b = 0) →
  ((r+3)^3 + a*(r+3) + b + 360 = 0) →
  ((s-2)^3 + a*(s-2) + b + 360 = 0) →
  (b = -1330/27 ∨ b = -6340/27) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l3693_369317


namespace NUMINAMATH_CALUDE_rotation_implies_equilateral_l3693_369381

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Rotation of a point around another point by a given angle -/
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Theorem: If rotating a triangle 60° around point A moves B to C, then the triangle is equilateral -/
theorem rotation_implies_equilateral (t : Triangle) :
  rotate t.A (π / 3) t.B = t.C → is_equilateral t := by sorry

end NUMINAMATH_CALUDE_rotation_implies_equilateral_l3693_369381


namespace NUMINAMATH_CALUDE_circles_intersect_l3693_369387

theorem circles_intersect : ∃ (x y : ℝ), 
  ((x + 1)^2 + (y + 2)^2 = 4) ∧ ((x - 1)^2 + (y + 1)^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3693_369387


namespace NUMINAMATH_CALUDE_cos_90_degrees_equals_zero_l3693_369314

theorem cos_90_degrees_equals_zero : 
  let cos_def : ℝ → ℝ := λ θ => (Real.cos θ)
  let unit_circle_point : ℝ × ℝ := (0, 1)
  cos_def (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_equals_zero_l3693_369314


namespace NUMINAMATH_CALUDE_tenth_term_is_44_l3693_369326

/-- Arithmetic sequence with first term 8 and common difference 4 -/
def arithmetic_sequence (n : ℕ) : ℕ := 8 + 4 * (n - 1)

/-- The 10th term of the arithmetic sequence is 44 -/
theorem tenth_term_is_44 : arithmetic_sequence 10 = 44 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_44_l3693_369326


namespace NUMINAMATH_CALUDE_equation_simplification_l3693_369309

theorem equation_simplification (x : ℝ) :
  x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2 ↔ 10 * x / 3 = 1 + (12 - 3 * x) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_simplification_l3693_369309


namespace NUMINAMATH_CALUDE_restaurant_bill_total_l3693_369395

theorem restaurant_bill_total (num_people : ℕ) (individual_payment : ℚ) (total_bill : ℚ) : 
  num_people = 9 → 
  individual_payment = 514.19 → 
  total_bill = num_people * individual_payment → 
  total_bill = 4627.71 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_total_l3693_369395


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3693_369312

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = -3
  condition : 11 * (a 5) = 5 * (a 8) - 13
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- The common difference of the arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℚ := seq.a 2 - seq.a 1

/-- The sum of the first n terms of the arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

/-- The main theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (common_difference seq = 5/9) ∧
  (∃ n : ℕ, sum_n_terms seq n = -29/3 ∧ 
    ∀ m : ℕ, sum_n_terms seq m ≥ sum_n_terms seq n) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3693_369312


namespace NUMINAMATH_CALUDE_xyz_equation_l3693_369390

theorem xyz_equation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1)
  (h2 : x + 1 / z = 5)
  (h3 : y + 1 / x = 29) :
  z + 1 / y = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_xyz_equation_l3693_369390


namespace NUMINAMATH_CALUDE_log_3_bounds_l3693_369349

theorem log_3_bounds :
  2/5 < Real.log 3 / Real.log 10 ∧ Real.log 3 / Real.log 10 < 1/2 := by
  have h1 : (3 : ℝ)^5 = 243 := by norm_num
  have h2 : (3 : ℝ)^6 = 729 := by norm_num
  have h3 : (2 : ℝ)^8 = 256 := by norm_num
  have h4 : (2 : ℝ)^10 = 1024 := by norm_num
  have h5 : (10 : ℝ)^2 = 100 := by norm_num
  have h6 : (10 : ℝ)^3 = 1000 := by norm_num
  sorry

end NUMINAMATH_CALUDE_log_3_bounds_l3693_369349


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3693_369364

theorem quadratic_roots_relation (r s : ℝ) (p q : ℝ) : 
  (3 * r^2 + 4 * r + 2 = 0) →
  (3 * s^2 + 4 * s + 2 = 0) →
  ((1/r^2) + (1/s^2) = -p) →
  ((1/r^2) * (1/s^2) = q) →
  p = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3693_369364


namespace NUMINAMATH_CALUDE_unique_divisible_by_396_l3693_369384

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = x * 100000 + y * 10000 + 2 * 1000 + 4 * 100 + 3 * 10 + z

theorem unique_divisible_by_396 :
  ∃! n : ℕ, is_valid_number n ∧ n % 396 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_396_l3693_369384


namespace NUMINAMATH_CALUDE_line_l_equation_l3693_369376

-- Define points A and B
def A : ℝ × ℝ := (3, 3)
def B : ℝ × ℝ := (5, 2)

-- Define lines l1 and l2
def l1 (x y : ℝ) : Prop := 3 * x - y - 1 = 0
def l2 (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the intersection point of l1 and l2
def intersection : ℝ × ℝ := (1, 2)

-- Define the property that l passes through the intersection
def passes_through_intersection (l : ℝ → ℝ → Prop) : Prop :=
  l (intersection.1) (intersection.2)

-- Define the property of equal distance from A and B to l
def equal_distance (l : ℝ → ℝ → Prop) : Prop :=
  ∃ d : ℝ, d > 0 ∧
    (∃ x y : ℝ, l x y ∧ (x - A.1)^2 + (y - A.2)^2 = d^2) ∧
    (∃ x y : ℝ, l x y ∧ (x - B.1)^2 + (y - B.2)^2 = d^2)

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := x - 6 * y + 11 = 0 ∨ x + 2 * y - 5 = 0

-- Theorem statement
theorem line_l_equation :
  ∀ l : ℝ → ℝ → Prop,
    passes_through_intersection l →
    equal_distance l →
    (∀ x y : ℝ, l x y ↔ line_l x y) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_l3693_369376


namespace NUMINAMATH_CALUDE_prime_square_plus_one_triples_l3693_369352

theorem prime_square_plus_one_triples :
  ∀ a b c : ℕ,
    Prime (a^2 + 1) →
    Prime (b^2 + 1) →
    (a^2 + 1) * (b^2 + 1) = c^2 + 1 →
    ((a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_plus_one_triples_l3693_369352


namespace NUMINAMATH_CALUDE_regression_y_intercept_l3693_369334

/-- Empirical regression equation for height prediction -/
def height_prediction (x : ℝ) (a : ℝ) : ℝ := 3 * x + a

/-- Average height of the 50 classmates -/
def average_height : ℝ := 170

/-- Average shoe size of the 50 classmates -/
def average_shoe_size : ℝ := 40

/-- Theorem stating that the y-intercept (a) of the regression line is 50 -/
theorem regression_y_intercept :
  ∃ (a : ℝ), height_prediction average_shoe_size a = average_height ∧ a = 50 := by
  sorry

end NUMINAMATH_CALUDE_regression_y_intercept_l3693_369334


namespace NUMINAMATH_CALUDE_no_real_solution_nonzero_z_l3693_369359

theorem no_real_solution_nonzero_z (x y z : ℝ) : 
  x - y = 2 → xy + z^2 + 1 = 0 → z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_nonzero_z_l3693_369359


namespace NUMINAMATH_CALUDE_stratified_sample_grad_count_l3693_369363

/-- Represents the number of students to be sampled from each stratum in a stratified sampling -/
structure StratifiedSample where
  total : ℕ
  junior : ℕ
  undergrad : ℕ
  grad : ℕ

/-- Calculates the stratified sample given the total population and sample size -/
def calculateStratifiedSample (totalPopulation : ℕ) (juniorCount : ℕ) (undergradCount : ℕ) (sampleSize : ℕ) : StratifiedSample :=
  let gradCount := totalPopulation - juniorCount - undergradCount
  let sampleRatio := sampleSize / totalPopulation
  { total := sampleSize,
    junior := juniorCount * sampleRatio,
    undergrad := undergradCount * sampleRatio,
    grad := sampleSize - (juniorCount * sampleRatio) - (undergradCount * sampleRatio) }

theorem stratified_sample_grad_count 
  (totalPopulation : ℕ) (juniorCount : ℕ) (undergradCount : ℕ) (sampleSize : ℕ)
  (h1 : totalPopulation = 5600)
  (h2 : juniorCount = 1300)
  (h3 : undergradCount = 3000)
  (h4 : sampleSize = 280) :
  (calculateStratifiedSample totalPopulation juniorCount undergradCount sampleSize).grad = 65 := by
  sorry

#eval (calculateStratifiedSample 5600 1300 3000 280).grad

end NUMINAMATH_CALUDE_stratified_sample_grad_count_l3693_369363


namespace NUMINAMATH_CALUDE_largest_prime_with_special_property_l3693_369342

theorem largest_prime_with_special_property : 
  (∃ (p : ℕ), Prime p ∧ 
    (∃ (a b : ℕ), 
      a > 0 ∧ b > 0 ∧ 
      (p : ℚ) = (b / 2 : ℚ) * Real.sqrt ((a - b : ℚ) / (a + b : ℚ))) ∧
    (∀ (q : ℕ), Prime q → 
      (∃ (a b : ℕ), 
        a > 0 ∧ b > 0 ∧ 
        (q : ℚ) = (b / 2 : ℚ) * Real.sqrt ((a - b : ℚ) / (a + b : ℚ))) → 
      q ≤ p)) ∧
  (∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧ 
    (5 : ℚ) = (b / 2 : ℚ) * Real.sqrt ((a - b : ℚ) / (a + b : ℚ))) := by
  sorry


end NUMINAMATH_CALUDE_largest_prime_with_special_property_l3693_369342


namespace NUMINAMATH_CALUDE_triangle_existence_l3693_369356

theorem triangle_existence (y : ℕ+) : 
  (y + 1 + 6 > y^2 + 2*y + 3) ∧ 
  (y + 1 + (y^2 + 2*y + 3) > 6) ∧ 
  (6 + (y^2 + 2*y + 3) > y + 1) ↔ 
  y = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_existence_l3693_369356


namespace NUMINAMATH_CALUDE_sequence_properties_l3693_369354

theorem sequence_properties (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 2*n) :
  (a 2 = 5 ∧ a 3 = 11 ∧ a 4 = 19) ∧
  (∀ n : ℕ, a n = n^2 + n - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l3693_369354


namespace NUMINAMATH_CALUDE_blue_cube_problem_l3693_369399

theorem blue_cube_problem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_cube_problem_l3693_369399


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3693_369362

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 - 20 * x - 8 * y + 72 = 0

/-- The circle is inscribed in a rectangle -/
def is_inscribed (circle : (ℝ → ℝ → Prop)) (rectangle : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), circle x y → (x, y) ∈ rectangle

/-- One pair of sides of the rectangle is parallel to the y-axis -/
def sides_parallel_to_y_axis (rectangle : Set (ℝ × ℝ)) : Prop :=
  ∃ (x₁ x₂ : ℝ), ∀ (y : ℝ), (x₁, y) ∈ rectangle ∨ (x₂, y) ∈ rectangle

/-- The area of the rectangle -/
def rectangle_area (rectangle : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem -/
theorem inscribed_circle_rectangle_area :
  ∀ (rectangle : Set (ℝ × ℝ)),
  is_inscribed circle_equation rectangle →
  sides_parallel_to_y_axis rectangle →
  rectangle_area rectangle = 28 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l3693_369362


namespace NUMINAMATH_CALUDE_zhang_bin_is_journalist_l3693_369332

structure Person where
  name : String
  isJournalist : Bool
  statement : Bool

def liZhiming : Person := { name := "Li Zhiming", isJournalist := false, statement := true }
def zhangBin : Person := { name := "Zhang Bin", isJournalist := false, statement := true }
def wangDawei : Person := { name := "Wang Dawei", isJournalist := false, statement := true }

theorem zhang_bin_is_journalist :
  ∀ (li : Person) (zhang : Person) (wang : Person),
    li.name = "Li Zhiming" →
    zhang.name = "Zhang Bin" →
    wang.name = "Wang Dawei" →
    (li.isJournalist ∨ zhang.isJournalist ∨ wang.isJournalist) →
    (li.isJournalist → ¬zhang.isJournalist ∧ ¬wang.isJournalist) →
    (zhang.isJournalist → ¬li.isJournalist ∧ ¬wang.isJournalist) →
    (wang.isJournalist → ¬li.isJournalist ∧ ¬zhang.isJournalist) →
    li.statement = li.isJournalist →
    zhang.statement = ¬zhang.isJournalist →
    wang.statement = ¬li.statement →
    (li.statement ∨ zhang.statement ∨ wang.statement) →
    (li.statement → ¬zhang.statement ∧ ¬wang.statement) →
    (zhang.statement → ¬li.statement ∧ ¬wang.statement) →
    (wang.statement → ¬li.statement ∧ ¬zhang.statement) →
    zhang.isJournalist := by
  sorry

#check zhang_bin_is_journalist

end NUMINAMATH_CALUDE_zhang_bin_is_journalist_l3693_369332


namespace NUMINAMATH_CALUDE_f_always_positive_sum_reciprocals_geq_nine_l3693_369319

-- Problem 1
def f (x : ℝ) : ℝ := x^6 - x^3 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, f x > 0 := by
  sorry

-- Problem 2
theorem sum_reciprocals_geq_nine {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c = 1) : 1/a + 1/b + 1/c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_sum_reciprocals_geq_nine_l3693_369319


namespace NUMINAMATH_CALUDE_aria_cookie_spending_l3693_369365

/-- The number of days in March -/
def days_in_march : ℕ := 31

/-- The number of cookies Aria purchased each day -/
def cookies_per_day : ℕ := 4

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 19

/-- The total amount Aria spent on cookies in March -/
def total_spent : ℕ := days_in_march * cookies_per_day * cost_per_cookie

theorem aria_cookie_spending :
  total_spent = 2356 := by sorry

end NUMINAMATH_CALUDE_aria_cookie_spending_l3693_369365


namespace NUMINAMATH_CALUDE_icosagon_diagonals_from_vertex_l3693_369340

/-- The number of sides in an icosagon -/
def icosagon_sides : ℕ := 20

/-- The number of diagonals from a single vertex in an icosagon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

theorem icosagon_diagonals_from_vertex :
  diagonals_from_vertex icosagon_sides = 17 := by sorry

end NUMINAMATH_CALUDE_icosagon_diagonals_from_vertex_l3693_369340


namespace NUMINAMATH_CALUDE_highlighter_count_l3693_369328

/-- The number of highlighters in Kaya's teacher's desk -/
theorem highlighter_count : 
  let pink : ℕ := 12
  let yellow : ℕ := 15
  let blue : ℕ := 8
  let green : ℕ := 6
  let orange : ℕ := 4
  pink + yellow + blue + green + orange = 45 := by
  sorry

end NUMINAMATH_CALUDE_highlighter_count_l3693_369328


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3693_369302

theorem cubic_root_sum (a b c : ℝ) (p q r : ℕ+) : 
  a^3 - 3*a^2 - 7*a - 1 = 0 →
  b^3 - 3*b^2 - 7*b - 1 = 0 →
  c^3 - 3*c^2 - 7*c - 1 = 0 →
  a ≠ b →
  b ≠ c →
  c ≠ a →
  (1 / (a^(1/3) - b^(1/3)) + 1 / (b^(1/3) - c^(1/3)) + 1 / (c^(1/3) - a^(1/3)))^2 = p * q^(1/3) / r →
  Nat.gcd p.val r.val = 1 →
  ∀ (prime : ℕ), prime.Prime → ¬(∃ (k : ℕ), q = prime^3 * k) →
  100 * p + 10 * q + r = 1913 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3693_369302


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3693_369321

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 4 * x + 3 ∧
  (∀ (y : ℝ), y * |y| = 4 * y + 3 → x ≤ y) ∧
  x = -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3693_369321


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3693_369324

theorem max_sum_of_factors (p q : ℕ+) (h : p * q = 100) : 
  ∃ (a b : ℕ+), a * b = 100 ∧ a + b ≤ p + q ∧ a + b = 101 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3693_369324


namespace NUMINAMATH_CALUDE_only_one_claim_impossible_l3693_369355

-- Define the possible ring scores
def RingScores : List Nat := [1, 3, 5, 7, 9]

-- Define a structure for each person's claim
structure Claim where
  shots : Nat
  hits : Nat
  total_score : Nat

-- Define the claims
def claim_A : Claim := { shots := 5, hits := 5, total_score := 35 }
def claim_B : Claim := { shots := 6, hits := 6, total_score := 36 }
def claim_C : Claim := { shots := 3, hits := 3, total_score := 24 }
def claim_D : Claim := { shots := 4, hits := 3, total_score := 21 }

-- Function to check if a claim is possible
def is_claim_possible (c : Claim) : Prop :=
  ∃ (scores : List Nat),
    scores.length = c.hits ∧
    scores.all (· ∈ RingScores) ∧
    scores.sum = c.total_score

-- Theorem stating that only one claim is impossible
theorem only_one_claim_impossible :
  is_claim_possible claim_A ∧
  is_claim_possible claim_B ∧
  ¬is_claim_possible claim_C ∧
  is_claim_possible claim_D :=
sorry

end NUMINAMATH_CALUDE_only_one_claim_impossible_l3693_369355


namespace NUMINAMATH_CALUDE_game_terminates_l3693_369301

/-- Represents the state of knowledge for a player -/
structure PlayerKnowledge where
  lower : ℕ
  upper : ℕ

/-- Represents the game state -/
structure GameState where
  r₁ : ℕ
  r₂ : ℕ
  a_knowledge : PlayerKnowledge
  b_knowledge : PlayerKnowledge

/-- Updates a player's knowledge based on the game state -/
def update_knowledge (state : GameState) (is_player_a : Bool) : PlayerKnowledge :=
  sorry

/-- Checks if a player can determine the other's number -/
def can_determine (knowledge : PlayerKnowledge) : Bool :=
  sorry

/-- The main theorem stating that the game terminates -/
theorem game_terminates (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (n : ℕ), ∃ (final_state : GameState),
    (final_state.r₁ = a + b ∨ final_state.r₂ = a + b) ∧
    (can_determine final_state.a_knowledge ∨ can_determine final_state.b_knowledge) :=
  sorry

end NUMINAMATH_CALUDE_game_terminates_l3693_369301


namespace NUMINAMATH_CALUDE_triangle_cosine_C_l3693_369394

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : 0 < B ∧ B < π / 2)
  (h3 : 0 < C ∧ C < π / 2)
  (h4 : A + B + C = π)

-- State the theorem
theorem triangle_cosine_C (t : Triangle) 
  (h5 : f t.A = 3 / 2)
  (h6 : ∃ (D : ℝ), D = Real.sqrt 2 ∧ D * Real.sin (t.B / 2) = t.c * Real.sin (t.A / 2))
  (h7 : ∃ (D : ℝ), D = 2 ∧ D * Real.sin (t.A / 2) = t.b * Real.sin (t.C / 2)) :
  Real.cos t.C = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_cosine_C_l3693_369394


namespace NUMINAMATH_CALUDE_fraction_problem_l3693_369348

theorem fraction_problem (x : ℚ) : 
  (3/4 : ℚ) * x * (2/5 : ℚ) * 5100 = 765.0000000000001 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3693_369348


namespace NUMINAMATH_CALUDE_circle_area_around_equilateral_triangle_l3693_369337

theorem circle_area_around_equilateral_triangle :
  let side_length : ℝ := 12
  let circumradius : ℝ := side_length / Real.sqrt 3
  let circle_area : ℝ := Real.pi * circumradius^2
  circle_area = 48 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_around_equilateral_triangle_l3693_369337


namespace NUMINAMATH_CALUDE_prime_pair_existence_l3693_369393

theorem prime_pair_existence (n : ℕ) : 
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p + 2 = q ∧ Prime (2^n + p) ∧ Prime (2^n + q)) ↔ n = 1 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_pair_existence_l3693_369393


namespace NUMINAMATH_CALUDE_planter_cost_theorem_l3693_369380

/-- Represents the cost and quantity of a type of plant in a planter --/
structure PlantInfo where
  quantity : ℕ
  price : ℚ

/-- Calculates the total cost for a rectangle-shaped pool's corner planters --/
def total_cost (palm_fern : PlantInfo) (creeping_jenny : PlantInfo) (geranium : PlantInfo) : ℚ :=
  let cost_per_pot := palm_fern.quantity * palm_fern.price + 
                      creeping_jenny.quantity * creeping_jenny.price + 
                      geranium.quantity * geranium.price
  4 * cost_per_pot

/-- Theorem stating the total cost for the planters --/
theorem planter_cost_theorem (palm_fern : PlantInfo) (creeping_jenny : PlantInfo) (geranium : PlantInfo)
  (h1 : palm_fern.quantity = 1)
  (h2 : palm_fern.price = 15)
  (h3 : creeping_jenny.quantity = 4)
  (h4 : creeping_jenny.price = 4)
  (h5 : geranium.quantity = 4)
  (h6 : geranium.price = 7/2) :
  total_cost palm_fern creeping_jenny geranium = 180 := by
  sorry

end NUMINAMATH_CALUDE_planter_cost_theorem_l3693_369380


namespace NUMINAMATH_CALUDE_max_value_ln_minus_x_l3693_369351

open Real

theorem max_value_ln_minus_x :
  ∃ (x : ℝ), 0 < x ∧ x ≤ exp 1 ∧
  (∀ (y : ℝ), 0 < y ∧ y ≤ exp 1 → log y - y ≤ log x - x) ∧
  log x - x = -1 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_ln_minus_x_l3693_369351


namespace NUMINAMATH_CALUDE_marks_lost_is_one_l3693_369375

/-- Represents an examination with given parameters -/
structure Examination where
  totalQuestions : Nat
  correctAnswers : Nat
  marksPerCorrect : Nat
  totalScore : Int

/-- Calculates the marks lost per wrong answer -/
def marksLostPerWrongAnswer (exam : Examination) : Rat :=
  let wrongAnswers := exam.totalQuestions - exam.correctAnswers
  let totalCorrectMarks := exam.correctAnswers * exam.marksPerCorrect
  let totalLostMarks := totalCorrectMarks - exam.totalScore
  totalLostMarks / wrongAnswers

/-- Theorem stating that for the given examination parameters, 
    the marks lost per wrong answer is 1 -/
theorem marks_lost_is_one : 
  let exam : Examination := {
    totalQuestions := 80,
    correctAnswers := 42,
    marksPerCorrect := 4,
    totalScore := 130
  }
  marksLostPerWrongAnswer exam = 1 := by
  sorry

end NUMINAMATH_CALUDE_marks_lost_is_one_l3693_369375


namespace NUMINAMATH_CALUDE_man_walking_time_l3693_369338

/-- The man's usual time to cover the distance -/
def usual_time : ℝ := 72

/-- The man's usual speed -/
def usual_speed : ℝ := 1

/-- The factor by which the man's speed is reduced -/
def speed_reduction_factor : ℝ := 0.75

/-- The additional time taken when walking at reduced speed -/
def additional_time : ℝ := 24

theorem man_walking_time :
  (usual_speed * usual_time = speed_reduction_factor * usual_speed * (usual_time + additional_time)) →
  usual_time = 72 := by
  sorry

end NUMINAMATH_CALUDE_man_walking_time_l3693_369338


namespace NUMINAMATH_CALUDE_linear_decreasing_slope_l3693_369366

/-- A function that represents a linear equation with slope (m-3) and y-intercept 4 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x + 4

/-- The property that the function decreases as x increases -/
def decreasing (m : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → f m x₁ > f m x₂

theorem linear_decreasing_slope (m : ℝ) : decreasing m → m < 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_decreasing_slope_l3693_369366


namespace NUMINAMATH_CALUDE_square_area_rational_l3693_369300

theorem square_area_rational (s : ℚ) : ∃ (a : ℚ), a = s^2 :=
  sorry

end NUMINAMATH_CALUDE_square_area_rational_l3693_369300


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l3693_369318

/-- The number of players in the team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def num_quadruplets : ℕ := 4

/-- The number of starters to be chosen -/
def num_starters : ℕ := 5

/-- The number of quadruplets that must be in the starting lineup -/
def quadruplets_in_lineup : ℕ := 3

/-- The number of ways to choose the starting lineup -/
def ways_to_choose_lineup : ℕ := Nat.choose num_quadruplets quadruplets_in_lineup * 
  Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)

theorem starting_lineup_combinations : ways_to_choose_lineup = 264 := by sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l3693_369318


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3693_369397

theorem polynomial_factorization (y : ℝ) : 3 * y^2 - 27 = 3 * (y + 3) * (y - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3693_369397


namespace NUMINAMATH_CALUDE_pentagon_properties_independent_l3693_369320

/-- A pentagon is a polygon with 5 sides --/
structure Pentagon where
  sides : Fin 5 → ℝ
  angles : Fin 5 → ℝ

/-- A pentagon is equilateral if all its sides have the same length --/
def Pentagon.isEquilateral (p : Pentagon) : Prop :=
  ∀ i j : Fin 5, p.sides i = p.sides j

/-- A pentagon is equiangular if all its angles are equal --/
def Pentagon.isEquiangular (p : Pentagon) : Prop :=
  ∀ i j : Fin 5, p.angles i = p.angles j

/-- The properties of equal angles and equal sides in a pentagon are independent --/
theorem pentagon_properties_independent :
  (∃ p : Pentagon, p.isEquiangular ∧ ¬p.isEquilateral) ∧
  (∃ q : Pentagon, q.isEquilateral ∧ ¬q.isEquiangular) := by
  sorry

end NUMINAMATH_CALUDE_pentagon_properties_independent_l3693_369320


namespace NUMINAMATH_CALUDE_f_neg_six_value_l3693_369315

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_periodic : ∀ x, f (x + 6) = f x
axiom f_defined : ∀ x, -3 ≤ x → x ≤ 3 → f x = (x + 1) * (x - (1/2 : ℝ))

-- Theorem to prove
theorem f_neg_six_value : f (-6) = -1/2 := by sorry

end NUMINAMATH_CALUDE_f_neg_six_value_l3693_369315


namespace NUMINAMATH_CALUDE_range_of_a_l3693_369357

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3693_369357


namespace NUMINAMATH_CALUDE_middle_letter_value_is_eight_l3693_369398

/-- Represents a three-letter word in Scrabble -/
structure ScrabbleWord where
  first_letter_value : ℕ
  middle_letter_value : ℕ
  last_letter_value : ℕ

/-- Calculates the total value of a ScrabbleWord before tripling -/
def word_value (word : ScrabbleWord) : ℕ :=
  word.first_letter_value + word.middle_letter_value + word.last_letter_value

/-- Theorem: Given the conditions, the middle letter's value is 8 -/
theorem middle_letter_value_is_eight 
  (word : ScrabbleWord)
  (h1 : word.first_letter_value = 1)
  (h2 : word.last_letter_value = 1)
  (h3 : 3 * (word_value word) = 30) :
  word.middle_letter_value = 8 := by
  sorry


end NUMINAMATH_CALUDE_middle_letter_value_is_eight_l3693_369398


namespace NUMINAMATH_CALUDE_convergence_condition_l3693_369311

/-- The iteration function for calculating 1/a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * (2 - a * x)

/-- The sequence generated by the iteration -/
def iterSeq (a : ℝ) (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f a (iterSeq a x₀ n)

theorem convergence_condition (a : ℝ) (x₀ : ℝ) (h : a > 0) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |iterSeq a x₀ n - 1/a| < ε) ↔ (0 < x₀ ∧ x₀ < 2/a) :=
sorry

end NUMINAMATH_CALUDE_convergence_condition_l3693_369311


namespace NUMINAMATH_CALUDE_scribbled_digits_sum_l3693_369385

theorem scribbled_digits_sum (x y : ℕ) (a b c : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 →
  ∃ k : ℕ, x * y = k * 100000 + a * 10000 + 3 * 1000 + b * 100 + 1 * 10 + 2 →
  a < 10 ∧ b < 10 ∧ c < 10 →
  a + b + c = 6 := by
sorry

end NUMINAMATH_CALUDE_scribbled_digits_sum_l3693_369385


namespace NUMINAMATH_CALUDE_office_age_problem_l3693_369303

theorem office_age_problem (total_persons : Nat) (total_avg : ℚ) (group1_persons : Nat) 
  (group1_avg : ℚ) (person15_age : Nat) (group2_persons : Nat) :
  total_persons = 20 →
  total_avg = 15 →
  group1_persons = 5 →
  group1_avg = 14 →
  person15_age = 86 →
  group2_persons = 9 →
  let total_age : ℚ := total_persons * total_avg
  let group1_age : ℚ := group1_persons * group1_avg
  let remaining_age : ℚ := total_age - group1_age - person15_age
  let group2_age : ℚ := remaining_age - (total_persons - group1_persons - group2_persons - 1) * total_avg
  let group2_avg : ℚ := group2_age / group2_persons
  group2_avg = 23/3 := by sorry

end NUMINAMATH_CALUDE_office_age_problem_l3693_369303


namespace NUMINAMATH_CALUDE_flowers_in_vase_l3693_369346

theorem flowers_in_vase (initial_flowers : ℕ) (removed_flowers : ℕ) : 
  initial_flowers = 13 → removed_flowers = 7 → initial_flowers - removed_flowers = 6 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_vase_l3693_369346


namespace NUMINAMATH_CALUDE_bakery_storage_ratio_l3693_369310

/-- Given the conditions of a bakery's storage room, prove that the ratio of sugar to flour is 1 to 1. -/
theorem bakery_storage_ratio : ∀ (sugar flour baking_soda : ℝ),
  sugar = 2400 →
  flour = 10 * baking_soda →
  flour = 8 * (baking_soda + 60) →
  sugar / flour = 1 := by
  sorry

end NUMINAMATH_CALUDE_bakery_storage_ratio_l3693_369310


namespace NUMINAMATH_CALUDE_quiz_probabilities_l3693_369383

/-- Represents a quiz with multiple-choice and true/false questions -/
structure Quiz where
  total_questions : ℕ
  multiple_choice : ℕ
  true_false : ℕ
  h_total : total_questions = multiple_choice + true_false

/-- Calculates the probability of an event in a quiz draw -/
def probability (q : Quiz) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / (q.total_questions * (q.total_questions - 1))

theorem quiz_probabilities (q : Quiz) 
    (h_total : q.total_questions = 5)
    (h_mc : q.multiple_choice = 3)
    (h_tf : q.true_false = 2) :
  let p1 := probability q (q.true_false * q.multiple_choice)
  let p2 := 1 - probability q (q.true_false * (q.true_false - 1))
  p1 = 3/10 ∧ p2 = 9/10 := by
  sorry


end NUMINAMATH_CALUDE_quiz_probabilities_l3693_369383


namespace NUMINAMATH_CALUDE_quadratic_equation_with_root_one_l3693_369396

theorem quadratic_equation_with_root_one (a : ℝ) (h : a ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 - a) ∧ f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_root_one_l3693_369396


namespace NUMINAMATH_CALUDE_plants_eaten_first_day_l3693_369304

theorem plants_eaten_first_day (total : ℕ) (remaining : ℕ) :
  total = 30 ∧ 
  remaining = 4 ∧
  (∃ x y : ℕ, x + y + remaining + 1 = total ∧ y = (x + y + 1) / 2) →
  x = 20 :=
by sorry

end NUMINAMATH_CALUDE_plants_eaten_first_day_l3693_369304


namespace NUMINAMATH_CALUDE_dividend_calculation_l3693_369316

theorem dividend_calculation (quotient divisor remainder : ℝ) 
  (hq : quotient = -427.86)
  (hd : divisor = 52.7)
  (hr : remainder = -14.5) :
  (quotient * divisor) + remainder = -22571.122 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3693_369316


namespace NUMINAMATH_CALUDE_exponent_equation_l3693_369368

theorem exponent_equation (x s : ℕ) (h : (2^x) * (25^s) = 5 * (10^16)) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_l3693_369368


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3693_369373

def largest_perfect_cube_divisor (n : ℕ) : ℕ := sorry

def cube_root (n : ℕ) : ℕ := sorry

def sum_of_prime_exponents (n : ℕ) : ℕ := sorry

theorem sum_of_exponents_15_factorial : 
  sum_of_prime_exponents (cube_root (largest_perfect_cube_divisor (Nat.factorial 15))) = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3693_369373


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3693_369322

def arithmeticSequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem fifth_term_of_arithmetic_sequence 
  (a : ℤ) (d : ℤ) 
  (h12 : arithmeticSequence a d 12 = 25)
  (h13 : arithmeticSequence a d 13 = 29) :
  arithmeticSequence a d 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3693_369322


namespace NUMINAMATH_CALUDE_soccer_team_selection_l3693_369371

theorem soccer_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quad_starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 7 →
  quad_starters = 2 →
  (Nat.choose quadruplets quad_starters) * (Nat.choose (total_players - quadruplets) (starters - quad_starters)) = 4752 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l3693_369371


namespace NUMINAMATH_CALUDE_sector_max_area_l3693_369386

/-- Given a sector with circumference 20cm, its area is maximized when the radius is 5cm -/
theorem sector_max_area (R : ℝ) : 
  let circumference := 20
  let arc_length := circumference - 2 * R
  let area := (1 / 2) * arc_length * R
  (∀ r : ℝ, area ≤ ((1 / 2) * (circumference - 2 * r) * r)) → R = 5 := by
sorry

end NUMINAMATH_CALUDE_sector_max_area_l3693_369386


namespace NUMINAMATH_CALUDE_indeterminate_remainder_l3693_369307

theorem indeterminate_remainder (a b c d m n x y : ℤ) 
  (eq1 : a * x + b * y = m)
  (eq2 : c * x + d * y = n)
  (rem64 : ∃ k : ℤ, a * x + b * y = 64 * k + 37) :
  ∀ r : ℤ, ¬ (∀ k : ℤ, c * x + d * y = 5 * k + r ∧ 0 ≤ r ∧ r < 5) :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_remainder_l3693_369307


namespace NUMINAMATH_CALUDE_sphere_radius_regular_tetrahedron_l3693_369369

/-- The radius of a sphere touching all edges of a regular tetrahedron with edge length √2 --/
theorem sphere_radius_regular_tetrahedron : 
  ∀ (tetrahedron_edge : ℝ) (sphere_radius : ℝ),
  tetrahedron_edge = Real.sqrt 2 →
  sphere_radius = 
    (1 / 2) * ((tetrahedron_edge * Real.sqrt 6) / 3) →
  sphere_radius = 1 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_regular_tetrahedron_l3693_369369


namespace NUMINAMATH_CALUDE_julia_spent_114_l3693_369388

/-- The total amount Julia spent on food for her animals -/
def total_spent (weekly_total : ℕ) (rabbit_weeks : ℕ) (parrot_weeks : ℕ) (rabbit_food_cost : ℕ) : ℕ :=
  let parrot_food_cost := weekly_total - rabbit_food_cost
  rabbit_weeks * rabbit_food_cost + parrot_weeks * parrot_food_cost

/-- Proof that Julia spent $114 on food for her animals -/
theorem julia_spent_114 :
  total_spent 30 5 3 12 = 114 := by
  sorry

end NUMINAMATH_CALUDE_julia_spent_114_l3693_369388


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3693_369343

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ (b : ℝ), (Complex.I : ℂ) * b = (a + Complex.I) / (1 + Complex.I)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3693_369343


namespace NUMINAMATH_CALUDE_right_triangle_trig_values_l3693_369333

-- Define the right triangle XYZ
structure RightTriangle where
  XY : ℝ
  XZ : ℝ
  YZ : ℝ
  right_angle : XY^2 + XZ^2 = YZ^2

-- Define the theorem
theorem right_triangle_trig_values (t : RightTriangle) 
  (hyp : t.YZ = t.XZ) -- YZ is the hypotenuse
  (cos_z : Real.cos (Real.arccos ((8 * Real.sqrt 91) / 91)) = (8 * Real.sqrt 91) / 91)
  (xz_val : t.XZ = Real.sqrt 91) :
  t.XY = 8 ∧ Real.sin (Real.arccos ((8 * Real.sqrt 91) / 91)) = Real.sqrt 65 / 13 :=
by
  sorry


end NUMINAMATH_CALUDE_right_triangle_trig_values_l3693_369333


namespace NUMINAMATH_CALUDE_initial_average_production_l3693_369353

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h1 : n = 1)
  (h2 : today_production = 60)
  (h3 : new_average = 55) :
  ∃ initial_average : ℕ, initial_average = 50 ∧ 
    (initial_average * n + today_production) / (n + 1) = new_average := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l3693_369353


namespace NUMINAMATH_CALUDE_no_common_tangent_for_three_circles_l3693_369325

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Represents a configuration of three circles -/
structure ThreeCircleConfig where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  h12 : are_externally_tangent c1 c2
  h23 : are_externally_tangent c2 c3
  h31 : are_externally_tangent c3 c1

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a line is tangent to a circle -/
def is_tangent_to (l : Line) (c : Circle) : Prop :=
  let (x, y) := c.center
  (l.a * x + l.b * y + l.c)^2 = (l.a^2 + l.b^2) * c.radius^2

/-- The main theorem -/
theorem no_common_tangent_for_three_circles (config : ThreeCircleConfig) 
    (h1 : config.c1.radius = 3)
    (h2 : config.c2.radius = 4)
    (h3 : config.c3.radius = 5) :
  ¬∃ (l : Line), is_tangent_to l config.c1 ∧ is_tangent_to l config.c2 ∧ is_tangent_to l config.c3 :=
by sorry

end NUMINAMATH_CALUDE_no_common_tangent_for_three_circles_l3693_369325


namespace NUMINAMATH_CALUDE_cindy_marbles_problem_l3693_369339

def friends_given_marbles (initial_marbles : ℕ) (marbles_per_friend : ℕ) (remaining_marbles_multiplier : ℕ) (final_multiplied_marbles : ℕ) : ℕ :=
  (initial_marbles - final_multiplied_marbles / remaining_marbles_multiplier) / marbles_per_friend

theorem cindy_marbles_problem :
  friends_given_marbles 500 80 4 720 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_problem_l3693_369339


namespace NUMINAMATH_CALUDE_tan_problem_l3693_369344

theorem tan_problem (α : Real) (h : Real.tan (α + π/3) = 2) :
  (Real.sin (α + 4*π/3) + Real.cos (2*π/3 - α)) /
  (Real.cos (π/6 - α) - Real.sin (α + 5*π/6)) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_problem_l3693_369344


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_negative_four_squared_l3693_369347

theorem arithmetic_square_root_of_negative_four_squared : Real.sqrt ((-4)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_negative_four_squared_l3693_369347


namespace NUMINAMATH_CALUDE_profit_increase_l3693_369361

theorem profit_increase (initial_profit : ℝ) (march_to_april : ℝ) : 
  (initial_profit * (1 + march_to_april / 100) * 0.8 * 1.5 = initial_profit * 1.5600000000000001) →
  march_to_april = 30 :=
by sorry

end NUMINAMATH_CALUDE_profit_increase_l3693_369361


namespace NUMINAMATH_CALUDE_extremum_at_one_l3693_369367

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

theorem extremum_at_one (a : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1 ∨ f a x ≥ f a 1) → 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_extremum_at_one_l3693_369367


namespace NUMINAMATH_CALUDE_max_cookie_price_l3693_369350

theorem max_cookie_price (k p : ℕ) 
  (h1 : 8 * k + 3 * p < 200)
  (h2 : 4 * k + 5 * p > 150) :
  k ≤ 19 ∧ ∃ (k' p' : ℕ), k' = 19 ∧ 8 * k' + 3 * p' < 200 ∧ 4 * k' + 5 * p' > 150 :=
sorry

end NUMINAMATH_CALUDE_max_cookie_price_l3693_369350


namespace NUMINAMATH_CALUDE_expression_equals_nine_l3693_369378

theorem expression_equals_nine : 3 * 3 - 3 + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_nine_l3693_369378


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3693_369391

/-- Given two vectors a and b in ℝ², where a = (1,2) and b = (3,x), 
    if (a + b) is perpendicular to a, then x = -4. -/
theorem perpendicular_vectors_x_value : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![3, x]
  (∀ i : Fin 2, (a + b) i * a i = 0) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3693_369391
