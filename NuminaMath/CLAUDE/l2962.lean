import Mathlib

namespace NUMINAMATH_CALUDE_abc_perfect_cube_l2962_296248

theorem abc_perfect_cube (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ∃ (n : ℤ), (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = n) :
  ∃ (k : ℤ), a * b * c = k^3 := by
sorry

end NUMINAMATH_CALUDE_abc_perfect_cube_l2962_296248


namespace NUMINAMATH_CALUDE_expression_is_equation_l2962_296223

/-- Definition of an equation -/
def is_equation (e : Prop) : Prop :=
  ∃ (x : ℝ) (f g : ℝ → ℝ), e = (f x = g x)

/-- The expression 2x - 1 = 3 -/
def expression : Prop :=
  ∃ x : ℝ, 2 * x - 1 = 3

/-- Theorem: The given expression is an equation -/
theorem expression_is_equation : is_equation expression :=
sorry

end NUMINAMATH_CALUDE_expression_is_equation_l2962_296223


namespace NUMINAMATH_CALUDE_share_ratio_problem_l2962_296271

theorem share_ratio_problem (total : ℕ) (john_share : ℕ) :
  total = 4800 →
  john_share = 1600 →
  ∃ (jose_share binoy_share : ℕ),
    total = john_share + jose_share + binoy_share ∧
    2 * jose_share = 4 * john_share ∧
    3 * jose_share = 6 * john_share ∧
    binoy_share = 3 * john_share :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_problem_l2962_296271


namespace NUMINAMATH_CALUDE_truck_distance_proof_l2962_296201

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The distance traveled by the truck -/
def truck_distance : ℕ := arithmetic_sum 5 7 30

theorem truck_distance_proof : truck_distance = 3195 := by
  sorry

end NUMINAMATH_CALUDE_truck_distance_proof_l2962_296201


namespace NUMINAMATH_CALUDE_fraction_simplification_l2962_296238

theorem fraction_simplification : (5 * 6 - 3 * 4) / (6 + 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2962_296238


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l2962_296216

theorem two_digit_number_problem : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  (n / 10) * 2 = (n % 10) * 3 ∧
  (n / 10) = (n % 10) + 3 ∧
  n = 63 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l2962_296216


namespace NUMINAMATH_CALUDE_fifth_row_dots_l2962_296215

/-- Represents the number of green dots in a row -/
def greenDots : ℕ → ℕ
  | 0 => 3  -- First row (index 0) has 3 dots
  | n + 1 => greenDots n + 3  -- Each subsequent row increases by 3 dots

/-- The theorem stating that the fifth row has 15 green dots -/
theorem fifth_row_dots : greenDots 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifth_row_dots_l2962_296215


namespace NUMINAMATH_CALUDE_pascal_diagonal_sum_equals_fibonacci_l2962_296224

/-- Sum of numbers in the n-th diagonal of Pascal's triangle -/
def b (n : ℕ) : ℕ := sorry

/-- n-th term of the Fibonacci sequence -/
def a : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => a (n + 1) + a n

/-- Theorem stating that b_n equals a_n for all n -/
theorem pascal_diagonal_sum_equals_fibonacci (n : ℕ) : b n = a n := by
  sorry

end NUMINAMATH_CALUDE_pascal_diagonal_sum_equals_fibonacci_l2962_296224


namespace NUMINAMATH_CALUDE_davids_math_marks_l2962_296241

def english_marks : ℝ := 74
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 67
def biology_marks : ℝ := 90
def average_marks : ℝ := 75.6
def num_subjects : ℕ := 5

theorem davids_math_marks :
  let total_marks := average_marks * num_subjects
  let known_marks := english_marks + physics_marks + chemistry_marks + biology_marks
  let math_marks := total_marks - known_marks
  math_marks = 65 := by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l2962_296241


namespace NUMINAMATH_CALUDE_candy_car_cost_proof_l2962_296239

/-- The cost of a candy car given initial amount and change received -/
def candy_car_cost (initial_amount change : ℚ) : ℚ :=
  initial_amount - change

/-- Theorem stating the cost of the candy car is $0.45 -/
theorem candy_car_cost_proof (initial_amount change : ℚ) 
  (h1 : initial_amount = 1.80)
  (h2 : change = 1.35) : 
  candy_car_cost initial_amount change = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_candy_car_cost_proof_l2962_296239


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l2962_296294

theorem logarithmic_equation_solution (x : ℝ) (h1 : x > 1) :
  (Real.log x - 1) / Real.log 5 + 
  (Real.log (x^2 - 1)) / (Real.log 5 / 2) + 
  (Real.log (x - 1)) / (Real.log (1/5)) = 3 →
  x = Real.sqrt (5 * Real.sqrt 5 + 1) := by
sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l2962_296294


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l2962_296249

theorem intersection_point_of_lines (x y : ℝ) :
  y = x ∧ y = -x + 2 → (x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l2962_296249


namespace NUMINAMATH_CALUDE_identify_geometric_bodies_l2962_296269

/-- Represents the possible geometric bodies --/
inductive GeometricBody
  | TriangularPrism
  | QuadrangularPyramid
  | Cone
  | Frustum
  | TriangularFrustum
  | TriangularPyramid

/-- Represents a view of a geometric body --/
structure View where
  -- We'll assume some properties of the view, but won't define them explicitly
  dummy : Unit

/-- A function that determines if a set of views corresponds to a specific geometric body --/
def viewsMatchBody (views : List View) (body : GeometricBody) : Bool :=
  sorry -- The actual implementation would depend on how we define views

/-- The theorem stating that given the correct views, we can identify the four bodies --/
theorem identify_geometric_bodies 
  (views1 views2 views3 views4 : List View) 
  (h1 : viewsMatchBody views1 GeometricBody.TriangularPrism)
  (h2 : viewsMatchBody views2 GeometricBody.QuadrangularPyramid)
  (h3 : viewsMatchBody views3 GeometricBody.Cone)
  (h4 : viewsMatchBody views4 GeometricBody.Frustum) :
  ∃ (bodies : List GeometricBody), 
    bodies = [GeometricBody.TriangularPrism, 
              GeometricBody.QuadrangularPyramid, 
              GeometricBody.Cone, 
              GeometricBody.Frustum] ∧
    (∀ (views : List View), 
      views ∈ [views1, views2, views3, views4] → 
      ∃ (body : GeometricBody), body ∈ bodies ∧ viewsMatchBody views body) :=
by
  sorry


end NUMINAMATH_CALUDE_identify_geometric_bodies_l2962_296269


namespace NUMINAMATH_CALUDE_cubic_greater_than_quadratic_l2962_296275

theorem cubic_greater_than_quadratic (x : ℝ) (h : x > 1) : x^3 > x^2 - x + 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_greater_than_quadratic_l2962_296275


namespace NUMINAMATH_CALUDE_find_other_number_l2962_296217

theorem find_other_number (x y : ℤ) : 
  (3 * x + 4 * y = 151) → 
  ((x = 19 ∨ y = 19) → (x = 25 ∨ y = 25)) := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l2962_296217


namespace NUMINAMATH_CALUDE_train_speed_l2962_296299

/-- The speed of a train given its length, time to cross a moving person, and the person's speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (person_speed : ℝ) :
  train_length = 500 →
  crossing_time = 29.997600191984642 →
  person_speed = 3 →
  ∃ (train_speed : ℝ), abs (train_speed - 63) < 0.1 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2962_296299


namespace NUMINAMATH_CALUDE_halfway_fraction_l2962_296289

theorem halfway_fraction : (3 : ℚ) / 4 + ((5 : ℚ) / 6 - (3 : ℚ) / 4) / 2 = (19 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l2962_296289


namespace NUMINAMATH_CALUDE_point_distance_range_l2962_296227

/-- Given points A(0,1) and B(0,4), and a point P on the line 2x-y+m=0 such that |PA| = 1/2|PB|,
    the range of values for m is -2√5 ≤ m ≤ 2√5. -/
theorem point_distance_range (m : ℝ) : 
  (∃ (x y : ℝ), 2*x - y + m = 0 ∧ 
    (x^2 + (y-1)^2)^(1/2) = 1/2 * (x^2 + (y-4)^2)^(1/2)) 
  ↔ -2 * Real.sqrt 5 ≤ m ∧ m ≤ 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_point_distance_range_l2962_296227


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2962_296257

theorem quadratic_equation_roots (k : ℝ) :
  let f := fun x : ℝ => x^2 - 2*(k-1)*x + k^2
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  (k ≤ 1/2) ∧
  (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁*x₂ + x₁ + x₂ - 1 = 0 → k = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2962_296257


namespace NUMINAMATH_CALUDE_die_roll_probability_l2962_296274

def roll_outcome := Fin 6

def is_valid_outcome (m n : roll_outcome) : Prop :=
  m.val + 1 = 2 * (n.val + 1)

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 3

theorem die_roll_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_die_roll_probability_l2962_296274


namespace NUMINAMATH_CALUDE_remainder_11_power_1995_mod_5_l2962_296234

theorem remainder_11_power_1995_mod_5 : 11^1995 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_power_1995_mod_5_l2962_296234


namespace NUMINAMATH_CALUDE_triangle_problem_l2962_296292

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧
  Real.sqrt 3 * a = 2 * c * Real.sin A ∧  -- Given condition
  c = Real.sqrt 7 ∧  -- Given condition
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →  -- Area condition
  C = π/3 ∧ a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2962_296292


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l2962_296254

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes,
    with at least one empty box -/
def distributeWithEmpty (n : ℕ) (k : ℕ) : ℕ := k * (k-1)^n

theorem ball_distribution_problem :
  distribute 6 3 - distributeWithEmpty 6 3 = 537 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l2962_296254


namespace NUMINAMATH_CALUDE_linear_function_constant_point_l2962_296208

theorem linear_function_constant_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_constant_point_l2962_296208


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l2962_296270

theorem tangent_line_to_ln_curve (b : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1/2 * x + b = Real.log x) ∧ 
  (∀ y : ℝ, y > 0 → 1/2 * y + b ≥ Real.log y)) → 
  b = Real.log 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_curve_l2962_296270


namespace NUMINAMATH_CALUDE_class_sports_census_suitable_l2962_296245

/-- Represents a survey --/
inductive Survey
  | LightBulbLifespan
  | ClassSportsActivity
  | YangtzeRiverFish
  | PlasticBagDisposal

/-- Represents the characteristics of a survey --/
structure SurveyCharacteristics where
  population_size : ℕ
  data_collection_time : ℕ
  resource_intensity : ℕ

/-- Determines if a survey is feasible and practical for a census --/
def is_census_suitable (s : Survey) (c : SurveyCharacteristics) : Prop :=
  c.population_size ≤ 1000 ∧ c.data_collection_time ≤ 7 ∧ c.resource_intensity ≤ 5

/-- The characteristics of the class sports activity survey --/
def class_sports_characteristics : SurveyCharacteristics :=
  { population_size := 30
  , data_collection_time := 1
  , resource_intensity := 2 }

/-- Theorem stating that the class sports activity survey is suitable for a census --/
theorem class_sports_census_suitable :
  is_census_suitable Survey.ClassSportsActivity class_sports_characteristics :=
sorry

end NUMINAMATH_CALUDE_class_sports_census_suitable_l2962_296245


namespace NUMINAMATH_CALUDE_coin_difference_l2962_296228

theorem coin_difference (total_coins quarters : ℕ) 
  (h1 : total_coins = 77)
  (h2 : quarters = 29) : 
  total_coins - quarters = 48 := by
  sorry

end NUMINAMATH_CALUDE_coin_difference_l2962_296228


namespace NUMINAMATH_CALUDE_number_calculation_l2962_296214

theorem number_calculation (n : ℝ) : 
  0.1 * 0.3 * ((Real.sqrt (0.5 * n))^2) = 90 → n = 6000 := by
sorry

end NUMINAMATH_CALUDE_number_calculation_l2962_296214


namespace NUMINAMATH_CALUDE_odd_periodic_function_sum_l2962_296283

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period T if f(x + T) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem odd_periodic_function_sum (f : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_period : HasPeriod f 4) :
  f 2005 + f 2006 + f 2007 = f 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_sum_l2962_296283


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2962_296256

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 / 9 = 0) ↔ (y = (3/2) * x ∨ y = -(3/2) * x) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2962_296256


namespace NUMINAMATH_CALUDE_trajectory_of_G_l2962_296276

/-- The ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Point P on the ellipse C -/
def point_P (x₀ y₀ : ℝ) : Prop := ellipse_C x₀ y₀

/-- Relation between vectors PG and GO -/
def vector_relation (x₀ y₀ x y : ℝ) : Prop :=
  (x - x₀, y - y₀) = (2 * (-x), 2 * (-y))

/-- Trajectory of point G -/
def trajectory_G (x y : ℝ) : Prop := 9*x^2/4 + 3*y^2 = 1

theorem trajectory_of_G (x₀ y₀ x y : ℝ) :
  point_P x₀ y₀ → vector_relation x₀ y₀ x y → trajectory_G x y := by sorry

end NUMINAMATH_CALUDE_trajectory_of_G_l2962_296276


namespace NUMINAMATH_CALUDE_tangent_line_a_range_l2962_296220

/-- The line equation ax + y - 2 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y - 2 = 0

/-- The first circle equation (x-1)² + y² = 1 -/
def circle1_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The second circle equation x² + (y-1)² = 1/4 -/
def circle2_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1/4

/-- The line is tangent to both circles -/
def is_tangent_to_both_circles (a : ℝ) : Prop :=
  ∃ x y, line_equation a x y ∧ 
         ((circle1_equation x y ∧ ¬∃ x' y', x' ≠ x ∧ y' ≠ y ∧ line_equation a x' y' ∧ circle1_equation x' y') ∨
          (circle2_equation x y ∧ ¬∃ x' y', x' ≠ x ∧ y' ≠ y ∧ line_equation a x' y' ∧ circle2_equation x' y'))

theorem tangent_line_a_range :
  ∀ a : ℝ, is_tangent_to_both_circles a ↔ -Real.sqrt 3 < a ∧ a < 3/4 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_a_range_l2962_296220


namespace NUMINAMATH_CALUDE_cats_dogs_ratio_l2962_296263

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
theorem cats_dogs_ratio (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) (num_dogs : ℕ) : 
  cat_ratio * num_dogs = dog_ratio * num_cats → 
  cat_ratio = 3 → 
  dog_ratio = 4 → 
  num_cats = 18 → 
  num_dogs = 24 := by
sorry

end NUMINAMATH_CALUDE_cats_dogs_ratio_l2962_296263


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2962_296232

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Sum of all permutations of a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  100 * (n.hundreds + n.tens + n.ones) +
  10 * (n.hundreds + n.tens + n.ones) +
  (n.hundreds + n.tens + n.ones)

/-- The sum of digits of a three-digit number -/
def sumOfDigits (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

theorem unique_three_digit_number :
  ∀ n : ThreeDigitNumber,
    sumOfPermutations n = 4410 ∧
    Even (sumOfDigits n) →
    n.hundreds = 4 ∧ n.tens = 4 ∧ n.ones = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2962_296232


namespace NUMINAMATH_CALUDE_stating_isosceles_triangles_properties_l2962_296247

/-- 
Represents the number of isosceles triangles with vertices of the same color 
in a regular (6n+1)-gon with k red vertices and the rest blue.
-/
def P (n : ℕ) (k : ℕ) : ℕ := sorry

/-- 
Theorem stating the properties of P for a regular (6n+1)-gon 
with k red vertices and the rest blue.
-/
theorem isosceles_triangles_properties (n : ℕ) (k : ℕ) : 
  (P n (k + 1) - P n k = 3 * k - 9 * n) ∧ 
  (P n k = 3 * n * (6 * n + 1) - 9 * k * n + (3 * k * (k - 1)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_stating_isosceles_triangles_properties_l2962_296247


namespace NUMINAMATH_CALUDE_honey_jar_problem_l2962_296240

/-- Represents the process of drawing out honey and replacing with sugar solution --/
def draw_and_replace (initial_honey : ℝ) (percent : ℝ) : ℝ :=
  initial_honey * (1 - percent)

/-- The amount of honey remaining after three iterations --/
def remaining_honey (initial_honey : ℝ) : ℝ :=
  draw_and_replace (draw_and_replace (draw_and_replace initial_honey 0.3) 0.4) 0.5

/-- Theorem stating that if 315 grams of honey remain after the process, 
    the initial amount was 1500 grams --/
theorem honey_jar_problem (initial_honey : ℝ) :
  remaining_honey initial_honey = 315 → initial_honey = 1500 := by
  sorry

end NUMINAMATH_CALUDE_honey_jar_problem_l2962_296240


namespace NUMINAMATH_CALUDE_sum_distances_specific_triangle_l2962_296278

/-- The sum of distances from a point to the vertices of a triangle, expressed as x + y√z --/
def sum_distances (A B C P : ℝ × ℝ) : ℝ × ℝ × ℕ :=
  sorry

/-- Theorem stating the sum of distances for specific triangle and point --/
theorem sum_distances_specific_triangle :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (10, 2)
  let C : ℝ × ℝ := (5, 4)
  let P : ℝ × ℝ := (3, 1)
  let (x, y, z) := sum_distances A B C P
  x + y + z = 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_distances_specific_triangle_l2962_296278


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2962_296286

theorem coin_flip_probability (n : ℕ) : n = 6 →
  (1 + n + n * (n - 1) / 2 : ℚ) / 2^n = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2962_296286


namespace NUMINAMATH_CALUDE_unique_interior_point_is_centroid_l2962_296236

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def isInside (P : LatticePoint) (T : LatticeTriangle) : Prop :=
  sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def isOnBoundary (P : LatticePoint) (T : LatticeTriangle) : Prop :=
  sorry

/-- The centroid of a triangle -/
def centroid (T : LatticeTriangle) : LatticePoint :=
  sorry

/-- Main theorem -/
theorem unique_interior_point_is_centroid (T : LatticeTriangle) (P : LatticePoint) :
  (∀ Q : LatticePoint, isOnBoundary Q T → (Q = T.A ∨ Q = T.B ∨ Q = T.C)) →
  isInside P T →
  (∀ Q : LatticePoint, isInside Q T → Q = P) →
  P = centroid T :=
by sorry

end NUMINAMATH_CALUDE_unique_interior_point_is_centroid_l2962_296236


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2962_296277

def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |3*x - 15|

theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 3 10, g x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 3 10, g x = max) ∧
    (∀ x ∈ Set.Icc 3 10, min ≤ g x) ∧ 
    (∃ x ∈ Set.Icc 3 10, g x = min) ∧
    max + min = -21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2962_296277


namespace NUMINAMATH_CALUDE_age_and_marriage_relations_l2962_296295

-- Define the people
inductive Person : Type
| Roman : Person
| Oleg : Person
| Ekaterina : Person
| Zhanna : Person

-- Define the age relation
def olderThan : Person → Person → Prop := sorry

-- Define the marriage relation
def marriedTo : Person → Person → Prop := sorry

-- Theorem statement
theorem age_and_marriage_relations :
  -- Each person has a different age
  (∀ p q : Person, p ≠ q → (olderThan p q ∨ olderThan q p)) →
  -- Each husband is older than his wife
  (∀ p q : Person, marriedTo p q → olderThan p q) →
  -- Zhanna is older than Oleg
  olderThan Person.Zhanna Person.Oleg →
  -- There are exactly two married couples
  (∃! (p1 p2 q1 q2 : Person),
    p1 ≠ p2 ∧ q1 ≠ q2 ∧ p1 ≠ q1 ∧ p1 ≠ q2 ∧ p2 ≠ q1 ∧ p2 ≠ q2 ∧
    marriedTo p1 p2 ∧ marriedTo q1 q2) →
  -- Conclusion: Oleg is older than Ekaterina and Roman is the oldest and married to Zhanna
  olderThan Person.Oleg Person.Ekaterina ∧
  marriedTo Person.Roman Person.Zhanna ∧
  (∀ p : Person, p ≠ Person.Roman → olderThan Person.Roman p) :=
by sorry

end NUMINAMATH_CALUDE_age_and_marriage_relations_l2962_296295


namespace NUMINAMATH_CALUDE_polynomial_value_l2962_296261

theorem polynomial_value (a b c d : ℝ) : 
  (∀ x, a * x^5 + b * x^3 + c * x + d = 
    (fun x => a * x^5 + b * x^3 + c * x + d) x) →
  (a * 0^5 + b * 0^3 + c * 0 + d = -5) →
  (a * (-3)^5 + b * (-3)^3 + c * (-3) + d = 7) →
  (a * 3^5 + b * 3^3 + c * 3 + d = -17) := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l2962_296261


namespace NUMINAMATH_CALUDE_power_minus_self_even_l2962_296244

theorem power_minus_self_even (a n : ℕ+) : 
  ∃ k : ℤ, (a^n.val - a : ℤ) = 2 * k := by sorry

end NUMINAMATH_CALUDE_power_minus_self_even_l2962_296244


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l2962_296264

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem number_puzzle_solution :
  ∀ (A B C : ℕ),
  (sum_of_digits A = B) →
  (sum_of_digits B = C) →
  (A + B + C = 60) →
  (A = 44 ∨ A = 50 ∨ A = 47) :=
by sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l2962_296264


namespace NUMINAMATH_CALUDE_larry_win_probability_l2962_296266

/-- The probability of Larry winning a turn-based game against Julius, where:
  * Larry throws first
  * The probability of Larry knocking off the bottle is 3/5
  * The probability of Julius knocking off the bottle is 1/3
  * The winner is the first to knock off the bottle -/
theorem larry_win_probability (p_larry : ℝ) (p_julius : ℝ) 
  (h1 : p_larry = 3/5) 
  (h2 : p_julius = 1/3) :
  p_larry + (1 - p_larry) * (1 - p_julius) * p_larry / (1 - (1 - p_larry) * (1 - p_julius)) = 9/11 :=
by sorry

end NUMINAMATH_CALUDE_larry_win_probability_l2962_296266


namespace NUMINAMATH_CALUDE_max_reflections_theorem_l2962_296293

/-- The angle between the two reflecting lines in degrees -/
def angle : ℝ := 12

/-- The maximum angle of incidence in degrees -/
def max_incidence : ℝ := 90

/-- The maximum number of reflections possible -/
def max_reflections : ℕ := 7

/-- Theorem stating the maximum number of reflections possible given the angle between lines -/
theorem max_reflections_theorem : 
  ∀ n : ℕ, (n : ℝ) * angle ≤ max_incidence ↔ n ≤ max_reflections :=
by sorry

end NUMINAMATH_CALUDE_max_reflections_theorem_l2962_296293


namespace NUMINAMATH_CALUDE_combined_mean_of_three_sets_l2962_296255

theorem combined_mean_of_three_sets (set1_count : ℕ) (set1_mean : ℚ)
                                    (set2_count : ℕ) (set2_mean : ℚ)
                                    (set3_count : ℕ) (set3_mean : ℚ) :
  set1_count = 7 ∧ set1_mean = 15 ∧
  set2_count = 8 ∧ set2_mean = 20 ∧
  set3_count = 5 ∧ set3_mean = 12 →
  (set1_count * set1_mean + set2_count * set2_mean + set3_count * set3_mean) /
  (set1_count + set2_count + set3_count) = 325 / 20 := by
sorry

end NUMINAMATH_CALUDE_combined_mean_of_three_sets_l2962_296255


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2962_296262

theorem absolute_value_equation_solution :
  ∃! n : ℚ, |n + 4| = 3 - n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2962_296262


namespace NUMINAMATH_CALUDE_regions_divisible_by_six_l2962_296265

/-- Represents a triangle with sides divided into congruent segments --/
structure DividedTriangle where
  segments : ℕ
  segments_pos : segments > 0

/-- Calculates the number of regions formed in a divided triangle --/
def num_regions (t : DividedTriangle) : ℕ :=
  t.segments^2 + (2*t.segments - 1) * (t.segments - 1) - r t
where
  /-- Number of points where three lines intersect (excluding vertices) --/
  r (t : DividedTriangle) : ℕ := sorry

/-- The main theorem stating that the number of regions is divisible by 6 --/
theorem regions_divisible_by_six (t : DividedTriangle) (h : t.segments = 2002) :
  6 ∣ num_regions t := by sorry

end NUMINAMATH_CALUDE_regions_divisible_by_six_l2962_296265


namespace NUMINAMATH_CALUDE_unique_prime_square_product_l2962_296221

theorem unique_prime_square_product (a b c : ℕ) : 
  (Nat.Prime (a^2 + 1)) ∧ 
  (Nat.Prime (b^2 + 1)) ∧ 
  ((a^2 + 1) * (b^2 + 1) = c^2 + 1) →
  a = 2 ∧ b = 1 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_square_product_l2962_296221


namespace NUMINAMATH_CALUDE_parabola_points_m_range_l2962_296210

/-- The parabola equation -/
def parabola (a x : ℝ) : ℝ := a * x^2 - 2 * a * x - 3

theorem parabola_points_m_range (a x₁ x₂ y₁ y₂ m : ℝ) : 
  a ≠ 0 →
  parabola a x₁ = y₁ →
  parabola a x₂ = y₂ →
  -2 < x₁ →
  x₁ < 0 →
  m < x₂ →
  x₂ < m + 1 →
  y₁ ≠ y₂ →
  (0 < m ∧ m ≤ 1) ∨ m ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_points_m_range_l2962_296210


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2962_296207

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2962_296207


namespace NUMINAMATH_CALUDE_problem_statement_l2962_296202

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- State the theorem
theorem problem_statement (m : ℝ) :
  (∀ x, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1 : ℝ) 1) →
  (m = 1 ∧
   ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
     1/a + 1/(2*b) + 1/(3*c) = m →
     a + 2*b + 3*c ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2962_296202


namespace NUMINAMATH_CALUDE_initial_peanuts_count_l2962_296206

theorem initial_peanuts_count (initial final added : ℕ) 
  (h1 : added = 4)
  (h2 : final = 8)
  (h3 : final = initial + added) : 
  initial = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_peanuts_count_l2962_296206


namespace NUMINAMATH_CALUDE_least_equal_bulbs_l2962_296298

def tulip_pack_size : ℕ := 15
def daffodil_pack_size : ℕ := 16

theorem least_equal_bulbs :
  ∃ (n : ℕ), n > 0 ∧ n % tulip_pack_size = 0 ∧ n % daffodil_pack_size = 0 ∧
  ∀ (m : ℕ), (m > 0 ∧ m % tulip_pack_size = 0 ∧ m % daffodil_pack_size = 0) → m ≥ n :=
by
  use 240
  sorry

end NUMINAMATH_CALUDE_least_equal_bulbs_l2962_296298


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_base_11_perfect_square_eleven_is_smallest_l2962_296259

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 5 → (∃ n : ℕ, 4 * b + 5 = n * n) → b ≥ 11 :=
by
  sorry

theorem base_11_perfect_square : 
  ∃ n : ℕ, 4 * 11 + 5 = n * n :=
by
  sorry

theorem eleven_is_smallest : 
  ∀ b : ℕ, b > 5 ∧ b < 11 → ¬(∃ n : ℕ, 4 * b + 5 = n * n) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_base_11_perfect_square_eleven_is_smallest_l2962_296259


namespace NUMINAMATH_CALUDE_triangle_max_area_l2962_296212

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area is √3 when a = 2 and (2+b)(sin A - sin B) = (c-b)sin C -/
theorem triangle_max_area (a b c A B C : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_a : a = 2)
    (h_angles : A + B + C = π)
    (h_sine_law : (2 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
    (∀ a' b' c' A' B' C', 
      a' > 0 ∧ b' > 0 ∧ c' > 0 → 
      a' + b' > c' ∧ b' + c' > a' ∧ c' + a' > b' →
      a' = 2 →
      A' + B' + C' = π →
      (2 + b') * (Real.sin A' - Real.sin B') = (c' - b') * Real.sin C' →
      1/2 * b * c * Real.sin A ≥ 1/2 * b' * c' * Real.sin A') ∧
    1/2 * b * c * Real.sin A ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2962_296212


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_3136_l2962_296218

theorem largest_prime_factor_of_3136 (p : Nat) : 
  Nat.Prime p ∧ p ∣ 3136 → p ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_3136_l2962_296218


namespace NUMINAMATH_CALUDE_square_side_length_from_hexagons_l2962_296243

/-- The side length of a square formed by repositioning two congruent hexagons cut from a rectangle -/
def square_side_length (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  sorry

/-- The height of each hexagon cut from the rectangle -/
def hexagon_height (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  sorry

theorem square_side_length_from_hexagons
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (h1 : rectangle_width = 9)
  (h2 : rectangle_height = 27)
  (h3 : square_side_length rectangle_width rectangle_height =
        2 * hexagon_height rectangle_width rectangle_height) :
  square_side_length rectangle_width rectangle_height = 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_from_hexagons_l2962_296243


namespace NUMINAMATH_CALUDE_expansion_equality_l2962_296209

theorem expansion_equality (m : ℝ) : (m + 2) * (m - 3) = m^2 - m - 6 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l2962_296209


namespace NUMINAMATH_CALUDE_linear_function_existence_l2962_296267

theorem linear_function_existence (k : ℝ) (h1 : k ≠ 0) 
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → k * x1 + 3 < k * x2 + 3) :
  ∃ y : ℝ, y = k * (-2) + 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_existence_l2962_296267


namespace NUMINAMATH_CALUDE_prove_newly_added_groups_l2962_296250

/-- Represents the number of groups of students recently added to the class -/
def newly_added_groups : ℕ := 2

theorem prove_newly_added_groups :
  let tables : ℕ := 6
  let students_per_table : ℕ := 3
  let bathroom_students : ℕ := 3
  let canteen_students : ℕ := 3 * bathroom_students
  let students_per_group : ℕ := 4
  let foreign_students : ℕ := 3 * 3  -- 3 each from 3 countries
  let total_students : ℕ := 47
  newly_added_groups = 
    (total_students - (tables * students_per_table + bathroom_students + canteen_students + foreign_students)) / students_per_group :=
by
  sorry

#check prove_newly_added_groups

end NUMINAMATH_CALUDE_prove_newly_added_groups_l2962_296250


namespace NUMINAMATH_CALUDE_cosine_rational_values_l2962_296273

theorem cosine_rational_values (α : ℚ) (h : ∃ (r : ℚ), r = Real.cos (α * Real.pi)) :
  Real.cos (α * Real.pi) = 0 ∨
  Real.cos (α * Real.pi) = 1 ∨
  Real.cos (α * Real.pi) = -1 ∨
  Real.cos (α * Real.pi) = (1/2) ∨
  Real.cos (α * Real.pi) = -(1/2) :=
by sorry

end NUMINAMATH_CALUDE_cosine_rational_values_l2962_296273


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l2962_296272

/-- The number of ropes after n cuts -/
def num_ropes (n : ℕ) : ℕ := 1 + 4 * n

/-- The problem statement -/
theorem rope_cutting_problem :
  ∃ n : ℕ, num_ropes n = 2021 ∧ n = 505 := by sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l2962_296272


namespace NUMINAMATH_CALUDE_trig_simplification_l2962_296280

theorem trig_simplification (α : Real) :
  Real.sin (-α) * Real.cos (π + α) * Real.tan (2 * π + α) = Real.sin α ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l2962_296280


namespace NUMINAMATH_CALUDE_students_disliking_both_l2962_296226

theorem students_disliking_both (total : ℕ) (fries : ℕ) (burgers : ℕ) (both : ℕ) :
  total = 25 →
  fries = 15 →
  burgers = 10 →
  both = 6 →
  total - (fries + burgers - both) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_students_disliking_both_l2962_296226


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2962_296205

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x < -1 → x^2 - 1 > 0) ∧ 
  (∃ y : ℝ, y^2 - 1 > 0 ∧ ¬(y < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2962_296205


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2962_296204

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (a : ℚ) (b : ℕ), x = a * Real.sqrt b ∧ 
  (∀ (c : ℚ) (d : ℕ), x = c * Real.sqrt d → b ≤ d)

theorem simplest_quadratic_radical : 
  is_simplest_quadratic_radical (-Real.sqrt 3) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 0.5) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/7)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 8) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2962_296204


namespace NUMINAMATH_CALUDE_four_bottles_cost_l2962_296279

/-- The cost of a certain number of bottles of mineral water -/
def cost (bottles : ℕ) : ℚ :=
  if bottles = 3 then 3/2 else (3/2) * (bottles : ℚ) / 3

/-- Theorem: The cost of 4 bottles of mineral water is 2, given that 3 bottles cost 1.50 -/
theorem four_bottles_cost : cost 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_bottles_cost_l2962_296279


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l2962_296290

/-- The number of divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Check if a number has exactly 12 divisors -/
def has_12_divisors (n : ℕ+) : Prop := num_divisors n = 12

theorem smallest_with_12_divisors :
  ∃ (n : ℕ+), has_12_divisors n ∧ ∀ (m : ℕ+), has_12_divisors m → n ≤ m :=
by
  use 288
  sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l2962_296290


namespace NUMINAMATH_CALUDE_sampling_is_systematic_l2962_296203

/-- Represents a student ID number -/
structure StudentID where
  lastThreeDigits : Nat
  inv_range : 1 ≤ lastThreeDigits ∧ lastThreeDigits ≤ 818

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | Stratified
  | Systematic
  | RandomNumberTable

/-- Represents the selection criteria for inspection -/
def isSelected (id : StudentID) : Bool :=
  id.lastThreeDigits % 100 = 16

/-- Theorem stating that the sampling method is systematic -/
theorem sampling_is_systematic (ids : List StudentID) 
  (h1 : ∀ id ∈ ids, 1 ≤ id.lastThreeDigits ∧ id.lastThreeDigits ≤ 818) 
  (h2 : ∀ id ∈ ids, isSelected id ↔ id.lastThreeDigits % 100 = 16) : 
  SamplingMethod.Systematic = SamplingMethod.Systematic := by
  sorry

end NUMINAMATH_CALUDE_sampling_is_systematic_l2962_296203


namespace NUMINAMATH_CALUDE_economics_law_tournament_l2962_296219

theorem economics_law_tournament (n : ℕ) (m : ℕ) : 
  220 < n → n < 254 →
  m < n →
  (n - 2 * m)^2 = n →
  ∀ k : ℕ, (220 < k ∧ k < 254 ∧ k < n ∧ (k - 2 * (n - k))^2 = k) → n - m ≤ k - (n - k) →
  n - m = 105 := by
sorry

end NUMINAMATH_CALUDE_economics_law_tournament_l2962_296219


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l2962_296231

/-- Given an initial angle of 60 degrees and a clockwise rotation of 600 degrees,
    the resulting new acute angle is 60 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 →
  rotation = 600 →
  let effective_rotation := rotation % 360
  let new_angle := (effective_rotation - initial_angle) % 180
  new_angle = 60 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l2962_296231


namespace NUMINAMATH_CALUDE_jessica_watermelons_l2962_296253

/-- The number of watermelons Jessica has left -/
def watermelons_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Proof that Jessica has 8 watermelons left -/
theorem jessica_watermelons : watermelons_left 35 27 = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_watermelons_l2962_296253


namespace NUMINAMATH_CALUDE_area_of_triangle_l2962_296282

noncomputable def m (x y : ℝ) : ℝ × ℝ := (2 * Real.cos x, y - 2 * Real.sqrt 3 * Real.sin x * Real.cos x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (1, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + 1

theorem area_of_triangle (x y a b c : ℝ) : 
  (∃ k : ℝ, m x y = k • n x) → 
  f (c / 2) = 3 → 
  c = 2 * Real.sqrt 6 → 
  a + b = 6 → 
  (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_area_of_triangle_l2962_296282


namespace NUMINAMATH_CALUDE_inequality_solution_l2962_296237

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -7/6 ∨ x > -4/3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2962_296237


namespace NUMINAMATH_CALUDE_min_tiles_for_floor_l2962_296235

/-- Calculates the minimum number of rectangular tiles needed to cover a rectangular floor -/
def min_tiles_needed (tile_length inch_per_foot tile_width floor_length floor_width : ℕ) : ℕ :=
  let floor_area := (floor_length * inch_per_foot) * (floor_width * inch_per_foot)
  let tile_area := tile_length * tile_width
  (floor_area + tile_area - 1) / tile_area

/-- The minimum number of 5x6 inch tiles needed to cover a 3x4 foot floor is 58 -/
theorem min_tiles_for_floor :
  min_tiles_needed 5 12 6 3 4 = 58 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_floor_l2962_296235


namespace NUMINAMATH_CALUDE_ladder_slide_l2962_296268

theorem ladder_slide (ladder_length : ℝ) (initial_distance : ℝ) (slip_distance : ℝ) :
  ladder_length = 30 →
  initial_distance = 8 →
  slip_distance = 6 →
  let initial_height : ℝ := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height : ℝ := initial_height - slip_distance
  let new_distance : ℝ := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_ladder_slide_l2962_296268


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2962_296258

theorem infinitely_many_solutions (b : ℝ) : 
  (∀ x, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2962_296258


namespace NUMINAMATH_CALUDE_catch_up_solution_l2962_296252

/-- Represents the problem of a car catching up to a truck -/
def CatchUpProblem (truckSpeed carInitialSpeed carSpeedIncrease distance : ℝ) : Prop :=
  ∃ (t : ℝ),
    t > 0 ∧
    (carInitialSpeed * t + carSpeedIncrease * t * (t - 1) / 2) = (truckSpeed * t + distance)

/-- The solution to the catch-up problem -/
theorem catch_up_solution :
  CatchUpProblem 40 50 5 135 →
  ∃ (t : ℝ), t = 6 ∧ CatchUpProblem 40 50 5 135 := by sorry

#check catch_up_solution

end NUMINAMATH_CALUDE_catch_up_solution_l2962_296252


namespace NUMINAMATH_CALUDE_solution_is_rhombus_l2962_296297

def is_solution (x y : ℝ) : Prop :=
  max (|x + y|) (|x - y|) = 1

def rhombus_vertices : Set (ℝ × ℝ) :=
  {(-1, 0), (1, 0), (0, -1), (0, 1)}

theorem solution_is_rhombus :
  {p : ℝ × ℝ | is_solution p.1 p.2} = rhombus_vertices := by sorry

end NUMINAMATH_CALUDE_solution_is_rhombus_l2962_296297


namespace NUMINAMATH_CALUDE_triangle_inradius_l2962_296222

/-- The inradius of a triangle with side lengths 7, 24, and 25 is 3 -/
theorem triangle_inradius (a b c : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_inradius_l2962_296222


namespace NUMINAMATH_CALUDE_unique_solution_l2962_296260

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Checks if a number satisfies the first division scheme -/
def satisfies_first_scheme (n : FourDigitNumber) : Prop :=
  ∃ (d : ℕ), d < 10 ∧ 
  (n.value / d = 10 + (n.value / 100 % 10)) ∧
  (n.value % d = (n.value / 10 % 10) * 10 + (n.value % 10))

/-- Checks if a number satisfies the second division scheme -/
def satisfies_second_scheme (n : FourDigitNumber) : Prop :=
  ∃ (d : ℕ), d < 10 ∧
  (n.value / d = 168) ∧
  (n.value % d = 0)

/-- The main theorem stating that 1512 is the only number satisfying both schemes -/
theorem unique_solution : 
  ∃! (n : FourDigitNumber), 
    satisfies_first_scheme n ∧ 
    satisfies_second_scheme n ∧ 
    n.value = 1512 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2962_296260


namespace NUMINAMATH_CALUDE_triangle_side_length_l2962_296211

/-- Represents a triangle with sides a, b, c and angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- The theorem stating the relationship between sides and angles in the given triangle -/
theorem triangle_side_length (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.b = 3)
  (h3 : 3 * t.α + 2 * t.β = Real.pi)
  (h4 : t.α + t.β + t.γ = Real.pi)
  (h5 : 0 < t.a ∧ 0 < t.b ∧ 0 < t.c)
  (h6 : 0 < t.α ∧ 0 < t.β ∧ 0 < t.γ)
  (h7 : t.a / (Real.sin t.α) = t.b / (Real.sin t.β))
  (h8 : t.b / (Real.sin t.β) = t.c / (Real.sin t.γ)) :
  t.c = 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2962_296211


namespace NUMINAMATH_CALUDE_holiday_duty_arrangements_l2962_296229

def staff_count : ℕ := 6
def days_count : ℕ := 3
def staff_per_day : ℕ := 2

def arrangement_count (n m k : ℕ) (restricted_days : ℕ) : ℕ :=
  (Nat.choose n k * Nat.choose (n - k) k) -
  (restricted_days * Nat.choose (n - 1) k * Nat.choose (n - k - 1) k) +
  (Nat.choose (n - 2) k * Nat.choose (n - k - 2) k)

theorem holiday_duty_arrangements :
  arrangement_count staff_count days_count staff_per_day 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_holiday_duty_arrangements_l2962_296229


namespace NUMINAMATH_CALUDE_five_digit_numbers_without_specific_digits_l2962_296291

/-- The number of digits allowed in each place (excluding the first place) -/
def allowed_digits : ℕ := 8

/-- The number of digits allowed in the first place -/
def first_place_digits : ℕ := 7

/-- The total number of places in the number -/
def total_places : ℕ := 5

/-- The expected total count of valid numbers -/
def expected_total : ℕ := 28672

theorem five_digit_numbers_without_specific_digits (d : ℕ) (h : d ≠ 7) :
  first_place_digits * (allowed_digits ^ (total_places - 1)) = expected_total :=
sorry

end NUMINAMATH_CALUDE_five_digit_numbers_without_specific_digits_l2962_296291


namespace NUMINAMATH_CALUDE_max_value_of_f_l2962_296284

-- Define the function f
def f (x : ℝ) : ℝ := x * (6 - 2*x)^2

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 3 ∧ 
  (∀ x, x ∈ Set.Icc 0 3 → f x ≤ f c) ∧
  f c = 16 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2962_296284


namespace NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_l2962_296281

theorem quadratic_has_two_distinct_roots (a : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^2 - (2*a - 1)*x + a^2 - a
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_two_distinct_roots_l2962_296281


namespace NUMINAMATH_CALUDE_symmetry_of_abs_f_shifted_l2962_296251

-- Define a function f with domain ℝ
variable (f : ℝ → ℝ)

-- Define the property of |f(x)| being an even function
def abs_f_is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |f x| = |f (-x)|

-- State the theorem
theorem symmetry_of_abs_f_shifted (h : abs_f_is_even f) :
  ∀ y : ℝ, |f ((1 - y) - 1)| = |f ((1 + y) - 1)| :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_abs_f_shifted_l2962_296251


namespace NUMINAMATH_CALUDE_probability_through_D_l2962_296213

/-- Represents a point on the grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points --/
def numPaths (start finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of choosing a specific path --/
def pathProbability (start finish : Point) : ℚ :=
  (1 / 2) ^ (finish.x - start.x + finish.y - start.y)

theorem probability_through_D (A D B : Point)
  (hA : A = ⟨0, 0⟩)
  (hD : D = ⟨3, 1⟩)
  (hB : B = ⟨6, 3⟩) :
  (numPaths A D * numPaths D B : ℚ) * pathProbability A B / numPaths A B = 20 / 63 :=
sorry

end NUMINAMATH_CALUDE_probability_through_D_l2962_296213


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_fifth_l2962_296246

theorem reciprocal_of_negative_one_fifth :
  ((-1 : ℚ) / 5)⁻¹ = -5 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_fifth_l2962_296246


namespace NUMINAMATH_CALUDE_correlation_relationships_l2962_296242

-- Define the type for relationships
inductive Relationship
  | AppleProductionClimate
  | StudentID
  | TreeDiameterHeight
  | PointCoordinates

-- Define a predicate for correlation relationships
def IsCorrelation (r : Relationship) : Prop :=
  match r with
  | Relationship.AppleProductionClimate => true
  | Relationship.TreeDiameterHeight => true
  | _ => false

-- Theorem statement
theorem correlation_relationships :
  (∀ r : Relationship, IsCorrelation r ↔ 
    (r = Relationship.AppleProductionClimate ∨ 
     r = Relationship.TreeDiameterHeight)) := by
  sorry

end NUMINAMATH_CALUDE_correlation_relationships_l2962_296242


namespace NUMINAMATH_CALUDE_triangle_altitude_sum_perfect_square_l2962_296287

theorem triangle_altitude_sum_perfect_square (x y z : ℤ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  (∃ (h_x h_y h_z : ℝ), 
    h_x > 0 ∧ h_y > 0 ∧ h_z > 0 ∧
    (h_x = h_y + h_z ∨ h_y = h_x + h_z ∨ h_z = h_x + h_y) ∧
    x * h_x = y * h_y ∧ y * h_y = z * h_z) →
  ∃ (n : ℤ), x^2 + y^2 + z^2 = n^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_sum_perfect_square_l2962_296287


namespace NUMINAMATH_CALUDE_largest_divisible_n_l2962_296296

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ (n + 11) ∣ (n^4 + 119) ∧ 
  ∀ (m : ℕ), m > n → ¬((m + 11) ∣ (m^4 + 119)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l2962_296296


namespace NUMINAMATH_CALUDE_problem_statement_l2962_296225

theorem problem_statement (p : ℝ) (h : 126 * 3^8 = p) : 126 * 3^6 = (1/9) * p := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2962_296225


namespace NUMINAMATH_CALUDE_grid_solution_l2962_296288

/-- Represents the possible values in the grid -/
inductive GridValue
  | Two
  | Zero
  | One
  | Five
  | Blank

/-- Represents a 5x5 grid -/
def Grid := Fin 5 → Fin 5 → GridValue

/-- Check if a grid satisfies the row and column constraints -/
def isValidGrid (g : Grid) : Prop :=
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.Two) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.Zero) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.One) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.Five) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.Two) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.Zero) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.One) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.Five)

/-- Check if the diagonal constraint is satisfied -/
def validDiagonal (g : Grid) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i i ≠ g j j

/-- The main theorem stating the solution -/
theorem grid_solution (g : Grid) 
  (hvalid : isValidGrid g) 
  (hdiag : validDiagonal g) : 
  g 4 0 = GridValue.One ∧ 
  g 4 1 = GridValue.Five ∧ 
  g 4 2 = GridValue.Blank ∧ 
  g 4 3 = GridValue.Zero ∧ 
  g 4 4 = GridValue.Two :=
sorry

end NUMINAMATH_CALUDE_grid_solution_l2962_296288


namespace NUMINAMATH_CALUDE_average_of_numbers_l2962_296200

def numbers : List ℝ := [2, 3, 4, 7, 9]

theorem average_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l2962_296200


namespace NUMINAMATH_CALUDE_min_product_of_three_numbers_l2962_296233

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 1/36 :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_three_numbers_l2962_296233


namespace NUMINAMATH_CALUDE_train_length_calculation_l2962_296285

/-- Calculates the length of a train given the speeds of a jogger and the train,
    the initial distance between them, and the time taken for the train to pass the jogger. -/
def train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  (train_speed - jogger_speed) * passing_time - initial_distance

/-- Theorem stating that under the given conditions, the length of the train is 120 meters. -/
theorem train_length_calculation :
  let jogger_speed : ℝ := 9 * (1000 / 3600)  -- 9 kmph in m/s
  let train_speed : ℝ := 45 * (1000 / 3600)  -- 45 kmph in m/s
  let initial_distance : ℝ := 240  -- meters
  let passing_time : ℝ := 36  -- seconds
  train_length jogger_speed train_speed initial_distance passing_time = 120 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l2962_296285


namespace NUMINAMATH_CALUDE_circus_crowns_l2962_296230

theorem circus_crowns (total_feathers : ℕ) (feathers_per_crown : ℕ) (h1 : total_feathers = 6538) (h2 : feathers_per_crown = 7) :
  total_feathers / feathers_per_crown = 934 := by
sorry

end NUMINAMATH_CALUDE_circus_crowns_l2962_296230
