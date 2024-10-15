import Mathlib

namespace NUMINAMATH_CALUDE_gcd_equation_solutions_l4041_404177

theorem gcd_equation_solutions (x y d : ℕ) :
  d = Nat.gcd x y →
  d + x * y / d = x + y →
  (∃ k : ℕ, (x = d ∧ y = d * k) ∨ (x = d * k ∧ y = d)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_equation_solutions_l4041_404177


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l4041_404140

/-- Given a geometric sequence where the first term is 3 and the second term is 6,
    prove that the seventh term is 192. -/
theorem seventh_term_of_geometric_sequence :
  ∀ (a : ℕ → ℝ), 
    a 1 = 3 →
    a 2 = 6 →
    (∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) →
    a 7 = 192 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l4041_404140


namespace NUMINAMATH_CALUDE_sugar_amount_l4041_404160

/-- Represents the amounts of ingredients in a bakery storage room. -/
structure BakeryStorage where
  sugar : ℝ
  flour : ℝ
  bakingSoda : ℝ

/-- Checks if the given storage satisfies the bakery's ratios. -/
def satisfiesRatios (storage : BakeryStorage) : Prop :=
  storage.sugar / storage.flour = 5 / 4 ∧
  storage.flour / storage.bakingSoda = 10 / 1

/-- Checks if adding 60 pounds of baking soda changes the ratio as specified. -/
def satisfiesNewRatio (storage : BakeryStorage) : Prop :=
  storage.flour / (storage.bakingSoda + 60) = 8 / 1

/-- Theorem: Given the conditions, the amount of sugar in the storage is 3000 pounds. -/
theorem sugar_amount (storage : BakeryStorage) 
  (h1 : satisfiesRatios storage) 
  (h2 : satisfiesNewRatio storage) : 
  storage.sugar = 3000 := by
sorry

end NUMINAMATH_CALUDE_sugar_amount_l4041_404160


namespace NUMINAMATH_CALUDE_complex_subtraction_zero_implies_equality_l4041_404198

theorem complex_subtraction_zero_implies_equality (a b : ℂ) : a - b = 0 → a = b := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_zero_implies_equality_l4041_404198


namespace NUMINAMATH_CALUDE_fifth_triple_is_pythagorean_l4041_404119

/-- A Pythagorean triple is a tuple of three positive integers (a, b, c) such that a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The fifth group in the sequence of Pythagorean triples -/
def fifth_pythagorean_triple : ℕ × ℕ × ℕ := (11, 60, 61)

theorem fifth_triple_is_pythagorean :
  let (a, b, c) := fifth_pythagorean_triple
  is_pythagorean_triple a b c :=
by sorry

end NUMINAMATH_CALUDE_fifth_triple_is_pythagorean_l4041_404119


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l4041_404195

-- Define the number of available toppings
def n : ℕ := 8

-- Define the number of toppings to choose
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem pizza_toppings_combinations : combination n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l4041_404195


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4041_404117

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

/-- The proposition to be proved -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (d : ℝ) (h_arith : arithmetic_sequence a d) (h_nonzero : ∃ n, a n ≠ 0)
  (h_eq : 2 * a 4 - (a 7)^2 + 2 * a 10 = 0) :
  a 7 = 4 * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4041_404117


namespace NUMINAMATH_CALUDE_other_sales_percentage_l4041_404176

/-- The percentage of sales for notebooks -/
def notebook_sales : ℝ := 42

/-- The percentage of sales for markers -/
def marker_sales : ℝ := 26

/-- The total percentage of sales -/
def total_sales : ℝ := 100

/-- The percentage of sales that were not notebooks or markers -/
def other_sales : ℝ := total_sales - (notebook_sales + marker_sales)

theorem other_sales_percentage : other_sales = 32 := by
  sorry

end NUMINAMATH_CALUDE_other_sales_percentage_l4041_404176


namespace NUMINAMATH_CALUDE_teacher_assignment_count_l4041_404179

/-- The number of ways to assign four teachers to three classes --/
def total_assignments : ℕ := 36

/-- The number of ways to assign teachers A and B to the same class --/
def ab_same_class : ℕ := 6

/-- The number of ways to assign four teachers to three classes with A and B in different classes --/
def valid_assignments : ℕ := total_assignments - ab_same_class

theorem teacher_assignment_count :
  valid_assignments = 30 :=
sorry

end NUMINAMATH_CALUDE_teacher_assignment_count_l4041_404179


namespace NUMINAMATH_CALUDE_equation_solutions_l4041_404184

theorem equation_solutions :
  (∀ x : ℝ, 6*x - 7 = 4*x - 5 ↔ x = 1) ∧
  (∀ x : ℝ, 5*(x + 8) - 5 = 6*(2*x - 7) ↔ x = 11) ∧
  (∀ x : ℝ, x - (x - 1)/2 = 2 - (x + 2)/5 ↔ x = 11/7) ∧
  (∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4041_404184


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l4041_404192

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  length : ℝ
  width : ℝ
  height : ℝ
  radius : ℝ
  length_eq_twice_height : length = 2 * height
  surface_area_eq_288 : 2 * (length * width + width * height + length * height) = 288
  edge_sum_eq_96 : 4 * (length + width + height) = 96
  inscribed_in_sphere : (2 * radius) ^ 2 = length ^ 2 + width ^ 2 + height ^ 2

/-- The radius of the sphere containing the inscribed box is 4√5 -/
theorem inscribed_box_radius (box : InscribedBox) : box.radius = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l4041_404192


namespace NUMINAMATH_CALUDE_nunzio_pizza_consumption_l4041_404144

/-- Represents the number of pieces in a whole pizza -/
def pieces_per_pizza : ℕ := 8

/-- Represents the number of pizzas Nunzio eats in the given period -/
def total_pizzas : ℕ := 27

/-- Represents the number of days in the given period -/
def total_days : ℕ := 72

/-- Calculates the number of pizza pieces Nunzio eats per day -/
def pieces_per_day : ℕ := (total_pizzas * pieces_per_pizza) / total_days

/-- Theorem stating that Nunzio eats 3 pieces of pizza per day -/
theorem nunzio_pizza_consumption : pieces_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_nunzio_pizza_consumption_l4041_404144


namespace NUMINAMATH_CALUDE_max_twin_prime_sum_200_l4041_404171

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_twin_prime (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ q - p = 2

def max_twin_prime_sum : ℕ := 396

theorem max_twin_prime_sum_200 :
  ∀ p q : ℕ,
  p ≤ 200 → q ≤ 200 →
  is_twin_prime p q →
  p + q ≤ max_twin_prime_sum :=
sorry

end NUMINAMATH_CALUDE_max_twin_prime_sum_200_l4041_404171


namespace NUMINAMATH_CALUDE_sum_of_exterior_angles_constant_l4041_404141

/-- A convex polygon with n sides, where n ≥ 3 -/
structure ConvexPolygon where
  n : ℕ
  sides_ge_three : n ≥ 3

/-- The sum of exterior angles of a convex polygon -/
def sum_of_exterior_angles (p : ConvexPolygon) : ℝ := sorry

/-- Theorem: The sum of exterior angles of any convex polygon is 360° -/
theorem sum_of_exterior_angles_constant (p : ConvexPolygon) :
  sum_of_exterior_angles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_of_exterior_angles_constant_l4041_404141


namespace NUMINAMATH_CALUDE_line_passes_through_intersection_and_perpendicular_l4041_404135

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line3 (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0
def line4 (x y : ℝ) : Prop := 2 * x + 3 * y - 2 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem line_passes_through_intersection_and_perpendicular :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    line4 x y ∧
    perpendicular 
      ((3 : ℝ) / 2) -- slope of line3
      (-(2 : ℝ) / 3) -- slope of line4
  := by sorry

end NUMINAMATH_CALUDE_line_passes_through_intersection_and_perpendicular_l4041_404135


namespace NUMINAMATH_CALUDE_circular_board_holes_l4041_404193

/-- The number of holes on the circular board -/
def n : ℕ := 91

/-- Proposition: The number of holes on the circular board satisfies all conditions -/
theorem circular_board_holes :
  n < 100 ∧
  ∃ k : ℕ, k > 0 ∧ 2 * k ≡ 1 [ZMOD n] ∧
  ∃ m : ℕ, m > 0 ∧ 4 * m ≡ 2 * k [ZMOD n] ∧
  6 ≡ 0 [ZMOD n] :=
by sorry

end NUMINAMATH_CALUDE_circular_board_holes_l4041_404193


namespace NUMINAMATH_CALUDE_number_problem_l4041_404196

theorem number_problem (x : ℝ) : (x / 6) * 12 = 9 → x = 4.5 := by sorry

end NUMINAMATH_CALUDE_number_problem_l4041_404196


namespace NUMINAMATH_CALUDE_q_gt_one_not_sufficient_nor_necessary_l4041_404112

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- Theorem: "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem q_gt_one_not_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  (∃ a q, GeometricSequence a q ∧ q > 1 ∧ ¬IncreasingSequence a) ∧
  (∃ a q, GeometricSequence a q ∧ IncreasingSequence a ∧ ¬(q > 1)) :=
sorry

end NUMINAMATH_CALUDE_q_gt_one_not_sufficient_nor_necessary_l4041_404112


namespace NUMINAMATH_CALUDE_absolute_value_equation_l4041_404147

theorem absolute_value_equation (x : ℝ) : |x + 3| = |x - 5| → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l4041_404147


namespace NUMINAMATH_CALUDE_inner_rectangle_length_l4041_404118

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def rectangleArea (dim : RectDimensions) : ℝ :=
  dim.length * dim.width

/-- Represents the three regions of the rug -/
structure RugRegions where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Theorem stating that the length of the inner rectangle is 4 feet -/
theorem inner_rectangle_length (rug : RugRegions) : rug.inner.length = 4 :=
  by
  -- Assuming the following conditions:
  have inner_width : rug.inner.width = 2 := by sorry
  have middle_surround : rug.middle.length = rug.inner.length + 4 ∧ 
                         rug.middle.width = rug.inner.width + 4 := by sorry
  have outer_surround : rug.outer.length = rug.middle.length + 4 ∧ 
                        rug.outer.width = rug.middle.width + 4 := by sorry
  have areas_arithmetic_progression : 
    (rectangleArea rug.middle - rectangleArea rug.inner) = 
    (rectangleArea rug.outer - rectangleArea rug.middle) := by sorry
  
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_inner_rectangle_length_l4041_404118


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l4041_404137

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 40) :
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 1000 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l4041_404137


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_ten_l4041_404102

/-- Represents the number of classes. -/
def num_classes : ℕ := 3

/-- Represents the total number of spots to be allocated. -/
def total_spots : ℕ := 6

/-- Represents the minimum number of spots each class must receive. -/
def min_spots_per_class : ℕ := 1

/-- A function that calculates the number of ways to allocate spots among classes. -/
def allocation_schemes (n c m : ℕ) : ℕ := sorry

/-- Theorem stating that the number of allocation schemes is 10. -/
theorem allocation_schemes_eq_ten : 
  allocation_schemes total_spots num_classes min_spots_per_class = 10 := by sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_ten_l4041_404102


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l4041_404101

/-- Given a circle ρ = 2cos θ, and a point Q on the extension of a chord OP such that OP/PQ = 2/3,
    prove that the trajectory of Q is a circle with equation ρ = 5cos θ. -/
theorem trajectory_of_Q (θ : Real) (ρ ρ_0 : Real → Real) :
  (∀ θ, ρ_0 θ = 2 * Real.cos θ) →  -- Given circle equation
  (∀ θ, ρ_0 θ / (ρ θ - ρ_0 θ) = 2 / 3) →  -- Ratio condition
  (∀ θ, ρ θ = 5 * Real.cos θ) :=  -- Trajectory equation to prove
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l4041_404101


namespace NUMINAMATH_CALUDE_bakery_sales_percentage_l4041_404152

theorem bakery_sales_percentage (cake_percent cookie_percent : ℝ) 
  (h_cake : cake_percent = 42)
  (h_cookie : cookie_percent = 25) :
  100 - (cake_percent + cookie_percent) = 33 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sales_percentage_l4041_404152


namespace NUMINAMATH_CALUDE_certain_number_problem_l4041_404149

theorem certain_number_problem :
  ∃! x : ℝ,
    (28 + x + 42 + 78 + 104) / 5 = 62 ∧
    (48 + 62 + 98 + 124 + x) / 5 = 78 ∧
    x = 58 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4041_404149


namespace NUMINAMATH_CALUDE_distance_point_to_line_l4041_404190

/-- The distance from a point to a line in 3D space --/
def distancePointToLine (p : ℝ × ℝ × ℝ) (l1 l2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_point_to_line :
  let p := (2, -2, 3)
  let l1 := (1, 3, -1)
  let l2 := (0, 0, 2)
  distancePointToLine p l1 l2 = Real.sqrt 2750 / 19 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l4041_404190


namespace NUMINAMATH_CALUDE_giraffe_count_prove_giraffe_count_l4041_404182

/-- The number of giraffes at a zoo, given certain conditions. -/
theorem giraffe_count : ℕ → ℕ → Prop :=
  fun (giraffes : ℕ) (other_animals : ℕ) =>
    giraffes = 3 * other_animals ∧
    giraffes = other_animals + 290 →
    giraffes = 435

/-- Proof of the giraffe count theorem. -/
theorem prove_giraffe_count : ∃ (g o : ℕ), giraffe_count g o :=
  sorry

end NUMINAMATH_CALUDE_giraffe_count_prove_giraffe_count_l4041_404182


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4041_404132

/-- Two-dimensional vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Parallelism of two vectors -/
def parallel (v w : Vec2D) : Prop :=
  ∃ (μ : ℝ), μ ≠ 0 ∧ v.x = μ * w.x ∧ v.y = μ * w.y

theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : Vec2D := ⟨x, 1⟩
  let b : Vec2D := ⟨3, 6⟩
  parallel b a → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4041_404132


namespace NUMINAMATH_CALUDE_tim_total_score_l4041_404164

/-- The score for a single line in the game -/
def single_line_score : ℕ := 1000

/-- The score multiplier for a tetris -/
def tetris_multiplier : ℕ := 8

/-- The number of single lines Tim scored -/
def tim_singles : ℕ := 6

/-- The number of tetrises Tim scored -/
def tim_tetrises : ℕ := 4

/-- Theorem: Tim's total score is 38000 points -/
theorem tim_total_score : 
  tim_singles * single_line_score + tim_tetrises * (tetris_multiplier * single_line_score) = 38000 := by
  sorry

end NUMINAMATH_CALUDE_tim_total_score_l4041_404164


namespace NUMINAMATH_CALUDE_office_work_distribution_l4041_404159

theorem office_work_distribution (P : ℕ) : P > 0 → (6 / 7 : ℚ) * P * (6 / 5 : ℚ) = P → P ≥ 35 := by
  sorry

end NUMINAMATH_CALUDE_office_work_distribution_l4041_404159


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l4041_404188

theorem product_of_repeating_decimal_and_eight :
  let s : ℚ := 456 / 999
  8 * s = 1216 / 333 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l4041_404188


namespace NUMINAMATH_CALUDE_students_with_B_in_donovans_class_l4041_404104

theorem students_with_B_in_donovans_class 
  (christopher_total : ℕ) 
  (christopher_B : ℕ) 
  (donovan_total : ℕ) 
  (h1 : christopher_total = 20) 
  (h2 : christopher_B = 12) 
  (h3 : donovan_total = 30) 
  (h4 : (christopher_B : ℚ) / christopher_total = (donovan_B : ℚ) / donovan_total) :
  donovan_B = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_students_with_B_in_donovans_class_l4041_404104


namespace NUMINAMATH_CALUDE_aluminum_decoration_problem_l4041_404194

def available_lengths : List ℕ := [3, 6, 9, 12, 15, 19, 21, 30]

def is_valid_combination (combination : List ℕ) : Prop :=
  combination.all (· ∈ available_lengths) ∧ combination.sum = 50

theorem aluminum_decoration_problem :
  ∀ combination : List ℕ,
    is_valid_combination combination ↔
      combination = [19, 19, 12] ∨ combination = [19, 19] :=
by sorry

end NUMINAMATH_CALUDE_aluminum_decoration_problem_l4041_404194


namespace NUMINAMATH_CALUDE_triangles_count_is_120_l4041_404178

/-- Represents the configuration of points on two line segments -/
structure PointConfiguration :=
  (total_points : ℕ)
  (points_on_segment1 : ℕ)
  (points_on_segment2 : ℕ)
  (end_point : ℕ)

/-- Calculates the number of triangles that can be formed with the given point configuration -/
def count_triangles (config : PointConfiguration) : ℕ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the specific configuration results in 120 triangles -/
theorem triangles_count_is_120 :
  let config := PointConfiguration.mk 11 6 4 1
  count_triangles config = 120 :=
by sorry

end NUMINAMATH_CALUDE_triangles_count_is_120_l4041_404178


namespace NUMINAMATH_CALUDE_tangent_line_at_one_condition_holds_iff_l4041_404163

-- Define the function f(x) = 2x³ - 3ax²
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * a * x^2

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * a * x

theorem tangent_line_at_one (a : ℝ) (h : a = 2) :
  ∃ m b : ℝ, m = -6 ∧ b = 2 ∧
  ∀ x : ℝ, f a x + (f_prime a 1) * (x - 1) = m * x + b :=
sorry

theorem condition_holds_iff (a : ℝ) :
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc 0 2 →
    ∃ x₂ : ℝ, x₂ ∈ Set.Icc 0 1 ∧ f a x₁ ≥ f_prime a x₂) ↔
  a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_condition_holds_iff_l4041_404163


namespace NUMINAMATH_CALUDE_riza_son_age_l4041_404154

/-- Represents the age difference between Riza and her son -/
def age_difference : ℕ := 25

/-- Represents the sum of Riza's and her son's current ages -/
def current_age_sum : ℕ := 105

/-- Represents Riza's son's current age -/
def son_age : ℕ := (current_age_sum - age_difference) / 2

theorem riza_son_age : son_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_riza_son_age_l4041_404154


namespace NUMINAMATH_CALUDE_twentieth_number_is_381_l4041_404124

/-- The last number of the nth row in the sequence -/
def last_number (n : ℕ) : ℕ := n^2

/-- The 20th number in the 20th row of the sequence -/
def twentieth_number : ℕ := last_number 19 + 20

theorem twentieth_number_is_381 : twentieth_number = 381 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_number_is_381_l4041_404124


namespace NUMINAMATH_CALUDE_sufficient_necessary_but_not_sufficient_l4041_404169

-- Define propositions p and q
variable (p q : Prop)

-- Define what it means for p to be a sufficient condition for q
def sufficient (p q : Prop) : Prop := p → q

-- Define what it means for p to be a necessary and sufficient condition for q
def necessary_and_sufficient (p q : Prop) : Prop := p ↔ q

-- Theorem stating that "p is a sufficient condition for q" is a necessary but not sufficient condition for "p is a necessary and sufficient condition for q"
theorem sufficient_necessary_but_not_sufficient :
  (∀ p q, necessary_and_sufficient p q → sufficient p q) ∧
  ¬(∀ p q, sufficient p q → necessary_and_sufficient p q) :=
sorry

end NUMINAMATH_CALUDE_sufficient_necessary_but_not_sufficient_l4041_404169


namespace NUMINAMATH_CALUDE_temperature_altitude_relationship_l4041_404138

/-- Given that the ground temperature is 20°C and the temperature decreases by 6°C
    for every 1000m increase in altitude, prove that the functional relationship
    between temperature t(°C) and altitude h(m) is t = -0.006h + 20. -/
theorem temperature_altitude_relationship (h : ℝ) :
  let ground_temp : ℝ := 20
  let temp_decrease_per_km : ℝ := 6
  let altitude_increase : ℝ := 1000
  let t : ℝ → ℝ := fun h => -((temp_decrease_per_km / altitude_increase) * h) + ground_temp
  t h = -0.006 * h + 20 := by
  sorry

end NUMINAMATH_CALUDE_temperature_altitude_relationship_l4041_404138


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l4041_404185

def g (x : ℝ) : ℝ := (x - 3)^2 - 7

theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l4041_404185


namespace NUMINAMATH_CALUDE_pie_slices_sold_today_l4041_404142

theorem pie_slices_sold_today (total : ℕ) (yesterday : ℕ) (today : ℕ) 
  (h1 : total = 7) 
  (h2 : yesterday = 5) 
  (h3 : total = yesterday + today) : 
  today = 2 := by
  sorry

end NUMINAMATH_CALUDE_pie_slices_sold_today_l4041_404142


namespace NUMINAMATH_CALUDE_max_min_value_l4041_404186

def f (x y : ℝ) : ℝ := x^3 + (y-4)*x^2 + (y^2-4*y+4)*x + (y^3-4*y^2+4*y)

theorem max_min_value (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) 
  (h1 : f a b = f b c) (h2 : f b c = f c a) : 
  ∃ (m : ℝ), m = 1 ∧ 
  ∀ (x y z : ℝ), x ≠ y → y ≠ z → z ≠ x → f x y = f y z → f y z = f z x →
  min (x^4 - 4*x^3 + 4*x^2) (min (y^4 - 4*y^3 + 4*y^2) (z^4 - 4*z^3 + 4*z^2)) ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_min_value_l4041_404186


namespace NUMINAMATH_CALUDE_inequality_proof_l4041_404162

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b - c)^2 * (b + c) / a + (c - a)^2 * (c + a) / b + (a - b)^2 * (a + b) / c ≥ 
  2 * (a^2 + b^2 + c^2 - a*b - b*c - c*a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4041_404162


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4041_404108

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - 2*x₁ - 8 = 0) ∧ 
  (x₂^2 - 2*x₂ - 8 = 0) ∧ 
  x₁ = 4 ∧ 
  x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4041_404108


namespace NUMINAMATH_CALUDE_bird_photo_combinations_l4041_404155

/-- Represents the number of pairs of birds -/
def num_pairs : ℕ := 5

/-- Calculates the number of ways to photograph birds with alternating genders -/
def photo_combinations (n : ℕ) : ℕ :=
  let female_choices := List.range n
  let male_choices := List.range (n - 1)
  (female_choices.foldl (· * ·) 1) * (male_choices.foldl (· * ·) 1)

/-- Theorem stating the number of ways to photograph the birds -/
theorem bird_photo_combinations :
  photo_combinations num_pairs = 2880 := by
  sorry

end NUMINAMATH_CALUDE_bird_photo_combinations_l4041_404155


namespace NUMINAMATH_CALUDE_largest_value_l4041_404143

theorem largest_value (a b c d e : ℝ) 
  (ha : a = 15372 + 2/3074)
  (hb : b = 15372 - 2/3074)
  (hc : c = 15372 / (2/3074))
  (hd : d = 15372 * (2/3074))
  (he : e = 15372.3074) :
  c > a ∧ c > b ∧ c > d ∧ c > e :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l4041_404143


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l4041_404110

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon :
  num_diagonals 25 = 275 := by sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l4041_404110


namespace NUMINAMATH_CALUDE_min_guaranteed_meeting_distance_l4041_404134

/-- Represents the state of a player on the train -/
structure PlayerState :=
  (position : Real)
  (facing_forward : Bool)
  (at_front : Bool)
  (at_end : Bool)

/-- Represents the game state -/
structure GameState :=
  (alice : PlayerState)
  (bob : PlayerState)
  (total_distance : Real)

/-- Defines the train length -/
def train_length : Real := 1

/-- Theorem stating the minimum guaranteed meeting distance -/
theorem min_guaranteed_meeting_distance :
  ∀ (initial_state : GameState),
  ∃ (strategy : GameState → GameState),
  ∀ (final_state : GameState),
  (final_state.alice.position = final_state.bob.position) →
  (final_state.total_distance ≤ 1.5) :=
sorry

end NUMINAMATH_CALUDE_min_guaranteed_meeting_distance_l4041_404134


namespace NUMINAMATH_CALUDE_probability_6_or_7_heads_in_8_flips_l4041_404153

def n : ℕ := 8  -- number of coin flips

-- Define the probability of getting exactly k heads in n flips
def prob_k_heads (k : ℕ) : ℚ :=
  (n.choose k) / 2^n

-- Define the probability of getting exactly 6 or 7 heads in n flips
def prob_6_or_7_heads : ℚ :=
  prob_k_heads 6 + prob_k_heads 7

-- Theorem statement
theorem probability_6_or_7_heads_in_8_flips :
  prob_6_or_7_heads = 9 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_6_or_7_heads_in_8_flips_l4041_404153


namespace NUMINAMATH_CALUDE_function_rate_comparison_l4041_404150

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - x
def g (x : ℝ) : ℝ := x^2 + x

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := 2*x - 1
def g' (x : ℝ) : ℝ := 2*x + 1

theorem function_rate_comparison :
  (∃ x : ℝ, f' x = 2 * g' x ∧ x = -3/2) ∧
  (¬ ∃ x : ℝ, f' x = g' x) := by
  sorry


end NUMINAMATH_CALUDE_function_rate_comparison_l4041_404150


namespace NUMINAMATH_CALUDE_train_speed_l4041_404139

/-- Proves that a train with given length and time to cross a stationary object has the specified speed -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (speed : ℝ) : 
  train_length = 250 →
  crossing_time = 12.857142857142858 →
  speed = (train_length / 1000) / (crossing_time / 3600) →
  speed = 70 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l4041_404139


namespace NUMINAMATH_CALUDE_collinear_implies_relation_vector_relation_implies_coordinates_l4041_404181

-- Define points A, B, and C
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (3, -1)
def C : ℝ → ℝ → ℝ × ℝ := λ a b => (a, b)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r.1 - p.1 = t * (q.1 - p.1) ∧ r.2 - p.2 = t * (q.2 - p.2)

-- Define vector multiplication
def vec_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Part 1: Collinearity implies a = 2 - b
theorem collinear_implies_relation (a b : ℝ) :
  collinear A B (C a b) → a = 2 - b := by sorry

-- Part 2: AC = 2AB implies C = (5, -3)
theorem vector_relation_implies_coordinates (a b : ℝ) :
  C a b - A = vec_mult 2 (B - A) → C a b = (5, -3) := by sorry

end NUMINAMATH_CALUDE_collinear_implies_relation_vector_relation_implies_coordinates_l4041_404181


namespace NUMINAMATH_CALUDE_survey_result_l4041_404189

/-- Represents the result of a stratified sampling survey -/
structure SurveyResult where
  totalPopulation : ℕ
  sampleSize : ℕ
  physicsInSample : ℕ
  historyInPopulation : ℕ

/-- Checks if the survey result is valid based on the given conditions -/
def isValidSurvey (s : SurveyResult) : Prop :=
  s.totalPopulation = 1500 ∧
  s.sampleSize = 120 ∧
  s.physicsInSample = 80 ∧
  s.sampleSize - s.physicsInSample > 0 ∧
  s.sampleSize < s.totalPopulation

/-- Theorem stating the result of the survey -/
theorem survey_result (s : SurveyResult) (h : isValidSurvey s) :
  s.historyInPopulation = 500 := by
  sorry

#check survey_result

end NUMINAMATH_CALUDE_survey_result_l4041_404189


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l4041_404111

/-- Given three equally spaced parallel lines intersecting a circle and creating
    chords of lengths 42, 36, and 36, prove that the distance between two
    adjacent parallel lines is 2√2006. -/
theorem parallel_lines_distance (r : ℝ) (d : ℝ) : 
  (∃ (x y z : ℝ), x = 42 ∧ y = 36 ∧ z = 36 ∧
   21 * x * 21 + (d/2)^2 * x = 21 * r^2 + 21 * r^2 ∧
   18 * y * 18 + (d/2)^2 * y = 18 * r^2 + 18 * r^2) →
  d = 2 * Real.sqrt 2006 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l4041_404111


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l4041_404103

/-- Given two rectangles of equal area, where one has dimensions 8 by 45 and the other has width 24,
    prove that the length of the second rectangle is 15. -/
theorem equal_area_rectangles (area : ℝ) (length₁ width₁ width₂ : ℝ) 
  (h₁ : area = length₁ * width₁)
  (h₂ : length₁ = 8)
  (h₃ : width₁ = 45)
  (h₄ : width₂ = 24) :
  area / width₂ = 15 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l4041_404103


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l4041_404156

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, 0 < x ∧ x < 2 ↔ ∀ y : ℝ, 0 < y ∧ y < x → f y > f x :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l4041_404156


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l4041_404129

theorem cube_sum_of_roots (p q r : ℝ) : 
  (p^3 - 2*p^2 + p - 3 = 0) → 
  (q^3 - 2*q^2 + q - 3 = 0) → 
  (r^3 - 2*r^2 + r - 3 = 0) → 
  p^3 + q^3 + r^3 = 11 := by sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l4041_404129


namespace NUMINAMATH_CALUDE_total_wrappers_collected_l4041_404170

/-- The total number of wrappers collected by four friends is the sum of their individual collections. -/
theorem total_wrappers_collected
  (andy_wrappers : ℕ)
  (max_wrappers : ℕ)
  (zoe_wrappers : ℕ)
  (mia_wrappers : ℕ)
  (h1 : andy_wrappers = 34)
  (h2 : max_wrappers = 15)
  (h3 : zoe_wrappers = 25)
  (h4 : mia_wrappers = 19) :
  andy_wrappers + max_wrappers + zoe_wrappers + mia_wrappers = 93 := by
  sorry

end NUMINAMATH_CALUDE_total_wrappers_collected_l4041_404170


namespace NUMINAMATH_CALUDE_sara_bouquets_l4041_404125

theorem sara_bouquets (red yellow blue : ℕ) 
  (h_red : red = 42) 
  (h_yellow : yellow = 63) 
  (h_blue : blue = 54) : 
  Nat.gcd red (Nat.gcd yellow blue) = 21 := by
  sorry

end NUMINAMATH_CALUDE_sara_bouquets_l4041_404125


namespace NUMINAMATH_CALUDE_divisors_of_2121_with_units_digit_1_l4041_404199

/-- The number of positive integer divisors of 2121 with a units digit of 1 is 4. -/
theorem divisors_of_2121_with_units_digit_1 : 
  (Finset.filter (fun d => d % 10 = 1) (Nat.divisors 2121)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2121_with_units_digit_1_l4041_404199


namespace NUMINAMATH_CALUDE_shaniqua_haircut_price_l4041_404145

/-- The amount Shaniqua makes for each haircut -/
def haircut_price : ℚ := sorry

/-- The amount Shaniqua makes for each style -/
def style_price : ℚ := 25

/-- The total number of haircuts Shaniqua gave -/
def num_haircuts : ℕ := 8

/-- The total number of styles Shaniqua gave -/
def num_styles : ℕ := 5

/-- The total amount Shaniqua made -/
def total_amount : ℚ := 221

theorem shaniqua_haircut_price : 
  haircut_price * num_haircuts + style_price * num_styles = total_amount ∧ 
  haircut_price = 12 := by sorry

end NUMINAMATH_CALUDE_shaniqua_haircut_price_l4041_404145


namespace NUMINAMATH_CALUDE_fruits_in_good_condition_l4041_404120

theorem fruits_in_good_condition 
  (oranges : ℕ) 
  (bananas : ℕ) 
  (rotten_oranges_percent : ℚ) 
  (rotten_bananas_percent : ℚ) 
  (h1 : oranges = 600) 
  (h2 : bananas = 400) 
  (h3 : rotten_oranges_percent = 15/100) 
  (h4 : rotten_bananas_percent = 5/100) : 
  (oranges + bananas - (oranges * rotten_oranges_percent + bananas * rotten_bananas_percent)) / (oranges + bananas) = 89/100 := by
sorry

end NUMINAMATH_CALUDE_fruits_in_good_condition_l4041_404120


namespace NUMINAMATH_CALUDE_min_value_theorem_l4041_404168

theorem min_value_theorem (a b : ℝ) (h1 : a * b > 0) (h2 : 2 * a + b = 5) :
  (∀ x y : ℝ, x * y > 0 ∧ 2 * x + y = 5 → 
    2 / (a + 1) + 1 / (b + 1) ≤ 2 / (x + 1) + 1 / (y + 1)) ∧
  2 / (a + 1) + 1 / (b + 1) = 9 / 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4041_404168


namespace NUMINAMATH_CALUDE_miriam_pushups_l4041_404113

/-- Calculates the number of push-ups Miriam does on Friday given her schedule for the week. -/
theorem miriam_pushups (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ) 
  (h1 : monday = 5)
  (h2 : tuesday = 7)
  (h3 : wednesday = 2 * tuesday)
  (h4 : thursday = (monday + tuesday + wednesday) / 2)
  : monday + tuesday + wednesday + thursday = 39 := by
  sorry

end NUMINAMATH_CALUDE_miriam_pushups_l4041_404113


namespace NUMINAMATH_CALUDE_root_condition_implies_k_range_l4041_404158

theorem root_condition_implies_k_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
    2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
    |x₁ - 2*n| = k * Real.sqrt x₁ ∧
    |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_k_range_l4041_404158


namespace NUMINAMATH_CALUDE_sugar_salt_price_l4041_404136

/-- Given the price of 2 kg sugar and 5 kg salt, and the price of 1 kg sugar,
    prove the price of 3 kg sugar and 1 kg salt. -/
theorem sugar_salt_price
  (total_price : ℝ)
  (sugar_price : ℝ)
  (h1 : total_price = 5.5)
  (h2 : sugar_price = 1.5)
  (h3 : 2 * sugar_price + 5 * ((total_price - 2 * sugar_price) / 5) = total_price) :
  3 * sugar_price + ((total_price - 2 * sugar_price) / 5) = 5 :=
by sorry

end NUMINAMATH_CALUDE_sugar_salt_price_l4041_404136


namespace NUMINAMATH_CALUDE_positive_integer_power_equality_l4041_404175

theorem positive_integer_power_equality (a b : ℕ+) :
  a ^ b.val = b ^ (a.val ^ 2) ↔ (a, b) = (1, 1) ∨ (a, b) = (2, 16) ∨ (a, b) = (3, 27) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_power_equality_l4041_404175


namespace NUMINAMATH_CALUDE_math_score_calculation_math_score_is_83_l4041_404107

theorem math_score_calculation (average_three : ℝ) (average_decrease : ℝ) : ℝ :=
  let total_three := 3 * average_three
  let new_average := average_three - average_decrease
  let total_four := 4 * new_average
  total_four - total_three

theorem math_score_is_83 :
  math_score_calculation 95 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_math_score_calculation_math_score_is_83_l4041_404107


namespace NUMINAMATH_CALUDE_system_solutions_l4041_404106

/-- The polynomial f(t) = t³ - 4t² - 16t + 60 -/
def f (t : ℤ) : ℤ := t^3 - 4*t^2 - 16*t + 60

/-- The system of equations -/
def system (x y z : ℤ) : Prop :=
  f x = y ∧ f y = z ∧ f z = x

/-- The theorem stating the only integer solutions to the system -/
theorem system_solutions :
  ∀ x y z : ℤ, system x y z ↔ (x = 3 ∧ y = 3 ∧ z = 3) ∨ 
                               (x = 5 ∧ y = 5 ∧ z = 5) ∨ 
                               (x = -4 ∧ y = -4 ∧ z = -4) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l4041_404106


namespace NUMINAMATH_CALUDE_water_to_dean_height_ratio_l4041_404174

-- Define the heights and water depth
def ron_height : ℝ := 14
def height_difference : ℝ := 8
def water_depth : ℝ := 12

-- Define Dean's height
def dean_height : ℝ := ron_height - height_difference

-- Theorem statement
theorem water_to_dean_height_ratio :
  (water_depth / dean_height) = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_to_dean_height_ratio_l4041_404174


namespace NUMINAMATH_CALUDE_complete_square_sum_l4041_404148

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 10*x + 15 = 0 ↔ (x + b)^2 = c) → 
  b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l4041_404148


namespace NUMINAMATH_CALUDE_upstream_speed_is_25_l4041_404127

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ  -- Speed in still water
  downstream : ℝ  -- Speed downstream

/-- Calculates the speed of the man rowing upstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem: Given the conditions, the upstream speed is 25 kmph -/
theorem upstream_speed_is_25 (s : RowingSpeed) 
  (h1 : s.stillWater = 32) 
  (h2 : s.downstream = 39) : 
  upstreamSpeed s = 25 := by
  sorry

#eval upstreamSpeed { stillWater := 32, downstream := 39 }

end NUMINAMATH_CALUDE_upstream_speed_is_25_l4041_404127


namespace NUMINAMATH_CALUDE_ten_men_joined_l4041_404126

/-- Represents the number of men who joined the camp -/
def men_joined : ℕ := sorry

/-- The initial number of men in the camp -/
def initial_men : ℕ := 10

/-- The initial duration of the food supply in days -/
def initial_duration : ℕ := 50

/-- The new duration of the food supply after more men join -/
def new_duration : ℕ := 25

/-- The total amount of food available in man-days -/
def total_food : ℕ := initial_men * initial_duration

/-- Theorem stating that 10 men joined the camp -/
theorem ten_men_joined : men_joined = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_men_joined_l4041_404126


namespace NUMINAMATH_CALUDE_percent_relation_l4041_404109

theorem percent_relation (x y : ℝ) (h : 0.3 * (x - y) = 0.2 * (x + y)) : y = 0.2 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l4041_404109


namespace NUMINAMATH_CALUDE_planes_distance_l4041_404122

/-- The total distance traveled by two planes moving towards each other -/
def total_distance (speed : ℝ) (time : ℝ) : ℝ :=
  2 * speed * time

/-- Theorem: The total distance traveled by two planes moving towards each other
    at 283 miles per hour for 2 hours is 1132 miles. -/
theorem planes_distance :
  total_distance 283 2 = 1132 :=
by sorry

end NUMINAMATH_CALUDE_planes_distance_l4041_404122


namespace NUMINAMATH_CALUDE_ellipse_equation_parabola_equation_l4041_404173

-- Problem 1
theorem ellipse_equation (focal_distance : ℝ) (point : ℝ × ℝ) : 
  focal_distance = 4 ∧ point = (3, -2 * Real.sqrt 6) →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧
    (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/36 + y^2/32 = 1) :=
sorry

-- Problem 2
theorem parabola_equation (hyperbola : ℝ → ℝ → Prop) (directrix : ℝ) :
  (∀ x y : ℝ, hyperbola x y ↔ x^2 - y^2/3 = 1) ∧
  directrix = -1/2 →
  ∀ x y : ℝ, y^2 = 2*x ↔ y^2 = 2*x ∧ x ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_parabola_equation_l4041_404173


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l4041_404121

/-- Polynomial of degree 5 -/
def P (a b c d : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + d

theorem polynomial_value_at_three
  (a b c d : ℝ)
  (h1 : P a b c d 0 = -5)
  (h2 : P a b c d (-3) = 7) :
  P a b c d 3 = -17 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l4041_404121


namespace NUMINAMATH_CALUDE_complex_number_problem_l4041_404131

theorem complex_number_problem (z : ℂ) 
  (h1 : ∃ (r1 : ℝ), z / (1 + z^2) = r1)
  (h2 : ∃ (r2 : ℝ), z^2 / (1 + z) = r2) :
  z = -1/2 + (Complex.I * Real.sqrt 3)/2 ∨ 
  z = -1/2 - (Complex.I * Real.sqrt 3)/2 :=
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l4041_404131


namespace NUMINAMATH_CALUDE_cross_product_of_a_and_b_l4041_404180

def a : ℝ × ℝ × ℝ := (3, 4, -5)
def b : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (u₁, u₂, u₃) := u
  let (v₁, v₂, v₃) := v
  (u₂ * v₃ - u₃ * v₂, u₃ * v₁ - u₁ * v₃, u₁ * v₂ - u₂ * v₁)

theorem cross_product_of_a_and_b :
  cross_product a b = (11, -22, -11) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_of_a_and_b_l4041_404180


namespace NUMINAMATH_CALUDE_even_decreasing_properties_l4041_404130

/-- A function that is even and monotonically decreasing on (0, +∞) -/
def EvenDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x ∧ x < y → f y < f x)

theorem even_decreasing_properties (f : ℝ → ℝ) (hf : EvenDecreasingFunction f) :
  (∃ a : ℝ, f (2 * a) ≥ f (-a)) ∧ 
  (f π ≤ f (-3)) ∧
  (f (-Real.sqrt 3 / 2) < f (4 / 5)) ∧
  (∃ a : ℝ, f (a^2 + 1) ≥ f 1) := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_properties_l4041_404130


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4041_404123

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_properties 
  (a₁ : ℤ) 
  (d : ℤ) 
  (h1 : a₁ = 23)
  (h2 : ∀ n : ℕ, n ≤ 6 → arithmetic_sequence a₁ d n > 0)
  (h3 : arithmetic_sequence a₁ d 7 < 0) :
  (d = -4) ∧ 
  (∃ n : ℕ, sum_arithmetic_sequence a₁ d n = 78 ∧ 
    ∀ m : ℕ, sum_arithmetic_sequence a₁ d m ≤ 78) ∧
  (∃ n : ℕ, n = 12 ∧ sum_arithmetic_sequence a₁ d n > 0 ∧ 
    ∀ m : ℕ, m > 12 → sum_arithmetic_sequence a₁ d m ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4041_404123


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l4041_404172

/-- Given two rectangles of equal area, where one rectangle measures 15 inches by 24 inches
    and the other has a length of 8 inches, prove that the width of the second rectangle
    is 45 inches. -/
theorem equal_area_rectangles_width (area carol_length carol_width jordan_length : ℕ)
    (h1 : area = carol_length * carol_width)
    (h2 : carol_length = 15)
    (h3 : carol_width = 24)
    (h4 : jordan_length = 8) :
    area / jordan_length = 45 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l4041_404172


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l4041_404116

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 171 → n = 19 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l4041_404116


namespace NUMINAMATH_CALUDE_empty_solution_set_l4041_404183

def f (x : ℝ) : ℝ := x^2 + x

theorem empty_solution_set :
  {x : ℝ | f (x - 2) + f x < 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_l4041_404183


namespace NUMINAMATH_CALUDE_range_of_a_l4041_404157

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1) 1 → a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 4*x + a^2 > 0

-- Define the range of a
def range_a : Set ℝ := Set.Icc (-2) (-1) ∪ Set.Ioo 2 6

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ range_a :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4041_404157


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l4041_404114

/-- Represents a repeating decimal with a two-digit repeating part -/
def RepeatingDecimal (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

theorem repeating_decimal_ratio :
  (RepeatingDecimal 5 4) / (RepeatingDecimal 1 8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l4041_404114


namespace NUMINAMATH_CALUDE_sector_cone_properties_l4041_404191

/-- Represents a cone formed from a sector of a circular sheet -/
structure SectorCone where
  sheet_radius : ℝ
  num_sectors : ℕ

/-- Calculate the height of a cone formed from a sector of a circular sheet -/
def cone_height (c : SectorCone) : ℝ :=
  sorry

/-- Calculate the volume of a cone formed from a sector of a circular sheet -/
def cone_volume (c : SectorCone) : ℝ :=
  sorry

theorem sector_cone_properties (c : SectorCone) 
  (h_radius : c.sheet_radius = 12)
  (h_sectors : c.num_sectors = 4) :
  cone_height c = 3 * Real.sqrt 15 ∧ 
  cone_volume c = 9 * Real.pi * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_sector_cone_properties_l4041_404191


namespace NUMINAMATH_CALUDE_product_over_sum_minus_four_l4041_404146

theorem product_over_sum_minus_four :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) /
  (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 - 4) = 362880 / 41 := by
  sorry

end NUMINAMATH_CALUDE_product_over_sum_minus_four_l4041_404146


namespace NUMINAMATH_CALUDE_triangular_front_view_solids_l4041_404167

-- Define the types of solids
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

-- Define a predicate for solids that can have a triangular front view
def hasTriangularFrontView (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True
  | Solid.Cone => True
  | _ => False

-- Theorem stating which solids can have a triangular front view
theorem triangular_front_view_solids :
  ∀ s : Solid, hasTriangularFrontView s ↔
    (s = Solid.TriangularPyramid ∨
     s = Solid.SquarePyramid ∨
     s = Solid.TriangularPrism ∨
     s = Solid.Cone) :=
by sorry

end NUMINAMATH_CALUDE_triangular_front_view_solids_l4041_404167


namespace NUMINAMATH_CALUDE_max_value_sqrt_inequality_l4041_404166

theorem max_value_sqrt_inequality (x : ℝ) (h1 : 3 ≤ x) (h2 : x ≤ 6) :
  ∃ (k : ℝ), (∀ y : ℝ, y ≥ k → ∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ y) ∧
  (∀ z : ℝ, z > k → ¬∃ x : ℝ, 3 ≤ x ∧ x ≤ 6 ∧ Real.sqrt (x - 3) + Real.sqrt (6 - x) ≥ z) ∧
  k = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_inequality_l4041_404166


namespace NUMINAMATH_CALUDE_total_silverware_l4041_404197

/-- The number of types of silverware --/
def num_types : ℕ := 4

/-- The initial number of each type for personal use --/
def initial_personal : ℕ := 5

/-- The number of extra pieces of each type for guests --/
def extra_for_guests : ℕ := 10

/-- The reduction in the number of spoons --/
def spoon_reduction : ℕ := 4

/-- The reduction in the number of butter knives --/
def butter_knife_reduction : ℕ := 4

/-- The reduction in the number of steak knives --/
def steak_knife_reduction : ℕ := 5

/-- The reduction in the number of forks --/
def fork_reduction : ℕ := 3

/-- The theorem stating the total number of silverware pieces Stephanie will buy --/
theorem total_silverware : 
  (initial_personal + extra_for_guests - spoon_reduction) +
  (initial_personal + extra_for_guests - butter_knife_reduction) +
  (initial_personal + extra_for_guests - steak_knife_reduction) +
  (initial_personal + extra_for_guests - fork_reduction) = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_silverware_l4041_404197


namespace NUMINAMATH_CALUDE_fraction_numerator_greater_than_denominator_l4041_404115

theorem fraction_numerator_greater_than_denominator
  (x : ℝ)
  (h1 : -1 ≤ x)
  (h2 : x ≤ 3)
  : 4 * x + 2 > 8 - 3 * x ↔ 6 / 7 < x ∧ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_numerator_greater_than_denominator_l4041_404115


namespace NUMINAMATH_CALUDE_ratio_proof_l4041_404100

theorem ratio_proof (a b c k : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c = 56) 
  (h4 : c - a = 32) (h5 : a = 3 * k) (h6 : b = 5 * k) : 
  c / b = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proof_l4041_404100


namespace NUMINAMATH_CALUDE_sum_of_valid_numbers_mod_1000_l4041_404105

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0) ∧
  n % 99 = 0 ∧
  150 % (n / 100) = 0 ∧
  168 % (n % 100) = 0

def sum_of_valid_numbers : ℕ := sorry

theorem sum_of_valid_numbers_mod_1000 :
  sum_of_valid_numbers % 1000 = 108 := by sorry

end NUMINAMATH_CALUDE_sum_of_valid_numbers_mod_1000_l4041_404105


namespace NUMINAMATH_CALUDE_H_range_l4041_404151

def H (x : ℝ) : ℝ := |x + 2| - |x - 4| + 3

theorem H_range : 
  (∀ x, 5 ≤ H x ∧ H x ≤ 9) ∧ 
  (∃ x, H x = 9) ∧
  (∀ ε > 0, ∃ x, H x < 5 + ε) :=
sorry

end NUMINAMATH_CALUDE_H_range_l4041_404151


namespace NUMINAMATH_CALUDE_equation_holds_iff_b_equals_c_l4041_404161

theorem equation_holds_iff_b_equals_c (a b c : ℕ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_less_than_10 : a < 10 ∧ b < 10 ∧ c < 10) :
  (10 * a + b + 1) * (10 * a + c) = 100 * a^2 + 100 * a + b + c ↔ b = c :=
sorry

end NUMINAMATH_CALUDE_equation_holds_iff_b_equals_c_l4041_404161


namespace NUMINAMATH_CALUDE_light_flash_duration_l4041_404133

theorem light_flash_duration (flash_interval : ℕ) (num_flashes : ℕ) (seconds_per_hour : ℕ) : 
  flash_interval = 12 →
  num_flashes = 300 →
  seconds_per_hour = 3600 →
  (flash_interval * num_flashes) / seconds_per_hour = 1 := by
  sorry

end NUMINAMATH_CALUDE_light_flash_duration_l4041_404133


namespace NUMINAMATH_CALUDE_rotate_point_D_l4041_404128

def rotate90Clockwise (x y : ℝ) : ℝ × ℝ := (y, -x)

theorem rotate_point_D : 
  let D : ℝ × ℝ := (-3, 2)
  rotate90Clockwise D.1 D.2 = (2, -3) := by sorry

end NUMINAMATH_CALUDE_rotate_point_D_l4041_404128


namespace NUMINAMATH_CALUDE_minimal_fraction_difference_l4041_404165

theorem minimal_fraction_difference (p q : ℕ+) : 
  (3 : ℚ) / 5 < p / q ∧ p / q < (2 : ℚ) / 3 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < p' / q' ∧ p' / q' < (2 : ℚ) / 3 → q' ≥ q) →
  q - p = 11 := by
  sorry

end NUMINAMATH_CALUDE_minimal_fraction_difference_l4041_404165


namespace NUMINAMATH_CALUDE_seventeen_sided_polygon_diagonals_l4041_404187

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 17 sides has 119 diagonals -/
theorem seventeen_sided_polygon_diagonals :
  num_diagonals 17 = 119 := by sorry

end NUMINAMATH_CALUDE_seventeen_sided_polygon_diagonals_l4041_404187
