import Mathlib

namespace NUMINAMATH_CALUDE_find_d_l2514_251466

theorem find_d : ∃ d : ℚ, 
  (∃ n : ℤ, n = ⌊d⌋ ∧ 3 * n^2 + 19 * n - 28 = 0) ∧
  (4 * (d - ↑⌊d⌋)^2 - 11 * (d - ↑⌊d⌋) + 3 = 0) ∧
  (0 ≤ d - ↑⌊d⌋ ∧ d - ↑⌊d⌋ < 1) ∧
  d = -29/4 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l2514_251466


namespace NUMINAMATH_CALUDE_coplanar_vectors_lambda_l2514_251426

/-- Given three vectors a, b, and c in ℝ³, if they are coplanar and have specific coordinates,
    then the third coordinate of c equals 9. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) : 
  a = (2, -1, 3) → b = (-1, 4, -2) → c.1 = 7 → c.2.1 = 7 →
  (∃ (m n : ℝ), c = m • a + n • b) →
  c.2.2 = 9 := by
sorry

end NUMINAMATH_CALUDE_coplanar_vectors_lambda_l2514_251426


namespace NUMINAMATH_CALUDE_quadratic_roots_conditions_l2514_251419

/-- The quadratic equation (2m+1)x^2 + 4mx + 2m-3 = 0 has:
    1. Two distinct real roots iff m ∈ (-3/4, -1/2) ∪ (-1/2, ∞)
    2. Two equal real roots iff m = -3/4
    3. No real roots iff m ∈ (-∞, -3/4) -/
theorem quadratic_roots_conditions (m : ℝ) :
  let a := 2*m + 1
  let b := 4*m
  let c := 2*m - 3
  let discriminant := b^2 - 4*a*c
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0) ↔ 
    (m > -3/4 ∧ m ≠ -1/2) ∧
  (∃ x : ℝ, a*x^2 + b*x + c = 0 ∧ discriminant = 0) ↔ 
    (m = -3/4) ∧
  (∀ x : ℝ, a*x^2 + b*x + c ≠ 0) ↔ 
    (m < -3/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_conditions_l2514_251419


namespace NUMINAMATH_CALUDE_megan_candy_count_l2514_251445

theorem megan_candy_count (mary_initial : ℕ) (megan : ℕ) : 
  mary_initial = 3 * megan →
  mary_initial + 10 = 25 →
  megan = 5 := by
sorry

end NUMINAMATH_CALUDE_megan_candy_count_l2514_251445


namespace NUMINAMATH_CALUDE_correct_answer_l2514_251495

theorem correct_answer : ∃ x : ℤ, (x + 3 = 45) ∧ (x - 3 = 39) := by
  sorry

end NUMINAMATH_CALUDE_correct_answer_l2514_251495


namespace NUMINAMATH_CALUDE_digits_after_decimal_point_l2514_251498

theorem digits_after_decimal_point : ∃ (n : ℕ), 
  (5^7 : ℚ) / (10^5 * 15625) = (1 : ℚ) / 10^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_digits_after_decimal_point_l2514_251498


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2514_251414

/-- The area of a square inscribed in a right triangle with hypotenuse 100 units and one leg 35 units -/
theorem inscribed_square_area (h : ℝ) (l : ℝ) (s : ℝ) 
  (hyp : h = 100)  -- hypotenuse length
  (leg : l = 35)   -- one leg length
  (square : s^2 = l * (h - l)) : -- s is the side length of the inscribed square
  s^2 = 2275 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2514_251414


namespace NUMINAMATH_CALUDE_fraction_problem_l2514_251422

theorem fraction_problem (t k : ℚ) (f : ℚ) : 
  t = f * (k - 32) → t = 75 → k = 167 → f = 5/9 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2514_251422


namespace NUMINAMATH_CALUDE_sphere_with_n_plus_one_points_l2514_251471

open Set

variable {α : Type*} [MetricSpace α]

theorem sphere_with_n_plus_one_points
  (m n : ℕ)
  (points : Finset α)
  (h_card : points.card = m * n + 1)
  (h_distance : ∀ (subset : Finset α), subset ⊆ points → subset.card = m + 1 →
    ∃ (x y : α), x ∈ subset ∧ y ∈ subset ∧ x ≠ y ∧ dist x y ≤ 1) :
  ∃ (center : α), ∃ (subset : Finset α),
    subset ⊆ points ∧
    subset.card = n + 1 ∧
    ∀ x ∈ subset, dist center x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_sphere_with_n_plus_one_points_l2514_251471


namespace NUMINAMATH_CALUDE_least_expensive_trip_l2514_251421

-- Define the triangle
def triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 5000^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 4000^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 - (C.1 - A.1)^2 - (C.2 - A.2)^2

-- Define travel costs
def car_cost (distance : ℝ) : ℝ := 0.20 * distance
def train_cost (distance : ℝ) : ℝ := 150 + 0.15 * distance

-- Define the total trip cost
def trip_cost (AB BC CA : ℝ) (mode_AB mode_BC mode_CA : Bool) : ℝ :=
  (if mode_AB then train_cost AB else car_cost AB) +
  (if mode_BC then train_cost BC else car_cost BC) +
  (if mode_CA then train_cost CA else car_cost CA)

-- Theorem statement
theorem least_expensive_trip (A B C : ℝ × ℝ) :
  triangle A B C →
  ∃ (mode_AB mode_BC mode_CA : Bool),
    ∀ (other_mode_AB other_mode_BC other_mode_CA : Bool),
      trip_cost 5000 22500 4000 mode_AB mode_BC mode_CA ≤
      trip_cost 5000 22500 4000 other_mode_AB other_mode_BC other_mode_CA ∧
      trip_cost 5000 22500 4000 mode_AB mode_BC mode_CA = 5130 :=
sorry

end NUMINAMATH_CALUDE_least_expensive_trip_l2514_251421


namespace NUMINAMATH_CALUDE_grace_total_pennies_l2514_251407

/-- The value of a coin in pennies -/
def coin_value : ℕ := 10

/-- The value of a nickel in pennies -/
def nickel_value : ℕ := 5

/-- The number of coins Grace has -/
def grace_coins : ℕ := 10

/-- The number of nickels Grace has -/
def grace_nickels : ℕ := 10

/-- The total number of pennies Grace will have after exchanging her coins and nickels -/
theorem grace_total_pennies : 
  grace_coins * coin_value + grace_nickels * nickel_value = 150 := by
  sorry

end NUMINAMATH_CALUDE_grace_total_pennies_l2514_251407


namespace NUMINAMATH_CALUDE_b_power_a_equals_sixteen_l2514_251452

theorem b_power_a_equals_sixteen (a b : ℝ) : 
  b = Real.sqrt (2 - a) + Real.sqrt (a - 2) - 4 → b^a = 16 := by
sorry

end NUMINAMATH_CALUDE_b_power_a_equals_sixteen_l2514_251452


namespace NUMINAMATH_CALUDE_chocolate_distribution_l2514_251486

theorem chocolate_distribution (total_chocolates : ℕ) (total_children : ℕ) 
  (boys : ℕ) (girls : ℕ) (chocolates_per_boy : ℕ) :
  total_chocolates = 3000 →
  total_children = 120 →
  boys = 60 →
  girls = 60 →
  chocolates_per_boy = 2 →
  (total_chocolates - boys * chocolates_per_boy) / girls = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l2514_251486


namespace NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l2514_251467

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_through_isosceles_triangle_vertices (a b c : ℝ) (h_isosceles : a = c) 
  (h_sides : a = 5 ∧ c = 5) (h_base : b = 4) :
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = (13125/1764) * π := by
sorry

end NUMINAMATH_CALUDE_circle_area_through_isosceles_triangle_vertices_l2514_251467


namespace NUMINAMATH_CALUDE_max_segment_product_l2514_251430

-- Define the segment AB of unit length
def unitSegment : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define a function to calculate the product of segment lengths
def segmentProduct (a b c : ℝ) : ℝ :=
  a * (a + b) * 1 * b * (b + c) * c

-- Theorem statement
theorem max_segment_product :
  ∃ (max : ℝ), max = Real.sqrt 5 / 125 ∧
  ∀ (a b c : ℝ), a ∈ unitSegment → b ∈ unitSegment → c ∈ unitSegment →
  a + b + c = 1 → segmentProduct a b c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_segment_product_l2514_251430


namespace NUMINAMATH_CALUDE_max_value_a_l2514_251418

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 50) 
  (h5 : d > 10) : 
  a ≤ 2924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 ∧ 
    d' > 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l2514_251418


namespace NUMINAMATH_CALUDE_garden_area_l2514_251433

/-- Calculates the area of a garden given property dimensions and garden proportions -/
theorem garden_area 
  (property_width : ℝ) 
  (property_length : ℝ) 
  (garden_width_ratio : ℝ) 
  (garden_length_ratio : ℝ) 
  (h1 : property_width = 1000) 
  (h2 : property_length = 2250) 
  (h3 : garden_width_ratio = 1 / 8) 
  (h4 : garden_length_ratio = 1 / 10) : 
  garden_width_ratio * property_width * garden_length_ratio * property_length = 28125 := by
  sorry

#check garden_area

end NUMINAMATH_CALUDE_garden_area_l2514_251433


namespace NUMINAMATH_CALUDE_problem_solution_l2514_251413

theorem problem_solution (a b c d m n : ℕ+) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 1989)
  (h2 : a + b + c + d = m^2)
  (h3 : max a (max b (max c d)) = n^2) :
  m = 9 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2514_251413


namespace NUMINAMATH_CALUDE_digit_sum_equals_sixteen_l2514_251458

/-- Given distinct digits a, b, and c satisfying aba + aba = cbc,
    prove that a + b + c = 16 -/
theorem digit_sum_equals_sixteen
  (a b c : ℕ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_digits : a < 10 ∧ b < 10 ∧ c < 10)
  (h_equation : 100 * a + 10 * b + a + 100 * a + 10 * b + a = 100 * c + 10 * b + c) :
  a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_equals_sixteen_l2514_251458


namespace NUMINAMATH_CALUDE_sum_common_terms_example_l2514_251484

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℕ
  diff : ℕ
  last : ℕ

/-- Calculates the sum of common terms between two arithmetic sequences -/
def sumCommonTerms (seq1 seq2 : ArithmeticSequence) : ℕ :=
  sorry

theorem sum_common_terms_example :
  let seq1 : ArithmeticSequence := ⟨2, 4, 210⟩
  let seq2 : ArithmeticSequence := ⟨2, 6, 212⟩
  sumCommonTerms seq1 seq2 = 1872 :=
by sorry

end NUMINAMATH_CALUDE_sum_common_terms_example_l2514_251484


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l2514_251482

theorem scientific_notation_equivalence : 
  6390000 = 6.39 * (10 ^ 6) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l2514_251482


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2514_251427

theorem fraction_evaluation : (450 : ℚ) / (6 * 5 - 10 / 2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2514_251427


namespace NUMINAMATH_CALUDE_eight_real_numbers_inequality_l2514_251405

theorem eight_real_numbers_inequality (x : Fin 8 → ℝ) (h : Function.Injective x) :
  ∃ i j : Fin 8, i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (π / 7) := by
  sorry

end NUMINAMATH_CALUDE_eight_real_numbers_inequality_l2514_251405


namespace NUMINAMATH_CALUDE_chair_color_probability_l2514_251460

theorem chair_color_probability (black_chairs brown_chairs : ℕ) 
  (h1 : black_chairs = 15) (h2 : brown_chairs = 18) : 
  let total_chairs := black_chairs + brown_chairs
  (black_chairs : ℚ) / total_chairs * ((black_chairs - 1) / (total_chairs - 1)) + 
  (brown_chairs : ℚ) / total_chairs * ((brown_chairs - 1) / (total_chairs - 1)) = 
  (15 : ℚ) / 33 * 14 / 32 + 18 / 33 * 17 / 32 := by
sorry

end NUMINAMATH_CALUDE_chair_color_probability_l2514_251460


namespace NUMINAMATH_CALUDE_inner_square_probability_10x10_l2514_251465

/-- Represents a square checkerboard -/
structure Checkerboard where
  size : ℕ
  total_squares : ℕ
  perimeter_squares : ℕ
  inner_squares : ℕ

/-- Creates a 10x10 checkerboard -/
def create_10x10_board : Checkerboard :=
  { size := 10,
    total_squares := 100,
    perimeter_squares := 36,
    inner_squares := 64 }

/-- Calculates the probability of choosing an inner square -/
def inner_square_probability (board : Checkerboard) : ℚ :=
  board.inner_squares / board.total_squares

theorem inner_square_probability_10x10 :
  inner_square_probability create_10x10_board = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_probability_10x10_l2514_251465


namespace NUMINAMATH_CALUDE_linear_function_k_value_l2514_251440

theorem linear_function_k_value : ∀ k : ℝ, 
  (∀ x y : ℝ, y = k * x + 3) →  -- Linear function condition
  (2 = k * 1 + 3) →             -- Passes through (1, 2)
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_linear_function_k_value_l2514_251440


namespace NUMINAMATH_CALUDE_fixed_point_theorem_proof_l2514_251431

def fixed_point_theorem (f : ℝ → ℝ) (h_inverse : Function.Bijective f) : Prop :=
  let f_inv := Function.invFun f
  (f_inv (-(-1) + 2) = 2) → (f ((-3) - 1) = -3)

theorem fixed_point_theorem_proof (f : ℝ → ℝ) (h_inverse : Function.Bijective f) :
  fixed_point_theorem f h_inverse := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_proof_l2514_251431


namespace NUMINAMATH_CALUDE_alcohol_mixture_problem_l2514_251456

/-- Given a 1 gallon container of 75% alcohol solution, if 0.4 gallon is drained off and
    replaced with x% alcohol solution to produce a 1 gallon 65% alcohol solution,
    then x = 50%. -/
theorem alcohol_mixture_problem (x : ℝ) : 
  (0.75 * (1 - 0.4) + 0.4 * (x / 100) = 0.65) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_problem_l2514_251456


namespace NUMINAMATH_CALUDE_weight_sum_determination_l2514_251443

/-- Given the weights of four people in pairs, prove that the sum of the weights of two specific people can be determined. -/
theorem weight_sum_determination (a b c d : ℝ) 
  (h1 : a + b = 280)
  (h2 : b + c = 230)
  (h3 : c + d = 250)
  (h4 : a + d = 300) :
  a + c = 250 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_determination_l2514_251443


namespace NUMINAMATH_CALUDE_theresas_work_hours_l2514_251476

theorem theresas_work_hours (total_weeks : ℕ) (target_average : ℕ) 
  (week1 week2 week3 week4 : ℕ) (additional_task : ℕ) :
  total_weeks = 5 →
  target_average = 12 →
  week1 = 10 →
  week2 = 14 →
  week3 = 11 →
  week4 = 9 →
  additional_task = 1 →
  ∃ (week5 : ℕ), 
    (week1 + week2 + week3 + week4 + week5 + additional_task) / total_weeks = target_average ∧
    week5 = 15 :=
by sorry

end NUMINAMATH_CALUDE_theresas_work_hours_l2514_251476


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2514_251478

def circle_radius : ℝ := 2
def inner_circle_radius : ℝ := 1
def num_points : ℕ := 6
def num_symmetrical_parts : ℕ := 3

theorem shaded_area_theorem :
  let sector_angle : ℝ := 2 * Real.pi / num_points
  let sector_area : ℝ := (1 / 2) * circle_radius^2 * sector_angle
  let triangle_area : ℝ := (1 / 2) * circle_radius * inner_circle_radius * Real.sin (sector_angle / 2)
  let quadrilateral_area : ℝ := 2 * triangle_area
  let part_area : ℝ := sector_area + quadrilateral_area
  num_symmetrical_parts * part_area = 2 * Real.pi + 3 := by sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2514_251478


namespace NUMINAMATH_CALUDE_vector_parallelism_transitive_l2514_251461

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (v w : V) : Prop := ∃ (k : ℝ), v = k • w

theorem vector_parallelism_transitive (a b c : V) 
  (hab : parallel a b) (hbc : parallel b c) (hb : b ≠ 0) : 
  parallel a c := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_transitive_l2514_251461


namespace NUMINAMATH_CALUDE_not_in_second_quadrant_l2514_251472

/-- A linear function y = x - 2 -/
def f (x : ℝ) : ℝ := x - 2

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem: The graph of y = x - 2 does not pass through the second quadrant -/
theorem not_in_second_quadrant :
  ¬ ∃ (x : ℝ), second_quadrant x (f x) := by
  sorry

end NUMINAMATH_CALUDE_not_in_second_quadrant_l2514_251472


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2514_251477

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n + q

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 4 + a 8 = π →
  a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2514_251477


namespace NUMINAMATH_CALUDE_infinitely_many_planes_through_point_parallel_to_line_l2514_251464

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Predicate to check if a point is outside a line -/
def isOutside (P : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Predicate to check if a plane passes through a point -/
def passesThroughPoint (plane : Plane3D) (P : Point3D) : Prop :=
  sorry

/-- Predicate to check if a plane is parallel to a line -/
def isParallelToLine (plane : Plane3D) (l : Line3D) : Prop :=
  sorry

/-- The main theorem -/
theorem infinitely_many_planes_through_point_parallel_to_line
  (P : Point3D) (l : Line3D) (h : isOutside P l) :
  ∃ (f : ℕ → Plane3D), Function.Injective f ∧
    (∀ n : ℕ, passesThroughPoint (f n) P ∧ isParallelToLine (f n) l) :=
  sorry

end NUMINAMATH_CALUDE_infinitely_many_planes_through_point_parallel_to_line_l2514_251464


namespace NUMINAMATH_CALUDE_sector_arc_length_l2514_251453

/-- Given a sector with central angle π/3 and radius 3, its arc length is π. -/
theorem sector_arc_length (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 3) :
  θ * r = π := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2514_251453


namespace NUMINAMATH_CALUDE_olivias_correct_answers_l2514_251409

theorem olivias_correct_answers 
  (total_problems : ℕ) 
  (correct_points : ℤ) 
  (incorrect_points : ℤ) 
  (total_score : ℤ) 
  (h1 : total_problems = 15)
  (h2 : correct_points = 6)
  (h3 : incorrect_points = -3)
  (h4 : total_score = 45) :
  ∃ (correct_answers : ℕ),
    correct_answers * correct_points + 
    (total_problems - correct_answers) * incorrect_points = total_score ∧
    correct_answers = 10 := by
  sorry

end NUMINAMATH_CALUDE_olivias_correct_answers_l2514_251409


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_ten_thousand_l2514_251444

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

theorem smallest_n_divisible_by_ten_thousand :
  ∀ n : ℕ, n > 0 → n < 9375 → ¬(10000 ∣ sum_of_naturals n) ∧ (10000 ∣ sum_of_naturals 9375) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_ten_thousand_l2514_251444


namespace NUMINAMATH_CALUDE_max_value_condition_l2514_251493

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2*a - 1) * x

theorem max_value_condition (a : ℝ) :
  (∀ x > 0, f a x ≤ f a 1) → a > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_condition_l2514_251493


namespace NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l2514_251417

-- Define the expression
def expression (a b c : ℝ) : ℝ := (40 * a^6 * b^8 * c^14) ^ (1/3)

-- Define a function to calculate the sum of exponents outside the radical
def sum_of_exponents_outside_radical (a b c : ℝ) : ℕ :=
  let simplified := expression a b c
  -- This is a placeholder. In a real implementation, we would need to
  -- analyze the simplified expression to determine the exponents.
  8

-- The theorem to prove
theorem sum_of_exponents_is_eight :
  ∀ a b c : ℝ, sum_of_exponents_outside_radical a b c = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_is_eight_l2514_251417


namespace NUMINAMATH_CALUDE_expansion_coefficient_ratio_l2514_251400

theorem expansion_coefficient_ratio (n : ℕ) : 
  (∀ a b : ℝ, (4 : ℝ)^n / (2 : ℝ)^n = 64) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_ratio_l2514_251400


namespace NUMINAMATH_CALUDE_f_values_l2514_251439

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 4 else 4 * x - 1

theorem f_values : f (-3) = -5 ∧ f 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_values_l2514_251439


namespace NUMINAMATH_CALUDE_green_ball_probability_l2514_251406

/-- Represents a container of balls -/
structure Container where
  red : Nat
  green : Nat

/-- The probability of selecting a specific container -/
def containerProb : ℚ := 1 / 3

/-- The probability of selecting a green ball from a given container -/
def greenBallProb (c : Container) : ℚ := c.green / (c.red + c.green)

/-- The containers A, B, and C -/
def containerA : Container := ⟨5, 5⟩
def containerB : Container := ⟨3, 3⟩
def containerC : Container := ⟨3, 3⟩

/-- The theorem stating the probability of selecting a green ball -/
theorem green_ball_probability :
  containerProb * greenBallProb containerA +
  containerProb * greenBallProb containerB +
  containerProb * greenBallProb containerC = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2514_251406


namespace NUMINAMATH_CALUDE_no_square_possible_l2514_251494

/-- Represents the lengths of sticks available -/
def stick_lengths : List ℕ := [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

/-- The total length of all sticks -/
def total_length : ℕ := stick_lengths.sum

/-- Predicate to check if a square can be formed -/
def can_form_square (lengths : List ℕ) : Prop :=
  ∃ (side_length : ℕ), side_length > 0 ∧ 
  4 * side_length = lengths.sum ∧
  ∃ (subset : List ℕ), subset.sum = side_length ∧ subset.toFinset ⊆ lengths.toFinset

theorem no_square_possible : ¬(can_form_square stick_lengths) := by
  sorry

end NUMINAMATH_CALUDE_no_square_possible_l2514_251494


namespace NUMINAMATH_CALUDE_unique_paintable_number_l2514_251473

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def paints_every_nth (start : ℕ) (step : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = start + k * step

def is_paintable (h t u : ℕ) : Prop :=
  h = 4 ∧
  t % 2 ≠ 0 ∧
  is_prime u ∧
  ∀ n : ℕ, n > 0 →
    (paints_every_nth 1 4 n ∨ paints_every_nth 3 t n ∨ paints_every_nth 5 u n) ∧
    ¬(paints_every_nth 1 4 n ∧ paints_every_nth 3 t n) ∧
    ¬(paints_every_nth 1 4 n ∧ paints_every_nth 5 u n) ∧
    ¬(paints_every_nth 3 t n ∧ paints_every_nth 5 u n)

theorem unique_paintable_number :
  ∀ h t u : ℕ, is_paintable h t u → 100 * t + 10 * u + h = 354 :=
sorry

end NUMINAMATH_CALUDE_unique_paintable_number_l2514_251473


namespace NUMINAMATH_CALUDE_abundant_product_l2514_251446

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- A number is abundant if the sum of its divisors is greater than twice the number -/
def abundant (n : ℕ) : Prop := sigma n > 2 * n

/-- If a is abundant, then ab is abundant for any positive integer b -/
theorem abundant_product {a b : ℕ} (ha : a > 0) (hb : b > 0) (hab : abundant a) : abundant (a * b) := by
  sorry

end NUMINAMATH_CALUDE_abundant_product_l2514_251446


namespace NUMINAMATH_CALUDE_bella_steps_l2514_251475

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℝ := 10560

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℝ := 3

/-- The length of Bella's step in feet -/
def step_length : ℝ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps : ℕ := 880

theorem bella_steps :
  (distance / (1 + speed_ratio)) / step_length = steps := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_l2514_251475


namespace NUMINAMATH_CALUDE_wood_cutting_l2514_251488

/-- Given a piece of wood that can be sawed into 9 sections of 4 meters each,
    prove that 11 cuts are needed to saw it into 3-meter sections. -/
theorem wood_cutting (wood_length : ℕ) (num_long_sections : ℕ) (long_section_length : ℕ) 
  (short_section_length : ℕ) (h1 : wood_length = num_long_sections * long_section_length)
  (h2 : num_long_sections = 9) (h3 : long_section_length = 4) (h4 : short_section_length = 3) : 
  (wood_length / short_section_length) - 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_wood_cutting_l2514_251488


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2514_251416

theorem trigonometric_identity (α : ℝ) (m : ℝ) (h : Real.sin α - Real.cos α = m) :
  (Real.sin (4 * α) + Real.sin (10 * α) - Real.sin (6 * α)) /
  (Real.cos (2 * α) + 1 - 2 * Real.sin (4 * α) ^ 2) = 2 * (1 - m ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2514_251416


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2514_251411

def expansion (x : ℝ) := (1 + x^2) * (1 - x)^5

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expansion))) 0 / 6 = -15 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2514_251411


namespace NUMINAMATH_CALUDE_triangle_properties_l2514_251429

theorem triangle_properties (a b c A B C : ℝ) (h1 : a * Real.sin A - c * Real.sin C = (a - b) * Real.sin B)
  (h2 : c = 2 * Real.sqrt 3) (h3 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :
  C = π / 3 ∧ a + b + c = 6 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2514_251429


namespace NUMINAMATH_CALUDE_min_both_mozart_and_bach_l2514_251404

theorem min_both_mozart_and_bach 
  (total : ℕ) 
  (mozart : ℕ) 
  (bach : ℕ) 
  (h1 : total = 100) 
  (h2 : mozart = 87) 
  (h3 : bach = 70) 
  : ℕ :=
by
  sorry

#check min_both_mozart_and_bach

end NUMINAMATH_CALUDE_min_both_mozart_and_bach_l2514_251404


namespace NUMINAMATH_CALUDE_probability_both_types_selected_l2514_251481

def num_type_a : ℕ := 3
def num_type_b : ℕ := 2
def total_tvs : ℕ := num_type_a + num_type_b
def num_selected : ℕ := 3

theorem probability_both_types_selected :
  (Nat.choose num_type_a 2 * Nat.choose num_type_b 1 +
   Nat.choose num_type_a 1 * Nat.choose num_type_b 2) /
  Nat.choose total_tvs num_selected = 9 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_both_types_selected_l2514_251481


namespace NUMINAMATH_CALUDE_sum_with_rearrangement_not_1999_nines_sum_with_rearrangement_1010_divisible_by_10_l2514_251454

-- Define a function to represent digit rearrangement
def digitRearrangement (n : ℕ) : ℕ := sorry

-- Define a function to check if a number consists of 1999 nines
def is1999Nines (n : ℕ) : Prop := sorry

-- Part (a)
theorem sum_with_rearrangement_not_1999_nines (n : ℕ) : 
  ¬(is1999Nines (n + digitRearrangement n)) := sorry

-- Part (b)
theorem sum_with_rearrangement_1010_divisible_by_10 (n : ℕ) : 
  n + digitRearrangement n = 1010 → n % 10 = 0 := sorry

end NUMINAMATH_CALUDE_sum_with_rearrangement_not_1999_nines_sum_with_rearrangement_1010_divisible_by_10_l2514_251454


namespace NUMINAMATH_CALUDE_john_weekly_income_l2514_251415

/-- Represents the number of crab baskets John reels in each time he collects crabs -/
def baskets_per_collection : ℕ := 3

/-- Represents the number of crabs each basket holds -/
def crabs_per_basket : ℕ := 4

/-- Represents the number of times John collects crabs per week -/
def collections_per_week : ℕ := 2

/-- Represents the selling price of each crab in dollars -/
def price_per_crab : ℕ := 3

/-- Calculates John's weekly income from selling crabs -/
def weekly_income : ℕ := baskets_per_collection * crabs_per_basket * collections_per_week * price_per_crab

/-- Theorem stating that John's weekly income from selling crabs is $72 -/
theorem john_weekly_income : weekly_income = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_weekly_income_l2514_251415


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l2514_251455

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  (∀ m : ℕ, is_three_digit m → is_7_heavy m → 103 ≤ m) ∧ 
  is_three_digit 103 ∧ 
  is_7_heavy 103 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l2514_251455


namespace NUMINAMATH_CALUDE_cricket_innings_calculation_l2514_251437

/-- The number of innings played by a cricket player -/
def innings : ℕ := sorry

/-- The current average runs per innings -/
def current_average : ℚ := 22

/-- The increase in average after scoring 92 runs in the next innings -/
def average_increase : ℚ := 5

/-- The runs scored in the next innings -/
def next_innings_runs : ℕ := 92

theorem cricket_innings_calculation :
  (innings * current_average + next_innings_runs) / (innings + 1) = current_average + average_increase →
  innings = 13 := by sorry

end NUMINAMATH_CALUDE_cricket_innings_calculation_l2514_251437


namespace NUMINAMATH_CALUDE_conference_duration_theorem_l2514_251448

/-- Calculates the duration of a conference in minutes, excluding the lunch break. -/
def conference_duration (total_hours : ℕ) (total_minutes : ℕ) (lunch_break : ℕ) : ℕ :=
  total_hours * 60 + total_minutes - lunch_break

/-- Proves that a conference lasting 8 hours and 40 minutes with a 15-minute lunch break
    has an active session time of 505 minutes. -/
theorem conference_duration_theorem :
  conference_duration 8 40 15 = 505 := by
  sorry

end NUMINAMATH_CALUDE_conference_duration_theorem_l2514_251448


namespace NUMINAMATH_CALUDE_line_through_origin_l2514_251442

variable (m n p : ℝ)
variable (h : m ≠ 0 ∨ n ≠ 0 ∨ p ≠ 0)

def line_set : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | ∃ (k : ℝ), x = k * m ∧ y = k * n ∧ z = k * p}

theorem line_through_origin (m n p : ℝ) (h : m ≠ 0 ∨ n ≠ 0 ∨ p ≠ 0) :
  ∃ (a b c : ℝ), line_set m n p = {(x, y, z) | a * x + b * y + c * z = 0} ∧
  (0, 0, 0) ∈ line_set m n p :=
sorry

end NUMINAMATH_CALUDE_line_through_origin_l2514_251442


namespace NUMINAMATH_CALUDE_value_of_x_l2514_251483

theorem value_of_x (x y z : ℚ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 80) :
  x = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l2514_251483


namespace NUMINAMATH_CALUDE_T_bounds_l2514_251449

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ 1 ∧ y = (3*x + 5) / (x + 3)}

theorem T_bounds :
  ∃ (q P : ℝ),
    (∀ y ∈ T, q ≤ y) ∧
    (∀ y ∈ T, y ≤ P) ∧
    q ∈ T ∧
    P ∉ T :=
sorry

end NUMINAMATH_CALUDE_T_bounds_l2514_251449


namespace NUMINAMATH_CALUDE_infinite_solutions_ratio_l2514_251459

theorem infinite_solutions_ratio (a b c : ℚ) : 
  (∀ x, a * x^2 + b * x + c = (x - 1) * (2 * x + 1)) → 
  a = 2 ∧ b = -1 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_ratio_l2514_251459


namespace NUMINAMATH_CALUDE_sum_of_subtraction_equation_l2514_251402

theorem sum_of_subtraction_equation :
  ∀ A B : ℕ,
    A ≠ B →
    A < 10 →
    B < 10 →
    (80 + A) - (10 * B + 2) = 45 →
    A + B = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_subtraction_equation_l2514_251402


namespace NUMINAMATH_CALUDE_blue_markers_count_l2514_251424

theorem blue_markers_count (total_markers red_markers : ℕ) 
  (h1 : total_markers = 3343)
  (h2 : red_markers = 2315) :
  total_markers - red_markers = 1028 := by
  sorry

end NUMINAMATH_CALUDE_blue_markers_count_l2514_251424


namespace NUMINAMATH_CALUDE_quadratic_sum_real_roots_l2514_251423

/-- A quadratic polynomial with positive leading coefficient and real roots -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  real_roots : b^2 - 4*a*c ≥ 0

/-- The sum of two QuadraticPolynomials -/
def add_poly (P Q : QuadraticPolynomial) : QuadraticPolynomial :=
  { a := P.a + Q.a,
    b := P.b + Q.b,
    c := P.c + Q.c,
    a_pos := by 
      apply add_pos P.a_pos Q.a_pos
    real_roots := sorry }

/-- Two QuadraticPolynomials have a common root -/
def have_common_root (P Q : QuadraticPolynomial) : Prop :=
  ∃ x : ℝ, P.a * x^2 + P.b * x + P.c = 0 ∧ Q.a * x^2 + Q.b * x + Q.c = 0

theorem quadratic_sum_real_roots (P₁ P₂ P₃ : QuadraticPolynomial)
  (h₁₂ : have_common_root P₁ P₂)
  (h₂₃ : have_common_root P₂ P₃)
  (h₁₃ : have_common_root P₁ P₃) :
  ∃ x : ℝ, (P₁.a + P₂.a + P₃.a) * x^2 + (P₁.b + P₂.b + P₃.b) * x + (P₁.c + P₂.c + P₃.c) = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_real_roots_l2514_251423


namespace NUMINAMATH_CALUDE_gcd_2024_2048_l2514_251425

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2024_2048_l2514_251425


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l2514_251496

theorem geometric_sequence_first_term
  (a : ℕ → ℕ)
  (h1 : a 2 = 3)
  (h2 : a 3 = 9)
  (h3 : a 4 = 27)
  (h4 : a 5 = 81)
  (h5 : a 6 = 243)
  (h_geometric : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n) :
  a 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l2514_251496


namespace NUMINAMATH_CALUDE_rice_mixture_cost_l2514_251420

/-- Proves that mixing two varieties of rice in a given ratio results in the specified cost per kg -/
theorem rice_mixture_cost 
  (cost1 : ℝ) 
  (cost2 : ℝ) 
  (ratio : ℝ) 
  (mixture_cost : ℝ) 
  (h1 : cost1 = 7) 
  (h2 : cost2 = 8.75) 
  (h3 : ratio = 2.5) 
  (h4 : mixture_cost = 7.5) : 
  (ratio * cost1 + cost2) / (ratio + 1) = mixture_cost := by
  sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_l2514_251420


namespace NUMINAMATH_CALUDE_scientific_notation_of_2600000_l2514_251485

theorem scientific_notation_of_2600000 :
  2600000 = 2.6 * (10 ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2600000_l2514_251485


namespace NUMINAMATH_CALUDE_amber_bronze_selection_l2514_251480

/-- Represents a cell in the grid -/
inductive Cell
| Amber
| Bronze

/-- Represents the grid -/
def Grid (a b : ℕ) := Fin (a + b + 1) → Fin (a + b + 1) → Cell

/-- Counts the number of amber cells in the grid -/
def countAmber (g : Grid a b) : ℕ := sorry

/-- Counts the number of bronze cells in the grid -/
def countBronze (g : Grid a b) : ℕ := sorry

/-- Represents a selection of cells -/
def Selection (a b : ℕ) := Fin (a + b) → Fin (a + b + 1) × Fin (a + b + 1)

/-- Checks if a selection is valid (no two cells in the same row or column) -/
def isValidSelection (s : Selection a b) : Prop := sorry

/-- Counts the number of amber cells in a selection -/
def countAmberInSelection (g : Grid a b) (s : Selection a b) : ℕ := sorry

/-- Counts the number of bronze cells in a selection -/
def countBronzeInSelection (g : Grid a b) (s : Selection a b) : ℕ := sorry

theorem amber_bronze_selection (a b : ℕ) (g : Grid a b) 
  (ha : a > 0) (hb : b > 0)
  (hamber : countAmber g ≥ a^2 + a*b - b)
  (hbronze : countBronze g ≥ b^2 + a*b - a) :
  ∃ (s : Selection a b), 
    isValidSelection s ∧ 
    countAmberInSelection g s = a ∧ 
    countBronzeInSelection g s = b := by
  sorry

end NUMINAMATH_CALUDE_amber_bronze_selection_l2514_251480


namespace NUMINAMATH_CALUDE_school_children_count_prove_school_children_count_l2514_251432

theorem school_children_count : ℕ → Prop :=
  fun total_children =>
    let total_bananas := 2 * total_children
    total_bananas = 4 * (total_children - 350) →
    total_children = 700

-- Proof
theorem prove_school_children_count :
  ∃ (n : ℕ), school_children_count n :=
by
  sorry

end NUMINAMATH_CALUDE_school_children_count_prove_school_children_count_l2514_251432


namespace NUMINAMATH_CALUDE_time_addition_theorem_l2514_251435

/-- Represents time in a 12-hour format -/
structure Time12 where
  hours : Nat
  minutes : Nat
  seconds : Nat
  isPM : Bool

/-- Represents a duration in hours, minutes, and seconds -/
structure Duration where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a time and returns the resulting time -/
def addDuration (t : Time12) (d : Duration) : Time12 := sorry

/-- Computes the sum of hours, minutes, and seconds for a given time -/
def sumComponents (t : Time12) : Nat := sorry

theorem time_addition_theorem (initialTime : Time12) (duration : Duration) :
  initialTime = Time12.mk 3 0 0 true →
  duration = Duration.mk 300 55 30 →
  (addDuration initialTime duration = Time12.mk 3 55 30 true) ∧
  (sumComponents (addDuration initialTime duration) = 88) := by sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l2514_251435


namespace NUMINAMATH_CALUDE_yogurt_expiration_probability_l2514_251412

def total_boxes : ℕ := 6
def expired_boxes : ℕ := 2
def selected_boxes : ℕ := 2

def probability_at_least_one_expired : ℚ := 3/5

theorem yogurt_expiration_probability :
  (Nat.choose total_boxes selected_boxes - Nat.choose (total_boxes - expired_boxes) selected_boxes) /
  Nat.choose total_boxes selected_boxes = probability_at_least_one_expired :=
sorry

end NUMINAMATH_CALUDE_yogurt_expiration_probability_l2514_251412


namespace NUMINAMATH_CALUDE_inconsistent_age_sum_l2514_251401

theorem inconsistent_age_sum (total_students : ℕ) (class_avg_age : ℝ)
  (group1_size group2_size group3_size unknown_size : ℕ)
  (group1_avg_age group2_avg_age group3_avg_age : ℝ)
  (unknown_sum_age : ℝ) :
  total_students = 25 →
  class_avg_age = 18 →
  group1_size = 8 →
  group2_size = 10 →
  group3_size = 5 →
  unknown_size = 2 →
  group1_avg_age = 16 →
  group2_avg_age = 20 →
  group3_avg_age = 17 →
  unknown_sum_age = 35 →
  total_students = group1_size + group2_size + group3_size + unknown_size →
  ¬(class_avg_age * total_students =
    group1_avg_age * group1_size + group2_avg_age * group2_size +
    group3_avg_age * group3_size + unknown_sum_age) :=
by sorry

end NUMINAMATH_CALUDE_inconsistent_age_sum_l2514_251401


namespace NUMINAMATH_CALUDE_snacks_ryan_can_buy_l2514_251468

def one_way_travel_time : ℝ := 2
def snack_cost (round_trip_time : ℝ) : ℝ := 10 * round_trip_time
def ryan_budget : ℝ := 2000

theorem snacks_ryan_can_buy :
  (ryan_budget / snack_cost (2 * one_way_travel_time)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_snacks_ryan_can_buy_l2514_251468


namespace NUMINAMATH_CALUDE_original_class_size_l2514_251497

theorem original_class_size (A B C : ℕ) (N : ℕ) (D : ℕ) :
  A = 40 →
  B = 32 →
  C = 36 →
  D = N * A →
  D + 8 * B = (N + 8) * C →
  N = 8 := by
sorry

end NUMINAMATH_CALUDE_original_class_size_l2514_251497


namespace NUMINAMATH_CALUDE_cube_surface_area_l2514_251403

/-- Given a cube with volume 125 cubic cm, its surface area is 150 square cm. -/
theorem cube_surface_area (volume : ℝ) (side_length : ℝ) (surface_area : ℝ) : 
  volume = 125 → 
  side_length ^ 3 = volume →
  surface_area = 6 * side_length ^ 2 →
  surface_area = 150 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2514_251403


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2514_251470

theorem quadratic_root_implies_m_value :
  ∀ m : ℝ, ((-1)^2 - 2*(-1) + m = 0) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_value_l2514_251470


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l2514_251469

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 7 ↔ 3 < x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l2514_251469


namespace NUMINAMATH_CALUDE_research_project_hours_difference_l2514_251457

/-- The research project problem -/
theorem research_project_hours_difference
  (total_payment : ℝ)
  (wage_difference : ℝ)
  (wage_ratio : ℝ)
  (h1 : total_payment = 480)
  (h2 : wage_difference = 8)
  (h3 : wage_ratio = 1.5) :
  ∃ (hours_p hours_q : ℝ),
    hours_q - hours_p = 10 ∧
    hours_p * (wage_ratio * (total_payment / hours_q)) = total_payment ∧
    hours_q * (total_payment / hours_q) = total_payment ∧
    wage_ratio * (total_payment / hours_q) = (total_payment / hours_q) + wage_difference :=
by sorry


end NUMINAMATH_CALUDE_research_project_hours_difference_l2514_251457


namespace NUMINAMATH_CALUDE_simplify_expression_l2514_251441

theorem simplify_expression (b : ℝ) : 3*b*(3*b^3 + 2*b) - 2*b^2 = 9*b^4 + 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2514_251441


namespace NUMINAMATH_CALUDE_remainder_thirteen_150_mod_11_l2514_251489

theorem remainder_thirteen_150_mod_11 : 13^150 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_thirteen_150_mod_11_l2514_251489


namespace NUMINAMATH_CALUDE_salary_distribution_l2514_251436

def salary : ℚ := 140000

def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5
def remaining : ℚ := 14000

def food_fraction : ℚ := 1/5

theorem salary_distribution :
  food_fraction * salary + rent_fraction * salary + clothes_fraction * salary + remaining = salary :=
by sorry

end NUMINAMATH_CALUDE_salary_distribution_l2514_251436


namespace NUMINAMATH_CALUDE_shoes_in_box_l2514_251434

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 5

/-- The probability of selecting two matching shoes at random -/
def prob_matching : ℚ := 1 / 9

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- Theorem stating that given the conditions, the total number of shoes is 10 -/
theorem shoes_in_box :
  (num_pairs = 5) →
  (prob_matching = 1 / 9) →
  (total_shoes = 10) := by
  sorry


end NUMINAMATH_CALUDE_shoes_in_box_l2514_251434


namespace NUMINAMATH_CALUDE_bucket_capacity_proof_l2514_251490

/-- The capacity of a bucket that satisfies the given conditions -/
def bucket_capacity : ℝ :=
  let tank_capacity : ℝ := 48
  let small_bucket_capacity : ℝ := 3
  3

theorem bucket_capacity_proof :
  let tank_capacity : ℝ := 48
  let small_bucket_capacity : ℝ := 3
  (tank_capacity / bucket_capacity : ℝ) = (tank_capacity / small_bucket_capacity) - 4 := by
  sorry

#check bucket_capacity_proof

end NUMINAMATH_CALUDE_bucket_capacity_proof_l2514_251490


namespace NUMINAMATH_CALUDE_solve_equation_l2514_251438

theorem solve_equation (x : ℝ) : 3 * x + 12 = (1/3) * (7 * x + 42) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2514_251438


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l2514_251428

/-- The number of ways to partition n into at most k parts, where the order doesn't matter -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 11 ways to partition 7 into at most 4 parts -/
theorem seven_balls_four_boxes : partition_count 7 4 = 11 := by sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l2514_251428


namespace NUMINAMATH_CALUDE_final_comic_book_count_l2514_251491

def initial_books : ℕ := 22
def books_bought : ℕ := 6

theorem final_comic_book_count :
  (initial_books / 2 + books_bought : ℕ) = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_final_comic_book_count_l2514_251491


namespace NUMINAMATH_CALUDE_sequence_properties_l2514_251474

/-- Sequence a_n with sum S_n satisfying S_n = 2a_n - 3 for n ∈ ℕ* -/
def S (a : ℕ+ → ℝ) (n : ℕ+) : ℝ := 2 * a n - 3

/-- Sequence b_n defined as b_n = (n-1)a_n -/
def b (a : ℕ+ → ℝ) (n : ℕ+) : ℝ := (n.val - 1) * a n

/-- Sum T_n of the first n terms of sequence b_n -/
def T (b : ℕ+ → ℝ) : ℕ+ → ℝ := fun n ↦ (Finset.range n.val).sum (fun i ↦ b ⟨i + 1, Nat.succ_pos i⟩)

theorem sequence_properties (a : ℕ+ → ℝ) (k : ℝ) :
  (∀ n : ℕ+, S a n = 2 * a n - 3) →
  (∀ n : ℕ+, a n = 3 * 2^(n.val - 1)) ∧
  (∀ n : ℕ+, T (b a) n = 3 * (n.val - 2) * 2^n.val + 6) ∧
  (∀ n : ℕ+, T (b a) n > k * a n + 16 * n.val - 26 → k < 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2514_251474


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l2514_251462

/-- For a normal distribution with mean 10.5, if a value 2 standard deviations
    below the mean is 8.5, then the standard deviation is 1. -/
theorem normal_distribution_std_dev (μ σ : ℝ) (x : ℝ) : 
  μ = 10.5 → x = μ - 2 * σ → x = 8.5 → σ = 1 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l2514_251462


namespace NUMINAMATH_CALUDE_stating_max_books_borrowed_is_eight_l2514_251450

/-- Represents the maximum number of books borrowed by a single student -/
def max_books_borrowed (total_students : ℕ) 
                       (zero_book_students : ℕ) 
                       (one_book_students : ℕ) 
                       (two_book_students : ℕ) 
                       (avg_books_per_student : ℕ) : ℕ :=
  let total_books := total_students * avg_books_per_student
  let remaining_students := total_students - (zero_book_students + one_book_students + two_book_students)
  let accounted_books := one_book_students + 2 * two_book_students
  let remaining_books := total_books - accounted_books
  remaining_books - (3 * (remaining_students - 1))

/-- 
Theorem stating that given the conditions in the problem, 
the maximum number of books borrowed by a single student is 8.
-/
theorem max_books_borrowed_is_eight :
  max_books_borrowed 35 2 12 10 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_stating_max_books_borrowed_is_eight_l2514_251450


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l2514_251463

theorem geometric_series_first_term 
  (sum : ℝ) 
  (sum_squares : ℝ) 
  (h1 : sum = 20) 
  (h2 : sum_squares = 80) : 
  ∃ (a r : ℝ), 
    a / (1 - r) = sum ∧ 
    a^2 / (1 - r^2) = sum_squares ∧ 
    a = 20 / 3 := by 
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l2514_251463


namespace NUMINAMATH_CALUDE_third_circle_radius_l2514_251492

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 34) (h₂ : r₂ = 14) : 
  π * r₃^2 = π * (r₁^2 - r₂^2) → r₃ = 8 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_third_circle_radius_l2514_251492


namespace NUMINAMATH_CALUDE_f_x_plus_one_l2514_251408

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 + 4*(x + 1) - 5

-- State the theorem
theorem f_x_plus_one (x : ℝ) : f (x + 1) = x^2 + 8*x + 7 := by
  sorry

end NUMINAMATH_CALUDE_f_x_plus_one_l2514_251408


namespace NUMINAMATH_CALUDE_rank_squared_inequality_l2514_251451

theorem rank_squared_inequality (A B : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : Matrix.rank A > Matrix.rank B) : 
  Matrix.rank (A ^ 2) ≥ Matrix.rank (B ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_rank_squared_inequality_l2514_251451


namespace NUMINAMATH_CALUDE_divisibility_by_1897_l2514_251479

theorem divisibility_by_1897 (n : ℕ) : 
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1897_l2514_251479


namespace NUMINAMATH_CALUDE_part1_part2_l2514_251447

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x - 2*a + 3|

-- Part 1: When a = 2
theorem part1 : {x : ℝ | f x 2 ≤ 9} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} := by sorry

-- Part 2: When a ≠ 2
theorem part2 : ∀ a : ℝ, a ≠ 2 → 
  ((∀ x : ℝ, f x a ≥ 4) ↔ (a ≤ -2/3 ∨ a ≥ 14/3)) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2514_251447


namespace NUMINAMATH_CALUDE_subset_implies_range_l2514_251499

-- Define set A
def A : Set ℝ := {x | x^2 ≤ 5*x - 4}

-- Define set M parameterized by a
def M (a : ℝ) : Set ℝ := {x | x^2 - (a+2)*x + 2*a ≤ 0}

-- Theorem statement
theorem subset_implies_range (a : ℝ) : M a ⊆ A ↔ a ∈ Set.Icc 1 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_range_l2514_251499


namespace NUMINAMATH_CALUDE_cube_edge_increase_l2514_251487

theorem cube_edge_increase (e : ℝ) (h : e > 0) :
  let A := 6 * e^2
  let A' := 2.25 * A
  let e' := Real.sqrt (A' / 6)
  (e' - e) / e = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_increase_l2514_251487


namespace NUMINAMATH_CALUDE_remainder_when_z_plus_3_div_9_is_integer_l2514_251410

theorem remainder_when_z_plus_3_div_9_is_integer (z : ℤ) :
  ∃ k : ℤ, (z + 3) / 9 = k → z ≡ 6 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_remainder_when_z_plus_3_div_9_is_integer_l2514_251410
