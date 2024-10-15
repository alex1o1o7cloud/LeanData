import Mathlib

namespace NUMINAMATH_CALUDE_count_true_props_l3340_334033

def original_prop : Prop := ∀ x : ℝ, x^2 > 1 → x > 1

def converse_prop : Prop := ∀ x : ℝ, x > 1 → x^2 > 1

def inverse_prop : Prop := ∀ x : ℝ, x^2 ≤ 1 → x ≤ 1

def contrapositive_prop : Prop := ∀ x : ℝ, x ≤ 1 → x^2 ≤ 1

theorem count_true_props :
  (converse_prop ∧ inverse_prop ∧ ¬contrapositive_prop) ∨
  (converse_prop ∧ ¬inverse_prop ∧ contrapositive_prop) ∨
  (¬converse_prop ∧ inverse_prop ∧ contrapositive_prop) :=
sorry

end NUMINAMATH_CALUDE_count_true_props_l3340_334033


namespace NUMINAMATH_CALUDE_seven_faced_prism_has_five_lateral_faces_l3340_334030

/-- A prism is a three-dimensional shape with two identical ends (bases) and flat sides. -/
structure Prism where
  total_faces : ℕ
  base_faces : ℕ := 2

/-- Define a function that calculates the number of lateral faces of a prism. -/
def lateral_faces (p : Prism) : ℕ :=
  p.total_faces - p.base_faces

/-- Theorem stating that a prism with 7 faces has 5 lateral faces. -/
theorem seven_faced_prism_has_five_lateral_faces (p : Prism) (h : p.total_faces = 7) :
  lateral_faces p = 5 := by
  sorry


end NUMINAMATH_CALUDE_seven_faced_prism_has_five_lateral_faces_l3340_334030


namespace NUMINAMATH_CALUDE_fourth_power_sum_equals_108_to_fourth_l3340_334051

theorem fourth_power_sum_equals_108_to_fourth : ∃ m : ℕ+, 
  97^4 + 84^4 + 27^4 + 3^4 = m^4 ∧ m = 108 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_equals_108_to_fourth_l3340_334051


namespace NUMINAMATH_CALUDE_birth_rate_calculation_l3340_334085

/-- Represents the annual birth rate per 1000 people in a country. -/
def birth_rate : ℝ := sorry

/-- Represents the annual death rate per 1000 people in a country. -/
def death_rate : ℝ := 19.4

/-- Represents the number of years it takes for the population to double. -/
def doubling_time : ℝ := 35

/-- The Rule of 70 for population growth. -/
axiom rule_of_70 (growth_rate : ℝ) : 
  doubling_time = 70 / growth_rate

/-- The net growth rate is the difference between birth rate and death rate. -/
def net_growth_rate : ℝ := birth_rate - death_rate

theorem birth_rate_calculation : birth_rate = 21.4 := by sorry

end NUMINAMATH_CALUDE_birth_rate_calculation_l3340_334085


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3340_334068

theorem polynomial_remainder_theorem (x : ℝ) :
  let p (x : ℝ) := x^4 - 4*x^2 + 7
  let r := p 3
  r = 52 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3340_334068


namespace NUMINAMATH_CALUDE_min_value_exponential_sum_l3340_334077

theorem min_value_exponential_sum (x y : ℝ) (h : x + 2 * y = 1) :
  2^x + 4^y ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_sum_l3340_334077


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l3340_334082

/-- Represents a triangle in the sequence -/
structure Triangle where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { a := t.a / 2 - 1,
    b := t.b / 2,
    c := t.c / 2 + 1 }

/-- Checks if a triangle is valid (satisfies triangle inequality) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The initial triangle T₁ -/
def T₁ : Triangle :=
  { a := 1009, b := 1010, c := 1011 }

/-- Generates the sequence of triangles -/
def triangleSequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (triangleSequence n)

/-- Finds the index of the last valid triangle in the sequence -/
def lastValidTriangleIndex : ℕ := sorry

/-- The last valid triangle in the sequence -/
def lastValidTriangle : Triangle :=
  triangleSequence lastValidTriangleIndex

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℚ :=
  t.a + t.b + t.c

theorem last_triangle_perimeter :
  perimeter lastValidTriangle = 71 / 8 := by sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l3340_334082


namespace NUMINAMATH_CALUDE_all_trinomials_no_roots_l3340_334015

/-- Represents a quadratic trinomial ax² + bx + c -/
structure QuadraticTrinomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Calculates the discriminant of a quadratic trinomial -/
def discriminant (q : QuadraticTrinomial) : ℤ :=
  q.b ^ 2 - 4 * q.a * q.c

/-- Checks if a quadratic trinomial has no real roots -/
def has_no_real_roots (q : QuadraticTrinomial) : Prop :=
  discriminant q < 0

/-- Creates all permutations of three coefficients -/
def all_permutations (a b c : ℤ) : List QuadraticTrinomial :=
  [
    ⟨a, b, c⟩, ⟨a, c, b⟩,
    ⟨b, a, c⟩, ⟨b, c, a⟩,
    ⟨c, a, b⟩, ⟨c, b, a⟩
  ]

theorem all_trinomials_no_roots
  (a b c : ℤ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  ∀ q ∈ all_permutations a b c, has_no_real_roots q :=
sorry

end NUMINAMATH_CALUDE_all_trinomials_no_roots_l3340_334015


namespace NUMINAMATH_CALUDE_equation_system_solution_l3340_334032

theorem equation_system_solution (a b : ℝ) :
  (∃ (a' : ℝ), a' * 1 + 4 * (-1) = 23 ∧ 3 * 1 - b * (-1) = 5) →
  (∃ (b' : ℝ), a * 7 + 4 * (-3) = 23 ∧ 3 * 7 - b' * (-3) = 5) →
  (a^2 - 2*a*b + b^2 = 9) ∧
  (a * 3 + 4 * 2 = 23 ∧ 3 * 3 - b * 2 = 5) := by
sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3340_334032


namespace NUMINAMATH_CALUDE_suki_bag_weight_l3340_334086

-- Define the given quantities
def suki_bags : ℝ := 6.5
def jimmy_bags : ℝ := 4.5
def jimmy_bag_weight : ℝ := 18
def container_weight : ℝ := 8
def total_containers : ℕ := 28

-- Define the theorem
theorem suki_bag_weight :
  let total_weight := container_weight * total_containers
  let jimmy_total_weight := jimmy_bags * jimmy_bag_weight
  let suki_total_weight := total_weight - jimmy_total_weight
  suki_total_weight / suki_bags = 22 := by
sorry


end NUMINAMATH_CALUDE_suki_bag_weight_l3340_334086


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l3340_334098

def A : ℕ := 123456
def B : ℕ := 142857
def M : ℕ := 1000009
def N : ℕ := 750298

theorem multiplicative_inverse_modulo :
  (A * B * N) % M = 1 :=
sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l3340_334098


namespace NUMINAMATH_CALUDE_sum_units_digits_734_99_347_83_l3340_334096

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of the units digits of 734^99 and 347^83 is 7 -/
theorem sum_units_digits_734_99_347_83 : 
  (unitsDigit (734^99) + unitsDigit (347^83)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_units_digits_734_99_347_83_l3340_334096


namespace NUMINAMATH_CALUDE_bank_account_problem_l3340_334013

/-- The bank account problem -/
theorem bank_account_problem (A E : ℝ) 
  (h1 : A > E)  -- Al has more money than Eliot
  (h2 : A - E = (A + E) / 12)  -- Difference is 1/12 of sum
  (h3 : A * 1.1 = E * 1.15 + 22)  -- After increase, Al has $22 more
  : E = 146.67 := by
  sorry

end NUMINAMATH_CALUDE_bank_account_problem_l3340_334013


namespace NUMINAMATH_CALUDE_square_root_of_nine_l3340_334061

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l3340_334061


namespace NUMINAMATH_CALUDE_cherry_pie_degrees_l3340_334099

theorem cherry_pie_degrees (total : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h1 : total = 40)
  (h2 : chocolate = 15)
  (h3 : apple = 10)
  (h4 : blueberry = 7)
  (h5 : (total - (chocolate + apple + blueberry)) % 2 = 0) :
  let remaining := total - (chocolate + apple + blueberry)
  let cherry := remaining / 2
  (cherry : ℚ) / total * 360 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_degrees_l3340_334099


namespace NUMINAMATH_CALUDE_expression_value_l3340_334040

theorem expression_value (x y : ℝ) (h : x^2 - 4*x - 1 = 0) :
  (2*x - 3)^2 - (x + y)*(x - y) - y^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3340_334040


namespace NUMINAMATH_CALUDE_unique_solution_for_digit_sum_equation_l3340_334066

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that 402 is the only solution to n(S(n) - 1) = 2010 -/
theorem unique_solution_for_digit_sum_equation :
  ∀ n : ℕ, n > 0 → (n * (S n - 1) = 2010) ↔ n = 402 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_digit_sum_equation_l3340_334066


namespace NUMINAMATH_CALUDE_thumbtack_solution_l3340_334060

/-- Represents the problem of calculating remaining thumbtacks --/
structure ThumbTackProblem where
  total_cans : Nat
  total_tacks : Nat
  boards_tested : Nat
  tacks_per_board : Nat

/-- Calculates the number of remaining thumbtacks in each can --/
def remaining_tacks (problem : ThumbTackProblem) : Nat :=
  (problem.total_tacks / problem.total_cans) - (problem.boards_tested * problem.tacks_per_board)

/-- Theorem stating the solution to the specific problem --/
theorem thumbtack_solution :
  let problem : ThumbTackProblem := {
    total_cans := 3,
    total_tacks := 450,
    boards_tested := 120,
    tacks_per_board := 1
  }
  remaining_tacks problem = 30 := by sorry


end NUMINAMATH_CALUDE_thumbtack_solution_l3340_334060


namespace NUMINAMATH_CALUDE_middle_number_proof_l3340_334059

theorem middle_number_proof (x y z : ℕ) : 
  x < y ∧ y < z ∧ 
  x + y = 22 ∧ 
  x + z = 29 ∧ 
  y + z = 31 ∧ 
  x = 10 → 
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_middle_number_proof_l3340_334059


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3340_334027

theorem greatest_value_quadratic_inequality :
  ∀ a : ℝ, -a^2 + 9*a - 14 ≥ 0 → a ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3340_334027


namespace NUMINAMATH_CALUDE_jumping_jacks_ratio_l3340_334065

/-- The ratio of Brooke's jumping jacks to Sidney's jumping jacks is 3:1 -/
theorem jumping_jacks_ratio : 
  let sidney_jj := [20, 36, 40, 50]
  let brooke_jj := 438
  (brooke_jj : ℚ) / (sidney_jj.sum : ℚ) = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_jumping_jacks_ratio_l3340_334065


namespace NUMINAMATH_CALUDE_find_r_l3340_334017

theorem find_r (m : ℝ) (r : ℝ) 
  (h1 : 5 = m * 3^r) 
  (h2 : 45 = m * 9^(2*r)) : 
  r = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l3340_334017


namespace NUMINAMATH_CALUDE_minimum_value_of_sum_of_reciprocals_l3340_334002

theorem minimum_value_of_sum_of_reciprocals (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_sum_of_reciprocals_l3340_334002


namespace NUMINAMATH_CALUDE_composition_equality_l3340_334048

theorem composition_equality (δ φ : ℝ → ℝ) (h1 : ∀ x, δ x = 5 * x + 6) (h2 : ∀ x, φ x = 7 * x + 4) :
  (∀ x, δ (φ x) = 1) ↔ (∀ x, x = -5/7) :=
by sorry

end NUMINAMATH_CALUDE_composition_equality_l3340_334048


namespace NUMINAMATH_CALUDE_mangoes_per_neighbor_l3340_334024

-- Define the given conditions
def total_mangoes : ℕ := 560
def mangoes_to_family : ℕ := 50
def num_neighbors : ℕ := 12

-- Define the relationship between x and total mangoes
def mangoes_sold (total : ℕ) : ℕ := total / 2

-- Theorem statement
theorem mangoes_per_neighbor : 
  (total_mangoes - mangoes_sold total_mangoes - mangoes_to_family) / num_neighbors = 19 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_per_neighbor_l3340_334024


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3340_334046

theorem simplify_and_evaluate (a b : ℝ) : 
  a = 2 * Real.sin (60 * π / 180) - 3 * Real.tan (45 * π / 180) →
  b = 3 →
  1 - (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3340_334046


namespace NUMINAMATH_CALUDE_inverse_of_three_mod_forty_l3340_334034

theorem inverse_of_three_mod_forty :
  ∃ x : ℕ, x < 40 ∧ (3 * x) % 40 = 1 :=
by
  use 27
  sorry

end NUMINAMATH_CALUDE_inverse_of_three_mod_forty_l3340_334034


namespace NUMINAMATH_CALUDE_first_neighbor_height_l3340_334049

/-- The height of Lucille's house in feet -/
def lucille_height : ℝ := 80

/-- The height of the second neighbor's house in feet -/
def neighbor2_height : ℝ := 99

/-- The height difference between Lucille's house and the average height in feet -/
def height_difference : ℝ := 3

/-- The height of the first neighbor's house in feet -/
def neighbor1_height : ℝ := 70

theorem first_neighbor_height :
  (lucille_height + neighbor1_height + neighbor2_height) / 3 - height_difference = lucille_height :=
by sorry

end NUMINAMATH_CALUDE_first_neighbor_height_l3340_334049


namespace NUMINAMATH_CALUDE_sum_inequality_l3340_334036

theorem sum_inequality (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3340_334036


namespace NUMINAMATH_CALUDE_bicycle_distance_l3340_334019

/-- Proves that a bicycle traveling 1/2 as fast as a motorcycle moving at 40 miles per hour
    will cover a distance of 10 miles in 30 minutes. -/
theorem bicycle_distance (motorcycle_speed : ℝ) (bicycle_speed_ratio : ℝ) (time : ℝ) :
  motorcycle_speed = 40 →
  bicycle_speed_ratio = (1 : ℝ) / 2 →
  time = 30 / 60 →
  (bicycle_speed_ratio * motorcycle_speed) * time = 10 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_distance_l3340_334019


namespace NUMINAMATH_CALUDE_percentage_equality_l3340_334090

theorem percentage_equality (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (20 / 100) * (x + y) →
  y = (50 / 100) * x →
  P = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_equality_l3340_334090


namespace NUMINAMATH_CALUDE_birds_and_storks_l3340_334072

theorem birds_and_storks (initial_birds : ℕ) (initial_storks : ℕ) (joining_storks : ℕ) :
  initial_birds = 6 →
  initial_storks = 3 →
  joining_storks = 2 →
  initial_birds - (initial_storks + joining_storks) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_l3340_334072


namespace NUMINAMATH_CALUDE_parallel_implies_t_half_magnitude_when_t_one_l3340_334078

-- Define the vectors a and b as functions of t
def a (t : ℝ) : Fin 2 → ℝ := ![2 - t, 3]
def b (t : ℝ) : Fin 2 → ℝ := ![t, 1]

-- Theorem 1: If a and b are parallel, then t = 1/2
theorem parallel_implies_t_half :
  ∀ t : ℝ, (∃ k : ℝ, a t = k • b t) → t = 1/2 := by sorry

-- Theorem 2: When t = 1, |a - 4b| = √10
theorem magnitude_when_t_one :
  ‖(a 1) - 4 • (b 1)‖ = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_parallel_implies_t_half_magnitude_when_t_one_l3340_334078


namespace NUMINAMATH_CALUDE_common_tangents_exist_l3340_334064

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a line is a common tangent to two circles -/
def isCommonTangent (l : Line) (c1 c2 : Circle) : Prop := 
  isTangent l c1 ∧ isTangent l c2

/-- The line connecting the centers of two circles -/
def centerLine (c1 c2 : Circle) : Line := sorry

/-- Checks if a line intersects another line -/
def intersects (l1 l2 : Line) : Prop := sorry

/-- Theorem: For any two circles, there exist common tangents in two cases -/
theorem common_tangents_exist (c1 c2 : Circle) : 
  ∃ (l1 l2 : Line), 
    (isCommonTangent l1 c1 c2 ∧ ¬intersects l1 (centerLine c1 c2)) ∧
    (isCommonTangent l2 c1 c2 ∧ intersects l2 (centerLine c1 c2)) := by
  sorry

end NUMINAMATH_CALUDE_common_tangents_exist_l3340_334064


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3340_334041

-- Define the concept of opposite for integers
def opposite (n : ℤ) : ℤ := -n

-- Theorem statement
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3340_334041


namespace NUMINAMATH_CALUDE_tshirt_pricing_l3340_334007

theorem tshirt_pricing (first_batch_cost second_batch_cost : ℕ)
  (quantity_ratio : ℚ) (price_difference : ℕ) (first_batch_selling_price : ℕ)
  (min_total_profit : ℕ) :
  first_batch_cost = 4000 →
  second_batch_cost = 5400 →
  quantity_ratio = 3/2 →
  price_difference = 5 →
  first_batch_selling_price = 70 →
  min_total_profit = 4060 →
  ∃ (first_batch_unit_cost : ℕ) (second_batch_min_selling_price : ℕ),
    first_batch_unit_cost = 50 ∧
    second_batch_min_selling_price = 66 ∧
    (second_batch_cost : ℚ) / (first_batch_unit_cost - price_difference) = quantity_ratio * ((first_batch_cost : ℚ) / first_batch_unit_cost) ∧
    (first_batch_selling_price - first_batch_unit_cost) * (first_batch_cost / first_batch_unit_cost) +
    (second_batch_min_selling_price - (first_batch_unit_cost - price_difference)) * (second_batch_cost / (first_batch_unit_cost - price_difference)) ≥ min_total_profit :=
by sorry

end NUMINAMATH_CALUDE_tshirt_pricing_l3340_334007


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3340_334021

/-- Given a quadratic equation of the form (x^2 - bx + b^2) / (ax^2 - c) = (m-1) / (m+1),
    if the roots are numerically equal but of opposite signs, and c = b^2,
    then m = (a-1) / (a+1) -/
theorem quadratic_equation_roots (a b m : ℝ) :
  (∃ x y : ℝ, x = -y ∧ x ≠ 0 ∧
    (x^2 - b*x + b^2) / (a*x^2 - b^2) = (m-1) / (m+1) ∧
    (y^2 - b*y + b^2) / (a*y^2 - b^2) = (m-1) / (m+1)) →
  m = (a-1) / (a+1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3340_334021


namespace NUMINAMATH_CALUDE_overlapping_rectangles_area_l3340_334004

/-- Given two overlapping rectangles, prove the area of the non-overlapping part of one rectangle -/
theorem overlapping_rectangles_area (a b c d overlap : ℕ) : 
  a * b = 80 → 
  c * d = 108 → 
  overlap = 37 → 
  c * d - (a * b - overlap) = 65 := by
sorry

end NUMINAMATH_CALUDE_overlapping_rectangles_area_l3340_334004


namespace NUMINAMATH_CALUDE_replacement_preserves_mean_and_variance_l3340_334053

def initial_set : List ℤ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
def new_set : List ℤ := [-5, -5, -3, -2, -1, 0, 1, 1, 2, 3, 4, 5]

def mean (s : List ℤ) : ℚ := (s.sum : ℚ) / s.length

def variance (s : List ℤ) : ℚ :=
  let m := mean s
  (s.map (λ x => ((x : ℚ) - m) ^ 2)).sum / s.length

theorem replacement_preserves_mean_and_variance :
  mean initial_set = mean new_set ∧ variance initial_set = variance new_set :=
sorry

end NUMINAMATH_CALUDE_replacement_preserves_mean_and_variance_l3340_334053


namespace NUMINAMATH_CALUDE_number_1349_is_valid_l3340_334031

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  (n / 100 % 10 = 3 * (n / 1000)) ∧
  (n % 10 = 3 * (n / 100 % 10))

theorem number_1349_is_valid : is_valid_number 1349 := by
  sorry

end NUMINAMATH_CALUDE_number_1349_is_valid_l3340_334031


namespace NUMINAMATH_CALUDE_line_point_k_value_l3340_334042

/-- A line contains the points (3,5), (1,k), and (7,9). Prove that k = 3. -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), m * 3 + b = 5 ∧ m * 1 + b = k ∧ m * 7 + b = 9) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_value_l3340_334042


namespace NUMINAMATH_CALUDE_multiplicative_inverse_154_mod_257_l3340_334028

theorem multiplicative_inverse_154_mod_257 : ∃ x : ℕ, x < 257 ∧ (154 * x) % 257 = 1 :=
  by
    use 20
    sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_154_mod_257_l3340_334028


namespace NUMINAMATH_CALUDE_absolute_value_equals_negative_l3340_334094

theorem absolute_value_equals_negative (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negative_l3340_334094


namespace NUMINAMATH_CALUDE_fraction_value_implies_m_l3340_334043

theorem fraction_value_implies_m (m : ℚ) : (m - 5) / m = 2 → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_implies_m_l3340_334043


namespace NUMINAMATH_CALUDE_smallest_x_power_inequality_l3340_334092

theorem smallest_x_power_inequality : 
  ∃ x : ℕ, (∀ y : ℕ, 27^y > 3^24 → x ≤ y) ∧ 27^x > 3^24 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_power_inequality_l3340_334092


namespace NUMINAMATH_CALUDE_binary_sum_equals_116_l3340_334026

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010101₂ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The binary representation of 11111₂ -/
def binary2 : List Bool := [true, true, true, true, true]

/-- Theorem stating that the sum of 1010101₂ and 11111₂ in decimal is 116 -/
theorem binary_sum_equals_116 : 
  binary_to_decimal binary1 + binary_to_decimal binary2 = 116 := by
  sorry


end NUMINAMATH_CALUDE_binary_sum_equals_116_l3340_334026


namespace NUMINAMATH_CALUDE_chairs_to_remove_l3340_334087

/-- The number of chairs in each row -/
def chairs_per_row : ℕ := 15

/-- The initial number of chairs set up -/
def initial_chairs : ℕ := 225

/-- The number of expected attendees -/
def expected_attendees : ℕ := 180

/-- Theorem: The number of chairs to be removed is 45 -/
theorem chairs_to_remove :
  ∃ (removed : ℕ),
    removed = initial_chairs - expected_attendees ∧
    removed % chairs_per_row = 0 ∧
    (initial_chairs - removed) ≥ expected_attendees ∧
    (initial_chairs - removed) % chairs_per_row = 0 ∧
    removed = 45 := by
  sorry

end NUMINAMATH_CALUDE_chairs_to_remove_l3340_334087


namespace NUMINAMATH_CALUDE_rods_in_mile_l3340_334050

/-- Represents the number of furlongs in a mile -/
def furlongs_per_mile : ℕ := 10

/-- Represents the number of rods in a furlong -/
def rods_per_furlong : ℕ := 50

/-- Theorem stating that one mile is equal to 500 rods -/
theorem rods_in_mile : furlongs_per_mile * rods_per_furlong = 500 := by
  sorry

end NUMINAMATH_CALUDE_rods_in_mile_l3340_334050


namespace NUMINAMATH_CALUDE_tech_club_enrollment_l3340_334088

theorem tech_club_enrollment (total : ℕ) (cs : ℕ) (electronics : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : cs = 90)
  (h3 : electronics = 60)
  (h4 : both = 20) :
  total - (cs + electronics - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tech_club_enrollment_l3340_334088


namespace NUMINAMATH_CALUDE_circle_properties_l3340_334003

theorem circle_properties (k : ℚ) : 
  let circle_eq (x y : ℚ) := x^2 + 2*x + y^2 = 1992
  ∃ (x y : ℚ), 
    circle_eq 42 12 ∧ 
    circle_eq x y ∧ 
    y - 12 = k * (x - 42) := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3340_334003


namespace NUMINAMATH_CALUDE_alien_sequence_valid_l3340_334038

/-- Represents a symbol in the alien sequence -/
inductive AlienSymbol
| percent
| exclamation
| ampersand
| plus
| zero

/-- Represents the possible operations -/
inductive Operation
| addition
| subtraction
| multiplication
| division
| exponentiation

/-- Represents a mapping of symbols to digits or operations -/
structure SymbolMapping where
  base : ℕ
  digit_map : AlienSymbol → Fin base
  operation : AlienSymbol → Option Operation
  equality : AlienSymbol

/-- Converts a list of alien symbols to a natural number given a symbol mapping -/
def alien_to_nat (mapping : SymbolMapping) (symbols : List AlienSymbol) : ℕ := sorry

/-- Checks if a list of alien symbols represents a valid equation given a symbol mapping -/
def is_valid_equation (mapping : SymbolMapping) (symbols : List AlienSymbol) : Prop := sorry

/-- The alien sequence -/
def alien_sequence : List AlienSymbol :=
  [AlienSymbol.percent, AlienSymbol.exclamation, AlienSymbol.ampersand,
   AlienSymbol.plus, AlienSymbol.exclamation, AlienSymbol.zero,
   AlienSymbol.plus, AlienSymbol.plus, AlienSymbol.exclamation,
   AlienSymbol.exclamation, AlienSymbol.exclamation]

theorem alien_sequence_valid :
  ∃ (mapping : SymbolMapping), is_valid_equation mapping alien_sequence := by
  sorry

#check alien_sequence_valid

end NUMINAMATH_CALUDE_alien_sequence_valid_l3340_334038


namespace NUMINAMATH_CALUDE_square_less_than_triple_l3340_334054

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l3340_334054


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3340_334083

theorem at_least_one_greater_than_one (x y : ℝ) (h : x + y > 2) : max x y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3340_334083


namespace NUMINAMATH_CALUDE_company_employees_l3340_334093

theorem company_employees (december_employees : ℕ) (increase_percentage : ℚ) :
  december_employees = 470 →
  increase_percentage = 15 / 100 →
  ∃ (january_employees : ℕ),
    (january_employees : ℚ) * (1 + increase_percentage) = december_employees ∧
    january_employees = 409 :=
by sorry

end NUMINAMATH_CALUDE_company_employees_l3340_334093


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3340_334016

def P (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + b*x + c

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x, ∃ q, P a b c x = (x - 1)^3 * q) ↔ (a = -6 ∧ b = 8 ∧ c = -3) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3340_334016


namespace NUMINAMATH_CALUDE_solution_set_theorem_l3340_334056

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def f_def (f : ℝ → ℝ) : Prop :=
  ∀ x ≥ 0, f x = x^2 - 2*x

theorem solution_set_theorem (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_def : f_def f) : 
  {x : ℝ | f (x + 1) < 3} = Set.Ioo (-4 : ℝ) 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l3340_334056


namespace NUMINAMATH_CALUDE_unique_b_c_solution_l3340_334057

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

-- State the theorem
theorem unique_b_c_solution :
  ∃! (b c : ℝ), 
    (∃ a : ℝ, A a ≠ B b c) ∧ 
    (∃ a : ℝ, A a ∪ B b c = {-3, 4}) ∧
    (∃ a : ℝ, A a ∩ B b c = {-3}) ∧
    b = 6 ∧ c = 9 := by
  sorry


end NUMINAMATH_CALUDE_unique_b_c_solution_l3340_334057


namespace NUMINAMATH_CALUDE_stone_slab_area_l3340_334091

/-- Given 50 square stone slabs with a length of 120 cm each, 
    the total floor area covered is 72 square meters. -/
theorem stone_slab_area (n : ℕ) (length_cm : ℝ) (total_area_m2 : ℝ) : 
  n = 50 → 
  length_cm = 120 → 
  total_area_m2 = (n * (length_cm / 100)^2) → 
  total_area_m2 = 72 := by
sorry

end NUMINAMATH_CALUDE_stone_slab_area_l3340_334091


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3340_334039

theorem trigonometric_identity (α : Real) : 
  (Real.sin (45 * π / 180 + α))^2 - (Real.sin (30 * π / 180 - α))^2 - 
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180 + 2 * α) = 
  Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3340_334039


namespace NUMINAMATH_CALUDE_roberto_outfits_l3340_334095

/-- Represents the number of different outfits Roberto can create --/
def number_of_outfits (trousers shirts jackets constrained_trousers constrained_jackets : ℕ) : ℕ :=
  ((trousers - constrained_trousers) * jackets + constrained_trousers * constrained_jackets) * shirts

/-- Theorem stating the number of outfits Roberto can create given his wardrobe constraints --/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 7
  let jackets : ℕ := 3
  let constrained_trousers : ℕ := 2
  let constrained_jackets : ℕ := 2
  number_of_outfits trousers shirts jackets constrained_trousers constrained_jackets = 91 :=
by
  sorry


end NUMINAMATH_CALUDE_roberto_outfits_l3340_334095


namespace NUMINAMATH_CALUDE_max_value_a_squared_b_l3340_334045

theorem max_value_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) :
  a^2 * b ≤ 54 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ * (a₀ + b₀) = 27 ∧ a₀^2 * b₀ = 54 := by
  sorry

end NUMINAMATH_CALUDE_max_value_a_squared_b_l3340_334045


namespace NUMINAMATH_CALUDE_egg_problem_l3340_334063

/-- The initial number of eggs in the basket -/
def initial_eggs : ℕ := 120

/-- The number of broken eggs -/
def broken_eggs : ℕ := 20

/-- The total price in fillérs -/
def total_price : ℕ := 600

/-- Proves that the initial number of eggs was 120 -/
theorem egg_problem :
  initial_eggs = 120 ∧
  broken_eggs = 20 ∧
  total_price = 600 ∧
  (total_price : ℚ) / initial_eggs + 1 = total_price / (initial_eggs - broken_eggs) :=
by sorry

end NUMINAMATH_CALUDE_egg_problem_l3340_334063


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3340_334069

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_problem : 
  Nat.gcd (factorial 7) ((factorial 10) / (factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3340_334069


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l3340_334023

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 : ℚ) + ((1 / 3 : ℚ) - (1 / 8 : ℚ)) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l3340_334023


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3340_334047

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3340_334047


namespace NUMINAMATH_CALUDE_max_white_pieces_correct_l3340_334055

/-- Represents a game board with m rows and n columns -/
structure Board (m n : ℕ) where
  white_pieces : Finset (ℕ × ℕ)
  no_same_row_col : ∀ (i j k l : ℕ), (i, j) ∈ white_pieces → (k, l) ∈ white_pieces → i = k ∨ j = l → (i, j) = (k, l)

/-- The maximum number of white pieces that can be placed on the board -/
def max_white_pieces (m n : ℕ) : ℕ := m + n - 1

/-- Theorem stating that the maximum number of white pieces is m + n - 1 -/
theorem max_white_pieces_correct (m n : ℕ) :
  ∀ (b : Board m n), b.white_pieces.card ≤ max_white_pieces m n :=
by sorry

end NUMINAMATH_CALUDE_max_white_pieces_correct_l3340_334055


namespace NUMINAMATH_CALUDE_stereo_trade_in_value_l3340_334097

theorem stereo_trade_in_value (old_cost new_cost discount_percent out_of_pocket : ℚ) 
  (h1 : old_cost = 250)
  (h2 : new_cost = 600)
  (h3 : discount_percent = 25)
  (h4 : out_of_pocket = 250) :
  let discounted_price := new_cost * (1 - discount_percent / 100)
  let trade_in_value := discounted_price - out_of_pocket
  trade_in_value / old_cost * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_stereo_trade_in_value_l3340_334097


namespace NUMINAMATH_CALUDE_expression_simplification_l3340_334071

theorem expression_simplification (x : ℝ) : 
  3*x - 4*(2 + x^2) + 5*(3 - x) - 6*(1 - 2*x + x^2) = 10*x - 10*x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3340_334071


namespace NUMINAMATH_CALUDE_boys_camp_total_l3340_334006

theorem boys_camp_total (total : ℝ) 
  (h1 : 0.2 * total = total_school_a)
  (h2 : 0.3 * total_school_a = science_school_a)
  (h3 : total_school_a - science_school_a = 28) : 
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l3340_334006


namespace NUMINAMATH_CALUDE_clock_problem_l3340_334008

/-- Represents the original cost of the clock to the shop -/
def original_cost : ℝ := 250

/-- Represents the first selling price of the clock -/
def first_sell_price : ℝ := original_cost * 1.2

/-- Represents the buy-back price of the clock -/
def buy_back_price : ℝ := first_sell_price * 0.5

/-- Represents the second selling price of the clock -/
def second_sell_price : ℝ := buy_back_price * 1.8

theorem clock_problem :
  (original_cost - buy_back_price = 100) →
  (second_sell_price = 270) := by
  sorry

end NUMINAMATH_CALUDE_clock_problem_l3340_334008


namespace NUMINAMATH_CALUDE_log_inequality_characterization_l3340_334081

theorem log_inequality_characterization (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha_neq_1 : a ≠ 1) :
  (Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)) ↔
  (b = 1 ∧ a ≠ 1) ∨ (a > b ∧ b > 1) ∨ (b > 1 ∧ 1 > a) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_characterization_l3340_334081


namespace NUMINAMATH_CALUDE_x_minus_y_squared_l3340_334058

theorem x_minus_y_squared (x y : ℝ) : 
  y = Real.sqrt (2 * x - 3) + Real.sqrt (3 - 2 * x) - 4 →
  x = 3 / 2 →
  x - y^2 = -29 / 2 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_squared_l3340_334058


namespace NUMINAMATH_CALUDE_value_of_a_l3340_334010

def f (x : ℝ) : ℝ := 3 * x - 1

def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a, 5}

theorem value_of_a : ∃ a : ℝ, (∀ x ∈ A a, f x ∈ B a) ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_value_of_a_l3340_334010


namespace NUMINAMATH_CALUDE_retailer_profit_calculation_l3340_334044

/-- Calculates the actual profit percentage for a retailer who marks up goods
    by a certain percentage and then offers a discount. -/
theorem retailer_profit_calculation 
  (cost_price : ℝ) 
  (markup_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (markup_percentage_is_40 : markup_percentage = 40)
  (discount_percentage_is_25 : discount_percentage = 25)
  : let marked_price := cost_price * (1 + markup_percentage / 100)
    let selling_price := marked_price * (1 - discount_percentage / 100)
    let profit := selling_price - cost_price
    let profit_percentage := (profit / cost_price) * 100
    profit_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_calculation_l3340_334044


namespace NUMINAMATH_CALUDE_hyperbola_and_parabola_properties_l3340_334037

-- Define the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the parabola
def parabola_equation (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem hyperbola_and_parabola_properties :
  -- Length of real axis
  (∃ a : ℝ, a = 3 ∧ 2 * a = 6) ∧
  -- Length of imaginary axis
  (∃ b : ℝ, b = 4 ∧ 2 * b = 8) ∧
  -- Eccentricity
  (∃ e : ℝ, e = 5 / 3) ∧
  -- Parabola equation
  (∀ x y : ℝ, hyperbola_equation x y →
    (x = 0 ∧ y = 0 → parabola_equation x y) ∧
    (x = -3 ∧ y = 0 → parabola_equation x y)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_parabola_properties_l3340_334037


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3340_334080

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, 1 < x ∧ x < Real.pi / 2 → (x - 1) * Real.tan x > 0) ∧
  (∃ x : ℝ, (x - 1) * Real.tan x > 0 ∧ ¬(1 < x ∧ x < Real.pi / 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3340_334080


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l3340_334076

theorem percentage_equation_solution :
  ∃ x : ℝ, (12.4 * 350) + (9.9 * 275) = (8.6 * x) + (5.3 * (2250 - x)) := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l3340_334076


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l3340_334009

-- Define the function f
def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x - 3)

-- State the theorem
theorem tangent_slope_at_zero :
  (deriv f) 0 = -6 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l3340_334009


namespace NUMINAMATH_CALUDE_coffee_break_probabilities_l3340_334025

/-- Represents the state of knowledge among scientists -/
structure ScientistGroup where
  total : Nat
  initial_knowers : Nat
  
/-- Represents the outcome after the coffee break -/
structure CoffeeBreakOutcome where
  final_knowers : Nat

/-- Probability of a specific outcome after the coffee break -/
def probability_of_outcome (group : ScientistGroup) (outcome : CoffeeBreakOutcome) : ℚ :=
  sorry

/-- Expected number of scientists who know the news after the coffee break -/
def expected_final_knowers (group : ScientistGroup) : ℚ :=
  sorry

theorem coffee_break_probabilities (group : ScientistGroup) 
  (h1 : group.total = 18) 
  (h2 : group.initial_knowers = 10) : 
  probability_of_outcome group ⟨13⟩ = 0 ∧ 
  probability_of_outcome group ⟨14⟩ = 1120 / 2431 ∧
  expected_final_knowers group = 14 + 12 / 17 :=
  sorry

end NUMINAMATH_CALUDE_coffee_break_probabilities_l3340_334025


namespace NUMINAMATH_CALUDE_bryden_payment_is_correct_l3340_334022

/-- The face value of a state quarter in dollars -/
def quarter_value : ℝ := 0.25

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 6

/-- The percentage of face value the collector offers, expressed as a decimal -/
def collector_offer_percentage : ℝ := 16

/-- The discount percentage applied to the total payment, expressed as a decimal -/
def discount_percentage : ℝ := 0.1

/-- The amount Bryden receives for his state quarters -/
def bryden_payment : ℝ :=
  (bryden_quarters : ℝ) * quarter_value * collector_offer_percentage * (1 - discount_percentage)

theorem bryden_payment_is_correct :
  bryden_payment = 21.6 := by sorry

end NUMINAMATH_CALUDE_bryden_payment_is_correct_l3340_334022


namespace NUMINAMATH_CALUDE_teacher_student_arrangements_l3340_334005

/-- The number of arrangements for 4 teachers and 4 students in various scenarios -/
theorem teacher_student_arrangements :
  let n_teachers : ℕ := 4
  let n_students : ℕ := 4
  let factorial (n : ℕ) : ℕ := Nat.factorial n
  let choose (n k : ℕ) : ℕ := Nat.choose n k
  
  -- (I) Students stand together
  (factorial (n_teachers + 1) * factorial n_students = 2880) ∧
  
  -- (II) No two students adjacent
  (factorial n_teachers * choose (n_teachers + 1) n_students * factorial n_students = 2880) ∧
  
  -- (III) Teachers and students alternate
  (2 * factorial n_teachers * factorial n_students = 1152) :=
by sorry

end NUMINAMATH_CALUDE_teacher_student_arrangements_l3340_334005


namespace NUMINAMATH_CALUDE_pollen_diameter_scientific_notation_l3340_334073

/-- Expresses a given number in scientific notation -/
def scientific_notation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem pollen_diameter_scientific_notation :
  scientific_notation 0.0000021 = (2.1, -6) :=
sorry

end NUMINAMATH_CALUDE_pollen_diameter_scientific_notation_l3340_334073


namespace NUMINAMATH_CALUDE_power_of_six_with_nine_tens_digit_l3340_334084

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem power_of_six_with_nine_tens_digit :
  ∃ (k : ℕ), k > 0 ∧ tens_digit (6^k) = 9 ∧ ∀ (m : ℕ), m > 0 ∧ m < k → tens_digit (6^m) ≠ 9 :=
sorry

end NUMINAMATH_CALUDE_power_of_six_with_nine_tens_digit_l3340_334084


namespace NUMINAMATH_CALUDE_fourth_student_in_sample_l3340_334075

def systematic_sample (total_students : ℕ) (sample_size : ℕ) (sample : Finset ℕ) : Prop :=
  sample.card = sample_size ∧
  ∃ k : ℕ, ∀ i ∈ sample, ∃ j : ℕ, i = 1 + j * (total_students / sample_size)

theorem fourth_student_in_sample 
  (total_students : ℕ) (sample_size : ℕ) (sample : Finset ℕ) 
  (h1 : total_students = 52)
  (h2 : sample_size = 4)
  (h3 : 3 ∈ sample)
  (h4 : 29 ∈ sample)
  (h5 : 42 ∈ sample)
  (h6 : systematic_sample total_students sample_size sample) :
  16 ∈ sample :=
sorry

end NUMINAMATH_CALUDE_fourth_student_in_sample_l3340_334075


namespace NUMINAMATH_CALUDE_parallelogram_area_l3340_334035

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 12) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 60 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3340_334035


namespace NUMINAMATH_CALUDE_max_score_is_43_l3340_334012

-- Define the sightseeing point type
structure SightseeingPoint where
  score : ℕ
  time : ℚ
  cost : ℕ

-- Define the list of sightseeing points
def sightseeingPoints : List SightseeingPoint := [
  ⟨10, 2/3, 1000⟩,
  ⟨7, 1/2, 700⟩,
  ⟨6, 1/3, 300⟩,
  ⟨8, 2/3, 800⟩,
  ⟨5, 1/4, 200⟩,
  ⟨9, 2/3, 900⟩,
  ⟨8, 1/2, 900⟩,
  ⟨8, 2/5, 600⟩,
  ⟨5, 1/5, 400⟩,
  ⟨6, 1/4, 600⟩
]

-- Define a function to check if a selection of points is valid
def isValidSelection (selection : List Bool) : Prop :=
  let selectedPoints := List.zipWith (λ s p => if s then some p else none) selection sightseeingPoints
  let totalTime := selectedPoints.filterMap id |>.map SightseeingPoint.time |>.sum
  let totalCost := selectedPoints.filterMap id |>.map SightseeingPoint.cost |>.sum
  totalTime < 3 ∧ totalCost ≤ 3500

-- Define a function to calculate the total score of a selection
def totalScore (selection : List Bool) : ℕ :=
  let selectedPoints := List.zipWith (λ s p => if s then some p else none) selection sightseeingPoints
  selectedPoints.filterMap id |>.map SightseeingPoint.score |>.sum

-- State the theorem
theorem max_score_is_43 :
  ∃ (selection : List Bool),
    selection.length = sightseeingPoints.length ∧
    isValidSelection selection ∧
    totalScore selection = 43 ∧
    ∀ (otherSelection : List Bool),
      otherSelection.length = sightseeingPoints.length →
      isValidSelection otherSelection →
      totalScore otherSelection ≤ 43 := by
  sorry

end NUMINAMATH_CALUDE_max_score_is_43_l3340_334012


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3340_334000

theorem complex_fraction_equality : Complex.I * 2 / (1 + Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3340_334000


namespace NUMINAMATH_CALUDE_xyz_equals_four_l3340_334001

theorem xyz_equals_four (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 4 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_four_l3340_334001


namespace NUMINAMATH_CALUDE_goals_theorem_l3340_334070

def goals_problem (bruce_goals michael_goals jack_goals sarah_goals : ℕ) : Prop :=
  bruce_goals = 4 ∧
  michael_goals = 2 * bruce_goals ∧
  jack_goals = bruce_goals - 1 ∧
  sarah_goals = jack_goals / 2 ∧
  michael_goals + jack_goals + sarah_goals = 12

theorem goals_theorem :
  ∃ (bruce_goals michael_goals jack_goals sarah_goals : ℕ),
    goals_problem bruce_goals michael_goals jack_goals sarah_goals :=
by
  sorry

end NUMINAMATH_CALUDE_goals_theorem_l3340_334070


namespace NUMINAMATH_CALUDE_mandarin_ducks_count_l3340_334014

/-- The number of pairs of mandarin ducks -/
def num_pairs : ℕ := 3

/-- The number of ducks in each pair -/
def ducks_per_pair : ℕ := 2

/-- The total number of mandarin ducks -/
def total_ducks : ℕ := num_pairs * ducks_per_pair

theorem mandarin_ducks_count : total_ducks = 6 := by
  sorry

end NUMINAMATH_CALUDE_mandarin_ducks_count_l3340_334014


namespace NUMINAMATH_CALUDE_solve_system_l3340_334029

theorem solve_system (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 3 + 1 / x) :
  y = 3 / 2 + Real.sqrt 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3340_334029


namespace NUMINAMATH_CALUDE_cricket_match_playtime_l3340_334018

-- Define the total duration of the match in minutes
def total_duration : ℕ := 12 * 60 + 35

-- Define the lunch break duration in minutes
def lunch_break : ℕ := 15

-- Theorem to prove the actual playtime
theorem cricket_match_playtime :
  total_duration - lunch_break = 740 := by
  sorry

end NUMINAMATH_CALUDE_cricket_match_playtime_l3340_334018


namespace NUMINAMATH_CALUDE_total_eyes_in_pond_l3340_334067

/-- The number of eyes an animal has -/
def eyes_per_animal : ℕ := 2

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- The number of crocodiles in the pond -/
def num_crocodiles : ℕ := 6

/-- The total number of animals in the pond -/
def total_animals : ℕ := num_frogs + num_crocodiles

/-- Theorem: The total number of animal eyes in the pond is 52 -/
theorem total_eyes_in_pond : num_frogs * eyes_per_animal + num_crocodiles * eyes_per_animal = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_eyes_in_pond_l3340_334067


namespace NUMINAMATH_CALUDE_pages_copied_example_l3340_334079

/-- Given a cost per page in cents, a flat service charge in cents, and a total budget in cents,
    calculate the maximum number of pages that can be copied. -/
def max_pages_copied (cost_per_page : ℕ) (service_charge : ℕ) (total_budget : ℕ) : ℕ :=
  (total_budget - service_charge) / cost_per_page

/-- Prove that with a cost of 3 cents per page, a flat service charge of 500 cents,
    and a total budget of 5000 cents, the maximum number of pages that can be copied is 1500. -/
theorem pages_copied_example : max_pages_copied 3 500 5000 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_example_l3340_334079


namespace NUMINAMATH_CALUDE_part_one_part_two_l3340_334074

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |2*x + a|

-- Part I
theorem part_one : 
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 2} := by sorry

-- Part II
theorem part_two : 
  (∃ x₀ : ℝ, f a x₀ + |x₀ - 2| < 3) → -7 < a ∧ a < -1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3340_334074


namespace NUMINAMATH_CALUDE_prob_three_green_in_seven_trials_l3340_334062

/-- The number of green marbles -/
def green_marbles : ℕ := 8

/-- The number of purple marbles -/
def purple_marbles : ℕ := 4

/-- The total number of marbles -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The number of successful trials (picking green marbles) -/
def num_success : ℕ := 3

/-- The probability of picking a green marble in a single trial -/
def prob_green : ℚ := green_marbles / total_marbles

/-- The probability of picking a purple marble in a single trial -/
def prob_purple : ℚ := purple_marbles / total_marbles

/-- The probability of picking exactly three green marbles in seven trials -/
theorem prob_three_green_in_seven_trials :
  (Nat.choose num_trials num_success : ℚ) * prob_green ^ num_success * prob_purple ^ (num_trials - num_success) = 280 / 729 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_green_in_seven_trials_l3340_334062


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3340_334020

theorem absolute_value_simplification (x : ℝ) (h : x < -2) : 1 - |1 + x| = -2 - x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3340_334020


namespace NUMINAMATH_CALUDE_dragon_population_l3340_334052

theorem dragon_population (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 117) 
  (h2 : total_legs = 108) : 
  ∃ (three_headed six_headed : ℕ), 
    three_headed = 15 ∧ 
    six_headed = 12 ∧ 
    3 * three_headed + 6 * six_headed = total_heads ∧ 
    4 * (three_headed + six_headed) = total_legs :=
by sorry

end NUMINAMATH_CALUDE_dragon_population_l3340_334052


namespace NUMINAMATH_CALUDE_consecutive_odd_product_l3340_334089

theorem consecutive_odd_product (m : ℕ) (N : ℤ) : 
  Odd N → 
  N = (m - 1) * m * (m + 1) - ((m - 1) + m + (m + 1)) → 
  (∃ k : ℕ, m = 2 * k + 1) ∧ 
  N = (m - 2) * m * (m + 2) ∧ 
  Odd (m - 2) ∧ Odd m ∧ Odd (m + 2) :=
sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_l3340_334089


namespace NUMINAMATH_CALUDE_binary_representation_sqrt_theorem_l3340_334011

/-- Given a positive integer d that is not a perfect square, s(n) denotes the number of digits 1 
    among the first n digits in the binary representation of √d -/
def s (d : ℕ+) (n : ℕ) : ℕ := sorry

/-- The theorem states that for a positive integer d that is not a perfect square, 
    there exists an integer A such that for all integers n ≥ A, 
    s(n) > √(2n) - 2 -/
theorem binary_representation_sqrt_theorem (d : ℕ+) 
    (h : ∀ (m : ℕ), m * m ≠ d) : 
    ∃ A : ℕ, ∀ n : ℕ, n ≥ A → s d n > Real.sqrt (2 * n) - 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_sqrt_theorem_l3340_334011
