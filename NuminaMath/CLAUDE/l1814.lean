import Mathlib

namespace NUMINAMATH_CALUDE_inequality_system_solutions_l1814_181482

theorem inequality_system_solutions : 
  {x : ℤ | x ≥ 0 ∧ 5*x - 1 < 3*(x + 1) ∧ (1 - x) / 3 ≤ 1} = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l1814_181482


namespace NUMINAMATH_CALUDE_bicycles_in_garage_l1814_181442

theorem bicycles_in_garage (cars : ℕ) (total_wheels : ℕ) (bicycle_wheels : ℕ) (car_wheels : ℕ) : 
  cars = 16 → 
  total_wheels = 82 → 
  bicycle_wheels = 2 → 
  car_wheels = 4 → 
  ∃ bicycles : ℕ, bicycles * bicycle_wheels + cars * car_wheels = total_wheels ∧ bicycles = 9 :=
by sorry

end NUMINAMATH_CALUDE_bicycles_in_garage_l1814_181442


namespace NUMINAMATH_CALUDE_common_elements_count_l1814_181427

def S := Finset.range 2005
def T := Finset.range 2005

def multiples_of_4 (n : ℕ) : ℕ := (n + 1) * 4
def multiples_of_6 (n : ℕ) : ℕ := (n + 1) * 6

def S_set := S.image multiples_of_4
def T_set := T.image multiples_of_6

theorem common_elements_count : (S_set ∩ T_set).card = 668 := by
  sorry

end NUMINAMATH_CALUDE_common_elements_count_l1814_181427


namespace NUMINAMATH_CALUDE_additional_group_average_weight_l1814_181476

theorem additional_group_average_weight 
  (initial_count : ℕ) 
  (additional_count : ℕ) 
  (weight_increase : ℝ) 
  (final_average : ℝ) : 
  initial_count = 30 →
  additional_count = 30 →
  weight_increase = 10 →
  final_average = 40 →
  let total_count := initial_count + additional_count
  let initial_average := final_average - weight_increase
  let initial_total_weight := initial_count * initial_average
  let final_total_weight := total_count * final_average
  let additional_total_weight := final_total_weight - initial_total_weight
  additional_total_weight / additional_count = 50 := by
sorry

end NUMINAMATH_CALUDE_additional_group_average_weight_l1814_181476


namespace NUMINAMATH_CALUDE_evaluate_expression_l1814_181459

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 - 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1814_181459


namespace NUMINAMATH_CALUDE_cloth_seller_gain_percentage_l1814_181454

/-- Calculates the gain percentage for a cloth seller -/
theorem cloth_seller_gain_percentage 
  (total_cloth : ℝ) 
  (profit_cloth : ℝ) 
  (total_cloth_positive : total_cloth > 0)
  (profit_ratio : profit_cloth = total_cloth / 3) :
  (profit_cloth / total_cloth) * 100 = 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_cloth_seller_gain_percentage_l1814_181454


namespace NUMINAMATH_CALUDE_fraction_proof_l1814_181460

theorem fraction_proof (N : ℝ) (F : ℝ) (h1 : N = 8) (h2 : 0.5 * N = F * N + 2) : F = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proof_l1814_181460


namespace NUMINAMATH_CALUDE_count_factors_l1814_181444

/-- The number of distinct, whole-number factors of 3^5 * 5^3 * 7^2 -/
def num_factors : ℕ := 72

/-- The prime factorization of the number -/
def prime_factorization : List (ℕ × ℕ) := [(3, 5), (5, 3), (7, 2)]

/-- Theorem stating that the number of distinct, whole-number factors of 3^5 * 5^3 * 7^2 is 72 -/
theorem count_factors : 
  (List.prod (prime_factorization.map (fun (p, e) => e + 1))) = num_factors := by
  sorry

end NUMINAMATH_CALUDE_count_factors_l1814_181444


namespace NUMINAMATH_CALUDE_tom_walking_distance_l1814_181437

/-- Tom's walking rate in miles per minute -/
def walking_rate : ℚ := 2 / 36

/-- The time Tom walks in minutes -/
def walking_time : ℚ := 9

/-- The distance Tom walks in miles -/
def walking_distance : ℚ := walking_rate * walking_time

theorem tom_walking_distance :
  walking_distance = 1/2 := by sorry

end NUMINAMATH_CALUDE_tom_walking_distance_l1814_181437


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1814_181431

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) : (x - 3*y) - (y - 2*x) = 3*x - 4*y := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : 5*a*b^2 - 3*(2*a^2*b - 2*(a^2*b - 2*a*b^2)) = -7*a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1814_181431


namespace NUMINAMATH_CALUDE_f_2014_equals_2_l1814_181426

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_2014_equals_2
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x * f (x + 2) = 1)
  (h2 : f 1 = 3)
  (h3 : f 2 = 2) :
  f 2014 = 2 :=
sorry

end NUMINAMATH_CALUDE_f_2014_equals_2_l1814_181426


namespace NUMINAMATH_CALUDE_exists_perfect_pair_with_122_l1814_181490

/-- Two natural numbers form a perfect pair if their sum and product are both perfect squares. -/
def IsPerfectPair (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), a + b = x^2 ∧ a * b = y^2

/-- There exists a natural number that forms a perfect pair with 122. -/
theorem exists_perfect_pair_with_122 : ∃ (n : ℕ), IsPerfectPair 122 n := by
  sorry

end NUMINAMATH_CALUDE_exists_perfect_pair_with_122_l1814_181490


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1814_181494

theorem solution_set_inequality (x : ℝ) : 
  (2 * x) / (x - 1) < 1 ↔ -1 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1814_181494


namespace NUMINAMATH_CALUDE_angle_value_l1814_181414

theorem angle_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan β = 1/2) (h4 : Real.tan (α - β) = 1/3) : α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l1814_181414


namespace NUMINAMATH_CALUDE_sequence_general_term_l1814_181456

/-- Given a sequence {aₙ} where the sequence of differences forms an arithmetic
    sequence with first term 1 and common difference 1, prove that the general
    term formula for {aₙ} is n(n+1)/2. -/
theorem sequence_general_term (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) - a n = n) →
  a 1 = 1 →
  ∀ n : ℕ, a n = n * (n + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1814_181456


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l1814_181421

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical (x y z : ℝ) :
  x = 2 ∧ y = 2 * Real.sqrt 3 ∧ z = 4 →
  ∃ (r θ : ℝ),
    r = 4 ∧
    θ = π / 3 ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ ∧
    z = z :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l1814_181421


namespace NUMINAMATH_CALUDE_parallel_line_length_l1814_181418

/-- A triangle with a base of 24 inches and a parallel line dividing it into two equal areas -/
structure DividedTriangle where
  /-- The length of the base of the triangle -/
  base : ℝ
  /-- The length of the parallel line dividing the triangle -/
  parallel_line : ℝ
  /-- The base of the triangle is 24 inches -/
  base_length : base = 24
  /-- The parallel line divides the triangle into two equal areas -/
  equal_areas : parallel_line^2 = (1/2) * base^2

/-- The length of the parallel line in the divided triangle is 12√2 -/
theorem parallel_line_length (t : DividedTriangle) : t.parallel_line = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_length_l1814_181418


namespace NUMINAMATH_CALUDE_quadratic_range_for_x_less_than_neg_two_l1814_181471

/-- Represents a quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.yValue (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_range_for_x_less_than_neg_two
  (f : QuadraticFunction)
  (h_a_pos : f.a > 0)
  (h_vertex : f.yValue (-1) = -6)
  (h_y_at_neg_two : f.yValue (-2) = -5)
  (x : ℝ)
  (h_x : x < -2) :
  f.yValue x > -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_for_x_less_than_neg_two_l1814_181471


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l1814_181419

theorem complex_magnitude_equality (t : ℝ) : 
  t > 0 → (Complex.abs (-4 + t * Complex.I) = 2 * Real.sqrt 13 ↔ t = 6) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l1814_181419


namespace NUMINAMATH_CALUDE_magnitude_a_plus_b_unique_k_parallel_l1814_181487

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, 2]
def c : Fin 2 → ℝ := ![4, 1]

/-- The magnitude of the sum of vectors a and b is 5 -/
theorem magnitude_a_plus_b : ‖a + b‖ = 5 := by sorry

/-- The unique value of k such that a + k*c is parallel to 2*a - b is 3 -/
theorem unique_k_parallel : ∃! k : ℝ, ∃ t : ℝ, a + k • c = t • (2 • a - b) ∧ k = 3 := by sorry

end NUMINAMATH_CALUDE_magnitude_a_plus_b_unique_k_parallel_l1814_181487


namespace NUMINAMATH_CALUDE_golden_section_length_l1814_181402

/-- Given a segment AB of length 2 with C as its golden section point (AC > BC),
    the length of AC is √5 - 1 -/
theorem golden_section_length (A B C : ℝ) : 
  (B - A = 2) →
  (C - A) / (B - C) = (1 + Real.sqrt 5) / 2 →
  C - A > B - C →
  C - A = Real.sqrt 5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_golden_section_length_l1814_181402


namespace NUMINAMATH_CALUDE_reconstruct_triangle_l1814_181458

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the external angle bisector
def externalAngleBisector (A B C : Point) : Point → Prop := sorry

-- Define the perpendicular from a point to a line
def perpendicularFoot (P A B : Point) : Point := sorry

-- Define the statement
theorem reconstruct_triangle (A' B' C' : Point) :
  ∃ (A B C : Point),
    -- A'B'C' is formed by external angle bisectors of ABC
    externalAngleBisector B C A A' ∧
    externalAngleBisector A C B B' ∧
    externalAngleBisector A B C C' ∧
    -- A, B, C are feet of perpendiculars from A', B', C' to opposite sides of A'B'C'
    A = perpendicularFoot A' B' C' ∧
    B = perpendicularFoot B' A' C' ∧
    C = perpendicularFoot C' A' B' :=
by
  sorry

end NUMINAMATH_CALUDE_reconstruct_triangle_l1814_181458


namespace NUMINAMATH_CALUDE_power_product_eight_l1814_181450

theorem power_product_eight (a b : ℕ+) (h : (2 ^ a.val) ^ b.val = 2 ^ 2) :
  2 ^ a.val * 2 ^ b.val = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_eight_l1814_181450


namespace NUMINAMATH_CALUDE_emily_sees_leo_l1814_181438

/-- The time Emily can see Leo given their speeds and distances -/
theorem emily_sees_leo (emily_speed leo_speed : ℝ) (initial_distance final_distance : ℝ) : 
  emily_speed = 15 →
  leo_speed = 10 →
  initial_distance = 0.75 →
  final_distance = 0.6 →
  (initial_distance + final_distance) / (emily_speed - leo_speed) * 60 = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_emily_sees_leo_l1814_181438


namespace NUMINAMATH_CALUDE_logarithm_simplification_l1814_181411

open Real

theorem logarithm_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy_neq_1 : y ≠ 1) :
  (log x^2 / log y^8) * (log y^5 / log x^4) * (log x^3 / log y^5) * (log y^8 / log x^3) * (log x^4 / log y^3) = 
  (1/3) * (log x / log y) := by
sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l1814_181411


namespace NUMINAMATH_CALUDE_overall_percentage_l1814_181452

theorem overall_percentage (grade1 grade2 grade3 : ℚ) 
  (h1 : grade1 = 50 / 100)
  (h2 : grade2 = 70 / 100)
  (h3 : grade3 = 90 / 100) :
  (grade1 + grade2 + grade3) / 3 = 70 / 100 := by
  sorry

end NUMINAMATH_CALUDE_overall_percentage_l1814_181452


namespace NUMINAMATH_CALUDE_prob_two_rolls_eq_one_sixty_fourth_l1814_181400

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The desired sum on each roll -/
def target_sum : ℕ := 9

/-- The set of all possible outcomes when rolling two dice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_sides) (Finset.range num_sides)

/-- The set of outcomes that sum to the target -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun (a, b) => a + b + 2 = target_sum)

/-- The probability of rolling the target sum once -/
def prob_single_roll : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem prob_two_rolls_eq_one_sixty_fourth :
  prob_single_roll * prob_single_roll = 1 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_rolls_eq_one_sixty_fourth_l1814_181400


namespace NUMINAMATH_CALUDE_only_one_true_iff_in_range_l1814_181478

/-- The proposition p: no solution for the quadratic inequality -/
def p (a : ℝ) : Prop := a > 0 ∧ ∀ x, x^2 + (a-1)*x + a^2 > 0

/-- The proposition q: probability condition -/
def q (a : ℝ) : Prop := a > 0 ∧ (min a 4 + 2) / 6 ≥ 5/6

/-- The main theorem -/
theorem only_one_true_iff_in_range (a : ℝ) :
  (p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ a > 1/3 ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_only_one_true_iff_in_range_l1814_181478


namespace NUMINAMATH_CALUDE_triangle_problem_l1814_181488

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * (b^2 + c^2 - a^2) →
  (1/2) * b * c * Real.sin A = 3/2 →
  (A = π/3 ∧ ((b*c - 4*Real.sqrt 3) * Real.cos A + a*c * Real.cos B) / (a^2 - b^2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1814_181488


namespace NUMINAMATH_CALUDE_new_city_total_buildings_l1814_181435

/-- Represents the number of buildings of each type in Pittsburgh -/
structure PittsburghBuildings where
  stores : ℕ
  hospitals : ℕ
  schools : ℕ
  police_stations : ℕ

/-- Calculates the number of buildings for the new city based on Pittsburgh's numbers -/
def new_city_buildings (p : PittsburghBuildings) : ℕ × ℕ × ℕ × ℕ :=
  (p.stores / 2, p.hospitals * 2, p.schools - 50, p.police_stations + 5)

/-- Theorem stating that the total number of buildings in the new city is 2175 -/
theorem new_city_total_buildings (p : PittsburghBuildings) 
  (h1 : p.stores = 2000)
  (h2 : p.hospitals = 500)
  (h3 : p.schools = 200)
  (h4 : p.police_stations = 20) :
  let (new_stores, new_hospitals, new_schools, new_police) := new_city_buildings p
  new_stores + new_hospitals + new_schools + new_police = 2175 := by
  sorry

#check new_city_total_buildings

end NUMINAMATH_CALUDE_new_city_total_buildings_l1814_181435


namespace NUMINAMATH_CALUDE_inequality_proof_l1814_181443

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  6 * a * b * c ≤ a * b * (a + b) + b * c * (b + c) + a * c * (a + c) ∧
  a * b * (a + b) + b * c * (b + c) + a * c * (a + c) ≤ 2 * (a^3 + b^3 + c^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1814_181443


namespace NUMINAMATH_CALUDE_probability_at_least_one_shot_l1814_181432

/-- The probability of making at least one shot out of three, given a success rate of 3/5 for each shot. -/
theorem probability_at_least_one_shot (success_rate : ℝ) (num_shots : ℕ) : 
  success_rate = 3/5 → num_shots = 3 → 1 - (1 - success_rate)^num_shots = 0.936 := by
  sorry


end NUMINAMATH_CALUDE_probability_at_least_one_shot_l1814_181432


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1814_181404

theorem perfect_square_trinomial (a b : ℝ) : a^2 + 6*a*b + 9*b^2 = (a + 3*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1814_181404


namespace NUMINAMATH_CALUDE_fibonacci_determinant_l1814_181416

/-- An arbitrary Fibonacci sequence -/
def FibonacciSequence (u : ℕ → ℤ) : Prop :=
  ∀ n, u (n + 2) = u n + u (n + 1)

/-- The main theorem about the determinant of consecutive Fibonacci terms -/
theorem fibonacci_determinant (u : ℕ → ℤ) (h : FibonacciSequence u) :
  ∀ n : ℕ, u (n - 1) * u (n + 1) - u n ^ 2 = (-1) ^ n :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_determinant_l1814_181416


namespace NUMINAMATH_CALUDE_pure_imaginary_z_l1814_181477

theorem pure_imaginary_z (z : ℂ) : 
  (∃ (a : ℝ), z = Complex.I * a) → 
  Complex.abs (z - 1) = Complex.abs (-1 + Complex.I) → 
  z = Complex.I ∨ z = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_z_l1814_181477


namespace NUMINAMATH_CALUDE_rabbit_count_l1814_181439

/-- Given a total number of heads and a relationship between rabbit and chicken feet,
    prove the number of rabbits. -/
theorem rabbit_count (total_heads : ℕ) (rabbit_feet chicken_feet : ℕ → ℕ) : 
  total_heads = 40 →
  (∀ x, rabbit_feet x = 10 * chicken_feet (total_heads - x) - 8) →
  (∃ x, x = 33 ∧ 
        rabbit_feet x = 4 * x ∧ 
        chicken_feet (total_heads - x) = 2 * (total_heads - x)) :=
by sorry

end NUMINAMATH_CALUDE_rabbit_count_l1814_181439


namespace NUMINAMATH_CALUDE_product_trailing_zeroes_l1814_181409

/-- The number of trailing zeroes in a positive integer -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- The product of 25^5, 150^4, and 2008^3 -/
def largeProduct : ℕ := 25^5 * 150^4 * 2008^3

theorem product_trailing_zeroes :
  trailingZeroes largeProduct = 13 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeroes_l1814_181409


namespace NUMINAMATH_CALUDE_solution_exists_for_all_primes_l1814_181403

theorem solution_exists_for_all_primes (p : ℕ) (hp : Nat.Prime p) :
  ∃ n : ℤ, (6 * n^2 + 5 * n + 1) % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_for_all_primes_l1814_181403


namespace NUMINAMATH_CALUDE_g_squared_difference_l1814_181429

-- Define the function g
def g : ℝ → ℝ := λ x => 3

-- State the theorem
theorem g_squared_difference (x : ℝ) : g ((x - 1)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_squared_difference_l1814_181429


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l1814_181468

/-- Given a curve y = x^3 and a point (1, 1) on this curve, 
    the equation of the tangent line at this point is 3x - y - 2 = 0 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3) → -- The curve equation
  (1 = 1^3) → -- The point (1, 1) satisfies the curve equation
  (3*x - y - 2 = 0) -- The equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l1814_181468


namespace NUMINAMATH_CALUDE_candy_difference_l1814_181413

theorem candy_difference (anna_per_house billy_per_house anna_houses billy_houses : ℕ) 
  (h1 : anna_per_house = 14)
  (h2 : billy_per_house = 11)
  (h3 : anna_houses = 60)
  (h4 : billy_houses = 75) :
  anna_per_house * anna_houses - billy_per_house * billy_houses = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l1814_181413


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l1814_181440

theorem radio_loss_percentage (original_price sold_price : ℚ) :
  original_price = 490 →
  sold_price = 465.50 →
  (original_price - sold_price) / original_price * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l1814_181440


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l1814_181453

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (hS : S = 8) 
  (hsum : sum_first_two = 5) : 
  ∃ a : ℝ, (a = 8 * (1 - Real.sqrt (3/8)) ∨ a = 8 * (1 + Real.sqrt (3/8))) ∧ 
    (∃ r : ℝ, S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l1814_181453


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1814_181470

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 9 = 0 ∧ x₂^2 + m*x₂ + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1814_181470


namespace NUMINAMATH_CALUDE_sqrt_equation_implies_value_l1814_181466

theorem sqrt_equation_implies_value (a b : ℝ) :
  (Real.sqrt (a - 2 * b + 4) + (a + b - 5) ^ 2 = 0) →
  (4 * Real.sqrt a - Real.sqrt 24 / Real.sqrt b = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_implies_value_l1814_181466


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l1814_181412

theorem cos_sixty_degrees : Real.cos (π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l1814_181412


namespace NUMINAMATH_CALUDE_sequence_a_bounds_l1814_181433

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => sequence_a n + (1 : ℚ)/(n+1)^2 * (sequence_a n)^2

theorem sequence_a_bounds : ∀ n : ℕ, (n+1 : ℚ)/(n+2) < sequence_a n ∧ sequence_a n < n+1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_bounds_l1814_181433


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l1814_181473

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l1814_181473


namespace NUMINAMATH_CALUDE_john_total_distance_l1814_181481

/-- Calculates the total distance cycled given a constant speed and total cycling time -/
def total_distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: Given John's cycling conditions, he cycles 18 miles in total -/
theorem john_total_distance :
  let speed : ℝ := 6  -- miles per hour
  let time_before_rest : ℝ := 2  -- hours
  let time_after_rest : ℝ := 1  -- hour
  let total_time : ℝ := time_before_rest + time_after_rest
  total_distance speed total_time = 18 := by
  sorry

#check john_total_distance

end NUMINAMATH_CALUDE_john_total_distance_l1814_181481


namespace NUMINAMATH_CALUDE_train_speed_l1814_181449

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 385 →
  bridge_length = 140 →
  time = 42 →
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1814_181449


namespace NUMINAMATH_CALUDE_greatest_base7_digit_sum_l1814_181484

/-- Represents a base-7 digit (0 to 6) -/
def Base7Digit := Fin 7

/-- Represents a base-7 number as a list of digits -/
def Base7Number := List Base7Digit

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : Base7Number :=
  sorry

/-- Calculates the sum of digits in a base-7 number -/
def digitSum (num : Base7Number) : ℕ :=
  sorry

/-- Checks if a base-7 number is less than 1729 in decimal -/
def isLessThan1729 (num : Base7Number) : Prop :=
  sorry

theorem greatest_base7_digit_sum :
  ∃ (n : Base7Number), isLessThan1729 n ∧
    digitSum n = 22 ∧
    ∀ (m : Base7Number), isLessThan1729 m → digitSum m ≤ 22 :=
  sorry

end NUMINAMATH_CALUDE_greatest_base7_digit_sum_l1814_181484


namespace NUMINAMATH_CALUDE_four_hearts_probability_l1814_181474

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Diamonds | Clubs | Spades

/-- Represents the rank of a card -/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- A function that maps a card index to its suit -/
def card_to_suit : Fin 52 → Suit := sorry

/-- A function that maps a card index to its rank -/
def card_to_rank : Fin 52 → Rank := sorry

/-- The number of hearts in a standard deck -/
def hearts_count : Nat := 13

/-- Theorem: The probability of drawing four hearts as the top four cards from a standard 52-card deck is 286/108290 -/
theorem four_hearts_probability (d : Deck) : 
  (hearts_count * (hearts_count - 1) * (hearts_count - 2) * (hearts_count - 3)) / 
  (d.cards.card * (d.cards.card - 1) * (d.cards.card - 2) * (d.cards.card - 3)) = 286 / 108290 :=
sorry

end NUMINAMATH_CALUDE_four_hearts_probability_l1814_181474


namespace NUMINAMATH_CALUDE_polygon_sides_proof_l1814_181462

theorem polygon_sides_proof (x y : ℕ) : 
  (x - 2) * 180 + (y - 2) * 180 = 21 * (x + y + x * (x - 3) / 2 + y * (y - 3) / 2) - 39 →
  x * (x - 3) / 2 + y * (y - 3) / 2 - (x + y) = 99 →
  ((x = 17 ∧ y = 3) ∨ (x = 3 ∧ y = 17)) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_proof_l1814_181462


namespace NUMINAMATH_CALUDE_sandbox_dimension_ratio_l1814_181406

theorem sandbox_dimension_ratio 
  (V₁ V₂ : ℝ) 
  (h₁ : V₁ = 10) 
  (h₂ : V₂ = 80) 
  (k : ℝ) 
  (h₃ : V₂ = k^3 * V₁) : 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_dimension_ratio_l1814_181406


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l1814_181479

theorem inequality_system_solutions : 
  {x : ℕ | 5 * x - 6 ≤ 2 * (x + 3) ∧ (x : ℚ) / 4 - 1 < (x - 2 : ℚ) / 3} = {0, 1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l1814_181479


namespace NUMINAMATH_CALUDE_square_area_12cm_l1814_181464

/-- The area of a square with side length 12 cm is 144 square centimeters. -/
theorem square_area_12cm (s : ℝ) (h : s = 12) : s^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_12cm_l1814_181464


namespace NUMINAMATH_CALUDE_range_of_x_l1814_181472

theorem range_of_x (x y : ℝ) (h1 : x + y = 1) (h2 : y ≤ 2) : x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1814_181472


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1814_181467

/-- Calculate the percentage increase given an initial and new price -/
def percentage_increase (initial_price new_price : ℚ) : ℚ :=
  ((new_price - initial_price) / initial_price) * 100

/-- Theorem: The percentage increase from R$ 5.00 to R$ 5.55 is 11% -/
theorem price_increase_percentage :
  let initial_price : ℚ := 5
  let new_price : ℚ := (111 : ℚ) / 20
  percentage_increase initial_price new_price = 11 := by
sorry

#eval percentage_increase 5 (111 / 20)

end NUMINAMATH_CALUDE_price_increase_percentage_l1814_181467


namespace NUMINAMATH_CALUDE_two_roots_condition_l1814_181496

-- Define the equation
def f (x a : ℝ) : ℝ := 4 * x^2 - 16 * |x| + (2 * a + |x| - x)^2 - 16

-- Define the condition for exactly two distinct roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0 ∧
  ∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂

-- State the theorem
theorem two_roots_condition :
  ∀ a : ℝ, has_two_distinct_roots a ↔ (a > -6 ∧ a ≤ -2) ∨ (a > 2 ∧ a < Real.sqrt 8) :=
sorry

end NUMINAMATH_CALUDE_two_roots_condition_l1814_181496


namespace NUMINAMATH_CALUDE_ladder_slide_l1814_181415

theorem ladder_slide (L d s : ℝ) (h1 : L = 20) (h2 : d = 4) (h3 : s = 3) :
  ∃ y : ℝ, y = Real.sqrt (400 - (2 * Real.sqrt 96 - 3)^2) - 4 :=
sorry

end NUMINAMATH_CALUDE_ladder_slide_l1814_181415


namespace NUMINAMATH_CALUDE_sam_distance_l1814_181461

/-- Given Marguerite's cycling distance and time, and Sam's cycling time,
    prove that Sam's distance is equal to (Marguerite's distance / Marguerite's time) * Sam's time,
    assuming they cycle at the same average speed. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
    (h1 : marguerite_distance > 0)
    (h2 : marguerite_time > 0)
    (h3 : sam_time > 0) :
  let sam_distance := (marguerite_distance / marguerite_time) * sam_time
  sam_distance = (marguerite_distance / marguerite_time) * sam_time :=
by
  sorry

#check sam_distance

end NUMINAMATH_CALUDE_sam_distance_l1814_181461


namespace NUMINAMATH_CALUDE_largest_fraction_l1814_181486

theorem largest_fraction : 
  (8 : ℚ) / 9 > 7 / 8 ∧ 
  (8 : ℚ) / 9 > 66 / 77 ∧ 
  (8 : ℚ) / 9 > 55 / 66 ∧ 
  (8 : ℚ) / 9 > 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l1814_181486


namespace NUMINAMATH_CALUDE_rotation_of_D_l1814_181495

/-- Rotates a point 90 degrees clockwise about the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotation_of_D : 
  let D : ℝ × ℝ := (-3, -8)
  rotate90Clockwise D = (-8, 3) := by
sorry

end NUMINAMATH_CALUDE_rotation_of_D_l1814_181495


namespace NUMINAMATH_CALUDE_original_price_calculation_l1814_181408

/-- Given a sale price and a percent decrease, calculate the original price of an item. -/
theorem original_price_calculation (sale_price : ℝ) (percent_decrease : ℝ) 
  (h1 : sale_price = 75)
  (h2 : percent_decrease = 25) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - percent_decrease / 100) = sale_price ∧ 
    original_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1814_181408


namespace NUMINAMATH_CALUDE_area_of_large_rectangle_l1814_181463

/-- The width of each smaller rectangle in feet -/
def small_rectangle_width : ℝ := 8

/-- The number of identical rectangles stacked vertically -/
def num_rectangles : ℕ := 3

/-- The length of each smaller rectangle in feet -/
def small_rectangle_length : ℝ := 2 * small_rectangle_width

/-- The width of the larger rectangle ABCD in feet -/
def large_rectangle_width : ℝ := small_rectangle_width

/-- The length of the larger rectangle ABCD in feet -/
def large_rectangle_length : ℝ := num_rectangles * small_rectangle_length

/-- The area of the larger rectangle ABCD in square feet -/
def large_rectangle_area : ℝ := large_rectangle_width * large_rectangle_length

theorem area_of_large_rectangle : large_rectangle_area = 384 := by
  sorry

end NUMINAMATH_CALUDE_area_of_large_rectangle_l1814_181463


namespace NUMINAMATH_CALUDE_circular_arrangement_exists_l1814_181485

theorem circular_arrangement_exists : ∃ (a : Fin 12 → Fin 12), Function.Bijective a ∧
  ∀ (i j : Fin 12), i ≠ j → |a i - a j| ≠ |i.val - j.val| := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_exists_l1814_181485


namespace NUMINAMATH_CALUDE_discriminant_of_quadratic_poly_l1814_181455

/-- The discriminant of a quadratic polynomial ax² + bx + c is b² - 4ac -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The quadratic polynomial 3x² + (3 + 1/3)x + 1/3 -/
def quadratic_poly (x : ℚ) : ℚ := 3*x^2 + (3 + 1/3)*x + 1/3

theorem discriminant_of_quadratic_poly :
  discriminant 3 (3 + 1/3) (1/3) = 64/9 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_quadratic_poly_l1814_181455


namespace NUMINAMATH_CALUDE_star_neg_x_not_2x_squared_l1814_181434

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem stating that x ⋆ (-x) = 2x^2 is false
theorem star_neg_x_not_2x_squared : ¬ ∀ x : ℝ, star x (-x) = 2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_star_neg_x_not_2x_squared_l1814_181434


namespace NUMINAMATH_CALUDE_sum_equals_four_l1814_181499

/-- Custom binary operation on real numbers -/
def custom_op (x y : ℝ) : ℝ := x * (1 - y)

/-- The solution set of the inequality -/
def solution_set : Set ℝ := Set.Ioo 2 3

/-- Theorem stating the sum of a and b equals 4 -/
theorem sum_equals_four (a b : ℝ) 
  (h : ∀ x ∈ solution_set, custom_op (x - a) (x - b) > 0) 
  (h_unique : ∀ x ∉ solution_set, custom_op (x - a) (x - b) ≤ 0) : 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_four_l1814_181499


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l1814_181410

/-- Given two points on a line and another line equation, prove the value of k -/
theorem parallel_lines_k_value (k : ℝ) : 
  (∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b → 
    (x = 3 ∧ y = -12) ∨ (x = k ∧ y = 22)) ∧
   (∀ x y : ℝ, 4 * x + 6 * y = 36 → y = m * x + (36 / 6 - 4 * x / 6))) →
  k = -48 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l1814_181410


namespace NUMINAMATH_CALUDE_prop_A_prop_B_l1814_181498

-- Define the function f(x) = (x-2)^2
def f (x : ℝ) : ℝ := (x - 2)^2

-- Proposition A: f(x+2) is an even function
theorem prop_A : ∀ x : ℝ, f (x + 2) = f (-x + 2) := by sorry

-- Proposition B: f(x) is decreasing on (-∞, 2) and increasing on (2, +∞)
theorem prop_B :
  (∀ x y : ℝ, x < y → y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x → x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_prop_A_prop_B_l1814_181498


namespace NUMINAMATH_CALUDE_penalty_kicks_count_l1814_181424

theorem penalty_kicks_count (total_players : ℕ) (goalies : ℕ) : 
  total_players = 20 ∧ goalies = 3 → 
  (total_players - goalies) * goalies + goalies * (goalies - 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_count_l1814_181424


namespace NUMINAMATH_CALUDE_parallelepiped_volume_l1814_181428

/-- A rectangular parallelepiped divided into eight parts -/
structure Parallelepiped where
  volume_A : ℝ
  volume_C : ℝ
  volume_B_prime : ℝ
  volume_C_prime : ℝ

/-- The theorem stating that the total volume of the parallelepiped is 790 -/
theorem parallelepiped_volume 
  (p : Parallelepiped) 
  (h1 : p.volume_A = 40)
  (h2 : p.volume_C = 300)
  (h3 : p.volume_B_prime = 360)
  (h4 : p.volume_C_prime = 90) :
  p.volume_A + p.volume_C + p.volume_B_prime + p.volume_C_prime = 790 :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_l1814_181428


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1814_181407

theorem square_area_from_diagonal (a b : ℝ) :
  let diagonal := a + b
  ∃ s : ℝ, s > 0 ∧ s * s = (1/2) * diagonal * diagonal :=
by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1814_181407


namespace NUMINAMATH_CALUDE_pig_price_calculation_l1814_181441

/-- Given a total of 3 pigs and 10 hens costing Rs. 1200 in total,
    with hens costing an average of Rs. 30 each,
    prove that the average price of a pig is Rs. 300. -/
theorem pig_price_calculation (total_cost : ℕ) (num_pigs num_hens : ℕ) (avg_hen_price : ℕ) :
  total_cost = 1200 →
  num_pigs = 3 →
  num_hens = 10 →
  avg_hen_price = 30 →
  (total_cost - num_hens * avg_hen_price) / num_pigs = 300 := by
  sorry

end NUMINAMATH_CALUDE_pig_price_calculation_l1814_181441


namespace NUMINAMATH_CALUDE_problem_solution_l1814_181405

theorem problem_solution (x : ℤ) : x - (28 - (37 - (15 - 18))) = 57 → x = 69 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1814_181405


namespace NUMINAMATH_CALUDE_brenda_spay_cats_l1814_181423

/-- Represents the number of cats Brenda needs to spay -/
def num_cats : ℕ := sorry

/-- Represents the number of dogs Brenda needs to spay -/
def num_dogs : ℕ := sorry

/-- The total number of animals Brenda needs to spay -/
def total_animals : ℕ := 21

theorem brenda_spay_cats :
  (num_cats + num_dogs = total_animals) →
  (num_dogs = 2 * num_cats) →
  num_cats = 7 := by
  sorry

end NUMINAMATH_CALUDE_brenda_spay_cats_l1814_181423


namespace NUMINAMATH_CALUDE_election_results_l1814_181422

/-- Election results theorem -/
theorem election_results 
  (total_students : ℕ) 
  (voter_turnout : ℚ) 
  (vote_percent_A vote_percent_B vote_percent_C vote_percent_D vote_percent_E : ℚ) : 
  total_students = 5000 →
  voter_turnout = 3/5 →
  vote_percent_A = 2/5 →
  vote_percent_B = 1/4 →
  vote_percent_C = 1/5 →
  vote_percent_D = 1/10 →
  vote_percent_E = 1/20 →
  (↑total_students * voter_turnout * vote_percent_A - ↑total_students * voter_turnout * vote_percent_B : ℚ) = 450 ∧
  (↑total_students * voter_turnout * (vote_percent_C + vote_percent_D + vote_percent_E) : ℚ) = 1050 := by
  sorry

end NUMINAMATH_CALUDE_election_results_l1814_181422


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1814_181493

-- Define the inequality function
def f (x : ℝ) : ℝ := x^2 - 2*x - 8

-- Define the solution set
def solution_set : Set ℝ := {x | x ≤ -2 ∨ x ≥ 4}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1814_181493


namespace NUMINAMATH_CALUDE_ten_bulb_signals_l1814_181475

/-- The number of different signals that can be transmitted using a given number of light bulbs -/
def signalCount (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of different signals that can be transmitted using 10 light bulbs, 
    each of which can be either on or off, is equal to 2^10 (1024) -/
theorem ten_bulb_signals : signalCount 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_bulb_signals_l1814_181475


namespace NUMINAMATH_CALUDE_polynomial_divisibility_existence_l1814_181483

theorem polynomial_divisibility_existence : ∃ (r s : ℝ),
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    8 * x^4 - 4 * x^3 - 42 * x^2 + 45 * x - 10 = 
    ((x - r)^2 * (x - s) * (x - 1)) * q x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_existence_l1814_181483


namespace NUMINAMATH_CALUDE_infinitely_many_primes_with_quadratic_nonresidue_l1814_181451

theorem infinitely_many_primes_with_quadratic_nonresidue (a : ℤ) 
  (h_odd : Odd a) (h_not_square : ∀ n : ℤ, n ^ 2 ≠ a) :
  ∃ (S : Set ℕ), (∀ p ∈ S, Prime p) ∧ 
  Set.Infinite S ∧ 
  (∀ p ∈ S, ¬ ∃ x : ℤ, x ^ 2 ≡ a [ZMOD p]) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_with_quadratic_nonresidue_l1814_181451


namespace NUMINAMATH_CALUDE_kellys_games_l1814_181447

/-- Kelly's nintendo games problem -/
theorem kellys_games (initial_games : ℕ) (given_away : ℕ) (remaining_games : ℕ) : 
  initial_games = 106 → given_away = 64 → remaining_games = initial_games - given_away → remaining_games = 42 := by
  sorry

end NUMINAMATH_CALUDE_kellys_games_l1814_181447


namespace NUMINAMATH_CALUDE_rotate_line_theorem_l1814_181469

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Rotates a line counterclockwise by π/2 around a given point -/
def rotateLine (l : Line) (px py : ℝ) : Line :=
  sorry

theorem rotate_line_theorem (l : Line) :
  l.a = 2 ∧ l.b = -1 ∧ l.c = -2 →
  let rotated := rotateLine l 0 (-2)
  rotated.a = 1 ∧ rotated.b = 2 ∧ rotated.c = 4 :=
sorry

end NUMINAMATH_CALUDE_rotate_line_theorem_l1814_181469


namespace NUMINAMATH_CALUDE_randy_wipes_days_l1814_181430

/-- Calculates the number of days Randy can use wipes given the number of packs and wipes per pack -/
def days_of_wipes (walks_per_day : ℕ) (paws : ℕ) (packs : ℕ) (wipes_per_pack : ℕ) : ℕ :=
  let wipes_per_day := walks_per_day * paws
  let total_wipes := packs * wipes_per_pack
  total_wipes / wipes_per_day

/-- Theorem stating that Randy needs wipes for 90 days -/
theorem randy_wipes_days :
  days_of_wipes 2 4 6 120 = 90 := by
  sorry

end NUMINAMATH_CALUDE_randy_wipes_days_l1814_181430


namespace NUMINAMATH_CALUDE_blue_marbles_count_l1814_181417

theorem blue_marbles_count (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) 
  (h_total : total = 20)
  (h_red : red = 7)
  (h_prob : prob_red_or_white = 3/4) :
  ∃ blue : ℕ, blue = 5 ∧ 
    (red : ℚ) / total + (total - red - blue : ℚ) / total = prob_red_or_white :=
by sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l1814_181417


namespace NUMINAMATH_CALUDE_complex_real_condition_l1814_181480

theorem complex_real_condition (a : ℝ) :
  (Complex.I * (a - 1) = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1814_181480


namespace NUMINAMATH_CALUDE_chuzhou_gdp_scientific_notation_l1814_181489

/-- The GDP of Chuzhou City in 2022 in billions of yuan -/
def chuzhou_gdp : ℝ := 3600

/-- Conversion factor from billion to scientific notation -/
def billion_to_scientific : ℝ := 10^9

theorem chuzhou_gdp_scientific_notation :
  chuzhou_gdp * billion_to_scientific = 3.6 * 10^12 := by
  sorry

end NUMINAMATH_CALUDE_chuzhou_gdp_scientific_notation_l1814_181489


namespace NUMINAMATH_CALUDE_f_value_at_pi_over_4_f_monotone_increasing_l1814_181401

noncomputable def f (x : ℝ) : ℝ := (Real.sin (2 * x) + 2 * (Real.cos x) ^ 2) / Real.cos x

def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2

theorem f_value_at_pi_over_4 :
  f (Real.pi / 4) = 2 * Real.sqrt 2 :=
sorry

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Ioo 0 (Real.pi / 4)) :=
sorry

end NUMINAMATH_CALUDE_f_value_at_pi_over_4_f_monotone_increasing_l1814_181401


namespace NUMINAMATH_CALUDE_weight_probability_l1814_181457

/-- The probability that the weight of five eggs is less than 30 grams -/
def prob_less_than_30 : ℝ := 0.3

/-- The probability that the weight of five eggs is between [30, 40] grams -/
def prob_between_30_and_40 : ℝ := 0.5

/-- The probability that the weight of five eggs does not exceed 40 grams -/
def prob_not_exceed_40 : ℝ := prob_less_than_30 + prob_between_30_and_40

theorem weight_probability : prob_not_exceed_40 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_weight_probability_l1814_181457


namespace NUMINAMATH_CALUDE_f_inequality_l1814_181497

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f x = f (x + p)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem f_inequality (f : ℝ → ℝ) 
  (h1 : is_periodic f 4)
  (h2 : is_increasing_on f 0 2)
  (h3 : is_symmetric_about (fun x ↦ f (x + 2)) 0) :
  f 4.5 < f 7 ∧ f 7 < f 6.5 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1814_181497


namespace NUMINAMATH_CALUDE_point_on_number_line_l1814_181436

theorem point_on_number_line (a : ℝ) : 
  (∃ (A : ℝ), A = 2 * a + 1 ∧ |A| = 3) → (a = 1 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l1814_181436


namespace NUMINAMATH_CALUDE_grid_constant_l1814_181492

/-- A function representing the assignment of positive integers to grid points -/
def GridAssignment := ℤ → ℤ → ℕ+

/-- The condition that each value is the arithmetic mean of its neighbors -/
def is_arithmetic_mean (f : GridAssignment) : Prop :=
  ∀ x y : ℤ, (f x y : ℚ) = ((f (x-1) y + f (x+1) y + f x (y-1) + f x (y+1)) : ℚ) / 4

/-- The main theorem: if a grid assignment satisfies the arithmetic mean condition,
    then it is constant across the entire grid -/
theorem grid_constant (f : GridAssignment) (h : is_arithmetic_mean f) :
  ∀ x y x' y' : ℤ, f x y = f x' y' :=
sorry

end NUMINAMATH_CALUDE_grid_constant_l1814_181492


namespace NUMINAMATH_CALUDE_registration_methods_l1814_181425

/-- The number of students signing up for interest groups -/
def num_students : ℕ := 4

/-- The number of interest groups available -/
def num_groups : ℕ := 3

/-- Theorem stating the total number of registration methods -/
theorem registration_methods :
  (num_groups ^ num_students : ℕ) = 81 := by
  sorry

end NUMINAMATH_CALUDE_registration_methods_l1814_181425


namespace NUMINAMATH_CALUDE_gcf_of_2000_and_7700_l1814_181446

theorem gcf_of_2000_and_7700 : Nat.gcd 2000 7700 = 100 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_2000_and_7700_l1814_181446


namespace NUMINAMATH_CALUDE_total_length_of_objects_l1814_181448

/-- Given the lengths of various objects and their relationships, prove their total length. -/
theorem total_length_of_objects (pencil_length : ℝ) 
  (h1 : pencil_length = 12) 
  (h2 : ∃ pen_length rubber_length, 
    pen_length = rubber_length + 3 ∧ 
    pencil_length = pen_length + 2)
  (h3 : ∃ ruler_length, 
    ruler_length = 3 * rubber_length ∧ 
    ruler_length = pen_length * 1.2)
  (h4 : ∃ marker_length, marker_length = ruler_length / 2)
  (h5 : ∃ scissors_length, scissors_length = pencil_length * 0.75) :
  ∃ total_length, total_length = 69.5 ∧ 
    total_length = rubber_length + pen_length + pencil_length + 
                   marker_length + ruler_length + scissors_length :=
by sorry

end NUMINAMATH_CALUDE_total_length_of_objects_l1814_181448


namespace NUMINAMATH_CALUDE_quadratic_point_relation_l1814_181420

/-- The quadratic function f(x) = x^2 - 4x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 3

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-2, f (-2))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (1, f 1)

/-- Point C on the graph of f -/
def C : ℝ × ℝ := (4, f 4)

theorem quadratic_point_relation :
  A.2 > C.2 ∧ C.2 > B.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_point_relation_l1814_181420


namespace NUMINAMATH_CALUDE_class_average_weight_l1814_181465

theorem class_average_weight (students_a : ℕ) (students_b : ℕ) (avg_weight_a : ℝ) (avg_weight_b : ℝ)
  (h1 : students_a = 36)
  (h2 : students_b = 44)
  (h3 : avg_weight_a = 40)
  (h4 : avg_weight_b = 35) :
  let total_students := students_a + students_b
  let total_weight := students_a * avg_weight_a + students_b * avg_weight_b
  total_weight / total_students = 37.25 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l1814_181465


namespace NUMINAMATH_CALUDE_t_shirt_problem_l1814_181445

/-- Represents a t-shirt package with its size and price -/
structure Package where
  size : Nat
  price : Rat

/-- Calculates the total number of t-shirts and the discounted price -/
def calculate_total (small medium large : Package) 
                    (small_qty medium_qty large_qty : Nat) : Nat × Rat :=
  let total_shirts := small.size * small_qty + medium.size * medium_qty + large.size * large_qty
  let total_price := small.price * small_qty + medium.price * medium_qty + large.price * large_qty
  let total_packages := small_qty + medium_qty + large_qty
  let discounted_price := if total_packages > 25 
                          then total_price * (1 - 5 / 100) 
                          else total_price
  (total_shirts, discounted_price)

theorem t_shirt_problem :
  let small : Package := ⟨6, 12⟩
  let medium : Package := ⟨12, 20⟩
  let large : Package := ⟨20, 30⟩
  let (total_shirts, discounted_price) := calculate_total small medium large 15 10 4
  total_shirts = 290 ∧ discounted_price = 475 := by
  sorry

end NUMINAMATH_CALUDE_t_shirt_problem_l1814_181445


namespace NUMINAMATH_CALUDE_rational_function_value_l1814_181491

-- Define the polynomials p and q
def p (k m : ℝ) (x : ℝ) : ℝ := k * x + m
def q (x : ℝ) : ℝ := (x + 4) * (x - 1)

-- State the theorem
theorem rational_function_value (k m : ℝ) :
  (p k m 0) / (q 0) = 0 →
  (p k m 2) / (q 2) = -1 →
  (p k m (-1)) / (q (-1)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l1814_181491
