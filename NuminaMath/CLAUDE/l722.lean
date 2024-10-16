import Mathlib

namespace NUMINAMATH_CALUDE_butterfly_distribution_theorem_l722_72286

/-- Represents the movement rules for butterflies on a cube --/
structure ButterflyMovement where
  adjacent : ℕ  -- Number of butterflies moving to each adjacent vertex
  opposite : ℕ  -- Number of butterflies moving to the opposite vertex
  flyaway : ℕ   -- Number of butterflies flying away

/-- Represents the state of butterflies on a cube --/
structure CubeState where
  vertices : Fin 8 → ℕ  -- Number of butterflies at each vertex

/-- Defines the condition for equal distribution of butterflies --/
def is_equally_distributed (state : CubeState) : Prop :=
  ∀ i j : Fin 8, state.vertices i = state.vertices j

/-- Defines the evolution of the cube state according to movement rules --/
def evolve (initial : CubeState) (rules : ButterflyMovement) : ℕ → CubeState
  | 0 => initial
  | n+1 => sorry  -- Implementation of evolution step

/-- Main theorem: N must be a multiple of 45 for equal distribution --/
theorem butterfly_distribution_theorem 
  (N : ℕ) 
  (initial : CubeState) 
  (rules : ButterflyMovement) 
  (h_initial : ∃ v : Fin 8, initial.vertices v = N ∧ ∀ w : Fin 8, w ≠ v → initial.vertices w = 0)
  (h_rules : rules.adjacent = 3 ∧ rules.opposite = 1 ∧ rules.flyaway = 1) :
  (∃ t : ℕ, is_equally_distributed (evolve initial rules t)) ↔ ∃ k : ℕ, N = 45 * k :=
sorry

end NUMINAMATH_CALUDE_butterfly_distribution_theorem_l722_72286


namespace NUMINAMATH_CALUDE_middle_term_of_five_term_arithmetic_sequence_l722_72269

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem middle_term_of_five_term_arithmetic_sequence 
  (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) 
  (h_first : a 1 = 21) (h_last : a 5 = 53) : 
  a 3 = 37 := by
sorry

end NUMINAMATH_CALUDE_middle_term_of_five_term_arithmetic_sequence_l722_72269


namespace NUMINAMATH_CALUDE_investment_sum_l722_72299

/-- 
Given a sum P invested at two different simple interest rates for two years,
prove that P = 12000 if the difference in interest is 720.
-/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 720) → P = 12000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l722_72299


namespace NUMINAMATH_CALUDE_smallest_product_of_digits_1234_l722_72219

/-- Given a list of four distinct digits, returns all possible pairs of two-digit numbers that can be formed using each digit exactly once. -/
def generatePairs (digits : List Nat) : List (Nat × Nat) :=
  sorry

/-- Calculates the product of a pair of numbers. -/
def pairProduct (pair : Nat × Nat) : Nat :=
  pair.1 * pair.2

/-- Finds the smallest product among a list of pairs. -/
def smallestProduct (pairs : List (Nat × Nat)) : Nat :=
  sorry

theorem smallest_product_of_digits_1234 :
  let digits := [1, 2, 3, 4]
  let pairs := generatePairs digits
  smallestProduct pairs = 312 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_of_digits_1234_l722_72219


namespace NUMINAMATH_CALUDE_expression_simplification_l722_72261

theorem expression_simplification (a b : ℝ) (ha : a = -1) (hb : b = 1) :
  (4/5) * a * b - (2 * a * b^2 - 4 * (-(1/5) * a * b + 3 * a^2 * b)) + 2 * a * b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l722_72261


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l722_72298

theorem mixed_number_calculation : 
  26 * (2 + 4/7 - (3 + 1/3)) + (3 + 1/5 + 2 + 3/7) = -(14 + 223/735) := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l722_72298


namespace NUMINAMATH_CALUDE_tom_trip_cost_l722_72297

/-- Calculates the total cost of Tom's trip to Barbados --/
def total_trip_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
  (insurance_coverage : ℚ) (flight_cost : ℚ) (num_nights : ℕ) (lodging_cost_per_night : ℚ) 
  (transportation_cost : ℚ) (food_cost_per_day : ℚ) (exchange_rate : ℚ) 
  (conversion_fee_rate : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let out_of_pocket_medical := medical_cost * (1 - insurance_coverage)
  let local_expenses := (num_nights * lodging_cost_per_night + transportation_cost + 
    num_nights * food_cost_per_day)
  let conversion_fee := local_expenses * exchange_rate * conversion_fee_rate / exchange_rate
  out_of_pocket_medical + flight_cost + local_expenses + conversion_fee

/-- Theorem stating that the total cost of Tom's trip is $3060.10 --/
theorem tom_trip_cost : 
  total_trip_cost 10 45 250 0.8 1200 7 150 200 60 2 0.03 = 3060.1 := by
  sorry

#eval total_trip_cost 10 45 250 0.8 1200 7 150 200 60 2 0.03

end NUMINAMATH_CALUDE_tom_trip_cost_l722_72297


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l722_72207

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

/-- The set of roots of f -/
def roots_of_f : Set ℂ := {z : ℂ | f z = 0}

/-- n^th roots of unity -/
def nth_roots_of_unity (n : ℕ) : Set ℂ := {z : ℂ | z^n = 1}

/-- Statement: 9 is the smallest positive integer n such that all roots of z^6 - z^3 + 1 = 0 are n^th roots of unity -/
theorem smallest_n_for_roots_of_unity :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ∃ z ∈ roots_of_f, z ∉ nth_roots_of_unity m) ∧
  (∀ z ∈ roots_of_f, z ∈ nth_roots_of_unity n) ∧
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l722_72207


namespace NUMINAMATH_CALUDE_valid_arrangement_iff_odd_l722_72277

/-- A permutation of numbers from 1 to n -/
def OuterRingPermutation (n : ℕ) := Fin n → Fin n

/-- Checks if a permutation satisfies the rotation property -/
def SatisfiesRotationProperty (n : ℕ) (p : OuterRingPermutation n) : Prop :=
  ∀ k : Fin n, ∃! j : Fin n, (p j - j : ℤ) ≡ k [ZMOD n]

/-- The main theorem: a valid arrangement exists if and only if n is odd -/
theorem valid_arrangement_iff_odd (n : ℕ) (h : n ≥ 3) :
  (∃ p : OuterRingPermutation n, SatisfiesRotationProperty n p) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_iff_odd_l722_72277


namespace NUMINAMATH_CALUDE_gina_hourly_wage_l722_72206

-- Define Gina's painting rates
def rose_rate : ℝ := 6
def lily_rate : ℝ := 7
def sunflower_rate : ℝ := 5
def orchid_rate : ℝ := 8

-- Define the orders
def order1_roses : ℝ := 6
def order1_lilies : ℝ := 14
def order1_sunflowers : ℝ := 4
def order1_payment : ℝ := 120

def order2_orchids : ℝ := 10
def order2_roses : ℝ := 2
def order2_payment : ℝ := 80

def order3_sunflowers : ℝ := 8
def order3_orchids : ℝ := 4
def order3_payment : ℝ := 70

-- Define the theorem
theorem gina_hourly_wage :
  let total_time := (order1_roses / rose_rate + order1_lilies / lily_rate + order1_sunflowers / sunflower_rate) +
                    (order2_orchids / orchid_rate + order2_roses / rose_rate) +
                    (order3_sunflowers / sunflower_rate + order3_orchids / orchid_rate)
  let total_payment := order1_payment + order2_payment + order3_payment
  let hourly_wage := total_payment / total_time
  ∃ ε > 0, |hourly_wage - 36.08| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_gina_hourly_wage_l722_72206


namespace NUMINAMATH_CALUDE_trig_expression_equality_l722_72266

theorem trig_expression_equality : 
  (Real.sin (24 * π / 180) * Real.cos (18 * π / 180) + Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) / 
  (Real.sin (28 * π / 180) * Real.cos (12 * π / 180) + Real.cos (152 * π / 180) * Real.cos (92 * π / 180)) = 
  Real.sin (18 * π / 180) / Real.sin (26 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l722_72266


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l722_72203

/-- The ratio of the area of an inscribed circle to the area of a right triangle -/
theorem inscribed_circle_area_ratio (h a r : ℝ) (h_pos : h > 0) (a_pos : a > 0) (r_pos : r > 0) 
  (h_gt_a : h > a) :
  let b := Real.sqrt (h^2 - a^2)
  let s := (a + b + h) / 2
  let triangle_area := (1 / 2) * a * b
  let circle_area := π * r^2
  (r * s = triangle_area) →
  (circle_area / triangle_area = π * a * (h^2 - a^2) / (2 * (a + b + h))) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l722_72203


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l722_72291

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 + z^2 - 5*z + 3) ≤ 128 * Real.sqrt 3 / 27 := by
  sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l722_72291


namespace NUMINAMATH_CALUDE_candy_distribution_l722_72214

theorem candy_distribution (x : ℕ) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l722_72214


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l722_72279

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (l : Line) 
  (p : Point) 
  (given_line : Line) :
  l.a = 1 ∧ l.b = 3 ∧ l.c = -2 →
  p.x = -1 ∧ p.y = 1 →
  given_line.a = 1 ∧ given_line.b = 3 ∧ given_line.c = 4 →
  p.liesOn l ∧ l.isParallelTo given_line :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l722_72279


namespace NUMINAMATH_CALUDE_initial_investment_solution_exists_l722_72241

/-- Simple interest calculation function -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the initial investment given the conditions -/
theorem initial_investment (P : ℝ) (r : ℝ) :
  (simpleInterest P r 2 = 480) →
  (simpleInterest P r 7 = 680) →
  P = 400 := by
  sorry

/-- Proof of the existence of a solution -/
theorem solution_exists : ∃ (P r : ℝ),
  (simpleInterest P r 2 = 480) ∧
  (simpleInterest P r 7 = 680) ∧
  P = 400 := by
  sorry

end NUMINAMATH_CALUDE_initial_investment_solution_exists_l722_72241


namespace NUMINAMATH_CALUDE_hamburger_combinations_count_l722_72257

/-- The number of condiments available for hamburgers -/
def num_condiments : ℕ := 8

/-- The number of patty options available for hamburgers -/
def num_patty_options : ℕ := 4

/-- Calculates the number of different hamburger combinations -/
def num_hamburger_combinations : ℕ := 2^num_condiments * num_patty_options

theorem hamburger_combinations_count :
  num_hamburger_combinations = 1024 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_count_l722_72257


namespace NUMINAMATH_CALUDE_perfect_square_primes_l722_72293

theorem perfect_square_primes (p : ℕ) : 
  Nat.Prime p ∧ ∃ (n : ℕ), (2^(p+1) - 4) / p = n^2 ↔ p = 3 ∨ p = 7 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_primes_l722_72293


namespace NUMINAMATH_CALUDE_flour_for_cookies_l722_72281

/-- Given a recipe where 24 cookies require 1.5 cups of flour,
    calculate the amount of flour needed for 72 cookies. -/
theorem flour_for_cookies (original_cookies : ℕ) (original_flour : ℚ) (new_cookies : ℕ) :
  original_cookies = 24 →
  original_flour = 3/2 →
  new_cookies = 72 →
  (original_flour / original_cookies) * new_cookies = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_flour_for_cookies_l722_72281


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_specific_integers_l722_72296

theorem sqrt_equality_implies_specific_integers (a b : ℕ) :
  0 < a → 0 < b → a < b →
  Real.sqrt (1 + Real.sqrt (21 + 12 * Real.sqrt 3)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_specific_integers_l722_72296


namespace NUMINAMATH_CALUDE_max_value_of_sum_l722_72267

theorem max_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≤ 1 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 1 ∧
    1 / (a' + b' + 1) + 1 / (b' + c' + 1) + 1 / (c' + a' + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l722_72267


namespace NUMINAMATH_CALUDE_bobby_pancakes_left_l722_72217

/-- The number of pancakes Bobby has left after making and serving breakfast -/
def pancakes_left (standard_batch : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) (friends_ate : ℕ) : ℕ :=
  let total_made := standard_batch + 2 * standard_batch + standard_batch
  let total_eaten := bobby_ate + dog_ate + friends_ate
  total_made - total_eaten

/-- Theorem stating that Bobby has 50 pancakes left -/
theorem bobby_pancakes_left : 
  pancakes_left 21 5 7 22 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bobby_pancakes_left_l722_72217


namespace NUMINAMATH_CALUDE_apple_preference_percentage_l722_72254

theorem apple_preference_percentage (total_responses : ℕ) (apple_responses : ℕ) 
  (h1 : total_responses = 300) (h2 : apple_responses = 70) :
  (apple_responses : ℚ) / (total_responses : ℚ) * 100 = 23 := by
  sorry

end NUMINAMATH_CALUDE_apple_preference_percentage_l722_72254


namespace NUMINAMATH_CALUDE_total_students_l722_72248

theorem total_students (N : ℕ) 
  (provincial_total : ℕ) (provincial_sample : ℕ)
  (experimental_sample : ℕ) (regular_sample : ℕ) (sino_canadian_sample : ℕ)
  (h1 : provincial_total = 96)
  (h2 : provincial_sample = 12)
  (h3 : experimental_sample = 21)
  (h4 : regular_sample = 25)
  (h5 : sino_canadian_sample = 43)
  (h6 : N * provincial_sample = provincial_total * (provincial_sample + experimental_sample + regular_sample + sino_canadian_sample)) :
  N = 808 := by
sorry

end NUMINAMATH_CALUDE_total_students_l722_72248


namespace NUMINAMATH_CALUDE_missing_chess_pieces_l722_72210

/-- The number of pieces in a standard chess set -/
def standard_chess_set_pieces : ℕ := 32

/-- The number of pieces present -/
def present_pieces : ℕ := 28

/-- The number of missing pieces -/
def missing_pieces : ℕ := standard_chess_set_pieces - present_pieces

theorem missing_chess_pieces : missing_pieces = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_chess_pieces_l722_72210


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l722_72287

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - 6 * x
  (f 0 = 0 ∧ f (3/2) = 0) ∧
  ∀ x : ℝ, f x = 0 → (x = 0 ∨ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l722_72287


namespace NUMINAMATH_CALUDE_flower_bed_width_l722_72247

theorem flower_bed_width (area : ℝ) (length : ℝ) (width : ℝ) :
  area = 35 →
  length = 7 →
  area = length * width →
  width = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_flower_bed_width_l722_72247


namespace NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l722_72200

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_sqrt_64_over_9_l722_72200


namespace NUMINAMATH_CALUDE_duck_cow_legs_heads_l722_72272

theorem duck_cow_legs_heads :
  ∀ (D : ℕ),
  let C : ℕ := 16
  let H : ℕ := D + C
  let L : ℕ := 2 * D + 4 * C
  L - 2 * H = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_cow_legs_heads_l722_72272


namespace NUMINAMATH_CALUDE_bruce_purchase_amount_l722_72213

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1125 for his purchase -/
theorem bruce_purchase_amount :
  total_amount 9 70 9 55 = 1125 := by
  sorry

end NUMINAMATH_CALUDE_bruce_purchase_amount_l722_72213


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_correct_l722_72295

/-- The quadratic function y = 2x^2 - 8x + 10 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 10

/-- The axis of symmetry of the quadratic function -/
def axis_of_symmetry : ℝ := 2

/-- Theorem: The axis of symmetry of the quadratic function f(x) = 2x^2 - 8x + 10 is x = 2 -/
theorem axis_of_symmetry_is_correct : 
  ∀ x : ℝ, f (axis_of_symmetry + x) = f (axis_of_symmetry - x) :=
by
  sorry

#check axis_of_symmetry_is_correct

end NUMINAMATH_CALUDE_axis_of_symmetry_is_correct_l722_72295


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l722_72212

theorem integer_roots_of_polynomial (a b c : ℚ) : 
  ∃ (p q : ℤ), p ≠ q ∧ 
    (∀ x : ℂ, x^4 + a*x^2 + b*x + c = 0 ↔ 
      (x = 2 - Real.sqrt 3 ∨ x = p ∨ x = q ∨ x = 2 + Real.sqrt 3)) ∧
    p = -1 ∧ q = -3 := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l722_72212


namespace NUMINAMATH_CALUDE_cosine_axis_of_symmetry_l722_72252

/-- The axis of symmetry for a cosine function translated to the left by π/6 units -/
theorem cosine_axis_of_symmetry (k : ℤ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.cos (2 * (x + π / 6))
  ∃ (x : ℝ), x = -π / 6 + k * π / 2 ∧ 
    (∀ (y : ℝ), f (x - y) = f (x + y)) :=
sorry

end NUMINAMATH_CALUDE_cosine_axis_of_symmetry_l722_72252


namespace NUMINAMATH_CALUDE_det_evaluation_l722_72201

theorem det_evaluation (x z : ℝ) : 
  Matrix.det !![1, x, z; 1, x + z, 2*z; 1, x, x + 2*z] = z * (3*x + z) := by
  sorry

end NUMINAMATH_CALUDE_det_evaluation_l722_72201


namespace NUMINAMATH_CALUDE_parallel_line_slope_intercept_l722_72292

/-- The slope-intercept form of a line parallel to 4x + y - 2 = 0 and passing through (3, 2) -/
theorem parallel_line_slope_intercept :
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), 4 * x + y - 2 = 0 → y = -4 * x + b) ∧ 
    (2 = m * 3 + b) ∧
    (∀ (x y : ℝ), y = m * x + b ↔ y = -4 * x + 14) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_intercept_l722_72292


namespace NUMINAMATH_CALUDE_eulers_formula_l722_72273

/-- A planar graph is a graph that can be embedded in the plane without edge crossings. -/
structure PlanarGraph where
  V : Type u  -- Vertex type
  E : Type v  -- Edge type
  F : Type w  -- Face type
  vertex_count : Nat
  edge_count : Nat
  face_count : Nat
  is_connected : Bool

/-- Euler's formula for planar graphs -/
theorem eulers_formula (G : PlanarGraph) :
  G.is_connected → G.vertex_count - G.edge_count + G.face_count = 2 := by
  sorry

#check eulers_formula

end NUMINAMATH_CALUDE_eulers_formula_l722_72273


namespace NUMINAMATH_CALUDE_swimming_pool_area_l722_72202

/-- Theorem: Area of a rectangular swimming pool --/
theorem swimming_pool_area (w l : ℝ) (h1 : l = 3 * w + 10) (h2 : 2 * w + 2 * l = 320) :
  w * l = 4593.75 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_area_l722_72202


namespace NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l722_72233

/-- The number of ways to arrange animals in cages. -/
def arrange_animals (num_chickens num_dogs num_cats : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial num_chickens * Nat.factorial num_dogs * Nat.factorial num_cats

/-- The theorem stating the number of arrangements for the given problem. -/
theorem happy_valley_kennel_arrangements :
  arrange_animals 3 3 4 = 5184 :=
by sorry

end NUMINAMATH_CALUDE_happy_valley_kennel_arrangements_l722_72233


namespace NUMINAMATH_CALUDE_best_candidate_is_C_l722_72209

structure Participant where
  name : String
  average_score : Float
  variance : Float

def participants : List Participant := [
  { name := "A", average_score := 8.5, variance := 1.7 },
  { name := "B", average_score := 8.8, variance := 2.1 },
  { name := "C", average_score := 9.1, variance := 1.7 },
  { name := "D", average_score := 9.1, variance := 2.5 }
]

def is_best_candidate (p : Participant) : Prop :=
  ∀ q ∈ participants,
    (p.average_score > q.average_score ∨
    (p.average_score = q.average_score ∧ p.variance ≤ q.variance))

theorem best_candidate_is_C :
  ∃ p ∈ participants, p.name = "C" ∧ is_best_candidate p :=
by sorry

end NUMINAMATH_CALUDE_best_candidate_is_C_l722_72209


namespace NUMINAMATH_CALUDE_kevin_ran_17_miles_l722_72278

/-- Calculates the total distance Kevin ran given his running segments -/
def kevin_total_distance (speed1 speed2 speed3 : ℝ) (time1 time2 time3 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3

/-- Theorem stating that Kevin's total distance is 17 miles -/
theorem kevin_ran_17_miles :
  kevin_total_distance 10 20 8 0.5 0.5 0.25 = 17 := by
  sorry

#eval kevin_total_distance 10 20 8 0.5 0.5 0.25

end NUMINAMATH_CALUDE_kevin_ran_17_miles_l722_72278


namespace NUMINAMATH_CALUDE_different_subject_book_choices_l722_72275

def chinese_books : ℕ := 8
def math_books : ℕ := 6
def english_books : ℕ := 5

theorem different_subject_book_choices :
  chinese_books * math_books + 
  chinese_books * english_books + 
  math_books * english_books = 118 := by
  sorry

end NUMINAMATH_CALUDE_different_subject_book_choices_l722_72275


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l722_72221

/-- Represents a workshop with its production quantity -/
structure Workshop where
  production : ℕ

/-- Represents a stratified sampling scenario -/
structure StratifiedSampling where
  workshops : List Workshop
  sampleSizes : List ℕ

def StratifiedSampling.totalSampleSize (s : StratifiedSampling) : ℕ :=
  s.sampleSizes.sum

def StratifiedSampling.isValid (s : StratifiedSampling) : Prop :=
  s.workshops.length = s.sampleSizes.length ∧ 
  s.sampleSizes.all (· > 0)

theorem stratified_sampling_theorem (s : StratifiedSampling) 
  (h1 : s.workshops = [⟨120⟩, ⟨90⟩, ⟨60⟩])
  (h2 : s.sampleSizes.length = 3)
  (h3 : s.sampleSizes[2] = 2)
  (h4 : s.isValid)
  (h5 : ∀ s' : StratifiedSampling, s'.workshops = s.workshops → 
        s'.isValid → s'.totalSampleSize ≥ s.totalSampleSize) :
  s.totalSampleSize = 9 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l722_72221


namespace NUMINAMATH_CALUDE_sophia_estimate_l722_72231

theorem sophia_estimate (x y a b : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : a > 0) (h4 : b > 0) :
  (x + a) - (y - b) > x - y := by
  sorry

end NUMINAMATH_CALUDE_sophia_estimate_l722_72231


namespace NUMINAMATH_CALUDE_two_stretches_to_similar_triangle_l722_72216

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define a stretch transformation
structure Stretch where
  center : Point2D
  coefficient : ℝ

-- Define similarity between triangles
def Similar (t1 t2 : Triangle) : Prop := sorry

-- Define the application of a stretch to a triangle
def ApplyStretch (s : Stretch) (t : Triangle) : Triangle := sorry

-- Theorem statement
theorem two_stretches_to_similar_triangle 
  (ABC : Triangle) (DEF : Triangle) (h : DEF.A.x = DEF.A.y ∧ DEF.B.x = DEF.B.y) :
  ∃ (S1 S2 : Stretch), Similar (ApplyStretch S2 (ApplyStretch S1 ABC)) DEF := by
  sorry

end NUMINAMATH_CALUDE_two_stretches_to_similar_triangle_l722_72216


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l722_72263

theorem arithmetic_mean_problem (a b c : ℝ) :
  (a + b + c + 105) / 4 = 93 →
  (a + b + c) / 3 = 89 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l722_72263


namespace NUMINAMATH_CALUDE_opposite_of_one_seventh_l722_72251

theorem opposite_of_one_seventh :
  ∀ x : ℚ, x + (1 / 7) = 0 ↔ x = -(1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_seventh_l722_72251


namespace NUMINAMATH_CALUDE_quadratic_roots_positive_conditions_l722_72290

theorem quadratic_roots_positive_conditions (a b c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) →
  (b^2 - 4*a*c ≥ 0 ∧ a*c > 0 ∧ a*b < 0) ∧
  ¬(b^2 - 4*a*c ≥ 0 ∧ a*c > 0 ∧ a*b < 0 → 
    ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_positive_conditions_l722_72290


namespace NUMINAMATH_CALUDE_math_department_candidates_l722_72271

theorem math_department_candidates :
  ∀ (m : ℕ),
    (∃ (cs_candidates : ℕ),
      cs_candidates = 7 ∧
      (Nat.choose cs_candidates 2) * m = 84) →
    m = 4 :=
by sorry

end NUMINAMATH_CALUDE_math_department_candidates_l722_72271


namespace NUMINAMATH_CALUDE_car_speed_l722_72225

/-- Given a car that travels 375 km in 3 hours, its speed is 125 km/h -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 375 ∧ time = 3 → speed = distance / time → speed = 125 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l722_72225


namespace NUMINAMATH_CALUDE_cubic_root_sum_l722_72232

theorem cubic_root_sum (p q : ℝ) : 
  (∃ x : ℂ, x^3 + p*x + q = 0 ∧ x = 2 + Complex.I) → p + q = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l722_72232


namespace NUMINAMATH_CALUDE_minimum_cost_for_227_students_l722_72218

/-- Represents the cost structure for notebooks -/
structure NotebookPricing where
  single_cost : ℝ
  dozen_cost : ℝ
  bulk_dozen_cost : ℝ
  bulk_threshold : ℕ

/-- Calculates the minimum cost for a given number of notebooks -/
def minimum_cost (pricing : NotebookPricing) (num_students : ℕ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem minimum_cost_for_227_students :
  let pricing : NotebookPricing := {
    single_cost := 0.3,
    dozen_cost := 3.0,
    bulk_dozen_cost := 2.7,
    bulk_threshold := 10
  }
  minimum_cost pricing 227 = 51.3 := by sorry

end NUMINAMATH_CALUDE_minimum_cost_for_227_students_l722_72218


namespace NUMINAMATH_CALUDE_regression_line_not_most_points_l722_72264

/-- A type representing a scatter plot of data points. -/
structure ScatterPlot where
  points : Set (ℝ × ℝ)

/-- A type representing a line in 2D space. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The regression line for a given scatter plot. -/
noncomputable def regressionLine (plot : ScatterPlot) : Line :=
  sorry

/-- The number of points a line passes through in a scatter plot. -/
def pointsPassed (line : Line) (plot : ScatterPlot) : ℕ :=
  sorry

/-- The statement that the regression line passes through the most points. -/
def regressionLinePassesMostPoints (plot : ScatterPlot) : Prop :=
  ∀ l : Line, pointsPassed (regressionLine plot) plot ≥ pointsPassed l plot

/-- Theorem stating that the regression line does not necessarily pass through the most points. -/
theorem regression_line_not_most_points :
  ∃ plot : ScatterPlot, ¬(regressionLinePassesMostPoints plot) :=
sorry

end NUMINAMATH_CALUDE_regression_line_not_most_points_l722_72264


namespace NUMINAMATH_CALUDE_stream_current_is_three_l722_72204

/-- Represents the rowing scenario described in the problem -/
structure RowingScenario where
  r : ℝ  -- man's rowing speed in still water (miles per hour)
  c : ℝ  -- speed of the stream's current (miles per hour)
  distance : ℝ  -- distance traveled (miles)
  timeDiffNormal : ℝ  -- time difference between upstream and downstream at normal rate (hours)
  timeDiffTripled : ℝ  -- time difference between upstream and downstream at tripled rate (hours)

/-- The theorem stating that given the problem conditions, the stream's current is 3 mph -/
theorem stream_current_is_three 
  (scenario : RowingScenario)
  (h1 : scenario.distance = 20)
  (h2 : scenario.timeDiffNormal = 6)
  (h3 : scenario.timeDiffTripled = 1.5)
  (h4 : scenario.distance / (scenario.r + scenario.c) + scenario.timeDiffNormal = 
        scenario.distance / (scenario.r - scenario.c))
  (h5 : scenario.distance / (3 * scenario.r + scenario.c) + scenario.timeDiffTripled = 
        scenario.distance / (3 * scenario.r - scenario.c))
  : scenario.c = 3 := by
  sorry

#check stream_current_is_three

end NUMINAMATH_CALUDE_stream_current_is_three_l722_72204


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_b_l722_72236

theorem gcd_of_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 528 * k) :
  Nat.gcd (3 * b^3 + b^2 + 4 * b + 66).natAbs b.natAbs = 66 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_b_l722_72236


namespace NUMINAMATH_CALUDE_pat_initial_stickers_l722_72243

/-- The number of stickers Pat had at the end of the week -/
def end_stickers : ℕ := 61

/-- The number of stickers Pat earned during the week -/
def earned_stickers : ℕ := 22

/-- The number of stickers Pat had on the first day of the week -/
def initial_stickers : ℕ := end_stickers - earned_stickers

theorem pat_initial_stickers : initial_stickers = 39 := by
  sorry

end NUMINAMATH_CALUDE_pat_initial_stickers_l722_72243


namespace NUMINAMATH_CALUDE_new_rectangle_area_l722_72222

theorem new_rectangle_area (a b : ℝ) (h : a > b) :
  let base := a^2 + b^2 + a
  let height := a^2 + b^2 - b
  base * height = a^4 + a^3 + 2*a^2*b^2 + a*b^3 - a*b + b^4 - b^3 - b^2 :=
by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l722_72222


namespace NUMINAMATH_CALUDE_only_classmate_exercise_comprehensive_comprehensive_investigation_survey_l722_72224

/-- Represents a survey option -/
inductive SurveyOption
  | ClassmateExercise
  | CarCrashResistance
  | GalaViewership
  | ShoeSoleBending

/-- Defines the characteristics of a comprehensive investigation -/
def isComprehensive (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.ClassmateExercise => true
  | _ => false

/-- Theorem stating that only the classmate exercise survey is comprehensive -/
theorem only_classmate_exercise_comprehensive :
  ∀ s : SurveyOption, isComprehensive s ↔ s = SurveyOption.ClassmateExercise :=
by sorry

/-- Main theorem proving which survey is suitable for a comprehensive investigation -/
theorem comprehensive_investigation_survey :
  ∃! s : SurveyOption, isComprehensive s :=
by sorry

end NUMINAMATH_CALUDE_only_classmate_exercise_comprehensive_comprehensive_investigation_survey_l722_72224


namespace NUMINAMATH_CALUDE_labeling_periodic_l722_72245

/-- Represents the labeling of vertices at a given time -/
def Labeling := Fin 1993 → Int

/-- The rule for updating labels -/
def update_label (l : Labeling) (n : Fin 1993) : Int :=
  if l (n - 1) = l (n + 1) then 1 else -1

/-- The next labeling based on the current one -/
def next_labeling (l : Labeling) : Labeling :=
  fun n => update_label l n

/-- The labeling after t steps -/
def labeling_at_time (initial : Labeling) : ℕ → Labeling
  | 0 => initial
  | t + 1 => next_labeling (labeling_at_time initial t)

theorem labeling_periodic (initial : Labeling) :
  ∃ n : ℕ, n > 1 ∧ labeling_at_time initial n = labeling_at_time initial 1 := by
  sorry

end NUMINAMATH_CALUDE_labeling_periodic_l722_72245


namespace NUMINAMATH_CALUDE_function_zero_points_theorem_l722_72276

open Real

theorem function_zero_points_theorem (f : ℝ → ℝ) (a : ℝ) (x₁ x₂ : ℝ) 
  (h_f : ∀ x, f x = log x - a * x)
  (h_zero : f x₁ = 0 ∧ f x₂ = 0)
  (h_distinct : x₁ < x₂) :
  (0 < a ∧ a < 1 / Real.exp 1) ∧ 
  (2 / (x₁ + x₂) < a) := by
  sorry

end NUMINAMATH_CALUDE_function_zero_points_theorem_l722_72276


namespace NUMINAMATH_CALUDE_sin_double_alpha_l722_72270

theorem sin_double_alpha (α : Real) (h : Real.cos (π / 4 - α) = 4 / 5) : 
  Real.sin (2 * α) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_alpha_l722_72270


namespace NUMINAMATH_CALUDE_certain_number_exists_l722_72265

theorem certain_number_exists : ∃ x : ℝ, 
  (x * 0.0729 * 28.9) / (0.0017 * 0.025 * 8.1) = 382.5 ∧ 
  abs (x - 50.35) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l722_72265


namespace NUMINAMATH_CALUDE_distance_from_point_to_line_l722_72208

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a line in polar form ρ cos θ = a -/
structure PolarLine where
  a : ℝ

/-- Calculates the distance from a point in polar coordinates to a polar line -/
def distanceFromPointToLine (p : PolarPoint) (l : PolarLine) : ℝ :=
  sorry

theorem distance_from_point_to_line :
  let p : PolarPoint := ⟨1, Real.pi / 2⟩
  let l : PolarLine := ⟨2⟩
  distanceFromPointToLine p l = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_line_l722_72208


namespace NUMINAMATH_CALUDE_complex_point_on_line_l722_72205

theorem complex_point_on_line (a : ℝ) : 
  let z₁ : ℂ := 1 - a * Complex.I
  let z₂ : ℂ := (2 + Complex.I) ^ 2
  let z : ℂ := z₁ / z₂
  (5 * z.re - 5 * z.im + 3 = 0) → a = 22 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l722_72205


namespace NUMINAMATH_CALUDE_equation_solution_l722_72228

theorem equation_solution : 
  ∃ (x : ℚ), (3/4 : ℚ) + 4/x = 1 ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l722_72228


namespace NUMINAMATH_CALUDE_complement_intersection_equality_l722_72239

def S : Set Nat := {1,2,3,4,5}
def M : Set Nat := {1,4}
def N : Set Nat := {2,4}

theorem complement_intersection_equality : 
  (S \ M) ∩ (S \ N) = {3,5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equality_l722_72239


namespace NUMINAMATH_CALUDE_sin_pi_12_plus_theta_l722_72211

theorem sin_pi_12_plus_theta (θ : Real) 
  (h : Real.cos (5 * Real.pi / 12 - θ) = 1 / 3) : 
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_12_plus_theta_l722_72211


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l722_72242

theorem consecutive_integers_average (c : ℕ) (d : ℚ) : 
  (c > 0) →
  (d = (2 * c + (2 * c + 1) + (2 * c + 2) + (2 * c + 3) + (2 * c + 4) + (2 * c + 5) + (2 * c + 6)) / 7) →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = 2 * c + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l722_72242


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l722_72256

theorem smallest_five_digit_congruent_to_3_mod_17 : ∃ n : ℕ,
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit positive integer
  (n % 17 = 3) ∧              -- congruent to 3 modulo 17
  (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 3 → m ≥ n) ∧  -- smallest such integer
  n = 10012 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_3_mod_17_l722_72256


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l722_72227

/-- Given an arithmetic sequence {a_n} where a_2 = 3 and S_4 = 16, prove S_9 = 81 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 2 = 3 →                            -- given condition
  S 4 = 16 →                           -- given condition
  S 9 = 81 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l722_72227


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l722_72246

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 1 = 0 → x₂^2 - 2*x₂ - 1 = 0 → x₁^2 + x₂^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l722_72246


namespace NUMINAMATH_CALUDE_a_in_range_l722_72223

/-- A function f(x) = ax^2 + (a-3)x + 1 that is decreasing on [-1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

/-- The property that f is decreasing on [-1, +∞) -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, -1 ≤ x ∧ x < y → f a y < f a x

/-- The theorem stating that if f is decreasing on [-1, +∞), then a is in [-3, 0) -/
theorem a_in_range (a : ℝ) : is_decreasing_on_interval a → a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_a_in_range_l722_72223


namespace NUMINAMATH_CALUDE_hcf_problem_l722_72288

theorem hcf_problem (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b = 55 →
  Nat.lcm a b = 120 →
  (1 : ℚ) / a + (1 : ℚ) / b = 11 / 120 →
  Nat.gcd a b = 5 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l722_72288


namespace NUMINAMATH_CALUDE_library_repacking_l722_72249

theorem library_repacking (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) : 
  initial_boxes = 1500 → 
  books_per_initial_box = 45 → 
  books_per_new_box = 47 → 
  (initial_boxes * books_per_initial_box) % books_per_new_box = 8 := by
  sorry

end NUMINAMATH_CALUDE_library_repacking_l722_72249


namespace NUMINAMATH_CALUDE_total_initial_tickets_l722_72294

def dave_tiger_original_price : ℝ := 43
def dave_tiger_discount_rate : ℝ := 0.20
def dave_keychain_price : ℝ := 5.5
def dave_tickets_left : ℝ := 55

def alex_dinosaur_original_price : ℝ := 65
def alex_dinosaur_discount_rate : ℝ := 0.15
def alex_tickets_left : ℝ := 42

theorem total_initial_tickets : 
  let dave_tiger_discounted_price := dave_tiger_original_price * (1 - dave_tiger_discount_rate)
  let dave_total_spent := dave_tiger_discounted_price + dave_keychain_price
  let dave_initial_tickets := dave_total_spent + dave_tickets_left

  let alex_dinosaur_discounted_price := alex_dinosaur_original_price * (1 - alex_dinosaur_discount_rate)
  let alex_initial_tickets := alex_dinosaur_discounted_price + alex_tickets_left

  dave_initial_tickets + alex_initial_tickets = 192.15 := by sorry

end NUMINAMATH_CALUDE_total_initial_tickets_l722_72294


namespace NUMINAMATH_CALUDE_pizza_bill_division_l722_72237

theorem pizza_bill_division (total_price : ℝ) (num_people : ℕ) (individual_payment : ℝ) :
  total_price = 40 →
  num_people = 5 →
  individual_payment = total_price / num_people →
  individual_payment = 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_bill_division_l722_72237


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l722_72258

/-- Given two lines l₁ and l₂, prove that their distance is √10/5 -/
theorem distance_between_parallel_lines (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | 2*x + 3*m*y - m + 2 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | m*x + 6*y - 4 = 0}
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ → (x₂, y₂) ∈ l₂ → 
    (2 * (y₂ - y₁) = 3*m * (x₂ - x₁))) →  -- parallel condition
  (∃ (d : ℝ), d = Real.sqrt 10 / 5 ∧
    ∀ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ l₁ → p₂ ∈ l₂ →
      d ≤ Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_parallel_lines_l722_72258


namespace NUMINAMATH_CALUDE_monday_is_42_l722_72244

/-- Represents the temperature on each day of the week --/
structure WeekTemperatures where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The average temperature for Monday to Thursday is 48 degrees --/
def avg_mon_to_thu (w : WeekTemperatures) : Prop :=
  (w.monday + w.tuesday + w.wednesday + w.thursday) / 4 = 48

/-- The average temperature for Tuesday to Friday is 46 degrees --/
def avg_tue_to_fri (w : WeekTemperatures) : Prop :=
  (w.tuesday + w.wednesday + w.thursday + w.friday) / 4 = 46

/-- The temperature on Friday is 34 degrees --/
def friday_temp (w : WeekTemperatures) : Prop :=
  w.friday = 34

/-- Some day has a temperature of 42 degrees --/
def some_day_42 (w : WeekTemperatures) : Prop :=
  w.monday = 42 ∨ w.tuesday = 42 ∨ w.wednesday = 42 ∨ w.thursday = 42 ∨ w.friday = 42

theorem monday_is_42 (w : WeekTemperatures) 
  (h1 : avg_mon_to_thu w) 
  (h2 : avg_tue_to_fri w) 
  (h3 : friday_temp w) 
  (h4 : some_day_42 w) : 
  w.monday = 42 := by
  sorry

end NUMINAMATH_CALUDE_monday_is_42_l722_72244


namespace NUMINAMATH_CALUDE_range_of_a_l722_72220

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x y : ℝ, x - y + a = 0 ∧ x^2 + y^2 - 2*x = 1

def q (a : ℝ) : Prop := ∀ x : ℝ, Real.exp x - a > 1

-- State the theorem
theorem range_of_a (a : ℝ) : p a ∧ ¬(q a) → -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l722_72220


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l722_72255

/-- Represents a contestant's score for a single day -/
structure DailyScore where
  scored : ℚ
  attempted : ℚ

/-- Represents a contestant's scores for the three-day contest -/
structure ContestScore where
  day1 : DailyScore
  day2 : DailyScore
  day3 : DailyScore

def Charlie : ContestScore :=
  { day1 := { scored := 200, attempted := 300 },
    day2 := { scored := 160, attempted := 200 },
    day3 := { scored := 90, attempted := 100 } }

def totalAttempted (score : ContestScore) : ℚ :=
  score.day1.attempted + score.day2.attempted + score.day3.attempted

def totalScored (score : ContestScore) : ℚ :=
  score.day1.scored + score.day2.scored + score.day3.scored

def successRatio (score : ContestScore) : ℚ :=
  totalScored score / totalAttempted score

def dailySuccessRatio (day : DailyScore) : ℚ :=
  day.scored / day.attempted

theorem delta_max_success_ratio :
  ∀ delta : ContestScore,
    totalAttempted delta = 600 →
    dailySuccessRatio delta.day1 < dailySuccessRatio Charlie.day1 →
    dailySuccessRatio delta.day2 < dailySuccessRatio Charlie.day2 →
    dailySuccessRatio delta.day3 < dailySuccessRatio Charlie.day3 →
    delta.day1.attempted ≠ Charlie.day1.attempted →
    delta.day2.attempted ≠ Charlie.day2.attempted →
    delta.day3.attempted ≠ Charlie.day3.attempted →
    successRatio delta ≤ 399 / 600 :=
by sorry

#check delta_max_success_ratio

end NUMINAMATH_CALUDE_delta_max_success_ratio_l722_72255


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l722_72215

/-- Given a hyperbola and a line intersecting at two points, prove a relation between a and b -/
theorem hyperbola_line_intersection (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (∃ (P Q : ℝ × ℝ),
    -- P and Q lie on the hyperbola
    (P.1^2 / a - P.2^2 / b = 1) ∧
    (Q.1^2 / a - Q.2^2 / b = 1) ∧
    -- P and Q lie on the line
    (P.1 + P.2 = 1) ∧
    (Q.1 + Q.2 = 1) ∧
    -- OP is perpendicular to OQ
    (P.1 * Q.1 + P.2 * Q.2 = 0)) →
  1 / a - 1 / b = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l722_72215


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_l722_72253

/-- The probability of selecting at least one female student when choosing 3 students from a group of 3 male and 2 female students. -/
theorem prob_at_least_one_female (n_male : ℕ) (n_female : ℕ) (n_select : ℕ) : 
  n_male = 3 → n_female = 2 → n_select = 3 →
  (1 : ℚ) - (Nat.choose n_male n_select : ℚ) / (Nat.choose (n_male + n_female) n_select : ℚ) = 9/10 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_l722_72253


namespace NUMINAMATH_CALUDE_fraction_simplification_l722_72259

theorem fraction_simplification (x y z : ℚ) :
  x = 3 ∧ y = 4 ∧ z = 2 →
  (10 * x * y^3) / (15 * x^2 * y * z) = 16 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l722_72259


namespace NUMINAMATH_CALUDE_raspberry_pie_degrees_l722_72238

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The total number of students in the class -/
def total_students : ℕ := 48

/-- The number of students preferring chocolate pie -/
def chocolate_pref : ℕ := 18

/-- The number of students preferring apple pie -/
def apple_pref : ℕ := 10

/-- The number of students preferring blueberry pie -/
def blueberry_pref : ℕ := 8

/-- Theorem stating that the number of degrees for raspberry pie in the pie chart is 45 -/
theorem raspberry_pie_degrees : 
  let remaining := total_students - (chocolate_pref + apple_pref + blueberry_pref)
  let raspberry_pref := remaining / 2
  (raspberry_pref : ℚ) / total_students * full_circle = 45 := by sorry

end NUMINAMATH_CALUDE_raspberry_pie_degrees_l722_72238


namespace NUMINAMATH_CALUDE_trig_expressions_given_tan_alpha_l722_72262

theorem trig_expressions_given_tan_alpha (α : Real) (h : Real.tan α = -2) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = 5 ∧
  1 / (Real.sin α * Real.cos α) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expressions_given_tan_alpha_l722_72262


namespace NUMINAMATH_CALUDE_reciprocal_problem_l722_72250

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 3) : 50 * (1 / x) = 400 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l722_72250


namespace NUMINAMATH_CALUDE_C_power_50_l722_72282

def C : Matrix (Fin 2) (Fin 2) ℤ := !![2, 1; -4, -1]

theorem C_power_50 : C^50 = !![4^49 + 1, 4^49; -4^50, -2 * 4^49 + 1] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l722_72282


namespace NUMINAMATH_CALUDE_digits_of_3_15_times_5_10_l722_72285

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The number of digits in 3^15 * 5^10 is 14 -/
theorem digits_of_3_15_times_5_10 : num_digits (3^15 * 5^10) = 14 := by sorry

end NUMINAMATH_CALUDE_digits_of_3_15_times_5_10_l722_72285


namespace NUMINAMATH_CALUDE_total_students_l722_72235

theorem total_students (S : ℕ) (T : ℕ) : 
  T = 6 * S - 78 →
  T - S = 2222 →
  T = 2682 := by
sorry

end NUMINAMATH_CALUDE_total_students_l722_72235


namespace NUMINAMATH_CALUDE_max_value_of_expression_l722_72289

theorem max_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (2 * a) / (a^2 + b) + b / (a + b^2) ≤ (2 * Real.sqrt 3 + 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l722_72289


namespace NUMINAMATH_CALUDE_inequality_solution_l722_72274

theorem inequality_solution (x : ℕ+) : 
  (12 * x + 5 < 10 * x + 15) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l722_72274


namespace NUMINAMATH_CALUDE_sum_of_roots_l722_72280

theorem sum_of_roots (x : ℝ) : 
  (∃ a b : ℝ, (2*x + 3)*(x - 4) + (2*x + 3)*(x - 6) = 0 ∧ 
   {y : ℝ | (2*y + 3)*(y - 4) + (2*y + 3)*(y - 6) = 0} = {a, b} ∧
   a + b = 7/2) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l722_72280


namespace NUMINAMATH_CALUDE_star_six_three_l722_72240

def star (a b : ℝ) : ℝ := 4*a + 3*b - 2*a*b

theorem star_six_three : star 6 3 = -3 := by sorry

end NUMINAMATH_CALUDE_star_six_three_l722_72240


namespace NUMINAMATH_CALUDE_multiplication_properties_l722_72229

theorem multiplication_properties : 
  (∀ n : ℝ, n * 0 = 0) ∧ 
  (∀ n : ℝ, n * 1 = n) ∧ 
  (∀ n : ℝ, n * (-1) = -n) ∧ 
  (∃ a b : ℝ, a + b = 0 ∧ a * b ≠ 1) := by
sorry

end NUMINAMATH_CALUDE_multiplication_properties_l722_72229


namespace NUMINAMATH_CALUDE_delegates_without_badges_l722_72234

theorem delegates_without_badges (total : ℕ) (preprinted : ℕ) : 
  total = 36 → preprinted = 16 → (total - preprinted - (total - preprinted) / 2) = 10 := by
sorry

end NUMINAMATH_CALUDE_delegates_without_badges_l722_72234


namespace NUMINAMATH_CALUDE_number_added_after_doubling_l722_72226

theorem number_added_after_doubling (x y : ℝ) : x = 4 → 3 * (2 * x + y) = 51 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_added_after_doubling_l722_72226


namespace NUMINAMATH_CALUDE_derivative_sin_cos_plus_one_l722_72283

theorem derivative_sin_cos_plus_one (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin x * (Real.cos x + 1)
  (deriv f) x = Real.cos (2 * x) + Real.cos x := by
sorry

end NUMINAMATH_CALUDE_derivative_sin_cos_plus_one_l722_72283


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_count_l722_72284

theorem quadratic_integer_roots_count :
  ∃! (S : Finset ℝ), 
    (∀ a ∈ S, ∃ r s : ℤ, r^2 + a*r + 9*a = 0 ∧ s^2 + a*s + 9*a = 0) ∧
    (∀ a : ℝ, (∃ r s : ℤ, r^2 + a*r + 9*a = 0 ∧ s^2 + a*s + 9*a = 0) → a ∈ S) ∧
    S.card = 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_count_l722_72284


namespace NUMINAMATH_CALUDE_equation_represents_parabola_l722_72268

/-- The equation represents a parabola if it can be transformed into the form x² + bx + c = Ay + B --/
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c A B : ℝ, a ≠ 0 ∧ 
  ∀ x y : ℝ, f x y ↔ a * x^2 + b * x + c = A * y + B

/-- The given equation |y-3| = √((x+4)² + y²) --/
def given_equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 4)^2 + y^2)

theorem equation_represents_parabola : is_parabola given_equation := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_parabola_l722_72268


namespace NUMINAMATH_CALUDE_batsman_average_increase_proof_l722_72230

def batsman_average_increase (total_innings : ℕ) (last_innings_score : ℕ) (final_average : ℚ) : ℚ :=
  let initial_total := (total_innings - 1) * (total_innings * final_average - last_innings_score) / total_innings
  let initial_average := initial_total / (total_innings - 1)
  final_average - initial_average

theorem batsman_average_increase_proof :
  batsman_average_increase 12 65 32 = 3 := by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_proof_l722_72230


namespace NUMINAMATH_CALUDE_reciprocal_of_2024_l722_72260

theorem reciprocal_of_2024 : (2024⁻¹ : ℚ) = 1 / 2024 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2024_l722_72260
