import Mathlib

namespace NUMINAMATH_CALUDE_original_number_proof_l2172_217274

theorem original_number_proof (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Both parts are positive
  a ≤ b ∧          -- a is the smaller part
  a = 35 ∧         -- The smallest part is 35
  a / 7 = b / 9 →  -- The seventh part of the first equals the ninth part of the second
  a + b = 80       -- The original number is 80
  := by sorry

end NUMINAMATH_CALUDE_original_number_proof_l2172_217274


namespace NUMINAMATH_CALUDE_product_of_roots_l2172_217295

theorem product_of_roots (x : ℝ) : 
  ((x + 3) * (x - 4) = 22) → 
  (∃ y : ℝ, ((y + 3) * (y - 4) = 22) ∧ (x * y = -34)) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2172_217295


namespace NUMINAMATH_CALUDE_expression_value_l2172_217262

theorem expression_value : 3^(0^(2^2)) + ((3^1)^0)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2172_217262


namespace NUMINAMATH_CALUDE_min_value_theorem_l2172_217211

theorem min_value_theorem (x y : ℝ) (h : x^2 + y^2 + x*y = 315) :
  ∃ (m : ℝ), m = 105 ∧ ∀ z, x^2 + y^2 - x*y ≥ z → z ≤ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2172_217211


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l2172_217291

def repeating_decimal_03 : ℚ := 1 / 33
def repeating_decimal_8 : ℚ := 8 / 9

theorem product_of_repeating_decimals : 
  repeating_decimal_03 * repeating_decimal_8 = 8 / 297 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l2172_217291


namespace NUMINAMATH_CALUDE_sector_arc_length_l2172_217234

/-- Given a circular sector with area 24π cm² and central angle 216°, 
    its arc length is (12√10π)/5 cm. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 24 * Real.pi ∧ 
  angle = 216 →
  arc_length = (12 * Real.sqrt 10 * Real.pi) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2172_217234


namespace NUMINAMATH_CALUDE_monotonicity_condition_necessary_not_sufficient_l2172_217204

def f (a : ℝ) (x : ℝ) : ℝ := |a - 3*x|

theorem monotonicity_condition (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x ≤ f a y) ↔ a ≤ 3 :=
sorry

theorem necessary_not_sufficient :
  (∀ a : ℝ, (∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x ≤ f a y) → a = 3) ∧
  (∃ a : ℝ, a = 3 ∧ ¬(∀ x y : ℝ, 1 ≤ x ∧ x < y → f a x ≤ f a y)) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_condition_necessary_not_sufficient_l2172_217204


namespace NUMINAMATH_CALUDE_pollywogs_disappear_in_44_days_l2172_217225

/-- The number of days it takes for all pollywogs to disappear from the pond -/
def days_to_disappear (initial_pollywogs : ℕ) (maturation_rate : ℕ) (catching_rate : ℕ) (catching_duration : ℕ) : ℕ :=
  let combined_rate := maturation_rate + catching_rate
  let pollywogs_after_catching := initial_pollywogs - combined_rate * catching_duration
  let remaining_days := pollywogs_after_catching / maturation_rate
  catching_duration + remaining_days

/-- Theorem stating that it takes 44 days for all pollywogs to disappear from the pond -/
theorem pollywogs_disappear_in_44_days :
  days_to_disappear 2400 50 10 20 = 44 := by
  sorry

end NUMINAMATH_CALUDE_pollywogs_disappear_in_44_days_l2172_217225


namespace NUMINAMATH_CALUDE_dot_product_equals_25_l2172_217251

def a : ℝ × ℝ := (1, 2)

theorem dot_product_equals_25 (b : ℝ × ℝ) 
  (h : a - (1/5 : ℝ) • b = (-2, 1)) : 
  a • b = 25 := by sorry

end NUMINAMATH_CALUDE_dot_product_equals_25_l2172_217251


namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_family_l2172_217273

/-- The fixed point of a family of parabolas -/
theorem fixed_point_of_parabola_family :
  ∀ t : ℝ, (4 : ℝ) * 3^2 + 2 * t * 3 - 3 * t = 36 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_family_l2172_217273


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2172_217226

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  seventh_term : a 7 = -8
  seventeenth_term : a 17 = -28

/-- The general term formula for the arithmetic sequence -/
def generalTerm (seq : ArithmeticSequence) : ℕ → ℤ := 
  fun n => -2 * n + 6

/-- The sum of the first n terms of the arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) : ℕ → ℤ :=
  fun n => -n^2 + 5*n

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧ 
  (∃ k, ∀ n, sumOfTerms seq n ≤ sumOfTerms seq k) ∧
  (sumOfTerms seq 2 = 6 ∧ sumOfTerms seq 3 = 6) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2172_217226


namespace NUMINAMATH_CALUDE_adams_sandwiches_l2172_217281

/-- The number of sandwiches Adam bought -/
def num_sandwiches : ℕ := 3

/-- The cost of each sandwich in dollars -/
def sandwich_cost : ℕ := 3

/-- The cost of the water bottle in dollars -/
def water_cost : ℕ := 2

/-- The total cost of Adam's shopping in dollars -/
def total_cost : ℕ := 11

theorem adams_sandwiches :
  num_sandwiches * sandwich_cost + water_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_adams_sandwiches_l2172_217281


namespace NUMINAMATH_CALUDE_monotonic_increasing_cubic_l2172_217269

/-- A cubic function with parameters m and n. -/
def f (m n : ℝ) (x : ℝ) : ℝ := 4 * x^3 + m * x^2 + (m - 3) * x + n

/-- The derivative of f with respect to x. -/
def f' (m : ℝ) (x : ℝ) : ℝ := 12 * x^2 + 2 * m * x + (m - 3)

theorem monotonic_increasing_cubic (m n : ℝ) :
  (∀ x : ℝ, Monotone (f m n)) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_cubic_l2172_217269


namespace NUMINAMATH_CALUDE_remainder_sum_theorem_l2172_217292

theorem remainder_sum_theorem (n : ℕ) : 
  (∃ a b c : ℕ, 
    0 < a ∧ a < 29 ∧
    0 < b ∧ b < 41 ∧
    0 < c ∧ c < 59 ∧
    n % 29 = a ∧
    n % 41 = b ∧
    n % 59 = c ∧
    a + b + c = n) → 
  (n = 79 ∨ n = 114) :=
by sorry

end NUMINAMATH_CALUDE_remainder_sum_theorem_l2172_217292


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l2172_217268

-- Define the polynomial
def p (x : ℝ) : ℝ := 6 * (x^5 + 2*x^3 + x^2 + 3)

-- Define a function to get the coefficients of the expanded polynomial
def coefficients (p : ℝ → ℝ) : List ℝ := sorry

-- Define a function to calculate the sum of squares of a list of numbers
def sum_of_squares (l : List ℝ) : ℝ := sorry

-- Theorem statement
theorem sum_of_squares_of_coefficients :
  sum_of_squares (coefficients p) = 540 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l2172_217268


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2172_217288

open Real

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) +
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) +
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2172_217288


namespace NUMINAMATH_CALUDE_prime_square_sum_not_perfect_square_l2172_217275

theorem prime_square_sum_not_perfect_square
  (p q : ℕ) (hp : Prime p) (hq : Prime q)
  (h_perfect_square : ∃ a : ℕ, a > 0 ∧ p + q^2 = a^2) :
  ∀ n : ℕ, n > 0 → ¬∃ b : ℕ, b > 0 ∧ p^2 + q^n = b^2 :=
by sorry

end NUMINAMATH_CALUDE_prime_square_sum_not_perfect_square_l2172_217275


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2172_217289

def A : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}
def B : Set ℝ := {x : ℝ | (x+4)*(x-2) > 0}

theorem set_intersection_theorem : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2172_217289


namespace NUMINAMATH_CALUDE_find_divisor_with_remainder_relation_l2172_217258

theorem find_divisor_with_remainder_relation : ∃ (A : ℕ), 
  (312 % A = 2 * (270 % A)) ∧ 
  (270 % A = 2 * (211 % A)) ∧ 
  (A = 19) := by
sorry

end NUMINAMATH_CALUDE_find_divisor_with_remainder_relation_l2172_217258


namespace NUMINAMATH_CALUDE_dot_product_calculation_l2172_217239

theorem dot_product_calculation (a b : ℝ × ℝ) : 
  a = (2, 1) → a - 2 • b = (1, 1) → a • b = 1 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_calculation_l2172_217239


namespace NUMINAMATH_CALUDE_impossible_lake_system_dima_is_mistaken_l2172_217221

/-- Represents a lake system with a given number of lakes, outgoing rivers per lake, and incoming rivers per lake. -/
structure LakeSystem where
  num_lakes : ℕ
  outgoing_rivers_per_lake : ℕ
  incoming_rivers_per_lake : ℕ

/-- Theorem stating that a non-empty lake system with 3 outgoing and 4 incoming rivers per lake is impossible. -/
theorem impossible_lake_system : ¬∃ (ls : LakeSystem), ls.num_lakes > 0 ∧ ls.outgoing_rivers_per_lake = 3 ∧ ls.incoming_rivers_per_lake = 4 := by
  sorry

/-- Corollary stating that Dima's claim about the lake system in Vrunlandia is incorrect. -/
theorem dima_is_mistaken : ¬∃ (ls : LakeSystem), ls.num_lakes > 0 ∧ ls.outgoing_rivers_per_lake = 3 ∧ ls.incoming_rivers_per_lake = 4 := by
  exact impossible_lake_system

end NUMINAMATH_CALUDE_impossible_lake_system_dima_is_mistaken_l2172_217221


namespace NUMINAMATH_CALUDE_black_area_after_changes_l2172_217272

/-- Represents the fraction of the square that is black -/
def black_fraction : ℕ → ℚ
  | 0 => 1/2  -- Initially half the square is black
  | (n+1) => (3/4) * black_fraction n  -- Each change keeps 3/4 of the previous black area

/-- The number of changes applied to the square -/
def num_changes : ℕ := 6

theorem black_area_after_changes :
  black_fraction num_changes = 729/8192 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_changes_l2172_217272


namespace NUMINAMATH_CALUDE_correct_sampling_order_l2172_217249

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Define the properties of each sampling method
def isSimpleRandom (method : SamplingMethod) : Prop :=
  method = SamplingMethod.SimpleRandom

def isSystematic (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Systematic

def isStratified (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified

-- Define the properties of the given methods
def method1Properties (method : SamplingMethod) : Prop :=
  isSimpleRandom method

def method2Properties (method : SamplingMethod) : Prop :=
  isSystematic method

def method3Properties (method : SamplingMethod) : Prop :=
  isStratified method

-- Theorem statement
theorem correct_sampling_order :
  ∃ (m1 m2 m3 : SamplingMethod),
    method1Properties m1 ∧
    method2Properties m2 ∧
    method3Properties m3 ∧
    m1 = SamplingMethod.SimpleRandom ∧
    m2 = SamplingMethod.Systematic ∧
    m3 = SamplingMethod.Stratified :=
by
  sorry

end NUMINAMATH_CALUDE_correct_sampling_order_l2172_217249


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l2172_217213

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Fibonacci sequence modulo 9 repeats every 24 terms -/
axiom fib_mod_9_period : ∀ n, fib n % 9 = fib (n % 24) % 9

/-- The 6th Fibonacci number modulo 9 is 8 -/
axiom fib_6_mod_9 : fib 6 % 9 = 8

/-- The 150th Fibonacci number modulo 9 -/
theorem fib_150_mod_9 : fib 150 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l2172_217213


namespace NUMINAMATH_CALUDE_sin_theta_equals_sqrt2_over_2_l2172_217241

theorem sin_theta_equals_sqrt2_over_2 (θ : Real) (a : Real) (h1 : a ≠ 0) 
  (h2 : ∃ (x y : Real), x = a ∧ y = a ∧ Real.cos θ * Real.cos θ + Real.sin θ * Real.sin θ = 1 ∧ 
    Real.cos θ * x - Real.sin θ * y = 0 ∧ Real.sin θ * x + Real.cos θ * y = 0) : 
  |Real.sin θ| = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_equals_sqrt2_over_2_l2172_217241


namespace NUMINAMATH_CALUDE_max_vertex_product_sum_l2172_217271

/-- The set of numbers that can be assigned to the faces of the cube -/
def CubeNumbers : Finset ℕ := {1, 2, 3, 4, 8, 9}

/-- A valid assignment of numbers to the faces of a cube -/
structure CubeAssignment where
  faces : Fin 6 → ℕ
  valid : ∀ i, faces i ∈ CubeNumbers
  distinct : ∀ i j, i ≠ j → faces i ≠ faces j

/-- The sum of products at the vertices of a cube given a face assignment -/
def vertexProductSum (assignment : CubeAssignment) : ℕ :=
  let a := assignment.faces 0
  let b := assignment.faces 1
  let c := assignment.faces 2
  let d := assignment.faces 3
  let e := assignment.faces 4
  let f := assignment.faces 5
  (a + b) * (c + d) * (e + f)

/-- The maximum sum of products at the vertices of a cube -/
theorem max_vertex_product_sum :
  ∃ (assignment : CubeAssignment), ∀ (other : CubeAssignment),
    vertexProductSum assignment ≥ vertexProductSum other ∧
    vertexProductSum assignment = 729 :=
  sorry

end NUMINAMATH_CALUDE_max_vertex_product_sum_l2172_217271


namespace NUMINAMATH_CALUDE_machine_theorem_l2172_217236

def machine_step (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def machine_4_steps (n : ℕ) : ℕ :=
  machine_step (machine_step (machine_step (machine_step n)))

theorem machine_theorem :
  ∀ n : ℕ, n > 0 → (machine_4_steps n = 10 ↔ n = 3 ∨ n = 160) := by
  sorry

end NUMINAMATH_CALUDE_machine_theorem_l2172_217236


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2172_217203

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem tangent_slope_at_one :
  (deriv f) 1 = 1 / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2172_217203


namespace NUMINAMATH_CALUDE_ounces_per_cup_l2172_217257

theorem ounces_per_cup (container_capacity : ℕ) (soap_per_cup : ℕ) (total_soap : ℕ) :
  container_capacity = 40 ∧ soap_per_cup = 3 ∧ total_soap = 15 →
  ∃ (ounces_per_cup : ℕ), ounces_per_cup = 8 ∧ container_capacity = ounces_per_cup * (total_soap / soap_per_cup) :=
by sorry

end NUMINAMATH_CALUDE_ounces_per_cup_l2172_217257


namespace NUMINAMATH_CALUDE_inequality_proof_l2172_217240

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b / c) + (a * c / b) + (b * c / a) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2172_217240


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2172_217286

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2172_217286


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_3i_l2172_217264

theorem imaginary_part_of_2_minus_3i :
  Complex.im (2 - 3 * Complex.I) = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_3i_l2172_217264


namespace NUMINAMATH_CALUDE_arithmetic_sum_problem_l2172_217293

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ + a₃ = 2 and a₃ + a₅ = 4, prove a₅ + a₇ = 6. -/
theorem arithmetic_sum_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum1 : a 1 + a 3 = 2) 
  (h_sum2 : a 3 + a 5 = 4) : 
  a 5 + a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_problem_l2172_217293


namespace NUMINAMATH_CALUDE_candle_illumination_theorem_l2172_217222

/-- Represents a wall in a room -/
structure Wall where
  -- Add necessary properties for a wall

/-- Represents a candle in a room -/
structure Candle where
  -- Add necessary properties for a candle

/-- Represents a room with walls and a candle -/
structure Room where
  walls : List Wall
  candle : Candle

/-- Predicate to check if a wall is completely illuminated by a candle -/
def is_completely_illuminated (w : Wall) (c : Candle) : Prop :=
  sorry

/-- Theorem stating that for a room with n walls (where n is 10 or 6),
    there exists a configuration where a single candle can be placed
    such that no wall is completely illuminated -/
theorem candle_illumination_theorem (n : Nat) (h : n = 10 ∨ n = 6) :
  ∃ (r : Room), r.walls.length = n ∧ ∀ w ∈ r.walls, ¬is_completely_illuminated w r.candle :=
sorry

end NUMINAMATH_CALUDE_candle_illumination_theorem_l2172_217222


namespace NUMINAMATH_CALUDE_range_of_m_l2172_217247

theorem range_of_m (x m : ℝ) : 
  (∀ x, (1/3 < x ∧ x < 1/2) → |x - m| < 1) →
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2172_217247


namespace NUMINAMATH_CALUDE_not_divisible_by_twelve_l2172_217245

theorem not_divisible_by_twelve (m : ℕ) (h1 : m > 0) 
  (h2 : ∃ (j : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 9 + (1 : ℚ) / m = j) : 
  ¬(12 ∣ m) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_twelve_l2172_217245


namespace NUMINAMATH_CALUDE_pyramid_height_l2172_217237

theorem pyramid_height (perimeter : ℝ) (apex_to_vertex : ℝ) (h_perimeter : perimeter = 32) (h_apex : apex_to_vertex = 12) :
  let side := perimeter / 4
  let center_to_corner := side * Real.sqrt 2 / 2
  Real.sqrt (apex_to_vertex ^ 2 - center_to_corner ^ 2) = 4 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_l2172_217237


namespace NUMINAMATH_CALUDE_smallest_cube_multiple_l2172_217298

theorem smallest_cube_multiple : 
  (∃ (x : ℕ+) (M : ℤ), 3960 * x.val = M^3) ∧ 
  (∀ (y : ℕ+) (N : ℤ), 3960 * y.val = N^3 → y.val ≥ 9075) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_multiple_l2172_217298


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2172_217276

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h1 : a 4 * a 8 = 4) : a 5 * a 6 * a 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2172_217276


namespace NUMINAMATH_CALUDE_carries_mom_payment_ratio_l2172_217267

/-- The ratio of the amount Carrie's mom pays to the total cost of all clothes -/
theorem carries_mom_payment_ratio :
  let shirt_count : ℕ := 4
  let pants_count : ℕ := 2
  let jacket_count : ℕ := 2
  let shirt_price : ℕ := 8
  let pants_price : ℕ := 18
  let jacket_price : ℕ := 60
  let carries_payment : ℕ := 94
  let total_cost : ℕ := shirt_count * shirt_price + pants_count * pants_price + jacket_count * jacket_price
  let moms_payment : ℕ := total_cost - carries_payment
  moms_payment * 2 = total_cost :=
by sorry

end NUMINAMATH_CALUDE_carries_mom_payment_ratio_l2172_217267


namespace NUMINAMATH_CALUDE_probability_of_four_white_balls_l2172_217246

def total_balls : ℕ := 25
def white_balls : ℕ := 10
def black_balls : ℕ := 15
def drawn_balls : ℕ := 4

theorem probability_of_four_white_balls : 
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls) = 3 / 181 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_four_white_balls_l2172_217246


namespace NUMINAMATH_CALUDE_estimate_negative_sqrt_17_l2172_217253

theorem estimate_negative_sqrt_17 : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end NUMINAMATH_CALUDE_estimate_negative_sqrt_17_l2172_217253


namespace NUMINAMATH_CALUDE_equation_solutions_l2172_217243

theorem equation_solutions :
  (∃ x : ℝ, x = 1/3 ∧ 3/(1-6*x) = 2/(6*x+1) - (8+9*x)/(36*x^2-1)) ∧
  (∃ z : ℝ, z = -3/7 ∧ 3/(1-z^2) = 2/((1+z)^2) - 5/((1-z)^2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2172_217243


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l2172_217278

theorem floor_ceiling_sum : ⌊(-0.123 : ℝ)⌋ + ⌈(4.567 : ℝ)⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l2172_217278


namespace NUMINAMATH_CALUDE_sea_horse_penguin_ratio_l2172_217279

/-- The number of sea horses at the zoo -/
def sea_horses : ℕ := 70

/-- The number of penguins at the zoo -/
def penguins : ℕ := sea_horses + 85

/-- The ratio of sea horses to penguins -/
def ratio : ℕ × ℕ := (14, 31)

/-- Theorem stating that the ratio of sea horses to penguins is 14:31 -/
theorem sea_horse_penguin_ratio :
  (sea_horses : ℚ) / (penguins : ℚ) = (ratio.1 : ℚ) / (ratio.2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_sea_horse_penguin_ratio_l2172_217279


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l2172_217277

theorem sum_of_fifth_powers (a b x y : ℝ) 
  (eq1 : a*x + b*y = 3)
  (eq2 : a*x^2 + b*y^2 = 7)
  (eq3 : a*x^3 + b*y^3 = 6)
  (eq4 : a*x^4 + b*y^4 = 42) :
  a*x^5 + b*y^5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l2172_217277


namespace NUMINAMATH_CALUDE_multiples_of_three_is_closed_set_l2172_217287

-- Define a closed set
def is_closed_set (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

-- Define the set A
def A : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

-- Theorem statement
theorem multiples_of_three_is_closed_set : is_closed_set A := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_three_is_closed_set_l2172_217287


namespace NUMINAMATH_CALUDE_area_of_region_l2172_217238

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 77 ∧ 
   A = Real.pi * (((x + 8)^2 + (y - 3)^2) / 4) ∧
   x^2 + y^2 - 8 = 6*y - 16*x + 4) :=
by sorry

end NUMINAMATH_CALUDE_area_of_region_l2172_217238


namespace NUMINAMATH_CALUDE_point_on_line_m_value_l2172_217202

/-- A point with coordinates (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line defined by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Predicate to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

theorem point_on_line_m_value :
  ∀ (m : ℝ),
  let A : Point := ⟨2, m⟩
  let L : Line := ⟨-2, 3⟩
  pointOnLine A L → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_m_value_l2172_217202


namespace NUMINAMATH_CALUDE_industrial_machine_output_l2172_217297

theorem industrial_machine_output (shirts_per_minute : ℕ) 
  (yesterday_minutes : ℕ) (today_shirts : ℕ) (total_shirts : ℕ) :
  yesterday_minutes = 12 →
  today_shirts = 14 →
  total_shirts = 156 →
  yesterday_minutes * shirts_per_minute + today_shirts = total_shirts →
  shirts_per_minute = 11 :=
by sorry

end NUMINAMATH_CALUDE_industrial_machine_output_l2172_217297


namespace NUMINAMATH_CALUDE_probability_two_in_same_box_is_12_25_l2172_217220

def num_balls : ℕ := 3
def num_boxes : ℕ := 5

def total_placements : ℕ := num_boxes ^ num_balls

def two_in_same_box_placements : ℕ := 
  (num_balls.choose 2) * (num_boxes.choose 1) * (num_boxes - 1)

def probability_two_in_same_box : ℚ := 
  two_in_same_box_placements / total_placements

theorem probability_two_in_same_box_is_12_25 : 
  probability_two_in_same_box = 12 / 25 := by sorry

end NUMINAMATH_CALUDE_probability_two_in_same_box_is_12_25_l2172_217220


namespace NUMINAMATH_CALUDE_linear_equation_result_l2172_217284

theorem linear_equation_result (m x : ℝ) : 
  (m^2 - 1 = 0) → 
  (m - 1 ≠ 0) → 
  ((m^2 - 1)*x^2 - (m - 1)*x - 8 = 0) →
  200*(x - m)*(x + 2*m) - 10*m = 2010 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_result_l2172_217284


namespace NUMINAMATH_CALUDE_min_value_of_f_l2172_217285

-- Define the function
def f (x : ℝ) : ℝ := x^4 - 4*x + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem min_value_of_f : ∃ (x : ℝ), x ∈ interval ∧ f x = 0 ∧ ∀ (y : ℝ), y ∈ interval → f y ≥ f x := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2172_217285


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2172_217250

theorem min_value_of_expression (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0)
  (hx : x₁^2 - 4*a*x₁ + 3*a^2 = 0 ∧ x₂^2 - 4*a*x₂ + 3*a^2 = 0) :
  ∃ (m : ℝ), m = (4 * Real.sqrt 3) / 3 ∧ 
  ∀ (y₁ y₂ : ℝ), y₁^2 - 4*a*y₁ + 3*a^2 = 0 ∧ y₂^2 - 4*a*y₂ + 3*a^2 = 0 → 
  y₁ + y₂ + a / (y₁ * y₂) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2172_217250


namespace NUMINAMATH_CALUDE_manuscript_cost_example_l2172_217294

def manuscript_cost (total_pages : ℕ) (revised_once : ℕ) (revised_twice : ℕ) (revised_thrice : ℕ) 
  (initial_cost : ℕ) (revision_cost : ℕ) : ℕ :=
  let no_revision := total_pages - (revised_once + revised_twice + revised_thrice)
  let cost_no_revision := no_revision * initial_cost
  let cost_revised_once := revised_once * (initial_cost + revision_cost)
  let cost_revised_twice := revised_twice * (initial_cost + 2 * revision_cost)
  let cost_revised_thrice := revised_thrice * (initial_cost + 3 * revision_cost)
  cost_no_revision + cost_revised_once + cost_revised_twice + cost_revised_thrice

theorem manuscript_cost_example : 
  manuscript_cost 300 55 35 25 8 6 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_example_l2172_217294


namespace NUMINAMATH_CALUDE_pencil_length_l2172_217224

/-- The total length of a pencil with purple, black, and blue sections -/
theorem pencil_length (purple_length black_length blue_length : ℝ) 
  (h1 : purple_length = 3)
  (h2 : black_length = 2)
  (h3 : blue_length = 1) :
  purple_length + black_length + blue_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l2172_217224


namespace NUMINAMATH_CALUDE_tourist_distribution_eq_105_l2172_217207

/-- The number of ways to distribute 8 tourists among 4 guides with exactly 2 tourists per guide -/
def tourist_distribution : ℕ :=
  (Nat.choose 8 2 * Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2) / 24

theorem tourist_distribution_eq_105 : tourist_distribution = 105 := by
  sorry

end NUMINAMATH_CALUDE_tourist_distribution_eq_105_l2172_217207


namespace NUMINAMATH_CALUDE_sign_of_a_l2172_217206

theorem sign_of_a (a b c : ℝ) (n : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : ((-2)^8 * a^3 * b^3 * c^(n-1)) * ((-3)^3 * a^2 * b^5 * c^(n+1)) > 0) : 
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_sign_of_a_l2172_217206


namespace NUMINAMATH_CALUDE_four_divided_by_p_l2172_217218

theorem four_divided_by_p (q p : ℝ) 
  (h1 : 4 / q = 18) 
  (h2 : p - q = 0.2777777777777778) : 
  4 / p = 8 := by sorry

end NUMINAMATH_CALUDE_four_divided_by_p_l2172_217218


namespace NUMINAMATH_CALUDE_num_terms_eq_original_number_div_10_l2172_217283

/-- The number of terms when 100^10 is written as the sum of tens -/
def num_terms : ℕ := 10^19

/-- The original number -/
def original_number : ℕ := 100^10

theorem num_terms_eq_original_number_div_10 : 
  num_terms = original_number / 10 := by sorry

end NUMINAMATH_CALUDE_num_terms_eq_original_number_div_10_l2172_217283


namespace NUMINAMATH_CALUDE_range_of_a_l2172_217214

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := StrictMono (fun x => Real.log x / Real.log a)

-- Define the theorem
theorem range_of_a :
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  {a : ℝ | (-2 < a ∧ a ≤ 1) ∨ (a ≥ 2)} = {a : ℝ | a ∈ Set.Ioc (-2) 1 ∪ Set.Ici 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2172_217214


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2172_217266

theorem complex_equation_solution (z : ℂ) : (1 - z = z * Complex.I) → z = (1/2 : ℂ) - (1/2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2172_217266


namespace NUMINAMATH_CALUDE_triangle_height_from_rectangle_l2172_217265

/-- Given a 9x27 rectangle cut into two congruent trapezoids and rearranged to form a triangle with base 9, the height of the resulting triangle is 54 units. -/
theorem triangle_height_from_rectangle (rectangle_length : ℝ) (rectangle_width : ℝ) (triangle_base : ℝ) :
  rectangle_length = 27 →
  rectangle_width = 9 →
  triangle_base = rectangle_width →
  (1 / 2 : ℝ) * triangle_base * 54 = rectangle_length * rectangle_width :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_from_rectangle_l2172_217265


namespace NUMINAMATH_CALUDE_simplify_expression_l2172_217290

theorem simplify_expression (a : ℝ) : 3 * a^2 - a * (2 * a - 1) = a^2 + a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2172_217290


namespace NUMINAMATH_CALUDE_pattern_proof_l2172_217231

theorem pattern_proof (x : ℝ) (hx : x > 0) 
  (h1 : x + 1 / x ≥ 2)
  (h2 : x + 4 / x^2 ≥ 3)
  (h3 : x + 27 / x^3 ≥ 4)
  (h4 : ∃ a : ℝ, x + a / x^4 ≥ 5) :
  ∃ a : ℝ, x + a / x^4 ≥ 5 ∧ a = 256 := by
sorry

end NUMINAMATH_CALUDE_pattern_proof_l2172_217231


namespace NUMINAMATH_CALUDE_remainder_17_power_53_mod_5_l2172_217227

theorem remainder_17_power_53_mod_5 : 17^53 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_power_53_mod_5_l2172_217227


namespace NUMINAMATH_CALUDE_right_triangle_with_angle_ratio_l2172_217205

theorem right_triangle_with_angle_ratio (a b c : ℝ) (h_right : a + b + c = 180) 
  (h_largest : c = 90) (h_ratio : a / b = 3 / 2) : 
  c = 90 ∧ a = 54 ∧ b = 36 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_angle_ratio_l2172_217205


namespace NUMINAMATH_CALUDE_product_without_x3_x2_terms_l2172_217280

theorem product_without_x3_x2_terms (m n : ℝ) : 
  (∀ x : ℝ, (x^2 + m*x) * (x^2 - 2*x + n) = x^4 + m*n*x) → 
  m = 2 ∧ n = 4 := by
sorry

end NUMINAMATH_CALUDE_product_without_x3_x2_terms_l2172_217280


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l2172_217235

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2√(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  focal_length = 2 * Real.sqrt 10 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 7 - y^2 / 3 = 1) :=
by sorry

/-- The focal length of the hyperbola x²/7 - y²/3 = 1 is 2√10 -/
theorem specific_hyperbola_focal_length :
  let focal_length := 2 * Real.sqrt ((Real.sqrt 7)^2 + (Real.sqrt 3)^2)
  focal_length = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l2172_217235


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2172_217261

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence definition
  0 < a 1 → a 1 < a 2 →
  a 2 > Real.sqrt (a 1 * a 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2172_217261


namespace NUMINAMATH_CALUDE_secant_slope_positive_l2172_217230

open Real

noncomputable def f (x : ℝ) : ℝ := 2^x + x^3

theorem secant_slope_positive (x₁ x₂ : ℝ) (h : x₁ ≠ x₂) :
  (f x₁ - f x₂) / (x₁ - x₂) > 0 :=
sorry

end NUMINAMATH_CALUDE_secant_slope_positive_l2172_217230


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_proof_1008_smallest_l2172_217270

theorem smallest_four_digit_multiple_of_18 : ℕ → Prop :=
  fun n => (n ≥ 1000) ∧ (n < 10000) ∧ (n % 18 = 0) ∧
    ∀ m : ℕ, (m ≥ 1000) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m

theorem proof_1008_smallest : smallest_four_digit_multiple_of_18 1008 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_proof_1008_smallest_l2172_217270


namespace NUMINAMATH_CALUDE_exists_checkered_square_l2172_217233

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100 x 100 board -/
def Board := Fin 100 → Fin 100 → Color

/-- Checks if a cell is adjacent to the border -/
def is_border_adjacent (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square starting at (i, j) is monochrome -/
def is_monochrome (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Color,
    board i j = c ∧
    board i (j + 1) = c ∧
    board (i + 1) j = c ∧
    board (i + 1) (j + 1) = c

/-- Checks if a 2x2 square starting at (i, j) is checkered -/
def is_checkered (board : Board) (i j : Fin 100) : Prop :=
  (board i j = board (i + 1) (j + 1) ∧
   board i (j + 1) = board (i + 1) j ∧
   board i j ≠ board i (j + 1))

/-- Main theorem -/
theorem exists_checkered_square (board : Board) 
  (h1 : ∀ i j : Fin 100, is_border_adjacent i j → board i j = Color.Black)
  (h2 : ∀ i j : Fin 100, ¬is_monochrome board i j) :
  ∃ i j : Fin 100, is_checkered board i j :=
sorry

end NUMINAMATH_CALUDE_exists_checkered_square_l2172_217233


namespace NUMINAMATH_CALUDE_different_color_probability_l2172_217299

/-- The probability of drawing two balls of different colors from a bag containing 3 white balls and 2 black balls -/
theorem different_color_probability (total : Nat) (white : Nat) (black : Nat) 
  (h1 : total = 5) 
  (h2 : white = 3) 
  (h3 : black = 2) 
  (h4 : total = white + black) : 
  (white * black : ℚ) / (total.choose 2) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l2172_217299


namespace NUMINAMATH_CALUDE_radish_distribution_l2172_217232

theorem radish_distribution (total : ℕ) (groups : ℕ) (first_basket : ℕ) 
  (h1 : total = 88)
  (h2 : groups = 4)
  (h3 : first_basket = 37)
  (h4 : total % groups = 0) : 
  (total - first_basket) - first_basket = 14 := by
  sorry

end NUMINAMATH_CALUDE_radish_distribution_l2172_217232


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2172_217242

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2 * x + 2 = 0 ∧ k * y^2 - 2 * y + 2 = 0) ↔ 
  (k < 1/2 ∧ k ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2172_217242


namespace NUMINAMATH_CALUDE_otimes_twice_2h_l2172_217263

-- Define the operation ⊗
def otimes (x y : ℝ) : ℝ := x^3 - y

-- Theorem statement
theorem otimes_twice_2h (h : ℝ) : otimes (2*h) (otimes (2*h) (2*h)) = 2*h := by
  sorry

end NUMINAMATH_CALUDE_otimes_twice_2h_l2172_217263


namespace NUMINAMATH_CALUDE_driveways_shoveled_is_9_l2172_217256

/-- The number of driveways Jimmy shoveled -/
def driveways_shoveled : ℕ :=
  let candy_bar_price : ℚ := 75/100
  let candy_bar_discount : ℚ := 20/100
  let candy_bars_bought : ℕ := 2
  let lollipop_price : ℚ := 25/100
  let lollipops_bought : ℕ := 4
  let sales_tax : ℚ := 5/100
  let snow_shoveling_fraction : ℚ := 1/6
  let driveway_price : ℚ := 3/2

  let discounted_candy_price := candy_bar_price * (1 - candy_bar_discount)
  let total_candy_cost := (discounted_candy_price * candy_bars_bought)
  let total_lollipop_cost := (lollipop_price * lollipops_bought)
  let subtotal := total_candy_cost + total_lollipop_cost
  let total_with_tax := subtotal * (1 + sales_tax)
  let total_earned := total_with_tax / snow_shoveling_fraction
  let driveways := (total_earned / driveway_price).floor

  driveways.toNat

theorem driveways_shoveled_is_9 :
  driveways_shoveled = 9 := by sorry

end NUMINAMATH_CALUDE_driveways_shoveled_is_9_l2172_217256


namespace NUMINAMATH_CALUDE_hotel_room_charges_l2172_217223

theorem hotel_room_charges (G R P : ℝ) 
  (h1 : R = G * (1 + 0.60))
  (h2 : P = R * (1 - 0.50)) :
  P = G * (1 - 0.20) := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l2172_217223


namespace NUMINAMATH_CALUDE_cameron_typing_speed_l2172_217260

/-- The number of words Cameron could type per minute before breaking his arm -/
def words_per_minute : ℕ := 10

/-- The number of words Cameron could type per minute after breaking his arm -/
def words_after_injury : ℕ := 8

/-- The time period in minutes -/
def time_period : ℕ := 5

/-- The difference in words typed over the time period -/
def word_difference : ℕ := 10

theorem cameron_typing_speed :
  words_per_minute = 10 ∧
  words_after_injury = 8 ∧
  time_period = 5 ∧
  word_difference = 10 ∧
  time_period * words_per_minute - time_period * words_after_injury = word_difference :=
by sorry

end NUMINAMATH_CALUDE_cameron_typing_speed_l2172_217260


namespace NUMINAMATH_CALUDE_grocery_store_deal_cans_l2172_217217

theorem grocery_store_deal_cans (bulk_price : ℝ) (bulk_cans : ℕ) (store_price : ℝ) (price_difference : ℝ) : 
  bulk_price = 12 →
  bulk_cans = 48 →
  store_price = 6 →
  price_difference = 0.25 →
  (store_price / ((bulk_price / bulk_cans) + price_difference)) = 12 := by
sorry

end NUMINAMATH_CALUDE_grocery_store_deal_cans_l2172_217217


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l2172_217219

theorem parallelogram_perimeter (a b c d : ℕ) : 
  a^2 + b^2 = 130 →  -- sum of squares of diagonals
  c^2 + d^2 = 65 →   -- sum of squares of sides
  c + d = 11 →       -- sum of sides
  c * d = 28 →       -- product of sides
  2 * (c + d) = 22   -- perimeter
  := by sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l2172_217219


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2172_217254

theorem largest_prime_factor_of_1729 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 1729 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l2172_217254


namespace NUMINAMATH_CALUDE_f_of_five_equals_62_l2172_217248

/-- Given a function f(x) = 2x^2 + y where f(2) = 20, prove that f(5) = 62 -/
theorem f_of_five_equals_62 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : 
  f 5 = 62 := by
sorry

end NUMINAMATH_CALUDE_f_of_five_equals_62_l2172_217248


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l2172_217201

theorem arithmetic_sequence_squares (m : ℤ) : 
  (∃ (a d : ℝ), 
    (16 + m : ℝ) = (a : ℝ) ^ 2 ∧ 
    (100 + m : ℝ) = (a + d) ^ 2 ∧ 
    (484 + m : ℝ) = (a + 2 * d) ^ 2) ↔ 
  m = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l2172_217201


namespace NUMINAMATH_CALUDE_height_comparison_l2172_217215

theorem height_comparison (ashis_height babji_height : ℝ) 
  (h : ashis_height = babji_height * 1.25) : 
  (ashis_height - babji_height) / ashis_height = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l2172_217215


namespace NUMINAMATH_CALUDE_min_fourth_integer_l2172_217208

theorem min_fourth_integer (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  A = 3 * B →
  B = C - 2 →
  (A + B + C + D) / 4 = 16 →
  D ≥ 52 :=
by sorry

end NUMINAMATH_CALUDE_min_fourth_integer_l2172_217208


namespace NUMINAMATH_CALUDE_ajay_walking_distance_l2172_217229

/-- Ajay's walking problem -/
theorem ajay_walking_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 3) 
  (h2 : time = 16.666666666666668) : 
  speed * time = 50 := by
  sorry

end NUMINAMATH_CALUDE_ajay_walking_distance_l2172_217229


namespace NUMINAMATH_CALUDE_paiges_team_size_l2172_217228

theorem paiges_team_size (total_points : ℕ) (paige_points : ℕ) (others_points : ℕ) :
  total_points = 41 →
  paige_points = 11 →
  others_points = 6 →
  ∃ (team_size : ℕ), team_size = (total_points - paige_points) / others_points + 1 ∧ team_size = 6 :=
by sorry

end NUMINAMATH_CALUDE_paiges_team_size_l2172_217228


namespace NUMINAMATH_CALUDE_product_of_roots_equation_l2172_217209

theorem product_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 4) * (a - 2) + (a - 2) * (a - 6) = 0 ∧ 
               (b - 4) * (b - 2) + (b - 2) * (b - 6) = 0 ∧ 
               a ≠ b ∧ 
               a * b = 10) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_equation_l2172_217209


namespace NUMINAMATH_CALUDE_no_double_apply_add_2015_l2172_217296

theorem no_double_apply_add_2015 : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := by
  sorry

end NUMINAMATH_CALUDE_no_double_apply_add_2015_l2172_217296


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_expansion_l2172_217252

theorem binomial_coefficient_third_term_expansion (x : ℤ) :
  Nat.choose 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_expansion_l2172_217252


namespace NUMINAMATH_CALUDE_specific_trapezoid_diagonals_l2172_217282

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  midline : ℝ
  height : ℝ
  diagonal_angle : ℝ

/-- The diagonals of a trapezoid -/
def trapezoid_diagonals (t : Trapezoid) : ℝ × ℝ := sorry

/-- Theorem stating the diagonals of a specific trapezoid -/
theorem specific_trapezoid_diagonals :
  let t : Trapezoid := {
    midline := 7,
    height := 15 * Real.sqrt 3 / 7,
    diagonal_angle := 2 * π / 3  -- 120° in radians
  }
  trapezoid_diagonals t = (6, 10) := by sorry

end NUMINAMATH_CALUDE_specific_trapezoid_diagonals_l2172_217282


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2172_217216

def numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

theorem arithmetic_mean_problem (x : ℕ) :
  (numbers.sum + x) / (numbers.length + 1) = 12 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2172_217216


namespace NUMINAMATH_CALUDE_polynomial_rearrangement_l2172_217244

theorem polynomial_rearrangement (x : ℝ) : 
  x^4 + 2*x^3 - 3*x^2 - 4*x + 1 = (x+1)^4 - 2*(x+1)^3 - 3*(x+1)^2 + 4*(x+1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_rearrangement_l2172_217244


namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l2172_217200

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l2172_217200


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2172_217210

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2172_217210


namespace NUMINAMATH_CALUDE_square_of_binomial_l2172_217212

theorem square_of_binomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 14*x + k = (x - a)^2) ↔ k = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2172_217212


namespace NUMINAMATH_CALUDE_shepherds_sheep_count_l2172_217259

theorem shepherds_sheep_count :
  ∀ a b : ℕ,
  (∃ n : ℕ, a = n * n) →  -- a is a perfect square
  (∃ m : ℕ, b = m * m) →  -- b is a perfect square
  97 ≤ a + b →            -- lower bound of total sheep
  a + b ≤ 108 →           -- upper bound of total sheep
  a > b →                 -- Noémie has more sheep than Tristan
  a ≥ 4 →                 -- Each shepherd has at least 2 sheep (2² = 4)
  b ≥ 4 →                 -- Each shepherd has at least 2 sheep (2² = 4)
  Odd (a + b) →           -- Total number of sheep is odd
  a = 81 ∧ b = 16 :=      -- Conclusion: Noémie has 81 sheep, Tristan has 16 sheep
by sorry

end NUMINAMATH_CALUDE_shepherds_sheep_count_l2172_217259


namespace NUMINAMATH_CALUDE_no_perfect_square_solution_l2172_217255

theorem no_perfect_square_solution :
  ¬ ∃ (x y z t : ℕ+), 
    (x * y - z * t = x + y) ∧
    (x + y = z + t) ∧
    (∃ (a b : ℕ+), x * y = a * a ∧ z * t = b * b) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_solution_l2172_217255
