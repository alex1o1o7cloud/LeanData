import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2242_224223

theorem complex_modulus_problem (x y : ℝ) :
  (Complex.I * (x + 2 * Complex.I) = y - Complex.I) →
  Complex.abs (x - y * Complex.I) = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2242_224223


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_progression_l2242_224278

theorem arithmetic_to_geometric_progression 
  (x y z : ℝ) 
  (h1 : y^2 - x*y = z^2 - y^2) : 
  z^2 = y * (2*y - x) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_progression_l2242_224278


namespace NUMINAMATH_CALUDE_necklace_price_l2242_224211

def total_cost : ℕ := 240000
def necklace_count : ℕ := 3

theorem necklace_price (necklace_price : ℕ) 
  (h1 : necklace_count * necklace_price + 3 * necklace_price = total_cost) :
  necklace_price = 40000 := by
  sorry

end NUMINAMATH_CALUDE_necklace_price_l2242_224211


namespace NUMINAMATH_CALUDE_rectangle_to_square_l2242_224279

/-- A rectangle can be cut into three parts to form a square --/
theorem rectangle_to_square :
  ∃ (a b c : ℕ × ℕ),
    -- The original rectangle is 25 × 4
    25 * 4 = (a.1 * a.2) + (b.1 * b.2) + (c.1 * c.2) ∧
    -- The three parts can form a square
    ∃ (s : ℕ), s * s = (a.1 * a.2) + (b.1 * b.2) + (c.1 * c.2) ∧
    -- There are exactly three parts
    a ≠ b ∧ b ≠ c ∧ a ≠ c :=
by sorry


end NUMINAMATH_CALUDE_rectangle_to_square_l2242_224279


namespace NUMINAMATH_CALUDE_price_A_base_correct_minimum_amount_spent_l2242_224203

-- Define the price of type A seedlings at the base
def price_A_base : ℝ := 20

-- Define the price of type B seedlings at the base
def price_B_base : ℝ := 30

-- Define the total number of bundles to purchase
def total_bundles : ℕ := 100

-- Define the discount rate
def discount_rate : ℝ := 0.9

-- Theorem for part 1
theorem price_A_base_correct :
  price_A_base * (300 / price_A_base) = 
  (5/4 * price_A_base) * (300 / (5/4 * price_A_base) + 3) := by sorry

-- Theorem for part 2
theorem minimum_amount_spent :
  let m := min (total_bundles / 2) total_bundles
  ∃ (n : ℕ), n ≤ total_bundles - n ∧
    discount_rate * (price_A_base * m + price_B_base * (total_bundles - m)) = 2250 := by sorry

end NUMINAMATH_CALUDE_price_A_base_correct_minimum_amount_spent_l2242_224203


namespace NUMINAMATH_CALUDE_negative_represents_spending_l2242_224202

/-- Represents a monetary transaction -/
inductive Transaction
| receive (amount : ℤ)
| spend (amount : ℤ)

/-- Converts a transaction to an integer representation -/
def transactionToInt : Transaction → ℤ
| Transaction.receive amount => amount
| Transaction.spend amount => -amount

theorem negative_represents_spending (t : Transaction) : 
  (∃ (a : ℤ), a > 0 ∧ transactionToInt (Transaction.receive a) = a) →
  (∀ (b : ℤ), b > 0 → transactionToInt (Transaction.spend b) = -b) :=
by sorry

end NUMINAMATH_CALUDE_negative_represents_spending_l2242_224202


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_count_l2242_224257

theorem systematic_sampling_interval_count 
  (total_employees : ℕ) 
  (sample_size : ℕ) 
  (interval_start : ℕ) 
  (interval_end : ℕ) 
  (h1 : total_employees = 840)
  (h2 : sample_size = 42)
  (h3 : interval_start = 481)
  (h4 : interval_end = 720) :
  (interval_end - interval_start + 1) / (total_employees / sample_size) = 12 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_count_l2242_224257


namespace NUMINAMATH_CALUDE_xy_yz_zx_over_x2_y2_z2_l2242_224215

theorem xy_yz_zx_over_x2_y2_z2 (x y z a b c : ℝ) 
  (h_distinct_xyz : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_distinct_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_nonzero_abc : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a * x + b * y + c * z = 0) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = -1 :=
by sorry

end NUMINAMATH_CALUDE_xy_yz_zx_over_x2_y2_z2_l2242_224215


namespace NUMINAMATH_CALUDE_min_degree_of_specific_polynomial_l2242_224213

/-- A polynomial function from ℝ to ℝ -/
def PolynomialFunction := ℝ → ℝ

/-- The degree of a polynomial function -/
def degree (f : PolynomialFunction) : ℕ := sorry

theorem min_degree_of_specific_polynomial (f : PolynomialFunction)
  (h1 : f (-2) = 3)
  (h2 : f (-1) = -3)
  (h3 : f 1 = -3)
  (h4 : f 2 = 6)
  (h5 : f 3 = 5) :
  degree f = 4 ∧ ∀ g : PolynomialFunction, 
    g (-2) = 3 → g (-1) = -3 → g 1 = -3 → g 2 = 6 → g 3 = 5 → 
    degree g ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_degree_of_specific_polynomial_l2242_224213


namespace NUMINAMATH_CALUDE_train_speed_l2242_224289

/-- Proves that a train with given parameters has a speed of 45 km/hr -/
theorem train_speed (train_length : Real) (crossing_time : Real) (total_length : Real) :
  train_length = 100 →
  crossing_time = 30 →
  total_length = 275 →
  (total_length - train_length) / crossing_time * 3.6 = 45 :=
by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2242_224289


namespace NUMINAMATH_CALUDE_solve_for_y_l2242_224298

theorem solve_for_y (x y : ℚ) (h1 : x = 103) (h2 : x^3 * y - 2 * x^2 * y + x * y = 103030) : y = 10 / 103 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2242_224298


namespace NUMINAMATH_CALUDE_rest_stop_distance_l2242_224224

/-- Proves that the distance between rest stops is 10 miles for a man walking 50 miles in 320 minutes with given conditions. -/
theorem rest_stop_distance (walking_speed : ℝ) (rest_duration : ℝ) (total_distance : ℝ) (total_time : ℝ) 
  (h1 : walking_speed = 10) -- walking speed in mph
  (h2 : rest_duration = 5 / 60) -- rest duration in hours
  (h3 : total_distance = 50) -- total distance in miles
  (h4 : total_time = 320 / 60) -- total time in hours
  : ∃ (x : ℝ), x = 10 ∧ 
    (total_distance / walking_speed + rest_duration * (total_distance / x - 1) = total_time) := by
  sorry


end NUMINAMATH_CALUDE_rest_stop_distance_l2242_224224


namespace NUMINAMATH_CALUDE_original_ratio_l2242_224268

theorem original_ratio (x y : ℕ) (h1 : y = 48) (h2 : (x + 12) / y = 1/2) : x / y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_original_ratio_l2242_224268


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_45_l2242_224206

/-- Given a natural number n, returns the sum of n consecutive positive integers starting from a -/
def consecutive_sum (n : ℕ) (a : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Predicate that checks if there exists a starting integer a such that n consecutive integers starting from a sum to 45 -/
def exists_consecutive_sum (n : ℕ) : Prop :=
  ∃ a : ℕ, a > 0 ∧ consecutive_sum n a = 45

theorem max_consecutive_integers_sum_45 :
  (∀ k : ℕ, k > 9 → ¬ exists_consecutive_sum k) ∧
  exists_consecutive_sum 9 :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_45_l2242_224206


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l2242_224285

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 3

theorem f_monotone_decreasing :
  ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → f x1 > f x2 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l2242_224285


namespace NUMINAMATH_CALUDE_coefficient_of_y_l2242_224271

theorem coefficient_of_y (x y a : ℝ) : 
  5 * x + y = 19 →
  x + a * y = 1 →
  3 * x + 2 * y = 10 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l2242_224271


namespace NUMINAMATH_CALUDE_sum_x_y_l2242_224250

theorem sum_x_y (x y : ℝ) 
  (h1 : |x| + x + y - 2 = 14)
  (h2 : x + |y| - y + 3 = 20) : 
  x + y = 31/5 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_l2242_224250


namespace NUMINAMATH_CALUDE_inequality_proof_l2242_224230

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2242_224230


namespace NUMINAMATH_CALUDE_door_opening_probability_l2242_224234

/-- Represents the probability of opening a door on the second attempt -/
def probability_second_attempt (total_keys : ℕ) (working_keys : ℕ) (discard : Bool) : ℚ :=
  if discard then
    (working_keys : ℚ) / total_keys * working_keys / (total_keys - 1)
  else
    (working_keys : ℚ) / total_keys * working_keys / total_keys

/-- The main theorem about the probability of opening the door on the second attempt -/
theorem door_opening_probability :
  let total_keys : ℕ := 4
  let working_keys : ℕ := 2
  probability_second_attempt total_keys working_keys true = 1/3 ∧
  probability_second_attempt total_keys working_keys false = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_door_opening_probability_l2242_224234


namespace NUMINAMATH_CALUDE_equal_x_y_l2242_224245

-- Define the geometric configuration
structure GeometricConfiguration where
  a₁ : ℝ
  a₂ : ℝ
  b₁ : ℝ
  b₂ : ℝ
  x : ℝ
  y : ℝ

-- Define the theorem
theorem equal_x_y (config : GeometricConfiguration) 
  (h1 : config.a₁ = config.a₂) 
  (h2 : config.b₁ = config.b₂) : 
  config.x = config.y := by
  sorry


end NUMINAMATH_CALUDE_equal_x_y_l2242_224245


namespace NUMINAMATH_CALUDE_symmetric_points_count_l2242_224226

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^2 + 4 * x + 1 else 2 / Real.exp x

theorem symmetric_points_count :
  ∃! (p : ℕ), p = 2 ∧
  ∃ (S : Finset (ℝ × ℝ)),
    S.card = p ∧
    (∀ (x y : ℝ), (x, y) ∈ S → y = f x) ∧
    (∀ (x y : ℝ), (x, y) ∈ S → (-x, -y) ∈ S) ∧
    (∀ (x y : ℝ), (x, y) ∈ S → (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_count_l2242_224226


namespace NUMINAMATH_CALUDE_find_d_l2242_224240

theorem find_d (A B C D : ℝ) : 
  (A + B + C) / 3 = 130 →
  (A + B + C + D) / 4 = 126 →
  D = 114 := by
sorry

end NUMINAMATH_CALUDE_find_d_l2242_224240


namespace NUMINAMATH_CALUDE_green_beads_count_l2242_224227

/-- The number of green beads initially in a container -/
def initial_green_beads (total : ℕ) (brown red taken left : ℕ) : ℕ :=
  total - brown - red

/-- Theorem stating the number of green beads initially in the container -/
theorem green_beads_count (brown red taken left : ℕ) 
  (h1 : brown = 2)
  (h2 : red = 3)
  (h3 : taken = 2)
  (h4 : left = 4) :
  initial_green_beads (taken + left) brown red taken left = 1 := by
  sorry

#check green_beads_count

end NUMINAMATH_CALUDE_green_beads_count_l2242_224227


namespace NUMINAMATH_CALUDE_cost_type_B_calculation_l2242_224296

/-- The cost of purchasing type B books given the total number of books and the number of type A books purchased. -/
def cost_type_B (total_books : ℕ) (price_A : ℕ) (price_B : ℕ) (x : ℕ) : ℕ :=
  price_B * (total_books - x)

/-- Theorem stating that the cost of purchasing type B books is 8(100-x) yuan -/
theorem cost_type_B_calculation (x : ℕ) (h : x ≤ 100) :
  cost_type_B 100 10 8 x = 8 * (100 - x) := by
  sorry

end NUMINAMATH_CALUDE_cost_type_B_calculation_l2242_224296


namespace NUMINAMATH_CALUDE_equation_solution_l2242_224205

theorem equation_solution : ∃ x : ℝ, 24 - (4 * 2) = 5 + x ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2242_224205


namespace NUMINAMATH_CALUDE_inequality_proof_l2242_224286

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄) (h4 : x₄ ≥ 2) 
  (h5 : x₂ + x₃ + x₄ ≥ x₁) : 
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2242_224286


namespace NUMINAMATH_CALUDE_log_56342_between_consecutive_integers_l2242_224258

theorem log_56342_between_consecutive_integers :
  ∃ (c d : ℕ), c + 1 = d ∧ (c : ℝ) < Real.log 56342 / Real.log 10 ∧ Real.log 56342 / Real.log 10 < d ∧ c + d = 9 :=
by
  -- Assuming 10000 < 56342 < 100000
  have h1 : 10000 < 56342 := by sorry
  have h2 : 56342 < 100000 := by sorry
  sorry

end NUMINAMATH_CALUDE_log_56342_between_consecutive_integers_l2242_224258


namespace NUMINAMATH_CALUDE_city_male_population_l2242_224264

theorem city_male_population (total_population : ℕ) (num_parts : ℕ) (male_parts : ℕ) :
  total_population = 1000 →
  num_parts = 5 →
  male_parts = 2 →
  (total_population / num_parts) * male_parts = 400 := by
sorry

end NUMINAMATH_CALUDE_city_male_population_l2242_224264


namespace NUMINAMATH_CALUDE_seven_point_circle_triangles_l2242_224242

/-- The number of triangles formed by intersections of chords in a circle -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose (Nat.choose n 4) 3

/-- Theorem: Given 7 points on a circle with the specified conditions, 
    the number of triangles formed is 6545 -/
theorem seven_point_circle_triangles : num_triangles 7 = 6545 := by
  sorry

end NUMINAMATH_CALUDE_seven_point_circle_triangles_l2242_224242


namespace NUMINAMATH_CALUDE_derivative_implies_antiderivative_l2242_224208

theorem derivative_implies_antiderivative (f : ℝ → ℝ) :
  (∀ x, deriv f x = 6 * x^2 + 5) →
  ∃ c, ∀ x, f x = 2 * x^3 + 5 * x + c :=
sorry

end NUMINAMATH_CALUDE_derivative_implies_antiderivative_l2242_224208


namespace NUMINAMATH_CALUDE_intersection_line_slope_l2242_224267

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_line_slope (x y : ℝ) : 
  (x^2 + y^2 + 6*x - 8*y - 40 = 0) ∧ 
  (x^2 + y^2 + 22*x - 2*y + 20 = 0) →
  (∃ m : ℝ, m = 8/3 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁^2 + y₁^2 + 6*x₁ - 8*y₁ - 40 = 0) ∧ 
      (x₁^2 + y₁^2 + 22*x₁ - 2*y₁ + 20 = 0) ∧
      (x₂^2 + y₂^2 + 6*x₂ - 8*y₂ - 40 = 0) ∧ 
      (x₂^2 + y₂^2 + 22*x₂ - 2*y₂ + 20 = 0) ∧
      (x₁ ≠ x₂) →
      m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l2242_224267


namespace NUMINAMATH_CALUDE_circular_bead_arrangements_l2242_224221

/-- The number of red beads -/
def num_red : ℕ := 3

/-- The number of blue beads -/
def num_blue : ℕ := 2

/-- The total number of beads -/
def total_beads : ℕ := num_red + num_blue

/-- The symmetry group of the circular arrangement -/
def symmetry_group : ℕ := 2 * total_beads

/-- The number of fixed arrangements under the identity rotation -/
def fixed_identity : ℕ := (total_beads.choose num_red)

/-- The number of fixed arrangements under each reflection -/
def fixed_reflection : ℕ := 2

/-- The number of reflections in the symmetry group -/
def num_reflections : ℕ := total_beads

/-- The total number of fixed arrangements under all symmetries -/
def total_fixed : ℕ := fixed_identity + num_reflections * fixed_reflection

/-- The number of distinct arrangements of beads on the circular ring -/
def distinct_arrangements : ℕ := total_fixed / symmetry_group

theorem circular_bead_arrangements :
  distinct_arrangements = 2 :=
sorry

end NUMINAMATH_CALUDE_circular_bead_arrangements_l2242_224221


namespace NUMINAMATH_CALUDE_chip_drawing_probability_l2242_224262

theorem chip_drawing_probability : 
  let total_chips : ℕ := 14
  let tan_chips : ℕ := 5
  let pink_chips : ℕ := 3
  let violet_chips : ℕ := 6
  let favorable_outcomes : ℕ := (Nat.factorial pink_chips) * (Nat.factorial tan_chips) * (Nat.factorial violet_chips)
  let total_outcomes : ℕ := Nat.factorial total_chips
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 168168 := by
sorry

end NUMINAMATH_CALUDE_chip_drawing_probability_l2242_224262


namespace NUMINAMATH_CALUDE_factorization_proof_l2242_224243

theorem factorization_proof (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2242_224243


namespace NUMINAMATH_CALUDE_divisibility_problem_l2242_224254

theorem divisibility_problem (n m k : ℕ) (h1 : n = 859722) (h2 : m = 456) (h3 : k = 54) :
  (n + k) % m = 0 :=
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2242_224254


namespace NUMINAMATH_CALUDE_strawberry_picking_total_weight_l2242_224266

theorem strawberry_picking_total_weight 
  (marco_weight : ℕ) 
  (dad_weight : ℕ) 
  (h1 : marco_weight = 8) 
  (h2 : dad_weight = 32) : 
  marco_weight + dad_weight = 40 := by
sorry

end NUMINAMATH_CALUDE_strawberry_picking_total_weight_l2242_224266


namespace NUMINAMATH_CALUDE_four_distinct_roots_l2242_224244

/-- The equation x^2 - 4|x| + 5 = m has four distinct real roots if and only if 1 < m < 5 -/
theorem four_distinct_roots (m : ℝ) :
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^2 - 4 * |x| + 5 = m ↔ x = a ∨ x = b ∨ x = c ∨ x = d)) ↔
  1 < m ∧ m < 5 := by
  sorry

end NUMINAMATH_CALUDE_four_distinct_roots_l2242_224244


namespace NUMINAMATH_CALUDE_multiples_of_seven_between_50_and_150_l2242_224241

theorem multiples_of_seven_between_50_and_150 :
  (Finset.filter (fun n => 50 ≤ 7 * n ∧ 7 * n ≤ 150) (Finset.range (150 / 7 + 1))).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_seven_between_50_and_150_l2242_224241


namespace NUMINAMATH_CALUDE_common_point_and_tangent_l2242_224219

theorem common_point_and_tangent (t : ℝ) (h : t ≠ 0) :
  let f := fun x : ℝ => x^3 + a*x
  let g := fun x : ℝ => b*x^2 + c
  let f' := fun x : ℝ => 3*x^2 + a
  let g' := fun x : ℝ => 2*b*x
  f t = 0 ∧ g t = 0 ∧ f' t = g' t →
  a = -t^2 ∧ b = t ∧ c = -t^3 :=
by sorry

end NUMINAMATH_CALUDE_common_point_and_tangent_l2242_224219


namespace NUMINAMATH_CALUDE_collatz_7_11_collatz_10_probability_l2242_224265

-- Define the Collatz operation
def collatz (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

-- Define the Collatz sequence
def collatz_seq (a₀ : ℕ) : ℕ → ℕ
  | 0 => a₀
  | n + 1 => collatz (collatz_seq a₀ n)

-- Statement 1: When a₀ = 7, a₁₁ = 5
theorem collatz_7_11 : collatz_seq 7 11 = 5 := by sorry

-- Helper function to check if a number is odd
def is_odd (n : ℕ) : Bool := n % 2 ≠ 0

-- Statement 2: When a₀ = 10, the probability of randomly selecting two numbers
-- from aᵢ (i = 1,2,3,4,5,6), at least one of which is odd, is 3/5
theorem collatz_10_probability :
  let seq := List.range 6 |> List.map (collatz_seq 10)
  let total_pairs := seq.length.choose 2
  let odd_pairs := (seq.filterMap (fun n => if is_odd n then some n else none)).length
  (total_pairs - (seq.length - odd_pairs).choose 2) / total_pairs = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_collatz_7_11_collatz_10_probability_l2242_224265


namespace NUMINAMATH_CALUDE_division_result_l2242_224210

theorem division_result (h : 144 * 177 = 25488) : 254.88 / 0.177 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l2242_224210


namespace NUMINAMATH_CALUDE_boys_less_than_four_sevenths_l2242_224247

/-- Represents a class of students with two hiking trips -/
structure HikingClass where
  boys : ℕ
  girls : ℕ
  boys_trip1 : ℕ
  girls_trip1 : ℕ
  boys_trip2 : ℕ
  girls_trip2 : ℕ

/-- The conditions of the hiking trips -/
def validHikingClass (c : HikingClass) : Prop :=
  c.boys_trip1 < (2 * (c.boys_trip1 + c.girls_trip1)) / 5 ∧
  c.boys_trip2 < (2 * (c.boys_trip2 + c.girls_trip2)) / 5 ∧
  c.boys_trip1 + c.boys_trip2 ≥ c.boys ∧
  c.girls_trip1 ≤ c.girls ∧
  c.girls_trip2 ≤ c.girls

/-- The main theorem to prove -/
theorem boys_less_than_four_sevenths (c : HikingClass) 
  (h : validHikingClass c) : 
  c.boys < (4 * (c.boys + c.girls)) / 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_less_than_four_sevenths_l2242_224247


namespace NUMINAMATH_CALUDE_highway_length_is_105_l2242_224218

/-- The length of a highway where two cars meet after traveling from opposite ends -/
def highway_length (speed1 speed2 time : ℝ) : ℝ :=
  speed1 * time + speed2 * time

/-- Theorem: The highway length is 105 miles given the specific conditions -/
theorem highway_length_is_105 :
  highway_length 15 20 3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_is_105_l2242_224218


namespace NUMINAMATH_CALUDE_larger_number_proof_l2242_224287

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 1365) 
  (h2 : y = 6 * x + 15) : 
  y = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2242_224287


namespace NUMINAMATH_CALUDE_simplify_fraction_l2242_224260

theorem simplify_fraction : (84 : ℚ) / 1764 * 21 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2242_224260


namespace NUMINAMATH_CALUDE_triangle_area_l2242_224212

theorem triangle_area (A B C : Real) (a b c : Real) :
  A = π / 3 →
  a = Real.sqrt 3 →
  c = 1 →
  (∃ S : Real, S = (Real.sqrt 3) / 2 ∧ S = (1 / 2) * a * c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2242_224212


namespace NUMINAMATH_CALUDE_complex_modulus_l2242_224253

theorem complex_modulus (a b : ℝ) (h : (1 + 2*a*Complex.I) * Complex.I = 1 - b*Complex.I) : 
  Complex.abs (a + b*Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2242_224253


namespace NUMINAMATH_CALUDE_statutory_capital_scientific_notation_l2242_224270

/-- The statutory capital of the Asian Infrastructure Investment Bank in U.S. dollars -/
def statutory_capital : ℝ := 100000000000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Theorem stating that the statutory capital in scientific notation is 1 × 10^11 -/
theorem statutory_capital_scientific_notation :
  ∃ (sn : ScientificNotation),
    sn.coefficient = 1 ∧
    sn.exponent = 11 ∧
    statutory_capital = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
by sorry

end NUMINAMATH_CALUDE_statutory_capital_scientific_notation_l2242_224270


namespace NUMINAMATH_CALUDE_optimal_newspaper_sales_l2242_224282

/-- Represents the daily newspaper sales data --/
structure NewspaperSalesData where
  costPrice : ℝ
  sellingPrice : ℝ
  returnPrice : ℝ
  highSalesDays : ℕ
  highSalesAmount : ℕ
  lowSalesDays : ℕ
  lowSalesAmount : ℕ

/-- Calculates the monthly profit based on the number of copies purchased daily --/
def monthlyProfit (data : NewspaperSalesData) (dailyPurchase : ℕ) : ℝ :=
  let soldProfit := data.sellingPrice - data.costPrice
  let returnLoss := data.costPrice - data.returnPrice
  let totalSold := data.highSalesDays * (min dailyPurchase data.highSalesAmount) +
                   data.lowSalesDays * (min dailyPurchase data.lowSalesAmount)
  let totalReturned := (data.highSalesDays + data.lowSalesDays) * dailyPurchase - totalSold
  soldProfit * totalSold - returnLoss * totalReturned

/-- Theorem stating the optimal daily purchase and maximum monthly profit --/
theorem optimal_newspaper_sales (data : NewspaperSalesData)
  (h1 : data.costPrice = 0.12)
  (h2 : data.sellingPrice = 0.20)
  (h3 : data.returnPrice = 0.04)
  (h4 : data.highSalesDays = 20)
  (h5 : data.highSalesAmount = 400)
  (h6 : data.lowSalesDays = 10)
  (h7 : data.lowSalesAmount = 250) :
  (∀ x : ℕ, monthlyProfit data x ≤ monthlyProfit data 400) ∧
  monthlyProfit data 400 = 840 := by
  sorry


end NUMINAMATH_CALUDE_optimal_newspaper_sales_l2242_224282


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2242_224246

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 32 →
  a 4 + a 5 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2242_224246


namespace NUMINAMATH_CALUDE_equation_implies_fraction_value_l2242_224293

theorem equation_implies_fraction_value (a b : ℝ) :
  a^2 + b^2 - 4*a - 2*b + 5 = 0 →
  (Real.sqrt a + b) / (2 * Real.sqrt a + b + 1) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equation_implies_fraction_value_l2242_224293


namespace NUMINAMATH_CALUDE_product_equals_243_l2242_224276

theorem product_equals_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l2242_224276


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2242_224277

def is_divisible_by_all (n : ℕ) : Prop :=
  (n + 3) % 9 = 0 ∧ (n + 3) % 70 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0

theorem smallest_number_divisible_by_all : 
  ∀ n : ℕ, n < 3147 → ¬(is_divisible_by_all n) ∧ is_divisible_by_all 3147 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2242_224277


namespace NUMINAMATH_CALUDE_simplify_and_sum_l2242_224216

theorem simplify_and_sum (d : ℝ) (a b c : ℝ) (h : d ≠ 0) :
  (15 * d + 18 + 12 * d^2) + (5 * d + 2) = a * d + b + c * d^2 →
  a + b + c = 52 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_l2242_224216


namespace NUMINAMATH_CALUDE_min_typical_parallelepipeds_is_four_l2242_224261

/-- A typical parallelepiped has all dimensions different -/
structure TypicalParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  all_different : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- A cube with side length s -/
structure Cube where
  side : ℝ

/-- The minimum number of typical parallelepipeds into which a cube can be cut -/
def min_typical_parallelepipeds_in_cube (c : Cube) : ℕ :=
  4

/-- Theorem stating that the minimum number of typical parallelepipeds 
    into which a cube can be cut is 4 -/
theorem min_typical_parallelepipeds_is_four (c : Cube) :
  min_typical_parallelepipeds_in_cube c = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_typical_parallelepipeds_is_four_l2242_224261


namespace NUMINAMATH_CALUDE_water_in_altered_solution_l2242_224283

/-- Represents the ratios of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the new ratio after altering the original ratio -/
def alter_ratio (original : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * original.bleach,
    detergent := original.detergent,
    water := 2 * original.water }

/-- Theorem: Given the conditions, the altered solution contains 150 liters of water -/
theorem water_in_altered_solution :
  let original_ratio : SolutionRatio := ⟨2, 40, 100⟩
  let altered_ratio := alter_ratio original_ratio
  let detergent_volume : ℚ := 60
  (detergent_volume * altered_ratio.water) / altered_ratio.detergent = 150 := by
  sorry

end NUMINAMATH_CALUDE_water_in_altered_solution_l2242_224283


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2242_224269

theorem polynomial_division_remainder : ∀ (z : ℝ),
  ∃ (r : ℝ),
    3 * z^3 - 4 * z^2 - 14 * z + 3 = (3 * z + 5) * (z^2 - 3 * z + 1/3) + r ∧
    r = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2242_224269


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l2242_224272

/-- The minimum distance a bug must crawl on the surface of a right circular cone --/
theorem bug_crawl_distance (r h a b θ : ℝ) (hr : r = 500) (hh : h = 300) 
  (ha : a = 100) (hb : b = 400) (hθ : θ = π / 2) : 
  let d := Real.sqrt ((b * Real.cos θ - a)^2 + (b * Real.sin θ)^2)
  d = Real.sqrt 170000 := by
sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l2242_224272


namespace NUMINAMATH_CALUDE_janes_calculation_l2242_224214

theorem janes_calculation (a b c : ℝ) 
  (h1 : a + b + c = 11) 
  (h2 : a + b - c = 19) : 
  a + b = 15 := by
sorry

end NUMINAMATH_CALUDE_janes_calculation_l2242_224214


namespace NUMINAMATH_CALUDE_complex_square_roots_l2242_224229

theorem complex_square_roots (z : ℂ) : 
  z^2 = -45 - 28*I ↔ z = 2 - 7*I ∨ z = -2 + 7*I :=
sorry

end NUMINAMATH_CALUDE_complex_square_roots_l2242_224229


namespace NUMINAMATH_CALUDE_multiple_solutions_exist_four_wheelers_not_unique_l2242_224284

/-- Represents the number of wheels on a vehicle -/
inductive WheelCount
  | two
  | four

/-- Represents the parking lot with 2 wheelers and 4 wheelers -/
structure ParkingLot where
  twoWheelers : ℕ
  fourWheelers : ℕ

/-- Calculates the total number of wheels in the parking lot -/
def totalWheels (lot : ParkingLot) : ℕ :=
  2 * lot.twoWheelers + 4 * lot.fourWheelers

/-- Theorem stating that multiple solutions exist for a given total wheel count -/
theorem multiple_solutions_exist (totalWheelCount : ℕ) :
  ∃ (lot1 lot2 : ParkingLot), lot1 ≠ lot2 ∧ totalWheels lot1 = totalWheelCount ∧ totalWheels lot2 = totalWheelCount :=
sorry

/-- Theorem stating that the number of 4 wheelers cannot be uniquely determined -/
theorem four_wheelers_not_unique (totalWheelCount : ℕ) :
  ¬∃! (fourWheelerCount : ℕ), ∃ (twoWheelerCount : ℕ), totalWheels {twoWheelers := twoWheelerCount, fourWheelers := fourWheelerCount} = totalWheelCount :=
sorry

end NUMINAMATH_CALUDE_multiple_solutions_exist_four_wheelers_not_unique_l2242_224284


namespace NUMINAMATH_CALUDE_min_sum_distances_l2242_224225

/-- The minimum sum of distances from a point on the x-axis to two fixed points -/
theorem min_sum_distances (P : ℝ × ℝ) (A B : ℝ × ℝ) : 
  A = (1, 1) → B = (3, 4) → P.2 = 0 → 
  ∀ Q : ℝ × ℝ, Q.2 = 0 → Real.sqrt 29 ≤ dist P A + dist P B :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l2242_224225


namespace NUMINAMATH_CALUDE_harveys_steak_sales_l2242_224217

/-- Calculates the total number of steaks sold given the initial count, 
    the count after the first sale, and the count of the second sale. -/
def total_steaks_sold (initial : Nat) (after_first_sale : Nat) (second_sale : Nat) : Nat :=
  (initial - after_first_sale) + second_sale

/-- Theorem stating that given Harvey's specific situation, 
    the total number of steaks sold is 17. -/
theorem harveys_steak_sales : 
  total_steaks_sold 25 12 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_harveys_steak_sales_l2242_224217


namespace NUMINAMATH_CALUDE_total_profit_is_27_l2242_224252

/-- Given the following conditions:
  1. Natasha has 3 times as much money as Carla
  2. Carla has twice as much money as Cosima
  3. Natasha has $60
  4. Sergio has 1.5 times as much money as Cosima
  5. Natasha buys 4 items at $15 each
  6. Carla buys 6 items at $10 each
  7. Cosima buys 5 items at $8 each
  8. Sergio buys 3 items at $12 each
  9. Profit margins: Natasha 10%, Carla 15%, Cosima 12%, Sergio 20%

  Prove that the total profit after selling all goods is $27. -/
theorem total_profit_is_27 (natasha_money carla_money cosima_money sergio_money : ℚ)
  (natasha_items carla_items cosima_items sergio_items : ℕ)
  (natasha_price carla_price cosima_price sergio_price : ℚ)
  (natasha_margin carla_margin cosima_margin sergio_margin : ℚ) :
  natasha_money = 60 ∧
  natasha_money = 3 * carla_money ∧
  carla_money = 2 * cosima_money ∧
  sergio_money = 1.5 * cosima_money ∧
  natasha_items = 4 ∧
  carla_items = 6 ∧
  cosima_items = 5 ∧
  sergio_items = 3 ∧
  natasha_price = 15 ∧
  carla_price = 10 ∧
  cosima_price = 8 ∧
  sergio_price = 12 ∧
  natasha_margin = 0.1 ∧
  carla_margin = 0.15 ∧
  cosima_margin = 0.12 ∧
  sergio_margin = 0.2 →
  natasha_items * natasha_price * natasha_margin +
  carla_items * carla_price * carla_margin +
  cosima_items * cosima_price * cosima_margin +
  sergio_items * sergio_price * sergio_margin = 27 := by
  sorry


end NUMINAMATH_CALUDE_total_profit_is_27_l2242_224252


namespace NUMINAMATH_CALUDE_turtle_fraction_l2242_224251

theorem turtle_fraction (trey kris kristen : ℕ) : 
  trey = 7 * kris →
  trey = kristen + 9 →
  kristen = 12 →
  kris / kristen = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_turtle_fraction_l2242_224251


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2242_224255

theorem polynomial_simplification (x : ℝ) :
  (15 * x^10 + 10 * x^9 + 5 * x^8) + (3 * x^12 + 2 * x^10 + x^9 + 3 * x^7 + 4 * x^4 + 6 * x^2 + 9) =
  3 * x^12 + 17 * x^10 + 11 * x^9 + 5 * x^8 + 3 * x^7 + 4 * x^4 + 6 * x^2 + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2242_224255


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l2242_224275

/-- Given polynomials f, g, and h, prove their sum is equal to the specified polynomial -/
theorem sum_of_polynomials :
  let f : ℝ → ℝ := λ x => -4 * x^2 + 2 * x - 5
  let g : ℝ → ℝ := λ x => -6 * x^2 + 4 * x - 9
  let h : ℝ → ℝ := λ x => 6 * x^2 + 6 * x + 2
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l2242_224275


namespace NUMINAMATH_CALUDE_wood_measurement_l2242_224297

theorem wood_measurement (x : ℝ) : 
  (∃ rope : ℝ, rope = x + 4.5 ∧ rope / 2 = x + 1) → 
  (1/2 : ℝ) * (x + 4.5) = x - 1 :=
by sorry

end NUMINAMATH_CALUDE_wood_measurement_l2242_224297


namespace NUMINAMATH_CALUDE_max_playtime_is_180_minutes_l2242_224263

/-- Represents an arcade bundle with tokens, playtime in hours, and cost --/
structure Bundle where
  tokens : ℕ
  playtime : ℕ
  cost : ℕ

/-- Mike's weekly pay in dollars --/
def weekly_pay : ℕ := 100

/-- Mike's arcade budget in dollars (half of weekly pay) --/
def arcade_budget : ℕ := weekly_pay / 2

/-- Cost of snacks in dollars --/
def snack_cost : ℕ := 5

/-- Available bundles at the arcade --/
def bundles : List Bundle := [
  ⟨50, 1, 25⟩,   -- Bundle A
  ⟨120, 3, 45⟩,  -- Bundle B
  ⟨200, 5, 60⟩   -- Bundle C
]

/-- Remaining budget after buying snacks --/
def remaining_budget : ℕ := arcade_budget - snack_cost

/-- Function to calculate total playtime in minutes for a given bundle and quantity --/
def total_playtime (bundle : Bundle) (quantity : ℕ) : ℕ :=
  bundle.playtime * quantity * 60

/-- Theorem: The maximum playtime Mike can achieve is 180 minutes --/
theorem max_playtime_is_180_minutes :
  ∃ (bundle : Bundle) (quantity : ℕ),
    bundle ∈ bundles ∧
    bundle.cost * quantity ≤ remaining_budget ∧
    total_playtime bundle quantity = 180 ∧
    ∀ (other_bundle : Bundle) (other_quantity : ℕ),
      other_bundle ∈ bundles →
      other_bundle.cost * other_quantity ≤ remaining_budget →
      total_playtime other_bundle other_quantity ≤ 180 :=
sorry

end NUMINAMATH_CALUDE_max_playtime_is_180_minutes_l2242_224263


namespace NUMINAMATH_CALUDE_pizzeria_sales_l2242_224292

theorem pizzeria_sales (small_price large_price total_sales small_count : ℕ) 
  (h1 : small_price = 2)
  (h2 : large_price = 8)
  (h3 : total_sales = 40)
  (h4 : small_count = 8) : 
  ∃ large_count : ℕ, 
    large_count = 3 ∧ 
    small_price * small_count + large_price * large_count = total_sales :=
by
  sorry

end NUMINAMATH_CALUDE_pizzeria_sales_l2242_224292


namespace NUMINAMATH_CALUDE_chloe_profit_l2242_224220

/-- Calculates the profit from selling chocolate-dipped strawberries -/
def calculate_profit (buy_price_per_dozen : ℕ) (sell_price_per_half_dozen : ℕ) (dozens_sold : ℕ) : ℕ :=
  let cost := buy_price_per_dozen * dozens_sold
  let revenue := sell_price_per_half_dozen * 2 * dozens_sold
  revenue - cost

/-- Proves that Chloe's profit is $500 given the specified conditions -/
theorem chloe_profit :
  calculate_profit 50 30 50 = 500 := by
  sorry

end NUMINAMATH_CALUDE_chloe_profit_l2242_224220


namespace NUMINAMATH_CALUDE_competition_scores_l2242_224274

theorem competition_scores (n d : ℕ) : 
  n > 1 → 
  d > 0 → 
  d * (n * (n + 1)) / 2 = 26 * n → 
  ((n = 3 ∧ d = 13) ∨ (n = 12 ∧ d = 4) ∨ (n = 25 ∧ d = 2)) := by
  sorry

end NUMINAMATH_CALUDE_competition_scores_l2242_224274


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l2242_224256

-- Define a line by two points
def Line (x₁ y₁ x₂ y₂ : ℝ) := {(x, y) : ℝ × ℝ | ∃ t, x = x₁ + t * (x₂ - x₁) ∧ y = y₁ + t * (y₂ - y₁)}

-- Define the y-axis
def YAxis := {(x, y) : ℝ × ℝ | x = 0}

-- Theorem statement
theorem line_intersection_y_axis :
  ∃! p : ℝ × ℝ, p ∈ Line 2 9 4 15 ∧ p ∈ YAxis ∧ p = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l2242_224256


namespace NUMINAMATH_CALUDE_nancy_small_gardens_l2242_224233

/-- Given the total number of seeds, seeds planted in the big garden, and seeds per small garden,
    calculate the number of small gardens Nancy had. -/
def number_of_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Prove that Nancy had 6 small gardens given the conditions. -/
theorem nancy_small_gardens :
  number_of_small_gardens 52 28 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_nancy_small_gardens_l2242_224233


namespace NUMINAMATH_CALUDE_classroom_chairs_l2242_224259

theorem classroom_chairs (blue_chairs : ℕ) (green_chairs : ℕ) (white_chairs : ℕ) 
  (h1 : blue_chairs = 10)
  (h2 : green_chairs = 3 * blue_chairs)
  (h3 : white_chairs = blue_chairs + green_chairs - 13) :
  blue_chairs + green_chairs + white_chairs = 67 := by
  sorry

end NUMINAMATH_CALUDE_classroom_chairs_l2242_224259


namespace NUMINAMATH_CALUDE_special_polygon_properties_l2242_224273

/-- A polygon where each interior angle is 4 times the exterior angle at the same vertex -/
structure SpecialPolygon where
  vertices : ℕ
  interior_angle : Fin vertices → ℝ
  exterior_angle : Fin vertices → ℝ
  angle_relation : ∀ i, interior_angle i = 4 * exterior_angle i
  sum_exterior_angles : (Finset.univ.sum exterior_angle) = 360

theorem special_polygon_properties (Q : SpecialPolygon) :
  (Finset.univ.sum Q.interior_angle = 1440) ∧
  (∀ i j, Q.interior_angle i = Q.interior_angle j) := by
  sorry

#check special_polygon_properties

end NUMINAMATH_CALUDE_special_polygon_properties_l2242_224273


namespace NUMINAMATH_CALUDE_initial_garlic_cloves_l2242_224288

/-- 
Given that Maria used 86 cloves of garlic for roast chicken and has 7 cloves left,
prove that she initially stored 93 cloves of garlic.
-/
theorem initial_garlic_cloves (used : ℕ) (left : ℕ) (h1 : used = 86) (h2 : left = 7) :
  used + left = 93 := by
  sorry

end NUMINAMATH_CALUDE_initial_garlic_cloves_l2242_224288


namespace NUMINAMATH_CALUDE_chads_birthday_money_l2242_224299

/-- Chad's savings problem -/
theorem chads_birthday_money (
  savings_rate : ℝ)
  (other_earnings : ℝ)
  (total_savings : ℝ)
  (birthday_money : ℝ) :
  savings_rate = 0.4 →
  other_earnings = 900 →
  total_savings = 460 →
  savings_rate * (other_earnings + birthday_money) = total_savings →
  birthday_money = 250 := by
  sorry

end NUMINAMATH_CALUDE_chads_birthday_money_l2242_224299


namespace NUMINAMATH_CALUDE_rectangle_breadth_l2242_224290

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) :
  square_area = 625 →
  rectangle_area = 100 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  rectangle_area = rectangle_length * (10 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l2242_224290


namespace NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l2242_224204

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.first + (n - 1) * seq.diff

/-- The property of the arithmetic sequence we're interested in -/
def hasSpecificTerms (seq : ArithmeticSequence) : Prop :=
  seq.nthTerm 5 = 26 ∧ seq.nthTerm 8 = 50

theorem tenth_term_of_specific_sequence 
  (seq : ArithmeticSequence) 
  (h : hasSpecificTerms seq) : 
  seq.nthTerm 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_specific_sequence_l2242_224204


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l2242_224295

def m : ℕ := 2023^2 + 3^2023

theorem units_digit_of_m_squared_plus_3_to_m (m : ℕ) : (m^2 + 3^m) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l2242_224295


namespace NUMINAMATH_CALUDE_library_visitor_average_l2242_224222

/-- Calculates the average number of visitors per day for a month in a library --/
def average_visitors_per_day (sunday_visitors : ℕ) (weekday_visitors : ℕ) 
  (holiday_increase_percent : ℚ) (total_days : ℕ) (sundays : ℕ) (holidays : ℕ) : ℚ :=
  let weekdays := total_days - sundays
  let regular_weekdays := weekdays - holidays
  let holiday_visitors := weekday_visitors * (1 + holiday_increase_percent)
  let total_visitors := sunday_visitors * sundays + 
                        weekday_visitors * regular_weekdays + 
                        holiday_visitors * holidays
  total_visitors / total_days

/-- Theorem stating that the average number of visitors per day is 256 --/
theorem library_visitor_average : 
  average_visitors_per_day 540 240 (1/4) 30 4 4 = 256 := by
  sorry

end NUMINAMATH_CALUDE_library_visitor_average_l2242_224222


namespace NUMINAMATH_CALUDE_campers_rowing_morning_l2242_224235

theorem campers_rowing_morning (total_rowing : ℕ) (afternoon_rowing : ℕ) 
  (h1 : total_rowing = 34) 
  (h2 : afternoon_rowing = 21) : 
  total_rowing - afternoon_rowing = 13 := by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_morning_l2242_224235


namespace NUMINAMATH_CALUDE_sphere_radii_ratio_l2242_224291

/-- Given four spheres arranged such that each sphere touches three others and a plane,
    with two spheres having radius R and two spheres having radius r,
    prove that the ratio of the larger radius to the smaller radius is 2 + √3. -/
theorem sphere_radii_ratio (R r : ℝ) (h1 : R > 0) (h2 : r > 0)
  (h3 : R^2 + r^2 = 4*R*r) : R/r = 2 + Real.sqrt 3 ∨ r/R = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radii_ratio_l2242_224291


namespace NUMINAMATH_CALUDE_add_9999_seconds_to_8am_l2242_224209

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- Converts a natural number to a Time structure -/
def natToTime (n : Nat) : Time :=
  sorry

theorem add_9999_seconds_to_8am (startTime endTime : Time) :
  startTime = { hours := 8, minutes := 0, seconds := 0 } →
  endTime = addSeconds startTime 9999 →
  endTime = { hours := 10, minutes := 46, seconds := 39 } :=
sorry

end NUMINAMATH_CALUDE_add_9999_seconds_to_8am_l2242_224209


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2242_224201

theorem consecutive_even_numbers_sum (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4) →  -- a, b, c are consecutive even numbers
  a + b + c = 246 →                                -- their sum is 246
  b = 82                                           -- the second number is 82
:= by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2242_224201


namespace NUMINAMATH_CALUDE_no_fermat_in_sequence_l2242_224207

/-- The general term of the second-order arithmetic sequence -/
def a (n k : ℕ) : ℕ := (k - 2) * n * (n - 1) / 2 + n

/-- Fermat number of order m -/
def fermat (m : ℕ) : ℕ := 2^(2^m) + 1

/-- Statement: There are no Fermat numbers in the sequence for k > 2 -/
theorem no_fermat_in_sequence (k : ℕ) (h : k > 2) :
  ∀ (n m : ℕ), a n k ≠ fermat m :=
sorry

end NUMINAMATH_CALUDE_no_fermat_in_sequence_l2242_224207


namespace NUMINAMATH_CALUDE_intersection_point_l2242_224236

/-- The quadratic function f(x) = x^2 - 5x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 5*x + 1

/-- The y-axis is the set of points with x-coordinate 0 -/
def yAxis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

theorem intersection_point :
  (0, 1) ∈ yAxis ∧ f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l2242_224236


namespace NUMINAMATH_CALUDE_chord_equation_through_midpoint_l2242_224231

/-- The equation of a line containing a chord of an ellipse, where the chord passes through a given point and has that point as its midpoint. -/
theorem chord_equation_through_midpoint (x y : ℝ) :
  (4 * x^2 + 9 * y^2 = 144) →  -- Ellipse equation
  (3 : ℝ)^2 * 4 + 2^2 * 9 < 144 →  -- Point (3, 2) is inside the ellipse
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (4 * x₁^2 + 9 * y₁^2 = 144) ∧  -- Point (x₁, y₁) is on the ellipse
    (4 * x₂^2 + 9 * y₂^2 = 144) ∧  -- Point (x₂, y₂) is on the ellipse
    (x₁ + x₂) / 2 = 3 ∧  -- (3, 2) is the midpoint of (x₁, y₁) and (x₂, y₂)
    (y₁ + y₂) / 2 = 2 ∧
    2 * x + 3 * y - 12 = 0  -- Equation of the line containing the chord
  := by sorry

end NUMINAMATH_CALUDE_chord_equation_through_midpoint_l2242_224231


namespace NUMINAMATH_CALUDE_abs_neg_gt_neg_implies_positive_l2242_224280

theorem abs_neg_gt_neg_implies_positive (a : ℝ) : |(-a)| > -a → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_gt_neg_implies_positive_l2242_224280


namespace NUMINAMATH_CALUDE_lower_bound_of_exponential_sum_l2242_224238

theorem lower_bound_of_exponential_sum (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b + c = 1 → 
  ∃ m : ℝ, m = 4 ∧ ∀ x : ℝ, (2^a + 2^b + 2^c < x ↔ m ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_lower_bound_of_exponential_sum_l2242_224238


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2242_224228

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 6 → a^2 > 36) ∧ (∃ a, a^2 > 36 ∧ a ≤ 6) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2242_224228


namespace NUMINAMATH_CALUDE_circle_radius_l2242_224232

theorem circle_radius (x y : ℝ) : 
  (x^2 - 10*x + y^2 - 8*y + 29 = 0) → 
  (∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = 2*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l2242_224232


namespace NUMINAMATH_CALUDE_largest_number_with_sum_14_l2242_224281

def is_valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def all_valid_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, is_valid_digit d

theorem largest_number_with_sum_14 :
  ∀ n : ℕ,
    all_valid_digits n →
    digit_sum n = 14 →
    n ≤ 3332 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_14_l2242_224281


namespace NUMINAMATH_CALUDE_logarithm_problem_l2242_224248

theorem logarithm_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1)
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
  (h2 : x * y^2 = 729) :
  (Real.log (x / y) / Real.log 3)^2 = (206 - 90 * Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_logarithm_problem_l2242_224248


namespace NUMINAMATH_CALUDE_special_rectangle_dimensions_and_perimeter_l2242_224294

/-- A rectangle with integer sides where the area equals twice the perimeter -/
structure SpecialRectangle where
  a : ℕ
  b : ℕ
  h1 : a ≠ b
  h2 : a * b = 2 * (2 * a + 2 * b)

theorem special_rectangle_dimensions_and_perimeter (rect : SpecialRectangle) :
  (rect.a = 12 ∧ rect.b = 6) ∨ (rect.a = 6 ∧ rect.b = 12) ∧
  2 * (rect.a + rect.b) = 36 := by
  sorry

#check special_rectangle_dimensions_and_perimeter

end NUMINAMATH_CALUDE_special_rectangle_dimensions_and_perimeter_l2242_224294


namespace NUMINAMATH_CALUDE_circle_condition_l2242_224237

theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x + 2*y - m = 0) ↔ m > -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l2242_224237


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2242_224200

theorem quadratic_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - a ≠ 0) → a < -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2242_224200


namespace NUMINAMATH_CALUDE_r_amount_calculation_l2242_224249

def total_amount : ℝ := 5000

theorem r_amount_calculation (p_amount q_amount r_amount : ℝ) 
  (h1 : p_amount + q_amount + r_amount = total_amount)
  (h2 : r_amount = (2/3) * (p_amount + q_amount)) :
  r_amount = 2000 := by
  sorry

end NUMINAMATH_CALUDE_r_amount_calculation_l2242_224249


namespace NUMINAMATH_CALUDE_weaver_output_increase_l2242_224239

theorem weaver_output_increase (first_day_output : ℝ) (total_days : ℕ) (total_output : ℝ) :
  first_day_output = 5 ∧ total_days = 30 ∧ total_output = 390 →
  ∃ (daily_increase : ℝ),
    daily_increase = 16/29 ∧
    total_output = total_days * first_day_output + (total_days * (total_days - 1) / 2) * daily_increase :=
by sorry

end NUMINAMATH_CALUDE_weaver_output_increase_l2242_224239
