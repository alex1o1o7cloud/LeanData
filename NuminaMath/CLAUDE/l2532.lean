import Mathlib

namespace NUMINAMATH_CALUDE_point_on_line_l2532_253241

/-- Given a line with equation x = 2y + 3 and two points (m, n) and (m + 2, n + k) on this line,
    prove that k = 0. -/
theorem point_on_line (m n k : ℝ) : 
  (m = 2 * n + 3) → 
  (m + 2 = 2 * (n + k) + 3) → 
  k = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2532_253241


namespace NUMINAMATH_CALUDE_system_no_solution_l2532_253255

/-- The coefficient matrix of the system of equations -/
def A (n : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![n, 1, 1;
     1, n, 1;
     1, 1, n]

/-- The theorem stating that the system has no solution iff n = -2 -/
theorem system_no_solution (n : ℝ) :
  (∀ x y z : ℝ, n * x + y + z ≠ 1 ∨ x + n * y + z ≠ 1 ∨ x + y + n * z ≠ 1) ↔ n = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_no_solution_l2532_253255


namespace NUMINAMATH_CALUDE_curve_perimeter_ge_twice_diagonal_curve_perimeter_eq_twice_diagonal_l2532_253286

/-- A closed curve in 2D space -/
structure ClosedCurve where
  -- Add necessary fields/axioms for a closed curve

/-- A rectangle in 2D space -/
structure Rectangle where
  -- Add necessary fields for a rectangle (e.g., width, height, position)

/-- The perimeter of a closed curve -/
noncomputable def perimeter (c : ClosedCurve) : ℝ :=
  sorry

/-- The diagonal length of a rectangle -/
def diagonal (r : Rectangle) : ℝ :=
  sorry

/-- Predicate to check if a curve intersects all sides of a rectangle -/
def intersectsAllSides (c : ClosedCurve) (r : Rectangle) : Prop :=
  sorry

theorem curve_perimeter_ge_twice_diagonal 
  (c : ClosedCurve) (r : Rectangle) 
  (h : intersectsAllSides c r) : 
  perimeter c ≥ 2 * diagonal r :=
sorry

/-- Condition for equality -/
def equalityCondition (c : ClosedCurve) (r : Rectangle) : Prop :=
  sorry

theorem curve_perimeter_eq_twice_diagonal 
  (c : ClosedCurve) (r : Rectangle) 
  (h1 : intersectsAllSides c r)
  (h2 : equalityCondition c r) : 
  perimeter c = 2 * diagonal r :=
sorry

end NUMINAMATH_CALUDE_curve_perimeter_ge_twice_diagonal_curve_perimeter_eq_twice_diagonal_l2532_253286


namespace NUMINAMATH_CALUDE_number_calculation_l2532_253211

theorem number_calculation (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 25 → (40/100 : ℝ) * N = 300 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2532_253211


namespace NUMINAMATH_CALUDE_sqrt_four_squared_l2532_253221

theorem sqrt_four_squared : (Real.sqrt 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_l2532_253221


namespace NUMINAMATH_CALUDE_distinct_collections_eq_sixteen_l2532_253233

/-- The number of vowels in MATHCOUNTS -/
def num_vowels : ℕ := 3

/-- The number of consonants in MATHCOUNTS excluding T -/
def num_consonants_without_t : ℕ := 5

/-- The number of T's in MATHCOUNTS -/
def num_t : ℕ := 2

/-- The number of vowels to be selected -/
def vowels_to_select : ℕ := 3

/-- The number of consonants to be selected -/
def consonants_to_select : ℕ := 2

/-- The function to calculate the number of distinct collections -/
def distinct_collections : ℕ :=
  (Nat.choose num_vowels vowels_to_select) * (Nat.choose num_consonants_without_t consonants_to_select) +
  (Nat.choose num_vowels vowels_to_select) * (Nat.choose num_consonants_without_t (consonants_to_select - 1)) +
  (Nat.choose num_vowels vowels_to_select) * (Nat.choose num_consonants_without_t (consonants_to_select - 2))

theorem distinct_collections_eq_sixteen :
  distinct_collections = 16 := by sorry

end NUMINAMATH_CALUDE_distinct_collections_eq_sixteen_l2532_253233


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2010th_term_l2532_253269

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  p : ℚ
  q : ℚ
  first_term : ℚ := p
  second_term : ℚ := 7
  third_term : ℚ := 3*p - q
  fourth_term : ℚ := 5*p + q
  is_arithmetic : ∃ d : ℚ, second_term = first_term + d ∧ 
                           third_term = second_term + d ∧ 
                           fourth_term = third_term + d

/-- The 2010th term of the arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1) * ((seq.fourth_term - seq.first_term) / 3)

/-- Theorem stating that the 2010th term is 6253 -/
theorem arithmetic_sequence_2010th_term (seq : ArithmeticSequence) :
  nth_term seq 2010 = 6253 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2010th_term_l2532_253269


namespace NUMINAMATH_CALUDE_paintings_per_room_l2532_253200

theorem paintings_per_room (total_paintings : ℕ) (num_rooms : ℕ) 
  (h1 : total_paintings = 32) 
  (h2 : num_rooms = 4) 
  (h3 : total_paintings % num_rooms = 0) : 
  total_paintings / num_rooms = 8 := by
  sorry

end NUMINAMATH_CALUDE_paintings_per_room_l2532_253200


namespace NUMINAMATH_CALUDE_adams_pants_l2532_253217

/-- The number of pairs of pants Adam initially took out -/
def P : ℕ := 31

/-- The number of jumpers Adam took out -/
def jumpers : ℕ := 4

/-- The number of pajama sets Adam took out -/
def pajama_sets : ℕ := 4

/-- The number of t-shirts Adam took out -/
def tshirts : ℕ := 20

/-- The number of friends who donate the same amount as Adam -/
def friends : ℕ := 3

/-- The total number of articles of clothing being donated -/
def total_donated : ℕ := 126

theorem adams_pants :
  P = 31 ∧
  (4 * (P + jumpers + 2 * pajama_sets + tshirts) / 2 = total_donated) :=
sorry

end NUMINAMATH_CALUDE_adams_pants_l2532_253217


namespace NUMINAMATH_CALUDE_insufficient_funds_for_all_l2532_253220

theorem insufficient_funds_for_all 
  (num_workers : ℕ) 
  (total_salary : ℕ) 
  (item_cost : ℕ) 
  (h1 : num_workers = 5) 
  (h2 : total_salary = 1500) 
  (h3 : item_cost = 320) : 
  ∃ (worker : ℕ), worker ≤ num_workers ∧ total_salary < num_workers * item_cost :=
sorry

end NUMINAMATH_CALUDE_insufficient_funds_for_all_l2532_253220


namespace NUMINAMATH_CALUDE_only_log29_undetermined_l2532_253272

-- Define the given logarithms
def log7 : ℝ := 0.8451
def log10 : ℝ := 1

-- Define a function to represent whether a logarithm can be determined
def can_determine (x : ℝ) : Prop := 
  ∃ (f : ℝ → ℝ → ℝ), f log7 log10 = Real.log x

-- State the theorem
theorem only_log29_undetermined :
  ¬(can_determine 29) ∧ 
  can_determine (5/9) ∧ 
  can_determine 35 ∧ 
  can_determine 700 ∧ 
  can_determine 0.6 := by
  sorry


end NUMINAMATH_CALUDE_only_log29_undetermined_l2532_253272


namespace NUMINAMATH_CALUDE_divisors_of_720_l2532_253229

theorem divisors_of_720 : ∃ (n : ℕ), n = 720 ∧ (Finset.card (Finset.filter (λ x => n % x = 0) (Finset.range (n + 1)))) = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_720_l2532_253229


namespace NUMINAMATH_CALUDE_fraction_equality_l2532_253226

theorem fraction_equality (x y : ℝ) (h : (1/x + 1/y)/(1/x - 1/y) = 101) : 
  (x - y)/(x + y) = -1/5101 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2532_253226


namespace NUMINAMATH_CALUDE_unique_prime_in_form_l2532_253216

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def number_form (A : ℕ) : ℕ := 305200 + A

theorem unique_prime_in_form :
  ∃! A : ℕ, A < 10 ∧ is_prime (number_form A) ∧ number_form A = 305201 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_form_l2532_253216


namespace NUMINAMATH_CALUDE_cost_of_juices_l2532_253292

/-- The cost of juices and sandwiches problem -/
theorem cost_of_juices (sandwich_cost juice_cost : ℚ) : 
  (2 * sandwich_cost = 6) →
  (sandwich_cost + juice_cost = 5) →
  (5 * juice_cost = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_juices_l2532_253292


namespace NUMINAMATH_CALUDE_percent_profit_calculation_l2532_253260

theorem percent_profit_calculation (C S : ℝ) (h : 55 * C = 50 * S) : 
  (S - C) / C * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percent_profit_calculation_l2532_253260


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2532_253228

/-- An arithmetic sequence with specific terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = 3)
  (h_a4 : a 4 = 7)
  (h_ak : ∃ k : ℕ, a k = 15) :
  ∃ k : ℕ, a k = 15 ∧ k = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2532_253228


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_power_5_l2532_253261

theorem nearest_integer_to_3_plus_sqrt2_power_5 :
  ∃ n : ℤ, n = 1926 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^5 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^5 - (m : ℝ)| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_power_5_l2532_253261


namespace NUMINAMATH_CALUDE_train_passing_time_l2532_253263

theorem train_passing_time (fast_length slow_length : ℝ) (time_slow_observes : ℝ) :
  fast_length = 150 →
  slow_length = 200 →
  time_slow_observes = 6 →
  ∃ time_fast_observes : ℝ,
    time_fast_observes = 8 ∧
    fast_length / time_slow_observes = slow_length / time_fast_observes :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l2532_253263


namespace NUMINAMATH_CALUDE_new_assistant_drawing_time_main_theorem_l2532_253201

/-- Represents a beer barrel with two taps -/
structure BeerBarrel where
  capacity : ℕ
  midwayTapRate : ℚ  -- litres per minute
  lowerTapRate : ℚ   -- litres per minute

/-- Calculates the time taken to empty half the barrel using the midway tap -/
def timeToHalfEmpty (barrel : BeerBarrel) : ℚ :=
  (barrel.capacity / 2) / barrel.midwayTapRate

/-- Calculates the additional time the lower tap was used -/
def additionalLowerTapTime : ℕ := 24

/-- Theorem: The new assistant drew beer for 150 minutes -/
theorem new_assistant_drawing_time (barrel : BeerBarrel)
    (h1 : barrel.capacity = 36)
    (h2 : barrel.midwayTapRate = 1 / 6)
    (h3 : barrel.lowerTapRate = 1 / 4)
    : ℚ :=
  150

/-- Main theorem to prove -/
theorem main_theorem (barrel : BeerBarrel)
    (h1 : barrel.capacity = 36)
    (h2 : barrel.midwayTapRate = 1 / 6)
    (h3 : barrel.lowerTapRate = 1 / 4)
    : new_assistant_drawing_time barrel h1 h2 h3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_new_assistant_drawing_time_main_theorem_l2532_253201


namespace NUMINAMATH_CALUDE_pizza_slices_left_l2532_253290

theorem pizza_slices_left (total_slices : ℕ) (eaten_slices : ℕ) (h1 : total_slices = 32) (h2 : eaten_slices = 25) :
  total_slices - eaten_slices = 7 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_l2532_253290


namespace NUMINAMATH_CALUDE_abcd_addition_l2532_253295

theorem abcd_addition (a b c d : Nat) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ d ≥ 0 ∧ d ≤ 9 →
  (1000 * a + 100 * b + 10 * c + d) + (1000 * a + 100 * b + 10 * c + d) = 5472 →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_abcd_addition_l2532_253295


namespace NUMINAMATH_CALUDE_milk_for_nine_cookies_l2532_253284

/-- The number of pints in a quart -/
def pints_per_quart : ℚ := 2

/-- The number of cookies that can be baked with 3 quarts of milk -/
def cookies_per_three_quarts : ℚ := 18

/-- The number of cookies we want to bake -/
def target_cookies : ℚ := 9

/-- The function that calculates the number of pints of milk needed for a given number of cookies -/
def milk_needed (cookies : ℚ) : ℚ :=
  (3 * pints_per_quart * cookies) / cookies_per_three_quarts

theorem milk_for_nine_cookies :
  milk_needed target_cookies = 3 := by
  sorry

end NUMINAMATH_CALUDE_milk_for_nine_cookies_l2532_253284


namespace NUMINAMATH_CALUDE_unique_factorization_2210_l2532_253240

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ a * b = 2210

theorem unique_factorization_2210 :
  ∃! p : ℕ × ℕ, valid_factorization p.1 p.2 ∧ p.1 ≤ p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_factorization_2210_l2532_253240


namespace NUMINAMATH_CALUDE_no_prime_divisible_by_35_l2532_253264

/-- A number is prime if it has exactly two distinct positive divisors -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_35 :
  ¬∃ p : ℕ, isPrime p ∧ 35 ∣ p :=
by sorry

end NUMINAMATH_CALUDE_no_prime_divisible_by_35_l2532_253264


namespace NUMINAMATH_CALUDE_shelleys_weight_l2532_253202

/-- Given the weights of three people on a scale in pairs, find one person's weight -/
theorem shelleys_weight (p s r : ℕ) : 
  p + s = 151 → s + r = 132 → p + r = 115 → s = 84 := by
  sorry

end NUMINAMATH_CALUDE_shelleys_weight_l2532_253202


namespace NUMINAMATH_CALUDE_part_one_part_two_l2532_253214

-- Define α
variable (α : Real)

-- Given condition
axiom tan_alpha : Real.tan α = 3

-- Theorem for part (I)
theorem part_one : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5/7 := by
  sorry

-- Theorem for part (II)
theorem part_two :
  (Real.sin α + Real.cos α)^2 = 8/5 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2532_253214


namespace NUMINAMATH_CALUDE_water_boiling_time_l2532_253239

/-- Time for water to boil away given initial conditions -/
theorem water_boiling_time 
  (T₀ : ℝ) (Tₘ : ℝ) (t : ℝ) (c : ℝ) (L : ℝ)
  (h₁ : T₀ = 20)
  (h₂ : Tₘ = 100)
  (h₃ : t = 10)
  (h₄ : c = 4200)
  (h₅ : L = 2.3e6) :
  ∃ t₁ : ℝ, t₁ ≥ 67.5 ∧ t₁ < 68.5 :=
by sorry

end NUMINAMATH_CALUDE_water_boiling_time_l2532_253239


namespace NUMINAMATH_CALUDE_root_modulus_preservation_l2532_253275

theorem root_modulus_preservation (a b c : ℂ) :
  (∀ z : ℂ, z^3 + a*z^2 + b*z + c = 0 → Complex.abs z = 1) →
  (∀ z : ℂ, z^3 + Complex.abs a*z^2 + Complex.abs b*z + Complex.abs c = 0 → Complex.abs z = 1) :=
by sorry

end NUMINAMATH_CALUDE_root_modulus_preservation_l2532_253275


namespace NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_q_l2532_253274

-- Define the conditions
def p (x : ℝ) : Prop := |x - 1| < 2
def q (x : ℝ) : Prop := x^2 - 5*x - 6 < 0

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_but_not_necessary (p q : ℝ → Prop) : Prop :=
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x)

-- State the theorem
theorem p_sufficient_but_not_necessary_for_q :
  sufficient_but_not_necessary p q := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_but_not_necessary_for_q_l2532_253274


namespace NUMINAMATH_CALUDE_product_equals_difference_of_powers_l2532_253231

theorem product_equals_difference_of_powers : 
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * 
  (3^32 + 5^32) * (3^64 + 5^64) * (3^128 + 5^128) = 3^256 - 5^256 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_difference_of_powers_l2532_253231


namespace NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l2532_253219

/-- Given a cone whose lateral surface, when unfolded, forms a semicircle with area 8π,
    the volume of the cone is (8√3π)/3 -/
theorem cone_volume_from_lateral_surface (r h : ℝ) : 
  r > 0 → h > 0 → 
  (π * r * (r^2 + h^2).sqrt / 2 = 8 * π) → 
  (1/3 * π * r^2 * h = 8 * Real.sqrt 3 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_lateral_surface_l2532_253219


namespace NUMINAMATH_CALUDE_teacher_distribution_l2532_253235

theorem teacher_distribution (n : ℕ) (m : ℕ) :
  n = 6 →
  m = 3 →
  (Nat.choose n 3) * (Nat.choose (n - 3) 2) * (Nat.factorial m) = 360 := by
  sorry

end NUMINAMATH_CALUDE_teacher_distribution_l2532_253235


namespace NUMINAMATH_CALUDE_distance_between_centers_l2532_253256

/-- The distance between the centers of inscribed and circumscribed circles in a right triangle -/
theorem distance_between_centers (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let r := (a * b) / (2 * s)
  let x_i := r
  let y_i := r
  let x_o := c / 2
  let y_o := 0
  Real.sqrt ((x_o - x_i)^2 + (y_o - y_i)^2) = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_centers_l2532_253256


namespace NUMINAMATH_CALUDE_checkerboard_covering_l2532_253246

/-- Represents an L-shaped piece that can cover three squares on a checkerboard -/
inductive LPiece
| mk : LPiece

/-- Represents a square on the checkerboard -/
structure Square where
  x : Nat
  y : Nat

/-- Represents a checkerboard with one square removed -/
structure Checkerboard (n : Nat) where
  sideLength : Nat
  removedSquare : Square
  validSideLength : sideLength = 2^n
  validRemovedSquare : removedSquare.x < sideLength ∧ removedSquare.y < sideLength

/-- Represents a covering of the checkerboard with L-shaped pieces -/
def Covering (n : Nat) := List (Square × Square × Square)

/-- Checks if a covering is valid for a given checkerboard -/
def isValidCovering (n : Nat) (board : Checkerboard n) (covering : Covering n) : Prop :=
  -- Each L-piece covers exactly three squares
  -- No gaps or overlaps in the covering
  -- The removed square is not covered
  sorry

/-- Main theorem: Any checkerboard with one square removed can be covered by L-shaped pieces -/
theorem checkerboard_covering (n : Nat) (h : n > 0) (board : Checkerboard n) :
  ∃ (covering : Covering n), isValidCovering n board covering :=
sorry

end NUMINAMATH_CALUDE_checkerboard_covering_l2532_253246


namespace NUMINAMATH_CALUDE_weighted_sum_square_inequality_l2532_253268

theorem weighted_sum_square_inequality (a b x y : ℝ) 
  (h1 : a + b = 1) (h2 : a ≥ 0) (h3 : b ≥ 0) : 
  a * x^2 + b * y^2 - (a * x + b * y)^2 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_weighted_sum_square_inequality_l2532_253268


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l2532_253249

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 10)
  (h2 : sum_first_two = 6) :
  ∃ (a : ℝ), 
    (a = 10 - 10 * Real.sqrt (2/5) ∨ a = 10 + 10 * Real.sqrt (2/5)) ∧ 
    (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l2532_253249


namespace NUMINAMATH_CALUDE_tree_planting_equation_system_l2532_253280

theorem tree_planting_equation_system :
  ∀ (x y : ℕ),
  (x + y = 20) →
  (3 * x + 2 * y = 52) →
  (∀ (total_pioneers total_trees boys_trees girls_trees : ℕ),
    total_pioneers = 20 →
    total_trees = 52 →
    boys_trees = 3 →
    girls_trees = 2 →
    x + y = total_pioneers ∧
    3 * x + 2 * y = total_trees) :=
by sorry

end NUMINAMATH_CALUDE_tree_planting_equation_system_l2532_253280


namespace NUMINAMATH_CALUDE_cosine_identity_l2532_253227

/-- Proves that 2cos(16°)cos(29°) - cos(13°) = √2/2 -/
theorem cosine_identity : 
  2 * Real.cos (16 * π / 180) * Real.cos (29 * π / 180) - Real.cos (13 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l2532_253227


namespace NUMINAMATH_CALUDE_line_parallel_plane_implies_parallel_to_all_lines_false_l2532_253207

/-- A line in 3D space --/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space --/
structure Plane3D where
  -- Define properties of a plane

/-- Defines when a line is parallel to a plane --/
def line_parallel_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is contained in a plane --/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when two lines are parallel --/
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- The statement to be proven false --/
theorem line_parallel_plane_implies_parallel_to_all_lines_false :
  ¬ (∀ (l : Line3D) (p : Plane3D),
    line_parallel_plane l p →
    ∀ (l' : Line3D), line_in_plane l' p →
    lines_parallel l l') :=
  sorry

end NUMINAMATH_CALUDE_line_parallel_plane_implies_parallel_to_all_lines_false_l2532_253207


namespace NUMINAMATH_CALUDE_mad_hatter_win_condition_l2532_253258

/-- Represents the fraction of voters for each candidate and undecided voters -/
structure VoteDistribution where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ
  undecided : ℝ

/-- Represents the final vote count after undecided voters have voted -/
structure FinalVotes where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ

def minimum_fraction_for_mad_hatter (initial_votes : VoteDistribution) : ℝ :=
  0.7

theorem mad_hatter_win_condition (initial_votes : VoteDistribution) 
  (h1 : initial_votes.mad_hatter = 0.2)
  (h2 : initial_votes.march_hare = 0.25)
  (h3 : initial_votes.dormouse = 0.3)
  (h4 : initial_votes.undecided = 0.25)
  (h5 : initial_votes.mad_hatter + initial_votes.march_hare + initial_votes.dormouse + initial_votes.undecided = 1) :
  ∀ (final_votes : FinalVotes),
    (final_votes.mad_hatter ≥ initial_votes.mad_hatter + initial_votes.undecided * minimum_fraction_for_mad_hatter initial_votes) →
    (final_votes.march_hare ≤ initial_votes.march_hare + initial_votes.undecided * (1 - minimum_fraction_for_mad_hatter initial_votes)) →
    (final_votes.dormouse ≤ initial_votes.dormouse + initial_votes.undecided * (1 - minimum_fraction_for_mad_hatter initial_votes)) →
    (final_votes.mad_hatter + final_votes.march_hare + final_votes.dormouse = 1) →
    (final_votes.mad_hatter ≥ final_votes.march_hare ∧ final_votes.mad_hatter ≥ final_votes.dormouse) :=
  sorry

end NUMINAMATH_CALUDE_mad_hatter_win_condition_l2532_253258


namespace NUMINAMATH_CALUDE_alex_savings_l2532_253225

def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def grocery_percentage : ℝ := 0.05
def num_trips : ℕ := 40
def grocery_value : ℝ := 800

theorem alex_savings (initial_savings : ℝ) : 
  initial_savings + 
  (num_trips : ℝ) * trip_charge + 
  grocery_percentage * grocery_value = 
  car_cost :=
by sorry

end NUMINAMATH_CALUDE_alex_savings_l2532_253225


namespace NUMINAMATH_CALUDE_exists_remarkable_polygon_for_n_gt_4_l2532_253294

/-- A remarkable polygon is a grid polygon that is not a rectangle and can form a similar polygon from several of its copies. -/
structure RemarkablePolygon (n : ℕ) where
  cells : ℕ
  not_rectangle : cells ≠ 4
  can_form_similar : True  -- Simplified condition for similarity

/-- For all integers n > 4, there exists a remarkable polygon with n cells. -/
theorem exists_remarkable_polygon_for_n_gt_4 (n : ℕ) (h : n > 4) :
  ∃ (P : RemarkablePolygon n), P.cells = n :=
sorry

end NUMINAMATH_CALUDE_exists_remarkable_polygon_for_n_gt_4_l2532_253294


namespace NUMINAMATH_CALUDE_interest_calculation_l2532_253277

def deposit : ℝ := 30000
def term : ℝ := 3
def interest_rate : ℝ := 0.047
def tax_rate : ℝ := 0.2

def pre_tax_interest : ℝ := deposit * interest_rate * term
def after_tax_interest : ℝ := pre_tax_interest * (1 - tax_rate)
def total_withdrawal : ℝ := deposit + after_tax_interest

theorem interest_calculation :
  after_tax_interest = 3372 ∧ total_withdrawal = 33372 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l2532_253277


namespace NUMINAMATH_CALUDE_rectangle_similarity_symmetry_l2532_253237

/-- A rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Two rectangles are similar if their aspect ratios are equal -/
def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r1.height = r2.width / r2.height

/-- A rectangle can be formed from congruent copies of another rectangle -/
def can_form (r1 r2 : Rectangle) : Prop :=
  ∃ (m n p q : ℕ), m * r1.width + n * r1.height = r2.width ∧
                   p * r1.width + q * r1.height = r2.height

theorem rectangle_similarity_symmetry (A B : Rectangle) :
  (∃ C : Rectangle, similar C B ∧ can_form C A) →
  (∃ D : Rectangle, similar D A ∧ can_form D B) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_similarity_symmetry_l2532_253237


namespace NUMINAMATH_CALUDE_power_mod_eleven_l2532_253222

theorem power_mod_eleven : 3^225 ≡ 1 [MOD 11] := by sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l2532_253222


namespace NUMINAMATH_CALUDE_cyrus_shot_percentage_l2532_253238

def total_shots : ℕ := 20
def missed_shots : ℕ := 4

def shots_made : ℕ := total_shots - missed_shots

def percentage_made : ℚ := (shots_made : ℚ) / (total_shots : ℚ) * 100

theorem cyrus_shot_percentage :
  percentage_made = 80 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_shot_percentage_l2532_253238


namespace NUMINAMATH_CALUDE_subtract_negatives_l2532_253297

theorem subtract_negatives : -2 - 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_l2532_253297


namespace NUMINAMATH_CALUDE_lea_binders_purchase_l2532_253276

/-- The number of binders Léa bought -/
def num_binders : ℕ := 3

/-- The cost of one book in dollars -/
def book_cost : ℕ := 16

/-- The cost of one binder in dollars -/
def binder_cost : ℕ := 2

/-- The cost of one notebook in dollars -/
def notebook_cost : ℕ := 1

/-- The number of notebooks Léa bought -/
def num_notebooks : ℕ := 6

/-- The total cost of Léa's purchases in dollars -/
def total_cost : ℕ := 28

theorem lea_binders_purchase :
  book_cost + binder_cost * num_binders + notebook_cost * num_notebooks = total_cost :=
by sorry

end NUMINAMATH_CALUDE_lea_binders_purchase_l2532_253276


namespace NUMINAMATH_CALUDE_juliet_supporter_in_capulet_probability_l2532_253273

-- Define the population distribution
def montague_pop : ℚ := 5/8
def capulet_pop : ℚ := 3/16
def verona_pop : ℚ := 1/8
def mercutio_pop : ℚ := 1 - (montague_pop + capulet_pop + verona_pop)

-- Define the support rates
def montague_romeo_rate : ℚ := 4/5
def capulet_juliet_rate : ℚ := 7/10
def verona_romeo_rate : ℚ := 13/20
def mercutio_juliet_rate : ℚ := 11/20

-- Define the total Juliet supporters
def total_juliet_supporters : ℚ := capulet_pop * capulet_juliet_rate + mercutio_pop * mercutio_juliet_rate

-- Define the probability
def prob_juliet_in_capulet : ℚ := (capulet_pop * capulet_juliet_rate) / total_juliet_supporters

-- Theorem statement
theorem juliet_supporter_in_capulet_probability :
  ∃ (ε : ℚ), abs (prob_juliet_in_capulet - 66/100) < ε ∧ ε < 1/100 :=
sorry

end NUMINAMATH_CALUDE_juliet_supporter_in_capulet_probability_l2532_253273


namespace NUMINAMATH_CALUDE_xiaoming_multiplication_l2532_253213

theorem xiaoming_multiplication (a : ℝ) : 
  20.18 * a = 20.18 * (a - 1) + 2270.25 → a = 113.5 := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_multiplication_l2532_253213


namespace NUMINAMATH_CALUDE_function_zeros_count_l2532_253208

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_zeros_count
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_periodic : is_periodic f 3)
  (h_sin : ∀ x ∈ Set.Ioo 0 (3/2), f x = Real.sin (Real.pi * x))
  (h_zero : f (3/2) = 0) :
  ∃ S : Finset ℝ, S.card = 7 ∧ (∀ x ∈ S, x ∈ Set.Icc 0 6 ∧ f x = 0) ∧
    (∀ x ∈ Set.Icc 0 6, f x = 0 → x ∈ S) :=
sorry

end NUMINAMATH_CALUDE_function_zeros_count_l2532_253208


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l2532_253212

/-- A police emergency number is a positive integer that ends in 133 in decimal representation. -/
def PoliceEmergencyNumber (n : ℕ) : Prop :=
  n > 0 ∧ n % 1000 = 133

/-- Every police emergency number has a prime divisor greater than 7. -/
theorem police_emergency_number_prime_divisor (n : ℕ) (h : PoliceEmergencyNumber n) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l2532_253212


namespace NUMINAMATH_CALUDE_nice_sequence_divisibility_exists_nice_sequence_not_divisible_l2532_253271

/-- Definition of a nice sequence -/
def NiceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ (∀ n, a (2 * n) = 2 * a n)

theorem nice_sequence_divisibility (a : ℕ → ℕ) (p : ℕ) (hp : Prime p) (h_nice : NiceSequence a) (h_p_gt_a1 : p > a 1) :
  ∃ k, p ∣ a k := by
  sorry

theorem exists_nice_sequence_not_divisible (p : ℕ) (hp : Prime p) (h_p_gt_2 : p > 2) :
  ∃ a : ℕ → ℕ, NiceSequence a ∧ ∀ n, ¬(p ∣ a n) := by
  sorry

end NUMINAMATH_CALUDE_nice_sequence_divisibility_exists_nice_sequence_not_divisible_l2532_253271


namespace NUMINAMATH_CALUDE_face_mask_profit_l2532_253262

/-- Calculates the profit from selling face masks given the following conditions:
  * 12 boxes of face masks were bought at $9 per box
  * Each box contains 50 masks
  * 6 boxes were repacked and sold at $5 per 25 pieces
  * The remaining 300 masks were sold in baggies at 10 pieces for $3
-/
def calculate_profit : ℤ :=
  let boxes_bought := 12
  let cost_per_box := 9
  let masks_per_box := 50
  let repacked_boxes := 6
  let price_per_25_pieces := 5
  let remaining_masks := 300
  let price_per_10_pieces := 3

  let total_cost := boxes_bought * cost_per_box
  let revenue_repacked := repacked_boxes * (masks_per_box / 25) * price_per_25_pieces
  let revenue_baggies := (remaining_masks / 10) * price_per_10_pieces
  let total_revenue := revenue_repacked + revenue_baggies
  
  total_revenue - total_cost

/-- Theorem stating that the profit from selling face masks under the given conditions is $42 -/
theorem face_mask_profit : calculate_profit = 42 := by
  sorry

end NUMINAMATH_CALUDE_face_mask_profit_l2532_253262


namespace NUMINAMATH_CALUDE_tony_age_at_period_end_l2532_253283

/-- Represents Tony's work and payment details -/
structure TonyWork where
  hoursPerDay : ℕ
  payPerHourPerYear : ℚ
  daysWorked : ℕ
  totalEarned : ℚ

/-- Proves Tony's age at the end of the six-month period -/
theorem tony_age_at_period_end (tw : TonyWork) 
  (h1 : tw.hoursPerDay = 2)
  (h2 : tw.payPerHourPerYear = 1/2)
  (h3 : tw.daysWorked = 50)
  (h4 : tw.totalEarned = 630) :
  ∃ (age : ℕ), age = 13 ∧ 
  (∃ (x : ℕ), x ≤ tw.daysWorked ∧
    (age - 1) * tw.hoursPerDay * tw.payPerHourPerYear * x + 
    age * tw.hoursPerDay * tw.payPerHourPerYear * (tw.daysWorked - x) = tw.totalEarned) :=
sorry

end NUMINAMATH_CALUDE_tony_age_at_period_end_l2532_253283


namespace NUMINAMATH_CALUDE_units_digit_factorial_product_15_l2532_253270

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def productFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.foldl (·*·) 1

theorem units_digit_factorial_product_15 :
  unitsDigit (productFactorials 15) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_product_15_l2532_253270


namespace NUMINAMATH_CALUDE_number_difference_l2532_253299

theorem number_difference (a b : ℕ) : 
  a + b = 20000 → 7 * a = b → b - a = 15000 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2532_253299


namespace NUMINAMATH_CALUDE_division_theorem_l2532_253245

theorem division_theorem (a b q r : ℕ) (h1 : a = 1270) (h2 : b = 17) (h3 : q = 74) (h4 : r = 12) :
  a = b * q + r ∧ r < b := by
  sorry

end NUMINAMATH_CALUDE_division_theorem_l2532_253245


namespace NUMINAMATH_CALUDE_wrong_number_calculation_l2532_253236

def wrong_number (n : ℕ) (initial_avg : ℚ) (correct_num : ℕ) (correct_avg : ℚ) : ℚ :=
  n * initial_avg + correct_num - (n * correct_avg)

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg : ℚ) (correct_num : ℕ) :
  n = 10 →
  initial_avg = 5 →
  correct_num = 36 →
  correct_avg = 6 →
  wrong_number n initial_avg correct_num correct_avg = 26 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_calculation_l2532_253236


namespace NUMINAMATH_CALUDE_circle_symmetry_l2532_253206

/-- Given two circles and a line of symmetry, prove the value of a parameter -/
theorem circle_symmetry (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - a*x + 2*y + 1 = 0 ↔ 
    ∃ x' y' : ℝ, x'^2 + y'^2 = 1 ∧ 
    (x + x')/2 - (y + y')/2 = 1 ∧
    (x - x')^2 + (y - y')^2 = ((x + x')/2 - x)^2 + ((y + y')/2 - y)^2) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2532_253206


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2532_253204

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : 
  num_diagonals_nonagon = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2532_253204


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2532_253267

theorem simplify_and_evaluate (x y : ℚ) 
  (hx : x = -1) 
  (hy : y = 1/3) : 
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2532_253267


namespace NUMINAMATH_CALUDE_square_sum_from_system_l2532_253210

theorem square_sum_from_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = (14400 - 4056) / 169 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_system_l2532_253210


namespace NUMINAMATH_CALUDE_expression_equality_l2532_253287

theorem expression_equality :
  ((-3)^2 ≠ -3^2) ∧
  ((-3)^2 = 3^2) ∧
  ((-2)^3 = -2^3) ∧
  (|-2|^3 = |-2^3|) :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l2532_253287


namespace NUMINAMATH_CALUDE_p_properties_l2532_253218

-- Define the polynomial
def p (x y : ℝ) : ℝ := x^2 * y^3 - 3 * x * y^3 - 2

-- Define the degree of a monomial
def monomial_degree (m : ℕ × ℕ) : ℕ := m.1 + m.2

-- Define the degree of a polynomial
def polynomial_degree (p : ℝ → ℝ → ℝ) : ℕ :=
  -- This definition is a placeholder and needs to be implemented
  sorry

-- Define the number of terms in a polynomial
def number_of_terms (p : ℝ → ℝ → ℝ) : ℕ :=
  -- This definition is a placeholder and needs to be implemented
  sorry

-- Theorem stating the properties of the polynomial p
theorem p_properties :
  polynomial_degree p = 5 ∧ number_of_terms p = 3 := by
  sorry

end NUMINAMATH_CALUDE_p_properties_l2532_253218


namespace NUMINAMATH_CALUDE_total_whales_count_l2532_253230

/-- The total number of whales observed across three trips -/
def total_whales : ℕ := by sorry

/-- The number of male whales observed in the first trip -/
def first_trip_males : ℕ := 28

/-- The number of female whales observed in the first trip -/
def first_trip_females : ℕ := 2 * first_trip_males

/-- The number of baby whales observed in the second trip -/
def second_trip_babies : ℕ := 8

/-- The number of whales in each family group (baby + two parents) -/
def whales_per_family : ℕ := 3

/-- The number of male whales observed in the third trip -/
def third_trip_males : ℕ := first_trip_males / 2

/-- The number of female whales observed in the third trip -/
def third_trip_females : ℕ := first_trip_females

/-- Theorem stating that the total number of whales observed is 178 -/
theorem total_whales_count : total_whales = 178 := by sorry

end NUMINAMATH_CALUDE_total_whales_count_l2532_253230


namespace NUMINAMATH_CALUDE_animal_sightings_l2532_253285

theorem animal_sightings (january : ℕ) (february : ℕ) (march : ℕ) 
  (h1 : february = 3 * january)
  (h2 : march = february / 2)
  (h3 : january + february + march = 143) :
  january = 26 := by
sorry

end NUMINAMATH_CALUDE_animal_sightings_l2532_253285


namespace NUMINAMATH_CALUDE_blue_candy_probability_l2532_253266

/-- The probability of selecting a blue candy from a bag with green, blue, and red candies. -/
theorem blue_candy_probability
  (green : ℕ) (blue : ℕ) (red : ℕ)
  (h_green : green = 5)
  (h_blue : blue = 3)
  (h_red : red = 4) :
  (blue : ℚ) / (green + blue + red) = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_blue_candy_probability_l2532_253266


namespace NUMINAMATH_CALUDE_number_order_l2532_253244

theorem number_order (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_order_l2532_253244


namespace NUMINAMATH_CALUDE_inverse_mod_59_l2532_253254

theorem inverse_mod_59 (h : (17⁻¹ : ZMod 59) = 23) : (42⁻¹ : ZMod 59) = 36 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_59_l2532_253254


namespace NUMINAMATH_CALUDE_segment_endpoint_l2532_253279

/-- Given a line segment from (1, 3) to (x, 7) with length 15 and x < 0, prove x = 1 - √209 -/
theorem segment_endpoint (x : ℝ) : 
  x < 0 → 
  Real.sqrt ((1 - x)^2 + (3 - 7)^2) = 15 → 
  x = 1 - Real.sqrt 209 := by
sorry

end NUMINAMATH_CALUDE_segment_endpoint_l2532_253279


namespace NUMINAMATH_CALUDE_pages_left_to_read_l2532_253243

theorem pages_left_to_read (total_pages read_pages : ℕ) : 
  total_pages = 17 → read_pages = 11 → total_pages - read_pages = 6 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l2532_253243


namespace NUMINAMATH_CALUDE_hyperbola_center_l2532_253296

/-- The hyperbola is defined by the equation 9x^2 - 54x - 16y^2 + 128y - 400 = 0 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 16 * y^2 + 128 * y - 400 = 0

/-- The center of a hyperbola is the point (h, k) where h and k are the coordinates that make
    the equation symmetric about the vertical and horizontal axes passing through (h, k) -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y, hyperbola_equation x y ↔ hyperbola_equation (2*h - x) y ∧ hyperbola_equation x (2*k - y)

/-- The center of the hyperbola defined by 9x^2 - 54x - 16y^2 + 128y - 400 = 0 is (3, 4) -/
theorem hyperbola_center : is_center 3 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2532_253296


namespace NUMINAMATH_CALUDE_original_number_proof_l2532_253293

theorem original_number_proof (h : 204 / 12.75 = 16) : 
  ∃ x : ℝ, x / 1.275 = 1.6 ∧ x = 2.04 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2532_253293


namespace NUMINAMATH_CALUDE_total_wings_count_l2532_253253

theorem total_wings_count (num_planes : ℕ) (wings_per_plane : ℕ) 
  (h1 : num_planes = 54) 
  (h2 : wings_per_plane = 2) : 
  num_planes * wings_per_plane = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_wings_count_l2532_253253


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2532_253247

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l2532_253247


namespace NUMINAMATH_CALUDE_temperature_below_freezing_is_negative_three_l2532_253223

/-- The freezing point of water in degrees Celsius -/
def freezing_point : ℝ := 0

/-- The temperature difference in degrees Celsius -/
def temperature_difference : ℝ := 3

/-- The temperature below freezing point -/
def temperature_below_freezing : ℝ := freezing_point - temperature_difference

theorem temperature_below_freezing_is_negative_three :
  temperature_below_freezing = -3 := by sorry

end NUMINAMATH_CALUDE_temperature_below_freezing_is_negative_three_l2532_253223


namespace NUMINAMATH_CALUDE_largest_positive_integer_binary_op_l2532_253205

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem largest_positive_integer_binary_op :
  ∀ m : ℕ+, m > 4 → binary_op m ≥ 18 ∧ binary_op 4 < 18 :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_integer_binary_op_l2532_253205


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2532_253281

theorem expand_and_simplify (a b : ℝ) : (a + b) * (a - 4 * b) = a^2 - 3*a*b - 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2532_253281


namespace NUMINAMATH_CALUDE_base_two_representation_123_l2532_253215

theorem base_two_representation_123 :
  ∃ (a b c d e f g : Nat),
    123 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 ∧
    a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 0 ∧ f = 1 ∧ g = 1 :=
by sorry

end NUMINAMATH_CALUDE_base_two_representation_123_l2532_253215


namespace NUMINAMATH_CALUDE_quadratic_root_implies_quintic_root_l2532_253265

theorem quadratic_root_implies_quintic_root (r : ℝ) : 
  r^2 - r - 2 = 0 → r^5 - 11*r - 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_quintic_root_l2532_253265


namespace NUMINAMATH_CALUDE_sheet_length_is_48_l2532_253250

/-- Represents the dimensions of a rectangular sheet and the resulting box after cutting squares from corners. -/
structure SheetDimensions where
  length : ℝ
  width : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Theorem stating that given specific dimensions and volume, the length of the sheet must be 48 meters. -/
theorem sheet_length_is_48 (d : SheetDimensions)
  (h1 : d.width = 36)
  (h2 : d.cutSize = 4)
  (h3 : d.boxVolume = 4480)
  (h4 : d.boxVolume = (d.length - 2 * d.cutSize) * (d.width - 2 * d.cutSize) * d.cutSize) :
  d.length = 48 := by
  sorry


end NUMINAMATH_CALUDE_sheet_length_is_48_l2532_253250


namespace NUMINAMATH_CALUDE_train_crossing_time_l2532_253234

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) (time_cross_pole : ℝ) :
  train_length = 900 →
  platform_length = 1050 →
  time_cross_pole = 18 →
  (train_length + platform_length) / (train_length / time_cross_pole) = 39 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l2532_253234


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2532_253209

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of a point being on the ellipse C -/
def on_ellipse_C (p : ℝ × ℝ) : Prop := ellipse_C p.1 p.2

/-- Definition of the line l -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- Definition of a point being on the line l -/
def on_line_l (k m : ℝ) (p : ℝ × ℝ) : Prop := line_l k m p.1 p.2

/-- Definition of the right vertex of the ellipse C -/
def right_vertex : ℝ × ℝ := (2, 0)

/-- Definition of the circle with diameter AB passing through a point -/
def circle_AB_passes_through (A B p : ℝ × ℝ) : Prop :=
  (p.1 - A.1) * (p.1 - B.1) + (p.2 - A.2) * (p.2 - B.2) = 0

/-- The main theorem -/
theorem ellipse_intersection_fixed_point :
  ∀ (k m : ℝ) (A B : ℝ × ℝ),
    on_ellipse_C A ∧ on_ellipse_C B ∧
    on_line_l k m A ∧ on_line_l k m B ∧
    A ≠ right_vertex ∧ B ≠ right_vertex ∧
    circle_AB_passes_through A B right_vertex →
    on_line_l k m (1/2, 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l2532_253209


namespace NUMINAMATH_CALUDE_product_of_numbers_l2532_253224

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 48) : x * y = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2532_253224


namespace NUMINAMATH_CALUDE_cristina_speed_l2532_253278

/-- Cristina's running speed in a race with Nicky -/
theorem cristina_speed (head_start : ℝ) (nicky_speed : ℝ) (catch_up_time : ℝ) 
  (h1 : head_start = 36)
  (h2 : nicky_speed = 3)
  (h3 : catch_up_time = 12) :
  (head_start + nicky_speed * catch_up_time) / catch_up_time = 6 :=
by sorry

end NUMINAMATH_CALUDE_cristina_speed_l2532_253278


namespace NUMINAMATH_CALUDE_min_shoeing_time_for_scenario_l2532_253257

/-- The minimum time needed for blacksmiths to shoe horses -/
def min_shoeing_time (blacksmiths horses : ℕ) (time_per_shoe : ℕ) : ℕ :=
  let total_shoes := horses * 4
  let total_time := total_shoes * time_per_shoe
  (total_time + blacksmiths - 1) / blacksmiths

/-- Theorem stating the minimum time needed for the given scenario -/
theorem min_shoeing_time_for_scenario :
  min_shoeing_time 48 60 5 = 25 := by
sorry

#eval min_shoeing_time 48 60 5

end NUMINAMATH_CALUDE_min_shoeing_time_for_scenario_l2532_253257


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l2532_253242

theorem tax_reduction_theorem (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 100) : 
  (1 - x / 100) * 1.1 = 0.825 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l2532_253242


namespace NUMINAMATH_CALUDE_unique_function_is_lcm_l2532_253289

def satisfies_conditions (f : ℕ → ℕ → ℕ) : Prop :=
  (∀ m n, f m n = f n m) ∧
  (∀ n, f n n = n) ∧
  (∀ m n, n > m → (n - m) * f m n = n * f m (n - m))

theorem unique_function_is_lcm :
  ∀ f : ℕ → ℕ → ℕ, satisfies_conditions f → ∀ m n, f m n = Nat.lcm m n := by
  sorry

end NUMINAMATH_CALUDE_unique_function_is_lcm_l2532_253289


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l2532_253248

/-- Given information about blanket purchases and average price, 
    prove the unknown rate of two blankets. -/
theorem unknown_blanket_rate 
  (blanket_count_1 blanket_count_2 blanket_count_unknown : ℕ)
  (price_1 price_2 average_price : ℚ)
  (h1 : blanket_count_1 = 3)
  (h2 : blanket_count_2 = 3)
  (h3 : blanket_count_unknown = 2)
  (h4 : price_1 = 100)
  (h5 : price_2 = 150)
  (h6 : average_price = 150)
  (h7 : (blanket_count_1 * price_1 + blanket_count_2 * price_2 + 
         blanket_count_unknown * unknown_rate) / 
        (blanket_count_1 + blanket_count_2 + blanket_count_unknown) = 
        average_price) :
  unknown_rate = 225 := by
  sorry


end NUMINAMATH_CALUDE_unknown_blanket_rate_l2532_253248


namespace NUMINAMATH_CALUDE_keith_stored_bales_l2532_253282

/-- The number of bales Keith stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Theorem: Keith stored 67 bales in the barn -/
theorem keith_stored_bales :
  bales_stored 22 89 = 67 := by
  sorry

end NUMINAMATH_CALUDE_keith_stored_bales_l2532_253282


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_range_l2532_253251

theorem quadratic_always_nonnegative_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_range_l2532_253251


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2532_253259

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the containment relation of a line in a plane
variable (contained_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_relationship
  (m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane m β)
  (h2 : perpendicular_plane_plane α β) :
  parallel_line_plane m α ∨ contained_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2532_253259


namespace NUMINAMATH_CALUDE_a_zero_iff_multiple_of_ten_sum_a_1_to_2005_l2532_253291

def a (n : ℕ+) : ℕ :=
  (7 * n.val) % 10

theorem a_zero_iff_multiple_of_ten (n : ℕ+) : a n = 0 ↔ 10 ∣ n.val := by
  sorry

theorem sum_a_1_to_2005 : (Finset.range 2005).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩) = 9025 := by
  sorry

end NUMINAMATH_CALUDE_a_zero_iff_multiple_of_ten_sum_a_1_to_2005_l2532_253291


namespace NUMINAMATH_CALUDE_two_sided_icing_count_l2532_253288

/-- Represents a cubic cake with icing on specific faces -/
structure CubeCake where
  size : Nat
  has_top_icing : Bool
  has_bottom_icing : Bool
  has_side_icing : Bool
  has_middle_layer_icing : Bool

/-- Counts the number of 1×1×1 sub-cubes with icing on exactly two sides -/
def count_two_sided_icing (cake : CubeCake) : Nat :=
  sorry

/-- The main theorem stating that a 5×5×5 cake with specific icing has 24 sub-cubes with icing on two sides -/
theorem two_sided_icing_count :
  let cake : CubeCake := {
    size := 5,
    has_top_icing := true,
    has_bottom_icing := false,
    has_side_icing := true,
    has_middle_layer_icing := true
  }
  count_two_sided_icing cake = 24 := by sorry

end NUMINAMATH_CALUDE_two_sided_icing_count_l2532_253288


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2532_253298

open Set

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,3}
def B : Set Nat := {3,5}

theorem intersection_complement_equality : A ∩ (U \ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2532_253298


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l2532_253252

/-- The value of m for which the parabola y = x^2 + 2x + 3 and the hyperbola y^2 - mx^2 = 5 are tangent -/
def tangency_m : ℝ := -26

/-- The equation of the parabola -/
def parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

/-- The equation of the hyperbola -/
def hyperbola (x y m : ℝ) : Prop := y^2 - m*x^2 = 5

/-- Theorem stating that the parabola and hyperbola are tangent when m = -26 -/
theorem parabola_hyperbola_tangency :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = parabola x₀ ∧ 
    hyperbola x₀ y₀ tangency_m ∧
    ∀ (x y : ℝ), y = parabola x → hyperbola x y tangency_m → x = x₀ ∧ y = y₀ :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l2532_253252


namespace NUMINAMATH_CALUDE_equilateral_triangle_isosceles_points_l2532_253203

/-- An equilateral triangle in a 2D plane -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : sorry

/-- A point in the 2D plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a triangle is isosceles -/
def is_isosceles (P Q R : Point) : Prop := sorry

/-- Predicate to check if a point is inside a triangle -/
def is_inside (P : Point) (triangle : EquilateralTriangle) : Prop := sorry

theorem equilateral_triangle_isosceles_points (ABC : EquilateralTriangle) :
  ∃ (points : Finset Point),
    points.card = 10 ∧
    ∀ P ∈ points,
      is_inside P ABC ∧
      is_isosceles P ABC.B ABC.C ∧
      is_isosceles P ABC.A ABC.B ∧
      is_isosceles P ABC.A ABC.C :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_isosceles_points_l2532_253203


namespace NUMINAMATH_CALUDE_expression_defined_iff_in_interval_l2532_253232

/-- The expression log(x-2) / sqrt(5-x) is defined if and only if x is in the open interval (2, 5) -/
theorem expression_defined_iff_in_interval (x : ℝ) :
  (∃ y : ℝ, y = (Real.log (x - 2)) / Real.sqrt (5 - x)) ↔ 2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_defined_iff_in_interval_l2532_253232
