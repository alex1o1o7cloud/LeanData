import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l922_92218

theorem inequality_proof (a b c : ℝ) (h : a * b + b * c + c * a = 1) :
  |a - b| / |1 + c^2| + |b - c| / |1 + a^2| ≥ |c - a| / |1 + b^2| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l922_92218


namespace NUMINAMATH_CALUDE_range_of_m_l922_92212

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - m| < 1 → x^2 - 8*x + 12 < 0) ∧ 
  (∃ x, x^2 - 8*x + 12 < 0 ∧ |x - m| ≥ 1)) →
  (3 ≤ m ∧ m ≤ 5) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l922_92212


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l922_92263

/-- There exist infinitely many ordered quadruples (x, y, z, w) of real numbers
    satisfying the given conditions. -/
theorem infinitely_many_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ × ℝ)), Set.Infinite S ∧
    ∀ (x y z w : ℝ), (x, y, z, w) ∈ S →
      (x + y = 3 ∧ x * y - z^2 = w ∧ w + z = 4) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l922_92263


namespace NUMINAMATH_CALUDE_percentage_spent_l922_92226

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 5000)
  (h2 : remaining_amount = 3500) :
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_spent_l922_92226


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_257_l922_92267

theorem modular_inverse_of_3_mod_257 : ∃ x : ℕ, x < 257 ∧ (3 * x) % 257 = 1 :=
  by
    use 86
    sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_257_l922_92267


namespace NUMINAMATH_CALUDE_divisibility_condition_l922_92206

def is_divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

theorem divisibility_condition (a b : ℕ) : 
  (a ≤ 9 ∧ b ≤ 9) →
  (is_divisible_by (62684 * 10 + a * 10 + b) 8 ∧ 
   is_divisible_by (62684 * 10 + a * 10 + b) 5) →
  (b = 0 ∧ (a = 0 ∨ a = 8)) := by
  sorry

#check divisibility_condition

end NUMINAMATH_CALUDE_divisibility_condition_l922_92206


namespace NUMINAMATH_CALUDE_alyssa_allowance_proof_l922_92283

/-- Alyssa's weekly allowance -/
def weekly_allowance : ℝ := 240

theorem alyssa_allowance_proof :
  ∃ (A : ℝ),
    A > 0 ∧
    A / 2 + A / 5 + A / 4 + 12 = A ∧
    A = weekly_allowance :=
by sorry

end NUMINAMATH_CALUDE_alyssa_allowance_proof_l922_92283


namespace NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l922_92291

theorem sandwich_non_condiment_percentage
  (total_weight : ℝ)
  (condiment_weight : ℝ)
  (h1 : total_weight = 150)
  (h2 : condiment_weight = 45) :
  (total_weight - condiment_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_non_condiment_percentage_l922_92291


namespace NUMINAMATH_CALUDE_paco_cookies_theorem_l922_92240

/-- Represents the number of cookies Paco has after all actions --/
def remaining_cookies (initial_salty initial_sweet initial_chocolate : ℕ)
  (eaten_sweet eaten_salty : ℕ) (given_chocolate received_chocolate : ℕ) :
  ℕ × ℕ × ℕ :=
  let remaining_sweet := initial_sweet - eaten_sweet
  let remaining_salty := initial_salty - eaten_salty
  let remaining_chocolate := initial_chocolate - given_chocolate + received_chocolate
  (remaining_sweet, remaining_salty, remaining_chocolate)

/-- Theorem stating the final number of cookies Paco has --/
theorem paco_cookies_theorem :
  remaining_cookies 97 34 45 15 56 22 7 = (19, 41, 30) := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_theorem_l922_92240


namespace NUMINAMATH_CALUDE_machine_quality_l922_92249

/-- Represents a packaging machine --/
structure PackagingMachine where
  weight : Real → Real  -- Random variable representing packaging weight

/-- Defines the expected value of a random variable --/
def expectedValue (X : Real → Real) : Real :=
  sorry

/-- Defines the variance of a random variable --/
def variance (X : Real → Real) : Real :=
  sorry

/-- Determines if a packaging machine has better quality --/
def betterQuality (m1 m2 : PackagingMachine) : Prop :=
  expectedValue m1.weight = expectedValue m2.weight ∧
  variance m1.weight > variance m2.weight →
  sorry  -- This represents that m2 has better quality

/-- Theorem stating which machine has better quality --/
theorem machine_quality (A B : PackagingMachine) :
  betterQuality A B → sorry  -- This represents that B has better quality
:= by sorry

end NUMINAMATH_CALUDE_machine_quality_l922_92249


namespace NUMINAMATH_CALUDE_polynomial_simplification_l922_92239

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5 * x - 7) - (x^2 + 9 * x - 3) = x^2 - 4 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l922_92239


namespace NUMINAMATH_CALUDE_more_cats_than_dogs_l922_92296

theorem more_cats_than_dogs : 
  let num_dogs : ℕ := 9
  let num_cats : ℕ := 23
  num_cats - num_dogs = 14 := by sorry

end NUMINAMATH_CALUDE_more_cats_than_dogs_l922_92296


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l922_92201

theorem quadratic_form_equivalence :
  ∀ (x : ℝ), 3 * x^2 + 9 * x + 20 = 3 * (x + 3/2)^2 + 53/4 ∧
  ∃ (h : ℝ), h = -3/2 ∧ ∀ (x : ℝ), 3 * x^2 + 9 * x + 20 = 3 * (x - h)^2 + 53/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l922_92201


namespace NUMINAMATH_CALUDE_special_function_properties_l922_92205

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (f (1/3) = 1) ∧
  (∀ x : ℝ, x > 0 → f x > 0)

theorem special_function_properties (f : ℝ → ℝ) (h : special_function f) :
  (f 0 = 0) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, x < -2/3 → f x + f (2 + x) < 2) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l922_92205


namespace NUMINAMATH_CALUDE_binary_to_decimal_11001001_l922_92213

/-- Converts a list of binary digits to its decimal equivalent -/
def binaryToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 2^(digits.length - 1 - i)) 0

/-- The binary representation of the number -/
def binaryNumber : List Nat := [1, 1, 0, 0, 1, 0, 0, 1]

theorem binary_to_decimal_11001001 :
  binaryToDecimal binaryNumber = 201 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_11001001_l922_92213


namespace NUMINAMATH_CALUDE_min_value_K_l922_92287

theorem min_value_K (α β γ : ℝ) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  (α + 3*γ)/(α + 2*β + γ) + 4*β/(α + β + 2*γ) - 8*γ/(α + β + 3*γ) ≥ 2/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_K_l922_92287


namespace NUMINAMATH_CALUDE_cos_215_minus_1_l922_92200

theorem cos_215_minus_1 : Real.cos (215 * π / 180) - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_215_minus_1_l922_92200


namespace NUMINAMATH_CALUDE_min_overs_for_max_wickets_l922_92251

/-- Represents the maximum number of wickets a bowler can take in one over -/
def max_wickets_per_over : ℕ := 3

/-- Represents the maximum number of wickets a bowler can take in the innings -/
def max_wickets_in_innings : ℕ := 10

/-- Theorem stating the minimum number of overs required to potentially take the maximum wickets -/
theorem min_overs_for_max_wickets :
  ∃ (overs : ℕ), overs * max_wickets_per_over ≥ max_wickets_in_innings ∧
  ∀ (n : ℕ), n < overs → n * max_wickets_per_over < max_wickets_in_innings :=
by sorry

end NUMINAMATH_CALUDE_min_overs_for_max_wickets_l922_92251


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l922_92232

theorem average_of_six_numbers 
  (total_average : ℝ) 
  (second_pair_average : ℝ) 
  (third_pair_average : ℝ) 
  (h1 : total_average = 3.9) 
  (h2 : second_pair_average = 3.85) 
  (h3 : third_pair_average = 4.45) : 
  ∃ first_pair_average : ℝ, first_pair_average = 3.4 ∧ 
  6 * total_average = 2 * first_pair_average + 2 * second_pair_average + 2 * third_pair_average :=
sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l922_92232


namespace NUMINAMATH_CALUDE_farmer_wheat_harvest_l922_92273

theorem farmer_wheat_harvest (estimated_harvest additional_harvest : ℕ) 
  (h1 : estimated_harvest = 213489)
  (h2 : additional_harvest = 13257) :
  estimated_harvest + additional_harvest = 226746 := by
  sorry

end NUMINAMATH_CALUDE_farmer_wheat_harvest_l922_92273


namespace NUMINAMATH_CALUDE_vase_capacity_l922_92234

/-- The number of flowers each vase can hold -/
def flowers_per_vase (carnations : ℕ) (roses : ℕ) (vases : ℕ) : ℕ :=
  (carnations + roses) / vases

/-- Proof that each vase can hold 6 flowers -/
theorem vase_capacity : flowers_per_vase 7 47 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_vase_capacity_l922_92234


namespace NUMINAMATH_CALUDE_probability_no_consecutive_ones_l922_92246

/-- Represents a binary sequence -/
def BinarySequence := List Bool

/-- Checks if a binary sequence contains two consecutive 1s -/
def hasConsecutiveOnes : BinarySequence → Bool :=
  fun seq => sorry

/-- Generates all valid 12-digit binary sequences starting with 1 -/
def generateSequences : List BinarySequence :=
  sorry

/-- Counts the number of sequences without consecutive 1s -/
def countValidSequences : Nat :=
  sorry

/-- The total number of possible 12-digit sequences starting with 1 -/
def totalSequences : Nat := 2^11

theorem probability_no_consecutive_ones :
  (countValidSequences : ℚ) / totalSequences = 233 / 2048 :=
sorry

end NUMINAMATH_CALUDE_probability_no_consecutive_ones_l922_92246


namespace NUMINAMATH_CALUDE_parabola_directrix_l922_92261

/-- The equation of the directrix of the parabola y = -4x^2 - 16x + 1 -/
theorem parabola_directrix : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = -4 * x^2 - 16 * x + 1 ↔ y = a * (x - b)^2 + c) →
    (∃ d : ℝ, d = 273 / 16 ∧ 
      ∀ x y : ℝ, y = d → 
        (x - b)^2 + (y - c)^2 = (y - (c - 1 / (4 * |a|)))^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l922_92261


namespace NUMINAMATH_CALUDE_lauren_change_l922_92258

/-- Represents the grocery items with their prices and discounts --/
structure GroceryItems where
  hamburger_meat_price : ℝ
  hamburger_meat_discount : ℝ
  hamburger_buns_price : ℝ
  lettuce_price : ℝ
  tomato_price : ℝ
  tomato_weight : ℝ
  onion_price : ℝ
  onion_weight : ℝ
  pickles_price : ℝ
  pickles_coupon : ℝ
  potatoes_price : ℝ
  soda_price : ℝ
  soda_discount : ℝ

/-- Calculates the total cost of the grocery items including tax --/
def calculateTotalCost (items : GroceryItems) (tax_rate : ℝ) : ℝ :=
  let hamburger_meat_cost := 2 * items.hamburger_meat_price * (1 - items.hamburger_meat_discount)
  let hamburger_buns_cost := items.hamburger_buns_price
  let tomato_cost := items.tomato_price * items.tomato_weight
  let onion_cost := items.onion_price * items.onion_weight
  let pickles_cost := items.pickles_price - items.pickles_coupon
  let soda_cost := items.soda_price * (1 - items.soda_discount)
  let subtotal := hamburger_meat_cost + hamburger_buns_cost + items.lettuce_price + 
                  tomato_cost + onion_cost + pickles_cost + items.potatoes_price + soda_cost
  subtotal * (1 + tax_rate)

/-- Proves that Lauren's change from a $50 bill is $24.67 --/
theorem lauren_change (items : GroceryItems) (tax_rate : ℝ) :
  items.hamburger_meat_price = 3.5 →
  items.hamburger_meat_discount = 0.15 →
  items.hamburger_buns_price = 1.5 →
  items.lettuce_price = 1 →
  items.tomato_price = 2 →
  items.tomato_weight = 1.5 →
  items.onion_price = 0.75 →
  items.onion_weight = 0.5 →
  items.pickles_price = 2.5 →
  items.pickles_coupon = 1 →
  items.potatoes_price = 4 →
  items.soda_price = 5.99 →
  items.soda_discount = 0.07 →
  tax_rate = 0.06 →
  50 - calculateTotalCost items tax_rate = 24.67 := by
  sorry

end NUMINAMATH_CALUDE_lauren_change_l922_92258


namespace NUMINAMATH_CALUDE_ab_value_l922_92229

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l922_92229


namespace NUMINAMATH_CALUDE_custom_operation_theorem_l922_92235

def custom_operation (M N : Set ℕ) : Set ℕ :=
  {x | x ∈ M ∨ x ∈ N ∧ x ∉ M ∩ N}

def M : Set ℕ := {0, 2, 4, 6, 8, 10}
def N : Set ℕ := {0, 3, 6, 9, 12, 15}

theorem custom_operation_theorem :
  custom_operation (custom_operation M N) M = N := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_theorem_l922_92235


namespace NUMINAMATH_CALUDE_least_tiles_required_l922_92262

/-- The length of the room in centimeters -/
def room_length : ℕ := 1517

/-- The breadth of the room in centimeters -/
def room_breadth : ℕ := 902

/-- The greatest common divisor of the room length and breadth -/
def tile_side : ℕ := Nat.gcd room_length room_breadth

/-- The area of the room in square centimeters -/
def room_area : ℕ := room_length * room_breadth

/-- The area of a single tile in square centimeters -/
def tile_area : ℕ := tile_side * tile_side

/-- The number of tiles required to pave the room -/
def num_tiles : ℕ := (room_area + tile_area - 1) / tile_area

theorem least_tiles_required :
  num_tiles = 814 :=
sorry

end NUMINAMATH_CALUDE_least_tiles_required_l922_92262


namespace NUMINAMATH_CALUDE_multiply_and_add_equality_l922_92215

theorem multiply_and_add_equality : 45 * 28 + 72 * 45 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_equality_l922_92215


namespace NUMINAMATH_CALUDE_golden_hyperbola_eccentricity_l922_92252

theorem golden_hyperbola_eccentricity :
  ∀ e : ℝ, e > 1 → e^2 - e = 1 → e = (Real.sqrt 5 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_hyperbola_eccentricity_l922_92252


namespace NUMINAMATH_CALUDE_range_of_m_l922_92217

def has_two_distinct_negative_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

def satisfies_conditions (m : ℝ) : Prop :=
  (has_two_distinct_negative_roots m ∨ has_no_real_roots m) ∧
  ¬(has_two_distinct_negative_roots m ∧ has_no_real_roots m)

theorem range_of_m : 
  {m : ℝ | satisfies_conditions m} = {m : ℝ | 1 < m ∧ m ≤ 2 ∨ 3 ≤ m} :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l922_92217


namespace NUMINAMATH_CALUDE_harmonic_mean_counterexample_l922_92286

theorem harmonic_mean_counterexample :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (2 / (1/a + 1/b) < Real.sqrt (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_counterexample_l922_92286


namespace NUMINAMATH_CALUDE_distance_walked_l922_92299

/-- Proves that the distance walked is 18 miles given specific conditions on speed changes and time. -/
theorem distance_walked (speed : ℝ) (time : ℝ) : 
  speed > 0 → 
  time > 0 → 
  (speed + 1) * (3 * time / 4) = speed * time → 
  (speed - 1) * (time + 3) = speed * time → 
  speed * time = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l922_92299


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_l922_92233

/-- Given a point P(-3, 4) on the terminal side of angle α, prove that sin α + 2cos α = -2/5 -/
theorem sin_plus_two_cos (α : Real) (P : ℝ × ℝ) (h : P = (-3, 4)) : 
  Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_l922_92233


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l922_92219

def polynomial (x b₂ b₁ : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial x b₂ b₁ = 0} ⊆ possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l922_92219


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l922_92290

theorem repeating_decimal_subtraction : 
  ∃ (x y : ℚ), (∀ n : ℕ, (10 * x - x.floor) * 10^n % 10 = 4) ∧ 
               (∀ n : ℕ, (10 * y - y.floor) * 10^n % 10 = 6) ∧ 
               (x - y = -2/9) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l922_92290


namespace NUMINAMATH_CALUDE_prove_average_marks_l922_92259

def average_marks (M P C : ℝ) : Prop :=
  M + P = 40 ∧ C = P + 20 → (M + C) / 2 = 30

theorem prove_average_marks :
  ∀ M P C : ℝ, average_marks M P C :=
by
  sorry

end NUMINAMATH_CALUDE_prove_average_marks_l922_92259


namespace NUMINAMATH_CALUDE_externally_tangent_circles_l922_92214

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The equation of circle C₁ -/
def C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- The equation of circle C₂ -/
def C2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + m = 0

theorem externally_tangent_circles (m : ℝ) :
  (∃ c1 : ℝ × ℝ, ∃ r1 : ℝ, ∀ x y : ℝ, C1 x y ↔ (x - c1.1)^2 + (y - c1.2)^2 = r1^2) →
  (∃ c2 : ℝ × ℝ, ∃ r2 : ℝ, ∀ x y : ℝ, C2 x y m ↔ (x - c2.1)^2 + (y - c2.2)^2 = r2^2) →
  (∃ c1 c2 : ℝ × ℝ, ∃ r1 r2 : ℝ, externally_tangent c1 c2 r1 r2) →
  m = -3 := by
  sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_l922_92214


namespace NUMINAMATH_CALUDE_unit_vectors_parallel_to_a_l922_92203

def vector_a : ℝ × ℝ := (12, 5)

theorem unit_vectors_parallel_to_a :
  let magnitude := Real.sqrt (vector_a.1^2 + vector_a.2^2)
  let unit_vector := (vector_a.1 / magnitude, vector_a.2 / magnitude)
  (unit_vector = (12/13, 5/13) ∨ unit_vector = (-12/13, -5/13)) ∧
  (∀ v : ℝ × ℝ, (v.1^2 + v.2^2 = 1 ∧ ∃ k : ℝ, v = (k * vector_a.1, k * vector_a.2)) →
    (v = (12/13, 5/13) ∨ v = (-12/13, -5/13))) :=
by sorry

end NUMINAMATH_CALUDE_unit_vectors_parallel_to_a_l922_92203


namespace NUMINAMATH_CALUDE_expected_bounces_l922_92253

/-- The expected number of bounces for a ball on a rectangular billiard table -/
theorem expected_bounces (table_length table_width ball_travel : ℝ) 
  (h_length : table_length = 3)
  (h_width : table_width = 1)
  (h_travel : ball_travel = 2) :
  ∃ (E : ℝ), E = 1 + (2 / Real.pi) * (Real.arccos (3/4) + Real.arccos (1/4) - Real.arcsin (3/4)) :=
by sorry

end NUMINAMATH_CALUDE_expected_bounces_l922_92253


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l922_92281

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 11 = 50) 
  (h_a4 : a 4 = 13) : 
  a 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l922_92281


namespace NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l922_92230

theorem consecutive_product_not_perfect_power :
  ∀ x y : ℤ, ∀ n : ℕ, n > 1 → x * (x + 1) ≠ y^n := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_perfect_power_l922_92230


namespace NUMINAMATH_CALUDE_share_ratio_l922_92294

theorem share_ratio (total : ℚ) (a b c : ℚ) : 
  total = 510 →
  b = (1/4) * c →
  a + b + c = total →
  a = 360 →
  a / b = 12 := by
sorry

end NUMINAMATH_CALUDE_share_ratio_l922_92294


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l922_92248

def is_geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_sequence_condition (a b c : ℝ) :
  (is_geometric_sequence a b c → b^2 = a*c) ∧
  ¬(b^2 = a*c → is_geometric_sequence a b c) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l922_92248


namespace NUMINAMATH_CALUDE_triangular_projections_imply_triangular_pyramid_l922_92276

/-- Represents the shape of a projection in an orthographic view -/
inductive Projection
  | Triangular
  | Circular
  | Rectangular
  | Trapezoidal

/-- Represents a geometric solid -/
inductive GeometricSolid
  | Cone
  | TriangularPyramid
  | TriangularPrism
  | FrustumOfPyramid

/-- Represents the orthographic views of a solid -/
structure OrthographicViews where
  front : Projection
  top : Projection
  side : Projection

/-- Determines if a set of orthographic views corresponds to a triangular pyramid -/
def isTriangularPyramid (views : OrthographicViews) : Prop :=
  views.front = Projection.Triangular ∧
  views.top = Projection.Triangular ∧
  views.side = Projection.Triangular

theorem triangular_projections_imply_triangular_pyramid (views : OrthographicViews) :
  isTriangularPyramid views → GeometricSolid.TriangularPyramid = 
    match views with
    | ⟨Projection.Triangular, Projection.Triangular, Projection.Triangular⟩ => GeometricSolid.TriangularPyramid
    | _ => GeometricSolid.Cone  -- This is just a placeholder for other cases
    :=
  sorry

end NUMINAMATH_CALUDE_triangular_projections_imply_triangular_pyramid_l922_92276


namespace NUMINAMATH_CALUDE_lucy_max_notebooks_l922_92289

/-- The amount of money Lucy has in cents -/
def lucys_money : ℕ := 2550

/-- The cost of each notebook in cents -/
def notebook_cost : ℕ := 240

/-- The maximum number of notebooks Lucy can buy -/
def max_notebooks : ℕ := lucys_money / notebook_cost

theorem lucy_max_notebooks :
  max_notebooks = 10 ∧
  max_notebooks * notebook_cost ≤ lucys_money ∧
  (max_notebooks + 1) * notebook_cost > lucys_money :=
by sorry

end NUMINAMATH_CALUDE_lucy_max_notebooks_l922_92289


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l922_92208

theorem square_plus_reciprocal_square (x : ℝ) (h : x + (1/x) = 4) : x^2 + (1/x^2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l922_92208


namespace NUMINAMATH_CALUDE_april_coffee_cost_l922_92295

/-- The number of coffees Jon buys per day -/
def coffees_per_day : ℕ := 2

/-- The cost of one coffee in dollars -/
def cost_per_coffee : ℕ := 2

/-- The number of days in April -/
def days_in_april : ℕ := 30

/-- The total cost of coffee for Jon in April -/
def total_cost : ℕ := coffees_per_day * cost_per_coffee * days_in_april

theorem april_coffee_cost : total_cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_april_coffee_cost_l922_92295


namespace NUMINAMATH_CALUDE_janes_earnings_is_75_l922_92285

/-- The amount of money Jane earned for planting flower bulbs -/
def janes_earnings : ℚ :=
  let price_per_bulb : ℚ := 1/2
  let tulip_bulbs : ℕ := 20
  let iris_bulbs : ℕ := tulip_bulbs / 2
  let daffodil_bulbs : ℕ := 30
  let crocus_bulbs : ℕ := daffodil_bulbs * 3
  let total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  (total_bulbs : ℚ) * price_per_bulb

/-- Theorem stating that Jane earned $75 for planting flower bulbs -/
theorem janes_earnings_is_75 : janes_earnings = 75 := by
  sorry

end NUMINAMATH_CALUDE_janes_earnings_is_75_l922_92285


namespace NUMINAMATH_CALUDE_intersection_area_zero_l922_92265

-- Define the triangle vertices
def P : ℝ × ℝ := (3, -2)
def Q : ℝ × ℝ := (5, 4)
def R : ℝ × ℝ := (1, 1)

-- Define the reflection function across y = 0
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Define the triangle and its reflection
def triangle : Set (ℝ × ℝ) := {p | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ 
  p = (a * P.1 + b * Q.1 + c * R.1, a * P.2 + b * Q.2 + c * R.2)}

def reflectedTriangle : Set (ℝ × ℝ) := {p | ∃ q ∈ triangle, p = reflect q}

-- State the theorem
theorem intersection_area_zero : 
  MeasureTheory.volume (triangle ∩ reflectedTriangle) = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_area_zero_l922_92265


namespace NUMINAMATH_CALUDE_right_triangle_median_geometric_mean_l922_92278

theorem right_triangle_median_geometric_mean (c : ℝ) (h : c > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    c^2 = a^2 + b^2 ∧
    (c / 2)^2 = a * b ∧
    a + b = c * Real.sqrt (3 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_median_geometric_mean_l922_92278


namespace NUMINAMATH_CALUDE_eggs_left_l922_92260

theorem eggs_left (initial_eggs : ℕ) (taken_eggs : ℕ) (h1 : initial_eggs = 47) (h2 : taken_eggs = 5) :
  initial_eggs - taken_eggs = 42 := by
sorry

end NUMINAMATH_CALUDE_eggs_left_l922_92260


namespace NUMINAMATH_CALUDE_intersection_point_d_equals_two_l922_92211

/-- A function f(x) = 4x + c where c is an integer -/
def f (c : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + c

/-- The inverse of f -/
noncomputable def f_inv (c : ℤ) : ℝ → ℝ := λ x ↦ (x - c) / 4

theorem intersection_point_d_equals_two (c d : ℤ) :
  f c 2 = d ∧ f_inv c d = 2 → d = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_point_d_equals_two_l922_92211


namespace NUMINAMATH_CALUDE_notebook_cost_l922_92227

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : 
  total_students = 30 →
  total_cost = 1584 →
  ∃ (students_bought notebooks_per_student cost_per_notebook : Nat),
    students_bought = 20 ∧
    students_bought * notebooks_per_student * cost_per_notebook = total_cost ∧
    cost_per_notebook ≥ notebooks_per_student ∧
    cost_per_notebook = 11 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l922_92227


namespace NUMINAMATH_CALUDE_trays_needed_to_refill_l922_92204

/-- The number of ice cubes Dylan used in his glass -/
def dylan_glass_cubes : ℕ := 8

/-- The number of ice cubes used per glass for lemonade -/
def lemonade_glass_cubes : ℕ := 2 * dylan_glass_cubes

/-- The total number of glasses served (including Dylan's) -/
def total_glasses : ℕ := 5 + 1

/-- The number of spaces in each ice cube tray -/
def tray_spaces : ℕ := 14

/-- The fraction of total ice cubes used -/
def fraction_used : ℚ := 4/5

/-- The total number of ice cubes used -/
def total_used : ℕ := dylan_glass_cubes + lemonade_glass_cubes * total_glasses

/-- The initial total number of ice cubes -/
def initial_total : ℚ := (total_used : ℚ) / fraction_used

theorem trays_needed_to_refill : 
  ⌈initial_total / tray_spaces⌉ = 10 := by sorry

end NUMINAMATH_CALUDE_trays_needed_to_refill_l922_92204


namespace NUMINAMATH_CALUDE_rose_difference_l922_92275

theorem rose_difference (total : ℕ) (red_fraction : ℚ) (yellow_fraction : ℚ)
  (h_total : total = 48)
  (h_red : red_fraction = 3/8)
  (h_yellow : yellow_fraction = 5/16) :
  ↑total * red_fraction - ↑total * yellow_fraction = 3 :=
by sorry

end NUMINAMATH_CALUDE_rose_difference_l922_92275


namespace NUMINAMATH_CALUDE_value_of_expression_l922_92224

theorem value_of_expression (x : ℝ) (h : 7 * x + 6 = 3 * x - 18) : 
  3 * (2 * x + 4) = -24 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l922_92224


namespace NUMINAMATH_CALUDE_yarn_parts_count_l922_92243

/-- Given a yarn of 10 meters cut into equal parts, where 3 parts equal 6 meters,
    prove that the yarn was cut into 5 parts. -/
theorem yarn_parts_count (total_length : ℝ) (used_parts : ℕ) (used_length : ℝ) :
  total_length = 10 →
  used_parts = 3 →
  used_length = 6 →
  (total_length / (used_length / used_parts : ℝ) : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_yarn_parts_count_l922_92243


namespace NUMINAMATH_CALUDE_petya_counterexample_l922_92228

theorem petya_counterexample : ∃ (a b : ℕ), 
  (a^5 % b^2 = 0) ∧ (a^2 % b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_petya_counterexample_l922_92228


namespace NUMINAMATH_CALUDE_solve_equation_l922_92236

theorem solve_equation : ∃ x : ℝ, 3 * x + 36 = 48 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l922_92236


namespace NUMINAMATH_CALUDE_jinas_mascots_l922_92268

/-- The number of mascots Jina has -/
def total_mascots (initial_teddies : ℕ) (bunny_multiplier : ℕ) (koalas : ℕ) (additional_teddies_per_bunny : ℕ) : ℕ :=
  let bunnies := initial_teddies * bunny_multiplier
  let additional_teddies := bunnies * additional_teddies_per_bunny
  initial_teddies + bunnies + koalas + additional_teddies

/-- Theorem stating the total number of mascots Jina has -/
theorem jinas_mascots :
  total_mascots 5 3 1 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_jinas_mascots_l922_92268


namespace NUMINAMATH_CALUDE_fourth_root_squared_l922_92292

theorem fourth_root_squared (x : ℝ) : (x^(1/4))^2 = 16 → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_squared_l922_92292


namespace NUMINAMATH_CALUDE_total_path_satisfies_conditions_l922_92274

/-- The total length of Gyeongyeon's travel path --/
def total_path : ℝ := 2200

/-- Gyeongyeon's travel segments --/
structure TravelSegments where
  bicycle : ℝ
  first_walk : ℝ
  bus : ℝ
  final_walk : ℝ

/-- Conditions of Gyeongyeon's travel --/
def travel_conditions (d : ℝ) : Prop :=
  ∃ (segments : TravelSegments),
    segments.bicycle = d / 2 ∧
    segments.first_walk = 300 ∧
    segments.bus = (d / 2 - 300) / 2 ∧
    segments.final_walk = 400 ∧
    segments.bicycle + segments.first_walk + segments.bus + segments.final_walk = d

/-- Theorem stating that the total path length satisfies the travel conditions --/
theorem total_path_satisfies_conditions : travel_conditions total_path := by
  sorry


end NUMINAMATH_CALUDE_total_path_satisfies_conditions_l922_92274


namespace NUMINAMATH_CALUDE_expression_equality_l922_92244

theorem expression_equality : 40 + 5 * 12 / (180 / 3) = 41 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l922_92244


namespace NUMINAMATH_CALUDE_initial_sodium_chloride_percentage_l922_92241

/-- Proves that given a tank with 10,000 gallons of solution, if 5,500 gallons of water evaporate
    and the remaining solution is 11.11111111111111% sodium chloride, then the initial percentage
    of sodium chloride was 5%. -/
theorem initial_sodium_chloride_percentage
  (initial_volume : ℝ)
  (evaporated_volume : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 10000)
  (h2 : evaporated_volume = 5500)
  (h3 : final_percentage = 11.11111111111111)
  (h4 : final_percentage = (100 * initial_volume * (initial_percentage / 100)) /
                           (initial_volume - evaporated_volume)) :
  initial_percentage = 5 :=
by sorry

end NUMINAMATH_CALUDE_initial_sodium_chloride_percentage_l922_92241


namespace NUMINAMATH_CALUDE_triangle_angle_C_l922_92298

theorem triangle_angle_C (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  3 * Real.sin A + 4 * Real.cos B = 6 →
  4 * Real.sin B + 3 * Real.cos A = 1 →
  C = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l922_92298


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l922_92254

theorem complex_modulus_equality (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) :
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l922_92254


namespace NUMINAMATH_CALUDE_tim_bought_three_goats_l922_92269

/-- Proves that Tim bought 3 goats given the conditions of the problem -/
theorem tim_bought_three_goats
  (goat_cost : ℕ)
  (llama_count : ℕ → ℕ)
  (llama_cost : ℕ → ℕ)
  (total_spent : ℕ)
  (h1 : goat_cost = 400)
  (h2 : ∀ g, llama_count g = 2 * g)
  (h3 : ∀ g, llama_cost g = goat_cost + goat_cost / 2)
  (h4 : total_spent = 4800)
  (h5 : ∀ g, total_spent = g * goat_cost + llama_count g * llama_cost g) :
  ∃ g : ℕ, g = 3 ∧ total_spent = g * goat_cost + llama_count g * llama_cost g :=
sorry

end NUMINAMATH_CALUDE_tim_bought_three_goats_l922_92269


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l922_92256

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) :
  ∃ (q' r' : ℕ+), Nat.gcd q'.val r'.val = 10 ∧
    ∀ (q'' r'' : ℕ+), Nat.gcd q''.val r''.val < 10 →
      ¬(Nat.gcd p q''.val = 210 ∧ Nat.gcd p r''.val = 770) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l922_92256


namespace NUMINAMATH_CALUDE_smallest_covering_set_smallest_n_is_five_l922_92237

theorem smallest_covering_set (n : ℕ) : Prop :=
  ∃ (k a : Fin n → ℕ),
    (∀ i j : Fin n, i < j → 1 < k i ∧ k i < k j) ∧
    (∀ N : ℤ, ∃ i : Fin n, (k i : ℤ) ∣ (N - (a i : ℤ)))

theorem smallest_n_is_five :
  (∃ n : ℕ, smallest_covering_set n) ∧
  (∀ m : ℕ, smallest_covering_set m → m ≥ 5) ∧
  smallest_covering_set 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_set_smallest_n_is_five_l922_92237


namespace NUMINAMATH_CALUDE_nitrogen_atomic_weight_l922_92280

/-- The atomic weight of nitrogen in a compound with given properties -/
theorem nitrogen_atomic_weight (molecular_weight : ℝ) (hydrogen_weight : ℝ) (bromine_weight : ℝ) :
  molecular_weight = 98 →
  hydrogen_weight = 1.008 →
  bromine_weight = 79.904 →
  molecular_weight = 4 * hydrogen_weight + bromine_weight + 14.064 :=
by sorry

end NUMINAMATH_CALUDE_nitrogen_atomic_weight_l922_92280


namespace NUMINAMATH_CALUDE_B_equals_set_l922_92220

def A : Set ℤ := {-2, -1, 1, 2, 3, 4}

def B : Set ℕ := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_set : B = {1, 4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_B_equals_set_l922_92220


namespace NUMINAMATH_CALUDE_m_range_theorem_l922_92202

/-- The range of m satisfying the given conditions -/
def m_range (m : ℝ) : Prop :=
  m ≥ 3 ∨ m < 0 ∨ (0 < m ∧ m ≤ 5/2)

/-- Line and parabola have no intersections -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x y : ℝ, x - 2*y + 3 = 0 → y^2 = m*x → m ≠ 0 → False

/-- Equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (5 - 2*m) + y^2 / m = 1 → m * (5 - 2*m) < 0

/-- Main theorem -/
theorem m_range_theorem (m : ℝ) :
  (no_intersection m ∨ is_hyperbola m) ∧ ¬(no_intersection m ∧ is_hyperbola m) →
  m_range m :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l922_92202


namespace NUMINAMATH_CALUDE_base_7_conversion_correct_l922_92282

/-- Converts a list of digits in base 7 to its decimal (base 10) representation -/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The decimal number we want to convert -/
def decimalNumber : Nat := 1987

/-- The proposed base 7 representation -/
def base7Digits : List Nat := [6, 3, 5, 3, 5]

/-- Theorem stating that the conversion is correct -/
theorem base_7_conversion_correct :
  toDecimal base7Digits = decimalNumber := by sorry

end NUMINAMATH_CALUDE_base_7_conversion_correct_l922_92282


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l922_92257

theorem sum_of_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (2 - x) * (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 →
  a₀ + a₆ = -30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l922_92257


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l922_92223

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 14

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 15

/-- The number of books read -/
def books_read : ℕ := 11

/-- The number of movies watched -/
def movies_watched : ℕ := 40

theorem crazy_silly_school_series :
  (num_books = num_movies + 1) ∧
  (num_books = 15) ∧
  (books_read = 11) ∧
  (movies_watched = 40) →
  num_movies = 14 := by
sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l922_92223


namespace NUMINAMATH_CALUDE_max_absolute_value_of_z_l922_92231

theorem max_absolute_value_of_z (z : ℂ) : 
  Complex.abs (z - (3 + 4*I)) ≤ 2 → Complex.abs z ≤ 7 ∧ ∃ w : ℂ, Complex.abs (w - (3 + 4*I)) ≤ 2 ∧ Complex.abs w = 7 :=
sorry

end NUMINAMATH_CALUDE_max_absolute_value_of_z_l922_92231


namespace NUMINAMATH_CALUDE_lunch_cost_with_tip_l922_92209

theorem lunch_cost_with_tip (total_cost : ℝ) (tip_percentage : ℝ) (cost_before_tip : ℝ) : 
  total_cost = 60.24 →
  tip_percentage = 0.20 →
  total_cost = cost_before_tip * (1 + tip_percentage) →
  cost_before_tip = 50.20 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_with_tip_l922_92209


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l922_92238

theorem six_digit_multiple_of_nine :
  ∀ (d : ℕ), d < 10 →
  (456780 + d) % 9 = 0 ↔ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l922_92238


namespace NUMINAMATH_CALUDE_square_area_l922_92255

theorem square_area (side_length : ℝ) (h : side_length = 7) : side_length ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l922_92255


namespace NUMINAMATH_CALUDE_child_playing_time_l922_92250

/-- Calculates the playing time for each child in a game where 6 children take turns playing for 120 minutes, with only two children playing at a time. -/
theorem child_playing_time (total_time : ℕ) (num_children : ℕ) (players_per_game : ℕ) :
  total_time = 120 ∧ num_children = 6 ∧ players_per_game = 2 →
  (total_time * players_per_game) / num_children = 40 := by
  sorry

end NUMINAMATH_CALUDE_child_playing_time_l922_92250


namespace NUMINAMATH_CALUDE_min_balls_to_draw_correct_l922_92210

/-- Represents the number of balls of each color in the container -/
structure BallContainer :=
  (red : ℕ)
  (green : ℕ)
  (yellow : ℕ)
  (blue : ℕ)
  (purple : ℕ)
  (orange : ℕ)

/-- The initial distribution of balls in the container -/
def initialContainer : BallContainer :=
  { red := 40
  , green := 25
  , yellow := 20
  , blue := 15
  , purple := 10
  , orange := 5 }

/-- The minimum number of balls of a single color we want to guarantee -/
def targetCount : ℕ := 18

/-- Function to calculate the minimum number of balls to draw -/
def minBallsToDraw (container : BallContainer) (target : ℕ) : ℕ :=
  sorry

theorem min_balls_to_draw_correct :
  minBallsToDraw initialContainer targetCount = 82 :=
sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_correct_l922_92210


namespace NUMINAMATH_CALUDE_rectangle_x_value_l922_92271

/-- A rectangular construction with specified side lengths -/
structure RectConstruction where
  top_left : ℝ
  top_middle : ℝ
  top_right : ℝ
  bottom_left : ℝ
  bottom_middle : ℝ
  bottom_right : ℝ

/-- The theorem stating that X = 5 in the given rectangular construction -/
theorem rectangle_x_value (r : RectConstruction) 
  (h1 : r.top_left = 2)
  (h2 : r.top_right = 3)
  (h3 : r.bottom_left = 4)
  (h4 : r.bottom_middle = 1)
  (h5 : r.bottom_right = 5)
  (h6 : r.top_left + r.top_middle + r.top_right = r.bottom_left + r.bottom_middle + r.bottom_right) :
  r.top_middle = 5 := by
  sorry

#check rectangle_x_value

end NUMINAMATH_CALUDE_rectangle_x_value_l922_92271


namespace NUMINAMATH_CALUDE_imaginary_unit_calculation_l922_92242

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_unit_calculation : i * (1 + i)^2 = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_calculation_l922_92242


namespace NUMINAMATH_CALUDE_estimate_city_standards_l922_92270

/-- Estimates the number of students meeting standards in a population based on a sample. -/
def estimate_meeting_standards (sample_size : ℕ) (sample_meeting : ℕ) (total_population : ℕ) : ℕ :=
  (total_population * sample_meeting) / sample_size

/-- Theorem stating the estimated number of students meeting standards in the city -/
theorem estimate_city_standards : 
  let sample_size := 1000
  let sample_meeting := 950
  let total_population := 1200000
  estimate_meeting_standards sample_size sample_meeting total_population = 1140000 := by
  sorry

end NUMINAMATH_CALUDE_estimate_city_standards_l922_92270


namespace NUMINAMATH_CALUDE_no_multiple_with_smaller_digit_sum_l922_92288

/-- The number composed of m digits all being ones -/
def ones_number (m : ℕ) : ℕ :=
  (10^m - 1) / 9

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number composed of m ones has no multiple with digit sum less than m -/
theorem no_multiple_with_smaller_digit_sum (m : ℕ) :
  ∀ k : ℕ, k > 0 → digit_sum (k * ones_number m) ≥ m :=
sorry

end NUMINAMATH_CALUDE_no_multiple_with_smaller_digit_sum_l922_92288


namespace NUMINAMATH_CALUDE_joan_balloons_l922_92207

/-- The number of blue balloons Joan has now, given her initial count and the number lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

theorem joan_balloons : remaining_balloons 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l922_92207


namespace NUMINAMATH_CALUDE_wage_comparison_l922_92222

/-- Proves that given the wage relationships between Erica, Robin, and Charles,
    Charles earns approximately 170% more than Erica. -/
theorem wage_comparison (erica robin charles : ℝ) 
  (h1 : robin = erica * 1.30)
  (h2 : charles = robin * 1.3076923076923077) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ 
  charles = erica * (2.70 + ε) :=
sorry

end NUMINAMATH_CALUDE_wage_comparison_l922_92222


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l922_92264

theorem greatest_perimeter_of_special_triangle :
  ∀ a b c : ℕ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (b = 4 * a ∨ a = 4 * b ∨ c = 4 * a ∨ a = 4 * c ∨ b = 4 * c ∨ c = 4 * b) →
  (a = 12 ∨ b = 12 ∨ c = 12) →
  (a + b > c ∧ b + c > a ∧ a + c > b) →
  a + b + c ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l922_92264


namespace NUMINAMATH_CALUDE_rowing_distance_l922_92293

/-- The distance to a destination given rowing conditions and round trip time -/
theorem rowing_distance (v : ℝ) (w : ℝ) (c : ℝ) (t : ℝ) (h1 : v > 0) (h2 : w ≥ 0) (h3 : c ≥ 0) (h4 : t > 0) :
  let d := (t * (v + c) * (v + c - w)) / ((v + c) + (v + c - w))
  d = 45 / 11 ↔ v = 4 ∧ w = 1 ∧ c = 2 ∧ t = 3/2 := by
  sorry

#check rowing_distance

end NUMINAMATH_CALUDE_rowing_distance_l922_92293


namespace NUMINAMATH_CALUDE_real_y_condition_l922_92266

theorem real_y_condition (x : ℝ) :
  (∃ y : ℝ, 4 * y^2 - 2 * x * y + 2 * x + 9 = 0) ↔ (x ≤ -3 ∨ x ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_real_y_condition_l922_92266


namespace NUMINAMATH_CALUDE_john_running_distance_l922_92247

def monday_distance : ℕ := 1700
def tuesday_distance : ℕ := monday_distance + 200
def wednesday_distance : ℕ := (7 * tuesday_distance) / 10
def thursday_distance : ℕ := 2 * wednesday_distance
def friday_distance : ℕ := 3500

def total_distance : ℕ := monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance

theorem john_running_distance : total_distance = 10090 := by
  sorry

end NUMINAMATH_CALUDE_john_running_distance_l922_92247


namespace NUMINAMATH_CALUDE_newspaper_cost_difference_l922_92272

/-- Grant's yearly newspaper expenditure -/
def grant_yearly_cost : ℝ := 200

/-- Juanita's weekday newspaper cost -/
def juanita_weekday_cost : ℝ := 0.5

/-- Juanita's Sunday newspaper cost -/
def juanita_sunday_cost : ℝ := 2

/-- Number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Number of weekdays in a week -/
def weekdays_per_week : ℕ := 6

/-- Juanita's weekly newspaper cost -/
def juanita_weekly_cost : ℝ := juanita_weekday_cost * weekdays_per_week + juanita_sunday_cost

/-- Juanita's yearly newspaper cost -/
def juanita_yearly_cost : ℝ := juanita_weekly_cost * weeks_per_year

theorem newspaper_cost_difference : juanita_yearly_cost - grant_yearly_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_cost_difference_l922_92272


namespace NUMINAMATH_CALUDE_rectangle_length_l922_92221

/-- Given a rectangle where the length is three times the width, and decreasing the length by 5
    while increasing the width by 5 results in a square, prove that the original length is 15. -/
theorem rectangle_length (w : ℝ) (h1 : w > 0) : 
  (∃ l : ℝ, l = 3 * w ∧ l - 5 = w + 5) → 3 * w = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l922_92221


namespace NUMINAMATH_CALUDE_tuesday_attendance_theorem_l922_92277

/-- Represents the attendance status of students at Dunkley S.S. over two days -/
structure AttendanceData where
  total_students : ℕ
  monday_absent_rate : ℚ
  tuesday_return_rate : ℚ
  tuesday_absent_rate : ℚ

/-- Calculates the percentage of students present on Tuesday -/
def tuesday_present_percentage (data : AttendanceData) : ℚ :=
  let monday_present := 1 - data.monday_absent_rate
  let tuesday_present_from_monday := monday_present * (1 - data.tuesday_absent_rate)
  let tuesday_present_from_absent := data.monday_absent_rate * data.tuesday_return_rate
  (tuesday_present_from_monday + tuesday_present_from_absent) * 100

/-- Theorem stating that given the conditions, the percentage of students present on Tuesday is 82% -/
theorem tuesday_attendance_theorem (data : AttendanceData) 
  (h1 : data.total_students > 0)
  (h2 : data.monday_absent_rate = 1/10)
  (h3 : data.tuesday_return_rate = 1/10)
  (h4 : data.tuesday_absent_rate = 1/10) :
  tuesday_present_percentage data = 82 := by
  sorry


end NUMINAMATH_CALUDE_tuesday_attendance_theorem_l922_92277


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l922_92216

def i : ℂ := Complex.I

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 1 / (1 + i)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l922_92216


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l922_92279

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 + b*x - a < 0 ↔ -2 < x ∧ x < 3) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l922_92279


namespace NUMINAMATH_CALUDE_juan_reading_speed_l922_92297

/-- Proves that Juan reads 250 pages per hour given the conditions of the problem -/
theorem juan_reading_speed (lunch_trip : ℝ) (book_pages : ℕ) (office_to_lunch : ℝ) 
  (h1 : lunch_trip = 2 * office_to_lunch)
  (h2 : book_pages = 4000)
  (h3 : office_to_lunch = 4)
  (h4 : lunch_trip = (book_pages : ℝ) / (250 : ℝ)) : 
  (book_pages : ℝ) / (2 * lunch_trip) = 250 := by
sorry

end NUMINAMATH_CALUDE_juan_reading_speed_l922_92297


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l922_92245

theorem quadratic_solution_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) → 
  (3 * q^2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = 122 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l922_92245


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_divided_by_five_l922_92225

theorem units_digit_of_sum_of_powers_divided_by_five :
  ∃ n : ℕ, (2^2023 + 3^2023) / 5 ≡ n [ZMOD 10] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_divided_by_five_l922_92225


namespace NUMINAMATH_CALUDE_correct_operation_l922_92284

theorem correct_operation (a : ℝ) : 2 * a^2 * (3 * a) = 6 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l922_92284
