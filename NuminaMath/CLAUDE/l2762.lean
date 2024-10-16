import Mathlib

namespace NUMINAMATH_CALUDE_digit_count_proof_l2762_276266

theorem digit_count_proof (total_count : ℕ) (available_digits : ℕ) 
  (h1 : total_count = 28672) 
  (h2 : available_digits = 8) : 
  ∃ n : ℕ, available_digits ^ n = total_count ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_proof_l2762_276266


namespace NUMINAMATH_CALUDE_remaining_tanning_time_l2762_276236

/-- Calculates the remaining tanning time for the last two weeks of a month. -/
theorem remaining_tanning_time 
  (monthly_limit : ℕ) 
  (daily_time : ℕ) 
  (days_per_week : ℕ) 
  (weeks : ℕ) 
  (h1 : monthly_limit = 200)
  (h2 : daily_time = 30)
  (h3 : days_per_week = 2)
  (h4 : weeks = 2) :
  monthly_limit - (daily_time * days_per_week * weeks) = 80 := by
  sorry

#check remaining_tanning_time

end NUMINAMATH_CALUDE_remaining_tanning_time_l2762_276236


namespace NUMINAMATH_CALUDE_binary_op_property_l2762_276213

-- Define a binary operation on a type S
def binary_op (S : Type) := S → S → S

-- State the theorem
theorem binary_op_property {S : Type} (op : binary_op S) 
  (h : ∀ (a b : S), op (op a b) a = b) :
  ∀ (a b : S), op a (op b a) = b := by
  sorry

end NUMINAMATH_CALUDE_binary_op_property_l2762_276213


namespace NUMINAMATH_CALUDE_ultra_savings_interest_theorem_l2762_276231

/-- Represents the Ultra Savings Account investment scenario -/
structure UltraSavingsAccount where
  principal : ℝ
  rate : ℝ
  years : ℕ

/-- Calculates the final balance after compound interest -/
def finalBalance (account : UltraSavingsAccount) : ℝ :=
  account.principal * (1 + account.rate) ^ account.years

/-- Calculates the interest earned -/
def interestEarned (account : UltraSavingsAccount) : ℝ :=
  finalBalance account - account.principal

/-- Theorem stating that the interest earned is approximately $328.49 -/
theorem ultra_savings_interest_theorem (account : UltraSavingsAccount) 
  (h1 : account.principal = 1500)
  (h2 : account.rate = 0.02)
  (h3 : account.years = 10) : 
  ∃ ε > 0, |interestEarned account - 328.49| < ε :=
sorry

end NUMINAMATH_CALUDE_ultra_savings_interest_theorem_l2762_276231


namespace NUMINAMATH_CALUDE_specific_pentagon_area_l2762_276212

/-- A pentagon with specific side lengths that can be divided into a right triangle and a trapezoid -/
structure SpecificPentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  triangle_base : ℝ
  triangle_height : ℝ
  trapezoid_base1 : ℝ
  trapezoid_base2 : ℝ
  trapezoid_height : ℝ
  side1_eq : side1 = 15
  side2_eq : side2 = 20
  side3_eq : side3 = 27
  side4_eq : side4 = 24
  side5_eq : side5 = 20
  triangle_base_eq : triangle_base = 15
  triangle_height_eq : triangle_height = 20
  trapezoid_base1_eq : trapezoid_base1 = 20
  trapezoid_base2_eq : trapezoid_base2 = 27
  trapezoid_height_eq : trapezoid_height = 24

/-- The area of the specific pentagon is 714 square units -/
theorem specific_pentagon_area (p : SpecificPentagon) : 
  (1/2 * p.triangle_base * p.triangle_height) + 
  (1/2 * (p.trapezoid_base1 + p.trapezoid_base2) * p.trapezoid_height) = 714 := by
  sorry

end NUMINAMATH_CALUDE_specific_pentagon_area_l2762_276212


namespace NUMINAMATH_CALUDE_gaming_chair_price_proof_l2762_276222

/-- The price of a set of toy organizers -/
def toy_organizer_price : ℝ := 78

/-- The number of toy organizer sets ordered -/
def toy_organizer_sets : ℕ := 3

/-- The number of gaming chairs ordered -/
def gaming_chairs : ℕ := 2

/-- The delivery fee percentage -/
def delivery_fee_percent : ℝ := 0.05

/-- The total amount Leon paid -/
def total_paid : ℝ := 420

/-- The price of a gaming chair -/
def gaming_chair_price : ℝ := 83

theorem gaming_chair_price_proof :
  gaming_chair_price * gaming_chairs + toy_organizer_price * toy_organizer_sets +
  (gaming_chair_price * gaming_chairs + toy_organizer_price * toy_organizer_sets) * delivery_fee_percent =
  total_paid := by sorry

end NUMINAMATH_CALUDE_gaming_chair_price_proof_l2762_276222


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2762_276277

theorem contrapositive_equivalence :
  (∀ a : ℝ, a > 1 → a^2 > 1) ↔ (∀ a : ℝ, a^2 ≤ 1 → a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2762_276277


namespace NUMINAMATH_CALUDE_inequality_proof_l2762_276204

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 3*c) / (a + 2*b + c) + 4*b / (a + b + 2*c) - 8*c / (a + b + 3*c) ≥ -17 + 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2762_276204


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_80_l2762_276238

theorem last_three_digits_of_7_to_80 : 7^80 ≡ 961 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_80_l2762_276238


namespace NUMINAMATH_CALUDE_prob_one_defective_in_two_l2762_276206

/-- Given a set of 4 items with 3 genuine and 1 defective, the probability
of selecting exactly one defective item when randomly choosing 2 items is 1/2. -/
theorem prob_one_defective_in_two (n : ℕ) (k : ℕ) (d : ℕ) :
  n = 4 →
  k = 2 →
  d = 1 →
  (n.choose k) = 6 →
  (d * (n - d).choose (k - 1)) = 3 →
  (d * (n - d).choose (k - 1)) / (n.choose k) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_prob_one_defective_in_two_l2762_276206


namespace NUMINAMATH_CALUDE_gcd_9155_4892_l2762_276240

theorem gcd_9155_4892 : Nat.gcd 9155 4892 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9155_4892_l2762_276240


namespace NUMINAMATH_CALUDE_sin_pi_sufficient_not_necessary_l2762_276232

open Real

theorem sin_pi_sufficient_not_necessary :
  (∀ x : ℝ, x = π → sin x = 0) ∧
  (∃ x : ℝ, x ≠ π ∧ sin x = 0) := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_sufficient_not_necessary_l2762_276232


namespace NUMINAMATH_CALUDE_acid_dilution_l2762_276200

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution results in a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.40 →
  water_added = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_dilution_l2762_276200


namespace NUMINAMATH_CALUDE_all_dice_same_probability_l2762_276244

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The number of dice being tossed -/
def numberOfDice : ℕ := 5

/-- The probability of all dice showing the same number -/
def probabilityAllSame : ℚ := 1 / (standardDieSides ^ (numberOfDice - 1))

theorem all_dice_same_probability :
  probabilityAllSame = 1 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_all_dice_same_probability_l2762_276244


namespace NUMINAMATH_CALUDE_tangent_line_and_perpendicular_points_l2762_276250

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_and_perpendicular_points :
  -- Part 1: Equation of tangent line at (1, -1)
  (∀ x y : ℝ, (x = 1 ∧ y = f 1) → (2*x - y - 3 = 0)) ∧
  -- Part 2: Points where tangent is perpendicular to y = -1/2x + 3
  (∀ x : ℝ, (f' x = 2) → (x = 1 ∨ x = -1)) ∧
  (∀ x : ℝ, (x = 1 ∨ x = -1) → f x = -1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_perpendicular_points_l2762_276250


namespace NUMINAMATH_CALUDE_katie_soccer_granola_l2762_276299

/-- The number of boxes of granola bars needed for a soccer game --/
def granola_boxes_needed (num_kids : ℕ) (bars_per_kid : ℕ) (bars_per_box : ℕ) : ℕ :=
  (num_kids * bars_per_kid + bars_per_box - 1) / bars_per_box

theorem katie_soccer_granola : granola_boxes_needed 30 2 12 = 5 := by
  sorry

end NUMINAMATH_CALUDE_katie_soccer_granola_l2762_276299


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2762_276291

theorem complex_equation_solution (z : ℂ) : z * (1 - 2*I) = 2 + I → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2762_276291


namespace NUMINAMATH_CALUDE_data_transmission_time_l2762_276278

theorem data_transmission_time (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) :
  blocks = 80 →
  chunks_per_block = 640 →
  transmission_rate = 160 →
  (blocks * chunks_per_block : ℚ) / transmission_rate = 320 :=
by sorry

end NUMINAMATH_CALUDE_data_transmission_time_l2762_276278


namespace NUMINAMATH_CALUDE_sequence_difference_l2762_276293

def arithmetic_sum (a₁ aₙ : ℤ) (n : ℕ) : ℤ := n * (a₁ + aₙ) / 2

def sequence_1_sum : ℤ := arithmetic_sum 2 2021 674
def sequence_2_sum : ℤ := arithmetic_sum 3 2022 674

theorem sequence_difference : sequence_1_sum - sequence_2_sum = -544 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l2762_276293


namespace NUMINAMATH_CALUDE_andre_purchase_total_l2762_276284

/-- Calculates the discounted price given the original price and discount percentage. -/
def discountedPrice (originalPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  originalPrice * (1 - discountPercentage / 100)

/-- Calculates the total price for multiple items with the same price. -/
def totalPrice (itemPrice : ℚ) (quantity : ℕ) : ℚ :=
  itemPrice * quantity

theorem andre_purchase_total : 
  let treadmillOriginalPrice : ℚ := 1350
  let treadmillDiscount : ℚ := 30
  let plateOriginalPrice : ℚ := 60
  let plateQuantity : ℕ := 2
  let plateDiscount : ℚ := 15
  
  discountedPrice treadmillOriginalPrice treadmillDiscount + 
  discountedPrice (totalPrice plateOriginalPrice plateQuantity) plateDiscount = 1047 := by
sorry

end NUMINAMATH_CALUDE_andre_purchase_total_l2762_276284


namespace NUMINAMATH_CALUDE_set_operations_l2762_276296

-- Define the sets A and B
def A : Set ℝ := {x | x = 0 ∨ ∃ y, x = |y|}
def B : Set ℝ := {-1, 0, 1}

-- State the theorem
theorem set_operations (h : A ⊆ B) :
  (A ∩ B = {0, 1}) ∧
  (A ∪ B = {-1, 0, 1}) ∧
  (B \ A = {-1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2762_276296


namespace NUMINAMATH_CALUDE_product_equals_quadratic_l2762_276286

theorem product_equals_quadratic : ∃ m : ℤ, 72516 * 9999 = m^2 - 5*m + 7 ∧ m = 26926 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_quadratic_l2762_276286


namespace NUMINAMATH_CALUDE_rectangle_perimeter_10_l2762_276270

/-- A rectangle with sides a and b. -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0

/-- The perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.a + r.b)

/-- The sum of three sides of a rectangle. -/
def sum_three_sides (r : Rectangle) : Set ℝ := {2 * r.a + r.b, r.a + 2 * r.b}

/-- Theorem stating that there exists a rectangle with perimeter 10,
    given that the sum of the lengths of three different sides can be equal to 6 or 9. -/
theorem rectangle_perimeter_10 :
  ∃ r : Rectangle, (6 ∈ sum_three_sides r ∨ 9 ∈ sum_three_sides r) ∧ perimeter r = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_10_l2762_276270


namespace NUMINAMATH_CALUDE_gcd_of_squares_plus_one_l2762_276207

theorem gcd_of_squares_plus_one (n : ℕ+) : 
  Nat.gcd (n.val^2 + 1) ((n.val + 1)^2 + 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_squares_plus_one_l2762_276207


namespace NUMINAMATH_CALUDE_dry_grapes_weight_l2762_276239

-- Define the parameters
def fresh_water_content : Real := 0.90
def dried_water_content : Real := 0.20
def fresh_grapes_weight : Real := 5

-- Define the theorem
theorem dry_grapes_weight :
  let non_water_content := (1 - fresh_water_content) * fresh_grapes_weight
  let dry_grapes_weight := non_water_content / (1 - dried_water_content)
  dry_grapes_weight = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_dry_grapes_weight_l2762_276239


namespace NUMINAMATH_CALUDE_ratio_transformation_l2762_276261

/-- Given an original ratio of 2:3, prove that adding 2 to each term results in a ratio of 4:5 -/
theorem ratio_transformation (x : ℚ) : x = 2 → (2 + x) / (3 + x) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transformation_l2762_276261


namespace NUMINAMATH_CALUDE_annual_grass_cutting_cost_l2762_276272

/-- The annual cost of grass cutting given specific conditions -/
theorem annual_grass_cutting_cost
  (initial_height : ℝ)
  (growth_rate : ℝ)
  (cutting_threshold : ℝ)
  (cost_per_cut : ℝ)
  (h1 : initial_height = 2)
  (h2 : growth_rate = 0.5)
  (h3 : cutting_threshold = 4)
  (h4 : cost_per_cut = 100)
  : ℝ :=
by
  -- Prove that the annual cost of grass cutting is $300
  sorry

#check annual_grass_cutting_cost

end NUMINAMATH_CALUDE_annual_grass_cutting_cost_l2762_276272


namespace NUMINAMATH_CALUDE_max_snacks_is_11_l2762_276259

/-- Represents the number of snacks in a pack -/
inductive SnackPack
  | Single : SnackPack
  | Pack4 : SnackPack
  | Pack7 : SnackPack

/-- The cost of a snack pack in dollars -/
def cost : SnackPack → ℕ
  | SnackPack.Single => 2
  | SnackPack.Pack4 => 6
  | SnackPack.Pack7 => 9

/-- The number of snacks in a pack -/
def snacks : SnackPack → ℕ
  | SnackPack.Single => 1
  | SnackPack.Pack4 => 4
  | SnackPack.Pack7 => 7

/-- The budget in dollars -/
def budget : ℕ := 15

/-- A purchase is a list of snack packs -/
def Purchase := List SnackPack

/-- The total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.foldl (fun acc pack => acc + cost pack) 0

/-- The total number of snacks in a purchase -/
def totalSnacks (p : Purchase) : ℕ :=
  p.foldl (fun acc pack => acc + snacks pack) 0

/-- A purchase is valid if its total cost is within the budget -/
def isValidPurchase (p : Purchase) : Prop :=
  totalCost p ≤ budget

/-- The theorem stating that 11 is the maximum number of snacks that can be purchased -/
theorem max_snacks_is_11 :
  ∀ p : Purchase, isValidPurchase p → totalSnacks p ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_snacks_is_11_l2762_276259


namespace NUMINAMATH_CALUDE_expression_evaluation_l2762_276219

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : 3 * x - y / 3 ≠ 0) :
  (3 * x - y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹)^2 = (y + 9 * x)^2 / (3 * x^2 * y^2 * (9 * x - y)) :=
by sorry


end NUMINAMATH_CALUDE_expression_evaluation_l2762_276219


namespace NUMINAMATH_CALUDE_sine_cosine_problem_l2762_276249

theorem sine_cosine_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < π/2) 
  (h2 : Real.sin x + Real.cos x = -1/5) : 
  (Real.sin x - Real.cos x = 7/5) ∧ 
  ((Real.sin (π + x) + Real.sin (3*π/2 - x)) / (Real.tan (π - x) + Real.sin (π/2 - x)) = 3/11) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_problem_l2762_276249


namespace NUMINAMATH_CALUDE_unique_a_value_l2762_276221

-- Define the inequality function
def inequality (a x : ℝ) : Prop :=
  (a * x - 20) * Real.log (2 * a / x) ≤ 0

-- State the theorem
theorem unique_a_value : 
  ∃! a : ℝ, ∀ x : ℝ, x > 0 → inequality a x :=
by
  -- The unique value of a is √10
  use Real.sqrt 10
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_unique_a_value_l2762_276221


namespace NUMINAMATH_CALUDE_smallest_odd_divisor_of_difference_of_squares_l2762_276283

theorem smallest_odd_divisor_of_difference_of_squares (m n : ℕ) : 
  Odd m → Odd n → n < m → 
  (∃ (k : ℕ), ∀ (a b : ℕ), Odd a → Odd b → b < a → k ∣ (a^2 - b^2)) → 
  (∃ (d : ℕ), Odd d ∧ d ∣ (m^2 - n^2) ∧ 
    ∀ (e : ℕ), Odd e → e ∣ (m^2 - n^2) → d ≤ e) → 
  ∃ (d : ℕ), d = 1 ∧ Odd d ∧ d ∣ (m^2 - n^2) ∧ 
    ∀ (e : ℕ), Odd e → e ∣ (m^2 - n^2) → d ≤ e :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_divisor_of_difference_of_squares_l2762_276283


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2762_276276

theorem modulus_of_complex_number (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2762_276276


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2762_276233

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y = 2 * x * y) :
  x + y ≥ 9 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2762_276233


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2762_276297

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a*b*c) ≥ 216 := by
  sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a*b*c) = 216 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2762_276297


namespace NUMINAMATH_CALUDE_candy_remainder_l2762_276230

theorem candy_remainder : (31254389 : ℕ) % 6 = 5 := by sorry

end NUMINAMATH_CALUDE_candy_remainder_l2762_276230


namespace NUMINAMATH_CALUDE_exists_valid_matrix_with_equal_products_l2762_276251

/-- A 3x3 matrix of natural numbers -/
def Matrix3x3 := Fin 3 → Fin 3 → ℕ

/-- Check if all elements in a matrix are distinct and not exceeding 40 -/
def is_valid_matrix (m : Matrix3x3) : Prop :=
  (∀ i j, m i j ≤ 40) ∧
  (∀ i j i' j', (i, j) ≠ (i', j') → m i j ≠ m i' j')

/-- Calculate the product of a row in the matrix -/
def row_product (m : Matrix3x3) (i : Fin 3) : ℕ :=
  (m i 0) * (m i 1) * (m i 2)

/-- Calculate the product of a column in the matrix -/
def col_product (m : Matrix3x3) (j : Fin 3) : ℕ :=
  (m 0 j) * (m 1 j) * (m 2 j)

/-- Calculate the product of the main diagonal -/
def main_diag_product (m : Matrix3x3) : ℕ :=
  (m 0 0) * (m 1 1) * (m 2 2)

/-- Calculate the product of the anti-diagonal -/
def anti_diag_product (m : Matrix3x3) : ℕ :=
  (m 0 2) * (m 1 1) * (m 2 0)

/-- Check if all products in the matrix are equal to a given value -/
def all_products_equal (m : Matrix3x3) (p : ℕ) : Prop :=
  (∀ i, row_product m i = p) ∧
  (∀ j, col_product m j = p) ∧
  (main_diag_product m = p) ∧
  (anti_diag_product m = p)

/-- The main theorem stating the existence of a valid matrix with all products equal to 216 -/
theorem exists_valid_matrix_with_equal_products :
  ∃ (m : Matrix3x3), is_valid_matrix m ∧ all_products_equal m 216 := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_matrix_with_equal_products_l2762_276251


namespace NUMINAMATH_CALUDE_mandarin_ducks_count_l2762_276248

/-- The number of pairs of mandarin ducks -/
def num_pairs : ℕ := 3

/-- The number of ducks in each pair -/
def ducks_per_pair : ℕ := 2

/-- The total number of mandarin ducks -/
def total_ducks : ℕ := num_pairs * ducks_per_pair

theorem mandarin_ducks_count : total_ducks = 6 := by
  sorry

end NUMINAMATH_CALUDE_mandarin_ducks_count_l2762_276248


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2762_276226

/-- 
A geometric sequence is defined by its first term and common ratio.
This theorem proves that for a geometric sequence where the third term is 18
and the sixth term is 162, the first term is 2 and the common ratio is 3.
-/
theorem geometric_sequence_problem (a r : ℝ) : 
  a * r^2 = 18 → a * r^5 = 162 → a = 2 ∧ r = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2762_276226


namespace NUMINAMATH_CALUDE_sum_first_15_odd_integers_l2762_276257

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun n => 2 * n + 1) = 225 := by sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_integers_l2762_276257


namespace NUMINAMATH_CALUDE_remainder_789987_div_8_l2762_276225

theorem remainder_789987_div_8 : 789987 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_789987_div_8_l2762_276225


namespace NUMINAMATH_CALUDE_fraction_equality_l2762_276287

theorem fraction_equality (a b c d : ℝ) 
  (h1 : a / b = 4)
  (h2 : b / c = 1 / 3)
  (h3 : c / d = 6) :
  d / a = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2762_276287


namespace NUMINAMATH_CALUDE_nap_time_calculation_l2762_276241

/-- Calculates the remaining time for a nap given flight duration and time spent on activities --/
def time_for_nap (flight_duration : ℕ) (reading : ℕ) (movies : ℕ) (dinner : ℕ) (radio : ℕ) (games : ℕ) : ℕ :=
  flight_duration - (reading + movies + dinner + radio + games)

/-- Converts hours and minutes to minutes --/
def to_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

theorem nap_time_calculation :
  let flight_duration := to_minutes 11 20
  let reading := to_minutes 2 0
  let movies := to_minutes 4 0
  let dinner := 30
  let radio := 40
  let games := to_minutes 1 10
  let nap_time := time_for_nap flight_duration reading movies dinner radio games
  nap_time = to_minutes 3 0 := by sorry

end NUMINAMATH_CALUDE_nap_time_calculation_l2762_276241


namespace NUMINAMATH_CALUDE_wallpaper_area_proof_l2762_276268

theorem wallpaper_area_proof (total_area overlap_area double_layer triple_layer : ℝ) : 
  overlap_area = 180 →
  double_layer = 30 →
  triple_layer = 45 →
  total_area - 2 * double_layer - 3 * triple_layer = overlap_area →
  total_area = 375 := by
  sorry

end NUMINAMATH_CALUDE_wallpaper_area_proof_l2762_276268


namespace NUMINAMATH_CALUDE_book_sale_revenue_l2762_276203

theorem book_sale_revenue (total_books : ℕ) (sold_price : ℕ) (remaining_books : ℕ) : 
  (2 * total_books = 3 * remaining_books) →
  (sold_price = 5) →
  (remaining_books = 50) →
  (2 * total_books / 3 * sold_price = 500) :=
by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l2762_276203


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l2762_276229

/-- Given two vectors a and b in a vector space, the area_parallelogram function
    computes the area of the parallelogram generated by these vectors. -/
def area_parallelogram (a b : V) : ℝ := sorry

variable (V : Type) [AddCommGroup V] [Module ℝ V]
variable (a b : V)

/-- The theorem states that if the area of the parallelogram generated by vectors a and b
    is 15, then the area of the parallelogram generated by vectors 3a + 4b and 2a - 6b is 390. -/
theorem parallelogram_area_theorem (h : area_parallelogram a b = 15) :
  area_parallelogram (3 • a + 4 • b) (2 • a - 6 • b) = 390 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l2762_276229


namespace NUMINAMATH_CALUDE_combined_average_score_l2762_276216

theorem combined_average_score (score_u score_b score_c : ℝ)
  (ratio_u ratio_b ratio_c : ℕ) :
  score_u = 65 →
  score_b = 80 →
  score_c = 77 →
  ratio_u = 4 →
  ratio_b = 6 →
  ratio_c = 5 →
  (score_u * ratio_u + score_b * ratio_b + score_c * ratio_c) / (ratio_u + ratio_b + ratio_c) = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_average_score_l2762_276216


namespace NUMINAMATH_CALUDE_potato_slab_length_difference_l2762_276262

theorem potato_slab_length_difference (total_length first_piece_length : ℕ) 
  (h1 : total_length = 600)
  (h2 : first_piece_length = 275) :
  total_length - first_piece_length - first_piece_length = 50 :=
by sorry

end NUMINAMATH_CALUDE_potato_slab_length_difference_l2762_276262


namespace NUMINAMATH_CALUDE_count_integer_pairs_l2762_276201

theorem count_integer_pairs : 
  let w_count := (Finset.range 450).filter (fun w => w % 23 = 5) |>.card
  let n_count := (Finset.range 450).filter (fun n => n % 17 = 7) |>.card
  w_count * n_count = 540 := by
sorry

end NUMINAMATH_CALUDE_count_integer_pairs_l2762_276201


namespace NUMINAMATH_CALUDE_min_value_exponential_function_l2762_276255

theorem min_value_exponential_function :
  ∀ x : ℝ, 4 * Real.exp x + Real.exp (-x) ≥ 4 ∧
  ∃ x₀ : ℝ, 4 * Real.exp x₀ + Real.exp (-x₀) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_function_l2762_276255


namespace NUMINAMATH_CALUDE_inverse_of_inverse_nine_l2762_276289

def f (x : ℝ) : ℝ := 5 * x + 7

theorem inverse_of_inverse_nine :
  let f_inv (x : ℝ) := (x - 7) / 5
  f_inv (f_inv 9) = -33 / 25 := by
sorry

end NUMINAMATH_CALUDE_inverse_of_inverse_nine_l2762_276289


namespace NUMINAMATH_CALUDE_line_equation_proof_l2762_276282

/-- Proves that the equation of a line with slope -2 and y-intercept 3 is 2x + y - 3 = 0 -/
theorem line_equation_proof (x y : ℝ) : 
  let slope : ℝ := -2
  let y_intercept : ℝ := 3
  let line_equation := fun (x y : ℝ) => 2 * x + y - 3 = 0
  line_equation x y ↔ y = slope * x + y_intercept :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2762_276282


namespace NUMINAMATH_CALUDE_reflect_F_theorem_l2762_276260

/-- Reflects a point over the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point over the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- The composition of two reflections -/
def double_reflect (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_eq_x (reflect_x_axis p)

theorem reflect_F_theorem :
  let F : ℝ × ℝ := (1, 1)
  double_reflect F = (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_reflect_F_theorem_l2762_276260


namespace NUMINAMATH_CALUDE_intersection_y_coordinate_l2762_276285

/-- The y-coordinate of the intersection point between a line and a parabola -/
theorem intersection_y_coordinate (x : ℝ) : 
  x > 0 ∧ 
  (x - 1)^2 + 1 = -2*x + 11 → 
  -2*x + 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_coordinate_l2762_276285


namespace NUMINAMATH_CALUDE_angle_quadrant_l2762_276205

theorem angle_quadrant (θ : Real) : 
  (Real.sin θ * Real.cos θ > 0) → 
  (0 < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2) ∨
  (Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_quadrant_l2762_276205


namespace NUMINAMATH_CALUDE_quadratic_max_l2762_276280

/-- Given a quadratic function f(x) = ax^2 + bx + c where a < 0,
    and x₀ satisfies 2ax + b = 0, then for all x ∈ ℝ, f(x) ≤ f(x₀) -/
theorem quadratic_max (a b c : ℝ) (x₀ : ℝ) (h₁ : a < 0) (h₂ : 2 * a * x₀ + b = 0) :
  ∀ x : ℝ, a * x^2 + b * x + c ≤ a * x₀^2 + b * x₀ + c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_l2762_276280


namespace NUMINAMATH_CALUDE_shaded_area_hexagon_with_semicircles_l2762_276281

/-- The area of the shaded region in a regular hexagon with inscribed semicircles -/
theorem shaded_area_hexagon_with_semicircles (s : ℝ) (h : s = 3) :
  let hexagon_area := 3 * Real.sqrt 3 / 2 * s^2
  let semicircle_area := π * (s/2)^2 / 2
  let total_semicircle_area := 3 * semicircle_area
  hexagon_area - total_semicircle_area = 13.5 * Real.sqrt 3 - 27 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_hexagon_with_semicircles_l2762_276281


namespace NUMINAMATH_CALUDE_pullups_calculation_l2762_276290

/-- Calculates the number of pull-ups done per visit given the total pull-ups per week and visits per day -/
def pullups_per_visit (total_pullups : ℕ) (visits_per_day : ℕ) : ℚ :=
  total_pullups / (visits_per_day * 7)

/-- Theorem: If a person does 70 pull-ups per week and visits a room 5 times per day, 
    then the number of pull-ups done each visit is 2 -/
theorem pullups_calculation :
  pullups_per_visit 70 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pullups_calculation_l2762_276290


namespace NUMINAMATH_CALUDE_original_number_problem_l2762_276246

theorem original_number_problem (x : ℝ) : 
  (x + 0.25 * x) - (x - 0.30 * x) = 22 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l2762_276246


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2762_276220

theorem quadratic_inequality_range (m : ℝ) : 
  (∃ x ∈ Set.Icc 2 4, x^2 - 2*x + 5 - m < 0) ↔ m > 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2762_276220


namespace NUMINAMATH_CALUDE_auditorium_seats_l2762_276227

theorem auditorium_seats (initial_seats : ℕ) (final_seats : ℕ) (seat_increase : ℕ) : 
  initial_seats = 320 →
  final_seats = 420 →
  seat_increase = 4 →
  ∃ (initial_rows : ℕ),
    initial_rows > 0 ∧
    initial_seats % initial_rows = 0 ∧
    (initial_seats / initial_rows + seat_increase) * (initial_rows + 1) = final_seats ∧
    initial_rows + 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_auditorium_seats_l2762_276227


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocals_l2762_276252

theorem quadratic_roots_sum_reciprocals (a b : ℝ) : 
  (a^2 + 8*a + 4 = 0) → (b^2 + 8*b + 4 = 0) → (a / b + b / a = 14) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocals_l2762_276252


namespace NUMINAMATH_CALUDE_largest_expression_l2762_276273

def y : ℝ := 0.0002

theorem largest_expression (a b c d e : ℝ) 
  (ha : a = 5 + y)
  (hb : b = 5 - y)
  (hc : c = 5 * y)
  (hd : d = 5 / y)
  (he : e = y / 5) :
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_expression_l2762_276273


namespace NUMINAMATH_CALUDE_balloon_ratio_l2762_276247

theorem balloon_ratio : 
  let initial_blue : ℕ := 303
  let initial_purple : ℕ := 453
  let total_initial : ℕ := initial_blue + initial_purple
  let kept : ℕ := 378
  (kept : ℚ) / total_initial = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_balloon_ratio_l2762_276247


namespace NUMINAMATH_CALUDE_star_equation_solution_l2762_276218

/-- The star operation defined on real numbers -/
def star (x y : ℝ) : ℝ := 5*x - 2*y + 2*x*y

/-- Theorem stating that 4 star y = 22 if and only if y = 1/3 -/
theorem star_equation_solution :
  ∀ y : ℝ, star 4 y = 22 ↔ y = 1/3 := by sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2762_276218


namespace NUMINAMATH_CALUDE_intersection_M_N_l2762_276243

def M : Set ℝ := {0, 1, 2, 3}
def N : Set ℝ := {x | x^2 + x - 6 < 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2762_276243


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2762_276256

/-- A line passing through (2, 3) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (2, 3) -/
  passes_through_point : m * 2 + b = 3
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b ≠ 0 → -b/m = b

/-- The equation of the line is either x + y - 5 = 0 or 3x - 2y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, y = l.m * x + l.b → x + y = 5) ∨
  (∀ x y, y = l.m * x + l.b → 3*x - 2*y = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2762_276256


namespace NUMINAMATH_CALUDE_race_outcomes_l2762_276271

theorem race_outcomes (n : ℕ) (k : ℕ) (h : n = 6 ∧ k = 4) : 
  n * (n - 1) * (n - 2) * (n - 3) = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l2762_276271


namespace NUMINAMATH_CALUDE_triangle_base_length_l2762_276237

theorem triangle_base_length 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : height = 8) 
  (h2 : area = 24) 
  (h3 : area = (1/2) * height * base) : 
  base = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2762_276237


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2762_276254

theorem arithmetic_expression_equality : 72 + (120 / 15) + (18 * 19) - 250 - (360 / 6) = 112 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2762_276254


namespace NUMINAMATH_CALUDE_symmetric_roots_iff_b_eq_two_or_four_l2762_276267

/-- The polynomial in question -/
def P (b : ℝ) (z : ℂ) : ℂ :=
  z^5 - 8*z^4 + 12*b*z^3 - 4*(3*b^2 + 4*b - 4)*z^2 + 2*z + 2

/-- The roots of the polynomial form a symmetric pattern around the origin -/
def symmetric_roots (b : ℝ) : Prop :=
  ∃ (r : Finset ℂ), Finset.card r = 5 ∧ 
    (∀ z ∈ r, P b z = 0) ∧
    (∀ z ∈ r, -z ∈ r)

/-- The main theorem stating the condition for symmetric roots -/
theorem symmetric_roots_iff_b_eq_two_or_four :
  ∀ b : ℝ, symmetric_roots b ↔ b = 2 ∨ b = 4 := by sorry

end NUMINAMATH_CALUDE_symmetric_roots_iff_b_eq_two_or_four_l2762_276267


namespace NUMINAMATH_CALUDE_ellipse_constants_l2762_276263

/-- An ellipse with given foci and a point on its curve -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  point : ℝ × ℝ

/-- The standard form constants of an ellipse -/
structure EllipseConstants where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

/-- Theorem: For an ellipse with foci at (1, 3) and (1, 7) passing through (12, 0),
    the constants in the standard form equation are as given -/
theorem ellipse_constants (e : Ellipse) 
    (h_focus1 : e.focus1 = (1, 3))
    (h_focus2 : e.focus2 = (1, 7))
    (h_point : e.point = (12, 0)) :
    ∃ (c : EllipseConstants), 
      c.a = (Real.sqrt 130 + Real.sqrt 170) / 2 ∧
      c.b = Real.sqrt (((Real.sqrt 130 + Real.sqrt 170) / 2)^2 - 4^2) ∧
      c.h = 1 ∧
      c.k = 5 ∧
      c.a > 0 ∧
      c.b > 0 ∧
      (e.point.1 - c.h)^2 / c.a^2 + (e.point.2 - c.k)^2 / c.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_constants_l2762_276263


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2762_276294

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem x_eq_2_sufficient_not_necessary :
  let a : ℝ → ℝ × ℝ := λ x ↦ (1, x)
  let b : ℝ → ℝ × ℝ := λ x ↦ (x, 4)
  (∀ x, x = 2 → parallel (a x) (b x)) ∧
  ¬(∀ x, parallel (a x) (b x) → x = 2) :=
by sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l2762_276294


namespace NUMINAMATH_CALUDE_least_seven_binary_digits_l2762_276209

/-- The number of binary digits required to represent a positive integer -/
def binary_digits (n : ℕ+) : ℕ :=
  (Nat.log 2 n.val).succ

/-- Predicate to check if a number has exactly 7 binary digits -/
def has_seven_binary_digits (n : ℕ+) : Prop :=
  binary_digits n = 7

theorem least_seven_binary_digits :
  ∃ (n : ℕ+), has_seven_binary_digits n ∧
    ∀ (m : ℕ+), has_seven_binary_digits m → n ≤ m ∧
    n = 64 := by sorry

end NUMINAMATH_CALUDE_least_seven_binary_digits_l2762_276209


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2762_276269

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x + 1 / y) ≥ (1 / 5 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2762_276269


namespace NUMINAMATH_CALUDE_divisor_problem_l2762_276223

theorem divisor_problem (N D : ℕ) (h1 : N % D = 255) (h2 : (2 * N) % D = 112) : D = 398 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2762_276223


namespace NUMINAMATH_CALUDE_series_one_over_sqrt_n_diverges_l2762_276258

theorem series_one_over_sqrt_n_diverges :
  ¬ Summable (fun n : ℕ => 1 / Real.sqrt n) := by sorry

end NUMINAMATH_CALUDE_series_one_over_sqrt_n_diverges_l2762_276258


namespace NUMINAMATH_CALUDE_problem_statement_l2762_276265

theorem problem_statement :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ∧
  (∀ x : ℝ, 0 < x → x < π / 2 → x > Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2762_276265


namespace NUMINAMATH_CALUDE_acid_solution_volume_l2762_276228

/-- Given a solution with 1.6 litres of pure acid and a concentration of 20%,
    prove that the total volume of the solution is 8 litres. -/
theorem acid_solution_volume (pure_acid : ℝ) (concentration : ℝ) (total_volume : ℝ) 
    (h1 : pure_acid = 1.6)
    (h2 : concentration = 0.2)
    (h3 : pure_acid = concentration * total_volume) : 
  total_volume = 8 := by
sorry

end NUMINAMATH_CALUDE_acid_solution_volume_l2762_276228


namespace NUMINAMATH_CALUDE_remaining_blocks_l2762_276242

/-- The number of blocks Jess must walk to complete her errands and arrive at work. -/
def total_blocks : ℕ := 11 + 6 + 8

/-- The number of blocks Jess has already walked. -/
def walked_blocks : ℕ := 5

/-- Theorem stating the number of remaining blocks Jess must walk. -/
theorem remaining_blocks : total_blocks - walked_blocks = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_blocks_l2762_276242


namespace NUMINAMATH_CALUDE_rectangle_area_l2762_276224

theorem rectangle_area (length width : ℝ) (h1 : length = 20) (h2 : length = 4 * width) : length * width = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2762_276224


namespace NUMINAMATH_CALUDE_michelle_crayon_boxes_l2762_276202

/-- Given Michelle has 35 crayons in total and each box holds 5 crayons, 
    prove that the number of boxes Michelle has is 7. -/
theorem michelle_crayon_boxes 
  (total_crayons : ℕ) 
  (crayons_per_box : ℕ) 
  (h1 : total_crayons = 35)
  (h2 : crayons_per_box = 5) : 
  total_crayons / crayons_per_box = 7 := by
  sorry

#check michelle_crayon_boxes

end NUMINAMATH_CALUDE_michelle_crayon_boxes_l2762_276202


namespace NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l2762_276264

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l2762_276264


namespace NUMINAMATH_CALUDE_christina_bank_transfer_l2762_276288

/-- Calculates the remaining balance after a transfer --/
def remaining_balance (initial : ℕ) (transfer : ℕ) : ℕ :=
  initial - transfer

theorem christina_bank_transfer :
  remaining_balance 27004 69 = 26935 := by
  sorry

end NUMINAMATH_CALUDE_christina_bank_transfer_l2762_276288


namespace NUMINAMATH_CALUDE_expression_simplification_l2762_276275

theorem expression_simplification (a : ℝ) (h : a = 2 * Real.cos (π / 3) + 1) :
  (a - a^2 / (a + 1)) / (a^2 / (a^2 - 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2762_276275


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2762_276298

/-- The modulus of the complex number z = (1+3i)/(1-i) is equal to √5 -/
theorem modulus_of_complex_fraction : 
  let z : ℂ := (1 + 3*I) / (1 - I)
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2762_276298


namespace NUMINAMATH_CALUDE_billy_watched_79_videos_l2762_276245

/-- The number of videos Billy watches before finding one he likes -/
def total_videos_watched (suggestions_per_attempt : ℕ) (unsuccessful_attempts : ℕ) (position_of_liked_video : ℕ) : ℕ :=
  suggestions_per_attempt * unsuccessful_attempts + (position_of_liked_video - 1)

/-- Theorem stating that Billy watches 79 videos before finding one he likes -/
theorem billy_watched_79_videos :
  total_videos_watched 15 5 5 = 79 := by
sorry

end NUMINAMATH_CALUDE_billy_watched_79_videos_l2762_276245


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2762_276292

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x > 0 ∧ (1057 + x) % 23 = 0 ∧ ∀ y : ℕ, y > 0 ∧ (1057 + y) % 23 = 0 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2762_276292


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2762_276279

/-- Two 2D vectors are parallel if the cross product of their coordinates is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two vectors a and b, prove that if they are parallel, then x = 3 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, x)
  let b : ℝ × ℝ := (2, x - 1)
  are_parallel a b → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2762_276279


namespace NUMINAMATH_CALUDE_min_draw_for_eight_same_color_l2762_276217

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat
  white : Nat

/-- The minimum number of balls to draw to ensure at least n of the same color -/
def minDrawToEnsure (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The theorem to prove -/
theorem min_draw_for_eight_same_color (counts : BallCounts)
    (h_red : counts.red = 15)
    (h_green : counts.green = 12)
    (h_blue : counts.blue = 10)
    (h_yellow : counts.yellow = 7)
    (h_white : counts.white = 6)
    (h_total : counts.red + counts.green + counts.blue + counts.yellow + counts.white = 50) :
    minDrawToEnsure counts 8 = 35 := by
  sorry

end NUMINAMATH_CALUDE_min_draw_for_eight_same_color_l2762_276217


namespace NUMINAMATH_CALUDE_unique_coin_distribution_l2762_276210

/-- A structure representing the coin distribution in the piggy bank -/
structure CoinDistribution where
  one_ruble : ℕ
  two_rubles : ℕ
  five_rubles : ℕ

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- Theorem stating the unique solution to the coin distribution problem -/
theorem unique_coin_distribution : 
  ∃! (d : CoinDistribution), 
    d.one_ruble + d.two_rubles + d.five_rubles = 1000 ∧ 
    d.one_ruble + 2 * d.two_rubles + 5 * d.five_rubles = 2000 ∧
    is_prime d.one_ruble ∧
    d.one_ruble = 3 ∧ d.two_rubles = 996 ∧ d.five_rubles = 1 := by
  sorry


end NUMINAMATH_CALUDE_unique_coin_distribution_l2762_276210


namespace NUMINAMATH_CALUDE_daves_remaining_apps_l2762_276208

/-- Represents the number of apps and files on Dave's phone -/
structure PhoneContent where
  apps : ℕ
  files : ℕ

/-- The initial state of Dave's phone -/
def initial : PhoneContent := { apps := 11, files := 3 }

/-- The final state of Dave's phone after deletion -/
def final : PhoneContent := { apps := 2, files := 24 }

/-- Theorem stating that the final number of apps on Dave's phone is 2 -/
theorem daves_remaining_apps :
  final.apps = 2 ∧
  final.files = 24 ∧
  final.files = final.apps + 22 :=
by sorry

end NUMINAMATH_CALUDE_daves_remaining_apps_l2762_276208


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l2762_276211

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate1 germination_rate2 : ℚ) : 
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate1 = 1/5 →
  germination_rate2 = 7/20 →
  (((seeds_plot1 : ℚ) * germination_rate1 + (seeds_plot2 : ℚ) * germination_rate2) / 
   ((seeds_plot1 : ℚ) + (seeds_plot2 : ℚ))) = 13/50 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l2762_276211


namespace NUMINAMATH_CALUDE_mo_tea_consumption_l2762_276295

/-- Represents Mo's drinking habits and weather conditions for a week -/
structure MoDrinkingHabits where
  n : ℕ  -- number of hot chocolate cups on rainy mornings
  t : ℕ  -- number of tea cups on non-rainy mornings
  rainyDays : ℕ
  nonRainyDays : ℕ

/-- Theorem stating Mo's tea consumption on non-rainy mornings -/
theorem mo_tea_consumption (habits : MoDrinkingHabits) : habits.t = 4 :=
  by
  have h1 : habits.rainyDays = 2 := by sorry
  have h2 : habits.nonRainyDays = 7 - habits.rainyDays := by sorry
  have h3 : habits.n * habits.rainyDays + habits.t * habits.nonRainyDays = 26 := by sorry
  have h4 : habits.t * habits.nonRainyDays = habits.n * habits.rainyDays + 14 := by sorry
  sorry

#check mo_tea_consumption

end NUMINAMATH_CALUDE_mo_tea_consumption_l2762_276295


namespace NUMINAMATH_CALUDE_horner_method_correct_l2762_276274

def horner_polynomial (x : ℚ) : ℚ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v4 (x : ℚ) : ℚ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 79
  v3 * x - 8

theorem horner_method_correct :
  horner_v4 (-4) = 220 :=
sorry

end NUMINAMATH_CALUDE_horner_method_correct_l2762_276274


namespace NUMINAMATH_CALUDE_complement_of_A_l2762_276234

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x + 2 > 4}

theorem complement_of_A : Set.compl A = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2762_276234


namespace NUMINAMATH_CALUDE_min_m_and_x_range_l2762_276235

theorem min_m_and_x_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + 3*b^2 = 3) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → Real.sqrt 5 * a + b ≤ m) ∧
            (∀ m' : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → Real.sqrt 5 * a + b ≤ m') → m ≤ m') ∧
            m = 4) ∧
  (∀ x : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → 2 * |x - 1| + |x| ≥ Real.sqrt 5 * a + b) →
            (x ≤ -2/3 ∨ x ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_m_and_x_range_l2762_276235


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2762_276214

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 512 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l2762_276214


namespace NUMINAMATH_CALUDE_circle_center_reflection_l2762_276253

/-- Reflects a point (x, y) about the line y = x -/
def reflect_about_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem circle_center_reflection :
  let original_center : ℝ × ℝ := (8, -3)
  reflect_about_y_equals_x original_center = (-3, 8) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_reflection_l2762_276253


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2762_276215

theorem geometric_sequence_sum (a : ℝ) : 
  (a + 2*a + 4*a + 8*a = 1) →  -- Sum of first 4 terms equals 1
  (a + 2*a + 4*a + 8*a + 16*a + 32*a + 64*a + 128*a = 17) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2762_276215
