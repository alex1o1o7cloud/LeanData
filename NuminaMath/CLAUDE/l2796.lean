import Mathlib

namespace NUMINAMATH_CALUDE_point_divides_segment_l2796_279611

/-- Given two points A and B in 2D space, and a point P, this function checks if P divides the line segment AB in the given ratio m:n -/
def divides_segment (A B P : ℝ × ℝ) (m n : ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x, y) := P
  x = (m * x₂ + n * x₁) / (m + n) ∧
  y = (m * y₂ + n * y₁) / (m + n)

/-- The theorem states that the point (3.5, 8.5) divides the line segment between (2, 10) and (8, 4) in the ratio 1:3 -/
theorem point_divides_segment :
  divides_segment (2, 10) (8, 4) (3.5, 8.5) 1 3 := by
  sorry

end NUMINAMATH_CALUDE_point_divides_segment_l2796_279611


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2796_279657

def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_pos_geo : is_positive_geometric_sequence a)
  (h_eq : 1 / (a 2 * a 4) + 2 / (a 4 * a 4) + 1 / (a 4 * a 6) = 81) :
  1 / a 3 + 1 / a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2796_279657


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l2796_279629

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers are consecutive primes -/
def areConsecutivePrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ b = a + 2 ∧ c = b + 2

/-- The theorem stating the smallest perimeter of a scalene triangle with consecutive prime side lengths and prime perimeter -/
theorem smallest_prime_perimeter_triangle :
  ∀ a b c : ℕ,
    areConsecutivePrimes a b c →
    isPrime (a + b + c) →
    a + b + c ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l2796_279629


namespace NUMINAMATH_CALUDE_total_difference_is_90q_minus_250_l2796_279680

/-- The total difference in money between Charles and Richard in cents -/
def total_difference (q : ℤ) : ℤ :=
  let charles_quarters := 6 * q + 2
  let charles_dimes := 3 * q - 2
  let richard_quarters := 2 * q + 10
  let richard_dimes := 4 * q + 3
  let quarter_value := 25
  let dime_value := 10
  (charles_quarters - richard_quarters) * quarter_value + 
  (charles_dimes - richard_dimes) * dime_value

theorem total_difference_is_90q_minus_250 (q : ℤ) : 
  total_difference q = 90 * q - 250 := by
  sorry

end NUMINAMATH_CALUDE_total_difference_is_90q_minus_250_l2796_279680


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2796_279691

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2796_279691


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l2796_279659

theorem largest_divisor_of_product (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) = 480 * k) ∧
  (∀ (m : ℕ), m > 480 → ∃ (n : ℕ), Odd n ∧ ¬(∃ (k : ℕ), (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) = m * k)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l2796_279659


namespace NUMINAMATH_CALUDE_chess_game_probability_l2796_279653

theorem chess_game_probability (prob_draw prob_B_win : ℚ) 
  (h1 : prob_draw = 1/2) 
  (h2 : prob_B_win = 1/3) : 
  1 - prob_draw - prob_B_win = 1/6 := by
sorry

end NUMINAMATH_CALUDE_chess_game_probability_l2796_279653


namespace NUMINAMATH_CALUDE_class_8_3_final_score_l2796_279638

/-- The final score of a choir competition is calculated based on three categories:
    singing quality, spirit, and coordination. Each category has a specific weight
    in the final score calculation. -/
def final_score (singing_quality : ℝ) (spirit : ℝ) (coordination : ℝ)
                (singing_weight : ℝ) (spirit_weight : ℝ) (coordination_weight : ℝ) : ℝ :=
  singing_quality * singing_weight + spirit * spirit_weight + coordination * coordination_weight

/-- Theorem stating that the final score of Class 8-3 in the choir competition is 81.8 points -/
theorem class_8_3_final_score :
  final_score 92 80 70 0.4 0.3 0.3 = 81.8 := by
  sorry

end NUMINAMATH_CALUDE_class_8_3_final_score_l2796_279638


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2796_279610

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, (0 < x ∧ x < 2) → (x^2 - x - 2 < 0)) ∧ 
  (∃ x, (x^2 - x - 2 < 0) ∧ ¬(0 < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2796_279610


namespace NUMINAMATH_CALUDE_other_number_from_hcf_lcm_and_one_number_l2796_279641

/-- Given two positive integers with known HCF, LCM, and one of the numbers,
    prove that the other number is as calculated. -/
theorem other_number_from_hcf_lcm_and_one_number
  (a b : ℕ+) 
  (hcf : Nat.gcd a b = 16)
  (lcm : Nat.lcm a b = 396)
  (ha : a = 36) :
  b = 176 := by
  sorry

end NUMINAMATH_CALUDE_other_number_from_hcf_lcm_and_one_number_l2796_279641


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2796_279634

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt x) / 19 = 4 ∧ x = 5776 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2796_279634


namespace NUMINAMATH_CALUDE_least_possible_third_side_l2796_279628

theorem least_possible_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a = 8 → b = 15 →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_third_side_l2796_279628


namespace NUMINAMATH_CALUDE_constant_expression_l2796_279642

theorem constant_expression (x : ℝ) (h : x ≥ 4/7) :
  -4*x + |4 - 7*x| - |1 - 3*x| + 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_constant_expression_l2796_279642


namespace NUMINAMATH_CALUDE_monthly_income_A_l2796_279672

/-- Given the average monthly incomes of pairs of individuals, prove the monthly income of A. -/
theorem monthly_income_A (income_AB income_BC income_AC : ℚ) 
  (h1 : (income_A + income_B) / 2 = 5050)
  (h2 : (income_B + income_C) / 2 = 6250)
  (h3 : (income_A + income_C) / 2 = 5200)
  : income_A = 4000 := by
  sorry

where
  income_A : ℚ := sorry
  income_B : ℚ := sorry
  income_C : ℚ := sorry

end NUMINAMATH_CALUDE_monthly_income_A_l2796_279672


namespace NUMINAMATH_CALUDE_disc_purchase_problem_l2796_279600

theorem disc_purchase_problem (price_a price_b total_spent : ℚ) (num_b : ℕ) :
  price_a = 21/2 ∧ 
  price_b = 17/2 ∧ 
  total_spent = 93 ∧ 
  num_b = 6 →
  ∃ (num_a : ℕ), num_a + num_b = 10 ∧ 
    num_a * price_a + num_b * price_b = total_spent :=
by sorry

end NUMINAMATH_CALUDE_disc_purchase_problem_l2796_279600


namespace NUMINAMATH_CALUDE_print_output_l2796_279602

-- Define a simple output function to represent PRINT
def print (a : ℕ) (b : ℕ) : String :=
  s!"{a}, {b}"

-- Theorem statement
theorem print_output : print 3 (3 + 2) = "3, 5" := by
  sorry

end NUMINAMATH_CALUDE_print_output_l2796_279602


namespace NUMINAMATH_CALUDE_square_difference_formula_l2796_279692

theorem square_difference_formula : 30^2 - 2*(30*5) + 5^2 = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l2796_279692


namespace NUMINAMATH_CALUDE_total_amount_pigs_and_hens_l2796_279612

/-- The total amount spent on buying pigs and hens -/
def total_amount (num_pigs : ℕ) (num_hens : ℕ) (price_pig : ℕ) (price_hen : ℕ) : ℕ :=
  num_pigs * price_pig + num_hens * price_hen

/-- Theorem stating that the total amount spent on 3 pigs at Rs. 300 each and 10 hens at Rs. 30 each is Rs. 1200 -/
theorem total_amount_pigs_and_hens :
  total_amount 3 10 300 30 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_pigs_and_hens_l2796_279612


namespace NUMINAMATH_CALUDE_square_of_complex_is_real_implies_m_is_plus_minus_one_l2796_279689

theorem square_of_complex_is_real_implies_m_is_plus_minus_one (m : ℝ) :
  (∃ (r : ℝ), (m + Complex.I)^2 = r) → (m = 1 ∨ m = -1) := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_is_real_implies_m_is_plus_minus_one_l2796_279689


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2796_279677

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2796_279677


namespace NUMINAMATH_CALUDE_evergreen_marching_band_max_size_l2796_279669

theorem evergreen_marching_band_max_size :
  ∃ (n : ℕ),
    (∀ k : ℕ, 15 * k < 800 → 15 * k ≤ 15 * n) ∧
    (15 * n < 800) ∧
    (15 * n % 19 = 2) ∧
    (15 * n = 750) := by
  sorry

end NUMINAMATH_CALUDE_evergreen_marching_band_max_size_l2796_279669


namespace NUMINAMATH_CALUDE_paige_folders_proof_l2796_279614

def number_of_folders (initial_files deleted_files files_per_folder : ℕ) : ℕ :=
  (initial_files - deleted_files) / files_per_folder

theorem paige_folders_proof (initial_files deleted_files files_per_folder : ℕ) 
  (h1 : initial_files = 27)
  (h2 : deleted_files = 9)
  (h3 : files_per_folder = 6)
  : number_of_folders initial_files deleted_files files_per_folder = 3 := by
  sorry

#eval number_of_folders 27 9 6

end NUMINAMATH_CALUDE_paige_folders_proof_l2796_279614


namespace NUMINAMATH_CALUDE_min_output_no_loss_l2796_279697

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the sales revenue function
def sales_revenue (x : ℝ) : ℝ := 25 * x

-- Define the domain constraint
def in_domain (x : ℝ) : Prop := 0 < x ∧ x < 240

-- Theorem statement
theorem min_output_no_loss :
  ∃ (x_min : ℝ), x_min = 150 ∧
  in_domain x_min ∧
  (∀ x : ℝ, in_domain x → sales_revenue x ≥ total_cost x → x ≥ x_min) :=
sorry

end NUMINAMATH_CALUDE_min_output_no_loss_l2796_279697


namespace NUMINAMATH_CALUDE_third_difference_of_cubic_is_six_l2796_279648

/-- Finite difference operator -/
def finiteDifference (f : ℕ → ℝ) : ℕ → ℝ := fun n ↦ f (n + 1) - f n

/-- Third finite difference -/
def thirdFiniteDifference (f : ℕ → ℝ) : ℕ → ℝ :=
  finiteDifference (finiteDifference (finiteDifference f))

/-- Cubic function -/
def cubicFunction : ℕ → ℝ := fun n ↦ (n : ℝ) ^ 3

theorem third_difference_of_cubic_is_six :
  ∀ n, thirdFiniteDifference cubicFunction n = 6 := by sorry

end NUMINAMATH_CALUDE_third_difference_of_cubic_is_six_l2796_279648


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_three_digits_l2796_279626

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3 ∨ d = 6

def contains_all_required_digits (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 3 ∈ n.digits 10 ∧ 6 ∈ n.digits 10

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem smallest_valid_number_last_three_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 2 = 0 ∧
    m % 3 = 0 ∧
    is_valid_number m ∧
    contains_all_required_digits m ∧
    (∀ k : ℕ, k > 0 ∧ k % 2 = 0 ∧ k % 3 = 0 ∧ is_valid_number k ∧ contains_all_required_digits k → m ≤ k) ∧
    last_three_digits m = 326 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_three_digits_l2796_279626


namespace NUMINAMATH_CALUDE_max_pencils_is_seven_l2796_279690

/-- The maximum number of pencils Alice can purchase given the conditions --/
def max_pencils : ℕ :=
  let pin_cost : ℕ := 3
  let pen_cost : ℕ := 4
  let pencil_cost : ℕ := 9
  let total_budget : ℕ := 72
  let min_purchase : ℕ := pin_cost + pen_cost
  let remaining_budget : ℕ := total_budget - min_purchase
  remaining_budget / pencil_cost

/-- Theorem stating that the maximum number of pencils Alice can purchase is 7 --/
theorem max_pencils_is_seven : max_pencils = 7 := by
  sorry

#eval max_pencils -- This will evaluate to 7

end NUMINAMATH_CALUDE_max_pencils_is_seven_l2796_279690


namespace NUMINAMATH_CALUDE_boat_speed_is_54_l2796_279682

/-- Represents the speed of a boat in still water -/
def boat_speed (v : ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧
  (v - 18) * (2 * t) = (v + 18) * t

/-- Theorem: The speed of the boat in still water is 54 kmph -/
theorem boat_speed_is_54 : boat_speed 54 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_is_54_l2796_279682


namespace NUMINAMATH_CALUDE_probability_no_adjacent_birch_is_two_forty_fifths_l2796_279671

def total_trees : ℕ := 15
def birch_trees : ℕ := 6
def non_birch_trees : ℕ := 9

def probability_no_adjacent_birch : ℚ :=
  (Nat.choose (non_birch_trees + 1) birch_trees) / (Nat.choose total_trees birch_trees)

theorem probability_no_adjacent_birch_is_two_forty_fifths :
  probability_no_adjacent_birch = 2 / 45 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_adjacent_birch_is_two_forty_fifths_l2796_279671


namespace NUMINAMATH_CALUDE_apple_prices_l2796_279635

/-- The unit price of Zhao Tong apples in yuan per jin -/
def zhao_tong_price : ℚ := 5

/-- The unit price of Tianma from Xiaocaoba in yuan per jin -/
def tianma_price : ℚ := 50

/-- The original purchase price of apples in yuan per jin -/
def original_purchase_price : ℚ := 4

/-- The cost of 1 jin of Zhao Tong apples and 2 jin of Tianma from Xiaocaoba -/
def cost1 : ℚ := 105

/-- The cost of 3 jin of Zhao Tong apples and 5 jin of Tianma from Xiaocaoba -/
def cost2 : ℚ := 265

/-- The original cost for transporting apples -/
def original_transport_cost : ℚ := 240

/-- The new cost for transporting apples -/
def new_transport_cost : ℚ := 300

/-- The increase in purchase price -/
def price_increase : ℚ := 1

theorem apple_prices :
  (zhao_tong_price + 2 * tianma_price = cost1) ∧
  (3 * zhao_tong_price + 5 * tianma_price = cost2) ∧
  (original_transport_cost / original_purchase_price = new_transport_cost / (original_purchase_price + price_increase)) := by
  sorry

end NUMINAMATH_CALUDE_apple_prices_l2796_279635


namespace NUMINAMATH_CALUDE_equation_solution_l2796_279687

theorem equation_solution :
  let y : ℝ := 1 + Real.sqrt (19 / 3)
  3 / y - (4 / y) * (2 / y) = (3 / 2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2796_279687


namespace NUMINAMATH_CALUDE_line_slope_one_implies_a_equals_one_l2796_279695

/-- Given a line passing through points (-2, a) and (a, 4) with slope 1, prove that a = 1 -/
theorem line_slope_one_implies_a_equals_one (a : ℝ) :
  (4 - a) / (a + 2) = 1 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_one_implies_a_equals_one_l2796_279695


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l2796_279670

/-- The gain percent on a cycle sale --/
theorem cycle_gain_percent (cost_price selling_price : ℝ) 
  (h1 : cost_price = 900)
  (h2 : selling_price = 1180) : 
  (selling_price - cost_price) / cost_price * 100 = 31.11 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l2796_279670


namespace NUMINAMATH_CALUDE_sum_of_roots_l2796_279649

theorem sum_of_roots (x : ℝ) : 
  (x^2 = 9*x - 20) → (∃ y : ℝ, y^2 = 9*y - 20 ∧ x + y = 9) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2796_279649


namespace NUMINAMATH_CALUDE_triangle_areas_equal_l2796_279618

theorem triangle_areas_equal :
  let a : ℝ := 24
  let b : ℝ := 24
  let c : ℝ := 34
  let right_triangle_area := (1/2) * a * b
  let s := (a + b + c) / 2
  let general_triangle_area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  right_triangle_area = general_triangle_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_areas_equal_l2796_279618


namespace NUMINAMATH_CALUDE_min_value_of_f_l2796_279637

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else x + 6/x - 7

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 2 * Real.sqrt 6 - 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2796_279637


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2796_279693

theorem inequality_solution_set (k : ℝ) :
  let S := {x : ℝ | k * x^2 - (k + 2) * x + 2 < 0}
  (k = 0 → S = {x : ℝ | x < 1}) ∧
  (0 < k ∧ k < 2 → S = {x : ℝ | x < 1 ∨ x > 2/k}) ∧
  (k = 2 → S = {x : ℝ | x ≠ 1}) ∧
  (k > 2 → S = {x : ℝ | x < 2/k ∨ x > 1}) ∧
  (k < 0 → S = {x : ℝ | 2/k < x ∧ x < 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2796_279693


namespace NUMINAMATH_CALUDE_totalLives_eq_110_l2796_279664

/-- The total number of lives for remaining players after some quit and bonus lives are added -/
def totalLives : ℕ :=
  let initialPlayers : ℕ := 16
  let quitPlayers : ℕ := 7
  let remainingPlayers : ℕ := initialPlayers - quitPlayers
  let playersWithTenLives : ℕ := 3
  let playersWithEightLives : ℕ := 4
  let playersWithSixLives : ℕ := 2
  let bonusLives : ℕ := 4
  
  let livesBeforeBonus : ℕ := 
    playersWithTenLives * 10 + 
    playersWithEightLives * 8 + 
    playersWithSixLives * 6
  
  let totalBonusLives : ℕ := remainingPlayers * bonusLives
  
  livesBeforeBonus + totalBonusLives

theorem totalLives_eq_110 : totalLives = 110 := by
  sorry

end NUMINAMATH_CALUDE_totalLives_eq_110_l2796_279664


namespace NUMINAMATH_CALUDE_inequality_proof_l2796_279617

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2796_279617


namespace NUMINAMATH_CALUDE_english_speakers_l2796_279613

theorem english_speakers (total : ℕ) (hindi : ℕ) (both : ℕ) (english : ℕ) : 
  total = 40 → 
  hindi = 30 → 
  both ≥ 10 → 
  total = hindi + english - both → 
  english = 20 := by
sorry

end NUMINAMATH_CALUDE_english_speakers_l2796_279613


namespace NUMINAMATH_CALUDE_banquet_arrangement_theorem_l2796_279673

/-- Represents the problem of arranging tables for a banquet. -/
structure BanquetArrangement where
  guests : Nat
  initial_tables : Nat
  seats_per_table : Nat

/-- Calculates the minimum number of tables needed for the banquet arrangement. -/
def min_tables_needed (arrangement : BanquetArrangement) : Nat :=
  sorry

/-- Theorem stating that for the given banquet arrangement, 11 tables are needed. -/
theorem banquet_arrangement_theorem :
  let arrangement : BanquetArrangement := {
    guests := 44,
    initial_tables := 15,
    seats_per_table := 4
  }
  min_tables_needed arrangement = 11 := by sorry

end NUMINAMATH_CALUDE_banquet_arrangement_theorem_l2796_279673


namespace NUMINAMATH_CALUDE_line_intercept_sum_l2796_279660

/-- Given a line with equation 3x + 5y + k = 0, where the sum of its x-intercept and y-intercept is 16, prove that k = -30. -/
theorem line_intercept_sum (k : ℝ) : 
  (∃ (x y : ℝ), 3 * x + 5 * y + k = 0 ∧ 
   (3 * 0 + 5 * y + k = 0 → 3 * x + 5 * 0 + k = 0 → x + y = 16)) → 
  k = -30 := by
sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l2796_279660


namespace NUMINAMATH_CALUDE_area_triangle_dbc_l2796_279679

/-- Given a triangle ABC with vertices A(0,8), B(0,0), C(10,0), and midpoints D of AB and E of BC,
    the area of triangle DBC is 20. -/
theorem area_triangle_dbc (A B C D E : ℝ × ℝ) : 
  A = (0, 8) → 
  B = (0, 0) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  (1 / 2 : ℝ) * 10 * 4 = 20 := by
sorry

end NUMINAMATH_CALUDE_area_triangle_dbc_l2796_279679


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2796_279604

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 > 0 →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2796_279604


namespace NUMINAMATH_CALUDE_multiplication_mistake_correction_l2796_279621

theorem multiplication_mistake_correction (α : ℝ) :
  1.2 * α = 1.23 * α - 0.3 → 1.23 * α = 111 := by
sorry

end NUMINAMATH_CALUDE_multiplication_mistake_correction_l2796_279621


namespace NUMINAMATH_CALUDE_disco_probabilities_l2796_279603

/-- Represents the content of a music case -/
structure MusicCase where
  disco : ℕ
  techno : ℕ

/-- The probability of selecting a disco tape from a given music case -/
def prob_disco (case : MusicCase) : ℚ :=
  case.disco / (case.disco + case.techno)

/-- The probability of selecting a second disco tape when the first is returned -/
def prob_disco_returned (case : MusicCase) : ℚ :=
  prob_disco case

/-- The probability of selecting a second disco tape when the first is not returned -/
def prob_disco_not_returned (case : MusicCase) : ℚ :=
  (case.disco - 1) / (case.disco + case.techno - 1)

/-- Theorem stating the probabilities for the given scenario -/
theorem disco_probabilities (case : MusicCase) (h : case = ⟨20, 10⟩) :
  prob_disco case = 2/3 ∧
  prob_disco_returned case = 2/3 ∧
  prob_disco_not_returned case = 19/29 := by
  sorry


end NUMINAMATH_CALUDE_disco_probabilities_l2796_279603


namespace NUMINAMATH_CALUDE_geometric_increasing_condition_l2796_279606

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The condition "q > 1" is neither sufficient nor necessary for a geometric sequence to be increasing -/
theorem geometric_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  ¬(((q > 1) → increasing_sequence a) ∧ (increasing_sequence a → (q > 1))) :=
sorry

end NUMINAMATH_CALUDE_geometric_increasing_condition_l2796_279606


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l2796_279605

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the symmetric point relation
def symmetric_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- Midpoint of (x₁, y₁) and (x₂, y₂) lies on the line of symmetry
  line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ∧
  -- The line connecting (x₁, y₁) and (x₂, y₂) is perpendicular to the line of symmetry
  (y₂ - y₁) = (x₂ - x₁)

-- Theorem statement
theorem symmetric_point_theorem :
  symmetric_point 2 1 (-2) (-3) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l2796_279605


namespace NUMINAMATH_CALUDE_correct_statements_l2796_279622

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| GeneralToGeneral
| PartToWhole
| GeneralToSpecific
| SpecificToSpecific
| SpecificToGeneral

-- Define a function to describe the correct direction for each reasoning type
def correct_direction (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Define the statements
def statement (n : Nat) : ReasoningType × ReasoningDirection :=
  match n with
  | 1 => (ReasoningType.Inductive, ReasoningDirection.GeneralToGeneral)
  | 2 => (ReasoningType.Inductive, ReasoningDirection.PartToWhole)
  | 3 => (ReasoningType.Deductive, ReasoningDirection.GeneralToSpecific)
  | 4 => (ReasoningType.Analogical, ReasoningDirection.SpecificToSpecific)
  | 5 => (ReasoningType.Analogical, ReasoningDirection.SpecificToGeneral)
  | _ => (ReasoningType.Inductive, ReasoningDirection.PartToWhole) -- Default case

-- Define a function to check if a statement is correct
def is_correct (n : Nat) : Prop :=
  let (rt, rd) := statement n
  rd = correct_direction rt

-- Theorem stating that statements 2, 3, and 4 are the correct ones
theorem correct_statements :
  (is_correct 2 ∧ is_correct 3 ∧ is_correct 4) ∧
  (∀ n, n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 → ¬is_correct n) :=
sorry


end NUMINAMATH_CALUDE_correct_statements_l2796_279622


namespace NUMINAMATH_CALUDE_ben_has_fifteen_shirts_l2796_279624

/-- The number of new shirts Alex has -/
def alex_shirts : ℕ := 4

/-- The number of additional shirts Joe has compared to Alex -/
def joe_extra_shirts : ℕ := 3

/-- The number of additional shirts Ben has compared to Joe -/
def ben_extra_shirts : ℕ := 8

/-- The number of new shirts Joe has -/
def joe_shirts : ℕ := alex_shirts + joe_extra_shirts

/-- The number of new shirts Ben has -/
def ben_shirts : ℕ := joe_shirts + ben_extra_shirts

theorem ben_has_fifteen_shirts : ben_shirts = 15 := by
  sorry

end NUMINAMATH_CALUDE_ben_has_fifteen_shirts_l2796_279624


namespace NUMINAMATH_CALUDE_alices_favorite_number_l2796_279668

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem alices_favorite_number :
  ∃! n : ℕ,
    100 < n ∧ n < 200 ∧
    n % 13 = 0 ∧
    n % 3 ≠ 0 ∧
    (sum_of_digits n) % 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l2796_279668


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2796_279640

theorem hemisphere_surface_area (r : ℝ) (h : r > 0) : 
  π * r^2 = 3 → 2 * π * r^2 + π * r^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2796_279640


namespace NUMINAMATH_CALUDE_katherine_bottle_caps_l2796_279652

def initial_bottle_caps : ℕ := 34
def eaten_bottle_caps : ℕ := 8
def remaining_bottle_caps : ℕ := 26

theorem katherine_bottle_caps :
  initial_bottle_caps = eaten_bottle_caps + remaining_bottle_caps :=
by sorry

end NUMINAMATH_CALUDE_katherine_bottle_caps_l2796_279652


namespace NUMINAMATH_CALUDE_escalator_time_l2796_279608

theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) :
  escalator_speed = 8 →
  person_speed = 2 →
  escalator_length = 160 →
  escalator_length / (escalator_speed + person_speed) = 16 := by
  sorry

end NUMINAMATH_CALUDE_escalator_time_l2796_279608


namespace NUMINAMATH_CALUDE_max_sum_is_38_l2796_279666

/-- Represents the setup of numbers in the grid -/
structure Grid :=
  (a b c d e : ℕ)

/-- The set of available numbers -/
def availableNumbers : Finset ℕ := {2, 3, 8, 9, 14, 15}

/-- Checks if the grid satisfies the equality condition -/
def isValidGrid (g : Grid) : Prop :=
  g.a + g.b + g.e = g.a + g.c + g.e ∧
  g.a + g.c + g.e = g.b + g.d + g.e

/-- Checks if the grid uses numbers from the available set -/
def usesAvailableNumbers (g : Grid) : Prop :=
  g.a ∈ availableNumbers ∧
  g.b ∈ availableNumbers ∧
  g.c ∈ availableNumbers ∧
  g.d ∈ availableNumbers ∧
  g.e ∈ availableNumbers

/-- Calculates the sum of the grid -/
def gridSum (g : Grid) : ℕ := g.a + g.b + g.e

/-- Theorem: The maximum sum of a valid grid using the available numbers is 38 -/
theorem max_sum_is_38 :
  ∃ (g : Grid), isValidGrid g ∧ usesAvailableNumbers g ∧
  (∀ (h : Grid), isValidGrid h ∧ usesAvailableNumbers h → gridSum h ≤ gridSum g) ∧
  gridSum g = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_is_38_l2796_279666


namespace NUMINAMATH_CALUDE_equation_solution_l2796_279676

theorem equation_solution :
  let x : ℝ := 32
  let equation (number : ℝ) := 35 - (23 - (15 - x)) = 12 * 2 / (1 / number)
  ∃ (number : ℝ), equation number ∧ number = -4.8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2796_279676


namespace NUMINAMATH_CALUDE_erased_number_problem_l2796_279650

theorem erased_number_problem (n : Nat) (x : Nat) : 
  n = 69 → 
  x ≤ n →
  x ≥ 1 →
  (((n * (n + 1)) / 2 - x) : ℚ) / (n - 1 : ℚ) = 35 + (7 : ℚ) / 17 →
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_problem_l2796_279650


namespace NUMINAMATH_CALUDE_phi_value_for_symmetric_sine_l2796_279644

theorem phi_value_for_symmetric_sine (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  f (π / 3) = 1 / 2 →
  f (-π / 3) = 1 / 2 →
  ∃ k : ℤ, φ = 2 * k * π - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_for_symmetric_sine_l2796_279644


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l2796_279636

def is_arithmetic (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, s (n + 1) = s n + d

def seq1 (n : ℕ) : ℚ := n + 4
def seq2 (n : ℕ) : ℚ := if n % 2 = 0 then 3 - 3 * (n / 2) else 0
def seq3 (n : ℕ) : ℚ := 0
def seq4 (n : ℕ) : ℚ := (n + 1) / 10

theorem arithmetic_sequence_count :
  (is_arithmetic seq1) ∧
  (¬ is_arithmetic seq2) ∧
  (is_arithmetic seq3) ∧
  (is_arithmetic seq4) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l2796_279636


namespace NUMINAMATH_CALUDE_students_in_lunchroom_l2796_279646

theorem students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end NUMINAMATH_CALUDE_students_in_lunchroom_l2796_279646


namespace NUMINAMATH_CALUDE_income_p_is_3000_l2796_279698

/-- The monthly income of three people given their pairwise averages -/
def monthly_income (avg_pq avg_qr avg_pr : ℚ) : ℚ × ℚ × ℚ :=
  let p := 2 * (avg_pq + avg_pr - avg_qr)
  let q := 2 * (avg_pq + avg_qr - avg_pr)
  let r := 2 * (avg_qr + avg_pr - avg_pq)
  (p, q, r)

theorem income_p_is_3000 (avg_pq avg_qr avg_pr : ℚ) :
  avg_pq = 2050 → avg_qr = 5250 → avg_pr = 6200 →
  (monthly_income avg_pq avg_qr avg_pr).1 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_income_p_is_3000_l2796_279698


namespace NUMINAMATH_CALUDE_rowing_speed_l2796_279675

theorem rowing_speed (downstream_distance upstream_distance : ℝ)
                     (total_time : ℝ)
                     (current_speed : ℝ)
                     (h1 : downstream_distance = 3.5)
                     (h2 : upstream_distance = 3.5)
                     (h3 : total_time = 5/3)
                     (h4 : current_speed = 2) :
  ∃ still_water_speed : ℝ,
    still_water_speed = 5 ∧
    downstream_distance / (still_water_speed + current_speed) +
    upstream_distance / (still_water_speed - current_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_rowing_speed_l2796_279675


namespace NUMINAMATH_CALUDE_managers_salary_l2796_279620

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (avg_increase : ℝ) : 
  num_employees = 20 → 
  avg_salary = 1500 → 
  avg_increase = 1000 → 
  (num_employees * avg_salary + (num_employees + 1) * avg_increase) / (num_employees + 1) - avg_salary = 22500 :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l2796_279620


namespace NUMINAMATH_CALUDE_tinas_pens_l2796_279683

theorem tinas_pens (pink green blue : ℕ) : 
  green = pink - 9 →
  blue = green + 3 →
  pink + green + blue = 21 →
  pink = 12 := by
sorry

end NUMINAMATH_CALUDE_tinas_pens_l2796_279683


namespace NUMINAMATH_CALUDE_m_range_for_three_integer_solutions_l2796_279630

def inequality_system (x m : ℝ) : Prop :=
  2 * x - 1 ≤ 5 ∧ x - m > 0

def has_three_integer_solutions (m : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℤ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    inequality_system x₁ m ∧ inequality_system x₂ m ∧ inequality_system x₃ m ∧
    ∀ x : ℤ, inequality_system x m → x = x₁ ∨ x = x₂ ∨ x = x₃

theorem m_range_for_three_integer_solutions :
  ∀ m : ℝ, has_three_integer_solutions m ↔ 0 ≤ m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_m_range_for_three_integer_solutions_l2796_279630


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2796_279609

theorem quadratic_equation_solution (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x - 2 = 0 ∧ (x + 1) / (x - 1) = 3) → 
  k = -1 ∧ ∃ y : ℝ, y ≠ 2 ∧ y^2 + k*y - 2 = 0 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2796_279609


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_l2796_279623

theorem complex_modulus_sqrt (z : ℂ) (h : z^2 = -15 + 8*I) : Complex.abs z = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_l2796_279623


namespace NUMINAMATH_CALUDE_election_votes_theorem_l2796_279656

/-- Represents an election with two candidates -/
structure Election where
  total_votes : ℕ
  winning_percentage : ℚ
  vote_majority : ℕ

/-- Theorem: If the winning candidate receives 70% of the votes and wins by a 320 vote majority,
    then the total number of votes polled is 800. -/
theorem election_votes_theorem (e : Election) 
  (h1 : e.winning_percentage = 70 / 100)
  (h2 : e.vote_majority = 320) :
  e.total_votes = 800 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l2796_279656


namespace NUMINAMATH_CALUDE_maria_carrots_next_day_l2796_279632

/-- The number of carrots Maria picked the next day -/
def carrots_picked_next_day (initial_carrots thrown_out final_total : ℕ) : ℕ :=
  final_total - (initial_carrots - thrown_out)

/-- Theorem stating that Maria picked 15 carrots the next day -/
theorem maria_carrots_next_day : 
  carrots_picked_next_day 48 11 52 = 15 := by
  sorry

end NUMINAMATH_CALUDE_maria_carrots_next_day_l2796_279632


namespace NUMINAMATH_CALUDE_peanut_seed_germination_probability_l2796_279667

/-- The probability of exactly k successes in n independent trials,
    each with probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability that exactly 2 out of 4 planted seeds will germinate,
    given that the probability of each seed germinating is 4/5. -/
theorem peanut_seed_germination_probability :
  binomial_probability 4 2 (4/5) = 96/625 := by
  sorry

end NUMINAMATH_CALUDE_peanut_seed_germination_probability_l2796_279667


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_m_value_l2796_279662

-- Define the point P
def P (m : ℤ) : ℝ × ℝ := (2 - m, m - 4)

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

-- Theorem statement
theorem point_in_third_quadrant_m_value :
  ∀ m : ℤ, in_third_quadrant (P m) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_m_value_l2796_279662


namespace NUMINAMATH_CALUDE_tangent_lines_imply_a_greater_than_three_l2796_279625

/-- The function f(x) = -x^3 + ax^2 - 2x --/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 2*x

/-- The derivative of f(x) --/
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x - 2

/-- The condition for a line to be tangent to f(x) at point t --/
def is_tangent (a : ℝ) (t : ℝ) : Prop :=
  -1 + t^3 - a*t^2 + 2*t = (-3*t^2 + 2*a*t - 2)*(-t)

/-- The theorem statement --/
theorem tangent_lines_imply_a_greater_than_three (a : ℝ) :
  (∃ t₁ t₂ t₃ : ℝ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧
    is_tangent a t₁ ∧ is_tangent a t₂ ∧ is_tangent a t₃) →
  a > 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_imply_a_greater_than_three_l2796_279625


namespace NUMINAMATH_CALUDE_cube_root_1728_l2796_279699

theorem cube_root_1728 (a b : ℕ+) (h1 : (1728 : ℝ)^(1/3) = a * b^(1/3)) 
  (h2 : ∀ c d : ℕ+, (1728 : ℝ)^(1/3) = c * d^(1/3) → b ≤ d) : 
  a + b = 13 := by sorry

end NUMINAMATH_CALUDE_cube_root_1728_l2796_279699


namespace NUMINAMATH_CALUDE_square_roots_combination_l2796_279633

theorem square_roots_combination : ∃ (a : ℚ), a * Real.sqrt 2 = Real.sqrt 8 ∧
  (∀ (b : ℚ), b * Real.sqrt 3 ≠ Real.sqrt 6) ∧
  (∀ (c : ℚ), c * Real.sqrt 2 ≠ Real.sqrt 12) ∧
  (∀ (d : ℚ), d * Real.sqrt 12 ≠ Real.sqrt 18) := by
  sorry

end NUMINAMATH_CALUDE_square_roots_combination_l2796_279633


namespace NUMINAMATH_CALUDE_junior_count_l2796_279658

theorem junior_count (total : ℕ) (junior_percent : ℚ) (senior_percent : ℚ)
  (h_total : total = 40)
  (h_junior_percent : junior_percent = 1/5)
  (h_senior_percent : senior_percent = 1/10)
  (h_equal_team : ∃ (x : ℕ), x * 5 = junior_percent * total ∧ x * 10 = senior_percent * total) :
  ∃ (j : ℕ), j = 12 ∧ j + (total - j) = total ∧
  (junior_percent * j).num = (senior_percent * (total - j)).num :=
sorry

end NUMINAMATH_CALUDE_junior_count_l2796_279658


namespace NUMINAMATH_CALUDE_schedule_ways_eq_840_l2796_279696

/-- The number of periods in a day -/
def num_periods : ℕ := 8

/-- The number of mathematics courses -/
def num_courses : ℕ := 4

/-- The number of ways to schedule the mathematics courses -/
def schedule_ways : ℕ := (num_periods - 1).choose num_courses * num_courses.factorial

/-- Theorem stating that the number of ways to schedule the mathematics courses is 840 -/
theorem schedule_ways_eq_840 : schedule_ways = 840 := by sorry

end NUMINAMATH_CALUDE_schedule_ways_eq_840_l2796_279696


namespace NUMINAMATH_CALUDE_inequalities_propositions_l2796_279651

theorem inequalities_propositions :
  (∀ a b : ℝ, a * b > 0 → a > b → 1 / a < 1 / b) ∧
  (∀ a b : ℝ, a > abs b → a^2 > b^2) ∧
  (∃ a b c d : ℝ, a > b ∧ a > d ∧ a - c ≤ b - d) ∧
  (∃ a b m : ℝ, a < b ∧ m > 0 ∧ a / b ≥ (a + m) / (b + m)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_propositions_l2796_279651


namespace NUMINAMATH_CALUDE_accidental_multiplication_l2796_279619

theorem accidental_multiplication (x : ℕ) : x * 9 = 153 → x * 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_accidental_multiplication_l2796_279619


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2796_279685

theorem fraction_equals_zero (x : ℝ) : 
  (x^2 - 4) / (x - 2) = 0 ∧ x ≠ 2 → x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2796_279685


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1500th_term_l2796_279643

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_1500th_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_term1 : a 1 = m)
  (h_term2 : a 2 = m + 2*n)
  (h_term3 : a 3 = 5*m - n)
  (h_term4 : a 4 = 3*m + 3*n)
  (h_term5 : a 5 = 7*m - n)
  : a 1500 = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1500th_term_l2796_279643


namespace NUMINAMATH_CALUDE_square_side_ratio_l2796_279661

theorem square_side_ratio (area_ratio : ℚ) : 
  area_ratio = 50 / 98 → 
  ∃ (a b c : ℕ), (a : ℚ) * Real.sqrt (b : ℚ) / (c : ℚ) = Real.sqrt (area_ratio) ∧ 
                  a = 5 ∧ b = 2 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_side_ratio_l2796_279661


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2796_279678

theorem sum_of_four_numbers : 8765 + 7658 + 6587 + 5876 = 28868 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2796_279678


namespace NUMINAMATH_CALUDE_no_pentagon_decagon_tiling_l2796_279607

/-- The interior angle of a regular pentagon in degrees -/
def pentagon_angle : ℝ := 108

/-- The interior angle of a regular decagon in degrees -/
def decagon_angle : ℝ := 144

/-- The sum of angles at a vertex in a tiling -/
def vertex_angle_sum : ℝ := 360

/-- Theorem stating the impossibility of tiling with regular pentagons and decagons -/
theorem no_pentagon_decagon_tiling : 
  ¬ ∃ (p d : ℕ), p * pentagon_angle + d * decagon_angle = vertex_angle_sum :=
sorry

end NUMINAMATH_CALUDE_no_pentagon_decagon_tiling_l2796_279607


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2796_279654

theorem complex_division_simplification : 
  let i : ℂ := Complex.I
  (2 - 3 * i) / (1 + i) = -1/2 - 5/2 * i := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2796_279654


namespace NUMINAMATH_CALUDE_equation_solution_l2796_279645

theorem equation_solution : ∃ m : ℝ, (243 : ℝ) ^ (1/5) = 3^m ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2796_279645


namespace NUMINAMATH_CALUDE_three_officers_from_six_people_l2796_279615

/-- The number of ways to choose three distinct officers from a group of people. -/
def chooseThreeOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- Theorem stating that choosing three distinct officers from 6 people results in 120 ways. -/
theorem three_officers_from_six_people : chooseThreeOfficers 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_officers_from_six_people_l2796_279615


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_l2796_279663

theorem simplify_trigonometric_expression :
  let x := Real.sin (15 * π / 180)
  let y := Real.cos (15 * π / 180)
  Real.sqrt (x^4 + 4 * y^2) - Real.sqrt (y^4 + 4 * x^2) = (1 / 2) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_l2796_279663


namespace NUMINAMATH_CALUDE_mica_sandwich_options_l2796_279694

-- Define the types of sandwich components
def BreadTypes : ℕ := 6
def MeatTypes : ℕ := 7
def CheeseTypes : ℕ := 6

-- Define the restricted combinations
def TurkeySwissCombinations : ℕ := BreadTypes
def SourdoughChickenCombinations : ℕ := CheeseTypes
def SalamiRyeCombinations : ℕ := CheeseTypes

-- Define the total number of restricted combinations
def TotalRestrictedCombinations : ℕ :=
  TurkeySwissCombinations + SourdoughChickenCombinations + SalamiRyeCombinations

-- Define the total number of possible sandwich combinations
def TotalPossibleCombinations : ℕ := BreadTypes * MeatTypes * CheeseTypes

-- Define the number of sandwiches Mica could order
def MicaSandwichOptions : ℕ := TotalPossibleCombinations - TotalRestrictedCombinations

-- Theorem statement
theorem mica_sandwich_options :
  MicaSandwichOptions = 234 := by sorry

end NUMINAMATH_CALUDE_mica_sandwich_options_l2796_279694


namespace NUMINAMATH_CALUDE_simplify_expression_l2796_279616

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 - b^3 = a - b) :
  a / b + b / a + 1 / (a * b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2796_279616


namespace NUMINAMATH_CALUDE_solve_for_x_l2796_279686

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2796_279686


namespace NUMINAMATH_CALUDE_prob_three_red_before_green_l2796_279655

/-- A hat containing red and green chips -/
structure ChipHat :=
  (red_chips : ℕ)
  (green_chips : ℕ)

/-- The probability of drawing all red chips before all green chips -/
def prob_all_red_before_green (hat : ChipHat) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem prob_three_red_before_green :
  let hat := ChipHat.mk 3 3
  prob_all_red_before_green hat = 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_three_red_before_green_l2796_279655


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2796_279627

theorem sum_with_radical_conjugate :
  let x : ℝ := 12 - Real.sqrt 50
  let y : ℝ := 12 + Real.sqrt 50
  x + y = 24 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2796_279627


namespace NUMINAMATH_CALUDE_plane_equation_correct_l2796_279688

/-- The equation of a line in 2D Cartesian coordinates -/
def line_equation (A B C : ℝ) (x y : ℝ) : Prop :=
  A * x + B * y + C = 0 ∧ A^2 + B^2 ≠ 0

/-- The equation of a plane in 3D Cartesian coordinates -/
def plane_equation (A B C D : ℝ) (x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0 ∧ A^2 + B^2 + C^2 ≠ 0

/-- Theorem stating that the given plane equation is correct -/
theorem plane_equation_correct :
  ∀ (A B C D : ℝ) (x y z : ℝ),
  plane_equation A B C D x y z ↔
  (A * x + B * y + C * z + D = 0 ∧ A^2 + B^2 + C^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l2796_279688


namespace NUMINAMATH_CALUDE_absolute_value_square_l2796_279631

theorem absolute_value_square (a b : ℚ) : |a| = b → a^2 = (-b)^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_l2796_279631


namespace NUMINAMATH_CALUDE_linda_has_34_candies_l2796_279681

/-- The number of candies Linda and Chloe have together -/
def total_candies : ℕ := 62

/-- The number of candies Chloe has -/
def chloe_candies : ℕ := 28

/-- The number of candies Linda has -/
def linda_candies : ℕ := total_candies - chloe_candies

theorem linda_has_34_candies : linda_candies = 34 := by
  sorry

end NUMINAMATH_CALUDE_linda_has_34_candies_l2796_279681


namespace NUMINAMATH_CALUDE_eight_people_seating_l2796_279684

/-- 
P(n) represents the number of valid seating arrangements for n people,
where each person must sit in their original seat or an adjacent seat.
-/
def P : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => P (n + 1) + P n

/-- The number of valid seating arrangements for 8 people is 34. -/
theorem eight_people_seating : P 8 = 34 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_seating_l2796_279684


namespace NUMINAMATH_CALUDE_exam_correct_answers_l2796_279665

/-- Proves the number of correct answers in an exam with given conditions -/
theorem exam_correct_answers 
  (total_questions : ℕ) 
  (correct_score : ℤ) 
  (wrong_score : ℤ) 
  (total_score : ℤ) 
  (h1 : total_questions = 70)
  (h2 : correct_score = 3)
  (h3 : wrong_score = -1)
  (h4 : total_score = 38) :
  ∃ (correct wrong : ℕ),
    correct + wrong = total_questions ∧
    correct_score * correct + wrong_score * wrong = total_score ∧
    correct = 27 := by
  sorry

end NUMINAMATH_CALUDE_exam_correct_answers_l2796_279665


namespace NUMINAMATH_CALUDE_division_of_four_by_negative_two_l2796_279601

theorem division_of_four_by_negative_two : 4 / (-2 : ℚ) = -2 := by sorry

end NUMINAMATH_CALUDE_division_of_four_by_negative_two_l2796_279601


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_count_l2796_279639

theorem stratified_sampling_third_year_count 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 2000) 
  (h2 : third_year_students = 630) 
  (h3 : sample_size = 200) :
  (sample_size * third_year_students) / total_students = 63 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_count_l2796_279639


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2796_279674

/-- Represents a cube with holes cut through each face. -/
structure CubeWithHoles where
  cubeSideLength : ℝ
  holeSideLength : ℝ

/-- Calculates the total surface area of a cube with holes. -/
def totalSurfaceArea (c : CubeWithHoles) : ℝ :=
  let originalSurfaceArea := 6 * c.cubeSideLength ^ 2
  let holeArea := 6 * c.holeSideLength ^ 2
  let remainingExteriorArea := originalSurfaceArea - holeArea
  let interiorSurfaceArea := 6 * 4 * c.holeSideLength * c.cubeSideLength
  remainingExteriorArea + interiorSurfaceArea

/-- Theorem stating that the total surface area of the given cube with holes is 72 square meters. -/
theorem cube_with_holes_surface_area :
  totalSurfaceArea { cubeSideLength := 3, holeSideLength := 1 } = 72 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2796_279674


namespace NUMINAMATH_CALUDE_elvins_phone_bill_l2796_279647

/-- Elvin's monthly telephone bill -/
def monthly_bill (call_charge : ℕ) (internet_charge : ℕ) : ℕ :=
  call_charge + internet_charge

theorem elvins_phone_bill 
  (internet_charge : ℕ) 
  (first_month_call_charge : ℕ) 
  (h1 : monthly_bill first_month_call_charge internet_charge = 50)
  (h2 : monthly_bill (2 * first_month_call_charge) internet_charge = 76) :
  monthly_bill (2 * first_month_call_charge) internet_charge = 76 :=
by
  sorry

#check elvins_phone_bill

end NUMINAMATH_CALUDE_elvins_phone_bill_l2796_279647
