import Mathlib

namespace NUMINAMATH_GPT_sandy_receives_correct_change_l2023_202377

-- Define the costs of each item
def cost_cappuccino : ℕ := 2
def cost_iced_tea : ℕ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℕ := 1

-- Define the quantities ordered
def qty_cappuccino : ℕ := 3
def qty_iced_tea : ℕ := 2
def qty_cafe_latte : ℕ := 2
def qty_espresso : ℕ := 2

-- Calculate the total cost
def total_cost : ℝ := (qty_cappuccino * cost_cappuccino) + 
                      (qty_iced_tea * cost_iced_tea) + 
                      (qty_cafe_latte * cost_cafe_latte) + 
                      (qty_espresso * cost_espresso)

-- Define the amount paid
def amount_paid : ℝ := 20

-- Calculate the change
def change : ℝ := amount_paid - total_cost

theorem sandy_receives_correct_change : change = 3 := by
  -- Detailed steps would go here
  sorry

end NUMINAMATH_GPT_sandy_receives_correct_change_l2023_202377


namespace NUMINAMATH_GPT_football_games_this_year_l2023_202372

theorem football_games_this_year 
  (total_games : ℕ) 
  (games_last_year : ℕ) 
  (games_this_year : ℕ) 
  (h1 : total_games = 9) 
  (h2 : games_last_year = 5) 
  (h3 : total_games = games_last_year + games_this_year) : 
  games_this_year = 4 := 
sorry

end NUMINAMATH_GPT_football_games_this_year_l2023_202372


namespace NUMINAMATH_GPT_average_girls_score_l2023_202301

open Function

variable (C c D d : ℕ)
variable (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)

-- Conditions
def CedarBoys := avgCedarBoys = 85
def CedarGirls := avgCedarGirls = 80
def CedarCombined := avgCedarCombined = 83
def DeltaBoys := avgDeltaBoys = 76
def DeltaGirls := avgDeltaGirls = 95
def DeltaCombined := avgDeltaCombined = 87
def CombinedBoys := avgCombinedBoys = 73

-- Correct answer
def CombinedGirls (avgCombinedGirls : ℤ) := avgCombinedGirls = 86

-- Final statement
theorem average_girls_score (C c D d : ℕ)
    (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)
    (H1 : CedarBoys avgCedarBoys)
    (H2 : CedarGirls avgCedarGirls)
    (H3 : CedarCombined avgCedarCombined)
    (H4 : DeltaBoys avgDeltaBoys)
    (H5 : DeltaGirls avgDeltaGirls)
    (H6 : DeltaCombined avgDeltaCombined)
    (H7 : CombinedBoys avgCombinedBoys) :
    ∃ avgCombinedGirls, CombinedGirls avgCombinedGirls :=
sorry

end NUMINAMATH_GPT_average_girls_score_l2023_202301


namespace NUMINAMATH_GPT_rationalize_denominator_l2023_202369

theorem rationalize_denominator :
  (3 : ℝ) / Real.sqrt 48 = Real.sqrt 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l2023_202369


namespace NUMINAMATH_GPT_cone_base_diameter_l2023_202373

theorem cone_base_diameter
  (h_cone : ℝ) (r_sphere : ℝ) (waste_percentage : ℝ) (d : ℝ) :
  h_cone = 9 → r_sphere = 9 → waste_percentage = 0.75 → 
  (V_cone = 1/3 * π * (d/2)^2 * h_cone) →
  (V_sphere = 4/3 * π * r_sphere^3) →
  (V_cone = (1 - waste_percentage) * V_sphere) →
  d = 9 :=
by
  intros h_cond r_cond waste_cond v_cone_eq v_sphere_eq v_cone_sphere_eq
  sorry

end NUMINAMATH_GPT_cone_base_diameter_l2023_202373


namespace NUMINAMATH_GPT_domain_of_f2x_l2023_202334

theorem domain_of_f2x (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = f x) : 
  ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, f y = f (2 * x) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f2x_l2023_202334


namespace NUMINAMATH_GPT_divisor_of_1025_l2023_202302

theorem divisor_of_1025 : ∃ k : ℕ, 41 * k = 1025 :=
  sorry

end NUMINAMATH_GPT_divisor_of_1025_l2023_202302


namespace NUMINAMATH_GPT_inequality_ratios_l2023_202325

theorem inequality_ratios (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (c / a) > (d / b) :=
sorry

end NUMINAMATH_GPT_inequality_ratios_l2023_202325


namespace NUMINAMATH_GPT_find_side_c_l2023_202336

theorem find_side_c (a C S : ℝ) (ha : a = 3) (hC : C = 120) (hS : S = (15 * Real.sqrt 3) / 4) : 
  ∃ (c : ℝ), c = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_side_c_l2023_202336


namespace NUMINAMATH_GPT_quadratic_root_range_specific_m_value_l2023_202350

theorem quadratic_root_range (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1^2 - 2 * (1 - m) * x1 + m^2 = 0 ∧ x2^2 - 2 * (1 - m) * x2 + m^2 = 0 ↔ m ≤ 1/2 :=
by
  sorry

theorem specific_m_value (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2 * (1 - m) * x1 + m^2 = 0)
  (h2 : x2^2 - 2 * (1 - m) * x2 + m^2 = 0) (h3 : x1^2 + 12 * m + x2^2 = 10) : 
  m = -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_range_specific_m_value_l2023_202350


namespace NUMINAMATH_GPT_tan_alpha_eq_neg_one_third_l2023_202364

open Real

theorem tan_alpha_eq_neg_one_third
  (h : cos (π / 4 - α) / cos (π / 4 + α) = 1 / 2) :
  tan α = -1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg_one_third_l2023_202364


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l2023_202387

theorem monotonic_increasing_interval : ∀ x : ℝ, (x > 2) → ((x-3) * Real.exp x > 0) :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l2023_202387


namespace NUMINAMATH_GPT_sin_identity_l2023_202371

variable (α : ℝ)
axiom alpha_def : α = Real.pi / 7

theorem sin_identity : (Real.sin (3 * α)) ^ 2 - (Real.sin α) ^ 2 = Real.sin (2 * α) * Real.sin (3 * α) := 
by 
  sorry

end NUMINAMATH_GPT_sin_identity_l2023_202371


namespace NUMINAMATH_GPT_solve_inequality_l2023_202388

variable {x : ℝ}

theorem solve_inequality :
  (x - 8) / (x^2 - 4 * x + 13) ≥ 0 ↔ x ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2023_202388


namespace NUMINAMATH_GPT_michael_total_revenue_l2023_202398

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def qty_large : ℕ := 2
def qty_medium : ℕ := 2
def qty_small : ℕ := 3

def total_revenue : ℕ :=
  (price_large * qty_large) +
  (price_medium * qty_medium) +
  (price_small * qty_small)

theorem michael_total_revenue : total_revenue = 97 :=
  by sorry

end NUMINAMATH_GPT_michael_total_revenue_l2023_202398


namespace NUMINAMATH_GPT_relationship_among_abc_l2023_202344

noncomputable
def a := 0.2 ^ 1.5

noncomputable
def b := 2 ^ 0.1

noncomputable
def c := 0.2 ^ 1.3

theorem relationship_among_abc : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l2023_202344


namespace NUMINAMATH_GPT_inequality_proof_l2023_202317

theorem inequality_proof (x y : ℝ) (h : |x - 2 * y| = 5) : x^2 + y^2 ≥ 5 := 
  sorry

end NUMINAMATH_GPT_inequality_proof_l2023_202317


namespace NUMINAMATH_GPT_find_a_if_f_is_odd_l2023_202361

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a_if_f_is_odd (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = - f a x) → a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_f_is_odd_l2023_202361


namespace NUMINAMATH_GPT_JacobNeed_l2023_202386

-- Definitions of the conditions
def jobEarningsBeforeTax : ℝ := 25 * 15
def taxAmount : ℝ := 0.10 * jobEarningsBeforeTax
def jobEarningsAfterTax : ℝ := jobEarningsBeforeTax - taxAmount

def cookieEarnings : ℝ := 5 * 30

def tutoringEarnings : ℝ := 100 * 4

def lotteryWinnings : ℝ := 700 - 20
def friendShare : ℝ := 0.30 * lotteryWinnings
def netLotteryWinnings : ℝ := lotteryWinnings - friendShare

def giftFromSisters : ℝ := 700 * 2

def totalEarnings : ℝ := jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters

def travelGearExpenses : ℝ := 3 + 47

def netSavings : ℝ := totalEarnings - travelGearExpenses

def tripCost : ℝ := 8000

-- Statement to be proven
theorem JacobNeed (jobEarningsBeforeTax taxAmount jobEarningsAfterTax cookieEarnings tutoringEarnings 
netLotteryWinnings giftFromSisters totalEarnings travelGearExpenses netSavings tripCost : ℝ) : 
  (jobEarningsAfterTax == (25 * 15) - (0.10 * (25 * 15))) → 
  (cookieEarnings == 5 * 30) →
  (tutoringEarnings == 100 * 4) →
  (netLotteryWinnings == (700 - 20) - (0.30 * (700 - 20))) →
  (giftFromSisters == 700 * 2) →
  (totalEarnings == jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters) →
  (travelGearExpenses == 3 + 47) →
  (netSavings == totalEarnings - travelGearExpenses) →
  (tripCost == 8000) →
  (tripCost - netSavings = 5286.50) :=
by
  intros
  sorry

end NUMINAMATH_GPT_JacobNeed_l2023_202386


namespace NUMINAMATH_GPT_coin_draws_expected_value_l2023_202376

theorem coin_draws_expected_value :
  ∃ f : ℕ → ℝ, (∀ (n : ℕ), n ≥ 4 → f n = (3 : ℝ)) := sorry

end NUMINAMATH_GPT_coin_draws_expected_value_l2023_202376


namespace NUMINAMATH_GPT_largest_possible_perimeter_l2023_202366

theorem largest_possible_perimeter (y : ℤ) (hy1 : 3 ≤ y) (hy2 : y < 16) : 7 + 9 + y ≤ 31 := 
by
  sorry

end NUMINAMATH_GPT_largest_possible_perimeter_l2023_202366


namespace NUMINAMATH_GPT_domain_of_f_2x_minus_3_l2023_202335

noncomputable def f (x : ℝ) := 2 * x + 1

theorem domain_of_f_2x_minus_3 :
  (∀ x, 1 ≤ 2 * x - 3 ∧ 2 * x - 3 ≤ 5 → (2 ≤ x ∧ x ≤ 4)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_2x_minus_3_l2023_202335


namespace NUMINAMATH_GPT_inverse_matrix_equation_of_line_l_l2023_202352

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 2], ![3, 4]]
noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![-2, 1], ![3/2, -1/2]]

theorem inverse_matrix :
  M⁻¹ = M_inv :=
by
  sorry

def transformed_line (x y : ℚ) : Prop := 2 * (x + 2 * y) - (3 * x + 4 * y) = 4 

theorem equation_of_line_l (x y : ℚ) :
  transformed_line x y → x + 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_inverse_matrix_equation_of_line_l_l2023_202352


namespace NUMINAMATH_GPT_find_x_l2023_202393

theorem find_x (x : ℝ) : 
  (∀ (y : ℝ), 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := sorry

end NUMINAMATH_GPT_find_x_l2023_202393


namespace NUMINAMATH_GPT_find_number_l2023_202310

theorem find_number (p q N : ℝ) (h1 : N / p = 8) (h2 : N / q = 18) (h3 : p - q = 0.20833333333333334) : N = 3 :=
sorry

end NUMINAMATH_GPT_find_number_l2023_202310


namespace NUMINAMATH_GPT_convex_polyhedron_faces_same_edges_l2023_202349

theorem convex_polyhedron_faces_same_edges (n : ℕ) (f : Fin n → ℕ) 
  (n_ge_4 : 4 ≤ n)
  (h : ∀ i : Fin n, 3 ≤ f i ∧ f i ≤ n - 1) : 
  ∃ (i j : Fin n), i ≠ j ∧ f i = f j := 
by
  sorry

end NUMINAMATH_GPT_convex_polyhedron_faces_same_edges_l2023_202349


namespace NUMINAMATH_GPT_francie_has_3_dollars_remaining_l2023_202382

def francies_remaining_money : ℕ :=
  let allowance1 := 5 * 8
  let allowance2 := 6 * 6
  let total_savings := allowance1 + allowance2
  let remaining_after_clothes := total_savings / 2
  remaining_after_clothes - 35

theorem francie_has_3_dollars_remaining :
  francies_remaining_money = 3 := 
sorry

end NUMINAMATH_GPT_francie_has_3_dollars_remaining_l2023_202382


namespace NUMINAMATH_GPT_number_of_ways_to_choose_4_captains_from_15_l2023_202389

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_4_captains_from_15_l2023_202389


namespace NUMINAMATH_GPT_determine_k_l2023_202358

variable (x y z w : ℝ)

theorem determine_k
  (h₁ : 9 / (x + y + w) = k / (x + z + w))
  (h₂ : k / (x + z + w) = 12 / (z - y)) :
  k = 21 :=
sorry

end NUMINAMATH_GPT_determine_k_l2023_202358


namespace NUMINAMATH_GPT_cost_per_rose_l2023_202332

theorem cost_per_rose (P : ℝ) (h1 : 5 * 12 = 60) (h2 : 0.8 * 60 * P = 288) : P = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cost_per_rose_l2023_202332


namespace NUMINAMATH_GPT_largest_possible_product_l2023_202305

theorem largest_possible_product : 
  ∃ S1 S2 : Finset ℕ, 
  (S1 ∪ S2 = {1, 3, 4, 6, 7, 8, 9} ∧ S1 ∩ S2 = ∅ ∧ S1.prod id = S2.prod id) ∧ 
  (S1.prod id = 504 ∧ S2.prod id = 504) :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_product_l2023_202305


namespace NUMINAMATH_GPT_remainder_mul_three_division_l2023_202307

theorem remainder_mul_three_division
    (N : ℤ) (k : ℤ)
    (h1 : N = 1927 * k + 131) :
    ((3 * N) % 43) = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_mul_three_division_l2023_202307


namespace NUMINAMATH_GPT_part1_part2_l2023_202342

noncomputable def f (x m : ℝ) := |x + 1| + |m - x|

theorem part1 (x : ℝ) : (f x 3) ≥ 6 ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2 (m : ℝ) : (∀ x, f x m ≥ 8) ↔ (m ≥ 7 ∨ m ≤ -9) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l2023_202342


namespace NUMINAMATH_GPT_percent_of_dollar_is_37_l2023_202311

variable (coins_value_in_cents : ℕ)
variable (percent_of_one_dollar : ℕ)

def value_of_pennies : ℕ := 2 * 1
def value_of_nickels : ℕ := 3 * 5
def value_of_dimes : ℕ := 2 * 10

def total_coin_value : ℕ := value_of_pennies + value_of_nickels + value_of_dimes

theorem percent_of_dollar_is_37
  (h1 : total_coin_value = coins_value_in_cents)
  (h2 : percent_of_one_dollar = (coins_value_in_cents * 100) / 100) : 
  percent_of_one_dollar = 37 := 
by
  sorry

end NUMINAMATH_GPT_percent_of_dollar_is_37_l2023_202311


namespace NUMINAMATH_GPT_complement_fraction_irreducible_l2023_202353

theorem complement_fraction_irreducible (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.gcd (b - a) b = 1 :=
sorry

end NUMINAMATH_GPT_complement_fraction_irreducible_l2023_202353


namespace NUMINAMATH_GPT_initial_tax_rate_l2023_202399

variable (R : ℝ)

theorem initial_tax_rate
  (income : ℝ := 48000)
  (new_rate : ℝ := 0.30)
  (savings : ℝ := 7200)
  (tax_savings : income * (R / 100) - income * new_rate = savings) :
  R = 45 := by
  sorry

end NUMINAMATH_GPT_initial_tax_rate_l2023_202399


namespace NUMINAMATH_GPT_area_of_enclosed_shape_l2023_202390

noncomputable def enclosed_area : ℝ := 
∫ x in (0 : ℝ)..(2/3 : ℝ), (2 * x - 3 * x^2)

theorem area_of_enclosed_shape : enclosed_area = 4 / 27 := by
  sorry

end NUMINAMATH_GPT_area_of_enclosed_shape_l2023_202390


namespace NUMINAMATH_GPT_difference_highest_lowest_score_l2023_202315

-- Definitions based on conditions
def total_innings : ℕ := 46
def avg_innings : ℕ := 61
def highest_score : ℕ := 202
def avg_excl_highest_lowest : ℕ := 58
def innings_excl_highest_lowest : ℕ := 44

-- Calculated total runs
def total_runs : ℕ := total_innings * avg_innings
def total_runs_excl_highest_lowest : ℕ := innings_excl_highest_lowest * avg_excl_highest_lowest
def sum_of_highest_lowest : ℕ := total_runs - total_runs_excl_highest_lowest
def lowest_score : ℕ := sum_of_highest_lowest - highest_score

theorem difference_highest_lowest_score 
  (h1: total_runs = total_innings * avg_innings)
  (h2: avg_excl_highest_lowest * innings_excl_highest_lowest = total_runs_excl_highest_lowest)
  (h3: sum_of_highest_lowest = total_runs - total_runs_excl_highest_lowest)
  (h4: highest_score = 202)
  (h5: lowest_score = sum_of_highest_lowest - highest_score)
  : highest_score - lowest_score = 150 :=
by
  -- We only need to state the theorem, so we can skip the proof.
  -- The exact statements of conditions and calculations imply the result.
  sorry

end NUMINAMATH_GPT_difference_highest_lowest_score_l2023_202315


namespace NUMINAMATH_GPT_minimum_x_plus_2y_exists_l2023_202303

theorem minimum_x_plus_2y_exists (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) :
  ∃ z : ℝ, z = x + 2 * y ∧ z = -2 * Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_GPT_minimum_x_plus_2y_exists_l2023_202303


namespace NUMINAMATH_GPT_prime_quadruples_l2023_202345

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_quadruples {p₁ p₂ p₃ p₄ : ℕ} (prime_p₁ : is_prime p₁) (prime_p₂ : is_prime p₂) (prime_p₃ : is_prime p₃) (prime_p₄ : is_prime p₄)
  (h1 : p₁ < p₂) (h2 : p₂ < p₃) (h3 : p₃ < p₄) (eq_condition : p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882) :
  (p₁ = 2 ∧ p₂ = 5 ∧ p₃ = 19 ∧ p₄ = 37) ∨
  (p₁ = 2 ∧ p₂ = 11 ∧ p₃ = 19 ∧ p₄ = 31) ∨
  (p₁ = 2 ∧ p₂ = 13 ∧ p₃ = 19 ∧ p₄ = 29) :=
sorry

end NUMINAMATH_GPT_prime_quadruples_l2023_202345


namespace NUMINAMATH_GPT_seventeen_in_base_three_l2023_202391

theorem seventeen_in_base_three : (17 : ℕ) = 1 * 3^2 + 2 * 3^1 + 2 * 3^0 :=
by
  -- This is the arithmetic representation of the conversion,
  -- proving that 17 in base 10 equals 122 in base 3
  sorry

end NUMINAMATH_GPT_seventeen_in_base_three_l2023_202391


namespace NUMINAMATH_GPT_ratio_of_falls_l2023_202365

variable (SteveFalls : ℕ) (StephFalls : ℕ) (SonyaFalls : ℕ)
variable (H1 : SteveFalls = 3)
variable (H2 : StephFalls = SteveFalls + 13)
variable (H3 : SonyaFalls = 6)

theorem ratio_of_falls : SonyaFalls / (StephFalls / 2) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_of_falls_l2023_202365


namespace NUMINAMATH_GPT_cost_of_each_cake_l2023_202330

-- Define the conditions
def cakes : ℕ := 3
def payment_by_john : ℕ := 18
def total_payment : ℕ := payment_by_john * 2

-- Statement to prove that each cake costs $12
theorem cost_of_each_cake : (total_payment / cakes) = 12 := by
  sorry

end NUMINAMATH_GPT_cost_of_each_cake_l2023_202330


namespace NUMINAMATH_GPT_distinct_solutions_equation_l2023_202363

theorem distinct_solutions_equation (a b : ℝ) (h1 : a ≠ b) (h2 : a > b) (h3 : ∀ x, (3 * x - 9) / (x^2 + 3 * x - 18) = x + 1) (sol_a : x = a) (sol_b : x = b) :
  a - b = 1 :=
sorry

end NUMINAMATH_GPT_distinct_solutions_equation_l2023_202363


namespace NUMINAMATH_GPT_monthly_income_of_P_l2023_202392

theorem monthly_income_of_P (P Q R : ℕ) (h1 : P + Q = 10100) (h2 : Q + R = 12500) (h3 : P + R = 10400) : 
  P = 4000 := 
by 
  sorry

end NUMINAMATH_GPT_monthly_income_of_P_l2023_202392


namespace NUMINAMATH_GPT_tomatoes_multiplier_l2023_202356

theorem tomatoes_multiplier (before_vacation : ℕ) (grown_during_vacation : ℕ)
  (h1 : before_vacation = 36)
  (h2 : grown_during_vacation = 3564) :
  (before_vacation + grown_during_vacation) / before_vacation = 100 :=
by
  -- Insert proof here later
  sorry

end NUMINAMATH_GPT_tomatoes_multiplier_l2023_202356


namespace NUMINAMATH_GPT_binary_remainder_div_4_is_1_l2023_202367

def binary_to_base_10_last_two_digits (b1 b0 : Nat) : Nat :=
  2 * b1 + b0

noncomputable def remainder_of_binary_by_4 (n : Nat) : Nat :=
  match n with
  | 111010110101 => binary_to_base_10_last_two_digits 0 1
  | _ => 0

theorem binary_remainder_div_4_is_1 :
  remainder_of_binary_by_4 111010110101 = 1 := by
  sorry

end NUMINAMATH_GPT_binary_remainder_div_4_is_1_l2023_202367


namespace NUMINAMATH_GPT_min_value_when_a_is_half_range_of_a_for_positivity_l2023_202397

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 2*x + a) / x

theorem min_value_when_a_is_half : 
  ∀ x ∈ Set.Ici (1 : ℝ), f x (1/2) ≥ (7 / 2) := 
by 
  sorry

theorem range_of_a_for_positivity :
  ∀ x ∈ Set.Ici (1 : ℝ), f x a > 0 ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_when_a_is_half_range_of_a_for_positivity_l2023_202397


namespace NUMINAMATH_GPT_largest_valid_four_digit_number_l2023_202355

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end NUMINAMATH_GPT_largest_valid_four_digit_number_l2023_202355


namespace NUMINAMATH_GPT_income_calculation_l2023_202338

-- Define the conditions
def ratio (i e : ℕ) : Prop := 9 * e = 8 * i
def savings (i e : ℕ) : Prop := i - e = 4000

-- The theorem statement
theorem income_calculation (i e : ℕ) (h1 : ratio i e) (h2 : savings i e) : i = 36000 := by
  sorry

end NUMINAMATH_GPT_income_calculation_l2023_202338


namespace NUMINAMATH_GPT_gcd_180_126_l2023_202357

theorem gcd_180_126 : Nat.gcd 180 126 = 18 := by
  sorry

end NUMINAMATH_GPT_gcd_180_126_l2023_202357


namespace NUMINAMATH_GPT_lineup_possibilities_l2023_202327

theorem lineup_possibilities (total_players : ℕ) (all_stars_in_lineup : ℕ) (injured_player : ℕ) :
  total_players = 15 ∧ all_stars_in_lineup = 2 ∧ injured_player = 1 →
  Nat.choose 12 4 = 495 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_lineup_possibilities_l2023_202327


namespace NUMINAMATH_GPT_number_of_paper_cups_is_40_l2023_202333

noncomputable def cost_paper_plate : ℝ := sorry
noncomputable def cost_paper_cup : ℝ := sorry
noncomputable def num_paper_cups_in_second_purchase : ℝ := sorry

-- Conditions
axiom first_condition : 100 * cost_paper_plate + 200 * cost_paper_cup = 7.50
axiom second_condition : 20 * cost_paper_plate + num_paper_cups_in_second_purchase * cost_paper_cup = 1.50

-- Goal
theorem number_of_paper_cups_is_40 : num_paper_cups_in_second_purchase = 40 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_paper_cups_is_40_l2023_202333


namespace NUMINAMATH_GPT_problem_statement_l2023_202339

theorem problem_statement (a : ℤ)
  (h : (2006 - a) * (2004 - a) = 2005) :
  (2006 - a) ^ 2 + (2004 - a) ^ 2 = 4014 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2023_202339


namespace NUMINAMATH_GPT_min_value_x_plus_one_over_x_plus_two_l2023_202381

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1/(x + 2) ∧ y ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_one_over_x_plus_two_l2023_202381


namespace NUMINAMATH_GPT_lightest_ball_box_is_blue_l2023_202380

-- Define the weights and counts of balls
def yellow_ball_weight : ℕ := 50
def yellow_ball_count_per_box : ℕ := 50
def white_ball_weight : ℕ := 45
def white_ball_count_per_box : ℕ := 60
def blue_ball_weight : ℕ := 55
def blue_ball_count_per_box : ℕ := 40

-- Calculate the total weight of balls per type
def yellow_box_weight : ℕ := yellow_ball_weight * yellow_ball_count_per_box
def white_box_weight : ℕ := white_ball_weight * white_ball_count_per_box
def blue_box_weight : ℕ := blue_ball_weight * blue_ball_count_per_box

theorem lightest_ball_box_is_blue :
  (blue_box_weight < yellow_box_weight) ∧ (blue_box_weight < white_box_weight) :=
by
  -- Proof can go here
  sorry

end NUMINAMATH_GPT_lightest_ball_box_is_blue_l2023_202380


namespace NUMINAMATH_GPT_all_points_lie_on_parabola_l2023_202340

noncomputable def parabola_curve (u : ℝ) : ℝ × ℝ :=
  let x := 3^u - 4
  let y := 9^u - 7 * 3^u - 2
  (x, y)

theorem all_points_lie_on_parabola (u : ℝ) :
  let (x, y) := parabola_curve u
  y = x^2 + x - 6 := sorry

end NUMINAMATH_GPT_all_points_lie_on_parabola_l2023_202340


namespace NUMINAMATH_GPT_earth_surface_inhabitable_fraction_l2023_202351

theorem earth_surface_inhabitable_fraction :
  (1 / 3 : ℝ) * (2 / 3 : ℝ) = 2 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_earth_surface_inhabitable_fraction_l2023_202351


namespace NUMINAMATH_GPT_scientific_notation_correct_l2023_202324

/-- Define the number 42.39 million as 42.39 * 10^6 and prove that it is equivalent to 4.239 * 10^7 -/
def scientific_notation_of_42_39_million : Prop :=
  (42.39 * 10^6 = 4.239 * 10^7)

theorem scientific_notation_correct : scientific_notation_of_42_39_million :=
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l2023_202324


namespace NUMINAMATH_GPT_no_solution_exists_l2023_202343

theorem no_solution_exists (x y : ℝ) : 9^(y + 1) / (1 + 4 / x^2) ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l2023_202343


namespace NUMINAMATH_GPT_john_spent_l2023_202385

-- Given definitions from the conditions.
def total_time_in_hours := 4
def additional_minutes := 35
def break_time_per_break := 10
def number_of_breaks := 5
def cost_per_5_minutes := 0.75
def playing_cost (total_time_in_hours additional_minutes break_time_per_break number_of_breaks : ℕ) 
  (cost_per_5_minutes : ℝ) : ℝ :=
  let total_minutes := total_time_in_hours * 60 + additional_minutes
  let break_time := number_of_breaks * break_time_per_break
  let actual_playing_time := total_minutes - break_time
  let number_of_intervals := actual_playing_time / 5
  number_of_intervals * cost_per_5_minutes

-- Statement to be proved.
theorem john_spent (total_time_in_hours := 4) (additional_minutes := 35) (break_time_per_break := 10) 
  (number_of_breaks := 5) (cost_per_5_minutes := 0.75) :
  playing_cost total_time_in_hours additional_minutes break_time_per_break number_of_breaks cost_per_5_minutes = 33.75 := 
by
  sorry

end NUMINAMATH_GPT_john_spent_l2023_202385


namespace NUMINAMATH_GPT_non_neg_ints_less_than_pi_l2023_202368

-- Define the condition: non-negative integers with absolute value less than π
def condition (x : ℕ) : Prop := |(x : ℝ)| < Real.pi

-- Prove that the set satisfying the condition is {0, 1, 2, 3}
theorem non_neg_ints_less_than_pi :
  {x : ℕ | condition x} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_non_neg_ints_less_than_pi_l2023_202368


namespace NUMINAMATH_GPT_calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l2023_202346

-- Define the necessary probability events and conditions.
variable {p : ℝ} (calc_action : ℕ → ℝ)

-- Condition: initially, the display shows 0.
def initial_display : ℕ := 0

-- Events for part (a): addition only, randomly chosen numbers from 0 to 9.
def random_addition_event (n : ℕ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Events for part (b): both addition and multiplication allowed.
def random_operation_event (n : ℕ) : Prop := (n % 2 = 0 ∧ n % 2 = 1) ∨ -- addition
                                               (n ≠ 0 ∧ n % 2 = 1 ∧ (n/2) % 2 = 1) -- multiplication

-- Statements to be proved based on above definitions.
theorem calc_addition_even_odd_probability :
  calc_action 0 = 1 / 2 → random_addition_event initial_display := sorry

theorem calc_addition_multiplication_even_probability :
  calc_action (initial_display + 1) > 1 / 2 → random_operation_event (initial_display + 1) := sorry

end NUMINAMATH_GPT_calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l2023_202346


namespace NUMINAMATH_GPT_general_term_sequence_l2023_202313

variable {a : ℕ → ℝ}
variable {n : ℕ}

def sequence_condition (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n ≥ 1 → (n+1) * (a (n + 1))^2 - n * (a n)^2 + (a n) * (a (n + 1)) = 0)

theorem general_term_sequence (a : ℕ → ℝ) (h : sequence_condition a) :
  ∀ n : ℕ, n > 0 → a n = 1 / n := by
  sorry

end NUMINAMATH_GPT_general_term_sequence_l2023_202313


namespace NUMINAMATH_GPT_perimeter_of_square_with_area_36_l2023_202316

theorem perimeter_of_square_with_area_36 : 
  ∀ (A : ℝ), A = 36 → (∃ P : ℝ, P = 24 ∧ (∃ s : ℝ, s^2 = A ∧ P = 4 * s)) :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_with_area_36_l2023_202316


namespace NUMINAMATH_GPT_rotated_triangle_surface_area_l2023_202396

theorem rotated_triangle_surface_area :
  ∀ (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (ACLength : ℝ) (BCLength : ℝ) (right_angle : ℝ -> ℝ -> ℝ -> Prop)
    (pi_def : Real) (surface_area : ℝ -> ℝ -> ℝ),
    (right_angle 90 0 90) → (ACLength = 3) → (BCLength = 4) →
    surface_area ACLength BCLength = 24 * pi_def  :=
by
  sorry

end NUMINAMATH_GPT_rotated_triangle_surface_area_l2023_202396


namespace NUMINAMATH_GPT_football_championship_min_games_l2023_202383

theorem football_championship_min_games :
  (∃ (teams : Finset ℕ) (games : Finset (ℕ × ℕ)),
    teams.card = 20 ∧
    (∀ (a b c : ℕ), a ∈ teams → b ∈ teams → c ∈ teams → a ≠ b → b ≠ c → c ≠ a →
      (a, b) ∈ games ∨ (b, c) ∈ games ∨ (c, a) ∈ games) ∧
    games.card = 90) :=
sorry

end NUMINAMATH_GPT_football_championship_min_games_l2023_202383


namespace NUMINAMATH_GPT_price_of_book_l2023_202326

-- Definitions based on the problem conditions
def money_xiaowang_has (p : ℕ) : ℕ := 2 * p - 6
def money_xiaoli_has (p : ℕ) : ℕ := 2 * p - 31

def combined_money (p : ℕ) : ℕ := money_xiaowang_has p + money_xiaoli_has p

-- Lean statement to prove the price of each book
theorem price_of_book (p : ℕ) : combined_money p = 3 * p → p = 37 :=
by
  sorry

end NUMINAMATH_GPT_price_of_book_l2023_202326


namespace NUMINAMATH_GPT_geom_progression_common_ratio_l2023_202374

theorem geom_progression_common_ratio (x y z r : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : ∃ a, a ≠ 0 ∧ x * (2 * y - z) = a ∧ y * (2 * z - x) = a * r ∧ z * (2 * x - y) = a * r^2) :
  r^2 + r + 1 = 0 :=
sorry

end NUMINAMATH_GPT_geom_progression_common_ratio_l2023_202374


namespace NUMINAMATH_GPT_symmetric_point_of_A_l2023_202370

theorem symmetric_point_of_A (a b : ℝ) 
  (h1 : 2 * a - 4 * b + 9 = 0) 
  (h2 : ∃ t : ℝ, (a, b) = (1 - 4 * t, 4 + 2 * t)) : 
  (a, b) = (1, 4) :=
sorry

end NUMINAMATH_GPT_symmetric_point_of_A_l2023_202370


namespace NUMINAMATH_GPT_range_of_first_term_in_geometric_sequence_l2023_202300

theorem range_of_first_term_in_geometric_sequence (q a₁ : ℝ)
  (h_q : |q| < 1)
  (h_sum : a₁ / (1 - q) = q) :
  -2 < a₁ ∧ a₁ ≤ 0.25 ∧ a₁ ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_first_term_in_geometric_sequence_l2023_202300


namespace NUMINAMATH_GPT_parabola_properties_l2023_202348

-- Given conditions
variables (a b c : ℝ)
variable (h_vertex : ∃ a b c : ℝ, (∀ x, a * (x+1)^2 + 4 = ax^2 + b * x + c))
variable (h_intersection : ∃ A : ℝ, 2 < A ∧ A < 3 ∧ a * A^2 + b * A + c = 0)

-- Define the proof problem
theorem parabola_properties (h_vertex : (b = 2 * a)) (h_a : a < 0) (h_c : c = 4 + a) : 
  ∃ x : ℕ, x = 2 ∧ 
  (∀ a b c : ℝ, a * b * c < 0 → false) ∧ 
  (-4 < a ∧ a < -1 → false) ∧
  (a * c + 2 * b > 1 → false) :=
sorry

end NUMINAMATH_GPT_parabola_properties_l2023_202348


namespace NUMINAMATH_GPT_calc_num_articles_l2023_202379

-- Definitions based on the conditions
def cost_price (C : ℝ) : ℝ := C
def selling_price (C : ℝ) : ℝ := 1.10000000000000004 * C
def num_articles (n : ℝ) (C : ℝ) (S : ℝ) : Prop := 55 * C = n * S

-- Proof Statement
theorem calc_num_articles (C : ℝ) : ∃ n : ℝ, num_articles n C (selling_price C) ∧ n = 50 :=
by sorry

end NUMINAMATH_GPT_calc_num_articles_l2023_202379


namespace NUMINAMATH_GPT_evaluate_expression_l2023_202375

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 5)) = 15 / 16 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2023_202375


namespace NUMINAMATH_GPT_ab_value_l2023_202354

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a^4 + b^4 = 5 / 8) : ab = (Real.sqrt 3) / 4 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l2023_202354


namespace NUMINAMATH_GPT_factorize_expression_l2023_202329

-- Define variables m and n
variables (m n : ℤ)

-- The theorem stating the equality
theorem factorize_expression : m^3 * n - m * n = m * n * (m - 1) * (m + 1) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l2023_202329


namespace NUMINAMATH_GPT_probability_red_white_red_l2023_202347

-- Definitions and assumptions
def total_marbles := 10
def red_marbles := 4
def white_marbles := 6

def P_first_red : ℚ := red_marbles / total_marbles
def P_second_white_given_first_red : ℚ := white_marbles / (total_marbles - 1)
def P_third_red_given_first_red_and_second_white : ℚ := (red_marbles - 1) / (total_marbles - 2)

-- The target probability hypothesized
theorem probability_red_white_red :
  P_first_red * P_second_white_given_first_red * P_third_red_given_first_red_and_second_white = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_white_red_l2023_202347


namespace NUMINAMATH_GPT_min_sum_distances_to_corners_of_rectangle_center_l2023_202341

theorem min_sum_distances_to_corners_of_rectangle_center (P A B C D : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (1, 0))
  (hC : C = (1, 1))
  (hD : D = (0, 1))
  (hP_center : P = (0.5, 0.5)) :
  ∀ Q, (dist Q A + dist Q B + dist Q C + dist Q D) ≥ (dist P A + dist P B + dist P C + dist P D) := 
sorry

end NUMINAMATH_GPT_min_sum_distances_to_corners_of_rectangle_center_l2023_202341


namespace NUMINAMATH_GPT_f_g_of_1_l2023_202395

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 5 * x + 6
def g (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

-- The statement we need to prove
theorem f_g_of_1 : f (g 1) = 132 := by
  sorry

end NUMINAMATH_GPT_f_g_of_1_l2023_202395


namespace NUMINAMATH_GPT_savings_from_discount_l2023_202394

-- Define the initial price
def initial_price : ℝ := 475.00

-- Define the discounted price
def discounted_price : ℝ := 199.00

-- The theorem to prove the savings amount
theorem savings_from_discount : initial_price - discounted_price = 276.00 :=
by 
  -- This is where the actual proof would go
  sorry

end NUMINAMATH_GPT_savings_from_discount_l2023_202394


namespace NUMINAMATH_GPT_greatest_possible_integer_radius_l2023_202304

theorem greatest_possible_integer_radius :
  ∃ r : ℤ, (50 < (r : ℝ)^2) ∧ ((r : ℝ)^2 < 75) ∧ 
  (∀ s : ℤ, (50 < (s : ℝ)^2) ∧ ((s : ℝ)^2 < 75) → s ≤ r) :=
sorry

end NUMINAMATH_GPT_greatest_possible_integer_radius_l2023_202304


namespace NUMINAMATH_GPT_square_difference_l2023_202362

theorem square_difference : (601^2 - 599^2 = 2400) :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_square_difference_l2023_202362


namespace NUMINAMATH_GPT_sin_sq_sub_cos_sq_l2023_202322

-- Given condition
variable {α : ℝ}
variable (h : Real.sin α = Real.sqrt 5 / 5)

-- Proof goal
theorem sin_sq_sub_cos_sq (h : Real.sin α = Real.sqrt 5 / 5) : Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 := sorry

end NUMINAMATH_GPT_sin_sq_sub_cos_sq_l2023_202322


namespace NUMINAMATH_GPT_square_side_length_l2023_202337

theorem square_side_length (x : ℝ) 
  (h : x^2 = 6^2 + 8^2) : x = 10 := 
by sorry

end NUMINAMATH_GPT_square_side_length_l2023_202337


namespace NUMINAMATH_GPT_polygon_sides_from_diagonals_l2023_202312

theorem polygon_sides_from_diagonals (n : ℕ) (h : ↑((n * (n - 3)) / 2) = 14) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_from_diagonals_l2023_202312


namespace NUMINAMATH_GPT_real_numbers_division_l2023_202308

def is_non_neg (x : ℝ) : Prop := x ≥ 0

theorem real_numbers_division :
  ∀ x : ℝ, x < 0 ∨ is_non_neg x :=
by
  intro x
  by_cases h : x < 0
  · left
    exact h
  · right
    push_neg at h
    exact h

end NUMINAMATH_GPT_real_numbers_division_l2023_202308


namespace NUMINAMATH_GPT_A_wins_match_prob_correct_l2023_202359

def probA_wins_game : ℝ := 0.6
def probB_wins_game : ℝ := 0.4

def probA_wins_match : ℝ :=
  let probA_wins_first_two := probA_wins_game * probA_wins_game
  let probA_wins_first_and_third := probA_wins_game * probB_wins_game * probA_wins_game
  let probA_wins_last_two := probB_wins_game * probA_wins_game * probA_wins_game
  probA_wins_first_two + probA_wins_first_and_third + probA_wins_last_two

theorem A_wins_match_prob_correct : probA_wins_match = 0.648 := by
  sorry

end NUMINAMATH_GPT_A_wins_match_prob_correct_l2023_202359


namespace NUMINAMATH_GPT_order_of_four_l2023_202360

theorem order_of_four {m n p q : ℝ} (hmn : m < n) (hpq : p < q) (h1 : (p - m) * (p - n) < 0) (h2 : (q - m) * (q - n) < 0) : m < p ∧ p < q ∧ q < n :=
by
  sorry

end NUMINAMATH_GPT_order_of_four_l2023_202360


namespace NUMINAMATH_GPT_expression_evaluation_l2023_202378

-- Using the given conditions
def a : ℕ := 3
def b : ℕ := a^2 + 2 * a + 5
def c : ℕ := b^2 - 14 * b + 45

-- We need to assume that none of the denominators are zero.
lemma non_zero_denominators : (a + 1 ≠ 0) ∧ (b - 3 ≠ 0) ∧ (c + 7 ≠ 0) :=
  by {
    -- Proof goes here
  sorry }

theorem expression_evaluation :
  (a = 3) →
  ((a^2 + 2*a + 5) = b) →
  ((b^2 - 14*b + 45) = c) →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (↑(a + 3) / ↑(a + 1) * ↑(b - 1) / ↑(b - 3) * ↑(c + 9) / ↑(c + 7) = 4923 / 2924) :=
  by {
    -- Proof goes here
  sorry }

end NUMINAMATH_GPT_expression_evaluation_l2023_202378


namespace NUMINAMATH_GPT_least_incorrect_option_is_A_l2023_202320

def dozen_units : ℕ := 12
def chairs_needed : ℕ := 4

inductive CompletionOption
| dozen
| dozens
| dozen_of
| dozens_of

def correct_option (op : CompletionOption) : Prop :=
  match op with
  | CompletionOption.dozen => dozen_units >= chairs_needed
  | CompletionOption.dozens => False
  | CompletionOption.dozen_of => False
  | CompletionOption.dozens_of => False

theorem least_incorrect_option_is_A : correct_option CompletionOption.dozen :=
by {
  sorry
}

end NUMINAMATH_GPT_least_incorrect_option_is_A_l2023_202320


namespace NUMINAMATH_GPT_total_and_average_games_l2023_202318

def football_games_per_month : List Nat := [29, 35, 48, 43, 56, 36]
def baseball_games_per_month : List Nat := [15, 19, 23, 14, 18, 17]
def basketball_games_per_month : List Nat := [17, 21, 14, 32, 22, 27]

def total_games (games_per_month : List Nat) : Nat :=
  List.sum games_per_month

def average_games (total : Nat) (months : Nat) : Nat :=
  total / months

theorem total_and_average_games :
  total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month = 486
  ∧ average_games (total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month) 6 = 81 :=
by
  sorry

end NUMINAMATH_GPT_total_and_average_games_l2023_202318


namespace NUMINAMATH_GPT_find_angles_l2023_202309

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  2 * y = x + z

theorem find_angles (a : ℝ) (h1 : 0 < a) (h2 : a < 360)
  (h3 : is_arithmetic_sequence (Real.sin a) (Real.sin (2 * a)) (Real.sin (3 * a))) :
  a = 90 ∨ a = 270 := by
  sorry

end NUMINAMATH_GPT_find_angles_l2023_202309


namespace NUMINAMATH_GPT_scout_earnings_weekend_l2023_202331

-- Define the conditions
def base_pay_per_hour : ℝ := 10.00
def saturday_hours : ℝ := 6
def saturday_customers : ℝ := 5
def saturday_tip_per_customer : ℝ := 5.00
def sunday_hours : ℝ := 8
def sunday_customers_with_3_tip : ℝ := 5
def sunday_customers_with_7_tip : ℝ := 5
def sunday_tip_3_per_customer : ℝ := 3.00
def sunday_tip_7_per_customer : ℝ := 7.00
def overtime_multiplier : ℝ := 1.5

-- Statement to prove earnings for the weekend is $255.00
theorem scout_earnings_weekend : 
  (base_pay_per_hour * saturday_hours + saturday_customers * saturday_tip_per_customer) +
  (base_pay_per_hour * overtime_multiplier * sunday_hours + 
   sunday_customers_with_3_tip * sunday_tip_3_per_customer +
   sunday_customers_with_7_tip * sunday_tip_7_per_customer) = 255 :=
by
  sorry

end NUMINAMATH_GPT_scout_earnings_weekend_l2023_202331


namespace NUMINAMATH_GPT_problem_l2023_202321

noncomputable def x : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem (x y : ℝ) (hx : x = Real.sqrt 3 + Real.sqrt 2) (hy : y = Real.sqrt 3 - Real.sqrt 2) :
  x * y^2 - x^2 * y = -2 * Real.sqrt 2 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_problem_l2023_202321


namespace NUMINAMATH_GPT_sum_areas_of_eight_disks_l2023_202384

noncomputable def eight_disks_sum_areas (C_radius disk_count : ℝ) 
  (cover_C : ℝ) (no_overlap : ℝ) (tangent_neighbors : ℝ) : ℕ :=
  let r := (2 - Real.sqrt 2)
  let area_one_disk := Real.pi * r^2
  let total_area := disk_count * area_one_disk
  let a := 48
  let b := 32
  let c := 2
  a + b + c

theorem sum_areas_of_eight_disks : eight_disks_sum_areas 1 8 1 1 1 = 82 :=
  by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_sum_areas_of_eight_disks_l2023_202384


namespace NUMINAMATH_GPT_combined_age_in_ten_years_l2023_202314

theorem combined_age_in_ten_years (B A: ℕ) (hA : A = 20) (h1: A + 10 = 2 * (B + 10)): 
  (A + 10) + (B + 10) = 45 := 
by
  sorry

end NUMINAMATH_GPT_combined_age_in_ten_years_l2023_202314


namespace NUMINAMATH_GPT_eval_f_neg_2_l2023_202319

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem eval_f_neg_2 : f (-2) = 19 :=
by
  sorry

end NUMINAMATH_GPT_eval_f_neg_2_l2023_202319


namespace NUMINAMATH_GPT_todd_runs_faster_l2023_202328

-- Define the times taken by Brian and Todd
def brian_time : ℕ := 96
def todd_time : ℕ := 88

-- The theorem stating the problem
theorem todd_runs_faster : brian_time - todd_time = 8 :=
by
  -- Solution here
  sorry

end NUMINAMATH_GPT_todd_runs_faster_l2023_202328


namespace NUMINAMATH_GPT_p_implies_q_not_q_implies_p_l2023_202323

def p (a : ℝ) := a = Real.sqrt 2

def q (a : ℝ) := ∀ x y : ℝ, y = -(x : ℝ) → (x^2 + (y - a)^2 = 1)

theorem p_implies_q_not_q_implies_p (a : ℝ) : (p a → q a) ∧ (¬(q a → p a)) := 
    sorry

end NUMINAMATH_GPT_p_implies_q_not_q_implies_p_l2023_202323


namespace NUMINAMATH_GPT_total_savings_over_12_weeks_l2023_202306

-- Define the weekly savings and durations for each period
def weekly_savings_period_1 : ℕ := 5
def duration_period_1 : ℕ := 4

def weekly_savings_period_2 : ℕ := 10
def duration_period_2 : ℕ := 4

def weekly_savings_period_3 : ℕ := 20
def duration_period_3 : ℕ := 4

-- Define the total savings calculation for each period
def total_savings_period_1 : ℕ := weekly_savings_period_1 * duration_period_1
def total_savings_period_2 : ℕ := weekly_savings_period_2 * duration_period_2
def total_savings_period_3 : ℕ := weekly_savings_period_3 * duration_period_3

-- Prove that the total savings over 12 weeks equals $140.00
theorem total_savings_over_12_weeks : total_savings_period_1 + total_savings_period_2 + total_savings_period_3 = 140 := 
by 
  sorry

end NUMINAMATH_GPT_total_savings_over_12_weeks_l2023_202306
