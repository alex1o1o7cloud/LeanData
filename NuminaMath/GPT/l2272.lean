import Mathlib

namespace number_of_blue_stamps_l2272_227216

theorem number_of_blue_stamps (
    red_stamps : ℕ := 20
) (
    yellow_stamps : ℕ := 7
) (
    price_per_red_stamp : ℝ := 1.1
) (
    price_per_blue_stamp : ℝ := 0.8
) (
    total_earnings : ℝ := 100
) (
    price_per_yellow_stamp : ℝ := 2
) : red_stamps = 20 ∧ yellow_stamps = 7 ∧ price_per_red_stamp = 1.1 ∧ price_per_blue_stamp = 0.8 ∧ total_earnings = 100 ∧ price_per_yellow_stamp = 2 → ∃ (blue_stamps : ℕ), blue_stamps = 80 :=
by
  sorry

end number_of_blue_stamps_l2272_227216


namespace correct_calculation_l2272_227229

theorem correct_calculation :
  (-7 * a * b^2 + 4 * a * b^2 = -3 * a * b^2) ∧
  ¬ (2 * x + 3 * y = 5 * x * y) ∧
  ¬ (6 * x^2 - (-x^2) = 5 * x^2) ∧
  ¬ (4 * m * n - 3 * m * n = 1) :=
by
  sorry

end correct_calculation_l2272_227229


namespace find_m_l2272_227287

theorem find_m (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 2 * y - 4 = 0) →
  (∀ x y : ℝ, x - 2 * y + m = 0) →
  (m = 7 ∨ m = -3) :=
by
  sorry

end find_m_l2272_227287


namespace exam_results_l2272_227248

variable (E F G H : Prop)

def emma_statement : Prop := E → F
def frank_statement : Prop := F → ¬G
def george_statement : Prop := G → H
def exactly_two_asing : Prop :=
  (E ∧ F ∧ ¬G ∧ ¬H) ∨ (¬E ∧ F ∧ G ∧ ¬H) ∨
  (¬E ∧ ¬F ∧ G ∧ H) ∨ (¬E ∧ F ∧ ¬G ∧ H) ∨
  (E ∧ ¬F ∧ ¬G ∧ H)

theorem exam_results :
  (E ∧ F) ∨ (G ∧ H) :=
by {
  sorry
}

end exam_results_l2272_227248


namespace gabby_needs_more_money_l2272_227234

theorem gabby_needs_more_money (cost_saved : ℕ) (initial_saved : ℕ) (additional_money : ℕ) (cost_remaining : ℕ) :
  cost_saved = 65 → initial_saved = 35 → additional_money = 20 → cost_remaining = (cost_saved - initial_saved) - additional_money → cost_remaining = 10 :=
by
  intros h_cost_saved h_initial_saved h_additional_money h_cost_remaining
  simp [h_cost_saved, h_initial_saved, h_additional_money] at h_cost_remaining
  exact h_cost_remaining

end gabby_needs_more_money_l2272_227234


namespace probability_calculation_l2272_227290

noncomputable def probability_at_least_seven_at_least_three_times : ℚ :=
  let p := 1 / 4
  let q := 3 / 4
  (4 * p^3 * q) + (p^4)

theorem probability_calculation :
  probability_at_least_seven_at_least_three_times = 13 / 256 :=
by sorry

end probability_calculation_l2272_227290


namespace largest_divisor_of_expression_l2272_227262

theorem largest_divisor_of_expression (x : ℤ) (h : x % 2 = 1) : 
  324 ∣ (12 * x + 3) * (12 * x + 9) * (6 * x + 6) :=
sorry

end largest_divisor_of_expression_l2272_227262


namespace total_stock_worth_is_15000_l2272_227285

-- Define the total worth of the stock
variable (X : ℝ)

-- Define the conditions
def stock_condition_1 := 0.20 * X -- Worth of 20% of the stock
def stock_condition_2 := 0.10 * (0.20 * X) -- Profit from 20% of the stock
def stock_condition_3 := 0.80 * X -- Worth of 80% of the stock
def stock_condition_4 := 0.05 * (0.80 * X) -- Loss from 80% of the stock
def overall_loss := 0.04 * X - 0.02 * X

-- The question rewritten as a theorem statement
theorem total_stock_worth_is_15000 (h1 : overall_loss X = 300) : X = 15000 :=
by sorry

end total_stock_worth_is_15000_l2272_227285


namespace max_brownies_l2272_227281

theorem max_brownies (m n : ℕ) (h : (m - 2) * (n - 2) = 2 * m + 2 * n - 4) : m * n ≤ 60 :=
sorry

end max_brownies_l2272_227281


namespace gaussian_guardians_points_l2272_227264

theorem gaussian_guardians_points :
  let Daniel := 7
  let Curtis := 8
  let Sid := 2
  let Emily := 11
  let Kalyn := 6
  let Hyojeong := 12
  let Ty := 1
  let Winston := 7
  Daniel + Curtis + Sid + Emily + Kalyn + Hyojeong + Ty + Winston = 54 :=
by
  sorry

end gaussian_guardians_points_l2272_227264


namespace problem1_problem2_l2272_227247

-- Problem (1)
theorem problem1 (a : ℝ) (h : a = 1) (p q : ℝ → Prop) 
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0) 
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1) :
  (∀ x, (p x ∧ q x) ↔ (2 < x ∧ x < 3)) :=
by sorry

-- Problem (2)
theorem problem2 (a : ℝ) (p q : ℝ → Prop)
  (hp : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (hq : ∀ x, q x ↔ (x - 3)^2 < 1)
  (hnpc : ∀ x, ¬p x → ¬q x) 
  (hnpc_not_necessary : ∃ x, ¬p x ∧ q x) :
  (4 / 3 ≤ a ∧ a ≤ 2) :=
by sorry

end problem1_problem2_l2272_227247


namespace NinaCalculationCorrectAnswer_l2272_227226

variable (y : ℝ)

noncomputable def NinaMistakenCalculation (y : ℝ) : ℝ :=
(y + 25) * 5

noncomputable def NinaCorrectCalculation (y : ℝ) : ℝ :=
(y - 25) / 5

theorem NinaCalculationCorrectAnswer (hy : (NinaMistakenCalculation y) = 200) :
  (NinaCorrectCalculation y) = -2 := by
  sorry

end NinaCalculationCorrectAnswer_l2272_227226


namespace new_commission_percentage_l2272_227259

theorem new_commission_percentage
  (fixed_salary : ℝ)
  (total_sales : ℝ)
  (sales_threshold : ℝ)
  (previous_commission_rate : ℝ)
  (additional_earnings : ℝ)
  (prev_commission : ℝ)
  (extra_sales : ℝ)
  (new_commission : ℝ)
  (new_remuneration : ℝ) :
  fixed_salary = 1000 →
  total_sales = 12000 →
  sales_threshold = 4000 →
  previous_commission_rate = 0.05 →
  additional_earnings = 600 →
  prev_commission = previous_commission_rate * total_sales →
  extra_sales = total_sales - sales_threshold →
  new_remuneration = fixed_salary + new_commission * extra_sales →
  new_remuneration = prev_commission + additional_earnings →
  new_commission = 2.5 / 100 :=
by
  intros
  sorry

end new_commission_percentage_l2272_227259


namespace count_valid_five_digit_numbers_l2272_227228

-- Define the conditions
def is_five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def is_divisible_by (a b : ℕ) : Prop := b ∣ a

def quotient_remainder_sum_divisible_by (n q r : ℕ) : Prop :=
  (n = 100 * q + r) ∧ ((q + r) % 7 = 0)

-- Define the theorem
theorem count_valid_five_digit_numbers : 
  ∃ k, k = 8160 ∧ ∀ n, is_five_digit_number n ∧ 
    is_divisible_by n 13 ∧ 
    ∃ q r, quotient_remainder_sum_divisible_by n q r → 
    k = 8160 :=
sorry

end count_valid_five_digit_numbers_l2272_227228


namespace no_common_points_implies_parallel_l2272_227206

variable (a : Type) (P : Type) [LinearOrder P] [AddGroupWithOne P]
variable (has_no_common_point : a → P → Prop)
variable (is_parallel : a → P → Prop)

theorem no_common_points_implies_parallel (a_line : a) (a_plane : P) :
  has_no_common_point a_line a_plane ↔ is_parallel a_line a_plane :=
sorry

end no_common_points_implies_parallel_l2272_227206


namespace train_cross_pole_time_l2272_227215

def speed_kmh := 90 -- speed of the train in km/hr
def length_m := 375 -- length of the train in meters

/-- Convert speed from km/hr to m/s -/
def convert_speed (v_kmh : ℕ) : ℕ := v_kmh * 1000 / 3600

/-- Calculate the time it takes for the train to cross the pole -/
def time_to_cross_pole (length_m : ℕ) (speed_m_s : ℕ) : ℕ := length_m / speed_m_s

theorem train_cross_pole_time :
  time_to_cross_pole length_m (convert_speed speed_kmh) = 15 :=
by
  sorry

end train_cross_pole_time_l2272_227215


namespace unique_sequence_count_l2272_227279

def is_valid_sequence (a : Fin 5 → ℕ) :=
  a 0 = 1 ∧
  a 1 > a 0 ∧
  a 2 > a 1 ∧
  a 3 > a 2 ∧
  a 4 = 15 ∧
  (a 1) ^ 2 ≤ a 0 * a 2 + 1 ∧
  (a 2) ^ 2 ≤ a 1 * a 3 + 1 ∧
  (a 3) ^ 2 ≤ a 2 * a 4 + 1

theorem unique_sequence_count : 
  ∃! (a : Fin 5 → ℕ), is_valid_sequence a :=
sorry

end unique_sequence_count_l2272_227279


namespace sum_of_coeffs_l2272_227295

theorem sum_of_coeffs (a_5 a_4 a_3 a_2 a_1 a : ℤ) (h_eq : (x - 2)^5 = a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) (h_a : a = -32) :
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end sum_of_coeffs_l2272_227295


namespace vanessa_savings_weeks_l2272_227204

-- Definitions of given conditions
def dress_cost : ℕ := 80
def vanessa_savings : ℕ := 20
def weekly_allowance : ℕ := 30
def weekly_spending : ℕ := 10

-- Required amount to save 
def required_savings : ℕ := dress_cost - vanessa_savings

-- Weekly savings calculation
def weekly_savings : ℕ := weekly_allowance - weekly_spending

-- Number of weeks needed to save the required amount
def weeks_needed_to_save (required_savings weekly_savings : ℕ) : ℕ :=
  required_savings / weekly_savings

-- Axiom representing the correctness of our calculation
theorem vanessa_savings_weeks : weeks_needed_to_save required_savings weekly_savings = 3 := 
  by
  sorry

end vanessa_savings_weeks_l2272_227204


namespace smallest_range_of_sample_l2272_227233

open Real

theorem smallest_range_of_sample {a b c d e f g : ℝ}
  (h1 : (a + b + c + d + e + f + g) / 7 = 8)
  (h2 : d = 10)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e ∧ e ≤ f ∧ f ≤ g) :
  ∃ r, r = g - a ∧ r = 8 :=
by
  sorry

end smallest_range_of_sample_l2272_227233


namespace div_by_10_3pow_l2272_227275

theorem div_by_10_3pow
    (m : ℤ)
    (n : ℕ)
    (h : (3^n + m) % 10 = 0) :
    (3^(n + 4) + m) % 10 = 0 := by
  sorry

end div_by_10_3pow_l2272_227275


namespace solution_l2272_227254

noncomputable def f : ℝ → ℝ := sorry

lemma problem_conditions:
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (-x + 1) = f (x + 1)) ∧ f (-1) = 1 :=
sorry

theorem solution : f 2017 = -1 :=
sorry

end solution_l2272_227254


namespace least_x_l2272_227243

theorem least_x (x p : ℕ) (h1 : 0 < x) (h2: Nat.Prime p) (h3: ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 11 * p * q) : x ≥ 66 := 
sorry

end least_x_l2272_227243


namespace jindra_gray_fields_counts_l2272_227211

-- Definitions for the problem setup
noncomputable def initial_gray_fields: ℕ := 7
noncomputable def rotation_90_gray_fields: ℕ := 8
noncomputable def rotation_180_gray_fields: ℕ := 4

-- Statement of the theorem to be proved
theorem jindra_gray_fields_counts:
  initial_gray_fields = 7 ∧
  rotation_90_gray_fields = 8 ∧
  rotation_180_gray_fields = 4 := by
  sorry

end jindra_gray_fields_counts_l2272_227211


namespace specialCollectionAtEndOfMonth_l2272_227223

noncomputable def specialCollectionBooksEndOfMonth (initialBooks loanedBooks returnedPercentage : ℕ) :=
  initialBooks - (loanedBooks - loanedBooks * returnedPercentage / 100)

theorem specialCollectionAtEndOfMonth :
  specialCollectionBooksEndOfMonth 150 80 65 = 122 :=
by
  sorry

end specialCollectionAtEndOfMonth_l2272_227223


namespace min_vertical_distance_between_graphs_l2272_227299

noncomputable def min_distance (x : ℝ) : ℝ :=
  |x| - (-x^2 - 4 * x - 2)

theorem min_vertical_distance_between_graphs :
  ∃ x : ℝ, ∀ y : ℝ, min_distance x ≤ min_distance y := 
    sorry

end min_vertical_distance_between_graphs_l2272_227299


namespace one_eighth_of_2_pow_33_eq_2_pow_x_l2272_227253

theorem one_eighth_of_2_pow_33_eq_2_pow_x (x : ℕ) : (1 / 8) * (2 : ℝ) ^ 33 = (2 : ℝ) ^ x → x = 30 := by
  intro h
  sorry

end one_eighth_of_2_pow_33_eq_2_pow_x_l2272_227253


namespace power_of_power_evaluation_l2272_227208

theorem power_of_power_evaluation : (3^3)^2 = 729 := 
by
  -- Replace this with the actual proof
  sorry

end power_of_power_evaluation_l2272_227208


namespace remainder_when_ab_div_by_40_l2272_227296

theorem remainder_when_ab_div_by_40 (a b : ℤ) (k j : ℤ)
  (ha : a = 80 * k + 75)
  (hb : b = 90 * j + 85):
  (a + b) % 40 = 0 :=
by sorry

end remainder_when_ab_div_by_40_l2272_227296


namespace n_c_equation_l2272_227277

theorem n_c_equation (n c : ℕ) (hn : 0 < n) (hc : 0 < c) :
  (∀ x : ℕ, (↑x + n * ↑x / 100) * (1 - c / 100) = x) →
  (n^2 / c^2 = (100 + n) / (100 - c)) :=
by sorry

end n_c_equation_l2272_227277


namespace interval_for_x_l2272_227232

theorem interval_for_x (x : ℝ) 
  (hx1 : 1/x < 2) 
  (hx2 : 1/x > -3) : 
  x > 1/2 ∨ x < -1/3 :=
  sorry

end interval_for_x_l2272_227232


namespace cube_edge_percentage_growth_l2272_227249

theorem cube_edge_percentage_growth (p : ℝ) 
  (h : (1 + p / 100) ^ 2 - 1 = 0.96) : p = 40 :=
by
  sorry

end cube_edge_percentage_growth_l2272_227249


namespace Maria_score_l2272_227255

theorem Maria_score (x : ℝ) (y : ℝ) (h1 : x = y + 50) (h2 : (x + y) / 2 = 105) : x = 130 :=
by
  sorry

end Maria_score_l2272_227255


namespace find_exercise_books_l2272_227261

theorem find_exercise_books
  (pencil_ratio pen_ratio exercise_book_ratio eraser_ratio : ℕ)
  (total_pencils total_ratio_units : ℕ)
  (h1 : pencil_ratio = 10)
  (h2 : pen_ratio = 2)
  (h3 : exercise_book_ratio = 3)
  (h4 : eraser_ratio = 4)
  (h5 : total_pencils = 150)
  (h6 : total_ratio_units = pencil_ratio + pen_ratio + exercise_book_ratio + eraser_ratio) :
  (total_pencils / pencil_ratio) * exercise_book_ratio = 45 :=
by
  sorry

end find_exercise_books_l2272_227261


namespace solve_for_g2_l2272_227238

-- Let g : ℝ → ℝ be a function satisfying the given condition
variable (g : ℝ → ℝ)

-- The given condition
def condition (x : ℝ) : Prop :=
  g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The main theorem we aim to prove
theorem solve_for_g2 (h : ∀ x, condition g x) : g 2 = 0 :=
by
  sorry

end solve_for_g2_l2272_227238


namespace mask_digit_identification_l2272_227286

theorem mask_digit_identification :
  ∃ (elephant_mask mouse_mask pig_mask panda_mask : ℕ),
    (4 * 4 = 16) ∧
    (7 * 7 = 49) ∧
    (8 * 8 = 64) ∧
    (9 * 9 = 81) ∧
    elephant_mask = 6 ∧
    mouse_mask = 4 ∧
    pig_mask = 8 ∧
    panda_mask = 1 :=
by
  sorry

end mask_digit_identification_l2272_227286


namespace base10_to_base7_l2272_227239

theorem base10_to_base7 : 
  ∃ a b c d : ℕ, a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 = 729 ∧ a = 2 ∧ b = 0 ∧ c = 6 ∧ d = 1 :=
sorry

end base10_to_base7_l2272_227239


namespace determine_a_l2272_227251

theorem determine_a (a : ℝ) (h : ∃ r : ℝ, (a / (1+1*I : ℂ) + (1+1*I : ℂ) / 2).im = 0) : a = 1 :=
sorry

end determine_a_l2272_227251


namespace red_marble_count_l2272_227292

theorem red_marble_count (x y : ℕ) (total_yellow : ℕ) (total_diff : ℕ) 
  (jar1_ratio_red jar1_ratio_yellow : ℕ) (jar2_ratio_red jar2_ratio_yellow : ℕ) 
  (h1 : jar1_ratio_red = 7) (h2 : jar1_ratio_yellow = 2) 
  (h3 : jar2_ratio_red = 5) (h4 : jar2_ratio_yellow = 3) 
  (h5 : 2 * x + 3 * y = 50) (h6 : 8 * y = 9 * x + 20) :
  7 * x + 2 = 5 * y :=
sorry

end red_marble_count_l2272_227292


namespace sum_of_plane_angles_l2272_227242

theorem sum_of_plane_angles (v f p : ℕ) (h : v = p) :
    (2 * π * (v - f) = 2 * π * (p - 2)) :=
by sorry

end sum_of_plane_angles_l2272_227242


namespace hyperbola_equation_l2272_227269

-- Definition of the ellipse given in the problem
def ellipse (x y : ℝ) := y^2 / 5 + x^2 = 1

-- Definition of the conditions for the hyperbola:
-- 1. The hyperbola shares a common focus with the ellipse.
-- 2. Distance from the focus to the asymptote of the hyperbola is 1.
def hyperbola (x y : ℝ) (c : ℝ) :=
  ∃ a b : ℝ, c = 2 ∧ a^2 + b^2 = c^2 ∧
             (b = 1 ∧ y = if x = 0 then 0 else x * (a / b))

-- The statement we need to prove
theorem hyperbola_equation : 
  (∃ a b : ℝ, ellipse x y ∧ hyperbola x y 2 ∧ b = 1 ∧ a^2 = 3) → 
  (y^2 / 3 - x^2 = 1) :=
sorry

end hyperbola_equation_l2272_227269


namespace compound_difference_l2272_227270

noncomputable def monthly_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let monthly_rate := annual_rate / 12
  let periods := 12 * years
  principal * (1 + monthly_rate) ^ periods

noncomputable def semi_annual_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let semi_annual_rate := annual_rate / 2
  let periods := 2 * years
  principal * (1 + semi_annual_rate) ^ periods

theorem compound_difference (principal : ℝ) (annual_rate : ℝ) (years : ℝ) :
  monthly_compound_amount principal annual_rate years - semi_annual_compound_amount principal annual_rate years = 23.36 :=
by
  let principal := 8000
  let annual_rate := 0.08
  let years := 3
  sorry

end compound_difference_l2272_227270


namespace distinct_negative_real_roots_l2272_227294

def poly (p : ℝ) (x : ℝ) : ℝ := x^4 + 2*p*x^3 + x^2 + 2*p*x + 1

theorem distinct_negative_real_roots (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ poly p x1 = 0 ∧ poly p x2 = 0) ↔ p > 3/4 :=
sorry

end distinct_negative_real_roots_l2272_227294


namespace abs_value_condition_l2272_227240

theorem abs_value_condition (m : ℝ) (h : |m - 1| = m - 1) : m ≥ 1 :=
by {
  sorry
}

end abs_value_condition_l2272_227240


namespace Bryan_did_258_pushups_l2272_227230

-- Define the conditions
def sets : ℕ := 15
def pushups_per_set : ℕ := 18
def pushups_fewer_last_set : ℕ := 12

-- Define the planned total push-ups
def planned_total_pushups : ℕ := sets * pushups_per_set

-- Define the actual push-ups in the last set
def last_set_pushups : ℕ := pushups_per_set - pushups_fewer_last_set

-- Define the total push-ups Bryan did
def total_pushups : ℕ := (sets - 1) * pushups_per_set + last_set_pushups

-- The theorem to prove
theorem Bryan_did_258_pushups :
  total_pushups = 258 := by
  sorry

end Bryan_did_258_pushups_l2272_227230


namespace greatest_odd_factors_l2272_227278

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end greatest_odd_factors_l2272_227278


namespace range_of_cos_neg_alpha_l2272_227225

theorem range_of_cos_neg_alpha (α : ℝ) (h : 12 * (Real.sin α)^2 + Real.cos α > 11) :
  -1 / 4 < Real.cos (-α) ∧ Real.cos (-α) < 1 / 3 := 
sorry

end range_of_cos_neg_alpha_l2272_227225


namespace min_value_expression_l2272_227201

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 8) : 
  (x + 3 * y) * (y + 3 * z) * (3 * x * z + 1) ≥ 72 :=
sorry

end min_value_expression_l2272_227201


namespace annual_interest_income_l2272_227221

variables (totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate : ℝ)
           (firstInterest secondInterest totalInterest : ℝ)

def investment_conditions : Prop :=
  totalInvestment = 32000 ∧
  firstRate = 0.0575 ∧
  secondRate = 0.0625 ∧
  firstBondPrincipal = 20000 ∧
  secondBondPrincipal = totalInvestment - firstBondPrincipal

def calculate_interest (principal rate : ℝ) : ℝ := principal * rate

def total_annual_interest (firstInterest secondInterest : ℝ) : ℝ :=
  firstInterest + secondInterest

theorem annual_interest_income
  (hc : investment_conditions totalInvestment firstBondPrincipal secondBondPrincipal firstRate secondRate) :
  total_annual_interest (calculate_interest firstBondPrincipal firstRate)
    (calculate_interest secondBondPrincipal secondRate) = 1900 :=
by {
  sorry
}

end annual_interest_income_l2272_227221


namespace jennifer_initial_pears_l2272_227200

def initialPears (P: ℕ) : Prop := (P + 20 + 2 * P - 6 = 44)

theorem jennifer_initial_pears (P: ℕ) (h : initialPears P) : P = 10 := by
  sorry

end jennifer_initial_pears_l2272_227200


namespace greatest_divisor_under_100_l2272_227246

theorem greatest_divisor_under_100 (d : ℕ) :
  d ∣ 780 ∧ d < 100 ∧ d ∣ 180 ∧ d ∣ 240 ↔ d ≤ 60 := by
  sorry

end greatest_divisor_under_100_l2272_227246


namespace inequality_solution_l2272_227271

theorem inequality_solution (x : ℝ) : 
  x^2 - 9 * x + 20 < 1 ↔ (9 - Real.sqrt 5) / 2 < x ∧ x < (9 + Real.sqrt 5) / 2 := 
by
  sorry

end inequality_solution_l2272_227271


namespace binary_to_base4_conversion_l2272_227217

theorem binary_to_base4_conversion : 
  let binary := (1*2^7 + 1*2^6 + 0*2^5 + 1*2^4 + 1*2^3 + 0*2^2 + 0*2^1 + 1*2^0) 
  let base4 := (3*4^3 + 1*4^2 + 2*4^1 + 1*4^0)
  binary = base4 := by
  sorry

end binary_to_base4_conversion_l2272_227217


namespace total_distance_dog_runs_l2272_227280

-- Define the distance between Xiaoqiang's home and his grandmother's house in meters
def distance_home_to_grandma : ℕ := 1000

-- Define Xiaoqiang's walking speed in meters per minute
def xiaoqiang_speed : ℕ := 50

-- Define the dog's running speed in meters per minute
def dog_speed : ℕ := 200

-- Define the time Xiaoqiang takes to reach his grandmother's house
def xiaoqiang_time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the total distance the dog runs given the speeds and distances
theorem total_distance_dog_runs (d x_speed dog_speed : ℕ) 
  (hx : x_speed > 0) (hd : dog_speed > 0) : (d / x_speed) * dog_speed = 4000 :=
  sorry

end total_distance_dog_runs_l2272_227280


namespace range_of_x_l2272_227288

theorem range_of_x (x : ℝ) (h : 2 * x + 1 ≤ 0) : x ≤ -1 / 2 := 
  sorry

end range_of_x_l2272_227288


namespace factor_check_l2272_227227

theorem factor_check :
  ∃ (f : ℕ → ℕ) (x : ℝ), f 1 = (x^2 - 2 * x + 3) ∧ f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 :=
by
  let f : ℕ → ℕ := sorry -- Define a sequence or function for the proof context
  let x : ℝ := sorry -- Define the variable x in our context
  have h₁ : f 1 = (x^2 - 2 * x + 3) := sorry -- Establish the first factor
  have h₂ : f 2 = 29 * 37 * x^4 + 2 * x^2 + 9 := sorry -- Establish the polynomial expression
  exact ⟨f, x, h₁, h₂⟩ -- Use existential quantifier to capture the required form

end factor_check_l2272_227227


namespace line_tangent_to_circle_l2272_227291

theorem line_tangent_to_circle {m : ℝ} : 
  (∀ x y : ℝ, y = m * x) → (∀ x y : ℝ, x^2 + y^2 - 4 * x + 2 = 0) → 
  (m = 1 ∨ m = -1) := 
by 
  sorry

end line_tangent_to_circle_l2272_227291


namespace shaded_fraction_l2272_227214

theorem shaded_fraction {S : ℝ} (h : 0 < S) :
  let frac_area := ∑' n : ℕ, (1/(4:ℝ)^1) * (1/(4:ℝ)^n)
  1/3 = frac_area :=
by
  sorry

end shaded_fraction_l2272_227214


namespace express_y_l2272_227250

theorem express_y (x y : ℝ) (h : 3 * x + 2 * y = 1) : y = (1 - 3 * x) / 2 :=
by {
  sorry
}

end express_y_l2272_227250


namespace total_heads_l2272_227252

variables (H C : ℕ)

theorem total_heads (h_hens: H = 22) (h_feet: 2 * H + 4 * C = 140) : H + C = 46 :=
by
  sorry

end total_heads_l2272_227252


namespace common_ratio_of_geometric_sequence_l2272_227203

variables (a : ℕ → ℝ) (q : ℝ)
axiom h1 : a 1 = 2
axiom h2 : ∀ n : ℕ, a (n + 1) - a n ≠ 0 -- Common difference is non-zero
axiom h3 : a 3 = (a 1) * q
axiom h4 : a 11 = (a 1) * q^2
axiom h5 : a 11 = a 1 + 5 * (a 3 - a 1)

theorem common_ratio_of_geometric_sequence : q = 4 := 
by sorry

end common_ratio_of_geometric_sequence_l2272_227203


namespace sum_of_integers_ending_in_2_between_100_and_600_l2272_227222

theorem sum_of_integers_ending_in_2_between_100_and_600 :
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  ∃ S : ℤ, S = n * (a + l) / 2 ∧ S = 17350 := 
by
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  use n * (a + l) / 2
  sorry

end sum_of_integers_ending_in_2_between_100_and_600_l2272_227222


namespace correct_substitution_l2272_227276

theorem correct_substitution (x y : ℝ) (h1 : y = 1 - x) (h2 : x - 2 * y = 4) : x - 2 * (1 - x) = 4 → x - 2 + 2 * x = 4 := by
  sorry

end correct_substitution_l2272_227276


namespace arithmetic_sequence_sum_l2272_227268

-- Define the variables and conditions
def a : ℕ := 71
def d : ℕ := 2
def l : ℕ := 99

-- Calculate the number of terms in the sequence
def n : ℕ := ((l - a) / d) + 1

-- Define the sum of the arithmetic sequence
def S : ℕ := (n * (a + l)) / 2

-- Statement to be proven
theorem arithmetic_sequence_sum :
  3 * S = 3825 :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_sum_l2272_227268


namespace circumference_of_circle_inscribing_rectangle_l2272_227267

theorem circumference_of_circle_inscribing_rectangle (a b : ℝ) (h₁ : a = 9) (h₂ : b = 12) :
  ∃ C : ℝ, C = 15 * Real.pi := by
  sorry

end circumference_of_circle_inscribing_rectangle_l2272_227267


namespace solve_system_of_equations_l2272_227283

theorem solve_system_of_equations :
  ∃ (x y z w : ℤ), 
    x - y + z - w = 2 ∧
    x^2 - y^2 + z^2 - w^2 = 6 ∧
    x^3 - y^3 + z^3 - w^3 = 20 ∧
    x^4 - y^4 + z^4 - w^4 = 66 ∧
    (x, y, z, w) = (1, 3, 0, 2) := 
  by
    sorry

end solve_system_of_equations_l2272_227283


namespace proof_f_3_eq_9_ln_3_l2272_227213

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x

theorem proof_f_3_eq_9_ln_3 (a : ℝ) (h : deriv (deriv (f a)) 1 = 3) : f a 3 = 9 * Real.log 3 :=
by
  sorry

end proof_f_3_eq_9_ln_3_l2272_227213


namespace polynomial_identity_l2272_227219

theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = 
  (a - b) * (b - c) * (c - a) * (a + b + c) :=
sorry

end polynomial_identity_l2272_227219


namespace simplify_expression_l2272_227265

theorem simplify_expression (x : ℝ) :
  (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * (x - 1) + 1 = x^4 :=
sorry

end simplify_expression_l2272_227265


namespace regular_polygon_num_sides_l2272_227274

def diag_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem regular_polygon_num_sides (n : ℕ) (h : diag_formula n = 20) : n = 8 :=
by
  sorry

end regular_polygon_num_sides_l2272_227274


namespace width_of_domain_of_g_l2272_227289

variable (h : ℝ → ℝ) (dom_h : ∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x)

noncomputable def g (x : ℝ) : ℝ := h (x / 3)

theorem width_of_domain_of_g :
  (∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x) →
  (∀ y : ℝ, -30 ≤ y ∧ y ≤ 30 → h (y / 3) = h (y / 3)) →
  (∃ a b : ℝ, a = -30 ∧ b = 30 ∧  (∃ w : ℝ, w = b - a ∧ w = 60)) :=
by
  sorry

end width_of_domain_of_g_l2272_227289


namespace eldest_sibling_age_correct_l2272_227235

-- Definitions and conditions
def youngest_sibling_age (x : ℝ) := x
def second_youngest_sibling_age (x : ℝ) := x + 4
def third_youngest_sibling_age (x : ℝ) := x + 8
def fourth_youngest_sibling_age (x : ℝ) := x + 12
def fifth_youngest_sibling_age (x : ℝ) := x + 16
def sixth_youngest_sibling_age (x : ℝ) := x + 20
def seventh_youngest_sibling_age (x : ℝ) := x + 28
def eldest_sibling_age (x : ℝ) := x + 32

def combined_age_of_eight_siblings (x : ℝ) : ℝ := 
  youngest_sibling_age x +
  second_youngest_sibling_age x +
  third_youngest_sibling_age x +
  fourth_youngest_sibling_age x +
  fifth_youngest_sibling_age x +
  sixth_youngest_sibling_age x +
  seventh_youngest_sibling_age x +
  eldest_sibling_age x

-- Proving the combined age part
theorem eldest_sibling_age_correct (x : ℝ) (h : combined_age_of_eight_siblings x - youngest_sibling_age (x + 24) = 140) : 
  eldest_sibling_age x = 34.5 := by
  sorry

end eldest_sibling_age_correct_l2272_227235


namespace two_numbers_equal_l2272_227297

variables {a b c : ℝ}
variable (h1 : a + b^2 + c^2 = a^2 + b + c^2)
variable (h2 : a^2 + b + c^2 = a^2 + b^2 + c)

theorem two_numbers_equal (h1 : a + b^2 + c^2 = a^2 + b + c^2) (h2 : a^2 + b + c^2 = a^2 + b^2 + c) :
  a = b ∨ a = c ∨ b = c :=
by
  sorry

end two_numbers_equal_l2272_227297


namespace area_of_rectangle_l2272_227263

theorem area_of_rectangle (A G Y : ℝ) 
  (hG : G = 0.15 * A) 
  (hY : Y = 21) 
  (hG_plus_Y : G + Y = 0.5 * A) : 
  A = 60 := 
by 
  -- proof goes here
  sorry

end area_of_rectangle_l2272_227263


namespace gcd_160_200_360_l2272_227236

theorem gcd_160_200_360 : Nat.gcd (Nat.gcd 160 200) 360 = 40 := by
  sorry

end gcd_160_200_360_l2272_227236


namespace find_total_income_l2272_227209

theorem find_total_income (I : ℝ)
  (h1 : 0.6 * I + 0.3 * I + 0.005 * (I - (0.6 * I + 0.3 * I)) + 50000 = I) : 
  I = 526315.79 :=
by
  sorry

end find_total_income_l2272_227209


namespace problem_solution_l2272_227212

theorem problem_solution :
  (19 * 19 - 12 * 12) / ((19 / 12) - (12 / 19)) = 228 :=
by sorry

end problem_solution_l2272_227212


namespace area_of_rhombus_l2272_227218

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 10) : 
  1 / 2 * d1 * d2 = 30 :=
by 
  rw [h1, h2]
  norm_num

end area_of_rhombus_l2272_227218


namespace x2004_y2004_l2272_227298

theorem x2004_y2004 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
  x^2004 + y^2004 = 2^2004 := 
by
  sorry

end x2004_y2004_l2272_227298


namespace solve_for_y_l2272_227224

theorem solve_for_y (y : ℚ) : 
  y + 5 / 8 = 2 / 9 + 1 / 2 → 
  y = 7 / 72 := 
by 
  intro h1
  sorry

end solve_for_y_l2272_227224


namespace a679b_multiple_of_72_l2272_227260

-- Define conditions
def is_divisible_by_8 (n : Nat) : Prop :=
  n % 8 = 0

def sum_of_digits_is_divisible_by_9 (n : Nat) : Prop :=
  (n.digits 10).sum % 9 = 0

-- Define the given problem
theorem a679b_multiple_of_72 (a b : Nat) : 
  is_divisible_by_8 (7 * 100 + 9 * 10 + b) →
  sum_of_digits_is_divisible_by_9 (a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b) → 
  a = 3 ∧ b = 2 :=
by 
  sorry

end a679b_multiple_of_72_l2272_227260


namespace circle_eq_tangent_x_axis_l2272_227272

theorem circle_eq_tangent_x_axis (h k r : ℝ) (x y : ℝ)
  (h_center : h = -5)
  (k_center : k = 4)
  (tangent_x_axis : r = 4) :
  (x + 5)^2 + (y - 4)^2 = 16 :=
sorry

end circle_eq_tangent_x_axis_l2272_227272


namespace intersection_M_N_l2272_227220

def M : Set ℝ := {y | ∃ x : ℝ, y = x - |x|}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {0} :=
  sorry

end intersection_M_N_l2272_227220


namespace modulo_problem_l2272_227258

theorem modulo_problem :
  (47 ^ 2051 - 25 ^ 2051) % 5 = 3 := by
  sorry

end modulo_problem_l2272_227258


namespace quadratic_solution_m_l2272_227237

theorem quadratic_solution_m (m : ℝ) : (x = 2) → (x^2 - m*x + 8 = 0) → (m = 6) := 
by
  sorry

end quadratic_solution_m_l2272_227237


namespace min_rectangles_to_cover_square_exactly_l2272_227245

theorem min_rectangles_to_cover_square_exactly (a b n : ℕ) : 
  (a = 3) → (b = 4) → (n = 12) → 
  (∀ (x : ℕ), x * a * b = n * n → x = 12) :=
by intros; sorry

end min_rectangles_to_cover_square_exactly_l2272_227245


namespace truncated_cone_volume_l2272_227244

noncomputable def volume_of_truncated_cone (R r h : ℝ) : ℝ :=
  let V_large := (1 / 3) * Real.pi * R^2 * (h + h)  -- Height of larger cone is h + x = h + h
  let V_small := (1 / 3) * Real.pi * r^2 * h       -- Height of smaller cone is h
  V_large - V_small

theorem truncated_cone_volume (R r h : ℝ) (hR : R = 8) (hr : r = 4) (hh : h = 6) :
  volume_of_truncated_cone R r h = 224 * Real.pi :=
by
  sorry

end truncated_cone_volume_l2272_227244


namespace population_increase_l2272_227282

theorem population_increase (P : ℝ) (h₁ : 11000 * (1 + P / 100) * (1 + P / 100) = 13310) : 
  P = 10 :=
sorry

end population_increase_l2272_227282


namespace function_inequality_l2272_227231

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_inequality (f : ℝ → ℝ) (h1 : ∀ x : ℝ, x ≥ 1 → f x ≤ x)
  (h2 : ∀ x : ℝ, x ≥ 1 → f (2 * x) / Real.sqrt 2 ≤ f x) :
  ∀ x ≥ 1, f x < Real.sqrt (2 * x) :=
sorry

end function_inequality_l2272_227231


namespace benjie_is_6_years_old_l2272_227207

-- Definitions based on conditions
def margo_age_in_3_years := 4
def years_until_then := 3
def age_difference := 5

-- Current age of Margo
def margo_current_age := margo_age_in_3_years - years_until_then

-- Current age of Benjie
def benjie_current_age := margo_current_age + age_difference

-- The theorem we need to prove
theorem benjie_is_6_years_old : benjie_current_age = 6 :=
by
  -- Proof
  sorry

end benjie_is_6_years_old_l2272_227207


namespace eugene_total_pencils_l2272_227266

def initial_pencils : ℕ := 51
def additional_pencils : ℕ := 6
def total_pencils : ℕ := initial_pencils + additional_pencils

theorem eugene_total_pencils : total_pencils = 57 := by
  sorry

end eugene_total_pencils_l2272_227266


namespace largest_fraction_sum_l2272_227293

theorem largest_fraction_sum : 
  (max (max (max (max 
  ((1 : ℚ) / 3 + (1 : ℚ) / 4) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 5)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 2)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 9)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 6)) = (5 : ℚ) / 6 
:= 
by
  sorry

end largest_fraction_sum_l2272_227293


namespace ellipse_eccentricity_l2272_227273

theorem ellipse_eccentricity (a b c : ℝ) (h_eq : a * a = 16) (h_b : b * b = 12) (h_c : c * c = a * a - b * b) :
  c / a = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_l2272_227273


namespace betty_boxes_l2272_227210

theorem betty_boxes (total_oranges boxes_capacity : ℕ) (h1 : total_oranges = 24) (h2 : boxes_capacity = 8) : total_oranges / boxes_capacity = 3 :=
by sorry

end betty_boxes_l2272_227210


namespace find_constants_l2272_227205

theorem find_constants (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ -2 → 
  (x^3 - x - 4) / ((x - 1) * (x - 4) * (x + 2)) = 
  A / (x - 1) + B / (x - 4) + C / (x + 2)) →
  A = 4 / 9 ∧ B = 28 / 9 ∧ C = -1 / 3 :=
by
  sorry

end find_constants_l2272_227205


namespace complex_multiplication_value_l2272_227257

theorem complex_multiplication_value (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i :=
by
  sorry

end complex_multiplication_value_l2272_227257


namespace find_f_neg2_l2272_227284

-- Define the function f and the given conditions
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 4

theorem find_f_neg2 (a b : ℝ) (h₁ : f 2 a b = 6) : f (-2) a b = -14 :=
by
  sorry

end find_f_neg2_l2272_227284


namespace vector_dot_product_l2272_227241

open Complex

def a : Complex := (1 : ℝ) + (-(2 : ℝ)) * Complex.I
def b : Complex := (-3 : ℝ) + (4 : ℝ) * Complex.I
def c : Complex := (3 : ℝ) + (2 : ℝ) * Complex.I

-- Note: Using real coordinates to simulate vector operations.
theorem vector_dot_product :
  let a_vec := (1, -2)
  let b_vec := (-3, 4)
  let c_vec := (3, 2)
  let linear_combination := (a_vec.1 + 2 * b_vec.1, a_vec.2 + 2 * b_vec.2)
  (linear_combination.1 * c_vec.1 + linear_combination.2 * c_vec.2) = -3 := 
by
  sorry

end vector_dot_product_l2272_227241


namespace smallest_positive_integer_l2272_227202

theorem smallest_positive_integer (n : ℕ) (h1 : 0 < n) (h2 : ∃ k1 : ℕ, 3 * n = k1^2) (h3 : ∃ k2 : ℕ, 4 * n = k2^3) : 
  n = 54 := 
sorry

end smallest_positive_integer_l2272_227202


namespace adults_wearing_hats_l2272_227256

theorem adults_wearing_hats (total_adults : ℕ) (percent_men : ℝ) (percent_men_hats : ℝ) 
  (percent_women_hats : ℝ) (num_hats : ℕ) 
  (h1 : total_adults = 3600) 
  (h2 : percent_men = 0.40) 
  (h3 : percent_men_hats = 0.15) 
  (h4 : percent_women_hats = 0.25) 
  (h5 : num_hats = 756) : 
  (percent_men * total_adults) * percent_men_hats + (total_adults - (percent_men * total_adults)) * percent_women_hats = num_hats := 
sorry

end adults_wearing_hats_l2272_227256
