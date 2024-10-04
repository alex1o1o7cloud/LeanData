import Mathlib

namespace sale_price_is_207_l155_155855

-- Define a namespace for our problem
namespace BicyclePrice

-- Define the conditions as constants
def priceAtStoreP : ℝ := 200
def regularPriceIncreasePercentage : ℝ := 0.15
def salePriceDecreasePercentage : ℝ := 0.10

-- Define the regular price at Store Q
def regularPriceAtStoreQ : ℝ := priceAtStoreP * (1 + regularPriceIncreasePercentage)

-- Define the sale price at Store Q
def salePriceAtStoreQ : ℝ := regularPriceAtStoreQ * (1 - salePriceDecreasePercentage)

-- The final theorem we need to prove
theorem sale_price_is_207 : salePriceAtStoreQ = 207 := by
  sorry

end BicyclePrice

end sale_price_is_207_l155_155855


namespace year_2049_is_Jisi_l155_155575

-- Define Heavenly Stems
def HeavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]

-- Define Earthly Branches
def EarthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

-- Define the indices of Ding (丁) and You (酉) based on 2017
def Ding_index : Nat := 3
def You_index : Nat := 9

-- Define the year difference
def year_difference : Nat := 2049 - 2017

-- Calculate the indices for the Heavenly Stem and Earthly Branch in 2049
def HeavenlyStem_index_2049 : Nat := (Ding_index + year_difference) % 10
def EarthlyBranch_index_2049 : Nat := (You_index + year_difference) % 12

theorem year_2049_is_Jisi : 
  HeavenlyStems[HeavenlyStem_index_2049]? = some "Ji" ∧ EarthlyBranches[EarthlyBranch_index_2049]? = some "Si" :=
by
  sorry

end year_2049_is_Jisi_l155_155575


namespace correct_calculation_l155_155297

variable (a : ℝ)

theorem correct_calculation (a : ℝ) : (2 * a)^2 / (4 * a) = a := by
  sorry

end correct_calculation_l155_155297


namespace impossible_pawn_placement_l155_155054

theorem impossible_pawn_placement :
  ¬(∃ a b c : ℕ, a + b + c = 50 ∧ 
  ∀ (x y z : ℕ), 2 * a ≤ x ∧ x ≤ 2 * b ∧ 2 * b ≤ y ∧ y ≤ 2 * c ∧ 2 * c ≤ z ∧ z ≤ 2 * a) := sorry

end impossible_pawn_placement_l155_155054


namespace Jenna_total_cost_l155_155662

theorem Jenna_total_cost :
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  total_cost = 468 :=
by
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  show total_cost = 468
  sorry

end Jenna_total_cost_l155_155662


namespace next_special_year_after_2009_l155_155290

def is_special_year (n : ℕ) : Prop :=
  ∃ d1 d2 d3 d4 : ℕ,
    (2000 ≤ n) ∧ (n < 10000) ∧
    (d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n) ∧
    (d1 ≠ 0) ∧
    ∀ (p q r s : ℕ),
    (p * 1000 + q * 100 + r * 10 + s < n) →
    (p ≠ d1 ∨ q ≠ d2 ∨ r ≠ d3 ∨ s ≠ d4)

theorem next_special_year_after_2009 : ∃ y : ℕ, is_special_year y ∧ y > 2009 ∧ y = 2022 :=
  sorry

end next_special_year_after_2009_l155_155290


namespace max_difference_second_largest_second_smallest_l155_155596

theorem max_difference_second_largest_second_smallest :
  ∀ (a b c d e f g h : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧ g < h ∧
  a + b + c = 27 ∧
  a + b + c + d + e + f + g + h = 152 ∧
  f + g + h = 87 →
  g - b = 26 :=
by
  intros;
  sorry

end max_difference_second_largest_second_smallest_l155_155596


namespace inequality_of_f_log2015_l155_155676

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_of_f_log2015 :
  (∀ x : ℝ, deriv f x > f x) →
  f (Real.log 2015) > 2015 * f 0 :=
by sorry

end inequality_of_f_log2015_l155_155676


namespace remainder_when_N_divided_by_1000_l155_155115

def number_of_factors_of_5 (n : Nat) : Nat :=
  if n = 0 then 0 
  else n / 5 + number_of_factors_of_5 (n / 5)

def total_factors_of_5_upto (n : Nat) : Nat := 
  match n with
  | 0 => 0
  | n + 1 => number_of_factors_of_5 (n + 1) + total_factors_of_5_upto n

def product_factorial_5s : Nat := total_factors_of_5_upto 100

def N : Nat := product_factorial_5s

theorem remainder_when_N_divided_by_1000 : N % 1000 = 124 := by
  sorry

end remainder_when_N_divided_by_1000_l155_155115


namespace fraction_of_girls_l155_155797

variable {T G B : ℕ}
variable (ratio : ℚ)

theorem fraction_of_girls (X : ℚ) (h1 : ∀ (G : ℕ) (T : ℕ), X * G = (1/4) * T)
  (h2 : ratio = 5 / 3) (h3 : ∀ (G : ℕ) (B : ℕ), B / G = ratio) :
  X = 2 / 3 :=
by 
  sorry

end fraction_of_girls_l155_155797


namespace radius_squared_of_intersection_circle_l155_155576

def parabola1 (x y : ℝ) := y = (x - 2) ^ 2
def parabola2 (x y : ℝ) := x + 6 = (y - 5) ^ 2

theorem radius_squared_of_intersection_circle
    (x y : ℝ)
    (h₁ : parabola1 x y)
    (h₂ : parabola2 x y) :
    ∃ r, r ^ 2 = 83 / 4 :=
sorry

end radius_squared_of_intersection_circle_l155_155576


namespace percent_of_x_is_z_l155_155026

variable {x y z : ℝ}

theorem percent_of_x_is_z 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z / x = 1.2 := 
sorry

end percent_of_x_is_z_l155_155026


namespace possible_strings_after_moves_l155_155680

theorem possible_strings_after_moves : 
  let initial_string := "HHMMMMTT"
  let moves := [("HM", "MH"), ("MT", "TM"), ("TH", "HT")]
  let binom := Nat.choose 8 4
  binom = 70 := by
  sorry

end possible_strings_after_moves_l155_155680


namespace quadratic_solution_range_l155_155794

theorem quadratic_solution_range (t : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x - t = 0 ∧ -1 < x ∧ x < 4) ↔ (-1 ≤ t ∧ t < 8) := 
sorry

end quadratic_solution_range_l155_155794


namespace selling_price_per_book_l155_155464

noncomputable def fixed_costs : ℝ := 35630
noncomputable def variable_cost_per_book : ℝ := 11.50
noncomputable def num_books : ℕ := 4072
noncomputable def total_production_costs : ℝ := fixed_costs + variable_cost_per_book * num_books

theorem selling_price_per_book :
  (total_production_costs / num_books : ℝ) = 20.25 := by
  sorry

end selling_price_per_book_l155_155464


namespace tan_of_pi_over_two_minus_alpha_l155_155223

theorem tan_of_pi_over_two_minus_alpha (α : ℝ) (h : Real.sin (Real.pi + α) = -1/3) : 
  Real.tan (Real.pi / 2 - α) = 2 * Real.sqrt 2 ∨ Real.tan (Real.pi / 2 - α) = -2 * Real.sqrt 2 := 
sorry

end tan_of_pi_over_two_minus_alpha_l155_155223


namespace three_digit_integers_with_at_least_two_identical_digits_l155_155646

/-- Prove that the number of positive three-digit integers less than 700 that have at least two identical digits is 162. -/
theorem three_digit_integers_with_at_least_two_identical_digits : 
  ∃ n : ℕ, (n = 162) ∧ (count_three_digit_integers_with_at_least_two_identical_digits n) :=
by
  sorry

/-- Define a function to count the number of three-digit integers less than 700 with at least two identical digits -/
noncomputable def count_three_digit_integers_with_at_least_two_identical_digits (n : ℕ) : Prop :=
  n = 162

end three_digit_integers_with_at_least_two_identical_digits_l155_155646


namespace johns_profit_l155_155436

noncomputable def earnings : ℕ := 30000
noncomputable def purchase_price : ℕ := 18000
noncomputable def trade_in_value : ℕ := 6000

noncomputable def depreciation : ℕ := purchase_price - trade_in_value
noncomputable def profit : ℕ := earnings - depreciation

theorem johns_profit : profit = 18000 := by
  sorry

end johns_profit_l155_155436


namespace induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l155_155956

open Nat

theorem induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25 :
  ∀ n : ℕ, n > 0 → 25 ∣ (2^(n+2) * 3^n + 5*n - 4) :=
by
  intro n hn
  sorry

end induction_two_pow_n_plus_two_times_three_pow_n_plus_five_n_minus_four_divisible_by_25_l155_155956


namespace simplify_expression_l155_155136

theorem simplify_expression : (- (1 / 343 : ℝ)) ^ (-2 / 3 : ℝ) = 49 :=
by 
  sorry

end simplify_expression_l155_155136


namespace tshirts_per_package_l155_155432

def number_of_packages := 28
def total_white_tshirts := 56
def white_tshirts_per_package : Nat :=
  total_white_tshirts / number_of_packages

theorem tshirts_per_package :
  white_tshirts_per_package = 2 :=
by
  -- Assuming the definitions and the proven facts
  sorry

end tshirts_per_package_l155_155432


namespace sandwiches_ordered_l155_155803

-- Definitions of the given conditions
def sandwichCost : ℕ := 5
def payment : ℕ := 20
def change : ℕ := 5

-- Statement to prove how many sandwiches Jack ordered
theorem sandwiches_ordered : (payment - change) / sandwichCost = 3 := by
  -- Sorry to skip the proof
  sorry

end sandwiches_ordered_l155_155803


namespace AndrewAge_l155_155874

variable (a f g : ℚ)
axiom h1 : f = 8 * a
axiom h2 : g = 3 * f
axiom h3 : g - a = 72

theorem AndrewAge : a = 72 / 23 :=
by
  sorry

end AndrewAge_l155_155874


namespace scientific_notation_l155_155326

theorem scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) (h₁ : x = 5853) (h₂ : 1 ≤ |a|) (h₃ : |a| < 10) (h₄ : x = a * 10^n) : 
  a = 5.853 ∧ n = 3 :=
by sorry

end scientific_notation_l155_155326


namespace next_term_geometric_sequence_l155_155979

noncomputable def geometric_term (a r : ℕ) (n : ℕ) : ℕ :=
a * r^n

theorem next_term_geometric_sequence (y : ℕ) :
  ∀ a₁ a₂ a₃ a₄, a₁ = 3 → a₂ = 9 * y → a₃ = 27 * y^2 → a₄ = 81 * y^3 →
  geometric_term 3 (3 * y) 4 = 243 * y^4 :=
by
  intros a₁ a₂ a₃ a₄ h₁ h₂ h₃ h₄
  sorry

end next_term_geometric_sequence_l155_155979


namespace total_area_of_paths_l155_155382

theorem total_area_of_paths:
  let bed_width := 4
  let bed_height := 3
  let num_beds_width := 3
  let num_beds_height := 5
  let path_width := 2

  let total_bed_width := num_beds_width * bed_width
  let total_path_width := (num_beds_width + 1) * path_width
  let total_width := total_bed_width + total_path_width

  let total_bed_height := num_beds_height * bed_height
  let total_path_height := (num_beds_height + 1) * path_width
  let total_height := total_bed_height + total_path_height

  let total_area_greenhouse := total_width * total_height
  let total_area_beds := num_beds_width * num_beds_height * bed_width * bed_height

  let total_area_paths := total_area_greenhouse - total_area_beds

  total_area_paths = 360 :=
by sorry

end total_area_of_paths_l155_155382


namespace jerry_probability_l155_155807

noncomputable def biased_coin_process : ℕ := 56669  -- Define function to represent the outcome

theorem jerry_probability : 
  ∃ p q: ℕ, 
  nat.gcd p q = 1 ∧
  p = 5120 ∧ 
  q = 59049 ∧ 
  biased_coin_process = p + q := 
by
  use 5120
  use 59049
  sorry

end jerry_probability_l155_155807


namespace quadratic_inequality_sum_l155_155399

theorem quadratic_inequality_sum (a b : ℝ) (h1 : 1 < 2) 
 (h2 : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) 
 (h3 : 1 + 2 = a)  (h4 : 1 * 2 = b) : 
 a + b = 5 := 
by 
sorry

end quadratic_inequality_sum_l155_155399


namespace abs_neg_two_eq_two_l155_155829

theorem abs_neg_two_eq_two : |(-2 : ℤ)| = 2 := 
by 
  sorry

end abs_neg_two_eq_two_l155_155829


namespace probability_x_gt_9y_in_rectangle_l155_155002

theorem probability_x_gt_9y_in_rectangle :
  let a := 1007
  let b := 1008
  let area_triangle := (a * a / 18 : ℚ)
  let area_rectangle := (a * b : ℚ)
  area_triangle / area_rectangle = (1 : ℚ) / 18 :=
by
  sorry

end probability_x_gt_9y_in_rectangle_l155_155002


namespace train_pass_time_is_38_seconds_l155_155859

noncomputable def speed_of_jogger_kmhr : ℝ := 9
noncomputable def speed_of_train_kmhr : ℝ := 45
noncomputable def lead_distance_m : ℝ := 260
noncomputable def train_length_m : ℝ := 120

noncomputable def speed_of_jogger_ms : ℝ := speed_of_jogger_kmhr * (1000 / 3600)
noncomputable def speed_of_train_ms : ℝ := speed_of_train_kmhr * (1000 / 3600)

noncomputable def relative_speed_ms : ℝ := speed_of_train_ms - speed_of_jogger_ms
noncomputable def total_distance_m : ℝ := lead_distance_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_m / relative_speed_ms

theorem train_pass_time_is_38_seconds :
  time_to_pass_jogger_s = 38 := 
sorry

end train_pass_time_is_38_seconds_l155_155859


namespace find_alpha_l155_155908

theorem find_alpha (P : Real × Real) (h: P = (Real.sin (50 * Real.pi / 180), 1 + Real.cos (50 * Real.pi / 180))) :
  ∃ α : Real, α = 65 * Real.pi / 180 := by
  sorry

end find_alpha_l155_155908


namespace arithmetic_sequence_a8_l155_155374

/-- In an arithmetic sequence with the given sum of terms, prove the value of a_8 is 14. -/
theorem arithmetic_sequence_a8 (a : ℕ → ℕ) (d : ℕ) (h1 : ∀ (n : ℕ), a (n+1) = a n + d)
    (h2 : a 2 + a 7 + a 8 + a 9 + a 14 = 70) : a 8 = 14 :=
  sorry

end arithmetic_sequence_a8_l155_155374


namespace arithmetic_sequence_seventh_term_l155_155622

noncomputable def a3 := (2 : ℚ) / 11
noncomputable def a11 := (5 : ℚ) / 6

noncomputable def a7 := (a3 + a11) / 2

theorem arithmetic_sequence_seventh_term :
  a7 = 67 / 132 := by
  sorry

end arithmetic_sequence_seventh_term_l155_155622


namespace f_cos_x_l155_155922

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (hx : f (Real.sin x) = 2 - Real.cos (2 * x)) :
  f (Real.cos x) = 2 + (Real.cos x)^2 :=
sorry

end f_cos_x_l155_155922


namespace min_value_expression_l155_155764

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2/x + 3/y = 1) :
  x/2 + y/3 = 4 :=
sorry

end min_value_expression_l155_155764


namespace total_pages_in_book_l155_155346

-- Define the given conditions
def chapters : Nat := 41
def days : Nat := 30
def pages_per_day : Nat := 15

-- Define the statement to be proven
theorem total_pages_in_book : (days * pages_per_day) = 450 := by
  sorry

end total_pages_in_book_l155_155346


namespace calculate_expression_l155_155483

theorem calculate_expression : 
  (0.25 ^ 16) * ((-4) ^ 17) = -4 := 
by
  sorry

end calculate_expression_l155_155483


namespace circle_radius_seven_l155_155762

theorem circle_radius_seven (k : ℝ) :
  (∃ x y : ℝ, (x^2 + 12 * x + y^2 + 8 * y - k = 0)) ↔ (k = -3) :=
by
  sorry

end circle_radius_seven_l155_155762


namespace smaller_prime_factor_l155_155405

theorem smaller_prime_factor (a b : ℕ) (prime_a : Nat.Prime a) (prime_b : Nat.Prime b) (distinct : a ≠ b)
  (product : a * b = 316990099009901) :
  min a b = 4002001 :=
  sorry

end smaller_prime_factor_l155_155405


namespace possible_values_of_x3_y3_z3_l155_155114

def matrix_data (x y z : ℝ) := 
  ![
    [x, y, z],
    [y, z, x],
    [z, x, y]
  ]

theorem possible_values_of_x3_y3_z3 (x y z : ℝ) (I : Matrix (Fin 3) (Fin 3) ℝ) :
  let N := matrix_data x y z in
  N ⬝ N = 2 • I ∧ x * y * z = -2 →
  ∃ (k : ℝ), k ∈ { -6 + 2 * Real.sqrt 2, -6 - 2 * Real.sqrt 2 } ∧ k = x^3 + y^3 + z^3 :=
by
  sorry

end possible_values_of_x3_y3_z3_l155_155114


namespace bus_stoppage_time_l155_155491

theorem bus_stoppage_time (speed_excl_stoppages speed_incl_stoppages : ℕ) (h1 : speed_excl_stoppages = 54) (h2 : speed_incl_stoppages = 45) : 
  ∃ (t : ℕ), t = 10 := by
  sorry

end bus_stoppage_time_l155_155491


namespace cricket_target_runs_l155_155540

-- Define the conditions
def first_20_overs_run_rate : ℝ := 4.2
def remaining_30_overs_run_rate : ℝ := 8
def overs_20 : ℤ := 20
def overs_30 : ℤ := 30

-- State the proof problem
theorem cricket_target_runs : 
  (first_20_overs_run_rate * (overs_20 : ℝ)) + (remaining_30_overs_run_rate * (overs_30 : ℝ)) = 324 :=
by
  sorry

end cricket_target_runs_l155_155540


namespace polynomial_has_real_root_l155_155398

open Real Polynomial

variable {c d : ℝ}
variable {P : Polynomial ℝ}

theorem polynomial_has_real_root (hP1 : ∀ n : ℕ, c * |(n : ℝ)|^3 ≤ |P.eval (n : ℝ)|)
                                (hP2 : ∀ n : ℕ, |P.eval (n : ℝ)| ≤ d * |(n : ℝ)|^3)
                                (hc : 0 < c) (hd : 0 < d) : 
                                ∃ x : ℝ, P.eval x = 0 :=
sorry

end polynomial_has_real_root_l155_155398


namespace find_circle_center_l155_155497

def circle_center (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y - 16 = 0

theorem find_circle_center (x y : ℝ) :
  circle_center x y ↔ (x, y) = (3, 4) :=
by
  sorry

end find_circle_center_l155_155497


namespace no_valid_formation_l155_155602

-- Define the conditions related to the formation:
-- s : number of rows
-- t : number of musicians per row
-- Total musicians = s * t = 400
-- t is divisible by 4
-- 10 ≤ t ≤ 50
-- Additionally, the brass section needs to form a triangle in the first three rows
-- while maintaining equal distribution of musicians from each section in every row.

theorem no_valid_formation (s t : ℕ) (h_mul : s * t = 400) 
  (h_div : t % 4 = 0) 
  (h_range : 10 ≤ t ∧ t ≤ 50) 
  (h_triangle : ∀ (r1 r2 r3 : ℕ), r1 < r2 ∧ r2 < r3 → r1 + r2 + r3 = 100 → false) : 
  x = 0 := by
  sorry

end no_valid_formation_l155_155602


namespace standard_deviation_less_than_mean_l155_155401

theorem standard_deviation_less_than_mean 
  (μ : ℝ) (σ : ℝ) (x : ℝ) 
  (h1 : μ = 14.5) 
  (h2 : σ = 1.5) 
  (h3 : x = 11.5) : 
  (μ - x) / σ = 2 :=
by
  rw [h1, h2, h3]
  norm_num

end standard_deviation_less_than_mean_l155_155401


namespace sandy_gain_percent_is_10_l155_155028

def total_cost (purchase_price repair_costs : ℕ) := purchase_price + repair_costs

def gain (selling_price total_cost : ℕ) := selling_price - total_cost

def gain_percent (gain total_cost : ℕ) := (gain / total_cost : ℚ) * 100

theorem sandy_gain_percent_is_10 
  (purchase_price : ℕ := 900)
  (repair_costs : ℕ := 300)
  (selling_price : ℕ := 1320) :
  gain_percent (gain selling_price (total_cost purchase_price repair_costs)) 
               (total_cost purchase_price repair_costs) = 10 := 
by
  simp [total_cost, gain, gain_percent]
  sorry

end sandy_gain_percent_is_10_l155_155028


namespace value_of_4_inch_cube_l155_155450

noncomputable def value_per_cubic_inch (n : ℕ) : ℝ :=
  match n with
  | 1 => 300
  | _ => 1.1 ^ (n - 1) * 300

def cube_volume (n : ℕ) : ℝ :=
  n^3

noncomputable def total_value (n : ℕ) : ℝ :=
  cube_volume n * value_per_cubic_inch n

theorem value_of_4_inch_cube : total_value 4 = 25555 := by
  admit

end value_of_4_inch_cube_l155_155450


namespace fractions_expressible_iff_prime_l155_155905

noncomputable def is_good_fraction (a b n : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = n

theorem fractions_expressible_iff_prime (n : ℕ) (hn : n > 1) :
  (∀ (a b : ℕ), b < n → ∃ (k l : ℤ), k * a + l * n = b) ↔ Prime n :=
sorry

end fractions_expressible_iff_prime_l155_155905


namespace complex_div_eq_half_sub_half_i_l155_155569

theorem complex_div_eq_half_sub_half_i (i : ℂ) (hi : i^2 = -1) : 
  (i^3 / (1 - i)) = (1 / 2) - (1 / 2) * i :=
by
  sorry

end complex_div_eq_half_sub_half_i_l155_155569


namespace circumcircle_diameter_triangle_ABC_l155_155802

theorem circumcircle_diameter_triangle_ABC
  (A : ℝ) (BC : ℝ) (R : ℝ)
  (hA : A = 60) (hBC : BC = 4)
  (hR_formula : 2 * R = BC / Real.sin (A * Real.pi / 180)) :
  2 * R = 8 * Real.sqrt 3 / 3 :=
by
  sorry

end circumcircle_diameter_triangle_ABC_l155_155802


namespace puppies_given_to_friends_l155_155821

def original_puppies : ℕ := 8
def current_puppies : ℕ := 4

theorem puppies_given_to_friends : original_puppies - current_puppies = 4 :=
by
  sorry

end puppies_given_to_friends_l155_155821


namespace jelly_cost_l155_155337

theorem jelly_cost (B J : ℕ) 
  (h1 : 15 * (6 * B + 7 * J) = 315) 
  (h2 : 0 ≤ B) 
  (h3 : 0 ≤ J) : 
  15 * J * 7 = 315 := 
sorry

end jelly_cost_l155_155337


namespace find_number_l155_155216

theorem find_number (x : ℤ) (h : (85 + x) * 1 = 9637) : x = 9552 :=
by
  sorry

end find_number_l155_155216


namespace saeyoung_yen_value_l155_155198

-- Define the exchange rate
def exchange_rate : ℝ := 17.25

-- Define Saeyoung's total yuan
def total_yuan : ℝ := 1000 + 10

-- Define the total yen based on the exchange rate
def total_yen : ℝ := total_yuan * exchange_rate

-- State the theorem
theorem saeyoung_yen_value : total_yen = 17422.5 :=
by
  sorry

end saeyoung_yen_value_l155_155198


namespace correct_sampling_methods_l155_155041

/-- 
Given:
1. A group of 500 senior year students with the following blood type distribution: 200 with blood type O,
125 with blood type A, 125 with blood type B, and 50 with blood type AB.
2. A task to select a sample of 20 students to study the relationship between blood type and color blindness.
3. A high school soccer team consisting of 11 players, and the need to draw 2 players to investigate their study load.
4. Sampling methods: I. Random sampling, II. Systematic sampling, III. Stratified sampling.

Prove:
The correct sampling methods are: Stratified sampling (III) for the blood type-color blindness study and
Random sampling (I) for the soccer team study.
-/ 

theorem correct_sampling_methods (students : Finset ℕ) (blood_type_O blood_type_A blood_type_B blood_type_AB : ℕ)
  (sample_size_students soccer_team_size draw_size_soccer_team : ℕ)
  (sampling_methods : Finset ℕ) : 
  (students.card = 500) →
  (blood_type_O = 200) →
  (blood_type_A = 125) →
  (blood_type_B = 125) →
  (blood_type_AB = 50) →
  (sample_size_students = 20) →
  (soccer_team_size = 11) →
  (draw_size_soccer_team = 2) →
  (sampling_methods = {1, 2, 3}) →
  (s = (3, 1)) :=
by
  sorry

end correct_sampling_methods_l155_155041


namespace expansion_correct_l155_155628

noncomputable def P (x y : ℝ) : ℝ := 2 * x^25 - 5 * x^8 + 2 * x * y^3 - 9

noncomputable def M (x : ℝ) : ℝ := 3 * x^7

theorem expansion_correct (x y : ℝ) :
  (P x y) * (M x) = 6 * x^32 - 15 * x^15 + 6 * x^8 * y^3 - 27 * x^7 :=
by
  sorry

end expansion_correct_l155_155628


namespace find_y_from_equation_l155_155074

theorem find_y_from_equation :
  ∀ y : ℕ, (12 ^ 3 * 6 ^ 4) / y = 5184 → y = 432 :=
by
  sorry

end find_y_from_equation_l155_155074


namespace fish_kept_l155_155113

theorem fish_kept (Leo_caught Agrey_more Sierra_more Leo_fish Returned : ℕ) 
                  (Agrey_caught : Agrey_more = 20) 
                  (Sierra_caught : Sierra_more = 15) 
                  (Leo_caught_cond : Leo_fish = 40) 
                  (Returned_cond : Returned = 30) : 
                  (Leo_fish + (Leo_fish + Agrey_more) + ((Leo_fish + Agrey_more) + Sierra_more) - Returned) = 145 :=
by
  sorry

end fish_kept_l155_155113


namespace total_value_of_button_collection_l155_155659

theorem total_value_of_button_collection:
  (∀ (n : ℕ) (v : ℕ), n = 2 → v = 8 → has_same_value → total_value = 10 * (v / n)) →
  has_same_value :=
  sorry

end total_value_of_button_collection_l155_155659


namespace no_arithmetic_progression_exists_l155_155336

theorem no_arithmetic_progression_exists 
  (a : ℕ) (d : ℕ) (a_n : ℕ → ℕ) 
  (h_seq : ∀ n, a_n n = a + n * d) :
  ¬ ∃ (a_n : ℕ → ℕ), (∀ n, a_n (n+1) > a_n n ∧ 
  ∀ n, (a_n n) * (a_n (n+1)) * (a_n (n+2)) * (a_n (n+3)) * (a_n (n+4)) * 
        (a_n (n+5)) * (a_n (n+6)) * (a_n (n+7)) * (a_n (n+8)) * (a_n (n+9)) % 
        ((a_n n) + (a_n (n+1)) + (a_n (n+2)) + (a_n (n+3)) + (a_n (n+4)) + 
         (a_n (n+5)) + (a_n (n+6)) + (a_n (n+7)) + (a_n (n+8)) + (a_n (n+9)) ) = 0 ) := 
sorry

end no_arithmetic_progression_exists_l155_155336


namespace last_recess_break_duration_l155_155712

-- Definitions based on the conditions
def first_recess_break : ℕ := 15
def second_recess_break : ℕ := 15
def lunch_break : ℕ := 30
def total_outside_class_time : ℕ := 80

-- The theorem we need to prove
theorem last_recess_break_duration :
  total_outside_class_time = first_recess_break + second_recess_break + lunch_break + 20 :=
sorry

end last_recess_break_duration_l155_155712


namespace measure_of_angle_x_l155_155072

theorem measure_of_angle_x :
  ∀ (angle_ABC angle_BDE angle_DBE angle_ABD x : ℝ),
    angle_ABC = 132 ∧
    angle_BDE = 31 ∧
    angle_DBE = 30 ∧
    angle_ABD = 180 - 132 →
    x = 180 - (angle_BDE + angle_DBE) →
    x = 119 :=
by
  intros angle_ABC angle_BDE angle_DBE angle_ABD x h h2
  sorry

end measure_of_angle_x_l155_155072


namespace rachel_steps_l155_155488

theorem rachel_steps (x : ℕ) (h1 : x + 325 = 892) : x = 567 :=
sorry

end rachel_steps_l155_155488


namespace stripe_area_is_480pi_l155_155728

noncomputable def stripeArea (diameter : ℝ) (height : ℝ) (width : ℝ) (revolutions : ℕ) : ℝ :=
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let stripeLength := circumference * revolutions
  let area := width * stripeLength
  area

theorem stripe_area_is_480pi : stripeArea 40 90 4 3 = 480 * Real.pi :=
  by
    show stripeArea 40 90 4 3 = 480 * Real.pi
    sorry

end stripe_area_is_480pi_l155_155728


namespace sum_primes_between_1_and_20_l155_155981

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l155_155981


namespace ellipse_foci_coordinates_l155_155070

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) → ∃ c : ℝ, (c = 4) ∧ (x = c ∨ x = -c) ∧ (y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l155_155070


namespace sum_of_solutions_l155_155634

theorem sum_of_solutions :
  let a := -48
  let b := 110
  let c := 165
  ( ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) → x1 ≠ x2 → (x1 + x2) = 55 / 24 ) :=
by
  let a := -48
  let b := 110
  let c := 165
  sorry

end sum_of_solutions_l155_155634


namespace hamburgers_served_l155_155184

-- Definitions for the conditions
def hamburgers_made : ℕ := 9
def hamburgers_left_over : ℕ := 6

-- The main statement to prove
theorem hamburgers_served : hamburgers_made - hamburgers_left_over = 3 := by
  sorry

end hamburgers_served_l155_155184


namespace wrapping_paper_area_l155_155463

theorem wrapping_paper_area (length width : ℕ) (h1 : width = 6) (h2 : 2 * (length + width) = 28) : length * width = 48 :=
by
  sorry

end wrapping_paper_area_l155_155463


namespace sunil_total_amount_proof_l155_155831

theorem sunil_total_amount_proof
  (CI : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) (P : ℝ) (A : ℝ)
  (h1 : CI = 492)
  (h2 : r = 0.05)
  (h3 : n = 1)
  (h4 : t = 2)
  (h5 : CI = P * ((1 + r / n) ^ (n * t) - 1))
  (h6 : A = P + CI) :
  A = 5292 :=
by
  -- Skip the proof.
  sorry

end sunil_total_amount_proof_l155_155831


namespace compute_a1d1_a2d2_a3d3_l155_155547

theorem compute_a1d1_a2d2_a3d3
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, x^6 + 2 * x^5 + x^4 + x^3 + x^2 + 2 * x + 1 = (x^2 + a1*x + d1) * (x^2 + a2*x + d2) * (x^2 + a3*x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 2 :=
by
  sorry

end compute_a1d1_a2d2_a3d3_l155_155547


namespace mass_of_fat_max_mass_of_carbohydrates_l155_155176

-- Definitions based on conditions
def total_mass : ℤ := 500
def fat_percentage : ℚ := 5 / 100
def protein_to_mineral_ratio : ℤ := 4

-- Lean 4 statement for Part 1: mass of fat
theorem mass_of_fat : (total_mass : ℚ) * fat_percentage = 25 := sorry

-- Definitions to utilize in Part 2
def max_percentage_protein_carbs : ℚ := 85 / 100
def mass_protein (x : ℚ) : ℚ := protein_to_mineral_ratio * x

-- Lean 4 statement for Part 2: maximum mass of carbohydrates
theorem max_mass_of_carbohydrates (x : ℚ) :
  x ≥ 50 → (total_mass - 25 - x - mass_protein x) ≤ 225 := sorry

end mass_of_fat_max_mass_of_carbohydrates_l155_155176


namespace neg_exists_le_zero_iff_forall_gt_zero_l155_155145

variable (m : ℝ)

theorem neg_exists_le_zero_iff_forall_gt_zero :
  (¬ ∃ x : ℤ, (x:ℝ)^2 + 2 * x + m ≤ 0) ↔ ∀ x : ℤ, (x:ℝ)^2 + 2 * x + m > 0 :=
by
  sorry

end neg_exists_le_zero_iff_forall_gt_zero_l155_155145


namespace unique_real_root_eq_l155_155146

theorem unique_real_root_eq (x : ℝ) : (∃! x, x = Real.sin x + 1993) :=
sorry

end unique_real_root_eq_l155_155146


namespace margie_change_l155_155814

theorem margie_change : 
  let cost_per_apple := 0.30
  let cost_per_orange := 0.40
  let number_of_apples := 5
  let number_of_oranges := 4
  let total_money := 10.00
  let total_cost_of_apples := cost_per_apple * number_of_apples
  let total_cost_of_oranges := cost_per_orange * number_of_oranges
  let total_cost_of_fruits := total_cost_of_apples + total_cost_of_oranges
  let change_received := total_money - total_cost_of_fruits
  change_received = 6.90 :=
by
  sorry

end margie_change_l155_155814


namespace equilibrium_mass_l155_155288

variable (l m2 S g : ℝ) (m1 : ℝ)

-- Given conditions
def length_of_rod : ℝ := 0.5 -- length l in meters
def mass_of_rod : ℝ := 2 -- mass m2 in kg
def distance_S : ℝ := 0.1 -- distance S in meters
def gravity : ℝ := 9.8 -- gravitational acceleration in m/s^2

-- Equivalence statement
theorem equilibrium_mass (h1 : l = length_of_rod)
                         (h2 : m2 = mass_of_rod)
                         (h3 : S = distance_S)
                         (h4 : g = gravity) :
  m1 = 10 := sorry

end equilibrium_mass_l155_155288


namespace intersection_eq_l155_155516

def setA : Set ℝ := {x | -1 < x ∧ x < 1}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_eq : setA ∩ setB = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_eq_l155_155516


namespace max_experiments_fibonacci_search_l155_155655

-- Define the conditions and the theorem
def is_unimodal (f : ℕ → ℕ) : Prop :=
  ∃ k, ∀ n m, (n < k ∧ k ≤ m) → f n < f k ∧ f k > f m

def fibonacci_search_experiments (n : ℕ) : ℕ :=
  -- Placeholder function representing the steps of Fibonacci search
  if n <= 1 then n else fibonacci_search_experiments (n - 1) + fibonacci_search_experiments (n - 2)

theorem max_experiments_fibonacci_search (f : ℕ → ℕ) (n : ℕ) (hn : n = 33) (hf : is_unimodal f) : fibonacci_search_experiments n ≤ 7 :=
  sorry

end max_experiments_fibonacci_search_l155_155655


namespace prime_N_k_iff_k_eq_2_l155_155902

-- Define the function to generate the number N_k based on k
def N_k (k : ℕ) : ℕ := (10^(2 * k) - 1) / 99

-- Define the main theorem to prove
theorem prime_N_k_iff_k_eq_2 (k : ℕ) : Nat.Prime (N_k k) ↔ k = 2 :=
by
  sorry

end prime_N_k_iff_k_eq_2_l155_155902


namespace sin_ratio_in_triangle_l155_155102

theorem sin_ratio_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h : (b + c) / (c + a) = 4 / 5 ∧ (c + a) / (a + b) = 5 / 6) :
  (Real.sin A + Real.sin C) / Real.sin B = 2 :=
sorry

end sin_ratio_in_triangle_l155_155102


namespace right_triangle_sides_l155_155063

theorem right_triangle_sides (r R : ℝ) (a b c : ℝ) 
    (r_eq : r = 8)
    (R_eq : R = 41)
    (right_angle : a^2 + b^2 = c^2)
    (inradius : 2*r = a + b - c)
    (circumradius : 2*R = c) :
    (a = 18 ∧ b = 80 ∧ c = 82) ∨ (a = 80 ∧ b = 18 ∧ c = 82) :=
by
  sorry

end right_triangle_sides_l155_155063


namespace sum_of_first_15_terms_l155_155153

theorem sum_of_first_15_terms (S : ℕ → ℕ) (h1 : S 5 = 48) (h2 : S 10 = 60) : S 15 = 72 :=
sorry

end sum_of_first_15_terms_l155_155153


namespace total_cost_of_books_l155_155921

theorem total_cost_of_books
  (C1 : ℝ) (C2 : ℝ)
  (h1 : C1 = 315)
  (h2 : 0.85 * C1 = 1.19 * C2) :
  C1 + C2 = 2565 :=
by 
  sorry

end total_cost_of_books_l155_155921


namespace verify_parabola_D_l155_155911

def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

def parabola_vertex (y : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, y x = vertex_form (-1) h k x

-- Given conditions
def h : ℝ := 2
def k : ℝ := 3

-- Possible expressions
def parabola_A (x : ℝ) : ℝ := -((x + 2)^2) - 3
def parabola_B (x : ℝ) : ℝ := -((x - 2)^2) - 3
def parabola_C (x : ℝ) : ℝ := -((x + 2)^2) + 3
def parabola_D (x : ℝ) : ℝ := -((x - 2)^2) + 3

theorem verify_parabola_D : parabola_vertex parabola_D 2 3 :=
by
  -- Placeholder for the proof
  sorry

end verify_parabola_D_l155_155911


namespace calculate_final_price_l155_155677

noncomputable def final_price (j_init p_init : ℝ) (j_inc p_inc : ℝ) (tax discount : ℝ) (j_quantity p_quantity : ℕ) : ℝ :=
  let j_new := j_init + j_inc
  let p_new := p_init * (1 + p_inc)
  let total_price := (j_new * j_quantity) + (p_new * p_quantity)
  let tax_amount := total_price * tax
  let price_with_tax := total_price + tax_amount
  let final_price := if j_quantity > 1 ∧ p_quantity >= 3 then price_with_tax * (1 - discount) else price_with_tax
  final_price

theorem calculate_final_price :
  final_price 30 100 10 (0.20) (0.07) (0.10) 2 5 = 654.84 :=
by
  sorry

end calculate_final_price_l155_155677


namespace product_zero_probability_l155_155710

def set : Finset ℤ := {-3, -2, -1, 0, 0, 2, 4, 5}

noncomputable def probability_product_zero : ℚ :=
  let total_ways := (Finset.card set).choose 2
  let favorable_ways := 6
  favorable_ways / total_ways
  
theorem product_zero_probability :
  probability_product_zero = 3 / 14 := by
  sorry

end product_zero_probability_l155_155710


namespace quadratic_no_real_roots_l155_155215

theorem quadratic_no_real_roots (m : ℝ) : (∀ x, x^2 - 2 * x + m ≠ 0) ↔ m > 1 := 
by sorry

end quadratic_no_real_roots_l155_155215


namespace solve_for_m_l155_155577

def power_function_monotonic (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2 * m - 3 < 0)

theorem solve_for_m (m : ℝ) (h : power_function_monotonic m) : m = 2 :=
sorry

end solve_for_m_l155_155577


namespace arithmetic_sequence_k_is_10_l155_155536

noncomputable def a_n (n : ℕ) (d : ℝ) : ℝ := (n - 1) * d

theorem arithmetic_sequence_k_is_10 (d : ℝ) (h : d ≠ 0) : 
  (∃ k : ℕ, a_n k d = (a_n 1 d) + (a_n 2 d) + (a_n 3 d) + (a_n 4 d) + (a_n 5 d) + (a_n 6 d) + (a_n 7 d) ∧ k = 10) := 
by
  sorry

end arithmetic_sequence_k_is_10_l155_155536


namespace proof_problem_l155_155095

def M : Set ℝ := { x | x > -1 }

theorem proof_problem : {0} ⊆ M := by
  sorry

end proof_problem_l155_155095


namespace range_of_a_for_f_monotonic_l155_155520

-- Define the function f
def f (x a : ℝ) := real.sqrt(x * (x - a))

-- Define the condition that f(x) is monotonically increasing in the interval (0,1)
def is_monotonically_increasing_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y

-- The main theorem we want to prove
theorem range_of_a_for_f_monotonic (a : ℝ) :
  is_monotonically_increasing_in_interval (λ x, f x a) 0 1 → a ≤ 0 :=
by
  sorry

end range_of_a_for_f_monotonic_l155_155520


namespace quadratic_roots_expression_l155_155789

theorem quadratic_roots_expression {m n : ℝ}
  (h₁ : m^2 + m - 12 = 0)
  (h₂ : n^2 + n - 12 = 0)
  (h₃ : m + n = -1) :
  m^2 + 2 * m + n = 11 :=
by {
  sorry
}

end quadratic_roots_expression_l155_155789


namespace diamond_value_l155_155839

def diamond (a b : ℕ) : ℚ := 1 / (a : ℚ) + 2 / (b : ℚ)

theorem diamond_value : ∀ (a b : ℕ), a + b = 10 ∧ a * b = 24 → diamond a b = 2 / 3 := by
  intros a b h
  sorry

end diamond_value_l155_155839


namespace sum_of_inserted_numbers_in_arithmetic_sequence_l155_155378

theorem sum_of_inserted_numbers_in_arithmetic_sequence :
  ∃ a2 a3 : ℤ, 2015 > a2 ∧ a2 > a3 ∧ a3 > 131 ∧ (2015 - a2) = (a2 - a3) ∧ (a2 - a3) = (a3 - 131) ∧ (a2 + a3) = 2146 := 
by
  sorry

end sum_of_inserted_numbers_in_arithmetic_sequence_l155_155378


namespace min_n_constant_term_l155_155499

theorem min_n_constant_term (x : ℕ) (hx : x > 0) : 
  ∃ n : ℕ, 
  (∀ r : ℕ, (2 * n = 5 * r) → n ≥ 5) ∧ 
  (∃ r : ℕ, (2 * n = 5 * r) ∧ n = 5) := by
  sorry

end min_n_constant_term_l155_155499


namespace factorization_c_minus_d_l155_155066

theorem factorization_c_minus_d : 
  ∃ (c d : ℤ), (∀ (x : ℤ), (4 * x^2 - 17 * x - 15 = (4 * x + c) * (x + d))) ∧ (c - d = 8) :=
by
  sorry

end factorization_c_minus_d_l155_155066


namespace price_of_10_pound_bag_l155_155179

variables (P : ℝ) -- price of the 10-pound bag
def cost (n5 n10 n25 : ℕ) := n5 * 13.85 + n10 * P + n25 * 32.25

theorem price_of_10_pound_bag (h : ∃ (n5 n10 n25 : ℕ), n5 * 5 + n10 * 10 + n25 * 25 ≥ 65
  ∧ n5 * 5 + n10 * 10 + n25 * 25 ≤ 80 
  ∧ cost P n5 n10 n25 = 98.77) : 
  P = 20.42 :=
by
  -- Proof skipped
  sorry

end price_of_10_pound_bag_l155_155179


namespace jason_remaining_pokemon_cards_l155_155660

theorem jason_remaining_pokemon_cards :
  (3 - 2) = 1 :=
by 
  sorry

end jason_remaining_pokemon_cards_l155_155660


namespace percentage_problem_l155_155603

theorem percentage_problem
    (x : ℕ) (h1 : (x:ℝ) / 100 * 20 = 8) :
    x = 40 :=
by
    sorry

end percentage_problem_l155_155603


namespace ceilings_left_correct_l155_155395

def total_ceilings : ℕ := 28
def ceilings_painted_this_week : ℕ := 12
def ceilings_painted_next_week : ℕ := ceilings_painted_this_week / 4
def ceilings_left_to_paint : ℕ := total_ceilings - (ceilings_painted_this_week + ceilings_painted_next_week)

theorem ceilings_left_correct : ceilings_left_to_paint = 13 := by
  sorry

end ceilings_left_correct_l155_155395


namespace verify_inequality_l155_155778

variable {x y : ℝ}

theorem verify_inequality (h : x^2 + x * y + y^2 = (x + y)^2 - x * y ∧ (x + y)^2 - x * y = (x + y - real.sqrt (x * y)) * (x + y + real.sqrt (x * y))) :
  x + y + real.sqrt (x * y) ≤ 3 * (x + y - real.sqrt (x * y)) := by
  sorry

end verify_inequality_l155_155778


namespace quadratic_has_real_root_iff_b_in_intervals_l155_155527

theorem quadratic_has_real_root_iff_b_in_intervals (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ∈ set.Icc (-∞ : ℝ) (-10) ∪ set.Icc 10 (∞ : ℝ)) :=
by by sorry

end quadratic_has_real_root_iff_b_in_intervals_l155_155527


namespace sequence_geometric_series_l155_155773

noncomputable def f (x : ℝ) := sorry -- Define f as per given conditions

-- Define the sequence a_n
def a (n : ℕ) := (1/2) * ((4/3)^n)

-- Define the sum S_n of the first n terms of sequence a_n
def S : ℕ → ℝ
| 0 := a 0
| n + 1 := S n + a (n + 1)

-- Problem conditions as hypotheses
theorem sequence_geometric_series :
  (∀ x y > 0, f (x * y) = f x + f y) ∧
  (∀ n : ℕ, f (a n) = f (S n + 2) - f 4) →
  (∀ n : ℕ, a n = 1/2 * (4/3)^n) :=
by
  intros
  sorry -- Proof goes here


end sequence_geometric_series_l155_155773


namespace proposition_4_correct_l155_155785

section

variables {Point Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Definitions of perpendicular and parallel relationships
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (x y : Line) : Prop := sorry

theorem proposition_4_correct (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end

end proposition_4_correct_l155_155785


namespace find_original_b_l155_155826

variable {a b c : ℝ}
variable (H_inv_prop : a * b = c) (H_a_increase : 1.20 * a * 80 = c)

theorem find_original_b : b = 96 :=
  by
  sorry

end find_original_b_l155_155826


namespace no_matching_formula_l155_155106

def formula_A (x : ℕ) : ℕ := 4 * x - 2
def formula_B (x : ℕ) : ℕ := x^3 - x^2 + 2 * x
def formula_C (x : ℕ) : ℕ := 2 * x^2
def formula_D (x : ℕ) : ℕ := x^2 + 2 * x + 1

theorem no_matching_formula :
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_A x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_B x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_C x) ∧
  (¬ ∀ x, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → y = formula_D x)
  :=
by
  sorry

end no_matching_formula_l155_155106


namespace solve_inequality_l155_155010

theorem solve_inequality (x : ℝ) (h : x < 4) : (x - 2) / (x - 4) ≥ 3 := sorry

end solve_inequality_l155_155010


namespace problem1_problem2_l155_155333

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^2 + 3 * b^2 + 2 * a * b - 4 * a^2 - 4 * b = 3 * b^2 + 2 * a * b - 4 * b :=
by sorry

-- Problem 2
theorem problem2 (a b : ℝ) : 2 * (5 * a - 3 * b) - 3 = 10 * a - 6 * b - 3 :=
by sorry

end problem1_problem2_l155_155333


namespace sum_primes_between_1_and_20_l155_155983

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l155_155983


namespace geom_seq_value_l155_155509

noncomputable def geom_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ (n : ℕ), a (n + 1) = a n * q

theorem geom_seq_value
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom_seq : geom_sequence a q)
  (h_a5 : a 5 = 2)
  (h_a6_a8 : a 6 * a 8 = 8) :
  (a 2018 - a 2016) / (a 2014 - a 2012) = 2 :=
sorry

end geom_seq_value_l155_155509


namespace equal_product_groups_exist_l155_155156

def numbers : List ℕ := [21, 22, 34, 39, 44, 45, 65, 76, 133, 153]

theorem equal_product_groups_exist :
  ∃ (g1 g2 : List ℕ), 
    g1.length = 5 ∧ g2.length = 5 ∧ 
    g1.prod = g2.prod ∧ g1.prod = 349188840 ∧ 
    (g1 ++ g2 = numbers ∨ g1 ++ g2 = numbers.reverse) :=
by
  sorry

end equal_product_groups_exist_l155_155156


namespace oula_deliveries_count_l155_155817

-- Define the conditions for the problem
def num_deliveries_Oula (O : ℕ) (T : ℕ) : Prop :=
  T = (3 / 4 : ℚ) * O ∧ (100 * O - 100 * T = 2400)

-- Define the theorem we want to prove
theorem oula_deliveries_count : ∃ (O : ℕ), ∃ (T : ℕ), num_deliveries_Oula O T ∧ O = 96 :=
sorry

end oula_deliveries_count_l155_155817


namespace no_three_consecutive_geometric_l155_155270

open Nat

def a (n : ℕ) : ℤ := 3^n - 2^n

theorem no_three_consecutive_geometric :
  ∀ (k : ℕ), ¬ (∃ n m : ℕ, m = n + 1 ∧ k = m + 1 ∧ (a n) * (a k) = (a m)^2) :=
by
  sorry

end no_three_consecutive_geometric_l155_155270


namespace concyclic_four_points_of_triangle_l155_155149

open Complex

noncomputable def is_concyclic (a b c d : ℂ) : Prop :=
  let cross_ratio (z ∗ : ℂ) := (z.1 - z.3) * (z.2 - z.4) / (z.1 - z.4) * (z.2 - z.3)
  cross_ratio (a, b, c, d) ∈ ℝ

theorem concyclic_four_points_of_triangle
  (a b c : ℂ)
  (h_b : b = 1)
  (h_c : c = -1)
  (h_ratio : norm (a) = sqrt 3)
  (is_concyclic a b c d
: ∃ d,
is_concyclic ((a + b * 2) / 3) ((c - b) / 3) ((c + a * 2 - b) / 3) ((a - c * 2) / 3)) :=
sorry

end concyclic_four_points_of_triangle_l155_155149


namespace erdos_ginzburg_ziv_2047_l155_155003

open Finset

theorem erdos_ginzburg_ziv_2047 (s : Finset ℕ) (h : s.card = 2047) : 
  ∃ t ⊆ s, t.card = 1024 ∧ (t.sum id) % 1024 = 0 :=
sorry

end erdos_ginzburg_ziv_2047_l155_155003


namespace sequence_value_at_20_l155_155657

open Nat

def arithmetic_sequence (a : ℕ → ℤ) : Prop := 
  a 1 = 1 ∧ ∀ n, a (n + 1) - a n = 4

theorem sequence_value_at_20 (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 20 = 77 :=
sorry

end sequence_value_at_20_l155_155657


namespace power_division_simplify_l155_155616

theorem power_division_simplify :
  ((9^9 / 9^8)^2 * 3^4) / 2^4 = 410 + 1/16 := by
  sorry

end power_division_simplify_l155_155616


namespace three_pow_sub_cube_eq_two_l155_155850

theorem three_pow_sub_cube_eq_two (k : ℕ) (h : 30^k ∣ 929260) : 3^k - k^3 = 2 := 
sorry

end three_pow_sub_cube_eq_two_l155_155850


namespace find_4_digit_number_l155_155493

theorem find_4_digit_number (a b c d : ℕ) 
(h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182)
(h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
(h6 : 0 ≤ c) (h7 : c ≤ 9) (h8 : 1 ≤ d) (h9 : d ≤ 9) : 
1000 * a + 100 * b + 10 * c + d = 1909 :=
sorry

end find_4_digit_number_l155_155493


namespace students_not_in_biology_l155_155442

theorem students_not_in_biology (total_students : ℕ) (percent_enrolled : ℝ) (students_enrolled : ℕ) (students_not_enrolled : ℕ) : 
  total_students = 880 ∧ percent_enrolled = 32.5 ∧ total_students - students_enrolled = students_not_enrolled ∧ students_enrolled = 286 ∧ students_not_enrolled = 594 :=
by
  sorry

end students_not_in_biology_l155_155442


namespace abs_neg_two_l155_155567

def absolute_value (x : Int) : Int :=
  if x >= 0 then x else -x

theorem abs_neg_two : absolute_value (-2) = 2 := 
by 
  sorry

end abs_neg_two_l155_155567


namespace suff_condition_not_necc_condition_l155_155265

variable (x : ℝ)

def A : Prop := 0 < x ∧ x < 5
def B : Prop := |x - 2| < 3

theorem suff_condition : A x → B x := by
  sorry

theorem not_necc_condition : B x → ¬ A x := by
  sorry

end suff_condition_not_necc_condition_l155_155265


namespace union_intersection_l155_155945

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {1, 3, 4}
def C : Set ℝ := {x | x^2 - 3 * x + 2 > 0}
def D : Set ℕ := {0, 3, 4}

theorem union_intersection:
  (A ∪ B).inter (C ∩ Set.Ioo (-∞:ℝ) (∞:ℝ)) = D := by
  sorry

end union_intersection_l155_155945


namespace toms_dad_gave_him_dimes_l155_155583

theorem toms_dad_gave_him_dimes (original_dimes final_dimes dimes_given : ℕ)
  (h1 : original_dimes = 15)
  (h2 : final_dimes = 48)
  (h3 : final_dimes = original_dimes + dimes_given) :
  dimes_given = 33 :=
by
  -- Since the main goal here is just the statement, proof is omitted with sorry
  sorry

end toms_dad_gave_him_dimes_l155_155583


namespace ball_picking_problem_proof_l155_155968

-- Define the conditions
def red_balls : ℕ := 8
def white_balls : ℕ := 7

-- Define the questions
def num_ways_to_pick_one_ball : ℕ :=
  red_balls + white_balls

def num_ways_to_pick_two_different_color_balls : ℕ :=
  red_balls * white_balls

-- Define the correct answers
def correct_answer_to_pick_one_ball : ℕ := 15
def correct_answer_to_pick_two_different_color_balls : ℕ := 56

-- State the theorem to be proved
theorem ball_picking_problem_proof :
  (num_ways_to_pick_one_ball = correct_answer_to_pick_one_ball) ∧
  (num_ways_to_pick_two_different_color_balls = correct_answer_to_pick_two_different_color_balls) :=
by
  sorry

end ball_picking_problem_proof_l155_155968


namespace three_digit_numbers_m_l155_155369

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem three_digit_numbers_m (n : ℕ) :
  100 ≤ n ∧ n ≤ 999 ∧ sum_of_digits n = 12 ∧ 100 ≤ 2 * n ∧ 2 * n ≤ 999 ∧ sum_of_digits (2 * n) = 6 → ∃! (m : ℕ), n = m :=
sorry

end three_digit_numbers_m_l155_155369


namespace g_two_eq_one_l155_155812

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0

theorem g_two_eq_one : g 2 = 1 := by
  sorry

end g_two_eq_one_l155_155812


namespace find_rate_of_interest_l155_155719

noncomputable def interest_rate (P R : ℝ) : Prop :=
  (400 = P * (1 + 4 * R / 100)) ∧ (500 = P * (1 + 6 * R / 100))

theorem find_rate_of_interest (R : ℝ) (P : ℝ) (h : interest_rate P R) :
  R = 25 :=
by
  sorry

end find_rate_of_interest_l155_155719


namespace average_other_marbles_l155_155125

def total_marbles : ℕ := 10 -- Define a hypothetical total number for computation
def clear_marbles : ℕ := total_marbles * 40 / 100
def black_marbles : ℕ := total_marbles * 20 / 100
def other_marbles : ℕ := total_marbles - clear_marbles - black_marbles
def marbles_taken : ℕ := 5

theorem average_other_marbles :
  marbles_taken * other_marbles / total_marbles = 2 := by
  sorry

end average_other_marbles_l155_155125


namespace solution_comparison_l155_155546

variables (a a' b b' : ℝ)

theorem solution_comparison (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-(b / a) < -(b' / a')) ↔ (b' / a' < b / a) :=
by sorry

end solution_comparison_l155_155546


namespace largest_possible_b_l155_155414

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c ≤ b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l155_155414


namespace train_length_l155_155190

theorem train_length (L : ℝ) 
  (h1 : (L / 20) = ((L + 1500) / 70)) : L = 600 := by
  sorry

end train_length_l155_155190


namespace all_numbers_in_S_are_rational_l155_155410

theorem all_numbers_in_S_are_rational
  (S : Set ℝ) (hS_finite : S.Finite)
  (hS_sub : ∀ s ∈ S, ∃ a b ∈ S ∪ {0, 1}, a ≠ s ∧ b ≠ s ∧ s = (a + b) / 2) :
  ∀ s ∈ S, s ∈ ℚ :=
by
  sorry

end all_numbers_in_S_are_rational_l155_155410


namespace range_of_m_min_of_squares_l155_155386

-- 1. Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 4)

-- 2. State the condition that f(x) ≤ -m^2 + 6m holds for all x
def condition (m : ℝ) : Prop := ∀ x : ℝ, f x ≤ -m^2 + 6 * m

-- 3. State the range of m to be proven
theorem range_of_m : ∀ m : ℝ, condition m → 1 ≤ m ∧ m ≤ 5 := 
sorry

-- 4. Auxiliary condition for part 2
def m_0 : ℝ := 5

-- 5. State the condition 3a + 4b + 5c = m_0
def sum_condition (a b c : ℝ) : Prop := 3 * a + 4 * b + 5 * c = m_0

-- 6. State the minimum value problem
theorem min_of_squares (a b c : ℝ) : sum_condition a b c → a^2 + b^2 + c^2 ≥ 1 / 2 := 
sorry

end range_of_m_min_of_squares_l155_155386


namespace speed_of_boat_in_still_water_l155_155852

variables (Vb Vs : ℝ)

-- Conditions
def condition_1 : Prop := Vb + Vs = 11
def condition_2 : Prop := Vb - Vs = 5

theorem speed_of_boat_in_still_water (h1 : condition_1 Vb Vs) (h2 : condition_2 Vb Vs) : Vb = 8 := 
by sorry

end speed_of_boat_in_still_water_l155_155852


namespace abigail_saving_period_l155_155323

-- Define the conditions
def amount_saved_each_month : ℕ := 4000
def total_amount_saved : ℕ := 48000

-- State the theorem
theorem abigail_saving_period : total_amount_saved / amount_saved_each_month = 12 := by
  -- Proof would go here
  sorry

end abigail_saving_period_l155_155323


namespace average_other_color_marbles_l155_155127

def percentage_clear : ℝ := 0.4
def percentage_black : ℝ := 0.2
def total_percentage : ℝ := 1.0
def total_marbles_taken : ℝ := 5.0

theorem average_other_color_marbles :
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black in
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors in
  expected_other_color_marbles = 2 := by
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors
  show expected_other_color_marbles = 2
  sorry

end average_other_color_marbles_l155_155127


namespace no_two_exact_cubes_between_squares_l155_155110

theorem no_two_exact_cubes_between_squares :
  ∀ (n a b : ℤ), ¬ (n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2) :=
by
  intros n a b
  sorry

end no_two_exact_cubes_between_squares_l155_155110


namespace flower_bed_can_fit_l155_155539

noncomputable def flower_bed_fits_in_yard : Prop :=
  let yard_side := 70
  let yard_area := yard_side ^ 2
  let building1 := (20 * 10)
  let building2 := (25 * 15)
  let building3 := (30 * 30)
  let tank_radius := 10 / 2
  let tank_area := Real.pi * tank_radius^2
  let total_occupied_area := building1 + building2 + building3 + 2*tank_area
  let available_area := yard_area - total_occupied_area
  let flower_bed_radius := 10 / 2
  let flower_bed_area := Real.pi * flower_bed_radius^2
  let buffer_area := (yard_side - 2 * flower_bed_radius)^2
  available_area >= flower_bed_area ∧ buffer_area >= flower_bed_area

theorem flower_bed_can_fit : flower_bed_fits_in_yard := 
  sorry

end flower_bed_can_fit_l155_155539


namespace maximum_k_for_transportation_l155_155738

theorem maximum_k_for_transportation (k : ℕ) (h : k ≤ 26) :
  (∀ (weights : list ℕ), (∀ x ∈ weights, x ≤ k) ∧ weights.sum = 1500 →
   ∃ (distribution : list (list ℕ)), (∀ d ∈ distribution, d.sum ≤ 80) ∧
                                     distribution.length ≤ 25 ∧
                                     (∀ x ∈ distribution, ∀ y ∈ x, y ∈ weights)) :=
sorry

end maximum_k_for_transportation_l155_155738


namespace range_of_m_l155_155222

open Set

variable {α : Type}

noncomputable def A (m : ℝ) : Set ℝ := {x | m+1 ≤ x ∧ x ≤ 2*m-1}
noncomputable def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

theorem range_of_m (m : ℝ) (hA : A m ⊆ B) (hA_nonempty : A m ≠ ∅) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end range_of_m_l155_155222


namespace min_area_triangle_ABC_l155_155309

def point (α : Type*) := (α × α)

def area_of_triangle (A B C : point ℤ) : ℚ :=
  (1/2 : ℚ) * abs (36 * (C.snd) - 15 * (C.fst))

theorem min_area_triangle_ABC :
  ∃ (C : point ℤ), area_of_triangle (0, 0) (36, 15) C = 3 / 2 :=
by
  sorry

end min_area_triangle_ABC_l155_155309


namespace min_value_of_quadratic_function_l155_155076

-- Given the quadratic function y = x^2 + 4x - 5
def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 4*x - 5

-- Statement of the proof in Lean 4
theorem min_value_of_quadratic_function :
  ∃ (x_min y_min : ℝ), y_min = quadratic_function x_min ∧
  ∀ x : ℝ, quadratic_function x ≥ y_min ∧ x_min = -2 ∧ y_min = -9 :=
by
  sorry

end min_value_of_quadratic_function_l155_155076


namespace range_of_a_l155_155232

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 2) * x + 6 * a - 1 else a ^ x

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 2 / 3) :=
by
  sorry

end range_of_a_l155_155232


namespace average_salary_of_all_workers_l155_155689

-- Definitions of conditions
def num_technicians : ℕ := 7
def num_total_workers : ℕ := 12
def num_other_workers : ℕ := num_total_workers - num_technicians

def avg_salary_technicians : ℝ := 12000
def avg_salary_others : ℝ := 6000

-- Total salary calculations
def total_salary_technicians : ℝ := num_technicians * avg_salary_technicians
def total_salary_others : ℝ := num_other_workers * avg_salary_others

def total_salary : ℝ := total_salary_technicians + total_salary_others

-- Proof statement: the average salary of all workers is 9500
theorem average_salary_of_all_workers : total_salary / num_total_workers = 9500 :=
by
  sorry

end average_salary_of_all_workers_l155_155689


namespace exists_nonneg_coefs_some_n_l155_155848

-- Let p(x) be a polynomial with real coefficients
variable (p : Polynomial ℝ)

-- Assumption: p(x) > 0 for all x >= 0
axiom positive_poly : ∀ x : ℝ, x ≥ 0 → p.eval x > 0 

theorem exists_nonneg_coefs_some_n :
  ∃ n : ℕ, ∀ k : ℕ, Polynomial.coeff ((1 + Polynomial.X)^n * p) k ≥ 0 :=
sorry

end exists_nonneg_coefs_some_n_l155_155848


namespace find_t_for_area_of_triangle_l155_155709

theorem find_t_for_area_of_triangle :
  ∃ (t : ℝ), 
  (∀ (A B C T U: ℝ × ℝ),
    A = (0, 10) → 
    B = (3, 0) → 
    C = (9, 0) → 
    T = (3/10 * (10 - t), t) →
    U = (9/10 * (10 - t), t) →
    2 * 15 = 3/10 * (10 - t) ^ 2) →
  t = 2.93 :=
by sorry

end find_t_for_area_of_triangle_l155_155709


namespace second_smallest_five_digit_in_pascal_l155_155428

theorem second_smallest_five_digit_in_pascal :
  ∃ (x : ℕ), (x > 10000) ∧ (∀ y : ℕ, (y ≠ 10000) → (y < x) → (y < 10000)) ∧ (x = 10001) :=
sorry

end second_smallest_five_digit_in_pascal_l155_155428


namespace abs_inequality_solution_l155_155886

theorem abs_inequality_solution (x : ℝ) : |2 * x + 1| - 2 * |x - 1| > 0 ↔ x > 1 / 4 :=
sorry

end abs_inequality_solution_l155_155886


namespace perpendicular_vectors_x_l155_155237

theorem perpendicular_vectors_x 
  (x : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (x, -2))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : 
  x = 4 := 
  by 
  sorry

end perpendicular_vectors_x_l155_155237


namespace selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l155_155277

-- Problem conditions
def cost_price : ℕ := 70
def max_price : ℕ := 99
def initial_price : ℕ := 110
def initial_sales : ℕ := 20
def price_drop_rate : ℕ := 1
def sales_increase_rate : ℕ := 2
def sales_increase_per_yuan : ℕ := 2
def profit_target : ℕ := 1200

-- Selling price for given sales volume
def selling_price_for_sales_volume (sales_volume : ℕ) : ℕ :=
  initial_price - (sales_volume - initial_sales) / sales_increase_per_yuan

-- Functional relationship between sales volume (y) and price (x)
def sales_volume_function (x : ℕ) : ℕ :=
  initial_sales + sales_increase_rate * (initial_price - x)

-- Profit for given price and resulting sales volume
def daily_profit (x : ℕ) : ℤ :=
  (x - cost_price) * (sales_volume_function x)

-- Part 1: Selling price for 30 items sold
theorem selling_price_30_items : selling_price_for_sales_volume 30 = 105 :=
by
  sorry

-- Part 2: Functional relationship between sales volume and selling price
theorem sales_volume_functional_relationship (x : ℕ) (hx : 70 ≤ x ∧ x ≤ 99) :
  sales_volume_function x = 240 - 2 * x :=
by
  sorry

-- Part 3: Selling price for a daily profit of 1200 yuan
theorem selling_price_for_1200_profit {x : ℕ} (hx : 70 ≤ x ∧ x ≤ 99) :
  daily_profit x = 1200 → x = 90 :=
by
  sorry

end selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l155_155277


namespace no_integer_solutions_l155_155758

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 + 24 := 
by {
  sorry
}

end no_integer_solutions_l155_155758


namespace barbara_total_cost_l155_155475

-- Definitions based on the given conditions
def steak_cost_per_pound : ℝ := 15.00
def steak_quantity : ℝ := 4.5
def chicken_cost_per_pound : ℝ := 8.00
def chicken_quantity : ℝ := 1.5

def expected_total_cost : ℝ := 42.00

-- The main proposition we need to prove
theorem barbara_total_cost :
  steak_cost_per_pound * steak_quantity + chicken_cost_per_pound * chicken_quantity = expected_total_cost :=
by
  sorry

end barbara_total_cost_l155_155475


namespace union_A_B_compl_inter_A_B_l155_155917

-- Definitions based on the conditions
def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

def B : Set ℝ := {x | 2 * x - 9 ≥ 6 - 3 * x}

-- The first proof statement
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ 2} := by
  sorry

-- The second proof statement
theorem compl_inter_A_B : U \ (A ∩ B) = {x : ℝ | x < 3 ∨ x ≥ 4} := by
  sorry

end union_A_B_compl_inter_A_B_l155_155917


namespace misread_weight_l155_155568

-- Definitions based on given conditions in part (a)
def initial_avg_weight : ℝ := 58.4
def num_boys : ℕ := 20
def correct_weight : ℝ := 61
def correct_avg_weight : ℝ := 58.65

-- The Lean theorem statement that needs to be proved
theorem misread_weight :
  let incorrect_total_weight := initial_avg_weight * num_boys
  let correct_total_weight := correct_avg_weight * num_boys
  let weight_diff := correct_total_weight - incorrect_total_weight
  correct_weight - weight_diff = 56 := sorry

end misread_weight_l155_155568


namespace bulbs_on_perfect_squares_l155_155535

def is_on (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k * k

theorem bulbs_on_perfect_squares (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 100) :
  (∀ i : ℕ, 1 ≤ i → i ≤ 100 → ∃ j : ℕ, i = j * j ↔ is_on i) := sorry

end bulbs_on_perfect_squares_l155_155535


namespace parabola_vertex_expression_l155_155912

theorem parabola_vertex_expression (h k : ℝ) :
  (h = 2 ∧ k = 3) →
  ∃ (a : ℝ), (a ≠ 0) ∧
    (∀ x y : ℝ, y = a * (x - h)^2 + k ↔ y = -(x - 2)^2 + 3) :=
by
  sorry

end parabola_vertex_expression_l155_155912


namespace next_term_geometric_sequence_l155_155975

theorem next_term_geometric_sequence (y : ℝ) : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  a₀ = 3 ∧ 
  a₁ = 9 * y ∧ 
  a₂ = 27 * y^2 ∧ 
  a₃ = 81 * y^3 ∧ 
  a₄ = a₃ * 3 * y 
  → a₄ = 243 * y^4 := by
  sorry

end next_term_geometric_sequence_l155_155975


namespace basketball_campers_l155_155870

theorem basketball_campers (total_campers soccer_campers football_campers : ℕ)
  (h_total : total_campers = 88)
  (h_soccer : soccer_campers = 32)
  (h_football : football_campers = 32) :
  total_campers - soccer_campers - football_campers = 24 :=
by
  sorry

end basketball_campers_l155_155870


namespace find_num_chickens_l155_155555

-- Definitions based on problem conditions
def num_dogs : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2
def total_legs_seen : ℕ := 12

-- Proof problem: Prove the number of chickens Mrs. Hilt saw
theorem find_num_chickens (C : ℕ) (h1 : num_dogs * legs_per_dog + C * legs_per_chicken = total_legs_seen) : C = 2 := 
sorry

end find_num_chickens_l155_155555


namespace janna_sleep_hours_l155_155806

-- Define the sleep hours from Monday to Sunday with the specified conditions
def sleep_hours_monday : ℕ := 7
def sleep_hours_tuesday : ℕ := 7 + 1 / 2
def sleep_hours_wednesday : ℕ := 7
def sleep_hours_thursday : ℕ := 7 + 1 / 2
def sleep_hours_friday : ℕ := 7 + 1
def sleep_hours_saturday : ℕ := 8
def sleep_hours_sunday : ℕ := 8

-- Calculate the total sleep hours in a week
noncomputable def total_sleep_hours : ℕ :=
  sleep_hours_monday +
  sleep_hours_tuesday +
  sleep_hours_wednesday +
  sleep_hours_thursday +
  sleep_hours_friday +
  sleep_hours_saturday +
  sleep_hours_sunday

-- The statement we want to prove
theorem janna_sleep_hours : total_sleep_hours = 53 := by
  sorry

end janna_sleep_hours_l155_155806


namespace online_sale_discount_l155_155158

theorem online_sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (total_paid : ℕ) : 
  purchase_amount = 250 → 
  discount_per_100 = 10 → 
  total_paid = purchase_amount - (purchase_amount / 100) * discount_per_100 → 
  total_paid = 230 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end online_sale_discount_l155_155158


namespace inequality_proof_l155_155673

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2) +
    (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2) +
    (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2)
  ) ≤ 8 := 
by
  sorry

end inequality_proof_l155_155673


namespace remainder_of_concatenated_natural_digits_l155_155302

theorem remainder_of_concatenated_natural_digits : 
  let digits := concat_nat_digits 198 in
  digits % 9 = 6 := 
by 
  sorry

end remainder_of_concatenated_natural_digits_l155_155302


namespace ellipse_focus_distance_l155_155341

theorem ellipse_focus_distance : ∀ (x y : ℝ), 9 * x^2 + y^2 = 900 → 2 * Real.sqrt (10^2 - 30^2) = 40 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end ellipse_focus_distance_l155_155341


namespace ratio_of_discount_l155_155805

theorem ratio_of_discount (price_pair1 price_pair2 : ℕ) (total_paid : ℕ) (discount_percent : ℕ) (h1 : price_pair1 = 40)
    (h2 : price_pair2 = 60) (h3 : total_paid = 60) (h4 : discount_percent = 50) :
    (price_pair1 * discount_percent / 100) / (price_pair1 + (price_pair2 - price_pair1 * discount_percent / 100)) = 1 / 4 :=
by
  sorry

end ratio_of_discount_l155_155805


namespace max_value_of_m_l155_155915

noncomputable def f (x m n : ℝ) : ℝ := x^2 + m*x + n^2
noncomputable def g (x m n : ℝ) : ℝ := x^2 + (m+2)*x + n^2 + m + 1

theorem max_value_of_m (m n t : ℝ) :
  (∀(t : ℝ), f t m n ≥ 0 ∨ g t m n ≥ 0) → m ≤ 1 :=
by
  intro h
  sorry

end max_value_of_m_l155_155915


namespace quadratic_has_real_root_l155_155526

theorem quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ (b ≤ -10 ∨ b ≥ 10) := sorry

end quadratic_has_real_root_l155_155526


namespace combined_6th_grade_percentage_l155_155412

noncomputable def percentage_of_6th_graders 
  (parkPercent : Fin 7 → ℚ) 
  (riversidePercent : Fin 7 → ℚ) 
  (totalParkside : ℕ) 
  (totalRiverside : ℕ) 
  : ℚ := 
    let num6thParkside := parkPercent 6 * totalParkside
    let num6thRiverside := riversidePercent 6 * totalRiverside
    let total6thGraders := num6thParkside + num6thRiverside
    let totalStudents := totalParkside + totalRiverside
    (total6thGraders / totalStudents) * 100

theorem combined_6th_grade_percentage :
  let parkPercent := ![(14.0 : ℚ) / 100, 13 / 100, 16 / 100, 15 / 100, 12 / 100, 15 / 100, 15 / 100]
  let riversidePercent := ![(13.0 : ℚ) / 100, 16 / 100, 13 / 100, 15 / 100, 14 / 100, 15 / 100, 14 / 100]
  percentage_of_6th_graders parkPercent riversidePercent 150 250 = 15 := 
  by
  sorry

end combined_6th_grade_percentage_l155_155412


namespace required_moles_H2SO4_l155_155360

-- Definitions for the problem
def moles_NaCl := 2
def moles_H2SO4_needed := 2
def moles_HCl_produced := 2
def moles_NaHSO4_produced := 2

-- Condition representing stoichiometry of the reaction
axiom reaction_stoichiometry : ∀ (moles_NaCl moles_H2SO4 moles_HCl moles_NaHSO4 : ℕ), 
  moles_NaCl = moles_HCl ∧ moles_HCl = moles_NaHSO4 → moles_NaCl = moles_H2SO4

-- Proof statement we want to establish
theorem required_moles_H2SO4 : 
  ∃ (moles_H2SO4 : ℕ), moles_H2SO4 = 2 ∧ ∀ (moles_NaCl : ℕ), moles_NaCl = 2 → moles_H2SO4_needed = 2 := by
  sorry

end required_moles_H2SO4_l155_155360


namespace geometric_sequence_fifth_term_l155_155457

theorem geometric_sequence_fifth_term 
    (a₁ : ℕ) (a₄ : ℕ) (r : ℕ) (a₅ : ℕ)
    (h₁ : a₁ = 3) (h₂ : a₄ = 240) 
    (h₃ : a₄ = a₁ * r^3) 
    (h₄ : a₅ = a₁ * r^4) : 
    a₅ = 768 :=
by
  sorry

end geometric_sequence_fifth_term_l155_155457


namespace horse_revolutions_l155_155458

theorem horse_revolutions :
  ∀ (r_1 r_2 : ℝ) (n : ℕ),
    r_1 = 30 → r_2 = 10 → n = 25 → (r_1 * n) / r_2 = 75 := by
  sorry

end horse_revolutions_l155_155458


namespace largest_four_digit_number_divisible_by_six_l155_155426

theorem largest_four_digit_number_divisible_by_six : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 2 = 0) ∧ (n % 3 = 0) ∧ 
  (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 2 = 0) ∧ (m % 3 = 0) → m ≤ n) ∧ n = 9960 := 
by { sorry }

end largest_four_digit_number_divisible_by_six_l155_155426


namespace larger_exceeds_smaller_times_l155_155343

theorem larger_exceeds_smaller_times {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_diff : a ≠ b)
  (h_eq : a^3 - b^3 = 3 * (2 * a^2 * b - 3 * a * b^2 + b^3)) : a = 4 * b :=
sorry

end larger_exceeds_smaller_times_l155_155343


namespace product_of_numbers_is_86_l155_155929

-- Definitions of the two conditions
def sum_eq_24 (x y : ℝ) : Prop := x + y = 24
def sum_of_squares_eq_404 (x y : ℝ) : Prop := x^2 + y^2 = 404

-- The theorem to prove the product of the two numbers
theorem product_of_numbers_is_86 (x y : ℝ) (h1 : sum_eq_24 x y) (h2 : sum_of_squares_eq_404 x y) : x * y = 86 :=
  sorry

end product_of_numbers_is_86_l155_155929


namespace max_covered_squares_l155_155454

-- Definitions representing the conditions
def checkerboard_squares : ℕ := 1 -- side length of each square on the checkerboard
def card_side_len : ℕ := 2 -- side length of the card

-- Theorem statement representing the question and answer
theorem max_covered_squares : ∀ n, 
  (∃ board_side squared_len, 
    checkerboard_squares = 1 ∧ card_side_len = 2 ∧
    (board_side = checkerboard_squares ∧ squared_len = card_side_len) ∧
    n ≤ 16) →
  n = 16 :=
  sorry

end max_covered_squares_l155_155454


namespace calculate_plot_size_in_acres_l155_155015

theorem calculate_plot_size_in_acres :
  let bottom_edge_cm : ℝ := 15
  let top_edge_cm : ℝ := 10
  let height_cm : ℝ := 10
  let cm_to_miles : ℝ := 3
  let miles_to_acres : ℝ := 640
  let trapezoid_area_cm2 := (bottom_edge_cm + top_edge_cm) * height_cm / 2
  let trapezoid_area_miles2 := trapezoid_area_cm2 * (cm_to_miles ^ 2)
  (trapezoid_area_miles2 * miles_to_acres) = 720000 :=
by
  sorry

end calculate_plot_size_in_acres_l155_155015


namespace fred_initial_money_l155_155763

def initial_money (book_count : ℕ) (average_cost : ℕ) (money_left : ℕ) : ℕ :=
  book_count * average_cost + money_left

theorem fred_initial_money :
  initial_money 6 37 14 = 236 :=
by
  sorry

end fred_initial_money_l155_155763


namespace find_initial_amount_l155_155196

theorem find_initial_amount
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1050)
  (hR : R = 8)
  (hT : T = 5) :
  P = 750 :=
by
  have hSI : P * R * T / 100 = 1050 - P := sorry
  have hFormulaSimplified : P * 0.4 = 1050 - P := sorry
  have hFinal : P * 1.4 = 1050 := sorry
  exact sorry

end find_initial_amount_l155_155196


namespace janet_dresses_total_pockets_l155_155938

theorem janet_dresses_total_pockets :
  let dresses := 24
  let with_pockets := dresses / 2
  let with_two_pockets := with_pockets / 3
  let with_three_pockets := with_pockets - with_two_pockets
  let total_two_pockets := with_two_pockets * 2
  let total_three_pockets := with_three_pockets * 3
  total_two_pockets + total_three_pockets = 32 := by
  sorry

end janet_dresses_total_pockets_l155_155938


namespace sum_of_three_numbers_l155_155023

theorem sum_of_three_numbers (x y z : ℝ) (h₁ : x + y = 29) (h₂ : y + z = 46) (h₃ : z + x = 53) : x + y + z = 64 :=
by
  sorry

end sum_of_three_numbers_l155_155023


namespace vector_transitivity_l155_155301

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem vector_transitivity (h1 : a = b) (h2 : b = c) : a = c :=
by {
  sorry
}

end vector_transitivity_l155_155301


namespace tan_two_x_is_odd_l155_155175

noncomputable def tan_two_x (x : ℝ) : ℝ := Real.tan (2 * x)

theorem tan_two_x_is_odd :
  ∀ x : ℝ,
  (∀ k : ℤ, x ≠ (k * Real.pi / 2) + (Real.pi / 4)) →
  tan_two_x (-x) = -tan_two_x x :=
by
  sorry

end tan_two_x_is_odd_l155_155175


namespace min_S_n_at_24_l155_155077

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n : ℤ) * (2 * n - 48)

theorem min_S_n_at_24 : (∀ n : ℕ, n > 0 → S_n n ≥ S_n 24) ∧ S_n 24 < S_n 25 :=
by 
  sorry

end min_S_n_at_24_l155_155077


namespace compare_abc_l155_155117

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 2
noncomputable def c : ℝ := 9 ^ (1 / 2 : ℝ)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end compare_abc_l155_155117


namespace parabola_translation_left_by_two_units_l155_155407

/-- 
The parabola y = x^2 + 4x + 5 is obtained by translating the parabola y = x^2 + 1. 
Prove that this translation is 2 units to the left.
-/
theorem parabola_translation_left_by_two_units :
  ∀ x : ℝ, (x^2 + 4*x + 5) = ((x+2)^2 + 1) :=
by
  intro x
  sorry

end parabola_translation_left_by_two_units_l155_155407


namespace new_percentage_water_is_correct_l155_155863

def initial_volume : ℕ := 120
def initial_percentage_water : ℚ := 20 / 100
def added_water : ℕ := 8

def initial_volume_water : ℚ := initial_percentage_water * initial_volume
def initial_volume_wine : ℚ := initial_volume - initial_volume_water
def new_volume_water : ℚ := initial_volume_water + added_water
def new_total_volume : ℚ := initial_volume + added_water

def calculate_new_percentage_water : ℚ :=
  (new_volume_water / new_total_volume) * 100

theorem new_percentage_water_is_correct :
  calculate_new_percentage_water = 25 := 
by
  sorry

end new_percentage_water_is_correct_l155_155863


namespace isosceles_obtuse_triangle_smallest_angle_l155_155192

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β : ℝ), 0 < α ∧ α = 1.5 * 90 ∧ α + 2 * β = 180 ∧ β = 22.5 := by
  sorry

end isosceles_obtuse_triangle_smallest_angle_l155_155192


namespace molecular_weight_of_moles_l155_155022

-- Approximate atomic weights
def atomic_weight_N := 14.01
def atomic_weight_O := 16.00

-- Molecular weight of N2O3
def molecular_weight_N2O3 := (2 * atomic_weight_N) + (3 * atomic_weight_O)

-- Given the total molecular weight of some moles of N2O3
def total_molecular_weight : ℝ := 228

-- We aim to prove that the total molecular weight of some moles of N2O3 equals 228 g
theorem molecular_weight_of_moles (h: molecular_weight_N2O3 ≠ 0) :
  total_molecular_weight = 228 := by
  sorry

end molecular_weight_of_moles_l155_155022


namespace distance_between_truck_and_car_l155_155468

noncomputable def truck_speed : ℝ := 65
noncomputable def car_speed : ℝ := 85
noncomputable def time_duration_hours : ℝ := 3 / 60

theorem distance_between_truck_and_car :
  let relative_speed := car_speed - truck_speed in
  let distance := relative_speed * time_duration_hours in
  distance = 1 :=
by
  let relative_speed := car_speed - truck_speed
  let distance := relative_speed * time_duration_hours
  have h1 : distance = 1 := sorry
  exact h1

end distance_between_truck_and_car_l155_155468


namespace largest_digit_A_l155_155933

theorem largest_digit_A (A : ℕ) (h1 : (31 + A) % 3 = 0) (h2 : 96 % 4 = 0) : 
  A ≤ 7 ∧ (∀ a, a > 7 → ¬((31 + a) % 3 = 0 ∧ 96 % 4 = 0)) :=
by
  sorry

end largest_digit_A_l155_155933


namespace arithmetic_sequence_n_value_l155_155116

theorem arithmetic_sequence_n_value
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (n : ℕ)
  (hS9 : S 9 = 18)
  (ha_n_minus_4 : a (n-4) = 30)
  (hSn : S n = 336)
  (harithmetic_sequence : ∀ k, a (k + 1) - a k = a 2 - a 1) :
  n = 21 :=
sorry

end arithmetic_sequence_n_value_l155_155116


namespace find_f_l155_155766

noncomputable def f (x : ℕ) : ℚ := (1/4) * x * (x + 1) * (2 * x + 1)

lemma f_initial_condition : f 1 = 3 / 2 := by
  sorry

lemma f_functional_equation (x y : ℕ) :
  f (x + y) = (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2 := by
  sorry

theorem find_f (x : ℕ) : f x = (1 / 4) * x * (x + 1) * (2 * x + 1) := by
  sorry

end find_f_l155_155766


namespace students_and_confucius_same_arrival_time_l155_155031

noncomputable def speed_of_students_walking (x : ℝ) : ℝ := x

noncomputable def speed_of_bullock_cart (x : ℝ) : ℝ := 1.5 * x

noncomputable def time_for_students_to_school (x : ℝ) : ℝ := 30 / x

noncomputable def time_for_confucius_to_school (x : ℝ) : ℝ := 30 / (1.5 * x) + 1

theorem students_and_confucius_same_arrival_time (x : ℝ) (h1 : 0 < x) :
  30 / x = 30 / (1.5 * x) + 1 :=
by
  sorry

end students_and_confucius_same_arrival_time_l155_155031


namespace correct_statements_are_two_l155_155403

def statement1 : Prop := 
  ∀ (data : Type) (eq : data → data → Prop), 
    (∃ (t : data), eq t t) → 
    (∀ (d1 d2 : data), eq d1 d2 → d1 = d2)

def statement2 : Prop := 
  ∀ (samplevals : Type) (regress_eqn : samplevals → samplevals → Prop), 
    (∃ (s : samplevals), regress_eqn s s) → 
    (∀ (sv1 sv2 : samplevals), regress_eqn sv1 sv2 → sv1 = sv2)

def statement3 : Prop := 
  ∀ (predvals : Type) (pred_eqn : predvals → predvals → Prop), 
    (∃ (p : predvals), pred_eqn p p) → 
    (∀ (pp1 pp2 : predvals), pred_eqn pp1 pp2 → pp1 = pp2)

def statement4 : Prop := 
  ∀ (observedvals : Type) (linear_eqn : observedvals → observedvals → Prop), 
    (∃ (o : observedvals), linear_eqn o o) → 
    (∀ (ov1 ov2 : observedvals), linear_eqn ov1 ov2 → ov1 = ov2)

def correct_statements_count : ℕ := 2

theorem correct_statements_are_two : 
  (statement1 ∧ statement2 ∧ ¬ statement3 ∧ ¬ statement4) → 
  correct_statements_count = 2 := by
  sorry

end correct_statements_are_two_l155_155403


namespace equation_b_not_symmetric_about_x_axis_l155_155298

def equationA (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equationB (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equationC (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1
def equationD (x y : ℝ) : Prop := x + y^2 = -1

def symmetric_about_x_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, f x y ↔ f x (-y)

theorem equation_b_not_symmetric_about_x_axis : 
  ¬ symmetric_about_x_axis (equationB) :=
sorry

end equation_b_not_symmetric_about_x_axis_l155_155298


namespace total_birds_and_storks_l155_155033

theorem total_birds_and_storks
  (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ)
  (hb : initial_birds = 3) (hs : initial_storks = 4) (has : additional_storks = 6) :
  initial_birds + (initial_storks + additional_storks) = 13 :=
by
  sorry

end total_birds_and_storks_l155_155033


namespace verify_inequality_l155_155777

variable {x y : ℝ}

theorem verify_inequality (h : x^2 + x * y + y^2 = (x + y)^2 - x * y ∧ (x + y)^2 - x * y = (x + y - real.sqrt (x * y)) * (x + y + real.sqrt (x * y))) :
  x + y + real.sqrt (x * y) ≤ 3 * (x + y - real.sqrt (x * y)) := by
  sorry

end verify_inequality_l155_155777


namespace olympic_medals_l155_155373

theorem olympic_medals :
  ∃ (a b c : ℕ),
    (a + b + c = 100) ∧
    (3 * a - 153 = 0) ∧
    (c - b = 7) ∧
    (a = 51) ∧
    (a - 13 = 38) ∧
    (c = 28) :=
by
  sorry

end olympic_medals_l155_155373


namespace production_in_three_minutes_l155_155004

noncomputable def production_rate_per_machine (total_bottles : ℕ) (num_machines : ℕ) : ℕ :=
  total_bottles / num_machines

noncomputable def production_per_minute (machines : ℕ) (rate_per_machine : ℕ) : ℕ :=
  machines * rate_per_machine

noncomputable def total_production (production_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  production_per_minute * minutes

theorem production_in_three_minutes :
  ∀ (total_bottles : ℕ) (num_machines : ℕ) (machines : ℕ) (minutes : ℕ),
  total_bottles = 16 → num_machines = 4 → machines = 8 → minutes = 3 →
  total_production (production_per_minute machines (production_rate_per_machine total_bottles num_machines)) minutes = 96 :=
by
  intros total_bottles num_machines machines minutes h_total_bottles h_num_machines h_machines h_minutes
  sorry

end production_in_three_minutes_l155_155004


namespace no_positive_integer_solution_l155_155820

theorem no_positive_integer_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ¬ (x^2 * y^4 - x^4 * y^2 + 4 * x^2 * y^2 * z^2 + x^2 * z^4 - y^2 * z^4 = 0) :=
sorry

end no_positive_integer_solution_l155_155820


namespace product_of_roots_l155_155523

theorem product_of_roots : ∀ x : ℝ, (x + 3) * (x - 4) = 17 → (∃ a b : ℝ, (x = a ∨ x = b) ∧ a * b = -29) :=
by
  sorry

end product_of_roots_l155_155523


namespace ratio_of_w_y_l155_155963

variable (w x y z : ℚ)

theorem ratio_of_w_y (h1 : w / x = 4 / 3)
                     (h2 : y / z = 3 / 2)
                     (h3 : z / x = 1 / 3) :
                     w / y = 8 / 3 := by
  sorry

end ratio_of_w_y_l155_155963


namespace water_flow_total_l155_155692

theorem water_flow_total
  (R1 R2 R3 : ℕ)
  (h1 : R2 = 36)
  (h2 : R2 = (3 / 2) * R1)
  (h3 : R3 = (5 / 4) * R2)
  : R1 + R2 + R3 = 105 :=
sorry

end water_flow_total_l155_155692


namespace shift_line_down_4_units_l155_155417

theorem shift_line_down_4_units :
  ∀ (x : ℝ), y = - (3 / 4) * x → (y - 4 = - (3 / 4) * x - 4) := by
  sorry

end shift_line_down_4_units_l155_155417


namespace tank_insulation_cost_l155_155321

theorem tank_insulation_cost (l w h : ℝ) (cost_per_sqft : ℝ) (SA : ℝ) (C : ℝ) 
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost_per_sqft : cost_per_sqft = 20) 
  (h_SA : SA = 2 * l * w + 2 * l * h + 2 * w * h)
  (h_C : C = SA * cost_per_sqft) :
  C = 1440 := 
by
  -- proof will be filled in here
  sorry

end tank_insulation_cost_l155_155321


namespace gym_hours_per_week_l155_155940

-- Definitions for conditions
def timesAtGymEachWeek : ℕ := 3
def weightliftingTimeEachDay : ℕ := 1
def warmupCardioFraction : ℚ := 1 / 3

-- The theorem to prove
theorem gym_hours_per_week : (timesAtGymEachWeek * (weightliftingTimeEachDay + weightliftingTimeEachDay * warmupCardioFraction) = 4) := 
by
  sorry

end gym_hours_per_week_l155_155940


namespace solve_congruence_l155_155960

theorem solve_congruence : ∃ (a m : ℕ), 10 * x + 3 ≡ 7 [MOD 18] ∧ x ≡ a [MOD m] ∧ a < m ∧ m ≥ 2 ∧ a + m = 13 := 
sorry

end solve_congruence_l155_155960


namespace circle_radius_l155_155376

theorem circle_radius {C : ℝ → ℝ → Prop} (h1 : C 4 0) (h2 : C (-4) 0) : ∃ r : ℝ, r = 4 :=
by
  -- sorry for brevity
  sorry

end circle_radius_l155_155376


namespace remaining_rice_l155_155706

theorem remaining_rice {q_0 : ℕ} {c : ℕ} {d : ℕ} 
    (h_q0 : q_0 = 52) 
    (h_c : c = 9) 
    (h_d : d = 3) : 
    q_0 - (c * d) = 25 := 
  by 
    -- Proof to be written here
    sorry

end remaining_rice_l155_155706


namespace express_y_in_terms_of_x_l155_155065

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 1) : y = 1 - 5 * x :=
by
  sorry

end express_y_in_terms_of_x_l155_155065


namespace paint_cans_for_25_rooms_l155_155952

theorem paint_cans_for_25_rooms (cans rooms : ℕ) (H1 : cans * 30 = rooms) (H2 : cans * 25 = rooms - 5 * cans) :
  cans = 15 :=
by
  sorry

end paint_cans_for_25_rooms_l155_155952


namespace factorial_equation_solution_l155_155069

theorem factorial_equation_solution (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  a.factorial * b.factorial = a.factorial + b.factorial + c.factorial → a = 3 ∧ b = 3 ∧ c = 4 := by
  sorry

end factorial_equation_solution_l155_155069


namespace euler_line_of_isosceles_triangle_l155_155143

theorem euler_line_of_isosceles_triangle (A B : ℝ × ℝ) (hA : A = (2,0)) (hB : B = (0,4)) (C : ℝ × ℝ) (hC1 : dist A C = dist B C) :
  ∃ a b c : ℝ, a * (C.1 - 2) + b * (C.2 - 0) + c = 0 ∧ x - 2 * y + 3 = 0 :=
by
  sorry

end euler_line_of_isosceles_triangle_l155_155143


namespace smallest_b_no_inverse_mod75_and_mod90_l155_155429

theorem smallest_b_no_inverse_mod75_and_mod90 :
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, n > 0 → n < b →  ¬ (n.gcd 75 > 1 ∧ n.gcd 90 > 1)) ∧ 
  (b.gcd 75 > 1 ∧ b.gcd 90 > 1) ∧ 
  b = 15 := 
by
  sorry

end smallest_b_no_inverse_mod75_and_mod90_l155_155429


namespace fraction_female_attendees_on_time_l155_155197

theorem fraction_female_attendees_on_time (A : ℝ) (h1 : A > 0) :
  let males_fraction := 3/5
  let males_on_time := 7/8
  let not_on_time := 0.155
  let total_on_time_fraction := 1 - not_on_time
  let males := males_fraction * A
  let males_arrived_on_time := males_on_time * males
  let females := (1 - males_fraction) * A
  let females_arrived_on_time_fraction := (total_on_time_fraction * A - males_arrived_on_time) / females
  females_arrived_on_time_fraction = 4/5 :=
by
  sorry

end fraction_female_attendees_on_time_l155_155197


namespace perpendicular_lines_l155_155359

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, x * a + 3 * y - 1 = 0) ∧ (∃ x y : ℝ, 2 * x + (a - 1) * y + 1 = 0) ∧
  (∀ m1 m2 : ℝ, m1 = - a / 3 → m2 = - 2 / (a - 1) → m1 * m2 = -1) →
  a = 3 / 5 :=
sorry

end perpendicular_lines_l155_155359


namespace total_amount_invested_l155_155731

theorem total_amount_invested (x y : ℝ) (hx : 0.06 * x = 0.05 * y + 160) (hy : 0.05 * y = 6000) :
  x + y = 222666.67 :=
by
  sorry

end total_amount_invested_l155_155731


namespace division_of_monomials_l155_155331

variable (x : ℝ) -- ensure x is defined as a variable, here assuming x is a real number

theorem division_of_monomials (x : ℝ) : (2 * x^3 / x^2) = 2 * x := 
by 
  sorry

end division_of_monomials_l155_155331


namespace next_term_geometric_sequence_l155_155977

theorem next_term_geometric_sequence (y : ℝ) (h0 : y ≠ 0) :
  let r := 3 * y in
  let term := 81 * y^3 in
  term * r = 243 * y^4 :=
by
  let r := 3 * y
  let term := 81 * y^3
  have h : term * r = 243 * y^4 := sorry
  exact h

end next_term_geometric_sequence_l155_155977


namespace find_x_in_triangle_XYZ_l155_155256

theorem find_x_in_triangle_XYZ (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ) (hx : y = 7) (hz : z = 6) (hcos : cos_Y_minus_Z = 47 / 64) : 
    ∃ x : ℝ, x = Real.sqrt 63.75 :=
by
  -- The proof will go here, but it is skipped for now.
  sorry

end find_x_in_triangle_XYZ_l155_155256


namespace exists_same_color_points_distance_one_l155_155868

theorem exists_same_color_points_distance_one
    (color : ℝ × ℝ → Fin 3)
    (h : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_distance_one_l155_155868


namespace fraction_always_irreducible_l155_155560

-- Define the problem statement
theorem fraction_always_irreducible (n : ℤ) : gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_always_irreducible_l155_155560


namespace equivalent_prop_l155_155139

theorem equivalent_prop (x : ℝ) : (x > 1 → (x - 1) * (x + 3) > 0) ↔ ((x - 1) * (x + 3) ≤ 0 → x ≤ 1) :=
sorry

end equivalent_prop_l155_155139


namespace shortest_distance_l155_155151

-- Define the line and the circle
def is_on_line (P : ℝ × ℝ) : Prop := P.snd = P.fst - 1

def is_on_circle (Q : ℝ × ℝ) : Prop := Q.fst^2 + Q.snd^2 + 4 * Q.fst - 2 * Q.snd + 4 = 0

-- Define the square of the Euclidean distance between two points
def dist_squared (P Q : ℝ × ℝ) : ℝ := (P.fst - Q.fst)^2 + (P.snd - Q.snd)^2

-- State the theorem regarding the shortest distance between the points on the line and the circle
theorem shortest_distance : ∃ P Q : ℝ × ℝ, is_on_line P ∧ is_on_circle Q ∧ dist_squared P Q = 1 := sorry

end shortest_distance_l155_155151


namespace adam_earnings_l155_155191

def lawns_to_mow : ℕ := 12
def lawns_forgotten : ℕ := 8
def earnings_per_lawn : ℕ := 9

theorem adam_earnings : (lawns_to_mow - lawns_forgotten) * earnings_per_lawn = 36 := by
  sorry

end adam_earnings_l155_155191


namespace probability_of_Y_l155_155971

variable (P_X : ℝ) (P_X_and_Y : ℝ) (P_Y : ℝ)

theorem probability_of_Y (h1 : P_X = 1 / 7)
                         (h2 : P_X_and_Y = 0.031746031746031744) :
  P_Y = 0.2222222222222222 :=
sorry

end probability_of_Y_l155_155971


namespace marcel_corn_l155_155948

theorem marcel_corn (C : ℕ) (H1 : ∃ D, D = C / 2) (H2 : 27 = C + C / 2 + 8 + 4) : C = 10 :=
sorry

end marcel_corn_l155_155948


namespace smallest_whole_number_larger_than_perimeter_l155_155295

theorem smallest_whole_number_larger_than_perimeter (s : ℝ) (h1 : 7 + 23 > s) (h2 : 7 + s > 23) (h3 : 23 + s > 7) : 
  60 = Int.ceil (7 + 23 + s - 1) :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l155_155295


namespace total_guests_l155_155707

theorem total_guests (G : ℕ) 
  (hwomen: ∃ n, n = G / 2)
  (hmen: 15 = 15)
  (hchildren: ∃ n, n = G - (G / 2 + 15))
  (men_leaving: ∃ n, n = 1/5 * 15)
  (children_leaving: 4 = 4)
  (people_stayed: 43 = G - ((1/5 * 15) + 4))
  : G = 50 := by
  sorry

end total_guests_l155_155707


namespace train_actual_speed_l155_155044
-- Import necessary libraries

-- Define the given conditions and question
def departs_time := 6
def planned_speed := 100
def scheduled_arrival_time := 18
def actual_arrival_time := 16
def distance (t₁ t₂ : ℕ) (s : ℕ) : ℕ := s * (t₂ - t₁)
def actual_speed (d t₁ t₂ : ℕ) : ℕ := d / (t₂ - t₁)

-- Proof problem statement
theorem train_actual_speed:
  actual_speed (distance departs_time scheduled_arrival_time planned_speed) departs_time actual_arrival_time = 120 := by sorry

end train_actual_speed_l155_155044


namespace equilibrium_problems_l155_155205

-- Definition of equilibrium constant and catalyst relations

def q1 := False -- Any concentration of substances in equilibrium constant
def q2 := False -- Catalysts changing equilibrium constant
def q3 := False -- No shift if equilibrium constant doesn't change
def q4 := False -- ΔH > 0 if K decreases with increasing temperature
def q5 := True  -- Stoichiometric differences affecting equilibrium constants
def q6 := True  -- Equilibrium shift not necessarily changing equilibrium constant
def q7 := True  -- Extent of reaction indicated by both equilibrium constant and conversion rate

-- The theorem includes our problem statements

theorem equilibrium_problems :
  q1 = False ∧ q2 = False ∧ q3 = False ∧
  q4 = False ∧ q5 = True ∧ q6 = True ∧ q7 = True := by
  sorry

end equilibrium_problems_l155_155205


namespace chessboard_tiling_l155_155484

theorem chessboard_tiling (chessboard : Fin 8 × Fin 8 → Prop) (colors : Fin 8 × Fin 8 → Bool)
  (removed_squares : (Fin 8 × Fin 8) × (Fin 8 × Fin 8))
  (h_diff_colors : colors removed_squares.1 ≠ colors removed_squares.2) :
  ∃ f : (Fin 8 × Fin 8) → (Fin 8 × Fin 8), ∀ x, chessboard x → chessboard (f x) :=
by
  sorry

end chessboard_tiling_l155_155484


namespace Jakes_brother_has_more_l155_155804

-- Define the number of comic books Jake has
def Jake_comics : ℕ := 36

-- Define the total number of comic books Jake and his brother have together
def total_comics : ℕ := 87

-- Prove Jake's brother has 15 more comic books than Jake
theorem Jakes_brother_has_more : ∃ B, B > Jake_comics ∧ B + Jake_comics = total_comics ∧ B - Jake_comics = 15 :=
by
  sorry

end Jakes_brother_has_more_l155_155804


namespace combined_average_score_girls_l155_155050

open BigOperators

variable (A a B b C c : ℕ) -- number of boys and girls at each school
variable (x : ℕ) -- common value for number of boys and girls

axiom Adams_HS : 74 * (A : ℤ) + 81 * (a : ℤ) = 77 * (A + a)
axiom Baker_HS : 83 * (B : ℤ) + 92 * (b : ℤ) = 86 * (B + b)
axiom Carter_HS : 78 * (C : ℤ) + 85 * (c : ℤ) = 80 * (C + c)

theorem combined_average_score_girls :
  (A = a ∧ B = b ∧ C = c) →
  (A = B ∧ B = C) →
  (81 * (A : ℤ) + 92 * (B : ℤ) + 85 * (C : ℤ)) / (A + B + C) = 86 := 
by
  intro h1 h2
  sorry

end combined_average_score_girls_l155_155050


namespace sequence_contains_prime_l155_155897

-- Define the conditions for being square-free and relatively prime
def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m^2 ∣ n → m = 1

def are_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Statement of the problem
theorem sequence_contains_prime :
  ∀ (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ 14 → 2 ≤ a i ∧ a i ≤ 1995 ∧ is_square_free (a i)) →
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 14 → are_relatively_prime (a i) (a j)) →
  ∃ i, 1 ≤ i ∧ i ≤ 14 ∧ is_prime (a i) :=
sorry

end sequence_contains_prime_l155_155897


namespace typists_retype_time_l155_155422

theorem typists_retype_time
  (x y : ℕ)
  (h1 : (x / 2) + (y / 2) = 25)
  (h2 : 1 / x + 1 / y = 1 / 12) :
  (x = 20 ∧ y = 30) ∨ (x = 30 ∧ y = 20) :=
by
  sorry

end typists_retype_time_l155_155422


namespace symmetric_graph_increasing_interval_l155_155531

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_graph_increasing_interval :
  (∀ x : ℝ, f (-x) = -f x) → -- f is odd
  (∀ x y : ℝ, 3 ≤ x → x < y → y ≤ 7 → f x < f y) → -- f is increasing in [3,7]
  (∀ x : ℝ, 3 ≤ x → x ≤ 7 → f x ≤ 5) → -- f has a maximum value of 5 in [3,7]
  (∀ x y : ℝ, -7 ≤ x → x < y → y ≤ -3 → f x < f y) ∧ -- f is increasing in [-7,-3]
  (∀ x : ℝ, -7 ≤ x → x ≤ -3 → f x ≥ -5) -- f has a minimum value of -5 in [-7,-3]
:= sorry

end symmetric_graph_increasing_interval_l155_155531


namespace geometric_sequence_S12_l155_155900

theorem geometric_sequence_S12 (S : ℕ → ℝ) (S_4_eq : S 4 = 20) (S_8_eq : S 8 = 30) :
  S 12 = 35 :=
by
  sorry

end geometric_sequence_S12_l155_155900


namespace width_of_barrier_l155_155727

theorem width_of_barrier (r1 r2 : ℝ) (h : 2 * π * r1 - 2 * π * r2 = 16 * π) : r1 - r2 = 8 :=
by
  -- The proof would be inserted here, but is not required as per instructions.
  sorry

end width_of_barrier_l155_155727


namespace solution_exists_l155_155060

noncomputable def verifySolution (x y z : ℝ) : Prop := 
  x^2 - y = (z - 1)^2 ∧
  y^2 - z = (x - 1)^2 ∧
  z^2 - x = (y- 1)^2 

theorem solution_exists (x y z : ℝ) (h : verifySolution x y z) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x, y, z) = (-2.93122, 2.21124, 0.71998) ∨ 
  (x, y, z) = (2.21124, 0.71998, -2.93122) ∨ 
  (x, y, z) = (0.71998, -2.93122, 2.21124) :=
sorry

end solution_exists_l155_155060


namespace total_photos_in_gallery_l155_155392

def initial_photos : ℕ := 800
def photos_first_day : ℕ := (2 * initial_photos) / 3
def photos_second_day : ℕ := photos_first_day + 180

theorem total_photos_in_gallery : initial_photos + photos_first_day + photos_second_day = 2046 := by
  -- the proof can be provided here
  sorry

end total_photos_in_gallery_l155_155392


namespace total_pieces_eq_21_l155_155460

-- Definitions based on conditions
def red_pieces : Nat := 5
def yellow_pieces : Nat := 7
def green_pieces : Nat := 11

-- Derived definitions from conditions
def red_cuts : Nat := red_pieces - 1
def yellow_cuts : Nat := yellow_pieces - 1
def green_cuts : Nat := green_pieces - 1

-- Total cuts and the resulting total pieces
def total_cuts : Nat := red_cuts + yellow_cuts + green_cuts
def total_pieces : Nat := total_cuts + 1

-- Prove the total number of pieces is 21
theorem total_pieces_eq_21 : total_pieces = 21 := by
  sorry

end total_pieces_eq_21_l155_155460


namespace range_of_a_l155_155784

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a + 1) * x > a + 1 ↔ x > 1) : a > -1 := 
by
  sorry

end range_of_a_l155_155784


namespace probability_of_each_category_selected_l155_155845

theorem probability_of_each_category_selected :
  let total_items := 8 in
  let swim_items := 1 in
  let ball_items := 3 in
  let track_items := 4 in
  let total_combinations := Nat.choose total_items 4 in
  let valid_combinations := Nat.choose swim_items 1 * 
                            Nat.choose ball_items 1 * 
                            Nat.choose track_items 2 +
                            Nat.choose swim_items 1 * 
                            Nat.choose ball_items 2 * 
                            Nat.choose track_items 1 in
  (total_combinations ≠ 0) ->
  (valid_combinations / total_combinations : ℚ) = 3 / 7 :=
by
  /- conditions definition -/
  let total_items := 8
  let swim_items := 1
  let ball_items := 3
  let track_items := 4
  /- combination calculations -/
  let total_combinations := Nat.choose total_items 4
  let valid_combinations := Nat.choose swim_items 1 * 
                            Nat.choose ball_items 1 * 
                            Nat.choose track_items 2 +
                            Nat.choose swim_items 1 * 
                            Nat.choose ball_items 2 * 
                            Nat.choose track_items 1 
  /- proof body -/
  intro h
  sorry

end probability_of_each_category_selected_l155_155845


namespace evaluate_expression_l155_155627

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = -3) :
  (2 * x)^2 * (y^2)^3 * z^2 = 1 / 81 :=
by
  -- Proof omitted
  sorry

end evaluate_expression_l155_155627


namespace redistribute_marbles_l155_155206

theorem redistribute_marbles :
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  (d + m + p + v) / n = 15 :=
by
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  sorry

end redistribute_marbles_l155_155206


namespace expression_square_minus_three_times_l155_155338

-- Defining the statement
theorem expression_square_minus_three_times (a b : ℝ) : a^2 - 3 * b = a^2 - 3 * b := 
by
  sorry

end expression_square_minus_three_times_l155_155338


namespace simplify_exponentiation_l155_155135

theorem simplify_exponentiation (x : ℕ) :
  (x^5 * x^3)^2 = x^16 := 
by {
  sorry -- proof will go here
}

end simplify_exponentiation_l155_155135


namespace infinite_series_sum_l155_155760

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) * (1 / 1998)^n) = (3992004 / 3988009) :=
by sorry

end infinite_series_sum_l155_155760


namespace parity_of_f_max_value_of_f_l155_155642

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log2 (2^x + 1)

-- Statement 1: If a = -1/2, then f(x) is an even function
theorem parity_of_f (a : ℝ) : a = -1/2 → ∀ x : ℝ, f a x = f a (-x) := sorry

-- Statement 2: Given a > 0 and the minimum value condition of y = f(x) + f⁻¹(x),
-- show that a = 1 and the maximum value of f(x) on [1,2] is 2 + log_2 5
theorem max_value_of_f (a : ℝ) (h : a > 0)
  (h_min : ∀ x ∈ Icc 1 2, f a x + invFun (f a) x = 1 + Real.log2 3) :
  a = 1 ∧ ∀ x ∈ Icc 1 2, f a 2 = 2 + Real.log2 5 := sorry

end parity_of_f_max_value_of_f_l155_155642


namespace outfit_choices_l155_155363

/-- Given 8 shirts, 8 pairs of pants, and 8 hats, each in 8 colors,
only 6 colors have a matching shirt, pair of pants, and hat.
Each item in the outfit must be of a different color.
Prove that the number of valid outfits is 368. -/
theorem outfit_choices (shirts pants hats colors : ℕ)
  (matching_colors : ℕ)
  (h_shirts : shirts = 8)
  (h_pants : pants = 8)
  (h_hats : hats = 8)
  (h_colors : colors = 8)
  (h_matching_colors : matching_colors = 6) :
  (shirts * pants * hats) - 3 * (matching_colors * colors) = 368 := 
by {
  sorry
}

end outfit_choices_l155_155363


namespace range_y_over_x_l155_155347

theorem range_y_over_x {x y : ℝ} (h : (x-4)^2 + (y-2)^2 ≤ 4) : 
  ∃ k : ℝ, k = y / x ∧ 0 ≤ k ∧ k ≤ 4/3 :=
sorry

end range_y_over_x_l155_155347


namespace sum_primes_between_1_and_20_l155_155989

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l155_155989


namespace quadratic_inequality_l155_155121

theorem quadratic_inequality
  (a b c : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by
  sorry

end quadratic_inequality_l155_155121


namespace nth_equation_pattern_l155_155815

theorem nth_equation_pattern (n: ℕ) :
  (∀ k : ℕ, 1 ≤ k → ∃ a b c d : ℕ, (a * c ≠ 0) ∧ (b * d ≠ 0) ∧ (a = k) ∧ (b = k + 1) → 
    (a + 3 * (2 * a)) / (b + 3 * (2 * b)) = a / b) :=
by
  sorry

end nth_equation_pattern_l155_155815


namespace number_of_problems_l155_155566

theorem number_of_problems (Terry_score : ℤ) (points_right : ℤ) (points_wrong : ℤ) (wrong_ans : ℤ) 
  (h_score : Terry_score = 85) (h_points_right : points_right = 4) 
  (h_points_wrong : points_wrong = -1) (h_wrong_ans : wrong_ans = 3) : 
  ∃ (total_problems : ℤ), total_problems = 25 :=
by
  sorry

end number_of_problems_l155_155566


namespace selection_probability_correct_l155_155250

def percentage_women : ℝ := 0.55
def percentage_men : ℝ := 0.45

def women_below_35 : ℝ := 0.20
def women_35_to_50 : ℝ := 0.35
def women_above_50 : ℝ := 0.45

def men_below_35 : ℝ := 0.30
def men_35_to_50 : ℝ := 0.40
def men_above_50 : ℝ := 0.30

def women_below_35_lawyers : ℝ := 0.35
def women_below_35_doctors : ℝ := 0.45
def women_below_35_engineers : ℝ := 0.20

def women_35_to_50_lawyers : ℝ := 0.25
def women_35_to_50_doctors : ℝ := 0.50
def women_35_to_50_engineers : ℝ := 0.25

def women_above_50_lawyers : ℝ := 0.20
def women_above_50_doctors : ℝ := 0.30
def women_above_50_engineers : ℝ := 0.50

def men_below_35_lawyers : ℝ := 0.40
def men_below_35_doctors : ℝ := 0.30
def men_below_35_engineers : ℝ := 0.30

def men_35_to_50_lawyers : ℝ := 0.45
def men_35_to_50_doctors : ℝ := 0.25
def men_35_to_50_engineers : ℝ := 0.30

def men_above_50_lawyers : ℝ := 0.30
def men_above_50_doctors : ℝ := 0.40
def men_above_50_engineers : ℝ := 0.30

theorem selection_probability_correct :
  (percentage_women * women_below_35 * women_below_35_lawyers +
   percentage_men * men_above_50 * men_above_50_engineers +
   percentage_women * women_35_to_50 * women_35_to_50_doctors +
   percentage_men * men_35_to_50 * men_35_to_50_doctors) = 0.22025 :=
by
  sorry

end selection_probability_correct_l155_155250


namespace beth_overall_score_l155_155479

-- Definitions for conditions
def percent_score (score_pct : ℕ) (total_problems : ℕ) : ℕ :=
  (score_pct * total_problems) / 100

def total_correct_answers : ℕ :=
  percent_score 60 15 + percent_score 85 20 + percent_score 75 25

def total_problems : ℕ := 15 + 20 + 25

def combined_percentage : ℕ :=
  (total_correct_answers * 100) / total_problems

-- The statement to be proved
theorem beth_overall_score : combined_percentage = 75 := by
  sorry

end beth_overall_score_l155_155479


namespace circle_line_intersection_l155_155643

theorem circle_line_intersection (x y a : ℝ) (A B C O : ℝ × ℝ) :
  (x + y = 1) ∧ ((x^2 + y^2) = a) ∧ 
  (O = (0, 0)) ∧ 
  (x^2 + y^2 = a ∧ (A.1^2 + A.2^2 = a) ∧ (B.1^2 + B.2^2 = a) ∧ (C.1^2 + C.2^2 = a) ∧ 
  (A.1 + B.1 = C.1) ∧ (A.2 + B.2 = C.2)) -> 
  a = 2 := 
sorry

end circle_line_intersection_l155_155643


namespace average_marbles_of_other_colors_l155_155128

theorem average_marbles_of_other_colors :
  let total_percentage := 100
  let clear_percentage := 40
  let black_percentage := 20
  let other_percentage := total_percentage - clear_percentage - black_percentage
  let marbles_taken := 5
  (other_percentage / 100) * marbles_taken = 2 :=
by
  sorry

end average_marbles_of_other_colors_l155_155128


namespace earnings_per_visit_l155_155543

-- Define the conditions of the problem
def website_visits_per_month : ℕ := 30000
def earning_per_day : Real := 10
def days_in_month : ℕ := 30

-- Prove that John gets $0.01 per visit
theorem earnings_per_visit :
  (earning_per_day * days_in_month) / website_visits_per_month = 0.01 :=
by
  sorry

end earnings_per_visit_l155_155543


namespace question1_1_question1_2_question2_l155_155551

open Set

noncomputable def universal_set : Set ℝ := univ

def setA : Set ℝ := { x | x^2 - 9 * x + 18 ≥ 0 }

def setB : Set ℝ := { x | -2 < x ∧ x < 9 }

def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem question1_1 : ∀ x, x ∈ setA ∨ x ∈ setB :=
by sorry

theorem question1_2 : ∀ x, x ∈ (universal_set \ setA) ∩ setB ↔ (3 < x ∧ x < 6) :=
by sorry

theorem question2 (a : ℝ) (h : setC a ⊆ setB) : -2 ≤ a ∧ a ≤ 8 :=
by sorry

end question1_1_question1_2_question2_l155_155551


namespace time_to_cover_escalator_l155_155849

noncomputable def escalator_speed : ℝ := 8
noncomputable def person_speed : ℝ := 2
noncomputable def escalator_length : ℝ := 160
noncomputable def combined_speed : ℝ := escalator_speed + person_speed

theorem time_to_cover_escalator :
  escalator_length / combined_speed = 16 := by
  sorry

end time_to_cover_escalator_l155_155849


namespace sum_of_primes_1_to_20_l155_155998

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l155_155998


namespace sequence_a4_eq_5_over_3_l155_155377

theorem sequence_a4_eq_5_over_3 :
  ∀ (a : ℕ → ℚ), a 1 = 1 → (∀ n > 1, a n = 1 / a (n - 1) + 1) → a 4 = 5 / 3 :=
by
  intro a ha1 H
  sorry

end sequence_a4_eq_5_over_3_l155_155377


namespace problem_statement_l155_155678

-- Definitions based on problem conditions
def p (a b c : ℝ) : Prop := a > b → (a * c^2 > b * c^2)

def q : Prop := ∃ x_0 : ℝ, (x_0 > 0) ∧ (x_0 - 1 + Real.log x_0 = 0)

-- Main theorem
theorem problem_statement : (¬ (∀ a b c : ℝ, p a b c)) ∧ q :=
by sorry

end problem_statement_l155_155678


namespace find_original_money_sandy_took_l155_155958

noncomputable def originalMoney (remainingMoney : ℝ) (clothingPercent electronicsPercent foodPercent additionalSpendPercent salesTaxPercent : ℝ) : Prop :=
  let X := (remainingMoney / (1 - ((clothingPercent + electronicsPercent + foodPercent) + additionalSpendPercent) * (1 + salesTaxPercent)))
  abs (X - 397.73) < 0.01

theorem find_original_money_sandy_took :
  originalMoney 140 0.25 0.15 0.10 0.20 0.08 :=
sorry

end find_original_money_sandy_took_l155_155958


namespace annual_interest_payment_l155_155440

noncomputable def principal : ℝ := 9000
noncomputable def rate : ℝ := 9 / 100
noncomputable def time : ℝ := 1
noncomputable def interest : ℝ := principal * rate * time

theorem annual_interest_payment : interest = 810 := by
  sorry

end annual_interest_payment_l155_155440


namespace survey_steps_correct_l155_155105

theorem survey_steps_correct :
  ∀ steps : (ℕ → ℕ), (steps 1 = 2) → (steps 2 = 4) → (steps 3 = 3) → (steps 4 = 1) → True :=
by
  intros steps h1 h2 h3 h4
  exact sorry

end survey_steps_correct_l155_155105


namespace passes_after_6_l155_155292

-- Define the sequence a_n where a_n represents the number of ways the ball is in A's hands after n passes
def passes : ℕ → ℕ
| 0       => 1       -- Initially, the ball is in A's hands (1 way)
| (n + 1) => 2^n - passes n

-- Theorem to prove the number of different passing methods after 6 passes
theorem passes_after_6 : passes 6 = 22 := by
  sorry

end passes_after_6_l155_155292


namespace inequality_proof_l155_155779

theorem inequality_proof
  (x y : ℝ) (h1 : x^2 + x * y + y^2 = (x + y)^2 - x * y) 
  (h2 : x + y ≥ 2 * Real.sqrt (x * y)) : 
  x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := 
by
  sorry

end inequality_proof_l155_155779


namespace points_on_hyperbola_l155_155503

theorem points_on_hyperbola {s : ℝ} :
  let x := Real.exp s - Real.exp (-s)
  let y := 5 * (Real.exp s + Real.exp (-s))
  (y^2 / 100 - x^2 / 4 = 1) :=
by
  sorry

end points_on_hyperbola_l155_155503


namespace probability_is_correct_l155_155594

noncomputable def probability_hare_given_claims : ℝ :=
  let P_A := 1/2 -- Probability the individual is a hare
  let P_notA := 1/2 -- Probability the individual is not a hare (rabbit)
  let P_B_given_A := 1/4 -- Probability a hare claims not to be a hare
  let P_C_given_A := 3/4 -- Probability a hare claims not to be a rabbit
  let P_B_given_notA := 2/3 -- Probability a rabbit claims not to be a hare
  let P_C_given_notA := 1/3 -- Probability a rabbit claims not to be a rabbit
  let P_A_and_B_and_C := P_A * P_B_given_A * P_C_given_A -- Joint probability A ∩ B ∩ C
  let P_notA_and_B_and_C := P_notA * P_B_given_notA * P_C_given_notA -- Joint probability ¬A ∩ B ∩ C
  let P_B_and_C := P_A_and_B_and_C + P_notA_and_B_and_C -- Total probability of B ∩ C
  P_A_and_B_and_C / P_B_and_C -- Conditional probability A | (B ∩ C)

theorem probability_is_correct : probability_hare_given_claims = 27 / 59 := 
  by 
    -- Establish the values directly as per the conditions
    let P_A : ℝ := 1/2
    let P_B_given_A : ℝ := 1/4
    let P_C_given_A : ℝ := 3/4
    let P_notA : ℝ := 1/2
    let P_B_given_notA : ℝ := 2/3
    let P_C_given_notA : ℝ := 1/3
    let P_A_and_B_and_C : ℝ := P_A * P_B_given_A * P_C_given_A
    let P_notA_and_B_and_C : ℝ := P_notA * P_B_given_notA * P_C_given_notA
    let P_B_and_C : ℝ := P_A_and_B_and_C + P_notA_and_B_and_C
    have P_B_and_C_value : P_B_and_C = 59 / 288 := by sorry
    have P_A_and_B_and_C_value : P_A_and_B_and_C = 3 / 32 := by sorry
    have prob_value : (3 / 32) * (288 / 59) = 27 / 59 :=
      by sorry
    exact prob_value

end probability_is_correct_l155_155594


namespace find_a_l155_155759

def system_of_equations (a x y : ℝ) : Prop :=
  y - 2 = a * (x - 4) ∧ (2 * x) / (|y| + y) = Real.sqrt x

def domain_constraints (x y : ℝ) : Prop :=
  y > 0 ∧ x ≥ 0

def valid_a (a : ℝ) : Prop :=
  (∃ x y, domain_constraints x y ∧ system_of_equations a x y)

theorem find_a :
  ∀ a : ℝ, valid_a a ↔
  ((a < 0.5 ∧ ∃ y, y = 2 - 4 * a ∧ y > 0) ∨ 
   (∃ x y, x = 4 ∧ y = 2 ∧ x ≥ 0 ∧ y > 0) ∨
   (0 < a ∧ a ≠ 0.25 ∧ a < 0.5 ∧ ∃ x y, x = (1 - 2 * a) / a ∧ y = (1 - 2 * a) / a)) :=
by sorry

end find_a_l155_155759


namespace find_y_l155_155243

-- Define the known values and the proportion relation
variable (x y : ℝ)
variable (h1 : 0.75 / x = y / 7)
variable (h2 : x = 1.05)

theorem find_y : y = 5 :=
by
sorry

end find_y_l155_155243


namespace point_not_in_image_of_plane_l155_155388

def satisfies_plane (P : ℝ × ℝ × ℝ) (A B C D : ℝ) : Prop :=
  let (x, y, z) := P
  A * x + B * y + C * z + D = 0

theorem point_not_in_image_of_plane :
  let A := (2, -3, 1)
  let aA := 1
  let aB := 1
  let aC := -2
  let aD := 2
  let k := 5 / 2
  let a'A := aA
  let a'B := aB
  let a'C := aC
  let a'D := k * aD
  ¬ satisfies_plane A a'A a'B a'C a'D :=
by
  -- TODO: Proof needed
  sorry

end point_not_in_image_of_plane_l155_155388


namespace min_value_expr_l155_155633

theorem min_value_expr : 
  ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -3 := 
sorry

end min_value_expr_l155_155633


namespace remainder_of_87_pow_88_plus_7_l155_155697

theorem remainder_of_87_pow_88_plus_7 :
  (87^88 + 7) % 88 = 8 :=
by sorry

end remainder_of_87_pow_88_plus_7_l155_155697


namespace uber_profit_l155_155433

-- Define conditions
def income : ℕ := 30000
def initial_cost : ℕ := 18000
def trade_in : ℕ := 6000

-- Define depreciation cost
def depreciation_cost : ℕ := initial_cost - trade_in

-- Define the profit
def profit : ℕ := income - depreciation_cost

-- The theorem to be proved
theorem uber_profit : profit = 18000 := by 
  sorry

end uber_profit_l155_155433


namespace find_perimeter_and_sin2A_of_triangle_l155_155083

theorem find_perimeter_and_sin2A_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 3) (h_B : B = Real.pi / 3) (h_area : 6 * Real.sqrt 3 = 6 * Real.sqrt 3)
  (h_S : S_ABC = 6 * Real.sqrt 3) : 
  (a + b + c = 18) ∧ (Real.sin (2 * A) = (39 * Real.sqrt 3) / 98) := 
by 
  -- The proof will be placed here. Assuming a valid proof exists.
  sorry

end find_perimeter_and_sin2A_of_triangle_l155_155083


namespace quadratic_inequality_solution_l155_155496

theorem quadratic_inequality_solution (d : ℝ) 
  (h1 : 0 < d) 
  (h2 : d < 16) : 
  ∃ x : ℝ, (x^2 - 8*x + d < 0) :=
  sorry

end quadratic_inequality_solution_l155_155496


namespace count_pos_three_digit_ints_with_same_digits_l155_155647

-- Define a structure to encapsulate the conditions for a three-digit number less than 700 with at least two digits the same.
structure valid_int (n : ℕ) : Prop :=
  (three_digit : 100 ≤ n ∧ n < 700)
  (same_digits : ∃ d₁ d₂ d₃ : ℕ, ((100 * d₁ + 10 * d₂ + d₃ = n) ∧ (d₁ = d₂ ∨ d₂ = d₃ ∨ d₁ = d₃)))

-- The number of integers satisfying the conditions
def count_valid_ints : ℕ :=
  168

-- The theorem to prove
theorem count_pos_three_digit_ints_with_same_digits : 
  (∃ n, valid_int n) → 168 :=
by
  -- Since the proof is not required, we add sorry here.
  sorry

end count_pos_three_digit_ints_with_same_digits_l155_155647


namespace largest_integer_solving_inequality_l155_155202

theorem largest_integer_solving_inequality :
  ∃ (x : ℤ), (7 - 5 * x > 22) ∧ ∀ (y : ℤ), (7 - 5 * y > 22) → x ≥ y ∧ x = -4 :=
by
  sorry

end largest_integer_solving_inequality_l155_155202


namespace expected_value_of_win_is_2_5_l155_155037

noncomputable def expected_value_of_win : ℚ := 
  (1/6) * (6 - 1) + (1/6) * (6 - 2) + (1/6) * (6 - 3) + 
  (1/6) * (6 - 4) + (1/6) * (6 - 5) + (1/6) * (6 - 6)

theorem expected_value_of_win_is_2_5 : expected_value_of_win = 5 / 2 := 
by
  -- Proof steps will go here
  sorry

end expected_value_of_win_is_2_5_l155_155037


namespace roots_eq_s_l155_155670

theorem roots_eq_s (n c d : ℝ) (h₁ : c * d = 6) (h₂ : c + d = n)
  (h₃ : c^2 + 1 / d = c^2 + d^2 + 1 / c): 
  (n + 217 / 6) = d^2 + 1/ c * (n + c + d)
  :=
by
  -- The proof will go here
  sorry

end roots_eq_s_l155_155670


namespace asha_win_probability_l155_155327

theorem asha_win_probability :
  let P_Lose := (3 : ℚ) / 8
  let P_Tie := (1 : ℚ) / 4
  P_Lose + P_Tie < 1 → 1 - P_Lose - P_Tie = (3 : ℚ) / 8 := 
by
  sorry

end asha_win_probability_l155_155327


namespace solution_set_of_abs_inequality_l155_155287

theorem solution_set_of_abs_inequality (x : ℝ) : 
  (x < 5 ↔ |x - 8| - |x - 4| > 2) :=
sorry

end solution_set_of_abs_inequality_l155_155287


namespace essentially_different_proportions_l155_155919

theorem essentially_different_proportions (x y z t : α) [DecidableEq α] 
  (h1 : x ≠ y) (h2 : x ≠ z) (h3 : x ≠ t) (h4 : y ≠ z) (h5 : y ≠ t) (h6 : z ≠ t) : 
  ∃ n : ℕ, n = 3 := by
  sorry

end essentially_different_proportions_l155_155919


namespace range_of_z_l155_155796

theorem range_of_z (x y : ℝ) (hx1 : x - 2 * y + 1 ≥ 0) (hx2 : y ≥ x) (hx3 : x ≥ 0) :
  ∃ z, z = x^2 + y^2 ∧ 0 ≤ z ∧ z ≤ 2 :=
by
  sorry

end range_of_z_l155_155796


namespace prob_neither_prime_nor_composite_l155_155245

theorem prob_neither_prime_nor_composite :
  (1 / 95 : ℚ) = 1 / 95 := by
  sorry

end prob_neither_prime_nor_composite_l155_155245


namespace C_can_complete_work_in_100_days_l155_155439

-- Definitions for conditions
def A_work_rate : ℚ := 1 / 20
def B_work_rate : ℚ := 1 / 15
def work_done_by_A_and_B : ℚ := 6 * (1 / 20 + 1 / 15)
def remaining_work : ℚ := 1 - work_done_by_A_and_B
def work_done_by_A_in_5_days : ℚ := 5 * (1 / 20)
def work_done_by_C_in_5_days : ℚ := remaining_work - work_done_by_A_in_5_days
def C_work_rate_in_5_days : ℚ := work_done_by_C_in_5_days / 5

-- Statement to prove
theorem C_can_complete_work_in_100_days : 
  work_done_by_C_in_5_days ≠ 0 → 1 / C_work_rate_in_5_days = 100 :=
by
  -- proof of the theorem
  sorry

end C_can_complete_work_in_100_days_l155_155439


namespace statement_1_equiv_statement_2_equiv_l155_155559

-- Statement 1
variable (A B C : Prop)

theorem statement_1_equiv : ((A ∨ B) → C) ↔ (A → C) ∧ (B → C) :=
by
  sorry

-- Statement 2
theorem statement_2_equiv : (A → (B ∧ C)) ↔ (A → B) ∧ (A → C) :=
by
  sorry

end statement_1_equiv_statement_2_equiv_l155_155559


namespace abs_inequality_solution_l155_155579

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 1| < 1) ↔ (0 < x ∧ x < 2) :=
sorry

end abs_inequality_solution_l155_155579


namespace sym_sum_ineq_l155_155447

theorem sym_sum_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x + y + z = 1 / x + 1 / y + 1 / z) : x * y + y * z + z * x ≥ 3 :=
by
  sorry

end sym_sum_ineq_l155_155447


namespace smallest_third_term_arith_seq_l155_155843

theorem smallest_third_term_arith_seq {a d : ℕ} 
  (h1 : a > 0) 
  (h2 : d > 0) 
  (sum_eq : 5 * a + 10 * d = 80) : 
  a + 2 * d = 16 := 
by {
  sorry
}

end smallest_third_term_arith_seq_l155_155843


namespace average_beef_sales_l155_155122

theorem average_beef_sales 
  (thursday_sales : ℕ)
  (friday_sales : ℕ)
  (saturday_sales : ℕ)
  (h_thursday : thursday_sales = 210)
  (h_friday : friday_sales = 2 * thursday_sales)
  (h_saturday : saturday_sales = 150) :
  (thursday_sales + friday_sales + saturday_sales) / 3 = 260 :=
by sorry

end average_beef_sales_l155_155122


namespace trader_gain_percentage_l155_155615

theorem trader_gain_percentage (C : ℝ) (h1 : 95 * C = (95 * C - cost_of_95_pens) + (19 * C)) :
  100 * (19 * C / (95 * C)) = 20 := 
by {
  sorry
}

end trader_gain_percentage_l155_155615


namespace sum_of_primes_1_to_20_l155_155985

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l155_155985


namespace expression_value_is_one_l155_155588

theorem expression_value_is_one :
  let a1 := 121
  let b1 := 19
  let a2 := 91
  let b2 := 13
  (a1^2 - b1^2) / (a2^2 - b2^2) * ((a2 - b2) * (a2 + b2)) / ((a1 - b1) * (a1 + b1)) = 1 := by
  sorry

end expression_value_is_one_l155_155588


namespace group_interval_eq_l155_155656

noncomputable def group_interval (a b m h : ℝ) : ℝ := abs (a - b)

theorem group_interval_eq (a b m h : ℝ) 
  (h1 : h = m / abs (a - b)) :
  abs (a - b) = m / h := 
by 
  sorry

end group_interval_eq_l155_155656


namespace f_1982_value_l155_155389

noncomputable def f (n : ℕ) : ℕ := sorry  -- placeholder for the function definition

axiom f_condition_2 : f 2 = 0
axiom f_condition_3 : f 3 > 0
axiom f_condition_9999 : f 9999 = 3333
axiom f_add_condition (m n : ℕ) : f (m+n) - f m - f n = 0 ∨ f (m+n) - f m - f n = 1

open Nat

theorem f_1982_value : f 1982 = 660 :=
by
  sorry  -- proof goes here

end f_1982_value_l155_155389


namespace avg_salary_feb_mar_apr_may_l155_155279

def avg_salary_4_months : ℝ := 8000
def salary_jan : ℝ := 3700
def salary_may : ℝ := 6500
def total_salary_4_months := 4 * avg_salary_4_months
def total_salary_feb_mar_apr := total_salary_4_months - salary_jan
def total_salary_feb_mar_apr_may := total_salary_feb_mar_apr + salary_may

theorem avg_salary_feb_mar_apr_may : total_salary_feb_mar_apr_may / 4 = 8700 := by
  sorry

end avg_salary_feb_mar_apr_may_l155_155279


namespace solve_for_x_l155_155008

theorem solve_for_x : ∀ x : ℝ, 3^(2 * x) = Real.sqrt 27 → x = 3 / 4 :=
by
  intro x h
  sorry

end solve_for_x_l155_155008


namespace number_added_is_10_l155_155974

theorem number_added_is_10 (x y a : ℕ) (h1 : y = 40) 
  (h2 : x * 4 = 3 * y) 
  (h3 : (x + a) * 5 = 4 * (y + a)) : a = 10 := 
by
  sorry

end number_added_is_10_l155_155974


namespace find_other_root_l155_155310

theorem find_other_root (m : ℝ) :
  (∃ m : ℝ, (∀ x : ℝ, (x = -6 → (x^2 + m * x - 6 = 0))) → (x^2 + m * x - 6 = (x + 6) * (x - 1)) → (∀ x : ℝ, (x^2 + 5 * x - 6 = 0) → (x = -6 ∨ x = 1))) :=
sorry

end find_other_root_l155_155310


namespace simplify_sqrt_expression_l155_155273

theorem simplify_sqrt_expression :
  sqrt (5 * 3) * sqrt (3^4 * 5^2) = 15 * sqrt 15 :=
by sorry

end simplify_sqrt_expression_l155_155273


namespace triangle_area_eq_l155_155340

theorem triangle_area_eq :
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  area = 9 / 4 :=
by
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  sorry

end triangle_area_eq_l155_155340


namespace exists_n_divisible_l155_155383

theorem exists_n_divisible (k : ℕ) (m : ℤ) (hk : k > 0) (hm : m % 2 = 1) : 
  ∃ n : ℕ, n > 0 ∧ 2^k ∣ (n^n - m) :=
by
  sorry

end exists_n_divisible_l155_155383


namespace sufficient_but_not_necessary_condition_l155_155625

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a - Real.sin x

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, f' a x > 0 → (a > 1)) ∧ (¬∀ x, f' a x ≥ 0 → (a > 1)) := sorry

end sufficient_but_not_necessary_condition_l155_155625


namespace speed_of_first_bus_l155_155972

theorem speed_of_first_bus (v : ℕ) (h : (v + 60) * 4 = 460) : v = 55 :=
by
  sorry

end speed_of_first_bus_l155_155972


namespace complement_of_M_l155_155668

def M : Set ℝ := {x | x^2 - 2 * x > 0}

def U : Set ℝ := Set.univ

theorem complement_of_M :
  (U \ M) = (Set.Icc 0 2) :=
by
  sorry

end complement_of_M_l155_155668


namespace total_time_taken_l155_155955

theorem total_time_taken 
  (R : ℝ) -- Rickey's speed
  (T_R : ℝ := 40) -- Rickey's time
  (T_P : ℝ := (40 * (4 / 3))) -- Prejean's time derived from given conditions
  (P : ℝ := (3 / 4) * R) -- Prejean's speed
  (k : ℝ := 40 * R) -- constant k for distance
 
  (h1 : T_R = 40)
  (h2 : T_P = 40 * (4 / 3))
  -- Main goal: Prove total time taken equals 93.33 minutes
  : (T_R + T_P) = 93.33 := 
  sorry

end total_time_taken_l155_155955


namespace area_of_triangle_example_l155_155423

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_example : 
  area_of_triangle (3, 3) (3, 10) (12, 19) = 31.5 :=
by
  sorry

end area_of_triangle_example_l155_155423


namespace problem1_l155_155032

theorem problem1 (a : ℝ) 
    (circle_eqn : ∀ (x y : ℝ), x^2 + y^2 - 2*a*x + a = 0)
    (line_eqn : ∀ (x y : ℝ), a*x + y + 1 = 0)
    (chord_length : ∀ (x y : ℝ), (ax + y + 1 = 0) ∧ (x^2 + y^2 - 2*a*x + a = 0)  -> ((x - x')^2 + (y - y')^2 = 4)) : 
    a = -2 := sorry

end problem1_l155_155032


namespace exist_two_divisible_by_n_l155_155823

theorem exist_two_divisible_by_n (n : ℤ) (a : Fin (n.toNat + 1) → ℤ) :
  ∃ (i j : Fin (n.toNat + 1)), i ≠ j ∧ (a i - a j) % n = 0 :=
by
  sorry

end exist_two_divisible_by_n_l155_155823


namespace smallest_number_of_pets_l155_155635

noncomputable def smallest_common_multiple (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

theorem smallest_number_of_pets : smallest_common_multiple 3 15 9 = 45 :=
by
  sorry

end smallest_number_of_pets_l155_155635


namespace cubics_sum_l155_155096

theorem cubics_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : x^3 + y^3 = 640 :=
by
  sorry

end cubics_sum_l155_155096


namespace wood_needed_l155_155325

variable (total_needed : ℕ) (friend_pieces : ℕ) (brother_pieces : ℕ)

/-- Alvin's total needed wood is 376 pieces, he got 123 from his friend and 136 from his brother.
    Prove that Alvin needs 117 more pieces. -/
theorem wood_needed (h1 : total_needed = 376) (h2 : friend_pieces = 123) (h3 : brother_pieces = 136) :
  total_needed - (friend_pieces + brother_pieces) = 117 := by
  sorry

end wood_needed_l155_155325


namespace pats_stick_length_correct_l155_155558

noncomputable def jane_stick_length : ℕ := 22
noncomputable def sarah_stick_length : ℕ := jane_stick_length + 24
noncomputable def uncovered_pats_stick : ℕ := sarah_stick_length / 2
noncomputable def covered_pats_stick : ℕ := 7
noncomputable def total_pats_stick : ℕ := uncovered_pats_stick + covered_pats_stick

theorem pats_stick_length_correct : total_pats_stick = 30 := by
  sorry

end pats_stick_length_correct_l155_155558


namespace find_prime_p_l155_155067

theorem find_prime_p (p : ℕ) (hp : Nat.Prime p) (hp_plus_10 : Nat.Prime (p + 10)) (hp_plus_14 : Nat.Prime (p + 14)) : p = 3 := 
sorry

end find_prime_p_l155_155067


namespace units_digit_of_power_l155_155241

theorem units_digit_of_power (base : ℕ) (exp : ℕ) (units_base : ℕ) (units_exp_mod : ℕ) :
  (base % 10 = units_base) → (exp % 2 = units_exp_mod) → (units_base = 9) → (units_exp_mod = 0) →
  (base ^ exp % 10 = 1) :=
by
  intros
  sorry

end units_digit_of_power_l155_155241


namespace dimes_turned_in_l155_155878

theorem dimes_turned_in (total_coins nickels quarters : ℕ) (h1 : total_coins = 11) (h2 : nickels = 2) (h3 : quarters = 7) : 
  ∃ dimes : ℕ, dimes + nickels + quarters = total_coins ∧ dimes = 2 :=
by
  sorry

end dimes_turned_in_l155_155878


namespace mean_points_scored_is_48_l155_155612

def class_points : List ℤ := [50, 57, 49, 57, 32, 46, 65, 28]

theorem mean_points_scored_is_48 : (class_points.sum / class_points.length) = 48 := by
  sorry

end mean_points_scored_is_48_l155_155612


namespace initial_depth_dug_l155_155723

theorem initial_depth_dug :
  (∀ days : ℕ, 75 * 8 * days / D = 140 * 6 * days / 70) → D = 50 :=
by
  sorry

end initial_depth_dug_l155_155723


namespace hyperbola_foci_distance_l155_155498

theorem hyperbola_foci_distance (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 9) :
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 34 := 
by
  sorry

end hyperbola_foci_distance_l155_155498


namespace cost_of_one_unit_each_l155_155035

variables (x y z : ℝ)

theorem cost_of_one_unit_each
  (h1 : 2 * x + 3 * y + z = 130)
  (h2 : 3 * x + 5 * y + z = 205) :
  x + y + z = 55 :=
by
  sorry

end cost_of_one_unit_each_l155_155035


namespace m_in_A_l155_155675

variable (x : ℝ)
variable (A : Set ℝ := {x | x ≤ 2})
noncomputable def m : ℝ := Real.sqrt 2

theorem m_in_A : m ∈ A :=
sorry

end m_in_A_l155_155675


namespace union_complement_l155_155357

open Set

universe u

variable {α : Type u}

def U : Set α := {1, 2, 3, 4, 5}
def A : Set α := {1, 3}
def B : Set α := {1, 2, 4}

theorem union_complement (U A B : Set α) : 
  U = {1, 2, 3, 4, 5} ∧ A = {1, 3} ∧ B = {1, 2, 4} → ((U \ B) ∪ A) = {1, 3, 5} :=
by 
  sorry

end union_complement_l155_155357


namespace smallest_n_for_quadratic_factorization_l155_155342

theorem smallest_n_for_quadratic_factorization :
  ∃ (n : ℤ), (∀ A B : ℤ, A * B = 50 → n = 5 * B + A) ∧ (∀ m : ℤ, 
    (∀ A B : ℤ, A * B = 50 → m ≤ 5 * B + A) → n ≤ m) :=
by
  sorry

end smallest_n_for_quadratic_factorization_l155_155342


namespace ellipses_have_equal_focal_length_l155_155150

-- Define ellipses and their focal lengths
def ellipse1_focal_length : ℝ := 8
def k_condition (k : ℝ) : Prop := 0 < k ∧ k < 9
def ellipse2_focal_length (k : ℝ) : ℝ := 8

-- The main statement
theorem ellipses_have_equal_focal_length (k : ℝ) (hk : k_condition k) :
  ellipse1_focal_length = ellipse2_focal_length k :=
sorry

end ellipses_have_equal_focal_length_l155_155150


namespace refill_cost_calculation_l155_155238

variables (total_spent : ℕ) (refills : ℕ)

def one_refill_cost (total_spent refills : ℕ) : ℕ := total_spent / refills

theorem refill_cost_calculation (h1 : total_spent = 40) (h2 : refills = 4) :
  one_refill_cost total_spent refills = 10 :=
by
  sorry

end refill_cost_calculation_l155_155238


namespace simplify_eval_expression_l155_155824

theorem simplify_eval_expression (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ((a^2 - 1) / (a^2 - 2 * a + 1) - 1 / (1 - a)) / (1 / (a^2 - a)) = 1 :=
  sorry

end simplify_eval_expression_l155_155824


namespace one_girl_made_a_mistake_l155_155079

variables (c_M c_K c_L c_O : ℤ)

theorem one_girl_made_a_mistake (h₁ : c_M + c_K = c_L + c_O + 12) (h₂ : c_K + c_L = c_M + c_O - 7) :
  false := by
  -- Proof intentionally missing
  sorry

end one_girl_made_a_mistake_l155_155079


namespace intersection_points_count_l155_155838

noncomputable def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := x ^ 2 - 4 * x + 4

theorem intersection_points_count : ∃! x y : ℝ, 0 < x ∧ f x = g x ∧ y ≠ x ∧ f y = g y :=
sorry

end intersection_points_count_l155_155838


namespace minimum_guests_needed_l155_155282

theorem minimum_guests_needed (total_food : ℕ) (max_food_per_guest : ℕ) (guests_needed : ℕ) : 
  total_food = 323 → max_food_per_guest = 2 → guests_needed = Nat.ceil (323 / 2) → guests_needed = 162 :=
by
  intros
  sorry

end minimum_guests_needed_l155_155282


namespace correct_rounded_result_l155_155209

-- Definition of rounding to the nearest hundred
def rounded_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 < 50 then n / 100 * 100 else (n / 100 + 1) * 100

-- Given conditions
def sum : ℕ := 68 + 57

-- The theorem to prove
theorem correct_rounded_result : rounded_to_nearest_hundred sum = 100 :=
by
  -- Proof skipped
  sorry

end correct_rounded_result_l155_155209


namespace calculation_l155_155053

theorem calculation : 2005^2 - 2003 * 2007 = 4 :=
by
  have h1 : 2003 = 2005 - 2 := by rfl
  have h2 : 2007 = 2005 + 2 := by rfl
  sorry

end calculation_l155_155053


namespace janet_total_pockets_l155_155936

theorem janet_total_pockets
  (total_dresses : ℕ)
  (dresses_with_pockets : ℕ)
  (dresses_with_2_pockets : ℕ)
  (dresses_with_3_pockets : ℕ)
  (pockets_from_2 : ℕ)
  (pockets_from_3 : ℕ)
  (total_pockets : ℕ)
  (h1 : total_dresses = 24)
  (h2 : dresses_with_pockets = total_dresses / 2)
  (h3 : dresses_with_2_pockets = dresses_with_pockets / 3)
  (h4 : dresses_with_3_pockets = dresses_with_pockets - dresses_with_2_pockets)
  (h5 : pockets_from_2 = 2 * dresses_with_2_pockets)
  (h6 : pockets_from_3 = 3 * dresses_with_3_pockets)
  (h7 : total_pockets = pockets_from_2 + pockets_from_3)
  : total_pockets = 32 := 
by
  sorry

end janet_total_pockets_l155_155936


namespace range_of_values_l155_155506

theorem range_of_values (x : ℝ) : (x^2 - 5 * x + 6 < 0) ↔ (2 < x ∧ x < 3) :=
sorry

end range_of_values_l155_155506


namespace smallest_number_divisibility_l155_155713

theorem smallest_number_divisibility :
  ∃ n : ℕ, (n - 7) % 12 = 0 ∧
           (n - 7) % 16 = 0 ∧
           (n - 7) % 18 = 0 ∧
           (n - 7) % 21 = 0 ∧
           (n - 7) % 28 = 0 ∧
           n = 1015 :=
by {
  let n := 1015,
  use n,
  split, { norm_num }, split, { norm_num }, split, { norm_num }, split, { norm_num }, split, { norm_num }, 
  exact rfl,
  sorry,
}

end smallest_number_divisibility_l155_155713


namespace range_of_m_l155_155100

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 ≥ m * x * (x + y)) ↔ (m ∈ Set.Icc (-6:ℝ) 2) :=
by
  sorry

end range_of_m_l155_155100


namespace sausage_cutting_l155_155459

theorem sausage_cutting (red_pieces yellow_pieces green_pieces total_pieces : ℕ) 
  (h_red : red_pieces = 5)
  (h_yellow : yellow_pieces = 7)
  (h_green : green_pieces = 11)
  (h_total : total_pieces = (red_pieces - 1) + (yellow_pieces - 1) + (green_pieces - 1) + 1) :
  total_pieces = 21 := 
by {
  rw [h_red, h_yellow, h_green],
  sorry
}

end sausage_cutting_l155_155459


namespace valid_sequences_length_21_l155_155362

def valid_sequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else valid_sequences (n - 3) + valid_sequences (n - 4)

theorem valid_sequences_length_21 : valid_sequences 21 = 38 :=
by
  sorry

end valid_sequences_length_21_l155_155362


namespace alpha_cubic_expression_l155_155384

theorem alpha_cubic_expression (α : ℝ) (hα : α^2 - 8 * α - 5 = 0) : α^3 - 7 * α^2 - 13 * α + 6 = 11 :=
sorry

end alpha_cubic_expression_l155_155384


namespace sum_of_squares_first_28_l155_155332

theorem sum_of_squares_first_28 : 
  (28 * (28 + 1) * (2 * 28 + 1)) / 6 = 7722 := by
  sorry

end sum_of_squares_first_28_l155_155332


namespace sum_of_first_110_terms_l155_155698

theorem sum_of_first_110_terms
  (a d : ℤ)
  (S : ℕ → ℤ)
  (h1 : S 10 = 100)
  (h2 : S 100 = 10)
  (h_sum : ∀ n, S n = n * (2 * a + (n - 1) * d) / 2) :
  S 110 = -110 :=
by {
  sorry,
}

end sum_of_first_110_terms_l155_155698


namespace farmer_price_per_dozen_l155_155178

noncomputable def price_per_dozen 
(farmer_chickens : ℕ) 
(eggs_per_chicken : ℕ) 
(total_money_made : ℕ) 
(total_weeks : ℕ) 
(eggs_per_dozen : ℕ) 
: ℕ :=
total_money_made / (total_weeks * (farmer_chickens * eggs_per_chicken) / eggs_per_dozen)

theorem farmer_price_per_dozen 
  (farmer_chickens : ℕ) 
  (eggs_per_chicken : ℕ) 
  (total_money_made : ℕ) 
  (total_weeks : ℕ) 
  (eggs_per_dozen : ℕ) 
  (h_chickens : farmer_chickens = 46) 
  (h_eggs_per_chicken : eggs_per_chicken = 6) 
  (h_money : total_money_made = 552) 
  (h_weeks : total_weeks = 8) 
  (h_dozen : eggs_per_dozen = 12) 
: price_per_dozen farmer_chickens eggs_per_chicken total_money_made total_weeks eggs_per_dozen = 3 := 
by 
  rw [h_chickens, h_eggs_per_chicken, h_money, h_weeks, h_dozen]
  have : (552 : ℕ) / (8 * (46 * 6) / 12) = 3 := by norm_num
  exact this

end farmer_price_per_dozen_l155_155178


namespace smallest_difference_l155_155419

variable (DE EF FD : ℕ)

def is_valid_triangle (DE EF FD : ℕ) : Prop :=
  DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference (h1 : DE < EF)
                           (h2 : EF ≤ FD)
                           (h3 : DE + EF + FD = 1024)
                           (h4 : is_valid_triangle DE EF FD) :
  ∃ d, d = EF - DE ∧ d = 1 :=
by
  sorry

end smallest_difference_l155_155419


namespace solution_for_a_if_fa_eq_a_l155_155221

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 2) / (x - 2)

theorem solution_for_a_if_fa_eq_a (a : ℝ) (h : f a = a) : a = -1 :=
sorry

end solution_for_a_if_fa_eq_a_l155_155221


namespace outlet_two_rate_l155_155042

/-- Definitions and conditions for the problem -/
def tank_volume_feet : ℝ := 20
def inlet_rate_cubic_inches_per_min : ℝ := 5
def outlet_one_rate_cubic_inches_per_min : ℝ := 9
def empty_time_minutes : ℝ := 2880
def cubic_feet_to_cubic_inches : ℝ := 1728
def tank_volume_cubic_inches := tank_volume_feet * cubic_feet_to_cubic_inches

/-- Statement to prove the rate of the other outlet pipe -/
theorem outlet_two_rate (x : ℝ) :
  tank_volume_cubic_inches / empty_time_minutes = outlet_one_rate_cubic_inches_per_min + x - inlet_rate_cubic_inches_per_min → 
  x = 8 :=
by
  sorry

end outlet_two_rate_l155_155042


namespace dave_spent_102_dollars_l155_155884

noncomputable def total_cost (books_animals books_space books_trains cost_per_book : ℕ) : ℕ :=
  (books_animals + books_space + books_trains) * cost_per_book

theorem dave_spent_102_dollars :
  total_cost 8 6 3 6 = 102 := by
  sorry

end dave_spent_102_dollars_l155_155884


namespace cross_section_area_ratio_correct_l155_155931

variable (α : ℝ)
noncomputable def cross_section_area_ratio : ℝ := 2 * (Real.cos α)

theorem cross_section_area_ratio_correct (α : ℝ) : 
  cross_section_area_ratio α = 2 * Real.cos α :=
by
  unfold cross_section_area_ratio
  sorry

end cross_section_area_ratio_correct_l155_155931


namespace find_real_solutions_l155_155630

theorem find_real_solutions (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 5) :
  ( (x - 3) * (x - 4) * (x - 5) * (x - 4) * (x - 3) ) / ( (x - 4) * (x - 5) ) = -1 ↔ x = 10 / 3 ∨ x = 2 / 3 :=
by sorry

end find_real_solutions_l155_155630


namespace Daisy_vs_Bess_l155_155894

-- Define the conditions
def Bess_daily : ℕ := 2
def Brownie_multiple : ℕ := 3
def total_pails_per_week : ℕ := 77
def days_per_week : ℕ := 7

-- Define the weekly production for Bess
def Bess_weekly : ℕ := Bess_daily * days_per_week

-- Define the weekly production for Brownie
def Brownie_weekly : ℕ := Brownie_multiple * Bess_weekly

-- Farmer Red's total weekly milk production is the sum of Bess, Brownie, and Daisy's production
-- We need to prove the difference in weekly production between Daisy and Bess is 7 pails.
theorem Daisy_vs_Bess (Daisy_weekly : ℕ) (h : Bess_weekly + Brownie_weekly + Daisy_weekly = total_pails_per_week) :
  Daisy_weekly - Bess_weekly = 7 :=
by
  sorry

end Daisy_vs_Bess_l155_155894


namespace jogger_ahead_distance_l155_155601

-- Definitions of conditions
def jogger_speed : ℝ := 9  -- km/hr
def train_speed : ℝ := 45  -- km/hr
def train_length : ℝ := 150  -- meters
def passing_time : ℝ := 39  -- seconds

-- The main statement that we want to prove
theorem jogger_ahead_distance : 
  let relative_speed := (train_speed - jogger_speed) * (5 / 18)  -- conversion to m/s
  let distance_covered := relative_speed * passing_time
  let jogger_ahead := distance_covered - train_length
  jogger_ahead = 240 :=
by
  sorry

end jogger_ahead_distance_l155_155601


namespace chocolate_bars_per_box_l155_155969

-- Definitions for the given conditions
def total_chocolate_bars : ℕ := 849
def total_boxes : ℕ := 170

-- The statement to prove
theorem chocolate_bars_per_box : total_chocolate_bars / total_boxes = 5 :=
by 
  -- Proof is omitted here
  sorry

end chocolate_bars_per_box_l155_155969


namespace common_number_l155_155704

theorem common_number (a b c d e u v w : ℝ) (h1 : (a + b + c + d + e) / 5 = 7) 
                                            (h2 : (u + v + w) / 3 = 10) 
                                            (h3 : (a + b + c + d + e + u + v + w) / 8 = 8) 
                                            (h4 : a + b + c + d + e = 35) 
                                            (h5 : u + v + w = 30) 
                                            (h6 : a + b + c + d + e + u + v + w = 64) 
                                            (h7 : 35 + 30 = 65):
  d = u := 
by
  sorry

end common_number_l155_155704


namespace gain_percentage_l155_155186

theorem gain_percentage (SP1 SP2 CP: ℝ) (h1 : SP1 = 102) (h2 : SP2 = 144) (h3 : SP1 = CP - 0.15 * CP) :
  ((SP2 - CP) / CP) * 100 = 20 := by
sorry

end gain_percentage_l155_155186


namespace problem_trip_l155_155754

noncomputable def validate_trip (a b c : ℕ) (t : ℕ) : Prop :=
  a ≥ 1 ∧ a + b + c ≤ 10 ∧ 60 * t = 9 * c - 10 * b

theorem problem_trip (a b c t : ℕ) (h : validate_trip a b c t) : a^2 + b^2 + c^2 = 26 :=
sorry

end problem_trip_l155_155754


namespace problem_1_problem_2_l155_155229

noncomputable def f (x k : ℝ) : ℝ := (2 * k * x) / (x * x + 6 * k)

theorem problem_1 (k m : ℝ) (hk : k > 0)
  (hsol : ∀ x, (f x k) > m ↔ x < -3 ∨ x > -2) :
  ∀ x, 5 * m * x ^ 2 + k * x + 3 > 0 ↔ -1 < x ∧ x < 3 / 2 :=
sorry

theorem problem_2 (k : ℝ) (hk : k > 0)
  (hsol : ∃ (x : ℝ), x > 3 ∧ (f x k) > 1) :
  k > 6 :=
sorry

end problem_1_problem_2_l155_155229


namespace max_min_diff_w_l155_155086

theorem max_min_diff_w (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 4) :
  let w := a^2 + a*b + b^2
  let w1 := max (0^2 + 0*b + b^2) (4^2 + 4*b + b^2)
  let w2 := (2-2)^2 + 12
  w1 - w2 = 4 :=
by
  -- skip the proof
  sorry

end max_min_diff_w_l155_155086


namespace evaluate_nested_fraction_l155_155490

theorem evaluate_nested_fraction :
  (1 / (3 - (1 / (2 - (1 / (3 - (1 / (2 - (1 / 2))))))))) = 11 / 26 :=
by
  sorry

end evaluate_nested_fraction_l155_155490


namespace simplify_frac_op_l155_155247

-- Definition of the operation *
def frac_op (a b c d : ℚ) : ℚ := (a * c) * (d / (b + 1))

-- Proof problem stating the specific operation result
theorem simplify_frac_op :
  frac_op 5 11 9 4 = 15 :=
by
  sorry

end simplify_frac_op_l155_155247


namespace find_x_l155_155411

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x n : ℕ) (h₀ : n = 4) (h₁ : ¬(is_prime (2 * n + x))) : x = 1 :=
by
  sorry

end find_x_l155_155411


namespace john_salary_increase_l155_155112

theorem john_salary_increase :
  let initial_salary : ℝ := 30
  let final_salary : ℝ := ((30 * 1.1) * 1.15) * 1.05
  (final_salary - initial_salary) / initial_salary * 100 = 32.83 := by
  sorry

end john_salary_increase_l155_155112


namespace total_white_roses_l155_155554

-- Define the constants
def n_b : ℕ := 5
def n_t : ℕ := 7
def r_b : ℕ := 5
def r_t : ℕ := 12

-- State the theorem
theorem total_white_roses :
  n_t * r_t + n_b * r_b = 109 :=
by
  -- Automatic proof can be here; using sorry as placeholder
  sorry

end total_white_roses_l155_155554


namespace next_term_geometric_sequence_l155_155976

theorem next_term_geometric_sequence (y : ℝ) : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ), 
  a₀ = 3 ∧ 
  a₁ = 9 * y ∧ 
  a₂ = 27 * y^2 ∧ 
  a₃ = 81 * y^3 ∧ 
  a₄ = a₃ * 3 * y 
  → a₄ = 243 * y^4 := by
  sorry

end next_term_geometric_sequence_l155_155976


namespace smallest_c_inequality_l155_155500

theorem smallest_c_inequality (x : ℕ → ℝ) (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10) :
  ∃ c : ℝ, (∀ x : ℕ → ℝ, x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 = 10 →
    |x 0| + |x 1| + |x 2| + |x 3| + |x 4| + |x 5| + |x 6| + |x 7| + |x 8| ≥ c * |x 4|) ∧ c = 9 := 
by
  sorry

end smallest_c_inequality_l155_155500


namespace mixture_solution_l155_155029

theorem mixture_solution (x y : ℝ) :
  (0.30 * x + 0.40 * y = 32) →
  (x + y = 100) →
  (x = 80) :=
by
  intros h₁ h₂
  sorry

end mixture_solution_l155_155029


namespace num_divisors_of_factorial_9_multiple_3_l155_155623

-- Define the prime factorization of 9!
def factorial_9 := 2^7 * 3^4 * 5 * 7

-- Define the conditions for the exponents a, b, c, d
def valid_exponents (a b c d : ℕ) : Prop :=
  (0 ≤ a ∧ a ≤ 7) ∧ (1 ≤ b ∧ b ≤ 4) ∧ (0 ≤ c ∧ c ≤ 1) ∧ (0 ≤ d ∧ d ≤ 1)

-- Define the number of valid exponent combinations
def num_valid_combinations : ℕ :=
  8 * 4 * 2 * 2

-- Theorem stating that the number of divisors of 9! that are multiples of 3 is 128
theorem num_divisors_of_factorial_9_multiple_3 : num_valid_combinations = 128 := by
  sorry

end num_divisors_of_factorial_9_multiple_3_l155_155623


namespace solve_for_q_l155_155007

theorem solve_for_q : 
  let n : ℤ := 63
  let m : ℤ := 14
  ∀ (q : ℤ),
  (7 : ℤ) / 9 = n / 81 ∧
  (7 : ℤ) / 9 = (m + n) / 99 ∧
  (7 : ℤ) / 9 = (q - m) / 135 → 
  q = 119 :=
by
  sorry

end solve_for_q_l155_155007


namespace total_presents_l155_155284

-- Definitions based on the problem conditions
def numChristmasPresents : ℕ := 60
def numBirthdayPresents : ℕ := numChristmasPresents / 2

-- Theorem statement
theorem total_presents : numChristmasPresents + numBirthdayPresents = 90 :=
by
  -- Proof is omitted
  sorry

end total_presents_l155_155284


namespace inverse_variation_z_x_square_l155_155580

theorem inverse_variation_z_x_square (x z : ℝ) (K : ℝ) 
  (h₀ : z * x^2 = K) 
  (h₁ : x = 3 ∧ z = 2)
  (h₂ : z = 8) :
  x = 3 / 2 := 
by 
  sorry

end inverse_variation_z_x_square_l155_155580


namespace smallest_w_value_l155_155304

theorem smallest_w_value (w : ℕ) (hw : w > 0) :
  (∀ k : ℕ, (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (10^2 ∣ 936 * w)) ↔ w = 900 := 
sorry

end smallest_w_value_l155_155304


namespace arthur_reading_pages_l155_155195

theorem arthur_reading_pages :
  let total_goal : ℕ := 800
  let pages_read_from_500_book : ℕ := 500 * 80 / 100 -- 80% of 500 pages
  let pages_read_from_1000_book : ℕ := 1000 / 5 -- 1/5 of 1000 pages
  let total_pages_read : ℕ := pages_read_from_500_book + pages_read_from_1000_book
  let remaining_pages : ℕ := total_goal - total_pages_read
  remaining_pages = 200 :=
by
  -- placeholder for actual proof
  sorry

end arthur_reading_pages_l155_155195


namespace range_of_a_for_min_value_at_x_eq_1_l155_155230

noncomputable def f (a x : ℝ) : ℝ := a*x^3 + (a-1)*x^2 - x + 2

theorem range_of_a_for_min_value_at_x_eq_1 :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a 1 ≤ f a x) → a ≤ 3 / 5 :=
by
  sorry

end range_of_a_for_min_value_at_x_eq_1_l155_155230


namespace students_per_class_l155_155619

theorem students_per_class
  (cards_per_student : Nat)
  (periods_per_day : Nat)
  (cost_per_pack : Nat)
  (total_spent : Nat)
  (cards_per_pack : Nat)
  (students_per_class : Nat)
  (H1 : cards_per_student = 10)
  (H2 : periods_per_day = 6)
  (H3 : cost_per_pack = 3)
  (H4 : total_spent = 108)
  (H5 : cards_per_pack = 50)
  (H6 : students_per_class = 30)
  :
  students_per_class = (total_spent / cost_per_pack * cards_per_pack / cards_per_student / periods_per_day) :=
sorry

end students_per_class_l155_155619


namespace probability_two_girls_from_twelve_l155_155036

theorem probability_two_girls_from_twelve : 
  let total_members := 12
  let boys := 4
  let girls := 8
  let choose_two_total := Nat.choose total_members 2
  let choose_two_girls := Nat.choose girls 2
  let probability := (choose_two_girls : ℚ) / (choose_two_total : ℚ)
  probability = (14 / 33) := by
  -- Proof goes here
  sorry

end probability_two_girls_from_twelve_l155_155036


namespace total_students_in_class_l155_155371

theorem total_students_in_class (R S : ℕ) (h1 : 2 + 12 + 14 + R = S) (h2 : 2 * S = 40 + 3 * R) : S = 44 :=
by
  sorry

end total_students_in_class_l155_155371


namespace min_value_of_a_plus_2b_l155_155085

theorem min_value_of_a_plus_2b (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: 1 / (a + 1) + 1 / (b + 1) = 1) : 
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_value_of_a_plus_2b_l155_155085


namespace avg_price_of_pen_l155_155034

theorem avg_price_of_pen 
  (total_pens : ℕ) (total_pencils : ℕ) (total_cost : ℕ) 
  (avg_price_pencil : ℕ) (total_pens_cost : ℕ) (total_pencils_cost : ℕ)
  (total_cost_eq : total_cost = total_pens_cost + total_pencils_cost)
  (total_pencils_cost_eq : total_pencils_cost = total_pencils * avg_price_pencil)
  (pencils_count : total_pencils = 75) (pens_count : total_pens = 30) 
  (avg_price_pencil_eq : avg_price_pencil = 2)
  (total_cost_eq' : total_cost = 450) :
  total_pens_cost / total_pens = 10 :=
by
  sorry

end avg_price_of_pen_l155_155034


namespace trapezoid_area_l155_155735

theorem trapezoid_area (x : ℝ) :
  let base1 := 4 * x
  let base2 := 6 * x
  let height := x
  (base1 + base2) / 2 * height = 5 * x^2 :=
by
  sorry

end trapezoid_area_l155_155735


namespace statement_B_false_l155_155058

def f (x : ℝ) : ℝ := 3 * x

def diamondsuit (x y : ℝ) : ℝ := abs (f x - f y)

theorem statement_B_false (x y : ℝ) : 3 * diamondsuit x y ≠ diamondsuit (3 * x) (3 * y) :=
by
  sorry

end statement_B_false_l155_155058


namespace square_areas_l155_155254

theorem square_areas (z : ℂ) 
  (h1 : ¬ (2 : ℂ) * z^2 = z)
  (h2 : ¬ (3 : ℂ) * z^3 = z)
  (sz : (3 * z^3 - z) = (I * (2 * z^2 - z)) ∨ (3 * z^3 - z) = (-I * (2 * z^2 - z))) :
  ∃ (areas : Finset ℝ), areas = {85, 4500} :=
by {
  sorry
}

end square_areas_l155_155254


namespace solve_ff_eq_x_l155_155684

def f (x : ℝ) : ℝ := x^2 + 2 * x - 5

theorem solve_ff_eq_x :
  ∀ x : ℝ, f (f x) = x ↔ (x = ( -1 + Real.sqrt 21 ) / 2) ∨ (x = ( -1 - Real.sqrt 21 ) / 2) ∨
                          (x = ( -3 + Real.sqrt 17 ) / 2) ∨ (x = ( -3 - Real.sqrt 17 ) / 2) := 
by
  sorry

end solve_ff_eq_x_l155_155684


namespace pattern_proof_l155_155311

theorem pattern_proof (h1 : 1 = 6) (h2 : 2 = 36) (h3 : 3 = 363) (h4 : 4 = 364) (h5 : 5 = 365) : 36 = 3636 := by
  sorry

end pattern_proof_l155_155311


namespace spent_on_basil_seeds_l155_155481

-- Define the variables and conditions
variables (S cost_soil num_plants price_per_plant net_profit total_revenue total_expenses : ℝ)
variables (h1 : cost_soil = 8)
variables (h2 : num_plants = 20)
variables (h3 : price_per_plant = 5)
variables (h4 : net_profit = 90)

-- Definition of total revenue as the multiplication of number of plants and price per plant
def revenue_eq : Prop := total_revenue = num_plants * price_per_plant

-- Definition of total expenses as the sum of soil cost and cost of basil seeds
def expenses_eq : Prop := total_expenses = cost_soil + S

-- Definition of net profit
def profit_eq : Prop := net_profit = total_revenue - total_expenses

-- The theorem to prove
theorem spent_on_basil_seeds : S = 2 :=
by
  -- Since we define variables and conditions as inputs,
  -- the proof itself is omitted as per instructions
  sorry

end spent_on_basil_seeds_l155_155481


namespace smallest_number_property_l155_155714

theorem smallest_number_property : 
  ∃ n, ((n - 7) % 12 = 0) ∧ ((n - 7) % 16 = 0) ∧ ((n - 7) % 18 = 0) ∧ ((n - 7) % 21 = 0) ∧ ((n - 7) % 28 = 0) ∧ n = 1015 :=
by
  sorry  -- Proof is omitted

end smallest_number_property_l155_155714


namespace ratio_of_areas_l155_155180

noncomputable def large_square_side : ℝ := 4
noncomputable def large_square_area : ℝ := large_square_side ^ 2
noncomputable def inscribed_square_side : ℝ := 1  -- As it fits in the definition from the problem description
noncomputable def inscribed_square_area : ℝ := inscribed_square_side ^ 2

theorem ratio_of_areas :
  (inscribed_square_area / large_square_area) = 1 / 16 :=
by
  sorry

end ratio_of_areas_l155_155180


namespace barbara_total_cost_l155_155477

-- Define conditions
def steak_weight : ℝ := 4.5
def steak_price_per_pound : ℝ := 15.0
def chicken_weight : ℝ := 1.5
def chicken_price_per_pound : ℝ := 8.0

-- Define total cost formula
def total_cost := (steak_weight * steak_price_per_pound) + (chicken_weight * chicken_price_per_pound)

-- Prove that the total cost equals $79.50
theorem barbara_total_cost : total_cost = 79.50 := by
  sorry

end barbara_total_cost_l155_155477


namespace trigonometric_identity_proof_l155_155761

noncomputable def four_sin_40_minus_tan_40 : ℝ :=
  4 * Real.sin (40 * Real.pi / 180) - Real.tan (40 * Real.pi / 180)

theorem trigonometric_identity_proof : four_sin_40_minus_tan_40 = Real.sqrt 3 := by
  sorry

end trigonometric_identity_proof_l155_155761


namespace number_of_green_fish_and_carp_drawn_is_6_l155_155858

-- Definitions/parameters from the problem
def total_fish := 80 + 20 + 40 + 40 + 20
def sample_size := 20
def number_of_green_fish := 20
def number_of_carp := 40
def probability_of_being_drawn := sample_size / total_fish

-- Theorem to prove the combined number of green fish and carp drawn is 6
theorem number_of_green_fish_and_carp_drawn_is_6 :
  (number_of_green_fish + number_of_carp) * probability_of_being_drawn = 6 := 
by {
  -- Placeholder for the actual proof
  sorry
}

end number_of_green_fish_and_carp_drawn_is_6_l155_155858


namespace algebraic_expression_value_l155_155352

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
by
  sorry

end algebraic_expression_value_l155_155352


namespace perimeter_of_park_is_66_l155_155534

-- Given width and length of the flower bed
variables (w l : ℝ)
-- Given that the length is four times the width
variable (h1 : l = 4 * w)
-- Given the area of the flower bed
variable (h2 : l * w = 100)
-- Given the width of the walkway
variable (walkway_width : ℝ := 2)

-- The total width and length of the park, including the walkway
def w_park := w + 2 * walkway_width
def l_park := l + 2 * walkway_width

-- The proof statement: perimeter of the park equals 66 meters
theorem perimeter_of_park_is_66 :
  2 * (l_park + w_park) = 66 :=
by
  -- The full proof can be filled in here
  sorry

end perimeter_of_park_is_66_l155_155534


namespace gcd_abcd_dcba_l155_155140

-- Definitions based on the conditions
def abcd (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d
def dcba (a b c d : ℕ) : ℕ := 1000 * d + 100 * c + 10 * b + a
def consecutive_digits (a b c d : ℕ) : Prop := (b = a + 1) ∧ (c = a + 2) ∧ (d = a + 3)

-- Theorem statement
theorem gcd_abcd_dcba (a b c d : ℕ) (h : consecutive_digits a b c d) : 
  Nat.gcd (abcd a b c d + dcba a b c d) 1111 = 1111 :=
sorry

end gcd_abcd_dcba_l155_155140


namespace quadratic_has_real_root_of_b_interval_l155_155529

variable (b : ℝ)

theorem quadratic_has_real_root_of_b_interval
  (h : ∃ x : ℝ, x^2 + b * x + 25 = 0) : b ∈ Iic (-10) ∪ Ici 10 :=
by
  sorry

end quadratic_has_real_root_of_b_interval_l155_155529


namespace compute_f_g_f_l155_155119

def f (x : ℤ) : ℤ := 2 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

theorem compute_f_g_f (x : ℤ) : f (g (f 3)) = 108 := 
  by 
  sorry

end compute_f_g_f_l155_155119


namespace average_other_marbles_l155_155124

def total_marbles : ℕ := 10 -- Define a hypothetical total number for computation
def clear_marbles : ℕ := total_marbles * 40 / 100
def black_marbles : ℕ := total_marbles * 20 / 100
def other_marbles : ℕ := total_marbles - clear_marbles - black_marbles
def marbles_taken : ℕ := 5

theorem average_other_marbles :
  marbles_taken * other_marbles / total_marbles = 2 := by
  sorry

end average_other_marbles_l155_155124


namespace max_correct_questions_l155_155533

theorem max_correct_questions (a b c : ℕ) (h1 : a + b + c = 60) (h2 : 3 * a - 2 * c = 126) : a ≤ 49 :=
sorry

end max_correct_questions_l155_155533


namespace translate_A_coordinates_l155_155253

-- Definitions
def A_initial : ℝ × ℝ := (-3, 2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 - d)

-- Final coordinates after transformation
def A' : ℝ × ℝ :=
  let A_translated := translate_right A_initial 4
  translate_down A_translated 3

-- Proof statement
theorem translate_A_coordinates :
  A' = (1, -1) :=
by
  simp [A', translate_right, translate_down, A_initial]
  sorry

end translate_A_coordinates_l155_155253


namespace remainder_when_divide_by_66_l155_155409

-- Define the conditions as predicates
def condition_1 (n : ℕ) : Prop := ∃ l : ℕ, n % 22 = 7
def condition_2 (n : ℕ) : Prop := ∃ m : ℕ, n % 33 = 18

-- Define the main theorem
theorem remainder_when_divide_by_66 (n : ℕ) (h1 : condition_1 n) (h2 : condition_2 n) : n % 66 = 51 :=
  sorry

end remainder_when_divide_by_66_l155_155409


namespace probability_of_pink_gumball_l155_155444

theorem probability_of_pink_gumball 
  (P B : ℕ) 
  (total_gumballs : P + B > 0)
  (prob_blue_blue : ((B : ℚ) / (B + P))^2 = 16 / 49) : 
  (B + P > 0) → ((P : ℚ) / (B + P) = 3 / 7) :=
by
  sorry

end probability_of_pink_gumball_l155_155444


namespace arrangements_of_people_l155_155415

theorem arrangements_of_people : 
  let n := 5 in
  let total_arrangements := n.factorial in
  let A_left_end := (n - 1).factorial in
  let A_B_adjacent : ℕ := 2 * (n - 1).factorial in
  let A_left_end_B_adjacent := (n - 2).factorial in
  (total_arrangements - A_left_end - A_B_adjacent + A_left_end_B_adjacent) = 54 := sorry

end arrangements_of_people_l155_155415


namespace det_transformed_matrix_l155_155924

variables {p q r s : ℝ} -- Defining the variables over the real numbers

-- Defining the first determinant condition as an axiom
axiom det_initial_matrix : (p * s - q * r) = 10

-- Stating the theorem to be proved
theorem det_transformed_matrix : 
  (p + 2 * r) * s - (q + 2 * s) * r = 10 :=
by
  sorry -- Placeholder for the actual proof

end det_transformed_matrix_l155_155924


namespace Hillary_left_with_amount_l155_155472

theorem Hillary_left_with_amount :
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  remaining_amount = 25 :=
by
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  sorry

end Hillary_left_with_amount_l155_155472


namespace probability_of_one_pair_one_triplet_l155_155891

-- Define the necessary conditions
def six_sided_die_rolls (n : ℕ) : ℕ := 6 ^ n

def successful_outcomes : ℕ :=
  6 * 20 * 5 * 3 * 4

def total_outcomes : ℕ :=
  six_sided_die_rolls 6

def probability_success : ℚ :=
  successful_outcomes / total_outcomes

-- The theorem we want to prove
theorem probability_of_one_pair_one_triplet :
  probability_success = 25/162 :=
sorry

end probability_of_one_pair_one_triplet_l155_155891


namespace points_per_touchdown_l155_155658

theorem points_per_touchdown (P : ℕ) (games : ℕ) (touchdowns_per_game : ℕ) (two_point_conversions : ℕ) (two_point_conversion_value : ℕ) (total_points : ℕ) :
  touchdowns_per_game = 4 →
  games = 15 →
  two_point_conversions = 6 →
  two_point_conversion_value = 2 →
  total_points = (4 * P * 15 + 6 * two_point_conversion_value) →
  total_points = 372 →
  P = 6 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end points_per_touchdown_l155_155658


namespace total_distance_traveled_l155_155613

theorem total_distance_traveled :
  let day1_distance := 5 * 7
  let day2_distance_part1 := 6 * 6
  let day2_distance_part2 := 3 * 3
  let day3_distance := 7 * 5
  let total_distance := day1_distance + day2_distance_part1 + day2_distance_part2 + day3_distance
  total_distance = 115 :=
by
  sorry

end total_distance_traveled_l155_155613


namespace square_area_from_hexagon_l155_155120

theorem square_area_from_hexagon (hex_side length square_side : ℝ) (h1 : hex_side = 4) (h2 : length = 6 * hex_side)
  (h3 : square_side = length / 4) : square_side ^ 2 = 36 :=
by 
  sorry

end square_area_from_hexagon_l155_155120


namespace final_weight_is_sixteen_l155_155261

def initial_weight : ℤ := 0
def weight_after_jellybeans : ℤ := initial_weight + 2
def weight_after_brownies : ℤ := weight_after_jellybeans * 3
def weight_after_more_jellybeans : ℤ := weight_after_brownies + 2
def final_weight : ℤ := weight_after_more_jellybeans * 2

theorem final_weight_is_sixteen : final_weight = 16 := by
  sorry

end final_weight_is_sixteen_l155_155261


namespace train_pass_tree_l155_155593

theorem train_pass_tree
  (L : ℝ) (S : ℝ) (conv_factor : ℝ) 
  (hL : L = 275)
  (hS : S = 90)
  (hconv : conv_factor = 5 / 18) :
  L / (S * conv_factor) = 11 :=
by
  sorry

end train_pass_tree_l155_155593


namespace hexahedron_has_six_faces_l155_155239

-- Definition based on the condition
def is_hexahedron (P : Type) := 
  ∃ (f : P → ℕ), ∀ (x : P), f x = 6

-- Theorem statement based on the question and correct answer
theorem hexahedron_has_six_faces (P : Type) (h : is_hexahedron P) : 
  ∀ (x : P), ∃ (f : P → ℕ), f x = 6 :=
by 
  sorry

end hexahedron_has_six_faces_l155_155239


namespace determine_d_and_vertex_l155_155841

-- Definition of the quadratic equation
def g (x d : ℝ) : ℝ := 3 * x^2 + 12 * x + d

-- The proof problem
theorem determine_d_and_vertex (d : ℝ) :
  (∃ x : ℝ, g x d = 0 ∧ ∀ y : ℝ, g y d ≥ g x d) ↔ (d = 12 ∧ ∀ x : ℝ, 3 > 0 ∧ (g x d ≥ g 0 d)) := 
by 
  sorry

end determine_d_and_vertex_l155_155841


namespace distance_ratio_l155_155872

-- Define the distances as given in the conditions
def distance_from_city_sky_falls := 8 -- Distance in miles
def distance_from_city_rocky_mist := 400 -- Distance in miles

theorem distance_ratio : distance_from_city_rocky_mist / distance_from_city_sky_falls = 50 := 
by
  -- Proof skipped
  sorry

end distance_ratio_l155_155872


namespace chalk_pieces_l155_155724

theorem chalk_pieces (boxes: ℕ) (pieces_per_box: ℕ) (total_chalk: ℕ) 
  (hb: boxes = 194) (hp: pieces_per_box = 18) : 
  total_chalk = 194 * 18 :=
by 
  sorry

end chalk_pieces_l155_155724


namespace number_of_seeds_in_bucket_B_l155_155157

theorem number_of_seeds_in_bucket_B :
  ∃ (x : ℕ), 
    ∃ (y : ℕ), 
    ∃ (z : ℕ), 
      y = x + 10 ∧ 
      z = 30 ∧ 
      x + y + z = 100 ∧
      x = 30 :=
by {
  -- the proof is omitted.
  sorry
}

end number_of_seeds_in_bucket_B_l155_155157


namespace solution_set_lg2_l155_155226

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_deriv_lt : ∀ x : ℝ, deriv f x < 1

theorem solution_set_lg2 : { x : ℝ | f (Real.log x ^ 2) < Real.log x ^ 2 } = { x : ℝ | (1/10 : ℝ) < x ∧ x < 10 } :=
by
  sorry

end solution_set_lg2_l155_155226


namespace find_ab_sum_eq_42_l155_155636

noncomputable def find_value (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem find_ab_sum_eq_42 (a b : ℝ) (h1 : a + b = 6) (h2 : a * b = 7) : find_value a b = 42 := by
  sorry

end find_ab_sum_eq_42_l155_155636


namespace a_divisible_by_11_iff_b_divisible_by_11_l155_155445

-- Define the relevant functions
def a (n : ℕ) : ℕ := n^5 + 5^n
def b (n : ℕ) : ℕ := n^5 * 5^n + 1

-- State that for a positive integer n, a(n) is divisible by 11 if and only if b(n) is also divisible by 11
theorem a_divisible_by_11_iff_b_divisible_by_11 (n : ℕ) (hn : 0 < n) : 
  (a n % 11 = 0) ↔ (b n % 11 = 0) :=
sorry

end a_divisible_by_11_iff_b_divisible_by_11_l155_155445


namespace card_pair_probability_l155_155599

theorem card_pair_probability :
  let total_cards := 52
  let pair_removed_cards := total_cards - 2
  let remaining_cards := pair_removed_cards
  let choose_two : ℕ := remaining_cards.choose 2
  let total_ways := 12 * (4.choose 2) + 1 * (2.choose 2)
  let pair_probability := (total_ways : ℚ) / choose_two
  let m := 73
  let n := 1225
  m.gcd n = 1 ∧ pair_probability = (m : ℚ) / n ∧ m + n = 1298 := by
  sorry

end card_pair_probability_l155_155599


namespace gcd_subtraction_result_l155_155881

theorem gcd_subtraction_result : gcd 8100 270 - 8 = 262 := by
  sorry

end gcd_subtraction_result_l155_155881


namespace perimeter_triangle_ABC_l155_155572

-- Define the conditions and statement
theorem perimeter_triangle_ABC 
  (r : ℝ) (AP PB altitude : ℝ) 
  (h1 : r = 30) 
  (h2 : AP = 26) 
  (h3 : PB = 32) 
  (h4 : altitude = 96) :
  (2 * (58 + 34.8)) = 185.6 :=
by
  sorry

end perimeter_triangle_ABC_l155_155572


namespace traveled_distance_l155_155614

def distance_first_day : ℕ := 5 * 7
def distance_second_day_part1 : ℕ := 6 * 6
def distance_second_day_part2 : ℕ := (6 / 2) * 3
def distance_third_day : ℕ := 7 * 5

def total_distance : ℕ := distance_first_day + distance_second_day_part1 + distance_second_day_part2 + distance_third_day

theorem traveled_distance : total_distance = 115 := by
  unfold total_distance
  unfold distance_first_day distance_second_day_part1 distance_second_day_part2 distance_third_day
  norm_num
  rfl

end traveled_distance_l155_155614


namespace yuna_correct_multiplication_l155_155438

theorem yuna_correct_multiplication (x : ℕ) (h : 4 * x = 60) : 8 * x = 120 :=
by
  sorry

end yuna_correct_multiplication_l155_155438


namespace correct_calculation_l155_155364

def original_number (x : ℕ) : Prop := x + 12 = 48

theorem correct_calculation (x : ℕ) (h : original_number x) : x + 22 = 58 := by
  sorry

end correct_calculation_l155_155364


namespace mean_score_of_sophomores_l155_155466

open Nat

variable (s j : ℕ)
variable (m m_s m_j : ℝ)

theorem mean_score_of_sophomores :
  (s + j = 150) →
  (m = 85) →
  (j = 80 / 100 * s) →
  (m_s = 125 / 100 * m_j) →
  (s * m_s + j * m_j = 12750) →
  m_s = 94 := by intros; sorry

end mean_score_of_sophomores_l155_155466


namespace correct_calculation_l155_155589

variable (a : ℕ)

theorem correct_calculation : 
  ¬(a + a = a^2) ∧ ¬(a^3 * a = a^3) ∧ ¬(a^8 / a^2 = a^4) ∧ ((a^3)^2 = a^6) := 
by
  sorry

end correct_calculation_l155_155589


namespace exists_happy_configuration_l155_155943

noncomputable def all_children_happy (n: ℕ) (a: Fin n → ℕ) (xA xB xC: ℕ) : Prop :=
  let total_raisins := (n * (n + 1)) / 2
  let child_raises_at (x: ℕ) : ℕ := 
    (Finset.range n).sum (λ k => if xA = k ∨ xB = k ∨ xC = k then a k else 0)
  let unhappy (x: ℕ) (r: ℕ) : Prop :=
    ∃ y ∈ Finset.range n, y ≠ x ∧ child_raises_at y > r
  ∀ x ∈ {xA, xB, xC}, ¬ unhappy x (child_raises_at x)

theorem exists_happy_configuration : ∀ n: ℕ, (2 ≤ n ∧ n ≤ 8) ↔ ∃ (a: Fin n → ℕ) (xA xB xC: ℕ), all_children_happy n a xA xB xC := 
  by
    sorry

end exists_happy_configuration_l155_155943


namespace max_gcd_sequence_l155_155334

noncomputable def a (n : ℕ) : ℕ := n^3 + 4
noncomputable def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_sequence : (∀ n : ℕ, 0 < n → d n ≤ 433) ∧ (∃ n : ℕ, 0 < n ∧ d n = 433) :=
by sorry

end max_gcd_sequence_l155_155334


namespace johns_profit_l155_155435

noncomputable def earnings : ℕ := 30000
noncomputable def purchase_price : ℕ := 18000
noncomputable def trade_in_value : ℕ := 6000

noncomputable def depreciation : ℕ := purchase_price - trade_in_value
noncomputable def profit : ℕ := earnings - depreciation

theorem johns_profit : profit = 18000 := by
  sorry

end johns_profit_l155_155435


namespace fraction_irreducible_l155_155563

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_irreducible_l155_155563


namespace boundary_length_is_25_point_7_l155_155187

-- Define the side length derived from the given area.
noncomputable def sideLength (area : ℝ) : ℝ :=
  Real.sqrt area

-- Define the length of each segment when the square's side is divided into four equal parts.
noncomputable def segmentLength (side : ℝ) : ℝ :=
  side / 4

-- Define the total boundary length, which includes the circumference of the quarter-circle arcs and the straight segments.
noncomputable def totalBoundaryLength (area : ℝ) : ℝ :=
  let side := sideLength area
  let segment := segmentLength side
  let arcsLength := 2 * Real.pi * segment  -- the full circle's circumference
  let straightLength := 4 * segment
  arcsLength + straightLength

-- State the theorem that the total boundary length is approximately 25.7 units.
theorem boundary_length_is_25_point_7 :
  totalBoundaryLength 100 = 5 * Real.pi + 10 :=
by sorry

end boundary_length_is_25_point_7_l155_155187


namespace nat_exponent_sum_eq_l155_155631

theorem nat_exponent_sum_eq (n p q : ℕ) : n^p + n^q = n^2010 ↔ (n = 2 ∧ p = 2009 ∧ q = 2009) :=
by
  sorry

end nat_exponent_sum_eq_l155_155631


namespace nine_points_unit_square_l155_155147

theorem nine_points_unit_square :
  ∀ (points : List (ℝ × ℝ)), points.length = 9 → 
  (∀ (x : ℝ × ℝ), x ∈ points → 0 ≤ x.1 ∧ x.1 ≤ 1 ∧ 0 ≤ x.2 ∧ x.2 ≤ 1) → 
  ∃ (A B C : ℝ × ℝ), A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ 
  (1 / 8 : ℝ) ≤ abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 :=
by
  sorry

end nine_points_unit_square_l155_155147


namespace arthur_muffins_l155_155742

variable (arthur_baked : ℕ)
variable (james_baked : ℕ := 1380)
variable (times_as_many : ℕ := 12)

theorem arthur_muffins : arthur_baked * times_as_many = james_baked -> arthur_baked = 115 := by
  sorry

end arthur_muffins_l155_155742


namespace smallest_angle_of_isosceles_trapezoid_l155_155372

def is_isosceles_trapezoid (a b c d : ℝ) : Prop :=
  a = c ∧ b = d ∧ a + b + c + d = 360 ∧ a + 3 * b = 150

theorem smallest_angle_of_isosceles_trapezoid (a b : ℝ) (h1 : is_isosceles_trapezoid a b a (a + 2 * b))
  : a = 47 :=
sorry

end smallest_angle_of_isosceles_trapezoid_l155_155372


namespace inequality_proof_l155_155780

theorem inequality_proof
  (x y : ℝ) (h1 : x^2 + x * y + y^2 = (x + y)^2 - x * y) 
  (h2 : x + y ≥ 2 * Real.sqrt (x * y)) : 
  x + y + Real.sqrt (x * y) ≤ 3 * (x + y - Real.sqrt (x * y)) := 
by
  sorry

end inequality_proof_l155_155780


namespace highest_number_paper_l155_155246

theorem highest_number_paper (n : ℕ) (h : 1 / (n : ℝ) = 0.01020408163265306) : n = 98 :=
sorry

end highest_number_paper_l155_155246


namespace minimum_students_to_share_birthday_l155_155522

theorem minimum_students_to_share_birthday (k : ℕ) (m : ℕ) (n : ℕ) (hcond1 : k = 366) (hcond2 : m = 2) (hineq : n > k * m) : n ≥ 733 := 
by
  -- since k = 366 and m = 2
  have hk : k = 366 := hcond1
  have hm : m = 2 := hcond2
  -- thus: n > 366 * 2
  have hn : n > 732 := by
    rw [hk, hm] at hineq
    exact hineq
  -- hence, n ≥ 733
  exact Nat.succ_le_of_lt hn

end minimum_students_to_share_birthday_l155_155522


namespace remainder_of_65_power_65_plus_65_mod_97_l155_155171

theorem remainder_of_65_power_65_plus_65_mod_97 :
  (65^65 + 65) % 97 = 33 :=
by
  sorry

end remainder_of_65_power_65_plus_65_mod_97_l155_155171


namespace solve_for_square_l155_155350

theorem solve_for_square (x : ℤ) (s : ℤ) 
  (h1 : s + x = 80) 
  (h2 : 3 * (s + x) - 2 * x = 164) : 
  s = 42 :=
by 
  -- Include the implementation with sorry
  sorry

end solve_for_square_l155_155350


namespace fourth_vs_third_difference_l155_155375

def first_competitor_distance : ℕ := 22

def second_competitor_distance : ℕ := first_competitor_distance + 1

def third_competitor_distance : ℕ := second_competitor_distance - 2

def fourth_competitor_distance : ℕ := 24

theorem fourth_vs_third_difference : 
  fourth_competitor_distance - third_competitor_distance = 3 := by
  sorry

end fourth_vs_third_difference_l155_155375


namespace f_is_odd_l155_155088

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (α - 2) * x ^ α

theorem f_is_odd (α : ℝ) (hα : α = 3) : ∀ x : ℝ, f α (-x) = -f α x :=
by sorry

end f_is_odd_l155_155088


namespace count_8_digit_even_ending_l155_155521

theorem count_8_digit_even_ending : 
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  (choices_first_digit * choices_middle_digits * choices_last_digit) = 45000000 :=
by
  let choices_first_digit := 9
  let choices_middle_digits := 10 ^ 6
  let choices_last_digit := 5
  sorry

end count_8_digit_even_ending_l155_155521


namespace unique_tangent_lines_through_point_l155_155730

theorem unique_tangent_lines_through_point (P : ℝ × ℝ) (hP : P = (2, 4)) :
  ∃! l : ℝ × ℝ → Prop, (l P) ∧ (∀ p : ℝ × ℝ, l p → p ∈ {p : ℝ × ℝ | p.2 ^ 2 = 8 * p.1}) := sorry

end unique_tangent_lines_through_point_l155_155730


namespace arithmetic_mean_twice_y_l155_155638

theorem arithmetic_mean_twice_y (y x : ℝ) (h1 : (8 + y + 24 + 6 + x) / 5 = 12) (h2 : x = 2 * y) :
  y = 22 / 3 ∧ x = 44 / 3 :=
by
  sorry

end arithmetic_mean_twice_y_l155_155638


namespace janet_dresses_total_pockets_l155_155939

theorem janet_dresses_total_pockets :
  let dresses := 24
  let with_pockets := dresses / 2
  let with_two_pockets := with_pockets / 3
  let with_three_pockets := with_pockets - with_two_pockets
  let total_two_pockets := with_two_pockets * 2
  let total_three_pockets := with_three_pockets * 3
  total_two_pockets + total_three_pockets = 32 := by
  sorry

end janet_dresses_total_pockets_l155_155939


namespace trig_identity_proof_l155_155893

theorem trig_identity_proof
  (h1: Float.sin 50 = Float.cos 40)
  (h2: Float.tan 45 = 1)
  (h3: Float.tan 10 = Float.sin 10 / Float.cos 10)
  (h4: Float.sin 80 = Float.cos 10) :
  Float.sin 50 * (Float.tan 45 + Float.sqrt 3 * Float.tan 10) = 1 :=
by
  sorry

end trig_identity_proof_l155_155893


namespace find_value_of_a_l155_155686

variable (a b : ℝ)

def varies_inversely (a : ℝ) (b_minus_one_sq : ℝ) : ℝ :=
  a * b_minus_one_sq

theorem find_value_of_a 
  (h₁ : ∀ b : ℝ, varies_inversely a ((b - 1) ^ 2) = 64)
  (h₂ : b = 5) : a = 4 :=
by sorry

end find_value_of_a_l155_155686


namespace locker_count_proof_l155_155328

theorem locker_count_proof (cost_per_digit : ℕ := 3)
  (total_cost : ℚ := 224.91) :
  (N : ℕ) = 2151 :=
by
  sorry

end locker_count_proof_l155_155328


namespace alice_ride_top_speed_l155_155199

-- Define the conditions
variables (x y : Real) -- x is the hours at 25 mph, y is the hours at 15 mph.
def distance_eq : Prop := 25 * x + 15 * y + 10 * (9 - x - y) = 162
def time_eq : Prop := x + y ≤ 9

-- Define the final answer
def final_answer : Prop := x = 2.7

-- The statement to prove
theorem alice_ride_top_speed : distance_eq x y ∧ time_eq x y → final_answer x := sorry

end alice_ride_top_speed_l155_155199


namespace youseff_blocks_from_office_l155_155173

def blocks_to_office (x : ℕ) : Prop :=
  let walk_time := x  -- it takes x minutes to walk
  let bike_time := (20 * x) / 60  -- it takes (20 / 60) * x = (1 / 3) * x minutes to ride a bike
  walk_time = bike_time + 4  -- walking takes 4 more minutes than biking

theorem youseff_blocks_from_office (x : ℕ) (h : blocks_to_office x) : x = 6 :=
  sorry

end youseff_blocks_from_office_l155_155173


namespace isosceles_triangle_area_l155_155214

theorem isosceles_triangle_area (a b h : ℝ) (h_eq : h = a / (2 * Real.sqrt 3)) :
  (1 / 2 * a * h) = (a^2 * Real.sqrt 3) / 12 :=
by
  -- Define the necessary parameters and conditions
  let area := (1 / 2) * a * h
  have h := h_eq
  -- Substitute and prove the calculated area
  sorry

end isosceles_triangle_area_l155_155214


namespace half_abs_diff_squares_l155_155163

/-- Half of the absolute value of the difference of the squares of 23 and 19 is 84. -/
theorem half_abs_diff_squares : (1 / 2 : ℝ) * |(23^2 : ℝ) - (19^2 : ℝ)| = 84 :=
by
  sorry

end half_abs_diff_squares_l155_155163


namespace average_age_before_new_students_joined_l155_155402

/-
Problem: Given that the original strength of the class was 18, 
18 new students with an average age of 32 years joined the class, 
and the average age decreased by 4 years, prove that 
the average age of the class before the new students joined was 40 years.
-/

def original_strength := 18
def new_students := 18
def average_age_new_students := 32
def decrease_in_average_age := 4
def original_average_age := 40

theorem average_age_before_new_students_joined :
  (original_strength * original_average_age + new_students * average_age_new_students) / (original_strength + new_students) = original_average_age - decrease_in_average_age :=
by
  sorry

end average_age_before_new_students_joined_l155_155402


namespace find_n_tangent_eq_1234_l155_155632

theorem find_n_tangent_eq_1234 (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : Real.tan (n * Real.pi / 180) = Real.tan (1234 * Real.pi / 180)) : n = -26 := 
by 
  sorry

end find_n_tangent_eq_1234_l155_155632


namespace zack_initial_marbles_l155_155847

theorem zack_initial_marbles :
  ∃ M : ℕ, (∃ k : ℕ, M = 3 * k + 5) ∧ (M - 5 - 60 = 5) ∧ M = 70 := by
sorry

end zack_initial_marbles_l155_155847


namespace min_value_fraction_sum_l155_155772

theorem min_value_fraction_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  ∃ (x y : ℝ), x = 2/5 ∧ y = 3/5 ∧ (∃ (k : ℝ), k = 4/x + 9/y ∧ k = 25) :=
by
  sorry

end min_value_fraction_sum_l155_155772


namespace minimum_effort_to_qualify_l155_155739

def minimum_effort_to_qualify_for_mop (AMC_points_per_effort : ℕ := 6 * 1/3)
                                       (AIME_points_per_effort : ℕ := 10 * 1/7)
                                       (USAMO_points_per_effort : ℕ := 1 * 1/10)
                                       (required_amc_aime_points : ℕ := 200)
                                       (required_usamo_points : ℕ := 21) : ℕ :=
  let max_amc_points : ℕ := 150
  let effort_amc : ℕ := (max_amc_points / AMC_points_per_effort) * 3
  let remaining_aime_points : ℕ := 200 - max_amc_points
  let effort_aime : ℕ := (remaining_aime_points / AIME_points_per_effort) * 7
  let effort_usamo : ℕ := required_usamo_points * 10
  let total_effort : ℕ := effort_amc + effort_aime + effort_usamo
  total_effort

theorem minimum_effort_to_qualify : minimum_effort_to_qualify_for_mop 6 (10 * 1/7) (1 * 1/10) 200 21 = 320 := by
  sorry

end minimum_effort_to_qualify_l155_155739


namespace max_type_A_stationery_l155_155485

-- Define the variables and constraints
variables (x y : ℕ)

-- Define the conditions as hypotheses
def condition1 : Prop := 3 * x + 2 * (x - 2) + y = 66
def condition2 : Prop := 3 * x ≤ 33

-- The statement to prove
theorem max_type_A_stationery : condition1 x y ∧ condition2 x → x ≤ 11 :=
by sorry

end max_type_A_stationery_l155_155485


namespace find_k_l155_155211

-- Define the problem's conditions and constants
variables (S x y : ℝ)

-- Define the main theorem to prove k = 8 given the conditions
theorem find_k (h1 : 0.75 * x + ((S - 0.75 * x) * x) / (x + y) - (S * x) / (x + y) = 18) :
  (x * y / 3) / (x + y) = 8 := by 
  sorry

end find_k_l155_155211


namespace geom_seq_find_b3_l155_155541

-- Given conditions
def is_geometric_seq (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

def geom_seq_condition (b : ℕ → ℝ) : Prop :=
  is_geometric_seq b ∧ b 2 * b 3 * b 4 = 8

-- Proof statement: We need to prove that b 3 = 2
theorem geom_seq_find_b3 (b : ℕ → ℝ) (h : geom_seq_condition b) : b 3 = 2 :=
  sorry

end geom_seq_find_b3_l155_155541


namespace max_height_reached_l155_155598

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_reached : ∃ t : ℝ, h t = 161 :=
by
  sorry

end max_height_reached_l155_155598


namespace gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l155_155637

theorem gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4 (k : Int) :
  Int.gcd ((360 * k)^2 + 6 * (360 * k) + 8) (360 * k + 4) = 4 := 
sorry

end gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l155_155637


namespace paint_cans_for_25_rooms_l155_155953

theorem paint_cans_for_25_rooms (cans rooms : ℕ) (H1 : cans * 30 = rooms) (H2 : cans * 25 = rooms - 5 * cans) :
  cans = 15 :=
by
  sorry

end paint_cans_for_25_rooms_l155_155953


namespace Mr_Blue_potato_yield_l155_155553

-- Definitions based on the conditions
def steps_length (steps : ℕ) : ℕ := steps * 3
def garden_length : ℕ := steps_length 18
def garden_width : ℕ := steps_length 25

def area_garden : ℕ := garden_length * garden_width
def yield_potatoes (area : ℕ) : ℚ := area * (3/4)

-- Statement of the proof
theorem Mr_Blue_potato_yield :
  yield_potatoes area_garden = 3037.5 := by
  sorry

end Mr_Blue_potato_yield_l155_155553


namespace distance_from_P_to_AD_l155_155276

-- Definitions of points and circles
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 0, y := 4}
def D : Point := {x := 0, y := 0}
def C : Point := {x := 4, y := 0}
def M : Point := {x := 2, y := 0}
def radiusM : ℝ := 2
def radiusA : ℝ := 4

-- Definition of the circles
def circleM (P : Point) : Prop := (P.x - M.x)^2 + P.y^2 = radiusM^2
def circleA (P : Point) : Prop := P.x^2 + (P.y - A.y)^2 = radiusA^2

-- Definition of intersection point \(P\) of the two circles
def is_intersection (P : Point) : Prop := circleM P ∧ circleA P

-- Distance from point \(P\) to line \(\overline{AD}\) computed as the x-coordinate
def distance_to_line_AD (P : Point) : ℝ := P.x

-- The theorem to prove
theorem distance_from_P_to_AD :
  ∃ P : Point, is_intersection P ∧ distance_to_line_AD P = 16/5 :=
by {
  -- Use "sorry" as the proof placeholder
  sorry
}

end distance_from_P_to_AD_l155_155276


namespace next_term_geometric_sequence_l155_155978

theorem next_term_geometric_sequence (y : ℝ) (h0 : y ≠ 0) :
  let r := 3 * y in
  let term := 81 * y^3 in
  term * r = 243 * y^4 :=
by
  let r := 3 * y
  let term := 81 * y^3
  have h : term * r = 243 * y^4 := sorry
  exact h

end next_term_geometric_sequence_l155_155978


namespace parabola_vertex_expression_l155_155913

theorem parabola_vertex_expression (h k : ℝ) :
  (h = 2 ∧ k = 3) →
  ∃ (a : ℝ), (a ≠ 0) ∧
    (∀ x y : ℝ, y = a * (x - h)^2 + k ↔ y = -(x - 2)^2 + 3) :=
by
  sorry

end parabola_vertex_expression_l155_155913


namespace inequality_holds_for_all_y_l155_155899

theorem inequality_holds_for_all_y (x : ℝ) :
  (∀ y : ℝ, y^2 - (5^x - 1) * (y - 1) > 0) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end inequality_holds_for_all_y_l155_155899


namespace find_three_leaf_clovers_l155_155606

-- Define the conditions
def total_leaves : Nat := 1000

-- Define the statement
theorem find_three_leaf_clovers (n : Nat) (h : 3 * n + 4 = total_leaves) : n = 332 :=
  sorry

end find_three_leaf_clovers_l155_155606


namespace exactly_three_correct_is_impossible_l155_155703

theorem exactly_three_correct_is_impossible (n : ℕ) (hn : n = 5) (f : Fin n → Fin n) :
  (∃ S : Finset (Fin n), S.card = 3 ∧ ∀ i ∈ S, f i = i) → False :=
by
  intros h
  sorry

end exactly_three_correct_is_impossible_l155_155703


namespace max_common_initial_segment_l155_155695

theorem max_common_initial_segment (m n : ℕ) (h_coprime : Nat.gcd m n = 1) : 
  ∃ L, L = m + n - 2 := 
sorry

end max_common_initial_segment_l155_155695


namespace total_cookies_l155_155854

-- Define the number of bags and the number of cookies per bag
def bags : ℕ := 37
def cookies_per_bag : ℕ := 19

-- State the theorem
theorem total_cookies : bags * cookies_per_bag = 703 :=
by
  sorry

end total_cookies_l155_155854


namespace product_of_repeating145_and_11_equals_1595_over_999_l155_155750

-- Defining the repeating decimal as a fraction
def repeating145_as_fraction : ℚ :=
  145 / 999

-- Stating the main theorem
theorem product_of_repeating145_and_11_equals_1595_over_999 :
  11 * repeating145_as_fraction = 1595 / 999 :=
by
  sorry

end product_of_repeating145_and_11_equals_1595_over_999_l155_155750


namespace simplify_expression_l155_155681

theorem simplify_expression : 2023^2 - 2022 * 2024 = 1 := by
  sorry

end simplify_expression_l155_155681


namespace sale_discount_l155_155160

theorem sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (discount_multiple : ℕ)
  (h1 : purchase_amount = 250)
  (h2 : discount_per_100 = 10)
  (h3 : discount_multiple = purchase_amount / 100) :
  purchase_amount - discount_per_100 * discount_multiple = 230 := 
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end sale_discount_l155_155160


namespace value_of_f_at_5_l155_155946

theorem value_of_f_at_5 (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = - f x) 
  (h_period : ∀ x, f (x + 4) = f x)
  (h_func : ∀ x, -2 ≤ x ∧ x < 0 → f x = 3 * x + 1) : 
  f 5 = 2 :=
  sorry

end value_of_f_at_5_l155_155946


namespace truck_distance_on_7_gallons_l155_155470

theorem truck_distance_on_7_gallons :
  ∀ (d : ℝ) (g₁ g₂ : ℝ), d = 240 → g₁ = 5 → g₂ = 7 → (d / g₁) * g₂ = 336 :=
by
  intros d g₁ g₂ h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end truck_distance_on_7_gallons_l155_155470


namespace optimal_solution_for_z_is_1_1_l155_155264

def x := 1
def y := 1
def z (x y : ℝ) := 2 * x + y

theorem optimal_solution_for_z_is_1_1 :
  ∀ (x y : ℝ), z x y ≥ z 1 1 := 
by
  simp [z]
  sorry

end optimal_solution_for_z_is_1_1_l155_155264


namespace wings_per_person_l155_155729

-- Define the number of friends
def number_of_friends : ℕ := 15

-- Define the number of wings already cooked
def wings_already_cooked : ℕ := 7

-- Define the number of additional wings cooked
def additional_wings_cooked : ℕ := 45

-- Define the number of friends who don't eat chicken
def friends_not_eating : ℕ := 2

-- Calculate the total number of chicken wings
def total_chicken_wings : ℕ := wings_already_cooked + additional_wings_cooked

-- Calculate the number of friends who will eat chicken
def friends_eating : ℕ := number_of_friends - friends_not_eating

-- Define the statement we want to prove
theorem wings_per_person : total_chicken_wings / friends_eating = 4 := by
  sorry

end wings_per_person_l155_155729


namespace solve_for_y_l155_155009

theorem solve_for_y : ∃ y : ℝ, y = -2 ∧ y^2 + 6 * y + 8 = -(y + 2) * (y + 6) :=
by
  use -2
  sorry

end solve_for_y_l155_155009


namespace total_pictures_on_wall_l155_155001

theorem total_pictures_on_wall (oil_paintings watercolor_paintings : ℕ) (h1 : oil_paintings = 9) (h2 : watercolor_paintings = 7) :
  oil_paintings + watercolor_paintings = 16 := 
by
  sorry

end total_pictures_on_wall_l155_155001


namespace average_other_color_marbles_l155_155126

def percentage_clear : ℝ := 0.4
def percentage_black : ℝ := 0.2
def total_percentage : ℝ := 1.0
def total_marbles_taken : ℝ := 5.0

theorem average_other_color_marbles :
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black in
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors in
  expected_other_color_marbles = 2 := by
  let percentage_other_colors := total_percentage - percentage_clear - percentage_black
  let expected_other_color_marbles := total_marbles_taken * percentage_other_colors
  show expected_other_color_marbles = 2
  sorry

end average_other_color_marbles_l155_155126


namespace smallest_class_size_l155_155537

theorem smallest_class_size :
  ∀ (x : ℕ), 4 * x + 3 > 50 → 4 * x + 3 = 51 :=
by
  sorry

end smallest_class_size_l155_155537


namespace minimum_value_of_alpha_beta_l155_155349

open Polynomial

noncomputable def min_value_alpha_beta (m : ℝ) (h : m^2 ≥ 1) : ℝ :=
let α := root_of_quadratic (-2 * m) (2 - m^2) in
let β := root_of_quadratic_disjoint α (-2 * m) (2 - m^2) in
(α^2 + β^2)

theorem minimum_value_of_alpha_beta {α β m : ℝ} (h : m^2 ≥ 1) :
  (α^2 + β^2 = 2) :=
by
  have h_eq : (α^2 + β^2 = 6 * m^2 - 4), {
    sorry -- Proof showing how this expression is derived
  },
  have h_min : 6 * m^2 - 4 ≥ 2, {
    sorry -- Proof showing how minimum is derived
  },
  exact h_min

end minimum_value_of_alpha_beta_l155_155349


namespace village_current_population_l155_155932

def initial_population : ℕ := 4675
def died_by_bombardment : ℕ := (5*initial_population + 99) / 100 -- Equivalent to rounding (5/100) * 4675
def remaining_after_bombardment : ℕ := initial_population - died_by_bombardment
def left_due_to_fear : ℕ := (20*remaining_after_bombardment + 99) / 100 -- Equivalent to rounding (20/100) * remaining
def current_population : ℕ := remaining_after_bombardment - left_due_to_fear

theorem village_current_population : current_population = 3553 := by
  sorry

end village_current_population_l155_155932


namespace ratio_of_Victoria_to_Beacon_l155_155830

def Richmond_population : ℕ := 3000
def Beacon_population : ℕ := 500
def Victoria_population : ℕ := Richmond_population - 1000
def ratio_Victoria_Beacon : ℕ := Victoria_population / Beacon_population

theorem ratio_of_Victoria_to_Beacon : ratio_Victoria_Beacon = 4 := 
by
  unfold ratio_Victoria_Beacon Victoria_population Richmond_population Beacon_population
  sorry

end ratio_of_Victoria_to_Beacon_l155_155830


namespace anthony_pets_ratio_l155_155051

variable (C D : ℕ)

theorem anthony_pets_ratio
  (h1 : C + D = 12)
  (h2 : (C / 2 : ℕ) + (D + 7) + (C + D) = 27) :
  C / (C + D) = 2 / 3 :=
by
  sorry

end anthony_pets_ratio_l155_155051


namespace candles_left_in_room_l155_155607

-- Define the variables and conditions
def total_candles : ℕ := 40
def alyssa_used : ℕ := total_candles / 2
def remaining_candles_after_alyssa : ℕ := total_candles - alyssa_used
def chelsea_used : ℕ := (7 * remaining_candles_after_alyssa) / 10
def final_remaining_candles : ℕ := remaining_candles_after_alyssa - chelsea_used

-- The theorem we need to prove
theorem candles_left_in_room : final_remaining_candles = 6 := by
  sorry

end candles_left_in_room_l155_155607


namespace train_length_l155_155322

namespace TrainProblem

def speed_kmh : ℤ := 60
def time_sec : ℤ := 18
def speed_ms : ℚ := (speed_kmh : ℚ) * (1000 / 1) * (1 / 3600)
def length_meter := speed_ms * (time_sec : ℚ)

theorem train_length :
  length_meter = 300.06 := by
  sorry

end TrainProblem

end train_length_l155_155322


namespace tree_break_height_l155_155611

-- Define the problem conditions and prove the required height h
theorem tree_break_height (height_tree : ℝ) (distance_shore : ℝ) (height_break : ℝ) : 
  height_tree = 20 → distance_shore = 6 → 
  (distance_shore ^ 2 + height_break ^ 2 = (height_tree - height_break) ^ 2) →
  height_break = 9.1 :=
by
  intros h_tree_eq h_shore_eq hyp_eq
  have h_tree_20 := h_tree_eq
  have h_shore_6 := h_shore_eq
  have hyp := hyp_eq
  sorry -- Proof of the theorem is omitted

end tree_break_height_l155_155611


namespace candy_distribution_problem_l155_155846

theorem candy_distribution_problem (n : ℕ) :
  (n - 1) * (n - 2) / 2 - 3 * (n/2 - 1) / 6 = n + 1 → n = 18 :=
sorry

end candy_distribution_problem_l155_155846


namespace ratio_areas_ACEF_ADC_l155_155108

-- Define the basic geometric setup
variables (A B C D E F : Point) 
variables (BC CD DE : ℝ) 
variable (α : ℝ)
variables (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) 

-- Assuming the given conditions, we want to prove the ratio of areas
noncomputable def ratio_areas (α : ℝ) : ℝ := 4 * (1 - α)

theorem ratio_areas_ACEF_ADC (h1 : 0 < α) (h2 : α < 1/2) (h3 : CD = DE) (h4 : CD = α * BC) :
  ratio_areas α = 4 * (1 - α) :=
sorry

end ratio_areas_ACEF_ADC_l155_155108


namespace sequence_general_term_l155_155833

/-- The general term formula for the sequence 0.3, 0.33, 0.333, 0.3333, … is (1 / 3) * (1 - 1 / 10 ^ n). -/
theorem sequence_general_term (n : ℕ) : 
  (∃ a : ℕ → ℚ, (∀ n, a n = 0.3 + 0.03 * (10 ^ (n + 1) - 1) / 10 ^ (n + 1))) ↔
  ∀ n, (0.3 + 0.03 * (10 ^ (n + 1) - 1) / 10 ^ (n + 1)) = (1 / 3) * (1 - 1 / 10 ^ n) :=
sorry

end sequence_general_term_l155_155833


namespace cost_of_fencing_l155_155441

theorem cost_of_fencing (d : ℝ) (rate : ℝ) (C : ℝ) (cost : ℝ) : 
  d = 22 → rate = 3 → C = Real.pi * d → cost = C * rate → cost = 207 :=
by
  intros
  sorry

end cost_of_fencing_l155_155441


namespace unique_solution_pairs_count_l155_155361

theorem unique_solution_pairs_count :
  ∃! (p : ℝ × ℝ), (p.1 + 2 * p.2 = 2 ∧ (|abs p.1 - 2 * abs p.2| = 2) ∧
       ∃! q, (q = (2, 0) ∨ q = (0, 1)) ∧ p = q) := 
sorry

end unique_solution_pairs_count_l155_155361


namespace solve_trigonometric_equation_l155_155137

theorem solve_trigonometric_equation (x : ℝ) : 
  (2 * (Real.sin x)^6 + 2 * (Real.cos x)^6 - 3 * (Real.sin x)^4 - 3 * (Real.cos x)^4) = Real.cos (2 * x) ↔ 
  ∃ (k : ℤ), x = (π / 2) * (2 * k + 1) :=
sorry

end solve_trigonometric_equation_l155_155137


namespace eval_polynomial_at_3_l155_155892

theorem eval_polynomial_at_3 : (3 : ℤ) ^ 3 + (3 : ℤ) ^ 2 + 3 + 1 = 40 := by
  sorry

end eval_polynomial_at_3_l155_155892


namespace ratio_of_sequence_l155_155582

variables (a b c : ℝ)

-- Condition 1: arithmetic sequence
def arithmetic_sequence : Prop := 2 * b = a + c

-- Condition 2: geometric sequence
def geometric_sequence : Prop := c^2 = a * b

-- Theorem stating the ratio of a:b:c
theorem ratio_of_sequence (h1 : arithmetic_sequence a b c) (h2 : geometric_sequence a b c) : 
  (a = 4 * b) ∧ (c = -2 * b) :=
sorry

end ratio_of_sequence_l155_155582


namespace next_term_geometric_sequence_l155_155980

noncomputable def geometric_term (a r : ℕ) (n : ℕ) : ℕ :=
a * r^n

theorem next_term_geometric_sequence (y : ℕ) :
  ∀ a₁ a₂ a₃ a₄, a₁ = 3 → a₂ = 9 * y → a₃ = 27 * y^2 → a₄ = 81 * y^3 →
  geometric_term 3 (3 * y) 4 = 243 * y^4 :=
by
  intros a₁ a₂ a₃ a₄ h₁ h₂ h₃ h₄
  sorry

end next_term_geometric_sequence_l155_155980


namespace find_n_value_l155_155217

theorem find_n_value : ∃ n : ℤ, 3^3 - 7 = 4^2 + n ∧ n = 4 :=
by
  use 4
  sorry

end find_n_value_l155_155217


namespace matthew_and_zac_strawberries_l155_155941

theorem matthew_and_zac_strawberries (total_strawberries jonathan_and_matthew_strawberries zac_strawberries : ℕ) (h1 : total_strawberries = 550) (h2 : jonathan_and_matthew_strawberries = 350) (h3 : zac_strawberries = 200) : (total_strawberries - (jonathan_and_matthew_strawberries - zac_strawberries) = 400) :=
by { sorry }

end matthew_and_zac_strawberries_l155_155941


namespace perimeter_of_T_shaped_figure_l155_155836

theorem perimeter_of_T_shaped_figure :
  let a := 3    -- width of the horizontal rectangle
  let b := 5    -- height of the horizontal rectangle
  let c := 2    -- width of the vertical rectangle
  let d := 4    -- height of the vertical rectangle
  let overlap := 1 -- overlap length
  2 * a + 2 * b + 2 * c + 2 * d - 2 * overlap = 26 := by
  sorry

end perimeter_of_T_shaped_figure_l155_155836


namespace g_x_even_l155_155387

theorem g_x_even (a b c : ℝ) (g : ℝ → ℝ):
  (∀ x, g x = a * x^6 + b * x^4 - c * x^2 + 5)
  → g 32 = 3
  → g 32 + g (-32) = 6 :=
by
  sorry

end g_x_even_l155_155387


namespace three_digit_integers_with_two_identical_digits_less_than_700_l155_155645

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def less_than_700 (n : ℕ) : Prop :=
  n < 700

def has_at_least_two_identical_digits (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  digits.nodup = false

theorem three_digit_integers_with_two_identical_digits_less_than_700 : 
  ∃ (s : Finset ℕ), 
  (∀ n ∈ s, is_three_digit n ∧ less_than_700 n ∧ has_at_least_two_identical_digits n) ∧
  s.card = 156 := by
  sorry

end three_digit_integers_with_two_identical_digits_less_than_700_l155_155645


namespace one_third_of_1206_is_300_percent_of_134_l155_155123

theorem one_third_of_1206_is_300_percent_of_134 :
  let number := 1206
  let fraction := 1 / 3
  let computed_one_third := fraction * number
  let whole := 134
  let expected_percent := 300
  let percent := (computed_one_third / whole) * 100
  percent = expected_percent := by
  let number := 1206
  let fraction := 1 / 3
  have computed_one_third : ℝ := fraction * number
  let whole := 134
  let expected_percent := 300
  have percent : ℝ := (computed_one_third / whole) * 100
  exact sorry

end one_third_of_1206_is_300_percent_of_134_l155_155123


namespace fraction_always_irreducible_l155_155561

-- Define the problem statement
theorem fraction_always_irreducible (n : ℤ) : gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_always_irreducible_l155_155561


namespace older_brother_catches_up_in_half_hour_l155_155871

-- Defining the parameters according to the conditions
def speed_younger_brother := 4 -- kilometers per hour
def speed_older_brother := 20 -- kilometers per hour
def initial_distance := 8 -- kilometers

-- Calculate the relative speed difference
def speed_difference := speed_older_brother - speed_younger_brother

theorem older_brother_catches_up_in_half_hour:
  ∃ t : ℝ, initial_distance = speed_difference * t ∧ t = 0.5 := by
  use 0.5
  sorry

end older_brother_catches_up_in_half_hour_l155_155871


namespace part1_part2_l155_155385

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Part (1) 
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ a) → -2 ≤ a ∧ a ≤ 1 := by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → f a x ≥ a) → -3 ≤ a ∧ a ≤ 1 := by
  sorry

end part1_part2_l155_155385


namespace total_votes_l155_155851

-- Define the conditions
variable (V : ℝ) -- total number of votes polled
variable (w : ℝ) -- votes won by the winning candidate
variable (l : ℝ) -- votes won by the losing candidate
variable (majority : ℝ) -- majority votes

-- Define the specific values for the problem
def candidate_win_percentage (V : ℝ) : ℝ := 0.70 * V
def candidate_lose_percentage (V : ℝ) : ℝ := 0.30 * V

-- Define the majority condition
def majority_condition (V : ℝ) : Prop := (candidate_win_percentage V - candidate_lose_percentage V) = 240

-- The proof statement
theorem total_votes (V : ℝ) (h : majority_condition V) : V = 600 := by
  sorry

end total_votes_l155_155851


namespace smallest_fraction_of_land_l155_155693

noncomputable def smallest_share (n : ℕ) : ℚ :=
  if n = 150 then 1 / (2 * 3^49) else 0

theorem smallest_fraction_of_land :
  smallest_share 150 = 1 / (2 * 3^49) :=
sorry

end smallest_fraction_of_land_l155_155693


namespace geometric_progression_term_count_l155_155571

theorem geometric_progression_term_count
  (q : ℝ) (b4 : ℝ) (S : ℝ) (b1 : ℝ)
  (h1 : q = 1 / 3)
  (h2 : b4 = b1 * (q ^ 3))
  (h3 : S = b1 * (1 - q ^ 5) / (1 - q))
  (h4 : b4 = 1 / 54)
  (h5 : S = 121 / 162) :
  5 = 5 := sorry

end geometric_progression_term_count_l155_155571


namespace min_students_same_place_l155_155883

-- Define the context of the problem
def classSize := 45
def numberOfChoices := 6

-- The proof statement
theorem min_students_same_place : 
  ∃ (n : ℕ), 8 ≤ n ∧ n = Nat.ceil (classSize / numberOfChoices) :=
by
  sorry

end min_students_same_place_l155_155883


namespace fraction_exp_3_4_cubed_l155_155482

def fraction_exp (a b n : ℕ) : ℚ := (a : ℚ) ^ n / (b : ℚ) ^ n

theorem fraction_exp_3_4_cubed : fraction_exp 3 4 3 = 27 / 64 :=
by
  sorry

end fraction_exp_3_4_cubed_l155_155482


namespace expand_and_simplify_l155_155212

theorem expand_and_simplify (x : ℝ) : 
  -2 * (4 * x^3 - 5 * x^2 + 3 * x - 7) = -8 * x^3 + 10 * x^2 - 6 * x + 14 :=
sorry

end expand_and_simplify_l155_155212


namespace exists_infinitely_many_pairs_of_finite_sets_l155_155390

noncomputable def x (n : ℕ) : ℕ := Nat.choose (2 * n) n

theorem exists_infinitely_many_pairs_of_finite_sets :
  ∃ (A B : Finset ℕ), A ∩ B = ∅ ∧ (A ∪ B).Nonempty ∧
  (∏ j in A, x j) / (∏ j in B, x j) = 2012 :=
sorry

end exists_infinitely_many_pairs_of_finite_sets_l155_155390


namespace length_less_than_twice_width_l155_155573

def length : ℝ := 24
def width : ℝ := 13.5

theorem length_less_than_twice_width : 2 * width - length = 3 := by
  sorry

end length_less_than_twice_width_l155_155573


namespace binomial_probability_4_l155_155220

noncomputable def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ := 
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem binomial_probability_4 (n : ℕ) (p : ℝ) (ξ : ℕ → ℝ)
  (H1 : (ξ 0) = (n*p))
  (H2 : (ξ 1) = (n*p*(1-p))) :
  binomial_pmf n 4 p = 10 / 243 :=
by {
  sorry 
}

end binomial_probability_4_l155_155220


namespace range_of_m_l155_155903

noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then 3 + 3 * x
  else if x <= 3 then -1
  else x + 5

theorem range_of_m (m : ℝ) (x : ℝ) (hx : f x ≥ 1 / m - 4) :
  m < 0 ∨ m = 1 :=
sorry

end range_of_m_l155_155903


namespace maria_fraction_of_remaining_distance_l155_155890

theorem maria_fraction_of_remaining_distance (total_distance remaining_distance distance_travelled : ℕ) 
(h_total : total_distance = 480) 
(h_first_stop : distance_travelled = total_distance / 2) 
(h_remaining : remaining_distance = total_distance - distance_travelled)
(h_final_leg : remaining_distance - distance_travelled = 180) : 
(distance_travelled / remaining_distance) = (1 / 4) := 
by
  sorry

end maria_fraction_of_remaining_distance_l155_155890


namespace fraction_irreducible_l155_155562

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_irreducible_l155_155562


namespace expressions_equal_iff_sum_zero_l155_155057

theorem expressions_equal_iff_sum_zero (p q r : ℝ) : (p + qr = (p + q) * (p + r)) ↔ (p + q + r = 0) :=
sorry

end expressions_equal_iff_sum_zero_l155_155057


namespace number_of_pairs_of_socks_l155_155416

theorem number_of_pairs_of_socks (n : ℕ) (h : 2 * n^2 - n = 112) : n = 16 := sorry

end number_of_pairs_of_socks_l155_155416


namespace smallest_number_of_cubes_l155_155451

theorem smallest_number_of_cubes (l w d : ℕ) (hl : l = 36) (hw : w = 45) (hd : d = 18) : 
  ∃ n : ℕ, n = 40 ∧ (∃ s : ℕ, l % s = 0 ∧ w % s = 0 ∧ d % s = 0 ∧ (l / s) * (w / s) * (d / s) = n) := 
by
  sorry

end smallest_number_of_cubes_l155_155451


namespace product_of_last_two_digits_l155_155367

theorem product_of_last_two_digits (n A B : ℤ) 
  (h1 : n % 8 = 0) 
  (h2 : 10 * A + B = n % 100) 
  (h3 : A + B = 14) : 
  A * B = 48 := 
sorry

end product_of_last_two_digits_l155_155367


namespace james_drove_75_miles_l155_155380

noncomputable def james_total_distance : ℝ :=
  let speed1 := 30  -- mph
  let time1 := 0.5  -- hours
  let speed2 := 2 * speed1
  let time2 := 2 * time1
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  distance1 + distance2

theorem james_drove_75_miles : james_total_distance = 75 := by 
  sorry

end james_drove_75_miles_l155_155380


namespace max_value_set_x_graph_transformation_l155_155354

noncomputable def function_y (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6)) + 2

theorem max_value_set_x :
  ∃ k : ℤ, ∀ x : ℝ, x = k * Real.pi + Real.pi / 6 → function_y x = 4 :=
by
  sorry

theorem graph_transformation :
  ∀ x : ℝ, ∃ y : ℝ, (y = Real.sin x → y = 2 * Real.sin (2 * x + (Real.pi / 6)) + 2) :=
by
  sorry

end max_value_set_x_graph_transformation_l155_155354


namespace solve_system_of_equations_l155_155685

theorem solve_system_of_equations:
  (∀ (x y : ℝ), 2 * y - x - 2 * x * y = -1 ∧ 4 * x ^ 2 * y ^ 2 + x ^ 2 + 4 * y ^ 2 - 4 * x * y = 61 →
  (x, y) = (-6, -1/2) ∨ (x, y) = (1, 3) ∨ (x, y) = (1, -5/2) ∨ (x, y) = (5, -1/2)) :=
by
  sorry

end solve_system_of_equations_l155_155685


namespace positive_real_inequality_l155_155502

theorem positive_real_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end positive_real_inequality_l155_155502


namespace fraction_subtraction_equals_one_l155_155617

theorem fraction_subtraction_equals_one (x : ℝ) (h : x ≠ 1) : (x / (x - 1)) - (1 / (x - 1)) = 1 := 
by sorry

end fraction_subtraction_equals_one_l155_155617


namespace meal_arrangement_exactly_two_correct_l155_155581

noncomputable def meal_arrangement_count : ℕ :=
  let total_people := 13
  let meal_types := ["B", "B", "B", "B", "C", "C", "C", "F", "F", "F", "V", "V", "V"]
  let choose_2_people := (total_people.choose 2)
  let derangement_7 := 1854  -- Derangement of BBCCCVVV
  let derangement_9 := 133496  -- Derangement of BBCCFFFVV
  choose_2_people * (derangement_7 + derangement_9)

theorem meal_arrangement_exactly_two_correct : meal_arrangement_count = 10482600 := by
  sorry

end meal_arrangement_exactly_two_correct_l155_155581


namespace multiply_and_simplify_l155_155618

variable (a b : ℝ)

theorem multiply_and_simplify :
  (3 * a + 2 * b) * (a - 2 * b) = 3 * a^2 - 4 * a * b - 4 * b^2 :=
by
  sorry

end multiply_and_simplify_l155_155618


namespace least_number_to_add_l155_155307

theorem least_number_to_add (k n : ℕ) (h : k = 1015) (m : n = 25) : 
  ∃ x : ℕ, (k + x) % n = 0 ∧ x = 10 := by
  sorry

end least_number_to_add_l155_155307


namespace average_first_20_multiples_of_17_l155_155586

theorem average_first_20_multiples_of_17 :
  (20 / 2 : ℝ) * (17 + 17 * 20) / 20 = 178.5 := by
  sorry

end average_first_20_multiples_of_17_l155_155586


namespace non_monotonic_interval_l155_155368

def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem non_monotonic_interval (k : ℝ) :
  ¬(∀ x1 x2 ∈ set.Ioo (k-1) (k+1), x1 < x2 → f x1 ≤ f x2) ↔ (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
begin
  sorry
end

end non_monotonic_interval_l155_155368


namespace probability_three_correct_letters_l155_155702

-- Define the conditions and the theorem statement
noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

def derangements (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | k => (k - 1) * (derangements (k - 1) + derangements (k - 2))

theorem probability_three_correct_letters :
  let total_permutations := factorial 5,
      choose_three_correct := binomial_coefficient 5 3,
      derange_two := derangements 2,
      favorable_outcomes := choose_three_correct * derange_two
  in favorable_outcomes / total_permutations = 1 / 12 := 
by
  sorry

end probability_three_correct_letters_l155_155702


namespace cubic_sum_of_reciprocals_roots_l155_155228

theorem cubic_sum_of_reciprocals_roots :
  ∀ (a b c : ℝ),
  a ≠ b → b ≠ c → c ≠ a →
  0 < a ∧ a < 1 → 0 < b ∧ b < 1 → 0 < c ∧ c < 1 →
  (24 * a^3 - 38 * a^2 + 18 * a - 1 = 0) ∧
  (24 * b^3 - 38 * b^2 + 18 * b - 1 = 0) ∧
  (24 * c^3 - 38 * c^2 + 18 * c - 1 = 0) →
  ((1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 2 / 3) :=
by intros a b c neq_ab neq_bc neq_ca a_range b_range c_range roots_eqns
   sorry

end cubic_sum_of_reciprocals_roots_l155_155228


namespace series_sum_eq_neg_one_l155_155055

   noncomputable def sum_series : ℝ :=
     ∑' k : ℕ, if k = 0 then 0 else (12 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))

   theorem series_sum_eq_neg_one : sum_series = -1 :=
   sorry
   
end series_sum_eq_neg_one_l155_155055


namespace total_campers_went_rowing_l155_155449

-- Definitions based on given conditions
def morning_campers : ℕ := 36
def afternoon_campers : ℕ := 13
def evening_campers : ℕ := 49

-- Theorem statement to be proven
theorem total_campers_went_rowing : morning_campers + afternoon_campers + evening_campers = 98 :=
by sorry

end total_campers_went_rowing_l155_155449


namespace intersection_of_A_and_B_l155_155514

def setA (x : ℝ) : Prop := -1 < x ∧ x < 1
def setB (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def setC (x : ℝ) : Prop := 0 ≤ x ∧ x < 1

theorem intersection_of_A_and_B : {x : ℝ | setA x} ∩ {x | setB x} = {x | setC x} := by
  sorry

end intersection_of_A_and_B_l155_155514


namespace chicken_cost_l155_155111

theorem chicken_cost (total_money hummus_price hummus_count bacon_price vegetables_price apple_price apple_count chicken_price : ℕ)
  (h_total_money : total_money = 60)
  (h_hummus_price : hummus_price = 5)
  (h_hummus_count : hummus_count = 2)
  (h_bacon_price : bacon_price = 10)
  (h_vegetables_price : vegetables_price = 10)
  (h_apple_price : apple_price = 2)
  (h_apple_count : apple_count = 5)
  (h_remaining_money : chicken_price = total_money - (hummus_count * hummus_price + bacon_price + vegetables_price + apple_count * apple_price)) :
  chicken_price = 20 := 
by sorry

end chicken_cost_l155_155111


namespace probability_square_not_touching_outer_edge_l155_155557

theorem probability_square_not_touching_outer_edge :
  let total_squares := 10 * 10
  let perimeter_squares := 10 + 10 + (10 - 2) + (10 - 2)
  let non_perimeter_squares := total_squares - perimeter_squares
  (non_perimeter_squares / total_squares) = (16 / 25) :=
by
  let total_squares := 10 * 10
  let perimeter_squares := 10 + 10 + (10 - 2) + (10 - 2)
  let non_perimeter_squares := total_squares - perimeter_squares
  have h : non_perimeter_squares / total_squares = 16 / 25 := by sorry
  exact h

end probability_square_not_touching_outer_edge_l155_155557


namespace speed_of_man_in_still_water_l155_155303

theorem speed_of_man_in_still_water
  (V_m V_s : ℝ)
  (cond1 : V_m + V_s = 5)
  (cond2 : V_m - V_s = 7) :
  V_m = 6 :=
by
  sorry

end speed_of_man_in_still_water_l155_155303


namespace seventh_fisherman_right_neighbor_l155_155822

theorem seventh_fisherman_right_neighbor (f1 f2 f3 f4 f5 f6 f7 : ℕ) (L1 L2 L3 L4 L5 L6 L7 : ℕ) :
  (L2 * f1 = 12 ∨ L3 * f2 = 12 ∨ L4 * f3 = 12 ∨ L5 * f4 = 12 ∨ L6 * f5 = 12 ∨ L7 * f6 = 12 ∨ L1 * f7 = 12) → 
  (L2 * f1 = 14 ∨ L3 * f2 = 18 ∨ L4 * f3 = 32 ∨ L5 * f4 = 48 ∨ L6 * f5 = 70 ∨ L7 * f6 = x ∨ L1 * f7 = 12) →
  (12 * 12 * 20 * 24 * 32 * 42 * 56) / (12 * 14 * 18 * 32 * 48 * 70) = x :=
by
  sorry

end seventh_fisherman_right_neighbor_l155_155822


namespace find_x_value_l155_155073

theorem find_x_value (x : ℝ) (hx : x ≠ 0) : 
    (1/x) + (3/x) / (6/x) = 1 → x = 2 := 
by 
    intro h
    sorry

end find_x_value_l155_155073


namespace sum_of_primes_1_to_20_l155_155986

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l155_155986


namespace correct_average_weight_l155_155690

theorem correct_average_weight (n : ℕ) (incorrect_avg_weight : ℝ) (initial_avg_weight : ℝ)
  (misread_weight correct_weight : ℝ) (boys_count : ℕ) :
  incorrect_avg_weight = 58.4 →
  n = 20 →
  misread_weight = 56 →
  correct_weight = 65 →
  boys_count = n →
  initial_avg_weight = (incorrect_avg_weight * n + (correct_weight - misread_weight)) / boys_count →
  initial_avg_weight = 58.85 :=
by
  intro h1 h2 h3 h4 h5 h_avg
  sorry

end correct_average_weight_l155_155690


namespace simplify_sqrt_expr_l155_155274

/-- Simplify the given radical expression and prove its equivalence to the expected result. -/
theorem simplify_sqrt_expr :
  (Real.sqrt (5 * 3) * Real.sqrt ((3 ^ 4) * (5 ^ 2)) = 225 * Real.sqrt 15) := 
by
  sorry

end simplify_sqrt_expr_l155_155274


namespace farmer_tomatoes_l155_155177

theorem farmer_tomatoes (t p l : ℕ) (H1 : t = 97) (H2 : p = 83) : l = t - p → l = 14 :=
by {
  sorry
}

end farmer_tomatoes_l155_155177


namespace problem1_problem2_l155_155770

open BigOperators  -- to use ∑ notation

noncomputable def Sn (n : ℕ+) (a : ℤ) : ℤ := 2^n + a

noncomputable def an (n : ℕ+) (a : ℤ) : ℤ :=
  if n = 1 then 2 + a
  else 2^(n - 1)

theorem problem1 (a : ℤ) (h : ∀ n : ℕ+, Sn n a = 2^n + a) :
  a = -1 ∧ (∀ n : ℕ+, an n a = 2^(n - 1)) :=
sorry

noncomputable def bn (n : ℕ+) (a : ℤ) : ℚ :=
  Real.log (an n a) / Real.log 4 + 1

noncomputable def S'n (n : ℕ+) (a : ℤ) : ℚ := ∑ i in Finset.range n, bn (i + 1) a

theorem problem2 (a : ℤ) (h1 : a = -1) (h2 : ∀ n : ℕ+, an n a = 2^(n - 1)) :
  {n : ℕ | 2 * S'n n a ≤ 5} = {1, 2} :=
sorry

end problem1_problem2_l155_155770


namespace max_f_value_no_min_f_value_l155_155231

open Classical
open Real

noncomputable theory

def domain_M (x : ℝ) : Prop :=
  3 - 4 * x + x ^ 2 > 0

def f (x : ℝ) : ℝ :=
  2 ^ (x + 2) - 3 * 4 ^ x

theorem max_f_value :
  ∃ x, domain_M x ∧ f x = 4 / 3 :=
sorry

theorem no_min_f_value :
  ¬∃ m x, domain_M x ∧ f x = m ∧ ∀ y, domain_M y → f y ≥ m :=
sorry

end max_f_value_no_min_f_value_l155_155231


namespace factorization_correct_l155_155748

def expression (x : ℝ) : ℝ := 16 * x^3 + 4 * x^2
def factored_expression (x : ℝ) : ℝ := 4 * x^2 * (4 * x + 1)

theorem factorization_correct (x : ℝ) : expression x = factored_expression x := 
by 
  sorry

end factorization_correct_l155_155748


namespace area_regular_octagon_l155_155087

theorem area_regular_octagon (AB BC: ℝ) (hAB: AB = 2) (hBC: BC = 2) :
  let side_length := 2 * Real.sqrt 2
  let triangle_area := (AB * AB) / 2
  let total_triangle_area := 4 * triangle_area
  let side_length_rect := 4 + 2 * Real.sqrt 2
  let rect_area := side_length_rect * side_length_rect
  let octagon_area := rect_area - total_triangle_area
  octagon_area = 16 + 8 * Real.sqrt 2 :=
by sorry

end area_regular_octagon_l155_155087


namespace divisor_is_four_l155_155790

theorem divisor_is_four (d n : ℤ) (k j : ℤ) 
  (h1 : n % d = 3) 
  (h2 : 2 * n % d = 2): d = 4 :=
sorry

end divisor_is_four_l155_155790


namespace total_ticket_revenue_l155_155461

theorem total_ticket_revenue (total_seats : Nat) (price_adult_ticket : Nat) (price_child_ticket : Nat)
  (theatre_full : Bool) (child_tickets : Nat) (adult_tickets := total_seats - child_tickets)
  (rev_adult := adult_tickets * price_adult_ticket) (rev_child := child_tickets * price_child_ticket) :
  total_seats = 250 →
  price_adult_ticket = 6 →
  price_child_ticket = 4 →
  theatre_full = true →
  child_tickets = 188 →
  rev_adult + rev_child = 1124 := 
by
  intros h_total_seats h_price_adult h_price_child h_theatre_full h_child_tickets
  sorry

end total_ticket_revenue_l155_155461


namespace cost_of_each_television_l155_155877

-- Define the conditions
def number_of_televisions : Nat := 5
def number_of_figurines : Nat := 10
def cost_per_figurine : Nat := 1
def total_spent : Nat := 260

-- Define the proof problem
theorem cost_of_each_television (T : Nat) :
  (number_of_televisions * T + number_of_figurines * cost_per_figurine = total_spent) → (T = 50) :=
by
  sorry

end cost_of_each_television_l155_155877


namespace sum_of_primes_1_to_20_l155_155999

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l155_155999


namespace cans_needed_eq_l155_155951

axiom Paula_initial_rooms : ℕ
axiom Paula_lost_cans : ℕ
axiom Paula_after_loss_rooms : ℕ
axiom cans_for_25_rooms : ℕ

theorem cans_needed_eq :
  Paula_initial_rooms = 30 →
  Paula_lost_cans = 3 →
  Paula_after_loss_rooms = 25 →
  cans_for_25_rooms = 15 :=
by
  intros
  sorry

end cans_needed_eq_l155_155951


namespace animals_on_farm_l155_155324

theorem animals_on_farm (cows : ℕ) (sheep : ℕ) (pigs : ℕ) 
  (h1 : cows = 12) 
  (h2 : sheep = 2 * cows) 
  (h3 : pigs = 3 * sheep) : 
  cows + sheep + pigs = 108 := 
by
  sorry

end animals_on_farm_l155_155324


namespace problem_statement_l155_155925

noncomputable def decimalPartSqrtFive : ℝ := Real.sqrt 5 - 2
def integerPartSqrtThirteen : ℕ := 3

theorem problem_statement :
  decimalPartSqrtFive + integerPartSqrtThirteen - Real.sqrt 5 = 1 :=
by
  sorry

end problem_statement_l155_155925


namespace printer_cost_comparison_l155_155154

-- Definitions based on the given conditions
def in_store_price : ℝ := 150.00
def discount_rate : ℝ := 0.10
def installment_payment : ℝ := 28.00
def number_of_installments : ℕ := 5
def shipping_handling_charge : ℝ := 12.50

-- Discounted in-store price calculation
def discounted_in_store_price : ℝ := in_store_price * (1 - discount_rate)

-- Total cost from the television advertiser
def tv_advertiser_total_cost : ℝ := (number_of_installments * installment_payment) + shipping_handling_charge

-- Proof statement
theorem printer_cost_comparison :
  discounted_in_store_price - tv_advertiser_total_cost = -17.50 :=
by
  sorry

end printer_cost_comparison_l155_155154


namespace modulo_remainder_even_l155_155743

theorem modulo_remainder_even (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2018) : 
  ((2019 - k)^12 + 2018) % 2019 = (k^12 + 2018) % 2019 := 
by
  sorry

end modulo_remainder_even_l155_155743


namespace ducks_arrival_quantity_l155_155473

variable {initial_ducks : ℕ} (arrival_ducks : ℕ)

def initial_geese (initial_ducks : ℕ) := 2 * initial_ducks - 10

def remaining_geese (initial_ducks : ℕ) := initial_geese initial_ducks - 10

def remaining_ducks (initial_ducks arrival_ducks : ℕ) := initial_ducks + arrival_ducks

theorem ducks_arrival_quantity :
  initial_ducks = 25 →
  remaining_geese initial_ducks = 30 →
  remaining_geese initial_ducks = remaining_ducks initial_ducks arrival_ducks + 1 →
  arrival_ducks = 4 :=
by
sorry

end ducks_arrival_quantity_l155_155473


namespace find_n_for_sum_l155_155816

theorem find_n_for_sum (n : ℕ) : ∃ n, n * (2 * n - 1) = 2009 ^ 2 :=
by
  sorry

end find_n_for_sum_l155_155816


namespace find_4_digit_number_l155_155495

theorem find_4_digit_number (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182) :
  1000 * a + 100 * b + 10 * c + d = 1909 :=
by {
  sorry
}

end find_4_digit_number_l155_155495


namespace remainder_2023_mul_7_div_45_l155_155166

/-- The remainder when the product of 2023 and 7 is divided by 45 is 31. -/
theorem remainder_2023_mul_7_div_45 : 
  (2023 * 7) % 45 = 31 := 
by
  sorry

end remainder_2023_mul_7_div_45_l155_155166


namespace determine_a_l155_155487

theorem determine_a (a : ℕ) (h : a / (a + 36) = 9 / 10) : a = 324 :=
sorry

end determine_a_l155_155487


namespace triangle_angles_ratio_l155_155413

theorem triangle_angles_ratio (A B C : ℕ) 
  (hA : A = 20)
  (hB : B = 3 * A)
  (hSum : A + B + C = 180) :
  (C / A) = 5 := 
by
  sorry

end triangle_angles_ratio_l155_155413


namespace prob_chair_theorem_l155_155869

def numAvailableChairs : ℕ := 10 - 1

def totalWaysToChooseTwoChairs : ℕ := Nat.choose numAvailableChairs 2

def adjacentPairs : ℕ :=
  let pairs := [(1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9)]
  pairs.length

def probNextToEachOther : ℚ := adjacentPairs / totalWaysToChooseTwoChairs

def probNotNextToEachOther : ℚ := 1 - probNextToEachOther

theorem prob_chair_theorem : probNotNextToEachOther = 5/6 :=
by
  sorry

end prob_chair_theorem_l155_155869


namespace seq_arithmetic_l155_155775

theorem seq_arithmetic (a : ℕ → ℕ) (h : ∀ p q : ℕ, a p + a q = a (p + q)) (h1 : a 1 = 2) :
  ∀ n : ℕ, a n = 2 * n :=
by
  sorry

end seq_arithmetic_l155_155775


namespace greatest_possible_avg_speed_l155_155109

theorem greatest_possible_avg_speed {initial_odometer : ℕ} (t : ℕ) (max_speed : ℕ) 
  (final_odometer : ℕ) : 
  initial_odometer = 12321 ∧ t = 4 ∧ max_speed = 65 ∧ 
  (∀ n : ℕ, palindrome n → initial_odometer < final_odometer ∧ final_odometer ≤ initial_odometer + 260) →
  (final_odometer - initial_odometer) / t = 50 :=
by sorry

end greatest_possible_avg_speed_l155_155109


namespace school_boys_count_l155_155016

theorem school_boys_count (B G : ℕ) (h1 : B + G = 1150) (h2 : G = (B / 1150) * 100) : B = 1058 := 
by 
  sorry

end school_boys_count_l155_155016


namespace sum_of_primes_1_to_20_l155_155997

theorem sum_of_primes_1_to_20 : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := by
  sorry

end sum_of_primes_1_to_20_l155_155997


namespace lines_perpendicular_to_same_plane_are_parallel_l155_155609

variables {Point Line Plane : Type}
variables (a b c : Line) (α β γ : Plane)
variables (perp_line_to_plane : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)
variables (subset_line_in_plane : Line → Plane → Prop)

-- The conditions
axiom a_perp_alpha : perp_line_to_plane a α
axiom b_perp_alpha : perp_line_to_plane b α

-- The statement to prove
theorem lines_perpendicular_to_same_plane_are_parallel :
  parallel_lines a b :=
by sorry

end lines_perpendicular_to_same_plane_are_parallel_l155_155609


namespace complex_fraction_identity_l155_155669

theorem complex_fraction_identity (c d : ℂ) (h_nonzero_c : c ≠ 0) (h_nonzero_d : d ≠ 0) (h_condition : c^2 + c * d + d^2 = 0) : 
  (c^12 + d^12) / (c + d)^12 = -2 :=
by sorry

end complex_fraction_identity_l155_155669


namespace value_of_x_y_l155_155093

noncomputable def real_ln : ℝ → ℝ := sorry

theorem value_of_x_y (x y : ℝ) (h : 3 * x - y ≤ real_ln (x + 2 * y - 3) + real_ln (2 * x - 3 * y + 5)) :
  x + y = 16 / 7 :=
sorry

end value_of_x_y_l155_155093


namespace middle_managers_to_be_selected_l155_155456

def total_employees : ℕ := 160
def senior_managers : ℕ := 10
def middle_managers : ℕ := 30
def staff_members : ℕ := 120
def total_to_be_selected : ℕ := 32

theorem middle_managers_to_be_selected : 
  (middle_managers * total_to_be_selected / total_employees) = 6 := by
  sorry

end middle_managers_to_be_selected_l155_155456


namespace amount_allocated_to_food_l155_155443

theorem amount_allocated_to_food (total_amount : ℝ) (household_ratio food_ratio misc_ratio : ℝ) 
  (h₁ : total_amount = 1800) (h₂ : household_ratio = 5) (h₃ : food_ratio = 4) (h₄ : misc_ratio = 1) :
  food_ratio / (household_ratio + food_ratio + misc_ratio) * total_amount = 720 :=
by
  sorry

end amount_allocated_to_food_l155_155443


namespace ratio_of_cost_to_marked_price_l155_155734

variable (p : ℝ)

theorem ratio_of_cost_to_marked_price :
  let selling_price := (3/4) * p
  let cost_price := (5/8) * selling_price
  cost_price / p = 15 / 32 :=
by
  let selling_price := (3 / 4) * p
  let cost_price := (5 / 8) * selling_price
  sorry

end ratio_of_cost_to_marked_price_l155_155734


namespace sum_xyz_is_sqrt_13_l155_155224

variable (x y z : ℝ)

-- The conditions
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z

axiom eq1 : x^2 + y^2 + x * y = 3
axiom eq2 : y^2 + z^2 + y * z = 4
axiom eq3 : z^2 + x^2 + z * x = 7 

-- The theorem statement: Prove that x + y + z = sqrt(13)
theorem sum_xyz_is_sqrt_13 : x + y + z = Real.sqrt 13 :=
by
  sorry

end sum_xyz_is_sqrt_13_l155_155224


namespace floor_sufficient_but_not_necessary_l155_155244

theorem floor_sufficient_but_not_necessary {x y : ℝ} : 
  (∀ x y : ℝ, (⌊x⌋₊ = ⌊y⌋₊) → abs (x - y) < 1) ∧ 
  ¬ (∀ x y : ℝ, abs (x - y) < 1 → (⌊x⌋₊ = ⌊y⌋₊)) :=
by
  sorry

end floor_sufficient_but_not_necessary_l155_155244


namespace find_original_number_l155_155162

theorem find_original_number (n : ℝ) (h : n / 2 = 9) : n = 18 :=
sorry

end find_original_number_l155_155162


namespace main_theorem_l155_155391

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition: f is symmetric about x = 1
def symmetric_about_one (a b c : ℝ) : Prop := 
  ∀ x : ℝ, f a b c (1 - x) = f a b c (1 + x)

-- Main statement
theorem main_theorem (a b c : ℝ) (h₁ : 0 < a) (h₂ : symmetric_about_one a b c) :
  ∀ x : ℝ, f a b c (2^x) > f a b c (3^x) :=
sorry

end main_theorem_l155_155391


namespace number_of_distinct_configurations_l155_155446

-- Define the conditions
def numConfigurations (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 else n + 1

-- Theorem statement
theorem number_of_distinct_configurations (n : ℕ) : 
  numConfigurations n = if n % 2 = 1 then 2 else n + 1 :=
by
  sorry -- Proof intentionally left out

end number_of_distinct_configurations_l155_155446


namespace simplify_exponent_product_l155_155169

theorem simplify_exponent_product :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_product_l155_155169


namespace exists_same_color_points_at_distance_one_l155_155866

theorem exists_same_color_points_at_distance_one (coloring : ℝ × ℝ → Fin 3) :
  ∃ (p q : ℝ × ℝ), (coloring p = coloring q) ∧ (dist p q = 1) := sorry

end exists_same_color_points_at_distance_one_l155_155866


namespace supplement_of_angle_l155_155649

-- Condition: The complement of angle α is 54 degrees 32 minutes
theorem supplement_of_angle (α : ℝ) (h : α = 90 - (54 + 32 / 60)) :
  180 - α = 144 + 32 / 60 := by
sorry

end supplement_of_angle_l155_155649


namespace ice_cube_count_l155_155480

theorem ice_cube_count (cubes_per_tray : ℕ) (tray_count : ℕ) (H1: cubes_per_tray = 9) (H2: tray_count = 8) :
  cubes_per_tray * tray_count = 72 :=
by
  sorry

end ice_cube_count_l155_155480


namespace find_total_people_find_children_l155_155711

variables (x m : ℕ)

-- Given conditions translated into Lean

def group_b_more_people (x : ℕ) := x + 4
def sum_is_18_times_difference (x : ℕ) := (x + (x + 4)) = 18 * ((x + 4) - x)
def children_b_less_than_three_times (m : ℕ) := (3 * m) - 2
def adult_ticket_price := 100
def children_ticket_price := (100 * 60) / 100
def same_amount_spent (x m : ℕ) := 100 * (x - m) + (100 * 60 / 100) * m = 100 * ((group_b_more_people x) - (children_b_less_than_three_times m)) + (100 * 60 / 100) * (children_b_less_than_three_times m)

-- Proving the two propositions (question == answer given conditions)

theorem find_total_people (x : ℕ) (hx : sum_is_18_times_difference x) : x = 34 ∧ (group_b_more_people x) = 38 :=
by {
  sorry -- proof for x = 34 and group_b_people = 38 given that sum_is_18_times_difference x
}

theorem find_children (m : ℕ) (x : ℕ) (hx : sum_is_18_times_difference x) (hm : same_amount_spent x m) : m = 6 ∧ (children_b_less_than_three_times m) = 16 :=
by {
  sorry -- proof for m = 6 and children_b_people = 16 given sum_is_18_times_difference x and same_amount_spent x m
}

end find_total_people_find_children_l155_155711


namespace min_trucks_needed_l155_155133

theorem min_trucks_needed (n : ℕ) (w : ℕ) (t : ℕ) (total_weight : ℕ) (max_box_weight : ℕ) : 
    (total_weight = 10) → 
    (max_box_weight = 1) → 
    (t = 3) →
    (n * max_box_weight = total_weight) →
    (n ≥ 10) →
    ∀ min_trucks : ℕ, (min_trucks * t ≥ total_weight) → 
    min_trucks = 5 :=
by
  intro total_weight_eq max_box_weight_eq truck_capacity box_total_weight_eq n_lower_bound min_trucks min_trucks_condition
  sorry

end min_trucks_needed_l155_155133


namespace lighting_candles_correct_l155_155973

noncomputable def time_to_light_candles (initial_length : ℝ) : ℝ :=
  let burn_rate_1 := initial_length / 300
  let burn_rate_2 := initial_length / 240
  let t := (5 * 60 + 43) - (5 * 60) -- 11:17 AM is 342.857 minutes before 5 PM
  if ((initial_length - burn_rate_2 * t) = 3 * (initial_length - burn_rate_1 * t)) then 11 + 17 / 60 else 0 -- Check if the condition is met

theorem lighting_candles_correct :
  ∀ (initial_length : ℝ), time_to_light_candles initial_length = 11 + 17 / 60 :=
by
  intros initial_length
  sorry  -- Proof goes here

end lighting_candles_correct_l155_155973


namespace average_marbles_of_other_colors_l155_155129

theorem average_marbles_of_other_colors :
  let total_percentage := 100
  let clear_percentage := 40
  let black_percentage := 20
  let other_percentage := total_percentage - clear_percentage - black_percentage
  let marbles_taken := 5
  (other_percentage / 100) * marbles_taken = 2 :=
by
  sorry

end average_marbles_of_other_colors_l155_155129


namespace positive_integer_solutions_eq_8_2_l155_155353

-- Define the variables and conditions in the problem
def positive_integer_solution_count_eq (n m : ℕ) : Prop :=
  ∀ (x₁ x₂ x₃ x₄ : ℕ),
    x₂ = m →
    (x₁ + x₂ + x₃ + x₄ = n) →
    (x₁ > 0 ∧ x₃ > 0 ∧ x₄ > 0) →
    -- Number of positive integer solutions should be 10
    (x₁ + x₃ + x₄ = 6)

-- Statement of the theorem
theorem positive_integer_solutions_eq_8_2 : positive_integer_solution_count_eq 8 2 := sorry

end positive_integer_solutions_eq_8_2_l155_155353


namespace beta_angle_relationship_l155_155431

theorem beta_angle_relationship (α β γ : ℝ) (h1 : β - α = 3 * γ) (h2 : α + β + γ = 180) : β = 90 + γ :=
sorry

end beta_angle_relationship_l155_155431


namespace barbara_total_cost_l155_155478

-- Define conditions
def steak_weight : ℝ := 4.5
def steak_price_per_pound : ℝ := 15.0
def chicken_weight : ℝ := 1.5
def chicken_price_per_pound : ℝ := 8.0

-- Define total cost formula
def total_cost := (steak_weight * steak_price_per_pound) + (chicken_weight * chicken_price_per_pound)

-- Prove that the total cost equals $79.50
theorem barbara_total_cost : total_cost = 79.50 := by
  sorry

end barbara_total_cost_l155_155478


namespace hcf_two_numbers_l155_155098

theorem hcf_two_numbers
  (x y : ℕ) 
  (h_lcm : Nat.lcm x y = 560)
  (h_prod : x * y = 42000) : Nat.gcd x y = 75 :=
by
  sorry

end hcf_two_numbers_l155_155098


namespace hyperbola_equation_of_midpoint_l155_155092

-- Define the hyperbola E
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Given conditions
variables (a b : ℝ) (hapos : a > 0) (hbpos : b > 0)
variables (F : ℝ × ℝ) (hF : F = (-2, 0))
variables (M : ℝ × ℝ) (hM : M = (-3, -1))

-- The statement requiring proof
theorem hyperbola_equation_of_midpoint (hE : hyperbola a b (-2) 0) 
(hFocus : a^2 + b^2 = 4) : 
  (∃ a' b', a' = 3 ∧ b' = 1 ∧ hyperbola a' b' (-3) (-1)) :=
sorry

end hyperbola_equation_of_midpoint_l155_155092


namespace water_distribution_scheme_l155_155174

theorem water_distribution_scheme (a b c : ℚ) : 
  a + b + c = 1 ∧ 
  (∀ x : ℂ, ∃ n : ℕ, x^n = 1 → x = 1) ∧
  (∀ (x : ℂ), (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11 + x^12 + x^13 + x^14 + x^15 + x^16 + x^17 + x^18 + x^19 + x^20 + x^21 + x^22 = 0) → false) → 
  a = 0 ∧ b = 0 ∧ c = 1 :=
by
  sorry

end water_distribution_scheme_l155_155174


namespace range_abs_plus_one_l155_155885

 theorem range_abs_plus_one : 
   ∀ y : ℝ, (∃ x : ℝ, y = |x| + 1) ↔ y ≥ 1 := 
 by
   sorry
 
end range_abs_plus_one_l155_155885


namespace apple_tree_distribution_l155_155799

-- Definition of the problem
noncomputable def paths := 4

-- Definition of the apple tree positions
structure Position where
  x : ℕ -- Coordinate x
  y : ℕ -- Coordinate y

-- Definition of the initial condition: one existing apple tree
def existing_apple_tree : Position := {x := 0, y := 0}

-- Problem: proving the existence of a configuration with three new apple trees
theorem apple_tree_distribution :
  ∃ (p1 p2 p3 : Position),
    (p1 ≠ existing_apple_tree) ∧ (p2 ≠ existing_apple_tree) ∧ (p3 ≠ existing_apple_tree) ∧
    -- Ensure each path has equal number of trees on both sides
    (∃ (path1 path2 : ℕ), 
      -- Horizontal path balance
      path1 = (if p1.x > 0 then 1 else 0) + (if p2.x > 0 then 1 else 0) + (if p3.x > 0 then 1 else 0) + 1 ∧
      path2 = (if p1.x < 0 then 1 else 0) + (if p2.x < 0 then 1 else 0) + (if p3.x < 0 then 1 else 0) ∧
      path1 = path2) ∧
    (∃ (path3 path4 : ℕ), 
      -- Vertical path balance
      path3 = (if p1.y > 0 then 1 else 0) + (if p2.y > 0 then 1 else 0) + (if p3.y > 0 then 1 else 0) + 1 ∧
      path4 = (if p1.y < 0 then 1 else 0) + (if p2.y < 0 then 1 else 0) + (if p3.y < 0 then 1 else 0) ∧
      path3 = path4)
  := by sorry

end apple_tree_distribution_l155_155799


namespace age_difference_l155_155862

theorem age_difference (S M : ℕ) 
  (h1 : S = 35)
  (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 37 :=
by
  sorry

end age_difference_l155_155862


namespace mean_of_five_integers_l155_155144

theorem mean_of_five_integers
  (p q r s t : ℤ)
  (h1 : (p + q + r) / 3 = 9)
  (h2 : (s + t) / 2 = 14) :
  (p + q + r + s + t) / 5 = 11 :=
by
  sorry

end mean_of_five_integers_l155_155144


namespace product_variation_l155_155840

theorem product_variation (a b c : ℕ) (h1 : a * b = c) (h2 : b' = 10 * b) (h3 : ∃ d : ℕ, d = a * b') : d = 720 :=
by
  sorry

end product_variation_l155_155840


namespace geometric_sequence_sum_reciprocal_ratio_l155_155099

theorem geometric_sequence_sum_reciprocal_ratio
  (a : ℚ) (r : ℚ) (n : ℕ) (S S' : ℚ)
  (h1 : a = 1/4)
  (h2 : r = 2)
  (h3 : S = a * (1 - r^n) / (1 - r))
  (h4 : S' = (1/a) * (1 - (1/r)^n) / (1 - 1/r)) :
  S / S' = 32 :=
sorry

end geometric_sequence_sum_reciprocal_ratio_l155_155099


namespace total_seashells_found_l155_155666

-- Defining the conditions
def joan_daily_seashells : ℕ := 6
def jessica_daily_seashells : ℕ := 8
def length_of_vacation : ℕ := 7

-- Stating the theorem
theorem total_seashells_found : 
  (joan_daily_seashells + jessica_daily_seashells) * length_of_vacation = 98 :=
by
  sorry

end total_seashells_found_l155_155666


namespace maria_trip_time_l155_155489

theorem maria_trip_time 
(s_highway : ℕ) (s_mountain : ℕ) (d_highway : ℕ) (d_mountain : ℕ) (t_mountain : ℕ) (t_break : ℕ) : 
  (s_highway = 4 * s_mountain) -> 
  (t_mountain = d_mountain / s_mountain) -> 
  t_mountain = 40 -> 
  t_break = 15 -> 
  d_highway = 100 -> 
  d_mountain = 20 ->
  s_mountain = d_mountain / t_mountain -> 
  s_highway = 4 * s_mountain -> 
  d_highway / s_highway = 50 ->
  40 + 50 + 15 = 105 := 
by 
  sorry

end maria_trip_time_l155_155489


namespace johnny_needs_45_planks_l155_155808

theorem johnny_needs_45_planks
  (legs_per_table : ℕ)
  (planks_per_leg : ℕ)
  (surface_planks_per_table : ℕ)
  (number_of_tables : ℕ)
  (h1 : legs_per_table = 4)
  (h2 : planks_per_leg = 1)
  (h3 : surface_planks_per_table = 5)
  (h4 : number_of_tables = 5) :
  number_of_tables * (legs_per_table * planks_per_leg + surface_planks_per_table) = 45 :=
by
  sorry

end johnny_needs_45_planks_l155_155808


namespace perimeter_original_square_l155_155319

theorem perimeter_original_square (s : ℝ) (h1 : (3 / 4) * s^2 = 48) : 4 * s = 32 :=
by
  sorry

end perimeter_original_square_l155_155319


namespace interval_of_increase_l155_155834

noncomputable def u (x : ℝ) : ℝ := x^2 - 5*x + 6

def increasing_interval (f : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ interval → y ∈ interval → x < y → f x < f y

noncomputable def f (x : ℝ) : ℝ := Real.log (u x)

theorem interval_of_increase :
  increasing_interval f {x : ℝ | 3 < x} :=
sorry

end interval_of_increase_l155_155834


namespace sum_first_110_terms_l155_155700

variable (a d : ℕ → ℤ) [is_arithmetic_sequence: ∀ n, a (n + 1) = a n + d n]

-- Given that the sum of the first 10 terms is 100
def sum_first_10_terms := (∑ i in Finset.range 10, a i) = 100

-- Given that the sum of the first 100 terms is 10
def sum_first_100_terms := (∑ i in Finset.range 100, a i) = 10

-- Prove that the sum of the first 110 terms is -110
theorem sum_first_110_terms (h1 : sum_first_10_terms a) (h2 : sum_first_100_terms a) : 
  (∑ i in Finset.range 110, a i) = -110 :=
sorry

end sum_first_110_terms_l155_155700


namespace compute_xy_l155_155420

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 108) : x * y = 0 := 
sorry

end compute_xy_l155_155420


namespace next_year_property_appears_l155_155289

def no_smaller_rearrangement (n: Nat) : Prop :=
  ∀ (l: List Nat), (l.permutations.map (λ p, p.foldl (λ acc d, acc * 10 + d) 0)).all (λ m, m >= n)

def next_year_with_property (current: Nat) : Nat :=
  if h : current = 2022 then 2022
  else if ∃ n, n > current ∧ no_smaller_rearrangement n then
    Classical.some (Classical.some_spec h)
  else current

theorem next_year_property_appears : next_year_with_property 2009 = 2022 := by
  sorry

end next_year_property_appears_l155_155289


namespace contrapositive_example_l155_155281

theorem contrapositive_example (x : ℝ) :
  (x ^ 2 < 1 → -1 < x ∧ x < 1) ↔ (x ≥ 1 ∨ x ≤ -1 → x ^ 2 ≥ 1) :=
sorry

end contrapositive_example_l155_155281


namespace james_initial_bars_l155_155381

def initial_chocolate_bars (sold_last_week sold_this_week needs_to_sell : ℕ) : ℕ :=
  sold_last_week + sold_this_week + needs_to_sell

theorem james_initial_bars : 
  initial_chocolate_bars 5 7 6 = 18 :=
by 
  sorry

end james_initial_bars_l155_155381


namespace infinite_geometric_series_l155_155118

theorem infinite_geometric_series
  (p q r : ℝ)
  (h_series : ∑' n : ℕ, p / q^(n+1) = 9) :
  (∑' n : ℕ, p / (p + r)^(n+1)) = (9 * (q - 1)) / (9 * q + r - 10) :=
by 
  sorry

end infinite_geometric_series_l155_155118


namespace find_first_term_and_common_difference_l155_155756

variable (n : ℕ)
variable (a_1 d : ℚ)

-- Definition of the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (n : ℕ) (a_1 d : ℚ) : ℚ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom sum_condition : ∀ (n : ℕ), sum_arithmetic_seq n a_1 d = n^2 / 2

-- Theorem to prove
theorem find_first_term_and_common_difference 
  (a_1 d : ℚ) 
  (sum_condition : ∀ (n : ℕ), sum_arithmetic_seq n a_1 d = n^2 / 2) 
: a_1 = 1/2 ∧ d = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end find_first_term_and_common_difference_l155_155756


namespace lower_bound_for_x_l155_155370

variable {x y : ℝ}  -- declaring x and y as real numbers

theorem lower_bound_for_x 
  (h₁ : 3 < x) (h₂ : x < 6)
  (h₃ : 6 < y) (h₄ : y < 8)
  (h₅ : y - x = 4) : 
  ∃ ε > 0, 3 + ε = x := 
sorry

end lower_bound_for_x_l155_155370


namespace probability_at_least_one_tree_survives_l155_155045

noncomputable def prob_at_least_one_survives (survival_rate_A survival_rate_B : ℚ) (n_A n_B : ℕ) : ℚ :=
  1 - ((1 - survival_rate_A)^(n_A) * (1 - survival_rate_B)^(n_B))

theorem probability_at_least_one_tree_survives :
  prob_at_least_one_survives (5/6) (4/5) 2 2 = 899 / 900 :=
by
  sorry

end probability_at_least_one_tree_survives_l155_155045


namespace positive_root_exists_iff_p_range_l155_155059

theorem positive_root_exists_iff_p_range (p : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x^4 + 4 * p * x^3 + x^2 + 4 * p * x + 4 = 0) ↔ 
  p ∈ Set.Iio (-Real.sqrt 2 / 2) ∪ Set.Ioi (Real.sqrt 2 / 2) :=
by
  sorry

end positive_root_exists_iff_p_range_l155_155059


namespace probability_outside_circle_is_7_over_9_l155_155564

noncomputable def probability_point_outside_circle :
    ℚ :=
sorry

theorem probability_outside_circle_is_7_over_9 :
    probability_point_outside_circle = 7 / 9 :=
sorry

end probability_outside_circle_is_7_over_9_l155_155564


namespace savings_account_final_amount_l155_155400

noncomputable def final_amount (P R : ℝ) (t : ℕ) : ℝ :=
  P * (1 + R) ^ t

theorem savings_account_final_amount :
  final_amount 2500 0.06 21 = 8017.84 :=
by
  sorry

end savings_account_final_amount_l155_155400


namespace intersection_of_A_and_B_l155_155513

def setA (x : ℝ) : Prop := -1 < x ∧ x < 1
def setB (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def setC (x : ℝ) : Prop := 0 ≤ x ∧ x < 1

theorem intersection_of_A_and_B : {x : ℝ | setA x} ∩ {x | setB x} = {x | setC x} := by
  sorry

end intersection_of_A_and_B_l155_155513


namespace inequality_AM_GM_l155_155765

variable {a b c d : ℝ}
variable (h₁ : 0 < a)
variable (h₂ : 0 < b)
variable (h₃ : 0 < c)
variable (h₄ : 0 < d)

theorem inequality_AM_GM :
  (c / a * (8 * b + c) + d / b * (8 * c + d) + a / c * (8 * d + a) + b / d * (8 * a + b)) ≥ 9 * (a + b + c + d) :=
sorry

end inequality_AM_GM_l155_155765


namespace no_such_ab_l155_155888

theorem no_such_ab (a b : ℤ) : ¬ (2006^2 ∣ a^2006 + b^2006 + 1) :=
sorry

end no_such_ab_l155_155888


namespace isosceles_triangle_condition_l155_155751

-- Theorem statement
theorem isosceles_triangle_condition (N : ℕ) (h : N > 2) : 
  (∃ N1 : ℕ, N = N1 ∧ N1 = 10) ∨ (∃ N2 : ℕ, N = N2 ∧ N2 = 11) :=
by sorry

end isosceles_triangle_condition_l155_155751


namespace relationship_among_a_b_c_l155_155511

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_deriv : ∀ x ≠ 0, f'' x + f x / x > 0)

noncomputable def a : ℝ := (1 / Real.exp 1) * f (1 / Real.exp 1)
noncomputable def b : ℝ := -Real.exp 1 * f (-Real.exp 1)
noncomputable def c : ℝ := f 1

theorem relationship_among_a_b_c :
  a < c ∧ c < b :=
by
  -- sorry to skip the proof steps
  sorry

end relationship_among_a_b_c_l155_155511


namespace SarahsNumber_is_2880_l155_155959

def SarahsNumber (n : ℕ) : Prop :=
  (144 ∣ n) ∧ (45 ∣ n) ∧ (1000 ≤ n ∧ n ≤ 3000)

theorem SarahsNumber_is_2880 : SarahsNumber 2880 :=
  by
  sorry

end SarahsNumber_is_2880_l155_155959


namespace solve_equation_l155_155138

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (x / (x + 1) = 2 / (x^2 - 1)) ↔ (x = 2) :=
by
  sorry

end solve_equation_l155_155138


namespace altitude_correct_l155_155691

-- Define the given sides and area of the triangle
def AB : ℝ := 30
def BC : ℝ := 17
def AC : ℝ := 25
def area_ABC : ℝ := 120

-- The length of the altitude from the vertex C to the base AB
def height_C_to_AB : ℝ := 8

-- Problem statement to be proven
theorem altitude_correct : (1 / 2) * AB * height_C_to_AB = area_ABC :=
by
  sorry

end altitude_correct_l155_155691


namespace smallest_integer_satisfying_mod_conditions_l155_155062

theorem smallest_integer_satisfying_mod_conditions :
  ∃ n : ℕ, n > 0 ∧ 
  (n % 3 = 2) ∧ 
  (n % 5 = 4) ∧ 
  (n % 7 = 6) ∧ 
  (n % 11 = 10) ∧ 
  n = 1154 := 
sorry

end smallest_integer_satisfying_mod_conditions_l155_155062


namespace function_value_corresponds_to_multiple_independent_variables_l155_155591

theorem function_value_corresponds_to_multiple_independent_variables
  {α β : Type*} (f : α → β) :
  ∃ (b : β), ∃ (a1 a2 : α), a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b :=
sorry

end function_value_corresponds_to_multiple_independent_variables_l155_155591


namespace functional_equation_solution_l155_155810

theorem functional_equation_solution (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, (x + y) * (f x - f y) = a * (x - y) * f (x + y)) :
  (a = 1 → ∃ α β : ℝ, ∀ x : ℝ, f x = α * x^2 + β * x) ∧
  (a ≠ 1 ∧ a ≠ 0 → ∀ x : ℝ, f x = 0) ∧
  (a = 0 → ∃ c : ℝ, ∀ x : ℝ, f x = c) :=
by sorry

end functional_equation_solution_l155_155810


namespace sum_primes_between_1_and_20_l155_155991

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l155_155991


namespace abs_inequality_l155_155769

variables (a b c : ℝ)

theorem abs_inequality (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_inequality_l155_155769


namespace intersection_empty_set_l155_155916

def M : Set ℝ := { y | ∃ x, x > 0 ∧ y = 2^x }
def N : Set ℝ := { y | ∃ x, y = Real.sqrt (2*x - x^2) }

theorem intersection_empty_set :
  M ∩ N = ∅ :=
by
  sorry

end intersection_empty_set_l155_155916


namespace shop_discount_percentage_l155_155965

-- Definitions based on conditions
def original_price := 800
def price_paid := 560
def discount_amount := original_price - price_paid
def percentage_discount := (discount_amount / original_price) * 100

-- Proposition to prove
theorem shop_discount_percentage : percentage_discount = 30 := by
  sorry

end shop_discount_percentage_l155_155965


namespace percentage_of_women_employees_l155_155329

variable (E W M : ℝ)

-- Introduce conditions
def total_employees_are_married : Prop := 0.60 * E = (1 / 3) * M + 0.6842 * W
def total_employees_count : Prop := W + M = E
def percentage_of_women : Prop := W = 0.7601 * E

-- State the theorem to prove
theorem percentage_of_women_employees :
  total_employees_are_married E W M ∧ total_employees_count E W M → percentage_of_women E W :=
by sorry

end percentage_of_women_employees_l155_155329


namespace clean_room_time_l155_155393

theorem clean_room_time :
  let lisa_time := 8
  let kay_time := 12
  let ben_time := 16
  let combined_work_rate := (1 / lisa_time) + (1 / kay_time) + (1 / ben_time)
  let total_time := 1 / combined_work_rate
  total_time = 48 / 13 :=
by
  sorry

end clean_room_time_l155_155393


namespace complement_of_union_l155_155358

open Set

variable (U A B : Set ℕ)
variable (u_def : U = {0, 1, 2, 3, 4, 5, 6})
variable (a_def : A = {1, 3})
variable (b_def : B = {3, 5})

theorem complement_of_union :
  (U \ (A ∪ B)) = {0, 2, 4, 6} :=
by
  sorry

end complement_of_union_l155_155358


namespace complex_expression_evaluation_l155_155746

theorem complex_expression_evaluation : (i : ℂ) * (1 + i : ℂ)^2 = -2 := 
by
  sorry

end complex_expression_evaluation_l155_155746


namespace find_fx_when_x_positive_l155_155906

def isOddFunction {α : Type} [AddGroup α] [Neg α] (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)
variable (h_odd : isOddFunction f)
variable (h_neg : ∀ x : ℝ, x < 0 → f x = -x^2 + x)

theorem find_fx_when_x_positive : ∀ x : ℝ, x > 0 → f x = x^2 + x :=
by
  sorry

end find_fx_when_x_positive_l155_155906


namespace dreamy_bookstore_sales_l155_155688

theorem dreamy_bookstore_sales :
  let total_sales_percent := 100
  let notebooks_percent := 45
  let bookmarks_percent := 25
  let neither_notebooks_nor_bookmarks_percent := total_sales_percent - (notebooks_percent + bookmarks_percent)
  neither_notebooks_nor_bookmarks_percent = 30 :=
by {
  sorry
}

end dreamy_bookstore_sales_l155_155688


namespace jenna_costume_l155_155664

def cost_of_skirts (skirt_count : ℕ) (material_per_skirt : ℕ) : ℕ :=
  skirt_count * material_per_skirt

def cost_of_bodice (shirt_material : ℕ) (sleeve_material_per : ℕ) (sleeve_count : ℕ) : ℕ :=
  shirt_material + (sleeve_material_per * sleeve_count)

def total_material (skirt_material : ℕ) (bodice_material : ℕ) : ℕ :=
  skirt_material + bodice_material

def total_cost (total_material : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  total_material * cost_per_sqft

theorem jenna_costume : 
  cost_of_skirts 3 48 + cost_of_bodice 2 5 2 = 156 → total_cost 156 3 = 468 :=
by
  sorry

end jenna_costume_l155_155664


namespace find_4_digit_number_l155_155494

theorem find_4_digit_number (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182) :
  1000 * a + 100 * b + 10 * c + d = 1909 :=
by {
  sorry
}

end find_4_digit_number_l155_155494


namespace max_k_condition_l155_155737

theorem max_k_condition (k : ℕ) (total_goods : ℕ) (num_platforms : ℕ) (platform_capacity : ℕ) :
  total_goods = 1500 ∧ num_platforms = 25 ∧ platform_capacity = 80 → 
  (∀ (c : ℕ), 1 ≤ c ∧ c ≤ k → c ∣ k) → 
  (∀ (total : ℕ), total ≤ num_platforms * platform_capacity → total ≥ total_goods) → 
  k ≤ 26 := 
sorry

end max_k_condition_l155_155737


namespace gcd_polynomials_l155_155907

noncomputable def b : ℤ := sorry -- since b is given as an odd multiple of 997

theorem gcd_polynomials (h : ∃ k : ℤ, b = 997 * (2 * k + 1)) :
  Int.gcd (3 * b^2 + 41 * b + 101) (b + 17) = 1 :=
sorry

end gcd_polynomials_l155_155907


namespace jeans_vs_scarves_l155_155316

theorem jeans_vs_scarves :
  ∀ (ties belts black_shirts white_shirts : ℕ),
  ties = 34 →
  belts = 40 →
  black_shirts = 63 →
  white_shirts = 42 →
  let total_shirts := black_shirts + white_shirts in
  let jeans := (2 * total_shirts) / 3 in
  let total_ties_and_belts := ties + belts in
  let scarves := total_ties_and_belts / 2 in
  jeans - scarves = 33 :=
by
  intros ties belts black_shirts white_shirts ht hb hbs hws
  let total_shirts := black_shirts + white_shirts
  let jeans := (2 * total_shirts) / 3
  let total_ties_and_belts := ties + belts
  let scarves := total_ties_and_belts / 2
  show jeans - scarves = 33
  sorry

end jeans_vs_scarves_l155_155316


namespace power_function_value_l155_155233

variable (f : ℝ → ℝ)

-- Condition: power function passes through (1/2, 8)
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f (1 / 2) = 8

-- Question: What is f(2)?
theorem power_function_value (h : passes_through_point f) : f 2 = 1 / 8 := 
by
  sorry

end power_function_value_l155_155233


namespace estimate_y_value_at_x_equals_3_l155_155082

noncomputable def estimate_y (x : ℝ) (a : ℝ) : ℝ :=
  (1 / 3) * x + a

theorem estimate_y_value_at_x_equals_3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ) (y1 y2 y3 y4 y5 y6 y7 y8 : ℝ),
    (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 2 * (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8)) →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 8 →
    estimate_y 3 (1 / 6) = 7 / 6 := by
  intro x1 x2 x3 x4 x5 x6 x7 x8 y1 y2 y3 y4 y5 y6 y7 y8 h_sum hx
  sorry

end estimate_y_value_at_x_equals_3_l155_155082


namespace simplify_exponent_multiplication_l155_155167

theorem simplify_exponent_multiplication :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_multiplication_l155_155167


namespace sequence_sum_S15_S22_S31_l155_155644

def sequence_sum (n : ℕ) : ℤ :=
  match n with
  | 0     => 0
  | m + 1 => sequence_sum m + (-1)^m * (3 * (m + 1) - 1)

theorem sequence_sum_S15_S22_S31 :
  sequence_sum 15 + sequence_sum 22 - sequence_sum 31 = -57 := 
sorry

end sequence_sum_S15_S22_S31_l155_155644


namespace simplify_exponent_product_l155_155170

theorem simplify_exponent_product :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_product_l155_155170


namespace count_valid_subsets_l155_155767

open Set

theorem count_valid_subsets :
  ∀ (A : Set ℕ), (A ⊆ {1, 2, 3, 4, 5, 6, 7}) → 
  (∀ (a : ℕ), a ∈ A → (8 - a) ∈ A) → A ≠ ∅ → 
  ∃! (n : ℕ), n = 15 :=
  by
    sorry

end count_valid_subsets_l155_155767


namespace a_is_perfect_square_l155_155351

variable (a b : ℕ)
variable (h1 : 0 < a) 
variable (h2 : 0 < b)
variable (h3 : b % 2 = 1)
variable (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b)

theorem a_is_perfect_square (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b % 2 = 1) 
  (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b) : ∃ n : ℕ, a = n ^ 2 :=
sorry

end a_is_perfect_square_l155_155351


namespace min_value_expression_minimum_value_of_expression_l155_155811

variable (x y z : ℝ)

def positive_reals (x y z : ℝ) := (0 < x) ∧ (0 < y) ∧ (0 < z)

theorem min_value_expression (h1 : positive_reals x y z) (h2 : x + 2 * y + 3 * z = 6) :
  (1 / x + 4 / y + 9 / z) ≥ 98 / 3 :=
sorry

-- Alternatively, if we want to state the minimum value as an infimum
theorem minimum_value_of_expression (h1 : positive_reals x y z) (h2 : x + 2 * y + 3 * z = 6) :
  ∃ t : ℝ, (1 / x + 4 / y + 9 / z) = t ∧ t = 98 / 3 :=
sorry

end min_value_expression_minimum_value_of_expression_l155_155811


namespace champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l155_155651

-- Define the structure and relationship between teams in the tournament
structure Tournament (Team : Type) :=
  (competes : Team → Team → Prop) -- teams play against each other
  (no_ties : ∀ A B : Team, (competes A B ∧ ¬competes B A) ∨ (competes B A ∧ ¬competes A B)) -- no ties
  (superior : Team → Team → Prop) -- superiority relationship
  (superior_def : ∀ A B : Team, superior A B ↔ (competes A B ∧ ¬competes B A) ∨ (∃ C : Team, superior A C ∧ superior C B))

-- The main theorem based on the given questions
theorem champion_team_exists {Team : Type} (tournament : Tournament Team) :
  ∃ champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B) :=
  sorry

theorem unique_champion_wins_all {Team : Type} (tournament : Tournament Team)
  (h : ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B)) :
  ∃! champion : Team, (∀ B : Team, champion ≠ B → tournament.superior champion B ∧ tournament.competes champion B ∧ ¬tournament.competes B champion) :=
  sorry

theorem not_exactly_two_champions {Team : Type} (tournament : Tournament Team) :
  ¬∃ A B : Team, A ≠ B ∧ (∀ C : Team, C ≠ A → tournament.superior A C) ∧ (∀ C : Team, C ≠ B → tournament.superior B C) :=
  sorry

end champion_team_exists_unique_champion_wins_all_not_exactly_two_champions_l155_155651


namespace problem_I_problem_II_l155_155783

-- Problem (I)
theorem problem_I (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1|) : 
  ∀ x, (f x < |x| + 1) → (0 < x ∧ x < 2) :=
by
  intro x hx
  have fx_def : f x = |2 * x - 1| := h x
  sorry

-- Problem (II)
theorem problem_II (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1|) :
  ∀ x y, (|x - y - 1| ≤ 1 / 3) → (|2 * y + 1| ≤ 1 / 6) → (f x ≤ 5 / 6) :=
by
  intro x y hx hy
  have fx_def : f x = |2 * x - 1| := h x
  sorry

end problem_I_problem_II_l155_155783


namespace triangle_side_inequality_l155_155510

theorem triangle_side_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : 1 = 1 / 2 * b * c) : b ≥ Real.sqrt 2 :=
sorry

end triangle_side_inequality_l155_155510


namespace pages_left_in_pad_l155_155404

-- Definitions from conditions
def total_pages : ℕ := 120
def science_project_pages (total : ℕ) : ℕ := total * 25 / 100
def math_homework_pages : ℕ := 10

-- Proving the final number of pages left
theorem pages_left_in_pad :
  let remaining_pages_after_usage := total_pages - science_project_pages total_pages - math_homework_pages
  let pages_left_after_art_project := remaining_pages_after_usage / 2
  pages_left_after_art_project = 40 :=
by
  sorry

end pages_left_in_pad_l155_155404


namespace total_seeds_in_watermelons_l155_155584

def slices1 : ℕ := 40
def seeds_per_slice1 : ℕ := 60
def slices2 : ℕ := 30
def seeds_per_slice2 : ℕ := 80
def slices3 : ℕ := 50
def seeds_per_slice3 : ℕ := 40

theorem total_seeds_in_watermelons :
  (slices1 * seeds_per_slice1) + (slices2 * seeds_per_slice2) + (slices3 * seeds_per_slice3) = 6800 := by
  sorry

end total_seeds_in_watermelons_l155_155584


namespace min_value_problem_l155_155227

theorem min_value_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 4) :
  (x + 1) * (2 * y + 1) / (x * y) ≥ 9 / 2 :=
by
  sorry

end min_value_problem_l155_155227


namespace sum_primes_between_1_and_20_l155_155992

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l155_155992


namespace sale_discount_l155_155161

theorem sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (discount_multiple : ℕ)
  (h1 : purchase_amount = 250)
  (h2 : discount_per_100 = 10)
  (h3 : discount_multiple = purchase_amount / 100) :
  purchase_amount - discount_per_100 * discount_multiple = 230 := 
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end sale_discount_l155_155161


namespace proposition_q_must_be_true_l155_155795

theorem proposition_q_must_be_true (p q : Prop) (h1 : p ∨ q) (h2 : ¬ p) : q :=
by
  sorry

end proposition_q_must_be_true_l155_155795


namespace repeating_decimals_fraction_l155_155879

theorem repeating_decimals_fraction :
  (0.81:ℚ) / (0.36:ℚ) = 9 / 4 :=
by
  have h₁ : (0.81:ℚ) = 81 / 99 := sorry
  have h₂ : (0.36:ℚ) = 36 / 99 := sorry
  sorry

end repeating_decimals_fraction_l155_155879


namespace mean_median_sum_is_11_l155_155901

theorem mean_median_sum_is_11 (m n : ℕ) (h1 : m + 5 < n)
  (h2 : (m + (m + 3) + (m + 5) + n + (n + 1) + (2 * n - 1)) / 6 = n)
  (h3 : (m + 5 + n) / 2 = n) : m + n = 11 := by
  sorry

end mean_median_sum_is_11_l155_155901


namespace fraction_add_eq_l155_155430

theorem fraction_add_eq (n : ℤ) :
  (3 + n) = 4 * ((4 + n) - 5) → n = 1 := sorry

end fraction_add_eq_l155_155430


namespace angles_in_arithmetic_progression_in_cyclic_quadrilateral_angles_not_in_geometric_progression_in_cyclic_quadrilateral_l155_155747

-- Problem part (a)
theorem angles_in_arithmetic_progression_in_cyclic_quadrilateral 
  (α β γ δ : ℝ) 
  (angle_sum : α + β + γ + δ = 360) 
  (opposite_angles_sum : ∀ (α β γ δ : ℝ), α + γ = 180 ∧ β + δ = 180) 
  (arithmetic_progression : ∃ (d : ℝ) (α : ℝ), β = α + d ∧ γ = α + 2*d ∧ δ = α + 3*d ∧ d ≠ 0):
  (∃ α β γ δ, α + β + γ + δ = 360 ∧ α + γ = 180 ∧ β + δ = 180 ∧ β = α + d ∧ γ = α + 2*d ∧ δ = α + 3*d ∧ d ≠ 0) :=
sorry

-- Problem part (b)
theorem angles_not_in_geometric_progression_in_cyclic_quadrilateral 
  (α β γ δ : ℝ) 
  (angle_sum : α + β + γ + δ = 360) 
  (opposite_angles_sum : ∀ (α β γ δ : ℝ), α + γ = 180 ∧ β + δ = 180) 
  (geometric_progression : ∃ (r : ℝ) (α : ℝ), β = α * r ∧ γ = α * r^2 ∧ δ = α * r^3 ∧ r ≠ 1 ∧ r > 0):
  ¬(∃ α β γ δ, α + β + γ + δ = 360 ∧ α + γ = 180 ∧ β + δ = 180 ∧ β = α * r ∧ γ = α * r^2 ∧ δ = α * r^3 ∧ r ≠ 1) :=
sorry

end angles_in_arithmetic_progression_in_cyclic_quadrilateral_angles_not_in_geometric_progression_in_cyclic_quadrilateral_l155_155747


namespace initial_boys_count_l155_155717

theorem initial_boys_count (B : ℕ) (boys girls : ℕ)
  (h1 : boys = 3 * B)                             -- The ratio of boys to girls is 3:4
  (h2 : girls = 4 * B)                            -- The ratio of boys to girls is 3:4
  (h3 : boys - 10 = 4 * (girls - 20))             -- The final ratio after transfer is 4:5
  : boys = 90 :=                                  -- Prove initial boys count was 90
by 
  sorry

end initial_boys_count_l155_155717


namespace selection_count_l155_155653

theorem selection_count (word : String) (vowels : Finset Char) (consonants : Finset Char)
  (hword : word = "УЧЕБНИК")
  (hvowels : vowels = {'У', 'Е', 'И'})
  (hconsonants : consonants = {'Ч', 'Б', 'Н', 'К'})
  :
  vowels.card * consonants.card = 12 :=
by {
  sorry
}

end selection_count_l155_155653


namespace sum_primes_between_1_and_20_l155_155990

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_1_and_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_primes_between_1_and_20 : (primes_between_1_and_20.filter is_prime).sum = 77 := by
  sorry

end sum_primes_between_1_and_20_l155_155990


namespace river_current_speed_l155_155733

/--
Given conditions:
- The rower realized the hat was missing 15 minutes after passing under the bridge.
- The rower caught the hat 15 minutes later.
- The total distance the hat traveled from the bridge is 1 kilometer.
Prove that the speed of the river current is 2 km/h.
-/
theorem river_current_speed (t1 t2 d : ℝ) (h_t1 : t1 = 15 / 60) (h_t2 : t2 = 15 / 60) (h_d : d = 1) : 
  d / (t1 + t2) = 2 := by
sorry

end river_current_speed_l155_155733


namespace find_particular_number_l155_155040

theorem find_particular_number (A B : ℤ) (x : ℤ) (hA : A = 14) (hB : B = 24)
  (h : (((A + x) * A - B) / B = 13)) : x = 10 :=
by {
  -- You can add an appropriate lemma or proof here if necessary
  sorry
}

end find_particular_number_l155_155040


namespace complex_coordinates_l155_155255

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩

-- Define the complex number (1 + i)
def z1 : ℂ := 1 + i

-- Define the complex number i
def z2 : ℂ := i

-- The problem statement to be proven: the given complex number equals 1 - i
theorem complex_coordinates : (z1 / z2) = 1 - i :=
  sorry

end complex_coordinates_l155_155255


namespace canonical_form_lines_l155_155970

theorem canonical_form_lines (x y z : ℝ) :
  (2 * x - y + 3 * z - 1 = 0) →
  (5 * x + 4 * y - z - 7 = 0) →
  (∃ (k : ℝ), x = -11 * k ∧ y = 17 * k + 2 ∧ z = 13 * k + 1) :=
by
  intros h1 h2
  sorry

end canonical_form_lines_l155_155970


namespace percentage_discount_l155_155740

theorem percentage_discount (P S : ℝ) (hP : P = 50) (hS : S = 35) : (P - S) / P * 100 = 30 := by
  sorry

end percentage_discount_l155_155740


namespace hostel_food_duration_l155_155039

noncomputable def food_last_days (total_food_units daily_consumption_new: ℝ) : ℝ :=
  total_food_units / daily_consumption_new

theorem hostel_food_duration:
  let x : ℝ := 1 -- assuming x is a positive real number
  let men_initial := 100
  let women_initial := 100
  let children_initial := 50
  let total_days := 40
  let consumption_man := 3 * x
  let consumption_woman := 2 * x
  let consumption_child := 1 * x
  let food_sufficient_for := 250
  let total_food_units := 550 * x * 40
  let men_leave := 30
  let women_leave := 20
  let children_leave := 10
  let men_new := men_initial - men_leave
  let women_new := women_initial - women_leave
  let children_new := children_initial - children_leave
  let daily_consumption_new := 210 * x + 160 * x + 40 * x 
  (food_last_days total_food_units daily_consumption_new) = 22000 / 410 := 
by
  sorry

end hostel_food_duration_l155_155039


namespace monkey_reaches_top_l155_155864

def monkey_climb_time (tree_height : ℕ) (climb_per_hour : ℕ) (slip_per_hour : ℕ) 
  (rest_hours : ℕ) (cycle_hours : ℕ) : ℕ :=
  if (tree_height % (climb_per_hour - slip_per_hour) > climb_per_hour) 
    then (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours
    else (tree_height / (climb_per_hour - slip_per_hour)) + cycle_hours - 1

theorem monkey_reaches_top :
  monkey_climb_time 253 7 4 1 4 = 109 := 
sorry

end monkey_reaches_top_l155_155864


namespace quadratic_ineq_solution_set_l155_155927

theorem quadratic_ineq_solution_set {m : ℝ} :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
sorry

end quadratic_ineq_solution_set_l155_155927


namespace ball_arrangements_l155_155818

-- Define the structure of the boxes and balls
structure BallDistributions where
  white_balls_box1 : ℕ
  black_balls_box1 : ℕ
  white_balls_box2 : ℕ
  black_balls_box2 : ℕ
  white_balls_box3 : ℕ
  black_balls_box3 : ℕ

-- Problem conditions
def valid_distribution (d : BallDistributions) : Prop :=
  d.white_balls_box1 + d.black_balls_box1 ≥ 2 ∧
  d.white_balls_box2 + d.black_balls_box2 ≥ 2 ∧
  d.white_balls_box3 + d.black_balls_box3 ≥ 2 ∧
  d.white_balls_box1 ≥ 1 ∧
  d.black_balls_box1 ≥ 1 ∧
  d.white_balls_box2 ≥ 1 ∧
  d.black_balls_box2 ≥ 1 ∧
  d.white_balls_box3 ≥ 1 ∧
  d.black_balls_box3 ≥ 1

def total_white_balls (d : BallDistributions) : ℕ :=
  d.white_balls_box1 + d.white_balls_box2 + d.white_balls_box3

def total_black_balls (d : BallDistributions) : ℕ :=
  d.black_balls_box1 + d.black_balls_box2 + d.black_balls_box3

def correct_distribution (d : BallDistributions) : Prop :=
  total_white_balls d = 4 ∧ total_black_balls d = 5

-- Main theorem to prove
theorem ball_arrangements : ∃ (d : BallDistributions), valid_distribution d ∧ correct_distribution d ∧ (number_of_distributions = 18) :=
  sorry

end ball_arrangements_l155_155818


namespace units_digit_5_pow_2023_l155_155587

theorem units_digit_5_pow_2023 : ∀ n : ℕ, (n > 0) → (5^n % 10 = 5) → (5^2023 % 10 = 5) :=
by
  intros n hn hu
  have h_units_digit : ∀ k : ℕ, (k > 0) → 5^k % 10 = 5 := by
    intro k hk
    sorry -- pattern proof not included
  exact h_units_digit 2023 (by norm_num)

end units_digit_5_pow_2023_l155_155587


namespace smallest_value_of_Q_l155_155056

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

noncomputable def A := Q (-1)
noncomputable def B := Q (0)
noncomputable def C := (2 : ℝ)^2
def D := 1 - 2 + 3 - 4 + 5
def E := 2 -- assuming all zeros are real

theorem smallest_value_of_Q :
  min (min (min (min A B) C) D) E = 2 :=
by sorry

end smallest_value_of_Q_l155_155056


namespace sum_of_primes_between_1_and_20_l155_155996

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l155_155996


namespace floor_width_l155_155957

theorem floor_width
  (widthX lengthX : ℝ) (widthY lengthY : ℝ)
  (hX : widthX = 10) (lX : lengthX = 18) (lY : lengthY = 20)
  (h : lengthX * widthX = lengthY * widthY) :
  widthY = 9 := 
by
  -- proof goes here
  sorry

end floor_width_l155_155957


namespace triangle_side_relation_l155_155819

theorem triangle_side_relation
  (A B C : ℝ)
  (a b c : ℝ)
  (h : 3 * (Real.sin (A / 2)) * (Real.sin (B / 2)) * (Real.cos (C / 2)) + (Real.sin (3 * A / 2)) * (Real.sin (3 * B / 2)) * (Real.cos (3 * C / 2)) = 0)
  (law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  a^3 + b^3 = c^3 :=
by
  sorry

end triangle_side_relation_l155_155819


namespace smallest_non_unit_digit_multiple_of_five_l155_155294

theorem smallest_non_unit_digit_multiple_of_five :
  ∀ (d : ℕ), ((d = 0) ∨ (d = 5)) → (d ≠ 1 ∧ d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 6 ∧ d ≠ 7 ∧ d ≠ 8 ∧ d ≠ 9) :=
by {
  sorry
}

end smallest_non_unit_digit_multiple_of_five_l155_155294


namespace student_competition_distribution_l155_155218

theorem student_competition_distribution :
  ∃ f : Fin 4 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ x : Fin 4, f x = i ∧ ∃ y : Fin 4, f y = j) ∧ 
  (Finset.univ.image f).card = 3 := 
sorry

end student_competition_distribution_l155_155218


namespace simplify_exponent_multiplication_l155_155168

theorem simplify_exponent_multiplication :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end simplify_exponent_multiplication_l155_155168


namespace sum_of_lengths_of_square_sides_l155_155574

theorem sum_of_lengths_of_square_sides (side_length : ℕ) (h1 : side_length = 9) : 
  (4 * side_length) = 36 :=
by
  -- Here we would normally write the proof
  sorry

end sum_of_lengths_of_square_sides_l155_155574


namespace minuend_is_12_point_5_l155_155966

theorem minuend_is_12_point_5 (x y : ℝ) (h : x + y + (x - y) = 25) : x = 12.5 := by
  sorry

end minuend_is_12_point_5_l155_155966


namespace smallest_is_57_l155_155080

noncomputable def smallest_of_four_numbers (a b c d : ℕ) : ℕ :=
  if h1 : a + b + c = 234 ∧ a + b + d = 251 ∧ a + c + d = 284 ∧ b + c + d = 299
  then Nat.min (Nat.min a b) (Nat.min c d)
  else 0

theorem smallest_is_57 (a b c d : ℕ) (h1 : a + b + c = 234) (h2 : a + b + d = 251)
  (h3 : a + c + d = 284) (h4 : b + c + d = 299) :
  smallest_of_four_numbers a b c d = 57 :=
sorry

end smallest_is_57_l155_155080


namespace minimal_value_expression_l155_155672

theorem minimal_value_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a + (ab)^(1/3) + (abc)^(1/4)) ≥ (1/3 + 1/(3 * (3^(1/3))) + 1/(3 * (3^(1/4)))) :=
sorry

end minimal_value_expression_l155_155672


namespace verify_parabola_D_l155_155910

def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

def parabola_vertex (y : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, y x = vertex_form (-1) h k x

-- Given conditions
def h : ℝ := 2
def k : ℝ := 3

-- Possible expressions
def parabola_A (x : ℝ) : ℝ := -((x + 2)^2) - 3
def parabola_B (x : ℝ) : ℝ := -((x - 2)^2) - 3
def parabola_C (x : ℝ) : ℝ := -((x + 2)^2) + 3
def parabola_D (x : ℝ) : ℝ := -((x - 2)^2) + 3

theorem verify_parabola_D : parabola_vertex parabola_D 2 3 :=
by
  -- Placeholder for the proof
  sorry

end verify_parabola_D_l155_155910


namespace minjeong_walk_distance_l155_155267

noncomputable def park_side_length : ℕ := 40
noncomputable def square_sides : ℕ := 4

theorem minjeong_walk_distance (side_length : ℕ) (sides : ℕ) (h : side_length = park_side_length) (h2 : sides = square_sides) : 
  side_length * sides = 160 := by
  sorry

end minjeong_walk_distance_l155_155267


namespace expected_value_sum_of_two_marbles_l155_155787

open Finset

-- Define the set of 5 marbles
def marbles : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the pairs of marbles
def marble_pairs := marbles.powerset.filter (λ s, s.card = 2)

-- Calculate the sum of the pairs
def pair_sums : Finset ℕ := marble_pairs.image (λ s, s.sum id)

-- Calculate the expected value
def expected_value : ℚ := (pair_sums.sum id : ℚ) / marble_pairs.card

/-- The expected value of the sum of the numbers on two different marbles drawn at random from 5 marbles numbered 1 through 5 is 6. -/
theorem expected_value_sum_of_two_marbles : expected_value = 6 := by
  sorry

end expected_value_sum_of_two_marbles_l155_155787


namespace domain_of_h_l155_155827

def domain_f : Set ℝ := {x | -10 ≤ x ∧ x ≤ 3}

def h_dom := {x | -3 * x ∈ domain_f}

theorem domain_of_h :
  h_dom = {x | x ≥ 10 / 3} :=
by
  sorry

end domain_of_h_l155_155827


namespace train_travel_section_marked_l155_155837

-- Definition of the metro structure with the necessary conditions.
structure Metro (Station : Type) :=
  (lines : List (Station × Station))
  (travel_time : Station → Station → ℕ)
  (terminal_turnaround : Station → Station)
  (transfer_station : Station → Station)

variable {Station : Type}

/-- The function that defines the bipolar coloring of the metro stations. -/
def station_color (s : Station) : ℕ := sorry  -- Placeholder for actual coloring function.

theorem train_travel_section_marked 
  (metro : Metro Station)
  (initial_station : Station)
  (end_station : Station)
  (travel_time : ℕ)
  (marked_section : Station × Station)
  (h_start : initial_station = marked_section.fst)
  (h_end : end_station = marked_section.snd)
  (h_travel_time : travel_time = 2016)
  (h_condition : ∀ s1 s2, (s1, s2) ∈ metro.lines → metro.travel_time s1 s2 = 1 ∧ 
                metro.terminal_turnaround s1 ≠ s1 ∧ metro.transfer_station s1 ≠ s2) :
  ∃ (time : ℕ), time = 2016 ∧ ∃ s1 s2, (s1, s2) = marked_section :=
sorry

end train_travel_section_marked_l155_155837


namespace sum_of_primes_between_1_and_20_l155_155995

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l155_155995


namespace online_sale_discount_l155_155159

theorem online_sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (total_paid : ℕ) : 
  purchase_amount = 250 → 
  discount_per_100 = 10 → 
  total_paid = purchase_amount - (purchase_amount / 100) * discount_per_100 → 
  total_paid = 230 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end online_sale_discount_l155_155159


namespace total_presents_l155_155283

-- Definitions based on the problem conditions
def numChristmasPresents : ℕ := 60
def numBirthdayPresents : ℕ := numChristmasPresents / 2

-- Theorem statement
theorem total_presents : numChristmasPresents + numBirthdayPresents = 90 :=
by
  -- Proof is omitted
  sorry

end total_presents_l155_155283


namespace sum_of_primes_1_to_20_l155_155987

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l155_155987


namespace video_views_l155_155048

theorem video_views (initial_views : ℕ) (increase_factor : ℕ) (additional_views : ℕ) :
  initial_views = 4000 →
  increase_factor = 10 →
  additional_views = 50000 →
  let views_after_4_days := initial_views + increase_factor * initial_views in
  let total_views := views_after_4_days + additional_views in
  total_views = 94000 :=
by
  intros h_initial_views h_increase_factor h_additional_views
  have views_after_4_days_def : views_after_4_days = initial_views + increase_factor * initial_views
  have total_views_def : total_views = views_after_4_days + additional_views
  rw [h_initial_views, h_increase_factor, h_additional_views]
  rw [views_after_4_days_def, total_views_def]
  sorry

end video_views_l155_155048


namespace problem_solution_l155_155213

theorem problem_solution (x : ℝ) : 
  (x < -2 ∨ (-2 < x ∧ x ≤ 0) ∨ (0 < x ∧ x < 2) ∨ (2 ≤ x ∧ x < (15 - Real.sqrt 257) / 8) ∨ ((15 + Real.sqrt 257) / 8 < x)) ↔ 
  (x^2 - 1) / (x + 2) ≥ 3 / (x - 2) + 7 / 4 := sorry

end problem_solution_l155_155213


namespace balls_into_boxes_l155_155786

theorem balls_into_boxes : (4 ^ 5 = 1024) :=
by
  -- The proof is omitted; the statement is required
  sorry

end balls_into_boxes_l155_155786


namespace total_views_l155_155047

def first_day_views : ℕ := 4000
def views_after_4_days : ℕ := 40000 + first_day_views
def views_after_6_days : ℕ := views_after_4_days + 50000

theorem total_views : views_after_6_days = 94000 := by
  have h1 : first_day_views = 4000 := rfl
  have h2 : views_after_4_days = 40000 + first_day_views := rfl
  have h3 : views_after_6_days = views_after_4_days + 50000 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end total_views_l155_155047


namespace solve_linear_system_l155_155275

open Matrix

-- Define a namespace
namespace LinearSystem

-- Define the system of linear equations in matrix form
def A : Matrix (Fin 4) (Fin 4) ℚ := ![
  ![-1, -2, -6, 3],
  ![2, 5, 14, -7],
  ![3, 7, 20, -10],
  ![0, -1, -2, 1]
]

def b : Vector ℚ 4 := ![-1, 3, 4, -1]

-- Define the parametric solution
def x (α β : ℚ) : Vector ℚ 4 := ![
  -1 - 2 * α + β,
  1 - 2 * α + β,
  α,
  β
]

-- The main theorem to prove
theorem solve_linear_system : 
  ∃ (α β : ℚ), A.mulVec (x α β) = b := 
by
  sorry 

end LinearSystem

end solve_linear_system_l155_155275


namespace max_profit_at_nine_l155_155518

noncomputable def profit (x : ℝ) : ℝ := - (1 / 3) * x^3 + 81 * x - 23

theorem max_profit_at_nine :
  ∃ x : ℝ, x = 9 ∧ ∀ (ε : ℝ), ε > 0 → 
  (profit (9 - ε) < profit 9 ∧ profit (9 + ε) < profit 9) := 
by
  sorry

end max_profit_at_nine_l155_155518


namespace find_positive_integers_l155_155068

theorem find_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^2 - Nat.factorial y = 2019 ↔ x = 45 ∧ y = 3 :=
by
  sorry

end find_positive_integers_l155_155068


namespace number_of_solutions_l155_155107

theorem number_of_solutions :
  (∃ (a b c : ℕ), 4 * a = 6 * c ∧ 168 * a = 6 * a * b * c) → 
  ∃ (s : Finset ℕ), s.card = 6 :=
by sorry

end number_of_solutions_l155_155107


namespace wheat_grains_approximation_l155_155800

theorem wheat_grains_approximation :
  let total_grains : ℕ := 1536
  let wheat_per_sample : ℕ := 28
  let sample_size : ℕ := 224
  let wheat_estimate : ℕ := total_grains * wheat_per_sample / sample_size
  wheat_estimate = 169 := by
  sorry

end wheat_grains_approximation_l155_155800


namespace pregnant_dogs_count_l155_155620

-- Definitions as conditions stated in the problem
def total_puppies (P : ℕ) : ℕ := 4 * P
def total_shots (P : ℕ) : ℕ := 2 * total_puppies P
def total_cost (P : ℕ) : ℕ := total_shots P * 5

-- Proof statement without proof
theorem pregnant_dogs_count : ∃ P : ℕ, total_cost P = 120 → P = 3 :=
by sorry

end pregnant_dogs_count_l155_155620


namespace chris_sick_weeks_l155_155882

theorem chris_sick_weeks :
  ∀ (h1 : ∀ w : ℕ, w = 4 → 2 * w = 8),
    ∀ (h2 : ∀ h w : ℕ, h = 20 → ∀ m : ℕ, 2 * (w * m) = 160),
    ∀ (h3 : ∀ h : ℕ, h = 180 → 180 - 160 = 20),
    ∀ (h4 : ∀ h w : ℕ, h = 20 → w = 20 → 20 / 20 = 1),
    180 - 160 = (20 / 20) * 20 :=
by
  intros
  sorry

end chris_sick_weeks_l155_155882


namespace sculptures_not_on_display_eq_1200_l155_155873

-- Define the number of pieces of art in the gallery
def total_pieces_art := 2700

-- Define the number of pieces on display (1/3 of total pieces)
def pieces_on_display := total_pieces_art / 3

-- Define the number of pieces not on display
def pieces_not_on_display := total_pieces_art - pieces_on_display

-- Define the number of sculptures on display (1/6 of pieces on display)
def sculptures_on_display := pieces_on_display / 6

-- Define the number of paintings not on display (1/3 of pieces not on display)
def paintings_not_on_display := pieces_not_on_display / 3

-- Prove the number of sculptures not on display
theorem sculptures_not_on_display_eq_1200 :
  total_pieces_art = 2700 →
  pieces_on_display = total_pieces_art / 3 →
  pieces_not_on_display = total_pieces_art - pieces_on_display →
  sculptures_on_display = pieces_on_display / 6 →
  paintings_not_on_display = pieces_not_on_display / 3 →
  pieces_not_on_display - paintings_not_on_display = 1200 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end sculptures_not_on_display_eq_1200_l155_155873


namespace solve_eqn_l155_155683

theorem solve_eqn (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  3 ^ x = 2 ^ x * y + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) := by
  sorry

end solve_eqn_l155_155683


namespace sector_central_angle_l155_155909

-- Definitions and constants
def arc_length := 4 -- arc length of the sector in cm
def area := 2       -- area of the sector in cm²

-- The central angle of the sector we want to prove
def theta := 4      -- radian measure of the central angle

-- Main statement to prove
theorem sector_central_angle : 
  ∃ (r : ℝ), (1 / 2) * theta * r^2 = area ∧ theta * r = arc_length :=
by
  -- No proof is required as per the instruction
  sorry

end sector_central_angle_l155_155909


namespace sum_primes_between_1_and_20_l155_155982

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l155_155982


namespace slices_with_both_onions_and_olives_l155_155320

noncomputable def slicesWithBothToppings (total_slices slices_with_onions slices_with_olives : Nat) : Nat :=
  slices_with_onions + slices_with_olives - total_slices

theorem slices_with_both_onions_and_olives 
  (total_slices : Nat) (slices_with_onions : Nat) (slices_with_olives : Nat) :
  total_slices = 18 ∧ slices_with_onions = 10 ∧ slices_with_olives = 10 →
  slicesWithBothToppings total_slices slices_with_onions slices_with_olives = 2 :=
by
  sorry

end slices_with_both_onions_and_olives_l155_155320


namespace largest_four_digit_divisible_by_6_l155_155425

theorem largest_four_digit_divisible_by_6 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 6 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 6 = 0 → m ≤ n :=
begin
  use 9996,
  split,
  { exact nat.le_refl 9996 },
  split,
  { dec_trivial },
  split,
  { exact nat.zero_mod _ },
  { intros m h1 h2 h3,
    exfalso,
    sorry }
end

end largest_four_digit_divisible_by_6_l155_155425


namespace pentagon_coloring_l155_155075

theorem pentagon_coloring (convex : Prop) (unequal_sides : Prop)
  (colors : Prop) (adjacent_diff_color : Prop) :
  ∃ n : ℕ, n = 30 := by
  -- Definitions for conditions (in practical terms, these might need to be more elaborate)
  let convex := true           -- Simplified representation
  let unequal_sides := true    -- Simplified representation
  let colors := true           -- Simplified representation
  let adjacent_diff_color := true -- Simplified representation
  
  -- Proof that the number of coloring methods is 30
  existsi 30
  sorry

end pentagon_coloring_l155_155075


namespace system_consistent_and_solution_l155_155344

theorem system_consistent_and_solution (a x : ℝ) : 
  (a = -10 ∧ x = -1/3) ∨ (a = -8 ∧ x = -1) ∨ (a = 4 ∧ x = -2) ↔ 
  3 * x^2 - x - a - 10 = 0 ∧ (a + 4) * x + a + 12 = 0 := by
  sorry

end system_consistent_and_solution_l155_155344


namespace total_outfits_l155_155437

-- Define the quantities of each item.
def red_shirts : ℕ := 7
def green_shirts : ℕ := 8
def pants : ℕ := 10
def blue_hats : ℕ := 10
def red_hats : ℕ := 10
def scarves : ℕ := 5

-- The total number of outfits without having the same color of shirts and hats.
theorem total_outfits : 
  (red_shirts * pants * blue_hats * scarves) + (green_shirts * pants * red_hats * scarves) = 7500 := 
by sorry

end total_outfits_l155_155437


namespace extreme_points_range_of_a_l155_155782

noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2

-- Problem 1: Extreme points
theorem extreme_points (a : ℝ) : 
  (a ≤ 0 → ∃! x, ∀ y, f y a ≤ f x a) ∧
  (0 < a ∧ a < 1/2 → ∃ x1 x2, x1 ≠ x2 ∧ ∀ y, f y a ≤ f x1 a ∨ f y a ≤ f x2 a) ∧
  (a = 1/2 → ∀ x y, f y a ≤ f x a → x = y) ∧
  (a > 1/2 → ∃ x1 x2, x1 ≠ x2 ∧ ∀ y, f y a ≤ f x1 a ∨ f y a ≤ f x2 a) :=
sorry

-- Problem 2: Range of values for 'a'
theorem range_of_a (a : ℝ) : 
  (∀ x, f x a + Real.exp x ≥ x^3 + x) ↔ (a ≤ Real.exp 1 - 2) :=
sorry

end extreme_points_range_of_a_l155_155782


namespace revision_cost_per_page_is_4_l155_155408

-- Definitions based on conditions
def initial_cost_per_page := 6
def total_pages := 100
def revised_once_pages := 35
def revised_twice_pages := 15
def no_revision_pages := total_pages - revised_once_pages - revised_twice_pages
def total_cost := 860

-- Theorem to be proved
theorem revision_cost_per_page_is_4 : 
  ∃ x : ℝ, 
    ((initial_cost_per_page * total_pages) + 
     (revised_once_pages * x) + 
     (revised_twice_pages * (2 * x)) = total_cost) ∧ x = 4 :=
by
  sorry

end revision_cost_per_page_is_4_l155_155408


namespace simplified_expression_value_l155_155006

theorem simplified_expression_value (x : ℝ) (h : x = -2) :
  (x - 2)^2 - 4 * x * (x - 1) + (2 * x + 1) * (2 * x - 1) = 7 := 
  by
    -- We are given x = -2
    simp [h]
    -- sorry added to skip the actual solution in Lean
    sorry

end simplified_expression_value_l155_155006


namespace non_congruent_rectangles_count_l155_155183

theorem non_congruent_rectangles_count :
  (∃ (l w : ℕ), l + w = 50 ∧ l ≠ w) ∧
  (∀ (l w : ℕ), l + w = 50 ∧ l ≠ w → l > w) →
  (∃ (n : ℕ), n = 24) :=
by
  sorry

end non_congruent_rectangles_count_l155_155183


namespace inequality_solution_l155_155355

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + (a - 1) < 0) ↔ a < -1/3 :=
by
  sorry

end inequality_solution_l155_155355


namespace sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l155_155716

-- Define a convex n-gon and prove that the sum of its interior angles is (n-2) * 180 degrees
theorem sum_interior_angles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (sum_of_angles : ℝ), sum_of_angles = (n-2) * 180 :=
sorry

-- Define a convex n-gon and prove that the number of triangles formed by dividing with non-intersecting diagonals is n-2
theorem number_of_triangles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (num_of_triangles : ℕ), num_of_triangles = n-2 :=
sorry

end sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l155_155716


namespace barbara_total_cost_l155_155476

-- Definitions based on the given conditions
def steak_cost_per_pound : ℝ := 15.00
def steak_quantity : ℝ := 4.5
def chicken_cost_per_pound : ℝ := 8.00
def chicken_quantity : ℝ := 1.5

def expected_total_cost : ℝ := 42.00

-- The main proposition we need to prove
theorem barbara_total_cost :
  steak_cost_per_pound * steak_quantity + chicken_cost_per_pound * chicken_quantity = expected_total_cost :=
by
  sorry

end barbara_total_cost_l155_155476


namespace larger_number_is_70380_l155_155306

theorem larger_number_is_70380 (A B : ℕ) 
    (hcf : Nat.gcd A B = 20) 
    (lcm : Nat.lcm A B = 20 * 9 * 17 * 23) :
    max A B = 70380 :=
  sorry

end larger_number_is_70380_l155_155306


namespace blue_to_red_marble_ratio_l155_155291

-- Define the given conditions and the result.
theorem blue_to_red_marble_ratio (total_marble yellow_marble : ℕ) 
  (h1 : total_marble = 19)
  (h2 : yellow_marble = 5)
  (red_marble : ℕ)
  (h3 : red_marble = yellow_marble + 3) : 
  ∃ blue_marble : ℕ, (blue_marble = total_marble - (yellow_marble + red_marble)) 
  ∧ (blue_marble / (gcd blue_marble red_marble)) = 3 
  ∧ (red_marble / (gcd blue_marble red_marble)) = 4 :=
by {
  --existence of blue_marble and the ratio
  sorry
}

end blue_to_red_marble_ratio_l155_155291


namespace sum_of_primes_1_to_20_l155_155988

-- Definition of prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of range from 1 to 20
def primes_in_range : List ℕ :=
  List.filter is_prime (List.range' 1 20) 

-- Sum of the prime numbers in the range 1 to 20
def sum_primes_in_range : ℕ :=
  primes_in_range.sum

-- Statement of the problem: sum of all prime numbers between 1 and 20 is 77
theorem sum_of_primes_1_to_20 : sum_primes_in_range = 77 := by
  sorry

end sum_of_primes_1_to_20_l155_155988


namespace x_equals_1_over_16_l155_155801

-- Given conditions
def distance_center_to_tangents_intersection : ℚ := 3 / 8
def radius_of_circle : ℚ := 3 / 16
def distance_center_to_CD : ℚ := 1 / 2

-- Calculated total distance
def total_distance_center_to_C : ℚ := distance_center_to_tangents_intersection + radius_of_circle

-- Problem statement
theorem x_equals_1_over_16 (x : ℚ) 
    (h : total_distance_center_to_C = x + distance_center_to_CD) : 
    x = 1 / 16 := 
by
  -- Proof is omitted, based on the provided solution steps
  sorry

end x_equals_1_over_16_l155_155801


namespace calculation_l155_155926

def operation_e (x y z : ℕ) : ℕ := 3 * x * y * z

theorem calculation :
  operation_e 3 (operation_e 4 5 6) 1 = 3240 :=
by
  sorry

end calculation_l155_155926


namespace div_expression_l155_155744

theorem div_expression : (124 : ℝ) / (8 + 14 * 3) = 2.48 := by
  sorry

end div_expression_l155_155744


namespace largest_four_digit_number_divisible_by_6_l155_155424

theorem largest_four_digit_number_divisible_by_6 :
  ∃ n, n = 9996 ∧ ∀ m, (m ≤ 9999 ∧ m % 6 = 0) → m ≤ n :=
begin
  sorry
end

end largest_four_digit_number_divisible_by_6_l155_155424


namespace sufficient_but_not_necessary_condition_l155_155832

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  ((x + 1) * (x - 3) < 0 → x > -1) ∧ ¬ (x > -1 → (x + 1) * (x - 3) < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l155_155832


namespace pizza_slices_left_l155_155592

-- Lean definitions for conditions
def total_slices : ℕ := 24
def slices_eaten_dinner : ℕ := total_slices / 3
def slices_after_dinner : ℕ := total_slices - slices_eaten_dinner

def slices_eaten_yves : ℕ := slices_after_dinner / 5
def slices_after_yves : ℕ := slices_after_dinner - slices_eaten_yves

def slices_eaten_oldest_siblings : ℕ := 3 * 3
def slices_after_oldest_siblings : ℕ := slices_after_yves - slices_eaten_oldest_siblings

def num_remaining_siblings : ℕ := 7 - 3
def slices_eaten_remaining_siblings : ℕ := num_remaining_siblings * 2
def slices_final : ℕ := if slices_after_oldest_siblings < slices_eaten_remaining_siblings then 0 else slices_after_oldest_siblings - slices_eaten_remaining_siblings

-- Proposition to prove
theorem pizza_slices_left : slices_final = 0 := by sorry

end pizza_slices_left_l155_155592


namespace chimney_problem_l155_155330

variable (x : ℕ) -- number of bricks in the chimney
variable (t : ℕ)
variables (brenda_hours brandon_hours : ℕ)

def brenda_rate := x / brenda_hours
def brandon_rate := x / brandon_hours
def combined_rate := (brenda_rate + brandon_rate - 15) * t

theorem chimney_problem (h1 : brenda_hours = 9)
    (h2 : brandon_hours = 12)
    (h3 : t = 6)
    (h4 : combined_rate = x) : x = 540 := sorry

end chimney_problem_l155_155330


namespace polynomial_divisibility_l155_155204

theorem polynomial_divisibility (A B : ℝ) 
    (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^(205 : ℕ) + A * x + B = 0) : 
    A + B = -1 :=
by
  sorry

end polynomial_divisibility_l155_155204


namespace area_of_triangle_AMN_l155_155455

theorem area_of_triangle_AMN
  (α : ℝ) -- Angle at vertex A
  (S : ℝ) -- Area of triangle ABC
  (area_AMN_eq : ∀ (α : ℝ) (S : ℝ), ∃ (area_AMN : ℝ), area_AMN = S * (Real.cos α)^2) :
  ∃ area_AMN, area_AMN = S * (Real.cos α)^2 := by
  sorry

end area_of_triangle_AMN_l155_155455


namespace no_rel_prime_a_b_c_div_conditions_l155_155889

open Nat

theorem no_rel_prime_a_b_c_div_conditions :
  ∀ a b c : ℕ, (Nat.coprime a b) ∧ (Nat.coprime b c) ∧ (Nat.coprime c a) →
  (a + b) ∣ (c^2) → (b + c) ∣ (a^2) → (c + a) ∣ (b^2) → False :=
by
  intro a b c h_coprime h1 h2 h3
  sorry

end no_rel_prime_a_b_c_div_conditions_l155_155889


namespace find_blue_sea_glass_pieces_l155_155052

-- Define all required conditions and the proof problem.
theorem find_blue_sea_glass_pieces (B : ℕ) : 
  let BlancheRed := 3
  let RoseRed := 9
  let DorothyRed := 2 * (BlancheRed + RoseRed)
  let DorothyBlue := 3 * B
  let DorothyTotal := 57
  DorothyTotal = DorothyRed + DorothyBlue → B = 11 :=
by {
  sorry
}

end find_blue_sea_glass_pieces_l155_155052


namespace minimum_seats_occupied_l155_155155

-- Define the conditions
def initial_seat_count : Nat := 150
def people_initially_leaving_up_to_two_empty_seats := true
def eventually_rule_changes_to_one_empty_seat := true

-- Define the function which checks the minimum number of occupied seats needed
def fewest_occupied_seats (total_seats : Nat) (initial_rule : Bool) (final_rule : Bool) : Nat :=
  if initial_rule && final_rule && total_seats = 150 then 57 else 0

-- The main theorem we need to prove
theorem minimum_seats_occupied {total_seats : Nat} : 
  total_seats = initial_seat_count → 
  people_initially_leaving_up_to_two_empty_seats → 
  eventually_rule_changes_to_one_empty_seat → 
  fewest_occupied_seats total_seats people_initially_leaving_up_to_two_empty_seats eventually_rule_changes_to_one_empty_seat = 57 :=
by
  intro h1 h2 h3
  sorry

end minimum_seats_occupied_l155_155155


namespace no_adjacent_same_color_probability_zero_l155_155501

-- Define the number of each color bead
def num_red_beads : ℕ := 5
def num_white_beads : ℕ := 3
def num_blue_beads : ℕ := 2

-- Define the total number of beads
def total_beads : ℕ := num_red_beads + num_white_beads + num_blue_beads

-- Calculate the probability that no two neighboring beads are the same color
noncomputable def probability_no_adjacent_same_color : ℚ :=
  if (num_red_beads > num_white_beads + num_blue_beads + 1) then 0 else sorry

theorem no_adjacent_same_color_probability_zero :
  probability_no_adjacent_same_color = 0 :=
by {
  sorry
}

end no_adjacent_same_color_probability_zero_l155_155501


namespace rectangle_area_l155_155142

noncomputable def side_of_square : ℝ := Real.sqrt 625

noncomputable def radius_of_circle : ℝ := side_of_square

noncomputable def length_of_rectangle : ℝ := (2 / 5) * radius_of_circle

def breadth_of_rectangle : ℝ := 10

theorem rectangle_area :
  length_of_rectangle * breadth_of_rectangle = 100 := 
by
  simp [length_of_rectangle, breadth_of_rectangle, radius_of_circle, side_of_square]
  sorry

end rectangle_area_l155_155142


namespace volume_of_pure_water_added_l155_155597

theorem volume_of_pure_water_added 
  (V0 : ℝ) (P0 : ℝ) (Pf : ℝ) 
  (V0_eq : V0 = 50) 
  (P0_eq : P0 = 0.30) 
  (Pf_eq : Pf = 0.1875) : 
  ∃ V : ℝ, V = 30 ∧ (15 / (V0 + V)) = Pf := 
by
  sorry

end volume_of_pure_water_added_l155_155597


namespace find_abc_and_sqrt_l155_155774

theorem find_abc_and_sqrt (a b c : ℤ) (h1 : 3 * a - 2 * b - 1 = 9) (h2 : a + 2 * b = -8) (h3 : c = Int.floor (2 + Real.sqrt 7)) :
  a = 2 ∧ b = -2 ∧ c = 4 ∧ (Real.sqrt (a - b + c) = 2 * Real.sqrt 2 ∨ Real.sqrt (a - b + c) = -2 * Real.sqrt 2) :=
by
  -- proof details go here
  sorry

end find_abc_and_sqrt_l155_155774


namespace find_x_l155_155542

theorem find_x (x : ℝ) (h : 6 * x + 3 * x + 4 * x + 2 * x = 360) : x = 24 :=
sorry

end find_x_l155_155542


namespace marathons_total_distance_l155_155317

theorem marathons_total_distance :
  ∀ (m y : ℕ),
  (26 + 385 / 1760 : ℕ) = 26 ∧ 385 % 1760 = 385 →
  15 * 26 + 15 * 385 / 1760 = m + 495 / 1760 ∧
  15 * 385 % 1760 = 495 →
  0 ≤ 495 ∧ 495 < 1760 →
  y = 495 := by
  intros
  sorry

end marathons_total_distance_l155_155317


namespace larger_integer_value_l155_155285

-- Define the conditions as Lean definitions
def quotient_condition (a b : ℕ) : Prop := a / b = 5 / 2
def product_condition (a b : ℕ) : Prop := a * b = 160
def larger_integer (a b : ℕ) : ℕ := if a > b then a else b

-- State the theorem with conditions and expected outcome
theorem larger_integer_value (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) :
  larger_integer a b = 20 :=
sorry -- Proof to be provided

end larger_integer_value_l155_155285


namespace find_F1C_CG1_l155_155954

variable {A B C D E F G H E1 F1 G1 H1 : Type*}
variables (AE EB BF FC CG GD DH HA E1A AH1 F1C CG1 : ℝ) (a : ℝ)

axiom convex_quadrilateral (AE EB BF FC CG GD DH HA : ℝ) : 
  AE / EB * BF / FC * CG / GD * DH / HA = 1 

axiom quadrilaterals_similar 
  (E1F1 EF F1G1 FG G1H1 GH H1E1 HE : Prop) :
  E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True)

axiom given_ratio (E1A AH1 : ℝ) (a : ℝ) :
  E1A / AH1 = a

theorem find_F1C_CG1
  (conv : AE / EB * BF / FC * CG / GD * DH / HA = 1)
  (parallel_lines : E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True))
  (ratio : E1A / AH1 = a) :
  F1C / CG1 = a := 
sorry

end find_F1C_CG1_l155_155954


namespace true_compound_propositions_l155_155768

-- Definitions of the propositions
def p1 := ∀ x : ℝ, 0 < x → 3^x > 2^x
def p2 := ∃ θ : ℝ, sin θ + cos θ = 3 / 2

def q1 := p1 ∨ p2
def q2 := p1 ∧ p2
def q3 := ¬p1 ∨ p2
def q4 := p1 ∧ ¬p2

-- Theorem stating the desired truth values
theorem true_compound_propositions : q1 ∧ q4 :=
by {
  have h_p1 : p1 := ... -- Proof that p1 is true, left as exercise
  have h_np2 : ¬p2 := ... -- Proof that p2 is false, left as exercise
  have h_q1 : q1 := Or.inl h_p1, -- Using h_p1 to show q1 is true
  have h_q4 : q4 := And.intro h_p1 h_np2, -- Using h_p1 and h_np2 to show q4 is true
  exact And.intro h_q1 h_q4
}

end true_compound_propositions_l155_155768


namespace problem_expression_value_l155_155335

theorem problem_expression_value :
  (100 - (3010 - 301)) + (3010 - (301 - 100)) = 200 :=
by
  sorry

end problem_expression_value_l155_155335


namespace exists_inverse_C_l155_155545

open Matrix

def matrix_2_z := Matrix (Fin 2) (Fin 2) ℤ

variables {A C : matrix_2_z} {I : matrix_2_z}

theorem exists_inverse_C (hA : A ^ 2 + (5 : ℤ) • (1 : matrix_2_z) = 0) :
  ∃ C : matrix_2_z, invertible C ∧ (A = C⁻¹ • (matrix_of_fun ![![1, 2], ![-3, -1]]) • C ∨
                                    A = C⁻¹ • (matrix_of_fun ![![0, 1], ![-5, 0]]) • C) :=
sorry

end exists_inverse_C_l155_155545


namespace range_of_a_l155_155081

def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 2) * x + 5

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 4 → f x a ≤ f (x+1) a) : a ≥ -2 := 
by
  sorry

end range_of_a_l155_155081


namespace original_stone_count_145_l155_155687

theorem original_stone_count_145 : 
  ∃ (n : ℕ), (n ≡ 1 [MOD 18]) ∧ (n = 145) :=
by
  sorry

end original_stone_count_145_l155_155687


namespace total_gallons_needed_l155_155648

def gas_can_capacity : ℝ := 5.0
def number_of_cans : ℝ := 4.0
def total_gallons_of_gas : ℝ := gas_can_capacity * number_of_cans

theorem total_gallons_needed : total_gallons_of_gas = 20.0 := by
  -- proof goes here
  sorry

end total_gallons_needed_l155_155648


namespace inequality_proof_equality_condition_l155_155550

variable {x y z : ℝ}

def positive_reals (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0

theorem inequality_proof (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry -- Proof goes here

theorem equality_condition (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * z ∧ y = z :=
sorry -- Proof goes here

end inequality_proof_equality_condition_l155_155550


namespace uber_profit_l155_155434

-- Define conditions
def income : ℕ := 30000
def initial_cost : ℕ := 18000
def trade_in : ℕ := 6000

-- Define depreciation cost
def depreciation_cost : ℕ := initial_cost - trade_in

-- Define the profit
def profit : ℕ := income - depreciation_cost

-- The theorem to be proved
theorem uber_profit : profit = 18000 := by 
  sorry

end uber_profit_l155_155434


namespace lcm_36_90_eq_180_l155_155071

theorem lcm_36_90_eq_180 : Nat.lcm 36 90 = 180 := 
by 
  sorry

end lcm_36_90_eq_180_l155_155071


namespace trapezoid_side_length_l155_155188

theorem trapezoid_side_length (s : ℝ) (A : ℝ) (x : ℝ) (y : ℝ) :
  s = 1 ∧ A = 1 ∧ y = 1/2 ∧ (1/2) * ((x + y) * y) = 1/4 → x = 1/2 :=
by
  intro h
  rcases h with ⟨hs, hA, hy, harea⟩
  sorry

end trapezoid_side_length_l155_155188


namespace find_f_log_value_l155_155640

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 1 then 2^x + 1 else sorry

theorem find_f_log_value (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_spec : ∀ x, 0 < x → x < 1 → f x = 2^x + 1) :
  f (Real.logb (1/2) (1/15)) = -31/15 :=
sorry

end find_f_log_value_l155_155640


namespace sum_of_three_numbers_is_71_point_5_l155_155248

noncomputable def sum_of_three_numbers (a b c : ℝ) : ℝ :=
a + b + c

theorem sum_of_three_numbers_is_71_point_5 (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 48) (h3 : c + a = 60) :
  sum_of_three_numbers a b c = 71.5 :=
by
  unfold sum_of_three_numbers
  sorry

end sum_of_three_numbers_is_71_point_5_l155_155248


namespace probability_three_defective_before_two_good_correct_l155_155725

noncomputable def probability_three_defective_before_two_good 
  (total_items : ℕ) 
  (good_items : ℕ) 
  (defective_items : ℕ) 
  (sequence_length : ℕ) : ℚ := 
  -- We will skip the proof part and just acknowledge the result as mentioned
  (1 / 55 : ℚ)

theorem probability_three_defective_before_two_good_correct :
  probability_three_defective_before_two_good 12 9 3 5 = 1 / 55 := 
by sorry

end probability_three_defective_before_two_good_correct_l155_155725


namespace intersection_of_sets_l155_155944

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x | x > 2 ∨ x < 1}

theorem intersection_of_sets :
  (A ∪ B) ∩ C = {0, 3, 4} :=
by
  sorry

end intersection_of_sets_l155_155944


namespace evaluate_f_at_neg_three_l155_155788

def f (x : ℝ) : ℝ := 4 * x - 2

theorem evaluate_f_at_neg_three : f (-3) = -14 := by
  sorry

end evaluate_f_at_neg_three_l155_155788


namespace train_cross_time_l155_155721

open Real

noncomputable def length_train1 := 190 -- in meters
noncomputable def length_train2 := 160 -- in meters
noncomputable def speed_train1 := 60 * (5/18) --speed_kmhr_to_msec 60 km/hr to m/s
noncomputable def speed_train2 := 40 * (5/18) -- speed_kmhr_to_msec 40 km/hr to m/s
noncomputable def relative_speed := speed_train1 + speed_train2 -- relative speed

theorem train_cross_time :
  (length_train1 + length_train2) / relative_speed = 350 / ((60 * (5/18)) + (40 * (5/18))) :=
by
  sorry -- The proof will be here initially just to validate the Lean statement

end train_cross_time_l155_155721


namespace age_difference_l155_155853

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 12) : A - C = 12 :=
sorry

end age_difference_l155_155853


namespace symmetric_point_m_eq_one_l155_155017

theorem symmetric_point_m_eq_one (m : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (-3, -1))
  (symmetric : A.1 = B.1 ∧ A.2 = -B.2) : 
  m = 1 :=
by
  sorry

end symmetric_point_m_eq_one_l155_155017


namespace find_number_l155_155532

theorem find_number (x : ℚ) : (x + (-5/12) - (-5/2) = 1/3) → x = -7/4 :=
by
  sorry

end find_number_l155_155532


namespace problem_m_n_sum_l155_155448

theorem problem_m_n_sum (m n : ℕ) 
  (h1 : m^2 + n^2 = 3789) 
  (h2 : Nat.gcd m n + Nat.lcm m n = 633) : 
  m + n = 87 :=
sorry

end problem_m_n_sum_l155_155448


namespace find_xyz_l155_155517

theorem find_xyz (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 45) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15) (h3 : x + y + z = 5) : x * y * z = 10 :=
by
  sorry

end find_xyz_l155_155517


namespace find_intervals_of_monotonicity_find_value_of_a_l155_155505

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + a + 1

theorem find_intervals_of_monotonicity (k : ℤ) (a : ℝ) :
  ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3), MonotoneOn (λ x => f x a) (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) :=
sorry

theorem find_value_of_a (a : ℝ) (max_value_condition : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) :
  a = 1 :=
sorry

end find_intervals_of_monotonicity_find_value_of_a_l155_155505


namespace payment_per_mile_l155_155665

theorem payment_per_mile (miles_one_way : ℝ) (total_payment : ℝ) (total_miles_round_trip : ℝ) (payment_per_mile : ℝ) :
  miles_one_way = 400 → total_payment = 320 → total_miles_round_trip = miles_one_way * 2 →
  payment_per_mile = total_payment / total_miles_round_trip → payment_per_mile = 0.4 :=
by
  intros h_miles_one_way h_total_payment h_total_miles_round_trip h_payment_per_mile
  sorry

end payment_per_mile_l155_155665


namespace manager_salary_4200_l155_155720

theorem manager_salary_4200
    (avg_salary_employees : ℕ → ℕ → ℕ) 
    (total_salary_employees : ℕ → ℕ → ℕ)
    (new_avg_salary : ℕ → ℕ → ℕ)
    (total_salary_with_manager : ℕ → ℕ → ℕ) 
    (n_employees : ℕ)
    (employee_salary : ℕ) 
    (n_total : ℕ)
    (total_salary_before : ℕ)
    (avg_increase : ℕ)
    (new_employee_salary : ℕ) 
    (total_salary_after : ℕ) 
    (manager_salary : ℕ) :
    n_employees = 15 →
    employee_salary = 1800 →
    avg_increase = 150 →
    avg_salary_employees n_employees employee_salary = 1800 →
    total_salary_employees n_employees employee_salary = 27000 →
    new_avg_salary employee_salary avg_increase = 1950 →
    new_employee_salary = 1950 →
    total_salary_with_manager (n_employees + 1) new_employee_salary = 31200 →
    total_salary_before = 27000 →
    total_salary_after = 31200 →
    manager_salary = total_salary_after - total_salary_before →
    manager_salary = 4200 := 
by 
  intros 
  sorry

end manager_salary_4200_l155_155720


namespace average_age_of_club_l155_155103

theorem average_age_of_club (S_f S_m S_c : ℕ) (females males children : ℕ) (avg_females avg_males avg_children : ℕ) :
  females = 12 →
  males = 20 →
  children = 8 →
  avg_females = 28 →
  avg_males = 40 →
  avg_children = 10 →
  S_f = avg_females * females →
  S_m = avg_males * males →
  S_c = avg_children * children →
  (S_f + S_m + S_c) / (females + males + children) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end average_age_of_club_l155_155103


namespace washing_machines_removed_per_box_l155_155252

theorem washing_machines_removed_per_box 
  (crates : ℕ) (boxes_per_crate : ℕ) (washing_machines_per_box : ℕ) 
  (total_removed : ℕ) (total_crates : ℕ) (total_boxes_per_crate : ℕ) 
  (total_washing_machines_per_box : ℕ) 
  (h1 : crates = total_crates) (h2 : boxes_per_crate = total_boxes_per_crate) 
  (h3 : washing_machines_per_box = total_washing_machines_per_box) 
  (h4 : total_removed = 60) (h5 : total_crates = 10) 
  (h6 : total_boxes_per_crate = 6) 
  (h7 : total_washing_machines_per_box = 4):
  total_removed / (total_crates * total_boxes_per_crate) = 1 :=
by
  sorry

end washing_machines_removed_per_box_l155_155252


namespace simplify_expression_l155_155682

variables (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b)

theorem simplify_expression :
  (a^4 - a^2 * b^2) / (a - b)^2 / (a * (a + b) / b^2) * (b^2 / a) = b^4 / (a - b) :=
by
  sorry

end simplify_expression_l155_155682


namespace part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l155_155512

def A (x : ℝ) : Prop := x^2 - 4 * x - 5 ≥ 0
def B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

theorem part1_a_eq_neg1_inter (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by sorry

theorem part1_a_eq_neg1_union (a : ℝ) (h : a = -1) : 
  {x : ℝ | A x} ∪ {x : ℝ | B x a} = {x : ℝ | x ≤ 1 ∨ x ≥ 5} :=
by sorry

theorem part2_a_range (a : ℝ) : 
  ({x : ℝ | A x} ∩ {x : ℝ | B x a} = {x : ℝ | B x a}) → 
  a ∈ {a : ℝ | a > 2 ∨ a ≤ -3} :=
by sorry

end part1_a_eq_neg1_inter_part1_a_eq_neg1_union_part2_a_range_l155_155512


namespace no_six_digit_number_divisible_by_30_l155_155293

/-
  The claim is to show that forming a six-digit integer with the digits 2, 3, 3, 6, 0, and 5
  will result in no number that is divisible by 30.
-/

theorem no_six_digit_number_divisible_by_30 :
  ¬ ∃ n : ℕ, (n ∈ { n | nat.digits 10 n = [2, 3, 3, 6, 0, 5].perm }) ∧ (30 ∣ n) :=
begin
  sorry
end

end no_six_digit_number_divisible_by_30_l155_155293


namespace carl_garden_area_l155_155201

theorem carl_garden_area (total_posts : ℕ) (post_interval : ℕ) (x_posts_on_shorter : ℕ) (y_posts_on_longer : ℕ)
  (h1 : total_posts = 26)
  (h2 : post_interval = 5)
  (h3 : y_posts_on_longer = 2 * x_posts_on_shorter)
  (h4 : 2 * x_posts_on_shorter + 2 * y_posts_on_longer - 4 = total_posts) :
  (x_posts_on_shorter - 1) * post_interval * (y_posts_on_longer - 1) * post_interval = 900 := 
by
  sorry

end carl_garden_area_l155_155201


namespace triangle_area_l155_155165

theorem triangle_area (base height : ℝ) (h_base : base = 4.5) (h_height : height = 6) :
  (base * height) / 2 = 13.5 := 
by
  rw [h_base, h_height]
  norm_num

-- sorry
-- The later use of sorry statement is commented out because the proof itself has been provided in by block.

end triangle_area_l155_155165


namespace find_number_l155_155318

variable (x : ℕ)
variable (result : ℕ)

theorem find_number (h : x * 9999 = 4690640889) : x = 469131 :=
by
  sorry

end find_number_l155_155318


namespace sum_primes_between_1_and_20_l155_155984

/-- Define a helper predicate to check if a number is a prime number -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

/-- Define the set of prime numbers between 1 and 20 -/
def primes_between_1_and_20 : list ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

/-- Verify that each number in the defined set is prime -/
lemma all_primes_between_1_and_20 : ∀ n ∈ primes_between_1_and_20, is_prime n :=
by {
  intros n h,
  fin_cases h;
  unfold is_prime;
  { split, repeat { norm_num }, intros m hm1 hm2, interval_cases m; norm_num },
}

/-- Sum of primes between 1 and 20 is 77 -/
theorem sum_primes_between_1_and_20 : primes_between_1_and_20.sum = 77 :=
by {
  -- Calculate the sum manually or use computational method
  norm_num,
}

end sum_primes_between_1_and_20_l155_155984


namespace cost_of_notebooks_and_markers_l155_155570

theorem cost_of_notebooks_and_markers 
  (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.30) 
  (h2 : 5 * x + 3 * y = 11.65) : 
  2 * x + y = 4.35 :=
by
  sorry

end cost_of_notebooks_and_markers_l155_155570


namespace circles_externally_tangent_l155_155696

noncomputable def circle1_center : ℝ × ℝ := (-1, 1)
noncomputable def circle1_radius : ℝ := 2
noncomputable def circle2_center : ℝ × ℝ := (2, -3)
noncomputable def circle2_radius : ℝ := 3

noncomputable def distance_centers : ℝ :=
  Real.sqrt ((circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2)

theorem circles_externally_tangent :
  distance_centers = circle1_radius + circle2_radius :=
by
  -- The proof will show that the distance between the centers is equal to the sum of the radii, 
  -- indicating they are externally tangent.
  sorry

end circles_externally_tangent_l155_155696


namespace probability_xy_earlier_than_xm_l155_155715

variable {Ω : Type*} [Fintype Ω] [ProbabilitySpace Ω]

-- Define the possibilities of arrival for Xiao Jun, Xiao Yan and Xiao Ming
def arrival_order : Finset (Finset.univ : Finset (Fin 3)) := by
  exact {finset₁ | ∃ (x : Ω), true}

-- Define the event that Xiao Yan arrives earlier than Xiao Ming
def event_xy_earlier_than_xm (ω : Ω) : Prop := by
  sorry

-- Prove the probability of Xiao Yan arrives earlier than Xiao Ming is 1/2
theorem probability_xy_earlier_than_xm :
  Probability (λ ω, event_xy_earlier_than_xm ω) = 1/2 :=
by
  sorry

end probability_xy_earlier_than_xm_l155_155715


namespace hose_filling_time_l155_155189

theorem hose_filling_time :
  ∀ (P A B C : ℝ), 
  (P / 3 = A + B) →
  (P / 5 = A + C) →
  (P / 4 = B + C) →
  (P / (A + B + C) = 2.55) :=
by
  intros P A B C hAB hAC hBC
  sorry

end hose_filling_time_l155_155189


namespace distance_after_3_minutes_l155_155467

-- Conditions: speeds of the truck and car, and the time interval in hours
def v_truck : ℝ := 65 -- in km/h
def v_car : ℝ := 85 -- in km/h
def t : ℝ := 3 / 60 -- convert 3 minutes to hours

-- Statement to prove: The distance between the truck and the car after 3 minutes is 1 km
theorem distance_after_3_minutes : (v_car - v_truck) * t = 1 := 
by
  sorry

end distance_after_3_minutes_l155_155467


namespace leak_time_to_empty_l155_155172

def pump_rate : ℝ := 0.1 -- P = 0.1 tanks/hour
def effective_rate : ℝ := 0.05 -- P - L = 0.05 tanks/hour

theorem leak_time_to_empty (P L : ℝ) (hp : P = pump_rate) (he : P - L = effective_rate) :
  1 / L = 20 := by
  sorry

end leak_time_to_empty_l155_155172


namespace remainder_theorem_example_l155_155061

def polynomial (x : ℝ) : ℝ := x^15 + 3

theorem remainder_theorem_example :
  polynomial (-2) = -32765 :=
by
  -- Substitute x = -2 in the polynomial and show the remainder is -32765
  sorry

end remainder_theorem_example_l155_155061


namespace base10_to_base4_of_255_l155_155021

theorem base10_to_base4_of_255 :
  (255 : ℕ) = 3 * 4^3 + 3 * 4^2 + 3 * 4^1 + 3 * 4^0 :=
by
  sorry

end base10_to_base4_of_255_l155_155021


namespace square_side_length_theorem_l155_155314

-- Define the properties of the geometric configurations
def is_tangent_to_extension_segments (circle_radius : ℝ) (segment_length : ℝ) : Prop :=
  segment_length = circle_radius

def angle_between_tangents_from_point (angle : ℝ) : Prop :=
  angle = 60 

def square_side_length (side : ℝ) : Prop :=
  side = 4 * (Real.sqrt 2 - 1)

-- Main theorem
theorem square_side_length_theorem (circle_radius : ℝ) (segment_length : ℝ) (angle : ℝ) (side : ℝ)
  (h1 : is_tangent_to_extension_segments circle_radius segment_length)
  (h2 : angle_between_tangents_from_point angle) :
  square_side_length side :=
by
  sorry

end square_side_length_theorem_l155_155314


namespace possible_values_for_n_l155_155556

theorem possible_values_for_n (n : ℕ) (h1 : ∀ a b c : ℤ, (a = n-1) ∧ (b = n) ∧ (c = n+1) → 
    (∃ f g : ℤ, f = 2*a - b ∧ g = 2*b - a)) 
    (h2 : ∃ a b c : ℤ, (a = 0 ∨ b = 0 ∨ c = 0) ∧ (a + b + c = 0)) : 
    ∃ k : ℕ, n = 3^k := 
sorry

end possible_values_for_n_l155_155556


namespace addends_are_negative_l155_155296

theorem addends_are_negative (a b : ℤ) (h1 : a + b < a) (h2 : a + b < b) : a < 0 ∧ b < 0 := 
sorry

end addends_are_negative_l155_155296


namespace determine_b_l155_155101

theorem determine_b (a b c y1 y2 : ℝ) 
  (h1 : y1 = a * 2^2 + b * 2 + c)
  (h2 : y2 = a * (-2)^2 + b * (-2) + c)
  (h3 : y1 - y2 = -12) : 
  b = -3 := 
by
  sorry

end determine_b_l155_155101


namespace number_of_beavers_in_second_group_l155_155595

-- Define the number of beavers and the time for the first group
def numBeavers1 := 20
def time1 := 3

-- Define the time for the second group
def time2 := 5

-- Define the total work done (which is constant)
def work := numBeavers1 * time1

-- Define the number of beavers in the second group
def numBeavers2 := 12

-- Theorem stating the mathematical equivalence
theorem number_of_beavers_in_second_group : numBeavers2 * time2 = work :=
by
  -- remaining proof steps would go here
  sorry

end number_of_beavers_in_second_group_l155_155595


namespace basketball_team_selection_l155_155038

theorem basketball_team_selection :
  ∑ k in { k | k ≤ 5 }, (nat.choose 16 k) - 2 * (nat.choose 14 3) + (nat.choose 12 1) = 3652 := 
sorry

end basketball_team_selection_l155_155038


namespace price_reduction_for_1920_profit_maximum_profit_calculation_l155_155181

-- Definitions based on given conditions
def cost_price : ℝ := 12
def base_price : ℝ := 20
def base_quantity_sold : ℝ := 240
def increment_per_dollar : ℝ := 40

-- Profit function
def profit (x : ℝ) : ℝ := (base_price - cost_price - x) * (base_quantity_sold + increment_per_dollar * x)

-- Prove price reduction for $1920 profit per day
theorem price_reduction_for_1920_profit : ∃ x : ℝ, profit x = 1920 ∧ x = 8 := by
  sorry

-- Prove maximum profit calculation
theorem maximum_profit_calculation : ∃ x y : ℝ, x = 4 ∧ y = 2560 ∧ ∀ z, profit z ≤ y := by
  sorry

end price_reduction_for_1920_profit_maximum_profit_calculation_l155_155181


namespace ball_distribution_ways_l155_155240

theorem ball_distribution_ways :
  ∃ (ways : ℕ), ways = 10 ∧
    ∀ (balls boxes : ℕ), 
    balls = 6 ∧ boxes = 4 ∧ 
    (∀ (b : ℕ), b < boxes → b > 0) →
    ways = 10 :=
sorry

end ball_distribution_ways_l155_155240


namespace find_4_digit_number_l155_155492

theorem find_4_digit_number (a b c d : ℕ) 
(h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182)
(h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
(h6 : 0 ≤ c) (h7 : c ≤ 9) (h8 : 1 ≤ d) (h9 : d ≤ 9) : 
1000 * a + 100 * b + 10 * c + d = 1909 :=
sorry

end find_4_digit_number_l155_155492


namespace train_length_l155_155043

-- Definitions of the conditions as Lean terms/functions
def V (L : ℕ) := (L + 170) / 15
def U (L : ℕ) := (L + 250) / 20

-- The theorem to prove that the length of the train is 70 meters.
theorem train_length : ∃ L : ℕ, (V L = U L) → L = 70 := by
  sorry

end train_length_l155_155043


namespace square_difference_l155_155236

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x + 1) * (x - 1) = 9800 :=
by {
  sorry
}

end square_difference_l155_155236


namespace wise_men_guarantee_successful_task_l155_155835

theorem wise_men_guarantee_successful_task (h_sum : ∀ (S : Finset ℕ), S.card = 7 → (∑ x in S, x) = 100)
  (h_distinct : ∀ (S : Finset ℕ), S.card = 7 → S ≠ ∅ → (∀ x ∈ S, x ≠ 0))
  (a4 : ℕ) (h_a4 : a4 = 22) :
  ∃ S : Finset ℕ, S.card = 7 ∧ (∑ x in S, x = 100) ∧ (∃ l : list ℕ, l.sorted (≤) ∧ l = S.val) :=
by
  sorry

end wise_men_guarantee_successful_task_l155_155835


namespace maximum_marks_l155_155027

theorem maximum_marks (passing_percentage : ℝ) (score : ℝ) (shortfall : ℝ) (total_marks : ℝ) : 
  passing_percentage = 30 → 
  score = 212 → 
  shortfall = 16 → 
  total_marks = (score + shortfall) * 100 / passing_percentage → 
  total_marks = 760 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  assumption

end maximum_marks_l155_155027


namespace inverse_function_property_l155_155914

noncomputable def f (a x : ℝ) : ℝ := (x - a) * |x|

theorem inverse_function_property (a : ℝ) :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, f a (g x) = x) ↔ a = 0 :=
by sorry

end inverse_function_property_l155_155914


namespace half_abs_diff_squares_23_19_l155_155164

theorem half_abs_diff_squares_23_19 : 
  (| (23^2 - 19^2) |) / 2 = 84 := by
  sorry

end half_abs_diff_squares_23_19_l155_155164


namespace ceilings_left_to_paint_l155_155000

theorem ceilings_left_to_paint
    (floors : ℕ)
    (rooms_per_floor : ℕ)
    (ceilings_painted_this_week : ℕ)
    (hallways_per_floor : ℕ)
    (hallway_ceilings_per_hallway : ℕ)
    (ceilings_painted_ratio : ℚ)
    : floors = 4
    → rooms_per_floor = 7
    → ceilings_painted_this_week = 12
    → hallways_per_floor = 1
    → hallway_ceilings_per_hallway = 1
    → ceilings_painted_ratio = 1 / 4
    → (floors * rooms_per_floor + floors * hallways_per_floor * hallway_ceilings_per_hallway 
        - ceilings_painted_this_week 
        - (ceilings_painted_ratio * ceilings_painted_this_week + floors * hallway_ceilings_per_hallway) = 13) :=
by
  intros
  sorry

end ceilings_left_to_paint_l155_155000


namespace hyperbola_eccentricity_l155_155896

theorem hyperbola_eccentricity : 
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  e = Real.sqrt 5 / 2 := 
by
  let a := 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  sorry

end hyperbola_eccentricity_l155_155896


namespace count_polynomials_l155_155486

theorem count_polynomials (n : ℕ) (h_n : n = 30) :
  (∑ (a : ℕ) in finset.range 10, 
  ∑ (b : ℕ) in finset.range 10, 
  ∑ (c : ℕ) in finset.range 10, 
  ∑ (d : ℕ) in finset.range 10, if a - b + c - d = n then 1 else 0) = 5456 :=
by sorry

end count_polynomials_l155_155486


namespace sum_of_first_110_terms_l155_155699

theorem sum_of_first_110_terms
  (a d : ℝ)
  (h1 : (10 : ℝ) * (2 * a + (10 - 1) * d) / 2 = 100)
  (h2 : (100 : ℝ) * (2 * a + (100 - 1) * d) / 2 = 10) :
  (110 : ℝ) * (2 * a + (110 - 1) * d) / 2 = -110 :=
  sorry

end sum_of_first_110_terms_l155_155699


namespace sum_product_distinct_zero_l155_155947

open BigOperators

theorem sum_product_distinct_zero {n : ℕ} (h : n ≥ 3) (a : Fin n → ℝ) (ha : Function.Injective a) :
  (∑ i, (a i) * ∏ j in Finset.univ \ {i}, (1 / (a i - a j))) = 0 := 
by
  sorry

end sum_product_distinct_zero_l155_155947


namespace initial_oranges_l155_155708

theorem initial_oranges (O : ℕ) (h1 : (1 / 4 : ℚ) * (1 / 2 : ℚ) * O = 39) (h2 : (1 / 8 : ℚ) * (1 / 2 : ℚ) * O = 4 + 78 - (1 / 4 : ℚ) * (1 / 2 : ℚ) * O) :
  O = 96 :=
by
  sorry

end initial_oranges_l155_155708


namespace quadratic_eq_distinct_solutions_l155_155755

theorem quadratic_eq_distinct_solutions (b : ℤ) (k : ℤ) (h1 : 1 ≤ b ∧ b ≤ 100) :
  ∃ n : ℕ, n = 27 ∧ (x^2 + (2 * b + 3) * x + b^2 = 0 →
    12 * b + 9 = k^2 → 
    (∃ m n : ℤ, x = m ∧ x = n ∧ m ≠ n)) :=
sorry

end quadratic_eq_distinct_solutions_l155_155755


namespace distance_after_3_minutes_l155_155469

-- Define the speeds of the truck and the car in km/h.
def v_truck : ℝ := 65
def v_car : ℝ := 85

-- Define the time in hours.
def time_in_hours : ℝ := 3 / 60

-- Define the relative speed.
def v_relative : ℝ := v_car - v_truck

-- Define the expected distance between the truck and the car after 3 minutes.
def expected_distance : ℝ := 1

-- State the theorem: the distance between the truck and the car after 3 minutes is 1 km.
theorem distance_after_3_minutes : (v_relative * time_in_hours) = expected_distance := 
by {
  -- Here, we would provide the proof, but we are adding 'sorry' to skip the proof.
  sorry
}

end distance_after_3_minutes_l155_155469


namespace sufficient_but_not_necessary_l155_155263

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > |b|) → (a^3 > b^3) ∧ ¬((a^3 > b^3) → (a > |b|)) :=
by
  sorry

end sufficient_but_not_necessary_l155_155263


namespace john_weekly_calories_l155_155268

-- Define the calorie calculation for each meal type
def breakfast_calories : ℝ := 500
def morning_snack_calories : ℝ := 150
def lunch_calories : ℝ := breakfast_calories + 0.25 * breakfast_calories
def afternoon_snack_calories : ℝ := lunch_calories - 0.30 * lunch_calories
def dinner_calories : ℝ := 2 * lunch_calories

-- Total calories for Friday
def friday_calories : ℝ := breakfast_calories + morning_snack_calories + lunch_calories + afternoon_snack_calories + dinner_calories

-- Additional treats on Saturday and Sunday
def dessert_calories : ℝ := 350
def energy_drink_calories : ℝ := 220

-- Total calories for each day
def saturday_calories : ℝ := friday_calories + dessert_calories
def sunday_calories : ℝ := friday_calories + 2 * energy_drink_calories
def weekday_calories : ℝ := friday_calories

-- Proof statement
theorem john_weekly_calories : 
  friday_calories = 2962.5 ∧ 
  saturday_calories = 3312.5 ∧ 
  sunday_calories = 3402.5 ∧ 
  weekday_calories = 2962.5 :=
by 
  -- proof expressions would go here
  sorry

end john_weekly_calories_l155_155268


namespace red_to_green_ratio_l155_155251

theorem red_to_green_ratio (total_flowers green_flowers blue_percentage yellow_flowers : ℕ)
  (h1 : total_flowers = 96)
  (h2 : green_flowers = 9)
  (h3 : blue_percentage = 50)
  (h4 : yellow_flowers = 12) :
  let blue_flowers := (blue_percentage * total_flowers) / 100
  let red_flowers := total_flowers - (green_flowers + blue_flowers + yellow_flowers)
  (red_flowers : ℚ) / green_flowers = 3 := 
by
  sorry

end red_to_green_ratio_l155_155251


namespace sin_A_value_l155_155090

theorem sin_A_value
  (f : ℝ → ℝ)
  (cos_B : ℝ)
  (f_C_div_2 : ℝ)
  (C_acute : Prop) :
  (∀ x, f x = Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2) →
  cos_B = 1 / 3 →
  f (C / 2) = -1 / 4 →
  (0 < C ∧ C < Real.pi / 2) →
  Real.sin (Real.arcsin (Real.sqrt 3 / 2) + Real.arcsin (2 * Real.sqrt 2 / 3)) = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 :=
by
  intros
  sorry

end sin_A_value_l155_155090


namespace complex_fraction_sum_real_parts_l155_155641

theorem complex_fraction_sum_real_parts (a b : ℝ) (h : (⟨0, 1⟩ / ⟨1, 1⟩ : ℂ) = a + b * ⟨0, 1⟩) : a + b = 1 := by
  sorry

end complex_fraction_sum_real_parts_l155_155641


namespace even_function_on_neg_interval_l155_155011

theorem even_function_on_neg_interval
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_incr : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → f x₁ ≤ f x₂)
  (h_min : ∀ x : ℝ, 1 ≤ x → x ≤ 3 → 0 ≤ f x) :
  (∀ x : ℝ, -3 ≤ x → x ≤ -1 → 0 ≤ f x) ∧ (∀ x₁ x₂ : ℝ, -3 ≤ x₁ → x₁ < x₂ → x₂ ≤ -1 → f x₁ ≥ f x₂) :=
sorry

end even_function_on_neg_interval_l155_155011


namespace extreme_value_l155_155781

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)

theorem extreme_value (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, f x a = a - Real.log a - 1 ∧ (∀ y : ℝ, f y a ≤ f x a) :=
sorry

end extreme_value_l155_155781


namespace slips_with_3_l155_155825

variable (total_slips : ℕ) (expected_value : ℚ) (num_slips_with_3 : ℕ)

def num_slips_with_9 := total_slips - num_slips_with_3

def expected_value_calc (total_slips expected_value : ℚ) (num_slips_with_3 num_slips_with_9 : ℕ) : ℚ :=
  (num_slips_with_3 / total_slips) * 3 + (num_slips_with_9 / total_slips) * 9

theorem slips_with_3 (h1 : total_slips = 15) (h2 : expected_value = 5.4)
  (h3 : expected_value_calc total_slips expected_value num_slips_with_3 (num_slips_with_9 total_slips num_slips_with_3) = expected_value) :
  num_slips_with_3 = 9 :=
by
  rw [h1, h2] at h3
  sorry

end slips_with_3_l155_155825


namespace max_k_l155_155736

-- Define the conditions
def warehouse_weight : ℕ := 1500
def num_platforms : ℕ := 25
def platform_capacity : ℕ := 80

-- Define what we need to prove
theorem max_k (k : ℕ) : k ≤ 26 → 
  (∀ (containers : list ℕ), 
  (∀ c ∈ containers, 1 ≤ c ∧ c ≤ k) ∧ 
  containers.sum = warehouse_weight → 
  ∃ (platforms : list (list ℕ)),
  platforms.length = num_platforms ∧ 
  (∀ p ∈ platforms, p.sum ≤ platform_capacity) ∧ 
  list.join platforms = containers) :=
begin
  -- the proof would go here
  intros k hk containers hcontainers,
  sorry
end

end max_k_l155_155736


namespace total_weight_is_40_l155_155552

def marco_strawberries_weight : ℕ := 8
def dad_strawberries_weight : ℕ := 32
def total_strawberries_weight := marco_strawberries_weight + dad_strawberries_weight

theorem total_weight_is_40 : total_strawberries_weight = 40 := by
  sorry

end total_weight_is_40_l155_155552


namespace janet_total_pockets_l155_155937

theorem janet_total_pockets
  (total_dresses : ℕ)
  (dresses_with_pockets : ℕ)
  (dresses_with_2_pockets : ℕ)
  (dresses_with_3_pockets : ℕ)
  (pockets_from_2 : ℕ)
  (pockets_from_3 : ℕ)
  (total_pockets : ℕ)
  (h1 : total_dresses = 24)
  (h2 : dresses_with_pockets = total_dresses / 2)
  (h3 : dresses_with_2_pockets = dresses_with_pockets / 3)
  (h4 : dresses_with_3_pockets = dresses_with_pockets - dresses_with_2_pockets)
  (h5 : pockets_from_2 = 2 * dresses_with_2_pockets)
  (h6 : pockets_from_3 = 3 * dresses_with_3_pockets)
  (h7 : total_pockets = pockets_from_2 + pockets_from_3)
  : total_pockets = 32 := 
by
  sorry

end janet_total_pockets_l155_155937


namespace tangerines_left_l155_155210

def total_tangerines : ℕ := 27
def tangerines_eaten : ℕ := 18

theorem tangerines_left : total_tangerines - tangerines_eaten = 9 := by
  sorry

end tangerines_left_l155_155210


namespace minimum_value_is_138_l155_155942

-- Definition of problem conditions and question
def is_digit (n : ℕ) : Prop := n < 10
def digits (A : ℕ) : List ℕ := A.digits 10

def multiple_of_3_not_9 (A : ℕ) : Prop :=
  A % 3 = 0 ∧ A % 9 ≠ 0

def product_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· * ·) 1

def sum_of_digits (A : ℕ) : ℕ :=
  (digits A).foldl (· + ·) 0

def given_condition (A : ℕ) : Prop :=
  A % 9 = 0 → False ∧
  (A + product_of_digits A) % 9 = 0

-- Main goal: Prove that the minimum value A == 138 satisfies the given conditions
theorem minimum_value_is_138 : ∃ A, A = 138 ∧
  multiple_of_3_not_9 A ∧
  given_condition A :=
sorry

end minimum_value_is_138_l155_155942


namespace find_x_l155_155578

def seq : ℕ → ℕ
| 0 => 2
| 1 => 5
| 2 => 11
| 3 => 20
| 4 => 32
| 5 => 47
| (n+6) => seq (n+5) + 3 * (n + 1)

theorem find_x : seq 6 = 65 := by
  sorry

end find_x_l155_155578


namespace height_of_model_tower_l155_155280

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem height_of_model_tower (h_city : ℝ) (v_city : ℝ) (v_model : ℝ) (h_model : ℝ) :
  h_city = 80 → v_city = 200000 → v_model = 0.05 →
  let volume_ratio := v_city / v_model in
  let scale_factor := volume_ratio^(1/3) in
  h_model = h_city / scale_factor →
  h_model = 0.5 :=
by
  intros h_city_eq v_city_eq v_model_eq volume_ratio scale_factor h_model_eq
  sorry 

end height_of_model_tower_l155_155280


namespace range_of_a_l155_155269

noncomputable def line_eq (a : ℝ) (x y : ℝ) : ℝ := 3 * x - 2 * y + a 

def pointA : ℝ × ℝ := (3, 1)
def pointB : ℝ × ℝ := (-4, 6)

theorem range_of_a :
  (line_eq a pointA.1 pointA.2) * (line_eq a pointB.1 pointB.2) < 0 ↔ -7 < a ∧ a < 24 := sorry

end range_of_a_l155_155269


namespace sum_of_primes_between_1_and_20_l155_155994

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l155_155994


namespace find_a_l155_155235

def A : Set ℝ := {2, 3}
def B (a : ℝ) : Set ℝ := {1, 2, a}

theorem find_a (a : ℝ) : A ⊆ B a → a = 3 :=
by
  intro h
  sorry

end find_a_l155_155235


namespace PS_length_correct_l155_155152

variable {Triangle : Type}

noncomputable def PR := 15

noncomputable def PS_length (PS SR : ℝ) (PR : ℝ) : Prop :=
  PS + SR = PR ∧ (PS / SR) = (3 / 4)

theorem PS_length_correct : 
  ∃ PS SR : ℝ, PS_length PS SR PR ∧ PS = (45 / 7) :=
sorry

end PS_length_correct_l155_155152


namespace andy_last_problem_l155_155875

theorem andy_last_problem (start_num : ℕ) (num_solved : ℕ) (result : ℕ) : 
  start_num = 78 → 
  num_solved = 48 → 
  result = start_num + num_solved - 1 → 
  result = 125 :=
by
  sorry

end andy_last_problem_l155_155875


namespace adjusted_retail_price_l155_155465

variable {a : ℝ} {m n : ℝ}

theorem adjusted_retail_price (h : 0 ≤ m ∧ 0 ≤ n) : (a * (1 + m / 100) * (n / 100)) = a * (1 + m / 100) * (n / 100) :=
by
  sorry

end adjusted_retail_price_l155_155465


namespace midline_equation_l155_155249

theorem midline_equation (a b : ℝ) (K1 K2 : ℝ)
  (h1 : K1^2 = (a^2) / 4 + b^2)
  (h2 : K2^2 = a^2 + (b^2) / 4) :
  16 * K2^2 - 4 * K1^2 = 15 * a^2 :=
by
  sorry

end midline_equation_l155_155249


namespace find_g_two_fifths_l155_155694

noncomputable def g : ℝ → ℝ :=
sorry -- The function g(x) is not explicitly defined.

theorem find_g_two_fifths :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g x = 0 → g 0 = 0) ∧
  (∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 5) = g x / 3)
  → g (2 / 5) = 1 / 3 :=
sorry

end find_g_two_fifths_l155_155694


namespace probability_three_digit_divisible_by_5_with_ones_digit_9_l155_155701

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ones_digit (n : ℕ) : ℕ := n % 10

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem probability_three_digit_divisible_by_5_with_ones_digit_9 : 
  ∀ (M : ℕ), is_three_digit M → ones_digit M = 9 → ¬ is_divisible_by_5 M := by
  intros M h1 h2
  sorry

end probability_three_digit_divisible_by_5_with_ones_digit_9_l155_155701


namespace perimeter_of_park_l155_155732

def length := 300
def breadth := 200

theorem perimeter_of_park : 2 * (length + breadth) = 1000 := by
  sorry

end perimeter_of_park_l155_155732


namespace sum_of_lengths_of_edges_geometric_progression_l155_155967

theorem sum_of_lengths_of_edges_geometric_progression :
  ∃ (a r : ℝ), (a / r) * a * (a * r) = 8 ∧ 2 * (a / r * a + a * a * r + a * r * a / r) = 32 ∧ 
  4 * ((a / r) + a + (a * r)) = 32 :=
by
  sorry

end sum_of_lengths_of_edges_geometric_progression_l155_155967


namespace num_solutions_eq_3_l155_155674

theorem num_solutions_eq_3 : 
  ∃ (x1 x2 x3 : ℝ), (∀ x : ℝ, 2^x - 2 * (⌊x⌋:ℝ) - 1 = 0 → x = x1 ∨ x = x2 ∨ x = x3) 
  ∧ ¬ ∃ x4, (2^x4 - 2 * (⌊x4⌋:ℝ) - 1 = 0 ∧ x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3) :=
sorry

end num_solutions_eq_3_l155_155674


namespace problem_B_height_l155_155345

noncomputable def point_B_height (cos : ℝ → ℝ) : ℝ :=
  let θ := 30 * (Real.pi / 180)
  let cos30 := cos θ
  let original_vertical_height := 1 / 2
  let additional_height := cos30 * (1 / 2)
  original_vertical_height + additional_height

theorem problem_B_height : 
  point_B_height Real.cos = (2 + Real.sqrt 3) / 4 := 
by 
  sorry

end problem_B_height_l155_155345


namespace f_x1_plus_f_x2_always_greater_than_zero_l155_155508

theorem f_x1_plus_f_x2_always_greater_than_zero
  {f : ℝ → ℝ}
  (h1 : ∀ x, f (-x) = -f (x + 2))
  (h2 : ∀ x > 1, ∀ y > 1, x < y → f y < f x)
  (h3 : ∃ x₁ x₂ : ℝ, 1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) :
  ∀ x₁ x₂ : ℝ, (1 + x₁ * x₂ < x₁ + x₂ ∧ x₁ + x₂ < 2) → f x₁ + f x₂ > 0 := by
  sorry

end f_x1_plus_f_x2_always_greater_than_zero_l155_155508


namespace smallest_even_number_of_sum_1194_l155_155928

-- Defining the given condition
def sum_of_three_consecutive_even_numbers (x : ℕ) : Prop :=
  x + (x + 2) + (x + 4) = 1194

-- Stating the theorem to prove the smallest even number
theorem smallest_even_number_of_sum_1194 :
  ∃ x : ℕ, sum_of_three_consecutive_even_numbers x ∧ x = 396 :=
by
  sorry

end smallest_even_number_of_sum_1194_l155_155928


namespace g_ln_1_over_2017_l155_155225

theorem g_ln_1_over_2017 (a : ℝ) (h_a_pos : 0 < a) (h_a_neq_1 : a ≠ 1) (f g : ℝ → ℝ)
  (h_f_add : ∀ m n : ℝ, f (m + n) = f m + f n - 1)
  (h_g : ∀ x : ℝ, g x = f x + a^x / (a^x + 1))
  (h_g_ln_2017 : g (Real.log 2017) = 2018) :
  g (Real.log (1 / 2017)) = -2015 :=
sorry

end g_ln_1_over_2017_l155_155225


namespace meaning_of_negative_angle_l155_155753

-- Condition: a counterclockwise rotation of 30 degrees is denoted as +30 degrees.
-- Here, we set up two simple functions to represent the meaning of positive and negative angles.

def counterclockwise (angle : ℝ) : Prop :=
  angle > 0

def clockwise (angle : ℝ) : Prop :=
  angle < 0

-- Question: What is the meaning of -45 degrees?
theorem meaning_of_negative_angle : clockwise 45 :=
by
  -- we know from the problem that a positive angle (like 30 degrees) indicates counterclockwise rotation,
  -- therefore a negative angle (like -45 degrees), by definition, implies clockwise rotation.
  sorry

end meaning_of_negative_angle_l155_155753


namespace find_halfway_between_l155_155406

def halfway_between (a b : ℚ) : ℚ := (a + b) / 2

theorem find_halfway_between :
  halfway_between (1/8 : ℚ) (1/3 : ℚ) = 11/48 :=
by
  -- declare needed intermediate calculations (common denominators, etc.)
  sorry

end find_halfway_between_l155_155406


namespace points_on_line_with_slope_l155_155842

theorem points_on_line_with_slope :
  ∃ a b : ℝ, 
  (a - 3) ≠ 0 ∧ (b - 5) ≠ 0 ∧
  (7 - 5) / (a - 3) = 4 ∧ (b - 5) / (-1 - 3) = 4 ∧
  a = 7 / 2 ∧ b = -11 := 
by
  existsi 7 / 2
  existsi -11
  repeat {split}
  all_goals { sorry }

end points_on_line_with_slope_l155_155842


namespace prop_B_contrapositive_correct_l155_155024

/-
Proposition B: The contrapositive of the proposition 
"If x^2 < 1, then -1 < x < 1" is 
"If x ≥ 1 or x ≤ -1, then x^2 ≥ 1".
-/
theorem prop_B_contrapositive_correct :
  (∀ (x : ℝ), x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ (x : ℝ), (x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
sorry

end prop_B_contrapositive_correct_l155_155024


namespace exists_natural_multiple_of_2015_with_digit_sum_2015_l155_155207

-- Definition of sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Proposition that we need to prove
theorem exists_natural_multiple_of_2015_with_digit_sum_2015 :
  ∃ n : ℕ, (2015 ∣ n) ∧ sum_of_digits n = 2015 :=
sorry

end exists_natural_multiple_of_2015_with_digit_sum_2015_l155_155207


namespace measles_cases_in_1990_l155_155650

noncomputable def measles_cases_1970 := 480000
noncomputable def measles_cases_2000 := 600
noncomputable def years_between := 2000 - 1970
noncomputable def total_decrease := measles_cases_1970 - measles_cases_2000
noncomputable def decrease_per_year := total_decrease / years_between
noncomputable def years_from_1970_to_1990 := 1990 - 1970
noncomputable def decrease_to_1990 := years_from_1970_to_1990 * decrease_per_year
noncomputable def measles_cases_1990 := measles_cases_1970 - decrease_to_1990

theorem measles_cases_in_1990 : measles_cases_1990 = 160400 := by
  sorry

end measles_cases_in_1990_l155_155650


namespace checkerboards_that_cannot_be_covered_l155_155585

-- Define the dimensions of the checkerboards
def checkerboard_4x6 := (4, 6)
def checkerboard_3x7 := (3, 7)
def checkerboard_5x5 := (5, 5)
def checkerboard_7x4 := (7, 4)
def checkerboard_5x6 := (5, 6)

-- Define a function to calculate the number of squares
def num_squares (dims : Nat × Nat) : Nat := dims.1 * dims.2

-- Define a function to check if a board can be exactly covered by dominoes
def can_be_covered_by_dominoes (dims : Nat × Nat) : Bool := (num_squares dims) % 2 == 0

-- Statement to be proven
theorem checkerboards_that_cannot_be_covered :
  ¬ can_be_covered_by_dominoes checkerboard_3x7 ∧ ¬ can_be_covered_by_dominoes checkerboard_5x5 :=
by
  sorry

end checkerboards_that_cannot_be_covered_l155_155585


namespace complex_roots_equilateral_l155_155548

noncomputable def omega : ℂ := -1/2 + Complex.I * Real.sqrt 3 / 2

theorem complex_roots_equilateral (z1 z2 p q : ℂ) (h₁ : z2 = omega * z1) (h₂ : -p = (1 + omega) * z1) (h₃ : q = omega * z1 ^ 2) :
  p^2 / q = 1 + Complex.I * Real.sqrt 3 :=
by sorry

end complex_roots_equilateral_l155_155548


namespace bicycle_count_l155_155018

theorem bicycle_count (T : ℕ) (B : ℕ) (h1 : T = 14) (h2 : 2 * B + 3 * T = 90) : B = 24 :=
by {
  sorry
}

end bicycle_count_l155_155018


namespace remainder_product_mod_5_l155_155286

theorem remainder_product_mod_5 
  (a b c : ℕ) 
  (ha : a % 5 = 1) 
  (hb : b % 5 = 2) 
  (hc : c % 5 = 3) : 
  (a * b * c) % 5 = 1 :=
by
  sorry

end remainder_product_mod_5_l155_155286


namespace set_union_is_all_real_l155_155089

-- Define the universal set U as the real numbers
def U := ℝ

-- Define the set M as {x | x > 0}
def M : Set ℝ := {x | x > 0}

-- Define the set N as {x | x^2 ≥ x}
def N : Set ℝ := {x | x^2 ≥ x}

-- Prove the relationship M ∪ N = ℝ
theorem set_union_is_all_real : M ∪ N = U := by
  sorry

end set_union_is_all_real_l155_155089


namespace least_distinct_values_l155_155861

variable (L : List Nat) (h_len : L.length = 2023) (mode : Nat) 
variable (h_mode_unique : ∀ x ∈ L, L.count x ≤ 15 → x = mode)
variable (h_mode_count : L.count mode = 15)

theorem least_distinct_values : ∃ k, k = 145 ∧ (∀ d ∈ L, List.count d L ≤ 15) :=
by
  sorry

end least_distinct_values_l155_155861


namespace regular_polygon_sides_l155_155792

-- Define the measure of each exterior angle
def exterior_angle (n : ℕ) (angle : ℝ) : Prop :=
  angle = 40.0

-- Define the sum of exterior angles of any polygon
def sum_exterior_angles (n : ℕ) (total_angle : ℝ) : Prop :=
  total_angle = 360.0

-- Theorem to prove
theorem regular_polygon_sides (n : ℕ) :
  (exterior_angle n 40.0) ∧ (sum_exterior_angles n 360.0) → n = 9 :=
by
  sorry

end regular_polygon_sides_l155_155792


namespace extreme_value_of_f_range_of_values_for_a_l155_155091

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem extreme_value_of_f :
  ∃ x_min : ℝ, f x_min = 1 :=
sorry

theorem range_of_values_for_a :
  ∀ a : ℝ, (∀ x : ℝ, f x ≥ (x^3) / 6 + a) → a ≤ 1 :=
sorry

end extreme_value_of_f_range_of_values_for_a_l155_155091


namespace cube_side_length_l155_155605

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = 1 / 4 * 6 * n^3) : n = 4 := 
by 
  sorry

end cube_side_length_l155_155605


namespace number_of_pencils_bought_l155_155365

-- Define the conditions
def cost_of_glue : ℕ := 270
def cost_per_pencil : ℕ := 210
def amount_paid : ℕ := 1000
def change_received : ℕ := 100

-- Define the statement to prove
theorem number_of_pencils_bought : 
  ∃ (n : ℕ), cost_of_glue + (cost_per_pencil * n) = amount_paid - change_received :=
by {
  sorry 
}

end number_of_pencils_bought_l155_155365


namespace literature_more_than_science_science_less_than_literature_percent_l155_155964

theorem literature_more_than_science (l s : ℕ) (h : 8 * s = 5 * l) : (l - s) / s = 3 / 5 :=
by {
  -- definition and given condition will be provided
  sorry
}

theorem science_less_than_literature_percent (l s : ℕ) (h : 8 * s = 5 * l) : ((l - s : ℚ) / l) * 100 = 37.5 :=
by {
  -- definition and given condition will be provided
  sorry
}

end literature_more_than_science_science_less_than_literature_percent_l155_155964


namespace avg_bc_eq_28_l155_155012

variable (A B C : ℝ)

-- Conditions
def avg_abc_eq_30 : Prop := (A + B + C) / 3 = 30
def avg_ab_eq_25 : Prop := (A + B) / 2 = 25
def b_eq_16 : Prop := B = 16

-- The Proved Statement
theorem avg_bc_eq_28 (h1 : avg_abc_eq_30 A B C) (h2 : avg_ab_eq_25 A B) (h3 : b_eq_16 B) : (B + C) / 2 = 28 := 
by
  sorry

end avg_bc_eq_28_l155_155012


namespace pow_mod_sub_l155_155749

theorem pow_mod_sub (a b : ℕ) (n : ℕ) (h1 : a ≡ 5 [MOD 6]) (h2 : b ≡ 4 [MOD 6]) : (a^n - b^n) % 6 = 1 :=
by
  let a := 47
  let b := 22
  let n := 1987
  sorry

end pow_mod_sub_l155_155749


namespace sam_total_coins_l155_155271

theorem sam_total_coins (nickel_count : ℕ) (dime_count : ℕ) (total_value_cents : ℤ) (nickel_value : ℤ) (dime_value : ℤ)
  (h₁ : nickel_count = 12)
  (h₂ : total_value_cents = 240)
  (h₃ : nickel_value = 5)
  (h₄ : dime_value = 10)
  (h₅ : nickel_count * nickel_value + dime_count * dime_value = total_value_cents) :
  nickel_count + dime_count = 30 := 
  sorry

end sam_total_coins_l155_155271


namespace unique_two_digit_number_l155_155920

-- Definition of the problem in Lean
def is_valid_number (n : ℕ) : Prop :=
  n % 4 = 1 ∧ n % 17 = 1 ∧ 10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 69 :=
by
  sorry

end unique_two_digit_number_l155_155920


namespace problem_statement_l155_155671

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem problem_statement : f (f (f (f (f (f 2))))) = 4 :=
by
  sorry

end problem_statement_l155_155671


namespace circle_land_represents_30105_l155_155308

-- Definitions based on the problem's conditions
def circleLandNumber (digits : List (ℕ × ℕ)) : ℕ :=
  digits.foldl (λ acc (d_circle : ℕ × ℕ) => acc + d_circle.fst * 10^d_circle.snd) 0

-- Example 207
def number_207 : List (ℕ × ℕ) := [(2, 2), (0, 0), (7, 0)]

-- Example 4520
def number_4520 : List (ℕ × ℕ) := [(4, 3), (5, 1), (2, 0), (0, 0)]

-- The diagram to analyze
def given_diagram : List (ℕ × ℕ) := [(3, 4), (1, 2), (5, 0)]

-- The statement proving the given diagram represents 30105 in Circle Land
theorem circle_land_represents_30105 : circleLandNumber given_diagram = 30105 :=
  sorry

end circle_land_represents_30105_l155_155308


namespace max_value_of_f_l155_155130

noncomputable def f (x : ℝ) : ℝ := (log x) / (x ^ 2)

-- Define the domain
def domain := Set.Ioi 0

-- Theorem statement
theorem max_value_of_f :
  ∃ x ∈ domain, (∀ y ∈ domain, f y ≤ f x) ∧ x = Real.sqrt Real.exp 1 ∧ f x = 1 / (2 * Real.exp 1) :=
by
  sorry

end max_value_of_f_l155_155130


namespace polynomial_modulo_problem_l155_155809

open scoped Nat BigOperators

theorem polynomial_modulo_problem :
  let n := 2016
  let k := 2015
  let total_monomials := (4031.choose 2016)
  let N := 3^total_monomials / 3^(2^n)
  let v3 (n : Nat) : Nat := 
    if n = 0 then 0 else
      v3 (n / 3) + 1
  v3(N) % 2011 = 188 :=
by
  sorry

end polynomial_modulo_problem_l155_155809


namespace circles_condition_l155_155020

noncomputable def circles_intersect_at (p1 p2 : ℝ × ℝ) (m c : ℝ) : Prop :=
  p1 = (1, 3) ∧ p2 = (m, 1) ∧ (∃ (x y : ℝ), (x - y + c / 2 = 0) ∧ 
    (p1.1 - x)^2 + (p1.2 - y)^2 = (p2.1 - x)^2 + (p2.2 - y)^2)

theorem circles_condition (m c : ℝ) (h : circles_intersect_at (1, 3) (m, 1) m c) : m + c = 3 :=
sorry

end circles_condition_l155_155020


namespace partition_two_houses_l155_155654

-- Declare types and assumptions
variable {V : Type*} [Fintype V]

-- Assume a graph G representing the enemy relationships
variable (G : SimpleGraph V)

-- Each member has at most 3 enemies (degree constraint)
variable (h_max_enemies : ∀ v : V, G.degree v ≤ 3)

-- Statement of the proof problem
theorem partition_two_houses (G : SimpleGraph V) (h : ∀ v : V, G.degree v ≤ 3) : 
  ∃ (A B : Finset V), A ∩ B = ∅ ∧ A ∪ B = Finset.univ ∧ 
  (∀ v ∈ A, (Finset.filter (λ w, G.adj v w) A).card ≤ 1) ∧ 
  (∀ v ∈ B, (Finset.filter (λ w, G.adj v w) B).card ≤ 1) :=
by
  sorry

end partition_two_houses_l155_155654


namespace sum_of_roots_eq_2_l155_155752

open Polynomial

def cubic_poly : Polynomial ℝ :=
  Polynomial.C (5 : ℝ) * X^3 + Polynomial.C (-10 : ℝ) * X^2 + Polynomial.C (1 : ℝ) * X - Polynomial.C (24 : ℝ)

theorem sum_of_roots_eq_2 : (root_sum cubic_poly) = 2 := 
sorry

end sum_of_roots_eq_2_l155_155752


namespace relationship_of_AT_l155_155610

def S : ℝ := 300
def PC : ℝ := S + 500
def total_cost : ℝ := 2200

theorem relationship_of_AT (AT : ℝ) 
  (h1: S + PC + AT = total_cost) : 
  AT = S + PC - 400 :=
by
  sorry

end relationship_of_AT_l155_155610


namespace starting_number_l155_155880

theorem starting_number (x : ℝ) (h : (x + 26) / 2 = 19) : x = 12 :=
by
  sorry

end starting_number_l155_155880


namespace determine_selling_price_for_daily_profit_determine_max_profit_and_selling_price_l155_155722

-- Cost price per souvenir
def cost_price : ℕ := 40

-- Minimum selling price
def min_selling_price : ℕ := 44

-- Maximum selling price
def max_selling_price : ℕ := 60

-- Units sold if selling price is min_selling_price
def units_sold_at_min_price : ℕ := 300

-- Units sold decreases by 10 for every 1 yuan increase in selling price
def decrease_in_units (increase : ℕ) : ℕ := 10 * increase

-- Daily profit for a given increase in selling price
def daily_profit (increase : ℕ) : ℕ := (increase + min_selling_price - cost_price) * (units_sold_at_min_price - decrease_in_units increase)

-- Maximum profit calculation
def maximizing_daily_profit (increase : ℕ) : ℕ := (increase + min_selling_price - cost_price) * (units_sold_at_min_price - decrease_in_units increase) 

-- Statement for Problem Part 1
theorem determine_selling_price_for_daily_profit : ∃ P, P = 52 ∧ daily_profit (P - min_selling_price) = 2640 := 
sorry

-- Statement for Problem Part 2
theorem determine_max_profit_and_selling_price : ∃ P, P = 57 ∧ maximizing_daily_profit (P - min_selling_price) = 2890 := 
sorry

end determine_selling_price_for_daily_profit_determine_max_profit_and_selling_price_l155_155722


namespace exponentiation_comparison_l155_155299

theorem exponentiation_comparison :
  1.7 ^ 0.3 > 0.9 ^ 0.3 :=
by sorry

end exponentiation_comparison_l155_155299


namespace total_growing_space_l155_155741

noncomputable def garden_area : ℕ :=
  let area_3x3 := 3 * 3
  let total_area_3x3 := 2 * area_3x3
  let area_4x3 := 4 * 3
  let total_area_4x3 := 2 * area_4x3
  total_area_3x3 + total_area_4x3

theorem total_growing_space : garden_area = 42 :=
by
  sorry

end total_growing_space_l155_155741


namespace bug_paths_from_A_to_B_l155_155452

-- Define the positions A and B and intermediate red and blue points in the lattice
inductive Position
| A
| B
| red1
| red2
| blue1
| blue2

open Position

-- Define the possible directed paths in the lattice
def paths : List (Position × Position) :=
[(A, red1), (A, red2), 
 (red1, blue1), (red1, blue2), 
 (red2, blue1), (red2, blue2), 
 (blue1, B), (blue1, B), (blue1, B), 
 (blue2, B), (blue2, B), (blue2, B)]

-- Define a function that calculates the number of unique paths from A to B without repeating any path
def count_paths : ℕ := sorry

-- The mathematical problem statement
theorem bug_paths_from_A_to_B : count_paths = 24 := sorry

end bug_paths_from_A_to_B_l155_155452


namespace profit_percentage_is_twenty_percent_l155_155025

def selling_price : ℕ := 900
def profit : ℕ := 150
def cost_price : ℕ := selling_price - profit
def profit_percentage : ℕ := (profit * 100) / cost_price

theorem profit_percentage_is_twenty_percent : profit_percentage = 20 := by
  sorry

end profit_percentage_is_twenty_percent_l155_155025


namespace jogger_distance_ahead_l155_155860

def speed_jogger_kmph : ℕ := 9
def speed_train_kmph : ℕ := 45
def length_train_m : ℕ := 120
def time_to_pass_jogger_s : ℕ := 36

theorem jogger_distance_ahead :
  let relative_speed_mps := (speed_train_kmph - speed_jogger_kmph) * 1000 / 3600
  let distance_covered_m := relative_speed_mps * time_to_pass_jogger_s
  let jogger_distance_ahead : ℕ := distance_covered_m - length_train_m
  jogger_distance_ahead = 240 :=
by
  sorry

end jogger_distance_ahead_l155_155860


namespace a_term_b_value_c_value_d_value_l155_155097

theorem a_term (a x : ℝ) (h1 : a * (x + 1) = x^3 + 3 * x^2 + 3 * x + 1) : a = x^2 + 2 * x + 1 :=
sorry

theorem b_value (a x b : ℝ) (h1 : a - 1 = 0) (h2 : x = 0 ∨ x = b) : b = -2 :=
sorry

theorem c_value (p c b : ℝ) (h1 : p * c^4 = 32) (h2 : p * c = b^2) (h3 : 0 < c) : c = 2 :=
sorry

theorem d_value (A B d : ℝ) (P : ℝ → ℝ) (c : ℝ) (h1 : P (A * B) = P A + P B) (h2 : P A = 1) (h3 : P B = c) (h4 : A = 10^ P A) (h5 : B = 10^ P B) (h6 : d = A * B) : d = 1000 :=
sorry

end a_term_b_value_c_value_d_value_l155_155097


namespace hammers_in_comparison_group_l155_155094

theorem hammers_in_comparison_group (H W x : ℝ) (h1 : 2 * H + 2 * W = 1 / 3 * (x * H + 5 * W)) (h2 : W = 2 * H) :
  x = 8 :=
sorry

end hammers_in_comparison_group_l155_155094


namespace minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l155_155538

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions based on the problem statements
axiom a1_neg : a 1 < 0
axiom S2015_neg : S 2015 < 0
axiom S2016_pos : S 2016 > 0

-- Defining n value where S_n reaches its minimum
def n_min := 1008

theorem minimum_S_n_at_1008 : S n_min = S 1008 := sorry

-- Additional theorems to satisfy the provided conditions
theorem a1008_neg : a 1008 < 0 := sorry
theorem a1009_pos : a 1009 > 0 := sorry
theorem common_difference_pos : ∀ n : ℕ, a (n + 1) - a n > 0 := sorry

end minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l155_155538


namespace equilateral_triangle_grid_l155_155626

noncomputable def number_of_triangles (n : ℕ) : ℕ :=
1 + 3 + 5 + 7 + 9 + 1 + 2 + 3 + 4 + 3 + 1 + 2 + 3 + 1 + 2 + 1

theorem equilateral_triangle_grid (n : ℕ) (h : n = 5) : number_of_triangles n = 48 := by
  sorry

end equilateral_triangle_grid_l155_155626


namespace calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l155_155014

noncomputable def volume_of_parallelepiped (R : ℝ) : ℝ := R^3 * Real.sqrt 6

noncomputable def diagonal_A_C_prime (R: ℝ) : ℝ := R * Real.sqrt 6

noncomputable def volume_of_rotation (R: ℝ) : ℝ := R^3 * Real.sqrt 12

theorem calculate_volume_and_diagonal (R : ℝ) : 
  volume_of_parallelepiped R = R^3 * Real.sqrt 6 ∧ 
  diagonal_A_C_prime R = R * Real.sqrt 6 :=
by sorry

theorem calculate_volume_and_surface_rotation (R : ℝ) :
  volume_of_rotation R = R^3 * Real.sqrt 12 :=
by sorry

theorem calculate_radius_given_volume (V : ℝ) (h : V = 0.034786) : 
  ∃ R : ℝ, V = volume_of_parallelepiped R :=
by sorry

end calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l155_155014


namespace tom_has_65_fruits_left_l155_155418

def initial_fruits : ℕ := 40 + 70 + 30 + 15

def sold_oranges : ℕ := (1 / 4) * 40
def sold_apples : ℕ := (2 / 3) * 70
def sold_bananas : ℕ := (5 / 6) * 30
def sold_kiwis : ℕ := (60 / 100) * 15

def fruits_remaining : ℕ :=
  40 - sold_oranges +
  70 - sold_apples +
  30 - sold_bananas +
  15 - sold_kiwis

theorem tom_has_65_fruits_left :
  fruits_remaining = 65 := by
  sorry

end tom_has_65_fruits_left_l155_155418


namespace basis_vetors_correct_options_l155_155300

def is_basis (e1 e2 : ℝ × ℝ) : Prop :=
  e1 ≠ (0, 0) ∧ e2 ≠ (0, 0) ∧ e1.1 * e2.2 - e1.2 * e2.1 ≠ 0

def option_A : ℝ × ℝ := (0, 0)
def option_A' : ℝ × ℝ := (1, 2)

def option_B : ℝ × ℝ := (2, -1)
def option_B' : ℝ × ℝ := (1, 2)

def option_C : ℝ × ℝ := (-1, -2)
def option_C' : ℝ × ℝ := (1, 2)

def option_D : ℝ × ℝ := (1, 1)
def option_D' : ℝ × ℝ := (1, 2)

theorem basis_vetors_correct_options:
  ¬ is_basis option_A option_A' ∧ ¬ is_basis option_C option_C' ∧ 
  is_basis option_B option_B' ∧ is_basis option_D option_D' := 
by
  sorry

end basis_vetors_correct_options_l155_155300


namespace Faye_created_rows_l155_155339

theorem Faye_created_rows (total_crayons : ℕ) (crayons_per_row : ℕ) (rows : ℕ) 
  (h1 : total_crayons = 210) (h2 : crayons_per_row = 30) : rows = 7 :=
by
  sorry

end Faye_created_rows_l155_155339


namespace sum_of_products_l155_155030

def is_positive (x : ℝ) := 0 < x

theorem sum_of_products 
  (x y z : ℝ) 
  (hx : is_positive x)
  (hy : is_positive y)
  (hz : is_positive z)
  (h1 : x^2 + x * y + y^2 = 27)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 52) :
  x * y + y * z + z * x = 30 :=
  sorry

end sum_of_products_l155_155030


namespace trajectory_midpoint_Q_line_l_l155_155507

noncomputable def circle := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 3)^2 = 9}
def point_p := (5, -1 : ℝ)

theorem trajectory_midpoint_Q :
  ∀ Q : ℝ × ℝ,
  (∃ A ∈ circle, Q = ((A.1 + 5) / 2, (A.2 - 1) / 2)) →
  (2 * Q.1 - 8)^2 + (2 * Q.2 - 2)^2 = 9 :=
by
  sorry

theorem line_l :
  ∀ (line_eq : String),
  ∃ l : ℝ × ℝ → ℝ,
  (∀ (x y : ℝ), line_eq = "3x + 4y - 11 = 0" → l (x, y) = 3 * x + 4 * y - 11) ∨
  (∀ (x : ℝ), line_eq = "x = 5" → l (x, 0).1 = 5) :=
by
  sorry

end trajectory_midpoint_Q_line_l_l155_155507


namespace parabola_line_intersect_at_one_point_l155_155887

theorem parabola_line_intersect_at_one_point :
  ∃ a : ℝ, (∀ x : ℝ, (ax^2 + 5 * x + 2 = -2 * x + 1)) ↔ a = 49 / 4 :=
by sorry

end parabola_line_intersect_at_one_point_l155_155887


namespace trapezoid_perimeter_is_correct_l155_155934

noncomputable def trapezoid_perimeter_proof : ℝ :=
  let EF := 60
  let θ := Real.pi / 4 -- 45 degrees in radians
  let h := 30 * Real.sqrt 2
  let GH := EF + 2 * h / Real.tan θ
  let EG := h / Real.tan θ
  EF + GH + 2 * EG -- Perimeter calculation

theorem trapezoid_perimeter_is_correct :
  trapezoid_perimeter_proof = 180 + 60 * Real.sqrt 2 := 
by
  sorry

end trapezoid_perimeter_is_correct_l155_155934


namespace exists_same_color_points_distance_one_l155_155867

theorem exists_same_color_points_distance_one
    (color : ℝ × ℝ → Fin 3)
    (h : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_distance_one_l155_155867


namespace num_four_digit_snappy_numbers_divisible_by_25_l155_155182

def is_snappy (n : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by_25 (n : ℕ) : Prop :=
  let last_two_digits := n % 100
  last_two_digits = 0 ∨ last_two_digits = 25 ∨ last_two_digits = 50 ∨ last_two_digits = 75

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem num_four_digit_snappy_numbers_divisible_by_25 : 
  ∃ n, n = 3 ∧ (∀ x, is_four_digit x ∧ is_snappy x ∧ is_divisible_by_25 x ↔ x = 5225 ∨ x = 0550 ∨ x = 5775)
:=
sorry

end num_four_digit_snappy_numbers_divisible_by_25_l155_155182


namespace arithmetic_seq_S11_l155_155104

def Sn (n : ℕ) (a₁ d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1)) / 2 * d

theorem arithmetic_seq_S11 (a₁ d : ℤ)
  (h1 : a₁ = -11)
  (h2 : (Sn 10 a₁ d) / 10 - (Sn 8 a₁ d) / 8 = 2) :
  Sn 11 a₁ d = -11 :=
by
  sorry

end arithmetic_seq_S11_l155_155104


namespace coordinates_of_foci_l155_155013

-- Given conditions
def equation_of_hyperbola : Prop := ∃ (x y : ℝ), (x^2 / 4) - (y^2 / 5) = 1

-- The mathematical goal translated into a theorem
theorem coordinates_of_foci (x y : ℝ) (a b c : ℝ) (ha : a^2 = 4) (hb : b^2 = 5) (hc : c^2 = a^2 + b^2) :
  equation_of_hyperbola →
  ((x = 3 ∨ x = -3) ∧ y = 0) :=
sorry

end coordinates_of_foci_l155_155013


namespace isosceles_triangle_perimeter_l155_155923

theorem isosceles_triangle_perimeter (x y : ℝ) (h : 4 * x ^ 2 + 17 * y ^ 2 - 16 * x * y - 4 * y + 4 = 0):
  x = 4 ∧ y = 2 → 2 * x + y = 10 :=
by
  intros
  sorry

end isosceles_triangle_perimeter_l155_155923


namespace sum_of_pairwise_relatively_prime_numbers_l155_155844

theorem sum_of_pairwise_relatively_prime_numbers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
    (h4 : a * b * c = 302400) (h5 : Nat.gcd a b = 1) (h6 : Nat.gcd b c = 1) (h7 : Nat.gcd a c = 1) :
    a + b + c = 320 :=
sorry

end sum_of_pairwise_relatively_prime_numbers_l155_155844


namespace enrollment_inversely_proportional_l155_155793

theorem enrollment_inversely_proportional :
  ∃ k : ℝ, (40 * 2000 = k) → (s * 2500 = k) → s = 32 :=
by
  sorry

end enrollment_inversely_proportional_l155_155793


namespace union_complement_A_eq_l155_155356

open Set

variable (U A B : Set ℕ)

-- Define the sets U, A, and B
def U := {1, 2, 3, 4, 5}
def A := {1, 3}
def B := {1, 2, 4}

theorem union_complement_A_eq : ((U \ B) ∪ A) = {1, 3, 5} := 
by
  sorry

end union_complement_A_eq_l155_155356


namespace C_finishes_work_in_days_l155_155525

theorem C_finishes_work_in_days :
  (∀ (unit : ℝ) (A B C combined: ℝ),
    combined = 1 / 4 ∧
    A = 1 / 12 ∧
    B = 1 / 24 ∧
    combined = A + B + 1 / C) → 
    C = 8 :=
  sorry

end C_finishes_work_in_days_l155_155525


namespace find_z_value_l155_155565

theorem find_z_value (k : ℝ) (y z : ℝ) (h1 : (y = 2) → (z = 1)) (h2 : y ^ 3 * z ^ (1/3) = k) : 
  (y = 4) → z = 1 / 512 :=
by
  sorry

end find_z_value_l155_155565


namespace final_weight_of_box_l155_155259

theorem final_weight_of_box (w1 w2 w3 w4 : ℝ) (h1 : w1 = 2) (h2 : w2 = 3 * w1) (h3 : w3 = w2 + 2) (h4 : w4 = 2 * w3) : w4 = 16 :=
by
  sorry

end final_weight_of_box_l155_155259


namespace intersection_eq_zero_set_l155_155266

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | x^2 ≤ 0}

theorem intersection_eq_zero_set : M ∩ N = {0} := by
  sorry

end intersection_eq_zero_set_l155_155266


namespace number_of_correct_answers_is_95_l155_155652

variable (x y : ℕ) -- Define x as the number of correct answers and y as the number of wrong answers

-- Define the conditions
axiom h1 : x + y = 150
axiom h2 : 5 * x - 2 * y = 370

-- State the goal we want to prove
theorem number_of_correct_answers_is_95 : x = 95 :=
by
  sorry

end number_of_correct_answers_is_95_l155_155652


namespace roger_forgot_lawns_l155_155132

theorem roger_forgot_lawns
  (dollars_per_lawn : ℕ)
  (total_lawns : ℕ)
  (total_earned : ℕ)
  (actual_mowed_lawns : ℕ)
  (forgotten_lawns : ℕ)
  (h1 : dollars_per_lawn = 9)
  (h2 : total_lawns = 14)
  (h3 : total_earned = 54)
  (h4 : actual_mowed_lawns = total_earned / dollars_per_lawn) :
  forgotten_lawns = total_lawns - actual_mowed_lawns :=
  sorry

end roger_forgot_lawns_l155_155132


namespace domain_range_equal_l155_155624

noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

theorem domain_range_equal {a b : ℝ} (hb : b > 0) :
  (∀ y, ∃ x, f a b x = y) ↔ (a = -4 ∨ a = 0) :=
sorry

end domain_range_equal_l155_155624


namespace age_ratio_l155_155148

theorem age_ratio (R D : ℕ) (h1 : R + 2 = 26) (h2 : D = 18) : R / D = 4 / 3 :=
sorry

end age_ratio_l155_155148


namespace ratio_equivalence_l155_155718

theorem ratio_equivalence (a b : ℝ) (hb : b ≠ 0) (h : a / b = 5 / 4) : (4 * a + 3 * b) / (4 * a - 3 * b) = 4 :=
sorry

end ratio_equivalence_l155_155718


namespace total_money_shared_l155_155258

theorem total_money_shared (ratio_jonah ratio_kira ratio_liam kira_share : ℕ)
  (h_ratio : ratio_jonah = 2) (h_ratio2 : ratio_kira = 3) (h_ratio3 : ratio_liam = 8)
  (h_kira : kira_share = 45) :
  (ratio_jonah * (kira_share / ratio_kira) + kira_share + ratio_liam * (kira_share / ratio_kira)) = 195 := 
by
  sorry

end total_money_shared_l155_155258


namespace solve_for_A_l155_155242

variable (x y : ℝ)

theorem solve_for_A (A : ℝ) : (2 * x - y) ^ 2 + A = (2 * x + y) ^ 2 → A = 8 * x * y :=
by
  intro h
  sorry

end solve_for_A_l155_155242


namespace fraction_sum_l155_155745

variable (a b : ℝ)

theorem fraction_sum
  (hb : b + 1 ≠ 0) :
  (a / (b + 1)) + (2 * a / (b + 1)) - (3 * a / (b + 1)) = 0 :=
by sorry

end fraction_sum_l155_155745


namespace neg_p_implies_neg_q_l155_155084

variable {x : ℝ}

def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

theorem neg_p_implies_neg_q (h : ¬ p x) : ¬ q x :=
sorry

end neg_p_implies_neg_q_l155_155084


namespace total_cement_used_l155_155131

def cement_used_lexi : ℝ := 10
def cement_used_tess : ℝ := 5.1

theorem total_cement_used : cement_used_lexi + cement_used_tess = 15.1 :=
by sorry

end total_cement_used_l155_155131


namespace trains_cross_each_other_in_given_time_l155_155421

noncomputable def trains_crossing_time (length1 length2 speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1 := (speed1_kmph * 1000) / 3600
  let speed2 := (speed2_kmph * 1000) / 3600
  let relative_speed := speed1 + speed2
  let total_distance := length1 + length2
  total_distance / relative_speed

theorem trains_cross_each_other_in_given_time :
  trains_crossing_time 300 400 36 18 = 46.67 :=
by
  -- expected proof here
  sorry

end trains_cross_each_other_in_given_time_l155_155421


namespace intersection_of_A_and_B_l155_155234

open Set

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 9} :=
by
  sorry

end intersection_of_A_and_B_l155_155234


namespace power_inequality_l155_155904

theorem power_inequality (a b c d : ℝ) (ha : 0 < a) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end power_inequality_l155_155904


namespace salt_concentration_solution_l155_155019

theorem salt_concentration_solution
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 2 * x + 3 * y = 35)
  (h3 : 3 * y + 2 * z = 45) :
  x = 10 ∧ y = 5 ∧ z = 15 := by
  sorry

end salt_concentration_solution_l155_155019


namespace total_views_correct_l155_155046

-- Definitions based on the given conditions
def initial_views : ℕ := 4000
def views_increase := 10 * initial_views
def additional_views := 50000
def total_views_after_6_days := initial_views + views_increase + additional_views

-- The theorem we are going to state
theorem total_views_correct :
  total_views_after_6_days = 94000 :=
sorry

end total_views_correct_l155_155046


namespace list_price_is_40_l155_155049

theorem list_price_is_40 (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to indicate we're skipping the proof.
  sorry

end list_price_is_40_l155_155049


namespace age_ratio_l155_155397

-- Definitions of the ages based on the given conditions.
def Rachel_age : ℕ := 12  -- Rachel's age
def Father_age_when_Rachel_25 : ℕ := 60

-- Defining Mother, Father, Grandfather ages based on given conditions.
def Grandfather_age (R : ℕ) (F : ℕ) : ℕ := 2 * (F - 5)
def Father_age (R : ℕ) : ℕ := Father_age_when_Rachel_25 - (25 - R)

-- Proving the ratio of Grandfather's age to Rachel's age is 7:1
theorem age_ratio (R : ℕ) (F : ℕ) (G : ℕ) :
  R = Rachel_age →
  F = Father_age R →
  G = Grandfather_age R F →
  G / R = 7 := by
  exact sorry

end age_ratio_l155_155397


namespace real_root_interval_l155_155528

theorem real_root_interval (b : ℝ) (p : ℝ → ℝ)
  (h_poly : p = λ x, x^2 + b * x + 25)
  (h_real_root : ∃ x : ℝ, p x = 0) :
  b ∈ set.Iic (-10) ∪ set.Ici 10 :=
sorry

end real_root_interval_l155_155528


namespace sum_of_solutions_eq_zero_l155_155776

theorem sum_of_solutions_eq_zero (x : ℝ) :
  (∃ x_1 x_2 : ℝ, (|x_1 - 20| + |x_2 + 20| = 2020) ∧ (x_1 + x_2 = 0)) :=
sorry

end sum_of_solutions_eq_zero_l155_155776


namespace ratio_of_female_to_male_members_l155_155471

theorem ratio_of_female_to_male_members 
  (f m : ℕ) 
  (avg_age_female : ℕ) 
  (avg_age_male : ℕ)
  (avg_age_all : ℕ) 
  (H1 : avg_age_female = 45)
  (H2 : avg_age_male = 25)
  (H3 : avg_age_all = 35)
  (H4 : (f + m) ≠ 0) :
  (45 * f + 25 * m) / (f + m) = 35 → f = m :=
by sorry

end ratio_of_female_to_male_members_l155_155471


namespace hot_drinks_sales_l155_155604

theorem hot_drinks_sales (x: ℝ) (h: x = 4) : abs ((-2.35 * x + 155.47) - 146) < 1 :=
by sorry

end hot_drinks_sales_l155_155604


namespace final_weight_is_sixteen_l155_155262

def initial_weight : ℤ := 0
def weight_after_jellybeans : ℤ := initial_weight + 2
def weight_after_brownies : ℤ := weight_after_jellybeans * 3
def weight_after_more_jellybeans : ℤ := weight_after_brownies + 2
def final_weight : ℤ := weight_after_more_jellybeans * 2

theorem final_weight_is_sixteen : final_weight = 16 := by
  sorry

end final_weight_is_sixteen_l155_155262


namespace project_assignment_l155_155621

open Nat

def binom (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem project_assignment :
  let A := 3
  let B := 1
  let C := 2
  let D := 2
  let total_projects := 8
  A + B + C + D = total_projects →
  (binom 8 3) * (binom 5 1) * (binom 4 2) * (binom 2 2) = 1680 :=
by
  intros
  sorry

end project_assignment_l155_155621


namespace spelling_bee_students_count_l155_155798

theorem spelling_bee_students_count (x : ℕ) (h1 : x / 2 * 1 / 4 * 2 = 30) : x = 240 :=
by
  sorry

end spelling_bee_students_count_l155_155798


namespace intersection_eq_l155_155515

def setA : Set ℝ := {x | -1 < x ∧ x < 1}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_eq : setA ∩ setB = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_eq_l155_155515


namespace average_pages_per_book_l155_155949

theorem average_pages_per_book :
  let pages := [120, 150, 180, 210, 240]
  let num_books := 5
  let total_pages := pages.sum
  total_pages / num_books = 180 := by
  sorry

end average_pages_per_book_l155_155949


namespace percentage_more_research_l155_155667

-- Defining the various times spent
def acclimation_period : ℝ := 1
def learning_basics_period : ℝ := 2
def dissertation_fraction : ℝ := 0.5
def total_time : ℝ := 7

-- Defining the time spent on each activity
def dissertation_period := dissertation_fraction * acclimation_period
def research_period := total_time - acclimation_period - learning_basics_period - dissertation_period

-- The main theorem to prove
theorem percentage_more_research : 
  ((research_period - learning_basics_period) / learning_basics_period) * 100 = 75 :=
by
  -- Placeholder for the proof
  sorry

end percentage_more_research_l155_155667


namespace number_of_members_l155_155600

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end number_of_members_l155_155600


namespace monotone_range_of_f_l155_155519

theorem monotone_range_of_f {f : ℝ → ℝ} (a : ℝ) 
  (h : ∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≤ y → f x ≤ f y) : a ≤ 0 :=
sorry

end monotone_range_of_f_l155_155519


namespace exists_same_color_points_at_distance_one_l155_155865

theorem exists_same_color_points_at_distance_one (coloring : ℝ × ℝ → Fin 3) :
  ∃ (p q : ℝ × ℝ), (coloring p = coloring q) ∧ (dist p q = 1) := sorry

end exists_same_color_points_at_distance_one_l155_155865


namespace calculate_retail_price_l155_155185

/-- Define the wholesale price of the machine. -/
def wholesale_price : ℝ := 90

/-- Define the profit rate as 20% of the wholesale price. -/
def profit_rate : ℝ := 0.20

/-- Define the discount rate as 10% of the retail price. -/
def discount_rate : ℝ := 0.10

/-- Calculate the profit based on the wholesale price. -/
def profit : ℝ := profit_rate * wholesale_price

/-- Calculate the selling price after the discount. -/
def selling_price (retail_price : ℝ) : ℝ := retail_price * (1 - discount_rate)

/-- Calculate the total selling price as the wholesale price plus profit. -/
def total_selling_price : ℝ := wholesale_price + profit

/-- State the theorem we need to prove. -/
theorem calculate_retail_price : ∃ R : ℝ, selling_price R = total_selling_price → R = 120 := by
  sorry

end calculate_retail_price_l155_155185


namespace candles_left_l155_155608

theorem candles_left (total_candles : ℕ) (alyssa_fraction_used : ℚ) (chelsea_fraction_used : ℚ) 
  (h_total : total_candles = 40) 
  (h_alyssa : alyssa_fraction_used = (1 / 2)) 
  (h_chelsea : chelsea_fraction_used = (70 / 100)) : 
  total_candles - (alyssa_fraction_used * total_candles).toNat - (chelsea_fraction_used * (total_candles - (alyssa_fraction_used * total_candles).toNat)).toNat = 6 :=
by 
  sorry

end candles_left_l155_155608


namespace prime_numbers_eq_l155_155203

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_eq 
  (p q r : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (h : p * (p - 7) + q * (q - 7) = r * (r - 7)) :
  (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 5 ∧ r = 7) ∨
  (p = 7 ∧ q = 5 ∧ r = 5) ∨ (p = 5 ∧ q = 7 ∧ r = 5) ∨
  (p = 5 ∧ q = 7 ∧ r = 2) ∨ (p = 7 ∧ q = 5 ∧ r = 2) ∨
  (p = 7 ∧ q = 3 ∧ r = 3) ∨ (p = 3 ∧ q = 7 ∧ r = 3) ∨
  (∃ (a : ℕ), is_prime a ∧ p = a ∧ q = 7 ∧ r = a) ∨
  (∃ (a : ℕ), is_prime a ∧ p = 7 ∧ q = a ∧ r = a) :=
sorry

end prime_numbers_eq_l155_155203


namespace find_other_number_l155_155961

-- Define LCM and HCF conditions
def lcm_a_b := 2310
def hcf_a_b := 83
def number_a := 210

-- Define the problem to find the other number
def number_b : ℕ :=
  lcm_a_b * hcf_a_b / number_a

-- Statement: Prove that the other number is 913
theorem find_other_number : number_b = 913 := by
  -- Placeholder for proof
  sorry

end find_other_number_l155_155961


namespace unpainted_cubes_l155_155313

theorem unpainted_cubes (n : ℕ) (cubes_per_face : ℕ) (faces : ℕ) (total_cubes : ℕ) (painted_cubes : ℕ) :
  n = 6 → cubes_per_face = 4 → faces = 6 → total_cubes = 216 → painted_cubes = 24 → 
  total_cubes - painted_cubes = 192 := by
  intros
  sorry

end unpainted_cubes_l155_155313


namespace sanya_towels_count_l155_155005

-- Defining the conditions based on the problem
def towels_per_hour := 7
def hours_per_day := 2
def days_needed := 7

-- The main statement to prove
theorem sanya_towels_count : 
  (towels_per_hour * hours_per_day * days_needed = 98) :=
by
  sorry

end sanya_towels_count_l155_155005


namespace annie_spent_on_candies_l155_155876

theorem annie_spent_on_candies : 
  ∀ (num_classmates : ℕ) (candies_per_classmate : ℕ) (candies_left : ℕ) (cost_per_candy : ℚ),
  num_classmates = 35 →
  candies_per_classmate = 2 →
  candies_left = 12 →
  cost_per_candy = 0.1 →
  (num_classmates * candies_per_classmate + candies_left) * cost_per_candy = 8.2 :=
by
  intros num_classmates candies_per_classmate candies_left cost_per_candy
         h_classmates h_candies_per_classmate h_candies_left h_cost_per_candy
  simp [h_classmates, h_candies_per_classmate, h_candies_left, h_cost_per_candy]
  sorry

end annie_spent_on_candies_l155_155876


namespace values_of_x_l155_155504

theorem values_of_x (x : ℤ) :
  (∃ t : ℤ, x = 105 * t + 22) ∨ (∃ t : ℤ, x = 105 * t + 37) ↔ 
  (5 * x^3 - x + 17) % 15 = 0 ∧ (2 * x^2 + x - 3) % 7 = 0 :=
by {
  sorry
}

end values_of_x_l155_155504


namespace no_such_function_exists_l155_155208

noncomputable def f : ℕ → ℕ := sorry

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), (∀ n > 1, f n = f (f (n-1)) + f (f (n+1))) ∧ (∀ n, f n > 0) :=
sorry

end no_such_function_exists_l155_155208


namespace percentage_waiting_for_parts_l155_155257

def totalComputers : ℕ := 20
def unfixableComputers : ℕ := (20 * 20) / 100
def fixedRightAway : ℕ := 8
def waitingForParts : ℕ := totalComputers - (unfixableComputers + fixedRightAway)

theorem percentage_waiting_for_parts : (waitingForParts : ℝ) / totalComputers * 100 = 40 := 
by 
  have : waitingForParts = 8 := sorry
  have : (8 / 20 : ℝ) * 100 = 40 := sorry
  exact sorry

end percentage_waiting_for_parts_l155_155257


namespace product_three_power_l155_155453

theorem product_three_power (w : ℕ) (hW : w = 132) (hProd : ∃ (k : ℕ), 936 * w = 2^5 * 11^2 * k) : 
  ∃ (n : ℕ), (936 * w) = (2^5 * 11^2 * (3^3 * n)) :=
by 
  sorry

end product_three_power_l155_155453


namespace Auston_height_in_cm_l155_155474

theorem Auston_height_in_cm : 
  (60 : ℝ) * 2.54 = 152.4 :=
by sorry

end Auston_height_in_cm_l155_155474


namespace greatest_possible_positive_integer_difference_l155_155524

theorem greatest_possible_positive_integer_difference (x y : ℤ) (hx : 4 < x) (hx' : x < 6) (hy : 6 < y) (hy' : y < 10) :
  y - x = 4 :=
sorry

end greatest_possible_positive_integer_difference_l155_155524


namespace simplification_l155_155134

-- Define all relevant powers
def pow2_8 : ℤ := 2^8
def pow4_5 : ℤ := 4^5
def pow2_3 : ℤ := 2^3
def pow_neg2_2 : ℤ := (-2)^2

-- Define the expression inside the parentheses
def inner_expr : ℤ := pow2_3 - pow_neg2_2

-- Define the exponentiation of the inner expression
def inner_expr_pow11 : ℤ := inner_expr^11

-- Define the entire expression
def full_expr : ℤ := (pow2_8 + pow4_5) * inner_expr_pow11

-- State the proof goal
theorem simplification : full_expr = 5368709120 := by
  sorry

end simplification_l155_155134


namespace problem_statement_l155_155530

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.arccos (inner a b / (‖a‖ * ‖b‖))

theorem problem_statement
  (a b : EuclideanSpace ℝ (Fin 3))
  (h_angle_ab : angle_between_vectors a b = Real.pi / 3)
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 1) :
  angle_between_vectors a (a + 2 • b) = Real.pi / 6 :=
sorry

end problem_statement_l155_155530


namespace range_of_a_l155_155771

noncomputable def exists_unique_y (a : ℝ) (x : ℝ) : Prop :=
∃! (y : ℝ), y ∈ Set.Icc (-1) 1 ∧ x + y^2 * Real.exp y = a

theorem range_of_a (e : ℝ) (H_e : e = Real.exp 1) :
  (∀ x ∈ Set.Icc 0 1, exists_unique_y a x) →
  a ∈ Set.Ioc (1 + 1/e) e :=
by
  sorry

end range_of_a_l155_155771


namespace exists_numbers_with_prime_sum_and_product_l155_155379

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem exists_numbers_with_prime_sum_and_product :
  ∃ a b c : ℕ, is_prime (a + b + c) ∧ is_prime (a * b * c) :=
  by
    -- First import the prime definitions and variables.
    let a := 1
    let b := 1
    let c := 3
    have h1 : is_prime (a + b + c) := by sorry
    have h2 : is_prime (a * b * c) := by sorry
    exact ⟨a, b, c, h1, h2⟩

end exists_numbers_with_prime_sum_and_product_l155_155379


namespace sum_and_count_l155_155305

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count (x y : ℕ) (hx : x = sum_of_integers 30 50) (hy : y = count_even_integers 30 50) :
  x + y = 851 :=
by
  -- proof goes here
  sorry

end sum_and_count_l155_155305


namespace point_2023_0_cannot_lie_on_line_l155_155366

-- Define real numbers a and c with the condition ac > 0
variables (a c : ℝ)

-- The condition ac > 0
def ac_positive := (a * c > 0)

-- The statement that (2023, 0) cannot be on the line y = ax + c given the condition a * c > 0
theorem point_2023_0_cannot_lie_on_line (h : ac_positive a c) : ¬ (0 = 2023 * a + c) :=
sorry

end point_2023_0_cannot_lie_on_line_l155_155366


namespace classrooms_student_rabbit_difference_l155_155064

-- Definitions from conditions
def students_per_classroom : Nat := 20
def rabbits_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Theorem statement
theorem classrooms_student_rabbit_difference :
  (students_per_classroom * number_of_classrooms) - (rabbits_per_classroom * number_of_classrooms) = 102 := by
  sorry

end classrooms_student_rabbit_difference_l155_155064


namespace find_side_AB_l155_155935

theorem find_side_AB 
  (B C : ℝ) (BC : ℝ) (hB : B = 45) (hC : C = 45) (hBC : BC = 10) : 
  ∃ AB : ℝ, AB = 5 * Real.sqrt 2 :=
by
  -- We add 'sorry' here to indicate that the proof is not provided.
  sorry

end find_side_AB_l155_155935


namespace germany_fraction_closest_japan_fraction_closest_l155_155757

noncomputable def fraction_approx (a b : ℕ) : ℚ := a / b

theorem germany_fraction_closest :
  abs (fraction_approx 23 150 - fraction_approx 1 7) < 
  min (abs (fraction_approx 23 150 - fraction_approx 1 5))
      (min (abs (fraction_approx 23 150 - fraction_approx 1 6))
           (min (abs (fraction_approx 23 150 - fraction_approx 1 8))
                (abs (fraction_approx 23 150 - fraction_approx 1 9)))) :=
by sorry

theorem japan_fraction_closest :
  abs (fraction_approx 27 150 - fraction_approx 1 6) < 
  min (abs (fraction_approx 27 150 - fraction_approx 1 5))
      (min (abs (fraction_approx 27 150 - fraction_approx 1 7))
           (min (abs (fraction_approx 27 150 - fraction_approx 1 8))
                (abs (fraction_approx 27 150 - fraction_approx 1 9)))) :=
by sorry

end germany_fraction_closest_japan_fraction_closest_l155_155757


namespace largest_possible_pencils_in_each_package_l155_155396

def ming_pencils : ℕ := 48
def catherine_pencils : ℕ := 36
def lucas_pencils : ℕ := 60

theorem largest_possible_pencils_in_each_package (d : ℕ) (h_ming: ming_pencils % d = 0) (h_catherine: catherine_pencils % d = 0) (h_lucas: lucas_pencils % d = 0) : d ≤ ming_pencils ∧ d ≤ catherine_pencils ∧ d ≤ lucas_pencils ∧ (∀ e, (ming_pencils % e = 0 ∧ catherine_pencils % e = 0 ∧ lucas_pencils % e = 0) → e ≤ d) → d = 12 :=
by 
  sorry

end largest_possible_pencils_in_each_package_l155_155396


namespace final_weight_of_box_l155_155260

theorem final_weight_of_box (w1 w2 w3 w4 : ℝ) (h1 : w1 = 2) (h2 : w2 = 3 * w1) (h3 : w3 = w2 + 2) (h4 : w4 = 2 * w3) : w4 = 16 :=
by
  sorry

end final_weight_of_box_l155_155260


namespace solution_set_of_inequality_l155_155141

variable (f : ℝ → ℝ)
variable (h_inc : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y)

theorem solution_set_of_inequality :
  {x | 0 < x ∧ f x > f (2 * x - 4)} = {x | 2 < x ∧ x < 4} :=
by
  sorry

end solution_set_of_inequality_l155_155141


namespace supplement_of_angle_l155_155639

theorem supplement_of_angle (A : ℝ) (h : 90 - A = A - 18) : 180 - A = 126 := by
    sorry

end supplement_of_angle_l155_155639


namespace cobbler_works_fri_hours_l155_155857

-- Conditions
def mending_rate : ℕ := 3  -- Pairs of shoes per hour
def mon_to_thu_days : ℕ := 4
def hours_per_day : ℕ := 8
def weekly_mended_pairs : ℕ := 105

-- Translate the conditions
def hours_mended_mon_to_thu : ℕ := mon_to_thu_days * hours_per_day
def pairs_mended_mon_to_thu : ℕ := mending_rate * hours_mended_mon_to_thu
def pairs_mended_fri : ℕ := weekly_mended_pairs - pairs_mended_mon_to_thu

-- Theorem statement to prove the desired question
theorem cobbler_works_fri_hours : (pairs_mended_fri / mending_rate) = 3 := by
  sorry

end cobbler_works_fri_hours_l155_155857


namespace increasing_function_range_b_l155_155791

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (b - 3 / 2) * x + b - 1 else -x^2 + (2 - b) * x

theorem increasing_function_range_b :
  (∀ x y, x < y → f b x ≤ f b y) ↔ (3 / 2 < b ∧ b ≤ 2 ) := 
by
  sorry

end increasing_function_range_b_l155_155791


namespace option_d_correct_l155_155590

theorem option_d_correct (a b : ℝ) (h : a * b < 0) : 
  (a / b + b / a) ≤ -2 := by
  sorry

end option_d_correct_l155_155590


namespace Nina_now_l155_155278

def Lisa_age (l m n : ℝ) := l + m + n = 36
def Nina_age (l n : ℝ) := n - 5 = 2 * l
def Mike_age (l m : ℝ) := m + 2 = (l + 2) / 2

theorem Nina_now (l m n : ℝ) (h1 : Lisa_age l m n) (h2 : Nina_age l n) (h3 : Mike_age l m) : n = 34.6 := by
  sorry

end Nina_now_l155_155278


namespace problem_solution_l155_155348

def lean_problem (a : ℝ) : Prop :=
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1)^x₁ > (2 * a - 1)^x₂) →
  a > 1 / 2 ∧ a ≤ 2 / 3

theorem problem_solution (a : ℝ) : lean_problem a :=
  sorry -- Proof is to be filled in

end problem_solution_l155_155348


namespace consecutive_grouping_probability_l155_155726

theorem consecutive_grouping_probability :
  let green_factorial := Nat.factorial 4
  let orange_factorial := Nat.factorial 3
  let blue_factorial := Nat.factorial 5
  let block_arrangements := Nat.factorial 3
  let total_arrangements := Nat.factorial 12
  (block_arrangements * green_factorial * orange_factorial * blue_factorial) / total_arrangements = 1 / 4620 :=
by
  let green_factorial := Nat.factorial 4
  let orange_factorial := Nat.factorial 3
  let blue_factorial := Nat.factorial 5
  let block_arrangements := Nat.factorial 3
  let total_arrangements := Nat.factorial 12
  have h : (block_arrangements * green_factorial * orange_factorial * blue_factorial) = 103680 := sorry
  have h1 : (total_arrangements) = 479001600 := sorry
  calc
    (block_arrangements * green_factorial * orange_factorial * blue_factorial) / total_arrangements
    _ = 103680 / 479001600 := by rw [h, h1]
    _ = 1 / 4620 := sorry

end consecutive_grouping_probability_l155_155726


namespace bill_initial_amount_l155_155194

/-- Suppose Ann has $777 and Bill gives Ann $167,
    after which they both have the same amount of money. 
    Prove that Bill initially had $1111. -/
theorem bill_initial_amount (A B : ℕ) (h₁ : A = 777) (h₂ : B - 167 = A + 167) : B = 1111 :=
by
  -- Proof goes here
  sorry

end bill_initial_amount_l155_155194


namespace tom_catches_48_trout_l155_155394

variable (melanie_tom_catch_ratio : ℕ := 3)
variable (melanie_catch : ℕ := 16)

theorem tom_catches_48_trout (h1 : melanie_catch = 16) (h2 : melanie_tom_catch_ratio = 3) : (melanie_tom_catch_ratio * melanie_catch) = 48 :=
by
  sorry

end tom_catches_48_trout_l155_155394


namespace find_triples_l155_155895

theorem find_triples (a m n : ℕ) (h1 : a ≥ 2) (h2 : m ≥ 2) :
  a^n + 203 ∣ a^(m * n) + 1 → ∃ (k : ℕ), (k ≥ 1) := 
sorry

end find_triples_l155_155895


namespace constant_expression_l155_155219

-- Suppose x is a real number
variable {x : ℝ}

-- Define the expression sum
def expr_sum (x : ℝ) : ℝ :=
|3 * x - 1| + |4 * x - 1| + |5 * x - 1| + |6 * x - 1| + 
|7 * x - 1| + |8 * x - 1| + |9 * x - 1| + |10 * x - 1| + 
|11 * x - 1| + |12 * x - 1| + |13 * x - 1| + |14 * x - 1| + 
|15 * x - 1| + |16 * x - 1| + |17 * x - 1|

-- The Lean statement of the problem to be proven
theorem constant_expression : (∃ x : ℝ, expr_sum x = 5) :=
sorry

end constant_expression_l155_155219


namespace middle_number_l155_155828

theorem middle_number (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : a + b = 18) (h4 : a + c = 23) (h5 : b + c = 27) : b = 11 := by
  sorry

end middle_number_l155_155828


namespace programs_produce_same_result_l155_155193

-- Define Program A's computation
def programA_sum : ℕ := (List.range (1000 + 1)).sum -- Sum of numbers from 0 to 1000

-- Define Program B's computation
def programB_sum : ℕ := (List.range (1000 + 1)).reverse.sum -- Sum of numbers from 1000 down to 0

theorem programs_produce_same_result : programA_sum = programB_sum :=
  sorry

end programs_produce_same_result_l155_155193


namespace ratio_of_original_to_reversed_l155_155705

def original_number : ℕ := 21
def reversed_number : ℕ := 12

theorem ratio_of_original_to_reversed : 
  (original_number : ℚ) / (reversed_number : ℚ) = 7 / 4 := by
  sorry

end ratio_of_original_to_reversed_l155_155705


namespace Benny_total_hours_l155_155200

def hours_per_day : ℕ := 7
def days_worked : ℕ := 14

theorem Benny_total_hours : hours_per_day * days_worked = 98 := by
  sorry

end Benny_total_hours_l155_155200


namespace total_weekly_reading_time_l155_155918

def morning_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def morning_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

def evening_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def evening_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

theorem total_weekly_reading_time :
  let morning_minutes := 30
  let evening_minutes := 60
  let weekdays := 5
  let weekend_days := 2
  morning_reading_weekdays morning_minutes weekdays +
  morning_reading_weekends morning_minutes +
  evening_reading_weekdays evening_minutes weekdays +
  evening_reading_weekends evening_minutes = 810 :=
by
  sorry

end total_weekly_reading_time_l155_155918


namespace simplify_expression_l155_155272

theorem simplify_expression (x : ℝ) : (3 * x)^5 + (5 * x) * (x^4) - 7 * x^5 = 241 * x^5 := 
by
  sorry

end simplify_expression_l155_155272


namespace repeating_decimal_sum_l155_155629

theorem repeating_decimal_sum :
  let x := (0.3333333333333333 : ℚ) -- 0.\overline{3}
  let y := (0.0707070707070707 : ℚ) -- 0.\overline{07}
  let z := (0.008008008008008 : ℚ)  -- 0.\overline{008}
  x + y + z = 418 / 999 := by
sorry

end repeating_decimal_sum_l155_155629


namespace number_of_fish_given_to_dog_l155_155544

-- Define the conditions
def condition1 (D C : ℕ) : Prop := C = D / 2
def condition2 (D C : ℕ) : Prop := D + C = 60

-- Theorem to prove the number of fish given to the dog
theorem number_of_fish_given_to_dog (D : ℕ) (C : ℕ) (h1 : condition1 D C) (h2 : condition2 D C) : D = 40 :=
by
  sorry

end number_of_fish_given_to_dog_l155_155544


namespace confidence_k_squared_l155_155930

-- Define the condition for 95% confidence relation between events A and B
def confidence_95 (A B : Prop) : Prop := 
  -- Placeholder for the actual definition, assume 95% confidence implies a specific condition
  True

-- Define the data value and critical value condition
def K_squared : ℝ := sorry  -- Placeholder for the actual K² value

theorem confidence_k_squared (A B : Prop) (h : confidence_95 A B) : K_squared > 3.841 := 
by
  sorry  -- Proof is not required, only the statement

end confidence_k_squared_l155_155930


namespace cans_needed_eq_l155_155950

axiom Paula_initial_rooms : ℕ
axiom Paula_lost_cans : ℕ
axiom Paula_after_loss_rooms : ℕ
axiom cans_for_25_rooms : ℕ

theorem cans_needed_eq :
  Paula_initial_rooms = 30 →
  Paula_lost_cans = 3 →
  Paula_after_loss_rooms = 25 →
  cans_for_25_rooms = 15 :=
by
  intros
  sorry

end cans_needed_eq_l155_155950


namespace sum_of_primes_between_1_and_20_l155_155993

theorem sum_of_primes_between_1_and_20 :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 :=
by
  sorry

end sum_of_primes_between_1_and_20_l155_155993


namespace f_decreasing_on_0_1_l155_155679

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x⁻¹

theorem f_decreasing_on_0_1 : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end f_decreasing_on_0_1_l155_155679


namespace jenna_costume_l155_155663

def cost_of_skirts (skirt_count : ℕ) (material_per_skirt : ℕ) : ℕ :=
  skirt_count * material_per_skirt

def cost_of_bodice (shirt_material : ℕ) (sleeve_material_per : ℕ) (sleeve_count : ℕ) : ℕ :=
  shirt_material + (sleeve_material_per * sleeve_count)

def total_material (skirt_material : ℕ) (bodice_material : ℕ) : ℕ :=
  skirt_material + bodice_material

def total_cost (total_material : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  total_material * cost_per_sqft

theorem jenna_costume : 
  cost_of_skirts 3 48 + cost_of_bodice 2 5 2 = 156 → total_cost 156 3 = 468 :=
by
  sorry

end jenna_costume_l155_155663


namespace Jenna_total_cost_l155_155661

theorem Jenna_total_cost :
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  total_cost = 468 :=
by
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  show total_cost = 468
  sorry

end Jenna_total_cost_l155_155661


namespace problem_number_of_true_propositions_l155_155962

open Set

variable {α : Type*} {A B : Set α}

def card (s : Set α) : ℕ := sorry -- The actual definition of cardinality is complex and in LF (not imperative here).

-- Statement of the problem translated into a Lean statement
theorem problem_number_of_true_propositions :
  (∀ {A B : Set ℕ}, A ∩ B = ∅ ↔ card (A ∪ B) = card A + card B) ∧
  (∀ {A B : Set ℕ}, A ⊆ B → card A ≤ card B) ∧
  (∀ {A B : Set ℕ}, A ⊂ B → card A < card B) →
   (3 = 3) :=
by 
  sorry


end problem_number_of_true_propositions_l155_155962


namespace jeans_more_than_scarves_l155_155315

def num_ties := 34
def num_belts := 40
def num_black_shirts := 63
def num_white_shirts := 42
def num_jeans := (2 / 3) * (num_black_shirts + num_white_shirts)
def num_scarves := (1 / 2) * (num_ties + num_belts)

theorem jeans_more_than_scarves : num_jeans - num_scarves = 33 := by
  sorry

end jeans_more_than_scarves_l155_155315


namespace max_value_xy_l155_155549

open Real

theorem max_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 5 * y < 100) :
  ∃ (c : ℝ), c = 3703.7 ∧ ∀ (x' y' : ℝ), 0 < x' → 0 < y' → 2 * x' + 5 * y' < 100 → x' * y' * (100 - 2 * x' - 5 * y') ≤ c :=
sorry

end max_value_xy_l155_155549


namespace time_for_A_to_complete_work_l155_155856

-- Defining the work rates and the condition
def workRateA (a : ℕ) : ℚ := 1 / a
def workRateB : ℚ := 1 / 12
def workRateC : ℚ := 1 / 24
def combinedWorkRate (a : ℕ) : ℚ := workRateA a + workRateB + workRateC
def togetherWorkRate : ℚ := 1 / 4

-- Stating the theorem
theorem time_for_A_to_complete_work : 
  ∃ (a : ℕ), combinedWorkRate a = togetherWorkRate ∧ a = 8 :=
by
  sorry

end time_for_A_to_complete_work_l155_155856


namespace largest_four_digit_divisible_by_six_l155_155427

theorem largest_four_digit_divisible_by_six :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n ≤ 9999) ∧ 
    (n % 6 = 0) ∧ 
    (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 6 = 0) → m ≤ n) :=
begin
  existsi 9996,
  split, 
  exact ⟨by norm_num, by norm_num⟩,
  split, 
  exact dec_trivial,
  intro m,
  intro h,
  exact ⟨by norm_num [h.1], by norm_num [h.2]⟩
end

end largest_four_digit_divisible_by_six_l155_155427


namespace value_of_expression_l155_155898

theorem value_of_expression (x y : ℚ) (hx : x = 2/3) (hy : y = 5/8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 := 
by
  sorry

end value_of_expression_l155_155898


namespace find_x_from_percents_l155_155312

theorem find_x_from_percents (x : ℝ) (h : 0.65 * x = 0.20 * 487.50) : x = 150 :=
by
  -- Distilled condition from problem
  have h1 : 0.65 * x = 0.20 * 487.50 := h
  -- Start actual logic here
  sorry

end find_x_from_percents_l155_155312


namespace rectangle_perimeter_is_70_l155_155462

-- Define the length and width of the rectangle
def length : ℕ := 19
def width : ℕ := 16

-- Define the perimeter function for a rectangle
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem statement asserting that the perimeter of the given rectangle is 70 cm
theorem rectangle_perimeter_is_70 :
  perimeter length width = 70 := 
sorry

end rectangle_perimeter_is_70_l155_155462


namespace sequence_general_formula_l155_155078

theorem sequence_general_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2 - 2 * n + 2):
  (a 1 = 1) ∧ (∀ n, 1 < n → a n = S n - S (n - 1)) → 
  (∀ n, a n = if n = 1 then 1 else 2 * n - 3) :=
by
  intro h
  sorry

end sequence_general_formula_l155_155078


namespace tan_sum_half_l155_155813

theorem tan_sum_half (a b : ℝ) (h1 : Real.cos a + Real.cos b = 3/5) (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1 / 3 := 
by
  sorry

end tan_sum_half_l155_155813
