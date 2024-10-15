import Mathlib

namespace NUMINAMATH_GPT_sequence_a6_value_l218_21872

theorem sequence_a6_value 
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n : ℕ, n ≥ 1 → (1 / a n) + (1 / a (n + 2)) = 2 / a (n + 1)) :
  a 6 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a6_value_l218_21872


namespace NUMINAMATH_GPT_myrtle_eggs_count_l218_21862

-- Definition for daily egg production
def daily_eggs : ℕ := 3 * 3

-- Definition for the number of days Myrtle is gone
def days_gone : ℕ := 7

-- Definition for total eggs laid
def total_eggs : ℕ := daily_eggs * days_gone

-- Definition for eggs taken by neighbor
def eggs_taken_by_neighbor : ℕ := 12

-- Definition for eggs remaining after neighbor takes some
def eggs_after_neighbor : ℕ := total_eggs - eggs_taken_by_neighbor

-- Definition for eggs dropped by Myrtle
def eggs_dropped_by_myrtle : ℕ := 5

-- Definition for total remaining eggs Myrtle has
def eggs_remaining : ℕ := eggs_after_neighbor - eggs_dropped_by_myrtle

-- Theorem statement
theorem myrtle_eggs_count : eggs_remaining = 46 := by
  sorry

end NUMINAMATH_GPT_myrtle_eggs_count_l218_21862


namespace NUMINAMATH_GPT_largest_n_satisfies_l218_21860

noncomputable def sin_plus_cos_bound (n : ℕ) (x : ℝ) : Prop :=
  (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * Real.sqrt n)

theorem largest_n_satisfies :
  ∃ (n : ℕ), (∀ x : ℝ, sin_plus_cos_bound n x) ∧
  ∀ m : ℕ, (∀ x : ℝ, sin_plus_cos_bound m x) → m ≤ 2 := 
sorry

end NUMINAMATH_GPT_largest_n_satisfies_l218_21860


namespace NUMINAMATH_GPT_chocolate_cost_is_75_l218_21821

def candy_bar_cost : ℕ := 25
def juice_pack_cost : ℕ := 50
def num_quarters : ℕ := 11
def total_cost_in_cents : ℕ := num_quarters * candy_bar_cost
def num_candy_bars : ℕ := 3
def num_pieces_of_chocolate : ℕ := 2

def chocolate_cost_in_cents (x : ℕ) : Prop :=
  (num_candy_bars * candy_bar_cost) + (num_pieces_of_chocolate * x) + juice_pack_cost = total_cost_in_cents

theorem chocolate_cost_is_75 : chocolate_cost_in_cents 75 :=
  sorry

end NUMINAMATH_GPT_chocolate_cost_is_75_l218_21821


namespace NUMINAMATH_GPT_Mason_fathers_age_indeterminate_l218_21800

theorem Mason_fathers_age_indeterminate
  (Mason_age : ℕ) (Sydney_age Mason_father_age D : ℕ)
  (hM : Mason_age = 20)
  (hS_M : Mason_age = Sydney_age / 3)
  (hS_F : Mason_father_age - D = Sydney_age) :
  ¬ ∃ F, Mason_father_age = F :=
by {
  sorry
}

end NUMINAMATH_GPT_Mason_fathers_age_indeterminate_l218_21800


namespace NUMINAMATH_GPT_Mona_joined_groups_l218_21875

theorem Mona_joined_groups (G : ℕ) (h : G * 4 - 3 = 33) : G = 9 :=
by
  sorry

end NUMINAMATH_GPT_Mona_joined_groups_l218_21875


namespace NUMINAMATH_GPT_arithmetic_result_l218_21822

theorem arithmetic_result :
  1325 + (572 / 52) - 225 + (2^3) = 1119 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_result_l218_21822


namespace NUMINAMATH_GPT_abs_eq_neg_l218_21856

theorem abs_eq_neg (x : ℝ) (h : |x + 6| = -(x + 6)) : x ≤ -6 :=
by 
  sorry

end NUMINAMATH_GPT_abs_eq_neg_l218_21856


namespace NUMINAMATH_GPT_mean_cost_of_diesel_l218_21843

-- Define the diesel rates and the number of years.
def dieselRates : List ℝ := [1.2, 1.3, 1.8, 2.1]
def years : ℕ := 4

-- Define the mean calculation and the proof requirement.
theorem mean_cost_of_diesel (h₁ : dieselRates = [1.2, 1.3, 1.8, 2.1]) 
                               (h₂ : years = 4) : 
  (dieselRates.sum / years) = 1.6 :=
by
  sorry

end NUMINAMATH_GPT_mean_cost_of_diesel_l218_21843


namespace NUMINAMATH_GPT_sum_of_digits_of_special_two_digit_number_l218_21865

theorem sum_of_digits_of_special_two_digit_number (x : ℕ) (h1 : 1 ≤ x ∧ x < 10) 
  (h2 : ∃ (n : ℕ), n = 11 * x + 30) 
  (h3 : ∃ (sum_digits : ℕ), sum_digits = (x + 3) + x) 
  (h4 : (11 * x + 30) % ((x + 3) + x) = 3)
  (h5 : (11 * x + 30) / ((x + 3) + x) = 7) :
  (x + 3) + x = 7 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_special_two_digit_number_l218_21865


namespace NUMINAMATH_GPT_minimum_value_l218_21887

-- Define the geometric sequence and its conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (positive : ∀ n, 0 < a n)
variable (geometric_seq : ∀ n, a (n+1) = q * a n)
variable (condition1 : a 6 = a 5 + 2 * a 4)
variable (m n : ℕ)
variable (condition2 : ∀ m n, sqrt (a m * a n) = 2 * a 1 → a m = a n)

-- Prove that the minimum value of 1/m + 9/n is 4
theorem minimum_value : m + n = 4 → (∀ x y : ℝ, (0 < x ∧ 0 < y) → (1 / x + 9 / y) ≥ 4) :=
sorry

end NUMINAMATH_GPT_minimum_value_l218_21887


namespace NUMINAMATH_GPT_greg_spent_on_shirt_l218_21813

-- Define the conditions in Lean
variables (S H : ℤ)
axiom condition1 : H = 2 * S + 9
axiom condition2 : S + H = 300

-- State the theorem to prove
theorem greg_spent_on_shirt : S = 97 :=
by
  sorry

end NUMINAMATH_GPT_greg_spent_on_shirt_l218_21813


namespace NUMINAMATH_GPT_Alan_shells_l218_21808

theorem Alan_shells (l b a : ℕ) (h1 : l = 36) (h2 : b = l / 3) (h3 : a = 4 * b) : a = 48 :=
by
sorry

end NUMINAMATH_GPT_Alan_shells_l218_21808


namespace NUMINAMATH_GPT_larger_angle_at_3_30_l218_21840

def hour_hand_angle_3_30 : ℝ := 105.0
def minute_hand_angle_3_30 : ℝ := 180.0
def smaller_angle_between_hands : ℝ := abs (minute_hand_angle_3_30 - hour_hand_angle_3_30)
def larger_angle_between_hands : ℝ := 360.0 - smaller_angle_between_hands

theorem larger_angle_at_3_30 :
  larger_angle_between_hands = 285.0 := 
  sorry

end NUMINAMATH_GPT_larger_angle_at_3_30_l218_21840


namespace NUMINAMATH_GPT_cost_500_pencils_is_25_dollars_l218_21855

def cost_of_500_pencils (cost_per_pencil : ℕ) (pencils : ℕ) (cents_per_dollar : ℕ) : ℕ :=
  (cost_per_pencil * pencils) / cents_per_dollar

theorem cost_500_pencils_is_25_dollars : cost_of_500_pencils 5 500 100 = 25 := by
  sorry

end NUMINAMATH_GPT_cost_500_pencils_is_25_dollars_l218_21855


namespace NUMINAMATH_GPT_total_amount_spent_l218_21889

variable (your_spending : ℝ) (friend_spending : ℝ)
variable (h1 : friend_spending = your_spending + 3) (h2 : friend_spending = 10)

theorem total_amount_spent : your_spending + friend_spending = 17 :=
by sorry

end NUMINAMATH_GPT_total_amount_spent_l218_21889


namespace NUMINAMATH_GPT_fraction_value_l218_21836

variable (x y : ℝ)

theorem fraction_value (h : 1/x + 1/y = 2) : (2*x + 5*x*y + 2*y) / (x - 3*x*y + y) = -9 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l218_21836


namespace NUMINAMATH_GPT_four_consecutive_product_divisible_by_12_l218_21804

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end NUMINAMATH_GPT_four_consecutive_product_divisible_by_12_l218_21804


namespace NUMINAMATH_GPT_negation_of_P_is_exists_ge_1_l218_21891

theorem negation_of_P_is_exists_ge_1 :
  let P := ∀ x : ℤ, x < 1
  ¬P ↔ ∃ x : ℤ, x ≥ 1 := by
  sorry

end NUMINAMATH_GPT_negation_of_P_is_exists_ge_1_l218_21891


namespace NUMINAMATH_GPT_unique_n_for_solutions_l218_21829

theorem unique_n_for_solutions :
  ∃! (n : ℕ), (∀ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (3 * x + 3 * y + 2 * z = n)) → 
  ((∃ (s : ℕ), s = 10) ∧ (n = 17)) :=
sorry

end NUMINAMATH_GPT_unique_n_for_solutions_l218_21829


namespace NUMINAMATH_GPT_value_of_c_over_b_l218_21880

def is_median (a b c : ℤ) (m : ℤ) : Prop :=
a < b ∧ b < c ∧ m = b

def in_geometric_progression (p q r : ℤ) : Prop :=
∃ k : ℤ, k ≠ 0 ∧ q = p * k ∧ r = q * k

theorem value_of_c_over_b (a b c p q r : ℤ) 
  (h1 : (a + b + c) / 3 = (b / 2))
  (h2 : a * b * c = 0)
  (h3 : a < b ∧ b < c ∧ a = 0)
  (h4 : p < q ∧ q < r ∧ r ≠ 0)
  (h5 : in_geometric_progression p q r)
  (h6 : a^2 + b^2 + c^2 = (p + q + r)^2) : 
  c / b = 2 := 
sorry

end NUMINAMATH_GPT_value_of_c_over_b_l218_21880


namespace NUMINAMATH_GPT_cost_of_largest_pot_is_2_52_l218_21812

/-
Mark bought a set of 6 flower pots of different sizes at a total pre-tax cost.
Each pot cost 0.4 more than the next one below it in size.
The total cost, including a sales tax of 7.5%, was $9.80.
Prove that the cost of the largest pot before sales tax was $2.52.
-/

def cost_smallest_pot (x : ℝ) : Prop :=
  let total_cost := x + (x + 0.4) + (x + 0.8) + (x + 1.2) + (x + 1.6) + (x + 2.0)
  let pre_tax_cost := total_cost / 1.075
  let pre_tax_total_cost := (9.80 / 1.075)
  (total_cost = 6 * x + 6 ∧ total_cost = pre_tax_total_cost) →
  (x + 2.0 = 2.52)

theorem cost_of_largest_pot_is_2_52 :
  ∃ x : ℝ, cost_smallest_pot x :=
sorry

end NUMINAMATH_GPT_cost_of_largest_pot_is_2_52_l218_21812


namespace NUMINAMATH_GPT_quotient_of_integers_l218_21802

theorem quotient_of_integers
  (a b : ℤ)
  (h : 1996 * a + b / 96 = a + b) :
  b / a = 2016 ∨ a / b = 2016 := 
sorry

end NUMINAMATH_GPT_quotient_of_integers_l218_21802


namespace NUMINAMATH_GPT_double_rooms_percentage_l218_21869

theorem double_rooms_percentage (S : ℝ) (h1 : 0 < S)
  (h2 : ∃ Sd : ℝ, Sd = 0.75 * S)
  (h3 : ∃ Ss : ℝ, Ss = 0.25 * S):
  (0.375 * S) / (0.625 * S) * 100 = 60 := 
by 
  sorry

end NUMINAMATH_GPT_double_rooms_percentage_l218_21869


namespace NUMINAMATH_GPT_problem1_problem2_l218_21878

-- For problem 1: Prove the quotient is 5.
def f (n : ℕ) : ℕ := 
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a + b + c + a * b + b * c + c * a + a * b * c

theorem problem1 : (625 / f 625) = 5 :=
by
  sorry

-- For problem 2: Prove the set of numbers.
def three_digit_numbers_satisfying_quotient : Finset ℕ :=
  {199, 299, 399, 499, 599, 699, 799, 899, 999}

theorem problem2 (n : ℕ) : (100 ≤ n ∧ n < 1000) ∧ n / f n = 1 ↔ n ∈ three_digit_numbers_satisfying_quotient :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l218_21878


namespace NUMINAMATH_GPT_transformed_inequality_solution_l218_21811

variable {a b c d : ℝ}

theorem transformed_inequality_solution (H : ∀ x : ℝ, ((-1 < x ∧ x < -1/3) ∨ (1/2 < x ∧ x < 1)) → 
  (b / (x + a) + (x + d) / (x + c) < 0)) :
  ∀ x : ℝ, ((1 < x ∧ x < 3) ∨ (-2 < x ∧ x < -1)) ↔ (bx / (ax - 1) + (dx - 1) / (cx - 1) < 0) :=
sorry

end NUMINAMATH_GPT_transformed_inequality_solution_l218_21811


namespace NUMINAMATH_GPT_find_number_l218_21847

-- Define the conditions
variables (x : ℝ)
axiom condition : (4/3) * x = 45

-- Prove the main statement
theorem find_number : x = 135 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l218_21847


namespace NUMINAMATH_GPT_smallest_base_b_l218_21803

theorem smallest_base_b (b : ℕ) (n : ℕ) : b > 3 ∧ 3 * b + 4 = n ^ 2 → b = 4 := 
by
  sorry

end NUMINAMATH_GPT_smallest_base_b_l218_21803


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_for_geometric_sequence_l218_21868

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {c : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n+1) = r * a_n n

theorem necessary_and_sufficient_condition_for_geometric_sequence :
  (∀ n : ℕ, S_n n = 2^n + c) →
  (∀ n : ℕ, a_n n = S_n n - S_n (n-1)) →
  is_geometric_sequence a_n ↔ c = -1 :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_for_geometric_sequence_l218_21868


namespace NUMINAMATH_GPT_transport_cost_l218_21874

theorem transport_cost (cost_per_kg : ℝ) (weight_g : ℝ) : 
  (cost_per_kg = 30000) → (weight_g = 400) → 
  ((weight_g / 1000) * cost_per_kg = 12000) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_transport_cost_l218_21874


namespace NUMINAMATH_GPT_box_volume_possible_l218_21859

theorem box_volume_possible (x : ℕ) (V : ℕ) (H1 : V = 40 * x^3) (H2 : (2 * x) * (4 * x) * (5 * x) = V) : 
  V = 320 :=
by 
  have x_possible_values := x
  -- checking if V = 320 and x = 2 satisfies the given conditions
  sorry

end NUMINAMATH_GPT_box_volume_possible_l218_21859


namespace NUMINAMATH_GPT_focal_chord_length_perpendicular_l218_21852

theorem focal_chord_length_perpendicular (x1 y1 x2 y2 : ℝ)
  (h_parabola : y1^2 = 4 * x1 ∧ y2^2 = 4 * x2)
  (h_perpendicular : x1 = x2) :
  abs (y1 - y2) = 4 :=
by sorry

end NUMINAMATH_GPT_focal_chord_length_perpendicular_l218_21852


namespace NUMINAMATH_GPT_johns_shirt_percentage_increase_l218_21839

variable (P S : ℕ)

theorem johns_shirt_percentage_increase :
  P = 50 →
  S + P = 130 →
  ((S - P) * 100 / P) = 60 := by
  sorry

end NUMINAMATH_GPT_johns_shirt_percentage_increase_l218_21839


namespace NUMINAMATH_GPT_sand_art_l218_21873

theorem sand_art (len_blue_rect : ℕ) (area_blue_rect : ℕ) (side_red_square : ℕ) (sand_per_sq_inch : ℕ) (h1 : len_blue_rect = 7) (h2 : area_blue_rect = 42) (h3 : side_red_square = 5) (h4 : sand_per_sq_inch = 3) :
  (area_blue_rect * sand_per_sq_inch) + (side_red_square * side_red_square * sand_per_sq_inch) = 201 :=
by
  sorry

end NUMINAMATH_GPT_sand_art_l218_21873


namespace NUMINAMATH_GPT_A_and_B_finish_work_together_in_12_days_l218_21882

theorem A_and_B_finish_work_together_in_12_days 
  (T_B : ℕ) 
  (T_A : ℕ)
  (h1 : T_B = 18) 
  (h2 : T_A = 2 * T_B) : 
  1 / (1 / T_A + 1 / T_B) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_A_and_B_finish_work_together_in_12_days_l218_21882


namespace NUMINAMATH_GPT_calculate_weight_l218_21844

theorem calculate_weight (W : ℝ) (h : 0.75 * W + 2 = 62) : W = 80 :=
by
  sorry

end NUMINAMATH_GPT_calculate_weight_l218_21844


namespace NUMINAMATH_GPT_B_pow_101_eq_B_pow_5_l218_21897

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 0]]

theorem B_pow_101_eq_B_pow_5 : B^101 = B := 
by sorry

end NUMINAMATH_GPT_B_pow_101_eq_B_pow_5_l218_21897


namespace NUMINAMATH_GPT_Cid_charges_5_for_car_wash_l218_21826

theorem Cid_charges_5_for_car_wash (x : ℝ) :
  5 * 20 + 10 * 30 + 15 * x = 475 → x = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Cid_charges_5_for_car_wash_l218_21826


namespace NUMINAMATH_GPT_simplify_expression_l218_21877

theorem simplify_expression (x y : ℝ) (h : x ≠ y) : (x^2 - x * y) / (x - y)^2 = x / (x - y) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l218_21877


namespace NUMINAMATH_GPT_triangle_area_given_conditions_l218_21863

theorem triangle_area_given_conditions (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6) (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_given_conditions_l218_21863


namespace NUMINAMATH_GPT_fraction_zero_l218_21823

theorem fraction_zero (x : ℝ) (h : x ≠ 3) : (2 * x^2 - 6 * x) / (x - 3) = 0 ↔ x = 0 := 
by
  sorry

end NUMINAMATH_GPT_fraction_zero_l218_21823


namespace NUMINAMATH_GPT_repair_cost_total_l218_21867

-- Define the inputs
def labor_cost_rate : ℤ := 75
def labor_hours : ℤ := 16
def part_cost : ℤ := 1200

-- Define the required computation and proof statement
def total_repair_cost : ℤ :=
  let labor_cost := labor_cost_rate * labor_hours
  labor_cost + part_cost

theorem repair_cost_total : total_repair_cost = 2400 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_repair_cost_total_l218_21867


namespace NUMINAMATH_GPT_perfect_apples_count_l218_21842

-- Definitions (conditions)
def total_apples := 30
def too_small_fraction := (1 : ℚ) / 6
def not_ripe_fraction := (1 : ℚ) / 3
def too_small_apples := (too_small_fraction * total_apples : ℚ)
def not_ripe_apples := (not_ripe_fraction * total_apples : ℚ)

-- Statement of the theorem (proof problem)
theorem perfect_apples_count : total_apples - too_small_apples - not_ripe_apples = 15 := by
  sorry

end NUMINAMATH_GPT_perfect_apples_count_l218_21842


namespace NUMINAMATH_GPT_no_valid_x_for_given_circle_conditions_l218_21841

theorem no_valid_x_for_given_circle_conditions :
  ∀ x : ℝ,
    ¬ ((x - 15)^2 + 18^2 = 225 ∧ (x - 15)^2 + (-18)^2 = 225) :=
by
  sorry

end NUMINAMATH_GPT_no_valid_x_for_given_circle_conditions_l218_21841


namespace NUMINAMATH_GPT_count_valid_m_l218_21837

theorem count_valid_m (h : 1260 > 0) :
  ∃! (n : ℕ), n = 3 := by
  sorry

end NUMINAMATH_GPT_count_valid_m_l218_21837


namespace NUMINAMATH_GPT_cost_of_each_scoop_l218_21817

theorem cost_of_each_scoop (x : ℝ) 
  (pierre_scoops : ℝ := 3)
  (mom_scoops : ℝ := 4)
  (total_bill : ℝ := 14) 
  (h : 7 * x = total_bill) :
  x = 2 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_each_scoop_l218_21817


namespace NUMINAMATH_GPT_seeking_the_cause_from_the_result_means_sufficient_condition_l218_21885

-- Define the necessary entities for the conditions
inductive Condition
| Necessary
| Sufficient
| NecessaryAndSufficient
| NecessaryOrSufficient

-- Define the statement of the proof problem
theorem seeking_the_cause_from_the_result_means_sufficient_condition :
  (seeking_the_cause_from_the_result : Condition) = Condition.Sufficient :=
sorry

end NUMINAMATH_GPT_seeking_the_cause_from_the_result_means_sufficient_condition_l218_21885


namespace NUMINAMATH_GPT_red_light_cherries_cost_price_min_value_m_profit_l218_21824

-- Define the constants and cost conditions
def cost_price_red_light_cherries (x : ℝ) (y : ℝ) : Prop :=
  (6000 / (2 * x) - 100 = 1000 / x)

-- Define sales conditions and profit requirement
def min_value_m (m : ℝ) (profit : ℝ) : Prop :=
  (20 * 3 * m + 20 * (20 - 0.5 * m) + (28 - 20) * (50 - 3 * m - 20) >= profit)

-- Define the main proof goal statements
theorem red_light_cherries_cost_price :
  ∃ x, cost_price_red_light_cherries x 6000 ∧ 20 = x :=
sorry

theorem min_value_m_profit :
  ∃ m, min_value_m m 770 ∧ m >= 5 :=
sorry

end NUMINAMATH_GPT_red_light_cherries_cost_price_min_value_m_profit_l218_21824


namespace NUMINAMATH_GPT_pam_bags_equiv_gerald_bags_l218_21896

theorem pam_bags_equiv_gerald_bags :
  ∀ (total_apples pam_bags apples_per_gerald_bag : ℕ), 
    total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 → 
    (total_apples / pam_bags) / apples_per_gerald_bag = 3 :=
by
  intros total_apples pam_bags apples_per_gerald_bag h
  obtain ⟨ht, hp, hg⟩ : total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 := h
  sorry

end NUMINAMATH_GPT_pam_bags_equiv_gerald_bags_l218_21896


namespace NUMINAMATH_GPT_math_problem_l218_21881

theorem math_problem 
  (a1 : (10^4 + 500) = 100500)
  (a2 : (25^4 + 500) = 390625500)
  (a3 : (40^4 + 500) = 256000500)
  (a4 : (55^4 + 500) = 915062500)
  (a5 : (70^4 + 500) = 24010062500)
  (b1 : (5^4 + 500) = 625+500)
  (b2 : (20^4 + 500) = 160000500)
  (b3 : (35^4 + 500) = 150062500)
  (b4 : (50^4 + 500) = 625000500)
  (b5 : (65^4 + 500) = 1785062500) :
  ( (100500 * 390625500 * 256000500 * 915062500 * 24010062500) / (625+500 * 160000500 * 150062500 * 625000500 * 1785062500) = 240) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l218_21881


namespace NUMINAMATH_GPT_girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l218_21849

namespace PhotoArrangement

/-- There are 4 boys and 3 girls. -/
def boys : ℕ := 4
def girls : ℕ := 3

/-- Number of ways to arrange given conditions -/
def arrangementsWithGirlsAtEnds : ℕ := 720
def arrangementsWithNoGirlsNextToEachOther : ℕ := 1440
def arrangementsWithGirlAtoRightOfGirlB : ℕ := 2520

-- Problem 1: If there are girls at both ends
theorem girls_at_ends (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlsAtEnds := by
  sorry

-- Problem 2: If no two girls are standing next to each other
theorem no_girls_next_to_each_other (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithNoGirlsNextToEachOther := by
  sorry

-- Problem 3: If girl A must be to the right of girl B
theorem girl_A_right_of_girl_B (b g : ℕ) (h_b : b = boys) (h_g : g = girls) :
  ∃ n, n = arrangementsWithGirlAtoRightOfGirlB := by
  sorry

end PhotoArrangement

end NUMINAMATH_GPT_girls_at_ends_no_girls_next_to_each_other_girl_A_right_of_girl_B_l218_21849


namespace NUMINAMATH_GPT_percentage_of_female_officers_on_duty_l218_21894

-- Declare the conditions
def total_officers_on_duty : ℕ := 100
def female_officers_on_duty : ℕ := 50
def total_female_officers : ℕ := 250

-- The theorem to prove
theorem percentage_of_female_officers_on_duty :
  (female_officers_on_duty / total_female_officers) * 100 = 20 := 
sorry

end NUMINAMATH_GPT_percentage_of_female_officers_on_duty_l218_21894


namespace NUMINAMATH_GPT_part1_l218_21879

theorem part1 (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^2005 + z^2006 + z^2008 + z^2009 = -2 :=
  sorry

end NUMINAMATH_GPT_part1_l218_21879


namespace NUMINAMATH_GPT_union_of_sets_l218_21807

def A : Set Int := {-1, 2, 3, 5}
def B : Set Int := {2, 4, 5}

theorem union_of_sets :
  A ∪ B = {-1, 2, 3, 4, 5} := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l218_21807


namespace NUMINAMATH_GPT_sequence_term_101_l218_21816

theorem sequence_term_101 :
  ∃ a : ℕ → ℚ, a 1 = 2 ∧ (∀ n : ℕ, 2 * a (n+1) - 2 * a n = 1) ∧ a 101 = 52 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_101_l218_21816


namespace NUMINAMATH_GPT_unique_solution_of_equation_l218_21854

theorem unique_solution_of_equation :
  ∃! (x : Fin 8 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + 
                                  (x 2 - x 3)^2 + (x 3 - x 4)^2 + 
                                  (x 4 - x 5)^2 + (x 5 - x 6)^2 + 
                                  (x 6 - x 7)^2 + (x 7)^2 = 1 / 9 :=
sorry

end NUMINAMATH_GPT_unique_solution_of_equation_l218_21854


namespace NUMINAMATH_GPT_union_of_A_B_l218_21814

def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }

def B : Set ℝ := { x | 1 < x ∧ x ≤ 3 }

theorem union_of_A_B : A ∪ B = { x | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_B_l218_21814


namespace NUMINAMATH_GPT_minimum_value_of_expression_l218_21820

theorem minimum_value_of_expression (p q r s t u : ℝ) 
  (hpqrsu_pos : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ 0 < t ∧ 0 < u) 
  (sum_eq : p + q + r + s + t + u = 8) : 
  98 ≤ (2 / p + 4 / q + 9 / r + 16 / s + 25 / t + 36 / u) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l218_21820


namespace NUMINAMATH_GPT_trapezoid_area_l218_21825

theorem trapezoid_area 
  (diagonals_perpendicular : ∀ A B C D : ℝ, (A ≠ B → C ≠ D → A * C + B * D = 0)) 
  (diagonal_length : ∀ B D : ℝ, B ≠ D → (B - D) = 17) 
  (height_of_trapezoid : ∀ (height : ℝ), height = 15) : 
  ∃ (area : ℝ), area = 4335 / 16 := 
sorry

end NUMINAMATH_GPT_trapezoid_area_l218_21825


namespace NUMINAMATH_GPT_increase_by_percentage_l218_21866

-- Define the initial number.
def initial_number : ℝ := 75

-- Define the percentage increase as a decimal.
def percentage_increase : ℝ := 1.5

-- Define the expected final result after applying the increase.
def expected_result : ℝ := 187.5

-- The proof statement.
theorem increase_by_percentage : initial_number * (1 + percentage_increase) = expected_result :=
by
  sorry

end NUMINAMATH_GPT_increase_by_percentage_l218_21866


namespace NUMINAMATH_GPT_greene_family_admission_cost_l218_21810

theorem greene_family_admission_cost (x : ℝ) (h1 : ∀ y : ℝ, y = x - 13) (h2 : ∀ z : ℝ, z = x + (x - 13)) :
  x = 45 :=
by
  sorry

end NUMINAMATH_GPT_greene_family_admission_cost_l218_21810


namespace NUMINAMATH_GPT_a_range_l218_21809

noncomputable def f (a x : ℝ) : ℝ := x^3 + a*x^2 - 2*x + 5

noncomputable def f' (a x : ℝ) : ℝ := 3*x^2 + 2*a*x - 2

theorem a_range (a : ℝ) :
  (∃ x y : ℝ, (1/3 < x ∧ x < 1/2) ∧ (1/3 < y ∧ y < 1/2) ∧ f' a x = 0 ∧ f' a y = 0) ↔
  a ∈ Set.Ioo (5/4) (5/2) :=
by
  sorry

end NUMINAMATH_GPT_a_range_l218_21809


namespace NUMINAMATH_GPT_sin_two_alpha_sub_pi_eq_24_div_25_l218_21888

noncomputable def pi_div_2 : ℝ := Real.pi / 2

theorem sin_two_alpha_sub_pi_eq_24_div_25
  (α : ℝ) 
  (h1 : pi_div_2 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan (α + Real.pi / 4) = -1 / 7) : 
  Real.sin (2 * α - Real.pi) = 24 / 25 := 
sorry

end NUMINAMATH_GPT_sin_two_alpha_sub_pi_eq_24_div_25_l218_21888


namespace NUMINAMATH_GPT_max_sum_of_solutions_l218_21864

theorem max_sum_of_solutions (x y : ℤ) (h : 3 * x ^ 2 + 5 * y ^ 2 = 345) :
  x + y ≤ 13 :=
sorry

end NUMINAMATH_GPT_max_sum_of_solutions_l218_21864


namespace NUMINAMATH_GPT_laura_total_owed_l218_21893

-- Define the principal amounts charged each month
def january_charge : ℝ := 35
def february_charge : ℝ := 45
def march_charge : ℝ := 55
def april_charge : ℝ := 25

-- Define the respective interest rates for each month, as decimals
def january_interest_rate : ℝ := 0.05
def february_interest_rate : ℝ := 0.07
def march_interest_rate : ℝ := 0.04
def april_interest_rate : ℝ := 0.06

-- Define the interests accrued for each month's charges
def january_interest : ℝ := january_charge * january_interest_rate
def february_interest : ℝ := february_charge * february_interest_rate
def march_interest : ℝ := march_charge * march_interest_rate
def april_interest : ℝ := april_charge * april_interest_rate

-- Define the totals including original charges and their respective interests
def january_total : ℝ := january_charge + january_interest
def february_total : ℝ := february_charge + february_interest
def march_total : ℝ := march_charge + march_interest
def april_total : ℝ := april_charge + april_interest

-- Define the total amount owed a year later
def total_owed : ℝ := january_total + february_total + march_total + april_total

-- Prove that the total amount owed a year later is $168.60
theorem laura_total_owed :
  total_owed = 168.60 := by
  sorry

end NUMINAMATH_GPT_laura_total_owed_l218_21893


namespace NUMINAMATH_GPT_spring_work_compression_l218_21833

theorem spring_work_compression :
  ∀ (k : ℝ) (F : ℝ) (x : ℝ), 
  (F = 10) → (x = 1 / 100) → (k = F / x) → (W = 5) :=
by
sorry

end NUMINAMATH_GPT_spring_work_compression_l218_21833


namespace NUMINAMATH_GPT_pos_int_divides_l218_21851

theorem pos_int_divides (n : ℕ) (h₀ : 0 < n) (h₁ : (n - 1) ∣ (n^3 + 4)) : n = 2 ∨ n = 6 :=
by sorry

end NUMINAMATH_GPT_pos_int_divides_l218_21851


namespace NUMINAMATH_GPT_find_balanced_grid_pairs_l218_21890

-- Define a balanced grid condition
def is_balanced_grid (m n : ℕ) (grid : ℕ → ℕ → Prop) : Prop :=
  ∀ i j, i < m → j < n →
    (∀ k, k < m → grid i k = grid i j) ∧ (∀ l, l < n → grid l j = grid i j)

-- Main theorem statement
theorem find_balanced_grid_pairs (m n : ℕ) :
  (∃ grid, is_balanced_grid m n grid) ↔ (m = n ∨ m = n / 2 ∨ n = 2 * m) :=
by
  sorry

end NUMINAMATH_GPT_find_balanced_grid_pairs_l218_21890


namespace NUMINAMATH_GPT_negative_number_from_operations_l218_21834

theorem negative_number_from_operations :
  (∀ (a b : Int), a + b < 0 → a = -1 ∧ b = -3) ∧
  (∀ (a b : Int), a - b < 0 → a = 1 ∧ b = 4) ∧
  (∀ (a b : Int), a * b > 0 → a = 3 ∧ b = -2) ∧
  (∀ (a b : Int), a / b = 0 → a = 0 ∧ b = -7) :=
by
  sorry

end NUMINAMATH_GPT_negative_number_from_operations_l218_21834


namespace NUMINAMATH_GPT_angle_sum_triangle_l218_21848

theorem angle_sum_triangle (A B C : ℝ) (hA : A = 75) (hB : B = 40) (h_sum : A + B + C = 180) : C = 65 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_triangle_l218_21848


namespace NUMINAMATH_GPT_ellie_total_distance_after_six_steps_l218_21876

-- Define the initial conditions and parameters
def initial_position : ℚ := 0
def target_distance : ℚ := 5
def step_fraction : ℚ := 1 / 4
def steps : ℕ := 6

-- Define the function that calculates the sum of the distances walked
def distance_walked (n : ℕ) : ℚ :=
  let first_term := target_distance * step_fraction
  let common_ratio := 3 / 4
  first_term * (1 - common_ratio^n) / (1 - common_ratio)

-- Define the theorem we want to prove
theorem ellie_total_distance_after_six_steps :
  distance_walked steps = 16835 / 4096 :=
by 
  sorry

end NUMINAMATH_GPT_ellie_total_distance_after_six_steps_l218_21876


namespace NUMINAMATH_GPT_proof_value_g_expression_l218_21805

noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom g_invertible : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x
axiom g_table : ∀ x, (x = 1 → g x = 4) ∧ (x = 2 → g x = 5) ∧ (x = 3 → g x = 7) ∧ (x = 4 → g x = 9) ∧ (x = 5 → g x = 10)

theorem proof_value_g_expression :
  g (g 2) + g (g_inv 9) + g_inv (g_inv 7) = 21 :=
by
  sorry

end NUMINAMATH_GPT_proof_value_g_expression_l218_21805


namespace NUMINAMATH_GPT_hotel_cost_l218_21845

theorem hotel_cost (x y : ℕ) (h1 : 3 * x + 6 * y = 1020) (h2 : x + 5 * y = 700) :
  5 * (x + y) = 1100 :=
sorry

end NUMINAMATH_GPT_hotel_cost_l218_21845


namespace NUMINAMATH_GPT_beads_problem_l218_21861

noncomputable def number_of_blue_beads (total_beads : ℕ) (beads_with_blue_neighbor : ℕ) (beads_with_green_neighbor : ℕ) : ℕ :=
  let beads_with_both_neighbors := beads_with_blue_neighbor + beads_with_green_neighbor - total_beads
  let beads_with_only_blue_neighbor := beads_with_blue_neighbor - beads_with_both_neighbors
  (2 * beads_with_only_blue_neighbor + beads_with_both_neighbors) / 2

theorem beads_problem : number_of_blue_beads 30 26 20 = 18 := by 
  -- ...
  sorry

end NUMINAMATH_GPT_beads_problem_l218_21861


namespace NUMINAMATH_GPT_c_impossible_value_l218_21846

theorem c_impossible_value (a b c : ℤ) (h : (∀ x : ℤ, (x + a) * (x + b) = x^2 + c * x - 8)) : c ≠ 4 :=
by
  sorry

end NUMINAMATH_GPT_c_impossible_value_l218_21846


namespace NUMINAMATH_GPT_inequality_solution_l218_21898

-- Declare the constants m and n
variables (m n : ℝ)

-- State the conditions
def condition1 (x : ℝ) := m < 0
def condition2 := n = -m / 2

-- State the theorem
theorem inequality_solution (x : ℝ) (h1 : condition1 m n) (h2 : condition2 m n) : 
  nx - m < 0 ↔ x < -2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l218_21898


namespace NUMINAMATH_GPT_set_intersection_l218_21858

def M := {x : ℝ | x^2 > 4}
def N := {x : ℝ | 1 < x ∧ x ≤ 3}
def complement_M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def intersection := N ∩ complement_M

theorem set_intersection : intersection = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_set_intersection_l218_21858


namespace NUMINAMATH_GPT_divide_by_repeating_decimal_l218_21892

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end NUMINAMATH_GPT_divide_by_repeating_decimal_l218_21892


namespace NUMINAMATH_GPT_component_unqualified_l218_21850

/-- 
    The specified diameter range for a component is within [19.98, 20.02].
    The measured diameter of the component is 19.9.
    Prove that the component is unqualified.
-/
def is_unqualified (diameter_measured : ℝ) : Prop :=
    diameter_measured < 19.98 ∨ diameter_measured > 20.02

theorem component_unqualified : is_unqualified 19.9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_component_unqualified_l218_21850


namespace NUMINAMATH_GPT_total_weight_proof_l218_21835
-- Import the entire math library

-- Assume the conditions as given variables
variables (w r s : ℕ)
-- Assign values to the given conditions
def weight_per_rep := 15
def reps_per_set := 10
def number_of_sets := 3

-- Calculate total weight moved
def total_weight_moved := w * r * s

-- The theorem to prove the total weight moved
theorem total_weight_proof : total_weight_moved weight_per_rep reps_per_set number_of_sets = 450 :=
by
  -- Provide the expected result directly, proving the statement
  sorry

end NUMINAMATH_GPT_total_weight_proof_l218_21835


namespace NUMINAMATH_GPT_problem_statement_l218_21828

noncomputable def C : ℝ := 49
noncomputable def D : ℝ := 3.75

theorem problem_statement : C + D = 52.75 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l218_21828


namespace NUMINAMATH_GPT_martha_initial_apples_l218_21883

theorem martha_initial_apples :
  ∀ (jane_apples james_apples keep_apples more_to_give initial_apples : ℕ),
    jane_apples = 5 →
    james_apples = jane_apples + 2 →
    keep_apples = 4 →
    more_to_give = 4 →
    initial_apples = jane_apples + james_apples + keep_apples + more_to_give →
    initial_apples = 20 :=
by
  intros jane_apples james_apples keep_apples more_to_give initial_apples
  intro h_jane
  intro h_james
  intro h_keep
  intro h_more
  intro h_initial
  exact sorry

end NUMINAMATH_GPT_martha_initial_apples_l218_21883


namespace NUMINAMATH_GPT_cubic_function_decreasing_l218_21871

theorem cubic_function_decreasing (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → (a ≤ 0) := 
by 
  sorry

end NUMINAMATH_GPT_cubic_function_decreasing_l218_21871


namespace NUMINAMATH_GPT_smallest_n_satisfying_equation_l218_21831

theorem smallest_n_satisfying_equation : ∃ (k : ℤ), (∃ (n : ℤ), n > 0 ∧ n % 2 = 1 ∧ (n ^ 3 + 2 * n ^ 2 = k ^ 2) ∧ ∀ m : ℤ, (m > 0 ∧ m < n ∧ m % 2 = 1) → ¬ (∃ j : ℤ, m ^ 3 + 2 * m ^ 2 = j ^ 2)) ∧ k % 2 = 1 :=
sorry

end NUMINAMATH_GPT_smallest_n_satisfying_equation_l218_21831


namespace NUMINAMATH_GPT_quadratic_equation_must_be_minus_2_l218_21801

-- Define the main problem statement
theorem quadratic_equation_must_be_minus_2 (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x ^ |m| - 3 * x - 7 = 0) →
  (∀ (h : |m| = 2), m - 2 ≠ 0) →
  m = -2 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_must_be_minus_2_l218_21801


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l218_21830

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x + 1) * (x - 3) < 0 → x > -1 ∧ ((x > -1) → (x + 1) * (x - 3) < 0) = false :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l218_21830


namespace NUMINAMATH_GPT_articleWords_l218_21827

-- Define the number of words per page for larger and smaller types
def wordsLargerType : Nat := 1800
def wordsSmallerType : Nat := 2400

-- Define the total number of pages and the number of pages in smaller type
def totalPages : Nat := 21
def smallerTypePages : Nat := 17

-- The number of pages in larger type
def largerTypePages : Nat := totalPages - smallerTypePages

-- Calculate the total number of words in the article
def totalWords : Nat := (largerTypePages * wordsLargerType) + (smallerTypePages * wordsSmallerType)

-- Prove that the total number of words in the article is 48,000
theorem articleWords : totalWords = 48000 := 
by
  sorry

end NUMINAMATH_GPT_articleWords_l218_21827


namespace NUMINAMATH_GPT_amount_c_l218_21806

theorem amount_c (a b c d : ℝ) :
  a + c = 350 →
  b + d = 450 →
  a + d = 400 →
  c + d = 500 →
  a + b + c + d = 750 →
  c = 225 :=
by 
  intros h1 h2 h3 h4 h5
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_amount_c_l218_21806


namespace NUMINAMATH_GPT_find_constants_l218_21886

theorem find_constants :
  ∃ A B C D : ℚ,
    (∀ x : ℚ,
      x ≠ 2 → x ≠ 3 → x ≠ 5 → x ≠ -1 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 5) * (x + 1)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x + 1)) ∧
  A = -5/9 ∧ B = 0 ∧ C = 4/9 ∧ D = -1/9 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l218_21886


namespace NUMINAMATH_GPT_simplification_correct_l218_21815

noncomputable def given_equation (x : ℚ) : Prop := 
  x / (2 * x - 1) - 3 = 2 / (1 - 2 * x)

theorem simplification_correct (x : ℚ) (h : given_equation x) : 
  x - 3 * (2 * x - 1) = -2 :=
sorry

end NUMINAMATH_GPT_simplification_correct_l218_21815


namespace NUMINAMATH_GPT_correct_option_is_A_l218_21832

variable (a b : ℤ)

-- Option A condition
def optionA : Prop := 3 * a^2 * b / b = 3 * a^2

-- Option B condition
def optionB : Prop := a^12 / a^3 = a^4

-- Option C condition
def optionC : Prop := (a + b)^2 = a^2 + b^2

-- Option D condition
def optionD : Prop := (-2 * a^2)^3 = 8 * a^6

theorem correct_option_is_A : 
  optionA a b ∧ ¬optionB a ∧ ¬optionC a b ∧ ¬optionD a :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_A_l218_21832


namespace NUMINAMATH_GPT_p_at_zero_l218_21819

-- Define the quartic monic polynomial
noncomputable def p (x : ℝ) : ℝ := sorry

-- Conditions
axiom p_monic : true -- p is a monic polynomial, we represent it by an axiom here for simplicity
axiom p_neg2 : p (-2) = -4
axiom p_1 : p (1) = -1
axiom p_3 : p (3) = -9
axiom p_5 : p (5) = -25

-- The theorem to be proven
theorem p_at_zero : p 0 = -30 := by
  sorry

end NUMINAMATH_GPT_p_at_zero_l218_21819


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_eqn_l218_21857

theorem sum_of_roots_of_quadratic_eqn (A B : ℝ) 
  (h₁ : 3 * A ^ 2 - 9 * A + 6 = 0)
  (h₂ : 3 * B ^ 2 - 9 * B + 6 = 0)
  (h_distinct : A ≠ B):
  A + B = 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_eqn_l218_21857


namespace NUMINAMATH_GPT_average_alligators_l218_21853

theorem average_alligators (t s n : ℕ) (h1 : t = 50) (h2 : s = 20) (h3 : n = 3) :
  (t - s) / n = 10 :=
by 
  sorry

end NUMINAMATH_GPT_average_alligators_l218_21853


namespace NUMINAMATH_GPT_prime_power_divides_binomial_l218_21818

theorem prime_power_divides_binomial {p n k α : ℕ} (hp : Nat.Prime p) 
  (h : p^α ∣ Nat.choose n k) : p^α ≤ n := 
sorry

end NUMINAMATH_GPT_prime_power_divides_binomial_l218_21818


namespace NUMINAMATH_GPT_r_values_if_polynomial_divisible_l218_21838

noncomputable
def find_r_iff_divisible (r : ℝ) : Prop :=
  (10 * (r^2 * (1 - 2*r))) = -6 ∧ 
  (2 * r + (1 - 2*r)) = 1 ∧ 
  (r^2 + 2 * r * (1 - 2*r)) = -5.2

theorem r_values_if_polynomial_divisible (r : ℝ) :
  (find_r_iff_divisible r) ↔ 
  (r = (2 + Real.sqrt 30) / 5 ∨ r = (2 - Real.sqrt 30) / 5) := 
by
  sorry

end NUMINAMATH_GPT_r_values_if_polynomial_divisible_l218_21838


namespace NUMINAMATH_GPT_number_of_boxes_sold_on_saturday_l218_21870

theorem number_of_boxes_sold_on_saturday (S : ℝ) 
  (h : S + 1.5 * S + 1.95 * S + 2.34 * S + 2.574 * S = 720) : 
  S = 77 := 
sorry

end NUMINAMATH_GPT_number_of_boxes_sold_on_saturday_l218_21870


namespace NUMINAMATH_GPT_area_ratio_PQR_to_STU_l218_21884

-- Given Conditions
def triangle_PQR_sides (a b c : Nat) : Prop :=
  a = 9 ∧ b = 40 ∧ c = 41

def triangle_STU_sides (x y z : Nat) : Prop :=
  x = 7 ∧ y = 24 ∧ z = 25

-- Theorem Statement (math proof problem)
theorem area_ratio_PQR_to_STU :
  (∃ (a b c x y z : Nat), triangle_PQR_sides a b c ∧ triangle_STU_sides x y z) →
  9 * 40 / (7 * 24) = 15 / 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_area_ratio_PQR_to_STU_l218_21884


namespace NUMINAMATH_GPT_cindy_gave_lisa_marbles_l218_21899

-- Definitions for the given conditions
def cindy_initial_marbles : ℕ := 20
def lisa_initial_marbles := cindy_initial_marbles - 5
def lisa_final_marbles := lisa_initial_marbles + 19

-- Theorem we need to prove
theorem cindy_gave_lisa_marbles :
  ∃ n : ℕ, lisa_final_marbles = lisa_initial_marbles + n ∧ n = 19 :=
by
  sorry

end NUMINAMATH_GPT_cindy_gave_lisa_marbles_l218_21899


namespace NUMINAMATH_GPT_algebraic_expression_independence_l218_21895

theorem algebraic_expression_independence (a b : ℝ) (h : ∀ x : ℝ, (x^2 + a*x - (b*x^2 - x - 3)) = 3) : a - b = -2 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_independence_l218_21895
