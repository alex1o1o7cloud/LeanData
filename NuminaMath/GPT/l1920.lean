import Mathlib

namespace NUMINAMATH_GPT_milk_production_l1920_192007

theorem milk_production (a b c d e f : ℕ) (h₁ : a > 0) (h₂ : c > 0) (h₃ : f > 0) : 
  ((d * e * b * f) / (100 * a * c)) = (d * e * b * f / (100 * a * c)) :=
by
  sorry

end NUMINAMATH_GPT_milk_production_l1920_192007


namespace NUMINAMATH_GPT_men_in_hotel_l1920_192034

theorem men_in_hotel (n : ℕ) (A : ℝ) (h1 : 8 * 3 = 24)
  (h2 : A = 32.625 / n)
  (h3 : 24 + (A + 5) = 32.625) :
  n = 9 := 
  by
  sorry

end NUMINAMATH_GPT_men_in_hotel_l1920_192034


namespace NUMINAMATH_GPT_maximum_possible_angle_Z_l1920_192014

theorem maximum_possible_angle_Z (X Y Z : ℝ) (h1 : Z ≤ Y) (h2 : Y ≤ X) (h3 : 2 * X = 6 * Z) (h4 : X + Y + Z = 180) : Z = 36 :=
by
  sorry

end NUMINAMATH_GPT_maximum_possible_angle_Z_l1920_192014


namespace NUMINAMATH_GPT_Mika_water_left_l1920_192062

theorem Mika_water_left :
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  initial_amount - used_amount = 5 / 4 :=
by
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  show initial_amount - used_amount = 5 / 4
  sorry

end NUMINAMATH_GPT_Mika_water_left_l1920_192062


namespace NUMINAMATH_GPT_part1_part2_l1920_192008

-- Part 1
noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1/3) * a * x ^ 3 - (1/2) * x ^ 2

noncomputable def f' (x a : ℝ) : ℝ := x * Real.exp x - a * x ^ 2 - x

noncomputable def g (x a : ℝ) : ℝ := f' x a / x

theorem part1 (a : ℝ) (h : a > 0) : g a a > 0 := by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x, f' x a = 0) : a > 0 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1920_192008


namespace NUMINAMATH_GPT_taxes_taken_out_l1920_192031

theorem taxes_taken_out
  (gross_pay : ℕ)
  (retirement_percentage : ℝ)
  (net_pay_after_taxes : ℕ)
  (tax_amount : ℕ) :
  gross_pay = 1120 →
  retirement_percentage = 0.25 →
  net_pay_after_taxes = 740 →
  tax_amount = gross_pay - (gross_pay * retirement_percentage) - net_pay_after_taxes :=
by
  sorry

end NUMINAMATH_GPT_taxes_taken_out_l1920_192031


namespace NUMINAMATH_GPT_head_start_l1920_192095

theorem head_start (V_b : ℝ) (S : ℝ) : 
  ((7 / 4) * V_b) = V_b → 
  196 = (196 - S) → 
  S = 84 := 
sorry

end NUMINAMATH_GPT_head_start_l1920_192095


namespace NUMINAMATH_GPT_square_side_length_l1920_192076

theorem square_side_length (length_rect width_rect : ℕ) (h_length : length_rect = 400) (h_width : width_rect = 300)
  (h_perimeter : 4 * side_length = 2 * (2 * (length_rect + width_rect))) : side_length = 700 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_square_side_length_l1920_192076


namespace NUMINAMATH_GPT_bushes_needed_l1920_192005

theorem bushes_needed
  (num_sides : ℕ) (side_length : ℝ) (bush_fill : ℝ) (total_length : ℝ) (num_bushes : ℕ) :
  num_sides = 3 ∧ side_length = 16 ∧ bush_fill = 4 ∧ total_length = num_sides * side_length ∧ num_bushes = total_length / bush_fill →
  num_bushes = 12 := by
  sorry

end NUMINAMATH_GPT_bushes_needed_l1920_192005


namespace NUMINAMATH_GPT_final_price_after_discounts_l1920_192018

theorem final_price_after_discounts (m : ℝ) : (0.8 * m - 10) = selling_price :=
by
  sorry

end NUMINAMATH_GPT_final_price_after_discounts_l1920_192018


namespace NUMINAMATH_GPT_slope_of_line_l1920_192047

theorem slope_of_line (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 / x + 4 / y = 0) : 
  ∃ m : ℝ, m = -4 / 3 := 
sorry

end NUMINAMATH_GPT_slope_of_line_l1920_192047


namespace NUMINAMATH_GPT_range_g_l1920_192082

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + 2 * Real.arcsin x

theorem range_g : 
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -Real.pi / 2 ≤ g x ∧ g x ≤ 3 * Real.pi / 2) := 
by {
  sorry
}

end NUMINAMATH_GPT_range_g_l1920_192082


namespace NUMINAMATH_GPT_area_increase_percentage_l1920_192067

variable (r : ℝ) (π : ℝ := Real.pi)

theorem area_increase_percentage (h₁ : r > 0) (h₂ : π > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  (new_area - original_area) / original_area * 100 = 525 := 
by
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  sorry

end NUMINAMATH_GPT_area_increase_percentage_l1920_192067


namespace NUMINAMATH_GPT_fred_earned_63_dollars_l1920_192043

-- Definitions for the conditions
def initial_money_fred : ℕ := 23
def initial_money_jason : ℕ := 46
def money_per_car : ℕ := 5
def money_per_lawn : ℕ := 10
def money_per_dog : ℕ := 3
def total_money_after_chores : ℕ := 86
def cars_washed : ℕ := 4
def lawns_mowed : ℕ := 3
def dogs_walked : ℕ := 7

-- The equivalent proof problem in Lean
theorem fred_earned_63_dollars :
  (initial_money_fred + (cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = total_money_after_chores) → 
  ((cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = 63) :=
by
  sorry

end NUMINAMATH_GPT_fred_earned_63_dollars_l1920_192043


namespace NUMINAMATH_GPT_distance_travelled_first_hour_l1920_192044

noncomputable def initial_distance (x : ℕ) : Prop :=
  let distance_travelled := (12 / 2) * (2 * x + (12 - 1) * 2)
  distance_travelled = 552

theorem distance_travelled_first_hour : ∃ x : ℕ, initial_distance x ∧ x = 35 :=
by
  use 35
  unfold initial_distance
  sorry

end NUMINAMATH_GPT_distance_travelled_first_hour_l1920_192044


namespace NUMINAMATH_GPT_part_a_part_b_l1920_192093

theorem part_a {a b c : ℝ} : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 :=
sorry

theorem part_b {a b c : ℝ} : (a + b + c) ^ 2 ≥ 3 * (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1920_192093


namespace NUMINAMATH_GPT_fraction_of_left_handed_non_throwers_l1920_192015

theorem fraction_of_left_handed_non_throwers 
  (total_players : ℕ) (throwers : ℕ) (right_handed_players : ℕ) (all_throwers_right_handed : throwers ≤ right_handed_players) 
  (total_players_eq : total_players = 70) 
  (throwers_eq : throwers = 46) 
  (right_handed_players_eq : right_handed_players = 62) 
  : (total_players - throwers) = 24 → ((right_handed_players - throwers) = 16 → (24 - 16) = 8 → ((8 : ℚ) / 24 = 1/3)) := 
by 
  intros;
  sorry

end NUMINAMATH_GPT_fraction_of_left_handed_non_throwers_l1920_192015


namespace NUMINAMATH_GPT_situps_together_l1920_192072

theorem situps_together (hani_rate diana_rate : ℕ) (diana_situps diana_time hani_situps total_situps : ℕ)
  (h1 : hani_rate = diana_rate + 3)
  (h2 : diana_rate = 4)
  (h3 : diana_situps = 40)
  (h4 : diana_time = diana_situps / diana_rate)
  (h5 : hani_situps = hani_rate * diana_time)
  (h6 : total_situps = diana_situps + hani_situps) : 
  total_situps = 110 :=
sorry

end NUMINAMATH_GPT_situps_together_l1920_192072


namespace NUMINAMATH_GPT_exponent_division_is_equal_l1920_192042

variable (a : ℝ) 

theorem exponent_division_is_equal :
  (a^11) / (a^2) = a^9 := 
sorry

end NUMINAMATH_GPT_exponent_division_is_equal_l1920_192042


namespace NUMINAMATH_GPT_evaluate_expression_l1920_192000

theorem evaluate_expression :
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  (sum1 / sum2 - sum2 / sum1) = 11 / 30 :=
by
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1920_192000


namespace NUMINAMATH_GPT_event_day_price_l1920_192094

theorem event_day_price (original_price : ℝ) (first_discount second_discount : ℝ)
  (h1 : original_price = 250) (h2 : first_discount = 0.4) (h3 : second_discount = 0.25) : 
  ∃ discounted_price : ℝ, 
  discounted_price = (original_price * (1 - first_discount)) * (1 - second_discount) → 
  discounted_price = 112.5 :=
by
  use (250 * (1 - 0.4) * (1 - 0.25))
  sorry

end NUMINAMATH_GPT_event_day_price_l1920_192094


namespace NUMINAMATH_GPT_speed_of_boat_l1920_192050

-- Given conditions
variables (V_b : ℝ) (V_s : ℝ) (T : ℝ) (D : ℝ)

-- Problem statement in Lean
theorem speed_of_boat (h1 : V_s = 5) (h2 : T = 1) (h3 : D = 45) :
  D = T * (V_b + V_s) → V_b = 40 := 
by
  intro h4
  rw [h1, h2, h3] at h4
  linarith

end NUMINAMATH_GPT_speed_of_boat_l1920_192050


namespace NUMINAMATH_GPT_range_of_sum_of_reciprocals_l1920_192064

theorem range_of_sum_of_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  ∃ (r : ℝ), ∀ t ∈ Set.Ici (3 + 2 * Real.sqrt 2), t = (1 / x + 1 / y) := 
sorry

end NUMINAMATH_GPT_range_of_sum_of_reciprocals_l1920_192064


namespace NUMINAMATH_GPT_sum_of_ages_is_20_l1920_192024

-- Given conditions
variables (age_kiana age_twin : ℕ)
axiom product_of_ages : age_kiana * age_twin * age_twin = 162

-- Required proof
theorem sum_of_ages_is_20 : age_kiana + age_twin + age_twin = 20 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_is_20_l1920_192024


namespace NUMINAMATH_GPT_bananas_per_friend_l1920_192053

-- Define constants and conditions
def totalBananas : Nat := 40
def totalFriends : Nat := 40

-- Define the main theorem to prove
theorem bananas_per_friend : totalBananas / totalFriends = 1 := by
  sorry

end NUMINAMATH_GPT_bananas_per_friend_l1920_192053


namespace NUMINAMATH_GPT_company_pays_300_per_month_l1920_192074

theorem company_pays_300_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box_per_month : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1080000)
  (h5 : cost_per_box_per_month = 0.5) :
  (total_volume / (length * width * height)) * cost_per_box_per_month = 300 := by
  sorry

end NUMINAMATH_GPT_company_pays_300_per_month_l1920_192074


namespace NUMINAMATH_GPT_average_of_pqrs_l1920_192098

theorem average_of_pqrs (p q r s : ℚ) (h : (5/4) * (p + q + r + s) = 20) : ((p + q + r + s) / 4) = 4 :=
sorry

end NUMINAMATH_GPT_average_of_pqrs_l1920_192098


namespace NUMINAMATH_GPT_bears_on_each_shelf_l1920_192052

theorem bears_on_each_shelf (initial_bears : ℕ) (additional_bears : ℕ) (shelves : ℕ) (total_bears : ℕ) (bears_per_shelf : ℕ) :
  initial_bears = 5 → additional_bears = 7 → shelves = 2 → total_bears = initial_bears + additional_bears → bears_per_shelf = total_bears / shelves → bears_per_shelf = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_bears_on_each_shelf_l1920_192052


namespace NUMINAMATH_GPT_variance_transformed_is_8_l1920_192003

variables {n : ℕ} (x : Fin n → ℝ)

-- Given: the variance of x₁, x₂, ..., xₙ is 2.
def variance_x (x : Fin n → ℝ) : ℝ := sorry

axiom variance_x_is_2 : variance_x x = 2

-- Variance of 2 * x₁ + 3, 2 * x₂ + 3, ..., 2 * xₙ + 3
def variance_transformed (x : Fin n → ℝ) : ℝ :=
  variance_x (fun i => 2 * x i + 3)

-- Prove that the variance is 8.
theorem variance_transformed_is_8 : variance_transformed x = 8 :=
  sorry

end NUMINAMATH_GPT_variance_transformed_is_8_l1920_192003


namespace NUMINAMATH_GPT_algebraic_expression_value_l1920_192075

theorem algebraic_expression_value (a : ℝ) (h : a = Real.sqrt 6 + 2) : a^2 - 4 * a + 4 = 6 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1920_192075


namespace NUMINAMATH_GPT_train_speed_kmh_l1920_192029

variable (length_of_train_meters : ℕ) (time_to_cross_seconds : ℕ)

theorem train_speed_kmh (h1 : length_of_train_meters = 50) (h2 : time_to_cross_seconds = 6) :
  (length_of_train_meters * 3600) / (time_to_cross_seconds * 1000) = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_kmh_l1920_192029


namespace NUMINAMATH_GPT_all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l1920_192022

theorem all_palindromes_divisible_by_11 : 
  (∀ a b : ℕ, 1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 →
    (1001 * a + 110 * b) % 11 = 0 ) := sorry

theorem probability_palindrome_divisible_by_11 : 
  (∀ (palindromes : ℕ → Prop), 
  (∀ n, palindromes n ↔ ∃ (a b : ℕ), 
  1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 ∧ 
  n = 1001 * a + 110 * b) → 
  (∀ n, palindromes n → n % 11 = 0) →
  ∃ p : ℝ, p = 1) := sorry

end NUMINAMATH_GPT_all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l1920_192022


namespace NUMINAMATH_GPT_smallest_angle_CBD_l1920_192086

-- Definitions for given conditions
def angle_ABC : ℝ := 40
def angle_ABD : ℝ := 15

-- Theorem statement
theorem smallest_angle_CBD : ∃ (angle_CBD : ℝ), angle_CBD = angle_ABC - angle_ABD := by
  use 25
  sorry

end NUMINAMATH_GPT_smallest_angle_CBD_l1920_192086


namespace NUMINAMATH_GPT_man_walking_time_l1920_192049

theorem man_walking_time (D V_w V_m T : ℝ) (t : ℝ) :
  D = V_w * T →
  D_w = V_m * t →
  D - V_m * t = V_w * (T - t) →
  T - (T - t) = 16 →
  t = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_man_walking_time_l1920_192049


namespace NUMINAMATH_GPT_james_marbles_l1920_192088

def marbles_in_bag_D (bag_C : ℕ) := 2 * bag_C - 1
def marbles_in_bag_E (bag_A : ℕ) := bag_A / 2
def marbles_in_bag_G (bag_E : ℕ) := bag_E

theorem james_marbles :
    ∀ (A B C D E F G : ℕ),
      A = 4 →
      B = 3 →
      C = 5 →
      D = marbles_in_bag_D C →
      E = marbles_in_bag_E A →
      F = 3 →
      G = marbles_in_bag_G E →
      28 - (D + F) + 4 = 20 := by
    intros A B C D E F G hA hB hC hD hE hF hG
    sorry

end NUMINAMATH_GPT_james_marbles_l1920_192088


namespace NUMINAMATH_GPT_calculate_truncated_cone_volume_l1920_192040

noncomputable def volume_of_truncated_cone (R₁ R₂ h : ℝ) :
    ℝ := ((1 / 3) * Real.pi * h * (R₁ ^ 2 + R₁ * R₂ + R₂ ^ 2))

theorem calculate_truncated_cone_volume : 
    volume_of_truncated_cone 10 5 10 = (1750 / 3) * Real.pi := by
sorry

end NUMINAMATH_GPT_calculate_truncated_cone_volume_l1920_192040


namespace NUMINAMATH_GPT_warriors_won_40_games_l1920_192096

variable (H F W K R S : ℕ)

-- Conditions as given in the problem
axiom hawks_won_more_games_than_falcons : H > F
axiom knights_won_more_than_30 : K > 30
axiom warriors_won_more_than_knights_but_fewer_than_royals : W > K ∧ W < R
axiom squires_tied_with_falcons : S = F

-- The proof statement
theorem warriors_won_40_games : W = 40 :=
sorry

end NUMINAMATH_GPT_warriors_won_40_games_l1920_192096


namespace NUMINAMATH_GPT_evaluate_f_at_2_l1920_192068

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem evaluate_f_at_2 : f 2 = 259 := 
by
  -- Substitute x = 2 into the polynomial and simplify the expression.
  sorry

end NUMINAMATH_GPT_evaluate_f_at_2_l1920_192068


namespace NUMINAMATH_GPT_at_least_one_angle_not_greater_than_60_l1920_192019

theorem at_least_one_angle_not_greater_than_60 (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (hSum : A + B + C = 180) : false :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_angle_not_greater_than_60_l1920_192019


namespace NUMINAMATH_GPT_count_natural_numbers_perfect_square_l1920_192027

theorem count_natural_numbers_perfect_square :
  ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (n1^2 - 19 * n1 + 91) = m^2 ∧ (n2^2 - 19 * n2 + 91) = k^2 ∧
  ∀ n : ℕ, (n^2 - 19 * n + 91) = p^2 → n = n1 ∨ n = n2 := sorry

end NUMINAMATH_GPT_count_natural_numbers_perfect_square_l1920_192027


namespace NUMINAMATH_GPT_find_c_l1920_192046

theorem find_c (x : ℝ) (c : ℝ) (h1 : 3 * x + 5 = 4) (h2 : c * x + 6 = 3) : c = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1920_192046


namespace NUMINAMATH_GPT_min_value_l1920_192023

theorem min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : (a - 1) * 1 + 1 * (2 * b) = 0) :
    (2 / a) + (1 / b) = 8 :=
  sorry

end NUMINAMATH_GPT_min_value_l1920_192023


namespace NUMINAMATH_GPT_range_of_a_for_inequality_l1920_192080

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ p q : ℝ, (0 < p ∧ p < 1) → (0 < q ∧ q < 1) → p ≠ q → (f a p - f a q) / (p - q) > 1) ↔ 3 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_l1920_192080


namespace NUMINAMATH_GPT_sum_of_arithmetic_series_l1920_192097

theorem sum_of_arithmetic_series (a1 an : ℕ) (d n : ℕ) (s : ℕ) :
  a1 = 2 ∧ an = 100 ∧ d = 2 ∧ n = (an - a1) / d + 1 ∧ s = n * (a1 + an) / 2 → s = 2550 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_series_l1920_192097


namespace NUMINAMATH_GPT_sum_of_remaining_two_scores_l1920_192009

open Nat

theorem sum_of_remaining_two_scores :
  ∃ x y : ℕ, x + y = 160 ∧ (65 + 75 + 85 + 95 + x + y) / 6 = 80 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remaining_two_scores_l1920_192009


namespace NUMINAMATH_GPT_students_not_reading_l1920_192010

theorem students_not_reading (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_reading : ℚ) (frac_boys_reading : ℚ)
  (h1 : total_girls = 12) (h2 : total_boys = 10)
  (h3 : frac_girls_reading = 5 / 6) (h4 : frac_boys_reading = 4 / 5) :
  let girls_not_reading := total_girls - total_girls * frac_girls_reading
  let boys_not_reading := total_boys - total_boys * frac_boys_reading
  let total_not_reading := girls_not_reading + boys_not_reading
  total_not_reading = 4 := sorry

end NUMINAMATH_GPT_students_not_reading_l1920_192010


namespace NUMINAMATH_GPT_chameleon_color_change_l1920_192078

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end NUMINAMATH_GPT_chameleon_color_change_l1920_192078


namespace NUMINAMATH_GPT_players_odd_sum_probability_l1920_192017

theorem players_odd_sum_probability :
  let tiles := (1:ℕ) :: (2:ℕ) :: (3:ℕ) :: (4:ℕ) :: (5:ℕ) :: (6:ℕ) :: (7:ℕ) :: (8:ℕ) :: (9:ℕ) :: (10:ℕ) :: (11:ℕ) :: []
  let m := 1
  let n := 26
  m + n = 27 :=
by
  sorry

end NUMINAMATH_GPT_players_odd_sum_probability_l1920_192017


namespace NUMINAMATH_GPT_cristina_speed_cristina_running_speed_l1920_192083

theorem cristina_speed 
  (head_start : ℕ)
  (nicky_speed : ℕ)
  (catch_up_time : ℕ)
  (distance : ℕ := head_start + (nicky_speed * catch_up_time))
  : distance / catch_up_time = 6
  := by
  sorry

-- Given conditions used as definitions in Lean 4:
-- head_start = 36 (meters)
-- nicky_speed = 3 (meters/second)
-- catch_up_time = 12 (seconds)

theorem cristina_running_speed
  (head_start : ℕ := 36)
  (nicky_speed : ℕ := 3)
  (catch_up_time : ℕ := 12)
  : (head_start + (nicky_speed * catch_up_time)) / catch_up_time = 6
  := by
  sorry

end NUMINAMATH_GPT_cristina_speed_cristina_running_speed_l1920_192083


namespace NUMINAMATH_GPT_remainder_three_l1920_192092

-- Define the condition that x % 6 = 3
def condition (x : ℕ) : Prop := x % 6 = 3

-- Proof statement that if condition is met, then (3 * x) % 6 = 3
theorem remainder_three {x : ℕ} (h : condition x) : (3 * x) % 6 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_three_l1920_192092


namespace NUMINAMATH_GPT_probability_of_odd_sum_rows_columns_l1920_192004

open BigOperators

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_odd_sums : ℚ :=
  let even_arrangements := factorial 4
  let odd_positions := factorial 12
  let total_arrangements := factorial 16
  (even_arrangements * odd_positions : ℚ) / total_arrangements

theorem probability_of_odd_sum_rows_columns :
  probability_odd_sums = 1 / 1814400 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_odd_sum_rows_columns_l1920_192004


namespace NUMINAMATH_GPT_solve_system_l1920_192065

noncomputable def solutions (a b c : ℝ) : Prop :=
  a^4 - b^4 = c ∧ b^4 - c^4 = a ∧ c^4 - a^4 = b

theorem solve_system :
  { (a, b, c) | solutions a b c } =
  { (0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0) } :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1920_192065


namespace NUMINAMATH_GPT_planes_parallel_or_intersect_l1920_192059

variables {Plane : Type} {Line : Type}
variables (α β : Plane) (a b : Line)

-- Conditions
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def not_parallel (l1 l2 : Line) : Prop := sorry

-- Given conditions
axiom h₁ : line_in_plane a α
axiom h₂ : line_in_plane b β
axiom h₃ : not_parallel a b

-- The theorem statement
theorem planes_parallel_or_intersect : (exists l : Line, line_in_plane l α ∧ line_in_plane l β) ∨ (α = β) :=
sorry

end NUMINAMATH_GPT_planes_parallel_or_intersect_l1920_192059


namespace NUMINAMATH_GPT_find_frac_a_b_c_l1920_192011

theorem find_frac_a_b_c (a b c : ℝ) (h1 : a = 2 * b) (h2 : a^2 + b^2 = c^2) : (a + b) / c = (3 * Real.sqrt 5) / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_frac_a_b_c_l1920_192011


namespace NUMINAMATH_GPT_number_of_markings_l1920_192041

def markings (L : ℕ → ℕ) := ∀ n, (n > 0) → L n = L (n - 1) + 1

theorem number_of_markings : ∃ L : ℕ → ℕ, (∀ n, n = 1 → L n = 2) ∧ markings L ∧ L 200 = 201 := 
sorry

end NUMINAMATH_GPT_number_of_markings_l1920_192041


namespace NUMINAMATH_GPT_triangle_perimeter_l1920_192036

-- Define the given sides of the triangle
def side_a := 15
def side_b := 6
def side_c := 12

-- Define the function to calculate the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- The theorem stating that the perimeter of the given triangle is 33
theorem triangle_perimeter : perimeter side_a side_b side_c = 33 := by
  -- We can include the proof later
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1920_192036


namespace NUMINAMATH_GPT_m_cubed_plus_m_inv_cubed_l1920_192032

theorem m_cubed_plus_m_inv_cubed (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 1 = 971 :=
sorry

end NUMINAMATH_GPT_m_cubed_plus_m_inv_cubed_l1920_192032


namespace NUMINAMATH_GPT_number_of_five_ruble_coins_l1920_192091

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_five_ruble_coins_l1920_192091


namespace NUMINAMATH_GPT_find_min_value_l1920_192026

theorem find_min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) : 
  (1 / (2 * a)) + (2 / (b - 1)) ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_min_value_l1920_192026


namespace NUMINAMATH_GPT_prime_of_two_pow_sub_one_prime_l1920_192045

theorem prime_of_two_pow_sub_one_prime {n : ℕ} (h : Nat.Prime (2^n - 1)) : Nat.Prime n :=
sorry

end NUMINAMATH_GPT_prime_of_two_pow_sub_one_prime_l1920_192045


namespace NUMINAMATH_GPT_ratio_of_fractions_l1920_192020

theorem ratio_of_fractions (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) : 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 :=
sorry

end NUMINAMATH_GPT_ratio_of_fractions_l1920_192020


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1920_192038

noncomputable def a : ℝ := (0.6:ℝ) ^ (0.2:ℝ)
noncomputable def b : ℝ := (0.2:ℝ) ^ (0.2:ℝ)
noncomputable def c : ℝ := (0.2:ℝ) ^ (0.6:ℝ)

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  -- The proof can be added here if needed
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1920_192038


namespace NUMINAMATH_GPT_total_students_l1920_192006

theorem total_students (rank_right rank_left : ℕ) (h1 : rank_right = 16) (h2 : rank_left = 6) : rank_right + rank_left - 1 = 21 := by
  sorry

end NUMINAMATH_GPT_total_students_l1920_192006


namespace NUMINAMATH_GPT_find_k_value_l1920_192060

theorem find_k_value (k : ℚ) (h1 : (3, -5) ∈ {p : ℚ × ℚ | p.snd = k * p.fst}) (h2 : k ≠ 0) : k = -5 / 3 :=
sorry

end NUMINAMATH_GPT_find_k_value_l1920_192060


namespace NUMINAMATH_GPT_f_4_1981_eq_l1920_192085

def f : ℕ → ℕ → ℕ
| 0, y     => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_4_1981_eq : f 4 1981 = 2 ^ 16 - 3 := sorry

end NUMINAMATH_GPT_f_4_1981_eq_l1920_192085


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1920_192066

-- Given a line equation kx - y + 1 - 3k = 0
def line_equation (k x y : ℝ) : Prop := k * x - y + 1 - 3 * k = 0

-- We need to prove that this line passes through the point (3,1)
theorem line_passes_through_fixed_point (k : ℝ) : line_equation k 3 1 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1920_192066


namespace NUMINAMATH_GPT_R_and_D_calculation_l1920_192055

-- Define the given conditions and required calculation
def R_and_D_t : ℝ := 2640.92
def delta_APL_t_plus_1 : ℝ := 0.12

theorem R_and_D_calculation :
  (R_and_D_t / delta_APL_t_plus_1) = 22008 := by sorry

end NUMINAMATH_GPT_R_and_D_calculation_l1920_192055


namespace NUMINAMATH_GPT_total_people_3522_l1920_192071

def total_people (M W: ℕ) : ℕ := M + W

theorem total_people_3522 
    (M W: ℕ) 
    (h1: M / 9 * 45 + W / 12 * 60 = 17760)
    (h2: M % 9 = 0)
    (h3: W % 12 = 0) : 
    total_people M W = 3552 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_people_3522_l1920_192071


namespace NUMINAMATH_GPT_product_is_zero_l1920_192084

variables {a b c d : ℤ}

def system_of_equations (a b c d : ℤ) :=
  2 * a + 3 * b + 5 * c + 7 * d = 34 ∧
  3 * (d + c) = b ∧
  3 * b + c = a ∧
  c - 1 = d

theorem product_is_zero (h : system_of_equations a b c d) : 
  a * b * c * d = 0 :=
sorry

end NUMINAMATH_GPT_product_is_zero_l1920_192084


namespace NUMINAMATH_GPT_total_cost_is_correct_l1920_192013

def gravel_cost_per_cubic_foot : ℝ := 8
def discount_rate : ℝ := 0.10
def volume_in_cubic_yards : ℝ := 8
def conversion_factor : ℝ := 27

-- The initial cost for the given volume of gravel in cubic feet
noncomputable def initial_cost : ℝ := gravel_cost_per_cubic_foot * (volume_in_cubic_yards * conversion_factor)

-- The discount amount
noncomputable def discount_amount : ℝ := initial_cost * discount_rate

-- Total cost after applying discount
noncomputable def total_cost_after_discount : ℝ := initial_cost - discount_amount

theorem total_cost_is_correct : total_cost_after_discount = 1555.20 :=
sorry

end NUMINAMATH_GPT_total_cost_is_correct_l1920_192013


namespace NUMINAMATH_GPT_weighted_average_plants_per_hour_l1920_192021

theorem weighted_average_plants_per_hour :
  let heath_carrot_plants_100 := 100 * 275
  let heath_carrot_plants_150 := 150 * 325
  let heath_total_plants := heath_carrot_plants_100 + heath_carrot_plants_150
  let heath_total_time := 10 + 20
  
  let jake_potato_plants_50 := 50 * 300
  let jake_potato_plants_100 := 100 * 400
  let jake_total_plants := jake_potato_plants_50 + jake_potato_plants_100
  let jake_total_time := 12 + 18

  let total_plants := heath_total_plants + jake_total_plants
  let total_time := heath_total_time + jake_total_time
  let weighted_average := total_plants / total_time
  weighted_average = 2187.5 :=
by
  sorry

end NUMINAMATH_GPT_weighted_average_plants_per_hour_l1920_192021


namespace NUMINAMATH_GPT_phi_value_l1920_192057

noncomputable def f (x φ : ℝ) := Real.sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|) (h2 : f (π / 3) φ > f (π / 2) φ) : φ = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_phi_value_l1920_192057


namespace NUMINAMATH_GPT_greatest_number_of_large_chips_l1920_192087

theorem greatest_number_of_large_chips (s l p : ℕ) (h1 : s + l = 60) (h2 : s = l + p) 
  (hp_prime : Nat.Prime p) (hp_div : p ∣ l) : l ≤ 29 :=
by
  sorry

end NUMINAMATH_GPT_greatest_number_of_large_chips_l1920_192087


namespace NUMINAMATH_GPT_doughnuts_per_box_l1920_192079

theorem doughnuts_per_box
  (total_doughnuts : ℕ)
  (boxes_sold : ℕ)
  (doughnuts_given_away : ℕ)
  (doughnuts_per_box : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : boxes_sold = 27)
  (h3 : doughnuts_given_away = 30) :
  doughnuts_per_box = (total_doughnuts - doughnuts_given_away) / boxes_sold := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_doughnuts_per_box_l1920_192079


namespace NUMINAMATH_GPT_equal_areas_greater_perimeter_l1920_192039

noncomputable def side_length_square := Real.sqrt 3 + 3

noncomputable def length_rectangle := Real.sqrt 72 + 3 * Real.sqrt 6
noncomputable def width_rectangle := Real.sqrt 2

noncomputable def area_square := (side_length_square) ^ 2

noncomputable def area_rectangle := length_rectangle * width_rectangle

noncomputable def perimeter_square := 4 * side_length_square

noncomputable def perimeter_rectangle := 2 * (length_rectangle + width_rectangle)

theorem equal_areas : area_square = area_rectangle := sorry

theorem greater_perimeter : perimeter_square < perimeter_rectangle := sorry

end NUMINAMATH_GPT_equal_areas_greater_perimeter_l1920_192039


namespace NUMINAMATH_GPT_jill_arrives_30_minutes_before_jack_l1920_192035

theorem jill_arrives_30_minutes_before_jack
    (d : ℝ) (s_jill : ℝ) (s_jack : ℝ) (t_diff : ℝ)
    (h_d : d = 2)
    (h_s_jill : s_jill = 12)
    (h_s_jack : s_jack = 3)
    (h_t_diff : t_diff = 30) :
    ((d / s_jack) * 60 - (d / s_jill) * 60) = t_diff :=
by
  sorry

end NUMINAMATH_GPT_jill_arrives_30_minutes_before_jack_l1920_192035


namespace NUMINAMATH_GPT_people_on_bus_now_l1920_192089

variable (x : ℕ)

def original_people_on_bus : ℕ := 38
def people_got_on_bus (x : ℕ) : ℕ := x
def people_left_bus (x : ℕ) : ℕ := x + 9

theorem people_on_bus_now (x : ℕ) : original_people_on_bus - people_left_bus x + people_got_on_bus x = 29 := 
by
  sorry

end NUMINAMATH_GPT_people_on_bus_now_l1920_192089


namespace NUMINAMATH_GPT_parker_savings_l1920_192002

-- Define the costs of individual items and meals
def burger_cost : ℝ := 5
def fries_cost : ℝ := 3
def drink_cost : ℝ := 3
def special_meal_cost : ℝ := 9.5
def kids_burger_cost : ℝ := 3
def kids_fries_cost : ℝ := 2
def kids_drink_cost : ℝ := 2
def kids_meal_cost : ℝ := 5

-- Define the number of meals Mr. Parker buys
def adult_meals : ℕ := 2
def kids_meals : ℕ := 2

-- Define the total cost of individual items for adults and children
def total_individual_cost_adults : ℝ :=
  adult_meals * (burger_cost + fries_cost + drink_cost)

def total_individual_cost_children : ℝ :=
  kids_meals * (kids_burger_cost + kids_fries_cost + kids_drink_cost)

-- Define the total cost of meal deals
def total_meals_cost : ℝ :=
  adult_meals * special_meal_cost + kids_meals * kids_meal_cost

-- Define the total cost of individual items for both adults and children
def total_individual_cost : ℝ :=
  total_individual_cost_adults + total_individual_cost_children

-- Define the savings
def savings : ℝ := total_individual_cost - total_meals_cost

theorem parker_savings : savings = 7 :=
by
  sorry

end NUMINAMATH_GPT_parker_savings_l1920_192002


namespace NUMINAMATH_GPT_problem_dividing_remainder_l1920_192061

-- The conditions exported to Lean
def tiling_count (n : ℕ) : ℕ :=
  -- This function counts the number of valid tilings for a board size n with all colors used
  sorry

def remainder_when_divide (num divisor : ℕ) : ℕ := num % divisor

-- The statement problem we need to prove
theorem problem_dividing_remainder :
  remainder_when_divide (tiling_count 9) 1000 = 545 := 
sorry

end NUMINAMATH_GPT_problem_dividing_remainder_l1920_192061


namespace NUMINAMATH_GPT_xyz_inequality_l1920_192030

theorem xyz_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z ≥ 1) :
    (x^4 + y) * (y^4 + z) * (z^4 + x) ≥ (x + y^2) * (y + z^2) * (z + x^2) :=
by
  sorry

end NUMINAMATH_GPT_xyz_inequality_l1920_192030


namespace NUMINAMATH_GPT_lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l1920_192058

/- Define lines l1 and l2 -/
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

/- Prove intersection condition -/
theorem lines_intersect (a : ℝ) : (∃ x y, l1 a x y ∧ l2 a x y) ↔ (a ≠ -1 ∧ a ≠ 2) := 
sorry

/- Prove perpendicular condition -/
theorem lines_perpendicular (a : ℝ) : (∃ x1 y1 x2 y2, l1 a x1 y1 ∧ l2 a x2 y2 ∧ x1 * x2 + y1 * y2 = 0) ↔ (a = 2 / 3) :=
sorry

/- Prove coincident condition -/
theorem lines_coincide (a : ℝ) : (∀ x y, l1 a x y ↔ l2 a x y) ↔ (a = 2) := 
sorry

/- Prove parallel condition -/
theorem lines_parallel (a : ℝ) : (∀ x1 y1 x2 y2, l1 a x1 y1 → l2 a x2 y2 → (x1 * y2 - y1 * x2) = 0) ↔ (a = -1) := 
sorry

end NUMINAMATH_GPT_lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l1920_192058


namespace NUMINAMATH_GPT_line_through_intersections_of_circles_l1920_192081

-- Define the first circle
def circle₁ (x y : ℝ) : Prop :=
  x^2 + y^2 = 10

-- Define the second circle
def circle₂ (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 20

-- The statement of the mathematically equivalent proof problem
theorem line_through_intersections_of_circles : 
    (∃ (x y : ℝ), circle₁ x y ∧ circle₂ x y) → (∃ (x y : ℝ), x + 3 * y - 5 = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_line_through_intersections_of_circles_l1920_192081


namespace NUMINAMATH_GPT_current_at_time_l1920_192070

noncomputable def I (t : ℝ) : ℝ := 5 * (Real.sin (100 * Real.pi * t + Real.pi / 3))

theorem current_at_time (t : ℝ) (h : t = 1 / 200) : I t = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_current_at_time_l1920_192070


namespace NUMINAMATH_GPT_determine_k_l1920_192077

theorem determine_k (k : ℕ) : 2^2004 - 2^2003 - 2^2002 + 2^2001 = k * 2^2001 → k = 3 :=
by
  intro h
  -- now we would proceed to prove it, but we'll skip proof here
  sorry

end NUMINAMATH_GPT_determine_k_l1920_192077


namespace NUMINAMATH_GPT_pascals_triangle_53_rows_l1920_192056

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end NUMINAMATH_GPT_pascals_triangle_53_rows_l1920_192056


namespace NUMINAMATH_GPT_geometric_sequence_form_l1920_192028

-- Definitions for sequences and common difference/ratio
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ (m n : ℕ), a n = a m + (n - m) * d

def isGeometricSeq (b : ℕ → ℝ) (q : ℝ) :=
  ∀ (m n : ℕ), b n = b m * q ^ (n - m)

-- Problem statement: given an arithmetic sequence, find the form of the corresponding geometric sequence
theorem geometric_sequence_form
  (b : ℕ → ℝ) (q : ℝ) (m n : ℕ) (b_m : ℝ) (q_pos : q > 0) :
  (∀ (m n : ℕ), b n = b m * q ^ (n - m)) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_form_l1920_192028


namespace NUMINAMATH_GPT_leftover_potatoes_l1920_192033

theorem leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ)
    (h1 : fries_per_potato = 25) (h2 : total_potatoes = 15) (h3 : required_fries = 200) :
    (total_potatoes - required_fries / fries_per_potato) = 7 :=
sorry

end NUMINAMATH_GPT_leftover_potatoes_l1920_192033


namespace NUMINAMATH_GPT_x_is_one_if_pure_imaginary_l1920_192090

theorem x_is_one_if_pure_imaginary
  (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x^2 + 3 * x + 2 ≠ 0) :
  x = 1 :=
sorry

end NUMINAMATH_GPT_x_is_one_if_pure_imaginary_l1920_192090


namespace NUMINAMATH_GPT_union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l1920_192099

open Set

variables {α : Type*} (A B C : Set α)

-- Commutativity
theorem union_comm : A ∪ B = B ∪ A := sorry
theorem inter_comm : A ∩ B = B ∩ A := sorry

-- Associativity
theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry
theorem inter_assoc : A ∩ (B ∩ C) = (A ∩ B) ∩ C := sorry

-- Distributivity
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) := sorry

-- Idempotence
theorem union_idem : A ∪ A = A := sorry
theorem inter_idem : A ∩ A = A := sorry

-- De Morgan's Laws
theorem de_morgan_union : compl (A ∪ B) = compl A ∩ compl B := sorry
theorem de_morgan_inter : compl (A ∩ B) = compl A ∪ compl B := sorry

end NUMINAMATH_GPT_union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l1920_192099


namespace NUMINAMATH_GPT_angle_between_hour_and_minute_hand_at_5_oclock_l1920_192016

theorem angle_between_hour_and_minute_hand_at_5_oclock : 
  let degrees_in_circle := 360
  let hours_in_clock := 12
  let angle_per_hour := degrees_in_circle / hours_in_clock
  let hour_hand_position := 5
  let minute_hand_position := 0
  let angle := (hour_hand_position - minute_hand_position) * angle_per_hour
  angle = 150 :=
by sorry

end NUMINAMATH_GPT_angle_between_hour_and_minute_hand_at_5_oclock_l1920_192016


namespace NUMINAMATH_GPT_three_hour_classes_per_week_l1920_192069

theorem three_hour_classes_per_week (x : ℕ) : 
  (24 * (3 * x + 4 + 4) = 336) → x = 2 := by {
  sorry
}

end NUMINAMATH_GPT_three_hour_classes_per_week_l1920_192069


namespace NUMINAMATH_GPT_find_original_number_l1920_192001

theorem find_original_number (n a b: ℤ) 
  (h1 : n > 1000) 
  (h2 : n + 79 = a^2) 
  (h3 : n + 204 = b^2) 
  (h4 : b^2 - a^2 = 125) : 
  n = 3765 := 
by 
  sorry

end NUMINAMATH_GPT_find_original_number_l1920_192001


namespace NUMINAMATH_GPT_pyramid_boxes_l1920_192063

theorem pyramid_boxes (a₁ a₂ aₙ : ℕ) (d : ℕ) (n : ℕ) (Sₙ : ℕ) 
  (h₁ : a₁ = 12) 
  (h₂ : a₂ = 15) 
  (h₃ : aₙ = 39) 
  (h₄ : d = 3) 
  (h₅ : a₂ = a₁ + d)
  (h₆ : aₙ = a₁ + (n - 1) * d) 
  (h₇ : Sₙ = n * (a₁ + aₙ) / 2) :
  Sₙ = 255 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_boxes_l1920_192063


namespace NUMINAMATH_GPT_figure_100_squares_l1920_192025

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 2 * n + 1

theorem figure_100_squares : f 100 = 1020201 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_figure_100_squares_l1920_192025


namespace NUMINAMATH_GPT_compute_expression_l1920_192037

theorem compute_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b^3) / (a^2 - 2*a*b + b^2 + a*b) = 5 :=
by
  have h : a = 3 := h1
  have k : b = 2 := h2
  rw [h, k]
  sorry

end NUMINAMATH_GPT_compute_expression_l1920_192037


namespace NUMINAMATH_GPT_mean_of_quadrilateral_angles_l1920_192051

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mean_of_quadrilateral_angles_l1920_192051


namespace NUMINAMATH_GPT_find_d_l1920_192054

theorem find_d (a d : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * d) : d = 49 :=
sorry

end NUMINAMATH_GPT_find_d_l1920_192054


namespace NUMINAMATH_GPT_count_four_digit_multiples_of_5_l1920_192012

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end NUMINAMATH_GPT_count_four_digit_multiples_of_5_l1920_192012


namespace NUMINAMATH_GPT_william_max_riding_time_l1920_192048

theorem william_max_riding_time (x : ℝ) :
  (2 * x + 2 * 1.5 + 2 * (1 / 2 * x) = 21) → (x = 6) :=
by
  sorry

end NUMINAMATH_GPT_william_max_riding_time_l1920_192048


namespace NUMINAMATH_GPT_quadratic_graph_above_x_axis_l1920_192073

theorem quadratic_graph_above_x_axis (a b c : ℝ) :
  ¬ ((b^2 - 4*a*c < 0) ↔ ∀ x : ℝ, a*x^2 + b*x + c > 0) :=
sorry

end NUMINAMATH_GPT_quadratic_graph_above_x_axis_l1920_192073
