import Mathlib

namespace NUMINAMATH_GPT_paths_H_to_J_via_I_l205_20594

def binom (n k : ℕ) : ℕ := Nat.choose n k

def paths_from_H_to_I : ℕ :=
  binom 7 2  -- Calculate the number of paths from H(0,7) to I(5,5)

def paths_from_I_to_J : ℕ :=
  binom 8 3  -- Calculate the number of paths from I(5,5) to J(8,0)

theorem paths_H_to_J_via_I : paths_from_H_to_I * paths_from_I_to_J = 1176 := by
  -- This theorem states that the number of paths from H to J through I is 1176
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_paths_H_to_J_via_I_l205_20594


namespace NUMINAMATH_GPT_exists_special_cubic_polynomial_l205_20506

theorem exists_special_cubic_polynomial :
  ∃ P : Polynomial ℝ, 
    Polynomial.degree P = 3 ∧ 
    (∀ x : ℝ, Polynomial.IsRoot P x → x > 0) ∧
    (∀ x : ℝ, Polynomial.IsRoot (Polynomial.derivative P) x → x < 0) ∧
    (∃ x y : ℝ, Polynomial.IsRoot P x ∧ Polynomial.IsRoot (Polynomial.derivative P) y ∧ x ≠ y) :=
by
  sorry

end NUMINAMATH_GPT_exists_special_cubic_polynomial_l205_20506


namespace NUMINAMATH_GPT_find_d_l205_20565

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) :
  d = 180 :=
sorry

end NUMINAMATH_GPT_find_d_l205_20565


namespace NUMINAMATH_GPT_midpoint_sum_four_times_l205_20567

theorem midpoint_sum_four_times (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = -4) (h3 : x2 = -2) (h4 : y2 = 10) :
  4 * ((x1 + x2) / 2 + (y1 + y2) / 2) = 24 :=
by
  rw [h1, h2, h3, h4]
  -- simplifying to get the desired result
  sorry

end NUMINAMATH_GPT_midpoint_sum_four_times_l205_20567


namespace NUMINAMATH_GPT_remainder_of_first_105_sum_div_5280_l205_20578

theorem remainder_of_first_105_sum_div_5280:
  let n := 105
  let d := 5280
  let sum := n * (n + 1) / 2
  sum % d = 285 := by
  sorry

end NUMINAMATH_GPT_remainder_of_first_105_sum_div_5280_l205_20578


namespace NUMINAMATH_GPT_string_length_is_correct_l205_20582

noncomputable def calculate_string_length (circumference height : ℝ) (loops : ℕ) : ℝ :=
  let vertical_distance_per_loop := height / loops
  let hypotenuse_length := Real.sqrt ((circumference ^ 2) + (vertical_distance_per_loop ^ 2))
  loops * hypotenuse_length

theorem string_length_is_correct : calculate_string_length 6 16 5 = 34 := 
  sorry

end NUMINAMATH_GPT_string_length_is_correct_l205_20582


namespace NUMINAMATH_GPT_product_of_squares_l205_20534

theorem product_of_squares (x : ℝ) (h : |5 * x| + 4 = 49) : x^2 * (if x = 9 then 9 else -9)^2 = 6561 :=
by
  sorry

end NUMINAMATH_GPT_product_of_squares_l205_20534


namespace NUMINAMATH_GPT_intersection_P_Q_l205_20592

def P (x : ℝ) : Prop := x + 2 ≥ x^2

def Q (x : ℕ) : Prop := x ≤ 3

theorem intersection_P_Q :
  {x : ℕ | P x} ∩ {x : ℕ | Q x} = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l205_20592


namespace NUMINAMATH_GPT_adjust_collection_amount_l205_20523

/-- Define the error caused by mistaking half-dollars for dollars -/
def halfDollarError (x : ℕ) : ℤ := 50 * x

/-- Define the error caused by mistaking quarters for nickels -/
def quarterError (x : ℕ) : ℤ := 20 * x

/-- Define the total error based on the given conditions -/
def totalError (x : ℕ) : ℤ := halfDollarError x - quarterError x

theorem adjust_collection_amount (x : ℕ) : totalError x = 30 * x := by
  sorry

end NUMINAMATH_GPT_adjust_collection_amount_l205_20523


namespace NUMINAMATH_GPT_option_A_correct_l205_20551

theorem option_A_correct (x y : ℝ) (hy : y ≠ 0) :
  (-2 * x^2 * y + y) / y = -2 * x^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_option_A_correct_l205_20551


namespace NUMINAMATH_GPT_ellipse_foci_on_x_axis_l205_20500

variable {a b : ℝ}

theorem ellipse_foci_on_x_axis (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) (hc : ∀ x y : ℝ, (a * x^2 + b * y^2 = 1) → (1 / a > 1 / b ∧ 1 / b > 0))
  : 0 < a ∧ a < b :=
sorry

end NUMINAMATH_GPT_ellipse_foci_on_x_axis_l205_20500


namespace NUMINAMATH_GPT_average_test_score_first_25_percent_l205_20585

theorem average_test_score_first_25_percent (x : ℝ) :
  (0.25 * x) + (0.50 * 65) + (0.25 * 90) = 1 * 75 → x = 80 :=
by
  sorry

end NUMINAMATH_GPT_average_test_score_first_25_percent_l205_20585


namespace NUMINAMATH_GPT_least_five_digit_congruent_to_six_mod_seventeen_l205_20530

theorem least_five_digit_congruent_to_six_mod_seventeen : ∃ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 17 = 6 ∧ ∀ y : ℕ, y ≥ 10000 ∧ y < 100000 ∧ y % 17 = 6 → x ≤ y :=
  by
    sorry

end NUMINAMATH_GPT_least_five_digit_congruent_to_six_mod_seventeen_l205_20530


namespace NUMINAMATH_GPT_find_pumpkin_seed_packets_l205_20576

variable (P : ℕ)

-- Problem assumptions (conditions)
def pumpkin_seed_cost : ℝ := 2.50
def tomato_seed_cost_total : ℝ := 1.50 * 4
def chili_pepper_seed_cost_total : ℝ := 0.90 * 5
def total_spent : ℝ := 18.00

-- Main theorem to prove
theorem find_pumpkin_seed_packets (P : ℕ) (h : (pumpkin_seed_cost * P) + tomato_seed_cost_total + chili_pepper_seed_cost_total = total_spent) : P = 3 := by sorry

end NUMINAMATH_GPT_find_pumpkin_seed_packets_l205_20576


namespace NUMINAMATH_GPT_exp_f_f_increasing_inequality_l205_20557

noncomputable def f (a b : ℝ) (x : ℝ) :=
  (a * x + b) / (x^2 + 1)

-- Conditions
variable (a b : ℝ)
axiom h_odd : ∀ x : ℝ, f a b (-x) = - f a b x
axiom h_value : f a b (1/2) = 2/5

-- Proof statements
theorem exp_f : f a b x = x / (x^2 + 1) := sorry

theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : 
  f a b x1 < f a b x2 := sorry

theorem inequality (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  f a b (2 * x - 1) + f a b x < 0 := sorry

end NUMINAMATH_GPT_exp_f_f_increasing_inequality_l205_20557


namespace NUMINAMATH_GPT_solve_star_eq_l205_20561

noncomputable def star (a b : ℤ) : ℤ := if a = b then 2 else sorry

axiom star_assoc : ∀ (a b c : ℤ), star a (star b c) = (star a b) - c
axiom star_self_eq_two : ∀ (a : ℤ), star a a = 2

theorem solve_star_eq : ∀ (x : ℤ), star 100 (star 5 x) = 20 → x = 20 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_solve_star_eq_l205_20561


namespace NUMINAMATH_GPT_charity_fundraising_l205_20504

theorem charity_fundraising (num_people : ℕ) (amount_event1 amount_event2 : ℕ) (total_amount_per_person : ℕ) :
  num_people = 8 →
  amount_event1 = 2000 →
  amount_event2 = 1000 →
  total_amount_per_person = (amount_event1 + amount_event2) / num_people →
  total_amount_per_person = 375 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_charity_fundraising_l205_20504


namespace NUMINAMATH_GPT_juice_cans_count_l205_20542

theorem juice_cans_count :
  let original_price := 12 
  let discount := 2 
  let tub_sale_price := original_price - discount 
  let tub_quantity := 2 
  let ice_cream_total := tub_quantity * tub_sale_price 
  let total_payment := 24 
  let juice_cost_per_5cans := 2 
  let remaining_amount := total_payment - ice_cream_total 
  let sets_of_juice_cans := remaining_amount / juice_cost_per_5cans 
  let cans_per_set := 5 
  2 * cans_per_set = 10 :=
by
  sorry

end NUMINAMATH_GPT_juice_cans_count_l205_20542


namespace NUMINAMATH_GPT_fraction_equality_l205_20577

variable (a b : ℚ)

theorem fraction_equality (h : (4 * a + 3 * b) / (4 * a - 3 * b) = 4) : a / b = 5 / 4 := by
  sorry

end NUMINAMATH_GPT_fraction_equality_l205_20577


namespace NUMINAMATH_GPT_solve_system_of_odes_l205_20566

theorem solve_system_of_odes (C₁ C₂ : ℝ) :
  ∃ (x y : ℝ → ℝ),
    (∀ t, x t = (C₁ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, y t = (C₁ + C₂ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, deriv x t = 2 * x t + y t) ∧
    (∀ t, deriv y t = 4 * y t - x t) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_odes_l205_20566


namespace NUMINAMATH_GPT_value_of_4b_minus_a_l205_20535

theorem value_of_4b_minus_a (a b : ℕ) (h1 : a > b) (h2 : x^2 - 20*x + 96 = (x - a)*(x - b)) : 4*b - a = 20 :=
  sorry

end NUMINAMATH_GPT_value_of_4b_minus_a_l205_20535


namespace NUMINAMATH_GPT_nicholas_bottle_caps_l205_20519

theorem nicholas_bottle_caps (N : ℕ) (h : N + 85 = 93) : N = 8 :=
by
  sorry

end NUMINAMATH_GPT_nicholas_bottle_caps_l205_20519


namespace NUMINAMATH_GPT_terminating_decimal_zeros_l205_20570

-- Define a generic environment for terminating decimal and problem statement
def count_zeros (d : ℚ) : ℕ :=
  -- This function needs to count the zeros after the decimal point and before
  -- the first non-zero digit, but its actual implementation is skipped here.
  sorry

-- Define the specific fraction in question
def my_fraction : ℚ := 1 / (2^3 * 5^5)

-- State what we need to prove: the number of zeros after the decimal point
-- in the terminating representation of my_fraction should be 4
theorem terminating_decimal_zeros : count_zeros my_fraction = 4 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_terminating_decimal_zeros_l205_20570


namespace NUMINAMATH_GPT_cube_root_of_neg_eight_l205_20550

theorem cube_root_of_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_of_neg_eight_l205_20550


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l205_20591

theorem breadth_of_rectangular_plot (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 432) : b = 12 := 
sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l205_20591


namespace NUMINAMATH_GPT_second_watermelon_correct_weight_l205_20595

-- Define various weights involved as given in the conditions
def first_watermelon_weight : ℝ := 9.91
def total_watermelon_weight : ℝ := 14.02

-- Define the weight of the second watermelon
def second_watermelon_weight : ℝ :=
  total_watermelon_weight - first_watermelon_weight

-- State the theorem to prove that the weight of the second watermelon is 4.11 pounds
theorem second_watermelon_correct_weight : second_watermelon_weight = 4.11 :=
by
  -- This ensures the statement can be built successfully in Lean 4
  sorry

end NUMINAMATH_GPT_second_watermelon_correct_weight_l205_20595


namespace NUMINAMATH_GPT_part1_part2_l205_20584

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x * abs (x^2 - a)

-- Define the two main proofs to be shown
theorem part1 (a : ℝ) (h : a = 1) : 
  ∃ I1 I2 : Set ℝ, I1 = Set.Icc (-1 - Real.sqrt 2) (-1) ∧ I2 = Set.Icc (-1 + Real.sqrt 2) (1) ∧ 
  ∀ x ∈ I1 ∪ I2, ∀ y ∈ I1 ∪ I2, x ≤ y → f y 1 ≤ f x 1 :=
sorry

theorem part2 (a : ℝ) (h : a ≥ 0) (h_roots : ∀ m : ℝ, (∃ x : ℝ, x > 0 ∧ f x a = m) ∧ (∃ x : ℝ, x < 0 ∧ f x a = m)) : 
  ∃ m : ℝ, m = 4 / (Real.exp 2) :=
sorry

end NUMINAMATH_GPT_part1_part2_l205_20584


namespace NUMINAMATH_GPT_gcd_of_four_sum_1105_l205_20539

theorem gcd_of_four_sum_1105 (a b c d : ℕ) (h_sum : a + b + c + d = 1105)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (h_neq_ab : a ≠ b) (h_neq_ac : a ≠ c) (h_neq_ad : a ≠ d)
  (h_neq_bc : b ≠ c) (h_neq_bd : b ≠ d) (h_neq_cd : c ≠ d)
  (h_gcd_ab : gcd a b > 1) (h_gcd_ac : gcd a c > 1) (h_gcd_ad : gcd a d > 1)
  (h_gcd_bc : gcd b c > 1) (h_gcd_bd : gcd b d > 1) (h_gcd_cd : gcd c d > 1) :
  gcd a (gcd b (gcd c d)) = 221 := by
  sorry

end NUMINAMATH_GPT_gcd_of_four_sum_1105_l205_20539


namespace NUMINAMATH_GPT_probability_two_black_balls_l205_20521

theorem probability_two_black_balls (white_balls black_balls drawn_balls : ℕ) 
  (h_w : white_balls = 4) (h_b : black_balls = 7) (h_d : drawn_balls = 2) :
  let total_ways := Nat.choose (white_balls + black_balls) drawn_balls
  let black_ways := Nat.choose black_balls drawn_balls
  (black_ways / total_ways : ℚ) = 21 / 55 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_black_balls_l205_20521


namespace NUMINAMATH_GPT_pages_with_money_l205_20583

def cost_per_page : ℝ := 3.5
def total_money : ℝ := 15 * 100

theorem pages_with_money : ⌊total_money / cost_per_page⌋ = 428 :=
by sorry

end NUMINAMATH_GPT_pages_with_money_l205_20583


namespace NUMINAMATH_GPT_pentagonal_tiles_count_l205_20593

theorem pentagonal_tiles_count (t p : ℕ) (h1 : t + p = 30) (h2 : 3 * t + 5 * p = 100) : p = 5 :=
sorry

end NUMINAMATH_GPT_pentagonal_tiles_count_l205_20593


namespace NUMINAMATH_GPT_Tim_has_16_pencils_l205_20569

variable (T_Sarah T_Tyrah T_Tim : Nat)

-- Conditions
def condition1 : Prop := T_Tyrah = 6 * T_Sarah
def condition2 : Prop := T_Tim = 8 * T_Sarah
def condition3 : Prop := T_Tyrah = 12

-- Theorem to prove
theorem Tim_has_16_pencils (h1 : condition1 T_Sarah T_Tyrah) (h2 : condition2 T_Sarah T_Tim) (h3 : condition3 T_Tyrah) : T_Tim = 16 :=
by
  sorry

end NUMINAMATH_GPT_Tim_has_16_pencils_l205_20569


namespace NUMINAMATH_GPT_product_of_roots_l205_20573

theorem product_of_roots : ∀ x : ℝ, (x + 3) * (x - 4) = 17 → (∃ a b : ℝ, (x = a ∨ x = b) ∧ a * b = -29) :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_l205_20573


namespace NUMINAMATH_GPT_five_letters_three_mailboxes_l205_20589

theorem five_letters_three_mailboxes : (∃ n : ℕ, n = 5) ∧ (∃ m : ℕ, m = 3) → ∃ k : ℕ, k = m^n :=
by
  sorry

end NUMINAMATH_GPT_five_letters_three_mailboxes_l205_20589


namespace NUMINAMATH_GPT_predicted_value_y_at_x_5_l205_20520

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem predicted_value_y_at_x_5 :
  let x_values := [-2, -1, 0, 1, 2]
  let y_values := [5, 4, 2, 2, 1]
  let x_bar := mean x_values
  let y_bar := mean y_values
  let a_hat := y_bar
  (∀ x, y = -x + a_hat) →
  (x = 5 → y = -2.2) :=
by
  sorry

end NUMINAMATH_GPT_predicted_value_y_at_x_5_l205_20520


namespace NUMINAMATH_GPT_ben_current_age_l205_20586

theorem ben_current_age (a b c : ℕ) 
  (h1 : a + b + c = 36) 
  (h2 : c = 2 * a - 4) 
  (h3 : b + 5 = 3 * (a + 5) / 4) : 
  b = 5 := 
by
  sorry

end NUMINAMATH_GPT_ben_current_age_l205_20586


namespace NUMINAMATH_GPT_pieces_bound_l205_20590

open Finset

variable {n : ℕ} (B W : ℕ)

theorem pieces_bound (n : ℕ) (B W : ℕ) (hB : B ≤ n^2) (hW : W ≤ n^2) :
    B ≤ n^2 ∨ W ≤ n^2 := 
by
  sorry

end NUMINAMATH_GPT_pieces_bound_l205_20590


namespace NUMINAMATH_GPT_find_OH_squared_l205_20563

variables (A B C : ℝ) (a b c R OH : ℝ)

-- Conditions
def circumcenter (O : ℝ) := true  -- Placeholder, as the actual definition relies on geometric properties
def orthocenter (H : ℝ) := true   -- Placeholder, as the actual definition relies on geometric properties

axiom eqR : R = 5
axiom sumSquares : a^2 + b^2 + c^2 = 50

-- Problem statement
theorem find_OH_squared : OH^2 = 175 :=
by
  sorry

end NUMINAMATH_GPT_find_OH_squared_l205_20563


namespace NUMINAMATH_GPT_perfect_square_condition_l205_20548

theorem perfect_square_condition (x m : ℝ) (h : ∃ k : ℝ, x^2 + x + 2*m = k^2) : m = 1/8 := 
sorry

end NUMINAMATH_GPT_perfect_square_condition_l205_20548


namespace NUMINAMATH_GPT_missing_dog_number_l205_20574

theorem missing_dog_number {S : Finset ℕ} (h₁ : S =  Finset.range 25 \ {24}) (h₂ : S.sum id = 276) :
  (∃ y ∈ S, y = (S.sum id - y) / (S.card - 1)) ↔ 24 ∉ S :=
by
  sorry

end NUMINAMATH_GPT_missing_dog_number_l205_20574


namespace NUMINAMATH_GPT_points_on_line_y1_gt_y2_l205_20503

theorem points_on_line_y1_gt_y2 (y1 y2 : ℝ) : 
    (∀ x y, y = -x + 3 → 
    ((x = -4 → y = y1) ∧ (x = 2 → y = y2))) → 
    y1 > y2 :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_y1_gt_y2_l205_20503


namespace NUMINAMATH_GPT_shorter_leg_right_triangle_l205_20533

theorem shorter_leg_right_triangle (a b c : ℕ) (h0 : a^2 + b^2 = c^2) (h1 : c = 39) (h2 : a < b) : a = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_shorter_leg_right_triangle_l205_20533


namespace NUMINAMATH_GPT_new_population_difference_l205_20581

def population_eagles : ℕ := 150
def population_falcons : ℕ := 200
def population_hawks : ℕ := 320
def population_owls : ℕ := 270
def increase_rate : ℕ := 10

theorem new_population_difference :
  let least_populous := min population_eagles (min population_falcons (min population_hawks population_owls))
  let most_populous := max population_eagles (max population_falcons (max population_hawks population_owls))
  let increased_least_populous := least_populous + least_populous * increase_rate / 100
  most_populous - increased_least_populous = 155 :=
by
  sorry

end NUMINAMATH_GPT_new_population_difference_l205_20581


namespace NUMINAMATH_GPT_measure_four_messzely_l205_20580

theorem measure_four_messzely (c3 c5 : ℕ) (hc3 : c3 = 3) (hc5 : c5 = 5) : 
  ∃ (x y z : ℕ), x = 4 ∧ x + y * c3 + z * c5 = 4 := 
sorry

end NUMINAMATH_GPT_measure_four_messzely_l205_20580


namespace NUMINAMATH_GPT_length_of_bridge_l205_20502

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (time_to_pass_bridge : ℝ) 
  (train_length_eq : train_length = 400)
  (train_speed_kmh_eq : train_speed_kmh = 60) 
  (time_to_pass_bridge_eq : time_to_pass_bridge = 72)
  : ∃ (bridge_length : ℝ), bridge_length = 800.24 := 
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l205_20502


namespace NUMINAMATH_GPT_betty_gave_stuart_percentage_l205_20572

theorem betty_gave_stuart_percentage (P : ℝ) 
  (betty_marbles : ℝ := 60) 
  (stuart_initial_marbles : ℝ := 56) 
  (stuart_final_marbles : ℝ := 80)
  (increase_in_stuart_marbles : ℝ := stuart_final_marbles - stuart_initial_marbles)
  (betty_to_stuart : ℝ := (P / 100) * betty_marbles) :
  56 + ((P / 100) * betty_marbles) = 80 → P = 40 :=
by
  intros h
  -- Sorry is used since the proof steps are not required
  sorry

end NUMINAMATH_GPT_betty_gave_stuart_percentage_l205_20572


namespace NUMINAMATH_GPT_problem_solution_l205_20588

def count_multiples_of_5_not_15 : ℕ := 
  let count_up_to (m n : ℕ) := n / m
  let multiples_of_5_up_to_300 := count_up_to 5 299
  let multiples_of_15_up_to_300 := count_up_to 15 299
  multiples_of_5_up_to_300 - multiples_of_15_up_to_300

theorem problem_solution : count_multiples_of_5_not_15 = 40 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l205_20588


namespace NUMINAMATH_GPT_part1_part2_l205_20526

-- Definitions for the sets A and B
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Proof problem (1): A ∩ B = {2} implies a = -5 or a = 1
theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := 
sorry

-- Proof problem (2): A ∪ B = A implies a > 3
theorem part2 (a : ℝ) (h : A ∪ B a = A) : 3 < a :=
sorry

end NUMINAMATH_GPT_part1_part2_l205_20526


namespace NUMINAMATH_GPT_correct_statements_about_opposite_numbers_l205_20554

/-- Definition of opposite numbers: two numbers are opposite if one is the negative of the other --/
def is_opposite (a b : ℝ) : Prop := a = -b

theorem correct_statements_about_opposite_numbers (a b : ℝ) :
  (is_opposite a b ↔ a + b = 0) ∧
  (a + b = 0 ↔ is_opposite a b) ∧
  ((is_opposite a b ∧ a ≠ 0 ∧ b ≠ 0) ↔ (a / b = -1)) ∧
  ((a / b = -1 ∧ b ≠ 0) ↔ is_opposite a b) :=
by {
  sorry -- Proof is omitted
}

end NUMINAMATH_GPT_correct_statements_about_opposite_numbers_l205_20554


namespace NUMINAMATH_GPT_log_exp_identity_l205_20505

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem log_exp_identity : 2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := 
by
  sorry

end NUMINAMATH_GPT_log_exp_identity_l205_20505


namespace NUMINAMATH_GPT_find_x_given_inverse_relationship_l205_20546

theorem find_x_given_inverse_relationship :
  ∀ (x y: ℝ), (0 < x ∧ 0 < y) ∧ ((x^3 * y = 64) ↔ (x = 2 ∧ y = 8)) ∧ (y = 500) →
  x = 2 / 5 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_find_x_given_inverse_relationship_l205_20546


namespace NUMINAMATH_GPT_find_a_l205_20528

theorem find_a (a : ℝ) (h : 6 * a + 4 = 0) : a = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l205_20528


namespace NUMINAMATH_GPT_smallest_abc_sum_l205_20511

theorem smallest_abc_sum : 
  ∃ (a b c : ℕ), (a * c + 2 * b * c + a + 2 * b = c^2 + c + 6) ∧ (∀ (a' b' c' : ℕ), (a' * c' + 2 * b' * c' + a' + 2 * b' = c'^2 + c' + 6) → (a' + b' + c' ≥ a + b + c)) → (a, b, c) = (2, 1, 1) := 
by
  sorry

end NUMINAMATH_GPT_smallest_abc_sum_l205_20511


namespace NUMINAMATH_GPT_Mark_sold_1_box_less_than_n_l205_20552

variable (M A n : ℕ)

theorem Mark_sold_1_box_less_than_n (h1 : n = 8)
 (h2 : A = n - 2)
 (h3 : M + A < n)
 (h4 : M ≥ 1) 
 (h5 : A ≥ 1)
 : M = 1 := 
sorry

end NUMINAMATH_GPT_Mark_sold_1_box_less_than_n_l205_20552


namespace NUMINAMATH_GPT_incorrect_option_C_l205_20524

theorem incorrect_option_C (a b d : ℝ) (h₁ : ∀ x : ℝ, x ≠ d → x^2 + a * x + b > 0) (h₂ : a > 0) :
  ¬∀ x₁ x₂ : ℝ, (x₁ * x₂ > 0) → ((x₁, x₂) ∈ {p : (ℝ × ℝ) | p.1^2 + a * p.1 - b < 0 ∧ p.2^2 + a * p.2 - b < 0}) :=
sorry

end NUMINAMATH_GPT_incorrect_option_C_l205_20524


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l205_20514

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m ^ 2 - 2 * m + 1) / m) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l205_20514


namespace NUMINAMATH_GPT_sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l205_20549

theorem sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine : 
  Real.sqrt (3^3 + 3^3 + 3^3) = 9 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l205_20549


namespace NUMINAMATH_GPT_compute_product_l205_20564

theorem compute_product (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : x1^3 - 3 * x1 * y1^2 = 1005) 
  (h2 : y1^3 - 3 * x1^2 * y1 = 1004)
  (h3 : x2^3 - 3 * x2 * y2^2 = 1005)
  (h4 : y2^3 - 3 * x2^2 * y2 = 1004)
  (h5 : x3^3 - 3 * x3 * y3^2 = 1005)
  (h6 : y3^3 - 3 * x3^2 * y3 = 1004) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 502 := 
sorry

end NUMINAMATH_GPT_compute_product_l205_20564


namespace NUMINAMATH_GPT_maximum_k_l205_20596

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

-- Prove that the maximum integer value k satisfying k(x - 2) < f(x) for all x > 2 is 4.
theorem maximum_k (x : ℝ) (hx : x > 2) : ∃ k : ℤ, k = 4 ∧ (∀ x > 2, k * (x - 2) < f x) :=
sorry

end NUMINAMATH_GPT_maximum_k_l205_20596


namespace NUMINAMATH_GPT_largest_integer_solution_l205_20537

theorem largest_integer_solution : ∃ x : ℤ, (x ≤ 10) ∧ (∀ y : ℤ, (y > 10 → (y / 4 + 5 / 6 < 7 / 2) = false)) :=
sorry

end NUMINAMATH_GPT_largest_integer_solution_l205_20537


namespace NUMINAMATH_GPT_neg_pi_lt_neg_three_l205_20501

theorem neg_pi_lt_neg_three (h : Real.pi > 3) : -Real.pi < -3 :=
sorry

end NUMINAMATH_GPT_neg_pi_lt_neg_three_l205_20501


namespace NUMINAMATH_GPT_fourth_vertex_exists_l205_20559

structure Point :=
  (x : ℚ)
  (y : ℚ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  let M_AC := Point.mk ((A.x + C.x) / 2) ((A.y + C.y) / 2)
  let M_BD := Point.mk ((B.x + D.x) / 2) ((B.y + D.y) / 2)
  is_midpoint M_AC A C ∧ is_midpoint M_BD B D ∧ M_AC = M_BD

theorem fourth_vertex_exists (A B C : Point) (hA : A = ⟨-1, 0⟩) (hB : B = ⟨3, 0⟩) (hC : C = ⟨1, -5⟩) :
  ∃ D : Point, (D = ⟨1, 5⟩ ∨ D = ⟨-3, -5⟩) ∧ is_parallelogram A B C D :=
by
  sorry

end NUMINAMATH_GPT_fourth_vertex_exists_l205_20559


namespace NUMINAMATH_GPT_evaluation_of_expression_l205_20543

theorem evaluation_of_expression :
  10 * (1 / 8) - 6.4 / 8 + 1.2 * 0.125 = 0.6 :=
by sorry

end NUMINAMATH_GPT_evaluation_of_expression_l205_20543


namespace NUMINAMATH_GPT_solve_for_a_l205_20517

noncomputable def special_otimes (a b : ℝ) : ℝ :=
  if a > b then a^2 + b else a + b^2

theorem solve_for_a (a : ℝ) : special_otimes a (-2) = 4 → a = Real.sqrt 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_a_l205_20517


namespace NUMINAMATH_GPT_percentage_of_green_eyed_brunettes_l205_20513

def conditions (a b c d : ℝ) : Prop :=
  (a / (a + b) = 0.65) ∧
  (b / (b + c) = 0.7) ∧
  (c / (c + d) = 0.1)

theorem percentage_of_green_eyed_brunettes (a b c d : ℝ) (h : conditions a b c d) :
  d / (a + b + c + d) = 0.54 :=
sorry

end NUMINAMATH_GPT_percentage_of_green_eyed_brunettes_l205_20513


namespace NUMINAMATH_GPT_jellybeans_red_l205_20545

-- Define the individual quantities of each color of jellybean.
def b := 14
def p := 26
def o := 40
def pk := 7
def y := 21
def T := 237

-- Prove that the number of red jellybeans is 129.
theorem jellybeans_red : T - (b + p + o + pk + y) = 129 := by
  -- (optional: you can include intermediate steps if needed, but it's not required here)
  sorry

end NUMINAMATH_GPT_jellybeans_red_l205_20545


namespace NUMINAMATH_GPT_value_of_expression_l205_20527

theorem value_of_expression (n : ℕ) (a : ℝ) (h1 : 6 * 11 * n ≠ 0) (h2 : a ^ (2 * n) = 5) : 2 * a ^ (6 * n) - 4 = 246 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l205_20527


namespace NUMINAMATH_GPT_each_boy_receives_52_l205_20547

theorem each_boy_receives_52 {boys girls : ℕ} (h_ratio : boys / gcd boys girls = 5 ∧ girls / gcd boys girls = 7) (h_total : boys + girls = 180) (h_share : 3900 ∣ boys) :
  3900 / boys = 52 :=
by
  sorry

end NUMINAMATH_GPT_each_boy_receives_52_l205_20547


namespace NUMINAMATH_GPT_real_and_imag_parts_of_z_l205_20562

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem real_and_imag_parts_of_z :
  ∀ (i : ℂ), i * i = -1 → 
  ∀ (z : ℂ), z = i * (-1 + 2 * i) → real_part z = -2 ∧ imag_part z = -1 :=
by 
  intros i hi z hz
  sorry

end NUMINAMATH_GPT_real_and_imag_parts_of_z_l205_20562


namespace NUMINAMATH_GPT_distinct_balls_boxes_l205_20536

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_distinct_balls_boxes_l205_20536


namespace NUMINAMATH_GPT_total_matches_played_l205_20553

def home_team_wins := 3
def home_team_draws := 4
def home_team_losses := 0
def rival_team_wins := 2 * home_team_wins
def rival_team_draws := home_team_draws
def rival_team_losses := 0

theorem total_matches_played :
  home_team_wins + home_team_draws + home_team_losses + rival_team_wins + rival_team_draws + rival_team_losses = 17 :=
by
  sorry

end NUMINAMATH_GPT_total_matches_played_l205_20553


namespace NUMINAMATH_GPT_find_line_equation_l205_20507

-- Definitions: Point and Line in 2D
structure Point2D where
  x : ℝ
  y : ℝ

structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Line passes through the point
def line_through_point (L : Line2D) (P : Point2D) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

-- Perpendicular lines condition: if Line L1 and Line L2 are perpendicular.
def perpendicular (L1 L2 : Line2D) : Prop :=
  L1.a * L2.a + L1.b * L2.b = 0

-- Define line1 and line2 as given
def line1 : Line2D := {a := 1, b := -2, c := 0} -- corresponds to x - 2y + m = 0

-- Define point P (-1, 3)
def P : Point2D := {x := -1, y := 3}

-- Required line passing through point P and perpendicular to line1
def required_line : Line2D := {a := 2, b := 1, c := -1}

-- The proof goal
theorem find_line_equation : (line_through_point required_line P) ∧ (perpendicular line1 required_line) :=
by
  sorry

end NUMINAMATH_GPT_find_line_equation_l205_20507


namespace NUMINAMATH_GPT_max_value_of_a_l205_20599

noncomputable def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem max_value_of_a (a b c d : ℝ) (h_deriv_bounds : ∀ x, 0 ≤ x → x ≤ 1 → abs (3 * a * x^2 + 2 * b * x + c) ≤ 1) (h_a_nonzero : a ≠ 0) :
  a ≤ 8 / 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_a_l205_20599


namespace NUMINAMATH_GPT_convert_C_to_F_l205_20522

theorem convert_C_to_F (C F : ℝ) (h1 : C = 40) (h2 : C = 5 / 9 * (F - 32)) : F = 104 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_convert_C_to_F_l205_20522


namespace NUMINAMATH_GPT_no_real_solution_l205_20509

theorem no_real_solution (x : ℝ) : ¬ (x^3 + 2 * (x + 1)^3 + 3 * (x + 2)^3 = 6 * (x + 4)^3) :=
sorry

end NUMINAMATH_GPT_no_real_solution_l205_20509


namespace NUMINAMATH_GPT_volume_in_cubic_yards_l205_20525

theorem volume_in_cubic_yards (V : ℝ) (conversion_factor : ℝ) (hV : V = 216) (hcf : conversion_factor = 27) :
  V / conversion_factor = 8 := by
  sorry

end NUMINAMATH_GPT_volume_in_cubic_yards_l205_20525


namespace NUMINAMATH_GPT_sum_first_7_terms_is_105_l205_20560

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables {a d : ℕ}
axiom a4_is_15 : arithmetic_seq a d 4 = 15

-- Goal/theorem to be proven
theorem sum_first_7_terms_is_105 : sum_arithmetic_seq a d 7 = 105 :=
sorry

end NUMINAMATH_GPT_sum_first_7_terms_is_105_l205_20560


namespace NUMINAMATH_GPT_monotone_increasing_solve_inequality_l205_20598

section MathProblem

variable {f : ℝ → ℝ}

theorem monotone_increasing (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y) 
(h₂ : ∀ x : ℝ, 1 < x → 0 < f x) : 
∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := sorry

theorem solve_inequality (h₃ : f 2 = 1) (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x + f y) 
(h₅ : ∀ x : ℝ, 1 < x → 0 < f x) :
∀ x : ℝ, 0 < x → f x + f (x - 3) ≤ 2 → 3 < x ∧ x ≤ 4 := sorry

end MathProblem

end NUMINAMATH_GPT_monotone_increasing_solve_inequality_l205_20598


namespace NUMINAMATH_GPT_diamond_comm_l205_20508

def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^2 - b^2

theorem diamond_comm (x y : ℝ) : diamond x y = diamond y x := by
  sorry

end NUMINAMATH_GPT_diamond_comm_l205_20508


namespace NUMINAMATH_GPT_Jim_remaining_distance_l205_20531

theorem Jim_remaining_distance (t d r : ℕ) (h₁ : t = 1200) (h₂ : d = 923) (h₃ : r = t - d) : r = 277 := 
by 
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_Jim_remaining_distance_l205_20531


namespace NUMINAMATH_GPT_range_of_k_l205_20532

theorem range_of_k (x : ℝ) (h1 : 0 < x) (h2 : x < 2) (h3 : x / Real.exp x < 1 / (k + 2 * x - x^2)) :
    0 ≤ k ∧ k < Real.exp 1 - 1 :=
sorry

end NUMINAMATH_GPT_range_of_k_l205_20532


namespace NUMINAMATH_GPT_weekly_charge_for_motel_l205_20529

theorem weekly_charge_for_motel (W : ℝ) (h1 : ∀ t : ℝ, t = 3 * 4 → t = 12)
(h2 : ∀ cost_weekly : ℝ, cost_weekly = 12 * W)
(h3 : ∀ cost_monthly : ℝ, cost_monthly = 3 * 1000)
(h4 : cost_monthly + 360 = 12 * W) : 
W = 280 := 
sorry

end NUMINAMATH_GPT_weekly_charge_for_motel_l205_20529


namespace NUMINAMATH_GPT_A_share_in_profit_l205_20510

-- Given conditions:
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12600

-- The statement we need to prove:
theorem A_share_in_profit :
  (3 / 10) * total_profit = 3780 := by
  sorry

end NUMINAMATH_GPT_A_share_in_profit_l205_20510


namespace NUMINAMATH_GPT_hyperbola_foci_eccentricity_l205_20541

-- Definitions and conditions
def hyperbola_eq := (x y : ℝ) → (x^2 / 4) - (y^2 / 12) = 1

-- Proof goals: Coordinates of the foci and eccentricity
theorem hyperbola_foci_eccentricity (x y : ℝ) : 
  (∃ c : ℝ, (x^2 / 4) - (y^2 / 12) = 1 ∧ (x = 4 ∧ y = 0) ∨ (x = -4 ∧ y = 0)) ∧ 
  (∃ e : ℝ, e = 2) :=
sorry

end NUMINAMATH_GPT_hyperbola_foci_eccentricity_l205_20541


namespace NUMINAMATH_GPT_john_younger_than_mark_l205_20538

variable (Mark_age John_age Parents_age : ℕ)
variable (h_mark : Mark_age = 18)
variable (h_parents_age_relation : Parents_age = 5 * John_age)
variable (h_parents_when_mark_born : Parents_age = 22 + Mark_age)

theorem john_younger_than_mark : Mark_age - John_age = 10 :=
by
  -- We state the theorem and leave the proof as sorry
  sorry

end NUMINAMATH_GPT_john_younger_than_mark_l205_20538


namespace NUMINAMATH_GPT_sin_theta_val_sin_2theta_pi_div_6_val_l205_20515

open Real

theorem sin_theta_val (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) 
  (hcos : cos (θ + π / 6) = 1 / 3) : 
  sin θ = (2 * sqrt 6 - 1) / 6 := 
by sorry

theorem sin_2theta_pi_div_6_val (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (hcos : cos (θ + π / 6) = 1 / 3) : 
  sin (2 * θ + π / 6) = (4 * sqrt 6 + 7) / 18 := 
by sorry

end NUMINAMATH_GPT_sin_theta_val_sin_2theta_pi_div_6_val_l205_20515


namespace NUMINAMATH_GPT_balloons_total_l205_20579

theorem balloons_total (number_of_groups balloons_per_group : ℕ)
  (h1 : number_of_groups = 7) (h2 : balloons_per_group = 5) : 
  number_of_groups * balloons_per_group = 35 := by
  sorry

end NUMINAMATH_GPT_balloons_total_l205_20579


namespace NUMINAMATH_GPT_complement_of_angle_correct_l205_20587

def complement_of_angle (a : ℚ) : ℚ := 90 - a

theorem complement_of_angle_correct : complement_of_angle (40 + 30/60) = 49 + 30/60 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_complement_of_angle_correct_l205_20587


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l205_20568

theorem arithmetic_sequence_sum (d : ℕ) (y : ℕ) (x : ℕ) (h_y : y = 39) (h_d : d = 6) 
  (h_x : x = y - d) : 
  x + y = 72 := by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l205_20568


namespace NUMINAMATH_GPT_consecutive_odd_numbers_l205_20518

/- 
  Out of some consecutive odd numbers, 9 times the first number 
  is equal to the addition of twice the third number and adding 9 
  to twice the second. Let x be the first number, then we aim to prove that 
  9 * x = 2 * (x + 4) + 2 * (x + 2) + 9 ⟹ x = 21 / 5
-/

theorem consecutive_odd_numbers (x : ℚ) (h : 9 * x = 2 * (x + 4) + 2 * (x + 2) + 9) : x = 21 / 5 :=
sorry

end NUMINAMATH_GPT_consecutive_odd_numbers_l205_20518


namespace NUMINAMATH_GPT_exists_point_at_distance_l205_20555

def Line : Type := sorry
def Point : Type := sorry
def distance (P Q : Point) : ℝ := sorry

variables (L : Line) (d : ℝ) (P : Point)

def is_at_distance (Q : Point) (L : Line) (d : ℝ) := ∃ Q, distance Q L = d

theorem exists_point_at_distance :
  ∃ Q : Point, is_at_distance Q L d :=
sorry

end NUMINAMATH_GPT_exists_point_at_distance_l205_20555


namespace NUMINAMATH_GPT_family_of_sets_properties_l205_20540

variable {X : Type}
variable {t n k : ℕ}
variable (A : Fin t → Set X)
variable (card : Set X → ℕ)
variable (h_card : ∀ (i j : Fin t), i ≠ j → card (A i ∩ A j) = k)

theorem family_of_sets_properties :
  (k = 0 → t ≤ n+1) ∧ (k ≠ 0 → t ≤ n) :=
by
  sorry

end NUMINAMATH_GPT_family_of_sets_properties_l205_20540


namespace NUMINAMATH_GPT_alice_coins_percentage_l205_20512

theorem alice_coins_percentage :
  let penny := 1
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let total_cents := penny + dime + quarter + half_dollar
  (total_cents / 100) * 100 = 86 :=
by
  sorry

end NUMINAMATH_GPT_alice_coins_percentage_l205_20512


namespace NUMINAMATH_GPT_f_of_2_l205_20575

def f (x : ℝ) : ℝ := sorry

theorem f_of_2 : f 2 = 20 / 3 :=
    sorry

end NUMINAMATH_GPT_f_of_2_l205_20575


namespace NUMINAMATH_GPT_three_four_five_six_solution_l205_20544

-- State that the equation 3^x + 4^x = 5^x is true when x=2
axiom three_four_five_solution : 3^2 + 4^2 = 5^2

-- We need to prove the following theorem
theorem three_four_five_six_solution : 3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end NUMINAMATH_GPT_three_four_five_six_solution_l205_20544


namespace NUMINAMATH_GPT_chess_probability_l205_20571

theorem chess_probability (P_draw P_B_win : ℚ) (h_draw : P_draw = 1/2) (h_B_win : P_B_win = 1/3) :
  (1 - P_draw - P_B_win = 1/6) ∧ -- Statement A is correct
  (P_draw + (1 - P_draw - P_B_win) ≠ 1/2) ∧ -- Statement B is incorrect as it's not 1/2
  (1 - P_draw - P_B_win ≠ 2/3) ∧ -- Statement C is incorrect as it's not 2/3
  (P_draw + P_B_win ≠ 1/2) := -- Statement D is incorrect as it's not 1/2
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_chess_probability_l205_20571


namespace NUMINAMATH_GPT_min_value_of_x_under_conditions_l205_20558

noncomputable def S (x y z : ℝ) : ℝ := (z + 1)^2 / (2 * x * y * z)

theorem min_value_of_x_under_conditions :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x^2 + y^2 + z^2 = 1 →
  (∃ x_min : ℝ, S x y z = S x_min x_min (Real.sqrt 2 - 1) ∧ x_min = Real.sqrt (Real.sqrt 2 - 1)) :=
by
  intros x y z hx hy hz hxyz
  use Real.sqrt (Real.sqrt 2 - 1)
  sorry

end NUMINAMATH_GPT_min_value_of_x_under_conditions_l205_20558


namespace NUMINAMATH_GPT_water_glass_ounces_l205_20556

theorem water_glass_ounces (glasses_per_day : ℕ) (days_per_week : ℕ)
    (bottle_ounces : ℕ) (bottle_fills_per_week : ℕ)
    (total_glasses_per_week : ℕ)
    (total_ounces_per_week : ℕ)
    (glasses_per_week_eq : glasses_per_day * days_per_week = total_glasses_per_week)
    (ounces_per_week_eq : bottle_ounces * bottle_fills_per_week = total_ounces_per_week)
    (ounce_per_glass : ℕ)
    (glasses_per_week : ℕ)
    (ounces_per_week : ℕ) :
    total_ounces_per_week / total_glasses_per_week = 5 :=
by
  sorry

end NUMINAMATH_GPT_water_glass_ounces_l205_20556


namespace NUMINAMATH_GPT_square_tablecloth_side_length_l205_20516

theorem square_tablecloth_side_length (area : ℝ) (h : area = 5) : ∃ a : ℝ, a > 0 ∧ a * a = 5 := 
by
  use Real.sqrt 5
  constructor
  · apply Real.sqrt_pos.2; linarith
  · exact Real.mul_self_sqrt (by linarith [h])

end NUMINAMATH_GPT_square_tablecloth_side_length_l205_20516


namespace NUMINAMATH_GPT_music_commercials_ratio_l205_20597

theorem music_commercials_ratio (T C: ℕ) (hT: T = 112) (hC: C = 40) : (T - C) / C = 9 / 5 := by
  sorry

end NUMINAMATH_GPT_music_commercials_ratio_l205_20597
