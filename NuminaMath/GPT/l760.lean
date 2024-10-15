import Mathlib

namespace NUMINAMATH_GPT_find_Z_l760_76020

theorem find_Z (Z : ℝ) (h : (100 + 20 / Z) * Z = 9020) : Z = 90 :=
sorry

end NUMINAMATH_GPT_find_Z_l760_76020


namespace NUMINAMATH_GPT_carter_average_goals_l760_76084

theorem carter_average_goals (C : ℝ)
  (h1 : C + (1 / 2) * C + (C - 3) = 7) : C = 4 :=
by
  sorry

end NUMINAMATH_GPT_carter_average_goals_l760_76084


namespace NUMINAMATH_GPT_percent_uni_no_job_choice_l760_76091

variable (P_ND_JC P_JC P_UD P_U_NJC P_NJC : ℝ)
variable (h1 : P_ND_JC = 0.18)
variable (h2 : P_JC = 0.40)
variable (h3 : P_UD = 0.37)

theorem percent_uni_no_job_choice :
  (P_UD - (P_JC - P_ND_JC)) / (1 - P_JC) = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_percent_uni_no_job_choice_l760_76091


namespace NUMINAMATH_GPT_final_volume_of_syrup_l760_76071

-- Definitions based on conditions extracted from step a)
def quarts_to_cups (q : ℚ) : ℚ := q * 4
def reduce_volume (v : ℚ) : ℚ := v / 12
def add_sugar (v : ℚ) (s : ℚ) : ℚ := v + s

theorem final_volume_of_syrup :
  let initial_volume_in_quarts := 6
  let sugar_added := 1
  let initial_volume_in_cups := quarts_to_cups initial_volume_in_quarts
  let reduced_volume := reduce_volume initial_volume_in_cups
  add_sugar reduced_volume sugar_added = 3 :=
by
  sorry

end NUMINAMATH_GPT_final_volume_of_syrup_l760_76071


namespace NUMINAMATH_GPT_calculation_l760_76059

noncomputable def distance_from_sphere_center_to_plane (S P Q R : Point) (r PQ QR RP : ℝ) : ℝ := 
  let a := PQ / 2
  let b := QR / 2
  let c := RP / 2
  let s := (PQ + QR + RP) / 2
  let K := Real.sqrt (s * (s - PQ) * (s - QR) * (s - RP))
  let R := (PQ * QR * RP) / (4 * K)
  Real.sqrt (r^2 - R^2)

theorem calculation 
  (P Q R S : Point) 
  (r : ℝ) 
  (PQ QR RP : ℝ)
  (h1 : PQ = 17)
  (h2 : QR = 18)
  (h3 : RP = 19)
  (h4 : r = 25) :
  distance_from_sphere_center_to_plane S P Q R r PQ QR RP = 35 * Real.sqrt 7 / 8 → 
  ∃ (x y z : ℕ), x + y + z = 50 ∧ (x.gcd z = 1) ∧ ¬ ∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ y := 
by {
  sorry
}

end NUMINAMATH_GPT_calculation_l760_76059


namespace NUMINAMATH_GPT_solution_set_of_inequality_l760_76064

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_f1_zero : f 1 = 0) : 
  { x | f x > 0 } = { x | x < -1 ∨ 1 < x } := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l760_76064


namespace NUMINAMATH_GPT_pauly_cannot_make_more_omelets_l760_76068

-- Pauly's omelet data
def total_eggs : ℕ := 36
def plain_omelet_eggs : ℕ := 3
def cheese_omelet_eggs : ℕ := 4
def vegetable_omelet_eggs : ℕ := 5

-- Requested omelets
def requested_plain_omelets : ℕ := 4
def requested_cheese_omelets : ℕ := 2
def requested_vegetable_omelets : ℕ := 3

-- Number of eggs used for each type of requested omelet
def total_requested_eggs : ℕ :=
  (requested_plain_omelets * plain_omelet_eggs) +
  (requested_cheese_omelets * cheese_omelet_eggs) +
  (requested_vegetable_omelets * vegetable_omelet_eggs)

-- The remaining number of eggs
def remaining_eggs : ℕ := total_eggs - total_requested_eggs

theorem pauly_cannot_make_more_omelets :
  remaining_eggs < min plain_omelet_eggs (min cheese_omelet_eggs vegetable_omelet_eggs) :=
by
  sorry

end NUMINAMATH_GPT_pauly_cannot_make_more_omelets_l760_76068


namespace NUMINAMATH_GPT_difference_of_digits_l760_76028

theorem difference_of_digits (X Y : ℕ) (h1 : 10 * X + Y < 100) 
  (h2 : 72 = (10 * X + Y) - (10 * Y + X)) : (X - Y) = 8 :=
sorry

end NUMINAMATH_GPT_difference_of_digits_l760_76028


namespace NUMINAMATH_GPT_find_polynomial_l760_76038

noncomputable def polynomial_satisfies_conditions (P : Polynomial ℝ) : Prop :=
  P.eval 0 = 0 ∧ ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1

theorem find_polynomial (P : Polynomial ℝ) (h : polynomial_satisfies_conditions P) : P = Polynomial.X :=
  sorry

end NUMINAMATH_GPT_find_polynomial_l760_76038


namespace NUMINAMATH_GPT_apples_per_box_l760_76006

-- Defining the given conditions
variable (apples_per_crate : ℤ)
variable (number_of_crates : ℤ)
variable (rotten_apples : ℤ)
variable (number_of_boxes : ℤ)

-- Stating the facts based on given conditions
def total_apples := apples_per_crate * number_of_crates
def remaining_apples := total_apples - rotten_apples

-- The statement to prove
theorem apples_per_box 
    (hc1 : apples_per_crate = 180)
    (hc2 : number_of_crates = 12)
    (hc3 : rotten_apples = 160)
    (hc4 : number_of_boxes = 100) :
    (remaining_apples apples_per_crate number_of_crates rotten_apples) / number_of_boxes = 20 := 
sorry

end NUMINAMATH_GPT_apples_per_box_l760_76006


namespace NUMINAMATH_GPT_inequality_a_b_l760_76057

theorem inequality_a_b (a b : ℝ) (h : a > b ∧ b > 0) : (1/a) < (1/b) := 
by
  sorry

end NUMINAMATH_GPT_inequality_a_b_l760_76057


namespace NUMINAMATH_GPT_mustard_found_at_third_table_l760_76062

variable (a b T : ℝ)
def found_mustard_at_first_table := (a = 0.25)
def found_mustard_at_second_table := (b = 0.25)
def total_mustard_found := (T = 0.88)

theorem mustard_found_at_third_table
  (h1 : found_mustard_at_first_table a)
  (h2 : found_mustard_at_second_table b)
  (h3 : total_mustard_found T) :
  T - (a + b) = 0.38 := by
  sorry

end NUMINAMATH_GPT_mustard_found_at_third_table_l760_76062


namespace NUMINAMATH_GPT_average_chore_time_l760_76049

theorem average_chore_time 
  (times : List ℕ := [4, 3, 2, 1, 0])
  (counts : List ℕ := [2, 4, 2, 1, 1]) 
  (total_students : ℕ := 10)
  (total_time : ℕ := List.sum (List.zipWith (λ t c => t * c) times counts)) :
  (total_time : ℚ) / total_students = 2.5 := by
  sorry

end NUMINAMATH_GPT_average_chore_time_l760_76049


namespace NUMINAMATH_GPT_runway_show_duration_l760_76001

theorem runway_show_duration
  (evening_wear_time : ℝ) (bathing_suits_time : ℝ) (formal_wear_time : ℝ) (casual_wear_time : ℝ)
  (evening_wear_sets : ℕ) (bathing_suits_sets : ℕ) (formal_wear_sets : ℕ) (casual_wear_sets : ℕ)
  (num_models : ℕ) :
  evening_wear_time = 4 → bathing_suits_time = 2 → formal_wear_time = 3 → casual_wear_time = 2.5 →
  evening_wear_sets = 4 → bathing_suits_sets = 2 → formal_wear_sets = 3 → casual_wear_sets = 5 →
  num_models = 10 →
  (evening_wear_time * evening_wear_sets + bathing_suits_time * bathing_suits_sets
   + formal_wear_time * formal_wear_sets + casual_wear_time * casual_wear_sets) * num_models = 415 :=
by
  intros
  sorry

end NUMINAMATH_GPT_runway_show_duration_l760_76001


namespace NUMINAMATH_GPT_product_gcf_lcm_l760_76044

def gcf (a b c : Nat) : Nat := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : Nat) : Nat := Nat.lcm (Nat.lcm a b) c

theorem product_gcf_lcm :
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  A * B = 432 :=
by
  let A := gcf 6 18 24
  let B := lcm 6 18 24
  have hA : A = Nat.gcd (Nat.gcd 6 18) 24 := rfl
  have hB : B = Nat.lcm (Nat.lcm 6 18) 24 := rfl
  sorry

end NUMINAMATH_GPT_product_gcf_lcm_l760_76044


namespace NUMINAMATH_GPT_bowling_ball_weight_l760_76004

variable {b c : ℝ}

theorem bowling_ball_weight :
  (10 * b = 4 * c) ∧ (3 * c = 108) → b = 14.4 :=
by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l760_76004


namespace NUMINAMATH_GPT_combine_heaps_l760_76060

def heaps_similar (x y : ℕ) : Prop :=
  x ≤ 2 * y ∧ y ≤ 2 * x

theorem combine_heaps (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
  f 0 = n ∧
  ∀ k, k < n → (∃ i j, i + j = k ∧ heaps_similar (f i) (f j)) ∧ 
  (∃ k, f k = n) :=
by
  sorry

end NUMINAMATH_GPT_combine_heaps_l760_76060


namespace NUMINAMATH_GPT_volume_ratio_spheres_l760_76096

theorem volume_ratio_spheres (r1 r2 r3 v1 v2 v3 : ℕ)
  (h_rad_ratio : r1 = 1 ∧ r2 = 2 ∧ r3 = 3)
  (h_vol_ratio : v1 = r1^3 ∧ v2 = r2^3 ∧ v3 = r3^3) :
  v3 = 3 * (v1 + v2) := by
  -- main proof goes here
  sorry

end NUMINAMATH_GPT_volume_ratio_spheres_l760_76096


namespace NUMINAMATH_GPT_find_certain_number_l760_76036

-- Define the given operation a # b
def sOperation (a b : ℝ) : ℝ :=
  a * b - b + b^2

-- State the theorem to find the value of the certain number
theorem find_certain_number (x : ℝ) (h : sOperation 3 x = 48) : x = 6 :=
sorry

end NUMINAMATH_GPT_find_certain_number_l760_76036


namespace NUMINAMATH_GPT_smallest_prime_dividing_7pow15_plus_9pow17_l760_76085

theorem smallest_prime_dividing_7pow15_plus_9pow17 :
  Nat.Prime 2 ∧ (∀ p : ℕ, Nat.Prime p → p ∣ (7^15 + 9^17) → 2 ≤ p) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_dividing_7pow15_plus_9pow17_l760_76085


namespace NUMINAMATH_GPT_num_factors_48_l760_76056

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end NUMINAMATH_GPT_num_factors_48_l760_76056


namespace NUMINAMATH_GPT_y_intercept_of_line_l760_76015

theorem y_intercept_of_line : ∀ (x y : ℝ), (5 * x - 2 * y - 10 = 0) → (x = 0) → (y = -5) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l760_76015


namespace NUMINAMATH_GPT_fraction_calculation_l760_76034

noncomputable def improper_frac_1 : ℚ := 21 / 8
noncomputable def improper_frac_2 : ℚ := 33 / 14
noncomputable def improper_frac_3 : ℚ := 37 / 12
noncomputable def improper_frac_4 : ℚ := 35 / 8
noncomputable def improper_frac_5 : ℚ := 179 / 9

theorem fraction_calculation :
  (improper_frac_1 - (2 / 3) * improper_frac_2) / ((improper_frac_3 + improper_frac_4) / improper_frac_5) = 59 / 21 :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l760_76034


namespace NUMINAMATH_GPT_other_equation_l760_76070

-- Define the variables for the length of the rope and the depth of the well
variables (x y : ℝ)

-- Given condition
def cond1 : Prop := (1/4) * x = y + 3

-- The proof goal
theorem other_equation (h : cond1 x y) : (1/5) * x = y + 2 :=
sorry

end NUMINAMATH_GPT_other_equation_l760_76070


namespace NUMINAMATH_GPT_Suma_work_time_l760_76086

theorem Suma_work_time (W : ℝ) (h1 : W > 0) :
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  suma_time = 8 :=
by 
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  exact sorry

end NUMINAMATH_GPT_Suma_work_time_l760_76086


namespace NUMINAMATH_GPT_smallest_prime_perimeter_l760_76053

def is_prime (n : ℕ) := Nat.Prime n
def is_triangle (a b c : ℕ) := a + b > c ∧ a + c > b ∧ b + c > a
def is_scalene (a b c : ℕ) := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_scalene a b c ∧ a ≥ 5
  ∧ is_prime (a + b + c) ∧ a + b + c = 23 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_perimeter_l760_76053


namespace NUMINAMATH_GPT_roots_quadratic_l760_76002

theorem roots_quadratic (m x₁ x₂ : ℝ) (h : m < 0) (h₁ : x₁ < x₂) (hx : ∀ x, (x^2 - x - 6 = m) ↔ (x = x₁ ∨ x = x₂)) : 
  -2 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_roots_quadratic_l760_76002


namespace NUMINAMATH_GPT_positive_integers_satisfying_inequality_l760_76067

-- Define the assertion that there are exactly 5 positive integers x satisfying the given inequality
theorem positive_integers_satisfying_inequality :
  (∃! x : ℕ, 4 < x ∧ x < 10 ∧ (10 * x)^4 > x^8 ∧ x^8 > 2^16) :=
sorry

end NUMINAMATH_GPT_positive_integers_satisfying_inequality_l760_76067


namespace NUMINAMATH_GPT_no_solution_exists_l760_76099

theorem no_solution_exists (x y z : ℕ) (hx : x > 2) (hy : y > 1) (h : x^y + 1 = z^2) : false := 
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l760_76099


namespace NUMINAMATH_GPT_fraction_spent_by_Rica_is_one_fifth_l760_76024

-- Define the conditions
variable (totalPrizeMoney : ℝ) (fractionReceived : ℝ) (amountLeft : ℝ)
variable (h1 : totalPrizeMoney = 1000) (h2 : fractionReceived = 3 / 8) (h3 : amountLeft = 300)

-- Define Rica's original prize money
noncomputable def RicaOriginalPrizeMoney (totalPrizeMoney fractionReceived : ℝ) : ℝ :=
  fractionReceived * totalPrizeMoney

-- Define amount spent by Rica
noncomputable def AmountSpent (originalPrizeMoney amountLeft : ℝ) : ℝ :=
  originalPrizeMoney - amountLeft

-- Define the fraction of prize money spent by Rica
noncomputable def FractionSpent (amountSpent originalPrizeMoney : ℝ) : ℝ :=
  amountSpent / originalPrizeMoney

-- Main theorem to prove
theorem fraction_spent_by_Rica_is_one_fifth :
  let totalPrizeMoney := 1000
  let fractionReceived := 3 / 8
  let amountLeft := 300
  let RicaOriginalPrizeMoney := fractionReceived * totalPrizeMoney
  let AmountSpent := RicaOriginalPrizeMoney - amountLeft
  let FractionSpent := AmountSpent / RicaOriginalPrizeMoney
  FractionSpent = 1 / 5 :=
by {
  -- Proof details are omitted as per instructions
  sorry
}

end NUMINAMATH_GPT_fraction_spent_by_Rica_is_one_fifth_l760_76024


namespace NUMINAMATH_GPT_gcd_lcm_product_l760_76025

theorem gcd_lcm_product (a b : ℤ) (h1 : Int.gcd a b = 8) (h2 : Int.lcm a b = 24) : a * b = 192 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l760_76025


namespace NUMINAMATH_GPT_parabola_focus_and_directrix_l760_76092

theorem parabola_focus_and_directrix :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ a b : ℝ, (a, b) = (0, 1) ∧ y = -1) :=
by
  -- Here, we would provide definitions and logical steps if we were completing the proof.
  -- For now, we will leave it unfinished.
  sorry

end NUMINAMATH_GPT_parabola_focus_and_directrix_l760_76092


namespace NUMINAMATH_GPT_total_votes_l760_76040

variable (V : ℝ)

theorem total_votes (h : 0.70 * V - 0.30 * V = 160) : V = 400 := by
  sorry

end NUMINAMATH_GPT_total_votes_l760_76040


namespace NUMINAMATH_GPT_simplify_fraction_l760_76032

theorem simplify_fraction : (2 / 520) + (23 / 40) = 301 / 520 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l760_76032


namespace NUMINAMATH_GPT_geese_flew_away_l760_76027

theorem geese_flew_away (initial remaining flown_away : ℕ) (h_initial: initial = 51) (h_remaining: remaining = 23) : flown_away = 28 :=
by
  sorry

end NUMINAMATH_GPT_geese_flew_away_l760_76027


namespace NUMINAMATH_GPT_remainder_3n_mod_7_l760_76035

theorem remainder_3n_mod_7 (n : ℤ) (k : ℤ) (h : n = 7*k + 1) :
  (3 * n) % 7 = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_3n_mod_7_l760_76035


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l760_76023

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a^2 ≠ 4) → (a ≠ 2) ∧ ¬ ((a ≠ 2) → (a^2 ≠ 4)) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l760_76023


namespace NUMINAMATH_GPT_additional_amount_needed_l760_76093

-- Definitions of the conditions
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost : ℝ := 6.00
def lotions_count : ℕ := 3
def free_shipping_threshold : ℝ := 50.00

-- Calculating the total amount spent
def total_spent : ℝ :=
  shampoo_cost + conditioner_cost + lotions_count * lotion_cost

-- Required statement for the proof
theorem additional_amount_needed : 
  total_spent + 12.00 = free_shipping_threshold :=
by 
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_additional_amount_needed_l760_76093


namespace NUMINAMATH_GPT_claire_earnings_l760_76061

theorem claire_earnings
  (total_flowers : ℕ)
  (tulips : ℕ)
  (white_roses : ℕ)
  (price_per_red_rose : ℚ)
  (sell_fraction : ℚ)
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : price_per_red_rose = 0.75)
  (h5 : sell_fraction = 1/2) : 
  (total_flowers - tulips - white_roses) * sell_fraction * price_per_red_rose = 75 :=
by
  sorry

end NUMINAMATH_GPT_claire_earnings_l760_76061


namespace NUMINAMATH_GPT_minimum_value_of_f_l760_76073

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / x

theorem minimum_value_of_f (h : 1 < x) : ∃ y, f x = y ∧ (∀ z, (f z) ≥ 2*sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l760_76073


namespace NUMINAMATH_GPT_remainder_when_squared_l760_76050

theorem remainder_when_squared (n : ℕ) (h : n % 8 = 6) : (n * n) % 32 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_when_squared_l760_76050


namespace NUMINAMATH_GPT_teacher_age_frequency_l760_76081

theorem teacher_age_frequency (f_less_than_30 : ℝ) (f_between_30_and_50 : ℝ) (h1 : f_less_than_30 = 0.3) (h2 : f_between_30_and_50 = 0.5) :
  1 - f_less_than_30 - f_between_30_and_50 = 0.2 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_teacher_age_frequency_l760_76081


namespace NUMINAMATH_GPT_min_value_of_z_l760_76021

theorem min_value_of_z (a x y : ℝ) (h1 : a > 0) (h2 : x ≥ 1) (h3 : x + y ≤ 3) (h4 : y ≥ a * (x - 3)) :
  (∃ (x y : ℝ), 2 * x + y = 1) → a = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_of_z_l760_76021


namespace NUMINAMATH_GPT_total_blocks_per_day_l760_76063

def blocks_to_park : ℕ := 4
def blocks_to_hs : ℕ := 7
def blocks_to_home : ℕ := 11
def walks_per_day : ℕ := 3

theorem total_blocks_per_day :
  (blocks_to_park + blocks_to_hs + blocks_to_home) * walks_per_day = 66 :=
by
  sorry

end NUMINAMATH_GPT_total_blocks_per_day_l760_76063


namespace NUMINAMATH_GPT_tap_B_filling_time_l760_76087

theorem tap_B_filling_time : 
  ∀ (r_A r_B : ℝ), 
  (r_A + r_B = 1 / 30) → 
  (r_B * 40 = 2 / 3) → 
  (1 / r_B = 60) := 
by
  intros r_A r_B h₁ h₂
  sorry

end NUMINAMATH_GPT_tap_B_filling_time_l760_76087


namespace NUMINAMATH_GPT_find_line_equation_proj_origin_l760_76047

theorem find_line_equation_proj_origin (P : ℝ × ℝ) (hP : P = (-2, 1)) :
    ∃ (a b c : ℝ), a * 2 + b * (-1) + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_line_equation_proj_origin_l760_76047


namespace NUMINAMATH_GPT_find_number_of_hens_l760_76029

theorem find_number_of_hens
  (H C : ℕ)
  (h1 : H + C = 48)
  (h2 : 2 * H + 4 * C = 140) :
  H = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_hens_l760_76029


namespace NUMINAMATH_GPT_original_price_doubled_l760_76003

variable (P : ℝ)

-- Given condition: Original price plus 20% equals 351
def price_increased (P : ℝ) : Prop :=
  P + 0.20 * P = 351

-- The goal is to prove that 2 times the original price is 585
theorem original_price_doubled (P : ℝ) (h : price_increased P) : 2 * P = 585 :=
sorry

end NUMINAMATH_GPT_original_price_doubled_l760_76003


namespace NUMINAMATH_GPT_range_m_l760_76039

def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

noncomputable def problem :=
  ∀ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

theorem range_m (m : ℝ) : problem := 
  sorry

end NUMINAMATH_GPT_range_m_l760_76039


namespace NUMINAMATH_GPT_quadratic_eq_has_nonzero_root_l760_76011

theorem quadratic_eq_has_nonzero_root (b c : ℝ) (h : c ≠ 0) (h_eq : c^2 + b * c + c = 0) : b + c = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_eq_has_nonzero_root_l760_76011


namespace NUMINAMATH_GPT_union_sets_l760_76022

def M (a : ℕ) : Set ℕ := {a, 0}
def N : Set ℕ := {1, 2}

theorem union_sets (a : ℕ) (h_inter : M a ∩ N = {2}) : M a ∪ N = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l760_76022


namespace NUMINAMATH_GPT_people_stools_chairs_l760_76026

def numberOfPeopleStoolsAndChairs (x y z : ℕ) : Prop :=
  2 * x + 3 * y + 4 * z = 32 ∧
  x > y ∧
  x > z ∧
  x < y + z

theorem people_stools_chairs :
  ∃ (x y z : ℕ), numberOfPeopleStoolsAndChairs x y z ∧ x = 5 ∧ y = 2 ∧ z = 4 :=
by
  sorry

end NUMINAMATH_GPT_people_stools_chairs_l760_76026


namespace NUMINAMATH_GPT_tangent_line_eq_l760_76078

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 - 2 * x - 1)

theorem tangent_line_eq :
  let x := 1
  let y := f x
  ∃ (m : ℝ), m = -2 * Real.exp 1 ∧ (∀ (x y : ℝ), y = m * (x - 1) + f 1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l760_76078


namespace NUMINAMATH_GPT_speed_of_jakes_dad_second_half_l760_76000

theorem speed_of_jakes_dad_second_half :
  let distance_to_park := 22
  let total_time := 0.5
  let time_half_journey := total_time / 2
  let speed_first_half := 28
  let distance_first_half := speed_first_half * time_half_journey
  let remaining_distance := distance_to_park - distance_first_half
  let time_second_half := time_half_journey
  let speed_second_half := remaining_distance / time_second_half
  speed_second_half = 60 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_jakes_dad_second_half_l760_76000


namespace NUMINAMATH_GPT_range_of_a_l760_76046

theorem range_of_a (x y : ℝ) (a : ℝ) :
  (0 < x ∧ x ≤ 2) ∧ (0 < y ∧ y ≤ 2) ∧ (x * y = 2) ∧ (6 - 2 * x - y ≥ a * (2 - x) * (4 - y)) →
  a ≤ 1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l760_76046


namespace NUMINAMATH_GPT_find_length_of_field_l760_76043

variables (L : ℝ) -- Length of the field
variables (width_field : ℝ := 55) -- Width of the field, given as 55 meters.
variables (width_path : ℝ := 2.5) -- Width of the path around the field, given as 2.5 meters.
variables (area_path : ℝ := 1200) -- Area of the path, given as 1200 square meters.

theorem find_length_of_field
  (h : area_path = (L + 2 * width_path) * (width_field + 2 * width_path) - L * width_field)
  : L = 180 :=
by sorry

end NUMINAMATH_GPT_find_length_of_field_l760_76043


namespace NUMINAMATH_GPT_least_four_digit_9_heavy_l760_76042

def is_9_heavy (n : ℕ) : Prop := n % 9 > 5

def four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem least_four_digit_9_heavy : ∃ n, four_digit n ∧ is_9_heavy n ∧ ∀ m, (four_digit m ∧ is_9_heavy m) → n ≤ m :=
by
  exists 1005
  sorry

end NUMINAMATH_GPT_least_four_digit_9_heavy_l760_76042


namespace NUMINAMATH_GPT_max_gcd_14m_plus_4_9m_plus_2_l760_76095

theorem max_gcd_14m_plus_4_9m_plus_2 (m : ℕ) (h : m > 0) : ∃ M, M = 8 ∧ ∀ k, gcd (14 * m + 4) (9 * m + 2) = k → k ≤ M :=
by
  sorry

end NUMINAMATH_GPT_max_gcd_14m_plus_4_9m_plus_2_l760_76095


namespace NUMINAMATH_GPT_negation_of_exists_sin_gt_one_l760_76033

theorem negation_of_exists_sin_gt_one : 
  (¬ ∃ x : ℝ, Real.sin x > 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := 
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_sin_gt_one_l760_76033


namespace NUMINAMATH_GPT_evaluate_expression_l760_76097

theorem evaluate_expression :
  -25 + 7 * ((8 / 4) ^ 2) = 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l760_76097


namespace NUMINAMATH_GPT_sum_of_three_largest_ge_50_l760_76005

theorem sum_of_three_largest_ge_50 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℕ) :
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧ a₁ ≠ a₇ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧ a₂ ≠ a₇ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧ a₃ ≠ a₇ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧ a₄ ≠ a₇ ∧
  a₅ ≠ a₆ ∧ a₅ ≠ a₇ ∧
  a₆ ≠ a₇ ∧
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0 ∧ a₅ > 0 ∧ a₆ > 0 ∧ a₇ > 0 ∧
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 100 →
  ∃ (x y z : ℕ), (x ≠ y ∧ x ≠ z ∧ y ≠ z) ∧ (x > 0 ∧ y > 0 ∧ z > 0) ∧ (x + y + z ≥ 50) :=
by sorry

end NUMINAMATH_GPT_sum_of_three_largest_ge_50_l760_76005


namespace NUMINAMATH_GPT_car_distance_l760_76054

variable (v_x v_y : ℝ) (Δt_x : ℝ) (d_x : ℝ)

theorem car_distance (h_vx : v_x = 35) (h_vy : v_y = 50) (h_Δt : Δt_x = 1.2)
  (h_dx : d_x = v_x * Δt_x):
  d_x + v_x * (d_x / (v_y - v_x)) = 98 := 
by sorry

end NUMINAMATH_GPT_car_distance_l760_76054


namespace NUMINAMATH_GPT_monotonicity_and_range_l760_76072

noncomputable def f (a x : ℝ) : ℝ := (a * x - 2) * Real.exp x - Real.exp (a - 2)

theorem monotonicity_and_range (a x : ℝ) :
  ( (a = 0 → ∀ x, f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x < (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x > (2 - a) / a, f a x > f a (x + 1) ) ∧
  (a < 0 → ∀ x > (2 - a) / a, f a x < f a (x + 1) ∧ ∀ x < (2 - a) / a, f a x > f a (x + 1) ) ∧
  (∀ x > 1, f a x > 0 ↔ a ∈ Set.Ici 1)) 
:=
sorry

end NUMINAMATH_GPT_monotonicity_and_range_l760_76072


namespace NUMINAMATH_GPT_cone_base_diameter_l760_76030

theorem cone_base_diameter {r l : ℝ} 
  (h₁ : π * r * l + π * r^2 = 3 * π) 
  (h₂ : 2 * π * r = π * l) : 
  2 * r = 2 :=
by
  sorry

end NUMINAMATH_GPT_cone_base_diameter_l760_76030


namespace NUMINAMATH_GPT_employee_earnings_l760_76098

theorem employee_earnings (regular_rate overtime_rate first3_days_h second2_days_h total_hours overtime_hours : ℕ)
  (h1 : regular_rate = 30)
  (h2 : overtime_rate = 45)
  (h3 : first3_days_h = 6)
  (h4 : second2_days_h = 12)
  (h5 : total_hours = first3_days_h * 3 + second2_days_h * 2)
  (h6 : total_hours = 42)
  (h7 : overtime_hours = total_hours - 40)
  (h8 : overtime_hours = 2) :
  (40 * regular_rate + overtime_hours * overtime_rate) = 1290 := 
sorry

end NUMINAMATH_GPT_employee_earnings_l760_76098


namespace NUMINAMATH_GPT_Jan_older_than_Cindy_l760_76007

noncomputable def Cindy_age : ℕ := 5
noncomputable def Greg_age : ℕ := 16

variables (Marcia_age Jan_age : ℕ)

axiom Greg_and_Marcia : Greg_age = Marcia_age + 2
axiom Marcia_and_Jan : Marcia_age = 2 * Jan_age

theorem Jan_older_than_Cindy : (Jan_age - Cindy_age) = 2 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_Jan_older_than_Cindy_l760_76007


namespace NUMINAMATH_GPT_ellipse_foci_y_axis_range_l760_76079

theorem ellipse_foci_y_axis_range (k : ℝ) :
  (∃ x y : ℝ, x^2 + k * y^2 = 4 ∧ (∃ c1 c2 : ℝ, y = 0 → c1^2 + c2^2 = 4)) ↔ 0 < k ∧ k < 1 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_y_axis_range_l760_76079


namespace NUMINAMATH_GPT_max_objective_function_value_l760_76076

def objective_function (x1 x2 : ℝ) := 4 * x1 + 6 * x2

theorem max_objective_function_value :
  ∃ x1 x2 : ℝ, 
    (x1 >= 0) ∧ 
    (x2 >= 0) ∧ 
    (x1 + x2 <= 18) ∧ 
    (0.5 * x1 + x2 <= 12) ∧ 
    (2 * x1 <= 24) ∧ 
    (2 * x2 <= 18) ∧ 
    (∀ y1 y2 : ℝ, 
      (y1 >= 0) ∧ 
      (y2 >= 0) ∧ 
      (y1 + y2 <= 18) ∧ 
      (0.5 * y1 + y2 <= 12) ∧ 
      (2 * y1 <= 24) ∧ 
      (2 * y2 <= 18) -> 
      objective_function y1 y2 <= objective_function x1 x2) ∧
    (objective_function x1 x2 = 84) :=
by
  use 12, 6
  sorry

end NUMINAMATH_GPT_max_objective_function_value_l760_76076


namespace NUMINAMATH_GPT_domain_f_l760_76083

noncomputable def f (x : ℝ) := Real.sqrt (3 - x) + Real.log (x - 1)

theorem domain_f : { x : ℝ | 1 < x ∧ x ≤ 3 } = { x : ℝ | True } ∩ { x : ℝ | x ≤ 3 } ∩ { x : ℝ | x > 1 } :=
by
  sorry

end NUMINAMATH_GPT_domain_f_l760_76083


namespace NUMINAMATH_GPT_find_smallest_number_l760_76051

theorem find_smallest_number 
  : ∃ x : ℕ, (x - 18) % 14 = 0 ∧ (x - 18) % 26 = 0 ∧ (x - 18) % 28 = 0 ∧ (x - 18) / Nat.lcm 14 (Nat.lcm 26 28) = 746 ∧ x = 271562 := by
  sorry

end NUMINAMATH_GPT_find_smallest_number_l760_76051


namespace NUMINAMATH_GPT_locus_of_midpoint_l760_76017

theorem locus_of_midpoint (x y : ℝ) (h : y ≠ 0) :
  (∃ P : ℝ × ℝ, P = (2*x, 2*y) ∧ ((P.1^2 + (P.2-3)^2 = 9))) →
  (x^2 + (y - 3/2)^2 = 9/4) :=
by
  sorry

end NUMINAMATH_GPT_locus_of_midpoint_l760_76017


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_product_336_l760_76052

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y z : ℤ), x * y * z = 336 ∧ x + 1 = y ∧ y + 1 = z ∧ x + y + z = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_product_336_l760_76052


namespace NUMINAMATH_GPT_edges_sum_l760_76031

def edges_triangular_pyramid : ℕ := 6
def edges_triangular_prism : ℕ := 9

theorem edges_sum : edges_triangular_pyramid + edges_triangular_prism = 15 :=
by
  sorry

end NUMINAMATH_GPT_edges_sum_l760_76031


namespace NUMINAMATH_GPT_problem_not_equivalent_l760_76074

theorem problem_not_equivalent :
  (0.0000396 ≠ 3.9 * 10^(-5)) ∧ 
  (0.0000396 = 3.96 * 10^(-5)) ∧ 
  (0.0000396 = 396 * 10^(-7)) ∧ 
  (0.0000396 = (793 / 20000) * 10^(-5)) ∧ 
  (0.0000396 = 198 / 5000000) :=
by
  sorry

end NUMINAMATH_GPT_problem_not_equivalent_l760_76074


namespace NUMINAMATH_GPT_smallest_positive_angle_l760_76018

theorem smallest_positive_angle (deg : ℤ) (k : ℤ) (h : deg = -2012) : ∃ m : ℤ, m = 148 ∧ 0 ≤ m ∧ m < 360 ∧ (∃ n : ℤ, deg + 360 * n = m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_l760_76018


namespace NUMINAMATH_GPT_x_squared_eq_1_iff_x_eq_1_l760_76041

theorem x_squared_eq_1_iff_x_eq_1 (x : ℝ) : (x^2 = 1 → x = 1) ↔ false ∧ (x = 1 → x^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_x_squared_eq_1_iff_x_eq_1_l760_76041


namespace NUMINAMATH_GPT_find_equation_with_new_roots_l760_76012

variable {p q r s : ℝ}

theorem find_equation_with_new_roots 
  (h_eq : ∀ x, x^2 - p * x + q = 0 ↔ (x = r ∧ x = s))
  (h_r_nonzero : r ≠ 0)
  (h_s_nonzero : s ≠ 0)
  : 
  ∀ x, (x^2 - ((q^2 + 1) * (p^2 - 2 * q) / q^2) * x + (q + 1/q)^2) = 0 ↔ 
       (x = r^2 + 1/(s^2) ∧ x = s^2 + 1/(r^2)) := 
sorry

end NUMINAMATH_GPT_find_equation_with_new_roots_l760_76012


namespace NUMINAMATH_GPT_three_colors_sufficient_l760_76089

-- Definition of the tessellation problem with specified conditions.
def tessellation (n : ℕ) (x_divisions : ℕ) (y_divisions : ℕ) : Prop :=
  n = 8 ∧ x_divisions = 2 ∧ y_divisions = 2

-- Definition of the adjacency property.
def no_adjacent_same_color {α : Type} (coloring : ℕ → ℕ → α) : Prop :=
  ∀ (i j : ℕ), i < 8 → j < 8 →
  (i > 0 → coloring i j ≠ coloring (i-1) j) ∧ 
  (j > 0 → coloring i j ≠ coloring i (j-1)) ∧
  (i < 7 → coloring i j ≠ coloring (i+1) j) ∧ 
  (j < 7 → coloring i j ≠ coloring i (j+1)) ∧
  (i > 0 ∧ j > 0 → coloring i j ≠ coloring (i-1) (j-1)) ∧
  (i < 7 ∧ j < 7 → coloring i j ≠ coloring (i+1) (j+1)) ∧
  (i > 0 ∧ j < 7 → coloring i j ≠ coloring (i-1) (j+1)) ∧
  (i < 7 ∧ j > 0 → coloring i j ≠ coloring (i+1) (j-1))

-- The main theorem that needs to be proved.
theorem three_colors_sufficient : ∃ (k : ℕ) (coloring : ℕ → ℕ → ℕ), k = 3 ∧ 
  tessellation 8 2 2 ∧ 
  no_adjacent_same_color coloring := by
  sorry 

end NUMINAMATH_GPT_three_colors_sufficient_l760_76089


namespace NUMINAMATH_GPT_impossible_ratio_5_11_l760_76082

theorem impossible_ratio_5_11:
  ∀ (b g: ℕ), 
  b + g ≥ 66 →
  b + 11 = g - 13 →
  ¬(5 * b = 11 * (b + 24) ∧ b ≥ 21) := 
by
  intros b g h1 h2 h3
  sorry

end NUMINAMATH_GPT_impossible_ratio_5_11_l760_76082


namespace NUMINAMATH_GPT_largest_fraction_of_consecutive_odds_is_three_l760_76037

theorem largest_fraction_of_consecutive_odds_is_three
  (p q r s : ℕ)
  (h1 : 0 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h_odd1 : p % 2 = 1)
  (h_odd2 : q % 2 = 1)
  (h_odd3 : r % 2 = 1)
  (h_odd4 : s % 2 = 1)
  (h_consecutive1 : q = p + 2)
  (h_consecutive2 : r = q + 2)
  (h_consecutive3 : s = r + 2) :
  (r + s) / (p + q) = 3 :=
sorry

end NUMINAMATH_GPT_largest_fraction_of_consecutive_odds_is_three_l760_76037


namespace NUMINAMATH_GPT_original_price_of_color_TV_l760_76088

theorem original_price_of_color_TV
  (x : ℝ)  -- Let the variable x represent the original price
  (h1 : x * 1.4 * 0.8 - x = 144)  -- Condition as equation
  : x = 1200 := 
sorry  -- Proof to be filled in later

end NUMINAMATH_GPT_original_price_of_color_TV_l760_76088


namespace NUMINAMATH_GPT_composite_a2_b2_l760_76045

-- Introduce the main definitions according to the conditions stated in a)
theorem composite_a2_b2 (x1 x2 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (a b : ℤ) 
  (ha : a = -(x1 + x2)) (hb : b = x1 * x2 - 1) : 
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ (a^2 + b^2) = m * n := 
by 
  sorry

end NUMINAMATH_GPT_composite_a2_b2_l760_76045


namespace NUMINAMATH_GPT_unique_zero_of_f_l760_76016

theorem unique_zero_of_f (f : ℝ → ℝ) (h1 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 16) 
  (h2 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 8) (h3 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 4) 
  (h4 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 2) : ¬ ∃ x, f x = 0 ∧ 2 ≤ x ∧ x < 16 := 
by
  sorry

end NUMINAMATH_GPT_unique_zero_of_f_l760_76016


namespace NUMINAMATH_GPT_wheels_on_floor_l760_76066

def number_of_wheels (n_people : Nat) (w_per_person : Nat) : Nat :=
  n_people * w_per_person

theorem wheels_on_floor (n_people : Nat) (w_per_person : Nat) (h_people : n_people = 40) (h_wheels : w_per_person = 4) :
  number_of_wheels n_people w_per_person = 160 := by
  sorry

end NUMINAMATH_GPT_wheels_on_floor_l760_76066


namespace NUMINAMATH_GPT_product_not_50_l760_76077

theorem product_not_50 :
  (1 / 2 * 100 = 50) ∧
  (-5 * -10 = 50) ∧
  ¬(5 * 11 = 50) ∧
  (2 * 25 = 50) ∧
  (5 / 2 * 20 = 50) :=
by
  sorry

end NUMINAMATH_GPT_product_not_50_l760_76077


namespace NUMINAMATH_GPT_inequality_solution_set_l760_76055

theorem inequality_solution_set : 
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l760_76055


namespace NUMINAMATH_GPT_find_a_10_l760_76075

def seq (a : ℕ → ℚ) : Prop :=
∀ n, a (n + 1) = 2 * a n / (a n + 2)

def initial_value (a : ℕ → ℚ) : Prop :=
a 1 = 1

theorem find_a_10 (a : ℕ → ℚ) (h1 : initial_value a) (h2 : seq a) : 
  a 10 = 2 / 11 := 
sorry

end NUMINAMATH_GPT_find_a_10_l760_76075


namespace NUMINAMATH_GPT_jennifer_remaining_money_l760_76069

noncomputable def money_spent_on_sandwich (initial_money : ℝ) : ℝ :=
  let sandwich_cost := (1/5) * initial_money
  let discount := (10/100) * sandwich_cost
  sandwich_cost - discount

noncomputable def money_spent_on_ticket (initial_money : ℝ) : ℝ :=
  (1/6) * initial_money

noncomputable def money_spent_on_book (initial_money : ℝ) : ℝ :=
  (1/2) * initial_money

noncomputable def money_after_initial_expenses (initial_money : ℝ) (gift : ℝ) : ℝ :=
  initial_money - money_spent_on_sandwich initial_money - money_spent_on_ticket initial_money - money_spent_on_book initial_money + gift

noncomputable def money_spent_on_cosmetics (remaining_money : ℝ) : ℝ :=
  (1/4) * remaining_money

noncomputable def money_after_cosmetics (remaining_money : ℝ) : ℝ :=
  remaining_money - money_spent_on_cosmetics remaining_money

noncomputable def money_spent_on_tshirt (remaining_money : ℝ) : ℝ :=
  let tshirt_cost := (1/3) * remaining_money
  let tax := (5/100) * tshirt_cost
  tshirt_cost + tax

noncomputable def remaining_money (initial_money : ℝ) (gift : ℝ) : ℝ :=
  let after_initial := money_after_initial_expenses initial_money gift
  let after_cosmetics := after_initial - money_spent_on_cosmetics after_initial
  after_cosmetics - money_spent_on_tshirt after_cosmetics

theorem jennifer_remaining_money : remaining_money 90 30 = 21.35 := by
  sorry

end NUMINAMATH_GPT_jennifer_remaining_money_l760_76069


namespace NUMINAMATH_GPT_num_distinct_prime_factors_330_l760_76019

theorem num_distinct_prime_factors_330 : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, Nat.Prime x ∧ 330 % x = 0 := 
sorry

end NUMINAMATH_GPT_num_distinct_prime_factors_330_l760_76019


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l760_76094

theorem arithmetic_sequence_sum :
  ∀ (x y : ℤ), (∃ (n m : ℕ), (3 + n * 6 = x) ∧ (3 + m * 6 = y) ∧ x + 6 = y ∧ y + 6 = 33) → x + y = 60 :=
by
  intro x y h
  obtain ⟨n, m, hn, hm, hx, hy⟩ := h
  exact sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l760_76094


namespace NUMINAMATH_GPT_two_pairs_of_dice_probability_l760_76058

noncomputable def two_pairs_probability : ℚ :=
  5 / 36

theorem two_pairs_of_dice_probability :
  ∃ p : ℚ, p = two_pairs_probability := 
by 
  use 5 / 36
  sorry

end NUMINAMATH_GPT_two_pairs_of_dice_probability_l760_76058


namespace NUMINAMATH_GPT_num_administrative_personnel_l760_76010

noncomputable def total_employees : ℕ := 280
noncomputable def sample_size : ℕ := 56
noncomputable def ordinary_staff_sample : ℕ := 49

theorem num_administrative_personnel (n : ℕ) (h1 : total_employees = 280) 
(h2 : sample_size = 56) (h3 : ordinary_staff_sample = 49) : 
n = 35 := 
by
  have h_proportion : (sample_size - ordinary_staff_sample) / sample_size = n / total_employees := by sorry
  have h_sol : n = (sample_size - ordinary_staff_sample) * (total_employees / sample_size) := by sorry
  have h_n : n = 35 := by sorry
  exact h_n

end NUMINAMATH_GPT_num_administrative_personnel_l760_76010


namespace NUMINAMATH_GPT_bricks_required_l760_76065

theorem bricks_required (L_courtyard W_courtyard L_brick W_brick : Real)
  (hcourtyard : L_courtyard = 35) 
  (wcourtyard : W_courtyard = 24) 
  (hbrick_len : L_brick = 0.15) 
  (hbrick_wid : W_brick = 0.08) : 
  (L_courtyard * W_courtyard) / (L_brick * W_brick) = 70000 := 
by
  sorry

end NUMINAMATH_GPT_bricks_required_l760_76065


namespace NUMINAMATH_GPT_exam_correct_answers_count_l760_76014

theorem exam_correct_answers_count (x y : ℕ) (h1 : x + y = 80) (h2 : 4 * x - y = 130) : x = 42 :=
by {
  -- (proof to be completed later)
  sorry
}

end NUMINAMATH_GPT_exam_correct_answers_count_l760_76014


namespace NUMINAMATH_GPT_cyclic_quadrilateral_angles_l760_76009

theorem cyclic_quadrilateral_angles (ABCD_cyclic : True) (P_interior : True)
  (x y z t : ℝ) (h1 : x + y + z + t = 360)
  (h2 : x + t = 180) :
  x = 180 - y - z :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_angles_l760_76009


namespace NUMINAMATH_GPT_basketball_students_l760_76008

variable (C B_inter_C B_union_C B : ℕ)

theorem basketball_students (hC : C = 5) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 9) (hInclusionExclusion : B_union_C = B + C - B_inter_C) : B = 7 := by
  sorry

end NUMINAMATH_GPT_basketball_students_l760_76008


namespace NUMINAMATH_GPT_hyperbola_asymptote_slope_l760_76090

theorem hyperbola_asymptote_slope
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c ≠ -a ∧ c ≠ a)
  (H1 : (c ≠ -a ∧ c ≠ a) ∧ (a ≠ 0) ∧ (b ≠ 0))
  (H_perp : (c + a) * (c - a) * (a * a * a * a) + (b * b * b * b) = 0) :
  abs (b / a) = 1 :=
by
  sorry  -- Proof here is not required as per the given instructions

end NUMINAMATH_GPT_hyperbola_asymptote_slope_l760_76090


namespace NUMINAMATH_GPT_imaginary_part_of_complex_number_l760_76080

open Complex

theorem imaginary_part_of_complex_number :
  ∀ (i : ℂ), i^2 = -1 → im ((2 * I) / (2 + I^3)) = 4 / 5 :=
by
  intro i hi
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_number_l760_76080


namespace NUMINAMATH_GPT_problem_statement_l760_76013

-- Define what it means to be a quadratic equation
def is_quadratic (eqn : String) : Prop :=
  -- In the context of this solution, we'll define a quadratic equation as one
  -- that fits the form ax^2 + bx + c = 0 where a, b, c are constants and a ≠ 0.
  eqn = "x^2 - 2 = 0"

-- We need to formulate a theorem that checks the validity of which equation is quadratic.
theorem problem_statement :
  is_quadratic "x^2 - 2 = 0" :=
sorry

end NUMINAMATH_GPT_problem_statement_l760_76013


namespace NUMINAMATH_GPT_boy_late_l760_76048

noncomputable def time_late (D V1 V2 : ℝ) (early : ℝ) : ℝ :=
  let T1 := D / V1
  let T2 := D / V2
  let T1_mins := T1 * 60
  let T2_mins := T2 * 60
  let actual_on_time := T2_mins + early
  T1_mins - actual_on_time

theorem boy_late :
  time_late 2.5 5 10 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_boy_late_l760_76048
