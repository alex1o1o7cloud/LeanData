import Mathlib

namespace NUMINAMATH_GPT_product_of_fractions_l93_9370

theorem product_of_fractions :
  (1 / 2) * (2 / 3) * (3 / 4) * (3 / 2) = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l93_9370


namespace NUMINAMATH_GPT_A_intersect_B_eq_l93_9375

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x ≤ 1
def A_cap_B (x : ℝ) : Prop := x ∈ {y | A y} ∧ x ∈ {y | B y}

theorem A_intersect_B_eq (x : ℝ) : (A_cap_B x) ↔ (x ∈ Set.Ioc 0 1) :=
by
  sorry

end NUMINAMATH_GPT_A_intersect_B_eq_l93_9375


namespace NUMINAMATH_GPT_cubic_sum_identity_l93_9365

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 10) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 100 :=
by sorry

end NUMINAMATH_GPT_cubic_sum_identity_l93_9365


namespace NUMINAMATH_GPT_cos_neg_30_eq_sqrt_3_div_2_l93_9353

theorem cos_neg_30_eq_sqrt_3_div_2 : 
  Real.cos (-30 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_neg_30_eq_sqrt_3_div_2_l93_9353


namespace NUMINAMATH_GPT_find_x1_l93_9367

theorem find_x1 (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
    (h5 : (1 - x1)^3 + (x1 - x2)^3 + (x2 - x3)^3 + x3^3 = 1 / 8) : x1 = 3 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_x1_l93_9367


namespace NUMINAMATH_GPT_no_valid_a_l93_9346

theorem no_valid_a : ¬ ∃ (a : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 
  ∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 2 * x₁^2 + (3 * a + 1) * x₁ + a^2 = 0 ∧ 2 * x₂^2 + (3 * a + 1) * x₂ + a^2 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_valid_a_l93_9346


namespace NUMINAMATH_GPT_solution_set_nonempty_implies_a_range_l93_9338

theorem solution_set_nonempty_implies_a_range (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_nonempty_implies_a_range_l93_9338


namespace NUMINAMATH_GPT_solve_floor_equation_l93_9317

theorem solve_floor_equation (x : ℝ) (hx : (∃ (y : ℤ), (x^3 - 40 * (y : ℝ) - 78 = 0) ∧ (y : ℝ) ≤ x ∧ x < (y + 1 : ℝ))) :
  x = -5.45 ∨ x = -4.96 ∨ x = -1.26 ∨ x = 6.83 ∨ x = 7.10 :=
by sorry

end NUMINAMATH_GPT_solve_floor_equation_l93_9317


namespace NUMINAMATH_GPT_player_B_wins_l93_9382

variable {R : Type*} [Ring R]

noncomputable def polynomial_game (n : ℕ) (f : Polynomial R) : Prop :=
  (f.degree = 2 * n) ∧ (∃ (a b : R) (x y : R), f.eval x = 0 ∨ f.eval y = 0)

theorem player_B_wins (n : ℕ) (f : Polynomial ℝ)
  (h1 : n ≥ 2)
  (h2 : f.degree = 2 * n) :
  polynomial_game n f :=
by
  sorry

end NUMINAMATH_GPT_player_B_wins_l93_9382


namespace NUMINAMATH_GPT_farmer_children_l93_9324

noncomputable def numberOfChildren 
  (totalLeft : ℕ)
  (eachChildCollected : ℕ)
  (eatenApples : ℕ)
  (soldApples : ℕ) : ℕ :=
  let totalApplesEaten := eatenApples * 2
  let initialCollection := eachChildCollected * (totalLeft + totalApplesEaten + soldApples) / eachChildCollected
  initialCollection / eachChildCollected

theorem farmer_children (totalLeft : ℕ) (eachChildCollected : ℕ) (eatenApples : ℕ) (soldApples : ℕ) : 
  totalLeft = 60 → eachChildCollected = 15 → eatenApples = 4 → soldApples = 7 → 
  numberOfChildren totalLeft eachChildCollected eatenApples soldApples = 5 := 
by
  intro h_totalLeft h_eachChildCollected h_eatenApples h_soldApples
  unfold numberOfChildren
  simp
  sorry

end NUMINAMATH_GPT_farmer_children_l93_9324


namespace NUMINAMATH_GPT_total_pupils_count_l93_9371

theorem total_pupils_count (girls boys : ℕ) (h1 : girls = 692) (h2 : girls = boys + 458) : girls + boys = 926 :=
by 
  sorry

end NUMINAMATH_GPT_total_pupils_count_l93_9371


namespace NUMINAMATH_GPT_base_representation_l93_9351

theorem base_representation (b : ℕ) (h₁ : b^2 ≤ 125) (h₂ : 125 < b^3) :
  (∀ b, b = 12 → 125 % b % 2 = 1) → b = 12 := 
by
  sorry

end NUMINAMATH_GPT_base_representation_l93_9351


namespace NUMINAMATH_GPT_division_simplification_l93_9379

theorem division_simplification : 180 / (12 + 13 * 3) = 60 / 17 := by
  sorry

end NUMINAMATH_GPT_division_simplification_l93_9379


namespace NUMINAMATH_GPT_value_that_number_exceeds_l93_9374

theorem value_that_number_exceeds (V : ℤ) (h : 69 = V + 3 * (86 - 69)) : V = 18 :=
by
  sorry

end NUMINAMATH_GPT_value_that_number_exceeds_l93_9374


namespace NUMINAMATH_GPT_area_of_region_l93_9320

-- Define the equation as a predicate
def region (x y : ℝ) : Prop := x^2 + y^2 + 6*x = 2*y + 10

-- The proof statement
theorem area_of_region : (∃ (x y : ℝ), region x y) → ∃ A : ℝ, A = 20 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_area_of_region_l93_9320


namespace NUMINAMATH_GPT_janet_earnings_per_hour_l93_9350

theorem janet_earnings_per_hour :
  let text_posts := 150
  let image_posts := 80
  let video_posts := 20
  let rate_text := 0.25
  let rate_image := 0.30
  let rate_video := 0.40
  text_posts * rate_text + image_posts * rate_image + video_posts * rate_video = 69.50 :=
by
  sorry

end NUMINAMATH_GPT_janet_earnings_per_hour_l93_9350


namespace NUMINAMATH_GPT_inequality_always_holds_l93_9396

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
sorry

end NUMINAMATH_GPT_inequality_always_holds_l93_9396


namespace NUMINAMATH_GPT_find_a_find_A_l93_9348

-- Part (I)
theorem find_a (b c : ℝ) (A : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = 5 * Real.pi / 6) :
  ∃ a : ℝ, a = 2 * Real.sqrt 7 :=
by {
  sorry
}

-- Part (II)
theorem find_A (b c : ℝ) (C : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hC : C = Real.pi / 2 + A) :
  ∃ A : ℝ, A = Real.pi / 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_find_A_l93_9348


namespace NUMINAMATH_GPT_divisible_by_101_l93_9369

theorem divisible_by_101 (n : ℕ) : (101 ∣ (10^n - 1)) ↔ (∃ k : ℕ, n = 4 * k) :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_101_l93_9369


namespace NUMINAMATH_GPT_rational_solution_unique_l93_9339

theorem rational_solution_unique
  (n : ℕ) (x y : ℚ)
  (hn : Odd n)
  (hx_eqn : x ^ n + 2 * y = y ^ n + 2 * x) :
  x = y :=
sorry

end NUMINAMATH_GPT_rational_solution_unique_l93_9339


namespace NUMINAMATH_GPT_sum_abcd_l93_9359

theorem sum_abcd (a b c d : ℤ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 8) : 
  a + b + c + d = -6 :=
sorry

end NUMINAMATH_GPT_sum_abcd_l93_9359


namespace NUMINAMATH_GPT_marbles_count_l93_9354

theorem marbles_count (M : ℕ)
  (h_blue : (M / 2) = n_blue)
  (h_red : (M / 4) = n_red)
  (h_green : 27 = n_green)
  (h_yellow : 14 = n_yellow)
  (h_total : (n_blue + n_red + n_green + n_yellow) = M) :
  M = 164 :=
by
  sorry

end NUMINAMATH_GPT_marbles_count_l93_9354


namespace NUMINAMATH_GPT_range_of_a_inequality_solution_set_l93_9301

noncomputable def quadratic_condition_holds (a : ℝ) : Prop :=
∀ (x : ℝ), x^2 - 2 * a * x + a > 0

theorem range_of_a (a : ℝ) (h : quadratic_condition_holds a) : 0 < a ∧ a < 1 := sorry

theorem inequality_solution_set (a x : ℝ) (h1 : 0 < a) (h2 : a < 1) : (a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1) ↔ x > 3 := sorry

end NUMINAMATH_GPT_range_of_a_inequality_solution_set_l93_9301


namespace NUMINAMATH_GPT_constant_term_is_21_l93_9384

def poly1 (x : ℕ) := x^3 + x^2 + 3
def poly2 (x : ℕ) := 2*x^4 + x^2 + 7
def expanded_poly (x : ℕ) := poly1 x * poly2 x

theorem constant_term_is_21 : expanded_poly 0 = 21 := by
  sorry

end NUMINAMATH_GPT_constant_term_is_21_l93_9384


namespace NUMINAMATH_GPT_cheryl_used_total_amount_l93_9333

theorem cheryl_used_total_amount :
  let bought_A := (5 / 8 : ℚ)
  let bought_B := (2 / 9 : ℚ)
  let bought_C := (2 / 5 : ℚ)
  let leftover_A := (1 / 12 : ℚ)
  let leftover_B := (5 / 36 : ℚ)
  let leftover_C := (1 / 10 : ℚ)
  let used_A := bought_A - leftover_A
  let used_B := bought_B - leftover_B
  let used_C := bought_C - leftover_C
  used_A + used_B + used_C = 37 / 40 :=
by 
  sorry

end NUMINAMATH_GPT_cheryl_used_total_amount_l93_9333


namespace NUMINAMATH_GPT_gain_percent_correct_l93_9340

theorem gain_percent_correct (C S : ℝ) (h : 50 * C = 28 * S) : 
  ( (S - C) / C ) * 100 = 1100 / 14 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_correct_l93_9340


namespace NUMINAMATH_GPT_intervals_of_monotonicity_and_min_value_l93_9345

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

theorem intervals_of_monotonicity_and_min_value : 
  (∀ x, (x < -1 → f x < f (x + 0.0001)) ∧ (x > -1 ∧ x < 3 → f x > f (x + 0.0001)) ∧ (x > 3 → f x < f (x + 0.0001))) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≥ f 2) :=
by
  sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_and_min_value_l93_9345


namespace NUMINAMATH_GPT_teddy_bears_ordered_l93_9307

theorem teddy_bears_ordered (days : ℕ) (T : ℕ)
  (h1 : 20 * days + 100 = T)
  (h2 : 23 * days - 20 = T) :
  T = 900 ∧ days = 40 := 
by 
  sorry

end NUMINAMATH_GPT_teddy_bears_ordered_l93_9307


namespace NUMINAMATH_GPT_conclusion1_conclusion2_conclusion3_l93_9392

-- Define the Δ operation
def delta (m n : ℚ) : ℚ := (m + n) / (1 + m * n)

-- 1. Proof that (-2^2) Δ 4 = 0
theorem conclusion1 : delta (-4) 4 = 0 := sorry

-- 2. Proof that (1/3) Δ (1/4) = 3 Δ 4
theorem conclusion2 : delta (1/3) (1/4) = delta 3 4 := sorry

-- 3. Proof that (-m) Δ n = m Δ (-n)
theorem conclusion3 (m n : ℚ) : delta (-m) n = delta m (-n) := sorry

end NUMINAMATH_GPT_conclusion1_conclusion2_conclusion3_l93_9392


namespace NUMINAMATH_GPT_probability_at_least_two_same_post_l93_9397

theorem probability_at_least_two_same_post : 
  let volunteers := 3
  let posts := 4
  let total_assignments := posts ^ volunteers
  let different_post_assignments := Nat.factorial posts / (Nat.factorial (posts - volunteers))
  let probability_all_different := different_post_assignments / total_assignments
  let probability_two_same := 1 - probability_all_different
  (1 - (Nat.factorial posts / (total_assignments * Nat.factorial (posts - volunteers)))) = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_two_same_post_l93_9397


namespace NUMINAMATH_GPT_six_digit_number_l93_9331

theorem six_digit_number : ∃ x : ℕ, 100000 ≤ x ∧ x < 1000000 ∧ 3 * x = (x - 300000) * 10 + 3 ∧ x = 428571 :=
by
sorry

end NUMINAMATH_GPT_six_digit_number_l93_9331


namespace NUMINAMATH_GPT_find_initial_workers_l93_9329

-- Define the initial number of workers.
def initial_workers (W : ℕ) (A : ℕ) : Prop :=
  -- Condition 1: W workers can complete work A in 25 days.
  ( W * 25 = A )  ∧
  -- Condition 2: (W + 10) workers can complete work A in 15 days.
  ( (W + 10) * 15 = A )

-- The theorem states that given the conditions, the initial number of workers is 15.
theorem find_initial_workers {W A : ℕ} (h : initial_workers W A) : W = 15 :=
  sorry

end NUMINAMATH_GPT_find_initial_workers_l93_9329


namespace NUMINAMATH_GPT_smallest_N_div_a3_possible_values_of_a3_l93_9347

-- Problem (a)
theorem smallest_N_div_a3 (a : Fin 10 → Nat) (h : StrictMono a) :
  Nat.lcm (a 0) (Nat.lcm (a 1) (Nat.lcm (a 2) (Nat.lcm (a 3) (Nat.lcm (a 4) (Nat.lcm (a 5) (Nat.lcm (a 6) (Nat.lcm (a 7) (Nat.lcm (a 8) (a 9))))))))) / (a 2) = 8 :=
sorry

-- Problem (b)
theorem possible_values_of_a3 (a : Nat) (h_a3_range : 1 ≤ a ∧ a ≤ 1000) :
  a = 315 ∨ a = 630 ∨ a = 945 :=
sorry

end NUMINAMATH_GPT_smallest_N_div_a3_possible_values_of_a3_l93_9347


namespace NUMINAMATH_GPT_correct_expression_l93_9308

theorem correct_expression (a b c : ℝ) : a - b + c = a - (b - c) :=
by
  sorry

end NUMINAMATH_GPT_correct_expression_l93_9308


namespace NUMINAMATH_GPT_find_the_triplet_l93_9383

theorem find_the_triplet (x y z : ℕ) (h : x + y + z = x * y * z) : (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_the_triplet_l93_9383


namespace NUMINAMATH_GPT_factor_ax2_minus_ay2_l93_9398

variable (a x y : ℝ)

theorem factor_ax2_minus_ay2 : a * x^2 - a * y^2 = a * (x + y) * (x - y) := 
sorry

end NUMINAMATH_GPT_factor_ax2_minus_ay2_l93_9398


namespace NUMINAMATH_GPT_roots_of_quadratic_serve_as_eccentricities_l93_9394

theorem roots_of_quadratic_serve_as_eccentricities :
  ∀ (x1 x2 : ℝ), x1 * x2 = 1 ∧ x1 + x2 = 79 → (x1 > 1 ∧ x2 < 1) → 
  (x1 > 1 ∧ x2 < 1) ∧ x1 > 1 ∧ x2 < 1 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_serve_as_eccentricities_l93_9394


namespace NUMINAMATH_GPT_maximum_value_of_k_l93_9344

theorem maximum_value_of_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
    (h4 : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) : k ≤ 1.5 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_k_l93_9344


namespace NUMINAMATH_GPT_find_function_l93_9302

theorem find_function (f : ℕ → ℕ) (h : ∀ m n, f (m + f n) = f (f m) + f n) :
  ∃ d, d > 0 ∧ (∀ m, ∃ k, f m = k * d) :=
sorry

end NUMINAMATH_GPT_find_function_l93_9302


namespace NUMINAMATH_GPT_odd_function_characterization_l93_9342

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_characterization_l93_9342


namespace NUMINAMATH_GPT_sum_a2_a4_a6_l93_9386

theorem sum_a2_a4_a6 : ∀ {a : ℕ → ℕ}, (∀ i, a (i+1) = (1 / 2 : ℝ) * a i) → a 2 = 32 → a 2 + a 4 + a 6 = 42 :=
by
  intros a ha h2
  sorry

end NUMINAMATH_GPT_sum_a2_a4_a6_l93_9386


namespace NUMINAMATH_GPT_array_element_count_l93_9368

theorem array_element_count (A : Finset ℕ) 
  (h1 : ∀ n ∈ A, n ≠ 1 → (∃ a ∈ [2, 3, 5], a ∣ n)) 
  (h2 : ∀ n ∈ A, (2 * n ∈ A ∨ 3 * n ∈ A ∨ 5 * n ∈ A) ↔ (n ∈ A ∧ 2 * n ∈ A ∧ 3 * n ∈ A ∧ 5 * n ∈ A)) 
  (card_A_range : 300 ≤ A.card ∧ A.card ≤ 400) : 
  A.card = 364 := 
sorry

end NUMINAMATH_GPT_array_element_count_l93_9368


namespace NUMINAMATH_GPT_remaining_kibble_l93_9327

def starting_kibble : ℕ := 12
def mary_kibble_morning : ℕ := 1
def mary_kibble_evening : ℕ := 1
def frank_kibble_afternoon : ℕ := 1
def frank_kibble_late_evening : ℕ := 2 * frank_kibble_afternoon

theorem remaining_kibble : starting_kibble - (mary_kibble_morning + mary_kibble_evening + frank_kibble_afternoon + frank_kibble_late_evening) = 7 := by
  sorry

end NUMINAMATH_GPT_remaining_kibble_l93_9327


namespace NUMINAMATH_GPT_smallest_n_l93_9381

theorem smallest_n(vc: ℕ) (n: ℕ) : 
    (vc = 25) ∧ ∃ y o i : ℕ, ((25 * n = 10 * y) ∨ (25 * n = 18 * o) ∨ (25 * n = 20 * i)) → 
    n = 16 := by
    -- We state that given conditions should imply n = 16.
    sorry

end NUMINAMATH_GPT_smallest_n_l93_9381


namespace NUMINAMATH_GPT_unique_solution_l93_9361

def unique_ordered_pair : Prop :=
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ 
               (∃ x : ℝ, x = (m : ℝ)^(1/3) - (n : ℝ)^(1/3) ∧ x^6 + 4 * x^3 - 36 * x^2 + 4 = 0) ∧
               m = 2 ∧ n = 4

theorem unique_solution : unique_ordered_pair := sorry

end NUMINAMATH_GPT_unique_solution_l93_9361


namespace NUMINAMATH_GPT_fraction_spent_is_one_third_l93_9362

-- Define the initial conditions and money variables
def initial_money := 32
def cost_bread := 3
def cost_candy := 2
def remaining_money_after_all := 18

-- Define the calculation for the money left after buying bread and candy bar
def money_left_after_bread_candy := initial_money - cost_bread - cost_candy

-- Define the calculation for the money spent on turkey
def money_spent_on_turkey := money_left_after_bread_candy - remaining_money_after_all

-- The fraction of the remaining money spent on the Turkey
noncomputable def fraction_spent_on_turkey := (money_spent_on_turkey : ℚ) / money_left_after_bread_candy

-- State the theorem that verifies the fraction spent on turkey is 1/3
theorem fraction_spent_is_one_third : fraction_spent_on_turkey = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_spent_is_one_third_l93_9362


namespace NUMINAMATH_GPT_log_sum_l93_9304

theorem log_sum : Real.logb 2 1 + Real.logb 3 9 = 2 := by
  sorry

end NUMINAMATH_GPT_log_sum_l93_9304


namespace NUMINAMATH_GPT_largest_integer_satisfying_inequality_l93_9364

theorem largest_integer_satisfying_inequality :
  ∃ x : ℤ, (6 * x - 5 < 3 * x + 4) ∧ (∀ y : ℤ, (6 * y - 5 < 3 * y + 4) → y ≤ x) ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_satisfying_inequality_l93_9364


namespace NUMINAMATH_GPT_roses_after_trading_equals_36_l93_9390

-- Definitions of the given conditions
def initial_roses_given : ℕ := 24
def roses_after_trade (n : ℕ) : ℕ := n
def remaining_roses_after_first_wilt (roses : ℕ) : ℕ := roses / 2
def remaining_roses_after_second_wilt (roses : ℕ) : ℕ := roses / 2
def roses_remaining_second_day : ℕ := 9

-- The statement we want to prove
theorem roses_after_trading_equals_36 (n : ℕ) (h : roses_remaining_second_day = 9) :
  ( ∃ x, roses_after_trade x = n ∧ remaining_roses_after_first_wilt (remaining_roses_after_first_wilt x) = roses_remaining_second_day ) →
  n = 36 :=
by
  sorry

end NUMINAMATH_GPT_roses_after_trading_equals_36_l93_9390


namespace NUMINAMATH_GPT_range_of_a_l93_9377

noncomputable def p (x : ℝ) : Prop := (1 / (x - 3)) ≥ 1

noncomputable def q (x a : ℝ) : Prop := abs (x - a) < 1

theorem range_of_a (a : ℝ) : (∀ x, p x → q x a) ∧ (∃ x, ¬ (p x) ∧ (q x a)) ↔ (3 < a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l93_9377


namespace NUMINAMATH_GPT_x_plus_q_eq_five_l93_9380

theorem x_plus_q_eq_five (x q : ℝ) (h1 : abs (x - 5) = q) (h2 : x < 5) : x + q = 5 :=
by
  sorry

end NUMINAMATH_GPT_x_plus_q_eq_five_l93_9380


namespace NUMINAMATH_GPT_discriminant_of_quadratic_polynomial_l93_9312

theorem discriminant_of_quadratic_polynomial :
  let a := 5
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ) 
  let Δ := b^2 - 4 * a * c
  Δ = (576/25 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_discriminant_of_quadratic_polynomial_l93_9312


namespace NUMINAMATH_GPT_kamari_toys_eq_65_l93_9393

-- Define the number of toys Kamari has
def number_of_toys_kamari_has : ℕ := sorry

-- Define the number of toys Anais has in terms of K
def number_of_toys_anais_has (K : ℕ) : ℕ := K + 30

-- Define the total number of toys
def total_number_of_toys (K A : ℕ) := K + A

-- Prove that the number of toys Kamari has is 65
theorem kamari_toys_eq_65 : ∃ K : ℕ, (number_of_toys_anais_has K) = K + 30 ∧ total_number_of_toys K (number_of_toys_anais_has K) = 160 ∧ K = 65 :=
by
  sorry

end NUMINAMATH_GPT_kamari_toys_eq_65_l93_9393


namespace NUMINAMATH_GPT_coffee_last_days_l93_9316

theorem coffee_last_days (coffee_weight : ℕ) (cups_per_lb : ℕ) (angie_daily : ℕ) (bob_daily : ℕ) (carol_daily : ℕ) 
  (angie_coffee_weight : coffee_weight = 3) (cups_brewing_rate : cups_per_lb = 40)
  (angie_consumption : angie_daily = 3) (bob_consumption : bob_daily = 2) (carol_consumption : carol_daily = 4) : 
  ((coffee_weight * cups_per_lb) / (angie_daily + bob_daily + carol_daily) = 13) := by
  sorry

end NUMINAMATH_GPT_coffee_last_days_l93_9316


namespace NUMINAMATH_GPT_magnitude_of_c_is_correct_l93_9323

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def b : ℝ × ℝ := (-1, 2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2
noncomputable def c : ℝ × ℝ := (a.1 - (dot_product a b) * b.1, a.2 - (dot_product a b) * b.2)

noncomputable def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt ((u.1 ^ 2) + (u.2 ^ 2))

theorem magnitude_of_c_is_correct :
  magnitude c = 8 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_c_is_correct_l93_9323


namespace NUMINAMATH_GPT_apartments_decrease_l93_9311

theorem apartments_decrease (p_initial e_initial p e q : ℕ) (h1: p_initial = 5) (h2: e_initial = 2) (h3: q = 1)
    (first_mod: p = p_initial - 2) (e_first_mod: e = e_initial + 3) (q_eq: q = 1)
    (second_mod: p = p - 2) (e_second_mod: e = e + 3) :
    p_initial * e_initial * q > p * e * q := by
  sorry

end NUMINAMATH_GPT_apartments_decrease_l93_9311


namespace NUMINAMATH_GPT_area_of_sector_l93_9318

theorem area_of_sector (l : ℝ) (α : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : l = 3)
  (h2 : α = 1)
  (h3 : l = α * r) : 
  S = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_sector_l93_9318


namespace NUMINAMATH_GPT_sum_first_9_terms_l93_9373

-- Definitions of the arithmetic sequence and sum.
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + a m

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Conditions
def a_n (n : ℕ) : ℤ := sorry -- we assume this function gives the n-th term of the arithmetic sequence
def S_n (n : ℕ) : ℤ := sorry -- sum of first n terms
axiom a_5_eq_2 : a_n 5 = 2
axiom arithmetic_sequence_proof : arithmetic_sequence a_n
axiom sum_first_n_proof : sum_first_n a_n S_n

-- Statement to prove
theorem sum_first_9_terms : S_n 9 = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_9_terms_l93_9373


namespace NUMINAMATH_GPT_find_certain_value_l93_9356

noncomputable def certain_value 
  (total_area : ℝ) (smaller_part : ℝ) (difference_fraction : ℝ) : ℝ :=
  (total_area - 2 * smaller_part) / difference_fraction

theorem find_certain_value (total_area : ℝ) (smaller_part : ℝ) (X : ℝ) : 
  total_area = 700 → 
  smaller_part = 315 → 
  (total_area - 2 * smaller_part) / (1/5) = X → 
  X = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_find_certain_value_l93_9356


namespace NUMINAMATH_GPT_quadratic_roots_bc_minus_two_l93_9325

theorem quadratic_roots_bc_minus_two (b c : ℝ) 
  (h1 : 1 + -2 = -b) 
  (h2 : 1 * -2 = c) : b * c = -2 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_bc_minus_two_l93_9325


namespace NUMINAMATH_GPT_other_divisor_l93_9376

theorem other_divisor (x : ℕ) (h1 : 261 % 37 = 2) (h2 : 261 % x = 2) (h3 : 259 = 261 - 2) :
  ∃ x : ℕ, 259 % 37 = 0 ∧ 259 % x = 0 ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_other_divisor_l93_9376


namespace NUMINAMATH_GPT_sequence_root_formula_l93_9310

theorem sequence_root_formula {a : ℕ → ℝ} 
    (h1 : ∀ n, (a (n + 1))^2 = (a n)^2 + 4)
    (h2 : a 1 = 1)
    (h3 : ∀ n, a n > 0) :
    ∀ n, a n = Real.sqrt (4 * n - 3) := 
sorry

end NUMINAMATH_GPT_sequence_root_formula_l93_9310


namespace NUMINAMATH_GPT_smallest_c_ineq_l93_9343

noncomputable def smallest_c {d : ℕ → ℕ} (h_d : ∀ n > 0, d n ≤ d n + 1) := Real.sqrt 3

theorem smallest_c_ineq (d : ℕ → ℕ) (h_d : ∀ n > 0, (d n) ≤ d n + 1) :
  ∀ n : ℕ, n > 0 → d n ≤ smallest_c h_d * (Real.sqrt n) :=
sorry

end NUMINAMATH_GPT_smallest_c_ineq_l93_9343


namespace NUMINAMATH_GPT_g_6_eq_1_l93_9300

variable (f : ℝ → ℝ)

noncomputable def g (x : ℝ) := f x + 1 - x

theorem g_6_eq_1 
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1) :
  g f 6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_g_6_eq_1_l93_9300


namespace NUMINAMATH_GPT_arrange_abc_l93_9399

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom cos_a_eq_a : Real.cos a = a
axiom sin_cos_b_eq_b : Real.sin (Real.cos b) = b
axiom cos_sin_c_eq_c : Real.cos (Real.sin c) = c

theorem arrange_abc : b < a ∧ a < c := 
by
  sorry

end NUMINAMATH_GPT_arrange_abc_l93_9399


namespace NUMINAMATH_GPT_sqrt_square_of_neg_four_l93_9372

theorem sqrt_square_of_neg_four : Real.sqrt ((-4:Real)^2) = 4 := by
  sorry

end NUMINAMATH_GPT_sqrt_square_of_neg_four_l93_9372


namespace NUMINAMATH_GPT_part1_real_values_part2_imaginary_values_l93_9358

namespace ComplexNumberProblem

-- Definitions of conditions for part 1
def imaginaryZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 = 0

def realPositive (x : ℝ) : Prop :=
  x^2 - 2*x - 2 > 0

-- Definition of question for part 1
def realValues (x : ℝ) : Prop :=
  x = -1 ∨ x = -2

-- Proof problem for part 1
theorem part1_real_values (x : ℝ) (h1 : imaginaryZero x) (h2 : realPositive x) : realValues x :=
by
  have h : realValues x := sorry
  exact h

-- Definitions of conditions for part 2
def realPartOne (x : ℝ) : Prop :=
  x^2 - 2*x - 2 = 1

def imaginaryNonZero (x : ℝ) : Prop :=
  x^2 + 3*x + 2 ≠ 0

-- Definition of question for part 2
def imaginaryValues (x : ℝ) : Prop :=
  x = 3

-- Proof problem for part 2
theorem part2_imaginary_values (x : ℝ) (h1 : realPartOne x) (h2 : imaginaryNonZero x) : imaginaryValues x :=
by
  have h : imaginaryValues x := sorry
  exact h

end ComplexNumberProblem

end NUMINAMATH_GPT_part1_real_values_part2_imaginary_values_l93_9358


namespace NUMINAMATH_GPT_similarity_coefficient_interval_l93_9363

-- Definitions
def similarTriangles (x y z p : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ x = k * y ∧ y = k * z ∧ z = k * p

-- Theorem statement
theorem similarity_coefficient_interval (x y z p k : ℝ) (h_sim : similarTriangles x y z p) :
  0 ≤ k ∧ k ≤ 2 :=
sorry

end NUMINAMATH_GPT_similarity_coefficient_interval_l93_9363


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l93_9395

theorem arithmetic_sequence_sum :
  ∃ a : ℕ → ℤ, 
    a 3 = 7 ∧ a 4 = 11 ∧ a 5 = 15 ∧ 
    (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 = 54) := 
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_sum_l93_9395


namespace NUMINAMATH_GPT_tanya_work_days_l93_9387

theorem tanya_work_days (days_sakshi : ℕ) (efficiency_increase : ℚ) (work_rate_sakshi : ℚ) (work_rate_tanya : ℚ) (days_tanya : ℚ) :
  days_sakshi = 15 ->
  efficiency_increase = 1.25 ->
  work_rate_sakshi = 1 / days_sakshi ->
  work_rate_tanya = work_rate_sakshi * efficiency_increase ->
  days_tanya = 1 / work_rate_tanya ->
  days_tanya = 12 :=
by
  intros h_sakshi h_efficiency h_work_rate_sakshi h_work_rate_tanya h_days_tanya
  sorry

end NUMINAMATH_GPT_tanya_work_days_l93_9387


namespace NUMINAMATH_GPT_temperature_on_last_day_l93_9330

noncomputable def last_day_temperature (T1 T2 T3 T4 T5 T6 T7 : ℕ) (mean : ℕ) : ℕ :=
  8 * mean - (T1 + T2 + T3 + T4 + T5 + T6 + T7)

theorem temperature_on_last_day 
  (T1 T2 T3 T4 T5 T6 T7 mean x : ℕ)
  (hT1 : T1 = 82) (hT2 : T2 = 80) (hT3 : T3 = 84) 
  (hT4 : T4 = 86) (hT5 : T5 = 88) (hT6 : T6 = 90) 
  (hT7 : T7 = 88) (hmean : mean = 86) 
  (hx : x = last_day_temperature T1 T2 T3 T4 T5 T6 T7 mean) :
  x = 90 := by
  sorry

end NUMINAMATH_GPT_temperature_on_last_day_l93_9330


namespace NUMINAMATH_GPT_initial_marbles_l93_9309

theorem initial_marbles (M : ℕ) :
  (M - 5) % 3 = 0 ∧ M = 65 :=
by
  sorry

end NUMINAMATH_GPT_initial_marbles_l93_9309


namespace NUMINAMATH_GPT_geom_seq_product_l93_9319

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  2 * a 3 - (a 8) ^ 2 + 2 * a 13 = 0

def geometric_seq (b : ℕ → ℤ) (a8 : ℤ) : Prop :=
  b 8 = a8

theorem geom_seq_product (a b : ℕ → ℤ) (a8 : ℤ) 
  (h1 : arithmetic_seq a)
  (h2 : geometric_seq b a8)
  (h3 : a8 = 4)
: b 4 * b 12 = 16 := sorry

end NUMINAMATH_GPT_geom_seq_product_l93_9319


namespace NUMINAMATH_GPT_quadratic_square_binomial_l93_9321

theorem quadratic_square_binomial (a r s : ℚ) (h1 : a = r^2) (h2 : 2 * r * s = 26) (h3 : s^2 = 9) :
  a = 169/9 := sorry

end NUMINAMATH_GPT_quadratic_square_binomial_l93_9321


namespace NUMINAMATH_GPT_negative_integers_abs_le_4_l93_9326

theorem negative_integers_abs_le_4 (x : Int) (h1 : x < 0) (h2 : abs x ≤ 4) : 
  x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4 :=
by
  sorry

end NUMINAMATH_GPT_negative_integers_abs_le_4_l93_9326


namespace NUMINAMATH_GPT_rational_solutions_are_integers_l93_9366

-- Given two integers a and b, and two equations with rational solutions
variables (a b : ℤ)

-- The first equation is y - 2x = a
def eq1 (y x : ℚ) : Prop := y - 2 * x = a

-- The second equation is y^2 - xy + x^2 = b
def eq2 (y x : ℚ) : Prop := y^2 - x * y + x^2 = b

-- We want to prove that if y and x are rational solutions, they must be integers
theorem rational_solutions_are_integers (y x : ℚ) (h1 : eq1 a y x) (h2 : eq2 b y x) : 
    ∃ (y_int x_int : ℤ), y = y_int ∧ x = x_int :=
sorry

end NUMINAMATH_GPT_rational_solutions_are_integers_l93_9366


namespace NUMINAMATH_GPT_total_notebooks_eq_216_l93_9337

theorem total_notebooks_eq_216 (n : ℕ) 
  (h1 : total_notebooks = n^2 + 20)
  (h2 : total_notebooks = (n + 1)^2 - 9) : 
  total_notebooks = 216 := 
by 
  sorry

end NUMINAMATH_GPT_total_notebooks_eq_216_l93_9337


namespace NUMINAMATH_GPT_find_m_range_l93_9352

noncomputable def f (x m : ℝ) : ℝ := x * abs (x - m) + 2 * x - 3

theorem find_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ m ≤ f x₂ m)
    ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_find_m_range_l93_9352


namespace NUMINAMATH_GPT_total_prime_dates_in_non_leap_year_l93_9389

def prime_dates_in_non_leap_year (days_in_months : List (Nat × Nat)) : Nat :=
  let prime_numbers := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  days_in_months.foldl 
    (λ acc (month, days) => 
      acc + (prime_numbers.filter (λ day => day ≤ days)).length) 
    0

def month_days : List (Nat × Nat) :=
  [(2, 28), (3, 31), (5, 31), (7, 31), (11,30)]

theorem total_prime_dates_in_non_leap_year : prime_dates_in_non_leap_year month_days = 52 :=
  sorry

end NUMINAMATH_GPT_total_prime_dates_in_non_leap_year_l93_9389


namespace NUMINAMATH_GPT_sqrt_range_l93_9335

theorem sqrt_range (a : ℝ) : 2 * a - 1 ≥ 0 ↔ a ≥ 1 / 2 :=
by sorry

end NUMINAMATH_GPT_sqrt_range_l93_9335


namespace NUMINAMATH_GPT_squared_diagonal_inequality_l93_9305

theorem squared_diagonal_inequality 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) :
  let AB := (x1 - x2)^2 + (y1 - y2)^2
  let BC := (x2 - x3)^2 + (y2 - y3)^2
  let CD := (x3 - x4)^2 + (y3 - y4)^2
  let DA := (x1 - x4)^2 + (y1 - y4)^2
  let AC := (x1 - x3)^2 + (y1 - y3)^2
  let BD := (x2 - x4)^2 + (y2 - y4)^2
  AC + BD ≤ AB + BC + CD + DA := 
by
  sorry

end NUMINAMATH_GPT_squared_diagonal_inequality_l93_9305


namespace NUMINAMATH_GPT_sum_of_six_angles_l93_9385

theorem sum_of_six_angles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle3 + angle5 = 180)
  (h2 : angle2 + angle4 + angle6 = 180) :
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 360 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_six_angles_l93_9385


namespace NUMINAMATH_GPT_find_num_pennies_l93_9378

def total_value (nickels : ℕ) (dimes : ℕ) (pennies : ℕ) : ℕ :=
  5 * nickels + 10 * dimes + pennies

def num_pennies (nickels_value: ℕ) (dimes_value: ℕ) (total: ℕ): ℕ :=
  total - (nickels_value + dimes_value)

theorem find_num_pennies : 
  ∀ (total : ℕ) (num_nickels : ℕ) (num_dimes: ℕ),
  total = 59 → num_nickels = 4 → num_dimes = 3 → num_pennies (5 * num_nickels) (10 * num_dimes) total = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_num_pennies_l93_9378


namespace NUMINAMATH_GPT_series_convergence_p_geq_2_l93_9355

noncomputable def ai_series_converges (a : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, a i ^ 2 = l

noncomputable def bi_series_converges (b : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, ∑' i, b i ^ 2 = l

theorem series_convergence_p_geq_2 
  (a b : ℕ → ℝ) 
  (h₁ : ai_series_converges a)
  (h₂ : bi_series_converges b) 
  (p : ℝ) (hp : p ≥ 2) : 
  ∃ l : ℝ, ∑' i, |a i - b i| ^ p = l := 
sorry

end NUMINAMATH_GPT_series_convergence_p_geq_2_l93_9355


namespace NUMINAMATH_GPT_pies_in_each_row_l93_9336

theorem pies_in_each_row (pecan_pies apple_pies rows : Nat) (hpecan : pecan_pies = 16) (happle : apple_pies = 14) (hrows : rows = 30) :
  (pecan_pies + apple_pies) / rows = 1 :=
by
  sorry

end NUMINAMATH_GPT_pies_in_each_row_l93_9336


namespace NUMINAMATH_GPT_chantel_final_bracelets_count_l93_9328

def bracelets_made_in_first_5_days : ℕ := 5 * 2

def bracelets_after_giving_away_at_school : ℕ := bracelets_made_in_first_5_days - 3

def bracelets_made_in_next_4_days : ℕ := 4 * 3

def total_bracelets_before_soccer_giveaway : ℕ := bracelets_after_giving_away_at_school + bracelets_made_in_next_4_days

def bracelets_after_giving_away_at_soccer : ℕ := total_bracelets_before_soccer_giveaway - 6

theorem chantel_final_bracelets_count : bracelets_after_giving_away_at_soccer = 13 :=
sorry

end NUMINAMATH_GPT_chantel_final_bracelets_count_l93_9328


namespace NUMINAMATH_GPT_base_n_not_divisible_by_11_l93_9388

theorem base_n_not_divisible_by_11 :
  ∀ n, 2 ≤ n ∧ n ≤ 100 → (6 + 2*n + 5*n^2 + 4*n^3 + 2*n^4 + 4*n^5) % 11 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_base_n_not_divisible_by_11_l93_9388


namespace NUMINAMATH_GPT_proof_inequality_l93_9315

theorem proof_inequality (n : ℕ) (a b : ℝ) (c : ℝ) (h_n : 1 ≤ n) (h_a : 1 ≤ a) (h_b : 1 ≤ b) (h_c : 0 < c) : 
  ((ab + c)^n - c) / ((b + c)^n - c) ≤ a^n :=
sorry

end NUMINAMATH_GPT_proof_inequality_l93_9315


namespace NUMINAMATH_GPT_seven_pow_eight_mod_100_l93_9334

theorem seven_pow_eight_mod_100 :
  (7 ^ 8) % 100 = 1 := 
by {
  -- here can be the steps of the proof, but for now we use sorry
  sorry
}

end NUMINAMATH_GPT_seven_pow_eight_mod_100_l93_9334


namespace NUMINAMATH_GPT_period_change_l93_9360

theorem period_change {f : ℝ → ℝ} (T : ℝ) (hT : 0 < T) (h_period : ∀ x, f (x + T) = f x) (α : ℝ) (hα : 0 < α) :
  ∀ x, f (α * (x + T / α)) = f (α * x) :=
by
  sorry

end NUMINAMATH_GPT_period_change_l93_9360


namespace NUMINAMATH_GPT_area_of_sector_l93_9313

theorem area_of_sector (r l : ℝ) (h1 : l + 2 * r = 12) (h2 : l / r = 2) : (1 / 2) * l * r = 9 :=
by
  sorry

end NUMINAMATH_GPT_area_of_sector_l93_9313


namespace NUMINAMATH_GPT_iron_wire_left_l93_9341

-- Given conditions as variables
variable (initial_usage : ℚ) (additional_usage : ℚ)

-- Conditions as hypotheses
def conditions := initial_usage = 2 / 9 ∧ additional_usage = 3 / 9

-- The goal to prove
theorem iron_wire_left (h : conditions initial_usage additional_usage):
  1 - initial_usage - additional_usage = 4 / 9 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_iron_wire_left_l93_9341


namespace NUMINAMATH_GPT_find_amount_l93_9349

theorem find_amount (N : ℝ) (hN : N = 24) (A : ℝ) (hA : A = 0.6667 * N - 0.25 * N) : A = 10.0008 :=
by
  rw [hN] at hA
  sorry

end NUMINAMATH_GPT_find_amount_l93_9349


namespace NUMINAMATH_GPT_union_A_B_comp_U_A_inter_B_range_of_a_l93_9391

namespace ProofProblem

def A : Set ℝ := { x | 2 ≤ x ∧ x ≤ 8 }
def B : Set ℝ := { x | 1 < x ∧ x < 6 }
def C (a : ℝ) : Set ℝ := { x | x > a }
def U : Set ℝ := Set.univ

theorem union_A_B : A ∪ B = { x | 1 < x ∧ x ≤ 8 } := by
  sorry

theorem comp_U_A_inter_B : (U \ A) ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

theorem range_of_a (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_union_A_B_comp_U_A_inter_B_range_of_a_l93_9391


namespace NUMINAMATH_GPT_greatest_possible_value_x_y_l93_9314

noncomputable def max_x_y : ℕ :=
  let s1 := 150
  let s2 := 210
  let s3 := 270
  let s4 := 330
  (3 * (s3 + s4) - (s1 + s2 + s3 + s4))

theorem greatest_possible_value_x_y :
  max_x_y = 840 := by
  sorry

end NUMINAMATH_GPT_greatest_possible_value_x_y_l93_9314


namespace NUMINAMATH_GPT_find_n_l93_9306

theorem find_n (n : ℕ) (h : 7^(2*n) = (1/7)^(n-12)) : n = 4 :=
sorry

end NUMINAMATH_GPT_find_n_l93_9306


namespace NUMINAMATH_GPT_hanna_has_money_l93_9357

variable (total_roses money_spent : ℕ)
variable (rose_price : ℕ := 2)

def hanna_gives_roses (total_roses : ℕ) : Bool :=
  (1 / 3 * total_roses + 1 / 2 * total_roses) = 125

theorem hanna_has_money (H : hanna_gives_roses total_roses) : money_spent = 300 := sorry

end NUMINAMATH_GPT_hanna_has_money_l93_9357


namespace NUMINAMATH_GPT_difference_of_squares_example_l93_9322

theorem difference_of_squares_example : 204^2 - 202^2 = 812 := by
  sorry

end NUMINAMATH_GPT_difference_of_squares_example_l93_9322


namespace NUMINAMATH_GPT_boat_b_takes_less_time_l93_9332

theorem boat_b_takes_less_time (A_speed_still : ℝ) (B_speed_still : ℝ)
  (A_current : ℝ) (B_current : ℝ) (distance_downstream : ℝ)
  (A_speed_downstream : A_speed_still + A_current = 26)
  (B_speed_downstream : B_speed_still + B_current = 28)
  (A_time : A_speed_still + A_current = 26 → distance_downstream / (A_speed_still + A_current) = 4.6154)
  (B_time : B_speed_still + B_current = 28 → distance_downstream / (B_speed_still + B_current) = 4.2857) :
  distance_downstream / (B_speed_still + B_current) < distance_downstream / (A_speed_still + A_current) :=
by sorry

end NUMINAMATH_GPT_boat_b_takes_less_time_l93_9332


namespace NUMINAMATH_GPT_oliver_final_money_l93_9303

-- Define the initial conditions as variables and constants
def initial_amount : Nat := 9
def savings : Nat := 5
def earnings : Nat := 6
def spent_frisbee : Nat := 4
def spent_puzzle : Nat := 3
def spent_stickers : Nat := 2
def movie_ticket_price : Nat := 10
def movie_ticket_discount : Nat := 20 -- 20%
def snack_price : Nat := 3
def snack_discount : Nat := 1
def birthday_gift : Nat := 8

-- Define the final amount of money Oliver has left based on the problem statement
def final_amount : Nat :=
  let total_money := initial_amount + savings + earnings
  let total_spent := spent_frisbee + spent_puzzle + spent_stickers
  let remaining_after_spending := total_money - total_spent
  let discounted_movie_ticket := movie_ticket_price * (100 - movie_ticket_discount) / 100
  let discounted_snack := snack_price - snack_discount
  let total_spent_after_discounts := discounted_movie_ticket + discounted_snack
  let remaining_after_discounts := remaining_after_spending - total_spent_after_discounts
  remaining_after_discounts + birthday_gift

-- Lean theorem statement to prove that Oliver ends up with $9
theorem oliver_final_money : final_amount = 9 := by
  sorry

end NUMINAMATH_GPT_oliver_final_money_l93_9303
