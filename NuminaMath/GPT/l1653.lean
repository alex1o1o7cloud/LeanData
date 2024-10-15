import Mathlib

namespace NUMINAMATH_GPT_unbroken_seashells_l1653_165310

theorem unbroken_seashells (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) 
  (h_unbroken : unbroken_seashells = total_seashells - broken_seashells) : 
  unbroken_seashells = 3 :=
by 
  rw [h_total, h_broken] at h_unbroken
  exact h_unbroken

end NUMINAMATH_GPT_unbroken_seashells_l1653_165310


namespace NUMINAMATH_GPT_ln_gt_ln_sufficient_for_x_gt_y_l1653_165309

noncomputable def ln : ℝ → ℝ := sorry  -- Assuming ln is imported from Mathlib

-- Conditions
variable (x y : ℝ)
axiom ln_gt_ln_of_x_gt_y (hxy : x > y) (hx_pos : 0 < x) (hy_pos : 0 < y) : ln x > ln y

theorem ln_gt_ln_sufficient_for_x_gt_y (h : ln x > ln y) : x > y := sorry

end NUMINAMATH_GPT_ln_gt_ln_sufficient_for_x_gt_y_l1653_165309


namespace NUMINAMATH_GPT_inequality_lemma_l1653_165351

theorem inequality_lemma (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (b * c + c * d + d * a - 1)) +
  (1 / (a * b + c * d + d * a - 1)) +
  (1 / (a * b + b * c + d * a - 1)) +
  (1 / (a * b + b * c + c * d - 1)) ≤ 2 :=
sorry

end NUMINAMATH_GPT_inequality_lemma_l1653_165351


namespace NUMINAMATH_GPT_sum_of_square_face_is_13_l1653_165355

-- Definitions based on conditions
variables (x₁ x₂ x₃ x₄ x₅ : ℕ)

-- Conditions
axiom h₁ : x₁ + x₂ + x₃ = 7
axiom h₂ : x₁ + x₂ + x₄ = 8
axiom h₃ : x₁ + x₃ + x₄ = 9
axiom h₄ : x₂ + x₃ + x₄ = 10

-- Properties
axiom h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 15

-- Goal to prove
theorem sum_of_square_face_is_13 (h₁ : x₁ + x₂ + x₃ = 7) (h₂ : x₁ + x₂ + x₄ = 8) 
  (h₃ : x₁ + x₃ + x₄ = 9) (h₄ : x₂ + x₃ + x₄ = 10) (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ = 15): 
  x₅ + x₁ + x₂ + x₄ = 13 :=
sorry

end NUMINAMATH_GPT_sum_of_square_face_is_13_l1653_165355


namespace NUMINAMATH_GPT_pizza_party_l1653_165365

theorem pizza_party (boys girls : ℕ) :
  (7 * boys + 3 * girls ≤ 59) ∧ (6 * boys + 2 * girls ≥ 49) ∧ (boys + girls ≤ 10) → 
  boys = 8 ∧ girls = 1 := 
by sorry

end NUMINAMATH_GPT_pizza_party_l1653_165365


namespace NUMINAMATH_GPT_shirts_production_l1653_165374

-- Definitions
def constant_rate (r : ℕ) : Prop := ∀ n : ℕ, 8 * n * r = 160 * n

theorem shirts_production (r : ℕ) (h : constant_rate r) : 16 * r = 32 :=
by sorry

end NUMINAMATH_GPT_shirts_production_l1653_165374


namespace NUMINAMATH_GPT_smallest_n_for_cubic_sum_inequality_l1653_165320

theorem smallest_n_for_cubic_sum_inequality :
  ∃ n : ℕ, (∀ (a b c : ℕ), (a + b + c) ^ 3 ≤ n * (a ^ 3 + b ^ 3 + c ^ 3)) ∧ n = 9 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_cubic_sum_inequality_l1653_165320


namespace NUMINAMATH_GPT_like_terms_sum_l1653_165394

theorem like_terms_sum (n m : ℕ) 
  (h1 : n + 1 = 3) 
  (h2 : m - 1 = 3) : 
  m + n = 6 := 
  sorry

end NUMINAMATH_GPT_like_terms_sum_l1653_165394


namespace NUMINAMATH_GPT_dolls_completion_time_l1653_165361

def time_to_complete_dolls (craft_time_per_doll break_time_per_three_dolls total_dolls start_time : Nat) : Nat :=
  let total_craft_time := craft_time_per_doll * total_dolls
  let total_breaks := (total_dolls / 3) * break_time_per_three_dolls
  let total_time := total_craft_time + total_breaks
  (start_time + total_time) % 1440 -- 1440 is the number of minutes in a day

theorem dolls_completion_time :
  time_to_complete_dolls 105 30 10 600 = 300 := -- 600 is 10:00 AM in minutes, 300 is 5:00 AM in minutes
sorry

end NUMINAMATH_GPT_dolls_completion_time_l1653_165361


namespace NUMINAMATH_GPT_smallest_sum_of_factors_of_12_factorial_l1653_165343

theorem smallest_sum_of_factors_of_12_factorial :
  ∃ (x y z w : Nat), x * y * z * w = Nat.factorial 12 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ x + y + z + w = 147 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_factors_of_12_factorial_l1653_165343


namespace NUMINAMATH_GPT_cheryl_same_color_probability_l1653_165322

/-- Defines the probability of Cheryl picking 3 marbles of the same color from the given box setup. -/
def probability_cheryl_picks_same_color : ℚ :=
  let total_ways := (Nat.choose 9 3) * (Nat.choose 6 3) * (Nat.choose 3 3)
  let favorable_ways := 3 * (Nat.choose 6 3)
  (favorable_ways : ℚ) / (total_ways : ℚ)

/-- Theorem stating the probability that Cheryl picks 3 marbles of the same color is 1/28. -/
theorem cheryl_same_color_probability :
  probability_cheryl_picks_same_color = 1 / 28 :=
by
  sorry

end NUMINAMATH_GPT_cheryl_same_color_probability_l1653_165322


namespace NUMINAMATH_GPT_parabola_int_x_axis_for_all_m_l1653_165331

theorem parabola_int_x_axis_for_all_m {n : ℝ} :
  (∀ m : ℝ, (9 * m^2 - 4 * m - 4 * n) ≥ 0) → (n ≤ -1 / 9) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_int_x_axis_for_all_m_l1653_165331


namespace NUMINAMATH_GPT_range_of_quadratic_function_is_geq_11_over_4_l1653_165364

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - x + 3

-- Define the range of the quadratic function
def range_of_quadratic_function := {y : ℝ | ∃ x : ℝ, quadratic_function x = y}

-- Prove the statement
theorem range_of_quadratic_function_is_geq_11_over_4 : range_of_quadratic_function = {y : ℝ | y ≥ 11 / 4} :=
by
  sorry

end NUMINAMATH_GPT_range_of_quadratic_function_is_geq_11_over_4_l1653_165364


namespace NUMINAMATH_GPT_find_m_l1653_165319

theorem find_m (m : ℝ) : 
  (∀ (x y : ℝ), (y = x + m ∧ x = 0) → y = m) ∧
  (∀ (x y : ℝ), (y = 2 * x - 2 ∧ x = 0) → y = -2) ∧
  (∀ (x : ℝ), (∃ y : ℝ, (y = x + m ∧ x = 0) ∧ (y = 2 * x - 2 ∧ x = 0))) → 
  m = -2 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_l1653_165319


namespace NUMINAMATH_GPT_find_inverse_sum_l1653_165300

def f (x : ℝ) : ℝ := x * |x|^2

theorem find_inverse_sum :
  (∃ x : ℝ, f x = 8) ∧ (∃ y : ℝ, f y = -64) → 
  (∃ a b : ℝ, f a = 8 ∧ f b = -64 ∧ a + b = 6) :=
sorry

end NUMINAMATH_GPT_find_inverse_sum_l1653_165300


namespace NUMINAMATH_GPT_determine_x_l1653_165334

variable {m x : ℝ}

theorem determine_x (h₁ : m > 25)
    (h₂ : ((m / 100) * m = (m - 20) / 100 * (m + x))) : 
    x = 20 * m / (m - 20) := 
sorry

end NUMINAMATH_GPT_determine_x_l1653_165334


namespace NUMINAMATH_GPT_unique_ordered_triple_l1653_165338

theorem unique_ordered_triple (a b c : ℕ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : a^3 + b^3 + c^3 + 648 = (a + b + c)^3) :
  (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) :=
sorry

end NUMINAMATH_GPT_unique_ordered_triple_l1653_165338


namespace NUMINAMATH_GPT_final_amount_after_bets_l1653_165376

theorem final_amount_after_bets :
  let initial_amount := 128
  let num_bets := 8
  let num_wins := 4
  let num_losses := 4
  let bonus_per_win_after_loss := 10
  let win_multiplier := 3 / 2
  let loss_multiplier := 1 / 2
  ∃ final_amount : ℝ,
    (final_amount =
      initial_amount * (win_multiplier ^ num_wins) * (loss_multiplier ^ num_losses) + 2 * bonus_per_win_after_loss) ∧
    final_amount = 60.5 :=
sorry

end NUMINAMATH_GPT_final_amount_after_bets_l1653_165376


namespace NUMINAMATH_GPT_div_count_27n5_l1653_165362

theorem div_count_27n5 
  (n : ℕ) 
  (h : (120 * n^3).divisors.card = 120) 
  : (27 * n^5).divisors.card = 324 :=
sorry

end NUMINAMATH_GPT_div_count_27n5_l1653_165362


namespace NUMINAMATH_GPT_car_speeds_midpoint_condition_l1653_165302

theorem car_speeds_midpoint_condition 
  (v k : ℝ) (h_k : k > 1) 
  (A B C D : ℝ) (AB AD CD : ℝ)
  (h_midpoint : AD = AB / 2) 
  (h_CD_AD : CD / AD = 1 / 2)
  (h_D_midpoint : D = (A + B) / 2) 
  (h_C_on_return : C = D - CD) 
  (h_speeds : (v > 0) ∧ (k * v > v)) 
  (h_AB_AD : AB = 2 * AD) :
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_car_speeds_midpoint_condition_l1653_165302


namespace NUMINAMATH_GPT_negation_of_exist_prop_l1653_165378

theorem negation_of_exist_prop :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_negation_of_exist_prop_l1653_165378


namespace NUMINAMATH_GPT_find_tricycles_l1653_165323

theorem find_tricycles (b t w : ℕ) 
  (sum_children : b + t + w = 10)
  (sum_wheels : 2 * b + 3 * t = 26) :
  t = 6 :=
by sorry

end NUMINAMATH_GPT_find_tricycles_l1653_165323


namespace NUMINAMATH_GPT_marie_erasers_l1653_165391

theorem marie_erasers (initial_erasers : ℕ) (lost_erasers : ℕ) (final_erasers : ℕ) :
  initial_erasers = 95 → lost_erasers = 42 → final_erasers = initial_erasers - lost_erasers → final_erasers = 53 :=
by
  intros h_initial h_lost h_final
  rw [h_initial, h_lost] at h_final
  exact h_final

end NUMINAMATH_GPT_marie_erasers_l1653_165391


namespace NUMINAMATH_GPT_brocard_inequality_part_a_brocard_inequality_part_b_l1653_165305

variable (α β γ φ : ℝ)

theorem brocard_inequality_part_a (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  φ^3 ≤ (α - φ) * (β - φ) * (γ - φ) := 
sorry

theorem brocard_inequality_part_b (h_sum_angles : α + β + γ = π) (h_brocard : 0 < φ ∧ φ < π/2) :
  8 * φ^3 ≤ α * β * γ := 
sorry

end NUMINAMATH_GPT_brocard_inequality_part_a_brocard_inequality_part_b_l1653_165305


namespace NUMINAMATH_GPT_athlete_A_most_stable_l1653_165335

noncomputable def athlete_A_variance : ℝ := 0.019
noncomputable def athlete_B_variance : ℝ := 0.021
noncomputable def athlete_C_variance : ℝ := 0.020
noncomputable def athlete_D_variance : ℝ := 0.022

theorem athlete_A_most_stable :
  athlete_A_variance < athlete_B_variance ∧
  athlete_A_variance < athlete_C_variance ∧
  athlete_A_variance < athlete_D_variance :=
by {
  sorry
}

end NUMINAMATH_GPT_athlete_A_most_stable_l1653_165335


namespace NUMINAMATH_GPT_quadratic_real_roots_l1653_165366

theorem quadratic_real_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l1653_165366


namespace NUMINAMATH_GPT_function_equality_l1653_165381

theorem function_equality (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, f n < f (n + 1) )
  (h2 : f 2 = 2)
  (h3 : ∀ m n : ℕ, f (m * n) = f m * f n) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end NUMINAMATH_GPT_function_equality_l1653_165381


namespace NUMINAMATH_GPT_leadership_board_stabilizes_l1653_165392

theorem leadership_board_stabilizes :
  ∃ n : ℕ, 2 ^ n - 1 ≤ 2020 ∧ 2020 < 2 ^ (n + 1) - 1 := by
  sorry

end NUMINAMATH_GPT_leadership_board_stabilizes_l1653_165392


namespace NUMINAMATH_GPT_balloons_difference_l1653_165357

theorem balloons_difference (yours friends : ℝ) (hyours : yours = -7) (hfriends : friends = 4.5) :
  friends - yours = 11.5 :=
by
  rw [hyours, hfriends]
  sorry

end NUMINAMATH_GPT_balloons_difference_l1653_165357


namespace NUMINAMATH_GPT_toy_production_difference_l1653_165363

variables (w t : ℕ)
variable  (t_nonneg : 0 < t) -- assuming t is always non-negative for a valid working hour.
variable  (h : w = 3 * t)

theorem toy_production_difference : 
  (w * t) - ((w + 5) * (t - 3)) = 4 * t + 15 :=
by
  sorry

end NUMINAMATH_GPT_toy_production_difference_l1653_165363


namespace NUMINAMATH_GPT_B_catches_up_with_A_l1653_165387

theorem B_catches_up_with_A :
  let d := 140
  let vA := 10
  let vB := 20
  let tA := d / vA
  let tB := d / vB
  tA - tB = 7 := 
by
  -- Definitions
  let d := 140
  let vA := 10
  let vB := 20
  let tA := d / vA
  let tB := d / vB
  -- Goal
  show tA - tB = 7
  sorry

end NUMINAMATH_GPT_B_catches_up_with_A_l1653_165387


namespace NUMINAMATH_GPT_inequality_solution_l1653_165307

theorem inequality_solution (a x : ℝ) : 
  (a = 0 → ¬(x^2 - 2*a*x - 3*a^2 < 0)) ∧
  (a > 0 → (-a < x ∧ x < 3*a) ↔ (x^2 - 2*a*x - 3*a^2 < 0)) ∧
  (a < 0 → (3*a < x ∧ x < -a) ↔ (x^2 - 2*a*x - 3*a^2 < 0)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1653_165307


namespace NUMINAMATH_GPT_jaime_saves_enough_l1653_165398

-- Definitions of the conditions
def weekly_savings : ℕ := 50
def bi_weekly_expense : ℕ := 46
def target_savings : ℕ := 135

-- The proof goal
theorem jaime_saves_enough : ∃ weeks : ℕ, 2 * ((weeks * weekly_savings - bi_weekly_expense) / 2) = target_savings := 
sorry

end NUMINAMATH_GPT_jaime_saves_enough_l1653_165398


namespace NUMINAMATH_GPT_boys_left_hand_to_girl_l1653_165396

-- Definitions based on the given conditions
def num_boys : ℕ := 40
def num_girls : ℕ := 28
def boys_right_hand_to_girl : ℕ := 18

-- Statement to prove
theorem boys_left_hand_to_girl : (num_boys - (num_boys - boys_right_hand_to_girl)) = boys_right_hand_to_girl := by
  sorry

end NUMINAMATH_GPT_boys_left_hand_to_girl_l1653_165396


namespace NUMINAMATH_GPT_red_candies_count_l1653_165388

def total_candies : ℕ := 3409
def blue_candies : ℕ := 3264

theorem red_candies_count : total_candies - blue_candies = 145 := by
  sorry

end NUMINAMATH_GPT_red_candies_count_l1653_165388


namespace NUMINAMATH_GPT_find_cos_7theta_l1653_165311

theorem find_cos_7theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (7 * θ) = 1105 / 16384 :=
by
  sorry

end NUMINAMATH_GPT_find_cos_7theta_l1653_165311


namespace NUMINAMATH_GPT_sqrt_47_minus_2_range_l1653_165399

theorem sqrt_47_minus_2_range (h : 6 < Real.sqrt 47 ∧ Real.sqrt 47 < 7) : 4 < Real.sqrt 47 - 2 ∧ Real.sqrt 47 - 2 < 5 := by
  sorry

end NUMINAMATH_GPT_sqrt_47_minus_2_range_l1653_165399


namespace NUMINAMATH_GPT_bumper_cars_number_of_tickets_l1653_165339

theorem bumper_cars_number_of_tickets (Ferris_Wheel Roller_Coaster Jeanne_Has Jeanne_Buys : ℕ)
  (h1 : Ferris_Wheel = 5)
  (h2 : Roller_Coaster = 4)
  (h3 : Jeanne_Has = 5)
  (h4 : Jeanne_Buys = 8) :
  Ferris_Wheel + Roller_Coaster + (13 - (Ferris_Wheel + Roller_Coaster)) = 13 - (Ferris_Wheel + Roller_Coaster) :=
by
  sorry

end NUMINAMATH_GPT_bumper_cars_number_of_tickets_l1653_165339


namespace NUMINAMATH_GPT_parabola_directrix_l1653_165313

theorem parabola_directrix (x y : ℝ) :
  (∃ a b c : ℝ, y = (a * x^2 + b * x + c) / 12 ∧ a = 1 ∧ b = -6 ∧ c = 5) →
  y = -10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1653_165313


namespace NUMINAMATH_GPT_find_rate_squares_sum_l1653_165359

theorem find_rate_squares_sum {b j s : ℤ} 
(H1 : 3 * b + 2 * j + 2 * s = 112)
(H2 : 2 * b + 3 * j + 4 * s = 129) : b^2 + j^2 + s^2 = 1218 :=
by sorry

end NUMINAMATH_GPT_find_rate_squares_sum_l1653_165359


namespace NUMINAMATH_GPT_total_value_of_coins_l1653_165344

theorem total_value_of_coins :
  (∀ (coins : List (String × ℕ)), coins.length = 12 →
    (∃ Q N : ℕ, 
      Q = 4 ∧ N = 8 ∧
      (∀ (coin : String × ℕ), coin ∈ coins → 
        (coin = ("quarter", Q) → Q = 4 ∧ (Q * 25 = 100)) ∧ 
        (coin = ("nickel", N) → N = 8 ∧ (N * 5 = 40)) ∧
      (Q * 25 + N * 5 = 140)))) :=
sorry

end NUMINAMATH_GPT_total_value_of_coins_l1653_165344


namespace NUMINAMATH_GPT_product_gt_one_l1653_165383

theorem product_gt_one 
  (m : ℚ) (b : ℚ)
  (hm : m = 3 / 4)
  (hb : b = 5 / 2) :
  m * b > 1 := 
by
  sorry

end NUMINAMATH_GPT_product_gt_one_l1653_165383


namespace NUMINAMATH_GPT_largest_integer_mod_l1653_165397

theorem largest_integer_mod (a : ℕ) (h₁ : a < 100) (h₂ : a % 5 = 2) : a = 97 :=
by sorry

end NUMINAMATH_GPT_largest_integer_mod_l1653_165397


namespace NUMINAMATH_GPT_bella_bakes_most_cookies_per_batch_l1653_165389

theorem bella_bakes_most_cookies_per_batch (V : ℝ) :
  let alex_cookies := V / 9
  let bella_cookies := V / 7
  let carlo_cookies := V / 8
  let dana_cookies := V / 10
  alex_cookies < bella_cookies ∧ carlo_cookies < bella_cookies ∧ dana_cookies < bella_cookies :=
sorry

end NUMINAMATH_GPT_bella_bakes_most_cookies_per_batch_l1653_165389


namespace NUMINAMATH_GPT_square_difference_l1653_165371

theorem square_difference :
  153^2 - 147^2 = 1800 :=
by
  sorry

end NUMINAMATH_GPT_square_difference_l1653_165371


namespace NUMINAMATH_GPT_lemonade_stand_total_profit_l1653_165329

theorem lemonade_stand_total_profit :
  let day1_revenue := 21 * 4
  let day1_expenses := 10 + 5 + 3
  let day1_profit := day1_revenue - day1_expenses

  let day2_revenue := 18 * 5
  let day2_expenses := 12 + 6 + 4
  let day2_profit := day2_revenue - day2_expenses

  let day3_revenue := 25 * 4
  let day3_expenses := 8 + 4 + 3 + 2
  let day3_profit := day3_revenue - day3_expenses

  let total_profit := day1_profit + day2_profit + day3_profit

  total_profit = 217 := by
    sorry

end NUMINAMATH_GPT_lemonade_stand_total_profit_l1653_165329


namespace NUMINAMATH_GPT_smallest_real_solution_l1653_165382

theorem smallest_real_solution (x : ℝ) : 
  (x * |x| = 3 * x + 4) → x = 4 :=
by {
  sorry -- Proof omitted as per the instructions
}

end NUMINAMATH_GPT_smallest_real_solution_l1653_165382


namespace NUMINAMATH_GPT_minimum_ticket_cost_l1653_165379

theorem minimum_ticket_cost :
  let num_people := 12
  let num_adults := 8
  let num_children := 4
  let adult_ticket_cost := 100
  let child_ticket_cost := 50
  let group_ticket_cost := 70
  num_people = num_adults + num_children →
  (num_people >= 10) →
  ∃ (cost : ℕ), cost = min (num_adults * adult_ticket_cost + num_children * child_ticket_cost) (group_ticket_cost * num_people) ∧
  cost = min (group_ticket_cost * 10 + child_ticket_cost * (num_people - 10)) (group_ticket_cost * num_people) →
  cost = 800 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_minimum_ticket_cost_l1653_165379


namespace NUMINAMATH_GPT_calculate_expression_l1653_165318

theorem calculate_expression : 
  (-7 : ℤ)^7 / (7 : ℤ)^4 + 2^6 - 8^2 = -343 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1653_165318


namespace NUMINAMATH_GPT_range_of_a_l1653_165301

open Real

noncomputable def f (x : ℝ) := x - sqrt (x^2 + x)

noncomputable def g (x a : ℝ) := log x / log 27 - log x / log 9 + a * log x / log 3

theorem range_of_a (a : ℝ) : (∀ x1 ∈ Set.Ioi 1, ∃ x2 ∈ Set.Icc 3 9, f x1 > g x2 a) → a ≤ -1/12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1653_165301


namespace NUMINAMATH_GPT_power_function_odd_f_m_plus_1_l1653_165368

noncomputable def f (x : ℝ) (m : ℝ) := x^(2 + m)

theorem power_function_odd_f_m_plus_1 (m : ℝ) (h_odd : ∀ x : ℝ, f (-x) m = -f x m)
  (h_domain : -1 ≤ m) : f (m + 1) m = 1 := by
  sorry

end NUMINAMATH_GPT_power_function_odd_f_m_plus_1_l1653_165368


namespace NUMINAMATH_GPT_remainder_of_f_when_divided_by_x_plus_2_l1653_165395

def f (x : ℝ) : ℝ := x^4 - 6 * x^3 + 11 * x^2 + 8 * x - 20

theorem remainder_of_f_when_divided_by_x_plus_2 : f (-2) = 72 := by
  sorry

end NUMINAMATH_GPT_remainder_of_f_when_divided_by_x_plus_2_l1653_165395


namespace NUMINAMATH_GPT_find_p_minus_q_l1653_165390

theorem find_p_minus_q (x y p q : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 3 / (x * p) = 8) (h2 : 5 / (y * q) = 18)
  (hminX : ∀ x', x' ≠ 0 → 3 / (x' * 3) ≠ 1 / 8)
  (hminY : ∀ y', y' ≠ 0 → 5 / (y' * 5) ≠ 1 / 18) :
  p - q = 0 :=
sorry

end NUMINAMATH_GPT_find_p_minus_q_l1653_165390


namespace NUMINAMATH_GPT_largest_r_satisfying_condition_l1653_165386

theorem largest_r_satisfying_condition :
  ∃ M : ℕ, ∀ (a : ℕ → ℕ) (r : ℝ) (h : ∀ n : ℕ, a n ≤ a (n + 2) ∧ a (n + 2) ≤ Real.sqrt (a n ^ 2 + r * a (n + 1))),
  (∀ n : ℕ, n ≥ M → a (n + 2) = a n) → r = 2 := 
by
  sorry

end NUMINAMATH_GPT_largest_r_satisfying_condition_l1653_165386


namespace NUMINAMATH_GPT_binary_arithmetic_correct_l1653_165340

theorem binary_arithmetic_correct :
  (2^3 + 2^2 + 2^0) + (2^2 + 2^1 + 2^0) - (2^3 + 2^2 + 2^1) + (2^3 + 2^0) + (2^3 + 2^1) = 2^4 + 2^3 + 2^0 :=
by sorry

end NUMINAMATH_GPT_binary_arithmetic_correct_l1653_165340


namespace NUMINAMATH_GPT_evaluate_f_g_l1653_165341

def g (x : ℝ) : ℝ := 3 * x
def f (x : ℝ) : ℝ := x - 6

theorem evaluate_f_g :
  f (g 3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_g_l1653_165341


namespace NUMINAMATH_GPT_overall_average_of_marks_l1653_165336

theorem overall_average_of_marks (n total_boys passed_boys failed_boys avg_passed avg_failed : ℕ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 105)
  (h3 : failed_boys = 15)
  (h4 : total_boys = passed_boys + failed_boys)
  (h5 : avg_passed = 39)
  (h6 : avg_failed = 15) :
  ((passed_boys * avg_passed + failed_boys * avg_failed) / total_boys = 36) :=
by
  sorry

end NUMINAMATH_GPT_overall_average_of_marks_l1653_165336


namespace NUMINAMATH_GPT_max_n_for_polynomial_l1653_165304

theorem max_n_for_polynomial (P : Polynomial ℤ) (hdeg : P.degree = 2022) :
  ∃ n ≤ 2022, ∀ {a : Fin n → ℤ}, 
    (∀ i, P.eval (a i) = i) ↔ n = 2022 :=
by sorry

end NUMINAMATH_GPT_max_n_for_polynomial_l1653_165304


namespace NUMINAMATH_GPT_toby_peanut_butter_servings_l1653_165337

theorem toby_peanut_butter_servings :
  let bread_calories := 100
  let peanut_butter_calories_per_serving := 200
  let total_calories := 500
  let bread_pieces := 1
  ∃ (servings : ℕ), total_calories = (bread_calories * bread_pieces) + (peanut_butter_calories_per_serving * servings) → servings = 2 := by
  sorry

end NUMINAMATH_GPT_toby_peanut_butter_servings_l1653_165337


namespace NUMINAMATH_GPT_max_hot_dogs_with_300_dollars_l1653_165306

def num_hot_dogs (dollars : ℕ) 
  (cost_8 : ℚ) (count_8 : ℕ) 
  (cost_20 : ℚ) (count_20 : ℕ)
  (cost_250 : ℚ) (count_250 : ℕ) : ℕ :=
  sorry

theorem max_hot_dogs_with_300_dollars : 
  num_hot_dogs 300 1.55 8 3.05 20 22.95 250 = 3258 :=
sorry

end NUMINAMATH_GPT_max_hot_dogs_with_300_dollars_l1653_165306


namespace NUMINAMATH_GPT_chair_cost_l1653_165330

/--
Nadine went to a garage sale and spent $56. She bought a table for $34 and 2 chairs.
Each chair cost the same amount.
Prove that one chair cost $11.
-/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) (total_cost : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  total_cost = 56 - 34 →
  total_cost / num_chairs = 11 :=
by
  sorry

end NUMINAMATH_GPT_chair_cost_l1653_165330


namespace NUMINAMATH_GPT_twice_x_minus_3_gt_4_l1653_165333

theorem twice_x_minus_3_gt_4 (x : ℝ) : 2 * x - 3 > 4 :=
sorry

end NUMINAMATH_GPT_twice_x_minus_3_gt_4_l1653_165333


namespace NUMINAMATH_GPT_age_of_son_l1653_165360

theorem age_of_son (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 20 := 
sorry

end NUMINAMATH_GPT_age_of_son_l1653_165360


namespace NUMINAMATH_GPT_smallest_value_of_linear_expression_l1653_165303

theorem smallest_value_of_linear_expression :
  (∃ a, 8 * a^2 + 6 * a + 5 = 7 ∧ (∃ b, b = 3 * a + 2 ∧ ∀ c, (8 * c^2 + 6 * c + 5 = 7 → 3 * c + 2 ≥ b))) → -1 = b :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_linear_expression_l1653_165303


namespace NUMINAMATH_GPT_range_x_plus_y_l1653_165356

theorem range_x_plus_y (x y: ℝ) (h: x^2 + y^2 - 4 * x + 3 = 0) : 
  2 - Real.sqrt 2 ≤ x + y ∧ x + y ≤ 2 + Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_range_x_plus_y_l1653_165356


namespace NUMINAMATH_GPT_percentage_of_500_l1653_165358

theorem percentage_of_500 : (110 / 100) * 500 = 550 := 
  by
  -- Here we would provide the proof (placeholder)
  sorry

end NUMINAMATH_GPT_percentage_of_500_l1653_165358


namespace NUMINAMATH_GPT_closest_point_on_line_l1653_165347

theorem closest_point_on_line :
  ∀ (x y : ℝ), (4, -2) = (4, -2) →
    y = 3 * x - 1 →
    (∃ (p : ℝ × ℝ), p = (-0.5, -2.5) ∧ p = (-0.5, -2.5))
  := by
    -- The proof of the theorem goes here
    sorry

end NUMINAMATH_GPT_closest_point_on_line_l1653_165347


namespace NUMINAMATH_GPT_sum_of_coefficients_is_1_l1653_165393

-- Given conditions:
def polynomial_expansion (x y : ℤ) := (x - 2 * y) ^ 18

-- Proof statement:
theorem sum_of_coefficients_is_1 : (polynomial_expansion 1 1) = 1 := by
  -- The proof itself is omitted as per the instruction
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_is_1_l1653_165393


namespace NUMINAMATH_GPT_oranges_per_box_l1653_165345

theorem oranges_per_box (total_oranges : ℝ) (total_boxes : ℝ) (h1 : total_oranges = 26500) (h2 : total_boxes = 2650) : 
  total_oranges / total_boxes = 10 :=
by 
  sorry

end NUMINAMATH_GPT_oranges_per_box_l1653_165345


namespace NUMINAMATH_GPT_money_problem_l1653_165370

theorem money_problem
  (A B C : ℕ)
  (h1 : A + B + C = 450)
  (h2 : B + C = 350)
  (h3 : C = 100) :
  A + C = 200 :=
by
  sorry

end NUMINAMATH_GPT_money_problem_l1653_165370


namespace NUMINAMATH_GPT_mitzi_amount_brought_l1653_165316

-- Define the amounts spent on different items
def ticket_cost : ℕ := 30
def food_cost : ℕ := 13
def tshirt_cost : ℕ := 23

-- Define the amount of money left
def amount_left : ℕ := 9

-- Define the total amount spent
def total_spent : ℕ :=
  ticket_cost + food_cost + tshirt_cost

-- Define the total amount brought to the amusement park
def amount_brought : ℕ :=
  total_spent + amount_left

-- Prove that the amount of money Mitzi brought to the amusement park is 75
theorem mitzi_amount_brought : amount_brought = 75 := by
  sorry

end NUMINAMATH_GPT_mitzi_amount_brought_l1653_165316


namespace NUMINAMATH_GPT_man_l1653_165373

-- Define the speeds and values given in the problem conditions
def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

-- Define the man's speed in still water as a variable
def man_speed_in_still_water : ℝ := man_speed_with_current - speed_of_current

-- The theorem we need to prove
theorem man's_speed_against_current_is_correct :
  (man_speed_in_still_water - speed_of_current = man_speed_against_current) :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_man_l1653_165373


namespace NUMINAMATH_GPT_find_interval_solution_l1653_165375

def interval_solution : Set ℝ := {x | 2 < x / (3 * x - 7) ∧ x / (3 * x - 7) <= 7}

theorem find_interval_solution (x : ℝ) :
  x ∈ interval_solution ↔
  x ∈ Set.Ioc (49 / 20 : ℝ) (14 / 5 : ℝ) := 
sorry

end NUMINAMATH_GPT_find_interval_solution_l1653_165375


namespace NUMINAMATH_GPT_x5_y5_z5_value_is_83_l1653_165346

noncomputable def find_x5_y5_z5_value (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧ 
  (x^3 + y^3 + z^3 = 15) ∧
  (x^4 + y^4 + z^4 = 35) ∧
  (x^2 + y^2 + z^2 < 10) →
  x^5 + y^5 + z^5 = 83

theorem x5_y5_z5_value_is_83 (x y z : ℝ) :
  find_x5_y5_z5_value x y z :=
  sorry

end NUMINAMATH_GPT_x5_y5_z5_value_is_83_l1653_165346


namespace NUMINAMATH_GPT_parallelogram_area_l1653_165332

def base := 12 -- in meters
def height := 6 -- in meters

theorem parallelogram_area : base * height = 72 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1653_165332


namespace NUMINAMATH_GPT_ratio_alisha_to_todd_is_two_to_one_l1653_165380

-- Definitions
def total_gumballs : ℕ := 45
def todd_gumballs : ℕ := 4
def bobby_gumballs (A : ℕ) : ℕ := 4 * A - 5
def remaining_gumballs : ℕ := 6

-- Condition stating Hector's gumball distribution
def hector_gumballs_distribution (A : ℕ) : Prop :=
  todd_gumballs + A + bobby_gumballs A + remaining_gumballs = total_gumballs

-- Definition for the ratio of the gumballs given to Alisha to Todd
def ratio_alisha_todd (A : ℕ) : ℕ × ℕ :=
  (A / 4, todd_gumballs / 4)

-- Theorem stating the problem
theorem ratio_alisha_to_todd_is_two_to_one : ∃ (A : ℕ), hector_gumballs_distribution A → ratio_alisha_todd A = (2, 1) :=
sorry

end NUMINAMATH_GPT_ratio_alisha_to_todd_is_two_to_one_l1653_165380


namespace NUMINAMATH_GPT_multiple_people_sharing_carriage_l1653_165354

theorem multiple_people_sharing_carriage (x : ℝ) : 
  (x / 3) + 2 = (x - 9) / 2 :=
sorry

end NUMINAMATH_GPT_multiple_people_sharing_carriage_l1653_165354


namespace NUMINAMATH_GPT_proof_problem_l1653_165317

def from_base (b : ℕ) (digits : List ℕ) : ℕ :=
digits.foldr (λ (d acc) => d + b * acc) 0

def problem : Prop :=
  let a := from_base 8 [2, 3, 4, 5] -- 2345 base 8
  let b := from_base 5 [1, 4, 0]    -- 140 base 5
  let c := from_base 4 [1, 0, 3, 2] -- 1032 base 4
  let d := from_base 8 [2, 9, 1, 0] -- 2910 base 8
  let result := (a / b + c - d : ℤ)
  result = -1502

theorem proof_problem : problem :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1653_165317


namespace NUMINAMATH_GPT_kim_knit_sweaters_total_l1653_165350

theorem kim_knit_sweaters_total :
  ∀ (M T W R F : ℕ), 
    M = 8 →
    T = M + 2 →
    W = T - 4 →
    R = T - 4 →
    F = M / 2 →
    M + T + W + R + F = 34 :=
by
  intros M T W R F hM hT hW hR hF
  rw [hM, hT, hW, hR, hF]
  norm_num
  sorry

end NUMINAMATH_GPT_kim_knit_sweaters_total_l1653_165350


namespace NUMINAMATH_GPT_neq_is_necessary_but_not_sufficient_l1653_165377

theorem neq_is_necessary_but_not_sufficient (a b : ℝ) : (a ≠ b) → ¬ (∀ a b : ℝ, (a ≠ b) → (a / b + b / a > 2)) ∧ (∀ a b : ℝ, (a / b + b / a > 2) → (a ≠ b)) :=
by {
    sorry
}

end NUMINAMATH_GPT_neq_is_necessary_but_not_sufficient_l1653_165377


namespace NUMINAMATH_GPT_count_integer_solutions_l1653_165352

theorem count_integer_solutions :
  (2 * 9^2 + 5 * 9 * -4 + 3 * (-4)^2 = 30) →
  ∃ S : Finset (ℤ × ℤ), (∀ x y : ℤ, ((2 * x ^ 2 + 5 * x * y + 3 * y ^ 2 = 30) ↔ (x, y) ∈ S)) ∧ 
  S.card = 16 :=
by sorry

end NUMINAMATH_GPT_count_integer_solutions_l1653_165352


namespace NUMINAMATH_GPT_divide_45_to_get_900_l1653_165325

theorem divide_45_to_get_900 (x : ℝ) (h : 45 / x = 900) : x = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_divide_45_to_get_900_l1653_165325


namespace NUMINAMATH_GPT_calculate_abc_over_def_l1653_165385

theorem calculate_abc_over_def
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  (a * b * c) / (d * e * f) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_abc_over_def_l1653_165385


namespace NUMINAMATH_GPT_average_monthly_balance_is_150_l1653_165328

-- Define the balances for each month
def balance_jan : ℕ := 100
def balance_feb : ℕ := 200
def balance_mar : ℕ := 150
def balance_apr : ℕ := 150

-- Define the number of months
def num_months : ℕ := 4

-- Define the total sum of balances
def total_balance : ℕ := balance_jan + balance_feb + balance_mar + balance_apr

-- Define the average balance
def average_balance : ℕ := total_balance / num_months

-- Goal is to prove that the average monthly balance is 150 dollars
theorem average_monthly_balance_is_150 : average_balance = 150 :=
by
  sorry

end NUMINAMATH_GPT_average_monthly_balance_is_150_l1653_165328


namespace NUMINAMATH_GPT_polynomial_real_roots_l1653_165314

theorem polynomial_real_roots :
  ∀ x : ℝ, (x^4 - 3 * x^3 + 3 * x^2 - x - 6 = 0) ↔ (x = 3 ∨ x = 2 ∨ x = -1) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_real_roots_l1653_165314


namespace NUMINAMATH_GPT_mathematicians_contemporaries_probability_l1653_165372

noncomputable def probability_contemporaries : ℚ :=
  let overlap_area : ℚ := 129600
  let total_area : ℚ := 360000
  overlap_area / total_area

theorem mathematicians_contemporaries_probability :
  probability_contemporaries = 18 / 25 :=
by
  sorry

end NUMINAMATH_GPT_mathematicians_contemporaries_probability_l1653_165372


namespace NUMINAMATH_GPT_fairy_tale_island_counties_l1653_165342

theorem fairy_tale_island_counties :
  let initial_elves := 1
  let initial_dwarves := 1
  let initial_centaurs := 1

  let first_year_elves := initial_elves
  let first_year_dwarves := initial_dwarves * 3
  let first_year_centaurs := initial_centaurs * 3

  let second_year_elves := first_year_elves * 4
  let second_year_dwarves := first_year_dwarves
  let second_year_centaurs := first_year_centaurs * 4

  let third_year_elves := second_year_elves * 6
  let third_year_dwarves := second_year_dwarves * 6
  let third_year_centaurs := second_year_centaurs

  let total_counties := third_year_elves + third_year_dwarves + third_year_centaurs

  total_counties = 54 :=
by
  sorry

end NUMINAMATH_GPT_fairy_tale_island_counties_l1653_165342


namespace NUMINAMATH_GPT_expand_product_l1653_165327

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1653_165327


namespace NUMINAMATH_GPT_largest_prime_factor_of_12321_l1653_165384

-- Definitions based on the given conditions
def n := 12321
def a := 111
def p₁ := 3
def p₂ := 37

-- Given conditions as hypotheses
theorem largest_prime_factor_of_12321 (h1 : n = a^2) (h2 : a = p₁ * p₂) (hp₁_prime : Prime p₁) (hp₂_prime : Prime p₂) :
  p₂ = 37 ∧ ∀ p, Prime p → p ∣ n → p ≤ 37 := 
by 
  sorry

end NUMINAMATH_GPT_largest_prime_factor_of_12321_l1653_165384


namespace NUMINAMATH_GPT_sufficient_condition_for_line_perpendicular_to_plane_l1653_165312

variables {Plane Line : Type}
variables (α β γ : Plane) (m n l : Line)

-- Definitions of perpendicularity and inclusion
def perp (l : Line) (p : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def parallel (p₁ p₂ : Plane) : Prop := sorry -- definition of parallel planes
def incl (l : Line) (p : Plane) : Prop := sorry -- definition of a line being in a plane

-- The given conditions
axiom n_perp_α : perp n α
axiom n_perp_β : perp n β
axiom m_perp_α : perp m α

-- The proof goal
theorem sufficient_condition_for_line_perpendicular_to_plane :
  perp m β :=
by
    sorry

end NUMINAMATH_GPT_sufficient_condition_for_line_perpendicular_to_plane_l1653_165312


namespace NUMINAMATH_GPT_div_equiv_l1653_165349

theorem div_equiv : (0.75 / 25) = (7.5 / 250) :=
by
  sorry

end NUMINAMATH_GPT_div_equiv_l1653_165349


namespace NUMINAMATH_GPT_solve_system_l1653_165315

variable {a b c : ℝ}
variable {x y z : ℝ}
variable {e1 e2 e3 : ℤ} -- Sign variables should be integers to express ±1 more easily 

axiom ax1 : x * (x + y) + z * (x - y) = a
axiom ax2 : y * (y + z) + x * (y - z) = b
axiom ax3 : z * (z + x) + y * (z - x) = c

theorem solve_system :
  (e1 = 1 ∨ e1 = -1) ∧ (e2 = 1 ∨ e2 = -1) ∧ (e3 = 1 ∨ e3 = -1) →
  x = (1/2) * (e1 * Real.sqrt (a + b) - e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) ∧
  y = (1/2) * (e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) - e3 * Real.sqrt (c + a)) ∧
  z = (1/2) * (-e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) :=
sorry -- proof goes here

end NUMINAMATH_GPT_solve_system_l1653_165315


namespace NUMINAMATH_GPT_students_count_l1653_165321

noncomputable def num_students (N T : ℕ) : Prop :=
  T = 72 * N ∧ (T - 200) / (N - 5) = 92

theorem students_count (N T : ℕ) : num_students N T → N = 13 :=
by
  sorry

end NUMINAMATH_GPT_students_count_l1653_165321


namespace NUMINAMATH_GPT_total_books_in_class_l1653_165367

theorem total_books_in_class (Tables : ℕ) (BooksPerTable : ℕ) (TotalBooks : ℕ) 
  (h1 : Tables = 500)
  (h2 : BooksPerTable = (2 * Tables) / 5)
  (h3 : TotalBooks = Tables * BooksPerTable) :
  TotalBooks = 100000 := 
sorry

end NUMINAMATH_GPT_total_books_in_class_l1653_165367


namespace NUMINAMATH_GPT_factor_polynomial_l1653_165326

variable (x : ℝ)

theorem factor_polynomial : (270 * x^3 - 90 * x^2 + 18 * x) = 18 * x * (15 * x^2 - 5 * x + 1) :=
by 
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1653_165326


namespace NUMINAMATH_GPT_unique_solution_of_equation_l1653_165353

theorem unique_solution_of_equation (x y : ℝ) (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_of_equation_l1653_165353


namespace NUMINAMATH_GPT_petya_cannot_form_figure_c_l1653_165324

-- Define the rhombus and its properties, including rotation
noncomputable def is_rotatable_rhombus (r : ℕ) : Prop := sorry

-- Define the larger shapes and their properties in terms of whether they can be formed using rotations of the rhombus.
noncomputable def can_form_figure_a (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_b (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_c (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_d (rhombus : ℕ) : Prop := sorry

-- Statement: Petya cannot form the figure (c) using the rhombus and allowed transformations.
theorem petya_cannot_form_figure_c (rhombus : ℕ) (h : is_rotatable_rhombus rhombus) :
  ¬ can_form_figure_c rhombus := sorry

end NUMINAMATH_GPT_petya_cannot_form_figure_c_l1653_165324


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1653_165369

variable (a : ℕ → ℤ)

def arithmetic_sequence_condition_1 := a 5 = 3
def arithmetic_sequence_condition_2 := a 6 = -2

theorem arithmetic_sequence_sum :
  arithmetic_sequence_condition_1 a →
  arithmetic_sequence_condition_2 a →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1653_165369


namespace NUMINAMATH_GPT_mixed_number_expression_l1653_165308

theorem mixed_number_expression :
  (7 + 1/2 - (5 + 3/4)) * (3 + 1/6 + (2 + 1/8)) = 9 + 25/96 :=
by
  -- here we would provide the proof steps
  sorry

end NUMINAMATH_GPT_mixed_number_expression_l1653_165308


namespace NUMINAMATH_GPT_george_second_half_questions_l1653_165348

noncomputable def george_first_half_questions : ℕ := 6
noncomputable def points_per_question : ℕ := 3
noncomputable def george_final_score : ℕ := 30

theorem george_second_half_questions :
  (george_final_score - (george_first_half_questions * points_per_question)) / points_per_question = 4 :=
by
  sorry

end NUMINAMATH_GPT_george_second_half_questions_l1653_165348
