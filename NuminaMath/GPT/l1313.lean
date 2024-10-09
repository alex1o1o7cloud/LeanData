import Mathlib

namespace star_7_2_l1313_131365

def star (a b : ℕ) := 4 * a - 4 * b

theorem star_7_2 : star 7 2 = 20 := 
by
  sorry

end star_7_2_l1313_131365


namespace part1_part2_l1313_131305

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem part1 (x : ℝ) : |f (-x)| + |f x| ≥ 4 * |x| := 
by
  sorry

theorem part2 (x a : ℝ) (h : |x - a| < 1 / 2) : |f x - f a| < |a| + 5 / 4 := 
by
  sorry

end part1_part2_l1313_131305


namespace find_constant_k_l1313_131302

theorem find_constant_k (k : ℤ) :
    (∀ x : ℝ, -x^2 - (k + 7) * x - 8 = - (x - 2) * (x - 4)) → k = -13 :=
by 
    intros h
    sorry

end find_constant_k_l1313_131302


namespace wood_burned_afternoon_l1313_131367

theorem wood_burned_afternoon (burned_morning burned_afternoon bundles_start bundles_end : ℕ) 
  (h_burned_morning : burned_morning = 4)
  (h_bundles_start : bundles_start = 10) 
  (h_bundles_end : bundles_end = 3)
  (total_burned : bundles_start - bundles_end = burned_morning + burned_afternoon) :
  burned_afternoon = 3 :=
by {
  -- Proof placeholder
  sorry
}

end wood_burned_afternoon_l1313_131367


namespace part1_part2_l1313_131314

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1 (a : ℝ) (h : a = 2) : 
  {x : ℝ | f x a ≥ 4 - abs (x - 4)} = {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 5} :=
by
  sorry

theorem part2 (set_is : {x : ℝ | 1 ≤ x ∧ x ≤ 2}) : 
  ∃ a : ℝ, 
    (∀ x : ℝ, abs (f (2*x + a) a - 2*f x a) ≤ 2 → (1 ≤ x ∧ x ≤ 2)) ∧ 
    a = 3 :=
by
  sorry

end part1_part2_l1313_131314


namespace cube_volume_l1313_131336

-- Define the condition: the surface area of the cube is 54
def surface_area_of_cube (x : ℝ) : Prop := 6 * x^2 = 54

-- Define the theorem that states the volume of the cube given the surface area condition
theorem cube_volume : ∃ (x : ℝ), surface_area_of_cube x ∧ x^3 = 27 := by
  sorry

end cube_volume_l1313_131336


namespace calculate_t_minus_d_l1313_131343

def tom_pays : ℕ := 150
def dorothy_pays : ℕ := 190
def sammy_pays : ℕ := 240
def nancy_pays : ℕ := 320
def total_expenses := tom_pays + dorothy_pays + sammy_pays + nancy_pays
def individual_share := total_expenses / 4
def tom_needs_to_pay := individual_share - tom_pays
def dorothy_needs_to_pay := individual_share - dorothy_pays
def sammy_should_receive := sammy_pays - individual_share
def nancy_should_receive := nancy_pays - individual_share
def t := tom_needs_to_pay
def d := dorothy_needs_to_pay

theorem calculate_t_minus_d : t - d = 40 :=
by
  sorry

end calculate_t_minus_d_l1313_131343


namespace problem1_problem2_problem3_problem4_l1313_131363

theorem problem1 : 12 - (-1) + (-7) = 6 := by
  sorry

theorem problem2 : -3.5 * (-3 / 4) / (7 / 8) = 3 := by
  sorry

theorem problem3 : (1 / 3 - 1 / 6 - 1 / 12) * (-12) = -1 := by
  sorry

theorem problem4 : (-2)^4 / (-4) * (-1/2)^2 - 1^2 = -2 := by
  sorry

end problem1_problem2_problem3_problem4_l1313_131363


namespace smallest_second_term_l1313_131381

theorem smallest_second_term (a d : ℕ) (h1 : 5 * a + 10 * d = 95) (h2 : a > 0) (h3 : d > 0) : 
  a + d = 10 :=
sorry

end smallest_second_term_l1313_131381


namespace average_weight_l1313_131337

theorem average_weight (Ishmael Ponce Jalen : ℝ) 
  (h1 : Ishmael = Ponce + 20) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Jalen = 160) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
by 
  sorry

end average_weight_l1313_131337


namespace scott_runs_84_miles_in_a_month_l1313_131358

-- Define the number of miles Scott runs from Monday to Wednesday in a week.
def milesMonToWed : ℕ := 3 * 3

-- Define the number of miles Scott runs on Thursday and Friday in a week.
def milesThuFri : ℕ := 3 * 2 * 2

-- Define the total number of miles Scott runs in a week.
def totalMilesPerWeek : ℕ := milesMonToWed + milesThuFri

-- Define the number of weeks in a month.
def weeksInMonth : ℕ := 4

-- Define the total number of miles Scott runs in a month.
def totalMilesInMonth : ℕ := totalMilesPerWeek * weeksInMonth

-- Statement to prove that Scott runs 84 miles in a month with 4 weeks.
theorem scott_runs_84_miles_in_a_month : totalMilesInMonth = 84 := by
  -- The proof is omitted for this example.
  sorry

end scott_runs_84_miles_in_a_month_l1313_131358


namespace tan_alpha_add_pi_div_four_l1313_131385

theorem tan_alpha_add_pi_div_four {α : ℝ} (h1 : α ∈ Set.Ioo 0 (Real.pi)) (h2 : Real.cos α = -4/5) :
  Real.tan (α + Real.pi / 4) = 1 / 7 :=
sorry

end tan_alpha_add_pi_div_four_l1313_131385


namespace circumcircle_radius_of_right_triangle_l1313_131334

theorem circumcircle_radius_of_right_triangle (r : ℝ) (BC : ℝ) (R : ℝ) 
  (h1 : r = 3) (h2 : BC = 10) : R = 7.25 := 
by
  sorry

end circumcircle_radius_of_right_triangle_l1313_131334


namespace olivia_earnings_l1313_131306

-- Define Olivia's hourly wage
def wage : ℕ := 9

-- Define the hours worked on each day
def hours_monday : ℕ := 4
def hours_wednesday : ℕ := 3
def hours_friday : ℕ := 6

-- Define the total hours worked
def total_hours : ℕ := hours_monday + hours_wednesday + hours_friday

-- Define the total earnings
def total_earnings : ℕ := total_hours * wage

-- State the theorem
theorem olivia_earnings : total_earnings = 117 :=
by
  sorry

end olivia_earnings_l1313_131306


namespace boys_at_reunion_l1313_131338

theorem boys_at_reunion (n : ℕ) (h : n * (n - 1) = 56) : n = 8 :=
sorry

end boys_at_reunion_l1313_131338


namespace probability_of_target_hit_l1313_131326

theorem probability_of_target_hit  :
  let A_hits := 0.9
  let B_hits := 0.8
  ∃ (P_A P_B : ℝ), 
  P_A = A_hits ∧ P_B = B_hits ∧ 
  (∀ events_independent : Prop, 
   events_independent → P_A * P_B = (0.1) * (0.2)) →
  1 - (0.1 * 0.2) = 0.98
:= 
  sorry

end probability_of_target_hit_l1313_131326


namespace swimming_pool_width_l1313_131391

theorem swimming_pool_width 
  (V : ℝ) (L : ℝ) (B1 : ℝ) (B2 : ℝ) (h : ℝ)
  (h_volume : V = (h / 2) * (B1 + B2) * L) 
  (h_V : V = 270) 
  (h_L : L = 12) 
  (h_B1 : B1 = 1) 
  (h_B2 : B2 = 4) : 
  h = 9 :=
  sorry

end swimming_pool_width_l1313_131391


namespace arccos_sqrt_half_l1313_131354

theorem arccos_sqrt_half : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 := by
  sorry

end arccos_sqrt_half_l1313_131354


namespace trigonometric_identity_l1313_131389

theorem trigonometric_identity 
  (deg7 deg37 deg83 : ℝ)
  (h7 : deg7 = 7) 
  (h37 : deg37 = 37) 
  (h83 : deg83 = 83) 
  : (Real.sin (deg7 * Real.pi / 180) * Real.cos (deg37 * Real.pi / 180) - Real.sin (deg83 * Real.pi / 180) * Real.sin (deg37 * Real.pi / 180) = -1/2) :=
sorry

end trigonometric_identity_l1313_131389


namespace toothpaste_last_day_l1313_131384

theorem toothpaste_last_day (total_toothpaste : ℝ)
  (dad_use_per_brush : ℝ) (dad_brushes_per_day : ℕ)
  (mom_use_per_brush : ℝ) (mom_brushes_per_day : ℕ)
  (anne_use_per_brush : ℝ) (anne_brushes_per_day : ℕ)
  (brother_use_per_brush : ℝ) (brother_brushes_per_day : ℕ)
  (sister_use_per_brush : ℝ) (sister_brushes_per_day : ℕ)
  (grandfather_use_per_brush : ℝ) (grandfather_brushes_per_day : ℕ)
  (guest_use_per_brush : ℝ) (guest_brushes_per_day : ℕ) (guest_days : ℕ)
  (total_usage_per_day : ℝ) :
  total_toothpaste = 80 →
  dad_use_per_brush * dad_brushes_per_day = 16 →
  mom_use_per_brush * mom_brushes_per_day = 12 →
  anne_use_per_brush * anne_brushes_per_day = 8 →
  brother_use_per_brush * brother_brushes_per_day = 4 →
  sister_use_per_brush * sister_brushes_per_day = 2 →
  grandfather_use_per_brush * grandfather_brushes_per_day = 6 →
  guest_use_per_brush * guest_brushes_per_day * guest_days = 6 * 4 →
  total_usage_per_day = 54 →
  80 / 54 = 1 → 
  total_toothpaste / total_usage_per_day = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end toothpaste_last_day_l1313_131384


namespace find_radius_l1313_131353

def setA : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def setB (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem find_radius (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ setA ∧ p ∈ setB r) ↔ (r = 3 ∨ r = 7) :=
by
  sorry

end find_radius_l1313_131353


namespace range_of_a_l1313_131360

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (4 * (x - 1) > 3 * x - 1) ∧ (5 * x > 3 * x + 2 * a) ↔ (x > 3)) ↔ (a ≤ 3) :=
by
  sorry

end range_of_a_l1313_131360


namespace simplify_fraction_l1313_131316

theorem simplify_fraction : 
  1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end simplify_fraction_l1313_131316


namespace floor_sum_even_l1313_131392

theorem floor_sum_even (a b c : ℕ) (h1 : a^2 + b^2 + 1 = c^2) : 
    ((a / 2) + (c / 2)) % 2 = 0 := 
  sorry

end floor_sum_even_l1313_131392


namespace total_chickens_on_farm_l1313_131303

noncomputable def total_chickens (H R : ℕ) : ℕ := H + R

theorem total_chickens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H = 67) : total_chickens H R = 75 := 
by
  sorry

end total_chickens_on_farm_l1313_131303


namespace fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l1313_131319

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

theorem fixed_point_when_a_2_b_neg2 :
  (∃ x : ℝ, f 2 (-2) x = x) → (x = -1 ∨ x = 2) :=
sorry

theorem range_of_a_for_two_fixed_points (a : ℝ) :
  (∀ b : ℝ, a ≠ 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = x1 ∧ f a b x2 = x2)) → (0 < a ∧ a < 2) :=
sorry

end fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l1313_131319


namespace age_sum_squares_l1313_131308

theorem age_sum_squares (a b c : ℕ) (h1 : 5 * a + 2 * b = 3 * c) (h2 : 3 * c^2 = 4 * a^2 + b^2) (h3 : Nat.gcd (Nat.gcd a b) c = 1) : a^2 + b^2 + c^2 = 18 :=
sorry

end age_sum_squares_l1313_131308


namespace quadratic_roots_condition_l1313_131371

theorem quadratic_roots_condition (k : ℝ) : 
  (∀ (r s : ℝ), r + s = -k ∧ r * s = 12 → (r + 3) + (s + 3) = k) → k = 3 := 
by 
  sorry

end quadratic_roots_condition_l1313_131371


namespace arun_age_l1313_131372

variable (A S G M : ℕ)

theorem arun_age (h1 : A - 6 = 18 * G)
                 (h2 : G + 2 = M)
                 (h3 : M = 5)
                 (h4 : S = A - 8) : A = 60 :=
by sorry

end arun_age_l1313_131372


namespace simplify_fractional_expression_l1313_131341

variable {a b c : ℝ}

theorem simplify_fractional_expression 
  (h_nonzero_a : a ≠ 0)
  (h_nonzero_b : b ≠ 0)
  (h_nonzero_c : c ≠ 0)
  (h_sum : a + b + c = 1) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 
  3 / (2 * (-b - c + b * c)) :=
sorry

end simplify_fractional_expression_l1313_131341


namespace total_pizzas_bought_l1313_131321

theorem total_pizzas_bought (slices_small : ℕ) (slices_medium : ℕ) (slices_large : ℕ) 
                            (num_small : ℕ) (num_medium : ℕ) (total_slices : ℕ) :
  slices_small = 6 → 
  slices_medium = 8 → 
  slices_large = 12 → 
  num_small = 4 → 
  num_medium = 5 → 
  total_slices = 136 → 
  (total_slices = num_small * slices_small + num_medium * slices_medium + 72) →
  15 = num_small + num_medium + 6 :=
by
  intros
  sorry

end total_pizzas_bought_l1313_131321


namespace trigonometric_identity_l1313_131318

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) - Real.sin α * Real.cos α = -1 :=
sorry

end trigonometric_identity_l1313_131318


namespace digits_to_replace_l1313_131348

theorem digits_to_replace (a b c d e f : ℕ) :
  (a = 1) →
  (b < 5) →
  (c = 8) →
  (d = 1) →
  (e = 0) →
  (f = 4) →
  (100 * a + 10 * b + c)^2 = 10000 * d + 1000 * e + 100 * f + 10 * f + f :=
  by
    intros ha hb hc hd he hf 
    sorry

end digits_to_replace_l1313_131348


namespace bob_age_sum_digits_l1313_131352

theorem bob_age_sum_digits
  (A B C : ℕ)  -- Define ages for Alice (A), Bob (B), and Carl (C)
  (h1 : C = 2)  -- Carl's age is 2
  (h2 : B = A + 2)  -- Bob is 2 years older than Alice
  (h3 : ∃ n, A = 2 * n ∧ n > 0 ∧ n ≤ 8 )  -- Alice's age is a multiple of Carl's age today, marking the second of the 8 such multiples 
  : ∃ n, (B + n) % (C + n) = 0 ∧ (B + n) = 50 :=  -- Prove that the next time Bob's age is a multiple of Carl's, Bob's age will be 50
sorry

end bob_age_sum_digits_l1313_131352


namespace g_x_squared_plus_2_l1313_131349

namespace PolynomialProof

open Polynomial

noncomputable def g (x : ℚ) : ℚ := sorry

theorem g_x_squared_plus_2 (x : ℚ) (h : g (x^2 - 2) = x^4 - 6*x^2 + 8) :
  g (x^2 + 2) = x^4 + 2*x^2 + 2 :=
sorry

end PolynomialProof

end g_x_squared_plus_2_l1313_131349


namespace find_a_l1313_131333

theorem find_a (x a : ℝ) (h₁ : x = 2) (h₂ : (4 - x) / 2 + a = 4) : a = 3 :=
by
  -- Proof steps will go here
  sorry

end find_a_l1313_131333


namespace school_B_saving_l1313_131387

def cost_A (kg_price : ℚ) (kg_amount : ℚ) : ℚ :=
  kg_price * kg_amount

def effective_kg_B (total_kg : ℚ) (extra_percentage : ℚ) : ℚ :=
  total_kg / (1 + extra_percentage)

def cost_B (kg_price : ℚ) (effective_kg : ℚ) : ℚ :=
  kg_price * effective_kg

theorem school_B_saving
  (kg_amount : ℚ) (price_A: ℚ) (discount: ℚ) (extra_percentage : ℚ) 
  (expected_saving : ℚ)
  (h1 : kg_amount = 56)
  (h2 : price_A = 8.06)
  (h3 : discount = 0.56)
  (h4 : extra_percentage = 0.05)
  (h5 : expected_saving = 51.36) :
  cost_A price_A kg_amount - cost_B (price_A - discount) (effective_kg_B kg_amount extra_percentage) = expected_saving := 
by 
  sorry

end school_B_saving_l1313_131387


namespace total_spent_by_pete_and_raymond_l1313_131300

def pete_initial_amount := 250
def pete_spending_on_stickers := 4 * 5
def pete_spending_on_candy := 3 * 10
def pete_spending_on_toy_car := 2 * 25
def pete_spending_on_keychain := 5
def pete_total_spent := pete_spending_on_stickers + pete_spending_on_candy + pete_spending_on_toy_car + pete_spending_on_keychain
def raymond_initial_amount := 250
def raymond_left_dimes := 7 * 10
def raymond_left_quarters := 4 * 25
def raymond_left_nickels := 5 * 5
def raymond_left_pennies := 3 * 1
def raymond_total_left := raymond_left_dimes + raymond_left_quarters + raymond_left_nickels + raymond_left_pennies
def raymond_total_spent := raymond_initial_amount - raymond_total_left
def total_spent := pete_total_spent + raymond_total_spent

theorem total_spent_by_pete_and_raymond : total_spent = 157 := by
  have h1 : pete_total_spent = 105 := sorry
  have h2 : raymond_total_spent = 52 := sorry
  exact sorry

end total_spent_by_pete_and_raymond_l1313_131300


namespace average_first_21_multiples_of_6_l1313_131331

-- Define the arithmetic sequence and its conditions.
def arithmetic_sequence (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

-- Define the problem statement.
theorem average_first_21_multiples_of_6 :
  let a1 := 6
  let d := 6
  let n := 21
  let an := arithmetic_sequence a1 d n
  (a1 + an) / 2 = 66 := by
  sorry

end average_first_21_multiples_of_6_l1313_131331


namespace parrots_are_red_l1313_131332

-- Definitions for fractions.
def total_parrots : ℕ := 160
def green_fraction : ℚ := 5 / 8
def blue_fraction : ℚ := 1 / 4

-- Definition for calculating the number of parrots.
def number_of_green_parrots : ℚ := green_fraction * total_parrots
def number_of_blue_parrots : ℚ := blue_fraction * total_parrots
def number_of_red_parrots : ℚ := total_parrots - number_of_green_parrots - number_of_blue_parrots

-- The theorem to prove.
theorem parrots_are_red : number_of_red_parrots = 20 := by
  -- Proof is omitted.
  sorry

end parrots_are_red_l1313_131332


namespace largest_four_digit_number_divisible_by_2_5_9_11_l1313_131398

theorem largest_four_digit_number_divisible_by_2_5_9_11 : ∃ n : ℤ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∀ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d → a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (n % 2 = 0) ∧ 
  (n % 5 = 0) ∧ 
  (n % 9 = 0) ∧ 
  (n % 11 = 0) ∧ 
  (n = 8910) := 
by
  sorry

end largest_four_digit_number_divisible_by_2_5_9_11_l1313_131398


namespace correct_choice_2point5_l1313_131379

def set_M : Set ℝ := {x | -2 < x ∧ x < 3}

theorem correct_choice_2point5 : 2.5 ∈ set_M :=
by {
  -- sorry is added to close the proof for now
  sorry
}

end correct_choice_2point5_l1313_131379


namespace frac_sum_is_one_l1313_131327

theorem frac_sum_is_one (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end frac_sum_is_one_l1313_131327


namespace lines_do_not_form_triangle_l1313_131344

noncomputable def line1 (x y : ℝ) := 3 * x - y + 2 = 0
noncomputable def line2 (x y : ℝ) := 2 * x + y + 3 = 0
noncomputable def line3 (m x y : ℝ) := m * x + y = 0

theorem lines_do_not_form_triangle (m : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y) →
  (∀ x y : ℝ, (line1 x y → line3 m x y) ∨ (line2 x y → line3 m x y) ∨ 
    (line1 x y ∧ line2 x y → line3 m x y)) →
  (m = -3 ∨ m = 2 ∨ m = -1) :=
by
  sorry

end lines_do_not_form_triangle_l1313_131344


namespace problem1_problem2_l1313_131320

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

-- Problem 1: If ¬p is true, find the range of values for x
theorem problem1 {x : ℝ} (h : ¬ p x) : x > 2 ∨ x < -1 :=
by
  -- Proof omitted
  sorry

-- Problem 2: If ¬q is a sufficient but not necessary condition for ¬p, find the range of values for m
theorem problem2 {m : ℝ} (h : ∀ x : ℝ, ¬ q x m → ¬ p x) : m > 1 ∨ m < -2 :=
by
  -- Proof omitted
  sorry

end problem1_problem2_l1313_131320


namespace exactly_one_root_in_interval_l1313_131324

theorem exactly_one_root_in_interval (p q : ℝ) (h : q * (q + p + 1) < 0) :
    ∃ x : ℝ, 0 < x ∧ x < 1 ∧ (x^2 + p * x + q = 0) := sorry

end exactly_one_root_in_interval_l1313_131324


namespace bottle_cost_l1313_131382

-- Definitions of the conditions
def total_cost := 30
def wine_extra_cost := 26

-- Statement of the problem in Lean 4
theorem bottle_cost : 
  ∃ x : ℕ, (x + (x + wine_extra_cost) = total_cost) ∧ x = 2 :=
by
  sorry

end bottle_cost_l1313_131382


namespace find_reciprocal_square_sum_of_roots_l1313_131315

theorem find_reciprocal_square_sum_of_roots :
  ∃ (a b c : ℝ), 
    (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (a^3 - 6 * a^2 - a + 3 = 0) ∧ 
    (b^3 - 6 * b^2 - b + 3 = 0) ∧ 
    (c^3 - 6 * c^2 - c + 3 = 0) ∧ 
    (a + b + c = 6) ∧
    (a * b + b * c + c * a = -1) ∧
    (a * b * c = -3)) 
    → (1 / a^2 + 1 / b^2 + 1 / c^2 = 37 / 9) :=
sorry

end find_reciprocal_square_sum_of_roots_l1313_131315


namespace wade_final_profit_l1313_131312

theorem wade_final_profit :
  let tips_per_customer_friday := 2.00
  let customers_friday := 28
  let tips_per_customer_saturday := 2.50
  let customers_saturday := 3 * customers_friday
  let tips_per_customer_sunday := 1.50
  let customers_sunday := 36
  let cost_ingredients_per_hotdog := 1.25
  let price_per_hotdog := 4.00
  let truck_maintenance_daily_cost := 50.00
  let total_taxes := 150.00
  let revenue_tips_friday := tips_per_customer_friday * customers_friday
  let revenue_hotdogs_friday := customers_friday * price_per_hotdog
  let cost_ingredients_friday := customers_friday * cost_ingredients_per_hotdog
  let revenue_friday := revenue_tips_friday + revenue_hotdogs_friday
  let total_costs_friday := cost_ingredients_friday + truck_maintenance_daily_cost
  let profit_friday := revenue_friday - total_costs_friday
  let revenue_tips_saturday := tips_per_customer_saturday * customers_saturday
  let revenue_hotdogs_saturday := customers_saturday * price_per_hotdog
  let cost_ingredients_saturday := customers_saturday * cost_ingredients_per_hotdog
  let revenue_saturday := revenue_tips_saturday + revenue_hotdogs_saturday
  let total_costs_saturday := cost_ingredients_saturday + truck_maintenance_daily_cost
  let profit_saturday := revenue_saturday - total_costs_saturday
  let revenue_tips_sunday := tips_per_customer_sunday * customers_sunday
  let revenue_hotdogs_sunday := customers_sunday * price_per_hotdog
  let cost_ingredients_sunday := customers_sunday * cost_ingredients_per_hotdog
  let revenue_sunday := revenue_tips_sunday + revenue_hotdogs_sunday
  let total_costs_sunday := cost_ingredients_sunday + truck_maintenance_daily_cost
  let profit_sunday := revenue_sunday - total_costs_sunday
  let total_profit := profit_friday + profit_saturday + profit_sunday
  let final_profit := total_profit - total_taxes
  final_profit = 427.00 :=
by
  sorry

end wade_final_profit_l1313_131312


namespace total_coughs_after_20_minutes_l1313_131335

def coughs_in_n_minutes (rate_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  rate_per_minute * minutes

def total_coughs (georgia_rate_per_minute : ℕ) (minutes : ℕ) (multiplier : ℕ) : ℕ :=
  let georgia_coughs := coughs_in_n_minutes georgia_rate_per_minute minutes
  let robert_rate_per_minute := georgia_rate_per_minute * multiplier
  let robert_coughs := coughs_in_n_minutes robert_rate_per_minute minutes
  georgia_coughs + robert_coughs

theorem total_coughs_after_20_minutes :
  total_coughs 5 20 2 = 300 :=
by
  sorry

end total_coughs_after_20_minutes_l1313_131335


namespace avg_visitors_is_correct_l1313_131309

-- Define the number of days in the month
def days_in_month : ℕ := 30

-- Define the average number of visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average number of visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Define the number of Sundays in the month
def sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def other_days_in_month : ℕ := days_in_month - sundays_in_month

-- Define the total visitors on Sundays
def total_visitors_sundays : ℕ := sundays_in_month * avg_visitors_sunday

-- Define the total visitors on other days
def total_visitors_other_days : ℕ := other_days_in_month * avg_visitors_other_days

-- Define the total number of visitors in the month
def total_visitors : ℕ := total_visitors_sundays + total_visitors_other_days

-- Define the average number of visitors per day
def avg_visitors_per_day : ℕ := total_visitors / days_in_month

-- The theorem to prove
theorem avg_visitors_is_correct : avg_visitors_per_day = 276 := by
  sorry

end avg_visitors_is_correct_l1313_131309


namespace answer_keys_count_l1313_131355

theorem answer_keys_count 
  (test_questions : ℕ)
  (true_answers : ℕ)
  (false_answers : ℕ)
  (min_score : ℕ)
  (conditions : test_questions = 10 ∧ true_answers = 5 ∧ false_answers = 5 ∧ min_score >= 4) :
  ∃ (count : ℕ), count = 22 := by
  sorry

end answer_keys_count_l1313_131355


namespace largest_number_is_A_l1313_131362

-- Definitions of the numbers
def numA := 8.45678
def numB := 8.456777777 -- This should be represented properly with an infinite sequence in a real formal proof
def numC := 8.456767676 -- This should be represented properly with an infinite sequence in a real formal proof
def numD := 8.456756756 -- This should be represented properly with an infinite sequence in a real formal proof
def numE := 8.456745674 -- This should be represented properly with an infinite sequence in a real formal proof

-- Lean statement to prove that numA is the largest number
theorem largest_number_is_A : numA > numB ∧ numA > numC ∧ numA > numD ∧ numA > numE :=
by
  -- Proof not provided, sorry to skip
  sorry

end largest_number_is_A_l1313_131362


namespace snake_count_l1313_131307

def neighborhood : Type := {n : ℕ // n = 200}

def percentage (total : ℕ) (percent : ℕ) : ℕ := total * percent / 100

def owns_only_dogs (total : ℕ) : ℕ := percentage total 13
def owns_only_cats (total : ℕ) : ℕ := percentage total 10
def owns_only_snakes (total : ℕ) : ℕ := percentage total 5
def owns_only_rabbits (total : ℕ) : ℕ := percentage total 7
def owns_only_birds (total : ℕ) : ℕ := percentage total 3
def owns_only_exotic (total : ℕ) : ℕ := percentage total 6
def owns_dogs_and_cats (total : ℕ) : ℕ := percentage total 8
def owns_dogs_cats_exotic (total : ℕ) : ℕ := percentage total 9
def owns_cats_and_snakes (total : ℕ) : ℕ := percentage total 4
def owns_cats_and_birds (total : ℕ) : ℕ := percentage total 2
def owns_snakes_and_rabbits (total : ℕ) : ℕ := percentage total 5
def owns_snakes_and_birds (total : ℕ) : ℕ := percentage total 3
def owns_rabbits_and_birds (total : ℕ) : ℕ := percentage total 1
def owns_all_except_snakes (total : ℕ) : ℕ := percentage total 2
def owns_all_except_birds (total : ℕ) : ℕ := percentage total 1
def owns_three_with_exotic (total : ℕ) : ℕ := percentage total 11
def owns_only_chameleons (total : ℕ) : ℕ := percentage total 3
def owns_only_hedgehogs (total : ℕ) : ℕ := percentage total 2

def exotic_pet_owners (total : ℕ) : ℕ :=
  owns_only_exotic total + owns_dogs_cats_exotic total + owns_all_except_snakes total +
  owns_all_except_birds total + owns_three_with_exotic total + owns_only_chameleons total +
  owns_only_hedgehogs total

def exotic_pet_owners_with_snakes (total : ℕ) : ℕ :=
  percentage (exotic_pet_owners total) 25

def total_snake_owners (total : ℕ) : ℕ :=
  owns_only_snakes total + owns_cats_and_snakes total +
  owns_snakes_and_rabbits total + owns_snakes_and_birds total +
  exotic_pet_owners_with_snakes total

theorem snake_count (nh : neighborhood) : total_snake_owners (nh.val) = 51 :=
by
  sorry

end snake_count_l1313_131307


namespace scientific_notation_of_116_million_l1313_131374

theorem scientific_notation_of_116_million : 116000000 = 1.16 * 10^7 :=
sorry

end scientific_notation_of_116_million_l1313_131374


namespace stickers_decorate_l1313_131390

theorem stickers_decorate (initial_stickers bought_stickers birthday_stickers given_stickers remaining_stickers stickers_used : ℕ)
    (h1 : initial_stickers = 20)
    (h2 : bought_stickers = 12)
    (h3 : birthday_stickers = 20)
    (h4 : given_stickers = 5)
    (h5 : remaining_stickers = 39) :
    (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers = stickers_used) →
    stickers_used = 8 
:= by {sorry}

end stickers_decorate_l1313_131390


namespace complementary_angles_positive_difference_l1313_131351

/-- Two angles are complementary if their sum is 90 degrees.
    If the measures of these angles are in the ratio 3:1,
    then their positive difference is 45 degrees. -/
theorem complementary_angles_positive_difference (x : ℝ) (h1 : (3 * x) + x = 90) :
  abs ((3 * x) - x) = 45 :=
by
  sorry

end complementary_angles_positive_difference_l1313_131351


namespace range_of_m_l1313_131329

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then (1 / 3)^(-x) - 2 
  else 2 * Real.log x / Real.log 3

theorem range_of_m :
  {m : ℝ | f m > 1} = {m : ℝ | m < -Real.sqrt 3} ∪ {m : ℝ | 1 < m} :=
by
  sorry

end range_of_m_l1313_131329


namespace greatest_drop_in_price_is_august_l1313_131373

-- Define the months and their respective price changes
def price_changes : List (String × ℝ) :=
  [("January", -1.00), ("February", 1.50), ("March", -3.00), ("April", 2.50), 
   ("May", -0.75), ("June", -2.25), ("July", 1.00), ("August", -4.00)]

-- Define the statement that August has the greatest drop in price
theorem greatest_drop_in_price_is_august :
  ∀ month ∈ price_changes, month.snd ≤ -4.00 → month.fst = "August" :=
by
  sorry

end greatest_drop_in_price_is_august_l1313_131373


namespace vectors_orthogonal_dot_product_l1313_131386

theorem vectors_orthogonal_dot_product (y : ℤ) :
  (3 * -2) + (4 * y) + (-1 * 5) = 0 → y = 11 / 4 :=
by
  sorry

end vectors_orthogonal_dot_product_l1313_131386


namespace determine_a_l1313_131370

theorem determine_a (a b c : ℕ) (h_b : b = 5) (h_c : c = 6) (h_order : c > b ∧ b > a ∧ a > 2) :
(a - 2) * (b - 2) * (c - 2) = 4 * (b - 2) + 4 * (c - 2) → a = 4 :=
by 
  sorry

end determine_a_l1313_131370


namespace tamara_total_earnings_l1313_131394

-- Definitions derived from the conditions in the problem statement.
def pans : ℕ := 2
def pieces_per_pan : ℕ := 8
def price_per_piece : ℕ := 2

-- Theorem stating the required proof problem.
theorem tamara_total_earnings : 
  (pans * pieces_per_pan * price_per_piece) = 32 :=
by
  sorry

end tamara_total_earnings_l1313_131394


namespace intersection_of_M_and_N_l1313_131330

open Set

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, y = 2 - |x|}

theorem intersection_of_M_and_N : M ∩ N = {y | 0 ≤ y ∧ y ≤ 2} :=
by sorry

end intersection_of_M_and_N_l1313_131330


namespace vehicles_travelled_last_year_l1313_131364

theorem vehicles_travelled_last_year (V : ℕ) : 
  (∀ (x : ℕ), (96 : ℕ) * (V / 100000000) = 2880) → V = 3000000000 := 
by 
  sorry

end vehicles_travelled_last_year_l1313_131364


namespace outlinedSquareDigit_l1313_131317

-- We define the conditions for three-digit powers of 2 and 3
def isThreeDigitPowerOf (base : ℕ) (n : ℕ) : Prop :=
  let power := base ^ n
  power >= 100 ∧ power < 1000

-- Define the sets of three-digit powers of 2 and 3
def threeDigitPowersOf2 : List ℕ := [128, 256, 512]
def threeDigitPowersOf3 : List ℕ := [243, 729]

-- Define the condition that the digit in the outlined square should be common as a last digit in any power of 2 and 3 that's three-digit long
def commonLastDigitOfPowers (a b : List ℕ) : Option ℕ :=
  let aLastDigits := a.map (λ x => x % 10)
  let bLastDigits := b.map (λ x => x % 10)
  (aLastDigits.inter bLastDigits).head?

theorem outlinedSquareDigit : (commonLastDigitOfPowers threeDigitPowersOf2 threeDigitPowersOf3) = some 3 :=
by
  sorry

end outlinedSquareDigit_l1313_131317


namespace proof_problem_l1313_131356

-- Definitions based on the conditions from the problem
def optionA (A : Set α) : Prop := ∅ ∩ A = ∅

def optionC : Prop := { y | ∃ x, y = 1 / x } = { z | ∃ t, z = 1 / t }

-- The main theorem statement
theorem proof_problem (A : Set α) : optionA A ∧ optionC := by
  -- Placeholder for the proof
  sorry

end proof_problem_l1313_131356


namespace distinct_units_digits_of_squares_mod_6_l1313_131361

theorem distinct_units_digits_of_squares_mod_6 : 
  ∃ (s : Finset ℕ), s = {0, 1, 4, 3} ∧ s.card = 4 :=
by
  sorry

end distinct_units_digits_of_squares_mod_6_l1313_131361


namespace max_lateral_surface_area_l1313_131325

theorem max_lateral_surface_area (x y : ℝ) (h₁ : x + y = 10) : 
  2 * π * x * y ≤ 50 * π :=
by
  sorry

end max_lateral_surface_area_l1313_131325


namespace sum_of_possible_values_l1313_131301

variable (N K : ℝ)

theorem sum_of_possible_values (h1 : N ≠ 0) (h2 : N - (3 / N) = K) : N + (K / N) = K := 
sorry

end sum_of_possible_values_l1313_131301


namespace tallest_building_model_height_l1313_131350

def height_campus : ℝ := 120
def volume_campus : ℝ := 30000
def volume_model : ℝ := 0.03
def height_model : ℝ := 1.2

theorem tallest_building_model_height :
  (volume_campus / volume_model)^(1/3) = (height_campus / height_model) :=
by
  sorry

end tallest_building_model_height_l1313_131350


namespace amaya_movie_watching_time_l1313_131395

theorem amaya_movie_watching_time :
  let t1 := 30 + 5
  let t2 := 20 + 7
  let t3 := 10 + 12
  let t4 := 15 + 8
  let t5 := 25 + 15
  let t6 := 15 + 10
  t1 + t2 + t3 + t4 + t5 + t6 = 172 :=
by
  sorry

end amaya_movie_watching_time_l1313_131395


namespace first_day_exceeds_target_l1313_131345

-- Definitions based on the conditions
def initial_count : ℕ := 5
def daily_growth_factor : ℕ := 3
def target_count : ℕ := 200

-- The proof problem in Lean
theorem first_day_exceeds_target : ∃ n : ℕ, 5 * 3 ^ n > 200 ∧ ∀ m < n, ¬ (5 * 3 ^ m > 200) :=
by
  sorry

end first_day_exceeds_target_l1313_131345


namespace calculation_correct_l1313_131368

def expression : ℝ := 200 * 375 * 0.0375 * 5

theorem calculation_correct : expression = 14062.5 := 
by
  sorry

end calculation_correct_l1313_131368


namespace point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l1313_131357

-- Question (1): Proving that the point (-2,0) lies on the graph
theorem point_on_graph (k : ℝ) (hk : k ≠ 0) : k * (-2 + 2) = 0 := 
by sorry

-- Question (2): Finding the value of k given a shifted graph passing through a point
theorem find_k_shifted_graph_passing (k : ℝ) : (k * (1 + 2) + 2 = -2) → k = -4/3 := 
by sorry

-- Question (3): Proving the range of k for the function's y-intercept within given limits
theorem y_axis_intercept_range (k : ℝ) (hk : -2 < 2 * k ∧ 2 * k < 0) : -1 < k ∧ k < 0 := 
by sorry

end point_on_graph_find_k_shifted_graph_passing_y_axis_intercept_range_l1313_131357


namespace six_digit_number_contains_7_l1313_131304

theorem six_digit_number_contains_7
  (a b k : ℤ)
  (h1 : 100 ≤ 7 * a + k ∧ 7 * a + k < 1000)
  (h2 : 100 ≤ 7 * b + k ∧ 7 * b + k < 1000) :
  7 ∣ (1000 * (7 * a + k) + (7 * b + k)) :=
by
  sorry

end six_digit_number_contains_7_l1313_131304


namespace find_nonnegative_solutions_l1313_131340

theorem find_nonnegative_solutions :
  ∀ (x y z : ℕ), 5^x + 7^y = 2^z ↔ (x = 0 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 5) :=
by
  sorry

end find_nonnegative_solutions_l1313_131340


namespace money_left_eq_l1313_131369

theorem money_left_eq :
  let initial_money := 56
  let notebooks := 7
  let cost_per_notebook := 4
  let books := 2
  let cost_per_book := 7
  let money_left := initial_money - (notebooks * cost_per_notebook + books * cost_per_book)
  money_left = 14 :=
by
  sorry

end money_left_eq_l1313_131369


namespace line_intersects_parabola_once_l1313_131323

theorem line_intersects_parabola_once (k : ℝ) : 
  (∃! y : ℝ, -3 * y^2 + 2 * y + 7 = k) ↔ k = 22 / 3 :=
by {
  sorry
}

end line_intersects_parabola_once_l1313_131323


namespace find_k_l1313_131388

noncomputable def a_squared : ℝ := 9
noncomputable def b_squared (k : ℝ) : ℝ := 4 + k
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

noncomputable def c_squared_1 (k : ℝ) : ℝ := 5 - k
noncomputable def c_squared_2 (k : ℝ) : ℝ := k - 5

theorem find_k (k : ℝ) :
  (eccentricity (Real.sqrt (c_squared_1 k)) (Real.sqrt a_squared) = 4 / 5 →
   k = -19 / 25) ∨ 
  (eccentricity (Real.sqrt (c_squared_2 k)) (Real.sqrt (b_squared k)) = 4 / 5 →
   k = 21) :=
sorry

end find_k_l1313_131388


namespace average_marks_correct_l1313_131377

-- Definitions used in the Lean 4 statement, reflecting conditions in the problem
def total_students_class1 : ℕ := 25 
def average_marks_class1 : ℕ := 40 
def total_students_class2 : ℕ := 30 
def average_marks_class2 : ℕ := 60 

-- Calculate the total marks for both classes
def total_marks_class1 : ℕ := total_students_class1 * average_marks_class1 
def total_marks_class2 : ℕ := total_students_class2 * average_marks_class2 
def total_marks : ℕ := total_marks_class1 + total_marks_class2 

-- Calculate the total number of students
def total_students : ℕ := total_students_class1 + total_students_class2 

-- Define the average of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_students 

-- The theorem to be proved
theorem average_marks_correct : average_marks_all_students = (2800 : ℚ) / 55 := 
by 
  sorry

end average_marks_correct_l1313_131377


namespace total_bill_calculation_l1313_131376

theorem total_bill_calculation (n : ℕ) (amount_per_person : ℝ) (total_amount : ℝ) :
  n = 9 → amount_per_person = 514.19 → total_amount = 4627.71 → 
  n * amount_per_person = total_amount :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_bill_calculation_l1313_131376


namespace bob_day3_miles_l1313_131396

noncomputable def total_miles : ℕ := 70
noncomputable def day1_miles : ℕ := total_miles * 20 / 100
noncomputable def remaining_after_day1 : ℕ := total_miles - day1_miles
noncomputable def day2_miles : ℕ := remaining_after_day1 * 50 / 100
noncomputable def remaining_after_day2 : ℕ := remaining_after_day1 - day2_miles
noncomputable def day3_miles : ℕ := remaining_after_day2

theorem bob_day3_miles : day3_miles = 28 :=
by
  -- Insert proof here
  sorry

end bob_day3_miles_l1313_131396


namespace find_g_l1313_131393

noncomputable def g (x : ℝ) : ℝ := 2 - 4 * x

theorem find_g :
  g 0 = 2 ∧ (∀ x y : ℝ, g (x * y) = g ((3 * x ^ 2 + y ^ 2) / 4) + 3 * (x - y) ^ 2) → ∀ x : ℝ, g x = 2 - 4 * x :=
by
  sorry

end find_g_l1313_131393


namespace circumscribed_quadrilateral_converse_arithmetic_progression_l1313_131378

theorem circumscribed_quadrilateral (a b c d : ℝ) (k : ℝ) (h1 : b = a + k) (h2 : d = a + 2 * k) (h3 : c = a + 3 * k) :
  a + c = b + d :=
by
  sorry

theorem converse_arithmetic_progression (a b c d : ℝ) (h : a + c = b + d) :
  ∃ k : ℝ, b = a + k ∧ d = a + 2 * k ∧ c = a + 3 * k :=
by
  sorry

end circumscribed_quadrilateral_converse_arithmetic_progression_l1313_131378


namespace eval_polynomial_at_3_l1313_131380

def f (x : ℝ) : ℝ := 2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

theorem eval_polynomial_at_3 : f 3 = 130 :=
by
  -- proof can be completed here following proper steps or using Horner's method
  sorry

end eval_polynomial_at_3_l1313_131380


namespace cos_beta_zero_l1313_131339

theorem cos_beta_zero (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) (h3 : Real.cos α = 1 / 2) (h4 : Real.cos (α + β) = -1 / 2) : Real.cos β = 0 :=
sorry

end cos_beta_zero_l1313_131339


namespace average_infection_rate_l1313_131383

theorem average_infection_rate (x : ℝ) : 
  (1 + x + x * (1 + x) = 196) → x = 13 :=
by
  intro h
  sorry

end average_infection_rate_l1313_131383


namespace rate_per_sq_meter_is_900_l1313_131322

/-- The length of the room L is 7 (meters). -/
def L : ℝ := 7

/-- The width of the room W is 4.75 (meters). -/
def W : ℝ := 4.75

/-- The total cost of paving the floor is Rs. 29,925. -/
def total_cost : ℝ := 29925

/-- The rate per square meter for the slabs is Rs. 900. -/
theorem rate_per_sq_meter_is_900 :
  total_cost / (L * W) = 900 :=
by
  sorry

end rate_per_sq_meter_is_900_l1313_131322


namespace salmon_total_l1313_131359

def num_male : ℕ := 712261
def num_female : ℕ := 259378
def num_total : ℕ := 971639

theorem salmon_total :
  num_male + num_female = num_total :=
by
  -- proof will be provided here
  sorry

end salmon_total_l1313_131359


namespace average_of_side_lengths_of_squares_l1313_131313

theorem average_of_side_lengths_of_squares :
  let side_length1 := Real.sqrt 25
  let side_length2 := Real.sqrt 64
  let side_length3 := Real.sqrt 144
  (side_length1 + side_length2 + side_length3) / 3 = 25 / 3 :=
by
  sorry

end average_of_side_lengths_of_squares_l1313_131313


namespace customers_in_other_countries_l1313_131399

-- Definitions for conditions
def total_customers : ℕ := 7422
def us_customers : ℕ := 723

-- Statement to prove
theorem customers_in_other_countries : total_customers - us_customers = 6699 :=
by
  sorry

end customers_in_other_countries_l1313_131399


namespace reduced_rates_apply_two_days_l1313_131310

-- Definition of total hours in a week
def total_hours_in_week : ℕ := 7 * 24

-- Given fraction of the week with reduced rates
def reduced_rate_fraction : ℝ := 0.6428571428571429

-- Total hours covered by reduced rates
def reduced_rate_hours : ℝ := reduced_rate_fraction * total_hours_in_week

-- Hours per day with reduced rates on weekdays (8 p.m. to 8 a.m.)
def hours_weekday_night : ℕ := 12

-- Total weekdays with reduced rates
def total_weekdays : ℕ := 5

-- Total reduced rate hours on weekdays
def reduced_rate_hours_weekdays : ℕ := total_weekdays * hours_weekday_night

-- Remaining hours for 24 hour reduced rates
def remaining_reduced_rate_hours : ℝ := reduced_rate_hours - reduced_rate_hours_weekdays

-- Prove that the remaining reduced rate hours correspond to exactly 2 full days
theorem reduced_rates_apply_two_days : remaining_reduced_rate_hours = 2 * 24 := 
by
  sorry

end reduced_rates_apply_two_days_l1313_131310


namespace even_function_and_monotonicity_l1313_131342

noncomputable def f (x : ℝ) : ℝ := sorry

theorem even_function_and_monotonicity (f_symm : ∀ x : ℝ, f x = f (-x))
  (f_inc_neg : ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → x1 ≤ 0 → x2 ≤ 0 → f x1 < f x2)
  (n : ℕ) (hn : n > 0) :
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) := 
sorry

end even_function_and_monotonicity_l1313_131342


namespace correct_operation_l1313_131347

theorem correct_operation :
  (∀ a : ℝ, (a^5 * a^3 = a^15) = false) ∧
  (∀ a : ℝ, (a^5 - a^3 = a^2) = false) ∧
  (∀ a : ℝ, ((-a^5)^2 = a^10) = true) ∧
  (∀ a : ℝ, (a^6 / a^3 = a^2) = false) :=
by
  sorry

end correct_operation_l1313_131347


namespace sufficient_but_not_necessary_condition_l1313_131366

theorem sufficient_but_not_necessary_condition :
  (∀ (x : ℝ), x^2 - 2 * x < 0 → 0 < x ∧ x < 4)
  ∧ ¬(∀ (x : ℝ), 0 < x ∧ x < 4 → x^2 - 2 * x < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1313_131366


namespace M_is_listed_correctly_l1313_131311

noncomputable def M : Set ℕ := { m | ∃ n : ℕ+, 3 / (5 - m : ℝ) = n }

theorem M_is_listed_correctly : M = { 2, 4 } :=
by
  sorry

end M_is_listed_correctly_l1313_131311


namespace population_total_l1313_131375

theorem population_total (total_population layers : ℕ) (ratio_A ratio_B ratio_C : ℕ) 
(sample_capacity : ℕ) (prob_ab_in_C : ℚ) 
(h1 : ratio_A = 3)
(h2 : ratio_B = 6)
(h3 : ratio_C = 1)
(h4 : sample_capacity = 20)
(h5 : prob_ab_in_C = 1 / 21)
(h6 : total_population = 10 * ratio_C) :
  total_population = 70 := 
by 
  sorry

end population_total_l1313_131375


namespace yanna_kept_apples_l1313_131328

-- Define the given conditions
def initial_apples : ℕ := 60
def percentage_given_to_zenny : ℝ := 0.40
def percentage_given_to_andrea : ℝ := 0.25

-- Prove the main statement
theorem yanna_kept_apples : 
  let apples_given_to_zenny := (percentage_given_to_zenny * initial_apples)
  let apples_remaining_after_zenny := (initial_apples - apples_given_to_zenny)
  let apples_given_to_andrea := (percentage_given_to_andrea * apples_remaining_after_zenny)
  let apples_kept := (apples_remaining_after_zenny - apples_given_to_andrea)
  apples_kept = 27 :=
by
  sorry

end yanna_kept_apples_l1313_131328


namespace find_number_l1313_131397

theorem find_number (x : ℤ) (h : 27 + 2 * x = 39) : x = 6 :=
sorry

end find_number_l1313_131397


namespace Martha_needs_54_cakes_l1313_131346

theorem Martha_needs_54_cakes :
  let n_children := 3
  let n_cakes_per_child := 18
  let n_cakes_total := 54
  n_cakes_total = n_children * n_cakes_per_child :=
by
  sorry

end Martha_needs_54_cakes_l1313_131346
