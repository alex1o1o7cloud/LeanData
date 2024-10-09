import Mathlib

namespace boatman_current_speed_and_upstream_time_l1032_103220

variables (v : ℝ) (v_T : ℝ) (t_up : ℝ) (t_total : ℝ) (dist : ℝ) (d1 : ℝ) (d2 : ℝ)

theorem boatman_current_speed_and_upstream_time
  (h1 : dist = 12.5)
  (h2 : d1 = 3)
  (h3 : d2 = 5)
  (h4 : t_total = 8)
  (h5 : ∀ t, t = d1 / (v - v_T))
  (h6 : ∀ t, t = d2 / (v + v_T))
  (h7 : dist / (v - v_T) + dist / (v + v_T) = t_total) :
  v_T = 5 / 6 ∧ t_up = 5 := by
  sorry

end boatman_current_speed_and_upstream_time_l1032_103220


namespace distinct_values_in_expression_rearrangement_l1032_103211

theorem distinct_values_in_expression_rearrangement : 
  ∀ (exp : ℕ), exp = 3 → 
  (∃ n : ℕ, n = 3 ∧ 
    let a := exp ^ (exp ^ exp)
    let b := exp ^ ((exp ^ exp) ^ exp)
    let c := ((exp ^ exp) ^ exp) ^ exp
    let d := (exp ^ (exp ^ exp)) ^ exp
    let e := (exp ^ exp) ^ (exp ^ exp)
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :=
by
  sorry

end distinct_values_in_expression_rearrangement_l1032_103211


namespace remainder_3x_minus_6_divides_P_l1032_103254

def P(x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 8 * x^4 + 3 * x^3 - 5
def D(x : ℝ) : ℝ := 3 * x - 6

theorem remainder_3x_minus_6_divides_P :
  P 2 = 915 :=
by
  sorry

end remainder_3x_minus_6_divides_P_l1032_103254


namespace remainder_127_14_l1032_103282

theorem remainder_127_14 : ∃ r : ℤ, r = 127 - (14 * 9) ∧ r = 1 := by
  sorry

end remainder_127_14_l1032_103282


namespace min_colors_correctness_l1032_103285

noncomputable def min_colors_no_monochromatic_cycle (n : ℕ) : ℕ :=
if n ≤ 2 then 1 else 2

theorem min_colors_correctness (n : ℕ) (h₀ : n > 0) :
  (min_colors_no_monochromatic_cycle n = 1 ∧ n ≤ 2) ∨
  (min_colors_no_monochromatic_cycle n = 2 ∧ n ≥ 3) :=
by
  sorry

end min_colors_correctness_l1032_103285


namespace find_initial_number_l1032_103213

theorem find_initial_number (x : ℝ) (h : x + 12.808 - 47.80600000000004 = 3854.002) : x = 3889 := by
  sorry

end find_initial_number_l1032_103213


namespace each_spider_eats_seven_bugs_l1032_103250

theorem each_spider_eats_seven_bugs (initial_bugs : ℕ) (reduction_rate : ℚ) (spiders_introduced : ℕ) (bugs_left : ℕ) (result : ℕ)
  (h1 : initial_bugs = 400)
  (h2 : reduction_rate = 0.80)
  (h3 : spiders_introduced = 12)
  (h4 : bugs_left = 236)
  (h5 : result = initial_bugs * (4 / 5) - bugs_left) :
  (result / spiders_introduced) = 7 :=
by
  sorry

end each_spider_eats_seven_bugs_l1032_103250


namespace matches_C_won_l1032_103218

variable (A_wins B_wins D_wins total_matches wins_C : ℕ)

theorem matches_C_won 
  (hA : A_wins = 3)
  (hB : B_wins = 1)
  (hD : D_wins = 0)
  (htot : total_matches = 6)
  (h_sum_wins: A_wins + B_wins + D_wins + wins_C = total_matches)
  : wins_C = 2 :=
by
  sorry

end matches_C_won_l1032_103218


namespace car_speed_first_hour_l1032_103295

theorem car_speed_first_hour (speed1 speed2 avg_speed : ℕ) (h1 : speed2 = 70) (h2 : avg_speed = 95) :
  (2 * avg_speed) = speed1 + speed2 → speed1 = 120 :=
by
  sorry

end car_speed_first_hour_l1032_103295


namespace remainder_sum_div7_l1032_103286

theorem remainder_sum_div7 (a b c : ℕ) (h1 : a * b * c ≡ 2 [MOD 7])
  (h2 : 3 * c ≡ 4 [MOD 7])
  (h3 : 4 * b ≡ 2 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_sum_div7_l1032_103286


namespace find_a_2016_l1032_103234

theorem find_a_2016 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 1, a (n + 1) = 3 * S n)
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)):
  a 2016 = 3 * 4 ^ 2014 := 
by 
  sorry

end find_a_2016_l1032_103234


namespace trailing_zeros_1_to_100_l1032_103235

def count_multiples (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def trailing_zeros_in_range (n : ℕ) : ℕ :=
  let multiples_of_5 := count_multiples n 5
  let multiples_of_25 := count_multiples n 25
  multiples_of_5 + multiples_of_25

theorem trailing_zeros_1_to_100 : trailing_zeros_in_range 100 = 24 := by
  sorry

end trailing_zeros_1_to_100_l1032_103235


namespace milkshake_cost_proof_l1032_103230

-- Define the problem
def milkshake_cost (total_money : ℕ) (hamburger_cost : ℕ) (n_hamburgers : ℕ)
                   (n_milkshakes : ℕ) (remaining_money : ℕ) : ℕ :=
  let total_hamburgers_cost := n_hamburgers * hamburger_cost
  let money_after_hamburgers := total_money - total_hamburgers_cost
  let milkshake_cost := (money_after_hamburgers - remaining_money) / n_milkshakes
  milkshake_cost

-- Statement to prove
theorem milkshake_cost_proof : milkshake_cost 120 4 8 6 70 = 3 :=
by
  -- we skip the proof steps as the problem statement does not require it
  sorry

end milkshake_cost_proof_l1032_103230


namespace double_angle_cosine_l1032_103261

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l1032_103261


namespace bicycle_stock_decrease_l1032_103258

-- Define the conditions and the problem
theorem bicycle_stock_decrease (m : ℕ) (jan_to_oct_decrease june_to_oct_decrease monthly_decrease : ℕ) 
  (h1: monthly_decrease = 4)
  (h2: jan_to_oct_decrease = 36)
  (h3: june_to_oct_decrease = 4 * monthly_decrease):
  m * monthly_decrease = jan_to_oct_decrease - june_to_oct_decrease → m = 5 := 
by
  sorry

end bicycle_stock_decrease_l1032_103258


namespace lilly_daily_savings_l1032_103208

-- Conditions
def days_until_birthday : ℕ := 22
def flowers_to_buy : ℕ := 11
def cost_per_flower : ℕ := 4

-- Definition we want to prove
def total_cost : ℕ := flowers_to_buy * cost_per_flower
def daily_savings : ℕ := total_cost / days_until_birthday

theorem lilly_daily_savings : daily_savings = 2 := by
  sorry

end lilly_daily_savings_l1032_103208


namespace income_day_3_is_750_l1032_103260

-- Define the given incomes for the specific days
def income_day_1 : ℝ := 250
def income_day_2 : ℝ := 400
def income_day_4 : ℝ := 400
def income_day_5 : ℝ := 500

-- Define the total number of days and the average income over these days
def total_days : ℝ := 5
def average_income : ℝ := 460

-- Define the total income based on the average
def total_income : ℝ := total_days * average_income

-- Define the income on the third day
def income_day_3 : ℝ := total_income - (income_day_1 + income_day_2 + income_day_4 + income_day_5)

-- Claim: The income on the third day is $750
theorem income_day_3_is_750 : income_day_3 = 750 := by
  sorry

end income_day_3_is_750_l1032_103260


namespace problem_AD_l1032_103201

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin x + Real.cos x

open Real

theorem problem_AD :
  (∀ x, 0 < x ∧ x < π / 4 → f x < f (x + 0.01) ∧ g x < g (x + 0.01)) ∧
  (∃ x, x = π / 4 ∧ f x + g x = 1 / 2 + sqrt 2) :=
by
  sorry

end problem_AD_l1032_103201


namespace div_condition_l1032_103202

theorem div_condition (N : ℤ) : (∃ k : ℤ, N^2 - 71 = k * (7 * N + 55)) ↔ (N = 57 ∨ N = -8) := 
by
  sorry

end div_condition_l1032_103202


namespace quadratic_completeness_l1032_103289

noncomputable def quad_eqn : Prop :=
  ∃ b c : ℤ, (∀ x : ℝ, (x^2 - 10 * x + 15 = 0) ↔ ((x + b)^2 = c)) ∧ b + c = 5

theorem quadratic_completeness : quad_eqn :=
sorry

end quadratic_completeness_l1032_103289


namespace alpha_beta_range_l1032_103205

theorem alpha_beta_range (α β : ℝ) (h1 : - (π / 2) < α) (h2 : α < β) (h3 : β < π) : 
- 3 * (π / 2) < α - β ∧ α - β < 0 :=
by
  sorry

end alpha_beta_range_l1032_103205


namespace remainder_of_76_pow_k_mod_7_is_6_l1032_103249

theorem remainder_of_76_pow_k_mod_7_is_6 (k : ℕ) (hk : k % 2 = 1) : (76 ^ k) % 7 = 6 :=
sorry

end remainder_of_76_pow_k_mod_7_is_6_l1032_103249


namespace infinite_expressible_terms_l1032_103240

theorem infinite_expressible_terms
  (a : ℕ → ℕ)
  (h1 : ∀ n, a n < a (n + 1)) :
  ∃ f : ℕ → ℕ, (∀ n, a (f n) = (f n).succ * a 1 + (f n).succ.succ * a 2) ∧
    ∀ i j, i ≠ j → f i ≠ f j :=
by
  sorry

end infinite_expressible_terms_l1032_103240


namespace discrim_of_quadratic_eqn_l1032_103274

theorem discrim_of_quadratic_eqn : 
  let a := 3
  let b := -2
  let c := -1
  b^2 - 4 * a * c = 16 := 
by
  sorry

end discrim_of_quadratic_eqn_l1032_103274


namespace number_of_unanswered_questions_l1032_103219

theorem number_of_unanswered_questions (n p q : ℕ) (h1 : p = 8) (h2 : q = 5) (h3 : n = 20)
(h4: ∃ s, s % 13 = 0) (hy : y = 0 ∨ y = 13) : 
  ∃ k, k = 20 ∨ k = 7 := by
  sorry

end number_of_unanswered_questions_l1032_103219


namespace flower_beds_fraction_l1032_103273

noncomputable def area_triangle (leg: ℝ) : ℝ := (leg * leg) / 2
noncomputable def area_rectangle (length width: ℝ) : ℝ := length * width
noncomputable def area_trapezoid (a b height: ℝ) : ℝ := ((a + b) * height) / 2

theorem flower_beds_fraction : 
  ∀ (leg len width a b height total_length: ℝ),
    a = 30 →
    b = 40 →
    height = 6 →
    total_length = 60 →
    leg = 5 →
    len = 20 →
    width = 5 →
    (area_rectangle len width + 2 * area_triangle leg) / (area_trapezoid a b height + area_rectangle len width) = 125 / 310 :=
by
  intros
  sorry

end flower_beds_fraction_l1032_103273


namespace unique_root_exists_maximum_value_lnx_l1032_103255

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 / Real.exp x

theorem unique_root_exists (k : ℝ) :
  ∃ a, a = 1 ∧ (∃ x ∈ Set.Ioo k (k+1), f x = g x) :=
sorry

theorem maximum_value_lnx (p q : ℝ) :
  (∃ x, (x = min p q) ∧ Real.log x = ( 4 / Real.exp 2 )) :=
sorry

end unique_root_exists_maximum_value_lnx_l1032_103255


namespace quadratic_value_at_point_a_l1032_103294

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

open Real

theorem quadratic_value_at_point_a
  (a b c : ℝ)
  (axis : ℝ)
  (sym : ∀ x, quadratic a b c (2 * axis - x) = quadratic a b c x)
  (at_zero : quadratic a b c 0 = -3) :
  quadratic a b c 20 = -3 := by
  -- proof steps would go here
  sorry

end quadratic_value_at_point_a_l1032_103294


namespace triangle_sides_from_rhombus_l1032_103221

variable (m p q : ℝ)

def is_triangle_side_lengths (BC AC AB : ℝ) :=
  (BC = p + q) ∧
  (AC = m * (p + q) / p) ∧
  (AB = m * (p + q) / q)

theorem triangle_sides_from_rhombus :
  ∃ BC AC AB : ℝ, is_triangle_side_lengths m p q BC AC AB :=
by
  use p + q
  use m * (p + q) / p
  use m * (p + q) / q
  sorry

end triangle_sides_from_rhombus_l1032_103221


namespace walking_time_l1032_103290

theorem walking_time (v : ℕ) (d : ℕ) (h1 : v = 10) (h2 : d = 4) : 
    ∃ (T : ℕ), T = 24 := 
by
  sorry

end walking_time_l1032_103290


namespace convert_base_8_to_7_l1032_103284

def convert_base_8_to_10 (n : Nat) : Nat :=
  let d2 := n / 100 % 10
  let d1 := n / 10 % 10
  let d0 := n % 10
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def convert_base_10_to_7 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else 
    let rec helper (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else helper (n / 7) ((n % 7) :: acc)
    helper n []

def represent_in_base_7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem convert_base_8_to_7 :
  represent_in_base_7 (convert_base_10_to_7 (convert_base_8_to_10 653)) = 1150 :=
by
  sorry

end convert_base_8_to_7_l1032_103284


namespace find_square_tiles_l1032_103283

variable {s p : ℕ}

theorem find_square_tiles (h1 : s + p = 30) (h2 : 4 * s + 5 * p = 110) : s = 20 :=
by
  sorry

end find_square_tiles_l1032_103283


namespace find_x_l1032_103256

theorem find_x (x : ℝ) : x * 2.25 - (5 * 0.85) / 2.5 = 5.5 → x = 3.2 :=
by
  sorry

end find_x_l1032_103256


namespace arithmetic_common_difference_l1032_103246

-- Defining the arithmetic sequence sum function S
def S (a₁ a₂ a₃ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if n = 3 then a₁ + a₂ + a₃
  else if n = 2 then a₁ + a₂
  else 0

-- The Lean statement for the problem
theorem arithmetic_common_difference (a₁ a₂ a₃ d : ℝ) 
  (h1 : a₂ = a₁ + d) 
  (h2 : a₃ = a₂ + d)
  (h3 : 2 * S a₁ a₂ a₃ d 3 = 3 * S a₁ a₂ a₃ d 2 + 6) :
  d = 2 :=
by
  sorry

end arithmetic_common_difference_l1032_103246


namespace red_paint_four_times_blue_paint_total_painted_faces_is_1625_l1032_103237

/-- Given a structure of twenty-five layers of cubes -/
def structure_layers := 25

/-- The number of painted faces from each vertical view -/
def vertical_faces_per_view : ℕ :=
  (structure_layers * (structure_layers + 1)) / 2

/-- The total number of red-painted faces (4 vertical views) -/
def total_red_faces : ℕ :=
  4 * vertical_faces_per_view

/-- The total number of blue-painted faces (1 top view) -/
def total_blue_faces : ℕ :=
  vertical_faces_per_view

theorem red_paint_four_times_blue_paint :
  total_red_faces = 4 * total_blue_faces :=
by sorry

theorem total_painted_faces_is_1625 :
  (4 * vertical_faces_per_view + vertical_faces_per_view) = 1625 :=
by sorry

end red_paint_four_times_blue_paint_total_painted_faces_is_1625_l1032_103237


namespace base_conversion_l1032_103252

theorem base_conversion (b2_to_b10_step : 101101 = 1 * 2 ^ 5 + 0 * 2 ^ 4 + 1 * 2 ^ 3 + 1 * 2 ^ 2 + 0 * 2 + 1)
  (b10_to_b7_step1 : 45 / 7 = 6) (b10_to_b7_step2 : 45 % 7 = 3) (b10_to_b7_step3 : 6 / 7 = 0) (b10_to_b7_step4 : 6 % 7 = 6) :
  101101 = 45 ∧ 45 = 63 :=
by {
  -- Conversion steps from the proof will be filled in here
  sorry
}

end base_conversion_l1032_103252


namespace maximum_value_of_function_l1032_103209

theorem maximum_value_of_function :
  ∀ (x : ℝ), -2 < x ∧ x < 0 → x + 1 / x ≤ -2 :=
by
  sorry

end maximum_value_of_function_l1032_103209


namespace general_term_of_sequence_l1032_103288

theorem general_term_of_sequence (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 4 * n) : 
  a n = 2 * n - 5 :=
by
  -- Proof can be completed here
  sorry

end general_term_of_sequence_l1032_103288


namespace not_always_possible_to_predict_winner_l1032_103224

def football_championship (teams : Fin 16 → ℕ) : Prop :=
  ∃ i j : Fin 16, i ≠ j ∧ teams i = teams j ∧
  ∀ pairs : Fin 16 → Fin 16 × Fin 16,
  (∀ k : Fin 8, (pairs k).fst ≠ (pairs k).snd ∧
               teams (pairs k).fst ≠ teams (pairs k).snd) ∨
  ∃ k : Fin 8, (pairs k).fst = i ∧ (pairs k).snd = j

theorem not_always_possible_to_predict_winner :
  ∀ teams : Fin 16 → ℕ, (∃ i j : Fin 16, i ≠ j ∧ teams i = teams j) →
  ∃ pairs : Fin 16 → Fin 16 × Fin 16,
  (∃ k : Fin 8, teams (pairs k).fst = 15 ∧ teams (pairs k).snd = 15) ↔
  ¬ ∀ pairs : Fin 16 → Fin 16 × Fin 16,
  (∀ k : Fin 8, (pairs k).fst ≠ (pairs k).snd ∧ teams (pairs k).fst ≠ teams (pairs k).snd) :=
by
  sorry

end not_always_possible_to_predict_winner_l1032_103224


namespace residue_of_11_pow_2048_mod_19_l1032_103276

theorem residue_of_11_pow_2048_mod_19 :
  (11 ^ 2048) % 19 = 16 := 
by
  sorry

end residue_of_11_pow_2048_mod_19_l1032_103276


namespace remainder_of_sum_l1032_103272

theorem remainder_of_sum (h1 : 9375 % 5 = 0) (h2 : 9376 % 5 = 1) (h3 : 9377 % 5 = 2) (h4 : 9378 % 5 = 3) :
  (9375 + 9376 + 9377 + 9378) % 5 = 1 :=
by
  sorry

end remainder_of_sum_l1032_103272


namespace team_sports_competed_l1032_103298

theorem team_sports_competed (x : ℕ) (n : ℕ) 
  (h1 : (97 + n) / x = 90) 
  (h2 : (73 + n) / x = 87) : 
  x = 8 := 
by sorry

end team_sports_competed_l1032_103298


namespace payment_to_Y_is_227_27_l1032_103253

-- Define the conditions
def total_payment_per_week (x y : ℝ) : Prop :=
  x + y = 500

def x_payment_is_120_percent_of_y (x y : ℝ) : Prop :=
  x = 1.2 * y

-- Formulate the problem as a theorem to be proven
theorem payment_to_Y_is_227_27 (Y : ℝ) (X : ℝ) 
  (h1 : total_payment_per_week X Y) 
  (h2 : x_payment_is_120_percent_of_y X Y) : 
  Y = 227.27 :=
by
  sorry

end payment_to_Y_is_227_27_l1032_103253


namespace rate_of_interest_l1032_103236

theorem rate_of_interest (R : ℝ) (h : 5000 * 2 * R / 100 + 3000 * 4 * R / 100 = 2200) : R = 10 := by
  sorry

end rate_of_interest_l1032_103236


namespace evaluate_g_at_3_l1032_103259

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

theorem evaluate_g_at_3 : g 3 = 79 := by
  sorry

end evaluate_g_at_3_l1032_103259


namespace hansel_album_duration_l1032_103292

theorem hansel_album_duration 
    (initial_songs : ℕ)
    (additional_songs : ℕ)
    (duration_per_song : ℕ)
    (h_initial : initial_songs = 25)
    (h_additional : additional_songs = 10)
    (h_duration : duration_per_song = 3):
    initial_songs * duration_per_song + additional_songs * duration_per_song = 105 := 
by
  sorry

end hansel_album_duration_l1032_103292


namespace problem_statement_l1032_103200

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 65 / 2) + 5 / 2)

theorem problem_statement :
  ∃ a b c : ℕ, (x ^ 100 = 2 * x ^ 98 + 16 * x ^ 96 + 13 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 42) ∧ (a + b + c = 337) :=
by
  sorry

end problem_statement_l1032_103200


namespace andy_cavity_per_candy_cane_l1032_103245

theorem andy_cavity_per_candy_cane 
  (cavities_per_candy_cane : ℝ)
  (candy_caned_from_parents : ℝ := 2)
  (candy_caned_each_teacher : ℝ := 3)
  (num_teachers : ℝ := 4)
  (allowance_factor : ℝ := 1/7)
  (total_cavities : ℝ := 16) :
  let total_given_candy : ℝ := candy_caned_from_parents + candy_caned_each_teacher * num_teachers
  let total_bought_candy : ℝ := allowance_factor * total_given_candy
  let total_candy : ℝ := total_given_candy + total_bought_candy
  total_candy / total_cavities = cavities_per_candy_cane :=
by
  sorry

end andy_cavity_per_candy_cane_l1032_103245


namespace chickens_count_l1032_103203

def total_animals := 13
def total_legs := 44
def legs_per_chicken := 2
def legs_per_buffalo := 4

theorem chickens_count : 
  (∃ c b : ℕ, c + b = total_animals ∧ legs_per_chicken * c + legs_per_buffalo * b = total_legs ∧ c = 4) :=
by
  sorry

end chickens_count_l1032_103203


namespace new_average_commission_is_250_l1032_103226

-- Definitions based on the problem conditions
def C : ℝ := 1000
def n : ℝ := 6
def increase_in_average_commission : ℝ := 150

-- Theorem stating the new average commission is $250
theorem new_average_commission_is_250 (x : ℝ) (h1 : x + increase_in_average_commission = (5 * x + C) / n) :
  x + increase_in_average_commission = 250 := by
  sorry

end new_average_commission_is_250_l1032_103226


namespace range_of_a_l1032_103244

theorem range_of_a (a m : ℝ) (hp : 3 * a < m ∧ m < 4 * a) 
  (hq : 1 < m ∧ m < 3 / 2) :
  1 / 3 ≤ a ∧ a ≤ 3 / 8 :=
by
  sorry

end range_of_a_l1032_103244


namespace total_time_spent_l1032_103207

variable (B I E M EE ST ME : ℝ)

def learn_basic_rules : ℝ := B
def learn_intermediate_level : ℝ := I
def learn_expert_level : ℝ := E
def learn_master_level : ℝ := M
def endgame_exercises : ℝ := EE
def middle_game_strategy_tactics : ℝ := ST
def mentoring : ℝ := ME

theorem total_time_spent :
  B = 2 →
  I = 75 * B →
  E = 50 * (B + I) →
  M = 30 * E →
  EE = 0.25 * I →
  ST = 2 * EE →
  ME = 0.5 * E →
  B + I + E + M + EE + ST + ME = 235664.5 :=
by
  intros hB hI hE hM hEE hST hME
  rw [hB, hI, hE, hM, hEE, hST, hME]
  sorry

end total_time_spent_l1032_103207


namespace train_length_approx_500_l1032_103251

noncomputable def length_of_train (speed_km_per_hr : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  speed_m_per_s * time_sec

theorem train_length_approx_500 :
  length_of_train 120 15 = 500 :=
by
  sorry

end train_length_approx_500_l1032_103251


namespace find_k_l1032_103242

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 1 / x + 5
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := 2 * x^2 - k

theorem find_k (k : ℝ) : 
  (f 3 - g 3 k = 6) → k = -23/3 := 
by
  sorry

end find_k_l1032_103242


namespace maximum_area_l1032_103277

variable {l w : ℝ}

theorem maximum_area (h1 : l + w = 200) (h2 : l ≥ 90) (h3 : w ≥ 50) (h4 : l ≤ 2 * w) : l * w ≤ 10000 :=
sorry

end maximum_area_l1032_103277


namespace john_total_spending_l1032_103291

def t_shirt_price : ℝ := 20
def num_t_shirts : ℝ := 3
def t_shirt_offer_discount : ℝ := 0.50
def t_shirt_total_cost : ℝ := (2 * t_shirt_price) + (t_shirt_price * t_shirt_offer_discount)

def pants_price : ℝ := 50
def num_pants : ℝ := 2
def pants_total_cost : ℝ := pants_price * num_pants

def jacket_original_price : ℝ := 80
def jacket_discount : ℝ := 0.25
def jacket_total_cost : ℝ := jacket_original_price * (1 - jacket_discount)

def hat_price : ℝ := 15

def shoes_original_price : ℝ := 60
def shoes_discount : ℝ := 0.10
def shoes_total_cost : ℝ := shoes_original_price * (1 - shoes_discount)

def clothes_tax_rate : ℝ := 0.05
def shoes_tax_rate : ℝ := 0.08

def clothes_total_cost : ℝ := t_shirt_total_cost + pants_total_cost + jacket_total_cost + hat_price
def total_cost_before_tax : ℝ := clothes_total_cost + shoes_total_cost

def clothes_tax : ℝ := clothes_total_cost * clothes_tax_rate
def shoes_tax : ℝ := shoes_total_cost * shoes_tax_rate

def total_cost_including_tax : ℝ := total_cost_before_tax + clothes_tax + shoes_tax

theorem john_total_spending :
  total_cost_including_tax = 294.57 := by
  sorry

end john_total_spending_l1032_103291


namespace integer_divisibility_l1032_103278

theorem integer_divisibility (m n : ℕ) (hm : m > 1) (hn : n > 1) (h1 : n ∣ 4^m - 1) (h2 : 2^m ∣ n - 1) : n = 2^m + 1 :=
by sorry

end integer_divisibility_l1032_103278


namespace blue_crayons_l1032_103239

variables (B G : ℕ)

theorem blue_crayons (h1 : 24 = 8 + B + G + 6) (h2 : G = (2 / 3) * B) : B = 6 :=
by 
-- This is where the proof would go
sorry

end blue_crayons_l1032_103239


namespace area_of_red_flowers_is_54_l1032_103227

noncomputable def total_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def red_yellow_area (total : ℝ) : ℝ :=
  total / 2

noncomputable def red_area (red_yellow : ℝ) : ℝ :=
  red_yellow / 2

theorem area_of_red_flowers_is_54 :
  total_area 18 12 / 2 / 2 = 54 := 
  by
    sorry

end area_of_red_flowers_is_54_l1032_103227


namespace cubic_inequality_l1032_103275

theorem cubic_inequality (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0) (hne : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end cubic_inequality_l1032_103275


namespace equivalent_form_l1032_103241

theorem equivalent_form (x y : ℝ) (h : y = x + 1/x) :
  (x^4 + x^3 - 3*x^2 + x + 2 = 0) ↔ (x^2 * (y^2 + y - 5) = 0) :=
sorry

end equivalent_form_l1032_103241


namespace relationship_between_x_x_squared_and_x_cubed_l1032_103231

theorem relationship_between_x_x_squared_and_x_cubed (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : x < x^3 ∧ x^3 < x^2 :=
by
  sorry

end relationship_between_x_x_squared_and_x_cubed_l1032_103231


namespace quadratic_roots_solution_l1032_103279

theorem quadratic_roots_solution (x : ℝ) (h : x > 0) (h_roots : 7 * x^2 - 8 * x - 6 = 0) : (x = 6 / 7) ∨ (x = 1) :=
sorry

end quadratic_roots_solution_l1032_103279


namespace eggs_problem_solution_l1032_103264

theorem eggs_problem_solution :
  ∃ (n x : ℕ), 
  (120 * n = 206 * x) ∧
  (n = 103) ∧
  (x = 60) :=
by sorry

end eggs_problem_solution_l1032_103264


namespace p_necessary_not_sufficient_q_l1032_103263

theorem p_necessary_not_sufficient_q (x : ℝ) : (|x| = 2) → (x = 2) → (|x| = 2 ∧ (x ≠ 2 ∨ x = -2)) := by
  intros h_p h_q
  sorry

end p_necessary_not_sufficient_q_l1032_103263


namespace visitors_surveyed_l1032_103238

-- Given definitions
def total_visitors : ℕ := 400
def visitors_not_enjoyed_nor_understood : ℕ := 100
def E := total_visitors / 2
def U := total_visitors / 2

-- Using condition that 3/4th visitors enjoyed and understood
def enjoys_and_understands := (3 * total_visitors) / 4

-- Assert the equivalence of total number of visitors calculation
theorem visitors_surveyed:
  total_visitors = enjoys_and_understands + visitors_not_enjoyed_nor_understood :=
by
  sorry

end visitors_surveyed_l1032_103238


namespace find_a1_plus_a2_l1032_103267

theorem find_a1_plus_a2 (x : ℝ) (a0 a1 a2 a3 : ℝ) 
  (h : (1 - 2/x)^3 = a0 + a1 * (1/x) + a2 * (1/x)^2 + a3 * (1/x)^3) : 
  a1 + a2 = 6 :=
by
  sorry

end find_a1_plus_a2_l1032_103267


namespace irr_sqrt6_l1032_103299

open Real

theorem irr_sqrt6 : ¬ ∃ (q : ℚ), (↑q : ℝ) = sqrt 6 := by
  sorry

end irr_sqrt6_l1032_103299


namespace inverse_h_l1032_103225

-- definitions of f, g, and h
def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := 3 * x + 7
def h (x : ℝ) : ℝ := f (g x)

-- statement of the problem
theorem inverse_h (x : ℝ) : (∃ y : ℝ, h y = x) ∧ ∀ y : ℝ, h y = x → y = (x - 23) / 12 :=
by
  sorry

end inverse_h_l1032_103225


namespace kitchen_upgrade_total_cost_l1032_103216

-- Defining the given conditions
def num_cabinet_knobs : ℕ := 18
def cost_per_cabinet_knob : ℚ := 2.50

def num_drawer_pulls : ℕ := 8
def cost_per_drawer_pull : ℚ := 4

-- Definition of the total cost function
def total_cost : ℚ :=
  (num_cabinet_knobs * cost_per_cabinet_knob) + (num_drawer_pulls * cost_per_drawer_pull)

-- The theorem to prove the total cost is $77.00
theorem kitchen_upgrade_total_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_total_cost_l1032_103216


namespace john_total_spent_is_correct_l1032_103204

noncomputable def john_spent_total (original_cost : ℝ) (discount_rate : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let discounted_cost := original_cost - (discount_rate / 100 * original_cost)
  let cost_with_tax := discounted_cost + (sales_tax_rate / 100 * discounted_cost)
  let lightsaber_cost := 2 * original_cost
  let lightsaber_cost_with_tax := lightsaber_cost + (sales_tax_rate / 100 * lightsaber_cost)
  cost_with_tax + lightsaber_cost_with_tax

theorem john_total_spent_is_correct :
  john_spent_total 1200 20 8 = 3628.80 :=
by
  sorry

end john_total_spent_is_correct_l1032_103204


namespace f_one_f_a_f_f_a_l1032_103206

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3

theorem f_one : f 1 = 5 := by
  sorry

theorem f_a (a : ℝ) : f a = 2 * a + 3 := by
  sorry

theorem f_f_a (a : ℝ) : f (f a) = 4 * a + 9 := by
  sorry

end f_one_f_a_f_f_a_l1032_103206


namespace angle_bisector_length_l1032_103212

open Real
open Complex

-- Definitions for the problem
def side_lengths (AC BC : ℝ) : Prop :=
  AC = 6 ∧ BC = 9

def angle_C (angle : ℝ) : Prop :=
  angle = 120

-- Main statement to prove
theorem angle_bisector_length (AC BC angle x : ℝ)
  (h1 : side_lengths AC BC)
  (h2 : angle_C angle) :
  x = 18 / 5 :=
  sorry

end angle_bisector_length_l1032_103212


namespace coin_same_side_probability_l1032_103210

noncomputable def probability_same_side_5_tosses (p : ℚ) := (p ^ 5) + (p ^ 5)

theorem coin_same_side_probability : probability_same_side_5_tosses (1/2) = 1/16 := by
  sorry

end coin_same_side_probability_l1032_103210


namespace valid_three_digit_numbers_count_l1032_103265

def count_three_digit_numbers : ℕ := 900

def count_invalid_numbers : ℕ := (90 + 90 - 9)

def count_valid_three_digit_numbers : ℕ := 900 - (90 + 90 - 9)

theorem valid_three_digit_numbers_count :
  count_valid_three_digit_numbers = 729 :=
by
  show 900 - (90 + 90 - 9) = 729
  sorry

end valid_three_digit_numbers_count_l1032_103265


namespace cost_difference_per_square_inch_l1032_103271

theorem cost_difference_per_square_inch (width1 height1 width2 height2 : ℕ) (cost1 cost2 : ℕ)
  (h_size1 : width1 = 24 ∧ height1 = 16)
  (h_cost1 : cost1 = 672)
  (h_size2 : width2 = 48 ∧ height2 = 32)
  (h_cost2 : cost2 = 1152) :
  (cost1 / (width1 * height1) : ℚ) - (cost2 / (width2 * height2) : ℚ) = 1 := 
by
  sorry

end cost_difference_per_square_inch_l1032_103271


namespace fraction_of_students_with_buddy_l1032_103262

theorem fraction_of_students_with_buddy (t s : ℕ) (h1 : (t / 4) = (3 * s / 5)) :
  (t / 4 + 3 * s / 5) / (t + s) = 6 / 17 :=
by
  sorry

end fraction_of_students_with_buddy_l1032_103262


namespace distance_between_points_is_sqrt_5_l1032_103257

noncomputable def distance_between_polar_points : ℝ :=
  let xA := 1 * Real.cos (3/4 * Real.pi)
  let yA := 1 * Real.sin (3/4 * Real.pi)
  let xB := 2 * Real.cos (Real.pi / 4)
  let yB := 2 * Real.sin (Real.pi / 4)
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2)

theorem distance_between_points_is_sqrt_5 :
  distance_between_polar_points = Real.sqrt 5 :=
by
  sorry

end distance_between_points_is_sqrt_5_l1032_103257


namespace binomial_inequality_l1032_103297

theorem binomial_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : (1 + x)^n ≥ 1 + n * x :=
by sorry

end binomial_inequality_l1032_103297


namespace percentage_of_salt_in_second_solution_l1032_103247

-- Define the data and initial conditions
def original_solution_salt_percentage := 0.15
def replaced_solution_salt_percentage (x: ℝ) := x
def resulting_solution_salt_percentage := 0.16

-- State the question as a theorem
theorem percentage_of_salt_in_second_solution (S : ℝ) (x : ℝ) :
  0.15 * S - 0.0375 * S + x * (S / 4) = 0.16 * S → x = 0.19 :=
by 
  sorry

end percentage_of_salt_in_second_solution_l1032_103247


namespace coin_exchange_proof_l1032_103223

/-- Prove the coin combination that Petya initially had -/
theorem coin_exchange_proof (x y z : ℕ) (hx : 20 * x + 15 * y + 10 * z = 125) : x = 0 ∧ y = 1 ∧ z = 11 :=
by
  sorry

end coin_exchange_proof_l1032_103223


namespace total_lives_after_third_level_l1032_103268

def initial_lives : ℕ := 2

def extra_lives_first_level : ℕ := 6
def modifier_first_level (lives : ℕ) : ℕ := lives / 2

def extra_lives_second_level : ℕ := 11
def challenge_second_level (lives : ℕ) : ℕ := lives - 3

def reward_third_level (lives_first_two_levels : ℕ) : ℕ := 2 * lives_first_two_levels

theorem total_lives_after_third_level :
  let lives_first_level := modifier_first_level extra_lives_first_level
  let lives_after_first_level := initial_lives + lives_first_level
  let lives_second_level := challenge_second_level extra_lives_second_level
  let lives_after_second_level := lives_after_first_level + lives_second_level
  let total_gained_lives_first_two_levels := lives_first_level + lives_second_level
  let third_level_reward := reward_third_level total_gained_lives_first_two_levels
  lives_after_second_level + third_level_reward = 35 :=
by
  sorry

end total_lives_after_third_level_l1032_103268


namespace deepak_present_age_l1032_103287

theorem deepak_present_age (x : ℕ) (rahul deepak rohan : ℕ) 
  (h_ratio : rahul = 5 * x ∧ deepak = 2 * x ∧ rohan = 3 * x)
  (h_rahul_future_age : rahul + 8 = 28) :
  deepak = 8 := 
by
  sorry

end deepak_present_age_l1032_103287


namespace algebra_expression_value_l1032_103228

theorem algebra_expression_value (m : ℝ) (h : m^2 - 3 * m - 1 = 0) : 2 * m^2 - 6 * m + 5 = 7 := by
  sorry

end algebra_expression_value_l1032_103228


namespace find_added_number_l1032_103217

theorem find_added_number (R D Q X : ℕ) (hR : R = 5) (hD : D = 3 * Q) (hDiv : 113 = D * Q + R) (hD_def : D = 3 * R + X) : 
  X = 3 :=
by
  -- Provide the conditions as assumptions
  sorry

end find_added_number_l1032_103217


namespace circle_intersection_range_l1032_103248

theorem circle_intersection_range (a : ℝ) :
  (0 < a ∧ a < 2 * Real.sqrt 2) ∨ (-2 * Real.sqrt 2 < a ∧ a < 0) ↔
  (let C := { p : ℝ × ℝ | (p.1 - a) ^ 2 + (p.2 - a) ^ 2 = 4 };
   let O := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 4 };
   ∀ p, p ∈ C → p ∈ O) :=
sorry

end circle_intersection_range_l1032_103248


namespace smallest_n_l1032_103270

theorem smallest_n (n : ℕ) (h : 0 < n) (h1 : 813 * n % 30 = 1224 * n % 30) : n = 10 := 
sorry

end smallest_n_l1032_103270


namespace solution_inequality_set_l1032_103293

-- Define the inequality condition
def inequality (x : ℝ) : Prop := x^2 - 3 * x - 10 ≤ 0

-- Define the interval solution set
def solution_set := Set.Icc (-2 : ℝ) 5

-- The statement that we want to prove
theorem solution_inequality_set : {x : ℝ | inequality x} = solution_set :=
  sorry

end solution_inequality_set_l1032_103293


namespace bill_new_profit_percentage_l1032_103269

theorem bill_new_profit_percentage 
  (original_SP : ℝ)
  (profit_percent : ℝ)
  (increment : ℝ)
  (CP : ℝ)
  (CP_new : ℝ)
  (SP_new : ℝ)
  (Profit_new : ℝ)
  (new_profit_percent : ℝ) :
  original_SP = 439.99999999999966 →
  profit_percent = 0.10 →
  increment = 28 →
  CP = original_SP / (1 + profit_percent) →
  CP_new = CP * (1 - profit_percent) →
  SP_new = original_SP + increment →
  Profit_new = SP_new - CP_new →
  new_profit_percent = (Profit_new / CP_new) * 100 →
  new_profit_percent = 30 :=
by
  -- sorry to skip the proof
  sorry

end bill_new_profit_percentage_l1032_103269


namespace tan_five_pi_over_four_l1032_103215

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l1032_103215


namespace car_speed_l1032_103266

theorem car_speed (t_60 : ℝ := 60) (t_12 : ℝ := 12) (t_dist : ℝ := 1) :
  ∃ v : ℝ, v = 50 ∧ (t_60 / 60 + t_12 = 3600 / v) := 
by
  sorry

end car_speed_l1032_103266


namespace value_of_b_l1032_103229

noncomputable def k := 675

theorem value_of_b (a b : ℝ) (h1 : a * b = k) (h2 : a + b = 60) (h3 : a = 3 * b) (h4 : a = -12) :
  b = -56.25 := by
  sorry

end value_of_b_l1032_103229


namespace remaining_lemon_heads_after_eating_l1032_103243

-- Assume initial number of lemon heads is given
variables (initial_lemon_heads : ℕ)

-- Patricia eats 15 lemon heads
def remaining_lemon_heads (initial_lemon_heads : ℕ) : ℕ :=
  initial_lemon_heads - 15

theorem remaining_lemon_heads_after_eating :
  ∀ (initial_lemon_heads : ℕ), remaining_lemon_heads initial_lemon_heads = initial_lemon_heads - 15 :=
by
  intros
  rfl

end remaining_lemon_heads_after_eating_l1032_103243


namespace remainder_of_division_l1032_103281

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l1032_103281


namespace unique_combined_friends_count_l1032_103280

theorem unique_combined_friends_count 
  (james_friends : ℕ)
  (susan_friends : ℕ)
  (john_multiplier : ℕ)
  (shared_friends : ℕ)
  (maria_shared_friends : ℕ)
  (maria_friends : ℕ)
  (h_james : james_friends = 90)
  (h_susan : susan_friends = 50)
  (h_john : ∃ (john_friends : ℕ), john_friends = john_multiplier * susan_friends ∧ john_multiplier = 4)
  (h_shared : shared_friends = 35)
  (h_maria_shared : maria_shared_friends = 10)
  (h_maria : maria_friends = 80) :
  ∃ (total_unique_friends : ℕ), total_unique_friends = 325 :=
by
  -- Proof is omitted
  sorry

end unique_combined_friends_count_l1032_103280


namespace paint_cost_per_quart_l1032_103214

theorem paint_cost_per_quart
  (total_cost : ℝ)
  (coverage_per_quart : ℝ)
  (side_length : ℝ)
  (cost_per_quart : ℝ) 
  (h1 : total_cost = 192)
  (h2 : coverage_per_quart = 10)
  (h3 : side_length = 10) 
  (h4 : cost_per_quart = total_cost / ((6 * side_length ^ 2) / coverage_per_quart))
  : cost_per_quart = 3.20 := 
by 
  sorry

end paint_cost_per_quart_l1032_103214


namespace average_speed_is_70_l1032_103222

theorem average_speed_is_70 
  (distance1 distance2 : ℕ) (time1 time2 : ℕ)
  (h1 : distance1 = 80) (h2 : distance2 = 60)
  (h3 : time1 = 1) (h4 : time2 = 1) :
  (distance1 + distance2) / (time1 + time2) = 70 := 
by 
  sorry

end average_speed_is_70_l1032_103222


namespace smallest_logarithmic_term_l1032_103232

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_logarithmic_term (x₀ : ℝ) (hx₀ : f x₀ = 0) (h_interval : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) := 
by
  sorry

end smallest_logarithmic_term_l1032_103232


namespace remainder_and_division_l1032_103296

theorem remainder_and_division (n : ℕ) (h1 : n = 1680) (h2 : n % 9 = 0) : 
  1680 % 1677 = 3 :=
by {
  sorry
}

end remainder_and_division_l1032_103296


namespace sequence_properties_l1032_103233

def f (x : ℝ) : ℝ := x^3 + 3 * x

variables {a_5 a_8 : ℝ}
variables {S_12 : ℝ}

axiom a5_condition : (a_5 - 1)^3 + 3 * a_5 = 4
axiom a8_condition : (a_8 - 1)^3 + 3 * a_8 = 2

theorem sequence_properties : (a_5 > a_8) ∧ (S_12 = 12) :=
by {
  sorry
}

end sequence_properties_l1032_103233
