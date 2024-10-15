import Mathlib

namespace NUMINAMATH_GPT_linear_function_k_range_l1274_127413

theorem linear_function_k_range (k b : ℝ) (h1 : k ≠ 0) (h2 : ∃ x : ℝ, (x = 2) ∧ (-3 = k * x + b)) (h3 : 0 < b ∧ b < 1) : -2 < k ∧ k < -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_k_range_l1274_127413


namespace NUMINAMATH_GPT_average_speed_round_trip_l1274_127431

def time_to_walk_uphill := 30 -- in minutes
def time_to_walk_downhill := 10 -- in minutes
def distance_one_way := 1 -- in km

theorem average_speed_round_trip :
  (2 * distance_one_way) / ((time_to_walk_uphill + time_to_walk_downhill) / 60) = 3 := by
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l1274_127431


namespace NUMINAMATH_GPT_sum_of_four_smallest_divisors_l1274_127411

-- Define a natural number n and divisors d1, d2, d3, d4
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = k * d

-- Primary problem condition (sum of four divisors equals 2n)
def sum_of_divisors_eq (n d1 d2 d3 d4 : ℕ) : Prop := d1 + d2 + d3 + d4 = 2 * n

-- Assume the four divisors of n are distinct
def distinct (d1 d2 d3 d4 : ℕ) : Prop := d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

-- State the Lean proof problem
theorem sum_of_four_smallest_divisors (n d1 d2 d3 d4 : ℕ) (h1 : d1 < d2) (h2 : d2 < d3) (h3 : d3 < d4) 
    (h_div1 : is_divisor d1 n) (h_div2 : is_divisor d2 n) (h_div3 : is_divisor d3 n) (h_div4 : is_divisor d4 n)
    (h_sum : sum_of_divisors_eq n d1 d2 d3 d4) (h_distinct : distinct d1 d2 d3 d4) : 
    (d1 + d2 + d3 + d4 = 10 ∨ d1 + d2 + d3 + d4 = 11 ∨ d1 + d2 + d3 + d4 = 12) := 
sorry

end NUMINAMATH_GPT_sum_of_four_smallest_divisors_l1274_127411


namespace NUMINAMATH_GPT_fixed_point_l1274_127480

noncomputable def func (a : ℝ) (x : ℝ) : ℝ := a^(x-1)

theorem fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : func a 1 = 1 :=
by {
  -- We need to prove that func a 1 = 1 for any a > 0 and a ≠ 1
  sorry
}

end NUMINAMATH_GPT_fixed_point_l1274_127480


namespace NUMINAMATH_GPT_find_percentage_l1274_127440

theorem find_percentage (P : ℝ) : 
  (P / 100) * 100 - 40 = 30 → P = 70 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_percentage_l1274_127440


namespace NUMINAMATH_GPT_price_increase_percentage_l1274_127417

theorem price_increase_percentage (x : ℝ) :
  (0.9 * (1 + x / 100) * 0.9259259259259259 = 1) → x = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_price_increase_percentage_l1274_127417


namespace NUMINAMATH_GPT_shaded_region_area_l1274_127432

theorem shaded_region_area (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) (T : ℝ):
  d = 3 → L = 24 → L = n * d → n * 2 = 16 → r = d / 2 → 
  A = (1 / 2) * π * r ^ 2 → T = 16 * A → T = 18 * π :=
  by
  intros d_eq L_eq Ln_eq semicircle_count r_eq A_eq T_eq_total
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1274_127432


namespace NUMINAMATH_GPT_evaluate_fg_l1274_127419

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x - 5

theorem evaluate_fg : f (g 4) = 9 := by
  sorry

end NUMINAMATH_GPT_evaluate_fg_l1274_127419


namespace NUMINAMATH_GPT_find_b_l1274_127460

theorem find_b (b : ℤ) :
  ∃ (r₁ r₂ : ℤ), (r₁ = -9) ∧ (r₁ * r₂ = 36) ∧ (r₁ + r₂ = -b) → b = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_b_l1274_127460


namespace NUMINAMATH_GPT_food_company_total_food_l1274_127422

theorem food_company_total_food (boxes : ℕ) (kg_per_box : ℕ) (full_boxes : boxes = 388) (weight_per_box : kg_per_box = 2) :
  boxes * kg_per_box = 776 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_food_company_total_food_l1274_127422


namespace NUMINAMATH_GPT_no_base_satisfies_l1274_127462

def e : ℕ := 35

theorem no_base_satisfies :
  ∀ (base : ℝ), (1 / 5)^e * (1 / 4)^18 ≠ 1 / 2 * (base)^35 :=
by
  sorry

end NUMINAMATH_GPT_no_base_satisfies_l1274_127462


namespace NUMINAMATH_GPT_solve_for_x_l1274_127441

theorem solve_for_x (x : ℝ) (h : 1 - 1 / (1 - x) ^ 3 = 1 / (1 - x)) : x = 1 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1274_127441


namespace NUMINAMATH_GPT_factorize_expression_l1274_127443

theorem factorize_expression (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1274_127443


namespace NUMINAMATH_GPT_cone_height_l1274_127457

theorem cone_height (V : ℝ) (π : ℝ) (r h : ℝ) (sqrt2 : ℝ) :
  V = 9720 * π →
  sqrt2 = Real.sqrt 2 →
  h = r * sqrt2 →
  V = (1/3) * π * r^2 * h →
  h = 38.7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cone_height_l1274_127457


namespace NUMINAMATH_GPT_frequency_even_numbers_facing_up_l1274_127420

theorem frequency_even_numbers_facing_up (rolls : ℕ) (event_occurrences : ℕ) (h_rolls : rolls = 100) (h_event : event_occurrences = 47) : (event_occurrences / (rolls : ℝ)) = 0.47 :=
by
  sorry

end NUMINAMATH_GPT_frequency_even_numbers_facing_up_l1274_127420


namespace NUMINAMATH_GPT_sum_of_polynomials_l1274_127492

open Polynomial

noncomputable def f : ℚ[X] := -4 * X^2 + 2 * X - 5
noncomputable def g : ℚ[X] := -6 * X^2 + 4 * X - 9
noncomputable def h : ℚ[X] := 6 * X^2 + 6 * X + 2

theorem sum_of_polynomials :
  f + g + h = -4 * X^2 + 12 * X - 12 :=
by sorry

end NUMINAMATH_GPT_sum_of_polynomials_l1274_127492


namespace NUMINAMATH_GPT_find_a6_l1274_127425

-- Define the geometric sequence conditions
noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the specific sequence with given initial conditions and sum of first three terms
theorem find_a6 : 
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (0 < q) ∧ (q ≠ 1) ∧ geom_seq a q ∧ 
    a 1 = 96 ∧ 
    (a 1 + a 2 + a 3 = 168) ∧
    a 6 = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_a6_l1274_127425


namespace NUMINAMATH_GPT_prove_b_zero_l1274_127477

variables {a b c : ℕ}

theorem prove_b_zero (h1 : ∃ (a b c : ℕ), a^5 + 4 * b^5 = c^5 ∧ c % 2 = 0) : b = 0 :=
sorry

end NUMINAMATH_GPT_prove_b_zero_l1274_127477


namespace NUMINAMATH_GPT_kim_trip_time_l1274_127438

-- Definitions
def distance_freeway : ℝ := 120
def distance_mountain : ℝ := 25
def speed_ratio : ℝ := 4
def time_mountain : ℝ := 75

-- The problem statement
theorem kim_trip_time : ∃ t_freeway t_total : ℝ,
  t_freeway = distance_freeway / (speed_ratio * (distance_mountain / time_mountain)) ∧
  t_total = time_mountain + t_freeway ∧
  t_total = 165 := by
  sorry

end NUMINAMATH_GPT_kim_trip_time_l1274_127438


namespace NUMINAMATH_GPT_find_speed_l1274_127490

theorem find_speed (v : ℝ) (t : ℝ) (h : t = 5 * v^2) (ht : t = 20) : v = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_l1274_127490


namespace NUMINAMATH_GPT_min_value_expr_min_value_achieved_l1274_127423

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4*x + 1/x^4 ≥ 5 :=
by
  sorry

theorem min_value_achieved (x : ℝ) : x = 1 → 4*x + 1/x^4 = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_min_value_achieved_l1274_127423


namespace NUMINAMATH_GPT_monthly_earnings_l1274_127427

variable (e : ℕ) (s : ℕ) (p : ℕ) (t : ℕ)

-- conditions
def half_monthly_savings := s = e / 2
def car_price := p = 16000
def saving_months := t = 8
def total_saving := s * t = p

theorem monthly_earnings : ∀ (e s p t : ℕ), 
  half_monthly_savings e s → 
  car_price p → 
  saving_months t → 
  total_saving s t p → 
  e = 4000 :=
by
  intros e s p t h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_monthly_earnings_l1274_127427


namespace NUMINAMATH_GPT_find_original_number_l1274_127471

theorem find_original_number
  (n : ℤ)
  (h : (2 * (n + 2) - 2) / 2 = 7) :
  n = 6 := 
sorry

end NUMINAMATH_GPT_find_original_number_l1274_127471


namespace NUMINAMATH_GPT_buckets_required_l1274_127468

theorem buckets_required (C : ℚ) (N : ℕ) (h : 250 * (4/5 : ℚ) * C = N * C) : N = 200 :=
by
  sorry

end NUMINAMATH_GPT_buckets_required_l1274_127468


namespace NUMINAMATH_GPT_c_share_l1274_127434

theorem c_share (A B C : ℕ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : A + B + C = 406) : 
    C = 232 := by 
    sorry

end NUMINAMATH_GPT_c_share_l1274_127434


namespace NUMINAMATH_GPT_square_side_length_l1274_127483

theorem square_side_length (a b s : ℝ) 
  (h_area : a * b = 54) 
  (h_square_condition : 3 * a = b / 2) : 
  s = 9 :=
by 
  sorry

end NUMINAMATH_GPT_square_side_length_l1274_127483


namespace NUMINAMATH_GPT_work_duration_l1274_127448

theorem work_duration (p q r : ℕ) (Wp Wq Wr : ℕ) (t1 t2 : ℕ) (T : ℝ) :
  (Wp = 20) → (Wq = 12) → (Wr = 30) →
  (t1 = 4) → (t2 = 4) →
  (T = (t1 + t2 + (4/15 * Wr) / (1/(Wr) + 1/(Wq) + 1/(Wp)))) →
  T = 9.6 :=
by
  intros;
  sorry

end NUMINAMATH_GPT_work_duration_l1274_127448


namespace NUMINAMATH_GPT_recreation_spent_percent_l1274_127479

variable (W : ℝ) -- Assume W is the wages last week

-- Conditions
def last_week_spent_on_recreation (W : ℝ) : ℝ := 0.25 * W
def this_week_wages (W : ℝ) : ℝ := 0.70 * W
def this_week_spent_on_recreation (W : ℝ) : ℝ := 0.50 * (this_week_wages W)

-- Proof statement
theorem recreation_spent_percent (W : ℝ) :
  (this_week_spent_on_recreation W / last_week_spent_on_recreation W) * 100 = 140 := by
  sorry

end NUMINAMATH_GPT_recreation_spent_percent_l1274_127479


namespace NUMINAMATH_GPT_remainder_of_5n_minus_9_l1274_127400

theorem remainder_of_5n_minus_9 (n : ℤ) (h : n % 11 = 3) : (5 * n - 9) % 11 = 6 :=
by
  sorry -- Proof is omitted, as per instruction.

end NUMINAMATH_GPT_remainder_of_5n_minus_9_l1274_127400


namespace NUMINAMATH_GPT_two_rel_prime_exists_l1274_127415

theorem two_rel_prime_exists (A : Finset ℕ) (h1 : A.card = 2011) (h2 : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 4020) : 
  ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ Nat.gcd a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_two_rel_prime_exists_l1274_127415


namespace NUMINAMATH_GPT_find_f1_l1274_127418

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f1
  (h1 : ∀ x : ℝ, |f x - x^2| ≤ 1/4)
  (h2 : ∀ x : ℝ, |f x + 1 - x^2| ≤ 3/4) :
  f 1 = 3/4 := 
sorry

end NUMINAMATH_GPT_find_f1_l1274_127418


namespace NUMINAMATH_GPT_find_width_of_room_l1274_127402

variable (length : ℕ) (total_carpet_owned : ℕ) (additional_carpet_needed : ℕ)
variable (total_area : ℕ) (width : ℕ)

theorem find_width_of_room
  (h1 : length = 11) 
  (h2 : total_carpet_owned = 16) 
  (h3 : additional_carpet_needed = 149)
  (h4 : total_area = total_carpet_owned + additional_carpet_needed) 
  (h5 : total_area = length * width) :
  width = 15 := by
    sorry

end NUMINAMATH_GPT_find_width_of_room_l1274_127402


namespace NUMINAMATH_GPT_total_animals_seen_l1274_127466

theorem total_animals_seen (lions_sat : ℕ) (elephants_sat : ℕ) 
                           (buffaloes_sun : ℕ) (leopards_sun : ℕ)
                           (rhinos_mon : ℕ) (warthogs_mon : ℕ) 
                           (h_sat : lions_sat = 3 ∧ elephants_sat = 2)
                           (h_sun : buffaloes_sun = 2 ∧ leopards_sun = 5)
                           (h_mon : rhinos_mon = 5 ∧ warthogs_mon = 3) :
  lions_sat + elephants_sat + buffaloes_sun + leopards_sun + rhinos_mon + warthogs_mon = 20 := by
  sorry

end NUMINAMATH_GPT_total_animals_seen_l1274_127466


namespace NUMINAMATH_GPT_range_of_b_l1274_127406

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  - (1/2) * (x - 2)^2 + b * Real.log x

theorem range_of_b (b : ℝ) : (∀ x : ℝ, 1 < x → f x b ≤ f 1 b) → b ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1274_127406


namespace NUMINAMATH_GPT_teacher_arrangements_l1274_127454

theorem teacher_arrangements (T : Fin 30 → ℕ) (h1 : T 1 < T 2 ∧ T 2 < T 3 ∧ T 3 < T 4 ∧ T 4 < T 5)
  (h2 : ∀ i : Fin 4, T (i + 1) ≥ T i + 3)
  (h3 : 1 ≤ T 1)
  (h4 : T 5 ≤ 26) :
  ∃ n : ℕ, n = 26334 := by
  sorry

end NUMINAMATH_GPT_teacher_arrangements_l1274_127454


namespace NUMINAMATH_GPT_average_after_discard_l1274_127446

theorem average_after_discard (sum_50 : ℝ) (avg_50 : sum_50 = 2200) (a b : ℝ) (h1 : a = 45) (h2 : b = 55) :
  (sum_50 - (a + b)) / 48 = 43.75 :=
by
  -- Given conditions: sum_50 = 2200, a = 45, b = 55
  -- We need to prove (sum_50 - (a + b)) / 48 = 43.75
  sorry

end NUMINAMATH_GPT_average_after_discard_l1274_127446


namespace NUMINAMATH_GPT_inequality_proof_l1274_127429

variable {a b c : ℝ}

theorem inequality_proof (h : a > b) : (a / (c^2 + 1)) > (b / (c^2 + 1)) := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1274_127429


namespace NUMINAMATH_GPT_Tom_sold_games_for_240_l1274_127405

-- Define the value of games and perform operations as per given conditions
def original_value : ℕ := 200
def tripled_value : ℕ := 3 * original_value
def sold_percentage : ℕ := 40
def sold_value : ℕ := (sold_percentage * tripled_value) / 100

-- Assert the proof problem
theorem Tom_sold_games_for_240 : sold_value = 240 := 
by
  sorry

end NUMINAMATH_GPT_Tom_sold_games_for_240_l1274_127405


namespace NUMINAMATH_GPT_ten_numbers_property_l1274_127445

theorem ten_numbers_property (x : ℕ → ℝ) (h : ∀ i : ℕ, 1 ≤ i → i ≤ 9 → x i + 2 * x (i + 1) = 1) : 
  x 1 + 512 * x 10 = 171 :=
by
  sorry

end NUMINAMATH_GPT_ten_numbers_property_l1274_127445


namespace NUMINAMATH_GPT_ratio_D_to_C_l1274_127495

-- Defining the terms and conditions
def speed_ratio (C Ch D : ℝ) : Prop :=
  (C = 2 * Ch) ∧
  (D / Ch = 6)

-- The theorem statement
theorem ratio_D_to_C (C Ch D : ℝ) (h : speed_ratio C Ch D) : (D / C = 3) :=
by
  sorry

end NUMINAMATH_GPT_ratio_D_to_C_l1274_127495


namespace NUMINAMATH_GPT_part1_part2_l1274_127433

noncomputable def f (x : ℝ) : ℝ := (Real.exp (-x) - Real.exp x) / 2

theorem part1 (h_odd : ∀ x, f (-x) = -f x) (g : ℝ → ℝ) (h_even : ∀ x, g (-x) = g x)
  (h_g_def : ∀ x, g x = f x + Real.exp x) :
  ∀ x, f x = (Real.exp (-x) - Real.exp x) / 2 := sorry

theorem part2 : {x : ℝ | f x ≥ 3 / 4} = {x | x ≤ -Real.log 2} := sorry

end NUMINAMATH_GPT_part1_part2_l1274_127433


namespace NUMINAMATH_GPT_sara_initial_quarters_l1274_127410

theorem sara_initial_quarters (total_quarters dad_gift initial_quarters : ℕ) (h1 : dad_gift = 49) (h2 : total_quarters = 70) (h3 : total_quarters = initial_quarters + dad_gift) : initial_quarters = 21 :=
by sorry

end NUMINAMATH_GPT_sara_initial_quarters_l1274_127410


namespace NUMINAMATH_GPT_eighth_term_of_arithmetic_sequence_l1274_127486

noncomputable def arithmetic_sequence (n : ℕ) (a1 an : ℚ) (k : ℕ) : ℚ :=
  a1 + (k - 1) * ((an - a1) / (n - 1))

theorem eighth_term_of_arithmetic_sequence :
  ∀ (a1 a30 : ℚ), a1 = 5 → a30 = 86 → 
  arithmetic_sequence 30 a1 a30 8 = 592 / 29 :=
by
  intros a1 a30 h_a1 h_a30
  rw [h_a1, h_a30]
  dsimp [arithmetic_sequence]
  sorry

end NUMINAMATH_GPT_eighth_term_of_arithmetic_sequence_l1274_127486


namespace NUMINAMATH_GPT_base5_division_l1274_127424

theorem base5_division :
  ∀ (a b : ℕ), a = 1121 ∧ b = 12 → 
   ∃ (q r : ℕ), (a = b * q + r) ∧ (r < b) ∧ (q = 43) :=
by sorry

end NUMINAMATH_GPT_base5_division_l1274_127424


namespace NUMINAMATH_GPT_problem_statement_l1274_127430

noncomputable def a : ℝ := Real.tan (1 / 2)
noncomputable def b : ℝ := Real.tan (2 / Real.pi)
noncomputable def c : ℝ := Real.sqrt 3 / Real.pi

theorem problem_statement : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1274_127430


namespace NUMINAMATH_GPT_problem1_correct_problem2_correct_l1274_127487

noncomputable def problem1_solution_set : Set ℝ := {x | x ≤ -3 ∨ x ≥ 1}

noncomputable def problem2_solution_set : Set ℝ := {x | (-3 ≤ x ∧ x < 1) ∨ (3 < x ∧ x ≤ 7)}

theorem problem1_correct (x : ℝ) :
  (4 - x) / (x^2 + x + 1) ≤ 1 ↔ x ∈ problem1_solution_set :=
sorry

theorem problem2_correct (x : ℝ) :
  (1 < |x - 2| ∧ |x - 2| ≤ 5) ↔ x ∈ problem2_solution_set :=
sorry

end NUMINAMATH_GPT_problem1_correct_problem2_correct_l1274_127487


namespace NUMINAMATH_GPT_effect_on_revenue_l1274_127472

-- Define the conditions using parameters and variables

variables {P Q : ℝ} -- Original price and quantity of TV sets

def new_price (P : ℝ) : ℝ := P * 1.60 -- New price after 60% increase
def new_quantity (Q : ℝ) : ℝ := Q * 0.80 -- New quantity after 20% decrease

def original_revenue (P Q : ℝ) : ℝ := P * Q -- Original revenue
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q) -- New revenue

theorem effect_on_revenue
  (P Q : ℝ) :
  new_revenue P Q = original_revenue P Q * 1.28 :=
by
  sorry

end NUMINAMATH_GPT_effect_on_revenue_l1274_127472


namespace NUMINAMATH_GPT_joanne_main_job_hours_l1274_127465

theorem joanne_main_job_hours (h : ℕ) (earn_main_job : ℝ) (earn_part_time : ℝ) (hours_part_time : ℕ) (days_week : ℕ) (total_weekly_earn : ℝ) :
  earn_main_job = 16.00 →
  earn_part_time = 13.50 →
  hours_part_time = 2 →
  days_week = 5 →
  total_weekly_earn = 775 →
  days_week * earn_main_job * h + days_week * earn_part_time * hours_part_time = total_weekly_earn →
  h = 8 :=
by
  sorry

end NUMINAMATH_GPT_joanne_main_job_hours_l1274_127465


namespace NUMINAMATH_GPT_solve_system_eq_l1274_127404

theorem solve_system_eq (x y : ℚ) 
  (h1 : 3 * x - 7 * y = 31) 
  (h2 : 5 * x + 2 * y = -10) : 
  x = -336 / 205 := 
sorry

end NUMINAMATH_GPT_solve_system_eq_l1274_127404


namespace NUMINAMATH_GPT_vincent_total_loads_l1274_127403

def loads_wednesday : Nat := 2 + 1 + 3

def loads_thursday : Nat := 2 * loads_wednesday

def loads_friday : Nat := loads_thursday / 2

def loads_saturday : Nat := loads_wednesday / 3

def total_loads : Nat := loads_wednesday + loads_thursday + loads_friday + loads_saturday

theorem vincent_total_loads : total_loads = 20 := by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_vincent_total_loads_l1274_127403


namespace NUMINAMATH_GPT_min_seats_occupied_l1274_127439

theorem min_seats_occupied (n : ℕ) (h : n = 150) : ∃ k : ℕ, k = 37 ∧ ∀ m : ℕ, m > k → ∃ i : ℕ, i < k ∧ m - k ≥ 2 := sorry

end NUMINAMATH_GPT_min_seats_occupied_l1274_127439


namespace NUMINAMATH_GPT_lines_are_perpendicular_l1274_127493

-- Define the first line equation
def line1 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x - y + 3 = 0

-- Definition to determine the perpendicularity of two lines
def are_perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

theorem lines_are_perpendicular :
  are_perpendicular (-1) (1) := 
by
  sorry

end NUMINAMATH_GPT_lines_are_perpendicular_l1274_127493


namespace NUMINAMATH_GPT_cannot_be_sum_of_six_consecutive_odd_integers_l1274_127447

theorem cannot_be_sum_of_six_consecutive_odd_integers (S : ℕ) :
  (S = 90 ∨ S = 150) ->
  ∀ n : ℤ, ¬(S = n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10)) :=
by
  intro h
  intro n
  cases h
  case inl => 
    sorry
  case inr => 
    sorry

end NUMINAMATH_GPT_cannot_be_sum_of_six_consecutive_odd_integers_l1274_127447


namespace NUMINAMATH_GPT_inequality_must_hold_l1274_127485

theorem inequality_must_hold (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := 
sorry

end NUMINAMATH_GPT_inequality_must_hold_l1274_127485


namespace NUMINAMATH_GPT_police_speed_l1274_127491

/-- 
A thief runs away from a location with a speed of 20 km/hr.
A police officer starts chasing him from a location 60 km away after 1 hour.
The police officer catches the thief after 4 hours.
Prove that the speed of the police officer is 40 km/hr.
-/
theorem police_speed
  (thief_speed : ℝ)
  (police_start_distance : ℝ)
  (police_chase_time : ℝ)
  (time_head_start : ℝ)
  (police_distance_to_thief : ℝ)
  (thief_distance_after_time : ℝ)
  (total_distance_police_officer : ℝ) :
  thief_speed = 20 ∧
  police_start_distance = 60 ∧
  police_chase_time = 4 ∧
  time_head_start = 1 ∧
  police_distance_to_thief = police_start_distance + 100 ∧
  thief_distance_after_time = thief_speed * police_chase_time + thief_speed * time_head_start ∧
  total_distance_police_officer = police_start_distance + (thief_speed * (police_chase_time + time_head_start)) →
  (total_distance_police_officer / police_chase_time) = 40 := by
  sorry

end NUMINAMATH_GPT_police_speed_l1274_127491


namespace NUMINAMATH_GPT_binary_addition_to_decimal_l1274_127458

theorem binary_addition_to_decimal : (0b111111111 + 0b1000001 = 576) :=
by {
  sorry
}

end NUMINAMATH_GPT_binary_addition_to_decimal_l1274_127458


namespace NUMINAMATH_GPT_placement_ways_l1274_127421

theorem placement_ways (rows cols crosses : ℕ) (h1 : rows = 3) (h2 : cols = 4) (h3 : crosses = 4)
  (condition : ∀ r : Fin rows, ∃ c : Fin cols, r < rows ∧ c < cols) : 
  (∃ n, n = (3 * 6 * 2) → n = 36) :=
by 
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_placement_ways_l1274_127421


namespace NUMINAMATH_GPT_find_k_value_l1274_127494

theorem find_k_value : 
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 16 * 12 ^ 1001 :=
by
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  sorry

end NUMINAMATH_GPT_find_k_value_l1274_127494


namespace NUMINAMATH_GPT_area_inside_quadrilateral_BCDE_outside_circle_l1274_127499

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3) / 2 * side_length ^ 2

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

theorem area_inside_quadrilateral_BCDE_outside_circle :
  let side_length := 2
  let hex_area := hexagon_area side_length
  let hex_area_large := hexagon_area (2 * side_length)
  let circle_radius := 3
  let circle_area_A := circle_area circle_radius
  let total_area_of_interest := hex_area_large - circle_area_A
  let area_of_one_region := total_area_of_interest / 6
  area_of_one_region = 4 * Real.sqrt 3 - (3 / 2) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_inside_quadrilateral_BCDE_outside_circle_l1274_127499


namespace NUMINAMATH_GPT_zoo_visitors_l1274_127482

theorem zoo_visitors (P : ℕ) (h : 3 * P = 3750) : P = 1250 :=
by 
  sorry

end NUMINAMATH_GPT_zoo_visitors_l1274_127482


namespace NUMINAMATH_GPT_ratio_of_blue_fish_to_total_fish_l1274_127496

-- Define the given conditions
def total_fish : ℕ := 30
def blue_spotted_fish : ℕ := 5
def half (n : ℕ) : ℕ := n / 2

-- Calculate the number of blue fish using the conditions
def blue_fish : ℕ := blue_spotted_fish * 2

-- Define the ratio of blue fish to total fish
def ratio (num denom : ℕ) : ℚ := num / denom

-- The theorem to prove
theorem ratio_of_blue_fish_to_total_fish :
  ratio blue_fish total_fish = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_blue_fish_to_total_fish_l1274_127496


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1274_127455

theorem eccentricity_of_ellipse (k : ℝ) (h_k : k > 0)
  (focus : ∃ (x : ℝ), (x, 0) = ⟨3, 0⟩) :
  ∃ e : ℝ, e = (Real.sqrt 3 / 2) := 
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1274_127455


namespace NUMINAMATH_GPT_value_of_x_l1274_127444

theorem value_of_x (x : ℝ) : 3 - 5 + 7 = 6 - x → x = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_x_l1274_127444


namespace NUMINAMATH_GPT_probability_john_david_chosen_l1274_127481

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem probability_john_david_chosen :
  let total_workers := 6
  let choose_two := choose total_workers 2
  let favorable_outcomes := 1
  choose_two = 15 → (favorable_outcomes / choose_two : ℝ) = 1 / 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_john_david_chosen_l1274_127481


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_a_l1274_127473

-- Definitions and statements for Problem 1
def f1 (x : ℝ) : ℝ := -12 * x ^ 2 - 2 * x + 2

theorem problem1_solution_set :
  (∃ a b : ℝ, a = -12 ∧ b = -2 ∧
    ∀ x : ℝ, f1 x > 0 → -1 / 2 < x ∧ x < 1 / 3) :=
by sorry

-- Definitions and statements for Problem 2
def f2 (x a : ℝ) : ℝ := a * x ^ 2 - x + 2

theorem problem2_range_of_a :
  (∃ b : ℝ, b = -1 ∧
    ∀ a : ℝ, (∀ x : ℝ, f2 x a < 0 → false) → a ≥ 1 / 8) :=
by sorry

end NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_a_l1274_127473


namespace NUMINAMATH_GPT_magic_triangle_largest_S_l1274_127451

theorem magic_triangle_largest_S :
  ∃ (S : ℕ) (a b c d e f g : ℕ),
    (10 ≤ a) ∧ (a ≤ 16) ∧
    (10 ≤ b) ∧ (b ≤ 16) ∧
    (10 ≤ c) ∧ (c ≤ 16) ∧
    (10 ≤ d) ∧ (d ≤ 16) ∧
    (10 ≤ e) ∧ (e ≤ 16) ∧
    (10 ≤ f) ∧ (f ≤ 16) ∧
    (10 ≤ g) ∧ (g ≤ 16) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧
    (e ≠ f) ∧ (e ≠ g) ∧
    (f ≠ g) ∧
    (S = a + b + c) ∧
    (S = c + d + e) ∧
    (S = e + f + a) ∧
    (S = g + b + c) ∧
    (S = g + d + e) ∧
    (S = g + f + a) ∧
    ((a + b + c) + (c + d + e) + (e + f + a) = 91 - g) ∧
    (S = 26) := sorry

end NUMINAMATH_GPT_magic_triangle_largest_S_l1274_127451


namespace NUMINAMATH_GPT_pizza_slices_left_l1274_127489

theorem pizza_slices_left (total_slices : ℕ) (angeli_slices : ℚ) (marlon_slices : ℚ) 
  (H1 : total_slices = 8) (H2 : angeli_slices = 3/2) (H3 : marlon_slices = 3/2) :
  total_slices - (angeli_slices + marlon_slices) = 5 :=
by
  sorry

end NUMINAMATH_GPT_pizza_slices_left_l1274_127489


namespace NUMINAMATH_GPT_accurate_scale_l1274_127437

-- Definitions for the weights on each scale
variables (a b c d e x : ℝ)

-- Given conditions
def condition1 := c = b - 0.3
def condition2 := d = c - 0.1
def condition3 := e = a - 0.1
def condition4 := c = e - 0.1
def condition5 := 5 * x = a + b + c + d + e

-- Proof statement
theorem accurate_scale 
  (h1 : c = b - 0.3)
  (h2 : d = c - 0.1)
  (h3 : e = a - 0.1)
  (h4 : c = e - 0.1)
  (h5 : 5 * x = a + b + c + d + e) : e = x :=
by
  sorry

end NUMINAMATH_GPT_accurate_scale_l1274_127437


namespace NUMINAMATH_GPT_find_integer_pairs_l1274_127436

theorem find_integer_pairs (a b : ℤ) (h₁ : 1 < a) (h₂ : 1 < b) 
    (h₃ : a ∣ (b + 1)) (h₄ : b ∣ (a^3 - 1)) : 
    ∃ (s : ℤ), (s ≥ 2 ∧ (a, b) = (s, s^3 - 1)) ∨ (s ≥ 3 ∧ (a, b) = (s, s - 1)) :=
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l1274_127436


namespace NUMINAMATH_GPT_gcd_lcm_product_24_60_l1274_127469

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_24_60_l1274_127469


namespace NUMINAMATH_GPT_mean_greater_than_median_l1274_127409

theorem mean_greater_than_median (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5 
  let median := x + 4 
  mean - median = 4 :=
by 
  sorry

end NUMINAMATH_GPT_mean_greater_than_median_l1274_127409


namespace NUMINAMATH_GPT_proportional_segments_l1274_127475

theorem proportional_segments (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 d1 d2 d3 d4 : ℕ)
  (hA : a1 = 1 ∧ a2 = 2 ∧ a3 = 3 ∧ a4 = 4)
  (hB : b1 = 1 ∧ b2 = 2 ∧ b3 = 2 ∧ b4 = 4)
  (hC : c1 = 3 ∧ c2 = 5 ∧ c3 = 9 ∧ c4 = 13)
  (hD : d1 = 1 ∧ d2 = 2 ∧ d3 = 2 ∧ d4 = 3) :
  (b1 * b4 = b2 * b3) :=
by
  sorry

end NUMINAMATH_GPT_proportional_segments_l1274_127475


namespace NUMINAMATH_GPT_find_x_l1274_127452

theorem find_x (x : ℝ) (h : x + 5 * 12 / (180 / 3) = 41) : x = 40 :=
sorry

end NUMINAMATH_GPT_find_x_l1274_127452


namespace NUMINAMATH_GPT_remainder_ab_div_48_is_15_l1274_127474

noncomputable def remainder_ab_div_48 (a b : ℕ) (ha : a % 8 = 3) (hb : b % 6 = 5) : ℕ :=
  (a * b) % 48

theorem remainder_ab_div_48_is_15 {a b : ℕ} (ha : a % 8 = 3) (hb : b % 6 = 5) : remainder_ab_div_48 a b ha hb = 15 :=
  sorry

end NUMINAMATH_GPT_remainder_ab_div_48_is_15_l1274_127474


namespace NUMINAMATH_GPT_S8_eq_90_l1274_127442

-- Definitions and given conditions
def arithmetic_seq (a : ℕ → ℤ) : Prop := ∃ d, ∀ n, a (n + 1) - a n = d
def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop := ∀ n, S n = (n * (a 1 + a n)) / 2
def condition_a4 (a : ℕ → ℤ) : Prop := a 4 = 18 - a 5

-- Prove that S₈ = 90
theorem S8_eq_90 (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arithmetic_seq a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond : condition_a4 a) : S 8 = 90 :=
by
  sorry

end NUMINAMATH_GPT_S8_eq_90_l1274_127442


namespace NUMINAMATH_GPT_problem_proof_l1274_127456

-- Definitions based on conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Theorem to prove
theorem problem_proof (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1274_127456


namespace NUMINAMATH_GPT_original_price_of_article_l1274_127450

theorem original_price_of_article :
  ∃ P : ℝ, (P * 0.55 * 0.85 = 920) ∧ P = 1968.04 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_article_l1274_127450


namespace NUMINAMATH_GPT_replace_digits_divisible_by_13_l1274_127497

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

theorem replace_digits_divisible_by_13 :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ 
  (3 * 10^6 + x * 10^4 + y * 10^2 + 3) % 13 = 0 ∧
  (x = 2 ∧ y = 3 ∨ 
   x = 5 ∧ y = 2 ∨ 
   x = 8 ∧ y = 1 ∨ 
   x = 9 ∧ y = 5 ∨ 
   x = 6 ∧ y = 6 ∨ 
   x = 3 ∧ y = 7 ∨ 
   x = 0 ∧ y = 8) :=
by
  sorry

end NUMINAMATH_GPT_replace_digits_divisible_by_13_l1274_127497


namespace NUMINAMATH_GPT_solve_for_a_plus_b_l1274_127408

theorem solve_for_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, a * (x + b) = 3 * x + 12) → a + b = 7 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_for_a_plus_b_l1274_127408


namespace NUMINAMATH_GPT_last_two_digits_sum_is_32_l1274_127476

-- Definitions for digit representation
variables (z a r l m : ℕ)

-- Numbers definitions
def ZARAZA := z * 10^5 + a * 10^4 + r * 10^3 + a * 10^2 + z * 10 + a
def ALMAZ := a * 10^4 + l * 10^3 + m * 10^2 + a * 10 + z

-- Condition that ZARAZA is divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Condition that ALMAZ is divisible by 28
def divisible_by_28 (n : ℕ) : Prop := n % 28 = 0

-- The theorem to prove
theorem last_two_digits_sum_is_32
  (hz4 : divisible_by_4 (ZARAZA z a r))
  (ha28 : divisible_by_28 (ALMAZ a l m z))
  : (ZARAZA z a r + ALMAZ a l m z) % 100 = 32 :=
by sorry

end NUMINAMATH_GPT_last_two_digits_sum_is_32_l1274_127476


namespace NUMINAMATH_GPT_solve_inequality_range_of_a_l1274_127426

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |2 * x + 2|

theorem solve_inequality : {x : ℝ | f x > 5} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 4 / 3} :=
by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x, ¬ (f x < a)) : a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_range_of_a_l1274_127426


namespace NUMINAMATH_GPT_pages_per_day_difference_l1274_127484

theorem pages_per_day_difference :
  let songhee_pages := 288
  let songhee_days := 12
  let eunju_pages := 243
  let eunju_days := 9
  let songhee_per_day := songhee_pages / songhee_days
  let eunju_per_day := eunju_pages / eunju_days
  eunju_per_day - songhee_per_day = 3 := by
  sorry

end NUMINAMATH_GPT_pages_per_day_difference_l1274_127484


namespace NUMINAMATH_GPT_range_of_a_l1274_127464

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 1) ^ 2 > 4 → x > a) → a ≥ 1 := sorry

end NUMINAMATH_GPT_range_of_a_l1274_127464


namespace NUMINAMATH_GPT_cos_of_angle_complement_l1274_127459

theorem cos_of_angle_complement (α : ℝ) (h : 90 - α = 30) : Real.cos α = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_of_angle_complement_l1274_127459


namespace NUMINAMATH_GPT_Peter_buys_more_hot_dogs_than_hamburgers_l1274_127435

theorem Peter_buys_more_hot_dogs_than_hamburgers :
  let chicken := 16
  let hamburgers := chicken / 2
  (exists H : Real, 16 + hamburgers + H + H / 2 = 39 ∧ (H - hamburgers = 2)) := sorry

end NUMINAMATH_GPT_Peter_buys_more_hot_dogs_than_hamburgers_l1274_127435


namespace NUMINAMATH_GPT_interval_satisfies_ineq_l1274_127478

theorem interval_satisfies_ineq (p : ℝ) (h1 : 18 * p < 10) (h2 : 0.5 < p) : 0.5 < p ∧ p < 5 / 9 :=
by {
  sorry -- Proof not required, only the statement.
}

end NUMINAMATH_GPT_interval_satisfies_ineq_l1274_127478


namespace NUMINAMATH_GPT_smallest_integer_in_consecutive_set_l1274_127407

theorem smallest_integer_in_consecutive_set (n : ℤ) (h : n + 6 < 2 * (n + 3)) : n > 0 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_in_consecutive_set_l1274_127407


namespace NUMINAMATH_GPT_sum_of_numbers_l1274_127467

theorem sum_of_numbers (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) (h3 : x^2 - y^2 = 50) : x + y = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1274_127467


namespace NUMINAMATH_GPT_average_of_a_b_l1274_127461

theorem average_of_a_b (a b : ℚ) (h1 : b = 2 * a) (h2 : (4 + 6 + 8 + a + b) / 5 = 17) : (a + b) / 2 = 33.5 := 
by
  sorry

end NUMINAMATH_GPT_average_of_a_b_l1274_127461


namespace NUMINAMATH_GPT_initial_average_runs_l1274_127453

theorem initial_average_runs (A : ℝ) (h : 10 * A + 65 = 11 * (A + 3)) : A = 32 :=
  by sorry

end NUMINAMATH_GPT_initial_average_runs_l1274_127453


namespace NUMINAMATH_GPT_smallest_solution_proof_l1274_127414

noncomputable def smallest_solution : ℝ :=
  let n := 11
  let a := 0.533
  n + a

theorem smallest_solution_proof :
  ∃ (x : ℝ), ⌊x^2⌋ - ⌊x⌋^2 = 21 ∧ x = smallest_solution :=
by
  use smallest_solution
  sorry

end NUMINAMATH_GPT_smallest_solution_proof_l1274_127414


namespace NUMINAMATH_GPT_birds_in_tree_l1274_127488

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) (h : initial_birds = 21.0) (h_flew : birds_flew_away = 14.0) : 
initial_birds - birds_flew_away = 7.0 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_birds_in_tree_l1274_127488


namespace NUMINAMATH_GPT_minimum_tasks_for_18_points_l1274_127401

def task_count (points : ℕ) : ℕ :=
  if points <= 9 then
    (points / 3) * 1
  else if points <= 15 then
    3 + (points - 9 + 2) / 3 * 2
  else
    3 + 4 + (points - 15 + 2) / 3 * 3

theorem minimum_tasks_for_18_points : task_count 18 = 10 := by
  sorry

end NUMINAMATH_GPT_minimum_tasks_for_18_points_l1274_127401


namespace NUMINAMATH_GPT_total_worth_of_presents_l1274_127449

-- Define the costs as given in the conditions
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def bracelet_cost : ℕ := 2 * ring_cost

-- Define the total worth of the presents
def total_worth : ℕ := ring_cost + car_cost + bracelet_cost

-- Statement: Prove the total worth is 14000
theorem total_worth_of_presents : total_worth = 14000 :=
by
  -- Here is the proof statement
  sorry

end NUMINAMATH_GPT_total_worth_of_presents_l1274_127449


namespace NUMINAMATH_GPT_sum_of_largest_three_l1274_127412

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end NUMINAMATH_GPT_sum_of_largest_three_l1274_127412


namespace NUMINAMATH_GPT_right_triangle_x_value_l1274_127428

variable (BM MA BC CA x h d : ℝ)

theorem right_triangle_x_value (BM MA BC CA x h d : ℝ)
  (h4 : BM + MA = BC + CA)
  (h5 : BM = x)
  (h6 : BC = h)
  (h7 : CA = d) :
  x = h * d / (2 * h + d) := 
sorry

end NUMINAMATH_GPT_right_triangle_x_value_l1274_127428


namespace NUMINAMATH_GPT_rational_expression_iff_rational_square_l1274_127498

theorem rational_expression_iff_rational_square (x : ℝ) :
  (∃ r : ℚ, x^2 + (Real.sqrt (x^4 + 1)) - 1 / (x^2 + (Real.sqrt (x^4 + 1))) = r) ↔
  (∃ q : ℚ, x^2 = q) := by
  sorry

end NUMINAMATH_GPT_rational_expression_iff_rational_square_l1274_127498


namespace NUMINAMATH_GPT_range_of_a_l1274_127416

-- Given definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x + 1 

-- Monotonicity condition on the interval [1, 2]
def is_monotonic (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → (x ≤ y → f x a ≤ f y a) ∨ (x ≤ y → f x a ≥ f y a)

-- The proof objective
theorem range_of_a (a : ℝ) : is_monotonic a → (a ≤ -2 ∨ a ≥ -1) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1274_127416


namespace NUMINAMATH_GPT_smallest_n_l1274_127463

-- Define the conditions as predicates
def condition1 (n : ℕ) : Prop := (n + 2018) % 2020 = 0
def condition2 (n : ℕ) : Prop := (n + 2020) % 2018 = 0

-- The main theorem statement using these conditions
theorem smallest_n (n : ℕ) : 
  (∃ n, condition1 n ∧ condition2 n ∧ (∀ m, condition1 m ∧ condition2 m → n ≤ m)) ↔ n = 2030102 := 
by 
    sorry

end NUMINAMATH_GPT_smallest_n_l1274_127463


namespace NUMINAMATH_GPT_share_of_B_is_2400_l1274_127470

noncomputable def share_of_B (total_profit : ℝ) (B_investment : ℝ) (A_months B_months C_months D_months : ℝ) : ℝ :=
  let A_investment := 3 * B_investment
  let C_investment := (3/2) * B_investment
  let D_investment := (1/2) * A_investment
  let A_inv_months := A_investment * A_months
  let B_inv_months := B_investment * B_months
  let C_inv_months := C_investment * C_months
  let D_inv_months := D_investment * D_months
  let total_inv_months := A_inv_months + B_inv_months + C_inv_months + D_inv_months
  (B_inv_months / total_inv_months) * total_profit

theorem share_of_B_is_2400 :
  share_of_B 27000 (1000 : ℝ) 12 6 9 8 = 2400 := 
sorry

end NUMINAMATH_GPT_share_of_B_is_2400_l1274_127470
