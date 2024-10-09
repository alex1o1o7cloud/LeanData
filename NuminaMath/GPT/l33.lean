import Mathlib

namespace ratio_of_dividends_l33_3355

-- Definitions based on conditions
def expected_earnings : ℝ := 0.80
def actual_earnings : ℝ := 1.10
def additional_per_increment : ℝ := 0.04
def increment_size : ℝ := 0.10

-- Definition for the base dividend D which remains undetermined
variable (D : ℝ)

-- Stating the theorem
theorem ratio_of_dividends 
  (h1 : actual_earnings = 1.10)
  (h2 : expected_earnings = 0.80)
  (h3 : additional_per_increment = 0.04)
  (h4 : increment_size = 0.10) :
  let additional_earnings := actual_earnings - expected_earnings
  let increments := additional_earnings / increment_size
  let additional_dividend := increments * additional_per_increment
  let total_dividend := D + additional_dividend
  let ratio := total_dividend / actual_earnings
  ratio = (D + 0.12) / 1.10 :=
by
  sorry

end ratio_of_dividends_l33_3355


namespace find_k_l33_3320

theorem find_k (x k : ℝ) (h₁ : (x^2 - k) * (x - k) = x^3 - k * (x^2 + x + 3))
               (h₂ : k ≠ 0) : k = -3 :=
by
  sorry

end find_k_l33_3320


namespace car_second_half_speed_l33_3386

theorem car_second_half_speed (D : ℝ) (V : ℝ) :
  let average_speed := 60  -- km/hr
  let first_half_speed := 75 -- km/hr
  average_speed = D / ((D / 2) / first_half_speed + (D / 2) / V) ->
  V = 150 :=
by
  sorry

end car_second_half_speed_l33_3386


namespace acres_left_untouched_l33_3335

def total_acres := 65057
def covered_acres := 64535

theorem acres_left_untouched : total_acres - covered_acres = 522 :=
by
  sorry

end acres_left_untouched_l33_3335


namespace trader_cloth_sold_l33_3347

variable (x : ℕ)
variable (profit_per_meter total_profit : ℕ)

theorem trader_cloth_sold (h_profit_per_meter : profit_per_meter = 55)
  (h_total_profit : total_profit = 2200) :
  55 * x = 2200 → x = 40 :=
by 
  sorry

end trader_cloth_sold_l33_3347


namespace nathan_dice_roll_probability_l33_3350

noncomputable def probability_nathan_rolls : ℚ :=
  let prob_less4_first_die : ℚ := 3 / 8
  let prob_greater5_second_die : ℚ := 3 / 8
  prob_less4_first_die * prob_greater5_second_die

theorem nathan_dice_roll_probability : probability_nathan_rolls = 9 / 64 := by
  sorry

end nathan_dice_roll_probability_l33_3350


namespace arithmetic_sequence_sum_l33_3384

variable {a : ℕ → ℕ}

-- Defining the arithmetic sequence condition
axiom arithmetic_sequence_condition : a 3 + a 7 = 37

-- The goal is to prove that the total of a_2 + a_4 + a_6 + a_8 is 74
theorem arithmetic_sequence_sum : a 2 + a 4 + a 6 + a 8 = 74 :=
by
  sorry

end arithmetic_sequence_sum_l33_3384


namespace system1_solution_system2_solution_l33_3343

-- System (1)
theorem system1_solution (x y : ℚ) (h1 : 4 * x + 8 * y = 12) (h2 : 3 * x - 2 * y = 5) :
  x = 2 ∧ y = 1 / 2 := by
  sorry

-- System (2)
theorem system2_solution (x y : ℚ) (h1 : (1/2) * x - (y + 1) / 3 = 1) (h2 : 6 * x + 2 * y = 10) :
  x = 2 ∧ y = -1 := by
  sorry

end system1_solution_system2_solution_l33_3343


namespace probability_of_rolling_3_or_5_is_1_over_4_l33_3323

def fair_8_sided_die := {outcome : Fin 8 // true}

theorem probability_of_rolling_3_or_5_is_1_over_4 :
  (1 / 4 : ℚ) = 2 / 8 :=
by sorry

end probability_of_rolling_3_or_5_is_1_over_4_l33_3323


namespace meet_second_time_4_5_minutes_l33_3302

-- Define the initial conditions
def opposite_ends := true      -- George and Henry start from opposite ends
def pass_in_center := 1.5      -- They pass each other in the center after 1.5 minutes
def no_time_lost := true       -- No time lost in turning
def constant_speeds := true    -- They maintain their respective speeds

-- Prove that they pass each other the second time after 4.5 minutes
theorem meet_second_time_4_5_minutes :
  opposite_ends ∧ pass_in_center = 1.5 ∧ no_time_lost ∧ constant_speeds → 
  ∃ t : ℝ, t = 4.5 := by
  sorry

end meet_second_time_4_5_minutes_l33_3302


namespace value_of_3W5_l33_3367

def W (a b : ℕ) : ℕ := b + 7 * a - a ^ 2

theorem value_of_3W5 : W 3 5 = 17 := by 
  sorry

end value_of_3W5_l33_3367


namespace exercise_l33_3316

noncomputable def g (x : ℝ) : ℝ := x^3
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem exercise : f (g 3) = 1457 := by
  sorry

end exercise_l33_3316


namespace least_whole_number_subtracted_from_ratio_l33_3346

theorem least_whole_number_subtracted_from_ratio (x : ℕ) : 
  (6 - x) / (7 - x) < 16 / 21 := by
  sorry

end least_whole_number_subtracted_from_ratio_l33_3346


namespace girl_scout_cookie_sales_l33_3310

theorem girl_scout_cookie_sales :
  ∃ C P : ℝ, C + P = 1585 ∧ 1.25 * C + 0.75 * P = 1586.25 ∧ P = 790 :=
by
  sorry

end girl_scout_cookie_sales_l33_3310


namespace largest_of_three_numbers_l33_3354

noncomputable def hcf := 23
noncomputable def factors := [11, 12, 13]

/-- The largest of the three numbers, given the H.C.F is 23 and the other factors of their L.C.M are 11, 12, and 13, is 39468. -/
theorem largest_of_three_numbers : hcf * factors.prod = 39468 := by
  sorry

end largest_of_three_numbers_l33_3354


namespace largest_divisor_of_even_n_cube_difference_l33_3397

theorem largest_divisor_of_even_n_cube_difference (n : ℤ) (h : Even n) : 6 ∣ (n^3 - n) := by
  sorry

end largest_divisor_of_even_n_cube_difference_l33_3397


namespace expand_expression_l33_3395

variable {R : Type} [CommRing R]
variable (a b x : R)

theorem expand_expression (a b x : R) :
  (a * x^2 + b) * (5 * x^3) = 35 * x^5 + (-15) * x^3 :=
by
  -- The proof goes here
  sorry

end expand_expression_l33_3395


namespace fill_two_thirds_of_bucket_time_l33_3372

theorem fill_two_thirds_of_bucket_time (fill_entire_bucket_time : ℝ) (h : fill_entire_bucket_time = 3) : (2 / 3) * fill_entire_bucket_time = 2 :=
by 
  sorry

end fill_two_thirds_of_bucket_time_l33_3372


namespace speed_of_first_half_of_journey_l33_3368

theorem speed_of_first_half_of_journey
  (total_time : ℝ)
  (speed_second_half : ℝ)
  (total_distance : ℝ)
  (first_half_distance : ℝ)
  (second_half_distance : ℝ)
  (time_second_half : ℝ)
  (time_first_half : ℝ)
  (speed_first_half : ℝ) :
  total_time = 15 →
  speed_second_half = 24 →
  total_distance = 336 →
  first_half_distance = total_distance / 2 →
  second_half_distance = total_distance / 2 →
  time_second_half = second_half_distance / speed_second_half →
  time_first_half = total_time - time_second_half →
  speed_first_half = first_half_distance / time_first_half →
  speed_first_half = 21 :=
by intros; sorry

end speed_of_first_half_of_journey_l33_3368


namespace no_such_a_b_exists_l33_3341

open Set

def A (a b : ℝ) : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, a * x + b) }

def B : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, 3 * (x : ℝ) ^ 2 + 15) }

def C : Set (ℝ × ℝ) :=
  { p | p.1 ^ 2 + p.2 ^ 2 ≤ 144 }

theorem no_such_a_b_exists :
  ¬ ∃ (a b : ℝ), 
    ((A a b ∩ B).Nonempty) ∧ ((a, b) ∈ C) :=
sorry

end no_such_a_b_exists_l33_3341


namespace find_x_l33_3370

-- Define the condition from the problem statement
def condition1 (x : ℝ) : Prop := 70 = 0.60 * x + 22

-- Translate the question to the Lean statement form
theorem find_x (x : ℝ) (h : condition1 x) : x = 80 :=
by {
  sorry
}

end find_x_l33_3370


namespace peter_bought_large_glasses_l33_3308

-- Define the conditions as Lean definitions
def total_money : ℕ := 50
def cost_small_glass : ℕ := 3
def cost_large_glass : ℕ := 5
def small_glasses_bought : ℕ := 8
def change_left : ℕ := 1

-- Define the number of large glasses bought
def large_glasses_bought (total_money : ℕ) (cost_small_glass : ℕ) (cost_large_glass : ℕ) (small_glasses_bought : ℕ) (change_left : ℕ) : ℕ :=
  let total_spent := total_money - change_left
  let spent_on_small := cost_small_glass * small_glasses_bought
  let spent_on_large := total_spent - spent_on_small
  spent_on_large / cost_large_glass

-- The theorem to be proven
theorem peter_bought_large_glasses : large_glasses_bought total_money cost_small_glass cost_large_glass small_glasses_bought change_left = 5 :=
by
  sorry

end peter_bought_large_glasses_l33_3308


namespace hyperbola_asymptotes_l33_3396

variable (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)

theorem hyperbola_asymptotes (e : ℝ) (h_ecc : e = (Real.sqrt 5) / 2)
  (h_hyperbola : e = Real.sqrt (1 + (b^2 / a^2))) :
  (∀ x : ℝ, y = x * (b / a) ∨ y = -x * (b / a)) :=
by
  -- Here, the proof would follow logically from the given conditions.
  sorry

end hyperbola_asymptotes_l33_3396


namespace video_streaming_budget_l33_3326

theorem video_streaming_budget 
  (weekly_food_budget : ℕ) 
  (weeks : ℕ) 
  (total_food_budget : ℕ) 
  (rent : ℕ) 
  (phone : ℕ) 
  (savings_rate : ℝ)
  (total_savings : ℕ) 
  (total_expenses : ℕ) 
  (known_expenses: ℕ) 
  (total_spending : ℕ):
  weekly_food_budget = 100 →
  weeks = 4 →
  total_food_budget = weekly_food_budget * weeks →
  rent = 1500 →
  phone = 50 →
  savings_rate = 0.10 →
  total_savings = 198 →
  total_expenses = total_food_budget + rent + phone →
  total_spending = (total_savings : ℝ) / savings_rate →
  known_expenses = total_expenses →
  total_spending - known_expenses = 30 :=
by sorry

end video_streaming_budget_l33_3326


namespace add_to_fraction_eq_l33_3342

theorem add_to_fraction_eq (n : ℤ) : (4 + n : ℤ) / (7 + n) = (2 : ℤ) / 3 → n = 2 := 
by {
  sorry
}

end add_to_fraction_eq_l33_3342


namespace evaluate_expression_l33_3311

theorem evaluate_expression (x : ℝ) (h : x = 2) : 4 * x ^ 2 + 1 / 2 = 16.5 := by
  sorry

end evaluate_expression_l33_3311


namespace possible_values_sin_plus_cos_l33_3333

variable (x : ℝ)

theorem possible_values_sin_plus_cos (h : 2 * Real.cos x - 3 * Real.sin x = 2) :
    ∃ (values : Set ℝ), values = {3, -31 / 13} ∧ (Real.sin x + 3 * Real.cos x) ∈ values := by
  sorry

end possible_values_sin_plus_cos_l33_3333


namespace largest_lcm_18_l33_3330

theorem largest_lcm_18 :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 :=
by sorry

end largest_lcm_18_l33_3330


namespace number_of_ways_to_select_officers_l33_3325

-- Definitions based on conditions
def boys : ℕ := 6
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def officers_to_select : ℕ := 3

-- Number of ways to choose 3 individuals out of 10
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_choices : ℕ := choose total_people officers_to_select

-- Number of ways to choose 3 boys out of 6 (0 girls)
def all_boys_choices : ℕ := choose boys officers_to_select

-- Number of ways to choose at least 1 girl
def at_least_one_girl_choices : ℕ := total_choices - all_boys_choices

-- Theorem to prove the number of ways to select the officers
theorem number_of_ways_to_select_officers :
  at_least_one_girl_choices = 100 := by
  sorry

end number_of_ways_to_select_officers_l33_3325


namespace correct_statements_for_function_l33_3324

-- Definitions and the problem statement
def f (x b c : ℝ) := x * |x| + b * x + c

theorem correct_statements_for_function (b c : ℝ) :
  (c = 0 → ∀ x, f x b c = -f (-x) b c) ∧
  (b = 0 ∧ c > 0 → ∀ x, f x b c = 0 → x = 0) ∧
  (∀ x, f x b c = f (-x) b (-c)) :=
sorry

end correct_statements_for_function_l33_3324


namespace proteges_57_l33_3393

def divisors (n : ℕ) : List ℕ := (List.range (n + 1)).filter (λ d => n % d = 0)

def units_digit (n : ℕ) : ℕ := n % 10

def proteges (n : ℕ) : List ℕ := (divisors n).map units_digit

theorem proteges_57 : proteges 57 = [1, 3, 9, 7] :=
sorry

end proteges_57_l33_3393


namespace fish_caught_by_twentieth_fisherman_l33_3392

theorem fish_caught_by_twentieth_fisherman :
  ∀ (total_fishermen total_fish fish_per_fisherman nineten_fishermen : ℕ),
  total_fishermen = 20 →
  total_fish = 10000 →
  fish_per_fisherman = 400 →
  nineten_fishermen = 19 →
  (total_fishermen * fish_per_fisherman) - (nineten_fishermen * fish_per_fisherman) = 2400 :=
by
  intros
  sorry

end fish_caught_by_twentieth_fisherman_l33_3392


namespace second_hand_travel_distance_l33_3300

theorem second_hand_travel_distance (r : ℝ) (minutes : ℝ) (π : ℝ) (h : r = 10 ∧ minutes = 45 ∧ π = Real.pi) : 
  (minutes / 60) * 60 * (2 * π * r) = 900 * π := 
by sorry

end second_hand_travel_distance_l33_3300


namespace time_for_second_train_to_cross_l33_3361

def length_first_train : ℕ := 100
def speed_first_train : ℕ := 10
def length_second_train : ℕ := 150
def speed_second_train : ℕ := 15
def distance_between_trains : ℕ := 50

def total_distance : ℕ := length_first_train + length_second_train + distance_between_trains
def relative_speed : ℕ := speed_second_train - speed_first_train

theorem time_for_second_train_to_cross :
  total_distance / relative_speed = 60 :=
by
  -- Definitions and intermediate steps would be handled in the proof here
  sorry

end time_for_second_train_to_cross_l33_3361


namespace shortest_distance_correct_l33_3383

noncomputable def shortest_distance_a_to_c1 (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c)

theorem shortest_distance_correct (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  shortest_distance_a_to_c1 a b c h₁ h₂ = Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c) :=
by
  -- This is where the proof would go.
  sorry

end shortest_distance_correct_l33_3383


namespace linear_function_difference_l33_3366

variable (g : ℝ → ℝ)
variable (h_linear : ∀ x y, g (x + y) = g x + g y)
variable (h_value : g 8 - g 4 = 16)

theorem linear_function_difference : g 16 - g 4 = 48 := by
  sorry

end linear_function_difference_l33_3366


namespace sqrt_defined_iff_ge_neg1_l33_3313

theorem sqrt_defined_iff_ge_neg1 (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x + 1)) ↔ x ≥ -1 := by
  sorry

end sqrt_defined_iff_ge_neg1_l33_3313


namespace solution_set_of_inequality_l33_3305

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 - 7*x + 12 < 0 ↔ 3 < x ∧ x < 4 :=
by {
  sorry
}

end solution_set_of_inequality_l33_3305


namespace problem_xyz_inequality_l33_3352

theorem problem_xyz_inequality (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h_eq : x^2 + y^2 + z^2 + x * y * z = 4) :
  x * y * z ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ x * y * z + 2 :=
by 
  sorry

end problem_xyz_inequality_l33_3352


namespace M_gt_N_l33_3303

variable (a : ℝ)

def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

theorem M_gt_N : M a > N a := by
  sorry

end M_gt_N_l33_3303


namespace proof_f_2017_l33_3328

-- Define the conditions provided in the problem
variable (f : ℝ → ℝ)
variable (hf : ∀ x, f (-x) = -f x) -- f is an odd function
variable (h1 : ∀ x, f (-x + 1) = f (x + 1))
variable (h2 : f (-1) = 1)

-- Define the Lean statement that proves the correct answer
theorem proof_f_2017 : f 2017 = -1 :=
sorry

end proof_f_2017_l33_3328


namespace trash_can_ratio_l33_3391

theorem trash_can_ratio (streets_trash_cans total_trash_cans : ℕ) 
(h_streets : streets_trash_cans = 14) 
(h_total : total_trash_cans = 42) : 
(total_trash_cans - streets_trash_cans) / streets_trash_cans = 2 :=
by {
  sorry
}

end trash_can_ratio_l33_3391


namespace area_of_hexagon_l33_3379

def isRegularHexagon (A B C D E F : Type) : Prop := sorry
def isInsideQuadrilateral (P : Type) (A B C D : Type) : Prop := sorry
def areaTriangle (P X Y : Type) : Real := sorry

theorem area_of_hexagon (A B C D E F P : Type)
    (h1 : isRegularHexagon A B C D E F)
    (h2 : isInsideQuadrilateral P A B C D)
    (h3 : areaTriangle P B C = 20)
    (h4 : areaTriangle P A D = 23) :
    ∃ area : Real, area = 189 :=
sorry

end area_of_hexagon_l33_3379


namespace find_x_l33_3381

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 135) (h2 : x > 0) : x = 11.25 :=
by
  sorry

end find_x_l33_3381


namespace max_distinct_counts_proof_l33_3377

-- Define the number of boys (B) and girls (G)
def B : ℕ := 29
def G : ℕ := 15

-- Define the maximum distinct dance counts achievable
def max_distinct_counts : ℕ := 29

-- The theorem to prove
theorem max_distinct_counts_proof:
  ∃ (distinct_counts : ℕ), distinct_counts = max_distinct_counts ∧ distinct_counts <= B + G := 
by
  sorry

end max_distinct_counts_proof_l33_3377


namespace find_pq_l33_3340

noncomputable def p_and_q (p q : ℝ) := 
  (Complex.I * 2 - 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0} ∧ 
  - (Complex.I * 2 + 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0}

theorem find_pq : ∃ (p q : ℝ), p_and_q p q ∧ p + q = 38 :=
by
  sorry

end find_pq_l33_3340


namespace f_neg4_plus_f_0_range_of_a_l33_3321

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else if x < 0 then -log (-x) / log 2 else 0

/- Prove that f(-4) + f(0) = -2 given the function properties -/
theorem f_neg4_plus_f_0 : f (-4) + f 0 = -2 :=
sorry

/- Prove the range of a such that f(a) > f(-a) is a > 1 or -1 < a < 0 given the function properties -/
theorem range_of_a (a : ℝ) : f a > f (-a) ↔ a > 1 ∨ (-1 < a ∧ a < 0) :=
sorry

end f_neg4_plus_f_0_range_of_a_l33_3321


namespace beth_cans_of_corn_l33_3319

theorem beth_cans_of_corn (C P : ℕ) (h1 : P = 2 * C + 15) (h2 : P = 35) : C = 10 :=
by
  sorry

end beth_cans_of_corn_l33_3319


namespace sum_of_base4_numbers_is_correct_l33_3337

-- Define the four base numbers
def n1 : ℕ := 2 * 4^2 + 1 * 4^1 + 2 * 4^0
def n2 : ℕ := 1 * 4^2 + 0 * 4^1 + 3 * 4^0
def n3 : ℕ := 3 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Define the expected sum in base 4 interpreted as a natural number
def expected_sum : ℕ := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

-- State the theorem
theorem sum_of_base4_numbers_is_correct : n1 + n2 + n3 = expected_sum := by
  sorry

end sum_of_base4_numbers_is_correct_l33_3337


namespace solution_set_inequality_l33_3345

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := by
  sorry

end solution_set_inequality_l33_3345


namespace fx_root_and_decreasing_l33_3359

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - Real.log x / Real.log 2

theorem fx_root_and_decreasing (a x0 : ℝ) (h0 : 0 < a) (hx0 : 0 < x0) (h_cond : a < x0) (hf_root : f x0 = 0) 
  (hf_decreasing : ∀ x y : ℝ, x < y → f y < f x) : f a > 0 := 
sorry

end fx_root_and_decreasing_l33_3359


namespace division_of_expression_l33_3374

theorem division_of_expression (x y : ℝ) (hy : y ≠ 0) (hx : x ≠ 0) : (12 * x^2 * y) / (-6 * x * y) = -2 * x := by
  sorry

end division_of_expression_l33_3374


namespace exactly_two_roots_iff_l33_3378

theorem exactly_two_roots_iff (a : ℝ) : 
  (∃! (x : ℝ), x^2 + 2 * x + 2 * |x + 1| = a) ↔ a > -1 :=
by
  sorry

end exactly_two_roots_iff_l33_3378


namespace percentage_emails_moved_to_work_folder_l33_3398

def initialEmails : ℕ := 400
def trashedEmails : ℕ := initialEmails / 2
def remainingEmailsAfterTrash : ℕ := initialEmails - trashedEmails
def emailsLeftInInbox : ℕ := 120
def emailsMovedToWorkFolder : ℕ := remainingEmailsAfterTrash - emailsLeftInInbox

theorem percentage_emails_moved_to_work_folder :
  (emailsMovedToWorkFolder * 100 / remainingEmailsAfterTrash) = 40 := by
  sorry

end percentage_emails_moved_to_work_folder_l33_3398


namespace max_andy_l33_3371

def max_cookies_eaten_by_andy (total : ℕ) (k1 k2 a b c : ℤ) : Prop :=
  a + b + c = total ∧ b = 2 * a + 2 ∧ c = a - 3

theorem max_andy (total : ℕ) (a : ℤ) :
  (∀ b c, max_cookies_eaten_by_andy total 2 (-3) a b c) → a ≤ 7 :=
by
  intros H
  sorry

end max_andy_l33_3371


namespace expression_equality_l33_3364

theorem expression_equality : 1 + 2 / (3 + 4 / 5) = 29 / 19 := by
  sorry

end expression_equality_l33_3364


namespace problem_statement_l33_3356

theorem problem_statement {x y z : ℝ} (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  1 < 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) ∧ 1 / (1 + x) + 1 / (1 + y) + 1 / (1 + z) < 2 :=
by
  sorry

end problem_statement_l33_3356


namespace find_digit_D_l33_3334

def is_digit (n : ℕ) : Prop := n < 10

theorem find_digit_D (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C)
  (h5 : B ≠ D) (h6 : C ≠ D) (h7 : is_digit A) (h8 : is_digit B) (h9 : is_digit C) (h10 : is_digit D) :
  (1000 * A + 100 * B + 10 * C + D) * 2 = 5472 → D = 6 := 
by
  sorry

end find_digit_D_l33_3334


namespace opposite_of_abs_frac_l33_3389

theorem opposite_of_abs_frac (h : 0 < (1 : ℝ) / 2023) : -|((1 : ℝ) / 2023)| = -(1 / 2023) := by
  sorry

end opposite_of_abs_frac_l33_3389


namespace geometric_progression_first_term_l33_3338

theorem geometric_progression_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 8) 
  (h2 : a + a * r = 5) : 
  a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) := 
by sorry

end geometric_progression_first_term_l33_3338


namespace find_angle_A_l33_3307

theorem find_angle_A (A B C a b c : ℝ) 
  (h_triangle: a = Real.sqrt 2)
  (h_sides: b = 2 * Real.sin B + Real.cos B)
  (h_b_eq: b = Real.sqrt 2)
  (h_a_lt_b: a < b)
  : A = Real.pi / 6 := sorry

end find_angle_A_l33_3307


namespace probabilities_equal_l33_3360

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end probabilities_equal_l33_3360


namespace curve_meets_line_once_l33_3373

theorem curve_meets_line_once (a : ℝ) (h : a > 0) :
  (∃! P : ℝ × ℝ, (∃ θ : ℝ, P.1 = a + 4 * Real.cos θ ∧ P.2 = 1 + 4 * Real.sin θ)
  ∧ (3 * P.1 + 4 * P.2 = 5)) → a = 7 :=
sorry

end curve_meets_line_once_l33_3373


namespace maximum_lambda_l33_3304

theorem maximum_lambda (a b : ℝ) : (27 / 4) * a^2 * b^2 * (a + b)^2 ≤ (a^2 + a * b + b^2)^3 := 
sorry

end maximum_lambda_l33_3304


namespace parallel_lines_slope_eq_l33_3315

theorem parallel_lines_slope_eq (m : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y - 3 = 0 → 6 * x + m * y + 1 = 0) → m = 4 :=
by
  sorry

end parallel_lines_slope_eq_l33_3315


namespace ratio_of_abc_l33_3385

theorem ratio_of_abc (a b c : ℝ) (h1 : a ≠ 0) (h2 : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : a / b = 1 / 2 ∧ a / c = 1 / 3 := 
sorry

end ratio_of_abc_l33_3385


namespace car_return_speed_l33_3301

variable (d : ℕ) (r : ℕ)
variable (H0 : d = 180)
variable (H1 : ∀ t1 : ℕ, t1 = d / 90)
variable (H2 : ∀ t2 : ℕ, t2 = d / r)
variable (H3 : ∀ avg_rate : ℕ, avg_rate = 2 * d / (d / 90 + d / r))
variable (H4 : avg_rate = 60)

theorem car_return_speed : r = 45 :=
by sorry

end car_return_speed_l33_3301


namespace markers_last_group_correct_l33_3357

-- Definition of conditions in Lean 4
def total_students : ℕ := 30
def boxes_of_markers : ℕ := 22
def markers_per_box : ℕ := 5
def students_in_first_group : ℕ := 10
def markers_per_student_first_group : ℕ := 2
def students_in_second_group : ℕ := 15
def markers_per_student_second_group : ℕ := 4

-- Calculate total markers allocated to the first and second groups
def markers_used_by_first_group : ℕ := students_in_first_group * markers_per_student_first_group
def markers_used_by_second_group : ℕ := students_in_second_group * markers_per_student_second_group

-- Total number of markers available
def total_markers : ℕ := boxes_of_markers * markers_per_box

-- Markers left for last group
def markers_remaining : ℕ := total_markers - (markers_used_by_first_group + markers_used_by_second_group)

-- Number of students in the last group
def students_in_last_group : ℕ := total_students - (students_in_first_group + students_in_second_group)

-- Number of markers per student in the last group
def markers_per_student_last_group : ℕ := markers_remaining / students_in_last_group

-- The proof problem in Lean 4
theorem markers_last_group_correct : markers_per_student_last_group = 6 :=
  by
  -- Proof is to be filled here
  sorry

end markers_last_group_correct_l33_3357


namespace max_regions_quadratic_trinomials_l33_3358

theorem max_regions_quadratic_trinomials (a b c : Fin 100 → ℝ) :
  ∃ R, (∀ (n : ℕ), n ≤ 100 → R = n^2 + 1) → R = 10001 := 
  sorry

end max_regions_quadratic_trinomials_l33_3358


namespace value_of_m_l33_3375

-- Definitions of the conditions
def base6_num (m : ℕ) : ℕ := 2 + m * 6^2
def dec_num (d : ℕ) := d = 146

-- Theorem to prove
theorem value_of_m (m : ℕ) (h1 : base6_num m = 146) : m = 4 := 
sorry

end value_of_m_l33_3375


namespace quadratic_roots_range_l33_3339

theorem quadratic_roots_range (m : ℝ) :
  (∃ p n : ℝ, p > 0 ∧ n < 0 ∧ 2 * p^2 + (m + 1) * p + m = 0 ∧ 2 * n^2 + (m + 1) * n + m = 0) →
  m < 0 :=
by
  sorry

end quadratic_roots_range_l33_3339


namespace div_neg_cancel_neg_div_example_l33_3344

theorem div_neg_cancel (x y : Int) (h : y ≠ 0) : (-x) / (-y) = x / y := by
  sorry

theorem neg_div_example : (-64 : Int) / (-32) = 2 := by
  apply div_neg_cancel
  norm_num

end div_neg_cancel_neg_div_example_l33_3344


namespace parabolas_intersect_l33_3390

theorem parabolas_intersect :
  let eq1 (x : ℝ) := 3 * x^2 - 4 * x + 2
  let eq2 (x : ℝ) := -x^2 + 6 * x + 8
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = -0.5 ∧ y = 4.75) ∧
  (∃ x y : ℝ, y = eq1 x ∧ y = eq2 x ∧ x = 3 ∧ y = 17) :=
by sorry

end parabolas_intersect_l33_3390


namespace proposition_A_l33_3348

variables {m n : Line} {α β : Plane}

def parallel (x y : Line) : Prop := sorry -- definition for parallel lines
def perpendicular (x : Line) (P : Plane) : Prop := sorry -- definition for perpendicular line to plane
def parallel_planes (P Q : Plane) : Prop := sorry -- definition for parallel planes

theorem proposition_A (hmn : parallel m n) (hperp_mα : perpendicular m α) (hperp_nβ : perpendicular n β) : parallel_planes α β :=
sorry

end proposition_A_l33_3348


namespace employees_count_l33_3380

-- Let E be the number of employees excluding the manager
def E (employees : ℕ) : ℕ := employees

-- Let T be the total salary of employees excluding the manager
def T (employees : ℕ) : ℕ := employees * 1500

-- Conditions given in the problem
def average_salary (employees : ℕ) : ℕ := T employees / E employees
def new_average_salary (employees : ℕ) : ℕ := (T employees + 22500) / (E employees + 1)

theorem employees_count : (average_salary employees = 1500) ∧ (new_average_salary employees = 2500) ∧ (manager_salary = 22500) → (E employees = 20) :=
  by sorry

end employees_count_l33_3380


namespace reciprocal_and_fraction_l33_3332

theorem reciprocal_and_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : (2/5) * a = 20) : 
  b = (1/a) ∧ (1/3) * a = (50/3) := 
by 
  sorry

end reciprocal_and_fraction_l33_3332


namespace correct_propositions_l33_3327

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_propositions :
  ¬ ∀ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 → ∃ k : ℤ, x1 - x2 = k * Real.pi ∧
  (∀ (x : ℝ), f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (- (Real.pi / 6)) = 0) ∧
  ¬ ∀ (x : ℝ), f x = f (-x - Real.pi / 6) :=
sorry

end correct_propositions_l33_3327


namespace area_of_shaded_region_l33_3317

theorem area_of_shaded_region
  (r_large : ℝ) (r_small : ℝ) (n_small : ℕ) (π : ℝ)
  (A_large : ℝ) (A_small : ℝ) (A_7_small : ℝ) (A_shaded : ℝ)
  (h1 : r_large = 20)
  (h2 : r_small = 10)
  (h3 : n_small = 7)
  (h4 : π = 3.14)
  (h5 : A_large = π * r_large^2)
  (h6 : A_small = π * r_small^2)
  (h7 : A_7_small = n_small * A_small)
  (h8 : A_shaded = A_large - A_7_small) :
  A_shaded = 942 :=
by
  sorry

end area_of_shaded_region_l33_3317


namespace find_value_of_p_l33_3399

theorem find_value_of_p (p : ℝ) :
  (∀ x y, (x = 0 ∧ y = -2) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 1/2 ∧ y = 0) → y = p*x^2 + 5*x + p) ∧
  (∀ x y, (x = 2 ∧ y = 0) → y = p*x^2 + 5*x + p) →
  p = -2 :=
by
  sorry

end find_value_of_p_l33_3399


namespace original_price_of_dish_l33_3322

theorem original_price_of_dish : 
  ∀ (P : ℝ), 
  1.05 * P - 1.035 * P = 0.54 → 
  P = 36 :=
by
  intros P h
  sorry

end original_price_of_dish_l33_3322


namespace interest_years_l33_3376

theorem interest_years (P : ℝ) (R : ℝ) (N : ℝ) (H1 : P = 2400) (H2 : (P * (R + 1) * N) / 100 - (P * R * N) / 100 = 72) : N = 3 :=
by
  -- Proof can be filled in here
  sorry

end interest_years_l33_3376


namespace max_hours_at_regular_rate_l33_3363

-- Define the maximum hours at regular rate H
def max_regular_hours (H : ℕ) : Prop := 
  let regular_rate := 16
  let overtime_rate := 16 + (0.75 * 16)
  let total_hours := 60
  let total_compensation := 1200
  16 * H + 28 * (total_hours - H) = total_compensation

theorem max_hours_at_regular_rate : ∃ H, max_regular_hours H ∧ H = 40 :=
sorry

end max_hours_at_regular_rate_l33_3363


namespace max_area_l33_3351

theorem max_area (l w : ℝ) (h : l + 3 * w = 500) : l * w ≤ 62500 :=
by
  sorry

end max_area_l33_3351


namespace find_box_depth_l33_3349

-- Definitions and conditions
noncomputable def length : ℝ := 1.6
noncomputable def width : ℝ := 1.0
noncomputable def edge : ℝ := 0.2
noncomputable def number_of_blocks : ℝ := 120

-- The goal is to find the depth of the box
theorem find_box_depth (d : ℝ) :
  length * width * d = number_of_blocks * (edge ^ 3) →
  d = 0.6 := 
sorry

end find_box_depth_l33_3349


namespace hyperbola_asymptote_l33_3312

theorem hyperbola_asymptote :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (y = (1/2) * x) ∨ (y = -(1/2) * x) :=
by
  intros x y h
  sorry

end hyperbola_asymptote_l33_3312


namespace negation_of_p_l33_3306

def p : Prop := ∀ x : ℝ, x^2 - 2*x + 2 ≤ Real.sin x
def not_p : Prop := ∃ x : ℝ, x^2 - 2*x + 2 > Real.sin x

theorem negation_of_p : ¬ p ↔ not_p := by
  sorry

end negation_of_p_l33_3306


namespace combined_weight_l33_3362

variables (G D C : ℝ)

def grandmother_weight (G D C : ℝ) := G + D + C = 150
def daughter_weight (D : ℝ) := D = 42
def child_weight (G C : ℝ) := C = 1/5 * G

theorem combined_weight (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_weight D) (h3 : child_weight G C) : D + C = 60 :=
by
  sorry

end combined_weight_l33_3362


namespace workout_total_correct_l33_3387

structure Band := 
  (A : ℕ) 
  (B : ℕ) 
  (C : ℕ)

structure Equipment := 
  (leg_weight_squat : ℕ) 
  (dumbbell : ℕ) 
  (leg_weight_lunge : ℕ) 
  (kettlebell : ℕ)

def total_weight (bands : Band) (equip : Equipment) : ℕ := 
  let squat_total := bands.A + bands.B + bands.C + (2 * equip.leg_weight_squat) + equip.dumbbell
  let lunge_total := bands.A + bands.C + (2 * equip.leg_weight_lunge) + equip.kettlebell
  squat_total + lunge_total

theorem workout_total_correct (bands : Band) (equip : Equipment) : 
  bands = ⟨7, 5, 3⟩ → 
  equip = ⟨10, 15, 8, 18⟩ → 
  total_weight bands equip = 94 :=
by 
  -- Insert your proof steps here
  sorry

end workout_total_correct_l33_3387


namespace part_1_part_2_l33_3353

def f (x a : ℝ) : ℝ := |x - a| + 5 * x

theorem part_1 (x : ℝ) : (|x + 1| + 5 * x ≤ 5 * x + 3) ↔ (x ∈ Set.Icc (-4 : ℝ) 2) :=
by
  sorry

theorem part_2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) :=
by
  sorry

end part_1_part_2_l33_3353


namespace carl_garden_area_l33_3309

theorem carl_garden_area (x : ℕ) (longer_side_post_count : ℕ) (total_posts : ℕ) 
  (shorter_side_length : ℕ) (longer_side_length : ℕ) 
  (posts_per_gap : ℕ) (spacing : ℕ) :
  -- Conditions
  total_posts = 20 → 
  posts_per_gap = 4 → 
  spacing = 4 → 
  longer_side_post_count = 2 * x → 
  2 * x + 2 * (2 * x) - 4 = total_posts →
  shorter_side_length = (x - 1) * spacing → 
  longer_side_length = (longer_side_post_count - 1) * spacing →
  -- Conclusion
  shorter_side_length * longer_side_length = 336 :=
by
  sorry

end carl_garden_area_l33_3309


namespace sqrt_9_eq_3_or_neg3_l33_3369

theorem sqrt_9_eq_3_or_neg3 :
  { x : ℝ | x^2 = 9 } = {3, -3} :=
sorry

end sqrt_9_eq_3_or_neg3_l33_3369


namespace money_raised_is_correct_l33_3382

noncomputable def total_money_raised : ℝ :=
  let ticket_sales := 120 * 2.50 + 80 * 4.50 + 40 * 8.00 + 15 * 14.00
  let donations := 3 * 20.00 + 2 * 55.00 + 75.00 + 95.00 + 150.00
  ticket_sales + donations

theorem money_raised_is_correct :
  total_money_raised = 1680 := by
  sorry

end money_raised_is_correct_l33_3382


namespace money_left_after_purchase_l33_3329

noncomputable def total_cost : ℝ := 250 + 25 + 35 + 45 + 90

def savings_erika : ℝ := 155

noncomputable def savings_rick : ℝ := total_cost / 2

def savings_sam : ℝ := 175

def combined_cost_cake_flowers_skincare : ℝ := 25 + 35 + 45

noncomputable def savings_amy : ℝ := 2 * combined_cost_cake_flowers_skincare

noncomputable def total_savings : ℝ := savings_erika + savings_rick + savings_sam + savings_amy

noncomputable def money_left : ℝ := total_savings - total_cost

theorem money_left_after_purchase : money_left = 317.5 := by
  sorry

end money_left_after_purchase_l33_3329


namespace smallest_positive_multiple_45_l33_3336

theorem smallest_positive_multiple_45 (x : ℕ) (h : x > 0) : ∃ (y : ℕ), y > 0 ∧ 45 * y = 45 :=
by
  use 1
  sorry

end smallest_positive_multiple_45_l33_3336


namespace f_properties_l33_3388

theorem f_properties (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^2) - f (y^2) ≤ (f x + y) * (x - f y)) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end f_properties_l33_3388


namespace sin_225_plus_alpha_l33_3394

theorem sin_225_plus_alpha (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = 5 / 13) :
    Real.sin (5 * Real.pi / 4 + α) = -5 / 13 :=
by
  sorry

end sin_225_plus_alpha_l33_3394


namespace sum_of_distinct_integers_eq_zero_l33_3314

theorem sum_of_distinct_integers_eq_zero 
  (a b c d : ℤ) 
  (distinct : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  (prod_eq_25 : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end sum_of_distinct_integers_eq_zero_l33_3314


namespace knights_and_liars_l33_3331

/--
Suppose we have a set of natives, each of whom is either a liar or a knight.
Each native declares to all others: "You are all liars."
This setup implies that there must be exactly one knight among them.
-/
theorem knights_and_liars (natives : Type) (is_knight : natives → Prop) (is_liar : natives → Prop)
  (h1 : ∀ x, is_knight x ∨ is_liar x) 
  (h2 : ∀ x y, x ≠ y → (is_knight x → is_liar y) ∧ (is_liar x → is_knight y))
  : ∃! x, is_knight x :=
by
  sorry

end knights_and_liars_l33_3331


namespace power_sum_l33_3365

theorem power_sum (n : ℕ) : (-2 : ℤ)^n + (-2 : ℤ)^(n+1) = 2^n := by
  sorry

end power_sum_l33_3365


namespace range_of_f_l33_3318

noncomputable def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ ({0, 1, 2, 3} : Finset ℕ), f x = y} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l33_3318
