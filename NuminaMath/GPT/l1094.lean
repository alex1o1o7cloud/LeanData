import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1094_109434

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ),
    (∀ (n : ℕ), a_n n = 1 + (n - 1) * d) →  -- first condition
    d ≠ 0 →  -- second condition
    (∀ (n : ℕ), S_n n = n / 2 * (2 * 1 + (n - 1) * d)) →  -- third condition
    (1 * (1 + 4 * d) = (1 + d) ^ 2) →  -- fourth condition
    S_n 8 = 64 :=  -- conclusion
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1094_109434


namespace NUMINAMATH_GPT_gain_percentage_l1094_109440

variables (C S : ℝ) (hC : C > 0)
variables (hS : S > 0)

def cost_price := 25 * C
def selling_price := 25 * S
def gain := 10 * S 

theorem gain_percentage (h_eq : 25 * S = 25 * C + 10 * S):
  (S = C) → 
  ((gain / cost_price) * 100 = 40) :=
by
  sorry

end NUMINAMATH_GPT_gain_percentage_l1094_109440


namespace NUMINAMATH_GPT_Wayne_blocks_l1094_109436

theorem Wayne_blocks (initial_blocks : ℕ) (additional_blocks : ℕ) (total_blocks : ℕ) 
  (h1 : initial_blocks = 9) (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 :=
by {
  -- h1: initial_blocks = 9
  -- h2: additional_blocks = 6
  -- h3: total_blocks = initial_blocks + additional_blocks
  sorry
}

end NUMINAMATH_GPT_Wayne_blocks_l1094_109436


namespace NUMINAMATH_GPT_minor_axis_length_of_ellipse_l1094_109424

theorem minor_axis_length_of_ellipse :
  ∀ (x y : ℝ), (9 * x^2 + y^2 = 36) → 4 = 4 :=
by
  intros x y h
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_minor_axis_length_of_ellipse_l1094_109424


namespace NUMINAMATH_GPT_gcd_m_n_l1094_109409

-- Define the numbers m and n
def m : ℕ := 555555555
def n : ℕ := 1111111111

-- State the problem: Prove that gcd(m, n) = 1
theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_gcd_m_n_l1094_109409


namespace NUMINAMATH_GPT_min_b_factors_l1094_109462

theorem min_b_factors (x r s b : ℕ) (h : r * s = 1998) (fact : (x + r) * (x + s) = x^2 + b * x + 1998) : b = 91 :=
sorry

end NUMINAMATH_GPT_min_b_factors_l1094_109462


namespace NUMINAMATH_GPT_range_of_values_for_a_l1094_109403

noncomputable def problem_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)

theorem range_of_values_for_a (a : ℝ) :
  problem_statement a → a ≤ 5 :=
  sorry

end NUMINAMATH_GPT_range_of_values_for_a_l1094_109403


namespace NUMINAMATH_GPT_line_equation_l1094_109433

theorem line_equation {x y : ℝ} (m b : ℝ) (h1 : m = 2) (h2 : b = -3) :
    (∃ (f : ℝ → ℝ), (∀ x, f x = m * x + b) ∧ (∀ x, 2 * x - f x - 3 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l1094_109433


namespace NUMINAMATH_GPT_walk_to_school_l1094_109442

theorem walk_to_school (W P : ℕ) (h1 : W + P = 41) (h2 : W = P + 3) : W = 22 :=
by 
  sorry

end NUMINAMATH_GPT_walk_to_school_l1094_109442


namespace NUMINAMATH_GPT_total_orchids_l1094_109401

-- Conditions
def current_orchids : ℕ := 2
def additional_orchids : ℕ := 4

-- Proof statement
theorem total_orchids : current_orchids + additional_orchids = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_orchids_l1094_109401


namespace NUMINAMATH_GPT_cuboid_volume_is_correct_l1094_109451

-- Definition of cuboid edges and volume calculation
def cuboid_volume (a b c : ℕ) : ℕ := a * b * c

-- Given conditions
def edge1 : ℕ := 2
def edge2 : ℕ := 5
def edge3 : ℕ := 3

-- Theorem statement
theorem cuboid_volume_is_correct : cuboid_volume edge1 edge2 edge3 = 30 := 
by sorry

end NUMINAMATH_GPT_cuboid_volume_is_correct_l1094_109451


namespace NUMINAMATH_GPT_find_right_triangle_sides_l1094_109464

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_condition (a b c : ℕ) : Prop :=
  a * b = 3 * (a + b + c)

theorem find_right_triangle_sides :
  ∃ (a b c : ℕ),
    is_right_triangle a b c ∧ area_condition a b c ∧
    ((a = 7 ∧ b = 24 ∧ c = 25) ∨
     (a = 8 ∧ b = 15 ∧ c = 17) ∨
     (a = 9 ∧ b = 12 ∧ c = 15)) :=
sorry

end NUMINAMATH_GPT_find_right_triangle_sides_l1094_109464


namespace NUMINAMATH_GPT_smallest_x_l1094_109426

theorem smallest_x (x : ℕ) :
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 279 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_l1094_109426


namespace NUMINAMATH_GPT_mod_inverse_11_mod_1105_l1094_109406

theorem mod_inverse_11_mod_1105 : (11 * 201) % 1105 = 1 :=
  by 
    sorry

end NUMINAMATH_GPT_mod_inverse_11_mod_1105_l1094_109406


namespace NUMINAMATH_GPT_total_stars_l1094_109494

/-- Let n be the number of students, and s be the number of stars each student makes.
    We need to prove that the total number of stars is n * s. --/
theorem total_stars (n : ℕ) (s : ℕ) (h_n : n = 186) (h_s : s = 5) : n * s = 930 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_stars_l1094_109494


namespace NUMINAMATH_GPT_calculation_result_l1094_109443

theorem calculation_result :
  5 * 7 - 6 * 8 + 9 * 2 + 7 * 3 = 26 :=
by sorry

end NUMINAMATH_GPT_calculation_result_l1094_109443


namespace NUMINAMATH_GPT_combined_length_of_trains_l1094_109474

theorem combined_length_of_trains
  (speed_A_kmph : ℕ) (speed_B_kmph : ℕ)
  (platform_length : ℕ) (time_A_sec : ℕ) (time_B_sec : ℕ)
  (h_speed_A : speed_A_kmph = 72) (h_speed_B : speed_B_kmph = 90)
  (h_platform_length : platform_length = 300)
  (h_time_A : time_A_sec = 30) (h_time_B : time_B_sec = 24) :
  let speed_A_ms := speed_A_kmph * 5 / 18
  let speed_B_ms := speed_B_kmph * 5 / 18
  let distance_A := speed_A_ms * time_A_sec
  let distance_B := speed_B_ms * time_B_sec
  let length_A := distance_A - platform_length
  let length_B := distance_B - platform_length
  length_A + length_B = 600 :=
by
  sorry

end NUMINAMATH_GPT_combined_length_of_trains_l1094_109474


namespace NUMINAMATH_GPT_initial_sum_simple_interest_l1094_109492

theorem initial_sum_simple_interest :
  ∃ P : ℝ, (P * (3/100) + P * (5/100) + P * (4/100) + P * (6/100) = 100) ∧ (P = 5000 / 9) :=
by
  sorry

end NUMINAMATH_GPT_initial_sum_simple_interest_l1094_109492


namespace NUMINAMATH_GPT_neg_a_pow4_div_neg_a_eq_neg_a_pow3_l1094_109499

variable (a : ℝ)

theorem neg_a_pow4_div_neg_a_eq_neg_a_pow3 : (-a)^4 / (-a) = -a^3 := sorry

end NUMINAMATH_GPT_neg_a_pow4_div_neg_a_eq_neg_a_pow3_l1094_109499


namespace NUMINAMATH_GPT_temp_drop_of_8_deg_is_neg_8_l1094_109444

theorem temp_drop_of_8_deg_is_neg_8 (rise_3_deg : ℤ) (h : rise_3_deg = 3) : ∀ drop_8_deg, drop_8_deg = -8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_temp_drop_of_8_deg_is_neg_8_l1094_109444


namespace NUMINAMATH_GPT_trains_cross_time_l1094_109428

def speed_in_m_per_s (speed_in_km_per_hr : Float) : Float :=
  (speed_in_km_per_hr * 1000) / 3600

def relative_speed (speed1 : Float) (speed2 : Float) : Float :=
  speed1 + speed2

def total_distance (length1 : Float) (length2 : Float) : Float :=
  length1 + length2

def time_to_cross (total_dist : Float) (relative_spd : Float) : Float :=
  total_dist / relative_spd

theorem trains_cross_time 
  (length_train1 : Float := 270)
  (speed_train1 : Float := 120)
  (length_train2 : Float := 230.04)
  (speed_train2 : Float := 80) :
  time_to_cross (total_distance length_train1 length_train2) 
                (relative_speed (speed_in_m_per_s speed_train1) 
                                (speed_in_m_per_s speed_train2)) = 9 := 
by
  sorry

end NUMINAMATH_GPT_trains_cross_time_l1094_109428


namespace NUMINAMATH_GPT_percentage_of_women_lawyers_l1094_109483

theorem percentage_of_women_lawyers
  (T : ℝ) 
  (h1 : 0.70 * T = W) 
  (h2 : 0.28 * T = WL) : 
  ((WL / W) * 100 = 40) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_women_lawyers_l1094_109483


namespace NUMINAMATH_GPT_max_sum_of_integer_pairs_l1094_109435

theorem max_sum_of_integer_pairs (x y : ℤ) (h : (x-1)^2 + (y+2)^2 = 36) : 
  max (x + y) = 5 :=
sorry

end NUMINAMATH_GPT_max_sum_of_integer_pairs_l1094_109435


namespace NUMINAMATH_GPT_solve_inequality_l1094_109415

-- Define conditions
def valid_x (x : ℝ) : Prop := x ≠ -3 ∧ x ≠ -8/3

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x + 3) > (4 * x + 5) / (3 * x + 8)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (-3 < x ∧ x < -8/3) ∨ ((1 - Real.sqrt 89) / 4 < x ∧ x < (1 + Real.sqrt 89) / 4)

-- Prove the equivalence
theorem solve_inequality (x : ℝ) (h : valid_x x) : inequality x ↔ solution_set x :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1094_109415


namespace NUMINAMATH_GPT_triangle_third_side_length_l1094_109432

theorem triangle_third_side_length {x : ℝ}
    (h1 : 3 > 0)
    (h2 : 7 > 0)
    (h3 : 3 + 7 > x)
    (h4 : x + 3 > 7)
    (h5 : x + 7 > 3) :
    4 < x ∧ x < 10 := by
  sorry

end NUMINAMATH_GPT_triangle_third_side_length_l1094_109432


namespace NUMINAMATH_GPT_range_of_a_l1094_109455

open Set

variable {a : ℝ}
def M (a : ℝ) : Set ℝ := { x : ℝ | (2 * a - 1) < x ∧ x < (4 * a) }
def N : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

theorem range_of_a (h : N ⊆ M a) : 1 / 2 ≤ a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_a_l1094_109455


namespace NUMINAMATH_GPT_trader_profit_l1094_109452

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def discount_price (P : ℝ) : ℝ := 0.95 * P
noncomputable def selling_price (P : ℝ) : ℝ := 1.52 * P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def percent_profit (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) (hP : 0 < P) : percent_profit P = 52 := by 
  sorry

end NUMINAMATH_GPT_trader_profit_l1094_109452


namespace NUMINAMATH_GPT_james_sushi_rolls_l1094_109420

def fish_for_sushi : ℕ := 40
def total_fish : ℕ := 400
def bad_fish_percentage : ℕ := 20

theorem james_sushi_rolls :
  let good_fish := total_fish - (bad_fish_percentage * total_fish / 100)
  good_fish / fish_for_sushi = 8 :=
by
  sorry

end NUMINAMATH_GPT_james_sushi_rolls_l1094_109420


namespace NUMINAMATH_GPT_intersection_of_sets_l1094_109421

def setA : Set ℝ := {x | x^2 < 8}
def setB : Set ℝ := {x | 1 - x ≤ 0}
def setIntersection : Set ℝ := {x | x ∈ setA ∧ x ∈ setB}

theorem intersection_of_sets :
    setIntersection = {x | 1 ≤ x ∧ x < 2 * Real.sqrt 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1094_109421


namespace NUMINAMATH_GPT_general_formula_a_n_general_formula_b_n_l1094_109487

-- Prove general formula for the sequence a_n
theorem general_formula_a_n (S : Nat → Nat) (a : Nat → Nat) (h₁ : ∀ n, S n = 2^(n+1) - 2) :
  (∀ n, a n = S n - S (n - 1)) → ∀ n, a n = 2^n :=
by
  sorry

-- Prove general formula for the sequence b_n
theorem general_formula_b_n (a b : Nat → Nat) (h₁ : ∀ n, a n = 2^n) :
  (∀ n, b n = a n + a (n + 1)) → ∀ n, b n = 3 * 2^n :=
by
  sorry

end NUMINAMATH_GPT_general_formula_a_n_general_formula_b_n_l1094_109487


namespace NUMINAMATH_GPT_sum_first_110_terms_l1094_109469

noncomputable def sum_arithmetic (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem sum_first_110_terms (a1 d : ℚ) (h1 : sum_arithmetic a1 d 10 = 100)
  (h2 : sum_arithmetic a1 d 100 = 10) : sum_arithmetic a1 d 110 = -110 := by
  sorry

end NUMINAMATH_GPT_sum_first_110_terms_l1094_109469


namespace NUMINAMATH_GPT_num_positive_integers_condition_l1094_109485

theorem num_positive_integers_condition : 
  ∃! n : ℤ, 0 < n ∧ n < 50 ∧ (n + 2) % (50 - n) = 0 :=
by
  sorry

end NUMINAMATH_GPT_num_positive_integers_condition_l1094_109485


namespace NUMINAMATH_GPT_sum_of_solutions_l1094_109465

def equation (x : ℝ) : Prop := (6 * x) / 30 = 8 / x

theorem sum_of_solutions : ∀ x1 x2 : ℝ, equation x1 → equation x2 → x1 + x2 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1094_109465


namespace NUMINAMATH_GPT_tilling_time_in_minutes_l1094_109489

-- Definitions
def plot_width : ℕ := 110
def plot_length : ℕ := 120
def tiller_width : ℕ := 2
def tilling_rate : ℕ := 2 -- 2 seconds per foot

-- Theorem: The time to till the entire plot in minutes
theorem tilling_time_in_minutes : (plot_width / tiller_width * plot_length * tilling_rate) / 60 = 220 := by
  sorry

end NUMINAMATH_GPT_tilling_time_in_minutes_l1094_109489


namespace NUMINAMATH_GPT_seq_a2020_l1094_109473

def seq (a : ℕ → ℕ) : Prop :=
(∀ n : ℕ, (a n + a (n+1) ≠ a (n+2) + a (n+3))) ∧
(∀ n : ℕ, (a n + a (n+1) + a (n+2) ≠ a (n+3) + a (n+4) + a (n+5))) ∧
(a 1 = 0)

theorem seq_a2020 (a : ℕ → ℕ) (h : seq a) : a 2020 = 1 :=
sorry

end NUMINAMATH_GPT_seq_a2020_l1094_109473


namespace NUMINAMATH_GPT_grant_school_students_l1094_109402

theorem grant_school_students (S : ℕ) 
  (h1 : S / 3 = x) 
  (h2 : x / 4 = 15) : 
  S = 180 := 
sorry

end NUMINAMATH_GPT_grant_school_students_l1094_109402


namespace NUMINAMATH_GPT_adam_change_is_correct_l1094_109471

-- Define the conditions
def adam_money : ℝ := 5.00
def airplane_cost : ℝ := 4.28
def change : ℝ := adam_money - airplane_cost

-- State the theorem
theorem adam_change_is_correct : change = 0.72 := 
by {
  -- Proof can be added later
  sorry
}

end NUMINAMATH_GPT_adam_change_is_correct_l1094_109471


namespace NUMINAMATH_GPT_minimum_value_of_f_on_interval_l1094_109467

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * x^2 + Real.log x

theorem minimum_value_of_f_on_interval :
  (∀ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x ≥ f (Real.exp 1)) ∧
  ∃ x ∈ (Set.Icc (1 / Real.exp 1) (Real.exp 1)), f x = f (Real.exp 1) := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_on_interval_l1094_109467


namespace NUMINAMATH_GPT_average_weight_a_b_l1094_109448

theorem average_weight_a_b (A B C : ℝ) 
    (h1 : (A + B + C) / 3 = 45) 
    (h2 : (B + C) / 2 = 44) 
    (h3 : B = 33) : 
    (A + B) / 2 = 40 := 
by 
  sorry

end NUMINAMATH_GPT_average_weight_a_b_l1094_109448


namespace NUMINAMATH_GPT_weeks_to_cover_expense_l1094_109495

-- Definitions and the statement of the problem
def hourly_rate : ℕ := 20
def monthly_expense : ℕ := 1200
def weekday_hours : ℕ := 3
def saturday_hours : ℕ := 5

theorem weeks_to_cover_expense : 
  ∀ (w : ℕ), (5 * weekday_hours + saturday_hours) * hourly_rate * w ≥ monthly_expense → w >= 3 := 
sorry

end NUMINAMATH_GPT_weeks_to_cover_expense_l1094_109495


namespace NUMINAMATH_GPT_sequence_property_l1094_109418

noncomputable def U : ℕ → ℕ
| 0       => 0  -- This definition is added to ensure U 1 corresponds to U_1 = 1
| (n + 1) => U n + (n + 1)

theorem sequence_property (n : ℕ) : U n + U (n + 1) = (n + 1) * (n + 1) :=
  sorry

end NUMINAMATH_GPT_sequence_property_l1094_109418


namespace NUMINAMATH_GPT_movie_theater_people_l1094_109490

def totalSeats : ℕ := 750
def emptySeats : ℕ := 218
def peopleWatching := totalSeats - emptySeats

theorem movie_theater_people :
  peopleWatching = 532 := by
  sorry

end NUMINAMATH_GPT_movie_theater_people_l1094_109490


namespace NUMINAMATH_GPT_interest_rate_b_to_c_l1094_109472

open Real

noncomputable def calculate_rate_b_to_c (P : ℝ) (r1 : ℝ) (t : ℝ) (G : ℝ) : ℝ :=
  let I_a_b := P * (r1 / 100) * t
  let I_b_c := I_a_b + G
  (100 * I_b_c) / (P * t)

theorem interest_rate_b_to_c :
  calculate_rate_b_to_c 3200 12 5 400 = 14.5 := by
  sorry

end NUMINAMATH_GPT_interest_rate_b_to_c_l1094_109472


namespace NUMINAMATH_GPT_correct_minutes_added_l1094_109459

theorem correct_minutes_added :
  let time_lost_per_day : ℚ := 3 + 1/4
  let start_time := 1 -- in P.M. on March 15
  let end_time := 3 -- in P.M. on March 22
  let total_days := 7 -- days from March 15 to March 22
  let extra_hours := 2 -- hours on March 22 from 1 P.M. to 3 P.M.
  let total_hours := (total_days * 24) + extra_hours
  let time_lost_per_minute := time_lost_per_day / (24 * 60)
  let total_time_lost := total_hours * time_lost_per_minute
  let total_time_lost_minutes := total_time_lost * 60
  n = total_time_lost_minutes 
→ n = 221 / 96 := 
sorry

end NUMINAMATH_GPT_correct_minutes_added_l1094_109459


namespace NUMINAMATH_GPT_stone_breadth_l1094_109427

theorem stone_breadth 
  (hall_length_m : ℕ) (hall_breadth_m : ℕ)
  (stone_length_dm : ℕ) (num_stones : ℕ)
  (hall_area_dm2 : ℕ) (stone_area_dm2 : ℕ) 
  (hall_length_dm hall_breadth_dm : ℕ) (b : ℕ) :
  hall_length_m = 36 → hall_breadth_m = 15 →
  stone_length_dm = 8 → num_stones = 1350 →
  hall_length_dm = hall_length_m * 10 → hall_breadth_dm = hall_breadth_m * 10 →
  hall_area_dm2 = hall_length_dm * hall_breadth_dm →
  stone_area_dm2 = stone_length_dm * b →
  hall_area_dm2 = num_stones * stone_area_dm2 →
  b = 5 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_stone_breadth_l1094_109427


namespace NUMINAMATH_GPT_rancher_lasso_probability_l1094_109446

theorem rancher_lasso_probability : 
  let p_success := 1 / 2
  let p_failure := 1 - p_success
  (1 - p_failure ^ 3) = (7 / 8) := by
  sorry

end NUMINAMATH_GPT_rancher_lasso_probability_l1094_109446


namespace NUMINAMATH_GPT_discount_limit_l1094_109400

theorem discount_limit {cost_price selling_price : ℕ} (x : ℚ)
  (h1: cost_price = 100)
  (h2: selling_price = 150)
  (h3: ∃ p : ℚ, p = 1.2 * cost_price) : selling_price * (x / 10) - cost_price ≥ 0.2 * cost_price ↔ x ≤ 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_discount_limit_l1094_109400


namespace NUMINAMATH_GPT_greatest_three_digit_number_l1094_109479

theorem greatest_three_digit_number : ∃ n : ℕ, n < 1000 ∧ n >= 100 ∧ (n + 1) % 8 = 0 ∧ (n - 4) % 7 = 0 ∧ n = 967 :=
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_number_l1094_109479


namespace NUMINAMATH_GPT_fruit_basket_count_l1094_109441

-- Define the number of apples and oranges
def apples := 7
def oranges := 12

-- Condition: A fruit basket must contain at least two pieces of fruit
def min_pieces_of_fruit := 2

-- Problem: Prove that there are 101 different fruit baskets containing at least two pieces of fruit
theorem fruit_basket_count (n_apples n_oranges n_min_pieces : Nat) (h_apples : n_apples = apples) (h_oranges : n_oranges = oranges) (h_min_pieces : n_min_pieces = min_pieces_of_fruit) :
  (n_apples = 7) ∧ (n_oranges = 12) ∧ (n_min_pieces = 2) → (104 - 3 = 101) :=
by
  sorry

end NUMINAMATH_GPT_fruit_basket_count_l1094_109441


namespace NUMINAMATH_GPT_correct_group_l1094_109414

def atomic_number (element : String) : Nat :=
  match element with
  | "Be" => 4
  | "C" => 6
  | "B" => 5
  | "Cl" => 17
  | "O" => 8
  | "Li" => 3
  | "Al" => 13
  | "S" => 16
  | "Si" => 14
  | "Mg" => 12
  | _ => 0

def is_descending (lst : List Nat) : Bool :=
  match lst with
  | [] => true
  | [x] => true
  | x :: y :: xs => if x > y then is_descending (y :: xs) else false

theorem correct_group : is_descending [atomic_number "Cl", atomic_number "O", atomic_number "Li"] = true ∧
                        is_descending [atomic_number "Be", atomic_number "C", atomic_number "B"] = false ∧
                        is_descending [atomic_number "Al", atomic_number "S", atomic_number "Si"] = false ∧
                        is_descending [atomic_number "C", atomic_number "S", atomic_number "Mg"] = false :=
by
  -- Prove the given theorem based on the atomic number function and is_descending condition
  sorry

end NUMINAMATH_GPT_correct_group_l1094_109414


namespace NUMINAMATH_GPT_inequality_abc_l1094_109486

variable (a b c : ℝ)

theorem inequality_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  a / (a^3 - a^2 + 3) + b / (b^3 - b^2 + 3) + c / (c^3 - c^2 + 3) ≤ 1 := 
sorry

end NUMINAMATH_GPT_inequality_abc_l1094_109486


namespace NUMINAMATH_GPT_split_terms_addition_l1094_109445

theorem split_terms_addition : 
  (-2017 - (2/3)) + (2016 + (3/4)) + (-2015 - (5/6)) + (16 + (1/2)) = -2000 - (1/4) :=
by
  sorry

end NUMINAMATH_GPT_split_terms_addition_l1094_109445


namespace NUMINAMATH_GPT_bill_profit_difference_l1094_109430

theorem bill_profit_difference (P SP NSP NP : ℝ) 
  (h1 : SP = 1.10 * P)
  (h2 : SP = 659.9999999999994)
  (h3 : NP = 0.90 * P)
  (h4 : NSP = 1.30 * NP) :
  NSP - SP = 42 := 
sorry

end NUMINAMATH_GPT_bill_profit_difference_l1094_109430


namespace NUMINAMATH_GPT_value_of_z_l1094_109498

theorem value_of_z (z y : ℝ) (h1 : (12)^3 * z^3 / 432 = y) (h2 : y = 864) : z = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_z_l1094_109498


namespace NUMINAMATH_GPT_sufficient_condition_l1094_109411

theorem sufficient_condition (A B C D : Prop) (h : C → D): C → (A > B) := 
by 
  sorry

end NUMINAMATH_GPT_sufficient_condition_l1094_109411


namespace NUMINAMATH_GPT_modified_cube_edges_l1094_109413

/--
A solid cube with a side length of 4 has different-sized solid cubes removed from three of its corners:
- one corner loses a cube of side length 1,
- another corner loses a cube of side length 2,
- and a third corner loses a cube of side length 1.

The total number of edges of the modified solid is 22.
-/
theorem modified_cube_edges :
  let original_edges := 12
  let edges_removed_1x1 := 6
  let edges_added_2x2 := 16
  original_edges - 2 * edges_removed_1x1 + edges_added_2x2 = 22 := by
  sorry

end NUMINAMATH_GPT_modified_cube_edges_l1094_109413


namespace NUMINAMATH_GPT_sum_of_roots_l1094_109453

theorem sum_of_roots (x₁ x₂ : ℝ) (h1 : x₁^2 = 2 * x₁ + 1) (h2 : x₂^2 = 2 * x₂ + 1) :
  x₁ + x₂ = 2 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1094_109453


namespace NUMINAMATH_GPT_neg_root_sufficient_not_necessary_l1094_109456

theorem neg_root_sufficient_not_necessary (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (a < 0) :=
sorry

end NUMINAMATH_GPT_neg_root_sufficient_not_necessary_l1094_109456


namespace NUMINAMATH_GPT_factorize_expression_l1094_109477

theorem factorize_expression (y a : ℝ) : 
  3 * y * a ^ 2 - 6 * y * a + 3 * y = 3 * y * (a - 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1094_109477


namespace NUMINAMATH_GPT_current_books_l1094_109405

def initial_books : ℕ := 743
def sold_instore_saturday : ℕ := 37
def sold_online_saturday : ℕ := 128
def sold_instore_sunday : ℕ := 2 * sold_instore_saturday
def sold_online_sunday : ℕ := sold_online_saturday + 34
def total_books_sold_saturday : ℕ := sold_instore_saturday + sold_online_saturday
def total_books_sold_sunday : ℕ := sold_instore_sunday + sold_online_sunday
def total_books_sold_weekend : ℕ := total_books_sold_saturday + total_books_sold_sunday
def books_received_shipment : ℕ := 160
def net_change_books : ℤ := books_received_shipment - total_books_sold_weekend

theorem current_books
  (initial_books : ℕ) 
  (sold_instore_saturday : ℕ) 
  (sold_online_saturday : ℕ) 
  (sold_instore_sunday : ℕ)
  (sold_online_sunday : ℕ)
  (total_books_sold_saturday : ℕ)
  (total_books_sold_sunday : ℕ)
  (total_books_sold_weekend : ℕ)
  (books_received_shipment : ℕ)
  (net_change_books : ℤ) : (initial_books - net_change_books) = 502 := 
by {
  sorry
}

end NUMINAMATH_GPT_current_books_l1094_109405


namespace NUMINAMATH_GPT_calculate_v_sum_l1094_109481

def v (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem calculate_v_sum :
  v (2) + v (-2) + v (1) + v (-1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_v_sum_l1094_109481


namespace NUMINAMATH_GPT_f_monotonically_decreasing_in_interval_l1094_109422

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 3

theorem f_monotonically_decreasing_in_interval :
  ∀ x y : ℝ, -2 < x ∧ x < 1 → -2 < y ∧ y < 1 → (y > x → f y < f x) :=
by
  sorry

end NUMINAMATH_GPT_f_monotonically_decreasing_in_interval_l1094_109422


namespace NUMINAMATH_GPT_total_oranges_after_increase_l1094_109423

theorem total_oranges_after_increase :
  let Mary := 122
  let Jason := 105
  let Tom := 85
  let Sarah := 134
  let increase_rate := 0.10
  let new_Mary := Mary + Mary * increase_rate
  let new_Jason := Jason + Jason * increase_rate
  let new_Tom := Tom + Tom * increase_rate
  let new_Sarah := Sarah + Sarah * increase_rate
  let total_new_oranges := new_Mary + new_Jason + new_Tom + new_Sarah
  Float.round total_new_oranges = 491 := 
by
  sorry

end NUMINAMATH_GPT_total_oranges_after_increase_l1094_109423


namespace NUMINAMATH_GPT_payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l1094_109429

variable (x : ℕ)
variable (hx : x > 10)

noncomputable def option1_payment (x : ℕ) : ℕ := 200 * x + 8000
noncomputable def option2_payment (x : ℕ) : ℕ := 180 * x + 9000

theorem payment_option1 (x : ℕ) (hx : x > 10) : option1_payment x = 200 * x + 8000 :=
by sorry

theorem payment_option2 (x : ℕ) (hx : x > 10) : option2_payment x = 180 * x + 9000 :=
by sorry

theorem cost_effective_option (x : ℕ) (hx : x > 10) (h30 : x = 30) : option1_payment 30 < option2_payment 30 :=
by sorry

theorem most_cost_effective_plan (h30 : x = 30) : (10000 + 3600 = 13600) :=
by sorry

end NUMINAMATH_GPT_payment_option1_payment_option2_cost_effective_option_most_cost_effective_plan_l1094_109429


namespace NUMINAMATH_GPT_remaining_stock_weighs_120_l1094_109407

noncomputable def total_remaining_weight (green_beans_weight rice_weight sugar_weight : ℕ) :=
  let remaining_rice := rice_weight - (rice_weight / 3)
  let remaining_sugar := sugar_weight - (sugar_weight / 5)
  let remaining_stock := remaining_rice + remaining_sugar + green_beans_weight
  remaining_stock

theorem remaining_stock_weighs_120 : total_remaining_weight 60 30 50 = 120 :=
by
  have h1: 60 - 30 = 30 := by norm_num
  have h2: 60 - 10 = 50 := by norm_num
  have h3: 30 - (30 / 3) = 20 := by norm_num
  have h4: 50 - (50 / 5) = 40 := by norm_num
  have h5: 20 + 40 + 60 = 120 := by norm_num
  exact h5

end NUMINAMATH_GPT_remaining_stock_weighs_120_l1094_109407


namespace NUMINAMATH_GPT_proof_BH_length_equals_lhs_rhs_l1094_109497

noncomputable def calculate_BH_length : ℝ :=
  let AB := 3
  let BC := 4
  let CA := 5
  let AG := 4  -- Since AB < AG
  let AH := 6  -- AG < AH
  let GI := 3
  let HI := 8
  let GH := Real.sqrt (GI ^ 2 + HI ^ 2)
  let p := 3
  let q := 2
  let r := 73
  let s := 1
  3 + 2 * Real.sqrt 73

theorem proof_BH_length_equals_lhs_rhs :
  let BH := 3 + 2 * Real.sqrt 73
  calculate_BH_length = BH := by
    sorry

end NUMINAMATH_GPT_proof_BH_length_equals_lhs_rhs_l1094_109497


namespace NUMINAMATH_GPT_binom_2000_3_eq_l1094_109431

theorem binom_2000_3_eq : Nat.choose 2000 3 = 1331000333 := by
  sorry

end NUMINAMATH_GPT_binom_2000_3_eq_l1094_109431


namespace NUMINAMATH_GPT_total_length_of_rubber_pen_pencil_l1094_109457

variable (rubber pen pencil : ℕ)

theorem total_length_of_rubber_pen_pencil 
  (h1 : pen = rubber + 3)
  (h2 : pen = pencil - 2)
  (h3 : pencil = 12) : rubber + pen + pencil = 29 := by
  sorry

end NUMINAMATH_GPT_total_length_of_rubber_pen_pencil_l1094_109457


namespace NUMINAMATH_GPT_least_number_to_add_l1094_109412

theorem least_number_to_add (x : ℕ) (h : 1055 % 23 = 20) : x = 3 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_least_number_to_add_l1094_109412


namespace NUMINAMATH_GPT_average_speed_l1094_109463

theorem average_speed (D : ℝ) :
  let time_by_bus := D / 80
  let time_walking := D / 16
  let time_cycling := D / 120
  let total_time := time_by_bus + time_walking + time_cycling
  let total_distance := 2 * D
  total_distance / total_time = 24 := by
  sorry

end NUMINAMATH_GPT_average_speed_l1094_109463


namespace NUMINAMATH_GPT_breakfast_calories_l1094_109425

variable (B : ℝ) 

def lunch_calories := 1.25 * B
def dinner_calories := 2.5 * B
def shakes_calories := 900
def total_calories := 3275

theorem breakfast_calories:
  (B + lunch_calories B + dinner_calories B + shakes_calories = total_calories) → B = 500 :=
by
  sorry

end NUMINAMATH_GPT_breakfast_calories_l1094_109425


namespace NUMINAMATH_GPT_hot_dog_cost_l1094_109437

variables (h d : ℝ)

theorem hot_dog_cost :
  (3 * h + 4 * d = 10) →
  (2 * h + 3 * d = 7) →
  d = 1 :=
by
  intros h_eq d_eq
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_hot_dog_cost_l1094_109437


namespace NUMINAMATH_GPT_distributeCandies_l1094_109484

-- Define the conditions as separate definitions.

-- Number of candies
def candies : ℕ := 10

-- Number of boxes
def boxes : ℕ := 5

-- Condition that each box gets at least one candy
def atLeastOne (candyDist : Fin boxes → ℕ) : Prop :=
  ∀ b, candyDist b > 0

-- Function to count the number of ways to distribute candies
noncomputable def countWaysToDistribute (candies : ℕ) (boxes : ℕ) : ℕ :=
  -- Function to compute the number of ways
  -- (assuming a correct implementation is provided)
  sorry -- Placeholder for the actual counting implementation

-- Theorem to prove the number of distributions
theorem distributeCandies : countWaysToDistribute candies boxes = 7 := 
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_distributeCandies_l1094_109484


namespace NUMINAMATH_GPT_radius_of_circular_film_l1094_109410

theorem radius_of_circular_film (r_canister h_canister t_film R: ℝ) 
  (V: ℝ) (h1: r_canister = 5) (h2: h_canister = 10) 
  (h3: t_film = 0.2) (h4: V = 250 * Real.pi): R = 25 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circular_film_l1094_109410


namespace NUMINAMATH_GPT_solve_equation_l1094_109417

theorem solve_equation (m n : ℝ) (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : m ≠ n) :
  ∀ x : ℝ, ((x + m)^2 - 3 * (x + n)^2 = m^2 - 3 * n^2) ↔ (x = 0 ∨ x = m - 3 * n) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1094_109417


namespace NUMINAMATH_GPT_find_multiplier_n_l1094_109439

variable (x y n : ℝ)

theorem find_multiplier_n (h1 : 5 * x = n * y) 
  (h2 : x * y ≠ 0) 
  (h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998) : 
  n = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_multiplier_n_l1094_109439


namespace NUMINAMATH_GPT_crayons_count_l1094_109408

-- Definitions based on the conditions given in the problem
def total_crayons : Nat := 96
def benny_crayons : Nat := 12
def fred_crayons : Nat := 2 * benny_crayons
def jason_crayons (sarah_crayons : Nat) : Nat := 3 * sarah_crayons

-- Stating the proof goal
theorem crayons_count (sarah_crayons : Nat) :
  fred_crayons + benny_crayons + jason_crayons sarah_crayons + sarah_crayons = total_crayons →
  sarah_crayons = 15 ∧
  fred_crayons = 24 ∧
  jason_crayons sarah_crayons = 45 ∧
  benny_crayons = 12 :=
by
  sorry

end NUMINAMATH_GPT_crayons_count_l1094_109408


namespace NUMINAMATH_GPT_not_enough_funds_to_buy_two_books_l1094_109449

def storybook_cost : ℝ := 25.5
def sufficient_funds (amount : ℝ) : Prop := amount >= 50

theorem not_enough_funds_to_buy_two_books : ¬ sufficient_funds (2 * storybook_cost) :=
by
  sorry

end NUMINAMATH_GPT_not_enough_funds_to_buy_two_books_l1094_109449


namespace NUMINAMATH_GPT_sasha_lives_on_seventh_floor_l1094_109419

theorem sasha_lives_on_seventh_floor (N : ℕ) (x : ℕ) 
(h1 : x = (1/3 : ℝ) * N) 
(h2 : N - ((1/3 : ℝ) * N + 1) = (1/2 : ℝ) * N) :
  N + 1 = 7 := 
sorry

end NUMINAMATH_GPT_sasha_lives_on_seventh_floor_l1094_109419


namespace NUMINAMATH_GPT_price_of_olives_l1094_109470

theorem price_of_olives 
  (cherries_price : ℝ)
  (total_cost_with_discount : ℝ)
  (num_bags : ℕ)
  (discount : ℝ)
  (olives_price : ℝ) :
  cherries_price = 5 →
  total_cost_with_discount = 540 →
  num_bags = 50 →
  discount = 0.10 →
  (0.9 * (num_bags * cherries_price + num_bags * olives_price) = total_cost_with_discount) →
  olives_price = 7 :=
by
  intros h_cherries_price h_total_cost h_num_bags h_discount h_equation
  sorry

end NUMINAMATH_GPT_price_of_olives_l1094_109470


namespace NUMINAMATH_GPT_salary_for_may_l1094_109488

theorem salary_for_may (J F M A May : ℝ) 
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8200)
  (h3 : J = 5700) : 
  May = 6500 :=
by 
  have eq1 : J + F + M + A = 32000 := by
    linarith
  have eq2 : F + M + A + May = 32800 := by
    linarith
  have eq3 : May - J = 800 := by
    linarith [eq1, eq2]
  have eq4 : May = 6500 := by
    linarith [eq3, h3]
  exact eq4

end NUMINAMATH_GPT_salary_for_may_l1094_109488


namespace NUMINAMATH_GPT_find_m_n_l1094_109461

theorem find_m_n (m n : ℕ) (h : (1/5 : ℝ)^m * (1/4 : ℝ)^n = 1 / (10 : ℝ)^4) : m = 4 ∧ n = 2 :=
sorry

end NUMINAMATH_GPT_find_m_n_l1094_109461


namespace NUMINAMATH_GPT_smallest_number_four_solutions_sum_four_squares_l1094_109454

def is_sum_of_four_squares (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a^2 + b^2 + c^2 + d^2

theorem smallest_number_four_solutions_sum_four_squares :
  ∃ n : ℕ,
    is_sum_of_four_squares n ∧
    (∃ (a1 b1 c1 d1 a2 b2 c2 d2 a3 b3 c3 d3 a4 b4 c4 d4 : ℕ),
      n = a1^2 + b1^2 + c1^2 + d1^2 ∧
      n = a2^2 + b2^2 + c2^2 + d2^2 ∧
      n = a3^2 + b3^2 + c3^2 + d3^2 ∧
      n = a4^2 + b4^2 + c4^2 + d4^2 ∧
      (a1, b1, c1, d1) ≠ (a2, b2, c2, d2) ∧
      (a1, b1, c1, d1) ≠ (a3, b3, c3, d3) ∧
      (a1, b1, c1, d1) ≠ (a4, b4, c4, d4) ∧
      (a2, b2, c2, d2) ≠ (a3, b3, c3, d3) ∧
      (a2, b2, c2, d2) ≠ (a4, b4, c4, d4) ∧
      (a3, b3, c3, d3) ≠ (a4, b4, c4, d4)) ∧
    (∀ m : ℕ,
      m < 635318657 →
      ¬ (∃ (a5 b5 c5 d5 a6 b6 c6 d6 a7 b7 c7 d7 a8 b8 c8 d8 : ℕ),
        m = a5^2 + b5^2 + c5^2 + d5^2 ∧
        m = a6^2 + b6^2 + c6^2 + d6^2 ∧
        m = a7^2 + b7^2 + c7^2 + d7^2 ∧
        m = a8^2 + b8^2 + c8^2 + d8^2 ∧
        (a5, b5, c5, d5) ≠ (a6, b6, c6, d6) ∧
        (a5, b5, c5, d5) ≠ (a7, b7, c7, d7) ∧
        (a5, b5, c5, d5) ≠ (a8, b8, c8, d8) ∧
        (a6, b6, c6, d6) ≠ (a7, b7, c7, d7) ∧
        (a6, b6, c6, d6) ≠ (a8, b8, c8, d8) ∧
        (a7, b7, c7, d7) ≠ (a8, b8, c8, d8))) :=
  sorry

end NUMINAMATH_GPT_smallest_number_four_solutions_sum_four_squares_l1094_109454


namespace NUMINAMATH_GPT_number_of_segments_l1094_109478

theorem number_of_segments (tangent_chords : ℕ) (angle_ABC : ℝ) (h : angle_ABC = 80) :
  tangent_chords = 18 :=
sorry

end NUMINAMATH_GPT_number_of_segments_l1094_109478


namespace NUMINAMATH_GPT_dog_food_l1094_109493

theorem dog_food (weights : List ℕ) (h_weights : weights = [20, 40, 10, 30, 50]) (h_ratio : ∀ w ∈ weights, 1 ≤ w / 10):
  (weights.sum / 10) = 15 := by
  sorry

end NUMINAMATH_GPT_dog_food_l1094_109493


namespace NUMINAMATH_GPT_area_of_AFCH_l1094_109480

-- Define the lengths of the sides of the rectangles
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the problem statement
theorem area_of_AFCH :
  let intersection_area := min BC FG * min EF AB
  let total_area := AB * FG
  let outer_ring_area := total_area - intersection_area
  intersection_area + outer_ring_area / 2 = 52.5 :=
by
  -- Use the values of AB, BC, EF, and FG to compute
  sorry

end NUMINAMATH_GPT_area_of_AFCH_l1094_109480


namespace NUMINAMATH_GPT_units_digit_35_87_plus_93_49_l1094_109466

theorem units_digit_35_87_plus_93_49 : (35^87 + 93^49) % 10 = 8 := by
  sorry

end NUMINAMATH_GPT_units_digit_35_87_plus_93_49_l1094_109466


namespace NUMINAMATH_GPT_anna_least_days_l1094_109404

theorem anna_least_days (borrow : ℝ) (interest_rate : ℝ) (days : ℕ) :
  (borrow = 20) → (interest_rate = 0.10) → borrow + (borrow * interest_rate * days) ≥ 2 * borrow → days ≥ 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_anna_least_days_l1094_109404


namespace NUMINAMATH_GPT_percentage_concentration_acid_l1094_109460

-- Definitions based on the given conditions
def volume_acid : ℝ := 1.6
def total_volume : ℝ := 8.0

-- Lean statement to prove the percentage concentration is 20%
theorem percentage_concentration_acid : (volume_acid / total_volume) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_concentration_acid_l1094_109460


namespace NUMINAMATH_GPT_composition_of_homotheties_l1094_109450

-- Define points A1 and A2 and the coefficients k1 and k2
variables (A1 A2 : ℂ) (k1 k2 : ℂ)

-- Definition of homothety
def homothety (A : ℂ) (k : ℂ) (z : ℂ) : ℂ := k * (z - A) + A

-- Translation vector in case 1
noncomputable def translation_vector (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 = 1 then (1 - k1) * A1 + (k1 - 1) * A2 else 0 

-- Center A in case 2
noncomputable def center (A1 A2 : ℂ) (k1 k2 : ℂ) : ℂ :=
  if k1 * k2 ≠ 1 then (k2 * (1 - k1) * A1 + (1 - k2) * A2) / (k1 * k2 - 1) else 0

-- The final composition of two homotheties
noncomputable def composition (A1 A2 : ℂ) (k1 k2 : ℂ) (z : ℂ) : ℂ :=
  if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
  else homothety (center A1 A2 k1 k2) (k1 * k2) z

-- The theorem to prove
theorem composition_of_homotheties 
  (A1 A2 : ℂ) (k1 k2 : ℂ) : ∀ z : ℂ,
  composition A1 A2 k1 k2 z = if k1 * k2 = 1 then z + translation_vector A1 A2 k1 k2
                              else homothety (center A1 A2 k1 k2) (k1 * k2) z := 
by sorry

end NUMINAMATH_GPT_composition_of_homotheties_l1094_109450


namespace NUMINAMATH_GPT_sum_of_A_and_B_zero_l1094_109482

theorem sum_of_A_and_B_zero
  (A B C : ℝ)
  (h1 : A ≠ B)
  (h2 : C ≠ 0)
  (f g : ℝ → ℝ)
  (h3 : ∀ x, f x = A * x + B + C)
  (h4 : ∀ x, g x = B * x + A - C)
  (h5 : ∀ x, f (g x) - g (f x) = 2 * C) : A + B = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_A_and_B_zero_l1094_109482


namespace NUMINAMATH_GPT_first_train_speed_l1094_109496

-- Definitions
def train_speeds_opposite (v₁ v₂ t : ℝ) : Prop := v₁ * t + v₂ * t = 910

def train_problem_conditions (v₁ v₂ t : ℝ) : Prop :=
  train_speeds_opposite v₁ v₂ t ∧ v₂ = 80 ∧ t = 6.5

-- Theorem
theorem first_train_speed (v : ℝ) (h : train_problem_conditions v 80 6.5) : v = 60 :=
  sorry

end NUMINAMATH_GPT_first_train_speed_l1094_109496


namespace NUMINAMATH_GPT_point_on_transformed_graph_l1094_109475

theorem point_on_transformed_graph 
  (f : ℝ → ℝ)
  (h1 : f 12 = 5)
  (x y : ℝ)
  (h2 : 1.5 * y = (f (3 * x) + 3) / 3)
  (point_x : x = 4)
  (point_y : y = 16 / 9) 
  : x + y = 52 / 9 :=
by
  sorry

end NUMINAMATH_GPT_point_on_transformed_graph_l1094_109475


namespace NUMINAMATH_GPT_find_vector_p_l1094_109476

noncomputable def vector_proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := v.1 * u.1 + v.2 * u.2
  let dot_u := u.1 * u.1 + u.2 * u.2
  let scale := dot_uv / dot_u
  (scale * u.1, scale * u.2)

theorem find_vector_p :
  ∃ p : ℝ × ℝ,
    vector_proj (5, -2) p = p ∧
    vector_proj (2, 6) p = p ∧
    p = (14 / 73, 214 / 73) :=
by
  sorry

end NUMINAMATH_GPT_find_vector_p_l1094_109476


namespace NUMINAMATH_GPT_jason_current_cards_l1094_109468

-- Define Jason's initial number of Pokemon cards
def jason_initial_cards : ℕ := 1342

-- Define the number of Pokemon cards Alyssa bought
def alyssa_bought_cards : ℕ := 536

-- Define the number of Pokemon cards Jason has now
def jason_final_cards (initial_cards bought_cards : ℕ) : ℕ :=
  initial_cards - bought_cards

-- Theorem statement verifying the final number of Pokemon cards Jason has
theorem jason_current_cards : jason_final_cards jason_initial_cards alyssa_bought_cards = 806 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jason_current_cards_l1094_109468


namespace NUMINAMATH_GPT_some_number_value_l1094_109416

theorem some_number_value (x : ℕ) (some_number : ℕ) : x = 5 → ((x / 5) + some_number = 4) → some_number = 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_some_number_value_l1094_109416


namespace NUMINAMATH_GPT_range_of_x_for_odd_monotonic_function_l1094_109438

theorem range_of_x_for_odd_monotonic_function 
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_increasing_on_R : ∀ x y : ℝ, x ≤ y → f x ≤ f y) :
  ∀ x : ℝ, (0 < x) → ( (|f (Real.log x) - f (Real.log (1 / x))| / 2) < f 1 ) → (Real.exp (-1) < x ∧ x < Real.exp 1) := 
by
  sorry

end NUMINAMATH_GPT_range_of_x_for_odd_monotonic_function_l1094_109438


namespace NUMINAMATH_GPT_smallest_n_exists_l1094_109458

theorem smallest_n_exists (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : 8 / 15 < n / (n + k)) (h4 : n / (n + k) < 7 / 13) : 
  n = 15 :=
  sorry

end NUMINAMATH_GPT_smallest_n_exists_l1094_109458


namespace NUMINAMATH_GPT_total_cost_correct_l1094_109491

noncomputable def total_cost : ℝ :=
  let first_path_area := 5 * 100
  let first_path_cost := first_path_area * 2
  let second_path_area := 4 * 80
  let second_path_cost := second_path_area * 1.5
  let diagonal_length := Real.sqrt ((100:ℝ)^2 + (80:ℝ)^2)
  let third_path_area := 6 * diagonal_length
  let third_path_cost := third_path_area * 3
  let circular_path_area := Real.pi * (10:ℝ)^2
  let circular_path_cost := circular_path_area * 4
  first_path_cost + second_path_cost + third_path_cost + circular_path_cost

theorem total_cost_correct : total_cost = 5040.64 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1094_109491


namespace NUMINAMATH_GPT_curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l1094_109447

-- Define the curve G as a set of points (x, y) satisfying the equation x^3 + y^3 - 6xy = 0
def curveG (x y : ℝ) : Prop :=
  x^3 + y^3 - 6 * x * y = 0

-- Prove symmetry of curveG with respect to the line y = x
theorem curveG_symmetric (x y : ℝ) (h : curveG x y) : curveG y x :=
  sorry

-- Prove unique common point with the line x + y - 6 = 0
theorem curveG_unique_common_point : ∃! p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 + p.2 = 6 :=
  sorry

-- Prove curveG has at least one common point with the line x - y + 1 = 0
theorem curveG_common_points_x_y : ∃ p : ℝ × ℝ, curveG p.1 p.2 ∧ p.1 - p.2 + 1 = 0 :=
  sorry

-- Prove the maximum distance from any point on the curveG to the origin is 3√2
theorem curveG_max_distance : ∀ p : ℝ × ℝ, curveG p.1 p.2 → p.1 > 0 → p.2 > 0 → (p.1^2 + p.2^2 ≤ 18) :=
  sorry

end NUMINAMATH_GPT_curveG_symmetric_curveG_unique_common_point_curveG_common_points_x_y_curveG_max_distance_l1094_109447
