import Mathlib

namespace NUMINAMATH_GPT_owen_wins_with_n_bullseyes_l1706_170649

-- Define the parameters and conditions
def initial_score_lead : ℕ := 60
def total_shots : ℕ := 120
def bullseye_points : ℕ := 9
def minimum_points_per_shot : ℕ := 3
def max_points_per_shot : ℕ := 9
def n : ℕ := 111

-- Define the condition for Owen's winning requirement
theorem owen_wins_with_n_bullseyes :
  6 * 111 + 360 > 1020 :=
by
  sorry

end NUMINAMATH_GPT_owen_wins_with_n_bullseyes_l1706_170649


namespace NUMINAMATH_GPT_negation_at_most_three_l1706_170650

theorem negation_at_most_three :
  ¬ (∀ n : ℕ, n ≤ 3) ↔ (∃ n : ℕ, n ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_negation_at_most_three_l1706_170650


namespace NUMINAMATH_GPT_solve_for_x_l1706_170616

theorem solve_for_x (h_perimeter_square : ∀(s : ℝ), 4 * s = 64)
  (h_height_triangle : ∀(h : ℝ), h = 48)
  (h_area_equal : ∀(s h x : ℝ), s * s = 1/2 * h * x) : 
  x = 32 / 3 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1706_170616


namespace NUMINAMATH_GPT_village_population_l1706_170681

theorem village_population (P : ℝ) (h : 0.9 * P = 45000) : P = 50000 :=
by
  sorry

end NUMINAMATH_GPT_village_population_l1706_170681


namespace NUMINAMATH_GPT_neg_pow_eq_pow_four_l1706_170674

variable (a : ℝ)

theorem neg_pow_eq_pow_four (a : ℝ) : (-a)^4 = a^4 :=
sorry

end NUMINAMATH_GPT_neg_pow_eq_pow_four_l1706_170674


namespace NUMINAMATH_GPT_total_money_shared_l1706_170614

theorem total_money_shared (T : ℝ) (h : 0.75 * T = 4500) : T = 6000 :=
by
  sorry

end NUMINAMATH_GPT_total_money_shared_l1706_170614


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1706_170604

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

theorem eccentricity_of_ellipse {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
                                 (h_angle : Real.cos (Real.pi / 6) = b / a) :
    eccentricity a b = (Real.sqrt 6) / 3 := by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1706_170604


namespace NUMINAMATH_GPT_solve_for_m_l1706_170602

noncomputable def has_positive_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (m / (x - 3) - 1 / (3 - x) = 2)

theorem solve_for_m (m : ℝ) : has_positive_root m → m = -1 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l1706_170602


namespace NUMINAMATH_GPT_find_correct_fraction_l1706_170628

theorem find_correct_fraction
  (mistake_frac : ℚ) (n : ℕ) (delta : ℚ)
  (correct_frac : ℚ) (number : ℕ)
  (h1 : mistake_frac = 5 / 6)
  (h2 : number = 288)
  (h3 : mistake_frac * number = correct_frac * number + delta)
  (h4 : delta = 150) :
  correct_frac = 5 / 32 :=
by
  sorry

end NUMINAMATH_GPT_find_correct_fraction_l1706_170628


namespace NUMINAMATH_GPT_marble_leftovers_l1706_170666

theorem marble_leftovers :
  ∃ r p : ℕ, (r % 8 = 5) ∧ (p % 8 = 7) ∧ ((r + p) % 10 = 0) → ((r + p) % 8 = 4) :=
by { sorry }

end NUMINAMATH_GPT_marble_leftovers_l1706_170666


namespace NUMINAMATH_GPT_find_the_number_l1706_170697

theorem find_the_number (x : ℝ) (h : 150 - x = x + 68) : x = 41 :=
sorry

end NUMINAMATH_GPT_find_the_number_l1706_170697


namespace NUMINAMATH_GPT_range_of_a_l1706_170684

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x ^ 2 + 2 * a * x + 1 > 0) → (0 ≤ a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1706_170684


namespace NUMINAMATH_GPT_number_of_valid_n_l1706_170699

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ m n : ℕ, n = 2^m * 5^n

def has_nonzero_thousandths_digit (n : ℕ) : Prop :=
  -- Placeholder for a formal definition to check the non-zero thousandths digit.
  sorry

theorem number_of_valid_n : 
  (∃ l : List ℕ, 
    l.length = 10 ∧ 
    ∀ n ∈ l, n <= 200 ∧ is_terminating_decimal n ∧ has_nonzero_thousandths_digit n) :=
sorry

end NUMINAMATH_GPT_number_of_valid_n_l1706_170699


namespace NUMINAMATH_GPT_initial_amount_is_3_l1706_170611

-- Define the initial amount of water in the bucket
def initial_water_amount (total water_added : ℝ) : ℝ :=
  total - water_added

-- Define the variables
def total : ℝ := 9.8
def water_added : ℝ := 6.8

-- State the problem
theorem initial_amount_is_3 : initial_water_amount total water_added = 3 := 
  by
    sorry

end NUMINAMATH_GPT_initial_amount_is_3_l1706_170611


namespace NUMINAMATH_GPT_neg_p_equiv_exists_leq_l1706_170636

-- Define the given proposition p
def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- State the equivalence we need to prove
theorem neg_p_equiv_exists_leq :
  ¬ p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by {
  sorry  -- Proof is skipped as per instructions
}

end NUMINAMATH_GPT_neg_p_equiv_exists_leq_l1706_170636


namespace NUMINAMATH_GPT_sin_2x_and_tan_fraction_l1706_170625

open Real

theorem sin_2x_and_tan_fraction (x : ℝ) (h : sin (π + x) + cos (π + x) = 1 / 2) :
  (sin (2 * x) = -3 / 4) ∧ ((1 + tan x) / (sin x * cos (x - π / 4)) = -8 * sqrt 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_sin_2x_and_tan_fraction_l1706_170625


namespace NUMINAMATH_GPT_average_interest_rate_l1706_170609

theorem average_interest_rate (I : ℝ) (r1 r2 : ℝ) (y : ℝ)
  (h0 : I = 6000)
  (h1 : r1 = 0.05)
  (h2 : r2 = 0.07)
  (h3 : 0.05 * (6000 - y) = 0.07 * y) :
  ((r1 * (I - y) + r2 * y) / I) = 0.05833 :=
by
  sorry

end NUMINAMATH_GPT_average_interest_rate_l1706_170609


namespace NUMINAMATH_GPT_integer_solution_count_eq_eight_l1706_170601

theorem integer_solution_count_eq_eight : ∃ S : Finset (ℤ × ℤ), (∀ s ∈ S, 2 * s.1 ^ 2 + s.1 * s.2 - s.2 ^ 2 = 14 ∧ (s.1 = s.1 ∧ s.2 = s.2)) ∧ S.card = 8 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_count_eq_eight_l1706_170601


namespace NUMINAMATH_GPT_bob_hair_growth_time_l1706_170629

theorem bob_hair_growth_time (initial_length final_length growth_rate monthly_to_yearly_conversion : ℝ) 
  (initial_cut : initial_length = 6) 
  (current_length : final_length = 36) 
  (growth_per_month : growth_rate = 0.5) 
  (months_in_year : monthly_to_yearly_conversion = 12) : 
  (final_length - initial_length) / (growth_rate * monthly_to_yearly_conversion) = 5 :=
by
  sorry

end NUMINAMATH_GPT_bob_hair_growth_time_l1706_170629


namespace NUMINAMATH_GPT_inequality_x2_y4_z6_l1706_170617

variable (x y z : ℝ)

theorem inequality_x2_y4_z6
    (hx : 0 < x)
    (hy : 0 < y)
    (hz : 0 < z) :
    x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_x2_y4_z6_l1706_170617


namespace NUMINAMATH_GPT_probability_both_tell_truth_l1706_170668

variable (P_A : ℝ) (P_B : ℝ)

theorem probability_both_tell_truth (hA : P_A = 0.75) (hB : P_B = 0.60) : P_A * P_B = 0.45 :=
by
  rw [hA, hB]
  norm_num

end NUMINAMATH_GPT_probability_both_tell_truth_l1706_170668


namespace NUMINAMATH_GPT_correct_operation_l1706_170658

theorem correct_operation (a b : ℤ) : -3 * (a - b) = -3 * a + 3 * b := 
sorry

end NUMINAMATH_GPT_correct_operation_l1706_170658


namespace NUMINAMATH_GPT_scientific_notation_correct_l1706_170685

noncomputable def significant_figures : ℝ := 274
noncomputable def decimal_places : ℝ := 8
noncomputable def scientific_notation_rep : ℝ := 2.74 * (10^8)

theorem scientific_notation_correct :
  274000000 = scientific_notation_rep :=
sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1706_170685


namespace NUMINAMATH_GPT_children_on_bus_l1706_170690

/-- Prove the number of children on the bus after the bus stop equals 14 given the initial conditions -/
theorem children_on_bus (initial_children : ℕ) (children_got_off : ℕ) (extra_children_got_on : ℕ) (final_children : ℤ) :
  initial_children = 5 →
  children_got_off = 63 →
  extra_children_got_on = 9 →
  final_children = (initial_children - children_got_off) + (children_got_off + extra_children_got_on) →
  final_children = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_children_on_bus_l1706_170690


namespace NUMINAMATH_GPT_minimize_y_l1706_170634

noncomputable def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

theorem minimize_y (a b c : ℝ) : ∃ x : ℝ, (∀ x0 : ℝ, y x a b c ≤ y x0 a b c) ∧ x = (a + b + c) / 3 :=
by
  sorry

end NUMINAMATH_GPT_minimize_y_l1706_170634


namespace NUMINAMATH_GPT_yellow_percentage_l1706_170627

theorem yellow_percentage (s w : ℝ) 
  (h_cross : w * w + 4 * w * (s - 2 * w) = 0.49 * s * s) : 
  (w / s) ^ 2 = 0.2514 :=
by
  sorry

end NUMINAMATH_GPT_yellow_percentage_l1706_170627


namespace NUMINAMATH_GPT_roshini_spent_on_sweets_l1706_170644

variable (initial_amount friends_amount total_friends_amount sweets_amount : ℝ)

noncomputable def Roshini_conditions (initial_amount friends_amount total_friends_amount sweets_amount : ℝ) :=
  initial_amount = 10.50 ∧ friends_amount = 6.80 ∧ sweets_amount = 3.70 ∧ 2 * 3.40 = 6.80

theorem roshini_spent_on_sweets :
  ∀ (initial_amount friends_amount total_friends_amount sweets_amount : ℝ),
    Roshini_conditions initial_amount friends_amount total_friends_amount sweets_amount →
    initial_amount - friends_amount = sweets_amount :=
by
  intros initial_amount friends_amount total_friends_amount sweets_amount h
  cases h
  sorry

end NUMINAMATH_GPT_roshini_spent_on_sweets_l1706_170644


namespace NUMINAMATH_GPT_unique_solution_to_functional_eq_l1706_170671

theorem unique_solution_to_functional_eq :
  (∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 6 * x^2 * f y + 2 * x^2 * y^2) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_to_functional_eq_l1706_170671


namespace NUMINAMATH_GPT_sum_of_roots_l1706_170639

theorem sum_of_roots (f : ℝ → ℝ) :
  (∀ x : ℝ, f (2 + x) = f (2 - x)) →
  (∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) →
  a + b + c + d = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1706_170639


namespace NUMINAMATH_GPT_area_of_region_l1706_170689

noncomputable def region_area : ℝ :=
  sorry

theorem area_of_region :
  region_area = sorry := 
sorry

end NUMINAMATH_GPT_area_of_region_l1706_170689


namespace NUMINAMATH_GPT_cube_sum_div_by_nine_l1706_170655

theorem cube_sum_div_by_nine (n : ℕ) (hn : 0 < n) : (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 := by sorry

end NUMINAMATH_GPT_cube_sum_div_by_nine_l1706_170655


namespace NUMINAMATH_GPT_tom_initial_foreign_exchange_l1706_170606

theorem tom_initial_foreign_exchange (x : ℝ) (y₀ y₁ y₂ y₃ y₄ : ℝ) :
  y₀ = x / 2 - 5 ∧
  y₁ = y₀ / 2 - 5 ∧
  y₂ = y₁ / 2 - 5 ∧
  y₃ = y₂ / 2 - 5 ∧
  y₄ = y₃ / 2 - 5 ∧
  y₄ - 5 = 100
  → x = 3355 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tom_initial_foreign_exchange_l1706_170606


namespace NUMINAMATH_GPT_measure_angle_F_correct_l1706_170665

noncomputable def measure_angle_D : ℝ := 80
noncomputable def measure_angle_F : ℝ := 70 / 3
noncomputable def measure_angle_E (angle_F : ℝ) : ℝ := 2 * angle_F + 30
noncomputable def angle_sum_property (angle_D angle_E angle_F : ℝ) : Prop :=
  angle_D + angle_E + angle_F = 180

theorem measure_angle_F_correct : measure_angle_F = 70 / 3 :=
by
  let angle_D := measure_angle_D
  let angle_F := measure_angle_F
  have h1 : measure_angle_E angle_F = 2 * angle_F + 30 := rfl
  have h2 : angle_sum_property angle_D (measure_angle_E angle_F) angle_F := sorry
  sorry

end NUMINAMATH_GPT_measure_angle_F_correct_l1706_170665


namespace NUMINAMATH_GPT_find_rectangle_pairs_l1706_170657

theorem find_rectangle_pairs (w l : ℕ) (hw : w > 0) (hl : l > 0) (h : w * l = 18) : 
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) ∨
  (w, l) = (6, 3) ∨ (w, l) = (9, 2) ∨ (w, l) = (18, 1) :=
by
  sorry

end NUMINAMATH_GPT_find_rectangle_pairs_l1706_170657


namespace NUMINAMATH_GPT_smallest_norm_of_v_l1706_170632

variables (v : ℝ × ℝ)

def vector_condition (v : ℝ × ℝ) : Prop :=
  ‖(v.1 - 2, v.2 + 4)‖ = 10

theorem smallest_norm_of_v
  (hv : vector_condition v) :
  ‖v‖ ≥ 10 - 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_smallest_norm_of_v_l1706_170632


namespace NUMINAMATH_GPT_strictly_increasing_interval_l1706_170637

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3)

theorem strictly_increasing_interval :
  (∀ k : ℤ, ∀ x : ℝ, 
    (2 * k * Real.pi - 5 * Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 6) 
    → (f x) < (f (x + 1))) :=
by 
  sorry

end NUMINAMATH_GPT_strictly_increasing_interval_l1706_170637


namespace NUMINAMATH_GPT_at_least_two_equal_l1706_170667

noncomputable def positive_reals (x y z : ℝ) : Prop :=
x > 0 ∧ y > 0 ∧ z > 0

noncomputable def triangle_inequality_for_n (x y z : ℝ) (n : ℕ) : Prop :=
(x^n + y^n > z^n) ∧ (y^n + z^n > x^n) ∧ (z^n + x^n > y^n)

theorem at_least_two_equal (x y z : ℝ) 
  (pos : positive_reals x y z) 
  (triangle_ineq: ∀ n : ℕ, n > 0 → triangle_inequality_for_n x y z n) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end NUMINAMATH_GPT_at_least_two_equal_l1706_170667


namespace NUMINAMATH_GPT_jana_height_l1706_170610

theorem jana_height (Jess_height : ℕ) (h1 : Jess_height = 72) 
  (Kelly_height : ℕ) (h2 : Kelly_height = Jess_height - 3) 
  (Jana_height : ℕ) (h3 : Jana_height = Kelly_height + 5) : 
  Jana_height = 74 := by
  subst h1
  subst h2
  subst h3
  sorry

end NUMINAMATH_GPT_jana_height_l1706_170610


namespace NUMINAMATH_GPT_measure_of_y_l1706_170660

theorem measure_of_y (y : ℕ) (h₁ : 40 + 2 * y + y = 180) : y = 140 / 3 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_y_l1706_170660


namespace NUMINAMATH_GPT_total_opponents_runs_l1706_170678

theorem total_opponents_runs (team_scores : List ℕ) (opponent_scores : List ℕ) :
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] →
  ∃ lost_games won_games opponent_lost_scores opponent_won_scores,
    lost_games = [1, 3, 5, 7, 9, 11] ∧
    won_games = [2, 4, 6, 8, 10, 12] ∧
    (∀ (t : ℕ), t ∈ lost_games → ∃ o : ℕ, o = t + 1 ∧ o ∈ opponent_scores) ∧
    (∀ (t : ℕ), t ∈ won_games → ∃ o : ℕ, o = t / 2 ∧ o ∈ opponent_scores) ∧
    opponent_scores = opponent_lost_scores ++ opponent_won_scores ∧
    opponent_lost_scores = [2, 4, 6, 8, 10, 12] ∧
    opponent_won_scores = [1, 2, 3, 4, 5, 6] →
  opponent_scores.sum = 63 :=
by
  sorry

end NUMINAMATH_GPT_total_opponents_runs_l1706_170678


namespace NUMINAMATH_GPT_average_is_4_l1706_170648

theorem average_is_4 (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) : 
  (p + q + r + s) / 4 = 4 := 
by 
  sorry 

end NUMINAMATH_GPT_average_is_4_l1706_170648


namespace NUMINAMATH_GPT_total_sounds_produced_l1706_170653

-- Defining the total number of nails for one customer and the number of customers
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3

-- Proving the total number of nail trimming sounds for 3 customers = 60
theorem total_sounds_produced : nails_per_person * number_of_customers = 60 := by
  sorry

end NUMINAMATH_GPT_total_sounds_produced_l1706_170653


namespace NUMINAMATH_GPT_range_of_k_l1706_170603

theorem range_of_k (a k : ℝ) : 
  (∀ x y : ℝ, y^2 - x * y + 2 * x + k = 0 → (x = a ∧ y = -a)) →
  k ≤ 1/2 :=
by sorry

end NUMINAMATH_GPT_range_of_k_l1706_170603


namespace NUMINAMATH_GPT_find_slower_train_speed_l1706_170686

theorem find_slower_train_speed (l : ℝ) (vf : ℝ) (t : ℝ) (v_s : ℝ) 
  (h1 : l = 37.5)   -- Length of each train
  (h2 : vf = 46)   -- Speed of the faster train in km/hr
  (h3 : t = 27)    -- Time in seconds to pass the slower train
  (h4 : (2 * l) = ((46 - v_s) * (5 / 18) * 27))   -- Distance covered at relative speed
  : v_s = 36 := 
sorry

end NUMINAMATH_GPT_find_slower_train_speed_l1706_170686


namespace NUMINAMATH_GPT_equation_of_circle_l1706_170673

-- Definitions directly based on conditions 
noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)
noncomputable def directrix_of_parabola : ℝ × ℝ -> Prop
  | (x, _) => x = -1

-- The statement of the problem: equation of the circle with given conditions
theorem equation_of_circle : ∃ (r : ℝ), (∀ (x y : ℝ), (x - 1)^2 + y^2 = r^2) ∧ r = 2 :=
sorry

end NUMINAMATH_GPT_equation_of_circle_l1706_170673


namespace NUMINAMATH_GPT_total_cleaning_time_l1706_170647

-- Definition for the problem conditions
def time_to_clean_egg (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ) : ℕ :=
  (num_eggs * seconds_per_egg) / seconds_per_minute

def time_to_clean_toilet_paper (minutes_per_roll : ℕ) (num_rolls : ℕ) : ℕ :=
  num_rolls * minutes_per_roll

-- Main statement to prove the total cleaning time
theorem total_cleaning_time
  (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ)
  (minutes_per_roll : ℕ) (num_rolls : ℕ) :
  seconds_per_egg = 15 →
  num_eggs = 60 →
  seconds_per_minute = 60 →
  minutes_per_roll = 30 →
  num_rolls = 7 →
  time_to_clean_egg seconds_per_egg num_eggs seconds_per_minute +
  time_to_clean_toilet_paper minutes_per_roll num_rolls = 225 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_cleaning_time_l1706_170647


namespace NUMINAMATH_GPT_alex_wins_if_picks_two_l1706_170670

theorem alex_wins_if_picks_two (matches_left : ℕ) (alex_picks bob_picks : ℕ) :
  matches_left = 30 →
  1 ≤ alex_picks ∧ alex_picks ≤ 6 →
  1 ≤ bob_picks ∧ bob_picks ≤ 6 →
  alex_picks = 2 →
  (∀ n, (n % 7 ≠ 0) → ¬ (∃ k, matches_left - k ≤ 0 ∧ (matches_left - k) % 7 = 0)) :=
by sorry

end NUMINAMATH_GPT_alex_wins_if_picks_two_l1706_170670


namespace NUMINAMATH_GPT_trigonometric_identity_l1706_170696

open Real

theorem trigonometric_identity
  (theta : ℝ)
  (h : cos (π / 6 - theta) = 2 * sqrt 2 / 3) : 
  cos (π / 3 + theta) = 1 / 3 ∨ cos (π / 3 + theta) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1706_170696


namespace NUMINAMATH_GPT_subject_selection_ways_l1706_170654

theorem subject_selection_ways :
  let compulsory := 3 -- Chinese, Mathematics, English
  let choose_one := 2
  let choose_two := 6
  compulsory + choose_one * choose_two = 12 :=
by
  sorry

end NUMINAMATH_GPT_subject_selection_ways_l1706_170654


namespace NUMINAMATH_GPT_hockey_season_length_l1706_170635

theorem hockey_season_length (total_games_per_month : ℕ) (total_games_season : ℕ) 
  (h1 : total_games_per_month = 13) (h2 : total_games_season = 182) : 
  total_games_season / total_games_per_month = 14 := 
by 
  sorry

end NUMINAMATH_GPT_hockey_season_length_l1706_170635


namespace NUMINAMATH_GPT_tile_floor_multiple_of_seven_l1706_170662

theorem tile_floor_multiple_of_seven (n : ℕ) (a : ℕ)
  (h1 : n * n = 7 * a)
  (h2 : 4 * a / 7 + 3 * a / 7 = a) :
  ∃ k : ℕ, n = 7 * k := by
  sorry

end NUMINAMATH_GPT_tile_floor_multiple_of_seven_l1706_170662


namespace NUMINAMATH_GPT_freshman_class_count_l1706_170664

theorem freshman_class_count : ∃ n : ℤ, n < 500 ∧ n % 25 = 24 ∧ n % 19 = 11 ∧ n = 49 := by
  sorry

end NUMINAMATH_GPT_freshman_class_count_l1706_170664


namespace NUMINAMATH_GPT_soap_box_width_l1706_170630

theorem soap_box_width
  (carton_length : ℝ) (carton_width : ℝ) (carton_height : ℝ)
  (box_length : ℝ) (box_height : ℝ) (max_boxes : ℝ) (carton_volume : ℝ)
  (box_volume : ℝ) (W : ℝ) : 
  carton_length = 25 →
  carton_width = 42 →
  carton_height = 60 →
  box_length = 6 →
  box_height = 6 →
  max_boxes = 250 →
  carton_volume = carton_length * carton_width * carton_height →
  box_volume = box_length * W * box_height →
  max_boxes * box_volume = carton_volume →
  W = 7 :=
sorry

end NUMINAMATH_GPT_soap_box_width_l1706_170630


namespace NUMINAMATH_GPT_ordered_pairs_unique_solution_l1706_170641

theorem ordered_pairs_unique_solution :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_ordered_pairs_unique_solution_l1706_170641


namespace NUMINAMATH_GPT_probability_C_calc_l1706_170638

noncomputable section

-- Define the given probabilities
def prob_A : ℚ := 3 / 8
def prob_B : ℚ := 1 / 4
def prob_C : ℚ := 3 / 16
def prob_D : ℚ := prob_C

-- The sum of probabilities equals 1
theorem probability_C_calc :
  prob_A + prob_B + prob_C + prob_D = 1 :=
by
  -- Simplifying directly, we can assert the correctness of given prob_C
  sorry

end NUMINAMATH_GPT_probability_C_calc_l1706_170638


namespace NUMINAMATH_GPT_nate_search_time_l1706_170692

theorem nate_search_time
  (rowsG : Nat) (cars_per_rowG : Nat)
  (rowsH : Nat) (cars_per_rowH : Nat)
  (rowsI : Nat) (cars_per_rowI : Nat)
  (walk_speed : Nat) : Nat :=
  let total_cars : Nat := rowsG * cars_per_rowG + rowsH * cars_per_rowH + rowsI * cars_per_rowI
  let total_minutes : Nat := total_cars / walk_speed
  if total_cars % walk_speed == 0 then total_minutes else total_minutes + 1

/-- Given:
- rows in Section G = 18, cars per row in Section G = 12
- rows in Section H = 25, cars per row in Section H = 10
- rows in Section I = 17, cars per row in Section I = 11
- Nate's walking speed is 8 cars per minute
Prove: Nate took 82 minutes to search the parking lot
-/
example : nate_search_time 18 12 25 10 17 11 8 = 82 := by
  sorry

end NUMINAMATH_GPT_nate_search_time_l1706_170692


namespace NUMINAMATH_GPT_average_salary_all_workers_l1706_170652

theorem average_salary_all_workers 
  (n : ℕ) (avg_salary_technicians avg_salary_rest total_avg_salary : ℝ)
  (h1 : n = 7) 
  (h2 : avg_salary_technicians = 8000) 
  (h3 : avg_salary_rest = 6000)
  (h4 : total_avg_salary = avg_salary_technicians) : 
  total_avg_salary = 8000 :=
by sorry

end NUMINAMATH_GPT_average_salary_all_workers_l1706_170652


namespace NUMINAMATH_GPT_distance_incenters_ACD_BCD_l1706_170698

noncomputable def distance_between_incenters (AC : ℝ) (angle_ABC : ℝ) (angle_BAC : ℝ) : ℝ :=
  -- Use the given conditions to derive the distance value
  -- Skipping the detailed calculations, denoted by "sorry"
  sorry

theorem distance_incenters_ACD_BCD :
  distance_between_incenters 1 (30 : ℝ) (60 : ℝ) = 0.5177 := sorry

end NUMINAMATH_GPT_distance_incenters_ACD_BCD_l1706_170698


namespace NUMINAMATH_GPT_contrapositive_of_square_inequality_l1706_170607

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x > y → x^2 > y^2) ↔ (x^2 ≤ y^2 → x ≤ y) :=
sorry

end NUMINAMATH_GPT_contrapositive_of_square_inequality_l1706_170607


namespace NUMINAMATH_GPT_packs_of_string_cheese_l1706_170688

theorem packs_of_string_cheese (cost_per_piece: ℕ) (pieces_per_pack: ℕ) (total_cost_dollars: ℕ) 
                                (h1: cost_per_piece = 10) 
                                (h2: pieces_per_pack = 20) 
                                (h3: total_cost_dollars = 6) : 
  (total_cost_dollars * 100) / (cost_per_piece * pieces_per_pack) = 3 := 
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_packs_of_string_cheese_l1706_170688


namespace NUMINAMATH_GPT_θ_values_l1706_170615

-- Define the given conditions
def terminal_side_coincides (θ : ℝ) : Prop :=
  ∃ k : ℤ, 7 * θ = θ + 360 * k

def θ_in_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

-- The main theorem
theorem θ_values (θ : ℝ) (h_terminal : terminal_side_coincides θ) (h_range : θ_in_range θ) :
  θ = 0 ∨ θ = 60 ∨ θ = 120 ∨ θ = 180 ∨ θ = 240 ∨ θ = 300 :=
sorry

end NUMINAMATH_GPT_θ_values_l1706_170615


namespace NUMINAMATH_GPT_robot_distance_proof_l1706_170626

noncomputable def distance (south1 south2 south3 east1 east2 : ℝ) : ℝ :=
  Real.sqrt ((south1 + south2 + south3)^2 + (east1 + east2)^2)

theorem robot_distance_proof :
  distance 1.2 1.8 1.0 1.0 2.0 = 5.0 :=
by
  sorry

end NUMINAMATH_GPT_robot_distance_proof_l1706_170626


namespace NUMINAMATH_GPT_ellipse_eq_from_hyperbola_l1706_170676

noncomputable def hyperbola_eq : Prop :=
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = -1) →
  (x^2 / 4 + y^2 / 16 = 1)

theorem ellipse_eq_from_hyperbola :
  hyperbola_eq :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eq_from_hyperbola_l1706_170676


namespace NUMINAMATH_GPT_simplify_sin_cos_expr_cos_pi_six_alpha_expr_l1706_170675

open Real

-- Problem (1)
theorem simplify_sin_cos_expr (x : ℝ) :
  (sin x ^ 2 / (sin x - cos x)) - ((sin x + cos x) / (tan x ^ 2 - 1)) - sin x = cos x :=
sorry

-- Problem (2)
theorem cos_pi_six_alpha_expr (α : ℝ) (h : cos (π / 6 - α) = sqrt 3 / 3) :
  cos (5 * π / 6 + α) + cos (4 * π / 3 + α) ^ 2 = (2 - sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_simplify_sin_cos_expr_cos_pi_six_alpha_expr_l1706_170675


namespace NUMINAMATH_GPT_total_budget_l1706_170682

theorem total_budget (s_ticket : ℕ) (s_drinks_food : ℕ) (k_ticket : ℕ) (k_drinks : ℕ) (k_food : ℕ) 
  (h1 : s_ticket = 14) (h2 : s_drinks_food = 6) (h3 : k_ticket = 14) (h4 : k_drinks = 2) (h5 : k_food = 4) : 
  s_ticket + s_drinks_food + k_ticket + k_drinks + k_food = 40 := 
by
  sorry

end NUMINAMATH_GPT_total_budget_l1706_170682


namespace NUMINAMATH_GPT_greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l1706_170605

theorem greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30 :
  ∃ d, d ∣ 480 ∧ d < 60 ∧ d ∣ 90 ∧ (∀ e, e ∣ 480 → e < 60 → e ∣ 90 → e ≤ d) ∧ d = 30 :=
sorry

end NUMINAMATH_GPT_greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l1706_170605


namespace NUMINAMATH_GPT_henry_age_is_29_l1706_170622

-- Definitions and conditions
variable (Henry_age Jill_age : ℕ)

-- Condition 1: Sum of the present age of Henry and Jill is 48
def sum_of_ages : Prop := Henry_age + Jill_age = 48

-- Condition 2: Nine years ago, Henry was twice the age of Jill
def age_relation_nine_years_ago : Prop := Henry_age - 9 = 2 * (Jill_age - 9)

-- Theorem to prove
theorem henry_age_is_29 (H: ℕ) (J: ℕ)
  (h1 : sum_of_ages H J) 
  (h2 : age_relation_nine_years_ago H J) : H = 29 :=
by
  sorry

end NUMINAMATH_GPT_henry_age_is_29_l1706_170622


namespace NUMINAMATH_GPT_maximum_books_l1706_170677

theorem maximum_books (dollars : ℝ) (price_per_book : ℝ) (n : ℕ) 
    (h1 : dollars = 12) (h2 : price_per_book = 1.25) : n ≤ 9 :=
    sorry

end NUMINAMATH_GPT_maximum_books_l1706_170677


namespace NUMINAMATH_GPT_integer_values_of_b_l1706_170642

theorem integer_values_of_b (b : ℤ) :
  (∃ x : ℤ, x^3 + 2*x^2 + b*x + 18 = 0) ↔ 
  b = -21 ∨ b = 19 ∨ b = -17 ∨ b = -4 ∨ b = 3 :=
by
  sorry

end NUMINAMATH_GPT_integer_values_of_b_l1706_170642


namespace NUMINAMATH_GPT_penalty_kicks_calculation_l1706_170633

def totalPlayers := 24
def goalkeepers := 4
def nonGoalkeeperShootsAgainstOneGoalkeeper := totalPlayers - 1
def totalPenaltyKicks := goalkeepers * nonGoalkeeperShootsAgainstOneGoalkeeper

theorem penalty_kicks_calculation : totalPenaltyKicks = 92 := by
  sorry

end NUMINAMATH_GPT_penalty_kicks_calculation_l1706_170633


namespace NUMINAMATH_GPT_angle_BAO_eq_angle_CAH_l1706_170618

noncomputable def is_triangle (A B C : Type) : Prop := sorry
noncomputable def orthocenter (A B C H : Type) : Prop := sorry
noncomputable def circumcenter (A B C O : Type) : Prop := sorry
noncomputable def angle (A B C : Type) : Type := sorry

theorem angle_BAO_eq_angle_CAH (A B C H O : Type) 
  (hABC : is_triangle A B C)
  (hH : orthocenter A B C H)
  (hO : circumcenter A B C O):
  angle B A O = angle C A H := 
  sorry

end NUMINAMATH_GPT_angle_BAO_eq_angle_CAH_l1706_170618


namespace NUMINAMATH_GPT_brothers_percentage_fewer_trees_l1706_170640

theorem brothers_percentage_fewer_trees (total_trees initial_days brother_days : ℕ) (trees_per_day : ℕ) (total_brother_trees : ℕ) (percentage_fewer : ℕ):
  initial_days = 2 →
  brother_days = 3 →
  trees_per_day = 20 →
  total_trees = 196 →
  total_brother_trees = total_trees - (trees_per_day * initial_days) →
  percentage_fewer = ((total_brother_trees / brother_days - trees_per_day) * 100) / trees_per_day →
  percentage_fewer = 60 :=
by
  sorry

end NUMINAMATH_GPT_brothers_percentage_fewer_trees_l1706_170640


namespace NUMINAMATH_GPT_remainder_of_7n_mod_4_l1706_170694

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_of_7n_mod_4_l1706_170694


namespace NUMINAMATH_GPT_six_diggers_five_hours_l1706_170624

theorem six_diggers_five_hours (holes_per_hour_per_digger : ℝ) 
  (h1 : 3 * holes_per_hour_per_digger * 3 = 3) :
  6 * (holes_per_hour_per_digger) * 5 = 10 :=
by
  -- The proof will go here, but we only need to state the theorem
  sorry

end NUMINAMATH_GPT_six_diggers_five_hours_l1706_170624


namespace NUMINAMATH_GPT_ratio_of_a_and_b_l1706_170669

theorem ratio_of_a_and_b (x y a b : ℝ) (h1 : x / y = 3) (h2 : (2 * a - x) / (3 * b - y) = 3) : a / b = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_a_and_b_l1706_170669


namespace NUMINAMATH_GPT_option_d_l1706_170661

variable {R : Type*} [LinearOrderedField R]

theorem option_d (a b c d : R) (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by 
  sorry

end NUMINAMATH_GPT_option_d_l1706_170661


namespace NUMINAMATH_GPT_infinite_rationals_sqrt_rational_l1706_170683

theorem infinite_rationals_sqrt_rational : ∃ᶠ x : ℚ in Filter.atTop, ∃ y : ℚ, y = Real.sqrt (x^2 + x + 1) :=
sorry

end NUMINAMATH_GPT_infinite_rationals_sqrt_rational_l1706_170683


namespace NUMINAMATH_GPT_answer_l1706_170680

def p : Prop := ∃ x > Real.exp 1, (1 / 2)^x > Real.log x
def q : Prop := ∀ a b : Real, a > 1 → b > 1 → Real.log a / Real.log b + 2 * (Real.log b / Real.log a) ≥ 2 * Real.sqrt 2

theorem answer : ¬ p ∧ q :=
by
  have h1 : ¬ p := sorry
  have h2 : q := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_answer_l1706_170680


namespace NUMINAMATH_GPT_elberta_money_l1706_170600

theorem elberta_money (granny_smith : ℕ) (anjou : ℕ) (elberta : ℕ) 
  (h1 : granny_smith = 120) 
  (h2 : anjou = granny_smith / 4) 
  (h3 : elberta = anjou + 5) : 
  elberta = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_elberta_money_l1706_170600


namespace NUMINAMATH_GPT_Milly_study_time_l1706_170621

theorem Milly_study_time :
  let math_time := 60
  let geo_time := math_time / 2
  let mean_time := (math_time + geo_time) / 2
  let total_study_time := math_time + geo_time + mean_time
  total_study_time = 135 := by
  sorry

end NUMINAMATH_GPT_Milly_study_time_l1706_170621


namespace NUMINAMATH_GPT_harold_catches_up_at_12_miles_l1706_170643

/-- 
Proof Problem: Given that Adrienne starts walking from X to Y at 3 miles per hour and one hour later Harold starts walking from X to Y at 4 miles per hour, prove that Harold covers 12 miles when he catches up to Adrienne.
-/
theorem harold_catches_up_at_12_miles :
  (∀ (T : ℕ), (ad_distance : ℕ) = 3 * (T + 1) → (ha_distance : ℕ) = 4 * T → ad_distance = ha_distance) →
  (∃ T : ℕ, ha_distance = 12) :=
by
  sorry

end NUMINAMATH_GPT_harold_catches_up_at_12_miles_l1706_170643


namespace NUMINAMATH_GPT_combination_2586_1_eq_2586_l1706_170608

noncomputable def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem combination_2586_1_eq_2586 : combination 2586 1 = 2586 := by
  sorry

end NUMINAMATH_GPT_combination_2586_1_eq_2586_l1706_170608


namespace NUMINAMATH_GPT_arman_sister_age_l1706_170651

-- Define the conditions
variables (S : ℝ) -- Arman's sister's age four years ago
variable (A : ℝ) -- Arman's age four years ago

-- Given conditions as hypotheses
axiom h1 : A = 6 * S -- Arman is six times older than his sister
axiom h2 : A + 8 = 40 -- In 4 years, Arman's age will be 40 (hence, A in 4 years should be A + 8)

-- Main theorem to prove
theorem arman_sister_age (h1 : A = 6 * S) (h2 : A + 8 = 40) : S = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arman_sister_age_l1706_170651


namespace NUMINAMATH_GPT_part1_part2_l1706_170663

def star (a b c d : ℝ) : ℝ := a * c - b * d

-- Part (1)
theorem part1 : star (-4) 3 2 (-6) = 10 := by
  sorry

-- Part (2)
theorem part2 (m : ℝ) (h : ∀ x : ℝ, star x (2 * x - 1) (m * x + 1) m = 0 → (m ≠ 0 → (((1 - 2 * m) ^ 2 - 4 * m * m) ≥ 0))) :
  (m ≤ 1 / 4 ∨ m < 0) ∧ m ≠ 0 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1706_170663


namespace NUMINAMATH_GPT_find_cos_A_l1706_170646

theorem find_cos_A
  (A C : ℝ)
  (AB CD : ℝ)
  (AD BC : ℝ)
  (α : ℝ)
  (h1 : A = C)
  (h2 : AB = 150)
  (h3 : CD = 150)
  (h4 : AD ≠ BC)
  (h5 : AB + BC + CD + AD = 560)
  (h6 : A = α)
  (h7 : C = α)
  (BD₁ BD₂ : ℝ)
  (h8 : BD₁^2 = AD^2 + 150^2 - 2 * 150 * AD * Real.cos α)
  (h9 : BD₂^2 = BC^2 + 150^2 - 2 * 150 * BC * Real.cos α)
  (h10 : BD₁ = BD₂) :
  Real.cos A = 13 / 15 := 
sorry

end NUMINAMATH_GPT_find_cos_A_l1706_170646


namespace NUMINAMATH_GPT_percentage_calculation_l1706_170613

def percentage_less_than_50000_towns : Float := 85

def percentage_less_than_20000_towns : Float := 20
def percentage_20000_to_49999_towns : Float := 65

theorem percentage_calculation :
  percentage_less_than_50000_towns = percentage_less_than_20000_towns + percentage_20000_to_49999_towns :=
by
  sorry

end NUMINAMATH_GPT_percentage_calculation_l1706_170613


namespace NUMINAMATH_GPT_arithmetic_mean_is_ten_l1706_170631

theorem arithmetic_mean_is_ten (a b x : ℝ) (h₁ : a = 4) (h₂ : b = 16) (h₃ : x = (a + b) / 2) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_is_ten_l1706_170631


namespace NUMINAMATH_GPT_probability_two_females_one_male_l1706_170656

theorem probability_two_females_one_male :
  let total_contestants := 8
  let num_females := 5
  let num_males := 3
  let choose3 := Nat.choose total_contestants 3
  let choose2f := Nat.choose num_females 2
  let choose1m := Nat.choose num_males 1
  let favorable_outcomes := choose2f * choose1m
  choose3 ≠ 0 → (favorable_outcomes / choose3 : ℚ) = 15 / 28 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_females_one_male_l1706_170656


namespace NUMINAMATH_GPT_john_final_push_time_l1706_170612

theorem john_final_push_time :
  ∃ t : ℝ, (∀ (d_j d_s : ℝ), d_j = 4.2 * t ∧ d_s = 3.7 * t ∧ (d_j = d_s + 14)) → t = 28 :=
by
  sorry

end NUMINAMATH_GPT_john_final_push_time_l1706_170612


namespace NUMINAMATH_GPT_fixed_point_C_D_intersection_l1706_170623

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 2) = 1

noncomputable def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4 ∧ P.2 ≠ 0

noncomputable def line_CD_fixed_point (t : ℝ) (C D : ℝ × ℝ) : Prop :=
  let x1 := (36 - 2 * t^2) / (18 + t^2)
  let y1 := (12 * t) / (18 + t^2)
  let x2 := (2 * t^2 - 4) / (2 + t^2)
  let y2 := -(4 * t) / (t^2 + 2)
  C = (x1, y1) ∧ D = (x2, y2) →
  let k_CD := (4 * t) / (6 - t^2)
  ∀ (x y : ℝ), y + (4 * t) / (t^2 + 2) = k_CD * (x - (2 * t^2 - 4) / (t^2 + 2)) →
  y = 0 → x = 1

theorem fixed_point_C_D_intersection :
  ∀ (t : ℝ) (C D : ℝ × ℝ), point_on_line (4, t) →
  ellipse_equation C.1 C.2 →
  ellipse_equation D.1 D.2 →
  line_CD_fixed_point t C D :=
by
  intros t C D point_on_line_P ellipse_C ellipse_D
  sorry

end NUMINAMATH_GPT_fixed_point_C_D_intersection_l1706_170623


namespace NUMINAMATH_GPT_find_multiplier_l1706_170672

/-- Define the number -/
def number : ℝ := -10.0

/-- Define the multiplier m -/
def m : ℝ := 0.4

/-- Given conditions and prove the correct multiplier -/
theorem find_multiplier (number : ℝ) (m : ℝ) 
  (h1 : ∃ m : ℝ, m * number - 8 = -12) 
  (h2 : number = -10.0) : m = 0.4 :=
by
  -- We skip the actual steps and provide the answer using sorry
  sorry

end NUMINAMATH_GPT_find_multiplier_l1706_170672


namespace NUMINAMATH_GPT_min_stamps_for_target_value_l1706_170619

theorem min_stamps_for_target_value :
  ∃ (c f : ℕ), 5 * c + 7 * f = 50 ∧ ∀ (c' f' : ℕ), 5 * c' + 7 * f' = 50 → c + f ≤ c' + f' → c + f = 8 :=
by
  sorry

end NUMINAMATH_GPT_min_stamps_for_target_value_l1706_170619


namespace NUMINAMATH_GPT_mario_age_difference_l1706_170659

variable (Mario_age Maria_age : ℕ)

def age_conditions (Mario_age Maria_age difference : ℕ) : Prop :=
  Mario_age + Maria_age = 7 ∧
  Mario_age = 4 ∧
  Mario_age - Maria_age = difference

theorem mario_age_difference : ∃ (difference : ℕ), age_conditions 4 (4 - difference) difference ∧ difference = 1 := by
  sorry

end NUMINAMATH_GPT_mario_age_difference_l1706_170659


namespace NUMINAMATH_GPT_tickets_required_l1706_170687

theorem tickets_required (cost_ferris_wheel : ℝ) (cost_roller_coaster : ℝ) 
  (discount_multiple_rides : ℝ) (coupon_value : ℝ) 
  (total_cost_with_discounts : ℝ) : 
  cost_ferris_wheel = 2.0 ∧ 
  cost_roller_coaster = 7.0 ∧ 
  discount_multiple_rides = 1.0 ∧ 
  coupon_value = 1.0 → 
  total_cost_with_discounts = 7.0 :=
by
  sorry

end NUMINAMATH_GPT_tickets_required_l1706_170687


namespace NUMINAMATH_GPT_maximum_value_of_expression_l1706_170679

variable (x y z : ℝ)

theorem maximum_value_of_expression (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) : 
  x + y^3 + z^4 ≤ 1 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l1706_170679


namespace NUMINAMATH_GPT_decreasing_function_range_l1706_170645

theorem decreasing_function_range (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_range_l1706_170645


namespace NUMINAMATH_GPT_ariana_total_owe_l1706_170693

-- Definitions based on the conditions
def first_bill_principal : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_overdue_months : ℕ := 2

def second_bill_principal : ℕ := 130
def second_bill_late_fee : ℕ := 50
def second_bill_overdue_months : ℕ := 6

def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80

-- Theorem
theorem ariana_total_owe : 
  first_bill_principal + 
    (first_bill_principal : ℝ) * first_bill_interest_rate * (first_bill_overdue_months : ℝ) +
    second_bill_principal + 
    second_bill_late_fee * second_bill_overdue_months + 
    third_bill_first_month_fee + 
    third_bill_second_month_fee = 790 := 
by 
  sorry

end NUMINAMATH_GPT_ariana_total_owe_l1706_170693


namespace NUMINAMATH_GPT_total_blocks_correct_l1706_170691

-- Definitions given by the conditions in the problem
def red_blocks : ℕ := 18
def yellow_blocks : ℕ := red_blocks + 7
def blue_blocks : ℕ := red_blocks + 14

-- Theorem stating the goal to prove
theorem total_blocks_correct : red_blocks + yellow_blocks + blue_blocks = 75 := by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_total_blocks_correct_l1706_170691


namespace NUMINAMATH_GPT_least_positive_integer_n_l1706_170620

theorem least_positive_integer_n : ∃ (n : ℕ), (1 / (n : ℝ) - 1 / (n + 1) < 1 / 100) ∧ ∀ m, m < n → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 100) :=
sorry

end NUMINAMATH_GPT_least_positive_integer_n_l1706_170620


namespace NUMINAMATH_GPT_f_zero_f_odd_f_range_l1706_170695

-- Condition 1: The function f is defined on ℝ
-- Condition 2: f(x + y) = f(x) + f(y)
-- Condition 3: f(1/3) = 1
-- Condition 4: f(x) < 0 when x > 0

variables (f : ℝ → ℝ)
axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_third : f (1/3) = 1
axiom f_neg_positive : ∀ x : ℝ, 0 < x → f x < 0

-- Question 1: Find the value of f(0)
theorem f_zero : f 0 = 0 := sorry

-- Question 2: Prove that f is an odd function
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

-- Question 3: Find the range of x where f(x) + f(2 + x) < 2
theorem f_range : ∀ x : ℝ, f x + f (2 + x) < 2 → -2/3 < x := sorry

end NUMINAMATH_GPT_f_zero_f_odd_f_range_l1706_170695
