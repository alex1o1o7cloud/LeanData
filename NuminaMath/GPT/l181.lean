import Mathlib

namespace NUMINAMATH_GPT_percentage_cleared_all_sections_l181_18175

def total_candidates : ℝ := 1200
def cleared_none : ℝ := 0.05 * total_candidates
def cleared_one_section : ℝ := 0.25 * total_candidates
def cleared_four_sections : ℝ := 0.20 * total_candidates
def cleared_two_sections : ℝ := 0.245 * total_candidates
def cleared_three_sections : ℝ := 300

-- Let x be the percentage of candidates who cleared all sections
def cleared_all_sections (x: ℝ) : Prop :=
  let total_cleared := (cleared_none + 
                        cleared_one_section + 
                        cleared_four_sections + 
                        cleared_two_sections + 
                        cleared_three_sections + 
                        x * total_candidates / 100)
  total_cleared = total_candidates

theorem percentage_cleared_all_sections :
  ∃ x, cleared_all_sections x ∧ x = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_cleared_all_sections_l181_18175


namespace NUMINAMATH_GPT_initial_cards_eq_4_l181_18160

theorem initial_cards_eq_4 (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_cards_eq_4_l181_18160


namespace NUMINAMATH_GPT_max_tied_teams_round_robin_l181_18177

theorem max_tied_teams_round_robin (n : ℕ) (h: n = 8) :
  ∃ k, (k <= n) ∧ (∀ m, m > k → k * m < n * (n - 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_max_tied_teams_round_robin_l181_18177


namespace NUMINAMATH_GPT_point_on_x_axis_l181_18167

theorem point_on_x_axis (m : ℝ) (P : ℝ × ℝ) (hP : P = (m + 3, m - 1)) (hx : P.2 = 0) :
  P = (4, 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l181_18167


namespace NUMINAMATH_GPT_remainder_product_l181_18121

theorem remainder_product (x y : ℤ) 
  (hx : x % 792 = 62) 
  (hy : y % 528 = 82) : 
  (x * y) % 66 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_product_l181_18121


namespace NUMINAMATH_GPT_eq_b_minus_a_l181_18132

   -- Definition for rotating a point counterclockwise by 180° around another point
   def rotate_180 (h k x y : ℝ) : ℝ × ℝ :=
     (2 * h - x, 2 * k - y)

   -- Definition for reflecting a point about the line y = -x
   def reflect_y_eq_neg_x (x y : ℝ) : ℝ × ℝ :=
     (-y, -x)

   -- Given point Q(a, b)
   variables (a b : ℝ)

   -- Image of Q after the transformations
   def Q_transformed :=
     (5, -1)

   -- Image of Q after reflection about y = -x
   def Q_reflected :=
     reflect_y_eq_neg_x (5) (-1)

   -- Image of Q after 180° rotation around (2,3)
   def Q_original :=
     rotate_180 (2) (3) a b

   -- Statement we want to prove:
   theorem eq_b_minus_a : b - a = 6 :=
   by
     -- Calculation steps
     sorry
   
end NUMINAMATH_GPT_eq_b_minus_a_l181_18132


namespace NUMINAMATH_GPT_solve_for_x_l181_18150

theorem solve_for_x :
  ∃ x : ℝ, (24 / 36) = Real.sqrt (x / 36) ∧ x = 16 :=
by
  use 16
  sorry

end NUMINAMATH_GPT_solve_for_x_l181_18150


namespace NUMINAMATH_GPT_problem_statement_l181_18189

theorem problem_statement (x y : ℤ) (k : ℤ) (h : 4 * x - y = 3 * k) : 9 ∣ 4 * x^2 + 7 * x * y - 2 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l181_18189


namespace NUMINAMATH_GPT_talkingBirds_count_l181_18183

-- Define the conditions
def totalBirds : ℕ := 77
def nonTalkingBirds : ℕ := 13
def talkingBirds (T : ℕ) : Prop := T + nonTalkingBirds = totalBirds

-- Statement to prove
theorem talkingBirds_count : ∃ T, talkingBirds T ∧ T = 64 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_talkingBirds_count_l181_18183


namespace NUMINAMATH_GPT_factorization_proof_l181_18129

def factorization_problem (x : ℝ) : Prop := (x^2 - 1)^2 - 6 * (x^2 - 1) + 9 = (x - 2)^2 * (x + 2)^2

theorem factorization_proof (x : ℝ) : factorization_problem x :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_factorization_proof_l181_18129


namespace NUMINAMATH_GPT_part1_expression_for_f_part2_three_solutions_l181_18117

noncomputable def f1 (x : ℝ) := x^2

noncomputable def f2 (x : ℝ) := 8 / x

noncomputable def f (x : ℝ) := f1 x + f2 x

theorem part1_expression_for_f : ∀ x:ℝ, f x = x^2 + 8 / x := by
  sorry  -- This is where the proof would go

theorem part2_three_solutions (a : ℝ) (h : a > 3) : 
  ∃ x1 x2 x3 : ℝ, f x1 = f a ∧ f x2 = f a ∧ f x3 = f a ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 := by
  sorry  -- This is where the proof would go

end NUMINAMATH_GPT_part1_expression_for_f_part2_three_solutions_l181_18117


namespace NUMINAMATH_GPT_minimum_frosting_time_l181_18103

def ann_time_per_cake := 8 -- Ann's time per cake in minutes
def bob_time_per_cake := 6 -- Bob's time per cake in minutes
def carol_time_per_cake := 10 -- Carol's time per cake in minutes
def passing_time := 1 -- time to pass a cake from one person to another in minutes
def total_cakes := 10 -- total number of cakes to be frosted

theorem minimum_frosting_time : 
  (ann_time_per_cake + passing_time + bob_time_per_cake + passing_time + carol_time_per_cake) + (total_cakes - 1) * carol_time_per_cake = 116 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_frosting_time_l181_18103


namespace NUMINAMATH_GPT_evaluate_at_two_l181_18127

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 1

theorem evaluate_at_two : f 2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_at_two_l181_18127


namespace NUMINAMATH_GPT_cars_on_river_road_l181_18162

theorem cars_on_river_road (B C : ℕ) (h_ratio : B / C = 1 / 3) (h_fewer : C = B + 40) : C = 60 :=
sorry

end NUMINAMATH_GPT_cars_on_river_road_l181_18162


namespace NUMINAMATH_GPT_fraction_addition_l181_18134

theorem fraction_addition (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_addition_l181_18134


namespace NUMINAMATH_GPT_eval_expression_eq_one_l181_18156

theorem eval_expression_eq_one (x : ℝ) (hx1 : x^3 + 1 = (x+1)*(x^2 - x + 1)) (hx2 : x^3 - 1 = (x-1)*(x^2 + x + 1)) :
  ( ((x+1)^3 * (x^2 - x + 1)^3 / (x^3 + 1)^3)^2 * ((x-1)^3 * (x^2 + x + 1)^3 / (x^3 - 1)^3)^2 ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_eq_one_l181_18156


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_10_l181_18149

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable def a_n (a1 d : α) (n : ℕ) : α :=
a1 + (n - 1) • d

def sequence_sum (a1 d : α) (n : ℕ) : α :=
n • a1 + (n • (n - 1) / 2) • d

theorem arithmetic_sequence_sum_10 
  (a1 d : ℤ)
  (h1 : a_n a1 d 2 + a_n a1 d 4 = 4)
  (h2 : a_n a1 d 3 + a_n a1 d 5 = 10) :
  sequence_sum a1 d 10 = 95 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_10_l181_18149


namespace NUMINAMATH_GPT_find_positive_integers_unique_solution_l181_18168

theorem find_positive_integers_unique_solution :
  ∃ x r p n : ℕ,  
  0 < x ∧ 0 < r ∧ 0 < n ∧  Nat.Prime p ∧ 
  r > 1 ∧ n > 1 ∧ x^r - 1 = p^n ∧ 
  (x = 3 ∧ r = 2 ∧ p = 2 ∧ n = 3) := 
    sorry

end NUMINAMATH_GPT_find_positive_integers_unique_solution_l181_18168


namespace NUMINAMATH_GPT_ratio_of_areas_l181_18106

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l181_18106


namespace NUMINAMATH_GPT_park_area_calculation_l181_18115

def scale := 300 -- miles per inch
def short_diagonal := 10 -- inches
def real_length := short_diagonal * scale -- miles
def park_area := (1/2) * real_length * real_length -- square miles

theorem park_area_calculation : park_area = 4500000 := by
  sorry

end NUMINAMATH_GPT_park_area_calculation_l181_18115


namespace NUMINAMATH_GPT_sum_of_coordinates_l181_18142

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 3) : 
  let x := 2 / 3
  let y := 2 * f (3 * x) + 4
  x + y = 32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_l181_18142


namespace NUMINAMATH_GPT_Brenda_new_lead_l181_18197

noncomputable def Brenda_initial_lead : ℤ := 22
noncomputable def Brenda_play_points : ℤ := 15
noncomputable def David_play_points : ℤ := 32

theorem Brenda_new_lead : 
  Brenda_initial_lead + Brenda_play_points - David_play_points = 5 := 
by
  sorry

end NUMINAMATH_GPT_Brenda_new_lead_l181_18197


namespace NUMINAMATH_GPT_lowest_possible_students_l181_18145

theorem lowest_possible_students :
  ∃ n : ℕ, (n % 10 = 0 ∧ n % 24 = 0) ∧ n = 120 :=
by
  sorry

end NUMINAMATH_GPT_lowest_possible_students_l181_18145


namespace NUMINAMATH_GPT_choosing_ways_president_vp_committee_l181_18194

theorem choosing_ways_president_vp_committee :
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  total_choices = 2520 := by
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  have : total_choices = 2520 := by
    sorry
  exact this

end NUMINAMATH_GPT_choosing_ways_president_vp_committee_l181_18194


namespace NUMINAMATH_GPT_train_pass_bridge_in_56_seconds_l181_18180

noncomputable def time_for_train_to_pass_bridge 
(length_of_train : ℕ) (speed_of_train_kmh : ℕ) (length_of_bridge : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  total_distance / speed_of_train_ms

theorem train_pass_bridge_in_56_seconds :
  time_for_train_to_pass_bridge 560 45 140 = 56 := by
  sorry

end NUMINAMATH_GPT_train_pass_bridge_in_56_seconds_l181_18180


namespace NUMINAMATH_GPT_huahuan_initial_cards_l181_18171

theorem huahuan_initial_cards
  (a b c : ℕ) -- let a, b, c be the initial number of cards Huahuan, Yingying, and Nini have
  (total : a + b + c = 2712)
  (condition_after_50_rounds : ∃ d, b = a + d ∧ c = a + 2 * d) -- after 50 rounds, form an arithmetic sequence
  : a = 754 := sorry

end NUMINAMATH_GPT_huahuan_initial_cards_l181_18171


namespace NUMINAMATH_GPT_acute_triangle_iff_sum_of_squares_l181_18113

theorem acute_triangle_iff_sum_of_squares (a b c R : ℝ) 
  (hRpos : R > 0) 
  (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  (∀ α β γ, (a = 2 * R * Real.sin α) ∧ (b = 2 * R * Real.sin β) ∧ (c = 2 * R * Real.sin γ) → 
   (α < Real.pi / 2 ∧ β < Real.pi / 2 ∧ γ < Real.pi / 2)) ↔ 
  (a^2 + b^2 + c^2 > 8 * R^2) :=
sorry

end NUMINAMATH_GPT_acute_triangle_iff_sum_of_squares_l181_18113


namespace NUMINAMATH_GPT_number_of_pigs_l181_18104

theorem number_of_pigs (daily_feed_per_pig : ℕ) (weekly_feed_total : ℕ) (days_per_week : ℕ)
  (h1 : daily_feed_per_pig = 10) (h2 : weekly_feed_total = 140) (h3 : days_per_week = 7) : 
  (weekly_feed_total / days_per_week) / daily_feed_per_pig = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_pigs_l181_18104


namespace NUMINAMATH_GPT_solve_quadratic_solve_inequality_system_l181_18187

theorem solve_quadratic :
  ∀ x : ℝ, x^2 - 6 * x + 5 = 0 ↔ x = 1 ∨ x = 5 :=
sorry

theorem solve_inequality_system :
  ∀ x : ℝ, (x + 3 > 0 ∧ 2 * (x + 1) < 4) ↔ (-3 < x ∧ x < 1) :=
sorry

end NUMINAMATH_GPT_solve_quadratic_solve_inequality_system_l181_18187


namespace NUMINAMATH_GPT_find_eighth_number_l181_18120

def average_of_numbers (a b c d e f g h x : ℕ) : ℕ :=
  (a + b + c + d + e + f + g + h + x) / 9

theorem find_eighth_number (a b c d e f g h x : ℕ) (avg : ℕ) 
    (h_avg : average_of_numbers a b c d e f g h x = avg)
    (h_total_sum : a + b + c + d + e + f + g + h + x = 540)
    (h_x_val : x = 65) : a = 53 :=
by
  sorry

end NUMINAMATH_GPT_find_eighth_number_l181_18120


namespace NUMINAMATH_GPT_opposite_of_5_is_neg5_l181_18161

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_5_is_neg5_l181_18161


namespace NUMINAMATH_GPT_probability_A_more_than_B_sum_m_n_l181_18138

noncomputable def prob_A_more_than_B : ℚ :=
  0.6 + 0.4 * (1 / 2) * (1 - (63 / 512))

theorem probability_A_more_than_B : prob_A_more_than_B = 779 / 1024 := sorry

theorem sum_m_n : 779 + 1024 = 1803 := sorry

end NUMINAMATH_GPT_probability_A_more_than_B_sum_m_n_l181_18138


namespace NUMINAMATH_GPT_min_sum_of_squares_l181_18165

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 3 * b + 5 * c + 7 * d = 14) : 
  a^2 + b^2 + c^2 + d^2 ≥ 7 / 3 :=
sorry

end NUMINAMATH_GPT_min_sum_of_squares_l181_18165


namespace NUMINAMATH_GPT_negation_exists_ge_zero_l181_18148

theorem negation_exists_ge_zero (h : ∀ x > 0, x^2 - 3 * x + 2 < 0) :
  ∃ x > 0, x^2 - 3 * x + 2 ≥ 0 :=
sorry

end NUMINAMATH_GPT_negation_exists_ge_zero_l181_18148


namespace NUMINAMATH_GPT_f_is_monotonic_l181_18173

variable (f : ℝ → ℝ)

theorem f_is_monotonic (h : ∀ a b x : ℝ, a < x ∧ x < b → min (f a) (f b) < f x ∧ f x < max (f a) (f b)) :
  (∀ x y : ℝ, x ≤ y → f x <= f y) ∨ (∀ x y : ℝ, x ≤ y → f x >= f y) :=
sorry

end NUMINAMATH_GPT_f_is_monotonic_l181_18173


namespace NUMINAMATH_GPT_factor_x_minus_1_l181_18196

theorem factor_x_minus_1 (P Q R S : Polynomial ℂ) : 
  (P.eval 1 = 0) → 
  (P.eval (x^5) + x * Q.eval (x^5) + x^2 * R.eval (x^5) 
  = (x^4 + x^3 + x^2 + x + 1) * S.eval (x)) :=
sorry

end NUMINAMATH_GPT_factor_x_minus_1_l181_18196


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l181_18125

theorem part_a (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (ineq : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b := 
sorry

theorem part_b (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (ineq : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) := 
sorry

theorem part_c (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) :
  ¬ (a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) → 
     a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :=
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l181_18125


namespace NUMINAMATH_GPT_largest_expr_is_expr1_l181_18137

def U : ℝ := 3 * 2005 ^ 2006
def V : ℝ := 2005 ^ 2006
def W : ℝ := 2004 * 2005 ^ 2005
def X : ℝ := 3 * 2005 ^ 2005
def Y : ℝ := 2005 ^ 2005
def Z : ℝ := 2005 ^ 2004

def expr1 : ℝ := U - V
def expr2 : ℝ := V - W
def expr3 : ℝ := W - X
def expr4 : ℝ := X - Y
def expr5 : ℝ := Y - Z

theorem largest_expr_is_expr1 : 
  max (max (max expr1 expr2) (max expr3 expr4)) expr5 = expr1 := 
sorry

end NUMINAMATH_GPT_largest_expr_is_expr1_l181_18137


namespace NUMINAMATH_GPT_distance_to_x_axis_l181_18154

theorem distance_to_x_axis (x y : ℤ) (h : (x, y) = (-3, 5)) : |y| = 5 := by
  -- coordinates of point A are (-3, 5)
  sorry

end NUMINAMATH_GPT_distance_to_x_axis_l181_18154


namespace NUMINAMATH_GPT_present_price_after_discount_l181_18192

theorem present_price_after_discount :
  ∀ (P : ℝ), (∀ x : ℝ, (3 * x = P - 0.20 * P) ∧ (x = (P / 3) - 4)) → P = 60 → 0.80 * P = 48 :=
by
  intros P hP h60
  sorry

end NUMINAMATH_GPT_present_price_after_discount_l181_18192


namespace NUMINAMATH_GPT_player1_winning_strategy_l181_18119

/--
Player 1 has a winning strategy if and only if N is not an odd power of 2,
under the game rules where players alternately subtract proper divisors
and a player loses when given a prime number or 1.
-/
theorem player1_winning_strategy (N: ℕ) : 
  ¬ (∃ k: ℕ, k % 2 = 1 ∧ N = 2^k) ↔ (∃ strategy: ℕ → ℕ, ∀ n ≠ 1, n ≠ prime → n - strategy n = m) :=
sorry

end NUMINAMATH_GPT_player1_winning_strategy_l181_18119


namespace NUMINAMATH_GPT_range_of_x_l181_18169

theorem range_of_x (a : ℝ) (x : ℝ) (h₁ : a = 1) (h₂ : (x - a) * (x - 3 * a) < 0) (h₃ : 2 < x ∧ x ≤ 3) : 2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_GPT_range_of_x_l181_18169


namespace NUMINAMATH_GPT_percentage_of_original_price_l181_18101
-- Define the original price and current price in terms of real numbers
def original_price : ℝ := 25
def current_price : ℝ := 20

-- Lean statement to verify the correctness of the percentage calculation
theorem percentage_of_original_price :
  (current_price / original_price) * 100 = 80 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_original_price_l181_18101


namespace NUMINAMATH_GPT_arithmetic_progression_probability_l181_18110

def is_arithmetic_progression (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

noncomputable def probability_arithmetic_progression_diff_two : ℚ :=
  have total_outcomes : ℚ := 6 * 6 * 6
  have favorable_outcomes : ℚ := 12
  favorable_outcomes / total_outcomes

theorem arithmetic_progression_probability (d : ℕ) (h : d = 2) :
  probability_arithmetic_progression_diff_two = 1 / 18 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_progression_probability_l181_18110


namespace NUMINAMATH_GPT_data_set_average_l181_18163

theorem data_set_average (a : ℝ) (h : (2 + 3 + 3 + 4 + a) / 5 = 3) : a = 3 := 
sorry

end NUMINAMATH_GPT_data_set_average_l181_18163


namespace NUMINAMATH_GPT_relationship_among_abc_l181_18190

noncomputable def a := Real.log 2 / Real.log (1/5)
noncomputable def b := 3 ^ (3/5)
noncomputable def c := 4 ^ (1/5)

theorem relationship_among_abc : a < c ∧ c < b := 
by
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l181_18190


namespace NUMINAMATH_GPT_brianne_yard_length_l181_18107

theorem brianne_yard_length 
  (derrick_yard_length : ℝ)
  (h₁ : derrick_yard_length = 10)
  (alex_yard_length : ℝ)
  (h₂ : alex_yard_length = derrick_yard_length / 2)
  (brianne_yard_length : ℝ)
  (h₃ : brianne_yard_length = 6 * alex_yard_length) :
  brianne_yard_length = 30 :=
by sorry

end NUMINAMATH_GPT_brianne_yard_length_l181_18107


namespace NUMINAMATH_GPT_coefficients_sum_eq_four_l181_18123

noncomputable def simplified_coefficients_sum (y : ℚ → ℚ) : ℚ :=
  let A := 1
  let B := 3
  let C := 2
  let D := -2
  A + B + C + D

theorem coefficients_sum_eq_four : simplified_coefficients_sum (λ x => 
  (x^3 + 5*x^2 + 8*x + 4) / (x + 2)) = 4 := by
  sorry

end NUMINAMATH_GPT_coefficients_sum_eq_four_l181_18123


namespace NUMINAMATH_GPT_sin_neg_045_unique_solution_l181_18133

theorem sin_neg_045_unique_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 180) (h: ℝ) :
  (h = Real.sin x → h = -0.45) → 
  ∃! x, 0 ≤ x ∧ x < 180 ∧ Real.sin x = -0.45 :=
by sorry

end NUMINAMATH_GPT_sin_neg_045_unique_solution_l181_18133


namespace NUMINAMATH_GPT_gcd_2_pow_2018_2_pow_2029_l181_18178

theorem gcd_2_pow_2018_2_pow_2029 : Nat.gcd (2^2018 - 1) (2^2029 - 1) = 2047 :=
by
  sorry

end NUMINAMATH_GPT_gcd_2_pow_2018_2_pow_2029_l181_18178


namespace NUMINAMATH_GPT_sum_first_4_terms_l181_18102

-- Define the sequence and its properties
def a (n : ℕ) : ℝ := sorry   -- The actual definition will be derived based on n, a_1, and q
def S (n : ℕ) : ℝ := sorry   -- The sum of the first n terms, also will be derived

-- Define the initial sequence properties based on the given conditions
axiom h1 : 0 < a 1  -- The sequence is positive
axiom h2 : a 4 * a 6 = 1 / 4
axiom h3 : a 7 = 1 / 8

-- The goal is to prove the sum of the first 4 terms equals 15
theorem sum_first_4_terms : S 4 = 15 := by
  sorry

end NUMINAMATH_GPT_sum_first_4_terms_l181_18102


namespace NUMINAMATH_GPT_sum_f_positive_l181_18105

variable (a b c : ℝ)

def f (x : ℝ) := x^3 + x

theorem sum_f_positive (h1 : a + b > 0) (h2 : a + c > 0) (h3 : b + c > 0) :
  f a + f b + f c > 0 :=
sorry

end NUMINAMATH_GPT_sum_f_positive_l181_18105


namespace NUMINAMATH_GPT_cannot_lie_on_line_l181_18166

open Real

theorem cannot_lie_on_line (m b : ℝ) (h1 : m * b > 0) (h2 : b > 0) :
  (0, -2023) ≠ (0, b) :=
by
  sorry

end NUMINAMATH_GPT_cannot_lie_on_line_l181_18166


namespace NUMINAMATH_GPT_relationship_between_m_and_n_l181_18108

theorem relationship_between_m_and_n
  (b m n : ℝ)
  (h₁ : m = 2 * (-1 / 2) + b)
  (h₂ : n = 2 * 2 + b) :
  m < n :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_m_and_n_l181_18108


namespace NUMINAMATH_GPT_cost_of_first_house_l181_18124

theorem cost_of_first_house (C : ℝ) (h₀ : 2 * C + C = 600000) : C = 200000 := by
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_cost_of_first_house_l181_18124


namespace NUMINAMATH_GPT_find_f_neg_one_l181_18185

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x - 3*x + k else -(2^(-x) - 3*(-x) + k)

theorem find_f_neg_one (k : ℝ) (h : ∀ (x : ℝ), f k (-x) = -f k x) : f k (-1) = 2 :=
sorry

end NUMINAMATH_GPT_find_f_neg_one_l181_18185


namespace NUMINAMATH_GPT_total_amount_invested_l181_18109

theorem total_amount_invested (x y : ℝ) (hx : 0.06 * x = 0.05 * y + 160) (hy : 0.05 * y = 6000) :
  x + y = 222666.67 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_invested_l181_18109


namespace NUMINAMATH_GPT_initial_participants_l181_18179

theorem initial_participants (p : ℕ) (h1 : 0.6 * p = 0.6 * (p : ℝ)) (h2 : ∀ (n : ℕ), n = 4 * m → 30 = (2 / 5) * n * (1 / 4)) :
  p = 300 :=
by sorry

end NUMINAMATH_GPT_initial_participants_l181_18179


namespace NUMINAMATH_GPT_find_period_l181_18141

variable (x : ℕ)
variable (theo_daily : ℕ := 8)
variable (mason_daily : ℕ := 7)
variable (roxy_daily : ℕ := 9)
variable (total_water : ℕ := 168)

theorem find_period (h : (theo_daily + mason_daily + roxy_daily) * x = total_water) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_period_l181_18141


namespace NUMINAMATH_GPT_common_ratio_geometric_progression_l181_18191

theorem common_ratio_geometric_progression (r : ℝ) (a : ℝ) (h : a > 0) (h_r : r > 0) (h_eq : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) : r^3 + r^2 + r - 1 = 0 := 
by sorry

end NUMINAMATH_GPT_common_ratio_geometric_progression_l181_18191


namespace NUMINAMATH_GPT_move_point_right_l181_18147

theorem move_point_right (A B : ℤ) (hA : A = -3) (hAB : B = A + 4) : B = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_move_point_right_l181_18147


namespace NUMINAMATH_GPT_problem1_l181_18155

theorem problem1 : (- (1 / 12) - (1 / 16) + (3 / 4) - (1 / 6)) * (-48) = -21 :=
by
  sorry

end NUMINAMATH_GPT_problem1_l181_18155


namespace NUMINAMATH_GPT_average_speed_of_bus_trip_l181_18182

theorem average_speed_of_bus_trip
  (v : ℝ)
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_increment : ℝ)
  (original_time : ℝ := distance / v)
  (faster_time : ℝ := distance / (v + speed_increment))
  (h1 : distance = 360)
  (h2 : time_difference = 1)
  (h3 : speed_increment = 5)
  (h4 : original_time - time_difference = faster_time) :
  v = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_bus_trip_l181_18182


namespace NUMINAMATH_GPT_find_line_equation_l181_18100

-- Definition of a line passing through a point
def passes_through (l : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := l p.1 p.2

-- Definition of intercepts being opposite
def opposite_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ l a 0 ∧ l 0 (-a)

-- The line passing through the point (7, 1)
def line_exists (l : ℝ → ℝ → Prop) : Prop :=
  passes_through l (7, 1) ∧ opposite_intercepts l

-- Main theorem to prove the equation of the line
theorem find_line_equation (l : ℝ → ℝ → Prop) :
  line_exists l ↔ (∀ x y, l x y ↔ x - 7 * y = 0) ∨ (∀ x y, l x y ↔ x - y - 6 = 0) :=
sorry

end NUMINAMATH_GPT_find_line_equation_l181_18100


namespace NUMINAMATH_GPT_problem_solution_l181_18193

variable (x y : ℝ)

-- Conditions
axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : (4 * x - 3 * y) / (x + 4 * y) = 3

-- Goal
theorem problem_solution : (x - 4 * y) / (4 * x + 3 * y) = 11 / 63 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l181_18193


namespace NUMINAMATH_GPT_evaluate_expression_l181_18188

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l181_18188


namespace NUMINAMATH_GPT_find_x_satisfying_condition_l181_18136

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem find_x_satisfying_condition : ∀ x : ℝ, (A x ∪ B x = A x) ↔ (x = 2 ∨ x = -2 ∨ x = 0) := by
  sorry

end NUMINAMATH_GPT_find_x_satisfying_condition_l181_18136


namespace NUMINAMATH_GPT_solve_log_eq_l181_18139

theorem solve_log_eq : ∀ x : ℝ, (2 : ℝ) ^ (Real.log x / Real.log 3) = (1 / 4 : ℝ) → x = 1 / 9 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_log_eq_l181_18139


namespace NUMINAMATH_GPT_student_score_max_marks_l181_18195

theorem student_score_max_marks (M : ℝ)
  (pass_threshold : ℝ := 0.60 * M)
  (student_marks : ℝ := 80)
  (fail_by : ℝ := 40)
  (required_passing_score : ℝ := student_marks + fail_by) :
  pass_threshold = required_passing_score → M = 200 := 
by
  sorry

end NUMINAMATH_GPT_student_score_max_marks_l181_18195


namespace NUMINAMATH_GPT_find_n_for_k_eq_1_l181_18176

theorem find_n_for_k_eq_1 (n : ℤ) (h : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 1)) : n = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_n_for_k_eq_1_l181_18176


namespace NUMINAMATH_GPT_value_of_h_l181_18172

theorem value_of_h (h : ℝ) : (∃ x : ℝ, x^3 + h * x - 14 = 0 ∧ x = 3) → h = -13/3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_h_l181_18172


namespace NUMINAMATH_GPT_ceil_sqrt_225_l181_18146

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end NUMINAMATH_GPT_ceil_sqrt_225_l181_18146


namespace NUMINAMATH_GPT_tickets_spent_on_beanie_l181_18111

-- Define the initial number of tickets Jerry had.
def initial_tickets : ℕ := 4

-- Define the number of tickets Jerry won later.
def won_tickets : ℕ := 47

-- Define the current number of tickets Jerry has.
def current_tickets : ℕ := 49

-- The statement of the problem to prove the tickets spent on the beanie.
theorem tickets_spent_on_beanie :
  initial_tickets + won_tickets - 2 = current_tickets := by
  sorry

end NUMINAMATH_GPT_tickets_spent_on_beanie_l181_18111


namespace NUMINAMATH_GPT_intersection_M_complement_N_l181_18144

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def N : Set ℝ := {x | ∃ y : ℝ, y = 3*x^2 + 1 }

def complement_N : Set ℝ := {x | ¬ ∃ y : ℝ, y = 3*x^2 + 1}

theorem intersection_M_complement_N :
  (M ∩ complement_N) = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_complement_N_l181_18144


namespace NUMINAMATH_GPT_polynomial_value_l181_18152

def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_value (a b c d : ℝ) 
  (h1 : P 1 a b c d = 1993) 
  (h2 : P 2 a b c d = 3986) 
  (h3 : P 3 a b c d = 5979) :
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_value_l181_18152


namespace NUMINAMATH_GPT_minimize_quadratic_function_l181_18122

theorem minimize_quadratic_function :
  ∃ x : ℝ, ∀ y : ℝ, 3 * y^2 - 18 * y + 7 ≥ 3 * x^2 - 18 * x + 7 := 
by
  use 3
  intros y
  sorry

end NUMINAMATH_GPT_minimize_quadratic_function_l181_18122


namespace NUMINAMATH_GPT_count_less_than_threshold_is_zero_l181_18158

def numbers := [0.8, 0.5, 0.9]
def threshold := 0.4

theorem count_less_than_threshold_is_zero :
  (numbers.filter (λ x => x < threshold)).length = 0 :=
by
  sorry

end NUMINAMATH_GPT_count_less_than_threshold_is_zero_l181_18158


namespace NUMINAMATH_GPT_line_parabola_midpoint_l181_18174

theorem line_parabola_midpoint (a b : ℝ) 
  (r s : ℝ) 
  (intersects_parabola : ∀ x, x = r ∨ x = s → ax + b = x^2)
  (midpoint_cond : (r + s) / 2 = 5 ∧ (r^2 + s^2) / 2 = 101) :
  a + b = -41 :=
sorry

end NUMINAMATH_GPT_line_parabola_midpoint_l181_18174


namespace NUMINAMATH_GPT_number_of_albums_l181_18140

-- Definitions for the given conditions
def pictures_from_phone : ℕ := 7
def pictures_from_camera : ℕ := 13
def pictures_per_album : ℕ := 4

-- We compute the total number of pictures
def total_pictures : ℕ := pictures_from_phone + pictures_from_camera

-- Statement: Prove the number of albums is 5
theorem number_of_albums :
  total_pictures / pictures_per_album = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_albums_l181_18140


namespace NUMINAMATH_GPT_not_divisible_by_15_l181_18116

theorem not_divisible_by_15 (a : ℤ) : ¬ (15 ∣ (a^2 + a + 2)) :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_15_l181_18116


namespace NUMINAMATH_GPT_find_x_l181_18131

theorem find_x (x : ℤ) (h : (2 * x + 7) / 5 = 22) : x = 103 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l181_18131


namespace NUMINAMATH_GPT_manager_salary_l181_18181

theorem manager_salary (avg_salary_50 : ℕ) (num_employees : ℕ) (increment_new_avg : ℕ)
  (new_avg_salary : ℕ) (total_old_salary : ℕ) (total_new_salary : ℕ) (M : ℕ) :
  avg_salary_50 = 2000 →
  num_employees = 50 →
  increment_new_avg = 250 →
  new_avg_salary = avg_salary_50 + increment_new_avg →
  total_old_salary = num_employees * avg_salary_50 →
  total_new_salary = (num_employees + 1) * new_avg_salary →
  M = total_new_salary - total_old_salary →
  M = 14750 :=
by {
  sorry
}

end NUMINAMATH_GPT_manager_salary_l181_18181


namespace NUMINAMATH_GPT_counterexample_conjecture_l181_18130

theorem counterexample_conjecture 
    (odd_gt_5 : ℕ → Prop) 
    (is_prime : ℕ → Prop) 
    (conjecture : ∀ n, odd_gt_5 n → ∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) : 
    ∃ n, odd_gt_5 n ∧ ¬ (∃ p1 p2 p3, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ n = p1 + p2 + p3) :=
sorry

end NUMINAMATH_GPT_counterexample_conjecture_l181_18130


namespace NUMINAMATH_GPT_triangle_area_l181_18143

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem triangle_area :
  area_of_triangle 0 0 0 6 8 0 = 24 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l181_18143


namespace NUMINAMATH_GPT_function_machine_output_l181_18112

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 25 then step1 - 7 else step1 + 10
  step2

theorem function_machine_output : function_machine 15 = 38 :=
by
  sorry

end NUMINAMATH_GPT_function_machine_output_l181_18112


namespace NUMINAMATH_GPT_value_of_fraction_l181_18170

theorem value_of_fraction (x y : ℝ) (h : 1 / x - 1 / y = 2) : (x + x * y - y) / (x - x * y - y) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l181_18170


namespace NUMINAMATH_GPT_number_of_triangles_l181_18153

theorem number_of_triangles (x : ℕ) (h₁ : 2 + x > 6) (h₂ : 8 > x) : ∃! t, t = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_triangles_l181_18153


namespace NUMINAMATH_GPT_seokjin_paper_count_l181_18159

theorem seokjin_paper_count :
  ∀ (jimin_paper seokjin_paper : ℕ),
  jimin_paper = 41 →
  jimin_paper = seokjin_paper + 1 →
  seokjin_paper = 40 :=
by
  intros jimin_paper seokjin_paper h_jimin h_relation
  sorry

end NUMINAMATH_GPT_seokjin_paper_count_l181_18159


namespace NUMINAMATH_GPT_prob_blue_lower_than_yellow_l181_18198

noncomputable def prob_bin_k (k : ℕ) : ℝ :=
  3^(-k : ℤ)

noncomputable def prob_same_bin : ℝ :=
  ∑' k, 3^(-2*k : ℤ)

theorem prob_blue_lower_than_yellow :
  (1 - prob_same_bin) / 2 = 7 / 16 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_prob_blue_lower_than_yellow_l181_18198


namespace NUMINAMATH_GPT_probability_prime_sum_l181_18128

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def possible_outcomes : ℕ := 48

def prime_sums : Finset ℕ := {2, 3, 5, 7, 11, 13}

def prime_count : ℕ := 19

theorem probability_prime_sum :
  ((prime_count : ℚ) / possible_outcomes) = 19 / 48 := 
by
  sorry

end NUMINAMATH_GPT_probability_prime_sum_l181_18128


namespace NUMINAMATH_GPT_ratio_condition_equivalence_l181_18157

variable (a b c d : ℝ)

theorem ratio_condition_equivalence
  (h : (2 * a + 3 * b) / (b + 2 * c) = (3 * c + 2 * d) / (d + 2 * a)) :
  2 * a = 3 * c ∨ 2 * a + 3 * b + d + 2 * c = 0 :=
by
  sorry

end NUMINAMATH_GPT_ratio_condition_equivalence_l181_18157


namespace NUMINAMATH_GPT_ellipse_hyperbola_same_foci_l181_18126

theorem ellipse_hyperbola_same_foci (k : ℝ) (h1 : k > 0) :
  (∀ (x y : ℝ), (x^2 / 9 + y^2 / k^2 = 1) ↔ (x^2 / k - y^2 / 3 = 1)) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_same_foci_l181_18126


namespace NUMINAMATH_GPT_g_zero_l181_18184

variable (f g h : Polynomial ℤ) -- Assume f, g, h are polynomials over the integers

-- Condition: h(x) = f(x) * g(x)
axiom h_def : h = f * g

-- Condition: The constant term of f(x) is 2
axiom f_const : f.coeff 0 = 2

-- Condition: The constant term of h(x) is -6
axiom h_const : h.coeff 0 = -6

-- Proof statement that g(0) = -3
theorem g_zero : g.coeff 0 = -3 := by
  sorry

end NUMINAMATH_GPT_g_zero_l181_18184


namespace NUMINAMATH_GPT_smallest_k_for_g_l181_18151

theorem smallest_k_for_g (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x + k = -3) ↔ k ≤ -3/4 := sorry

end NUMINAMATH_GPT_smallest_k_for_g_l181_18151


namespace NUMINAMATH_GPT_parabola_point_ordinate_l181_18186

-- The definition of the problem as a Lean 4 statement
theorem parabola_point_ordinate (a : ℝ) (x₀ y₀ : ℝ) 
  (h₀ : 0 < a)
  (h₁ : x₀^2 = (1 / a) * y₀)
  (h₂ : dist (0, 1 / (4 * a)) (0, -1 / (4 * a)) = 1)
  (h₃ : dist (x₀, y₀) (0, 1 / (4 * a)) = 5) :
  y₀ = 9 / 2 := 
sorry

end NUMINAMATH_GPT_parabola_point_ordinate_l181_18186


namespace NUMINAMATH_GPT_reversed_digits_sum_l181_18135

theorem reversed_digits_sum (a b n : ℕ) (x y : ℕ) (ha : a < 10) (hb : b < 10) 
(hx : x = 10 * a + b) (hy : y = 10 * b + a) (hsq : x^2 + y^2 = n^2) : 
  x + y + n = 264 :=
sorry

end NUMINAMATH_GPT_reversed_digits_sum_l181_18135


namespace NUMINAMATH_GPT_find_m_plus_n_l181_18164

-- Define the number of ways Blair and Corey can draw the remaining cards
def num_ways_blair_and_corey_draw : ℕ := Nat.choose 50 2

-- Define the function q(a) as given in the problem
noncomputable def q (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / num_ways_blair_and_corey_draw

-- Define the problem statement to find the minimum value of a for which q(a) >= 1/2
noncomputable def minimum_a : ℤ :=
  if q 7 >= 1/2 then 7 else 36 -- According to the solution, these are the points of interest

-- The final statement to be proved
theorem find_m_plus_n : minimum_a = 7 ∨ minimum_a = 36 :=
  sorry

end NUMINAMATH_GPT_find_m_plus_n_l181_18164


namespace NUMINAMATH_GPT_algebraic_expression_value_l181_18118

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x - 2 = 0) :
  x * (x + 2) + (x + 1)^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l181_18118


namespace NUMINAMATH_GPT_sum_mod_9_l181_18199

theorem sum_mod_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  sorry

end NUMINAMATH_GPT_sum_mod_9_l181_18199


namespace NUMINAMATH_GPT_graveling_cost_is_969_l181_18114

-- Definitions for lawn dimensions
def lawn_length : ℝ := 75
def lawn_breadth : ℝ := 45

-- Definitions for road widths and costs
def road1_width : ℝ := 6
def road1_cost_per_sq_meter : ℝ := 0.90

def road2_width : ℝ := 5
def road2_cost_per_sq_meter : ℝ := 0.85

def road3_width : ℝ := 4
def road3_cost_per_sq_meter : ℝ := 0.80

def road4_width : ℝ := 3
def road4_cost_per_sq_meter : ℝ := 0.75

-- Calculate the area of each road
def road1_area : ℝ := road1_width * lawn_length
def road2_area : ℝ := road2_width * lawn_length
def road3_area : ℝ := road3_width * lawn_breadth
def road4_area : ℝ := road4_width * lawn_breadth

-- Calculate the cost of graveling each road
def road1_graveling_cost : ℝ := road1_area * road1_cost_per_sq_meter
def road2_graveling_cost : ℝ := road2_area * road2_cost_per_sq_meter
def road3_graveling_cost : ℝ := road3_area * road3_cost_per_sq_meter
def road4_graveling_cost : ℝ := road4_area * road4_cost_per_sq_meter

-- Calculate the total cost
def total_graveling_cost : ℝ := 
  road1_graveling_cost + road2_graveling_cost + road3_graveling_cost + road4_graveling_cost

-- Statement to be proved
theorem graveling_cost_is_969 : total_graveling_cost = 969 := by
  sorry

end NUMINAMATH_GPT_graveling_cost_is_969_l181_18114
