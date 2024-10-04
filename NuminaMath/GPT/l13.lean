import Mathlib

namespace sum_max_min_value_f_l13_13892

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + x) / (x ^ 2 + 1)

theorem sum_max_min_value_f : 
  let M := (⨆ x : ℝ, f x)
  let m := (⨅ x : ℝ, f x)
  M + m = 2 :=
by
-- Proof to be filled in
  sorry

end sum_max_min_value_f_l13_13892


namespace problem1_problem2_l13_13749

-- Definitions based on conditions in the problem
def seq_sum (a : ℕ) (n : ℕ) : ℕ := a * 2^n - 1
def a1 (a : ℕ) : ℕ := seq_sum a 1
def a4 (a : ℕ) : ℕ := seq_sum a 4 - seq_sum a 3

-- Problem statement 1
theorem problem1 (a : ℕ) (h : a = 3) : a1 a = 5 ∧ a4 a = 24 := by 
  sorry

-- Geometric sequence conditions
def is_geometric (a_n : ℕ → ℕ) : Prop :=
  ∃ q ≠ 1, ∀ n, a_n (n + 1) = q * a_n n

-- Definitions for the geometric sequence part
def a_n (a : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then 2 * a - 1
  else if n = 2 then 2 * a
  else if n = 3 then 4 * a
  else 0 -- Simplifying for the first few terms only

-- Problem statement 2
theorem problem2 : (∃ a : ℕ, is_geometric (a_n a)) → ∃ a : ℕ, a = 1 := by
  sorry

end problem1_problem2_l13_13749


namespace find_range_of_m_l13_13589

def has_two_distinct_negative_real_roots (m : ℝ) : Prop := 
  let Δ := m^2 - 4
  Δ > 0 ∧ -m > 0

def inequality_holds_for_all_real (m : ℝ) : Prop :=
  let Δ := (4 * (m - 2))^2 - 16
  Δ < 0

def problem_statement (m : ℝ) : Prop :=
  (has_two_distinct_negative_real_roots m ∨ inequality_holds_for_all_real m) ∧ 
  ¬(has_two_distinct_negative_real_roots m ∧ inequality_holds_for_all_real m)

theorem find_range_of_m (m : ℝ) : problem_statement m ↔ ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m)) :=
by
  sorry

end find_range_of_m_l13_13589


namespace angle_terminal_side_eq_l13_13018

theorem angle_terminal_side_eq (α : ℝ) : 
  (α = -4 * Real.pi / 3 + 2 * Real.pi) → (0 ≤ α ∧ α < 2 * Real.pi) → α = 2 * Real.pi / 3 := 
by 
  sorry

end angle_terminal_side_eq_l13_13018


namespace find_f_expression_l13_13765

theorem find_f_expression (f : ℝ → ℝ) (x : ℝ) (h : f (Real.log x) = 3 * x + 4) : 
  f x = 3 * Real.exp x + 4 := 
by
  sorry

end find_f_expression_l13_13765


namespace marching_band_total_weight_l13_13338

noncomputable def total_weight : ℕ :=
  let trumpet_weight := 5
  let clarinet_weight := 5
  let trombone_weight := 10
  let tuba_weight := 20
  let drum_weight := 15
  let trumpets := 6
  let clarinets := 9
  let trombones := 8
  let tubas := 3
  let drummers := 2
  (trumpets + clarinets) * trumpet_weight + trombones * trombone_weight + tubas * tuba_weight + drummers * drum_weight

theorem marching_band_total_weight : total_weight = 245 := by
  sorry

end marching_band_total_weight_l13_13338


namespace series_sum_eq_half_l13_13124

theorem series_sum_eq_half :
  ∑' (n : ℕ), 2^n / (3^(2^n) + 1) = 1 / 2 :=
sorry

end series_sum_eq_half_l13_13124


namespace inequality_proof_l13_13368

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (x^2 / (y - 1) + y^2 / (x - 1) ≥ 8) :=
  sorry

end inequality_proof_l13_13368


namespace jessica_withdrawal_l13_13780

/-- Jessica withdrew some money from her bank account, causing her account balance to decrease by 2/5.
    She then deposited an amount equal to 1/4 of the remaining balance. The final balance in her bank account is $750.
    Prove that Jessica initially withdrew $400. -/
theorem jessica_withdrawal (X W : ℝ) 
  (initial_eq : W = (2 / 5) * X)
  (remaining_eq : X * (3 / 5) + (1 / 4) * (X * (3 / 5)) = 750) :
  W = 400 := 
sorry

end jessica_withdrawal_l13_13780


namespace slope_of_line_l13_13210

noncomputable def line_eq (x y : ℝ) := x / 4 + y / 5 = 1

theorem slope_of_line : ∀ (x y : ℝ), line_eq x y → (∃ m b : ℝ, y = m * x + b ∧ m = -5 / 4) :=
sorry

end slope_of_line_l13_13210


namespace smallest_three_digit_n_l13_13859

theorem smallest_three_digit_n (n : ℕ) (h_pos : 100 ≤ n) (h_below : n ≤ 999) 
  (cond1 : n % 9 = 2) (cond2 : n % 6 = 4) : n = 118 :=
by {
  sorry
}

end smallest_three_digit_n_l13_13859


namespace find_number_of_boys_l13_13493

noncomputable def number_of_boys (B G : ℕ) : Prop :=
  (B : ℚ) / (G : ℚ) = 7.5 / 15.4 ∧ G = B + 174

theorem find_number_of_boys : ∃ B G : ℕ, number_of_boys B G ∧ B = 165 := 
by 
  sorry

end find_number_of_boys_l13_13493


namespace find_k_l13_13101

noncomputable def parabola_k : ℝ := 4

theorem find_k (k : ℝ) (h1 : ∀ x, y = k^2 - x^2) (h2 : k > 0)
    (h3 : ∀ A D : (ℝ × ℝ), A = (-k, 0) ∧ D = (k, 0))
    (h4 : ∀ V : (ℝ × ℝ), V = (0, k^2))
    (h5 : 2 * (2 * k + k^2) = 48) : k = 4 :=
  sorry

end find_k_l13_13101


namespace min_workers_to_profit_l13_13250

/-- Definitions of constants used in the problem. --/
def daily_maintenance_cost : ℕ := 500
def wage_per_hour : ℕ := 20
def widgets_per_hour_per_worker : ℕ := 5
def sell_price_per_widget : ℕ := 350 / 100 -- since the input is 3.50
def workday_hours : ℕ := 8

/-- Profit condition: the revenue should be greater than the cost. 
    The problem specifies that the number of workers must be at least 26 to make a profit. --/

theorem min_workers_to_profit (n : ℕ) :
  (widgets_per_hour_per_worker * workday_hours * sell_price_per_widget * n > daily_maintenance_cost + (workday_hours * wage_per_hour * n)) → n ≥ 26 :=
sorry


end min_workers_to_profit_l13_13250


namespace base7_to_base10_l13_13373

theorem base7_to_base10 (a b : ℕ) (h : 235 = 49 * 2 + 7 * 3 + 5) (h_ab : 100 + 10 * a + b = 124) : 
  (a + b) / 7 = 6 / 7 :=
by
  sorry

end base7_to_base10_l13_13373


namespace number_of_perfect_apples_l13_13618

theorem number_of_perfect_apples (total_apples : ℕ) (too_small_ratio : ℚ) (not_ripe_ratio : ℚ)
  (h_total : total_apples = 30)
  (h_too_small_ratio : too_small_ratio = 1 / 6)
  (h_not_ripe_ratio : not_ripe_ratio = 1 / 3) :
  (total_apples - (too_small_ratio * total_apples).natAbs - (not_ripe_ratio * total_apples).natAbs) = 15 := by
  sorry

end number_of_perfect_apples_l13_13618


namespace trivia_team_average_points_l13_13110

noncomputable def average_points_per_member (total_members didn't_show_up total_points : ℝ) : ℝ :=
  total_points / (total_members - didn't_show_up)

@[simp]
theorem trivia_team_average_points :
  let total_members := 8.0
  let didn't_show_up := 3.5
  let total_points := 12.5
  ∃ avg_points, avg_points = 2.78 ∧ avg_points = average_points_per_member total_members didn't_show_up total_points :=
by
  sorry

end trivia_team_average_points_l13_13110


namespace min_value_expression_l13_13396

theorem min_value_expression : ∃ x y : ℝ, (xy-2)^2 + (x^2 + y^2) = 4 :=
by
  sorry

end min_value_expression_l13_13396


namespace compound_interest_rate_l13_13981

theorem compound_interest_rate
  (A P : ℝ) (t n : ℝ)
  (HA : A = 1348.32)
  (HP : P = 1200)
  (Ht : t = 2)
  (Hn : n = 1) :
  ∃ r : ℝ, 0 ≤ r ∧ ((A / P) ^ (1 / (n * t)) - 1) = r ∧ r = 0.06 := 
sorry

end compound_interest_rate_l13_13981


namespace proof_by_contradiction_conditions_l13_13182

theorem proof_by_contradiction_conditions :
  ∀ (P Q : Prop),
    (∃ R : Prop, (R = ¬Q) ∧ (P → R) ∧ (R → P) ∧ (∀ T : Prop, (T = Q) → false)) →
    (∃ S : Prop, (S = ¬Q) ∧ P ∧ (∃ U : Prop, U) ∧ ¬Q) :=
by
  sorry

end proof_by_contradiction_conditions_l13_13182


namespace cos_double_angle_zero_l13_13486

variable (θ : ℝ)

-- Conditions
def tan_eq_one : Prop := Real.tan θ = 1

-- Objective
theorem cos_double_angle_zero (h : tan_eq_one θ) : Real.cos (2 * θ) = 0 :=
sorry

end cos_double_angle_zero_l13_13486


namespace find_missing_number_l13_13137

theorem find_missing_number (x : ℝ) (h : 0.00375 * x = 153.75) : x = 41000 :=
sorry

end find_missing_number_l13_13137


namespace quadratic_two_distinct_real_roots_l13_13931

theorem quadratic_two_distinct_real_roots (m : ℝ) : 
  ∀ x : ℝ, x^2 + m * x - 2 = 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0 :=
by
  sorry

end quadratic_two_distinct_real_roots_l13_13931


namespace gasoline_price_increase_percent_l13_13661

theorem gasoline_price_increase_percent {P Q : ℝ}
  (h₁ : P > 0)
  (h₂: Q > 0)
  (x : ℝ)
  (condition : P * Q * 1.08 = P * (1 + x/100) * Q * 0.90) :
  x = 20 :=
by {
  sorry
}

end gasoline_price_increase_percent_l13_13661


namespace scientific_notation_gdp_2022_l13_13716

def gdp_2022_fujian : ℝ := 53100 * 10^9

theorem scientific_notation_gdp_2022 : 
  (53100 * 10^9) = 5.31 * 10^12 :=
by
  -- The proof is based on the understanding that 53100 * 10^9 can be rewritten as 5.31 * 10^12
  -- However, this proof is currently omitted with a placeholder.
  sorry

end scientific_notation_gdp_2022_l13_13716


namespace inequality_a3_b3_c3_l13_13040

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3 * a * b * c > a * b * (a + b) + b * c * (b + c) + a * c * (a + c) :=
by
  sorry

end inequality_a3_b3_c3_l13_13040


namespace solution_set_inequality_l13_13681

theorem solution_set_inequality (x : ℝ) : (1 < x ∧ x < 3) ↔ (x^2 - 4*x + 3 < 0) :=
by sorry

end solution_set_inequality_l13_13681


namespace smaller_angle_3_40_pm_l13_13063

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l13_13063


namespace num_boys_in_circle_l13_13852

theorem num_boys_in_circle (n : ℕ) 
  (h : ∃ k, n = 2 * k ∧ k = 40 - 10) : n = 60 :=
by
  sorry

end num_boys_in_circle_l13_13852


namespace tan_product_l13_13565

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end tan_product_l13_13565


namespace smaller_angle_between_clock_hands_3_40_pm_l13_13076

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l13_13076


namespace boats_seating_problem_l13_13428

theorem boats_seating_problem 
  (total_boats : ℕ) (total_people : ℕ) 
  (big_boat_seats : ℕ) (small_boat_seats : ℕ) 
  (b s : ℕ) 
  (h1 : total_boats = 12) 
  (h2 : total_people = 58) 
  (h3 : big_boat_seats = 6) 
  (h4 : small_boat_seats = 4) 
  (h5 : b + s = 12) 
  (h6 : b * 6 + s * 4 = 58) 
  : b = 5 ∧ s = 7 :=
sorry

end boats_seating_problem_l13_13428


namespace problem_l13_13477

theorem problem
  (a b c d e : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (- (a * b))^2009 - (c + d)^2010 - e^2011 = 0 := 
by
  sorry

end problem_l13_13477


namespace cube_sum_eq_one_l13_13804

theorem cube_sum_eq_one (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 2) (h3 : abc = 1) : a^3 + b^3 + c^3 = 1 :=
sorry

end cube_sum_eq_one_l13_13804


namespace triangle_side_lengths_l13_13326

-- Define the variables a, b, and c
variables {a b c : ℝ}

-- Assume that a, b, and c are the lengths of the sides of a triangle
-- and the given equation holds
theorem triangle_side_lengths (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
    (h_eq : a^2 + 4*a*c + 3*c^2 - 3*a*b - 7*b*c + 2*b^2 = 0) : 
    a + c - 2*b = 0 :=
by
  sorry

end triangle_side_lengths_l13_13326


namespace solution_points_satisfy_equation_l13_13408

theorem solution_points_satisfy_equation (x y : ℝ) :
  x^2 * (y + y^2) = y^3 + x^4 → (y = x ∨ y = -x ∨ y = x^2) := sorry

end solution_points_satisfy_equation_l13_13408


namespace find_values_general_formula_l13_13585

variable (a_n S_n : ℕ → ℝ)

-- Conditions
axiom sum_sequence (n : ℕ) (hn : n > 0) :  S_n n = (1 / 3) * (a_n n - 1)

-- Questions
theorem find_values :
  (a_n 1 = 2) ∧ (a_n 2 = 5) ∧ (a_n 3 = 8) := sorry

theorem general_formula (n : ℕ) :
  n > 0 → a_n n = n + 1 := sorry

end find_values_general_formula_l13_13585


namespace value_of_sum_l13_13161

theorem value_of_sum (a b c : ℚ) (h1 : 2 * a + 3 * b + c = 27) (h2 : 4 * a + 6 * b + 5 * c = 71) :
  a + b + c = 115 / 9 :=
sorry

end value_of_sum_l13_13161


namespace team_total_mistakes_l13_13174

theorem team_total_mistakes (total_questions : ℕ) (riley_mistakes : ℕ) (ofelia_correction: (ℕ → ℕ) ) : total_questions = 35 → riley_mistakes = 3 → (∀ riley_correct_answers, riley_correct_answers = total_questions - riley_mistakes → ofelia_correction riley_correct_answers = (riley_correct_answers / 2) + 5) → (riley_mistakes + (total_questions - (ofelia_correction (total_questions - riley_mistakes)))) = 17 :=
by
  intros h1 h2 h3
  sorry

end team_total_mistakes_l13_13174


namespace roots_fourth_pow_sum_l13_13193

theorem roots_fourth_pow_sum :
  (∃ p q r : ℂ, (∀ z, (z = p ∨ z = q ∨ z = r) ↔ z^3 - z^2 + 2*z - 3 = 0) ∧ p^4 + q^4 + r^4 = 13) := by
sorry

end roots_fourth_pow_sum_l13_13193


namespace equation_for_pears_l13_13380

-- Define the conditions
def pearDist1 (x : ℕ) : ℕ := 4 * x + 12
def pearDist2 (x : ℕ) : ℕ := 6 * x

-- State the theorem to be proved
theorem equation_for_pears (x : ℕ) : pearDist1 x = pearDist2 x :=
by
  sorry

end equation_for_pears_l13_13380


namespace find_f_10_l13_13262

def f (x : Int) : Int := sorry

axiom condition_1 : f 1 + 1 > 0
axiom condition_2 : ∀ x y : Int, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom condition_3 : ∀ x : Int, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := by
  sorry

end find_f_10_l13_13262


namespace simplify_and_evaluate_l13_13649

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : 
  ( ( (x^2 - 4 * x + 4) / (x^2 - 4) ) / ( (x-2) / (x^2 + 2*x) ) ) + 3 = 6 :=
by
  sorry

end simplify_and_evaluate_l13_13649


namespace smallest_N_l13_13541

theorem smallest_N (N : ℕ) (h : 7 * N = 999999) : N = 142857 :=
sorry

end smallest_N_l13_13541


namespace roots_of_equation_l13_13862

theorem roots_of_equation :
  (∃ x, (18 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = 3 ∨ x = -4.5)) :=
by
  sorry

end roots_of_equation_l13_13862


namespace clock_angle_3_40_l13_13080

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l13_13080


namespace reciprocals_expression_value_l13_13003

theorem reciprocals_expression_value (a b : ℝ) (h : a * b = 1) : a^2 * b - (a - 2023) = 2023 := 
by 
  sorry

end reciprocals_expression_value_l13_13003


namespace vertex_angle_isosceles_triangle_l13_13498

theorem vertex_angle_isosceles_triangle (α : ℝ) (β : ℝ) (sum_of_angles : α + α + β = 180) (base_angle : α = 50) :
  β = 80 :=
by
  sorry

end vertex_angle_isosceles_triangle_l13_13498


namespace sum_of_distinct_integers_eq_zero_l13_13764

theorem sum_of_distinct_integers_eq_zero 
  (a b c d : ℤ) 
  (distinct : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  (prod_eq_25 : a * b * c * d = 25) : a + b + c + d = 0 := by
  sorry

end sum_of_distinct_integers_eq_zero_l13_13764


namespace courtyard_length_eq_40_l13_13324

/-- Defining the dimensions of a paving stone -/
def stone_length : ℝ := 4
def stone_width : ℝ := 2

/-- Defining the width of the courtyard -/
def courtyard_width : ℝ := 20

/-- Number of paving stones used -/
def num_stones : ℝ := 100

/-- Area covered by one paving stone -/
def stone_area : ℝ := stone_length * stone_width

/-- Total area covered by the paving stones -/
def total_area : ℝ := num_stones * stone_area

/-- The main statement to be proved -/
theorem courtyard_length_eq_40 (h1 : total_area = num_stones * stone_area)
(h2 : total_area = 800)
(h3 : courtyard_width = 20) : total_area / courtyard_width = 40 :=
by sorry

end courtyard_length_eq_40_l13_13324


namespace prove_y_l13_13457

theorem prove_y (x y : ℝ) (h1 : 3 * x^2 - 4 * x + 7 * y + 3 = 0) (h2 : 3 * x - 5 * y + 6 = 0) :
  25 * y^2 - 39 * y + 69 = 0 := sorry

end prove_y_l13_13457


namespace area_of_triangle_l13_13480

theorem area_of_triangle (s1 s2 s3 : ℕ) (h1 : s1^2 = 36) (h2 : s2^2 = 64) (h3 : s3^2 = 100) (h4 : s1^2 + s2^2 = s3^2) :
  (1 / 2 : ℚ) * s1 * s2 = 24 := by
  sorry

end area_of_triangle_l13_13480


namespace factor_polynomial_l13_13869

theorem factor_polynomial :
  ∀ (x : ℤ), 9 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 5 * x^2 = (x^2 + 4) * (9 * x^2 + 22 * x + 342) :=
by
  intro x
  sorry

end factor_polynomial_l13_13869


namespace total_cost_train_and_bus_l13_13553

noncomputable def trainFare := 3.75 + 2.35
noncomputable def busFare := 3.75
noncomputable def totalFare := trainFare + busFare

theorem total_cost_train_and_bus : totalFare = 9.85 :=
by
  -- We'll need a proof here if required.
  sorry

end total_cost_train_and_bus_l13_13553


namespace probability_B_in_A_is_17_over_24_l13_13023

open Set

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs p.1 + abs p.2 <= 2}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p ∈ set_A ∧ p.2 <= p.1 ^ 2}

noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry -- Assume we have means to compute the area of a set

theorem probability_B_in_A_is_17_over_24 :
  (area set_B / area set_A) = 17 / 24 :=
sorry

end probability_B_in_A_is_17_over_24_l13_13023


namespace total_snowfall_l13_13172

theorem total_snowfall (morning afternoon : ℝ) (h1 : morning = 0.125) (h2 : afternoon = 0.5) :
  morning + afternoon = 0.625 := by
  sorry

end total_snowfall_l13_13172


namespace justine_more_than_bailey_l13_13904

-- Definitions from conditions
def J : ℕ := 22 -- Justine's initial rubber bands
def B : ℕ := 12 -- Bailey's initial rubber bands

-- Theorem to prove
theorem justine_more_than_bailey : J - B = 10 := by
  -- Proof will be done here
  sorry

end justine_more_than_bailey_l13_13904


namespace tangent_line_touching_circle_l13_13155

theorem tangent_line_touching_circle (a : ℝ) : 
  (∃ (x y : ℝ), 5 * x + 12 * y + a = 0 ∧ (x - 1)^2 + y^2 = 1) → 
  (a = 8 ∨ a = -18) :=
by
  sorry

end tangent_line_touching_circle_l13_13155


namespace log_abs_monotone_decreasing_l13_13011

open Real

theorem log_abs_monotone_decreasing {a : ℝ} (h : ∀ x y, 0 < x ∧ x < y ∧ y ≤ a → |log x| ≥ |log y|) : 0 < a ∧ a ≤ 1 :=
by
  sorry

end log_abs_monotone_decreasing_l13_13011


namespace man_speed_km_per_hr_l13_13712

noncomputable def train_length : ℝ := 110
noncomputable def train_speed_km_per_hr : ℝ := 82
noncomputable def time_to_pass_man_sec : ℝ := 4.499640028797696

theorem man_speed_km_per_hr :
  ∃ (Vm_km_per_hr : ℝ), Vm_km_per_hr = 6.0084 :=
sorry

end man_speed_km_per_hr_l13_13712


namespace selling_price_is_320_l13_13966

noncomputable def sales_volume (x : ℝ) : ℝ := 8000 / x

def cost_price : ℝ := 180

def desired_profit : ℝ := 3500

def selling_price_for_desired_profit (x : ℝ) : Prop :=
  (x - cost_price) * sales_volume x = desired_profit

/-- The selling price of the small electrical appliance to achieve a daily sales profit 
    of $3500 dollars is $320 dollars. -/
theorem selling_price_is_320 : selling_price_for_desired_profit 320 :=
by
  -- We skip the proof as per instructions
  sorry

end selling_price_is_320_l13_13966


namespace solution_for_g0_l13_13381

variable (g : ℝ → ℝ)

def functional_eq_condition := ∀ x y : ℝ, g (x + y) = g x + g y - 1

theorem solution_for_g0 (h : functional_eq_condition g) : g 0 = 1 :=
by {
  sorry
}

end solution_for_g0_l13_13381


namespace fraction_given_to_friend_l13_13795

theorem fraction_given_to_friend (s u r g k : ℕ) 
  (h1: s = 135) 
  (h2: u = s / 3) 
  (h3: r = s - u) 
  (h4: k = 54) 
  (h5: g = r - k) :
  g / r = 2 / 5 := 
  by
  sorry

end fraction_given_to_friend_l13_13795


namespace percentage_increase_second_year_is_20_l13_13659

noncomputable def find_percentage_increase_second_year : ℕ :=
  let P₀ := 1000
  let P₁ := P₀ + (10 * P₀) / 100
  let Pf := 1320
  let P := (Pf - P₁) * 100 / P₁
  P

theorem percentage_increase_second_year_is_20 :
  find_percentage_increase_second_year = 20 :=
by
  sorry

end percentage_increase_second_year_is_20_l13_13659


namespace intersection_point_of_lines_l13_13993

theorem intersection_point_of_lines :
  ∃ (x y : ℚ), (2 * y = 3 * x - 6) ∧ (x + 5 * y = 10) ∧ (x = 50 / 17) ∧ (y = 24 / 17) :=
by
  sorry

end intersection_point_of_lines_l13_13993


namespace john_marble_choices_l13_13907

open Nat

theorem john_marble_choices :
  (choose 4 2) * (choose 12 3) = 1320 :=
by
  sorry

end john_marble_choices_l13_13907


namespace greatest_three_digit_div_by_3_6_5_l13_13226

theorem greatest_three_digit_div_by_3_6_5 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ n % 3 = 0 ∧ n % 6 = 0 ∧ n % 5 = 0 ∧ ∀ m : ℕ, (m < 1000 ∧ m ≥ 100 ∧ m % 3 = 0 ∧ m % 6 = 0 ∧ m % 5 = 0) → m ≤ n :=
begin
  use 990,
  split; try {linarith},
  split; try {linarith},
  split; try {norm_num},
  split; try {norm_num},
  split; try {norm_num},
  intros m hm,
  rcases hm with ⟨hm1, hm2, hm3, hm4, hm5⟩,
  have h_div : m % 30 = 0,
  {change (30 | m), exact ⟨_, by {field_simp *}⟩},
  rcases h_div with ⟨k, rfl⟩,
  have : k ≤ 33,
  {linarith},
  norm_num at this,
  linarith,
end

end greatest_three_digit_div_by_3_6_5_l13_13226


namespace set_intersection_l13_13030

open Set

variable (U : Set ℝ)
variable (A B : Set ℝ)

def complement (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem set_intersection (hU : U = univ)
                         (hA : A = {x : ℝ | x > 0})
                         (hB : B = {x : ℝ | x > 1}) :
  A ∩ complement B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end set_intersection_l13_13030


namespace find_largest_value_l13_13693

theorem find_largest_value
  (h1: 0 < Real.sin 2) (h2: Real.sin 2 < 1)
  (h3: Real.log 2 / Real.log (1 / 3) < 0)
  (h4: Real.log (1 / 3) / Real.log (1 / 2) > 1) :
  Real.log (1 / 3) / Real.log (1 / 2) > Real.sin 2 ∧ 
  Real.log (1 / 3) / Real.log (1 / 2) > Real.log 2 / Real.log (1 / 3) := by
  sorry

end find_largest_value_l13_13693


namespace findCorrectAnswer_l13_13448

-- Definitions
variable (x : ℕ)
def mistakenCalculation : Prop := 3 * x = 90
def correctAnswer : ℕ := x - 30

-- Theorem statement
theorem findCorrectAnswer (h : mistakenCalculation x) : correctAnswer x = 0 :=
sorry

end findCorrectAnswer_l13_13448


namespace measure_of_angle_B_in_triangle_l13_13171

theorem measure_of_angle_B_in_triangle
  {a b c : ℝ} {A B C : ℝ} 
  (h1 : a * c = b^2 - a^2)
  (h2 : A = Real.pi / 6)
  (h3 : a / Real.sin A = b / Real.sin B) 
  (h4 : b / Real.sin B = c / Real.sin C)
  (h5 : A + B + C = Real.pi) :
  B = Real.pi / 3 :=
by sorry

end measure_of_angle_B_in_triangle_l13_13171


namespace similar_triangles_same_heights_ratio_l13_13152

theorem similar_triangles_same_heights_ratio (h1 h2 : ℝ) 
  (sim_ratio : h1 / h2 = 1 / 4) : h1 / h2 = 1 / 4 :=
by
  sorry

end similar_triangles_same_heights_ratio_l13_13152


namespace num_ways_two_different_colors_l13_13748

theorem num_ways_two_different_colors 
  (red white blue : ℕ) 
  (total_balls : ℕ) 
  (choose : ℕ → ℕ → ℕ) 
  (h_red : red = 2) 
  (h_white : white = 3) 
  (h_blue : blue = 1) 
  (h_total : total_balls = red + white + blue) 
  (h_choose_total : choose total_balls 3 = 20)
  (h_choose_three_diff_colors : 2 * 3 * 1 = 6)
  (h_one_color : 1 = 1) :
  choose total_balls 3 - 6 - 1 = 13 := 
by
  sorry

end num_ways_two_different_colors_l13_13748


namespace total_earnings_correct_l13_13436

-- Define the conditions as initial parameters

def ticket_price : ℕ := 3
def weekday_visitors_per_day : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300

def total_weekday_visitors : ℕ := 5 * weekday_visitors_per_day
def total_weekend_visitors : ℕ := saturday_visitors + sunday_visitors
def total_visitors : ℕ := total_weekday_visitors + total_weekend_visitors

def total_earnings := total_visitors * ticket_price

-- Prove that the total earnings of the amusement park in a week is $3000
theorem total_earnings_correct : total_earnings = 3000 :=
by
  sorry

end total_earnings_correct_l13_13436


namespace B_more_than_C_l13_13431

variables (A B C : ℕ)
noncomputable def total_subscription : ℕ := 50000
noncomputable def total_profit : ℕ := 35000
noncomputable def A_profit : ℕ := 14700
noncomputable def A_subscr : ℕ := B + 4000

theorem B_more_than_C (B_subscr C_subscr : ℕ) (h1 : A_subscr + B_subscr + C_subscr = total_subscription)
    (h2 : 14700 * 50000 = 35000 * A_subscr) :
    B_subscr - C_subscr = 5000 :=
sorry

end B_more_than_C_l13_13431


namespace range_of_3a_minus_b_l13_13606

theorem range_of_3a_minus_b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3)
                             (h3 : 2 < a - b) (h4 : a - b < 4) :
    ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 11 ∧ x = 3 * a - b :=
sorry

end range_of_3a_minus_b_l13_13606


namespace sets_are_equal_l13_13636

theorem sets_are_equal :
  let M := {x | ∃ k : ℤ, x = 2 * k + 1}
  let N := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}
  M = N :=
by
  sorry

end sets_are_equal_l13_13636


namespace total_hours_before_midterms_l13_13976

-- Define the hours spent on each activity per week
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3

-- Sum up the total hours spent on extracurriculars per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Define semester information
def total_weeks_per_semester : ℕ := 12
def weeks_before_midterms : ℕ := total_weeks_per_semester / 2
def weeks_sick : ℕ := 2
def active_weeks_before_midterms : ℕ := weeks_before_midterms - weeks_sick

-- Define the theorem statement about total hours before midterms
theorem total_hours_before_midterms : total_hours_per_week * active_weeks_before_midterms = 52 := by
  -- We skip the actual proof here
  sorry

end total_hours_before_midterms_l13_13976


namespace count_marble_pairs_l13_13687

-- Define conditions:
structure Marbles :=
(red : ℕ) (green : ℕ) (blue : ℕ) (yellow : ℕ) (white : ℕ)

def tomsMarbles : Marbles :=
  { red := 1, green := 1, blue := 1, yellow := 3, white := 2 }

-- Define a function to count pairs of marbles:
def count_pairs (m : Marbles) : ℕ :=
  -- Count pairs of identical marbles:
  (if m.yellow >= 2 then 1 else 0) + 
  (if m.white >= 2 then 1 else 0) +
  -- Count pairs of different colored marbles:
  (Nat.choose 5 2)

-- Theorem statement:
theorem count_marble_pairs : count_pairs tomsMarbles = 12 :=
  by
    sorry

end count_marble_pairs_l13_13687


namespace find_f_neg2_l13_13633

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2^x + 3*x - 1 else -(2^(-x) + 3*(-x) - 1)

theorem find_f_neg2 : f (-2) = -9 :=
by sorry

end find_f_neg2_l13_13633


namespace inequality_proof_l13_13143

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  (a / (b^2) + b / (a^2) ≥ 1 / a + 1 / b) :=
by
  sorry

end inequality_proof_l13_13143


namespace carrie_payment_l13_13116

def num_shirts := 8
def cost_per_shirt := 12
def total_shirt_cost := num_shirts * cost_per_shirt

def num_pants := 4
def cost_per_pant := 25
def total_pant_cost := num_pants * cost_per_pant

def num_jackets := 4
def cost_per_jacket := 75
def total_jacket_cost := num_jackets * cost_per_jacket

def num_skirts := 3
def cost_per_skirt := 30
def total_skirt_cost := num_skirts * cost_per_skirt

def num_shoes := 2
def cost_per_shoe := 50
def total_shoe_cost := num_shoes * cost_per_shoe

def total_cost := total_shirt_cost + total_pant_cost + total_jacket_cost + total_skirt_cost + total_shoe_cost

def mom_share := (2 / 3 : ℚ) * total_cost
def carrie_share := total_cost - mom_share

theorem carrie_payment : carrie_share = 228.67 :=
by
  sorry

end carrie_payment_l13_13116


namespace inequality_and_equality_conditions_l13_13361

theorem inequality_and_equality_conditions
  (x y a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a ≥ 0)
  (h3 : b ≥ 0) :
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 ∧ ((a * b = 0) ∨ (x = y)) :=
by
  sorry

end inequality_and_equality_conditions_l13_13361


namespace largest_even_digit_multiple_of_9_under_1000_l13_13821

theorem largest_even_digit_multiple_of_9_under_1000 : 
  ∃ n : ℕ, (∀ d ∈ Int.digits 10 n, d % 2 = 0) ∧ n < 1000 ∧ n % 9 = 0 ∧ 
  (∀ m : ℕ, (∀ d ∈ Int.digits 10 m, d % 2 = 0) ∧ m < 1000 ∧ m % 9 = 0 → m ≤ n) ∧ n = 864 :=
sorry

end largest_even_digit_multiple_of_9_under_1000_l13_13821


namespace total_chairs_l13_13096

theorem total_chairs (living_room_chairs kitchen_chairs : ℕ) (h1 : living_room_chairs = 3) (h2 : kitchen_chairs = 6) :
  living_room_chairs + kitchen_chairs = 9 := by
  sorry

end total_chairs_l13_13096


namespace rancher_lasso_probability_l13_13830

theorem rancher_lasso_probability : 
  let p_success := 1 / 2
  let p_failure := 1 - p_success
  (1 - p_failure ^ 3) = (7 / 8) := by
  sorry

end rancher_lasso_probability_l13_13830


namespace preimage_of_3_1_is_2_1_l13_13146

-- Definition of the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

-- The Lean theorem statement
theorem preimage_of_3_1_is_2_1 : ∃ (x y : ℝ), f x y = (3, 1) ∧ (x = 2 ∧ y = 1) :=
by
  sorry

end preimage_of_3_1_is_2_1_l13_13146


namespace right_triangle_ratio_l13_13468

theorem right_triangle_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∃ (x y : ℝ), 5 * (x * y) = x^2 + y^2 ∧ 5 * (a^2 + b^2) = (x + y)^2 ∧ 
    ((x - y)^2 < x^2 + y^2 ∧ x^2 + y^2 < (x + y)^2)):
  (1/2 < a / b) ∧ (a / b < 2) := by
  sorry

end right_triangle_ratio_l13_13468


namespace total_pink_crayons_l13_13035

def mara_crayons := 40
def mara_pink_percent := 10
def luna_crayons := 50
def luna_pink_percent := 20

def pink_crayons (total_crayons : ℕ) (percent_pink : ℕ) : ℕ :=
  (percent_pink * total_crayons) / 100

def mara_pink_crayons := pink_crayons mara_crayons mara_pink_percent
def luna_pink_crayons := pink_crayons luna_crayons luna_pink_percent

theorem total_pink_crayons : mara_pink_crayons + luna_pink_crayons = 14 :=
by
  -- Proof can be written here.
  sorry

end total_pink_crayons_l13_13035


namespace cole_drive_time_l13_13119

theorem cole_drive_time (d : ℝ) (h1 : d / 75 + d / 105 = 1) : (d / 75) * 60 = 35 :=
by
  -- Using the given equation: d / 75 + d / 105 = 1
  -- We solve it step by step and finally show that the time it took to drive to work is 35 minutes.
  sorry

end cole_drive_time_l13_13119


namespace no_unhappy_days_l13_13669

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l13_13669


namespace power_mod_eight_l13_13823

theorem power_mod_eight (n : ℕ) : (3^101 + 5) % 8 = 0 :=
by
  sorry

end power_mod_eight_l13_13823


namespace intersection_A_B_eq_C_l13_13590

def A : Set ℝ := { x | 4 - x^2 ≥ 0 }
def B : Set ℝ := { x | x > -1 }
def C : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B_eq_C : A ∩ B = C := 
by {
  sorry
}

end intersection_A_B_eq_C_l13_13590


namespace set_intersection_complement_l13_13548

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
noncomputable def M : Set ℕ := {2, 3, 4, 5}
noncomputable def N : Set ℕ := {1, 4, 5, 7}

theorem set_intersection_complement :
  M ∩ (U \ N) = {2, 3} :=
by
  sorry

end set_intersection_complement_l13_13548


namespace series_sum_eq_half_l13_13130

theorem series_sum_eq_half : ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_eq_half_l13_13130


namespace smaller_angle_between_hands_at_3_40_l13_13069

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l13_13069


namespace product_of_coefficients_l13_13686

theorem product_of_coefficients (b c : ℤ)
  (H1 : ∀ r, r^2 - 2 * r - 1 = 0 → r^5 - b * r - c = 0):
  b * c = 348 :=
by
  -- Solution steps would go here
  sorry

end product_of_coefficients_l13_13686


namespace solve_floor_trig_eq_l13_13507

-- Define the floor function
def floor (x : ℝ) : ℤ := by 
  sorry

-- Define the condition and theorem
theorem solve_floor_trig_eq (x : ℝ) (n : ℤ) : 
  floor (Real.sin x + Real.cos x) = 1 ↔ (∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (2 * Real.pi * n + Real.pi / 2)) := 
  by 
  sorry

end solve_floor_trig_eq_l13_13507


namespace sum_of_first_40_terms_l13_13054

def a : ℕ → ℤ := sorry

def S (n : ℕ) : ℤ := (Finset.range n).sum a

theorem sum_of_first_40_terms :
  (∀ n : ℕ, a (n + 1) + (-1) ^ n * a n = n) →
  S 40 = 420 := 
sorry

end sum_of_first_40_terms_l13_13054


namespace emily_height_in_cm_l13_13301

theorem emily_height_in_cm 
  (inches_in_foot : ℝ) (cm_in_foot : ℝ) (emily_height_in_inches : ℝ)
  (h_if : inches_in_foot = 12) (h_cf : cm_in_foot = 30.5) (h_ehi : emily_height_in_inches = 62) :
  emily_height_in_inches * (cm_in_foot / inches_in_foot) = 157.6 :=
by
  sorry

end emily_height_in_cm_l13_13301


namespace solve_equation_l13_13991

-- Define the given equation
def equation (x : ℝ) : Prop := (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = -3

-- State the theorem indicating the solutions to the equation
theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  equation x ↔ x = -2 ∨ x = 3 / 2 :=
sorry

end solve_equation_l13_13991


namespace values_of_cos_0_45_l13_13604

-- Define the interval and the condition for the cos function
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 360
def cos_condition (x : ℝ) : Prop := Real.cos x = 0.45

-- Final theorem statement
theorem values_of_cos_0_45 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), in_interval x ∧ cos_condition x ↔ x = 1 ∨ x = 2 := 
sorry

end values_of_cos_0_45_l13_13604


namespace dichromate_molecular_weight_l13_13397

theorem dichromate_molecular_weight :
  let atomic_weight_Cr := 52.00
  let atomic_weight_O := 16.00
  let dichromate_num_Cr := 2
  let dichromate_num_O := 7
  (dichromate_num_Cr * atomic_weight_Cr + dichromate_num_O * atomic_weight_O) = 216.00 :=
by
  sorry

end dichromate_molecular_weight_l13_13397


namespace find_years_ago_twice_age_l13_13057

-- Definitions of given conditions
def age_sum (H J : ℕ) : Prop := H + J = 43
def henry_age : ℕ := 27
def jill_age : ℕ := 16

-- Definition of the problem to be proved
theorem find_years_ago_twice_age (X : ℕ) 
  (h1 : age_sum henry_age jill_age) 
  (h2 : henry_age = 27) 
  (h3 : jill_age = 16) : (27 - X = 2 * (16 - X)) → X = 5 := 
by 
  sorry

end find_years_ago_twice_age_l13_13057


namespace find_number_l13_13497

def correct_answer (N : ℚ) : ℚ := 5 / 16 * N
def incorrect_answer (N : ℚ) : ℚ := 5 / 6 * N
def condition (N : ℚ) : Prop := incorrect_answer N = correct_answer N + 150

theorem find_number (N : ℚ) (h : condition N) : N = 288 / 5 := by
  sorry

end find_number_l13_13497


namespace inequality_proof_l13_13310

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (3 / (a^3 + b^3 + c^3)) ≤ 
  ((1 / (a^3 + b^3 + abc)) + (1 / (b^3 + c^3 + abc)) + (1 / (c^3 + a^3 + abc))) ∧ 
  ((1 / (a^3 + b^3 + abc)) + (1 / (b^3 + c^3 + abc)) + (1 / (c^3 + a^3 + abc)) ≤ (1 / (abc))) := 
sorry

end inequality_proof_l13_13310


namespace cyclist_average_speed_l13_13251

theorem cyclist_average_speed (v : ℝ) 
  (h1 : 8 / v + 10 / 8 = 18 / 8.78) : v = 10 :=
by
  sorry

end cyclist_average_speed_l13_13251


namespace first_discount_percentage_l13_13247

theorem first_discount_percentage 
  (original_price final_price : ℝ) 
  (successive_discount1 successive_discount2 : ℝ) 
  (h1 : original_price = 10000)
  (h2 : final_price = 6840)
  (h3 : successive_discount1 = 0.10)
  (h4 : successive_discount2 = 0.05)
  : ∃ x, (1 - x / 100) * (1 - successive_discount1) * (1 - successive_discount2) * original_price = final_price ∧ x = 20 :=
by
  sorry

end first_discount_percentage_l13_13247


namespace carly_practice_time_l13_13444

-- conditions
def practice_time_butterfly_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def practice_time_backstroke_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def total_weekly_practice (butterfly_hours : ℕ) (backstroke_hours : ℕ) : ℕ :=
  butterfly_hours + backstroke_hours

def monthly_practice (weekly_hours : ℕ) (weeks_per_month : ℕ) : ℕ :=
  weekly_hours * weeks_per_month

-- Proof Problem Statement
theorem carly_practice_time :
  practice_time_butterfly_weekly 3 4 + practice_time_backstroke_weekly 2 6 * 4 = 96 :=
by
  sorry

end carly_practice_time_l13_13444


namespace effective_simple_interest_rate_proof_l13_13950

noncomputable def effective_simple_interest_rate : ℝ :=
  let P := 1
  let r1 := 0.10 / 2 -- Half-yearly rate for year 1
  let t1 := 2 -- number of compounding periods semi-annual
  let A1 := P * (1 + r1) ^ t1

  let r2 := 0.12 / 2 -- Half-yearly rate for year 2
  let t2 := 2
  let A2 := A1 * (1 + r2) ^ t2

  let r3 := 0.14 / 2 -- Half-yearly rate for year 3
  let t3 := 2
  let A3 := A2 * (1 + r3) ^ t3

  let r4 := 0.16 / 2 -- Half-yearly rate for year 4
  let t4 := 2
  let A4 := A3 * (1 + r4) ^ t4

  let CI := 993
  let P_actual := CI / (A4 - P)
  let effective_simple_interest := (CI / P_actual) * 100
  effective_simple_interest

theorem effective_simple_interest_rate_proof :
  effective_simple_interest_rate = 65.48 := by
  sorry

end effective_simple_interest_rate_proof_l13_13950


namespace sub_base8_l13_13556

theorem sub_base8 : (1352 - 674) == 1456 :=
by sorry

end sub_base8_l13_13556


namespace does_not_round_to_72_56_l13_13694

-- Definitions for the numbers in question
def numA := 72.558
def numB := 72.563
def numC := 72.55999
def numD := 72.564
def numE := 72.555

-- Function to round a number to the nearest hundredth
def round_nearest_hundredth (x : Float) : Float :=
  (Float.round (x * 100) / 100 : Float)

-- Lean statement for the equivalent proof problem
theorem does_not_round_to_72_56 :
  round_nearest_hundredth numA = 72.56 ∧
  round_nearest_hundredth numB = 72.56 ∧
  round_nearest_hundredth numC = 72.56 ∧
  round_nearest_hundredth numD = 72.56 ∧
  round_nearest_hundredth numE ≠ 72.56 :=
by
  sorry

end does_not_round_to_72_56_l13_13694


namespace worker_cellphone_surveys_l13_13714

theorem worker_cellphone_surveys 
  (regular_rate : ℕ) 
  (num_surveys : ℕ) 
  (higher_rate : ℕ)
  (total_earnings : ℕ) 
  (earned : ℕ → ℕ → ℕ)
  (higher_earned : ℕ → ℕ → ℕ) 
  (h1 : regular_rate = 10) 
  (h2 : num_surveys = 50) 
  (h3 : higher_rate = 13) 
  (h4 : total_earnings = 605) 
  (h5 : ∀ x, earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings)
  : (∃ x, x = 35 ∧ earned regular_rate (num_surveys - x) + higher_earned higher_rate x = total_earnings) :=
sorry

end worker_cellphone_surveys_l13_13714


namespace product_evaluation_l13_13929

noncomputable def product_term (n : ℕ) : ℚ :=
  1 - (1 / (n * n))

noncomputable def product_expression : ℚ :=
  10 * 71 * (product_term 2) * (product_term 3) * (product_term 4) * (product_term 5) *
  (product_term 6) * (product_term 7) * (product_term 8) * (product_term 9) * (product_term 10)

theorem product_evaluation : product_expression = 71 := by
  sorry

end product_evaluation_l13_13929


namespace intersection_of_sets_l13_13591

theorem intersection_of_sets:
  let A := {-2, -1, 0, 1}
  let B := {x : ℤ | x^3 + 1 ≤ 0 }
  A ∩ B = {-2, -1} :=
by
  sorry

end intersection_of_sets_l13_13591


namespace sin_double_angle_l13_13306

theorem sin_double_angle (A : ℝ) (h₁ : 0 < A) (h₂ : A < π / 2) (h₃ : Real.cos A = 3 / 5) :
  Real.sin (2 * A) = 24 / 25 := 
by
  sorry

end sin_double_angle_l13_13306


namespace no_p_dependence_l13_13593

theorem no_p_dependence (m : ℕ) (p : ℕ) (hp : Prime p) (hm : m < p)
  (n : ℕ) (hn : 0 < n) (k : ℕ) 
  (h : m^2 + n^2 + p^2 - 2*m*n - 2*m*p - 2*n*p = k^2) : 
  ∀ q : ℕ, Prime q → m < q → (m^2 + n^2 + q^2 - 2*m*n - 2*m*q - 2*n*q = k^2) :=
by sorry

end no_p_dependence_l13_13593


namespace jasper_time_to_raise_kite_l13_13184

-- Define the conditions
def rate_of_omar : ℝ := 240 / 12 -- Rate of Omar in feet per minute
def rate_of_jasper : ℝ := 3 * rate_of_omar -- Jasper's rate is 3 times Omar's rate

def height_jasper : ℝ := 600 -- Height Jasper raises his kite

-- Define the time function for Jasper
def time_for_jasper_to_raise (height : ℝ) (rate : ℝ) : ℝ := height / rate

-- The main statement to prove
theorem jasper_time_to_raise_kite : time_for_jasper_to_raise height_jasper rate_of_jasper = 10 := by
  sorry

end jasper_time_to_raise_kite_l13_13184


namespace factor_expression_equals_one_l13_13733

theorem factor_expression_equals_one (a b c : ℝ) :
  ((a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2) / ((a - b)^2 + (b - c)^2 + (c - a)^2) = 1 :=
by
  sorry

end factor_expression_equals_one_l13_13733


namespace perimeter_of_figure_l13_13163

theorem perimeter_of_figure (x : ℕ) (h : x = 3) : 
  let sides := [x, x + 1, 6, 10]
  (sides.sum = 23) := by 
  sorry

end perimeter_of_figure_l13_13163


namespace canoe_row_probability_l13_13236

-- Definitions based on conditions
def prob_left_works : ℚ := 3 / 5
def prob_right_works : ℚ := 3 / 5

-- The probability that you can still row the canoe
def prob_can_row : ℚ := 
  prob_left_works * prob_right_works +  -- both oars work
  prob_left_works * (1 - prob_right_works) +  -- left works, right breaks
  (1 - prob_left_works) * prob_right_works  -- left breaks, right works
  
theorem canoe_row_probability : prob_can_row = 21 / 25 := by
  -- Skip proof for now
  sorry

end canoe_row_probability_l13_13236


namespace smallest_number_l13_13963

theorem smallest_number (N : ℤ) : (∃ (k : ℤ), N = 24 * k + 34) ∧ ∀ n, (∃ (k : ℤ), n = 24 * k + 10) -> n ≥ 34 := sorry

end smallest_number_l13_13963


namespace math_problem_mod_1001_l13_13454

theorem math_problem_mod_1001 :
  (2^6 * 3^10 * 5^12 - 75^4 * (26^2 - 1)^2 + 3^10 - 50^6 + 5^12) % 1001 = 400 := by
  sorry

end math_problem_mod_1001_l13_13454


namespace problem_statement_l13_13794

open Set

variable (U P Q : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5, 6}) (hP : P = {1, 2, 3, 4}) (hQ : Q = {3, 4, 5})

theorem problem_statement : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end problem_statement_l13_13794


namespace number_solution_exists_l13_13824

theorem number_solution_exists (x : ℝ) (h : 0.80 * x = (4 / 5 * 15) + 20) : x = 40 :=
sorry

end number_solution_exists_l13_13824


namespace clock_angle_3_40_l13_13078

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l13_13078


namespace integer_solutions_l13_13736

theorem integer_solutions (x y : ℤ) : 
  (x^2 + x = y^4 + y^3 + y^2 + y) ↔ 
  (x, y) = (0, -1) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (5, 2) :=
by
  sorry

end integer_solutions_l13_13736


namespace discount_rate_for_1000_min_price_for_1_3_discount_l13_13549

def discounted_price (original_price : ℕ) : ℕ := 
  original_price * 80 / 100

def voucher_amount (discounted_price : ℕ) : ℕ :=
  if discounted_price < 400 then 30
  else if discounted_price < 500 then 60
  else if discounted_price < 700 then 100
  else if discounted_price < 900 then 130
  else 0 -- Can extend the rule as needed

def discount_rate (original_price : ℕ) : ℚ := 
  let total_discount := original_price * 20 / 100 + voucher_amount (discounted_price original_price)
  (total_discount : ℚ) / (original_price : ℚ)

theorem discount_rate_for_1000 : 
  discount_rate 1000 = 0.33 := 
by
  sorry

theorem min_price_for_1_3_discount :
  ∀ (x : ℕ), 500 ≤ x ∧ x ≤ 800 → 0.33 ≤ discount_rate x ↔ (625 ≤ x ∧ x ≤ 750) :=
by
  sorry

end discount_rate_for_1000_min_price_for_1_3_discount_l13_13549


namespace no_such_triangle_exists_l13_13500

theorem no_such_triangle_exists (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : b = 0.25 * (a + b + c)) :
  ¬ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end no_such_triangle_exists_l13_13500


namespace sunlovers_happy_days_l13_13679

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l13_13679


namespace negation_abs_lt_zero_l13_13741

theorem negation_abs_lt_zero : ¬ (∀ x : ℝ, |x| < 0) ↔ ∃ x : ℝ, |x| ≥ 0 := 
by 
  sorry

end negation_abs_lt_zero_l13_13741


namespace min_value_fraction_solve_inequality_l13_13320

-- Part 1
theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (f : ℝ → ℝ)
  (h3 : f 1 = 2) (h4 : ∀ x, f x = a * x^2 + b * x + 1) :
  (a + b = 1) → (∃ z, z = (1 / a + 4 / b) ∧ z = 9) := 
by {
  sorry
}

-- Part 2
theorem solve_inequality (a : ℝ) (x : ℝ) (h1 : b = -a - 1) (f : ℝ → ℝ)
  (h2 : ∀ x, f x = a * x^2 + b * x + 1) :
  (f x ≤ 0) → 
  (if a = 0 then 
      {x | x ≥ 1}
  else if a > 0 then
      if a = 1 then 
          {x | x = 1}
      else if 0 < a ∧ a < 1 then 
          {x | 1 ≤ x ∧ x ≤ 1 / a}
      else 
          {x | 1 / a ≤ x ∧ x ≤ 1}
  else 
      {x | x ≥ 1 ∨ x ≤ 1 / a}) :=
by {
  sorry
}

end min_value_fraction_solve_inequality_l13_13320


namespace student_chose_124_l13_13239

theorem student_chose_124 (x : ℤ) (h : 2 * x - 138 = 110) : x = 124 := 
by {
  sorry
}

end student_chose_124_l13_13239


namespace find_z_given_conditions_l13_13165

variable (x y z : ℤ)

theorem find_z_given_conditions :
  (x + y) / 2 = 4 →
  x + y + z = 0 →
  z = -8 := by
  sorry

end find_z_given_conditions_l13_13165


namespace sequence_10_eq_123_l13_13039

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 3
  | n + 2 => sequence (n + 1) + sequence n

theorem sequence_10_eq_123 : sequence 10 = 123 :=
by
  sorry

end sequence_10_eq_123_l13_13039


namespace quadratic_linear_term_l13_13009

theorem quadratic_linear_term (m : ℝ) 
  (h : 2 * m = 6) : -4 * (x : ℝ) + m * x = -x := by 
  sorry

end quadratic_linear_term_l13_13009


namespace inequality_solution_set_l13_13801

open Set

noncomputable def rational_expression (x : ℝ) : ℝ := (x^2 - 16) / (x^2 + 10*x + 25)

theorem inequality_solution_set :
  {x : ℝ | rational_expression x < 0} = Ioo (-4 : ℝ) 4 :=
by
  sorry

end inequality_solution_set_l13_13801


namespace simplify_and_evaluate_expression_l13_13042

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 + Real.sqrt 3) : 
  ( ( x + 3 ) / ( x^2 - 2*x + 1 ) * ( x - 1 ) / ( x^2 + 3*x ) + 1 / x ) = Real.sqrt 3 / 3 :=
by
  rw h
  sorry

end simplify_and_evaluate_expression_l13_13042


namespace P_1_lt_X_lt_5_l13_13151

-- Define the random variable following a normal distribution
noncomputable def X : ℝ → ℝ := Normal(3, σ^2)

-- Define the probability values given in the problem
def P_X_geq_5 : ℝ := 0.15

-- State the proposition to prove
theorem P_1_lt_X_lt_5 : P(1 < X < 5) = 0.7 := by
  sorry

end P_1_lt_X_lt_5_l13_13151


namespace tangent_line_at_one_e_l13_13205

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_one_e : ∀ (x y : ℝ), (x, y) = (1, Real.exp 1) → (y = 2 * Real.exp x * x - Real.exp 1) :=
by
  intro x y h
  sorry

end tangent_line_at_one_e_l13_13205


namespace largest_divisor_of_m_l13_13008

theorem largest_divisor_of_m (m : ℕ) (h1 : 0 < m) (h2 : 39 ∣ m^2) : 39 ∣ m := sorry

end largest_divisor_of_m_l13_13008


namespace clock_angle_3_40_l13_13079

/-
  Prove that the angle between the clock hands at 3:40 pm is 130 degrees,
  given the movement conditions of the clock hands.
-/
theorem clock_angle_3_40 : 
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  in angle_between = 130 :=
by
  let minute_angle_per_minute := 6
  let hour_angle_per_minute := 0.5
  let minutes := 40
  let minute_angle := minutes * minute_angle_per_minute
  let hours := 3
  let hour_angle := hours * 30 + minutes * hour_angle_per_minute
  let angle_between := minute_angle - hour_angle
  have step1 : minute_angle = 240 := by sorry
  have step2 : hour_angle = 110 := by sorry
  have step3 : angle_between = 130 := by sorry
  exact step3

end clock_angle_3_40_l13_13079


namespace goods_train_speed_l13_13097

theorem goods_train_speed 
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_to_cross : ℕ)
  (h_train : length_train = 270)
  (h_platform : length_platform = 250)
  (h_time : time_to_cross = 26) : 
  (length_train + length_platform) / time_to_cross = 20 := 
by
  sorry

end goods_train_speed_l13_13097


namespace A_pow_101_l13_13501

def A : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 0, 1],
  ![1, 0, 0],
  ![0, 1, 0]
]

theorem A_pow_101 :
  A ^ 101 = ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] := by
  sorry

end A_pow_101_l13_13501


namespace face_sum_l13_13371

theorem face_sum (a b c d e f : ℕ) (h : (a + d) * (b + e) * (c + f) = 1008) : 
  a + b + c + d + e + f = 173 :=
by
  sorry

end face_sum_l13_13371


namespace trigonometric_expression_l13_13311

theorem trigonometric_expression
  (α : ℝ)
  (h1 : Real.tan α = 3) : 
  (Real.sin α + 3 * Real.cos α) / (Real.cos α - 3 * Real.sin α) = -3/4 := 
by
  sorry

end trigonometric_expression_l13_13311


namespace machine_shirt_rate_l13_13280

theorem machine_shirt_rate (S : ℕ) 
  (worked_yesterday : ℕ) (worked_today : ℕ) (shirts_today : ℕ) 
  (h1 : worked_yesterday = 5)
  (h2 : worked_today = 12)
  (h3 : shirts_today = 72)
  (h4 : worked_today * S = shirts_today) : 
  S = 6 := 
by 
  sorry

end machine_shirt_rate_l13_13280


namespace repeating_decimal_sum_is_one_l13_13989

noncomputable def repeating_decimal_sum : ℝ :=
  let x := (1/3 : ℝ)
  let y := (2/3 : ℝ)
  x + y

theorem repeating_decimal_sum_is_one : repeating_decimal_sum = 1 := by
  sorry

end repeating_decimal_sum_is_one_l13_13989


namespace gumball_draw_probability_l13_13829

def prob_blue := 2 / 3
def prob_two_blue := (16 / 36)
def prob_pink := 1 - prob_blue

theorem gumball_draw_probability
    (h1 : prob_two_blue = prob_blue * prob_blue)
    (h2 : prob_blue + prob_pink = 1) :
    prob_pink = 1 / 3 := 
by
  sorry

end gumball_draw_probability_l13_13829


namespace range_of_m_l13_13169

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 5| < m^2 - m) ↔ m < -1 ∨ m > 2 := 
by
  sorry

end range_of_m_l13_13169


namespace range_of_a_if_p_is_false_l13_13879

theorem range_of_a_if_p_is_false :
  (∀ x : ℝ, x^2 + a * x + a ≥ 0) → (0 ≤ a ∧ a ≤ 4) := 
sorry

end range_of_a_if_p_is_false_l13_13879


namespace tetrahedron_vertices_identical_l13_13376

theorem tetrahedron_vertices_identical
  (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ)
  (h1 : a1 * a2 + a2 * a3 + a3 * a1 = b1 * b2 + b2 * b3 + b3 * b1)
  (h2 : a1 * a2 + a2 * a4 + a4 * a1 = b1 * b2 + b2 * b4 + b4 * b1)
  (h3 : a1 * a3 + a3 * a4 + a4 * a1 = b1 * b3 + b3 * b4 + b4 * b1)
  (h4 : a2 * a3 + a3 * a4 + a4 * a2 = b2 * b3 + b3 * b4 + b4 * b2) :
  multiset.of_list [a1, a2, a3, a4] = multiset.of_list [b1, b2, b3, b4] :=
by
  sorry

end tetrahedron_vertices_identical_l13_13376


namespace kanul_spent_on_raw_materials_l13_13021

theorem kanul_spent_on_raw_materials 
    (total_amount : ℝ)
    (spent_machinery : ℝ)
    (spent_cash_percent : ℝ)
    (spent_cash : ℝ)
    (amount_raw_materials : ℝ)
    (h_total : total_amount = 93750)
    (h_machinery : spent_machinery = 40000)
    (h_percent : spent_cash_percent = 20 / 100)
    (h_cash : spent_cash = spent_cash_percent * total_amount)
    (h_sum : total_amount = amount_raw_materials + spent_machinery + spent_cash) : 
    amount_raw_materials = 35000 :=
sorry

end kanul_spent_on_raw_materials_l13_13021


namespace real_roots_for_all_a_b_l13_13522

theorem real_roots_for_all_a_b (a b : ℝ) : ∃ x : ℝ, (x^2 / (x^2 - a^2) + x^2 / (x^2 - b^2) = 4) :=
sorry

end real_roots_for_all_a_b_l13_13522


namespace smallest_k_for_a_l13_13026

theorem smallest_k_for_a (a n : ℕ) (h : 10 ^ 2013 ≤ a^n ∧ a^n < 10 ^ 2014) : ∀ k : ℕ, k < 46 → ∃ n : ℕ, (10 ^ (k - 1)) ≤ a ∧ a < 10 ^ k :=
by sorry

end smallest_k_for_a_l13_13026


namespace correct_factorization_l13_13233

theorem correct_factorization :
  ∀ (x : ℝ), -x^2 + 2*x - 1 = - (x - 1)^2 :=
by
  intro x
  sorry

end correct_factorization_l13_13233


namespace three_digit_multiples_of_15_not_70_l13_13000

theorem three_digit_multiples_of_15_not_70 : 
  let is_three_digit := λ n : ℕ, 100 ≤ n ∧ n < 1000
  ∧ is_multiple_of_15 := λ n : ℕ, n % 15 = 0
  ∧ is_multiple_of_70 := λ n : ℕ, n % 70 = 0
  ∧ valid_num := λ n : ℕ, is_three_digit n ∧ is_multiple_of_15 n ∧ ¬is_multiple_of_70 n in
  (count (λ n, valid_num n) (list.range' 100 900) = 56) :=
by sorry

end three_digit_multiples_of_15_not_70_l13_13000


namespace cow_value_increase_l13_13631

theorem cow_value_increase :
  let starting_weight : ℝ := 732
  let increase_factor : ℝ := 1.35
  let price_per_pound : ℝ := 2.75
  let new_weight := starting_weight * increase_factor
  let value_at_new_weight := new_weight * price_per_pound
  let value_at_starting_weight := starting_weight * price_per_pound
  let increase_in_value := value_at_new_weight - value_at_starting_weight
  increase_in_value = 704.55 :=
by
  sorry

end cow_value_increase_l13_13631


namespace nathan_blankets_l13_13083

theorem nathan_blankets (b : ℕ) (hb : 21 = (b / 2) * 3) : b = 14 :=
by sorry

end nathan_blankets_l13_13083


namespace ratio_of_length_to_height_l13_13526

theorem ratio_of_length_to_height
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (vol_eq : 129024 = w * h * l)
  (w_eq : w = 8) :
  l / h = 7 := 
sorry

end ratio_of_length_to_height_l13_13526


namespace small_angle_at_3_40_is_130_degrees_l13_13065

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l13_13065


namespace find_f_value_l13_13266

def f (x : ℤ) : ℤ := sorry

theorem find_f_value :
  (f(1) + 1 > 0) ∧ 
  (∀ (x y : ℤ), f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y) ∧
  (∀ (x : ℤ), 2 * f(x) = f(x + 1) - x + 1) →
  f 10 = 1014 :=
by
  sorry

end find_f_value_l13_13266


namespace intersection_M_N_l13_13322

open Set

def M : Set ℝ := { x | x^2 - x - 2 ≤ 0 }
def N : Set ℝ := { x | 0 < x }

theorem intersection_M_N : M ∩ N = Set.Ioc 0 2 := by
  sorry

end intersection_M_N_l13_13322


namespace proof_GP_product_l13_13727

namespace GPProof

variables {a r : ℝ} {n : ℕ} (S S' P : ℝ)

def isGeometricProgression (a r : ℝ) (n : ℕ) :=
  ∀ i, 0 ≤ i ∧ i < n → ∃ k, ∃ b, b = (-1)^k * a * r^k ∧ k = i 

noncomputable def product (a r : ℝ) (n : ℕ) : ℝ :=
  a^n * r^(n*(n-1)/2) * (-1)^(n*(n-1)/2)

noncomputable def sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - (-r)^n) / (1 - (-r))

noncomputable def reciprocalSum (a r : ℝ) (n : ℕ) : ℝ :=
  (1 / a) * (1 - (-1/r)^n) / (1 + 1/r)

theorem proof_GP_product (hyp1 : isGeometricProgression a (-r) n) (hyp2 : S = sum a (-r) n) (hyp3 : S' = reciprocalSum a (-r) n) (hyp4 : P = product a (-r) n) :
  P = (S / S')^(n/2) :=
by
  sorry

end GPProof

end proof_GP_product_l13_13727


namespace triangle_side_value_l13_13905

theorem triangle_side_value
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 1)
  (h2 : b = 4)
  (h3 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h4 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) :
  c = Real.sqrt 13 :=
sorry

end triangle_side_value_l13_13905


namespace average_people_per_boat_correct_l13_13214

-- Define number of boats and number of people
def num_boats := 3.0
def num_people := 5.0

-- Definition for average people per boat
def avg_people_per_boat := num_people / num_boats

-- Theorem to prove the average number of people per boat is 1.67
theorem average_people_per_boat_correct : avg_people_per_boat = 1.67 := by
  sorry

end average_people_per_boat_correct_l13_13214


namespace determine_b_eq_l13_13986

theorem determine_b_eq (b : ℝ) : (∃! (x : ℝ), |x^2 + 3 * b * x + 4 * b| ≤ 3) ↔ b = 4 / 3 ∨ b = 1 := 
by sorry

end determine_b_eq_l13_13986


namespace transfer_deck_l13_13060

-- Define the conditions
variables {k n : ℕ}

-- Assume conditions explicitly
axiom k_gt_1 : k > 1
axiom cards_deck : 2*n = 2*n -- Implicitly states that we have 2n cards

-- Define the problem statement
theorem transfer_deck (k_gt_1 : k > 1) (cards_deck : 2*n = 2*n) : n = k - 1 :=
sorry

end transfer_deck_l13_13060


namespace find_x_value_l13_13004

theorem find_x_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 5 * x^2 + 15 * x * y = x^3 + 2 * x^2 * y + 3 * x * y^2) : x = 5 :=
sorry

end find_x_value_l13_13004


namespace product_of_numbers_l13_13927

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 85) : x * y = 18 := by
  sorry

end product_of_numbers_l13_13927


namespace find_original_number_l13_13717

-- Defining the conditions as given in the problem
def original_number_condition (x : ℤ) : Prop :=
  3 * (3 * x - 6) = 141

-- Stating the main theorem to be proven
theorem find_original_number (x : ℤ) (h : original_number_condition x) : x = 17 :=
sorry

end find_original_number_l13_13717


namespace smaller_angle_between_clock_hands_3_40_pm_l13_13075

theorem smaller_angle_between_clock_hands_3_40_pm :
  let minute_hand_angle : ℝ := 240
  let hour_hand_angle : ℝ := 110
  let angle_between_hands : ℝ := |minute_hand_angle - hour_hand_angle|
  angle_between_hands = 130.0 :=
by
  sorry

end smaller_angle_between_clock_hands_3_40_pm_l13_13075


namespace no_unhappy_days_l13_13667

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l13_13667


namespace impossible_distinct_values_l13_13767

theorem impossible_distinct_values :
  ∀ a b c : ℝ, 
  (a * (a - 4) = 12) → 
  (b * (b - 4) = 12) → 
  (c * (c - 4) = 12) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) → 
  false := 
sorry

end impossible_distinct_values_l13_13767


namespace trainee_teacher_arrangements_l13_13425

-- Define the main theorem
theorem trainee_teacher_arrangements :
  let n := 5
  let classes := 3
  let arrangements := 50
  (∃ (C : Fin classes → Finset (Fin n)) (h : ∀ i, C i ≠ ∅) (hXiaoLi : 0 ∈ C 0),
    (∀ i j, i ≠ j → C i ∩ C j = ∅) ∧ (Finset.univ = ⋃ i, C i)) ↔ (n = 5 ∧ classes = 3 ∧ arrangements = 50) := 
by 
  sorry

end trainee_teacher_arrangements_l13_13425


namespace cubic_equation_roots_l13_13650

theorem cubic_equation_roots :
  (∀ x : ℝ, (x^3 - 7*x^2 + 36 = 0) → (x = -2 ∨ x = 3 ∨ x = 6)) ∧
  ∃ (x1 x2 x3 : ℝ), (x1 * x2 = 18) ∧ (x1 * x2 * x3 = -36) :=
by
  sorry

end cubic_equation_roots_l13_13650


namespace triangle_inequality_l13_13913

variable {α β γ a b c : ℝ}

theorem triangle_inequality (h1: α ≥ β) (h2: β ≥ γ) (h3: a ≥ b) (h4: b ≥ c) (h5: α ≥ γ) (h6: a ≥ c) :
  a * α + b * β + c * γ ≥ a * β + b * γ + c * α :=
by
  sorry

end triangle_inequality_l13_13913


namespace complex_sum_cubics_eq_zero_l13_13786

-- Define the hypothesis: omega is a nonreal root of x^3 = 1
def is_nonreal_root_of_cubic (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω ≠ 1

-- Now state the theorem to prove the expression evaluates to 0
theorem complex_sum_cubics_eq_zero (ω : ℂ) (h : is_nonreal_root_of_cubic ω) :
  (2 - 2*ω + 2*ω^2)^3 + (2 + 2*ω - 2*ω^2)^3 = 0 :=
by
  -- This is where the proof would go. 
  sorry

end complex_sum_cubics_eq_zero_l13_13786


namespace sum_of_squares_of_roots_l13_13572

theorem sum_of_squares_of_roots : 
  (∃ r1 r2 : ℝ, r1 + r2 = 11 ∧ r1 * r2 = 12 ∧ (r1 ^ 2 + r2 ^ 2) = 97) := 
sorry

end sum_of_squares_of_roots_l13_13572


namespace total_weight_l13_13099

axiom D : ℕ -- Daughter's weight
axiom C : ℕ -- Grandchild's weight
axiom M : ℕ -- Mother's weight

-- Given conditions from the problem
axiom h1 : D + C = 60
axiom h2 : C = M / 5
axiom h3 : D = 50

-- The statement to be proven
theorem total_weight : M + D + C = 110 :=
by sorry

end total_weight_l13_13099


namespace circle_area_l13_13179

open Real

theorem circle_area (x y : ℝ) :
  (∃ r, (x + 2)^2 + (y - 3 / 2)^2 = r^2) →
  r = 7 / 2 →
  ∃ A, A = (π * (r)^2) ∧ A = (49/4) * π :=
by
  sorry

end circle_area_l13_13179


namespace a_share_calculation_l13_13696

noncomputable def investment_a : ℕ := 15000
noncomputable def investment_b : ℕ := 21000
noncomputable def investment_c : ℕ := 27000
noncomputable def total_investment : ℕ := investment_a + investment_b + investment_c -- 63000
noncomputable def b_share : ℕ := 1540
noncomputable def total_profit : ℕ := 4620  -- from the solution steps

theorem a_share_calculation :
  (investment_a * total_profit) / total_investment = 1100 := 
by
  sorry

end a_share_calculation_l13_13696


namespace necessary_and_sufficient_condition_l13_13092

theorem necessary_and_sufficient_condition 
  (a : ℕ) 
  (A B : ℝ) 
  (x y z : ℤ) 
  (h1 : (x^2 + y^2 + z^2 : ℝ) = (B * ↑a)^2) 
  (h2 : (x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) : ℝ) = (1 / 4) * (2 * A + B) * (B * (↑a)^4)) :
  B = 2 * A :=
by
  sorry

end necessary_and_sufficient_condition_l13_13092


namespace find_pairs_l13_13990

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ a b, (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3))
  ↔ (∃ a b, a > 0 ∧ b > 0 ∧ (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0) := by
  sorry

end find_pairs_l13_13990


namespace find_m_l13_13490

theorem find_m (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 2 * y - 4 = 0) →
  (∀ x y : ℝ, x - 2 * y + m = 0) →
  (m = 7 ∨ m = -3) :=
by
  sorry

end find_m_l13_13490


namespace sin_squared_minus_cos_squared_value_l13_13530

noncomputable def sin_squared_minus_cos_squared : Real :=
  (Real.sin (Real.pi / 12))^2 - (Real.cos (Real.pi / 12))^2

theorem sin_squared_minus_cos_squared_value :
  sin_squared_minus_cos_squared = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_squared_minus_cos_squared_value_l13_13530


namespace find_triangle_angles_l13_13177

theorem find_triangle_angles (a b h_a h_b : ℝ) (A B C : ℝ) :
  a ≤ h_a → b ≤ h_b →
  h_a ≤ b → h_b ≤ a →
  ∃ x y z : ℝ, (x = 90 ∧ y = 45 ∧ z = 45) ∧ 
  (x + y + z = 180) :=
by
  sorry

end find_triangle_angles_l13_13177


namespace perpendicular_lines_slope_l13_13601

theorem perpendicular_lines_slope {m : ℝ} : 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → 
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0 → (m * (-1/2)) = -1) → 
  m = 2 :=
by 
  intros h_perpendicular h_slope
  sorry

end perpendicular_lines_slope_l13_13601


namespace average_check_l13_13048

variable (a b c d e f g x : ℕ)

def sum_natural (l : List ℕ) : ℕ := l.foldr (λ x y => x + y) 0

theorem average_check (h1 : a = 54) (h2 : b = 55) (h3 : c = 57) (h4 : d = 58) (h5 : e = 59) (h6 : f = 63) (h7 : g = 65) (h8 : x = 65) (avg : 60 * 8 = 480) :
    sum_natural [a, b, c, d, e, f, g, x] = 480 :=
by
  sorry

end average_check_l13_13048


namespace electric_car_charging_cost_l13_13012

/-- The fractional equation for the given problem,
    along with the correct solution for the average charging cost per kilometer. -/
theorem electric_car_charging_cost (
    x : ℝ
) : 
    (200 / x = 4 * (200 / (x + 0.6))) → x = 0.2 :=
by
  intros h_eq
  sorry

end electric_car_charging_cost_l13_13012


namespace race_length_l13_13520

theorem race_length (members : ℕ) (member_distance : ℕ) (ralph_multiplier : ℕ) 
    (h1 : members = 4) (h2 : member_distance = 3) (h3 : ralph_multiplier = 2) : 
    members * member_distance + ralph_multiplier * member_distance = 18 :=
by
  -- Start the proof with sorry to denote missing steps.
  sorry

end race_length_l13_13520


namespace group_division_ways_l13_13537

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem group_division_ways : 
  choose 30 10 * choose 20 10 * choose 10 10 = Nat.factorial 30 / (Nat.factorial 10 * Nat.factorial 10 * Nat.factorial 10) := 
by
  sorry

end group_division_ways_l13_13537


namespace sunlovers_happy_days_l13_13678

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l13_13678


namespace tangent_product_l13_13567

theorem tangent_product (n θ : ℝ) 
  (h1 : ∀ n θ, tan (n * θ) = (sin (n * θ)) / (cos (n * θ)))
  (h2 : tan 8 θ = (8 * tan θ - 56 * (tan θ) ^ 3 + 56 * (tan θ) ^ 5 - 8 * (tan θ) ^ 7) / 
                  (1 - 28 * (tan θ) ^ 2 + 70 * (tan θ) ^ 4 - 28 * (tan θ) ^ 6))
  (h3 : tan (8 * (π / 8)) = 0)
  (h4 : tan (8 * (3 * π / 8)) = 0)
  (h5 : tan (8 * (5 * π / 8)) = 0) :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = 2 * sqrt 2 :=
sorry

end tangent_product_l13_13567


namespace total_trees_in_park_l13_13181

theorem total_trees_in_park (oak_planted_total maple_planted_total birch_planted_total : ℕ)
  (initial_oak initial_maple initial_birch : ℕ)
  (oak_removed_day2 maple_removed_day2 birch_removed_day2 : ℕ)
  (D1_oak_plant : ℕ) (D2_oak_plant : ℕ) (D1_maple_plant : ℕ) (D2_maple_plant : ℕ)
  (D1_birch_plant : ℕ) (D2_birch_plant : ℕ):
  initial_oak = 25 → initial_maple = 40 → initial_birch = 20 →
  oak_planted_total = 73 → maple_planted_total = 52 → birch_planted_total = 35 →
  D1_oak_plant = 29 → D2_oak_plant = 26 →
  D1_maple_plant = 26 → D2_maple_plant = 13 →
  D1_birch_plant = 10 → D2_birch_plant = 16 →
  oak_removed_day2 = 15 → maple_removed_day2 = 10 → birch_removed_day2 = 5 →
  (initial_oak + oak_planted_total - oak_removed_day2) +
  (initial_maple + maple_planted_total - maple_removed_day2) +
  (initial_birch + birch_planted_total - birch_removed_day2) = 215 :=
by
  intros h_initial_oak h_initial_maple h_initial_birch
         h_oak_planted_total h_maple_planted_total h_birch_planted_total
         h_D1_oak h_D2_oak h_D1_maple h_D2_maple h_D1_birch h_D2_birch
         h_oak_removed h_maple_removed h_birch_removed
  sorry

end total_trees_in_park_l13_13181


namespace jordan_book_pages_l13_13349

theorem jordan_book_pages (avg_first_4_days : ℕ)
                           (avg_next_2_days : ℕ)
                           (pages_last_day : ℕ)
                           (total_pages : ℕ) :
  avg_first_4_days = 42 → 
  avg_next_2_days = 38 → 
  pages_last_day = 20 → 
  total_pages = 4 * avg_first_4_days + 2 * avg_next_2_days + pages_last_day →
  total_pages = 264 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end jordan_book_pages_l13_13349


namespace min_time_to_same_side_l13_13437

def side_length : ℕ := 50
def speed_A : ℕ := 5
def speed_B : ℕ := 3

def time_to_same_side (side_length speed_A speed_B : ℕ) : ℕ :=
  30

theorem min_time_to_same_side :
  time_to_same_side side_length speed_A speed_B = 30 :=
by
  -- The proof goes here
  sorry

end min_time_to_same_side_l13_13437


namespace sum_of_coefficients_l13_13389

theorem sum_of_coefficients
  (d : ℝ)
  (g h : ℝ)
  (h1 : (8 * d^2 - 4 * d + g) * (5 * d^2 + h * d - 10) = 40 * d^4 - 75 * d^3 - 90 * d^2 + 5 * d + 20) :
  g + h = 15.5 :=
sorry

end sum_of_coefficients_l13_13389


namespace find_a7_l13_13774

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, ∃ r, a (n + m) = (a n) * (r ^ m)

def sequence_properties (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ a 3 = 3 ∧ a 11 = 27

theorem find_a7 (a : ℕ → ℝ) (h : sequence_properties a) : a 7 = 9 := 
sorry

end find_a7_l13_13774


namespace min_games_required_l13_13620

-- Given condition: max_games ≤ 15
def max_games := 15

-- Theorem statement to prove: minimum number of games that must be played is 8
theorem min_games_required (n : ℕ) (h : n ≤ max_games) : n = 8 :=
sorry

end min_games_required_l13_13620


namespace xyz_leq_36_l13_13784

theorem xyz_leq_36 {x y z : ℝ} 
    (hx0 : x > 0) (hy0 : y > 0) (hz0 : z > 0) 
    (hx2 : x ≤ 2) (hy3 : y ≤ 3) 
    (hxyz_sum : x + y + z = 11) : 
    x * y * z ≤ 36 := 
by
  sorry

end xyz_leq_36_l13_13784


namespace pradeep_failed_by_25_marks_l13_13797

theorem pradeep_failed_by_25_marks :
  (35 / 100 * 600 : ℝ) - 185 = 25 :=
by
  sorry

end pradeep_failed_by_25_marks_l13_13797


namespace sum_of_first_four_terms_l13_13772

noncomputable def sum_first_n_terms (a q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_first_four_terms :
  ∀ (a q : ℝ), a * (1 + q) = 7 → a * (q^6 - 1) / (q - 1) = 91 →
  a * (1 + q + q^2 + q^3) = 28 :=
by
  intros a q h₁ h₂
  -- Proof omitted
  sorry

end sum_of_first_four_terms_l13_13772


namespace find_f_10_l13_13271

def f (x : ℤ) : ℤ := sorry

noncomputable def h (x : ℤ) : ℤ := f(x) + x

axiom condition_1 : f(1) + 1 > 0

axiom condition_2 : ∀ (x y : ℤ), f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y

axiom condition_3 : ∀ (x : ℤ), 2 * f(x) = f(x + 1) - x + 1

theorem find_f_10 : f(10) = 1014 := sorry

end find_f_10_l13_13271


namespace al_initial_portion_l13_13112

theorem al_initial_portion (a b c : ℝ) 
  (h1 : a + b + c = 1200) 
  (h2 : a - 200 + 2 * b + 1.5 * c = 1800) : 
  a = 600 :=
sorry

end al_initial_portion_l13_13112


namespace compute_five_fold_application_l13_13632

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then -2 * x^2 else x^2 + 4 * x + 12

theorem compute_five_fold_application :
  f (f (f (f (f 2)))) = -449183247763232 :=
  by
    sorry

end compute_five_fold_application_l13_13632


namespace total_toothpicks_in_grid_l13_13936

theorem total_toothpicks_in_grid (l w : ℕ) (h₁ : l = 50) (h₂ : w = 20) : 
  (l + 1) * w + (w + 1) * l + 2 * (l * w) = 4070 :=
by
  sorry

end total_toothpicks_in_grid_l13_13936


namespace john_gallons_of_gas_l13_13783

theorem john_gallons_of_gas
  (rental_cost : ℝ)
  (gas_cost_per_gallon : ℝ)
  (mile_cost : ℝ)
  (miles_driven : ℝ)
  (total_cost : ℝ)
  (rental_cost_val : rental_cost = 150)
  (gas_cost_per_gallon_val : gas_cost_per_gallon = 3.50)
  (mile_cost_val : mile_cost = 0.50)
  (miles_driven_val : miles_driven = 320)
  (total_cost_val : total_cost = 338) :
  ∃ gallons_of_gas : ℝ, gallons_of_gas = 8 :=
by
  sorry

end john_gallons_of_gas_l13_13783


namespace mn_min_l13_13844

noncomputable def min_mn_value (m n : ℝ) : ℝ := m * n

theorem mn_min : 
  (∃ m n, m = Real.sin (2 * (π / 12)) ∧ n > 0 ∧ 
            Real.cos (2 * (π / 12 + n) - π / 4) = m ∧ 
            min_mn_value m n = π * 5 / 48) := by
  sorry

end mn_min_l13_13844


namespace mask_distribution_l13_13538

theorem mask_distribution (x : ℕ) (total_masks_3 : ℕ) (total_masks_4 : ℕ)
    (h1 : total_masks_3 = 3 * x + 20)
    (h2 : total_masks_4 = 4 * x - 25) :
    3 * x + 20 = 4 * x - 25 :=
by
  sorry

end mask_distribution_l13_13538


namespace percentage_equivalence_l13_13329

theorem percentage_equivalence (A B C P : ℝ)
  (hA : A = 0.80 * 600)
  (hB : B = 480)
  (hC : C = 960)
  (hP : P = (B / C) * 100) :
  A = P * 10 :=  -- Since P is the percentage, we use it to relate A to C
sorry

end percentage_equivalence_l13_13329


namespace remainder_proof_l13_13300

def nums : List ℕ := [83, 84, 85, 86, 87, 88, 89, 90]
def mod : ℕ := 17

theorem remainder_proof : (nums.sum % mod) = 3 := by sorry

end remainder_proof_l13_13300


namespace sum_G_correct_l13_13305

def G (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 + 1 else n^2

def sum_G (a b : ℕ) : ℕ :=
  List.sum (List.map G (List.range' a (b - a + 1)))

theorem sum_G_correct :
  sum_G 2 2007 = 8546520 := by
  sorry

end sum_G_correct_l13_13305


namespace two_digit_solution_l13_13139

def two_digit_number (x y : ℕ) : ℕ := 10 * x + y

theorem two_digit_solution :
  ∃ (x y : ℕ), 
    two_digit_number x y = 24 ∧ 
    two_digit_number x y = x^3 + y^2 ∧ 
    0 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 :=
by
  sorry

end two_digit_solution_l13_13139


namespace initial_average_age_is_16_l13_13807

-- Given conditions
variable (N : ℕ) (newPersons : ℕ) (avgNewPersonsAge : ℝ) (totalPersonsAfter : ℕ) (avgAgeAfter : ℝ)
variable (initial_avg_age : ℝ) -- This represents the initial average age (A) we need to prove

-- The specific values from the problem
def N_value : ℕ := 20
def newPersons_value : ℕ := 20
def avgNewPersonsAge_value : ℝ := 15
def totalPersonsAfter_value : ℕ := 40
def avgAgeAfter_value : ℝ := 15.5

-- Theorem statement to prove that the initial average age is 16 years
theorem initial_average_age_is_16 (h1 : N = N_value) (h2 : newPersons = newPersons_value) 
  (h3 : avgNewPersonsAge = avgNewPersonsAge_value) (h4 : totalPersonsAfter = totalPersonsAfter_value) 
  (h5 : avgAgeAfter = avgAgeAfter_value) : initial_avg_age = 16 := by
  sorry

end initial_average_age_is_16_l13_13807


namespace all_xi_equal_l13_13785

theorem all_xi_equal (P : Polynomial ℤ) (n : ℕ) (hn : n % 2 = 1) (x : Fin n → ℤ) 
  (hP : ∀ i : Fin n, P.eval (x i) = x ⟨i + 1, sorry⟩) : 
  ∀ i j : Fin n, x i = x j :=
by
  sorry

end all_xi_equal_l13_13785


namespace smaller_angle_3_40_pm_l13_13064

-- Definitions of the movements of the clock hands and the time condition
def minuteHandDegreesPerMinute : ℝ := 6
def hourHandDegreesPerMinute : ℝ := 0.5
def timeInMinutesSinceNoon : ℕ := 3 * 60 + 40 -- 220 minutes

-- Function to calculate the position of the minute hand at a given time
def minuteHandAngle (minutes: ℕ) : ℝ := minutes * minuteHandDegreesPerMinute

-- Function to calculate the position of the hour hand at a given time
def hourHandAngle (minutes: ℕ) : ℝ := minutes * hourHandDegreesPerMinute

-- Statement of the problem to be proven
theorem smaller_angle_3_40_pm : 
  let angleMinute := minuteHandAngle timeInMinutesSinceNoon,
      angleHour := hourHandAngle timeInMinutesSinceNoon,
      angleDiff := abs (angleMinute - angleHour)
  in (if angleDiff <= 180 then angleDiff else 360 - angleDiff) = 130 :=
by {
  sorry
}

end smaller_angle_3_40_pm_l13_13064


namespace find_f_10_l13_13263

noncomputable def f : ℤ → ℤ := sorry

axiom cond1 : f 1 + 1 > 0
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := 
by
  sorry 

end find_f_10_l13_13263


namespace exists_sum_of_squares_form_l13_13570

theorem exists_sum_of_squares_form (n : ℕ) (h : n % 25 = 9) :
  ∃ (a b c : ℕ), n = (a * (a + 1)) / 2 + (b * (b + 1)) / 2 + (c * (c + 1)) / 2 := 
by 
  sorry

end exists_sum_of_squares_form_l13_13570


namespace james_balloons_l13_13779

-- Definitions
def amy_balloons : ℕ := 513
def extra_balloons_james_has : ℕ := 709

-- Statement of the problem
theorem james_balloons : amy_balloons + extra_balloons_james_has = 1222 :=
by
  -- Placeholder for the actual proof
  sorry

end james_balloons_l13_13779


namespace tan_pi_by_eight_product_l13_13562

theorem tan_pi_by_eight_product :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = -real.sqrt 2 := by 
sorry

end tan_pi_by_eight_product_l13_13562


namespace intersection_eq_l13_13877

def A : Set ℝ := {-1, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def C : Set ℝ := {2}

theorem intersection_eq : A ∩ B = C := 
by {
  sorry
}

end intersection_eq_l13_13877


namespace solve_for_y_l13_13798

theorem solve_for_y : ∀ (y : ℝ), 4 + 2.3 * y = 1.7 * y - 20 → y = -40 :=
by
  sorry

end solve_for_y_l13_13798


namespace radishes_per_row_l13_13421

theorem radishes_per_row 
  (bean_seedlings : ℕ) (beans_per_row : ℕ) 
  (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
  (radishes : ℕ) (rows_per_bed : ℕ) (plant_beds : ℕ)
  (h1 : bean_seedlings = 64) (h2 : beans_per_row = 8)
  (h3 : pumpkin_seeds = 84) (h4 : pumpkins_per_row = 7)
  (h5 : radishes = 48) (h6 : rows_per_bed = 2) (h7 : plant_beds = 14) : 
  (radishes / ((plant_beds * rows_per_bed) - (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row))) = 6 := 
by sorry

end radishes_per_row_l13_13421


namespace phones_left_l13_13964

theorem phones_left (last_year_production : ℕ) 
                    (this_year_production : ℕ) 
                    (sold_phones : ℕ) 
                    (left_phones : ℕ) 
                    (h1 : last_year_production = 5000) 
                    (h2 : this_year_production = 2 * last_year_production) 
                    (h3 : sold_phones = this_year_production / 4) 
                    (h4 : left_phones = this_year_production - sold_phones) : 
                    left_phones = 7500 :=
by
  rw [h1, h2]
  simp only
  rw [h3, h4]
  norm_num
  sorry

end phones_left_l13_13964


namespace Tom_Brady_passing_yards_l13_13216

-- Definitions
def record := 5999
def current_yards := 4200
def games_left := 6

-- Proof problem statement
theorem Tom_Brady_passing_yards :
  (record + 1 - current_yards) / games_left = 300 := by
  sorry

end Tom_Brady_passing_yards_l13_13216


namespace boy_late_l13_13703

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

end boy_late_l13_13703


namespace nonneg_integer_representation_l13_13133

theorem nonneg_integer_representation (n : ℕ) : 
  ∃ x y : ℕ, n = (x + y) * (x + y) + 3 * x + y / 2 := 
sorry

end nonneg_integer_representation_l13_13133


namespace trader_profit_percent_l13_13552

-- Definitions based on the conditions
variables (P : ℝ) -- Original price of the car
def discount_price := 0.95 * P
def taxes := 0.03 * P
def maintenance := 0.02 * P
def total_cost := discount_price + taxes + maintenance 
def selling_price := 0.95 * P * 1.60
def profit := selling_price - total_cost

-- Theorem
theorem trader_profit_percent : (profit P / P) * 100 = 52 :=
by
  sorry

end trader_profit_percent_l13_13552


namespace hearty_buys_red_packages_l13_13483

-- Define the conditions
def packages_of_blue := 3
def beads_per_package := 40
def total_beads := 320

-- Calculate the number of blue beads
def blue_beads := packages_of_blue * beads_per_package

-- Calculate the number of red beads
def red_beads := total_beads - blue_beads

-- Prove that the number of red packages is 5
theorem hearty_buys_red_packages : (red_beads / beads_per_package) = 5 := by
  sorry

end hearty_buys_red_packages_l13_13483


namespace relationship_of_a_b_l13_13897

theorem relationship_of_a_b
  (a b : Real)
  (h1 : a < 0)
  (h2 : b > 0)
  (h3 : a + b < 0) : 
  -a > b ∧ b > -b ∧ -b > a := 
by
  sorry

end relationship_of_a_b_l13_13897


namespace starting_current_ratio_l13_13811

theorem starting_current_ratio (running_current : ℕ) (units : ℕ) (total_current : ℕ)
    (h1 : running_current = 40) 
    (h2 : units = 3) 
    (h3 : total_current = 240) 
    (h4 : total_current = running_current * (units * starter_ratio)) :
    starter_ratio = 2 := 
sorry

end starting_current_ratio_l13_13811


namespace find_min_of_S_l13_13577

noncomputable def S (k : ℝ) : ℝ :=
let x1 := Real.arcsin (k / 2) in
let x3 := π - Real.arcsin (k / 2) in
  (∫ x in 0..x1, k * Real.cos x - Real.sin (2 * x))
  + (∫ x in x1..(π / 2), Real.sin (2 * x) - k * Real.cos x)
  + |∫ x in (π / 2)..x3, k * Real.cos x - Real.sin (2 * x)|
  + |∫ x in x3..π, Real.sin (2 * x) - k * Real.cos x|

theorem find_min_of_S : ∃ k, 0 < k ∧ k < 2 ∧ S k = -1.0577 := 
sorry

end find_min_of_S_l13_13577


namespace sequence_result_l13_13336

theorem sequence_result :
  (1 + 2)^2 + 1 = 10 ∧
  (2 + 3)^2 + 1 = 26 ∧
  (4 + 5)^2 + 1 = 82 →
  (3 + 4)^2 + 1 = 50 :=
by sorry

end sequence_result_l13_13336


namespace sin_75_is_sqrt_6_add_sqrt_2_div_4_l13_13995

noncomputable def sin_75_angle (a : Real) (b : Real) : Real :=
  Real.sin (75 * Real.pi / 180)

theorem sin_75_is_sqrt_6_add_sqrt_2_div_4 :
  sin_75_angle π (π / 6) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end sin_75_is_sqrt_6_add_sqrt_2_div_4_l13_13995


namespace simplify_fraction_l13_13692

theorem simplify_fraction : (1 / (2 + (2/3))) = (3 / 8) :=
by
  sorry

end simplify_fraction_l13_13692


namespace find_side_a_l13_13345

theorem find_side_a
  (A : ℝ) (a b c : ℝ)
  (area : ℝ)
  (hA : A = 60)
  (h_area : area = (3 * real.sqrt 3) / 2)
  (h_bc_sum : b + c = 3 * real.sqrt 3)
  (h_area_formula : area = 1 / 2 * b * c * real.sin (A * real.pi / 180)) :
  a = 3 := by
  have h1 : real.sin (A * real.pi / 180) = real.sqrt 3 / 2, by sorry
  have h2 : (3 * real.sqrt 3) / 2 = 1 / 2 * b * c * (real.sqrt 3 / 2), by sorry
  have h3 : b * c = 6, by sorry
  have h4 : b + c = 3 * real.sqrt 3, by sorry
  have h5 : 3 * real.sqrt 3 * real.sqrt 3 = 27, by sorry
  have h6 : b^2 + c^2 = 3, by sorry
  have h7 : 1 / 2 * (15 - a^2) = 1, by sorry
  have h8 : 15 - a^2 = 6, by sorry
  have h9 : a^2 = 9, by sorry
  have h10 : a = real.sqrt 9, by sorry
  exact h10

end find_side_a_l13_13345


namespace circles_intersect_l13_13660

def circle1 := { x : ℝ × ℝ | (x.1 - 1)^2 + (x.2 + 2)^2 = 1 }
def circle2 := { x : ℝ × ℝ | (x.1 - 2)^2 + (x.2 + 1)^2 = 1 / 4 }

theorem circles_intersect :
  ∃ x : ℝ × ℝ, x ∈ circle1 ∧ x ∈ circle2 :=
sorry

end circles_intersect_l13_13660


namespace complex_expr_simplify_l13_13648

noncomputable def complex_demo : Prop :=
  let i := Complex.I
  7 * (4 + 2 * i) - 2 * i * (7 + 3 * i) = (34 : ℂ)

theorem complex_expr_simplify : 
  complex_demo :=
by
  -- proof skipped
  sorry

end complex_expr_simplify_l13_13648


namespace f_194_l13_13316

noncomputable def f : ℝ → ℝ := sorry -- function definition

theorem f_194 :
  (∀ x : ℝ, f(2 * x - 1) = -f(-(2 * x - 1))) ∧
  (∀ x : ℝ, f(x + 1) = f(-(x + 1))) ∧
  (∀ x : ℝ, x ∈ Ioo (-1 : ℝ) (1 : ℝ) → f(x) = Real.exp x) →
  f(194) = 1 := 
sorry

end f_194_l13_13316


namespace volume_is_correct_l13_13285

noncomputable def volume_of_target_cube (V₁ : ℝ) (A₂ : ℝ) : ℝ :=
  if h₁ : V₁ = 8 then
    let s₁ := (8 : ℝ)^(1/3)
    let A₁ := 6 * s₁^2
    if h₂ : A₂ = 2 * A₁ then
      let s₂ := (A₂ / 6)^(1/2)
      let V₂ := s₂^3
      V₂
    else 0
  else 0

theorem volume_is_correct : volume_of_target_cube 8 48 = 16 * Real.sqrt 2 :=
by
  sorry

end volume_is_correct_l13_13285


namespace smallest_benches_l13_13955

theorem smallest_benches (N : ℕ) (h1 : ∃ n, 8 * n = 40 ∧ 10 * n = 40) : N = 20 :=
sorry

end smallest_benches_l13_13955


namespace value_of_A_l13_13175

-- Definitions for values in the factor tree, ensuring each condition is respected.
def D : ℕ := 3 * 2 * 2
def E : ℕ := 5 * 2
def B : ℕ := 3 * D
def C : ℕ := 5 * E
def A : ℕ := B * C

-- Assertion of the correct value for A
theorem value_of_A : A = 1800 := by
  -- Mathematical equivalence proof problem placeholder
  sorry

end value_of_A_l13_13175


namespace compare_log_values_l13_13582

noncomputable def a : ℝ := (Real.log 2) / 2
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := (Real.log 5) / 5

theorem compare_log_values : c < a ∧ a < b := by
  -- Proof is omitted
  sorry

end compare_log_values_l13_13582


namespace apples_in_pile_l13_13535

/-- Assuming an initial pile of 8 apples and adding 5 more apples, there should be 13 apples in total. -/
theorem apples_in_pile (initial_apples added_apples : ℕ) (h1 : initial_apples = 8) (h2 : added_apples = 5) :
  initial_apples + added_apples = 13 :=
by
  sorry

end apples_in_pile_l13_13535


namespace a5_eq_11_l13_13624

variable (a : ℕ → ℚ) (S : ℕ → ℚ)
variable (n : ℕ) (d : ℚ) (a1 : ℚ)

-- The definitions as given in the conditions
def arithmetic_sequence (a : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

def sum_of_terms (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a1 + (n - 1) * d)

-- Given conditions
def cond1 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 3 + S 3 = 22

def cond2 (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  a 4 - S 4 = -15

-- The statement to prove
theorem a5_eq_11 (a : ℕ → ℚ) (S : ℕ → ℚ) (a1 : ℚ) (d : ℚ)
  (h_arith : arithmetic_sequence a a1 d)
  (h_sum : sum_of_terms S a1 d)
  (h1 : cond1 a S)
  (h2 : cond2 a S) : a 5 = 11 := by
  sorry

end a5_eq_11_l13_13624


namespace inequality_solution_l13_13802

theorem inequality_solution (x : ℝ) :
  2 * (2 * x - 1) > 3 * x - 1 → x > 1 :=
by
  sorry

end inequality_solution_l13_13802


namespace find_largest_integer_l13_13296

theorem find_largest_integer : ∃ (x : ℤ), x < 120 ∧ x % 8 = 7 ∧ x = 119 := 
by
  use 119
  sorry

end find_largest_integer_l13_13296


namespace sum_series_l13_13127

theorem sum_series :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
sorry

end sum_series_l13_13127


namespace probability_Z_l13_13424

variable (p_X p_Y p_Z p_W : ℚ)

def conditions :=
  (p_X = 1/4) ∧ (p_Y = 1/3) ∧ (p_W = 1/6) ∧ (p_X + p_Y + p_Z + p_W = 1)

theorem probability_Z (h : conditions p_X p_Y p_Z p_W) : p_Z = 1/4 :=
by
  obtain ⟨hX, hY, hW, hSum⟩ := h
  sorry

end probability_Z_l13_13424


namespace expression_S_max_value_S_l13_13891

section
variable (x t : ℝ)
def f (x : ℝ) := -3 * x^2 + 6 * x

-- Define the integral expression for S(t)
noncomputable def S (t : ℝ) := ∫ x in t..(t + 1), f x

-- Assert the expression for S(t)
theorem expression_S (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 2) :
  S t = -3 * t^2 + 3 * t + 2 :=
by
  sorry

-- Assert the maximum value of S(t)
theorem max_value_S :
  ∀ t, (0 ≤ t ∧ t ≤ 2) → S t ≤ 5 / 4 :=
by
  sorry

end

end expression_S_max_value_S_l13_13891


namespace find_abc_squared_sum_l13_13360

theorem find_abc_squared_sum (a b c : ℕ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a^3 + 32 * b + 2 * c = 2018) (h₃ : b^3 + 32 * a + 2 * c = 1115) :
  a^2 + b^2 + c^2 = 226 :=
sorry

end find_abc_squared_sum_l13_13360


namespace find_some_number_l13_13701

theorem find_some_number (some_number : ℝ) :
  (0.0077 * some_number) / (0.04 * 0.1 * 0.007) = 990.0000000000001 → 
  some_number = 3.6 :=
by
  intro h
  sorry

end find_some_number_l13_13701


namespace no_unhappy_days_l13_13668

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l13_13668


namespace marching_band_total_weight_l13_13339

noncomputable def total_weight : ℕ :=
  let trumpet_weight := 5
  let clarinet_weight := 5
  let trombone_weight := 10
  let tuba_weight := 20
  let drum_weight := 15
  let trumpets := 6
  let clarinets := 9
  let trombones := 8
  let tubas := 3
  let drummers := 2
  (trumpets + clarinets) * trumpet_weight + trombones * trombone_weight + tubas * tuba_weight + drummers * drum_weight

theorem marching_band_total_weight : total_weight = 245 := by
  sorry

end marching_band_total_weight_l13_13339


namespace diamond_evaluation_l13_13728

def diamond (X Y : ℚ) : ℚ := (2 * X + 3 * Y) / 5

theorem diamond_evaluation : diamond (diamond 3 15) 6 = 192 / 25 := 
by
  sorry

end diamond_evaluation_l13_13728


namespace abs_diff_31st_term_l13_13937

-- Define the sequences C and D
def C (n : ℕ) : ℤ := 40 + 20 * (n - 1)
def D (n : ℕ) : ℤ := 40 - 20 * (n - 1)

-- Question: What is the absolute value of the difference between the 31st term of C and D?
theorem abs_diff_31st_term : |C 31 - D 31| = 1200 := by
  sorry

end abs_diff_31st_term_l13_13937


namespace largest_unorderable_dumplings_l13_13495

theorem largest_unorderable_dumplings : 
  ∀ (a b c : ℕ), 43 ≠ 6 * a + 9 * b + 20 * c :=
by sorry

end largest_unorderable_dumplings_l13_13495


namespace max_subset_A_l13_13790

open Finset

theorem max_subset_A :
  ∃ A ⊆ (Icc 0 29),
    (∀ a b ∈ A, ∀ k : ℤ, ¬ ∃ n : ℤ, a + b + 30 * k = n * (n + 1)) ∧
    ∀ B ⊆ (Icc 0 29),
    (∀ a b ∈ B, ∀ k : ℤ, ¬ ∃ n : ℤ, a + b + 30 * k = n * (n + 1)) →
    card A ≥ card B :=
sorry

end max_subset_A_l13_13790


namespace find_876_last_three_digits_l13_13387

noncomputable def has_same_last_three_digits (N : ℕ) : Prop :=
  (N^2 - N) % 1000 = 0

theorem find_876_last_three_digits (N : ℕ) (h1 : has_same_last_three_digits N) (h2 : N > 99) (h3 : N < 1000) : 
  N % 1000 = 876 :=
sorry

end find_876_last_three_digits_l13_13387


namespace tanner_remaining_money_l13_13806
-- Import the entire Mathlib library

-- Define the conditions using constants
def s_Sep : ℕ := 17
def s_Oct : ℕ := 48
def s_Nov : ℕ := 25
def v_game : ℕ := 49

-- Define the total amount left and prove it equals 41
theorem tanner_remaining_money :
  (s_Sep + s_Oct + s_Nov - v_game) = 41 :=
by { sorry }

end tanner_remaining_money_l13_13806


namespace perfect_square_trinomial_l13_13002

theorem perfect_square_trinomial (k : ℝ) :
  ∃ k, (∀ x, (4 * x^2 - 2 * k * x + 1) = (2 * x + 1)^2 ∨ (4 * x^2 - 2 * k * x + 1) = (2 * x - 1)^2) → 
  (k = 2 ∨ k = -2) := by
  sorry

end perfect_square_trinomial_l13_13002


namespace total_legs_in_household_l13_13348

def number_of_legs (humans children dogs cats : ℕ) (human_legs child_legs dog_legs cat_legs : ℕ) : ℕ :=
  humans * human_legs + children * child_legs + dogs * dog_legs + cats * cat_legs

theorem total_legs_in_household : number_of_legs 2 3 2 1 2 2 4 4 = 22 :=
  by
    -- The statement ensures the total number of legs is 22, given the defined conditions.
    sorry

end total_legs_in_household_l13_13348


namespace average_speed_l13_13212

-- Define the speeds in the first and second hours
def speed_first_hour : ℝ := 90
def speed_second_hour : ℝ := 42

-- Define the time taken for each hour
def time_first_hour : ℝ := 1
def time_second_hour : ℝ := 1

-- Calculate the total distance and total time
def total_distance : ℝ := speed_first_hour + speed_second_hour
def total_time : ℝ := time_first_hour + time_second_hour

-- State the theorem for the average speed
theorem average_speed : total_distance / total_time = 66 := by
  sorry

end average_speed_l13_13212


namespace class_mean_score_l13_13492

theorem class_mean_score:
  ∀ (n: ℕ) (m: ℕ) (a b: ℕ),
  n + m = 50 →
  n * a = 3400 →
  m * b = 750 →
  a = 85 →
  b = 75 →
  (n * a + m * b) / (n + m) = 83 :=
by
  intros n m a b h1 h2 h3 h4 h5
  sorry

end class_mean_score_l13_13492


namespace system_solution_l13_13987

theorem system_solution (x y z : ℝ) 
  (h1 : 2 * x - 3 * y + z = 8) 
  (h2 : 4 * x - 6 * y + 2 * z = 16) 
  (h3 : x + y - z = 1) : 
  x = 11 / 3 ∧ y = 1 ∧ z = 11 / 3 :=
by
  sorry

end system_solution_l13_13987


namespace circle_center_l13_13204

theorem circle_center : ∃ (a b : ℝ), (∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y - 4 = 0 ↔ (x - a)^2 + (y - b)^2 = 9) ∧ a = 1 ∧ b = 2 :=
sorry

end circle_center_l13_13204


namespace percent_same_grades_l13_13335

theorem percent_same_grades 
    (total_students same_A same_B same_C same_D same_E : ℕ)
    (h_total_students : total_students = 40)
    (h_same_A : same_A = 3)
    (h_same_B : same_B = 5)
    (h_same_C : same_C = 6)
    (h_same_D : same_D = 2)
    (h_same_E : same_E = 1):
    ((same_A + same_B + same_C + same_D + same_E : ℚ) / total_students * 100) = 42.5 :=
by
  sorry

end percent_same_grades_l13_13335


namespace product_increase_2022_l13_13626

theorem product_increase_2022 (a b c : ℕ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 678) :
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2022 :=
by {
  -- The proof would go here, but it's not required per the instructions.
  sorry
}

end product_increase_2022_l13_13626


namespace books_sold_on_friday_l13_13630

theorem books_sold_on_friday
  (total_books : ℕ)
  (books_sold_mon : ℕ)
  (books_sold_tue : ℕ)
  (books_sold_wed : ℕ)
  (books_sold_thu : ℕ)
  (pct_unsold : ℚ)
  (initial_stock : total_books = 1400)
  (sold_mon : books_sold_mon = 62)
  (sold_tue : books_sold_tue = 62)
  (sold_wed : books_sold_wed = 60)
  (sold_thu : books_sold_thu = 48)
  (percentage_unsold : pct_unsold = 0.8057142857142857) :
  total_books - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + 40) = total_books * pct_unsold :=
by
  sorry

end books_sold_on_friday_l13_13630


namespace cost_of_dried_fruit_l13_13111

variable (x : ℝ)

theorem cost_of_dried_fruit 
  (h1 : 3 * 12 + 2.5 * x = 56) : 
  x = 8 := 
by 
  sorry

end cost_of_dried_fruit_l13_13111


namespace find_third_root_l13_13984

theorem find_third_root (a b : ℚ) 
  (h1 : a * 1^3 + (a + 3 * b) * 1^2 + (b - 4 * a) * 1 + (6 - a) = 0)
  (h2 : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (6 - a) = 0)
  : ∃ c : ℚ, c = 7 / 13 :=
sorry

end find_third_root_l13_13984


namespace band_weight_correct_l13_13340

universe u

structure InstrumentGroup where
  count : ℕ
  weight_per_instrument : ℕ

def total_weight (ig : InstrumentGroup) : ℕ :=
  ig.count * ig.weight_per_instrument

def total_band_weight : ℕ :=
  (total_weight ⟨6, 5⟩) + (total_weight ⟨9, 5⟩) +
  (total_weight ⟨8, 10⟩) + (total_weight ⟨3, 20⟩) + (total_weight ⟨2, 15⟩)

theorem band_weight_correct : total_band_weight = 245 := by
  rfl

end band_weight_correct_l13_13340


namespace same_color_difference_perfect_square_l13_13460

theorem same_color_difference_perfect_square :
  (∃ (f : ℤ → ℕ) (a b : ℤ), f a = f b ∧ a ≠ b ∧ ∃ (k : ℤ), a - b = k * k) :=
sorry

end same_color_difference_perfect_square_l13_13460


namespace problem1_correct_problem2_correct_l13_13391

noncomputable def problem1_arrangements : ℕ :=
  let boys := 4
  let girls := 3
  let spaces := boys + 1  -- the spaces for girls
  Nat.factorial boys * (Nat.factorial spaces / Nat.factorial (spaces - girls))

theorem problem1_correct : problem1_arrangements = 1440 :=
  sorry

noncomputable def problem2_selections : ℕ :=
  let total := 7
  let boys := 4
  (Nat.choose total 3) - (Nat.choose boys 3)

theorem problem2_correct : problem2_selections = 31 :=
  sorry

end problem1_correct_problem2_correct_l13_13391


namespace find_number_of_flowers_l13_13446
open Nat

theorem find_number_of_flowers (F : ℕ) (h_candles : choose 4 2 = 6) (h_groupings : 6 * choose F 8 = 54) : F = 9 :=
sorry

end find_number_of_flowers_l13_13446


namespace student_ticket_cost_l13_13413

theorem student_ticket_cost (cost_per_student_ticket : ℝ) :
  (12 * cost_per_student_ticket + 4 * 3 = 24) → cost_per_student_ticket = 1 :=
by
  intros h
  -- We should provide a complete proof here, but for illustration, we use sorry.
  sorry

end student_ticket_cost_l13_13413


namespace exists_valid_configuration_l13_13856

-- Define the nine circles
def circles : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the connections (adjacency list) where each connected pair must sum to 23
def lines : List (ℕ × ℕ) := [(1, 8), (8, 6), (8, 9), (9, 2), (2, 7), (7, 6), (7, 4), (4, 1), (4, 5), (5, 6), (5, 3), (6, 3)]

-- The main theorem that we need to prove: there exists a permutation of circles satisfying the line sum condition
theorem exists_valid_configuration: 
  ∃ (f : ℕ → ℕ), 
    (∀ x ∈ circles, f x ∈ circles) ∧ 
    (∀ (a b : ℕ), (a, b) ∈ lines → f a + f b = 23) :=
sorry

end exists_valid_configuration_l13_13856


namespace constant_function_of_inequality_l13_13189

theorem constant_function_of_inequality
  (f : ℤ → ℝ)
  (h_bound : ∃ M : ℝ, ∀ n : ℤ, f n ≤ M)
  (h_ineq : ∀ n : ℤ, f n ≤ (f (n - 1) + f (n + 1)) / 2) :
  ∀ m n : ℤ, f m = f n := by
  sorry

end constant_function_of_inequality_l13_13189


namespace average_weight_of_children_l13_13049

theorem average_weight_of_children
  (S_B S_G : ℕ)
  (avg_boys_weight : S_B = 8 * 160)
  (avg_girls_weight : S_G = 5 * 110) :
  (S_B + S_G) / 13 = 141 := 
by
  sorry

end average_weight_of_children_l13_13049


namespace geom_seq_m_value_l13_13625

/-- Given a geometric sequence {a_n} with a1 = 1 and common ratio q ≠ 1,
    if a_m = a_1 * a_2 * a_3 * a_4 * a_5, then m = 11. -/
theorem geom_seq_m_value (q : ℝ) (h_q : q ≠ 1) :
  ∃ (m : ℕ), (m = 11) ∧ (∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n, a (n + 1) = a n * q ) ∧ (a m = a 1 * a 2 * a 3 * a 4 * a 5)) :=
by
  sorry

end geom_seq_m_value_l13_13625


namespace solve_for_y_l13_13312

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 3
def g (x y : ℝ) : ℝ := 3 * x + y

-- State the theorem to be proven
theorem solve_for_y (x y : ℝ) : 2 * f x - 11 + g x y = f (x - 2) ↔ y = -5 * x + 10 :=
by
  sorry

end solve_for_y_l13_13312


namespace min_value_geq_four_l13_13787

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_geq_four (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  4 ≤ min_value_expression x y z :=
sorry

end min_value_geq_four_l13_13787


namespace correct_factorization_l13_13402

-- Define the conditions from the problem
def conditionA (a b : ℝ) : Prop := a * (a - b) - b * (b - a) = (a - b) * (a + b)
def conditionB (a b : ℝ) : Prop := a^2 - 4 * b^2 = (a + 4 * b) * (a - 4 * b)
def conditionC (a b : ℝ) : Prop := a^2 + 2 * a * b - b^2 = (a + b)^2
def conditionD (a : ℝ) : Prop := a^2 - a - 2 = a * (a - 1) - 2

-- Main theorem statement verifying that only conditionA holds
theorem correct_factorization (a b : ℝ) : 
  conditionA a b ∧ ¬ conditionB a b ∧ ¬ conditionC a b ∧ ¬ conditionD a :=
by 
  sorry

end correct_factorization_l13_13402


namespace min_value_of_frac_l13_13206

theorem min_value_of_frac (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : 2 * m + n = 1) (hm : m > 0) (hn : n > 0) :
  (1 / m) + (2 / n) = 8 :=
sorry

end min_value_of_frac_l13_13206


namespace senior_citizen_tickets_l13_13938

theorem senior_citizen_tickets (A S : ℕ) 
  (h1 : A + S = 510) 
  (h2 : 21 * A + 15 * S = 8748) : 
  S = 327 :=
by 
  -- Proof steps are omitted as instructed
  sorry

end senior_citizen_tickets_l13_13938


namespace largest_even_digit_multiple_of_nine_l13_13822

theorem largest_even_digit_multiple_of_nine : ∃ n : ℕ, (n < 1000) ∧ (∀ d ∈ digits 10 n, d % 2 = 0) ∧ (n % 9 = 0) ∧ n = 888 := 
by
  sorry

end largest_even_digit_multiple_of_nine_l13_13822


namespace find_f_10_l13_13256

variable {f : ℤ → ℤ}

-- Defining the conditions
axiom cond1 : f(1) + 1 > 0
axiom cond2 : ∀ x y : ℤ, f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y
axiom cond3 : ∀ x : ℤ, 2 * f(x) = f(x + 1) - x + 1

-- Goal to prove
theorem find_f_10 : f(10) = 1014 := by
  sorry

end find_f_10_l13_13256


namespace equal_real_roots_value_of_m_l13_13330

theorem equal_real_roots_value_of_m (m : ℝ) (h : (x^2 - 4*x + m = 0)) 
  (discriminant_zero : (16 - 4*m) = 0) : m = 4 :=
sorry

end equal_real_roots_value_of_m_l13_13330


namespace min_value_expr_l13_13634

theorem min_value_expr (x y z w : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) (hw : -2 < w ∧ w < 2) :
  2 ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w / 2)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w / 2))) :=
sorry

end min_value_expr_l13_13634


namespace line_through_point_with_equal_intercepts_l13_13136

/-- A line passing through point (-2, 3) and having equal intercepts
on the coordinate axes can have the equation y = -3/2 * x or x + y = 1. -/
theorem line_through_point_with_equal_intercepts (x y : Real) :
  (∃ (m : Real), (y = m * x) ∧ (y - m * (-2) = 3 ∧ y - m * 0 = 0))
  ∨ (∃ (a : Real), (x + y = a) ∧ (a = 1 ∧ (-2) + 3 = a)) :=
sorry

end line_through_point_with_equal_intercepts_l13_13136


namespace janice_trash_fraction_l13_13347

noncomputable def janice_fraction : ℚ :=
  let homework := 30
  let cleaning := homework / 2
  let walking_dog := homework + 5
  let total_tasks := homework + cleaning + walking_dog
  let total_time := 120
  let time_left := 35
  let time_spent := total_time - time_left
  let trash_time := time_spent - total_tasks
  trash_time / homework

theorem janice_trash_fraction : janice_fraction = 1 / 6 :=
by
  sorry

end janice_trash_fraction_l13_13347


namespace percentage_needed_to_pass_l13_13038

def MikeScore : ℕ := 212
def Shortfall : ℕ := 19
def MaxMarks : ℕ := 770

theorem percentage_needed_to_pass :
  (231.0 / (770.0 : ℝ)) * 100 = 30 := by
  -- placeholder for proof
  sorry

end percentage_needed_to_pass_l13_13038


namespace find_younger_age_l13_13203

def younger_age (y e : ℕ) : Prop :=
  (e = y + 20) ∧ (e - 5 = 5 * (y - 5))

theorem find_younger_age (y e : ℕ) (h : younger_age y e) : y = 10 :=
by sorry

end find_younger_age_l13_13203


namespace max_value_of_seq_diff_l13_13504

theorem max_value_of_seq_diff :
  ∀ (a : Fin 2017 → ℝ),
    a 0 = a 2016 →
    (∀ i : Fin 2015, |a i + a (i+2) - 2 * a (i+1)| ≤ 1) →
    ∃ b : ℝ, b = 508032 ∧ ∀ i j, 1 ≤ i → i < j → j ≤ 2017 → |a i - a j| ≤ b :=
  sorry

end max_value_of_seq_diff_l13_13504


namespace total_market_cost_l13_13637

-- Defining the variables for the problem
def pounds_peaches : Nat := 5 * 3
def pounds_apples : Nat := 4 * 3
def pounds_blueberries : Nat := 3 * 3

def cost_per_pound_peach := 2
def cost_per_pound_apple := 1
def cost_per_pound_blueberry := 1

-- Defining the total costs
def cost_peaches : Nat := pounds_peaches * cost_per_pound_peach
def cost_apples : Nat := pounds_apples * cost_per_pound_apple
def cost_blueberries : Nat := pounds_blueberries * cost_per_pound_blueberry

-- Total cost
def total_cost : Nat := cost_peaches + cost_apples + cost_blueberries

-- Theorem to prove the total cost is $51.00
theorem total_market_cost : total_cost = 51 := by
  sorry

end total_market_cost_l13_13637


namespace probability_of_three_blue_marbles_l13_13706

theorem probability_of_three_blue_marbles
  (red_marbles : ℕ) (blue_marbles : ℕ) (yellow_marbles : ℕ) (total_marbles : ℕ)
  (draws : ℕ) 
  (prob : ℚ) :
  red_marbles = 3 →
  blue_marbles = 4 →
  yellow_marbles = 13 →
  total_marbles = 20 →
  draws = 3 →
  prob = ((4 / 20) * (3 / 19) * (1 / 9)) →
  prob = 1 / 285 :=
by
  intros; 
  sorry

end probability_of_three_blue_marbles_l13_13706


namespace xyz_sum_divisible_l13_13354

-- Define variables and conditions
variable (p x y z : ℕ) [Fact (Prime p)]
variable (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < p)
variable (h_eq1 : x^3 % p = y^3 % p)
variable (h_eq2 : y^3 % p = z^3 % p)

-- Theorem statement
theorem xyz_sum_divisible (p x y z : ℕ) [Fact (Prime p)]
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < p)
  (h_eq1 : x^3 % p = y^3 % p)
  (h_eq2 : y^3 % p = z^3 % p) :
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := 
  sorry

end xyz_sum_divisible_l13_13354


namespace barn_painting_total_area_l13_13415

theorem barn_painting_total_area :
  let width := 12
  let length := 15
  let height := 5
  let divider_width := 12
  let divider_height := 5

  let external_wall_area := 2 * (width * height + length * height)
  let dividing_wall_area := 2 * (divider_width * divider_height)
  let ceiling_area := width * length
  let total_area := 2 * external_wall_area + dividing_wall_area + ceiling_area

  total_area = 840 := by
    sorry

end barn_painting_total_area_l13_13415


namespace rate_of_discount_l13_13235

theorem rate_of_discount (marked_price : ℝ) (selling_price : ℝ) (rate : ℝ)
  (h_marked : marked_price = 125) (h_selling : selling_price = 120)
  (h_rate : rate = ((marked_price - selling_price) / marked_price) * 100) :
  rate = 4 :=
by
  subst h_marked
  subst h_selling
  subst h_rate
  sorry

end rate_of_discount_l13_13235


namespace rows_count_mod_pascals_triangle_l13_13450

-- Define the modified Pascal's triangle function that counts the required rows.
def modified_pascals_triangle_satisfying_rows (n : ℕ) : ℕ := sorry

-- Statement of the problem
theorem rows_count_mod_pascals_triangle :
  modified_pascals_triangle_satisfying_rows 30 = 4 :=
sorry

end rows_count_mod_pascals_triangle_l13_13450


namespace john_money_l13_13910

theorem john_money (cost_given : ℝ) : cost_given = 14 :=
by
  have gift_cost := 28
  have half_cost := gift_cost / 2
  exact sorry

end john_money_l13_13910


namespace rectangle_area_l13_13584

theorem rectangle_area (a b : ℕ) 
  (h1 : 2 * (a + b) = 16)
  (h2 : a^2 + b^2 - 2 * a * b - 4 = 0) :
  a * b = 30 :=
by
  sorry

end rectangle_area_l13_13584


namespace savings_in_cents_l13_13098

def price_local : ℝ := 149.99
def price_payment : ℝ := 26.50
def number_payments : ℕ := 5
def fee_delivery : ℝ := 19.99

theorem savings_in_cents :
  (price_local - (number_payments * price_payment + fee_delivery)) * 100 = -250 := by
  sorry

end savings_in_cents_l13_13098


namespace probability_odd_sum_gt_10_l13_13187

theorem probability_odd_sum_gt_10 :
  let S := {1, 3, 5}
  let T := {2, 4, 6, 8}
  let U := {1, 2, 5}
  (1 / 3 * 1 / 3 * 1 / 9 + 1 / 3 * 1 / 4 * 1 / 9) = 7 / 108 :=
by
  sorry

end probability_odd_sum_gt_10_l13_13187


namespace xiao_ming_correct_answers_l13_13334

theorem xiao_ming_correct_answers :
  ∃ (m n : ℕ), m + n = 20 ∧ 5 * m - n = 76 ∧ m = 16 := 
by
  -- Definitions of points for correct and wrong answers
  let points_per_correct := 5 
  let points_deducted_per_wrong := 1

  -- Contestant's Scores and Conditions
  have contestant_a : 20 * points_per_correct - 0 * points_deducted_per_wrong = 100 := by sorry
  have contestant_b : 19 * points_per_correct - 1 * points_deducted_per_wrong = 94 := by sorry
  have contestant_c : 18 * points_per_correct - 2 * points_deducted_per_wrong = 88 := by sorry
  have contestant_d : 14 * points_per_correct - 6 * points_deducted_per_wrong = 64 := by sorry
  have contestant_e : 10 * points_per_correct - 10 * points_deducted_per_wrong = 40 := by sorry

  -- Xiao Ming's conditions translated to variables m and n
  have xiao_ming_conditions : (∃ m n : ℕ, m + n = 20 ∧ 5 * m - n = 76) := by sorry

  exact ⟨16, 4, rfl, rfl, rfl⟩

end xiao_ming_correct_answers_l13_13334


namespace solve_system_1_solve_system_2_solve_system_3_solve_system_4_l13_13524

-- System 1
theorem solve_system_1 (x y : ℝ) (h1 : x = y + 1) (h2 : 4 * x - 3 * y = 5) : x = 2 ∧ y = 1 :=
by
  sorry

-- System 2
theorem solve_system_2 (x y : ℝ) (h1 : 3 * x + y = 8) (h2 : x - y = 4) : x = 3 ∧ y = -1 :=
by
  sorry

-- System 3
theorem solve_system_3 (x y : ℝ) (h1 : 5 * x + 3 * y = 2) (h2 : 3 * x + 2 * y = 1) : x = 1 ∧ y = -1 :=
by
  sorry

-- System 4
theorem solve_system_4 (x y z : ℝ) (h1 : x + y = 3) (h2 : y + z = -2) (h3 : z + x = 9) : x = 7 ∧ y = -4 ∧ z = 2 :=
by
  sorry

end solve_system_1_solve_system_2_solve_system_3_solve_system_4_l13_13524


namespace percentage_weight_loss_measured_l13_13242

variable (W : ℝ)

def weight_after_loss (W : ℝ) := 0.85 * W
def weight_with_clothes (W : ℝ) := weight_after_loss W * 1.02

theorem percentage_weight_loss_measured (W : ℝ) :
  ((W - weight_with_clothes W) / W) * 100 = 13.3 := by
  sorry

end percentage_weight_loss_measured_l13_13242


namespace original_weight_of_beef_l13_13709

variable (W : ℝ)

def first_stage_weight := 0.80 * W
def second_stage_weight := 0.70 * (first_stage_weight W)
def third_stage_weight := 0.75 * (second_stage_weight W)

theorem original_weight_of_beef :
  third_stage_weight W = 392 → W = 933.33 :=
by
  intro h
  sorry

end original_weight_of_beef_l13_13709


namespace train_speed_l13_13234

-- Define the conditions as given in part (a)
def train_length : ℝ := 160
def crossing_time : ℝ := 6

-- Define the statement to prove
theorem train_speed :
  train_length / crossing_time = 26.67 :=
by
  sorry

end train_speed_l13_13234


namespace series_sum_eq_half_l13_13125

theorem series_sum_eq_half :
  ∑' (n : ℕ), 2^n / (3^(2^n) + 1) = 1 / 2 :=
sorry

end series_sum_eq_half_l13_13125


namespace find_all_functions_l13_13870

theorem find_all_functions 
  (f : ℤ → ℝ)
  (h1 : ∀ m n : ℤ, m < n → f m < f n)
  (h2 : ∀ m n : ℤ, ∃ k : ℤ, f m - f n = f k) :
  ∃ a t : ℝ, a > 0 ∧ (∀ n : ℤ, f n = a * (n + t)) :=
sorry

end find_all_functions_l13_13870


namespace find_f_10_l13_13254

variable {f : ℤ → ℤ}

-- Defining the conditions
axiom cond1 : f(1) + 1 > 0
axiom cond2 : ∀ x y : ℤ, f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y
axiom cond3 : ∀ x : ℤ, 2 * f(x) = f(x + 1) - x + 1

-- Goal to prove
theorem find_f_10 : f(10) = 1014 := by
  sorry

end find_f_10_l13_13254


namespace function_range_is_correct_l13_13662

noncomputable def function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = Real.log (x^2 - 6 * x + 17) }

theorem function_range_is_correct : function_range = {x : ℝ | x ≤ Real.log 8} :=
by
  sorry

end function_range_is_correct_l13_13662


namespace vector_parallel_x_value_l13_13323

theorem vector_parallel_x_value :
  ∀ (x : ℝ), let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  (∃ k : ℝ, b = (k * 3, k * 1)) → x = -9 :=
by
  intro x
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (x, -3)
  intro h
  sorry

end vector_parallel_x_value_l13_13323


namespace solve_for_x_l13_13044

theorem solve_for_x (x : ℚ) :
  (4 * x - 12) / 3 = (3 * x + 6) / 5 → 
  x = 78 / 11 :=
sorry

end solve_for_x_l13_13044


namespace smallest_integer_l13_13328

theorem smallest_integer (k : ℕ) : 
  (∀ (n : ℕ), n = 2^2 * 3^1 * 11^1 → 
  (∀ (f : ℕ), (f = 2^4 ∨ f = 3^3 ∨ f = 13^3) → f ∣ (n * k))) → 
  k = 79092 :=
  sorry

end smallest_integer_l13_13328


namespace largest_n_for_crates_l13_13713

theorem largest_n_for_crates (total_crates : ℕ) (min_oranges max_oranges : ℕ)
  (h1 : total_crates = 145)
  (h2 : min_oranges = 110)
  (h3 : max_oranges = 140) : 
  ∃ n : ℕ, n = 5 ∧ ∀ k : ℕ, k ≤ max_oranges - min_oranges + 1 → total_crates / k ≤ n :=
  by {
    sorry
  }

end largest_n_for_crates_l13_13713


namespace Sam_total_books_l13_13282

/-- Sam's book purchases -/
def Sam_bought_books : Real := 
  let used_adventure_books := 13.0
  let used_mystery_books := 17.0
  let new_crime_books := 15.0
  used_adventure_books + used_mystery_books + new_crime_books

theorem Sam_total_books : Sam_bought_books = 45.0 :=
by
  -- The proof will show that Sam indeed bought 45 books in total
  sorry

end Sam_total_books_l13_13282


namespace roses_remain_unchanged_l13_13392

variable (initial_roses : ℕ) (initial_orchids : ℕ) (final_orchids : ℕ)

def unchanged_roses (roses_now : ℕ) : Prop :=
  roses_now = initial_roses

theorem roses_remain_unchanged :
  initial_roses = 13 → 
  initial_orchids = 84 → 
  final_orchids = 91 →
  ∀ (roses_now : ℕ), unchanged_roses initial_roses roses_now :=
by
  intros _ _ _ _
  simp [unchanged_roses]
  sorry

end roses_remain_unchanged_l13_13392


namespace carla_needs_24_cans_l13_13115

variable (cans_chilis : ℕ) (cans_beans : ℕ) (tomato_multiplier : ℕ) (batch_factor : ℕ)

def cans_tomatoes (cans_beans : ℕ) (tomato_multiplier : ℕ) : ℕ :=
  cans_beans * tomato_multiplier

def normal_batch_cans (cans_chilis : ℕ) (cans_beans : ℕ) (tomato_cans : ℕ) : ℕ :=
  cans_chilis + cans_beans + tomato_cans

def total_cans (normal_cans : ℕ) (batch_factor : ℕ) : ℕ :=
  normal_cans * batch_factor

theorem carla_needs_24_cans : 
  cans_chilis = 1 → 
  cans_beans = 2 → 
  tomato_multiplier = 3 / 2 → 
  batch_factor = 4 → 
  total_cans (normal_batch_cans cans_chilis cans_beans (cans_tomatoes cans_beans tomato_multiplier)) batch_factor = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end carla_needs_24_cans_l13_13115


namespace expectedAdjacentBlackPairs_l13_13252

noncomputable def numberOfBlackPairsInCircleDeck (totalCards blackCards redCards : ℕ) : ℚ := 
  let probBlackNext := (blackCards - 1) / (totalCards - 1)
  blackCards * probBlackNext

theorem expectedAdjacentBlackPairs (totalCards blackCards redCards expectedPairs : ℕ) : 
  totalCards = 52 → 
  blackCards = 30 → 
  redCards = 22 → 
  expectedPairs = 870 / 51 → 
  numberOfBlackPairsInCircleDeck totalCards blackCards redCards = expectedPairs :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end expectedAdjacentBlackPairs_l13_13252


namespace solve_inequality_l13_13140

theorem solve_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by
  sorry

end solve_inequality_l13_13140


namespace minimum_value_of_sum_l13_13791

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
    1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a) >= 3 :=
by
  sorry

end minimum_value_of_sum_l13_13791


namespace smaller_angle_between_hands_at_3_40_l13_13070

noncomputable def smaller_angle (hour minute : ℕ) : ℝ :=
  let minute_angle := minute * 6
  let hour_angle := (hour % 12) * 30 + (minute * 0.5)
  let angle := abs (minute_angle - hour_angle)
  min angle (360 - angle)

theorem smaller_angle_between_hands_at_3_40 : smaller_angle 3 40 = 130.0 := 
by 
  sorry

end smaller_angle_between_hands_at_3_40_l13_13070


namespace find_a_l13_13886

theorem find_a (a : ℝ) : (∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5) → (∃ x : ℝ, x = 3) → a = 8 :=
by
  sorry

end find_a_l13_13886


namespace triangle_angles_l13_13176

-- Define the properties of the triangle
structure Triangle :=
  (a b c h_a h_b : ℝ)
  (altitudes_not_less_than_sides : h_a ≥ a ∧ h_b ≥ b)

-- Define the theorem: Show the angles are 90°, 45°, and 45° if conditions hold
theorem triangle_angles (T : Triangle) : 
  (T.a = T.b) ∧ 
  (T.h_a = T.a) ∧ 
  (T.h_b = T.b) → 
  -- Angles are 90°, 45°, and 45°
  sorry

end triangle_angles_l13_13176


namespace solve_inequality_l13_13451

theorem solve_inequality (x : ℝ) : (x^2 + 5 * x - 14 < 0) ↔ (-7 < x ∧ x < 2) :=
sorry

end solve_inequality_l13_13451


namespace count_triangles_in_figure_l13_13485

noncomputable def triangles_in_figure : ℕ := 53

theorem count_triangles_in_figure : triangles_in_figure = 53 := 
by sorry

end count_triangles_in_figure_l13_13485


namespace altered_solution_contains_correct_detergent_volume_l13_13208

-- Define the original and altered ratios.
def original_ratio : ℝ × ℝ × ℝ := (2, 25, 100)
def altered_ratio_bleach_to_detergent : ℝ × ℝ := (6, 25)
def altered_ratio_detergent_to_water : ℝ × ℝ := (25, 200)

-- Define the given condition about the amount of water in the altered solution.
def altered_solution_water_volume : ℝ := 300

-- Define a function for the total altered solution volume and detergent volume
noncomputable def altered_solution_detergent_volume (water_volume : ℝ) : ℝ :=
  let detergent_volume := (altered_ratio_detergent_to_water.1 * water_volume) / altered_ratio_detergent_to_water.2
  detergent_volume

-- The proof statement asserting the amount of detergent in the altered solution.
theorem altered_solution_contains_correct_detergent_volume :
  altered_solution_detergent_volume altered_solution_water_volume = 37.5 :=
by
  sorry

end altered_solution_contains_correct_detergent_volume_l13_13208


namespace polygon_sides_l13_13168

theorem polygon_sides (n : ℕ) (hn : (n - 2) * 180 = 5 * 360) : n = 12 :=
by
  sorry

end polygon_sides_l13_13168


namespace Mikey_leaves_l13_13513

theorem Mikey_leaves (initial_leaves : ℕ) (leaves_blew_away : ℕ) 
  (h1 : initial_leaves = 356) 
  (h2 : leaves_blew_away = 244) : 
  initial_leaves - leaves_blew_away = 112 :=
by
  -- proof steps would go here
  sorry

end Mikey_leaves_l13_13513


namespace polynomial_is_monic_l13_13700

noncomputable def f : ℝ → ℝ := sorry

variables (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + 6*x - 4)

theorem polynomial_is_monic (f : ℝ → ℝ) (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + x + b) : 
  ∀ x : ℝ, f x = x^2 + 6*x - 4 :=
by sorry

end polynomial_is_monic_l13_13700


namespace original_price_of_book_l13_13388

-- Define the conditions as Lean 4 statements
variable (P : ℝ)  -- Original price of the book
variable (P_new : ℝ := 480)  -- New price of the book
variable (increase_percentage : ℝ := 0.60)  -- Percentage increase in the price

-- Prove the question: original price equals to $300
theorem original_price_of_book :
  P + increase_percentage * P = P_new → P = 300 :=
by
  sorry

end original_price_of_book_l13_13388


namespace abcd_sum_is_12_l13_13579

theorem abcd_sum_is_12 (a b c d : ℤ) 
  (h1 : a + c = 2) 
  (h2 : a * c + b + d = -1) 
  (h3 : a * d + b * c = 18) 
  (h4 : b * d = 24) : 
  a + b + c + d = 12 :=
sorry

end abcd_sum_is_12_l13_13579


namespace travel_same_direction_time_l13_13521

variable (A B : Type) [MetricSpace A] (downstream_speed upstream_speed : ℝ)
  (H_A_downstream_speed : downstream_speed = 8)
  (H_A_upstream_speed : upstream_speed = 4)
  (H_B_downstream_speed : downstream_speed = 8)
  (H_B_upstream_speed : upstream_speed = 4)
  (H_equal_travel_time : (∃ x : ℝ, x * downstream_speed + (3 - x) * upstream_speed = 3)
                      ∧ (∃ x : ℝ, x * upstream_speed + (3 - x) * downstream_speed = 3))

theorem travel_same_direction_time (A_α_downstream B_β_upstream A_α_upstream B_β_downstream : ℝ)
  (H_travel_time : (∃ x : ℝ, x = 1) ∧ (A_α_upstream = 3 - A_α_downstream) ∧ (B_β_downstream = 3 - B_β_upstream)) :
  A_α_downstream = 1 → A_α_upstream = 3 - 1 → B_β_downstream = 1 → B_β_upstream = 3 - 1 → ∃ t, t = 1 :=
by
  sorry

end travel_same_direction_time_l13_13521


namespace number_of_birds_is_122_l13_13459

-- Defining the variables
variables (b m i : ℕ)

-- Define the conditions as part of an axiom
axiom heads_count : b + m + i = 300
axiom legs_count : 2 * b + 4 * m + 6 * i = 1112

-- We aim to prove the number of birds is 122
theorem number_of_birds_is_122 (h1 : b + m + i = 300) (h2 : 2 * b + 4 * m + 6 * i = 1112) : b = 122 := by
  sorry

end number_of_birds_is_122_l13_13459


namespace find_g_zero_l13_13383

variable {g : ℝ → ℝ}

theorem find_g_zero (h : ∀ x y : ℝ, g (x + y) = g x + g y - 1) : g 0 = 1 :=
sorry

end find_g_zero_l13_13383


namespace lines_intersect_at_point_l13_13838

noncomputable def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

noncomputable def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 3 * v, 2 - v)

theorem lines_intersect_at_point :
  ∃ s v : ℚ,
    line1 s = (15 / 7, 16 / 7) ∧
    line2 v = (15 / 7, 16 / 7) ∧
    s = 4 / 7 ∧
    v = -2 / 7 := by
  sorry

end lines_intersect_at_point_l13_13838


namespace phone_extension_permutations_l13_13908

theorem phone_extension_permutations : 
  (∃ (l : List ℕ), l = [5, 7, 8, 9, 0] ∧ Nat.factorial l.length = 120) :=
sorry

end phone_extension_permutations_l13_13908


namespace average_height_males_l13_13803

theorem average_height_males
  (M W H_m : ℝ)
  (h₀ : W ≠ 0)
  (h₁ : M = 2 * W)
  (h₂ : (M * H_m + W * 170) / (M + W) = 180) :
  H_m = 185 := 
sorry

end average_height_males_l13_13803


namespace tan_pi_by_eight_product_l13_13563

theorem tan_pi_by_eight_product :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = -real.sqrt 2 := by 
sorry

end tan_pi_by_eight_product_l13_13563


namespace change_in_us_volume_correct_l13_13925

-- Definition: Change in the total import and export volume of goods in a given year
def change_in_volume (country : String) : Float :=
  if country = "China" then 7.5
  else if country = "United States" then -6.4
  else 0

-- Theorem: The change in the total import and export volume of goods in the United States is correctly represented.
theorem change_in_us_volume_correct :
  change_in_volume "United States" = -6.4 := by
  sorry

end change_in_us_volume_correct_l13_13925


namespace find_coefficient_m_l13_13134

theorem find_coefficient_m :
  ∃ m : ℝ, (1 + 2 * x)^3 = 1 + 6 * x + m * x^2 + 8 * x^3 ∧ m = 12 := by
  sorry

end find_coefficient_m_l13_13134


namespace number_of_continents_collected_l13_13440

-- Definitions of the given conditions
def books_per_continent : ℕ := 122
def total_books : ℕ := 488

-- The mathematical statement to be proved
theorem number_of_continents_collected :
  total_books / books_per_continent = 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_continents_collected_l13_13440


namespace tan_product_l13_13564

theorem tan_product :
  (Real.tan (Real.pi / 8)) * (Real.tan (3 * Real.pi / 8)) * (Real.tan (5 * Real.pi / 8)) = 1 :=
sorry

end tan_product_l13_13564


namespace distance_between_QY_l13_13358

theorem distance_between_QY 
  (m_rate : ℕ) (j_rate : ℕ) (j_distance : ℕ) (headstart : ℕ) 
  (t : ℕ) 
  (h1 : m_rate = 3) 
  (h2 : j_rate = 4) 
  (h3 : j_distance = 24) 
  (h4 : headstart = 1) 
  (h5 : j_distance = j_rate * (t - headstart)) 
  (h6 : t = 7) 
  (distance_m : ℕ := m_rate * t) 
  (distance_j : ℕ := j_distance) :
  distance_j + distance_m = 45 :=
by 
  sorry

end distance_between_QY_l13_13358


namespace mutually_exclusive_shots_proof_l13_13423

/-- Definition of a mutually exclusive event to the event "at most one shot is successful". -/
def mutual_exclusive_at_most_one_shot_successful (both_shots_successful at_most_one_shot_successful : Prop) : Prop :=
  (at_most_one_shot_successful ↔ ¬both_shots_successful)

variable (both_shots_successful : Prop)
variable (at_most_one_shot_successful : Prop)

/-- Given two basketball shots, prove that "both shots are successful" is a mutually exclusive event to "at most one shot is successful". -/
theorem mutually_exclusive_shots_proof : mutual_exclusive_at_most_one_shot_successful both_shots_successful at_most_one_shot_successful :=
  sorry

end mutually_exclusive_shots_proof_l13_13423


namespace mod_inverse_13_1728_l13_13691

theorem mod_inverse_13_1728 :
  (13 * 133) % 1728 = 1 := by
  sorry

end mod_inverse_13_1728_l13_13691


namespace son_age_l13_13839

theorem son_age (S F : ℕ) (h1 : F = S + 30) (h2 : F + 2 = 2 * (S + 2)) : S = 28 :=
by
  sorry

end son_age_l13_13839


namespace part_a_part_b_l13_13723

-- Definitions from conditions
def X := {f : ℝ → ℤ // ∀ x, f x = ⌊x⌋}

-- Part (a) statement
theorem part_a : ∃ E : (ℝ → ℝ) → Prop, 
  (∃ f : ℝ → ℝ, E f ∧ (∀ x, f(x) = ⌊x⌋)) ∧ 
  (∀ f : ℝ → ℝ, E f → (∀ x, f(x) = ⌊x⌋)) :=
sorry

-- Part (b) statement
theorem part_b : ∃ E : (ℝ → ℝ) → Prop,  (∃! f : ℝ → ℝ, E f ∧ (∀ x, f(x) = ⌊x⌋)) :=
sorry

end part_a_part_b_l13_13723


namespace least_subtr_from_12702_to_div_by_99_l13_13229

theorem least_subtr_from_12702_to_div_by_99 : ∃ k : ℕ, 12702 - k = 99 * (12702 / 99) ∧ 0 ≤ k ∧ k < 99 :=
by
  sorry

end least_subtr_from_12702_to_div_by_99_l13_13229


namespace different_quantifiers_not_equiv_l13_13942

theorem different_quantifiers_not_equiv {x₀ : ℝ} :
  (∃ x₀ : ℝ, x₀^2 > 3) ↔ ¬ (∀ x₀ : ℝ, x₀^2 > 3) :=
by
  sorry

end different_quantifiers_not_equiv_l13_13942


namespace point_bisector_second_quadrant_l13_13900

theorem point_bisector_second_quadrant (a : ℝ) : 
  (a < 0 ∧ 2 > 0) ∧ (2 = -a) → a = -2 :=
by sorry

end point_bisector_second_quadrant_l13_13900


namespace polynomial_has_integer_root_l13_13025

noncomputable def P : Polynomial ℤ := sorry

theorem polynomial_has_integer_root
  (P : Polynomial ℤ)
  (h_deg : P.degree = 3)
  (h_infinite_sol : ∀ (x y : ℤ), x ≠ y → x * P.eval x = y * P.eval y → 
  ∃ (x y : ℤ), x ≠ y ∧ x * P.eval x = y * P.eval y) :
  ∃ k : ℤ, P.eval k = 0 :=
sorry

end polynomial_has_integer_root_l13_13025


namespace exterior_angle_sum_l13_13449

theorem exterior_angle_sum (n : ℕ) (h_n : 3 ≤ n) :
  let polygon_exterior_angle_sum := 360
  let triangle_exterior_angle_sum := 0
  (polygon_exterior_angle_sum + triangle_exterior_angle_sum = 360) :=
by sorry

end exterior_angle_sum_l13_13449


namespace sum_series_l13_13128

theorem sum_series :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
sorry

end sum_series_l13_13128


namespace problem_A_correct_problem_B_correct_problem_C_incorrect_problem_D_incorrect_l13_13087

variables (α : ℝ)

theorem problem_A_correct : tan (Real.pi + 1) = tan 1 :=
by 
  -- Using the periodicity of tangent
  exact Real.tan_add_pi 1

theorem problem_B_correct (α : ℝ) : sin (-α) / tan (2 * Real.pi - α) = cos α :=
by 
  -- Simplify using trigonometric identities
  have h1 : sin (-α) = -sin α := Real.sin_neg α,
  have h2 : tan (2 * Real.pi - α) = -tan α := by rw Real.tan_sub (2 * Real.pi) α; exact Real.tan_2pi_sub α,
  rw [h1, h2, neg_div_neg_eq],
  exact (cos_div_sin α).symm

/-
The following theorems are included here for completeness but are shown to be incorrect according to the given solution.
-/

theorem problem_C_incorrect (α : ℝ) : ¬ (sin (Real.pi - α) / cos (Real.pi + α) = tan α) :=
by 
  -- Simplify using trigonometric identities
  have h1 : sin (Real.pi - α) = sin α := Real.sin_pi_sub α,
  have h2 : cos (Real.pi + α) = -cos α := Real.cos_add_pi α,
  rw [h1, h2, neg_div],
  exact (neg_ne_self (tan α)).symm

theorem problem_D_incorrect (α : ℝ) : ¬ (cos (Real.pi - α) * tan(-Real.pi - α) / sin (2 * Real.pi - α) = 1) :=
by 
  -- Simplify using trigonometric identities
  have h1 : cos (Real.pi - α) = -cos α := Real.cos_pi_sub α,
  have h2 : tan (-Real.pi - α) = tan (α - 2 * Real.pi) := Real.tan_sub_pi α,
  have h3 : sin (2 * Real.pi - α) = -sin α := Real.sin_sub_pi α,
  rw [h1, h2, h3],
  -- The simplified form will show it differs from 1
  exact (neg_ne_self (1 : ℝ)).symm

end problem_A_correct_problem_B_correct_problem_C_incorrect_problem_D_incorrect_l13_13087


namespace band_weight_correct_l13_13341

universe u

structure InstrumentGroup where
  count : ℕ
  weight_per_instrument : ℕ

def total_weight (ig : InstrumentGroup) : ℕ :=
  ig.count * ig.weight_per_instrument

def total_band_weight : ℕ :=
  (total_weight ⟨6, 5⟩) + (total_weight ⟨9, 5⟩) +
  (total_weight ⟨8, 10⟩) + (total_weight ⟨3, 20⟩) + (total_weight ⟨2, 15⟩)

theorem band_weight_correct : total_band_weight = 245 := by
  rfl

end band_weight_correct_l13_13341


namespace max_value_of_PQ_l13_13754

noncomputable def maxDistance (P Q : ℝ × ℝ) : ℝ :=
  let dist (a b : ℝ × ℝ) : ℝ := Real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)
  let O1 : ℝ × ℝ := (0, 4)
  dist P Q

theorem max_value_of_PQ:
  ∀ (P Q : ℝ × ℝ),
    (P.1 ^ 2 + (P.2 - 4) ^ 2 = 1) →
    (Q.1 ^ 2 / 9 + Q.2 ^ 2 = 1) →
    maxDistance P Q ≤ 1 + 3 * Real.sqrt 3 :=
by
  sorry

end max_value_of_PQ_l13_13754


namespace complement_intersection_l13_13600

def U : Set ℝ := fun x => True
def A : Set ℝ := fun x => x < 0
def B : Set ℝ := fun x => x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2

theorem complement_intersection (hU : ∀ x : ℝ, U x) :
  ((compl A) ∩ B) = {0, 1, 2} :=
by {
  sorry
}

end complement_intersection_l13_13600


namespace number_of_points_determined_l13_13314

def A : Set ℕ := {5}
def B : Set ℕ := {1, 2}
def C : Set ℕ := {1, 3, 4}

theorem number_of_points_determined : (∃ n : ℕ, n = 33) :=
by
  -- sorry to skip the proof
  sorry

end number_of_points_determined_l13_13314


namespace intersection_A_B_l13_13356

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem intersection_A_B : (A ∩ B) = { x | 0 < x ∧ x ≤ 1 } := by
  sorry

end intersection_A_B_l13_13356


namespace larger_number_of_hcf_23_lcm_factors_13_15_l13_13658

theorem larger_number_of_hcf_23_lcm_factors_13_15 :
  ∃ A B, (Nat.gcd A B = 23) ∧ (A * B = 23 * 13 * 15) ∧ (A = 345 ∨ B = 345) := sorry

end larger_number_of_hcf_23_lcm_factors_13_15_l13_13658


namespace tangent_product_l13_13566

theorem tangent_product (n θ : ℝ) 
  (h1 : ∀ n θ, tan (n * θ) = (sin (n * θ)) / (cos (n * θ)))
  (h2 : tan 8 θ = (8 * tan θ - 56 * (tan θ) ^ 3 + 56 * (tan θ) ^ 5 - 8 * (tan θ) ^ 7) / 
                  (1 - 28 * (tan θ) ^ 2 + 70 * (tan θ) ^ 4 - 28 * (tan θ) ^ 6))
  (h3 : tan (8 * (π / 8)) = 0)
  (h4 : tan (8 * (3 * π / 8)) = 0)
  (h5 : tan (8 * (5 * π / 8)) = 0) :
  tan (π / 8) * tan (3 * π / 8) * tan (5 * π / 8) = 2 * sqrt 2 :=
sorry

end tangent_product_l13_13566


namespace range_of_x_l13_13201

theorem range_of_x (x : ℝ) (hx1 : 1 / x ≤ 3) (hx2 : 1 / x ≥ -2) : x ≥ 1 / 3 := 
sorry

end range_of_x_l13_13201


namespace expected_absolute_difference_after_10_days_l13_13944

def cat_fox_wealth_difference : ℝ := 1

theorem expected_absolute_difference_after_10_days :
  let p_cat_wins := 0.25
  let p_fox_wins := 0.25
  let p_both_police := 0.5
  let num_days := 10
  ∃ (X : ℕ → ℕ), 
    (X 0 = 0) ∧
    ∀ n, (X (n + 1) = (if (X n = 0) then 0.5 else 0) * X n) →
    (∑ k in range (num_days + 1), (k : ℝ) * (0.5 ^ k)) = cat_fox_wealth_difference := 
sorry

end expected_absolute_difference_after_10_days_l13_13944


namespace jellybean_count_l13_13393

theorem jellybean_count (initial_jellybeans : ℕ) (samantha_takes : ℕ) (shelby_eats : ℕ) :
  initial_jellybeans = 90 → samantha_takes = 24 → shelby_eats = 12 →
  let total_taken := samantha_takes + shelby_eats in
  let shannon_refills := total_taken / 2 in
  initial_jellybeans - total_taken + shannon_refills = 72 :=
by
  intros h_initial h_samantha h_shelby
  simp [h_initial, h_samantha, h_shelby]
  let total_taken := 24 + 12
  let shannon_refills := total_taken / 2
  have : (90 - total_taken + shannon_refills) = 72 := by norm_num
  exact this

end jellybean_count_l13_13393


namespace probability_more_boys_than_girls_l13_13770

open ProbabilityTheory MeasureTheory

-- Define the distribution of the number of children
def P_X (n : ℕ) : ℝ :=
  if n = 0 then 1/15 else
  if n = 1 then 6/15 else
  if n = 2 then 6/15 else
  if n = 3 then 2/15 else 0

-- Define the conditional probability P(B|A_i)
def P_B_given_A (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  if n = 1 then 1/2 else
  if n = 2 then 1/4 else
  if n = 3 then 1/2 else 0

-- Define the overall probability P(B) using the law of total probability
def P_B : ℝ :=
  0 * (1/15) + (1/2) * (6/15) + (1/4) * (6/15) + (1/2) * (2/15)

-- The main theorem stating the required probability
theorem probability_more_boys_than_girls : P_B = 11/30 :=
by
  -- This part is just to show the theorem structure.
  -- The actual proof is not provided here.
  sorry

end probability_more_boys_than_girls_l13_13770


namespace toys_per_week_production_l13_13961

-- Define the necessary conditions
def days_per_week : Nat := 4
def toys_per_day : Nat := 1500

-- Define the theorem to prove the total number of toys produced per week
theorem toys_per_week_production : 
  ∀ (days_per_week toys_per_day : Nat), 
    (days_per_week = 4) →
    (toys_per_day = 1500) →
    (days_per_week * toys_per_day = 6000) := 
by
  intros
  sorry

end toys_per_week_production_l13_13961


namespace jerry_total_miles_l13_13629

def monday : ℕ := 15
def tuesday : ℕ := 18
def wednesday : ℕ := 25
def thursday : ℕ := 12
def friday : ℕ := 10

def total : ℕ := monday + tuesday + wednesday + thursday + friday

theorem jerry_total_miles : total = 80 := by
  sorry

end jerry_total_miles_l13_13629


namespace solve_system_of_equations_l13_13652

theorem solve_system_of_equations (x y : ℝ) :
  (3 * x^2 + 4 * x * y + 12 * y^2 + 16 * y = -6) ∧
  (x^2 - 12 * x * y + 4 * y^2 - 10 * x + 12 * y = -7) →
  (x = 1 / 2) ∧ (y = -3 / 4) :=
by
  sorry

end solve_system_of_equations_l13_13652


namespace exists_base_for_1994_no_base_for_1993_l13_13647

-- Problem 1: Existence of a base for 1994 with identical digits
theorem exists_base_for_1994 :
  ∃ b : ℕ, 1 < b ∧ b < 1993 ∧ (∃ a : ℕ, ∀ n : ℕ, 1994 = a * ((b ^ n - 1) / (b - 1)) ∧ a = 2) :=
sorry

-- Problem 2: Non-existence of a base for 1993 with identical digits
theorem no_base_for_1993 :
  ¬∃ b : ℕ, 1 < b ∧ b < 1992 ∧ (∃ a : ℕ, ∀ n : ℕ, 1993 = a * ((b ^ n - 1) / (b - 1))) :=
sorry

end exists_base_for_1994_no_base_for_1993_l13_13647


namespace door_solution_l13_13412

def door_problem (x : ℝ) : Prop :=
  let w := x - 4
  let h := x - 2
  let diagonal := x
  (diagonal ^ 2 - (h) ^ 2 = (w) ^ 2)

theorem door_solution (x : ℝ) : door_problem x :=
  sorry

end door_solution_l13_13412


namespace no_unhappy_days_l13_13671

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l13_13671


namespace aftershave_lotion_volume_l13_13854

theorem aftershave_lotion_volume (V : ℝ) (h1 : 0.30 * V = 0.1875 * (V + 30)) : V = 50 := 
by 
-- sorry is added to indicate proof is omitted.
sorry

end aftershave_lotion_volume_l13_13854


namespace sum_of_first_seven_terms_l13_13164

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given condition
axiom a3_a4_a5_sum : a 3 + a 4 + a 5 = 12

-- Statement to prove
theorem sum_of_first_seven_terms (h : arithmetic_sequence a d) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end sum_of_first_seven_terms_l13_13164


namespace part1_part2_l13_13805

noncomputable def h (x : ℝ) : ℝ := x^2

noncomputable def phi (x : ℝ) : ℝ := 2 * Real.exp 1 * Real.log x

noncomputable def F (x : ℝ) : ℝ := h x - phi x

theorem part1 :
  ∃ (x : ℝ), x > 0 ∧ Real.log x = 1 ∧ F x = 0 :=
sorry

theorem part2 :
  ∃ (k b : ℝ), 
  (∀ x > 0, h x ≥ k * x + b) ∧
  (∀ x > 0, phi x ≤ k * x + b) ∧
  (k = 2 * Real.exp 1 ∧ b = -Real.exp 1) :=
sorry

end part1_part2_l13_13805


namespace rational_cos_rational_k_l13_13293

theorem rational_cos_rational_k (k : ℚ) (h1 : 0 ≤ k ∧ k ≤ 1/2) :
  (cos (k * real.pi)).is_rational ↔ (k = 0 ∨ k = 1/2 ∨ k = 1/3) :=
by sorry

end rational_cos_rational_k_l13_13293


namespace problem1_problem2_l13_13287

theorem problem1 : (Real.sqrt 2) * (Real.sqrt 6) + (Real.sqrt 3) = 3 * (Real.sqrt 3) :=
  sorry

theorem problem2 : (1 - Real.sqrt 2) * (2 - Real.sqrt 2) = 4 - 3 * (Real.sqrt 2) :=
  sorry

end problem1_problem2_l13_13287


namespace range_of_a_l13_13461

theorem range_of_a 
  (a : ℝ):
  (∀ x : ℝ, |x + 2| + |x - 1| > a^2 - 2 * a) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l13_13461


namespace m_value_quadratic_l13_13757

theorem m_value_quadratic (m : ℝ)
  (h1 : |m - 2| = 2)
  (h2 : m - 4 ≠ 0) :
  m = 0 :=
sorry

end m_value_quadratic_l13_13757


namespace range_of_x_l13_13607

theorem range_of_x (x : ℝ) (h : (x + 1) ^ 0 = 1) : x ≠ -1 :=
sorry

end range_of_x_l13_13607


namespace flowers_left_l13_13718

theorem flowers_left (flowers_picked_A : Nat) (flowers_picked_M : Nat) (flowers_given : Nat)
  (h_a : flowers_picked_A = 16)
  (h_m : flowers_picked_M = 16)
  (h_g : flowers_given = 18) :
  flowers_picked_A + flowers_picked_M - flowers_given = 14 :=
by
  sorry

end flowers_left_l13_13718


namespace range_of_m_l13_13479

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ∀ y : ℝ, (2 ≤ x ∧ x ≤ 3) → (3 ≤ y ∧ y ≤ 6) → m * x^2 - x * y + y^2 ≥ 0) ↔ (m ≥ 0) :=
by
  sorry

end range_of_m_l13_13479


namespace current_population_l13_13834

theorem current_population (initial_population deaths_leaving_percentage : ℕ) (current_population : ℕ) :
  initial_population = 3161 → deaths_leaving_percentage = 5 →
  deaths_leaving_percentage / 100 * initial_population + deaths_leaving_percentage * (initial_population - deaths_leaving_percentage / 100 * initial_population) / 100 = initial_population - current_population →
  current_population = 2553 :=
 by
  sorry

end current_population_l13_13834


namespace amusement_park_weekly_revenue_l13_13433

def ticket_price : ℕ := 3
def visitors_mon_to_fri_per_day : ℕ := 100
def visitors_saturday : ℕ := 200
def visitors_sunday : ℕ := 300

theorem amusement_park_weekly_revenue : 
  let total_visitors_weekdays := visitors_mon_to_fri_per_day * 5
  let total_visitors_weekend := visitors_saturday + visitors_sunday
  let total_visitors := total_visitors_weekdays + total_visitors_weekend
  let total_revenue := total_visitors * ticket_price
  total_revenue = 3000 := by
  sorry

end amusement_park_weekly_revenue_l13_13433


namespace red_beads_count_is_90_l13_13100

-- Define the arithmetic sequence for red beads
def red_bead_count (n : ℕ) : ℕ := 2 * n

-- The sum of the first n terms in our sequence
def sum_red_beads (n : ℕ) : ℕ := n * (n + 1)

-- Verify the number of terms n such that the sum of red beads remains under 100
def valid_num_terms : ℕ := Nat.sqrt 99

-- Calculate total number of red beads on the necklace
def total_red_beads : ℕ := sum_red_beads valid_num_terms

theorem red_beads_count_is_90 (num_beads : ℕ) (valid : num_beads = 99) : 
  total_red_beads = 90 :=
by
  -- Proof skipped
  sorry

end red_beads_count_is_90_l13_13100


namespace third_term_binomial_expansion_l13_13085

-- Let a, x be real numbers
variables (a x : ℝ)

-- Binomial theorem term for k = 2
def binomial_term (n k : ℕ) (x y : ℝ) : ℝ :=
  (Nat.choose n k) * x^(n-k) * y^k

theorem third_term_binomial_expansion :
  binomial_term 6 2 (a / Real.sqrt x) (-Real.sqrt x / a^2) = 15 / x :=
by
  sorry

end third_term_binomial_expansion_l13_13085


namespace initial_comparison_discount_comparison_B_based_on_discounted_A_l13_13841

noncomputable section

-- Definitions based on the problem conditions
def A_price (x : ℝ) : ℝ := x
def B_price (x : ℝ) : ℝ := (0.2 * 2 * x + 0.3 * 3 * x + 0.4 * 4 * x) / 3
def A_discount_price (x : ℝ) : ℝ := 0.9 * x

-- Initial comparison
theorem initial_comparison (x : ℝ) (h : 0 < x) : B_price x < A_price x :=
by {
  sorry
}

-- After A's discount comparison
theorem discount_comparison (x : ℝ) (h : 0 < x) : A_discount_price x < B_price x :=
by {
  sorry
}

-- B's price based on A’s discounted price comparison
theorem B_based_on_discounted_A (x : ℝ) (h : 0 < x) : B_price (A_discount_price x) < A_discount_price x :=
by {
  sorry
}

end initial_comparison_discount_comparison_B_based_on_discounted_A_l13_13841


namespace jasper_time_l13_13183

theorem jasper_time {omar_time : ℕ} {omar_height : ℕ} {jasper_height : ℕ} 
  (h1 : omar_time = 12)
  (h2 : omar_height = 240)
  (h3 : jasper_height = 600)
  (h4 : ∃ t : ℕ, t = (jasper_height * omar_time) / (3 * omar_height))
  : t = 10 :=
by sorry

end jasper_time_l13_13183


namespace original_class_strength_l13_13525

theorem original_class_strength 
  (orig_avg_age : ℕ) (new_students_num : ℕ) (new_avg_age : ℕ) 
  (avg_age_decrease : ℕ) (orig_strength : ℕ) :
  orig_avg_age = 40 →
  new_students_num = 12 →
  new_avg_age = 32 →
  avg_age_decrease = 4 →
  (orig_strength + new_students_num) * (orig_avg_age - avg_age_decrease) = orig_strength * orig_avg_age + new_students_num * new_avg_age →
  orig_strength = 12 := 
by
  intros
  sorry

end original_class_strength_l13_13525


namespace tan_alpha_values_l13_13755

theorem tan_alpha_values (α : ℝ) (h : Real.sin α + Real.cos α = 7 / 5) : 
  (Real.tan α = 4 / 3) ∨ (Real.tan α = 3 / 4) := 
  sorry

end tan_alpha_values_l13_13755


namespace complex_square_identity_l13_13722

theorem complex_square_identity (i : ℂ) (h_i_squared : i^2 = -1) :
  i * (1 + i)^2 = -2 :=
by
  sorry

end complex_square_identity_l13_13722


namespace wire_length_approx_is_correct_l13_13430

noncomputable def S : ℝ := 5.999999999999998
noncomputable def L : ℝ := (5 / 2) * S
noncomputable def W : ℝ := S + L

theorem wire_length_approx_is_correct : abs (W - 21) < 1e-16 := by
  sorry

end wire_length_approx_is_correct_l13_13430


namespace which_is_negative_l13_13826

theorem which_is_negative
    (A : ℤ := 2023)
    (B : ℤ := -2023)
    (C : ℚ := 1/2023)
    (D : ℤ := 0) :
    B < 0 :=
by
  sorry

end which_is_negative_l13_13826


namespace bus_speed_excluding_stoppages_l13_13868

theorem bus_speed_excluding_stoppages 
  (v : ℝ) 
  (speed_incl_stoppages : v * 54 / 60 = 45) : 
  v = 50 := 
  by 
    sorry

end bus_speed_excluding_stoppages_l13_13868


namespace probability_top_card_king_l13_13106

theorem probability_top_card_king :
  let total_cards := 52
  let total_kings := 4
  let probability := total_kings / total_cards
  probability = 1 / 13 :=
by
  -- sorry to skip the proof
  sorry

end probability_top_card_king_l13_13106


namespace trigonometric_value_existence_l13_13880

noncomputable def can_be_value_of_tan (n : ℝ) : Prop :=
∃ θ : ℝ, Real.tan θ = n

noncomputable def can_be_value_of_cot (n : ℝ) : Prop :=
∃ θ : ℝ, 1 / Real.tan θ = n

def can_be_value_of_sin (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.sin θ = n

def can_be_value_of_cos (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.cos θ = n

def can_be_value_of_sec (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.cos θ = n

def can_be_value_of_csc (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.sin θ = n

theorem trigonometric_value_existence (n : ℝ) : 
  can_be_value_of_tan n ∧ 
  can_be_value_of_cot n ∧ 
  can_be_value_of_sin n ∧ 
  can_be_value_of_cos n ∧ 
  can_be_value_of_sec n ∧ 
  can_be_value_of_csc n := 
sorry

end trigonometric_value_existence_l13_13880


namespace functional_equation_solution_l13_13466

variable (f : ℝ → ℝ)

-- Declare the conditions as hypotheses
axiom cond1 : ∀ x : ℝ, 0 < x → 0 < f x
axiom cond2 : f 1 = 1
axiom cond3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

-- State the theorem to be proved
theorem functional_equation_solution : ∀ x : ℝ, f x = x :=
sorry

end functional_equation_solution_l13_13466


namespace largest_common_divisor_476_330_l13_13818

theorem largest_common_divisor_476_330 :
  ∀ (S₁ S₂ : Finset ℕ), 
    S₁ = {1, 2, 4, 7, 14, 28, 17, 34, 68, 119, 238, 476} → 
    S₂ = {1, 2, 3, 5, 6, 10, 11, 15, 22, 30, 33, 55, 66, 110, 165, 330} → 
    ∃ D, D ∈ S₁ ∧ D ∈ S₂ ∧ ∀ x, x ∈ S₁ ∧ x ∈ S₂ → x ≤ D ∧ D = 2 :=
by
  intros S₁ S₂ hS₁ hS₂
  use 2
  sorry

end largest_common_divisor_476_330_l13_13818


namespace positive_integer_solutions_value_of_m_when_sum_is_zero_fixed_solution_integer_values_of_m_l13_13157

-- Definitions for the conditions
def eq1 (x y : ℝ) := x + 2 * y = 6
def eq2 (x y m : ℝ) := x - 2 * y + m * x + 5 = 0

-- Theorem for part (1)
theorem positive_integer_solutions :
  {x y : ℕ} → eq1 x y → (x = 4 ∧ y = 1) ∨ (x = 2 ∧ y = 2) :=
sorry

-- Theorem for part (2)
theorem value_of_m_when_sum_is_zero (x y : ℝ) (h : x + y = 0) :
  eq1 x y → ∃ m : ℝ, eq2 x y m → m = -13/6 :=
sorry

-- Theorem for part (3)
theorem fixed_solution (m : ℝ) : eq2 0 2.5 m :=
sorry

-- Theorem for part (4)
theorem integer_values_of_m (x : ℤ) :
  (∃ y : ℤ, eq1 x y ∧ ∃ m : ℤ, eq2 x y m) → m = -1 ∨ m = -3 :=
sorry

end positive_integer_solutions_value_of_m_when_sum_is_zero_fixed_solution_integer_values_of_m_l13_13157


namespace diff_reading_math_homework_l13_13642

-- Define the conditions as given in the problem
def pages_math_homework : ℕ := 3
def pages_reading_homework : ℕ := 4

-- The statement to prove that Rachel had 1 more page of reading homework than math homework
theorem diff_reading_math_homework : pages_reading_homework - pages_math_homework = 1 := by
  sorry

end diff_reading_math_homework_l13_13642


namespace smaller_angle_3_40_l13_13081

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l13_13081


namespace distance_to_place_l13_13237

theorem distance_to_place (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) (D : ℝ) :
  rowing_speed = 5 ∧ current_speed = 1 ∧ total_time = 1 →
  D = 2.4 :=
by
  -- Rowing Parameters
  let V_d := rowing_speed + current_speed
  let V_u := rowing_speed - current_speed
  
  -- Time Variables
  let T_d := total_time / (V_d + V_u)
  let T_u := total_time - T_d

  -- Distance Calculations
  let D1 := V_d * T_d
  let D2 := V_u * T_u

  -- Prove D is the same distance both upstream and downstream
  sorry

end distance_to_place_l13_13237


namespace pencils_initial_count_l13_13350

theorem pencils_initial_count (pencils_given : ℕ) (pencils_left : ℕ) (initial_pencils : ℕ) :
  pencils_given = 31 → pencils_left = 111 → initial_pencils = pencils_given + pencils_left → initial_pencils = 142 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end pencils_initial_count_l13_13350


namespace rocky_first_round_knockouts_l13_13643

theorem rocky_first_round_knockouts
  (total_fights : ℕ)
  (knockout_percentage : ℝ)
  (first_round_knockout_percentage : ℝ)
  (h1 : total_fights = 190)
  (h2 : knockout_percentage = 0.50)
  (h3 : first_round_knockout_percentage = 0.20) :
  (total_fights * knockout_percentage * first_round_knockout_percentage = 19) := 
by
  sorry

end rocky_first_round_knockouts_l13_13643


namespace range_of_a_l13_13024

theorem range_of_a (a : ℝ) (x1 x2 : ℝ)
  (h_poly: ∀ x, x * x + (a * a - 1) * x + (a - 2) = 0 → x = x1 ∨ x = x2)
  (h_order: x1 < 1 ∧ 1 < x2) : 
  -2 < a ∧ a < 1 := 
sorry

end range_of_a_l13_13024


namespace geometric_sequence_first_term_l13_13788

theorem geometric_sequence_first_term (S_3 S_6 : ℝ) (a_1 q : ℝ)
  (hS3 : S_3 = 6) (hS6 : S_6 = 54)
  (hS3_def : S_3 = a_1 * (1 - q^3) / (1 - q))
  (hS6_def : S_6 = a_1 * (1 - q^6) / (1 - q)) :
  a_1 = 6 / 7 := 
by
  sorry

end geometric_sequence_first_term_l13_13788


namespace max_possible_value_e_l13_13508

def b (n : ℕ) : ℕ := (7^n - 1) / 6

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n+1))

theorem max_possible_value_e (n : ℕ) : e n = 1 := by
  sorry

end max_possible_value_e_l13_13508


namespace expected_difference_after_10_days_l13_13947

noncomputable def cat_fox_expected_difference : ℕ → ℝ
| 0     := 0
| (n+1) := 0.25 * (cat_fox_expected_difference n + 1)  -- cat wins
                + 0.25 * (cat_fox_expected_difference n + 1)  -- fox wins
                + 0.5 * 0  -- both go to police, difference resets

theorem expected_difference_after_10_days :
  cat_fox_expected_difference 10 = 1 :=
sorry

end expected_difference_after_10_days_l13_13947


namespace cos_2beta_value_l13_13472

theorem cos_2beta_value (α β : ℝ) 
  (h1 : Real.sin (α - β) = 3/5) 
  (h2 : Real.cos (α + β) = -3/5) 
  (h3 : α - β ∈ Set.Ioo (π/2) π) 
  (h4 : α + β ∈ Set.Ioo (π/2) π) : 
  Real.cos (2 * β) = 24/25 := 
sorry

end cos_2beta_value_l13_13472


namespace xy_value_l13_13923

theorem xy_value (x y : ℝ) (h : x ≠ y) (h_eq : x^2 + 2 / x^2 = y^2 + 2 / y^2) : 
  x * y = Real.sqrt 2 ∨ x * y = -Real.sqrt 2 :=
by
  sorry

end xy_value_l13_13923


namespace smaller_angle_at_3_40_l13_13073

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l13_13073


namespace program_final_value_l13_13209

-- Define the program execution in a Lean function
def program_result (i : ℕ) (S : ℕ) : ℕ :=
  if i < 9 then S
  else program_result (i - 1) (S * i)

-- Initial conditions
def initial_i := 11
def initial_S := 1

-- The theorem to prove
theorem program_final_value : program_result initial_i initial_S = 990 := by
  sorry

end program_final_value_l13_13209


namespace S_is_finite_l13_13789

def A : Set (ℕ → ℕ) := {f | ∀ n, f n ∈ {i : ℕ | i < 2018}}

def begins_with (M T : ℕ → ℕ) (n : ℕ) : Prop := 
  ∀ i < n, M i = T i

def S (s : ℕ → ℕ) : Set (ℕ → ℕ) := {t | ∃ n, begins_with t s n}

theorem S_is_finite (S : Set (ℕ → ℕ)) : 
  (∀ (M ∈ A), ∃! T ∈ S, ∃ n, begins_with M T n) → 
  S.finite := 
sorry

end S_is_finite_l13_13789


namespace farmer_cows_more_than_goats_l13_13420

-- Definitions of the variables
variables (C P G x : ℕ)

-- Conditions given in the problem
def twice_as_many_pigs_as_cows : Prop := P = 2 * C
def more_cows_than_goats : Prop := C = G + x
def goats_count : Prop := G = 11
def total_animals : Prop := C + P + G = 56

-- The theorem to prove
theorem farmer_cows_more_than_goats
  (h1 : twice_as_many_pigs_as_cows C P)
  (h2 : more_cows_than_goats C G x)
  (h3 : goats_count G)
  (h4 : total_animals C P G) :
  C - G = 4 :=
sorry

end farmer_cows_more_than_goats_l13_13420


namespace internal_angle_bisectors_concurrent_l13_13726

theorem internal_angle_bisectors_concurrent
  (A B C D P : EuclideanSpace ℝ (Fin 3))
  (h_convex : convex_hull ℝ ({A, B, C, D} : Set (EuclideanSpace ℝ (Fin 3))) )
  (h_angles : ∃ (α β : ℝ), ∠PAD = 1 * α ∧ ∠PBA = 2 * α ∧ ∠DPA = 3 * α ∧
                            ∠CBP = 1 * β ∧ ∠BAP = 2 * β ∧ ∠BPC = 3 * β) :
  ∃ O : EuclideanSpace ℝ (Fin 3), 
    is_circumcenter O A B P ∧
    (O ∈ internal_angle_bisectors (triangle.mk D P A) ) ∧
    (O ∈ internal_angle_bisectors (triangle.mk P C B) ) ∧
    (O ∈ perpendicular_bisector (segment A B)) :=
sorry

end internal_angle_bisectors_concurrent_l13_13726


namespace total_pink_crayons_l13_13034

-- Define the conditions
def Mara_crayons : ℕ := 40
def Mara_pink_percent : ℕ := 10
def Luna_crayons : ℕ := 50
def Luna_pink_percent : ℕ := 20

-- Define the proof problem statement
theorem total_pink_crayons : 
  (Mara_crayons * Mara_pink_percent / 100) + (Luna_crayons * Luna_pink_percent / 100) = 14 := 
by sorry

end total_pink_crayons_l13_13034


namespace C_pow_50_l13_13353

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem C_pow_50 :
  C ^ 50 = !![-299, -100; 800, 249] := by
  sorry

end C_pow_50_l13_13353


namespace problem_inequality_l13_13515

variable (a b : ℝ)

theorem problem_inequality (h_pos : 0 < a) (h_pos' : 0 < b) (h_sum : a + b = 1) :
  (1 / a^2 - a^3) * (1 / b^2 - b^3) ≥ (31 / 8)^2 := 
  sorry

end problem_inequality_l13_13515


namespace sum_placed_on_SI_l13_13698

theorem sum_placed_on_SI :
  let P₁ := 4000
  let r₁ := 0.10
  let t₁ := 2
  let CI := P₁ * ((1 + r₁)^t₁ - 1)

  let SI := (1 / 2 * CI : ℝ)
  let r₂ := 0.08
  let t₂ := 3
  let P₂ := SI / (r₂ * t₂)

  P₂ = 1750 :=
by
  sorry

end sum_placed_on_SI_l13_13698


namespace max_min_values_monotonocity_l13_13317

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 - (1 / 2) * x ^ 2

theorem max_min_values (a : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (ha : a = 1) : 
  f a 0 = 0 ∧ f a 1 = 1 / 2 ∧ f a (1 / 3) = -1 / 54 :=
sorry

theorem monotonocity (a : ℝ) (hx : 0 < x ∧ x < (1 / (6 * a))) (ha : 0 < a) : 
  (3 * a * x ^ 2 - x) < 0 → (f a x) < (f a 0) :=
sorry

end max_min_values_monotonocity_l13_13317


namespace abs_diff_probs_l13_13013

def numRedMarbles := 1000
def numBlackMarbles := 1002
def totalMarbles := numRedMarbles + numBlackMarbles

def probSame : ℚ := 
  ((numRedMarbles * (numRedMarbles - 1)) / 2 + (numBlackMarbles * (numBlackMarbles - 1)) / 2) / (totalMarbles * (totalMarbles - 1) / 2)

def probDiff : ℚ :=
  (numRedMarbles * numBlackMarbles) / (totalMarbles * (totalMarbles - 1) / 2)

theorem abs_diff_probs : |probSame - probDiff| = 999 / 2003001 := 
by {
  sorry
}

end abs_diff_probs_l13_13013


namespace pupils_sent_up_exam_l13_13047

theorem pupils_sent_up_exam (average_marks : ℕ) (specific_scores : List ℕ) (new_average : ℕ) : 
  (average_marks = 39) → 
  (specific_scores = [25, 12, 15, 19]) → 
  (new_average = 44) → 
  ∃ n : ℕ, (n > 4) ∧ (average_marks * n) = 39 * n ∧ ((39 * n - specific_scores.sum) / (n - specific_scores.length)) = new_average →
  n = 21 :=
by
  intros h_avg h_scores h_new_avg
  sorry

end pupils_sent_up_exam_l13_13047


namespace emily_chairs_count_l13_13462

theorem emily_chairs_count 
  (C : ℕ) 
  (T : ℕ) 
  (time_per_furniture : ℕ)
  (total_time : ℕ) 
  (hT : T = 2) 
  (h_time : time_per_furniture = 8) 
  (h_total : 8 * C + 8 * T = 48) : 
  C = 4 := by
    sorry

end emily_chairs_count_l13_13462


namespace another_seat_in_sample_l13_13289

-- Definition of the problem
def total_students := 56
def sample_size := 4
def sample_set : Finset ℕ := {3, 17, 45}

-- Lean 4 statement for the proof problem
theorem another_seat_in_sample :
  (sample_set = sample_set ∪ {31}) ∧
  (31 ∉ sample_set) ∧
  (∀ x ∈ sample_set ∪ {31}, x ≤ total_students) :=
by
  sorry

end another_seat_in_sample_l13_13289


namespace solution_for_g0_l13_13382

variable (g : ℝ → ℝ)

def functional_eq_condition := ∀ x y : ℝ, g (x + y) = g x + g y - 1

theorem solution_for_g0 (h : functional_eq_condition g) : g 0 = 1 :=
by {
  sorry
}

end solution_for_g0_l13_13382


namespace cost_of_adult_ticket_l13_13815

theorem cost_of_adult_ticket
    (child_ticket_cost : ℝ)
    (total_tickets : ℕ)
    (total_receipts : ℝ)
    (adult_tickets_sold : ℕ)
    (A : ℝ)
    (child_tickets_sold : ℕ := total_tickets - adult_tickets_sold)
    (total_revenue_adult : ℝ := adult_tickets_sold * A)
    (total_revenue_child : ℝ := child_tickets_sold * child_ticket_cost) :
    child_ticket_cost = 4 →
    total_tickets = 130 →
    total_receipts = 840 →
    adult_tickets_sold = 90 →
    total_revenue_adult + total_revenue_child = total_receipts →
    A = 7.56 :=
by
  intros
  sorry

end cost_of_adult_ticket_l13_13815


namespace segment_parallel_l13_13655

open EuclideanGeometry

variables {P I : Point} {A B C A1 C1 A0 C0 : Point} {ABC : Triangle}

-- The conditions
def angle_bisectors_conditions (ABC : Triangle) (A1 C1 A0 C0 P I : Point) :=
  ABC.is_angle_bisector A A0 ∧ ABC.is_angle_bisector C C0 ∧
  incidence_geometry.line_through (A1 : Point) (A0 : Point) ∧
  incidence_geometry.line_through (C1 : Point) (C0 : Point) ∧
  incidence_geometry.line (A1C1: Line) ∧ incidence_geometry.line (A0C0: Line) ∧
  A1C1.inter_Lines A0C0 = P ∧ ABC.incenter = I

-- The statement to prove
theorem segment_parallel (ABC : Triangle) (A B C A1 C1 A0 C0 P I : Point)
  (h_cond : angle_bisectors_conditions ABC A1 C1 A0 C0 P I) : P.I ∥ A.C := 
sorry

end segment_parallel_l13_13655


namespace annie_extracurricular_hours_l13_13973

-- Definitions based on conditions
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3
def weeks_per_semester : ℕ := 12
def weeks_off_sick : ℕ := 2

-- Total hours of extracurricular activities per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Number of active weeks before midterms
def active_weeks_before_midterms : ℕ := weeks_per_semester - weeks_off_sick

-- Total hours of extracurricular activities before midterms
def total_hours_before_midterms : ℕ := total_hours_per_week * active_weeks_before_midterms

-- Proof statement
theorem annie_extracurricular_hours : total_hours_before_midterms = 130 := by
  sorry

end annie_extracurricular_hours_l13_13973


namespace distinct_sequences_count_l13_13890

noncomputable def number_of_distinct_sequences (n : ℕ) : ℕ :=
  if n = 6 then 12 else sorry

theorem distinct_sequences_count : number_of_distinct_sequences 6 = 12 := 
by 
  sorry

end distinct_sequences_count_l13_13890


namespace intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l13_13050

variable (a b : ℝ)

-- Define the equations given in the problem
def line1 := ∀ y : ℝ, 3 = (1/3) * y + a
def line2 := ∀ x : ℝ, 3 = (1/3) * x + b

-- The Lean statement for the proof
theorem intersecting_lines_at_3_3_implies_a_plus_b_eq_4 :
  (line1 3) ∧ (line2 3) → a + b = 4 :=
by 
  sorry

end intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l13_13050


namespace minimum_cardinality_of_three_sets_l13_13654

open Set

noncomputable def min_intersection_cardinality (X Y Z : Type) [Fintype X] [Fintype Y] [Fintype Z]
  (h1 : Fintype.card X + Fintype.card Y + Fintype.card Z = Fintype.card (X ∪ Y ∪ Z))
  (h2 : Fintype.card X = 50)
  (h3 : Fintype.card Y = 50)
  (h4 : (X ∩ Y ∩ Z).Nonempty) : ℕ :=
  min |X ∩ Y ∩ Z|

theorem minimum_cardinality_of_three_sets (X Y Z : Type) [Fintype X] [Fintype Y] [Fintype Z]
  (h1 : Fintype.card X + Fintype.card Y + Fintype.card Z = Fintype.card (X ∪ Y ∪ Z))
  (h2 : Fintype.card X = 50)
  (h3 : Fintype.card Y = 50)
  (h4 : (X ∩ Y ∩ Z).Nonempty) : min_intersection_cardinality X Y Z h1 h2 h3 h4 = 1 := sorry

end minimum_cardinality_of_three_sets_l13_13654


namespace probability_of_2_red_1_black_l13_13685

theorem probability_of_2_red_1_black :
  let P_red := 4 / 7
  let P_black := 3 / 7 
  let prob_RRB := P_red * P_red * P_black 
  let prob_RBR := P_red * P_black * P_red 
  let prob_BRR := P_black * P_red * P_red 
  let total_prob := 3 * prob_RRB
  total_prob = 144 / 343 :=
by
  sorry

end probability_of_2_red_1_black_l13_13685


namespace find_n_l13_13162

theorem find_n (n m : ℕ) (h : m = 4) (eq1 : (1/5)^m * (1/4)^n = 1/(10^4)) : n = 2 :=
by
  sorry

end find_n_l13_13162


namespace t_shirts_left_yesterday_correct_l13_13926

-- Define the conditions
def t_shirts_left_yesterday (x : ℕ) : Prop :=
  let t_shirts_sold_morning := (3 / 5) * x
  let t_shirts_sold_afternoon := 180
  t_shirts_sold_morning = t_shirts_sold_afternoon

-- Prove that x = 300 given the above conditions
theorem t_shirts_left_yesterday_correct (x : ℕ) (h : t_shirts_left_yesterday x) : x = 300 :=
by
  sorry

end t_shirts_left_yesterday_correct_l13_13926


namespace absolute_difference_center_l13_13052

theorem absolute_difference_center (x1 y1 x2 y2 : ℝ) 
 (h1: x1 = 8) (h2: y1 = -7) (h3: x2 = -4) (h4: y2 = 5) : 
|((x1 + x2) / 2 - (y1 + y2) / 2)| = 3 :=
by
  sorry

end absolute_difference_center_l13_13052


namespace no_square_number_divisible_by_six_in_range_l13_13464

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ (x : ℕ), (x ^ 2) % 6 = 0 ∧ 39 < x ^ 2 ∧ x ^ 2 < 120 :=
by
  sorry

end no_square_number_divisible_by_six_in_range_l13_13464


namespace final_combined_price_correct_l13_13812

theorem final_combined_price_correct :
  let i_p := 1000
  let d_1 := 0.10
  let d_2 := 0.20
  let t_1 := 0.08
  let t_2 := 0.06
  let s_p := 30
  let c_p := 50
  let t_a := 0.05
  let price_after_first_month := i_p * (1 - d_1) * (1 + t_1)
  let price_after_second_month := price_after_first_month * (1 - d_2) * (1 + t_2)
  let screen_protector_final := s_p * (1 + t_a)
  let case_final := c_p * (1 + t_a)
  price_after_second_month + screen_protector_final + case_final = 908.256 := by
  sorry  -- Proof not required

end final_combined_price_correct_l13_13812


namespace time_fraction_reduced_l13_13253

theorem time_fraction_reduced (T D : ℝ) (h1 : D = 30 * T) :
  D = 40 * ((3/4) * T) → 1 - (3/4) = 1/4 :=
sorry

end time_fraction_reduced_l13_13253


namespace total_hours_eq_52_l13_13977

def hours_per_week_on_extracurriculars : ℕ := 2 + 8 + 3  -- Total hours per week
def weeks_in_semester : ℕ := 12  -- Total weeks in a semester
def weeks_before_midterms : ℕ := weeks_in_semester / 2  -- Weeks before midterms
def sick_weeks : ℕ := 2  -- Weeks Annie takes off sick
def active_weeks_before_midterms : ℕ := weeks_before_midterms - sick_weeks  -- Active weeks before midterms

def total_extracurricular_hours_before_midterms : ℕ :=
  hours_per_week_on_extracurriculars * active_weeks_before_midterms

theorem total_hours_eq_52 :
  total_extracurricular_hours_before_midterms = 52 :=
by
  sorry

end total_hours_eq_52_l13_13977


namespace carla_chili_cans_l13_13114

theorem carla_chili_cans :
  let chilis_per_batch := 1
  let beans_per_batch := 2
  let tomatoes_per_batch := 1.5 * beans_per_batch
  let total_per_batch := chilis_per_batch + beans_per_batch + tomatoes_per_batch
  let quadruple_batch := 4 * total_per_batch
  in quadruple_batch = 24 := 
by
  let chilis_per_batch := 1
  let beans_per_batch := 2
  let tomatoes_per_batch := 1.5 * beans_per_batch
  let total_per_batch := chilis_per_batch + beans_per_batch + tomatoes_per_batch
  let quadruple_batch := 4 * total_per_batch
  show quadruple_batch = 24
  sorry

end carla_chili_cans_l13_13114


namespace brie_clothes_washer_l13_13230

theorem brie_clothes_washer (total_blouses total_skirts total_slacks : ℕ)
  (blouses_pct skirts_pct slacks_pct : ℝ)
  (h_blouses : total_blouses = 12)
  (h_skirts : total_skirts = 6)
  (h_slacks : total_slacks = 8)
  (h_blouses_pct : blouses_pct = 0.75)
  (h_skirts_pct : skirts_pct = 0.5)
  (h_slacks_pct : slacks_pct = 0.25) :
  let blouses_in_hamper := total_blouses * blouses_pct
  let skirts_in_hamper := total_skirts * skirts_pct
  let slacks_in_hamper := total_slacks * slacks_pct
  blouses_in_hamper + skirts_in_hamper + slacks_in_hamper = 14 := 
by
  sorry

end brie_clothes_washer_l13_13230


namespace card_area_after_reduction_width_l13_13720

def initial_length : ℕ := 5
def initial_width : ℕ := 8
def new_width := initial_width - 2
def expected_new_area : ℕ := 24

theorem card_area_after_reduction_width :
  initial_length * new_width = expected_new_area := 
by
  -- initial_length = 5, new_width = 8 - 2 = 6
  -- 5 * 6 = 30, which was corrected to 24 given the misinterpretation mentioned.
  sorry

end card_area_after_reduction_width_l13_13720


namespace set_intersection_l13_13505

def U : Set ℤ := {-1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1, 2}

theorem set_intersection :
  (U \ A) ∩ B = {1, 2} :=
by
  sorry

end set_intersection_l13_13505


namespace largest_integer_n_neg_quad_expr_l13_13737

theorem largest_integer_n_neg_quad_expr :
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, ((n^2 - 11 * n + 28 < 0) → (m < 7 ∧ m > 4) → m ≤ n) :=
by
  sorry

end largest_integer_n_neg_quad_expr_l13_13737


namespace clock_angle_l13_13071

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l13_13071


namespace probability_sequence_rw_10_l13_13837

noncomputable def probability_red_white_red : ℚ :=
  (4 / 10) * (6 / 9) * (3 / 8)

theorem probability_sequence_rw_10 :
    probability_red_white_red = 1 / 10 := by
  sorry

end probability_sequence_rw_10_l13_13837


namespace temperature_on_fourth_day_l13_13390

theorem temperature_on_fourth_day
  (t₁ t₂ t₃ : ℤ) 
  (avg : ℤ)
  (h₁ : t₁ = -36) 
  (h₂ : t₂ = 13) 
  (h₃ : t₃ = -10) 
  (h₄ : avg = -12) 
  : ∃ t₄ : ℤ, t₄ = -15 :=
by
  sorry

end temperature_on_fourth_day_l13_13390


namespace sam_dimes_proof_l13_13920

def initial_dimes : ℕ := 9
def remaining_dimes : ℕ := 2
def dimes_given : ℕ := 7

theorem sam_dimes_proof : initial_dimes - remaining_dimes = dimes_given :=
by
  sorry

end sam_dimes_proof_l13_13920


namespace matrix_count_l13_13299

-- A definition for the type of 3x3 matrices with 1's on the diagonal and * can be 0 or 1
def valid_matrix (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 0 = 1 ∧ 
  m 1 1 = 1 ∧ 
  m 2 2 = 1 ∧ 
  (m 0 1 = 0 ∨ m 0 1 = 1) ∧
  (m 0 2 = 0 ∨ m 0 2 = 1) ∧
  (m 1 0 = 0 ∨ m 1 0 = 1) ∧
  (m 1 2 = 0 ∨ m 1 2 = 1) ∧
  (m 2 0 = 0 ∨ m 2 0 = 1) ∧
  (m 2 1 = 0 ∨ m 2 1 = 1)

-- A definition to check that rows are distinct
def distinct_rows (m : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  m 0 ≠ m 1 ∧ m 1 ≠ m 2 ∧ m 0 ≠ m 2

-- Complete proof problem statement
theorem matrix_count : ∃ (n : ℕ), 
  (∀ m : Matrix (Fin 3) (Fin 3) ℕ, valid_matrix m → distinct_rows m) ∧ 
  n = 45 :=
by
  sorry

end matrix_count_l13_13299


namespace small_angle_at_3_40_is_130_degrees_l13_13066

-- Definitions based on the problem's conditions
def minute_hand_angle (minute : ℕ) : ℝ :=
  minute * 6

def hour_hand_angle (hour minute : ℕ) : ℝ :=
  (hour * 60 + minute) * 0.5

-- Statement to prove that the smaller angle at 3:40 is 130.0 degrees
theorem small_angle_at_3_40_is_130_degrees :
  let minute := 40 in
  let hour := 3 in
  let angle_between_hands := abs ((minute_hand_angle minute) - (hour_hand_angle hour minute)) in
  min angle_between_hands (360 - angle_between_hands) = 130.0 :=
by
  sorry

end small_angle_at_3_40_is_130_degrees_l13_13066


namespace bears_per_shelf_l13_13967

def bears_initial : ℕ := 6

def shipment : ℕ := 18

def shelves : ℕ := 4

theorem bears_per_shelf : (bears_initial + shipment) / shelves = 6 := by
  sorry

end bears_per_shelf_l13_13967


namespace complement_intersection_l13_13481

open Set -- Open namespace for set operations

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3, 4}

-- Define set B
def B : Set ℕ := {3, 4, 5, 6}

-- Define the intersection A ∩ B
def A_inter_B : Set ℕ := A ∩ B

-- Define the complement C_I(S) as I \ S, where S is a subset of I
def complement (S : Set ℕ) : Set ℕ := I \ S

-- Prove that the complement of A ∩ B in I is {1, 2, 5, 6}
theorem complement_intersection : complement A_inter_B = {1, 2, 5, 6} :=
by
  sorry -- Proof to be provided

end complement_intersection_l13_13481


namespace phones_left_is_7500_l13_13965

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def sold_phones : ℕ := this_year_production / 4
def phones_left : ℕ := this_year_production - sold_phones

theorem phones_left_is_7500 : phones_left = 7500 :=
by
  sorry

end phones_left_is_7500_l13_13965


namespace one_real_root_multiple_coinciding_roots_three_distinct_real_roots_three_coinciding_roots_at_origin_l13_13170

-- Definitions from conditions
def cubic_eq (x p q : ℝ) := x^3 + p * x + q

-- Correct answers in mathematical proofs
theorem one_real_root (p q : ℝ) : 4 * p^3 + 27 * q^2 > 0 → ∃ x : ℝ, cubic_eq x p q = 0 := sorry

theorem multiple_coinciding_roots (p q : ℝ) : 4 * p^3 + 27 * q^2 = 0 ∧ (p ≠ 0 ∨ q ≠ 0) → ∃ x : ℝ, cubic_eq x p q = 0 := sorry

theorem three_distinct_real_roots (p q : ℝ) : 4 * p^3 + 27 * q^2 < 0 → ∃ x₁ x₂ x₃ : ℝ, 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ cubic_eq x₁ p q = 0 ∧ cubic_eq x₂ p q = 0 ∧ cubic_eq x₃ p q = 0 := sorry

theorem three_coinciding_roots_at_origin : ∃ x : ℝ, cubic_eq x 0 0 = 0 := sorry

end one_real_root_multiple_coinciding_roots_three_distinct_real_roots_three_coinciding_roots_at_origin_l13_13170


namespace min_value_fraction_solve_inequality_l13_13319

-- Part 1
theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (f : ℝ → ℝ)
  (h3 : f 1 = 2) (h4 : ∀ x, f x = a * x^2 + b * x + 1) :
  (a + b = 1) → (∃ z, z = (1 / a + 4 / b) ∧ z = 9) := 
by {
  sorry
}

-- Part 2
theorem solve_inequality (a : ℝ) (x : ℝ) (h1 : b = -a - 1) (f : ℝ → ℝ)
  (h2 : ∀ x, f x = a * x^2 + b * x + 1) :
  (f x ≤ 0) → 
  (if a = 0 then 
      {x | x ≥ 1}
  else if a > 0 then
      if a = 1 then 
          {x | x = 1}
      else if 0 < a ∧ a < 1 then 
          {x | 1 ≤ x ∧ x ≤ 1 / a}
      else 
          {x | 1 / a ≤ x ∧ x ≤ 1}
  else 
      {x | x ≥ 1 ∨ x ≤ 1 / a}) :=
by {
  sorry
}

end min_value_fraction_solve_inequality_l13_13319


namespace min_people_liking_both_l13_13014

theorem min_people_liking_both (total : ℕ) (Beethoven : ℕ) (Chopin : ℕ) 
    (total_eq : total = 150) (Beethoven_eq : Beethoven = 120) (Chopin_eq : Chopin = 95) : 
    ∃ (both : ℕ), both = 65 := 
by 
  have H := Beethoven + Chopin - total
  sorry

end min_people_liking_both_l13_13014


namespace find_c_plus_one_over_b_l13_13377

variable (a b c : ℝ)
variable (habc : a * b * c = 1)
variable (ha : a + (1 / c) = 7)
variable (hb : b + (1 / a) = 35)

theorem find_c_plus_one_over_b : (c + (1 / b) = 11 / 61) :=
by
  have h1 : a * b * c = 1 := habc
  have h2 : a + (1 / c) = 7 := ha
  have h3 : b + (1 / a) = 35 := hb
  sorry

end find_c_plus_one_over_b_l13_13377


namespace sqrt_expression_equality_l13_13284

theorem sqrt_expression_equality :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) = 10 :=
by
  sorry

end sqrt_expression_equality_l13_13284


namespace values_of_a_l13_13302

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (π * x / 2)

def m_a (a : ℝ) : ℝ :=
if a ≤ 1 then 0 else abs (Real.log a)

def M_a (a : ℝ) : ℝ :=
if a ≤ 1 then Real.sin (π * a / 2) else 1

theorem values_of_a (a : ℝ) (ha : 0 < a) : 
  M_a a - m_a a = 1/2 ↔ a = 1/3 ∨ a = Real.sqrt 10 :=
by
  sorry

end values_of_a_l13_13302


namespace eval_expr_x_eq_3_y_eq_4_l13_13132

theorem eval_expr_x_eq_3_y_eq_4 : 
  ∀ (x y : ℕ), x = 3 → y = 4 → 5 * x^y + 6 * y^x + x * y = 801 := 
by 
  intros x y hx hy 
  rw [hx, hy]
  -- Proof omitted
  sorry

end eval_expr_x_eq_3_y_eq_4_l13_13132


namespace a_minus_b_range_l13_13474

noncomputable def range_of_a_minus_b (a b : ℝ) : Set ℝ :=
  {x | -2 < a ∧ a < 1 ∧ 0 < b ∧ b < 4 ∧ x = a - b}

theorem a_minus_b_range (a b : ℝ) (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) :
  ∃ x, range_of_a_minus_b a b x ∧ (-6 < x ∧ x < 1) :=
by
  sorry

end a_minus_b_range_l13_13474


namespace card_subsets_l13_13528

theorem card_subsets (A : Finset ℕ) (hA_card : A.card = 3) : (A.powerset.card = 8) :=
sorry

end card_subsets_l13_13528


namespace barbara_candies_l13_13554

theorem barbara_candies : (9 + 18) = 27 :=
by
  sorry

end barbara_candies_l13_13554


namespace divisibility_criterion_l13_13917

theorem divisibility_criterion (x y : ℕ) (h_two_digit : 10 ≤ x ∧ x < 100) :
  (1207 % x = 0) ↔ (x = 10 * (x / 10) + (x % 10) ∧ (x / 10)^3 + (x % 10)^3 = 344) :=
by
  sorry

end divisibility_criterion_l13_13917


namespace gcd_315_2016_l13_13810

def a : ℕ := 315
def b : ℕ := 2016

theorem gcd_315_2016 : Nat.gcd a b = 63 := 
by 
  sorry

end gcd_315_2016_l13_13810


namespace simplify_expression_l13_13041

theorem simplify_expression :
  (6^7 + 4^6) * (1^5 - (-1)^5)^10 = 290938368 :=
by
  sorry

end simplify_expression_l13_13041


namespace fraction_of_females_l13_13514

variable (participants_last_year males_last_year females_last_year males_this_year females_this_year participants_this_year : ℕ)

-- The conditions
def conditions :=
  males_last_year = 20 ∧
  participants_this_year = (110 * (participants_last_year/100)) ∧
  males_this_year = (105 * males_last_year / 100) ∧
  females_this_year = (120 * females_last_year / 100) ∧
  participants_last_year = males_last_year + females_last_year ∧
  participants_this_year = males_this_year + females_this_year

-- The proof statement
theorem fraction_of_females (h : conditions males_last_year females_last_year males_this_year females_this_year participants_last_year participants_this_year) :
  (females_this_year : ℚ) / (participants_this_year : ℚ) = 4 / 11 :=
  sorry

end fraction_of_females_l13_13514


namespace solve_for_a_l13_13344

noncomputable def area_of_triangle (b c : ℝ) : ℝ :=
  1 / 2 * b * c * Real.sin (Real.pi / 3)

theorem solve_for_a (a b c : ℝ) (hA : 60 = 60) 
  (h_area : area_of_triangle b c = 3 * Real.sqrt 3 / 2)
  (h_sum_bc : b + c = 3 * Real.sqrt 3) :
  a = 3 :=
sorry

end solve_for_a_l13_13344


namespace faculty_after_reduction_is_correct_l13_13105

-- Define the original number of faculty members
def original_faculty : ℝ := 253.25

-- Define the reduction percentage as a decimal
def reduction_percentage : ℝ := 0.23

-- Calculate the reduction amount
def reduction_amount : ℝ := original_faculty * reduction_percentage

-- Define the rounded reduction amount
def rounded_reduction_amount : ℝ := 58.25

-- Calculate the number of professors after the reduction
def professors_after_reduction : ℝ := original_faculty - rounded_reduction_amount

-- Statement to be proven: the number of professors after the reduction is 195
theorem faculty_after_reduction_is_correct : professors_after_reduction = 195 := by
  sorry

end faculty_after_reduction_is_correct_l13_13105


namespace Danny_more_wrappers_than_caps_l13_13452

-- Define the conditions
def bottle_caps_park := 11
def wrappers_park := 28

-- State the theorem representing the problem
theorem Danny_more_wrappers_than_caps:
  wrappers_park - bottle_caps_park = 17 :=
by
  sorry

end Danny_more_wrappers_than_caps_l13_13452


namespace average_wage_per_day_l13_13544

theorem average_wage_per_day :
  let num_male := 20
  let num_female := 15
  let num_child := 5
  let wage_male := 35
  let wage_female := 20
  let wage_child := 8
  let total_wages := (num_male * wage_male) + (num_female * wage_female) + (num_child * wage_child)
  let total_workers := num_male + num_female + num_child
  total_wages / total_workers = 26 := by
  sorry

end average_wage_per_day_l13_13544


namespace no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n_k_ge_3_positive_ints_m_n_exists_l13_13093

-- Proof Problem 1:
theorem no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n :
  ¬ ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + 2) = n * (n + 1) :=
by sorry

-- Proof Problem 2:
theorem k_ge_3_positive_ints_m_n_exists (k : ℕ) (hk : k ≥ 3) :
  (k = 3 → ¬ ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) ∧
  (k ≥ 4 → ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) :=
by sorry

end no_positive_ints_m_n_m_square_plus_2_equals_n_square_plus_n_k_ge_3_positive_ints_m_n_exists_l13_13093


namespace ab_range_l13_13144

theorem ab_range (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = |2 - x^2|)
  (h_a_lt_b : 0 < a ∧ a < b) (h_fa_eq_fb : f a = f b) :
  0 < a * b ∧ a * b < 2 := 
by
  sorry

end ab_range_l13_13144


namespace decreasing_implies_b_geq_4_l13_13331

-- Define the function and its derivative
def function (x : ℝ) (b : ℝ) : ℝ := x^3 - 3*b*x + 1

def derivative (x : ℝ) (b : ℝ) : ℝ := 3*x^2 - 3*b

theorem decreasing_implies_b_geq_4 (b : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → derivative x b ≤ 0) → b ≥ 4 :=
by
  intros h
  sorry

end decreasing_implies_b_geq_4_l13_13331


namespace person_walking_speed_on_escalator_l13_13279

theorem person_walking_speed_on_escalator 
  (v : ℝ) 
  (escalator_speed : ℝ := 15) 
  (escalator_length : ℝ := 180) 
  (time_taken : ℝ := 10)
  (distance_eq : escalator_length = (v + escalator_speed) * time_taken) : 
  v = 3 := 
by 
  -- The proof steps will be filled in if required
  sorry

end person_walking_speed_on_escalator_l13_13279


namespace percentage_discount_four_friends_l13_13954

theorem percentage_discount_four_friends 
  (num_friends : ℕ)
  (original_price : ℝ)
  (total_spent : ℝ)
  (item_per_friend : ℕ)
  (total_items : ℕ)
  (each_spent : ℝ)
  (discount_percentage : ℝ):
  num_friends = 4 →
  original_price = 20 →
  total_spent = 40 →
  item_per_friend = 1 →
  total_items = num_friends * item_per_friend →
  each_spent = total_spent / num_friends →
  discount_percentage = ((original_price - each_spent) / original_price) * 100 →
  discount_percentage = 50 :=
by
  sorry

end percentage_discount_four_friends_l13_13954


namespace minimal_moves_for_7_disks_l13_13935

/-- Mathematical model of the Tower of Hanoi problem with special rules --/
def tower_of_hanoi_moves (n : ℕ) : ℚ :=
  if n = 7 then 23 / 4 else sorry

/-- Proof problem for the minimal number of moves required to transfer all seven disks to rod C --/
theorem minimal_moves_for_7_disks : tower_of_hanoi_moves 7 = 23 / 4 := 
  sorry

end minimal_moves_for_7_disks_l13_13935


namespace ceil_minus_x_of_fractional_part_half_l13_13456

theorem ceil_minus_x_of_fractional_part_half (x : ℝ) (hx : x - ⌊x⌋ = 1 / 2) : ⌈x⌉ - x = 1 / 2 :=
by
 sorry

end ceil_minus_x_of_fractional_part_half_l13_13456


namespace time_to_cross_first_platform_l13_13849

noncomputable section

def train_length : ℝ := 310
def platform_1_length : ℝ := 110
def platform_2_length : ℝ := 250
def crossing_time_platform_2 : ℝ := 20

def total_distance_2 (train_length platform_2_length : ℝ) : ℝ :=
  train_length + platform_2_length

def train_speed (total_distance_2 crossing_time_platform_2 : ℝ) : ℝ :=
  total_distance_2 / crossing_time_platform_2

def total_distance_1 (train_length platform_1_length : ℝ) : ℝ :=
  train_length + platform_1_length

def crossing_time_platform_1 (total_distance_1 train_speed : ℝ) : ℝ :=
  total_distance_1 / train_speed

theorem time_to_cross_first_platform :
  crossing_time_platform_1 (total_distance_1 train_length platform_1_length)
                           (train_speed (total_distance_2 train_length platform_2_length)
                                        crossing_time_platform_2) 
  = 15 :=
by
  -- We would prove this in a detailed proof which is omitted here.
  sorry

end time_to_cross_first_platform_l13_13849


namespace value_of_expression_l13_13309

theorem value_of_expression (x y : ℝ) (h1 : x = Real.sqrt 5 + Real.sqrt 3) (h2 : y = Real.sqrt 5 - Real.sqrt 3) : x^2 + x * y + y^2 = 18 :=
by sorry

end value_of_expression_l13_13309


namespace Mabel_gave_away_daisies_l13_13031

-- Setting up the conditions
variables (d_total : ℕ) (p_per_daisy : ℕ) (p_remaining : ℕ)

-- stating the assumptions
def initial_petals (d_total p_per_daisy : ℕ) := d_total * p_per_daisy
def petals_given_away (d_total p_per_daisy p_remaining : ℕ) := initial_petals d_total p_per_daisy - p_remaining
def daisies_given_away (d_total p_per_daisy p_remaining : ℕ) := petals_given_away d_total p_per_daisy p_remaining / p_per_daisy

-- The main theorem
theorem Mabel_gave_away_daisies 
  (h1 : d_total = 5)
  (h2 : p_per_daisy = 8)
  (h3 : p_remaining = 24) :
  daisies_given_away d_total p_per_daisy p_remaining = 2 :=
sorry

end Mabel_gave_away_daisies_l13_13031


namespace find_x_value_l13_13623

theorem find_x_value (A B C x : ℝ) (hA : A = 40) (hB : B = 3 * x) (hC : C = 2 * x) (hSum : A + B + C = 180) : x = 28 :=
by
  sorry

end find_x_value_l13_13623


namespace sequence_a8_l13_13499

theorem sequence_a8 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a n + a (n + 1)) 
  (h2 : a 7 = 120) : 
  a 8 = 194 :=
sorry

end sequence_a8_l13_13499


namespace percentage_cut_l13_13088

theorem percentage_cut (S C : ℝ) (hS : S = 940) (hC : C = 611) :
  (C / S) * 100 = 65 := 
by
  rw [hS, hC]
  norm_num

end percentage_cut_l13_13088


namespace total_pink_crayons_l13_13036

def mara_crayons := 40
def mara_pink_percent := 10
def luna_crayons := 50
def luna_pink_percent := 20

def pink_crayons (total_crayons : ℕ) (percent_pink : ℕ) : ℕ :=
  (percent_pink * total_crayons) / 100

def mara_pink_crayons := pink_crayons mara_crayons mara_pink_percent
def luna_pink_crayons := pink_crayons luna_crayons luna_pink_percent

theorem total_pink_crayons : mara_pink_crayons + luna_pink_crayons = 14 :=
by
  -- Proof can be written here.
  sorry

end total_pink_crayons_l13_13036


namespace series_sum_eq_half_l13_13123

theorem series_sum_eq_half :
  ∑' (n : ℕ), 2^n / (3^(2^n) + 1) = 1 / 2 :=
sorry

end series_sum_eq_half_l13_13123


namespace decrease_of_negative_distance_l13_13488

theorem decrease_of_negative_distance (x : Int) (increase : Int → Int) (decrease : Int → Int) :
  (increase 30 = 30) → (decrease 5 = -5) → (decrease 5 = -5) :=
by
  intros
  sorry

end decrease_of_negative_distance_l13_13488


namespace sum_of_ages_l13_13851

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
def condition1 := a = 16 + b + c
def condition2 := a^2 = 1632 + (b + c)^2

-- Define the theorem to prove the question
theorem sum_of_ages : condition1 a b c → condition2 a b c → a + b + c = 102 := 
by 
  intros h1 h2
  sorry

end sum_of_ages_l13_13851


namespace find_a_l13_13887

theorem find_a (a : ℝ) :
  (∀ x : ℝ, ((x^2 - 4 * x + a) + |x - 3| ≤ 5) → x ≤ 3) →
  (∃ x : ℝ, x = 3 ∧ ((x^2 - 4 * x + a) + |x - 3| ≤ 5)) →
  a = 2 := 
by
  sorry

end find_a_l13_13887


namespace buses_trips_product_l13_13646

theorem buses_trips_product :
  ∃ (n k : ℕ), n > 3 ∧ n * (n - 1) * (2 * k - 1) = 600 ∧ (n * k = 52 ∨ n * k = 40) := 
by
  sorry

end buses_trips_product_l13_13646


namespace divisibility_criterion_l13_13916

theorem divisibility_criterion :
  (∃ x : ℕ, 10 ≤ x ∧ x < 100 ∧ (1207 % x = 0) ∧
  (let a := x / 10 in let b := x % 10 in a^3 + b^3 = 344)) ↔
  (1207 % 17 = 0 ∧ let a1 := 1 in let b1 := 7 in a1^3 + b1^3 = 344) ∨
  (1207 % 71 = 0 ∧ let a2 := 7 in let b2 := 1 in a2^3 + b2^3 = 344) :=
by sorry

end divisibility_criterion_l13_13916


namespace bisection_method_termination_condition_l13_13394

theorem bisection_method_termination_condition (x1 x2 : ℝ) (ε : ℝ) : Prop :=
  |x1 - x2| < ε

end bisection_method_termination_condition_l13_13394


namespace quadratic_sum_eq_504_l13_13864

theorem quadratic_sum_eq_504 :
  ∃ (a b c : ℝ), (∀ x : ℝ, 20 * x^2 + 160 * x + 800 = a * (x + b)^2 + c) ∧ a + b + c = 504 :=
by sorry

end quadratic_sum_eq_504_l13_13864


namespace tan_product_l13_13560

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end tan_product_l13_13560


namespace perfect_apples_l13_13619

theorem perfect_apples (total_apples : ℕ) (fraction_small fraction_unripe : ℚ) 
  (h_total_apples : total_apples = 30) 
  (h_fraction_small : fraction_small = 1 / 6) 
  (h_fraction_unripe : fraction_unripe = 1 / 3) : 
  total_apples * (1 - fraction_small - fraction_unripe) = 15 :=
  by
  rw [h_total_apples, h_fraction_small, h_fraction_unripe]
  have h : 1 - (1/6 + 1/3) = 1/2 := by norm_num
  rw h
  norm_num

end perfect_apples_l13_13619


namespace value_of_f_sum_l13_13745

variable (a b c m : ℝ)

def f (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f_sum :
  f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end value_of_f_sum_l13_13745


namespace find_x_l13_13898

theorem find_x (x : ℕ) : (x % 9 = 0) ∧ (x^2 > 144) ∧ (x < 30) → (x = 18 ∨ x = 27) :=
by 
  sorry

end find_x_l13_13898


namespace clock_angle_l13_13072

-- Conditions
def hour_position (h m : ℕ) : ℝ := (h % 12) * 30 + m * 0.5
def minute_position (m : ℕ) : ℝ := m * 6
def angle_between (pos1 pos2 : ℝ) : ℝ := abs (pos1 - pos2)

-- Given data
def h := 3
def m := 40

-- Calculate positions
def hour_pos := hour_position h m
def minute_pos := minute_position m

-- Calculate the smaller angle
def smaller_angle (pos1 pos2 : ℝ) : ℝ := if angle_between pos1 pos2 <= 180 then angle_between pos1 pos2 else 360 - angle_between pos1 pos2

-- Final statement to prove
theorem clock_angle : smaller_angle hour_pos minute_pos = 130.0 :=
  sorry

end clock_angle_l13_13072


namespace number_of_false_propositions_is_even_l13_13928

theorem number_of_false_propositions_is_even 
  (P Q : Prop) : 
  ∃ (n : ℕ), (P ∧ ¬P ∧ (¬Q → ¬P) ∧ (Q → P)) = false ∧ n % 2 = 0 := sorry

end number_of_false_propositions_is_even_l13_13928


namespace cake_angle_between_adjacent_pieces_l13_13443

theorem cake_angle_between_adjacent_pieces 
  (total_angle : ℝ := 360)
  (total_pieces : ℕ := 10)
  (eaten_pieces : ℕ := 1)
  (angle_per_piece := total_angle / total_pieces)
  (remaining_pieces := total_pieces - eaten_pieces)
  (new_angle_per_piece := total_angle / remaining_pieces) :
  (new_angle_per_piece - angle_per_piece = 4) := 
by
  sorry

end cake_angle_between_adjacent_pieces_l13_13443


namespace sunlovers_happy_days_l13_13676

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l13_13676


namespace trouser_sale_price_l13_13909

theorem trouser_sale_price 
  (original_price : ℝ) 
  (percent_decrease : ℝ) 
  (sale_price : ℝ) 
  (h : original_price = 100) 
  (p : percent_decrease = 0.25) 
  (s : sale_price = original_price * (1 - percent_decrease)) : 
  sale_price = 75 :=
by 
  sorry

end trouser_sale_price_l13_13909


namespace Bill_initial_money_l13_13470

theorem Bill_initial_money (joint_money : ℕ) (pizza_cost : ℕ) (num_pizzas : ℕ) (final_bill_amount : ℕ) (initial_joint_money_eq : joint_money = 42) (pizza_cost_eq : pizza_cost = 11) (num_pizzas_eq : num_pizzas = 3) (final_bill_amount_eq : final_bill_amount = 39) :
  ∃ b : ℕ, b = 30 :=
by
  sorry

end Bill_initial_money_l13_13470


namespace probability_queen_and_spade_l13_13217

def standard_deck : Finset (ℕ × Suit) := 
  Finset.range 52

inductive Card
| queen : Suit → Card
| other : ℕ → Suit → Card

inductive Suit
| hearts
| diamonds
| clubs
| spades

open Card Suit

def count_queens (deck : Finset (Card)) : ℕ :=
  deck.count (λ c => match c with
                    | queen _ => true
                    | _ => false)

def count_spades (deck : Finset (Card)) : ℕ :=
  deck.count (λ c => match c with
                    | queen spades => true
                    | other _ spades => true
                    | _ => false)

theorem probability_queen_and_spade
  (h_deck : ∀ c ∈ standard_deck, c = queen hearts ∨ c = queen diamonds ∨ c = queen clubs ∨ c = queen spades
  ∨ c = other 1 hearts ∨ c = other 1 diamonds ∨ c = other 1 clubs ∨ c = other 1 spades
  ∨ ... (other combinations for cards))
  (h_queens : count_queens standard_deck = 4)
  (h_spades : count_spades standard_deck = 13) :
  sorry : ℚ :=
begin
  -- Mathematically prove the probability is 4/17, proof is omitted for now
  sorry
end

end probability_queen_and_spade_l13_13217


namespace num_ways_to_pay_16_rubles_l13_13863

theorem num_ways_to_pay_16_rubles :
  ∃! (n : ℕ), n = 13 ∧ ∀ (x y z : ℕ), (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0) ∧ 
  (10 * x + 2 * y + 1 * z = 16) ∧ (x < 2) ∧ (y + z > 0) := sorry

end num_ways_to_pay_16_rubles_l13_13863


namespace peaches_left_in_baskets_l13_13059

theorem peaches_left_in_baskets :
  let initial_baskets := 5
  let initial_peaches_per_basket := 20
  let new_baskets := 4
  let new_peaches_per_basket := 25
  let peaches_removed_per_basket := 10

  let total_initial_peaches := initial_baskets * initial_peaches_per_basket
  let total_new_peaches := new_baskets * new_peaches_per_basket
  let total_peaches_before_removal := total_initial_peaches + total_new_peaches

  let total_baskets := initial_baskets + new_baskets
  let total_peaches_removed := total_baskets * peaches_removed_per_basket
  let peaches_left := total_peaches_before_removal - total_peaches_removed

  peaches_left = 110 := by
  sorry

end peaches_left_in_baskets_l13_13059


namespace probability_queen_then_spade_l13_13219

-- Define the size of the deck and the quantities for specific cards
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_spades : ℕ := 13

-- Define the probability calculation problem
theorem probability_queen_then_spade :
  (num_queens / deck_size : ℚ) * ((num_spades - 1) / (deck_size - 1) : ℚ) + ((num_queens - 1) / deck_size : ℚ) * (num_spades / (deck_size - 1) : ℚ) = 1 / deck_size :=
by sorry

end probability_queen_then_spade_l13_13219


namespace part1_part2_l13_13758

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - (1 / 5) ^ x

-- First part: f(x₁) > f(x₂) for any x₁, x₂ ∈ [1, +∞) with x₁ < x₂
theorem part1 (x₁ x₂ : ℝ) (h1 : 1 ≤ x₁) (h2 : 1 ≤ x₂) (h3 : x₁ < x₂) : f x₁ > f x₂ :=
sorry

-- Second part: f(√(x₁ x₂)) > √(f(x₁) f(x₂)) for any x₁, x₂ ∈ [1, +∞) with x₁ < x₂
theorem part2 (x₁ x₂ : ℝ) (h1 : 1 ≤ x₁) (h2 : 1 ≤ x₂) (h3 : x₁ < x₂) : 
  f (Real.sqrt (x₁ * x₂)) > Real.sqrt (f x₁ * f x₂) :=
sorry

end part1_part2_l13_13758


namespace reflect_curve_maps_onto_itself_l13_13369

theorem reflect_curve_maps_onto_itself (a b c : ℝ) :
    ∃ (x0 y0 : ℝ), 
    x0 = -a / 3 ∧ 
    y0 = 2 * a^3 / 27 - a * b / 3 + c ∧
    ∀ x y x' y', 
    y = x^3 + a * x^2 + b * x + c → 
    x' = 2 * x0 - x → 
    y' = 2 * y0 - y → 
    y' = x'^3 + a * x'^2 + b * x' + c := 
    by sorry

end reflect_curve_maps_onto_itself_l13_13369


namespace drum_y_capacity_filled_l13_13240

-- Definitions of the initial conditions
def capacity_of_drum_X (C : ℝ) (half_full_x : ℝ) := half_full_x = 1 / 2 * C
def capacity_of_drum_Y (C : ℝ) (two_c_y : ℝ) := two_c_y = 2 * C
def oil_in_drum_X (C : ℝ) (half_full_x : ℝ) := half_full_x = 1 / 2 * C
def oil_in_drum_Y (C : ℝ) (four_fifth_c_y : ℝ) := four_fifth_c_y = 4 / 5 * C

-- Theorem to prove the capacity filled in drum Y after pouring all oil from X
theorem drum_y_capacity_filled {C : ℝ} (hx : 1/2 * C = 1 / 2 * C) (hy : 2 * C = 2 * C) (ox : 1/2 * C = 1 / 2 * C) (oy : 4/5 * 2 * C = 4 / 5 * C) :
  ( (1/2 * C + 4/5 * C) / (2 * C) ) = 13 / 20 :=
by
  sorry

end drum_y_capacity_filled_l13_13240


namespace proof_part1_proof_part2_l13_13894

-- Definitions for conditions
def line_parametric (t : ℝ) : ℝ × ℝ := (t, sqrt 3 * t + sqrt 2 / 2)
def curve_polar (ρ θ : ℝ) : ℝ := ρ = 2 * cos (θ - π / 4)

-- Definitions of the solved parts
def slope_angle_of_line : ℝ := π / 3
def curve_rect_eq (x y : ℝ) : Prop := (x - sqrt 2 / 2)^2 + (y - sqrt 2 / 2)^2 = 1

-- Propositions to be proven
theorem proof_part1 : slope_angle_of_line = π / 3 ∧ (∀ x y, curve_rect_eq x y ↔ curve_polar (sqrt (x^2 + y^2)) (atan2 y x))
:= sorry

theorem proof_part2 (A B P : ℝ × ℝ) (hA : A ∈ line_intersection_points) (hB : B ∈ line_intersection_points) (hP : P = (0, sqrt 2 / 2)) :
  |distance P A| + |distance P B| = sqrt 10 / 2
:= sorry

end proof_part1_proof_part2_l13_13894


namespace motorboat_max_distance_l13_13084

/-- Given a motorboat which, when fully fueled, can travel exactly 40 km against the current 
    or 60 km with the current, proves that the maximum distance it can travel up the river and 
    return to the starting point with the available fuel is 24 km. -/
theorem motorboat_max_distance (upstream_dist : ℕ) (downstream_dist : ℕ) : 
  upstream_dist = 40 → downstream_dist = 60 → 
  ∃ max_round_trip_dist : ℕ, max_round_trip_dist = 24 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end motorboat_max_distance_l13_13084


namespace no_unhappy_days_l13_13664

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l13_13664


namespace find_f_10_l13_13260

def f (x : Int) : Int := sorry

axiom condition_1 : f 1 + 1 > 0
axiom condition_2 : ∀ x y : Int, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom condition_3 : ∀ x : Int, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := by
  sorry

end find_f_10_l13_13260


namespace problem_statement_l13_13893

-- Definitions related to the given conditions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi) / 6)

theorem problem_statement :
  (∀ x1 x2 : ℝ, (x1 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → (x2 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → x1 < x2 → f x1 < f x2) →
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) →
  f (-((5 * Real.pi) / 12)) = (Real.sqrt 3) / 2 :=
by
  intros h_mono h_symm
  sorry

end problem_statement_l13_13893


namespace average_of_numbers_in_range_l13_13295

-- Define the set of numbers we are considering
def numbers_in_range : List ℕ := [10, 15, 20, 25, 30]

-- Define the sum of these numbers
def sum_in_range : ℕ := 10 + 15 + 20 + 25 + 30

-- Define the number of elements in our range
def count_in_range : ℕ := 5

-- Prove that the average of numbers in the range is 20
theorem average_of_numbers_in_range : (sum_in_range / count_in_range) = 20 := by
  -- TODO: Proof to be written, for now we use sorry as a placeholder
  sorry

end average_of_numbers_in_range_l13_13295


namespace expected_value_absolute_difference_after_10_days_l13_13943

/-- Define the probability space and outcome -/
noncomputable def probability_cat_wins : ℝ := 0.25
noncomputable def probability_fox_wins : ℝ := 0.25
noncomputable def probability_both_police : ℝ := 0.50

/-- Define the random variable for absolute difference in wealth -/
noncomputable def X_n (n : ℕ) : ℝ := sorry

/-- Define the probability p_{0, n} -/
noncomputable def p (k n : ℕ) : ℝ := sorry

/-- Given the above conditions, the expected value of the absolute difference -/
theorem expected_value_absolute_difference_after_10_days : (∑ k in finset.range 11, k * p k 10) = 1 :=
sorry

end expected_value_absolute_difference_after_10_days_l13_13943


namespace average_age_of_troupe_l13_13621

theorem average_age_of_troupe
  (number_females : ℕ) (number_males : ℕ) 
  (average_age_females : ℕ) (average_age_males : ℕ)
  (total_people : ℕ) (total_age : ℕ)
  (h1 : number_females = 12) 
  (h2 : number_males = 18) 
  (h3 : average_age_females = 25) 
  (h4 : average_age_males = 30)
  (h5 : total_people = 30)
  (h6 : total_age = (25 * 12 + 30 * 18)) :
  total_age / total_people = 28 :=
by
  -- Proof goes here
  sorry

end average_age_of_troupe_l13_13621


namespace find_f_value_l13_13267

def f (x : ℤ) : ℤ := sorry

theorem find_f_value :
  (f(1) + 1 > 0) ∧ 
  (∀ (x y : ℤ), f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y) ∧
  (∀ (x : ℤ), 2 * f(x) = f(x + 1) - x + 1) →
  f 10 = 1014 :=
by
  sorry

end find_f_value_l13_13267


namespace probability_queen_then_spade_l13_13221

theorem probability_queen_then_spade (h_deck: ℕ) (h_queens: ℕ) (h_spades: ℕ) :
  h_deck = 52 ∧ h_queens = 4 ∧ h_spades = 13 →
  (1 / 52) * (12 / 51) + (3 / 52) * (13 / 51) = 18 / 221 :=
by
  sorry

end probability_queen_then_spade_l13_13221


namespace child_running_speed_l13_13704

theorem child_running_speed
  (c s t : ℝ)
  (h1 : (74 - s) * 3 = 165)
  (h2 : (74 + s) * t = 372) :
  c = 74 :=
by sorry

end child_running_speed_l13_13704


namespace spell_AMCB_paths_equals_24_l13_13985

def central_A_reachable_M : Nat := 4
def M_reachable_C : Nat := 2
def C_reachable_B : Nat := 3

theorem spell_AMCB_paths_equals_24 :
  central_A_reachable_M * M_reachable_C * C_reachable_B = 24 := by
  sorry

end spell_AMCB_paths_equals_24_l13_13985


namespace fraction_problem_l13_13374

theorem fraction_problem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : (2 * a - b) / (a + 4 * b) = 3) : 
  (a - 4 * b) / (2 * a + b) = 17 / 25 :=
by sorry

end fraction_problem_l13_13374


namespace only_odd_digit_square_l13_13471

def odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d % 2 = 1

theorem only_odd_digit_square (n : ℕ) : n^2 = n → odd_digits n → n = 1 ∨ n = 9 :=
by
  intros
  sorry

end only_odd_digit_square_l13_13471


namespace median_of_list_l13_13342

theorem median_of_list : 
  let list := (List.range 100).bind (λ n, List.replicate (n+1) (n+1)) in
  list.length = 5050 ∧ 
  list.nth (2525 - 1) = some 71 ∧ -- since List.nth is 0-indexed, we use (2525 - 1) and (2526 - 1)
  list.nth 2525 = some 71 → 
  (list.median = 71) :=
by
  -- Provide the infrastructure setup for the list construction and the median computation.
  sorry

end median_of_list_l13_13342


namespace golden_ticket_problem_l13_13445

open Real

/-- The golden ratio -/
noncomputable def φ := (1 + sqrt 5) / 2

/-- Assume the proportions and the resulting area -/
theorem golden_ticket_problem
  (a b : ℝ)
  (h : 0 + b * φ = 
        φ - (5 + sqrt 5) / (8 * φ)) :
  b / a = -4 / 3 :=
  sorry

end golden_ticket_problem_l13_13445


namespace geometric_sequence_sum_l13_13016

variable {a b : ℝ} -- Parameters for real numbers a and b
variable (a_ne_zero : a ≠ 0) -- condition a ≠ 0

/-- Proof that in the geometric sequence {a_n}, given a_5 + a_6 = a and a_15 + a_16 = b, 
    a_25 + a_26 = b^2 / a --/
theorem geometric_sequence_sum (a5_plus_a6 : ℕ → ℝ) (a15_plus_a16 : ℕ → ℝ) (a25_plus_a26 : ℕ → ℝ)
  (h1 : a5_plus_a6 5 + a5_plus_a6 6 = a)
  (h2 : a15_plus_a16 15 + a15_plus_a16 16 = b) :
  a25_plus_a26 25 + a25_plus_a26 26 = b^2 / a :=
  sorry

end geometric_sequence_sum_l13_13016


namespace arctan_sum_property_l13_13602

open Real

theorem arctan_sum_property (x y z : ℝ) :
  arctan x + arctan y + arctan z = π / 2 → x * y + y * z + x * z = 1 :=
by
  sorry

end arctan_sum_property_l13_13602


namespace no_unhappy_days_l13_13675

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l13_13675


namespace combinations_sum_l13_13828
open Nat

theorem combinations_sum : 
  let d := [1, 2, 3, 4]
  let count_combinations (n : Nat) := factorial n
  count_combinations 1 + count_combinations 2 + count_combinations 3 + count_combinations 4 = 64 :=
  by
    sorry

end combinations_sum_l13_13828


namespace base_conversion_min_sum_l13_13529

theorem base_conversion_min_sum (a b : ℕ) (h1 : 3 * a + 6 = 6 * b + 3) (h2 : 6 < a) (h3 : 6 < b) : a + b = 20 :=
sorry

end base_conversion_min_sum_l13_13529


namespace determine_a_l13_13599
open Set

-- Given Condition Definitions
def U : Set ℕ := {1, 3, 5, 7}
def M (a : ℤ) : Set ℕ := {1, Int.natAbs (a - 5)} -- using ℤ for a and natAbs to get |a - 5|

-- Problem statement
theorem determine_a (a : ℤ) (hM_subset_U : M a ⊆ U) (h_complement : U \ M a = {5, 7}) : a = 2 ∨ a = 8 :=
by sorry

end determine_a_l13_13599


namespace smallest_integer_b_gt_4_base_b_perfect_square_l13_13399

theorem smallest_integer_b_gt_4_base_b_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ n : ℕ, 2 * b + 5 = n^2 ∧ b = 10 :=
by
  sorry

end smallest_integer_b_gt_4_base_b_perfect_square_l13_13399


namespace quadratic_inequality_solution_l13_13680

theorem quadratic_inequality_solution
  (x : ℝ) :
  -2 * x^2 + x < -3 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 2) := by
  sorry

end quadratic_inequality_solution_l13_13680


namespace no_unhappy_days_l13_13666

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l13_13666


namespace num_first_and_second_year_students_total_l13_13940

-- Definitions based on conditions
def num_sampled_students : ℕ := 55
def num_first_year_students_sampled : ℕ := 10
def num_second_year_students_sampled : ℕ := 25
def num_third_year_students_total : ℕ := 400

-- Given that 20 students from the third year are sampled
def num_third_year_students_sampled := num_sampled_students - num_first_year_students_sampled - num_second_year_students_sampled

-- Proportion equality condition
theorem num_first_and_second_year_students_total (x : ℕ) :
  20 / 55 = 400 / (x + num_third_year_students_total) →
  x = 700 :=
by
  sorry

end num_first_and_second_year_students_total_l13_13940


namespace find_m_for_given_slope_l13_13211

theorem find_m_for_given_slope (m : ℝ) :
  (∃ (P Q : ℝ × ℝ),
    P = (-2, m) ∧ Q = (m, 4) ∧
    (Q.2 - P.2) / (Q.1 - P.1) = 1) → m = 1 :=
by
  sorry

end find_m_for_given_slope_l13_13211


namespace no_rearrangement_to_positive_and_negative_roots_l13_13627

theorem no_rearrangement_to_positive_and_negative_roots (a b c : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ a ≠ 0 ∧ b = -a * (x1 + x2) ∧ c = a * x1 * x2) →
  (∃ y1 y2 : ℝ, y1 > 0 ∧ y2 > 0 ∧ a ≠ 0 ∧ b != 0 ∧ c != 0 ∧ 
    (∃ b' c' : ℝ, b' ≠ b ∧ c' ≠ c ∧ 
      b' = -a * (y1 + y2) ∧ c' = a * y1 * y2)) →
  False := by
  sorry

end no_rearrangement_to_positive_and_negative_roots_l13_13627


namespace geom_sequence_third_term_l13_13768

theorem geom_sequence_third_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a n = a 1 * r ^ (n - 1)) (h_cond : a 1 * a 5 = a 3) : a 3 = 1 :=
sorry

end geom_sequence_third_term_l13_13768


namespace ratio_boys_to_girls_l13_13333

theorem ratio_boys_to_girls (total_students girls : ℕ) (h1 : total_students = 455) (h2 : girls = 175) :
  let boys := total_students - girls
  (boys : ℕ) / Nat.gcd boys girls = 8 / 1 ∧ (girls : ℕ) / Nat.gcd boys girls = 5 / 1 :=
by
  sorry

end ratio_boys_to_girls_l13_13333


namespace correctly_calculated_value_l13_13924

theorem correctly_calculated_value (n : ℕ) (h : 5 * n = 30) : n / 6 = 1 :=
sorry

end correctly_calculated_value_l13_13924


namespace neg_p_l13_13245

open Set

-- Definitions of sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

def A : Set ℤ := {x | is_odd x}
def B : Set ℤ := {x | is_even x}

-- Proposition p
def p : Prop := ∀ x ∈ A, 2 * x ∈ B

-- Negation of the proposition p
theorem neg_p : ¬p ↔ ∃ x ∈ A, ¬(2 * x ∈ B) := sorry

end neg_p_l13_13245


namespace moles_of_H2O_formed_l13_13138

theorem moles_of_H2O_formed
  (moles_H2SO4 : ℕ)
  (moles_H2O : ℕ)
  (H : moles_H2SO4 = 3)
  (H' : moles_H2O = 3) :
  moles_H2O = 3 :=
by
  sorry

end moles_of_H2O_formed_l13_13138


namespace A_inter_B_eq_l13_13635

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 > 1}

theorem A_inter_B_eq : A ∩ B = {-2, 2} := 
by
  sorry

end A_inter_B_eq_l13_13635


namespace solve_for_x_l13_13816

theorem solve_for_x (x y : ℝ) : 3 * x + 4 * y = 5 → x = (5 - 4 * y) / 3 :=
by
  intro h
  sorry

end solve_for_x_l13_13816


namespace arithmetic_geometric_sequence_l13_13881

open Real

noncomputable def a_4 (a1 q : ℝ) : ℝ := a1 * q^3
noncomputable def sum_five_terms (a1 q : ℝ) : ℝ := a1 * (1 - q^5) / (1 - q)

theorem arithmetic_geometric_sequence :
  ∀ (a1 q : ℝ),
    (a1 + a1 * q^2 = 10) →
    (a1 * q^3 + a1 * q^5 = 5 / 4) →
    (a_4 a1 q = 1) ∧ (sum_five_terms a1 q = 31 / 2) :=
by
  intros a1 q h1 h2
  sorry

end arithmetic_geometric_sequence_l13_13881


namespace speed_of_man_in_still_water_l13_13840

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 4 = 48 →
  (v_m - v_s) * 6 = 24 →
  v_m = 8 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end speed_of_man_in_still_water_l13_13840


namespace line_tangent_to_curve_iff_a_zero_l13_13318

noncomputable def f (x : ℝ) := Real.sin (2 * x)
noncomputable def l (x a : ℝ) := 2 * x + a

theorem line_tangent_to_curve_iff_a_zero (a : ℝ) :
  (∃ x₀ : ℝ, deriv f x₀ = 2 ∧ f x₀ = l x₀ a) → a = 0 :=
sorry

end line_tangent_to_curve_iff_a_zero_l13_13318


namespace initial_average_quiz_score_l13_13901

theorem initial_average_quiz_score 
  (n : ℕ) (A : ℝ) (dropped_avg : ℝ) (drop_score : ℝ)
  (students_before : n = 16)
  (students_after : n - 1 = 15)
  (dropped_avg_eq : dropped_avg = 64.0)
  (drop_score_eq : drop_score = 8) 
  (total_sum_before_eq : n * A = 16 * A)
  (total_sum_after_eq : (n - 1) * dropped_avg = 15 * 64):
  A = 60.5 := 
by
  sorry

end initial_average_quiz_score_l13_13901


namespace problem_π_digit_sequence_l13_13792

def f (n : ℕ) : ℕ :=
  match n with
  | 1  => 1
  | 2  => 4
  | 3  => 1
  | 4  => 5
  | 5  => 9
  | 6  => 2
  | 7  => 6
  | 8  => 5
  | 9  => 3
  | 10 => 5
  | _  => 0  -- for simplicity we define other cases arbitrarily

theorem problem_π_digit_sequence :
  ∃ n : ℕ, n > 0 ∧ f (f (f (f (f 10)))) = 1 := by
  sorry

end problem_π_digit_sequence_l13_13792


namespace valid_q_range_l13_13992

noncomputable def polynomial_has_nonneg_root (q : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ (x^4 + q*x^3 + x^2 + q*x + 4 = 0)

theorem valid_q_range (q : ℝ) : polynomial_has_nonneg_root q → q ≤ -2 * Real.sqrt 2 := 
sorry

end valid_q_range_l13_13992


namespace find_f_10_l13_13265

noncomputable def f : ℤ → ℤ := sorry

axiom cond1 : f 1 + 1 > 0
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := 
by
  sorry 

end find_f_10_l13_13265


namespace series_sum_eq_half_l13_13129

theorem series_sum_eq_half : ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_eq_half_l13_13129


namespace tire_usage_is_25714_l13_13416

-- Definitions based on conditions
def car_has_six_tires : Prop := (4 + 2 = 6)
def used_equally_over_miles (total_miles : ℕ) (number_of_tires : ℕ) : Prop := 
  (total_miles * 4) / number_of_tires = 25714

-- Theorem statement based on proof
theorem tire_usage_is_25714 (miles_driven : ℕ) (num_tires : ℕ) 
  (h1 : car_has_six_tires) 
  (h2 : miles_driven = 45000)
  (h3 : num_tires = 7) :
  used_equally_over_miles miles_driven num_tires :=
by
  sorry

end tire_usage_is_25714_l13_13416


namespace quadrilateral_offset_l13_13574

theorem quadrilateral_offset (d A h₂ x : ℝ)
  (h_da: d = 40)
  (h_A: A = 400)
  (h_h2 : h₂ = 9)
  (h_area : A = 1/2 * d * (x + h₂)) : 
  x = 11 :=
by sorry

end quadrilateral_offset_l13_13574


namespace difference_is_three_l13_13782

-- Define the range for two-digit numbers
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define whether a number is a multiple of three
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Identify the smallest and largest two-digit multiples of three
def smallest_two_digit_multiple_of_three : ℕ := 12
def largest_two_digit_multiple_of_three : ℕ := 99

-- Identify the smallest and largest two-digit non-multiples of three
def smallest_two_digit_non_multiple_of_three : ℕ := 10
def largest_two_digit_non_multiple_of_three : ℕ := 98

-- Calculate Joey's sum
def joeys_sum : ℕ := smallest_two_digit_multiple_of_three + largest_two_digit_multiple_of_three

-- Calculate Zoë's sum
def zoes_sum : ℕ := smallest_two_digit_non_multiple_of_three + largest_two_digit_non_multiple_of_three

-- Prove the difference between Joey's and Zoë's sums is 3
theorem difference_is_three : joeys_sum - zoes_sum = 3 :=
by
  -- The proof is not given, so we use sorry here
  sorry

end difference_is_three_l13_13782


namespace star_equiv_l13_13742

variable {m n x y : ℝ}

def star (m n : ℝ) : ℝ := (3 * m - 2 * n) ^ 2

theorem star_equiv (x y : ℝ) : star ((3 * x - 2 * y) ^ 2) ((2 * y - 3 * x) ^ 2) = (3 * x - 2 * y) ^ 4 := 
by
  sorry

end star_equiv_l13_13742


namespace market_value_of_stock_l13_13695

theorem market_value_of_stock 
  (yield : ℝ) 
  (dividend_percentage : ℝ) 
  (face_value : ℝ) 
  (market_value : ℝ) 
  (h1 : yield = 0.10) 
  (h2 : dividend_percentage = 0.07) 
  (h3 : face_value = 100) 
  (h4 : market_value = (dividend_percentage * face_value) / yield) :
  market_value = 70 := by
  sorry

end market_value_of_stock_l13_13695


namespace sum_of_altitudes_is_less_than_perimeter_l13_13914

theorem sum_of_altitudes_is_less_than_perimeter 
  (a b c h_a h_b h_c : ℝ) 
  (h_a_le_b : h_a ≤ b) 
  (h_b_le_c : h_b ≤ c) 
  (h_c_le_a : h_c ≤ a) 
  (strict_inequality : h_a < b ∨ h_b < c ∨ h_c < a) : h_a + h_b + h_c < a + b + c := 
by 
  sorry

end sum_of_altitudes_is_less_than_perimeter_l13_13914


namespace decreasing_function_range_l13_13166

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := - (1 / 2) * x^2 + m * real.log x

theorem decreasing_function_range {m : ℝ} :
  (∀ x > 1, deriv (λ x, f x m) x ≤ 0) ↔ m ≤ 1 :=
by
  -- Proof skipped
  sorry

end decreasing_function_range_l13_13166


namespace no_unhappy_days_l13_13672

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l13_13672


namespace squirrels_more_than_nuts_l13_13411

theorem squirrels_more_than_nuts 
  (squirrels : ℕ) 
  (nuts : ℕ) 
  (h_squirrels : squirrels = 4) 
  (h_nuts : nuts = 2) 
  : squirrels - nuts = 2 :=
by
  sorry

end squirrels_more_than_nuts_l13_13411


namespace tom_brady_average_yards_per_game_l13_13215

theorem tom_brady_average_yards_per_game 
  (record : ℕ) (current_yards : ℕ) (games_left : ℕ) 
  (h_record : record = 6000) 
  (h_current : current_yards = 4200) 
  (h_games : games_left = 6) : 
  (record - current_yards) / games_left = 300 := 
by {
  rw [h_record, h_current, h_games],
  norm_num,
  exact nat.div_eq_of_eq_mul_right (nat.succ_pos 5) rfl
}

end tom_brady_average_yards_per_game_l13_13215


namespace max_value_of_trig_expr_l13_13298

theorem max_value_of_trig_expr : 
  ∃ x, ∀ θ, (2 * Real.cos θ + 3 * Real.sin θ) ≤ (sqrt 13) := by
  sorry

end max_value_of_trig_expr_l13_13298


namespace f_10_l13_13258

namespace MathProof

variable (f : ℤ → ℤ)

-- Condition 1: f(1) + 1 > 0
axiom cond1 : f 1 + 1 > 0

-- Condition 2: f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y for any x, y ∈ ℤ
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y

-- Condition 3: 2 * f(x) = f(x + 1) - x + 1 for any x ∈ ℤ
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

-- We need to prove f(10) = 1014
theorem f_10 : f 10 = 1014 :=
by
  sorry

end MathProof

end f_10_l13_13258


namespace abs_lt_one_suff_but_not_necc_l13_13609

theorem abs_lt_one_suff_but_not_necc (x : ℝ) : (|x| < 1 → x^2 + x - 2 < 0) ∧ ¬(x^2 + x - 2 < 0 → |x| < 1) :=
by
  sorry

end abs_lt_one_suff_but_not_necc_l13_13609


namespace side_length_of_S2_l13_13918

variable (r s : ℝ)

theorem side_length_of_S2 (h1 : 2 * r + s = 2100) (h2 : 2 * r + 3 * s = 3400) : s = 650 := by
  sorry

end side_length_of_S2_l13_13918


namespace foodAdditivesPercentage_l13_13248

-- Define the given percentages
def microphotonicsPercentage : ℕ := 14
def homeElectronicsPercentage : ℕ := 24
def microorganismsPercentage : ℕ := 29
def industrialLubricantsPercentage : ℕ := 8

-- Define degrees representing basic astrophysics
def basicAstrophysicsDegrees : ℕ := 18

-- Define the total degrees in a circle
def totalDegrees : ℕ := 360

-- Define the total budget percentage
def totalBudgetPercentage : ℕ := 100

-- Prove that the remaining percentage for food additives is 20%
theorem foodAdditivesPercentage :
  let basicAstrophysicsPercentage := (basicAstrophysicsDegrees * totalBudgetPercentage) / totalDegrees
  let totalKnownPercentage := microphotonicsPercentage + homeElectronicsPercentage + microorganismsPercentage + industrialLubricantsPercentage + basicAstrophysicsPercentage
  totalBudgetPercentage - totalKnownPercentage = 20 :=
by
  let basicAstrophysicsPercentage := (basicAstrophysicsDegrees * totalBudgetPercentage) / totalDegrees
  let totalKnownPercentage := microphotonicsPercentage + homeElectronicsPercentage + microorganismsPercentage + industrialLubricantsPercentage + basicAstrophysicsPercentage
  sorry

end foodAdditivesPercentage_l13_13248


namespace speed_of_first_car_l13_13223

theorem speed_of_first_car 
  (distance_highway : ℕ)
  (time_to_meet : ℕ)
  (speed_second_car : ℕ)
  (total_distance_covered : distance_highway = time_to_meet * 40 + time_to_meet * speed_second_car): 
  5 * 40 + 5 * 60 = distance_highway := 
by
  /-
    Given:
      - distance_highway : ℕ (The length of the highway, which is 500 miles)
      - time_to_meet : ℕ (The time after which the two cars meet, which is 5 hours)
      - speed_second_car : ℕ (The speed of the second car, which is 60 mph)
      - total_distance_covered : distance_highway = time_to_meet * speed_of_first_car + time_to_meet * speed_second_car

    We need to prove:
      - 5 * 40 + 5 * 60 = distance_highway
  -/

  sorry

end speed_of_first_car_l13_13223


namespace license_plate_combinations_l13_13283

theorem license_plate_combinations :
  let choose := Nat.choose
  let fact := Nat.factorial
  (choose 26 2) * (fact 4 / (fact 2 * fact 2)) * 10 * 9 = 175500 :=
by
  let choose := Nat.choose
  let fact := Nat.factorial
  have h1 : (choose 26 2) = 325 := by sorry
  have h2 : (fact 4 / (fact 2 * fact 2)) = 6 := by sorry
  have h3 : (325 * 6 * 10 * 9) = 175500 := by sorry
  exact h3

end license_plate_combinations_l13_13283


namespace ploughing_problem_l13_13899

theorem ploughing_problem
  (hours_per_day_group1 : ℕ)
  (days_group1 : ℕ)
  (bulls_group1 : ℕ)
  (total_fields_group2 : ℕ)
  (hours_per_day_group2 : ℕ)
  (days_group2 : ℕ)
  (bulls_group2 : ℕ)
  (fields_group1 : ℕ)
  (fields_group2 : ℕ) :
    hours_per_day_group1 = 10 →
    days_group1 = 3 →
    bulls_group1 = 10 →
    hours_per_day_group2 = 8 →
    days_group2 = 2 →
    bulls_group2 = 30 →
    fields_group2 = 32 →
    480 * fields_group1 = 300 * fields_group2 →
    fields_group1 = 20 := by
  sorry

end ploughing_problem_l13_13899


namespace compute_C_pow_50_l13_13352

def matrixC : Matrix (Fin 2) (Fin 2) ℤ := ![[5, 2], [-16, -6]]

theorem compute_C_pow_50 :
  matrixC ^ 50 = ![[-299, -100], [800, 251]] := by
  sorry

end compute_C_pow_50_l13_13352


namespace max_lateral_surface_area_of_pyramid_l13_13274

theorem max_lateral_surface_area_of_pyramid (a h : ℝ) (r : ℝ) (h_eq : 2 * a^2 + h^2 = 4) (r_eq : r = 1) :
  ∃ (a : ℝ), (a = 1) :=
by
sorry

end max_lateral_surface_area_of_pyramid_l13_13274


namespace books_before_grant_l13_13379

-- Define the conditions 
def books_purchased_with_grant : ℕ := 2647
def total_books_now : ℕ := 8582

-- Prove the number of books before the grant
theorem books_before_grant : 
  (total_books_now - books_purchased_with_grant = 5935) := 
by
  sorry

end books_before_grant_l13_13379


namespace composite_for_large_n_l13_13198

theorem composite_for_large_n (m : ℕ) (hm : m > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → Nat.Prime (2^m * 2^(2^n) + 1) = false :=
sorry

end composite_for_large_n_l13_13198


namespace annie_extracurricular_hours_l13_13972

-- Definitions based on conditions
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3
def weeks_per_semester : ℕ := 12
def weeks_off_sick : ℕ := 2

-- Total hours of extracurricular activities per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Number of active weeks before midterms
def active_weeks_before_midterms : ℕ := weeks_per_semester - weeks_off_sick

-- Total hours of extracurricular activities before midterms
def total_hours_before_midterms : ℕ := total_hours_per_week * active_weeks_before_midterms

-- Proof statement
theorem annie_extracurricular_hours : total_hours_before_midterms = 130 := by
  sorry

end annie_extracurricular_hours_l13_13972


namespace no_unhappy_days_l13_13674

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l13_13674


namespace max_value_of_squares_l13_13911

theorem max_value_of_squares (a b c d : ℝ) 
  (h1 : a + b = 18) 
  (h2 : ab + c + d = 91) 
  (h3 : ad + bc = 187) 
  (h4 : cd = 105) : 
  a^2 + b^2 + c^2 + d^2 ≤ 107 :=
sorry

end max_value_of_squares_l13_13911


namespace largest_integer_n_neg_quad_expr_l13_13739

theorem largest_integer_n_neg_quad_expr :
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, ((n^2 - 11 * n + 28 < 0) → (m < 7 ∧ m > 4) → m ≤ n) :=
by
  sorry

end largest_integer_n_neg_quad_expr_l13_13739


namespace infinite_series_sum_l13_13447

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * n - 1) / 3 ^ (n + 1)) = 2 :=
by
  sorry

end infinite_series_sum_l13_13447


namespace probability_queen_then_spade_l13_13222

theorem probability_queen_then_spade (h_deck: ℕ) (h_queens: ℕ) (h_spades: ℕ) :
  h_deck = 52 ∧ h_queens = 4 ∧ h_spades = 13 →
  (1 / 52) * (12 / 51) + (3 / 52) * (13 / 51) = 18 / 221 :=
by
  sorry

end probability_queen_then_spade_l13_13222


namespace class_8_3_final_score_is_correct_l13_13771

def class_8_3_singing_quality : ℝ := 92
def class_8_3_spirit : ℝ := 80
def class_8_3_coordination : ℝ := 70

def final_score (singing_quality spirit coordination : ℝ) : ℝ :=
  0.4 * singing_quality + 0.3 * spirit + 0.3 * coordination

theorem class_8_3_final_score_is_correct :
  final_score class_8_3_singing_quality class_8_3_spirit class_8_3_coordination = 81.8 :=
by
  sorry

end class_8_3_final_score_is_correct_l13_13771


namespace find_ellipse_l13_13576

noncomputable def standard_equation_ellipse (x y : ℝ) : Prop :=
  (x^2 / 9 + y^2 / 3 = 1)
  ∨ (x^2 / 18 + y^2 / 9 = 1)
  ∨ (y^2 / (45 / 2) + x^2 / (45 / 4) = 1)

variables 
  (P1 P2 : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (a b : ℝ)

def passes_through_points (P1 P2 : ℝ × ℝ) : Prop :=
  ∀ equation : (ℝ → ℝ → Prop), 
    equation P1.1 P1.2 ∧ equation P2.1 P2.2

def focus_conditions (focus : ℝ × ℝ) : Prop :=
  -- Condition indicating focus, relationship with the minor axis etc., will be precisely defined here
  true -- Placeholder, needs correct mathematical condition

theorem find_ellipse : 
  passes_through_points P1 P2 
  → focus_conditions focus 
  → standard_equation_ellipse x y :=
sorry

end find_ellipse_l13_13576


namespace inequality_proof_l13_13883

theorem inequality_proof
  (a b c : ℝ)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 1) :
  (a^2 + b^2 + c^2) * ((a / (b + c)) + (b / (a + c)) + (c / (a + b))) ≥ 1/2 := by
  sorry

end inequality_proof_l13_13883


namespace greatest_number_of_quarters_l13_13865

def eva_has_us_coins : ℝ := 4.80
def quarters_and_dimes_have_same_count (q : ℕ) : Prop := (0.25 * q + 0.10 * q = eva_has_us_coins)

theorem greatest_number_of_quarters : ∃ (q : ℕ), quarters_and_dimes_have_same_count q ∧ q = 13 :=
sorry

end greatest_number_of_quarters_l13_13865


namespace evaluate_star_property_l13_13195

noncomputable def star (a b : ℕ) : ℕ := b ^ a

theorem evaluate_star_property (a b c m : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hm : 0 < m) :
  (star a b ≠ star b a) ∧
  (star a (star b c) ≠ star (star a b) c) ∧
  (star a (b ^ m) ≠ star (star a m) b) ∧
  ((star a b) ^ m ≠ star a (m * b)) :=
by
  sorry

end evaluate_star_property_l13_13195


namespace two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l13_13458

theorem two_pow_2023_add_three_pow_2023_mod_seven_not_zero : (2^2023 + 3^2023) % 7 ≠ 0 := 
by sorry

end two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l13_13458


namespace quadrilateral_side_squares_inequality_l13_13707

theorem quadrilateral_side_squares_inequality :
  ∀ (x1 y1 x2 y2 : ℝ),
    0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ y1 ∧ y1 ≤ 1 ∧
    0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ y2 ∧ y2 ≤ 1 →
    2 ≤ (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ∧ 
          (x1 - 1)^2 + y1^2 + (x2 - 1)^2 + (y1 - 1)^2 + x2^2 + (y2 - 1)^2 + x1^2 + y2^2 ≤ 4 :=
by
  intro x1 y1 x2 y2 h
  sorry

end quadrilateral_side_squares_inequality_l13_13707


namespace number_of_ways_to_choose_bases_l13_13744

theorem number_of_ways_to_choose_bases : ∀ (students bases : ℕ), students = 4 → bases = 4 → (bases^students) = 256 :=
by
  intros students bases h_students h_bases
  rw [h_students, h_bases]
  exact pow_succ' 4 3

end number_of_ways_to_choose_bases_l13_13744


namespace tan_five_pi_over_four_l13_13465

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l13_13465


namespace general_formula_l13_13598

theorem general_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end general_formula_l13_13598


namespace malingerers_exposed_l13_13833

theorem malingerers_exposed (a b c : Nat) (ha : a > b) (hc : c = b + 9) :
  let aabbb := 10000 * a + 1000 * a + 100 * b + 10 * b + b
  let abccc := 10000 * a + 1000 * b + 100 * c + 10 * c + c
  (aabbb - 1 = abccc) -> abccc = 10999 :=
by
  sorry

end malingerers_exposed_l13_13833


namespace tan_product_l13_13569

theorem tan_product (t : ℝ) (h1 : 1 - 7 * t^2 + 7 * t^4 - t^6 = 0)
  (h2 : t = tan (Real.pi / 8) ∨ t = tan (3 * Real.pi / 8) ∨ t = tan (5 * Real.pi / 8)) :
  tan (Real.pi / 8) * tan (3 * Real.pi / 8) * tan (5 * Real.pi / 8) = 1 := 
sorry

end tan_product_l13_13569


namespace max_ratio_l13_13578

-- Define conditions
def conditions (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ (x^3 + y^4 = x^2 * y)

-- Statement of the theorem
theorem max_ratio (A B : ℝ) 
  (hA : ∀ x y : ℝ, conditions x y → x ≤ A)
  (hB : ∀ x y : ℝ, conditions x y → y ≤ B) :
  A / B = 729 / 1024 :=
  sorry

end max_ratio_l13_13578


namespace sum_X_Y_l13_13871

-- Define the variables and assumptions
variable (X Y : ℕ)

-- Hypotheses
axiom h1 : Y + 2 = X
axiom h2 : X + 5 = Y

-- Theorem statement
theorem sum_X_Y : X + Y = 12 := by
  sorry

end sum_X_Y_l13_13871


namespace hexagon_area_is_32_l13_13571

noncomputable def area_of_hexagon : ℝ := 
  let p0 : ℝ × ℝ := (0, 0)
  let p1 : ℝ × ℝ := (2, 4)
  let p2 : ℝ × ℝ := (5, 4)
  let p3 : ℝ × ℝ := (7, 0)
  let p4 : ℝ × ℝ := (5, -4)
  let p5 : ℝ × ℝ := (2, -4)
  -- Triangle 1: p0, p1, p2
  let area_tri1 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 2: p2, p3, p4
  let area_tri2 := 1 / 2 * (8 : ℝ) * (2 : ℝ)
  -- Triangle 3: p4, p5, p0
  let area_tri3 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 4: p1, p2, p5
  let area_tri4 := 1 / 2 * (8 : ℝ) * (3 : ℝ)
  area_tri1 + area_tri2 + area_tri3 + area_tri4

theorem hexagon_area_is_32 : area_of_hexagon = 32 := 
by
  sorry

end hexagon_area_is_32_l13_13571


namespace sarith_laps_l13_13022

theorem sarith_laps 
  (k_speed : ℝ) (s_speed : ℝ) (k_laps : ℝ) (s_laps : ℝ) (distance_ratio : ℝ) :
  k_speed = 3 * s_speed →
  distance_ratio = 1 / 2 →
  k_laps = 12 →
  s_laps = (k_laps * 2 / 3) →
  s_laps = 8 :=
by
  intros
  sorry

end sarith_laps_l13_13022


namespace value_of_m_l13_13005

theorem value_of_m (m : ℝ) (h1 : m^2 - 2 * m - 1 = 2) (h2 : m ≠ 3) : m = -1 :=
sorry

end value_of_m_l13_13005


namespace peter_class_students_l13_13616

def total_students (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ) : ℕ :=
  students_with_two_hands + students_with_one_hand + students_with_three_hands + 1

theorem peter_class_students
  (students_with_two_hands students_with_one_hand students_with_three_hands : ℕ)
  (total_hands_without_peter : ℕ) :

  students_with_two_hands = 10 →
  students_with_one_hand = 3 →
  students_with_three_hands = 1 →
  total_hands_without_peter = 20 →
  total_students students_with_two_hands students_with_one_hand students_with_three_hands = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end peter_class_students_l13_13616


namespace no_unhappy_days_l13_13670

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l13_13670


namespace percent_increase_equilateral_triangles_l13_13719

noncomputable def side_length (n : ℕ) : ℕ :=
  if n = 0 then 3 else 2 ^ n * 3

noncomputable def perimeter (n : ℕ) : ℕ :=
  3 * side_length n

noncomputable def percent_increase (initial : ℕ) (final : ℕ) : ℚ := 
  ((final - initial) / initial) * 100

theorem percent_increase_equilateral_triangles :
  percent_increase (perimeter 0) (perimeter 4) = 1500 := by
  sorry

end percent_increase_equilateral_triangles_l13_13719


namespace candidate_D_votes_l13_13903

theorem candidate_D_votes :
  let total_votes := 10000
  let invalid_votes_percentage := 0.25
  let valid_votes := (1 - invalid_votes_percentage) * total_votes
  let candidate_A_percentage := 0.40
  let candidate_B_percentage := 0.30
  let candidate_C_percentage := 0.20
  let candidate_D_percentage := 1.0 - (candidate_A_percentage + candidate_B_percentage + candidate_C_percentage)
  let candidate_D_votes := candidate_D_percentage * valid_votes
  candidate_D_votes = 750 :=
by
  sorry

end candidate_D_votes_l13_13903


namespace solve_sin_cos_l13_13506

def int_part (x : ℝ) : ℤ := ⌊x⌋

theorem solve_sin_cos (x : ℝ) :
  int_part (Real.sin x + Real.cos x) = 1 ↔ ∃ n : ℤ, 2 * Real.pi * n ≤ x ∧ x ≤ (Real.pi / 2) + 2 * Real.pi * n :=
by
  sorry

end solve_sin_cos_l13_13506


namespace has_buried_correct_number_of_bones_l13_13555

def bones_received_per_month : ℕ := 10
def number_of_months : ℕ := 5
def bones_available : ℕ := 8

def total_bones_received : ℕ := bones_received_per_month * number_of_months
def bones_buried : ℕ := total_bones_received - bones_available

theorem has_buried_correct_number_of_bones : bones_buried = 42 := by
  sorry

end has_buried_correct_number_of_bones_l13_13555


namespace num_colorings_correct_l13_13414

open Finset

def num_colorings_6_points : ℕ :=
  let total_partitions := 
    1 + -- All points in one group
    6 + -- 5-1 pattern
    15 + -- 4-2 pattern
    15 + -- 4-1-1 pattern
    10 + -- 3-3 pattern
    60 + -- 3-2-1 pattern
    20 + -- 3-1-1-1 pattern
    15 + -- 2-2-2 pattern
    45 + -- 2-2-1-1 pattern
    15 + -- 2-1-1-1-1 pattern
    1 -- 1-1-1-1-1-1 pattern
  total_partitions

theorem num_colorings_correct : num_colorings_6_points = 203 :=
by
  sorry

end num_colorings_correct_l13_13414


namespace count_of_distinct_integer_sums_of_two_special_fractions_l13_13288

open Locale.Rat

def is_special_fraction (a b : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ a + b = 18

def special_fractions : Finset ℚ :=
  (Finset.range 18).psigma (λ a => Finset.filter (λ b => is_special_fraction a b) (Finset.range 18)).map (λ ⟨a, b, hab⟩ => (a : ℚ) / (b : ℚ))

def sums_of_two_special_fractions : Finset ℚ :=
  (special_fractions.product special_fractions).map (λ p => p.1 + p.2)

def integer_sums_of_two_special_fractions : Finset ℕ :=
  sums_of_two_special_fractions.filter_map (λ q => if q.den = 1 then some q.num.to_nat else none)

theorem count_of_distinct_integer_sums_of_two_special_fractions : integer_sums_of_two_special_fractions.card = 7 :=
by
  sorry

end count_of_distinct_integer_sums_of_two_special_fractions_l13_13288


namespace mary_earnings_max_hours_l13_13037

noncomputable def earnings (hours : ℕ) : ℝ :=
  if hours <= 40 then 
    hours * 10
  else if hours <= 60 then 
    (40 * 10) + ((hours - 40) * 13)
  else 
    (40 * 10) + (20 * 13) + ((hours - 60) * 16)

theorem mary_earnings_max_hours : 
  earnings 70 = 820 :=
by
  sorry

end mary_earnings_max_hours_l13_13037


namespace missing_fraction_is_correct_l13_13932

def sum_of_fractions (x : ℚ) : Prop :=
  (1/3 : ℚ) + (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + x = (45/100 : ℚ)

theorem missing_fraction_is_correct : sum_of_fractions (27/60 : ℚ) :=
  by sorry

end missing_fraction_is_correct_l13_13932


namespace rectangular_box_diagonals_l13_13845

noncomputable def interior_diagonals_sum (a b c : ℝ) : ℝ := 4 * Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + c * a) = 150)
  (h2 : 4 * (a + b + c) = 60)
  (h3 : a * b * c = 216) :
  interior_diagonals_sum a b c = 20 * Real.sqrt 3 :=
by
  sorry

end rectangular_box_diagonals_l13_13845


namespace cat_fox_wealth_difference_l13_13946

noncomputable def prob_coin_toss : ℕ := ((1/4 : ℝ) + (1/4 : ℝ) - (1/2 : ℝ))

-- define the random variable X_n representing the "absolute difference in wealth at end of n-th day"
noncomputable def X (n : ℕ) : ℝ := sorry

-- statement of the proof problem
theorem cat_fox_wealth_difference : ∃ E : ℝ, E = 1 ∧ E = classical.some X 10 := 
sorry

end cat_fox_wealth_difference_l13_13946


namespace ana_wins_probability_l13_13020

noncomputable def probability_ana_wins : ℚ := 
  let a := (1 / 2)^5
  let r := (1 / 2)^4
  a / (1 - r)

theorem ana_wins_probability :
  probability_ana_wins = 1 / 30 :=
by
  sorry

end ana_wins_probability_l13_13020


namespace sphere_volume_diameter_l13_13613

theorem sphere_volume_diameter {D : ℝ} : 
  (D^3/2 + (1/21) * (D^3/2)) = (π * D^3 / 6) ↔ π = 22 / 7 := 
sorry

end sphere_volume_diameter_l13_13613


namespace team_selection_l13_13422

theorem team_selection (boys girls : ℕ) (choose_boys choose_girls : ℕ) 
  (boy_count girl_count : ℕ) (h1 : boy_count = 10) (h2 : girl_count = 12) 
  (h3 : choose_boys = 5) (h4 : choose_girls = 3) :
    (Nat.choose boy_count choose_boys) * (Nat.choose girl_count choose_girls) = 55440 :=
by
  rw [h1, h2, h3, h4]
  sorry

end team_selection_l13_13422


namespace value_of_a2019_l13_13597

noncomputable def a : ℕ → ℝ
| 0 => 3
| (n + 1) => 1 / (1 - a n)

theorem value_of_a2019 : a 2019 = 2 / 3 :=
sorry

end value_of_a2019_l13_13597


namespace find_f_10_l13_13261

def f (x : Int) : Int := sorry

axiom condition_1 : f 1 + 1 > 0
axiom condition_2 : ∀ x y : Int, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom condition_3 : ∀ x : Int, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := by
  sorry

end find_f_10_l13_13261


namespace coin_flip_probability_l13_13487

theorem coin_flip_probability :
  let P : (ℕ → Bool) → ℕ → ℚ := λ seq n, if seq n then (1/2 : ℚ) else (1/2 : ℚ)
  let E : (ℕ → Bool) → Prop := λ seq, seq 0 ∧ ¬ (seq 1 ∨ seq 2 ∨ seq 3 ∨ seq 4)
  (∑' (seq : ℕ → Bool), if E seq then (∏ n, P seq n) else 0) = 1/32 := sorry

end coin_flip_probability_l13_13487


namespace a2_range_l13_13775

open Nat

noncomputable def a_seq (a : ℕ → ℝ) := ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)

theorem a2_range (a : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), n > 0 → (n + 1) * a n ≥ n * a (2 * n)) 
  (h2 : ∀ (m n : ℕ), m < n → a m ≤ a n) 
  (h3 : a 1 = 2) :
  (2 < a 2) ∧ (a 2 ≤ 4) :=
sorry

end a2_range_l13_13775


namespace simplify_division_l13_13523

theorem simplify_division :
  (2 * 10^12) / (4 * 10^5 - 1 * 10^4) = 5.1282 * 10^6 :=
by
  -- problem statement
  sorry

end simplify_division_l13_13523


namespace determine_fake_coin_l13_13410

theorem determine_fake_coin (N : ℕ) : 
  (∃ (n : ℕ), N = 2 * n + 2) ↔ (∃ (n : ℕ), N = 2 * n + 2) := by 
  sorry

end determine_fake_coin_l13_13410


namespace find_m_of_transformed_point_eq_l13_13617

theorem find_m_of_transformed_point_eq (m : ℝ) (h : m + 1 = 5) : m = 4 :=
by
  sorry

end find_m_of_transformed_point_eq_l13_13617


namespace point_in_second_quadrant_l13_13180

-- Define the point coordinates in the Cartesian plane
def x_coord : ℤ := -8
def y_coord : ℤ := 2

-- Define the quadrants based on coordinate conditions
def first_quadrant : Prop := x_coord > 0 ∧ y_coord > 0
def second_quadrant : Prop := x_coord < 0 ∧ y_coord > 0
def third_quadrant : Prop := x_coord < 0 ∧ y_coord < 0
def fourth_quadrant : Prop := x_coord > 0 ∧ y_coord < 0

-- Proof statement: The point (-8, 2) lies in the second quadrant
theorem point_in_second_quadrant : second_quadrant :=
by
  sorry

end point_in_second_quadrant_l13_13180


namespace largest_n_for_divisibility_l13_13690

theorem largest_n_for_divisibility (n : ℕ) (h : (n + 20) ∣ (n^3 + 1000)) : n ≤ 180 := 
sorry

example : ∃ n : ℕ, (n + 20) ∣ (n^3 + 1000) ∧ n = 180 :=
by
  use 180
  sorry

end largest_n_for_divisibility_l13_13690


namespace find_y_value_l13_13697

theorem find_y_value : (12 ^ 3 * 6 ^ 4) / 432 = 5184 := by
  sorry

end find_y_value_l13_13697


namespace probability_not_passing_l13_13813

noncomputable def probability_of_passing : ℚ := 4 / 7

theorem probability_not_passing (h : probability_of_passing = 4 / 7) : 1 - probability_of_passing = 3 / 7 :=
by
  sorry

end probability_not_passing_l13_13813


namespace sharon_trip_distance_l13_13291

theorem sharon_trip_distance
  (x : ℝ)
  (usual_speed : ℝ := x / 180)
  (reduced_speed : ℝ := usual_speed - 1/3)
  (time_before_storm : ℝ := (x / 3) / usual_speed)
  (time_during_storm : ℝ := (2 * x / 3) / reduced_speed)
  (total_trip_time : ℝ := 276)
  (h : time_before_storm + time_during_storm = total_trip_time) :
  x = 135 :=
sorry

end sharon_trip_distance_l13_13291


namespace b_l13_13439

def initial_marbles : Nat := 24
def lost_through_hole : Nat := 4
def given_away : Nat := 2 * lost_through_hole
def eaten_by_dog : Nat := lost_through_hole / 2

theorem b {m : Nat} (h₁ : m = initial_marbles - lost_through_hole)
  (h₂ : m - given_away = m₁)
  (h₃ : m₁ - eaten_by_dog = 10) :
  m₁ - eaten_by_dog = 10 := sorry

end b_l13_13439


namespace senior_citizen_tickets_l13_13939

theorem senior_citizen_tickets (A S : ℕ) 
  (h1 : A + S = 510) 
  (h2 : 21 * A + 15 * S = 8748) : 
  S = 327 :=
by 
  -- Proof steps are omitted as instructed
  sorry

end senior_citizen_tickets_l13_13939


namespace find_f_10_l13_13264

noncomputable def f : ℤ → ℤ := sorry

axiom cond1 : f 1 + 1 > 0
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := 
by
  sorry 

end find_f_10_l13_13264


namespace angle_sum_90_l13_13346

theorem angle_sum_90 (A B : ℝ) (h : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2) : A + B = Real.pi / 2 :=
sorry

end angle_sum_90_l13_13346


namespace ratio_of_shares_l13_13199

theorem ratio_of_shares 
    (sheila_share : ℕ → ℕ)
    (rose_share : ℕ)
    (total_rent : ℕ) 
    (h1 : ∀ P, sheila_share P = 5 * P)
    (h2 : rose_share = 1800)
    (h3 : ∀ P, sheila_share P + P + rose_share = total_rent) 
    (h4 : total_rent = 5400) :
    ∃ P, 1800 / P = 3 := 
by 
  sorry

end ratio_of_shares_l13_13199


namespace altered_solution_detergent_volume_l13_13090

theorem altered_solution_detergent_volume 
  (bleach : ℕ)
  (detergent : ℕ)
  (water : ℕ)
  (h1 : bleach / detergent = 4 / 40)
  (h2 : detergent / water = 40 / 100)
  (ratio_tripled : 3 * (bleach / detergent) = bleach / detergent)
  (ratio_halved : (detergent / water) / 2 = (detergent / water))
  (altered_water : water = 300) : 
  detergent = 60 := 
  sorry

end altered_solution_detergent_volume_l13_13090


namespace third_side_length_l13_13489

noncomputable def calc_third_side (a b : ℕ) (hypotenuse : Bool) : ℝ :=
if hypotenuse then
  Real.sqrt (a^2 + b^2)
else
  Real.sqrt (abs (a^2 - b^2))

theorem third_side_length (a b : ℕ) (h_right_triangle : (a = 8 ∧ b = 15)) :
  calc_third_side a b true = 17 ∨ calc_third_side 15 8 false = Real.sqrt 161 :=
by {
  sorry
}

end third_side_length_l13_13489


namespace base_conversion_addition_l13_13463

theorem base_conversion_addition :
  (214 % 8 / 32 % 5 + 343 % 9 / 133 % 4) = 9134 / 527 :=
by sorry

end base_conversion_addition_l13_13463


namespace find_p_q_l13_13752

noncomputable def cubicFunction (p q : ℝ) (x : ℂ) : ℂ :=
  2 * x^3 + p * x^2 + q * x

theorem find_p_q (p q : ℝ) :
  cubicFunction p q (2 * Complex.I - 3) = 0 ∧ 
  cubicFunction p q (-2 * Complex.I - 3) = 0 → 
  p = 12 ∧ q = 26 :=
by
  sorry

end find_p_q_l13_13752


namespace value_of_a_l13_13145

noncomputable def z (a : ℝ) : ℂ := (1 + a * complex.I) / (1 - complex.I)

theorem value_of_a (a : ℝ) (hz: z a = (b : ℂ) ∧ b.re = 0) : a = 1 := by
sorry

end value_of_a_l13_13145


namespace lampshire_parade_group_max_members_l13_13046

theorem lampshire_parade_group_max_members 
  (n : ℕ) 
  (h1 : 30 * n % 31 = 7)
  (h2 : 30 * n % 17 = 0)
  (h3 : 30 * n < 1500) :
  30 * n = 1020 :=
sorry

end lampshire_parade_group_max_members_l13_13046


namespace shopkeeper_gain_l13_13406

noncomputable def gain_percent (cost_per_kg : ℝ) (claimed_weight : ℝ) (actual_weight : ℝ) : ℝ :=
  let gain := cost_per_kg - (actual_weight / claimed_weight) * cost_per_kg
  (gain / ((actual_weight / claimed_weight) * cost_per_kg)) * 100

theorem shopkeeper_gain (c : ℝ) (cw aw : ℝ) (h : c = 1) (hw : cw = 1) (ha : aw = 0.75) : 
  gain_percent c cw aw = 33.33 :=
by sorry

end shopkeeper_gain_l13_13406


namespace cube_surface_area_l13_13058

theorem cube_surface_area (v : ℝ) (h : v = 1000) : ∃ (s : ℝ), s^3 = v ∧ 6 * s^2 = 600 :=
by
  sorry

end cube_surface_area_l13_13058


namespace probability_of_one_exactly_four_times_l13_13610

def roll_probability := (1 : ℝ) / 6
def non_one_probability := (5 : ℝ) / 6

lemma prob_roll_one_four_times :
  ∑ x in {1, 2, 3, 4, 5}, 
      roll_probability^4 * non_one_probability = 
    5 * (roll_probability^4 * non_one_probability) :=
by
  sorry

theorem probability_of_one_exactly_four_times :
  (5 : ℝ) * roll_probability^4 * non_one_probability = (25 : ℝ) / 7776 :=
by
  have key := prob_roll_one_four_times
  sorry

end probability_of_one_exactly_four_times_l13_13610


namespace factorial_expression_calculation_l13_13113

theorem factorial_expression_calculation :
  7 * (Nat.factorial 7) + 5 * (Nat.factorial 6) - 6 * (Nat.factorial 5) = 7920 :=
by
  sorry

end factorial_expression_calculation_l13_13113


namespace annie_extracurricular_hours_l13_13971

-- Definitions based on conditions
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3
def weeks_per_semester : ℕ := 12
def weeks_off_sick : ℕ := 2

-- Total hours of extracurricular activities per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Number of active weeks before midterms
def active_weeks_before_midterms : ℕ := weeks_per_semester - weeks_off_sick

-- Total hours of extracurricular activities before midterms
def total_hours_before_midterms : ℕ := total_hours_per_week * active_weeks_before_midterms

-- Proof statement
theorem annie_extracurricular_hours : total_hours_before_midterms = 130 := by
  sorry

end annie_extracurricular_hours_l13_13971


namespace regular_polygon_sides_l13_13683

theorem regular_polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : 
  n = 6 :=
sorry

end regular_polygon_sides_l13_13683


namespace count_boys_correct_l13_13996

def total_vans : ℕ := 5
def students_per_van : ℕ := 28
def number_of_girls : ℕ := 80

theorem count_boys_correct : 
  (total_vans * students_per_van) - number_of_girls = 60 := 
by
  sorry

end count_boys_correct_l13_13996


namespace sqrt_6_between_2_and_3_l13_13988

theorem sqrt_6_between_2_and_3 : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by
  sorry

end sqrt_6_between_2_and_3_l13_13988


namespace positive_real_solution_l13_13896

def polynomial (x : ℝ) : ℝ := x^4 + 10*x^3 - 2*x^2 + 12*x - 9

theorem positive_real_solution (h : polynomial 1 = 0) : polynomial 1 > 0 := sorry

end positive_real_solution_l13_13896


namespace paint_mixer_days_l13_13032

/-- Making an equal number of drums of paint each day, a paint mixer takes three days to make 18 drums of paint.
    We want to determine how many days it will take for him to make 360 drums of paint. -/
theorem paint_mixer_days (n : ℕ) (h1 : n > 0) 
  (h2 : 3 * n = 18) : 
  360 / n = 60 := by
  sorry

end paint_mixer_days_l13_13032


namespace youngest_child_age_l13_13056

variables (child_ages : Fin 5 → ℕ)

def child_ages_eq_intervals (x : ℕ) : Prop :=
  child_ages 0 = x ∧ child_ages 1 = x + 8 ∧ child_ages 2 = x + 16 ∧ child_ages 3 = x + 24 ∧ child_ages 4 = x + 32

def sum_of_ages_eq (child_ages : Fin 5 → ℕ) (sum : ℕ) : Prop :=
  (Finset.univ : Finset (Fin 5)).sum child_ages = sum

theorem youngest_child_age (child_ages : Fin 5 → ℕ) (h1 : ∃ x, child_ages_eq_intervals child_ages x) (h2 : sum_of_ages_eq child_ages 90) :
  ∃ x, x = 2 ∧ child_ages 0 = x :=
sorry

end youngest_child_age_l13_13056


namespace find_y_l13_13246

-- Definitions for the given conditions
variable (p y : ℕ) (h : p > 30)  -- Natural numbers, noting p > 30 condition

-- The initial amount of acid in ounces
def initial_acid_amount : ℕ := p * p / 100

-- The amount of acid after adding y ounces of water
def final_acid_amount : ℕ := (p - 15) * (p + y) / 100

-- Lean statement to prove y = 15p/(p-15)
theorem find_y (h1 : p > 30) (h2 : initial_acid_amount p = final_acid_amount p y) :
  y = 15 * p / (p - 15) :=
sorry

end find_y_l13_13246


namespace problem1_solution_problem2_solution_problem3_solution_l13_13799

-- Problem 1
theorem problem1_solution (x : ℝ) :
  (6 * x - 1) ^ 2 = 25 ↔ (x = 1 ∨ x = -2 / 3) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) :
  4 * x^2 - 1 = 12 * x ↔ (x = 3 / 2 + (Real.sqrt 10) / 2 ∨ x = 3 / 2 - (Real.sqrt 10) / 2) :=
sorry

-- Problem 3
theorem problem3_solution (x : ℝ) :
  x * (x - 7) = 8 * (7 - x) ↔ (x = 7 ∨ x = -8) :=
sorry

end problem1_solution_problem2_solution_problem3_solution_l13_13799


namespace find_k_l13_13657

noncomputable def distance_x (x : ℝ) := 5
noncomputable def distance_y (x k : ℝ) := |x^2 - k|
noncomputable def total_distance (x k : ℝ) := distance_x x + distance_y x k

theorem find_k (x k : ℝ) (hk : distance_y x k = 2 * distance_x x) (htot : total_distance x k = 30) :
  k = x^2 - 10 :=
sorry

end find_k_l13_13657


namespace extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l13_13588

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x / Real.exp x + x^2 / 2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := max (f a x) (g x)

theorem extreme_values_for_f_when_a_is_one :
  (∀ x : ℝ, (f 1 x) ≤ 0) ∧ f 1 0 = 0 ∧ f 1 1 = (1 / Real.exp 1) - 1 / 2 :=
sorry

theorem number_of_zeros_of_h (a : ℝ) :
  (0 ≤ a → 
   if 1 < a ∧ a < Real.exp 1 / 2 then
     ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ h a x1 = 0 ∧ h a x2 = 0
   else if 0 ≤ a ∧ a ≤ 1 ∨ a = Real.exp 1 / 2 then
     ∃ x : ℝ, 0 < x ∧ x < 1 ∧ h a x = 0
   else
     ∀ x : ℝ, x > 0 → h a x ≠ 0) :=
sorry

end extreme_values_for_f_when_a_is_one_number_of_zeros_of_h_l13_13588


namespace suitcase_combinations_l13_13860

def count_odd_numbers (n : Nat) : Nat := n / 2

def count_multiples_of_4 (n : Nat) : Nat := n / 4

def count_multiples_of_5 (n : Nat) : Nat := n / 5

theorem suitcase_combinations : count_odd_numbers 40 * count_multiples_of_4 40 * count_multiples_of_5 40 = 1600 :=
by
  sorry

end suitcase_combinations_l13_13860


namespace angle_at_3_40_pm_is_130_degrees_l13_13067

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l13_13067


namespace percentage_B_to_C_l13_13173

variables (total_students : ℕ)
variables (pct_A pct_B pct_C pct_A_to_C pct_B_to_C : ℝ)

-- Given conditions
axiom total_students_eq_100 : total_students = 100
axiom pct_A_eq_60 : pct_A = 60
axiom pct_B_eq_40 : pct_B = 40
axiom pct_A_to_C_eq_30 : pct_A_to_C = 30
axiom pct_C_eq_34 : pct_C = 34

-- Proof goal
theorem percentage_B_to_C :
  pct_B_to_C = 40 :=
sorry

end percentage_B_to_C_l13_13173


namespace square_diagonal_l13_13848

theorem square_diagonal (P : ℝ) (d : ℝ) (hP : P = 200 * Real.sqrt 2) :
  d = 100 :=
by
  sorry

end square_diagonal_l13_13848


namespace sum_of_solutions_of_equation_l13_13873

theorem sum_of_solutions_of_equation :
  let f := (fun x : ℝ => (x - 4) ^ 2)
  ∃ S : Set ℝ, (S = {x | f x = 16}) ∧ (∑ s in S, s) = 8 := 
by
  sorry

end sum_of_solutions_of_equation_l13_13873


namespace teresa_spends_40_dollars_l13_13378

-- Definitions of the conditions
def sandwich_cost : ℝ := 7.75
def num_sandwiches : ℝ := 2

def salami_cost : ℝ := 4.00

def brie_cost : ℝ := 3 * salami_cost

def olives_cost_per_pound : ℝ := 10.00
def amount_of_olives : ℝ := 0.25

def feta_cost_per_pound : ℝ := 8.00
def amount_of_feta : ℝ := 0.5

def french_bread_cost : ℝ := 2.00

-- Total cost calculation
def total_cost : ℝ :=
  num_sandwiches * sandwich_cost + salami_cost + brie_cost + olives_cost_per_pound * amount_of_olives + feta_cost_per_pound * amount_of_feta + french_bread_cost

-- Proof statement
theorem teresa_spends_40_dollars :
  total_cost = 40.0 :=
by
  sorry

end teresa_spends_40_dollars_l13_13378


namespace fraction_arithmetic_l13_13603

theorem fraction_arithmetic : ( (4 / 5 - 1 / 10) / (2 / 5) ) = 7 / 4 :=
  sorry

end fraction_arithmetic_l13_13603


namespace mans_rate_is_19_l13_13273

-- Define the given conditions
def downstream_speed : ℝ := 25
def upstream_speed : ℝ := 13

-- Define the man's rate in still water and state the theorem
theorem mans_rate_is_19 : (downstream_speed + upstream_speed) / 2 = 19 := by
  -- Proof goes here
  sorry

end mans_rate_is_19_l13_13273


namespace purchase_price_l13_13705

-- Define the context and conditions 
variables (P S : ℝ)
-- Define the conditions
axiom cond1 : S = P + 0.5 * S
axiom cond2 : S - P = 100

-- Define the main theorem
theorem purchase_price : P = 100 :=
by sorry

end purchase_price_l13_13705


namespace third_consecutive_even_l13_13532

theorem third_consecutive_even {a b c d : ℕ} (h1 : b = a + 2) (h2 : c = a + 4) (h3 : d = a + 6) (h_sum : a + b + c + d = 52) : c = 14 :=
by
  sorry

end third_consecutive_even_l13_13532


namespace shopkeeper_weight_l13_13708

/-- A shopkeeper sells his goods at cost price but uses a certain weight instead of kilogram weight.
    His profit percentage is 25%. Prove that the weight he uses is 0.8 kilograms. -/
theorem shopkeeper_weight (c s p : ℝ) (x : ℝ) (h1 : s = c * (1 + p / 100))
  (h2 : p = 25) (h3 : c = 1) (h4 : s = 1.25) : x = 0.8 :=
by
  sorry

end shopkeeper_weight_l13_13708


namespace points_on_circle_l13_13998

theorem points_on_circle (t : ℝ) : 
  ( (2 - 3 * t^2) / (2 + t^2) )^2 + ( 3 * t / (2 + t^2) )^2 = 1 := 
by 
  sorry

end points_on_circle_l13_13998


namespace original_savings_l13_13365

variable (S : ℝ)

noncomputable def savings_after_expenditures :=
  S - 0.20 * S - 0.40 * S - 1500 

theorem original_savings : savings_after_expenditures S = 2900 → S = 11000 :=
by
  intro h
  rw [savings_after_expenditures, sub_sub_sub_cancel_right] at h
  sorry

end original_savings_l13_13365


namespace tens_digit_of_8_pow_306_l13_13540

theorem tens_digit_of_8_pow_306 : ∀ n,  n % 6 = 0 -> (∃ m, 8 ^ n % 100 = m ∧ m / 10 % 10 = 6) :=
by
  intro n hn
  -- The corresponding exponent in the cycle of last two digits of 8^k in 68, 44, 52, 16, 28, 24
  have hcycle : 8^6 % 100 = 64 := by -- The sixth power of 8 mod cycle length (6)
    norm_num [pow_succ]
  have hmod : (306 % 6 = 0) := by -- This is given as the precursor condition
    rfl
  use (8 ^ 12 % 100); 
  split;
  apply hcycle;
  sorry

end tens_digit_of_8_pow_306_l13_13540


namespace total_area_of_map_l13_13429

def level1_area : ℕ := 40 * 20
def level2_area : ℕ := 15 * 15
def level3_area : ℕ := (25 * 12) / 2

def total_area : ℕ := level1_area + level2_area + level3_area

theorem total_area_of_map : total_area = 1175 := by
  -- Proof to be completed
  sorry

end total_area_of_map_l13_13429


namespace simplify_and_evaluate_l13_13200

theorem simplify_and_evaluate :
  ∀ (x y : ℝ), x = -1/2 → y = 3 → 3 * (2 * x^2 * y - x * y^2) - 2 * (-2 * y^2 * x + x^2 * y) = -3/2 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_l13_13200


namespace cards_not_in_box_correct_l13_13919

-- Total number of cards Robie had at the beginning.
def total_cards : ℕ := 75

-- Number of cards in each box.
def cards_per_box : ℕ := 10

-- Number of boxes Robie gave away.
def boxes_given_away : ℕ := 2

-- Number of boxes Robie has with him.
def boxes_with_rob : ℕ := 5

-- The number of cards not placed in a box.
def cards_not_in_box : ℕ :=
  total_cards - (boxes_given_away * cards_per_box + boxes_with_rob * cards_per_box)

theorem cards_not_in_box_correct : cards_not_in_box = 5 :=
by
  unfold cards_not_in_box
  unfold total_cards
  unfold boxes_given_away
  unfold cards_per_box
  unfold boxes_with_rob
  sorry

end cards_not_in_box_correct_l13_13919


namespace inequality_cube_l13_13027

theorem inequality_cube (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end inequality_cube_l13_13027


namespace candy_pieces_given_l13_13778

theorem candy_pieces_given (initial total : ℕ) (h1 : initial = 68) (h2 : total = 93) :
  total - initial = 25 :=
by
  sorry

end candy_pieces_given_l13_13778


namespace weight_of_apples_l13_13539

-- Definitions based on conditions
def total_weight : ℕ := 10
def weight_orange : ℕ := 1
def weight_grape : ℕ := 3
def weight_strawberry : ℕ := 3

-- Prove that the weight of apples is 3 kilograms
theorem weight_of_apples : (total_weight - (weight_orange + weight_grape + weight_strawberry)) = 3 :=
by
  sorry

end weight_of_apples_l13_13539


namespace election_win_percentage_l13_13622

theorem election_win_percentage (total_votes : ℕ) (james_percentage : ℝ) (additional_votes_needed : ℕ) (votes_needed_to_win_percentage : ℝ) :
    total_votes = 2000 →
    james_percentage = 0.005 →
    additional_votes_needed = 991 →
    votes_needed_to_win_percentage = (1001 / 2000) * 100 →
    votes_needed_to_win_percentage > 50.05 :=
by
  intros h_total_votes h_james_percentage h_additional_votes_needed h_votes_needed_to_win_percentage
  sorry

end election_win_percentage_l13_13622


namespace dice_total_correct_l13_13185

-- Define the problem conditions
def IvanDice (x : ℕ) : ℕ := x
def JerryDice (x : ℕ) : ℕ := (1 / 2 * x) ^ 2

-- Define the total dice function
def totalDice (x : ℕ) : ℕ := IvanDice x + JerryDice x

-- The theorem to prove the answer
theorem dice_total_correct (x : ℕ) : totalDice x = x + (1 / 4) * x ^ 2 := 
  sorry

end dice_total_correct_l13_13185


namespace series_value_l13_13122

theorem series_value : ∑ n in Nat.range ∞, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
by
  sorry

end series_value_l13_13122


namespace geometric_sequence_sum_l13_13997

-- Defining the geometric sequence related properties and conditions
theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * r) → 
  S 3 = a 0 + a 1 + a 2 →
  S 6 = a 3 + a 4 + a 5 →
  S 12 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 →
  S 3 = 3 →
  S 6 = 6 →
  S 12 = 45 :=
by
  sorry

end geometric_sequence_sum_l13_13997


namespace can_determine_counterfeit_coin_l13_13534

/-- 
Given 101 coins where 50 are counterfeit and each counterfeit coin 
differs by 1 gram from the genuine ones, prove that Petya can 
determine if a given coin is counterfeit with a single weighing 
using a balance scale.
-/
theorem can_determine_counterfeit_coin :
  ∃ (coins : Fin 101 → ℤ), 
    (∃ i : Fin 101, (1 ≤ i ∧ i ≤ 50 → coins i = 1) ∧ (51 ≤ i ∧ i ≤ 101 → coins i = 0)) →
    (∃ (b : ℤ), (0 < b → b ∣ 1) ∧ (¬(0 < b → b ∣ 1) → coins 101 = b)) :=
by
  sorry

end can_determine_counterfeit_coin_l13_13534


namespace sunlovers_happy_days_l13_13677

open Nat

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end sunlovers_happy_days_l13_13677


namespace inequality1_inequality2_l13_13800

theorem inequality1 (x : ℝ) : 2 * x - 1 > x - 3 → x > -2 := by
  sorry

theorem inequality2 (x : ℝ) : 
  (x - 3 * (x - 2) ≥ 4) ∧ ((x - 1) / 5 < (x + 1) / 2) → -7 / 3 < x ∧ x ≤ 1 := by
  sorry

end inequality1_inequality2_l13_13800


namespace nec_but_not_suff_condition_l13_13831

variables {p q : Prop}

theorem nec_but_not_suff_condition (hp : ¬p) : 
  (p ∨ q → False) ↔ (¬p) ∧ ¬(¬p → p ∨ q) :=
by {
  sorry
}

end nec_but_not_suff_condition_l13_13831


namespace prime_divisor_form_l13_13509

theorem prime_divisor_form {p q : ℕ} (hp : Nat.Prime p) (hpgt2 : p > 2) (hq : Nat.Prime q) (hq_dvd : q ∣ 2^p - 1) : 
  ∃ k : ℕ, q = 2 * k * p + 1 := 
sorry

end prime_divisor_form_l13_13509


namespace relatively_prime_solutions_l13_13467

theorem relatively_prime_solutions  (x y : ℤ) (h_rel_prime : gcd x y = 1) : 
  2 * (x^3 - x) = 5 * (y^3 - y) ↔ 
  (x = 0 ∧ (y = 1 ∨ y = -1)) ∨ 
  (x = 1 ∧ y = 0) ∨
  (x = -1 ∧ y = 0) ∨
  (x = 4 ∧ (y = 3 ∨ y = -3)) ∨ 
  (x = -4 ∧ (y = -3 ∨ y = 3)) ∨
  (x = 1 ∧ y = -1) ∨
  (x = -1 ∧ y = 1) ∨
  (x = 0 ∧ y = 0) :=
by sorry

end relatively_prime_solutions_l13_13467


namespace arcsin_neg_one_eq_neg_pi_div_two_l13_13559

theorem arcsin_neg_one_eq_neg_pi_div_two : arcsin (-1) = - (Real.pi / 2) := sorry

end arcsin_neg_one_eq_neg_pi_div_two_l13_13559


namespace total_surface_area_prime_rectangular_solid_l13_13731

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Prime n

def prime_edge_lengths (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c

def volume (a b c : ℕ) : ℕ := a * b * c

def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

-- The main theorem statement
theorem total_surface_area_prime_rectangular_solid :
  ∃ (a b c : ℕ), prime_edge_lengths a b c ∧ volume a b c = 105 ∧ surface_area a b c = 142 :=
sorry

end total_surface_area_prime_rectangular_solid_l13_13731


namespace cos_solution_count_l13_13605

theorem cos_solution_count :
  ∃ n : ℕ, n = 2 ∧ 0 ≤ x ∧ x < 360 → cos x = 0.45 :=
by
  sorry

end cos_solution_count_l13_13605


namespace minimum_value_condition_l13_13315

theorem minimum_value_condition (a b : ℝ) (h : 16 * a ^ 2 + 2 * a + 8 * a * b + b ^ 2 - 1 = 0) : 
  ∃ m : ℝ, m = 3 * a + b ∧ m ≥ -1 :=
sorry

end minimum_value_condition_l13_13315


namespace factorize_expression_l13_13734

theorem factorize_expression (x : ℝ) : 4 * x ^ 2 - 2 * x = 2 * x * (2 * x - 1) :=
by
  sorry

end factorize_expression_l13_13734


namespace claudia_ratio_of_kids_l13_13118

def claudia_art_class :=
  let saturday_kids := 20
  let sunday_kids := (300 - saturday_kids * 10) / 10
  sunday_kids / saturday_kids = 1 / 2

theorem claudia_ratio_of_kids :
  let saturday_kids := 20
  let sunday_kids := (300 - saturday_kids * 10) / 10
  (sunday_kids / saturday_kids = 1 / 2) :=
by
  sorry

end claudia_ratio_of_kids_l13_13118


namespace obtuse_triangle_of_sin_cos_sum_l13_13587

theorem obtuse_triangle_of_sin_cos_sum
  (A : ℝ) (hA : 0 < A ∧ A < π) 
  (h_eq : Real.sin A + Real.cos A = 12 / 25) :
  π / 2 < A ∧ A < π :=
sorry

end obtuse_triangle_of_sin_cos_sum_l13_13587


namespace statement_2_statement_3_l13_13141

variable {α : Type*} [LinearOrderedField α]

-- Given a quadratic function
def quadratic (a b c x : α) : α :=
  a * x^2 + b * x + c

-- Statement 2
theorem statement_2 (a b c p q : α) (hpq : p ≠ q) :
  quadratic a b c p = quadratic a b c q → quadratic a b c (p + q) = c :=
sorry

-- Statement 3
theorem statement_3 (a b c p q : α) (hpq : p ≠ q) :
  quadratic a b c (p + q) = c → (p + q = 0 ∨ quadratic a b c p = quadratic a b c q) :=
sorry

end statement_2_statement_3_l13_13141


namespace total_weight_fruits_in_good_condition_l13_13275

theorem total_weight_fruits_in_good_condition :
  let oranges_initial := 600
  let bananas_initial := 400
  let apples_initial := 300
  let avocados_initial := 200
  let grapes_initial := 100
  let pineapples_initial := 50

  let oranges_rotten := 0.15 * oranges_initial
  let bananas_rotten := 0.05 * bananas_initial
  let apples_rotten := 0.08 * apples_initial
  let avocados_rotten := 0.10 * avocados_initial
  let grapes_rotten := 0.03 * grapes_initial
  let pineapples_rotten := 0.20 * pineapples_initial

  let oranges_good := oranges_initial - oranges_rotten
  let bananas_good := bananas_initial - bananas_rotten
  let apples_good := apples_initial - apples_rotten
  let avocados_good := avocados_initial - avocados_rotten
  let grapes_good := grapes_initial - grapes_rotten
  let pineapples_good := pineapples_initial - pineapples_rotten

  let weight_per_orange := 150 / 1000 -- kg
  let weight_per_banana := 120 / 1000 -- kg
  let weight_per_apple := 100 / 1000 -- kg
  let weight_per_avocado := 80 / 1000 -- kg
  let weight_per_grape := 5 / 1000 -- kg
  let weight_per_pineapple := 1 -- kg

  oranges_good * weight_per_orange +
  bananas_good * weight_per_banana +
  apples_good * weight_per_apple +
  avocados_good * weight_per_avocado +
  grapes_good * weight_per_grape +
  pineapples_good * weight_per_pineapple = 204.585 :=
by
  sorry

end total_weight_fruits_in_good_condition_l13_13275


namespace paul_digs_the_well_l13_13186

theorem paul_digs_the_well (P : ℝ) (h1 : 1 / 16 + 1 / P + 1 / 48 = 1 / 8) : P = 24 :=
sorry

end paul_digs_the_well_l13_13186


namespace discarded_marble_weight_l13_13159

-- Define the initial weight of the marble block and the weights of the statues
def initial_weight : ℕ := 80
def weight_statue_1 : ℕ := 10
def weight_statue_2 : ℕ := 18
def weight_statue_3 : ℕ := 15
def weight_statue_4 : ℕ := 15

-- The proof statement: the discarded weight of marble is 22 pounds.
theorem discarded_marble_weight :
  initial_weight - (weight_statue_1 + weight_statue_2 + weight_statue_3 + weight_statue_4) = 22 :=
by
  sorry

end discarded_marble_weight_l13_13159


namespace sum_converges_to_one_l13_13983

noncomputable def series_sum (n: ℕ) : ℝ :=
  if n ≥ 2 then (6 * n^3 - 2 * n^2 - 2 * n + 1) / (n^6 - 2 * n^5 + 2 * n^4 - n^3 + n^2 - 2 * n)
  else 0

theorem sum_converges_to_one : 
  (∑' n, series_sum n) = 1 := by
  sorry

end sum_converges_to_one_l13_13983


namespace pyramid_property_l13_13968

-- Define the areas of the faces of the right-angled triangular pyramid.
variables (S_ABC S_ACD S_ADB S_BCD : ℝ)

-- Define the condition that the areas correspond to a right-angled triangular pyramid.
def right_angled_triangular_pyramid (S_ABC S_ACD S_ADB S_BCD : ℝ) : Prop :=
  S_BCD^2 = S_ABC^2 + S_ACD^2 + S_ADB^2

-- State the theorem to be proven.
theorem pyramid_property : right_angled_triangular_pyramid S_ABC S_ACD S_ADB S_BCD :=
sorry

end pyramid_property_l13_13968


namespace write_as_sum_1800_l13_13001

/-- The number of ways to write 1800 as the sum of 1s, 2s, and 3s, ignoring order, is 4^300. -/
theorem write_as_sum_1800 : 
  (∑ (n : ℕ) in finset.range 1801, if ∃ (s₁ s₂ s₃ : ℕ), s₁ + s₂ + s₃ = n ∧ s₁ + 2 * s₂ + 3 * s₃ = 1800 then 1 else 0) = 4^300 :=
sorry

end write_as_sum_1800_l13_13001


namespace find_n_in_geometric_sequence_l13_13343

def geometric_sequence (an : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ q : ℝ, ∀ k : ℕ, an (k + 1) = an k * q

theorem find_n_in_geometric_sequence (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h3 : ∀ q : ℝ, a n = a 1 * a 2 * a 3 * a 4 * a 5) :
  n = 11 :=
sorry

end find_n_in_geometric_sequence_l13_13343


namespace tetrahedron_vertex_equality_l13_13375

theorem tetrahedron_vertex_equality
  (r1 r2 r3 r4 j1 j2 j3 j4 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hr3 : r3 > 0) (hr4 : r4 > 0)
  (hj1 : j1 > 0) (hj2 : j2 > 0) (hj3 : j3 > 0) (hj4 : j4 > 0) 
  (h1 : r2 * r3 + r3 * r4 + r4 * r2 = j2 * j3 + j3 * j4 + j4 * j2)
  (h2 : r1 * r3 + r3 * r4 + r4 * r1 = j1 * j3 + j3 * j4 + j4 * j1)
  (h3 : r1 * r2 + r2 * r4 + r4 * r1 = j1 * j2 + j2 * j4 + j4 * j1)
  (h4 : r1 * r2 + r2 * r3 + r3 * r1 = j1 * j2 + j2 * j3 + j3 * j1) :
  r1 = j1 ∧ r2 = j2 ∧ r3 = j3 ∧ r4 = j4 := by
  sorry

end tetrahedron_vertex_equality_l13_13375


namespace find_distance_l13_13238

-- Definitions of given conditions
def speed : ℝ := 65 -- km/hr
def time  : ℝ := 3  -- hr

-- Statement: The distance is 195 km given the speed and time.
theorem find_distance (speed : ℝ) (time : ℝ) : (speed * time = 195) :=
by
  sorry

end find_distance_l13_13238


namespace remainder_division_39_l13_13959

theorem remainder_division_39 (N : ℕ) (k m R1 : ℕ) (hN1 : N = 39 * k + R1) (hN2 : N % 13 = 5) (hR1_lt_39 : R1 < 39) :
  R1 = 5 :=
by sorry

end remainder_division_39_l13_13959


namespace max_distance_from_curve_to_line_l13_13150

theorem max_distance_from_curve_to_line
  (θ : ℝ) (t : ℝ)
  (C_polar_eqn : ∀ θ, ∃ (ρ : ℝ), ρ = 2 * Real.cos θ)
  (line_eqn : ∀ t, ∃ (x y : ℝ), x = -1 + t ∧ y = 2 * t) :
  ∃ (max_dist : ℝ), max_dist = (4 * Real.sqrt 5 + 5) / 5 := sorry

end max_distance_from_curve_to_line_l13_13150


namespace sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l13_13091

theorem sufficient_but_not_necessary_condition_for_x_lt_3 (x : ℝ) : |x - 1| < 2 → x < 3 :=
by {
  sorry
}

theorem not_necessary_condition_for_x_lt_3 (x : ℝ) : (x < 3) → ¬(-1 < x ∧ x < 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_for_x_lt_3_not_necessary_condition_for_x_lt_3_l13_13091


namespace probability_two_boys_and_three_girls_l13_13699

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_boys_and_three_girls :
  binomial_probability 5 2 0.5 = 0.3125 :=
by
  sorry

end probability_two_boys_and_three_girls_l13_13699


namespace problem1_l13_13952

theorem problem1 (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) := by 
  sorry

end problem1_l13_13952


namespace total_earnings_correct_l13_13435

-- Define the conditions as initial parameters

def ticket_price : ℕ := 3
def weekday_visitors_per_day : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300

def total_weekday_visitors : ℕ := 5 * weekday_visitors_per_day
def total_weekend_visitors : ℕ := saturday_visitors + sunday_visitors
def total_visitors : ℕ := total_weekday_visitors + total_weekend_visitors

def total_earnings := total_visitors * ticket_price

-- Prove that the total earnings of the amusement park in a week is $3000
theorem total_earnings_correct : total_earnings = 3000 :=
by
  sorry

end total_earnings_correct_l13_13435


namespace clothing_needed_for_washer_l13_13231

def total_blouses : ℕ := 12
def total_skirts : ℕ := 6
def total_slacks : ℕ := 8

def blouses_in_hamper : ℕ := total_blouses * 75 / 100
def skirts_in_hamper : ℕ := total_skirts * 50 / 100
def slacks_in_hamper : ℕ := total_slacks * 25 / 100

def total_clothing_in_hamper : ℕ := blouses_in_hamper + skirts_in_hamper + slacks_in_hamper

theorem clothing_needed_for_washer : total_clothing_in_hamper = 14 := by
  rw [total_clothing_in_hamper, blouses_in_hamper, skirts_in_hamper, slacks_in_hamper]
  rw [Nat.mul_div_cancel_left _ (Nat.pos_of_ne_zero (by decide)), Nat.mul_div_cancel_left _ (by decide), Nat.mul_div_cancel_left _ (by decide)]
  exact rfl

end clothing_needed_for_washer_l13_13231


namespace gcd_3375_9180_l13_13575

-- Definition of gcd and the problem condition
theorem gcd_3375_9180 : Nat.gcd 3375 9180 = 135 := by
  sorry -- Proof can be filled in with the steps using the Euclidean algorithm

end gcd_3375_9180_l13_13575


namespace general_term_of_sequence_l13_13586

theorem general_term_of_sequence (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = (a n) ^ 2) :
  ∀ n : ℕ, n > 0 → a n = 3 ^ (2 ^ (n - 1)) :=
by
  intros n hn
  sorry

end general_term_of_sequence_l13_13586


namespace smallest_distance_l13_13194

open Complex

noncomputable def a := 2 + 4 * Complex.I
noncomputable def b := 8 + 6 * Complex.I

theorem smallest_distance (z w : ℂ)
    (hz : abs (z - a) = 2)
    (hw : abs (w - b) = 4) :
    abs (z - w) ≥ 2 * Real.sqrt 10 - 6 := by
  sorry

end smallest_distance_l13_13194


namespace hyperbola_eccentricity_l13_13153

theorem hyperbola_eccentricity (m : ℝ) (h1: ∃ x y : ℝ, (x^2 / 3) - (y^2 / m) = 1) (h2: ∀ a b : ℝ, a^2 = 3 ∧ b^2 = m ∧ (2 = Real.sqrt (1 + b^2 / a^2))) : m = -9 := 
sorry

end hyperbola_eccentricity_l13_13153


namespace functions_same_l13_13827

theorem functions_same (x : ℝ) : (∀ x, (y = x) → (∀ x, (y = (x^3 + x) / (x^2 + 1)))) :=
by sorry

end functions_same_l13_13827


namespace find_f_10_l13_13255

variable {f : ℤ → ℤ}

-- Defining the conditions
axiom cond1 : f(1) + 1 > 0
axiom cond2 : ∀ x y : ℤ, f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y
axiom cond3 : ∀ x : ℤ, 2 * f(x) = f(x + 1) - x + 1

-- Goal to prove
theorem find_f_10 : f(10) = 1014 := by
  sorry

end find_f_10_l13_13255


namespace second_number_removed_l13_13809

theorem second_number_removed (S : ℝ) (X : ℝ) (h1 : S / 50 = 38) (h2 : (S - 45 - X) / 48 = 37.5) : X = 55 :=
by
  sorry

end second_number_removed_l13_13809


namespace pens_to_sell_to_make_profit_l13_13956

theorem pens_to_sell_to_make_profit (initial_pens : ℕ) (purchase_price selling_price profit : ℝ) :
  initial_pens = 2000 →
  purchase_price = 0.15 →
  selling_price = 0.30 →
  profit = 150 →
  (initial_pens * selling_price - initial_pens * purchase_price = profit) →
  initial_pens * profit / (selling_price - purchase_price) = 1500 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end pens_to_sell_to_make_profit_l13_13956


namespace jam_event_probability_is_0_25_l13_13281

noncomputable theory

open MeasureTheory

def jam_event_prob : ℝ :=
  let area_of_unit_square := 1
  let area_of_triangle := (1 / 2) * 1 * 1
  area_of_triangle / area_of_unit_square

theorem jam_event_probability_is_0_25:
  jam_event_prob = 0.25 :=
by
  sorry

end jam_event_probability_is_0_25_l13_13281


namespace solve_for_x_l13_13921

theorem solve_for_x : ∃ (x : ℝ), (x - 5) ^ 2 = (1 / 16)⁻¹ ∧ (x = 9 ∨ x = 1) :=
by
  sorry

end solve_for_x_l13_13921


namespace sea_horses_count_l13_13089

theorem sea_horses_count (S P : ℕ) (h1 : 11 * S = 5 * P) (h2 : P = S + 85) : S = 70 :=
by
  sorry

end sea_horses_count_l13_13089


namespace correct_number_is_650_l13_13543

theorem correct_number_is_650 
  (n : ℕ) 
  (h : n - 152 = 346): 
  n + 152 = 650 :=
by
  sorry

end correct_number_is_650_l13_13543


namespace n_mod_5_division_of_grid_l13_13999

theorem n_mod_5_division_of_grid (n : ℕ) :
  (∃ m : ℕ, n^2 = 4 + 5 * m) ↔ n % 5 = 2 :=
by
  sorry

end n_mod_5_division_of_grid_l13_13999


namespace prob_four_ones_in_five_rolls_l13_13612

open ProbabilityTheory

theorem prob_four_ones_in_five_rolls :
  let p_one := (1 : ℝ) / 6;
      p_not_one := 5 / 6;
      single_sequence_prob := p_one^4 * p_not_one;
      total_prob := (5 * single_sequence_prob)
  in
  total_prob = (25 / 7776) := 
by 
  sorry

end prob_four_ones_in_five_rolls_l13_13612


namespace min_value_f_l13_13149

noncomputable def f (x : ℝ) :=
  2 / (x - 1) + 1 / (5 - x)

theorem min_value_f :
  ∃ x ∈ Ioo (1:ℝ) 5, f x = (3 + 2 * Real.sqrt 2) / 4 ∧
    (∀ y ∈ Ioo (1:ℝ) 5, f y ≥ (3 + 2 * Real.sqrt 2) / 4) :=
by
  sorry

end min_value_f_l13_13149


namespace total_handshakes_l13_13858

theorem total_handshakes (twins_num : ℕ) (triplets_num : ℕ) (twins_sets : ℕ) (triplets_sets : ℕ) (h_twins : twins_sets = 9) (h_triplets : triplets_sets = 6) (h_twins_num : twins_num = 2 * twins_sets) (h_triplets_num: triplets_num = 3 * triplets_sets) (h_handshakes : twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2) = 882): 
  (twins_num * (twins_num - 2) + triplets_num * (triplets_num - 3) + 2 * twins_num * (triplets_num / 2)) / 2 = 441 :=
by
  sorry

end total_handshakes_l13_13858


namespace find_x_l13_13228

theorem find_x (x : ℤ) (h : (2008 + x)^2 = x^2) : x = -1004 :=
sorry

end find_x_l13_13228


namespace probability_of_choosing_gulongzhong_l13_13158

def num_attractions : Nat := 4
def num_ways_gulongzhong : Nat := 1
def probability_gulongzhong : ℚ := num_ways_gulongzhong / num_attractions

theorem probability_of_choosing_gulongzhong : probability_gulongzhong = 1 / 4 := 
by 
  sorry

end probability_of_choosing_gulongzhong_l13_13158


namespace series_value_l13_13121

theorem series_value : ∑ n in Nat.range ∞, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
by
  sorry

end series_value_l13_13121


namespace tan_product_l13_13561

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end tan_product_l13_13561


namespace find_natural_number_n_l13_13573

theorem find_natural_number_n (n x y : ℕ) (h1 : n + 195 = x^3) (h2 : n - 274 = y^3) : 
  n = 2002 :=
by
  sorry

end find_natural_number_n_l13_13573


namespace arcsin_neg_one_eq_neg_half_pi_l13_13558

theorem arcsin_neg_one_eq_neg_half_pi :
  arcsin (-1) = - (Float.pi / 2) :=
by
  sorry

end arcsin_neg_one_eq_neg_half_pi_l13_13558


namespace purely_imaginary_necessary_not_sufficient_l13_13832

-- Definition of a purely imaginary number
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem purely_imaginary_necessary_not_sufficient (a b : ℝ) :
  a = 0 → (z : ℂ) = ⟨a, b⟩ → is_purely_imaginary z ↔ (a = 0 ∧ b ≠ 0) :=
by
  sorry

end purely_imaginary_necessary_not_sufficient_l13_13832


namespace rectangle_y_value_l13_13053

theorem rectangle_y_value 
  (y : ℝ)
  (A : (0, 0) = E ∧ (0, 5) = F ∧ (y, 5) = G ∧ (y, 0) = H)
  (area : 5 * y = 35)
  (y_pos : y > 0) :
  y = 7 :=
sorry

end rectangle_y_value_l13_13053


namespace min_value_when_a_is_negative_one_max_value_bounds_l13_13596

-- Conditions
def f (a x : ℝ) : ℝ := a * x^2 + x
def a1 : ℝ := -1
def a : ℝ := -2
def a_lower_bound : ℝ := -2
def a_upper_bound : ℝ := 0
def interval : Set ℝ := Set.Icc 0 2

-- Part I: Minimum value when a = -1
theorem min_value_when_a_is_negative_one : 
  ∃ x ∈ interval, f a1 x = -2 := 
by
  sorry

-- Part II: Maximum value criterions
theorem max_value_bounds (a : ℝ) (H : a ∈ Set.Icc a_lower_bound a_upper_bound) :
  (∀ x ∈ interval, 
    (a ≥ -1/4 → f a ( -1 / (2 * a) ) = -1 / (4 * a)) 
    ∧ (a < -1/4 → f a 2 = 4 * a + 2 )) :=
by
  sorry

end min_value_when_a_is_negative_one_max_value_bounds_l13_13596


namespace radius_of_circle_in_xy_plane_l13_13930

theorem radius_of_circle_in_xy_plane (theta : ℝ) :
  let x := 2 * Real.sin (Real.pi / 3) * Real.cos theta,
      y := 2 * Real.sin (Real.pi / 3) * Real.sin theta,
      radius := Real.sqrt (x^2 + y^2)
  in radius = Real.sqrt 3 :=
by
  have x_def : x = 2 * Real.sin (Real.pi / 3) * Real.cos theta := rfl
  have y_def : y = 2 * Real.sin (Real.pi / 3) * Real.sin theta := rfl
  have radius_def : radius = Real.sqrt (x^2 + y^2) := rfl
  sorry

end radius_of_circle_in_xy_plane_l13_13930


namespace tan_product_l13_13568

theorem tan_product (t : ℝ) (h1 : 1 - 7 * t^2 + 7 * t^4 - t^6 = 0)
  (h2 : t = tan (Real.pi / 8) ∨ t = tan (3 * Real.pi / 8) ∨ t = tan (5 * Real.pi / 8)) :
  tan (Real.pi / 8) * tan (3 * Real.pi / 8) * tan (5 * Real.pi / 8) = 1 := 
sorry

end tan_product_l13_13568


namespace factorable_iff_m_eq_2_l13_13455

theorem factorable_iff_m_eq_2 (m : ℤ) :
  (∃ (A B C D : ℤ), (x y : ℤ) -> (x^2 + 2*x*y + 2*x + m*y + 2*m = (x + A*y + B) * (x + C*y + D))) ↔ m = 2 :=
sorry

end factorable_iff_m_eq_2_l13_13455


namespace eval_expression_correct_l13_13867

noncomputable def evaluate_expression : ℝ :=
    3 + Real.sqrt 3 + (3 - Real.sqrt 3) / 6 + (1 / (Real.cos (Real.pi / 4) - 3))

theorem eval_expression_correct : 
  evaluate_expression = (3 * Real.sqrt 3 - 5 * Real.sqrt 2) / 34 :=
by
  -- Proof can be filled in later
  sorry

end eval_expression_correct_l13_13867


namespace paint_mixture_replacement_l13_13061

theorem paint_mixture_replacement :
  ∃ x y : ℝ,
    (0.5 * (1 - x) + 0.35 * x = 0.45) ∧
    (0.6 * (1 - y) + 0.45 * y = 0.55) ∧
    (x = 1 / 3) ∧
    (y = 1 / 3) :=
sorry

end paint_mixture_replacement_l13_13061


namespace octal_to_decimal_conversion_l13_13419

theorem octal_to_decimal_conversion : 
  let d8 := 8
  let f := fun (x: Nat) (y: Nat) => x * d8 ^ y
  7 * d8^0 + 6 * d8^1 + 3 * d8^2 = 247 := 
by
  let d8 := 8
  let f := fun (x: Nat) (y: Nat) => x * d8 ^ y
  sorry

end octal_to_decimal_conversion_l13_13419


namespace lesson_duration_tuesday_l13_13969

theorem lesson_duration_tuesday
  (monday_lessons : ℕ)
  (monday_duration : ℕ)
  (tuesday_lessons : ℕ)
  (wednesday_multiplier : ℕ)
  (total_time : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (H1 : monday_lessons = 6)
  (H2 : monday_duration = 30)
  (H3 : tuesday_lessons = 3)
  (H4 : wednesday_multiplier = 2)
  (H5 : total_time = 12)
  (H6 : monday_hours = monday_lessons * monday_duration / 60)
  (H7 : tuesday_hours = tuesday_lessons * T)
  (H8 : wednesday_hours = wednesday_multiplier * tuesday_hours)
  (H9 : monday_hours + tuesday_hours + wednesday_hours = total_time) :
  T = 1 := by
  sorry

end lesson_duration_tuesday_l13_13969


namespace arithmetic_geometric_mean_inequality_l13_13519

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end arithmetic_geometric_mean_inequality_l13_13519


namespace entrance_sum_2_to_3_pm_exit_sum_2_to_3_pm_no_crowd_control_at_4_pm_l13_13277

noncomputable def f : ℕ → ℕ
| n => if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
       else if 9 ≤ n ∧ n ≤ 32 then 360 * (3 ^ ((n - 8) / 12)) + 3000
       else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
       else 0

noncomputable def g : ℕ → ℕ
| n => if 1 ≤ n ∧ n ≤ 18 then 0
       else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
       else if 33 ≤ n ∧ n ≤ 45 then 8800
       else 0

theorem entrance_sum_2_to_3_pm : f 21 + f 22 + f 23 + f 24 = 17460 := by
  sorry

theorem exit_sum_2_to_3_pm : g 21 + g 22 + g 23 + g 24 = 9000 := by
  sorry

theorem no_crowd_control_at_4_pm : f 28 - g 28 < 80000 := by
  sorry

end entrance_sum_2_to_3_pm_exit_sum_2_to_3_pm_no_crowd_control_at_4_pm_l13_13277


namespace cosine_value_parallel_vectors_l13_13482

theorem cosine_value_parallel_vectors (α : ℝ) (h1 : ∃ (a : ℝ × ℝ) (b : ℝ × ℝ), a = (Real.cos (Real.pi / 3 + α), 1) ∧ b = (1, 4) ∧ a.1 * b.2 - a.2 * b.1 = 0) : 
  Real.cos (Real.pi / 3 - 2 * α) = 7 / 8 := by
  sorry

end cosine_value_parallel_vectors_l13_13482


namespace ninas_money_l13_13357

theorem ninas_money (C M : ℝ) (h1 : 6 * C = M) (h2 : 8 * (C - 1.15) = M) : M = 27.6 := 
by
  sorry

end ninas_money_l13_13357


namespace func_max_l13_13941

-- Condition: The function defined as y = 2 * cos x - 1
def func (x : ℝ) : ℝ := 2 * Real.cos x - 1

-- Proof statement: Prove that the function achieves its maximum value when x = 2 * k * π for k ∈ ℤ
theorem func_max (x : ℝ) (k : ℤ) : (∃ x, func x = 2 * 1 - 1) ↔ (∃ k : ℤ, x = 2 * k * Real.pi) := sorry

end func_max_l13_13941


namespace third_median_length_is_9_l13_13015

noncomputable def length_of_third_median_of_triangle (m₁ m₂ m₃ area : ℝ) : Prop :=
  ∃ median : ℝ, median = m₃

theorem third_median_length_is_9 :
  length_of_third_median_of_triangle 5 7 9 (6 * Real.sqrt 10) :=
by
  sorry

end third_median_length_is_9_l13_13015


namespace solve_equation_in_integers_l13_13922
-- Import the necessary library for Lean

-- Define the main theorem to solve the equation in integers
theorem solve_equation_in_integers :
  ∃ (xs : List (ℕ × ℕ)), (∀ x y, (3^x - 2^y = 1 → (x, y) ∈ xs)) ∧ xs = [(1, 1), (2, 3)] :=
by
  sorry

end solve_equation_in_integers_l13_13922


namespace sufficient_but_not_necessary_condition_l13_13244

theorem sufficient_but_not_necessary_condition :
  (∀ (x : ℝ), x^2 - 2 * x < 0 → 0 < x ∧ x < 4)
  ∧ ¬(∀ (x : ℝ), 0 < x ∧ x < 4 → x^2 - 2 * x < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l13_13244


namespace rotated_point_l13_13156

def point := (ℝ × ℝ × ℝ)

def rotate_point (A P : point) (θ : ℝ) : point :=
  -- Function implementing the rotation (the full definition would normally be placed here)
  sorry

def A : point := (1, 1, 1)
def P : point := (1, 1, 0)

theorem rotated_point (θ : ℝ) (hθ : θ = 60) :
  rotate_point A P θ = (1/3, 4/3, 1/3) :=
sorry

end rotated_point_l13_13156


namespace clock_angle_3_40_l13_13077

/-- The smaller angle between the hands of a 12-hour clock at 3:40 pm in degrees is 130.0. -/
theorem clock_angle_3_40 : 
  let minute_angle := 40 * 6,
      hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
      angle_between := abs (minute_angle - hour_angle) in
  real.to_decimal angle_between 1 = "130.0" := 
by {
  let minute_angle := 40 * 6,
  let hour_angle := 3 * 60 * 0.5 + 40 * 0.5,
  let angle_between := abs (minute_angle - hour_angle),
  sorry
}

end clock_angle_3_40_l13_13077


namespace equivalent_polar_coordinates_l13_13337

-- Definitions of given conditions and the problem statement
def polar_point_neg (r : ℝ) (θ : ℝ) : Prop := r = -3 ∧ θ = 5 * Real.pi / 6
def polar_point_pos (r : ℝ) (θ : ℝ) : Prop := r = 3 ∧ θ = 11 * Real.pi / 6
def angle_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

theorem equivalent_polar_coordinates :
  ∃ (r θ : ℝ), polar_point_neg r θ → polar_point_pos 3 (11 * Real.pi / 6) ∧ angle_range (11 * Real.pi / 6) :=
by
  sorry

end equivalent_polar_coordinates_l13_13337


namespace smallest_n_for_constant_term_l13_13469

theorem smallest_n_for_constant_term :
  ∃ (n : ℕ), (n > 0) ∧ ((∃ (r : ℕ), 2 * n = 5 * r) ∧ (∀ (m : ℕ), m > 0 → (∃ (r' : ℕ), 2 * m = 5 * r') → n ≤ m)) ∧ n = 5 :=
by
  sorry

end smallest_n_for_constant_term_l13_13469


namespace expression_divisible_by_10_l13_13915

theorem expression_divisible_by_10 (n : ℕ) : 10 ∣ (3 ^ (n + 2) - 2 ^ (n + 2) + 3 ^ n - 2 ^ n) :=
  sorry

end expression_divisible_by_10_l13_13915


namespace meet_at_starting_line_l13_13895

theorem meet_at_starting_line (henry_time margo_time : ℕ) (h_henry : henry_time = 7) (h_margo : margo_time = 12) : Nat.lcm henry_time margo_time = 84 :=
by
  rw [h_henry, h_margo]
  sorry

end meet_at_starting_line_l13_13895


namespace max_value_of_trig_expr_l13_13297

theorem max_value_of_trig_expr (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 :=
sorry

end max_value_of_trig_expr_l13_13297


namespace problem_l13_13051

theorem problem (q r : ℕ) (hq : 1259 = 23 * q + r) (hq_pos : 0 < q) (hr_pos : 0 < r) :
  q - r ≤ 37 :=
sorry

end problem_l13_13051


namespace B_grazed_months_l13_13702

-- Define the conditions
variables (A_cows B_cows C_cows D_cows : ℕ)
variables (A_months B_months C_months D_months : ℕ)
variables (A_rent total_rent : ℕ)

-- Given conditions
def A_condition := (A_cows = 24 ∧ A_months = 3)
def B_condition := (B_cows = 10)
def C_condition := (C_cows = 35 ∧ C_months = 4)
def D_condition := (D_cows = 21 ∧ D_months = 3)
def A_rent_condition := (A_rent = 720)
def total_rent_condition := (total_rent = 3250)

-- Define cow-months calculation
def cow_months (cows months : ℕ) : ℕ := cows * months

-- Define cost per cow-month
def cost_per_cow_month (rent cow_months : ℕ) : ℕ := rent / cow_months

-- Define B's months of grazing proof problem
theorem B_grazed_months
  (A_cows_months : cow_months 24 3 = 72)
  (B_cows := 10)
  (C_cows_months : cow_months 35 4 = 140)
  (D_cows_months : cow_months 21 3 = 63)
  (A_rent_condition : A_rent = 720)
  (total_rent_condition : total_rent = 3250) :
  ∃ (B_months : ℕ), 10 * B_months = 50 ∧ B_months = 5 := sorry

end B_grazed_months_l13_13702


namespace circle_area_with_radius_8_l13_13249

noncomputable def circle_radius : ℝ := 8
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_with_radius_8 :
  circle_area circle_radius = 64 * Real.pi :=
by
  sorry

end circle_area_with_radius_8_l13_13249


namespace correct_inequality_l13_13970

theorem correct_inequality :
  (1 / 2)^(2 / 3) < (1 / 2)^(1 / 3) ∧ (1 / 2)^(1 / 3) < 1 :=
by sorry

end correct_inequality_l13_13970


namespace tan_five_pi_over_four_l13_13735

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_five_pi_over_four_l13_13735


namespace scientific_notation_of_GDP_l13_13715

theorem scientific_notation_of_GDP
  (b : ℕ) (billion_val : b = 10^9) :
  ∀ (n : ℕ) (GDP_billion : n = 53100), 
  let GDP_scientific := (5.31 : ℝ) * 10^13 in
  (n * b : ℝ) = GDP_scientific := 
by
  intros
  unfold billion_val GDP_billion
  unfold GDP_scientific
  sorry

end scientific_notation_of_GDP_l13_13715


namespace max_souls_guaranteed_l13_13117

def initial_nuts : ℕ := 1001

def valid_N (N : ℕ) : Prop :=
  1 ≤ N ∧ N ≤ 1001

def nuts_transferred (N : ℕ) (T : ℕ) : Prop :=
  valid_N N ∧ T ≤ 71

theorem max_souls_guaranteed : (∀ N, valid_N N → ∃ T, nuts_transferred N T) :=
sorry

end max_souls_guaranteed_l13_13117


namespace remainder_of_3_pow_19_div_10_l13_13547

def w : ℕ := 3 ^ 19

theorem remainder_of_3_pow_19_div_10 : w % 10 = 7 := by
  sorry

end remainder_of_3_pow_19_div_10_l13_13547


namespace exists_sum_or_diff_divisible_by_1000_l13_13362

theorem exists_sum_or_diff_divisible_by_1000 (nums : Fin 502 → Nat) :
  ∃ a b : Nat, (∃ i j : Fin 502, nums i = a ∧ nums j = b ∧ i ≠ j) ∧
  (a - b) % 1000 = 0 ∨ (a + b) % 1000 = 0 :=
by
  sorry

end exists_sum_or_diff_divisible_by_1000_l13_13362


namespace ratio_of_bubbles_l13_13327

def bubbles_dawn_per_ounce : ℕ := 200000

def mixture_bubbles (bubbles_other_per_ounce : ℕ) : ℕ :=
  let half_ounce_dawn := bubbles_dawn_per_ounce / 2
  let half_ounce_other := bubbles_other_per_ounce / 2
  half_ounce_dawn + half_ounce_other

noncomputable def find_ratio (bubbles_other_per_ounce : ℕ) : ℚ :=
  (bubbles_other_per_ounce : ℚ) / bubbles_dawn_per_ounce

theorem ratio_of_bubbles
  (bubbles_other_per_ounce : ℕ)
  (h_mixture : mixture_bubbles bubbles_other_per_ounce = 150000) :
  find_ratio bubbles_other_per_ounce = 1 / 2 :=
by
  sorry

end ratio_of_bubbles_l13_13327


namespace sum_div_minuend_eq_two_l13_13684

variable (Subtrahend Minuend Difference : ℝ)

theorem sum_div_minuend_eq_two
  (h : Subtrahend + Difference = Minuend) :
  (Subtrahend + Minuend + Difference) / Minuend = 2 :=
by
  sorry

end sum_div_minuend_eq_two_l13_13684


namespace point_D_number_l13_13796

theorem point_D_number (x : ℝ) :
    (5 + 8 - 10 + x = -5 - 8 + 10 - x) ↔ x = -3 :=
by
  sorry

end point_D_number_l13_13796


namespace area_of_triangle_ABF_l13_13817

open Real

/-- Let A = (0, 0), B = (√3, 0), E an interior point of square ABCD such that ∠ABE = 90°, 
    and F the intersection of BD and AE.
    Given the length of side AB is √3, we aim to find the area of ΔABF. -/
theorem area_of_triangle_ABF : 
  ∃ (A B E F : ℝ × ℝ), 
  A = (0, 0) ∧ 
  B = (sqrt 3, 0) ∧
  (∃ (C D : ℝ × ℝ), (C = (sqrt 3, sqrt 3) ∧ D = (0, sqrt 3))) ∧
  E.1 = E.2 ∧ 
  (E.1 > 0 ∧ E.1 < sqrt 3) ∧
  (E.1, E.2) ∈ interior (convex_hull (set.ABC A B (0, sqrt 3))) ∧
  (∃ F : ℝ × ℝ, F = (sqrt 3 / 2, sqrt 3 / 2)) ∧
  ∃ (area : ℝ), area = 3 / 4 :=
sorry

end area_of_triangle_ABF_l13_13817


namespace apples_remaining_l13_13364

-- Define the initial conditions
def number_of_trees := 52
def apples_on_tree_before := 9
def apples_picked := 2

-- Define the target proof: the number of apples remaining on the tree
def apples_on_tree_after := apples_on_tree_before - apples_picked

-- The statement we aim to prove
theorem apples_remaining : apples_on_tree_after = 7 := sorry

end apples_remaining_l13_13364


namespace exact_sequence_a2007_l13_13104

theorem exact_sequence_a2007 (a : ℕ → ℤ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 0) 
  (exact : ∀ n m : ℕ, n > m → a n ^ 2 - a m ^ 2 = a (n - m) * a (n + m)) :
  a 2007 = -1 := 
sorry

end exact_sequence_a2007_l13_13104


namespace tan_22_5_eq_half_l13_13213

noncomputable def tan_h_LHS (θ : Real) := Real.tan θ / (1 - Real.tan θ ^ 2)

theorem tan_22_5_eq_half :
    tan_h_LHS (Real.pi / 8) = 1 / 2 :=
  sorry

end tan_22_5_eq_half_l13_13213


namespace shortest_distance_from_parabola_to_line_l13_13475

open Real

noncomputable def parabola_point (M : ℝ × ℝ) : Prop :=
  M.snd^2 = 6 * M.fst

noncomputable def distance_to_line (M : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * M.fst + b * M.snd + c) / sqrt (a^2 + b^2)

theorem shortest_distance_from_parabola_to_line (M : ℝ × ℝ) (h : parabola_point M) :
  distance_to_line M 3 (-4) 12 = 3 :=
by
  sorry

end shortest_distance_from_parabola_to_line_l13_13475


namespace diameter_circle_C_inscribed_within_D_l13_13725

noncomputable def circle_diameter_C (d_D : ℝ) (ratio : ℝ) : ℝ :=
  let R := d_D / 2
  let r := (R : ℝ) / (Real.sqrt 5)
  2 * r

theorem diameter_circle_C_inscribed_within_D 
  (d_D : ℝ) (ratio : ℝ) (h_dD_pos : 0 < d_D) (h_ratio : ratio = 4)
  (h_dD : d_D = 24) : 
  circle_diameter_C d_D ratio = 24 * Real.sqrt 5 / 5 :=
by
  sorry

end diameter_circle_C_inscribed_within_D_l13_13725


namespace wire_cut_l13_13835

theorem wire_cut (total_length : ℝ) (ratio : ℝ) (shorter longer : ℝ) (h_total : total_length = 21) (h_ratio : ratio = 2/5)
  (h_shorter : longer = (5/2) * shorter) (h_sum : total_length = shorter + longer) : shorter = 6 := 
by
  -- total_length = 21, ratio = 2/5, longer = (5/2) * shorter, total_length = shorter + longer, prove shorter = 6
  sorry

end wire_cut_l13_13835


namespace polynomial_equivalence_l13_13403

def polynomial_expression (x : ℝ) : ℝ :=
  (3 * x ^ 2 + 2 * x - 5) * (x - 2) - (x - 2) * (x ^ 2 - 5 * x + 28) + (4 * x - 7) * (x - 2) * (x + 4)

theorem polynomial_equivalence (x : ℝ) : 
  polynomial_expression x = 6 * x ^ 3 + 4 * x ^ 2 - 93 * x + 122 :=
by {
  sorry
}

end polynomial_equivalence_l13_13403


namespace total_hours_eq_52_l13_13978

def hours_per_week_on_extracurriculars : ℕ := 2 + 8 + 3  -- Total hours per week
def weeks_in_semester : ℕ := 12  -- Total weeks in a semester
def weeks_before_midterms : ℕ := weeks_in_semester / 2  -- Weeks before midterms
def sick_weeks : ℕ := 2  -- Weeks Annie takes off sick
def active_weeks_before_midterms : ℕ := weeks_before_midterms - sick_weeks  -- Active weeks before midterms

def total_extracurricular_hours_before_midterms : ℕ :=
  hours_per_week_on_extracurriculars * active_weeks_before_midterms

theorem total_hours_eq_52 :
  total_extracurricular_hours_before_midterms = 52 :=
by
  sorry

end total_hours_eq_52_l13_13978


namespace range_of_a_l13_13766

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| - 2 * a + 2 < 0) → (a > 2) :=
by
  sorry

end range_of_a_l13_13766


namespace probability_after_2019_rings_l13_13902

noncomputable def players_start_with_one : ℕ := 1
noncomputable def bell_ring_interval : ℕ := 15
noncomputable def num_rings : ℕ := 2019

-- Assuming a function that simulates the game results based on the conditions
def game_simulation (num_rings : ℕ) : ℚ := sorry

theorem probability_after_2019_rings :
  game_simulation num_rings = 1 / 4 := sorry

end probability_after_2019_rings_l13_13902


namespace smallest_positive_integer_n_l13_13953

theorem smallest_positive_integer_n (n : ℕ) :
  (∃ n1 n2 n3 : ℕ, 5 * n = n1 ^ 5 ∧ 6 * n = n2 ^ 6 ∧ 7 * n = n3 ^ 7) →
  n = 2^5 * 3^5 * 5^4 * 7^6 :=
by
  sorry

end smallest_positive_integer_n_l13_13953


namespace values_of_m_l13_13580

def A : Set ℝ := { -1, 2 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

theorem values_of_m (m : ℝ) : (A ∪ B m = A) ↔ (m = -1/2 ∨ m = 0 ∨ m = 1) := by
  sorry

end values_of_m_l13_13580


namespace find_g_zero_l13_13384

variable {g : ℝ → ℝ}

theorem find_g_zero (h : ∀ x y : ℝ, g (x + y) = g x + g y - 1) : g 0 = 1 :=
sorry

end find_g_zero_l13_13384


namespace probability_of_at_least_ten_heads_in_twelve_given_first_two_heads_l13_13395

-- Define a fair coin
inductive Coin
| Heads
| Tails

def fair_coin : List Coin := [Coin.Heads, Coin.Tails]

-- Define a function to calculate the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.descFactorial n k / k.factorial

-- Define a function to calculate the probability of at least 8 heads in 10 flips
def prob_at_least_eight_heads_in_ten : ℚ :=
  (binomial 10 8 + binomial 10 9 + binomial 10 10) / (2 ^ 10)

-- Define our theorem statement
theorem probability_of_at_least_ten_heads_in_twelve_given_first_two_heads :
    (prob_at_least_eight_heads_in_ten = 7 / 128) :=
  by
    -- The proof steps can be written here later
    sorry

end probability_of_at_least_ten_heads_in_twelve_given_first_two_heads_l13_13395


namespace solve_wire_cut_problem_l13_13948

def wire_cut_problem : Prop :=
  ∃ x y : ℝ, x + y = 35 ∧ y = (2/5) * x ∧ x = 25

theorem solve_wire_cut_problem : wire_cut_problem := by
  sorry

end solve_wire_cut_problem_l13_13948


namespace solution_count_l13_13594

/-- There are 91 solutions to the equation x + y + z = 15 given that x, y, z are all positive integers. -/
theorem solution_count (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 15) : 
  ∃! n, n = 91 := 
by sorry

end solution_count_l13_13594


namespace average_mb_per_hour_of_music_l13_13550

/--
Given a digital music library:
- It contains 14 days of music.
- The first 7 days use 10,000 megabytes of disk space.
- The next 7 days use 14,000 megabytes of disk space.
- Each day has 24 hours.

Prove that the average megabytes per hour of music in this library is 71 megabytes.
-/
theorem average_mb_per_hour_of_music
  (days_total : ℕ) 
  (days_first : ℕ) 
  (days_second : ℕ) 
  (mb_first : ℕ) 
  (mb_second : ℕ) 
  (hours_per_day : ℕ) 
  (total_mb : ℕ) 
  (total_hours : ℕ) :
  days_total = 14 →
  days_first = 7 →
  days_second = 7 →
  mb_first = 10000 →
  mb_second = 14000 →
  hours_per_day = 24 →
  total_mb = mb_first + mb_second →
  total_hours = days_total * hours_per_day →
  total_mb / total_hours = 71 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end average_mb_per_hour_of_music_l13_13550


namespace belfried_industries_tax_l13_13847

noncomputable def payroll_tax (payroll : ℕ) : ℕ :=
  if payroll <= 200000 then
    0
  else
    ((payroll - 200000) * 2) / 1000

theorem belfried_industries_tax : payroll_tax 300000 = 200 :=
by
  sorry

end belfried_industries_tax_l13_13847


namespace central_angle_of_sector_l13_13656

noncomputable def sector_area (α r : ℝ) : ℝ := (1/2) * α * r^2

theorem central_angle_of_sector :
  sector_area 3 2 = 6 :=
by
  unfold sector_area
  norm_num
  done

end central_angle_of_sector_l13_13656


namespace smaller_angle_at_3_40_l13_13074

-- Defining the context of a 12-hour clock
def degrees_per_minute : ℝ := 360 / 60
def degrees_per_hour : ℝ := 360 / 12

-- Defining the problem conditions:
def minute_position (minutes : ℝ) : ℝ :=
  minutes * degrees_per_minute

def hour_position (hour : ℝ) (minutes : ℝ) : ℝ :=
  (hour * 30) + (minutes * (degrees_per_hour / 60))

-- The specific condition given in the problem:
def minute_hand_3_40 := minute_position 40
def hour_hand_3_40 := hour_position 3 40

def angle_between_hands (minute_hand : ℝ) (hour_hand : ℝ) : ℝ :=
  abs (minute_hand - hour_hand)

def smaller_angle (angle : ℝ) : ℝ :=
  if angle > 180 then 360 - angle else angle

-- The theorem to prove:
theorem smaller_angle_at_3_40 : smaller_angle (angle_between_hands minute_hand_3_40 hour_hand_3_40) = 130 := by
  sorry

end smaller_angle_at_3_40_l13_13074


namespace decreasing_functions_l13_13595

noncomputable def f1 (x : ℝ) : ℝ := -x^2 + 1
noncomputable def f2 (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def f4 (x : ℝ) : ℝ := 3 ^ x

theorem decreasing_functions :
  (∀ x y : ℝ, 0 < x → x < y → f1 y < f1 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f2 y > f2 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f3 y > f3 x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f4 y > f4 x) :=
by {
  sorry
}

end decreasing_functions_l13_13595


namespace radius_of_shorter_tank_l13_13689

theorem radius_of_shorter_tank (h : ℝ) (r : ℝ) 
  (volume_eq : ∀ (π : ℝ), π * (10^2) * (2 * h) = π * (r^2) * h) : 
  r = 10 * Real.sqrt 2 := 
by 
  sorry

end radius_of_shorter_tank_l13_13689


namespace find_f_value_l13_13268

def f (x : ℤ) : ℤ := sorry

theorem find_f_value :
  (f(1) + 1 > 0) ∧ 
  (∀ (x y : ℤ), f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y) ∧
  (∀ (x : ℤ), 2 * f(x) = f(x + 1) - x + 1) →
  f 10 = 1014 :=
by
  sorry

end find_f_value_l13_13268


namespace find_d_l13_13473

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) :
  d = 180 :=
sorry

end find_d_l13_13473


namespace can_capacity_is_30_l13_13332

noncomputable def capacity_of_can (x: ℝ) : ℝ :=
  7 * x + 10

theorem can_capacity_is_30 :
  ∃ (x: ℝ), (4 * x + 10) / (3 * x) = 5 / 2 ∧ capacity_of_can x = 30 :=
by
  sorry

end can_capacity_is_30_l13_13332


namespace largest_even_digit_multiple_of_9_under_1000_l13_13820

noncomputable def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (digits n).sum

noncomputable def all_even_digits (n : ℕ) : Prop :=
  ∀ d ∈ digits n, is_even_digit d

theorem largest_even_digit_multiple_of_9_under_1000 :
  ∃ n, n < 1000 ∧ all_even_digits n ∧ sum_of_digits n % 9 = 0 ∧ n = 360 :=
begin
  use 360,
  split,
  { exact nat.lt_succ_self 359 },
  split,
  { intros d hd,
    unfold digits at hd,
    rw list.mem_iff_exists_get at hd,
    rcases hd with ⟨k, hk⟩,
    repeat { rw list.get? },
    simp only [digits] at hk,
    have : k < 3 := (nat.lt_of_succ_lt_succ (list.length_le_of_lt_some hk)).trans_le (by norm_num),
    interval_cases k,
    { rw [hk, nat.digits, nat.digits_aux'],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] },
    { rw [hk, nat.digits, nat.digits_aux', nat.div_eq, nat.mod_eq_of_lt, nat.add_eq_zero_eq_zero_and_eq_zero, nat.zero_eq, eq_self_iff_true, true_and],
      norm_num [is_even_digit] } },
  { simp only [sum_of_digits, digits],
    exact nat.digits_sum_eq 360 10,
    exact dec_trivial },
  { refl }
end

end largest_even_digit_multiple_of_9_under_1000_l13_13820


namespace water_left_over_l13_13934

theorem water_left_over (players : ℕ) (initial_liters : ℕ) (milliliters_per_player : ℕ) (water_spill_ml : ℕ) :
  players = 30 → initial_liters = 8 → milliliters_per_player = 200 → water_spill_ml = 250 →
  (initial_liters * 1000) - (players * milliliters_per_player + water_spill_ml) = 1750 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  change 8 * 1000 - (30 * 200 + 250) = 1750
  norm_num
  sorry

end water_left_over_l13_13934


namespace f_10_l13_13259

namespace MathProof

variable (f : ℤ → ℤ)

-- Condition 1: f(1) + 1 > 0
axiom cond1 : f 1 + 1 > 0

-- Condition 2: f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y for any x, y ∈ ℤ
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y

-- Condition 3: 2 * f(x) = f(x + 1) - x + 1 for any x ∈ ℤ
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

-- We need to prove f(10) = 1014
theorem f_10 : f 10 = 1014 :=
by
  sorry

end MathProof

end f_10_l13_13259


namespace find_A_l13_13653

theorem find_A :
  ∃ A B C : ℝ, 
  (1 : ℝ) / (x^3 - 7 * x^2 + 11 * x + 15) = 
  A / (x - 5) + B / (x + 3) + C / ((x + 3)^2) → 
  A = 1 / 64 := 
by 
  sorry

end find_A_l13_13653


namespace hyperbola_triangle_area_l13_13363

/-- The relationship between the hyperbola's asymptotes, tangent, and area proportion -/
theorem hyperbola_triangle_area (a b x0 y0 : ℝ) 
  (h_asymptote1 : ∀ x, y = (b / a) * x)
  (h_asymptote2 : ∀ x, y = -(b / a) * x)
  (h_tangent    : ∀ x y, (x0 * x) / (a ^ 2) - (y0 * y) / (b ^ 2) = 1)
  (h_condition  : (x0 ^ 2) * (a ^ 2) - (y0 ^ 2) * (b ^ 2) = (a ^ 2) * (b ^ 2)) :
  ∃ k : ℝ, k = a ^ 4 :=
sorry

end hyperbola_triangle_area_l13_13363


namespace james_huskies_count_l13_13906

theorem james_huskies_count 
  (H : ℕ) 
  (pitbulls : ℕ := 2) 
  (golden_retrievers : ℕ := 4) 
  (husky_pups_per_husky : ℕ := 3) 
  (pitbull_pups_per_pitbull : ℕ := 3) 
  (extra_pups_per_golden_retriever : ℕ := 2) 
  (pup_difference : ℕ := 30) :
  H + pitbulls + golden_retrievers + pup_difference = 3 * H + pitbulls * pitbull_pups_per_pitbull + golden_retrievers * (husky_pups_per_husky + extra_pups_per_golden_retriever) :=
sorry

end james_huskies_count_l13_13906


namespace negation_of_existence_l13_13759

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem negation_of_existence:
  (∃ x : ℝ, log_base 3 x ≤ 0) ↔ ∀ x : ℝ, log_base 3 x < 0 :=
by
  sorry

end negation_of_existence_l13_13759


namespace find_f_10_l13_13270

def f (x : ℤ) : ℤ := sorry

noncomputable def h (x : ℤ) : ℤ := f(x) + x

axiom condition_1 : f(1) + 1 > 0

axiom condition_2 : ∀ (x y : ℤ), f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y

axiom condition_3 : ∀ (x : ℤ), 2 * f(x) = f(x + 1) - x + 1

theorem find_f_10 : f(10) = 1014 := sorry

end find_f_10_l13_13270


namespace abs_sum_condition_l13_13747

theorem abs_sum_condition (a b : ℝ) (h₁ : |a| = 2) (h₂ : b = -1) : |a + b| = 1 ∨ |a + b| = 3 :=
by
  sorry

end abs_sum_condition_l13_13747


namespace no_x_axis_intersection_iff_l13_13167

theorem no_x_axis_intersection_iff (m : ℝ) :
    (∀ x : ℝ, x^2 - x + m ≠ 0) ↔ m > 1 / 4 :=
by
  sorry

end no_x_axis_intersection_iff_l13_13167


namespace distinct_sequences_count_l13_13160

-- Defining the set of letters in "PROBLEMS"
def letters : List Char := ['P', 'R', 'O', 'B', 'L', 'E', 'M']

-- Defining a sequence constraint: must start with 'S' and not end with 'M'
def valid_sequence (seq : List Char) : Prop :=
  seq.head? = some 'S' ∧ seq.getLast? ≠ some 'M'

-- Counting valid sequences according to the constraints
noncomputable def count_valid_sequences : Nat :=
  6 * 120

theorem distinct_sequences_count :
  count_valid_sequences = 720 := by
  sorry

end distinct_sequences_count_l13_13160


namespace units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l13_13994

-- Define the cycle of the units digits of powers of 7
def units_digit_of_7_power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1  -- 7^4, 7^8, ...
  | 1 => 7  -- 7^1, 7^5, ...
  | 2 => 9  -- 7^2, 7^6, ...
  | 3 => 3  -- 7^3, 7^7, ...
  | _ => 0  -- unreachable

-- The main theorem to prove
theorem units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared : 
  units_digit_of_7_power (3 ^ (5 ^ 2)) = 3 :=
by
  sorry

end units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l13_13994


namespace angle_at_3_40_pm_is_130_degrees_l13_13068

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l13_13068


namespace coal_consumption_rel_l13_13551

variables (Q a x y : ℝ)
variables (h₀ : 0 < x) (h₁ : x < a) (h₂ : Q ≠ 0) (h₃ : a ≠ 0) (h₄ : a - x ≠ 0)

theorem coal_consumption_rel :
  y = Q / (a - x) - Q / a :=
sorry

end coal_consumption_rel_l13_13551


namespace num_candidates_above_630_l13_13663

noncomputable def normal_distribution_candidates : Prop :=
  let μ := 530
  let σ := 50
  let total_candidates := 1000
  let probability_above_630 := (1 - 0.954) / 2  -- Probability of scoring above 630
  let expected_candidates_above_630 := total_candidates * probability_above_630
  expected_candidates_above_630 = 23

theorem num_candidates_above_630 : normal_distribution_candidates := by
  sorry

end num_candidates_above_630_l13_13663


namespace comparison_of_exponential_and_power_l13_13583

theorem comparison_of_exponential_and_power :
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  a > b :=
by
  let a := 2 ^ 0.6
  let b := 0.6 ^ 2
  sorry

end comparison_of_exponential_and_power_l13_13583


namespace probability_queen_then_spade_l13_13220

-- Define the size of the deck and the quantities for specific cards
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_spades : ℕ := 13

-- Define the probability calculation problem
theorem probability_queen_then_spade :
  (num_queens / deck_size : ℚ) * ((num_spades - 1) / (deck_size - 1) : ℚ) + ((num_queens - 1) / deck_size : ℚ) * (num_spades / (deck_size - 1) : ℚ) = 1 / deck_size :=
by sorry

end probability_queen_then_spade_l13_13220


namespace series_sum_eq_half_l13_13131

theorem series_sum_eq_half : ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_eq_half_l13_13131


namespace ellipse_standard_and_trajectory_l13_13147

theorem ellipse_standard_and_trajectory :
  ∀ a b x y : ℝ, 
  a > b ∧ 0 < b ∧ 
  (b^2 = a^2 - 1) ∧ 
  (9/4 + 6/(8) = 1) →
  (∃ x y : ℝ, (x / 2)^2 / 9 + (y)^2 / 8 = 1) ∧ 
  (x^2 / 9 - y^2 / 8 = 1 ∧ x ≠ 3 ∧ x ≠ -3) := 
  sorry

end ellipse_standard_and_trajectory_l13_13147


namespace can_identify_counterfeit_coin_l13_13533

theorem can_identify_counterfeit_coin (coins : Fin 101 → ℤ) :
  (∃ n : Fin 101, (∑ i, if i ≠ n then coins i else 0 = 50 ∧
                   ∑ i, if i = n then coins i else 0 = 1)) →
  (∑ i, coins i = 0 ∨ ∑ i, coins i % 2 = 0) :=
sorry

end can_identify_counterfeit_coin_l13_13533


namespace platform_length_l13_13109

/-- Given:
1. The speed of the train is 72 kmph.
2. The train crosses a platform in 32 seconds.
3. The train crosses a man standing on the platform in 18 seconds.

Prove:
The length of the platform is 280 meters.
-/
theorem platform_length
  (train_speed_kmph : ℕ)
  (cross_platform_time_sec cross_man_time_sec : ℕ)
  (h1 : train_speed_kmph = 72)
  (h2 : cross_platform_time_sec = 32)
  (h3 : cross_man_time_sec = 18) :
  ∃ (L_platform : ℕ), L_platform = 280 :=
by
  sorry

end platform_length_l13_13109


namespace hat_price_reduction_l13_13107

theorem hat_price_reduction (original_price : ℚ) (r1 r2 : ℚ) (price_after_reductions : ℚ) :
  original_price = 12 → r1 = 0.20 → r2 = 0.25 →
  price_after_reductions = original_price * (1 - r1) * (1 - r2) →
  price_after_reductions = 7.20 :=
by
  intros original_price_eq r1_eq r2_eq price_calc_eq
  sorry

end hat_price_reduction_l13_13107


namespace total_fraction_inspected_l13_13351

-- Define the fractions of products inspected by John, Jane, and Roy.
variables (J N R : ℝ)
-- Define the rejection rates for John, Jane, and Roy.
variables (rJ rN rR : ℝ)
-- Define the total rejection rate.
variable (r_total : ℝ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  (rJ = 0.007) ∧ (rN = 0.008) ∧ (rR = 0.01) ∧ (r_total = 0.0085) ∧
  (0.007 * J + 0.008 * N + 0.01 * R = 0.0085)

-- The proof statement that the total fraction of products inspected is 1.
theorem total_fraction_inspected (h : conditions J N R rJ rN rR r_total) : J + N + R = 1 :=
sorry

end total_fraction_inspected_l13_13351


namespace product_of_fractions_l13_13982

-- Define the fractions as ratios.
def fraction1 : ℚ := 2 / 5
def fraction2 : ℚ := 7 / 10

-- State the theorem that proves the product of the fractions is equal to the simplified result.
theorem product_of_fractions : fraction1 * fraction2 = 7 / 25 :=
by
  -- Skip the proof.
  sorry

end product_of_fractions_l13_13982


namespace intersection_of_diagonals_l13_13386

-- Define the four lines based on the given conditions
def line1 (k b x : ℝ) : ℝ := k*x + b
def line2 (k b x : ℝ) : ℝ := k*x - b
def line3 (m b x : ℝ) : ℝ := m*x + b
def line4 (m b x : ℝ) : ℝ := m*x - b

-- Define a function to represent the problem
noncomputable def point_of_intersection_of_diagonals (k m b : ℝ) : ℝ × ℝ :=
(0, 0)

-- State the theorem to be proved
theorem intersection_of_diagonals (k m b : ℝ) :
  point_of_intersection_of_diagonals k m b = (0, 0) :=
sorry

end intersection_of_diagonals_l13_13386


namespace english_score_l13_13853

theorem english_score (s1 s2 s3 e : ℕ) :
  (s1 + s2 + s3) = 276 → (s1 + s2 + s3 + e) = 376 → e = 100 :=
by
  intros h1 h2
  sorry

end english_score_l13_13853


namespace seunghye_saw_number_l13_13781

theorem seunghye_saw_number (x : ℝ) (h : 10 * x - x = 37.35) : x = 4.15 :=
by
  sorry

end seunghye_saw_number_l13_13781


namespace number_of_correct_answers_l13_13710

def total_questions := 30
def correct_points := 3
def incorrect_points := -1
def total_score := 78

theorem number_of_correct_answers (x : ℕ) :
  3 * x + incorrect_points * (total_questions - x) = total_score → x = 27 :=
by
  sorry

end number_of_correct_answers_l13_13710


namespace problem_l13_13478

theorem problem
  (a b c d e : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (- (a * b))^2009 - (c + d)^2010 - e^2011 = 0 := 
by
  sorry

end problem_l13_13478


namespace algebra_expression_value_l13_13753

theorem algebra_expression_value (a b : ℝ)
  (h1 : |a + 2| = 0)
  (h2 : (b - 5 / 2) ^ 2 = 0) : (2 * a + 3 * b) * (2 * b - 3 * a) = 26 := by
sorry

end algebra_expression_value_l13_13753


namespace work_completion_alternate_days_l13_13404

theorem work_completion_alternate_days (h₁ : ∀ (work : ℝ), ∃ a_days : ℝ, a_days = 12 → (∀ t : ℕ, t / a_days <= work / 12))
                                      (h₂ : ∀ (work : ℝ), ∃ b_days : ℝ, b_days = 36 → (∀ t : ℕ, t / b_days <= work / 36)) :
  ∃ days : ℝ, days = 18 := by
  sorry

end work_completion_alternate_days_l13_13404


namespace ring_width_eq_disk_radius_l13_13496

theorem ring_width_eq_disk_radius
  (r R1 R2 : ℝ)
  (h1 : R2 = 3 * r)
  (h2 : 7 * π * r^2 = π * (R1^2 - R2^2)) :
  R1 - R2 = r :=
by {
  sorry
}

end ring_width_eq_disk_radius_l13_13496


namespace find_a_l13_13760

variable {x : ℝ} {a b : ℝ}

def setA : Set ℝ := {x | Real.log x / Real.log 2 > 1}
def setB (a : ℝ) : Set ℝ := {x | x < a}
def setIntersection (b : ℝ) : Set ℝ := {x | b < x ∧ x < 2 * b + 3}

theorem find_a (h : setA ∩ setB a = setIntersection b) : a = 7 := 
by
  sorry

end find_a_l13_13760


namespace evaluate_complex_ratio_l13_13190

noncomputable def complex_ratio (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^3 + a^2 * b + a * b^2 + b^3 = 0) : ℂ :=
(a^12 + b^12) / (a + b)^12

theorem evaluate_complex_ratio (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^3 + a^2 * b + a * b^2 + b^3 = 0) :
  complex_ratio a b h1 h2 h3 = 1 / 32 :=
by
  sorry

end evaluate_complex_ratio_l13_13190


namespace sum_of_numbers_with_lcm_and_ratio_l13_13951

theorem sum_of_numbers_with_lcm_and_ratio 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 48)
  (h_ratio : a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3) : 
  a + b = 80 := 
by sorry

end sum_of_numbers_with_lcm_and_ratio_l13_13951


namespace larger_sign_diameter_l13_13688

theorem larger_sign_diameter (d k : ℝ) 
  (h1 : ∀ d, d > 0) 
  (h2 : ∀ k, (π * (k * d / 2)^2 = 49 * π * (d / 2)^2)) : 
  k = 7 :=
by
sorry

end larger_sign_diameter_l13_13688


namespace determinant_identity_l13_13476

variable (x y z w : ℝ)
variable (h1 : x * w - y * z = -3)

theorem determinant_identity :
  (x + z) * w - (y + w) * z = -3 :=
by sorry

end determinant_identity_l13_13476


namespace sandy_age_l13_13645

variable (S M : ℕ)

-- Conditions
def condition1 := M = S + 12
def condition2 := S * 9 = M * 7

theorem sandy_age : condition1 S M → condition2 S M → S = 42 := by
  intros h1 h2
  sorry

end sandy_age_l13_13645


namespace minimum_shift_value_l13_13793

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem minimum_shift_value :
  ∃ m > 0, ∀ x, f (x + m) = Real.sin x ∧ m = 3 * Real.pi / 2 :=
by
  sorry

end minimum_shift_value_l13_13793


namespace find_line_equation_l13_13536

noncomputable def y_line (m b x : ℝ) : ℝ := m * x + b
noncomputable def quadratic_y (x : ℝ) : ℝ := x ^ 2 + 8 * x + 7

noncomputable def equation_of_the_line : Prop :=
  ∃ (m b k : ℝ),
    (quadratic_y k = y_line m b k + 6 ∨ quadratic_y k = y_line m b k - 6) ∧
    (y_line m b 2 = 7) ∧ 
    b ≠ 0 ∧
    y_line 19.5 (-32) = y_line m b

theorem find_line_equation : equation_of_the_line :=
sorry

end find_line_equation_l13_13536


namespace largest_even_digit_multiple_of_9_below_1000_l13_13819

theorem largest_even_digit_multiple_of_9_below_1000 :
  ∃ n : ℕ, n = 882 ∧ n < 1000 ∧ (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d % 2 = 0) ∧ n % 9 = 0 :=
by
  existsi 882
  split
  { rfl }
  split
  { norm_num }
  split
  { intro d
    intro h
    fin_cases d with
    | h1 => norm_num
    | h2 => norm_num
    | h3 => norm_num }
  { norm_num }

end largest_even_digit_multiple_of_9_below_1000_l13_13819


namespace series_value_l13_13120

theorem series_value : ∑ n in Nat.range ∞, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
by
  sorry

end series_value_l13_13120


namespace problem1_problem2_problem3_problem4_l13_13441

theorem problem1 : -15 + (-23) - 26 - (-15) = -49 := 
by sorry

theorem problem2 : (- (1 / 2) + (2 / 3) - (1 / 4)) * (-24) = 2 := 
by sorry

theorem problem3 : -24 / (-6) * (- (1 / 4)) = -1 := 
by sorry

theorem problem4 : -1 ^ 2024 - (-2) ^ 3 - 3 ^ 2 + 2 / (2 / 3 * (3 / 2)) = 5 / 2 := 
by sorry

end problem1_problem2_problem3_problem4_l13_13441


namespace solve_investment_problem_l13_13850

def remaining_rate_proof (A I A1 R1 A2 R2 x : ℚ) : Prop :=
  let income1 := A1 * (R1 / 100)
  let income2 := A2 * (R2 / 100)
  let remaining := A - A1 - A2
  let required_income := I - (income1 + income2)
  let expected_rate_in_float := (required_income / remaining) * 100
  expected_rate_in_float = x

theorem solve_investment_problem :
  remaining_rate_proof 15000 800 5000 3 6000 4.5 9.5 :=
by
  -- proof goes here
  sorry

end solve_investment_problem_l13_13850


namespace range_of_a_l13_13746

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (1 - 2 * a) * x + 3 * a else Real.log x

theorem range_of_a (a : ℝ) : (-1 ≤ a ∧ a < 1/2) ↔
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) :=
by
  sorry

end range_of_a_l13_13746


namespace total_pink_crayons_l13_13033

-- Define the conditions
def Mara_crayons : ℕ := 40
def Mara_pink_percent : ℕ := 10
def Luna_crayons : ℕ := 50
def Luna_pink_percent : ℕ := 20

-- Define the proof problem statement
theorem total_pink_crayons : 
  (Mara_crayons * Mara_pink_percent / 100) + (Luna_crayons * Luna_pink_percent / 100) = 14 := 
by sorry

end total_pink_crayons_l13_13033


namespace total_profit_calculation_l13_13062

variable (investment_Tom : ℝ) (investment_Jose : ℝ) (time_Jose : ℝ) (share_Jose : ℝ) (total_time : ℝ) 
variable (total_profit : ℝ)

theorem total_profit_calculation 
  (h1 : investment_Tom = 30000) 
  (h2 : investment_Jose = 45000) 
  (h3 : time_Jose = 10) -- Jose joined 2 months later, so he invested for 10 months out of 12
  (h4 : share_Jose = 30000) 
  (h5 : total_time = 12) 
  : total_profit = 54000 :=
sorry

end total_profit_calculation_l13_13062


namespace correct_calculation_l13_13232

theorem correct_calculation (a b : ℝ) :
  2 * a^2 * b - 3 * a^2 * b = -a^2 * b ∧
  ¬ (a^3 * a^4 = a^12) ∧
  ¬ ((-2 * a^2 * b)^3 = -6 * a^6 * b^3) ∧
  ¬ ((a + b)^2 = a^2 + b^2) :=
by
  sorry

end correct_calculation_l13_13232


namespace num_floors_each_building_l13_13095

theorem num_floors_each_building
  (floors_each_building num_apartments_per_floor num_doors_per_apartment total_doors : ℕ)
  (h1 : floors_each_building = F)
  (h2 : num_apartments_per_floor = 6)
  (h3 : num_doors_per_apartment = 7)
  (h4 : total_doors = 1008)
  (eq1 : 2 * floors_each_building * num_apartments_per_floor * num_doors_per_apartment = total_doors) :
  F = 12 :=
sorry

end num_floors_each_building_l13_13095


namespace cube_surface_area_l13_13207

theorem cube_surface_area (a : ℝ) (h : a = 1) :
    6 * a^2 = 6 := by
  sorry

end cube_surface_area_l13_13207


namespace number_of_possible_sums_l13_13196

open Finset

theorem number_of_possible_sums (A : Finset ℕ) (C : Finset ℕ) (hA : A = range 121 \ erase (range 121) 0) (hC : C.card = 80) :
  ∃ n : ℕ, n = 3201 :=
by
  sorry  -- Proof to be completed

end number_of_possible_sums_l13_13196


namespace sum_of_two_squares_l13_13889

theorem sum_of_two_squares (n : ℕ) (k m : ℤ) : 2 * n = k^2 + m^2 → ∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end sum_of_two_squares_l13_13889


namespace marble_count_l13_13962

-- Definitions from conditions
variable (M P : ℕ)
def condition1 : Prop := M = 26 * P
def condition2 : Prop := M = 28 * (P - 1)

-- Theorem to be proved
theorem marble_count (h1 : condition1 M P) (h2 : condition2 M P) : M = 364 := 
by
  sorry

end marble_count_l13_13962


namespace roots_of_polynomial_fraction_l13_13355

theorem roots_of_polynomial_fraction (a b c : ℝ)
  (h1 : a + b + c = 6)
  (h2 : a * b + a * c + b * c = 11)
  (h3 : a * b * c = 6) :
  a / (b * c + 2) + b / (a * c + 2) + c / (a * b + 2) = 3 / 2 := 
by
  sorry

end roots_of_polynomial_fraction_l13_13355


namespace find_a_2016_l13_13592

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (n + 1) / n * a n

theorem find_a_2016 (a : ℕ → ℝ) (h : seq a) : a 2016 = 4032 :=
by
  sorry

end find_a_2016_l13_13592


namespace first_hour_rain_l13_13019

variable (x : ℝ)
variable (rain_1st_hour : ℝ) (rain_2nd_hour : ℝ)
variable (total_rain : ℝ)

-- Define the conditions
def condition_1 (x rain_2nd_hour : ℝ) : Prop :=
  rain_2nd_hour = 2 * x + 7

def condition_2 (x rain_2nd_hour total_rain : ℝ) : Prop :=
  x + rain_2nd_hour = total_rain

-- Prove the amount of rain in the first hour
theorem first_hour_rain (h1 : condition_1 x rain_2nd_hour)
                         (h2 : condition_2 x rain_2nd_hour total_rain)
                         (total_rain_is_22 : total_rain = 22) :
  x = 5 :=
by
  -- Proof steps go here
  sorry

end first_hour_rain_l13_13019


namespace find_m_l13_13882

def hyperbola_focus (x y : ℝ) (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 = 9 ∧ b^2 = -m ∧ (x - 0)^2 / a^2 - (y - 0)^2 / b^2 = 1

theorem find_m (m : ℝ) (H : hyperbola_focus 5 0 m) : m = -16 :=
by
  sorry

end find_m_l13_13882


namespace brown_eyed_brunettes_count_l13_13494

-- Definitions of conditions
variables (total_students blue_eyed_blondes brunettes brown_eyed_students : ℕ)
variable (brown_eyed_brunettes : ℕ)

-- Initial conditions
axiom h1 : total_students = 60
axiom h2 : blue_eyed_blondes = 18
axiom h3 : brunettes = 40
axiom h4 : brown_eyed_students = 24

-- Proof objective
theorem brown_eyed_brunettes_count :
  brown_eyed_brunettes = 24 - (24 - (20 - (20 - 18))) := sorry

end brown_eyed_brunettes_count_l13_13494


namespace correct_conclusion_l13_13029

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 2 else n * 2^n

theorem correct_conclusion (n : ℕ) (h₁ : ∀ k : ℕ, k > 0 → a_n (k + 1) - 2 * a_n k = 2^(k + 1)) :
  a_n n = n * 2 ^ n :=
by
  sorry

end correct_conclusion_l13_13029


namespace sum_series_l13_13126

theorem sum_series :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
sorry

end sum_series_l13_13126


namespace jennifer_money_left_l13_13188

theorem jennifer_money_left (initial_amount sandwich_fraction museum_fraction book_fraction : ℚ)
    (initial_eq : initial_amount = 90) 
    (sandwich_eq : sandwich_fraction = 1/5) 
    (museum_eq : museum_fraction = 1/6) 
    (book_eq : book_fraction = 1/2) :
    initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction) = 12 := 
by 
  sorry

end jennifer_money_left_l13_13188


namespace market_cost_l13_13638

theorem market_cost (peach_pies apple_pies blueberry_pies : ℕ) (fruit_per_pie : ℕ) 
  (price_per_pound_apple price_per_pound_blueberry price_per_pound_peach : ℕ) :
  peach_pies = 5 ∧
  apple_pies = 4 ∧
  blueberry_pies = 3 ∧
  fruit_per_pie = 3 ∧
  price_per_pound_apple = 1 ∧
  price_per_pound_blueberry = 1 ∧
  price_per_pound_peach = 2 →
  let total_peaches := peach_pies * fruit_per_pie in
  let total_apples := apple_pies * fruit_per_pie in
  let total_blueberries := blueberry_pies * fruit_per_pie in
  let cost_apples := total_apples * price_per_pound_apple in
  let cost_blueberries := total_blueberries * price_per_pound_blueberry in
  let cost_peaches := total_peaches * price_per_pound_peach in
  (cost_apples + cost_blueberries + cost_peaches = 51) :=
by
  intros
  sorry

end market_cost_l13_13638


namespace speed_in_m_per_s_eq_l13_13960

theorem speed_in_m_per_s_eq : (1 : ℝ) / 3.6 = (0.27777 : ℝ) :=
by sorry

end speed_in_m_per_s_eq_l13_13960


namespace simplify_sqrt_expression_l13_13370

theorem simplify_sqrt_expression (h : Real.sqrt 3 > 1) :
  Real.sqrt ((1 - Real.sqrt 3) ^ 2) = Real.sqrt 3 - 1 :=
by
  sorry

end simplify_sqrt_expression_l13_13370


namespace complementary_angle_measure_l13_13855

theorem complementary_angle_measure (A S C : ℝ) (h1 : A = 45) (h2 : A + S = 180) (h3 : A + C = 90) (h4 : S = 3 * C) : C = 45 :=
by
  sorry

end complementary_angle_measure_l13_13855


namespace part1_part2_l13_13313

-- Part (1)  
theorem part1 (m : ℝ) : (∀ x : ℝ, 1 < x ∧ x < 3 → 2 * m < x ∧ x < 1 - m) ↔ (m ≤ -2) :=
sorry

-- Part (2)
theorem part2 (m : ℝ) : (∀ x : ℝ, (1 < x ∧ x < 3) → ¬ (2 * m < x ∧ x < 1 - m)) ↔ (0 ≤ m) :=
sorry

end part1_part2_l13_13313


namespace find_a_l13_13888

theorem find_a (a : ℝ) :
  (∀ x : ℝ, ((x^2 - 4 * x + a) + |x - 3| ≤ 5) → x ≤ 3) →
  (∃ x : ℝ, x = 3 ∧ ((x^2 - 4 * x + a) + |x - 3| ≤ 5)) →
  a = 2 := 
by
  sorry

end find_a_l13_13888


namespace slope_of_parallel_line_l13_13398

theorem slope_of_parallel_line (a b c : ℝ) (h : a = 3 ∧ b = -6 ∧ c = 12) :
  ∃ m : ℝ, (∀ (x y : ℝ), 3 * x - 6 * y = 12 → y = m * x - 2) ∧ m = 1/2 := 
sorry

end slope_of_parallel_line_l13_13398


namespace hexagon_angle_sum_l13_13272

theorem hexagon_angle_sum (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) :
  a1 + a2 + a3 + a4 = 360 ∧ b1 + b2 + b3 + b4 = 360 → 
  a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 = 720 :=
by
  sorry

end hexagon_angle_sum_l13_13272


namespace part1_part2_l13_13912

/-- Definition of set A as roots of the equation x^2 - 3x + 2 = 0 --/
def set_A : Set ℝ := {x | x ^ 2 - 3 * x + 2 = 0}

/-- Definition of set B as roots of the equation x^2 + (a - 1)x + a^2 - 5 = 0 --/
def set_B (a : ℝ) : Set ℝ := {x | x ^ 2 + (a - 1) * x + a ^ 2 - 5 = 0}

/-- Proof for intersection condition --/
theorem part1 (a : ℝ) : (set_A ∩ set_B a = {2}) → (a = -3 ∨ a = 1) := by
  sorry

/-- Proof for union condition --/
theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -3 ∨ a > 7 / 3) := by
  sorry

end part1_part2_l13_13912


namespace train_length_is_140_l13_13426

noncomputable def train_length (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  let distance := speed_ms * time_s
  distance - bridge_length_m

theorem train_length_is_140 :
  train_length 45 30 235 = 140 := by
  sorry

end train_length_is_140_l13_13426


namespace find_a6_l13_13017

variable {a : ℕ → ℝ} -- Sequence a is indexed by natural numbers and the terms are real numbers.

-- Conditions
def a_is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)
def a1_eq_4 (a : ℕ → ℝ) := a 1 = 4
def a3_eq_a2_mul_a4 (a : ℕ → ℝ) := a 3 = a 2 * a 4

theorem find_a6 (a : ℕ → ℝ) 
  (h1 : a_is_geom_seq a)
  (h2 : a1_eq_4 a)
  (h3 : a3_eq_a2_mul_a4 a) : 
  a 6 = 1 / 8 ∨ a 6 = - (1 / 8) := 
by 
  sorry

end find_a6_l13_13017


namespace chloe_total_score_l13_13777

theorem chloe_total_score :
  let first_level_treasure_points := 9
  let first_level_bonus_points := 15
  let first_level_treasures := 6
  let second_level_treasure_points := 11
  let second_level_bonus_points := 20
  let second_level_treasures := 3

  let first_level_score := first_level_treasures * first_level_treasure_points + first_level_bonus_points
  let second_level_score := second_level_treasures * second_level_treasure_points + second_level_bonus_points

  first_level_score + second_level_score = 122 :=
by
  sorry

end chloe_total_score_l13_13777


namespace k_at_1_value_l13_13304

def h (x p : ℝ) := x^3 + p * x^2 + 2 * x + 20
def k (x p q r : ℝ) := x^4 + 2 * x^3 + q * x^2 + 50 * x + r

theorem k_at_1_value (p q r : ℝ) (h_distinct_roots : ∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ → h x₁ p = 0 → h x₂ p = 0 → h x₃ p = 0 → k x₁ p q r = 0 ∧ k x₂ p q r = 0 ∧ k x₃ p q r = 0):
  k 1 (-28) (2 - -28 * -30) (-20 * -30) = -155 :=
by
  sorry

end k_at_1_value_l13_13304


namespace no_unhappy_days_l13_13673

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
  sorry

end no_unhappy_days_l13_13673


namespace rabbits_in_cage_l13_13776

theorem rabbits_in_cage (heads legs : ℝ) (total_heads : heads = 40) 
  (condition : legs = 8 + 10 * (2 * (heads - rabbits))) :
  ∃ rabbits : ℝ, rabbits = 33 :=
by
  sorry

end rabbits_in_cage_l13_13776


namespace jack_change_l13_13628

theorem jack_change :
  let discountedCost1 := 4.50
  let discountedCost2 := 4.50
  let discountedCost3 := 5.10
  let cost4 := 7.00
  let totalDiscountedCost := discountedCost1 + discountedCost2 + discountedCost3 + cost4
  let tax := totalDiscountedCost * 0.05
  let taxRounded := 1.06 -- Tax rounded to nearest cent
  let totalCostWithTax := totalDiscountedCost + taxRounded
  let totalCostWithServiceFee := totalCostWithTax + 2.00
  let totalPayment := 20 + 10 + 4 * 1
  let change := totalPayment - totalCostWithServiceFee
  change = 9.84 :=
by
  sorry

end jack_change_l13_13628


namespace noah_garden_larger_by_75_l13_13510

-- Define the dimensions of Liam's garden
def length_liam : ℕ := 30
def width_liam : ℕ := 50

-- Define the dimensions of Noah's garden
def length_noah : ℕ := 35
def width_noah : ℕ := 45

-- Define the areas of the gardens
def area_liam : ℕ := length_liam * width_liam
def area_noah : ℕ := length_noah * width_noah

theorem noah_garden_larger_by_75 :
  area_noah - area_liam = 75 :=
by
  -- The proof goes here
  sorry

end noah_garden_larger_by_75_l13_13510


namespace Reese_initial_savings_l13_13366

theorem Reese_initial_savings (F M A R : ℝ) (savings : ℝ) :
  F = 0.2 * savings →
  M = 0.4 * savings →
  A = 1500 →
  R = 2900 →
  savings = 11000 :=
by
  sorry

end Reese_initial_savings_l13_13366


namespace podcast_length_l13_13367

theorem podcast_length (x : ℝ) (hx : x + 2 * x + 1.75 + 1 + 1 = 6) : x = 0.75 :=
by {
  -- We do not need the proof steps here
  sorry
}

end podcast_length_l13_13367


namespace water_leftover_l13_13933

theorem water_leftover (players : ℕ) (total_water_l : ℕ) (water_per_player_ml : ℕ) (spill_water_ml : ℕ)
  (h1 : players = 30) 
  (h2 : total_water_l = 8) 
  (h3 : water_per_player_ml = 200) 
  (h4 : spill_water_ml = 250) : 
  (total_water_l * 1000 - (players * water_per_player_ml + spill_water_ml) = 1750) :=
by
  -- conversion of total water to milliliters
  let total_water_ml := total_water_l * 1000
  -- calculation of total water used for players
  let total_water_used_for_players := players * water_per_player_ml
  -- calculation of total water including spill
  let total_water_used := total_water_used_for_players + spill_water_ml
  -- leftover water calculation
  have calculation : total_water_l * 1000 - (players * water_per_player_ml + spill_water_ml) = total_water_ml - total_water_used, by
    rw [total_water_ml, total_water_used, total_water_used_for_players]
  rw calculation
  -- conclusion by substituting known values
  rw [h1, h2, h3, h4]
  norm_num

end water_leftover_l13_13933


namespace equality_of_expressions_l13_13006

theorem equality_of_expressions (a b c : ℝ) (h : a = b + c + 2) : 
  a + b * c = (a + b) * (a + c) ↔ a = 0 ∨ a = 1 :=
by sorry

end equality_of_expressions_l13_13006


namespace towel_length_decrease_l13_13711

theorem towel_length_decrease (L B : ℝ) (HL1: L > 0) (HB1: B > 0)
  (length_percent_decr : ℝ) (breadth_decr : B' = 0.8 * B) 
  (area_decr : (L' * B') = 0.64 * (L * B)) :
  (L' = 0.8 * L) ∧ (length_percent_decrease = 20) := by
  sorry

end towel_length_decrease_l13_13711


namespace inequality_correct_l13_13608

theorem inequality_correct (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1/a) < (1/b) :=
sorry

end inequality_correct_l13_13608


namespace elly_candies_l13_13740

theorem elly_candies (a b c : ℝ) (h1 : a * b * c = 216) : 
  24 * 216 = 5184 :=
by
  sorry

end elly_candies_l13_13740


namespace average_minutes_per_day_l13_13857

theorem average_minutes_per_day (e : ℕ) (h_e_pos : 0 < e) : 
  let sixth_grade_minutes := 20
  let seventh_grade_minutes := 18
  let eighth_grade_minutes := 12
  
  let sixth_graders := 3 * e
  let seventh_graders := 4 * e
  let eighth_graders := e
  
  let total_minutes := sixth_grade_minutes * sixth_graders + seventh_grade_minutes * seventh_graders + eighth_grade_minutes * eighth_graders
  let total_students := sixth_graders + seventh_graders + eighth_graders
  
  (total_minutes / total_students) = 18 := by
sorry

end average_minutes_per_day_l13_13857


namespace students_neither_cs_nor_robotics_l13_13432

theorem students_neither_cs_nor_robotics
  (total_students : ℕ)
  (cs_students : ℕ)
  (robotics_students : ℕ)
  (both_cs_and_robotics : ℕ)
  (H1 : total_students = 150)
  (H2 : cs_students = 90)
  (H3 : robotics_students = 70)
  (H4 : both_cs_and_robotics = 20) :
  (total_students - (cs_students + robotics_students - both_cs_and_robotics)) = 10 :=
by
  sorry

end students_neither_cs_nor_robotics_l13_13432


namespace no_solution_abs_eq_2_l13_13729

theorem no_solution_abs_eq_2 (x : ℝ) :
  |x - 5| = |x + 3| + 2 → false :=
by sorry

end no_solution_abs_eq_2_l13_13729


namespace triangle_area_decrease_l13_13241

theorem triangle_area_decrease (B H : ℝ) : 
  let A_original := (B * H) / 2
  let H_new := 0.60 * H
  let B_new := 1.40 * B
  let A_new := (B_new * H_new) / 2
  A_new = 0.42 * A_original :=
by
  sorry

end triangle_area_decrease_l13_13241


namespace Olivia_steps_l13_13639

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem Olivia_steps :
  let x := 57 + 68
  let y := x - 15
  round_to_nearest_ten y = 110 := 
by
  sorry

end Olivia_steps_l13_13639


namespace find_f_10_l13_13269

def f (x : ℤ) : ℤ := sorry

noncomputable def h (x : ℤ) : ℤ := f(x) + x

axiom condition_1 : f(1) + 1 > 0

axiom condition_2 : ∀ (x y : ℤ), f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y

axiom condition_3 : ∀ (x : ℤ), 2 * f(x) = f(x + 1) - x + 1

theorem find_f_10 : f(10) = 1014 := sorry

end find_f_10_l13_13269


namespace sum_of_solutions_l13_13872

theorem sum_of_solutions (x : ℝ) : (∃ x₁ x₂ : ℝ, (x - 4)^2 = 16 ∧ x = x₁ ∨ x = x₂ ∧ x₁ + x₂ = 8) :=
by sorry

end sum_of_solutions_l13_13872


namespace find_m_l13_13010

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem find_m (a b m : ℝ) (h1 : f m a b = 0) (h2 : 3 * m^2 + 2 * a * m + b = 0)
  (h3 : f (m / 3) a b = 1 / 2) (h4 : m ≠ 0) : m = 3 / 2 :=
  sorry

end find_m_l13_13010


namespace smaller_angle_3_40_l13_13082

-- Definitions using the conditions provided in the problem
def is_12_hour_clock (clock : Type) := 
  ∀ t : ℕ, 1 ≤ t ∧ t ≤ 12

def time_is_3_40 (time : Type) := 
  ∃ h m : ℕ, h = 3 ∧ m = 40

-- The theorem that needs to be proven
theorem smaller_angle_3_40 (clock : Type) (time : Type)
  (h1 : is_12_hour_clock clock) 
  (h2 : time_is_3_40 time) : 
  ∃ alpha : ℝ, alpha = 130.0 :=
begin
  sorry
end

end smaller_angle_3_40_l13_13082


namespace constants_solution_l13_13876

theorem constants_solution :
  let a := 56 / 9
  let c := 5 / 3
  (∀ (v1 v2 : ℝ ^ 3),
    v1 = ![a, -1, c]
    ∧ v2 = ![8, 4, 6]
    → (v1.cross_product v2 = ![-14, -24, 34]))
    ↔ (a, c) = (56 / 9, 5 / 3) :=
by
  sorry

end constants_solution_l13_13876


namespace wuzhen_conference_arrangements_l13_13730

theorem wuzhen_conference_arrangements 
  (countries : Finset ℕ)
  (hotels : Finset ℕ)
  (h_countries_count : countries.card = 5)
  (h_hotels_count : hotels.card = 3) :
  ∃ f : ℕ → ℕ,
  (∀ c ∈ countries, f c ∈ hotels) ∧
  (∀ h ∈ hotels, ∃ c ∈ countries, f c = h) ∧
  (Finset.card (Set.toFinset (f '' countries)) = 3) ∧
  ∃ n : ℕ,
  n = 150 := 
sorry

end wuzhen_conference_arrangements_l13_13730


namespace max_students_total_l13_13224

def max_students_class (a b : ℕ) (h : 3 * a + 5 * b = 115) : ℕ :=
  a + b

theorem max_students_total :
  ∃ a b : ℕ, 3 * a + 5 * b = 115 ∧ max_students_class a b (by sorry) = 37 :=
sorry

end max_students_total_l13_13224


namespace greatest_three_digit_number_divisible_by_3_6_5_l13_13227

theorem greatest_three_digit_number_divisible_by_3_6_5 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 3 = 0) ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 3 = 0) ∧ (m % 6 = 0) ∧ (m % 5 = 0) → m ≤ n) ∧ n = 990 := 
by
  sorry

end greatest_three_digit_number_divisible_by_3_6_5_l13_13227


namespace area_of_region_l13_13135

theorem area_of_region (x y : ℝ) :
  x ≤ 2 * y ∧ y ≤ 2 * x ∧ x + y ≤ 60 →
  ∃ (A : ℝ), A = 600 :=
by
  sorry

end area_of_region_l13_13135


namespace arithmetic_geometric_sequence_ab_l13_13762

theorem arithmetic_geometric_sequence_ab :
  ∀ (a l m b n : ℤ), 
    (b < 0) → 
    (2 * a = -10) → 
    (b^2 = 9) → 
    ab = 15 :=
by
  intros a l m b n hb ha hb_eq
  sorry

end arithmetic_geometric_sequence_ab_l13_13762


namespace percent_palindromes_containing_7_l13_13400

theorem percent_palindromes_containing_7 : 
  let num_palindromes := 90
  let num_palindrome_with_7 := 19
  (num_palindrome_with_7 / num_palindromes * 100) = 21.11 := 
by
  sorry

end percent_palindromes_containing_7_l13_13400


namespace parallel_lines_a_l13_13007

-- Definitions of the lines
def l1 (a : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y - 1 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := 6 * x + a * y + 2 = 0

-- The main theorem to prove
theorem parallel_lines_a (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y → l2 a x y) → (a = 3) := 
sorry

end parallel_lines_a_l13_13007


namespace total_hours_before_midterms_l13_13974

-- Define the hours spent on each activity per week
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3

-- Sum up the total hours spent on extracurriculars per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Define semester information
def total_weeks_per_semester : ℕ := 12
def weeks_before_midterms : ℕ := total_weeks_per_semester / 2
def weeks_sick : ℕ := 2
def active_weeks_before_midterms : ℕ := weeks_before_midterms - weeks_sick

-- Define the theorem statement about total hours before midterms
theorem total_hours_before_midterms : total_hours_per_week * active_weeks_before_midterms = 52 := by
  -- We skip the actual proof here
  sorry

end total_hours_before_midterms_l13_13974


namespace clerks_needed_eq_84_l13_13418

def forms_processed_per_hour : ℕ := 25
def type_a_forms_count : ℕ := 3000
def type_b_forms_count : ℕ := 4000
def type_a_form_time_minutes : ℕ := 3
def type_b_form_time_minutes : ℕ := 4
def working_hours_per_day : ℕ := 5
def total_minutes_in_an_hour : ℕ := 60
def forms_time_needed (count : ℕ) (time_per_form : ℕ) : ℕ := count * time_per_form
def total_forms_time_needed : ℕ := forms_time_needed type_a_forms_count type_a_form_time_minutes +
                                    forms_time_needed type_b_forms_count type_b_form_time_minutes
def total_hours_needed : ℕ := total_forms_time_needed / total_minutes_in_an_hour
def clerk_hours_needed : ℕ := total_hours_needed / working_hours_per_day
def required_clerks : ℕ := Nat.ceil (clerk_hours_needed)

theorem clerks_needed_eq_84 :
  required_clerks = 84 :=
by
  sorry

end clerks_needed_eq_84_l13_13418


namespace ratio_of_sold_phones_to_production_l13_13843

def last_years_production : ℕ := 5000
def this_years_production : ℕ := 2 * last_years_production
def phones_left_in_factory : ℕ := 7500
def sold_phones : ℕ := this_years_production - phones_left_in_factory

theorem ratio_of_sold_phones_to_production : 
  (sold_phones : ℚ) / this_years_production = 1 / 4 := 
by
  sorry

end ratio_of_sold_phones_to_production_l13_13843


namespace amusement_park_weekly_revenue_l13_13434

def ticket_price : ℕ := 3
def visitors_mon_to_fri_per_day : ℕ := 100
def visitors_saturday : ℕ := 200
def visitors_sunday : ℕ := 300

theorem amusement_park_weekly_revenue : 
  let total_visitors_weekdays := visitors_mon_to_fri_per_day * 5
  let total_visitors_weekend := visitors_saturday + visitors_sunday
  let total_visitors := total_visitors_weekdays + total_visitors_weekend
  let total_revenue := total_visitors * ticket_price
  total_revenue = 3000 := by
  sorry

end amusement_park_weekly_revenue_l13_13434


namespace sugar_left_in_grams_l13_13359

theorem sugar_left_in_grams 
  (initial_ounces : ℝ) (spilled_ounces : ℝ) (conversion_factor : ℝ)
  (h_initial : initial_ounces = 9.8) (h_spilled : spilled_ounces = 5.2)
  (h_conversion : conversion_factor = 28.35) :
  (initial_ounces - spilled_ounces) * conversion_factor = 130.41 := 
by
  sorry

end sugar_left_in_grams_l13_13359


namespace megans_candy_l13_13512

variable (M : ℕ)

theorem megans_candy (h1 : M * 3 + 10 = 25) : M = 5 :=
by sorry

end megans_candy_l13_13512


namespace caroline_socks_gift_l13_13724

theorem caroline_socks_gift :
  ∀ (initial lost donated_fraction purchased total received),
    initial = 40 →
    lost = 4 →
    donated_fraction = 2 / 3 →
    purchased = 10 →
    total = 25 →
    received = total - (initial - lost - donated_fraction * (initial - lost) + purchased) →
    received = 3 :=
by
  intros initial lost donated_fraction purchased total received
  intro h_initial h_lost h_donated_fraction h_purchased h_total h_received
  sorry

end caroline_socks_gift_l13_13724


namespace remainder_4x_mod_7_l13_13491

theorem remainder_4x_mod_7 (x : ℤ) (k : ℤ) (h : x = 7 * k + 5) : (4 * x) % 7 = 6 :=
by
  sorry

end remainder_4x_mod_7_l13_13491


namespace people_per_pizza_l13_13197

def pizza_cost := 12 -- dollars per pizza
def babysitting_earnings_per_night := 4 -- dollars per night
def nights_babysitting := 15
def total_people := 15

theorem people_per_pizza : (babysitting_earnings_per_night * nights_babysitting / pizza_cost) = (total_people / ((babysitting_earnings_per_night * nights_babysitting / pizza_cost))) := 
by
  sorry

end people_per_pizza_l13_13197


namespace train_pass_platform_time_l13_13108

theorem train_pass_platform_time (l v t : ℝ) (h1 : v = l / t) (h2 : l > 0) (h3 : t > 0) :
  ∃ T : ℝ, T = 3.5 * t := by
  sorry

end train_pass_platform_time_l13_13108


namespace triangle_inequality_l13_13884

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end triangle_inequality_l13_13884


namespace problem_statement_l13_13861

theorem problem_statement (a b c x : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0)
  (eq1 : (a * x^4 / b * c)^3 = x^3)
  (sum_eq : a + b + c = 9) :
  (x = 1 ∨ x = -1) ∧ a = 1 ∧ b = 4 ∧ c = 4 :=
by
  sorry

end problem_statement_l13_13861


namespace tip_percentage_l13_13957

theorem tip_percentage (T : ℝ) 
  (total_cost meal_cost sales_tax : ℝ)
  (h1 : meal_cost = 61.48)
  (h2 : sales_tax = 0.07 * meal_cost)
  (h3 : total_cost = meal_cost + sales_tax + T * meal_cost)
  (h4 : total_cost ≤ 75) :
  T ≤ 0.1499 :=
by
  -- main proof goes here
  sorry

end tip_percentage_l13_13957


namespace f_10_l13_13257

namespace MathProof

variable (f : ℤ → ℤ)

-- Condition 1: f(1) + 1 > 0
axiom cond1 : f 1 + 1 > 0

-- Condition 2: f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y for any x, y ∈ ℤ
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y

-- Condition 3: 2 * f(x) = f(x + 1) - x + 1 for any x ∈ ℤ
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

-- We need to prove f(10) = 1014
theorem f_10 : f 10 = 1014 :=
by
  sorry

end MathProof

end f_10_l13_13257


namespace rational_quotient_of_arith_geo_subseq_l13_13641

theorem rational_quotient_of_arith_geo_subseq (A d : ℝ) (h_d_nonzero : d ≠ 0)
    (h_contains_geo : ∃ (q : ℝ) (k m n : ℕ), q ≠ 1 ∧ q ≠ 0 ∧ 
        A + k * d = (A + m * d) * q ∧ A + m * d = (A + n * d) * q)
    : ∃ (r : ℚ), A / d = r :=
  sorry

end rational_quotient_of_arith_geo_subseq_l13_13641


namespace calculate_expression_l13_13721

theorem calculate_expression : 4 * 6 * 8 + 24 / 4 - 10 = 188 := by
  sorry

end calculate_expression_l13_13721


namespace fair_die_probability_l13_13611

noncomputable def probability_of_rolling_four_ones_in_five_rolls 
  (prob_1 : ℚ) (prob_not_1 : ℚ) (n : ℕ) (k : ℕ) : ℚ :=
(binomial n k) * (prob_1 ^ k) * (prob_not_1 ^ (n - k))

theorem fair_die_probability :
  let n := 5
  let k := 4
  let prob_1 := 1 / 6
  let prob_not_1 := 5 / 6
  probability_of_rolling_four_ones_in_five_rolls prob_1 prob_not_1 n k = 25 / 7776 := by
  sorry

end fair_die_probability_l13_13611


namespace total_hours_eq_52_l13_13979

def hours_per_week_on_extracurriculars : ℕ := 2 + 8 + 3  -- Total hours per week
def weeks_in_semester : ℕ := 12  -- Total weeks in a semester
def weeks_before_midterms : ℕ := weeks_in_semester / 2  -- Weeks before midterms
def sick_weeks : ℕ := 2  -- Weeks Annie takes off sick
def active_weeks_before_midterms : ℕ := weeks_before_midterms - sick_weeks  -- Active weeks before midterms

def total_extracurricular_hours_before_midterms : ℕ :=
  hours_per_week_on_extracurriculars * active_weeks_before_midterms

theorem total_hours_eq_52 :
  total_extracurricular_hours_before_midterms = 52 :=
by
  sorry

end total_hours_eq_52_l13_13979


namespace polar_to_cartesian_coordinates_l13_13615

theorem polar_to_cartesian_coordinates (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = 5 * Real.pi / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (-Real.sqrt 3, 1) :=
by
  sorry

end polar_to_cartesian_coordinates_l13_13615


namespace age_problem_l13_13546

theorem age_problem (c b a : ℕ) (h1 : b = 2 * c) (h2 : a = b + 2) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end age_problem_l13_13546


namespace length_of_ON_l13_13154

noncomputable def proof_problem : Prop :=
  let hyperbola := { x : ℝ × ℝ | x.1 ^ 2 - x.2 ^ 2 = 1 }
  ∃ (F1 F2 P : ℝ × ℝ) (O : ℝ × ℝ) (N : ℝ × ℝ),
    O = (0, 0) ∧
    P ∈ hyperbola ∧
    N = ((P.1 + F1.1) / 2, (P.2 + F1.2) / 2) ∧
    dist P F1 = 5 ∧
    ∃ r : ℝ, r = 1.5 ∧ (dist O N = r)

theorem length_of_ON : proof_problem :=
sorry

end length_of_ON_l13_13154


namespace car_speed_5_hours_l13_13417

variable (T : ℝ)
variable (S : ℝ)

theorem car_speed_5_hours (h1 : T > 0) (h2 : 2 * T = S * 5.0) : S = 2 * T / 5.0 :=
sorry

end car_speed_5_hours_l13_13417


namespace shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l13_13055

noncomputable def k : ℝ := (1 / 20) * Real.log (1 / 4)
noncomputable def b : ℝ := Real.log 160
noncomputable def y (x : ℝ) : ℝ := Real.exp (k * x + b)

theorem shelf_life_at_30_degrees : y 30 = 20 := sorry

theorem temperature_condition_for_shelf_life (x : ℝ) : y x ≥ 80 → x ≤ 10 := sorry

end shelf_life_at_30_degrees_temperature_condition_for_shelf_life_l13_13055


namespace minimum_value_l13_13308

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 1) + 9 / y = 1) : 4 * x + y ≥ 21 :=
sorry

end minimum_value_l13_13308


namespace solve_inequality_l13_13651

noncomputable def g (x : ℝ) := Real.arcsin x + x^3

theorem solve_inequality (x : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1)
    (h2 : Real.arcsin (x^2) + Real.arcsin x + x^6 + x^3 > 0) :
    0 < x ∧ x ≤ 1 :=
by
  sorry

end solve_inequality_l13_13651


namespace problem_inequality_l13_13503

variable {n : ℕ}
variable (S_n : Finset (Fin n)) (f : Finset (Fin n) → ℝ)

axiom pos_f : ∀ A : Finset (Fin n), 0 < f A
axiom cond_f : ∀ (A : Finset (Fin n)) (x y : Fin n), x ≠ y → f (A ∪ {x}) * f (A ∪ {y}) ≤ f (A ∪ {x, y}) * f A

theorem problem_inequality (A B : Finset (Fin n)) : f A * f B ≤ f (A ∪ B) * f (A ∩ B) := sorry

end problem_inequality_l13_13503


namespace standard_equation_of_ellipse_locus_of_midpoint_M_l13_13148

-- Define the conditions of the ellipse
def isEllipse (a b c : ℝ) : Prop :=
  a = 2 ∧ c = Real.sqrt 3 ∧ b = Real.sqrt (a^2 - c^2)

-- Define the equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the locus of the midpoint M
def locus_midpoint (x y : ℝ) : Prop :=
  x^2 / 4 + 4 * y^2 = 1

theorem standard_equation_of_ellipse :
  ∃ a b c : ℝ, isEllipse a b c ∧ (∀ x y : ℝ, ellipse_equation x y) :=
sorry

theorem locus_of_midpoint_M :
  ∃ a b c : ℝ, isEllipse a b c ∧ (∀ x y : ℝ, locus_midpoint x y) :=
sorry

end standard_equation_of_ellipse_locus_of_midpoint_M_l13_13148


namespace water_formed_on_combining_l13_13294

theorem water_formed_on_combining (molar_mass_water : ℝ) (n_NaOH : ℝ) (n_HCl : ℝ) :
  n_NaOH = 1 ∧ n_HCl = 1 ∧ molar_mass_water = 18.01528 → 
  n_NaOH * molar_mass_water = 18.01528 :=
by sorry

end water_formed_on_combining_l13_13294


namespace inequality_of_triangle_tangents_l13_13028

theorem inequality_of_triangle_tangents
  (a b c x y z : ℝ)
  (h1 : a = y + z)
  (h2 : b = x + z)
  (h3 : c = x + y)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_tangents : z ≥ y ∧ y ≥ x) :
  (a * z + b * y + c * x ≥ (a^2 + b^2 + c^2) / 2) ∧
  ((a^2 + b^2 + c^2) / 2 ≥ a * x + b * y + c * z) :=
sorry

end inequality_of_triangle_tangents_l13_13028


namespace probability_girls_same_color_l13_13836

open Classical

noncomputable def probability_same_color_marbles : ℚ :=
(3/6) * (2/5) * (1/4) + (3/6) * (2/5) * (1/4)

theorem probability_girls_same_color :
  probability_same_color_marbles = 1/20 := by
  sorry

end probability_girls_same_color_l13_13836


namespace smallest_cube_with_divisor_l13_13192

theorem smallest_cube_with_divisor (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ (m : ℕ), m = (p * q * r^2) ^ 3 ∧ (p * q^3 * r^5 ∣ m) :=
by
  sorry

end smallest_cube_with_divisor_l13_13192


namespace machines_finish_together_in_2_hours_l13_13407

def machineA_time := 4
def machineB_time := 12
def machineC_time := 6

def machineA_rate := 1 / machineA_time
def machineB_rate := 1 / machineB_time
def machineC_rate := 1 / machineC_time

def combined_rate := machineA_rate + machineB_rate + machineC_rate
def total_time := 1 / combined_rate

-- We want to prove that the total_time for machines A, B, and C to finish the job together is 2 hours.
theorem machines_finish_together_in_2_hours : total_time = 2 := by
  sorry

end machines_finish_together_in_2_hours_l13_13407


namespace has_two_distinct_roots_and_ordered_l13_13761

-- Define the context and the conditions of the problem.
variables (a b c : ℝ) (h : a < b) (h2 : b < c)

-- Define the quadratic function derived from the problem.
def quadratic (x : ℝ) : ℝ :=
  (x - a) * (x - b) + (x - a) * (x - c) + (x - b) * (x - c)

-- State the main theorem.
theorem has_two_distinct_roots_and_ordered:
  ∃ x1 x2 : ℝ, quadratic a b c x1 = 0 ∧ quadratic a b c x2 = 0 ∧ a < x1 ∧ x1 < b ∧ b < x2 ∧ x2 < c :=
sorry

end has_two_distinct_roots_and_ordered_l13_13761


namespace proof_problem_l13_13321

noncomputable def p : Prop := ∃ x : ℝ, x^2 + 1 / x^2 ≤ 2
def q : Prop := ¬ p

theorem proof_problem : q ∧ (p ∨ q) :=
by
  -- Insert proof here
  sorry

end proof_problem_l13_13321


namespace intersection_point_on_y_eq_neg_x_l13_13878

theorem intersection_point_on_y_eq_neg_x 
  (α β : ℝ)
  (h1 : ∃ x y : ℝ, (x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1) ∧ 
                   (x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1) ∧ 
                   (y = -x)) :
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 :=
sorry

end intersection_point_on_y_eq_neg_x_l13_13878


namespace lead_atom_ratio_l13_13808

noncomputable def ratio_of_lead_atoms (average_weight : ℝ) 
  (weight_206 : ℕ) (weight_207 : ℕ) (weight_208 : ℕ) 
  (number_206 : ℕ) (number_207 : ℕ) (number_208 : ℕ) : Prop :=
  average_weight = 207.2 ∧ 
  weight_206 = 206 ∧ 
  weight_207 = 207 ∧ 
  weight_208 = 208 ∧ 
  number_208 = number_206 + number_207 →
  (number_206 : ℚ) / (number_207 : ℚ) = 3 / 2 ∧
  (number_208 : ℚ) / (number_207 : ℚ) = 5 / 2

theorem lead_atom_ratio : ratio_of_lead_atoms 207.2 206 207 208 3 2 5 :=
by sorry

end lead_atom_ratio_l13_13808


namespace jaden_time_difference_l13_13401

-- Define the conditions as hypotheses
def jaden_time_as_girl (distance : ℕ) (time : ℕ) : Prop :=
  distance = 20 ∧ time = 240

def jaden_time_as_woman (distance : ℕ) (time : ℕ) : Prop :=
  distance = 8 ∧ time = 240

-- Define the proof problem
theorem jaden_time_difference
  (d_girl t_girl d_woman t_woman : ℕ)
  (H_girl : jaden_time_as_girl d_girl t_girl)
  (H_woman : jaden_time_as_woman d_woman t_woman)
  : (t_woman / d_woman) - (t_girl / d_girl) = 18 :=
by
  sorry

end jaden_time_difference_l13_13401


namespace AM_QM_Muirhead_Inequality_l13_13045

open Real

theorem AM_QM_Muirhead_Inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  ((a + b + c) / 3 = sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c) ∧
  (sqrt ((a^2 + b^2 + c^2) / 3) = ((ab / c) + (bc / a) + (ca / b)) / 3 ↔ a = b ∧ b = c) :=
by sorry

end AM_QM_Muirhead_Inequality_l13_13045


namespace min_value_ineq_l13_13614

noncomputable def minimum_value (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) : ℝ :=
  1 / a + 4 / b

theorem min_value_ineq (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) :
  minimum_value a b ha hb h ≥ 16 :=
sorry

end min_value_ineq_l13_13614


namespace number_of_classes_l13_13773

-- Define the conditions
def first_term : ℕ := 27
def common_diff : ℤ := -2
def total_students : ℕ := 115

-- Define and prove the main statement
theorem number_of_classes : ∃ n : ℕ, n > 0 ∧ (first_term + (n - 1) * common_diff) * n / 2 = total_students ∧ n = 5 :=
by
  sorry

end number_of_classes_l13_13773


namespace abs_sum_inequality_solution_l13_13682

theorem abs_sum_inequality_solution (x : ℝ) : 
  (|x - 5| + |x + 1| < 8) ↔ (-2 < x ∧ x < 6) :=
sorry

end abs_sum_inequality_solution_l13_13682


namespace yellow_more_than_purple_l13_13094
-- Import math library for necessary definitions.

-- Define the problem conditions in Lean
def num_purple_candies : ℕ := 10
def num_total_candies : ℕ := 36

axiom exists_yellow_and_green_candies 
  (Y G : ℕ) 
  (h1 : G = Y - 2) 
  (h2 : 10 + Y + G = 36) : True

-- The theorem to prove
theorem yellow_more_than_purple 
  (Y : ℕ) 
  (hY : exists (G : ℕ), G = Y - 2 ∧ 10 + Y + G = 36) : Y - num_purple_candies = 4 :=
by {
  sorry -- proof is not required
}

end yellow_more_than_purple_l13_13094


namespace fans_attended_show_l13_13531

-- Definitions from the conditions
def total_seats : ℕ := 60000
def sold_percentage : ℝ := 0.75
def fans_stayed_home : ℕ := 5000

-- The proof statement
theorem fans_attended_show :
  let sold_seats := sold_percentage * total_seats
  let fans_attended := sold_seats - fans_stayed_home
  fans_attended = 40000 :=
by
  -- Auto-generated proof placeholder.
  sorry

end fans_attended_show_l13_13531


namespace total_distance_traveled_l13_13842

theorem total_distance_traveled :
  let car_speed1 := 90
  let car_time1 := 2
  let car_speed2 := 60
  let car_time2 := 1
  let train_speed := 100
  let train_time := 2.5
  let distance_car1 := car_speed1 * car_time1
  let distance_car2 := car_speed2 * car_time2
  let distance_train := train_speed * train_time
  distance_car1 + distance_car2 + distance_train = 490 := by
  sorry

end total_distance_traveled_l13_13842


namespace power_decomposition_l13_13875

theorem power_decomposition (n m : ℕ) (h1 : n ≥ 2) 
  (h2 : n * n = 1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19) 
  (h3 : Nat.succ 19 = 21) 
  : m + n = 15 := sorry

end power_decomposition_l13_13875


namespace distance_down_correct_l13_13405

-- Conditions
def rate_up : ℕ := 5  -- rate on the way up (miles per day)
def time_up : ℕ := 2  -- time to travel up (days)
def rate_factor : ℕ := 3 / 2  -- factor for the rate on the way down
def time_down := time_up  -- time to travel down is the same

-- Formula for computation
def distance_up : ℕ := rate_up * time_up
def rate_down : ℕ := rate_up * rate_factor
def distance_down : ℕ := rate_down * time_down

-- Theorem to be proved
theorem distance_down_correct : distance_down = 15 := by
  sorry

end distance_down_correct_l13_13405


namespace friends_share_difference_l13_13202

-- Define the initial conditions
def gift_cost : ℕ := 120
def initial_friends : ℕ := 10
def remaining_friends : ℕ := 6

-- Define the initial and new shares
def initial_share : ℕ := gift_cost / initial_friends
def new_share : ℕ := gift_cost / remaining_friends

-- Define the difference between the new share and the initial share
def share_difference : ℕ := new_share - initial_share

-- The theorem to be proved
theorem friends_share_difference : share_difference = 8 :=
by
  sorry

end friends_share_difference_l13_13202


namespace infinite_solutions_l13_13385

theorem infinite_solutions (x y : ℕ) (h : x ≥ 1 ∧ y ≥ 1) : ∃ (x y : ℕ), x^2 + y^2 = x^3 :=
by {
  sorry 
}

end infinite_solutions_l13_13385


namespace minimize_cost_per_km_l13_13825

section ship_cost_minimization

variables (u v k : ℝ) (fuel_cost other_cost total_cost_per_km: ℝ)

-- Condition 1: The fuel cost per unit time is directly proportional to the cube of its speed.
def fuel_cost_eq : Prop := u = k * v^3

-- Condition 2: When the speed of the ship is 10 km/h, the fuel cost is 35 yuan per hour.
def fuel_cost_at_10 : Prop := u = 35 ∧ v = 10

-- Condition 3: The other costs are 560 yuan per hour.
def other_cost_eq : Prop := other_cost = 560

-- Condition 4: The maximum speed of the ship is 25 km/h.
def max_speed : Prop := v ≤ 25

-- Prove that the speed of the ship that minimizes the cost per kilometer is 20 km/h.
theorem minimize_cost_per_km : 
  fuel_cost_eq u v k ∧ fuel_cost_at_10 u v ∧ other_cost_eq other_cost ∧ max_speed v → v = 20 :=
by
  sorry

end ship_cost_minimization

end minimize_cost_per_km_l13_13825


namespace diamond_3_7_l13_13581

def star (a b : ℕ) : ℕ := a^2 + 2*a*b + b^2
def diamond (a b : ℕ) : ℕ := star a b - a * b

theorem diamond_3_7 : diamond 3 7 = 79 :=
by 
  sorry

end diamond_3_7_l13_13581


namespace smallest_side_of_triangle_l13_13516

theorem smallest_side_of_triangle (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) : 
  a > c ∧ b > c :=
by
  sorry

end smallest_side_of_triangle_l13_13516


namespace arith_seq_a12_value_l13_13178

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ (a₄ : ℝ), a 4 = 1 ∧ a 7 = a 4 + 3 * d ∧ a 9 = a 4 + 5 * d

theorem arith_seq_a12_value
  (h₁ : arithmetic_sequence a (13 / 8))
  (h₂ : a 7 + a 9 = 15)
  (h₃ : a 4 = 1) :
  a 12 = 14 :=
sorry

end arith_seq_a12_value_l13_13178


namespace sequence_nth_term_l13_13750

theorem sequence_nth_term (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * (a n) + 3) : a 11 = 2^11 - 3 := 
sorry

end sequence_nth_term_l13_13750


namespace mn_sum_eq_neg_one_l13_13325

theorem mn_sum_eq_neg_one (m n : ℤ) (h : (∀ x : ℤ, (x + 2) * (x - 1) = x^2 + m * x + n)) :
  m + n = -1 :=
sorry

end mn_sum_eq_neg_one_l13_13325


namespace smallest_coterminal_angle_pos_radians_l13_13814

theorem smallest_coterminal_angle_pos_radians :
  ∀ (θ : ℝ), θ = -560 * (π / 180) → ∃ α : ℝ, α > 0 ∧ α = (8 * π) / 9 ∧ (∃ k : ℤ, θ + 2 * k * π = α) :=
by
  sorry

end smallest_coterminal_angle_pos_radians_l13_13814


namespace total_heads_l13_13102

theorem total_heads (D P : ℕ) (h1 : D = 9) (h2 : 4 * D + 2 * P = 42) : D + P = 12 :=
by
  sorry

end total_heads_l13_13102


namespace tetrahedron_edge_assignment_possible_l13_13518

theorem tetrahedron_edge_assignment_possible 
(s S a b : ℝ) 
(hs : s ≥ 0) (hS : S ≥ 0) (ha : a ≥ 0) (hb : b ≥ 0) :
  ∃ (e₁ e₂ e₃ e₄ e₅ e₆ : ℝ),
    e₁ ≥ 0 ∧ e₂ ≥ 0 ∧ e₃ ≥ 0 ∧ e₄ ≥ 0 ∧ e₅ ≥ 0 ∧ e₆ ≥ 0 ∧
    (e₁ + e₂ + e₃ = s) ∧ (e₁ + e₄ + e₅ = S) ∧
    (e₂ + e₄ + e₆ = a) ∧ (e₃ + e₅ + e₆ = b) := by
  sorry

end tetrahedron_edge_assignment_possible_l13_13518


namespace permutation_6_2_eq_30_l13_13276

theorem permutation_6_2_eq_30 :
  (Nat.factorial 6) / (Nat.factorial (6 - 2)) = 30 :=
by
  sorry

end permutation_6_2_eq_30_l13_13276


namespace solve_for_x_l13_13372

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 9) * x = 14) : x = 220.5 :=
by
  sorry

end solve_for_x_l13_13372


namespace equation_of_symmetric_line_l13_13640

theorem equation_of_symmetric_line
  (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  (∀ x : ℝ, ∃ y : ℝ, (x = a * y + b)) → (∀ x : ℝ, ∃ y : ℝ, (y = (1/a) * x - (b/a))) :=
by
  sorry

end equation_of_symmetric_line_l13_13640


namespace tangent_circumcircle_l13_13409

open EuclideanGeometry

variables {A B C P Q O A' S : Point}

-- Define conditions given in the problem
def conditions (hABC : Triangle A B C) (hP : P ∈ segment A B) (hQ : Q ∈ segment A C)
  (hPQ_parallel_BC : parallel PQ BC) (hBQ_CP_intersect_O : intersects BQ CP O)
  (hA'_symmetric_A : symmetric_with_respect_to A' A BC)
  (hAO'_intersection_S : intersects A' O ((circumcircle A P Q) S)) : Prop := sorry

-- Define the statement that needs to be proved
theorem tangent_circumcircle (hABC : Triangle A B C) (hP : P ∈ segment A B) (hQ : Q ∈ segment A C)
  (hPQ_parallel_BC : parallel PQ BC) (hBQ_CP_intersect_O : intersects BQ CP O)
  (hA'_symmetric_A : symmetric_with_respect_to A' A BC)
  (hAO'_intersection_S : intersects A' O ((circumcircle A P Q) S)) :
  tangential_circles (circumcircle B S C) (circumcircle A P Q) := 
sorry

end tangent_circumcircle_l13_13409


namespace calculate_result_l13_13286

def multiply (a b : ℕ) : ℕ := a * b
def subtract (a b : ℕ) : ℕ := a - b
def three_fifths (a : ℕ) : ℕ := 3 * a / 5

theorem calculate_result :
  let result := three_fifths (subtract (multiply 12 10) 20)
  result = 60 :=
by
  sorry

end calculate_result_l13_13286


namespace trapezoidal_field_perimeter_l13_13427

-- Definitions derived from the conditions
def length_of_longer_parallel_side : ℕ := 15
def length_of_shorter_parallel_side : ℕ := 9
def total_perimeter_of_rectangle : ℕ := 52

-- Correct Answer
def correct_perimeter_of_trapezoidal_field : ℕ := 46

-- Theorem statement
theorem trapezoidal_field_perimeter 
  (a b w : ℕ)
  (h1 : a = length_of_longer_parallel_side)
  (h2 : b = length_of_shorter_parallel_side)
  (h3 : 2 * (a + w) = total_perimeter_of_rectangle)
  (h4 : w = 11) -- from the solution calculation
  : a + b + 2 * w = correct_perimeter_of_trapezoidal_field :=
by
  sorry

end trapezoidal_field_perimeter_l13_13427


namespace not_perfect_squares_l13_13086

-- Definitions of the numbers as per conditions
def n1 : ℕ := 6^3032
def n2 : ℕ := 7^3033
def n3 : ℕ := 8^3034
def n4 : ℕ := 9^3035
def n5 : ℕ := 10^3036

-- Proof statement asserting which of these numbers are not perfect squares.
theorem not_perfect_squares : ¬ (∃ x : ℕ, x^2 = n2) ∧ ¬ (∃ x : ℕ, x^2 = n4) :=
by
  -- Proof is omitted
  sorry

end not_perfect_squares_l13_13086


namespace add_hex_numbers_l13_13278

theorem add_hex_numbers : (7 * 16^2 + 10 * 16^1 + 3) + (1 * 16^2 + 15 * 16^1 + 4) = 9 * 16^2 + 9 * 16^1 + 7 := by sorry

end add_hex_numbers_l13_13278


namespace find_fraction_l13_13191

def f (x : ℕ) : ℕ := 3 * x + 2
def g (x : ℕ) : ℕ := 2 * x - 3

theorem find_fraction : (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := 
by 
  sorry

end find_fraction_l13_13191


namespace tetrahedron_edge_assignment_possible_l13_13517

theorem tetrahedron_edge_assignment_possible 
(s S a b : ℝ) 
(hs : s ≥ 0) (hS : S ≥ 0) (ha : a ≥ 0) (hb : b ≥ 0) :
  ∃ (e₁ e₂ e₃ e₄ e₅ e₆ : ℝ),
    e₁ ≥ 0 ∧ e₂ ≥ 0 ∧ e₃ ≥ 0 ∧ e₄ ≥ 0 ∧ e₅ ≥ 0 ∧ e₆ ≥ 0 ∧
    (e₁ + e₂ + e₃ = s) ∧ (e₁ + e₄ + e₅ = S) ∧
    (e₂ + e₄ + e₆ = a) ∧ (e₃ + e₅ + e₆ = b) := by
  sorry

end tetrahedron_edge_assignment_possible_l13_13517


namespace numberOfZeros_l13_13453

noncomputable def g (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem numberOfZeros :
  ∃ x ∈ Set.Ioo 1 (Real.exp Real.pi), g x = 0 ∧ ∀ y ∈ Set.Ioo 1 (Real.exp Real.pi), g y = 0 → y = x := 
sorry

end numberOfZeros_l13_13453


namespace solve_system_addition_l13_13874

theorem solve_system_addition (a b : ℝ) (h1 : 3 * a + 7 * b = 1977) (h2 : 5 * a + b = 2007) : a + b = 498 :=
by
  sorry

end solve_system_addition_l13_13874


namespace base_for_784_as_CDEC_l13_13542

theorem base_for_784_as_CDEC : 
  ∃ (b : ℕ), 
  (b^3 ≤ 784 ∧ 784 < b^4) ∧ 
  (∃ C D : ℕ, C ≠ D ∧ 784 = (C * b^3 + D * b^2 + C * b + C) ∧ 
  b = 6) :=
sorry

end base_for_784_as_CDEC_l13_13542


namespace negation_of_forall_ge_zero_l13_13527

theorem negation_of_forall_ge_zero :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
sorry

end negation_of_forall_ge_zero_l13_13527


namespace probability_queen_and_spade_l13_13218

def standard_deck : Finset (ℕ × Suit) := 
  Finset.range 52

inductive Card
| queen : Suit → Card
| other : ℕ → Suit → Card

inductive Suit
| hearts
| diamonds
| clubs
| spades

open Card Suit

def count_queens (deck : Finset (Card)) : ℕ :=
  deck.count (λ c => match c with
                    | queen _ => true
                    | _ => false)

def count_spades (deck : Finset (Card)) : ℕ :=
  deck.count (λ c => match c with
                    | queen spades => true
                    | other _ spades => true
                    | _ => false)

theorem probability_queen_and_spade
  (h_deck : ∀ c ∈ standard_deck, c = queen hearts ∨ c = queen diamonds ∨ c = queen clubs ∨ c = queen spades
  ∨ c = other 1 hearts ∨ c = other 1 diamonds ∨ c = other 1 clubs ∨ c = other 1 spades
  ∨ ... (other combinations for cards))
  (h_queens : count_queens standard_deck = 4)
  (h_spades : count_spades standard_deck = 13) :
  sorry : ℚ :=
begin
  -- Mathematically prove the probability is 4/17, proof is omitted for now
  sorry
end

end probability_queen_and_spade_l13_13218


namespace find_a_l13_13885

theorem find_a (a : ℝ) : (∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5) → (∃ x : ℝ, x = 3) → a = 8 :=
by
  sorry

end find_a_l13_13885


namespace matrix_rank_at_least_two_l13_13502

open Matrix

-- Define the problem statement
theorem matrix_rank_at_least_two 
  {m n : ℕ} 
  (A : Matrix (Fin m) (Fin n) ℚ)
  (prime_count: ℕ)
  (h1 : ∀ i j, 1 ≤ list.countp (λ p, p.prime) (abs (A i j)))
  (h2 : list.countp (λ p, p.prime) (abs (A i j)) ≥ m + n) :
  rank A ≥ 2 := 
sorry

end matrix_rank_at_least_two_l13_13502


namespace min_operations_to_reach_goal_l13_13980

-- Define the initial and final configuration of the letters
structure Configuration where
  A : Char := 'A'
  B : Char := 'B'
  C : Char := 'C'
  D : Char := 'D'
  E : Char := 'E'
  F : Char := 'F'
  G : Char := 'G'

-- Define a valid rotation operation
inductive Rotation
| rotate_ABC : Rotation
| rotate_ABD : Rotation
| rotate_DEF : Rotation
| rotate_EFC : Rotation

-- Function representing a single rotation
def applyRotation : Configuration -> Rotation -> Configuration
| config, Rotation.rotate_ABC => 
  { A := config.C, B := config.A, C := config.B, D := config.D, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_ABD => 
  { A := config.B, B := config.D, D := config.A, C := config.C, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_DEF => 
  { D := config.E, E := config.F, F := config.D, A := config.A, B := config.B, C := config.C, G := config.G }
| config, Rotation.rotate_EFC => 
  { E := config.F, F := config.C, C := config.E, A := config.A, B := config.B, D := config.D, G := config.G }

-- Define the goal configuration
def goalConfiguration : Configuration := 
  { A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G' }

-- Function to apply multiple rotations
def applyRotations (config : Configuration) (rotations : List Rotation) : Configuration :=
  rotations.foldl applyRotation config

-- Main theorem statement 
theorem min_operations_to_reach_goal : 
  ∃ rotations : List Rotation, rotations.length = 3 ∧ applyRotations {A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G'} rotations = goalConfiguration :=
sorry

end min_operations_to_reach_goal_l13_13980


namespace largest_integer_n_neg_quad_expr_l13_13738

theorem largest_integer_n_neg_quad_expr :
  ∃ n : ℤ, n = 6 ∧ ∀ m : ℤ, ((n^2 - 11 * n + 28 < 0) → (m < 7 ∧ m > 4) → m ≤ n) :=
by
  sorry

end largest_integer_n_neg_quad_expr_l13_13738


namespace points_calculation_l13_13243

def points_per_enemy : ℕ := 9
def total_enemies : ℕ := 11
def enemies_destroyed : ℕ := total_enemies - 3
def total_points_earned : ℕ := enemies_destroyed * points_per_enemy

theorem points_calculation :
  total_points_earned = 72 := by
  sorry

end points_calculation_l13_13243


namespace geom_sequence_a1_l13_13756

noncomputable def a_n (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n-1)

theorem geom_sequence_a1 {a1 q : ℝ} 
  (h1 : 0 < q)
  (h2 : a_n a1 q 4 * a_n a1 q 8 = 2 * (a_n a1 q 5)^2)
  (h3 : a_n a1 q 2 = 1) :
  a1 = (Real.sqrt 2) / 2 :=
sorry

end geom_sequence_a1_l13_13756


namespace arcsin_neg_one_eq_neg_pi_div_two_l13_13557

theorem arcsin_neg_one_eq_neg_pi_div_two : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_neg_one_eq_neg_pi_div_two_l13_13557


namespace expected_difference_after_10_days_l13_13945

-- Define the initial state and transitions
noncomputable def initial_prob (k : ℤ) : ℝ :=
if k = 0 then 1 else 0

noncomputable def transition_prob (k : ℤ) (n : ℕ) : ℝ :=
0.5 * initial_prob k +
0.25 * initial_prob (k - 1) +
0.25 * initial_prob (k + 1)

-- Define event probability for having any wealth difference after n days
noncomputable def p_k_n (k : ℤ) (n : ℕ) : ℝ :=
if n = 0 then initial_prob k
else transition_prob k (n - 1)

-- Use expected value of absolute difference between wealths 
noncomputable def expected_value_abs_diff (n : ℕ) : ℝ :=
Σ' k, |k| * p_k_n k n

-- Finally, state the theorem
theorem expected_difference_after_10_days :
expected_value_abs_diff 10 = 1 :=
by
  sorry

end expected_difference_after_10_days_l13_13945


namespace negation_of_p_l13_13751

   -- Define the proposition p as an existential quantification
   def p : Prop := ∃ x₀ : ℝ, x₀^2 + 2 * x₀ + 3 > 0

   -- State the theorem that negation of p is a universal quantification
   theorem negation_of_p : ¬ p ↔ ∀ x : ℝ, x^2 + 2*x + 3 ≤ 0 :=
   by sorry
   
end negation_of_p_l13_13751


namespace hibiscus_flower_ratio_l13_13511

theorem hibiscus_flower_ratio (x : ℕ) 
  (h1 : 2 + x + 4 * x = 22) : x / 2 = 2 := 
sorry

end hibiscus_flower_ratio_l13_13511


namespace simplify_and_evaluate_expression_l13_13043

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 + Real.sqrt 3) :
  ((x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l13_13043


namespace fixed_point_for_any_k_l13_13303

-- Define the function f representing our quadratic equation
def f (k : ℝ) (x : ℝ) : ℝ :=
  8 * x^2 + 3 * k * x - 5 * k
  
-- The statement representing our proof problem
theorem fixed_point_for_any_k :
  ∀ (a b : ℝ), (∀ (k : ℝ), f k a = b) → (a, b) = (5, 200) :=
by
  sorry

end fixed_point_for_any_k_l13_13303


namespace parametric_plane_equation_l13_13103

-- Definitions to translate conditions
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ := (2 + 2 * s - t, 4 - 2 * s, 6 + s - 3 * t)

-- Theorem to prove the equivalence to plane equation
theorem parametric_plane_equation : 
  ∃ A B C D, A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧ 
  (∀ s t x y z, parametric_plane s t = (x, y, z) → 6 * x - 5 * y - 2 * z + 20 = 0) := by
  sorry

end parametric_plane_equation_l13_13103


namespace find_positive_integers_n_l13_13290

open Real Int

noncomputable def satisfies_conditions (x y z : ℝ) (n : ℕ) : Prop :=
  sqrt x + sqrt y + sqrt z = 1 ∧ 
  (∃ m : ℤ, sqrt (x + n) + sqrt (y + n) + sqrt (z + n) = m)

theorem find_positive_integers_n (n : ℕ) :
  (∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ satisfies_conditions x y z n) ↔
  (∃ k : ℤ, k ≥ 1 ∧ (k % 9 = 1 ∨ k % 9 = 8) ∧ n = (k^2 - 1) / 9) :=
by
  sorry

end find_positive_integers_n_l13_13290


namespace right_triangle_ratio_is_4_l13_13846

noncomputable def right_triangle_rectangle_ratio (b h xy : ℝ) : Prop :=
  (0.4 * (1/2) * b * h = 0.25 * xy) ∧ (xy = b * h) → (b / h = 4)

theorem right_triangle_ratio_is_4 (b h xy : ℝ) (h1 : 0.4 * (1/2) * b * h = 0.25 * xy)
(h2 : xy = b * h) : b / h = 4 :=
sorry

end right_triangle_ratio_is_4_l13_13846


namespace edwards_final_money_l13_13732

def small_lawn_rate : ℕ := 8
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def first_garden_rate : ℕ := 10
def second_garden_rate : ℕ := 12
def additional_garden_rate : ℕ := 15

def num_small_lawns : ℕ := 3
def num_medium_lawns : ℕ := 1
def num_large_lawns : ℕ := 1
def num_gardens_cleaned : ℕ := 5

def fuel_expense : ℕ := 10
def equipment_rental_expense : ℕ := 15
def initial_savings : ℕ := 7

theorem edwards_final_money : 
  (num_small_lawns * small_lawn_rate + 
   num_medium_lawns * medium_lawn_rate + 
   num_large_lawns * large_lawn_rate + 
   (first_garden_rate + second_garden_rate + (num_gardens_cleaned - 2) * additional_garden_rate) + 
   initial_savings - 
   (fuel_expense + equipment_rental_expense)) = 100 := 
  by 
  -- The proof goes here
  sorry

end edwards_final_money_l13_13732


namespace range_of_a_l13_13743

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x - 2 * a) * (a * x - 1) < 0 → (x > 1 / a ∨ x < 2 * a)) → (a ≤ -Real.sqrt 2 / 2) :=
by
  intro h
  sorry

end range_of_a_l13_13743


namespace count_triangles_in_figure_l13_13484

def rectangle_sim (r w l : ℕ) : Prop := 
  (number_of_small_right_triangles r w l = 24) ∧
  (number_of_isosceles_triangles r w l = 6) ∧
  (number_of_half_length_isosceles_triangles r w l = 8) ∧
  (number_of_large_right_triangles r w l = 12) ∧
  (number_of_full_width_isosceles_triangles r w l = 3)

theorem count_triangles_in_figure (r w l : ℕ) (H : rectangle_sim r w l) : 
  total_number_of_triangles r w l = 53 :=
sorry

end count_triangles_in_figure_l13_13484


namespace unique_solution_for_digits_l13_13225

theorem unique_solution_for_digits :
  ∃ (A B C D E : ℕ),
  (A < B ∧ B < C ∧ C < D ∧ D < E) ∧
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
   B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
   C ≠ D ∧ C ≠ E ∧
   D ≠ E) ∧
  (10 * A + B) * C = 10 * D + E ∧
  (A = 1 ∧ B = 3 ∧ C = 6 ∧ D = 7 ∧ E = 8) :=
sorry

end unique_solution_for_digits_l13_13225


namespace tetrahedron_min_green_edges_l13_13292

theorem tetrahedron_min_green_edges : 
  ∃ (green_edges : Finset (Fin 6)), 
  (∀ face : Finset (Fin 6), face.card = 3 → ∃ edge ∈ face, edge ∈ green_edges) ∧ green_edges.card = 3 :=
by sorry

end tetrahedron_min_green_edges_l13_13292


namespace flowers_given_l13_13644

theorem flowers_given (initial_flowers total_flowers flowers_given : ℕ) 
  (h1 : initial_flowers = 67) 
  (h2 : total_flowers = 90) 
  (h3 : total_flowers = initial_flowers + flowers_given) : 
  flowers_given = 23 :=
by {
  sorry
}

end flowers_given_l13_13644


namespace complement_U_A_is_singleton_one_l13_13307

-- Define the universe and subset
def U : Set ℝ := Set.Icc 0 1
def A : Set ℝ := Set.Ico 0 1

-- Define the complement of A relative to U
def complement_U_A : Set ℝ := U \ A

-- Theorem statement
theorem complement_U_A_is_singleton_one : complement_U_A = {1} := by
  sorry

end complement_U_A_is_singleton_one_l13_13307


namespace total_hours_before_midterms_l13_13975

-- Define the hours spent on each activity per week
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3

-- Sum up the total hours spent on extracurriculars per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Define semester information
def total_weeks_per_semester : ℕ := 12
def weeks_before_midterms : ℕ := total_weeks_per_semester / 2
def weeks_sick : ℕ := 2
def active_weeks_before_midterms : ℕ := weeks_before_midterms - weeks_sick

-- Define the theorem statement about total hours before midterms
theorem total_hours_before_midterms : total_hours_per_week * active_weeks_before_midterms = 52 := by
  -- We skip the actual proof here
  sorry

end total_hours_before_midterms_l13_13975


namespace popsicle_sticks_l13_13142

theorem popsicle_sticks (total_sticks : ℕ) (gino_sticks : ℕ) (my_sticks : ℕ) 
  (h1 : total_sticks = 113) (h2 : gino_sticks = 63) (h3 : total_sticks = gino_sticks + my_sticks) : 
  my_sticks = 50 :=
  sorry

end popsicle_sticks_l13_13142


namespace evaluate_expression_l13_13866

theorem evaluate_expression : (3^2)^4 * 2^3 = 52488 := by
  sorry

end evaluate_expression_l13_13866


namespace find_N_l13_13763

/--
If 15% of N is 45% of 2003, then N is 6009.
-/
theorem find_N (N : ℕ) (h : 15 / 100 * N = 45 / 100 * 2003) : 
  N = 6009 :=
sorry

end find_N_l13_13763


namespace algebraic_expression_value_l13_13769

theorem algebraic_expression_value (x y : ℝ) (h : 2 * x - y = 2) : 6 * x - 3 * y + 1 = 7 := 
by
  sorry

end algebraic_expression_value_l13_13769


namespace exam_paper_max_marks_l13_13545

/-- A candidate appearing for an examination has to secure 40% marks to pass paper i.
    The candidate secured 40 marks and failed by 20 marks.
    Prove that the maximum mark for paper i is 150. -/
theorem exam_paper_max_marks (p : ℝ) (s f : ℝ) (M : ℝ) (h1 : p = 0.40) (h2 : s = 40) (h3 : f = 20) (h4 : p * M = s + f) :
  M = 150 :=
sorry

end exam_paper_max_marks_l13_13545


namespace no_unhappy_days_l13_13665

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end no_unhappy_days_l13_13665


namespace problem_statement_l13_13958

noncomputable def f (x k : ℝ) : ℝ :=
  (1/5) * (x - k + 4500 / x)

noncomputable def fuel_consumption_100km (x k : ℝ) : ℝ :=
  100 / x * f x k

theorem problem_statement (x k : ℝ)
  (hx1 : 60 ≤ x) (hx2 : x ≤ 120)
  (hk1 : 60 ≤ k) (hk2 : k ≤ 100)
  (H : f 120 k = 11.5) :

  (∀ x, 60 ≤ x ∧ x ≤ 100 → f x k ≤ 9 ∧ 
  (if 75 ≤ k ∧ k ≤ 100 then fuel_consumption_100km (9000 / k) k = 20 - k^2 / 900
   else fuel_consumption_100km 120 k = 105 / 4 - k / 6)) :=
  sorry

end problem_statement_l13_13958


namespace ironing_pants_each_day_l13_13438

-- Given conditions:
def minutes_ironing_shirt := 5 -- minutes per day
def days_per_week := 5 -- days per week
def total_minutes_ironing_4_weeks := 160 -- minutes over 4 weeks

-- Target statement to prove:
theorem ironing_pants_each_day : 
  (total_minutes_ironing_4_weeks / 4 - minutes_ironing_shirt * days_per_week) /
  days_per_week = 3 :=
by 
sorry

end ironing_pants_each_day_l13_13438


namespace not_divisible_l13_13442

theorem not_divisible (n : ℕ) : ¬ ((4^n - 1) ∣ (5^n - 1)) :=
by
  sorry

end not_divisible_l13_13442


namespace total_earnings_l13_13949

theorem total_earnings (x y : ℕ) 
  (h1 : 2 * x * y = 250) : 
  58 * (x * y) = 7250 := 
by
  sorry

end total_earnings_l13_13949
