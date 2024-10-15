import Mathlib

namespace NUMINAMATH_GPT_sum_incorrect_correct_l29_2920

theorem sum_incorrect_correct (x : ℕ) (h : x + 9 = 39) :
  ((x - 5 + 14) + (x * 5 + 14)) = 203 :=
sorry

end NUMINAMATH_GPT_sum_incorrect_correct_l29_2920


namespace NUMINAMATH_GPT_find_initial_nickels_l29_2958

variable (initial_nickels current_nickels borrowed_nickels : ℕ)

def initial_nickels_equation (initial_nickels current_nickels borrowed_nickels : ℕ) : Prop :=
  initial_nickels - borrowed_nickels = current_nickels

theorem find_initial_nickels (h : initial_nickels_equation initial_nickels current_nickels borrowed_nickels) 
                             (h_current : current_nickels = 11) 
                             (h_borrowed : borrowed_nickels = 20) : 
                             initial_nickels = 31 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_nickels_l29_2958


namespace NUMINAMATH_GPT_cos_alpha_plus_pi_div_4_value_l29_2931

noncomputable def cos_alpha_plus_pi_div_4 (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) : Real :=
  Real.cos (α + π / 4)

theorem cos_alpha_plus_pi_div_4_value (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : Real.sin (α - 3 * π / 4) = 3 / 5) :
  cos_alpha_plus_pi_div_4 α h1 h2 = -4 / 5 :=
sorry

end NUMINAMATH_GPT_cos_alpha_plus_pi_div_4_value_l29_2931


namespace NUMINAMATH_GPT_min_value_l29_2991

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ m : ℝ, m = 3 + 2 * Real.sqrt 2 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (1 / x + 2 / y) ≥ m) := 
sorry

end NUMINAMATH_GPT_min_value_l29_2991


namespace NUMINAMATH_GPT_min_sum_of_intercepts_l29_2960

-- Definitions based on conditions
def line (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = a * b
def point_on_line (a b : ℝ) : Prop := line a b 1 1

-- Main theorem statement
theorem min_sum_of_intercepts (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_point : point_on_line a b) : 
  a + b >= 4 :=
sorry

end NUMINAMATH_GPT_min_sum_of_intercepts_l29_2960


namespace NUMINAMATH_GPT_max_value_of_b_l29_2905

theorem max_value_of_b (a b c : ℝ) (q : ℝ) (hq : q ≠ 0) 
  (h_geom : a = b / q ∧ c = b * q) 
  (h_arith : 2 * b + 4 = a + 6 + (b + 2) + (c + 1) - (b + 2)) :
  b ≤ 3 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_b_l29_2905


namespace NUMINAMATH_GPT_kara_forgot_medication_times_l29_2933

theorem kara_forgot_medication_times :
  let ounces_per_medication := 4
  let medication_times_per_day := 3
  let days_per_week := 7
  let total_weeks := 2
  let total_water_intaken := 160
  let expected_total_water := (ounces_per_medication * medication_times_per_day * days_per_week * total_weeks)
  let water_difference := expected_total_water - total_water_intaken
  let forget_times := water_difference / ounces_per_medication
  forget_times = 2 := by sorry

end NUMINAMATH_GPT_kara_forgot_medication_times_l29_2933


namespace NUMINAMATH_GPT_factors_multiple_of_120_l29_2964

theorem factors_multiple_of_120 (n : ℕ) (h : n = 2^12 * 3^15 * 5^9 * 7^5) :
  ∃ k : ℕ, k = 8100 ∧ ∀ d : ℕ, d ∣ n ∧ 120 ∣ d ↔ ∃ a b c d : ℕ, 3 ≤ a ∧ a ≤ 12 ∧ 1 ≤ b ∧ b ≤ 15 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 5 ∧ d = 2^a * 3^b * 5^c * 7^d :=
by
  sorry

end NUMINAMATH_GPT_factors_multiple_of_120_l29_2964


namespace NUMINAMATH_GPT_remainder_of_multiple_of_n_mod_7_l29_2950

theorem remainder_of_multiple_of_n_mod_7
  (n m : ℤ)
  (h1 : n % 7 = 1)
  (h2 : m % 7 = 3) :
  (m * n) % 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_multiple_of_n_mod_7_l29_2950


namespace NUMINAMATH_GPT_part1_part2_l29_2997

-- Statement for part (1)
theorem part1 (m : ℝ) : 
  (∀ x1 x2 : ℝ, (m - 1) * x1^2 + 3 * x1 - 2 = 0 ∧ 
               (m - 1) * x2^2 + 3 * x2 - 2 = 0 ∧ x1 ≠ x2) ↔ m > -1/8 :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 2 = 0 ∧ ∀ y : ℝ, (m - 1) * y^2 + 3 * y - 2 = 0 → y = x) ↔ 
  (m = 1 ∨ m = -1/8) :=
sorry

end NUMINAMATH_GPT_part1_part2_l29_2997


namespace NUMINAMATH_GPT_C_recurrence_S_recurrence_l29_2994

noncomputable def C (x : ℝ) : ℝ := 2 * Real.cos x
noncomputable def C_n (n : ℕ) (x : ℝ) : ℝ := 2 * Real.cos (n * x)
noncomputable def S_n (n : ℕ) (x : ℝ) : ℝ := Real.sin (n * x) / Real.sin x

theorem C_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  C_n n x = C x * C_n (n - 1) x - C_n (n - 2) x := sorry

theorem S_recurrence (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
  S_n n x = C x * S_n (n - 1) x - S_n (n - 2) x := sorry

end NUMINAMATH_GPT_C_recurrence_S_recurrence_l29_2994


namespace NUMINAMATH_GPT_total_trip_time_l29_2928

-- Definitions: conditions from the problem
def time_in_first_country : Nat := 2
def time_in_second_country := 2 * time_in_first_country
def time_in_third_country := 2 * time_in_first_country

-- Statement: prove that the total time spent is 10 weeks
theorem total_trip_time : time_in_first_country + time_in_second_country + time_in_third_country = 10 := by
  sorry

end NUMINAMATH_GPT_total_trip_time_l29_2928


namespace NUMINAMATH_GPT_third_square_length_l29_2940

theorem third_square_length 
  (A1 : 8 * 5 = 40) 
  (A2 : 10 * 7 = 70) 
  (A3 : 15 * 9 = 135) 
  (L : ℕ) 
  (A4 : 40 + 70 + L * 5 = 135) 
  : L = 5 := 
sorry

end NUMINAMATH_GPT_third_square_length_l29_2940


namespace NUMINAMATH_GPT_a_range_condition_l29_2929

theorem a_range_condition (a : ℝ) : 
  (∀ x y : ℝ, ((x + a)^2 + (y - a)^2 < 4) → (x = -1 ∧ y = -1)) → 
  -1 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_a_range_condition_l29_2929


namespace NUMINAMATH_GPT_appliance_costs_l29_2996

theorem appliance_costs (a b : ℕ) 
  (h1 : a + 2 * b = 2300) 
  (h2 : 2 * a + b = 2050) : 
  a = 600 ∧ b = 850 := 
by 
  sorry

end NUMINAMATH_GPT_appliance_costs_l29_2996


namespace NUMINAMATH_GPT_max_cube_sum_l29_2955

theorem max_cube_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 9) : x^3 + y^3 + z^3 ≤ 27 :=
sorry

end NUMINAMATH_GPT_max_cube_sum_l29_2955


namespace NUMINAMATH_GPT_problem_solution_l29_2977

theorem problem_solution (x y : ℕ) (hxy : x + y + x * y = 104) (hx : 0 < x) (hy : 0 < y) (hx30 : x < 30) (hy30 : y < 30) : 
  x + y = 20 := 
sorry

end NUMINAMATH_GPT_problem_solution_l29_2977


namespace NUMINAMATH_GPT_contradiction_method_l29_2923

theorem contradiction_method (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0) → a < 1 :=
sorry

end NUMINAMATH_GPT_contradiction_method_l29_2923


namespace NUMINAMATH_GPT_rachel_total_problems_l29_2954

theorem rachel_total_problems
    (problems_per_minute : ℕ)
    (minutes_before_bed : ℕ)
    (problems_next_day : ℕ) 
    (h1 : problems_per_minute = 5) 
    (h2 : minutes_before_bed = 12) 
    (h3 : problems_next_day = 16) : 
    problems_per_minute * minutes_before_bed + problems_next_day = 76 :=
by
  sorry

end NUMINAMATH_GPT_rachel_total_problems_l29_2954


namespace NUMINAMATH_GPT_range_of_f_l29_2924

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (1 - 2 * x)

theorem range_of_f : ∀ y, (∃ x, x ≤ (1 / 2) ∧ f x = y) ↔ y ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_GPT_range_of_f_l29_2924


namespace NUMINAMATH_GPT_number_of_valid_4_digit_integers_l29_2983

/-- 
Prove that the number of 4-digit positive integers that satisfy the following conditions:
1. Each of the first two digits must be 2, 3, or 5.
2. The last two digits cannot be the same.
3. Each of the last two digits must be 4, 6, or 9.
is equal to 54.
-/
theorem number_of_valid_4_digit_integers : 
  ∃ n : ℕ, n = 54 ∧ 
  ∀ d1 d2 d3 d4 : ℕ, 
    (d1 = 2 ∨ d1 = 3 ∨ d1 = 5) ∧ 
    (d2 = 2 ∨ d2 = 3 ∨ d2 = 5) ∧ 
    (d3 = 4 ∨ d3 = 6 ∨ d3 = 9) ∧ 
    (d4 = 4 ∨ d4 = 6 ∨ d4 = 9) ∧ 
    (d3 ≠ d4) → 
    n = 54 := 
sorry

end NUMINAMATH_GPT_number_of_valid_4_digit_integers_l29_2983


namespace NUMINAMATH_GPT_range_of_a_for_critical_points_l29_2944

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + a * x + 3

theorem range_of_a_for_critical_points : 
  ∀ a : ℝ, (∃ x : ℝ, deriv (f a) x = 0) ↔ (a < 0 ∨ a > 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_critical_points_l29_2944


namespace NUMINAMATH_GPT_find_number_90_l29_2935

theorem find_number_90 {x y : ℝ} (h1 : x = y + 0.11 * y) (h2 : x = 99.9) : y = 90 :=
sorry

end NUMINAMATH_GPT_find_number_90_l29_2935


namespace NUMINAMATH_GPT_number_of_correct_statements_l29_2986

def line : Type := sorry
def plane : Type := sorry
def parallel (x y : line) : Prop := sorry
def perpendicular (x : line) (y : plane) : Prop := sorry
def subset (x : line) (y : plane) : Prop := sorry
def skew (x y : line) : Prop := sorry

variable (m n : line) -- two different lines
variable (alpha beta : plane) -- two different planes

theorem number_of_correct_statements :
  (¬parallel m alpha ∨ subset n alpha ∧ parallel m n) ∧
  (parallel m alpha ∧ perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular n beta) ∧
  (subset m alpha ∧ subset n beta ∧ perpendicular m n) ∧
  (skew m n ∧ subset m alpha ∧ subset n beta ∧ parallel m beta ∧ parallel n alpha) :=
sorry

end NUMINAMATH_GPT_number_of_correct_statements_l29_2986


namespace NUMINAMATH_GPT_age_of_B_l29_2979

variable (a b : ℕ)

-- Conditions
def condition1 := a + 10 = 2 * (b - 10)
def condition2 := a = b + 5

-- The proof goal
theorem age_of_B (h1 : condition1 a b) (h2 : condition2 a b) : b = 35 := by
  sorry

end NUMINAMATH_GPT_age_of_B_l29_2979


namespace NUMINAMATH_GPT_product_of_distances_l29_2993

-- Definitions based on the conditions
def curve (x y : ℝ) : Prop := x * y = 2

-- The theorem to prove
theorem product_of_distances (x y : ℝ) (h : curve x y) : abs x * abs y = 2 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_product_of_distances_l29_2993


namespace NUMINAMATH_GPT_least_integer_greater_than_sqrt_450_l29_2900

theorem least_integer_greater_than_sqrt_450 : ∃ n : ℤ, 21^2 < 450 ∧ 450 < 22^2 ∧ n = 22 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_greater_than_sqrt_450_l29_2900


namespace NUMINAMATH_GPT_question_I_question_II_l29_2987

def f (x a : ℝ) : ℝ := |x - a| + 3 * x

theorem question_I (a : ℝ) (h_pos : a > 0) : 
  (f 1 x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) := by sorry

theorem question_II (a : ℝ) (h_pos : a > 0) : 
  (- (a / 2) = -1) ↔ (a = 2) := by sorry

end NUMINAMATH_GPT_question_I_question_II_l29_2987


namespace NUMINAMATH_GPT_find_number_l29_2995

theorem find_number :
  ∃ (x : ℤ), 
  x * (x + 6) = -8 ∧ 
  x^4 + (x + 6)^4 = 272 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l29_2995


namespace NUMINAMATH_GPT_number_of_girls_is_4_l29_2957

variable (x : ℕ)

def number_of_boys : ℕ := 12

def average_score_boys : ℕ := 84

def average_score_girls : ℕ := 92

def average_score_class : ℕ := 86

theorem number_of_girls_is_4 
  (h : average_score_class = 
    (average_score_boys * number_of_boys + average_score_girls * x) / (number_of_boys + x))
  : x = 4 := 
sorry

end NUMINAMATH_GPT_number_of_girls_is_4_l29_2957


namespace NUMINAMATH_GPT_problem_inequality_l29_2915

theorem problem_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^6 - a^2 + 4) * (b^6 - b^2 + 4) * (c^6 - c^2 + 4) * (d^6 - d^2 + 4) ≥ (a + b + c + d)^4 :=
by
  sorry

end NUMINAMATH_GPT_problem_inequality_l29_2915


namespace NUMINAMATH_GPT_maria_savings_l29_2963

-- Conditions
def sweater_cost : ℕ := 30
def scarf_cost : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def savings : ℕ := 500

-- The proof statement
theorem maria_savings : savings - (num_sweaters * sweater_cost + num_scarves * scarf_cost) = 200 :=
by
  sorry

end NUMINAMATH_GPT_maria_savings_l29_2963


namespace NUMINAMATH_GPT_Jason_attended_36_games_l29_2976

noncomputable def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (percentage_missed : ℕ) : ℕ :=
  let total_planned := planned_this_month + planned_last_month
  let missed_games := (percentage_missed * total_planned) / 100
  total_planned - missed_games

theorem Jason_attended_36_games :
  games_attended 24 36 40 = 36 :=
by
  sorry

end NUMINAMATH_GPT_Jason_attended_36_games_l29_2976


namespace NUMINAMATH_GPT_total_cost_correct_l29_2949

/-- Define the base car rental cost -/
def rental_cost : ℝ := 150

/-- Define cost per mile -/
def cost_per_mile : ℝ := 0.5

/-- Define miles driven on Monday -/
def miles_monday : ℝ := 620

/-- Define miles driven on Thursday -/
def miles_thursday : ℝ := 744

/-- Define the total cost Zach spent -/
def total_cost : ℝ := rental_cost + (miles_monday * cost_per_mile) + (miles_thursday * cost_per_mile)

/-- Prove that the total cost Zach spent is 832 dollars -/
theorem total_cost_correct : total_cost = 832 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l29_2949


namespace NUMINAMATH_GPT_depth_of_lost_ship_l29_2922

theorem depth_of_lost_ship (rate_of_descent : ℕ) (time_taken : ℕ) (h1 : rate_of_descent = 60) (h2 : time_taken = 60) :
  rate_of_descent * time_taken = 3600 :=
by {
  /-
  Proof steps would go here.
  -/
  sorry
}

end NUMINAMATH_GPT_depth_of_lost_ship_l29_2922


namespace NUMINAMATH_GPT_hyperbola_eq_l29_2975

/-- Given a hyperbola with center at the origin, 
    one focus at (-√5, 0), and a point P on the hyperbola such that 
    the midpoint of segment PF₁ has coordinates (0, 2), 
    then the equation of the hyperbola is x² - y²/4 = 1. --/
theorem hyperbola_eq (x y : ℝ) (P F1 : ℝ × ℝ) 
  (hF1 : F1 = (-Real.sqrt 5, 0)) 
  (hMidPoint : (P.1 + -Real.sqrt 5) / 2 = 0 ∧ (P.2 + 0) / 2 = 2) 
  : x^2 - y^2 / 4 = 1 := 
sorry

end NUMINAMATH_GPT_hyperbola_eq_l29_2975


namespace NUMINAMATH_GPT_range_of_b_l29_2925

open Real

theorem range_of_b {b x x1 x2 : ℝ} 
  (h1 : ∀ x : ℝ, x^2 - b * x + 1 > 0 ↔ x < x1 ∨ x > x2)
  (h2 : x1 < 1)
  (h3 : x2 > 1) : 
  b > 2 := sorry

end NUMINAMATH_GPT_range_of_b_l29_2925


namespace NUMINAMATH_GPT_water_formed_from_reaction_l29_2970

-- Definitions
def mol_mass_water : ℝ := 18.015
def water_formed_grams (moles_water : ℝ) : ℝ := moles_water * mol_mass_water

-- Statement
theorem water_formed_from_reaction (moles_water : ℝ) :
  18 = water_formed_grams moles_water :=
by sorry

end NUMINAMATH_GPT_water_formed_from_reaction_l29_2970


namespace NUMINAMATH_GPT_mary_income_more_than_tim_income_l29_2906

variables (J T M : ℝ)
variables (h1 : T = 0.60 * J) (h2 : M = 0.8999999999999999 * J)

theorem mary_income_more_than_tim_income : (M - T) / T * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_mary_income_more_than_tim_income_l29_2906


namespace NUMINAMATH_GPT_survivor_quitting_probability_l29_2951

noncomputable def probability_all_quitters_same_tribe : ℚ :=
  let total_contestants := 20
  let tribe_size := 10
  let total_quitters := 3
  let total_ways := (Nat.choose total_contestants total_quitters)
  let tribe_quitters_ways := (Nat.choose tribe_size total_quitters)
  (tribe_quitters_ways + tribe_quitters_ways) / total_ways

theorem survivor_quitting_probability :
  probability_all_quitters_same_tribe = 4 / 19 :=
by
  sorry

end NUMINAMATH_GPT_survivor_quitting_probability_l29_2951


namespace NUMINAMATH_GPT_calculate_tan_product_l29_2962

theorem calculate_tan_product :
  let A := 30
  let B := 40
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2.9 :=
by
  sorry

end NUMINAMATH_GPT_calculate_tan_product_l29_2962


namespace NUMINAMATH_GPT_cherry_orange_punch_ratio_l29_2937

theorem cherry_orange_punch_ratio 
  (C : ℝ)
  (h_condition1 : 4.5 + C + (C - 1.5) = 21) : 
  C / 4.5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_cherry_orange_punch_ratio_l29_2937


namespace NUMINAMATH_GPT_largest_c_for_range_l29_2918

noncomputable def g (x c : ℝ) : ℝ := x^2 - 6*x + c

theorem largest_c_for_range (c : ℝ) : (∃ x : ℝ, g x c = 2) ↔ c ≤ 11 := 
sorry

end NUMINAMATH_GPT_largest_c_for_range_l29_2918


namespace NUMINAMATH_GPT_find_x_l29_2969

theorem find_x :
  ∃ x : ℝ, (2020 + x)^2 = x^2 ∧ x = -1010 :=
sorry

end NUMINAMATH_GPT_find_x_l29_2969


namespace NUMINAMATH_GPT_range_of_a_l29_2903

theorem range_of_a (a : ℝ) (h₁ : 1/2 ≤ 1) (h₂ : a ≤ a + 1)
    (h_condition : ∀ x:ℝ, (1/2 ≤ x ∧ x ≤ 1) → (a ≤ x ∧ x ≤ a + 1)) :
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l29_2903


namespace NUMINAMATH_GPT_rhombus_perimeter_l29_2938

-- Define the lengths of the diagonals
def d1 : ℝ := 5  -- Length of the first diagonal
def d2 : ℝ := 12 -- Length of the second diagonal

-- Calculate the perimeter and state the theorem
theorem rhombus_perimeter : ((d1 / 2)^2 + (d2 / 2)^2).sqrt * 4 = 26 := by
  -- Sorry is placed here to denote the proof
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_l29_2938


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l29_2946

theorem simplify_sqrt_expression (h : Real.sqrt 3 > 1) :
  Real.sqrt ((1 - Real.sqrt 3) ^ 2) = Real.sqrt 3 - 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l29_2946


namespace NUMINAMATH_GPT_parabola_focus_coordinates_parabola_distance_to_directrix_l29_2945

-- Define constants and variables
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

noncomputable def focus_coordinates : ℝ × ℝ := (1, 0)

noncomputable def point : ℝ × ℝ := (4, 4)

noncomputable def directrix : ℝ := -1

noncomputable def distance_to_directrix : ℝ := 5

-- Proof statements
theorem parabola_focus_coordinates (x y : ℝ) (h : parabola_equation x y) : 
  focus_coordinates = (1, 0) :=
sorry

theorem parabola_distance_to_directrix (p : ℝ × ℝ) (d : ℝ) (h : p = point) (h_line : d = directrix) : 
  distance_to_directrix = 5 :=
  by
    -- Define and use the distance between point and vertical line formula
    sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_parabola_distance_to_directrix_l29_2945


namespace NUMINAMATH_GPT_sum_of_three_consecutive_integers_l29_2934

theorem sum_of_three_consecutive_integers (n m l : ℕ) (h1 : n + 1 = m) (h2 : m + 1 = l) (h3 : l = 13) : n + m + l = 36 := 
by sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_integers_l29_2934


namespace NUMINAMATH_GPT_determine_number_on_reverse_side_l29_2941

variable (n : ℕ) (k : ℕ) (shown_cards : ℕ → Prop)

theorem determine_number_on_reverse_side :
    -- Conditions
    (∀ i, 1 ≤ i ∧ i ≤ n → (shown_cards (i - 1) ↔ shown_cards i)) →
    -- Prove
    (k = 0 ∨ k = n ∨ (1 ≤ k ∧ k < n ∧ (shown_cards (k - 1) ∨ shown_cards (k + 1)))) →
    (∃ j, (j = 1 ∧ k = 0) ∨ (j = n - 1 ∧ k = n) ∨ 
          (j = k - 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k + 1)) ∨ 
          (j = k + 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k - 1))) :=
by
  sorry

end NUMINAMATH_GPT_determine_number_on_reverse_side_l29_2941


namespace NUMINAMATH_GPT_part1_part2_part3_l29_2990

-- Definitions based on conditions
def fractional_eq (x a : ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Part (1): Proof statement for a == -1 if x == 5 is a root
theorem part1 (x : ℝ) (a : ℝ) (h : x = 5) (heq : fractional_eq x a) : a = -1 :=
sorry

-- Part (2): Proof statement for a == 2 if the equation has a double root
theorem part2 (a : ℝ) (h_double_root : ∀ x, fractional_eq x a → x = 0 ∨ x = 2) : a = 2 :=
sorry

-- Part (3): Proof statement for a == -3 or == 2 if the equation has no solution
theorem part3 (a : ℝ) (h_no_solution : ¬∃ x, fractional_eq x a) : a = -3 ∨ a = 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l29_2990


namespace NUMINAMATH_GPT_stanley_total_cost_l29_2981

theorem stanley_total_cost (n_tires : ℕ) (price_per_tire : ℝ) (h_n : n_tires = 4) (h_price : price_per_tire = 60) : n_tires * price_per_tire = 240 := by
  sorry

end NUMINAMATH_GPT_stanley_total_cost_l29_2981


namespace NUMINAMATH_GPT_find_b_l29_2907

theorem find_b (b : ℕ) (h1 : 0 ≤ b) (h2 : b ≤ 20) (h3 : (746392847 - b) % 17 = 0) : b = 16 :=
sorry

end NUMINAMATH_GPT_find_b_l29_2907


namespace NUMINAMATH_GPT_trig_identity_sin_cos_l29_2978

theorem trig_identity_sin_cos
  (a : ℝ)
  (h : Real.sin (Real.pi / 3 - a) = 1 / 3) :
  Real.cos (5 * Real.pi / 6 - a) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_sin_cos_l29_2978


namespace NUMINAMATH_GPT_expr_undefined_iff_l29_2921

theorem expr_undefined_iff (x : ℝ) : (x^2 - 9 = 0) ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_expr_undefined_iff_l29_2921


namespace NUMINAMATH_GPT_six_digit_numbers_with_at_least_one_zero_is_368559_l29_2992

def total_six_digit_numbers : ℕ := 9 * 10 * 10 * 10 * 10 * 10

def six_digit_numbers_no_zero : ℕ := 9 * 9 * 9 * 9 * 9 * 9

def six_digit_numbers_with_at_least_one_zero : ℕ :=
  total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_at_least_one_zero_is_368559 :
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end NUMINAMATH_GPT_six_digit_numbers_with_at_least_one_zero_is_368559_l29_2992


namespace NUMINAMATH_GPT_find_f_neg1_plus_f_7_l29_2908

-- Given a function f : ℝ → ℝ
axiom f : ℝ → ℝ

-- f satisfies the property of an even function
axiom even_f : ∀ x : ℝ, f (-x) = f x

-- f satisfies the periodicity of period 2
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x

-- Also, we are given that f(1) = 1
axiom f_one : f 1 = 1

-- We need to prove that f(-1) + f(7) = 2
theorem find_f_neg1_plus_f_7 : f (-1) + f 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg1_plus_f_7_l29_2908


namespace NUMINAMATH_GPT_min_groups_required_l29_2939

/-!
  Prove that if a coach has 30 athletes and wants to arrange them into equal groups with no more than 12 athletes each, 
  then the minimum number of groups required is 3.
-/

theorem min_groups_required (total_athletes : ℕ) (max_athletes_per_group : ℕ) (h_total : total_athletes = 30) (h_max : max_athletes_per_group = 12) :
  ∃ (min_groups : ℕ), min_groups = total_athletes / 10 ∧ (total_athletes % 10 = 0) := by
  sorry

end NUMINAMATH_GPT_min_groups_required_l29_2939


namespace NUMINAMATH_GPT_determine_f_16_l29_2943

theorem determine_f_16 (a : ℝ) (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  (∀ x, a ^ (x - 4) + 1 = 2) →
  f 4 = 2 →
  f 16 = 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_f_16_l29_2943


namespace NUMINAMATH_GPT_factor_quadratic_polynomial_l29_2926

theorem factor_quadratic_polynomial :
  (∀ x : ℝ, x^4 - 36*x^2 + 25 = (x^2 - 6*x + 5) * (x^2 + 6*x + 5)) :=
by
  sorry

end NUMINAMATH_GPT_factor_quadratic_polynomial_l29_2926


namespace NUMINAMATH_GPT_number_of_cheesecakes_in_fridge_l29_2942

section cheesecake_problem

def cheesecakes_on_display : ℕ := 10
def cheesecakes_sold : ℕ := 7
def cheesecakes_left_to_be_sold : ℕ := 18

def cheesecakes_in_fridge (total_display : ℕ) (sold : ℕ) (left : ℕ) : ℕ :=
  left - (total_display - sold)

theorem number_of_cheesecakes_in_fridge :
  cheesecakes_in_fridge cheesecakes_on_display cheesecakes_sold cheesecakes_left_to_be_sold = 15 :=
by
  sorry

end cheesecake_problem

end NUMINAMATH_GPT_number_of_cheesecakes_in_fridge_l29_2942


namespace NUMINAMATH_GPT_no_valid_pairs_l29_2967

theorem no_valid_pairs : ∀ (a b : ℕ), (a > 0) → (b > 0) → (a ≥ b) → 
  a * b + 125 = 30 * Nat.lcm a b + 24 * Nat.gcd a b + a % b → 
  false := by
  sorry

end NUMINAMATH_GPT_no_valid_pairs_l29_2967


namespace NUMINAMATH_GPT_arrange_6_books_l29_2930

theorem arrange_6_books :
  Nat.factorial 6 = 720 :=
by
  sorry

end NUMINAMATH_GPT_arrange_6_books_l29_2930


namespace NUMINAMATH_GPT_remainder_division_l29_2910

theorem remainder_division (x : ℂ) (β : ℂ) (hβ : β^7 = 1) :
  (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 0 ->
  (x^63 + x^49 + x^35 + x^14 + 1) % (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_remainder_division_l29_2910


namespace NUMINAMATH_GPT_number_of_bricks_needed_l29_2966

theorem number_of_bricks_needed :
  ∀ (brick_length brick_width brick_height wall_length wall_height wall_width : ℝ),
  brick_length = 25 → 
  brick_width = 11.25 → 
  brick_height = 6 → 
  wall_length = 750 → 
  wall_height = 600 → 
  wall_width = 22.5 → 
  (wall_length * wall_height * wall_width) / (brick_length * brick_width * brick_height) = 6000 :=
by
  intros brick_length brick_width brick_height wall_length wall_height wall_width
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end NUMINAMATH_GPT_number_of_bricks_needed_l29_2966


namespace NUMINAMATH_GPT_area_OMVK_l29_2953

theorem area_OMVK :
  ∀ (S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK : ℝ),
    S_OKCL = 6 →
    S_ONAM = 12 →
    S_ONBM = 24 →
    S_ABCD = 4 * (S_OKCL + S_ONAM) →
    S_OMVK = S_ABCD - S_OKCL - S_ONAM - S_ONBM →
    S_OMVK = 30 :=
by
  intros S_OKCL S_ONAM S_ONBM S_ABCD S_OMVK h_OKCL h_ONAM h_ONBM h_ABCD h_OMVK
  rw [h_OKCL, h_ONAM, h_ONBM] at *
  sorry

end NUMINAMATH_GPT_area_OMVK_l29_2953


namespace NUMINAMATH_GPT_max_leap_years_in_200_years_l29_2912

-- Definitions based on conditions
def leap_year_occurrence (years : ℕ) : ℕ :=
  years / 4

-- Define the problem statement based on the given conditions and required proof
theorem max_leap_years_in_200_years : leap_year_occurrence 200 = 50 := 
by
  sorry

end NUMINAMATH_GPT_max_leap_years_in_200_years_l29_2912


namespace NUMINAMATH_GPT_find_number_l29_2917

theorem find_number (n x : ℤ) (h1 : n * x + 3 = 10 * x - 17) (h2 : x = 4) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l29_2917


namespace NUMINAMATH_GPT_no_such_function_l29_2902

theorem no_such_function :
  ¬ ∃ f : ℝ → ℝ, (∀ y x : ℝ, 0 < x → x < y → f y > (y - x) * (f x)^2) :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_l29_2902


namespace NUMINAMATH_GPT_triangle_perimeter_correct_l29_2989

def side_a : ℕ := 15
def side_b : ℕ := 8
def side_c : ℕ := 10
def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter_correct :
  perimeter side_a side_b side_c = 33 := by
sorry

end NUMINAMATH_GPT_triangle_perimeter_correct_l29_2989


namespace NUMINAMATH_GPT_probability_selecting_both_types_X_distribution_correct_E_X_correct_l29_2901

section DragonBoatFestival

/-- The total number of zongzi on the plate -/
def total_zongzi : ℕ := 10

/-- The total number of red bean zongzi -/
def red_bean_zongzi : ℕ := 2

/-- The total number of plain zongzi -/
def plain_zongzi : ℕ := 8

/-- The number of zongzi to select -/
def zongzi_to_select : ℕ := 3

/-- Probability of selecting at least one red bean zongzi and at least one plain zongzi -/
def probability_selecting_both : ℚ := 8 / 15

/-- Distribution of the number of red bean zongzi selected (X) -/
def X_distribution : ℕ → ℚ
| 0 => 7 / 15
| 1 => 7 / 15
| 2 => 1 / 15
| _ => 0

/-- Mathematical expectation of the number of red bean zongzi selected (E(X)) -/
def E_X : ℚ := 3 / 5

/-- Theorem stating the probability of selecting both types of zongzi -/
theorem probability_selecting_both_types :
  let p := probability_selecting_both
  p = 8 / 15 :=
by
  let p := probability_selecting_both
  sorry

/-- Theorem stating the probability distribution of the number of red bean zongzi selected -/
theorem X_distribution_correct :
  (X_distribution 0 = 7 / 15) ∧
  (X_distribution 1 = 7 / 15) ∧
  (X_distribution 2 = 1 / 15) :=
by
  sorry

/-- Theorem stating the mathematical expectation of the number of red bean zongzi selected -/
theorem E_X_correct :
  let E := E_X
  E = 3 / 5 :=
by
  let E := E_X
  sorry

end DragonBoatFestival

end NUMINAMATH_GPT_probability_selecting_both_types_X_distribution_correct_E_X_correct_l29_2901


namespace NUMINAMATH_GPT_find_a_l29_2968

noncomputable def f (x a : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem find_a : (∃ a : ℝ, ((∀ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a ≤ -3) ∧ (∃ x ∈ Set.Icc (-(1/3):ℝ) (1/3), f x a = -3)) ↔ a = Real.sqrt 6 + 2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l29_2968


namespace NUMINAMATH_GPT_min_value_of_expression_l29_2965

noncomputable def min_value_expression (a b c d : ℝ) : ℝ :=
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2

theorem min_value_of_expression (a b c d : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≥ 2) :
  min_value_expression a b c d = 1 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l29_2965


namespace NUMINAMATH_GPT_total_items_washed_l29_2936

def towels := 15
def shirts := 10
def loads := 20

def items_per_load : Nat := towels + shirts
def total_items : Nat := items_per_load * loads

theorem total_items_washed : total_items = 500 :=
by
  rw [total_items, items_per_load]
  -- step expansion:
  -- unfold items_per_load
  -- calc 
  -- 15 + 10 = 25  -- from definition
  -- 25 * 20 = 500  -- from multiplication
  sorry

end NUMINAMATH_GPT_total_items_washed_l29_2936


namespace NUMINAMATH_GPT_probability_of_Z_l29_2909

namespace ProbabilityProof

def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_X_or_Y_or_Z : ℚ := 0.4583333333333333

theorem probability_of_Z :
  ∃ P_Z : ℚ, P_Z = 0.0833333333333333 ∧ 
  P_X_or_Y_or_Z = P_X + P_Y + P_Z :=
by
  sorry

end ProbabilityProof

end NUMINAMATH_GPT_probability_of_Z_l29_2909


namespace NUMINAMATH_GPT_exists_n_divisible_l29_2927

theorem exists_n_divisible (k : ℕ) (m : ℤ) (hk : k > 0) (hm : m % 2 = 1) : 
  ∃ n : ℕ, n > 0 ∧ 2^k ∣ (n^n - m) :=
by
  sorry

end NUMINAMATH_GPT_exists_n_divisible_l29_2927


namespace NUMINAMATH_GPT_min_value_expression_l29_2974

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ (y : ℝ), y = x * sqrt 2 ∧ ∀ (u : ℝ), ∀ (hu : u > 0), 
     sqrt ((x^2 + u^2) * (4 * x^2 + u^2)) / (x * u) ≥ 3 * sqrt 2) := 
sorry

end NUMINAMATH_GPT_min_value_expression_l29_2974


namespace NUMINAMATH_GPT_coffee_tea_overlap_l29_2932

theorem coffee_tea_overlap (c t : ℕ) (h_c : c = 80) (h_t : t = 70) : 
  ∃ (b : ℕ), b = 50 := 
by 
  sorry

end NUMINAMATH_GPT_coffee_tea_overlap_l29_2932


namespace NUMINAMATH_GPT_find_n_if_pow_eqn_l29_2947

theorem find_n_if_pow_eqn (n : ℕ) :
  6 ^ 3 = 9 ^ n → n = 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_n_if_pow_eqn_l29_2947


namespace NUMINAMATH_GPT_find_a_l29_2985

theorem find_a (a : ℝ) (h : 2 * a + 2 * a / 4 = 4) : a = 8 / 5 := sorry

end NUMINAMATH_GPT_find_a_l29_2985


namespace NUMINAMATH_GPT_problem1_problem2_l29_2982

-- Define the first problem: For positive real numbers a and b,
-- with the condition a + b = 2, show that the minimum value of 
-- (1 / (1 + a) + 4 / (1 + b)) is 9/4.
theorem problem1 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 2) :
  1 / (1 + a) + 4 / (1 + b) ≥ 9 / 4 :=
sorry

-- Define the second problem: For any positive real numbers a and b,
-- prove that a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1).
theorem problem2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l29_2982


namespace NUMINAMATH_GPT_unit_digit_14_pow_100_l29_2916

theorem unit_digit_14_pow_100 : (14 ^ 100) % 10 = 6 :=
by
  sorry

end NUMINAMATH_GPT_unit_digit_14_pow_100_l29_2916


namespace NUMINAMATH_GPT_correct_sunset_time_proof_l29_2971

def Time := ℕ × ℕ  -- hours and minutes

def sunrise_time : Time := (7, 12)  -- 7:12 AM
def incorrect_daylight_duration : Time := (11, 15)  -- 11 hours 15 minutes as per newspaper

def add_time (t1 t2 : Time) : Time :=
  let (h1, m1) := t1
  let (h2, m2) := t2
  let minutes := m1 + m2
  let hours := h1 + h2 + minutes / 60
  (hours % 24, minutes % 60)

def correct_sunset_time : Time := (18, 27)  -- 18:27 in 24-hour format equivalent to 6:27 PM in 12-hour format

theorem correct_sunset_time_proof :
  add_time sunrise_time incorrect_daylight_duration = correct_sunset_time :=
by
  -- skipping the detailed proof for now
  sorry

end NUMINAMATH_GPT_correct_sunset_time_proof_l29_2971


namespace NUMINAMATH_GPT_solution_set_of_inequality_l29_2913

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem solution_set_of_inequality :
  { x : ℝ | f (x - 2) + f (x^2 - 4) < 0 } = Set.Ioo (-3 : ℝ) 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l29_2913


namespace NUMINAMATH_GPT_find_k_l29_2919

theorem find_k (x y k : ℤ) (h₁ : x = -1) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) :
  k = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l29_2919


namespace NUMINAMATH_GPT_max_self_intersections_polyline_7_l29_2998

def max_self_intersections (n : ℕ) : ℕ :=
  if h : n > 2 then (n * (n - 3)) / 2 else 0

theorem max_self_intersections_polyline_7 :
  max_self_intersections 7 = 14 := 
sorry

end NUMINAMATH_GPT_max_self_intersections_polyline_7_l29_2998


namespace NUMINAMATH_GPT_pencil_price_l29_2952

variable (P N : ℕ) -- This assumes the price of a pencil (P) and the price of a notebook (N) are natural numbers (non-negative integers).

-- Define the conditions
def conditions : Prop :=
  (P + N = 950) ∧ (N = P + 150)

-- The theorem to prove
theorem pencil_price (h : conditions P N) : P = 400 :=
by
  sorry

end NUMINAMATH_GPT_pencil_price_l29_2952


namespace NUMINAMATH_GPT_race_time_A_l29_2959

noncomputable def time_for_A_to_cover_distance (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) : ℝ :=
  let speed_of_B := distance / time_of_B
  let time_for_B_to_cover_remaining := remaining_distance_for_B / speed_of_B
  time_for_B_to_cover_remaining

theorem race_time_A (distance : ℝ) (time_of_B : ℝ) (remaining_distance_for_B : ℝ) :
  distance = 100 ∧ time_of_B = 25 ∧ remaining_distance_for_B = distance - 20 →
  time_for_A_to_cover_distance distance time_of_B remaining_distance_for_B = 20 :=
by
  intros h
  rcases h with ⟨h_distance, h_time_of_B, h_remaining_distance_for_B⟩
  rw [h_distance, h_time_of_B, h_remaining_distance_for_B]
  sorry

end NUMINAMATH_GPT_race_time_A_l29_2959


namespace NUMINAMATH_GPT_abs_add_gt_abs_sub_l29_2999

variables {a b : ℝ}

theorem abs_add_gt_abs_sub (h : a * b > 0) : |a + b| > |a - b| :=
sorry

end NUMINAMATH_GPT_abs_add_gt_abs_sub_l29_2999


namespace NUMINAMATH_GPT_frequency_of_hits_l29_2984

theorem frequency_of_hits (n m : ℕ) (h_n : n = 20) (h_m : m = 15) : (m / n : ℚ) = 0.75 := by
  sorry

end NUMINAMATH_GPT_frequency_of_hits_l29_2984


namespace NUMINAMATH_GPT_max_value_of_A_l29_2988

theorem max_value_of_A (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / ((a + b + c)^4 - 79 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_of_A_l29_2988


namespace NUMINAMATH_GPT_olivia_insurance_premium_l29_2956

theorem olivia_insurance_premium :
  ∀ (P : ℕ) (base_premium accident_percentage ticket_cost : ℤ) (tickets accidents : ℕ),
    base_premium = 50 →
    accident_percentage = P →
    ticket_cost = 5 →
    tickets = 3 →
    accidents = 1 →
    (base_premium + (accidents * base_premium * P / 100) + (tickets * ticket_cost) = 70) →
    P = 10 :=
by
  intros P base_premium accident_percentage ticket_cost tickets accidents
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_olivia_insurance_premium_l29_2956


namespace NUMINAMATH_GPT_sophie_oranges_per_day_l29_2904

/-- Sophie and Hannah together eat a certain number of fruits in 30 days.
    Given Hannah eats 40 grapes every day, prove that Sophie eats 20 oranges every day. -/
theorem sophie_oranges_per_day (total_fruits : ℕ) (grapes_per_day : ℕ) (days : ℕ)
  (total_days_fruits : total_fruits = 1800) (hannah_grapes : grapes_per_day = 40) (days_count : days = 30) :
  (total_fruits - grapes_per_day * days) / days = 20 :=
by
  sorry

end NUMINAMATH_GPT_sophie_oranges_per_day_l29_2904


namespace NUMINAMATH_GPT_no_integer_solutions_for_sum_of_squares_l29_2972

theorem no_integer_solutions_for_sum_of_squares :
  ∀ a b c : ℤ, a^2 + b^2 + c^2 ≠ 20122012 := 
by sorry

end NUMINAMATH_GPT_no_integer_solutions_for_sum_of_squares_l29_2972


namespace NUMINAMATH_GPT_complement_A_eq_l29_2973

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1}

theorem complement_A_eq :
  U \ A = {0, 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_A_eq_l29_2973


namespace NUMINAMATH_GPT_pentagon_triangle_ratio_l29_2911

theorem pentagon_triangle_ratio (p t s : ℝ) 
  (h₁ : 5 * p = 30) 
  (h₂ : 3 * t = 30)
  (h₃ : 4 * s = 30) : 
  p / t = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_pentagon_triangle_ratio_l29_2911


namespace NUMINAMATH_GPT_dan_money_left_l29_2914

def money_left (initial : ℝ) (candy_bar : ℝ) (chocolate : ℝ) (soda : ℝ) (gum : ℝ) : ℝ :=
  initial - candy_bar - chocolate - soda - gum

theorem dan_money_left :
  money_left 10 2 3 1.5 1.25 = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_dan_money_left_l29_2914


namespace NUMINAMATH_GPT_Gary_final_amount_l29_2948

theorem Gary_final_amount
(initial_amount dollars_snake dollars_hamster dollars_supplies : ℝ)
(h1 : initial_amount = 73.25)
(h2 : dollars_snake = 55.50)
(h3 : dollars_hamster = 25.75)
(h4 : dollars_supplies = 12.40) :
  initial_amount + dollars_snake - dollars_hamster - dollars_supplies = 90.60 :=
by
  sorry

end NUMINAMATH_GPT_Gary_final_amount_l29_2948


namespace NUMINAMATH_GPT_toothpaste_runs_out_in_two_days_l29_2961

noncomputable def toothpaste_capacity := 90
noncomputable def dad_usage_per_brushing := 4
noncomputable def mom_usage_per_brushing := 3
noncomputable def anne_usage_per_brushing := 2
noncomputable def brother_usage_per_brushing := 1
noncomputable def sister_usage_per_brushing := 1

noncomputable def dad_brushes_per_day := 4
noncomputable def mom_brushes_per_day := 4
noncomputable def anne_brushes_per_day := 4
noncomputable def brother_brushes_per_day := 4
noncomputable def sister_brushes_per_day := 2

noncomputable def total_daily_usage :=
  dad_usage_per_brushing * dad_brushes_per_day + 
  mom_usage_per_brushing * mom_brushes_per_day + 
  anne_usage_per_brushing * anne_brushes_per_day + 
  brother_usage_per_brushing * brother_brushes_per_day + 
  sister_usage_per_brushing * sister_brushes_per_day

theorem toothpaste_runs_out_in_two_days :
  toothpaste_capacity / total_daily_usage = 2 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_toothpaste_runs_out_in_two_days_l29_2961


namespace NUMINAMATH_GPT_episodes_per_season_l29_2980

theorem episodes_per_season
  (days_to_watch : ℕ)
  (episodes_per_day : ℕ)
  (seasons : ℕ) :
  days_to_watch = 10 →
  episodes_per_day = 6 →
  seasons = 4 →
  (episodes_per_day * days_to_watch) / seasons = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_episodes_per_season_l29_2980
