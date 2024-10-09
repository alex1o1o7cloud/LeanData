import Mathlib

namespace find_a_max_and_min_values_l877_87733

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + x + a)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + 3*x + a + 1)

theorem find_a (a : ℝ) : (f' a 0) = 2 → a = 1 :=
by {
  -- Proof omitted
  sorry
}

theorem max_and_min_values (a : ℝ) :
  (a = 1) →
  (Real.exp (-2) * (4 - 2 + 1) = (3 / Real.exp 2)) ∧
  (Real.exp (-1) * (1 - 1 + 1) = (1 / Real.exp 1)) ∧
  (Real.exp 2 * (4 + 2 + 1) = (7 * Real.exp 2)) :=
by {
  -- Proof omitted
  sorry
}

end find_a_max_and_min_values_l877_87733


namespace solutions_of_system_l877_87716

theorem solutions_of_system :
  ∀ (x y : ℝ), (x - 2 * y = 1) ∧ (x^3 - 8 * y^3 - 6 * x * y = 1) ↔ y = (x - 1) / 2 :=
by
  -- Since this is a statement-only task, the detailed proof is omitted.
  -- Insert actual proof here.
  sorry

end solutions_of_system_l877_87716


namespace line_through_point_parallel_l877_87711

theorem line_through_point_parallel (p : ℝ × ℝ) (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0)
  (hp : a * p.1 + b * p.2 + c = 0) :
  ∃ k : ℝ, a * p.1 + b * p.2 + k = 0 :=
by
  use - (a * p.1 + b * p.2)
  sorry

end line_through_point_parallel_l877_87711


namespace avg_gpa_8th_graders_l877_87707

theorem avg_gpa_8th_graders :
  ∀ (GPA_6th GPA_8th : ℝ),
    GPA_6th = 93 →
    (∀ GPA_7th : ℝ, GPA_7th = GPA_6th + 2 →
    (GPA_6th + GPA_7th + GPA_8th) / 3 = 93 →
    GPA_8th = 91) :=
by
  intros GPA_6th GPA_8th h1 GPA_7th h2 h3
  sorry

end avg_gpa_8th_graders_l877_87707


namespace bill_harry_combined_l877_87791

-- Definitions based on the given conditions
def sue_nuts := 48
def harry_nuts := 2 * sue_nuts
def bill_nuts := 6 * harry_nuts

-- The theorem we want to prove
theorem bill_harry_combined : bill_nuts + harry_nuts = 672 :=
by
  sorry

end bill_harry_combined_l877_87791


namespace value_of_a_l877_87757

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_a (a : ℝ) (f_symmetric : ∀ x y : ℝ, y = f x ↔ -y = 2^(-x + a)) (sum_f_condition : f (-2) + f (-4) = 1) :
  a = 2 :=
sorry

end value_of_a_l877_87757


namespace average_is_1380_l877_87753

def avg_of_numbers : Prop := 
  (1200 + 1300 + 1400 + 1510 + 1520 + 1530 + 1200) / 7 = 1380

theorem average_is_1380 : avg_of_numbers := by
  sorry

end average_is_1380_l877_87753


namespace max_area_triangle_l877_87789

/-- Given two fixed points A and B on the plane with distance 2 between them, 
and a point P moving such that the ratio of distances |PA| / |PB| = sqrt(2), 
prove that the maximum area of triangle PAB is 2 * sqrt(2). -/
theorem max_area_triangle 
  (A B P : EuclideanSpace ℝ (Fin 2)) 
  (hAB : dist A B = 2)
  (h_ratio : dist P A = Real.sqrt 2 * dist P B)
  (h_non_collinear : ¬ ∃ k : ℝ, ∃ l : ℝ, k ≠ l ∧ A = k • B ∧ P = l • B) 
  : ∃ S_max : ℝ, S_max = 2 * Real.sqrt 2 := 
sorry

end max_area_triangle_l877_87789


namespace total_travel_ways_l877_87792

-- Define the number of car departures
def car_departures : ℕ := 3

-- Define the number of train departures
def train_departures : ℕ := 4

-- Define the number of ship departures
def ship_departures : ℕ := 2

-- The total number of ways to travel from location A to location B
def total_ways : ℕ := car_departures + train_departures + ship_departures

-- The theorem stating the total number of ways to travel given the conditions
theorem total_travel_ways :
  total_ways = 9 :=
by
  -- Proof goes here
  sorry

end total_travel_ways_l877_87792


namespace quadratic_solution_symmetry_l877_87738

variable (a b c n : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : a * (-5)^2 + b * (-5) + c = -2.79)
variable (h₂ : a * 1^2 + b * 1 + c = -2.79)
variable (h₃ : a * 2^2 + b * 2 + c = 0)
variable (h₄ : a * 3^2 + b * 3 + c = n)

theorem quadratic_solution_symmetry :
  (x = 3 ∨ x = -7) ↔ (a * x^2 + b * x + c = n) :=
sorry

end quadratic_solution_symmetry_l877_87738


namespace find_k2_minus_b2_l877_87755

theorem find_k2_minus_b2 (k b : ℝ) (h1 : 3 = k * 1 + b) (h2 : 2 = k * (-1) + b) : k^2 - b^2 = -6 := 
by
  sorry

end find_k2_minus_b2_l877_87755


namespace determine_f_peak_tourism_season_l877_87712

noncomputable def f (n : ℕ) : ℝ := 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300

theorem determine_f :
  (∀ n : ℕ, f n = 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300) ∧
  (f 8 - f 2 = 400) ∧
  (f 2 = 100) :=
sorry

theorem peak_tourism_season (n : ℤ) :
  (6 ≤ n ∧ n ≤ 10) ↔ (200 * Real.cos (((Real.pi / 6) * n) + 2 * Real.pi / 3) + 300 >= 400) :=
sorry

end determine_f_peak_tourism_season_l877_87712


namespace arith_seq_100th_term_l877_87720

noncomputable def arithSeq (a : ℤ) (n : ℕ) : ℤ :=
  a - 1 + (n - 1) * ((a + 1) - (a - 1))

theorem arith_seq_100th_term (a : ℤ) : arithSeq a 100 = 197 := by
  sorry

end arith_seq_100th_term_l877_87720


namespace pages_read_in_a_year_l877_87736

-- Definition of the problem conditions
def novels_per_month := 4
def pages_per_novel := 200
def months_per_year := 12

-- Theorem statement corresponding to the problem
theorem pages_read_in_a_year (h1 : novels_per_month = 4) (h2 : pages_per_novel = 200) (h3 : months_per_year = 12) : 
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by
  sorry

end pages_read_in_a_year_l877_87736


namespace sum_of_first_9000_terms_of_geometric_sequence_l877_87756

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l877_87756


namespace max_sum_arithmetic_prog_l877_87725

theorem max_sum_arithmetic_prog (a d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 3 = 327)
  (h2 : S 57 = 57)
  (hS : ∀ n, S n = (n / 2) * (2 * a + (n - 1) * d)) :
  ∃ max_S : ℝ, max_S = 1653 := by
  sorry

end max_sum_arithmetic_prog_l877_87725


namespace find_a_perpendicular_lines_l877_87794

theorem find_a_perpendicular_lines (a : ℝ) :
    (∀ x y : ℝ, a * x - y + 2 * a = 0 → (2 * a - 1) * x + a * y + a = 0) →
    (a = 0 ∨ a = 1) :=
by
  intro h
  sorry

end find_a_perpendicular_lines_l877_87794


namespace number_of_common_tangents_l877_87710

theorem number_of_common_tangents 
  (circle1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (circle2 : ∀ x y : ℝ, 2 * y^2 - 6 * x - 8 * y + 9 = 0) : 
  ∃ n : ℕ, n = 3 :=
by
  -- Proof is skipped
  sorry

end number_of_common_tangents_l877_87710


namespace closest_years_l877_87761

theorem closest_years (a b c d : ℕ) (h1 : 10 * a + b + 10 * c + d = 10 * b + c) :
  (a = 1 ∧ b = 8 ∧ c = 6 ∧ d = 8) ∨ (a = 2 ∧ b = 3 ∧ c = 0 ∧ d =7) ↔
  ((10 * 1 + 8 + 10 * 6 + 8 = 10 * 8 + 6) ∧ (10 * 2 + 3 + 10 * 0 + 7 = 10 * 3 + 0)) :=
sorry

end closest_years_l877_87761


namespace expenses_opposite_to_income_l877_87739

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end expenses_opposite_to_income_l877_87739


namespace min_p_q_sum_l877_87740

theorem min_p_q_sum (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h : 162 * p = q^3) : p + q = 54 :=
sorry

end min_p_q_sum_l877_87740


namespace find_x_values_l877_87771

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem find_x_values :
  { x : ℕ | combination 10 x = combination 10 (3 * x - 2) } = {1, 3} :=
by
  sorry

end find_x_values_l877_87771


namespace pencil_fraction_white_part_l877_87737

theorem pencil_fraction_white_part
  (L : ℝ )
  (H1 : L = 9.333333333333332)
  (H2 : (1 / 8) * L + (7 / 12 * 7 / 8) * (7 / 8) * L + W * (7 / 8) * L = L) :
  W = 5 / 12 :=
by
  sorry

end pencil_fraction_white_part_l877_87737


namespace largest_possible_number_of_sweets_in_each_tray_l877_87767

-- Define the initial conditions as given in the problem statement
def tim_sweets : ℕ := 36
def peter_sweets : ℕ := 44

-- Define the statement that we want to prove
theorem largest_possible_number_of_sweets_in_each_tray :
  Nat.gcd tim_sweets peter_sweets = 4 :=
by
  sorry

end largest_possible_number_of_sweets_in_each_tray_l877_87767


namespace problem_l877_87735

def a := 1 / 4
def b := 1 / 2
def c := -3 / 4

def a_n (n : ℕ) : ℚ := 2 * n + 1
def S_n (n : ℕ) : ℚ := (n + 2) * n
def f (n : ℕ) : ℚ := 4 * a * n^2 + (4 * a + 2 * b) * n + (a + b + c)

theorem problem : ∀ n : ℕ, f n = S_n n := by
  sorry

end problem_l877_87735


namespace triangle_is_right_l877_87730

theorem triangle_is_right {A B C : ℝ} (h : A + B + C = 180) (h1 : A = B + C) : A = 90 :=
by
  sorry

end triangle_is_right_l877_87730


namespace quadratic_opposite_roots_l877_87796

theorem quadratic_opposite_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 + x2 = 0 ∧ x1 * x2 = k + 1) ↔ k = -2 :=
by
  sorry

end quadratic_opposite_roots_l877_87796


namespace sum_geometric_series_nine_l877_87776

noncomputable def geometric_series_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  S n = a 0 * (1 - a 1 ^ n) / (1 - a 1)

theorem sum_geometric_series_nine
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (S_3 : S 3 = 12)
  (S_6 : S 6 = 60) :
  S 9 = 252 := by
  sorry

end sum_geometric_series_nine_l877_87776


namespace mr_a_net_gain_l877_87709

theorem mr_a_net_gain 
  (initial_value : ℝ)
  (sale_profit_percentage : ℝ)
  (buyback_loss_percentage : ℝ)
  (final_sale_price : ℝ) 
  (buyback_price : ℝ)
  (net_gain : ℝ) :
  initial_value = 12000 →
  sale_profit_percentage = 0.15 →
  buyback_loss_percentage = 0.12 →
  final_sale_price = initial_value * (1 + sale_profit_percentage) →
  buyback_price = final_sale_price * (1 - buyback_loss_percentage) →
  net_gain = final_sale_price - buyback_price →
  net_gain = 1656 :=
by
  sorry

end mr_a_net_gain_l877_87709


namespace train_journey_duration_l877_87728

def battery_lifespan (talk_time standby_time : ℝ) :=
  talk_time <= 6 ∧ standby_time <= 210

def full_battery_usage (total_time : ℝ) :=
  (total_time / 2) / 6 + (total_time / 2) / 210 = 1

theorem train_journey_duration (t : ℝ) (h1 : battery_lifespan (t / 2) (t / 2)) (h2 : full_battery_usage t) :
  t = 35 / 3 :=
sorry

end train_journey_duration_l877_87728


namespace simplified_expression_l877_87763

variable {x y : ℝ}

theorem simplified_expression 
  (P : ℝ := x^2 + y^2) 
  (Q : ℝ := x^2 - y^2) : 
  ( (P + 3 * Q) / (P - Q) - (P - 3 * Q) / (P + Q) ) = (2 * x^4 - y^4) / (x^2 * y^2) := 
  by sorry

end simplified_expression_l877_87763


namespace bus_driver_total_compensation_l877_87778

-- Define the regular rate
def regular_rate : ℝ := 16

-- Define the number of regular hours
def regular_hours : ℕ := 40

-- Define the overtime rate as 75% higher than the regular rate
def overtime_rate : ℝ := regular_rate * 1.75

-- Define the total hours worked in the week
def total_hours_worked : ℕ := 48

-- Calculate the overtime hours
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Calculate the total compensation
def total_compensation : ℝ :=
  (regular_hours * regular_rate) + (overtime_hours * overtime_rate)

-- Theorem to prove that the total compensation is $864
theorem bus_driver_total_compensation : total_compensation = 864 := by
  -- Proof is omitted
  sorry

end bus_driver_total_compensation_l877_87778


namespace num_students_only_math_l877_87713

def oakwood_ninth_grade_problem 
  (total_students: ℕ)
  (students_in_math: ℕ)
  (students_in_foreign_language: ℕ)
  (students_in_science: ℕ)
  (students_in_all_three: ℕ)
  (students_total_from_ie: ℕ) :=
  (total_students = 120) ∧
  (students_in_math = 85) ∧
  (students_in_foreign_language = 65) ∧
  (students_in_science = 75) ∧
  (students_in_all_three = 20) ∧
  total_students = students_in_math + students_in_foreign_language + students_in_science 
  - (students_total_from_ie) + students_in_all_three - (students_in_all_three)

theorem num_students_only_math 
  (total_students: ℕ := 120)
  (students_in_math: ℕ := 85)
  (students_in_foreign_language: ℕ := 65)
  (students_in_science: ℕ := 75)
  (students_in_all_three: ℕ := 20)
  (students_total_from_ie: ℕ := 45) :
  oakwood_ninth_grade_problem total_students students_in_math students_in_foreign_language students_in_science students_in_all_three students_total_from_ie →
  ∃ (students_only_math: ℕ), students_only_math = 75 :=
by
  sorry

end num_students_only_math_l877_87713


namespace find_f_prime_2_l877_87781

theorem find_f_prime_2 (a : ℝ) (f' : ℝ → ℝ) 
    (h1 : f' 1 = -5)
    (h2 : ∀ x, f' x = 3 * a * x^2 + 2 * f' 2 * x) : f' 2 = -4 := by
    sorry

end find_f_prime_2_l877_87781


namespace exponent_sum_l877_87731

variables (a : ℝ) (m n : ℝ)

theorem exponent_sum (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end exponent_sum_l877_87731


namespace find_income_l877_87777

-- Definitions of percentages used in calculations
def rent_percentage : ℝ := 0.15
def education_percentage : ℝ := 0.15
def misc_percentage : ℝ := 0.10
def medical_percentage : ℝ := 0.15

-- Remaining amount after all expenses
def final_amount : ℝ := 5548

-- Income calculation function
def calc_income (X : ℝ) : ℝ :=
  let after_rent := X * (1 - rent_percentage)
  let after_education := after_rent * (1 - education_percentage)
  let after_misc := after_education * (1 - misc_percentage)
  let after_medical := after_misc * (1 - medical_percentage)
  after_medical

-- Theorem statement to prove the woman's income
theorem find_income (X : ℝ) (h : calc_income X = final_amount) : X = 10038.46 := by
  sorry

end find_income_l877_87777


namespace min_plus_max_value_of_x_l877_87750

theorem min_plus_max_value_of_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (10 - Real.sqrt 304) / 6
  let M := (10 + Real.sqrt 304) / 6
  m + M = 10 / 3 := by 
  sorry

end min_plus_max_value_of_x_l877_87750


namespace cone_lateral_surface_area_l877_87785

theorem cone_lateral_surface_area (r h : ℝ) (h_r : r = 3) (h_h : h = 4) : 
  (1/2) * (2 * Real.pi * r) * (Real.sqrt (r ^ 2 + h ^ 2)) = 15 * Real.pi := 
by
  sorry

end cone_lateral_surface_area_l877_87785


namespace expression_value_l877_87749

theorem expression_value :
    (2.502 + 0.064)^2 - ((2.502 - 0.064)^2) / (2.502 * 0.064) = 4.002 :=
by
  -- the proof goes here
  sorry

end expression_value_l877_87749


namespace least_number_subtracted_l877_87748

theorem least_number_subtracted (x : ℤ) (N : ℤ) :
  N = 2590 - x →
  (N % 9 = 6) →
  (N % 11 = 6) →
  (N % 13 = 6) →
  x = 10 :=
by
  sorry

end least_number_subtracted_l877_87748


namespace number_of_sets_X_l877_87714

noncomputable def finite_set_problem (M A B : Finset ℕ) : Prop :=
  (M.card = 10) ∧ 
  (A ⊆ M) ∧ 
  (B ⊆ M) ∧ 
  (A ∩ B = ∅) ∧ 
  (A.card = 2) ∧ 
  (B.card = 3) ∧ 
  (∃ (X : Finset ℕ), X ⊆ M ∧ ¬(A ⊆ X) ∧ ¬(B ⊆ X))

theorem number_of_sets_X (M A B : Finset ℕ) (h : finite_set_problem M A B) : 
  ∃ n : ℕ, n = 672 := 
sorry

end number_of_sets_X_l877_87714


namespace product_segment_doubles_l877_87788

-- Define the problem conditions and proof statement in Lean.
theorem product_segment_doubles
  (a b e : ℝ)
  (d : ℝ := (a * b) / e)
  (e' : ℝ := e / 2)
  (d' : ℝ := (a * b) / e') :
  d' = 2 * d := 
  sorry

end product_segment_doubles_l877_87788


namespace winning_votes_calculation_l877_87774

variables (V : ℚ) (winner_votes : ℚ)

-- Conditions
def percentage_of_votes_of_winner : ℚ := 0.60 * V
def percentage_of_votes_of_loser : ℚ := 0.40 * V
def vote_difference_spec : 0.60 * V - 0.40 * V = 288 := by sorry

-- Theorem to prove
theorem winning_votes_calculation (h1 : winner_votes = 0.60 * V)
  (h2 : 0.60 * V - 0.40 * V = 288) : winner_votes = 864 :=
by
  sorry

end winning_votes_calculation_l877_87774


namespace david_money_left_l877_87751

noncomputable section
open Real

def money_left_after_week (rate_per_hour : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  let total_hours := hours_per_day * days_per_week
  let total_money := total_hours * rate_per_hour
  let money_after_shoes := total_money / 2
  let money_after_mom := (total_money - money_after_shoes) / 2
  total_money - money_after_shoes - money_after_mom

theorem david_money_left :
  money_left_after_week 14 2 7 = 49 := by simp [money_left_after_week]; norm_num

end david_money_left_l877_87751


namespace one_cow_one_bag_l877_87706

theorem one_cow_one_bag {days_per_bag : ℕ} (h : 50 * days_per_bag = 50 * 50) : days_per_bag = 50 :=
by
  sorry

end one_cow_one_bag_l877_87706


namespace Ali_is_8_l877_87784

open Nat

-- Definitions of the variables based on the conditions
def YusafAge (UmarAge : ℕ) : ℕ := UmarAge / 2
def AliAge (YusafAge : ℕ) : ℕ := YusafAge + 3

-- The specific given conditions
def UmarAge : ℕ := 10
def Yusaf : ℕ := YusafAge UmarAge
def Ali : ℕ := AliAge Yusaf

-- The theorem to be proved
theorem Ali_is_8 : Ali = 8 :=
by
  sorry

end Ali_is_8_l877_87784


namespace train_length_l877_87718

theorem train_length (L : ℕ) (V : ℕ) (platform_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
    (h1 : V = L / time_pole) 
    (h2 : V = (L + platform_length) / time_platform) :
    L = 300 := 
by 
  -- The proof can be filled here
  sorry

end train_length_l877_87718


namespace penny_half_dollar_same_probability_l877_87786

def probability_penny_half_dollar_same : ℚ :=
  1 / 2

theorem penny_half_dollar_same_probability :
  probability_penny_half_dollar_same = 1 / 2 :=
by
  sorry

end penny_half_dollar_same_probability_l877_87786


namespace total_get_well_cards_l877_87779

def dozens_to_cards (d : ℕ) : ℕ := d * 12
def hundreds_to_cards (h : ℕ) : ℕ := h * 100

theorem total_get_well_cards 
  (d_hospital : ℕ) (h_hospital : ℕ)
  (d_home : ℕ) (h_home : ℕ) :
  d_hospital = 25 ∧ h_hospital = 7 ∧ d_home = 39 ∧ h_home = 3 →
  (dozens_to_cards d_hospital + hundreds_to_cards h_hospital +
   dozens_to_cards d_home + hundreds_to_cards h_home) = 1768 :=
by
  intros
  sorry

end total_get_well_cards_l877_87779


namespace smallest_integer_proof_l877_87766

theorem smallest_integer_proof :
  ∃ (x : ℤ), x^2 = 3 * x + 75 ∧ ∀ (y : ℤ), y^2 = 3 * y + 75 → x ≤ y := 
  sorry

end smallest_integer_proof_l877_87766


namespace find_constants_l877_87775

variable (x : ℝ)

theorem find_constants 
  (h : ∀ x, (6 * x^2 + 3 * x) / ((x - 4) * (x - 2)^3) = 
  (13.5 / (x - 4)) + (-27 / (x - 2)) + (-15 / (x - 2)^3)) :
  true :=
by {
  sorry
}

end find_constants_l877_87775


namespace cookie_problem_l877_87782

theorem cookie_problem (n : ℕ) (M A : ℕ) 
  (hM : M = n - 7) 
  (hA : A = n - 2) 
  (h_sum : M + A < n) 
  (hM_pos : M ≥ 1) 
  (hA_pos : A ≥ 1) : 
  n = 8 := 
sorry

end cookie_problem_l877_87782


namespace principal_amount_l877_87752

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) : 
  A = 1120 → r = 0.05 → t = 6 → P = 1120 / (1 + 0.05 * 6) :=
by
  intros h1 h2 h3
  sorry

end principal_amount_l877_87752


namespace value_of_m_l877_87732

def f (x m : ℝ) : ℝ := x^2 - 2 * x + m
def g (x m : ℝ) : ℝ := x^2 - 2 * x + 2 * m + 8

theorem value_of_m (m : ℝ) : (3 * f 5 m = g 5 m) → m = -22 :=
by
  intro h
  sorry

end value_of_m_l877_87732


namespace smallest_perfect_square_5336100_l877_87741

def smallestPerfectSquareDivisibleBy (a b c d : Nat) (s : Nat) : Prop :=
  ∃ k : Nat, s = k * k ∧ s % a = 0 ∧ s % b = 0 ∧ s % c = 0 ∧ s % d = 0

theorem smallest_perfect_square_5336100 :
  smallestPerfectSquareDivisibleBy 6 14 22 30 5336100 :=
sorry

end smallest_perfect_square_5336100_l877_87741


namespace find_positive_x_l877_87722

theorem find_positive_x (x : ℝ) (h1 : x * ⌊x⌋ = 72) (h2 : x > 0) : x = 9 :=
by 
  sorry

end find_positive_x_l877_87722


namespace determinant_problem_l877_87770

theorem determinant_problem (a b c d : ℝ)
  (h : Matrix.det ![![a, b], ![c, d]] = 4) :
  Matrix.det ![![a, 5*a + 3*b], ![c, 5*c + 3*d]] = 12 := by
  sorry

end determinant_problem_l877_87770


namespace geometric_sequence_l877_87715

theorem geometric_sequence (q : ℝ) (a : ℕ → ℝ) (h1 : q > 0) (h2 : a 2 = 1)
  (h3 : a 2 * a 10 = 2 * (a 5)^2) : ∀ n, a n = 2^((n-2:ℝ)/2) := by
  sorry

end geometric_sequence_l877_87715


namespace compute_expression_equals_375_l877_87754

theorem compute_expression_equals_375 : 15 * (30 / 6) ^ 2 = 375 := 
by 
  have frac_simplified : 30 / 6 = 5 := by sorry
  have power_calculated : 5 ^ 2 = 25 := by sorry
  have final_result : 15 * 25 = 375 := by sorry
  sorry

end compute_expression_equals_375_l877_87754


namespace equivalent_conditions_l877_87797

theorem equivalent_conditions 
  (f : ℕ+ → ℕ+)
  (H1 : ∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m))
  (H2 : ∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :
  (∀ (m n : ℕ+), m ≤ n → (f m + n) ∣ (f n + m)) ↔ 
  (∀ (m n : ℕ+), m ≥ n → (f m + n) ∣ (f n + m)) :=
sorry

end equivalent_conditions_l877_87797


namespace intersection_of_A_and_B_l877_87772

noncomputable def A : Set ℕ := {x | x > 0 ∧ x ≤ 3}
def B : Set ℕ := {x | 0 < x ∧ x < 4}

theorem intersection_of_A_and_B : 
  A ∩ B = {1, 2, 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l877_87772


namespace largest_int_less_than_100_by_7_l877_87773

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end largest_int_less_than_100_by_7_l877_87773


namespace car_second_hour_speed_l877_87798

theorem car_second_hour_speed (s1 s2 : ℕ) (h1 : s1 = 100) (avg : (s1 + s2) / 2 = 80) : s2 = 60 :=
by
  sorry

end car_second_hour_speed_l877_87798


namespace solve_equation_l877_87745

theorem solve_equation (x : ℝ) : 
  16 * (x - 1) ^ 2 - 9 = 0 ↔ (x = 7 / 4 ∨ x = 1 / 4) := by
  sorry

end solve_equation_l877_87745


namespace convex_polyhedron_has_triangular_face_l877_87701

def convex_polyhedron : Type := sorry -- placeholder for the type of convex polyhedra
def face (P : convex_polyhedron) : Type := sorry -- placeholder for the type of faces of a polyhedron
def vertex (P : convex_polyhedron) : Type := sorry -- placeholder for the type of vertices of a polyhedron
def edge (P : convex_polyhedron) : Type := sorry -- placeholder for the type of edges of a polyhedron

-- The number of edges meeting at a specific vertex
def vertex_degree (P : convex_polyhedron) (v : vertex P) : ℕ := sorry

-- Number of edges or vertices on a specific face
def face_sides (P : convex_polyhedron) (f : face P) : ℕ := sorry

-- A polyhedron is convex
def is_convex (P : convex_polyhedron) : Prop := sorry

-- A face is a triangle if it has 3 sides
def is_triangle (P : convex_polyhedron) (f : face P) := face_sides P f = 3

-- The problem statement in Lean 4
theorem convex_polyhedron_has_triangular_face
  (P : convex_polyhedron)
  (h1 : is_convex P)
  (h2 : ∀ v : vertex P, vertex_degree P v ≥ 4) :
  ∃ f : face P, is_triangle P f :=
sorry

end convex_polyhedron_has_triangular_face_l877_87701


namespace three_numbers_sum_div_by_three_l877_87727

theorem three_numbers_sum_div_by_three (s : Fin 7 → ℕ) : 
  ∃ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (s a + s b + s c) % 3 = 0 := 
sorry

end three_numbers_sum_div_by_three_l877_87727


namespace nadine_spent_money_l877_87762

theorem nadine_spent_money (table_cost : ℕ) (chair_cost : ℕ) (num_chairs : ℕ) 
    (h_table_cost : table_cost = 34) 
    (h_chair_cost : chair_cost = 11) 
    (h_num_chairs : num_chairs = 2) : 
    table_cost + num_chairs * chair_cost = 56 :=
by
  sorry

end nadine_spent_money_l877_87762


namespace cuboids_painted_l877_87717

-- Let's define the conditions first
def faces_per_cuboid : ℕ := 6
def total_faces_painted : ℕ := 36

-- Now, we state the theorem we want to prove
theorem cuboids_painted (n : ℕ) (h : total_faces_painted = n * faces_per_cuboid) : n = 6 :=
by
  -- Add proof here
  sorry

end cuboids_painted_l877_87717


namespace max_sides_13_eq_13_max_sides_1950_eq_1950_l877_87744

noncomputable def max_sides (n : ℕ) : ℕ := n

theorem max_sides_13_eq_13 : max_sides 13 = 13 :=
by {
  sorry
}

theorem max_sides_1950_eq_1950 : max_sides 1950 = 1950 :=
by {
  sorry
}

end max_sides_13_eq_13_max_sides_1950_eq_1950_l877_87744


namespace arithmetic_and_geometric_sequence_l877_87783

theorem arithmetic_and_geometric_sequence (a : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + 2) 
  (h_geom_seq : (a 2)^2 = a 0 * a 3) : 
  a 1 + a 2 = -10 := 
sorry

end arithmetic_and_geometric_sequence_l877_87783


namespace arithmetic_sequence_geo_ratio_l877_87795

theorem arithmetic_sequence_geo_ratio
  (a_n : ℕ → ℝ)
  (d : ℝ)
  (h_nonzero : d ≠ 0)
  (S : ℕ → ℝ)
  (h_seq : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_geo : (S 2) ^ 2 = S 1 * S 4) :
  (a_n 2 + a_n 3) / a_n 1 = 8 :=
by sorry

end arithmetic_sequence_geo_ratio_l877_87795


namespace expression_evaluation_l877_87793

theorem expression_evaluation :
  (40 - (2040 - 210)) + (2040 - (210 - 40)) = 80 :=
by
  sorry

end expression_evaluation_l877_87793


namespace round_nearest_hundredth_problem_l877_87708

noncomputable def round_nearest_hundredth (x : ℚ) : ℚ :=
  let shifted := x * 100
  let rounded := if (shifted - shifted.floor) < 0.5 then shifted.floor else shifted.ceil
  rounded / 100

theorem round_nearest_hundredth_problem :
  let A := 34.561
  let B := 34.558
  let C := 34.5539999
  let D := 34.5601
  let E := 34.56444
  round_nearest_hundredth A = 34.56 ∧
  round_nearest_hundredth B = 34.56 ∧
  round_nearest_hundredth C ≠ 34.56 ∧
  round_nearest_hundredth D = 34.56 ∧
  round_nearest_hundredth E = 34.56 :=
sorry

end round_nearest_hundredth_problem_l877_87708


namespace C_necessary_but_not_sufficient_for_A_l877_87719

variable {A B C : Prop}

-- Given conditions
def sufficient_not_necessary (h : A → B) (hn : ¬(B → A)) := h
def necessary_sufficient := B ↔ C

-- Prove that C is a necessary but not sufficient condition for A
theorem C_necessary_but_not_sufficient_for_A (h₁ : A → B) (hn : ¬(B → A)) (h₂ : B ↔ C) : (C → A) ∧ ¬(A → C) :=
  by
  sorry

end C_necessary_but_not_sufficient_for_A_l877_87719


namespace remainder_333_pow_333_mod_11_l877_87769

theorem remainder_333_pow_333_mod_11 : (333 ^ 333) % 11 = 5 := by
  sorry

end remainder_333_pow_333_mod_11_l877_87769


namespace four_cards_probability_l877_87768

theorem four_cards_probability :
  let deck_size := 52
  let suits_size := 13
  ∀ (C D H S : ℕ), 
  C = 1 ∧ D = 13 ∧ H = 13 ∧ S = 13 →
  (C / deck_size) *
  (D / (deck_size - 1)) *
  (H / (deck_size - 2)) *
  (S / (deck_size - 3)) = (2197 / 499800) :=
by
  intros deck_size suits_size C D H S h
  sorry

end four_cards_probability_l877_87768


namespace sqrt_mul_eq_6_l877_87723

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l877_87723


namespace min_value_f_l877_87704

def f (x : ℝ) : ℝ := 5 * x^2 + 10 * x + 20

theorem min_value_f : ∃ (x : ℝ), f x = 15 :=
by
  sorry

end min_value_f_l877_87704


namespace probability_three_girls_chosen_l877_87746

theorem probability_three_girls_chosen :
  let total_members := 15;
  let boys := 7;
  let girls := 8;
  let total_ways := Nat.choose total_members 3;
  let girls_ways := Nat.choose girls 3;
  total_ways = Nat.choose 15 3 ∧ girls_ways = Nat.choose 8 3 →
  (girls_ways : ℚ) / (total_ways : ℚ) = 8 / 65 := 
by  
  sorry

end probability_three_girls_chosen_l877_87746


namespace vasya_made_a_mistake_l877_87705

theorem vasya_made_a_mistake :
  ∀ x : ℝ, x^4 - 3*x^3 - 2*x^2 - 4*x + 1 = 0 → ¬ x < 0 :=
by sorry

end vasya_made_a_mistake_l877_87705


namespace cube_surface_area_increase_l877_87758

theorem cube_surface_area_increase (s : ℝ) :
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  (A_new - A_original) / A_original * 100 = 224 :=
by
  -- Definitions from the conditions
  let A_original := 6 * s^2
  let s' := 1.8 * s
  let A_new := 6 * s'^2
  -- Rest of the proof; replace sorry with the actual proof
  sorry

end cube_surface_area_increase_l877_87758


namespace range_of_a_l877_87703

theorem range_of_a (a : ℝ) (h : ¬ (1^2 - 2*1 + a > 0)) : 1 ≤ a := sorry

end range_of_a_l877_87703


namespace yellow_less_than_three_times_red_l877_87759

def num_red : ℕ := 40
def less_than_three_times (Y : ℕ) : Prop := Y < 120
def blue_half_yellow (Y B : ℕ) : Prop := B = Y / 2
def remaining_after_carlos (B : ℕ) : Prop := 40 + B = 90
def difference_three_times_red (Y : ℕ) : ℕ := 3 * num_red - Y

theorem yellow_less_than_three_times_red (Y B : ℕ) 
  (h1 : less_than_three_times Y) 
  (h2 : blue_half_yellow Y B) 
  (h3 : remaining_after_carlos B) : 
  difference_three_times_red Y = 20 := by
  sorry

end yellow_less_than_three_times_red_l877_87759


namespace largest_4_digit_divisible_by_50_l877_87742

-- Define the condition for a 4-digit number
def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define the largest 4-digit number
def largest_4_digit : ℕ := 9999

-- Define the property that a number is exactly divisible by 50
def divisible_by_50 (n : ℕ) : Prop := n % 50 = 0

-- Main statement to be proved
theorem largest_4_digit_divisible_by_50 :
  ∃ n, is_4_digit n ∧ divisible_by_50 n ∧ ∀ m, is_4_digit m → divisible_by_50 m → m ≤ n ∧ n = 9950 :=
by
  sorry

end largest_4_digit_divisible_by_50_l877_87742


namespace probability_two_white_balls_l877_87780

-- Definitions
def totalBalls : ℕ := 5
def whiteBalls : ℕ := 3
def blackBalls : ℕ := 2
def totalWaysToDrawTwoBalls : ℕ := Nat.choose totalBalls 2
def waysToDrawTwoWhiteBalls : ℕ := Nat.choose whiteBalls 2

-- Theorem statement
theorem probability_two_white_balls :
  (waysToDrawTwoWhiteBalls : ℚ) / totalWaysToDrawTwoBalls = 3 / 10 := by
  sorry

end probability_two_white_balls_l877_87780


namespace race_distance_l877_87724

theorem race_distance (T_A T_B : ℝ) (D : ℝ) (V_A V_B : ℝ)
  (h1 : T_A = 23)
  (h2 : T_B = 30)
  (h3 : V_A = D / 23)
  (h4 : V_B = (D - 56) / 30)
  (h5 : D = (D - 56) * (23 / 30) + 56) :
  D = 56 :=
by
  sorry

end race_distance_l877_87724


namespace solve_for_y_l877_87702

noncomputable def solve_quadratic := {y : ℂ // 4 + 3 * y^2 = 0.7 * y - 40}

theorem solve_for_y : 
  ∃ y : ℂ, (y = 0.1167 + 3.8273 * Complex.I ∨ y = 0.1167 - 3.8273 * Complex.I) ∧
            (4 + 3 * y^2 = 0.7 * y - 40) :=
by
  sorry

end solve_for_y_l877_87702


namespace sum_of_arithmetic_sequences_l877_87799

theorem sum_of_arithmetic_sequences (n : ℕ) (h : n ≠ 0) :
  (2 * n * (n + 3) = n * (n + 12)) → (n = 6) :=
by
  intro h_eq
  have h_nonzero : n ≠ 0 := h
  sorry

end sum_of_arithmetic_sequences_l877_87799


namespace angle_ABC_tangent_circle_l877_87729

theorem angle_ABC_tangent_circle 
  (BAC ACB : ℝ)
  (h1 : BAC = 70)
  (h2 : ACB = 45)
  (D : Type)
  (incenter : ∀ D : Type, Prop)  -- Represent the condition that D is the incenter
  : ∃ ABC : ℝ, ABC = 65 :=
by
  sorry

end angle_ABC_tangent_circle_l877_87729


namespace factorization_correct_l877_87747

def factor_expression (x : ℝ) : ℝ :=
  (12 * x^4 - 27 * x^3 + 45 * x) - (-3 * x^4 - 6 * x^3 + 9 * x)

theorem factorization_correct (x : ℝ) : 
  factor_expression x = 3 * x * (5 * x^3 - 7 * x^2 + 12) :=
by
  sorry

end factorization_correct_l877_87747


namespace line_solutions_l877_87760

-- Definition for points
def point := ℝ × ℝ

-- Conditions for lines and points
def line1 (p : point) : Prop := 3 * p.1 + 4 * p.2 = 2
def line2 (p : point) : Prop := 2 * p.1 + p.2 = -2
def line3 : Prop := ∃ p : point, line1 p ∧ line2 p

def lineL (p : point) : Prop := 2 * p.1 + p.2 = -2 -- Line l we need to prove
def perp_lineL : Prop := ∃ p : point, lineL p ∧ p.1 - 2 * p.2 = 1

-- Symmetry condition for the line
def symmetric_line (p : point) : Prop := 2 * p.1 + p.2 = 2 -- Symmetric line we need to prove

-- Main theorem to prove
theorem line_solutions :
  line3 →
  perp_lineL →
  (∀ p, lineL p ↔ 2 * p.1 + p.2 = -2) ∧
  (∀ p, symmetric_line p ↔ 2 * p.1 + p.2 = 2) :=
sorry

end line_solutions_l877_87760


namespace find_row_with_sum_2013_squared_l877_87765

-- Define the sum of the numbers in the nth row
def sum_of_row (n : ℕ) : ℕ := (2 * n - 1)^2

theorem find_row_with_sum_2013_squared : (∃ n : ℕ, sum_of_row n = 2013^2) ∧ (sum_of_row 1007 = 2013^2) :=
by
  sorry

end find_row_with_sum_2013_squared_l877_87765


namespace find_bullet_l877_87787

theorem find_bullet (x y : ℝ) (h₁ : 3 * x + y = 8) (h₂ : y = -1) : 2 * x - y = 7 :=
sorry

end find_bullet_l877_87787


namespace sqrt_neg9_sq_l877_87721

theorem sqrt_neg9_sq : Real.sqrt ((-9 : Real)^2) = 9 := 
by 
  sorry

end sqrt_neg9_sq_l877_87721


namespace cost_of_fencing_l877_87764

noncomputable def fencingCost :=
  let π := 3.14159
  let diameter := 32
  let costPerMeter := 1.50
  let circumference := π * diameter
  let totalCost := costPerMeter * circumference
  totalCost

theorem cost_of_fencing :
  let roundedCost := (fencingCost).round
  roundedCost = 150.80 :=
by
  sorry

end cost_of_fencing_l877_87764


namespace sample_size_is_100_l877_87743

-- Conditions:
def scores_from_students := 100
def sampling_method := "simple random sampling"
def goal := "statistical analysis of senior three students' exam performance"

-- Problem statement:
theorem sample_size_is_100 :
  scores_from_students = 100 →
  sampling_method = "simple random sampling" →
  goal = "statistical analysis of senior three students' exam performance" →
  scores_from_students = 100 := by
sorry

end sample_size_is_100_l877_87743


namespace groom_dog_time_l877_87726

theorem groom_dog_time :
  ∃ (D : ℝ), (5 * D + 3 * 0.5 = 14) ∧ (D = 2.5) :=
by
  sorry

end groom_dog_time_l877_87726


namespace store_credit_card_discount_proof_l877_87790

def full_price : ℕ := 125
def sale_discount_percentage : ℕ := 20
def coupon_discount : ℕ := 10
def total_savings : ℕ := 44

def sale_discount := full_price * sale_discount_percentage / 100
def price_after_sale_discount := full_price - sale_discount
def price_after_coupon := price_after_sale_discount - coupon_discount
def store_credit_card_discount := total_savings - sale_discount - coupon_discount
def discount_percentage_of_store_credit := (store_credit_card_discount * 100) / price_after_coupon

theorem store_credit_card_discount_proof : discount_percentage_of_store_credit = 10 := by
  sorry

end store_credit_card_discount_proof_l877_87790


namespace factor_exp_l877_87734

variable (x : ℤ)

theorem factor_exp : x * (x + 2) + (x + 2) = (x + 1) * (x + 2) :=
by
  sorry

end factor_exp_l877_87734


namespace cos_double_angle_nonpositive_l877_87700

theorem cos_double_angle_nonpositive (α β : ℝ) (φ : ℝ) 
  (h : Real.tan φ = 1 / (Real.cos α * Real.cos β + Real.tan α * Real.tan β)) : 
  Real.cos (2 * φ) ≤ 0 := 
sorry

end cos_double_angle_nonpositive_l877_87700
