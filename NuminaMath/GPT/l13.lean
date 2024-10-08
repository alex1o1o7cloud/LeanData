import Mathlib

namespace multiplication_result_l13_13910

theorem multiplication_result : 143 * 21 * 4 * 37 * 2 = 888888 := by
  sorry

end multiplication_result_l13_13910


namespace profit_loss_balance_l13_13944

-- Defining variables
variables (C L : Real)

-- Profit and loss equations according to problem conditions
theorem profit_loss_balance (h1 : 832 - C = C - L) (h2 : 992 = 0.55 * C) : 
  (C + 992 = 2795.64) :=
by
  -- Statement of the theorem
  sorry

end profit_loss_balance_l13_13944


namespace correct_calculation_l13_13103

variable (a : ℝ)

theorem correct_calculation : (-2 * a) ^ 3 = -8 * a ^ 3 := by
  sorry

end correct_calculation_l13_13103


namespace unshaded_area_eq_20_l13_13942

-- Define the dimensions of the first rectangle
def rect1_width := 4
def rect1_length := 12

-- Define the dimensions of the second rectangle
def rect2_width := 5
def rect2_length := 10

-- Define the dimensions of the overlapping region
def overlap_width := 4
def overlap_length := 5

-- Calculate area functions
def area (width length : ℕ) := width * length

-- Calculate areas of the individual rectangles and the overlapping region
def area_rect1 := area rect1_width rect1_length
def area_rect2 := area rect2_width rect2_length
def overlap_area := area overlap_width overlap_length

-- Calculate the total shaded area
def total_shaded_area := area_rect1 + area_rect2 - overlap_area

-- The total area of the combined figure (assumed to be the union of both rectangles) minus shaded area gives the unshaded area
def total_area := rect1_width * rect1_length + rect2_width * rect2_length
def unshaded_area := total_area - total_shaded_area

theorem unshaded_area_eq_20 : unshaded_area = 20 := by
  sorry

end unshaded_area_eq_20_l13_13942


namespace fraction_saved_l13_13228

variable {P : ℝ} (hP : P > 0)

theorem fraction_saved (f : ℝ) (hf0 : 0 ≤ f) (hf1 : f ≤ 1) (condition : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  sorry

end fraction_saved_l13_13228


namespace tracy_initial_candies_l13_13189

noncomputable def initial_candies : Nat := 80

theorem tracy_initial_candies
  (x : Nat)
  (hx1 : ∃ y : Nat, (1 ≤ y ∧ y ≤ 6) ∧ x = (5 * (44 + y)) / 3)
  (hx2 : x % 20 = 0) : x = initial_candies := by
  sorry

end tracy_initial_candies_l13_13189


namespace common_tangent_l13_13271

-- Definition of the ellipse and hyperbola
def ellipse (x y : ℝ) : Prop := 9 * x^2 + 16 * y^2 = 144
def hyperbola (x y : ℝ) : Prop := 7 * x^2 - 32 * y^2 = 224

-- The statement to prove
theorem common_tangent :
  (∀ x y : ℝ, ellipse x y → hyperbola x y → ((x + y + 5 = 0) ∨ (x + y - 5 = 0) ∨ (x - y + 5 = 0) ∨ (x - y - 5 = 0))) := 
sorry

end common_tangent_l13_13271


namespace min_possible_value_of_x_l13_13076

theorem min_possible_value_of_x :
  ∀ (x y : ℝ),
  (69 + 53 + 69 + 71 + 78 + x + y) / 7 = 66 →
  (∀ y ≤ 100, x ≥ 0) →
  x ≥ 22 :=
by
  intros x y h_avg h_y 
  -- proof steps go here
  sorry

end min_possible_value_of_x_l13_13076


namespace find_blue_beads_per_row_l13_13272

-- Given the conditions of the problem:
def number_of_purple_beads : ℕ := 50 * 20
def number_of_gold_beads : ℕ := 80
def total_cost : ℕ := 180

-- Define the main theorem to solve for the number of blue beads per row.
theorem find_blue_beads_per_row (x : ℕ) :
  (number_of_purple_beads + 40 * x + number_of_gold_beads = total_cost) → x = (total_cost - (number_of_purple_beads + number_of_gold_beads)) / 40 := 
by {
  -- Proof steps would go here
  sorry
}

end find_blue_beads_per_row_l13_13272


namespace line_intersects_circle_l13_13311

/-- The positional relationship between the line y = ax + 1 and the circle x^2 + y^2 - 2x - 3 = 0
    is always intersecting for any real number a. -/
theorem line_intersects_circle (a : ℝ) : 
    ∀ a : ℝ, ∃ x y : ℝ, y = a * x + 1 ∧ x^2 + y^2 - 2 * x - 3 = 0 :=
by
    sorry

end line_intersects_circle_l13_13311


namespace diane_total_harvest_l13_13581

def total_harvest (h1 i1 i2 : Nat) : Nat :=
  h1 + (h1 + i1) + ((h1 + i1) + i2)

theorem diane_total_harvest :
  total_harvest 2479 6085 7890 = 27497 := 
by 
  sorry

end diane_total_harvest_l13_13581


namespace geom_progression_vertex_ad_l13_13175

theorem geom_progression_vertex_ad
  (a b c d : ℝ)
  (geom_prog : a * c = b * b ∧ b * d = c * c)
  (vertex : (b, c) = (1, 3)) :
  a * d = 3 :=
sorry

end geom_progression_vertex_ad_l13_13175


namespace only_solution_xyz_l13_13048

theorem only_solution_xyz : 
  ∀ (x y z : ℕ), x^3 + 4 * y^3 = 16 * z^3 + 4 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro x y z
  intro h
  sorry

end only_solution_xyz_l13_13048


namespace faster_train_pass_time_l13_13785

-- Defining the conditions
def length_of_train : ℕ := 45 -- length in meters
def speed_of_faster_train : ℕ := 45 -- speed in km/hr
def speed_of_slower_train : ℕ := 36 -- speed in km/hr

-- Define relative speed
def relative_speed := (speed_of_faster_train - speed_of_slower_train) * 5 / 18 -- converting km/hr to m/s

-- Total distance to pass (sum of lengths of both trains)
def total_passing_distance := (2 * length_of_train) -- 2 trains of 45 meters each

-- Calculate the time to pass the slower train
def time_to_pass := total_passing_distance / relative_speed

-- The theorem to prove
theorem faster_train_pass_time : time_to_pass = 36 := by
  -- This is where the proof would be placed
  sorry

end faster_train_pass_time_l13_13785


namespace evaluate_expression_l13_13077

theorem evaluate_expression :
  (827 * 827) - ((827 - 1) * (827 + 1)) = 1 :=
sorry

end evaluate_expression_l13_13077


namespace part_a_constant_part_b_inequality_l13_13874

open Real

noncomputable def cubic_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem part_a_constant (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1 * x2 / x3^2) + cubic_root (x2 * x3 / x1^2) + cubic_root (x3 * x1 / x2^2)) = 
  const_value := sorry

theorem part_b_inequality (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1^2 / (x2 * x3)) + cubic_root (x2^2 / (x3 * x1)) + cubic_root (x3^2 / (x1 * x2))) < (-15 / 4) := sorry

end part_a_constant_part_b_inequality_l13_13874


namespace parallel_ne_implies_value_l13_13827

theorem parallel_ne_implies_value 
  (x : ℝ) 
  (m : ℝ × ℝ := (2 * x, 7)) 
  (n : ℝ × ℝ := (6, x + 4)) 
  (h1 : 2 * x * (x + 4) = 42) 
  (h2 : m ≠ n) :
  x = -7 :=
by {
  sorry
}

end parallel_ne_implies_value_l13_13827


namespace part_a_part_b_part_c_l13_13097

-- Part (a)
theorem part_a (m : ℤ) : (m^2 + 10) % (m - 2) = 0 ∧ (m^2 + 10) % (m + 4) = 0 ↔ m = -5 ∨ m = 9 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) : ∃ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 :=
sorry

-- Part (c)
theorem part_c (n : ℤ) : ∃ N : ℕ, ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 → m < N :=
sorry

end part_a_part_b_part_c_l13_13097


namespace intersection_of_M_and_N_l13_13797

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 = 2 * x}

theorem intersection_of_M_and_N : M ∩ N = {0} := 
by sorry

end intersection_of_M_and_N_l13_13797


namespace joe_total_spending_at_fair_l13_13501

-- Definitions based on conditions
def entrance_fee (age : ℕ) : ℝ := if age < 18 then 5 else 6
def ride_cost (rides : ℕ) : ℝ := rides * 0.5

-- Given conditions
def joe_age := 19
def twin_age := 6

def total_cost (joe_age : ℕ) (twin_age : ℕ) (rides_per_person : ℕ) :=
  entrance_fee joe_age + 2 * entrance_fee twin_age + 3 * ride_cost rides_per_person

-- The main statement to be proven
theorem joe_total_spending_at_fair : total_cost joe_age twin_age 3 = 20.5 :=
by
  sorry

end joe_total_spending_at_fair_l13_13501


namespace matchstick_equality_l13_13091

theorem matchstick_equality :
  abs ((22 : ℝ) / 7 - Real.pi) < 0.1 := 
sorry

end matchstick_equality_l13_13091


namespace teacher_work_months_l13_13300

variable (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) (total_earnings : ℕ)

def monthly_earnings (periods_per_day : ℕ) (pay_per_period : ℕ) (days_per_month : ℕ) : ℕ :=
  periods_per_day * pay_per_period * days_per_month

def number_of_months_worked (total_earnings : ℕ) (monthly_earnings : ℕ) : ℕ :=
  total_earnings / monthly_earnings

theorem teacher_work_months :
  let periods_per_day := 5
  let pay_per_period := 5
  let days_per_month := 24
  let total_earnings := 3600
  number_of_months_worked total_earnings (monthly_earnings periods_per_day pay_per_period days_per_month) = 6 :=
by
  sorry

end teacher_work_months_l13_13300


namespace difference_of_squares_l13_13628

theorem difference_of_squares {a b : ℝ} (h1 : a + b = 75) (h2 : a - b = 15) : a^2 - b^2 = 1125 :=
by
  sorry

end difference_of_squares_l13_13628


namespace a_and_b_together_time_eq_4_over_3_l13_13846

noncomputable def work_together_time (a b c h : ℝ) :=
  (1 / a) + (1 / b) + (1 / c) = (1 / (a - 6)) ∧
  (1 / a) + (1 / b) = 1 / h ∧
  (1 / (a - 6)) = (1 / (b - 1)) ∧
  (1 / (a - 6)) = 2 / c

theorem a_and_b_together_time_eq_4_over_3 (a b c h : ℝ) (h_wt : work_together_time a b c h) : 
  h = 4 / 3 :=
  sorry

end a_and_b_together_time_eq_4_over_3_l13_13846


namespace abs_diff_eq_five_l13_13237

theorem abs_diff_eq_five (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 7) : |a - b| = 5 :=
by
  sorry

end abs_diff_eq_five_l13_13237


namespace factorial_fraction_eq_zero_l13_13643

theorem factorial_fraction_eq_zero :
  ((5 * (Nat.factorial 7) - 35 * (Nat.factorial 6)) / Nat.factorial 8 = 0) :=
by
  sorry

end factorial_fraction_eq_zero_l13_13643


namespace difference_of_squares_example_l13_13169

theorem difference_of_squares_example : 625^2 - 375^2 = 250000 :=
by sorry

end difference_of_squares_example_l13_13169


namespace a_plus_b_eq_six_l13_13405

theorem a_plus_b_eq_six (a b : ℤ) (k : ℝ) (h1 : k = a + Real.sqrt b)
  (h2 : ∀ k > 0, |Real.log k / Real.log 2 - Real.log (k + 6) / Real.log 2| = 1) :
  a + b = 6 :=
by
  sorry

end a_plus_b_eq_six_l13_13405


namespace jake_not_drop_coffee_l13_13470

theorem jake_not_drop_coffee :
  let p_trip := 0.40
  let p_drop_trip := 0.25
  let p_step := 0.30
  let p_drop_step := 0.20
  let p_no_drop_trip := 1 - (p_trip * p_drop_trip)
  let p_no_drop_step := 1 - (p_step * p_drop_step)
  (p_no_drop_trip * p_no_drop_step) = 0.846 :=
by
  sorry

end jake_not_drop_coffee_l13_13470


namespace value_after_addition_l13_13799

theorem value_after_addition (x : ℕ) (h : x / 9 = 8) : x + 11 = 83 :=
by
  sorry

end value_after_addition_l13_13799


namespace nora_third_tree_oranges_l13_13731

theorem nora_third_tree_oranges (a b c total : ℕ)
  (h_a : a = 80)
  (h_b : b = 60)
  (h_total : total = 260)
  (h_sum : total = a + b + c) :
  c = 120 :=
by
  -- The proof should go here
  sorry

end nora_third_tree_oranges_l13_13731


namespace option_C_is_quadratic_l13_13851

-- Define what it means for an equation to be quadratic
def is_quadratic (p : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ (x : ℝ), p x ↔ a*x^2 + b*x + c = 0

-- Define the equation in option C
def option_C (x : ℝ) : Prop := (x - 1) * (x - 2) = 0

-- The theorem we need to prove
theorem option_C_is_quadratic : is_quadratic option_C :=
  sorry

end option_C_is_quadratic_l13_13851


namespace full_time_and_year_l13_13450

variable (Total F Y N FY : ℕ)

theorem full_time_and_year (h1 : Total = 130)
                            (h2 : F = 80)
                            (h3 : Y = 100)
                            (h4 : N = 20)
                            (h5 : Total = FY + (F - FY) + (Y - FY) + N) :
    FY = 90 := 
sorry

end full_time_and_year_l13_13450


namespace excluded_twins_lineup_l13_13481

/-- 
  Prove that the number of ways to choose 5 starters from 15 players,
  such that both Alice and Bob (twins) are not included together in the lineup, is 2717.
-/
theorem excluded_twins_lineup (n : ℕ) (k : ℕ) (t : ℕ) (u : ℕ) (h_n : n = 15) (h_k : k = 5) (h_t : t = 2) (h_u : u = 3) :
  ((n.choose k) - ((n - t).choose u)) = 2717 :=
by {
  sorry
}

end excluded_twins_lineup_l13_13481


namespace girls_percentage_l13_13890

theorem girls_percentage (total_students girls boys : ℕ) 
    (total_eq : total_students = 42)
    (ratio : 3 * girls = 4 * boys)
    (total_students_eq : total_students = girls + boys) : 
    (girls * 100 / total_students : ℚ) = 57.14 := 
by 
  sorry

end girls_percentage_l13_13890


namespace Angelina_speed_grocery_to_gym_l13_13049

-- Define parameters for distances and times
def distance_home_to_grocery : ℕ := 720
def distance_grocery_to_gym : ℕ := 480
def time_difference : ℕ := 40

-- Define speeds
variable (v : ℕ) -- speed in meters per second from home to grocery
def speed_home_to_grocery := v
def speed_grocery_to_gym := 2 * v

-- Define times using given speeds and distances
def time_home_to_grocery := distance_home_to_grocery / speed_home_to_grocery
def time_grocery_to_gym := distance_grocery_to_gym / speed_grocery_to_gym

-- Proof statement for the problem
theorem Angelina_speed_grocery_to_gym
  (v_pos : 0 < v)
  (condition : time_home_to_grocery - time_difference = time_grocery_to_gym) :
  speed_grocery_to_gym = 24 := by
  sorry

end Angelina_speed_grocery_to_gym_l13_13049


namespace part1_part2_l13_13631

def f (x : ℝ) : ℝ := abs (x - 5) + abs (x + 4)

theorem part1 (x : ℝ) : f x ≥ 12 ↔ x ≥ 13 / 2 ∨ x ≤ -11 / 2 :=
by
    sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x - 2 ^ (1 - 3 * a) - 1 ≥ 0) ↔ -2 / 3 ≤ a :=
by
    sorry

end part1_part2_l13_13631


namespace both_pipes_opened_together_for_2_minutes_l13_13715

noncomputable def fill_time (t : ℝ) : Prop :=
  let rate_p := 1 / 12
  let rate_q := 1 / 15
  let combined_rate := rate_p + rate_q
  let work_done_by_p_q := combined_rate * t
  let work_done_by_q := rate_q * 10.5
  work_done_by_p_q + work_done_by_q = 1

theorem both_pipes_opened_together_for_2_minutes : ∃ t : ℝ, fill_time t ∧ t = 2 :=
by
  use 2
  unfold fill_time
  sorry

end both_pipes_opened_together_for_2_minutes_l13_13715


namespace final_probability_l13_13812

def total_cards := 52
def kings := 4
def aces := 4
def chosen_cards := 3

namespace probability

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def prob_three_kings : ℚ :=
  (4 / 52) * (3 / 51) * (2 / 50)

def prob_exactly_two_aces : ℚ :=
  (choose 4 2 * choose 48 1) / choose 52 3

def prob_exactly_three_aces : ℚ :=
  (choose 4 3) / choose 52 3

def prob_at_least_two_aces : ℚ :=
  prob_exactly_two_aces + prob_exactly_three_aces

def prob_three_kings_or_two_aces : ℚ :=
  prob_three_kings + prob_at_least_two_aces

theorem final_probability :
  prob_three_kings_or_two_aces = 6 / 425 :=
by
  sorry

end probability

end final_probability_l13_13812


namespace max_2a_b_2c_l13_13421

theorem max_2a_b_2c (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 2 * a + b + 2 * c ≤ 3 :=
sorry

end max_2a_b_2c_l13_13421


namespace incorrect_statement_c_l13_13899

open Real

theorem incorrect_statement_c (p q: ℝ) : ¬(∀ x: ℝ, (x * abs x + p * x + q = 0 ↔ p^2 - 4 * q ≥ 0)) :=
sorry

end incorrect_statement_c_l13_13899


namespace polynomial_divisibility_l13_13948

theorem polynomial_divisibility (m : ℕ) (hm : 0 < m) :
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1) ^ (2 * m) - x ^ (2 * m) - 2 * x - 1 :=
by
  intro x
  sorry

end polynomial_divisibility_l13_13948


namespace john_total_jury_duty_days_l13_13642

-- Definitions based on the given conditions
def jury_selection_days : ℕ := 2
def trial_length_multiplier : ℕ := 4
def deliberation_days : ℕ := 6
def deliberation_hours_per_day : ℕ := 16
def hours_in_a_day : ℕ := 24

-- Define the total days John spends on jury duty
def total_days_on_jury_duty : ℕ :=
  jury_selection_days +
  (trial_length_multiplier * jury_selection_days) +
  (deliberation_days * deliberation_hours_per_day) / hours_in_a_day

-- The theorem to prove
theorem john_total_jury_duty_days : total_days_on_jury_duty = 14 := by
  sorry

end john_total_jury_duty_days_l13_13642


namespace part_one_part_two_l13_13977

-- Part (1)
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

-- Part (2)
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : 
  2 * a + b = 8 :=
sorry

end part_one_part_two_l13_13977


namespace proof_problem_l13_13039

noncomputable def p : Prop := ∃ x : ℝ, Real.sin x > 1
noncomputable def q : Prop := ∀ x : ℝ, Real.exp (-x) < 0

theorem proof_problem : ¬ (p ∨ q) :=
by sorry

end proof_problem_l13_13039


namespace grapes_difference_l13_13612

theorem grapes_difference (R A_i A_l : ℕ) 
  (hR : R = 25) 
  (hAi : A_i = R + 2) 
  (hTotal : R + A_i + A_l = 83) : 
  A_l - A_i = 4 := 
by
  sorry

end grapes_difference_l13_13612


namespace muffin_cost_l13_13233

theorem muffin_cost (m : ℝ) :
  let fruit_cup_cost := 3
  let francis_cost := 2 * m + 2 * fruit_cup_cost
  let kiera_cost := 2 * m + 1 * fruit_cup_cost
  let total_cost := 17
  (francis_cost + kiera_cost = total_cost) → m = 2 :=
by
  intro h
  sorry

end muffin_cost_l13_13233


namespace parallel_slope_l13_13382

theorem parallel_slope (x y : ℝ) (h : 3 * x + 6 * y = -21) : 
    ∃ m : ℝ, m = -1 / 2 :=
by
  sorry

end parallel_slope_l13_13382


namespace ordered_pairs_satisfying_condition_l13_13144

theorem ordered_pairs_satisfying_condition : 
  ∃! (pairs : Finset (ℕ × ℕ)),
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ 
      m > 0 ∧ n > 0 ∧ m ≥ n ∧ m^2 - n^2 = 144) ∧ 
    pairs.card = 4 := sorry

end ordered_pairs_satisfying_condition_l13_13144


namespace obtuse_equilateral_triangle_impossible_l13_13728

-- Define a scalene triangle 
def is_scalene_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ A + B + C = 180

-- Define acute triangles
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

-- Define right triangles
def is_right_triangle (A B C : ℝ) : Prop :=
  A = 90 ∨ B = 90 ∨ C = 90

-- Define isosceles triangles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

-- Define obtuse triangles
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

-- Define equilateral triangles
def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = a ∧ A = 60 ∧ B = 60 ∧ C = 60

theorem obtuse_equilateral_triangle_impossible :
  ¬ ∃ (a b c A B C : ℝ), is_equilateral_triangle a b c A B C ∧ is_obtuse_triangle A B C :=
by
  sorry

end obtuse_equilateral_triangle_impossible_l13_13728


namespace inequality_proof_l13_13378

theorem inequality_proof (b c : ℝ) (hb : 0 < b) (hc : 0 < c) :
  (b - c) ^ 2011 * (b + c) ^ 2011 * (c - b) ^ 2011 ≥ 
  (b ^ 2011 - c ^ 2011) * (b ^ 2011 + c ^ 2011) * (c ^ 2011 - b ^ 2011) := 
by
  sorry

end inequality_proof_l13_13378


namespace arithmetic_sequence_property_l13_13205

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
                                     (h2 : a 3 + a 11 = 40) :
  a 6 - a 7 + a 8 = 20 :=
by
  sorry

end arithmetic_sequence_property_l13_13205


namespace total_vegetables_correct_l13_13995

def cucumbers : ℕ := 70
def tomatoes : ℕ := 3 * cucumbers
def total_vegetables : ℕ := cucumbers + tomatoes

theorem total_vegetables_correct : total_vegetables = 280 :=
by
  sorry

end total_vegetables_correct_l13_13995


namespace unique_solution_a_eq_4_l13_13368

theorem unique_solution_a_eq_4 (a : ℝ) (h : ∀ x1 x2 : ℝ, (a * x1^2 + a * x1 + 1 = 0 ∧ a * x2^2 + a * x2 + 1 = 0) → x1 = x2) : a = 4 :=
sorry

end unique_solution_a_eq_4_l13_13368


namespace values_of_n_l13_13081

theorem values_of_n (a b d : ℕ) :
  7 * a + 77 * b + 7777 * d = 6700 →
  ∃ n : ℕ, ∃ (count : ℕ), count = 107 ∧ n = a + 2 * b + 4 * d := 
by
  sorry

end values_of_n_l13_13081


namespace arithmetic_sequence_general_term_l13_13861

theorem arithmetic_sequence_general_term (x : ℕ)
  (t1 t2 t3 : ℤ)
  (h1 : t1 = x - 1)
  (h2 : t2 = x + 1)
  (h3 : t3 = 2 * x + 3) :
  (∃ a : ℕ → ℤ, a 1 = t1 ∧ a 2 = t2 ∧ a 3 = t3 ∧ ∀ n, a n = 2 * n - 3) := 
sorry

end arithmetic_sequence_general_term_l13_13861


namespace math_problem_l13_13566

open Real

theorem math_problem
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x + y + z = 1) :
  ( (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 ) :=
by {
  sorry
}

end math_problem_l13_13566


namespace pages_of_shorter_book_is_10_l13_13402

theorem pages_of_shorter_book_is_10
  (x : ℕ) 
  (h_diff : ∀ (y : ℕ), x = y - 10)
  (h_divide : (x + 10) / 2 = x) 
  : x = 10 :=
by
  sorry

end pages_of_shorter_book_is_10_l13_13402


namespace amount_per_person_l13_13600

theorem amount_per_person (total_amount : ℕ) (num_persons : ℕ) (amount_each : ℕ)
  (h1 : total_amount = 42900) (h2 : num_persons = 22) (h3 : amount_each = 1950) :
  total_amount / num_persons = amount_each :=
by
  -- Proof to be filled
  sorry

end amount_per_person_l13_13600


namespace min_value_of_ratio_l13_13345

noncomputable def min_ratio (a b c d : ℕ) : ℝ :=
  let num := 1000 * a + 100 * b + 10 * c + d
  let denom := a + b + c + d
  (num : ℝ) / (denom : ℝ)

theorem min_value_of_ratio : 
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
  min_ratio a b c d = 60.5 :=
by
  sorry

end min_value_of_ratio_l13_13345


namespace band_total_l13_13718

theorem band_total (flutes_total clarinets_total trumpets_total pianists_total : ℕ)
                   (flutes_pct clarinets_pct trumpets_pct pianists_pct : ℚ)
                   (h_flutes : flutes_total = 20)
                   (h_clarinets : clarinets_total = 30)
                   (h_trumpets : trumpets_total = 60)
                   (h_pianists : pianists_total = 20)
                   (h_flutes_pct : flutes_pct = 0.8)
                   (h_clarinets_pct : clarinets_pct = 0.5)
                   (h_trumpets_pct : trumpets_pct = 1/3)
                   (h_pianists_pct : pianists_pct = 1/10) :
  flutes_total * flutes_pct + clarinets_total * clarinets_pct + 
  trumpets_total * trumpets_pct + pianists_total * pianists_pct = 53 := by
  sorry

end band_total_l13_13718


namespace box_dimension_triples_l13_13734

theorem box_dimension_triples (N : ℕ) :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ (1 / a + 1 / b + 1 / c = 1 / 8) → ∃ k, k = N := sorry 

end box_dimension_triples_l13_13734


namespace find_n_l13_13260

theorem find_n : ∃ (n : ℤ), -150 < n ∧ n < 150 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1600 * Real.pi / 180) :=
sorry

end find_n_l13_13260


namespace larger_integer_l13_13574

theorem larger_integer (a b : ℕ) (h_diff : a - b = 8) (h_prod : a * b = 224) : a = 16 :=
by
  sorry

end larger_integer_l13_13574


namespace spider_final_position_l13_13904

def circle_points : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def next_position (current : ℕ) : ℕ :=
  if current % 2 = 0 
  then (current + 3 - 1) % 7 + 1 -- Clockwise modulo operation for even
  else (current + 1 - 1) % 7 + 1 -- Clockwise modulo operation for odd

def spider_position_after_jumps (start : ℕ) (jumps : ℕ) : ℕ :=
  (Nat.iterate next_position jumps start)

theorem spider_final_position : spider_position_after_jumps 6 2055 = 2 := 
  by
  sorry

end spider_final_position_l13_13904


namespace speed_of_second_half_l13_13223

theorem speed_of_second_half (t d s1 d1 d2 : ℝ) (h_t : t = 30) (h_d : d = 672) (h_s1 : s1 = 21)
  (h_d1 : d1 = d / 2) (h_d2 : d2 = d / 2) (h_t1 : d1 / s1 = 16) (h_t2 : t - d1 / s1 = 14) :
  d2 / 14 = 24 :=
by sorry

end speed_of_second_half_l13_13223


namespace find_symmetric_point_l13_13573

def slope_angle (l : ℝ → ℝ → Prop) (θ : ℝ) := ∃ m, m = Real.tan θ ∧ ∀ x y, l x y ↔ y = m * (x - 1) + 1
def passes_through (l : ℝ → ℝ → Prop) (P : ℝ × ℝ) := l P.fst P.snd
def symmetric_point (A A' : ℝ × ℝ) (l : ℝ → ℝ → Prop) := 
  (A'.snd - A.snd = A'.fst - A.fst) ∧ 
  ((A'.fst + A.fst) / 2 + (A'.snd + A.snd) / 2 - 2 = 0)

theorem find_symmetric_point :
  ∃ l : ℝ → ℝ → Prop, 
    slope_angle l (135 : ℝ) ∧ 
    passes_through l (1, 1) ∧ 
    (∀ x y, l x y ↔ x + y = 2) ∧ 
    symmetric_point (3, 4) (-2, -1) l :=
by sorry

end find_symmetric_point_l13_13573


namespace probability_fully_lit_l13_13140

-- define the conditions of the problem
def characters : List String := ["K", "y", "o", "t", "o", " ", "G", "r", "a", "n", "d", " ", "H", "o", "t", "e", "l"]

-- define the length of the sequence
def length_sequence : ℕ := characters.length

-- theorem stating the probability of seeing the fully lit sign
theorem probability_fully_lit : (1 / length_sequence) = 1 / 5 :=
by
  -- The proof is omitted
  sorry

end probability_fully_lit_l13_13140


namespace basketball_team_points_l13_13414

theorem basketball_team_points (total_points : ℕ) (number_of_players : ℕ) (points_per_player : ℕ) 
  (h1 : total_points = 18) (h2 : number_of_players = 9) : points_per_player = 2 :=
by {
  sorry -- Proof goes here
}

end basketball_team_points_l13_13414


namespace find_M_l13_13988

theorem find_M : 995 + 997 + 999 + 1001 + 1003 = 5100 - 104 :=
by 
  sorry

end find_M_l13_13988


namespace rectangles_cannot_cover_large_rectangle_l13_13090

theorem rectangles_cannot_cover_large_rectangle (n m : ℕ) (a b c d: ℕ) : 
  n = 14 → m = 9 → a = 2 → b = 3 → c = 3 → d = 2 → 
  (∀ (v_rects : ℕ) (h_rects : ℕ), v_rects = 10 → h_rects = 11 →
    (∀ (rect_area : ℕ), rect_area = n * m →
      (∀ (small_rect_area : ℕ), 
        small_rect_area = (v_rects * (a * b)) + (h_rects * (c * d)) →
        small_rect_area = rect_area → 
        false))) :=
by
  intros n_eq m_eq a_eq b_eq c_eq d_eq
       v_rects h_rects v_rects_eq h_rects_eq
       rect_area rect_area_eq small_rect_area small_rect_area_eq area_sum_eq
  sorry

end rectangles_cannot_cover_large_rectangle_l13_13090


namespace number_of_multiples_of_4_l13_13620

theorem number_of_multiples_of_4 (a b : ℤ) (h1 : 100 < a) (h2 : b < 500) (h3 : a % 4 = 0) (h4 : b % 4 = 0) : 
  ∃ n : ℤ, n = 99 :=
by
  sorry

end number_of_multiples_of_4_l13_13620


namespace pencil_length_after_sharpening_l13_13871

-- Definition of the initial length of the pencil
def initial_length : ℕ := 22

-- Definition of the amount sharpened each day
def sharpened_each_day : ℕ := 2

-- Final length of the pencil after sharpening on Monday and Tuesday
def final_length (initial_length : ℕ) (sharpened_each_day : ℕ) : ℕ :=
  initial_length - sharpened_each_day * 2

-- Theorem stating that the final length is 18 inches
theorem pencil_length_after_sharpening : final_length initial_length sharpened_each_day = 18 := by
  sorry

end pencil_length_after_sharpening_l13_13871


namespace interval_contains_solution_l13_13178

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 2

theorem interval_contains_solution :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry

end interval_contains_solution_l13_13178


namespace total_dots_correct_l13_13499

/-- Define the initial conditions -/
def monday_ladybugs : ℕ := 8
def monday_dots_per_ladybug : ℕ := 6
def tuesday_ladybugs : ℕ := 5
def wednesday_ladybugs : ℕ := 4

/-- Define the derived conditions -/
def tuesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 1
def wednesday_dots_per_ladybug : ℕ := monday_dots_per_ladybug - 2

/-- Calculate the total number of dots -/
def monday_total_dots : ℕ := monday_ladybugs * monday_dots_per_ladybug
def tuesday_total_dots : ℕ := tuesday_ladybugs * tuesday_dots_per_ladybug
def wednesday_total_dots : ℕ := wednesday_ladybugs * wednesday_dots_per_ladybug
def total_dots : ℕ := monday_total_dots + tuesday_total_dots + wednesday_total_dots

/-- Prove the total dots equal to 89 -/
theorem total_dots_correct : total_dots = 89 := by
  sorry

end total_dots_correct_l13_13499


namespace arccos_cos_8_eq_1_point_72_l13_13452

noncomputable def arccos_cos_eight : Real :=
  Real.arccos (Real.cos 8)

theorem arccos_cos_8_eq_1_point_72 : arccos_cos_eight = 1.72 :=
by
  sorry

end arccos_cos_8_eq_1_point_72_l13_13452


namespace length_of_bridge_is_l13_13587

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 21.998240140788738
noncomputable def speed_kmph : ℝ := 36
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is : bridge_length = 119.98240140788738 :=
by
  have speed_mps_val : speed_mps = 10 := by
    norm_num [speed_kmph, speed_mps]
  have total_distance_val : total_distance = 219.98240140788738 := by
    norm_num [total_distance, speed_mps_val, time_to_cross_bridge]
  have bridge_length_val : bridge_length = 119.98240140788738 := by
    norm_num [bridge_length, total_distance_val, train_length]
  exact bridge_length_val

end length_of_bridge_is_l13_13587


namespace sum_of_abs_first_10_terms_l13_13167

noncomputable def sum_of_first_n_terms (n : ℕ) : ℤ := n^2 - 5 * n + 2

theorem sum_of_abs_first_10_terms : 
  let S := sum_of_first_n_terms 10
  let S3 := sum_of_first_n_terms 3
  (S - 2 * S3) = 60 := 
by
  sorry

end sum_of_abs_first_10_terms_l13_13167


namespace tangent_line_parallel_x_axis_l13_13736

def f (x : ℝ) : ℝ := x^4 - 4 * x

theorem tangent_line_parallel_x_axis :
  ∃ (m n : ℝ), (n = f m) ∧ (deriv f m = 0) ∧ (m, n) = (1, -3) := by
  sorry

end tangent_line_parallel_x_axis_l13_13736


namespace decrement_from_each_observation_l13_13051

theorem decrement_from_each_observation (n : Nat) (mean_original mean_updated decrement : ℝ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 191)
  (h4 : decrement = 9) :
  (mean_original - mean_updated) * (n : ℝ) / n = decrement :=
by
  sorry

end decrement_from_each_observation_l13_13051


namespace find_value_of_c_l13_13790

-- Mathematical proof problem in Lean 4 statement
theorem find_value_of_c (a b c d : ℝ)
  (h1 : a + c = 900)
  (h2 : b + c = 1100)
  (h3 : a + d = 700)
  (h4 : a + b + c + d = 2000) : 
  c = 200 :=
sorry

end find_value_of_c_l13_13790


namespace fraction_meaningful_condition_l13_13069

theorem fraction_meaningful_condition (x : ℝ) : 3 - x ≠ 0 ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_condition_l13_13069


namespace jeremys_school_distance_l13_13616

def distance_to_school (rush_hour_time : ℚ) (no_traffic_time : ℚ) (speed_increase : ℚ) (distance : ℚ) : Prop :=
  ∃ v : ℚ, distance = v * rush_hour_time ∧ distance = (v + speed_increase) * no_traffic_time

theorem jeremys_school_distance :
  distance_to_school (3/10 : ℚ) (1/5 : ℚ) 20 12 :=
sorry

end jeremys_school_distance_l13_13616


namespace joe_two_different_fruits_in_a_day_l13_13276

def joe_meal_event : Type := {meal : ℕ // meal = 4}
def joe_fruit_choice : Type := {fruit : ℕ // fruit ≤ 4}

noncomputable def prob_all_same_fruit : ℚ := (1 / 4) ^ 4 * 4
noncomputable def prob_at_least_two_diff_fruits : ℚ := 1 - prob_all_same_fruit

theorem joe_two_different_fruits_in_a_day :
  prob_at_least_two_diff_fruits = 63 / 64 :=
by
  sorry

end joe_two_different_fruits_in_a_day_l13_13276


namespace function_D_is_odd_function_D_is_decreasing_l13_13494

def f_D (x : ℝ) : ℝ := -x * |x|

theorem function_D_is_odd (x : ℝ) : f_D (-x) = -f_D x := by
  sorry

theorem function_D_is_decreasing (x y : ℝ) (h : x < y) : f_D x > f_D y := by
  sorry

end function_D_is_odd_function_D_is_decreasing_l13_13494


namespace exists_natural_numbers_with_digit_sum_condition_l13_13923

def digit_sum (x : ℕ) : ℕ :=
  x.digits 10 |>.sum

theorem exists_natural_numbers_with_digit_sum_condition :
  ∃ (a b c : ℕ), digit_sum (a + b) < 5 ∧ digit_sum (a + c) < 5 ∧ digit_sum (b + c) < 5 ∧ digit_sum (a + b + c) > 50 :=
by
  sorry

end exists_natural_numbers_with_digit_sum_condition_l13_13923


namespace max_last_digit_of_sequence_l13_13328

theorem max_last_digit_of_sequence :
  ∀ (s : Fin 1001 → ℕ), 
  (s 0 = 2) →
  (∀ (i : Fin 1000), (s i) * 10 + (s i.succ) ∈ {n | n % 17 = 0 ∨ n % 23 = 0}) →
  ∃ (d : ℕ), (d = s ⟨1000, sorry⟩) ∧ (∀ (d' : ℕ), d' = s ⟨1000, sorry⟩ → d' ≤ d) ∧ (d = 2) :=
by
  intros s h1 h2
  use 2
  sorry

end max_last_digit_of_sequence_l13_13328


namespace loss_percentage_is_26_l13_13774

/--
Given the cost price of a radio is Rs. 1500 and the selling price is Rs. 1110, 
prove that the loss percentage is 26%
-/
theorem loss_percentage_is_26 (cost_price selling_price : ℝ)
  (h₀ : cost_price = 1500)
  (h₁ : selling_price = 1110) :
  ((cost_price - selling_price) / cost_price) * 100 = 26 := 
by 
  sorry

end loss_percentage_is_26_l13_13774


namespace prime_sum_exists_even_n_l13_13458

theorem prime_sum_exists_even_n (n : ℕ) :
  (∃ a b c : ℤ, a + b + c = 0 ∧ Prime (a^n + b^n + c^n)) ↔ Even n := 
by
  sorry

end prime_sum_exists_even_n_l13_13458


namespace amount_a_receives_l13_13645

theorem amount_a_receives (a b c : ℕ) (h1 : a + b + c = 50000) (h2 : a = b + 4000) (h3 : b = c + 5000) :
  (21000 / 50000) * 36000 = 15120 :=
by
  sorry

end amount_a_receives_l13_13645


namespace select_defective_products_l13_13562

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem select_defective_products :
  let total_products := 200
  let defective_products := 3
  let selected_products := 5
  let ways_2_defective := choose defective_products 2 * choose (total_products - defective_products) 3
  let ways_3_defective := choose defective_products 3 * choose (total_products - defective_products) 2
  ways_2_defective + ways_3_defective = choose defective_products 2 * choose (total_products - defective_products) 3 + choose defective_products 3 * choose (total_products - defective_products) 2 :=
by
  sorry

end select_defective_products_l13_13562


namespace muffins_in_morning_l13_13703

variable (M : ℕ)

-- Conditions
def goal : ℕ := 20
def afternoon_sales : ℕ := 4
def additional_needed : ℕ := 4
def morning_sales (M : ℕ) : ℕ := M

-- Proof statement (no need to prove here, just state it)
theorem muffins_in_morning :
  morning_sales M + afternoon_sales + additional_needed = goal → M = 12 :=
sorry

end muffins_in_morning_l13_13703


namespace speed_of_current_l13_13047

-- Definitions of the given conditions
def downstream_time := 6 / 60 -- time in hours to travel 1 km downstream
def upstream_time := 10 / 60 -- time in hours to travel 1 km upstream

-- Definition of speeds
def downstream_speed := 1 / downstream_time -- speed in km/h downstream
def upstream_speed := 1 / upstream_time -- speed in km/h upstream

-- Theorem statement
theorem speed_of_current : 
  (downstream_speed - upstream_speed) / 2 = 2 := 
by 
  -- We skip the proof for now
  sorry

end speed_of_current_l13_13047


namespace increasing_exponential_function_l13_13986

theorem increasing_exponential_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → (a ^ x) < (a ^ y)) → (1 < a) :=
by
  sorry

end increasing_exponential_function_l13_13986


namespace range_of_a_l13_13037

noncomputable def inequality_always_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (a : ℝ) : inequality_always_holds a ↔ 0 ≤ a ∧ a < 1 := 
by
  sorry

end range_of_a_l13_13037


namespace min_value_fraction_l13_13852

theorem min_value_fraction {a : ℕ → ℕ} (h1 : a 1 = 10)
    (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) :
    ∃ n : ℕ, (n > 0) ∧ (n - 1 + 10 / n = 16 / 3) :=
by {
  sorry
}

end min_value_fraction_l13_13852


namespace solve_for_x_l13_13768

/-- Let f(x) = 2 - 1 / (2 - x)^3.
Proof that f(x) = 1 / (2 - x)^3 implies x = 1. -/
theorem solve_for_x (x : ℝ) (h : 2 - 1 / (2 - x)^3 = 1 / (2 - x)^3) : x = 1 :=
  sorry

end solve_for_x_l13_13768


namespace average_rate_of_change_interval_l13_13896

noncomputable def average_rate_of_change (f : ℝ → ℝ) (x₀ x₁ : ℝ) : ℝ :=
  (f x₁ - f x₀) / (x₁ - x₀)

theorem average_rate_of_change_interval (f : ℝ → ℝ) (x₀ x₁ : ℝ) :
  (f x₁ - f x₀) / (x₁ - x₀) = average_rate_of_change f x₀ x₁ := by
  sorry

end average_rate_of_change_interval_l13_13896


namespace shenzhen_vaccination_count_l13_13627

theorem shenzhen_vaccination_count :
  2410000 = 2.41 * 10^6 :=
  sorry

end shenzhen_vaccination_count_l13_13627


namespace total_payment_l13_13016

theorem total_payment (manicure_cost : ℚ) (tip_percentage : ℚ) (h_manicure_cost : manicure_cost = 30) (h_tip_percentage : tip_percentage = 30) : 
  manicure_cost + (tip_percentage / 100) * manicure_cost = 39 := 
by 
  sorry

end total_payment_l13_13016


namespace system_of_equations_correct_l13_13327

theorem system_of_equations_correct (x y : ℝ) (h1 : x + y = 2000) (h2 : y = x * 0.30) :
  x + y = 2000 ∧ y = x * 0.30 :=
by 
  exact ⟨h1, h2⟩

end system_of_equations_correct_l13_13327


namespace dennis_years_taught_l13_13163

theorem dennis_years_taught (A V D : ℕ) (h1 : V + A + D = 75) (h2 : V = A + 9) (h3 : V = D - 9) : D = 34 :=
sorry

end dennis_years_taught_l13_13163


namespace smallest_base10_integer_l13_13551

-- Definitions of the integers a and b as bases larger than 3.
variables {a b : ℕ}

-- Definitions of the base-10 representation of the given numbers.
def thirteen_in_a (a : ℕ) : ℕ := 1 * a + 3
def thirty_one_in_b (b : ℕ) : ℕ := 3 * b + 1

-- The proof statement.
theorem smallest_base10_integer (h₁ : a > 3) (h₂ : b > 3) :
  (∃ (n : ℕ), thirteen_in_a a = n ∧ thirty_one_in_b b = n) → ∃ n, n = 13 :=
by
  sorry

end smallest_base10_integer_l13_13551


namespace analytical_expression_when_x_in_5_7_l13_13393

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry
lemma symmetric_about_one (x : ℝ) : f (1 - x) = f (1 + x) := sorry
lemma values_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x ≤ 1) : f x = x := sorry

theorem analytical_expression_when_x_in_5_7 (x : ℝ) (h : 5 < x ∧ x ≤ 7) :
  f x = 6 - x :=
sorry

end analytical_expression_when_x_in_5_7_l13_13393


namespace arithmetic_mean_end_number_l13_13639

theorem arithmetic_mean_end_number (n : ℤ) :
  (100 + n) / 2 = 150 + 100 → n = 400 := by
  sorry

end arithmetic_mean_end_number_l13_13639


namespace disloyal_bound_l13_13392

variable {p n : ℕ}

/-- A number is disloyal if its GCD with n is not 1 -/
def isDisloyal (x : ℕ) (n : ℕ) := Nat.gcd x n ≠ 1

theorem disloyal_bound (p : ℕ) (n : ℕ) (hp : p.Prime) (hn : n % p^2 = 0) :
  (∃ D : Finset ℕ, (∀ x ∈ D, isDisloyal x n) ∧ D.card ≤ (n - 1) / p) :=
sorry

end disloyal_bound_l13_13392


namespace plan_A_fee_eq_nine_l13_13036

theorem plan_A_fee_eq_nine :
  ∃ F : ℝ, (0.25 * 60 + F = 0.40 * 60) ∧ (F = 9) :=
by
  sorry

end plan_A_fee_eq_nine_l13_13036


namespace instantaneous_acceleration_at_1_second_l13_13467

-- Assume the velocity function v(t) is given as:
def v (t : ℝ) : ℝ := t^2 + 2 * t + 3

-- We need to prove that the instantaneous acceleration at t = 1 second is 4 m/s^2.
theorem instantaneous_acceleration_at_1_second : 
  deriv v 1 = 4 :=
by 
  sorry

end instantaneous_acceleration_at_1_second_l13_13467


namespace complement_M_l13_13426

open Set

-- Definitions and conditions
def U : Set ℝ := univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- Theorem stating the complement of M with respect to the universal set U
theorem complement_M : compl M = {x | x < -2 ∨ x > 2} :=
by
  sorry

end complement_M_l13_13426


namespace baskets_containing_neither_l13_13028

-- Definitions representing the conditions
def total_baskets : ℕ := 15
def baskets_with_apples : ℕ := 10
def baskets_with_oranges : ℕ := 8
def baskets_with_both : ℕ := 5

-- Theorem statement to prove the number of baskets containing neither apples nor oranges
theorem baskets_containing_neither : total_baskets - (baskets_with_apples + baskets_with_oranges - baskets_with_both) = 2 :=
by
  sorry

end baskets_containing_neither_l13_13028


namespace sum_of_series_eq_half_l13_13793

theorem sum_of_series_eq_half :
  (∑' k : ℕ, 3^(2^k) / (9^(2^k) - 1)) = 1 / 2 :=
by
  sorry

end sum_of_series_eq_half_l13_13793


namespace ball_hits_ground_time_l13_13656

noncomputable def find_time_when_ball_hits_ground (a b c : ℝ) : ℝ :=
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)

theorem ball_hits_ground_time :
  find_time_when_ball_hits_ground (-16) 40 50 = (5 + 5 * Real.sqrt 3) / 4 :=
by
  sorry

end ball_hits_ground_time_l13_13656


namespace carrots_weight_l13_13597

theorem carrots_weight (carrots_bed1: ℕ) (carrots_bed2: ℕ) (carrots_bed3: ℕ) (carrots_per_pound: ℕ)
  (h_bed1: carrots_bed1 = 55)
  (h_bed2: carrots_bed2 = 101)
  (h_bed3: carrots_bed3 = 78)
  (h_c_per_p: carrots_per_pound = 6) :
  (carrots_bed1 + carrots_bed2 + carrots_bed3) / carrots_per_pound = 39 := by
  sorry

end carrots_weight_l13_13597


namespace sequence_le_zero_l13_13444

noncomputable def sequence_property (N : ℕ) (a : ℕ → ℝ) : Prop :=
  (a 0 = 0) ∧ (a N = 0) ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2)

theorem sequence_le_zero {N : ℕ} (a : ℕ → ℝ) (h : sequence_property N a) : 
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 :=
sorry

end sequence_le_zero_l13_13444


namespace continuous_piecewise_function_l13_13082

theorem continuous_piecewise_function (a c : ℝ) (h1 : 2 * a * 2 + 6 = 3 * 2 - 2) (h2 : 4 * (-2) + 2 * c = 3 * (-2) - 2) : 
  a + c = -1/2 := 
sorry

end continuous_piecewise_function_l13_13082


namespace initial_amount_of_money_l13_13756

-- Definitions based on conditions in a)
variables (n : ℚ) -- Bert left the house with n dollars
def after_hardware_store := (3 / 4) * n
def after_dry_cleaners := after_hardware_store - 9
def after_grocery_store := (1 / 2) * after_dry_cleaners
def after_bookstall := (2 / 3) * after_grocery_store
def after_donation := (4 / 5) * after_bookstall

-- Theorem statement
theorem initial_amount_of_money : after_donation = 27 → n = 72 :=
by
  sorry

end initial_amount_of_money_l13_13756


namespace roots_equal_condition_l13_13894

theorem roots_equal_condition (a c : ℝ) (h : a ≠ 0) :
    (∀ x1 x2, (a * x1 * x1 + 4 * a * x1 + c = 0) ∧ (a * x2 * x2 + 4 * a * x2 + c = 0) → x1 = x2) ↔ c = 4 * a := 
by
  sorry

end roots_equal_condition_l13_13894


namespace geom_sequence_arith_ratio_l13_13667

variable (a : ℕ → ℝ) (q : ℝ)
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_arith : 3 * a 0 + 2 * a 1 = 2 * (1/2) * a 2)

theorem geom_sequence_arith_ratio (ha : 3 * a 0 + 2 * a 1 = a 2) :
    (a 8 + a 9) / (a 6 + a 7) = 9 := sorry

end geom_sequence_arith_ratio_l13_13667


namespace total_cupcakes_l13_13593

theorem total_cupcakes (children : ℕ) (cupcakes_per_child : ℕ) (total_cupcakes : ℕ) 
  (h1 : children = 8) (h2 : cupcakes_per_child = 12) : total_cupcakes = 96 := 
by
  sorry

end total_cupcakes_l13_13593


namespace interval_of_a_l13_13040

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then Real.exp x + x^2 else Real.exp (-x) + x^2

theorem interval_of_a (a : ℝ) :
  f (-a) + f a ≤ 2 * f 1 → -1 ≤ a ∧ a ≤ 1 :=
sorry

end interval_of_a_l13_13040


namespace squirrel_spring_acorns_l13_13202

/--
A squirrel had stashed 210 acorns to last him the three winter months. 
It divided the pile into thirds, one for each month, and then took some 
from each third, leaving 60 acorns for each winter month. The squirrel 
combined the ones it took to eat in the first cold month of spring. 
Prove that the number of acorns the squirrel has for the beginning of spring 
is 30.
-/
theorem squirrel_spring_acorns :
  ∀ (initial_acorns acorns_per_month remaining_acorns_per_month acorns_taken_per_month : ℕ),
    initial_acorns = 210 →
    acorns_per_month = initial_acorns / 3 →
    remaining_acorns_per_month = 60 →
    acorns_taken_per_month = acorns_per_month - remaining_acorns_per_month →
    3 * acorns_taken_per_month = 30 :=
by
  intros initial_acorns acorns_per_month remaining_acorns_per_month acorns_taken_per_month
  sorry

end squirrel_spring_acorns_l13_13202


namespace christine_needs_32_tbs_aquafaba_l13_13547

-- Definitions for the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

def total_egg_whites : ℕ := egg_whites_per_cake * number_of_cakes
def total_tbs_aquafaba : ℕ := tablespoons_per_egg_white * total_egg_whites

-- Theorem statement
theorem christine_needs_32_tbs_aquafaba :
  total_tbs_aquafaba = 32 :=
by sorry

end christine_needs_32_tbs_aquafaba_l13_13547


namespace minimum_value_of_f_l13_13606

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x + 5) + abs (x + 6)

theorem minimum_value_of_f : ∃ x : ℝ, f x = 1 :=
by sorry

end minimum_value_of_f_l13_13606


namespace edge_ratio_of_cubes_l13_13428

theorem edge_ratio_of_cubes (a b : ℝ) (h : (a^3) / (b^3) = 64) : a / b = 4 :=
sorry

end edge_ratio_of_cubes_l13_13428


namespace sequence_a_correct_l13_13533

open Nat -- Opening the natural numbers namespace

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => (1 / 2 : ℝ) * a n

theorem sequence_a_correct : 
  (∀ n, 0 < a n) ∧ 
  a 1 = 1 ∧ 
  (∀ n, a (n + 1) = a n / 2) ∧
  a 2 = 1 / 2 ∧
  a 3 = 1 / 4 ∧
  ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end sequence_a_correct_l13_13533


namespace arithmetic_sequence_z_l13_13515

-- Define the arithmetic sequence and value of z
theorem arithmetic_sequence_z (z : ℤ) (arith_seq : 9 + 27 = 2 * z) : z = 18 := 
by 
  sorry

end arithmetic_sequence_z_l13_13515


namespace cube_relation_l13_13096

theorem cube_relation (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cube_relation_l13_13096


namespace complete_square_l13_13256

theorem complete_square 
  (x : ℝ) : 
  (2 * x^2 - 3 * x - 1 = 0) → 
  ((x - (3/4))^2 = (17/16)) :=
sorry

end complete_square_l13_13256


namespace sister_age_is_one_l13_13476

variable (B S : ℕ)

theorem sister_age_is_one (h : B = B * S) : S = 1 :=
by {
  sorry
}

end sister_age_is_one_l13_13476


namespace probability_of_sequence_l13_13012

theorem probability_of_sequence :
  let total_cards := 52
  let face_cards := 12
  let hearts := 13
  let first_card_face_prob := (face_cards : ℝ) / total_cards
  let second_card_heart_prob := (10 : ℝ) / (total_cards - 1)
  let third_card_face_prob := (11 : ℝ) / (total_cards - 2)
  let total_prob := first_card_face_prob * second_card_heart_prob * third_card_face_prob
  total_prob = 1 / 100.455 :=
by
  sorry

end probability_of_sequence_l13_13012


namespace INPUT_is_input_statement_l13_13134

-- Define what constitutes each type of statement
def isOutputStatement (stmt : String) : Prop :=
  stmt = "PRINT"

def isInputStatement (stmt : String) : Prop :=
  stmt = "INPUT"

def isConditionalStatement (stmt : String) : Prop :=
  stmt = "THEN"

def isEndStatement (stmt : String) : Prop :=
  stmt = "END"

-- The main theorem
theorem INPUT_is_input_statement : isInputStatement "INPUT" := by
  sorry

end INPUT_is_input_statement_l13_13134


namespace range_of_a_l13_13787

noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x - a

theorem range_of_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f x a = 0 ∧ f y a = 0 ∧ f z a = 0) ↔ (-9 < a ∧ a < 5/3) :=
by
  sorry

end range_of_a_l13_13787


namespace tumblonian_words_count_l13_13956

def numTumblonianWords : ℕ :=
  let alphabet_size := 6
  let max_word_length := 4
  let num_words n := alphabet_size ^ n
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4)

theorem tumblonian_words_count : numTumblonianWords = 1554 := by
  sorry

end tumblonian_words_count_l13_13956


namespace intersect_range_k_l13_13283

theorem intersect_range_k : 
  ∀ k : ℝ, (∃ x y : ℝ, x^2 - (kx + 2)^2 = 6) ↔ 
  -Real.sqrt (5 / 3) < k ∧ k < Real.sqrt (5 / 3) := 
by sorry

end intersect_range_k_l13_13283


namespace percentage_of_earrings_l13_13457

theorem percentage_of_earrings (B M R : ℕ) (hB : B = 10) (hM : M = 2 * R) (hTotal : B + M + R = 70) : 
  (B * 100) / M = 25 := 
by
  sorry

end percentage_of_earrings_l13_13457


namespace max_consecutive_sum_l13_13713

theorem max_consecutive_sum (a N : ℤ) (h₀ : N > 0) (h₁ : N * (2 * a + N - 1) = 90) : N = 90 :=
by
  -- Proof to be provided
  sorry

end max_consecutive_sum_l13_13713


namespace odd_times_even_is_even_l13_13858

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_times_even_is_even (a b : ℤ) (h₁ : is_odd a) (h₂ : is_even b) : is_even (a * b) :=
by sorry

end odd_times_even_is_even_l13_13858


namespace gcd_78_182_l13_13576

theorem gcd_78_182 : Nat.gcd 78 182 = 26 := by
  sorry

end gcd_78_182_l13_13576


namespace smallest_sum_of_18_consecutive_integers_is_perfect_square_l13_13155

theorem smallest_sum_of_18_consecutive_integers_is_perfect_square 
  (n : ℕ) 
  (S : ℕ) 
  (h1 : S = 9 * (2 * n + 17)) 
  (h2 : ∃ k : ℕ, 2 * n + 17 = k^2) 
  (h3 : ∀ m : ℕ, m < 5 → 2 * n + 17 ≠ m^2) : 
  S = 225 := 
by
  sorry

end smallest_sum_of_18_consecutive_integers_is_perfect_square_l13_13155


namespace hot_dogs_served_for_dinner_l13_13451

theorem hot_dogs_served_for_dinner
  (l t : ℕ) 
  (h_cond1 : l = 9) 
  (h_cond2 : t = 11) :
  ∃ d : ℕ, d = t - l ∧ d = 2 := by
  sorry

end hot_dogs_served_for_dinner_l13_13451


namespace original_number_is_0_2_l13_13285

theorem original_number_is_0_2 :
  ∃ x : ℝ, (1 / (1 / x - 1) - 1 = -0.75) ∧ x = 0.2 :=
by
  sorry

end original_number_is_0_2_l13_13285


namespace geometric_sequence_a3_l13_13210

theorem geometric_sequence_a3 (
  a : ℕ → ℝ
) 
(h1 : a 1 = 1)
(h5 : a 5 = 16)
(h_geometric : ∀ (n : ℕ), a (n + 1) / a n = a 2 / a 1) :
a 3 = 4 := by
  sorry

end geometric_sequence_a3_l13_13210


namespace initial_water_amount_gallons_l13_13274

theorem initial_water_amount_gallons 
  (cup_capacity_oz : ℕ)
  (rows : ℕ)
  (chairs_per_row : ℕ)
  (water_left_oz : ℕ)
  (oz_per_gallon : ℕ)
  (total_gallons : ℕ)
  (h1 : cup_capacity_oz = 6)
  (h2 : rows = 5)
  (h3 : chairs_per_row = 10)
  (h4 : water_left_oz = 84)
  (h5 : oz_per_gallon = 128)
  (h6 : total_gallons = (rows * chairs_per_row * cup_capacity_oz + water_left_oz) / oz_per_gallon) :
  total_gallons = 3 := 
by sorry

end initial_water_amount_gallons_l13_13274


namespace condition_of_inequality_l13_13399

theorem condition_of_inequality (x y : ℝ) (h : x^2 + y^2 ≤ 2 * (x + y - 1)) : x = 1 ∧ y = 1 :=
by
  sorry

end condition_of_inequality_l13_13399


namespace student_marks_equals_125_l13_13672

-- Define the maximum marks
def max_marks : ℕ := 500

-- Define the percentage required to pass
def pass_percentage : ℚ := 33 / 100

-- Define the marks required to pass
def pass_marks : ℚ := pass_percentage * max_marks

-- Define the marks by which the student failed
def fail_by_marks : ℕ := 40

-- Define the obtained marks by the student
def obtained_marks : ℚ := pass_marks - fail_by_marks

-- Prove that the obtained marks are 125
theorem student_marks_equals_125 : obtained_marks = 125 := by
  sorry

end student_marks_equals_125_l13_13672


namespace bhishma_speed_l13_13255

-- Given definitions based on conditions
def track_length : ℝ := 600
def bruce_speed : ℝ := 30
def time_meet : ℝ := 90

-- Main theorem we want to prove
theorem bhishma_speed : ∃ v : ℝ, v = 23.33 ∧ (bruce_speed * time_meet) = (v * time_meet + track_length) :=
  by
    sorry

end bhishma_speed_l13_13255


namespace isosceles_right_triangle_ratio_l13_13084

theorem isosceles_right_triangle_ratio {a : ℝ} (h_pos : 0 < a) :
  (a + 2 * a) / Real.sqrt (a^2 + a^2) = 3 * Real.sqrt 2 / 2 :=
sorry

end isosceles_right_triangle_ratio_l13_13084


namespace min_S6_minus_S4_l13_13834

variable {a₁ a₂ q : ℝ} (h1 : q > 1) (h2 : (q^2 - 1) * (a₁ + a₂) = 3)

theorem min_S6_minus_S4 : 
  ∃ (a₁ a₂ q : ℝ), q > 1 ∧ (q^2 - 1) * (a₁ + a₂) = 3 ∧ (q^4 * (a₁ + a₂) - (a₁ + a₂ + a₂ * q + a₂ * q^2) = 12) := sorry

end min_S6_minus_S4_l13_13834


namespace composite_2011_2014_composite_2012_2015_l13_13275

theorem composite_2011_2014 :
  let N := 2011 * 2012 * 2013 * 2014 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2011 * 2012 * 2013 * 2014 + 1
  sorry
  
theorem composite_2012_2015 :
  let N := 2012 * 2013 * 2014 * 2015 + 1
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ N = a * b := 
by
  let N := 2012 * 2013 * 2014 * 2015 + 1
  sorry

end composite_2011_2014_composite_2012_2015_l13_13275


namespace digits_difference_l13_13482

theorem digits_difference (d A B : ℕ) (h1 : d > 6) (h2 : (B + A) * d + 2 * A = d^2 + 7 * d + 2)
  (h3 : B + A = 10) (h4 : 2 * A = 8) : A - B = 3 :=
by 
  sorry

end digits_difference_l13_13482


namespace area_shaded_region_is_75_l13_13419

-- Define the side length of the larger square
def side_length_large_square : ℝ := 10

-- Define the side length of the smaller square
def side_length_small_square : ℝ := 5

-- Define the area of the larger square
def area_large_square : ℝ := side_length_large_square ^ 2

-- Define the area of the smaller square
def area_small_square : ℝ := side_length_small_square ^ 2

-- Define the area of the shaded region
def area_shaded_region : ℝ := area_large_square - area_small_square

-- The theorem that states the area of the shaded region is 75 square units
theorem area_shaded_region_is_75 : area_shaded_region = 75 := by
  -- The proof will be filled in here when required
  sorry

end area_shaded_region_is_75_l13_13419


namespace min_races_needed_l13_13844

noncomputable def minimum_races (total_horses : ℕ) (max_race_horses : ℕ) : ℕ :=
  if total_horses ≤ max_race_horses then 1 else
  if total_horses % max_race_horses = 0 then total_horses / max_race_horses else total_horses / max_race_horses + 1

/-- We need to show that the minimum number of races required to find the top 3 fastest horses
    among 35 horses, where a maximum of 4 horses can race together at a time, is 10. -/
theorem min_races_needed : minimum_races 35 4 = 10 :=
  sorry

end min_races_needed_l13_13844


namespace possible_teams_count_l13_13072

-- Defining the problem
def team_group_division : Prop :=
  ∃ (g1 g2 g3 g4 : ℕ), (g1 ≥ 2) ∧ (g2 ≥ 2) ∧ (g3 ≥ 2) ∧ (g4 ≥ 2) ∧
  (66 = (g1 * (g1 - 1) / 2) + (g2 * (g2 - 1) / 2) + (g3 * (g3 - 1) / 2) + 
       (g4 * (g4 - 1) / 2)) ∧ 
  ((g1 + g2 + g3 + g4 = 21) ∨ (g1 + g2 + g3 + g4 = 22) ∨ 
   (g1 + g2 + g3 + g4 = 23) ∨ (g1 + g2 + g3 + g4 = 24) ∨ 
   (g1 + g2 + g3 + g4 = 25))

-- Theorem statement to prove
theorem possible_teams_count : team_group_division :=
sorry

end possible_teams_count_l13_13072


namespace average_weight_l13_13530

theorem average_weight (men women : ℕ) (avg_weight_men avg_weight_women : ℝ) (total_people : ℕ) (combined_avg_weight : ℝ) 
  (h1 : men = 8) (h2 : avg_weight_men = 190) (h3 : women = 6) (h4 : avg_weight_women = 120) (h5 : total_people = 14) 
  (h6 : (men * avg_weight_men + women * avg_weight_women) / total_people = combined_avg_weight) : combined_avg_weight = 160 := 
  sorry

end average_weight_l13_13530


namespace union_of_sets_l13_13811

def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5, 7}
def union_result : Set ℕ := {1, 3, 5, 7}

theorem union_of_sets : A ∪ B = union_result := by
  sorry

end union_of_sets_l13_13811


namespace remainder_when_dividing_n_by_d_l13_13038

def n : ℕ := 25197638
def d : ℕ := 4
def r : ℕ := 2

theorem remainder_when_dividing_n_by_d :
  n % d = r :=
by
  sorry

end remainder_when_dividing_n_by_d_l13_13038


namespace min_value_fraction_l13_13702

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) = 2) : 
  ∃ x : ℝ, x = (8 * a + b) / (a * b) ∧ x = 9 :=
by
  sorry

end min_value_fraction_l13_13702


namespace total_capsules_in_july_l13_13706

theorem total_capsules_in_july : 
  let mondays := 4
  let tuesdays := 5
  let wednesdays := 5
  let thursdays := 4
  let fridays := 4
  let saturdays := 4
  let sundays := 5

  let capsules_monday := mondays * 2
  let capsules_tuesday := tuesdays * 3
  let capsules_wednesday := wednesdays * 2
  let capsules_thursday := thursdays * 3
  let capsules_friday := fridays * 2
  let capsules_saturday := saturdays * 4
  let capsules_sunday := sundays * 4

  let total_capsules := capsules_monday + capsules_tuesday + capsules_wednesday + capsules_thursday + capsules_friday + capsules_saturday + capsules_sunday

  let missed_capsules_tuesday := 3
  let missed_capsules_sunday := 4

  let total_missed_capsules := missed_capsules_tuesday + missed_capsules_sunday

  let total_consumed_capsules := total_capsules - total_missed_capsules
  total_consumed_capsules = 82 := 
by
  -- Details omitted, proof goes here
  sorry

end total_capsules_in_july_l13_13706


namespace opposite_sides_line_l13_13194

theorem opposite_sides_line (m : ℝ) :
  ( (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ) → (-7 < m ∧ m < 24) :=
by sorry

end opposite_sides_line_l13_13194


namespace isosceles_triangle_largest_angle_l13_13738

theorem isosceles_triangle_largest_angle (A B C : ℝ)
  (h1 : A = B)
  (h2 : C = 50) :
  A + B + C = 180 →
  C = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l13_13738


namespace commute_time_absolute_difference_l13_13596

theorem commute_time_absolute_difference (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : (x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end commute_time_absolute_difference_l13_13596


namespace division_result_l13_13026

def m : ℕ := 16 ^ 2024

theorem division_result : m / 8 = 8 * 16 ^ 2020 :=
by
  -- sorry for the actual proof
  sorry

end division_result_l13_13026


namespace soda_cost_l13_13979

-- Definitions based on conditions of the problem
variable (b s : ℤ)
variable (h1 : 4 * b + 3 * s = 540)
variable (h2 : 3 * b + 2 * s = 390)

-- The theorem to prove the cost of a soda
theorem soda_cost : s = 60 := by
  sorry

end soda_cost_l13_13979


namespace barbara_candies_left_l13_13863

def initial_candies: ℝ := 18.5
def candies_used_to_make_dessert: ℝ := 4.2
def candies_received_from_friend: ℝ := 6.8
def candies_eaten: ℝ := 2.7

theorem barbara_candies_left : 
  initial_candies - candies_used_to_make_dessert + candies_received_from_friend - candies_eaten = 18.4 := 
by
  sorry

end barbara_candies_left_l13_13863


namespace inequality_problem_l13_13975

-- Given a < b < 0, we want to prove a^2 > ab > b^2
theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
sorry

end inequality_problem_l13_13975


namespace number_of_small_pizzas_ordered_l13_13122

-- Define the problem conditions
def benBrothers : Nat := 2
def slicesPerPerson : Nat := 12
def largePizzaSlices : Nat := 14
def smallPizzaSlices : Nat := 8
def numLargePizzas : Nat := 2

-- Define the statement to prove
theorem number_of_small_pizzas_ordered : 
  ∃ (s : Nat), (benBrothers + 1) * slicesPerPerson - numLargePizzas * largePizzaSlices = s * smallPizzaSlices ∧ s = 1 :=
by
  sorry

end number_of_small_pizzas_ordered_l13_13122


namespace vehicle_speed_increase_l13_13266

/-- Vehicle dynamics details -/
structure Vehicle := 
  (initial_speed : ℝ) 
  (deceleration : ℝ)
  (initial_distance_from_A : ℝ)

/-- Given conditions -/
def conditions (A B C : Vehicle) : Prop :=
  A.initial_speed = 80 ∧
  B.initial_speed = 60 ∧
  C.initial_speed = 70 ∧ 
  C.deceleration = 2 ∧
  B.initial_distance_from_A = 40 ∧
  C.initial_distance_from_A = 260

/-- Prove A needs to increase its speed by 5 mph -/
theorem vehicle_speed_increase (A B C : Vehicle) (h : conditions A B C) : 
  ∃ dA : ℝ, dA = 5 ∧ A.initial_speed + dA > B.initial_speed → 
    (A.initial_distance_from_A / (A.initial_speed + dA - B.initial_speed)) < 
    (C.initial_distance_from_A / (A.initial_speed + dA + C.initial_speed - C.deceleration)) :=
sorry

end vehicle_speed_increase_l13_13266


namespace gloves_needed_l13_13615

theorem gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) (total_gloves : ℕ)
  (h1 : participants = 82)
  (h2 : gloves_per_participant = 2)
  (h3 : total_gloves = participants * gloves_per_participant) :
  total_gloves = 164 :=
by
  sorry

end gloves_needed_l13_13615


namespace quadratic_function_vertex_upwards_exists_l13_13754

theorem quadratic_function_vertex_upwards_exists :
  ∃ (a : ℝ), a > 0 ∧ ∃ (f : ℝ → ℝ), (∀ x, f x = a * (x - 1) * (x - 1) - 2) :=
by
  sorry

end quadratic_function_vertex_upwards_exists_l13_13754


namespace smallest_positive_integer_cube_ends_368_l13_13496

theorem smallest_positive_integer_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 368 ∧ n = 34 :=
by
  sorry

end smallest_positive_integer_cube_ends_368_l13_13496


namespace quad_eq_sum_ab_l13_13578

theorem quad_eq_sum_ab {a b : ℝ} (h1 : a < 0)
  (h2 : ∀ x : ℝ, (x = -1 / 2 ∨ x = 1 / 3) ↔ ax^2 + bx + 2 = 0) :
  a + b = -14 :=
by
  sorry

end quad_eq_sum_ab_l13_13578


namespace nancy_clay_pots_l13_13763

theorem nancy_clay_pots : 
  ∃ M : ℕ, (M + 2 * M + 14 = 50) ∧ M = 12 :=
sorry

end nancy_clay_pots_l13_13763


namespace expression_equals_100_l13_13325

-- Define the terms in the numerator and their squares
def num1 := 0.02
def num2 := 0.52
def num3 := 0.035

def num1_sq := num1^2
def num2_sq := num2^2
def num3_sq := num3^2

-- Define the terms in the denominator and their squares
def denom1 := 0.002
def denom2 := 0.052
def denom3 := 0.0035

def denom1_sq := denom1^2
def denom2_sq := denom2^2
def denom3_sq := denom3^2

-- Define the sums of the squares
def sum_numerator := num1_sq + num2_sq + num3_sq
def sum_denominator := denom1_sq + denom2_sq + denom3_sq

-- Define the final expression
def expression := sum_numerator / sum_denominator

-- Prove the expression equals the correct answer
theorem expression_equals_100 : expression = 100 := by sorry

end expression_equals_100_l13_13325


namespace differentiable_implies_continuous_l13_13689

-- Theorem: If a function f is differentiable at x0, then it is continuous at x0.
theorem differentiable_implies_continuous {f : ℝ → ℝ} {x₀ : ℝ} (h : DifferentiableAt ℝ f x₀) : 
  ContinuousAt f x₀ :=
sorry

end differentiable_implies_continuous_l13_13689


namespace compute_expression_l13_13925

theorem compute_expression (x y : ℝ) (hx : 1/x + 1/y = 4) (hy : x*y + x + y = 5) : 
  x^2 * y + x * y^2 + x^2 + y^2 = 18 := 
by 
  -- Proof goes here 
  sorry

end compute_expression_l13_13925


namespace cube_surface_area_l13_13439

theorem cube_surface_area (v : ℝ) (h : v = 1000) : ∃ (s : ℝ), s^3 = v ∧ 6 * s^2 = 600 :=
by
  sorry

end cube_surface_area_l13_13439


namespace functional_equation_solution_l13_13310

noncomputable def f (x : ℚ) : ℚ := sorry

theorem functional_equation_solution (f : ℚ → ℚ) (f_pos_rat : ∀ x : ℚ, 0 < x → 0 < f x) :
  (∀ x y : ℚ, 0 < x → 0 < y → f x + f y + 2 * x * y * f (x * y) = f (x * y) / f (x + y)) →
  (∀ x : ℚ, 0 < x → f x = 1 / x ^ 2) :=
by
  sorry

end functional_equation_solution_l13_13310


namespace num_real_solutions_abs_eq_l13_13895

theorem num_real_solutions_abs_eq :
  (∃ x y : ℝ, x ≠ y ∧ |x-1| = |x-2| + |x-3| + |x-4| 
    ∧ |y-1| = |y-2| + |y-3| + |y-4| 
    ∧ ∀ z : ℝ, |z-1| = |z-2| + |z-3| + |z-4| → (z = x ∨ z = y)) := sorry

end num_real_solutions_abs_eq_l13_13895


namespace correct_operation_l13_13263

theorem correct_operation (a b : ℝ) : a * b^2 - b^2 * a = 0 := by
  sorry

end correct_operation_l13_13263


namespace congruence_solutions_count_number_of_solutions_l13_13435

theorem congruence_solutions_count (x : ℕ) (hx_pos : x > 0) (hx_lt : x < 200) :
  (x + 17) % 52 = 75 % 52 ↔ x = 6 ∨ x = 58 ∨ x = 110 ∨ x = 162 :=
by sorry

theorem number_of_solutions :
  (∃ x : ℕ, (0 < x ∧ x < 200 ∧ (x + 17) % 52 = 75 % 52)) ∧
  (∃ x1 x2 x3 x4 : ℕ, x1 = 6 ∧ x2 = 58 ∧ x3 = 110 ∧ x4 = 162) ∧
  4 = 4 :=
by sorry

end congruence_solutions_count_number_of_solutions_l13_13435


namespace greatest_divisor_of_sum_of_arith_seq_l13_13306

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l13_13306


namespace find_x_l13_13347

theorem find_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 5 = 5 * y + 2) :
  x = (685 + 25 * Real.sqrt 745) / 6 :=
by
  sorry

end find_x_l13_13347


namespace five_year_salary_increase_l13_13295

noncomputable def salary_growth (S : ℝ) := S * (1.08)^5

theorem five_year_salary_increase (S : ℝ) : 
  salary_growth S = S * 1.4693 := 
sorry

end five_year_salary_increase_l13_13295


namespace exist_non_quadratic_residues_sum_l13_13388

noncomputable section

def is_quadratic_residue_mod (p a : ℤ) : Prop :=
  ∃ x : ℤ, x^2 ≡ a [ZMOD p]

theorem exist_non_quadratic_residues_sum {p : ℤ} (hp : p > 5) (hp_modeq : p ≡ 1 [ZMOD 4]) (a : ℤ) : 
  ∃ b c : ℤ, a = b + c ∧ ¬is_quadratic_residue_mod p b ∧ ¬is_quadratic_residue_mod p c :=
sorry

end exist_non_quadratic_residues_sum_l13_13388


namespace an_gt_bn_l13_13239

theorem an_gt_bn (a b : ℕ → ℕ) (h₁ : a 1 = 2013) (h₂ : ∀ n, a (n + 1) = 2013^(a n))
                            (h₃ : b 1 = 1) (h₄ : ∀ n, b (n + 1) = 2013^(2012 * (b n))) :
  ∀ n, a n > b n := 
sorry

end an_gt_bn_l13_13239


namespace find_x_plus_y_l13_13384

theorem find_x_plus_y (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := 
by
  sorry

end find_x_plus_y_l13_13384


namespace new_radius_of_circle_l13_13334

theorem new_radius_of_circle
  (r_1 : ℝ)
  (A_1 : ℝ := π * r_1^2)
  (r_2 : ℝ)
  (A_2 : ℝ := 0.64 * A_1) 
  (h1 : r_1 = 5) 
  (h2 : A_2 = π * r_2^2) : 
  r_2 = 4 :=
by 
  sorry

end new_radius_of_circle_l13_13334


namespace complex_power_six_l13_13634

theorem complex_power_six (i : ℂ) (hi : i * i = -1) : (1 + i)^6 = -8 * i :=
by
  sorry

end complex_power_six_l13_13634


namespace frogs_count_l13_13357

variables (Alex Brian Chris LeRoy Mike : Type) 

-- Definitions for the species
def toad (x : Type) : Prop := ∃ p : Prop, p -- Dummy definition for toads
def frog (x : Type) : Prop := ∃ p : Prop, ¬p -- Dummy definition for frogs

-- Conditions
axiom Alex_statement : (toad Alex) → (∃ x : ℕ, x = 3) ∧ (frog Alex) → (¬(∃ x : ℕ, x = 3))
axiom Brian_statement : (toad Brian) → (toad Mike) ∧ (frog Brian) → (frog Mike)
axiom Chris_statement : (toad Chris) → (toad LeRoy) ∧ (frog Chris) → (frog LeRoy)
axiom LeRoy_statement : (toad LeRoy) → (toad Chris) ∧ (frog LeRoy) → (frog Chris)
axiom Mike_statement : (toad Mike) → (∃ x : ℕ, x < 3) ∧ (frog Mike) → (¬(∃ x : ℕ, x < 3))

theorem frogs_count (total : ℕ) : total = 5 → 
  (∃ frog_count : ℕ, frog_count = 2) :=
by
  -- Leaving the proof as a sorry placeholder
  sorry

end frogs_count_l13_13357


namespace min_reciprocal_sum_l13_13983

theorem min_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : 
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_reciprocal_sum_l13_13983


namespace students_answered_both_correctly_l13_13371

theorem students_answered_both_correctly :
  ∀ (total_students set_problem function_problem both_incorrect x : ℕ),
    total_students = 50 → 
    set_problem = 40 →
    function_problem = 31 →
    both_incorrect = 4 →
    x = total_students - both_incorrect - (set_problem + function_problem - total_students) →
    x = 25 :=
by
  intros total_students set_problem function_problem both_incorrect x
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end students_answered_both_correctly_l13_13371


namespace machine_made_8_shirts_today_l13_13789

-- Define the conditions
def shirts_per_minute : ℕ := 2
def minutes_worked_today : ℕ := 4

-- Define the expected number of shirts made today
def shirts_made_today : ℕ := shirts_per_minute * minutes_worked_today

-- The theorem stating that the shirts made today should be 8
theorem machine_made_8_shirts_today : shirts_made_today = 8 := by
  sorry

end machine_made_8_shirts_today_l13_13789


namespace land_plot_side_length_l13_13214

theorem land_plot_side_length (A : ℝ) (h : A = Real.sqrt 1024) : Real.sqrt A = 32 := 
by sorry

end land_plot_side_length_l13_13214


namespace completing_the_square_l13_13536

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l13_13536


namespace leak_empty_time_l13_13248

theorem leak_empty_time (P L : ℝ) (h1 : P = 1 / 6) (h2 : P - L = 1 / 12) : 1 / L = 12 :=
by
  -- Proof to be provided
  sorry

end leak_empty_time_l13_13248


namespace lcm_of_numbers_with_ratio_and_hcf_l13_13460

theorem lcm_of_numbers_with_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 4 * x) (h3 : Nat.gcd a b = 3) : Nat.lcm a b = 36 := 
  sorry

end lcm_of_numbers_with_ratio_and_hcf_l13_13460


namespace man_speed_l13_13967

theorem man_speed (L T V_t V_m : ℝ) (hL : L = 400) (hT : T = 35.99712023038157) (hVt : V_t = 46 * 1000 / 3600) (hE : L = (V_t - V_m) * T) : V_m = 1.666666666666684 :=
by
  sorry

end man_speed_l13_13967


namespace fraction_is_two_thirds_l13_13471

noncomputable def fraction_of_price_of_ballet_slippers (f : ℚ) : Prop :=
  let price_high_heels := 60
  let num_ballet_slippers := 5
  let total_cost := 260
  price_high_heels + num_ballet_slippers * f * price_high_heels = total_cost

theorem fraction_is_two_thirds : fraction_of_price_of_ballet_slippers (2 / 3) := by
  sorry

end fraction_is_two_thirds_l13_13471


namespace number_of_men_in_engineering_department_l13_13475

theorem number_of_men_in_engineering_department (T : ℝ) (h1 : 0.30 * T = 180) : 
  0.70 * T = 420 :=
by 
  -- The proof will be done here, but for now, we skip it.
  sorry

end number_of_men_in_engineering_department_l13_13475


namespace opposite_of_negative_a_is_a_l13_13171

-- Define the problem:
theorem opposite_of_negative_a_is_a (a : ℝ) : -(-a) = a :=
by 
  sorry

end opposite_of_negative_a_is_a_l13_13171


namespace ellipse_ratio_sum_l13_13407

theorem ellipse_ratio_sum :
  (∃ x y : ℝ, 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0) →
  (∃ a b : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0 → 
    (y = a * x ∨ y = b * x)) ∧ (a + b = 9)) :=
  sorry

end ellipse_ratio_sum_l13_13407


namespace washer_dryer_cost_diff_l13_13238

-- conditions
def total_cost : ℕ := 1200
def washer_cost : ℕ := 710
def dryer_cost : ℕ := total_cost - washer_cost

-- proof statement
theorem washer_dryer_cost_diff : (washer_cost - dryer_cost) = 220 :=
by
  sorry

end washer_dryer_cost_diff_l13_13238


namespace quotient_of_division_l13_13491

theorem quotient_of_division:
  ∀ (n d r q : ℕ), n = 165 → d = 18 → r = 3 → q = (n - r) / d → q = 9 :=
by sorry

end quotient_of_division_l13_13491


namespace area_of_annulus_l13_13588

theorem area_of_annulus (R r t : ℝ) (h : R > r) (h_tangent : R^2 = r^2 + t^2) : 
  π * (R^2 - r^2) = π * t^2 :=
by 
  sorry

end area_of_annulus_l13_13588


namespace solution_set_empty_range_a_l13_13520

theorem solution_set_empty_range_a (a : ℝ) :
  (∀ x : ℝ, ¬((a - 1) * x^2 + 2 * (a - 1) * x - 4 ≥ 0)) ↔ -3 < a ∧ a ≤ 1 :=
by
  sorry

end solution_set_empty_range_a_l13_13520


namespace probability_five_heads_in_six_tosses_is_09375_l13_13145

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_exact_heads (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * (p^k) * ((1-p)^(n-k))
  
theorem probability_five_heads_in_six_tosses_is_09375 :
  probability_exact_heads 6 5 0.5 = 0.09375 :=
by
  sorry

end probability_five_heads_in_six_tosses_is_09375_l13_13145


namespace number_less_than_value_l13_13105

-- Definition for the conditions
def exceeds_condition (x y : ℕ) : Prop := x - 18 = 3 * (y - x)
def specific_value (x : ℕ) : Prop := x = 69

-- Statement of the theorem
theorem number_less_than_value : ∃ y : ℕ, (exceeds_condition 69 y) ∧ (specific_value 69) → y = 86 :=
by
  -- To be proved
  sorry

end number_less_than_value_l13_13105


namespace length_of_tangent_l13_13332

/-- 
Let O and O1 be the centers of the larger and smaller circles respectively with radii 8 and 3. 
The circles touch each other internally. Let A be the point of tangency and OM be the tangent from center O to the smaller circle. 
Prove that the length of this tangent is 4.
--/
theorem length_of_tangent {O O1 : Type} (radius_large : ℝ) (radius_small : ℝ) (OO1 : ℝ) 
  (OM O1M : ℝ) (h : 8 - 3 = 5) (h1 : OO1 = 5) (h2 : O1M = 3): OM = 4 :=
by
  sorry

end length_of_tangent_l13_13332


namespace find_m_for_q_find_m_for_pq_l13_13571

variable (m : ℝ)

-- Statement q: The equation represents a hyperbola if and only if m > 3
def q (m : ℝ) : Prop := m > 3

-- Statement p: The inequality holds if and only if m >= 1
def p (m : ℝ) : Prop := m ≥ 1

-- 1. If statement q is true, find the range of values for m.
theorem find_m_for_q (h : q m) : m > 3 := by
  exact h

-- 2. If (p ∨ q) is true and (p ∧ q) is false, find the range of values for m.
theorem find_m_for_pq (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end find_m_for_q_find_m_for_pq_l13_13571


namespace fg_of_3_eq_29_l13_13741

def f (x : ℝ) : ℝ := 2 * x - 3
def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem fg_of_3_eq_29 : f (g 3) = 29 := by
  sorry

end fg_of_3_eq_29_l13_13741


namespace largest_whole_number_m_satisfies_inequality_l13_13032

theorem largest_whole_number_m_satisfies_inequality :
  ∃ m : ℕ, (1 / 4 + m / 6 : ℚ) < 3 / 2 ∧ ∀ n : ℕ, (1 / 4 + n / 6 : ℚ) < 3 / 2 → n ≤ 7 :=
by
  sorry

end largest_whole_number_m_satisfies_inequality_l13_13032


namespace correct_operation_is_C_l13_13798

/--
Given the following statements:
1. \( a^3 \cdot a^2 = a^6 \)
2. \( (2a^3)^3 = 6a^9 \)
3. \( -6x^5 \div 2x^3 = -3x^2 \)
4. \( (-x-2)(x-2) = x^2 - 4 \)

Prove that the correct statement is \( -6x^5 \div 2x^3 = -3x^2 \) and the other statements are incorrect.
-/
theorem correct_operation_is_C (a x : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧
  ((2 * a^3)^3 ≠ 6 * a^9) ∧
  (-6 * x^5 / (2 * x^3) = -3 * x^2) ∧
  ((-x - 2) * (x - 2) ≠ x^2 - 4) := by
  sorry

end correct_operation_is_C_l13_13798


namespace sum_of_squares_of_solutions_l13_13112

theorem sum_of_squares_of_solutions :
  (∃ s₁ s₂ : ℝ, s₁ ≠ s₂ ∧ s₁ + s₂ = 17 ∧ s₁ * s₂ = 22) →
  ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 245 :=
by
  sorry

end sum_of_squares_of_solutions_l13_13112


namespace find_y_eq_1_div_5_l13_13505

theorem find_y_eq_1_div_5 (b : ℝ) (y : ℝ) (h1 : b > 2) (h2 : y > 0) (h3 : (3 * y)^(Real.log 3 / Real.log b) - (5 * y)^(Real.log 5 / Real.log b) = 0) :
  y = 1 / 5 :=
by
  sorry

end find_y_eq_1_div_5_l13_13505


namespace kyle_practice_time_l13_13004

-- Definitions for the conditions
def weightlifting_time : ℕ := 20  -- in minutes
def running_time : ℕ := 2 * weightlifting_time  -- twice the weightlifting time
def total_running_and_weightlifting_time : ℕ := weightlifting_time + running_time  -- total time for running and weightlifting
def shooting_time : ℕ := total_running_and_weightlifting_time  -- because it's half the practice time

-- Total daily practice time, in minutes
def total_practice_time_minutes : ℕ := shooting_time + total_running_and_weightlifting_time

-- Total daily practice time, in hours
def total_practice_time_hours : ℕ := total_practice_time_minutes / 60

-- Theorem stating that Kyle practices for 2 hours every day given the conditions
theorem kyle_practice_time : total_practice_time_hours = 2 := by
  sorry

end kyle_practice_time_l13_13004


namespace exponent_evaluation_l13_13953

theorem exponent_evaluation {a b : ℕ} (h₁ : 2 ^ a ∣ 200) (h₂ : ¬ (2 ^ (a + 1) ∣ 200))
                           (h₃ : 5 ^ b ∣ 200) (h₄ : ¬ (5 ^ (b + 1) ∣ 200)) :
  (1 / 3) ^ (b - a) = 3 :=
by sorry

end exponent_evaluation_l13_13953


namespace probability_three_dice_sum_to_fourth_l13_13647

-- Define the probability problem conditions
def total_outcomes : ℕ := 8^4
def favorable_outcomes : ℕ := 1120

-- Final probability for the problem
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Lean statement for the proof problem
theorem probability_three_dice_sum_to_fourth :
  probability favorable_outcomes total_outcomes = 35 / 128 :=
by sorry

end probability_three_dice_sum_to_fourth_l13_13647


namespace monotonicity_of_f_l13_13993

noncomputable def f (x : ℝ) : ℝ := - (2 * x) / (1 + x^2)

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y ∧ (y < -1 ∨ x > 1) → f x < f y) ∧
  (∀ x y : ℝ, x < y ∧ -1 < x ∧ y < 1 → f y < f x) := sorry

end monotonicity_of_f_l13_13993


namespace line_through_P_origin_line_through_P_perpendicular_to_l3_l13_13984

-- Define lines l1, l2, l3
def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x + y + 2 = 0
def l3 (x y : ℝ) := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Prove the equations of the lines passing through P
theorem line_through_P_origin : ∃ A B C : ℝ, A * -2 + B * 2 + C = 0 ∧ A * 0 + B * 0 + C = 0 ∧ A = 1 ∧ B = 1 ∧ C = 0 :=
by sorry

theorem line_through_P_perpendicular_to_l3 : ∃ A B C : ℝ, A * -2 + B * 2 + C = 0 ∧ A * P.1 + B * P.2 + C = 0 ∧ A = 2 ∧ B = 1 ∧ C = 2 :=
by sorry

end line_through_P_origin_line_through_P_perpendicular_to_l3_l13_13984


namespace minimum_value_of_y_l13_13726

noncomputable def y (x : ℝ) : ℝ :=
  x^2 + 12 * x + 108 / x^4

theorem minimum_value_of_y : ∃ x > 0, y x = 49 :=
by
  sorry

end minimum_value_of_y_l13_13726


namespace skateboarder_speed_l13_13100

theorem skateboarder_speed (d t : ℕ) (ft_per_mile hr_to_sec : ℕ)
  (h1 : d = 660) (h2 : t = 30) (h3 : ft_per_mile = 5280) (h4 : hr_to_sec = 3600) :
  ((d / t) / ft_per_mile) * hr_to_sec = 15 :=
by sorry

end skateboarder_speed_l13_13100


namespace tangent_intersection_x_l13_13539

theorem tangent_intersection_x :
  ∃ x : ℝ, 
    0 < x ∧ (∃ r1 r2 : ℝ, 
     (r1 = 3) ∧ 
     (r2 = 8) ∧ 
     (0, 0) = (0, 0) ∧ 
     (18, 0) = (18, 0) ∧
     (∀ t : ℝ, t > 0 → t = x / (18 - x) → t = r1 / r2) ∧ 
      x = 54 / 11) := 
sorry

end tangent_intersection_x_l13_13539


namespace minimum_value_of_expression_l13_13876

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  1 / x + 4 / y + 9 / z

theorem minimum_value_of_expression (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  min_value_expression x y z ≥ 36 :=
sorry

end minimum_value_of_expression_l13_13876


namespace total_profit_l13_13997

theorem total_profit (investment_B : ℝ) (period_B : ℝ) (profit_B : ℝ) (investment_A : ℝ) (period_A : ℝ) (total_profit : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : period_A = 2 * period_B)
  (h3 : profit_B = 6000)
  (h4 : profit_B / (profit_A * 6 + profit_B) = profit_B) : total_profit = 7 * 6000 :=
by 
  sorry

#print axioms total_profit

end total_profit_l13_13997


namespace hockey_team_helmets_l13_13133

theorem hockey_team_helmets (r b : ℕ) 
  (h1 : b = r - 6) 
  (h2 : r * 3 = b * 5) : 
  r + b = 24 :=
by
  sorry

end hockey_team_helmets_l13_13133


namespace olaf_total_cars_l13_13626

noncomputable def olaf_initial_cars : ℕ := 150
noncomputable def uncle_cars : ℕ := 5
noncomputable def grandpa_cars : ℕ := 2 * uncle_cars
noncomputable def dad_cars : ℕ := 10
noncomputable def mum_cars : ℕ := dad_cars + 5
noncomputable def auntie_cars : ℕ := 6
noncomputable def liam_cars : ℕ := dad_cars / 2
noncomputable def emma_cars : ℕ := uncle_cars / 3
noncomputable def grandma_cars : ℕ := 3 * auntie_cars

noncomputable def total_gifts : ℕ := 
  grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars + liam_cars + emma_cars + grandma_cars

noncomputable def total_cars_after_gifts : ℕ := olaf_initial_cars + total_gifts

theorem olaf_total_cars : total_cars_after_gifts = 220 := by
  sorry

end olaf_total_cars_l13_13626


namespace inequality_solution_l13_13265

noncomputable def solution_set_inequality : Set ℝ := {x | -2 < x ∧ x < 1 / 3}

theorem inequality_solution :
  {x : ℝ | (2 * x - 1) / (3 * x + 1) > 1} = solution_set_inequality :=
by
  sorry

end inequality_solution_l13_13265


namespace sum_of_possible_values_l13_13092

theorem sum_of_possible_values :
  ∀ x, (|x - 5| - 4 = 3) → x = 12 ∨ x = -2 → (12 + (-2) = 10) :=
by
  sorry

end sum_of_possible_values_l13_13092


namespace problem1_no_solution_problem2_solution_l13_13437

theorem problem1_no_solution (x : ℝ) 
  (h : (5*x - 4)/(x - 2) = (4*x + 10)/(3*x - 6) - 1) : false :=
by
  -- The original equation turns out to have no solution
  sorry

theorem problem2_solution (x : ℝ) 
  (h : 1 - (x - 2)/(2 + x) = 16/(x^2 - 4)) : x = 6 :=
by
  -- The equation has a solution x = 6
  sorry

end problem1_no_solution_problem2_solution_l13_13437


namespace cats_to_dogs_ratio_l13_13337

theorem cats_to_dogs_ratio
    (cats dogs : ℕ)
    (ratio : cats / dogs = 3 / 4)
    (num_cats : cats = 18) :
    dogs = 24 :=
by
    sorry

end cats_to_dogs_ratio_l13_13337


namespace intersection_points_l13_13095

-- Definitions and conditions
def is_ellipse (e : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, e x y ↔ x^2 + 2*y^2 = 2

def is_tangent_or_intersects (l : ℝ → ℝ) (e : ℝ → ℝ → Prop) : Prop :=
  ∃ z1 z2 : ℝ, (e z1 (l z1) ∨ e z2 (l z2))

def lines_intersect (l1 l2 : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, l1 x = l2 x

theorem intersection_points :
  ∀ (e : ℝ → ℝ → Prop) (l1 l2 : ℝ → ℝ),
  is_ellipse e →
  is_tangent_or_intersects l1 e →
  is_tangent_or_intersects l2 e →
  lines_intersect l1 l2 →
  ∃ n : ℕ, n = 2 ∨ n = 3 ∨ n = 4 :=
by
  intros e l1 l2 he hto1 hto2 hl
  sorry

end intersection_points_l13_13095


namespace runners_speeds_and_track_length_l13_13757

/-- Given two runners α and β on a circular track starting at point P and running with uniform speeds,
when α reaches the halfway point Q, β is 16 meters behind α. At a later time, their positions are 
symmetric with respect to the diameter PQ. In 1 2/15 seconds, β reaches point Q, and 13 13/15 seconds later, 
α finishes the race. This theorem calculates the speeds of the runners and the distance of the lap. -/
theorem runners_speeds_and_track_length (x y : ℕ)
    (distance : ℝ)
    (runner_speed_alpha runner_speed_beta : ℝ) 
    (half_track_time_alpha half_track_time_beta : ℝ)
    (mirror_time_alpha mirror_time_beta : ℝ)
    (additional_time_beta : ℝ) :
    half_track_time_alpha = 16 ∧ 
    half_track_time_beta = (272/15) ∧ 
    mirror_time_alpha = (17/15) * (272/15 - 16/32) ∧ 
    mirror_time_beta = (17/15) ∧ 
    additional_time_beta = (13 + (13/15))  ∧ 
    runner_speed_beta = (15/2) ∧ 
    runner_speed_alpha = (85/10) ∧ 
    distance = 272 :=
  sorry

end runners_speeds_and_track_length_l13_13757


namespace difference_Q_R_l13_13222

variable (P Q R : ℝ) (x : ℝ)

theorem difference_Q_R (h1 : 11 * x - 5 * x = 12100) : 19 * x - 11 * x = 16133.36 :=
by
  sorry

end difference_Q_R_l13_13222


namespace cost_of_product_l13_13552

theorem cost_of_product (x : ℝ) (a : ℝ) (h : a > 0) :
  (1 + a / 100) * (x / (1 + a / 100)) = x :=
by
  field_simp [ne_of_gt h]
  sorry

end cost_of_product_l13_13552


namespace tank_capacity_is_correct_l13_13087

-- Definition of the problem conditions
def initial_fraction := 1 / 3
def added_water := 180
def final_fraction := 2 / 3

-- Capacity of the tank
noncomputable def tank_capacity : ℕ := 540

-- Proof statement
theorem tank_capacity_is_correct (x : ℕ) :
  (initial_fraction * x + added_water = final_fraction * x) → x = tank_capacity := 
by
  -- This is where the proof would go
  sorry

end tank_capacity_is_correct_l13_13087


namespace calculate_total_cost_l13_13648

def initial_price_orange : ℝ := 40
def initial_price_mango : ℝ := 50
def price_increase_percentage : ℝ := 0.15

-- Hypotheses
def new_price (initial_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_price * (1 + percentage_increase)

noncomputable def total_cost (num_oranges num_mangoes : ℕ) : ℝ :=
  (num_oranges * new_price initial_price_orange price_increase_percentage) +
  (num_mangoes * new_price initial_price_mango price_increase_percentage)

theorem calculate_total_cost :
  total_cost 10 10 = 1035 := by
  sorry

end calculate_total_cost_l13_13648


namespace smallest_k_for_Δk_un_zero_l13_13367

def u (n : ℕ) : ℤ := n^3 - n

def Δ (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0     => u
  | (k+1) => λ n => Δ k u (n+1) - Δ k u n

theorem smallest_k_for_Δk_un_zero (u : ℕ → ℤ) (h : ∀ n, u n = n^3 - n) :
  ∀ n, Δ 4 u n = 0 ∧ (∀ k < 4, ∃ n, Δ k u n ≠ 0) :=
by
  sorry

end smallest_k_for_Δk_un_zero_l13_13367


namespace compare_abc_l13_13943

variable (a b c : ℝ)

noncomputable def define_a : ℝ := (2/3)^(1/3)
noncomputable def define_b : ℝ := (2/3)^(1/2)
noncomputable def define_c : ℝ := (3/5)^(1/2)

theorem compare_abc (h₁ : a = define_a) (h₂ : b = define_b) (h₃ : c = define_c) :
  a > b ∧ b > c := by
  sorry

end compare_abc_l13_13943


namespace find_k_l13_13474

-- Define the problem statement
theorem find_k (d : ℝ) (x : ℝ)
  (h_ratio : 3 * x / (5 * x) = 3 / 5)
  (h_diag : (10 * d)^2 = (3 * x)^2 + (5 * x)^2) :
  ∃ k : ℝ, (3 * x) * (5 * x) = k * d^2 ∧ k = 750 / 17 := by
  sorry

end find_k_l13_13474


namespace num_white_squares_in_24th_row_l13_13824

-- Define the function that calculates the total number of squares in the nth row
def total_squares (n : ℕ) : ℕ := 1 + 2 * (n - 1)

-- Define the function that calculates the number of white squares in the nth row
def white_squares (n : ℕ) : ℕ := (total_squares n - 2) / 2

-- Problem statement for the Lean 4 theorem
theorem num_white_squares_in_24th_row : white_squares 24 = 23 :=
by {
  -- Lean proof generation will be placed here
  sorry
}

end num_white_squares_in_24th_row_l13_13824


namespace bob_same_color_probability_is_1_over_28_l13_13416

def num_marriages : ℕ := 9
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3

def david_marbles : ℕ := 3
def alice_marbles : ℕ := 3
def bob_marbles : ℕ := 3

def total_ways : ℕ := 1680
def favorable_ways : ℕ := 60
def probability_bob_same_color := favorable_ways / total_ways

theorem bob_same_color_probability_is_1_over_28 : probability_bob_same_color = (1 : ℚ) / 28 := by
  sorry

end bob_same_color_probability_is_1_over_28_l13_13416


namespace arctan_sum_zero_l13_13591
open Real

variable (a b c : ℝ)
variable (h : a^2 + b^2 = c^2)

theorem arctan_sum_zero (h : a^2 + b^2 = c^2) :
  arctan (a / (b + c)) + arctan (b / (a + c)) + arctan (c / (a + b)) = 0 := 
sorry

end arctan_sum_zero_l13_13591


namespace number_of_boys_l13_13610

-- We define the conditions provided in the problem
def child_1_has_3_brothers : Prop := ∃ B G : ℕ, B - 1 = 3 ∧ G = 6
def child_2_has_4_brothers : Prop := ∃ B G : ℕ, B - 1 = 4 ∧ G = 5

theorem number_of_boys (B G : ℕ) (h1 : child_1_has_3_brothers) (h2 : child_2_has_4_brothers) : B = 4 :=
by
  sorry

end number_of_boys_l13_13610


namespace part_one_part_two_l13_13273

-- Part (1)
theorem part_one (a : ℝ) (h : a ≤ 2) (x : ℝ) :
  (|x - 1| + |x - a| ≥ 2 ↔ x ≤ 0.5 ∨ x ≥ 2.5) :=
sorry

-- Part (2)
theorem part_two (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, |x - 1| + |x - a| + |x - 1| ≥ 1) :
  a ≥ 2 :=
sorry

end part_one_part_two_l13_13273


namespace value_of_f_2018_l13_13759

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 3) * f x = -1
axiom initial_condition : f (-1) = 2

theorem value_of_f_2018 : f 2018 = -1 / 2 :=
by
  sorry

end value_of_f_2018_l13_13759


namespace find_constants_l13_13376

theorem find_constants :
  ∃ P Q : ℚ, (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (4 * x + 7) / (x^2 - 3 * x - 18) = P / (x - 6) + Q / (x + 3)) ∧
    P = 31 / 9 ∧ Q = 5 / 9 :=
by
  sorry

end find_constants_l13_13376


namespace find_n_l13_13603

theorem find_n (x y : ℝ) (h1 : (7 * x + 2 * y) / (x - n * y) = 23) (h2 : x / (2 * y) = 3 / 2) :
  ∃ n : ℝ, n = 2 := by
  sorry

end find_n_l13_13603


namespace compute_xy_l13_13063

theorem compute_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^3 + y^3 = 351) : x * y = 14 :=
by
  sorry

end compute_xy_l13_13063


namespace three_digit_numbers_divisible_by_11_are_550_or_803_l13_13139

theorem three_digit_numbers_divisible_by_11_are_550_or_803 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ ∃ (a b c : ℕ), N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 11 ∣ N ∧ (N / 11 = a^2 + b^2 + c^2)) → (N = 550 ∨ N = 803) :=
by
  sorry

end three_digit_numbers_divisible_by_11_are_550_or_803_l13_13139


namespace calculate_area_l13_13801

def leftmost_rectangle_area (height width : ℕ) : ℕ := height * width
def middle_rectangle_area (height width : ℕ) : ℕ := height * width
def rightmost_rectangle_area (height width : ℕ) : ℕ := height * width

theorem calculate_area : 
  let leftmost_segment_height := 7
  let bottom_width := 6
  let segment_above_3 := 3
  let segment_above_2 := 2
  let rightmost_width := 5
  leftmost_rectangle_area leftmost_segment_height bottom_width + 
  middle_rectangle_area segment_above_3 segment_above_3 + 
  rightmost_rectangle_area segment_above_2 rightmost_width = 
  61 := by
    sorry

end calculate_area_l13_13801


namespace most_reasonable_sample_l13_13740

-- Define what it means to be a reasonable sample
def is_reasonable_sample (sample : String) : Prop :=
  sample = "D"

-- Define the conditions for each sample
def sample_A := "A"
def sample_B := "B"
def sample_C := "C"
def sample_D := "D"

-- Define the problem statement
theorem most_reasonable_sample :
  is_reasonable_sample sample_D :=
sorry

end most_reasonable_sample_l13_13740


namespace visitors_on_previous_day_is_246_l13_13021

def visitors_on_previous_day : Nat := 246
def total_visitors_in_25_days : Nat := 949

theorem visitors_on_previous_day_is_246 :
  visitors_on_previous_day = 246 := 
by
  rfl

end visitors_on_previous_day_is_246_l13_13021


namespace sum_of_perimeters_l13_13287

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) : 
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * x + 4 * y := by
  sorry

end sum_of_perimeters_l13_13287


namespace sufficient_but_not_necessary_condition_l13_13219

def condition_p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0
def condition_q (x : ℝ) : Prop := |x - 2| < 1

theorem sufficient_but_not_necessary_condition : 
  (∀ x : ℝ, condition_p x → condition_q x) ∧ ¬(∀ x : ℝ, condition_q x → condition_p x) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l13_13219


namespace mary_biking_time_l13_13008

-- Define the conditions and the task
def total_time_away := 570 -- in minutes
def time_in_classes := 7 * 45 -- in minutes
def lunch_time := 40 -- in minutes
def additional_activities := 105 -- in minutes
def time_in_school_activities := time_in_classes + lunch_time + additional_activities

-- Define the total biking time based on given conditions
theorem mary_biking_time : 
  total_time_away - time_in_school_activities = 110 :=
by 
-- sorry is used to skip the proof step.
  sorry

end mary_biking_time_l13_13008


namespace total_ages_l13_13855

theorem total_ages (bride_age groom_age : ℕ) (h1 : bride_age = 102) (h2 : groom_age = bride_age - 19) : bride_age + groom_age = 185 :=
by
  sorry

end total_ages_l13_13855


namespace expression_change_l13_13236

variable (x b : ℝ)

-- The conditions
def expression (x : ℝ) : ℝ := x^3 - 5 * x + 1
def expr_change_plus (x b : ℝ) : ℝ := (x + b)^3 - 5 * (x + b) + 1
def expr_change_minus (x b : ℝ) : ℝ := (x - b)^3 - 5 * (x - b) + 1

-- The Lean statement to prove
theorem expression_change (h_b_pos : 0 < b) :
  expr_change_plus x b - expression x = 3 * b * x^2 + 3 * b^2 * x + b^3 - 5 * b ∨ 
  expr_change_minus x b - expression x = -3 * b * x^2 + 3 * b^2 * x - b^3 + 5 * b := 
by
  sorry

end expression_change_l13_13236


namespace rainfall_ratio_l13_13297

theorem rainfall_ratio (R1 R2 : ℕ) (H1 : R2 = 18) (H2 : R1 + R2 = 30) : R2 / R1 = 3 / 2 := by
  sorry

end rainfall_ratio_l13_13297


namespace number_of_oranges_l13_13816

theorem number_of_oranges (B T O : ℕ) (h₁ : B + T = 178) (h₂ : B + T + O = 273) : O = 95 :=
by
  -- Begin proof here
  sorry

end number_of_oranges_l13_13816


namespace find_a_l13_13180

-- Define the necessary variables
variables (a b : ℝ) (t : ℝ)

-- Given conditions
def b_val : ℝ := 2120
def t_val : ℝ := 0.5

-- The statement we need to prove
theorem find_a (h: b = b_val) (h2: t = t_val) (h3: t = a / b) : a = 1060 := by
  -- Placeholder for proof
  sorry

end find_a_l13_13180


namespace boxes_per_week_l13_13006

-- Define the given conditions
def cost_per_box : ℝ := 3.00
def weeks_in_year : ℝ := 52
def total_spent_per_year : ℝ := 312

-- The question we want to prove:
theorem boxes_per_week:
  (total_spent_per_year = cost_per_box * weeks_in_year * (total_spent_per_year / (weeks_in_year * cost_per_box))) → 
  (total_spent_per_year / (weeks_in_year * cost_per_box)) = 2 := sorry

end boxes_per_week_l13_13006


namespace transformation_g_from_f_l13_13685

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (8 * x + 3 * Real.pi / 2)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (4 * x)

theorem transformation_g_from_f :
  (∀ x, g x = f (x + Real.pi / 4) * 2) ∨ (∀ x, g x = f (x - Real.pi / 4) * 2) := 
by
  sorry

end transformation_g_from_f_l13_13685


namespace sqrt_condition_l13_13745

theorem sqrt_condition (x : ℝ) : (x - 3 ≥ 0) ↔ (x = 3) :=
by sorry

end sqrt_condition_l13_13745


namespace roots_squared_sum_l13_13497

theorem roots_squared_sum (p q r : ℂ) (h : ∀ x : ℂ, 3 * x ^ 3 - 3 * x ^ 2 + 6 * x - 9 = 0 → x = p ∨ x = q ∨ x = r) :
  p^2 + q^2 + r^2 = -3 :=
by
  sorry

end roots_squared_sum_l13_13497


namespace small_cone_altitude_l13_13675

noncomputable def frustum_height : ℝ := 18
noncomputable def lower_base_area : ℝ := 400 * Real.pi
noncomputable def upper_base_area : ℝ := 100 * Real.pi

theorem small_cone_altitude (h_frustum : frustum_height = 18) 
    (A_lower : lower_base_area = 400 * Real.pi) 
    (A_upper : upper_base_area = 100 * Real.pi) : 
    ∃ (h_small_cone : ℝ), h_small_cone = 18 := 
by
  sorry

end small_cone_altitude_l13_13675


namespace parallel_lines_eq_a_l13_13264

theorem parallel_lines_eq_a (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a + 1) * x - a * y = 0) → (a = -3/2 ∨ a = 0) :=
by sorry

end parallel_lines_eq_a_l13_13264


namespace work_completion_time_l13_13373

noncomputable def rate_b : ℝ := 1 / 24
noncomputable def rate_a : ℝ := 2 * rate_b
noncomputable def combined_rate : ℝ := rate_a + rate_b
noncomputable def completion_time : ℝ := 1 / combined_rate

theorem work_completion_time :
  completion_time = 8 :=
by
  sorry

end work_completion_time_l13_13373


namespace tan_A_in_right_triangle_l13_13456

theorem tan_A_in_right_triangle (AC : ℝ) (AB : ℝ) (BC : ℝ) (hAC : AC = Real.sqrt 20) (hAB : AB = 4) (h_right_triangle : AC^2 = AB^2 + BC^2) :
  Real.tan (Real.arcsin (AB / AC)) = 1 / 2 :=
by
  sorry

end tan_A_in_right_triangle_l13_13456


namespace power_mod_remainder_l13_13849

theorem power_mod_remainder :
  (7 ^ 2023) % 17 = 16 :=
sorry

end power_mod_remainder_l13_13849


namespace groupB_is_conditional_control_l13_13396

-- Definitions based on conditions
def groupA_medium (nitrogen_sources : Set String) : Prop := nitrogen_sources = {"urea"}
def groupB_medium (nitrogen_sources : Set String) : Prop := nitrogen_sources = {"urea", "nitrate"}

-- The property that defines a conditional control in this context.
def conditional_control (control_sources : Set String) (experimental_sources : Set String) : Prop :=
  control_sources ≠ experimental_sources ∧ "urea" ∈ control_sources ∧ "nitrate" ∈ experimental_sources

-- Prove that Group B's experiment forms a conditional control
theorem groupB_is_conditional_control :
  ∃ nitrogen_sourcesA nitrogen_sourcesB, groupA_medium nitrogen_sourcesA ∧ groupB_medium nitrogen_sourcesB ∧
  conditional_control nitrogen_sourcesA nitrogen_sourcesB :=
by
  sorry

end groupB_is_conditional_control_l13_13396


namespace fg_eq_neg7_l13_13429

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^2 + 2

theorem fg_eq_neg7 : f (g 2) = -7 :=
  by
    sorry

end fg_eq_neg7_l13_13429


namespace original_price_l13_13243

theorem original_price (sale_price : ℝ) (discount : ℝ) : 
  sale_price = 55 → discount = 0.45 → 
  ∃ (P : ℝ), 0.55 * P = sale_price ∧ P = 100 :=
by
  sorry

end original_price_l13_13243


namespace hillary_activities_l13_13665

-- Define the conditions
def swims_every : ℕ := 6
def runs_every : ℕ := 4
def cycles_every : ℕ := 16

-- Define the theorem to prove
theorem hillary_activities : Nat.lcm (Nat.lcm swims_every runs_every) cycles_every = 48 :=
by
  -- Provide a placeholder for the proof
  sorry

end hillary_activities_l13_13665


namespace inverse_function_problem_l13_13769

theorem inverse_function_problem
  (f : ℝ → ℝ)
  (f_inv : ℝ → ℝ)
  (h₁ : ∀ x, f (f_inv x) = x)
  (h₂ : ∀ x, f_inv (f x) = x)
  (a b : ℝ)
  (h₃ : f_inv (a - 1) + f_inv (b - 1) = 1) :
  f (a * b) = 3 :=
by
  sorry

end inverse_function_problem_l13_13769


namespace students_suggesting_bacon_l13_13919

theorem students_suggesting_bacon (S : ℕ) (M : ℕ) (h1: S = 310) (h2: M = 185) : S - M = 125 := 
by
  -- proof here
  sorry

end students_suggesting_bacon_l13_13919


namespace min_lcm_leq_six_floor_l13_13254

theorem min_lcm_leq_six_floor (n : ℕ) (h : n ≠ 4) (a : Fin n → ℕ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 2 * n) : 
  ∃ i j, i < j ∧ Nat.lcm (a i) (a j) ≤ 6 * (n / 2 + 1) :=
by
  sorry

end min_lcm_leq_six_floor_l13_13254


namespace tan_alpha_values_l13_13821

theorem tan_alpha_values (α : ℝ) (h : Real.sin α + Real.cos α = 7 / 5) : 
  (Real.tan α = 4 / 3) ∨ (Real.tan α = 3 / 4) := 
  sorry

end tan_alpha_values_l13_13821


namespace work_days_B_works_l13_13244

theorem work_days_B_works (x : ℕ) (A_work_rate B_work_rate : ℚ) (A_remaining_days : ℕ) (total_work : ℚ) :
  A_work_rate = (1 / 12) ∧
  B_work_rate = (1 / 15) ∧
  A_remaining_days = 4 ∧
  total_work = 1 →
  x * B_work_rate + A_remaining_days * A_work_rate = total_work →
  x = 10 :=
sorry

end work_days_B_works_l13_13244


namespace smallest_n_exists_l13_13723

theorem smallest_n_exists :
  ∃ n : ℕ, n > 0 ∧ 3^(3^(n + 1)) ≥ 3001 :=
by
  sorry

end smallest_n_exists_l13_13723


namespace probability_black_balls_l13_13872

variable {m1 m2 k1 k2 : ℕ}

/-- Given conditions:
  1. The total number of balls in both urns is 25.
  2. The probability of drawing one white ball from each urn is 0.54.
To prove: The probability of both drawn balls being black is 0.04.
-/
theorem probability_black_balls : 
  m1 + m2 = 25 → 
  (k1 * k2) * 50 = 27 * m1 * m2 → 
  ((m1 - k1) * (m2 - k2) : ℚ) / (m1 * m2) = 0.04 :=
by
  intros h1 h2
  sorry

end probability_black_balls_l13_13872


namespace problem_part1_problem_part2_l13_13302

def A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def CR_A : Set ℝ := { x | x < 3 ∨ x > 7 }

theorem problem_part1 : A ∪ B = { x | 3 ≤ x ∧ x ≤ 7 } := by
  sorry

theorem problem_part2 : (CR_A ∩ B) = { x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10) } := by
  sorry

end problem_part1_problem_part2_l13_13302


namespace smallest_integer_no_inverse_mod_77_66_l13_13926

theorem smallest_integer_no_inverse_mod_77_66 :
  ∃ a : ℕ, 0 < a ∧ a = 11 ∧ gcd a 77 > 1 ∧ gcd a 66 > 1 :=
by
  sorry

end smallest_integer_no_inverse_mod_77_66_l13_13926


namespace rectangle_area_l13_13195

-- Define the vertices of the rectangle
def V1 : ℝ × ℝ := (-7, 1)
def V2 : ℝ × ℝ := (1, 1)
def V3 : ℝ × ℝ := (1, -6)
def V4 : ℝ × ℝ := (-7, -6)

-- Define the function to compute the area of the rectangle given the vertices
noncomputable def area_of_rectangle (A B C D : ℝ × ℝ) : ℝ :=
  let length := abs (B.1 - A.1)
  let width := abs (A.2 - D.2)
  length * width

-- The statement to prove
theorem rectangle_area : area_of_rectangle V1 V2 V3 V4 = 56 := by
  sorry

end rectangle_area_l13_13195


namespace payment_for_30_kilograms_l13_13366

-- Define the price calculation based on quantity x
def payment_amount (x : ℕ) : ℕ :=
  if x ≤ 10 then 20 * x
  else 16 * x + 40

-- Prove that for x = 30, the payment amount y equals 520
theorem payment_for_30_kilograms : payment_amount 30 = 520 := by
  sorry

end payment_for_30_kilograms_l13_13366


namespace num_students_yes_R_l13_13632

noncomputable def num_students_total : ℕ := 800
noncomputable def num_students_yes_only_M : ℕ := 150
noncomputable def num_students_no_to_both : ℕ := 250

theorem num_students_yes_R : (num_students_total - num_students_no_to_both) - num_students_yes_only_M = 400 :=
by
  sorry

end num_students_yes_R_l13_13632


namespace people_on_train_after_third_stop_l13_13961

variable (initial_people : ℕ) (off_1 boarded_1 off_2 boarded_2 off_3 boarded_3 : ℕ)

def people_after_first_stop (initial : ℕ) (off_1 boarded_1 : ℕ) : ℕ :=
  initial - off_1 + boarded_1

def people_after_second_stop (first_stop : ℕ) (off_2 boarded_2 : ℕ) : ℕ :=
  first_stop - off_2 + boarded_2

def people_after_third_stop (second_stop : ℕ) (off_3 boarded_3 : ℕ) : ℕ :=
  second_stop - off_3 + boarded_3

theorem people_on_train_after_third_stop :
  people_after_third_stop (people_after_second_stop (people_after_first_stop initial_people off_1 boarded_1) off_2 boarded_2) off_3 boarded_3 = 42 :=
  by
    have initial_people := 48
    have off_1 := 12
    have boarded_1 := 7
    have off_2 := 15
    have boarded_2 := 9
    have off_3 := 6
    have boarded_3 := 11
    sorry

end people_on_train_after_third_stop_l13_13961


namespace ways_from_A_to_C_l13_13513

theorem ways_from_A_to_C (ways_A_to_B : ℕ) (ways_B_to_C : ℕ) (hA_to_B : ways_A_to_B = 3) (hB_to_C : ways_B_to_C = 4) : ways_A_to_B * ways_B_to_C = 12 :=
by
  sorry

end ways_from_A_to_C_l13_13513


namespace volume_of_inscribed_sphere_l13_13641

theorem volume_of_inscribed_sphere {cube_edge : ℝ} (h : cube_edge = 6) : 
  ∃ V : ℝ, V = 36 * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l13_13641


namespace count_students_in_meets_l13_13415

theorem count_students_in_meets (A B : Finset ℕ) (hA : A.card = 13) (hB : B.card = 12) (hAB : (A ∩ B).card = 6) :
  (A ∪ B).card = 19 :=
by
  sorry

end count_students_in_meets_l13_13415


namespace rounded_diff_greater_l13_13269

variable (x y ε : ℝ)
variable (h1 : x > y)
variable (h2 : y > 0)
variable (h3 : ε > 0)

theorem rounded_diff_greater : (x + ε) - (y - ε) > x - y :=
  by
  sorry

end rounded_diff_greater_l13_13269


namespace extreme_point_condition_l13_13117

variable {R : Type*} [OrderedRing R]

def f (x a b : R) : R := x^3 - a*x - b

theorem extreme_point_condition (a b x0 x1 : R) (h₁ : ∀ x : R, f x a b = x^3 - a*x - b)
  (h₂ : f x0 a b = x0^3 - a*x0 - b)
  (h₃ : f x1 a b = x1^3 - a*x1 - b)
  (has_extreme : ∃ x0 : R, 3*x0^2 = a) 
  (hx1_extreme : f x1 a b = f x0 a b) 
  (hx1_x0_diff : x1 ≠ x0) :
  x1 + 2*x0 = 0 :=
by
  sorry

end extreme_point_condition_l13_13117


namespace find_number_l13_13190

theorem find_number (x : ℝ) : x = 7 ∧ x^2 + 95 = (x - 19)^2 :=
by
  sorry

end find_number_l13_13190


namespace divides_both_numerator_and_denominator_l13_13448

theorem divides_both_numerator_and_denominator (x m : ℤ) :
  (x ∣ (5 * m + 6)) ∧ (x ∣ (8 * m + 7)) → (x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13) :=
by
  sorry

end divides_both_numerator_and_denominator_l13_13448


namespace polynomial_factors_l13_13537

theorem polynomial_factors (x : ℝ) : 
  (x^4 - 4*x^2 + 4) = (x^2 - 2*x + 2) * (x^2 + 2*x + 2) :=
by
  sorry

end polynomial_factors_l13_13537


namespace valid_permutations_count_l13_13800

def num_permutations (seq : List ℕ) : ℕ :=
  -- A dummy implementation, the real function would calculate the number of valid permutations.
  sorry

theorem valid_permutations_count : num_permutations [1, 2, 3, 4, 5, 6] = 32 :=
by
  sorry

end valid_permutations_count_l13_13800


namespace problem_1_problem_2_l13_13633

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  let numerator := |C1 - C2|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  let numerator := |A * x0 + B * y0 + C|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

theorem problem_1 : distance_between_parallel_lines 2 1 (-1) 1 = 2 * Real.sqrt 5 / 5 :=
  by sorry

theorem problem_2 : distance_point_to_line 2 1 (-1) 0 2 = Real.sqrt 5 / 5 :=
  by sorry

end problem_1_problem_2_l13_13633


namespace degree_poly_product_l13_13575

open Polynomial

-- Given conditions: p and q are polynomials with specified degrees
variables {R : Type*} [CommRing R]
variable (p q : R[X])
variable (hp : degree p = 3)
variable (hq : degree q = 6)

-- Proposition: The degree of p(x^2) * q(x^4) is 30
theorem degree_poly_product : degree (p.comp ((X : R[X])^2) * (q.comp ((X : R[X])^4))) = 30 :=
by sorry

end degree_poly_product_l13_13575


namespace new_mix_concentration_l13_13586

theorem new_mix_concentration 
  (capacity1 capacity2 capacity_mix : ℝ)
  (alc_percent1 alc_percent2 : ℝ)
  (amount1 amount2 : capacity1 = 3 ∧ capacity2 = 5 ∧ capacity_mix = 10)
  (percent1: alc_percent1 = 0.25)
  (percent2: alc_percent2 = 0.40)
  (total_volume : ℝ)
  (eight_liters : total_volume = 8) :
  (alc_percent1 * capacity1 + alc_percent2 * capacity2) / total_volume * 100 = 34.375 :=
by
  sorry

end new_mix_concentration_l13_13586


namespace product_gcd_lcm_15_9_l13_13288

theorem product_gcd_lcm_15_9 : Nat.gcd 15 9 * Nat.lcm 15 9 = 135 := 
by
  -- skipping proof as instructed
  sorry

end product_gcd_lcm_15_9_l13_13288


namespace intersecting_lines_l13_13454

theorem intersecting_lines (c d : ℝ) :
  (∀ x y, (x = (1/3) * y + c) ∧ (y = (1/3) * x + d) → x = 3 ∧ y = 6) →
  c + d = 6 :=
by
  sorry

end intersecting_lines_l13_13454


namespace number_of_orange_ribbons_l13_13854

/-- Define the total number of ribbons -/
def total_ribbons (yellow purple orange black total : ℕ) : Prop :=
  yellow + purple + orange + black = total

/-- Define the fractions -/
def fractions (total_ribbons yellow purple orange black : ℕ) : Prop :=
  yellow = total_ribbons / 4 ∧ purple = total_ribbons / 3 ∧ orange = total_ribbons / 12 ∧ black = 40

/-- Define the black ribbons fraction -/
def black_fraction (total_ribbons : ℕ) : Prop :=
  40 = total_ribbons / 3

theorem number_of_orange_ribbons :
  ∃ (total : ℕ), total_ribbons (total / 4) (total / 3) (total / 12) 40 total ∧ black_fraction total ∧ (total / 12 = 10) :=
by
  sorry

end number_of_orange_ribbons_l13_13854


namespace simplify_expression_l13_13441

theorem simplify_expression (a : ℝ) (h : a ≠ 1 ∧ a ≠ -1) : 
  1 - (1 / (1 + (a^2 / (1 - a^2)))) = a^2 :=
sorry

end simplify_expression_l13_13441


namespace rectangular_cube_length_l13_13885

theorem rectangular_cube_length (L : ℝ) (h1 : 2 * (L * 2) + 2 * (L * 0.5) + 2 * (2 * 0.5) = 24) : L = 4.6 := 
by {
  sorry
}

end rectangular_cube_length_l13_13885


namespace min_value_of_f_l13_13764

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_of_f : ∃ x : ℝ, f x = x^3 - 3 * x^2 + 1 ∧ (∀ y : ℝ, f y ≥ f 2) :=
by
  sorry

end min_value_of_f_l13_13764


namespace range_of_a_l13_13766

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l13_13766


namespace Mary_put_crayons_l13_13432

def initial_crayons : ℕ := 7
def final_crayons : ℕ := 10
def added_crayons (i f : ℕ) : ℕ := f - i

theorem Mary_put_crayons :
  added_crayons initial_crayons final_crayons = 3 := 
by
  sorry

end Mary_put_crayons_l13_13432


namespace annual_average_growth_rate_l13_13055

theorem annual_average_growth_rate (x : ℝ) :
  7200 * (1 + x)^2 = 8450 :=
sorry

end annual_average_growth_rate_l13_13055


namespace abhay_speed_l13_13671

theorem abhay_speed
    (A S : ℝ)
    (h1 : 30 / A = 30 / S + 2)
    (h2 : 30 / (2 * A) = 30 / S - 1) :
    A = 5 * Real.sqrt 6 :=
by
  sorry

end abhay_speed_l13_13671


namespace clarence_oranges_left_l13_13965

-- Definitions based on the conditions in the problem
def initial_oranges : ℕ := 5
def oranges_from_joyce : ℕ := 3
def total_oranges_after_joyce : ℕ := initial_oranges + oranges_from_joyce
def oranges_given_to_bob : ℕ := total_oranges_after_joyce / 2
def oranges_left : ℕ := total_oranges_after_joyce - oranges_given_to_bob

-- Proof statement that needs to be proven
theorem clarence_oranges_left : oranges_left = 4 :=
by
  sorry

end clarence_oranges_left_l13_13965


namespace measure_of_angle_y_l13_13992

def is_straight_angle (a : ℝ) := a = 180

theorem measure_of_angle_y (angle_ABC angle_ADB angle_BDA y : ℝ) 
  (h1 : angle_ABC = 117)
  (h2 : angle_ADB = 31)
  (h3 : angle_BDA = 28)
  (h4 : is_straight_angle (angle_ABC + (180 - angle_ABC)))
  : y = 86 := 
by 
  sorry

end measure_of_angle_y_l13_13992


namespace solve_for_m_l13_13114

theorem solve_for_m :
  (∀ (m : ℕ), 
   ((1:ℚ)^(m+1) / 5^(m+1) * 1^18 / 4^18 = 1 / (2 * 10^35)) → m = 34) := 
by apply sorry

end solve_for_m_l13_13114


namespace geometric_series_sum_eq_l13_13743

theorem geometric_series_sum_eq :
  let a := (1/3 : ℚ)
  let r := (1/3 : ℚ)
  let n := 8
  let S := a * (1 - r^n) / (1 - r)
  S = 3280 / 6561 :=
by
  sorry

end geometric_series_sum_eq_l13_13743


namespace max_value_of_function_is_seven_l13_13796

theorem max_value_of_function_is_seven:
  ∃ a: ℕ, (0 < a) ∧ 
  (∃ x: ℝ, (x + Real.sqrt (13 - 2 * a * x)) = 7 ∧
    ∀ y: ℝ, (y = x + Real.sqrt (13 - 2 * a * x)) → y ≤ 7) :=
sorry

end max_value_of_function_is_seven_l13_13796


namespace michael_payment_correct_l13_13240

def suit_price : ℕ := 430
def suit_discount : ℕ := 100
def shoes_price : ℕ := 190
def shoes_discount : ℕ := 30
def shirt_price : ℕ := 80
def tie_price: ℕ := 50
def combined_discount : ℕ := (shirt_price + tie_price) * 20 / 100

def total_price_paid : ℕ :=
    suit_price - suit_discount + shoes_price - shoes_discount + (shirt_price + tie_price - combined_discount)

theorem michael_payment_correct :
    total_price_paid = 594 :=
by
    -- skipping the proof
    sorry

end michael_payment_correct_l13_13240


namespace binomial_probability_l13_13390

theorem binomial_probability (n : ℕ) (p : ℝ) (h1 : (n * p = 300)) (h2 : (n * p * (1 - p) = 200)) :
    p = 1 / 3 :=
by
  sorry

end binomial_probability_l13_13390


namespace uncle_ben_eggs_l13_13999

noncomputable def total_eggs (total_chickens : ℕ) (roosters : ℕ) (non_egg_laying_hens : ℕ) (eggs_per_hen : ℕ) : ℕ :=
  let total_hens := total_chickens - roosters
  let egg_laying_hens := total_hens - non_egg_laying_hens
  egg_laying_hens * eggs_per_hen

theorem uncle_ben_eggs :
  total_eggs 440 39 15 3 = 1158 :=
by
  unfold total_eggs
  -- Correct steps to prove the theorem can be skipped with sorry
  sorry

end uncle_ben_eggs_l13_13999


namespace checkerboard_sums_l13_13177

-- Define the dimensions and the arrangement of the checkerboard
def n : ℕ := 10
def board (i j : ℕ) : ℕ := i * n + j + 1

-- Define corner positions
def top_left_corner : ℕ := board 0 0
def top_right_corner : ℕ := board 0 (n - 1)
def bottom_left_corner : ℕ := board (n - 1) 0
def bottom_right_corner : ℕ := board (n - 1) (n - 1)

-- Sum of the corners
def corner_sum : ℕ := top_left_corner + top_right_corner + bottom_left_corner + bottom_right_corner

-- Define the positions of the main diagonals
def main_diagonal (i : ℕ) : ℕ := board i i
def anti_diagonal (i : ℕ) : ℕ := board i (n - 1 - i)

-- Sum of the main diagonals
def diagonal_sum : ℕ := (Finset.range n).sum main_diagonal + (Finset.range n).sum anti_diagonal - (main_diagonal 0 + main_diagonal (n - 1))

-- Statement to prove
theorem checkerboard_sums : corner_sum = 202 ∧ diagonal_sum = 101 :=
by
-- Proof is not required as per the instructions
sorry

end checkerboard_sums_l13_13177


namespace intersection_A_B_l13_13747

def A := {x : ℝ | |x| < 1}
def B := {x : ℝ | -2 < x ∧ x < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end intersection_A_B_l13_13747


namespace sam_won_total_matches_l13_13029

/-- Sam's first 100 matches and he won 50% of them -/
def first_100_matches : ℕ := 100

/-- Sam won 50% of his first 100 matches -/
def win_rate_first : ℕ := 50

/-- Sam's next 100 matches and he won 60% of them -/
def next_100_matches : ℕ := 100

/-- Sam won 60% of his next 100 matches -/
def win_rate_next : ℕ := 60

/-- The total number of matches Sam won -/
def total_matches_won (first_100_matches: ℕ) (win_rate_first: ℕ) (next_100_matches: ℕ) (win_rate_next: ℕ) : ℕ :=
  (first_100_matches * win_rate_first) / 100 + (next_100_matches * win_rate_next) / 100

theorem sam_won_total_matches :
  total_matches_won first_100_matches win_rate_first next_100_matches win_rate_next = 110 :=
by
  sorry

end sam_won_total_matches_l13_13029


namespace relationship_between_variables_l13_13640

theorem relationship_between_variables
  (a b x y : ℚ)
  (h1 : x + y = a + b)
  (h2 : y - x < a - b)
  (h3 : b > a) :
  y < a ∧ a < b ∧ b < x :=
sorry

end relationship_between_variables_l13_13640


namespace more_than_10_weights_missing_l13_13668

/-- 
Given weights of 5, 24, and 43 grams with an equal number of each type
and that the total remaining mass is 606060...60 grams,
prove that more than 10 weights are missing.
-/
theorem more_than_10_weights_missing (total_mass : ℕ) (n : ℕ) (k : ℕ) 
  (total_mass_eq : total_mass = k * (5 + 24 + 43))
  (total_mass_mod : total_mass % 72 ≠ 0) :
  k < n - 10 :=
sorry

end more_than_10_weights_missing_l13_13668


namespace red_ball_value_l13_13083

theorem red_ball_value (r b g : ℕ) (blue_points green_points : ℕ)
  (h1 : blue_points = 4)
  (h2 : green_points = 5)
  (h3 : b = g)
  (h4 : r^4 * blue_points^b * green_points^g = 16000)
  (h5 : b = 6) :
  r = 1 :=
by
  sorry

end red_ball_value_l13_13083


namespace remainder_of_7_pow_205_mod_12_l13_13608

theorem remainder_of_7_pow_205_mod_12 : (7^205) % 12 = 7 :=
by
  sorry

end remainder_of_7_pow_205_mod_12_l13_13608


namespace total_fruits_picked_l13_13035

variable (L M P B : Nat)

theorem total_fruits_picked (hL : L = 25) (hM : M = 32) (hP : P = 12) (hB : B = 18) : L + M + P = 69 :=
by
  sorry

end total_fruits_picked_l13_13035


namespace messages_after_noon_l13_13131

theorem messages_after_noon (t n : ℕ) (h1 : t = 39) (h2 : n = 21) : t - n = 18 := by
  sorry

end messages_after_noon_l13_13131


namespace heartsuit_example_l13_13389

def heartsuit (x y: ℤ) : ℤ := 4 * x + 6 * y

theorem heartsuit_example : heartsuit 3 8 = 60 :=
by
  sorry

end heartsuit_example_l13_13389


namespace glass_original_water_l13_13417

theorem glass_original_water 
  (O : ℝ)  -- Ounces of water originally in the glass
  (evap_per_day : ℝ)  -- Ounces of water evaporated per day
  (total_days : ℕ)    -- Total number of days evaporation occurs
  (percent_evaporated : ℝ)  -- Percentage of the original amount that evaporated
  (h1 : evap_per_day = 0.06)  -- 0.06 ounces of water evaporated each day
  (h2 : total_days = 20)  -- Evaporation occurred over a period of 20 days
  (h3 : percent_evaporated = 0.12)  -- 12% of the original amount evaporated during this period
  (h4 : evap_per_day * total_days = 1.2)  -- 0.06 ounces per day for 20 days total gives 1.2 ounces
  (h5 : percent_evaporated * O = evap_per_day * total_days) :  -- 1.2 ounces is 12% of the original amount
  O = 10 :=  -- Prove that the original amount is 10 ounces
sorry

end glass_original_water_l13_13417


namespace checker_on_diagonal_l13_13052

theorem checker_on_diagonal
  (board : ℕ)
  (n_checkers : ℕ)
  (symmetric : (ℕ → ℕ → Prop))
  (diag_check : ∀ i j, symmetric i j -> symmetric j i)
  (num_checkers_odd : Odd n_checkers)
  (board_size : board = 25)
  (checkers : n_checkers = 25) :
  ∃ i, i < 25 ∧ symmetric i i := by
  sorry

end checker_on_diagonal_l13_13052


namespace sides_of_triangle_expr_negative_l13_13430

theorem sides_of_triangle_expr_negative (a b c : ℝ) 
(h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
(a - c)^2 - b^2 < 0 :=
sorry

end sides_of_triangle_expr_negative_l13_13430


namespace average_speed_correct_l13_13146

-- Definitions of distances and speeds
def distance1 := 50 -- miles
def speed1 := 20 -- miles per hour
def distance2 := 20 -- miles
def speed2 := 40 -- miles per hour
def distance3 := 30 -- miles
def speed3 := 15 -- miles per hour

-- Definition of total distance and total time
def total_distance := distance1 + distance2 + distance3
def time1 := distance1 / speed1
def time2 := distance2 / speed2
def time3 := distance3 / speed3
def total_time := time1 + time2 + time3

-- Definition of average speed
def average_speed := total_distance / total_time

-- Statement to be proven
theorem average_speed_correct : average_speed = 20 := 
by 
  -- Proof will be provided here
  sorry

end average_speed_correct_l13_13146


namespace jade_transactions_correct_l13_13860

-- Definitions for the conditions
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + (mabel_transactions * 10 / 100)
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := cal_transactions + 16

-- The theorem stating what we want to prove
theorem jade_transactions_correct : jade_transactions = 82 := by
  sorry

end jade_transactions_correct_l13_13860


namespace num_machines_first_scenario_l13_13870

theorem num_machines_first_scenario (r : ℝ) (n : ℕ) :
  (∀ r, (2 : ℝ) * r * 24 = 1) →
  (∀ r, (n : ℝ) * r * 6 = 1) →
  n = 8 :=
by
  intros h1 h2
  sorry

end num_machines_first_scenario_l13_13870


namespace jafaris_candy_l13_13350

-- Define the conditions
variable (candy_total : Nat)
variable (taquon_candy : Nat)
variable (mack_candy : Nat)

-- Assume the conditions from the problem
axiom candy_total_def : candy_total = 418
axiom taquon_candy_def : taquon_candy = 171
axiom mack_candy_def : mack_candy = 171

-- Define the statement to be proved
theorem jafaris_candy : (candy_total - (taquon_candy + mack_candy)) = 76 :=
by
  -- Proof goes here
  sorry

end jafaris_candy_l13_13350


namespace sum_of_inserted_numbers_l13_13778

theorem sum_of_inserted_numbers (x y : ℝ) (r : ℝ) 
  (h1 : 4 * r = x) 
  (h2 : 4 * r^2 = y) 
  (h3 : (2 / y) = ((1 / x) + (1 / 16))) :
  x + y = 8 :=
sorry

end sum_of_inserted_numbers_l13_13778


namespace percentage_both_colors_l13_13220

theorem percentage_both_colors
  (total_flags : ℕ)
  (even_flags : total_flags % 2 = 0)
  (C : ℕ)
  (total_flags_eq : total_flags = 2 * C)
  (blue_percent : ℕ)
  (blue_percent_eq : blue_percent = 60)
  (red_percent : ℕ)
  (red_percent_eq : red_percent = 65) :
  ∃ both_colors_percent : ℕ, both_colors_percent = 25 :=
by
  sorry

end percentage_both_colors_l13_13220


namespace problem_1_problem_2_l13_13489

-- Definitions and conditions for the problems
def A : Set ℝ := { x | abs (x - 2) < 3 }
def B (m : ℝ) : Set ℝ := { x | x^2 - 2 * x - m < 0 }

-- Problem (I)
theorem problem_1 : (A ∩ (Set.univ \ B 3)) = { x | 3 ≤ x ∧ x < 5 } :=
sorry

-- Problem (II)
theorem problem_2 (m : ℝ) : (A ∩ B m = { x | -1 < x ∧ x < 4 }) → m = 8 :=
sorry

end problem_1_problem_2_l13_13489


namespace simplify_fractions_l13_13484

theorem simplify_fractions:
  (3 / 462 : ℚ) + (28 / 42 : ℚ) = 311 / 462 := sorry

end simplify_fractions_l13_13484


namespace arithmetic_sequence_min_sum_l13_13602

theorem arithmetic_sequence_min_sum (x : ℝ) (d : ℝ) (h₁ : d > 0) :
  (∃ n : ℕ, n > 0 ∧ (n^2 - 4 * n < 0) ∧ (n = 6 ∨ n = 7)) :=
by
  sorry

end arithmetic_sequence_min_sum_l13_13602


namespace digit_place_value_ratio_l13_13679

theorem digit_place_value_ratio : 
  let num := 43597.2468
  let digit5_place_value := 10    -- tens place
  let digit2_place_value := 0.1   -- tenths place
  digit5_place_value / digit2_place_value = 100 := 
by 
  sorry

end digit_place_value_ratio_l13_13679


namespace does_not_balance_l13_13558

variables (square odot circ triangle O : ℝ)

-- Conditions represented as hypothesis
def condition1 : Prop := 4 * square = odot + circ
def condition2 : Prop := 2 * circ + odot = 2 * triangle

-- Statement to be proved
theorem does_not_balance (h1 : condition1 square odot circ) (h2 : condition2 circ odot triangle)
 : ¬(2 * triangle + square = triangle + odot + square) := 
sorry

end does_not_balance_l13_13558


namespace calculation_result_l13_13955

theorem calculation_result : 50 + 50 / 50 + 50 = 101 := by
  sorry

end calculation_result_l13_13955


namespace least_number_of_stamps_l13_13708

theorem least_number_of_stamps (p q : ℕ) (h : 5 * p + 4 * q = 50) : p + q = 11 :=
sorry

end least_number_of_stamps_l13_13708


namespace sum_of_remainders_mod_l13_13173

theorem sum_of_remainders_mod (a b c : ℕ) (h1 : a % 53 = 31) (h2 : b % 53 = 22) (h3 : c % 53 = 7) :
  (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_remainders_mod_l13_13173


namespace marathon_time_l13_13359

noncomputable def marathon_distance : ℕ := 26
noncomputable def first_segment_distance : ℕ := 10
noncomputable def first_segment_time : ℕ := 1
noncomputable def remaining_distance : ℕ := marathon_distance - first_segment_distance
noncomputable def pace_percentage : ℕ := 80
noncomputable def initial_pace : ℕ := first_segment_distance / first_segment_time
noncomputable def remaining_pace : ℕ := (initial_pace * pace_percentage) / 100
noncomputable def remaining_time : ℕ := remaining_distance / remaining_pace
noncomputable def total_time : ℕ := first_segment_time + remaining_time

theorem marathon_time : total_time = 3 := by
  -- Proof omitted: hence using sorry
  sorry

end marathon_time_l13_13359


namespace sum_X_Y_Z_l13_13057

theorem sum_X_Y_Z (X Y Z : ℕ) (hX : X ∈ Finset.range 10) (hY : Y ∈ Finset.range 10) (hZ : Z = 0)
     (div9 : (1 + 3 + 0 + 7 + 6 + 7 + 4 + X + 2 + 0 + Y + 0 + 0 + 8 + 0) % 9 = 0) 
     (div7 : (307674 * 10 + X * 20 + Y * 10 + 800) % 7 = 0) :
  X + Y + Z = 7 := 
sorry

end sum_X_Y_Z_l13_13057


namespace value_of_a_l13_13431

-- Conditions
def A (a : ℝ) : Set ℝ := {2, a}
def B (a : ℝ) : Set ℝ := {-1, a^2 - 2}

-- Theorem statement asserting the condition and the correct answer
theorem value_of_a (a : ℝ) : (A a ∩ B a).Nonempty → a = -2 :=
by
  sorry

end value_of_a_l13_13431


namespace waiter_customers_l13_13154

variable (initial_customers left_customers new_customers : ℕ)

theorem waiter_customers 
  (h1 : initial_customers = 33)
  (h2 : left_customers = 31)
  (h3 : new_customers = 26) :
  (initial_customers - left_customers + new_customers = 28) := 
by
  sorry

end waiter_customers_l13_13154


namespace snake_alligator_consumption_l13_13749

theorem snake_alligator_consumption :
  (616 / 7) = 88 :=
by
  sorry

end snake_alligator_consumption_l13_13749


namespace paint_per_door_l13_13969

variable (cost_per_pint : ℕ) (cost_per_gallon : ℕ) (num_doors : ℕ) (pints_per_gallon : ℕ) (savings : ℕ)

theorem paint_per_door :
  cost_per_pint = 8 →
  cost_per_gallon = 55 →
  num_doors = 8 →
  pints_per_gallon = 8 →
  savings = 9 →
  (pints_per_gallon / num_doors = 1) :=
by
  intros h_cpint h_cgallon h_nd h_pgallon h_savings
  sorry

end paint_per_door_l13_13969


namespace parabola_vertex_l13_13250

theorem parabola_vertex (c d : ℝ) (h : ∀ x : ℝ, - x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  (∃ v : ℝ × ℝ, v = (5, 1)) :=
sorry

end parabola_vertex_l13_13250


namespace range_of_m_l13_13914

noncomputable def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
noncomputable def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ¬p x → ¬q x m) → (m ≥ 9) :=
by
  sorry

end range_of_m_l13_13914


namespace decimal_to_fraction_l13_13882

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l13_13882


namespace find_ab_l13_13938

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) :
  a * b = 10 :=
by
  sorry

end find_ab_l13_13938


namespace total_copies_l13_13261

theorem total_copies (rate1 : ℕ) (rate2 : ℕ) (time : ℕ) (total : ℕ) 
  (h1 : rate1 = 25) (h2 : rate2 = 55) (h3 : time = 30) : 
  total = rate1 * time + rate2 * time := 
  sorry

end total_copies_l13_13261


namespace axis_of_symmetry_cosine_l13_13980

theorem axis_of_symmetry_cosine (x : ℝ) : 
  (∃ k : ℤ, 2 * x + π / 3 = k * π) → x = -π / 6 :=
sorry

end axis_of_symmetry_cosine_l13_13980


namespace divisibility_of_powers_l13_13377

theorem divisibility_of_powers (a b c d m : ℤ) (h_odd : m % 2 = 1)
  (h_sum_div : m ∣ (a + b + c + d))
  (h_sum_squares_div : m ∣ (a^2 + b^2 + c^2 + d^2)) : 
  m ∣ (a^4 + b^4 + c^4 + d^4 + 4 * a * b * c * d) :=
sorry

end divisibility_of_powers_l13_13377


namespace shaded_region_area_l13_13845

noncomputable def radius_large : ℝ := 10
noncomputable def radius_small : ℝ := 4

theorem shaded_region_area :
  let area_large := Real.pi * radius_large^2 
  let area_small := Real.pi * radius_small^2 
  (area_large - 2 * area_small) = 68 * Real.pi :=
by
  sorry

end shaded_region_area_l13_13845


namespace a5_value_l13_13418

theorem a5_value (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a2 - a1 = 2)
  (h2 : a3 - a2 = 4)
  (h3 : a4 - a3 = 8)
  (h4 : a5 - a4 = 16) :
  a5 = 31 := by
  sorry

end a5_value_l13_13418


namespace sum_of_three_distinct_l13_13909

def S : Set ℤ := {2, 5, 8, 11, 14, 17, 20}

theorem sum_of_three_distinct (S : Set ℤ) (h : S = {2, 5, 8, 11, 14, 17, 20}) :
  (∃ n : ℕ, n = 13 ∧ ∀ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c → 
    ∃ k : ℕ, a + b + c = 3 * k) := 
by  -- The proof goes here.
  sorry

end sum_of_three_distinct_l13_13909


namespace homework_duration_equation_l13_13153

-- Given conditions
def initial_duration : ℝ := 120
def final_duration : ℝ := 60
variable (x : ℝ)

-- The goal is to prove that the appropriate equation holds
theorem homework_duration_equation : initial_duration * (1 - x)^2 = final_duration := 
sorry

end homework_duration_equation_l13_13153


namespace negation_of_universal_proposition_l13_13820

variable (p : Prop)
variable (x : ℝ)

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 1 > 0)) ↔ (∃ x : ℝ, x^2 - 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l13_13820


namespace equal_distribution_arithmetic_sequence_l13_13363

theorem equal_distribution_arithmetic_sequence :
  ∃ a d : ℚ, (a - 2 * d) + (a - d) = (a + (a + d) + (a + 2 * d)) ∧
  5 * a = 5 ∧
  a + 2 * d = 2 / 3 :=
by
  sorry

end equal_distribution_arithmetic_sequence_l13_13363


namespace abs_monotonic_increasing_even_l13_13166

theorem abs_monotonic_increasing_even :
  (∀ x : ℝ, |x| = |(-x)|) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → |x1| ≤ |x2|) :=
by
  sorry

end abs_monotonic_increasing_even_l13_13166


namespace negation_of_proposition_l13_13599

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0)) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by
  sorry

end negation_of_proposition_l13_13599


namespace germs_per_dish_calc_l13_13490

theorem germs_per_dish_calc :
    let total_germs := 0.036 * 10^5
    let total_dishes := 36000 * 10^(-3)
    (total_germs / total_dishes) = 100 := by
    sorry

end germs_per_dish_calc_l13_13490


namespace coin_exchange_impossible_l13_13170

theorem coin_exchange_impossible :
  ∀ (n : ℕ), (n % 4 = 1) → (¬ (∃ k : ℤ, n + 4 * k = 26)) :=
by
  intros n h
  sorry

end coin_exchange_impossible_l13_13170


namespace cost_for_flour_for_two_cakes_l13_13480

theorem cost_for_flour_for_two_cakes 
    (packages_per_cake : ℕ)
    (cost_per_package : ℕ)
    (cakes : ℕ) 
    (total_cost : ℕ)
    (H1 : packages_per_cake = 2)
    (H2 : cost_per_package = 3)
    (H3 : cakes = 2)
    (H4 : total_cost = 12) :
    total_cost = cakes * packages_per_cake * cost_per_package := 
by 
    rw [H1, H2, H3]
    sorry

end cost_for_flour_for_two_cakes_l13_13480


namespace XiaoMing_reading_problem_l13_13019

theorem XiaoMing_reading_problem :
  ∀ (total_pages days first_days first_rate remaining_rate : ℕ),
    total_pages = 72 →
    days = 10 →
    first_days = 2 →
    first_rate = 5 →
    (first_days * first_rate) + ((days - first_days) * remaining_rate) ≥ total_pages →
    remaining_rate ≥ 8 :=
by
  intros total_pages days first_days first_rate remaining_rate
  intro h1 h2 h3 h4 h5
  sorry

end XiaoMing_reading_problem_l13_13019


namespace number_of_articles_l13_13744

-- Define the conditions
def gain := 1 / 9
def cp_one_article := 1  -- cost price of one article

-- Define the cost price for x articles
def cp (x : ℕ) := x * cp_one_article

-- Define the selling price for 45 articles
def sp (x : ℕ) := x / 45

-- Define the selling price equation considering gain
def sp_one_article := (cp_one_article * (1 + gain))

-- Main theorem to prove
theorem number_of_articles (x : ℕ) (h : sp x = sp_one_article) : x = 50 :=
by
  sorry

-- The theorem imports all necessary conditions and definitions and prepares the problem for proof.

end number_of_articles_l13_13744


namespace sequence_u5_eq_27_l13_13034

theorem sequence_u5_eq_27 (u : ℕ → ℝ) 
  (h_recurrence : ∀ n, u (n + 2) = 3 * u (n + 1) - 2 * u n)
  (h_u3 : u 3 = 15)
  (h_u6 : u 6 = 43) :
  u 5 = 27 :=
  sorry

end sequence_u5_eq_27_l13_13034


namespace tan_2x_value_l13_13468

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) := deriv f x

theorem tan_2x_value (x : ℝ) (h : f' x = 3 * f x) : Real.tan (2 * x) = (4/3) := by
  sorry

end tan_2x_value_l13_13468


namespace problem_l13_13017

theorem problem (a b c : ℝ) (h : 1/a + 1/b + 1/c = 1/(a + b + c)) : (a + b) * (b + c) * (a + c) = 0 := 
by
  sorry

end problem_l13_13017


namespace box_dimensions_l13_13971

theorem box_dimensions {a b c : ℕ} (h1 : a + c = 17) (h2 : a + b = 13) (h3 : 2 * (b + c) = 40) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  sorry
}

end box_dimensions_l13_13971


namespace max_a_inequality_l13_13776

theorem max_a_inequality (a : ℝ) :
  (∀ x : ℝ, x * a ≤ Real.exp (x - 1) + x^2 + 1) → a ≤ 3 := 
sorry

end max_a_inequality_l13_13776


namespace max_ounces_amber_can_get_l13_13126

theorem max_ounces_amber_can_get :
  let money := 7
  let candy_cost := 1
  let candy_ounces := 12
  let chips_cost := 1.40
  let chips_ounces := 17
  let max_ounces := max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces)
  max_ounces = 85 := 
by
  sorry

end max_ounces_amber_can_get_l13_13126


namespace find_smallest_d_l13_13361

theorem find_smallest_d (d : ℕ) : (5 + 6 + 2 + 4 + 8 + d) % 9 = 0 → d = 2 :=
by
  sorry

end find_smallest_d_l13_13361


namespace smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l13_13949

theorem smallest_integer_sum_of_squares_and_cubes :
  ∃ (n : ℕ) (a b c d : ℕ), n > 2 ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 ∧
  ∀ (m : ℕ) (x y u v : ℕ), (m > 2 ∧ m = x^2 + y^2 ∧ m = u^3 + v^3) → n ≤ m := 
sorry

theorem infinite_integers_sum_of_squares_and_cubes :
  ∀ (k : ℕ), ∃ (n : ℕ) (a b c d : ℕ), n = 1 + 2^(6*k) ∧ n = a^2 + b^2 ∧ n = c^3 + d^3 :=
sorry

end smallest_integer_sum_of_squares_and_cubes_infinite_integers_sum_of_squares_and_cubes_l13_13949


namespace van_distance_l13_13024

theorem van_distance (D : ℝ) (t_initial t_new : ℝ) (speed_new : ℝ) 
  (h1 : t_initial = 6) 
  (h2 : t_new = (3 / 2) * t_initial) 
  (h3 : speed_new = 30) 
  (h4 : D = speed_new * t_new) : 
  D = 270 :=
by
  sorry

end van_distance_l13_13024


namespace percentage_repeated_digits_five_digit_numbers_l13_13842

theorem percentage_repeated_digits_five_digit_numbers : 
  let total_five_digit_numbers := 90000
  let non_repeated_digits_number := 9 * 9 * 8 * 7 * 6
  let repeated_digits_number := total_five_digit_numbers - non_repeated_digits_number
  let y := (repeated_digits_number.toFloat / total_five_digit_numbers.toFloat) * 100 
  y = 69.8 :=
by
  sorry

end percentage_repeated_digits_five_digit_numbers_l13_13842


namespace general_term_sequence_l13_13808

def seq (a : ℕ → ℤ) : Prop :=
  a 0 = 3 ∧ a 1 = 9 ∧ ∀ n ≥ 2, a n = 4 * a (n - 1) - 3 * a (n - 2) - 4 * n + 2

theorem general_term_sequence (a : ℕ → ℤ) (h : seq a) : 
  ∀ n, a n = 3^n + n^2 + 3 * n + 2 :=
by
  sorry

end general_term_sequence_l13_13808


namespace set_intersection_complement_l13_13618

open Set

theorem set_intersection_complement (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  (U \ A) ∩ B = {3} :=
by
  sorry

end set_intersection_complement_l13_13618


namespace find_base_l13_13767

noncomputable def f (a x : ℝ) := 1 + (Real.log x) / (Real.log a)

theorem find_base (a : ℝ) (hinv_pass : (∀ y : ℝ, (∀ x : ℝ, f a x = y → x = 4 → y = 3))) : a = 2 :=
by
  sorry

end find_base_l13_13767


namespace new_average_age_l13_13958

theorem new_average_age (n_students : ℕ) (average_student_age : ℕ) (teacher_age : ℕ)
  (h_students : n_students = 50)
  (h_average_student_age : average_student_age = 14)
  (h_teacher_age : teacher_age = 65) :
  (n_students * average_student_age + teacher_age) / (n_students + 1) = 15 :=
by
  sorry

end new_average_age_l13_13958


namespace B_time_to_complete_work_l13_13577

variable {W : ℝ} {R_b : ℝ} {T_b : ℝ}

theorem B_time_to_complete_work (h1 : 3 * R_b * (T_b - 10) = R_b * T_b) : T_b = 15 :=
by
  sorry

end B_time_to_complete_work_l13_13577


namespace smallest_x_exists_l13_13532

theorem smallest_x_exists (x k m : ℤ) 
    (h1 : x + 3 = 7 * k) 
    (h2 : x - 5 = 8 * m) 
    (h3 : ∀ n : ℤ, ((n + 3) % 7 = 0) ∧ ((n - 5) % 8 = 0) → x ≤ n) : 
    x = 53 := by
  sorry

end smallest_x_exists_l13_13532


namespace systematic_sampling_methods_l13_13666

-- Definitions for sampling methods ①, ②, ④
def sampling_method_1 : Prop :=
  ∀ (l : ℕ), (l ≤ 15 ∧ l + 5 ≤ 15 ∧ l + 10 ≤ 15 ∨
              l ≤ 15 ∧ l + 5 ≤ 20 ∧ l + 10 ≤ 20) → True

def sampling_method_2 : Prop :=
  ∀ (t : ℕ), (t % 5 = 0) → True

def sampling_method_3 : Prop :=
  ∀ (n : ℕ), (n > 0) → True

def sampling_method_4 : Prop :=
  ∀ (row : ℕ) (seat : ℕ), (seat = 12) → True

-- Equivalence Proof Statement
theorem systematic_sampling_methods :
  sampling_method_1 ∧ sampling_method_2 ∧ sampling_method_4 :=
by sorry

end systematic_sampling_methods_l13_13666


namespace total_games_won_l13_13519

theorem total_games_won (Betsy_games : ℕ) (Helen_games : ℕ) (Susan_games : ℕ) 
    (hBetsy : Betsy_games = 5)
    (hHelen : Helen_games = 2 * Betsy_games)
    (hSusan : Susan_games = 3 * Betsy_games) : 
    Betsy_games + Helen_games + Susan_games = 30 :=
sorry

end total_games_won_l13_13519


namespace proof_problem_l13_13915

noncomputable def problem_equivalent_proof (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧
  (z + 6 = 2 * y - z) ∧
  (x + 8 * z = y + 2) →
  (x^2 + y^2 + z^2 = 21)

theorem proof_problem (x y z : ℝ) : problem_equivalent_proof x y z :=
by
  sorry

end proof_problem_l13_13915


namespace sum_product_of_integers_l13_13045

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end sum_product_of_integers_l13_13045


namespace intersection_of_M_and_complementN_l13_13231

def UniversalSet := Set ℝ
def setM : Set ℝ := {-1, 0, 1, 3}
def setN : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def complementSetN : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_of_M_and_complementN :
  setM ∩ complementSetN = {0, 1} :=
sorry

end intersection_of_M_and_complementN_l13_13231


namespace c_plus_d_l13_13974

theorem c_plus_d (c d : ℝ)
  (h1 : c^3 - 12 * c^2 + 15 * c - 36 = 0)
  (h2 : 6 * d^3 - 36 * d^2 - 150 * d + 1350 = 0) :
  c + d = 7 := 
  sorry

end c_plus_d_l13_13974


namespace kinetic_energy_reduction_collisions_l13_13936

theorem kinetic_energy_reduction_collisions (E_0 : ℝ) (n : ℕ) :
  (1 / 2)^n * E_0 = E_0 / 64 → n = 6 :=
by
  sorry

end kinetic_energy_reduction_collisions_l13_13936


namespace product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l13_13423

theorem product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half :
  (∀ x : ℝ, (x + 1/x = 3 * x) → (x = 1/Real.sqrt 2 ∨ x = -1/Real.sqrt 2)) →
  (∀ x y : ℝ, (x = 1/Real.sqrt 2) → (y = -1/Real.sqrt 2) →
  x * y = -1/2) :=
by
  intros h h1 h2
  sorry

end product_of_real_numbers_trippled_when_added_to_reciprocal_is_minus_one_half_l13_13423


namespace first_tribe_term_is_longer_l13_13878

def years_to_days_first_tribe (years : ℕ) : ℕ := 
  years * 12 * 30

def months_to_days_first_tribe (months : ℕ) : ℕ :=
  months * 30

def total_days_first_tribe (years months days : ℕ) : ℕ :=
  (years_to_days_first_tribe years) + (months_to_days_first_tribe months) + days

def years_to_days_second_tribe (years : ℕ) : ℕ := 
  years * 13 * 4 * 7

def moons_to_days_second_tribe (moons : ℕ) : ℕ :=
  moons * 4 * 7

def weeks_to_days_second_tribe (weeks : ℕ) : ℕ :=
  weeks * 7

def total_days_second_tribe (years moons weeks days : ℕ) : ℕ :=
  (years_to_days_second_tribe years) + (moons_to_days_second_tribe moons) + (weeks_to_days_second_tribe weeks) + days

theorem first_tribe_term_is_longer :
  total_days_first_tribe 7 1 18 > total_days_second_tribe 6 12 1 3 :=
by
  sorry

end first_tribe_term_is_longer_l13_13878


namespace maynard_dog_holes_l13_13293

open Real

theorem maynard_dog_holes (h_filled : ℝ) (h_unfilled : ℝ) (percent_filled : ℝ) 
  (percent_unfilled : ℝ) (total_holes : ℝ) :
  percent_filled = 0.75 →
  percent_unfilled = 0.25 →
  h_unfilled = 2 →
  h_filled = total_holes * percent_filled →
  total_holes = 8 :=
by
  intros hf pu hu hf_total
  sorry

end maynard_dog_holes_l13_13293


namespace find_c_find_A_l13_13088

open Real

noncomputable def acute_triangle_sides (A B C a b c : ℝ) : Prop :=
  a = b * cos C + (sqrt 3 / 3) * c * sin B

theorem find_c (A B C a b c : ℝ) (ha : a = 2) (hb : b = sqrt 7) 
  (hab : acute_triangle_sides A B C a b c) : c = 3 := 
sorry

theorem find_A (A B C : ℝ) (h : sqrt 3 * sin (2 * A - π / 6) - 2 * (sin (C - π / 12))^2 = 0)
  (h_range : π / 6 < A ∧ A < π / 2) : A = π / 4 :=
sorry

end find_c_find_A_l13_13088


namespace trig_expression_value_l13_13116

theorem trig_expression_value (α : ℝ) (h₁ : Real.tan (α + π / 4) = -1/2) (h₂ : π / 2 < α ∧ α < π) :
  (Real.sin (2 * α) - 2 * (Real.cos α)^2) / Real.sin (α - π / 4) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end trig_expression_value_l13_13116


namespace find_seventh_term_l13_13379

theorem find_seventh_term :
  ∃ r : ℚ, ∃ (a₁ a₇ a₁₀ : ℚ), 
    a₁ = 12 ∧ 
    a₁₀ = 78732 ∧ 
    a₇ = a₁ * r^6 ∧ 
    a₁₀ = a₁ * r^9 ∧ 
    a₇ = 8748 :=
by
  sorry

end find_seventh_term_l13_13379


namespace find_x2_plus_y2_l13_13783

theorem find_x2_plus_y2 : ∀ (x y : ℝ),
  3 * x + 4 * y = 30 →
  x + 2 * y = 13 →
  x^2 + y^2 = 36.25 :=
by
  intros x y h1 h2
  sorry

end find_x2_plus_y2_l13_13783


namespace boys_camp_total_l13_13398

theorem boys_camp_total (T : ℕ) 
  (h1 : 0.20 * T = (0.20 : ℝ) * T) 
  (h2 : (0.30 : ℝ) * (0.20 * T) = (0.30 : ℝ) * (0.20 * T)) 
  (h3 : (0.70 : ℝ) * (0.20 * T) = 63) :
  T = 450 :=
by
  sorry

end boys_camp_total_l13_13398


namespace true_discount_correct_l13_13073

noncomputable def true_discount (banker_gain : ℝ) (average_rate : ℝ) (time_years : ℝ) : ℝ :=
  let r := average_rate
  let t := time_years
  let exp_factor := Real.exp (-r * t)
  let face_value := banker_gain / (1 - exp_factor)
  face_value - (face_value * exp_factor)

theorem true_discount_correct : 
  true_discount 15.8 0.145 5 = 15.8 := 
by
  sorry

end true_discount_correct_l13_13073


namespace percent_of_x_is_y_l13_13877

variable (x y : ℝ)

theorem percent_of_x_is_y (h : 0.20 * (x - y) = 0.15 * (x + y)) : (y / x) * 100 = 100 / 7 :=
by
  sorry

end percent_of_x_is_y_l13_13877


namespace prove_x_ge_neg_one_sixth_l13_13340

variable (x y : ℝ)

theorem prove_x_ge_neg_one_sixth (h : x^4 * y^2 + y^4 + 2 * x^3 * y + 6 * x^2 * y + x^2 + 8 ≤ 0) :
  x ≥ -1 / 6 :=
sorry

end prove_x_ge_neg_one_sixth_l13_13340


namespace quiz_competition_top_three_orders_l13_13346

theorem quiz_competition_top_three_orders :
  let participants := 4
  let top_positions := 3
  let permutations := (Nat.factorial participants) / (Nat.factorial (participants - top_positions))
  permutations = 24 := 
by
  sorry

end quiz_competition_top_three_orders_l13_13346


namespace translate_function_right_by_2_l13_13802

theorem translate_function_right_by_2 (x : ℝ) : 
  (∀ x, (x - 2) ^ 2 + (x - 2) = x ^ 2 - 3 * x + 2) := 
by 
  sorry

end translate_function_right_by_2_l13_13802


namespace tim_age_difference_l13_13204

theorem tim_age_difference (j_turned_23_j_turned_35 : ∃ (j_age_when_james_23 : ℕ) (john_age_when_james_23 : ℕ), 
                                          j_age_when_james_23 = 23 ∧ john_age_when_james_23 = 35)
                           (tim_age : ℕ) (tim_age_eq : tim_age = 79)
                           (tim_age_twice_john_age_less_X : ∃ (X : ℕ) (john_age : ℕ), tim_age = 2 * john_age - X) :
  ∃ (X : ℕ), X = 15 :=
by
  sorry

end tim_age_difference_l13_13204


namespace problem1_problem2_l13_13507

theorem problem1 : |2 - Real.sqrt 3| - 2^0 - Real.sqrt 12 = 1 - 3 * Real.sqrt 3 := 
by 
  sorry
  
theorem problem2 : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1) ^ 2 = 14 + 4 * Real.sqrt 3 := 
by 
  sorry

end problem1_problem2_l13_13507


namespace sum_of_consecutive_integers_l13_13690

theorem sum_of_consecutive_integers (x y : ℕ) (h1 : y = x + 1) (h2 : x * y = 812) : x + y = 57 :=
by
  -- proof skipped
  sorry

end sum_of_consecutive_integers_l13_13690


namespace tank_capacity_l13_13011

theorem tank_capacity (x : ℝ) (h₁ : 0.25 * x = 60) (h₂ : 0.05 * x = 12) : x = 240 :=
sorry

end tank_capacity_l13_13011


namespace number_of_valid_3_digit_numbers_l13_13252

theorem number_of_valid_3_digit_numbers : 
  ∃ (n : ℕ), 
    (∀ (h t u : ℕ), 
      (n = h * 100 + t * 10 + u) ∧ 
      (1 ≤ h ∧ h ≤ 9) ∧ 
      (0 ≤ t ∧ t ≤ 9) ∧ 
      (0 ≤ u ∧ u ≤ 9) ∧ 
      (u ≥ 3 * t)) →
      n = 198 := 
by
  sorry

end number_of_valid_3_digit_numbers_l13_13252


namespace least_possible_value_of_z_minus_x_l13_13676

theorem least_possible_value_of_z_minus_x 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : x < y) 
  (h2 : y < z) 
  (h3 : y - x > 5) : 
  z - x = 9 :=
sorry

end least_possible_value_of_z_minus_x_l13_13676


namespace simplify_expression_l13_13570

noncomputable def i : ℂ := Complex.I

theorem simplify_expression : 7*(4 - 2*i) + 4*i*(3 - 2*i) = 36 - 2*i :=
by
  sorry

end simplify_expression_l13_13570


namespace percent_x_of_w_l13_13477

theorem percent_x_of_w (x y z w : ℝ)
  (h1 : x = 1.2 * y)
  (h2 : y = 0.7 * z)
  (h3 : w = 1.5 * z) : (x / w) * 100 = 56 :=
by
  sorry

end percent_x_of_w_l13_13477


namespace building_height_l13_13670

theorem building_height (h : ℕ) 
  (shadow_building : ℕ) 
  (shadow_pole : ℕ) 
  (height_pole : ℕ) 
  (ratio_proportional : shadow_building * height_pole = shadow_pole * h) 
  (shadow_building_val : shadow_building = 63) 
  (shadow_pole_val : shadow_pole = 32) 
  (height_pole_val : height_pole = 28) : 
  h = 55 := 
by 
  sorry

end building_height_l13_13670


namespace coordinate_inequality_l13_13025

theorem coordinate_inequality (x y : ℝ) :
  (xy > 0 → (x - 2)^2 + (y + 1)^2 < 5) ∧ (xy < 0 → (x - 2)^2 + (y + 1)^2 > 5) :=
by
  sorry

end coordinate_inequality_l13_13025


namespace no_three_in_range_l13_13828

theorem no_three_in_range (c : ℝ) : c > 4 → ¬ (∃ x : ℝ, x^2 + 2 * x + c = 3) :=
by
  sorry

end no_three_in_range_l13_13828


namespace find_subtracted_value_l13_13609

theorem find_subtracted_value (x y : ℕ) (h1 : x = 120) (h2 : 2 * x - y = 102) : y = 138 :=
by
  sorry

end find_subtracted_value_l13_13609


namespace total_wheels_in_garage_l13_13360

theorem total_wheels_in_garage 
    (num_bicycles : ℕ)
    (num_cars : ℕ)
    (wheels_per_bicycle : ℕ)
    (wheels_per_car : ℕ) 
    (num_bicycles_eq : num_bicycles = 9)
    (num_cars_eq : num_cars = 16)
    (wheels_per_bicycle_eq : wheels_per_bicycle = 2)
    (wheels_per_car_eq : wheels_per_car = 4) :
    num_bicycles * wheels_per_bicycle + num_cars * wheels_per_car = 82 := 
by
    sorry

end total_wheels_in_garage_l13_13360


namespace product_of_B_coords_l13_13898

structure Point where
  x : ℝ
  y : ℝ

def isMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

theorem product_of_B_coords :
  ∀ (M A B : Point), 
  isMidpoint M A B →
  M = ⟨3, 7⟩ →
  A = ⟨5, 3⟩ →
  (B.x * B.y) = 11 :=
by intro M A B hM hM_def hA_def; sorry

end product_of_B_coords_l13_13898


namespace elena_hike_total_miles_l13_13866

theorem elena_hike_total_miles (x1 x2 x3 x4 x5 : ℕ)
  (h1 : x1 + x2 = 36)
  (h2 : x2 + x3 = 40)
  (h3 : x3 + x4 + x5 = 45)
  (h4 : x1 + x4 = 38) : 
  x1 + x2 + x3 + x4 + x5 = 81 := 
sorry

end elena_hike_total_miles_l13_13866


namespace find_expression_value_l13_13455

theorem find_expression_value (a b : ℝ)
  (h1 : a^2 - a - 3 = 0)
  (h2 : b^2 - b - 3 = 0) :
  2 * a^3 + b^2 + 3 * a^2 - 11 * a - b + 5 = 23 :=
  sorry

end find_expression_value_l13_13455


namespace graph_f_intersects_x_eq_1_at_most_once_l13_13526

-- Define a function f from ℝ to ℝ
def f : ℝ → ℝ := sorry  -- Placeholder for the actual function

-- Define the domain of the function f (it's a generic function on ℝ for simplicity)
axiom f_unique : ∀ x y : ℝ, f x = f y → x = y  -- If f(x) = f(y), then x must equal y

-- Prove that the graph of y = f(x) intersects the line x = 1 at most once
theorem graph_f_intersects_x_eq_1_at_most_once : ∃ y : ℝ, (f 1 = y) ∨ (¬∃ y : ℝ, f 1 = y) :=
by
  -- Proof goes here
  sorry

end graph_f_intersects_x_eq_1_at_most_once_l13_13526


namespace yardwork_payment_l13_13773

theorem yardwork_payment :
  let earnings := [15, 20, 25, 40]
  let total_earnings := List.sum earnings
  let equal_share := total_earnings / earnings.length
  let high_earner := 40
  high_earner - equal_share = 15 :=
by
  sorry

end yardwork_payment_l13_13773


namespace length_of_lunch_break_is_48_minutes_l13_13653

noncomputable def paula_and_assistants_lunch_break : ℝ := sorry

theorem length_of_lunch_break_is_48_minutes
  (p h L : ℝ)
  (h_monday : (9 - L) * (p + h) = 0.6)
  (h_tuesday : (7 - L) * h = 0.3)
  (h_wednesday : (10 - L) * p = 0.1) :
  L = 0.8 :=
sorry

end length_of_lunch_break_is_48_minutes_l13_13653


namespace lines_parallel_if_perpendicular_to_plane_l13_13023

variables (m n l : Line) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop := sorry
def parallel (m n : Line) : Prop := sorry

theorem lines_parallel_if_perpendicular_to_plane
  (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end lines_parallel_if_perpendicular_to_plane_l13_13023


namespace divisibility_problem_l13_13043

theorem divisibility_problem (n : ℕ) : n-1 ∣ n^n - 7*n + 5*n^2024 + 3*n^2 - 2 := 
by
  sorry

end divisibility_problem_l13_13043


namespace negative_comparison_l13_13663

theorem negative_comparison : -2023 > -2024 :=
sorry

end negative_comparison_l13_13663


namespace polynomial_remainder_l13_13777

theorem polynomial_remainder (P : ℝ → ℝ) (h1 : P 19 = 16) (h2 : P 15 = 8) : 
  ∃ Q : ℝ → ℝ, ∀ x, P x = (x - 15) * (x - 19) * Q x + 2 * x - 22 :=
by
  sorry

end polynomial_remainder_l13_13777


namespace Onum_Lake_more_trout_l13_13937

theorem Onum_Lake_more_trout (O B R : ℕ) (hB : B = 75) (hR : R = O / 2) (hAvg : (O + B + R) / 3 = 75) : O - B = 25 :=
by
  sorry

end Onum_Lake_more_trout_l13_13937


namespace number_of_solutions_l13_13511

-- Define the equation and the constraints
def equation (x y z : ℕ) : Prop := 2 * x + 3 * y + z = 800

def positive_integer (n : ℕ) : Prop := n > 0

-- The main theorem statement
theorem number_of_solutions : ∃ s, s = 127 ∧ ∀ (x y z : ℕ), positive_integer x → positive_integer y → positive_integer z → equation x y z → s = 127 :=
by
  sorry

end number_of_solutions_l13_13511


namespace sum_of_values_satisfying_l13_13946

theorem sum_of_values_satisfying (x : ℝ) (h : Real.sqrt ((x - 2) ^ 2) = 8) :
  ∃ x1 x2 : ℝ, (Real.sqrt ((x1 - 2) ^ 2) = 8) ∧ (Real.sqrt ((x2 - 2) ^ 2) = 8) ∧ x1 + x2 = 4 := 
by
  sorry

end sum_of_values_satisfying_l13_13946


namespace seq_an_general_term_and_sum_l13_13758

theorem seq_an_general_term_and_sum
  (a_n : ℕ → ℕ)
  (S : ℕ → ℕ)
  (T : ℕ → ℕ)
  (H1 : ∀ n, S n = 2 * a_n n - a_n 1)
  (H2 : ∃ d : ℕ, a_n 1 = d ∧ a_n 2 + 1 = a_n 1 + d ∧ a_n 3 = a_n 2 + d) :
  (∀ n, a_n n = 2^n) ∧ (∀ n, T n = n * 2^(n + 1) + 2 - 2^(n + 1)) := 
  by
  sorry

end seq_an_general_term_and_sum_l13_13758


namespace hoseok_more_than_minyoung_l13_13683

-- Define the initial amounts and additional earnings
def initial_amount : ℕ := 1500000
def additional_min : ℕ := 320000
def additional_hos : ℕ := 490000

-- Define the new amounts
def new_amount_min : ℕ := initial_amount + additional_min
def new_amount_hos : ℕ := initial_amount + additional_hos

-- Define the proof problem: Hoseok's new amount - Minyoung's new amount = 170000
theorem hoseok_more_than_minyoung : (new_amount_hos - new_amount_min) = 170000 :=
by
  -- The proof is skipped.
  sorry

end hoseok_more_than_minyoung_l13_13683


namespace neg_q_is_true_l13_13659

variable (p q : Prop)

theorem neg_q_is_true (hp : p) (hq : ¬ q) : ¬ q :=
by
  exact hq

end neg_q_is_true_l13_13659


namespace percent_absent_is_correct_l13_13737

theorem percent_absent_is_correct (total_students boys girls absent_boys absent_girls : ℝ) 
(h1 : total_students = 100)
(h2 : boys = 50)
(h3 : girls = 50)
(h4 : absent_boys = boys * (1 / 5))
(h5 : absent_girls = girls * (1 / 4)):
  (absent_boys + absent_girls) / total_students * 100 = 22.5 :=
by 
  sorry

end percent_absent_is_correct_l13_13737


namespace position_of_21_over_19_in_sequence_l13_13182

def sequence_term (n : ℕ) : ℚ := (n + 3) / (n + 1)

theorem position_of_21_over_19_in_sequence :
  ∃ n : ℕ, sequence_term n = 21 / 19 ∧ n = 18 :=
by sorry

end position_of_21_over_19_in_sequence_l13_13182


namespace collective_apples_l13_13732

theorem collective_apples :
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8 := by
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  show (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8
  sorry

end collective_apples_l13_13732


namespace min_distance_origin_to_line_l13_13705

noncomputable def distance_from_origin_to_line(A B C : ℝ) : ℝ :=
  let d := |A * 0 + B * 0 + C| / (Real.sqrt (A^2 + B^2))
  d

theorem min_distance_origin_to_line : distance_from_origin_to_line 1 1 (-4) = 2 * Real.sqrt 2 := by 
  sorry

end min_distance_origin_to_line_l13_13705


namespace abs_eq_4_l13_13637

theorem abs_eq_4 (a : ℝ) : |a| = 4 ↔ a = 4 ∨ a = -4 :=
by
  sorry

end abs_eq_4_l13_13637


namespace monotonic_decreasing_interval_l13_13320

noncomputable def f (x : ℝ) : ℝ := x^3 - 15 * x^2 - 33 * x + 6

theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -1 ∧ b = 11 ∧ ∀ x, x > a ∧ x < b → (deriv f x) < 0 :=
by
  sorry

end monotonic_decreasing_interval_l13_13320


namespace find_ax5_plus_by5_l13_13225

variable (a b x y : ℝ)

-- Conditions
axiom h1 : a * x + b * y = 3
axiom h2 : a * x^2 + b * y^2 = 7
axiom h3 : a * x^3 + b * y^3 = 16
axiom h4 : a * x^4 + b * y^4 = 42

-- Theorem (what we need to prove)
theorem find_ax5_plus_by5 : a * x^5 + b * y^5 = 20 :=
sorry

end find_ax5_plus_by5_l13_13225


namespace proj_w_v_is_v_l13_13651

noncomputable def proj_w_v (v w : ℝ × ℝ) : ℝ × ℝ :=
  let c := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (c * w.1, c * w.2)

def v : ℝ × ℝ := (-3, 2)
def w : ℝ × ℝ := (4, -2)

theorem proj_w_v_is_v : proj_w_v v w = v := 
  sorry

end proj_w_v_is_v_l13_13651


namespace population_multiple_of_18_l13_13289

theorem population_multiple_of_18
  (a b c P : ℕ)
  (ha : P = a^2)
  (hb : P + 200 = b^2 + 1)
  (hc : b^2 + 301 = c^2) :
  ∃ k, P = 18 * k := 
sorry

end population_multiple_of_18_l13_13289


namespace proof_subset_l13_13888

def M : Set ℝ := {x | x ≥ 0}
def N : Set ℝ := {0, 1, 2}

theorem proof_subset : N ⊆ M := sorry

end proof_subset_l13_13888


namespace problem1_problem2_l13_13892

theorem problem1 : (1 * (-9)) - (-7) + (-6) - 5 = -13 := 
by 
  -- problem1 proof
  sorry

theorem problem2 : ((-5 / 12) + (2 / 3) - (3 / 4)) * (-12) = 6 := 
by 
  -- problem2 proof
  sorry

end problem1_problem2_l13_13892


namespace green_dots_third_row_l13_13746

noncomputable def row_difference (a b : Nat) : Nat := b - a

theorem green_dots_third_row (a1 a2 a4 a5 a3 d : Nat)
  (h_a1 : a1 = 3)
  (h_a2 : a2 = 6)
  (h_a4 : a4 = 12)
  (h_a5 : a5 = 15)
  (h_d : row_difference a2 a1 = d)
  (h_d_consistent : row_difference a2 a1 = row_difference a4 a3) :
  a3 = 9 :=
sorry

end green_dots_third_row_l13_13746


namespace minimum_value_l13_13127

theorem minimum_value (x : ℝ) (h : x > 1) : 2 * x + 7 / (x - 1) ≥ 2 * Real.sqrt 14 + 2 := by
  sorry

end minimum_value_l13_13127


namespace smallest_palindrome_in_bases_2_and_4_l13_13391

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let repr := n.digits base
  repr = repr.reverse

theorem smallest_palindrome_in_bases_2_and_4 (x : ℕ) :
  (x > 15) ∧ is_palindrome x 2 ∧ is_palindrome x 4 → x = 17 :=
by
  sorry

end smallest_palindrome_in_bases_2_and_4_l13_13391


namespace mod_equiv_pow_five_l13_13113

theorem mod_equiv_pow_five (m : ℤ) (hm : 0 ≤ m ∧ m < 11) (h : 12^5 ≡ m [ZMOD 11]) : m = 1 :=
by
  sorry

end mod_equiv_pow_five_l13_13113


namespace triangle_side_a_value_l13_13027

noncomputable def a_value (A B c : ℝ) : ℝ :=
  30 * Real.sqrt 2 - 10 * Real.sqrt 6

theorem triangle_side_a_value
  (A B : ℝ) (c : ℝ)
  (hA : A = 60)
  (hB : B = 45)
  (hc : c = 20) :
  a_value A B c = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 :=
by
  sorry

end triangle_side_a_value_l13_13027


namespace find_m_n_sum_l13_13630

theorem find_m_n_sum (m n : ℕ) (hm : m > 1) (hn : n > 1) 
  (h : 2005^2 + m^2 = 2004^2 + n^2) : 
  m + n = 211 :=
sorry

end find_m_n_sum_l13_13630


namespace domain_of_function_l13_13553

theorem domain_of_function : {x : ℝ | 3 - 2 * x - x ^ 2 ≥ 0 } = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end domain_of_function_l13_13553


namespace find_shop_width_l13_13931

def shop_width (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_square_foot : ℕ) : ℕ :=
  let annual_rent := monthly_rent * 12
  let total_area := annual_rent / annual_rent_per_square_foot
  total_area / length

theorem find_shop_width :
  shop_width 3600 20 144 = 15 :=
by 
  -- Here would go the proof, but we add sorry to skip it
  sorry

end find_shop_width_l13_13931


namespace find_m_and_p_l13_13712

-- Definition of a point being on the parabola y^2 = 2px
def on_parabola (m : ℝ) (p : ℝ) : Prop :=
  (-3)^2 = 2 * p * m

-- Definition of the distance from the point (m, -3) to the focus being 5
def distance_to_focus (m : ℝ) (p : ℝ) : Prop :=
  m + p / 2 = 5

theorem find_m_and_p (m p : ℝ) (hp : 0 < p) : 
  (on_parabola m p) ∧ (distance_to_focus m p) → 
  (m = 1 / 2 ∧ p = 9) ∨ (m = 9 / 2 ∧ p = 1) :=
by
  sorry

end find_m_and_p_l13_13712


namespace simplify_expression_l13_13523

variable (z : ℝ)

theorem simplify_expression :
  (z - 2 * z + 4 * z - 6 + 3 + 7 - 2) = (3 * z + 2) := by
  sorry

end simplify_expression_l13_13523


namespace ratio_of_area_of_smaller_circle_to_larger_rectangle_l13_13080

noncomputable def ratio_areas (w : ℝ) : ℝ :=
  (3.25 * Real.pi * w^2 / 4) / (1.5 * w^2)

theorem ratio_of_area_of_smaller_circle_to_larger_rectangle (w : ℝ) : 
  ratio_areas w = 13 * Real.pi / 24 := 
by 
  sorry

end ratio_of_area_of_smaller_circle_to_larger_rectangle_l13_13080


namespace line_equation_from_point_normal_l13_13184

theorem line_equation_from_point_normal :
  let M1 : ℝ × ℝ := (7, -8)
  let n : ℝ × ℝ := (-2, 3)
  ∃ C : ℝ, ∀ x y : ℝ, 2 * x - 3 * y + C = 0 ↔ (C = -38) := 
by
  sorry

end line_equation_from_point_normal_l13_13184


namespace sin_double_angle_l13_13374

open Real

theorem sin_double_angle (α : ℝ) (h : tan α = -3/5) : sin (2 * α) = -15/17 :=
by
  -- We are skipping the proof here
  sorry

end sin_double_angle_l13_13374


namespace transform_negation_l13_13316

-- Define the terms a, b, and c as real numbers
variables (a b c : ℝ)

-- State the theorem we want to prove
theorem transform_negation (a b c : ℝ) : 
  - (a - b + c) = -a + b - c :=
sorry

end transform_negation_l13_13316


namespace find_a_given_coefficient_l13_13138

theorem find_a_given_coefficient (a : ℝ) (h : (a^3 * 10 = 80)) : a = 2 :=
by
  sorry

end find_a_given_coefficient_l13_13138


namespace intersection_P_Q_l13_13751

-- Define the sets P and Q
def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {x | -1 ≤ x ∧ x < 1}

-- The proof statement
theorem intersection_P_Q : P ∩ Q = {-1, 0} :=
by
  sorry

end intersection_P_Q_l13_13751


namespace coeff_div_binom_eq_4_l13_13913

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def coeff_x5_expansion : ℚ :=
  binomial 8 2 * (-2) ^ 2

def binomial_coeff : ℚ :=
  binomial 8 2

theorem coeff_div_binom_eq_4 : 
  (coeff_x5_expansion / binomial_coeff) = 4 := by
  sorry

end coeff_div_binom_eq_4_l13_13913


namespace max_a_such_that_f_geq_a_min_value_under_constraint_l13_13486

-- Problem (1)
theorem max_a_such_that_f_geq_a :
  ∃ (a : ℝ), (∀ (x : ℝ), |x - (5/2)| + |x - a| ≥ a) ∧ a = 5 / 4 := sorry

-- Problem (2)
theorem min_value_under_constraint :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2 * y + 3 * z = 1 ∧
  (3 / x + 2 / y + 1 / z) = 16 + 8 * Real.sqrt 3 := sorry

end max_a_such_that_f_geq_a_min_value_under_constraint_l13_13486


namespace range_of_k_condition_l13_13199

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := (4 - k) / x

theorem range_of_k_condition (k x1 x2 y1 y2 : ℝ) 
    (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 < y2) 
    (hA : inverse_proportion_function k x1 = y1) 
    (hB : inverse_proportion_function k x2 = y2) : 
    k < 4 :=
sorry

end range_of_k_condition_l13_13199


namespace min_value_of_a_l13_13086

theorem min_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 + 2*x*y ≤ a*(x^2 + y^2)) → (a ≥ (Real.sqrt 5 + 1) / 2) := 
sorry

end min_value_of_a_l13_13086


namespace beef_weight_after_processing_l13_13280

noncomputable def initial_weight : ℝ := 840
noncomputable def lost_percentage : ℝ := 35
noncomputable def retained_percentage : ℝ := 100 - lost_percentage
noncomputable def final_weight : ℝ := retained_percentage / 100 * initial_weight

theorem beef_weight_after_processing : final_weight = 546 := by
  sorry

end beef_weight_after_processing_l13_13280


namespace solve_for_x_l13_13883

theorem solve_for_x (x : ℝ) (h : (4 + x) / (6 + x) = (1 + x) / (2 + x)) : x = 2 :=
sorry

end solve_for_x_l13_13883


namespace playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l13_13270

def hasWinningStrategyA (n : ℕ) : Prop :=
  n ≥ 8

def hasWinningStrategyB (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5

def draw (n : ℕ) : Prop :=
  n = 6 ∨ n = 7

theorem playerA_winning_strategy (n : ℕ) : n ≥ 8 → hasWinningStrategyA n :=
by
  sorry

theorem playerB_winning_strategy (n : ℕ) : (n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5) → hasWinningStrategyB n :=
by
  sorry

theorem no_winning_strategy (n : ℕ) : n = 6 ∨ n = 7 → draw n :=
by
  sorry

end playerA_winning_strategy_playerB_winning_strategy_no_winning_strategy_l13_13270


namespace roots_negative_reciprocal_condition_l13_13908

theorem roots_negative_reciprocal_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r * s = -1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) → c = -a :=
by
  sorry

end roots_negative_reciprocal_condition_l13_13908


namespace ada_original_seat_l13_13352

theorem ada_original_seat {positions : Fin 6 → Fin 6} 
  (Bea Ceci Dee Edie Fred Ada: Fin 6)
  (h1: Ada = 0)
  (h2: positions (Bea + 1) = Bea)
  (h3: positions (Ceci - 2) = Ceci)
  (h4: positions Dee = Edie ∧ positions Edie = Dee)
  (h5: positions Fred = Fred) :
  Ada = 1 → Bea = 1 → Ceci = 3 → Dee = 4 → Edie = 5 → Fred = 6 → Ada = 1 :=
by
  intros
  sorry

end ada_original_seat_l13_13352


namespace actual_revenue_is_60_percent_of_projected_l13_13531

variable (R : ℝ)

-- Condition: Projected revenue is 25% more than last year's revenue
def projected_revenue (R : ℝ) : ℝ := 1.25 * R

-- Condition: Actual revenue decreased by 25% compared to last year's revenue
def actual_revenue (R : ℝ) : ℝ := 0.75 * R

-- Theorem: Prove that the actual revenue is 60% of the projected revenue
theorem actual_revenue_is_60_percent_of_projected :
  (actual_revenue R) = 0.6 * (projected_revenue R) :=
  sorry

end actual_revenue_is_60_percent_of_projected_l13_13531


namespace quadratic_axis_of_symmetry_is_one_l13_13911

noncomputable def quadratic_axis_of_symmetry (b c : ℝ) : ℝ :=
  (-b / (2 * 1))

theorem quadratic_axis_of_symmetry_is_one
  (b c : ℝ)
  (hA : (0:ℝ)^2 + b * 0 + c = 3)
  (hB : (2:ℝ)^2 + b * 2 + c = 3) :
  quadratic_axis_of_symmetry b c = 1 :=
by
  sorry

end quadratic_axis_of_symmetry_is_one_l13_13911


namespace find_k_l13_13704

theorem find_k 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (hSn : ∀ n, S n = -2 + 2 * (1 / 3) ^ n) 
  (h_geom : ∀ n, a (n + 1) = a n * a 2 / a 1) :
  k = -2 :=
sorry

end find_k_l13_13704


namespace compute_expr_l13_13529

-- Definitions
def a := 150 / 5
def b := 40 / 8
def c := 16 / 32
def d := 3

def expr := 20 * (a - b + c + d)

-- Theorem
theorem compute_expr : expr = 570 :=
by
  sorry

end compute_expr_l13_13529


namespace smallest_positive_integer_square_begins_with_1989_l13_13693

theorem smallest_positive_integer_square_begins_with_1989 :
  ∃ (A : ℕ), (1989 * 10^0 ≤ A^2 ∧ A^2 < 1990 * 10^0) 
  ∨ (1989 * 10^1 ≤ A^2 ∧ A^2 < 1990 * 10^1) 
  ∨ (1989 * 10^2 ≤ A^2 ∧ A^2 < 1990 * 10^2)
  ∧ A = 446 :=
sorry

end smallest_positive_integer_square_begins_with_1989_l13_13693


namespace only_one_tuple_exists_l13_13862

theorem only_one_tuple_exists :
  ∃! (x : Fin 15 → ℝ),
    (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2
    + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7 - x 8)^2 + (x 8 - x 9)^2
    + (x 9 - x 10)^2 + (x 10 - x 11)^2 + (x 11 - x 12)^2 + (x 12 - x 13)^2
    + (x 13 - x 14)^2 + (x 14)^2 = 1 / 16 := by
  sorry

end only_one_tuple_exists_l13_13862


namespace totalCostOfFencing_l13_13565

def numberOfSides : ℕ := 4
def costPerSide : ℕ := 79

theorem totalCostOfFencing (n : ℕ) (c : ℕ) (hn : n = numberOfSides) (hc : c = costPerSide) : n * c = 316 :=
by 
  rw [hn, hc]
  exact rfl

end totalCostOfFencing_l13_13565


namespace largest_angle_of_triangle_ABC_l13_13803

theorem largest_angle_of_triangle_ABC (a b c : ℝ)
  (h₁ : a + b + 2 * c = a^2) 
  (h₂ : a + b - 2 * c = -1) : 
  ∃ C : ℝ, C = 120 :=
sorry

end largest_angle_of_triangle_ABC_l13_13803


namespace Carol_rectangle_length_l13_13364

theorem Carol_rectangle_length :
  (∃ (L : ℕ), (L * 15 = 4 * 30) → L = 8) :=
by
  sorry

end Carol_rectangle_length_l13_13364


namespace jane_earnings_in_two_weeks_l13_13093

-- Define the conditions in the lean environment
def number_of_chickens : ℕ := 10
def eggs_per_chicken_per_week : ℕ := 6
def selling_price_per_dozen : ℕ := 2

-- Statement of the proof problem
theorem jane_earnings_in_two_weeks :
  (number_of_chickens * eggs_per_chicken_per_week * 2) / 12 * selling_price_per_dozen = 20 :=
by
  sorry

end jane_earnings_in_two_weeks_l13_13093


namespace parts_production_equation_l13_13183

theorem parts_production_equation (x : ℝ) : 
  let apr := 50
  let may := 50 * (1 + x)
  let jun := 50 * (1 + x) * (1 + x)
  (apr + may + jun = 182) :=
sorry

end parts_production_equation_l13_13183


namespace divisible_by_13_l13_13188

theorem divisible_by_13 (a : ℤ) (h₀ : 0 ≤ a) (h₁ : a ≤ 13) : (51^2015 + a) % 13 = 0 → a = 1 :=
by
  sorry

end divisible_by_13_l13_13188


namespace unique_function_l13_13572

-- Define the function in the Lean environment
def f (n : ℕ) : ℕ := n

-- State the theorem with the given conditions and expected answer
theorem unique_function (f : ℕ → ℕ) : 
  (∀ x y : ℕ, 0 < x → 0 < y → f x + y * f (f x) < x * (1 + f y) + 2021) → (∀ x : ℕ, f x = x) :=
by
  intros h x
  -- Placeholder for the proof
  sorry

end unique_function_l13_13572


namespace problem1_problem2_l13_13449

/-- Problem 1: Prove the solution to the system of equations is x = 1/2 and y = 5 -/
theorem problem1 (x y : ℚ) (h1 : 2 * x - y = -4) (h2 : 4 * x - 5 * y = -23) : 
  x = 1 / 2 ∧ y = 5 := 
sorry

/-- Problem 2: Prove the value of the expression (x-3y)^{2} - (2x+y)(y-2x) when x = 2 and y = -1 is 40 -/
theorem problem2 (x y : ℚ) (h1 : x = 2) (h2 : y = -1) : 
  (x - 3 * y) ^ 2 - (2 * x + y) * (y - 2 * x) = 40 := 
sorry

end problem1_problem2_l13_13449


namespace original_price_of_candy_box_is_8_l13_13504

-- Define the given conditions
def candy_box_price_after_increase : ℝ := 10
def candy_box_increase_rate : ℝ := 1.25

-- Define the original price of the candy box
noncomputable def original_candy_box_price : ℝ := candy_box_price_after_increase / candy_box_increase_rate

-- The theorem to prove
theorem original_price_of_candy_box_is_8 :
  original_candy_box_price = 8 := by
  sorry

end original_price_of_candy_box_is_8_l13_13504


namespace sum_of_ten_numbers_in_circle_l13_13906

theorem sum_of_ten_numbers_in_circle : 
  ∀ (a b c d e f g h i j : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i ∧ 0 < j ∧
  a = Nat.gcd b j + 1 ∧ b = Nat.gcd a c + 1 ∧ c = Nat.gcd b d + 1 ∧ d = Nat.gcd c e + 1 ∧ 
  e = Nat.gcd d f + 1 ∧ f = Nat.gcd e g + 1 ∧ g = Nat.gcd f h + 1 ∧ 
  h = Nat.gcd g i + 1 ∧ i = Nat.gcd h j + 1 ∧ j = Nat.gcd i a + 1 → 
  a + b + c + d + e + f + g + h + i + j = 28 :=
by
  intros
  sorry

end sum_of_ten_numbers_in_circle_l13_13906


namespace find_value_of_x_cubed_plus_y_cubed_l13_13966

-- Definitions based on the conditions provided
variables (x y : ℝ)
variables (h1 : y + 3 = (x - 3)^2) (h2 : x + 3 = (y - 3)^2) (h3 : x ≠ y)

theorem find_value_of_x_cubed_plus_y_cubed :
  x^3 + y^3 = 217 :=
sorry

end find_value_of_x_cubed_plus_y_cubed_l13_13966


namespace solve_for_A_l13_13544

theorem solve_for_A (A : ℕ) (h1 : 3 + 68 * A = 691) (h2 : 68 * A < 1000) (h3 : 68 * A ≥ 100) : A = 8 :=
by
  sorry

end solve_for_A_l13_13544


namespace expand_polynomial_l13_13927

noncomputable def polynomial_expansion : Prop :=
  ∀ (x : ℤ), (x + 3) * (x^2 + 4 * x + 6) = x^3 + 7 * x^2 + 18 * x + 18

theorem expand_polynomial : polynomial_expansion :=
by
  sorry

end expand_polynomial_l13_13927


namespace prime_number_identity_l13_13538

theorem prime_number_identity (p m : ℕ) (h1 : Nat.Prime p) (h2 : m > 0) (h3 : 2 * p^2 + p + 9 = m^2) :
  p = 5 ∧ m = 8 :=
sorry

end prime_number_identity_l13_13538


namespace solve_eq_solution_l13_13234

def eq_solution (x y : ℕ) : Prop := 3 ^ x = 2 ^ x * y + 1

theorem solve_eq_solution (x y : ℕ) (h1 : x > 0) (h2 : y > 0) : 
  eq_solution x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) :=
sorry

end solve_eq_solution_l13_13234


namespace profit_in_2004_correct_l13_13589

-- We define the conditions as given in the problem
def annual_profit_2002 : ℝ := 10
def annual_growth_rate (p : ℝ) : ℝ := p

-- The expression for the annual profit in 2004 given the above conditions
def annual_profit_2004 (p : ℝ) : ℝ := annual_profit_2002 * (1 + p) * (1 + p)

-- The theorem to prove that the computed annual profit in 2004 matches the expected answer
theorem profit_in_2004_correct (p : ℝ) :
  annual_profit_2004 p = 10 * (1 + p)^2 := 
by 
  sorry

end profit_in_2004_correct_l13_13589


namespace geometric_sequence_fourth_term_l13_13162

theorem geometric_sequence_fourth_term (a₁ a₂ a₃ : ℝ) (r : ℝ)
    (h₁ : a₁ = 5^(3/4))
    (h₂ : a₂ = 5^(1/2))
    (h₃ : a₃ = 5^(1/4))
    (geometric_seq : a₂ = a₁ * r ∧ a₃ = a₂ * r) :
    a₃ * r = 1 := 
by
  sorry

end geometric_sequence_fourth_term_l13_13162


namespace nonnegative_integer_solutions_l13_13168

theorem nonnegative_integer_solutions (x : ℕ) :
  2 * x - 1 < 5 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
sorry

end nonnegative_integer_solutions_l13_13168


namespace value_of_fraction_l13_13071

theorem value_of_fraction (x y z w : ℕ) (h₁ : x = 4 * y) (h₂ : y = 3 * z) (h₃ : z = 5 * w) :
  x * z / (y * w) = 20 := by
  sorry

end value_of_fraction_l13_13071


namespace trigonometric_identity_l13_13595

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
sorry

end trigonometric_identity_l13_13595


namespace polynomial_horner_form_operations_l13_13229

noncomputable def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
  coeffs.foldr (fun a acc => a + acc * x) 0

theorem polynomial_horner_form_operations :
  let p := [1, 1, 2, 3, 4, 5]
  let x := 2
  horner_eval p x = ((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 ∧
  (∀ x, x = 2 → (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1 =  5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + 1 * x + 1)) ∧ 
  (∃ m a, m = 5 ∧ a = 5) := sorry

end polynomial_horner_form_operations_l13_13229


namespace leila_total_cakes_l13_13761

def cakes_monday : ℕ := 6
def cakes_friday : ℕ := 9
def cakes_saturday : ℕ := 3 * cakes_monday
def total_cakes : ℕ := cakes_monday + cakes_friday + cakes_saturday

theorem leila_total_cakes : total_cakes = 33 :=
by sorry

end leila_total_cakes_l13_13761


namespace solution_set_quadratic_ineq_all_real_l13_13755

theorem solution_set_quadratic_ineq_all_real (a b c : ℝ) :
  (∀ x : ℝ, (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) :=
by
  sorry

end solution_set_quadratic_ineq_all_real_l13_13755


namespace fish_count_l13_13483

theorem fish_count (initial_fish : ℝ) (bought_fish : ℝ) (total_fish : ℝ) 
  (h1 : initial_fish = 212.0) 
  (h2 : bought_fish = 280.0) 
  (h3 : total_fish = initial_fish + bought_fish) : 
  total_fish = 492.0 := 
by 
  sorry

end fish_count_l13_13483


namespace hall_volume_l13_13466

theorem hall_volume (l w : ℕ) (h : ℕ) 
    (cond1 : l = 18)
    (cond2 : w = 9)
    (cond3 : (2 * l * w) = (2 * l * h + 2 * w * h)) : 
    (l * w * h = 972) :=
by
  rw [cond1, cond2] at cond3
  have h_eq : h = 324 / 54 := sorry
  rw [h_eq]
  norm_num
  sorry

end hall_volume_l13_13466


namespace f_of_6_l13_13247

noncomputable def f (u : ℝ) : ℝ := 
  let x := (u + 2) / 4
  x^3 - x + 2

theorem f_of_6 : f 6 = 8 :=
by
  sorry

end f_of_6_l13_13247


namespace fraction_equals_decimal_l13_13488

theorem fraction_equals_decimal : (3 : ℝ) / 2 = 1.5 := 
sorry

end fraction_equals_decimal_l13_13488


namespace find_a_l13_13473

noncomputable def tangent_line (a : ℝ) (x : ℝ) := (3 * a * (1:ℝ)^2 + 1) * (x - 1) + (a * (1:ℝ)^3 + (1:ℝ) + 1)

theorem find_a : ∃ a : ℝ, tangent_line a 2 = 7 := 
sorry

end find_a_l13_13473


namespace expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l13_13916

noncomputable def A (x y : ℝ) := x^2 - 3 * x * y - y^2
noncomputable def B (x y : ℝ) := x^2 - 3 * x * y - 3 * y^2
noncomputable def M (x y : ℝ) := 2 * A x y - B x y

theorem expression_for_M (x y : ℝ) : M x y = x^2 - 3 * x * y + y^2 := by
  sorry

theorem value_of_M_when_x_eq_negative_2_and_y_eq_1 :
  M (-2) 1 = 11 := by
  sorry

end expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l13_13916


namespace probability_of_two_black_balls_is_one_fifth_l13_13118

noncomputable def probability_of_two_black_balls (W B : Nat) : ℚ :=
  let total_balls := W + B
  let prob_black1 := (B : ℚ) / total_balls
  let prob_black2_given_black1 := (B - 1 : ℚ) / (total_balls - 1)
  prob_black1 * prob_black2_given_black1

theorem probability_of_two_black_balls_is_one_fifth : 
  probability_of_two_black_balls 8 7 = 1 / 5 := 
by
  sorry

end probability_of_two_black_balls_is_one_fifth_l13_13118


namespace change_percentage_difference_l13_13370

theorem change_percentage_difference 
  (initial_yes : ℚ) (initial_no : ℚ) (initial_undecided : ℚ)
  (final_yes : ℚ) (final_no : ℚ) (final_undecided : ℚ)
  (h_initial : initial_yes = 0.4 ∧ initial_no = 0.3 ∧ initial_undecided = 0.3)
  (h_final : final_yes = 0.6 ∧ final_no = 0.1 ∧ final_undecided = 0.3) :
  (final_yes - initial_yes + initial_no - final_no) = 0.2 := by
sorry

end change_percentage_difference_l13_13370


namespace min_value_cos_sin_l13_13125

noncomputable def min_value_expression : ℝ :=
  -1 / 2

theorem min_value_cos_sin (θ : ℝ) (hθ1 : 0 ≤ θ) (hθ2 : θ ≤ 3 * Real.pi / 2) :
  ∃ (y : ℝ), y = Real.cos (θ / 3) * (1 - Real.sin θ) ∧ y = min_value_expression :=
sorry

end min_value_cos_sin_l13_13125


namespace smallest_number_plus_3_divisible_by_18_70_100_21_l13_13397

/-- 
The smallest number such that when increased by 3 is divisible by 18, 70, 100, and 21.
-/
theorem smallest_number_plus_3_divisible_by_18_70_100_21 : 
  ∃ n : ℕ, (∃ k : ℕ, n + 3 = k * 18) ∧ (∃ l : ℕ, n + 3 = l * 70) ∧ (∃ m : ℕ, n + 3 = m * 100) ∧ (∃ o : ℕ, n + 3 = o * 21) ∧ n = 6297 :=
sorry

end smallest_number_plus_3_divisible_by_18_70_100_21_l13_13397


namespace pizza_cost_l13_13386

theorem pizza_cost (soda_cost jeans_cost start_money quarters_left : ℝ) (quarters_value : ℝ) (total_left : ℝ) (pizza_cost : ℝ) :
  soda_cost = 1.50 → 
  jeans_cost = 11.50 → 
  start_money = 40 → 
  quarters_left = 97 → 
  quarters_value = 0.25 → 
  total_left = quarters_left * quarters_value → 
  pizza_cost = start_money - total_left - (soda_cost + jeans_cost) → 
  pizza_cost = 2.75 :=
by
  sorry

end pizza_cost_l13_13386


namespace dot_product_EC_ED_l13_13312

open Real

-- Assume we are in the plane and define points A, B, C, D and E
def squareSide : ℝ := 2

noncomputable def A : ℝ × ℝ := (0, 0)
noncomputable def B : ℝ × ℝ := (squareSide, 0)
noncomputable def D : ℝ × ℝ := (0, squareSide)
noncomputable def C : ℝ × ℝ := (squareSide, squareSide)
noncomputable def E : ℝ × ℝ := (squareSide / 2, 0) -- Midpoint of AB

-- Defining vectors EC and ED
noncomputable def vectorEC : ℝ × ℝ := (C.1 - E.1, C.2 - E.2)
noncomputable def vectorED : ℝ × ℝ := (D.1 - E.1, D.2 - E.2)

-- Goal: prove the dot product of vectorEC and vectorED is 3
theorem dot_product_EC_ED : vectorEC.1 * vectorED.1 + vectorEC.2 * vectorED.2 = 3 := by
  sorry

end dot_product_EC_ED_l13_13312


namespace mango_price_reduction_l13_13720

theorem mango_price_reduction (P R : ℝ) (M : ℕ)
  (hP_orig : 110 * P = 366.67)
  (hM : M * P = 360)
  (hR_red : (M + 12) * R = 360) :
  ((P - R) / P) * 100 = 10 :=
by sorry

end mango_price_reduction_l13_13720


namespace company_x_total_employees_l13_13149

-- Definitions for conditions
def initial_percentage : ℝ := 0.60
def Q2_hiring_males : ℕ := 30
def Q2_new_percentage : ℝ := 0.57
def Q3_hiring_females : ℕ := 50
def Q3_new_percentage : ℝ := 0.62
def Q4_hiring_males : ℕ := 40
def Q4_hiring_females : ℕ := 10
def Q4_new_percentage : ℝ := 0.58

-- Statement of the proof problem
theorem company_x_total_employees :
  ∃ (E : ℕ) (F : ℕ), 
    (F = initial_percentage * E ∧
     F = Q2_new_percentage * (E + Q2_hiring_males) ∧
     F + Q3_hiring_females = Q3_new_percentage * (E + Q2_hiring_males + Q3_hiring_females) ∧
     F + Q3_hiring_females + Q4_hiring_females = Q4_new_percentage * (E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females)) →
    E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females = 700 :=
sorry

end company_x_total_employees_l13_13149


namespace range_x_inequality_l13_13748

theorem range_x_inequality (a b x : ℝ) (ha : a ≠ 0) :
  (x ≥ 1/2) ∧ (x ≤ 5/2) →
  |a + b| + |a - b| ≥ |a| * (|x - 1| + |x - 2|) :=
by
  sorry

end range_x_inequality_l13_13748


namespace problem1_l13_13934

/-- Problem 1: Given the formula \( S = vt + \frac{1}{2}at^2 \) and the conditions
  when \( t=1, S=4 \) and \( t=2, S=10 \), prove that when \( t=3 \), \( S=18 \). -/
theorem problem1 (v a t S: ℝ) 
  (h₁ : t = 1 → S = 4 → S = v * t + 1 / 2 * a * t^2)
  (h₂ : t = 2 → S = 10 → S = v * t + 1 / 2 * a * t^2):
  t = 3 → S = v * t + 1 / 2 * a * t^2 → S = 18 := by
  sorry

end problem1_l13_13934


namespace axis_of_symmetry_of_quadratic_l13_13624

theorem axis_of_symmetry_of_quadratic (m : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * m * x - m^2 + 3 = -x^2 + 2 * m * x - m^2 + 3) ∧ (∃ x : ℝ, x + 2 = 0) → m = -2 :=
by
  sorry

end axis_of_symmetry_of_quadratic_l13_13624


namespace diff_of_squares_l13_13985

theorem diff_of_squares : (1001^2 - 999^2 = 4000) :=
by
  sorry

end diff_of_squares_l13_13985


namespace acute_triangle_area_relation_l13_13020

open Real

variables (A B C R : ℝ)
variables (acute_triangle : Prop)
variables (S p_star : ℝ)

-- Conditions
axiom acute_triangle_condition : acute_triangle
axiom area_formula : S = (R^2 / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))
axiom semiperimeter_formula : p_star = (R / 2) * (sin (2 * A) + sin (2 * B) + sin (2 * C))

-- Theorem to prove
theorem acute_triangle_area_relation (h : acute_triangle) : S = p_star * R := 
by {
  sorry 
}

end acute_triangle_area_relation_l13_13020


namespace length_of_platform_l13_13921

def len_train : ℕ := 300 -- length of the train in meters
def time_platform : ℕ := 39 -- time to cross the platform in seconds
def time_pole : ℕ := 26 -- time to cross the signal pole in seconds

theorem length_of_platform (L : ℕ) (h1 : len_train / time_pole = (len_train + L) / time_platform) : L = 150 :=
  sorry

end length_of_platform_l13_13921


namespace purely_imaginary_roots_iff_l13_13074

theorem purely_imaginary_roots_iff (z : ℂ) (k : ℝ) (i : ℂ) (h_i2 : i^2 = -1) :
  (∀ r : ℂ, (20 * r^2 + 6 * i * r - ↑k = 0) → (∃ b : ℝ, r = b * i)) ↔ (k = 9 / 5) :=
sorry

end purely_imaginary_roots_iff_l13_13074


namespace evaluate_expression_l13_13172

theorem evaluate_expression : (20 * 3 + 10) / (5 + 3) = 9 := by
  sorry

end evaluate_expression_l13_13172


namespace isosceles_vertex_angle_l13_13583

-- Let T be a type representing triangles, with a function base_angle returning the degree of a base angle,
-- and vertex_angle representing the degree of the vertex angle.
axiom Triangle : Type
axiom is_isosceles (t : Triangle) : Prop
axiom base_angle_deg (t : Triangle) : ℝ
axiom vertex_angle_deg (t : Triangle) : ℝ

theorem isosceles_vertex_angle (t : Triangle) (h_isosceles : is_isosceles t)
  (h_base_angle : base_angle_deg t = 50) : vertex_angle_deg t = 80 := by
  sorry

end isosceles_vertex_angle_l13_13583


namespace jake_sister_weight_ratio_l13_13422

theorem jake_sister_weight_ratio (Jake_initial_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (sister_weight : ℕ) 
(h₁ : Jake_initial_weight = 156) 
(h₂ : total_weight = 224) 
(h₃ : weight_loss = 20) 
(h₄ : total_weight = Jake_initial_weight + sister_weight) :
(Jake_initial_weight - weight_loss) / sister_weight = 2 := by
  sorry

end jake_sister_weight_ratio_l13_13422


namespace parameterization_of_line_l13_13031

theorem parameterization_of_line : 
  ∀ t : ℝ, ∃ f : ℝ → ℝ, (f t, 20 * t - 14) ∈ { p : ℝ × ℝ | ∃ (x y : ℝ), y = 2 * x - 40 ∧ p = (x, y) } ∧ f t = 10 * t + 13 :=
by
  sorry

end parameterization_of_line_l13_13031


namespace arman_hourly_rate_increase_l13_13290

theorem arman_hourly_rate_increase :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let total_payment := 770
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := total_payment - last_week_earnings
  let this_week_rate := this_week_earnings / this_week_hours
  let rate_increase := this_week_rate - last_week_rate
  rate_increase = 0.50 :=
by {
  sorry
}

end arman_hourly_rate_increase_l13_13290


namespace part1_part2_l13_13401

noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.sin x - 1/2 * Real.cos (2 * x) + a - 3/a + 1/2

theorem part1 (a : ℝ) (h₀ : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 0 1 := sorry

theorem part2 (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≥ 2) :
  (∃ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 2 3 := sorry

end part1_part2_l13_13401


namespace vector_properties_l13_13819

/-- The vectors a, b, and c used in the problem. --/
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-4, 2)
def c : ℝ × ℝ := (1, 2)

theorem vector_properties :
  ((∃ k : ℝ, b = k • a) ∧ (b.1 * c.1 + b.2 * c.2 = 0) ∧ (a.1*a.1 + a.2*a.2 = c.1*c.1 + c.2*c.2)) :=
  by sorry

end vector_properties_l13_13819


namespace rational_root_of_factors_l13_13686

theorem rational_root_of_factors (p : ℕ) (a : ℚ) (hprime : Nat.Prime p) 
  (f : Polynomial ℚ) (hf : f = Polynomial.X ^ p - Polynomial.C a)
  (hfactors : ∃ g h : Polynomial ℚ, f = g * h ∧ 1 ≤ g.degree ∧ 1 ≤ h.degree) : 
  ∃ r : ℚ, Polynomial.eval r f = 0 :=
sorry

end rational_root_of_factors_l13_13686


namespace simplify_fraction_subtraction_l13_13982

theorem simplify_fraction_subtraction :
  (5 / 15 : ℚ) - (2 / 45) = 13 / 45 :=
by
  -- (The proof will go here)
  sorry

end simplify_fraction_subtraction_l13_13982


namespace total_students_l13_13625

theorem total_students (a b c d e f : ℕ)  (h : a + b = 15) (h1 : a = 5) (h2 : b = 10) 
(h3 : c = 15) (h4 : d = 10) (h5 : e = 5) (h6 : f = 0) (h_total : a + b + c + d + e + f = 50) : a + b + c + d + e + f = 50 :=
by {exact h_total}

end total_students_l13_13625


namespace sum_of_reciprocals_l13_13684

variable (x y : ℝ)

theorem sum_of_reciprocals (h1 : x + y = 10) (h2 : x * y = 20) : 1 / x + 1 / y = 1 / 2 :=
by
  sorry

end sum_of_reciprocals_l13_13684


namespace power_equation_l13_13579

theorem power_equation (x a : ℝ) (h : x^(-a) = 3) : x^(2 * a) = 1 / 9 :=
sorry

end power_equation_l13_13579


namespace pentagon_rectangle_ratio_l13_13881

theorem pentagon_rectangle_ratio (p w l : ℝ) (h₁ : 5 * p = 20) (h₂ : l = 2 * w) (h₃ : 2 * l + 2 * w = 20) : p / w = 6 / 5 :=
by
  sorry

end pentagon_rectangle_ratio_l13_13881


namespace number_of_real_roots_l13_13436

theorem number_of_real_roots :
  ∃ (roots_count : ℕ), roots_count = 2 ∧
  (∀ x : ℝ, x^2 - |2 * x - 1| - 4 = 0 → (x = -1 - Real.sqrt 6 ∨ x = 3)) :=
sorry

end number_of_real_roots_l13_13436


namespace five_points_distance_ratio_ge_two_sin_54_l13_13433

theorem five_points_distance_ratio_ge_two_sin_54
  (points : Fin 5 → ℝ × ℝ)
  (distinct : Function.Injective points) :
  let distances := {d : ℝ | ∃ (i j : Fin 5), i ≠ j ∧ d = dist (points i) (points j)}
  ∃ (max_dist min_dist : ℝ), max_dist ∈ distances ∧ min_dist ∈ distances ∧ max_dist / min_dist ≥ 2 * Real.sin (54 * Real.pi / 180) := by
  sorry

end five_points_distance_ratio_ge_two_sin_54_l13_13433


namespace average_weight_increase_l13_13217

variable {W : ℝ} -- Total weight before replacement
variable {n : ℝ} -- Number of men in the group

theorem average_weight_increase
  (h1 : (W - 58 + 83) / n - W / n = 2.5) : n = 10 :=
by
  sorry

end average_weight_increase_l13_13217


namespace athlete_speed_200m_in_24s_is_30kmh_l13_13853

noncomputable def speed_in_kmh (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_meters / 1000) / (time_seconds / 3600)

theorem athlete_speed_200m_in_24s_is_30kmh :
  speed_in_kmh 200 24 = 30 := by
  sorry

end athlete_speed_200m_in_24s_is_30kmh_l13_13853


namespace largest_sum_of_digits_in_display_l13_13917

-- Define the conditions
def is_valid_hour (h : Nat) : Prop := 0 <= h ∧ h < 24
def is_valid_minute (m : Nat) : Prop := 0 <= m ∧ m < 60

-- Define helper functions to convert numbers to their digit sums
def digit_sum (n : Nat) : Nat :=
  n.digits 10 |>.sum

-- Define the largest possible sum of the digits condition
def largest_possible_digit_sum : Prop :=
  ∀ (h m : Nat), is_valid_hour h → is_valid_minute m → 
    digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) ≤ 24 ∧
    ∃ (h m : Nat), is_valid_hour h ∧ is_valid_minute m ∧ digit_sum (h.div 10 + h % 10) + digit_sum (m.div 10 + m % 10) = 24

-- The statement to prove
theorem largest_sum_of_digits_in_display : largest_possible_digit_sum :=
by
  sorry

end largest_sum_of_digits_in_display_l13_13917


namespace jessica_current_age_l13_13932

-- Definitions and conditions from the problem
def J (M : ℕ) : ℕ := M / 2
def M : ℕ := 60

-- Lean statement for the proof problem
theorem jessica_current_age : J M + 10 = 40 :=
by
  sorry

end jessica_current_age_l13_13932


namespace heather_payment_per_weed_l13_13120

noncomputable def seconds_in_hour : ℕ := 60 * 60

noncomputable def weeds_per_hour (seconds_per_weed : ℕ) : ℕ :=
  seconds_in_hour / seconds_per_weed

noncomputable def payment_per_weed (hourly_pay : ℕ) (weeds_per_hour : ℕ) : ℚ :=
  hourly_pay / weeds_per_hour

theorem heather_payment_per_weed (seconds_per_weed : ℕ) (hourly_pay : ℕ) :
  seconds_per_weed = 18 ∧ hourly_pay = 10 → payment_per_weed hourly_pay (weeds_per_hour seconds_per_weed) = 0.05 :=
by
  sorry

end heather_payment_per_weed_l13_13120


namespace woman_lawyer_probability_l13_13836

noncomputable def probability_of_woman_lawyer : ℚ :=
  let total_members : ℚ := 100
  let women_percentage : ℚ := 0.80
  let lawyer_percentage_women : ℚ := 0.40
  let women_members := women_percentage * total_members
  let women_lawyers := lawyer_percentage_women * women_members
  let probability := women_lawyers / total_members
  probability

theorem woman_lawyer_probability :
  probability_of_woman_lawyer = 0.32 := by
  sorry

end woman_lawyer_probability_l13_13836


namespace hypotenuse_of_isosceles_right_triangle_l13_13826

theorem hypotenuse_of_isosceles_right_triangle (a : ℝ) (hyp : a = 8) : 
  ∃ c : ℝ, c = a * Real.sqrt 2 :=
by
  use 8 * Real.sqrt 2
  sorry

end hypotenuse_of_isosceles_right_triangle_l13_13826


namespace total_reading_materials_l13_13897

theorem total_reading_materials (magazines newspapers : ℕ) (h1 : magazines = 425) (h2 : newspapers = 275) : 
  magazines + newspapers = 700 :=
by 
  sorry

end total_reading_materials_l13_13897


namespace females_watch_eq_seventy_five_l13_13385

-- Definition of conditions
def males_watch : ℕ := 85
def females_dont_watch : ℕ := 120
def total_watch : ℕ := 160
def total_dont_watch : ℕ := 180

-- Definition of the proof problem
theorem females_watch_eq_seventy_five :
  total_watch - males_watch = 75 :=
by
  sorry

end females_watch_eq_seventy_five_l13_13385


namespace smaller_rectangle_area_l13_13115

-- Define the lengths and widths of the rectangles
def bigRectangleLength : ℕ := 40
def bigRectangleWidth : ℕ := 20
def smallRectangleLength : ℕ := bigRectangleLength / 2
def smallRectangleWidth : ℕ := bigRectangleWidth / 2

-- Define the area of the rectangles
def area (length width : ℕ) : ℕ := length * width

-- Prove the area of the smaller rectangle
theorem smaller_rectangle_area : area smallRectangleLength smallRectangleWidth = 200 :=
by
  -- Skip the proof
  sorry

end smaller_rectangle_area_l13_13115


namespace math_problem_l13_13446

theorem math_problem
  (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : x^2 + y^2 = 18) :
  x^2 + y^2 = 18 :=
sorry

end math_problem_l13_13446


namespace expression_evaluation_l13_13964

theorem expression_evaluation : |1 - Real.sqrt 3| + 2 * Real.cos (Real.pi / 6) - Real.sqrt 12 - 2023 = -2024 := 
by {
    sorry
}

end expression_evaluation_l13_13964


namespace ashley_percentage_secured_l13_13994

noncomputable def marks_secured : ℕ := 332
noncomputable def max_marks : ℕ := 400
noncomputable def percentage_secured : ℕ := (marks_secured * 100) / max_marks

theorem ashley_percentage_secured 
    (h₁ : marks_secured = 332)
    (h₂ : max_marks = 400) :
    percentage_secured = 83 := by
  -- Proof goes here
  sorry

end ashley_percentage_secured_l13_13994


namespace find_smaller_number_l13_13889

theorem find_smaller_number (n m : ℕ) (h1 : n - m = 58)
  (h2 : n^2 % 100 = m^2 % 100) : m = 21 :=
by
  sorry

end find_smaller_number_l13_13889


namespace determinant_of_tan_matrix_l13_13549

theorem determinant_of_tan_matrix
  (A B C : ℝ)
  (h₁ : A = π / 4)
  (h₂ : A + B + C = π)
  : (Matrix.det ![
      ![Real.tan A, 1, 1],
      ![1, Real.tan B, 1],
      ![1, 1, Real.tan C]
    ]) = 2 :=
  sorry

end determinant_of_tan_matrix_l13_13549


namespace jacob_age_in_X_years_l13_13939

-- Definitions of the conditions
variable (J M X : ℕ)

theorem jacob_age_in_X_years
  (h1 : J = M - 14)
  (h2 : M + 9 = 2 * (J + 9))
  (h3 : J = 5) :
  J + X = 5 + X :=
by
  sorry

end jacob_age_in_X_years_l13_13939


namespace equilateral_triangle_perimeter_l13_13132

theorem equilateral_triangle_perimeter (s : ℝ) 
  (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  -- Proof steps (omitted)
  sorry

end equilateral_triangle_perimeter_l13_13132


namespace regular_polygon_sides_l13_13209

theorem regular_polygon_sides (ex_angle : ℝ) (hne_zero : ex_angle ≠ 0)
  (sum_ext_angles : ∀ (n : ℕ), n > 2 → n * ex_angle = 360) :
  ∃ (n : ℕ), n * 15 = 360 ∧ n = 24 :=
by 
  sorry

end regular_polygon_sides_l13_13209


namespace remainder_4032_125_l13_13174

theorem remainder_4032_125 : 4032 % 125 = 32 := by
  sorry

end remainder_4032_125_l13_13174


namespace jerry_age_l13_13054

theorem jerry_age (M J : ℕ) (h1 : M = 2 * J - 2) (h2 : M = 18) : J = 10 := by
  sorry

end jerry_age_l13_13054


namespace smallest_integer_is_840_l13_13123

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def all_divide (N : ℕ) : Prop :=
  (2 ∣ N) ∧ (3 ∣ N) ∧ (5 ∣ N) ∧ (7 ∣ N)

def no_prime_digit (N : ℕ) : Prop :=
  ∀ d ∈ N.digits 10, ¬ is_prime_digit d

def smallest_satisfying_N (N : ℕ) : Prop :=
  no_prime_digit N ∧ all_divide N ∧ ∀ M, no_prime_digit M → all_divide M → N ≤ M

theorem smallest_integer_is_840 : smallest_satisfying_N 840 :=
by
  sorry

end smallest_integer_is_840_l13_13123


namespace min_sum_of_factors_l13_13447

theorem min_sum_of_factors (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 3960) : 
  a + b + c = 72 :=
sorry

end min_sum_of_factors_l13_13447


namespace min_value_x_plus_y_l13_13907

theorem min_value_x_plus_y (x y : ℤ) (det : 3 < x * y ∧ x * y < 5) : x + y = -5 :=
sorry

end min_value_x_plus_y_l13_13907


namespace cube_of_720_diamond_1001_l13_13413

-- Define the operation \diamond
def diamond (a b : ℕ) : ℕ :=
  (Nat.factors (a * b)).toFinset.card

-- Define the specific numbers 720 and 1001
def n1 : ℕ := 720
def n2 : ℕ := 1001

-- Calculate the cubic of the result of diamond operation
def cube_of_diamond : ℕ := (diamond n1 n2) ^ 3

-- The statement to be proved
theorem cube_of_720_diamond_1001 : cube_of_diamond = 216 :=
by {
  sorry
}

end cube_of_720_diamond_1001_l13_13413


namespace polynomial_divisible_l13_13298

theorem polynomial_divisible (a b c : ℕ) :
  (X^(3 * a) + X^(3 * b + 1) + X^(3 * c + 2)) % (X^2 + X + 1) = 0 :=
by sorry

end polynomial_divisible_l13_13298


namespace min_value_of_expression_l13_13701

variable (a b : ℝ)

theorem min_value_of_expression (h : b ≠ 0) : 
  ∃ (a b : ℝ), (a^2 + b^2 + a / b + 1 / b^2) = Real.sqrt 3 :=
sorry

end min_value_of_expression_l13_13701


namespace lcm_5_711_is_3555_l13_13989

theorem lcm_5_711_is_3555 : Nat.lcm 5 711 = 3555 := by
  sorry

end lcm_5_711_is_3555_l13_13989


namespace solve_for_x_l13_13109

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 = -2 * x + 11) : x = 3 := 
sorry

end solve_for_x_l13_13109


namespace handshake_count_l13_13107

theorem handshake_count :
  let total_people := 5 * 4
  let handshakes_per_person := total_people - 1 - 3
  let total_handshakes_with_double_count := total_people * handshakes_per_person
  let total_handshakes := total_handshakes_with_double_count / 2
  total_handshakes = 160 :=
by
-- We include "sorry" to indicate that the proof is not provided.
sorry

end handshake_count_l13_13107


namespace interior_triangle_area_l13_13962

theorem interior_triangle_area (a b c : ℝ)
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (hpythagorean : a^2 + b^2 = c^2) :
  1/2 * a * b = 24 :=
by
  sorry

end interior_triangle_area_l13_13962


namespace find_y_l13_13246

theorem find_y (x : ℤ) (y : ℤ) (h : x = 5) (h1 : 3 * x = (y - x) + 4) : y = 16 :=
by
  sorry

end find_y_l13_13246


namespace octagon_diagonals_l13_13891

def total_lines (n : ℕ) : ℕ := n * (n - 1) / 2

theorem octagon_diagonals : total_lines 8 - 8 = 20 := 
by
  -- Calculate the total number of lines between any two points in an octagon
  have h1 : total_lines 8 = 28 := by sorry
  -- Subtract the number of sides of the octagon
  have h2 : 28 - 8 = 20 := by norm_num
  
  -- Combine results to conclude the theorem
  exact h2

end octagon_diagonals_l13_13891


namespace find_f_2011_l13_13197

open Function

variable {R : Type} [Field R]

def functional_equation (f : R → R) : Prop :=
  ∀ a b : R, f (a * f b) = a * b

theorem find_f_2011 (f : ℝ → ℝ) (h : functional_equation f) : f 2011 = 2011 :=
sorry

end find_f_2011_l13_13197


namespace remainder_x2023_plus_1_l13_13434

noncomputable def remainder (a b : Polynomial ℂ) : Polynomial ℂ :=
a % b

theorem remainder_x2023_plus_1 :
  remainder (Polynomial.X ^ 2023 + 1) (Polynomial.X ^ 8 - Polynomial.X ^ 6 + Polynomial.X ^ 4 - Polynomial.X ^ 2 + 1) =
  - Polynomial.X ^ 3 + 1 :=
by
  sorry

end remainder_x2023_plus_1_l13_13434


namespace line_properties_l13_13443

theorem line_properties : ∃ m x_intercept, 
  (∀ (x y : ℝ), 4 * x + 7 * y = 28 → y = m * x + 4) ∧ 
  (∀ (x y : ℝ), y = 0 → 4 * x + 7 * y = 28 → x = x_intercept) ∧ 
  m = -4 / 7 ∧ 
  x_intercept = 7 :=
by 
  sorry

end line_properties_l13_13443


namespace percent_of_men_tenured_l13_13998

theorem percent_of_men_tenured (total_professors : ℕ) (women_percent tenured_percent women_tenured_or_both_percent men_percent tenured_men_percent : ℝ)
  (h1 : women_percent = 70 / 100)
  (h2 : tenured_percent = 70 / 100)
  (h3 : women_tenured_or_both_percent = 90 / 100)
  (h4 : men_percent = 30 / 100)
  (h5 : total_professors > 0)
  (h6 : tenured_men_percent = (2/3)) :
  tenured_men_percent * 100 = 66.67 :=
by sorry

end percent_of_men_tenured_l13_13998


namespace angle_sum_around_point_l13_13342

theorem angle_sum_around_point (p q r s t : ℝ) (h : p + q + r + s + t = 360) : p = 360 - q - r - s - t :=
by
  sorry

end angle_sum_around_point_l13_13342


namespace dot_product_of_ab_ac_l13_13568

def vec_dot (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem dot_product_of_ab_ac :
  vec_dot (1, -2) (2, -2) = 6 := by
  sorry

end dot_product_of_ab_ac_l13_13568


namespace average_next_seven_consecutive_is_correct_l13_13781

-- Define the sum of seven consecutive integers starting at x.
def sum_seven_consecutive_integers (x : ℕ) : ℕ := 7 * x + 21

-- Define the next sequence of seven integers starting from y + 1.
def average_next_seven_consecutive_integers (x : ℕ) : ℕ :=
  let y := sum_seven_consecutive_integers x
  let start := y + 1
  (start + (start + 1) + (start + 2) + (start + 3) + (start + 4) + (start + 5) + (start + 6)) / 7

-- Problem statement
theorem average_next_seven_consecutive_is_correct (x : ℕ) : 
  average_next_seven_consecutive_integers x = 7 * x + 25 :=
by
  sorry

end average_next_seven_consecutive_is_correct_l13_13781


namespace total_cost_production_l13_13786

variable (FC MC : ℕ) (n : ℕ)

theorem total_cost_production : FC = 12000 → MC = 200 → n = 20 → (FC + MC * n = 16000) :=
by
  intro hFC hMC hn
  sorry

end total_cost_production_l13_13786


namespace mean_of_second_set_l13_13765

theorem mean_of_second_set (x : ℝ)
  (H1 : (28 + x + 70 + 88 + 104) / 5 = 67) :
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 :=
sorry

end mean_of_second_set_l13_13765


namespace mushroom_drying_l13_13791

theorem mushroom_drying (M M' : ℝ) (m1 m2 : ℝ) :
  M = 100 ∧ m1 = 0.01 * M ∧ m2 = 0.02 * M' ∧ m1 = 1 → M' = 50 :=
by
  sorry

end mushroom_drying_l13_13791


namespace find_x_value_l13_13102

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x)
  else Real.log x * Real.log 81

theorem find_x_value (x : ℝ) (h : f x = 1 / 4) : x = 3 :=
sorry

end find_x_value_l13_13102


namespace percent_increase_equilateral_triangles_l13_13469

noncomputable def side_length (n : ℕ) : ℕ :=
  if n = 0 then 3 else 2 ^ n * 3

noncomputable def perimeter (n : ℕ) : ℕ :=
  3 * side_length n

noncomputable def percent_increase (initial : ℕ) (final : ℕ) : ℚ := 
  ((final - initial) / initial) * 100

theorem percent_increase_equilateral_triangles :
  percent_increase (perimeter 0) (perimeter 4) = 1500 := by
  sorry

end percent_increase_equilateral_triangles_l13_13469


namespace population_scientific_notation_l13_13181

theorem population_scientific_notation : 
  (1.41: ℝ) * (10 ^ 9) = 1.41 * 10 ^ 9 := 
by
  sorry

end population_scientific_notation_l13_13181


namespace find_principal_l13_13009

theorem find_principal
  (R : ℝ) (T : ℕ) (interest_less_than_principal : ℝ) : 
  R = 0.05 → 
  T = 10 → 
  interest_less_than_principal = 3100 → 
  ∃ P : ℝ, P - ((P * R * T): ℝ) = P - interest_less_than_principal ∧ P = 6200 :=
by
  sorry

end find_principal_l13_13009


namespace maximize_revenue_l13_13809

def revenue (p : ℝ) : ℝ := 150 * p - 4 * p^2

theorem maximize_revenue : 
  ∃ p, 0 ≤ p ∧ p ≤ 30 ∧ p = 18.75 ∧ (∀ q, 0 ≤ q ∧ q ≤ 30 → revenue q ≤ revenue 18.75) :=
by
  sorry

end maximize_revenue_l13_13809


namespace faucet_fill_time_l13_13442

theorem faucet_fill_time (r : ℝ) (T1 T2 t : ℝ) (F1 F2 : ℕ) (h1 : T1 = 200) (h2 : t = 8) (h3 : F1 = 4) (h4 : F2 = 8) (h5 : T2 = 50) (h6 : r * F1 * t = T1) : 
(F2 * r) * t / (F1 * F2) = T2 -> by sorry := sorry

#check faucet_fill_time

end faucet_fill_time_l13_13442


namespace inequality_solution_addition_eq_seven_l13_13554

theorem inequality_solution_addition_eq_seven (b c : ℝ) :
  (∀ x : ℝ, -5 < 2 * x - 3 ∧ 2 * x - 3 < 5 → -1 < x ∧ x < 4) →
  (∀ x : ℝ, -x^2 + b * x + c = 0 ↔ (x = -1 ∨ x = 4)) →
  b + c = 7 :=
by
  intro h1 h2
  sorry

end inequality_solution_addition_eq_seven_l13_13554


namespace floor_painting_cost_l13_13534

noncomputable def floor_painting_problem : Prop := 
  ∃ (B L₁ L₂ B₂ Area₁ Area₂ CombinedCost : ℝ),
  L₁ = 2 * B ∧
  Area₁ = L₁ * B ∧
  484 = Area₁ * 3 ∧
  L₂ = 0.8 * L₁ ∧
  B₂ = 1.3 * B ∧
  Area₂ = L₂ * B₂ ∧
  CombinedCost = 484 + (Area₂ * 5) ∧
  CombinedCost = 1320.8

theorem floor_painting_cost : floor_painting_problem :=
by
  sorry

end floor_painting_cost_l13_13534


namespace monotone_increasing_interval_for_shifted_function_l13_13674

variable (f : ℝ → ℝ)

-- Given definition: f(x+1) is an even function
def even_function : Prop :=
  ∀ x, f (x+1) = f (-(x+1))

-- Given condition: f(x+1) is monotonically decreasing on [0, +∞)
def monotone_decreasing_on_nonneg : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f (x+1) ≥ f (y+1)

-- Theorem to prove: the interval on which f(x-1) is monotonically increasing is (-∞, 2]
theorem monotone_increasing_interval_for_shifted_function
  (h_even : even_function f)
  (h_mono_dec : monotone_decreasing_on_nonneg f) :
  ∀ x y, x ≤ 2 → y ≤ 2 → x ≤ y → f (x-1) ≤ f (y-1) :=
by
  sorry

end monotone_increasing_interval_for_shifted_function_l13_13674


namespace abs_floor_value_l13_13226

theorem abs_floor_value : (Int.floor (|(-56.3: Real)|)) = 56 := 
by
  sorry

end abs_floor_value_l13_13226


namespace right_triangle_ineq_l13_13929

variable (a b c : ℝ)
variable (h : c^2 = a^2 + b^2)

theorem right_triangle_ineq (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a^3 + b^3 + c^3) / (a * b * (a + b + c)) ≥ Real.sqrt 2 :=
by
  sorry

end right_triangle_ineq_l13_13929


namespace find_cost_price_of_article_l13_13920

theorem find_cost_price_of_article 
  (C : ℝ) 
  (h1 : 1.05 * C - 2 = 1.045 * C) 
  (h2 : 0.005 * C = 2) 
: C = 400 := 
by 
  sorry

end find_cost_price_of_article_l13_13920


namespace smallest_integer_in_set_l13_13262

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 144) (h2 : greatest = 153) : ∃ x : ℤ, x = 135 :=
by
  sorry

end smallest_integer_in_set_l13_13262


namespace symmetric_diff_cardinality_l13_13232

theorem symmetric_diff_cardinality (X Y : Finset ℤ) 
  (hX : X.card = 8) 
  (hY : Y.card = 10) 
  (hXY : (X ∩ Y).card = 6) : 
  (X \ Y ∪ Y \ X).card = 6 := 
by
  sorry

end symmetric_diff_cardinality_l13_13232


namespace jina_teddies_l13_13879

variable (T : ℕ)

def initial_teddies (bunnies : ℕ) (koala : ℕ) (add_teddies : ℕ) (total : ℕ) :=
  T + bunnies + add_teddies + koala

theorem jina_teddies (bunnies : ℕ) (koala : ℕ) (add_teddies : ℕ) (total : ℕ) :
  bunnies = 3 * T ∧ koala = 1 ∧ add_teddies = 2 * bunnies ∧ total = 51 → T = 5 :=
by
  sorry

end jina_teddies_l13_13879


namespace no_common_complex_roots_l13_13461

theorem no_common_complex_roots (a b : ℚ) :
  ¬ ∃ α : ℂ, (α^5 - α - 1 = 0) ∧ (α^2 + a * α + b = 0) :=
sorry

end no_common_complex_roots_l13_13461


namespace sugar_for_third_layer_l13_13335

theorem sugar_for_third_layer (s1 : ℕ) (s2 : ℕ) (s3 : ℕ) 
  (h1 : s1 = 2) 
  (h2 : s2 = 2 * s1) 
  (h3 : s3 = 3 * s2) : 
  s3 = 12 := 
sorry

end sugar_for_third_layer_l13_13335


namespace factor_expression_l13_13128

variable (x : ℝ)

theorem factor_expression :
  (4 * x ^ 3 + 100 * x ^ 2 - 28) - (-9 * x ^ 3 + 2 * x ^ 2 - 28) = 13 * x ^ 2 * (x + 7) :=
by
  sorry

end factor_expression_l13_13128


namespace stratified_sampling_l13_13058

theorem stratified_sampling
  (total_products : ℕ)
  (sample_size : ℕ)
  (workshop_products : ℕ)
  (h1 : total_products = 2048)
  (h2 : sample_size = 128)
  (h3 : workshop_products = 256) :
  (workshop_products / total_products) * sample_size = 16 := 
by
  rw [h1, h2, h3]
  norm_num
  
  sorry

end stratified_sampling_l13_13058


namespace my_op_five_four_l13_13336

-- Define the operation a * b
def my_op (a b : ℤ) := a^2 + a * b - b^2

-- Define the theorem to prove 5 * 4 = 29 given the defined operation my_op
theorem my_op_five_four : my_op 5 4 = 29 := 
by 
sorry

end my_op_five_four_l13_13336


namespace smallest_b_for_undefined_inverse_mod_70_77_l13_13795

theorem smallest_b_for_undefined_inverse_mod_70_77 (b : ℕ) :
  (∀ k, k < b → k * 1 % 70 ≠ 1 ∧ k * 1 % 77 ≠ 1) ∧ (b * 1 % 70 ≠ 1) ∧ (b * 1 % 77 ≠ 1) → b = 7 :=
by sorry

end smallest_b_for_undefined_inverse_mod_70_77_l13_13795


namespace trapezoid_segment_AB_length_l13_13912

/-
In the trapezoid shown, the ratio of the area of triangle ABC to the area of triangle ADC is 5:2.
If AB + CD = 240 cm, prove that the length of segment AB is 171.42857 cm.
-/

theorem trapezoid_segment_AB_length
  (AB CD : ℝ)
  (ratio_areas : ℝ := 5 / 2)
  (area_ratio_condition : AB / CD = ratio_areas)
  (length_sum_condition : AB + CD = 240) :
  AB = 171.42857 :=
sorry

end trapezoid_segment_AB_length_l13_13912


namespace binomial_expansion_example_l13_13930

theorem binomial_expansion_example : 7^3 + 3 * (7^2) * 2 + 3 * 7 * (2^2) + 2^3 = 729 := by
  sorry

end binomial_expansion_example_l13_13930


namespace total_weight_of_balls_l13_13143

theorem total_weight_of_balls :
  let weight_blue := 6
  let weight_brown := 3.12
  let weight_green := 4.5
  weight_blue + weight_brown + weight_green = 13.62 := by
  sorry

end total_weight_of_balls_l13_13143


namespace ellipse_equation_l13_13465

theorem ellipse_equation (a b c : ℝ) :
  (2 * a = 10) ∧ (c / a = 4 / 5) →
  ((x:ℝ)^2 / 25 + (y:ℝ)^2 / 9 = 1) ∨ ((x:ℝ)^2 / 9 + (y:ℝ)^2 / 25 = 1) :=
by
  sorry

end ellipse_equation_l13_13465


namespace tan_seven_pi_over_six_l13_13996
  
theorem tan_seven_pi_over_six :
  Real.tan (7 * Real.pi / 6) = 1 / Real.sqrt 3 :=
sorry

end tan_seven_pi_over_six_l13_13996


namespace remainder_2468135792_mod_101_l13_13242

theorem remainder_2468135792_mod_101 : 
  2468135792 % 101 = 47 := 
sorry

end remainder_2468135792_mod_101_l13_13242


namespace eagles_per_section_l13_13412

theorem eagles_per_section (total_eagles sections : ℕ) (h1 : total_eagles = 18) (h2 : sections = 3) :
  total_eagles / sections = 6 := by
  sorry

end eagles_per_section_l13_13412


namespace ratio_of_patients_l13_13216

def one_in_four_zx (current_patients : ℕ) : ℕ :=
  current_patients / 4

def previous_patients : ℕ :=
  26

def diagnosed_patients : ℕ :=
  13

def current_patients : ℕ :=
  diagnosed_patients * 4

theorem ratio_of_patients : 
  one_in_four_zx current_patients = diagnosed_patients → 
  (current_patients / previous_patients) = 2 := 
by 
  sorry

end ratio_of_patients_l13_13216


namespace area_percent_difference_l13_13804

theorem area_percent_difference (b h : ℝ) (hb : b > 0) (hh : h > 0) : 
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  percent_difference = 4 := 
by
  let area_B := (b * h) / 2
  let area_A := ((1.20 * b) * (0.80 * h)) / 2
  let percent_difference := ((area_B - area_A) / area_B) * 100
  sorry

end area_percent_difference_l13_13804


namespace probability_at_least_one_boy_one_girl_l13_13806

def boys := 12
def girls := 18
def total_members := 30
def committee_size := 6

def total_ways := Nat.choose total_members committee_size
def all_boys_ways := Nat.choose boys committee_size
def all_girls_ways := Nat.choose girls committee_size
def all_boys_or_girls_ways := all_boys_ways + all_girls_ways
def complementary_probability := all_boys_or_girls_ways / total_ways
def desired_probability := 1 - complementary_probability

theorem probability_at_least_one_boy_one_girl :
  desired_probability = (574287 : ℚ) / 593775 :=
  sorry

end probability_at_least_one_boy_one_girl_l13_13806


namespace sufficient_condition_inequalities_l13_13186

theorem sufficient_condition_inequalities (x a : ℝ) :
  (¬ (a-4 < x ∧ x < a+4) → ¬ (1 < x ∧ x < 2)) ↔ -2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end sufficient_condition_inequalities_l13_13186


namespace total_noodles_and_pirates_l13_13509

-- Condition definitions
def pirates : ℕ := 45
def noodles : ℕ := pirates - 7

-- Theorem stating the total number of noodles and pirates
theorem total_noodles_and_pirates : (noodles + pirates) = 83 := by
  sorry

end total_noodles_and_pirates_l13_13509


namespace special_blend_probability_l13_13200

/-- Define the probability variables and conditions -/
def visit_count : ℕ := 6
def special_blend_prob : ℚ := 3 / 4
def non_special_blend_prob : ℚ := 1 / 4

/-- The binomial coefficient for choosing 5 days out of 6 -/
def choose_6_5 : ℕ := Nat.choose 6 5

/-- The probability of serving the special blend exactly 5 times out of 6 -/
def prob_special_blend_5 : ℚ := (choose_6_5 : ℚ) * (special_blend_prob ^ 5) * (non_special_blend_prob ^ 1)

/-- Statement to prove the desired probability -/
theorem special_blend_probability :
  prob_special_blend_5 = 1458 / 4096 :=
by
  sorry

end special_blend_probability_l13_13200


namespace max_min_value_l13_13567

noncomputable def f (A B x a b : ℝ) : ℝ :=
  A * Real.sqrt (x - a) + B * Real.sqrt (b - x)

theorem max_min_value (A B a b : ℝ) (hA : A > 0) (hB : B > 0) (ha_lt_b : a < b) :
  (∀ x, a ≤ x ∧ x ≤ b → f A B x a b ≤ Real.sqrt ((A^2 + B^2) * (b - a))) ∧
  min (f A B a a b) (f A B b a b) ≤ f A B x a b :=
  sorry

end max_min_value_l13_13567


namespace ellipse_equation_and_slope_range_l13_13409

theorem ellipse_equation_and_slope_range (a b : ℝ) (e : ℝ) (k : ℝ) :
  a > b ∧ b > 0 ∧ e = (Real.sqrt 3) / 3 ∧
  ∃! ℓ : ℝ × ℝ, (ℓ.2 = 1 ∧ ℓ.1 = -2) ∧
  ∀ x y : ℝ, x^2 + y^2 = b^2 → y = x + 2 →
  ((x - 0)^2 + (y - 0)^2 = b^2) ∧
  (
    (a^2 = (3 * b^2)) ∧ (b = Real.sqrt 2) ∧
    a > 0 ∧
    (∀ x y : ℝ, x^2 / 3 + y^2 / 2 = 1) ∧
    (-((Real.sqrt 2) / 2) < k ∧ k < 0) ∨ (0 < k ∧ k < ((Real.sqrt 2) / 2))
  ) :=
by
  sorry

end ellipse_equation_and_slope_range_l13_13409


namespace angle_A_value_cos_A_minus_2x_value_l13_13348

open Real

-- Let A, B, and C be the internal angles of triangle ABC.
variable {A B C x : ℝ}

-- Given conditions
axiom triangle_angles : A + B + C = π
axiom sinC_eq_2sinAminusB : sin C = 2 * sin (A - B)
axiom B_is_pi_over_6 : B = π / 6
axiom cosAplusx_is_neg_third : cos (A + x) = -1 / 3

-- Proof goals
theorem angle_A_value : A = π / 3 := by sorry

theorem cos_A_minus_2x_value : cos (A - 2 * x) = 7 / 9 := by sorry

end angle_A_value_cos_A_minus_2x_value_l13_13348


namespace smaller_group_men_l13_13829

-- Define the main conditions of the problem
def men_work_days : ℕ := 36 * 18  -- 36 men for 18 days

-- Define the theorem we need to prove
theorem smaller_group_men (M : ℕ) (h: M * 72 = men_work_days) : M = 9 :=
by
  -- proof is not required
  sorry

end smaller_group_men_l13_13829


namespace derivative_at_pi_over_4_l13_13935

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem derivative_at_pi_over_4 : (deriv f (Real.pi / 4)) = -2 :=
by
  sorry

end derivative_at_pi_over_4_l13_13935


namespace range_of_a_l13_13279

theorem range_of_a (x a : ℝ) (h₁ : 0 < x) (h₂ : x < 2) (h₃ : a - 1 < x) (h₄ : x ≤ a) :
  1 ≤ a ∧ a < 2 :=
by
  sorry

end range_of_a_l13_13279


namespace six_div_one_minus_three_div_ten_equals_twenty_four_l13_13033

theorem six_div_one_minus_three_div_ten_equals_twenty_four :
  (6 : ℤ) / (1 - (3 : ℤ) / (10 : ℤ)) = 24 := 
by
  sorry

end six_div_one_minus_three_div_ten_equals_twenty_four_l13_13033


namespace infinite_solutions_iff_c_is_5_over_2_l13_13646

theorem infinite_solutions_iff_c_is_5_over_2 (c : ℝ) :
  (∀ y : ℝ, 3 * (2 + 2 * c * y) = 15 * y + 6) ↔ c = 5 / 2 :=
by 
  sorry

end infinite_solutions_iff_c_is_5_over_2_l13_13646


namespace shift_right_graph_l13_13662

theorem shift_right_graph (x : ℝ) :
  (3 : ℝ)^(x+1) = (3 : ℝ)^((x+1) - 1) :=
by 
  -- Here we prove that shifting the graph of y = 3^(x+1) to right by 1 unit 
  -- gives the graph of y = 3^x
  sorry

end shift_right_graph_l13_13662


namespace cookies_left_after_ted_leaves_l13_13521

theorem cookies_left_after_ted_leaves :
  let f : Nat := 2 -- trays per day
  let d : Nat := 6 -- days
  let e_f : Nat := 1 -- cookies eaten per day by Frank
  let t : Nat := 4 -- cookies eaten by Ted
  let c : Nat := 12 -- cookies per tray
  let total_cookies := f * c * d -- total cookies baked
  let cookies_eaten_by_frank := e_f * d -- total cookies eaten by Frank
  let cookies_before_ted := total_cookies - cookies_eaten_by_frank -- cookies before Ted
  let total_cookies_left := cookies_before_ted - t -- cookies left after Ted
  total_cookies_left = 134
:= by
  sorry

end cookies_left_after_ted_leaves_l13_13521


namespace dividend_is_217_l13_13680

-- Given conditions
def r : ℕ := 1
def q : ℕ := 54
def d : ℕ := 4

-- Define the problem as a theorem in Lean 4
theorem dividend_is_217 : (d * q) + r = 217 := by
  -- proof is omitted
  sorry

end dividend_is_217_l13_13680


namespace radius_of_inscribed_circle_l13_13760

variable (height : ℝ) (alpha : ℝ)

theorem radius_of_inscribed_circle (h : ℝ) (α : ℝ) : 
∃ r : ℝ, r = (h / 2) * (Real.tan (Real.pi / 4 - α / 4)) ^ 2 := 
sorry

end radius_of_inscribed_circle_l13_13760


namespace water_tank_height_l13_13341

theorem water_tank_height (r h : ℝ) (V : ℝ) (V_water : ℝ) (a b : ℕ) 
  (h_tank : h = 120) (r_tank : r = 20) (V_tank : V = (1/3) * π * r^2 * h) 
  (V_water_capacity : V_water = 0.4 * V) :
  a = 48 ∧ b = 2 ∧ V = 16000 * π ∧ V_water = 6400 * π ∧ 
  h_water = 48 * (2^(1/3) / 1) ∧ (a + b = 50) :=
by
  sorry

end water_tank_height_l13_13341


namespace peter_reads_one_book_18_hours_l13_13707

-- Definitions of conditions given in the problem
variables (P : ℕ)

-- Condition: Peter can read three times as fast as Kristin
def reads_three_times_as_fast (P : ℕ) : Prop :=
  ∀ (K : ℕ), K = 3 * P

-- Condition: Kristin reads half of her 20 books in 540 hours
def half_books_in_540_hours (K : ℕ) : Prop :=
  K = 54

-- Theorem stating the main proof problem: proving P equals 18 hours
theorem peter_reads_one_book_18_hours
  (H1 : reads_three_times_as_fast P)
  (H2 : half_books_in_540_hours (3 * P)) :
  P = 18 :=
sorry

end peter_reads_one_book_18_hours_l13_13707


namespace range_of_m_l13_13299

noncomputable def problem (x m : ℝ) (p q : Prop) : Prop :=
  (¬ p → ¬ q) ∧ (¬ q → ¬ p → False) ∧ (p ↔ |1 - (x - 1) / 3| ≤ 2) ∧ 
  (q ↔ x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0)

theorem range_of_m (m : ℝ) (x : ℝ) (p q : Prop) 
  (h : problem x m p q) : m ≥ 9 :=
sorry

end range_of_m_l13_13299


namespace Billy_is_45_l13_13137

variable (B J : ℕ)

-- Condition 1: Billy's age is three times Joe's age
def condition1 : Prop := B = 3 * J

-- Condition 2: The sum of their ages is 60
def condition2 : Prop := B + J = 60

-- The theorem we want to prove: Billy's age is 45
theorem Billy_is_45 (h1 : condition1 B J) (h2 : condition2 B J) : B = 45 := 
sorry

end Billy_is_45_l13_13137


namespace inverse_variation_y_at_x_l13_13818

variable (k x y : ℝ)

theorem inverse_variation_y_at_x :
  (∀ x y k, y = k / x → y = 6 → x = 3 → k = 18) → 
  k = 18 →
  x = 12 →
  y = 18 / 12 →
  y = 3 / 2 := by
  intros h1 h2 h3 h4
  sorry

end inverse_variation_y_at_x_l13_13818


namespace stationery_store_profit_l13_13850

variable (a : ℝ)

def store_cost : ℝ := 100 * a
def markup_price : ℝ := a * 1.2
def discount_price : ℝ := markup_price a * 0.8

def revenue_first_half : ℝ := 50 * markup_price a
def revenue_second_half : ℝ := 50 * discount_price a
def total_revenue : ℝ := revenue_first_half a + revenue_second_half a

def profit : ℝ := total_revenue a - store_cost a

theorem stationery_store_profit : profit a = 8 * a := 
by sorry

end stationery_store_profit_l13_13850


namespace part1_part2_l13_13940

def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + a + 1

-- Proof problem 1: Prove that if a = 2, then f(x) ≥ 0 is equivalent to x ≥ 3/2 or x ≤ 1.
theorem part1 (x : ℝ) : f 2 x ≥ 0 ↔ x ≥ (3 / 2 : ℝ) ∨ x ≤ 1 := sorry

-- Proof problem 2: Prove that for a∈[-2,2], if f(x) < 0 always holds, then x ∈ (1, 3/2).
theorem part2 (a x : ℝ) (ha : a ≥ -2 ∧ a ≤ 2) : (∀ x, f a x < 0) ↔ 1 < x ∧ x < (3 / 2 : ℝ) := sorry

end part1_part2_l13_13940


namespace electronics_sale_negation_l13_13014

variables (E : Type) (storeElectronics : E → Prop) (onSale : E → Prop)

theorem electronics_sale_negation
  (H : ¬ ∀ e, storeElectronics e → onSale e) :
  (∃ e, storeElectronics e ∧ ¬ onSale e) ∧ ¬ ∀ e, storeElectronics e → onSale e :=
by
  -- Proving that at least one electronic is not on sale follows directly from the negation of the universal statement
  sorry

end electronics_sale_negation_l13_13014


namespace john_coffees_per_day_l13_13353

theorem john_coffees_per_day (x : ℕ)
  (h1 : ∀ p : ℕ, p = 2)
  (h2 : ∀ p : ℕ, p = p + p / 2)
  (h3 : ∀ n : ℕ, n = x / 2)
  (h4 : ∀ d : ℕ, 2 * x - 3 * (x / 2) = 2) :
  x = 4 :=
by
  sorry

end john_coffees_per_day_l13_13353


namespace exists_periodic_sequence_of_period_ge_two_l13_13784

noncomputable def periodic_sequence (x : ℕ → ℝ) (p : ℕ) : Prop :=
  ∀ n, x (n + p) = x n

theorem exists_periodic_sequence_of_period_ge_two :
  ∀ (p : ℕ), p ≥ 2 →
  ∃ (x : ℕ → ℝ), periodic_sequence x p ∧ 
  ∀ n, x (n + 1) = x n - (1 / x n) :=
by {
  sorry
}

end exists_periodic_sequence_of_period_ge_two_l13_13784


namespace evan_amount_l13_13972

def adrian : ℤ := sorry
def brenda : ℤ := sorry
def charlie : ℤ := sorry
def dana : ℤ := sorry
def evan : ℤ := sorry

def amounts_sum : Prop := adrian + brenda + charlie + dana + evan = 72
def abs_diff_1 : Prop := abs (adrian - brenda) = 21
def abs_diff_2 : Prop := abs (brenda - charlie) = 8
def abs_diff_3 : Prop := abs (charlie - dana) = 6
def abs_diff_4 : Prop := abs (dana - evan) = 5
def abs_diff_5 : Prop := abs (evan - adrian) = 14

theorem evan_amount
  (h_sum : amounts_sum)
  (h_diff1 : abs_diff_1)
  (h_diff2 : abs_diff_2)
  (h_diff3 : abs_diff_3)
  (h_diff4 : abs_diff_4)
  (h_diff5 : abs_diff_5) :
  evan = 21 := sorry

end evan_amount_l13_13972


namespace exists_root_abs_leq_2_abs_c_div_b_l13_13068

theorem exists_root_abs_leq_2_abs_c_div_b (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h_real_roots : ∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ |x| ≤ 2 * |c / b| :=
by
  sorry

end exists_root_abs_leq_2_abs_c_div_b_l13_13068


namespace cups_of_rice_morning_l13_13227

variable (cupsMorning : Nat) -- Number of cups of rice Robbie eats in the morning
variable (cupsAfternoon : Nat := 2) -- Cups of rice in the afternoon
variable (cupsEvening : Nat := 5) -- Cups of rice in the evening
variable (fatPerCup : Nat := 10) -- Fat in grams per cup of rice
variable (weeklyFatIntake : Nat := 700) -- Total fat in grams per week

theorem cups_of_rice_morning :
  ((cupsMorning + cupsAfternoon + cupsEvening) * fatPerCup) = (weeklyFatIntake / 7) → cupsMorning = 3 :=
  by
    sorry

end cups_of_rice_morning_l13_13227


namespace distinct_integers_sum_l13_13394

theorem distinct_integers_sum (n : ℕ) (h : n > 3) (a : Fin n → ℤ)
  (h1 : ∀ i, 1 ≤ a i) (h2 : ∀ i j, i < j → a i < a j) (h3 : ∀ i, a i ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
  k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ a i + a j = a k + a l ∧ a k + a l = a m :=
by
  sorry

end distinct_integers_sum_l13_13394


namespace least_positive_integer_l13_13323

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l13_13323


namespace carlson_handkerchief_usage_l13_13198

def problem_statement : Prop :=
  let handkerchief_area := 25 * 25 -- Area in cm²
  let total_fabric_area := 3 * 10000 -- Total fabric area in cm²
  let days := 8
  let total_handkerchiefs := total_fabric_area / handkerchief_area
  let handkerchiefs_per_day := total_handkerchiefs / days
  handkerchiefs_per_day = 6

theorem carlson_handkerchief_usage : problem_statement := by
  sorry

end carlson_handkerchief_usage_l13_13198


namespace number_is_seven_point_five_l13_13698

theorem number_is_seven_point_five (x : ℝ) (h : x^2 + 100 = (x - 20)^2) : x = 7.5 :=
by
  sorry

end number_is_seven_point_five_l13_13698


namespace angie_total_taxes_l13_13344

theorem angie_total_taxes:
  ∀ (salary : ℕ) (N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over : ℕ),
  salary = 80 →
  N_1 = 12 → T_1 = 8 → U_1 = 5 →
  N_2 = 15 → T_2 = 6 → U_2 = 7 →
  N_3 = 10 → T_3 = 9 → U_3 = 6 →
  N_4 = 14 → T_4 = 7 → U_4 = 4 →
  left_over = 18 →
  T_1 + T_2 + T_3 + T_4 = 30 :=
by
  intros salary N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over
  sorry

end angie_total_taxes_l13_13344


namespace Tim_marbles_l13_13064

theorem Tim_marbles (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 :=
by
  sorry

end Tim_marbles_l13_13064


namespace room_dimension_l13_13561

theorem room_dimension
  (x : ℕ)
  (cost_per_sqft : ℕ := 4)
  (dimension_1 : ℕ := 15)
  (dimension_2 : ℕ := 12)
  (door_width : ℕ := 6)
  (door_height : ℕ := 3)
  (num_windows : ℕ := 3)
  (window_width : ℕ := 4)
  (window_height : ℕ := 3)
  (total_cost : ℕ := 3624) :
  (2 * (x * dimension_1) + 2 * (x * dimension_2) - (door_width * door_height + num_windows * (window_width * window_height))) * cost_per_sqft = total_cost →
  x = 18 :=
by
  sorry

end room_dimension_l13_13561


namespace complement_of_A_in_U_l13_13479

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ 0}
def C_UA : Set ℝ := U \ A

theorem complement_of_A_in_U :
  C_UA = {x | 0 < x ∧ x < 1} :=
sorry

end complement_of_A_in_U_l13_13479


namespace find_f_minus_2_l13_13691

namespace MathProof

def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 5

theorem find_f_minus_2 (a b c : ℝ) (h : f a b c 2 = 3) : f a b c (-2) = -13 := 
by
  sorry

end MathProof

end find_f_minus_2_l13_13691


namespace Mirella_read_purple_books_l13_13987

theorem Mirella_read_purple_books (P : ℕ) 
  (pages_per_purple_book : ℕ := 230)
  (pages_per_orange_book : ℕ := 510)
  (orange_books_read : ℕ := 4)
  (extra_orange_pages : ℕ := 890)
  (total_orange_pages : ℕ := orange_books_read * pages_per_orange_book)
  (total_purple_pages : ℕ := P * pages_per_purple_book)
  (condition : total_orange_pages - total_purple_pages = extra_orange_pages) :
  P = 5 := 
by 
  sorry

end Mirella_read_purple_books_l13_13987


namespace relation_correct_l13_13594

def M := {x : ℝ | x < 2}
def N := {x : ℝ | 0 < x ∧ x < 1}
def CR (S : Set ℝ) := {x : ℝ | x ∈ (Set.univ : Set ℝ) \ S}

theorem relation_correct : M ∪ CR N = (Set.univ : Set ℝ) :=
by sorry

end relation_correct_l13_13594


namespace probability_at_least_one_red_l13_13307

def total_balls : ℕ := 6
def red_balls : ℕ := 4
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_at_least_one_red :
  (choose_two red_balls + red_balls * (total_balls - red_balls - 1) / 2) / choose_two total_balls = 14 / 15 :=
sorry

end probability_at_least_one_red_l13_13307


namespace rhombus_area_l13_13111

theorem rhombus_area (R r : ℝ) : 
  ∃ A : ℝ, A = (8 * R^3 * r^3) / ((R^2 + r^2)^2) :=
by
  sorry

end rhombus_area_l13_13111


namespace fraction_min_sum_l13_13590

theorem fraction_min_sum (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) 
  (h_ineq : 45 * b < 110 * a ∧ 110 * a < 50 * b) :
  a = 3 ∧ b = 7 :=
sorry

end fraction_min_sum_l13_13590


namespace max_value_of_sequence_l13_13840

theorem max_value_of_sequence : 
  ∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → (∃ (a : ℝ), a = (m / (m^2 + 6 : ℝ)) ∧ a ≤ (n / (n^2 + 6 : ℝ))) :=
sorry

end max_value_of_sequence_l13_13840


namespace probability_all_quitters_same_tribe_l13_13500

theorem probability_all_quitters_same_tribe :
  ∀ (people : Finset ℕ) (tribe1 tribe2 : Finset ℕ) (choose : ℕ → ℕ → ℕ) (prob : ℚ),
  people.card = 20 →
  tribe1.card = 10 →
  tribe2.card = 10 →
  tribe1 ∪ tribe2 = people →
  tribe1 ∩ tribe2 = ∅ →
  choose 20 3 = 1140 →
  choose 10 3 = 120 →
  prob = (2 * choose 10 3) / choose 20 3 →
  prob = 20 / 95 :=
by
  intro people tribe1 tribe2 choose prob
  intros hp20 ht1 ht2 hu hi hchoose20 hchoose10 hprob
  sorry

end probability_all_quitters_same_tribe_l13_13500


namespace isosceles_base_length_l13_13179

theorem isosceles_base_length (x b : ℕ) (h1 : 2 * x + b = 40) (h2 : x = 15) : b = 10 :=
by
  sorry

end isosceles_base_length_l13_13179


namespace phase_shift_3cos_4x_minus_pi_over_4_l13_13838

theorem phase_shift_3cos_4x_minus_pi_over_4 :
    ∃ (φ : ℝ), y = 3 * Real.cos (4 * x - φ) ∧ φ = π / 16 :=
sorry

end phase_shift_3cos_4x_minus_pi_over_4_l13_13838


namespace sequence_gcd_is_index_l13_13719

theorem sequence_gcd_is_index (a : ℕ → ℕ) 
  (h : ∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) :
  ∀ i : ℕ, a i = i :=
by
  sorry

end sequence_gcd_is_index_l13_13719


namespace min_value_ineq_l13_13506

theorem min_value_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 3) : 
  (1 / a) + (4 / b) ≥ 3 :=
sorry

end min_value_ineq_l13_13506


namespace initial_bottle_caps_l13_13805

theorem initial_bottle_caps (bought_caps total_caps initial_caps : ℕ) 
  (hb : bought_caps = 41) (ht : total_caps = 43):
  initial_caps = 2 :=
by
  have h : total_caps = initial_caps + bought_caps := sorry
  have ha : initial_caps = total_caps - bought_caps := sorry
  exact sorry

end initial_bottle_caps_l13_13805


namespace euler_family_mean_age_l13_13658

theorem euler_family_mean_age : 
  let girls_ages := [5, 5, 10, 15]
  let boys_ages := [8, 12, 16]
  let children_ages := girls_ages ++ boys_ages
  let total_sum := List.sum children_ages
  let number_of_children := List.length children_ages
  (total_sum : ℚ) / number_of_children = 10.14 := 
by
  sorry

end euler_family_mean_age_l13_13658


namespace pages_left_to_read_l13_13106

-- Defining the given conditions
def total_pages : ℕ := 500
def read_first_night : ℕ := (20 * total_pages) / 100
def read_second_night : ℕ := (20 * total_pages) / 100
def read_third_night : ℕ := (30 * total_pages) / 100

-- The total pages read over the three nights
def total_read : ℕ := read_first_night + read_second_night + read_third_night

-- The remaining pages to be read
def remaining_pages : ℕ := total_pages - total_read

theorem pages_left_to_read : remaining_pages = 150 :=
by
  -- Leaving the proof as a placeholder
  sorry

end pages_left_to_read_l13_13106


namespace inverse_relation_a1600_inverse_relation_a400_l13_13119

variable (a b : ℝ)

def k := 400 

theorem inverse_relation_a1600 : (a * b = k) → (a = 1600) → (b = 0.25) :=
by
  sorry

theorem inverse_relation_a400 : (a * b = k) → (a = 400) → (b = 1) :=
by
  sorry

end inverse_relation_a1600_inverse_relation_a400_l13_13119


namespace no_integer_solution_for_z_l13_13959

theorem no_integer_solution_for_z (z : ℤ) (h : 2 / z = 2 / (z + 1) + 2 / (z + 25)) : false :=
by
  sorry

end no_integer_solution_for_z_l13_13959


namespace non_neg_sum_sq_inequality_l13_13193

theorem non_neg_sum_sq_inequality (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
sorry

end non_neg_sum_sq_inequality_l13_13193


namespace jelly_bean_probability_l13_13709

theorem jelly_bean_probability :
  ∀ (P_red P_orange P_green P_yellow : ℝ),
  P_red = 0.1 →
  P_orange = 0.4 →
  P_green = 0.2 →
  P_red + P_orange + P_green + P_yellow = 1 →
  P_yellow = 0.3 :=
by
  intros P_red P_orange P_green P_yellow h_red h_orange h_green h_sum
  sorry

end jelly_bean_probability_l13_13709


namespace cycle_final_selling_price_l13_13697

-- Lean 4 statement capturing the problem definition and final selling price
theorem cycle_final_selling_price (original_price : ℝ) (initial_discount_rate : ℝ) 
  (loss_rate : ℝ) (exchange_discount_rate : ℝ) (final_price : ℝ) :
  original_price = 1400 →
  initial_discount_rate = 0.05 →
  loss_rate = 0.25 →
  exchange_discount_rate = 0.10 →
  final_price = 
    (original_price * (1 - initial_discount_rate) * (1 - loss_rate) * (1 - exchange_discount_rate)) →
  final_price = 897.75 :=
by
  sorry

end cycle_final_selling_price_l13_13697


namespace sequence_formula_l13_13677

open Nat

def a : ℕ → ℤ
| 0     => 0  -- Defining a(0) though not used
| 1     => 1
| (n+2) => 3 * a (n+1) + 2^(n+2)

theorem sequence_formula (n : ℕ) (hn : n ≥ 1) :
  a n = 5 * 3^(n-1) - 2^(n+1) :=
by
  sorry

end sequence_formula_l13_13677


namespace max_probability_pc_l13_13284

variables (p1 p2 p3 : ℝ)
variable (h : p3 > p2 ∧ p2 > p1 ∧ p1 > 0)

def PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3)
def PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)
def PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3)

theorem max_probability_pc : PC > PA ∧ PC > PB := 
by 
  sorry

end max_probability_pc_l13_13284


namespace negate_exists_l13_13557

theorem negate_exists (x : ℝ) : 
  (¬ ∃ x : ℝ, x < Real.sin x ∨ x > Real.tan x) ↔ (∀ x : ℝ, x ≥ Real.sin x ∨ x ≤ Real.tan x) :=
by
  sorry

end negate_exists_l13_13557


namespace smallest_positive_a_l13_13700

/-- Define a function f satisfying the given conditions. -/
noncomputable def f : ℝ → ℝ :=
  sorry -- we'll define it later according to the problem

axiom condition1 : ∀ x > 0, f (2 * x) = 2 * f x

axiom condition2 : ∀ x, 1 < x ∧ x < 2 → f x = 2 - x

theorem smallest_positive_a :
  (∃ a > 0, f a = f 2020) ∧ ∀ b > 0, (f b = f 2020 → b ≥ 36) :=
  sorry

end smallest_positive_a_l13_13700


namespace quadratic_function_two_distinct_roots_l13_13782

-- Define the discriminant of the quadratic function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the function that checks the conditions for the quadratic to have two distinct real roots
theorem quadratic_function_two_distinct_roots (a : ℝ) : 
  (0 < a ∧ a < 2) → (discriminant a (-4) 2 > 0) :=
by
  sorry

end quadratic_function_two_distinct_roots_l13_13782


namespace calculator_display_after_50_presses_l13_13333

theorem calculator_display_after_50_presses :
  let initial_display := 3
  let operation (x : ℚ) := 1 / (1 - x)
  (Nat.iterate operation 50 initial_display) = 2 / 3 :=
by
  sorry

end calculator_display_after_50_presses_l13_13333


namespace max_ab_perpendicular_l13_13464

theorem max_ab_perpendicular (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 2 * a + b = 3) : ab <= (9 / 8) := 
sorry

end max_ab_perpendicular_l13_13464


namespace sufficient_not_necessary_condition_l13_13365

variable (x y : ℝ)

theorem sufficient_not_necessary_condition :
  (x > 1 ∧ y > 1) → (x + y > 2 ∧ x * y > 1) ∧
  ¬((x + y > 2 ∧ x * y > 1) → (x > 1 ∧ y > 1)) :=
by
  sorry

end sufficient_not_necessary_condition_l13_13365


namespace blueberries_cartons_proof_l13_13187

def total_needed_cartons : ℕ := 26
def strawberries_cartons : ℕ := 10
def cartons_to_buy : ℕ := 7

theorem blueberries_cartons_proof :
  strawberries_cartons + cartons_to_buy + 9 = total_needed_cartons :=
by
  sorry

end blueberries_cartons_proof_l13_13187


namespace rhombus_area_correct_l13_13305

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem rhombus_area_correct (x : ℝ) (h1 : rhombus_area 7 (abs (8 - x)) = 56) 
    (h2 : x ≠ 8) : x = -8 ∨ x = 24 :=
by
  sorry

end rhombus_area_correct_l13_13305


namespace fg_3_eq_7_l13_13343

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 2) ^ 2

theorem fg_3_eq_7 : f (g 3) = 7 :=
by
  sorry

end fg_3_eq_7_l13_13343


namespace divisor_iff_even_l13_13636

noncomputable def hasDivisor (k : ℕ) : Prop := 
  ∃ n : ℕ, n > 0 ∧ (8 * k * n - 1) ∣ (4 * k ^ 2 - 1) ^ 2

theorem divisor_iff_even (k : ℕ) (h : k > 0) : hasDivisor k ↔ (k % 2 = 0) :=
by
  sorry

end divisor_iff_even_l13_13636


namespace divisible_sum_or_difference_l13_13794

theorem divisible_sum_or_difference (a : Fin 52 → ℤ) :
  ∃ i j, (i ≠ j) ∧ (a i + a j) % 100 = 0 ∨ (a i - a j) % 100 = 0 :=
by
  sorry

end divisible_sum_or_difference_l13_13794


namespace brown_eyed_brunettes_count_l13_13682

/--
There are 50 girls in a group. Each girl is either blonde or brunette and either blue-eyed or brown-eyed.
14 girls are blue-eyed blondes. 31 girls are brunettes. 18 girls are brown-eyed.
Prove that the number of brown-eyed brunettes is equal to 13.
-/
theorem brown_eyed_brunettes_count
  (total_girls : ℕ)
  (blue_eyed_blondes : ℕ)
  (total_brunettes : ℕ)
  (total_brown_eyed : ℕ)
  (total_girls_eq : total_girls = 50)
  (blue_eyed_blondes_eq : blue_eyed_blondes = 14)
  (total_brunettes_eq : total_brunettes = 31)
  (total_brown_eyed_eq : total_brown_eyed = 18) :
  ∃ (brown_eyed_brunettes : ℕ), brown_eyed_brunettes = 13 :=
by sorry

end brown_eyed_brunettes_count_l13_13682


namespace abs_inequality_solution_l13_13253

theorem abs_inequality_solution (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 :=
by
  sorry

end abs_inequality_solution_l13_13253


namespace sarah_initial_money_l13_13062

-- Definitions based on conditions
def cost_toy_car := 11
def cost_scarf := 10
def cost_beanie := 14
def remaining_money := 7
def total_cost := 2 * cost_toy_car + cost_scarf + cost_beanie
def initial_money := total_cost + remaining_money

-- Statement of the theorem
theorem sarah_initial_money : initial_money = 53 :=
by
  sorry

end sarah_initial_money_l13_13062


namespace probability_not_within_square_B_l13_13211

theorem probability_not_within_square_B {A B : Type} 
  (area_A : ℝ) (perimeter_B : ℝ) (area_B : ℝ) (not_covered : ℝ) 
  (h1 : area_A = 30) 
  (h2 : perimeter_B = 16) 
  (h3 : area_B = 16) 
  (h4 : not_covered = area_A - area_B) :
  (not_covered / area_A) = 7 / 15 := by sorry

end probability_not_within_square_B_l13_13211


namespace evaluate_expression_l13_13918

theorem evaluate_expression (A B : ℝ) (hA : A = 2^7) (hB : B = 3^6) : (A ^ (1 / 3)) * (B ^ (1 / 2)) = 108 * 2 ^ (1 / 3) :=
by
  sorry

end evaluate_expression_l13_13918


namespace restaurant_cost_l13_13375

theorem restaurant_cost (total_people kids adult_cost : ℕ)
  (h1 : total_people = 12)
  (h2 : kids = 7)
  (h3 : adult_cost = 3) :
  total_people - kids * adult_cost = 15 := by
  sorry

end restaurant_cost_l13_13375


namespace intersection_A_B_l13_13322

-- Definition of sets A and B
def A : Set ℤ := {0, 1, 2, 3}
def B : Set ℤ := { x | -1 ≤ x ∧ x < 3 }

-- Statement to prove
theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := 
sorry

end intersection_A_B_l13_13322


namespace flowers_count_l13_13884

theorem flowers_count (save_per_day : ℕ) (days : ℕ) (flower_cost : ℕ) (total_savings : ℕ) (flowers : ℕ) 
    (h1 : save_per_day = 2) 
    (h2 : days = 22) 
    (h3 : flower_cost = 4) 
    (h4 : total_savings = save_per_day * days) 
    (h5 : flowers = total_savings / flower_cost) : 
    flowers = 11 := 
sorry

end flowers_count_l13_13884


namespace average_of_six_starting_from_d_plus_one_l13_13733

theorem average_of_six_starting_from_d_plus_one (c d : ℝ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (c + 6) = ((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 6 := 
by 
-- Proof omitted; end with sorry
sorry

end average_of_six_starting_from_d_plus_one_l13_13733


namespace rectangular_prism_volume_is_60_l13_13438

def rectangularPrismVolume (a b c : ℕ) : ℕ := a * b * c 

theorem rectangular_prism_volume_is_60 (a b c : ℕ) 
  (h_ge_2 : a ≥ 2) (h_ge_2_b : b ≥ 2) (h_ge_2_c : c ≥ 2)
  (h_one_face : 2 * ((a-2)*(b-2) + (b-2)*(c-2) + (a-2)*(c-2)) = 24)
  (h_two_faces : 4 * ((a-2) + (b-2) + (c-2)) = 28) :
  rectangularPrismVolume a b c = 60 := 
  by sorry

end rectangular_prism_volume_is_60_l13_13438


namespace div_by_seven_iff_multiple_of_three_l13_13886

theorem div_by_seven_iff_multiple_of_three (n : ℕ) (hn : 0 < n) : 
  (7 ∣ (2^n - 1)) ↔ (3 ∣ n) := 
sorry

end div_by_seven_iff_multiple_of_three_l13_13886


namespace Chloe_wins_l13_13817

theorem Chloe_wins (C M : ℕ) (h_ratio : 8 * M = 3 * C) (h_Max : M = 9) : C = 24 :=
by {
    sorry
}

end Chloe_wins_l13_13817


namespace cost_of_each_skin_l13_13522

theorem cost_of_each_skin
  (total_value : ℕ)
  (overall_profit : ℚ)
  (profit_first : ℚ)
  (profit_second : ℚ)
  (total_sell : ℕ)
  (equality : (1 : ℚ) + profit_first ≠ 0 ∧ (1 : ℚ) + profit_second ≠ 0) :
  total_value = 2250 → overall_profit = 0.4 → profit_first = 0.25 → profit_second = -0.5 →
  total_sell = 3150 →
  ∃ x y : ℚ, x = 2700 ∧ y = -450 :=
by
  sorry

end cost_of_each_skin_l13_13522


namespace min_value_xy_l13_13721

theorem min_value_xy {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : (2 / x) + (8 / y) = 1) : x * y ≥ 64 :=
sorry

end min_value_xy_l13_13721


namespace solve_for_y_l13_13218

theorem solve_for_y : 
  ∀ (y : ℚ), y = 45 / (8 - 3 / 7) → y = 315 / 53 :=
by
  intro y
  intro h
  -- proof steps would be placed here
  sorry

end solve_for_y_l13_13218


namespace x_pow_12_eq_one_l13_13792

theorem x_pow_12_eq_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 :=
sorry

end x_pow_12_eq_one_l13_13792


namespace problem1_problem2_l13_13427

-- Problem (1): Maximum value of (a + 1/a)(b + 1/b)
theorem problem1 {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (a + 1/a) * (b + 1/b) ≤ 25 / 4 := 
sorry

-- Problem (2): Minimum value of u = (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3
theorem problem2 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
sorry

end problem1_problem2_l13_13427


namespace ceil_floor_difference_l13_13835

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end ceil_floor_difference_l13_13835


namespace possible_values_of_n_l13_13582

theorem possible_values_of_n (E M n : ℕ) (h1 : M + 3 = n * (E - 3)) (h2 : E + n = 3 * (M - n)) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end possible_values_of_n_l13_13582


namespace turnips_total_l13_13598

def melanie_turnips := 139
def benny_turnips := 113

def total_turnips (melanie_turnips benny_turnips : Nat) : Nat :=
  melanie_turnips + benny_turnips

theorem turnips_total :
  total_turnips melanie_turnips benny_turnips = 252 :=
by
  sorry

end turnips_total_l13_13598


namespace sum_of_digits_of_product_in_base9_l13_13990

def base9_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  d1 * 9 + d0

def base10_to_base9 (n : ℕ) : ℕ :=
  let d0 := n % 9
  let d1 := (n / 9) % 9
  let d2 := (n / 81) % 9
  d2 * 100 + d1 * 10 + d0

def sum_of_digits_base9 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  d2 + d1 + d0

theorem sum_of_digits_of_product_in_base9 :
  let n1 := base9_to_decimal 36
  let n2 := base9_to_decimal 21
  let product := n1 * n2
  let base9_product := base10_to_base9 product
  sum_of_digits_base9 base9_product = 19 :=
by
  sorry

end sum_of_digits_of_product_in_base9_l13_13990


namespace euler_line_of_isosceles_triangle_l13_13722

theorem euler_line_of_isosceles_triangle (A B : ℝ × ℝ) (hA : A = (2,0)) (hB : B = (0,4)) (C : ℝ × ℝ) (hC1 : dist A C = dist B C) :
  ∃ a b c : ℝ, a * (C.1 - 2) + b * (C.2 - 0) + c = 0 ∧ x - 2 * y + 3 = 0 :=
by
  sorry

end euler_line_of_isosceles_triangle_l13_13722


namespace jake_weight_l13_13129

theorem jake_weight {J S : ℝ} (h1 : J - 20 = 2 * S) (h2 : J + S = 224) : J = 156 :=
by
  sorry

end jake_weight_l13_13129


namespace arithmetic_sequence_general_formula_l13_13351

def f (x : ℝ) : ℝ := x^2 - 4 * x + 2

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_sequence_general_formula (a : ℕ → ℝ) (x : ℝ)
  (h_arith : arithmetic_seq a)
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n, a n = 2 * n - 4 ∨ a n = 4 - 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l13_13351


namespace ratio_of_areas_l13_13788

theorem ratio_of_areas (side_length : ℝ) (h : side_length = 6) :
  let area_triangle := (side_length^2 * Real.sqrt 3) / 4
  let area_square := side_length^2
  (area_triangle / area_square) = Real.sqrt 3 / 4 :=
by
  sorry

end ratio_of_areas_l13_13788


namespace faster_train_length_is_150_l13_13201

def speed_faster_train_kmph : ℝ := 72
def speed_slower_train_kmph : ℝ := 36
def time_seconds : ℝ := 15

noncomputable def length_faster_train : ℝ :=
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  relative_speed_mps * time_seconds

theorem faster_train_length_is_150 :
  length_faster_train = 150 := by
sorry

end faster_train_length_is_150_l13_13201


namespace football_team_total_players_l13_13694

theorem football_team_total_players (P : ℕ) (throwers : ℕ) (left_handed : ℕ) (right_handed : ℕ) :
  throwers = 49 →
  right_handed = 63 →
  left_handed = (1/3) * (P - 49) →
  (P - 49) - left_handed = (2/3) * (P - 49) →
  70 = P :=
by
  intros h_throwers h_right_handed h_left_handed h_remaining
  sorry

end football_team_total_players_l13_13694


namespace middle_group_frequency_l13_13215

theorem middle_group_frequency (capacity : ℕ) (n_rectangles : ℕ) (A_mid A_other : ℝ) 
  (h_capacity : capacity = 300)
  (h_rectangles : n_rectangles = 9)
  (h_areas : A_mid = 1 / 5 * A_other)
  (h_total_area : A_mid + A_other = 1) : 
  capacity * A_mid = 50 := by
  sorry

end middle_group_frequency_l13_13215


namespace jose_initial_caps_l13_13830

-- Definition of conditions and the problem
def jose_starting_caps : ℤ :=
  let final_caps := 9
  let caps_from_rebecca := 2
  final_caps - caps_from_rebecca

-- Lean theorem to state the required proof
theorem jose_initial_caps : jose_starting_caps = 7 := by
  -- skip proof
  sorry

end jose_initial_caps_l13_13830


namespace irrational_of_sqrt_3_l13_13730

theorem irrational_of_sqrt_3 :
  ¬ (∃ (a b : ℤ), b ≠ 0 ∧ ↑a / ↑b = Real.sqrt 3) :=
sorry

end irrational_of_sqrt_3_l13_13730


namespace cosine_sum_sine_half_sum_leq_l13_13060

variable {A B C : ℝ}

theorem cosine_sum_sine_half_sum_leq (h : A + B + C = Real.pi) :
  (Real.cos A + Real.cos B + Real.cos C) ≤ (Real.sin (A / 2) + Real.sin (B / 2) + Real.sin (C / 2)) :=
sorry

end cosine_sum_sine_half_sum_leq_l13_13060


namespace difference_of_squares_example_l13_13267

theorem difference_of_squares_example : 169^2 - 168^2 = 337 :=
by
  -- The proof steps using the difference of squares formula is omitted here.
  sorry

end difference_of_squares_example_l13_13267


namespace inequality_a3_b3_c3_l13_13724

theorem inequality_a3_b3_c3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^3 + b^3 + c^3 ≥ (1/3) * (a^2 + b^2 + c^2) * (a + b + c) := 
by 
  sorry

end inequality_a3_b3_c3_l13_13724


namespace min_red_chips_l13_13410

variable (w b r : ℕ)

theorem min_red_chips :
  (b ≥ w / 3) → (b ≤ r / 4) → (w + b ≥ 70) → r ≥ 72 :=
by
  sorry

end min_red_chips_l13_13410


namespace no_solutions_of_pairwise_distinct_l13_13752

theorem no_solutions_of_pairwise_distinct 
  (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∀ x : ℝ, ¬(x^3 - a * x^2 + b^3 = 0 ∧ x^3 - b * x^2 + c^3 = 0 ∧ x^3 - c * x^2 + a^3 = 0) :=
by
  -- Proof to be completed
  sorry

end no_solutions_of_pairwise_distinct_l13_13752


namespace opposite_of_neg_two_is_two_l13_13901

theorem opposite_of_neg_two_is_two (x : ℤ) (h : -2 + x = 0) : x = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l13_13901


namespace number_of_seeds_in_bucket_B_l13_13623

theorem number_of_seeds_in_bucket_B :
  ∃ (x : ℕ), 
    ∃ (y : ℕ), 
    ∃ (z : ℕ), 
      y = x + 10 ∧ 
      z = 30 ∧ 
      x + y + z = 100 ∧
      x = 30 :=
by {
  -- the proof is omitted.
  sorry
}

end number_of_seeds_in_bucket_B_l13_13623


namespace congruent_triangles_have_equal_perimeters_and_areas_l13_13661

-- Definitions based on the conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (A B C : ℝ) -- angles of the triangle

def congruent_triangles (Δ1 Δ2 : Triangle) : Prop :=
  Δ1.a = Δ2.a ∧ Δ1.b = Δ2.b ∧ Δ1.c = Δ2.c ∧
  Δ1.A = Δ2.A ∧ Δ1.B = Δ2.B ∧ Δ1.C = Δ2.C

-- perimeters and areas (assuming some function calc_perimeter and calc_area for simplicity)
def perimeter (Δ : Triangle) : ℝ := Δ.a + Δ.b + Δ.c
def area (Δ : Triangle) : ℝ := sorry -- implement area calculation, e.g., using Heron's formula

-- Statement to be proved
theorem congruent_triangles_have_equal_perimeters_and_areas (Δ1 Δ2 : Triangle) :
  congruent_triangles Δ1 Δ2 →
  perimeter Δ1 = perimeter Δ2 ∧ area Δ1 = area Δ2 :=
sorry

end congruent_triangles_have_equal_perimeters_and_areas_l13_13661


namespace y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l13_13772

def y : ℕ := 36 + 48 + 72 + 144 + 216 + 432 + 1296

theorem y_is_multiple_of_12 : y % 12 = 0 := by
  sorry

theorem y_is_multiple_of_3 : y % 3 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_4 : y % 4 = 0 := by
  have h := y_is_multiple_of_12
  sorry

theorem y_is_multiple_of_6 : y % 6 = 0 := by
  have h := y_is_multiple_of_12
  sorry

end y_is_multiple_of_12_y_is_multiple_of_3_y_is_multiple_of_4_y_is_multiple_of_6_l13_13772


namespace arithmetic_sequence_common_difference_l13_13003

theorem arithmetic_sequence_common_difference (d : ℚ) (a₁ : ℚ) (h : a₁ = -10)
  (h₁ : ∀ n ≥ 10, a₁ + (n - 1) * d > 0) :
  10 / 9 < d ∧ d ≤ 5 / 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l13_13003


namespace number_of_real_solutions_l13_13617

-- Definition of the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- The main theorem stating the number of solutions
theorem number_of_real_solutions : 
  ∃ (n : ℕ), n = 32 ∧ ∀ x : ℝ, equation x → -50 ≤ x ∧ x ≤ 50 :=
sorry

end number_of_real_solutions_l13_13617


namespace exactly_one_even_needs_assumption_l13_13638

open Nat

theorem exactly_one_even_needs_assumption 
  {a b c : ℕ} 
  (h : (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) ∧ (a % 2 = 0 → b % 2 = 1) ∧ (a % 2 = 0 → c % 2 = 1) ∧ (b % 2 = 0 → c % 2 = 1)) :
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) → (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) → (¬(a % 2 = 0 ∧ b % 2 = 0) ∧ ¬(b % 2 = 0 ∧ c % 2 = 0) ∧ ¬(a % 2 = 0 ∧ c % 2 = 0)) := 
by
  sorry

end exactly_one_even_needs_assumption_l13_13638


namespace simplify_fraction_l13_13779

theorem simplify_fraction :
  5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end simplify_fraction_l13_13779


namespace successive_discounts_final_price_l13_13649

noncomputable def initial_price : ℝ := 10000
noncomputable def discount1 : ℝ := 0.20
noncomputable def discount2 : ℝ := 0.10
noncomputable def discount3 : ℝ := 0.05

theorem successive_discounts_final_price :
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let final_selling_price := price_after_second_discount * (1 - discount3)
  final_selling_price = 6840 := by
  sorry

end successive_discounts_final_price_l13_13649


namespace Carla_is_2_years_older_than_Karen_l13_13314

-- Define the current age of Karen.
def Karen_age : ℕ := 2

-- Define the current age of Frank given that in 5 years he will be 36 years old.
def Frank_age : ℕ := 36 - 5

-- Define the current age of Ty given that Frank will be 3 times his age in 5 years.
def Ty_age : ℕ := 36 / 3

-- Define Carla's current age given that Ty is currently 4 years more than two times Carla's age.
def Carla_age : ℕ := (Ty_age - 4) / 2

-- Define the difference in age between Carla and Karen.
def Carla_Karen_age_diff : ℕ := Carla_age - Karen_age

-- The statement to be proven.
theorem Carla_is_2_years_older_than_Karen : Carla_Karen_age_diff = 2 := by
  -- The proof is not required, so we use sorry.
  sorry

end Carla_is_2_years_older_than_Karen_l13_13314


namespace cyclist_speed_l13_13070

/-- 
  Two cyclists A and B start at the same time from Newton to Kingston, a distance of 50 miles. 
  Cyclist A travels 5 mph slower than cyclist B. After reaching Kingston, B immediately turns 
  back and meets A 10 miles from Kingston. --/
theorem cyclist_speed (a b : ℕ) (h1 : b = a + 5) (h2 : 40 / a = 60 / b) : a = 10 :=
by
  sorry

end cyclist_speed_l13_13070


namespace solve_inequality_l13_13780

theorem solve_inequality {x : ℝ} : (x^2 - 9 * x + 18 ≤ 0) ↔ 3 ≤ x ∧ x ≤ 6 :=
by
sorry

end solve_inequality_l13_13780


namespace problem_l13_13372

theorem problem (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : 1 / x - 1 / y = -6) :
  x + y = -4 / 5 :=
sorry

end problem_l13_13372


namespace factorize_expression_l13_13291

theorem factorize_expression (x a : ℝ) : 4 * x - x * a^2 = x * (2 - a) * (2 + a) :=
by 
  sorry

end factorize_expression_l13_13291


namespace cubic_sum_l13_13950

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by {
  sorry
}

end cubic_sum_l13_13950


namespace joe_initial_money_l13_13110

theorem joe_initial_money (cost_notebook cost_book money_left : ℕ) 
                          (num_notebooks num_books : ℕ)
                          (h1 : cost_notebook = 4) 
                          (h2 : cost_book = 7)
                          (h3 : num_notebooks = 7) 
                          (h4 : num_books = 2) 
                          (h5 : money_left = 14) :
  (num_notebooks * cost_notebook + num_books * cost_book + money_left) = 56 := by
  sorry

end joe_initial_money_l13_13110


namespace number_of_intersections_is_four_l13_13324

def LineA (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0
def LineB (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def LineC (x y : ℝ) : Prop := x - y + 1 = 0
def LineD (x y : ℝ) : Prop := y - 2 = 0

def is_intersection (L1 L2 : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := L1 p.1 p.2 ∧ L2 p.1 p.2

theorem number_of_intersections_is_four :
  (∃ p1 : ℝ × ℝ, is_intersection LineA LineB p1) ∧
  (∃ p2 : ℝ × ℝ, is_intersection LineC LineD p2) ∧
  (∃ p3 : ℝ × ℝ, is_intersection LineA LineD p3) ∧
  (∃ p4 : ℝ × ℝ, is_intersection LineB LineD p4) ∧
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :=
by
  sorry

end number_of_intersections_is_four_l13_13324


namespace part1_max_value_part2_three_distinct_real_roots_l13_13157

def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem part1_max_value (m : ℝ) (h_max : ∀ x, f x m ≤ f 2 m) : m = 6 := by
  sorry

theorem part2_three_distinct_real_roots (a : ℝ) (h_m : (m = 6))
  (h_a : ∀ x₁ x₂ x₃ : ℝ, f x₁ m = a ∧ f x₂ m = a ∧ f x₃ m = a →
     x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) : 0 < a ∧ a < 32 := by
  sorry

end part1_max_value_part2_three_distinct_real_roots_l13_13157


namespace correct_conclusion_l13_13841

noncomputable def proof_problem (a x : ℝ) (x1 x2 : ℝ) :=
  (a * (x - 1) * (x - 3) + 2 > 0 ∧ x1 < x2 ∧ 
   (∀ x, a * (x - 1) * (x - 3) + 2 > 0 ↔ x < x1 ∨ x > x2)) →
  (x1 + x2 = 4 ∧ 3 < x1 * x2 ∧ x1 * x2 < 4 ∧ 
   (∀ x, ((3 * a + 2) * x^2 - 4 * a * x + a < 0) ↔ (1 / x2 < x ∧ x < 1 / x1)))

theorem correct_conclusion (a x x1 x2 : ℝ) : 
proof_problem a x x1 x2 :=
by 
  unfold proof_problem 
  sorry

end correct_conclusion_l13_13841


namespace cupcakes_per_package_calculation_l13_13510

noncomputable def sarah_total_cupcakes := 38
noncomputable def cupcakes_eaten_by_todd := 14
noncomputable def number_of_packages := 3
noncomputable def remaining_cupcakes := sarah_total_cupcakes - cupcakes_eaten_by_todd
noncomputable def cupcakes_per_package := remaining_cupcakes / number_of_packages

theorem cupcakes_per_package_calculation : cupcakes_per_package = 8 := by
  sorry

end cupcakes_per_package_calculation_l13_13510


namespace domain_of_function_l13_13381

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x ≠ 1) ↔ (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) := by
  sorry

end domain_of_function_l13_13381


namespace complex_square_identity_l13_13498

theorem complex_square_identity (i : ℂ) (h_i_squared : i^2 = -1) :
  i * (1 + i)^2 = -2 :=
by
  sorry

end complex_square_identity_l13_13498


namespace geom_sequence_a1_value_l13_13121

-- Define the conditions and the statement
theorem geom_sequence_a1_value (a_1 a_6 : ℚ) (a_3 a_4 : ℚ)
  (h1 : a_1 + a_6 = 11)
  (h2 : a_3 * a_4 = 32 / 9) :
  (a_1 = 32 / 3 ∨ a_1 = 1 / 3) :=
by 
-- We will prove the theorem here (skipped with sorry)
sorry

end geom_sequence_a1_value_l13_13121


namespace find_a_l13_13349

theorem find_a (a b c : ℤ) (h_vertex : ∀ x, (x - 2)*(x - 2) * a + 3 = a*x*x + b*x + c) 
  (h_point : (a*(3 - 2)*(3 -2) + 3 = 6)) : a = 3 :=
by
  sorry

end find_a_l13_13349


namespace right_triangle_median_to_hypotenuse_l13_13960

theorem right_triangle_median_to_hypotenuse 
    {DEF : Type} [MetricSpace DEF] 
    (D E F M : DEF) 
    (h_triangle : dist D E = 15 ∧ dist D F = 20 ∧ dist E F = 25) 
    (h_midpoint : dist D M = dist E M ∧ dist D E = 2 * dist D M ∧ dist E F * dist E F = dist E D * dist E D + dist D F * dist D F) :
    dist F M = 12.5 :=
by sorry

end right_triangle_median_to_hypotenuse_l13_13960


namespace area_of_triangle_XPQ_l13_13142
open Real

/-- Given a triangle XYZ with area 15 square units and points P, Q, R on sides XY, YZ, and ZX respectively,
where XP = 3, PY = 6, and triangles XPQ and quadrilateral PYRQ have equal areas, 
prove that the area of triangle XPQ is 5/3 square units. -/
theorem area_of_triangle_XPQ 
  (Area_XYZ : ℝ) (h1 : Area_XYZ = 15)
  (XP PY : ℝ) (h2 : XP = 3) (h3 : PY = 6)
  (h4 : ∃ (Area_XPQ : ℝ) (Area_PYRQ : ℝ), Area_XPQ = Area_PYRQ) :
  ∃ (Area_XPQ : ℝ), Area_XPQ = 5/3 :=
sorry

end area_of_triangle_XPQ_l13_13142


namespace mr_lee_gain_l13_13657

noncomputable def cost_price_1 (revenue : ℝ) (profit_percentage : ℝ) : ℝ :=
  revenue / (1 + profit_percentage)

noncomputable def cost_price_2 (revenue : ℝ) (loss_percentage : ℝ) : ℝ :=
  revenue / (1 - loss_percentage)

theorem mr_lee_gain
    (revenue : ℝ)
    (profit_percentage : ℝ)
    (loss_percentage : ℝ)
    (revenue_1 : ℝ := 1.44)
    (revenue_2 : ℝ := 1.44)
    (profit_percent : ℝ := 0.20)
    (loss_percent : ℝ := 0.10):
  let cost_1 := cost_price_1 revenue_1 profit_percent
  let cost_2 := cost_price_2 revenue_2 loss_percent
  let total_cost := cost_1 + cost_2
  let total_revenue := revenue_1 + revenue_2
  total_revenue - total_cost = 0.08 :=
by
  sorry

end mr_lee_gain_l13_13657


namespace range_of_S_l13_13191

variable {a b x : ℝ}
def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem range_of_S (h1 : ∀ x ∈ Set.Icc 0 1, |f x a b| ≤ 1) :
  ∃ l u, -2 ≤ l ∧ u ≤ 9 / 4 ∧ ∀ (S : ℝ), (S = (a + 1) * (b + 1)) → l ≤ S ∧ S ≤ u :=
by
  sorry

end range_of_S_l13_13191


namespace find_angle_l13_13338

theorem find_angle (r1 r2 : ℝ) (h_r1 : r1 = 1) (h_r2 : r2 = 2) 
(h_shaded : ∀ α : ℝ, 0 < α ∧ α < 2 * π → 
  (360 / 360 * pi * r1^2 + (α / (2 * π)) * pi * r2^2 - (α / (2 * π)) * pi * r1^2 = (1/3) * (pi * r2^2))) : 
  (∀ α : ℝ, 0 < α ∧ α < 2 * π ↔ 
  α = π / 3 ) :=
by
  sorry

end find_angle_l13_13338


namespace convert_base7_to_base2_l13_13007

-- Definitions and conditions
def base7_to_decimal (n : ℕ) : ℕ :=
  2 * 7^1 + 5 * 7^0

def decimal_to_binary (n : ℕ) : ℕ :=
  -- Reversing the binary conversion steps
  -- 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 19
  1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Proof problem
theorem convert_base7_to_base2 : decimal_to_binary (base7_to_decimal 25) = 10011 :=
by {
  sorry
}

end convert_base7_to_base2_l13_13007


namespace div_of_abs_values_l13_13061

theorem div_of_abs_values (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x < y) : x / y = -2 := 
by
  sorry

end div_of_abs_values_l13_13061


namespace focus_of_parabola_y_eq_9x2_plus_6_l13_13141

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, b + (1 / (4 * a)))

theorem focus_of_parabola_y_eq_9x2_plus_6 :
  focus_of_parabola 9 6 = (0, 217 / 36) :=
by
  sorry

end focus_of_parabola_y_eq_9x2_plus_6_l13_13141


namespace sum_first_six_terms_geometric_seq_l13_13161

theorem sum_first_six_terms_geometric_seq :
  let a := (1 : ℚ) / 4 
  let r := (1 : ℚ) / 4 
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (1365 / 16384 : ℚ) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  sorry

end sum_first_six_terms_geometric_seq_l13_13161


namespace remainder_modulo_l13_13207

theorem remainder_modulo (y : ℕ) (hy : 5 * y ≡ 1 [MOD 17]) : (7 + y) % 17 = 14 :=
sorry

end remainder_modulo_l13_13207


namespace pollen_scientific_notation_correct_l13_13156

def moss_flower_pollen_diameter := 0.0000084
def pollen_scientific_notation := 8.4 * 10^(-6)

theorem pollen_scientific_notation_correct :
  moss_flower_pollen_diameter = pollen_scientific_notation :=
by
  -- Proof skipped
  sorry

end pollen_scientific_notation_correct_l13_13156


namespace length_of_train_is_correct_l13_13330

-- Definitions based on conditions
def speed_kmh := 90
def time_sec := 10

-- Convert speed from km/hr to m/s
def speed_ms := speed_kmh * (1000 / 3600)

-- Calculate the length of the train
def length_of_train := speed_ms * time_sec

-- Theorem to prove the length of the train
theorem length_of_train_is_correct : length_of_train = 250 := by
  sorry

end length_of_train_is_correct_l13_13330


namespace find_m_of_hyperbola_l13_13478

noncomputable def eccen_of_hyperbola (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2) / (a^2))

theorem find_m_of_hyperbola :
  ∃ (m : ℝ), (m > 0) ∧ (eccen_of_hyperbola 2 m = Real.sqrt 3) ∧ (m = 2 * Real.sqrt 2) :=
by
  sorry

end find_m_of_hyperbola_l13_13478


namespace biography_percentage_increase_l13_13601

variable {T : ℝ}
variable (hT : T > 0 ∧ T ≤ 10000)
variable (B : ℝ := 0.20 * T)
variable (B' : ℝ := 0.32 * T)
variable (percentage_increase : ℝ := ((B' - B) / B) * 100)

theorem biography_percentage_increase :
  percentage_increase = 60 :=
by
  sorry

end biography_percentage_increase_l13_13601


namespace scientific_notation_correct_l13_13108

theorem scientific_notation_correct :
  1200000000 = 1.2 * 10^9 := 
by
  sorry

end scientific_notation_correct_l13_13108


namespace total_cost_of_tshirts_l13_13303

theorem total_cost_of_tshirts
  (White_packs : ℕ := 3) (Blue_packs : ℕ := 2) (Red_packs : ℕ := 4) (Green_packs : ℕ := 1) 
  (White_price_per_pack : ℝ := 12) (Blue_price_per_pack : ℝ := 8) (Red_price_per_pack : ℝ := 10) (Green_price_per_pack : ℝ := 6) 
  (White_discount : ℝ := 0.10) (Blue_discount : ℝ := 0.05) (Red_discount : ℝ := 0.15) (Green_discount : ℝ := 0.00) :
  White_packs * White_price_per_pack * (1 - White_discount) +
  Blue_packs * Blue_price_per_pack * (1 - Blue_discount) +
  Red_packs * Red_price_per_pack * (1 - Red_discount) +
  Green_packs * Green_price_per_pack * (1 - Green_discount) = 87.60 := by
    sorry

end total_cost_of_tshirts_l13_13303


namespace vector_perpendicular_sets_l13_13208

-- Define the problem in Lean
theorem vector_perpendicular_sets (x : ℝ) : 
  let a := (Real.sin x, Real.cos x)
  let b := (Real.sin x + Real.cos x, Real.sin x - Real.cos x)
  a.1 * b.1 + a.2 * b.2 = 0 ↔ ∃ (k : ℤ), x = k * (π / 2) + (π / 8) :=
sorry

end vector_perpendicular_sets_l13_13208


namespace k_9_pow_4_eq_81_l13_13978

theorem k_9_pow_4_eq_81 
  (h k : ℝ → ℝ) 
  (hk1 : ∀ (x : ℝ), x ≥ 1 → h (k x) = x^3) 
  (hk2 : ∀ (x : ℝ), x ≥ 1 → k (h x) = x^4) 
  (k81_eq_9 : k 81 = 9) :
  (k 9)^4 = 81 :=
by
  sorry

end k_9_pow_4_eq_81_l13_13978


namespace pie_filling_cans_l13_13462

-- Conditions
def price_per_pumpkin : ℕ := 3
def total_pumpkins : ℕ := 83
def total_revenue : ℕ := 96
def pumpkins_per_can : ℕ := 3

-- Definition
def cans_of_pie_filling (price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_revenue / price_per_pumpkin
  let pumpkins_remaining := total_pumpkins - pumpkins_sold
  pumpkins_remaining / pumpkins_per_can

-- Theorem
theorem pie_filling_cans : cans_of_pie_filling price_per_pumpkin total_pumpkins total_revenue pumpkins_per_can = 17 :=
  by sorry

end pie_filling_cans_l13_13462


namespace find_sides_from_diagonals_l13_13339

-- Define the number of diagonals D
def D : ℕ := 20

-- Define the equation relating the number of sides (n) to D
def diagonal_formula (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Statement to prove
theorem find_sides_from_diagonals (n : ℕ) (h : D = diagonal_formula n) : n = 8 :=
sorry

end find_sides_from_diagonals_l13_13339


namespace task_completion_time_l13_13893

noncomputable def work_time (A B C : ℝ) : ℝ := 1 / (A + B + C)

theorem task_completion_time (x y z : ℝ) (h1 : 8 * (x + y) = 1) (h2 : 6 * (x + z) = 1) (h3 : 4.8 * (y + z) = 1) :
    work_time x y z = 4 :=
by
  sorry

end task_completion_time_l13_13893


namespace find_m_from_permutation_l13_13089

theorem find_m_from_permutation (A : Nat → Nat → Nat) (m : Nat) (hA : A 11 m = 11 * 10 * 9 * 8 * 7 * 6 * 5) : m = 7 :=
sorry

end find_m_from_permutation_l13_13089


namespace sum_of_real_numbers_l13_13655

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l13_13655


namespace rhombus_area_l13_13614

theorem rhombus_area (a b : ℝ) (h : (a - 1) ^ 2 + Real.sqrt (b - 4) = 0) : (1 / 2) * a * b = 2 := by
  sorry

end rhombus_area_l13_13614


namespace total_stars_l13_13902

theorem total_stars (students stars_per_student : ℕ) (h_students : students = 124) (h_stars_per_student : stars_per_student = 3) : students * stars_per_student = 372 := by
  sorry

end total_stars_l13_13902


namespace travel_time_difference_in_minutes_l13_13814

/-
A bus travels at an average speed of 40 miles per hour.
We need to prove that the difference in travel time between a 360-mile trip and a 400-mile trip equals 60 minutes.
-/

theorem travel_time_difference_in_minutes 
  (speed : ℝ) (distance1 distance2 : ℝ) 
  (h1 : speed = 40) 
  (h2 : distance1 = 360) 
  (h3 : distance2 = 400) :
  (distance2 / speed - distance1 / speed) * 60 = 60 := by
  sorry

end travel_time_difference_in_minutes_l13_13814


namespace find_a0_find_a2_find_sum_a1_a2_a3_a4_l13_13546

lemma problem_conditions (x : ℝ) : 
  (x - 2)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 :=
sorry

theorem find_a0 :
  a_0 = 16 :=
sorry

theorem find_a2 :
  a_2 = 24 :=
sorry

theorem find_sum_a1_a2_a3_a4 :
  a_1 + a_2 + a_3 + a_4 = -15 :=
sorry

end find_a0_find_a2_find_sum_a1_a2_a3_a4_l13_13546


namespace average_weight_of_boys_l13_13742

theorem average_weight_of_boys (n1 n2 : ℕ) (w1 w2 : ℚ) 
  (weight_avg_22_boys : w1 = 50.25) 
  (weight_avg_8_boys : w2 = 45.15) 
  (count_22_boys : n1 = 22) 
  (count_8_boys : n2 = 8) 
  : ((n1 * w1 + n2 * w2) / (n1 + n2) : ℚ) = 48.89 :=
by
  sorry

end average_weight_of_boys_l13_13742


namespace math_problem_l13_13833

theorem math_problem (x y : ℤ) (h1 : x = 12) (h2 : y = 18) : (x - y) * ((x + y) ^ 2) = -5400 := by
  sorry

end math_problem_l13_13833


namespace fraction_of_butterflies_flew_away_l13_13495

theorem fraction_of_butterflies_flew_away (original_butterflies : ℕ) (left_butterflies : ℕ) (h1 : original_butterflies = 9) (h2 : left_butterflies = 6) : (original_butterflies - left_butterflies) / original_butterflies = 1 / 3 :=
by
  sorry

end fraction_of_butterflies_flew_away_l13_13495


namespace incorrect_expression_l13_13687

theorem incorrect_expression (x y : ℝ) (h : x > y) : ¬ (1 - 3*x > 1 - 3*y) :=
sorry

end incorrect_expression_l13_13687


namespace desiree_age_l13_13654

variables (D C : ℕ)
axiom condition1 : D = 2 * C
axiom condition2 : D + 30 = (2 * (C + 30)) / 3 + 14

theorem desiree_age : D = 6 :=
by
  sorry

end desiree_age_l13_13654


namespace opposite_of_negative_2023_l13_13753

theorem opposite_of_negative_2023 : -(-2023) = 2023 :=
by
  sorry

end opposite_of_negative_2023_l13_13753


namespace max_abs_ax_plus_b_l13_13673

theorem max_abs_ax_plus_b (a b c : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, |x| ≤ 1 → |a * x + b| ≤ 2 :=
by
  sorry

end max_abs_ax_plus_b_l13_13673


namespace length_of_rectangular_garden_l13_13356

theorem length_of_rectangular_garden (P B : ℝ) (h₁ : P = 1200) (h₂ : B = 240) :
  ∃ L : ℝ, P = 2 * (L + B) ∧ L = 360 :=
by
  sorry

end length_of_rectangular_garden_l13_13356


namespace abs_condition_sufficient_not_necessary_l13_13619

theorem abs_condition_sufficient_not_necessary:
  (∀ x : ℝ, (-2 < x ∧ x < 3) → (-1 < x ∧ x < 3)) :=
by
  sorry

end abs_condition_sufficient_not_necessary_l13_13619


namespace weight_of_one_apple_l13_13847

-- Conditions
def total_weight_of_bag_with_apples : ℝ := 1.82
def weight_of_empty_bag : ℝ := 0.5
def number_of_apples : ℕ := 6

-- The proposition to prove: the weight of one apple
theorem weight_of_one_apple : (total_weight_of_bag_with_apples - weight_of_empty_bag) / number_of_apples = 0.22 := 
by
  sorry

end weight_of_one_apple_l13_13847


namespace calc_7_op_4_minus_4_op_7_l13_13541

def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

theorem calc_7_op_4_minus_4_op_7 : (op 7 4) - (op 4 7) = -12 := by
  sorry

end calc_7_op_4_minus_4_op_7_l13_13541


namespace total_children_count_l13_13823

theorem total_children_count (boys girls : ℕ) (hb : boys = 40) (hg : girls = 77) : boys + girls = 117 := by
  sorry

end total_children_count_l13_13823


namespace fewest_presses_to_original_l13_13319

theorem fewest_presses_to_original (x : ℝ) (hx : x = 16) (f : ℝ → ℝ)
    (hf : ∀ y : ℝ, f y = 1 / y) : (f (f x)) = x :=
by
  sorry

end fewest_presses_to_original_l13_13319


namespace total_people_in_class_l13_13099

def likes_both (n : ℕ) := n = 5
def likes_only_baseball (n : ℕ) := n = 2
def likes_only_football (n : ℕ) := n = 3
def likes_neither (n : ℕ) := n = 6

theorem total_people_in_class
  (h1 : likes_both n1)
  (h2 : likes_only_baseball n2)
  (h3 : likes_only_football n3)
  (h4 : likes_neither n4) :
  n1 + n2 + n3 + n4 = 16 :=
by 
  sorry

end total_people_in_class_l13_13099


namespace John_bought_new_socks_l13_13148

theorem John_bought_new_socks (initial_socks : ℕ) (thrown_away_socks : ℕ) (current_socks : ℕ) :
    initial_socks = 33 → thrown_away_socks = 19 → current_socks = 27 → 
    current_socks = (initial_socks - thrown_away_socks) + 13 :=
by
  sorry

end John_bought_new_socks_l13_13148


namespace problem_f_x_sum_neg_l13_13411

open Function

-- Definitions for monotonic decreasing and odd properties of the function
def isOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def isMonotonicallyDecreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → f y ≤ f x

-- The main theorem to prove
theorem problem_f_x_sum_neg
  (f : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_monotone : isMonotonicallyDecreasing f)
  (x₁ x₂ x₃ : ℝ)
  (h₁ : x₁ + x₂ > 0)
  (h₂ : x₂ + x₃ > 0)
  (h₃ : x₃ + x₁ > 0) :
  f x₁ + f x₂ + f x₃ < 0 :=
by
  sorry

end problem_f_x_sum_neg_l13_13411


namespace points_per_round_l13_13453

def total_points : ℕ := 78
def num_rounds : ℕ := 26

theorem points_per_round : total_points / num_rounds = 3 := by
  sorry

end points_per_round_l13_13453


namespace x_coord_sum_l13_13147

noncomputable def sum_x_coordinates (x : ℕ) : Prop :=
  (0 ≤ x ∧ x < 20) ∧ (∃ y, y ≡ 7 * x + 3 [MOD 20] ∧ y ≡ 13 * x + 18 [MOD 20])

theorem x_coord_sum : ∃ (x : ℕ), sum_x_coordinates x ∧ x = 15 := by 
  sorry

end x_coord_sum_l13_13147


namespace trigonometric_identity_l13_13843

theorem trigonometric_identity (α : ℝ) (h : Real.cos α + Real.sin α = 2 / 3) :
  (Real.sqrt 2 * Real.sin (2 * α - Real.pi / 4) + 1) / (1 + Real.tan α) = - 5 / 9 :=
sorry

end trigonometric_identity_l13_13843


namespace circumference_circle_l13_13716

theorem circumference_circle {d r : ℝ} (h1 : ∀ (d r : ℝ), d = 2 * r) : 
  ∃ C : ℝ, C = π * d ∨ C = 2 * π * r :=
by {
  sorry
}

end circumference_circle_l13_13716


namespace fibonacci_p_arithmetic_periodic_l13_13094

-- Define p-arithmetic system and its properties
def p_arithmetic (p : ℕ) : Prop :=
  ∀ (a : ℤ), a ≠ 0 → a^(p-1) = 1

-- Define extraction of sqrt(5)
def sqrt5_extractable (p : ℕ) : Prop :=
  ∃ (r : ℝ), r^2 = 5

-- Define Fibonacci sequence in p-arithmetic
def fibonacci_p_arithmetic (p : ℕ) (v : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, v (n+2) = v (n+1) + v n

-- Main Theorem
theorem fibonacci_p_arithmetic_periodic (p : ℕ) (v : ℕ → ℤ) :
  p_arithmetic p →
  sqrt5_extractable p →
  fibonacci_p_arithmetic p v →
  (∀ k : ℕ, v (k + p) = v k) :=
by
  intros _ _ _
  sorry

end fibonacci_p_arithmetic_periodic_l13_13094


namespace fg_neg_two_l13_13525

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x + 3

theorem fg_neg_two : f (g (-2)) = 2 := by
  sorry

end fg_neg_two_l13_13525


namespace totalPawnsLeft_l13_13556

def sophiaInitialPawns := 8
def chloeInitialPawns := 8
def sophiaLostPawns := 5
def chloeLostPawns := 1

theorem totalPawnsLeft : (sophiaInitialPawns - sophiaLostPawns) + (chloeInitialPawns - chloeLostPawns) = 10 := by
  sorry

end totalPawnsLeft_l13_13556


namespace arcsin_sqrt_three_over_two_l13_13185

theorem arcsin_sqrt_three_over_two : 
  ∃ θ, θ = Real.arcsin (Real.sqrt 3 / 2) ∧ θ = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_three_over_two_l13_13185


namespace triangle_side_lengths_values_l13_13569

theorem triangle_side_lengths_values :
  ∃ (m_values : Finset ℕ), m_values = {m ∈ Finset.range 750 | m ≥ 4} ∧ m_values.card = 746 :=
by
  sorry

end triangle_side_lengths_values_l13_13569


namespace inscribed_rectangle_sides_l13_13278

theorem inscribed_rectangle_sides {a b c : ℕ} (h₀ : a = 3) (h₁ : b = 4) (h₂ : c = 5) (ratio : ℚ) (h_ratio : ratio = 1 / 3) :
  ∃ (x y : ℚ), x = 20 / 29 ∧ y = 60 / 29 ∧ x = ratio * y :=
by
  sorry

end inscribed_rectangle_sides_l13_13278


namespace equal_semi_circles_radius_l13_13542

-- Define the segments and semicircles given in the problem as conditions.
def segment1 : ℝ := 12
def segment2 : ℝ := 22
def segment3 : ℝ := 22
def segment4 : ℝ := 16
def segment5 : ℝ := 22

def total_horizontal_path1 (r : ℝ) : ℝ := 2*r + segment1 + 2*r + segment1 + 2*r
def total_horizontal_path2 (r : ℝ) : ℝ := segment2 + 2*r + segment4 + 2*r + segment5

-- The theorem that proves the radius is 18.
theorem equal_semi_circles_radius : ∃ r : ℝ, total_horizontal_path1 r = total_horizontal_path2 r ∧ r = 18 := by
  use 18
  simp [total_horizontal_path1, total_horizontal_path2, segment1, segment2, segment3, segment4, segment5]
  sorry

end equal_semi_circles_radius_l13_13542


namespace element_with_36_36_percentage_is_O_l13_13358

-- Define the chemical formula N2O and atomic masses
def chemical_formula : String := "N2O"
def atomic_mass_N : Float := 14.01
def atomic_mass_O : Float := 16.00

-- Define the molar mass of N2O
def molar_mass_N2O : Float := (2 * atomic_mass_N) + (1 * atomic_mass_O)

-- Mass of nitrogen in N2O
def mass_N_in_N2O : Float := 2 * atomic_mass_N

-- Mass of oxygen in N2O
def mass_O_in_N2O : Float := 1 * atomic_mass_O

-- Mass percentages
def mass_percentage_N : Float := (mass_N_in_N2O / molar_mass_N2O) * 100
def mass_percentage_O : Float := (mass_O_in_N2O / molar_mass_N2O) * 100

-- Prove that the element with a mass percentage of 36.36% is oxygen
theorem element_with_36_36_percentage_is_O : mass_percentage_O = 36.36 := sorry

end element_with_36_36_percentage_is_O_l13_13358


namespace guy_has_sixty_cents_l13_13695

-- Definitions for the problem conditions
def lance_has (lance_cents : ℕ) : Prop := lance_cents = 70
def margaret_has (margaret_cents : ℕ) : Prop := margaret_cents = 75
def bill_has (bill_cents : ℕ) : Prop := bill_cents = 60
def total_has (total_cents : ℕ) : Prop := total_cents = 265

-- Problem Statement in Lean format
theorem guy_has_sixty_cents (lance_cents margaret_cents bill_cents total_cents guy_cents : ℕ) 
    (h_lance : lance_has lance_cents)
    (h_margaret : margaret_has margaret_cents)
    (h_bill : bill_has bill_cents)
    (h_total : total_has total_cents) :
    guy_cents = total_cents - (lance_cents + margaret_cents + bill_cents) → guy_cents = 60 :=
by
  intros h
  simp [lance_has, margaret_has, bill_has, total_has] at *
  rw [h_lance, h_margaret, h_bill, h_total] at h
  exact h

end guy_has_sixty_cents_l13_13695


namespace find_m_plus_n_l13_13301

theorem find_m_plus_n (m n : ℤ) 
  (H1 : (x^3 + m*x + n) * (x^2 - 3*x + 1) ≠ 1 * x^2 + 1 * x^3) 
  (H2 : (x^3 + m*x + n) * (x^2 - 3*x + 1) ≠ 1 * x^2 + 1 * x^3) : 
  m + n = -4 := 
by
  sorry

end find_m_plus_n_l13_13301


namespace compute_roots_sum_l13_13810

def roots_quadratic_eq_a_b (a b : ℂ) : Prop :=
  a^2 - 6 * a + 8 = 0 ∧ b^2 - 6 * b + 8 = 0

theorem compute_roots_sum (a b : ℂ) (ha : roots_quadratic_eq_a_b a b) :
  a^5 + a^3 * b^3 + b^5 = -568 := by
  sorry

end compute_roots_sum_l13_13810


namespace factorial_not_div_by_two_pow_l13_13540

theorem factorial_not_div_by_two_pow (n : ℕ) : ¬ (2^n ∣ n!) :=
sorry

end factorial_not_div_by_two_pow_l13_13540


namespace product_of_slopes_l13_13281

theorem product_of_slopes (p : ℝ) (hp : 0 < p) :
  let T := (p, 0)
  let parabola := fun x y => y^2 = 2*p*x
  let line := fun x y => y = x - p
  -- Define intersection points A and B on the parabola satisfying the line equation
  ∃ A B : ℝ × ℝ, 
  parabola A.1 A.2 ∧ line A.1 A.2 ∧
  parabola B.1 B.2 ∧ line B.1 B.2 ∧
  -- O is the origin
  let O := (0, 0)
  -- define slope function
  let slope (P Q : ℝ × ℝ) := (Q.2 - P.2) / (Q.1 - P.1)
  -- slopes of OA and OB
  let k_OA := slope O A
  let k_OB := slope O B
  -- product of slopes
  k_OA * k_OB = -2 := sorry

end product_of_slopes_l13_13281


namespace tickets_distribution_l13_13516

theorem tickets_distribution (people tickets : ℕ) (h_people : people = 9) (h_tickets : tickets = 24)
  (h_each_gets_at_least_one : ∀ (i : ℕ), i < people → (1 : ℕ) ≤ 1) :
  ∃ (count : ℕ), count ≥ 4 ∧ ∃ (f : ℕ → ℕ), (∀ i, i < people → 1 ≤ f i ∧ f i ≤ tickets) ∧ (∀ i < people, ∃ j < people, f i = f j) :=
  sorry

end tickets_distribution_l13_13516


namespace charlie_rope_first_post_l13_13517

theorem charlie_rope_first_post (X : ℕ) (h : X + 20 + 14 + 12 = 70) : X = 24 :=
sorry

end charlie_rope_first_post_l13_13517


namespace MiaShots_l13_13001

theorem MiaShots (shots_game1_to_5 : ℕ) (total_shots_game1_to_5 : ℕ) (initial_avg : ℕ → ℕ → Prop)
  (shots_game6 : ℕ) (new_avg_shots : ℕ → ℕ → Prop) (total_shots : ℕ) (new_avg : ℕ): 
  shots_game1_to_5 = 20 →
  total_shots_game1_to_5 = 50 →
  initial_avg shots_game1_to_5 total_shots_game1_to_5 →
  shots_game6 = 15 →
  new_avg_shots 29 65 →
  total_shots = total_shots_game1_to_5 + shots_game6 →
  new_avg = 45 →
  (∃ shots_made_game6 : ℕ, shots_made_game6 = 29 - shots_game1_to_5 ∧ shots_made_game6 = 9) :=
by
  sorry

end MiaShots_l13_13001


namespace min_value_of_expression_l13_13159

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  (1 / x + 4 / y) ≥ 9 :=
by
  sorry

end min_value_of_expression_l13_13159


namespace construct_triangle_l13_13315

variables (a : ℝ) (α : ℝ) (d : ℝ)

-- Helper definitions
def is_triangle_valid (a α d : ℝ) : Prop := sorry

-- The theorem to be proven
theorem construct_triangle (a α d : ℝ) : is_triangle_valid a α d :=
sorry

end construct_triangle_l13_13315


namespace exponentiation_addition_zero_l13_13151

theorem exponentiation_addition_zero : (-2)^(3^2) + 2^(3^2) = 0 := 
by 
  -- proof goes here
  sorry

end exponentiation_addition_zero_l13_13151


namespace jack_sugar_usage_l13_13735

theorem jack_sugar_usage (initial_sugar bought_sugar final_sugar x : ℕ) 
  (h1 : initial_sugar = 65) 
  (h2 : bought_sugar = 50) 
  (h3 : final_sugar = 97) 
  (h4 : final_sugar = initial_sugar - x + bought_sugar) : 
  x = 18 := 
by 
  sorry

end jack_sugar_usage_l13_13735


namespace sector_central_angle_l13_13259

theorem sector_central_angle (r l α : ℝ) (h1 : 2 * r + l = 6) (h2 : 1/2 * l * r = 2) :
  α = l / r → (α = 1 ∨ α = 4) :=
by
  sorry

end sector_central_angle_l13_13259


namespace ellipse_eq_range_m_l13_13245

theorem ellipse_eq_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (m - 1) + y^2 / (3 - m) = 1)) ↔ (1 < m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end ellipse_eq_range_m_l13_13245


namespace valid_license_plates_l13_13664

-- Define the number of vowels and the total alphabet letters.
def num_vowels : ℕ := 5
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the total number of valid license plates in Eldoria.
theorem valid_license_plates : num_vowels * num_letters * num_digits^3 = 130000 := by
  sorry

end valid_license_plates_l13_13664


namespace a_divisible_by_11_iff_b_divisible_by_11_l13_13268

-- Define the relevant functions
def a (n : ℕ) : ℕ := n^5 + 5^n
def b (n : ℕ) : ℕ := n^5 * 5^n + 1

-- State that for a positive integer n, a(n) is divisible by 11 if and only if b(n) is also divisible by 11
theorem a_divisible_by_11_iff_b_divisible_by_11 (n : ℕ) (hn : 0 < n) : 
  (a n % 11 = 0) ↔ (b n % 11 = 0) :=
sorry

end a_divisible_by_11_iff_b_divisible_by_11_l13_13268


namespace shortest_chord_through_M_l13_13750

noncomputable def point_M : ℝ × ℝ := (1, 0)
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y = 0

theorem shortest_chord_through_M :
  (∀ x y : ℝ, circle_C x y → x + y - 1 = 0) :=
by
  sorry

end shortest_chord_through_M_l13_13750


namespace initial_books_in_bin_l13_13711

variable (X : ℕ)

theorem initial_books_in_bin (h1 : X - 3 + 10 = 11) : X = 4 :=
by
  sorry

end initial_books_in_bin_l13_13711


namespace inequality_solution_l13_13717

theorem inequality_solution (x : ℝ) : 
  -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 3 ≤ x ∧ x < 4 → 
  (x + 6 ≥ 0) ∧ (x + 1 > 0) ∧ (5 - x > 0) ∧ (x ≠ 0) ∧ (x ≠ 1) ∧ (x ≠ 4) ∧
  ( (x - 3) / ((x - 1) * (4 - x)) ≥ 0 ) :=
sorry

end inequality_solution_l13_13717


namespace project_completion_time_l13_13512

theorem project_completion_time (initial_workers : ℕ) (initial_days : ℕ) (extra_workers : ℕ) (extra_days : ℕ) : 
  initial_workers = 10 →
  initial_days = 15 →
  extra_workers = 5 →
  extra_days = 5 →
  total_days = 6 := by
  sorry

end project_completion_time_l13_13512


namespace curve_B_is_not_good_l13_13970

-- Define the points A and B
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define the condition for being a "good curve"
def is_good_curve (C : ℝ × ℝ → Prop) : Prop :=
  ∃ M : ℝ × ℝ, C M ∧ abs (dist M A - dist M B) = 8

-- Define the curves
def curve_A (p : ℝ × ℝ) : Prop := p.1 + p.2 = 5
def curve_B (p : ℝ × ℝ) : Prop := p.1 ^ 2 + p.2 ^ 2 = 9
def curve_C (p : ℝ × ℝ) : Prop := (p.1 ^ 2) / 25 + (p.2 ^ 2) / 9 = 1
def curve_D (p : ℝ × ℝ) : Prop := p.1 ^ 2 = 16 * p.2

-- Prove that curve_B is not a "good curve"
theorem curve_B_is_not_good : ¬ is_good_curve curve_B := by
  sorry

end curve_B_is_not_good_l13_13970


namespace find_k_l13_13309

theorem find_k (k : ℝ) (h : (3 : ℝ)^2 - k * (3 : ℝ) - 6 = 0) : k = 1 :=
by
  sorry

end find_k_l13_13309


namespace carla_drive_distance_l13_13545

theorem carla_drive_distance
    (d1 d3 : ℕ) (gpm : ℕ) (gas_price total_cost : ℕ) 
    (x : ℕ)
    (hx : 2 * gas_price = 1)
    (gallon_cost : ℕ := total_cost / gas_price)
    (total_distance   : ℕ := gallon_cost * gpm)
    (total_errand_distance : ℕ := d1 + x + d3 + 2 * x)
    (h_distance : total_distance = total_errand_distance) :
  x = 10 :=
by
  -- begin
  -- proof construction
  sorry

end carla_drive_distance_l13_13545


namespace coefficient_of_ab_is_correct_l13_13472

noncomputable def a : ℝ := 15 / 7
noncomputable def b : ℝ := 15 / 2
noncomputable def ab : ℝ := 674.9999999999999
noncomputable def coeff_ab := ab / (a * b)

theorem coefficient_of_ab_is_correct :
  coeff_ab = 674.9999999999999 / ((15 * 15) / (7 * 2)) := sorry

end coefficient_of_ab_is_correct_l13_13472


namespace differentiable_function_zero_l13_13868

theorem differentiable_function_zero
    (f : ℝ → ℝ)
    (h_diff : Differentiable ℝ f)
    (h_zero : f 0 = 0)
    (h_ineq : ∀ x : ℝ, 0 < |f x| ∧ |f x| < 1/2 → |deriv f x| ≤ |f x * Real.log (|f x|)|) :
    ∀ x : ℝ, f x = 0 :=
by
  sorry

end differentiable_function_zero_l13_13868


namespace solve_system_equations_l13_13101

theorem solve_system_equations (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
    ∃ x y z : ℝ,  
      (x * y = (z - a) ^ 2) ∧
      (y * z = (x - b) ^ 2) ∧
      (z * x = (y - c) ^ 2) ∧
      x = ((b ^ 2 - a * c) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) ∧
      y = ((c ^ 2 - a * b) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) ∧
      z = ((a ^ 2 - b * c) ^ 2) / (a ^ 3 + b ^ 3 + c ^ 3 - 3 * a * b * c) :=
sorry

end solve_system_equations_l13_13101


namespace log_product_max_l13_13611

open Real

theorem log_product_max (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : log x + log y = 4) : log x * log y ≤ 4 := 
by
  sorry

end log_product_max_l13_13611


namespace train_cross_first_platform_l13_13864

noncomputable def time_to_cross_first_platform (L_t L_p1 L_p2 t2 : ℕ) : ℕ :=
  (L_t + L_p1) / ((L_t + L_p2) / t2)

theorem train_cross_first_platform :
  time_to_cross_first_platform 100 200 300 20 = 15 :=
by
  sorry

end train_cross_first_platform_l13_13864


namespace integer_roots_if_q_positive_no_integer_roots_if_q_negative_l13_13362

theorem integer_roots_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1) :=
sorry

theorem no_integer_roots_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬ ((∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1)) :=
sorry

end integer_roots_if_q_positive_no_integer_roots_if_q_negative_l13_13362


namespace base_7_digits_956_l13_13005

theorem base_7_digits_956 : ∃ n : ℕ, ∀ k : ℕ, 956 < 7^k → n = k ∧ 956 ≥ 7^(k-1) := sorry

end base_7_digits_956_l13_13005


namespace num_five_ruble_coins_l13_13013

theorem num_five_ruble_coins (total_coins a b c k : ℕ) (h1 : total_coins = 25)
    (h2 : a = 25 - 19) (h3 : b = 25 - 20) (h4 : c = 25 - 16)
    (h5 : k = total_coins - (a + b + c)) : k = 5 :=
by
  rw [h1, h2, h3, h4] at h5
  simp at h5
  exact h5

end num_five_ruble_coins_l13_13013


namespace ratio_of_unit_prices_l13_13559

def volume_y (v : ℝ) : ℝ := v
def price_y (p : ℝ) : ℝ := p
def volume_x (v : ℝ) : ℝ := 1.3 * v
def price_x (p : ℝ) : ℝ := 0.8 * p

theorem ratio_of_unit_prices (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (0.8 * p / (1.3 * v)) / (p / v) = 8 / 13 :=
by 
  sorry

end ratio_of_unit_prices_l13_13559


namespace largest_integral_x_l13_13710

theorem largest_integral_x (x : ℤ) (h1 : 1/4 < (x:ℝ)/6) (h2 : (x:ℝ)/6 < 7/9) : x ≤ 4 :=
by
  -- This is where the proof would go
  sorry

end largest_integral_x_l13_13710


namespace sequence_solution_l13_13235

theorem sequence_solution 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n / (2 + a n))
  (h2 : a 1 = 1) :
  ∀ n, a n = 1 / (2^n - 1) :=
sorry

end sequence_solution_l13_13235


namespace chef_additional_wings_l13_13607

theorem chef_additional_wings
    (n : ℕ) (w_initial : ℕ) (w_per_friend : ℕ) (w_additional : ℕ)
    (h1 : n = 4)
    (h2 : w_initial = 9)
    (h3 : w_per_friend = 4)
    (h4 : w_additional = 7) :
    n * w_per_friend - w_initial = w_additional :=
by
  sorry

end chef_additional_wings_l13_13607


namespace find_non_negative_integers_l13_13257

def has_exactly_two_distinct_solutions (a : ℕ) (m : ℕ) : Prop :=
  ∃ (x₁ x₂ : ℕ), (x₁ < m) ∧ (x₂ < m) ∧ (x₁ ≠ x₂) ∧ (x₁^2 + a) % m = 0 ∧ (x₂^2 + a) % m = 0

theorem find_non_negative_integers (a : ℕ) (m : ℕ := 2007) : 
  a < m ∧ has_exactly_two_distinct_solutions a m ↔ a = 446 ∨ a = 1115 ∨ a = 1784 :=
sorry

end find_non_negative_integers_l13_13257


namespace find_root_product_l13_13286

theorem find_root_product :
  (∃ r s t : ℝ, (∀ x : ℝ, (x - r) * (x - s) * (x - t) = x^3 - 15 * x^2 + 26 * x - 8) ∧
  (1 + r) * (1 + s) * (1 + t) = 50) :=
sorry

end find_root_product_l13_13286


namespace mean_first_second_fifth_sixth_diff_l13_13041

def six_numbers_arithmetic_mean_condition (a1 a2 a3 a4 a5 a6 A : ℝ) :=
  (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

def mean_first_four_numbers (a1 a2 a3 a4 A : ℝ) :=
  (a1 + a2 + a3 + a4) / 4 = A + 10

def mean_last_four_numbers (a3 a4 a5 a6 A : ℝ) :=
  (a3 + a4 + a5 + a6) / 4 = A - 7

theorem mean_first_second_fifth_sixth_diff (a1 a2 a3 a4 a5 a6 A : ℝ) :
  six_numbers_arithmetic_mean_condition a1 a2 a3 a4 a5 a6 A →
  mean_first_four_numbers a1 a2 a3 a4 A →
  mean_last_four_numbers a3 a4 a5 a6 A →
  ((a1 + a2 + a5 + a6) / 4) = A - 3 :=
by
  intros h1 h2 h3
  sorry

end mean_first_second_fifth_sixth_diff_l13_13041


namespace proof_multiple_l13_13887

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem proof_multiple (a b : ℕ) 
  (h₁ : is_multiple a 5) 
  (h₂ : is_multiple b 10) : 
  is_multiple b 5 ∧ 
  is_multiple (a + b) 5 ∧ 
  is_multiple (a + b) 2 :=
by
  sorry

end proof_multiple_l13_13887


namespace arith_sqrt_abs_neg_nine_l13_13079

theorem arith_sqrt_abs_neg_nine : Real.sqrt (abs (-9)) = 3 := by
  sorry

end arith_sqrt_abs_neg_nine_l13_13079


namespace a_share_correct_l13_13459

-- Investment periods for each individual in months
def investment_a := 12
def investment_b := 6
def investment_c := 4
def investment_d := 9
def investment_e := 7
def investment_f := 5

-- Investment multiplier for each individual
def multiplier_b := 2
def multiplier_c := 3
def multiplier_d := 4
def multiplier_e := 5
def multiplier_f := 6

-- Total annual gain
def total_gain := 38400

-- Calculate individual shares
def share_a (x : ℝ) := x * investment_a
def share_b (x : ℝ) := multiplier_b * x * investment_b
def share_c (x : ℝ) := multiplier_c * x * investment_c
def share_d (x : ℝ) := multiplier_d * x * investment_d
def share_e (x : ℝ) := multiplier_e * x * investment_e
def share_f (x : ℝ) := multiplier_f * x * investment_f

-- Calculate total investment
def total_investment (x : ℝ) :=
  share_a x + share_b x + share_c x + share_d x + share_e x + share_f x

-- Prove that a's share of the annual gain is Rs. 3360
theorem a_share_correct : 
  ∃ x : ℝ, (12 * x / total_investment x) * total_gain = 3360 := 
sorry

end a_share_correct_l13_13459


namespace find_p_l13_13739

noncomputable def f (p : ℝ) : ℝ := 2 * p - 20

theorem find_p : (f ∘ f ∘ f) p = 6 → p = 18.25 := by
  sorry

end find_p_l13_13739


namespace two_cubic_meters_to_cubic_feet_l13_13869

theorem two_cubic_meters_to_cubic_feet :
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  2 * cubic_meter_to_cubic_feet = 70.6294 :=
by
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  have h : 2 * cubic_meter_to_cubic_feet = 70.6294 := sorry
  exact h

end two_cubic_meters_to_cubic_feet_l13_13869


namespace runs_in_last_match_l13_13292

-- Definitions based on the conditions
def initial_bowling_average : ℝ := 12.4
def wickets_last_match : ℕ := 7
def decrease_average : ℝ := 0.4
def new_average : ℝ := initial_bowling_average - decrease_average
def approximate_wickets_before : ℕ := 145

-- The Lean statement of the problem
theorem runs_in_last_match (R : ℝ) :
  ((initial_bowling_average * approximate_wickets_before + R) / 
   (approximate_wickets_before + wickets_last_match) = new_average) →
   R = 28 :=
by
  sorry

end runs_in_last_match_l13_13292


namespace spade_evaluation_l13_13230

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_evaluation : spade 5 (spade 6 7 + 2) = -96 := by
  sorry

end spade_evaluation_l13_13230


namespace four_digit_numbers_l13_13775

theorem four_digit_numbers (n : ℕ) :
    (∃ a b c d : ℕ, 
        n = a * 1000 + b * 100 + c * 10 + d 
        ∧ 4 ≤ a ∧ a ≤ 9 
        ∧ 1 ≤ b ∧ b ≤ 9 
        ∧ 1 ≤ c ∧ c ≤ 9 
        ∧ 0 ≤ d ∧ d ≤ 9 
        ∧ b * c > 8) → n ∈ {n | 4000 ≤ n ∧ n < 10000}
           → n ∈ {n | 4000 ≤ n ∧ n < 10000 ∧ b * c > 8} := sorry

end four_digit_numbers_l13_13775


namespace rod_length_l13_13857

theorem rod_length (pieces : ℕ) (length_per_piece_cm : ℕ) (total_length_m : ℝ) :
  pieces = 35 → length_per_piece_cm = 85 → total_length_m = 29.75 :=
by
  intros h1 h2
  sorry

end rod_length_l13_13857


namespace rafael_earnings_l13_13176

theorem rafael_earnings 
  (hours_monday : ℕ) 
  (hours_tuesday : ℕ) 
  (hours_left : ℕ) 
  (rate_per_hour : ℕ) 
  (h_monday : hours_monday = 10) 
  (h_tuesday : hours_tuesday = 8) 
  (h_left : hours_left = 20) 
  (h_rate : rate_per_hour = 20) : 
  (hours_monday + hours_tuesday + hours_left) * rate_per_hour = 760 := 
by
  sorry

end rafael_earnings_l13_13176


namespace find_speed_of_B_l13_13065

namespace BicycleSpeed

variables (d : ℝ) (t_diff : ℝ) (v_A v_B : ℝ)

-- Given conditions
def given_conditions := 
d = 12 ∧ 
t_diff = 1/6 ∧ 
v_A = 1.2 * v_B ∧ 
(12 / v_B - 12 / v_A = t_diff)

theorem find_speed_of_B
  (h : given_conditions d t_diff v_A v_B) : 
  v_B = 12 :=
sorry

end BicycleSpeed

end find_speed_of_B_l13_13065


namespace hajar_score_l13_13294

variables (F H : ℕ)

theorem hajar_score 
  (h1 : F - H = 21)
  (h2 : F + H = 69)
  (h3 : F > H) :
  H = 24 :=
sorry

end hajar_score_l13_13294


namespace angle_bisector_inequality_l13_13856

noncomputable def triangle_ABC (A B C K M : Type) [Inhabited A] [Inhabited B] [Inhabited C] (AB BC CA AK CM AM MK KC : ℝ) 
  (Hbisector_CM : BM / MA = BC / CA)
  (Hbisector_AK : BK / KC = AB / AC)
  (Hcondition : AB > BC) : Prop :=
  AM > MK ∧ MK > KC

theorem angle_bisector_inequality (A B C K M : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (AB BC CA AK CM AM MK KC : ℝ)
  (Hbisector_CM : BM / MA = BC / CA)
  (Hbisector_AK : BK / KC = AB / AC)
  (Hcondition : AB > BC) : AM > MK ∧ MK > KC :=
by
  sorry

end angle_bisector_inequality_l13_13856


namespace mean_of_remaining_students_l13_13493

variable (k : ℕ) (h1 : k > 20)

def mean_of_class (mean : ℝ := 10) := mean
def mean_of_20_students (mean : ℝ := 16) := mean

theorem mean_of_remaining_students 
  (h2 : mean_of_class = 10)
  (h3 : mean_of_20_students = 16) :
  let remaining_students := (k - 20)
  let total_score_20 := 20 * mean_of_20_students
  let total_score_class := k * mean_of_class
  let total_score_remaining := total_score_class - total_score_20
  let mean_remaining := total_score_remaining / remaining_students
  mean_remaining = (10 * k - 320) / (k - 20) :=
sorry

end mean_of_remaining_students_l13_13493


namespace problem_1_problem_2_l13_13848

theorem problem_1 (h : Real.tan (α / 2) = 2) : Real.tan (α + Real.arctan 1) = -1/7 :=
by
  sorry

theorem problem_2 (h : Real.tan (α / 2) = 2) : (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 :=
by
  sorry

end problem_1_problem_2_l13_13848


namespace students_not_coming_l13_13976

-- Define the conditions
def pieces_per_student : ℕ := 4
def pieces_made_last_monday : ℕ := 40
def pieces_made_upcoming_monday : ℕ := 28

-- Define the number of students not coming to class
theorem students_not_coming :
  (pieces_made_last_monday / pieces_per_student) - 
  (pieces_made_upcoming_monday / pieces_per_student) = 3 :=
by sorry

end students_not_coming_l13_13976


namespace max_cube_side_length_max_parallelepiped_dimensions_l13_13296

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l13_13296


namespace range_of_m_l13_13622

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log x + m / x

theorem range_of_m (m : ℝ) :
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (f m b - f m a) / (b - a) < 1) →
  m ≥ 1 / 4 :=
by
  sorry

end range_of_m_l13_13622


namespace find_goods_train_speed_l13_13152

-- Definition of given conditions
def speed_of_man_train_kmph : ℝ := 120
def time_goods_train_seconds : ℝ := 9
def length_goods_train_meters : ℝ := 350

-- The proof statement
theorem find_goods_train_speed :
  let relative_speed_mps := (speed_of_man_train_kmph + goods_train_speed_kmph) * (5 / 18)
  ∃ (goods_train_speed_kmph : ℝ), relative_speed_mps = length_goods_train_meters / time_goods_train_seconds ∧ goods_train_speed_kmph = 20 :=
by {
  sorry
}

end find_goods_train_speed_l13_13152


namespace y_percentage_of_8950_l13_13922

noncomputable def x := 0.18 * 4750
noncomputable def y := 1.30 * x
theorem y_percentage_of_8950 : (y / 8950) * 100 = 12.42 := 
by 
  -- proof steps are omitted
  sorry

end y_percentage_of_8950_l13_13922


namespace average_after_15th_inning_l13_13321

theorem average_after_15th_inning (A : ℝ) 
    (h_avg_increase : (14 * A + 75) = 15 * (A + 3)) : 
    A + 3 = 33 :=
by {
  sorry
}

end average_after_15th_inning_l13_13321


namespace range_of_a_for_distinct_real_roots_l13_13560

theorem range_of_a_for_distinct_real_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔ (a < 2 ∧ a ≠ 1) :=
by
  sorry

end range_of_a_for_distinct_real_roots_l13_13560


namespace age_of_vanya_and_kolya_l13_13807

theorem age_of_vanya_and_kolya (P V K : ℕ) (hP : P = 10)
  (hV : V = P - 1) (hK : K = P - 5 + 1) : V = 9 ∧ K = 6 :=
by
  sorry

end age_of_vanya_and_kolya_l13_13807


namespace initial_money_l13_13395

-- Let M represent the initial amount of money Mrs. Hilt had.
variable (M : ℕ)

-- Condition 1: Mrs. Hilt bought a pencil for 11 cents.
def pencil_cost : ℕ := 11

-- Condition 2: She had 4 cents left after buying the pencil.
def amount_left : ℕ := 4

-- Proof problem statement: Prove that M = 15 given the above conditions.
theorem initial_money (h : M = pencil_cost + amount_left) : M = 15 :=
by
  sorry

end initial_money_l13_13395


namespace group_1991_l13_13770

theorem group_1991 (n : ℕ) (h1 : 1 ≤ n) (h2 : 1991 = 2 * n ^ 2 - 1) : n = 32 := 
sorry

end group_1991_l13_13770


namespace Valleyball_Soccer_League_members_l13_13681

theorem Valleyball_Soccer_League_members (cost_socks cost_tshirt total_expenditure cost_per_member: ℕ) (h1 : cost_socks = 6) (h2 : cost_tshirt = cost_socks + 8) (h3 : total_expenditure = 3740) (h4 : cost_per_member = cost_socks + 2 * cost_tshirt) : 
  total_expenditure = 3740 → cost_per_member = 34 → total_expenditure / cost_per_member = 110 :=
sorry

end Valleyball_Soccer_League_members_l13_13681


namespace meaningful_expression_range_l13_13318

theorem meaningful_expression_range (x : ℝ) : (3 * x + 9 ≥ 0) ∧ (x ≠ 2) ↔ (x ≥ -3 ∧ x ≠ 2) := by
  sorry

end meaningful_expression_range_l13_13318


namespace geometric_sequence_sum_l13_13727

theorem geometric_sequence_sum (S : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) : 
  (∀ n : ℕ, n > 0 → S n = 2^n + a) →
  (S 1 = 2 + a) →
  (∀ n ≥ 2, a_n n = S n - S (n - 1)) →
  (a_n 1 = 1) →
  a = -1 :=
by
  sorry

end geometric_sequence_sum_l13_13727


namespace tap_fills_tank_without_leakage_in_12_hours_l13_13015

theorem tap_fills_tank_without_leakage_in_12_hours 
  (R_t R_l : ℝ)
  (h1 : (R_t - R_l) * 18 = 1)
  (h2 : R_l * 36 = 1) :
  1 / R_t = 12 := 
by
  sorry

end tap_fills_tank_without_leakage_in_12_hours_l13_13015


namespace students_in_each_grade_l13_13865

theorem students_in_each_grade (total_students : ℕ) (total_grades : ℕ) (students_per_grade : ℕ) :
  total_students = 22800 → total_grades = 304 → students_per_grade = total_students / total_grades → students_per_grade = 75 :=
by
  intros h1 h2 h3
  sorry

end students_in_each_grade_l13_13865


namespace calculate_V3_at_2_l13_13124

def polynomial (x : ℕ) : ℕ :=
  (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem calculate_V3_at_2 : polynomial 2 = 71 := by
  sorry

end calculate_V3_at_2_l13_13124


namespace find_c1_minus_c2_l13_13400

-- Define the conditions of the problem
variables (c1 c2 : ℝ)
variables (x y : ℝ)
variables (h1 : (2 : ℝ) * x + 3 * y = c1)
variables (h2 : (3 : ℝ) * x + 2 * y = c2)
variables (sol_x : x = 2)
variables (sol_y : y = 1)

-- Define the theorem to be proven
theorem find_c1_minus_c2 : c1 - c2 = -1 := 
by
  sorry

end find_c1_minus_c2_l13_13400


namespace greatest_y_l13_13762

theorem greatest_y (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : y ≤ 24 :=
sorry

end greatest_y_l13_13762


namespace infinite_geometric_series_sum_l13_13945

theorem infinite_geometric_series_sum :
  let a : ℚ := 1
  let r : ℚ := 1 / 3
  ∑' (n : ℕ), a * r ^ n = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l13_13945


namespace soccer_goal_difference_l13_13258

theorem soccer_goal_difference (n : ℕ) (h : n = 2020) :
  ¬ ∃ g : Fin n → ℤ,
    (∀ i j : Fin n, i < j → (g i < g j)) ∧ 
    (∀ i : Fin n, ∃ x y : ℕ, x + y = n - 1 ∧ 3 * x = (n - 1 - x) ∧ g i = x - y) :=
by
  sorry

end soccer_goal_difference_l13_13258


namespace largest_consecutive_multiple_of_3_l13_13492

theorem largest_consecutive_multiple_of_3 (n : ℕ) 
  (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 72) : 3 * (n + 2) = 27 :=
by 
  sorry

end largest_consecutive_multiple_of_3_l13_13492


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l13_13383

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l13_13383


namespace min_value_fraction_sum_l13_13832

open Real

theorem min_value_fraction_sum (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h_sum : x + y + z = 2) :
    ∃ m, m = (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ∧ m = 9/4 :=
by
  sorry

end min_value_fraction_sum_l13_13832


namespace line_eq_l13_13010

theorem line_eq (x_1 y_1 x_2 y_2 : ℝ) (h1 : x_1 + x_2 = 8) (h2 : y_1 + y_2 = 2)
  (h3 : x_1^2 - 4 * y_1^2 = 4) (h4 : x_2^2 - 4 * y_2^2 = 4) :
  ∃ l : ℝ, ∀ x y : ℝ, x - y - 3 = l :=
by sorry

end line_eq_l13_13010


namespace map_distance_l13_13563

/--
On a map, 8 cm represents 40 km. Prove that 20 cm represents 100 km.
-/
theorem map_distance (scale_factor : ℕ) (distance_cm : ℕ) (distance_km : ℕ) 
  (h_scale : scale_factor = 5) (h_distance_cm : distance_cm = 20) : 
  distance_km = 20 * scale_factor := 
by {
  sorry
}

end map_distance_l13_13563


namespace smallest_sum_l13_13550

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  (∀ A B C D : ℕ, 
    5 * A = 25 * A - 27 * B ∧
    5 * B = 15 * A - 16 * B ∧
    3 * C = 25 * C - 27 * D ∧
    3 * D = 15 * C - 16 * D) ∧
  a = 4 ∧ b = 3 ∧ c = 27 ∧ d = 22 ∧ a + b + c + d = 56

theorem smallest_sum : problem_statement :=
  sorry

end smallest_sum_l13_13550


namespace find_three_numbers_l13_13165

theorem find_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : a + b - c = 10) 
  (h3 : a - b + c = 8) : 
  a = 9 ∧ b = 3.5 ∧ c = 2.5 := 
by 
  sorry

end find_three_numbers_l13_13165


namespace total_sum_is_750_l13_13508

-- Define the individual numbers
def joyce_number : ℕ := 30

def xavier_number (joyce : ℕ) : ℕ :=
  4 * joyce

def coraline_number (xavier : ℕ) : ℕ :=
  xavier + 50

def jayden_number (coraline : ℕ) : ℕ :=
  coraline - 40

def mickey_number (jayden : ℕ) : ℕ :=
  jayden + 20

def yvonne_number (xavier joyce : ℕ) : ℕ :=
  xavier + joyce

-- Prove the total sum is 750
theorem total_sum_is_750 :
  joyce_number + xavier_number joyce_number + coraline_number (xavier_number joyce_number) +
  jayden_number (coraline_number (xavier_number joyce_number)) +
  mickey_number (jayden_number (coraline_number (xavier_number joyce_number))) +
  yvonne_number (xavier_number joyce_number) joyce_number = 750 :=
by {
  -- Proof omitted for brevity
  sorry
}

end total_sum_is_750_l13_13508


namespace find_dividend_l13_13660

theorem find_dividend (partial_product : ℕ) (remainder : ℕ) (divisor quotient : ℕ) :
  partial_product = 2015 → 
  remainder = 0 →
  divisor = 105 → 
  quotient = 197 → 
  divisor * quotient + remainder = partial_product → 
  partial_product * 10 = 20685 :=
by {
  -- Proof skipped
  sorry
}

end find_dividend_l13_13660


namespace g_negative_example1_g_negative_example2_g_negative_example3_l13_13968

noncomputable def g (a : ℚ) : ℚ := sorry

axiom g_mul (a b : ℚ) (ha : 0 < a) (hb : 0 < b) : g (a * b) = g a + g b
axiom g_prime (p : ℕ) (hp : Nat.Prime p) : g (p * p) = p

theorem g_negative_example1 : g (8/81) < 0 := sorry
theorem g_negative_example2 : g (25/72) < 0 := sorry
theorem g_negative_example3 : g (49/18) < 0 := sorry

end g_negative_example1_g_negative_example2_g_negative_example3_l13_13968


namespace sum_a_eq_9_l13_13196

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sum_a_eq_9 (a2 a3 a4 a5 a6 a7 : ℤ) 
  (h1 : 0 ≤ a2 ∧ a2 < 2) (h2 : 0 ≤ a3 ∧ a3 < 3) (h3 : 0 ≤ a4 ∧ a4 < 4)
  (h4 : 0 ≤ a5 ∧ a5 < 5) (h5 : 0 ≤ a6 ∧ a6 < 6) (h6 : 0 ≤ a7 ∧ a7 < 7)
  (h_eq : (5 : ℚ) / 7 = (a2 : ℚ) / factorial 2 + (a3 : ℚ) / factorial 3 + (a4 : ℚ) / factorial 4 + 
                         (a5 : ℚ) / factorial 5 + (a6 : ℚ) / factorial 6 + (a7 : ℚ) / factorial 7) :
  a2 + a3 + a4 + a5 + a6 + a7 = 9 := 
sorry

end sum_a_eq_9_l13_13196


namespace exponentiation_and_multiplication_of_fractions_l13_13160

-- Let's define the required fractions
def a : ℚ := 3 / 4
def b : ℚ := 1 / 5

-- Define the expected result
def expected_result : ℚ := 81 / 1280

-- State the theorem to prove
theorem exponentiation_and_multiplication_of_fractions : (a^4) * b = expected_result := by 
  sorry

end exponentiation_and_multiplication_of_fractions_l13_13160


namespace determine_x_l13_13463

-- Definitions based on conditions
variables {x : ℝ}

-- Problem statement
theorem determine_x (h : (6 * x)^5 = (18 * x)^4) (hx : x ≠ 0) : x = 27 / 2 :=
by
  sorry

end determine_x_l13_13463


namespace additional_hours_needed_l13_13308

-- Define the conditions
def speed : ℕ := 5  -- kilometers per hour
def total_distance : ℕ := 30 -- kilometers
def hours_walked : ℕ := 3 -- hours

-- Define the statement to prove
theorem additional_hours_needed : total_distance / speed - hours_walked = 3 := 
by
  sorry

end additional_hours_needed_l13_13308


namespace max_value_of_sum_l13_13403

theorem max_value_of_sum (x y z : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (eq : x^2 + y^2 + z^2 + x + 2*y + 3*z = (13 : ℝ) / 4) : x + y + z ≤ 3 / 2 :=
sorry

end max_value_of_sum_l13_13403


namespace compare_negatives_l13_13905

theorem compare_negatives : -1 > -2 := 
by 
  sorry

end compare_negatives_l13_13905


namespace find_x_l13_13696

theorem find_x (x : ℝ) (A1 A2 : ℝ) (P1 P2 : ℝ)
    (hA1 : A1 = x^2 + 4*x + 4)
    (hA2 : A2 = 4*x^2 - 12*x + 9)
    (hP : P1 + P2 = 32)
    (hP1 : P1 = 4 * (x + 2))
    (hP2 : P2 = 4 * (2*x - 3)) :
    x = 3 :=
by
  sorry

end find_x_l13_13696


namespace distinct_terms_count_l13_13644

theorem distinct_terms_count
  (x y z w p q r s t : Prop)
  (h1 : ¬(x = y ∨ x = z ∨ x = w ∨ y = z ∨ y = w ∨ z = w))
  (h2 : ¬(p = q ∨ p = r ∨ p = s ∨ p = t ∨ q = r ∨ q = s ∨ q = t ∨ r = s ∨ r = t ∨ s = t)) :
  ∃ (n : ℕ), n = 20 := by
  sorry

end distinct_terms_count_l13_13644


namespace fifth_student_gold_stickers_l13_13528

theorem fifth_student_gold_stickers :
  ∀ s1 s2 s3 s4 s5 s6 : ℕ,
  s1 = 29 →
  s2 = 35 →
  s3 = 41 →
  s4 = 47 →
  s6 = 59 →
  (s2 - s1 = 6) →
  (s3 - s2 = 6) →
  (s4 - s3 = 6) →
  (s6 - s4 = 12) →
  s5 = s4 + (s2 - s1) →
  s5 = 53 := by
  intros s1 s2 s3 s4 s5 s6 hs1 hs2 hs3 hs4 hs6 hd1 hd2 hd3 hd6 heq
  subst_vars
  sorry

end fifth_student_gold_stickers_l13_13528


namespace func4_same_domain_range_as_func1_l13_13387

noncomputable def func1_domain : Set ℝ := {x | 0 < x}
noncomputable def func1_range : Set ℝ := {y | 0 < y}

noncomputable def func4_domain : Set ℝ := {x | 0 < x}
noncomputable def func4_range : Set ℝ := {y | 0 < y}

theorem func4_same_domain_range_as_func1 :
  (func4_domain = func1_domain) ∧ (func4_range = func1_range) :=
sorry

end func4_same_domain_range_as_func1_l13_13387


namespace range_of_m_l13_13585

open Set

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 7 }
def B (m : ℝ) : Set ℝ := { x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1) }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → m ≤ 4 :=
by
  intro h
  sorry

end range_of_m_l13_13585


namespace measure_angle_BRC_l13_13313

inductive Point : Type
| A 
| B 
| C 
| P 
| Q 
| R 

open Point

def is_inside_triangle (P : Point) (A B C : Point) : Prop := sorry

def intersection (a b c : Point) : Point := sorry

def length (a b : Point) : ℝ := sorry

def angle (a b c : Point) : ℝ := sorry

theorem measure_angle_BRC 
  (P : Point) (A B C : Point)
  (h_inside : is_inside_triangle P A B C)
  (hQ : Q = intersection A C P)
  (hR : R = intersection A B P)
  (h_lengths_equal : length A R = length R B ∧ length R B = length C P)
  (h_CQ_PQ : length C Q = length P Q) :
  angle B R C = 120 := 
sorry

end measure_angle_BRC_l13_13313


namespace total_branches_in_pine_tree_l13_13604

-- Definitions based on the conditions
def middle_branch : ℕ := 0 -- arbitrary assignment to represent the middle branch

def jumps_up_5 (b : ℕ) : ℕ := b + 5
def jumps_down_7 (b : ℕ) : ℕ := b - 7
def jumps_up_4 (b : ℕ) : ℕ := b + 4
def jumps_up_9 (b : ℕ) : ℕ := b + 9

-- The statement to be proven
theorem total_branches_in_pine_tree : 
  (jumps_up_9 (jumps_up_4 (jumps_down_7 (jumps_up_5 middle_branch))) = 11) →
  ∃ n, n = 23 :=
by
  sorry

end total_branches_in_pine_tree_l13_13604


namespace quad_root_sum_product_l13_13050

theorem quad_root_sum_product (α β : ℝ) (h₁ : α ≠ β) (h₂ : α * α - 5 * α - 2 = 0) (h₃ : β * β - 5 * β - 2 = 0) : 
  α + β + α * β = 3 := 
by
  sorry

end quad_root_sum_product_l13_13050


namespace insects_legs_l13_13771

theorem insects_legs (n : ℕ) (l : ℕ) (h₁ : n = 6) (h₂ : l = 6) : n * l = 36 :=
by sorry

end insects_legs_l13_13771


namespace abs_diff_condition_l13_13555

theorem abs_diff_condition {a b : ℝ} (h1 : |a| = 1) (h2 : |b - 1| = 2) (h3 : a > b) : a - b = 2 := 
sorry

end abs_diff_condition_l13_13555


namespace point_A_outside_circle_iff_l13_13130

-- Define the conditions
def B : ℝ := 16
def radius : ℝ := 4
def A_position (t : ℝ) : ℝ := 2 * t

-- Define the theorem
theorem point_A_outside_circle_iff (t : ℝ) : (A_position t < B - radius) ∨ (A_position t > B + radius) ↔ (t < 6 ∨ t > 10) :=
by
  sorry

end point_A_outside_circle_iff_l13_13130


namespace nabla_eq_37_l13_13104

def nabla (a b : ℤ) : ℤ := a * b + a - b

theorem nabla_eq_37 : nabla (-5) (-7) = 37 := by
  sorry

end nabla_eq_37_l13_13104


namespace simplify_and_evaluate_l13_13837

def expr (x : ℤ) : ℤ := (x + 2) * (x - 2) - (x - 1) ^ 2

theorem simplify_and_evaluate : expr (-1) = -7 := by
  sorry

end simplify_and_evaluate_l13_13837


namespace tangency_condition_for_parabola_and_line_l13_13605

theorem tangency_condition_for_parabola_and_line (k : ℚ) :
  (∀ x y : ℚ, (6 * x - 4 * y + k = 0) ↔ (y^2 = 16 * x)) ↔ (k = 32 / 3) :=
  sorry

end tangency_condition_for_parabola_and_line_l13_13605


namespace calculate_jessie_points_l13_13692

theorem calculate_jessie_points (total_points : ℕ) (some_players_points : ℕ) (players : ℕ) :
  total_points = 311 →
  some_players_points = 188 →
  players = 3 →
  (total_points - some_players_points) / players = 41 :=
by
  intros
  sorry

end calculate_jessie_points_l13_13692


namespace boys_belong_to_other_communities_l13_13548

-- Definitions for the given problem
def total_boys : ℕ := 850
def percent_muslims : ℝ := 0.34
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10
def percent_other : ℝ := 1 - (percent_muslims + percent_hindus + percent_sikhs)

-- Statement to prove that the number of boys belonging to other communities is 238
theorem boys_belong_to_other_communities : 
  (percent_other * total_boys) = 238 := by 
  sorry

end boys_belong_to_other_communities_l13_13548


namespace train_passing_time_l13_13859

theorem train_passing_time 
  (length_train : ℕ) 
  (speed_train_kmph : ℕ) 
  (time_to_pass : ℕ)
  (h1 : length_train = 60)
  (h2 : speed_train_kmph = 54)
  (h3 : time_to_pass = 4) :
  time_to_pass = length_train * 18 / (speed_train_kmph * 5) := by
  sorry

end train_passing_time_l13_13859


namespace worms_stolen_correct_l13_13282

-- Given conditions translated into Lean statements
def num_babies : ℕ := 6
def worms_per_baby_per_day : ℕ := 3
def papa_bird_worms : ℕ := 9
def mama_bird_initial_worms : ℕ := 13
def additional_worms_needed : ℕ := 34

-- From the conditions, determine the total number of worms needed for 3 days
def total_worms_needed : ℕ := worms_per_baby_per_day * num_babies * 3

-- Calculate how many worms they will have after catching additional worms
def total_worms_after_catching_more : ℕ := papa_bird_worms + mama_bird_initial_worms + additional_worms_needed

-- Amount suspected to be stolen
def worms_stolen : ℕ := total_worms_after_catching_more - total_worms_needed

theorem worms_stolen_correct : worms_stolen = 2 :=
by sorry

end worms_stolen_correct_l13_13282


namespace total_dogs_l13_13584

theorem total_dogs (D : ℕ) 
(h1 : 12 = 12)
(h2 : D / 2 = D / 2)
(h3 : D / 4 = D / 4)
(h4 : 10 = 10)
(h_eq : 12 + D / 2 + D / 4 + 10 = D) : 
D = 88 := by
sorry

end total_dogs_l13_13584


namespace pine_trees_multiple_of_27_l13_13380

noncomputable def numberOfPineTrees (n : ℕ) : ℕ := 27 * n

theorem pine_trees_multiple_of_27 (oak_trees : ℕ) (max_trees_per_row : ℕ) (rows_of_oak : ℕ) :
  oak_trees = 54 → max_trees_per_row = 27 → rows_of_oak = oak_trees / max_trees_per_row →
  ∃ n : ℕ, numberOfPineTrees n = 27 * n :=
by
  intros
  use (oak_trees - rows_of_oak * max_trees_per_row) / 27
  sorry

end pine_trees_multiple_of_27_l13_13380


namespace school_selection_theorem_l13_13030

-- Define the basic setup and conditions
def school_selection_problem : Prop :=
  let schools := ["A", "B", "C", "D"]
  let total_schools := 4
  let selected_schools := 2
  let combinations := Nat.choose total_schools selected_schools
  let favorable_outcomes := Nat.choose (total_schools - 1) (selected_schools - 1)
  let probability := (favorable_outcomes : ℚ) / (combinations : ℚ)
  probability = 1 / 2

-- Proof is yet to be provided
theorem school_selection_theorem : school_selection_problem := sorry

end school_selection_theorem_l13_13030


namespace line_equation_through_point_and_area_l13_13241

theorem line_equation_through_point_and_area (k b : ℝ) :
  (∃ (P : ℝ × ℝ), P = (4/3, 2)) ∧
  (∀ (A B : ℝ × ℝ), A = (- b / k, 0) ∧ B = (0, b) → 
  1 / 2 * abs ((- b / k) * b) = 6) →
  (y = k * x + b ↔ (y = -3/4 * x + 3 ∨ y = -3 * x + 6)) :=
by
  sorry

end line_equation_through_point_and_area_l13_13241


namespace largest_4_digit_congruent_to_17_mod_26_l13_13635

theorem largest_4_digit_congruent_to_17_mod_26 :
  ∃ x, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧ (∀ y, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) ∧ x = 9972 := 
by
  sorry

end largest_4_digit_congruent_to_17_mod_26_l13_13635


namespace royal_children_count_l13_13224

-- Defining the initial conditions
def king_age := 35
def queen_age := 35
def sons := 3
def daughters_min := 1
def initial_children_age := 35
def max_children := 20

-- Statement of the problem
theorem royal_children_count (d n C : ℕ) 
    (h1 : king_age = 35)
    (h2 : queen_age = 35)
    (h3 : sons = 3)
    (h4 : daughters_min ≥ 1)
    (h5 : initial_children_age = 35)
    (h6 : 70 + 2 * n = 35 + (d + sons) * n)
    (h7 : C = d + sons)
    (h8 : C ≤ max_children) : 
    C = 7 ∨ C = 9 := 
sorry

end royal_children_count_l13_13224


namespace james_calories_per_minute_l13_13629

variable (classes_per_week : ℕ) (hours_per_class : ℝ) (total_calories_per_week : ℕ)

theorem james_calories_per_minute
  (h1 : classes_per_week = 3)
  (h2 : hours_per_class = 1.5)
  (h3 : total_calories_per_week = 1890) :
  total_calories_per_week / (classes_per_week * (hours_per_class * 60)) = 7 := 
by
  sorry

end james_calories_per_minute_l13_13629


namespace P_sufficient_but_not_necessary_for_Q_l13_13941

def P (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def Q (x : ℝ) : Prop := x^2 - 2 * x + 1 > 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ ¬ (∀ x : ℝ, Q x → P x) :=
by 
  sorry

end P_sufficient_but_not_necessary_for_Q_l13_13941


namespace intersection_A_B_l13_13406

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l13_13406


namespace eating_time_175_seconds_l13_13880

variable (Ponchik_time Neznaika_time : ℝ)
variable (Ponchik_rate Neznaika_rate : ℝ)

theorem eating_time_175_seconds
    (hP_rate : Ponchik_rate = 1 / Ponchik_time)
    (hP_time : Ponchik_time = 5)
    (hN_rate : Neznaika_rate = 1 / Neznaika_time)
    (hN_time : Neznaika_time = 7)
    (combined_rate := Ponchik_rate + Neznaika_rate)
    (total_minutes := 1 / combined_rate)
    (total_seconds := total_minutes * 60):
    total_seconds = 175 := by
  sorry

end eating_time_175_seconds_l13_13880


namespace jan_total_skips_l13_13331

def jan_initial_speed : ℕ := 70
def jan_training_factor : ℕ := 2
def jan_skipping_time : ℕ := 5

theorem jan_total_skips :
  (jan_initial_speed * jan_training_factor) * jan_skipping_time = 700 := by
  sorry

end jan_total_skips_l13_13331


namespace arithmetic_sequence_product_l13_13502

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) 
  (h_inc : ∀ n, b (n + 1) - b n = d)
  (h_pos : d > 0)
  (h_prod : b 5 * b 6 = 21) 
  : b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l13_13502


namespace wang_payment_correct_l13_13822

noncomputable def first_trip_payment (x : ℝ) : ℝ := 0.9 * x
noncomputable def second_trip_payment (y : ℝ) : ℝ := 300 * 0.9 + (y - 300) * 0.8

theorem wang_payment_correct (x y: ℝ) 
  (cond1: 0.1 * x = 19)
  (cond2: (x + y) - (0.9 * x + ((y - 300) * 0.8 + 300 * 0.9)) = 67) :
  first_trip_payment x = 171 ∧ second_trip_payment y = 342 := 
by
  sorry

end wang_payment_correct_l13_13822


namespace team_formation_l13_13688

def nat1 : ℕ := 7  -- Number of natives who know mathematics and physics
def nat2 : ℕ := 6  -- Number of natives who know physics and chemistry
def nat3 : ℕ := 3  -- Number of natives who know chemistry and mathematics
def nat4 : ℕ := 4  -- Number of natives who know physics and biology

def totalWaysToFormTeam (n1 n2 n3 n4 : ℕ) : ℕ := (n1 + n2 + n3 + n4).choose 3
def waysFromSameGroup (n : ℕ) : ℕ := n.choose 3

def waysFromAllGroups (n1 n2 n3 n4 : ℕ) : ℕ := (waysFromSameGroup n1) + (waysFromSameGroup n2) + (waysFromSameGroup n3) + (waysFromSameGroup n4)

theorem team_formation : totalWaysToFormTeam nat1 nat2 nat3 nat4 - waysFromAllGroups nat1 nat2 nat3 nat4 = 1080 := 
by
    sorry

end team_formation_l13_13688


namespace cooper_age_l13_13613

variable (Cooper Dante Maria : ℕ)

-- Conditions
def sum_of_ages : Prop := Cooper + Dante + Maria = 31
def dante_twice_cooper : Prop := Dante = 2 * Cooper
def maria_one_year_older : Prop := Maria = Dante + 1

theorem cooper_age (h1 : sum_of_ages Cooper Dante Maria) (h2 : dante_twice_cooper Cooper Dante) (h3 : maria_one_year_older Dante Maria) : Cooper = 6 :=
by
  sorry

end cooper_age_l13_13613


namespace find_triples_l13_13957

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_triples (a b c : ℕ) :
  is_prime (a^2 + 1) ∧
  is_prime (b^2 + 1) ∧
  (a^2 + 1) * (b^2 + 1) = c^2 + 1 →
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3) :=
by
  sorry

end find_triples_l13_13957


namespace helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l13_13408

def heights_A : List ℝ := [3.6, -2.4, 2.8, -1.5, 0.9]
def heights_B : List ℝ := [3.8, -2, 4.1, -2.3]

theorem helicopter_A_highest_altitude :
  List.maximum heights_A = some 3.6 :=
by sorry

theorem helicopter_A_final_altitude :
  List.sum heights_A = 3.4 :=
by sorry

theorem helicopter_B_5th_performance :
  ∃ (x : ℝ), List.sum heights_B + x = 3.4 ∧ x = -0.2 :=
by sorry

end helicopter_A_highest_altitude_helicopter_A_final_altitude_helicopter_B_5th_performance_l13_13408


namespace fraction_difference_l13_13221

theorem fraction_difference : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := 
  sorry

end fraction_difference_l13_13221


namespace inequality_I_l13_13729

theorem inequality_I (a b x y : ℝ) (hx : x < a) (hy : y < b) : x * y < a * b :=
sorry

end inequality_I_l13_13729


namespace equal_wear_tires_l13_13991

theorem equal_wear_tires (t D d : ℕ) (h1 : t = 7) (h2 : D = 42000) (h3 : t * d = 6 * D) : d = 36000 :=
by
  sorry

end equal_wear_tires_l13_13991


namespace sum_of_solutions_eq_seven_l13_13564

theorem sum_of_solutions_eq_seven : 
  ∃ x : ℝ, x + 49/x = 14 ∧ (∀ y : ℝ, y + 49 / y = 14 → y = x) → x = 7 :=
by {
  sorry
}

end sum_of_solutions_eq_seven_l13_13564


namespace opposite_of_neg_2023_l13_13075

theorem opposite_of_neg_2023 : -(-2023) = 2023 := by
  sorry

end opposite_of_neg_2023_l13_13075


namespace decreasing_power_function_l13_13326

theorem decreasing_power_function (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(m^2 + m - 1) < (m^2 - m - 1) * (x + 1) ^ (m^2 + m - 1)) →
  m = -1 :=
sorry

end decreasing_power_function_l13_13326


namespace correct_calculation_B_l13_13924

theorem correct_calculation_B :
  (∀ (a : ℕ), 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) ∧
  (∀ (x : ℕ), 3 * x^2 * 4 * x^2 ≠ 12 * x^2) ∧
  (∀ (y : ℕ), 5 * y^3 * 3 * y^5 ≠ 8 * y^8) →
  (∀ (x : ℕ), 2 * x^2 * 3 * x^2 = 6 * x^4) := 
by
  sorry

end correct_calculation_B_l13_13924


namespace total_spent_l13_13650

def cost_sandwich : ℕ := 2
def cost_hamburger : ℕ := 2
def cost_hotdog : ℕ := 1
def cost_fruit_juice : ℕ := 2

def selene_sandwiches : ℕ := 3
def selene_fruit_juice : ℕ := 1
def tanya_hamburgers : ℕ := 2
def tanya_fruit_juice : ℕ := 2

def total_selene_spent : ℕ := (selene_sandwiches * cost_sandwich) + (selene_fruit_juice * cost_fruit_juice)
def total_tanya_spent : ℕ := (tanya_hamburgers * cost_hamburger) + (tanya_fruit_juice * cost_fruit_juice)

theorem total_spent : total_selene_spent + total_tanya_spent = 16 := by
  sorry

end total_spent_l13_13650


namespace initial_ratio_of_stamps_l13_13213

theorem initial_ratio_of_stamps (P Q : ℕ) (h1 : ((P - 8 : ℤ) : ℚ) / (Q + 8) = 6 / 5) (h2 : P - 8 = Q + 8) : P / Q = 6 / 5 :=
sorry

end initial_ratio_of_stamps_l13_13213


namespace solve_inequality_l13_13251

theorem solve_inequality : { x : ℝ | 3 * x^2 - 1 > 13 - 5 * x } = { x : ℝ | x < -7 ∨ x > 2 } :=
by
  sorry

end solve_inequality_l13_13251


namespace chessboard_edge_count_l13_13514

theorem chessboard_edge_count (n : ℕ) 
  (border_white : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ w : ℕ, w ≥ n)) 
  (border_black : ∀ (c : ℕ), c ∈ (Finset.range (4 * (n - 1))) → (∃ b : ℕ, b ≥ n)) :
  ∃ e : ℕ, e ≥ n :=
sorry

end chessboard_edge_count_l13_13514


namespace solve_for_x_l13_13304

theorem solve_for_x : ∃ x : ℝ, 5 * x + 9 * x = 570 - 12 * (x - 5) ∧ x = 315 / 13 :=
by
  sorry

end solve_for_x_l13_13304


namespace age_of_15th_student_l13_13136

noncomputable def average_age_15_students := 15
noncomputable def average_age_7_students_1 := 14
noncomputable def average_age_7_students_2 := 16
noncomputable def total_students := 15
noncomputable def group_students := 7

theorem age_of_15th_student :
  let total_age_15_students := total_students * average_age_15_students
  let total_age_7_students_1 := group_students * average_age_7_students_1
  let total_age_7_students_2 := group_students * average_age_7_students_2
  let total_age_14_students := total_age_7_students_1 + total_age_7_students_2
  let age_15th_student := total_age_15_students - total_age_14_students
  age_15th_student = 15 :=
by
  sorry

end age_of_15th_student_l13_13136


namespace simplified_radical_formula_l13_13963

theorem simplified_radical_formula (y : ℝ) (hy : 0 ≤ y):
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) :=
by
  sorry

end simplified_radical_formula_l13_13963


namespace ben_eggs_left_l13_13873

def initial_eggs : ℕ := 50
def day1_morning : ℕ := 5
def day1_afternoon : ℕ := 4
def day2_morning : ℕ := 8
def day2_evening : ℕ := 3
def day3_afternoon : ℕ := 6
def day3_night : ℕ := 2

theorem ben_eggs_left : initial_eggs - (day1_morning + day1_afternoon + day2_morning + day2_evening + day3_afternoon + day3_night) = 22 := 
by
  sorry

end ben_eggs_left_l13_13873


namespace S_30_value_l13_13066

noncomputable def geometric_sequence_sum (n : ℕ) : ℝ := sorry

axiom S_10 : geometric_sequence_sum 10 = 10
axiom S_20 : geometric_sequence_sum 20 = 30

theorem S_30_value : geometric_sequence_sum 30 = 70 :=
by
  sorry

end S_30_value_l13_13066


namespace marked_price_correct_l13_13485

theorem marked_price_correct
    (initial_price : ℝ)
    (initial_discount_rate : ℝ)
    (profit_margin_rate : ℝ)
    (final_discount_rate : ℝ)
    (purchase_price : ℝ)
    (final_selling_price : ℝ)
    (marked_price : ℝ)
    (h_initial_price : initial_price = 30)
    (h_initial_discount_rate : initial_discount_rate = 0.15)
    (h_profit_margin_rate : profit_margin_rate = 0.20)
    (h_final_discount_rate : final_discount_rate = 0.25)
    (h_purchase_price : purchase_price = initial_price * (1 - initial_discount_rate))
    (h_final_selling_price : final_selling_price = purchase_price * (1 + profit_margin_rate))
    (h_marked_price : marked_price * (1 - final_discount_rate) = final_selling_price) : 
    marked_price = 40.80 :=
by
  sorry

end marked_price_correct_l13_13485


namespace max_sum_nonneg_l13_13317

theorem max_sum_nonneg (a b c d : ℝ) (h : a + b + c + d = 0) : 
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := 
sorry

end max_sum_nonneg_l13_13317


namespace uneaten_chips_correct_l13_13954

def cookies_per_dozen : Nat := 12
def dozens : Nat := 4
def chips_per_cookie : Nat := 7

def total_cookies : Nat := dozens * cookies_per_dozen
def total_chips : Nat := total_cookies * chips_per_cookie
def eaten_cookies : Nat := total_cookies / 2
def uneaten_cookies : Nat := total_cookies - eaten_cookies

def uneaten_chips : Nat := uneaten_cookies * chips_per_cookie

theorem uneaten_chips_correct : uneaten_chips = 168 :=
by
  -- Placeholder for the proof
  sorry

end uneaten_chips_correct_l13_13954


namespace intersection_A_B_l13_13192

def setA : Set ℝ := {x | x^2 - 1 < 0}
def setB : Set ℝ := {x | x > 0}

theorem intersection_A_B : setA ∩ setB = {x | 0 < x ∧ x < 1} := 
by 
  sorry

end intersection_A_B_l13_13192


namespace boat_speed_in_still_water_l13_13404

variable (B S : ℝ)

def downstream_speed := 10
def upstream_speed := 4

theorem boat_speed_in_still_water :
  B + S = downstream_speed → 
  B - S = upstream_speed → 
  B = 7 :=
by
  intros h₁ h₂
  -- We would insert the proof steps here
  sorry

end boat_speed_in_still_water_l13_13404


namespace combined_weight_is_correct_l13_13424

-- Frank and Gwen's candy weights
def frank_candy : ℕ := 10
def gwen_candy : ℕ := 7

-- The combined weight of candy
def combined_weight : ℕ := frank_candy + gwen_candy

-- Theorem that states the combined weight is 17 pounds
theorem combined_weight_is_correct : combined_weight = 17 :=
by
  -- proves that 10 + 7 = 17
  sorry

end combined_weight_is_correct_l13_13424


namespace muffins_baked_by_James_correct_l13_13813

noncomputable def muffins_baked_by_James (muffins_baked_by_Arthur : ℝ) (ratio : ℝ) : ℝ :=
  muffins_baked_by_Arthur / ratio

theorem muffins_baked_by_James_correct :
  muffins_baked_by_James 115.0 12.0 = 9.5833 :=
by
  -- Add the proof here
  sorry

end muffins_baked_by_James_correct_l13_13813


namespace positive_real_solution_l13_13355

theorem positive_real_solution (x : ℝ) (h : 0 < x)
  (h_eq : (1/3) * (2 * x^2 + 3) = (x^2 - 40 * x - 8) * (x^2 + 20 * x + 4)) :
  x = 20 + Real.sqrt 409 :=
sorry

end positive_real_solution_l13_13355


namespace intersect_A_B_complement_l13_13249

-- Define the sets A and B
def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | x > 1}

-- Find the complement of B in ℝ
def B_complement := {x : ℝ | x ≤ 1}

-- Prove that the intersection of A and the complement of B is equal to (-1, 1]
theorem intersect_A_B_complement : A ∩ B_complement = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  -- Proof is to be provided
  sorry

end intersect_A_B_complement_l13_13249


namespace find_second_number_l13_13150

-- The Lean statement for the given math problem:

theorem find_second_number
  (x y z : ℝ)  -- Represent the three numbers
  (h1 : x = 2 * y)  -- The first number is twice the second
  (h2 : z = (1/3) * x)  -- The third number is one-third of the first
  (h3 : x + y + z = 110)  -- The sum of the three numbers is 110
  : y = 30 :=  -- The second number is 30
sorry

end find_second_number_l13_13150


namespace range_of_a_l13_13652

theorem range_of_a (f : ℝ → ℝ) (h_mono_dec : ∀ x1 x2, -2 ≤ x1 ∧ x1 ≤ 2 ∧ -2 ≤ x2 ∧ x2 ≤ 2 → x1 < x2 → f x1 > f x2) 
  (h_cond : ∀ a, -2 ≤ a + 1 ∧ a + 1 ≤ 2 ∧ -2 ≤ 2 * a ∧ 2 * a ≤ 2 → f (a + 1) < f (2 * a)) :
  { a : ℝ | -1 ≤ a ∧ a < 1 } :=
sorry

end range_of_a_l13_13652


namespace kindergarten_children_count_l13_13487

theorem kindergarten_children_count (D B C : ℕ) (hD : D = 18) (hB : B = 6) (hC : C + B = 12) : D + C + B = 30 :=
by
  sorry

end kindergarten_children_count_l13_13487


namespace solve_r_l13_13839

theorem solve_r (r : ℚ) :
  (r^2 - 5*r + 4) / (r^2 - 8*r + 7) = (r^2 - 2*r - 15) / (r^2 - r - 20) →
  r = -5/4 :=
by
  -- Proof would go here
  sorry

end solve_r_l13_13839


namespace nicky_speed_l13_13425

theorem nicky_speed
  (head_start : ℕ := 36)
  (cristina_speed : ℕ := 6)
  (time_to_catch_up : ℕ := 12)
  (distance_cristina_runs : ℕ := cristina_speed * time_to_catch_up)
  (distance_nicky_runs : ℕ := distance_cristina_runs - head_start)
  (nicky_speed : ℕ := distance_nicky_runs / time_to_catch_up) :
  nicky_speed = 3 :=
by
  sorry

end nicky_speed_l13_13425


namespace odd_function_h_l13_13580

noncomputable def f (x h k : ℝ) : ℝ := Real.log (abs ((1 / (x + 1)) + k)) + h

theorem odd_function_h (k : ℝ) (h : ℝ) (H : ∀ x : ℝ, x ≠ -1 → f x h k = -f (-x) h k) :
  h = Real.log 2 :=
sorry

end odd_function_h_l13_13580


namespace remainder_is_15_l13_13420

-- Definitions based on conditions
def S : ℕ := 476
def L : ℕ := S + 2395
def quotient : ℕ := 6

-- The proof statement
theorem remainder_is_15 : ∃ R : ℕ, L = quotient * S + R ∧ R = 15 := by
  sorry

end remainder_is_15_l13_13420


namespace range_of_a_if_p_and_not_q_l13_13543

open Real

def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a^2 = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a_if_p_and_not_q : 
  (∃ a : ℝ, (p a ∧ ¬q a)) → 
  (∀ a : ℝ, (p a ∧ ¬q a) → (-1 ≤ a ∧ a < 0)) :=
sorry

end range_of_a_if_p_and_not_q_l13_13543


namespace locus_centers_tangent_circles_l13_13928

theorem locus_centers_tangent_circles (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (3 - r)^2) →
  a^2 - 12 * a + 4 * b^2 = 0 :=
by
  sorry

end locus_centers_tangent_circles_l13_13928


namespace laborer_monthly_income_l13_13042

theorem laborer_monthly_income :
  (∃ (I D : ℤ),
    6 * I + D = 540 ∧
    4 * I - D = 270) →
  (∃ I : ℤ,
    I = 81) :=
by
  sorry

end laborer_monthly_income_l13_13042


namespace absolute_value_property_l13_13725

theorem absolute_value_property (a b c : ℤ) (h : |a - b| + |c - a| = 1) : |a - c| + |c - b| + |b - a| = 2 :=
sorry

end absolute_value_property_l13_13725


namespace simplify_expression_l13_13933

theorem simplify_expression (x y : ℝ) (hxy : x ≠ y) : 
  ((x - y) ^ 3 / (x - y) ^ 2) * (y - x) = -(x - y) ^ 2 := 
by
  sorry

end simplify_expression_l13_13933


namespace stratified_sampling_students_l13_13164

theorem stratified_sampling_students :
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  f + s = 70 :=
by
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  sorry

end stratified_sampling_students_l13_13164


namespace photo_counts_correct_l13_13592

open Real

-- Definitions based on the conditions from step a)
def animal_photos : ℕ := 20
def flower_photos : ℕ := 30 -- 1.5 * 20
def total_animal_flower_photos : ℕ := animal_photos + flower_photos
def scenery_abstract_photos_combined : ℕ := (4 / 10) * total_animal_flower_photos -- 40% of total_animal_flower_photos

def x : ℕ := scenery_abstract_photos_combined / 5
def scenery_photos : ℕ := 3 * x
def abstract_photos : ℕ := 2 * x
def total_photos : ℕ := animal_photos + flower_photos + scenery_photos + abstract_photos

-- The statement to prove
theorem photo_counts_correct :
  animal_photos = 20 ∧
  flower_photos = 30 ∧
  total_animal_flower_photos = 50 ∧
  scenery_abstract_photos_combined = 20 ∧
  scenery_photos = 12 ∧
  abstract_photos = 8 ∧
  total_photos = 70 :=
by
  sorry

end photo_counts_correct_l13_13592


namespace households_with_both_car_and_bike_l13_13056

theorem households_with_both_car_and_bike 
  (total_households : ℕ) 
  (households_without_either : ℕ) 
  (households_with_car : ℕ) 
  (households_with_bike_only : ℕ)
  (H1 : total_households = 90)
  (H2 : households_without_either = 11)
  (H3 : households_with_car = 44)
  (H4 : households_with_bike_only = 35)
  : ∃ B : ℕ, households_with_car - households_with_bike_only = B ∧ B = 9 := 
by
  sorry

end households_with_both_car_and_bike_l13_13056


namespace circles_externally_tangent_l13_13277

noncomputable def circle1_center : ℝ × ℝ := (-1, 1)
noncomputable def circle1_radius : ℝ := 2
noncomputable def circle2_center : ℝ × ℝ := (2, -3)
noncomputable def circle2_radius : ℝ := 3

noncomputable def distance_centers : ℝ :=
  Real.sqrt ((circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2)

theorem circles_externally_tangent :
  distance_centers = circle1_radius + circle2_radius :=
by
  -- The proof will show that the distance between the centers is equal to the sum of the radii, 
  -- indicating they are externally tangent.
  sorry

end circles_externally_tangent_l13_13277


namespace friend_spent_more_l13_13098

variable (total_spent : ℕ)
variable (friend_spent : ℕ)
variable (you_spent : ℕ)

-- Conditions
axiom total_is_11 : total_spent = 11
axiom friend_is_7 : friend_spent = 7
axiom spending_relation : total_spent = friend_spent + you_spent

-- Question
theorem friend_spent_more : friend_spent - you_spent = 3 :=
by
  sorry -- Here should be the formal proof

end friend_spent_more_l13_13098


namespace hyejin_math_score_l13_13875

theorem hyejin_math_score :
  let ethics := 82
  let korean_language := 90
  let science := 88
  let social_studies := 84
  let avg_score := 88
  let total_subjects := 5
  ∃ (M : ℕ), (ethics + korean_language + science + social_studies + M) / total_subjects = avg_score := by
    sorry

end hyejin_math_score_l13_13875


namespace decimal_89_to_binary_l13_13440

def decimal_to_binary (n : ℕ) : ℕ := sorry

theorem decimal_89_to_binary :
  decimal_to_binary 89 = 1011001 :=
sorry

end decimal_89_to_binary_l13_13440


namespace children_left_birthday_l13_13947

theorem children_left_birthday 
  (total_guests : ℕ := 60)
  (women : ℕ := 30)
  (men : ℕ := 15)
  (remaining_guests : ℕ := 50)
  (initial_children : ℕ := total_guests - women - men)
  (men_left : ℕ := men / 3)
  (total_left : ℕ := total_guests - remaining_guests)
  (children_left : ℕ := total_left - men_left) :
  children_left = 5 :=
by
  sorry

end children_left_birthday_l13_13947


namespace taxi_fare_range_l13_13815

theorem taxi_fare_range (x : ℝ) (h : 12.5 + 2.4 * (x - 3) = 19.7) : 5 < x ∧ x ≤ 6 :=
by
  -- Given conditions and the equation, we need to prove the inequalities.
  have fare_eq : 12.5 + 2.4 * (x - 3) = 19.7 := h
  sorry

end taxi_fare_range_l13_13815


namespace mean_of_five_numbers_l13_13206

theorem mean_of_five_numbers (a b c d e : ℚ) (h : a + b + c + d + e = 2/3) : 
  (a + b + c + d + e) / 5 = 2 / 15 := 
by 
  -- This is where the proof would go, but we'll omit it as per instructions
  sorry

end mean_of_five_numbers_l13_13206


namespace days_required_by_x_l13_13067

theorem days_required_by_x (x y : ℝ) 
  (h1 : (1 / x + 1 / y = 1 / 12)) 
  (h2 : (1 / y = 1 / 24)) : 
  x = 24 := 
by
  sorry

end days_required_by_x_l13_13067


namespace probability_same_color_l13_13669

theorem probability_same_color :
  let red_marble_prob := (5 / 21) * (4 / 20) * (3 / 19)
  let white_marble_prob := (6 / 21) * (5 / 20) * (4 / 19)
  let blue_marble_prob := (7 / 21) * (6 / 20) * (5 / 19)
  let green_marble_prob := (3 / 21) * (2 / 20) * (1 / 19)
  red_marble_prob + white_marble_prob + blue_marble_prob + green_marble_prob = 66 / 1330 := by
  sorry

end probability_same_color_l13_13669


namespace worker_wage_before_promotion_l13_13535

variable (W_new : ℝ)
variable (W : ℝ)

theorem worker_wage_before_promotion (h1 : W_new = 45) (h2 : W_new = 1.60 * W) :
  W = 28.125 := by
  sorry

end worker_wage_before_promotion_l13_13535


namespace fabric_cut_l13_13524

/-- Given a piece of fabric that is 2/3 meter long,
we can cut a piece measuring 1/2 meter
by folding the original piece into four equal parts and removing one part. -/
theorem fabric_cut :
  ∃ (f : ℚ), f = (2/3 : ℚ) → ∃ (half : ℚ), half = (1/2 : ℚ) ∧ half = f * (3/4 : ℚ) :=
by
  sorry

end fabric_cut_l13_13524


namespace range_of_m_l13_13445

open Set Real

theorem range_of_m (M N : Set ℝ) (m : ℝ) :
    (M = {x | x ≤ m}) →
    (N = {y | ∃ x : ℝ, y = 2^(-x)}) →
    (M ∩ N ≠ ∅) → m > 0 := by
  intros hM hN hMN
  sorry

end range_of_m_l13_13445


namespace wall_width_l13_13825

theorem wall_width
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_length : ℝ) (wall_height : ℝ)
  (num_bricks : ℕ)
  (brick_volume : ℝ := brick_length * brick_width * brick_height)
  (total_volume : ℝ := num_bricks * brick_volume) :
  brick_length = 0.20 → brick_width = 0.10 → brick_height = 0.08 →
  wall_length = 10 → wall_height = 8 → num_bricks = 12250 →
  total_volume = wall_length * wall_height * (0.245 : ℝ) :=
by 
  sorry

end wall_width_l13_13825


namespace solution_set_for_log_inequality_l13_13078

noncomputable def f : ℝ → ℝ := sorry

def isEven (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def isIncreasingOnNonNeg (f : ℝ → ℝ) := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_positive_at_third : Prop := f (1 / 3) > 0

theorem solution_set_for_log_inequality
  (hf_even : isEven f)
  (hf_increasing : isIncreasingOnNonNeg f)
  (hf_positive : f_positive_at_third) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | 0 < x ∧ x < 1/2} ∪ {x : ℝ | 2 < x} := sorry

end solution_set_for_log_inequality_l13_13078


namespace least_possible_value_a2008_l13_13354

theorem least_possible_value_a2008 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, a n < a (n + 1)) 
  (h2 : ∀ i j k l, 1 ≤ i → i < j → j ≤ k → k < l → i + l = j + k → a i + a l > a j + a k)
  : a 2008 ≥ 2015029 :=
sorry

end least_possible_value_a2008_l13_13354


namespace find_a_l13_13044

theorem find_a (a : ℝ) : 
  (∃ (r : ℕ), r = 3 ∧ 
  ((-1)^r * (Nat.choose 5 r : ℝ) * a^(5 - r) = -40)) ↔ a = 2 ∨ a = -2 :=
by
    sorry

end find_a_l13_13044


namespace power_equivalence_l13_13714

theorem power_equivalence (m : ℕ) : 16^6 = 4^m → m = 12 :=
by
  sorry

end power_equivalence_l13_13714


namespace problem_l13_13952

theorem problem (m n : ℕ) 
  (m_pos : 0 < m) 
  (n_pos : 0 < n) 
  (h1 : m + 8 < n) 
  (h2 : (m + (m + 3) + (m + 8) + n + (n + 3) + (2 * n - 1)) / 6 = n + 1) 
  (h3 : (m + 8 + n) / 2 = n + 1) : m + n = 16 :=
  sorry

end problem_l13_13952


namespace negation_p_l13_13951

open Nat

def p : Prop := ∀ n : ℕ, n^2 ≤ 2^n

theorem negation_p : ¬p ↔ ∃ n : ℕ, n^2 > 2^n :=
by
  sorry

end negation_p_l13_13951


namespace measured_weight_loss_l13_13212

variable (W : ℝ) (hW : W > 0)

noncomputable def final_weigh_in (initial_weight : ℝ) : ℝ :=
  (0.90 * initial_weight) * 1.02

theorem measured_weight_loss :
  final_weigh_in W = 0.918 * W → (W - final_weigh_in W) / W * 100 = 8.2 := 
by
  intro h
  unfold final_weigh_in at h
  -- skip detailed proof steps, focus on the statement
  sorry

end measured_weight_loss_l13_13212


namespace annual_growth_rate_proof_l13_13903

-- Lean 4 statement for the given problem
theorem annual_growth_rate_proof (profit_2021 : ℝ) (profit_2023 : ℝ) (r : ℝ)
  (h1 : profit_2021 = 3000)
  (h2 : profit_2023 = 4320)
  (h3 : profit_2023 = profit_2021 * (1 + r) ^ 2) :
  r = 0.2 :=
by sorry

end annual_growth_rate_proof_l13_13903


namespace race_course_length_l13_13053

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : 4 * (d - 69) = d) : d = 92 :=
by
  sorry

end race_course_length_l13_13053


namespace range_of_a_l13_13900

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x + a + 3 > 0) : 0 ≤ a := sorry

end range_of_a_l13_13900


namespace fifth_observation_l13_13002

theorem fifth_observation (O1 O2 O3 O4 O5 O6 O7 O8 O9 : ℝ)
  (h1 : O1 + O2 + O3 + O4 + O5 + O6 + O7 + O8 + O9 = 72)
  (h2 : O1 + O2 + O3 + O4 + O5 = 50)
  (h3 : O5 + O6 + O7 + O8 + O9 = 40) :
  O5 = 18 := 
  sorry

end fifth_observation_l13_13002


namespace smallest_n_term_dec_l13_13085

theorem smallest_n_term_dec (n : ℕ) (h_pos : 0 < n) (h : ∀ d, 0 < d → d = n + 150 → ∀ p, p ∣ d → (p = 2 ∨ p = 5)) :
  n = 10 :=
by {
  sorry
}

end smallest_n_term_dec_l13_13085


namespace subset_proof_l13_13135

-- Define the set B
def B : Set ℝ := { x | x ≥ 0 }

-- Define the set A as the set {1, 2}
def A : Set ℝ := {1, 2}

-- The proof problem: Prove that A ⊆ B
theorem subset_proof : A ⊆ B := sorry

end subset_proof_l13_13135


namespace intersection_point_l13_13046

-- Definitions of the lines
def line1 (x y : ℚ) : Prop := 8 * x - 5 * y = 10
def line2 (x y : ℚ) : Prop := 6 * x + 2 * y = 20

-- Theorem stating the intersection point
theorem intersection_point : line1 (60 / 23) (50 / 23) ∧ line2 (60 / 23) (50 / 23) :=
by {
  sorry
}

end intersection_point_l13_13046


namespace line_parabola_intersection_l13_13699

theorem line_parabola_intersection (k : ℝ) : 
    (∀ l p: ℝ → ℝ, l = (fun x => k * x + 1) ∧ p = (fun x => 4 * x ^ 2) → 
        (∃ x, l x = p x) ∧ (∀ x1 x2, l x1 = p x1 ∧ l x2 = p x2 → x1 = x2) 
    ↔ k = 0 ∨ k = 1) :=
sorry

end line_parabola_intersection_l13_13699


namespace melanie_dimes_final_l13_13678

-- Define a type representing the initial state of Melanie's dimes
variable {initial_dimes : ℕ} (h_initial : initial_dimes = 7)

-- Define a function representing the result after attempting to give away dimes
def remaining_dimes_after_giving (initial_dimes : ℕ) (given_dimes : ℕ) : ℕ :=
  if given_dimes <= initial_dimes then initial_dimes - given_dimes else initial_dimes

-- State the problem
theorem melanie_dimes_final (h_initial : initial_dimes = 7) (given_dimes_dad : ℕ) (h_given_dad : given_dimes_dad = 8) (received_dimes_mom : ℕ) (h_received_mom : received_dimes_mom = 4) :
  remaining_dimes_after_giving initial_dimes given_dimes_dad + received_dimes_mom = 11 :=
by
  sorry

end melanie_dimes_final_l13_13678


namespace members_not_playing_any_sport_l13_13621

theorem members_not_playing_any_sport {total_members badminton_players tennis_players both_players : ℕ}
  (h_total : total_members = 28)
  (h_badminton : badminton_players = 17)
  (h_tennis : tennis_players = 19)
  (h_both : both_players = 10) :
  total_members - (badminton_players + tennis_players - both_players) = 2 :=
by
  sorry

end members_not_playing_any_sport_l13_13621


namespace time_on_sideline_l13_13503

def total_game_time : ℕ := 90
def time_mark_played_first_period : ℕ := 20
def time_mark_played_second_period : ℕ := 35
def total_time_mark_played : ℕ := time_mark_played_first_period + time_mark_played_second_period

theorem time_on_sideline : total_game_time - total_time_mark_played = 35 := by
  sorry

end time_on_sideline_l13_13503


namespace coffee_shop_lattes_l13_13831

theorem coffee_shop_lattes (x : ℕ) (number_of_teas number_of_lattes : ℕ)
  (h1 : number_of_teas = 6)
  (h2 : number_of_lattes = 32)
  (h3 : number_of_lattes = x * number_of_teas + 8) :
  x = 4 :=
by
  sorry

end coffee_shop_lattes_l13_13831


namespace no_nat_solutions_for_m2_eq_n2_plus_2014_l13_13059

theorem no_nat_solutions_for_m2_eq_n2_plus_2014 :
  ∀ m n : ℕ, ¬(m^2 = n^2 + 2014) := by
sorry

end no_nat_solutions_for_m2_eq_n2_plus_2014_l13_13059


namespace intersection_of_A_and_B_l13_13329

def A : Set ℤ := { -3, -1, 0, 1 }
def B : Set ℤ := { x | (-2 < x) ∧ (x < 1) }

theorem intersection_of_A_and_B : A ∩ B = { -1, 0 } := by
  sorry

end intersection_of_A_and_B_l13_13329


namespace derivative_y_l13_13973

open Real

noncomputable def y (x : ℝ) : ℝ :=
  log (2 * x - 3 + sqrt (4 * x ^ 2 - 12 * x + 10)) -
  sqrt (4 * x ^ 2 - 12 * x + 10) * arctan (2 * x - 3)

theorem derivative_y (x : ℝ) : 
  (deriv y x) = - arctan (2 * x - 3) / sqrt (4 * x ^ 2 - 12 * x + 10) :=
by
  sorry

end derivative_y_l13_13973


namespace remaining_number_l13_13018

theorem remaining_number (S : Finset ℕ) (hS : S = Finset.range 51) :
  ∃ n ∈ S, n % 2 = 0 := 
sorry

end remaining_number_l13_13018


namespace combined_average_yield_l13_13203

theorem combined_average_yield (yield_A : ℝ) (price_A : ℝ) (yield_B : ℝ) (price_B : ℝ) (yield_C : ℝ) (price_C : ℝ) :
  yield_A = 0.20 → price_A = 100 → yield_B = 0.12 → price_B = 200 → yield_C = 0.25 → price_C = 300 →
  (yield_A * price_A + yield_B * price_B + yield_C * price_C) / (price_A + price_B + price_C) = 0.1983 :=
by
  intros hYA hPA hYB hPB hYC hPC
  sorry

end combined_average_yield_l13_13203


namespace fraction_simplification_l13_13981

theorem fraction_simplification (x y : ℚ) (hx : x = 4 / 6) (hy : y = 5 / 8) :
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  rw [hx, hy]
  sorry

end fraction_simplification_l13_13981


namespace central_angle_eq_one_l13_13867

noncomputable def radian_measure_of_sector (α r : ℝ) : Prop :=
  α * r = 2 ∧ (1 / 2) * α * r^2 = 2

-- Theorem stating the radian measure of the central angle is 1
theorem central_angle_eq_one (α r : ℝ) (h : radian_measure_of_sector α r) : α = 1 :=
by
  -- provide proof steps here
  sorry

end central_angle_eq_one_l13_13867


namespace expression_subtracted_from_3_pow_k_l13_13369

theorem expression_subtracted_from_3_pow_k (k : ℕ) (h : 15^k ∣ 759325) : 3^k - 0 = 1 :=
sorry

end expression_subtracted_from_3_pow_k_l13_13369


namespace plastic_bag_co2_release_l13_13527

def total_co2_canvas_bag_lb : ℕ := 600
def total_co2_canvas_bag_oz : ℕ := 9600
def plastic_bags_per_trip : ℕ := 8
def shopping_trips : ℕ := 300

theorem plastic_bag_co2_release :
  total_co2_canvas_bag_oz = 2400 * 4 :=
by
  sorry

end plastic_bag_co2_release_l13_13527


namespace perpendicular_condition_sufficient_but_not_necessary_l13_13022

theorem perpendicular_condition_sufficient_but_not_necessary (a : ℝ) :
  (a = -2) → ((∀ x y : ℝ, ax + (a + 1) * y + 1 = 0 → x + a * y + 2 = 0 ∧ (∃ t : ℝ, t ≠ 0 ∧ x = -t / (a + 1) ∧ y = (t / a))) →
  ¬ (a = -2) ∨ (a + 1 ≠ 0 ∧ ∃ k1 k2 : ℝ, k1 * k2 = -1 ∧ k1 = -a / (a + 1) ∧ k2 = -1 / a)) :=
by
  sorry

end perpendicular_condition_sufficient_but_not_necessary_l13_13022


namespace math_club_members_count_l13_13518

theorem math_club_members_count 
    (n_books : ℕ) 
    (n_borrow_each_member : ℕ) 
    (n_borrow_each_book : ℕ) 
    (total_borrow_count_books : n_books * n_borrow_each_book = 36) 
    (total_borrow_count_members : 2 * x = 36) 
    : x = 18 := 
by
  sorry

end math_club_members_count_l13_13518


namespace brownies_total_l13_13158

theorem brownies_total :
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  after_mooney_ate + additional_brownies = 36 :=
by
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  show after_mooney_ate + additional_brownies = 36
  sorry

end brownies_total_l13_13158


namespace sum_of_coefficients_is_7_l13_13000

noncomputable def v (n : ℕ) : ℕ := sorry

theorem sum_of_coefficients_is_7 : 
  (∀ n : ℕ, v (n + 1) - v n = 3 * n + 2) → (v 1 = 7) → (∃ a b c : ℝ, (a * n^2 + b * n + c = v n) ∧ (a + b + c = 7)) := 
by
  intros H1 H2
  sorry

end sum_of_coefficients_is_7_l13_13000
