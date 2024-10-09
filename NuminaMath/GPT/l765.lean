import Mathlib

namespace Irene_hours_worked_l765_76574

open Nat

theorem Irene_hours_worked (x totalHours : ℕ) : 
  (500 + 20 * x = 700) → 
  (totalHours = 40 + x) → 
  totalHours = 50 :=
by
  sorry

end Irene_hours_worked_l765_76574


namespace time_both_pipes_opened_l765_76523

def fill_rate_p := 1 / 10
def fill_rate_q := 1 / 15
def total_fill_rate := fill_rate_p + fill_rate_q -- Combined fill rate of both pipes

def remaining_fill_rate := 10 * fill_rate_q -- Fill rate of pipe q in 10 minutes

theorem time_both_pipes_opened (t : ℝ) :
  (t / 6) + (2 / 3) = 1 → t = 2 :=
by
  sorry

end time_both_pipes_opened_l765_76523


namespace calculate_power_of_fractions_l765_76541

-- Defining the fractions
def a : ℚ := 5 / 6
def b : ℚ := 3 / 5

-- The main statement to prove the given question
theorem calculate_power_of_fractions : a^3 + b^3 = (21457 : ℚ) / 27000 := by 
  sorry

end calculate_power_of_fractions_l765_76541


namespace range_of_a_l765_76528

theorem range_of_a (M : Set ℝ) (a : ℝ) :
  (M = {x | x^2 - 4 * x + 4 * a < 0}) →
  ¬(2 ∈ M) →
  (1 ≤ a) :=
by
  -- Given assumptions
  intros hM h2_notin_M
  -- Convert h2_notin_M to an inequality and prove the desired result
  sorry

end range_of_a_l765_76528


namespace boys_skip_count_l765_76561

theorem boys_skip_count 
  (x y : ℕ)
  (avg_jumps_boys : ℕ := 85)
  (avg_jumps_girls : ℕ := 92)
  (avg_jumps_all : ℕ := 88)
  (h1 : x = y + 10)
  (h2 : (85 * x + 92 * y) / (x + y) = 88) : x = 40 :=
  sorry

end boys_skip_count_l765_76561


namespace trigonometric_expression_evaluation_l765_76583

theorem trigonometric_expression_evaluation :
  (Real.cos (-585 * Real.pi / 180)) / 
  (Real.tan (495 * Real.pi / 180) + Real.sin (-690 * Real.pi / 180)) = Real.sqrt 2 :=
  sorry

end trigonometric_expression_evaluation_l765_76583


namespace total_chocolates_l765_76525

-- Definitions based on conditions
def chocolates_per_bag := 156
def number_of_bags := 20

-- Statement to prove
theorem total_chocolates : chocolates_per_bag * number_of_bags = 3120 :=
by
  -- skip the proof
  sorry

end total_chocolates_l765_76525


namespace curve_not_parabola_l765_76532

theorem curve_not_parabola (k : ℝ) : ¬(∃ (a b c d e f : ℝ), k * x^2 + y^2 = a * x^2 + b * x * y + c * y^2 + d * x + e * y + f ∧ b^2 = 4*a*c ∧ (a = 0 ∨ c = 0)) := sorry

end curve_not_parabola_l765_76532


namespace scientific_notation_chip_gate_width_l765_76567

theorem scientific_notation_chip_gate_width :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end scientific_notation_chip_gate_width_l765_76567


namespace parabola_vertex_position_l765_76509

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem parabola_vertex_position (x y : ℝ) :
  (∃ a b : ℝ, f a = y ∧ g b = y ∧ a = 1 ∧ b = -1)
  → (1 > -1) ∧ (f 1 > g (-1)) :=
by
  sorry

end parabola_vertex_position_l765_76509


namespace find_number_l765_76537

theorem find_number (x : ℝ) (h : 5020 - (1004 / x) = 4970) : x = 20.08 := 
by
  sorry

end find_number_l765_76537


namespace no_square_number_divisible_by_six_between_50_and_120_l765_76579

theorem no_square_number_divisible_by_six_between_50_and_120 :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n * n) ∧ (x % 6 = 0) ∧ (50 < x ∧ x < 120) := 
sorry

end no_square_number_divisible_by_six_between_50_and_120_l765_76579


namespace circle_radius_seven_l765_76542

theorem circle_radius_seven (k : ℝ) :
  (∃ x y : ℝ, (x^2 + 12 * x + y^2 + 8 * y - k = 0)) ↔ (k = -3) :=
by
  sorry

end circle_radius_seven_l765_76542


namespace number_of_valid_sequences_l765_76573

/--
The measures of the interior angles of a convex pentagon form an increasing arithmetic sequence.
Determine the number of such sequences possible if the pentagon is not equiangular, all of the angle
degree measures are positive integers less than 150 degrees, and the smallest angle is at least 60 degrees.
-/

theorem number_of_valid_sequences : ∃ n : ℕ, n = 5 ∧
  ∀ (x d : ℕ),
  x ≥ 60 ∧ x + 4 * d < 150 ∧ 5 * x + 10 * d = 540 ∧ (x + d ≠ x + 2 * d) := 
sorry

end number_of_valid_sequences_l765_76573


namespace inequality_correct_l765_76595

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c :=
sorry

end inequality_correct_l765_76595


namespace geometric_sequence_a3_l765_76511

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = a 1 * q ^ 3) (h2 : a 2 = a 1 * q) (h3 : a 5 = a 1 * q ^ 4) 
    (h4 : a 4 - a 2 = 6) (h5 : a 5 - a 1 = 15) : a 3 = 4 ∨ a 3 = -4 :=
by
  sorry

end geometric_sequence_a3_l765_76511


namespace sufficient_but_not_necessary_condition_l765_76548

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x^2 - 1 = 0 → x^3 - x = 0) ∧ ¬ (x^3 - x = 0 → x^2 - 1 = 0) := by
  sorry

end sufficient_but_not_necessary_condition_l765_76548


namespace minimum_value_of_f_at_zero_inequality_f_geq_term_l765_76585

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

theorem minimum_value_of_f_at_zero (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a y ≥ f a x ∧ f a x = 0) → a = 2 :=
by
  sorry

theorem inequality_f_geq_term (x : ℝ) (hx : x > 1) : 
  f 2 x ≥ 1 / x - Real.exp (1 - x) :=
by
  sorry

end minimum_value_of_f_at_zero_inequality_f_geq_term_l765_76585


namespace middle_rungs_widths_l765_76558

theorem middle_rungs_widths (a : ℕ → ℝ) (d : ℝ) :
  a 1 = 33 ∧ a 12 = 110 ∧ (∀ n, a (n + 1) = a n + 7) →
  (a 2 = 40 ∧ a 3 = 47 ∧ a 4 = 54 ∧ a 5 = 61 ∧
   a 6 = 68 ∧ a 7 = 75 ∧ a 8 = 82 ∧ a 9 = 89 ∧
   a 10 = 96 ∧ a 11 = 103) :=
by
  sorry

end middle_rungs_widths_l765_76558


namespace union_M_N_eq_N_l765_76570

def M := {x : ℝ | x^2 - 2 * x ≤ 0}
def N := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

theorem union_M_N_eq_N : M ∪ N = N := 
sorry

end union_M_N_eq_N_l765_76570


namespace no_infinite_pos_sequence_l765_76521

theorem no_infinite_pos_sequence (α : ℝ) (hα : 0 < α ∧ α < 1) :
  ¬(∃ a : ℕ → ℝ, (∀ n : ℕ, a n > 0) ∧ (∀ n : ℕ, 1 + a (n + 1) ≤ a n + (α / n) * a n)) :=
sorry

end no_infinite_pos_sequence_l765_76521


namespace cycle_selling_price_l765_76543

noncomputable def selling_price (cost_price : ℝ) (gain_percent : ℝ) : ℝ :=
  let gain_amount := (gain_percent / 100) * cost_price
  cost_price + gain_amount

theorem cycle_selling_price :
  selling_price 450 15.56 = 520.02 :=
by
  sorry

end cycle_selling_price_l765_76543


namespace prism_diagonal_length_l765_76544

theorem prism_diagonal_length (x y z : ℝ) (h1 : 4 * x + 4 * y + 4 * z = 24) (h2 : 2 * x * y + 2 * x * z + 2 * y * z = 11) : Real.sqrt (x^2 + y^2 + z^2) = 5 :=
  by
  sorry

end prism_diagonal_length_l765_76544


namespace fraction_addition_l765_76518

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 9 + 3 = (33 + 5 * d) / 9 := 
by 
  sorry

end fraction_addition_l765_76518


namespace arithmetic_sequence_formula_geometric_sequence_sum_l765_76598

variables {a_n S_n b_n T_n : ℕ → ℚ} {a_3 S_3 a_5 b_3 T_3 : ℚ} {q : ℚ}

def is_arithmetic_sequence (a_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, a_n n = a_1 + (n - 1) * d

def sum_first_n_arithmetic (S_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, S_n n = n / 2 * (2 * a_1 + (n - 1) * d)

def is_geometric_sequence (b_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, b_n n = b_1 * q^(n-1)

def sum_first_n_geometric (T_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, T_n n = if q = 1 then n * b_1 else b_1 * (1 - q^n) / (1 - q)

theorem arithmetic_sequence_formula {a_1 d : ℚ} (h_arith : is_arithmetic_sequence a_n a_1 d)
    (h_sum : sum_first_n_arithmetic S_n a_1 d) (h1 : a_n 3 = 5) (h2 : S_n 3 = 9) :
    ∀ n, a_n n = 2 * n - 1 := sorry

theorem geometric_sequence_sum {b_1 : ℚ} (h_geom : is_geometric_sequence b_n b_1 q)
    (h_sum : sum_first_n_geometric T_n b_1 q) (h3 : q > 0) (h4 : b_n 3 = a_n 5) (h5 : T_n 3 = 13) :
    ∀ n, T_n n = (3^n - 1) / 2 := sorry

end arithmetic_sequence_formula_geometric_sequence_sum_l765_76598


namespace arithmetic_sequence_solution_l765_76566

theorem arithmetic_sequence_solution :
  ∃ (a1 d : ℤ), 
    (a1 + 3*d + (a1 + 4*d) + (a1 + 5*d) + (a1 + 6*d) = 56) ∧
    ((a1 + 3*d) * (a1 + 6*d) = 187) ∧
    (
      (a1 = 5 ∧ d = 2) ∨
      (a1 = 23 ∧ d = -2)
    ) :=
by
  sorry

end arithmetic_sequence_solution_l765_76566


namespace sum_of_h_and_k_l765_76599

theorem sum_of_h_and_k (foci1 foci2 : ℝ × ℝ) (pt : ℝ × ℝ) (a b h k : ℝ) 
  (h_positive : a > 0) (b_positive : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = if (x, y) = pt then 1 else sorry)
  (foci_eq : foci1 = (1, 2) ∧ foci2 = (4, 2))
  (pt_eq : pt = (-1, 5)) :
  h + k = 4.5 :=
sorry

end sum_of_h_and_k_l765_76599


namespace nico_reads_wednesday_l765_76592

def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51
def pages_wednesday := total_pages - (pages_monday + pages_tuesday) 

theorem nico_reads_wednesday :
  pages_wednesday = 19 :=
by
  sorry

end nico_reads_wednesday_l765_76592


namespace total_number_of_people_l765_76514

-- Conditions
def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698

-- Theorem stating the total number of people is 803 given the conditions
theorem total_number_of_people : 
  number_of_parents + number_of_pupils = 803 :=
by
  sorry

end total_number_of_people_l765_76514


namespace proof_solution_l765_76545

noncomputable def proof_problem (x : ℝ) : Prop :=
  (⌈2 * x⌉₊ : ℝ) - (⌊2 * x⌋₊ : ℝ) = 0 → (⌈2 * x⌉₊ : ℝ) - 2 * x = 0

theorem proof_solution (x : ℝ) : proof_problem x :=
by
  sorry

end proof_solution_l765_76545


namespace problem_l765_76572

theorem problem
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 320) :
  x * y = 64 ∧ x^3 + y^3 = 4160 :=
by
  sorry

end problem_l765_76572


namespace annie_extracurricular_hours_l765_76510

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

end annie_extracurricular_hours_l765_76510


namespace maximum_value_of_M_l765_76590

noncomputable def M (x : ℝ) : ℝ :=
  (Real.sin x * (2 - Real.cos x)) / (5 - 4 * Real.cos x)

theorem maximum_value_of_M : 
  ∃ x : ℝ, M x = (Real.sqrt 3) / 4 :=
sorry

end maximum_value_of_M_l765_76590


namespace diameter_of_large_circle_l765_76522

-- Given conditions
def small_radius : ℝ := 3
def num_small_circles : ℕ := 6

-- Problem statement: Prove the diameter of the large circle
theorem diameter_of_large_circle (r : ℝ) (n : ℕ) (h_radius : r = small_radius) (h_num : n = num_small_circles) :
  ∃ (R : ℝ), R = 9 * 2 := 
sorry

end diameter_of_large_circle_l765_76522


namespace greatest_integer_difference_l765_76551

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) : y - x = 3 :=
sorry

end greatest_integer_difference_l765_76551


namespace percentage_reduction_in_price_l765_76577

-- Definitions for the conditions in the problem
def reduced_price_per_kg : ℕ := 30
def extra_oil_obtained_kg : ℕ := 10
def total_money_spent : ℕ := 1500

-- Definition of the original price per kg of oil
def original_price_per_kg : ℕ := 75

-- Statement to prove the percentage reduction
theorem percentage_reduction_in_price : 
  (original_price_per_kg - reduced_price_per_kg) * 100 / original_price_per_kg = 60 := by
  sorry

end percentage_reduction_in_price_l765_76577


namespace num_people_comparison_l765_76586

def num_people_1st_session (a : ℝ) : Prop := a > 0 -- Define the number for first session
def num_people_2nd_session (a : ℝ) : ℝ := 1.1 * a -- Define the number for second session
def num_people_3rd_session (a : ℝ) : ℝ := 0.99 * a -- Define the number for third session

theorem num_people_comparison (a b : ℝ) 
    (h1 : b = 0.99 * a): 
    a > b := 
by 
  -- insert the proof here
  sorry 

end num_people_comparison_l765_76586


namespace math_problem_l765_76587

variable (a a' b b' c c' : ℝ)

theorem math_problem 
  (h1 : a * a' > 0) 
  (h2 : a * c ≥ b * b) 
  (h3 : a' * c' ≥ b' * b') : 
  (a + a') * (c + c') ≥ (b + b') * (b + b') := 
by
  sorry

end math_problem_l765_76587


namespace average_disk_space_per_hour_l765_76500

theorem average_disk_space_per_hour :
  let days : ℕ := 15
  let total_mb : ℕ := 20000
  let hours_per_day : ℕ := 24
  let total_hours := days * hours_per_day
  total_mb / total_hours = 56 :=
by
  let days := 15
  let total_mb := 20000
  let hours_per_day := 24
  let total_hours := days * hours_per_day
  have h : total_mb / total_hours = 56 := sorry
  exact h

end average_disk_space_per_hour_l765_76500


namespace parking_savings_l765_76562

theorem parking_savings
  (weekly_rent : ℕ := 10)
  (monthly_rent : ℕ := 40)
  (weeks_in_year : ℕ := 52)
  (months_in_year : ℕ := 12)
  : weekly_rent * weeks_in_year - monthly_rent * months_in_year = 40 := 
by
  sorry

end parking_savings_l765_76562


namespace volleyballs_count_l765_76582

-- Definitions of sports item counts based on given conditions.
def soccer_balls := 20
def basketballs := soccer_balls + 5
def tennis_balls := 2 * soccer_balls
def baseballs := soccer_balls + 10
def hockey_pucks := tennis_balls / 2
def total_items := 180

-- Calculate the total number of known sports items.
def known_items_sum := soccer_balls + basketballs + tennis_balls + baseballs + hockey_pucks

-- Prove the number of volleyballs
theorem volleyballs_count : total_items - known_items_sum = 45 := by
  sorry

end volleyballs_count_l765_76582


namespace least_number_divisible_l765_76564

theorem least_number_divisible (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 9 = 4) (h3 : n % 18 = 4) : n = 130 := sorry

end least_number_divisible_l765_76564


namespace log_constant_expression_l765_76575

theorem log_constant_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hcond : x^2 + y^2 = 18 * x * y) :
  ∃ k : ℝ, (Real.log (x - y) / Real.log (Real.sqrt 2) - (1 / 2) * (Real.log x / Real.log (Real.sqrt 2) + Real.log y / Real.log (Real.sqrt 2))) = k :=
sorry

end log_constant_expression_l765_76575


namespace labor_cost_calculation_l765_76519

def num_men : Nat := 5
def num_women : Nat := 8
def num_boys : Nat := 10

def base_wage_man : Nat := 100
def base_wage_woman : Nat := 80
def base_wage_boy : Nat := 50

def efficiency_man_woman_ratio : Nat := 2
def efficiency_man_boy_ratio : Nat := 3

def overtime_rate_multiplier : Nat := 3 / 2 -- 1.5 as a ratio
def holiday_rate_multiplier : Nat := 2

def num_men_working_overtime : Nat := 3
def hours_worked_overtime : Nat := 10
def regular_workday_hours : Nat := 8

def is_holiday : Bool := true

theorem labor_cost_calculation : 
  (num_men * base_wage_man * holiday_rate_multiplier
    + num_women * base_wage_woman * holiday_rate_multiplier
    + num_boys * base_wage_boy * holiday_rate_multiplier
    + num_men_working_overtime * (hours_worked_overtime - regular_workday_hours) * (base_wage_man * overtime_rate_multiplier)) 
  = 4180 :=
by
  sorry

end labor_cost_calculation_l765_76519


namespace proof_intersection_complement_l765_76593

open Set

variable (U : Set ℝ) (A B : Set ℝ)

theorem proof_intersection_complement:
  U = univ ∧ A = {x | -1 < x ∧ x ≤ 5} ∧ B = {x | x < 2} →
  A ∩ (U \ B) = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  intros h
  rcases h with ⟨hU, hA, hB⟩
  simp [hU, hA, hB]
  sorry

end proof_intersection_complement_l765_76593


namespace number_of_new_students_l765_76556

theorem number_of_new_students (initial_students left_students final_students new_students : ℕ) 
  (h_initial : initial_students = 4) 
  (h_left : left_students = 3) 
  (h_final : final_students = 43) : 
  new_students = final_students - (initial_students - left_students) :=
by 
  sorry

end number_of_new_students_l765_76556


namespace inequality_l765_76571

variable {a b c : ℝ}

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  a * (a - 1) + b * (b - 1) + c * (c - 1) ≥ 0 := 
by 
  sorry

end inequality_l765_76571


namespace extreme_points_of_f_range_of_a_l765_76540

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≥ -1 then Real.log (x + 1) + a * (x^2 - x) 
  else 0

theorem extreme_points_of_f (a : ℝ) :
  (a < 0 → ∃ x, f a x = 0) ∧
  (0 ≤ a ∧ a ≤ 8/9 → ∃! x, f a x = 0) ∧
  (a > 8/9 → ∃ x₁ x₂, x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end extreme_points_of_f_range_of_a_l765_76540


namespace men_handshakes_l765_76501

theorem men_handshakes (n : ℕ) (h : n * (n - 1) / 2 = 435) : n = 30 :=
sorry

end men_handshakes_l765_76501


namespace ratio_x_w_l765_76563

variable {x y z w : ℕ}

theorem ratio_x_w (h1 : x / y = 24) (h2 : z / y = 8) (h3 : z / w = 1 / 12) : x / w = 1 / 4 := by
  sorry

end ratio_x_w_l765_76563


namespace min_value_of_expression_l765_76565

open Real

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h_perp : (x - 1) * 1 + 3 * y = 0) :
  ∃ (m : ℝ), m = 4 ∧ (∀ (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_ab_perp : (a - 1) * 1 + 3 * b = 0), (1 / a) + (1 / (3 * b)) ≥ m) :=
by
  use 4
  sorry

end min_value_of_expression_l765_76565


namespace age_difference_l765_76538

-- Define the present age of the son.
def S : ℕ := 22

-- Define the present age of the man.
variable (M : ℕ)

-- Given condition: In two years, the man's age will be twice the age of his son.
axiom condition : M + 2 = 2 * (S + 2)

-- Prove that the difference in present ages of the man and his son is 24 years.
theorem age_difference : M - S = 24 :=
by 
  -- We will fill in the proof here
  sorry

end age_difference_l765_76538


namespace chosen_number_l765_76533

theorem chosen_number (x : ℝ) (h1 : x / 9 - 100 = 10) : x = 990 :=
  sorry

end chosen_number_l765_76533


namespace total_ladybugs_l765_76524

theorem total_ladybugs (ladybugs_with_spots ladybugs_without_spots : ℕ) 
  (h1 : ladybugs_with_spots = 12170) 
  (h2 : ladybugs_without_spots = 54912) : 
  ladybugs_with_spots + ladybugs_without_spots = 67082 := 
by
  sorry

end total_ladybugs_l765_76524


namespace sum_of_g_of_nine_values_l765_76549

def f (x : ℝ) : ℝ := x^2 - 9 * x + 20
def g (y : ℝ) : ℝ := 3 * y - 4

theorem sum_of_g_of_nine_values : (g 9) = 19 := by
  sorry

end sum_of_g_of_nine_values_l765_76549


namespace polynomial_coeff_sum_l765_76578

theorem polynomial_coeff_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) (h : (2 * x - 3) ^ 5 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 160 :=
sorry

end polynomial_coeff_sum_l765_76578


namespace uncommon_card_cost_l765_76596

/--
Tom's deck contains 19 rare cards, 11 uncommon cards, and 30 common cards.
Each rare card costs $1.
Each common card costs $0.25.
The total cost of the deck is $32.
Prove that the cost of each uncommon card is $0.50.
-/
theorem uncommon_card_cost (x : ℝ): 
  let rare_count := 19
  let uncommon_count := 11
  let common_count := 30
  let rare_cost := 1
  let common_cost := 0.25
  let total_cost := 32
  (rare_count * rare_cost) + (common_count * common_cost) + (uncommon_count * x) = total_cost 
  → x = 0.5 :=
by
  sorry

end uncommon_card_cost_l765_76596


namespace employee_age_when_hired_l765_76513

theorem employee_age_when_hired
    (hire_year retire_year : ℕ)
    (rule_of_70 : ∀ A Y, A + Y = 70)
    (years_worked : ∀ hire_year retire_year, retire_year - hire_year = 19)
    (hire_year_eqn : hire_year = 1987)
    (retire_year_eqn : retire_year = 2006) :
  ∃ A : ℕ, A = 51 :=
by
  have Y := 19
  have A := 70 - Y
  use A
  sorry

end employee_age_when_hired_l765_76513


namespace max_log_expression_l765_76506

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem max_log_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x > y) :
  log_base x (x^2 / y^3) + log_base y (y^2 / x^3) = -2 :=
by
  sorry

end max_log_expression_l765_76506


namespace difference_quotient_correct_l765_76530

theorem difference_quotient_correct (a b : ℝ) :
  abs (3 * a - b) / abs (a + 2 * b) = abs (3 * a - b) / abs (a + 2 * b) :=
by
  sorry

end difference_quotient_correct_l765_76530


namespace number_of_sides_l765_76560

theorem number_of_sides (P l n : ℕ) (hP : P = 49) (hl : l = 7) (h : P = n * l) : n = 7 :=
by
  sorry

end number_of_sides_l765_76560


namespace most_cost_effective_way_cost_is_860_l765_76529

-- Definitions based on the problem conditions
def adult_cost := 150
def child_cost := 60
def group_cost_per_person := 100
def group_min_size := 5

-- Number of adults and children
def num_adults := 4
def num_children := 7

-- Calculate the total cost for the most cost-effective way
noncomputable def most_cost_effective_way_cost :=
  let group_tickets_count := 5  -- 4 adults + 1 child
  let remaining_children := num_children - 1
  group_tickets_count * group_cost_per_person + remaining_children * child_cost

-- Theorem to state the cost for the most cost-effective way
theorem most_cost_effective_way_cost_is_860 : most_cost_effective_way_cost = 860 := by
  sorry

end most_cost_effective_way_cost_is_860_l765_76529


namespace fixed_point_of_parabola_l765_76589

theorem fixed_point_of_parabola (s : ℝ) : ∃ y : ℝ, y = 4 * 3^2 + s * 3 - 3 * s ∧ (3, y) = (3, 36) :=
by
  sorry

end fixed_point_of_parabola_l765_76589


namespace intersection_eq_l765_76507

def M : Set ℝ := {x | ∃ y, y = Real.log (2 - x) / Real.log 3}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_eq : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_eq_l765_76507


namespace find_constants_PQR_l765_76534

theorem find_constants_PQR :
  ∃ P Q R : ℝ, 
    (6 * x + 2) / ((x - 4) * (x - 2) ^ 3) = P / (x - 4) + Q / (x - 2) + R / (x - 2) ^ 3 :=
by
  use 13 / 4
  use -6.5
  use -7
  sorry

end find_constants_PQR_l765_76534


namespace production_line_B_units_l765_76502

theorem production_line_B_units {x y z : ℕ} (h1 : x + y + z = 24000) (h2 : 2 * y = x + z) : y = 8000 :=
sorry

end production_line_B_units_l765_76502


namespace fourth_grade_students_l765_76516

theorem fourth_grade_students:
  (initial_students = 35) →
  (first_semester_left = 6) →
  (first_semester_joined = 4) →
  (first_semester_transfers = 2) →
  (second_semester_left = 3) →
  (second_semester_joined = 7) →
  (second_semester_transfers = 2) →
  final_students = initial_students - first_semester_left + first_semester_joined - second_semester_left + second_semester_joined :=
  sorry

end fourth_grade_students_l765_76516


namespace remaining_gift_card_value_correct_l765_76526

def initial_best_buy := 5
def initial_target := 3
def initial_walmart := 7
def initial_amazon := 2

def value_best_buy := 500
def value_target := 250
def value_walmart := 100
def value_amazon := 1000

def sent_best_buy := 1
def sent_walmart := 2
def sent_amazon := 1

def remaining_dollars : Nat :=
  (initial_best_buy - sent_best_buy) * value_best_buy +
  initial_target * value_target +
  (initial_walmart - sent_walmart) * value_walmart +
  (initial_amazon - sent_amazon) * value_amazon

theorem remaining_gift_card_value_correct : remaining_dollars = 4250 :=
  sorry

end remaining_gift_card_value_correct_l765_76526


namespace find_f2_l765_76594

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f2 (h : f a b (-2) = 2) : f a b 2 = 0 := by
  sorry

end find_f2_l765_76594


namespace trig_expression_value_l765_76520

open Real

theorem trig_expression_value : 
  (2 * cos (10 * (π / 180)) - sin (20 * (π / 180))) / cos (20 * (π / 180)) = sqrt 3 :=
by
  -- Proof should go here
  sorry

end trig_expression_value_l765_76520


namespace mason_hotdogs_proof_mason_ate_15_hotdogs_l765_76552

-- Define the weights of the items.
def weight_hotdog := 2 -- in ounces
def weight_burger := 5 -- in ounces
def weight_pie := 10 -- in ounces

-- Define Noah's consumption
def noah_burgers := 8

-- Define the total weight of hotdogs Mason ate
def mason_hotdogs_weight := 30

-- Calculate the number of hotdogs Mason ate
def hotdogs_mason_ate := mason_hotdogs_weight / weight_hotdog

-- Calculate the number of pies Jacob ate
def jacob_pies := noah_burgers - 3

-- Given conditions
theorem mason_hotdogs_proof :
  mason_hotdogs_weight / weight_hotdog = 3 * (noah_burgers - 3) :=
by
  sorry

-- Proving the number of hotdogs Mason ate equals 15
theorem mason_ate_15_hotdogs :
  hotdogs_mason_ate = 15 :=
by
  sorry

end mason_hotdogs_proof_mason_ate_15_hotdogs_l765_76552


namespace income_of_deceased_is_correct_l765_76581

-- Definitions based on conditions
def family_income_before_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def family_income_after_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def income_of_deceased (total_before: ℝ) (total_after: ℝ) : ℝ := total_before - total_after

-- Given conditions
def avg_income_before : ℝ := 782
def avg_income_after : ℝ := 650
def num_members_before : ℕ := 4
def num_members_after : ℕ := 3

-- Mathematical statement
theorem income_of_deceased_is_correct : 
  income_of_deceased (family_income_before_death avg_income_before num_members_before) 
                     (family_income_after_death avg_income_after num_members_after) = 1178 :=
by
  sorry

end income_of_deceased_is_correct_l765_76581


namespace remaining_cube_edge_length_l765_76550

theorem remaining_cube_edge_length (a b : ℕ) (h : a^3 = 98 + b^3) : b = 3 :=
sorry

end remaining_cube_edge_length_l765_76550


namespace simplify_and_evaluate_l765_76576

noncomputable def my_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m ^ 2 + 3 * m) / (m + 1))

theorem simplify_and_evaluate : my_expression (Real.sqrt 3) = 1 - Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l765_76576


namespace divisor_is_22_l765_76508

theorem divisor_is_22 (n d : ℤ) (h1 : n % d = 12) (h2 : (2 * n) % 11 = 2) : d = 22 :=
by
  sorry

end divisor_is_22_l765_76508


namespace sum_G_correct_l765_76553

def G (n : ℕ) : ℕ :=
  if n % 2 = 0 then n^2 + 1 else n^2

def sum_G (a b : ℕ) : ℕ :=
  List.sum (List.map G (List.range' a (b - a + 1)))

theorem sum_G_correct :
  sum_G 2 2007 = 8546520 := by
  sorry

end sum_G_correct_l765_76553


namespace students_in_lower_grades_l765_76580

noncomputable def seniors : ℕ := 300
noncomputable def percentage_cars_seniors : ℝ := 0.40
noncomputable def percentage_cars_remaining : ℝ := 0.10
noncomputable def total_percentage_cars : ℝ := 0.15

theorem students_in_lower_grades (X : ℝ) :
  (0.15 * (300 + X) = 120 + 0.10 * X) → X = 1500 :=
by
  intro h
  sorry

end students_in_lower_grades_l765_76580


namespace weight_problem_l765_76536

theorem weight_problem (w1 w2 w3 : ℝ) (h1 : w1 + w2 + w3 = 100)
  (h2 : w1 + 2 * w2 + w3 = 101) (h3 : w1 + w2 + 2 * w3 = 102) : 
  w1 ≥ 90 ∨ w2 ≥ 90 ∨ w3 ≥ 90 :=
by
  sorry

end weight_problem_l765_76536


namespace daria_amount_owed_l765_76597

variable (savings : ℝ)
variable (couch_price : ℝ)
variable (table_price : ℝ)
variable (lamp_price : ℝ)
variable (total_cost : ℝ)
variable (amount_owed : ℝ)

theorem daria_amount_owed (h_savings : savings = 500)
                          (h_couch : couch_price = 750)
                          (h_table : table_price = 100)
                          (h_lamp : lamp_price = 50)
                          (h_total_cost : total_cost = couch_price + table_price + lamp_price)
                          (h_amount_owed : amount_owed = total_cost - savings) :
                          amount_owed = 400 :=
by
  sorry

end daria_amount_owed_l765_76597


namespace count_multiples_5_or_7_but_not_35_l765_76591

def count_multiples (n d : ℕ) : ℕ :=
  n / d

def inclusion_exclusion (a b c : ℕ) : ℕ :=
  a + b - c

theorem count_multiples_5_or_7_but_not_35 : 
  count_multiples 3000 5 + count_multiples 3000 7 - count_multiples 3000 35 = 943 :=
by
  sorry

end count_multiples_5_or_7_but_not_35_l765_76591


namespace max_min_cos_sin_product_l765_76546

theorem max_min_cos_sin_product (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π / 12) (h4 : x + y + z = π / 2) :
  ∃ (maximum minimum : ℝ), maximum = (2 + Real.sqrt 3) / 8 ∧ minimum = 1 / 8 := by
  sorry

end max_min_cos_sin_product_l765_76546


namespace group_8_extracted_number_is_72_l765_76505

-- Definitions related to the problem setup
def individ_to_group (n : ℕ) : ℕ := n / 10 + 1
def unit_digit (n : ℕ) : ℕ := n % 10
def extraction_rule (k m : ℕ) : ℕ := (k + m - 1) % 10

-- Given condition: total individuals split into sequential groups and m = 5
def total_individuals : ℕ := 100
def total_groups : ℕ := 10
def m : ℕ := 5
def k_8 : ℕ := 8

-- The final theorem statement
theorem group_8_extracted_number_is_72 : ∃ n : ℕ, individ_to_group n = k_8 ∧ unit_digit n = extraction_rule k_8 m := by
  sorry

end group_8_extracted_number_is_72_l765_76505


namespace find_a_l765_76584

variable (a : ℝ)

def A := ({1, 2, a} : Set ℝ)
def B := ({1, a^2 - a} : Set ℝ)

theorem find_a (h : B a ⊆ A a) : a = -1 ∨ a = 0 :=
  sorry

end find_a_l765_76584


namespace integer_solution_for_equation_l765_76555

theorem integer_solution_for_equation :
  ∃ (M : ℤ), 14^2 * 35^2 = 10^2 * (M - 10)^2 ∧ M = 59 :=
by
  sorry

end integer_solution_for_equation_l765_76555


namespace find_d_l765_76512

theorem find_d 
  (d : ℝ)
  (d_gt_zero : d > 0)
  (line_eq : ∀ x : ℝ, (2 * x - 6 = 0) → x = 3)
  (y_intercept : ∀ y : ℝ, (2 * 0 - 6 = y) → y = -6)
  (area_condition : (1/2 * 3 * 6 = 9) → (1/2 * (d - 3) * (2 * d - 6) = 36)) :
  d = 9 :=
sorry

end find_d_l765_76512


namespace quadratic_interval_inequality_l765_76539

theorem quadratic_interval_inequality (a b c : ℝ) :
  (∀ x : ℝ, -1 / 2 < x ∧ x < 2 → a * x^2 + b * x + c > 0) →
  a < 0 ∧ c > 0 :=
sorry

end quadratic_interval_inequality_l765_76539


namespace remaining_balance_on_phone_card_l765_76527

theorem remaining_balance_on_phone_card (original_balance : ℝ) (cost_per_minute : ℝ) (call_duration : ℕ) :
  original_balance = 30 → cost_per_minute = 0.16 → call_duration = 22 →
  original_balance - (cost_per_minute * call_duration) = 26.48 :=
by
  intros
  sorry

end remaining_balance_on_phone_card_l765_76527


namespace avg_growth_rate_selling_price_reduction_l765_76531

open Real

-- Define the conditions for the first question
def sales_volume_aug : ℝ := 50000
def sales_volume_oct : ℝ := 72000

-- Define the conditions for the second question
def cost_price_per_unit : ℝ := 40
def initial_selling_price_per_unit : ℝ := 80
def initial_sales_volume_per_day : ℝ := 20
def additional_units_per_half_dollar_decrease : ℝ := 4
def desired_daily_profit : ℝ := 1400

-- First proof: monthly average growth rate
theorem avg_growth_rate (x : ℝ) :
  sales_volume_aug * (1 + x)^2 = sales_volume_oct → x = 0.2 :=
by {
  sorry
}

-- Second proof: reduction in selling price for daily profit
theorem selling_price_reduction (y : ℝ) :
  (initial_selling_price_per_unit - y - cost_price_per_unit) * (initial_sales_volume_per_day + additional_units_per_half_dollar_decrease * y / 0.5) = desired_daily_profit → y = 30 :=
by {
  sorry
}

end avg_growth_rate_selling_price_reduction_l765_76531


namespace students_not_taking_either_l765_76557

-- Definitions of the conditions
def total_students : ℕ := 28
def students_taking_french : ℕ := 5
def students_taking_spanish : ℕ := 10
def students_taking_both : ℕ := 4

-- Theorem stating the mathematical problem
theorem students_not_taking_either :
  total_students - (students_taking_french + students_taking_spanish + students_taking_both) = 9 :=
sorry

end students_not_taking_either_l765_76557


namespace complex_problem_l765_76569

theorem complex_problem (z : ℂ) (h : (i * z + z) = 2) : z = 1 - i :=
sorry

end complex_problem_l765_76569


namespace molecular_weight_of_one_mole_l765_76517

theorem molecular_weight_of_one_mole (molecular_weight_3_moles : ℕ) (h : molecular_weight_3_moles = 222) : (molecular_weight_3_moles / 3) = 74 := 
by
  sorry

end molecular_weight_of_one_mole_l765_76517


namespace min_value_of_quadratic_expression_l765_76554

theorem min_value_of_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, x^2 + 6 * x + 3 ≤ y) ∧ x^2 + 6 * x + 3 = -6 :=
sorry

end min_value_of_quadratic_expression_l765_76554


namespace complement_A_union_B_range_of_m_l765_76588

def setA : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.sqrt (x^2 - 5*x - 14) }
def setB : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (-x^2 - 7*x - 12) }
def setC (m : ℝ) : Set ℝ := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

theorem complement_A_union_B :
  (A ∪ B)ᶜ = Set.Ioo (-2 : ℝ) 7 :=
sorry

theorem range_of_m (m : ℝ) :
  (A ∪ setC m = A) → (m < 2 ∨ m ≥ 6) :=
sorry

end complement_A_union_B_range_of_m_l765_76588


namespace area_of_rectangle_EFGH_l765_76535

theorem area_of_rectangle_EFGH :
  ∀ (a b c : ℕ), 
    a = 7 → 
    b = 3 * a → 
    c = 2 * a → 
    (area : ℕ) = b * c → 
    area = 294 := 
by
  sorry

end area_of_rectangle_EFGH_l765_76535


namespace sqrt_36_eq_6_l765_76568

theorem sqrt_36_eq_6 : Real.sqrt 36 = 6 := by
  sorry

end sqrt_36_eq_6_l765_76568


namespace functional_inequality_solution_l765_76503

theorem functional_inequality_solution {f : ℝ → ℝ} 
  (h : ∀ x y : ℝ, f (x * y) ≤ y * f (x) + f (y)) : 
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_inequality_solution_l765_76503


namespace jugglers_count_l765_76547

-- Define the conditions
def num_balls_each_juggler := 6
def total_balls := 2268

-- Define the theorem to prove the number of jugglers
theorem jugglers_count : (total_balls / num_balls_each_juggler) = 378 :=
by
  sorry

end jugglers_count_l765_76547


namespace bike_travel_distance_l765_76504

-- Declaring the conditions as definitions
def speed : ℝ := 50 -- Speed in meters per second
def time : ℝ := 7 -- Time in seconds

-- Declaring the question and expected answer
def expected_distance : ℝ := 350 -- Expected distance in meters

-- The proof statement that needs to be proved
theorem bike_travel_distance : (speed * time = expected_distance) :=
by
  sorry

end bike_travel_distance_l765_76504


namespace average_salary_of_laborers_l765_76559

-- Define the main statement as a theorem
theorem average_salary_of_laborers 
  (total_workers : ℕ)
  (total_salary_all : ℕ)
  (supervisors : ℕ)
  (supervisor_salary : ℕ)
  (laborers : ℕ)
  (expected_laborer_salary : ℝ) :
  total_workers = 48 → 
  total_salary_all = 60000 →
  supervisors = 6 →
  supervisor_salary = 2450 →
  laborers = 42 →
  expected_laborer_salary = 1078.57 :=
sorry

end average_salary_of_laborers_l765_76559


namespace problem1_problem2_problem3_problem4_l765_76515

-- Problem 1
theorem problem1 : (2 / 19) * (8 / 25) + (17 / 25) / (19 / 2) = 2 / 19 := 
by sorry

-- Problem 2
theorem problem2 : (1 / 4) * 125 * (1 / 25) * 8 = 10 := 
by sorry

-- Problem 3
theorem problem3 : ((1 / 3) + (1 / 4)) / ((1 / 2) - (1 / 3)) = 7 / 2 := 
by sorry

-- Problem 4
theorem problem4 : ((1 / 6) + (1 / 8)) * 24 * (1 / 9) = 7 / 9 := 
by sorry

end problem1_problem2_problem3_problem4_l765_76515
