import Mathlib

namespace min_AC_plus_BD_l1595_159580

theorem min_AC_plus_BD (k : ℝ) (h : k ≠ 0) :
  (8 + 8 / k^2) + (8 + 2 * k^2) ≥ 24 :=
by
  sorry -- skipping the proof

end min_AC_plus_BD_l1595_159580


namespace triangle_side_lengths_l1595_159591

theorem triangle_side_lengths (a b c r : ℕ) (h : a / b / c = 25 / 29 / 36) (hinradius : r = 232) :
  (a = 725 ∧ b = 841 ∧ c = 1044) :=
by
  sorry

end triangle_side_lengths_l1595_159591


namespace geometric_sequence_product_l1595_159585

-- Defining the geometric sequence and the equation
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def satisfies_quadratic_roots (a : ℕ → ℝ) : Prop :=
  (a 2 = -1 ∧ a 18 = -16 / (-1 + 16 / -1) ∨
  a 18 = -1 ∧ a 2 = -16 / (-1 + 16 / -1))

-- Problem statement
theorem geometric_sequence_product (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : satisfies_quadratic_roots a) : 
  a 3 * a 10 * a 17 = -64 :=
sorry

end geometric_sequence_product_l1595_159585


namespace arccos_sin_three_l1595_159524

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end arccos_sin_three_l1595_159524


namespace total_cars_at_end_of_play_l1595_159542

def carsInFront : ℕ := 100
def carsInBack : ℕ := 2 * carsInFront
def additionalCars : ℕ := 300

theorem total_cars_at_end_of_play : carsInFront + carsInBack + additionalCars = 600 := by
  sorry

end total_cars_at_end_of_play_l1595_159542


namespace solve_equation_solve_inequality_system_l1595_159567

theorem solve_equation (x : ℝ) : x^2 - 2 * x - 4 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 :=
by
  sorry

theorem solve_inequality_system (x : ℝ) : (4 * (x - 1) < x + 2) ∧ ((x + 7) / 3 > x) ↔ x < 2 :=
by
  sorry

end solve_equation_solve_inequality_system_l1595_159567


namespace find_m_l1595_159578

theorem find_m (m : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) : m = 2 :=
sorry

end find_m_l1595_159578


namespace no_equal_prob_for_same_color_socks_l1595_159534

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l1595_159534


namespace intervals_equinumerous_l1595_159592

-- Definitions and statements
theorem intervals_equinumerous (a : ℝ) (h : 0 < a) : 
  ∃ (f : Set.Icc 0 1 → Set.Icc 0 a), Function.Bijective f :=
by
  sorry

end intervals_equinumerous_l1595_159592


namespace tank_loss_rate_after_first_repair_l1595_159570

def initial_capacity : ℕ := 350000
def first_loss_rate : ℕ := 32000
def first_loss_duration : ℕ := 5
def second_loss_duration : ℕ := 10
def filling_rate : ℕ := 40000
def filling_duration : ℕ := 3
def missing_gallons : ℕ := 140000

noncomputable def first_repair_loss_rate := (initial_capacity - (first_loss_rate * first_loss_duration) + (filling_rate * filling_duration) - (initial_capacity - missing_gallons)) / second_loss_duration

theorem tank_loss_rate_after_first_repair : first_repair_loss_rate = 10000 := by sorry

end tank_loss_rate_after_first_repair_l1595_159570


namespace molecular_weight_compound_l1595_159547

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def molecular_weight (n_H n_Br n_O : ℕ) : ℝ :=
  n_H * atomic_weight_H + n_Br * atomic_weight_Br + n_O * atomic_weight_O

theorem molecular_weight_compound : 
  molecular_weight 1 1 3 = 128.91 :=
by
  -- This is where the proof would go
  sorry

end molecular_weight_compound_l1595_159547


namespace savings_after_increase_l1595_159535

/-- A man saves 20% of his monthly salary. If on account of dearness of things
    he is to increase his monthly expenses by 20%, he is only able to save a
    certain amount per month. His monthly salary is Rs. 6250. -/
theorem savings_after_increase (monthly_salary : ℝ) (initial_savings_percentage : ℝ)
  (increase_expenses_percentage : ℝ) (final_savings : ℝ) :
  monthly_salary = 6250 ∧
  initial_savings_percentage = 0.20 ∧
  increase_expenses_percentage = 0.20 →
  final_savings = 250 :=
by
  sorry

end savings_after_increase_l1595_159535


namespace intersection_of_A_and_B_l1595_159584

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_of_A_and_B : A ∩ B = {-2, 2} :=
by
  sorry

end intersection_of_A_and_B_l1595_159584


namespace reflected_ray_eqn_l1595_159546

theorem reflected_ray_eqn : 
  ∃ a b c : ℝ, (∀ x y : ℝ, 2 * x - y + 5 = 0 → (a * x + b * y + c = 0)) → -- Condition for the line
  (∀ x y : ℝ, x = 1 ∧ y = 3 → (a * x + b * y + c = 0)) → -- Condition for point (1, 3)
  (a = 1 ∧ b = -5 ∧ c = 14) := -- Assertion about the line equation
by
  sorry

end reflected_ray_eqn_l1595_159546


namespace variable_v_value_l1595_159506

theorem variable_v_value (w x v : ℝ) (h1 : 2 / w + 2 / x = 2 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) :
  v = 0.25 :=
sorry

end variable_v_value_l1595_159506


namespace valve_XY_time_correct_l1595_159574

-- Given conditions
def valve_rates (x y z : ℝ) := (x + y + z = 1/2 ∧ x + z = 1/4 ∧ y + z = 1/3)
def total_fill_time (t : ℝ) (x y : ℝ) := t = 1 / (x + y)

-- The proof problem
theorem valve_XY_time_correct (x y z : ℝ) (t : ℝ) 
  (h : valve_rates x y z) : total_fill_time t x y → t = 2.4 :=
by
  -- Assume h defines the rates
  have h1 : x + y + z = 1/2 := h.1
  have h2 : x + z = 1/4 := h.2.1
  have h3 : y + z = 1/3 := h.2.2
  
  sorry

end valve_XY_time_correct_l1595_159574


namespace reciprocal_inequality_pos_reciprocal_inequality_neg_l1595_159508

theorem reciprocal_inequality_pos {a b : ℝ} (h : a < b) (ha : 0 < a) : (1 / a) > (1 / b) :=
sorry

theorem reciprocal_inequality_neg {a b : ℝ} (h : a < b) (hb : b < 0) : (1 / a) < (1 / b) :=
sorry

end reciprocal_inequality_pos_reciprocal_inequality_neg_l1595_159508


namespace problem_l1595_159521

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

axiom universal_set : U = {1, 2, 3, 4, 5, 6, 7}
axiom set_M : M = {3, 4, 5}
axiom set_N : N = {1, 3, 6}

def complement (U M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

theorem problem :
  {1, 6} = (complement U M) ∩ N :=
by
  sorry

end problem_l1595_159521


namespace sum_of_ages_is_220_l1595_159561

-- Definitions based on the conditions
def father_age (S : ℕ) := (7 * S) / 4
def sum_ages (F S : ℕ) := F + S

-- The proof statement
theorem sum_of_ages_is_220 (F S : ℕ) (h1 : 4 * F = 7 * S)
  (h2 : 3 * (F + 10) = 5 * (S + 10)) : sum_ages F S = 220 :=
by
  sorry

end sum_of_ages_is_220_l1595_159561


namespace score_recording_l1595_159545

theorem score_recording (avg : ℤ) (h : avg = 0) : 
  (9 = avg + 9) ∧ (-18 = avg - 18) ∧ (-2 = avg - 2) :=
by
  -- Proof steps go here
  sorry

end score_recording_l1595_159545


namespace find_n_modulo_l1595_159512

theorem find_n_modulo :
  ∀ n : ℤ, (0 ≤ n ∧ n < 25 ∧ -175 % 25 = n % 25) → n = 0 :=
by
  intros n h
  sorry

end find_n_modulo_l1595_159512


namespace intersection_of_M_and_complement_N_l1595_159530

def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | 2 * x < 2 }
def complement_N : Set ℝ := { x | x ≥ 1 }

theorem intersection_of_M_and_complement_N : M ∩ complement_N = { x | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_of_M_and_complement_N_l1595_159530


namespace total_eyes_correct_l1595_159513

-- Conditions
def boys := 21 * 2 + 2 * 1
def girls := 15 * 2 + 3 * 1
def cats := 8 * 2 + 2 * 1
def spiders := 4 * 8 + 1 * 6

-- Total count of eyes
def total_eyes := boys + girls + cats + spiders

theorem total_eyes_correct: total_eyes = 133 :=
by 
  -- Here the proof steps would go, which we are skipping
  sorry

end total_eyes_correct_l1595_159513


namespace first_solution_carbonation_l1595_159573

-- Definitions of given conditions in the problem
variable (C : ℝ) -- Percentage of carbonated water in the first solution
variable (L : ℝ) -- Percentage of lemonade in the first solution

-- The second solution is 55% carbonated water and 45% lemonade
def second_solution_carbonated : ℝ := 55
def second_solution_lemonade : ℝ := 45

-- The mixture is 65% carbonated water and 40% of the volume is the first solution
def mixture_carbonated : ℝ := 65
def first_solution_contribution : ℝ := 0.40
def second_solution_contribution : ℝ := 0.60

-- The relationship between the solution components
def equation := first_solution_contribution * C + second_solution_contribution * second_solution_carbonated = mixture_carbonated

-- The statement to prove: C = 80
theorem first_solution_carbonation :
  equation C →
  C = 80 :=
sorry

end first_solution_carbonation_l1595_159573


namespace geometric_progression_condition_l1595_159559

theorem geometric_progression_condition
  (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0)
  (a_seq : ℕ → ℝ) 
  (h_def : ∀ n, a_seq (n+2) = k * a_seq n * a_seq (n+1)) :
  (a_seq 1 = a ∧ a_seq 2 = b) ↔ a_seq 1 = a_seq 2 :=
by
  sorry

end geometric_progression_condition_l1595_159559


namespace vehicle_flow_mod_15_l1595_159563

theorem vehicle_flow_mod_15
  (vehicle_length : ℝ := 5)
  (max_speed : ℕ := 100)
  (speed_interval : ℕ := 10)
  (distance_multiplier : ℕ := 10)
  (N : ℕ := 2000) :
  (N % 15) = 5 := 
sorry

end vehicle_flow_mod_15_l1595_159563


namespace Tony_temp_above_fever_threshold_l1595_159562

def normal_temp : ℕ := 95
def illness_A : ℕ := 10
def illness_B : ℕ := 4
def illness_C : Int := -2
def fever_threshold : ℕ := 100

theorem Tony_temp_above_fever_threshold :
  let T := normal_temp + illness_A + illness_B + illness_C
  T = 107 ∧ (T - fever_threshold) = 7 := by
  -- conditions
  let t_0 := normal_temp
  let T_A := illness_A
  let T_B := illness_B
  let T_C := illness_C
  let F := fever_threshold
  -- calculations
  let T := t_0 + T_A + T_B + T_C
  show T = 107 ∧ (T - F) = 7
  sorry

end Tony_temp_above_fever_threshold_l1595_159562


namespace intersection_M_N_l1595_159533

def M : Set ℝ := { x | Real.exp (x - 1) > 1 }
def N : Set ℝ := { x | x^2 - 2*x - 3 < 0 }

theorem intersection_M_N :
  (M ∩ N : Set ℝ) = { x | 1 < x ∧ x < 3 } := 
by
  sorry

end intersection_M_N_l1595_159533


namespace solve_problems_l1595_159557

theorem solve_problems (x y : ℕ) (hx : x + y = 14) (hy : 7 * x - 12 * y = 60) : x = 12 :=
sorry

end solve_problems_l1595_159557


namespace expand_and_simplify_fraction_l1595_159572

theorem expand_and_simplify_fraction (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * ((7 / (x^2)) + 15 * (x^3) - 4 * x) = (3 / (x^2)) + (45 * (x^3) / 7) - (12 * x / 7) :=
by
  sorry

end expand_and_simplify_fraction_l1595_159572


namespace log_sum_exp_log_sub_l1595_159596

theorem log_sum : Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 = 1 := 
by sorry

theorem exp_log_sub : Real.exp (Real.log 3 / Real.log 2 * Real.log 2) - Real.exp (Real.log 8 / 3) = 1 := 
by sorry

end log_sum_exp_log_sub_l1595_159596


namespace metal_detector_time_on_less_crowded_days_l1595_159556

variable (find_parking_time walk_time crowded_metal_detector_time total_time_per_week : ℕ)
variable (week_days crowded_days less_crowded_days : ℕ)

theorem metal_detector_time_on_less_crowded_days
  (h1 : find_parking_time = 5)
  (h2 : walk_time = 3)
  (h3 : crowded_metal_detector_time = 30)
  (h4 : total_time_per_week = 130)
  (h5 : week_days = 5)
  (h6 : crowded_days = 2)
  (h7 : less_crowded_days = 3) :
  (total_time_per_week = (find_parking_time * week_days) + (walk_time * week_days) + (crowded_metal_detector_time * crowded_days) + (10 * less_crowded_days)) :=
sorry

end metal_detector_time_on_less_crowded_days_l1595_159556


namespace standard_deviations_below_l1595_159595

variable (σ : ℝ)
variable (mean : ℝ)
variable (score98 : ℝ)
variable (score58 : ℝ)

-- Conditions translated to Lean definitions
def condition_1 : Prop := score98 = mean + 3 * σ
def condition_2 : Prop := mean = 74
def condition_3 : Prop := σ = 8

-- Target statement: Prove that the score of 58 is 2 standard deviations below the mean
theorem standard_deviations_below : condition_1 σ mean score98 → condition_2 mean → condition_3 σ → score58 = 74 - 2 * σ :=
by
  intro h1 h2 h3
  sorry

end standard_deviations_below_l1595_159595


namespace average_cost_is_70_l1595_159517

noncomputable def C_before_gratuity (total_bill : ℝ) (gratuity_rate : ℝ) : ℝ :=
  total_bill / (1 + gratuity_rate)

noncomputable def average_cost_per_individual (C : ℝ) (total_people : ℝ) : ℝ :=
  C / total_people

theorem average_cost_is_70 :
  let total_bill := 756
  let gratuity_rate := 0.20
  let total_people := 9
  average_cost_per_individual (C_before_gratuity total_bill gratuity_rate) total_people = 70 :=
by
  sorry

end average_cost_is_70_l1595_159517


namespace minimum_guests_l1595_159550

-- Define the conditions as variables
def total_food : ℕ := 4875
def max_food_per_guest : ℕ := 3

-- Define the theorem we need to prove
theorem minimum_guests : ∃ g : ℕ, g * max_food_per_guest = total_food ∧ g >= 1625 := by
  sorry

end minimum_guests_l1595_159550


namespace sum_m_b_eq_neg_five_halves_l1595_159588

theorem sum_m_b_eq_neg_five_halves : 
  let x1 := 1 / 2
  let y1 := -1
  let x2 := -1 / 2
  let y2 := 2
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = -5 / 2 :=
by 
  sorry

end sum_m_b_eq_neg_five_halves_l1595_159588


namespace must_divide_a_l1595_159509

-- Definitions of positive integers and their gcd conditions
variables {a b c d : ℕ}

-- The conditions given in the problem
axiom h1 : gcd a b = 24
axiom h2 : gcd b c = 36
axiom h3 : gcd c d = 54
axiom h4 : 70 < gcd d a ∧ gcd d a < 100

-- We need to prove that 13 divides a
theorem must_divide_a : 13 ∣ a :=
by sorry

end must_divide_a_l1595_159509


namespace parity_of_f_find_a_l1595_159581

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x + a * Real.exp (-x)

theorem parity_of_f (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a ↔ a = 1 ∨ a = -1) ∧
  (∀ x : ℝ, f (-x) a = -f x a ↔ a = -1) ∧
  (∀ x : ℝ, ¬(f (-x) a = f x a) ∧ ¬(f (-x) a = -f x a) ↔ ¬(a = 1 ∨ a = -1)) :=
by
  sorry

theorem find_a (h : ∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), f x a ≥ f 0 a) : 
  a = 1 :=
by
  sorry

end parity_of_f_find_a_l1595_159581


namespace g_neg_9_equiv_78_l1595_159523

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (y : ℝ) : ℝ := 3 * (y / 2 - 3 / 2)^2 + 4 * (y / 2 - 3 / 2) - 6

theorem g_neg_9_equiv_78 : g (-9) = 78 := by
  sorry

end g_neg_9_equiv_78_l1595_159523


namespace calories_per_strawberry_l1595_159515

theorem calories_per_strawberry (x : ℕ) :
  (12 * x + 6 * 17 = 150) → x = 4 := by
  sorry

end calories_per_strawberry_l1595_159515


namespace platform_length_l1595_159576

theorem platform_length (speed_kmh : ℕ) (time_min : ℕ) (train_length_m : ℕ) (distance_covered_m : ℕ) : 
  speed_kmh = 90 → time_min = 1 → train_length_m = 750 → distance_covered_m = 1500 →
  train_length_m + (distance_covered_m - train_length_m) = 750 + (1500 - 750) :=
by sorry

end platform_length_l1595_159576


namespace sheets_bought_l1595_159565

variable (x y : ℕ)

-- Conditions based on the problem statement
def A_condition (x y : ℕ) : Prop := x + 40 = y
def B_condition (x y : ℕ) : Prop := 3 * x + 40 = y

-- Proven that if these conditions are met, then the number of sheets of stationery bought by A and B is 120
theorem sheets_bought (x y : ℕ) (hA : A_condition x y) (hB : B_condition x y) : y = 120 :=
by
  sorry

end sheets_bought_l1595_159565


namespace ratio_of_sopranos_to_altos_l1595_159511

theorem ratio_of_sopranos_to_altos (S A : ℕ) :
  (10 = 5 * S) ∧ (15 = 5 * A) → (S : ℚ) / (A : ℚ) = 2 / 3 :=
by sorry

end ratio_of_sopranos_to_altos_l1595_159511


namespace students_in_only_one_subject_l1595_159516

variables (A B C : ℕ) 
variables (A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ)

def students_in_one_subject (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ) : ℕ :=
  A + B + C - A_inter_B - A_inter_C - B_inter_C + A_inter_B_inter_C - 2 * A_inter_B_inter_C

theorem students_in_only_one_subject :
  ∀ (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ),
    A = 29 →
    B = 28 →
    C = 27 →
    A_inter_B = 13 →
    A_inter_C = 12 →
    B_inter_C = 11 →
    A_inter_B_inter_C = 5 →
    students_in_one_subject A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C = 27 :=
by
  intros A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C hA hB hC hAB hAC hBC hABC
  unfold students_in_one_subject
  rw [hA, hB, hC, hAB, hAC, hBC, hABC]
  norm_num
  sorry

end students_in_only_one_subject_l1595_159516


namespace triangle_sum_is_16_l1595_159548

-- Definition of the triangle operation
def triangle (a b c : ℕ) : ℕ := a * b - c

-- Lean theorem statement
theorem triangle_sum_is_16 : 
  triangle 2 4 3 + triangle 3 6 7 = 16 := 
by 
  sorry

end triangle_sum_is_16_l1595_159548


namespace linear_equation_a_neg2_l1595_159525

theorem linear_equation_a_neg2 (a : ℝ) :
  (∃ x y : ℝ, (a - 2) * x ^ (|a| - 1) + 3 * y = 1) ∧
  (∀ x : ℝ, x ≠ 0 → x ^ (|a| - 1) ≠ 1) →
  a = -2 :=
by
  sorry

end linear_equation_a_neg2_l1595_159525


namespace part1_part2_l1595_159505

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part1 (tangent_at_e : ∀ x : ℝ, f x e = 2 * e) : a = e := sorry

theorem part2 (m : ℝ) (a : ℝ) (hm : 0 < m) :
  (if m ≤ 1 / (2 * Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (2 * m) a 
   else if 1 / (2 * Real.exp 1) < m ∧ m < 1 / (Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (1 / (Real.exp 1)) a 
   else 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f m a) :=
  sorry

end part1_part2_l1595_159505


namespace height_of_picture_frame_l1595_159590

-- Definitions of lengths and perimeter
def length : ℕ := 10
def perimeter : ℕ := 44

-- Perimeter formula for a rectangle
def rectangle_perimeter (L H : ℕ) : ℕ := 2 * (L + H)

-- Theorem statement: Proving the height is 12 inches based on given conditions
theorem height_of_picture_frame : ∃ H : ℕ, rectangle_perimeter length H = perimeter ∧ H = 12 := by
  sorry

end height_of_picture_frame_l1595_159590


namespace machine_purchase_price_l1595_159543

theorem machine_purchase_price (P : ℝ) (h : 0.80 * P = 6400) : P = 8000 :=
by
  sorry

end machine_purchase_price_l1595_159543


namespace find_number_l1595_159594

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
sorry

end find_number_l1595_159594


namespace sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l1595_159555

theorem sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : 100^2 + 1^2 = p * q ∧ 65^2 + 76^2 = p * q) : p + q = 210 := 
sorry

end sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l1595_159555


namespace orthogonal_pairs_in_cube_is_36_l1595_159571

-- Define a cube based on its properties, i.e., having vertices, edges, and faces.
structure Cube :=
(vertices : Fin 8 → Fin 3)
(edges : Fin 12 → (Fin 2 → Fin 8))
(faces : Fin 6 → (Fin 4 → Fin 8))

-- Define orthogonal pairs of a cube as an axiom.
axiom orthogonal_line_plane_pairs (c : Cube) : ℕ

-- The main theorem stating the problem's conclusion.
theorem orthogonal_pairs_in_cube_is_36 (c : Cube): orthogonal_line_plane_pairs c = 36 :=
by { sorry }

end orthogonal_pairs_in_cube_is_36_l1595_159571


namespace x_zero_sufficient_not_necessary_for_sin_zero_l1595_159500

theorem x_zero_sufficient_not_necessary_for_sin_zero :
  (∀ x : ℝ, x = 0 → Real.sin x = 0) ∧ (∃ y : ℝ, Real.sin y = 0 ∧ y ≠ 0) :=
by
  sorry

end x_zero_sufficient_not_necessary_for_sin_zero_l1595_159500


namespace candle_burning_time_l1595_159502

theorem candle_burning_time :
  ∃ T : ℝ, 
    (∀ T, 0 ≤ T ∧ T ≤ 4 → thin_candle_length = 24 - 6 * T) ∧
    (∀ T, 0 ≤ T ∧ T ≤ 6 → thick_candle_length = 24 - 4 * T) ∧
    (2 * (24 - 6 * T) = 24 - 4 * T) →
    T = 3 :=
by
  sorry

end candle_burning_time_l1595_159502


namespace find_cost_price_l1595_159507

theorem find_cost_price (C : ℝ) (h1 : C * 1.05 = C + 0.05 * C)
  (h2 : 0.95 * C = C - 0.05 * C)
  (h3 : 1.05 * C - 4 = 1.045 * C) :
  C = 800 := sorry

end find_cost_price_l1595_159507


namespace sum_of_three_digit_numbers_l1595_159538

theorem sum_of_three_digit_numbers :
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  let Sum := n / 2 * (first_term + last_term)
  Sum = 494550 :=
by {
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  have n_def : n = 900 := by norm_num [n]
  let Sum := n / 2 * (first_term + last_term)
  have sum_def : Sum = 450 * (100 + 999) := by norm_num [Sum, first_term, last_term, n_def]
  have final_sum : Sum = 494550 := by norm_num [sum_def]
  exact final_sum
}

end sum_of_three_digit_numbers_l1595_159538


namespace cube_surface_area_example_l1595_159586

def cube_surface_area (V : ℝ) (S : ℝ) : Prop :=
  (∃ s : ℝ, s ^ 3 = V ∧ S = 6 * s ^ 2)

theorem cube_surface_area_example : cube_surface_area 8 24 :=
by
  sorry

end cube_surface_area_example_l1595_159586


namespace compute_product_l1595_159501

theorem compute_product : (100 - 5) * (100 + 5) = 9975 := by
  sorry

end compute_product_l1595_159501


namespace part_a_part_b_part_c_l1595_159510

-- Define the conditions
inductive Color
| blue
| red
| green
| yellow

-- Each square can be painted in one of the colors: blue, red, or green.
def square_colors : List Color := [Color.blue, Color.red, Color.green]

-- Each triangle can be painted in one of the colors: blue, red, or yellow.
def triangle_colors : List Color := [Color.blue, Color.red, Color.yellow]

-- Condition that polygons with a common side cannot share the same color
def different_color (c1 c2 : Color) : Prop := c1 ≠ c2

-- Part (a)
theorem part_a : ∃ n : Nat, n = 7 := sorry

-- Part (b)
theorem part_b : ∃ n : Nat, n = 43 := sorry

-- Part (c)
theorem part_c : ∃ n : Nat, n = 667 := sorry

end part_a_part_b_part_c_l1595_159510


namespace average_speed_l1595_159554

-- Define the average speed v
variable {v : ℝ}

-- Conditions
def day1_distance : ℝ := 160  -- 160 miles on the first day
def day2_distance : ℝ := 280  -- 280 miles on the second day
def time_difference : ℝ := 3  -- 3 hours difference

-- Theorem to prove the average speed
theorem average_speed (h1 : day1_distance / v + time_difference = day2_distance / v) : v = 40 := 
by 
  sorry  -- Proof is omitted

end average_speed_l1595_159554


namespace solution_set_equivalence_l1595_159587

noncomputable def f : ℝ → ℝ := sorry

axiom f_derivative : ∀ x : ℝ, deriv f x > 1 - f x
axiom f_at_0 : f 0 = 3

theorem solution_set_equivalence :
  {x : ℝ | (Real.exp x) * f x > (Real.exp x) + 2} = {x : ℝ | x > 0} :=
by sorry

end solution_set_equivalence_l1595_159587


namespace maximum_value_expression_l1595_159549

theorem maximum_value_expression (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 + 3 * a * b)) + Real.sqrt (Real.sqrt (b^2 + 3 * b * c)) +
   Real.sqrt (Real.sqrt (c^2 + 3 * c * d)) + Real.sqrt (Real.sqrt (d^2 + 3 * d * a))) ≤ 4 * Real.sqrt 2 :=
by 
  sorry

end maximum_value_expression_l1595_159549


namespace necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l1595_159532

-- Problem 1
theorem necessary_condition_for_q_implies_m_bounds (m : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 20 ≤ 0 → 1 - m^2 ≤ x ∧ x ≤ 1 + m^2) → (- Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
sorry

-- Problem 2
theorem necessary_but_not_sufficient_condition_for_not_q (m : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - 8 * x - 20 ≤ 0) → ¬ (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) → (m ≥ 3 ∨ m ≤ -3) :=
sorry

end necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l1595_159532


namespace crab_ratio_l1595_159520

theorem crab_ratio 
  (oysters_day1 : ℕ) 
  (crabs_day1 : ℕ) 
  (total_days : ℕ) 
  (oysters_ratio : ℕ) 
  (oysters_day2 : ℕ) 
  (total_oysters_crabs : ℕ) 
  (crabs_day2 : ℕ) 
  (ratio : ℚ) :
  oysters_day1 = 50 →
  crabs_day1 = 72 →
  oysters_ratio = 2 →
  oysters_day2 = oysters_day1 / oysters_ratio →
  total_oysters_crabs = 195 →
  total_oysters_crabs = oysters_day1 + crabs_day1 + oysters_day2 + crabs_day2 →
  crabs_day2 = total_oysters_crabs - (oysters_day1 + crabs_day1 + oysters_day2) →
  ratio = (crabs_day2 : ℚ) / crabs_day1 →
  ratio = 2 / 3 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end crab_ratio_l1595_159520


namespace amount_A_received_l1595_159577

-- Define the conditions
def total_amount : ℕ := 600
def ratio_a : ℕ := 1
def ratio_b : ℕ := 2

-- Define the total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b

-- Define the value of one part
def value_per_part : ℕ := total_amount / total_parts

-- Define the amount A gets
def amount_A_gets : ℕ := ratio_a * value_per_part

-- Lean statement to prove
theorem amount_A_received : amount_A_gets = 200 := by
  sorry

end amount_A_received_l1595_159577


namespace painting_frame_ratio_proof_l1595_159564

def framed_painting_ratio (x : ℝ) : Prop :=
  let width := 20
  let height := 20
  let side_border := x
  let top_bottom_border := 3 * x
  let framed_width := width + 2 * side_border
  let framed_height := height + 2 * top_bottom_border
  let painting_area := width * height
  let frame_area := painting_area
  let total_area := framed_width * framed_height - painting_area
  total_area = frame_area ∧ (width + 2 * side_border) ≤ (height + 2 * top_bottom_border) → 
  framed_width / framed_height = 4 / 7

theorem painting_frame_ratio_proof (x : ℝ) (h : framed_painting_ratio x) : (20 + 2 * x) / (20 + 6 * x) = 4 / 7 :=
  sorry

end painting_frame_ratio_proof_l1595_159564


namespace gym_class_students_l1595_159553

theorem gym_class_students :
  ∃ n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 6 = 3 ∧ n % 8 = 5 ∧ n % 9 = 2 ∧ (n = 165 ∨ n = 237) :=
by
  sorry

end gym_class_students_l1595_159553


namespace savings_percentage_correct_l1595_159528

def coat_price : ℝ := 120
def hat_price : ℝ := 30
def gloves_price : ℝ := 50

def coat_discount : ℝ := 0.20
def hat_discount : ℝ := 0.40
def gloves_discount : ℝ := 0.30

def original_total : ℝ := coat_price + hat_price + gloves_price
def coat_savings : ℝ := coat_price * coat_discount
def hat_savings : ℝ := hat_price * hat_discount
def gloves_savings : ℝ := gloves_price * gloves_discount
def total_savings : ℝ := coat_savings + hat_savings + gloves_savings

theorem savings_percentage_correct :
  (total_savings / original_total) * 100 = 25.5 := by
  sorry

end savings_percentage_correct_l1595_159528


namespace total_cost_of_shirt_and_sweater_l1595_159519

-- Define the given conditions
def price_of_shirt := 36.46
def diff_price_shirt_sweater := 7.43
def price_of_sweater := price_of_shirt + diff_price_shirt_sweater

-- Statement to prove
theorem total_cost_of_shirt_and_sweater :
  price_of_shirt + price_of_sweater = 80.35 :=
by
  -- Proof goes here
  sorry

end total_cost_of_shirt_and_sweater_l1595_159519


namespace Joan_paid_158_l1595_159589

theorem Joan_paid_158 (J K : ℝ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end Joan_paid_158_l1595_159589


namespace sum_arithmetic_sequence_100_to_110_l1595_159579

theorem sum_arithmetic_sequence_100_to_110 :
  let a := 100
  let l := 110
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 1155 := by
  sorry

end sum_arithmetic_sequence_100_to_110_l1595_159579


namespace proof_min_k_l1595_159583

-- Define the number of teachers
def num_teachers : ℕ := 200

-- Define what it means for a teacher to send a message to another teacher.
-- Represent this as a function where each teacher sends a message to exactly one other teacher.
def sends_message (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∀ i : Fin num_teachers, ∃ j : Fin num_teachers, teachers i = j

-- Define the main proposition: there exists a group of 67 teachers where no one sends a message to anyone else in the group.
def min_k (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∃ (k : ℕ) (reps : Fin k → Fin num_teachers), k ≥ 67 ∧
  ∀ (i j : Fin k), i ≠ j → teachers (reps i) ≠ reps j

theorem proof_min_k : ∀ (teachers : Fin num_teachers → Fin num_teachers),
  sends_message teachers → min_k teachers :=
sorry

end proof_min_k_l1595_159583


namespace ratio_volume_surface_area_l1595_159558

noncomputable def volume : ℕ := 10
noncomputable def surface_area : ℕ := 45

theorem ratio_volume_surface_area : volume / surface_area = 2 / 9 := by
  sorry

end ratio_volume_surface_area_l1595_159558


namespace smallest_angle_l1595_159503

theorem smallest_angle (k : ℝ) (h1 : 4 * k + 5 * k + 7 * k = 180) : 4 * k = 45 :=
by sorry

end smallest_angle_l1595_159503


namespace integer_solutions_to_equation_l1595_159575

theorem integer_solutions_to_equation :
  ∃ (x y : ℤ), 2 * x^2 + 8 * y^2 = 17 * x * y - 423 ∧
               ((x = 11 ∧ y = 19) ∨ (x = -11 ∧ y = -19)) :=
by
  sorry

end integer_solutions_to_equation_l1595_159575


namespace conference_center_people_count_l1595_159527

-- Definition of the conditions
def rooms : ℕ := 6
def capacity_per_room : ℕ := 80
def fraction_full : ℚ := 2/3

-- Total capacity of the conference center
def total_capacity := rooms * capacity_per_room

-- Number of people in the conference center when 2/3 full
def num_people := fraction_full * total_capacity

-- The theorem stating the problem
theorem conference_center_people_count :
  num_people = 320 := 
by
  -- This is a placeholder for the proof
  sorry

end conference_center_people_count_l1595_159527


namespace aarti_three_times_work_l1595_159540

theorem aarti_three_times_work (d : ℕ) (h : d = 5) : 3 * d = 15 :=
by
  sorry

end aarti_three_times_work_l1595_159540


namespace probability_number_greater_than_3_from_0_5_l1595_159518

noncomputable def probability_number_greater_than_3_in_0_5 : ℝ :=
  let total_interval_length := 5 - 0
  let event_interval_length := 5 - 3
  event_interval_length / total_interval_length

theorem probability_number_greater_than_3_from_0_5 :
  probability_number_greater_than_3_in_0_5 = 2 / 5 :=
by
  sorry

end probability_number_greater_than_3_from_0_5_l1595_159518


namespace sound_heard_in_4_seconds_l1595_159598

/-- Given the distance between a boy and his friend is 1200 meters,
    the speed of the car is 108 km/hr, and the speed of sound is 330 m/s,
    the duration after which the friend hears the whistle is 4 seconds. -/
theorem sound_heard_in_4_seconds :
  let distance := 1200  -- distance in meters
  let speed_of_car_kmh := 108  -- speed of car in km/hr
  let speed_of_sound := 330  -- speed of sound in m/s
  let speed_of_car := speed_of_car_kmh * 1000 / 3600  -- convert km/hr to m/s
  let effective_speed_of_sound := speed_of_sound - speed_of_car
  let time := distance / effective_speed_of_sound
  time = 4 := 
by
  sorry

end sound_heard_in_4_seconds_l1595_159598


namespace kitchen_length_l1595_159599

-- Define the conditions
def tile_area : ℕ := 6
def kitchen_width : ℕ := 48
def number_of_tiles : ℕ := 96

-- The total area is the number of tiles times the area of each tile
def total_area : ℕ := number_of_tiles * tile_area

-- Statement to prove the length of the kitchen
theorem kitchen_length : (total_area / kitchen_width) = 12 :=
by
  sorry

end kitchen_length_l1595_159599


namespace composite_shape_sum_l1595_159560

def triangular_prism_faces := 5
def triangular_prism_edges := 9
def triangular_prism_vertices := 6

def pentagonal_prism_additional_faces := 7
def pentagonal_prism_additional_edges := 10
def pentagonal_prism_additional_vertices := 5

def pyramid_additional_faces := 5
def pyramid_additional_edges := 5
def pyramid_additional_vertices := 1

def resulting_shape_faces := triangular_prism_faces - 1 + pentagonal_prism_additional_faces + pyramid_additional_faces
def resulting_shape_edges := triangular_prism_edges + pentagonal_prism_additional_edges + pyramid_additional_edges
def resulting_shape_vertices := triangular_prism_vertices + pentagonal_prism_additional_vertices + pyramid_additional_vertices

def sum_faces_edges_vertices := resulting_shape_faces + resulting_shape_edges + resulting_shape_vertices

theorem composite_shape_sum : sum_faces_edges_vertices = 51 :=
by
  unfold sum_faces_edges_vertices resulting_shape_faces resulting_shape_edges resulting_shape_vertices
  unfold triangular_prism_faces triangular_prism_edges triangular_prism_vertices
  unfold pentagonal_prism_additional_faces pentagonal_prism_additional_edges pentagonal_prism_additional_vertices
  unfold pyramid_additional_faces pyramid_additional_edges pyramid_additional_vertices
  simp
  sorry

end composite_shape_sum_l1595_159560


namespace inverse_linear_intersection_l1595_159552

theorem inverse_linear_intersection (m n : ℝ) 
  (h1 : n = 2 / m) 
  (h2 : n = m + 3) 
  : (1 / m) - (1 / n) = 3 / 2 := 
by sorry

end inverse_linear_intersection_l1595_159552


namespace chocolate_ice_cream_ordered_l1595_159514

theorem chocolate_ice_cream_ordered (V C : ℕ) (total_ice_cream : ℕ) (percentage_vanilla : ℚ) 
  (h_total : total_ice_cream = 220) 
  (h_percentage : percentage_vanilla = 0.20) 
  (h_vanilla_total : V = percentage_vanilla * total_ice_cream) 
  (h_vanilla_chocolate : V = 2 * C) 
  : C = 22 := 
by 
  sorry

end chocolate_ice_cream_ordered_l1595_159514


namespace complement_A_in_U_l1595_159531

noncomputable def U : Set ℝ := {x | x > -Real.sqrt 3}
noncomputable def A : Set ℝ := {x | 1 < 4 - x^2 ∧ 4 - x^2 ≤ 2}

theorem complement_A_in_U :
  (U \ A) = {x | -Real.sqrt 3 < x ∧ x ≤ -Real.sqrt 2} ∪ {x | Real.sqrt 2 ≤ x ∧ x < (Real.sqrt 3) ∨ Real.sqrt 3 ≤ x} :=
by
  sorry

end complement_A_in_U_l1595_159531


namespace cylinder_ellipse_major_axis_l1595_159536

theorem cylinder_ellipse_major_axis :
  ∀ (r : ℝ), r = 2 →
  ∀ (minor_axis : ℝ), minor_axis = 2 * r →
  ∀ (major_axis : ℝ), major_axis = 1.4 * minor_axis →
  major_axis = 5.6 :=
by
  intros r hr minor_axis hminor major_axis hmajor
  sorry

end cylinder_ellipse_major_axis_l1595_159536


namespace students_not_enrolled_in_either_l1595_159544

variable (total_students french_students german_students both_students : ℕ)

theorem students_not_enrolled_in_either (h1 : total_students = 60)
                                        (h2 : french_students = 41)
                                        (h3 : german_students = 22)
                                        (h4 : both_students = 9) :
    total_students - (french_students + german_students - both_students) = 6 := by
  sorry

end students_not_enrolled_in_either_l1595_159544


namespace jack_total_dollars_l1595_159582

-- Constants
def initial_dollars : ℝ := 45
def euro_amount : ℝ := 36
def yen_amount : ℝ := 1350
def ruble_amount : ℝ := 1500
def euro_to_dollar : ℝ := 2
def yen_to_dollar : ℝ := 0.009
def ruble_to_dollar : ℝ := 0.013
def transaction_fee_rate : ℝ := 0.01
def spending_rate : ℝ := 0.1

-- Convert each foreign currency to dollars
def euros_to_dollars : ℝ := euro_amount * euro_to_dollar
def yen_to_dollars : ℝ := yen_amount * yen_to_dollar
def rubles_to_dollars : ℝ := ruble_amount * ruble_to_dollar

-- Calculate transaction fees for each currency conversion
def euros_fee : ℝ := euros_to_dollars * transaction_fee_rate
def yen_fee : ℝ := yen_to_dollars * transaction_fee_rate
def rubles_fee : ℝ := rubles_to_dollars * transaction_fee_rate

-- Subtract transaction fees from the converted amounts
def euros_after_fee : ℝ := euros_to_dollars - euros_fee
def yen_after_fee : ℝ := yen_to_dollars - yen_fee
def rubles_after_fee : ℝ := rubles_to_dollars - rubles_fee

-- Calculate total dollars after conversion and fees
def total_dollars_before_spending : ℝ := initial_dollars + euros_after_fee + yen_after_fee + rubles_after_fee

-- Calculate 10% expenditure
def spending_amount : ℝ := total_dollars_before_spending * spending_rate

-- Calculate final amount after spending
def final_amount : ℝ := total_dollars_before_spending - spending_amount

theorem jack_total_dollars : final_amount = 132.85 := by
  sorry

end jack_total_dollars_l1595_159582


namespace total_packs_l1595_159529

theorem total_packs (cards_per_person cards_per_pack : ℕ) (num_people : ℕ) 
  (h1 : cards_per_person = 540) 
  (h2 : cards_per_pack = 20) 
  (h3 : num_people = 4) : 
  (cards_per_person / cards_per_pack) * num_people = 108 := 
by
  sorry

end total_packs_l1595_159529


namespace time_for_runnerA_to_complete_race_l1595_159593

variable (speedA : ℝ) -- speed of runner A in meters per second
variable (t : ℝ) -- time taken by runner A to complete the race in seconds
variable (tB : ℝ) -- time taken by runner B to complete the race in seconds

noncomputable def distanceA : ℝ := 1000 -- distance covered by runner A in meters
noncomputable def distanceB : ℝ := 950 -- distance covered by runner B in meters when A finishes
noncomputable def speedB : ℝ := distanceB / tB -- speed of runner B in meters per second

theorem time_for_runnerA_to_complete_race
    (h1 : distanceA = speedA * t)
    (h2 : distanceB = speedA * (t + 20)) :
    t = 400 :=
by
  sorry

end time_for_runnerA_to_complete_race_l1595_159593


namespace initial_marbles_l1595_159504

variable (C_initial : ℕ)
variable (marbles_given : ℕ := 42)
variable (marbles_left : ℕ := 5)

theorem initial_marbles :
  C_initial = marbles_given + marbles_left :=
sorry

end initial_marbles_l1595_159504


namespace total_percentage_increase_l1595_159551

def initial_time : ℝ := 45
def additive_A_increase : ℝ := 0.35
def additive_B_increase : ℝ := 0.20

theorem total_percentage_increase :
  let time_after_A := initial_time * (1 + additive_A_increase)
  let time_after_B := time_after_A * (1 + additive_B_increase)
  (time_after_B - initial_time) / initial_time * 100 = 62 :=
  sorry

end total_percentage_increase_l1595_159551


namespace exam_total_questions_l1595_159541

/-- 
In an examination, a student scores 4 marks for every correct answer 
and loses 1 mark for every wrong answer. The student secures 140 marks 
in total. Given that the student got 40 questions correct, 
prove that the student attempted a total of 60 questions. 
-/
theorem exam_total_questions (C W T : ℕ) 
  (score_correct : C = 40)
  (total_score : 4 * C - W = 140)
  (total_questions : T = C + W) : 
  T = 60 := 
by 
  -- Proof omitted
  sorry

end exam_total_questions_l1595_159541


namespace max_area_rectangle_l1595_159539

theorem max_area_rectangle (perimeter : ℕ) (a b : ℕ) (h1 : perimeter = 30) 
  (h2 : b = a + 3) : a * b = 54 :=
by
  sorry

end max_area_rectangle_l1595_159539


namespace determine_g_l1595_159566

def real_function (g : ℝ → ℝ) :=
  ∀ c d : ℝ, g (c + d) + g (c - d) = g (c) * g (d) + g (d)

def non_zero_function (g : ℝ → ℝ) :=
  ∃ x : ℝ, g x ≠ 0

theorem determine_g (g : ℝ → ℝ) (h1 : real_function g) (h2 : non_zero_function g) : g 0 = 1 ∧ ∀ x : ℝ, g (-x) = g x := 
sorry

end determine_g_l1595_159566


namespace division_proof_l1595_159537

-- Define the given condition
def given_condition : Prop :=
  2084.576 / 135.248 = 15.41

-- Define the problem statement we want to prove
def problem_statement : Prop :=
  23.8472 / 13.5786 = 1.756

-- Main theorem stating that under the given condition, the problem statement holds
theorem division_proof (h : given_condition) : problem_statement :=
by sorry

end division_proof_l1595_159537


namespace fraction_exp_3_4_cubed_l1595_159569

def fraction_exp (a b n : ℕ) : ℚ := (a : ℚ) ^ n / (b : ℚ) ^ n

theorem fraction_exp_3_4_cubed : fraction_exp 3 4 3 = 27 / 64 :=
by
  sorry

end fraction_exp_3_4_cubed_l1595_159569


namespace compute_R_at_3_l1595_159568

def R (x : ℝ) := 3 * x ^ 4 + x ^ 3 + x ^ 2 + x + 1

theorem compute_R_at_3 : R 3 = 283 := by
  sorry

end compute_R_at_3_l1595_159568


namespace simplify_fraction_l1595_159526

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end simplify_fraction_l1595_159526


namespace problem_solution_l1595_159597

def f (x : ℤ) : ℤ := 3 * x + 1
def g (x : ℤ) : ℤ := 4 * x - 3

theorem problem_solution :
  (f (g (f 3))) / (g (f (g 3))) = 112 / 109 := by
sorry

end problem_solution_l1595_159597


namespace cross_country_meet_winning_scores_l1595_159522

theorem cross_country_meet_winning_scores :
  ∃ (scores : Finset ℕ), scores.card = 13 ∧
    ∀ s ∈ scores, s ≥ 15 ∧ s ≤ 27 :=
by
  sorry

end cross_country_meet_winning_scores_l1595_159522
