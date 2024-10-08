import Mathlib

namespace ways_to_go_home_via_library_l69_69123

def ways_from_school_to_library := 2
def ways_from_library_to_home := 3

theorem ways_to_go_home_via_library : 
  ways_from_school_to_library * ways_from_library_to_home = 6 :=
by 
  sorry

end ways_to_go_home_via_library_l69_69123


namespace a_completes_in_12_days_l69_69146

def work_rate_a_b (r_A r_B : ℝ) := r_A + r_B = 1 / 3
def work_rate_b_c (r_B r_C : ℝ) := r_B + r_C = 1 / 2
def work_rate_a_c (r_A r_C : ℝ) := r_A + r_C = 1 / 3

theorem a_completes_in_12_days (r_A r_B r_C : ℝ) 
  (h1 : work_rate_a_b r_A r_B)
  (h2 : work_rate_b_c r_B r_C)
  (h3 : work_rate_a_c r_A r_C) : 
  1 / r_A = 12 :=
by
  sorry

end a_completes_in_12_days_l69_69146


namespace reasoning_classification_correct_l69_69050

def analogical_reasoning := "reasoning from specific to specific"
def inductive_reasoning := "reasoning from part to whole and from individual to general"
def deductive_reasoning := "reasoning from general to specific"

theorem reasoning_classification_correct : 
  (analogical_reasoning, inductive_reasoning, deductive_reasoning) =
  ("reasoning from specific to specific", "reasoning from part to whole and from individual to general", "reasoning from general to specific") := 
by 
  sorry

end reasoning_classification_correct_l69_69050


namespace sector_max_area_l69_69999

noncomputable def max_sector_area (R c : ℝ) : ℝ := 
  if h : R = c / 4 then c^2 / 16 else 0 -- This is just a skeleton, actual proof requires conditions
-- State the theorem that relates conditions to the maximum area.
theorem sector_max_area (R c α : ℝ) 
  (hc : c = 2 * R + R * α) : 
  (∃ R, R = c / 4) → max_sector_area R c = c^2 / 16 :=
by 
  sorry

end sector_max_area_l69_69999


namespace lex_apples_l69_69227

theorem lex_apples (A : ℕ) (h1 : A / 5 < 100) (h2 : A = (A / 5) + ((A / 5) + 9) + 42) : A = 85 :=
by
  sorry

end lex_apples_l69_69227


namespace evaluate_expression_l69_69681

theorem evaluate_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 8^(3/2) + 5) = 25 - 16 * Real.sqrt 2 := 
by
  sorry

end evaluate_expression_l69_69681


namespace zigzag_lines_divide_regions_l69_69551

-- Define the number of regions created by n zigzag lines
def regions (n : ℕ) : ℕ := (2 * n * (2 * n + 1)) / 2 + 1 - 2 * n

-- Main theorem
theorem zigzag_lines_divide_regions (n : ℕ) : ∃ k : ℕ, k = regions n := by
  sorry

end zigzag_lines_divide_regions_l69_69551


namespace math_problem_l69_69611

theorem math_problem (a b c d x : ℝ)
  (h1 : a = -(-b))
  (h2 : c = -1 / d)
  (h3 : |x| = 3) :
  x^3 + c * d * x^2 - (a - b) / 2 = 18 ∨ x^3 + c * d * x^2 - (a - b) / 2 = -36 :=
by sorry

end math_problem_l69_69611


namespace domain_log_sin_sqrt_l69_69848

theorem domain_log_sin_sqrt (x : ℝ) : 
  (2 < x ∧ x < (5 * Real.pi) / 3) ↔ 
  (∃ k : ℤ, (Real.pi / 3) + (4 * k * Real.pi) < x ∧ x < (5 * Real.pi / 3) + (4 * k * Real.pi) ∧ 2 < x) :=
by
  sorry

end domain_log_sin_sqrt_l69_69848


namespace money_out_of_pocket_l69_69906

theorem money_out_of_pocket
  (old_system_cost : ℝ)
  (trade_in_percent : ℝ)
  (new_system_cost : ℝ)
  (discount_percent : ℝ)
  (trade_in_value : ℝ)
  (discount_value : ℝ)
  (discounted_price : ℝ)
  (money_out_of_pocket : ℝ) :
  old_system_cost = 250 →
  trade_in_percent = 80 / 100 →
  new_system_cost = 600 →
  discount_percent = 25 / 100 →
  trade_in_value = old_system_cost * trade_in_percent →
  discount_value = new_system_cost * discount_percent →
  discounted_price = new_system_cost - discount_value →
  money_out_of_pocket = discounted_price - trade_in_value →
  money_out_of_pocket = 250 := by
  intros
  sorry

end money_out_of_pocket_l69_69906


namespace quadratic_function_through_point_l69_69575

theorem quadratic_function_through_point : 
  (∃ (a : ℝ), ∀ (x y : ℝ), y = a * x ^ 2 ∧ ((x, y) = (-1, 4)) → y = 4 * x ^ 2) :=
sorry

end quadratic_function_through_point_l69_69575


namespace extinction_prob_one_l69_69342

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end extinction_prob_one_l69_69342


namespace kids_played_on_Wednesday_l69_69160

def played_on_Monday : ℕ := 17
def played_on_Tuesday : ℕ := 15
def total_kids : ℕ := 34

theorem kids_played_on_Wednesday :
  total_kids - (played_on_Monday + played_on_Tuesday) = 2 :=
by sorry

end kids_played_on_Wednesday_l69_69160


namespace find_m_value_l69_69247

theorem find_m_value : 
  ∀ (u v : ℝ), 
    (3 * u^2 + 4 * u + 5 = 0) ∧ 
    (3 * v^2 + 4 * v + 5 = 0) ∧ 
    (u + v = -4/3) ∧ 
    (u * v = 5/3) → 
    ∃ m n : ℝ, 
      (x^2 + m * x + n = 0) ∧ 
      ((u^2 + 1) + (v^2 + 1) = -m) ∧ 
      (m = -4/9) :=
by {
  -- Insert proof here
  sorry
}

end find_m_value_l69_69247


namespace wilma_garden_rows_l69_69436

theorem wilma_garden_rows :
  ∃ (rows : ℕ),
    (∃ (yellow green red total : ℕ),
      yellow = 12 ∧
      green = 2 * yellow ∧
      red = 42 ∧
      total = yellow + green + red ∧
      total / 13 = rows ∧
      rows = 6) :=
sorry

end wilma_garden_rows_l69_69436


namespace union_of_sets_l69_69653

def M : Set Int := { -1, 0, 1 }
def N : Set Int := { 0, 1, 2 }

theorem union_of_sets : M ∪ N = { -1, 0, 1, 2 } := by
  sorry

end union_of_sets_l69_69653


namespace min_nSn_l69_69872

theorem min_nSn 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (m : ℕ)
  (h1 : m ≥ 2)
  (h2 : S (m-1) = -2) 
  (h3 : S m = 0) 
  (h4 : S (m+1) = 3) : 
  ∃ n : ℕ, n * S n = -9 :=
by {
  sorry
}

end min_nSn_l69_69872


namespace triangle_side_relationship_l69_69980

theorem triangle_side_relationship
  (a b c : ℝ)
  (habc : a < b + c)
  (ha_pos : a > 0) :
  a^2 < a * b + a * c :=
by
  sorry

end triangle_side_relationship_l69_69980


namespace breadth_of_rectangular_plot_is_18_l69_69360

/-- Problem statement:
The length of a rectangular plot is thrice its breadth. 
If the area of the rectangular plot is 972 sq m, 
this theorem proves that the breadth of the rectangular plot is 18 meters.
-/
theorem breadth_of_rectangular_plot_is_18 (b l : ℝ) (h_length : l = 3 * b) (h_area : l * b = 972) : b = 18 :=
by
  sorry

end breadth_of_rectangular_plot_is_18_l69_69360


namespace lena_calculation_l69_69955

def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + (10 - n % 10)

theorem lena_calculation :
  round_to_nearest_ten (63 + 2 * 29) = 120 :=
by
  sorry

end lena_calculation_l69_69955


namespace arithmetic_sequence_sum_l69_69369

variable (a_n : ℕ → ℕ)

theorem arithmetic_sequence_sum (h1: a_n 1 + a_n 2 = 5) (h2 : a_n 3 + a_n 4 = 7) (arith : ∀ n, a_n (n + 1) - a_n n = a_n 2 - a_n 1) :
  a_n 5 + a_n 6 = 9 := 
sorry

end arithmetic_sequence_sum_l69_69369


namespace speed_of_stream_l69_69064

-- Define the speed of the boat in still water
def speed_of_boat_in_still_water : ℝ := 39

-- Define the effective speed upstream and downstream
def effective_speed_upstream (v : ℝ) : ℝ := speed_of_boat_in_still_water - v
def effective_speed_downstream (v : ℝ) : ℝ := speed_of_boat_in_still_water + v

-- Define the condition that time upstream is twice the time downstream
def time_condition (D v : ℝ) : Prop := 
  (D / effective_speed_upstream v = 2 * (D / effective_speed_downstream v))

-- The main theorem stating the speed of the stream
theorem speed_of_stream (D : ℝ) (h : D > 0) : (v : ℝ) → time_condition D v → v = 13 :=
by
  sorry

end speed_of_stream_l69_69064


namespace algebra_expression_value_l69_69645

theorem algebra_expression_value (a : ℝ) (h : 3 * a ^ 2 + 2 * a - 1 = 0) : 3 * a ^ 2 + 2 * a - 2019 = -2018 := 
by 
  -- Proof goes here
  sorry

end algebra_expression_value_l69_69645


namespace islander_distances_l69_69742

theorem islander_distances (A B C D : ℕ) (k1 : A = 1 ∨ A = 2)
  (k2 : B = 2)
  (C_liar : C = 1) (is_knight : C ≠ 1) :
  C = 1 ∨ C = 3 ∨ C = 4 ∧ D = 2 :=
by {
  sorry
}

end islander_distances_l69_69742


namespace animal_shelter_cats_l69_69574

theorem animal_shelter_cats (D C x : ℕ) (h1 : 15 * C = 7 * D) (h2 : 15 * (C + x) = 11 * D) (h3 : D = 60) : x = 16 :=
by
  sorry

end animal_shelter_cats_l69_69574


namespace find_a_tangent_line_eq_l69_69433

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x - 1) * Real.exp x

theorem find_a (a : ℝ) : f 1 (-3) = 0 → a = 1 := by
  sorry

theorem tangent_line_eq (x : ℝ) (e : ℝ) : x = 1 ∧ f 1 x = Real.exp 1 → 
    (4 * Real.exp 1 * x - y - 3 * Real.exp 1 = 0) := by
  sorry

end find_a_tangent_line_eq_l69_69433


namespace linear_function_decreasing_iff_l69_69176

-- Define the conditions
def linear_function (m b x : ℝ) : ℝ := m * x + b

-- Define the condition for decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2

-- The theorem to prove
theorem linear_function_decreasing_iff (m b : ℝ) :
  (is_decreasing (linear_function m b)) ↔ (m < 0) :=
by
  sorry

end linear_function_decreasing_iff_l69_69176


namespace intercept_form_impossible_values_l69_69237

-- Define the problem statement
theorem intercept_form_impossible_values (m : ℝ) :
  (¬ (∃ a b c : ℝ, m ≠ 0 ∧ a * m = 0 ∧ b * m = 0 ∧ c * m = 1) ↔ (m = 4 ∨ m = -3 ∨ m = 5)) :=
sorry

end intercept_form_impossible_values_l69_69237


namespace trigonometric_identity_l69_69890

open Real

theorem trigonometric_identity :
  sin (72 * pi / 180) * cos (12 * pi / 180) - cos (72 * pi / 180) * sin (12 * pi / 180) = sqrt 3 / 2 :=
by
  sorry

end trigonometric_identity_l69_69890


namespace batsman_average_after_17_l69_69406

variable (x : ℝ)
variable (total_runs_16 : ℝ := 16 * x)
variable (runs_17 : ℝ := 90)
variable (new_total_runs : ℝ := total_runs_16 + runs_17)
variable (new_average : ℝ := new_total_runs / 17)

theorem batsman_average_after_17 :
  (total_runs_16 + runs_17 = 17 * (x + 3)) → new_average = x + 3 → new_average = 42 :=
by
  intros h1 h2
  sorry

end batsman_average_after_17_l69_69406


namespace tan_seven_pi_over_four_l69_69922

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by sorry

end tan_seven_pi_over_four_l69_69922


namespace project_completion_time_saving_l69_69944

/-- A theorem stating that if a project with initial and additional workforce configuration,
the project will be completed 10 days ahead of schedule. -/
theorem project_completion_time_saving
  (total_days : ℕ := 100)
  (initial_people : ℕ := 10)
  (initial_days : ℕ := 30)
  (initial_fraction : ℚ := 1 / 5)
  (additional_people : ℕ := 10)
  : (total_days - ((initial_days + (1 / (initial_people + additional_people * initial_fraction)) * (total_days * initial_fraction) / initial_fraction)) = 10) :=
sorry

end project_completion_time_saving_l69_69944


namespace stuart_initial_marbles_l69_69008

theorem stuart_initial_marbles
    (betty_marbles : ℕ)
    (stuart_marbles_after_given : ℕ)
    (percentage_given : ℚ)
    (betty_gave : ℕ):
    betty_marbles = 60 →
    stuart_marbles_after_given = 80 →
    percentage_given = 0.40 →
    betty_gave = percentage_given * betty_marbles →
    stuart_marbles_after_given = stuart_initial + betty_gave →
    stuart_initial = 56 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end stuart_initial_marbles_l69_69008


namespace find_x_plus_y_l69_69846

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := 
by
  sorry

end find_x_plus_y_l69_69846


namespace find_expression_l69_69328

theorem find_expression (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end find_expression_l69_69328


namespace problem1_solution_problem2_solution_l69_69903

-- Problem 1: f(x-2) = 3x - 5 implies f(x) = 3x + 1
def problem1 (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f (x - 2) = 3 * x - 5 → f x = 3 * x + 1

-- Problem 2: Quadratic function satisfying specific conditions
def is_quadratic (f : ℝ → ℝ) : Prop := 
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a*x^2 + b*x + c

def problem2 (f : ℝ → ℝ) : Prop :=
  is_quadratic f ∧
  (f 0 = 4) ∧
  (∀ x : ℝ, f (3 - x) = f x) ∧
  (∀ x : ℝ, f x ≥ 7/4) →
  (∀ x : ℝ, f x = x^2 - 3*x + 4)

-- Statements to be proved
theorem problem1_solution : ∀ f : ℝ → ℝ, problem1 x f := sorry
theorem problem2_solution : ∀ f : ℝ → ℝ, problem2 f := sorry

end problem1_solution_problem2_solution_l69_69903


namespace composite_quadratic_l69_69491

theorem composite_quadratic (m n : ℤ) (x1 x2 : ℤ)
  (h1 : 2 * x1^2 + m * x1 + 2 - n = 0)
  (h2 : 2 * x2^2 + m * x2 + 2 - n = 0)
  (h3 : x1 ≠ 0) 
  (h4 : x2 ≠ 0) :
  ∃ (k : ℕ), ∃ (l : ℕ), 
    (k > 1) ∧ (l > 1) ∧ (k * l = (m^2 + n^2) / 4) := sorry

end composite_quadratic_l69_69491


namespace Tim_weekly_water_intake_l69_69759

variable (daily_bottle_intake : ℚ)
variable (additional_intake : ℚ)
variable (quart_to_ounces : ℚ)
variable (days_in_week : ℕ := 7)

theorem Tim_weekly_water_intake (H1 : daily_bottle_intake = 2 * 1.5)
                              (H2 : additional_intake = 20)
                              (H3 : quart_to_ounces = 32) :
  (daily_bottle_intake * quart_to_ounces + additional_intake) * days_in_week = 812 := by
  sorry

end Tim_weekly_water_intake_l69_69759


namespace find_divisor_of_x_l69_69300

theorem find_divisor_of_x (x : ℕ) (q p : ℕ) (h1 : x % n = 5) (h2 : 4 * x % n = 2) : n = 9 :=
by
  sorry

end find_divisor_of_x_l69_69300


namespace find_fraction_l69_69293

variable {N : ℕ}
variable {f : ℚ}

theorem find_fraction (h1 : N = 150) (h2 : N - f * N = 60) : f = 3/5 := by
  sorry

end find_fraction_l69_69293


namespace order_exponents_l69_69221

theorem order_exponents :
  (2:ℝ) ^ 300 < (3:ℝ) ^ 200 ∧ (3:ℝ) ^ 200 < (10:ℝ) ^ 100 :=
by
  sorry

end order_exponents_l69_69221


namespace value_of_m_l69_69365

theorem value_of_m (m : ℚ) : 
  (m = - -(-(1/3) : ℚ) → m = -1/3) :=
by
  sorry

end value_of_m_l69_69365


namespace add_n_to_constant_l69_69508

theorem add_n_to_constant (y n : ℝ) (h_eq : y^4 - 20 * y + 1 = 22) (h_n : n = 3) : y^4 - 20 * y + 4 = 25 :=
by
  sorry

end add_n_to_constant_l69_69508


namespace find_f_neg_3_l69_69299

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def functional_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 - x) = f (1 + x)

def function_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x^2

theorem find_f_neg_3 
  (hf_even : even_function f) 
  (hf_condition : functional_condition f)
  (hf_interval : function_on_interval f) : 
  f (-3) = 1 := 
by
  sorry

end find_f_neg_3_l69_69299


namespace envelope_width_l69_69362

theorem envelope_width (L W A : ℝ) (hL : L = 4) (hA : A = 16) (hArea : A = L * W) : W = 4 := 
by
  -- We state the problem
  sorry

end envelope_width_l69_69362


namespace simplify_expression_l69_69343

theorem simplify_expression (y : ℝ) : (y - 2)^2 + 2 * (y - 2) * (5 + y) + (5 + y)^2 = (2*y + 3)^2 := 
by sorry

end simplify_expression_l69_69343


namespace average_bowling_score_l69_69930

-- Definitions of the scores
def g : ℕ := 120
def m : ℕ := 113
def b : ℕ := 85

-- Theorem statement: The average score is 106
theorem average_bowling_score : (g + m + b) / 3 = 106 := by
  sorry

end average_bowling_score_l69_69930


namespace find_k_l69_69306

noncomputable def sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  3 * 2^n + k

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a n * a (n + 2) = (a (n + 1))^2

theorem find_k
  (a : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a n = sequence_sum (n + 1) k - sequence_sum n k)
  (h2 : geometric_sequence a) :
  k = -3 :=
  by sorry

end find_k_l69_69306


namespace what_percent_of_y_l69_69464

-- Given condition
axiom y_pos : ℝ → Prop

noncomputable def math_problem (y : ℝ) (h : y_pos y) : Prop :=
  (8 * y / 20 + 3 * y / 10 = 0.7 * y)

-- The theorem to be proved
theorem what_percent_of_y (y : ℝ) (h : y > 0) : 8 * y / 20 + 3 * y / 10 = 0.7 * y :=
by
  sorry

end what_percent_of_y_l69_69464


namespace common_difference_of_AP_l69_69685

theorem common_difference_of_AP (a T_12 : ℝ) (d : ℝ) (n : ℕ) (h1 : a = 2) (h2 : T_12 = 90) (h3 : n = 12) 
(h4 : T_12 = a + (n - 1) * d) : d = 8 := 
by sorry

end common_difference_of_AP_l69_69685


namespace find_y_when_x_is_minus_2_l69_69364

theorem find_y_when_x_is_minus_2 :
  ∀ (x y t : ℝ), (x = 3 - 2 * t) → (y = 5 * t + 6) → (x = -2) → y = 37 / 2 :=
by
  intros x y t h1 h2 h3
  sorry

end find_y_when_x_is_minus_2_l69_69364


namespace time_gaps_l69_69674

theorem time_gaps (dist_a dist_b dist_c : ℕ) (time_a time_b time_c : ℕ) :
  dist_a = 130 →
  dist_b = 130 →
  dist_c = 130 →
  time_a = 36 →
  time_b = 45 →
  time_c = 42 →
  (time_b - time_a = 9) ∧ (time_c - time_a = 6) ∧ (time_b - time_c = 3) := by
  intros hdist_a hdist_b hdist_c htime_a htime_b htime_c
  sorry

end time_gaps_l69_69674


namespace extreme_values_range_of_a_inequality_of_zeros_l69_69741

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -2 * (Real.log x) - a / (x ^ 2) + 1

theorem extreme_values (a : ℝ) (h : a = 1) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≤ 0) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = 0) ∧
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≥ -3 + 2 * (Real.log 2)) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = -3 + 2 * (Real.log 2)) :=
sorry

theorem range_of_a :
  (∀ a : ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 0 < a ∧ a < 1) :=
sorry

theorem inequality_of_zeros (a : ℝ) (h : 0 < a) (h1 : a < 1) (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) (hx1x2 : x1 ≠ x2) :
  1 / (x1 ^ 2) + 1 / (x2 ^ 2) > 2 / a :=
sorry

end extreme_values_range_of_a_inequality_of_zeros_l69_69741


namespace fewest_students_possible_l69_69726

theorem fewest_students_possible :
  ∃ n : ℕ, n ≡ 2 [MOD 5] ∧ n ≡ 4 [MOD 6] ∧ n ≡ 6 [MOD 8] ∧ n = 22 :=
sorry

end fewest_students_possible_l69_69726


namespace arithmetic_sequence_ratios_l69_69341

theorem arithmetic_sequence_ratios
  (a : ℕ → ℝ) (b : ℕ → ℝ) (A : ℕ → ℝ) (B : ℕ → ℝ)
  (d1 d2 a1 b1 : ℝ)
  (hA_sum : ∀ n : ℕ, A n = n * a1 + (n * (n - 1)) * d1 / 2)
  (hB_sum : ∀ n : ℕ, B n = n * b1 + (n * (n - 1)) * d2 / 2)
  (h_ratio : ∀ n : ℕ, B n ≠ 0 → A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, b n ≠ 0 → a n / b n = (4 * n - 3) / (6 * n - 2) := sorry

end arithmetic_sequence_ratios_l69_69341


namespace gcd_60_90_150_l69_69649

theorem gcd_60_90_150 : Nat.gcd (Nat.gcd 60 90) 150 = 30 := 
by
  sorry

end gcd_60_90_150_l69_69649


namespace find_k_l69_69024

theorem find_k (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 12 = 0 → ∃ y : ℝ, y = x + 3 ∧ y^2 - k * y + 12 = 0) →
  k = 3 :=
sorry

end find_k_l69_69024


namespace percentage_of_volume_is_P_l69_69096

noncomputable def volumeOfSolutionP {P Q : ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : ℝ := 
(P / (P + Q)) * 100

theorem percentage_of_volume_is_P {P Q: ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : 
  volumeOfSolutionP h = 50 :=
sorry

end percentage_of_volume_is_P_l69_69096


namespace problem_l69_69119

open Real

theorem problem (x y : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_cond : x + y^(2016) ≥ 1) : 
  x^(2016) + y > 1 - 1/100 :=
by sorry

end problem_l69_69119


namespace quadratic_roots_expression_l69_69702

theorem quadratic_roots_expression {m n : ℝ}
  (h₁ : m^2 + m - 12 = 0)
  (h₂ : n^2 + n - 12 = 0)
  (h₃ : m + n = -1) :
  m^2 + 2 * m + n = 11 :=
by {
  sorry
}

end quadratic_roots_expression_l69_69702


namespace find_m_for_asymptotes_l69_69654

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 9 = 1

-- Definition of the asymptotes form
def asymptote_form (m : ℝ) (x y : ℝ) : Prop :=
  y - 1 = m * x + 2 * m ∨ y - 1 = -m * x - 2 * m

-- The main theorem to prove
theorem find_m_for_asymptotes :
  (∀ x y : ℝ, hyperbola x y → asymptote_form (4 / 3) x y) :=
sorry

end find_m_for_asymptotes_l69_69654


namespace parallel_line_through_point_l69_69786

theorem parallel_line_through_point :
  ∀ {x y : ℝ}, (3 * x + 4 * y + 1 = 0) ∧ (∃ (a b : ℝ), a = 1 ∧ b = 2 ∧ (3 * a + 4 * b + x0 = 0) → (x = -11)) :=
sorry

end parallel_line_through_point_l69_69786


namespace exists_perfect_square_of_the_form_l69_69596

theorem exists_perfect_square_of_the_form (k : ℕ) (h : k > 0) : ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * m = n * 2^k - 7 :=
by sorry

end exists_perfect_square_of_the_form_l69_69596


namespace stationery_cost_l69_69335

theorem stationery_cost (cost_per_pencil cost_per_pen : ℕ)
    (boxes : ℕ)
    (pencils_per_box pens_offset : ℕ)
    (total_cost : ℕ) :
    cost_per_pencil = 4 →
    boxes = 15 →
    pencils_per_box = 80 →
    pens_offset = 300 →
    cost_per_pen = 5 →
    total_cost = (boxes * pencils_per_box * cost_per_pencil) +
                 ((2 * (boxes * pencils_per_box + pens_offset)) * cost_per_pen) →
    total_cost = 18300 :=
by
  intros
  sorry

end stationery_cost_l69_69335


namespace local_minimum_interval_l69_69775

-- Definitions of the function and its derivative
def y (x a : ℝ) : ℝ := x^3 - 2 * a * x + a
def y_prime (x a : ℝ) : ℝ := 3 * x^2 - 2 * a

-- The proof problem statement
theorem local_minimum_interval (a : ℝ) : 
  (0 < a ∧ a < 3 / 2) ↔ ∃ (x : ℝ), (0 < x ∧ x < 1) ∧ y_prime x a = 0 :=
sorry

end local_minimum_interval_l69_69775


namespace find_a_plus_d_l69_69521

variables (a b c d e : ℝ)

theorem find_a_plus_d :
  a + b = 12 ∧ b + c = 9 ∧ c + d = 3 ∧ d + e = 7 ∧ e + a = 10 → a + d = 6 :=
by
  intros h
  have h1 : a + b = 12 := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : d + e = 7 := h.2.2.2.1
  have h5 : e + a = 10 := h.2.2.2.2
  sorry

end find_a_plus_d_l69_69521


namespace wendy_total_sales_correct_l69_69027

noncomputable def wendy_total_sales : ℝ :=
  let morning_apples := 40 * 1.50
  let morning_oranges := 30 * 1
  let morning_bananas := 10 * 0.75
  let afternoon_apples := 50 * 1.35
  let afternoon_oranges := 40 * 0.90
  let afternoon_bananas := 20 * 0.675
  let unsold_bananas := 20 * 0.375
  let unsold_oranges := 10 * 0.50
  let total_morning := morning_apples + morning_oranges + morning_bananas
  let total_afternoon := afternoon_apples + afternoon_oranges + afternoon_bananas
  let total_day_sales := total_morning + total_afternoon
  let total_unsold_sales := unsold_bananas + unsold_oranges
  total_day_sales + total_unsold_sales

theorem wendy_total_sales_correct :
  wendy_total_sales = 227 := by
  unfold wendy_total_sales
  sorry

end wendy_total_sales_correct_l69_69027


namespace sqrt_condition_l69_69236

theorem sqrt_condition (x : ℝ) : (3 * x - 5 ≥ 0) → (x ≥ 5 / 3) :=
by
  intros h
  have h1 : 3 * x ≥ 5 := by linarith
  have h2 : x ≥ 5 / 3 := by linarith
  exact h2

end sqrt_condition_l69_69236


namespace average_weight_of_24_boys_l69_69022

theorem average_weight_of_24_boys (A : ℝ) : 
  (24 * A + 8 * 45.15) / 32 = 48.975 → A = 50.25 :=
by
  intro h
  sorry

end average_weight_of_24_boys_l69_69022


namespace simplify_fraction_l69_69602

theorem simplify_fraction (a b : ℕ) (h1 : a = 252) (h2 : b = 248) :
  (1000 ^ 2 : ℤ) / ((a ^ 2 - b ^ 2) : ℤ) = 500 := by
  sorry

end simplify_fraction_l69_69602


namespace almond_butter_servings_l69_69422

def servings_of_almond_butter (tbsp_in_container : ℚ) (tbsp_per_serving : ℚ) : ℚ :=
  tbsp_in_container / tbsp_per_serving

def container_holds : ℚ := 37 + 2/3

def serving_size : ℚ := 3

theorem almond_butter_servings :
  servings_of_almond_butter container_holds serving_size = 12 + 5/9 := 
by
  sorry

end almond_butter_servings_l69_69422


namespace find_g_8_l69_69162

def g (x : ℝ) : ℝ := x^2 + x + 1

theorem find_g_8 : (∀ x : ℝ, g (2*x - 4) = x^2 + x + 1) → g 8 = 43 := 
by sorry

end find_g_8_l69_69162


namespace tan3theta_l69_69033

theorem tan3theta (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 := 
by
  sorry

end tan3theta_l69_69033


namespace eq_solutions_count_l69_69974

theorem eq_solutions_count : 
  ∃! (n : ℕ), n = 126 ∧ (∀ x y : ℕ, 2*x + 3*y = 768 → x > 0 ∧ y > 0 → ∃ t : ℤ, x = 384 + 3*t ∧ y = -2*t ∧ -127 ≤ t ∧ t <= -1) := sorry

end eq_solutions_count_l69_69974


namespace solve_equation_l69_69011

theorem solve_equation (x : ℝ) : 2 * x - 4 = 0 ↔ x = 2 :=
by sorry

end solve_equation_l69_69011


namespace simplify_evaluate_expression_l69_69239

theorem simplify_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6)) = Real.sqrt 2 :=
by
  sorry

end simplify_evaluate_expression_l69_69239


namespace almost_perfect_numbers_l69_69477

def d (n : Nat) : Nat := 
  -- Implement the function to count the number of positive divisors of n
  sorry

def f (n : Nat) : Nat := 
  -- Implement the function f(n) as given in the problem statement
  sorry

def isAlmostPerfect (n : Nat) : Prop := 
  f n = n

theorem almost_perfect_numbers :
  ∀ n, isAlmostPerfect n → n = 1 ∨ n = 3 ∨ n = 18 ∨ n = 36 :=
by
  sorry

end almost_perfect_numbers_l69_69477


namespace stockholm_to_uppsala_distance_l69_69958

-- Definition of conditions
def map_distance : ℝ := 45 -- in cm
def scale1 : ℝ := 10 -- first scale 1 cm : 10 km
def scale2 : ℝ := 5 -- second scale 1 cm : 5 km
def boundary : ℝ := 15 -- first 15 cm at scale 2

-- Calculation of the two parts
def part1_distance (boundary : ℝ) (scale2 : ℝ) := boundary * scale2
def remaining_distance (map_distance boundary : ℝ) := map_distance - boundary
def part2_distance (remaining_distance : ℝ) (scale1 : ℝ) := remaining_distance * scale1

-- Total distance
def total_distance (part1 part2: ℝ) := part1 + part2

theorem stockholm_to_uppsala_distance : 
  total_distance (part1_distance boundary scale2) 
                 (part2_distance (remaining_distance map_distance boundary) scale1) 
  = 375 := 
by
  -- Proof to be provided
  sorry

end stockholm_to_uppsala_distance_l69_69958


namespace no_int_solutions_for_equation_l69_69461

theorem no_int_solutions_for_equation : 
  ∀ x y : ℤ, x ^ 2022 + y^2 = 2 * y + 2 → false := 
by
  -- By the given steps in the solution, we can conclude that no integer solutions exist
  sorry

end no_int_solutions_for_equation_l69_69461


namespace pascal_fifth_element_row_20_l69_69208

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_fifth_element_row_20 : binom 20 4 = 4845 := sorry

end pascal_fifth_element_row_20_l69_69208


namespace expand_product_polynomials_l69_69494

noncomputable def poly1 : Polynomial ℤ := 5 * Polynomial.X + 3
noncomputable def poly2 : Polynomial ℤ := 7 * Polynomial.X^2 + 2 * Polynomial.X + 4
noncomputable def expanded_form : Polynomial ℤ := 35 * Polynomial.X^3 + 31 * Polynomial.X^2 + 26 * Polynomial.X + 12

theorem expand_product_polynomials :
  poly1 * poly2 = expanded_form := 
by
  sorry

end expand_product_polynomials_l69_69494


namespace second_shirt_price_l69_69452

-- Define the conditions
def price_first_shirt := 82
def price_third_shirt := 90
def min_avg_price_remaining_shirts := 104
def total_shirts := 10
def desired_avg_price := 100

-- Prove the price of the second shirt
theorem second_shirt_price : 
  ∀ (P : ℝ), 
  (price_first_shirt + P + price_third_shirt + 7 * min_avg_price_remaining_shirts = total_shirts * desired_avg_price) → 
  P = 100 :=
by
  sorry

end second_shirt_price_l69_69452


namespace claire_initial_balloons_l69_69385

theorem claire_initial_balloons (B : ℕ) (h : B - 12 - 9 + 11 = 39) : B = 49 :=
by sorry

end claire_initial_balloons_l69_69385


namespace mass_percentage_C_is_54_55_l69_69506

def mass_percentage (C: String) (percentage: ℝ) : Prop :=
  percentage = 54.55

theorem mass_percentage_C_is_54_55 :
  mass_percentage "C" 54.55 :=
by
  unfold mass_percentage
  rfl

end mass_percentage_C_is_54_55_l69_69506


namespace fewer_free_throws_l69_69787

noncomputable def Deshawn_free_throws : ℕ := 12
noncomputable def Kayla_free_throws : ℕ := Deshawn_free_throws + (Deshawn_free_throws / 2)
noncomputable def Annieka_free_throws : ℕ := 14

theorem fewer_free_throws :
  Annieka_free_throws = Kayla_free_throws - 4 :=
by
  sorry

end fewer_free_throws_l69_69787


namespace evaluate_expression_l69_69017

theorem evaluate_expression (a : ℕ) (h : a = 2) : (7 * a ^ 2 - 10 * a + 3) * (3 * a - 4) = 22 :=
by
  -- Here would be the proof which is omitted as per instructions
  sorry

end evaluate_expression_l69_69017


namespace find_train_speed_l69_69512

variable (bridge_length train_length train_crossing_time : ℕ)

def speed_of_train (bridge_length train_length train_crossing_time : ℕ) : ℕ :=
  (bridge_length + train_length) / train_crossing_time

theorem find_train_speed
  (bridge_length : ℕ) (train_length : ℕ) (train_crossing_time : ℕ)
  (h_bridge_length : bridge_length = 180)
  (h_train_length : train_length = 120)
  (h_train_crossing_time : train_crossing_time = 20) :
  speed_of_train bridge_length train_length train_crossing_time = 15 := by
  sorry

end find_train_speed_l69_69512


namespace janet_total_l69_69614

-- Definitions based on the conditions
variable (initial_collect : ℕ) (sold : ℕ) (better_cond : ℕ)
variable (twice_size : ℕ)

-- The conditions from part a)
def janet_initial_collection := initial_collect = 10
def janet_sells := sold = 6
def janet_gets_better := better_cond = 4
def brother_gives := twice_size = 2 * (initial_collect - sold + better_cond)

-- The proof statement based on part c)
theorem janet_total (initial_collect sold better_cond twice_size : ℕ) : 
    janet_initial_collection initial_collect →
    janet_sells sold →
    janet_gets_better better_cond →
    brother_gives initial_collect sold better_cond twice_size →
    (initial_collect - sold + better_cond + twice_size = 24) :=
by
  intros h1 h2 h3 h4
  sorry

end janet_total_l69_69614


namespace xy_squared_l69_69292

theorem xy_squared (x y : ℚ) (h1 : x + y = 9 / 20) (h2 : x - y = 1 / 20) :
  x^2 - y^2 = 9 / 400 :=
by
  sorry

end xy_squared_l69_69292


namespace find_y_l69_69806

theorem find_y (y : ℝ) (h : 3 * y / 7 = 21) : y = 49 := 
sorry

end find_y_l69_69806


namespace factor_expression_l69_69593

theorem factor_expression (x : ℝ) :
  (3*x^3 + 48*x^2 - 14) - (-9*x^3 + 2*x^2 - 14) =
  2*x^2 * (6*x + 23) :=
by
  sorry

end factor_expression_l69_69593


namespace meeting_attendance_l69_69537

theorem meeting_attendance (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + 2 * B = 11) : A + B = 6 :=
sorry

end meeting_attendance_l69_69537


namespace equation1_solutions_equation2_solutions_l69_69282

theorem equation1_solutions (x : ℝ) :
  (4 * x^2 = 12 * x) ↔ (x = 0 ∨ x = 3) := by
sorry

theorem equation2_solutions (x : ℝ) :
  ((3 / 4) * x^2 - 2 * x - (1 / 2) = 0) ↔ (x = (4 + Real.sqrt 22) / 3 ∨ x = (4 - Real.sqrt 22) / 3) := by
sorry

end equation1_solutions_equation2_solutions_l69_69282


namespace positive_integer_solutions_l69_69443

theorem positive_integer_solutions (a b : ℕ) (h_pos_ab : 0 < a ∧ 0 < b) :
  (∃ k : ℕ, k = a^2 / (2 * a * b^2 - b^3 + 1) ∧ 0 < k) ↔
  ∃ n : ℕ, (a = 2 * n ∧ b = 1) ∨ (a = n ∧ b = 2 * n) ∨ (a = 8 * n^4 - n ∧ b = 2 * n) :=
by
  sorry

end positive_integer_solutions_l69_69443


namespace geometric_sum_six_l69_69322

theorem geometric_sum_six (a r : ℚ) (n : ℕ) 
  (hn₁ : a = 1/4) 
  (hn₂ : r = 1/2) 
  (hS: a * (1 - r^n) / (1 - r) = 63/128) : 
  n = 6 :=
by
  -- Statement to be Proven
  rw [hn₁, hn₂] at hS
  sorry

end geometric_sum_six_l69_69322


namespace line_parameterization_l69_69098

theorem line_parameterization (r k : ℝ) (t : ℝ) :
  (∀ x y : ℝ, (x, y) = (r + 3 * t, 2 + k * t) → (y = 2 * x - 5) ) ∧
  (t = 0 → r = 7 / 2) ∧
  (t = 1 → k = 6) :=
by
  sorry

end line_parameterization_l69_69098


namespace triangle_angle_y_l69_69658

theorem triangle_angle_y (y : ℝ) (h : y + 3 * y + 45 = 180) : y = 33.75 :=
by
  have h1 : 4 * y + 45 = 180 := by sorry
  have h2 : 4 * y = 135 := by sorry
  have h3 : y = 33.75 := by sorry
  exact h3

end triangle_angle_y_l69_69658


namespace cryptarithm_solved_l69_69281

-- Definitions for the digits A, B, C
def valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

-- Given conditions, where A, B, C are distinct non-zero digits
def conditions (A B C : ℕ) : Prop :=
  valid_digit A ∧ valid_digit B ∧ valid_digit C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C

-- Definitions of the two-digit and three-digit numbers
def two_digit (A B : ℕ) : ℕ := 10 * A + B
def three_digit_rep (C : ℕ) : ℕ := 111 * C

-- Main statement of the proof problem
theorem cryptarithm_solved (A B C : ℕ) (h : conditions A B C) :
  two_digit A B + A * three_digit_rep C = 247 → A * 100 + B * 10 + C = 251 :=
sorry -- Proof goes here

end cryptarithm_solved_l69_69281


namespace trains_crossing_l69_69214

noncomputable def time_to_cross_each_other (v : ℝ) (L₁ L₂ : ℝ) (t₁ t₂ : ℝ) : ℝ :=
  (L₁ + L₂) / (2 * v)

theorem trains_crossing (v : ℝ) (t₁ t₂ : ℝ) (h1 : t₁ = 27) (h2 : t₂ = 17) :
  time_to_cross_each_other v (v * 27) (v * 17) t₁ t₂ = 22 :=
by
  -- Conditions
  have h3 : t₁ = 27 := h1
  have h4 : t₂ = 17 := h2
  -- Proof outline (not needed, just to ensure the setup is understood):
  -- Lengths
  let L₁ := v * 27
  let L₂ := v * 17
  -- Calculating Crossing Time
  have t := (L₁ + L₂) / (2 * v)
  -- Simplification leads to t = 22
  sorry

end trains_crossing_l69_69214


namespace tan_theta_eq_sqrt_3_of_f_maximum_l69_69230

theorem tan_theta_eq_sqrt_3_of_f_maximum (θ : ℝ) 
  (h : ∀ x : ℝ, 3 * Real.sin (x + (Real.pi / 6)) ≤ 3 * Real.sin (θ + (Real.pi / 6))) : 
  Real.tan θ = Real.sqrt 3 :=
sorry

end tan_theta_eq_sqrt_3_of_f_maximum_l69_69230


namespace gift_card_remaining_l69_69284

theorem gift_card_remaining (initial_amount : ℕ) (half_monday : ℕ) (quarter_tuesday : ℕ) : 
  initial_amount = 200 → 
  half_monday = initial_amount / 2 →
  quarter_tuesday = (initial_amount - half_monday) / 4 →
  initial_amount - half_monday - quarter_tuesday = 75 :=
by
  intros h_init h_half h_quarter
  rw [h_init, h_half, h_quarter]
  sorry

end gift_card_remaining_l69_69284


namespace total_amount_paid_correct_l69_69767

/--
Given:
1. The marked price of each article is $17.5.
2. A discount of 30% was applied to the total marked price of the pair of articles.

Prove:
The total amount paid for the pair of articles is $24.5.
-/
def total_amount_paid (marked_price_each : ℝ) (discount_rate : ℝ) : ℝ :=
  let marked_price_pair := marked_price_each * 2
  let discount := discount_rate * marked_price_pair
  marked_price_pair - discount

theorem total_amount_paid_correct :
  total_amount_paid 17.5 0.30 = 24.5 :=
by
  sorry

end total_amount_paid_correct_l69_69767


namespace farmer_rent_l69_69874

-- Definitions based on given conditions
def rent_per_acre_per_month : ℕ := 60
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Problem statement: 
-- Prove that the monthly rent to rent the rectangular plot is $600.
theorem farmer_rent : 
  (length_of_plot * width_of_plot) / square_feet_per_acre * rent_per_acre_per_month = 600 :=
by
  sorry

end farmer_rent_l69_69874


namespace square_perimeter_eq_area_perimeter_16_l69_69260

theorem square_perimeter_eq_area_perimeter_16 (s : ℕ) (h : s^2 = 4 * s) : 4 * s = 16 := by
  sorry

end square_perimeter_eq_area_perimeter_16_l69_69260


namespace div_decimals_l69_69253

theorem div_decimals : 0.45 / 0.005 = 90 := sorry

end div_decimals_l69_69253


namespace complex_ratio_max_min_diff_l69_69003

noncomputable def max_minus_min_complex_ratio (z w : ℂ) : ℝ :=
max (1 : ℝ) (0 : ℝ) - min (1 : ℝ) (0 : ℝ)

theorem complex_ratio_max_min_diff (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) : 
  max_minus_min_complex_ratio z w = 1 :=
by sorry

end complex_ratio_max_min_diff_l69_69003


namespace factor_expression_l69_69979

theorem factor_expression (x : ℝ) : 25 * x^2 + 10 * x = 5 * x * (5 * x + 2) :=
sorry

end factor_expression_l69_69979


namespace total_votes_l69_69201

theorem total_votes (A B C V : ℝ)
  (h1 : A = B + 0.10 * V)
  (h2 : A = C + 0.15 * V)
  (h3 : A - 3000 = B + 3000)
  (h4 : B + 3000 = A - 0.10 * V)
  (h5 : B + 3000 = C + 0.05 * V)
  : V = 60000 := 
sorry

end total_votes_l69_69201


namespace div_36_of_n_ge_5_l69_69029

noncomputable def n := Nat

theorem div_36_of_n_ge_5 (n : ℕ) (hn : n ≥ 5) (h2 : ¬ (n % 2 = 0)) (h3 : ¬ (n % 3 = 0)) : 36 ∣ (n^2 - 1) :=
by
  sorry

end div_36_of_n_ge_5_l69_69029


namespace intersection_A_B_l69_69884
-- Lean 4 code statement

def set_A : Set ℝ := {x | |x - 1| > 2}
def set_B : Set ℝ := {x | x * (x - 5) < 0}
def set_intersection : Set ℝ := {x | 3 < x ∧ x < 5}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection := by
  sorry

end intersection_A_B_l69_69884


namespace inequality_problem_l69_69473

noncomputable def a := (3 / 4) * Real.exp (2 / 5)
noncomputable def b := 2 / 5
noncomputable def c := (2 / 5) * Real.exp (3 / 4)

theorem inequality_problem : b < c ∧ c < a := by
  sorry

end inequality_problem_l69_69473


namespace largest_equal_cost_l69_69437

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_digit_sum (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem largest_equal_cost :
  ∃ (n : ℕ), n < 500 ∧ digit_sum n = binary_digit_sum n ∧ ∀ m < 500, digit_sum m = binary_digit_sum m → m ≤ 247 :=
by
  sorry

end largest_equal_cost_l69_69437


namespace dot_product_result_l69_69467

open scoped BigOperators

-- Define the vectors a and b
def a : ℝ × ℝ := (2, -3)
def b : ℝ × ℝ := (-1, 2)

-- Define the addition of two vectors
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_result : dot_product (vector_add a b) a = 5 := by
  sorry

end dot_product_result_l69_69467


namespace heights_inequality_l69_69951

theorem heights_inequality (a b c h_a h_b h_c p R : ℝ) (h₁ : a ≤ b) (h₂ : b ≤ c) :
  h_a + h_b + h_c ≤ (3 * b * (a^2 + a * c + c^2)) / (4 * p * R) :=
by
  sorry

end heights_inequality_l69_69951


namespace men_work_in_80_days_l69_69873

theorem men_work_in_80_days (x : ℕ) (work_eq_20men_56days : x * 80 = 20 * 56) : x = 14 :=
by 
  sorry

end men_work_in_80_days_l69_69873


namespace find_vector_b_coordinates_l69_69081

theorem find_vector_b_coordinates 
  (a b : ℝ × ℝ) 
  (h₁ : a = (-3, 4)) 
  (h₂ : ∃ m : ℝ, m < 0 ∧ b = (-3 * m, 4 * m)) 
  (h₃ : ‖b‖ = 10) : 
  b = (6, -8) := 
by
  sorry

end find_vector_b_coordinates_l69_69081


namespace geometric_sequence_sum_l69_69758

variable {α : Type*} 
variable [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → α) (h : is_geometric_sequence a) 
  (h1 : a 0 + a 1 = 20) 
  (h2 : a 2 + a 3 = 40) : 
  a 4 + a 5 = 80 :=
sorry

end geometric_sequence_sum_l69_69758


namespace trajectory_midpoint_chord_l69_69425

theorem trajectory_midpoint_chord (x y : ℝ) 
  (h₀ : y^2 = 4 * x) : (y^2 = 2 * x - 2) :=
sorry

end trajectory_midpoint_chord_l69_69425


namespace triangle_cosine_l69_69924

theorem triangle_cosine (LM : ℝ) (cos_N : ℝ) (LN : ℝ) (h1 : LM = 20) (h2 : cos_N = 3/5) :
  LM / LN = cos_N → LN = 100 / 3 :=
by
  intro h3
  sorry

end triangle_cosine_l69_69924


namespace translate_point_correct_l69_69067

def P : ℝ × ℝ := (2, 3)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

theorem translate_point_correct :
  translate_down (translate_left P 3) 4 = (-1, -1) :=
by
  sorry

end translate_point_correct_l69_69067


namespace ellipse_properties_l69_69311

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2) / 4 + y^2 = 1

noncomputable def trajectory_equation_midpoint (x y : ℝ) : Prop :=
  ((2 * x - 1)^2) / 4 + (2 * y - 1 / 2)^2 = 1

theorem ellipse_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y) ∧
  (∀ x y : ℝ, trajectory_equation_midpoint x y) :=
by
  sorry

end ellipse_properties_l69_69311


namespace episodes_per_season_l69_69417

theorem episodes_per_season (S : ℕ) (E : ℕ) (H1 : S = 12) (H2 : 2/3 * E = 160) : E / S = 20 :=
by
  sorry

end episodes_per_season_l69_69417


namespace sum_of_real_roots_l69_69057

theorem sum_of_real_roots (P : Polynomial ℝ) (hP : P = Polynomial.C 1 * X^4 - Polynomial.C 8 * X - Polynomial.C 2) :
  P.roots.sum = 2 :=
by {
  sorry
}

end sum_of_real_roots_l69_69057


namespace find_b_l69_69130

theorem find_b (b : ℝ) (h : ∃ x : ℝ, x^2 + b*x - 35 = 0 ∧ x = -5) : b = -2 :=
by
  sorry

end find_b_l69_69130


namespace goose_eggs_laid_l69_69345

theorem goose_eggs_laid (E : ℕ) 
    (H1 : ∃ h, h = (2 / 5) * E)
    (H2 : ∃ m, m = (11 / 15) * h)
    (H3 : ∃ s, s = (1 / 4) * m)
    (H4 : ∃ y, y = (2 / 7) * s)
    (H5 : y = 150) : 
    E = 7160 := 
sorry

end goose_eggs_laid_l69_69345


namespace triple_comp_g_of_2_l69_69581

def g (n : ℕ) : ℕ :=
  if n ≤ 3 then n^3 - 2 else 4 * n + 1

theorem triple_comp_g_of_2 : g (g (g 2)) = 101 := by
  sorry

end triple_comp_g_of_2_l69_69581


namespace average_speed_is_correct_l69_69495

-- Definitions for the conditions
def speed_first_hour : ℕ := 140
def speed_second_hour : ℕ := 40
def total_distance : ℕ := speed_first_hour + speed_second_hour
def total_time : ℕ := 2

-- The statement we need to prove
theorem average_speed_is_correct : total_distance / total_time = 90 := by
  -- We would place the proof here
  sorry

end average_speed_is_correct_l69_69495


namespace part1_part2_l69_69327

noncomputable def f (a x : ℝ) : ℝ := a * x + x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x, x ≥ Real.exp 1 → (a + 1 + Real.log x) ≥ 0) →
  a ≥ -2 :=
by
  sorry

theorem part2 (k : ℤ) :
  (∀ x, 1 < x → (k : ℝ) * (x - 1) < f 1 x) →
  k ≤ 3 :=
by
  sorry

end part1_part2_l69_69327


namespace focal_length_ellipse_l69_69045

theorem focal_length_ellipse :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 :=
by
  sorry

end focal_length_ellipse_l69_69045


namespace joyce_initial_eggs_l69_69427

theorem joyce_initial_eggs :
  ∃ E : ℕ, (E + 6 = 14) ∧ E = 8 :=
sorry

end joyce_initial_eggs_l69_69427


namespace dice_probability_four_less_than_five_l69_69892

noncomputable def probability_exactly_four_less_than_five (n : ℕ) : ℚ :=
  if n = 8 then (Nat.choose 8 4) * (1 / 2)^8 else 0

theorem dice_probability_four_less_than_five : probability_exactly_four_less_than_five 8 = 35 / 128 :=
by
  -- statement is correct, proof to be provided
  sorry

end dice_probability_four_less_than_five_l69_69892


namespace friends_Sarah_brought_l69_69564

def total_people_in_house : Nat := 15
def in_bedroom : Nat := 2
def living_room : Nat := 8
def Sarah : Nat := 1

theorem friends_Sarah_brought :
  total_people_in_house - (in_bedroom + Sarah + living_room) = 4 := by
  sorry

end friends_Sarah_brought_l69_69564


namespace sum_of_angles_l69_69302

namespace BridgeProblem

def is_isosceles (A B C : Type) (AB AC : ℝ) : Prop := AB = AC

def angle_bac (A B C : Type) : ℝ := 15

def angle_edf (D E F : Type) : ℝ := 45

theorem sum_of_angles (A B C D E F : Type) 
  (h_isosceles_ABC : is_isosceles A B C 1 1)
  (h_isosceles_DEF : is_isosceles D E F 1 1)
  (h_angle_BAC : angle_bac A B C = 15)
  (h_angle_EDF : angle_edf D E F = 45) :
  true := 
by 
  sorry

end BridgeProblem

end sum_of_angles_l69_69302


namespace edge_of_new_cube_l69_69266

theorem edge_of_new_cube (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) :
  ∃ d : ℝ, d^3 = a^3 + b^3 + c^3 ∧ d = 12 :=
by
  sorry

end edge_of_new_cube_l69_69266


namespace band_member_earnings_l69_69517

theorem band_member_earnings :
  let attendees := 500
  let ticket_price := 30
  let band_share_percentage := 70 / 100
  let band_members := 4
  let total_earnings := attendees * ticket_price
  let band_earnings := total_earnings * band_share_percentage
  let earnings_per_member := band_earnings / band_members
  earnings_per_member = 2625 := 
by {
  sorry
}

end band_member_earnings_l69_69517


namespace non_zero_real_solution_of_equation_l69_69990

noncomputable def equation_solution : Prop :=
  ∀ (x : ℝ), x ≠ 0 ∧ (7 * x) ^ 14 = (14 * x) ^ 7 → x = 2 / 7

theorem non_zero_real_solution_of_equation : equation_solution := sorry

end non_zero_real_solution_of_equation_l69_69990


namespace smallest_n_l69_69353

theorem smallest_n (n : ℕ) (h : 0 < n) : 
  (1 / (n : ℝ)) - (1 / (n + 1 : ℝ)) < 1 / 15 → n = 4 := sorry

end smallest_n_l69_69353


namespace equal_pieces_length_l69_69545

theorem equal_pieces_length (total_length_cm : ℕ) (num_pieces : ℕ) (num_equal_pieces : ℕ) (length_remaining_piece_mm : ℕ) :
  total_length_cm = 1165 ∧ num_pieces = 154 ∧ num_equal_pieces = 150 ∧ length_remaining_piece_mm = 100 →
  (total_length_cm * 10 - (num_pieces - num_equal_pieces) * length_remaining_piece_mm) / num_equal_pieces = 75 :=
by
  sorry

end equal_pieces_length_l69_69545


namespace sin_2phi_l69_69432

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l69_69432


namespace original_profit_percentage_l69_69171

theorem original_profit_percentage {P S : ℝ}
  (h1 : S = 1100)
  (h2 : P ≠ 0)
  (h3 : 1.17 * P = 1170) :
  (S - P) / P * 100 = 10 :=
by
  sorry

end original_profit_percentage_l69_69171


namespace locus_of_C_l69_69486

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)

theorem locus_of_C :
  ∀ (C : ℝ × ℝ), (C.2 = (b / a) * C.1 ∧ (a * b / Real.sqrt (a ^ 2 + b ^ 2) ≤ C.1) ∧ (C.1 ≤ a)) :=
sorry

end locus_of_C_l69_69486


namespace factor_expression_l69_69837

theorem factor_expression (a b c : ℝ) : 
  ( (a^2 - b^2)^4 + (b^2 - c^2)^4 + (c^2 - a^2)^4 ) / 
  ( (a - b)^4 + (b - c)^4 + (c - a)^4 ) = 1 := 
by sorry

end factor_expression_l69_69837


namespace find_base_length_of_isosceles_triangle_l69_69231

noncomputable def is_isosceles_triangle_with_base_len (a b : ℝ) : Prop :=
  a = 2 ∧ ((a + a + b = 5) ∨ (a + b + b = 5))

theorem find_base_length_of_isosceles_triangle :
  ∃ (b : ℝ), is_isosceles_triangle_with_base_len 2 b ∧ (b = 1.5 ∨ b = 2) :=
by
  sorry

end find_base_length_of_isosceles_triangle_l69_69231


namespace solve_for_a_l69_69007

theorem solve_for_a (x : ℤ) (a : ℤ) (h : 3 * x + 2 * a + 1 = 2) (hx : x = -1) : a = 2 :=
by
  sorry

end solve_for_a_l69_69007


namespace final_limes_count_l69_69182

def limes_initial : ℕ := 9
def limes_by_Sara : ℕ := 4
def limes_used_for_juice : ℕ := 5
def limes_given_to_neighbor : ℕ := 3

theorem final_limes_count :
  limes_initial + limes_by_Sara - limes_used_for_juice - limes_given_to_neighbor = 5 :=
by
  sorry

end final_limes_count_l69_69182


namespace part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l69_69630

-- Definition of a good number
def is_good (n : ℕ) : Prop := (n % 6 = 3)

-- Lean 4 statements

-- 1. 2001 is good
theorem part_a_2001_good : is_good 2001 :=
by sorry

-- 2. 3001 isn't good
theorem part_a_3001_not_good : ¬ is_good 3001 :=
by sorry

-- 3. The product of two good numbers is a good number
theorem part_b_product_of_good_is_good (x y : ℕ) (hx : is_good x) (hy : is_good y) : is_good (x * y) :=
by sorry

-- 4. If the product of two numbers is good, then at least one of the numbers is good
theorem part_c_product_good_then_one_good (x y : ℕ) (hxy : is_good (x * y)) : is_good x ∨ is_good y :=
by sorry

end part_a_2001_good_part_a_3001_not_good_part_b_product_of_good_is_good_part_c_product_good_then_one_good_l69_69630


namespace arithmetic_mean_l69_69983

variable (x b : ℝ)

theorem arithmetic_mean (hx : x ≠ 0) :
  ((x + b) / x + (x - 2 * b) / x) / 2 = 1 - b / (2 * x) := by
  sorry

end arithmetic_mean_l69_69983


namespace isosceles_triangle_perimeter_l69_69926

-- Define the conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

-- Define the side lengths
def side1 := 2
def side2 := 2
def base := 5

-- Define the perimeter
def perimeter (a b c : ℝ) := a + b + c

-- State the theorem
theorem isosceles_triangle_perimeter : isosceles_triangle side1 side2 base → perimeter side1 side2 base = 9 :=
  by sorry

end isosceles_triangle_perimeter_l69_69926


namespace pens_to_sell_to_make_profit_l69_69629

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

end pens_to_sell_to_make_profit_l69_69629


namespace train_length_l69_69372

theorem train_length (v : ℝ) (t : ℝ) (l_b : ℝ) (v_r : v = 52) (t_r : t = 34.61538461538461) (l_b_r : l_b = 140) : 
  ∃ l_t : ℝ, l_t = 360 :=
by
  have speed_ms := v * (1000 / 3600)
  have total_distance := speed_ms * t
  have length_train := total_distance - l_b
  use length_train
  sorry

end train_length_l69_69372


namespace scheduling_competitions_l69_69798

-- Define the problem conditions
def scheduling_conditions (gyms : ℕ) (sports : ℕ) (max_sports_per_gym : ℕ) : Prop :=
  gyms = 4 ∧ sports = 3 ∧ max_sports_per_gym = 2

-- Define the main statement
theorem scheduling_competitions :
  scheduling_conditions 4 3 2 →
  (number_of_arrangements = 60) :=
by
  sorry

end scheduling_competitions_l69_69798


namespace ned_weekly_revenue_l69_69229

-- Conditions
def normal_mouse_cost : ℕ := 120
def percentage_increase : ℕ := 30
def mice_sold_per_day : ℕ := 25
def days_store_is_open_per_week : ℕ := 4

-- Calculate cost of a left-handed mouse
def left_handed_mouse_cost : ℕ := normal_mouse_cost + (normal_mouse_cost * percentage_increase / 100)

-- Calculate daily revenue
def daily_revenue : ℕ := mice_sold_per_day * left_handed_mouse_cost

-- Calculate weekly revenue
def weekly_revenue : ℕ := daily_revenue * days_store_is_open_per_week

-- Theorem to prove
theorem ned_weekly_revenue : weekly_revenue = 15600 := 
by 
  sorry

end ned_weekly_revenue_l69_69229


namespace polynomial_divisibility_l69_69532

theorem polynomial_divisibility (C D : ℝ)
  (h : ∀ x, x^2 + x + 1 = 0 → x^102 + C * x + D = 0) :
  C + D = -1 := 
by 
  sorry

end polynomial_divisibility_l69_69532


namespace total_revenue_correct_l69_69259

noncomputable def revenue_calculation : ℕ :=
  let fair_tickets := 60
  let fair_price := 15
  let baseball_tickets := fair_tickets / 3
  let baseball_price := 10
  let play_tickets := 2 * fair_tickets
  let play_price := 12
  fair_tickets * fair_price
  + baseball_tickets * baseball_price
  + play_tickets * play_price

theorem total_revenue_correct : revenue_calculation = 2540 :=
  by
  sorry

end total_revenue_correct_l69_69259


namespace distance_by_land_l69_69412

theorem distance_by_land (distance_by_sea total_distance distance_by_land : ℕ)
  (h1 : total_distance = 601)
  (h2 : distance_by_sea = 150)
  (h3 : total_distance = distance_by_land + distance_by_sea) : distance_by_land = 451 := by
  sorry

end distance_by_land_l69_69412


namespace sum_of_products_l69_69599

theorem sum_of_products : 1 * 15 + 2 * 14 + 3 * 13 + 4 * 12 + 5 * 11 + 6 * 10 + 7 * 9 + 8 * 8 = 372 := by
  sorry

end sum_of_products_l69_69599


namespace karen_nuts_l69_69736

/-- Karen added 0.25 cup of walnuts to a batch of trail mix.
Later, she added 0.25 cup of almonds.
In all, Karen put 0.5 cups of nuts in the trail mix. -/
theorem karen_nuts (walnuts almonds : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_almonds : almonds = 0.25) : 
  walnuts + almonds = 0.5 := 
by
  sorry

end karen_nuts_l69_69736


namespace relationship_between_c_and_d_l69_69251

noncomputable def c : ℝ := Real.log 400 / Real.log 4
noncomputable def d : ℝ := Real.log 20 / Real.log 2

theorem relationship_between_c_and_d : c = d := by
  sorry

end relationship_between_c_and_d_l69_69251


namespace student_loses_one_mark_per_wrong_answer_l69_69772

noncomputable def marks_lost_per_wrong_answer (x : ℝ) : Prop :=
  let total_questions := 60
  let correct_answers := 42
  let wrong_answers := total_questions - correct_answers
  let marks_per_correct := 4
  let total_marks := 150
  correct_answers * marks_per_correct - wrong_answers * x = total_marks

theorem student_loses_one_mark_per_wrong_answer : marks_lost_per_wrong_answer 1 :=
by
  sorry

end student_loses_one_mark_per_wrong_answer_l69_69772


namespace coprime_ab_and_a_plus_b_l69_69200

theorem coprime_ab_and_a_plus_b (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a * b) (a + b) = 1 := by
  sorry

end coprime_ab_and_a_plus_b_l69_69200


namespace find_missing_number_l69_69280

noncomputable def missing_number : Prop :=
  ∃ (y x a b : ℝ),
    a = y + x ∧
    b = x + 630 ∧
    28 = y * a ∧
    660 = a * b ∧
    y = 13

theorem find_missing_number : missing_number :=
  sorry

end find_missing_number_l69_69280


namespace complex_sum_l69_69888

-- Define the given condition as a hypothesis
variables {z : ℂ} (h : z^2 + z + 1 = 0)

-- Define the statement to prove
theorem complex_sum (h : z^2 + z + 1 = 0) : z^96 + z^97 + z^98 + z^99 + z^100 + z^101 = 0 :=
sorry

end complex_sum_l69_69888


namespace smallest_a_plus_b_l69_69069

theorem smallest_a_plus_b (a b : ℕ) (h1: 0 < a) (h2: 0 < b) (h3 : 2^10 * 7^3 = a^b) : a + b = 31 :=
sorry

end smallest_a_plus_b_l69_69069


namespace kneading_time_is_correct_l69_69308

def total_time := 280
def rising_time_per_session := 120
def number_of_rising_sessions := 2
def baking_time := 30

def total_rising_time := rising_time_per_session * number_of_rising_sessions
def total_non_kneading_time := total_rising_time + baking_time
def kneading_time := total_time - total_non_kneading_time

theorem kneading_time_is_correct : kneading_time = 10 := by
  have h1 : total_rising_time = 240 := by
    sorry
  have h2 : total_non_kneading_time = 270 := by
    sorry
  have h3 : kneading_time = 10 := by
    sorry
  exact h3

end kneading_time_is_correct_l69_69308


namespace find_rth_term_l69_69989

theorem find_rth_term (n r : ℕ) (S : ℕ → ℕ) (hS : ∀ n, S n = 4 * n + 5 * n^2) :
  r > 0 → (S r) - (S (r - 1)) = 10 * r - 1 :=
by
  intro h
  have hr_pos := h
  sorry

end find_rth_term_l69_69989


namespace length_of_AC_l69_69615

theorem length_of_AC (AB : ℝ) (C : ℝ) (h1 : AB = 4) (h2 : 0 < C) (h3 : C < AB) (mean_proportional : C * C = AB * (AB - C)) :
  C = 2 * Real.sqrt 5 - 2 := 
sorry

end length_of_AC_l69_69615


namespace part1_optimal_strategy_part2_optimal_strategy_l69_69006

noncomputable def R (x1 x2 : ℝ) : ℝ := -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

theorem part1_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 + x2 = 5 ∧ x1 = 2 ∧ x2 = 3 ∧
    ∀ y1 y2, y1 + y2 = 5 → (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

theorem part2_optimal_strategy :
  ∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = 5 ∧
    ∀ y1 y2, (R y1 y2 - (y1 + y2) ≤ R x1 x2 - (x1 + x2)) := 
by
  sorry

end part1_optimal_strategy_part2_optimal_strategy_l69_69006


namespace profit_condition_maximize_profit_l69_69821

noncomputable def profit (x : ℕ) : ℕ := 
  (x + 10) * (300 - 10 * x)

theorem profit_condition (x : ℕ) : profit x = 3360 ↔ x = 2 ∨ x = 18 := by
  sorry

theorem maximize_profit : ∃ x, x = 10 ∧ profit x = 4000 := by
  sorry

end profit_condition_maximize_profit_l69_69821


namespace decimal_representation_l69_69028

theorem decimal_representation :
  (13 : ℝ) / (2 * 5^8) = 0.00001664 := 
  sorry

end decimal_representation_l69_69028


namespace koala_fiber_eaten_l69_69565

-- Definitions based on conditions
def absorbs_percentage : ℝ := 0.40
def fiber_absorbed : ℝ := 12

-- The theorem statement to prove the total amount of fiber eaten
theorem koala_fiber_eaten : 
  (fiber_absorbed / absorbs_percentage) = 30 :=
by 
  sorry

end koala_fiber_eaten_l69_69565


namespace number_of_possible_values_l69_69020

theorem number_of_possible_values (x : ℕ) (h1 : x > 6) (h2 : x + 4 > 0) :
  ∃ (n : ℕ), n = 24 := 
sorry

end number_of_possible_values_l69_69020


namespace number_of_children_proof_l69_69264

-- Let A be the number of mushrooms Anya has
-- Let V be the number of mushrooms Vitya has
-- Let S be the number of mushrooms Sasha has
-- Let xs be the list of mushrooms of other children

def mushrooms_distribution (A V S : ℕ) (xs : List ℕ) : Prop :=
  let n := 3 + xs.length
  -- First condition
  let total_mushrooms := A + V + S + xs.sum
  let equal_share := total_mushrooms / n
  (A / 2 = equal_share) ∧ (V + A / 2 = equal_share) ∧ (S = equal_share) ∧
  (∀ x ∈ xs, x = equal_share) ∧
  -- Second condition
  (S + A = V + xs.sum)

theorem number_of_children_proof (A V S : ℕ) (xs : List ℕ) :
  mushrooms_distribution A V S xs → 3 + xs.length = 6 :=
by
  intros h
  sorry

end number_of_children_proof_l69_69264


namespace apple_cost_l69_69255

theorem apple_cost (cost_per_pound : ℚ) (weight : ℚ) (total_cost : ℚ) : cost_per_pound = 1 ∧ weight = 18 → total_cost = 18 :=
by
  sorry

end apple_cost_l69_69255


namespace side_length_of_square_perimeter_of_square_l69_69854

theorem side_length_of_square {d s: ℝ} (h: d = 2 * Real.sqrt 2): s = 2 :=
by
  sorry

theorem perimeter_of_square {s P: ℝ} (h: s = 2): P = 8 :=
by
  sorry

end side_length_of_square_perimeter_of_square_l69_69854


namespace probability_of_three_tails_one_head_in_four_tosses_l69_69087

noncomputable def probability_three_tails_one_head (n : ℕ) : ℚ :=
  if n = 4 then 1 / 4 else 0

theorem probability_of_three_tails_one_head_in_four_tosses :
  probability_three_tails_one_head 4 = 1 / 4 :=
by sorry

end probability_of_three_tails_one_head_in_four_tosses_l69_69087


namespace domain_of_f_comp_l69_69447

theorem domain_of_f_comp (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ x^2 - 2 ∧ x^2 - 2 ≤ -1) →
  (∀ x, - (4 : ℝ) / 3 ≤ x ∧ x ≤ -1 → -2 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ -1) :=
by
  sorry

end domain_of_f_comp_l69_69447


namespace number_of_pairs_l69_69965

theorem number_of_pairs :
  (∃ (pairs : Finset (ℕ × ℕ)), (∀ (pair : ℕ × ℕ), pair ∈ pairs → 1 ≤ pair.1 ∧ pair.1 ≤ 30 ∧ 3 ≤ pair.2 ∧ pair.2 ≤ 30 ∧ (pair.1 % pair.2 = 0) ∧ (pair.1 % (pair.2 - 2) = 0)) ∧ pairs.card = 22) := by
  sorry

end number_of_pairs_l69_69965


namespace abs_inequality_solution_l69_69448

theorem abs_inequality_solution :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by
  sorry

end abs_inequality_solution_l69_69448


namespace cos_beta_calculation_l69_69683

variable (α β : ℝ)
variable (h1 : 0 < α ∧ α < π / 2) -- α is an acute angle
variable (h2 : 0 < β ∧ β < π / 2) -- β is an acute angle
variable (h3 : Real.cos α = Real.sqrt 5 / 5)
variable (h4 : Real.sin (α - β) = Real.sqrt 10 / 10)

theorem cos_beta_calculation :
  Real.cos β = Real.sqrt 2 / 2 :=
  sorry

end cos_beta_calculation_l69_69683


namespace dog_paws_ground_l69_69604

theorem dog_paws_ground (total_dogs : ℕ) (two_thirds_back_legs : ℕ) (remaining_dogs_four_legs : ℕ) (two_paws_per_back_leg_dog : ℕ) (four_paws_per_four_leg_dog : ℕ) :
  total_dogs = 24 →
  two_thirds_back_legs = 2 * total_dogs / 3 →
  remaining_dogs_four_legs = total_dogs - two_thirds_back_legs →
  two_paws_per_back_leg_dog = 2 →
  four_paws_per_four_leg_dog = 4 →
  (two_thirds_back_legs * two_paws_per_back_leg_dog + remaining_dogs_four_legs * four_paws_per_four_leg_dog) = 64 := 
by 
  sorry

end dog_paws_ground_l69_69604


namespace fraction_of_full_fare_half_ticket_l69_69802

theorem fraction_of_full_fare_half_ticket (F R : ℝ) 
  (h1 : F + R = 216) 
  (h2 : F + (1/2)*F + 2*R = 327) : 
  (1/2) = 1/2 :=
by
  sorry

end fraction_of_full_fare_half_ticket_l69_69802


namespace lincoln_high_school_students_l69_69297

theorem lincoln_high_school_students (total students_in_either_or_both_clubs students_in_photography students_in_science : ℕ)
  (h1 : total = 300)
  (h2 : students_in_photography = 120)
  (h3 : students_in_science = 140)
  (h4 : students_in_either_or_both_clubs = 220):
  ∃ x, x = 40 ∧ (students_in_photography + students_in_science - students_in_either_or_both_clubs = x) := 
by
  use 40
  sorry

end lincoln_high_school_students_l69_69297


namespace entire_meal_cost_correct_l69_69175

-- Define given conditions
def appetizer_cost : ℝ := 9.00
def entree_cost : ℝ := 20.00
def num_entrees : ℕ := 2
def dessert_cost : ℝ := 11.00
def tip_percentage : ℝ := 0.30

-- Calculate intermediate values
def total_cost_before_tip : ℝ := appetizer_cost + (entree_cost * num_entrees) + dessert_cost
def tip : ℝ := tip_percentage * total_cost_before_tip
def entire_meal_cost : ℝ := total_cost_before_tip + tip

-- Statement to be proved
theorem entire_meal_cost_correct : entire_meal_cost = 78.00 := by
  -- Proof will go here
  sorry

end entire_meal_cost_correct_l69_69175


namespace minimum_elements_union_l69_69446

open Set

def A : Finset ℕ := sorry
def B : Finset ℕ := sorry

variable (size_A : A.card = 25)
variable (size_B : B.card = 18)
variable (at_least_10_not_in_A : (B \ A).card ≥ 10)

theorem minimum_elements_union : (A ∪ B).card = 35 :=
by
  sorry

end minimum_elements_union_l69_69446


namespace cos_A_minus_B_minus_3pi_div_2_l69_69168

theorem cos_A_minus_B_minus_3pi_div_2 (A B : ℝ)
  (h1 : Real.tan B = 2 * Real.tan A)
  (h2 : Real.cos A * Real.sin B = 4 / 5) :
  Real.cos (A - B - 3 * Real.pi / 2) = 2 / 5 := 
sorry

end cos_A_minus_B_minus_3pi_div_2_l69_69168


namespace fraction_meaningful_l69_69285

theorem fraction_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ (f : ℝ → ℝ), f x = (x + 2) / (x - 1) :=
by
  sorry

end fraction_meaningful_l69_69285


namespace repeating_decimal_to_fraction_l69_69619

noncomputable def repeating_decimal := 0.6 + 3 / 100

theorem repeating_decimal_to_fraction :
  repeating_decimal = 19 / 30 :=
  sorry

end repeating_decimal_to_fraction_l69_69619


namespace arithmetic_sequence_sum_l69_69562

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (n a_1 d : α) : α :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem arithmetic_sequence_sum (a_1 d : α) :
  S 5 a_1 d = 5 → S 9 a_1 d = 27 → S 7 a_1 d = 14 :=
by
  sorry

end arithmetic_sequence_sum_l69_69562


namespace num_children_proof_l69_69407

-- Definitions and Main Problem
def legs_of_javier : ℕ := 2
def legs_of_wife : ℕ := 2
def legs_per_child : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_of_cat : ℕ := 4
def num_dogs : ℕ := 2
def num_cats : ℕ := 1
def total_legs : ℕ := 22

-- Proof problem: Prove that the number of children (num_children) is equal to 3
theorem num_children_proof : ∃ num_children : ℕ, legs_of_javier + legs_of_wife + (num_children * legs_per_child) + (num_dogs * legs_per_dog) + (num_cats * legs_of_cat) = total_legs ∧ num_children = 3 :=
by
  -- Proof goes here
  sorry

end num_children_proof_l69_69407


namespace Madeline_hours_left_over_l69_69498

theorem Madeline_hours_left_over :
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  total_hours_per_week - total_busy_hours = 46 :=
by
  let class_hours := 18
  let homework_hours_per_day := 4
  let homework_hours_per_week := homework_hours_per_day * 7
  let sleeping_hours_per_day := 8
  let sleeping_hours_per_week := sleeping_hours_per_day * 7
  let work_hours := 20
  let total_busy_hours := class_hours + homework_hours_per_week + sleeping_hours_per_week + work_hours
  let total_hours_per_week := 24 * 7
  have : total_hours_per_week - total_busy_hours = 168 - 122 := by rfl
  have : 168 - 122 = 46 := by rfl
  exact this

end Madeline_hours_left_over_l69_69498


namespace bing_location_subject_l69_69641

-- Defining entities
inductive City
| Beijing
| Shanghai
| Chongqing

inductive Subject
| Mathematics
| Chinese
| ForeignLanguage

inductive Teacher
| Jia
| Yi
| Bing

-- Defining the conditions
variables (works_in : Teacher → City) (teaches : Teacher → Subject)

axiom cond1_jia_not_beijing : works_in Teacher.Jia ≠ City.Beijing
axiom cond1_yi_not_shanghai : works_in Teacher.Yi ≠ City.Shanghai
axiom cond2_beijing_not_foreign : ∀ t, works_in t = City.Beijing → teaches t ≠ Subject.ForeignLanguage
axiom cond3_shanghai_math : ∀ t, works_in t = City.Shanghai → teaches t = Subject.Mathematics
axiom cond4_yi_not_chinese : teaches Teacher.Yi ≠ Subject.Chinese

-- The question
theorem bing_location_subject : 
  works_in Teacher.Bing = City.Beijing ∧ teaches Teacher.Bing = Subject.Chinese :=
by
  sorry

end bing_location_subject_l69_69641


namespace largest_n_satisfies_conditions_l69_69995

theorem largest_n_satisfies_conditions :
  ∃ (n m a : ℤ), n = 313 ∧ n^2 = (m + 1)^3 - m^3 ∧ ∃ (k : ℤ), 2 * n + 103 = k^2 :=
by
  sorry

end largest_n_satisfies_conditions_l69_69995


namespace train_length_l69_69568

theorem train_length 
  (t1 t2 : ℝ)
  (d2 : ℝ)
  (L : ℝ)
  (V : ℝ)
  (h1 : t1 = 18)
  (h2 : t2 = 27)
  (h3 : d2 = 150.00000000000006)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) :
  L = 300.0000000000001 :=
by
  sorry

end train_length_l69_69568


namespace maximum_value_of_expression_l69_69450

-- Define the given condition
def condition (a b c : ℝ) : Prop := a + 3 * b + c = 5

-- Define the objective function
def objective (a b c : ℝ) : ℝ := a * b + a * c + b * c

-- Main theorem statement
theorem maximum_value_of_expression (a b c : ℝ) (h : condition a b c) : 
  ∃ (a b c : ℝ), condition a b c ∧ objective a b c = 25 / 3 :=
sorry

end maximum_value_of_expression_l69_69450


namespace find_m_plus_n_l69_69694

-- Definitions
structure Triangle (A B C P M N : Type) :=
  (midpoint_AD_P : P)
  (intersection_M_AB : M)
  (intersection_N_AC : N)
  (vec_AB : ℝ)
  (vec_AM : ℝ)
  (vec_AC : ℝ)
  (vec_AN : ℝ)
  (m : ℝ)
  (n : ℝ)
  (AB_eq_AM_mul_m : vec_AB = m * vec_AM)
  (AC_eq_AN_mul_n : vec_AC = n * vec_AN)

-- The theorem to prove
theorem find_m_plus_n (A B C P M N : Type)
  (t : Triangle A B C P M N) :
  t.m + t.n = 4 :=
sorry

end find_m_plus_n_l69_69694


namespace symmetric_point_reflection_y_axis_l69_69713

theorem symmetric_point_reflection_y_axis (x y : ℝ) (h : (x, y) = (-2, 3)) :
  (-x, y) = (2, 3) :=
sorry

end symmetric_point_reflection_y_axis_l69_69713


namespace derivative_of_f_at_pi_over_2_l69_69945

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = -5 :=
sorry

end derivative_of_f_at_pi_over_2_l69_69945


namespace find_min_value_l69_69256

noncomputable def minValue (a b c : ℝ) : ℝ :=
  (3 / a) - (4 / b) + (5 / c)

theorem find_min_value (a b c : ℝ) (h1 : c > 0) (h2 : 4 * a^2 - 2 * a * b + 4 * b^2 = c) (h3 : ∀ x y : ℝ, |2 * a + b| ≥ |2 * x + y|) :
  minValue a b c = -2 :=
sorry

end find_min_value_l69_69256


namespace problem_statement_l69_69101

noncomputable def f (x : ℚ) : ℚ := (x^2 - x - 6) / (x^3 - 2 * x^2 - x + 2)

def a : ℕ := 1  -- number of holes
def b : ℕ := 2  -- number of vertical asymptotes
def c : ℕ := 1  -- number of horizontal asymptotes
def d : ℕ := 0  -- number of oblique asymptotes

theorem problem_statement : a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end problem_statement_l69_69101


namespace parabola_line_intersect_at_one_point_l69_69799

theorem parabola_line_intersect_at_one_point :
  ∃ a : ℝ, (∀ x : ℝ, (ax^2 + 5 * x + 2 = -2 * x + 1)) ↔ a = 49 / 4 :=
by sorry

end parabola_line_intersect_at_one_point_l69_69799


namespace expression_evaluation_l69_69988

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end expression_evaluation_l69_69988


namespace inequality_square_l69_69913

theorem inequality_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 :=
sorry

end inequality_square_l69_69913


namespace square_area_l69_69643

theorem square_area (s : ℕ) (h : s = 13) : s * s = 169 := by
  sorry

end square_area_l69_69643


namespace andy_remaining_demerits_l69_69825

-- Definitions based on conditions
def max_demerits : ℕ := 50
def demerits_per_late_instance : ℕ := 2
def late_instances : ℕ := 6
def joke_demerits : ℕ := 15

-- Calculation of total demerits for the month
def total_demerits : ℕ := (demerits_per_late_instance * late_instances) + joke_demerits

-- Proof statement: Andy can receive 23 more demerits without being fired
theorem andy_remaining_demerits : max_demerits - total_demerits = 23 :=
by
  -- Placeholder for proof
  sorry

end andy_remaining_demerits_l69_69825


namespace total_pushups_l69_69294

theorem total_pushups (zachary_pushups : ℕ) (david_more_pushups : ℕ) 
  (h1 : zachary_pushups = 44) (h2 : david_more_pushups = 58) : 
  zachary_pushups + (zachary_pushups + david_more_pushups) = 146 :=
by
  sorry

end total_pushups_l69_69294


namespace determine_m_l69_69309

def setA_is_empty (m: ℝ) : Prop :=
  { x : ℝ | m * x = 1 } = ∅

theorem determine_m (m: ℝ) (h: setA_is_empty m) : m = 0 :=
by sorry

end determine_m_l69_69309


namespace average_candies_l69_69826

theorem average_candies {a b c d e f : ℕ} (h₁ : a = 16) (h₂ : b = 22) (h₃ : c = 30) (h₄ : d = 26) (h₅ : e = 18) (h₆ : f = 20) :
  (a + b + c + d + e + f) / 6 = 22 := by
  sorry

end average_candies_l69_69826


namespace find_quotient_l69_69853

-- Definitions for the variables and conditions
variables (D d q r : ℕ)

-- Conditions
axiom eq1 : D = q * d + r
axiom eq2 : D + 65 = q * (d + 5) + r

-- Theorem statement
theorem find_quotient : q = 13 :=
by
  sorry

end find_quotient_l69_69853


namespace find_cost_price_l69_69031

-- Conditions
def initial_cost_price (C : ℝ) : Prop :=
  let SP := 1.07 * C
  let NCP := 0.92 * C
  let NSP := SP - 3
  NSP = 1.0304 * C

-- The problem is to prove the initial cost price C given the conditions
theorem find_cost_price (C : ℝ) (h : initial_cost_price C) : C = 75.7575 := 
  sorry

end find_cost_price_l69_69031


namespace max_diagonals_in_grid_l69_69781

-- Define the dimensions of the grid
def grid_width := 8
def grid_height := 5

-- Define the number of 1x2 rectangles
def number_of_1x2_rectangles := grid_width / 2 * grid_height

-- State the theorem
theorem max_diagonals_in_grid : number_of_1x2_rectangles = 20 := 
by 
  -- Simplify the expression
  sorry

end max_diagonals_in_grid_l69_69781


namespace gwen_spent_money_l69_69939

theorem gwen_spent_money (initial : ℕ) (remaining : ℕ) (spent : ℕ) 
  (h_initial : initial = 7) 
  (h_remaining : remaining = 5) 
  (h_spent : spent = initial - remaining) : 
  spent = 2 := 
sorry

end gwen_spent_money_l69_69939


namespace max_value_condition_min_value_condition_l69_69094

theorem max_value_condition (x : ℝ) (h : x < 0) : (x^2 + x + 1) / x ≤ -1 :=
sorry

theorem min_value_condition (x : ℝ) (h : x > -1) : ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
sorry

end max_value_condition_min_value_condition_l69_69094


namespace find_upper_book_pages_l69_69228

noncomputable def pages_in_upper_book (total_digits : ℕ) (page_diff : ℕ) : ℕ :=
  -- Here we would include the logic to determine the number of pages, but we are only focusing on the statement.
  207

theorem find_upper_book_pages :
  ∀ (total_digits page_diff : ℕ), total_digits = 999 → page_diff = 9 → pages_in_upper_book total_digits page_diff = 207 :=
by
  intros total_digits page_diff h1 h2
  sorry

end find_upper_book_pages_l69_69228


namespace initial_minutes_planA_equivalence_l69_69110

-- Conditions translated into Lean:
variable (x : ℝ)

-- Definitions for costs
def planA_cost_12 : ℝ := 0.60 + 0.06 * (12 - x)
def planB_cost_12 : ℝ := 0.08 * 12

-- Theorem we want to prove
theorem initial_minutes_planA_equivalence :
  (planA_cost_12 x = planB_cost_12) → x = 6 :=
by
  intro h
  -- complete proof is skipped with sorry
  sorry

end initial_minutes_planA_equivalence_l69_69110


namespace sqrt_of_square_of_neg_five_eq_five_l69_69516

theorem sqrt_of_square_of_neg_five_eq_five : Real.sqrt ((-5 : ℤ) ^ 2) = 5 := by
  sorry

end sqrt_of_square_of_neg_five_eq_five_l69_69516


namespace contrapositive_of_proposition_is_false_l69_69090

variables {a b : ℤ}

/-- Proposition: If a and b are both even, then a + b is even -/
def proposition (a b : ℤ) : Prop :=
  (∀ n m : ℤ, a = 2 * n ∧ b = 2 * m → ∃ k : ℤ, a + b = 2 * k)

/-- Contrapositive: If a and b are not both even, then a + b is not even -/
def contrapositive (a b : ℤ) : Prop :=
  ¬(∀ n m : ℤ, a = 2 * n ∧ b = 2 * m) → ¬(∃ k : ℤ, a + b = 2 * k)

/-- The contrapositive of the proposition "If a and b are both even, then a + b is even" -/
theorem contrapositive_of_proposition_is_false :
  (contrapositive a b) = false :=
sorry

end contrapositive_of_proposition_is_false_l69_69090


namespace find_t_l69_69894

variable (a b c : ℝ × ℝ)
variable (t : ℝ)

-- Definitions based on given conditions
def vec_a : ℝ × ℝ := (3, 1)
def vec_b : ℝ × ℝ := (1, 3)
def vec_c (t : ℝ) : ℝ × ℝ := (t, 2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Condition that (vec_a - vec_c) is perpendicular to vec_b
def perpendicular_condition (t : ℝ) : Prop :=
  dot_product (vec_a - vec_c t) vec_b = 0

-- Proof statement
theorem find_t : ∃ t : ℝ, perpendicular_condition t ∧ t = 0 := 
by
  sorry

end find_t_l69_69894


namespace elementary_sampling_count_l69_69380

theorem elementary_sampling_count :
  ∃ (a : ℕ), (a + (a + 600) + (a + 1200) = 3600) ∧
             (a = 600) ∧
             (a + 1200 = 1800) ∧
             (1800 * 1 / 100 = 18) :=
by {
  sorry
}

end elementary_sampling_count_l69_69380


namespace calculate_students_l69_69139

noncomputable def handshakes (m n : ℕ) : ℕ :=
  1/2 * (4 * 3 + 5 * (2 * (m - 2) + 2 * (n - 2)) + 8 * (m - 2) * (n - 2))

theorem calculate_students (m n : ℕ) (h_m : 3 ≤ m) (h_n : 3 ≤ n) (h_handshakes : handshakes m n = 1020) : m * n = 140 :=
by
  sorry

end calculate_students_l69_69139


namespace smallest_five_digit_in_pascals_triangle_l69_69102

/-- In Pascal's triangle, the smallest five-digit number is 10000. -/
theorem smallest_five_digit_in_pascals_triangle : 
  ∃ (n k : ℕ), (10000 = Nat.choose n k) ∧ (∀ m l : ℕ, Nat.choose m l < 10000) → (n > m) := 
sorry

end smallest_five_digit_in_pascals_triangle_l69_69102


namespace flagpole_height_l69_69002

theorem flagpole_height (h : ℕ)
  (shadow_flagpole : ℕ := 72)
  (height_pole : ℕ := 18)
  (shadow_pole : ℕ := 27)
  (ratio_shadow : shadow_flagpole / shadow_pole = 8 / 3) :
  h = 48 :=
by
  sorry

end flagpole_height_l69_69002


namespace describe_random_event_l69_69540

def idiom_A : Prop := "海枯石烂" = "extremely improbable or far into the future, not random"
def idiom_B : Prop := "守株待兔" = "represents a random event"
def idiom_C : Prop := "画饼充饥" = "unreal hopes, not random"
def idiom_D : Prop := "瓜熟蒂落" = "natural or expected outcome, not random"

theorem describe_random_event : idiom_B := 
by
  -- Proof omitted; conclusion follows from the given definitions
  sorry

end describe_random_event_l69_69540


namespace total_jokes_proof_l69_69843

-- Definitions of the conditions
def jokes_jessy_last_saturday : Nat := 11
def jokes_alan_last_saturday : Nat := 7
def jokes_jessy_next_saturday : Nat := 2 * jokes_jessy_last_saturday
def jokes_alan_next_saturday : Nat := 2 * jokes_alan_last_saturday

-- Sum of jokes over two Saturdays
def total_jokes : Nat := (jokes_jessy_last_saturday + jokes_alan_last_saturday) + (jokes_jessy_next_saturday + jokes_alan_next_saturday)

-- The proof problem
theorem total_jokes_proof : total_jokes = 54 := 
by
  sorry

end total_jokes_proof_l69_69843


namespace probability_red_ball_distribution_of_X_expected_value_of_X_l69_69190

theorem probability_red_ball :
  let pB₁ := 2 / 3
  let pB₂ := 1 / 3
  let pA_B₁ := 1 / 2
  let pA_B₂ := 3 / 4
  (pB₁ * pA_B₁ + pB₂ * pA_B₂) = 7 / 12 := by
  sorry

theorem distribution_of_X :
  let p_minus2 := 1 / 12
  let p_0 := 1 / 12
  let p_1 := 11 / 24
  let p_3 := 7 / 48
  let p_4 := 5 / 24
  let p_6 := 1 / 48
  (p_minus2 = 1 / 12) ∧ (p_0 = 1 / 12) ∧ (p_1 = 11 / 24) ∧ (p_3 = 7 / 48) ∧ (p_4 = 5 / 24) ∧ (p_6 = 1 / 48) := by
  sorry

theorem expected_value_of_X :
  let E_X := (-2 * (1 / 12) + 0 * (1 / 12) + 1 * (11 / 24) + 3 * (7 / 48) + 4 * (5 / 24) + 6 * (1 / 48))
  E_X = 27 / 16 := by
  sorry

end probability_red_ball_distribution_of_X_expected_value_of_X_l69_69190


namespace dave_won_tickets_l69_69138

theorem dave_won_tickets (initial_tickets spent_tickets final_tickets won_tickets : ℕ) 
  (h1 : initial_tickets = 25) 
  (h2 : spent_tickets = 22) 
  (h3 : final_tickets = 18) 
  (h4 : won_tickets = final_tickets - (initial_tickets - spent_tickets)) :
  won_tickets = 15 := 
by 
  sorry

end dave_won_tickets_l69_69138


namespace pears_left_l69_69746

theorem pears_left (keith_initial : ℕ) (keith_given : ℕ) (mike_initial : ℕ) 
  (hk : keith_initial = 47) (hg : keith_given = 46) (hm : mike_initial = 12) :
  (keith_initial - keith_given) + mike_initial = 13 := by
  sorry

end pears_left_l69_69746


namespace t_50_mod_7_l69_69845

theorem t_50_mod_7 (T : ℕ → ℕ) (h₁ : T 1 = 9) (h₂ : ∀ n > 1, T n = 9 ^ (T (n - 1))) :
  T 50 % 7 = 4 :=
sorry

end t_50_mod_7_l69_69845


namespace pizza_consumption_order_l69_69699

theorem pizza_consumption_order :
  let total_slices := 168
  let alex_slices := (1/6) * total_slices
  let beth_slices := (2/7) * total_slices
  let cyril_slices := (1/3) * total_slices
  let eve_slices_initial := (1/8) * total_slices
  let dan_slices_initial := total_slices - (alex_slices + beth_slices + cyril_slices + eve_slices_initial)
  let eve_slices := eve_slices_initial + 2
  let dan_slices := dan_slices_initial - 2
  (cyril_slices > beth_slices ∧ beth_slices > eve_slices ∧ eve_slices > alex_slices ∧ alex_slices > dan_slices) :=
  sorry

end pizza_consumption_order_l69_69699


namespace binom_10_3_eq_120_l69_69474

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l69_69474


namespace smallest_n_contains_digit9_and_terminating_decimal_l69_69270

-- Define the condition that a number contains the digit 9
def contains_digit_9 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 9

-- Define the condition that a number is of the form 2^a * 5^b
def is_form_of_2a_5b (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2 ^ a * 5 ^ b

-- Define the main theorem
theorem smallest_n_contains_digit9_and_terminating_decimal : 
  ∃ (n : ℕ), contains_digit_9 n ∧ is_form_of_2a_5b n ∧ (∀ m, (contains_digit_9 m ∧ is_form_of_2a_5b m) → n ≤ m) ∧ n = 12500 :=
  sorry

end smallest_n_contains_digit9_and_terminating_decimal_l69_69270


namespace find_principal_amount_l69_69061

-- Definitions based on conditions
def A : ℝ := 3969
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- The statement to be proved
theorem find_principal_amount : ∃ P : ℝ, A = P * (1 + r/n)^(n * t) ∧ P = 3600 :=
by
  use 3600
  sorry

end find_principal_amount_l69_69061


namespace digit_sum_square_l69_69753

theorem digit_sum_square (n : ℕ) (hn : 0 < n) :
  let A := (4 * (10 ^ (2 * n) - 1)) / 9
  let B := (8 * (10 ^ n - 1)) / 9
  ∃ k : ℕ, A + 2 * B + 4 = k ^ 2 := 
by
  sorry

end digit_sum_square_l69_69753


namespace best_discount_option_l69_69268

-- Define the original price
def original_price : ℝ := 100

-- Define the discount functions for each option
def option_A : ℝ := original_price * (1 - 0.20)
def option_B : ℝ := (original_price * (1 - 0.10)) * (1 - 0.10)
def option_C : ℝ := (original_price * (1 - 0.15)) * (1 - 0.05)
def option_D : ℝ := (original_price * (1 - 0.05)) * (1 - 0.15)

-- Define the theorem stating that option A gives the best price
theorem best_discount_option : option_A ≤ option_B ∧ option_A ≤ option_C ∧ option_A ≤ option_D :=
by {
  sorry
}

end best_discount_option_l69_69268


namespace chickens_egg_production_l69_69331

/--
Roberto buys 4 chickens for $20 each. The chickens cost $1 in total per week to feed.
Roberto used to buy 1 dozen eggs (12 eggs) a week, spending $2 per dozen.
After 81 weeks, the total cost of raising chickens will be cheaper than buying the eggs.
Prove that each chicken produces 3 eggs per week.
-/
theorem chickens_egg_production:
  let chicken_cost := 20
  let num_chickens := 4
  let weekly_feed_cost := 1
  let weekly_eggs_cost := 2
  let dozen_eggs := 12
  let weeks := 81

  -- Cost calculations
  let total_chicken_cost := num_chickens * chicken_cost
  let total_feed_cost := weekly_feed_cost * weeks
  let total_raising_cost := total_chicken_cost + total_feed_cost
  let total_buying_cost := weekly_eggs_cost * weeks

  -- Ensure cost condition
  (total_raising_cost <= total_buying_cost) →
  
  -- Egg production calculation
  (dozen_eggs / num_chickens) = 3 :=
by
  intros
  sorry

end chickens_egg_production_l69_69331


namespace unique_real_solution_l69_69833

theorem unique_real_solution :
  ∃! x : ℝ, -((x + 2) ^ 2) ≥ 0 :=
sorry

end unique_real_solution_l69_69833


namespace solution_set_is_circle_with_exclusion_l69_69559

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l69_69559


namespace suzy_final_books_l69_69648

def initial_books : ℕ := 98
def wednesday_checkouts : ℕ := 43
def thursday_returns : ℕ := 23
def thursday_checkouts : ℕ := 5
def friday_returns : ℕ := 7

theorem suzy_final_books :
  initial_books - wednesday_checkouts
  + thursday_returns - thursday_checkouts
  + friday_returns = 80 := by
sorry

end suzy_final_books_l69_69648


namespace find_interest_rate_l69_69534

-- Given conditions
def P : ℝ := 4099.999999999999
def t : ℕ := 2
def CI_minus_SI : ℝ := 41

-- Formulas for Simple Interest and Compound Interest
def SI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * r * (t : ℝ)
def CI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * ((1 + r) ^ t) - P

-- Main theorem to prove: the interest rate r is 0.1 (i.e., 10%)
theorem find_interest_rate (r : ℝ) : 
  (CI P r t - SI P r t = CI_minus_SI) → r = 0.1 :=
by
  sorry

end find_interest_rate_l69_69534


namespace sufficient_not_necessary_l69_69831

namespace ProofExample

variable {x : ℝ}

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | x < 2}

-- Theorem: "1 < x < 2" is a sufficient but not necessary condition for "x < 2" to hold.
theorem sufficient_not_necessary : 
  (∀ x, 1 < x ∧ x < 2 → x < 2) ∧ ¬(∀ x, x < 2 → 1 < x ∧ x < 2) := 
by
  sorry

end ProofExample

end sufficient_not_necessary_l69_69831


namespace prime_square_minus_one_divisible_by_24_l69_69451

theorem prime_square_minus_one_divisible_by_24 (n : ℕ) (h_prime : Prime n) (h_n_neq_2 : n ≠ 2) (h_n_neq_3 : n ≠ 3) : 24 ∣ (n^2 - 1) :=
sorry

end prime_square_minus_one_divisible_by_24_l69_69451


namespace function_has_two_zeros_l69_69041

/-- 
Given the function y = x + 1/(2x) + t has two zeros under the condition t > 0,
prove that the range of the real number t is (-∞, -√2).
-/
theorem function_has_two_zeros (t : ℝ) (ht : t > 0) : t < -Real.sqrt 2 :=
sorry

end function_has_two_zeros_l69_69041


namespace claudia_total_earnings_l69_69740

-- Definition of the problem conditions
def class_fee : ℕ := 10
def kids_saturday : ℕ := 20
def kids_sunday : ℕ := kids_saturday / 2

-- Theorem stating that Claudia makes $300.00 for the weekend
theorem claudia_total_earnings : (kids_saturday * class_fee) + (kids_sunday * class_fee) = 300 := 
by
  sorry

end claudia_total_earnings_l69_69740


namespace fraction_identity_l69_69917

theorem fraction_identity (f : ℚ) (h : 32 * f^2 = 2^3) : f = 1 / 2 :=
sorry

end fraction_identity_l69_69917


namespace find_abc_l69_69205

theorem find_abc
  {a b c : ℤ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30)
  (h2 : 1/a + 1/b + 1/c + 672/(a*b*c) = 1) :
  a * b * c = 2808 :=
sorry

end find_abc_l69_69205


namespace probability_snow_at_least_once_l69_69115

-- Defining the probability of no snow on the first five days
def no_snow_first_five_days : ℚ := (4 / 5) ^ 5

-- Defining the probability of no snow on the next five days
def no_snow_next_five_days : ℚ := (2 / 3) ^ 5

-- Total probability of no snow during the first ten days
def no_snow_first_ten_days : ℚ := no_snow_first_five_days * no_snow_next_five_days

-- Probability of snow at least once during the first ten days
def snow_at_least_once_first_ten_days : ℚ := 1 - no_snow_first_ten_days

-- Desired proof statement
theorem probability_snow_at_least_once :
  snow_at_least_once_first_ten_days = 726607 / 759375 := by
  sorry

end probability_snow_at_least_once_l69_69115


namespace range_of_f_l69_69134

noncomputable def f (x : ℝ) : ℝ := - (2 / (x - 1))

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, (0 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2) ∧ f x = y} = 
  {y : ℝ | y ≤ -2 ∨ 2 ≤ y} :=
by
  sorry

end range_of_f_l69_69134


namespace smallest_y_value_l69_69823

theorem smallest_y_value : 
  ∀ y : ℝ, (3 * y^2 + 15 * y - 90 = y * (y + 20)) → y ≥ -6 :=
by
  sorry

end smallest_y_value_l69_69823


namespace rectangle_maximized_area_side_length_l69_69756

theorem rectangle_maximized_area_side_length
  (x y : ℝ)
  (h_perimeter : 2 * x + 2 * y = 40)
  (h_max_area : x * y = 100) :
  x = 10 :=
by
  sorry

end rectangle_maximized_area_side_length_l69_69756


namespace correct_first_coupon_day_l69_69012

def is_redemption_valid (start_day : ℕ) (interval : ℕ) (num_coupons : ℕ) (closed_day : ℕ) : Prop :=
  ∀ n : ℕ, n < num_coupons → (start_day + n * interval) % 7 ≠ closed_day

def wednesday : ℕ := 3  -- Assuming Sunday = 0, Monday = 1, ..., Saturday = 6

theorem correct_first_coupon_day : 
  is_redemption_valid wednesday 10 6 0 :=
by {
  -- Proof goes here
  sorry
}

end correct_first_coupon_day_l69_69012


namespace value_of_square_reciprocal_l69_69242

theorem value_of_square_reciprocal (x : ℝ) (h : 18 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = Real.sqrt 20 := by
  sorry

end value_of_square_reciprocal_l69_69242


namespace min_distance_MN_l69_69868

open Real

noncomputable def f (x : ℝ) := exp x - (1 / 2) * x^2
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_MN (x1 x2 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 > 0) (h3 : f x1 = g x2) :
  abs (x2 - x1) = 2 :=
by
  sorry

end min_distance_MN_l69_69868


namespace min_cos_y_plus_sin_x_l69_69838

theorem min_cos_y_plus_sin_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.cos x = Real.sin (3 * x))
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) - Real.cos (2 * x)) :
  ∃ (v : ℝ), v = -1 - Real.sqrt (2 + Real.sqrt 2) / 2 :=
sorry

end min_cos_y_plus_sin_x_l69_69838


namespace Willey_Farm_Available_Capital_l69_69755

theorem Willey_Farm_Available_Capital 
  (total_acres : ℕ)
  (cost_per_acre_corn : ℕ)
  (cost_per_acre_wheat : ℕ)
  (acres_wheat : ℕ)
  (available_capital : ℕ) :
  total_acres = 4500 →
  cost_per_acre_corn = 42 →
  cost_per_acre_wheat = 35 →
  acres_wheat = 3400 →
  available_capital = (acres_wheat * cost_per_acre_wheat) + 
                      ((total_acres - acres_wheat) * cost_per_acre_corn) →
  available_capital = 165200 := sorry

end Willey_Farm_Available_Capital_l69_69755


namespace expression_evaluation_l69_69672

noncomputable def x : ℝ := (Real.sqrt 1.21) ^ 3
noncomputable def y : ℝ := (Real.sqrt 0.81) ^ 2
noncomputable def a : ℝ := 4 * Real.sqrt 0.81
noncomputable def b : ℝ := 2 * Real.sqrt 0.49
noncomputable def c : ℝ := 3 * Real.sqrt 1.21
noncomputable def d : ℝ := 2 * Real.sqrt 0.49
noncomputable def e : ℝ := (Real.sqrt 0.81) ^ 4

theorem expression_evaluation : ((x / Real.sqrt y) - (Real.sqrt a / b^2) + ((Real.sqrt c / Real.sqrt d) / (3 * e))) = 1.291343 := by 
  sorry

end expression_evaluation_l69_69672


namespace proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l69_69095

theorem proportion_false_if_x_is_0_75 (x : ℚ) (h1 : x = 0.75) : ¬ (x / 2 = 2 / 6) :=
by sorry

theorem correct_value_of_x_in_proportion (x : ℚ) (h1 : x / 2 = 2 / 6) : x = 2 / 3 :=
by sorry

end proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l69_69095


namespace length_of_real_axis_l69_69907

noncomputable def hyperbola_1 : Prop :=
  ∃ (x y: ℝ), (x^2 / 16) - (y^2 / 4) = 1

noncomputable def hyperbola_2 (a b: ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∃ (x y: ℝ), (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def same_eccentricity (a b: ℝ) : Prop :=
  (1 + b^2 / a^2) = (1 + 1 / 4 / 16)

noncomputable def area_of_triangle (a b: ℝ) : Prop :=
  (a * b) = 32

theorem length_of_real_axis (a b: ℝ) (ha : 0 < a) (hb : 0 < b) :
  hyperbola_1 ∧ hyperbola_2 a b ha hb ∧ same_eccentricity a b ∧ area_of_triangle a b →
  2 * a = 16 :=
by
  sorry

end length_of_real_axis_l69_69907


namespace smallest_enclosing_sphere_radius_l69_69571

noncomputable def radius_of_enclosing_sphere (r : ℝ) : ℝ :=
  let s := 6 -- side length of the cube
  let d := s * Real.sqrt 3 -- space diagonal of the cube
  (d + 2 * r) / 2

theorem smallest_enclosing_sphere_radius :
  radius_of_enclosing_sphere 2 = 3 * Real.sqrt 3 + 2 :=
by
  -- skipping the proof with sorry
  sorry

end smallest_enclosing_sphere_radius_l69_69571


namespace geometric_sequence_S20_l69_69864

-- Define the conditions and target statement
theorem geometric_sequence_S20
  (a : ℕ → ℝ) -- defining the sequence as a function from natural numbers to real numbers
  (q : ℝ) -- common ratio
  (h_pos : ∀ n, a n > 0) -- all terms are positive
  (h_geo : ∀ n, a (n + 1) = q * a n) -- geometric sequence property
  (S : ℕ → ℝ) -- sum function
  (h_S : ∀ n, S n = (a 1 * (1 - q ^ n)) / (1 - q)) -- sum formula for a geometric progression
  (h_S5 : S 5 = 3) -- given S_5 = 3
  (h_S15 : S 15 = 21) -- given S_15 = 21
  : S 20 = 45 := sorry

end geometric_sequence_S20_l69_69864


namespace tom_four_times_cindy_years_ago_l69_69336

variables (t c x : ℕ)

-- Conditions
axiom cond1 : t + 5 = 2 * (c + 5)
axiom cond2 : t - 13 = 3 * (c - 13)

-- Question to prove
theorem tom_four_times_cindy_years_ago :
  t - x = 4 * (c - x) → x = 19 :=
by
  intros h
  -- simply skip the proof for now
  sorry

end tom_four_times_cindy_years_ago_l69_69336


namespace Joel_laps_count_l69_69818

-- Definitions of conditions
def Yvonne_laps := 10
def sister_laps := Yvonne_laps / 2
def Joel_laps := sister_laps * 3

-- Statement to be proved
theorem Joel_laps_count : Joel_laps = 15 := by
  -- currently, proof is not required, so we defer it with 'sorry'
  sorry

end Joel_laps_count_l69_69818


namespace original_rope_length_l69_69232

variable (S : ℕ) (L : ℕ)

-- Conditions
axiom shorter_piece_length : S = 20
axiom longer_piece_length : L = 2 * S

-- Prove that the original length of the rope is 60 meters
theorem original_rope_length : S + L = 60 :=
by
  -- proof steps will go here
  sorry

end original_rope_length_l69_69232


namespace negation_universal_to_particular_l69_69567

theorem negation_universal_to_particular :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end negation_universal_to_particular_l69_69567


namespace polygon_sides_count_l69_69585

def sides_square : ℕ := 4
def sides_triangle : ℕ := 3
def sides_hexagon : ℕ := 6
def sides_heptagon : ℕ := 7
def sides_octagon : ℕ := 8
def sides_nonagon : ℕ := 9

def total_sides_exposed : ℕ :=
  let adjacent_1side := sides_square + sides_nonagon - 2 * 1
  let adjacent_2sides :=
    sides_triangle + sides_hexagon +
    sides_heptagon + sides_octagon - 4 * 2
  adjacent_1side + adjacent_2sides

theorem polygon_sides_count : total_sides_exposed = 27 := by
  sorry

end polygon_sides_count_l69_69585


namespace bowling_ball_weight_l69_69252

variables (b c k : ℝ)

def condition1 : Prop := 9 * b = 6 * c
def condition2 : Prop := c + k = 42
def condition3 : Prop := 3 * k = 2 * c

theorem bowling_ball_weight
  (h1 : condition1 b c)
  (h2 : condition2 c k)
  (h3 : condition3 c k) :
  b = 16.8 :=
sorry

end bowling_ball_weight_l69_69252


namespace right_triangles_not_1000_l69_69728

-- Definitions based on the conditions
def numPoints := 100
def numDiametricallyOppositePairs := numPoints / 2
def rightTrianglesPerPair := numPoints - 2
def totalRightTriangles := numDiametricallyOppositePairs * rightTrianglesPerPair

-- Theorem stating the final evaluation of the problem
theorem right_triangles_not_1000 :
  totalRightTriangles ≠ 1000 :=
by
  -- calculation shows it's impossible
  sorry

end right_triangles_not_1000_l69_69728


namespace sad_girls_count_l69_69795

variables (total_children happy_children sad_children neither_children : ℕ)
variables (total_boys total_girls happy_boys sad_children total_sad_boys : ℕ)

theorem sad_girls_count :
  total_children = 60 ∧ 
  happy_children = 30 ∧ 
  sad_children = 10 ∧ 
  neither_children = 20 ∧ 
  total_boys = 17 ∧ 
  total_girls = 43 ∧ 
  happy_boys = 6 ∧ 
  neither_boys = 5 ∧ 
  sad_children = total_sad_boys + (sad_children - total_sad_boys) ∧ 
  total_sad_boys = total_boys - happy_boys - neither_boys → 
  (sad_children - total_sad_boys = 4) := 
by
  intros h
  sorry

end sad_girls_count_l69_69795


namespace smallest_w_l69_69774

theorem smallest_w (w : ℕ) (hw : w > 0) (h1 : ∃ k1, 936 * w = 2^5 * k1) (h2 : ∃ k2, 936 * w = 3^3 * k2) (h3 : ∃ k3, 936 * w = 10^2 * k3) : 
  w = 300 :=
by
  sorry

end smallest_w_l69_69774


namespace race_distance_l69_69918

theorem race_distance {a b c : ℝ} (h1 : b = 0.9 * a) (h2 : c = 0.95 * b) :
  let andrei_distance := 1000
  let boris_distance := andrei_distance - 100
  let valentin_distance := boris_distance - 50
  let valentin_actual_distance := (c / a) * andrei_distance
  andrei_distance - valentin_actual_distance = 145 :=
by
  sorry

end race_distance_l69_69918


namespace compute_x_y_sum_l69_69111

theorem compute_x_y_sum (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 2)^4 + (Real.log y / Real.log 3)^4 + 8 = 8 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^Real.sqrt 2 + y^Real.sqrt 2 = 13 :=
by
  sorry

end compute_x_y_sum_l69_69111


namespace position_2023_l69_69867

def initial_position := "ABCD"

def rotate_180 (pos : String) : String :=
  match pos with
  | "ABCD" => "CDAB"
  | "CDAB" => "ABCD"
  | "DCBA" => "BADC"
  | "BADC" => "DCBA"
  | _ => pos

def reflect_horizontal (pos : String) : String :=
  match pos with
  | "ABCD" => "ABCD"
  | "CDAB" => "DCBA"
  | "DCBA" => "CDAB"
  | "BADC" => "BADC"
  | _ => pos

def transformation (n : ℕ) : String :=
  let cnt := n % 4
  if cnt = 1 then rotate_180 initial_position
  else if cnt = 2 then rotate_180 (rotate_180 initial_position)
  else if cnt = 3 then rotate_180 (reflect_horizontal (rotate_180 initial_position))
  else reflect_horizontal initial_position

theorem position_2023 : transformation 2023 = "DCBA" := by
  sorry

end position_2023_l69_69867


namespace range_of_m_l69_69348

theorem range_of_m (m : ℝ) : (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → -5 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l69_69348


namespace difference_two_numbers_l69_69576

theorem difference_two_numbers (a b : ℕ) (h₁ : a + b = 20250) (h₂ : b % 15 = 0) (h₃ : a = b / 3) : b - a = 10130 :=
by 
  sorry

end difference_two_numbers_l69_69576


namespace no_solution_system_l69_69991

theorem no_solution_system (a : ℝ) : 
  (∀ x : ℝ, (x - 2 * a > 0) → (3 - 2 * x > x - 6) → false) ↔ a ≥ 3 / 2 :=
by
  sorry

end no_solution_system_l69_69991


namespace max_smaller_rectangles_l69_69720

theorem max_smaller_rectangles (a : ℕ) (d : ℕ) (n : ℕ) 
    (ha : a = 100) (hd : d = 2) (hn : n = 50) : 
    n + 1 * (n + 1) = 2601 :=
by
  rw [hn]
  norm_num
  sorry

end max_smaller_rectangles_l69_69720


namespace john_finishes_ahead_l69_69910

noncomputable def InitialDistanceBehind : ℝ := 12
noncomputable def JohnSpeed : ℝ := 4.2
noncomputable def SteveSpeed : ℝ := 3.7
noncomputable def PushTime : ℝ := 28

theorem john_finishes_ahead :
  (JohnSpeed * PushTime - InitialDistanceBehind) - (SteveSpeed * PushTime) = 2 := by
  sorry

end john_finishes_ahead_l69_69910


namespace isabella_babysits_afternoons_per_week_l69_69296

-- Defining the conditions of Isabella's babysitting job
def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 5
def days_per_week (weeks : ℕ) (total_earnings : ℕ) : ℕ := total_earnings / (weeks * (hourly_rate * hours_per_day))

-- Total earnings after 7 weeks
def total_earnings : ℕ := 1050
def weeks : ℕ := 7

-- State the theorem
theorem isabella_babysits_afternoons_per_week :
  days_per_week weeks total_earnings = 6 :=
by
  sorry

end isabella_babysits_afternoons_per_week_l69_69296


namespace expression_square_l69_69286

theorem expression_square (a b c d : ℝ) :
  (2*a + b + 2*c - d)^2 - (3*a + 2*b + 3*c - 2*d)^2 - (4*a + 3*b + 4*c - 3*d)^2 + (5*a + 4*b + 5*c - 4*d)^2 =
  (2*(a + b + c - d))^2 := 
sorry

end expression_square_l69_69286


namespace complex_identity_l69_69097

theorem complex_identity (α β : ℝ) (h : Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = Complex.mk (-1 / 3) (5 / 8)) :
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = Complex.mk (-1 / 3) (-5 / 8) :=
by
  sorry

end complex_identity_l69_69097


namespace part1_a_range_part2_x_range_l69_69383
open Real

-- Definitions based on given conditions
def quad_func (a b x : ℝ) : ℝ :=
  a * x^2 + b * x + 2

def y_at_x1 (a b : ℝ) : Prop :=
  quad_func a b 1 = 1

def pos_on_interval (a b l r : ℝ) (x : ℝ) : Prop :=
  l < x ∧ x < r → 0 < quad_func a b x

-- Part 1 proof statement in Lean 4
theorem part1_a_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ x : ℝ, pos_on_interval a b 2 5 x) :
  a > 3 - 2 * sqrt 2 :=
sorry

-- Part 2 proof statement in Lean 4
theorem part2_x_range (a b : ℝ) (h1 : y_at_x1 a b) (h2 : ∀ a' : ℝ, -2 ≤ a' ∧ a' ≤ -1 → 0 < quad_func a' b x) :
  (1 - sqrt 17) / 4 < x ∧ x < (1 + sqrt 17) / 4 :=
sorry

end part1_a_range_part2_x_range_l69_69383


namespace evaluate_expression_l69_69184

theorem evaluate_expression :
  let x := 1.93
  let y := 51.3
  let z := 0.47
  Float.round (x * (y + z)) = 100 := by
sorry

end evaluate_expression_l69_69184


namespace find_k_l69_69274

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x + 2 * y - 7 = 0
def l2 (x y : ℝ) (k : ℝ) : Prop := 2 * x + k * x + 3 = 0

-- Define the condition for parallel lines in our context
def parallel (k : ℝ) : Prop := - (1 / 2) = -(2 / k)

-- Prove that under the given conditions, k must be 4
theorem find_k (k : ℝ) : parallel k → k = 4 :=
by
  intro h
  sorry

end find_k_l69_69274


namespace value_of_n_l69_69305

theorem value_of_n (a : ℝ) (n : ℕ) (h : ∃ (k : ℕ), (n - 2 * k = 0) ∧ (k = 4)) : n = 8 :=
sorry

end value_of_n_l69_69305


namespace terry_tomato_types_l69_69527

theorem terry_tomato_types (T : ℕ) (h1 : 2 * T * 4 * 2 = 48) : T = 3 :=
by
  -- Proof goes here
  sorry

end terry_tomato_types_l69_69527


namespace least_number_to_produce_multiple_of_112_l69_69257

theorem least_number_to_produce_multiple_of_112 : ∃ k : ℕ, 72 * k = 112 * m → k = 14 :=
by
  sorry

end least_number_to_produce_multiple_of_112_l69_69257


namespace original_mixture_percentage_l69_69267

def mixture_percentage_acid (a w : ℕ) : ℚ :=
  a / (a + w)

theorem original_mixture_percentage (a w : ℕ) :
  (a / (a + w+2) = 1 / 4) ∧ ((a + 2) / (a + w + 4) = 2 / 5) → 
  mixture_percentage_acid a w = 1 / 3 :=
by
  sorry

end original_mixture_percentage_l69_69267


namespace solve_fraction_equation_l69_69195

def fraction_equation (x : ℝ) : Prop :=
  1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) + 2 / (x - 1) = 5

theorem solve_fraction_equation (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 1) :
  fraction_equation x → 
  x = (-11 + Real.sqrt 257) / 4 ∨ x = (-11 - Real.sqrt 257) / 4 :=
by
  sorry

end solve_fraction_equation_l69_69195


namespace riding_mower_speed_l69_69665

theorem riding_mower_speed :
  (∃ R : ℝ, 
     (8 * (3 / 4) = 6) ∧       -- Jerry mows 6 acres with the riding mower
     (8 * (1 / 4) = 2) ∧       -- Jerry mows 2 acres with the push mower
     (2 / 1 = 2) ∧             -- Push mower takes 2 hours to mow 2 acres
     (5 - 2 = 3) ∧             -- Time spent on the riding mower is 3 hours
     (6 / 3 = R) ∧             -- Riding mower cuts 6 acres in 3 hours
     R = 2) :=                 -- Therefore, R (speed of riding mower in acres per hour) is 2
sorry

end riding_mower_speed_l69_69665


namespace dacid_physics_marks_l69_69044

theorem dacid_physics_marks 
  (english : ℕ := 73)
  (math : ℕ := 69)
  (chem : ℕ := 64)
  (bio : ℕ := 82)
  (avg_marks : ℕ := 76)
  (num_subjects : ℕ := 5)
  : ∃ physics : ℕ, physics = 92 :=
by
  let total_marks := avg_marks * num_subjects
  let known_marks := english + math + chem + bio
  have physics := total_marks - known_marks
  use physics
  sorry

end dacid_physics_marks_l69_69044


namespace marbles_difference_l69_69129

theorem marbles_difference : 10 - 8 = 2 :=
by
  sorry

end marbles_difference_l69_69129


namespace div_by_11_l69_69523

theorem div_by_11 (x y : ℤ) (k : ℤ) (h : 14 * x + 13 * y = 11 * k) : 11 ∣ (19 * x + 9 * y) :=
by
  sorry

end div_by_11_l69_69523


namespace perfume_price_l69_69750

variable (P : ℝ)

theorem perfume_price (h_increase : 1.10 * P = P + 0.10 * P)
    (h_decrease : 0.935 * P = 1.10 * P - 0.15 * 1.10 * P)
    (h_final_price : P - 0.935 * P = 78) : P = 1200 := 
by
  sorry

end perfume_price_l69_69750


namespace triangle_area_is_correct_l69_69587

noncomputable def triangle_area : ℝ :=
  let A := (3, 3)
  let B := (4.5, 7.5)
  let C := (7.5, 4.5)
  1 / 2 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2) : ℝ)|

theorem triangle_area_is_correct : triangle_area = 9 := by
  sorry

end triangle_area_is_correct_l69_69587


namespace minimum_value_of_reciprocal_squares_l69_69996

theorem minimum_value_of_reciprocal_squares
  (a b : ℝ)
  (h : a ≠ 0 ∧ b ≠ 0)
  (h_eq : (a^2) + 4 * (b^2) = 9)
  : (1/(a^2) + 1/(b^2)) = 1 :=
sorry

end minimum_value_of_reciprocal_squares_l69_69996


namespace total_distance_A_C_B_l69_69636

noncomputable section

open Real

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A : point := (-3, 5)
def B : point := (5, -3)
def C : point := (0, 0)

theorem total_distance_A_C_B :
  distance A C + distance C B = 2 * sqrt 34 :=
by
  sorry

end total_distance_A_C_B_l69_69636


namespace arithmetic_square_root_of_4_l69_69191

theorem arithmetic_square_root_of_4 : ∃ x : ℕ, x * x = 4 ∧ x = 2 := 
sorry

end arithmetic_square_root_of_4_l69_69191


namespace determinant_eval_l69_69196

open Matrix

noncomputable def matrix_example (α γ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin α, -Real.cos α],
    ![-Real.sin α, 0, 3 * Real.sin γ],
    ![2 * Real.cos α, -Real.sin γ, 0]]

theorem determinant_eval (α γ : ℝ) :
  det (matrix_example α γ) = 10 * Real.sin α * Real.sin γ * Real.cos α :=
sorry

end determinant_eval_l69_69196


namespace interest_rate_l69_69895

theorem interest_rate (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) : 
  P = 8000.000000000171 → t = 2 → d = 20 →
  (P * (1 + r/100)^2 - P - (P * r * t / 100) = d) → r = 5 :=
by
  intros hP ht hd heq
  sorry

end interest_rate_l69_69895


namespace question_correctness_l69_69723

theorem question_correctness (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
by sorry

end question_correctness_l69_69723


namespace only_root_is_4_l69_69616

noncomputable def equation_one (x : ℝ) : ℝ := (2 * x^2) / (x - 1) - (2 * x + 7) / 3 + (4 - 6 * x) / (x - 1) + 1

noncomputable def equation_two (x : ℝ) : ℝ := x^2 - 5 * x + 4

theorem only_root_is_4 (x : ℝ) (h: equation_one x = 0) (h_transformation: equation_two x = 0) : x = 4 := sorry

end only_root_is_4_l69_69616


namespace range_of_a_l69_69371

-- Definitions for the conditions
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0

-- Main theorem
theorem range_of_a (a : ℝ) (h : a < 0) : (¬ (∃ x, prop_p a x)) → (¬ (∃ x, ¬ prop_q x)) :=
sorry

end range_of_a_l69_69371


namespace total_volume_is_correct_l69_69148

theorem total_volume_is_correct :
  let carl_side := 3
  let carl_count := 3
  let kate_side := 1.5
  let kate_count := 4
  let carl_volume := carl_count * carl_side ^ 3
  let kate_volume := kate_count * kate_side ^ 3
  carl_volume + kate_volume = 94.5 :=
by
  sorry

end total_volume_is_correct_l69_69148


namespace medians_sum_square_l69_69472

-- Define the sides of the triangle
variables {a b c : ℝ}

-- Define diameters
variables {D : ℝ}

-- Define medians of the triangle
variables {m_a m_b m_c : ℝ}

-- Defining the theorem statement
theorem medians_sum_square :
  m_a ^ 2 + m_b ^ 2 + m_c ^ 2 = (3 / 4) * (a ^ 2 + b ^ 2 + c ^ 2) + (3 / 4) * D ^ 2 :=
sorry

end medians_sum_square_l69_69472


namespace molecular_weight_of_9_moles_CCl4_l69_69291

-- Define the atomic weight of Carbon (C) and Chlorine (Cl)
def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45

-- Define the molecular formula for carbon tetrachloride (CCl4)
def molecular_formula_CCl4 : ℝ := atomic_weight_C + (4 * atomic_weight_Cl)

-- Define the molecular weight of one mole of carbon tetrachloride (CCl4)
def molecular_weight_one_mole_CCl4 : ℝ := molecular_formula_CCl4

-- Define the number of moles
def moles_CCl4 : ℝ := 9

-- Define the result to check
def molecular_weight_nine_moles_CCl4 : ℝ := molecular_weight_one_mole_CCl4 * moles_CCl4

-- State the theorem to prove the molecular weight of 9 moles of carbon tetrachloride is 1384.29 grams
theorem molecular_weight_of_9_moles_CCl4 :
  molecular_weight_nine_moles_CCl4 = 1384.29 := by
  sorry

end molecular_weight_of_9_moles_CCl4_l69_69291


namespace bowling_ball_weight_l69_69382

theorem bowling_ball_weight :
  (∀ b c : ℝ, 9 * b = 2 * c → c = 35 → b = 70 / 9) :=
by
  intros b c h1 h2
  sorry

end bowling_ball_weight_l69_69382


namespace remainder_when_divided_by_20_l69_69928

theorem remainder_when_divided_by_20 
  (n r : ℤ) 
  (k : ℤ)
  (h1 : n % 20 = r) 
  (h2 : 2 * n % 10 = 2)
  (h3 : 0 ≤ r ∧ r < 20)
  : r = 1 := 
sorry

end remainder_when_divided_by_20_l69_69928


namespace intersection_A_complement_B_l69_69359

def A := { x : ℝ | x ≥ -1 }
def B := { x : ℝ | x > 2 }
def complement_B := { x : ℝ | x ≤ 2 }

theorem intersection_A_complement_B :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 2 } :=
sorry

end intersection_A_complement_B_l69_69359


namespace intersection_A_B_l69_69819

-- Conditions
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = 3 * x - 2 }

-- Question and proof statement
theorem intersection_A_B :
  A ∩ B = {1, 4} := by
  sorry

end intersection_A_B_l69_69819


namespace hyperbola_asymptote_l69_69392

theorem hyperbola_asymptote (a : ℝ) (h_cond : 0 < a)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2 - y^2 / 9 = 1) → (y = (3 / 5) * x))
  : a = 5 :=
sorry

end hyperbola_asymptote_l69_69392


namespace collinear_vectors_l69_69269

theorem collinear_vectors (m : ℝ) (h_collinear : 1 * m - (-2) * (-3) = 0) : m = 6 :=
by
  sorry

end collinear_vectors_l69_69269


namespace alloy_mixing_l69_69426

theorem alloy_mixing (x : ℕ) :
  (2 / 5) * 60 + (1 / 5) * x = 44 → x = 100 :=
by
  intros h1
  sorry

end alloy_mixing_l69_69426


namespace least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l69_69901

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3) + 2

theorem least_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem maximum_value_of_f :
  ∃ x, f x = 3 :=
sorry

theorem monotonically_increasing_intervals_of_f :
  ∀ k : ℤ, ∃ a b : ℝ, a = -Real.pi / 12 + k * Real.pi ∧ b = 5 * Real.pi / 12 + k * Real.pi ∧ ∀ x, a < x ∧ x < b → ∀ x', a ≤ x' ∧ x' ≤ x → f x' < f x :=
sorry

end least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l69_69901


namespace part_a_not_divisible_by_29_part_b_divisible_by_11_l69_69947
open Nat

-- Part (a): Checking divisibility of 5641713 by 29
def is_divisible_by_29 (n : ℕ) : Prop :=
  n % 29 = 0

theorem part_a_not_divisible_by_29 : ¬is_divisible_by_29 5641713 :=
  by sorry

-- Part (b): Checking divisibility of 1379235 by 11
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem part_b_divisible_by_11 : is_divisible_by_11 1379235 :=
  by sorry

end part_a_not_divisible_by_29_part_b_divisible_by_11_l69_69947


namespace count_two_digit_perfect_squares_divisible_by_4_l69_69588

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l69_69588


namespace quadratic_not_divisible_by_49_l69_69075

theorem quadratic_not_divisible_by_49 (n : ℤ) : ¬ (n^2 + 3 * n + 4) % 49 = 0 := 
by
  sorry

end quadratic_not_divisible_by_49_l69_69075


namespace multiplication_by_9_l69_69140

theorem multiplication_by_9 (n : ℕ) (h1 : n < 10) : 9 * n = 10 * (n - 1) + (10 - n) := 
sorry

end multiplication_by_9_l69_69140


namespace slope_angle_correct_l69_69764

def parametric_line (α : ℝ) : Prop :=
  α = 50 * (Real.pi / 180)

theorem slope_angle_correct : ∀ (t : ℝ),
  parametric_line 50 →
  ∀ α : ℝ, α = 140 * (Real.pi / 180) :=
by
  intro t
  intro h
  intro α
  sorry

end slope_angle_correct_l69_69764


namespace range_func_l69_69594

noncomputable def func (x : ℝ) : ℝ := x + 4 / x

theorem range_func (x : ℝ) (hx : x ≠ 0) : func x ≤ -4 ∨ func x ≥ 4 := by
  sorry

end range_func_l69_69594


namespace determine_a_l69_69535

theorem determine_a (a x : ℝ) (h : x = 1) (h_eq : a * x + 2 * x = 3) : a = 1 :=
by
  subst h
  simp at h_eq
  linarith

end determine_a_l69_69535


namespace find_percentage_of_alcohol_in_second_solution_l69_69790

def alcohol_content_second_solution (V2: ℕ) (p1 p2 p_final: ℕ) (V1 V_final: ℕ) : ℕ :=
  ((V_final * p_final) - (V1 * p1)) * 100 / V2

def percentage_correct : Prop :=
  alcohol_content_second_solution 125 20 12 15 75 200 = 12

theorem find_percentage_of_alcohol_in_second_solution : percentage_correct :=
by
  sorry

end find_percentage_of_alcohol_in_second_solution_l69_69790


namespace find_s_base_10_l69_69320

-- Defining the conditions of the problem
def s_in_base_b_equals_42 (b : ℕ) : Prop :=
  let factor_1 := b + 3
  let factor_2 := b + 4
  let factor_3 := b + 5
  let produced_number := factor_1 * factor_2 * factor_3
  produced_number = 2 * b^3 + 3 * b^2 + 2 * b + 5

-- The proof problem as a Lean 4 statement
theorem find_s_base_10 :
  (∃ b : ℕ, s_in_base_b_equals_42 b) →
  13 + 14 + 15 = 42 :=
sorry

end find_s_base_10_l69_69320


namespace simplify_expression_l69_69091

theorem simplify_expression (m n : ℝ) (h : m ≠ 0) : 
  (m^(4/3) - 27 * m^(1/3) * n) / 
  (m^(2/3) + 3 * (m * n)^(1/3) + 9 * n^(2/3)) / 
  (1 - 3 * (n / m)^(1/3)) - 
  (m^2)^(1/3) = 0 := 
sorry

end simplify_expression_l69_69091


namespace polygon_sides_eq_eight_l69_69625

theorem polygon_sides_eq_eight (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
sorry

end polygon_sides_eq_eight_l69_69625


namespace problem1_problem2_l69_69622

section proof_problem

-- Define the sets as predicate functions
def A (x : ℝ) : Prop := x > 1
def B (x : ℝ) : Prop := -2 < x ∧ x < 2
def C (x : ℝ) : Prop := -3 < x ∧ x < 5

-- Define the union and intersection of sets
def union (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∨ B x
def inter (A B : ℝ → Prop) (x : ℝ) : Prop := A x ∧ B x

-- Proving that (A ∪ B) ∩ C = {x | -2 < x < 5}
theorem problem1 : ∀ x, (inter (union A B) C) x ↔ (-2 < x ∧ x < 5) := 
by
  sorry

-- Proving the arithmetic expression result
theorem problem2 : 
  ((2 + 1/4) ^ (1/2)) - ((-9.6) ^ 0) - ((3 + 3/8) ^ (-2/3)) + ((1.5) ^ (-2)) = 1/2 := 
by
  sorry

end proof_problem

end problem1_problem2_l69_69622


namespace husk_estimation_l69_69570

-- Define the conditions: total rice, sample size, and number of husks in the sample
def total_rice : ℕ := 1520
def sample_size : ℕ := 144
def husks_in_sample : ℕ := 18

-- Define the expected amount of husks in the total batch of rice
def expected_husks : ℕ := 190

-- The theorem stating the problem
theorem husk_estimation 
  (h : (husks_in_sample / sample_size) * total_rice = expected_husks) :
  (18 / 144) * 1520 = 190 := 
sorry

end husk_estimation_l69_69570


namespace bob_makes_weekly_profit_l69_69234

def weekly_profit (p_cost p_sell : ℝ) (m_daily d_week : ℕ) : ℝ :=
  (p_sell - p_cost) * m_daily * (d_week : ℝ)

theorem bob_makes_weekly_profit :
  weekly_profit 0.75 1.5 12 7 = 63 := 
by
  sorry

end bob_makes_weekly_profit_l69_69234


namespace inverse_proposition_l69_69034

theorem inverse_proposition :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ y : ℝ, y^2 > 0 → y < 0) :=
by
  sorry

end inverse_proposition_l69_69034


namespace percent_of_Q_l69_69223

theorem percent_of_Q (P Q : ℝ) (h : (50 / 100) * P = (20 / 100) * Q) : P = 0.4 * Q :=
sorry

end percent_of_Q_l69_69223


namespace slower_train_speed_is_36_l69_69783

def speed_of_slower_train (v : ℕ) : Prop :=
  let length_of_each_train := 100
  let distance_covered := length_of_each_train * 2
  let time_taken := 72
  let faster_train_speed := 46
  let relative_speed := (faster_train_speed - v) * (1000 / 3600)
  distance_covered = relative_speed * time_taken

theorem slower_train_speed_is_36 : ∃ v, speed_of_slower_train v ∧ v = 36 :=
by
  use 36
  unfold speed_of_slower_train
  -- Prove that the equation holds when v = 36
  sorry

end slower_train_speed_is_36_l69_69783


namespace sqrt_expression_eq_l69_69178

theorem sqrt_expression_eq :
  Real.sqrt (3 * (7 + 4 * Real.sqrt 3)) = 2 * Real.sqrt 3 + 3 := 
  sorry

end sqrt_expression_eq_l69_69178


namespace geometric_sequence_first_term_l69_69363

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 162) : a = 2 := by
  sorry

end geometric_sequence_first_term_l69_69363


namespace value_of_a_l69_69457

theorem value_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + x + a^2 - 1 = 0 → x = 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end value_of_a_l69_69457


namespace functional_equation_solution_l69_69637

/-- For all functions f: ℝ → ℝ, that satisfy the given functional equation -/
def functional_equation (f: ℝ → ℝ) : Prop :=
  ∀ x y: ℝ, f (x + y * f (x + y)) = y ^ 2 + f (x * f (y + 1))

/-- The solution to the functional equation is f(x) = x -/
theorem functional_equation_solution :
  ∀ f: ℝ → ℝ, functional_equation f → (∀ x: ℝ, f x = x) :=
by
  intros f h x
  sorry

end functional_equation_solution_l69_69637


namespace number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l69_69662

-- Part (a)
theorem number_of_ways_to_choose_4_from_28 :
  (Nat.choose 28 4) = 20475 :=
sorry

-- Part (b)
theorem number_of_ways_to_choose_3_from_27_with_kolya_included :
  (Nat.choose 27 3) = 2925 :=
sorry

end number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l69_69662


namespace combination_30_2_l69_69739

theorem combination_30_2 : Nat.choose 30 2 = 435 := by
  sorry

end combination_30_2_l69_69739


namespace problem_l69_69301

theorem problem : 
  let N := 63745.2981
  let place_value_7 := 1000 -- The place value of the digit 7 (thousands place)
  let place_value_2 := 0.1 -- The place value of the digit 2 (tenths place)
  place_value_7 / place_value_2 = 10000 :=
by
  sorry

end problem_l69_69301


namespace matrix_multiplication_example_l69_69400

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![-4, 5]]
def vector1 : Fin 2 → ℤ := ![4, -2]
def scalar : ℤ := 2
def result : Fin 2 → ℤ := ![32, -52]

theorem matrix_multiplication_example :
  scalar • (matrix1.mulVec vector1) = result := by
  sorry

end matrix_multiplication_example_l69_69400


namespace count_solutions_inequalities_l69_69357

theorem count_solutions_inequalities :
  {x : ℤ | -5 * x ≥ 2 * x + 10} ∩ {x : ℤ | -3 * x ≤ 15} ∩ {x : ℤ | -6 * x ≥ 3 * x + 21} = {x : ℤ | x = -5 ∨ x = -4 ∨ x = -3} :=
by 
  sorry

end count_solutions_inequalities_l69_69357


namespace last_passenger_probability_l69_69809

noncomputable def probability_last_passenger_sits_correctly (n : ℕ) : ℝ :=
if n = 0 then 0 else 1 / 2

theorem last_passenger_probability (n : ℕ) :
  (probability_last_passenger_sits_correctly n) = 1 / 2 :=
by {
  sorry
}

end last_passenger_probability_l69_69809


namespace infinitely_many_triples_of_integers_l69_69642

theorem infinitely_many_triples_of_integers (k : ℕ) :
  ∃ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) ∧
                  (x^999 + y^1000 = z^1001) :=
by
  sorry

end infinitely_many_triples_of_integers_l69_69642


namespace min_value_of_reciprocal_sums_l69_69179

variable {a b : ℝ}

theorem min_value_of_reciprocal_sums (ha : a ≠ 0) (hb : b ≠ 0) (h : 4 * a ^ 2 + b ^ 2 = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) = 9 := by
  sorry

end min_value_of_reciprocal_sums_l69_69179


namespace second_part_of_ratio_l69_69073

theorem second_part_of_ratio (h_ratio : ∀ (x : ℝ), 25 = 0.5 * (25 + x)) : ∃ x : ℝ, x = 25 :=
by
  sorry

end second_part_of_ratio_l69_69073


namespace ants_meet_at_QS_l69_69982

theorem ants_meet_at_QS (P Q R S : Type)
  (dist_PQ : Nat)
  (dist_QR : Nat)
  (dist_PR : Nat)
  (ants_meet : 2 * (dist_PQ + (5 : Nat)) = dist_PQ + dist_QR + dist_PR)
  (perimeter : dist_PQ + dist_QR + dist_PR = 24)
  (distance_each_ant_crawls : (dist_PQ + 5) = 12) :
  5 = 5 :=
by
  sorry

end ants_meet_at_QS_l69_69982


namespace find_p_q_l69_69100

theorem find_p_q : 
  (∀ x : ℝ, (x - 2) * (x + 1) ∣ (x ^ 5 - x ^ 4 + x ^ 3 - p * x ^ 2 + q * x - 8)) → (p = -1 ∧ q = -10) :=
by
  sorry

end find_p_q_l69_69100


namespace total_pastries_l69_69332

variable (P x : ℕ)

theorem total_pastries (h1 : P = 28 * (10 + x)) (h2 : P = 49 * (4 + x)) : P = 392 := 
by 
  sorry

end total_pastries_l69_69332


namespace thomas_weekly_wage_l69_69968

theorem thomas_weekly_wage (monthly_wage : ℕ) (weeks_in_month : ℕ) (weekly_wage : ℕ) 
    (h1 : monthly_wage = 19500) (h2 : weeks_in_month = 4) :
    weekly_wage = 4875 :=
by
  have h3 : weekly_wage = monthly_wage / weeks_in_month := sorry
  rw [h1, h2] at h3
  exact h3

end thomas_weekly_wage_l69_69968


namespace correct_proposition_l69_69808

-- Definitions
def p (x : ℝ) : Prop := x > 2 → x > 1 ∧ ¬ (x > 1 → x > 2)

def q (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

-- Propositions
def p_and_q (x a b : ℝ) := p x ∧ q a b
def not_p_or_q (x a b : ℝ) := ¬ (p x) ∨ q a b
def p_and_not_q (x a b : ℝ) := p x ∧ ¬ (q a b)
def not_p_and_not_q (x a b : ℝ) := ¬ (p x) ∧ ¬ (q a b)

-- Main theorem
theorem correct_proposition (x a b : ℝ) (h_p : p x) (h_q : ¬ (q a b)) :
  (p_and_q x a b = false) ∧
  (not_p_or_q x a b = false) ∧
  (p_and_not_q x a b = true) ∧
  (not_p_and_not_q x a b = false) :=
by
  sorry

end correct_proposition_l69_69808


namespace exponent_subtraction_l69_69204

variable {a : ℝ} {m n : ℕ}

theorem exponent_subtraction (hm : a ^ m = 12) (hn : a ^ n = 3) : a ^ (m - n) = 4 :=
by
  sorry

end exponent_subtraction_l69_69204


namespace smallest_four_digit_in_pascals_triangle_l69_69009

theorem smallest_four_digit_in_pascals_triangle : ∃ n, ∃ k, k ≤ n ∧ 1000 ≤ Nat.choose n k :=
sorry

end smallest_four_digit_in_pascals_triangle_l69_69009


namespace money_allocation_l69_69542

theorem money_allocation (x y : ℝ) (h1 : x + 1/2 * y = 50) (h2 : y + 2/3 * x = 50) : 
  x + 1/2 * y = 50 ∧ y + 2/3 * x = 50 :=
by
  exact ⟨h1, h2⟩

end money_allocation_l69_69542


namespace find_a_l69_69712

theorem find_a (a : ℝ) (h : 2 * a + 3 = -3) : a = -3 := 
by 
  sorry

end find_a_l69_69712


namespace last_three_digits_of_3_pow_5000_l69_69347

theorem last_three_digits_of_3_pow_5000 : (3 ^ 5000) % 1000 = 1 := 
by
  -- skip the proof
  sorry

end last_three_digits_of_3_pow_5000_l69_69347


namespace prove_ellipse_and_dot_product_l69_69317

open Real

-- Assume the given conditions
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : a > b)
variable (e : ℝ) (he : e = sqrt 2 / 2)
variable (h_chord : 2 = 2 * sqrt (a^2 - 1))
variables (k : ℝ) (hk : k ≠ 0)

-- Given equation of points on the line and the ellipse
def ellipse_eq (x y : ℝ) : Prop := (x^2) / 2 + y^2 = 1
def line_eq (x y : ℝ) : Prop := y = k * (x - 1)

-- The points A and B lie on the ellipse and the line
variables (x1 y1 x2 y2 : ℝ)
variable (A : x1^2 / 2 + y1^2 = 1 ∧ y1 = k * (x1 - 1))
variable (B : x2^2 / 2 + y2^2 = 1 ∧ y2 = k * (x2 - 1))

-- Define the dot product condition
def MA_dot_MB (m : ℝ) : ℝ :=
  let x1_term := x1 - m
  let x2_term := x2 - m
  let dot_product := (x1_term * x2_term + y1 * y2)
  dot_product

-- The statement we need to prove
theorem prove_ellipse_and_dot_product :
  (a^2 = 2) ∧ (b = 1) ∧ (c = 1) ∧ (∃ (m : ℝ), m = 5 / 4 ∧ MA_dot_MB m = -7 / 16) :=
sorry

end prove_ellipse_and_dot_product_l69_69317


namespace solution_set_of_inequality_l69_69070

theorem solution_set_of_inequality (a : ℝ) :
  (a > 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x < a + 1}) ∧
  (a < 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x > a + 1}) ∧
  (a = 1 → {x : ℝ | ax + 1 < a^2 + x} = ∅) := 
  sorry

end solution_set_of_inequality_l69_69070


namespace blithe_initial_toys_l69_69797

-- Define the conditions as given in the problem
def lost_toys : ℤ := 6
def found_toys : ℤ := 9
def final_toys : ℤ := 43

-- Define the problem statement to prove the initial number of toys
theorem blithe_initial_toys (T : ℤ) (h : T - lost_toys + found_toys = final_toys) : T = 40 :=
sorry

end blithe_initial_toys_l69_69797


namespace sequence_sum_l69_69361

theorem sequence_sum (a b : ℤ) (h1 : ∃ d, d = 5 ∧ (∀ n : ℕ, (3 + n * d) = a ∨ (3 + (n-1) * d) = b ∨ (3 + (n-2) * d) = 33)) : 
  a + b = 51 :=
by
  sorry

end sequence_sum_l69_69361


namespace find_m_value_l69_69883

-- Condition: P(-m^2, 3) lies on the axis of symmetry of the parabola y^2 = mx
def point_on_axis_of_symmetry (m : ℝ) : Prop :=
  let P := (-m^2, 3)
  let axis_of_symmetry := (-m / 4)
  P.1 = axis_of_symmetry

theorem find_m_value (m : ℝ) (h : point_on_axis_of_symmetry m) : m = 1 / 4 :=
  sorry

end find_m_value_l69_69883


namespace gcd_840_1764_gcd_98_63_l69_69156

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := 
by sorry

theorem gcd_98_63 : Nat.gcd 98 63 = 7 :=
by sorry

end gcd_840_1764_gcd_98_63_l69_69156


namespace pond_87_5_percent_algae_free_on_day_17_l69_69147

/-- The algae in a local pond doubles every day. -/
def algae_doubles_every_day (coverage : ℕ → ℝ) : Prop :=
  ∀ n, coverage (n + 1) = 2 * coverage n

/-- The pond is completely covered in algae on day 20. -/
def pond_completely_covered_on_day_20 (coverage : ℕ → ℝ) : Prop :=
  coverage 20 = 1

/-- Determine the day on which the pond was 87.5% algae-free. -/
theorem pond_87_5_percent_algae_free_on_day_17 (coverage : ℕ → ℝ)
  (h1 : algae_doubles_every_day coverage)
  (h2 : pond_completely_covered_on_day_20 coverage) :
  coverage 17 = 0.125 :=
sorry

end pond_87_5_percent_algae_free_on_day_17_l69_69147


namespace distinct_real_roots_iff_l69_69511

-- Define f(x, a) := |x^2 - a| - x + 2
noncomputable def f (x a : ℝ) : ℝ := abs (x^2 - a) - x + 2

-- The proposition we need to prove
theorem distinct_real_roots_iff (a : ℝ) (h : 0 < a) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 4 < a :=
by
  sorry

end distinct_real_roots_iff_l69_69511


namespace hiking_hours_l69_69749

theorem hiking_hours
  (violet_water_per_hour : ℕ := 800)
  (dog_water_per_hour : ℕ := 400)
  (total_water : ℕ := 4800) :
  (total_water / (violet_water_per_hour + dog_water_per_hour) = 4) :=
by
  sorry

end hiking_hours_l69_69749


namespace number_of_houses_around_square_l69_69760

namespace HouseCounting

-- Definitions for the conditions
def M (k : ℕ) : ℕ := k
def J (k : ℕ) : ℕ := k

-- The main theorem stating the solution
theorem number_of_houses_around_square (n : ℕ)
  (h1 : M 5 % n = J 12 % n)
  (h2 : J 5 % n = M 30 % n) : n = 32 :=
sorry

end HouseCounting

end number_of_houses_around_square_l69_69760


namespace cyclist_north_speed_l69_69671

variable {v : ℝ} -- Speed of the cyclist going north.

-- Conditions: 
def speed_south := 15 -- Speed of the cyclist going south (15 kmph).
def time := 2 -- The time after which they are 50 km apart (2 hours).
def distance := 50 -- The distance they are apart after 2 hours (50 km).

-- Theorem statement:
theorem cyclist_north_speed :
    (v + speed_south) * time = distance → v = 10 := by
  intro h
  sorry

end cyclist_north_speed_l69_69671


namespace remainder_of_a_sq_plus_five_mod_seven_l69_69114

theorem remainder_of_a_sq_plus_five_mod_seven (a : ℕ) (h : a % 7 = 4) : (a^2 + 5) % 7 = 0 := 
by 
  sorry

end remainder_of_a_sq_plus_five_mod_seven_l69_69114


namespace DVDs_sold_168_l69_69590

-- Definitions of the conditions
def CDs_sold := ℤ
def DVDs_sold := ℤ

def ratio_condition (C D : ℤ) : Prop := D = 16 * C / 10
def total_condition (C D : ℤ) : Prop := D + C = 273

-- The main statement to prove
theorem DVDs_sold_168 (C D : ℤ) 
  (h1 : ratio_condition C D) 
  (h2 : total_condition C D) : D = 168 :=
sorry

end DVDs_sold_168_l69_69590


namespace min_value_l69_69325

open Real

noncomputable def y1 (x1 : ℝ) : ℝ := x1 * log x1
noncomputable def y2 (x2 : ℝ) : ℝ := x2 - 3

theorem min_value :
  ∃ (x1 x2 : ℝ), (x1 - x2)^2 + (y1 x1 - y2 x2)^2 = 2 :=
by
  sorry

end min_value_l69_69325


namespace shaded_region_area_l69_69548

theorem shaded_region_area
  (n : ℕ) (d : ℝ) 
  (h₁ : n = 25) 
  (h₂ : d = 10) 
  (h₃ : n > 0) : 
  (d^2 / n = 2) ∧ (n * (d^2 / (2 * n)) = 50) :=
by 
  sorry

end shaded_region_area_l69_69548


namespace find_area_of_triangle_l69_69062

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem find_area_of_triangle :
  let a := 10
  let b := 10
  let c := 12
  triangle_area a b c = 48 := 
by 
  sorry

end find_area_of_triangle_l69_69062


namespace find_x_l69_69273

theorem find_x (x : ℝ) : 
  45 - (28 - (37 - (x - 17))) = 56 ↔ x = 15 := 
by
  sorry

end find_x_l69_69273


namespace case_D_has_two_solutions_l69_69731

-- Definitions for the conditions of each case
structure CaseA :=
(b : ℝ) (A : ℝ) (B : ℝ)

structure CaseB :=
(a : ℝ) (c : ℝ) (B : ℝ)

structure CaseC :=
(a : ℝ) (b : ℝ) (A : ℝ)

structure CaseD :=
(a : ℝ) (b : ℝ) (A : ℝ)

-- Setting the values based on the given conditions
def caseA := CaseA.mk 10 45 70
def caseB := CaseB.mk 60 48 100
def caseC := CaseC.mk 14 16 45
def caseD := CaseD.mk 7 5 80

-- Define a function that checks if a case has two solutions
def has_two_solutions (a b c : ℝ) (A B : ℝ) : Prop := sorry

-- The theorem to prove that out of the given cases, only Case D has two solutions
theorem case_D_has_two_solutions :
  has_two_solutions caseA.b caseB.B caseC.a caseC.b caseC.A = false →
  has_two_solutions caseB.a caseB.c caseB.B caseC.b caseC.A = false →
  has_two_solutions caseC.a caseC.b caseC.A caseD.a caseD.b = false →
  has_two_solutions caseD.a caseD.b caseD.A caseA.b caseA.A = true :=
sorry

end case_D_has_two_solutions_l69_69731


namespace find_m_l69_69543

theorem find_m (m : ℝ) (h : ∀ x : ℝ, x - m > 5 ↔ x > 2) : m = -3 := by
  sorry

end find_m_l69_69543


namespace gumballs_problem_l69_69161

theorem gumballs_problem 
  (L x : ℕ)
  (h1 : 19 ≤ (17 + L + x) / 3 ∧ (17 + L + x) / 3 ≤ 25)
  (h2 : ∃ x_min x_max, x_max - x_min = 18 ∧ x_min = 19 ∧ x = x_min ∨ x = x_max) : 
  L = 21 :=
sorry

end gumballs_problem_l69_69161


namespace find_scalars_l69_69013

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 7], ![-3, -1]]
def M_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![-17, 7], ![-3, -20]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem find_scalars :
  ∃ p q : ℤ, M_squared = p • M + q • I ∧ (p, q) = (1, -19) := sorry

end find_scalars_l69_69013


namespace queenie_daily_earnings_l69_69351

/-- Define the overtime earnings per hour. -/
def overtime_pay_per_hour : ℤ := 5

/-- Define the total amount received. -/
def total_received : ℤ := 770

/-- Define the number of days worked. -/
def days_worked : ℤ := 5

/-- Define the number of overtime hours. -/
def overtime_hours : ℤ := 4

/-- State the theorem to find out Queenie's daily earnings. -/
theorem queenie_daily_earnings :
  ∃ D : ℤ, days_worked * D + overtime_hours * overtime_pay_per_hour = total_received ∧ D = 150 :=
by
  use 150
  sorry

end queenie_daily_earnings_l69_69351


namespace seventh_term_of_arithmetic_sequence_l69_69579

theorem seventh_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : 5 * a + 10 * d = 35)
  (h2 : a + 5 * d = 10) :
  a + 6 * d = 11 :=
by
  sorry

end seventh_term_of_arithmetic_sequence_l69_69579


namespace max_principals_ten_years_l69_69504

theorem max_principals_ten_years : 
  (∀ (P : ℕ → Prop), (∀ n, n ≥ 10 → ∀ i, ¬P (n - i)) → ∀ p, p ≤ 4 → 
  (∃ n ≤ 10, ∀ k, k ≥ n → P k)) :=
sorry

end max_principals_ten_years_l69_69504


namespace bobby_shoes_l69_69381

variable (Bonny_pairs Becky_pairs Bobby_pairs : ℕ)
variable (h1 : Bonny_pairs = 13)
variable (h2 : 2 * Becky_pairs - 5 = Bonny_pairs)
variable (h3 : Bobby_pairs = 3 * Becky_pairs)

theorem bobby_shoes : Bobby_pairs = 27 :=
by
  -- Use the conditions to prove the required theorem
  sorry

end bobby_shoes_l69_69381


namespace michael_laps_to_pass_donovan_l69_69393

theorem michael_laps_to_pass_donovan (track_length : ℕ) (donovan_lap_time : ℕ) (michael_lap_time : ℕ) 
  (h1 : track_length = 400) (h2 : donovan_lap_time = 48) (h3 : michael_lap_time = 40) : 
  michael_lap_time * 6 = donovan_lap_time * (michael_lap_time * 6 / track_length * michael_lap_time) :=
by
  sorry

end michael_laps_to_pass_donovan_l69_69393


namespace correct_total_cost_correct_remaining_donuts_l69_69307

-- Conditions
def budget : ℝ := 50
def cost_per_box : ℝ := 12
def discount_percentage : ℝ := 0.10
def number_of_boxes_bought : ℕ := 4
def donuts_per_box : ℕ := 12
def boxes_given_away : ℕ := 1
def additional_donuts_given_away : ℕ := 6

-- Calculations based on conditions
def total_cost_before_discount : ℝ := number_of_boxes_bought * cost_per_box
def discount_amount : ℝ := discount_percentage * total_cost_before_discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

def total_donuts : ℕ := number_of_boxes_bought * donuts_per_box
def total_donuts_given_away : ℕ := (boxes_given_away * donuts_per_box) + additional_donuts_given_away
def remaining_donuts : ℕ := total_donuts - total_donuts_given_away

-- Theorems to prove
theorem correct_total_cost : total_cost_after_discount = 43.20 := by
  -- proof here
  sorry

theorem correct_remaining_donuts : remaining_donuts = 30 := by
  -- proof here
  sorry

end correct_total_cost_correct_remaining_donuts_l69_69307


namespace prime_factor_condition_l69_69039

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 1
  | n + 2 => seq (n + 1) + seq n

theorem prime_factor_condition (p k : ℕ) (hp : Nat.Prime p) (h : p ∣ seq (2 * k) - 2) :
  p ∣ seq (2 * k - 1) - 1 :=
sorry

end prime_factor_condition_l69_69039


namespace solution_set_f_prime_pos_l69_69536

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

theorem solution_set_f_prime_pos : 
  {x : ℝ | 0 < x ∧ (deriv f x > 0)} = {x : ℝ | 2 < x} :=
by
  sorry

end solution_set_f_prime_pos_l69_69536


namespace max_value_quadratic_l69_69226

noncomputable def quadratic (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

theorem max_value_quadratic : ∀ x : ℝ, quadratic x ≤ -3 ∧ (∀ y : ℝ, quadratic y = -3 → (∀ z : ℝ, quadratic z ≤ quadratic y)) :=
by
  sorry

end max_value_quadratic_l69_69226


namespace candies_problem_l69_69312

theorem candies_problem (x : ℕ) (Nina : ℕ) (Oliver : ℕ) (total_candies : ℕ) (h1 : 4 * x = Mark) (h2 : 2 * Mark = Nina) (h3 : 6 * Nina = Oliver) (h4 : x + Mark + Nina + Oliver = total_candies) :
  x = 360 / 61 :=
by
  sorry

end candies_problem_l69_69312


namespace problem_proof_l69_69063

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem problem_proof :
  (∀ x, g (x + Real.pi) = g x) ∧ (∀ y, g (2 * (Real.pi / 12) - y) = g y) :=
by
  sorry

end problem_proof_l69_69063


namespace inequality_proof_l69_69462

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a / (a + 2 * b)^(1/3) + b / (b + 2 * c)^(1/3) + c / (c + 2 * a)^(1/3)) ≥ 1 := 
by
  sorry

end inequality_proof_l69_69462


namespace original_ratio_l69_69387

theorem original_ratio (x y : ℕ) (h1 : y = 15) (h2 : x + 10 = y) : x / y = 1 / 3 :=
by
  sorry

end original_ratio_l69_69387


namespace best_of_five_advantageous_l69_69782

theorem best_of_five_advantageous (p : ℝ) (h : p > 0.5) :
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    p2 > p1 :=
by 
    let p1 := p^2 + 2 * p^2 * (1 - p)
    let p2 := p^3 + 3 * p^3 * (1 - p) + 6 * p^3 * (1 - p)^2
    sorry -- an actual proof would go here

end best_of_five_advantageous_l69_69782


namespace equation_true_l69_69503

variables {AB BC CD AD AC BD : ℝ}

theorem equation_true :
  (AD * BC + AB * CD = AC * BD) ∧
  (AD * BC - AB * CD ≠ AC * BD) ∧
  (AB * BC + AC * CD ≠ AC * BD) ∧
  (AB * BC - AC * CD ≠ AC * BD) :=
by
  sorry

end equation_true_l69_69503


namespace bianca_birthday_money_l69_69647

/-- Define the number of friends Bianca has -/
def number_of_friends : ℕ := 5

/-- Define the amount of dollars each friend gave -/
def dollars_per_friend : ℕ := 6

/-- The total amount of dollars Bianca received -/
def total_dollars_received : ℕ := number_of_friends * dollars_per_friend

/-- Prove that the total amount of dollars Bianca received is 30 -/
theorem bianca_birthday_money : total_dollars_received = 30 :=
by
  sorry

end bianca_birthday_money_l69_69647


namespace danny_age_l69_69554

theorem danny_age (D : ℕ) (h : D - 19 = 3 * (26 - 19)) : D = 40 := by
  sorry

end danny_age_l69_69554


namespace gary_has_left_amount_l69_69657

def initial_amount : ℝ := 100
def cost_pet_snake : ℝ := 55
def cost_toy_car : ℝ := 12
def cost_novel : ℝ := 7.5
def cost_pack_stickers : ℝ := 3.25
def number_packs_stickers : ℕ := 3

theorem gary_has_left_amount : initial_amount - (cost_pet_snake + cost_toy_car + cost_novel + number_packs_stickers * cost_pack_stickers) = 15.75 :=
by
  sorry

end gary_has_left_amount_l69_69657


namespace tiles_per_row_24_l69_69911

noncomputable def num_tiles_per_row (area : ℝ) (tile_size : ℝ) : ℝ :=
  let side_length_ft := Real.sqrt area
  let side_length_in := side_length_ft * 12
  side_length_in / tile_size

theorem tiles_per_row_24 :
  num_tiles_per_row 324 9 = 24 :=
by
  sorry

end tiles_per_row_24_l69_69911


namespace jacks_paycheck_l69_69778

theorem jacks_paycheck (P : ℝ) (h1 : 0.2 * 0.8 * P = 20) : P = 125 :=
sorry

end jacks_paycheck_l69_69778


namespace tan_product_equals_three_l69_69143

noncomputable def tan_pi_div_9 := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_div_9 := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_div_9 := Real.tan (4 * Real.pi / 9)

theorem tan_product_equals_three 
  (h1 : tan_pi_div_9 ≠ 0)
  (h2 : tan_2pi_div_9 ≠ 0)
  (h3 : tan_4pi_div_9 ≠ 0)
  (h4 : tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3) :
  tan_pi_div_9 * tan_2pi_div_9 * tan_4pi_div_9 = 3 :=
by
  sorry

end tan_product_equals_three_l69_69143


namespace marketing_firm_l69_69652

variable (Total_households : ℕ) (A_only : ℕ) (A_and_B : ℕ) (B_to_A_and_B_ratio : ℕ)

def neither_soap_households : ℕ :=
  Total_households - (A_only + (B_to_A_and_B_ratio * A_and_B) + A_and_B)

theorem marketing_firm (h1 : Total_households = 300)
                       (h2 : A_only = 60)
                       (h3 : A_and_B = 40)
                       (h4 : B_to_A_and_B_ratio = 3)
                       : neither_soap_households 300 60 40 3 = 80 :=
by {
  sorry
}

end marketing_firm_l69_69652


namespace impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l69_69420

theorem impossible_to_get_60_pieces :
  ¬ ∃ (n m : ℕ), 1 + 7 * n + 11 * m = 60 :=
sorry

theorem possible_to_get_more_than_60_pieces :
  ∀ k > 60, ∃ (n m : ℕ), 1 + 7 * n + 11 * m = k :=
sorry

end impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l69_69420


namespace fractional_units_l69_69920

-- Define the mixed number and the smallest composite number
def mixed_number := 3 + 2/7
def smallest_composite := 4

-- To_struct fractional units of 3 2/7
theorem fractional_units (u : ℚ) (n : ℕ) (m : ℕ):
  u = 1/7 ∧ n = 23 ∧ m = 5 :=
by
  have h1 : u = 1 / 7 := sorry
  have h2 : mixed_number = 23 * u := sorry
  have h3 : smallest_composite - mixed_number = 5 * u := sorry
  have h4 : n = 23 := sorry
  have h5 : m = 5 := sorry
  exact ⟨h1, h4, h5⟩

end fractional_units_l69_69920


namespace cap_to_sunglasses_prob_l69_69187

-- Define the conditions
def num_people_wearing_sunglasses : ℕ := 60
def num_people_wearing_caps : ℕ := 40
def prob_sunglasses_and_caps : ℚ := 1 / 3

-- Define the statement to prove
theorem cap_to_sunglasses_prob : 
  (num_people_wearing_sunglasses * prob_sunglasses_and_caps) / num_people_wearing_caps = 1 / 2 :=
by
  sorry

end cap_to_sunglasses_prob_l69_69187


namespace washed_shirts_l69_69539

-- Definitions based on the conditions
def short_sleeve_shirts : ℕ := 39
def long_sleeve_shirts : ℕ := 47
def unwashed_shirts : ℕ := 66

-- The total number of shirts is the sum of short and long sleeve shirts
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts

-- The problem to prove that Oliver washed 20 shirts
theorem washed_shirts :
  total_shirts - unwashed_shirts = 20 := 
sorry

end washed_shirts_l69_69539


namespace ratio_of_drinking_speeds_l69_69346

def drinking_ratio(mala_portion usha_portion : ℚ) (same_time: Bool) (usha_fraction: ℚ) : ℚ :=
if same_time then mala_portion / usha_portion else 0

theorem ratio_of_drinking_speeds
  (mala_portion : ℚ)
  (usha_portion : ℚ)
  (same_time : Bool)
  (usha_fraction : ℚ)
  (usha_drank : usha_fraction = 2 / 10)
  (mala_drank : mala_portion = 1 - usha_fraction)
  (equal_time : same_time = tt)
  (ratio : drinking_ratio mala_portion usha_portion same_time usha_fraction = 4) :
  mala_portion / usha_portion = 4 :=
by
  sorry

end ratio_of_drinking_speeds_l69_69346


namespace sequence_problem_l69_69074

theorem sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1)) (h_eq : a 100 = a 96) :
  a 2018 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end sequence_problem_l69_69074


namespace legendre_symbol_two_l69_69815

theorem legendre_symbol_two (m : ℕ) [Fact (Nat.Prime m)] (hm : Odd m) :
  (legendreSym 2 m) = (-1 : ℤ) ^ ((m^2 - 1) / 8) :=
sorry

end legendre_symbol_two_l69_69815


namespace scalene_triangle_angle_difference_l69_69410

def scalene_triangle (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem scalene_triangle_angle_difference (x y : ℝ) :
  (x + y = 100) → scalene_triangle x y 80 → (x - y = 80) :=
by
  intros h1 h2
  sorry

end scalene_triangle_angle_difference_l69_69410


namespace cost_of_one_book_l69_69919

theorem cost_of_one_book (m : ℕ) (H1: 1100 < 900 + 9 * m ∧ 900 + 9 * m < 1200)
                                (H2: 1500 < 1300 + 13 * m ∧ 1300 + 13 * m < 1600) : 
                                m = 23 :=
by {
  sorry
}

end cost_of_one_book_l69_69919


namespace cos_expression_range_l69_69569

theorem cos_expression_range (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hSum : A + B + C = Real.pi) :
  -25 / 16 < 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) ∧ 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) < 6 :=
sorry

end cos_expression_range_l69_69569


namespace greatest_three_digit_multiple_of_17_l69_69460

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l69_69460


namespace total_watermelon_weight_l69_69144

theorem total_watermelon_weight :
  let w1 := 9.91
  let w2 := 4.112
  let w3 := 6.059
  w1 + w2 + w3 = 20.081 :=
by
  sorry

end total_watermelon_weight_l69_69144


namespace amount_spent_on_first_shop_l69_69248

-- Define the conditions
def booksFromFirstShop : ℕ := 65
def costFromSecondShop : ℕ := 2000
def booksFromSecondShop : ℕ := 35
def avgPricePerBook : ℕ := 85

-- Calculate the total books and the total amount spent
def totalBooks : ℕ := booksFromFirstShop + booksFromSecondShop
def totalAmountSpent : ℕ := totalBooks * avgPricePerBook

-- Prove the amount spent on the books from the first shop is Rs. 6500
theorem amount_spent_on_first_shop : 
  (totalAmountSpent - costFromSecondShop) = 6500 :=
by
  sorry

end amount_spent_on_first_shop_l69_69248


namespace solve_system_of_equations_l69_69813

theorem solve_system_of_equations (u v w : ℝ) (h₀ : u ≠ 0) (h₁ : v ≠ 0) (h₂ : w ≠ 0) :
  (3 / (u * v) + 15 / (v * w) = 2) ∧
  (15 / (v * w) + 5 / (w * u) = 2) ∧
  (5 / (w * u) + 3 / (u * v) = 2) →
  (u = 1 ∧ v = 3 ∧ w = 5) ∨
  (u = -1 ∧ v = -3 ∧ w = -5) :=
by
  sorry

end solve_system_of_equations_l69_69813


namespace proof_problem_l69_69014

variable {a b c d e f : ℝ}

theorem proof_problem :
  (a * b * c = 130) →
  (b * c * d = 65) →
  (d * e * f = 250) →
  (a * f / (c * d) = 0.5) →
  (c * d * e = 1000) :=
by
  intros h1 h2 h3 h4
  sorry

end proof_problem_l69_69014


namespace largest_AC_value_l69_69835

theorem largest_AC_value : ∃ (a b c d : ℕ), 
  a < 20 ∧ b < 20 ∧ c < 20 ∧ d < 20 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ (AC BD : ℝ), AC * BD = a * c + b * d ∧
  AC ^ 2 + BD ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 ∧
  AC = Real.sqrt 458) :=
sorry

end largest_AC_value_l69_69835


namespace train_speed_proof_l69_69418

noncomputable def speed_of_train (train_length : ℝ) (time_seconds : ℝ) (man_speed : ℝ) : ℝ :=
  let train_length_km := train_length / 1000
  let time_hours := time_seconds / 3600
  let relative_speed := train_length_km / time_hours
  relative_speed - man_speed

theorem train_speed_proof :
  speed_of_train 605 32.99736021118311 6 = 60.028 :=
by
  unfold speed_of_train
  -- Direct substitution and expected numerical simplification
  norm_num
  sorry

end train_speed_proof_l69_69418


namespace emails_in_afternoon_l69_69842

theorem emails_in_afternoon (A : ℕ) 
  (morning_emails : A + 3 = 10) : A = 7 :=
by {
    sorry
}

end emails_in_afternoon_l69_69842


namespace functions_satisfying_equation_l69_69183

theorem functions_satisfying_equation 
  (f g h : ℝ → ℝ)
  (H : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, f x = a * x^2 + b * x + c) ∧ 
    (∀ x : ℝ, g x = a * x^2 + b * x + c) ∧ 
    (∀ x : ℝ, h x = a * x + b) :=
sorry

end functions_satisfying_equation_l69_69183


namespace expand_expression_l69_69660

variables {R : Type*} [CommRing R] (x : R)

theorem expand_expression : (15 * x^2 + 5) * 3 * x^3 = 45 * x^5 + 15 * x^3 :=
by sorry

end expand_expression_l69_69660


namespace sum_series_1_to_60_l69_69711

-- Define what it means to be the sum of the first n natural numbers
def sum_n (n : Nat) : Nat := n * (n + 1) / 2

theorem sum_series_1_to_60 : sum_n 60 = 1830 :=
by
  sorry

end sum_series_1_to_60_l69_69711


namespace units_digit_of_sum_is_7_l69_69213

noncomputable def original_num (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
noncomputable def reversed_num (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

theorem units_digit_of_sum_is_7 (a b c : ℕ) (h : a = 2 * c - 3) :
  (original_num a b c + reversed_num a b c) % 10 = 7 := by
  sorry

end units_digit_of_sum_is_7_l69_69213


namespace compute_value_of_expression_l69_69315

theorem compute_value_of_expression (p q : ℝ) (h₁ : 3 * p ^ 2 - 5 * p - 12 = 0) (h₂ : 3 * q ^ 2 - 5 * q - 12 = 0) :
  (3 * p ^ 2 - 3 * q ^ 2) / (p - q) = 5 :=
by
  sorry

end compute_value_of_expression_l69_69315


namespace compare_two_sqrt_five_five_l69_69318

theorem compare_two_sqrt_five_five : 2 * Real.sqrt 5 < 5 :=
sorry

end compare_two_sqrt_five_five_l69_69318


namespace system1_solution_system2_solution_l69_69981

theorem system1_solution (x y : ℝ) (h₁ : y = 2 * x) (h₂ : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 := 
by sorry

theorem system2_solution (x y : ℝ) (h₁ : x - 3 * y = -2) (h₂ : 2 * x + 3 * y = 3) : x = (1 / 3) ∧ y = (7 / 9) := 
by sorry

end system1_solution_system2_solution_l69_69981


namespace determine_d_iff_l69_69733

theorem determine_d_iff (x : ℝ) : 
  (x ∈ Set.Ioo (-5/2) 3) ↔ (x * (2 * x + 3) < 15) :=
by
  sorry

end determine_d_iff_l69_69733


namespace rectangle_ratio_constant_l69_69373

theorem rectangle_ratio_constant (length width : ℝ) (d k : ℝ)
  (h1 : length/width = 5/2)
  (h2 : 2 * (length + width) = 28)
  (h3 : d^2 = length^2 + width^2)
  (h4 : (length * width) = k * d^2) :
  k = (10/29) := by
  sorry

end rectangle_ratio_constant_l69_69373


namespace cyclists_meet_time_l69_69035

/-- 
  Two cyclists start on a circular track from a given point but in opposite directions with speeds of 7 m/s and 8 m/s.
  The circumference of the circle is 180 meters.
  After what time will they meet at the starting point? 
-/
theorem cyclists_meet_time :
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  (circumference / (speed1 + speed2) = 12) :=
by
  let speed1 := 7 -- m/s
  let speed2 := 8 -- m/s
  let circumference := 180 -- meters
  sorry

end cyclists_meet_time_l69_69035


namespace find_x_l69_69015

theorem find_x (p q r s x : ℚ) (hpq : p ≠ q) (hq0 : q ≠ 0) 
    (h : (p + x) / (q - x) = r / s) 
    (hp : p = 3) (hq : q = 5) (hr : r = 7) (hs : s = 9) : 
    x = 1/2 :=
by {
  sorry
}

end find_x_l69_69015


namespace one_eighth_of_two_pow_36_eq_two_pow_y_l69_69384

theorem one_eighth_of_two_pow_36_eq_two_pow_y (y : ℕ) : (2^36 / 8 = 2^y) → (y = 33) :=
by
  sorry

end one_eighth_of_two_pow_36_eq_two_pow_y_l69_69384


namespace least_num_subtracted_l69_69367

theorem least_num_subtracted 
  {x : ℤ} 
  (h5 : (642 - x) % 5 = 4) 
  (h7 : (642 - x) % 7 = 4) 
  (h9 : (642 - x) % 9 = 4) : 
  x = 4 := 
sorry

end least_num_subtracted_l69_69367


namespace gcd_of_18_and_30_l69_69496

-- Define the numbers
def a := 18
def b := 30

-- The main theorem statement
theorem gcd_of_18_and_30 : Nat.gcd a b = 6 := by
  sorry

end gcd_of_18_and_30_l69_69496


namespace negation_proposition_l69_69501

-- Definitions based on the conditions
def original_proposition : Prop := ∃ x : ℝ, x^2 + 3*x + 2 < 0

-- Theorem requiring proof
theorem negation_proposition : (¬ original_proposition) = ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_proposition_l69_69501


namespace shopping_money_l69_69931

theorem shopping_money (X : ℝ) (h : 0.70 * X = 840) : X = 1200 :=
sorry

end shopping_money_l69_69931


namespace find_a_plus_d_l69_69037

theorem find_a_plus_d (a b c d : ℝ) (h₁ : ab + bc + ca + db = 42) (h₂ : b + c = 6) : a + d = 7 := 
sorry

end find_a_plus_d_l69_69037


namespace S8_is_255_l69_69748

-- Definitions and hypotheses
def geometric_sequence_sum (a : ℕ → ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  a 0 * (1 - q^n) / (1 - q)

variables (a : ℕ → ℚ) (q : ℚ)
variable (h_geo_seq : ∀ n, a (n + 1) = a n * q)
variable (h_S2 : geometric_sequence_sum a q 2 = 3)
variable (h_S4 : geometric_sequence_sum a q 4 = 15)

-- Goal
theorem S8_is_255 : geometric_sequence_sum a q 8 = 255 := 
by {
  -- skipping the proof
  sorry
}

end S8_is_255_l69_69748


namespace problem_solution_l69_69136

variable (α : ℝ)
variable (h : Real.cos α = 1 / 5)

theorem problem_solution : Real.cos (2 * α - 2017 * Real.pi) = 23 / 25 := by
  sorry

end problem_solution_l69_69136


namespace find_DF_l69_69533

noncomputable def triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def median (a b : ℝ) : ℝ := a / 2

theorem find_DF {DE EF DM DF : ℝ} (h1 : DE = 7) (h2 : EF = 10) (h3 : DM = 5) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l69_69533


namespace candy_distribution_l69_69966

theorem candy_distribution (n : ℕ) (h1 : n > 0) (h2 : 100 % n = 0) (h3 : 99 % n = 0) : n = 11 :=
sorry

end candy_distribution_l69_69966


namespace abs_fraction_inequality_solution_l69_69582

theorem abs_fraction_inequality_solution (x : ℝ) (h : x ≠ 2) :
  (abs ((3 * x - 2) / (x - 2)) > 3) ↔ (x < 4/3 ∨ x > 2) :=
by
  sorry

end abs_fraction_inequality_solution_l69_69582


namespace cone_apex_angle_l69_69663

theorem cone_apex_angle (R : ℝ) 
  (h1 : ∀ (θ : ℝ), (∃ (r : ℝ), r = R / 2 ∧ 2 * π * r = π * R)) :
  ∀ (θ : ℝ), θ = π / 3 :=
by
  sorry

end cone_apex_angle_l69_69663


namespace age_of_beckett_l69_69975

variables (B O S J : ℕ)

theorem age_of_beckett
  (h1 : B = O - 3)
  (h2 : S = O - 2)
  (h3 : J = 2 * S + 5)
  (h4 : B + O + S + J = 71) :
  B = 12 :=
by
  sorry

end age_of_beckett_l69_69975


namespace seashells_total_l69_69489

theorem seashells_total (tim_seashells sally_seashells : ℕ) (ht : tim_seashells = 37) (hs : sally_seashells = 13) :
  tim_seashells + sally_seashells = 50 := 
by 
  sorry

end seashells_total_l69_69489


namespace solve_inequality_l69_69676

theorem solve_inequality (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 6) ≥ 1) ↔ (x ∈ Set.Icc (-4) (-2) ∨ x ∈ Set.Icc 2 4) :=
sorry

end solve_inequality_l69_69676


namespace vinegar_evaporation_rate_l69_69703

def percentage_vinegar_evaporates_each_year (x : ℕ) : Prop :=
  let initial_vinegar : ℕ := 100
  let vinegar_left_after_first_year : ℕ := initial_vinegar - x
  let vinegar_left_after_two_years : ℕ := vinegar_left_after_first_year * (100 - x) / 100
  vinegar_left_after_two_years = 64

theorem vinegar_evaporation_rate :
  ∃ x : ℕ, percentage_vinegar_evaporates_each_year x ∧ x = 20 :=
by
  sorry

end vinegar_evaporation_rate_l69_69703


namespace am_gm_hm_inequality_l69_69822

theorem am_gm_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : a ≠ c) : 
  (a + b + c) / 3 > (a * b * c) ^ (1 / 3) ∧ (a * b * c) ^ (1 / 3) > 3 * a * b * c / (a * b + b * c + c * a) :=
by
  sorry

end am_gm_hm_inequality_l69_69822


namespace frosting_cupcakes_l69_69992

theorem frosting_cupcakes (R_Cagney R_Lacey R_Jamie : ℕ)
  (H1 : R_Cagney = 1 / 20)
  (H2 : R_Lacey = 1 / 30)
  (H3 : R_Jamie = 1 / 40)
  (TotalTime : ℕ)
  (H4 : TotalTime = 600) :
  (R_Cagney + R_Lacey + R_Jamie) * TotalTime = 65 :=
by
  sorry

end frosting_cupcakes_l69_69992


namespace geometric_sequence_sum_condition_l69_69639

theorem geometric_sequence_sum_condition
  (a_1 r : ℝ) 
  (S₄ : ℝ := a_1 * (1 + r + r^2 + r^3)) 
  (S₈ : ℝ := S₄ + a_1 * (r^4 + r^5 + r^6 + r^7)) 
  (h₁ : S₄ = 1) 
  (h₂ : S₈ = 3) :
  a_1 * r^16 * (1 + r + r^2 + r^3) = 8 := 
sorry

end geometric_sequence_sum_condition_l69_69639


namespace maryann_rescue_time_l69_69687

def time_to_free_cheaph (minutes : ℕ) : ℕ := 6
def time_to_free_expenh (minutes : ℕ) : ℕ := 8
def num_friends : ℕ := 3

theorem maryann_rescue_time : (time_to_free_cheaph 6 + time_to_free_expenh 8) * num_friends = 42 := 
by
  sorry

end maryann_rescue_time_l69_69687


namespace kite_area_eq_twenty_l69_69323

theorem kite_area_eq_twenty :
  let base := 10
  let height := 2
  let area_of_triangle := (1 / 2 : ℝ) * base * height
  let total_area := 2 * area_of_triangle
  total_area = 20 :=
by
  sorry

end kite_area_eq_twenty_l69_69323


namespace last_number_l69_69500

theorem last_number (A B C D E F G : ℕ)
  (h1 : A + B + C + D = 52)
  (h2 : D + E + F + G = 60)
  (h3 : E + F + G = 55)
  (h4 : D^2 = G) : G = 25 :=
by
  sorry

end last_number_l69_69500


namespace matt_climbing_speed_l69_69388

theorem matt_climbing_speed :
  ∃ (x : ℝ), (12 * 7 = 7 * x + 42) ∧ x = 6 :=
by {
  sorry
}

end matt_climbing_speed_l69_69388


namespace friendships_structure_count_l69_69697

/-- In a group of 8 individuals, where each person has exactly 3 friends within the group,
there are 420 different ways to structure these friendships. -/
theorem friendships_structure_count : 
  ∃ (structure_count : ℕ), 
    structure_count = 420 ∧ 
    (∀ (G : Fin 8 → Fin 8 → Prop), 
      (∀ i, ∃! (j₁ j₂ j₃ : Fin 8), G i j₁ ∧ G i j₂ ∧ G i j₃) ∧ 
      (∀ i j, G i j → G j i) ∧ 
      (structure_count = 420)) := 
by
  sorry

end friendships_structure_count_l69_69697


namespace distance_between_points_l69_69722

theorem distance_between_points :
  let A : ℝ × ℝ × ℝ := (1, -2, 3)
  let B : ℝ × ℝ × ℝ := (1, 2, 3)
  let C : ℝ × ℝ × ℝ := (1, 2, -3)
  dist B C = 6 :=
by
  sorry

end distance_between_points_l69_69722


namespace wall_print_costs_are_15_l69_69905

-- Define the cost of curtains, installation, total cost, and number of wall prints.
variable (cost_curtain : ℕ := 30)
variable (num_curtains : ℕ := 2)
variable (cost_installation : ℕ := 50)
variable (num_wall_prints : ℕ := 9)
variable (total_cost : ℕ := 245)

-- Define the total cost of curtains
def total_cost_curtains : ℕ := num_curtains * cost_curtain

-- Define the total fixed costs
def total_fixed_costs : ℕ := total_cost_curtains + cost_installation

-- Define the total cost of wall prints
def total_cost_wall_prints : ℕ := total_cost - total_fixed_costs

-- Define the cost per wall print
def cost_per_wall_print : ℕ := total_cost_wall_prints / num_wall_prints

-- Prove the cost per wall print is $15.00
theorem wall_print_costs_are_15 : cost_per_wall_print = 15 := by
  -- This is a placeholder for the proof
  sorry

end wall_print_costs_are_15_l69_69905


namespace necessary_and_sufficient_conditions_l69_69141

-- Definitions for sets A and B
def U : Set (ℝ × ℝ) := {p | true}

def A (m : ℝ) : Set (ℝ × ℝ) := {p | 2 * p.1 - p.2 + m > 0}

def B (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n ≤ 0}

-- Given point P(2, 3)
def P : ℝ × ℝ := (2, 3)

-- Complement of B
def B_complement (n : ℝ) : Set (ℝ × ℝ) := {p | p.1 + p.2 - n > 0}

-- Intersection of A and complement of B
def A_inter_B_complement (m n : ℝ) : Set (ℝ × ℝ) := A m ∩ B_complement n

-- Theorem stating the necessary and sufficient conditions for P to belong to A ∩ (complement of B)
theorem necessary_and_sufficient_conditions (m n : ℝ) : 
  P ∈ A_inter_B_complement m n ↔ m > -1 ∧ n < 5 :=
sorry

end necessary_and_sufficient_conditions_l69_69141


namespace greatest_multiple_of_5_and_6_less_than_800_l69_69207

theorem greatest_multiple_of_5_and_6_less_than_800 : 
  ∃ n : ℕ, n < 800 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 800 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
sorry

end greatest_multiple_of_5_and_6_less_than_800_l69_69207


namespace max_distance_covered_l69_69459

theorem max_distance_covered 
  (D : ℝ)
  (h1 : (D / 2) / 5 + (D / 2) / 4 = 6) : 
  D = 40 / 3 :=
by
  sorry

end max_distance_covered_l69_69459


namespace probability_of_blue_candy_l69_69374

theorem probability_of_blue_candy (green blue red : ℕ) (h1 : green = 5) (h2 : blue = 3) (h3 : red = 4) :
  (blue : ℚ) / (green + blue + red : ℚ) = 1 / 4 :=
by
  rw [h1, h2, h3]
  norm_num


end probability_of_blue_candy_l69_69374


namespace students_know_mothers_birthday_l69_69793

-- Defining the given conditions
def total_students : ℕ := 40
def A : ℕ := 10
def B : ℕ := 12
def C : ℕ := 22
def D : ℕ := 26

-- Statement to prove
theorem students_know_mothers_birthday : (B + C) = 22 :=
by
  sorry

end students_know_mothers_birthday_l69_69793


namespace no_infinite_set_exists_l69_69245

variable {S : Set ℕ} -- We assume S is a set of natural numbers

def satisfies_divisibility_condition (a b : ℕ) : Prop :=
  (a^2 + b^2 - a * b) ∣ (a * b)^2

theorem no_infinite_set_exists (h1 : Infinite S)
  (h2 : ∀ (a b : ℕ), a ∈ S → b ∈ S → satisfies_divisibility_condition a b) : false :=
  sorry

end no_infinite_set_exists_l69_69245


namespace height_difference_percentage_l69_69225

theorem height_difference_percentage (H_A H_B : ℝ) (h : H_B = H_A * 1.8181818181818183) :
  (H_A < H_B) → ((H_B - H_A) / H_B) * 100 = 45 := 
by 
  sorry

end height_difference_percentage_l69_69225


namespace conor_total_vegetables_weekly_l69_69976

def conor_vegetables_daily (e c p o z : ℕ) : ℕ :=
  e + c + p + o + z

def conor_vegetables_weekly (vegetables_daily days_worked : ℕ) : ℕ :=
  vegetables_daily * days_worked

theorem conor_total_vegetables_weekly :
  conor_vegetables_weekly (conor_vegetables_daily 12 9 8 15 7) 6 = 306 := by
  sorry

end conor_total_vegetables_weekly_l69_69976


namespace abc_inequality_l69_69186

theorem abc_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a = 1) :
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) ≥ (a * b + b * c + c * a)^2 :=
sorry

end abc_inequality_l69_69186


namespace parabola_focus_coordinates_l69_69529

theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  ∃ x y : ℝ, y = 4 * a * x^2 → (x, y) = (0, 1 / (16 * a)) :=
by
  sorry

end parabola_focus_coordinates_l69_69529


namespace curve_line_and_circle_l69_69449

theorem curve_line_and_circle : 
  ∀ x y : ℝ, (x^3 + x * y^2 = 2 * x) ↔ (x = 0 ∨ x^2 + y^2 = 2) :=
by
  sorry

end curve_line_and_circle_l69_69449


namespace mask_donation_equation_l69_69438

theorem mask_donation_equation (x : ℝ) : 
  1 + (1 + x) + (1 + x)^2 = 4.75 :=
sorry

end mask_donation_equation_l69_69438


namespace distance_from_circle_to_line_l69_69763

def polar_circle (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def polar_line (θ : ℝ) : Prop := θ = Real.pi / 6

theorem distance_from_circle_to_line : 
  ∃ d : ℝ, polar_circle ρ θ ∧ polar_line θ → d = Real.sqrt 3 := 
by
  sorry

end distance_from_circle_to_line_l69_69763


namespace solve_system_l69_69509

noncomputable def f (a b x : ℝ) : ℝ := a^x + b

theorem solve_system (a b : ℝ) :
  (f a b 1 = 4) ∧ (f a b 0 = 2) →
  a = 3 ∧ b = 1 :=
by
  sorry

end solve_system_l69_69509


namespace extrema_of_function_l69_69526

noncomputable def f (x : ℝ) := x / 8 + 2 / x

theorem extrema_of_function : 
  ∀ x ∈ Set.Ioo (-5 : ℝ) (10),
  (x ≠ 0) →
  (f (-4) = -1 ∧ f 4 = 1) ∧
  (∀ x ∈ Set.Ioc (-5) 0, f x ≤ -1) ∧
  (∀ x ∈ Set.Ioo 0 10, f x ≥ 1) := by
  sorry

end extrema_of_function_l69_69526


namespace value_of_expression_l69_69816

theorem value_of_expression (x y z : ℝ) (hz : z ≠ 0) 
    (h1 : 2 * x - 3 * y - z = 0) 
    (h2 : x + 3 * y - 14 * z = 0) : 
    (x^2 + 3 * x * y) / (y^2 + z^2) = 7 := 
by 
  sorry

end value_of_expression_l69_69816


namespace S_11_eq_zero_l69_69751

noncomputable def S (n : ℕ) : ℝ := sorry
variable (a_n : ℕ → ℝ) (d : ℝ)
variable (h1 : ∀ n, a_n (n+1) = a_n n + d) -- common difference d ≠ 0
variable (h2 : S 5 = S 6)

theorem S_11_eq_zero (h_nonzero : d ≠ 0) : S 11 = 0 := by
  sorry

end S_11_eq_zero_l69_69751


namespace maria_anna_ages_l69_69193

theorem maria_anna_ages : 
  ∃ (x y : ℝ), x + y = 44 ∧ x = 2 * (y - (- (1/2) * x + (3/2) * ((2/3) * y))) ∧ x = 27.5 ∧ y = 16.5 := by 
  sorry

end maria_anna_ages_l69_69193


namespace find_AX_l69_69082

variable (A B X C : Point)
variable (AB AC BC AX XB : ℝ)
variable (angleACX angleXCB : Angle)
variable (eqAngle : angleACX = angleXCB)

axiom length_AB : AB = 80
axiom length_AC : AC = 36
axiom length_BC : BC = 72

theorem find_AX (AB AC BC AX XB : ℝ) (angleACX angleXCB : Angle)
  (eqAngle : angleACX = angleXCB)
  (h1 : AB = 80)
  (h2 : AC = 36)
  (h3 : BC = 72) : AX = 80 / 3 :=
by
  sorry

end find_AX_l69_69082


namespace midpoint_quadrilateral_area_l69_69233

theorem midpoint_quadrilateral_area (R : ℝ) (hR : 0 < R) :
  ∃ (Q : ℝ), Q = R / 4 :=
by
  sorry

end midpoint_quadrilateral_area_l69_69233


namespace area_of_square_l69_69555

-- Define the problem setting and the conditions
def square (side_length : ℝ) : Prop :=
  ∃ (width height : ℝ), width * height = side_length^2
    ∧ width = 5
    ∧ side_length / height = 5 / height

-- State the theorem to be proven
theorem area_of_square (side_length : ℝ) (width height : ℝ) (h1 : width = 5) (h2: side_length = 5 + 2 * height): 
  square side_length → side_length^2 = 400 :=
by
  intro h
  sorry

end area_of_square_l69_69555


namespace oblique_projection_correctness_l69_69908

structure ProjectionConditions where
  intuitive_diagram_of_triangle_is_triangle : Prop
  intuitive_diagram_of_parallelogram_is_parallelogram : Prop

theorem oblique_projection_correctness (c : ProjectionConditions)
  (h1 : c.intuitive_diagram_of_triangle_is_triangle)
  (h2 : c.intuitive_diagram_of_parallelogram_is_parallelogram) :
  c.intuitive_diagram_of_triangle_is_triangle ∧ c.intuitive_diagram_of_parallelogram_is_parallelogram :=
by
  sorry

end oblique_projection_correctness_l69_69908


namespace conic_section_type_l69_69792

theorem conic_section_type (x y : ℝ) : 
  9 * x^2 - 36 * y^2 = 36 → 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1) :=
by
  sorry

end conic_section_type_l69_69792


namespace number_of_sets_l69_69830

theorem number_of_sets (A : Set ℕ) : ∃ s : Finset (Set ℕ), 
  (∀ x ∈ s, ({1} ⊂ x ∧ x ⊆ {1, 2, 3, 4})) ∧ s.card = 7 :=
sorry

end number_of_sets_l69_69830


namespace hyperbola_eccentricity_l69_69984

theorem hyperbola_eccentricity (a b : ℝ) (h_asymptote : a = 3 * b) : 
    (a^2 + b^2) / a^2 = 10 / 9 := 
by
    sorry

end hyperbola_eccentricity_l69_69984


namespace total_leftover_tarts_l69_69468

variable (cherry_tart blueberry_tart peach_tart : ℝ)
variable (h1 : cherry_tart = 0.08)
variable (h2 : blueberry_tart = 0.75)
variable (h3 : peach_tart = 0.08)

theorem total_leftover_tarts : 
  cherry_tart + blueberry_tart + peach_tart = 0.91 := 
by 
  sorry

end total_leftover_tarts_l69_69468


namespace relationship_between_a_and_b_l69_69481

theorem relationship_between_a_and_b (a b : ℝ) (h₀ : a ≠ 0) (max_point : ∃ x, (x = 0 ∨ x = 1/3) ∧ (∀ y, (y = 0 ∨ y = 1/3) → (3 * a * y^2 + 2 * b * y) = 0)) : a + 2 * b = 0 :=
sorry

end relationship_between_a_and_b_l69_69481


namespace required_decrease_l69_69530

noncomputable def price_after_increases (P : ℝ) : ℝ :=
  let P1 := 1.20 * P
  let P2 := 1.10 * P1
  1.15 * P2

noncomputable def price_after_discount (P : ℝ) : ℝ :=
  0.95 * price_after_increases P

noncomputable def price_after_tax (P : ℝ) : ℝ :=
  1.07 * price_after_discount P

theorem required_decrease (P : ℝ) (D : ℝ) : 
  (1 - D / 100) * price_after_tax P = P ↔ D = 35.1852 :=
by
  sorry

end required_decrease_l69_69530


namespace not_p_is_necessary_but_not_sufficient_l69_69220

-- Definitions based on the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = d

def not_p (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ n : ℕ, a (n + 2) - a (n + 1) ≠ d

def not_q (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ¬ is_arithmetic_sequence a d

-- Proof problem statement
theorem not_p_is_necessary_but_not_sufficient (d : ℝ) (a : ℕ → ℝ) :
  (not_p a d → not_q a d) ∧ (not_q a d → not_p a d) = False := 
sorry

end not_p_is_necessary_but_not_sufficient_l69_69220


namespace arithmetic_sequence_sum_l69_69519

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) (n : ℕ)
  (h₁ : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h₂ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₃ : 3 * a 5 - a 1 = 10) :
  S 13 = 117 := 
sorry

end arithmetic_sequence_sum_l69_69519


namespace bowling_ball_weight_l69_69814

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 5 * b = 4 * c) 
  (h2 : 2 * c = 80) : 
  b = 32 :=
by
  sorry

end bowling_ball_weight_l69_69814


namespace quadratic_real_roots_l69_69745

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots (k : ℝ) :
  discriminant (k - 1) 4 2 ≥ 0 ↔ k ≤ 3 ∧ k ≠ 1 :=
by
  sorry

end quadratic_real_roots_l69_69745


namespace total_pie_eaten_l69_69482

theorem total_pie_eaten (s1 s2 s3 : ℚ) (h1 : s1 = 8/9) (h2 : s2 = 5/6) (h3 : s3 = 2/3) :
  s1 + s2 + s3 = 43/18 := by
  sorry

end total_pie_eaten_l69_69482


namespace no_snuggly_numbers_l69_69125

def isSnuggly (n : Nat) : Prop :=
  ∃ (a b : Nat), 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    n = 10 * a + b ∧ 
    n = a + b^3 + 5

theorem no_snuggly_numbers : 
  ¬ ∃ n : Nat, 10 ≤ n ∧ n < 100 ∧ isSnuggly n :=
by
  sorry

end no_snuggly_numbers_l69_69125


namespace number_of_elderly_employees_in_sample_l69_69863

variables (total_employees young_employees sample_young_employees elderly_employees : ℕ)
variables (sample_total : ℕ)

def conditions (total_employees young_employees sample_young_employees elderly_employees : ℕ) :=
  total_employees = 430 ∧
  young_employees = 160 ∧
  sample_young_employees = 32 ∧
  (∃ M, M = 2 * elderly_employees ∧ elderly_employees + M + young_employees = total_employees)

theorem number_of_elderly_employees_in_sample
  (total_employees young_employees sample_young_employees elderly_employees : ℕ)
  (sample_total : ℕ) :
  conditions total_employees young_employees sample_young_employees elderly_employees →
  sample_total = 430 * 32 / 160 →
  sample_total = 90 * 32 / 430 :=
by
  sorry

end number_of_elderly_employees_in_sample_l69_69863


namespace flu_infection_equation_l69_69804

theorem flu_infection_equation (x : ℕ) (h : 1 + x + x^2 = 36) : 1 + x + x^2 = 36 :=
by
  sorry

end flu_infection_equation_l69_69804


namespace total_recruits_211_l69_69116

theorem total_recruits_211 (P N D : ℕ) (total : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170) 
  (h4 : ∃ (x y : ℕ), (x = 4 * y ∨ y = 4 * x) ∧ 
                      ((x, P) = (y, N) ∨ (x, N) = (y, D) ∨ (x, P) = (y, D))) :
  total = 211 :=
by
  sorry

end total_recruits_211_l69_69116


namespace ratio_of_sides_l69_69399

theorem ratio_of_sides (perimeter_pentagon perimeter_square : ℝ) (hp : perimeter_pentagon = 20) (hs : perimeter_square = 20) : (4:ℝ) / (5:ℝ) = (4:ℝ) / (5:ℝ) :=
by
  sorry

end ratio_of_sides_l69_69399


namespace cube_bug_probability_l69_69791

theorem cube_bug_probability :
  ∃ n : ℕ, (∃ p : ℚ, p = 547/2187) ∧ (p = n/6561) ∧ n = 1641 :=
by
  sorry

end cube_bug_probability_l69_69791


namespace soft_drink_company_bottle_count_l69_69963

theorem soft_drink_company_bottle_count
  (B : ℕ)
  (initial_small_bottles : ℕ := 6000)
  (percent_sold_small : ℝ := 0.12)
  (percent_sold_big : ℝ := 0.14)
  (bottles_remaining_total : ℕ := 18180) :
  (initial_small_bottles * (1 - percent_sold_small) + B * (1 - percent_sold_big) = bottles_remaining_total) → B = 15000 :=
by
  sorry

end soft_drink_company_bottle_count_l69_69963


namespace complex_expression_value_l69_69278

theorem complex_expression_value :
  (i^3 * (1 + i)^2 = 2) :=
by
  sorry

end complex_expression_value_l69_69278


namespace cost_of_hard_lenses_l69_69673

theorem cost_of_hard_lenses (x H : ℕ) (h1 : x + (x + 5) = 11)
    (h2 : 150 * (x + 5) + H * x = 1455) : H = 85 := by
  sorry

end cost_of_hard_lenses_l69_69673


namespace find_total_amount_l69_69054

variables (A B C : ℕ) (total_amount : ℕ) 

-- Conditions
def condition1 : Prop := B = 36
def condition2 : Prop := 100 * B / 45 = A
def condition3 : Prop := 100 * C / 30 = A

-- Proof statement
theorem find_total_amount (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 A C) :
  total_amount = 300 :=
sorry

end find_total_amount_l69_69054


namespace nontrivial_power_of_nat_l69_69812

theorem nontrivial_power_of_nat (n : ℕ) :
  (∃ A p : ℕ, 2^n + 1 = A^p ∧ p > 1) → n = 3 :=
by
  sorry

end nontrivial_power_of_nat_l69_69812


namespace total_carrots_l69_69025

theorem total_carrots (sally_carrots fred_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : fred_carrots = 4) : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l69_69025


namespace perpendicular_vectors_l69_69766

/-- Given vectors a and b which are perpendicular, find the value of m -/
theorem perpendicular_vectors (m : ℝ) (a b : ℝ × ℝ)
  (h1 : a = (2 * m, 1))
  (h2 : b = (1, m - 3))
  (h3 : (a.1 * b.1 + a.2 * b.2) = 0) : m = 1 :=
by
  sorry

end perpendicular_vectors_l69_69766


namespace integer_pairs_solution_l69_69403

def is_satisfied_solution (x y : ℤ) : Prop :=
  x^2 + y^2 = x + y + 2

theorem integer_pairs_solution :
  ∀ (x y : ℤ), is_satisfied_solution x y ↔ (x, y) = (-1, 0) ∨ (x, y) = (-1, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (0, 2) ∨ (x, y) = (1, -1) ∨ (x, y) = (1, 2) ∨ (x, y) = (2, 0) ∨ (x, y) = (2, 1) :=
by
  sorry

end integer_pairs_solution_l69_69403


namespace remainder_when_sum_divided_by_11_l69_69686

def sum_of_large_numbers : ℕ :=
  100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007

theorem remainder_when_sum_divided_by_11 : sum_of_large_numbers % 11 = 2 := by
  sorry

end remainder_when_sum_divided_by_11_l69_69686


namespace total_number_of_sweets_l69_69126

theorem total_number_of_sweets (num_crates : ℕ) (sweets_per_crate : ℕ) (total_sweets : ℕ) 
  (h1 : num_crates = 4) (h2 : sweets_per_crate = 16) : total_sweets = 64 := by
  sorry

end total_number_of_sweets_l69_69126


namespace A_not_losing_prob_correct_l69_69691

def probability_draw : ℚ := 1 / 2
def probability_A_wins : ℚ := 1 / 3
def probability_A_not_losing : ℚ := 5 / 6

theorem A_not_losing_prob_correct : 
  probability_draw + probability_A_wins = probability_A_not_losing := 
by sorry

end A_not_losing_prob_correct_l69_69691


namespace clubsuit_commute_l69_69483

-- Define the operation a ♣ b = a^3 * b - a * b^3
def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

-- Define the proposition to prove
theorem clubsuit_commute (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end clubsuit_commute_l69_69483


namespace largest_area_of_triangle_DEF_l69_69896

noncomputable def maxAreaTriangleDEF : Real :=
  let DE := 16.0
  let EF_to_FD := 25.0 / 24.0
  let max_area := 446.25
  max_area

theorem largest_area_of_triangle_DEF :
  ∀ (DE : Real) (EF FD : Real),
    DE = 16 ∧ EF / FD = 25 / 24 → 
    (∃ (area : Real), area ≤ maxAreaTriangleDEF) :=
by 
  sorry

end largest_area_of_triangle_DEF_l69_69896


namespace parameterized_line_solution_l69_69935

theorem parameterized_line_solution :
  ∃ s l : ℝ, s = 1 / 2 ∧ l = -10 ∧
    ∀ t : ℝ, ∃ x y : ℝ,
      (x = -7 + t * l → y = s + t * (-5)) ∧ (y = (1 / 2) * x + 4) :=
by
  sorry

end parameterized_line_solution_l69_69935


namespace distance_between_lines_l69_69181

noncomputable def distance_between_parallel_lines
  (a b m n : ℝ) : ℝ :=
  |m - n| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines
  (a b m n : ℝ) :
  distance_between_parallel_lines a b m n = 
  |m - n| / Real.sqrt (a^2 + b^2) :=
by
  sorry

end distance_between_lines_l69_69181


namespace solve_proportion_l69_69415

noncomputable def x : ℝ := 0.6

theorem solve_proportion (x : ℝ) (h : 0.75 / x = 10 / 8) : x = 0.6 :=
by
  sorry

end solve_proportion_l69_69415


namespace factor_expression_l69_69829

theorem factor_expression (x : ℤ) : 84 * x^7 - 270 * x^13 = 6 * x^7 * (14 - 45 * x^6) := 
by sorry

end factor_expression_l69_69829


namespace difference_of_interchanged_digits_l69_69337

theorem difference_of_interchanged_digits (X Y : ℕ) (h1 : X - Y = 3) :
  (10 * X + Y) - (10 * Y + X) = 27 := by
  sorry

end difference_of_interchanged_digits_l69_69337


namespace value_of_six_inch_cube_l69_69538

theorem value_of_six_inch_cube :
  let four_inch_cube_value := 400
  let four_inch_side_length := 4
  let six_inch_side_length := 6
  let volume (s : ℕ) : ℕ := s ^ 3
  (volume six_inch_side_length / volume four_inch_side_length) * four_inch_cube_value = 1350 := by
sorry

end value_of_six_inch_cube_l69_69538


namespace BANANA_distinct_arrangements_l69_69049

theorem BANANA_distinct_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 1) * (Nat.factorial 3) * (Nat.factorial 2)) = 60 := 
by
  sorry

end BANANA_distinct_arrangements_l69_69049


namespace greater_number_is_64_l69_69598

-- Proof statement: The greater number (y) is 64 given the conditions
theorem greater_number_is_64 (x y : ℕ) 
    (h1 : y = 2 * x) 
    (h2 : x + y = 96) : 
    y = 64 := 
sorry

end greater_number_is_64_l69_69598


namespace dave_used_tickets_for_toys_l69_69668

-- Define the given conditions
def number_of_tickets_won : ℕ := 18
def tickets_more_for_clothes : ℕ := 10

-- Define the main conjecture
theorem dave_used_tickets_for_toys (T : ℕ) : T + (T + tickets_more_for_clothes) = number_of_tickets_won → T = 4 :=
by {
  -- We'll need the proof here, but it's not required for the statement purpose.
  sorry
}

end dave_used_tickets_for_toys_l69_69668


namespace repair_cost_l69_69650

theorem repair_cost (purchase_price transport_cost sale_price : ℝ) (profit_percentage : ℝ) (repair_cost : ℝ) :
  purchase_price = 14000 →
  transport_cost = 1000 →
  sale_price = 30000 →
  profit_percentage = 50 →
  sale_price = (1 + profit_percentage / 100) * (purchase_price + repair_cost + transport_cost) →
  repair_cost = 5000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end repair_cost_l69_69650


namespace arithmetic_sequence_a4_a5_sum_l69_69151

theorem arithmetic_sequence_a4_a5_sum
  (a_n : ℕ → ℝ)
  (a1_a2_sum : a_n 1 + a_n 2 = -1)
  (a3_val : a_n 3 = 4)
  (h_arith : ∃ d : ℝ, ∀ (n : ℕ), a_n (n + 1) = a_n n + d) :
  a_n 4 + a_n 5 = 17 := 
by
  sorry

end arithmetic_sequence_a4_a5_sum_l69_69151


namespace sales_tax_difference_l69_69441

-- Definitions for the conditions
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.075
def tax_rate2 : ℝ := 0.05

-- Calculations based on the conditions
def tax1 := item_price * tax_rate1
def tax2 := item_price * tax_rate2

-- The proof statement
theorem sales_tax_difference :
  tax1 - tax2 = 1.25 :=
by
  sorry

end sales_tax_difference_l69_69441


namespace inverse_proportion_relation_l69_69670

variable (k : ℝ) (y1 y2 : ℝ) (h1 : y1 = - (2 / (-1))) (h2 : y2 = - (2 / (-2)))

theorem inverse_proportion_relation : y1 > y2 := by
  sorry

end inverse_proportion_relation_l69_69670


namespace problem_statement_l69_69811

noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

noncomputable def a : ℝ :=
1 / Real.logb (1 / 4) (1 / 2015) + 1 / Real.logb (1 / 504) (1 / 2015)

def b : ℝ := 2017

theorem problem_statement :
  (a + b + (a - b) * sgn (a - b)) / 2 = 2017 :=
sorry

end problem_statement_l69_69811


namespace height_of_frustum_l69_69566

-- Definitions based on the given conditions
def cuts_parallel_to_base (height: ℕ) (ratio: ℕ) : ℕ := 
  height * ratio

-- Define the problem
theorem height_of_frustum 
  (height_smaller_pyramid : ℕ) 
  (ratio_upper_to_lower: ℕ) 
  (h : height_smaller_pyramid = 3) 
  (r : ratio_upper_to_lower = 4) 
  : (cuts_parallel_to_base 3 2) - height_smaller_pyramid = 3 := 
by
  sorry

end height_of_frustum_l69_69566


namespace find_value_of_y_l69_69952

theorem find_value_of_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = 7) : y = 52 :=
by
  sorry

end find_value_of_y_l69_69952


namespace sum_even_sub_sum_odd_l69_69272

def sum_arith_seq (a1 an d : ℕ) (n : ℕ) : ℕ :=
  n * (a1 + an) / 2

theorem sum_even_sub_sum_odd :
  let n_even := 50
  let n_odd := 15
  let s_even := sum_arith_seq 2 100 2 n_even
  let s_odd := sum_arith_seq 1 29 2 n_odd
  s_even - s_odd = 2325 :=
by
  sorry

end sum_even_sub_sum_odd_l69_69272


namespace max_distance_between_bus_stops_l69_69942

theorem max_distance_between_bus_stops 
  (v_m : ℝ) (v_b : ℝ) (dist : ℝ) 
  (h1 : v_m = v_b / 3) (h2 : dist = 2) : 
  ∀ d : ℝ, d = 1.5 := sorry

end max_distance_between_bus_stops_l69_69942


namespace noncongruent_triangles_count_l69_69689

/-- Prove the number of noncongruent integer-sided triangles with positive area,
    perimeter less than 20, that are neither equilateral, isosceles, nor right triangles
    is 17 -/
theorem noncongruent_triangles_count:
  ∃ (s : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ s → a + b + c < 20 ∧ a + b > c ∧ a < b ∧ b < c ∧ 
         ¬(a = b ∨ b = c ∨ a = c) ∧ ¬(a * a + b * b = c * c)) ∧ 
    s.card = 17 := 
sorry

end noncongruent_triangles_count_l69_69689


namespace m_range_decrease_y_l69_69938

theorem m_range_decrease_y {m : ℝ} : (∀ x1 x2 : ℝ, x1 < x2 → (2 * m + 2) * x1 + 5 > (2 * m + 2) * x2 + 5) ↔ m < -1 :=
by
  sorry

end m_range_decrease_y_l69_69938


namespace positive_difference_l69_69805

theorem positive_difference (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 3 * y - 4 * x = 9) : 
  abs (y - x) = 129 / 7 - (30 - 129 / 7) := 
by {
  sorry
}

end positive_difference_l69_69805


namespace find_5a_plus_5b_l69_69788

noncomputable def g (x : ℝ) : ℝ := 5 * x - 4
noncomputable def f (a b x : ℝ) : ℝ := a * x + b
noncomputable def f_inv (a b x : ℝ) : ℝ := g x + 3

theorem find_5a_plus_5b (a b : ℝ) (h_inverse : ∀ x, f_inv a b (f a b x) = x) : 5 * a + 5 * b = 2 :=
by
  sorry

end find_5a_plus_5b_l69_69788


namespace congruent_triangles_implies_corresponding_sides_equal_corresponding_sides_equal_implies_congruent_triangles_not_congruent_triangles_implies_not_corresponding_sides_equal_not_corresponding_sides_equal_implies_not_congruent_triangles_four_equal_sides_implies_is_square_is_square_implies_four_equal_sides_not_four_equal_sides_implies_not_is_square_not_is_square_implies_not_four_equal_sides_l69_69277

namespace GeometricPropositions

-- Definitions for congruence in triangles and quadrilaterals:
def congruent_triangles (Δ1 Δ2 : Type) : Prop := sorry
def corresponding_sides_equal (Δ1 Δ2 : Type) : Prop := sorry

def four_equal_sides (Q : Type) : Prop := sorry
def is_square (Q : Type) : Prop := sorry

-- Propositions and their logical forms for triangles
theorem congruent_triangles_implies_corresponding_sides_equal (Δ1 Δ2 : Type) : congruent_triangles Δ1 Δ2 → corresponding_sides_equal Δ1 Δ2 := sorry

theorem corresponding_sides_equal_implies_congruent_triangles (Δ1 Δ2 : Type) : corresponding_sides_equal Δ1 Δ2 → congruent_triangles Δ1 Δ2 := sorry

theorem not_congruent_triangles_implies_not_corresponding_sides_equal (Δ1 Δ2 : Type) : ¬ congruent_triangles Δ1 Δ2 → ¬ corresponding_sides_equal Δ1 Δ2 := sorry

theorem not_corresponding_sides_equal_implies_not_congruent_triangles (Δ1 Δ2 : Type) : ¬ corresponding_sides_equal Δ1 Δ2 → ¬ congruent_triangles Δ1 Δ2 := sorry

-- Propositions and their logical forms for quadrilaterals
theorem four_equal_sides_implies_is_square (Q : Type) : four_equal_sides Q → is_square Q := sorry

theorem is_square_implies_four_equal_sides (Q : Type) : is_square Q → four_equal_sides Q := sorry

theorem not_four_equal_sides_implies_not_is_square (Q : Type) : ¬ four_equal_sides Q → ¬ is_square Q := sorry

theorem not_is_square_implies_not_four_equal_sides (Q : Type) : ¬ is_square Q → ¬ four_equal_sides Q := sorry

end GeometricPropositions

end congruent_triangles_implies_corresponding_sides_equal_corresponding_sides_equal_implies_congruent_triangles_not_congruent_triangles_implies_not_corresponding_sides_equal_not_corresponding_sides_equal_implies_not_congruent_triangles_four_equal_sides_implies_is_square_is_square_implies_four_equal_sides_not_four_equal_sides_implies_not_is_square_not_is_square_implies_not_four_equal_sides_l69_69277


namespace divisibility_l69_69612

def Q (X : ℤ) := (X - 1) ^ 3

def P_n (n : ℕ) (X : ℤ) : ℤ :=
  n * X ^ (n + 2) - (n + 2) * X ^ (n + 1) + (n + 2) * X - n

theorem divisibility (n : ℕ) (h : n > 0) : ∀ X : ℤ, Q X ∣ P_n n X :=
by
  sorry

end divisibility_l69_69612


namespace change_making_ways_l69_69290

-- Define the conditions
def is_valid_combination (quarters nickels pennies : ℕ) : Prop :=
  quarters ≤ 2 ∧ 25 * quarters + 5 * nickels + pennies = 50

-- Define the main statement
theorem change_making_ways : 
  ∃(num_ways : ℕ), (∀(quarters nickels pennies : ℕ), is_valid_combination quarters nickels pennies → num_ways = 18) :=
sorry

end change_making_ways_l69_69290


namespace no_four_digit_number_ending_in_47_is_divisible_by_5_l69_69177

theorem no_four_digit_number_ending_in_47_is_divisible_by_5 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (n % 100 = 47 → n % 10 ≠ 0 ∧ n % 10 ≠ 5) := by
  intro n
  intro Hn
  intro H47
  sorry

end no_four_digit_number_ending_in_47_is_divisible_by_5_l69_69177


namespace largest_fraction_l69_69105

theorem largest_fraction :
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  sorry

end largest_fraction_l69_69105


namespace num_tables_l69_69440

theorem num_tables (T : ℕ) : 
  (6 * T = (17 / 3) * T) → 
  T = 6 :=
sorry

end num_tables_l69_69440


namespace correct_operation_l69_69083

variable {a b : ℝ}

theorem correct_operation : (3 * a^2 * b - 3 * b * a^2 = 0) :=
by sorry

end correct_operation_l69_69083


namespace total_sleep_per_week_l69_69877

namespace TotalSleep

def hours_sleep_wd (days: Nat) : Nat := 6 * days
def hours_sleep_wknd (days: Nat) : Nat := 10 * days

theorem total_sleep_per_week : 
  hours_sleep_wd 5 + hours_sleep_wknd 2 = 50 := by
  sorry

end TotalSleep

end total_sleep_per_week_l69_69877


namespace height_of_pole_l69_69994

-- Definitions for the conditions
def ascends_first_minute := 2
def slips_second_minute := 1
def net_ascent_per_two_minutes := ascends_first_minute - slips_second_minute
def total_minutes := 17
def pairs_of_minutes := (total_minutes - 1) / 2  -- because the 17th minute is separate
def net_ascent_first_16_minutes := pairs_of_minutes * net_ascent_per_two_minutes

-- The final ascent in the 17th minute
def ascent_final_minute := 2

-- Total ascent
def total_ascent := net_ascent_first_16_minutes + ascent_final_minute

-- Statement to prove the height of the pole
theorem height_of_pole : total_ascent = 10 :=
by
  sorry

end height_of_pole_l69_69994


namespace number_of_skirts_l69_69717

theorem number_of_skirts (T Ca Cs S : ℕ) (hT : T = 50) (hCa : Ca = 20) (hCs : Cs = 15) (hS : T - Ca = S * Cs) : S = 2 := by
  sorry

end number_of_skirts_l69_69717


namespace train_speed_calculation_l69_69080

variable (p : ℝ) (h_p : p > 0)

/-- The speed calculation of a train that covers 200 meters in p seconds is correctly given by 720 / p km/hr. -/
theorem train_speed_calculation (h_p : p > 0) : (200 / p * 3.6 = 720 / p) :=
by
  sorry

end train_speed_calculation_l69_69080


namespace complete_the_square_l69_69142

theorem complete_the_square (y : ℤ) : y^2 + 14 * y + 60 = (y + 7)^2 + 11 :=
by
  sorry

end complete_the_square_l69_69142


namespace urea_formation_l69_69507

theorem urea_formation (CO2 NH3 : ℕ) (OCN2 H2O : ℕ) (h1 : CO2 = 3) (h2 : NH3 = 6) :
  (∀ x, CO2 * 1 + NH3 * 2 = x + (2 * x) + x) →
  OCN2 = 3 :=
by
  sorry

end urea_formation_l69_69507


namespace white_marbles_count_l69_69840

section Marbles

variable (total_marbles black_marbles red_marbles green_marbles white_marbles : Nat)

theorem white_marbles_count
  (h_total: total_marbles = 60)
  (h_black: black_marbles = 32)
  (h_red: red_marbles = 10)
  (h_green: green_marbles = 5)
  (h_color: total_marbles = black_marbles + red_marbles + green_marbles + white_marbles) : 
  white_marbles = 13 := 
by
  sorry 

end Marbles

end white_marbles_count_l69_69840


namespace cos_150_eq_negative_cos_30_l69_69972

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l69_69972


namespace valid_votes_other_candidate_l69_69470

theorem valid_votes_other_candidate (total_votes : ℕ)
  (invalid_percentage valid_percentage candidate1_percentage candidate2_percentage : ℕ)
  (h_invalid_valid_sum : invalid_percentage + valid_percentage = 100)
  (h_candidates_sum : candidate1_percentage + candidate2_percentage = 100)
  (h_invalid_percentage : invalid_percentage = 20)
  (h_candidate1_percentage : candidate1_percentage = 55)
  (h_total_votes : total_votes = 7500)
  (h_valid_percentage_eq : valid_percentage = 100 - invalid_percentage)
  (h_candidate2_percentage_eq : candidate2_percentage = 100 - candidate1_percentage) :
  ( ( candidate2_percentage * ( valid_percentage * total_votes / 100) ) / 100 ) = 2700 :=
  sorry

end valid_votes_other_candidate_l69_69470


namespace compute_expression_l69_69678

-- Given condition
def condition (x : ℝ) : Prop := x + 1/x = 3

-- Theorem to prove
theorem compute_expression (x : ℝ) (hx : condition x) : (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 8 := 
by
  sorry

end compute_expression_l69_69678


namespace people_visited_neither_l69_69310

-- Definitions based on conditions
def total_people : ℕ := 60
def visited_iceland : ℕ := 35
def visited_norway : ℕ := 23
def visited_both : ℕ := 31

-- Theorem statement
theorem people_visited_neither :
  total_people - (visited_iceland + visited_norway - visited_both) = 33 :=
by sorry

end people_visited_neither_l69_69310


namespace jill_trips_to_fill_tank_l69_69484

   -- Defining the conditions
   def tank_capacity : ℕ := 600
   def bucket_capacity : ℕ := 5
   def jack_buckets_per_trip : ℕ := 2
   def jill_buckets_per_trip : ℕ := 1
   def jack_trip_rate : ℕ := 3
   def jill_trip_rate : ℕ := 2

   -- Calculate the amount of water Jack and Jill carry per trip
   def jack_gallons_per_trip : ℕ := jack_buckets_per_trip * bucket_capacity
   def jill_gallons_per_trip : ℕ := jill_buckets_per_trip * bucket_capacity

   -- Grouping the trips in the time it takes for Jill to complete her trips
   def total_gallons_per_group : ℕ := (jack_trip_rate * jack_gallons_per_trip) + (jill_trip_rate * jill_gallons_per_trip)

   -- Calculate the number of groups needed to fill the tank
   def groups_needed : ℕ := tank_capacity / total_gallons_per_group

   -- Calculate the total trips Jill makes
   def jill_total_trips : ℕ := groups_needed * jill_trip_rate

   -- The proof statement
   theorem jill_trips_to_fill_tank : jill_total_trips = 30 :=
   by
     -- Skipping the proof
     sorry
   
end jill_trips_to_fill_tank_l69_69484


namespace max_value_of_expression_l69_69553

theorem max_value_of_expression (A M C : ℕ) (hA : 0 < A) (hM : 0 < M) (hC : 0 < C) (hSum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A + A + M + C ≤ 215 :=
sorry

end max_value_of_expression_l69_69553


namespace f_continuous_on_interval_f_not_bounded_variation_l69_69454

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else x * Real.sin (1 / x)

theorem f_continuous_on_interval : ContinuousOn f (Set.Icc 0 1) :=
sorry

theorem f_not_bounded_variation : ¬ BoundedVariationOn f (Set.Icc 0 1) :=
sorry

end f_continuous_on_interval_f_not_bounded_variation_l69_69454


namespace complex_number_condition_l69_69271

theorem complex_number_condition (z : ℂ) (h : z^2 + z + 1 = 0) :
  2 * z^96 + 3 * z^97 + 4 * z^98 + 5 * z^99 + 6 * z^100 = 3 + 5 * z := 
by 
  sorry

end complex_number_condition_l69_69271


namespace minimum_trips_needed_l69_69993

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def capacity : ℕ := 200

theorem minimum_trips_needed (masses : List ℕ) (capacity : ℕ) : 
  masses = [150, 60, 70, 71, 72, 100, 101, 102, 103] →
  capacity = 200 →
  ∃ trips : ℕ, trips = 5 :=
by
  sorry

end minimum_trips_needed_l69_69993


namespace rectangle_proof_right_triangle_proof_l69_69862

-- Definition of rectangle condition
def rectangle_condition (a b : ℕ) : Prop :=
  a * b = 2 * (a + b)

-- Definition of right triangle condition
def right_triangle_condition (a b : ℕ) : Prop :=
  a + b + Int.natAbs (Int.sqrt (a^2 + b^2)) = a * b / 2 ∧
  (∃ c : ℕ, c = Int.natAbs (Int.sqrt (a^2 + b^2)))

-- Recangle proof
theorem rectangle_proof : ∃! p : ℕ × ℕ, rectangle_condition p.1 p.2 := sorry

-- Right triangle proof
theorem right_triangle_proof : ∃! t : ℕ × ℕ, right_triangle_condition t.1 t.2 := sorry

end rectangle_proof_right_triangle_proof_l69_69862


namespace louis_current_age_l69_69514

/-- 
  In 6 years, Carla will be 30 years old. 
  The sum of the current ages of Carla and Louis is 55. 
  Prove that Louis is currently 31 years old.
--/
theorem louis_current_age (C L : ℕ) 
  (h1 : C + 6 = 30) 
  (h2 : C + L = 55) 
  : L = 31 := 
sorry

end louis_current_age_l69_69514


namespace days_per_week_l69_69969

def threeChildren := 3
def schoolYearWeeks := 25
def totalJuiceBoxes := 375

theorem days_per_week (d : ℕ) :
  (threeChildren * d * schoolYearWeeks = totalJuiceBoxes) → d = 5 :=
by
  sorry

end days_per_week_l69_69969


namespace diameter_is_twice_radius_l69_69558

theorem diameter_is_twice_radius {r d : ℝ} (h : d = 2 * r) : d = 2 * r :=
by {
  sorry
}

end diameter_is_twice_radius_l69_69558


namespace employees_females_l69_69544

theorem employees_females
  (total_employees : ℕ)
  (adv_deg_employees : ℕ)
  (coll_deg_employees : ℕ)
  (males_coll_deg : ℕ)
  (females_adv_deg : ℕ)
  (females_coll_deg : ℕ)
  (h1 : total_employees = 180)
  (h2 : adv_deg_employees = 90)
  (h3 : coll_deg_employees = 180 - 90)
  (h4 : males_coll_deg = 35)
  (h5 : females_adv_deg = 55)
  (h6 : females_coll_deg = 90 - 35) :
  females_coll_deg + females_adv_deg = 110 :=
by
  sorry

end employees_females_l69_69544


namespace max_levels_passable_prob_pass_three_levels_l69_69586

-- Define the condition for passing a level
def passes_level (n : ℕ) (sum : ℕ) : Prop :=
  sum > 2^n

-- Define the maximum sum possible for n dice rolls
def max_sum (n : ℕ) : ℕ :=
  6 * n

-- Define the probability of passing the n-th level
def prob_passing_level (n : ℕ) : ℚ :=
  if n = 1 then 2/3
  else if n = 2 then 5/6
  else if n = 3 then 20/27
  else 0 

-- Combine probabilities for passing the first three levels
def prob_passing_three_levels : ℚ :=
  (2/3) * (5/6) * (20/27)

-- Theorem statement for the maximum number of levels passable
theorem max_levels_passable : 4 = 4 :=
sorry

-- Theorem statement for the probability of passing the first three levels
theorem prob_pass_three_levels : prob_passing_three_levels = 100 / 243 :=
sorry

end max_levels_passable_prob_pass_three_levels_l69_69586


namespace add_fractions_l69_69973

theorem add_fractions : (7 / 12) + (3 / 8) = 23 / 24 := by
  sorry

end add_fractions_l69_69973


namespace find_second_expression_l69_69077

theorem find_second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 84) (h₂ : a = 32) : x = 88 :=
  sorry

end find_second_expression_l69_69077


namespace dave_files_left_l69_69215

theorem dave_files_left 
  (initial_apps : ℕ) 
  (initial_files : ℕ) 
  (apps_left : ℕ)
  (files_more_than_apps : ℕ) 
  (h1 : initial_apps = 11) 
  (h2 : initial_files = 3) 
  (h3 : apps_left = 2)
  (h4 : files_more_than_apps = 22) 
  : ∃ (files_left : ℕ), files_left = apps_left + files_more_than_apps :=
by
  use 24
  sorry

end dave_files_left_l69_69215


namespace area_isosceles_right_triangle_l69_69724

theorem area_isosceles_right_triangle 
( a : ℝ × ℝ )
( b : ℝ × ℝ )
( h_a : a = (Real.cos (2 / 3 * Real.pi), Real.sin (2 / 3 * Real.pi)) )
( is_isosceles_right_triangle : (a + b).fst * (a - b).fst + (a + b).snd * (a - b).snd = 0 
                                ∧ (a + b).fst * (a + b).fst + (a + b).snd * (a + b).snd 
                                = (a - b).fst * (a - b).fst + (a - b).snd * (a - b).snd ):
  1 / 2 * Real.sqrt ((1 - 1 / 2)^2 + (Real.sqrt 3 / 2 - -1 / 2)^2 )
 * Real.sqrt ((1 - -1 / 2)^2 + (Real.sqrt 3 / 2 - -1 / 2 )^2 ) = 1 :=
by
  sorry

end area_isosceles_right_triangle_l69_69724


namespace numbers_left_on_blackboard_l69_69933

theorem numbers_left_on_blackboard (n11 n12 n13 n14 n15 : ℕ)
    (h_n11 : n11 = 11) (h_n12 : n12 = 12) (h_n13 : n13 = 13) (h_n14 : n14 = 14) (h_n15 : n15 = 15)
    (total_numbers : n11 + n12 + n13 + n14 + n15 = 65) :
  ∃ (remaining1 remaining2 : ℕ), remaining1 = 12 ∧ remaining2 = 14 := 
sorry

end numbers_left_on_blackboard_l69_69933


namespace group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l69_69949

def mats_weaved (weavers mats days : ℕ) : ℕ :=
  (mats / days) * weavers

theorem group_a_mats_in_12_days (mats_req : ℕ) :
  let weavers := 4
  let mats_per_period := 4
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_b_mats_in_12_days (mats_req : ℕ) :
  let weavers := 6
  let mats_per_period := 9
  let period_days := 3
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_c_mats_in_12_days (mats_req : ℕ) :
  let weavers := 8
  let mats_per_period := 16
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

end group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l69_69949


namespace find_x_l69_69209

theorem find_x (x : ℝ) (h : (x * (x ^ 4) ^ (1/2)) ^ (1/4) = 2) : 
  x = 16 ^ (1/3) :=
sorry

end find_x_l69_69209


namespace infinitely_many_good_primes_infinitely_many_non_good_primes_l69_69078

def is_good_prime (p : ℕ) : Prop :=
∀ a b : ℕ, a ≡ b [ZMOD p] ↔ a^3 ≡ b^3 [ZMOD p]

theorem infinitely_many_good_primes :
  ∃ᶠ p in at_top, is_good_prime p := sorry

theorem infinitely_many_non_good_primes :
  ∃ᶠ p in at_top, ¬ is_good_prime p := sorry

end infinitely_many_good_primes_infinitely_many_non_good_primes_l69_69078


namespace axel_vowels_written_l69_69430

theorem axel_vowels_written (total_alphabets number_of_vowels n : ℕ) (h1 : total_alphabets = 10) (h2 : number_of_vowels = 5) (h3 : total_alphabets = number_of_vowels * n) : n = 2 :=
by
  sorry

end axel_vowels_written_l69_69430


namespace fraction_is_one_twelve_l69_69850

variables (A E : ℝ) (f : ℝ)

-- Given conditions
def condition1 : E = 200 := sorry
def condition2 : A - E = f * (A + E) := sorry
def condition3 : A * 1.10 = E * 1.20 + 20 := sorry

-- Proving the fraction f is 1/12
theorem fraction_is_one_twelve : E = 200 → (A - E = f * (A + E)) → (A * 1.10 = E * 1.20 + 20) → 
f = 1 / 12 :=
by
  intros hE hDiff hIncrease
  sorry

end fraction_is_one_twelve_l69_69850


namespace all_suits_different_in_groups_of_four_l69_69217

-- Define the alternation pattern of the suits in the deck of 36 cards
def suits : List String := ["spades", "clubs", "hearts", "diamonds"]

-- Formalize the condition that each 4-card group in the deck contains all different suits
def suits_includes_all (cards : List String) : Prop :=
  ∀ i j, i < 4 → j < 4 → i ≠ j → cards.get? i ≠ cards.get? j

-- The main theorem statement
theorem all_suits_different_in_groups_of_four (L : List String)
  (hL : L.length = 36)
  (hA : ∀ n, n < 9 → L.get? (4*n) = some "spades" ∧ L.get? (4*n + 1) = some "clubs" ∧ L.get? (4*n + 2) = some "hearts" ∧ L.get? (4*n + 3) = some "diamonds"):
  ∀ cut reversed_deck, (@List.append String (List.reverse (List.take cut L)) (List.drop cut L) = reversed_deck)
  → ∀ n, n < 9 → suits_includes_all (List.drop (4*n) (List.take 4 reversed_deck)) := sorry

end all_suits_different_in_groups_of_four_l69_69217


namespace sum_first_20_odds_is_400_l69_69319

-- Define the n-th odd positive integer
def odd_integer (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd positive integers as a function
def sum_first_n_odds (n : ℕ) : ℕ := (n * (2 * n + 1)) / 2

-- Theorem statement: sum of the first 20 odd positive integers is 400
theorem sum_first_20_odds_is_400 : sum_first_n_odds 20 = 400 := 
  sorry

end sum_first_20_odds_is_400_l69_69319


namespace smallest_discount_l69_69109

theorem smallest_discount (n : ℕ) (h1 : (1 - 0.12) * (1 - 0.18) = 0.88 * 0.82)
  (h2 : (1 - 0.08) * (1 - 0.08) * (1 - 0.08) = 0.92 * 0.92 * 0.92)
  (h3 : (1 - 0.20) * (1 - 0.10) = 0.80 * 0.90) :
  (29 > 27.84 ∧ 29 > 22.1312 ∧ 29 > 28) :=
by {
  sorry
}

end smallest_discount_l69_69109


namespace intersection_of_A_B_l69_69866

variable (A : Set ℝ) (B : Set ℝ)

theorem intersection_of_A_B (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {x : ℝ | 0 < x ∧ x < 3}) :
  A ∩ B = {1, 2} :=
  sorry

end intersection_of_A_B_l69_69866


namespace time_after_classes_l69_69505

def time_after_maths : Nat := 60
def time_after_history : Nat := 60 + 90
def time_after_break1 : Nat := time_after_history + 25
def time_after_geography : Nat := time_after_break1 + 45
def time_after_break2 : Nat := time_after_geography + 15
def time_after_science : Nat := time_after_break2 + 75

theorem time_after_classes (start_time : Nat := 12 * 60) : (start_time + time_after_science) % 1440 = 17 * 60 + 10 :=
by
  sorry

end time_after_classes_l69_69505


namespace events_mutually_exclusive_not_complementary_l69_69941

-- Define the set of balls and people
inductive Ball : Type
| b1 | b2 | b3 | b4

inductive Person : Type
| A | B | C | D

-- Define the event types
structure Event :=
  (p : Person)
  (b : Ball)

-- Define specific events as follows
def EventA : Event := { p := Person.A, b := Ball.b1 }
def EventB : Event := { p := Person.B, b := Ball.b1 }

-- We want to prove the relationship between two specific events:
-- "Person A gets ball number 1" and "Person B gets ball number 1"
-- Namely, that they are mutually exclusive but not complementary.

theorem events_mutually_exclusive_not_complementary :
  (∀ e : Event, (e = EventA → ¬ (e = EventB)) ∧ ¬ (e = EventA ∨ e = EventB)) :=
sorry

end events_mutually_exclusive_not_complementary_l69_69941


namespace probability_all_black_after_rotation_l69_69690

-- Define the conditions
def num_unit_squares : ℕ := 16
def num_colors : ℕ := 3
def prob_per_color : ℚ := 1 / 3

-- Define the type for probabilities
def prob_black_grid : ℚ := (1 / 81) * (11 / 27) ^ 12

-- The statement to be proven
theorem probability_all_black_after_rotation :
  (prob_black_grid =
    ((1 / 3) ^ 4) * ((11 / 27) ^ 12)) :=
sorry

end probability_all_black_after_rotation_l69_69690


namespace least_number_to_add_l69_69584

theorem least_number_to_add (a b n : ℕ) (h₁ : a = 1056) (h₂ : b = 29) (h₃ : (a + n) % b = 0) : n = 17 :=
sorry

end least_number_to_add_l69_69584


namespace solve_for_x_l69_69164

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end solve_for_x_l69_69164


namespace remaining_money_l69_69737

def potato_cost : ℕ := 6 * 2
def tomato_cost : ℕ := 9 * 3
def cucumber_cost : ℕ := 5 * 4
def banana_cost : ℕ := 3 * 5
def total_cost : ℕ := potato_cost + tomato_cost + cucumber_cost + banana_cost
def initial_money : ℕ := 500

theorem remaining_money : initial_money - total_cost = 426 :=
by
  sorry

end remaining_money_l69_69737


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l69_69762

theorem problem1 : 0 - (-22) = 22 := 
by 
  sorry

theorem problem2 : 8.5 - (-1.5) = 10 := 
by 
  sorry

theorem problem3 : (-13 : ℚ) - (4/7) - (-13 : ℚ) - (5/7) = 1/7 := 
by 
  sorry

theorem problem4 : (-1/2 : ℚ) - (1/4 : ℚ) = -3/4 := 
by 
  sorry

theorem problem5 : -51 + 12 + (-7) + (-11) + 36 = -21 := 
by 
  sorry

theorem problem6 : (5/6 : ℚ) + (-2/3) + 1 + (1/6) + (-1/3) = 1 := 
by 
  sorry

theorem problem7 : -13 + (-7) - 20 - (-40) + 16 = 16 := 
by 
  sorry

theorem problem8 : 4.7 - (-8.9) - 7.5 + (-6) = 0.1 := 
by 
  sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l69_69762


namespace polynomial_no_strictly_positive_roots_l69_69613

-- Define the necessary conditions and prove the main statement

variables (n : ℕ)
variables (a : Fin n → ℕ) (k : ℕ) (M : ℕ)

-- Axioms/Conditions
axiom pos_a (i : Fin n) : 0 < a i
axiom pos_k : 0 < k
axiom pos_M : 0 < M
axiom M_gt_1 : M > 1

axiom sum_reciprocals : (Finset.univ.sum (λ i => (1 : ℚ) / a i)) = k
axiom product_a : (Finset.univ.prod a) = M

noncomputable def polynomial_has_no_positive_roots : Prop :=
  ∀ x : ℝ, 0 < x →
    M * (1 + x)^k > (Finset.univ.prod (λ i => x + a i))

theorem polynomial_no_strictly_positive_roots (h : polynomial_has_no_positive_roots n a k M) : 
  ∀ x : ℝ, 0 < x → (M * (1 + x)^k - (Finset.univ.prod (λ i => x + a i)) ≠ 0) :=
by
  sorry

end polynomial_no_strictly_positive_roots_l69_69613


namespace room_breadth_l69_69262

theorem room_breadth :
  ∀ (length breadth carpet_width cost_per_meter total_cost : ℝ),
  length = 15 →
  carpet_width = 75 / 100 →
  cost_per_meter = 30 / 100 →
  total_cost = 36 →
  total_cost = cost_per_meter * (total_cost / cost_per_meter) →
  length * breadth = (total_cost / cost_per_meter) * carpet_width →
  breadth = 6 :=
by
  intros length breadth carpet_width cost_per_meter total_cost
  intros h_length h_carpet_width h_cost_per_meter h_total_cost h_total_cost_eq h_area_eq
  sorry

end room_breadth_l69_69262


namespace oldest_daily_cheese_l69_69623

-- Given conditions
def days_per_week : ℕ := 5
def weeks : ℕ := 4
def youngest_daily : ℕ := 1
def cheeses_per_pack : ℕ := 30
def packs_needed : ℕ := 2

-- Derived conditions
def total_days : ℕ := days_per_week * weeks
def total_cheeses : ℕ := packs_needed * cheeses_per_pack
def youngest_total_cheeses : ℕ := youngest_daily * total_days
def oldest_total_cheeses : ℕ := total_cheeses - youngest_total_cheeses

-- Prove that the oldest child wants 2 string cheeses per day
theorem oldest_daily_cheese : oldest_total_cheeses / total_days = 2 := by
  sorry

end oldest_daily_cheese_l69_69623


namespace cost_of_one_pack_of_gummy_bears_l69_69121

theorem cost_of_one_pack_of_gummy_bears
    (num_chocolate_bars : ℕ)
    (num_gummy_bears : ℕ)
    (num_chocolate_chips : ℕ)
    (total_cost : ℕ)
    (cost_per_chocolate_bar : ℕ)
    (cost_per_chocolate_chip : ℕ)
    (cost_of_one_gummy_bear_pack : ℕ)
    (h1 : num_chocolate_bars = 10)
    (h2 : num_gummy_bears = 10)
    (h3 : num_chocolate_chips = 20)
    (h4 : total_cost = 150)
    (h5 : cost_per_chocolate_bar = 3)
    (h6 : cost_per_chocolate_chip = 5)
    (h7 : num_chocolate_bars * cost_per_chocolate_bar +
          num_gummy_bears * cost_of_one_gummy_bear_pack +
          num_chocolate_chips * cost_per_chocolate_chip = total_cost) :
    cost_of_one_gummy_bear_pack = 2 := by
  sorry

end cost_of_one_pack_of_gummy_bears_l69_69121


namespace find_budget_l69_69725

variable (B : ℝ)

-- Conditions provided
axiom cond1 : 0.30 * B = 300

theorem find_budget : B = 1000 :=
by
  -- Notes:
  -- The proof will go here.
  sorry

end find_budget_l69_69725


namespace area_of_triangle_l69_69166

theorem area_of_triangle (a b : ℝ) 
  (hypotenuse : ℝ) (median : ℝ)
  (h_side : hypotenuse = 2)
  (h_median : median = 1)
  (h_sum : a + b = 1 + Real.sqrt 3) 
  (h_pythagorean :(a^2 + b^2 = 4)): 
  (1/2 * a * b) = (Real.sqrt 3 / 2) := 
sorry

end area_of_triangle_l69_69166


namespace arithmetic_sequence_mod_l69_69560

theorem arithmetic_sequence_mod :
  let a := 2
  let d := 5
  let l := 137
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  n = 28 ∧ S = 1946 →
  S % 20 = 6 :=
by
  intros h
  sorry

end arithmetic_sequence_mod_l69_69560


namespace arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l69_69189

-- Statement for Question 1
theorem arrangement_ways_13_books : 
  (Nat.factorial 13) = 6227020800 := 
sorry

-- Statement for Question 2
theorem arrangement_ways_13_books_with_4_arithmetic_together :
  (Nat.factorial 10) * (Nat.factorial 4) = 87091200 := 
sorry

-- Statement for Question 3
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_together :
  (Nat.factorial 5) * (Nat.factorial 4) * (Nat.factorial 6) = 2073600 := 
sorry

-- Statement for Question 4
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together :
  (Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 6) * (Nat.factorial 3) = 622080 := 
sorry

end arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l69_69189


namespace infinitely_many_m_l69_69279

theorem infinitely_many_m (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ᶠ m in Filter.atTop, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 :=
sorry

end infinitely_many_m_l69_69279


namespace maximize_f_at_1_5_l69_69695

noncomputable def f (x: ℝ) : ℝ := -3 * x^2 + 9 * x + 5

theorem maximize_f_at_1_5 : ∀ x: ℝ, f 1.5 ≥ f x := by
  sorry

end maximize_f_at_1_5_l69_69695


namespace total_pieces_correct_l69_69079

theorem total_pieces_correct :
  let bell_peppers := 10
  let onions := 7
  let zucchinis := 15
  let bell_peppers_slices := (2 * 20)  -- 25% of 10 bell peppers sliced into 20 slices each
  let bell_peppers_large_pieces := (7 * 10)  -- Remaining 75% cut into 10 pieces each
  let bell_peppers_smaller_pieces := (35 * 3)  -- Half of large pieces cut into 3 pieces each
  let onions_slices := (3 * 18)  -- 50% of onions sliced into 18 slices each
  let onions_pieces := (4 * 8)  -- Remaining 50% cut into 8 pieces each
  let zucchinis_slices := (4 * 15)  -- 30% of zucchinis sliced into 15 pieces each
  let zucchinis_pieces := (10 * 8)  -- Remaining 70% cut into 8 pieces each
  let total_slices := bell_peppers_slices + onions_slices + zucchinis_slices
  let total_pieces := bell_peppers_large_pieces + bell_peppers_smaller_pieces + onions_pieces + zucchinis_pieces
  total_slices + total_pieces = 441 :=
by
  sorry

end total_pieces_correct_l69_69079


namespace value_of_g_at_13_l69_69959

-- Define the function g
def g (n : ℕ) : ℕ := n^2 + n + 23

-- The theorem to prove
theorem value_of_g_at_13 : g 13 = 205 := by
  -- Rewrite using the definition of g
  unfold g
  -- Perform the arithmetic
  sorry

end value_of_g_at_13_l69_69959


namespace circle_eq_problem1_circle_eq_problem2_l69_69635

-- Problem 1
theorem circle_eq_problem1 :
  (∃ a b r : ℝ, (x - a)^2 + (y - b)^2 = r^2 ∧
  a - 2 * b - 3 = 0 ∧
  (2 - a)^2 + (-3 - b)^2 = r^2 ∧
  (-2 - a)^2 + (-5 - b)^2 = r^2) ↔
  (x + 1)^2 + (y + 2)^2 = 10 :=
sorry

-- Problem 2
theorem circle_eq_problem2 :
  (∃ D E F : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ∧
  (1:ℝ)^2 + (0:ℝ)^2 + D * 1 + E * 0 + F = 0 ∧
  (-1:ℝ)^2 + (-2:ℝ)^2 - D * 1 - 2 * E + F = 0 ∧
  (3:ℝ)^2 + (-2:ℝ)^2 + 3 * D - 2 * E + F = 0) ↔
  x^2 + y^2 - 2 * x + 4 * y + 1 = 0 :=
sorry

end circle_eq_problem1_circle_eq_problem2_l69_69635


namespace cube_iff_diagonal_perpendicular_l69_69847

-- Let's define the rectangular parallelepiped as a type
structure RectParallelepiped :=
-- Define the property of being a cube
(isCube : Prop)

-- Define the property q: any diagonal of the parallelepiped is perpendicular to the diagonal of its non-intersecting face
def diagonal_perpendicular (S : RectParallelepiped) : Prop := 
 sorry -- This depends on how you define diagonals and perpendicularity within the structure

-- Prove the biconditional relationship
theorem cube_iff_diagonal_perpendicular (S : RectParallelepiped) :
 S.isCube ↔ diagonal_perpendicular S :=
sorry

end cube_iff_diagonal_perpendicular_l69_69847


namespace estimate_contestants_l69_69326

theorem estimate_contestants :
  let total_contestants := 679
  let median_all_three := 188
  let median_two_tests := 159
  let median_one_test := 169
  total_contestants = 679 ∧
  median_all_three = 188 ∧
  median_two_tests = 159 ∧
  median_one_test = 169 →
  let approx_two_tests_per_pair := median_two_tests / 3
  let intersection_pairs_approx := approx_two_tests_per_pair + median_all_three
  let number_above_or_equal_median :=
    median_one_test + median_one_test + median_one_test -
    intersection_pairs_approx - intersection_pairs_approx - intersection_pairs_approx +
    median_all_three
  number_above_or_equal_median = 516 :=
by
  intros
  sorry

end estimate_contestants_l69_69326


namespace percentage_bob_is_36_l69_69603

def water_per_acre_corn : ℕ := 20
def water_per_acre_cotton : ℕ := 80
def water_per_acre_beans : ℕ := 2 * water_per_acre_corn

def acres_bob_corn : ℕ := 3
def acres_bob_cotton : ℕ := 9
def acres_bob_beans : ℕ := 12

def acres_brenda_corn : ℕ := 6
def acres_brenda_cotton : ℕ := 7
def acres_brenda_beans : ℕ := 14

def acres_bernie_corn : ℕ := 2
def acres_bernie_cotton : ℕ := 12

def water_bob : ℕ := (acres_bob_corn * water_per_acre_corn) +
                      (acres_bob_cotton * water_per_acre_cotton) +
                      (acres_bob_beans * water_per_acre_beans)

def water_brenda : ℕ := (acres_brenda_corn * water_per_acre_corn) +
                         (acres_brenda_cotton * water_per_acre_cotton) +
                         (acres_brenda_beans * water_per_acre_beans)

def water_bernie : ℕ := (acres_bernie_corn * water_per_acre_corn) +
                         (acres_bernie_cotton * water_per_acre_cotton)

def total_water : ℕ := water_bob + water_brenda + water_bernie

def percentage_bob : ℚ := (water_bob : ℚ) / (total_water : ℚ) * 100

theorem percentage_bob_is_36 : percentage_bob = 36 := by
  sorry

end percentage_bob_is_36_l69_69603


namespace x_over_y_l69_69238

theorem x_over_y (x y : ℝ) (h : 16 * x = 0.24 * 90 * y) : x / y = 1.35 :=
sorry

end x_over_y_l69_69238


namespace cryptarithmetic_puzzle_sol_l69_69520

theorem cryptarithmetic_puzzle_sol (A B C D : ℕ) 
  (h1 : A + B + C = D) 
  (h2 : B + C = 7) 
  (h3 : A - B = 1) : D = 9 := 
by 
  sorry

end cryptarithmetic_puzzle_sol_l69_69520


namespace total_accidents_l69_69909

noncomputable def A (k x : ℕ) : ℕ := 96 + k * x

theorem total_accidents :
  let k_morning := 1
  let k_evening := 3
  let x_morning := 2000
  let x_evening := 1000
  A k_morning x_morning + A k_evening x_evening = 5192 := by
  sorry

end total_accidents_l69_69909


namespace compute_abs_difference_l69_69531

theorem compute_abs_difference (x y : ℝ) 
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 3.6)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 4.5) : 
  |x - y| = 1.1 :=
by 
  sorry

end compute_abs_difference_l69_69531


namespace arithmetic_progression_12th_term_l69_69563

theorem arithmetic_progression_12th_term (a d n : ℤ) (h_a : a = 2) (h_d : d = 8) (h_n : n = 12) :
  a + (n - 1) * d = 90 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end arithmetic_progression_12th_term_l69_69563


namespace product_eval_l69_69240

theorem product_eval (a : ℝ) (h : a = 1) : (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 :=
by
  sorry

end product_eval_l69_69240


namespace car_speed_is_80_l69_69904

theorem car_speed_is_80 
  (d : ℝ) (t_delay : ℝ) (v_train_factor : ℝ)
  (t_car t_train : ℝ) (v : ℝ) :
  ((d = 75) ∧ (t_delay = 12.5 / 60) ∧ (v_train_factor = 1.5) ∧ 
   (d = v * t_car) ∧ (d = v_train_factor * v * (t_car - t_delay))) →
  v = 80 := 
sorry

end car_speed_is_80_l69_69904


namespace perfume_weight_is_six_ounces_l69_69561

def weight_in_pounds (ounces : ℕ) : ℕ := ounces / 16

def initial_weight := 5  -- Initial suitcase weight in pounds
def final_weight := 11   -- Final suitcase weight in pounds
def chocolate := 4       -- Weight of chocolate in pounds
def soap := 2 * 5        -- Weight of 2 bars of soap in ounces
def jam := 2 * 8         -- Weight of 2 jars of jam in ounces

def total_additional_weight :=
  chocolate + (weight_in_pounds soap) + (weight_in_pounds jam)

def perfume_weight_in_pounds := final_weight - initial_weight - total_additional_weight

def perfume_weight_in_ounces := perfume_weight_in_pounds * 16

theorem perfume_weight_is_six_ounces : perfume_weight_in_ounces = 6 := by sorry

end perfume_weight_is_six_ounces_l69_69561


namespace time_to_run_round_square_field_l69_69180

theorem time_to_run_round_square_field
  (side : ℝ) (speed_km_hr : ℝ)
  (h_side : side = 45)
  (h_speed_km_hr : speed_km_hr = 9) : 
  (4 * side / (speed_km_hr * 1000 / 3600)) = 72 := 
by 
  sorry

end time_to_run_round_square_field_l69_69180


namespace average_sales_l69_69730

/-- The sales for the first five months -/
def sales_first_five_months := [5435, 5927, 5855, 6230, 5562]

/-- The sale for the sixth month -/
def sale_sixth_month := 3991

/-- The correct average sale to be achieved -/
def correct_average_sale := 5500

theorem average_sales :
  (sales_first_five_months.sum + sale_sixth_month) / 6 = correct_average_sale :=
by
  sorry

end average_sales_l69_69730


namespace conor_total_vegetables_l69_69807

-- Definitions for each day of the week
def vegetables_per_day_mon_wed : Nat := 12 + 9 + 8 + 15 + 7
def vegetables_per_day_thu_sat : Nat := 7 + 5 + 4 + 10 + 4
def total_vegetables : Nat := 3 * vegetables_per_day_mon_wed + 3 * vegetables_per_day_thu_sat

-- Lean statement for the proof problem
theorem conor_total_vegetables : total_vegetables = 243 := by
  sorry

end conor_total_vegetables_l69_69807


namespace intersection_eq_T_l69_69664

noncomputable def S : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 3 * x + 2 }
noncomputable def T : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x ^ 2 - 1 }

theorem intersection_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_eq_T_l69_69664


namespace f_lt_2_l69_69624

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f (x + 2) = f (-x + 2)

axiom f_ge_2 (x : ℝ) (h : x ≥ 2) : f x = x^2 - 6 * x + 4

theorem f_lt_2 (x : ℝ) (h : x < 2) : f x = x^2 - 2 * x - 4 :=
by
  sorry

end f_lt_2_l69_69624


namespace find_m_l69_69419

theorem find_m (m l : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, m)) (h_b : b = (l, -2))
  (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ a = k • (a + 2 • b)) :
  m = -4 :=
by
  sorry

end find_m_l69_69419


namespace notebook_pen_cost_l69_69667

theorem notebook_pen_cost :
  ∃ (n p : ℕ), 15 * n + 4 * p = 160 ∧ n > p ∧ n + p = 18 := 
sorry

end notebook_pen_cost_l69_69667


namespace scientific_notation_conversion_l69_69708

theorem scientific_notation_conversion :
  (6.1 * 10^9 = (6.1 : ℝ) * 10^8) :=
sorry

end scientific_notation_conversion_l69_69708


namespace people_in_circle_l69_69040

theorem people_in_circle (n : ℕ) (h : ∃ k : ℕ, k * 2 + 7 = 18) : n = 22 :=
by
  sorry

end people_in_circle_l69_69040


namespace intersection_eq_l69_69800

def setA : Set ℝ := {x | -1 < x ∧ x < 1}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_eq : setA ∩ setB = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_eq_l69_69800


namespace option_B_more_cost_effective_l69_69621

def cost_option_A (x : ℕ) : ℕ := 60 + 18 * x
def cost_option_B (x : ℕ) : ℕ := 150 + 15 * x
def x : ℕ := 40

theorem option_B_more_cost_effective : cost_option_B x < cost_option_A x := by
  -- Placeholder for the proof steps
  sorry

end option_B_more_cost_effective_l69_69621


namespace total_distance_swam_l69_69651

theorem total_distance_swam (molly_swam_saturday : ℕ) (molly_swam_sunday : ℕ) (h1 : molly_swam_saturday = 400) (h2 : molly_swam_sunday = 300) : molly_swam_saturday + molly_swam_sunday = 700 := by 
    sorry

end total_distance_swam_l69_69651


namespace geometric_sequence_xz_eq_three_l69_69627

theorem geometric_sequence_xz_eq_three 
  (x y z : ℝ)
  (h1 : ∃ r : ℝ, x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -3 = z * r) :
  x * z = 3 :=
by
  -- skip the proof
  sorry

end geometric_sequence_xz_eq_three_l69_69627


namespace age_of_teacher_l69_69914

theorem age_of_teacher
    (n_students : ℕ)
    (avg_age_students : ℕ)
    (new_avg_age : ℕ)
    (n_total : ℕ)
    (H1 : n_students = 22)
    (H2 : avg_age_students = 21)
    (H3 : new_avg_age = avg_age_students + 1)
    (H4 : n_total = n_students + 1) :
    ((new_avg_age * n_total) - (avg_age_students * n_students) = 44) :=
by
    sorry

end age_of_teacher_l69_69914


namespace part1_part2_l69_69967

variable (x α β : ℝ)

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sqrt 3 * (Real.cos x) ^ 2 + 2 * (Real.sin x) * (Real.cos x) - Real.sqrt 3

theorem part1 (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  -Real.sqrt 3 ≤ f x ∧ f x ≤ 2 := 
sorry

theorem part2 (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2) 
(h1 : f (α / 2 - Real.pi / 6) = 8 / 5) 
(h2 : Real.cos (α + β) = -12 / 13) : 
  Real.sin β = 63 / 65 := 
sorry

end part1_part2_l69_69967


namespace find_f_of_7_over_2_l69_69150

variable (f : ℝ → ℝ)

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f (x - 2)
axiom f_definition : ∀ x, 0 < x ∧ x < 1 → f x = 3^x

theorem find_f_of_7_over_2 : f (7 / 2) = -Real.sqrt 3 :=
by
  sorry

end find_f_of_7_over_2_l69_69150


namespace integer_solutions_xy_l69_69042

theorem integer_solutions_xy :
  ∃ (x y : ℤ), (x + y + x * y = 500) ∧ 
               ((x = 0 ∧ y = 500) ∨ 
                (x = -2 ∧ y = -502) ∨ 
                (x = 2 ∧ y = 166) ∨ 
                (x = -4 ∧ y = -168)) :=
by
  sorry

end integer_solutions_xy_l69_69042


namespace negation_of_quadratic_prop_l69_69169

theorem negation_of_quadratic_prop :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 < 0 :=
by
  sorry

end negation_of_quadratic_prop_l69_69169


namespace tangent_lines_to_curve_at_l69_69743

noncomputable
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

noncomputable
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 2) * x

noncomputable
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 2)

theorem tangent_lines_to_curve_at (a : ℝ) :
  is_even_function (f' a) →
  (∀ x, f a x = - 2 → (2*x + (- f a x) = 0 ∨ 19*x - 4*(- f a x) - 27 = 0)) :=
by
  sorry

end tangent_lines_to_curve_at_l69_69743


namespace photos_difference_is_120_l69_69541

theorem photos_difference_is_120 (initial_photos : ℕ) (final_photos : ℕ) (first_day_factor : ℕ) (first_day_photos : ℕ) (second_day_photos : ℕ) : 
  initial_photos = 400 → 
  final_photos = 920 → 
  first_day_factor = 2 →
  first_day_photos = initial_photos / first_day_factor →
  final_photos = initial_photos + first_day_photos + second_day_photos →
  second_day_photos - first_day_photos = 120 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end photos_difference_is_120_l69_69541


namespace inequality_proof_l69_69431

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hSum : a + b + c = 1)

theorem inequality_proof :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 1 / 2 :=
by
  sorry

end inequality_proof_l69_69431


namespace greatest_mass_l69_69402

theorem greatest_mass (V : ℝ) (h : ℝ) (l : ℝ) 
    (ρ_Hg ρ_H2O ρ_Oil : ℝ) 
    (V1 V2 V3 : ℝ) 
    (m_Hg m_H2O m_Oil : ℝ)
    (ρ_Hg_val : ρ_Hg = 13.59) 
    (ρ_H2O_val : ρ_H2O = 1) 
    (ρ_Oil_val : ρ_Oil = 0.915) 
    (height_layers_equal : h = l) :
    ∀ V1 V2 V3 m_Hg m_H2O m_Oil, 
    V1 + V2 + V3 = 27 * (l^3) → 
    V2 = 7 * V1 → 
    V3 = 19 * V1 → 
    m_Hg = ρ_Hg * V1 → 
    m_H2O = ρ_H2O * V2 → 
    m_Oil = ρ_Oil * V3 → 
    m_Oil > m_Hg ∧ m_Oil > m_H2O := 
by 
    intros
    sorry

end greatest_mass_l69_69402


namespace triangle_area_l69_69157

theorem triangle_area (d : ℝ) (h : d = 8 * Real.sqrt 10) (ang : ∀ {α β γ : ℝ}, α = 45 ∨ β = 45 ∨ γ = 45) :
  ∃ A : ℝ, A = 160 :=
by
  sorry

end triangle_area_l69_69157


namespace original_price_of_goods_l69_69275

theorem original_price_of_goods
  (rebate_percent : ℝ := 0.06)
  (tax_percent : ℝ := 0.10)
  (total_paid : ℝ := 6876.1) :
  ∃ P : ℝ, (P - P * rebate_percent) * (1 + tax_percent) = total_paid ∧ P = 6650 :=
sorry

end original_price_of_goods_l69_69275


namespace range_of_m_l69_69197

def f (x : ℝ) : ℝ := x^2 - 4 * x - 6

theorem range_of_m (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ m → -10 ≤ f x ∧ f x ≤ -6) →
  2 ≤ m ∧ m ≤ 4 := 
sorry

end range_of_m_l69_69197


namespace gcd_91_49_l69_69985

theorem gcd_91_49 : Nat.gcd 91 49 = 7 :=
by
  -- Using the Euclidean algorithm
  -- 91 = 49 * 1 + 42
  -- 49 = 42 * 1 + 7
  -- 42 = 7 * 6 + 0
  sorry

end gcd_91_49_l69_69985


namespace calculate_savings_l69_69769

def income : ℕ := 5 * (45000 + 35000 + 7000 + 10000 + 13000)
def expenses : ℕ := 5 * (30000 + 10000 + 5000 + 4500 + 9000)
def initial_savings : ℕ := 849400
def total_savings : ℕ := initial_savings + income - expenses

theorem calculate_savings : total_savings = 1106900 := by
  -- proof to be filled in
  sorry

end calculate_savings_l69_69769


namespace translate_line_upwards_by_3_translate_line_right_by_3_l69_69480

theorem translate_line_upwards_by_3 (x : ℝ) :
  let y := 2 * x - 4
  let y' := y + 3
  y' = 2 * x - 1 := 
by
  let y := 2 * x - 4
  let y' := y + 3
  sorry

theorem translate_line_right_by_3 (x : ℝ) :
  let y := 2 * x - 4
  let y_up := y + 3
  let y_right := 2 * (x - 3) - 4
  y_right = 2 * x - 10 :=
by
  let y := 2 * x - 4
  let y_up := y + 3
  let y_right := 2 * (x - 3) - 4
  sorry

end translate_line_upwards_by_3_translate_line_right_by_3_l69_69480


namespace bracket_mul_l69_69682

def bracket (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

theorem bracket_mul : bracket 6 * bracket 3 = 28 := by
  sorry

end bracket_mul_l69_69682


namespace sequence_sum_l69_69088

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, 0 < a n)
  → (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) 
  → (∀ n : ℕ, a (n+1)^2 = a n * a (n+2))
  → S 3 = 13
  → a 1 = 1
  → (a 3 + a 4) / (a 1 + a 2) = 9 :=
sorry

end sequence_sum_l69_69088


namespace inequality_x_y_z_l69_69961

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) + y * (1 - z) + z * (1 - x) < 1 := 
by sorry

end inequality_x_y_z_l69_69961


namespace common_difference_of_arithmetic_seq_l69_69016

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * d

theorem common_difference_of_arithmetic_seq :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  arithmetic_sequence a d →
  (a 4 + a 8 = 10) →
  (a 10 = 6) →
  d = 1 / 4 :=
by
  intros a d h_seq h1 h2
  sorry

end common_difference_of_arithmetic_seq_l69_69016


namespace inverse_is_correct_l69_69710

-- Definitions
def original_proposition (n : ℤ) : Prop := n < 0 → n ^ 2 > 0
def inverse_proposition (n : ℤ) : Prop := n ^ 2 > 0 → n < 0

-- Theorem stating the inverse
theorem inverse_is_correct : 
  (∀ n : ℤ, original_proposition n) → (∀ n : ℤ, inverse_proposition n) :=
by
  sorry

end inverse_is_correct_l69_69710


namespace white_space_area_is_31_l69_69249

-- Definitions and conditions from the problem
def board_width : ℕ := 4
def board_length : ℕ := 18
def board_area : ℕ := board_width * board_length

def area_C : ℕ := 4 + 2 + 2
def area_O : ℕ := (4 * 3) - (2 * 1)
def area_D : ℕ := (4 * 3) - (2 * 1)
def area_E : ℕ := 4 + 3 + 3 + 3

def total_black_area : ℕ := area_C + area_O + area_D + area_E

def white_space_area : ℕ := board_area - total_black_area

-- Proof problem statement
theorem white_space_area_is_31 : white_space_area = 31 := by
  sorry

end white_space_area_is_31_l69_69249


namespace minimum_choir_size_l69_69463

theorem minimum_choir_size : ∃ (choir_size : ℕ), 
  (choir_size % 9 = 0) ∧ 
  (choir_size % 11 = 0) ∧ 
  (choir_size % 13 = 0) ∧ 
  (choir_size % 10 = 0) ∧ 
  (choir_size = 12870) :=
by
  sorry

end minimum_choir_size_l69_69463


namespace k_minus_2_divisible_by_3_l69_69887

theorem k_minus_2_divisible_by_3
  (k : ℕ)
  (a : ℕ → ℤ)
  (h_a0_pos : 0 < k)
  (h_seq : ∀ n ≥ 1, a n = (a (n - 1) + n^k) / n) :
  (k - 2) % 3 = 0 :=
sorry

end k_minus_2_divisible_by_3_l69_69887


namespace sara_total_spent_l69_69194

def ticket_cost : ℝ := 10.62
def num_tickets : ℕ := 2
def rent_cost : ℝ := 1.59
def buy_cost : ℝ := 13.95
def total_spent : ℝ := 36.78

theorem sara_total_spent : (num_tickets * ticket_cost) + rent_cost + buy_cost = total_spent := by
  sorry

end sara_total_spent_l69_69194


namespace carol_betty_age_ratio_l69_69688

theorem carol_betty_age_ratio:
  ∀ (C A B : ℕ), 
    C = 5 * A → 
    A = C - 12 → 
    B = 6 → 
    C / B = 5 / 2 :=
by
  intros C A B h1 h2 h3
  sorry

end carol_betty_age_ratio_l69_69688


namespace heap_holds_20_sheets_l69_69915

theorem heap_holds_20_sheets :
  ∀ (num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets : ℕ),
    num_bundles = 3 →
    num_bunches = 2 →
    num_heaps = 5 →
    sheets_per_bundle = 2 →
    sheets_per_bunch = 4 →
    total_sheets = 114 →
    (total_sheets - (num_bundles * sheets_per_bundle + num_bunches * sheets_per_bunch)) / num_heaps = 20 := 
by
  intros num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end heap_holds_20_sheets_l69_69915


namespace inequality_proof_l69_69377

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  27 * (a^3 + b^3 + c^3) + 1 ≥ 12 * (a^2 + b^2 + c^2) :=
by
  sorry

end inequality_proof_l69_69377


namespace mean_value_of_quadrilateral_angles_l69_69092

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l69_69092


namespace meaningful_fraction_l69_69244

theorem meaningful_fraction {a : ℝ} : 2 * a - 1 ≠ 0 ↔ a ≠ 1 / 2 :=
by sorry

end meaningful_fraction_l69_69244


namespace find_radius_l69_69355

theorem find_radius :
  ∃ (r : ℝ), 
  (∀ (x : ℝ), y = x^2 + r) ∧ 
  (∀ (x : ℝ), y = x) ∧ 
  (∀ (x : ℝ), x^2 + r = x) ∧ 
  (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 0) → 
  r = 1 / 4 :=
by
  sorry

end find_radius_l69_69355


namespace socks_problem_l69_69032

/-
  Theorem: Given x + y + z = 15, 2x + 4y + 5z = 36, and x, y, z ≥ 1, 
  the number of $2 socks Jack bought is x = 4.
-/

theorem socks_problem
  (x y z : ℕ)
  (h1 : x + y + z = 15)
  (h2 : 2 * x + 4 * y + 5 * z = 36)
  (h3 : 1 ≤ x)
  (h4 : 1 ≤ y)
  (h5 : 1 ≤ z) :
  x = 4 :=
  sorry

end socks_problem_l69_69032


namespace power_of_2_multiplication_l69_69053

theorem power_of_2_multiplication : (16^3) * (4^4) * (32^2) = 2^30 := by
  sorry

end power_of_2_multiplication_l69_69053


namespace problem_solution_l69_69998

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * (x ^ (x + 1)) = 246 :=
by
  sorry

end problem_solution_l69_69998


namespace susan_remaining_money_l69_69396

theorem susan_remaining_money :
  let initial_amount := 90
  let food_spent := 20
  let game_spent := 3 * food_spent
  let total_spent := food_spent + game_spent
  initial_amount - total_spent = 10 :=
by 
  sorry

end susan_remaining_money_l69_69396


namespace sum_of_roots_l69_69046

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - 2016 * x + 2015

theorem sum_of_roots (a b c : ℝ) (h1 : f a = c) (h2 : f b = c) (h3 : a ≠ b) :
  a + b = 2016 :=
by
  sorry

end sum_of_roots_l69_69046


namespace common_ratio_of_geometric_sequence_l69_69777

theorem common_ratio_of_geometric_sequence (a_1 q : ℝ) (hq : q ≠ 1) 
  (S : ℕ → ℝ) (hS: ∀ n, S n = a_1 * (1 - q^n) / (1 - q))
  (arithmetic_seq : 2 * S 7 = S 8 + S 9) :
  q = -2 :=
by sorry

end common_ratio_of_geometric_sequence_l69_69777


namespace line_perpendicular_intersection_l69_69409

noncomputable def line_equation (x y : ℝ) := 3 * x + y + 2 = 0

def is_perpendicular (m1 m2 : ℝ) := m1 * m2 = -1

theorem line_perpendicular_intersection (x y : ℝ) :
  (x - y + 2 = 0) →
  (2 * x + y + 1 = 0) →
  is_perpendicular (1 / 3) (-3) →
  line_equation x y := 
sorry

end line_perpendicular_intersection_l69_69409


namespace a_minus_b_7_l69_69607

theorem a_minus_b_7 (a b : ℤ) : (2 * y + a) * (y + b) = 2 * y^2 - 5 * y - 12 → a - b = 7 :=
by
  sorry

end a_minus_b_7_l69_69607


namespace simplify_1_simplify_2_l69_69752

theorem simplify_1 (a b : ℤ) : 2 * a - (a + b) = a - b :=
by
  sorry

theorem simplify_2 (x y : ℤ) : (x^2 - 2 * y^2) - 2 * (3 * y^2 - 2 * x^2) = 5 * x^2 - 8 * y^2 :=
by
  sorry

end simplify_1_simplify_2_l69_69752


namespace joan_payment_l69_69785

theorem joan_payment (cat_toy_cost cage_cost change_received : ℝ) 
  (h1 : cat_toy_cost = 8.77) 
  (h2 : cage_cost = 10.97) 
  (h3 : change_received = 0.26) : 
  cat_toy_cost + cage_cost - change_received = 19.48 := 
by 
  sorry

end joan_payment_l69_69785


namespace final_digit_is_two_l69_69824

-- Define initial conditions
def initial_ones : ℕ := 10
def initial_twos : ℕ := 10

-- Define the possible moves and the parity properties
def erase_identical (ones twos : ℕ) : ℕ × ℕ :=
  if ones ≥ 2 then (ones - 2, twos + 1)
  else (ones, twos - 1) -- for the case where two twos are removed

def erase_different (ones twos : ℕ) : ℕ × ℕ :=
  (ones, twos - 1)

-- Theorem stating that the final digit must be a two
theorem final_digit_is_two : 
∀ (ones twos : ℕ), ones = initial_ones → twos = initial_twos → 
(∃ n, ones + twos = n ∧ n = 1 ∧ (ones % 2 = 0)) → 
(∃ n, ones + twos = n ∧ n = 0 ∧ twos = 1) := 
by
  intros ones twos h_ones h_twos condition
  -- Constructing the proof should be done here
  sorry

end final_digit_is_two_l69_69824


namespace sum_reciprocals_of_factors_12_l69_69173

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l69_69173


namespace m_leq_neg3_l69_69261

theorem m_leq_neg3 (m : ℝ) (h : ∀ x ∈ Set.Icc (0 : ℝ) 1, x^2 - 4 * x ≥ m) : m ≤ -3 := 
  sorry

end m_leq_neg3_l69_69261


namespace rectangle_width_decrease_percent_l69_69865

theorem rectangle_width_decrease_percent (L W : ℝ) (h : L * W = L * W) :
  let L_new := 1.3 * L
  let W_new := W / 1.3 
  let percent_decrease := (1 - (W_new / W)) * 100
  percent_decrease = 23.08 :=
sorry

end rectangle_width_decrease_percent_l69_69865


namespace Timi_has_five_ears_l69_69869

theorem Timi_has_five_ears (seeing_ears_Imi seeing_ears_Dimi seeing_ears_Timi : ℕ)
  (H1 : seeing_ears_Imi = 8)
  (H2 : seeing_ears_Dimi = 7)
  (H3 : seeing_ears_Timi = 5)
  (total_ears : ℕ := (seeing_ears_Imi + seeing_ears_Dimi + seeing_ears_Timi) / 2) :
  total_ears - seeing_ears_Timi = 5 :=
by
  sorry -- Proof not required.

end Timi_has_five_ears_l69_69869


namespace loga_increasing_loga_decreasing_l69_69340

noncomputable def loga (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem loga_increasing (a : ℝ) (h₁ : a > 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a x < loga a y := by
  sorry 

theorem loga_decreasing (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a y < loga a x := by
  sorry

end loga_increasing_loga_decreasing_l69_69340


namespace range_of_third_side_l69_69128

theorem range_of_third_side (y : ℝ) : (2 < y) ↔ (y < 8) :=
by sorry

end range_of_third_side_l69_69128


namespace find_percentage_ryegrass_in_seed_mixture_X_l69_69076

open Real

noncomputable def percentage_ryegrass_in_seed_mixture_X (R : ℝ) : Prop := 
  let proportion_X : ℝ := 2 / 3
  let percentage_Y_ryegrass : ℝ := 25 / 100
  let proportion_Y : ℝ := 1 / 3
  let final_percentage_ryegrass : ℝ := 35 / 100
  final_percentage_ryegrass = (R / 100 * proportion_X) + (percentage_Y_ryegrass * proportion_Y)

/-
  Given the conditions:
  - Seed mixture Y is 25 percent ryegrass.
  - A mixture of seed mixtures X (66.67% of the mixture) and Y (33.33% of the mixture) contains 35 percent ryegrass.

  Prove:
  The percentage of ryegrass in seed mixture X is 40%.
-/
theorem find_percentage_ryegrass_in_seed_mixture_X : 
  percentage_ryegrass_in_seed_mixture_X 40 := 
  sorry

end find_percentage_ryegrass_in_seed_mixture_X_l69_69076


namespace only_n_eq_1_divides_2_pow_n_minus_1_l69_69573

theorem only_n_eq_1_divides_2_pow_n_minus_1 (n : ℕ) (h1 : 1 ≤ n) (h2 : n ∣ 2^n - 1) : n = 1 :=
sorry

end only_n_eq_1_divides_2_pow_n_minus_1_l69_69573


namespace transformed_coords_of_point_l69_69878

noncomputable def polar_to_rectangular_coordinates (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def transformed_coordinates (r θ : ℝ) : ℝ × ℝ :=
  let new_r := r ^ 3
  let new_θ := (3 * Real.pi / 2) * θ
  polar_to_rectangular_coordinates new_r new_θ

theorem transformed_coords_of_point (r θ : ℝ)
  (h_r : r = Real.sqrt (8^2 + 6^2))
  (h_cosθ : Real.cos θ = 8 / 10)
  (h_sinθ : Real.sin θ = 6 / 10)
  (coords_match : polar_to_rectangular_coordinates r θ = (8, 6)) :
  transformed_coordinates r θ = (-600, -800) :=
by
  -- The proof goes here
  sorry

end transformed_coords_of_point_l69_69878


namespace log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l69_69203

-- Define irrational numbers in Lean
def irrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Prove that log base 2 of 3 is irrational
theorem log_two_three_irrational : irrational (Real.log 3 / Real.log 2) := 
sorry

-- Prove that log base sqrt(2) of 3 is irrational
theorem log_sqrt2_three_irrational : 
  irrational (Real.log 3 / (1/2 * Real.log 2)) := 
sorry

-- Prove that log base (5 + 3sqrt(2)) of (3 + 5sqrt(2)) is irrational
theorem log_five_plus_three_sqrt2_irrational :
  irrational (Real.log (3 + 5 * Real.sqrt 2) / Real.log (5 + 3 * Real.sqrt 2)) := 
sorry

end log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l69_69203


namespace minimize_M_l69_69303

noncomputable def M (x y : ℝ) : ℝ := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9

theorem minimize_M : ∃ x y, M x y = 5 ∧ x = -3 ∧ y = -2 :=
by
  sorry

end minimize_M_l69_69303


namespace maximize_product_l69_69937

variable (x y : ℝ)
variable (h_xy_pos : x > 0 ∧ y > 0)
variable (h_sum : x + y = 35)

theorem maximize_product : x^5 * y^2 ≤ (25: ℝ)^5 * (10: ℝ)^2 :=
by
  -- Here we need to prove that the product x^5 y^2 is maximized at (x, y) = (25, 10)
  sorry

end maximize_product_l69_69937


namespace range_of_a_l69_69620

theorem range_of_a 
  (a : ℝ) 
  (h₀ : ∀ x : ℝ, (3 ≤ x ∧ x ≤ 4) ↔ (y = 2 * x + (3 - a))) : 
  9 ≤ a ∧ a ≤ 11 := 
sorry

end range_of_a_l69_69620


namespace sqrt_x_minus_1_meaningful_l69_69038

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) := by
  sorry

end sqrt_x_minus_1_meaningful_l69_69038


namespace instantaneous_velocity_at_4_seconds_l69_69747

-- Define the equation of motion
def s (t : ℝ) : ℝ := t^2 - 2 * t + 5

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t - 2

theorem instantaneous_velocity_at_4_seconds : v 4 = 6 := by
  -- Proof goes here
  sorry

end instantaneous_velocity_at_4_seconds_l69_69747


namespace polar_to_rectangular_l69_69250

theorem polar_to_rectangular : 
  ∀ (r θ : ℝ), 
  r = 8 → 
  θ = 7 * Real.pi / 4 → 
  (r * Real.cos θ, r * Real.sin θ) = (4 * Real.sqrt 2, -4 * Real.sqrt 2) :=
by 
  intros r θ hr hθ
  rw [hr, hθ]
  sorry

end polar_to_rectangular_l69_69250


namespace roots_of_equation_l69_69940

theorem roots_of_equation :
  ∀ x : ℝ, (x^4 + x^2 - 20 = 0) ↔ (x = 2 ∨ x = -2) :=
by
  -- This will be the proof.
  -- We are claiming that x is a root of the polynomial if and only if x = 2 or x = -2.
  sorry

end roots_of_equation_l69_69940


namespace triangle_angle_opposite_c_l69_69339

theorem triangle_angle_opposite_c (a b c : ℝ) (x : ℝ) 
  (ha : a = 2) (hb : b = 2) (hc : c = 4) : x = 180 :=
by 
  -- proof steps are not required as per the instruction
  sorry

end triangle_angle_opposite_c_l69_69339


namespace isosceles_triangle_perimeter_l69_69485

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a = 6 ∨ b = 6) 
(h_isosceles : a = b ∨ b = a) : 
  a + b + a = 15 ∨ b + a + b = 15 :=
by sorry

end isosceles_triangle_perimeter_l69_69485


namespace find_n_l69_69466

theorem find_n (n : ℕ) :
  (2^n - 1) % 3 = 0 ∧ (∃ m : ℤ, (2^n - 1) / 3 ∣ 4 * m^2 + 1) →
  ∃ j : ℕ, n = 2^j :=
by
  sorry

end find_n_l69_69466


namespace find_a_l69_69421

theorem find_a (x a : ℝ) : 
  (a + 2 = 0) ↔ (a = -2) :=
by
  sorry

end find_a_l69_69421


namespace perfect_square_of_factorials_l69_69133

open Nat

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfect_square_of_factorials :
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  is_perfect_square E3 :=
by
  -- definition of E1, E2, E3, E4, E5 as expressions given conditions
  let E1 := factorial 98 * factorial 99
  let E2 := factorial 98 * factorial 100
  let E3 := factorial 99 * factorial 100
  let E4 := factorial 99 * factorial 101
  let E5 := factorial 100 * factorial 101
  
  -- specify that E3 is the perfect square
  show is_perfect_square E3

  sorry

end perfect_square_of_factorials_l69_69133


namespace average_minutes_correct_l69_69172

noncomputable def average_minutes_run_per_day : ℚ :=
  let f (fifth_graders : ℕ) : ℚ := (48 * (4 * fifth_graders) + 30 * (2 * fifth_graders) + 10 * fifth_graders) / (4 * fifth_graders + 2 * fifth_graders + fifth_graders)
  f 1

theorem average_minutes_correct :
  average_minutes_run_per_day = 88 / 7 :=
by
  sorry

end average_minutes_correct_l69_69172


namespace regular_polygon_sides_l69_69435

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l69_69435


namespace distance_from_P_to_origin_l69_69036

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_from_P_to_origin :
  distance (-1) 2 0 0 = Real.sqrt 5 :=
by
  sorry

end distance_from_P_to_origin_l69_69036


namespace PartI_PartII_l69_69856

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Problem statement for (Ⅰ)
theorem PartI (x : ℝ) : (f x < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by sorry

-- Define conditions for PartII
variables (x y : ℝ)
def condition1 : Prop := |x - y - 1| ≤ 1 / 3
def condition2 : Prop := |2 * y + 1| ≤ 1 / 6

-- Problem statement for (Ⅱ)
theorem PartII (h1 : condition1 x y) (h2 : condition2 y) : f x < 1 :=
by sorry

end PartI_PartII_l69_69856


namespace simplify_and_evaluate_expression_l69_69375

theorem simplify_and_evaluate_expression (x y : ℚ) (h_x : x = -2) (h_y : y = 1/2) :
  (x + 2 * y)^2 - (x + y) * (x - y) = -11/4 := by
  sorry

end simplify_and_evaluate_expression_l69_69375


namespace remainder_of_5032_div_28_l69_69444

theorem remainder_of_5032_div_28 : 5032 % 28 = 20 :=
by
  sorry

end remainder_of_5032_div_28_l69_69444


namespace distance_between_centers_of_externally_tangent_circles_l69_69376

noncomputable def external_tangent_distance (R r : ℝ) (hR : R = 2) (hr : r = 3) (tangent : R > 0 ∧ r > 0) : ℝ :=
  R + r

theorem distance_between_centers_of_externally_tangent_circles :
  external_tangent_distance 2 3 (by rfl) (by rfl) (by norm_num) = 5 :=
sorry

end distance_between_centers_of_externally_tangent_circles_l69_69376


namespace find_x_in_interval_l69_69776

theorem find_x_in_interval (x : ℝ) 
  (h₁ : 4 ≤ (x + 1) / (3 * x - 7)) 
  (h₂ : (x + 1) / (3 * x - 7) < 9) : 
  x ∈ Set.Ioc (32 / 13) (29 / 11) := 
sorry

end find_x_in_interval_l69_69776


namespace volume_of_normal_block_is_3_l69_69631

variable (w d l : ℝ)
def V_normal : ℝ := w * d * l
def V_large : ℝ := (2 * w) * (2 * d) * (3 * l)

theorem volume_of_normal_block_is_3 (h : V_large w d l = 36) : V_normal w d l = 3 :=
by sorry

end volume_of_normal_block_is_3_l69_69631


namespace contractor_net_earnings_l69_69313

-- Definitions based on given conditions
def total_days : ℕ := 30
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.50
def absent_days : ℕ := 10

-- Calculation of the total amount received (involving both working days' pay and fines for absent days)
def worked_days : ℕ := total_days - absent_days
def total_earnings : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day
def net_earnings : ℝ := total_earnings - total_fine

-- The Theorem to be proved
theorem contractor_net_earnings : net_earnings = 425 := 
by 
  sorry

end contractor_net_earnings_l69_69313


namespace long_furred_brown_dogs_l69_69243

theorem long_furred_brown_dogs :
  ∀ (T L B N LB : ℕ), T = 60 → L = 45 → B = 35 → N = 12 →
  (LB = L + B - (T - N)) → LB = 32 :=
by
  intros T L B N LB hT hL hB hN hLB
  sorry

end long_furred_brown_dogs_l69_69243


namespace inequality_solution_set_l69_69124

theorem inequality_solution_set (x : ℝ) : (x^2 + x) / (2*x - 1) ≤ 1 ↔ x < 1 / 2 := 
sorry

end inequality_solution_set_l69_69124


namespace jean_initial_stuffies_l69_69298

variable (S : ℕ) (h1 : S * 2 / 3 / 4 = 10)

theorem jean_initial_stuffies : S = 60 :=
by
  sorry

end jean_initial_stuffies_l69_69298


namespace ratio_Cheryl_C_to_Cyrus_Y_l69_69021

noncomputable def Cheryl_C : ℕ := 126
noncomputable def Madeline_M : ℕ := 63
noncomputable def Total_pencils : ℕ := 231
noncomputable def Cyrus_Y : ℕ := Total_pencils - Cheryl_C - Madeline_M

theorem ratio_Cheryl_C_to_Cyrus_Y : 
  Cheryl_C = 2 * Madeline_M → 
  Madeline_M + Cheryl_C + Cyrus_Y = Total_pencils → 
  Cheryl_C / Cyrus_Y = 3 :=
by
  intros h1 h2
  sorry

end ratio_Cheryl_C_to_Cyrus_Y_l69_69021


namespace max_value_of_quadratic_function_l69_69071

def quadratic_function (x : ℝ) : ℝ := -x^2 + 2*x + 4

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 5 ∧ ∀ y : ℝ, quadratic_function y ≤ 5 :=
by
  sorry

end max_value_of_quadratic_function_l69_69071


namespace possible_values_a_l69_69072

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 - a * x - 7 else a / x

theorem possible_values_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → -2 * x - a ≥ 0) ∧
  (∀ x : ℝ, x > 1 → -a / (x^2) ≥ 0) ∧
  (-8 - a ≤ a) →
  a = -2 ∨ a = -3 ∨ a = -4 :=
sorry

end possible_values_a_l69_69072


namespace topsoil_cost_l69_69019

theorem topsoil_cost (cost_per_cubic_foot : ℕ) (cubic_yard_to_cubic_foot : ℕ) (volume_in_cubic_yards : ℕ) :
  cost_per_cubic_foot = 8 →
  cubic_yard_to_cubic_foot = 27 →
  volume_in_cubic_yards = 3 →
  volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 648 :=
by
  intros h1 h2 h3
  sorry

end topsoil_cost_l69_69019


namespace mixture_solution_l69_69394

theorem mixture_solution (x y : ℝ) :
  (0.30 * x + 0.40 * y = 32) →
  (x + y = 100) →
  (x = 80) :=
by
  intros h₁ h₂
  sorry

end mixture_solution_l69_69394


namespace total_birds_on_fence_l69_69497

variable (initial_birds : ℕ := 1)
variable (added_birds : ℕ := 4)

theorem total_birds_on_fence : initial_birds + added_birds = 5 := by
  sorry

end total_birds_on_fence_l69_69497


namespace largest_three_digit_number_l69_69546

def divisible_by_each_digit (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∀ d ∈ digits, d ≠ 0 ∧ n % d = 0

def sum_of_digits_divisible_by (n : ℕ) (k : ℕ) : Prop :=
  let sum := (n / 100) + ((n / 10) % 10) + (n % 10)
  sum % k = 0

theorem largest_three_digit_number : ∃ n : ℕ, n = 936 ∧
  n >= 100 ∧ n < 1000 ∧
  divisible_by_each_digit n ∧
  sum_of_digits_divisible_by n 6 :=
by
  -- Proof details are omitted
  sorry

end largest_three_digit_number_l69_69546


namespace log_inequality_l69_69592

theorem log_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
    (Real.log (c ^ 2) / Real.log (a + b) + Real.log (a ^ 2) / Real.log (b + c) + Real.log (b ^ 2) / Real.log (c + a)) ≥ 3 :=
sorry

end log_inequality_l69_69592


namespace intersection_nonempty_implies_t_lt_1_l69_69916

def M (x : ℝ) := x ≤ 1
def P (t : ℝ) (x : ℝ) := x > t

theorem intersection_nonempty_implies_t_lt_1 {t : ℝ} (h : ∃ x, M x ∧ P t x) : t < 1 :=
by
  sorry

end intersection_nonempty_implies_t_lt_1_l69_69916


namespace quadratic_roots_in_intervals_l69_69861

theorem quadratic_roots_in_intervals (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  ∃ x₁ x₂ : ℝ, (a < x₁ ∧ x₁ < b) ∧ (b < x₂ ∧ x₂ < c) ∧
  3 * x₁^2 - 2 * (a + b + c) * x₁ + (a * b + b * c + c * a) = 0 ∧
  3 * x₂^2 - 2 * (a + b + c) * x₂ + (a * b + b * c + c * a) = 0 :=
by
  sorry

end quadratic_roots_in_intervals_l69_69861


namespace larger_pie_flour_amount_l69_69107

variable (p1 : ℕ) (f1 : ℚ) (p2 : ℕ) (f2 : ℚ)

def prepared_pie_crusts (p1 p2 : ℕ) (f1 : ℚ) (f2 : ℚ) : Prop :=
  p1 * f1 = p2 * f2

theorem larger_pie_flour_amount (h : prepared_pie_crusts 40 25 (1/8) f2) : f2 = 1/5 :=
by
  sorry

end larger_pie_flour_amount_l69_69107


namespace baseball_card_devaluation_l69_69414

variable (x : ℝ) -- Note: x will represent the yearly percent decrease in decimal form (e.g., x = 0.10 for 10%)

theorem baseball_card_devaluation :
  (1 - x) * (1 - x) = 0.81 → x = 0.10 :=
by
  sorry

end baseball_card_devaluation_l69_69414


namespace store_profit_l69_69475

theorem store_profit :
  let selling_price : ℝ := 80
  let cost_price_profitable : ℝ := (selling_price - 0.60 * selling_price)
  let cost_price_loss : ℝ := (selling_price + 0.20 * selling_price)
  selling_price + selling_price - cost_price_profitable - cost_price_loss = 10 := by
  sorry

end store_profit_l69_69475


namespace trajectory_equation_l69_69198

-- Define the condition that the distance to the coordinate axes is equal.
def equidistantToAxes (x y : ℝ) : Prop :=
  abs x = abs y

-- State the theorem that we need to prove.
theorem trajectory_equation (x y : ℝ) (h : equidistantToAxes x y) : y^2 = x^2 :=
by sorry

end trajectory_equation_l69_69198


namespace birds_count_214_l69_69093

def two_legged_birds_count (b m i : Nat) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 3 * i = 686 → b = 214

theorem birds_count_214 (b m i : Nat) : two_legged_birds_count b m i :=
by
  sorry

end birds_count_214_l69_69093


namespace ned_games_l69_69370

theorem ned_games (F: ℕ) (bought_from_friend garage_sale non_working good total_games: ℕ) 
  (h₁: bought_from_friend = F)
  (h₂: garage_sale = 27)
  (h₃: non_working = 74)
  (h₄: good = 3)
  (h₅: total_games = non_working + good)
  (h₆: total_games = bought_from_friend + garage_sale) :
  F = 50 :=
by
  sorry

end ned_games_l69_69370


namespace largest_integer_x_l69_69030

theorem largest_integer_x (x : ℤ) : 
  (0.2 : ℝ) < (x : ℝ) / 7 ∧ (x : ℝ) / 7 < (7 : ℝ) / 12 → x = 4 :=
sorry

end largest_integer_x_l69_69030


namespace polygon_at_least_9_sides_l69_69929

theorem polygon_at_least_9_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → (∃ θ, θ < 45 ∧ (∀ j, 1 ≤ j ∧ j ≤ n → θ = 360 / n))):
  9 ≤ n :=
sorry

end polygon_at_least_9_sides_l69_69929


namespace no_solutions_iff_a_positive_and_discriminant_non_positive_l69_69962

theorem no_solutions_iff_a_positive_and_discriminant_non_positive (a b c : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, ¬ (a * x^2 + b * x + c < 0)) ↔ (a > 0 ∧ (b^2 - 4 * a * c) ≤ 0) :=
  sorry

end no_solutions_iff_a_positive_and_discriminant_non_positive_l69_69962


namespace simplify_and_evaluate_expression_l69_69719

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  a * (1 - 2 * a) + 2 * (a + 1) * (a - 1) = 2021 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l69_69719


namespace intersection_count_sum_l69_69693

theorem intersection_count_sum : 
  let m := 252
  let n := 252
  m + n = 504 := 
by {
  let m := 252 
  let n := 252 
  exact Eq.refl 504
}

end intersection_count_sum_l69_69693


namespace egg_count_l69_69265

theorem egg_count (E : ℕ) (son_daughter_eaten : ℕ) (rhea_husband_eaten : ℕ) (total_eaten : ℕ) (total_eggs : ℕ) (uneaten : ℕ) (trays : ℕ) 
  (H1 : son_daughter_eaten = 2 * 2 * 7)
  (H2 : rhea_husband_eaten = 4 * 2 * 7)
  (H3 : total_eaten = son_daughter_eaten + rhea_husband_eaten)
  (H4 : uneaten = 6)
  (H5 : total_eggs = total_eaten + uneaten)
  (H6 : trays = 2)
  (H7 : total_eggs = E * trays) : 
  E = 45 :=
by
  sorry

end egg_count_l69_69265


namespace cube_edge_length_l69_69471

-- Define edge length and surface area
variables (edge_length surface_area : ℝ)

-- Given condition
def surface_area_condition : Prop := surface_area = 294

-- Cube surface area formula
def cube_surface_area : Prop := surface_area = 6 * edge_length^2

-- Proof statement
theorem cube_edge_length (h1: surface_area_condition surface_area) (h2: cube_surface_area edge_length surface_area) : edge_length = 7 := 
by
  sorry

end cube_edge_length_l69_69471


namespace average_age_of_team_l69_69386

/--
The captain of a cricket team of 11 members is 26 years old and the wicket keeper is 
3 years older. If the ages of these two are excluded, the average age of the remaining 
players is one year less than the average age of the whole team. Prove that the average 
age of the whole team is 32 years.
-/
theorem average_age_of_team 
  (captain_age : Nat) (wicket_keeper_age : Nat) (remaining_9_average_age : Nat)
  (team_size : Nat) (total_team_age : Nat) (remaining_9_total_age : Nat)
  (A : Nat) :
  captain_age = 26 →
  wicket_keeper_age = captain_age + 3 →
  team_size = 11 →
  total_team_age = team_size * A →
  total_team_age = remaining_9_total_age + captain_age + wicket_keeper_age →
  remaining_9_total_age = 9 * (A - 1) →
  A = 32 :=
by
  sorry

end average_age_of_team_l69_69386


namespace total_savings_at_end_of_year_l69_69886

-- Defining constants for daily savings and the number of days in a year
def daily_savings : ℕ := 24
def days_in_year : ℕ := 365

-- Stating the theorem
theorem total_savings_at_end_of_year : daily_savings * days_in_year = 8760 :=
by
  sorry

end total_savings_at_end_of_year_l69_69886


namespace dalmatians_with_right_ear_spots_l69_69880

def TotalDalmatians := 101
def LeftOnlySpots := 29
def RightOnlySpots := 17
def NoEarSpots := 22

theorem dalmatians_with_right_ear_spots : 
  (TotalDalmatians - LeftOnlySpots - NoEarSpots) = 50 :=
by
  -- Proof goes here, but for now, we use sorry
  sorry

end dalmatians_with_right_ear_spots_l69_69880


namespace kishore_savings_l69_69849

-- Define the monthly expenses and condition
def expenses : Real :=
  5000 + 1500 + 4500 + 2500 + 2000 + 6100

-- Define the monthly salary and savings conditions
def salary (S : Real) : Prop :=
  expenses + 0.1 * S = S

-- Define the savings amount
def savings (S : Real) : Real :=
  0.1 * S

-- The theorem to prove
theorem kishore_savings : ∃ S : Real, salary S ∧ savings S = 2733.33 :=
by
  sorry

end kishore_savings_l69_69849


namespace find_p_and_q_l69_69978

theorem find_p_and_q (p q : ℝ)
    (M : Set ℝ := {x | x^2 + p * x - 2 = 0})
    (N : Set ℝ := {x | x^2 - 2 * x + q = 0})
    (h : M ∪ N = {-1, 0, 2}) :
    p = -1 ∧ q = 0 :=
sorry

end find_p_and_q_l69_69978


namespace round_robin_chess_l69_69899

/-- 
In a round-robin chess tournament, two boys and several girls participated. 
The boys together scored 8 points, while all the girls scored an equal number of points.
We are to prove that the number of girls could have participated in the tournament is 7 or 14,
given that a win is 1 point, a draw is 0.5 points, and a loss is 0 points.
-/
theorem round_robin_chess (n : ℕ) (x : ℚ) (h : 2 * n * x + 16 = n ^ 2 + 3 * n + 2) : n = 7 ∨ n = 14 :=
sorry

end round_robin_chess_l69_69899


namespace genevieve_coffee_drink_l69_69258

theorem genevieve_coffee_drink :
  let gallons := 4.5
  let small_thermos_count := 12
  let small_thermos_capacity_ml := 250
  let large_thermos_count := 6
  let large_thermos_capacity_ml := 500
  let genevieve_small_thermos_drink_count := 2
  let genevieve_large_thermos_drink_count := 1
  let ounces_per_gallon := 128
  let mls_per_ounce := 29.5735
  let total_mls := (gallons * ounces_per_gallon) * mls_per_ounce
  let genevieve_ml_drink := (genevieve_small_thermos_drink_count * small_thermos_capacity_ml) 
                            + (genevieve_large_thermos_drink_count * large_thermos_capacity_ml)
  let genevieve_ounces_drink := genevieve_ml_drink / mls_per_ounce
  genevieve_ounces_drink = 33.814 :=
by sorry

end genevieve_coffee_drink_l69_69258


namespace tagged_fish_in_second_catch_l69_69714

theorem tagged_fish_in_second_catch :
  ∀ (T : ℕ),
    (40 > 0) →
    (800 > 0) →
    (T / 40 = 40 / 800) →
    T = 2 := 
by
  intros T h1 h2 h3
  sorry

end tagged_fish_in_second_catch_l69_69714


namespace degree_measure_of_regular_hexagon_interior_angle_l69_69001

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l69_69001


namespace calculate_total_weight_l69_69084

variable (a b c d : ℝ)

-- Conditions
def I_II_weight := a + b = 156
def III_IV_weight := c + d = 195
def I_III_weight := a + c = 174
def II_IV_weight := b + d = 186

theorem calculate_total_weight (I_II_weight : a + b = 156) (III_IV_weight : c + d = 195)
    (I_III_weight : a + c = 174) (II_IV_weight : b + d = 186) :
    a + b + c + d = 355.5 :=
by
    sorry

end calculate_total_weight_l69_69084


namespace Ruth_sandwiches_l69_69986

theorem Ruth_sandwiches (sandwiches_left sandwiches_ruth sandwiches_brother sandwiches_first_cousin sandwiches_two_cousins total_sandwiches : ℕ)
  (h_ruth : sandwiches_ruth = 1)
  (h_brother : sandwiches_brother = 2)
  (h_first_cousin : sandwiches_first_cousin = 2)
  (h_two_cousins : sandwiches_two_cousins = 2)
  (h_left : sandwiches_left = 3) :
  total_sandwiches = sandwiches_left + sandwiches_two_cousins + sandwiches_first_cousin + sandwiches_ruth + sandwiches_brother :=
by
  sorry

end Ruth_sandwiches_l69_69986


namespace find_k_and_x2_l69_69492

theorem find_k_and_x2 (k : ℝ) (x2 : ℝ)
  (h1 : 2 * x2 = k)
  (h2 : 2 + x2 = 6) :
  k = 8 ∧ x2 = 4 :=
by
  sorry

end find_k_and_x2_l69_69492


namespace pauly_omelets_l69_69524

/-- Pauly is making omelets for his family. There are three dozen eggs, and he plans to use them all. 
Each omelet requires 4 eggs. Including himself, there are 3 people. 
Prove that each person will get 3 omelets. -/

def total_eggs := 3 * 12

def eggs_per_omelet := 4

def total_omelets := total_eggs / eggs_per_omelet

def number_of_people := 3

def omelets_per_person := total_omelets / number_of_people

theorem pauly_omelets : omelets_per_person = 3 :=
by
  -- Placeholder proof
  sorry

end pauly_omelets_l69_69524


namespace total_people_at_beach_l69_69216

-- Specifications of the conditions
def joined_people : ℕ := 100
def left_people : ℕ := 40
def family_count : ℕ := 3

-- Theorem stating the total number of people at the beach in the evening
theorem total_people_at_beach :
  joined_people - left_people + family_count = 63 := by
  sorry

end total_people_at_beach_l69_69216


namespace bus_passengers_final_count_l69_69640

theorem bus_passengers_final_count :
  let initial_passengers := 15
  let changes := [(3, -6), (-2, 4), (-7, 2), (3, -5)]
  let apply_change (acc : Int) (change : Int × Int) : Int :=
    acc + change.1 + change.2
  initial_passengers + changes.foldl apply_change 0 = 7 :=
by
  intros
  sorry

end bus_passengers_final_count_l69_69640


namespace abs_diff_61st_term_l69_69338

-- Define sequences C and D
def seqC (n : ℕ) : ℤ := 20 + 15 * (n - 1)
def seqD (n : ℕ) : ℤ := 20 - 15 * (n - 1)

-- Prove the absolute value of the difference between the 61st terms is 1800
theorem abs_diff_61st_term : (abs (seqC 61 - seqD 61) = 1800) :=
by
  sorry

end abs_diff_61st_term_l69_69338


namespace equation_of_line_l69_69476

theorem equation_of_line 
  (a : ℝ) (h : a < 3) 
  (C : ℝ × ℝ) 
  (hC : C = (-2, 3)) 
  (l_intersects_circle : ∃ A B : ℝ × ℝ, 
    (A.1^2 + A.2^2 + 2 * A.1 - 4 * A.2 + a = 0) ∧ 
    (B.1^2 + B.2^2 + 2 * B.1 - 4 * B.2 + a = 0) ∧ 
    (C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))) : 
  ∃ (m b : ℝ), 
    (m = 1) ∧ 
    (b = -5) ∧ 
    (∀ x y, y - 3 = m * (x + 2) ↔ x - y + 5 = 0) :=
by
  sorry

end equation_of_line_l69_69476


namespace parametric_to_standard_equation_l69_69413

theorem parametric_to_standard_equation (x y t : ℝ) 
(h1 : x = 4 * t + 1) 
(h2 : y = -2 * t - 5) : 
x + 2 * y + 9 = 0 :=
by
  sorry

end parametric_to_standard_equation_l69_69413


namespace part1_part2_l69_69936

theorem part1 (m : ℂ) : (m * (m + 2)).re = 0 ∧ (m^2 + m - 2).im ≠ 0 → m = 0 := by
  sorry

theorem part2 (m : ℝ) : (m * (m + 2) > 0 ∧ m^2 + m - 2 < 0) → 0 < m ∧ m < 1 := by
  sorry

end part1_part2_l69_69936


namespace billboards_color_schemes_is_55_l69_69453

def adjacent_color_schemes (n : ℕ) : ℕ :=
  if h : n = 8 then 55 else 0

theorem billboards_color_schemes_is_55 :
  adjacent_color_schemes 8 = 55 :=
sorry

end billboards_color_schemes_is_55_l69_69453


namespace eqn_y_value_l69_69254

theorem eqn_y_value (y : ℝ) (h : (2 / y) + ((3 / y) / (6 / y)) = 1.5) : y = 2 :=
sorry

end eqn_y_value_l69_69254


namespace harris_carrot_cost_l69_69638

-- Definitions stemming from the conditions
def carrots_per_day : ℕ := 1
def days_per_year : ℕ := 365
def carrots_per_bag : ℕ := 5
def cost_per_bag : ℕ := 2

-- Prove that Harris's total cost for carrots in one year is $146
theorem harris_carrot_cost : (days_per_year * carrots_per_day / carrots_per_bag) * cost_per_bag = 146 := by
  sorry

end harris_carrot_cost_l69_69638


namespace max_constant_N_l69_69366

theorem max_constant_N (a b c d : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0):
  (c^2 + d^2) ≠ 0 → ∃ N, N = 1 ∧ (a^2 + b^2) / (c^2 + d^2) ≤ 1 :=
by
  sorry

end max_constant_N_l69_69366


namespace log_comparison_l69_69634

open Real

noncomputable def a := log 4 / log 5  -- a = log_5(4)
noncomputable def b := log 5 / log 3  -- b = log_3(5)
noncomputable def c := log 5 / log 4  -- c = log_4(5)

theorem log_comparison : a < c ∧ c < b := 
by
  sorry

end log_comparison_l69_69634


namespace calculate_unshaded_perimeter_l69_69601

-- Defining the problem's conditions and results.
def total_length : ℕ := 20
def total_width : ℕ := 12
def shaded_area : ℕ := 65
def inner_shaded_width : ℕ := 5
def total_area : ℕ := total_length * total_width
def unshaded_area : ℕ := total_area - shaded_area

-- Define dimensions for the unshaded region based on the problem conditions.
def unshaded_width : ℕ := total_width - inner_shaded_width
def unshaded_height : ℕ := unshaded_area / unshaded_width

-- Calculate perimeter of the unshaded region.
def unshaded_perimeter : ℕ := 2 * (unshaded_width + unshaded_height)

-- Stating the theorem to be proved.
theorem calculate_unshaded_perimeter : unshaded_perimeter = 64 := 
sorry

end calculate_unshaded_perimeter_l69_69601


namespace number_of_blocks_needed_l69_69727

-- Define the dimensions of the fort
def fort_length : ℕ := 20
def fort_width : ℕ := 15
def fort_height : ℕ := 8

-- Define the thickness of the walls and the floor
def wall_thickness : ℕ := 2
def floor_thickness : ℕ := 1

-- Define the original volume of the fort
def V_original : ℕ := fort_length * fort_width * fort_height

-- Define the interior dimensions of the fort considering the thickness of the walls and floor
def interior_length : ℕ := fort_length - 2 * wall_thickness
def interior_width : ℕ := fort_width - 2 * wall_thickness
def interior_height : ℕ := fort_height - floor_thickness

-- Define the volume of the interior space
def V_interior : ℕ := interior_length * interior_width * interior_height

-- Statement to prove: number of blocks needed equals 1168
theorem number_of_blocks_needed : V_original - V_interior = 1168 := 
by 
  sorry

end number_of_blocks_needed_l69_69727


namespace infinite_integer_and_noninteger_terms_l69_69757

theorem infinite_integer_and_noninteger_terms (m : Nat) (h_m : m > 0) :
  ∃ (infinite_int_terms : Nat → Prop) (infinite_nonint_terms : Nat → Prop),
  (∀ n, ∃ k, infinite_int_terms k ∧ ∀ k, infinite_int_terms k → ∃ N, k = n + N + 1) ∧
  (∀ n, ∃ k, infinite_nonint_terms k ∧ ∀ k, infinite_nonint_terms k → ∃ N, k = n + N + 1) :=
sorry

end infinite_integer_and_noninteger_terms_l69_69757


namespace traffic_accident_emergency_number_l69_69932

theorem traffic_accident_emergency_number (A B C D : ℕ) (h1 : A = 122) (h2 : B = 110) (h3 : C = 120) (h4 : D = 114) : 
  A = 122 := 
by
  exact h1

end traffic_accident_emergency_number_l69_69932


namespace smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l69_69609

noncomputable def smaller_angle_at_715 : ℝ :=
  let hour_position := 7 * 30 + 30 / 4
  let minute_position := 15 * (360 / 60)
  let angle_between := abs (hour_position - minute_position)
  if angle_between > 180 then 360 - angle_between else angle_between

theorem smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m :
  smaller_angle_at_715 = 127.5 := 
sorry

end smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l69_69609


namespace remainder_of_expression_l69_69005

theorem remainder_of_expression :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by {
  -- Prove the expression step by step
  -- sorry
  sorry
}

end remainder_of_expression_l69_69005


namespace max_value_of_z_l69_69099

theorem max_value_of_z (x y z : ℝ) (h_add : x + y + z = 5) (h_mult : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end max_value_of_z_l69_69099


namespace total_cost_l69_69118

variable (a b : ℝ)

theorem total_cost (ha : a ≥ 0) (hb : b ≥ 0) : 3 * a + 4 * b = 3 * a + 4 * b :=
by sorry

end total_cost_l69_69118


namespace depth_of_first_hole_l69_69354

theorem depth_of_first_hole (n1 t1 n2 t2 : ℕ) (D : ℝ) (r : ℝ) 
  (h1 : n1 = 45) (h2 : t1 = 8) (h3 : n2 = 90) (h4 : t2 = 6) 
  (h5 : r = 1 / 12) (h6 : D = n1 * t1 * r) (h7 : n2 * t2 * r = 45) : 
  D = 30 := 
by 
  sorry

end depth_of_first_hole_l69_69354


namespace max_p_plus_q_l69_69704

theorem max_p_plus_q (p q : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → 2 * p * x^2 + q * x - p + 1 ≥ 0) : p + q ≤ 2 :=
sorry

end max_p_plus_q_l69_69704


namespace triangle_first_side_length_l69_69066

theorem triangle_first_side_length (x : ℕ) (h1 : x + 20 + 30 = 55) : x = 5 :=
by
  sorry

end triangle_first_side_length_l69_69066


namespace corrected_mean_of_observations_l69_69675

theorem corrected_mean_of_observations (mean : ℝ) (n : ℕ) (incorrect_observation : ℝ) (correct_observation : ℝ) 
  (h_mean : mean = 41) (h_n : n = 50) (h_incorrect_observation : incorrect_observation = 23) (h_correct_observation : correct_observation = 48) 
  (h_sum_incorrect : mean * n = 2050) : 
  (mean * n - incorrect_observation + correct_observation) / n = 41.5 :=
by
  sorry

end corrected_mean_of_observations_l69_69675


namespace multiply_polynomials_l69_69912

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end multiply_polynomials_l69_69912


namespace sum_of_squares_base_case_l69_69352

theorem sum_of_squares_base_case : 1^2 + 2^2 = (1 * 3 * 5) / 3 := by sorry

end sum_of_squares_base_case_l69_69352


namespace inverse_proportion_quadrants_l69_69263

theorem inverse_proportion_quadrants (a k : ℝ) (ha : a ≠ 0) (h : (3 * a, a) ∈ {p : ℝ × ℝ | p.snd = k / p.fst}) :
  k = 3 * a^2 ∧ k > 0 ∧
  (
    (∀ x y : ℝ, x > 0 → y = k / x → y > 0) ∨
    (∀ x y : ℝ, x < 0 → y = k / x → y < 0)
  ) :=
by
  sorry

end inverse_proportion_quadrants_l69_69263


namespace total_respondents_l69_69957

theorem total_respondents (x_preference resp_y : ℕ) (h1 : x_preference = 360) (h2 : 9 * resp_y = x_preference) : 
  resp_y + x_preference = 400 :=
by 
  sorry

end total_respondents_l69_69957


namespace count_big_boxes_l69_69465

theorem count_big_boxes (B : ℕ) (h : 7 * B + 4 * 9 = 71) : B = 5 :=
sorry

end count_big_boxes_l69_69465


namespace min_value_of_f_l69_69397

noncomputable def f (x : ℝ) : ℝ :=
  1 / (Real.sqrt (x^2 + 2)) + Real.sqrt (x^2 + 2)

theorem min_value_of_f :
  ∃ x : ℝ, f x = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end min_value_of_f_l69_69397


namespace problem_l69_69971

theorem problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1 / 2) :
  (1 - x) / (1 + x) * (1 - y) / (1 + y) * (1 - z) / (1 + z) ≥ 1 / 3 :=
by
  sorry

end problem_l69_69971


namespace cube_surface_area_l69_69170

noncomputable def volume (x : ℝ) : ℝ := x ^ 3

noncomputable def surface_area (x : ℝ) : ℝ := 6 * x ^ 2

theorem cube_surface_area (x : ℝ) :
  surface_area x = 6 * x ^ 2 :=
by sorry

end cube_surface_area_l69_69170


namespace paving_rate_l69_69832

theorem paving_rate
  (length : ℝ) (width : ℝ) (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) :
  total_cost / (length * width) = 800 := by
  sorry

end paving_rate_l69_69832


namespace number_of_students_in_club_l69_69104

variable (y : ℕ) -- Number of girls

def total_stickers_given (y : ℕ) : ℕ := y * y + (y + 3) * (y + 3)

theorem number_of_students_in_club :
  (total_stickers_given y = 640) → (2 * y + 3 = 35) := 
by
  intro h1
  sorry

end number_of_students_in_club_l69_69104


namespace sum_of_all_four_is_zero_l69_69023

variables {a b c d : ℤ}

theorem sum_of_all_four_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum_rows : a + b = c + d) 
  (h_product_columns : a * c = b * d) :
  a + b + c + d = 0 := 
sorry

end sum_of_all_four_is_zero_l69_69023


namespace arith_sign_change_geo_sign_change_l69_69026

-- Definitions for sequences
def arith_sequence (a₁ d : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => arith_sequence a₁ d n + d

def geo_sequence (a₁ r : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => geo_sequence a₁ r n * r

-- Problem statement
theorem arith_sign_change :
  ∀ (a₁ d : ℝ), (∃ N : ℕ, arith_sequence a₁ d N = 0) ∨ (∀ n m : ℕ, (arith_sequence a₁ d n) * (arith_sequence a₁ d m) ≥ 0) :=
sorry

theorem geo_sign_change :
  ∀ (a₁ r : ℝ), r < 0 → ∀ n : ℕ, (geo_sequence a₁ r n) * (geo_sequence a₁ r (n + 1)) < 0 :=
sorry

end arith_sign_change_geo_sign_change_l69_69026


namespace find_natural_number_n_l69_69773

def is_terminating_decimal (x : ℚ) : Prop :=
  ∃ (a b : ℕ), x = (a / b) ∧ (∃ (m n : ℕ), b = 2 ^ m * 5 ^ n)

theorem find_natural_number_n (n : ℕ) (h₁ : is_terminating_decimal (1 / n)) (h₂ : is_terminating_decimal (1 / (n + 1))) : n = 4 :=
by sorry

end find_natural_number_n_l69_69773


namespace cos_neg_300_eq_half_l69_69977

theorem cos_neg_300_eq_half : Real.cos (-300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_neg_300_eq_half_l69_69977


namespace combine_like_terms_l69_69796

theorem combine_like_terms (x y : ℝ) : 3 * x * y - 6 * x * y + (-2) * x * y = -5 * x * y := by
  sorry

end combine_like_terms_l69_69796


namespace carA_travel_time_l69_69618

theorem carA_travel_time 
    (speedA speedB distanceB : ℕ)
    (ratio : ℕ)
    (timeB : ℕ)
    (h_speedA : speedA = 50)
    (h_speedB : speedB = 100)
    (h_distanceB : distanceB = speedB * timeB)
    (h_ratio : distanceA / distanceB = ratio)
    (h_ratio_value : ratio = 3)
    (h_timeB : timeB = 1)
  : distanceA / speedA = 6 :=
by sorry

end carA_travel_time_l69_69618


namespace square_side_length_l69_69684

-- Variables for the conditions
variables (totalWire triangleWire : ℕ)
-- Definitions of the conditions
def totalLengthCondition := totalWire = 78
def triangleLengthCondition := triangleWire = 46

-- Goal is to prove the side length of the square
theorem square_side_length
  (h1 : totalLengthCondition totalWire)
  (h2 : triangleLengthCondition triangleWire)
  : (totalWire - triangleWire) / 4 = 8 := 
by
  rw [totalLengthCondition, triangleLengthCondition] at *
  sorry

end square_side_length_l69_69684


namespace find_certain_number_l69_69103

theorem find_certain_number (D S X : ℕ): 
  D = 20 → 
  S = 55 → 
  X + (D - S) = 3 * D - 90 →
  X = 5 := 
by
  sorry

end find_certain_number_l69_69103


namespace probability_of_selecting_one_male_and_one_female_l69_69390

noncomputable def probability_one_male_one_female : ℚ :=
  let total_ways := (Nat.choose 6 2) -- Total number of ways to select 2 out of 6
  let ways_one_male_one_female := (Nat.choose 3 1) * (Nat.choose 3 1) -- Ways to select 1 male and 1 female
  ways_one_male_one_female / total_ways

theorem probability_of_selecting_one_male_and_one_female :
  probability_one_male_one_female = 3 / 5 := by
  sorry

end probability_of_selecting_one_male_and_one_female_l69_69390


namespace find_A_l69_69059

variable (p q r s A : ℝ)

theorem find_A (H1 : (p + q + r + s) / 4 = 5) (H2 : (p + q + r + s + A) / 5 = 8) : A = 20 := 
by
  sorry

end find_A_l69_69059


namespace each_girl_gets_2_dollars_l69_69857

theorem each_girl_gets_2_dollars :
  let debt := 40
  let lulu_savings := 6
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  total_savings - debt = 6 → (total_savings - debt) / 3 = 2 :=
by
  sorry

end each_girl_gets_2_dollars_l69_69857


namespace infimum_of_function_l69_69206

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 1)^2

def is_lower_bound (M : ℝ) (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x ≥ M

def is_infimum (M : ℝ) (f : ℝ → ℝ) : Prop :=
  is_lower_bound M f ∧ ∀ L : ℝ, is_lower_bound L f → L ≤ M

theorem infimum_of_function :
  is_infimum 0.5 f :=
sorry

end infimum_of_function_l69_69206


namespace fraction_of_girls_on_trip_l69_69289

theorem fraction_of_girls_on_trip (b g : ℕ) (h : b = g) :
  ((2 / 3 * g) / (5 / 6 * b + 2 / 3 * g)) = 4 / 9 :=
by
  sorry

end fraction_of_girls_on_trip_l69_69289


namespace part1_part2_l69_69851

-- Define properties for the first part of the problem
def condition1 (weightA weightB : ℕ) : Prop :=
  weightA + weightB = 7500 ∧ weightA = 3 * weightB / 2

def question1_answer : Prop :=
  ∃ weightA weightB : ℕ, condition1 weightA weightB ∧ weightA = 4500 ∧ weightB = 3000

-- Combined condition for the second part of the problem scenarios
def condition2a (y : ℕ) : Prop := y ≤ 1800 ∧ 18 * y - 10 * y = 17400
def condition2b (y : ℕ) : Prop := 1800 < y ∧ y ≤ 3000 ∧ 18 * y - (15 * y - 9000) = 17400
def condition2c (y : ℕ) : Prop := y > 3000 ∧ 18 * y - (20 * y - 24000) = 17400

def question2_answer : Prop :=
  (∃ y : ℕ, condition2b y ∧ y = 2800) ∨ (∃ y : ℕ, condition2c y ∧ y = 3300)

-- The Lean statements for both parts of the problem
theorem part1 : question1_answer := sorry

theorem part2 : question2_answer := sorry

end part1_part2_l69_69851


namespace find_initial_period_l69_69784

theorem find_initial_period (P : ℝ) (T : ℝ) 
  (h1 : 1680 = (P * 4 * T) / 100)
  (h2 : 1680 = (P * 5 * 4) / 100) 
  : T = 5 := 
by 
  sorry

end find_initial_period_l69_69784


namespace coincide_green_square_pairs_l69_69817

structure Figure :=
  (green_squares : ℕ)
  (red_triangles : ℕ)
  (blue_triangles : ℕ)

theorem coincide_green_square_pairs (f : Figure) (hs : f.green_squares = 4)
  (rt : f.red_triangles = 3) (bt : f.blue_triangles = 6)
  (gs_coincide : ∀ n, n ≤ f.green_squares ⟶ n = f.green_squares) 
  (rt_coincide : ∃ n, n = 2) (bt_coincide : ∃ n, n = 2) 
  (red_blue_pairs : ∃ n, n = 3) : 
  ∃ pairs, pairs = 4 :=
by 
  sorry

end coincide_green_square_pairs_l69_69817


namespace border_material_length_l69_69493

noncomputable def area (r : ℝ) : ℝ := (22 / 7) * r^2

theorem border_material_length (r : ℝ) (C : ℝ) (border : ℝ) : 
  area r = 616 →
  C = 2 * (22 / 7) * r →
  border = C + 3 →
  border = 91 :=
by
  intro h_area h_circumference h_border
  sorry

end border_material_length_l69_69493


namespace divide_fractions_l69_69086

theorem divide_fractions :
  (7 / 3) / (5 / 4) = (28 / 15) :=
by
  sorry

end divide_fractions_l69_69086


namespace andrew_purchased_mangoes_l69_69617

variable (m : ℕ)

def cost_of_grapes := 8 * 70
def cost_of_mangoes (m : ℕ) := 55 * m
def total_cost (m : ℕ) := cost_of_grapes + cost_of_mangoes m

theorem andrew_purchased_mangoes :
  total_cost m = 1055 → m = 9 := by
  intros h_total_cost
  sorry

end andrew_purchased_mangoes_l69_69617


namespace inequalities_not_all_hold_l69_69552

theorem inequalities_not_all_hold (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
    ¬ (a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by
  sorry

end inequalities_not_all_hold_l69_69552


namespace min_value_of_squares_l69_69954

variable (a b t : ℝ)

theorem min_value_of_squares (ht : 0 < t) (habt : a + b = t) : 
  a^2 + b^2 ≥ t^2 / 2 := 
by
  sorry

end min_value_of_squares_l69_69954


namespace polynomial_coefficients_l69_69528

theorem polynomial_coefficients (x a₄ a₃ a₂ a₁ a₀ : ℝ) (h : (x - 1)^4 = a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀)
  : a₄ - a₃ + a₂ - a₁ = 15 := by
  sorry

end polynomial_coefficients_l69_69528


namespace inverse_function_evaluation_l69_69897

def g (x : ℕ) : ℕ :=
  if x = 1 then 4
  else if x = 2 then 5
  else if x = 3 then 2
  else if x = 4 then 3
  else if x = 5 then 1
  else 0  -- default case, though it shouldn't be used given the conditions

noncomputable def g_inv (y : ℕ) : ℕ :=
  if y = 4 then 1
  else if y = 5 then 2
  else if y = 2 then 3
  else if y = 3 then 4
  else if y = 1 then 5
  else 0  -- default case, though it shouldn't be used given the conditions

theorem inverse_function_evaluation : g_inv (g_inv (g_inv 4)) = 2 := by
  sorry

end inverse_function_evaluation_l69_69897


namespace total_bill_for_group_is_129_l69_69701

theorem total_bill_for_group_is_129 :
  let num_adults := 6
  let num_teenagers := 3
  let num_children := 1
  let cost_adult_meal := 9
  let cost_teenager_meal := 7
  let cost_child_meal := 5
  let cost_soda := 2.50
  let num_sodas := 10
  let cost_dessert := 4
  let num_desserts := 3
  let cost_appetizer := 6
  let num_appetizers := 2
  let total_bill := 
    (num_adults * cost_adult_meal) +
    (num_teenagers * cost_teenager_meal) +
    (num_children * cost_child_meal) +
    (num_sodas * cost_soda) +
    (num_desserts * cost_dessert) +
    (num_appetizers * cost_appetizer)
  total_bill = 129 := by
sorry

end total_bill_for_group_is_129_l69_69701


namespace total_travel_options_l69_69803

theorem total_travel_options (trains_A_to_B : ℕ) (ferries_B_to_C : ℕ) (flights_A_to_C : ℕ) 
  (h1 : trains_A_to_B = 3) (h2 : ferries_B_to_C = 2) (h3 : flights_A_to_C = 2) :
  (trains_A_to_B * ferries_B_to_C + flights_A_to_C = 8) :=
by
  sorry

end total_travel_options_l69_69803


namespace find_some_number_l69_69167

theorem find_some_number (some_number : ℝ) (h : (3.242 * some_number) / 100 = 0.045388) : some_number = 1.400 := 
sorry

end find_some_number_l69_69167


namespace P_finishes_in_15_minutes_more_l69_69754

variable (P Q : ℝ)

def rate_p := 1 / 4
def rate_q := 1 / 15
def time_together := 3
def total_job := 1

theorem P_finishes_in_15_minutes_more :
  let combined_rate := rate_p + rate_q
  let completed_job_in_3_hours := combined_rate * time_together
  let remaining_job := total_job - completed_job_in_3_hours
  let time_for_P_to_finish := remaining_job / rate_p
  let minutes_needed := time_for_P_to_finish * 60
  minutes_needed = 15 :=
by
  -- Proof steps go here
  sorry

end P_finishes_in_15_minutes_more_l69_69754


namespace find_y_l69_69860

-- Define the sequence from 1 to 50
def seq_sum : ℕ := (50 * 51) / 2

-- Define y and the average condition
def average_condition (y : ℚ) : Prop :=
  (seq_sum + y) / 51 = 51 * y

-- Theorem statement
theorem find_y (y : ℚ) (h : average_condition y) : y = 51 / 104 :=
by
  sorry

end find_y_l69_69860


namespace james_beats_old_record_l69_69891

def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def two_point_conversions : ℕ := 6
def points_per_two_point_conversion : ℕ := 2
def field_goals : ℕ := 8
def points_per_field_goal : ℕ := 3
def extra_points : ℕ := 20
def points_per_extra_point : ℕ := 1
def old_record : ℕ := 300

theorem james_beats_old_record :
  touchdowns_per_game * points_per_touchdown * games_in_season +
  two_point_conversions * points_per_two_point_conversion +
  field_goals * points_per_field_goal +
  extra_points * points_per_extra_point - old_record = 116 := by
  sorry -- Proof is omitted.

end james_beats_old_record_l69_69891


namespace length_segment_FF_l69_69655

-- Define the points F and F' based on the given conditions
def F : (ℝ × ℝ) := (4, 3)
def F' : (ℝ × ℝ) := (-4, 3)

-- The theorem to prove the length of the segment FF' is 8
theorem length_segment_FF' : dist F F' = 8 :=
by
  sorry

end length_segment_FF_l69_69655


namespace comparison_of_neg_square_roots_l69_69018

noncomputable def compare_square_roots : Prop :=
  -2 * Real.sqrt 11 > -3 * Real.sqrt 5

theorem comparison_of_neg_square_roots : compare_square_roots :=
by
  -- Omitting the proof details
  sorry

end comparison_of_neg_square_roots_l69_69018


namespace hexagon_area_correct_l69_69488

structure Point where
  x : ℝ
  y : ℝ

def hexagon : List Point := [
  { x := 0, y := 0 },
  { x := 2, y := 4 },
  { x := 6, y := 4 },
  { x := 8, y := 0 },
  { x := 6, y := -4 },
  { x := 2, y := -4 }
]

def area_of_hexagon (hex : List Point) : ℝ :=
  -- Assume a function that calculates the area of a polygon given a list of vertices
  sorry

theorem hexagon_area_correct : area_of_hexagon hexagon = 16 :=
  sorry

end hexagon_area_correct_l69_69488


namespace marty_combination_count_l69_69997

theorem marty_combination_count (num_colors : ℕ) (num_methods : ℕ) 
  (h1 : num_colors = 5) (h2 : num_methods = 4) : 
  num_colors * num_methods = 20 := by
  sorry

end marty_combination_count_l69_69997


namespace candies_share_equally_l69_69153

theorem candies_share_equally (mark_candies : ℕ) (peter_candies : ℕ) (john_candies : ℕ)
  (h_mark : mark_candies = 30) (h_peter : peter_candies = 25) (h_john : john_candies = 35) :
  (mark_candies + peter_candies + john_candies) / 3 = 30 :=
by
  sorry

end candies_share_equally_l69_69153


namespace find_first_number_l69_69135

theorem find_first_number : ∃ x : ℕ, x + 7314 = 3362 + 13500 ∧ x = 9548 :=
by
  -- This is where the proof would go
  sorry

end find_first_number_l69_69135


namespace calories_consummed_l69_69288

-- Definitions based on conditions
def calories_per_strawberry : ℕ := 4
def calories_per_ounce_of_yogurt : ℕ := 17
def strawberries_eaten : ℕ := 12
def yogurt_eaten_in_ounces : ℕ := 6

-- Theorem statement
theorem calories_consummed (c_straw : ℕ) (c_yogurt : ℕ) (straw : ℕ) (yogurt : ℕ) 
  (h1 : c_straw = calories_per_strawberry) 
  (h2 : c_yogurt = calories_per_ounce_of_yogurt) 
  (h3 : straw = strawberries_eaten) 
  (h4 : yogurt = yogurt_eaten_in_ounces) : 
  c_straw * straw + c_yogurt * yogurt = 150 :=
by 
  -- Derived conditions
  rw [h1, h2, h3, h4]
  sorry

end calories_consummed_l69_69288


namespace min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l69_69112

theorem min_n_consecutive_integers_sum_of_digits_is_multiple_of_8 
: ∃ n : ℕ, (∀ (nums : Fin n.succ → ℕ), 
              (∀ i j, i < j → nums i < nums j → nums j = nums i + 1) →
              ∃ i, (nums i) % 8 = 0) ∧ n = 15 := 
sorry

end min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l69_69112


namespace student_attempted_sums_l69_69550

theorem student_attempted_sums (right wrong : ℕ) (h1 : wrong = 2 * right) (h2 : right = 12) : right + wrong = 36 := sorry

end student_attempted_sums_l69_69550


namespace solve_equation_l69_69456

theorem solve_equation :
  ∃ x : ℝ, x = (Real.sqrt (x - 1/x)) + (Real.sqrt (1 - 1/x)) ∧ x = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end solve_equation_l69_69456


namespace line_equation_M_l69_69458

theorem line_equation_M (x y : ℝ) : 
  (∃ c1 m1 : ℝ, m1 = 2 / 3 ∧ c1 = 4 ∧ 
  (∃ m2 c2 : ℝ, m2 = 2 * m1 ∧ c2 = (1 / 2) * c1 ∧ y = m2 * x + c2)) → 
  y = (4 / 3) * x + 2 := 
sorry

end line_equation_M_l69_69458


namespace sequence_formula_l69_69626

theorem sequence_formula (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 3 * a n + 3 ^ n) → 
  ∀ n : ℕ, 0 < n → a n = n * 3 ^ (n - 1) :=
by
  sorry

end sequence_formula_l69_69626


namespace range_of_2a_plus_3b_l69_69923

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 :=
sorry

end range_of_2a_plus_3b_l69_69923


namespace Mickey_less_than_twice_Minnie_l69_69956

def Minnie_horses_per_day : ℕ := 10
def Mickey_horses_per_day : ℕ := 14

theorem Mickey_less_than_twice_Minnie :
  2 * Minnie_horses_per_day - Mickey_horses_per_day = 6 := by
  sorry

end Mickey_less_than_twice_Minnie_l69_69956


namespace chain_of_inequalities_l69_69583

theorem chain_of_inequalities (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  9 / (a + b + c) ≤ (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ∧ 
  (2 / (a + b) + 2 / (b + c) + 2 / (c + a)) ≤ (1 / a + 1 / b + 1 / c) := 
by 
  sorry

end chain_of_inequalities_l69_69583


namespace Nell_initial_cards_l69_69876

theorem Nell_initial_cards (given_away : ℕ) (now_has : ℕ) : 
  given_away = 276 → now_has = 252 → (now_has + given_away) = 528 :=
by
  intros h_given_away h_now_has
  sorry

end Nell_initial_cards_l69_69876


namespace evaluate_operations_l69_69721

def spadesuit (x y : ℝ) := (x + y) * (x - y)
def heartsuit (x y : ℝ) := x ^ 2 - y ^ 2

theorem evaluate_operations : spadesuit 5 (heartsuit 3 2) = 0 :=
by
  sorry

end evaluate_operations_l69_69721


namespace linear_term_coefficient_is_neg_two_l69_69859

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Define the specific quadratic equation
def specific_quadratic_eq (x : ℝ) : Prop :=
  quadratic_eq 1 (-2) (-1) x

-- The statement to prove the coefficient of the linear term
theorem linear_term_coefficient_is_neg_two : ∀ x : ℝ, specific_quadratic_eq x → ∀ a b c : ℝ, quadratic_eq a b c x → b = -2 :=
by
  intros x h_eq a b c h_quadratic_eq
  -- Proof is omitted
  sorry

end linear_term_coefficient_is_neg_two_l69_69859


namespace fourth_number_of_expression_l69_69789

theorem fourth_number_of_expression (x : ℝ) (h : 0.3 * 0.8 + 0.1 * x = 0.29) : x = 0.5 :=
by
  sorry

end fourth_number_of_expression_l69_69789


namespace original_number_is_509_l69_69696

theorem original_number_is_509 (n : ℕ) (h : n - 5 = 504) : n = 509 :=
by {
    sorry
}

end original_number_is_509_l69_69696


namespace right_triangle_angles_ratio_l69_69836

theorem right_triangle_angles_ratio (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3) :
  α = 67.5 ∧ β = 22.5 :=
sorry

end right_triangle_angles_ratio_l69_69836


namespace games_bought_l69_69010

def initial_money : ℕ := 35
def spent_money : ℕ := 7
def cost_per_game : ℕ := 4

theorem games_bought : (initial_money - spent_money) / cost_per_game = 7 := by
  sorry

end games_bought_l69_69010


namespace jacques_suitcase_weight_l69_69192

noncomputable def suitcase_weight_on_return : ℝ := 
  let initial_weight := 12
  let perfume_weight := (5 * 1.2) / 16
  let chocolate_weight := 4 + 1.5 + 3.25
  let soap_weight := (2 * 5) / 16
  let jam_weight := (8 + 6 + 10 + 12) / 16
  let sculpture_weight := 3.5 * 2.20462
  let shirts_weight := (3 * 300 * 0.03527396) / 16
  let cookies_weight := (450 * 0.03527396) / 16
  let wine_weight := (190 * 0.03527396) / 16
  initial_weight + perfume_weight + chocolate_weight + soap_weight + jam_weight + sculpture_weight + shirts_weight + cookies_weight + wine_weight

theorem jacques_suitcase_weight : suitcase_weight_on_return = 35.111288 := 
by 
  -- Calculation to verify that the total is 35.111288
  sorry

end jacques_suitcase_weight_l69_69192


namespace chocolate_bars_per_box_is_25_l69_69159

-- Define the conditions
def total_chocolate_bars : Nat := 400
def total_small_boxes : Nat := 16

-- Define the statement to be proved
def chocolate_bars_per_small_box : Nat := total_chocolate_bars / total_small_boxes

theorem chocolate_bars_per_box_is_25
  (h1 : total_chocolate_bars = 400)
  (h2 : total_small_boxes = 16) :
  chocolate_bars_per_small_box = 25 :=
by
  -- proof will go here
  sorry

end chocolate_bars_per_box_is_25_l69_69159


namespace sum_of_cosines_l69_69632

theorem sum_of_cosines :
  (Real.cos (2 * Real.pi / 7) + Real.cos (4 * Real.pi / 7) + Real.cos (6 * Real.pi / 7) = -1 / 2) := sorry

end sum_of_cosines_l69_69632


namespace triangle_area_and_fraction_of_square_l69_69295

theorem triangle_area_and_fraction_of_square 
  (a b c s : ℕ) 
  (h_triangle : a = 9 ∧ b = 40 ∧ c = 41)
  (h_square : s = 41)
  (h_right_angle : a^2 + b^2 = c^2) :
  let area_triangle := (a * b) / 2
  let area_square := s^2
  let fraction := (a * b) / (2 * s^2)
  area_triangle = 180 ∧ fraction = 180 / 1681 := 
by
  sorry

end triangle_area_and_fraction_of_square_l69_69295


namespace sum_term_S2018_l69_69889

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem sum_term_S2018 :
  ∃ (a S : ℕ → ℤ),
    arithmetic_sequence a ∧ 
    sum_first_n_terms a S ∧ 
    a 1 = -2018 ∧ 
    ((S 2015) / 2015 - (S 2013) / 2013 = 2) ∧ 
    S 2018 = -2018 
:= by
  sorry

end sum_term_S2018_l69_69889


namespace correct_statements_l69_69224

variables {d : ℝ} {S : ℕ → ℝ} {a : ℕ → ℝ}

axiom arithmetic_sequence (n : ℕ) : S n = n * a 1 + (n * (n - 1) / 2) * d

theorem correct_statements (h1 : S 6 = S 12) :
  (S 18 = 0) ∧ (d > 0 → a 6 + a 12 < 0) ∧ (d < 0 → |a 6| > |a 12|) :=
sorry

end correct_statements_l69_69224


namespace no_fermat_in_sequence_l69_69434

def sequence_term (n k : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

def is_fermat_number (a : ℕ) : Prop :=
  ∃ m : ℕ, a = 2^(2^m) + 1

theorem no_fermat_in_sequence (k n : ℕ) (hk : k > 2) (hn : n > 2) :
  ¬ is_fermat_number (sequence_term n k) :=
sorry

end no_fermat_in_sequence_l69_69434


namespace geometric_sequence_fifth_term_l69_69879

variable {a : ℕ → ℝ} (h1 : a 1 = 1) (h4 : a 4 = 8)

theorem geometric_sequence_fifth_term (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) :
  a 5 = 16 :=
sorry

end geometric_sequence_fifth_term_l69_69879


namespace negation_of_existential_prop_l69_69834

theorem negation_of_existential_prop :
  (¬ ∃ (x₀ : ℝ), x₀^2 + x₀ + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_prop_l69_69834


namespace B_completes_remaining_work_in_23_days_l69_69490

noncomputable def A_work_rate : ℝ := 1 / 45
noncomputable def B_work_rate : ℝ := 1 / 40
noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate
noncomputable def work_done_together_in_9_days : ℝ := combined_work_rate * 9
noncomputable def remaining_work : ℝ := 1 - work_done_together_in_9_days
noncomputable def days_B_completes_remaining_work : ℝ := remaining_work / B_work_rate

theorem B_completes_remaining_work_in_23_days :
  days_B_completes_remaining_work = 23 :=
by 
  -- Proof omitted - please fill in the proof steps
  sorry

end B_completes_remaining_work_in_23_days_l69_69490


namespace evaluate_f_neg_a_l69_69455

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem evaluate_f_neg_a (a : ℝ) (h : f a = 1 / 3) : f (-a) = 5 / 3 :=
by sorry

end evaluate_f_neg_a_l69_69455


namespace ratio_of_logs_l69_69852

noncomputable def log_base (b x : ℝ) := (Real.log x) / (Real.log b)

theorem ratio_of_logs (a b : ℝ) 
    (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : log_base 8 a = log_base 18 b)
    (h4 : log_base 18 b = log_base 32 (a + b)) : 
    b / a = (1 + Real.sqrt 5) / 2 :=
by sorry

end ratio_of_logs_l69_69852


namespace negation_of_exists_statement_l69_69858

theorem negation_of_exists_statement :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by
  sorry

end negation_of_exists_statement_l69_69858


namespace union_of_A_and_B_at_m_equals_3_range_of_m_if_A_union_B_equals_A_l69_69246

def set_A : Set ℝ := { x | x^2 - x - 12 ≤ 0 }
def set_B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Statement 1: Prove that when \( m = 3 \), \( A \cup B \) = \( \{ x \mid -3 \leq x \leq 5 \} \).
theorem union_of_A_and_B_at_m_equals_3 : set_A ∪ set_B 3 = { x | -3 ≤ x ∧ x ≤ 5 } :=
sorry

-- Statement 2: Prove that if \( A ∪ B = A \), then the range of \( m \) is \( (-\infty, \frac{5}{2}] \).
theorem range_of_m_if_A_union_B_equals_A (m : ℝ) : (set_A ∪ set_B m = set_A) → m ≤ 5 / 2 :=
sorry

end union_of_A_and_B_at_m_equals_3_range_of_m_if_A_union_B_equals_A_l69_69246


namespace derivative_at_neg_one_l69_69828

-- Define the function f
def f (x : ℝ) : ℝ := x ^ 6

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 6 * x ^ 5

-- The statement we want to prove
theorem derivative_at_neg_one : f' (-1) = -6 := sorry

end derivative_at_neg_one_l69_69828


namespace expression_equals_sqrt2_l69_69428

theorem expression_equals_sqrt2 :
  (1 + Real.pi)^0 + 2 - abs (-3) + 2 * Real.sin (Real.pi / 4) = Real.sqrt 2 := by
  sorry

end expression_equals_sqrt2_l69_69428


namespace solve_for_x_l69_69439

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : (9*x)^18 = (27*x)^9 ↔ x = 1/3 :=
by sorry

end solve_for_x_l69_69439


namespace polynomial_divisibility_l69_69608

open Polynomial

noncomputable def f (n : ℕ) : ℤ[X] :=
  (X + 1) ^ (2 * n + 1) + X ^ (n + 2)

noncomputable def p : ℤ[X] :=
  X ^ 2 + X + 1

theorem polynomial_divisibility (n : ℕ) : p ∣ f n :=
  sorry

end polynomial_divisibility_l69_69608


namespace petals_in_garden_l69_69316

def lilies_count : ℕ := 8
def tulips_count : ℕ := 5
def petals_per_lily : ℕ := 6
def petals_per_tulip : ℕ := 3

def total_petals : ℕ := lilies_count * petals_per_lily + tulips_count * petals_per_tulip

theorem petals_in_garden : total_petals = 63 := by
  sorry

end petals_in_garden_l69_69316


namespace beth_sells_half_of_coins_l69_69810

theorem beth_sells_half_of_coins (x y : ℕ) (h₁ : x = 125) (h₂ : y = 35) : (x + y) / 2 = 80 :=
by
  sorry

end beth_sells_half_of_coins_l69_69810


namespace bus_time_l69_69633

variable (t1 t2 t3 t4 : ℕ)

theorem bus_time
  (h1 : t1 = 25)
  (h2 : t2 = 40)
  (h3 : t3 = 15)
  (h4 : t4 = 10) :
  t1 + t2 + t3 + t4 = 90 := by
  sorry

end bus_time_l69_69633


namespace book_length_l69_69661

theorem book_length (P : ℕ) (h1 : 2323 = (P - 2323) + 90) : P = 4556 :=
by
  sorry

end book_length_l69_69661


namespace area_of_parallelogram_l69_69154

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 12) (h_height : height = 8) :
  base * height = 96 := by
  sorry

end area_of_parallelogram_l69_69154


namespace problem_solution_l69_69628

noncomputable def circles_intersect (m : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), (A ∈ { p | p.1^2 + p.2^2 = 1 }) ∧ (B ∈ { p | p.1^2 + p.2^2 = 1 }) ∧
  (A ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ (B ∈ { p | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0 }) ∧ 
  (dist A B = (4 * Real.sqrt 5) / 5)

theorem problem_solution (m : ℝ) : circles_intersect m ↔ (m = 1 ∨ m = -3) := by
  sorry

end problem_solution_l69_69628


namespace cube_side_length_l69_69771

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (hs : s ≠ 0) : s = 6 :=
by sorry

end cube_side_length_l69_69771


namespace vertical_asymptote_unique_d_values_l69_69605

theorem vertical_asymptote_unique_d_values (d : ℝ) :
  (∃! x : ℝ, ∃ c : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x^2 - 2*x + d) = 0) ↔ (d = 0 ∨ d = -3) := 
sorry

end vertical_asymptote_unique_d_values_l69_69605


namespace greatest_air_conditioning_but_no_racing_stripes_l69_69705

variable (total_cars : ℕ) (no_air_conditioning_cars : ℕ) (at_least_racing_stripes_cars : ℕ)
variable (total_cars_eq : total_cars = 100)
variable (no_air_conditioning_cars_eq : no_air_conditioning_cars = 37)
variable (at_least_racing_stripes_cars_ge : at_least_racing_stripes_cars ≥ 51)

theorem greatest_air_conditioning_but_no_racing_stripes
  (total_cars_eq : total_cars = 100)
  (no_air_conditioning_cars_eq : no_air_conditioning_cars = 37)
  (at_least_racing_stripes_cars_ge : at_least_racing_stripes_cars ≥ 51) :
  ∃ max_air_conditioning_no_racing_stripes : ℕ, max_air_conditioning_no_racing_stripes = 12 :=
by
  sorry

end greatest_air_conditioning_but_no_racing_stripes_l69_69705


namespace Mary_is_2_l69_69445

variable (M J : ℕ)

/-- Given the conditions from the problem, Mary's age can be determined to be 2. -/
theorem Mary_is_2 (h1 : J - 5 = M + 2) (h2 : J = 2 * M + 5) : M = 2 := by
  sorry

end Mary_is_2_l69_69445


namespace expression_values_l69_69155

-- Define the conditions as a predicate
def conditions (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a^2 - b * c = b^2 - a * c ∧ b^2 - a * c = c^2 - a * b

-- The main theorem statement
theorem expression_values (a b c : ℝ) (h : conditions a b c) :
  (∃ x : ℝ, x = (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b)) ∧ (x = 7 / 2 ∨ x = -7)) :=
by
  sorry

end expression_values_l69_69155


namespace put_letters_in_mailboxes_l69_69401

theorem put_letters_in_mailboxes :
  (3:ℕ)^4 = 81 :=
by
  sorry

end put_letters_in_mailboxes_l69_69401


namespace factorial_div_add_two_l69_69948

def factorial (n : ℕ) : ℕ :=
match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

theorem factorial_div_add_two :
  (factorial 50) / (factorial 48) + 2 = 2452 :=
by
  sorry

end factorial_div_add_two_l69_69948


namespace solution_set_of_inequality_l69_69518

theorem solution_set_of_inequality : 
  { x : ℝ | (x + 2) * (1 - x) > 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l69_69518


namespace no_solution_inequality_l69_69329

theorem no_solution_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| < 4 * x - 1 ∧ x < a) ↔ a ≤ (2/3) := by sorry

end no_solution_inequality_l69_69329


namespace area_of_lune_l69_69659

theorem area_of_lune :
  ∃ (A L : ℝ), A = (3/2) ∧ L = 2 ∧
  (Lune_area : ℝ) = (9 * Real.sqrt 3 / 4) - (55 * π / 24) →
  Lune_area = (9 * Real.sqrt 3 / 4) - (55 * π / 24) :=
by
  sorry

end area_of_lune_l69_69659


namespace number_line_distance_l69_69202

theorem number_line_distance (x : ℝ) : |x + 1| = 6 ↔ (x = 5 ∨ x = -7) :=
by
  sorry

end number_line_distance_l69_69202


namespace range_of_m_l69_69304

theorem range_of_m (x m : ℝ) (h1 : -2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2)
                   (h2 : x^2 - 2*x + 1 - m^2 ≤ 0)
                   (h3 : m > 0)
                   (h4 : (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m))
                   (h5 : ¬((x < 1 - m ∨ x > 1 + m) → (x < -2 ∨ x > 10))) :
                   m ≤ 3 :=
by
  sorry

end range_of_m_l69_69304


namespace segment_length_is_15_l69_69350

theorem segment_length_is_15 : 
  ∀ (x : ℝ), 
  ∀ (y1 y2 : ℝ), 
  x = 3 → 
  y1 = 5 → 
  y2 = 20 → 
  abs (y2 - y1) = 15 := by 
sorry

end segment_length_is_15_l69_69350


namespace range_of_x_l69_69358

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) : x > 1/3 :=
by
  sorry

end range_of_x_l69_69358


namespace mary_change_received_l69_69060

def cost_of_adult_ticket : ℝ := 2
def cost_of_child_ticket : ℝ := 1
def discount_first_child : ℝ := 0.5
def discount_second_child : ℝ := 0.75
def discount_third_child : ℝ := 1
def sales_tax_rate : ℝ := 0.08
def amount_paid : ℝ := 20

def total_ticket_cost_before_tax : ℝ :=
  cost_of_adult_ticket + (cost_of_child_ticket * discount_first_child) + 
  (cost_of_child_ticket * discount_second_child) + (cost_of_child_ticket * discount_third_child)

def sales_tax : ℝ :=
  total_ticket_cost_before_tax * sales_tax_rate

def total_ticket_cost_with_tax : ℝ :=
  total_ticket_cost_before_tax + sales_tax

def change_received : ℝ :=
  amount_paid - total_ticket_cost_with_tax

theorem mary_change_received :
  change_received = 15.41 :=
by
  sorry

end mary_change_received_l69_69060


namespace cubic_identity_l69_69934

theorem cubic_identity (a b c : ℝ) 
  (h1 : a + b + c = 12)
  (h2 : ab + ac + bc = 30)
  : a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end cubic_identity_l69_69934


namespace value_of_a8_l69_69469

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → 2 * a n + a (n + 1) = 0

theorem value_of_a8 (a : ℕ → ℝ) (h1 : seq a) (h2 : a 3 = -2) : a 8 = 64 :=
sorry

end value_of_a8_l69_69469


namespace unique_real_solution_for_cubic_l69_69004

theorem unique_real_solution_for_cubic {b : ℝ} :
  (∀ x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) → ∃! x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0)) ↔ b > 3 :=
sorry

end unique_real_solution_for_cubic_l69_69004


namespace matrix_A_to_power_4_l69_69950

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end matrix_A_to_power_4_l69_69950


namespace problem_statement_l69_69429

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def has_minimum_value_at (f : ℝ → ℝ) (a : ℝ) := ∀ x : ℝ, f a ≤ f x
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem problem_statement : is_even_function f4 ∧ has_minimum_value_at f4 0 :=
by
  sorry

end problem_statement_l69_69429


namespace mass_percentage_of_calcium_in_calcium_oxide_l69_69700

theorem mass_percentage_of_calcium_in_calcium_oxide
  (Ca_molar_mass : ℝ)
  (O_molar_mass : ℝ)
  (Ca_mass : Ca_molar_mass = 40.08)
  (O_mass : O_molar_mass = 16.00) :
  ((Ca_molar_mass / (Ca_molar_mass + O_molar_mass)) * 100) = 71.45 :=
by
  sorry

end mass_percentage_of_calcium_in_calcium_oxide_l69_69700


namespace pants_cost_l69_69591

/-- Given:
- 3 skirts with each costing $20.00
- 5 blouses with each costing $15.00
- The total spending is $180.00
- A discount on pants: buy 1 pair get 1 pair 1/2 off

Prove that each pair of pants costs $30.00 before the discount. --/
theorem pants_cost (cost_skirt cost_blouse total_amount : ℤ) (pants_discount: ℚ) (total_cost: ℤ) :
  cost_skirt = 20 ∧ cost_blouse = 15 ∧ total_amount = 180 
  ∧ pants_discount * 2 = 1 
  ∧ total_cost = 3 * cost_skirt + 5 * cost_blouse + 3/2 * pants_discount → 
  pants_discount = 30 := by
  sorry

end pants_cost_l69_69591


namespace proof_average_l69_69953

def average_two (x y : ℚ) : ℚ := (x + y) / 2
def average_three (x y z : ℚ) : ℚ := (x + y + z) / 3

theorem proof_average :
  average_three (2 * average_three 3 2 0) (average_two 0 3) (1 * 3) = 47 / 18 :=
by
  sorry

end proof_average_l69_69953


namespace coefficients_square_sum_l69_69379

theorem coefficients_square_sum (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1000 * x ^ 3 + 27 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 11090 := by
  sorry

end coefficients_square_sum_l69_69379


namespace ball_hits_ground_time_l69_69185

theorem ball_hits_ground_time :
  ∃ t : ℚ, -20 * t^2 + 30 * t + 50 = 0 ∧ t = 5 / 2 :=
sorry

end ball_hits_ground_time_l69_69185


namespace range_alpha_minus_beta_over_2_l69_69163

theorem range_alpha_minus_beta_over_2 (α β : ℝ) (h1 : -π / 2 ≤ α) (h2 : α < β) (h3 : β ≤ π / 2) :
  Set.Ico (-π / 2) 0 = {x : ℝ | ∃ α β : ℝ, -π / 2 ≤ α ∧ α < β ∧ β ≤ π / 2 ∧ x = (α - β) / 2} :=
by
  sorry

end range_alpha_minus_beta_over_2_l69_69163


namespace asha_win_probability_l69_69779

theorem asha_win_probability :
  let P_Lose := (3 : ℚ) / 8
  let P_Tie := (1 : ℚ) / 4
  P_Lose + P_Tie < 1 → 1 - P_Lose - P_Tie = (3 : ℚ) / 8 := 
by
  sorry

end asha_win_probability_l69_69779


namespace purple_tile_cost_correct_l69_69898

-- Definitions of given conditions
def turquoise_cost_per_tile : ℕ := 13
def wall1_area : ℕ := 5 * 8
def wall2_area : ℕ := 7 * 8
def total_area : ℕ := wall1_area + wall2_area
def tiles_per_square_foot : ℕ := 4
def total_tiles_needed : ℕ := total_area * tiles_per_square_foot
def turquoise_total_cost : ℕ := total_tiles_needed * turquoise_cost_per_tile
def savings : ℕ := 768
def purple_total_cost : ℕ := turquoise_total_cost - savings
def purple_cost_per_tile : ℕ := 11

-- Theorem stating the problem
theorem purple_tile_cost_correct :
  purple_total_cost / total_tiles_needed = purple_cost_per_tile :=
sorry

end purple_tile_cost_correct_l69_69898


namespace find_x_plus_y_of_parallel_vectors_l69_69871

theorem find_x_plus_y_of_parallel_vectors 
  (x y : ℝ) 
  (a b : ℝ × ℝ × ℝ)
  (ha : a = (x, 2, -2)) 
  (hb : b = (2, y, 4)) 
  (h_parallel : ∃ k : ℝ, a = k • b) 
  : x + y = -5 := 
by 
  sorry

end find_x_plus_y_of_parallel_vectors_l69_69871


namespace quotient_of_division_l69_69827

theorem quotient_of_division
  (larger smaller : ℕ)
  (h1 : larger - smaller = 1370)
  (h2 : larger = 1626)
  (h3 : ∃ q r, larger = smaller * q + r ∧ r = 15) :
  ∃ q, larger = smaller * q + 15 ∧ q = 6 :=
by
  sorry

end quotient_of_division_l69_69827


namespace sum_of_last_two_digits_l69_69964

-- Definitions based on given conditions
def six_power_twenty_five := 6^25
def fourteen_power_twenty_five := 14^25
def expression := six_power_twenty_five + fourteen_power_twenty_five
def modulo := 100

-- The statement we need to prove
theorem sum_of_last_two_digits : expression % modulo = 0 := by
  sorry

end sum_of_last_two_digits_l69_69964


namespace find_vector_b_l69_69174

def vector_collinear (a b : ℝ × ℝ) : Prop :=
    ∃ k : ℝ, (a.1 = k * b.1 ∧ a.2 = k * b.2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
    a.1 * b.1 + a.2 * b.2

theorem find_vector_b (a b : ℝ × ℝ) (h_collinear : vector_collinear a b) (h_dot : dot_product a b = -10) : b = (-4, 2) :=
    by
        sorry

end find_vector_b_l69_69174


namespace october_profit_condition_l69_69334

noncomputable def calculate_profit (price_reduction : ℝ) : ℝ :=
  (50 - price_reduction) * (500 + 20 * price_reduction)

theorem october_profit_condition (x : ℝ) (h : calculate_profit x = 28000) : x = 10 ∨ x = 15 := 
by
  sorry

end october_profit_condition_l69_69334


namespace sector_area_proof_l69_69106

-- Define the sector with its characteristics
structure sector :=
  (r : ℝ)            -- radius
  (theta : ℝ)        -- central angle

-- Given conditions
def sector_example : sector := {r := 1, theta := 2}

-- Definition of perimeter for a sector
def perimeter (sec : sector) : ℝ :=
  2 * sec.r + sec.theta * sec.r

-- Definition of area for a sector
def area (sec : sector) : ℝ :=
  0.5 * sec.r * (sec.theta * sec.r)

-- Theorem statement based on the problem statement
theorem sector_area_proof (sec : sector) (h1 : perimeter sec = 4) (h2 : sec.theta = 2) : area sec = 1 := 
  sorry

end sector_area_proof_l69_69106


namespace doctor_lindsay_adult_patients_per_hour_l69_69211

def number_of_adult_patients_per_hour (A : ℕ) : Prop :=
  let children_per_hour := 3
  let cost_per_adult := 50
  let cost_per_child := 25
  let daily_income := 2200
  let hours_worked := 8
  let income_per_hour := daily_income / hours_worked
  let income_from_children_per_hour := children_per_hour * cost_per_child
  let income_from_adults_per_hour := A * cost_per_adult
  income_from_adults_per_hour + income_from_children_per_hour = income_per_hour

theorem doctor_lindsay_adult_patients_per_hour : 
  ∃ A : ℕ, number_of_adult_patients_per_hour A ∧ A = 4 :=
sorry

end doctor_lindsay_adult_patients_per_hour_l69_69211


namespace total_promotional_items_l69_69680

def num_calendars : ℕ := 300
def num_date_books : ℕ := 200

theorem total_promotional_items : num_calendars + num_date_books = 500 := by
  sorry

end total_promotional_items_l69_69680


namespace max_diff_real_roots_l69_69960

-- Definitions of the quadratic equations
def eq1 (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq2 (a b c x : ℝ) : Prop := b * x^2 + c * x + a = 0
def eq3 (a b c x : ℝ) : Prop := c * x^2 + a * x + b = 0

-- The proof statement
theorem max_diff_real_roots (a b c : ℝ) (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  ∃ x y : ℝ, eq1 a b c x ∧ eq1 a b c y ∧ eq2 a b c x ∧ eq2 a b c y ∧ eq3 a b c x ∧ eq3 a b c y ∧ 
  abs (x - y) = 0 := sorry

end max_diff_real_roots_l69_69960


namespace walking_time_l69_69855

theorem walking_time (distance_walking_rate : ℕ) 
                     (distance : ℕ)
                     (rest_distance : ℕ) 
                     (rest_time : ℕ) 
                     (total_walking_time : ℕ) : 
  distance_walking_rate = 10 → 
  rest_distance = 10 → 
  rest_time = 7 → 
  distance = 50 → 
  total_walking_time = 328 → 
  total_walking_time = (distance / distance_walking_rate) * 60 + ((distance / rest_distance) - 1) * rest_time :=
by
  sorry

end walking_time_l69_69855


namespace remainder_47_mod_288_is_23_mod_24_l69_69606

theorem remainder_47_mod_288_is_23_mod_24 (m : ℤ) (h : m % 288 = 47) : m % 24 = 23 := 
sorry

end remainder_47_mod_288_is_23_mod_24_l69_69606


namespace mom_tshirts_count_l69_69770

def packages : ℕ := 71
def tshirts_per_package : ℕ := 6

theorem mom_tshirts_count : packages * tshirts_per_package = 426 := by
  sorry

end mom_tshirts_count_l69_69770


namespace value_of_m_l69_69349

theorem value_of_m : 5^2 + 7 = 4^3 + m → m = -32 :=
by
  intro h
  sorry

end value_of_m_l69_69349


namespace geometric_sequence_a2_a4_sum_l69_69068

theorem geometric_sequence_a2_a4_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), (∀ n, a n = a 1 * q ^ (n - 1)) ∧
    (a 2 * a 4 = 9) ∧
    (9 * (a 1 * (1 - q^4) / (1 - q)) = 10 * (a 1 * (1 - q^2) / (1 - q))) ∧
    (a 2 + a 4 = 10) :=
by
  sorry

end geometric_sequence_a2_a4_sum_l69_69068


namespace find_p_minus_q_l69_69127

theorem find_p_minus_q (p q : ℝ) (h : ∀ x, x^2 - 6 * x + q = 0 ↔ (x - p)^2 = 7) : p - q = 1 :=
sorry

end find_p_minus_q_l69_69127


namespace smallest_even_number_of_sum_1194_l69_69921

-- Defining the given condition
def sum_of_three_consecutive_even_numbers (x : ℕ) : Prop :=
  x + (x + 2) + (x + 4) = 1194

-- Stating the theorem to prove the smallest even number
theorem smallest_even_number_of_sum_1194 :
  ∃ x : ℕ, sum_of_three_consecutive_even_numbers x ∧ x = 396 :=
by
  sorry

end smallest_even_number_of_sum_1194_l69_69921


namespace overlapping_area_is_correct_l69_69356

-- Defining the coordinates of the grid points
def topLeft : (ℝ × ℝ) := (0, 2)
def topMiddle : (ℝ × ℝ) := (1.5, 2)
def topRight : (ℝ × ℝ) := (3, 2)
def middleLeft : (ℝ × ℝ) := (0, 1)
def center : (ℝ × ℝ) := (1.5, 1)
def middleRight : (ℝ × ℝ) := (3, 1)
def bottomLeft : (ℝ × ℝ) := (0, 0)
def bottomMiddle : (ℝ × ℝ) := (1.5, 0)
def bottomRight : (ℝ × ℝ) := (3, 0)

-- Defining the vertices of the triangles
def triangle1_points : List (ℝ × ℝ) := [topLeft, middleRight, bottomMiddle]
def triangle2_points : List (ℝ × ℝ) := [bottomLeft, topMiddle, middleRight]

-- Function to calculate the area of a polygon given the vertices -- placeholder here
noncomputable def area_of_overlapped_region (tr1 tr2 : List (ℝ × ℝ)) : ℝ := 
  -- Placeholder for the actual computation of the overlapped area
  1.2

-- Statement to prove
theorem overlapping_area_is_correct : 
  area_of_overlapped_region triangle1_points triangle2_points = 1.2 := sorry

end overlapping_area_is_correct_l69_69356


namespace number_of_terms_is_13_l69_69442

-- Define sum of first three terms
def sum_first_three (a d : ℤ) : ℤ := a + (a + d) + (a + 2 * d)

-- Define sum of last three terms when the number of terms is n
def sum_last_three (a d : ℤ) (n : ℕ) : ℤ := (a + (n - 3) * d) + (a + (n - 2) * d) + (a + (n - 1) * d)

-- Define sum of all terms in the sequence
def sum_all_terms (a d : ℤ) (n : ℕ) : ℤ := n / 2 * (2 * a + (n - 1) * d)

-- Given conditions
def condition_one (a d : ℤ) : Prop := sum_first_three a d = 34
def condition_two (a d : ℤ) (n : ℕ) : Prop := sum_last_three a d n = 146
def condition_three (a d : ℤ) (n : ℕ) : Prop := sum_all_terms a d n = 390

-- Theorem to prove that n = 13
theorem number_of_terms_is_13 (a d : ℤ) (n : ℕ) :
  condition_one a d →
  condition_two a d n →
  condition_three a d n →
  n = 13 :=
by sorry

end number_of_terms_is_13_l69_69442


namespace problem_statement_l69_69411

theorem problem_statement (pi : ℝ) (h : pi = 4 * Real.sin (52 * Real.pi / 180)) :
  (2 * pi * Real.sqrt (16 - pi ^ 2) - 8 * Real.sin (44 * Real.pi / 180)) /
  (Real.sqrt 3 - 2 * Real.sqrt 3 * (Real.sin (22 * Real.pi / 180)) ^ 2) = 8 * Real.sqrt 3 := 
  sorry

end problem_statement_l69_69411


namespace find_y_l69_69149

def custom_op (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem find_y (y : ℤ) (h : custom_op y 10 = 90) : y = 11 :=
by
  sorry

end find_y_l69_69149


namespace rectangle_area_l69_69510

theorem rectangle_area
  (line : ∀ x, 6 = x * x + 4 * x + 3 → x = -2 + Real.sqrt 7 ∨ x = -2 - Real.sqrt 7)
  (shorter_side : ∃ l, l = 2 * Real.sqrt 7 ∧ ∃ s, s = l + 3) :
  ∃ a, a = 28 + 12 * Real.sqrt 7 :=
by
  sorry

end rectangle_area_l69_69510


namespace most_noteworthy_figure_is_mode_l69_69666

-- Define the types of possible statistics
inductive Statistic
| Median
| Mean
| Mode
| WeightedMean

-- Define a structure for survey data (details abstracted)
structure SurveyData where
  -- fields abstracted for this problem

-- Define the concept of the most noteworthy figure
def most_noteworthy_figure (data : SurveyData) : Statistic :=
  Statistic.Mode

-- Theorem to prove the most noteworthy figure in a survey's data is the mode
theorem most_noteworthy_figure_is_mode (data : SurveyData) :
  most_noteworthy_figure data = Statistic.Mode :=
by
  sorry

end most_noteworthy_figure_is_mode_l69_69666


namespace parabola_focus_distance_l69_69677

theorem parabola_focus_distance
  (p : ℝ) (h : p > 0)
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 = 3 - p / 2) 
  (h2 : x2 = 2 - p / 2)
  (h3 : y1^2 = 2 * p * x1)
  (h4 : y2^2 = 2 * p * x2)
  (h5 : y1^2 / y2^2 = x1 / x2) : 
  p = 12 / 5 := 
sorry

end parabola_focus_distance_l69_69677


namespace existence_of_E_l69_69706

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 6) + (y^2 / 2) = 1

def point_on_x_axis (E : ℝ × ℝ) : Prop := E.snd = 0

def ea_dot_eb_constant (E A B : ℝ × ℝ) : ℝ :=
  let ea := (A.fst - E.fst, A.snd)
  let eb := (B.fst - E.fst, B.snd)
  ea.fst * eb.fst + ea.snd * eb.snd

noncomputable def E : ℝ × ℝ := (7/3, 0)

noncomputable def const_value : ℝ := (-5/9)

theorem existence_of_E :
  (∃ E, point_on_x_axis E ∧
        (∀ A B, ellipse_eq A.fst A.snd ∧ ellipse_eq B.fst B.snd →
                  ea_dot_eb_constant E A B = const_value)) :=
  sorry

end existence_of_E_l69_69706


namespace platform_length_150_l69_69577

def speed_kmph : ℕ := 54  -- Speed in km/hr

def speed_mps : ℚ := speed_kmph * 1000 / 3600  -- Speed in m/s

def time_pass_man : ℕ := 20  -- Time to pass a man in seconds
def time_pass_platform : ℕ := 30  -- Time to pass a platform in seconds

def length_train : ℚ := speed_mps * time_pass_man  -- Length of the train in meters

def length_platform (P : ℚ) : Prop :=
  length_train + P = speed_mps * time_pass_platform  -- The condition involving platform length

theorem platform_length_150 :
  length_platform 150 := by
  -- We would provide a proof here.
  sorry

end platform_length_150_l69_69577


namespace range_of_a_l69_69131

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end range_of_a_l69_69131


namespace sum_of_coeffs_l69_69943

theorem sum_of_coeffs (x y : ℤ) : (x - 3 * y) ^ 20 = 2 ^ 20 := by
  sorry

end sum_of_coeffs_l69_69943


namespace no_hikers_in_morning_l69_69416

-- Given Conditions
def morning_rowers : ℕ := 13
def afternoon_rowers : ℕ := 21
def total_rowers : ℕ := 34

-- Statement to be proven
theorem no_hikers_in_morning : (total_rowers - afternoon_rowers = morning_rowers) →
                              (total_rowers - afternoon_rowers = morning_rowers) →
                              0 = 34 - 21 - morning_rowers :=
by
  intros h1 h2
  sorry

end no_hikers_in_morning_l69_69416


namespace total_number_of_fish_l69_69499

theorem total_number_of_fish
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (blue_spotted_fish : ℕ)
  (h1 : blue_fish = total_fish / 3)
  (h2 : blue_spotted_fish = blue_fish / 2)
  (h3 : blue_spotted_fish = 10) :
  total_fish = 60 :=
by
  sorry

end total_number_of_fish_l69_69499


namespace johns_donation_is_correct_l69_69715

/-
Conditions:
1. Alice, Bob, and Carol donated different amounts.
2. The ratio of Alice's, Bob's, and Carol's donations is 3:2:5.
3. The sum of Alice's and Bob's donations is $120.
4. The average contribution increases by 50% and reaches $75 per person after John donates.

The statement to prove:
John's donation is $240.
-/

def donations_ratio : ℕ × ℕ × ℕ := (3, 2, 5)
def sum_Alice_Bob : ℕ := 120
def new_avg_after_john : ℕ := 75
def num_people_before_john : ℕ := 3
def avg_increase_factor : ℚ := 1.5

theorem johns_donation_is_correct (A B C J : ℕ) 
  (h1 : A * 2 = B * 3) 
  (h2 : B * 5 = C * 2) 
  (h3 : A + B = sum_Alice_Bob) 
  (h4 : (A + B + C) / num_people_before_john = 80) 
  (h5 : ((A + B + C + J) / (num_people_before_john + 1)) = new_avg_after_john) :
  J = 240 := 
sorry

end johns_donation_is_correct_l69_69715


namespace find_f_pi_over_4_l69_69108

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem find_f_pi_over_4
  (ω φ : ℝ)
  (hω_gt_0 : ω > 0)
  (hφ_lt_pi_over_2 : |φ| < Real.pi / 2)
  (h_mono_dec : ∀ x₁ x₂, (Real.pi / 6 < x₁ ∧ x₁ < Real.pi / 3 ∧ Real.pi / 3 < x₂ ∧ x₂ < 2 * Real.pi / 3) → f x₁ ω φ > f x₂ ω φ)
  (h_values_decreasing : f (Real.pi / 6) ω φ = 1 ∧ f (2 * Real.pi / 3) ω φ = -1) : 
  f (Real.pi / 4) 2 (Real.pi / 6) = Real.sqrt 3 / 2 :=
sorry

end find_f_pi_over_4_l69_69108


namespace arithmetic_mean_l69_69065

theorem arithmetic_mean (x y : ℝ) (h1 : x = Real.sqrt 2 - 1) (h2 : y = 1 / (Real.sqrt 2 - 1)) :
  (x + y) / 2 = Real.sqrt 2 := sorry

end arithmetic_mean_l69_69065


namespace sum_base_6_l69_69479

-- Define base 6 numbers
def n1 : ℕ := 1 * 6^3 + 4 * 6^2 + 5 * 6^1 + 2 * 6^0
def n2 : ℕ := 2 * 6^3 + 3 * 6^2 + 5 * 6^1 + 4 * 6^0

-- Define the expected result in base 6
def expected_sum : ℕ := 4 * 6^3 + 2 * 6^2 + 5 * 6^1 + 0 * 6^0

-- The theorem to prove
theorem sum_base_6 : n1 + n2 = expected_sum := by
    sorry

end sum_base_6_l69_69479


namespace minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l69_69610

theorem minimum_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  x + 2 * y ≥ 19 + 6 * Real.sqrt 2 := 
sorry

theorem minimum_value_x_add_2y_achieved (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 9/y = 1) : 
  ∃ x y, 0 < x ∧ 0 < y ∧ 1/x + 9/y = 1 ∧ x + 2 * y = 19 + 6 * Real.sqrt 2 :=
sorry

end minimum_value_x_add_2y_minimum_value_x_add_2y_achieved_l69_69610


namespace problem1_problem2_problem3_l69_69502

theorem problem1 : 128 + 52 / 13 = 132 :=
by
  sorry

theorem problem2 : 132 / 11 * 29 - 178 = 170 :=
by
  sorry

theorem problem3 : 45 * (320 / (4 * 5)) = 720 :=
by
  sorry

end problem1_problem2_problem3_l69_69502


namespace stratified_sampling_household_l69_69241

/-
  Given:
  - Total valid questionnaires: 500,000.
  - Number of people who purchased:
    - clothing, shoes, and hats: 198,000,
    - household goods: 94,000,
    - cosmetics: 116,000,
    - home appliances: 92,000.
  - Number of questionnaires selected from the "cosmetics" category: 116.
  
  Prove:
  - The number of questionnaires that should be selected from the "household goods" category is 94.
-/

theorem stratified_sampling_household (total_valid: ℕ)
  (clothing_shoes_hats: ℕ)
  (household_goods: ℕ)
  (cosmetics: ℕ)
  (home_appliances: ℕ)
  (sample_cosmetics: ℕ) :
  total_valid = 500000 →
  clothing_shoes_hats = 198000 →
  household_goods = 94000 →
  cosmetics = 116000 →
  home_appliances = 92000 →
  sample_cosmetics = 116 →
  (116 * household_goods = sample_cosmetics * cosmetics) →
  116 * 94000 = 116 * 116000 →
  94000 = 116000 →
  94 = 94 := by
  intros
  sorry

end stratified_sampling_household_l69_69241


namespace veranda_area_correct_l69_69199

-- Definitions of the room dimensions and veranda width
def room_length : ℝ := 18
def room_width : ℝ := 12
def veranda_width : ℝ := 2

-- Definition of the total length including veranda
def total_length : ℝ := room_length + 2 * veranda_width

-- Definition of the total width including veranda
def total_width : ℝ := room_width + 2 * veranda_width

-- Definition of the area of the entire space (room plus veranda)
def area_entire_space : ℝ := total_length * total_width

-- Definition of the area of the room
def area_room : ℝ := room_length * room_width

-- Definition of the area of the veranda
def area_veranda : ℝ := area_entire_space - area_room

-- Theorem statement to prove the area of the veranda
theorem veranda_area_correct : area_veranda = 136 := 
by
  sorry

end veranda_area_correct_l69_69199


namespace find_point_C_find_area_triangle_ABC_l69_69368

noncomputable section

-- Given points and equations
def point_B : ℝ × ℝ := (4, 4)
def eq_angle_bisector : ℝ × ℝ → Prop := λ p => p.2 = 0
def eq_altitude : ℝ × ℝ → Prop := λ p => p.1 - 2 * p.2 + 2 = 0

-- Target coordinates of point C
def point_C : ℝ × ℝ := (10, -8)

-- Coordinates of point A derived from given conditions
def point_A : ℝ × ℝ := (-2, 0)

-- Line equations derived from conditions
def eq_line_BC : ℝ × ℝ → Prop := λ p => 2 * p.1 + p.2 - 12 = 0
def eq_line_AC : ℝ × ℝ → Prop := λ p => 2 * p.1 + 3 * p.2 + 4 = 0

-- Prove the coordinates of point C
theorem find_point_C : ∃ C : ℝ × ℝ, eq_line_BC C ∧ eq_line_AC C ∧ C = point_C := by
  sorry

-- Prove the area of triangle ABC.
theorem find_area_triangle_ABC : ∃ S : ℝ, S = 48 := by
  sorry

end find_point_C_find_area_triangle_ABC_l69_69368


namespace area_of_rhombus_l69_69572

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 22) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 330 :=
by
  rw [h1, h2]
  norm_num

-- Here we state the theorem about the area of the rhombus given its diagonal lengths.

end area_of_rhombus_l69_69572


namespace mass_15_implies_age_7_l69_69158

-- Define the mass function m which depends on age a
variable (m : ℕ → ℕ)

-- Define the condition for the mass to be 15 kg
def is_age_when_mass_is_15 (a : ℕ) : Prop :=
  m a = 15

-- The problem statement to be proven
theorem mass_15_implies_age_7 : ∀ a, is_age_when_mass_is_15 m a → a = 7 :=
by
  -- Proof details would follow here
  sorry

end mass_15_implies_age_7_l69_69158


namespace Samantha_last_name_length_l69_69515

theorem Samantha_last_name_length :
  ∃ (S B : ℕ), S = B - 3 ∧ B - 2 = 2 * 4 ∧ S = 7 :=
by
  sorry

end Samantha_last_name_length_l69_69515


namespace simplify_expression_l69_69987

variable (x y : ℝ)

theorem simplify_expression : 3 * x^2 * y * (2 / (9 * x^3 * y)) = 2 / (3 * x) :=
by sorry

end simplify_expression_l69_69987


namespace remaining_strawberries_l69_69314

-- Define the constants based on conditions
def initial_kg1 : ℕ := 3
def initial_g1 : ℕ := 300
def given_kg1 : ℕ := 1
def given_g1 : ℕ := 900

-- Define the conversion from kilograms to grams
def kg_to_g (kg : ℕ) : ℕ := kg * 1000

-- Calculate initial total grams
def initial_total_g : ℕ := kg_to_g initial_kg1 + initial_g1

-- Calculate given total grams
def given_total_g : ℕ := kg_to_g given_kg1 + given_g1

-- Define the remaining grams
def remaining_g (initial_g : ℕ) (given_g : ℕ) : ℕ := initial_g - given_g

-- Statement to prove
theorem remaining_strawberries : remaining_g initial_total_g given_total_g = 1400 := by
sorry

end remaining_strawberries_l69_69314


namespace initial_investments_l69_69839

theorem initial_investments (x y : ℝ) : 
  -- Conditions
  5000 = y + (5000 - y) ∧
  (y * (1 + x / 100) = 2100) ∧
  ((5000 - y) * (1 + (x + 1) / 100) = 3180) →
  -- Conclusion
  y = 2000 ∧ (5000 - y) = 3000 := 
by 
  sorry

end initial_investments_l69_69839


namespace product_g_roots_l69_69222

noncomputable def f (x : ℝ) : ℝ := x^4 - x^3 + x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 3

theorem product_g_roots (x_1 x_2 x_3 x_4 : ℝ) (hx : ∀ x, (x = x_1 ∨ x = x_2 ∨ x = x_3 ∨ x = x_4) ↔ f x = 0) :
  g x_1 * g x_2 * g x_3 * g x_4 = 142 :=
by sorry

end product_g_roots_l69_69222


namespace probability_after_first_new_draw_is_five_ninths_l69_69970

-- Defining the conditions in Lean
def total_balls : ℕ := 10
def new_balls : ℕ := 6
def old_balls : ℕ := 4

def balls_remaining_after_first_draw : ℕ := total_balls - 1
def new_balls_after_first_draw : ℕ := new_balls - 1

-- Using the classic probability definition
def probability_of_drawing_second_new_ball := (new_balls_after_first_draw : ℚ) / (balls_remaining_after_first_draw : ℚ)

-- Stating the theorem to be proved
theorem probability_after_first_new_draw_is_five_ninths :
  probability_of_drawing_second_new_ball = 5/9 := sorry

end probability_after_first_new_draw_is_five_ninths_l69_69970


namespace number_of_teams_l69_69152

theorem number_of_teams (n : ℕ) (G : ℕ) (h1 : G = 28) (h2 : G = n * (n - 1) / 2) : n = 8 := 
  by
  -- Proof skipped
  sorry

end number_of_teams_l69_69152


namespace domain_ln_x_plus_one_l69_69557

theorem domain_ln_x_plus_one : 
  { x : ℝ | ∃ y : ℝ, y = x + 1 ∧ y > 0 } = { x : ℝ | x > -1 } :=
by
  sorry

end domain_ln_x_plus_one_l69_69557


namespace area_of_triangle_pqr_l69_69946

noncomputable def area_of_triangle (P Q R : ℝ) : ℝ :=
  let PQ := P + Q
  let PR := P + R
  let QR := Q + R
  if PQ^2 = PR^2 + QR^2 then
    1 / 2 * PR * QR
  else
    0

theorem area_of_triangle_pqr : 
  area_of_triangle 3 2 1 = 6 :=
by
  simp [area_of_triangle]
  sorry

end area_of_triangle_pqr_l69_69946


namespace necessary_not_sufficient_condition_l69_69398

theorem necessary_not_sufficient_condition (x : ℝ) : (x < 2) → (x^2 - x - 2 < 0) :=
by {
  sorry
}

end necessary_not_sufficient_condition_l69_69398


namespace curve_distance_bound_l69_69595

/--
Given the point A on the curve y = e^x and point B on the curve y = ln(x),
prove that |AB| >= a always holds if and only if a <= sqrt(2).
-/
theorem curve_distance_bound {A B : ℝ × ℝ} (a : ℝ)
  (hA : A.2 = Real.exp A.1) (hB : B.2 = Real.log B.1) :
  (dist A B ≥ a) ↔ (a ≤ Real.sqrt 2) :=
sorry

end curve_distance_bound_l69_69595


namespace inequality_proof_l69_69698

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a - b + c) * (1 / a - 1 / b + 1 / c) ≥ 1 :=
by
  sorry

end inequality_proof_l69_69698


namespace find_x_l69_69145

theorem find_x 
  (x y z : ℝ)
  (h1 : (20 + 40 + 60 + x) / 4 = (10 + 70 + y + z) / 4 + 9)
  (h2 : y + z = 110) 
  : x = 106 := 
by 
  sorry

end find_x_l69_69145


namespace jose_peanuts_l69_69925

def kenya_peanuts : Nat := 133
def difference_peanuts : Nat := 48

theorem jose_peanuts : (kenya_peanuts - difference_peanuts) = 85 := by
  sorry

end jose_peanuts_l69_69925


namespace correct_option_l69_69885

def M : Set ℝ := { x | x^2 - 4 = 0 }

theorem correct_option : -2 ∈ M :=
by
  -- Definitions and conditions from the problem
  -- Set M is defined as the set of all x such that x^2 - 4 = 0
  have hM : M = { x | x^2 - 4 = 0 } := rfl
  -- Goal is to show that -2 belongs to the set M
  sorry

end correct_option_l69_69885


namespace lower_upper_bound_f_l69_69841

-- definition of the function f(n, d) as given in the problem
def func_f (n : ℕ) (d : ℕ) : ℕ :=
  -- placeholder definition; actual definition would rely on the described properties
  sorry

theorem lower_upper_bound_f (n d : ℕ) (hn : 0 < n) (hd : 0 < d) :
  (n-1) * 2^d + 1 ≤ func_f n d ∧ func_f n d ≤ (n-1) * n^d + 1 :=
by
  sorry

end lower_upper_bound_f_l69_69841


namespace anne_find_bottle_caps_l69_69378

theorem anne_find_bottle_caps 
  (n_i n_f : ℕ) (h_initial : n_i = 10) (h_final : n_f = 15) : n_f - n_i = 5 :=
by
  sorry

end anne_find_bottle_caps_l69_69378


namespace total_profit_is_27_l69_69344

noncomputable def total_profit : ℕ :=
  let natasha_money := 60
  let carla_money := natasha_money / 3
  let cosima_money := carla_money / 2
  let sergio_money := 3 * cosima_money / 2

  let natasha_spent := 4 * 15
  let carla_spent := 6 * 10
  let cosima_spent := 5 * 8
  let sergio_spent := 3 * 12

  let natasha_profit := natasha_spent * 10 / 100
  let carla_profit := carla_spent * 15 / 100
  let cosima_profit := cosima_spent * 12 / 100
  let sergio_profit := sergio_spent * 20 / 100

  natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_is_27 : total_profit = 27 := by
  sorry

end total_profit_is_27_l69_69344


namespace usual_time_is_49_l69_69600

variable (R T : ℝ)
variable (h1 : R > 0) -- Usual rate is positive
variable (h2 : T > 0) -- Usual time is positive
variable (condition : T * R = (T - 7) * (7 / 6 * R)) -- Main condition derived from the problem

theorem usual_time_is_49 (h1 : R > 0) (h2 : T > 0) (condition : T * R = (T - 7) * (7 / 6 * R)) : T = 49 := by
  sorry -- Proof goes here

end usual_time_is_49_l69_69600


namespace cloth_gain_representation_l69_69844

theorem cloth_gain_representation (C S : ℝ) (h1 : S = 1.20 * C) (h2 : ∃ gain, gain = 60 * S - 60 * C) :
  ∃ meters : ℝ, meters = (60 * S - 60 * C) / S ∧ meters = 12 :=
by
  sorry

end cloth_gain_representation_l69_69844


namespace abs_eq_sqrt_five_l69_69744

theorem abs_eq_sqrt_five (x : ℝ) (h : |x| = Real.sqrt 5) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := 
sorry

end abs_eq_sqrt_five_l69_69744


namespace problem_solution_correct_l69_69556

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x = 1

def proposition_q : Prop :=
  {x : ℝ | x^2 - 3 * x + 2 < 0} = {x : ℝ | 1 < x ∧ x < 2}

theorem problem_solution_correct :
  (proposition_p ∧ proposition_q) ∧
  (proposition_p ∧ ¬proposition_q) = false ∧
  (¬proposition_p ∨ proposition_q) ∧
  (¬proposition_p ∨ ¬proposition_q) = false :=
by
  sorry

end problem_solution_correct_l69_69556


namespace gcd_polynomial_l69_69276

-- Given definitions based on the conditions
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

-- Given the conditions: a is a multiple of 1610
variables (a : ℕ) (h : is_multiple_of a 1610)

-- Main theorem: Prove that gcd(a^2 + 9a + 35, a + 5) = 15
theorem gcd_polynomial (h : is_multiple_of a 1610) : gcd (a^2 + 9*a + 35) (a + 5) = 15 :=
sorry

end gcd_polynomial_l69_69276


namespace total_balls_donated_l69_69549

def num_elem_classes_A := 4
def num_middle_classes_A := 5
def num_elem_classes_B := 5
def num_middle_classes_B := 3
def num_elem_classes_C := 6
def num_middle_classes_C := 4
def balls_per_class := 5

theorem total_balls_donated :
  (num_elem_classes_A + num_middle_classes_A) * balls_per_class +
  (num_elem_classes_B + num_middle_classes_B) * balls_per_class +
  (num_elem_classes_C + num_middle_classes_C) * balls_per_class =
  135 :=
by
  sorry

end total_balls_donated_l69_69549


namespace quadratic_intersection_l69_69058

theorem quadratic_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersection_l69_69058


namespace find_values_of_x_and_y_l69_69235

theorem find_values_of_x_and_y (x y : ℝ) :
  (2.5 * x = y^2 + 43) ∧ (2.1 * x = y^2 - 12) → (x = 137.5 ∧ y = Real.sqrt 300.75) :=
by
  sorry

end find_values_of_x_and_y_l69_69235


namespace remainder_when_divided_by_x_plus_2_l69_69132

variable (D E F : ℝ)

def q (x : ℝ) := D * x^4 + E * x^2 + F * x + 7

theorem remainder_when_divided_by_x_plus_2 :
  q D E F (-2) = 21 - 2 * F :=
by
  have hq2 : q D E F 2 = 21 := sorry
  sorry

end remainder_when_divided_by_x_plus_2_l69_69132


namespace product_value_l69_69765

theorem product_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 = 81 :=
  sorry

end product_value_l69_69765


namespace harry_total_travel_time_l69_69732

def bus_time_already_sitting : Nat := 15
def bus_time_remaining : Nat := 25
def walk_fraction := 1 / 2

def bus_time_total : Nat := bus_time_already_sitting + bus_time_remaining
def walk_time : Nat := bus_time_total * walk_fraction

theorem harry_total_travel_time : bus_time_total + walk_time = 60 := by
  sorry

end harry_total_travel_time_l69_69732


namespace area_of_square_inscribed_in_circle_l69_69212

theorem area_of_square_inscribed_in_circle (a : ℝ) :
  ∃ S : ℝ, S = (2 * a^2) / 3 :=
sorry

end area_of_square_inscribed_in_circle_l69_69212


namespace sarah_friends_apples_l69_69735

-- Definitions of initial conditions
def initial_apples : ℕ := 25
def left_apples : ℕ := 3
def apples_given_teachers : ℕ := 16
def apples_eaten : ℕ := 1

-- Theorem that states the number of friends who received apples
theorem sarah_friends_apples :
  (initial_apples - left_apples - apples_given_teachers - apples_eaten = 5) :=
by
  sorry

end sarah_friends_apples_l69_69735


namespace Mike_onions_grew_l69_69137

-- Define the data:
variables (nancy_onions dan_onions total_onions mike_onions : ℕ)

-- Conditions:
axiom Nancy_onions_grew : nancy_onions = 2
axiom Dan_onions_grew : dan_onions = 9
axiom Total_onions_grew : total_onions = 15

-- Theorem to prove:
theorem Mike_onions_grew (h : total_onions = nancy_onions + dan_onions + mike_onions) : mike_onions = 4 :=
by
  -- The proof is not provided, so we use sorry:
  sorry

end Mike_onions_grew_l69_69137


namespace combined_sum_correct_l69_69870

-- Define the sum of integers in a range
def sum_of_integers (a b : Int) : Int := (b - a + 1) * (a + b) / 2

-- Define the sum of squares of integers in a range
def sum_of_squares (a b : Int) : Int :=
  let sum_sq (n : Int) : Int := n * (n + 1) * (2 * n + 1) / 6
  sum_sq b - sum_sq (a - 1)

-- Define the combined sum function
def combined_sum (a b c d : Int) : Int :=
  sum_of_integers a b + sum_of_squares c d

-- Theorem statement: Prove the combined sum of integers from -50 to 40 and squares of integers from 10 to 40 is 21220
theorem combined_sum_correct :
  combined_sum (-50) 40 10 40 = 21220 :=
by
  -- leaving the proof as a sorry
  sorry

end combined_sum_correct_l69_69870


namespace final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l69_69424

-- Define the travel distances
def travel_distances : List ℤ := [17, -9, 7, 11, -15, -3]

-- Define the fuel consumption rate
def fuel_consumption_rate : ℝ := 0.08

-- Theorem stating the final position
theorem final_position_is_east_8km :
  List.sum travel_distances = 8 :=
by
  sorry

-- Theorem stating the total fuel consumption
theorem total_fuel_consumption_is_4_96liters :
  (List.sum (travel_distances.map fun x => |x| : List ℝ)) * fuel_consumption_rate = 4.96 :=
by
  sorry

end final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l69_69424


namespace frog_jump_plan_l69_69423

-- Define the vertices of the hexagon
inductive Vertex
| A | B | C | D | E | F

open Vertex

-- Define adjacency in the regular hexagon
def adjacent (v1 v2 : Vertex) : Prop :=
  match v1, v2 with
  | A, B | A, F | B, C | B, A | C, D | C, B | D, E | D, C | E, F | E, D | F, A | F, E => true
  | _, _ => false

-- Define the problem
def frog_jump_sequences_count : ℕ :=
  26

theorem frog_jump_plan : frog_jump_sequences_count = 26 := 
  sorry

end frog_jump_plan_l69_69423


namespace find_value_of_expression_l69_69578

theorem find_value_of_expression (m n : ℝ) 
  (h1 : m^2 + 2 * m * n = 3) 
  (h2 : m * n + n^2 = 4) : 
  m^2 + 3 * m * n + n^2 = 7 := 
by
  sorry

end find_value_of_expression_l69_69578


namespace linear_function_solution_l69_69043

theorem linear_function_solution (k : ℝ) (h₁ : k ≠ 0) (h₂ : 0 = k * (-2) + 3) :
  ∃ x : ℝ, k * (x - 5) + 3 = 0 ∧ x = 3 :=
by
  sorry

end linear_function_solution_l69_69043


namespace parallel_lines_solution_l69_69900

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y + 1 = 0 → 2 * x + a * y + 2 = 0 → (a = 1 ∨ a = -2)) :=
by
  sorry

end parallel_lines_solution_l69_69900


namespace total_broken_marbles_l69_69085

theorem total_broken_marbles (marbles_set1 marbles_set2 : ℕ) 
  (percentage_broken_set1 percentage_broken_set2 : ℚ) 
  (h1 : marbles_set1 = 50) 
  (h2 : percentage_broken_set1 = 0.1) 
  (h3 : marbles_set2 = 60) 
  (h4 : percentage_broken_set2 = 0.2) : 
  (marbles_set1 * percentage_broken_set1 + marbles_set2 * percentage_broken_set2 = 17) := 
by 
  sorry

end total_broken_marbles_l69_69085


namespace complementary_supplementary_angle_l69_69321

theorem complementary_supplementary_angle (x : ℝ) :
  (90 - x) * 3 = 180 - x → x = 45 :=
by 
  intro h
  sorry

end complementary_supplementary_angle_l69_69321


namespace num_of_negative_x_l69_69478

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l69_69478


namespace glass_cannot_all_be_upright_l69_69055

def glass_flip_problem :=
  ∀ (g : Fin 6 → ℤ),
    g 0 = 1 ∧ g 1 = 1 ∧ g 2 = 1 ∧ g 3 = 1 ∧ g 4 = 1 ∧ g 5 = -1 →
    (∀ (flip : Fin 4 → Fin 6 → ℤ),
      (∃ (i1 i2 i3 i4: Fin 6), 
        flip 0 = g i1 * -1 ∧ 
        flip 1 = g i2 * -1 ∧
        flip 2 = g i3 * -1 ∧
        flip 3 = g i4 * -1) →
      ∃ j, g j ≠ 1)

theorem glass_cannot_all_be_upright : glass_flip_problem :=
  sorry

end glass_cannot_all_be_upright_l69_69055


namespace third_student_number_l69_69188

theorem third_student_number (A B C D : ℕ) 
  (h1 : A + B + C + D = 531) 
  (h2 : A + B = C + D + 31) 
  (h3 : C = D + 22) : 
  C = 136 := 
by
  sorry

end third_student_number_l69_69188


namespace find_xy_pairs_l69_69692

theorem find_xy_pairs (x y: ℝ) :
  x + y + 4 = (12 * x + 11 * y) / (x ^ 2 + y ^ 2) ∧
  y - x + 3 = (11 * x - 12 * y) / (x ^ 2 + y ^ 2) ↔
  (x = 2 ∧ y = 1) ∨ (x = -2.5 ∧ y = -4.5) :=
by
  sorry

end find_xy_pairs_l69_69692


namespace cows_number_l69_69801

theorem cows_number (D C : ℕ) (L H : ℕ) 
  (h1 : L = 2 * D + 4 * C)
  (h2 : H = D + C)
  (h3 : L = 2 * H + 12) 
  : C = 6 := 
by
  sorry

end cows_number_l69_69801


namespace ray_walks_to_park_l69_69646

theorem ray_walks_to_park (x : ℤ) (h1 : 3 * (x + 7 + 11) = 66) : x = 4 :=
by
  -- solving steps are skipped
  sorry

end ray_walks_to_park_l69_69646


namespace proof_problem_l69_69120

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))

theorem proof_problem (
  a : ℝ
) (h1 : a > 1) :
  (∀ x, f a x = (a / (a^2 - 1)) * (Real.exp (Real.log a * x) - Real.exp (-Real.log a * x))) ∧
  (∀ x, f a (-x) = -f a x) ∧
  (∀ x1 x2, x1 < x2 → f a x1 < f a x2) ∧
  (∀ m, -1 < 1 - m ∧ 1 - m < m^2 - 1 ∧ m^2 - 1 < 1 → 1 < m ∧ m < Real.sqrt 2)
  :=
sorry

end proof_problem_l69_69120


namespace sin_of_alpha_l69_69893

theorem sin_of_alpha 
  (α : ℝ) 
  (h : Real.cos (α - Real.pi / 2) = 1 / 3) : 
  Real.sin α = 1 / 3 := 
by 
  sorry

end sin_of_alpha_l69_69893


namespace youngest_child_age_l69_69718

theorem youngest_child_age (x : ℝ) (h : x + (x + 1) + (x + 2) + (x + 3) = 12) : x = 1.5 :=
by sorry

end youngest_child_age_l69_69718


namespace hexagon_side_length_l69_69547

theorem hexagon_side_length (p : ℕ) (s : ℕ) (h₁ : p = 24) (h₂ : s = 6) : p / s = 4 := by
  sorry

end hexagon_side_length_l69_69547


namespace frequency_of_group_5_l69_69389

theorem frequency_of_group_5 (total_students freq1 freq2 freq3 freq4 : ℕ)
  (h_total: total_students = 50) 
  (h_freq1: freq1 = 7) 
  (h_freq2: freq2 = 12) 
  (h_freq3: freq3 = 13) 
  (h_freq4: freq4 = 8) :
  (50 - (7 + 12 + 13 + 8)) / 50 = 0.2 :=
by
  sorry

end frequency_of_group_5_l69_69389


namespace survey_representative_l69_69669

universe u

inductive SurveyOption : Type u
| A : SurveyOption  -- Selecting a class of students
| B : SurveyOption  -- Selecting 50 male students
| C : SurveyOption  -- Selecting 50 female students
| D : SurveyOption  -- Randomly selecting 50 eighth-grade students

def most_appropriate_survey : SurveyOption := SurveyOption.D

theorem survey_representative : most_appropriate_survey = SurveyOption.D := 
by sorry

end survey_representative_l69_69669


namespace tan_theta_l69_69048

theorem tan_theta (θ : ℝ) (h : Real.sin (θ / 2) - 2 * Real.cos (θ / 2) = 0) : Real.tan θ = -4 / 3 :=
sorry

end tan_theta_l69_69048


namespace total_amount_of_check_l69_69052

def numParts : Nat := 59
def price50DollarPart : Nat := 50
def price20DollarPart : Nat := 20
def num50DollarParts : Nat := 40

theorem total_amount_of_check : (num50DollarParts * price50DollarPart + (numParts - num50DollarParts) * price20DollarPart) = 2380 := by
  sorry

end total_amount_of_check_l69_69052


namespace slope_of_line_in_terms_of_angle_l69_69210

variable {x y : ℝ}

theorem slope_of_line_in_terms_of_angle (h : 2 * Real.sqrt 3 * x - 2 * y - 1 = 0) :
    ∃ α : ℝ, 0 ≤ α ∧ α < Real.pi ∧ Real.tan α = Real.sqrt 3 ∧ α = Real.pi / 3 :=
by
  sorry

end slope_of_line_in_terms_of_angle_l69_69210


namespace lemonade_quarts_water_l69_69597

-- Definitions derived from the conditions
def total_parts := 6 + 2 + 1 -- Sum of all ratio parts
def parts_per_gallon : ℚ := 1.5 / total_parts -- Volume per part in gallons
def parts_per_quart : ℚ := parts_per_gallon * 4 -- Volume per part in quarts
def water_needed : ℚ := 6 * parts_per_quart -- Quarts of water needed

-- Statement to prove
theorem lemonade_quarts_water : water_needed = 4 := 
by sorry

end lemonade_quarts_water_l69_69597


namespace chord_equation_l69_69513

variable {x y k b : ℝ}

-- Define the condition of the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 - 4 = 0

-- Define the condition that the point M(1, 1) is the midpoint
def midpoint_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1

-- Define the line equation in terms of its slope k and y-intercept b
def line (k b : ℝ) (x y : ℝ) : Prop := y = k * x + b

theorem chord_equation :
  (∃ (x₁ x₂ y₁ y₂ : ℝ), ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ midpoint_condition x₁ y₁ x₂ y₂) →
  (∃ (k b : ℝ), line k b x y ∧ k + b = 1 ∧ b = 1 - k) →
  y = -0.5 * x + 1.5 ↔ x + 2 * y - 3 = 0 :=
by
  sorry

end chord_equation_l69_69513


namespace y_intercepts_parabola_l69_69738

theorem y_intercepts_parabola : 
  ∀ (y : ℝ), ¬(0 = 3 * y^2 - 5 * y + 12) :=
by 
  -- Given x = 0, we have the equation 3 * y^2 - 5 * y + 12 = 0.
  -- The discriminant ∆ = b^2 - 4ac = (-5)^2 - 4 * 3 * 12 = 25 - 144 = -119 which is less than 0.
  -- Since the discriminant is negative, the quadratic equation has no real roots.
  sorry

end y_intercepts_parabola_l69_69738


namespace birthday_candles_l69_69794

theorem birthday_candles :
  ∀ (candles_Ambika : ℕ) (candles_Aniyah : ℕ),
  candles_Ambika = 4 →
  candles_Aniyah = 6 * candles_Ambika →
  (candles_Ambika + candles_Aniyah) / 2 = 14 :=
by
  intro candles_Ambika candles_Aniyah h1 h2
  rw [h1, h2]
  sorry

end birthday_candles_l69_69794


namespace eugene_pencils_left_l69_69089

-- Define the total number of pencils Eugene initially has
def initial_pencils : ℝ := 234.0

-- Define the number of pencils Eugene gives away
def pencils_given_away : ℝ := 35.0

-- Define the expected number of pencils left
def expected_pencils_left : ℝ := 199.0

-- Prove the number of pencils left after giving away 35.0 equals 199.0
theorem eugene_pencils_left : initial_pencils - pencils_given_away = expected_pencils_left := by
  -- This is where the proof would go, if needed
  sorry

end eugene_pencils_left_l69_69089


namespace initial_speed_is_correct_l69_69875

def initial_speed (v : ℝ) : Prop :=
  let D_total : ℝ := 70 * 5
  let D_2 : ℝ := 85 * 2
  let D_1 := v * 3
  D_total = D_1 + D_2

theorem initial_speed_is_correct :
  ∃ v : ℝ, initial_speed v ∧ v = 60 :=
by
  sorry

end initial_speed_is_correct_l69_69875


namespace expected_value_of_game_l69_69734

theorem expected_value_of_game :
  let heads_prob := 1 / 4
  let tails_prob := 1 / 2
  let edge_prob := 1 / 4
  let gain_heads := 4
  let loss_tails := -3
  let gain_edge := 0
  let expected_value := heads_prob * gain_heads + tails_prob * loss_tails + edge_prob * gain_edge
  expected_value = -0.5 :=
by
  sorry

end expected_value_of_game_l69_69734


namespace units_of_Product_C_sold_l69_69820

-- Definitions of commission rates
def commission_rate_A : ℝ := 0.05
def commission_rate_B : ℝ := 0.07
def commission_rate_C : ℝ := 0.10

-- Definitions of revenues per unit
def revenue_A : ℝ := 1500
def revenue_B : ℝ := 2000
def revenue_C : ℝ := 3500

-- Definition of units sold
def units_A : ℕ := 5
def units_B : ℕ := 3

-- Commission calculations for Product A and B
def commission_A : ℝ := commission_rate_A * revenue_A * units_A
def commission_B : ℝ := commission_rate_B * revenue_B * units_B

-- Previous average commission and new average commission
def previous_avg_commission : ℝ := 100
def new_avg_commission : ℝ := 250

-- The main proof statement
theorem units_of_Product_C_sold (x : ℝ) (h1 : new_avg_commission = previous_avg_commission + 150)
  (h2 : total_units = units_A + units_B + x)
  (h3 : total_new_commission = commission_A + commission_B + (commission_rate_C * revenue_C * x))
  : x = 12 :=
by
  sorry

end units_of_Product_C_sold_l69_69820


namespace isosceles_triangle_base_length_l69_69219

theorem isosceles_triangle_base_length (a b : ℝ) (h : a = 4 ∧ b = 4) : a + b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l69_69219


namespace range_of_a_l69_69707

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 9 ^ x - 2 * 3 ^ x + a - 3 > 0) → a > 4 :=
by
  sorry

end range_of_a_l69_69707


namespace katerina_weight_correct_l69_69709

-- We define the conditions
def total_weight : ℕ := 95
def alexa_weight : ℕ := 46

-- Define the proposition to prove: Katerina's weight is the total weight minus Alexa's weight, which should be 49.
theorem katerina_weight_correct : (total_weight - alexa_weight = 49) :=
by
  -- We use sorry to skip the proof.
  sorry

end katerina_weight_correct_l69_69709


namespace find_x_l69_69122

theorem find_x (x : ℝ) (h_pos : x > 0) (h_eq : x * (⌊x⌋) = 132) : x = 12 := sorry

end find_x_l69_69122


namespace find_length_of_BC_l69_69330

-- Define the geometrical objects and lengths
variable {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
variable (AB AC AM BC : ℝ)
variable (is_midpoint : Midpoint M B C)
variable (known_AB : AB = 7)
variable (known_AC : AC = 6)
variable (known_AM : AM = 4)

theorem find_length_of_BC : BC = Real.sqrt 106 := by
  sorry

end find_length_of_BC_l69_69330


namespace find_angle_B_l69_69882

-- Define the necessary trigonometric identities and dependencies
open Real

-- Declare the conditions under which we are working
theorem find_angle_B : 
  ∀ {a b A B : ℝ}, 
    a = 1 → 
    b = sqrt 3 → 
    A = π / 6 → 
    (B = π / 3 ∨ B = 2 * π / 3) := 
  by 
    intros a b A B ha hb hA
    sorry

end find_angle_B_l69_69882


namespace initial_bottle_caps_correct_l69_69408

-- Defining the variables based on the conditions
def bottle_caps_found : ℕ := 7
def total_bottle_caps_now : ℕ := 32
def initial_bottle_caps : ℕ := 25

-- Statement of the theorem
theorem initial_bottle_caps_correct:
  total_bottle_caps_now - bottle_caps_found = initial_bottle_caps :=
sorry

end initial_bottle_caps_correct_l69_69408


namespace rectangle_length_l69_69000

theorem rectangle_length (sq_side_len rect_width : ℕ) (sq_area : ℕ) (rect_len : ℕ) 
    (h1 : sq_side_len = 6) 
    (h2 : rect_width = 4) 
    (h3 : sq_area = sq_side_len * sq_side_len) 
    (h4 : sq_area = rect_width * rect_len) :
    rect_len = 9 := 
by 
  sorry

end rectangle_length_l69_69000


namespace wizard_answers_bal_l69_69881

-- Define the types for human and zombie as truth-tellers and liars respectively
inductive WizardType
| human : WizardType
| zombie : WizardType

-- Define the meaning of "bal"
inductive BalMeaning
| yes : BalMeaning
| no : BalMeaning

-- Question asked to the wizard
def question (w : WizardType) (b : BalMeaning) : Prop :=
  match w, b with
  | WizardType.human, BalMeaning.yes => true
  | WizardType.human, BalMeaning.no => false
  | WizardType.zombie, BalMeaning.yes => false
  | WizardType.zombie, BalMeaning.no => true

-- Theorem stating the wizard will answer "bal" to the given question
theorem wizard_answers_bal (w : WizardType) (b : BalMeaning) :
  question w b = true ↔ b = BalMeaning.yes :=
by
  sorry

end wizard_answers_bal_l69_69881


namespace fraction_difference_l69_69218

theorem fraction_difference (x y : ℝ) (h : x - y = 3 * x * y) : (1 / x) - (1 / y) = -3 :=
by
  sorry

end fraction_difference_l69_69218


namespace simplify_sqrt_is_cos_20_l69_69165

noncomputable def simplify_sqrt : ℝ :=
  let θ : ℝ := 160 * Real.pi / 180
  Real.sqrt (1 - Real.sin θ ^ 2)

theorem simplify_sqrt_is_cos_20 : simplify_sqrt = Real.cos (20 * Real.pi / 180) :=
  sorry

end simplify_sqrt_is_cos_20_l69_69165


namespace copier_cost_l69_69395

noncomputable def total_time : ℝ := 4 + 25 / 60
noncomputable def first_quarter_hour_cost : ℝ := 6
noncomputable def hourly_cost : ℝ := 8
noncomputable def time_after_first_quarter_hour : ℝ := total_time - 0.25
noncomputable def remaining_cost : ℝ := time_after_first_quarter_hour * hourly_cost
noncomputable def total_cost : ℝ := first_quarter_hour_cost + remaining_cost

theorem copier_cost :
  total_cost = 39.33 :=
by
  -- This statement remains to be proved.
  sorry

end copier_cost_l69_69395


namespace numbers_whose_triples_plus_1_are_primes_l69_69780

def is_prime (n : ℕ) : Prop := Nat.Prime n

def in_prime_range (n : ℕ) : Prop := 
  is_prime n ∧ 70 ≤ n ∧ n ≤ 110

def transformed_by_3_and_1 (x : ℕ) : ℕ := 3 * x + 1

theorem numbers_whose_triples_plus_1_are_primes :
  { x : ℕ | in_prime_range (transformed_by_3_and_1 x) } = {24, 26, 32, 34, 36} :=
by
  sorry

end numbers_whose_triples_plus_1_are_primes_l69_69780


namespace maximum_surface_area_of_cuboid_l69_69283

noncomputable def max_surface_area_of_inscribed_cuboid (R : ℝ) :=
  let (a, b, c) := (R, R, R) -- assuming cube dimensions where a=b=c
  2 * a * b + 2 * a * c + 2 * b * c

theorem maximum_surface_area_of_cuboid (R : ℝ) (h : ∃ a b c : ℝ, a^2 + b^2 + c^2 = 4 * R^2) :
  max_surface_area_of_inscribed_cuboid R = 8 * R^2 :=
sorry

end maximum_surface_area_of_cuboid_l69_69283


namespace math_proof_problem_l69_69525

theorem math_proof_problem (a : ℝ) : 
  (a^8 / a^4 ≠ a^4) ∧ ((a^2)^3 ≠ a^6) ∧ ((3*a)^3 ≠ 9*a^3) ∧ ((-a)^3 * (-a)^5 = a^8) := 
by 
  sorry

end math_proof_problem_l69_69525


namespace part1_part2_l69_69333

variable {a b : ℝ}

noncomputable def in_interval (x: ℝ) : Prop :=
  -1/2 < x ∧ x < 1/2

theorem part1 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1/3 * a + 1/6 * b) < 1/4 := 
by sorry

theorem part2 (h_a : in_interval a) (h_b : in_interval b) : 
  abs (1 - 4 * a * b) > 2 * abs (a - b) := 
by sorry

end part1_part2_l69_69333


namespace multiple_of_C_share_l69_69056

theorem multiple_of_C_share (A B C k : ℝ) : 
  3 * A = k * C ∧ 4 * B = k * C ∧ C = 84 ∧ A + B + C = 427 → k = 7 :=
by
  sorry

end multiple_of_C_share_l69_69056


namespace more_cats_than_dogs_l69_69589

-- Define the initial conditions
def initial_cats : ℕ := 28
def initial_dogs : ℕ := 18
def cats_adopted : ℕ := 3

-- Compute the number of cats after adoption
def cats_now : ℕ := initial_cats - cats_adopted

-- Define the target statement
theorem more_cats_than_dogs : cats_now - initial_dogs = 7 := by
  unfold cats_now
  unfold initial_cats
  unfold cats_adopted
  unfold initial_dogs
  sorry

end more_cats_than_dogs_l69_69589


namespace three_person_subcommittees_from_seven_l69_69047

-- Definition of the combinations formula (binomial coefficient)
def choose : ℕ → ℕ → ℕ
| n, k => if k = 0 then 1 else (n * choose (n - 1) (k - 1)) / k 

-- Problem statement in Lean 4
theorem three_person_subcommittees_from_seven : choose 7 3 = 35 :=
by
  -- We would fill in the steps here or use a sorry to skip the proof
  sorry

end three_person_subcommittees_from_seven_l69_69047


namespace prove_inequality_l69_69522

variable (x y z : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (h₃ : z > 0)
variable (h₄ : x + y + z = 1)

theorem prove_inequality :
  (3 * x^2 - x) / (1 + x^2) +
  (3 * y^2 - y) / (1 + y^2) +
  (3 * z^2 - z) / (1 + z^2) ≥ 0 :=
by
  sorry

end prove_inequality_l69_69522


namespace circle_condition_l69_69051

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4 * x - 2 * y + 5 * m = 0) ↔ m < 1 := by
  sorry

end circle_condition_l69_69051


namespace determine_y_l69_69927

theorem determine_y (x y : ℝ) (h₁ : x^2 = y - 7) (h₂ : x = 7) : y = 56 :=
sorry

end determine_y_l69_69927


namespace math_problem_l69_69580

noncomputable def proof_statement : Prop :=
  ∃ (a b m : ℝ),
    0 < a ∧ 0 < b ∧ 0 < m ∧
    (5 = m^2 * ((a^2 / b^2) + (b^2 / a^2)) + m * (a/b + b/a)) ∧
    m = (-1 + Real.sqrt 21) / 2

theorem math_problem : proof_statement :=
  sorry

end math_problem_l69_69580


namespace cistern_fill_time_l69_69324

theorem cistern_fill_time (F E : ℝ) (hF : F = 1 / 7) (hE : E = 1 / 9) : (1 / (F - E)) = 31.5 :=
by
  sorry

end cistern_fill_time_l69_69324


namespace proof_MrLalandeInheritance_l69_69902

def MrLalandeInheritance : Nat := 18000
def initialPayment : Nat := 3000
def monthlyInstallment : Nat := 2500
def numInstallments : Nat := 6

theorem proof_MrLalandeInheritance :
  initialPayment + numInstallments * monthlyInstallment = MrLalandeInheritance := 
by 
  sorry

end proof_MrLalandeInheritance_l69_69902


namespace uma_income_is_20000_l69_69404

/-- Given that the ratio of the incomes of Uma and Bala is 4 : 3, 
the ratio of their expenditures is 3 : 2, and both save $5000 at the end of the year, 
prove that Uma's income is $20000. -/
def uma_bala_income : Prop :=
  ∃ (x y : ℕ), (4 * x - 3 * y = 5000) ∧ (3 * x - 2 * y = 5000) ∧ (4 * x = 20000)
  
theorem uma_income_is_20000 : uma_bala_income :=
  sorry

end uma_income_is_20000_l69_69404


namespace solve_rational_equation_solve_quadratic_equation_l69_69405

-- Statement for the first equation
theorem solve_rational_equation (x : ℝ) (h : x ≠ 1) : 
  (x / (x - 1) + 2 / (1 - x) = 2) → (x = 0) :=
by intro h1; sorry

-- Statement for the second equation
theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 + 6 * x - 3 = 0) → (x = 1/2 ∨ x = -3) :=
by intro h1; sorry

end solve_rational_equation_solve_quadratic_equation_l69_69405


namespace possible_sums_of_products_neg11_l69_69487

theorem possible_sums_of_products_neg11 (a b c : ℤ) (h : a * b * c = -11) :
  a + b + c = -9 ∨ a + b + c = 11 ∨ a + b + c = 13 :=
sorry

end possible_sums_of_products_neg11_l69_69487


namespace equal_white_black_balls_l69_69391

theorem equal_white_black_balls (b w n x : ℕ) 
(h1 : x = n - x)
: (x = b + w - n + x - w) := sorry

end equal_white_black_balls_l69_69391


namespace composite_shape_perimeter_l69_69644

theorem composite_shape_perimeter :
  let r1 := 2.1
  let r2 := 3.6
  let π_approx := 3.14159
  let total_perimeter := π_approx * (r1 + r2)
  total_perimeter = 18.31 :=
by
  let radius1 := 2.1
  let radius2 := 3.6
  let total_radius := radius1 + radius2
  let pi_value := 3.14159
  let perimeter := pi_value * total_radius
  have calculation : perimeter = 18.31 := sorry
  exact calculation

end composite_shape_perimeter_l69_69644


namespace family_chocolate_chip_count_l69_69287

theorem family_chocolate_chip_count
  (batch_cookies : ℕ)
  (total_people : ℕ)
  (batches : ℕ)
  (choco_per_cookie : ℕ)
  (cookie_total : ℕ := batch_cookies * batches)
  (cookies_per_person : ℕ := cookie_total / total_people)
  (choco_per_person : ℕ := cookies_per_person * choco_per_cookie)
  (h1 : batch_cookies = 12)
  (h2 : total_people = 4)
  (h3 : batches = 3)
  (h4 : choco_per_cookie = 2)
  : choco_per_person = 18 := 
by sorry

end family_chocolate_chip_count_l69_69287


namespace dress_cost_l69_69113

theorem dress_cost (x : ℝ) 
  (h1 : 30 * x = 10 + x) 
  (h2 : 3 * ((10 + x) / 30) = x) : 
  x = 10 / 9 :=
by
  sorry

end dress_cost_l69_69113


namespace binary_preceding_and_following_l69_69679

theorem binary_preceding_and_following :
  ∀ (n : ℕ), n = 0b1010100 → (Nat.pred n = 0b1010011 ∧ Nat.succ n = 0b1010101) := by
  intros
  sorry

end binary_preceding_and_following_l69_69679


namespace num_turtles_on_sand_l69_69768

def num_turtles_total : ℕ := 42
def num_turtles_swept : ℕ := num_turtles_total / 3
def num_turtles_sand : ℕ := num_turtles_total - num_turtles_swept

theorem num_turtles_on_sand : num_turtles_sand = 28 := by
  sorry

end num_turtles_on_sand_l69_69768


namespace loss_percentage_initially_l69_69716

theorem loss_percentage_initially 
  (SP : ℝ) 
  (CP : ℝ := 400) 
  (h1 : SP + 100 = 1.05 * CP) : 
  (1 - SP / CP) * 100 = 20 := 
by 
  sorry

end loss_percentage_initially_l69_69716


namespace expected_male_teachers_in_sample_l69_69656

theorem expected_male_teachers_in_sample 
  (total_male total_female sample_size : ℕ) 
  (h1 : total_male = 56) 
  (h2 : total_female = 42) 
  (h3 : sample_size = 14) :
  (total_male * sample_size) / (total_male + total_female) = 8 :=
by
  sorry

end expected_male_teachers_in_sample_l69_69656


namespace computation_one_computation_two_l69_69729

-- Proof problem (1)
theorem computation_one :
  (-2)^3 + |(-3)| - Real.tan (Real.pi / 4) = -6 := by
  sorry

-- Proof problem (2)
theorem computation_two (a : ℝ) :
  (a + 2)^2 - a * (a - 4) = 8 * a + 4 := by
  sorry

end computation_one_computation_two_l69_69729


namespace reciprocal_inequality_l69_69761

variable (a b : ℝ)

theorem reciprocal_inequality (ha : a < 0) (hb : b > 0) : (1 / a) < (1 / b) := sorry

end reciprocal_inequality_l69_69761


namespace find_general_term_l69_69117

variable (a : ℕ → ℝ) (a1 : a 1 = 1)

def isGeometricSequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

def isArithmeticSequence (u v w : ℝ) :=
  2 * v = u + w

theorem find_general_term (h1 : a 1 = 1)
  (h2 : (isGeometricSequence a (1 / 2)))
  (h3 : isArithmeticSequence (1 / a 1) (1 / a 3) (1 / a 4 - 1)) :
  ∀ n, a n = (1 / 2) ^ (n - 1) :=
sorry

end find_general_term_l69_69117
