import Mathlib

namespace NUMINAMATH_GPT_q_value_at_2_l714_71496

-- Define the function q and the fact that (2, 3) is on its graph
def q : ℝ → ℝ := sorry

-- Condition: (2, 3) is on the graph of q(x)
axiom q_at_2 : q 2 = 3

-- Theorem: The value of q(2) is 3
theorem q_value_at_2 : q 2 = 3 := 
by 
  apply q_at_2

end NUMINAMATH_GPT_q_value_at_2_l714_71496


namespace NUMINAMATH_GPT_greatest_x_l714_71401

theorem greatest_x (x : ℕ) (h_pos : 0 < x) (h_ineq : (x^6) / (x^3) < 18) : x = 2 :=
by sorry

end NUMINAMATH_GPT_greatest_x_l714_71401


namespace NUMINAMATH_GPT_exists_right_triangle_area_eq_perimeter_l714_71420

theorem exists_right_triangle_area_eq_perimeter :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a + b + c = (a * b) / 2 ∧ a ≠ b ∧ 
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 12 ∧ b = 5 ∧ c = 13) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10)) :=
by
  sorry

end NUMINAMATH_GPT_exists_right_triangle_area_eq_perimeter_l714_71420


namespace NUMINAMATH_GPT_find_a_from_expansion_l714_71423

theorem find_a_from_expansion :
  (∃ a : ℝ, (∃ c : ℝ, (∃ d : ℝ, (∃ e : ℝ, (20 - 30 * a + 6 * a^2 = -16 ∧ (a = 2 ∨ a = 3))))))
:= sorry

end NUMINAMATH_GPT_find_a_from_expansion_l714_71423


namespace NUMINAMATH_GPT_inequality_condition_l714_71407

variables {a b c : ℝ} {x : ℝ}

theorem inequality_condition (h : a * a + b * b < c * c) : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end NUMINAMATH_GPT_inequality_condition_l714_71407


namespace NUMINAMATH_GPT_graph_passes_through_point_l714_71404

theorem graph_passes_through_point (a : ℝ) (x y : ℝ) (h : a < 0) : (1 - a)^0 - 1 = -1 :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l714_71404


namespace NUMINAMATH_GPT_train_length_l714_71487

/-- A train crosses a tree in 120 seconds. It takes 230 seconds to pass a platform 1100 meters long.
    How long is the train? -/
theorem train_length (L : ℝ) (V : ℝ)
    (h1 : V = L / 120)
    (h2 : V = (L + 1100) / 230) :
    L = 1200 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l714_71487


namespace NUMINAMATH_GPT_convoy_length_after_checkpoint_l714_71405

theorem convoy_length_after_checkpoint
  (L_initial : ℝ) (v_initial : ℝ) (v_final : ℝ) (t_fin : ℝ)
  (H_initial_len : L_initial = 300)
  (H_initial_speed : v_initial = 60)
  (H_final_speed : v_final = 40)
  (H_time_last_car : t_fin = (300 / 1000) / 60) :
  L_initial * v_final / v_initial - (v_final * ((300 / 1000) / 60)) = 200 :=
by
  sorry

end NUMINAMATH_GPT_convoy_length_after_checkpoint_l714_71405


namespace NUMINAMATH_GPT_sum_place_values_of_7s_l714_71442

theorem sum_place_values_of_7s (n : ℝ) (h : n = 87953.0727) : 
  let a := 7000
  let b := 0.07
  let c := 0.0007
  a + b + c = 7000.0707 :=
by
  sorry

end NUMINAMATH_GPT_sum_place_values_of_7s_l714_71442


namespace NUMINAMATH_GPT_problem_statement_l714_71494

theorem problem_statement (x : ℝ) :
  (x - 2)^4 + 5 * (x - 2)^3 + 10 * (x - 2)^2 + 10 * (x - 2) + 5 = (x - 2 + Real.sqrt 2)^4 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l714_71494


namespace NUMINAMATH_GPT_length_of_AB_in_triangle_l714_71403

open Real

theorem length_of_AB_in_triangle
  (AC BC : ℝ)
  (area : ℝ) :
  AC = 4 →
  BC = 3 →
  area = 3 * sqrt 3 →
  ∃ AB : ℝ, AB = sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_length_of_AB_in_triangle_l714_71403


namespace NUMINAMATH_GPT_sum_of_coefficients_shifted_function_l714_71467

def original_function (x : ℝ) : ℝ :=
  3*x^2 - 2*x + 6

def shifted_function (x : ℝ) : ℝ :=
  original_function (x + 5)

theorem sum_of_coefficients_shifted_function : 
  let a := 3
  let b := 28
  let c := 71
  a + b + c = 102 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_shifted_function_l714_71467


namespace NUMINAMATH_GPT_total_cost_function_range_of_x_minimum_cost_when_x_is_2_l714_71416

def transportation_cost (x : ℕ) : ℕ :=
  300 * x + 500 * (12 - x) + 400 * (10 - x) + 800 * (x - 2)

theorem total_cost_function (x : ℕ) : transportation_cost x = 200 * x + 8400 := by
  -- Simply restate the definition in the theorem form
  sorry

theorem range_of_x (x : ℕ) : 2 ≤ x ∧ x ≤ 10 := by
  -- Provide necessary constraints in theorem form
  sorry

theorem minimum_cost_when_x_is_2 : transportation_cost 2 = 8800 := by
  -- Final cost at minimum x
  sorry

end NUMINAMATH_GPT_total_cost_function_range_of_x_minimum_cost_when_x_is_2_l714_71416


namespace NUMINAMATH_GPT_rectangle_side_l714_71424

theorem rectangle_side (x : ℝ) (w : ℝ) (P : ℝ) (hP : P = 30) (h : 2 * (x + w) = P) : w = 15 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_rectangle_side_l714_71424


namespace NUMINAMATH_GPT_sum_of_inserted_numbers_l714_71413

theorem sum_of_inserted_numbers (x y : ℝ) (h1 : x^2 = 2 * y) (h2 : 2 * y = x + 20) :
  x + y = 4 ∨ x + y = 17.5 :=
sorry

end NUMINAMATH_GPT_sum_of_inserted_numbers_l714_71413


namespace NUMINAMATH_GPT_everett_weeks_worked_l714_71446

theorem everett_weeks_worked (daily_hours : ℕ) (total_hours : ℕ) (days_in_week : ℕ) 
  (h1 : daily_hours = 5) (h2 : total_hours = 140) (h3 : days_in_week = 7) : 
  (total_hours / (daily_hours * days_in_week) = 4) :=
by
  sorry

end NUMINAMATH_GPT_everett_weeks_worked_l714_71446


namespace NUMINAMATH_GPT_right_triangle_legs_l714_71483

theorem right_triangle_legs (a b : ℕ) (hypotenuse : ℕ) (h : hypotenuse = 39) : a^2 + b^2 = 39^2 → (a = 15 ∧ b = 36) ∨ (a = 36 ∧ b = 15) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_legs_l714_71483


namespace NUMINAMATH_GPT_gcd_of_1237_and_1849_l714_71459

def gcd_1237_1849 : ℕ := 1

theorem gcd_of_1237_and_1849 : Nat.gcd 1237 1849 = gcd_1237_1849 := by
  sorry

end NUMINAMATH_GPT_gcd_of_1237_and_1849_l714_71459


namespace NUMINAMATH_GPT_find_y_minus_x_l714_71456

theorem find_y_minus_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : Real.sqrt x + Real.sqrt y = 1) 
  (h5 : Real.sqrt (x / y) + Real.sqrt (y / x) = 10 / 3) : 
  y - x = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_y_minus_x_l714_71456


namespace NUMINAMATH_GPT_minimize_quadratic_function_l714_71486

def quadratic_function (x : ℝ) : ℝ := x^2 + 8*x + 7

theorem minimize_quadratic_function : ∃ x : ℝ, (∀ y : ℝ, quadratic_function y ≥ quadratic_function x) ∧ x = -4 :=
by
  sorry

end NUMINAMATH_GPT_minimize_quadratic_function_l714_71486


namespace NUMINAMATH_GPT_guest_bedroom_area_l714_71475

theorem guest_bedroom_area 
  (master_bedroom_bath_area : ℝ)
  (kitchen_guest_bath_living_area : ℝ)
  (total_rent : ℝ)
  (rate_per_sqft : ℝ)
  (num_guest_bedrooms : ℕ)
  (area_guest_bedroom : ℝ) :
  master_bedroom_bath_area = 500 →
  kitchen_guest_bath_living_area = 600 →
  total_rent = 3000 →
  rate_per_sqft = 2 →
  num_guest_bedrooms = 2 →
  (total_rent / rate_per_sqft) - (master_bedroom_bath_area + kitchen_guest_bath_living_area) / num_guest_bedrooms = area_guest_bedroom → 
  area_guest_bedroom = 200 := by
  sorry

end NUMINAMATH_GPT_guest_bedroom_area_l714_71475


namespace NUMINAMATH_GPT_courtyard_brick_problem_l714_71490

noncomputable def area_courtyard (length width : ℝ) : ℝ :=
  length * width

noncomputable def area_brick (length width : ℝ) : ℝ :=
  length * width

noncomputable def total_bricks_required (court_area brick_area : ℝ) : ℝ :=
  court_area / brick_area

theorem courtyard_brick_problem 
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (brick_width : ℝ)
  (H1 : courtyard_length = 18)
  (H2 : courtyard_width = 12)
  (H3 : brick_length = 15 / 100)
  (H4 : brick_width = 13 / 100) :
  
  total_bricks_required (area_courtyard courtyard_length courtyard_width * 10000) 
                        (area_brick brick_length brick_width) 
  = 11077 :=
by
  sorry

end NUMINAMATH_GPT_courtyard_brick_problem_l714_71490


namespace NUMINAMATH_GPT_equilibrium_proof_l714_71414

noncomputable def equilibrium_constant (Γ_eq B_eq : ℝ) : ℝ :=
(Γ_eq ^ 3) / (B_eq ^ 3)

theorem equilibrium_proof (Γ_eq B_eq : ℝ) (K_c : ℝ) (B_initial : ℝ) (Γ_initial : ℝ)
  (hΓ : Γ_eq = 0.25) (hB : B_eq = 0.15) (hKc : K_c = 4.63) 
  (ratio : Γ_eq = B_eq + B_initial) (hΓ_initial : Γ_initial = 0) :
  equilibrium_constant Γ_eq B_eq = K_c ∧ 
  B_initial = 0.4 ∧ 
  Γ_initial = 0 := 
by
  sorry

end NUMINAMATH_GPT_equilibrium_proof_l714_71414


namespace NUMINAMATH_GPT_probability_of_two_one_color_and_one_other_color_l714_71434

theorem probability_of_two_one_color_and_one_other_color
    (black_balls white_balls : ℕ)
    (total_drawn : ℕ)
    (draw_two_black_one_white : ℕ)
    (draw_one_black_two_white : ℕ)
    (total_ways : ℕ)
    (favorable_ways : ℕ)
    (probability : ℚ) :
    black_balls = 8 →
    white_balls = 7 →
    total_drawn = 3 →
    draw_two_black_one_white = 196 →
    draw_one_black_two_white = 168 →
    total_ways = 455 →
    favorable_ways = draw_two_black_one_white + draw_one_black_two_white →
    probability = favorable_ways / total_ways →
    probability = 4 / 5 :=
by sorry

end NUMINAMATH_GPT_probability_of_two_one_color_and_one_other_color_l714_71434


namespace NUMINAMATH_GPT_find_a_b_largest_x_l714_71495

def polynomial (a b x : ℤ) : ℤ := 2 * (a * x - 3) - 3 * (b * x + 5)

-- Given conditions
variables (a b : ℤ)
#check polynomial

-- Part 1: Prove the values of a and b
theorem find_a_b (h1 : polynomial a b 2 = -31) (h2 : a + b = 0) : a = -1 ∧ b = 1 :=
by sorry

-- Part 2: Given a and b found in Part 1, find the largest integer x such that P > 0
noncomputable def P (x : ℤ) : ℤ := -5 * x - 21

theorem largest_x {a b : ℤ} (ha : a = -1) (hb : b = 1) : ∃ x : ℤ, P x > 0 ∧ ∀ y : ℤ, (P y > 0 → y ≤ x) :=
by sorry

end NUMINAMATH_GPT_find_a_b_largest_x_l714_71495


namespace NUMINAMATH_GPT_sum_powers_of_i_l714_71484

-- Define the conditions
def i : ℂ := Complex.I -- Complex.I is the imaginary unit in ℂ (ℂ is the set of complex numbers)

-- The theorem statement
theorem sum_powers_of_i : (i + i^2 + i^3 + i^4) * 150 + 1 + i + i^2 + i^3 = 0 := by
  sorry

end NUMINAMATH_GPT_sum_powers_of_i_l714_71484


namespace NUMINAMATH_GPT_find_number_l714_71478

def valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (¬ (n % 10 = 6) ∧ n % 7 = 0)) ∧
  ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (¬ (n > 26) ∧ n % 10 = 8)) ∧
  ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (¬ (n % 13 = 0) ∧ n < 27))

theorem find_number : ∃ n : ℕ, valid_number n ∧ n = 91 := by
  sorry

end NUMINAMATH_GPT_find_number_l714_71478


namespace NUMINAMATH_GPT_wheat_flour_one_third_l714_71408

theorem wheat_flour_one_third (recipe_cups: ℚ) (third_recipe: ℚ) 
  (h1: recipe_cups = 5 + 2 / 3) (h2: third_recipe = recipe_cups / 3) :
  third_recipe = 1 + 8 / 9 :=
by
  sorry

end NUMINAMATH_GPT_wheat_flour_one_third_l714_71408


namespace NUMINAMATH_GPT_f_g_2_eq_36_l714_71438

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 4 * x - 2

theorem f_g_2_eq_36 : f (g 2) = 36 :=
by
  sorry

end NUMINAMATH_GPT_f_g_2_eq_36_l714_71438


namespace NUMINAMATH_GPT_range_of_a_l714_71441

noncomputable def quadratic_inequality_holds (a : ℝ) : Prop :=
  ∀ (x : ℝ), a * x^2 - a * x - 1 < 0 

theorem range_of_a (a : ℝ) : quadratic_inequality_holds a ↔ -4 < a ∧ a ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l714_71441


namespace NUMINAMATH_GPT_june_1_friday_l714_71406

open Nat

-- Define the days of the week as data type
inductive DayOfWeek : Type
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

open DayOfWeek

-- Define that June has 30 days
def june_days := 30

-- Hypotheses that June has exactly three Mondays and exactly three Thursdays
def three_mondays (d : DayOfWeek) : Prop := 
  ∃ days : Fin 30 → DayOfWeek, 
    (∀ n : Fin 30, days n = Monday → 3 ≤ n / 7) -- there are exactly three Mondays
  
def three_thursdays (d : DayOfWeek) : Prop := 
  ∃ days : Fin 30 → DayOfWeek, 
    (∀ n : Fin 30, days n = Thursday → 3 ≤ n / 7) -- there are exactly three Thursdays

-- Theorem to prove June 1 falls on a Friday given those conditions
theorem june_1_friday : ∀ (d : DayOfWeek), 
  three_mondays d → three_thursdays d → (d = Friday) :=
by
  sorry

end NUMINAMATH_GPT_june_1_friday_l714_71406


namespace NUMINAMATH_GPT_alphazia_lost_words_l714_71481

def alphazia_letters := 128
def forbidden_letters := 2
def total_forbidden_pairs := forbidden_letters * alphazia_letters

theorem alphazia_lost_words :
  let one_letter_lost := forbidden_letters
  let two_letter_lost := 2 * alphazia_letters
  one_letter_lost + two_letter_lost = 258 :=
by
  sorry

end NUMINAMATH_GPT_alphazia_lost_words_l714_71481


namespace NUMINAMATH_GPT_arun_weight_average_l714_71473

theorem arun_weight_average (w : ℝ) 
  (h1 : 64 < w ∧ w < 72) 
  (h2 : 60 < w ∧ w < 70) 
  (h3 : w ≤ 67) : 
  (64 + 67) / 2 = 65.5 := 
  by sorry

end NUMINAMATH_GPT_arun_weight_average_l714_71473


namespace NUMINAMATH_GPT_find_purchase_price_l714_71415

noncomputable def purchase_price (total_paid : ℝ) (interest_percent : ℝ) : ℝ :=
    total_paid / (1 + interest_percent)

theorem find_purchase_price :
  purchase_price 130 0.09090909090909092 = 119.09 :=
by
  -- Normally we would provide the full proof here, but it is omitted as per instructions
  sorry

end NUMINAMATH_GPT_find_purchase_price_l714_71415


namespace NUMINAMATH_GPT_modulo_residue_l714_71469

theorem modulo_residue : 
  ∃ (x : ℤ), 0 ≤ x ∧ x < 31 ∧ (-1237 % 31) = x := 
  sorry

end NUMINAMATH_GPT_modulo_residue_l714_71469


namespace NUMINAMATH_GPT_rate_is_correct_l714_71433

noncomputable def rate_of_interest (P A T : ℝ) : ℝ :=
  let SI := A - P
  (SI * 100) / (P * T)

theorem rate_is_correct :
  rate_of_interest 10000 18500 8 = 10.625 := 
by
  sorry

end NUMINAMATH_GPT_rate_is_correct_l714_71433


namespace NUMINAMATH_GPT_problem_l714_71429

variable {x y : ℝ}

theorem problem (h : x < y) : 3 - x > 3 - y :=
sorry

end NUMINAMATH_GPT_problem_l714_71429


namespace NUMINAMATH_GPT_alloy_problem_solution_l714_71477

theorem alloy_problem_solution (x y k n : ℝ) (H_weight : k * 4 * x + n * 3 * y = 10)
    (H_ratio : (kx + ny)/(k * 3 * x + n * 2 * y) = 3/7) :
    k * 4 * x = 4 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_alloy_problem_solution_l714_71477


namespace NUMINAMATH_GPT_quadratic_vertex_l714_71474

theorem quadratic_vertex (x y : ℝ) (h : y = -3 * x^2 + 2) : (x, y) = (0, 2) :=
sorry

end NUMINAMATH_GPT_quadratic_vertex_l714_71474


namespace NUMINAMATH_GPT_sum_of_angles_l714_71466

theorem sum_of_angles (x y : ℝ) (n : ℕ) :
  n = 16 →
  (∃ k l : ℕ, k = 3 ∧ l = 5 ∧ 
  x = (k * (360 / n)) / 2 ∧ y = (l * (360 / n)) / 2) →
  x + y = 90 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_angles_l714_71466


namespace NUMINAMATH_GPT_num_of_terms_in_arithmetic_sequence_l714_71480

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the first term, common difference, and last term of the sequence
def a : ℕ := 15
def d : ℕ := 4
def last_term : ℕ := 99

-- Define the number of terms in the sequence
def n : ℕ := 22

-- State the theorem
theorem num_of_terms_in_arithmetic_sequence : arithmetic_seq a d n = last_term :=
by
  sorry

end NUMINAMATH_GPT_num_of_terms_in_arithmetic_sequence_l714_71480


namespace NUMINAMATH_GPT_find_g_neg_6_l714_71412

noncomputable def g : ℤ → ℤ := sorry

-- Conditions from the problem
axiom g_condition_1 : g 1 - 1 > 0
axiom g_condition_2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom g_condition_3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The statement we need to prove
theorem find_g_neg_6 : g (-6) = 723 :=
by { sorry }

end NUMINAMATH_GPT_find_g_neg_6_l714_71412


namespace NUMINAMATH_GPT_triangle_area_interval_l714_71472

theorem triangle_area_interval (s : ℝ) :
  10 ≤ (s - 1)^(3 / 2) ∧ (s - 1)^(3 / 2) ≤ 50 → (5.64 ≤ s ∧ s ≤ 18.32) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_interval_l714_71472


namespace NUMINAMATH_GPT_ratio_a_over_b_l714_71465

-- Definitions of conditions
def func (a b x : ℝ) : ℝ := a * x^2 + b
def derivative (a b x : ℝ) : ℝ := 2 * a * x

-- Given conditions
variables (a b : ℝ)
axiom tangent_slope : derivative a b 1 = 2
axiom point_on_graph : func a b 1 = 3

-- Statement to prove
theorem ratio_a_over_b : a / b = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_a_over_b_l714_71465


namespace NUMINAMATH_GPT_compl_union_eq_l714_71468

-- Definitions
def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 4}
def B : Set ℤ := {2, 4}

-- The statement
theorem compl_union_eq : (Aᶜ ∩ U) ∪ B = {2, 4, 5, 6} :=
by sorry

end NUMINAMATH_GPT_compl_union_eq_l714_71468


namespace NUMINAMATH_GPT_math_problem_l714_71449

theorem math_problem
  (a b c d m : ℝ)
  (h1 : a = -b)
  (h2 : c = (1 / d) ∨ d = (1 / c))
  (h3 : |m| = 4) :
  (a + b = 0) ∧ (c * d = 1) ∧ (m = 4 ∨ m = -4) ∧
  ((a + b) / 3 + m^2 - 5 * (c * d) = 11) := by
  sorry

end NUMINAMATH_GPT_math_problem_l714_71449


namespace NUMINAMATH_GPT_greatest_possible_remainder_l714_71457

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 11 ∧ x % 11 = r ∧ r = 10 :=
by
  exists 10
  sorry

end NUMINAMATH_GPT_greatest_possible_remainder_l714_71457


namespace NUMINAMATH_GPT_cycle_time_to_library_l714_71498

theorem cycle_time_to_library 
  (constant_speed : Prop)
  (time_to_park : ℕ)
  (distance_to_park : ℕ)
  (distance_to_library : ℕ)
  (h1 : constant_speed)
  (h2 : time_to_park = 30)
  (h3 : distance_to_park = 5)
  (h4 : distance_to_library = 3) :
  (18 : ℕ) = (30 * distance_to_library / distance_to_park) :=
by
  intros
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_cycle_time_to_library_l714_71498


namespace NUMINAMATH_GPT_range_of_a_l714_71417

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (hf_odd : ∀ x, f (-x) = -f x)
  (hf_expr_pos : ∀ x, x > 0 → f x = -x^2 + ax - 1 - a)
  (hf_monotone : ∀ x y, x < y → f y ≤ f x) :
  -1 ≤ a ∧ a ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l714_71417


namespace NUMINAMATH_GPT_intersection_nonempty_iff_l714_71460

/-- Define sets A and B as described in the problem. -/
def A (x : ℝ) : Prop := -2 < x ∧ x ≤ 1
def B (x : ℝ) (k : ℝ) : Prop := x ≥ k

/-- The main theorem to prove the range of k where the intersection of A and B is non-empty. -/
theorem intersection_nonempty_iff (k : ℝ) : (∃ x, A x ∧ B x k) ↔ k ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_nonempty_iff_l714_71460


namespace NUMINAMATH_GPT_jane_average_speed_l714_71489

theorem jane_average_speed :
  let total_distance := 200
  let total_time := 6
  total_distance / total_time = 100 / 3 :=
by
  sorry

end NUMINAMATH_GPT_jane_average_speed_l714_71489


namespace NUMINAMATH_GPT_trig_identity_l714_71427

variable (α : ℝ)

theorem trig_identity (h : Real.sin (α - 70 * Real.pi / 180) = α) : 
  Real.cos (α + 20 * Real.pi / 180) = -α := by
  sorry

end NUMINAMATH_GPT_trig_identity_l714_71427


namespace NUMINAMATH_GPT_flagpole_breaking_height_l714_71418

theorem flagpole_breaking_height (x : ℝ) (h_pos : 0 < x) (h_ineq : x < 6)
    (h_pythagoras : (x^2 + 2^2 = 6^2)) : x = Real.sqrt 10 :=
by sorry

end NUMINAMATH_GPT_flagpole_breaking_height_l714_71418


namespace NUMINAMATH_GPT_sum_of_coefficients_l714_71440

theorem sum_of_coefficients (a b c : ℤ) (h : a - b + c = -1) : a + b + c = -1 := sorry

end NUMINAMATH_GPT_sum_of_coefficients_l714_71440


namespace NUMINAMATH_GPT_profit_starts_from_third_year_most_beneficial_option_l714_71488

-- Define the conditions of the problem
def investment_cost := 144
def maintenance_cost (n : ℕ) := 4 * n^2 + 20 * n
def revenue_per_year := 1

-- Define the net profit function
def net_profit (n : ℕ) : ℤ :=
(revenue_per_year * n : ℤ) - (maintenance_cost n) - investment_cost

-- Question 1: Prove the project starts to make a profit from the 3rd year
theorem profit_starts_from_third_year (n : ℕ) (h : 2 < n ∧ n < 18) : 
net_profit n > 0 ↔ 3 ≤ n := sorry

-- Question 2: Prove the most beneficial option for company's development
theorem most_beneficial_option : (∃ o, o = 1) ∧ (∃ t1 t2, t1 = 264 ∧ t2 = 264 ∧ t1 < t2) := sorry

end NUMINAMATH_GPT_profit_starts_from_third_year_most_beneficial_option_l714_71488


namespace NUMINAMATH_GPT_inequality_for_pos_a_b_c_d_l714_71451

theorem inequality_for_pos_a_b_c_d
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) * (b + c) * (c + d) * (d + a) * (1 + (abcd ^ (1/4)))^4
  ≥ 16 * abcd * (1 + a) * (1 + b) * (1 + c) * (1 + d) :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_pos_a_b_c_d_l714_71451


namespace NUMINAMATH_GPT_circles_touch_each_other_l714_71430

-- Define the radii of the two circles and the distance between their centers.
variables (R r d : ℝ)

-- Hypotheses: the condition and the relationships derived from the solution.
variables (x y t : ℝ)

-- The core relationships as conditions based on the problem and the solution.
axiom h1 : x + y = t
axiom h2 : x / y = R / r
axiom h3 : t / d = x / R

-- The proof statement
theorem circles_touch_each_other 
  (h1 : x + y = t) 
  (h2 : x / y = R / r) 
  (h3 : t / d = x / R) : 
  d = R + r := 
by 
  sorry

end NUMINAMATH_GPT_circles_touch_each_other_l714_71430


namespace NUMINAMATH_GPT_min_deg_q_l714_71476

-- Definitions of polynomials requirements
variables (p q r : Polynomial ℝ)

-- Given Conditions
def polynomials_relation : Prop := 5 * p + 6 * q = r
def deg_p : Prop := p.degree = 10
def deg_r : Prop := r.degree = 12

-- The main theorem we want to prove
theorem min_deg_q (h1 : polynomials_relation p q r) (h2 : deg_p p) (h3 : deg_r r) : q.degree ≥ 12 :=
sorry

end NUMINAMATH_GPT_min_deg_q_l714_71476


namespace NUMINAMATH_GPT_tan_monotone_increasing_interval_l714_71447

theorem tan_monotone_increasing_interval :
  ∀ k : ℤ, ∀ x : ℝ, 
  (-π / 2 + k * π < x + π / 4 ∧ x + π / 4 < π / 2 + k * π) ↔
  (k * π - 3 * π / 4 < x ∧ x < k * π + π / 4) :=
by sorry

end NUMINAMATH_GPT_tan_monotone_increasing_interval_l714_71447


namespace NUMINAMATH_GPT_compound_interest_doubling_time_l714_71428

theorem compound_interest_doubling_time :
  ∃ (t : ℕ), (0.15 : ℝ) = 0.15 ∧ ∀ (n : ℕ), (n = 1) →
               (2 : ℝ) < (1 + 0.15) ^ t ∧ t = 5 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_doubling_time_l714_71428


namespace NUMINAMATH_GPT_age_relation_l714_71470

variable (x y z : ℕ)

theorem age_relation (h1 : x > y) : (z > y) ↔ (∃ w, w > 0 ∧ y + z > 2 * x) :=
sorry

end NUMINAMATH_GPT_age_relation_l714_71470


namespace NUMINAMATH_GPT_inequality_proof_l714_71435

theorem inequality_proof (a b c d : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
  (a_geq_1 : 1 ≤ a) (b_geq_1 : 1 ≤ b) (c_geq_1 : 1 ≤ c)
  (abcd_eq_1 : a * b * c * d = 1)
  : 
  1 / (a^2 - a + 1)^2 + 1 / (b^2 - b + 1)^2 + 1 / (c^2 - c + 1)^2 + 1 / (d^2 - d + 1)^2 ≤ 4
  := sorry

end NUMINAMATH_GPT_inequality_proof_l714_71435


namespace NUMINAMATH_GPT_Maria_high_school_students_l714_71402

theorem Maria_high_school_students (M J : ℕ) (h1 : M = 4 * J) (h2 : M + J = 3600) : M = 2880 :=
sorry

end NUMINAMATH_GPT_Maria_high_school_students_l714_71402


namespace NUMINAMATH_GPT_cat_litter_container_weight_l714_71453

theorem cat_litter_container_weight :
  (∀ (cost_container : ℕ) (pounds_per_litterbox : ℕ) (cost_total : ℕ) (days : ℕ),
    cost_container = 21 ∧ pounds_per_litterbox = 15 ∧ cost_total = 210 ∧ days = 210 → 
    ∀ (weeks : ℕ), weeks = days / 7 →
    ∀ (containers : ℕ), containers = cost_total / cost_container →
    ∀ (cost_per_container : ℕ), cost_per_container = cost_total / containers →
    (∃ (pounds_per_container : ℕ), pounds_per_container = cost_container / cost_per_container ∧ pounds_per_container = 3)) :=
by
  intros cost_container pounds_per_litterbox cost_total days
  intros h weeks hw containers hc containers_cost hc_cost
  sorry

end NUMINAMATH_GPT_cat_litter_container_weight_l714_71453


namespace NUMINAMATH_GPT_union_when_a_eq_2_condition_1_condition_2_condition_3_l714_71497

open Set

def setA (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def setB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_when_a_eq_2 : setA 2 ∪ setB = {x | -1 ≤ x ∧ x ≤ 3} :=
sorry

theorem condition_1 (a : ℝ) : 
  (setA a ∪ setB = setB) → (0 ≤ a ∧ a ≤ 2) :=
sorry

theorem condition_2 (a : ℝ) :
  (∀ x, (x ∈ setA a ↔ x ∈ setB)) → (0 ≤ a ∧ a ≤ 2) :=
sorry

theorem condition_3 (a : ℝ) :
  (setA a ∩ setB = ∅) → (a < -2 ∨ 4 < a) :=
sorry

end NUMINAMATH_GPT_union_when_a_eq_2_condition_1_condition_2_condition_3_l714_71497


namespace NUMINAMATH_GPT_part1_part2_l714_71455

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 5 * Real.log x + a * x^2 - 6 * x
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := 5 / x + 2 * a * x - 6

theorem part1 (a : ℝ) (h_tangent : f_prime 1 a = 0) : a = 1 / 2 :=
by {
  sorry
}

theorem part2 (a : ℝ) (h_a : a = 1/2) :
  (∀ x, 0 < x → x < 1 → f_prime x a > 0) ∧
  (∀ x, 5 < x → f_prime x a > 0) ∧
  (∀ x, 1 < x → x < 5 → f_prime x a < 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_part2_l714_71455


namespace NUMINAMATH_GPT_calc1_calc2_l714_71421

noncomputable def calculation1 := -4^2

theorem calc1 : calculation1 = -16 := by
  sorry

noncomputable def calculation2 := (-3) - (-6)

theorem calc2 : calculation2 = 3 := by
  sorry

end NUMINAMATH_GPT_calc1_calc2_l714_71421


namespace NUMINAMATH_GPT_triangle_perimeter_l714_71432

theorem triangle_perimeter
  (x : ℝ) 
  (h : x^2 - 6 * x + 8 = 0)
  (a b c : ℝ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = x)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 10 := 
sorry

end NUMINAMATH_GPT_triangle_perimeter_l714_71432


namespace NUMINAMATH_GPT_factorial_inequality_l714_71482

theorem factorial_inequality (n : ℕ) : 2^n * n! < (n+1)^n :=
by
  sorry

end NUMINAMATH_GPT_factorial_inequality_l714_71482


namespace NUMINAMATH_GPT_dan_gave_marbles_l714_71422

-- Conditions as definitions in Lean 4
def original_marbles : ℕ := 64
def marbles_left : ℕ := 50
def marbles_given : ℕ := original_marbles - marbles_left

-- Theorem statement proving the question == answer given the conditions.
theorem dan_gave_marbles : marbles_given = 14 := by
  sorry

end NUMINAMATH_GPT_dan_gave_marbles_l714_71422


namespace NUMINAMATH_GPT_minimum_tenth_game_score_l714_71458

theorem minimum_tenth_game_score (S5 : ℕ) (score10 : ℕ) 
  (h1 : 18 + 15 + 16 + 19 = 68)
  (h2 : S5 ≤ 85)
  (h3 : (S5 + 68 + score10) / 10 > 17) : 
  score10 ≥ 18 := sorry

end NUMINAMATH_GPT_minimum_tenth_game_score_l714_71458


namespace NUMINAMATH_GPT_some_number_value_l714_71426

theorem some_number_value (a : ℤ) (x1 x2 : ℤ)
  (h1 : x1 + a = 10) (h2 : x2 + a = -10) (h_sum : x1 + x2 = 20) : a = -10 :=
by
  sorry

end NUMINAMATH_GPT_some_number_value_l714_71426


namespace NUMINAMATH_GPT_regression_line_estimate_l714_71425

theorem regression_line_estimate:
  (∀ (x y : ℝ), y = 1.23 * x + a ↔ a = 5 - 1.23 * 4) →
  ∃ (y : ℝ), y = 1.23 * 2 + 0.08 :=
by
  intro h
  use 2.54
  simp
  sorry

end NUMINAMATH_GPT_regression_line_estimate_l714_71425


namespace NUMINAMATH_GPT_polynomial_factorization_l714_71464

theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + ab + ac + b^2 + bc + c^2) :=
sorry

end NUMINAMATH_GPT_polynomial_factorization_l714_71464


namespace NUMINAMATH_GPT_fraction_calculation_l714_71439

theorem fraction_calculation : ( ( (1/2 : ℚ) + (1/5) ) / ( (3/7) - (1/14) ) * (2/3) ) = 98/75 :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l714_71439


namespace NUMINAMATH_GPT_retailer_marked_price_percentage_above_cost_l714_71462

noncomputable def cost_price : ℝ := 100
noncomputable def discount_rate : ℝ := 0.15
noncomputable def sales_profit_rate : ℝ := 0.275

theorem retailer_marked_price_percentage_above_cost :
  ∃ (MP : ℝ), ((MP - cost_price) / cost_price = 0.5) ∧ (((MP * (1 - discount_rate)) - cost_price) / cost_price = sales_profit_rate) :=
sorry

end NUMINAMATH_GPT_retailer_marked_price_percentage_above_cost_l714_71462


namespace NUMINAMATH_GPT_k_value_of_polynomial_square_l714_71445

theorem k_value_of_polynomial_square (k : ℤ) :
  (∃ (f : ℤ → ℤ), ∀ x, f x = x^2 + 6 * x + k^2) → (k = 3 ∨ k = -3) :=
by
  sorry

end NUMINAMATH_GPT_k_value_of_polynomial_square_l714_71445


namespace NUMINAMATH_GPT_compute_c_plus_d_l714_71491

-- Define the conditions
variables (c d : ℕ) 

-- Conditions:
-- Positive integers
axiom pos_c : 0 < c
axiom pos_d : 0 < d

-- Contains 630 terms
axiom term_count : d - c = 630

-- The product of the logarithms equals 2
axiom log_product : (Real.log d) / (Real.log c) = 2

-- Theorem to prove
theorem compute_c_plus_d : c + d = 1260 :=
sorry

end NUMINAMATH_GPT_compute_c_plus_d_l714_71491


namespace NUMINAMATH_GPT_new_salary_correct_l714_71448

-- Define the initial salary and percentage increase as given in the conditions
def initial_salary : ℝ := 10000
def percentage_increase : ℝ := 0.02

-- Define the function that calculates the new salary after a percentage increase
def new_salary (initial_salary : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_salary + (initial_salary * percentage_increase)

-- The theorem statement that proves the new salary is €10,200
theorem new_salary_correct :
  new_salary initial_salary percentage_increase = 10200 := by
  sorry

end NUMINAMATH_GPT_new_salary_correct_l714_71448


namespace NUMINAMATH_GPT_car_owners_without_motorcycles_l714_71485

theorem car_owners_without_motorcycles 
  (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (bicycle_owners : ℕ) (total_own_vehicle : ℕ)
  (h1 : total_adults = 400) (h2 : car_owners = 350) (h3 : motorcycle_owners = 60) (h4 : bicycle_owners = 30)
  (h5 : total_own_vehicle = total_adults)
  : (car_owners - 10 = 340) :=
by
  sorry

end NUMINAMATH_GPT_car_owners_without_motorcycles_l714_71485


namespace NUMINAMATH_GPT_rational_root_of_p_l714_71411

noncomputable def p (n : ℕ) (x : ℚ) : ℚ :=
  x^n + (2 + x)^n + (2 - x)^n

theorem rational_root_of_p :
  ∀ n : ℕ, n > 0 → (∃ x : ℚ, p n x = 0) ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_rational_root_of_p_l714_71411


namespace NUMINAMATH_GPT_max_profit_l714_71493

def fixed_cost : ℝ := 20
def variable_cost_per_unit : ℝ := 10

def total_cost (Q : ℝ) := fixed_cost + variable_cost_per_unit * Q

def revenue (Q : ℝ) := 40 * Q - Q^2

def profit (Q : ℝ) := revenue Q - total_cost Q

def Q_optimized : ℝ := 15

theorem max_profit : profit Q_optimized = 205 := by
  sorry -- Proof goes here.

end NUMINAMATH_GPT_max_profit_l714_71493


namespace NUMINAMATH_GPT_new_avg_weight_l714_71410

-- Definition of the conditions
def original_team_avg_weight : ℕ := 94
def original_team_size : ℕ := 7
def new_player_weight_1 : ℕ := 110
def new_player_weight_2 : ℕ := 60
def total_new_team_size : ℕ := original_team_size + 2

-- Computation of the total weight
def total_weight_original_team : ℕ := original_team_avg_weight * original_team_size
def total_weight_new_team : ℕ := total_weight_original_team + new_player_weight_1 + new_player_weight_2

-- Statement of the theorem
theorem new_avg_weight : total_weight_new_team / total_new_team_size = 92 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_new_avg_weight_l714_71410


namespace NUMINAMATH_GPT_simplify_and_evaluate_l714_71436

theorem simplify_and_evaluate : 
  (1 / (3 - 2) - 1 / (3 + 1)) / (3 / (3^2 - 1)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l714_71436


namespace NUMINAMATH_GPT_time_to_fill_cistern_l714_71463

def pipe_p_rate := (1: ℚ) / 10
def pipe_q_rate := (1: ℚ) / 15
def pipe_r_rate := - (1: ℚ) / 30
def combined_rate_p_q := pipe_p_rate + pipe_q_rate
def combined_rate_q_r := pipe_q_rate + pipe_r_rate
def initial_fill := 2 * combined_rate_p_q
def remaining_fill := 1 - initial_fill
def remaining_time := remaining_fill / combined_rate_q_r

theorem time_to_fill_cistern :
  remaining_time = 20 := by sorry

end NUMINAMATH_GPT_time_to_fill_cistern_l714_71463


namespace NUMINAMATH_GPT_sawing_time_determination_l714_71419

variable (totalLength pieceLength sawTime : Nat)

theorem sawing_time_determination
  (h1 : totalLength = 10)
  (h2 : pieceLength = 2)
  (h3 : sawTime = 10) :
  (totalLength / pieceLength - 1) * sawTime = 40 := by
  sorry

end NUMINAMATH_GPT_sawing_time_determination_l714_71419


namespace NUMINAMATH_GPT_caitlinAgeIsCorrect_l714_71471

-- Define Aunt Anna's age
def auntAnnAge : Nat := 48

-- Define the difference between Aunt Anna's age and 18
def ageDifference : Nat := auntAnnAge - 18

-- Define Brianna's age as twice the difference
def briannaAge : Nat := 2 * ageDifference

-- Define Caitlin's age as 6 years younger than Brianna
def caitlinAge : Nat := briannaAge - 6

-- Theorem to prove Caitlin's age
theorem caitlinAgeIsCorrect : caitlinAge = 54 := by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_caitlinAgeIsCorrect_l714_71471


namespace NUMINAMATH_GPT_solve_problem_l714_71479

def problem_statement (x y : ℕ) : Prop :=
  (x = 3) ∧ (y = 2) → (x^8 + 2 * x^4 * y^2 + y^4) / (x^4 + y^2) = 85

theorem solve_problem : problem_statement 3 2 :=
  by sorry

end NUMINAMATH_GPT_solve_problem_l714_71479


namespace NUMINAMATH_GPT_proof_problem_l714_71431

variable (a b c : ℝ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_prod : a * b * c = 1)
variable (h_ineq : a^2011 + b^2011 + c^2011 < (1 / a)^2011 + (1 / b)^2011 + (1 / c)^2011)

theorem proof_problem : a + b + c < 1 / a + 1 / b + 1 / c := 
  sorry

end NUMINAMATH_GPT_proof_problem_l714_71431


namespace NUMINAMATH_GPT_sam_dimes_example_l714_71409

theorem sam_dimes_example (x y : ℕ) (h₁ : x = 9) (h₂ : y = 7) : x + y = 16 :=
by 
  sorry

end NUMINAMATH_GPT_sam_dimes_example_l714_71409


namespace NUMINAMATH_GPT_second_butcher_packages_l714_71499

theorem second_butcher_packages (a b c: ℕ) (weight_per_package total_weight: ℕ)
    (first_butcher_packages: ℕ) (third_butcher_packages: ℕ)
    (cond1: a = 10) (cond2: b = 8) (cond3: weight_per_package = 4)
    (cond4: total_weight = 100):
    c = (total_weight - (first_butcher_packages * weight_per_package + third_butcher_packages * weight_per_package)) / weight_per_package →
    c = 7 := 
by 
  have first_butcher_packages := 10
  have third_butcher_packages := 8
  have weight_per_package := 4
  have total_weight := 100
  sorry

end NUMINAMATH_GPT_second_butcher_packages_l714_71499


namespace NUMINAMATH_GPT_generatrix_length_l714_71400

theorem generatrix_length (r : ℝ) (l : ℝ) 
  (h_radius : r = Real.sqrt 2) 
  (h_surface : 2 * Real.pi * r = Real.pi * l) : 
  l = 2 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_generatrix_length_l714_71400


namespace NUMINAMATH_GPT_simplify_fraction_l714_71443

theorem simplify_fraction : (3 / 462 + 17 / 42) = 95 / 231 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l714_71443


namespace NUMINAMATH_GPT_highest_daily_profit_and_total_profit_l714_71452

def cost_price : ℕ := 6
def standard_price : ℕ := 10

def price_relative (day : ℕ) : ℤ := 
  match day with
  | 1 => 3
  | 2 => 2
  | 3 => 1
  | 4 => -1
  | 5 => -2
  | _ => 0

def quantity_sold (day : ℕ) : ℕ :=
  match day with
  | 1 => 7
  | 2 => 12
  | 3 => 15
  | 4 => 32
  | 5 => 34
  | _ => 0

noncomputable def selling_price (day : ℕ) : ℤ := standard_price + price_relative day

noncomputable def profit_per_pen (day : ℕ) : ℤ := (selling_price day) - cost_price

noncomputable def daily_profit (day : ℕ) : ℤ := (profit_per_pen day) * (quantity_sold day)

theorem highest_daily_profit_and_total_profit 
  (h_highest_profit: daily_profit 4 = 96) 
  (h_total_profit: daily_profit 1 + daily_profit 2 + daily_profit 3 + daily_profit 4 + daily_profit 5 = 360) : 
  True :=
by
  sorry

end NUMINAMATH_GPT_highest_daily_profit_and_total_profit_l714_71452


namespace NUMINAMATH_GPT_arthur_walk_distance_l714_71492

def blocks_east : ℕ := 8
def blocks_north : ℕ := 15
def block_length : ℚ := 1 / 4

theorem arthur_walk_distance :
  (blocks_east + blocks_north) * block_length = 23 * (1 / 4) := by
  sorry

end NUMINAMATH_GPT_arthur_walk_distance_l714_71492


namespace NUMINAMATH_GPT_fraction_to_decimal_l714_71450

theorem fraction_to_decimal :
  (58 / 200 : ℝ) = 1.16 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l714_71450


namespace NUMINAMATH_GPT_prime_eq_sol_l714_71454

theorem prime_eq_sol {p q x y z : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
  sorry

end NUMINAMATH_GPT_prime_eq_sol_l714_71454


namespace NUMINAMATH_GPT_max_value_in_interval_l714_71437

variable {R : Type*} [OrderedCommRing R]

variables (f : R → R)
variables (odd_f : ∀ x, f (-x) = -f (x))
variables (f_increasing : ∀ x y, 0 < x → x < y → f x < f y)
variables (additive_f : ∀ x y, f (x + y) = f x + f y)
variables (f1_eq_2 : f 1 = 2)

theorem max_value_in_interval : ∀ x ∈ Set.Icc (-3 : R) (-2 : R), f x ≤ f (-2) ∧ f (-2) = -4 :=
by
  sorry

end NUMINAMATH_GPT_max_value_in_interval_l714_71437


namespace NUMINAMATH_GPT_triangle_areas_l714_71444

-- Define points based on the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Triangle DEF vertices
def D : Point := { x := 0, y := 4 }
def E : Point := { x := 6, y := 0 }
def F : Point := { x := 6, y := 5 }

-- Triangle GHI vertices
def G : Point := { x := 0, y := 8 }
def H : Point := { x := 0, y := 6 }
def I : Point := F  -- I and F are the same point

-- Auxiliary function to calculate area of a triangle given its vertices
def area (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- Prove that the areas are correct
theorem triangle_areas :
  area D E F = 15 ∧ area G H I = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_areas_l714_71444


namespace NUMINAMATH_GPT_remainder_of_polynomial_division_l714_71461

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * x^3 - 18 * x^2 + 31 * x - 40

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 5 * x - 10

-- Prove that the remainder when P(x) is divided by D(x) is -10
theorem remainder_of_polynomial_division : (P 2) = -10 := by
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_division_l714_71461
