import Mathlib

namespace average_of_possible_values_l1857_185709

theorem average_of_possible_values 
  (x : ℝ)
  (h : Real.sqrt (2 * x^2 + 5) = Real.sqrt 25) : 
  (x = Real.sqrt 10 ∨ x = -Real.sqrt 10) → (Real.sqrt 10 + (-Real.sqrt 10)) / 2 = 0 :=
by
  sorry

end average_of_possible_values_l1857_185709


namespace fraction_value_l1857_185754

theorem fraction_value : (1 + 3 + 5) / (10 + 6 + 2) = 1 / 2 := 
by
  sorry

end fraction_value_l1857_185754


namespace find_x_l1857_185738

theorem find_x (x : ℝ) (h : 2500 - 1002 / x = 2450) : x = 20.04 :=
by 
  sorry

end find_x_l1857_185738


namespace fraction_neither_cable_nor_vcr_l1857_185740

variable (T : ℕ)
variable (units_with_cable : ℕ := T / 5)
variable (units_with_vcrs : ℕ := T / 10)
variable (units_with_cable_and_vcrs : ℕ := (T / 5) / 3)

theorem fraction_neither_cable_nor_vcr (T : ℕ)
  (h1 : units_with_cable = T / 5)
  (h2 : units_with_vcrs = T / 10)
  (h3 : units_with_cable_and_vcrs = (units_with_cable / 3)) :
  (T - (units_with_cable + (units_with_vcrs - units_with_cable_and_vcrs))) / T = 7 / 10 := 
by
  sorry

end fraction_neither_cable_nor_vcr_l1857_185740


namespace exists_unique_continuous_extension_l1857_185727

noncomputable def F (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) : ℝ → ℝ :=
  sorry

theorem exists_unique_continuous_extension (f : ℚ → ℚ) (hf_bij : Function.Bijective f) (hf_mono : Monotone f) :
  ∃! F : ℝ → ℝ, Continuous F ∧ ∀ x : ℚ, F x = f x :=
sorry

end exists_unique_continuous_extension_l1857_185727


namespace rosalina_received_21_gifts_l1857_185733

def Emilio_gifts : Nat := 11
def Jorge_gifts : Nat := 6
def Pedro_gifts : Nat := 4

def total_gifts : Nat :=
  Emilio_gifts + Jorge_gifts + Pedro_gifts

theorem rosalina_received_21_gifts : total_gifts = 21 := by
  sorry

end rosalina_received_21_gifts_l1857_185733


namespace sin_squared_plus_sin_double_eq_one_l1857_185728

variable (α : ℝ)
variable (h : Real.tan α = 1 / 2)

theorem sin_squared_plus_sin_double_eq_one : Real.sin α ^ 2 + Real.sin (2 * α) = 1 :=
by
  -- sorry to indicate the proof is skipped
  sorry

end sin_squared_plus_sin_double_eq_one_l1857_185728


namespace dealer_profit_percentage_l1857_185759

-- Define the conditions
def cost_price_kg : ℕ := 1000
def given_weight_kg : ℕ := 575

-- Define the weight saved by the dealer
def weight_saved : ℕ := cost_price_kg - given_weight_kg

-- Define the profit percentage formula
def profit_percentage : ℕ → ℕ → ℚ := λ saved total_weight => (saved : ℚ) / (total_weight : ℚ) * 100

-- The main theorem statement
theorem dealer_profit_percentage : profit_percentage weight_saved cost_price_kg = 42.5 :=
by
  sorry

end dealer_profit_percentage_l1857_185759


namespace kims_morning_routine_total_time_l1857_185735

def time_spent_making_coffee := 5 -- in minutes
def time_spent_per_employee_status_update := 2 -- in minutes
def time_spent_per_employee_payroll_update := 3 -- in minutes
def number_of_employees := 9

theorem kims_morning_routine_total_time :
  time_spent_making_coffee +
  (time_spent_per_employee_status_update + time_spent_per_employee_payroll_update) * number_of_employees = 50 :=
by
  sorry

end kims_morning_routine_total_time_l1857_185735


namespace flour_needed_l1857_185772

theorem flour_needed (flour_per_24_cookies : ℝ) (cookies_per_recipe : ℕ) (desired_cookies : ℕ) 
  (h : flour_per_24_cookies = 1.5) (h1 : cookies_per_recipe = 24) (h2 : desired_cookies = 72) : 
  flour_per_24_cookies / cookies_per_recipe * desired_cookies = 4.5 := 
  by {
    -- The proof is omitted
    sorry
  }

end flour_needed_l1857_185772


namespace interior_angle_second_quadrant_l1857_185748

theorem interior_angle_second_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin α * Real.tan α < 0) : 
  π / 2 < α ∧ α < π :=
by
  sorry

end interior_angle_second_quadrant_l1857_185748


namespace range_of_x_l1857_185722

noncomputable def f : ℝ → ℝ := sorry -- Define the function f

variable (f_increasing : ∀ x y, x < y → f x < f y) -- f is increasing
variable (f_at_2 : f 2 = 0) -- f(2) = 0

theorem range_of_x (x : ℝ) : f (x - 2) > 0 ↔ x > 4 :=
by
  sorry

end range_of_x_l1857_185722


namespace percentage_of_y_l1857_185713

theorem percentage_of_y (y : ℝ) : (0.3 * 0.6 * y = 0.18 * y) :=
by {
  sorry
}

end percentage_of_y_l1857_185713


namespace trigonometric_relationship_l1857_185762

theorem trigonometric_relationship (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = (1 - Real.sin β) / Real.cos β) : 
  2 * α + β = π / 2 := 
sorry

end trigonometric_relationship_l1857_185762


namespace gcd_4557_1953_5115_l1857_185779

theorem gcd_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 :=
by
  -- We use 'sorry' to skip the proof part as per the instructions.
  sorry

end gcd_4557_1953_5115_l1857_185779


namespace polynomial_factor_l1857_185717

theorem polynomial_factor (a b : ℝ) :
  (∃ (c d : ℝ), a = 4 * c ∧ b = -3 * c + 4 * d ∧ 40 = 2 * c - 3 * d + 18 ∧ -20 = 2 * d - 9 ∧ 9 = 9) →
  a = 11 ∧ b = -121 / 4 :=
by
  sorry

end polynomial_factor_l1857_185717


namespace relative_error_comparison_l1857_185756

theorem relative_error_comparison :
  (0.05 / 25 = 0.002) ∧ (0.4 / 200 = 0.002) → (0.002 = 0.002) :=
by
  sorry

end relative_error_comparison_l1857_185756


namespace half_of_1_point_6_times_10_pow_6_l1857_185775

theorem half_of_1_point_6_times_10_pow_6 : (1.6 * 10^6) / 2 = 8 * 10^5 :=
by
  sorry

end half_of_1_point_6_times_10_pow_6_l1857_185775


namespace trailing_zeros_310_factorial_l1857_185753

def count_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem trailing_zeros_310_factorial :
  count_trailing_zeros 310 = 76 := by
sorry

end trailing_zeros_310_factorial_l1857_185753


namespace relationship_among_vars_l1857_185789

theorem relationship_among_vars {a b c d : ℝ} (h : (a + 2 * b) / (b + 2 * c) = (c + 2 * d) / (d + 2 * a)) :
  b = 2 * a ∨ a + b + c + d = 0 :=
sorry

end relationship_among_vars_l1857_185789


namespace visitors_inversely_proportional_l1857_185798

theorem visitors_inversely_proportional (k : ℝ) (v₁ v₂ t₁ t₂ : ℝ) (h1 : v₁ * t₁ = k) (h2 : t₁ = 20) (h3 : v₁ = 150) (h4 : t₂ = 30) : v₂ = 100 :=
by
  -- This is a placeholder line; the actual proof would go here.
  sorry

end visitors_inversely_proportional_l1857_185798


namespace find_ratio_of_hyperbola_asymptotes_l1857_185736

theorem find_ratio_of_hyperbola_asymptotes (a b : ℝ) (h : a > b) (hyp : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → |(2 * b / a)| = 1) : 
  a / b = 2 := 
by 
  sorry

end find_ratio_of_hyperbola_asymptotes_l1857_185736


namespace question1_question2_l1857_185731

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Problem 1: Prove the valid solution of x when f(x) = 3 and x ∈ [0, 4]
theorem question1 (h₀ : 0 ≤ 3) (h₁ : 4 ≥ 3) : 
  ∃ (x : ℝ), (f x = 3 ∧ 0 ≤ x ∧ x ≤ 4) → x = 3 :=
by
  sorry

-- Problem 2: Prove the range of f(x) when x ∈ [0, 4]
theorem question2 : 
  ∃ (a b : ℝ), (∀ x, 0 ≤ x ∧ x ≤ 4 → a ≤ f x ∧ f x ≤ b) → a = -1 ∧ b = 8 :=
by
  sorry

end question1_question2_l1857_185731


namespace length_of_platform_l1857_185761

theorem length_of_platform
  (length_of_train time_crossing_platform time_crossing_pole : ℝ) 
  (length_of_train_eq : length_of_train = 400)
  (time_crossing_platform_eq : time_crossing_platform = 45)
  (time_crossing_pole_eq : time_crossing_pole = 30) :
  ∃ (L : ℝ), (400 + L) / time_crossing_platform = length_of_train / time_crossing_pole :=
by {
  use 200,
  sorry
}

end length_of_platform_l1857_185761


namespace tree_distance_l1857_185734

theorem tree_distance 
  (num_trees : ℕ) (dist_first_to_fifth : ℕ) (length_of_road : ℤ) 
  (h1 : num_trees = 8) 
  (h2 : dist_first_to_fifth = 100) 
  (h3 : length_of_road = (dist_first_to_fifth * (num_trees - 1)) / 4 + 3 * dist_first_to_fifth) 
  :
  length_of_road = 175 := 
sorry

end tree_distance_l1857_185734


namespace fraction_pairs_l1857_185705

theorem fraction_pairs (n : ℕ) (h : n > 2009) : 
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 1 ≤ a ∧ a ≤ n ∧
  1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 1 ≤ d ∧ d ≤ n ∧
  1/a + 1/b = 1/c + 1/d := 
sorry

end fraction_pairs_l1857_185705


namespace M_k_max_l1857_185765

noncomputable def J_k (k : ℕ) : ℕ := 5^(k+3) * 2^(k+3) + 648

def M (k : ℕ) : ℕ := 
  if k < 3 then k + 3
  else 3

theorem M_k_max (k : ℕ) : M k = 3 :=
by sorry

end M_k_max_l1857_185765


namespace Jake_has_8_peaches_l1857_185715

variable (Steven Jill Jake : ℕ)

-- Conditions
axiom h1 : Steven = 15
axiom h2 : Steven = Jill + 14
axiom h3 : Jake = Steven - 7

-- Goal
theorem Jake_has_8_peaches : Jake = 8 := by
  sorry

end Jake_has_8_peaches_l1857_185715


namespace hall_volume_l1857_185745

theorem hall_volume (length width : ℝ) (h : ℝ) 
  (h_length : length = 6) 
  (h_width : width = 6) 
  (h_areas : 2 * (length * width) = 4 * (length * h)) :
  length * width * h = 108 :=
by
  sorry

end hall_volume_l1857_185745


namespace minimum_number_of_circles_l1857_185757

-- Define the problem conditions
def conditions_of_problem (circles : ℕ) (n : ℕ) (highlighted_lines : ℕ) (sides_of_regular_2011_gon : ℕ) : Prop :=
  circles ≥ n ∧ highlighted_lines = sides_of_regular_2011_gon

-- The main theorem we need to prove
theorem minimum_number_of_circles :
  ∀ (n circles highlighted_lines sides_of_regular_2011_gon : ℕ),
    sides_of_regular_2011_gon = 2011 ∧ (highlighted_lines = sides_of_regular_2011_gon * 2) ∧ conditions_of_problem circles n highlighted_lines sides_of_regular_2011_gon → n = 504 :=
by
  sorry

end minimum_number_of_circles_l1857_185757


namespace f_neg_def_l1857_185742

variable (f : ℝ → ℝ)
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_def_pos : ∀ x : ℝ, 0 < x → f x = x * (1 + x)

theorem f_neg_def (x : ℝ) (hx : x < 0) : f x = x * (1 - x) := by
  sorry

end f_neg_def_l1857_185742


namespace terminal_side_in_second_quadrant_l1857_185743

theorem terminal_side_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
    0 < α ∧ α < π :=
sorry

end terminal_side_in_second_quadrant_l1857_185743


namespace vertex_of_parabola_l1857_185797

theorem vertex_of_parabola (a b c : ℝ) :
  (∀ x y : ℝ, (x = -2 ∧ y = 5) ∨ (x = 4 ∧ y = 5) ∨ (x = 2 ∧ y = 2) →
    y = a * x^2 + b * x + c) →
  (∃ x_vertex : ℝ, x_vertex = 1) :=
by
  sorry

end vertex_of_parabola_l1857_185797


namespace good_subset_divisible_by_5_l1857_185716

noncomputable def num_good_subsets : ℕ :=
  (Nat.factorial 1000) / ((Nat.factorial 201) * (Nat.factorial (1000 - 201)))

theorem good_subset_divisible_by_5 : num_good_subsets / 5 = (1 / 5) * num_good_subsets := 
sorry

end good_subset_divisible_by_5_l1857_185716


namespace sequence_an_solution_l1857_185739

theorem sequence_an_solution {a : ℕ → ℝ} (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → (1 / a (n + 1) = 1 / a n + 1)) : ∀ n : ℕ, 0 < n → (a n = 1 / n) :=
by
  sorry

end sequence_an_solution_l1857_185739


namespace part1_part2_l1857_185721

theorem part1 (A B C a b c : ℝ) (h1 : 3 * a * Real.cos A = Real.sqrt 6 * (c * Real.cos B + b * Real.cos C)) :
    Real.tan (2 * A) = 2 * Real.sqrt 2 := sorry

theorem part2 (A B C a b c S : ℝ) 
  (h_sin_B : Real.sin (Real.pi / 2 + B) = 2 * Real.sqrt 2 / 3)
  (hc : c = 2 * Real.sqrt 2) :
    S = 2 * Real.sqrt 2 / 3 := sorry

end part1_part2_l1857_185721


namespace x_eq_1_iff_quadratic_eq_zero_l1857_185785

theorem x_eq_1_iff_quadratic_eq_zero :
  ∀ x : ℝ, (x = 1) ↔ (x^2 - 2 * x + 1 = 0) := by
  sorry

end x_eq_1_iff_quadratic_eq_zero_l1857_185785


namespace average_age_is_correct_l1857_185747

-- Define the conditions
def num_men : ℕ := 6
def num_women : ℕ := 9
def average_age_men : ℕ := 57
def average_age_women : ℕ := 52
def total_age_men : ℕ := num_men * average_age_men
def total_age_women : ℕ := num_women * average_age_women
def total_age : ℕ := total_age_men + total_age_women
def total_people : ℕ := num_men + num_women
def average_age_group : ℕ := total_age / total_people

-- The proof will require showing average_age_group is 54, left as sorry.
theorem average_age_is_correct : average_age_group = 54 := sorry

end average_age_is_correct_l1857_185747


namespace map_at_three_l1857_185770

variable (A B : Type)
variable (a : ℝ)
variable (f : ℝ → ℝ)
variable (h_map : ∀ x : ℝ, f x = a * x - 1)
variable (h_cond : f 2 = 3)

theorem map_at_three : f 3 = 5 := by
  sorry

end map_at_three_l1857_185770


namespace problem_f_neg2_equals_2_l1857_185729

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem_f_neg2_equals_2 (f : ℝ → ℝ) (b : ℝ) 
  (h_odd : is_odd_function f)
  (h_def : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 3 * x + b) 
  (h_b : b = 0) :
  f (-2) = 2 :=
by
  sorry

end problem_f_neg2_equals_2_l1857_185729


namespace solution_triple_root_system_l1857_185795

theorem solution_triple_root_system (x y z : ℝ) :
  (x - 1) * (y - 1) * (z - 1) = x * y * z - 1 ∧
  (x - 2) * (y - 2) * (z - 2) = x * y * z - 2 →
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  intro h
  sorry

end solution_triple_root_system_l1857_185795


namespace range_of_a_l1857_185725

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then (x - a) ^ 2 + Real.exp 1 else x / Real.log x + a + 10

theorem range_of_a (a : ℝ) :
    (∀ x, f x a ≥ f 2 a) → (2 ≤ a ∧ a ≤ 6) :=
by
  sorry

end range_of_a_l1857_185725


namespace acid_solution_l1857_185771

theorem acid_solution (m x : ℝ) (h1 : 0 < m) (h2 : m > 50)
  (h3 : (m / 100) * m = (m - 20) / 100 * (m + x)) : x = 20 * m / (m + 20) := 
sorry

end acid_solution_l1857_185771


namespace perpendicular_lines_slope_l1857_185723

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, x + a * y = 1 - a ∧ (a - 2) * x + 3 * y + 2 = 0) → a = 1 / 2 := 
by 
  sorry

end perpendicular_lines_slope_l1857_185723


namespace andrew_cookies_per_day_l1857_185760

/-- Number of days in May --/
def days_in_may : ℤ := 31

/-- Cost per cookie in dollars --/
def cost_per_cookie : ℤ := 15

/-- Total amount spent by Andrew on cookies in dollars --/
def total_amount_spent : ℤ := 1395

/-- Total number of cookies purchased by Andrew --/
def total_cookies : ℤ := total_amount_spent / cost_per_cookie

/-- Number of cookies purchased per day --/
def cookies_per_day : ℤ := total_cookies / days_in_may

theorem andrew_cookies_per_day : cookies_per_day = 3 := by
  sorry

end andrew_cookies_per_day_l1857_185760


namespace janet_hourly_wage_l1857_185780

theorem janet_hourly_wage : 
  ∃ x : ℝ, 
    (20 * x + (5 * 20 + 7 * 20) = 1640) ∧ 
    x = 70 :=
by
  use 70
  sorry

end janet_hourly_wage_l1857_185780


namespace necessary_but_not_sufficient_l1857_185776

def condition1 (a b : ℝ) : Prop :=
  a > b

def statement (a b : ℝ) : Prop :=
  a > b + 1

theorem necessary_but_not_sufficient (a b : ℝ) (h : condition1 a b) : 
  (∀ a b : ℝ, statement a b → condition1 a b) ∧ ¬ (∀ a b : ℝ, condition1 a b → statement a b) :=
by 
  -- Proof skipped
  sorry

end necessary_but_not_sufficient_l1857_185776


namespace B_and_C_mutually_exclusive_but_not_complementary_l1857_185781

-- Define the sample space of the cube
def faces : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define events based on conditions
def event_A (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_B (n : ℕ) : Prop := n = 1 ∨ n = 2
def event_C (n : ℕ) : Prop := n = 4 ∨ n = 5 ∨ n = 6

-- Define mutually exclusive events
def mutually_exclusive (A B : ℕ → Prop) : Prop := ∀ n, A n → ¬ B n

-- Define complementary events (for events over finite sample spaces like faces)
-- Events A and B are complementary if they partition the sample space faces
def complementary (A B : ℕ → Prop) : Prop := (∀ n, n ∈ faces → A n ∨ B n) ∧ (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n)

theorem B_and_C_mutually_exclusive_but_not_complementary :
  mutually_exclusive event_B event_C ∧ ¬ complementary event_B event_C := 
by
  sorry

end B_and_C_mutually_exclusive_but_not_complementary_l1857_185781


namespace andy_wrong_questions_l1857_185788

variables (a b c d : ℕ)

theorem andy_wrong_questions 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 6) 
  (h3 : c = 7) : 
  a = 20 :=
sorry

end andy_wrong_questions_l1857_185788


namespace harry_travel_time_l1857_185763

def t_bus1 : ℕ := 15
def t_bus2 : ℕ := 25
def t_bus_journey : ℕ := t_bus1 + t_bus2
def t_walk : ℕ := t_bus_journey / 2
def t_total : ℕ := t_bus_journey + t_walk

theorem harry_travel_time : t_total = 60 := by
  -- Will be proved afterwards
  sorry

end harry_travel_time_l1857_185763


namespace multiply_repeating_decimals_l1857_185787

noncomputable def repeating_decimal_03 : ℚ := 1 / 33
noncomputable def repeating_decimal_8 : ℚ := 8 / 9

theorem multiply_repeating_decimals : repeating_decimal_03 * repeating_decimal_8 = 8 / 297 := by 
  sorry

end multiply_repeating_decimals_l1857_185787


namespace quadratic_root_properties_l1857_185755

theorem quadratic_root_properties (b : ℝ) (t : ℝ) :
  (∀ x : ℝ, x^2 + b*x - 2 = 0 → (x = 2 ∨ x = t)) →
  b = -1 ∧ t = -1 :=
by
  sorry

end quadratic_root_properties_l1857_185755


namespace alpha_beta_sum_l1857_185750

theorem alpha_beta_sum (α β : ℝ) (h1 : α^3 - 3 * α^2 + 5 * α - 17 = 0) (h2 : β^3 - 3 * β^2 + 5 * β + 11 = 0) : α + β = 2 := 
by
  sorry

end alpha_beta_sum_l1857_185750


namespace f_f_f_f_f_3_eq_4_l1857_185712

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_f_f_f_f_3_eq_4 : f (f (f (f (f 3)))) = 4 := 
  sorry

end f_f_f_f_f_3_eq_4_l1857_185712


namespace shift_parabola_left_l1857_185799

theorem shift_parabola_left (x : ℝ) : (x + 1)^2 = y ↔ x^2 = y :=
sorry

end shift_parabola_left_l1857_185799


namespace four_digit_div_by_99_then_sum_div_by_18_l1857_185700

/-- 
If a whole number with at most four digits is divisible by 99, then 
the sum of its digits is divisible by 18. 
-/
theorem four_digit_div_by_99_then_sum_div_by_18 (n : ℕ) (h1 : n < 10000) (h2 : 99 ∣ n) : 
  18 ∣ (n.digits 10).sum := 
sorry

end four_digit_div_by_99_then_sum_div_by_18_l1857_185700


namespace number_of_players_l1857_185793

-- Definitions based on conditions in the problem
def cost_of_gloves : ℕ := 6
def cost_of_helmet : ℕ := cost_of_gloves + 7
def cost_of_cap : ℕ := 3
def total_expenditure : ℕ := 2968

-- Total cost for one player
def cost_per_player : ℕ := 2 * (cost_of_gloves + cost_of_helmet) + cost_of_cap

-- Statement to prove: number of players
theorem number_of_players : total_expenditure / cost_per_player = 72 := 
by
  sorry

end number_of_players_l1857_185793


namespace outfit_count_l1857_185719

theorem outfit_count 
  (S P T J : ℕ) 
  (hS : S = 8) 
  (hP : P = 5) 
  (hT : T = 4) 
  (hJ : J = 3) : 
  S * P * (T + 1) * (J + 1) = 800 := by 
  sorry

end outfit_count_l1857_185719


namespace fractional_addition_l1857_185778

theorem fractional_addition : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by
  sorry

end fractional_addition_l1857_185778


namespace sum_of_squares_of_coeffs_l1857_185784

theorem sum_of_squares_of_coeffs (c1 c2 c3 c4 : ℝ) (h1 : c1 = 3) (h2 : c2 = 6) (h3 : c3 = 15) (h4 : c4 = 6) :
  c1^2 + c2^2 + c3^2 + c4^2 = 306 :=
by
  sorry

end sum_of_squares_of_coeffs_l1857_185784


namespace gcd_lcm_product_l1857_185737

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 3 * 5^2) (h2 : b = 5^3) : 
  Nat.gcd a b * Nat.lcm a b = 9375 := by
  sorry

end gcd_lcm_product_l1857_185737


namespace roots_greater_than_two_l1857_185704

variable {x m : ℝ}

theorem roots_greater_than_two (h : ∀ x, x^2 - 2 * m * x + 4 = 0 → (∃ a b : ℝ, a > 2 ∧ b < 2 ∧ x = a ∨ x = b)) : 
  m > 2 :=
by
  sorry

end roots_greater_than_two_l1857_185704


namespace payment_ways_l1857_185744

-- Define basic conditions and variables
variables {x y z : ℕ}

-- Define the main problem as a Lean statement
theorem payment_ways : 
  ∃ (n : ℕ), n = 9 ∧ 
             (∀ x y z : ℕ, 
              x + y + z ≤ 10 ∧ 
              x + 2 * y + 5 * z = 18 ∧ 
              x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
              (x > 0 ∨ y > 0) ∧ (y > 0 ∨ z > 0) ∧ (z > 0 ∨ x > 0) → 
              n = 9) := 
sorry

end payment_ways_l1857_185744


namespace largest_sum_of_digits_l1857_185726

theorem largest_sum_of_digits (a b c : ℕ) (y : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9) (h4 : 1 ≤ y ∧ y ≤ 10) (h5 : (1000 * (a * 100 + b * 10 + c)) = 1000) : 
  a + b + c = 8 :=
sorry

end largest_sum_of_digits_l1857_185726


namespace division_of_fractions_l1857_185796

theorem division_of_fractions :
  (5 / 6) / (9 / 10) = 25 / 27 :=
by sorry

end division_of_fractions_l1857_185796


namespace percentage_salt_solution_l1857_185786

theorem percentage_salt_solution (P : ℝ) (V_initial V_added V_final : ℝ) (C_initial C_final : ℝ) :
  V_initial = 30 ∧ C_initial = 0.20 ∧ V_final = 60 ∧ C_final = 0.40 → 
  V_added = 30 → 
  (C_initial * V_initial + (P / 100) * V_added) / V_final = C_final →
  P = 60 :=
by
  intro h
  sorry

end percentage_salt_solution_l1857_185786


namespace no_odd_m_solution_l1857_185782

theorem no_odd_m_solution : ∀ (m n : ℕ), 0 < m → 0 < n → (5 * n = m * n - 3 * m) → ¬ Odd m :=
by
  intros m n hm hn h_eq
  sorry

end no_odd_m_solution_l1857_185782


namespace find_value_of_x_l1857_185751

theorem find_value_of_x :
  ∃ x : ℝ, (0.65 * x = 0.20 * 747.50) ∧ x = 230 :=
by
  sorry

end find_value_of_x_l1857_185751


namespace video_files_initial_l1857_185724

theorem video_files_initial (V : ℕ) (h1 : 4 + V - 23 = 2) : V = 21 :=
by 
  sorry

end video_files_initial_l1857_185724


namespace negation_proposition_iff_l1857_185701

-- Define propositions and their components
def P (x : ℝ) : Prop := x > 1
def Q (x : ℝ) : Prop := x^2 > 1

-- State the proof problem
theorem negation_proposition_iff (x : ℝ) : ¬ (P x → Q x) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by 
  sorry

end negation_proposition_iff_l1857_185701


namespace solve_system_l1857_185730

theorem solve_system (x y z : ℝ) 
  (h1 : 19 * (x + y) + 17 = 19 * (-x + y) - 21)
  (h2 : 5 * x - 3 * z = 11 * y - 7) : 
  x = -1 ∧ z = -11 * y / 3 + 2 / 3 :=
by sorry

end solve_system_l1857_185730


namespace shaded_region_area_is_48pi_l1857_185707

open Real

noncomputable def small_circle_radius : ℝ := 4
noncomputable def small_circle_area : ℝ := π * small_circle_radius^2
noncomputable def large_circle_radius : ℝ := 2 * small_circle_radius
noncomputable def large_circle_area : ℝ := π * large_circle_radius^2
noncomputable def shaded_region_area : ℝ := large_circle_area - small_circle_area

theorem shaded_region_area_is_48pi :
  shaded_region_area = 48 * π := by
    sorry

end shaded_region_area_is_48pi_l1857_185707


namespace each_shopper_will_receive_amount_l1857_185767

/-- Definitions of the given conditions -/
def isabella_has_more_than_sam : ℕ := 45
def isabella_has_more_than_giselle : ℕ := 15
def giselle_money : ℕ := 120

/-- Calculation based on the provided conditions -/
def isabella_money : ℕ := giselle_money + isabella_has_more_than_giselle
def sam_money : ℕ := isabella_money - isabella_has_more_than_sam
def total_money : ℕ := isabella_money + sam_money + giselle_money

/-- The total amount each shopper will receive when the donation is shared equally -/
def money_each_shopper_receives : ℕ := total_money / 3

/-- Main theorem to prove the statement derived from the problem -/
theorem each_shopper_will_receive_amount :
  money_each_shopper_receives = 115 := by
  sorry

end each_shopper_will_receive_amount_l1857_185767


namespace coordinates_of_point_A_in_third_quadrant_l1857_185703

def point_in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := abs y

def distance_to_y_axis (x : ℝ) : ℝ := abs x

theorem coordinates_of_point_A_in_third_quadrant 
  (x y : ℝ)
  (h1 : point_in_third_quadrant x y)
  (h2 : distance_to_x_axis y = 2)
  (h3 : distance_to_y_axis x = 3) :
  (x, y) = (-3, -2) :=
  sorry

end coordinates_of_point_A_in_third_quadrant_l1857_185703


namespace cos_sum_eq_one_l1857_185746

theorem cos_sum_eq_one (α β γ : ℝ) 
  (h1 : α + β + γ = Real.pi) 
  (h2 : Real.tan ((β + γ - α) / 4) + Real.tan ((γ + α - β) / 4) + Real.tan ((α + β - γ) / 4) = 1) :
  Real.cos α + Real.cos β + Real.cos γ = 1 :=
sorry

end cos_sum_eq_one_l1857_185746


namespace consecutive_numbers_sum_digits_l1857_185708

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem consecutive_numbers_sum_digits :
  ∃ n : ℕ, sum_of_digits n = 52 ∧ sum_of_digits (n + 4) = 20 := 
sorry

end consecutive_numbers_sum_digits_l1857_185708


namespace division_remainder_l1857_185732

theorem division_remainder (dividend divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 15) 
  (h_quotient : quotient = 9) 
  (h_dividend_eq : dividend = 136) 
  (h_eq : dividend = (divisor * quotient) + remainder) : 
  remainder = 1 :=
by
  sorry

end division_remainder_l1857_185732


namespace question_l1857_185792

-- Let x and y be real numbers.
variables (x y : ℝ)

-- Proposition A: x + y ≠ 8
def PropA : Prop := x + y ≠ 8

-- Proposition B: x ≠ 2 ∨ y ≠ 6
def PropB : Prop := x ≠ 2 ∨ y ≠ 6

-- We need to prove that PropA is a sufficient but not necessary condition for PropB.
theorem question : (PropA x y → PropB x y) ∧ ¬ (PropB x y → PropA x y) :=
sorry

end question_l1857_185792


namespace part_1_part_2_l1857_185752

-- Conditions and definitions
noncomputable def triangle_ABC (a b c S : ℝ) (A B C : ℝ) :=
  a * Real.sin B = -b * Real.sin (A + Real.pi / 3) ∧
  S = Real.sqrt 3 / 4 * c^2

-- 1. Prove A = 5 * Real.pi / 6
theorem part_1 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  A = 5 * Real.pi / 6 :=
  sorry

-- 2. Prove sin C = sqrt 7 / 14 given S = sqrt 3 / 4 * c^2
theorem part_2 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  Real.sin C = Real.sqrt 7 / 14 :=
  sorry

end part_1_part_2_l1857_185752


namespace given_system_solution_l1857_185741

noncomputable def solve_system : Prop :=
  ∃ x y z : ℝ, 
  x + y + z = 1 ∧ 
  x^2 + y^2 + z^2 = 1 ∧ 
  x^3 + y^3 + z^3 = 89 / 125 ∧ 
  (x = 2 / 5 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = 2 / 5 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 - Real.sqrt 33) / 10 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = 2 / 5 ∧ z = (3 + Real.sqrt 33) / 10 ∨ 
   x = (3 + Real.sqrt 33) / 10 ∧ y = (3 - Real.sqrt 33) / 10 ∧ z = 2 / 5 ∨ 
   x = (3 - Real.sqrt 33) / 10 ∧ y = (3 + Real.sqrt 33) / 10 ∧ z = 2 / 5)

theorem given_system_solution : solve_system :=
sorry

end given_system_solution_l1857_185741


namespace number_of_periods_l1857_185758

-- Definitions based on conditions
def students : ℕ := 32
def time_per_student : ℕ := 5
def period_duration : ℕ := 40

-- Theorem stating the equivalent proof problem
theorem number_of_periods :
  (students * time_per_student) / period_duration = 4 :=
sorry

end number_of_periods_l1857_185758


namespace motorcycle_wheels_l1857_185702

/--
In a parking lot, there are cars and motorcycles. Each car has 5 wheels (including one spare) 
and each motorcycle has a certain number of wheels. There are 19 cars in the parking lot.
Altogether all vehicles have 117 wheels. There are 11 motorcycles at the parking lot.
--/
theorem motorcycle_wheels (num_cars num_motorcycles total_wheels wheels_per_car wheels_per_motorcycle : ℕ)
  (h1 : wheels_per_car = 5) 
  (h2 : num_cars = 19) 
  (h3 : total_wheels = 117) 
  (h4 : num_motorcycles = 11) 
  : wheels_per_motorcycle = 2 :=
by
  sorry

end motorcycle_wheels_l1857_185702


namespace solve_g_eq_g_inv_l1857_185791

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5

noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem solve_g_eq_g_inv : 
  ∃ x : ℝ, g x = g_inv x ∧ x = 5 / 3 :=
by
  sorry

end solve_g_eq_g_inv_l1857_185791


namespace point_within_region_l1857_185764

theorem point_within_region (a : ℝ) (h : 2 * a + 2 < 4) : a < 1 := 
sorry

end point_within_region_l1857_185764


namespace meaningful_sqrt_range_l1857_185749

theorem meaningful_sqrt_range (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 :=
by
  sorry

end meaningful_sqrt_range_l1857_185749


namespace add_fractions_l1857_185773

theorem add_fractions :
  (8:ℚ) / 19 + 5 / 57 = 29 / 57 :=
sorry

end add_fractions_l1857_185773


namespace percentage_of_candidates_selected_in_State_A_is_6_l1857_185768

-- Definitions based on conditions
def candidates_appeared : ℕ := 8400
def candidates_selected_B : ℕ := (7 * candidates_appeared) / 100 -- 7% of 8400
def extra_candidates_selected : ℕ := 84
def candidates_selected_A : ℕ := candidates_selected_B - extra_candidates_selected

-- Definition based on the goal proof
def percentage_selected_A : ℕ := (candidates_selected_A * 100) / candidates_appeared

-- The theorem we need to prove
theorem percentage_of_candidates_selected_in_State_A_is_6 :
  percentage_selected_A = 6 :=
by
  sorry

end percentage_of_candidates_selected_in_State_A_is_6_l1857_185768


namespace range_of_x_l1857_185711

def valid_domain (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x ≠ 4)

theorem range_of_x : ∀ x : ℝ, valid_domain x ↔ (x ≤ 3) :=
by sorry

end range_of_x_l1857_185711


namespace share_apples_l1857_185794

theorem share_apples (h : 9 / 3 = 3) : true :=
sorry

end share_apples_l1857_185794


namespace complete_square_rewrite_l1857_185777

theorem complete_square_rewrite (j i : ℂ) :
  let c := 8
  let p := (3 * i / 8 : ℂ)
  let q := (137 / 8 : ℂ)
  (8 * j^2 + 6 * i * j + 16 = c * (j + p)^2 + q) →
  q / p = - (137 * i / 3) :=
by
  sorry

end complete_square_rewrite_l1857_185777


namespace intersection_points_number_of_regions_l1857_185718

-- Given n lines on a plane, any two of which are not parallel
-- and no three of which intersect at the same point,
-- prove the number of intersection points of these lines

theorem intersection_points (n : ℕ) (h_n : 0 < n) : 
  ∃ a_n : ℕ, a_n = n * (n - 1) / 2 := by
  sorry

-- Given n lines on a plane, any two of which are not parallel
-- and no three of which intersect at the same point,
-- prove the number of regions these lines form

theorem number_of_regions (n : ℕ) (h_n : 0 < n) :
  ∃ R_n : ℕ, R_n = n * (n + 1) / 2 + 1 := by
  sorry

end intersection_points_number_of_regions_l1857_185718


namespace Nora_to_Lulu_savings_ratio_l1857_185783

-- Definitions
def L : ℕ := 6
def T (N : ℕ) : Prop := N = 3 * (N / 3)
def total_savings (N : ℕ) : Prop := 6 + N + (N / 3) = 46

-- Theorem statement
theorem Nora_to_Lulu_savings_ratio (N : ℕ) (hN_T : T N) (h_total_savings : total_savings N) :
  N / L = 5 :=
by
  -- Proof will be provided here
  sorry

end Nora_to_Lulu_savings_ratio_l1857_185783


namespace fencing_required_l1857_185714

theorem fencing_required (length width area : ℕ) (length_eq : length = 30) (area_eq : area = 810) 
  (field_area : length * width = area) : 2 * length + width = 87 := 
by
  sorry

end fencing_required_l1857_185714


namespace distance_focus_directrix_l1857_185720

theorem distance_focus_directrix (y x p : ℝ) (h : y^2 = 4 * x) (hp : 2 * p = 4) : p = 2 :=
by sorry

end distance_focus_directrix_l1857_185720


namespace circle_equation_l1857_185774

theorem circle_equation :
  ∃ (a : ℝ) (x y : ℝ), 
    (2 * a + y - 1 = 0 ∧ (x = 3 ∧ y = 0) ∧ (x = 0 ∧ y = 1)) →
    (x - 1) ^ 2 + (y + 1) ^ 2 = 5 := by
  sorry

end circle_equation_l1857_185774


namespace system_of_equations_solution_l1857_185769

theorem system_of_equations_solution :
  ∃ (a b : ℤ), (2 * (2 : ℤ) + b = a ∧ (2 : ℤ) + b = 3 ∧ a = 5 ∧ b = 1) :=
by
  sorry

end system_of_equations_solution_l1857_185769


namespace sum_divisible_by_5_and_7_l1857_185790

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_divisible_by_5_and_7 (A B : ℕ) (hA_prime : is_prime A) 
  (hB_prime : is_prime B) (hA_minus_3_prime : is_prime (A - 3)) 
  (hA_plus_3_prime : is_prime (A + 3)) (hB_eq_2 : B = 2) : 
  5 ∣ (A + B + (A - 3) + (A + 3)) ∧ 7 ∣ (A + B + (A - 3) + (A + 3)) := by 
  sorry

end sum_divisible_by_5_and_7_l1857_185790


namespace max_rides_day1_max_rides_day2_l1857_185766

open List 

def daily_budget : ℤ := 10

def ride_prices_day1 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 5), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6)]

def ride_prices_day2 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 7), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6), ("Haunted house", 4)]

def max_rides (budget : ℤ) (prices : List (String × ℤ)) : ℤ :=
  sorry -- We'll assume this calculates the max number of rides correctly based on the given budget and prices.

theorem max_rides_day1 : max_rides daily_budget ride_prices_day1 = 3 := by
  sorry 

theorem max_rides_day2 : max_rides daily_budget ride_prices_day2 = 3 := by
  sorry 

end max_rides_day1_max_rides_day2_l1857_185766


namespace gcd_765432_654321_l1857_185706

-- Define the two integers 765432 and 654321
def a : ℕ := 765432
def b : ℕ := 654321

-- State the main theorem to prove the gcd
theorem gcd_765432_654321 : Nat.gcd a b = 3 := 
by 
  sorry

end gcd_765432_654321_l1857_185706


namespace seashells_left_l1857_185710

theorem seashells_left (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) :
  initial_seashells = 75 → given_seashells = 18 → remaining_seashells = initial_seashells - given_seashells → remaining_seashells = 57 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end seashells_left_l1857_185710
