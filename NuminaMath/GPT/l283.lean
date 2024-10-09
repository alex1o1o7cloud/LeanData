import Mathlib

namespace find_coordinates_of_C_l283_28373

structure Point where
  x : ℝ
  y : ℝ

def parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = C.x - D.x ∧ B.y - A.y = C.y - D.y) ∧
  (D.x - A.x = C.x - B.x ∧ D.y - A.y = C.y - B.y)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨7, 3⟩
def D : Point := ⟨3, 7⟩
def C : Point := ⟨8, 7⟩

theorem find_coordinates_of_C :
  parallelogram A B C D → C = ⟨8, 7⟩ :=
by
  intro h
  have h₁ := h.1.1
  have h₂ := h.1.2
  have h₃ := h.2.1
  have h₄ := h.2.2
  sorry

end find_coordinates_of_C_l283_28373


namespace directrix_of_parabola_l283_28339

theorem directrix_of_parabola (y x : ℝ) (h_eq : y^2 = 8 * x) :
  x = -2 :=
sorry

end directrix_of_parabola_l283_28339


namespace isosceles_triangle_angle_B_l283_28320

theorem isosceles_triangle_angle_B :
  ∀ (A B C : ℝ), (B = C) → (C = 3 * A) → (A + B + C = 180) → (B = 540 / 7) :=
by
  intros A B C h1 h2 h3
  sorry

end isosceles_triangle_angle_B_l283_28320


namespace number_of_integers_in_original_list_l283_28350

theorem number_of_integers_in_original_list :
  ∃ n m : ℕ, (m + 2) * (n + 1) = m * n + 15 ∧
             (m + 1) * (n + 2) = m * n + 16 ∧
             n = 4 :=
by {
  sorry
}

end number_of_integers_in_original_list_l283_28350


namespace negative_exponent_example_l283_28362

theorem negative_exponent_example : 3^(-2 : ℤ) = (1 : ℚ) / (3^2) :=
by sorry

end negative_exponent_example_l283_28362


namespace both_questions_correct_l283_28384

-- Define variables as constants
def nA : ℝ := 0.85  -- 85%
def nB : ℝ := 0.70  -- 70%
def nAB : ℝ := 0.60 -- 60%

theorem both_questions_correct:
  nAB = 0.60 := by
  sorry

end both_questions_correct_l283_28384


namespace lowest_fraction_combine_two_slowest_l283_28351

def rate_a (hours : ℕ) : ℚ := 1 / 4
def rate_b (hours : ℕ) : ℚ := 1 / 5
def rate_c (hours : ℕ) : ℚ := 1 / 8

theorem lowest_fraction_combine_two_slowest : 
  (rate_b 1 + rate_c 1) = 13 / 40 :=
by sorry

end lowest_fraction_combine_two_slowest_l283_28351


namespace min_value_expression_l283_28341

/--
  Prove that the minimum value of the expression (xy - 2)^2 + (x + y - 1)^2 
  for real numbers x and y is 2.
--/
theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, (a * b - 2)^2 + (a + b - 1)^2 ≥ (x * y - 2)^2 + (x + y - 1)^2 ) ∧ 
  (x * y - 2)^2 + (x + y - 1)^2 = 2 :=
by
  sorry

end min_value_expression_l283_28341


namespace move_line_up_l283_28337

/-- Define the original line equation as y = 3x - 2 -/
def original_line (x : ℝ) : ℝ := 3 * x - 2

/-- Define the resulting line equation as y = 3x + 4 -/
def resulting_line (x : ℝ) : ℝ := 3 * x + 4

theorem move_line_up (x : ℝ) : resulting_line x = original_line x + 6 :=
by
  sorry

end move_line_up_l283_28337


namespace intersection_of_A_and_B_l283_28311

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}
def B : Set ℝ := {y | 0 ≤ y ∧ y < 4}

theorem intersection_of_A_and_B : A ∩ B = {z | 2 ≤ z ∧ z < 4} :=
by
  sorry

end intersection_of_A_and_B_l283_28311


namespace binom_prod_l283_28399

theorem binom_prod : (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 := by
  sorry

end binom_prod_l283_28399


namespace min_value_expression_l283_28359

-- Define the given problem conditions and statement
theorem min_value_expression :
  ∀ (x y : ℝ), 0 < x → 0 < y → 6 ≤ (y / x) + (16 * x / (2 * x + y)) :=
by
  sorry

end min_value_expression_l283_28359


namespace school_sports_event_l283_28389

theorem school_sports_event (x y z : ℤ) (hx : x > y) (hy : y > z) (hz : z > 0)
  (points_A points_B points_E : ℤ) (ha : points_A = 22) (hb : points_B = 9) 
  (he : points_E = 9) (vault_winner_B : True) :
  ∃ n : ℕ, n = 5 ∧ second_place_grenade_throwing_team = 8^B :=
by
  sorry

end school_sports_event_l283_28389


namespace not_all_sets_of_10_segments_form_triangle_l283_28393

theorem not_all_sets_of_10_segments_form_triangle :
  ¬ ∀ (segments : Fin 10 → ℝ), ∃ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (segments a + segments b > segments c) ∧
    (segments a + segments c > segments b) ∧
    (segments b + segments c > segments a) :=
by
  sorry

end not_all_sets_of_10_segments_form_triangle_l283_28393


namespace range_of_a_squared_minus_2b_l283_28336

variable (a b : ℝ)

def quadratic_has_two_real_roots_in_01 (a b : ℝ) : Prop :=
  b ≥ 0 ∧ 1 + a + b ≥ 0 ∧ -2 ≤ a ∧ a ≤ 0 ∧ a^2 - 4 * b ≥ 0

theorem range_of_a_squared_minus_2b (a b : ℝ)
  (h : quadratic_has_two_real_roots_in_01 a b) : 0 ≤ a^2 - 2 * b ∧ a^2 - 2 * b ≤ 2 :=
sorry

end range_of_a_squared_minus_2b_l283_28336


namespace school_badminton_rackets_l283_28302

theorem school_badminton_rackets :
  ∃ (x y : ℕ), x + y = 30 ∧ 50 * x + 40 * y = 1360 ∧ x = 16 ∧ y = 14 :=
by
  sorry

end school_badminton_rackets_l283_28302


namespace system_of_equations_solution_exists_l283_28301

theorem system_of_equations_solution_exists :
  ∃ (x y : ℝ), 
    (4 * x^2 + 8 * x * y + 16 * y^2 + 2 * x + 20 * y = -7) ∧
    (2 * x^2 - 16 * x * y + 8 * y^2 - 14 * x + 20 * y = -11) ∧
    (x = 1/2) ∧ (y = -3/4) :=
by
  sorry

end system_of_equations_solution_exists_l283_28301


namespace max_P_l283_28312

noncomputable def P (a b : ℝ) : ℝ :=
  (a^2 + 6*b + 1) / (a^2 + a)

theorem max_P (a b x1 x2 x3 : ℝ) (h1 : a = x1 + x2 + x3) (h2 : a = x1 * x2 * x3) (h3 : ab = x1 * x2 + x2 * x3 + x3 * x1) 
    (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
    P a b ≤ (9 + Real.sqrt 3) / 9 := 
sorry

end max_P_l283_28312


namespace find_d_and_r_l283_28356

theorem find_d_and_r (d r : ℤ)
  (h1 : 1210 % d = r)
  (h2 : 1690 % d = r)
  (h3 : 2670 % d = r) :
  d - 4 * r = -20 := sorry

end find_d_and_r_l283_28356


namespace negation_proposition_l283_28382

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 < 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≥ 0) :=
by 
  sorry

end negation_proposition_l283_28382


namespace problem_statement_l283_28377

theorem problem_statement (n m : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : 
  (n * 5^n)^n = m * 5^9 ↔ n = 3 ∧ m = 27 :=
by {
  sorry
}

end problem_statement_l283_28377


namespace continuous_stripe_probability_is_3_16_l283_28319

-- Define the stripe orientation enumeration
inductive StripeOrientation
| diagonal
| straight

-- Define the face enumeration
inductive Face
| front
| back
| left
| right
| top
| bottom

-- Total number of stripe combinations (2^6 for each face having 2 orientations)
def total_combinations : ℕ := 2^6

-- Number of combinations for continuous stripes along length, width, and height
def length_combinations : ℕ := 2^2 -- 4 combinations
def width_combinations : ℕ := 2^2  -- 4 combinations
def height_combinations : ℕ := 2^2 -- 4 combinations

-- Total number of continuous stripe combinations across all dimensions
def total_continuous_stripe_combinations : ℕ :=
  length_combinations + width_combinations + height_combinations

-- Probability calculation
def continuous_stripe_probability : ℚ :=
  total_continuous_stripe_combinations / total_combinations

-- Final theorem statement
theorem continuous_stripe_probability_is_3_16 :
  continuous_stripe_probability = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_is_3_16_l283_28319


namespace total_amount_paid_l283_28322

def apples_kg := 8
def apples_rate := 70
def mangoes_kg := 9
def mangoes_rate := 65
def oranges_kg := 5
def oranges_rate := 50
def bananas_kg := 3
def bananas_rate := 30

def total_amount := (apples_kg * apples_rate) + (mangoes_kg * mangoes_rate) + (oranges_kg * oranges_rate) + (bananas_kg * bananas_rate)

theorem total_amount_paid : total_amount = 1485 := by
  sorry

end total_amount_paid_l283_28322


namespace carla_needs_24_cans_l283_28347

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

end carla_needs_24_cans_l283_28347


namespace carla_total_marbles_l283_28368

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem carla_total_marbles : initial_marbles + bought_marbles = 321.0 := by
  sorry

end carla_total_marbles_l283_28368


namespace sum_of_digits_9x_l283_28355

theorem sum_of_digits_9x (a b c d e : ℕ) (x : ℕ) :
  (1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9) →
  x = 10000 * a + 1000 * b + 100 * c + 10 * d + e →
  (b - a) + (c - b) + (d - c) + (e - d) + (10 - e) = 9 :=
by
  sorry

end sum_of_digits_9x_l283_28355


namespace largest_possible_value_of_b_l283_28379

theorem largest_possible_value_of_b (b : ℚ) (h : (3 * b + 4) * (b - 2) = 9 * b) : b ≤ 4 :=
sorry

end largest_possible_value_of_b_l283_28379


namespace arithmetic_expression_l283_28340

theorem arithmetic_expression : (5 * 7 - 6 + 2 * 12 + 2 * 6 + 7 * 3) = 86 :=
by
  sorry

end arithmetic_expression_l283_28340


namespace counterexample_to_conjecture_l283_28317

theorem counterexample_to_conjecture (n : ℕ) (h : n > 5) : 
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) ∨
  ¬ (∃ a b c : ℕ, (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (a + b + c = n)) :=
sorry

end counterexample_to_conjecture_l283_28317


namespace male_student_number_l283_28316

theorem male_student_number (year class_num student_num : ℕ) (h_year : year = 2011) (h_class : class_num = 6) (h_student : student_num = 23) : 
  (100000 * year + 1000 * class_num + 10 * student_num + 1 = 116231) :=
by
  sorry

end male_student_number_l283_28316


namespace range_of_a_if_ineq_has_empty_solution_l283_28360

theorem range_of_a_if_ineq_has_empty_solution (a : ℝ) :
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) → -2 ≤ a ∧ a < 6/5 :=
by
  sorry

end range_of_a_if_ineq_has_empty_solution_l283_28360


namespace minimum_value_of_expression_l283_28346

open Real

noncomputable def f (x y z : ℝ) : ℝ := (x + 2 * y) / (x * y * z)

theorem minimum_value_of_expression :
  ∀ (x y z : ℝ),
    x > 0 → y > 0 → z > 0 →
    x + y + z = 1 →
    x = 2 * y →
    f x y z = 8 :=
by
  intro x y z x_pos y_pos z_pos h_sum h_xy
  sorry

end minimum_value_of_expression_l283_28346


namespace total_area_for_building_l283_28331

theorem total_area_for_building (num_sections : ℕ) (area_per_section : ℝ) (open_space_percentage : ℝ) :
  num_sections = 7 →
  area_per_section = 9473 →
  open_space_percentage = 0.15 →
  (num_sections * (area_per_section * (1 - open_space_percentage))) = 56364.35 :=
by
  intros h1 h2 h3
  sorry

end total_area_for_building_l283_28331


namespace dragons_at_meeting_l283_28338

def dragon_meeting : Prop :=
  ∃ (x y : ℕ), 
    (2 * x + 7 * y = 26) ∧ 
    (x + y = 8)

theorem dragons_at_meeting : dragon_meeting :=
by
  sorry

end dragons_at_meeting_l283_28338


namespace ab_product_eq_four_l283_28321

theorem ab_product_eq_four (a b : ℝ) (h1: 0 < a) (h2: 0 < b) 
  (h3: (1/2) * (4 / a) * (6 / b) = 3) : 
  a * b = 4 :=
by 
  sorry

end ab_product_eq_four_l283_28321


namespace proof_problem_l283_28372

theorem proof_problem 
  (A a B b : ℝ) 
  (h1 : |A - 3 * a| ≤ 1 - a) 
  (h2 : |B - 3 * b| ≤ 1 - b) 
  (h3 : 0 < a) 
  (h4 : 0 < b) :
  (|((A * B) / 3) - 3 * (a * b)|) - 3 * (a * b) ≤ 1 - (a * b) :=
sorry

end proof_problem_l283_28372


namespace binary_arithmetic_l283_28397

theorem binary_arithmetic 
  : (0b10110 + 0b1011 - 0b11100 + 0b11101 = 0b100010) :=
by
  sorry

end binary_arithmetic_l283_28397


namespace manny_gave_2_marbles_l283_28334

-- Define the total number of marbles
def total_marbles : ℕ := 36

-- Define the ratio parts for Mario and Manny
def mario_ratio : ℕ := 4
def manny_ratio : ℕ := 5

-- Define the total ratio parts
def total_ratio : ℕ := mario_ratio + manny_ratio

-- Define the number of marbles Manny has after giving some away
def manny_marbles_now : ℕ := 18

-- Calculate the marbles per part based on the ratio and total marbles
def marbles_per_part : ℕ := total_marbles / total_ratio

-- Calculate the number of marbles Manny originally had
def manny_marbles_original : ℕ := manny_ratio * marbles_per_part

-- Formulate the theorem
theorem manny_gave_2_marbles : manny_marbles_original - manny_marbles_now = 2 := by
  sorry

end manny_gave_2_marbles_l283_28334


namespace number_of_zeros_l283_28365

noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.log x)

theorem number_of_zeros (n : ℕ) : (1 < x ∧ x < Real.exp Real.pi) → (∃! x : ℝ, g x = 0 ∧ 1 < x ∧ x < Real.exp Real.pi) → n = 1 :=
sorry

end number_of_zeros_l283_28365


namespace find_a_l283_28343

theorem find_a (a : ℝ) (M : Set ℝ) (N : Set ℝ) : 
  M = {1, 3} → N = {1 - a, 3} → (M ∪ N) = {1, 2, 3} → a = -1 :=
by
  intros hM hN hUnion
  sorry

end find_a_l283_28343


namespace translate_parabola_upwards_l283_28376

theorem translate_parabola_upwards (x y : ℝ) (h : y = x^2) : y + 1 = x^2 + 1 :=
by
  sorry

end translate_parabola_upwards_l283_28376


namespace trucks_in_yard_l283_28349

/-- The number of trucks in the yard is 23, given the conditions. -/
theorem trucks_in_yard (T : ℕ) (H1 : ∃ n : ℕ, n > 0)
  (H2 : ∃ k : ℕ, k = 5 * T)
  (H3 : T + 5 * T = 140) : T = 23 :=
sorry

end trucks_in_yard_l283_28349


namespace can_reach_4_white_l283_28325

/-
We define the possible states and operations on the urn as described.
-/

structure Urn :=
  (white : ℕ)
  (black : ℕ)

def operation1 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation2 (u : Urn) : Urn :=
  { white := u.white, black := u.black - 2 }

def operation3 (u : Urn) : Urn :=
  { white := u.white - 1, black := u.black - 1 }

def operation4 (u : Urn) : Urn :=
  { white := u.white - 2, black := u.black + 1 }

theorem can_reach_4_white : ∃ (u : Urn), u.white = 4 ∧ u.black > 0 :=
  sorry

end can_reach_4_white_l283_28325


namespace neg_p_sufficient_not_necessary_q_l283_28371

-- Definitions from the given conditions
def p (a : ℝ) : Prop := a ≥ 1
def q (a : ℝ) : Prop := a ≤ 2

-- The theorem stating the mathematical equivalence
theorem neg_p_sufficient_not_necessary_q (a : ℝ) : (¬ p a → q a) ∧ ¬ (q a → ¬ p a) := 
by sorry

end neg_p_sufficient_not_necessary_q_l283_28371


namespace sequence_formula_l283_28370

theorem sequence_formula (x : ℕ → ℤ) :
  x 1 = 1 →
  x 2 = -1 →
  (∀ n, n ≥ 2 → x (n-1) + x (n+1) = 2 * x n) →
  ∀ n, x n = -2 * n + 3 :=
by
  sorry

end sequence_formula_l283_28370


namespace find_values_l283_28335

theorem find_values (x y z : ℝ) :
  (x + y + z = 1) →
  (x^2 * y + y^2 * z + z^2 * x = x * y^2 + y * z^2 + z * x^2) →
  (x^3 + y^2 + z = y^3 + z^2 + x) →
  ( (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
    (x = 0 ∧ y = 0 ∧ z = 1) ∨
    (x = 2/3 ∧ y = -1/3 ∧ z = 2/3) ∨
    (x = 0 ∧ y = 1 ∧ z = 0) ∨
    (x = 1 ∧ y = 0 ∧ z = 0) ∨
    (x = -1 ∧ y = 1 ∧ z = 1) ) := 
sorry

end find_values_l283_28335


namespace unique_root_iff_l283_28328

def has_unique_solution (a : ℝ) : Prop :=
  ∃ (x : ℝ), ∀ (y : ℝ), (a * y^2 + 2 * y - 1 = 0 ↔ y = x)

theorem unique_root_iff (a : ℝ) : has_unique_solution a ↔ (a = 0 ∨ a = 1) := 
sorry

end unique_root_iff_l283_28328


namespace solve_trig_eqn_solution_set_l283_28326

theorem solve_trig_eqn_solution_set :
  {x : ℝ | ∃ k : ℤ, x = 3 * k * Real.pi + Real.pi / 4 ∨ x = 3 * k * Real.pi + 5 * Real.pi / 4} =
  {x : ℝ | 2 * Real.sin ((2 / 3) * x) = 1} :=
by
  sorry

end solve_trig_eqn_solution_set_l283_28326


namespace find_x_y_l283_28396

theorem find_x_y 
  (x y : ℝ) 
  (h1 : (15 + 30 + x + y) / 4 = 25) 
  (h2 : x = y + 10) :
  x = 32.5 ∧ y = 22.5 := 
by 
  sorry

end find_x_y_l283_28396


namespace pinedale_bus_speed_l283_28392

theorem pinedale_bus_speed 
  (stops_every_minutes : ℕ)
  (num_stops : ℕ)
  (distance_km : ℕ)
  (time_per_stop_minutes : stops_every_minutes = 5)
  (dest_stops : num_stops = 8)
  (dest_distance : distance_km = 40) 
  : (distance_km / (num_stops * stops_every_minutes / 60)) = 60 := 
by
  sorry

end pinedale_bus_speed_l283_28392


namespace Eleanor_books_l283_28327

theorem Eleanor_books (h p : ℕ) : 
    h + p = 12 ∧ 28 * h + 18 * p = 276 → h = 6 :=
by
  intro hp
  sorry

end Eleanor_books_l283_28327


namespace edric_hourly_rate_l283_28305

theorem edric_hourly_rate
  (monthly_salary : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (H1 : monthly_salary = 576)
  (H2 : hours_per_day = 8)
  (H3 : days_per_week = 6)
  (H4 : weeks_per_month = 4) :
  monthly_salary / weeks_per_month / days_per_week / hours_per_day = 3 := by
  sorry

end edric_hourly_rate_l283_28305


namespace total_expenditure_of_7_people_l283_28367

theorem total_expenditure_of_7_people :
  ∃ A : ℝ, 
    (6 * 11 + (A + 6) = 7 * A) ∧
    (6 * 11 = 66) ∧
    (∃ total : ℝ, total = 6 * 11 + (A + 6) ∧ total = 84) :=
by 
  sorry

end total_expenditure_of_7_people_l283_28367


namespace hexagon_largest_angle_measure_l283_28314

theorem hexagon_largest_angle_measure (x : ℝ) (a b c d e f : ℝ)
  (h_ratio: a = 2 * x) (h_ratio2: b = 3 * x)
  (h_ratio3: c = 3 * x) (h_ratio4: d = 4 * x)
  (h_ratio5: e = 4 * x) (h_ratio6: f = 6 * x)
  (h_sum: a + b + c + d + e + f = 720) :
  f = 2160 / 11 :=
by
  -- Proof is not required
  sorry

end hexagon_largest_angle_measure_l283_28314


namespace smallest_three_digit_multiple_of_13_l283_28378

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, (100 ≤ n) ∧ (n < 1000) ∧ (n % 13 = 0) ∧ (∀ k : ℕ, (100 ≤ k) ∧ (k < 1000) ∧ (k % 13 = 0) → n ≤ k) → n = 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l283_28378


namespace find_m_of_pure_imaginary_l283_28310

theorem find_m_of_pure_imaginary (m : ℝ) (h1 : (m^2 + m - 2) = 0) (h2 : (m^2 - 1) ≠ 0) : m = -2 :=
by
  sorry

end find_m_of_pure_imaginary_l283_28310


namespace Z_is_all_positive_integers_l283_28300

theorem Z_is_all_positive_integers (Z : Set ℕ) (h_nonempty : Z.Nonempty)
(h1 : ∀ x ∈ Z, 4 * x ∈ Z)
(h2 : ∀ x ∈ Z, (Nat.sqrt x) ∈ Z) : 
Z = { n : ℕ | n > 0 } :=
sorry

end Z_is_all_positive_integers_l283_28300


namespace sum_of_squares_and_product_l283_28391

theorem sum_of_squares_and_product (x y : ℤ) 
  (h1 : x^2 + y^2 = 290) 
  (h2 : x * y = 96) :
  x + y = 22 :=
sorry

end sum_of_squares_and_product_l283_28391


namespace reciprocal_of_one_twentieth_l283_28324

theorem reciprocal_of_one_twentieth : (1 / (1 / 20 : ℝ)) = 20 := 
by
  sorry

end reciprocal_of_one_twentieth_l283_28324


namespace sam_total_cans_l283_28390

theorem sam_total_cans (bags_sat : ℕ) (bags_sun : ℕ) (cans_per_bag : ℕ) 
  (h_sat : bags_sat = 3) (h_sun : bags_sun = 4) (h_cans : cans_per_bag = 9) : 
  (bags_sat + bags_sun) * cans_per_bag = 63 := 
by
  sorry

end sam_total_cans_l283_28390


namespace largest_value_of_c_l283_28313

theorem largest_value_of_c : ∃ c, (∀ x : ℝ, x^2 - 6 * x + c = 1 → c ≤ 10) :=
sorry

end largest_value_of_c_l283_28313


namespace appropriate_sampling_method_l283_28332

theorem appropriate_sampling_method (total_staff teachers admin_staff logistics_personnel sample_size : ℕ)
  (h1 : total_staff = 160)
  (h2 : teachers = 120)
  (h3 : admin_staff = 16)
  (h4 : logistics_personnel = 24)
  (h5 : sample_size = 20) :
  (sample_method : String) -> sample_method = "Stratified sampling" :=
sorry

end appropriate_sampling_method_l283_28332


namespace volume_calculation_l283_28345

noncomputable def enclosedVolume : Real :=
  let f (x y z : Real) : Real := x^2016 + y^2016 + z^2
  let V : Real := 360
  V

theorem volume_calculation : enclosedVolume = 360 :=
by
  sorry

end volume_calculation_l283_28345


namespace arithmetic_mean_reciprocals_first_four_primes_l283_28387

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l283_28387


namespace fraction_operation_correct_l283_28303

theorem fraction_operation_correct {a b : ℝ} :
  (0.2 * a + 0.5 * b) ≠ 0 →
  (2 * a + 5 * b) ≠ 0 →
  (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) :=
by
  intros h1 h2
  sorry

end fraction_operation_correct_l283_28303


namespace isosceles_triangle_legs_length_l283_28315

-- Define the given conditions in Lean
def perimeter (L B: ℕ) : ℕ := 2 * L + B
def base_length : ℕ := 8
def given_perimeter : ℕ := 20

-- State the theorem to be proven
theorem isosceles_triangle_legs_length :
  ∃ (L : ℕ), perimeter L base_length = given_perimeter ∧ L = 6 :=
by
  sorry

end isosceles_triangle_legs_length_l283_28315


namespace expected_balls_original_positions_l283_28380

noncomputable def expected_original_positions : ℝ :=
  8 * ((3/4:ℝ)^3)

theorem expected_balls_original_positions :
  expected_original_positions = 3.375 := by
  sorry

end expected_balls_original_positions_l283_28380


namespace g_f_neg4_eq_12_l283_28364

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 - 8

-- Define the assumption that g(f(4)) = 12
axiom g : ℝ → ℝ
axiom g_f4 : g (f 4) = 12

-- The theorem to prove that g(f(-4)) = 12
theorem g_f_neg4_eq_12 : g (f (-4)) = 12 :=
sorry -- proof placeholder

end g_f_neg4_eq_12_l283_28364


namespace area_inequality_l283_28361

open Real

variables (AB CD AD BC S : ℝ) (alpha beta : ℝ)
variables (α_pos : 0 < α ∧ α < π) (β_pos : 0 < β ∧ β < π)
variables (S_pos : 0 < S) (H1 : ConvexQuadrilateral AB CD AD BC S)

theorem area_inequality :
  AB * CD * sin α + AD * BC * sin β ≤ 2 * S ∧ 2 * S ≤ AB * CD + AD * BC :=
sorry

end area_inequality_l283_28361


namespace intersection_sums_l283_28374

theorem intersection_sums (x1 x2 x3 y1 y2 y3 : ℝ) (h1 : y1 = x1^3 - 6 * x1 + 4)
  (h2 : y2 = x2^3 - 6 * x2 + 4) (h3 : y3 = x3^3 - 6 * x3 + 4)
  (h4 : x1 + 3 * y1 = 3) (h5 : x2 + 3 * y2 = 3) (h6 : x3 + 3 * y3 = 3) :
  x1 + x2 + x3 = 0 ∧ y1 + y2 + y3 = 3 := 
by
  sorry

end intersection_sums_l283_28374


namespace arrange_cubes_bound_l283_28386

def num_ways_to_arrange_cubes_into_solids (n : ℕ) : ℕ := sorry

theorem arrange_cubes_bound (n : ℕ) (h : n = (2015^100)) :
  10^14 < num_ways_to_arrange_cubes_into_solids n ∧
  num_ways_to_arrange_cubes_into_solids n < 10^15 := sorry

end arrange_cubes_bound_l283_28386


namespace number_of_students_l283_28388

-- Define the conditions
variable (n : ℕ) (jayden_rank_best jayden_rank_worst : ℕ)
variable (h1 : jayden_rank_best = 100)
variable (h2 : jayden_rank_worst = 100)

-- Define the question
theorem number_of_students (h1 : jayden_rank_best = 100) (h2 : jayden_rank_worst = 100) : n = 199 := 
  sorry

end number_of_students_l283_28388


namespace digit_a_for_divisibility_l283_28323

theorem digit_a_for_divisibility (a : ℕ) (h1 : (8 * 10^3 + 7 * 10^2 + 5 * 10 + a) % 6 = 0) : a = 4 :=
sorry

end digit_a_for_divisibility_l283_28323


namespace percentage_of_women_attended_picnic_l283_28385

variable (E : ℝ) -- Total number of employees
variable (M : ℝ) -- The number of men
variable (W : ℝ) -- The number of women
variable (P : ℝ) -- Percentage of women who attended the picnic

-- Conditions
variable (h1 : M = 0.30 * E)
variable (h2 : W = E - M)
variable (h3 : 0.20 * M = 0.20 * 0.30 * E)
variable (h4 : 0.34 * E = 0.20 * 0.30 * E + P * (E - 0.30 * E))

-- Goal
theorem percentage_of_women_attended_picnic : P = 0.40 :=
by
  sorry

end percentage_of_women_attended_picnic_l283_28385


namespace triangle_area_correct_l283_28366
noncomputable def area_of_triangle_intercepts : ℝ :=
  let f (x : ℝ) : ℝ := (x - 3) ^ 2 * (x + 2)
  let x1 := 3
  let x2 := -2
  let y_intercept := f 0
  let base := x1 - x2
  let height := y_intercept
  1 / 2 * base * height

theorem triangle_area_correct :
  area_of_triangle_intercepts = 45 :=
by
  sorry

end triangle_area_correct_l283_28366


namespace find_n_l283_28306

theorem find_n (x y : ℝ) (n : ℝ) (h1 : x / (2 * y) = 3 / n) (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : n = 2 := by
  sorry

end find_n_l283_28306


namespace solve_for_x_l283_28309

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 5) * x = 2) : x = 17.5 :=
by
  -- Here we acknowledge the initial condition and conclusion without proving
  sorry

end solve_for_x_l283_28309


namespace find_y_l283_28394

-- Definitions of the given conditions
def is_straight_line (A B : Point) : Prop := 
  ∃ C D, A ≠ C ∧ B ≠ D

def angle (A B C : Point) : ℝ := sorry -- Assume angle is a function providing the angle in degrees

-- The proof problem statement
theorem find_y
  (A B C D X Y Z : Point)
  (hAB : is_straight_line A B)
  (hCD : is_straight_line C D)
  (hAXB : angle A X B = 180) 
  (hYXZ : angle Y X Z = 70)
  (hCYX : angle C Y X = 110) :
  angle X Y Z = 40 :=
sorry

end find_y_l283_28394


namespace unit_prices_min_total_cost_l283_28363

-- Part (1): Proving the unit prices of ingredients A and B.
theorem unit_prices (x y : ℝ)
    (h₁ : x + y = 68)
    (h₂ : 5 * x + 3 * y = 280) :
    x = 38 ∧ y = 30 :=
by
  -- Sorry, proof not provided
  sorry

-- Part (2): Proving the minimum cost calculation.
theorem min_total_cost (m : ℝ)
    (h₁ : m + (36 - m) = 36)
    (h₂ : m ≥ 2 * (36 - m)) :
    (38 * m + 30 * (36 - m)) = 1272 :=
by
  -- Sorry, proof not provided
  sorry

end unit_prices_min_total_cost_l283_28363


namespace monotonic_has_at_most_one_solution_l283_28352

def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y ∨ f y ≤ f x

theorem monotonic_has_at_most_one_solution (f : ℝ → ℝ) (c : ℝ) 
  (hf : monotonic f) : ∃! x : ℝ, f x = c :=
sorry

end monotonic_has_at_most_one_solution_l283_28352


namespace impossible_rearrange_reverse_l283_28333

theorem impossible_rearrange_reverse :
  ∀ (tokens : ℕ → ℕ), 
    (∀ i, (i % 2 = 1 ∧ i < 99 → tokens i = tokens (i + 2)) 
      ∧ (i % 2 = 0 ∧ i < 99 → tokens i = tokens (i + 2))) → ¬(∀ i, tokens i = 100 + 1 - tokens (i - 1)) :=
by
  intros tokens h
  sorry

end impossible_rearrange_reverse_l283_28333


namespace complex_is_1_sub_sqrt3i_l283_28381

open Complex

theorem complex_is_1_sub_sqrt3i (z : ℂ) (h : z * (1 + Real.sqrt 3 * I) = abs (1 + Real.sqrt 3 * I)) : z = 1 - Real.sqrt 3 * I :=
sorry

end complex_is_1_sub_sqrt3i_l283_28381


namespace probability_two_red_faces_eq_three_eighths_l283_28358

def cube_probability : ℚ :=
  let total_cubes := 64 -- Total number of smaller cubes
  let two_red_faces_cubes := 24 -- Number of smaller cubes with exactly two red faces
  two_red_faces_cubes / total_cubes

theorem probability_two_red_faces_eq_three_eighths :
  cube_probability = 3 / 8 :=
by
  -- proof goes here
  sorry

end probability_two_red_faces_eq_three_eighths_l283_28358


namespace period_is_3_years_l283_28330

def gain_of_B_per_annum (principal : ℕ) (rate_A rate_B : ℚ) : ℚ := 
  (rate_B - rate_A) * principal

def period (principal : ℕ) (rate_A rate_B : ℚ) (total_gain : ℚ) : ℚ := 
  total_gain / gain_of_B_per_annum principal rate_A rate_B

theorem period_is_3_years :
  period 1500 (10 / 100) (11.5 / 100) 67.5 = 3 :=
by
  sorry

end period_is_3_years_l283_28330


namespace possible_values_of_AD_l283_28307

-- Define the conditions as variables
variables {A B C D : ℝ}
variables {AB BC CD : ℝ}

-- Assume the given conditions
def conditions (A B C D : ℝ) (AB BC CD : ℝ) : Prop :=
  AB = 1 ∧ BC = 2 ∧ CD = 4

-- Define the proof goal: proving the possible values of AD
theorem possible_values_of_AD (h : conditions A B C D AB BC CD) :
  ∃ AD, AD = 1 ∨ AD = 3 ∨ AD = 5 ∨ AD = 7 :=
sorry

end possible_values_of_AD_l283_28307


namespace total_matches_played_l283_28344

theorem total_matches_played (home_wins : ℕ) (rival_wins : ℕ) (draws : ℕ) (home_wins_eq : home_wins = 3) (rival_wins_eq : rival_wins = 2 * home_wins) (draws_eq : draws = 4) (no_losses : ∀ (t : ℕ), t = 0) :
  home_wins + rival_wins + 2 * draws = 17 :=
by {
  sorry
}

end total_matches_played_l283_28344


namespace weird_fraction_implies_weird_power_fraction_l283_28308

theorem weird_fraction_implies_weird_power_fraction 
  (a b c : ℝ) (k : ℕ) 
  (h1 : (1/a) + (1/b) + (1/c) = (1/(a + b + c))) 
  (h2 : Odd k) : 
  (1 / (a^k) + 1 / (b^k) + 1 / (c^k) = 1 / (a^k + b^k + c^k)) := 
by 
  sorry

end weird_fraction_implies_weird_power_fraction_l283_28308


namespace mari_vs_kendra_l283_28375

-- Variable Definitions
variables (K M S : ℕ)  -- Number of buttons Kendra, Mari, and Sue made
variables (h1: 2*S = K) -- Sue made half as many as Kendra
variables (h2: S = 6)   -- Sue made 6 buttons
variables (h3: M = 64)  -- Mari made 64 buttons

-- Theorem Statement
theorem mari_vs_kendra (K M S : ℕ) (h1 : 2 * S = K) (h2 : S = 6) (h3 : M = 64) :
  M = 5 * K + 4 :=
sorry

end mari_vs_kendra_l283_28375


namespace tiling_remainder_is_888_l283_28369

noncomputable def boardTilingWithThreeColors (n : ℕ) : ℕ :=
  if n = 8 then
    4 * (21 * (3^3 - 3*2^3 + 3) +
         35 * (3^4 - 4*2^4 + 6) +
         35 * (3^5 - 5*2^5 + 10) +
         21 * (3^6 - 6*2^6 + 15) +
         7 * (3^7 - 7*2^7 + 21) +
         1 * (3^8 - 8*2^8 + 28))
  else
    0

theorem tiling_remainder_is_888 :
  boardTilingWithThreeColors 8 % 1000 = 888 :=
by
  sorry

end tiling_remainder_is_888_l283_28369


namespace geometric_sequence_value_l283_28354

theorem geometric_sequence_value (a : ℕ → ℝ) (h : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n+2) = a (n+1) * (a (n+1) / a n)) :
  a 3 * a 5 = 4 → a 4 = 2 :=
by
  sorry

end geometric_sequence_value_l283_28354


namespace bananas_each_child_l283_28353

theorem bananas_each_child (x : ℕ) (B : ℕ) 
  (h1 : 660 * x = B)
  (h2 : 330 * (x + 2) = B) : 
  x = 2 := 
by 
  sorry

end bananas_each_child_l283_28353


namespace base_7_divisibility_l283_28342

theorem base_7_divisibility (y : ℕ) :
  (934 + 7 * y) % 19 = 0 ↔ y = 3 :=
by
  sorry

end base_7_divisibility_l283_28342


namespace polygon_sides_l283_28304

theorem polygon_sides (n : ℕ) (h₁ : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 156 = (180 * (n - 2)) / n) : n = 15 := sorry

end polygon_sides_l283_28304


namespace marble_problem_l283_28383

theorem marble_problem (a : ℚ) :
  (a + 2 * a + 3 * 2 * a + 5 * (3 * 2 * a) + 2 * (5 * (3 * 2 * a)) = 212) ↔
  (a = 212 / 99) :=
by
  sorry

end marble_problem_l283_28383


namespace find_b_for_inf_solutions_l283_28329

theorem find_b_for_inf_solutions (x : ℝ) (b : ℝ) : 5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 :=
by
  intro h
  sorry

end find_b_for_inf_solutions_l283_28329


namespace cyclist_wait_time_l283_28357

theorem cyclist_wait_time
  (hiker_speed : ℝ)
  (hiker_speed_pos : hiker_speed = 4)
  (cyclist_speed : ℝ)
  (cyclist_speed_pos : cyclist_speed = 24)
  (waiting_time_minutes : ℝ)
  (waiting_time_minutes_pos : waiting_time_minutes = 5) :
  (waiting_time_minutes / 60) * cyclist_speed = 2 →
  (2 / hiker_speed) * 60 = 30 :=
by
  intros
  sorry

end cyclist_wait_time_l283_28357


namespace distinct_naturals_and_power_of_prime_l283_28398

theorem distinct_naturals_and_power_of_prime (a b : ℕ) (p k : ℕ) (h1 : a ≠ b) (h2 : a^2 + b ∣ b^2 + a) (h3 : ∃ (p : ℕ) (k : ℕ), b^2 + a = p^k) : (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2) :=
sorry

end distinct_naturals_and_power_of_prime_l283_28398


namespace scorpion_needs_10_millipedes_l283_28348

-- Define the number of segments required daily
def total_segments_needed : ℕ := 800

-- Define the segments already consumed by the scorpion
def segments_consumed : ℕ := 60 + 2 * (2 * 60)

-- Calculate the remaining segments needed
def remaining_segments_needed : ℕ := total_segments_needed - segments_consumed

-- Define the segments per millipede
def segments_per_millipede : ℕ := 50

-- Prove that the number of 50-segment millipedes to be eaten is 10
theorem scorpion_needs_10_millipedes 
  (h : remaining_segments_needed = 500) 
  (h2 : 500 / segments_per_millipede = 10) : 
  500 / segments_per_millipede = 10 := by
  sorry

end scorpion_needs_10_millipedes_l283_28348


namespace possible_values_of_N_l283_28318

theorem possible_values_of_N (N : ℤ) (h : N^2 - N = 12) : N = 4 ∨ N = -3 :=
sorry

end possible_values_of_N_l283_28318


namespace divisor_in_second_division_l283_28395

theorem divisor_in_second_division 
  (n : ℤ) 
  (h1 : (68 : ℤ) * 269 = n) 
  (d q : ℤ) 
  (h2 : n = d * q + 1) 
  (h3 : Prime 18291):
  d = 18291 := by
  sorry

end divisor_in_second_division_l283_28395
