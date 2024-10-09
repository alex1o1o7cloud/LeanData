import Mathlib

namespace ones_digit_of_9_pow_46_l553_55391

theorem ones_digit_of_9_pow_46 : (9 ^ 46) % 10 = 1 :=
by
  sorry

end ones_digit_of_9_pow_46_l553_55391


namespace rotational_homothety_commute_iff_centers_coincide_l553_55387

-- Define rotational homothety and its properties
structure RotationalHomothety (P : Type*) :=
(center : P)
(apply : P → P)
(is_homothety : ∀ p, apply (apply p) = apply p)

variables {P : Type*} [TopologicalSpace P] (H1 H2 : RotationalHomothety P)

-- Prove the equivalence statement
theorem rotational_homothety_commute_iff_centers_coincide :
  (H1.center = H2.center) ↔ (H1.apply ∘ H2.apply = H2.apply ∘ H1.apply) :=
sorry

end rotational_homothety_commute_iff_centers_coincide_l553_55387


namespace problem_statement_l553_55399

theorem problem_statement (x : ℝ) (h : 5 * x - 8 = 15 * x + 14) : 6 * (x + 3) = 4.8 :=
sorry

end problem_statement_l553_55399


namespace problem1_problem2_l553_55336

-- Problem 1
theorem problem1 : (-2)^2 * (1 / 4) + 4 / (4 / 9) + (-1)^2023 = 7 :=
by
  sorry

-- Problem 2
theorem problem2 : -1^4 + abs (2 - (-3)^2) + (1 / 2) / (-3 / 2) = 5 + 2 / 3 :=
by
  sorry

end problem1_problem2_l553_55336


namespace Jessie_initial_weight_l553_55382

def lost_first_week : ℕ := 56
def after_first_week : ℕ := 36

theorem Jessie_initial_weight :
  (after_first_week + lost_first_week = 92) :=
by
  sorry

end Jessie_initial_weight_l553_55382


namespace hyperbola_real_axis_length_l553_55394

theorem hyperbola_real_axis_length : 
  (∃ (x y : ℝ), (x^2 / 2) - (y^2 / 4) = 1) → real_axis_length = 2 * Real.sqrt 2 :=
by
  -- Proof is omitted
  sorry

end hyperbola_real_axis_length_l553_55394


namespace people_lost_l553_55373

-- Define the given constants
def win_ratio : ℕ := 4
def lose_ratio : ℕ := 1
def people_won : ℕ := 28

-- The statement to prove that 7 people lost
theorem people_lost (win_ratio lose_ratio people_won : ℕ) (H : win_ratio * 7 = people_won * lose_ratio) : 7 = people_won * lose_ratio / win_ratio :=
by { sorry }

end people_lost_l553_55373


namespace monthly_income_of_P_l553_55351

variable (P Q R : ℝ)

theorem monthly_income_of_P (h1 : (P + Q) / 2 = 5050) 
                           (h2 : (Q + R) / 2 = 6250) 
                           (h3 : (P + R) / 2 = 5200) : 
    P = 4000 := 
sorry

end monthly_income_of_P_l553_55351


namespace hyperbola_equation_l553_55362

theorem hyperbola_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 - 2 → (∃ k : ℝ, k ≠ 0 ∧ x * y = k) := 
by
  intros h
  sorry

end hyperbola_equation_l553_55362


namespace museum_rid_paintings_l553_55367

def initial_paintings : ℕ := 1795
def leftover_paintings : ℕ := 1322

theorem museum_rid_paintings : initial_paintings - leftover_paintings = 473 := by
  sorry

end museum_rid_paintings_l553_55367


namespace pete_numbers_count_l553_55357

theorem pete_numbers_count :
  ∃ x_values : Finset Nat, x_values.card = 4 ∧
  ∀ x ∈ x_values, ∃ y z : Nat, 
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x + y) * z = 14 ∧ (x * y) + z = 14 :=
by
  sorry

end pete_numbers_count_l553_55357


namespace triangle_side_length_l553_55348

theorem triangle_side_length (a b c x : ℕ) (A C : ℝ) (h1 : b = x) (h2 : a = x - 2) (h3 : c = x + 2)
  (h4 : C = 2 * A) (h5 : x + 2 = 10) : a = 8 :=
by
  sorry

end triangle_side_length_l553_55348


namespace quotient_transformation_l553_55316

theorem quotient_transformation (A B : ℕ) (h1 : B ≠ 0) (h2 : (A : ℝ) / B = 0.514) :
  ((10 * A : ℝ) / (B / 100)) = 514 :=
by
  -- skipping the proof
  sorry

end quotient_transformation_l553_55316


namespace complement_of_60_is_30_l553_55364

noncomputable def complement (angle : ℝ) : ℝ := 90 - angle

theorem complement_of_60_is_30 : complement 60 = 30 :=
by 
  sorry

end complement_of_60_is_30_l553_55364


namespace ratio_of_average_speeds_l553_55392

-- Conditions
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4
def distance_ab : ℕ := 600
def distance_ac : ℕ := 360

-- Theorem to prove the ratio of their average speeds
theorem ratio_of_average_speeds : (distance_ab / time_eddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 20 ∧
                                  (distance_ac / time_freddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 9 :=
by
  -- Solution steps go here if performing an actual proof
  sorry

end ratio_of_average_speeds_l553_55392


namespace suitable_for_comprehensive_survey_l553_55332

-- Define the conditions
def is_comprehensive_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  is_specific_group ∧ (group_size < 100)  -- assuming "small" means fewer than 100 individuals/items

def is_sampling_survey (group_size : ℕ) (is_specific_group : Bool) : Bool :=
  ¬is_comprehensive_survey group_size is_specific_group

-- Define the surveys
def option_A (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_comprehensive_survey group_size is_specific_group

def option_B (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_C (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

def option_D (group_size : ℕ) (is_specific_group : Bool) : Prop :=
  is_sampling_survey group_size is_specific_group

-- Question: Which of the following surveys is suitable for a comprehensive survey given conditions
theorem suitable_for_comprehensive_survey :
  ∀ (group_size_A group_size_B group_size_C group_size_D : ℕ) 
    (is_specific_group_A is_specific_group_B is_specific_group_C is_specific_group_D : Bool),
  option_A group_size_A is_specific_group_A ↔ 
  ((option_B group_size_B is_specific_group_B = false) ∧ 
   (option_C group_size_C is_specific_group_C = false) ∧ 
   (option_D group_size_D is_specific_group_D = false)) :=
by
  sorry

end suitable_for_comprehensive_survey_l553_55332


namespace five_times_number_equals_hundred_l553_55398

theorem five_times_number_equals_hundred (x : ℝ) (h : 5 * x = 100) : x = 20 :=
sorry

end five_times_number_equals_hundred_l553_55398


namespace import_tax_excess_amount_l553_55346

theorem import_tax_excess_amount 
    (tax_rate : ℝ) 
    (tax_paid : ℝ) 
    (total_value : ℝ)
    (X : ℝ) 
    (h1 : tax_rate = 0.07)
    (h2 : tax_paid = 109.2)
    (h3 : total_value = 2560) 
    (eq1 : tax_rate * (total_value - X) = tax_paid) :
    X = 1000 := sorry

end import_tax_excess_amount_l553_55346


namespace toothpicks_per_card_l553_55340

-- Define the conditions of the problem
def numCardsInDeck : ℕ := 52
def numCardsNotUsed : ℕ := 16
def numCardsUsed : ℕ := numCardsInDeck - numCardsNotUsed

def numBoxesToothpicks : ℕ := 6
def toothpicksPerBox : ℕ := 450
def totalToothpicksUsed : ℕ := numBoxesToothpicks * toothpicksPerBox

-- Prove the number of toothpicks used per card
theorem toothpicks_per_card : totalToothpicksUsed / numCardsUsed = 75 := 
  by sorry

end toothpicks_per_card_l553_55340


namespace original_cost_of_car_l553_55331

-- Conditions
variables (C : ℝ)
variables (spent_on_repairs : ℝ := 8000)
variables (selling_price : ℝ := 68400)
variables (profit_percent : ℝ := 54.054054054054056)

-- Statement to be proved
theorem original_cost_of_car :
  C + spent_on_repairs = selling_price - (profit_percent / 100) * C :=
sorry

end original_cost_of_car_l553_55331


namespace find_smaller_number_l553_55358

theorem find_smaller_number (L S : ℕ) (h1 : L - S = 2468) (h2 : L = 8 * S + 27) : S = 349 :=
by
  sorry

end find_smaller_number_l553_55358


namespace find_number_l553_55304

/-- 
  Given that 23% of a number x is equal to 150, prove that x equals 15000 / 23.
-/
theorem find_number (x : ℝ) (h : (23 / 100) * x = 150) : x = 15000 / 23 :=
by
  sorry

end find_number_l553_55304


namespace similarity_of_triangle_l553_55356

noncomputable def side_length (AB BC AC : ℝ) : Prop :=
  ∀ k : ℝ, k ≠ 1 → (AB, BC, AC) = (k * AB, k * BC, k * AC)

theorem similarity_of_triangle (AB BC AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : AC > 0) :
  side_length (2 * AB) (2 * BC) (2 * AC) = side_length AB BC AC :=
by sorry

end similarity_of_triangle_l553_55356


namespace number_subtracted_l553_55334

theorem number_subtracted (x y : ℕ) (h₁ : x = 48) (h₂ : 5 * x - y = 102) : y = 138 :=
by
  rw [h₁] at h₂
  sorry

end number_subtracted_l553_55334


namespace double_acute_angle_lt_180_l553_55352

theorem double_acute_angle_lt_180
  (α : ℝ) (h : 0 < α ∧ α < 90) : 2 * α < 180 := 
sorry

end double_acute_angle_lt_180_l553_55352


namespace jamie_workday_percent_l553_55349

theorem jamie_workday_percent
  (total_work_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (second_meeting_multiplier : ℕ)
  (break_minutes : ℕ)
  (total_minutes_per_hour : ℕ)
  (total_work_minutes : ℕ)
  (first_meeting_duration : ℕ)
  (second_meeting_duration : ℕ)
  (total_meeting_time : ℕ)
  (percentage_spent : ℚ) :
  total_work_hours = 10 →
  first_meeting_minutes = 60 →
  second_meeting_multiplier = 2 →
  break_minutes = 30 →
  total_minutes_per_hour = 60 →
  total_work_minutes = total_work_hours * total_minutes_per_hour →
  first_meeting_duration = first_meeting_minutes →
  second_meeting_duration = second_meeting_multiplier * first_meeting_duration →
  total_meeting_time = first_meeting_duration + second_meeting_duration + break_minutes →
  percentage_spent = (total_meeting_time : ℚ) / (total_work_minutes : ℚ) * 100 →
  percentage_spent = 35 :=
sorry

end jamie_workday_percent_l553_55349


namespace train_speed_is_72_kmh_l553_55395

-- Length of the train in meters
def length_train : ℕ := 600

-- Length of the platform in meters
def length_platform : ℕ := 600

-- Time to cross the platform in minutes
def time_crossing_platform : ℕ := 1

-- Convert meters to kilometers
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000

-- Convert minutes to hours
def minutes_to_hours (m : ℕ) : ℕ := m * 60

-- Speed of the train in km/hr given lengths in meters and time in minutes
def speed_train_kmh (distance_m : ℕ) (time_min : ℕ) : ℕ :=
  (meters_to_kilometers distance_m) / (minutes_to_hours time_min)

theorem train_speed_is_72_kmh :
  speed_train_kmh (length_train + length_platform) time_crossing_platform = 72 :=
by
  -- skipping the proof
  sorry

end train_speed_is_72_kmh_l553_55395


namespace blue_paint_amount_l553_55345

/-- 
Prove that if Giselle uses 15 quarts of white paint, then according to the ratio 4:3:5, she should use 12 quarts of blue paint.
-/
theorem blue_paint_amount (white_paint : ℚ) (h1 : white_paint = 15) : 
  let blue_ratio := 4;
  let white_ratio := 5;
  blue_ratio / white_ratio * white_paint = 12 :=
by
  sorry

end blue_paint_amount_l553_55345


namespace min_value_of_y_l553_55326

theorem min_value_of_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (∃ y : ℝ, y = 1 / a + 4 / b ∧ (∀ (a' b' : ℝ), a' > 0 → b' > 0 → a' + b' = 1 → 1 / a' + 4 / b' ≥ y)) ∧ 
  (∀ y : ℝ, y = 1 / a + 4 / b → y ≥ 9) :=
sorry

end min_value_of_y_l553_55326


namespace at_least_three_equal_l553_55309

theorem at_least_three_equal (a b c d : ℕ) (h1 : (a + b) ^ 2 ∣ c * d)
                                (h2 : (a + c) ^ 2 ∣ b * d)
                                (h3 : (a + d) ^ 2 ∣ b * c)
                                (h4 : (b + c) ^ 2 ∣ a * d)
                                (h5 : (b + d) ^ 2 ∣ a * c)
                                (h6 : (c + d) ^ 2 ∣ a * b) :
  ∃ x : ℕ, (x = a ∧ x = b ∧ x = c) ∨ (x = a ∧ x = b ∧ x = d) ∨ (x = a ∧ x = c ∧ x = d) ∨ (x = b ∧ x = c ∧ x = d) :=
sorry

end at_least_three_equal_l553_55309


namespace union_of_M_and_N_l553_55302

def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4, 5} :=
by
  sorry

end union_of_M_and_N_l553_55302


namespace trajectory_is_one_branch_of_hyperbola_l553_55368

open Real

-- Condition 1: Given points F1 and F2
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Condition 2: Moving point P such that |PF1| - |PF2| = 4
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  abs (dist P F1) - abs (dist P F2) = 4

-- Prove the trajectory of point P is one branch of a hyperbola
theorem trajectory_is_one_branch_of_hyperbola (P : ℝ × ℝ) (h : satisfies_condition P) : 
  (∃ a b : ℝ, ∀ x y: ℝ, satisfies_condition (x, y) → (((x^2 / a^2) - (y^2 / b^2) = 1) ∨ ((x^2 / a^2) - (y^2 / b^2) = -1))) :=
sorry

end trajectory_is_one_branch_of_hyperbola_l553_55368


namespace GAUSS_1998_LCM_l553_55335

/-- The periodicity of cycling the word 'GAUSS' -/
def period_GAUSS : ℕ := 5

/-- The periodicity of cycling the number '1998' -/
def period_1998 : ℕ := 4

/-- The least common multiple (LCM) of the periodicities of 'GAUSS' and '1998' is 20 -/
theorem GAUSS_1998_LCM : Nat.lcm period_GAUSS period_1998 = 20 :=
by
  sorry

end GAUSS_1998_LCM_l553_55335


namespace problem_statement_l553_55310

theorem problem_statement (m : ℤ) (h : (m + 2)^2 = 64) : (m + 1) * (m + 3) = 63 :=
sorry

end problem_statement_l553_55310


namespace line_intersects_ellipse_all_possible_slopes_l553_55330

theorem line_intersects_ellipse_all_possible_slopes (m : ℝ) :
  m^2 ≥ 1 / 5 ↔ ∃ x y : ℝ, (y = m * x - 3) ∧ (4 * x^2 + 25 * y^2 = 100) := sorry

end line_intersects_ellipse_all_possible_slopes_l553_55330


namespace find_function_l553_55343

theorem find_function (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m^2 + n^2) = f m ^ 2 + f n ^ 2)
  (h2 : f 1 > 0) : ∀ n : ℕ, f n = n := 
sorry

end find_function_l553_55343


namespace inverse_shifted_point_l553_55381

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def inverse_function (f g : ℝ → ℝ) : Prop := ∀ y, f (g y) = y ∧ ∀ x, g (f x) = x

theorem inverse_shifted_point
  (f : ℝ → ℝ)
  (hf_odd : odd_function f)
  (hf_point : f (-1) = 3)
  (g : ℝ → ℝ)
  (hg_inverse : inverse_function f g) :
  g (2 - 5) = 1 :=
by
  sorry

end inverse_shifted_point_l553_55381


namespace ratio_of_white_to_yellow_balls_l553_55353

theorem ratio_of_white_to_yellow_balls (original_white original_yellow extra_yellow : ℕ) 
(h1 : original_white = 32) 
(h2 : original_yellow = 32) 
(h3 : extra_yellow = 20) : 
(original_white : ℚ) / (original_yellow + extra_yellow) = 8 / 13 := 
by
  sorry

end ratio_of_white_to_yellow_balls_l553_55353


namespace find_parabola_equation_l553_55380

noncomputable def parabola_equation (a : ℝ) : Prop :=
  ∃ (F : ℝ × ℝ) (A : ℝ × ℝ), 
    F.1 = a / 4 ∧ F.2 = 0 ∧
    A.1 = 0 ∧ A.2 = a / 2 ∧
    (abs (F.1 * A.2) / 2) = 4

theorem find_parabola_equation :
  ∀ (a : ℝ), parabola_equation a → a = 8 ∨ a = -8 :=
by
  sorry

end find_parabola_equation_l553_55380


namespace angle_B_is_pi_div_3_sin_C_value_l553_55361

-- Definitions and conditions
variable (A B C a b c : ℝ)
variable (cos_cos_eq : (2 * a - c) * Real.cos B = b * Real.cos C)
variable (triangle_ineq : 0 < A ∧ A < Real.pi)
variable (sin_positive : Real.sin A > 0)
variable (a_eq_2 : a = 2)
variable (c_eq_3 : c = 3)

-- Proving B = π / 3 under given conditions
theorem angle_B_is_pi_div_3 : B = Real.pi / 3 := sorry

-- Proving sin C under given additional conditions
theorem sin_C_value : Real.sin C = 3 * Real.sqrt 14 / 14 := sorry

end angle_B_is_pi_div_3_sin_C_value_l553_55361


namespace cost_to_fill_half_of_can_B_l553_55360

theorem cost_to_fill_half_of_can_B (r h : ℝ) (cost_fill_V : ℝ) (cost_fill_V_eq : cost_fill_V = 16)
  (V_radius_eq : 2 * r = radius_of_can_V)
  (V_height_eq: h / 2 = height_of_can_V) :
  cost_fill_half_of_can_B = 4 :=
by
  sorry

end cost_to_fill_half_of_can_B_l553_55360


namespace inequality_solution_l553_55305

theorem inequality_solution {a b x : ℝ} 
  (h_sol_set : -1 < x ∧ x < 1) 
  (h1 : x - a > 2) 
  (h2 : b - 2 * x > 0) : 
  (a + b) ^ 2021 = -1 := 
by 
  sorry 

end inequality_solution_l553_55305


namespace head_start_distance_l553_55327

theorem head_start_distance (v_A v_B L H : ℝ) (h1 : v_A = 15 / 13 * v_B)
    (h2 : t_A = L / v_A) (h3 : t_B = (L - H) / v_B) (h4 : t_B = t_A - 0.25 * L / v_B) :
    H = 23 / 60 * L :=
sorry

end head_start_distance_l553_55327


namespace find_line_eq_l553_55341

theorem find_line_eq
  (l : ℝ → ℝ → Prop)
  (bisects_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y = 0 → l x y)
  (perpendicular_to_line : ∀ x y : ℝ, l x y ↔ y = -1/2 * x)
  : ∀ x y : ℝ, l x y ↔ 2*x - y = 0 := by
  sorry

end find_line_eq_l553_55341


namespace hyperbola_eccentricity_proof_l553_55317

noncomputable def ellipse_eccentricity (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (b ^ 2 + (a / 2) ^ 2 = a ^ 2)

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt ((a ^ 2 + b ^ 2) / a ^ 2)

theorem hyperbola_eccentricity_proof
  (a b : ℝ) (h : a > b ∧ b > 0) (h1 : ellipse_eccentricity a b h) :
  hyperbola_eccentricity a b = Real.sqrt 7 / 2 :=
by
  sorry

end hyperbola_eccentricity_proof_l553_55317


namespace sin_double_angle_l553_55388

theorem sin_double_angle (A : ℝ) (h₁ : 0 < A) (h₂ : A < π / 2) (h₃ : Real.cos A = 3 / 5) :
  Real.sin (2 * A) = 24 / 25 := 
by
  sorry

end sin_double_angle_l553_55388


namespace magician_starting_decks_l553_55313

def starting_decks (price_per_deck earned remaining_decks : ℕ) : ℕ :=
  earned / price_per_deck + remaining_decks

theorem magician_starting_decks :
  starting_decks 2 4 3 = 5 :=
by
  sorry

end magician_starting_decks_l553_55313


namespace asymptotes_of_hyperbola_l553_55308

theorem asymptotes_of_hyperbola :
  ∀ x y : ℝ, (y^2 / 4 - x^2 / 9 = 1) → (y = (2 / 3) * x ∨ y = -(2 / 3) * x) :=
by
  sorry

end asymptotes_of_hyperbola_l553_55308


namespace right_triangle_of_pythagorean_l553_55375

theorem right_triangle_of_pythagorean
  (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB BC CA : ℝ)
  (h : AB^2 = BC^2 + CA^2) : ∃ (c : ℕ), c = 90 :=
by
  sorry

end right_triangle_of_pythagorean_l553_55375


namespace B_and_C_complete_task_l553_55396

noncomputable def A_work_rate : ℚ := 1 / 12
noncomputable def B_work_rate : ℚ := 1.2 * A_work_rate
noncomputable def C_work_rate : ℚ := 2 * A_work_rate

theorem B_and_C_complete_task (B_work_rate C_work_rate : ℚ) 
    (A_work_rate : ℚ := 1 / 12) :
  B_work_rate = 1.2 * A_work_rate →
  C_work_rate = 2 * A_work_rate →
  (B_work_rate + C_work_rate) = 4 / 15 :=
by intros; sorry

end B_and_C_complete_task_l553_55396


namespace abs_inequality_solution_bounded_a_b_inequality_l553_55389

theorem abs_inequality_solution (x : ℝ) : (-4 < x ∧ x < 0) ↔ (|x + 1| + |x + 3| < 4) := sorry

theorem bounded_a_b_inequality (a b : ℝ) (h1 : -4 < a) (h2 : a < 0) (h3 : -4 < b) (h4 : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| := sorry

end abs_inequality_solution_bounded_a_b_inequality_l553_55389


namespace sufficient_but_not_necessary_l553_55369

noncomputable def p (m : ℝ) : Prop :=
  -6 ≤ m ∧ m ≤ 6

noncomputable def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 9 ≠ 0

theorem sufficient_but_not_necessary (m : ℝ) :
  (p m → q m) ∧ (q m → ¬ p m) :=
by
  sorry

end sufficient_but_not_necessary_l553_55369


namespace intersection_A_B_l553_55342

-- Conditions
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

-- Proof of the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end intersection_A_B_l553_55342


namespace fencing_required_l553_55311

variable (L W : ℝ)
variable (Area : ℝ := 20 * W)

theorem fencing_required (hL : L = 20) (hArea : L * W = 600) : 20 + 2 * W = 80 := by
  sorry

end fencing_required_l553_55311


namespace length_of_second_train_l553_55385

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (crossing_time : ℝ)
  (total_distance : ℝ)
  (relative_speed_mps : ℝ)
  (length_second_train : ℝ) :
  length_first_train = 130 ∧ 
  speed_first_train = 60 ∧
  speed_second_train = 40 ∧
  crossing_time = 10.439164866810657 ∧
  relative_speed_mps = (speed_first_train + speed_second_train) * (5/18) ∧
  total_distance = relative_speed_mps * crossing_time ∧
  length_first_train + length_second_train = total_distance →
  length_second_train = 160 :=
by
  sorry

end length_of_second_train_l553_55385


namespace butterfly_eq_roots_l553_55377

theorem butterfly_eq_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : a - b + c = 0)
    (h3 : (a + c)^2 - 4 * a * c = 0) : a = c :=
by
  sorry

end butterfly_eq_roots_l553_55377


namespace daisy_milk_problem_l553_55384

theorem daisy_milk_problem (total_milk : ℝ) (kids_percentage : ℝ) (remaining_milk : ℝ) (used_milk : ℝ) :
  total_milk = 16 →
  kids_percentage = 0.75 →
  remaining_milk = total_milk * (1 - kids_percentage) →
  used_milk = 2 →
  (used_milk / remaining_milk) * 100 = 50 :=
by
  intros _ _ _ _ 
  sorry

end daisy_milk_problem_l553_55384


namespace sin_sum_square_gt_sin_prod_l553_55319

theorem sin_sum_square_gt_sin_prod (α β γ : ℝ) (h1 : α + β + γ = Real.pi) 
  (h2 : 0 < Real.sin α) (h3 : Real.sin α < 1)
  (h4 : 0 < Real.sin β) (h5 : Real.sin β < 1)
  (h6 : 0 < Real.sin γ) (h7 : Real.sin γ < 1) :
  (Real.sin α + Real.sin β + Real.sin γ) ^ 2 > 9 * Real.sin α * Real.sin β * Real.sin γ := 
sorry

end sin_sum_square_gt_sin_prod_l553_55319


namespace degree_odd_of_polynomials_l553_55300

theorem degree_odd_of_polynomials 
  (d : ℕ) 
  (P Q : Polynomial ℝ) 
  (hP_deg : P.degree = d) 
  (h_eq : P^2 + 1 = (X^2 + 1) * Q^2) 
  : Odd d :=
sorry

end degree_odd_of_polynomials_l553_55300


namespace parabola_transformation_l553_55301

def original_parabola (x : ℝ) : ℝ := 3 * x^2

def shifted_left (x : ℝ) : ℝ := original_parabola (x + 1)

def shifted_down (x : ℝ) : ℝ := shifted_left x - 2

theorem parabola_transformation :
  shifted_down x = 3 * (x + 1)^2 - 2 :=
sorry

end parabola_transformation_l553_55301


namespace exists_rank_with_profit_2016_l553_55383

theorem exists_rank_with_profit_2016 : ∃ n : ℕ, n * (n + 1) / 2 = 2016 :=
by 
  sorry

end exists_rank_with_profit_2016_l553_55383


namespace symmetric_point_origin_l553_55318

theorem symmetric_point_origin (x y : Int) (hx : x = -(-4)) (hy : y = -(3)) :
    (x, y) = (4, -3) := by
  sorry

end symmetric_point_origin_l553_55318


namespace bryan_push_ups_l553_55325

theorem bryan_push_ups (sets : ℕ) (push_ups_per_set : ℕ) (fewer_in_last_set : ℕ) 
  (h1 : sets = 3) (h2 : push_ups_per_set = 15) (h3 : fewer_in_last_set = 5) :
  (sets - 1) * push_ups_per_set + (push_ups_per_set - fewer_in_last_set) = 40 := by 
  -- We are setting sorry here to skip the proof.
  sorry

end bryan_push_ups_l553_55325


namespace add_base6_numbers_l553_55379

def base6_to_base10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

def base10_to_base6 (n : ℕ) : (ℕ × ℕ × ℕ) := 
  (n / 6^2, (n % 6^2) / 6^1, (n % 6^2) % 6^1)

theorem add_base6_numbers : 
  let n1 := 3 * 6^1 + 5 * 6^0
  let n2 := 2 * 6^1 + 5 * 6^0
  let sum := n1 + n2
  base10_to_base6 sum = (1, 0, 4) :=
by
  -- Proof steps would go here
  sorry

end add_base6_numbers_l553_55379


namespace linear_function_details_l553_55386

variables (x y : ℝ)

noncomputable def linear_function (k b : ℝ) := k * x + b

def passes_through (k b x1 y1 x2 y2 : ℝ) : Prop :=
  y1 = linear_function k b x1 ∧ y2 = linear_function k b x2

def point_on_graph (k b x3 y3 : ℝ) : Prop :=
  y3 = linear_function k b x3

theorem linear_function_details :
  ∃ k b : ℝ, passes_through k b 3 5 (-4) (-9) ∧ point_on_graph k b (-1) (-3) :=
by
  -- to be proved
  sorry

end linear_function_details_l553_55386


namespace stratified_sampling_l553_55371

/-- Given a batch of 98 water heaters with 56 from Factory A and 42 from Factory B,
    and a stratified sample of 14 units is to be drawn, prove that the number 
    of water heaters sampled from Factory A is 8 and from Factory B is 6. --/

theorem stratified_sampling (batch_size A B sample_size : ℕ) 
  (h_batch : batch_size = 98) 
  (h_fact_a : A = 56) 
  (h_fact_b : B = 42) 
  (h_sample : sample_size = 14) : 
  (A * sample_size / batch_size = 8) ∧ (B * sample_size / batch_size = 6) := 
  by
    sorry

end stratified_sampling_l553_55371


namespace sums_remainders_equal_l553_55397

-- Definition and conditions
variables (A A' D S S' s s' : ℕ) 
variables (h1 : A > A') 
variables (h2 : A % D = S) 
variables (h3 : A' % D = S') 
variables (h4 : (A + A') % D = s) 
variables (h5 : (S + S') % D = s')

-- Proof statement
theorem sums_remainders_equal : s = s' := 
  sorry

end sums_remainders_equal_l553_55397


namespace sum_abs_eq_pos_or_neg_three_l553_55333

theorem sum_abs_eq_pos_or_neg_three (x y : Real) (h1 : abs x = 1) (h2 : abs y = 2) (h3 : x * y > 0) :
    x + y = 3 ∨ x + y = -3 :=
by
  sorry

end sum_abs_eq_pos_or_neg_three_l553_55333


namespace smallest_possible_value_l553_55347

theorem smallest_possible_value (a b c d : ℤ) 
  (h1 : a + b + c + d < 25) 
  (h2 : a > 8) 
  (h3 : b < 5) 
  (h4 : c % 2 = 1) 
  (h5 : d % 2 = 0) : 
  ∃ a' b' c' d' : ℤ, a' > 8 ∧ b' < 5 ∧ c' % 2 = 1 ∧ d' % 2 = 0 ∧ a' + b' + c' + d' < 25 ∧ (a' - b' + c' - d' = -4) := 
by 
  use 9, 4, 1, 10
  sorry

end smallest_possible_value_l553_55347


namespace molecular_weight_of_1_mole_l553_55366

theorem molecular_weight_of_1_mole (W_5 : ℝ) (W_1 : ℝ) (h : 5 * W_1 = W_5) (hW5 : W_5 = 490) : W_1 = 490 :=
by
  sorry

end molecular_weight_of_1_mole_l553_55366


namespace total_paint_correct_l553_55370

-- Define the current gallons of paint he has
def current_paint : ℕ := 36

-- Define the gallons of paint he bought
def bought_paint : ℕ := 23

-- Define the additional gallons of paint he needs
def needed_paint : ℕ := 11

-- The total gallons of paint he needs for finishing touches
def total_paint_needed : ℕ := current_paint + bought_paint + needed_paint

-- The proof statement to show that the total paint needed is 70
theorem total_paint_correct : total_paint_needed = 70 := by
  sorry

end total_paint_correct_l553_55370


namespace cos_alpha_plus_20_eq_neg_alpha_l553_55374

variable (α : ℝ)

theorem cos_alpha_plus_20_eq_neg_alpha (h : Real.sin (α - 70 * Real.pi / 180) = α) :
    Real.cos (α + 20 * Real.pi / 180) = -α :=
by
  sorry

end cos_alpha_plus_20_eq_neg_alpha_l553_55374


namespace blanket_thickness_after_foldings_l553_55354

theorem blanket_thickness_after_foldings (initial_thickness : ℕ) (folds : ℕ) (h1 : initial_thickness = 3) (h2 : folds = 4) :
  (initial_thickness * 2^folds) = 48 :=
by
  -- start with definitions as per the conditions
  rw [h1, h2]
  -- proof would follow
  sorry

end blanket_thickness_after_foldings_l553_55354


namespace sum_ratio_15_l553_55393

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequences
def sum_a (n : ℕ) := S n
def sum_b (n : ℕ) := T n

-- The ratio condition
def ratio_condition := ∀ n, a n * (n + 1) = b n * (3 * n + 21)

theorem sum_ratio_15
  (ha : sum_a 15 = 15 * a 8)
  (hb : sum_b 15 = 15 * b 8)
  (h_ratio : ratio_condition a b) :
  sum_a 15 / sum_b 15 = 5 :=
sorry

end sum_ratio_15_l553_55393


namespace halfway_between_one_third_and_one_eighth_l553_55303

theorem halfway_between_one_third_and_one_eighth : (1/3 + 1/8) / 2 = 11 / 48 :=
by
  -- The proof goes here
  sorry

end halfway_between_one_third_and_one_eighth_l553_55303


namespace arithmetic_sequence_10th_term_l553_55315

theorem arithmetic_sequence_10th_term (a_1 : ℕ) (d : ℕ) (n : ℕ) 
  (h1 : a_1 = 1) (h2 : d = 3) (h3 : n = 10) : (a_1 + (n - 1) * d) = 28 := by 
  sorry

end arithmetic_sequence_10th_term_l553_55315


namespace polynomial_factorization_proof_l553_55359

noncomputable def factorizable_binary_quadratic (m : ℚ) : Prop :=
  ∃ (a b : ℚ), (3*a - 5*b = 17) ∧ (a*b = -4) ∧ (m = 2*a + 3*b)

theorem polynomial_factorization_proof :
  ∀ (m : ℚ), factorizable_binary_quadratic m ↔ (m = 5 ∨ m = -58 / 15) :=
by
  sorry

end polynomial_factorization_proof_l553_55359


namespace sum_max_min_eq_four_l553_55355

noncomputable def f (x : ℝ) : ℝ :=
  (|2 * x| + x^3 + 2) / (|x| + 1)

-- Define the maximum value M and minimum value m
noncomputable def M : ℝ := sorry -- The maximum value of the function f(x)
noncomputable def m : ℝ := sorry -- The minimum value of the function f(x)

theorem sum_max_min_eq_four : M + m = 4 := by
  sorry

end sum_max_min_eq_four_l553_55355


namespace find_a_l553_55306

theorem find_a (f : ℕ → ℕ) (a : ℕ) 
  (h1 : ∀ x : ℕ, f (x + 1) = x) 
  (h2 : f a = 8) : a = 9 :=
sorry

end find_a_l553_55306


namespace geese_percentage_non_ducks_l553_55314

theorem geese_percentage_non_ducks :
  let total_birds := 100
  let geese := 0.20 * total_birds
  let swans := 0.30 * total_birds
  let herons := 0.15 * total_birds
  let ducks := 0.25 * total_birds
  let pigeons := 0.10 * total_birds
  let non_duck_birds := total_birds - ducks
  (geese / non_duck_birds) * 100 = 27 := 
by
  sorry

end geese_percentage_non_ducks_l553_55314


namespace triangle_right_angle_l553_55328

theorem triangle_right_angle {a b c : ℝ} {A B C : ℝ} (h : a * Real.cos A + b * Real.cos B = c * Real.cos C) :
  (A = Real.pi / 2) ∨ (B = Real.pi / 2) ∨ (C = Real.pi / 2) :=
sorry

end triangle_right_angle_l553_55328


namespace find_pairs_satisfying_conditions_l553_55344

theorem find_pairs_satisfying_conditions (x y : ℝ) :
    abs (x + y) = 3 ∧ x * y = -10 →
    (x = 5 ∧ y = -2) ∨ (x = -2 ∧ y = 5) ∨ (x = 2 ∧ y = -5) ∨ (x = -5 ∧ y = 2) :=
by
  sorry

end find_pairs_satisfying_conditions_l553_55344


namespace find_number_l553_55312

theorem find_number (x : ℝ) (h : 15 * x = 300) : x = 20 :=
by 
  sorry

end find_number_l553_55312


namespace find_speed_second_part_l553_55307

noncomputable def speed_second_part (x : ℝ) (v : ℝ) : Prop :=
  let t1 := x / 65       -- Time to cover the first x km at 65 kmph
  let t2 := 2 * x / v    -- Time to cover the second 2x km at v kmph
  let avg_time := 3 * x / 26    -- Average speed of the entire journey
  t1 + t2 = avg_time

theorem find_speed_second_part (x : ℝ) (v : ℝ) (h : speed_second_part x v) : v = 86.67 :=
sorry -- Proof of the claim

end find_speed_second_part_l553_55307


namespace part1_part2_l553_55337

open Real

noncomputable def f (x a : ℝ) : ℝ := 45 * abs (x - a) + 45 * abs (x - 5)

theorem part1 (a : ℝ) :
    (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≤ 2 ∨ a ≥ 8) :=
sorry

theorem part2 (a : ℝ) (ha : a = 2) :
    ∀ (x : ℝ), (f x 2 ≥ x^2 - 8*x + 15) ↔ (2 ≤ x ∧ x ≤ 5 + Real.sqrt 3) :=
sorry

end part1_part2_l553_55337


namespace sue_initially_borrowed_six_movies_l553_55350

variable (M : ℕ)
variable (initial_books : ℕ := 15)
variable (returned_books : ℕ := 8)
variable (returned_movies_fraction : ℚ := 1/3)
variable (additional_books : ℕ := 9)
variable (total_items : ℕ := 20)

theorem sue_initially_borrowed_six_movies (hM : total_items = initial_books - returned_books + additional_books + (M - returned_movies_fraction * M)) : 
  M = 6 := by
  sorry

end sue_initially_borrowed_six_movies_l553_55350


namespace mason_water_intake_l553_55329

theorem mason_water_intake
  (Theo_Daily : ℕ := 8)
  (Roxy_Daily : ℕ := 9)
  (Total_Weekly : ℕ := 168)
  (Days_Per_Week : ℕ := 7) :
  (∃ M : ℕ, M * Days_Per_Week = Total_Weekly - (Theo_Daily + Roxy_Daily) * Days_Per_Week ∧ M = 7) :=
  by
  sorry

end mason_water_intake_l553_55329


namespace conditionA_is_necessary_for_conditionB_l553_55365

-- Definitions for conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (area : ℝ) -- area of the triangle

def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

def conditionA (t1 t2 : Triangle) : Prop :=
  t1.area = t2.area ∧ t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Theorem statement
theorem conditionA_is_necessary_for_conditionB (t1 t2 : Triangle) :
  congruent t1 t2 → conditionA t1 t2 :=
by sorry

end conditionA_is_necessary_for_conditionB_l553_55365


namespace problem_I4_1_l553_55321

variable (A D E B C : Type) [Field A] [Field D] [Field E] [Field B] [Field C]
variable (AD DB DE BC : ℚ)
variable (a : ℚ)
variable (h1 : DE = BC) -- DE parallel to BC
variable (h2 : AD = 4)
variable (h3 : DB = 6)
variable (h4 : DE = 6)

theorem problem_I4_1 : a = 15 :=
  by
  sorry

end problem_I4_1_l553_55321


namespace rational_numbers_product_power_l553_55338

theorem rational_numbers_product_power (a b : ℚ) (h : |a - 2| + (2 * b + 1)^2 = 0) :
  (a * b)^2013 = -1 :=
sorry

end rational_numbers_product_power_l553_55338


namespace shiela_used_seven_colors_l553_55324

theorem shiela_used_seven_colors (total_blocks : ℕ) (blocks_per_color : ℕ) 
    (h1 : total_blocks = 49) (h2 : blocks_per_color = 7) : 
    total_blocks / blocks_per_color = 7 :=
by
  sorry

end shiela_used_seven_colors_l553_55324


namespace coffee_bean_price_l553_55390

theorem coffee_bean_price 
  (x : ℝ)
  (price_second : ℝ) (weight_first weight_second : ℝ)
  (total_weight : ℝ) (price_mixture : ℝ) 
  (value_mixture : ℝ) 
  (h1 : price_second = 12)
  (h2 : weight_first = 25)
  (h3 : weight_second = 25)
  (h4 : total_weight = 100)
  (h5 : price_mixture = 11.25)
  (h6 : value_mixture = total_weight * price_mixture)
  (h7 : weight_first + weight_second = total_weight) :
  25 * x + 25 * 12 = 100 * 11.25 → x = 33 :=
by
  intro h
  sorry

end coffee_bean_price_l553_55390


namespace vasya_100_using_fewer_sevens_l553_55378

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end vasya_100_using_fewer_sevens_l553_55378


namespace option_C_correct_l553_55323

theorem option_C_correct (x : ℝ) (hx : 0 < x) : x + 1 / x ≥ 2 :=
sorry

end option_C_correct_l553_55323


namespace quadratic_inequality_solution_l553_55339

variables {x : ℝ} {f : ℝ → ℝ}

def is_quadratic_and_opens_downwards (f : ℝ → ℝ) : Prop :=
  ∃ a b c, a < 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def is_symmetric_at_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = f (2 + x)

theorem quadratic_inequality_solution
  (h_quadratic : is_quadratic_and_opens_downwards f)
  (h_symmetric : is_symmetric_at_two f) :
  (1 - (Real.sqrt 14) / 4) < x ∧ x < (1 + (Real.sqrt 14) / 4) ↔
  f (Real.log ((1 / (1 / 4)) * (x^2 + x + 1 / 2))) <
  f (Real.log ((1 / (1 / 2)) * (2 * x^2 - x + 5 / 8))) :=
sorry

end quadratic_inequality_solution_l553_55339


namespace least_number_added_1789_l553_55376

def least_number_added_to_divisible (n d : ℕ) : ℕ := d - (n % d)

theorem least_number_added_1789 :
  least_number_added_to_divisible 1789 (Nat.lcm (Nat.lcm 5 6) (Nat.lcm 4 3)) = 11 :=
by
  -- Step definitions
  have lcm_5_6 := Nat.lcm 5 6
  have lcm_4_3 := Nat.lcm 4 3
  have lcm_total := Nat.lcm lcm_5_6 lcm_4_3
  -- Computation of the final result
  have remainder := 1789 % lcm_total
  have required_add := lcm_total - remainder
  -- Conclusion based on the computed values
  sorry

end least_number_added_1789_l553_55376


namespace max_mn_square_proof_l553_55320

noncomputable def max_mn_square (m n : ℕ) : ℕ :=
m^2 + n^2

theorem max_mn_square_proof (m n : ℕ) (h1 : 1 ≤ m ∧ m ≤ 2005) (h2 : 1 ≤ n ∧ n ≤ 2005) (h3 : (n^2 + 2 * m * n - 2 * m^2)^2 = 1) : 
max_mn_square m n ≤ 702036 :=
sorry

end max_mn_square_proof_l553_55320


namespace probability_greater_than_two_on_three_dice_l553_55372

theorem probability_greater_than_two_on_three_dice :
  (4 / 6 : ℚ) ^ 3 = (8 / 27 : ℚ) :=
by
  sorry

end probability_greater_than_two_on_three_dice_l553_55372


namespace complement_union_l553_55363

noncomputable def A : Set ℝ := { x : ℝ | x^2 - x - 2 ≤ 0 }
noncomputable def B : Set ℝ := { x : ℝ | 1 < x ∧ x ≤ 3 }
noncomputable def CR (S : Set ℝ) : Set ℝ := { x : ℝ | x ∉ S }

theorem complement_union (A B : Set ℝ) :
  (CR A ∪ B) = (Set.univ \ A ∪ Set.Ioo 1 3) := by
  sorry

end complement_union_l553_55363


namespace no_right_triangle_with_sqrt_2016_side_l553_55322

theorem no_right_triangle_with_sqrt_2016_side :
  ¬ ∃ (a b : ℤ), (a * a + b * b = 2016) ∨ (a * a + 2016 = b * b) :=
by
  sorry

end no_right_triangle_with_sqrt_2016_side_l553_55322
