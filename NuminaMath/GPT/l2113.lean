import Mathlib

namespace NUMINAMATH_GPT_ZacharysBusRideLength_l2113_211316

theorem ZacharysBusRideLength (vince_ride zach_ride : ℝ) 
  (h1 : vince_ride = 0.625) 
  (h2 : vince_ride = zach_ride + 0.125) : 
  zach_ride = 0.500 := 
by
  sorry

end NUMINAMATH_GPT_ZacharysBusRideLength_l2113_211316


namespace NUMINAMATH_GPT_cos_5theta_l2113_211393

theorem cos_5theta (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (5*θ) = -93/3125 :=
sorry

end NUMINAMATH_GPT_cos_5theta_l2113_211393


namespace NUMINAMATH_GPT_sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l2113_211348

noncomputable def calculate_time (distance1 distance2 speed1 speed2 : ℕ) : ℕ := 
  (distance1 / speed1) + (distance2 / speed2)

noncomputable def total_time_per_lap := calculate_time 200 100 4 6

theorem sofia_total_time_for_5_laps : total_time_per_lap * 5 = 335 := 
  by sorry

def converted_time (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / 60, total_seconds % 60)

theorem sofia_total_time_in_minutes_and_seconds :
  converted_time (total_time_per_lap * 5) = (5, 35) :=
  by sorry

end NUMINAMATH_GPT_sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l2113_211348


namespace NUMINAMATH_GPT_Tim_change_l2113_211392

theorem Tim_change (initial_amount paid_amount : ℕ) (h₀ : initial_amount = 50) (h₁ : paid_amount = 45) : initial_amount - paid_amount = 5 :=
by
  sorry

end NUMINAMATH_GPT_Tim_change_l2113_211392


namespace NUMINAMATH_GPT_sample_size_second_grade_l2113_211370

theorem sample_size_second_grade
    (total_students : ℕ)
    (ratio_first : ℕ)
    (ratio_second : ℕ)
    (ratio_third : ℕ)
    (sample_size : ℕ) :
    total_students = 2000 →
    ratio_first = 5 → ratio_second = 3 → ratio_third = 2 →
    sample_size = 20 →
    (20 * (3 / (5 + 3 + 2)) = 6) :=
by
  intros ht hr1 hr2 hr3 hs
  -- The proof would continue from here, but we're finished as the task only requires the statement.
  sorry

end NUMINAMATH_GPT_sample_size_second_grade_l2113_211370


namespace NUMINAMATH_GPT_linear_equations_not_always_solvable_l2113_211389

theorem linear_equations_not_always_solvable 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : 
  ¬(∀ x y : ℝ, (a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂) ↔ 
                   a₁ * b₂ - a₂ * b₁ ≠ 0) :=
sorry

end NUMINAMATH_GPT_linear_equations_not_always_solvable_l2113_211389


namespace NUMINAMATH_GPT_max_value_f_l2113_211317

theorem max_value_f (x y z : ℝ) (hxyz : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (1 - y * z + z) * (1 - x * z + x) * (1 - x * y + y) ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_f_l2113_211317


namespace NUMINAMATH_GPT_no_solution_for_inequalities_l2113_211350

theorem no_solution_for_inequalities (x : ℝ) : ¬ ((6 * x - 2 < (x + 2) ^ 2) ∧ ((x + 2) ^ 2 < 9 * x - 5)) :=
by sorry

end NUMINAMATH_GPT_no_solution_for_inequalities_l2113_211350


namespace NUMINAMATH_GPT_least_days_to_repay_twice_l2113_211344

-- Define the initial conditions
def borrowed_amount : ℝ := 15
def daily_interest_rate : ℝ := 0.10
def interest_per_day : ℝ := borrowed_amount * daily_interest_rate
def total_amount_to_repay : ℝ := 2 * borrowed_amount

-- Define the condition we want to prove
theorem least_days_to_repay_twice : ∃ (x : ℕ), (borrowed_amount + interest_per_day * x) ≥ total_amount_to_repay ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_least_days_to_repay_twice_l2113_211344


namespace NUMINAMATH_GPT_adults_riding_bicycles_l2113_211340

theorem adults_riding_bicycles (A : ℕ) (H1 : 15 * 3 + 2 * A = 57) : A = 6 :=
by
  sorry

end NUMINAMATH_GPT_adults_riding_bicycles_l2113_211340


namespace NUMINAMATH_GPT_right_triangle_max_area_l2113_211387

theorem right_triangle_max_area
  (a b : ℝ) (h_a_nonneg : 0 ≤ a) (h_b_nonneg : 0 ≤ b)
  (h_right_triangle : a^2 + b^2 = 20^2)
  (h_perimeter : a + b + 20 = 48) :
  (1 / 2) * a * b = 96 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_max_area_l2113_211387


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_eq_l2113_211399

theorem sum_of_roots_of_quadratic_eq :
  ∀ x : ℝ, x^2 + 2023 * x - 2024 = 0 → 
  x = -2023 := 
sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_eq_l2113_211399


namespace NUMINAMATH_GPT_min_value_x_plus_4y_l2113_211398

theorem min_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 2 * x * y) : x + 4 * y = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_4y_l2113_211398


namespace NUMINAMATH_GPT_power_function_through_point_l2113_211308

noncomputable def f : ℝ → ℝ := sorry

theorem power_function_through_point (h : ∀ x, ∃ a : ℝ, f x = x^a) (h1 : f 3 = 27) :
  f x = x^3 :=
sorry

end NUMINAMATH_GPT_power_function_through_point_l2113_211308


namespace NUMINAMATH_GPT_length_of_cloth_l2113_211337

theorem length_of_cloth (L : ℝ) (h : 35 = (L + 4) * (35 / L - 1)) : L = 10 :=
sorry

end NUMINAMATH_GPT_length_of_cloth_l2113_211337


namespace NUMINAMATH_GPT_monotonicity_f_l2113_211359

open Set

noncomputable def f (a x : ℝ) : ℝ := a * x / (x - 1)

theorem monotonicity_f (a : ℝ) (h : a ≠ 0) :
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → (if a > 0 then f a x1 > f a x2 else if a < 0 then f a x1 < f a x2 else False)) :=
by
  sorry

end NUMINAMATH_GPT_monotonicity_f_l2113_211359


namespace NUMINAMATH_GPT_tan_half_alpha_eq_one_third_l2113_211367

open Real

theorem tan_half_alpha_eq_one_third (α : ℝ) (h1 : 5 * sin (2 * α) = 6 * cos α) (h2 : 0 < α ∧ α < π / 2) :
  tan (α / 2) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_half_alpha_eq_one_third_l2113_211367


namespace NUMINAMATH_GPT_fraction_exponent_product_l2113_211394

theorem fraction_exponent_product :
  ( (5/6: ℚ)^2 * (2/3: ℚ)^3 = 50/243 ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_exponent_product_l2113_211394


namespace NUMINAMATH_GPT_simpsons_hats_l2113_211347

variable (S : ℕ)
variable (O : ℕ)

-- Define the conditions: O'Brien's hats before losing one
def obriens_hats_before : Prop := O = 2 * S + 5

-- Define the current number of O'Brien's hats
def obriens_current_hats : Prop := O = 34 + 1

-- Main theorem statement
theorem simpsons_hats : obriens_hats_before S O ∧ obriens_current_hats O → S = 15 := 
by
  sorry

end NUMINAMATH_GPT_simpsons_hats_l2113_211347


namespace NUMINAMATH_GPT_distance_between_X_and_Y_l2113_211390

def distance_XY := 31

theorem distance_between_X_and_Y
  (yolanda_rate : ℕ) (bob_rate : ℕ) (bob_walked : ℕ) (time_difference : ℕ) :
  yolanda_rate = 1 →
  bob_rate = 2 →
  bob_walked = 20 →
  time_difference = 1 →
  distance_XY = bob_walked + (bob_walked / bob_rate + time_difference) * yolanda_rate :=
by
  intros hy hb hbw htd
  sorry

end NUMINAMATH_GPT_distance_between_X_and_Y_l2113_211390


namespace NUMINAMATH_GPT_sum_of_conjugates_eq_30_l2113_211301

theorem sum_of_conjugates_eq_30 :
  (15 - Real.sqrt 2023) + (15 + Real.sqrt 2023) = 30 :=
sorry

end NUMINAMATH_GPT_sum_of_conjugates_eq_30_l2113_211301


namespace NUMINAMATH_GPT_value_of_x_for_real_y_l2113_211351

theorem value_of_x_for_real_y (x y : ℝ) (h : 4 * y^2 - 2 * x * y + 2 * x + 9 = 0) : x ≤ -3 ∨ x ≥ 12 :=
sorry

end NUMINAMATH_GPT_value_of_x_for_real_y_l2113_211351


namespace NUMINAMATH_GPT_equation_of_line_intersection_l2113_211369

theorem equation_of_line_intersection
  (h1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (h2 : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) :
  ∀ x y : ℝ, x - 2*y + 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_intersection_l2113_211369


namespace NUMINAMATH_GPT_part1_part2_l2113_211385

theorem part1 (x y : ℝ) (h1 : x + 3 * y = 26) (h2 : 2 * x + y = 22) : x = 8 ∧ y = 6 :=
by
  sorry

theorem part2 (m : ℝ) (h : 8 * m + 6 * (15 - m) ≤ 100) : m ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2113_211385


namespace NUMINAMATH_GPT_eval_expression_l2113_211326

theorem eval_expression (a : ℕ) (h : a = 2) : 
  8^3 + 4 * a * 8^2 + 6 * a^2 * 8 + a^3 = 1224 := 
by
  rw [h]
  sorry

end NUMINAMATH_GPT_eval_expression_l2113_211326


namespace NUMINAMATH_GPT_alex_casey_meet_probability_l2113_211330

noncomputable def probability_meet : ℚ :=
  let L := (1:ℚ) / 3;
  let area_of_square := 1;
  let area_of_triangles := (1 / 2) * L ^ 2;
  let area_of_meeting_region := area_of_square - 2 * area_of_triangles;
  area_of_meeting_region / area_of_square

theorem alex_casey_meet_probability :
  probability_meet = 8 / 9 :=
by
  sorry

end NUMINAMATH_GPT_alex_casey_meet_probability_l2113_211330


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2113_211319

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x, x ≥ a → x^2 - x - 2 ≥ 0) ∧ (∃ x, x ≥ a ∧ ¬(x^2 - x - 2 ≥ 0)) ↔ a ≥ 2 := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2113_211319


namespace NUMINAMATH_GPT_solve_abs_equation_l2113_211345

theorem solve_abs_equation (x : ℝ) (h : |2001 * x - 2001| = 2001) : x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_GPT_solve_abs_equation_l2113_211345


namespace NUMINAMATH_GPT_Hari_contribution_l2113_211396

theorem Hari_contribution (P T_P T_H : ℕ) (r1 r2 : ℕ) (H : ℕ) :
  P = 3500 → 
  T_P = 12 → 
  T_H = 7 → 
  r1 = 2 → 
  r2 = 3 →
  (P * T_P) * r2 = (H * T_H) * r1 →
  H = 9000 :=
by
  sorry

end NUMINAMATH_GPT_Hari_contribution_l2113_211396


namespace NUMINAMATH_GPT_vertex_not_neg2_2_l2113_211332

theorem vertex_not_neg2_2 (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : a * 1^2 + b * 1 + c = 0)
  (hsymm : ∀ x y, y = a * x^2 + b * x + c → y = a * (4 - x)^2 + b * (4 - x) + c) :
  ¬ ((-b) / (2 * a) = -2 ∧ a * (-2)^2 + b * (-2) + c = 2) :=
by
  sorry

end NUMINAMATH_GPT_vertex_not_neg2_2_l2113_211332


namespace NUMINAMATH_GPT_part_I_part_II_l2113_211328

namespace VectorProblems

def vector_a : ℝ × ℝ := (3, 2)
def vector_b : ℝ × ℝ := (-1, 2)
def vector_c : ℝ × ℝ := (4, 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem part_I (m : ℝ) :
  let u := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
  let v := (4 * m + vector_b.1, m + vector_b.2)
  dot_product u v > 0 →
  m ≠ 4 / 7 →
  m > -1 / 2 :=
sorry

theorem part_II (k : ℝ) :
  let u := (vector_a.1 + 4 * k, vector_a.2 + k)
  let v := (2 * vector_b.1 - vector_a.1, 2 * vector_b.2 - vector_a.2)
  dot_product u v = 0 →
  k = -11 / 18 :=
sorry

end VectorProblems

end NUMINAMATH_GPT_part_I_part_II_l2113_211328


namespace NUMINAMATH_GPT_minimize_distance_l2113_211321

noncomputable def f : ℝ → ℝ := λ x => x ^ 2
noncomputable def g : ℝ → ℝ := λ x => Real.log x
noncomputable def y : ℝ → ℝ := λ x => f x - g x

theorem minimize_distance (t : ℝ) (ht : t = Real.sqrt 2 / 2) :
  ∀ x > 0, y x ≥ y (Real.sqrt 2 / 2) := sorry

end NUMINAMATH_GPT_minimize_distance_l2113_211321


namespace NUMINAMATH_GPT_tenth_term_of_sequence_l2113_211361

variable (a : ℕ → ℚ) (n : ℕ)

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem tenth_term_of_sequence :
  let a₁ := (5 : ℚ)
  let r := (5 / 3 : ℚ)
  geometric_sequence a₁ r 10 = (9765625 / 19683 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_of_sequence_l2113_211361


namespace NUMINAMATH_GPT_solution_set_l2113_211312

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set (c1 : ∀ x : ℝ, f x + f' x > 1)
                     (c2 : f 0 = 2) :
  {x : ℝ | e^x * f x > e^x + 1} = {x : ℝ | 0 < x} :=
sorry

end NUMINAMATH_GPT_solution_set_l2113_211312


namespace NUMINAMATH_GPT_gym_monthly_income_l2113_211366

-- Define the conditions
def twice_monthly_charge : ℕ := 18
def monthly_charge_per_member : ℕ := 2 * twice_monthly_charge
def number_of_members : ℕ := 300

-- State the goal: the monthly income of the gym
def monthly_income : ℕ := 36 * 300

-- The theorem to prove
theorem gym_monthly_income : monthly_charge_per_member * number_of_members = 10800 :=
by
  sorry

end NUMINAMATH_GPT_gym_monthly_income_l2113_211366


namespace NUMINAMATH_GPT_pine_tree_taller_than_birch_l2113_211306

def height_birch : ℚ := 49 / 4
def height_pine : ℚ := 74 / 4

def height_difference : ℚ :=
  height_pine - height_birch

theorem pine_tree_taller_than_birch :
  height_difference = 25 / 4 :=
by
  sorry

end NUMINAMATH_GPT_pine_tree_taller_than_birch_l2113_211306


namespace NUMINAMATH_GPT_find_b_l2113_211334

theorem find_b (g : ℝ → ℝ) (a b : ℝ) (h1 : ∀ x, g (-x) = -g x) (h2 : ∃ x, g x ≠ 0) 
               (h3 : a > 0) (h4 : a ≠ 1) (h5 : ∀ x, (1 / (a ^ x - 1) - 1 / b) * g x = (1 / (a ^ (-x) - 1) - 1 / b) * g (-x)) :
    b = -2 :=
sorry

end NUMINAMATH_GPT_find_b_l2113_211334


namespace NUMINAMATH_GPT_incorrect_statement_l2113_211307

noncomputable def f (a b c : ℤ) (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem incorrect_statement
  (a b c : ℤ) (h₀ : a ≠ 0)
  (h₁ : 2 * a + b = 0)
  (h₂ : f a b c 1 = 3)
  (h₃ : f a b c 2 = 8) :
  ¬ (f a b c (-1) = 0) :=
sorry

end NUMINAMATH_GPT_incorrect_statement_l2113_211307


namespace NUMINAMATH_GPT_ratio_of_times_l2113_211333

theorem ratio_of_times (D S : ℝ) (hD : D = 27) (hS : S / 2 = D / 2 + 13.5) :
  D / S = 1 / 2 :=
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_ratio_of_times_l2113_211333


namespace NUMINAMATH_GPT_probability_of_selection_l2113_211353

theorem probability_of_selection (total_students : ℕ) (eliminated_students : ℕ) (groups : ℕ) (selected_students : ℕ)
(h1 : total_students = 1003) 
(h2 : eliminated_students = 3)
(h3 : groups = 20)
(h4 : selected_students = 50) : 
(selected_students : ℝ) / (total_students : ℝ) = 50 / 1003 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selection_l2113_211353


namespace NUMINAMATH_GPT_negation_of_forall_x_geq_1_l2113_211305

theorem negation_of_forall_x_geq_1 :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 1)) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_forall_x_geq_1_l2113_211305


namespace NUMINAMATH_GPT_alex_annual_income_l2113_211375

theorem alex_annual_income (q : ℝ) (B : ℝ)
  (H1 : 0.01 * q * 50000 + 0.01 * (q + 3) * (B - 50000) = 0.01 * (q + 0.5) * B) :
  B = 60000 :=
by sorry

end NUMINAMATH_GPT_alex_annual_income_l2113_211375


namespace NUMINAMATH_GPT_red_balls_estimate_l2113_211360

/-- There are several red balls and 4 black balls in a bag.
Each ball is identical except for color.
A ball is drawn and put back into the bag. This process is repeated 100 times.
Among those 100 draws, 40 times a black ball is drawn.
Prove that the number of red balls (x) is 6. -/
theorem red_balls_estimate (x : ℕ) (h_condition : (4 / (4 + x) = 40 / 100)) : x = 6 :=
by
    sorry

end NUMINAMATH_GPT_red_balls_estimate_l2113_211360


namespace NUMINAMATH_GPT_M_subset_P_l2113_211320

def M := {x : ℕ | ∃ a : ℕ, 0 < a ∧ x = a^2 + 1}
def P := {y : ℕ | ∃ b : ℕ, 0 < b ∧ y = b^2 - 4*b + 5}

theorem M_subset_P : M ⊂ P :=
by
  sorry

end NUMINAMATH_GPT_M_subset_P_l2113_211320


namespace NUMINAMATH_GPT_polynomial_solution_characterization_l2113_211346

theorem polynomial_solution_characterization (P : ℝ → ℝ → ℝ) (h : ∀ x y z : ℝ, P x (2 * y * z) + P y (2 * z * x) + P z (2 * x * y) = P (x + y + z) (x * y + y * z + z * x)) :
  ∃ (a b : ℝ), ∀ x y : ℝ, P x y = a * x + b * (x^2 + 2 * y) :=
sorry

end NUMINAMATH_GPT_polynomial_solution_characterization_l2113_211346


namespace NUMINAMATH_GPT_water_hyacinth_indicates_connection_l2113_211382

-- Definitions based on the conditions
def universally_interconnected : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (c : Type), (a ≠ c) ∧ (b ≠ c)

def connections_diverse : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (f : a → b), ∀ (x y : a), x ≠ y → f x ≠ f y

def connections_created : Prop :=
  ∃ (a b : Type), a ≠ b ∧ (∀ (f : a → b), False)

def connections_humanized : Prop :=
  ∀ (a b : Type), a ≠ b → (∃ c : Type, a = c) ∧ (∃ d : Type, b = d)

-- Problem statement
theorem water_hyacinth_indicates_connection : 
  universally_interconnected ∧ connections_diverse :=
by
  sorry

end NUMINAMATH_GPT_water_hyacinth_indicates_connection_l2113_211382


namespace NUMINAMATH_GPT_minutes_past_midnight_l2113_211381

-- Definitions for the problem

def degree_per_tick : ℝ := 30
def degree_per_minute_hand : ℝ := 6
def degree_per_hour_hand_hourly : ℝ := 30
def degree_per_hour_hand_minutes : ℝ := 0.5

def condition_minute_hand_degree := 300
def condition_hour_hand_degree := 70

-- Main theorem statement
theorem minutes_past_midnight :
  ∃ (h m: ℝ),
    degree_per_hour_hand_hourly * h + degree_per_hour_hand_minutes * m = condition_hour_hand_degree ∧
    degree_per_minute_hand * m = condition_minute_hand_degree ∧
    h * 60 + m = 110 :=
by
  sorry

end NUMINAMATH_GPT_minutes_past_midnight_l2113_211381


namespace NUMINAMATH_GPT_johns_meeting_distance_l2113_211322

theorem johns_meeting_distance (d t: ℝ) 
    (h1 : d = 40 * (t + 1.5))
    (h2 : d - 40 = 60 * (t - 2)) :
    d = 420 :=
by sorry

end NUMINAMATH_GPT_johns_meeting_distance_l2113_211322


namespace NUMINAMATH_GPT_digits_count_concatenated_l2113_211374

-- Define the conditions for the digit count of 2^n and 5^n
def digits_count_2n (n p : ℕ) : Prop := 10^(p-1) ≤ 2^n ∧ 2^n < 10^p
def digits_count_5n (n q : ℕ) : Prop := 10^(q-1) ≤ 5^n ∧ 5^n < 10^q

-- The main theorem to prove the number of digits when 2^n and 5^n are concatenated
theorem digits_count_concatenated (n p q : ℕ) 
  (h1 : digits_count_2n n p) 
  (h2 : digits_count_5n n q): 
  p + q = n + 1 := by 
  sorry

end NUMINAMATH_GPT_digits_count_concatenated_l2113_211374


namespace NUMINAMATH_GPT_john_average_speed_l2113_211378

/--
John drove continuously from 8:15 a.m. until 2:05 p.m. of the same day 
and covered a distance of 210 miles. Prove that his average speed in 
miles per hour was 36 mph.
-/
theorem john_average_speed :
  (210 : ℝ) / (((2 - 8) * 60 + 5 - 15) / 60) = 36 := by
  sorry

end NUMINAMATH_GPT_john_average_speed_l2113_211378


namespace NUMINAMATH_GPT_equal_division_of_cookie_l2113_211356

theorem equal_division_of_cookie (total_area : ℝ) (friends : ℕ) (area_per_person : ℝ) 
  (h1 : total_area = 81.12) 
  (h2 : friends = 6) 
  (h3 : area_per_person = total_area / friends) : 
  area_per_person = 13.52 :=
by 
  sorry

end NUMINAMATH_GPT_equal_division_of_cookie_l2113_211356


namespace NUMINAMATH_GPT_quadratic_to_vertex_form_addition_l2113_211324

theorem quadratic_to_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intro h_eq
  sorry

end NUMINAMATH_GPT_quadratic_to_vertex_form_addition_l2113_211324


namespace NUMINAMATH_GPT_ratio_of_a_over_5_to_b_over_4_l2113_211341

theorem ratio_of_a_over_5_to_b_over_4 (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) : (a/5) / (b/4) = 1 :=
sorry

end NUMINAMATH_GPT_ratio_of_a_over_5_to_b_over_4_l2113_211341


namespace NUMINAMATH_GPT_simplify_expression_l2113_211377

theorem simplify_expression (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  ( (1/(a-b) - 2 * a * b / (a^3 - a^2 * b + a * b^2 - b^3)) / 
    ((a^2 + a * b) / (a^3 + a^2 * b + a * b^2 + b^3) + 
    b / (a^2 + b^2)) ) = (a - b) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2113_211377


namespace NUMINAMATH_GPT_compare_log_values_l2113_211336

noncomputable def a : ℝ := (Real.log 2) / 2
noncomputable def b : ℝ := (Real.log 3) / 3
noncomputable def c : ℝ := (Real.log 5) / 5

theorem compare_log_values : c < a ∧ a < b := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_compare_log_values_l2113_211336


namespace NUMINAMATH_GPT_larger_of_two_numbers_l2113_211329

  theorem larger_of_two_numbers (x y : ℕ) 
    (h₁ : x + y = 37) 
    (h₂ : x - y = 5) 
    : x = 21 :=
  sorry
  
end NUMINAMATH_GPT_larger_of_two_numbers_l2113_211329


namespace NUMINAMATH_GPT_value_of_f_at_1_l2113_211388

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem value_of_f_at_1 : f 1 = 2 :=
by sorry

end NUMINAMATH_GPT_value_of_f_at_1_l2113_211388


namespace NUMINAMATH_GPT_min_value_of_vectors_l2113_211397

theorem min_value_of_vectors (m n : ℝ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : (m * (n - 2)) + 1 = 0) : (1 / m) + (2 / n) = 2 * Real.sqrt 2 + 3 / 2 :=
by sorry

end NUMINAMATH_GPT_min_value_of_vectors_l2113_211397


namespace NUMINAMATH_GPT_directrix_of_parabola_l2113_211386

theorem directrix_of_parabola : ∀ (x : ℝ), y = (x^2 - 8*x + 12) / 16 → ∃ (d : ℝ), d = -1/2 := 
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l2113_211386


namespace NUMINAMATH_GPT_min_m_value_l2113_211342

noncomputable def f (x a : ℝ) : ℝ := 2 ^ (abs (x - a))

theorem min_m_value :
  ∀ a, (∀ x, f (1 + x) a = f (1 - x) a) →
  ∃ m : ℝ, (∀ x : ℝ, x ≥ m → ∀ y : ℝ, y ≥ x → f y a ≥ f x a) ∧ m = 1 :=
by
  intros a h
  sorry

end NUMINAMATH_GPT_min_m_value_l2113_211342


namespace NUMINAMATH_GPT_total_points_other_five_l2113_211368

theorem total_points_other_five
  (x : ℕ) -- total number of points scored by the team
  (d : ℕ) (e : ℕ) (f : ℕ) (y : ℕ) -- points scored by Daniel, Emma, Fiona, and others respectively
  (hd : d = x / 3) -- Daniel scored 1/3 of the team's points
  (he : e = 3 * x / 8) -- Emma scored 3/8 of the team's points
  (hf : f = 18) -- Fiona scored 18 points
  (h_other : ∀ i, 1 ≤ i ∧ i ≤ 5 → y ≤ 15 / 5) -- Other 5 members scored no more than 3 points each
  (h_total : d + e + f + y = x) -- Total points equation
  : y = 14 := sorry -- Final number of points scored by the other 5 members

end NUMINAMATH_GPT_total_points_other_five_l2113_211368


namespace NUMINAMATH_GPT_number_of_friends_l2113_211302

/- Define the conditions -/
def sandwiches_per_friend : Nat := 3
def total_sandwiches : Nat := 12

/- Define the mathematical statement to be proven -/
theorem number_of_friends : (total_sandwiches / sandwiches_per_friend) = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l2113_211302


namespace NUMINAMATH_GPT_quadratic_roots_satisfy_condition_l2113_211357
variable (x1 x2 m : ℝ)

theorem quadratic_roots_satisfy_condition :
  ( ∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1 + x2 = -m) ∧ 
    (x1 * x2 = 5) ∧ (x1 = 2 * |x2| - 3) ) →
  m = -9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_satisfy_condition_l2113_211357


namespace NUMINAMATH_GPT_library_books_l2113_211352

theorem library_books (a : ℕ) (R L : ℕ) :
  (∃ R, a = 12 * R + 7) ∧ (∃ L, a = 25 * L - 5) ∧ 500 < a ∧ a < 650 → a = 595 :=
by
  sorry

end NUMINAMATH_GPT_library_books_l2113_211352


namespace NUMINAMATH_GPT_equation_nth_position_l2113_211363

theorem equation_nth_position (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 :=
by
  sorry

end NUMINAMATH_GPT_equation_nth_position_l2113_211363


namespace NUMINAMATH_GPT_fifty_times_reciprocal_of_eight_times_number_three_l2113_211314

theorem fifty_times_reciprocal_of_eight_times_number_three (x : ℚ) 
  (h : 8 * x = 3) : 50 * (1 / x) = 133 + 1 / 3 :=
sorry

end NUMINAMATH_GPT_fifty_times_reciprocal_of_eight_times_number_three_l2113_211314


namespace NUMINAMATH_GPT_problem_subtraction_of_negatives_l2113_211349

theorem problem_subtraction_of_negatives :
  12.345 - (-3.256) = 15.601 :=
sorry

end NUMINAMATH_GPT_problem_subtraction_of_negatives_l2113_211349


namespace NUMINAMATH_GPT_range_of_a_for_common_points_l2113_211331

theorem range_of_a_for_common_points (a : ℝ) : (∃ x : ℝ, x > 0 ∧ ax^2 = Real.exp x) ↔ a ≥ Real.exp 2 / 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_for_common_points_l2113_211331


namespace NUMINAMATH_GPT_find_p_l2113_211384

variable (a b p q r1 r2 : ℝ)

-- Given conditions
def roots_eq1 (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) : Prop :=
  -- Using Vieta's Formulas on x^2 + ax + b = 0
  ∀ (r1 r2 : ℝ), r1 + r2 = -a ∧ r1 * r2 = b

def roots_eq2 (r1 r2 : ℝ) (h_3 : r1^2 + r2^2 = -p) (h_4 : r1^2 * r2^2 = q) : Prop :=
  -- Using Vieta's Formulas on x^2 + px + q = 0
  ∀ (r1 r2 : ℝ), r1^2 + r2^2 = -p ∧ r1^2 * r2^2 = q

-- Theorems
theorem find_p (h_1 : r1 + r2 = -a) (h_2 : r1 * r2 = b) (h_3 : r1^2 + r2^2 = -p) :
  p = -a^2 + 2*b := by
  sorry

end NUMINAMATH_GPT_find_p_l2113_211384


namespace NUMINAMATH_GPT_complex_division_l2113_211325

theorem complex_division :
  (1 - 2 * Complex.I) / (2 + Complex.I) = -Complex.I :=
by sorry

end NUMINAMATH_GPT_complex_division_l2113_211325


namespace NUMINAMATH_GPT_orange_ratio_l2113_211391

theorem orange_ratio (total_oranges alice_oranges : ℕ) (h_total : total_oranges = 180) (h_alice : alice_oranges = 120) :
  alice_oranges / (total_oranges - alice_oranges) = 2 :=
by
  sorry

end NUMINAMATH_GPT_orange_ratio_l2113_211391


namespace NUMINAMATH_GPT_e_n_max_value_l2113_211376

def b (n : ℕ) : ℕ := (5^n - 1) / 4

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem e_n_max_value (n : ℕ) : e n = 1 := 
by sorry

end NUMINAMATH_GPT_e_n_max_value_l2113_211376


namespace NUMINAMATH_GPT_find_functions_l2113_211318

noncomputable def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (p q : ℝ), p ≠ q → (f q - f p) / (q - p) * 0 + f p - (f q - f p) / (q - p) * p = p * q

theorem find_functions (f : ℝ → ℝ) (c : ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = x * (c + x)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_functions_l2113_211318


namespace NUMINAMATH_GPT_area_of_polygon_ABCDEF_l2113_211327

-- Definitions based on conditions
def AB : ℕ := 8
def BC : ℕ := 10
def DC : ℕ := 5
def FA : ℕ := 7
def GF : ℕ := 3
def ED : ℕ := 7
def height_GF_ED : ℕ := 2

-- Area calculations based on given conditions
def area_ABCG : ℕ := AB * BC
def area_trapezoid_GFED : ℕ := (GF + ED) * height_GF_ED / 2

-- Proof statement
theorem area_of_polygon_ABCDEF :
  area_ABCG - area_trapezoid_GFED = 70 :=
by
  simp [area_ABCG, area_trapezoid_GFED]
  sorry

end NUMINAMATH_GPT_area_of_polygon_ABCDEF_l2113_211327


namespace NUMINAMATH_GPT_distance_between_P1_and_P2_l2113_211309

-- Define the two points
def P1 : ℝ × ℝ := (2, 3)
def P2 : ℝ × ℝ := (5, 10)

-- Define the distance function
noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Define the theorem we want to prove
theorem distance_between_P1_and_P2 :
  distance P1 P2 = Real.sqrt 58 :=
by sorry

end NUMINAMATH_GPT_distance_between_P1_and_P2_l2113_211309


namespace NUMINAMATH_GPT_square_distance_between_intersections_l2113_211354

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 7)^2 + y^2 = 4

-- Problem: Prove the square of the distance between intersection points P and Q
theorem square_distance_between_intersections :
  (∃ (x y1 y2 : ℝ), circle1 x y1 ∧ circle2 x y1 ∧ circle1 x y2 ∧ circle2 x y2 ∧ y1 ≠ y2) →
  ∃ d : ℝ, d^2 = 15.3664 :=
by
  sorry

end NUMINAMATH_GPT_square_distance_between_intersections_l2113_211354


namespace NUMINAMATH_GPT_train_distance_900_l2113_211362

theorem train_distance_900 (x t : ℝ) (H1 : x = 50 * t) (H2 : x - 100 = 40 * t) : 
  x + (x - 100) = 900 :=
by
  sorry

end NUMINAMATH_GPT_train_distance_900_l2113_211362


namespace NUMINAMATH_GPT_candies_distribution_l2113_211343

theorem candies_distribution (C : ℕ) (hC : C / 150 = C / 300 + 24) : C / 150 = 48 :=
by sorry

end NUMINAMATH_GPT_candies_distribution_l2113_211343


namespace NUMINAMATH_GPT_Tyler_scissors_count_l2113_211358

variable (S : ℕ)

def Tyler_initial_money : ℕ := 100
def cost_per_scissors : ℕ := 5
def number_of_erasers : ℕ := 10
def cost_per_eraser : ℕ := 4
def Tyler_remaining_money : ℕ := 20

theorem Tyler_scissors_count :
  Tyler_initial_money - (cost_per_scissors * S + number_of_erasers * cost_per_eraser) = Tyler_remaining_money →
  S = 8 :=
by
  sorry

end NUMINAMATH_GPT_Tyler_scissors_count_l2113_211358


namespace NUMINAMATH_GPT_unit_vector_norm_equal_l2113_211364

variables (a b : EuclideanSpace ℝ (Fin 2)) -- assuming 2D Euclidean space for simplicity

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ‖v‖ = 1

theorem unit_vector_norm_equal {a b : EuclideanSpace ℝ (Fin 2)}
  (ha : is_unit_vector a) (hb : is_unit_vector b) : ‖a‖ = ‖b‖ :=
by 
  sorry

end NUMINAMATH_GPT_unit_vector_norm_equal_l2113_211364


namespace NUMINAMATH_GPT_inequality_solution_l2113_211355

noncomputable def solve_inequality (x : ℝ) : Prop :=
  ((x - 3) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution : {x : ℝ | solve_inequality x} = 
  {x : ℝ | x < 2} ∪ {x : ℝ | 3 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | x > 7} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2113_211355


namespace NUMINAMATH_GPT_find_certain_number_l2113_211380

theorem find_certain_number (x : ℝ) (h : x + 12.952 - 47.95000000000027 = 3854.002) : x = 3889.000 :=
sorry

end NUMINAMATH_GPT_find_certain_number_l2113_211380


namespace NUMINAMATH_GPT_steven_amanda_hike_difference_l2113_211371

variable (Camila_hikes : ℕ)
variable (Camila_weeks : ℕ)
variable (hikes_per_week : ℕ)

def Amanda_hikes (Camila_hikes : ℕ) : ℕ := 8 * Camila_hikes

def Steven_hikes (Camila_hikes : ℕ)(Camila_weeks : ℕ)(hikes_per_week : ℕ) : ℕ :=
  Camila_hikes + Camila_weeks * hikes_per_week

theorem steven_amanda_hike_difference
  (hCamila : Camila_hikes = 7)
  (hWeeks : Camila_weeks = 16)
  (hHikesPerWeek : hikes_per_week = 4) :
  Steven_hikes Camila_hikes Camila_weeks hikes_per_week - Amanda_hikes Camila_hikes = 15 := by
  sorry

end NUMINAMATH_GPT_steven_amanda_hike_difference_l2113_211371


namespace NUMINAMATH_GPT_find_x_l2113_211383

def magic_constant (a b c d e f g h i : ℤ) : Prop :=
  a + b + c = d + e + f ∧ d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧ b + e + h = c + f + i ∧
  a + e + i = c + e + g

def given_magic_square (x : ℤ) : Prop :=
  magic_constant (4017) (2012) (0) 
                 (4015) (x - 2003) (11) 
                 (2014) (9) (x)

theorem find_x (x : ℤ) (h : given_magic_square x) : x = 4003 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l2113_211383


namespace NUMINAMATH_GPT_geometricSeqMinimumValue_l2113_211304

noncomputable def isMinimumValue (a : ℕ → ℝ) (n m : ℕ) (value : ℝ) : Prop :=
  ∀ b : ℝ, (1 / a n + b / a m) ≥ value

theorem geometricSeqMinimumValue {a : ℕ → ℝ}
  (h1 : ∀ n, a n > 0)
  (h2 : a 7 = (Real.sqrt 2) / 2)
  (h3 : ∀ n, ∀ m, a n * a m = a (n + m)) :
  isMinimumValue a 3 11 4 :=
sorry

end NUMINAMATH_GPT_geometricSeqMinimumValue_l2113_211304


namespace NUMINAMATH_GPT_problem_statement_l2113_211372

-- Define the repeating decimal 0.000272727... as x
noncomputable def repeatingDecimal : ℚ := 3 / 11000

-- Define the given condition for the question
def decimalRepeatsIndefinitely : Prop := 
  repeatingDecimal = 0.0002727272727272727  -- Representation for repeating decimal

-- Definitions of large powers of 10
def ten_pow_5 := 10^5
def ten_pow_3 := 10^3

-- The problem statement
theorem problem_statement : decimalRepeatsIndefinitely →
  (ten_pow_5 - ten_pow_3) * repeatingDecimal = 27 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2113_211372


namespace NUMINAMATH_GPT_connie_marbles_l2113_211339

-- Define the initial number of marbles that Connie had
def initial_marbles : ℝ := 73.5

-- Define the number of marbles that Connie gave away
def marbles_given : ℝ := 70.3

-- Define the expected number of marbles remaining
def marbles_remaining : ℝ := 3.2

-- State the theorem: prove that initial_marbles - marbles_given = marbles_remaining
theorem connie_marbles :
  initial_marbles - marbles_given = marbles_remaining :=
sorry

end NUMINAMATH_GPT_connie_marbles_l2113_211339


namespace NUMINAMATH_GPT_find_r_and_k_l2113_211379

-- Define the line equation
def line (x : ℝ) : ℝ := 5 * x - 7

-- Define the parameterization
def param (t r k : ℝ) : ℝ × ℝ := 
  (r + 3 * t, 2 + k * t)

-- Theorem stating that (r, k) = (9/5, 15) satisfies the given conditions
theorem find_r_and_k 
  (r k : ℝ)
  (H1 : param 0 r k = (r, 2))
  (H2 : line r = 2)
  (H3 : param 1 r k = (r + 3, 2 + k))
  (H4 : line (r + 3) = 2 + k)
  : (r, k) = (9/5, 15) :=
sorry

end NUMINAMATH_GPT_find_r_and_k_l2113_211379


namespace NUMINAMATH_GPT_rhombus_area_l2113_211365

theorem rhombus_area
  (d1 d2 : ℝ)
  (hd1 : d1 = 14)
  (hd2 : d2 = 20) :
  (d1 * d2) / 2 = 140 := by
  -- Problem: Given diagonals of length 14 cm and 20 cm,
  -- prove that the area of the rhombus is 140 square centimeters.
  sorry

end NUMINAMATH_GPT_rhombus_area_l2113_211365


namespace NUMINAMATH_GPT_f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l2113_211323

noncomputable def f (x : ℝ) : ℝ := x + 4 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 2^x + a

theorem f_monotone_on_0_to_2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 ≤ 2 → f x1 > f x2 :=
sorry

theorem find_range_a_part2 : (∀ x1 : ℝ, x1 ∈ (Set.Icc (1/2) 1) → 
  ∃ x2 : ℝ, x2 ∈ (Set.Icc 2 3) ∧ f x1 ≥ g x2 a) → a ≤ 1 :=
sorry

theorem find_range_a_part3 : (∃ x : ℝ, x ∈ (Set.Icc 0 2) ∧ f x ≤ g x a) → a ≥ 0 :=
sorry

end NUMINAMATH_GPT_f_monotone_on_0_to_2_find_range_a_part2_find_range_a_part3_l2113_211323


namespace NUMINAMATH_GPT_felix_chopped_at_least_91_trees_l2113_211338

def cost_to_sharpen := 5
def total_spent := 35
def trees_per_sharpen := 13

theorem felix_chopped_at_least_91_trees :
  (total_spent / cost_to_sharpen) * trees_per_sharpen = 91 := by
  sorry

end NUMINAMATH_GPT_felix_chopped_at_least_91_trees_l2113_211338


namespace NUMINAMATH_GPT_binomial_9_3_l2113_211311

theorem binomial_9_3 : (Nat.choose 9 3) = 84 := by
  sorry

end NUMINAMATH_GPT_binomial_9_3_l2113_211311


namespace NUMINAMATH_GPT_trader_excess_donations_l2113_211373

-- Define the conditions
def profit : ℤ := 1200
def allocation_percentage : ℤ := 60
def family_donation : ℤ := 250
def friends_donation : ℤ := (20 * family_donation) / 100 + family_donation
def total_family_friends_donation : ℤ := family_donation + friends_donation
def local_association_donation : ℤ := 15 * total_family_friends_donation / 10
def total_donations : ℤ := family_donation + friends_donation + local_association_donation
def allocated_amount : ℤ := allocation_percentage * profit / 100

-- Theorem statement (Question)
theorem trader_excess_donations : total_donations - allocated_amount = 655 :=
by
  sorry

end NUMINAMATH_GPT_trader_excess_donations_l2113_211373


namespace NUMINAMATH_GPT_maximum_marks_l2113_211395

theorem maximum_marks (M : ℝ) (P : ℝ) 
  (h1 : P = 0.45 * M) -- 45% of the maximum marks to pass
  (h2 : P = 210 + 40) -- Pradeep's marks plus failed marks

  : M = 556 := 
sorry

end NUMINAMATH_GPT_maximum_marks_l2113_211395


namespace NUMINAMATH_GPT_geometric_sequence_relation_l2113_211315

variables {a : ℕ → ℝ} {q : ℝ}
variables {m n p : ℕ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def are_in_geometric_sequence (a : ℕ → ℝ) (m n p : ℕ) : Prop :=
  a n ^ 2 = a m * a p

-- Theorem
theorem geometric_sequence_relation (h_geom : is_geometric_sequence a q) (h_order : are_in_geometric_sequence a m n p) (hq_ne_one : q ≠ 1) :
  2 * n = m + p :=
sorry

end NUMINAMATH_GPT_geometric_sequence_relation_l2113_211315


namespace NUMINAMATH_GPT_neg_square_positive_l2113_211313

theorem neg_square_positive :
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := sorry

end NUMINAMATH_GPT_neg_square_positive_l2113_211313


namespace NUMINAMATH_GPT_problem_statement_l2113_211303

theorem problem_statement (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) = x * f y + y * f x) →
  (∀ x : ℝ, x > 1 → f x < 0) →

  -- Conclusion 1: f(1) = 0, f(-1) = 0
  f 1 = 0 ∧ f (-1) = 0 ∧

  -- Conclusion 2: f(x) is an odd function: f(-x) = -f(x)
  (∀ x : ℝ, f (-x) = -f x) ∧

  -- Conclusion 3: f(x) is decreasing on (1, +∞)
  (∀ x1 x2 : ℝ, x1 > 1 → x2 > 1 → x1 < x2 → f x1 < f x2) := sorry

end NUMINAMATH_GPT_problem_statement_l2113_211303


namespace NUMINAMATH_GPT_solve_for_x_l2113_211300

theorem solve_for_x : ∃ x : ℝ, 3 * x - 48.2 = 0.25 * (4 * x + 56.8) → x = 31.2 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l2113_211300


namespace NUMINAMATH_GPT_intersection_points_range_l2113_211335

theorem intersection_points_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ a = x₁^3 - 3 * x₁ ∧
  a = x₂^3 - 3 * x₂ ∧ a = x₃^3 - 3 * x₃) ↔ (-2 < a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_intersection_points_range_l2113_211335


namespace NUMINAMATH_GPT_inequalities_hold_l2113_211310

theorem inequalities_hold 
  (x y z a b c : ℕ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)   -- Given that x, y, z are positive integers
  (ha : a > 0) (hb : b > 0) (hc : c > 0)   -- Given that a, b, c are positive integers
  (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 ≤ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ∧ 
  x^3 + y^3 + z^3 ≤ a^3 + b^3 + c^3 ∧ 
  x^2 * y * z + y^2 * z * x + z^2 * x * y ≤ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by
  sorry

end NUMINAMATH_GPT_inequalities_hold_l2113_211310
