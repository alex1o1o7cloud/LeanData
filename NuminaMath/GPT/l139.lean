import Mathlib

namespace problem_l139_13962

variables (y S : ℝ)

theorem problem (h : 5 * (2 * y + 3 * Real.sqrt 3) = S) : 10 * (4 * y + 6 * Real.sqrt 3) = 4 * S :=
sorry

end problem_l139_13962


namespace mean_proportional_l139_13963

theorem mean_proportional (a c x : ℝ) (ha : a = 9) (hc : c = 4) (hx : x^2 = a * c) : x = 6 := by
  sorry

end mean_proportional_l139_13963


namespace find_correct_quotient_l139_13996

theorem find_correct_quotient 
  (Q : ℕ)
  (D : ℕ)
  (h1 : D = 21 * Q)
  (h2 : D = 12 * 35) : 
  Q = 20 := 
by 
  sorry

end find_correct_quotient_l139_13996


namespace smallest_blocks_needed_for_wall_l139_13973

noncomputable def smallest_number_of_blocks (wall_length : ℕ) (wall_height : ℕ) (block_length1 : ℕ) (block_length2 : ℕ) (block_length3 : ℝ) : ℕ :=
  let blocks_per_odd_row := wall_length / block_length1
  let blocks_per_even_row := wall_length / block_length1 - 1 + 2
  let odd_rows := wall_height / 2 + 1
  let even_rows := wall_height / 2
  odd_rows * blocks_per_odd_row + even_rows * blocks_per_even_row

theorem smallest_blocks_needed_for_wall :
  smallest_number_of_blocks 120 7 2 1 1.5 = 423 :=
by
  sorry

end smallest_blocks_needed_for_wall_l139_13973


namespace tank_fewer_eggs_in_second_round_l139_13955

variables (T E_total T_r2_diff : ℕ)

theorem tank_fewer_eggs_in_second_round
  (h1 : E_total = 400)
  (h2 : E_total = (T + (T - 10)) + (30 + 60))
  (h3 : T_r2_diff = T - 30) :
  T_r2_diff = 130 := by
    sorry

end tank_fewer_eggs_in_second_round_l139_13955


namespace tangent_slope_l139_13980

noncomputable def f (x : ℝ) : ℝ := x - 1 + 1 / Real.exp x

noncomputable def f' (x : ℝ) : ℝ := 1 - 1 / Real.exp x

theorem tangent_slope (k : ℝ) (x₀ : ℝ) (y₀ : ℝ) 
  (h_tangent_point: (x₀ = -1) ∧ (y₀ = x₀ - 1 + 1 / Real.exp x₀))
  (h_tangent_line : ∀ x, y₀ = f x₀ + f' x₀ * (x - x₀)) :
  k = 1 - Real.exp 1 := 
sorry

end tangent_slope_l139_13980


namespace rate_of_current_l139_13928

-- Definitions of the conditions
def downstream_speed : ℝ := 30  -- in kmph
def upstream_speed : ℝ := 10    -- in kmph
def still_water_rate : ℝ := 20  -- in kmph

-- Calculating the rate of the current
def current_rate : ℝ := downstream_speed - still_water_rate

-- Proof statement
theorem rate_of_current :
  current_rate = 10 :=
by
  sorry

end rate_of_current_l139_13928


namespace trig_identity_l139_13905

theorem trig_identity (x : ℝ) (h0 : -3 * Real.pi / 2 < x) (h1 : x < -Real.pi) (h2 : Real.tan x = -3) :
  Real.sin x * Real.cos x = -3 / 10 :=
sorry

end trig_identity_l139_13905


namespace max_min_distance_inequality_l139_13910

theorem max_min_distance_inequality (n : ℕ) (D d : ℝ) (h1 : d > 0) 
    (exists_points : ∃ (points : Fin n → ℝ × ℝ), 
      (∀ i j : Fin n, i ≠ j → dist (points i) (points j) ≥ d) 
      ∧ (∀ i j : Fin n, dist (points i) (points j) ≤ D)) : 
    D / d > (Real.sqrt (n * Real.pi)) / 2 - 1 := 
  sorry

end max_min_distance_inequality_l139_13910


namespace age_difference_l139_13911

theorem age_difference (x y : ℕ) (h1 : 3 * x + 4 * x = 42) (h2 : 18 - y = (24 - y) / 2) : 
  y = 12 :=
  sorry

end age_difference_l139_13911


namespace range_of_a_l139_13978

open Set

theorem range_of_a (a : ℝ) (h1 : (∃ x, a^x > 1 ∧ x < 0) ∨ (∀ x, ax^2 - x + a ≥ 0))
  (h2 : ¬((∃ x, a^x > 1 ∧ x < 0) ∧ (∀ x, ax^2 - x + a ≥ 0))) :
  a ∈ (Ioo 0 (1/2)) ∪ (Ici 1) :=
by {
  sorry
}

end range_of_a_l139_13978


namespace A_salary_l139_13919

theorem A_salary (x y : ℝ) (h1 : x + y = 7000) (h2 : 0.05 * x = 0.15 * y) : x = 5250 :=
by
  sorry

end A_salary_l139_13919


namespace cone_base_radius_l139_13925

/-- A hemisphere of radius 3 rests on the base of a circular cone and is tangent to the cone's lateral surface along a circle. 
Given that the height of the cone is 9, prove that the base radius of the cone is 10.5. -/
theorem cone_base_radius
  (r_h : ℝ) (h : ℝ) (r : ℝ) 
  (hemisphere_tangent_cone : r_h = 3)
  (cone_height : h = 9)
  (tangent_circle_height : r - r_h = 3) :
  r = 10.5 := by
  sorry

end cone_base_radius_l139_13925


namespace perpendicular_condition_l139_13900

def line := Type
def plane := Type

variables {α : plane} {a b : line}

-- Conditions: define parallelism and perpendicularity
def parallel (a : line) (α : plane) : Prop := sorry
def perpendicular (a : line) (α : plane) : Prop := sorry
def perpendicular_lines (a b : line) : Prop := sorry

-- Given Hypotheses
variable (h1 : parallel a α)
variable (h2 : perpendicular b α)

-- Statement to prove
theorem perpendicular_condition (h1 : parallel a α) (h2 : perpendicular b α) :
  (perpendicular_lines b a) ∧ (¬ (perpendicular_lines b a → perpendicular b α)) := 
sorry

end perpendicular_condition_l139_13900


namespace circle_Γ_contains_exactly_one_l139_13934

-- Condition definitions
variables (z1 z2 : ℂ) (Γ : ℂ → ℂ → Prop)
variable (hz1z2 : z1 * z2 = 1)
variable (hΓ_passes : Γ (-1) 1)
variable (hΓ_not_passes : ¬Γ z1 z2)

-- Math proof problem
theorem circle_Γ_contains_exactly_one (hz1z2 : z1 * z2 = 1)
    (hΓ_passes : Γ (-1) 1) (hΓ_not_passes : ¬Γ z1 z2) : 
  (Γ 0 z1 ↔ ¬Γ 0 z2) ∨ (Γ 0 z2 ↔ ¬Γ 0 z1) :=
sorry

end circle_Γ_contains_exactly_one_l139_13934


namespace remainder_when_3y_divided_by_9_l139_13981

theorem remainder_when_3y_divided_by_9 (y : ℕ) (k : ℕ) (hy : y = 9 * k + 5) : (3 * y) % 9 = 6 :=
sorry

end remainder_when_3y_divided_by_9_l139_13981


namespace f_1996x_l139_13909

noncomputable def f : ℝ → ℝ := sorry

axiom f_equation (x y : ℝ) : f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

theorem f_1996x (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end f_1996x_l139_13909


namespace no_solution_pos_integers_l139_13951

theorem no_solution_pos_integers (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a + b + c + d - 3 ≠ a * b + c * d := 
by
  sorry

end no_solution_pos_integers_l139_13951


namespace sum_of_fractions_l139_13937

theorem sum_of_fractions : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) = 3 / 8) :=
by sorry

end sum_of_fractions_l139_13937


namespace second_number_more_than_first_l139_13916

-- Definitions of A and B based on the given ratio
def A : ℚ := 7 / 56
def B : ℚ := 8 / 56

-- Proof statement
theorem second_number_more_than_first : ((B - A) / A) * 100 = 100 / 7 :=
by
  -- skipped the proof
  sorry

end second_number_more_than_first_l139_13916


namespace repeating_decimal_to_fraction_l139_13935

theorem repeating_decimal_to_fraction : (0.2727272727 : ℝ) = 3 / 11 := 
sorry

end repeating_decimal_to_fraction_l139_13935


namespace minimum_value_of_x_squared_l139_13931

theorem minimum_value_of_x_squared : ∃ x : ℝ, x = 0 ∧ ∀ y : ℝ, y = x^2 → y ≥ 0 :=
by
  sorry

end minimum_value_of_x_squared_l139_13931


namespace find_value_m_sq_plus_2m_plus_n_l139_13944

noncomputable def m_n_roots (x : ℝ) : Prop := x^2 + x - 1001 = 0

theorem find_value_m_sq_plus_2m_plus_n
  (m n : ℝ)
  (hm : m_n_roots m)
  (hn : m_n_roots n)
  (h_sum : m + n = -1)
  (h_prod : m * n = -1001) :
  m^2 + 2 * m + n = 1000 :=
sorry

end find_value_m_sq_plus_2m_plus_n_l139_13944


namespace average_rate_l139_13915

theorem average_rate (distance_run distance_swim : ℝ) (rate_run rate_swim : ℝ) 
  (h1 : distance_run = 2) (h2 : distance_swim = 2) (h3 : rate_run = 10) (h4 : rate_swim = 5) : 
  (distance_run + distance_swim) / ((distance_run / rate_run) * 60 + (distance_swim / rate_swim) * 60) = 0.1111 :=
by
  sorry

end average_rate_l139_13915


namespace tan_three_halves_pi_sub_alpha_l139_13932

theorem tan_three_halves_pi_sub_alpha (α : ℝ) (h : Real.cos (π - α) = -3/5) :
    Real.tan (3 * π / 2 - α) = 3/4 ∨ Real.tan (3 * π / 2 - α) = -3/4 := by
  sorry

end tan_three_halves_pi_sub_alpha_l139_13932


namespace hyperbola_standard_equation_l139_13921

open Real

noncomputable def distance_from_center_to_focus (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

theorem hyperbola_standard_equation (a b c : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : b = sqrt 3 * c)
  (h4 : a + c = 3 * sqrt 3) :
  ∃ h : a^2 = 12 ∧ b = 3, y^2 / 12 - x^2 / 9 = 1 :=
sorry

end hyperbola_standard_equation_l139_13921


namespace coefficient_of_x3_in_expansion_l139_13949

theorem coefficient_of_x3_in_expansion :
  (∀ (x : ℝ), (Polynomial.coeff ((Polynomial.C x - 1)^5) 3) = 10) :=
by
  sorry

end coefficient_of_x3_in_expansion_l139_13949


namespace circle_rolling_start_point_l139_13946

theorem circle_rolling_start_point (x : ℝ) (h1 : ∃ x, (x + 2 * Real.pi = -1) ∨ (x - 2 * Real.pi = -1)) :
  x = -1 - 2 * Real.pi ∨ x = -1 + 2 * Real.pi :=
by
  sorry

end circle_rolling_start_point_l139_13946


namespace plot_length_60_l139_13947

/-- The length of a rectangular plot is 20 meters more than its breadth. If the cost of fencing the plot at Rs. 26.50 per meter is Rs. 5300, then the length of the plot in meters is 60. -/
theorem plot_length_60 (b l : ℝ) (h1 : l = b + 20) (h2 : 2 * (l + b) * 26.5 = 5300) : l = 60 :=
by
  sorry

end plot_length_60_l139_13947


namespace fresh_fruit_sold_l139_13923

variable (total_fruit frozen_fruit : ℕ)

theorem fresh_fruit_sold (h1 : total_fruit = 9792) (h2 : frozen_fruit = 3513) : 
  total_fruit - frozen_fruit = 6279 :=
by sorry

end fresh_fruit_sold_l139_13923


namespace A_is_9_years_older_than_B_l139_13992

-- Define the conditions
variables (A_years B_years : ℕ)

def given_conditions : Prop :=
  B_years = 39 ∧ A_years + 10 = 2 * (B_years - 10)

-- Theorem to prove the correct answer
theorem A_is_9_years_older_than_B (h : given_conditions A_years B_years) : A_years - B_years = 9 :=
by
  sorry

end A_is_9_years_older_than_B_l139_13992


namespace jacket_price_equation_l139_13960

theorem jacket_price_equation (x : ℝ) (h : 0.8 * (1 + 0.5) * x - x = 28) : 0.8 * (1 + 0.5) * x = x + 28 :=
by sorry

end jacket_price_equation_l139_13960


namespace rhombus_diagonal_length_l139_13982

theorem rhombus_diagonal_length 
  (side_length : ℕ) (shorter_diagonal : ℕ) (longer_diagonal : ℕ)
  (h1 : side_length = 34) (h2 : shorter_diagonal = 32) :
  longer_diagonal = 60 :=
sorry

end rhombus_diagonal_length_l139_13982


namespace collinear_R_S_T_l139_13971

theorem collinear_R_S_T
    (circle : Type)
    (P : circle)
    (A B C D : circle)
    (E F : Type → Type)
    (angle : ∀ (x y z : circle), ℝ)   -- Placeholder for angles
    (quadrilateral_inscribed_in_circle : ∀ (A B C D : circle), Prop)   -- Placeholder for the condition of the quadrilateral
    (extensions_intersect : ∀ (A B C D : circle) (E F : Type → Type), Prop)   -- Placeholder for extensions intersections
    (diagonals_intersect_at : ∀ (A C B D T : circle), Prop)   -- Placeholder for diagonals intersections
    (P_on_circle : ∀ (P : circle), Prop)        -- Point P is on the circle
    (PE_PF_intersect_again : ∀ (P R S : circle) (E F : Type → Type), Prop)   -- PE and PF intersect the circle again at R and S
    (R S T : circle) :
    quadrilateral_inscribed_in_circle A B C D →
    extensions_intersect A B C D E F →
    P_on_circle P →
    PE_PF_intersect_again P R S E F →
    diagonals_intersect_at A C B D T →
    ∃ collinearity : ∀ (R S T : circle), Prop,
    collinearity R S T := 
by
  intro h1 h2 h3 h4 h5
  sorry

end collinear_R_S_T_l139_13971


namespace distance_to_SFL_l139_13950

def distance_per_hour : ℕ := 27
def hours_travelled : ℕ := 3

theorem distance_to_SFL :
  (distance_per_hour * hours_travelled) = 81 := 
by
  sorry

end distance_to_SFL_l139_13950


namespace calculate_bus_stoppage_time_l139_13901

variable (speed_excl_stoppages speed_incl_stoppages distance_excl_stoppages distance_incl_stoppages distance_diff time_lost_stoppages : ℝ)

def bus_stoppage_time
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  Prop :=
  speed_excl_stoppages = 32 ∧
  speed_incl_stoppages = 16 ∧
  time_stopped = 30

theorem calculate_bus_stoppage_time 
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  bus_stoppage_time speed_excl_stoppages speed_incl_stoppages time_stopped :=
by
  have h1 : speed_excl_stoppages = 32 := by
    sorry
  have h2 : speed_incl_stoppages = 16 := by
    sorry
  have h3 : time_stopped = 30 := by
    sorry
  exact ⟨h1, h2, h3⟩

end calculate_bus_stoppage_time_l139_13901


namespace negate_proposition_l139_13956

theorem negate_proposition : (∀ x : ℝ, x^3 - x^2 + 1 ≤ 1) ↔ ¬ (∃ x : ℝ, x^3 - x^2 + 1 > 1) :=
by
  sorry

end negate_proposition_l139_13956


namespace solution_set_of_inequality_l139_13908

theorem solution_set_of_inequality (a : ℝ) :
  ¬ (∀ x : ℝ, ¬ (a * (x - a) * (a * x + a) ≥ 0)) ∧
  ¬ (∀ x : ℝ, (a - x ≤ 0 ∧ x - (-1) ≤ 0 → a * (x - a) * (a * x + a) ≥ 0)) :=
by
  sorry

end solution_set_of_inequality_l139_13908


namespace decimal_equiv_half_squared_l139_13972

theorem decimal_equiv_half_squared :
  ((1 / 2 : ℝ) ^ 2) = 0.25 := by
  sorry

end decimal_equiv_half_squared_l139_13972


namespace range_of_a_minus_b_l139_13943

theorem range_of_a_minus_b {a b : ℝ} (h1 : -2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) : 
  -4 < a - b ∧ a - b < 2 := 
by
  sorry

end range_of_a_minus_b_l139_13943


namespace abs_inequality_condition_l139_13952

theorem abs_inequality_condition (a : ℝ) : 
  (a < 2) ↔ ∀ x : ℝ, |x - 2| + |x| > a :=
sorry

end abs_inequality_condition_l139_13952


namespace PQRS_value_l139_13941

theorem PQRS_value
  (P Q R S : ℝ)
  (hP : 0 < P)
  (hQ : 0 < Q)
  (hR : 0 < R)
  (hS : 0 < S)
  (h1 : Real.log (P * Q) / Real.log 10 + Real.log (P * S) / Real.log 10 = 2)
  (h2 : Real.log (Q * S) / Real.log 10 + Real.log (Q * R) / Real.log 10 = 3)
  (h3 : Real.log (R * P) / Real.log 10 + Real.log (R * S) / Real.log 10 = 5) :
  P * Q * R * S = 100000 := 
sorry

end PQRS_value_l139_13941


namespace probability_quadrant_l139_13986

theorem probability_quadrant
    (r : ℝ) (x y : ℝ)
    (h : x^2 + y^2 ≤ r^2) :
    (∃ p : ℝ, p = (1 : ℚ)/4) :=
by
  sorry

end probability_quadrant_l139_13986


namespace regular_train_pass_time_l139_13918

-- Define the lengths of the trains
def high_speed_train_length : ℕ := 400
def regular_train_length : ℕ := 600

-- Define the observation time for the passenger on the high-speed train
def observation_time : ℕ := 3

-- Define the problem to find the time x for the regular train passenger
theorem regular_train_pass_time :
  ∃ (x : ℕ), (regular_train_length / observation_time) * x = high_speed_train_length :=
by 
  sorry

end regular_train_pass_time_l139_13918


namespace five_minus_a_l139_13927

theorem five_minus_a (a b : ℚ) (h1 : 5 + a = 3 - b) (h2 : 3 + b = 8 + a) : 5 - a = 17/2 :=
by
  sorry

end five_minus_a_l139_13927


namespace octagon_mass_is_19kg_l139_13993

-- Define the parameters given in the problem
def side_length_square_sheet := 1  -- side length in meters
def thickness_sheet := 0.3  -- thickness in cm (3 mm)
def density_steel := 7.8  -- density in g/cm³

-- Given the geometric transformations and constants, prove the mass of the octagon
theorem octagon_mass_is_19kg :
  ∃ mass : ℝ, (mass = 19) :=
by
  -- Placeholder for the proof.
  -- The detailed steps would include geometrical transformations and volume calculations,
  -- which have been rigorously defined in the problem and derived in the solution.
  sorry

end octagon_mass_is_19kg_l139_13993


namespace M_is_correct_ab_property_l139_13930

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 1|
def M : Set ℝ := {x | f x < 4}

theorem M_is_correct : M = {x | -2 < x ∧ x < 2} :=
sorry

theorem ab_property (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 2 * |a + b| < |4 + a * b| :=
sorry

end M_is_correct_ab_property_l139_13930


namespace ratio_of_polynomials_eq_962_l139_13959

open Real

theorem ratio_of_polynomials_eq_962 :
  (10^4 + 400) * (26^4 + 400) * (42^4 + 400) * (58^4 + 400) /
  ((2^4 + 400) * (18^4 + 400) * (34^4 + 400) * (50^4 + 400)) = 962 := 
sorry

end ratio_of_polynomials_eq_962_l139_13959


namespace trigonometric_identity_l139_13990

theorem trigonometric_identity 
  (θ : ℝ) 
  (h : Real.tan θ = 2) : 
  (Real.cos θ - Real.sin θ) / (Real.cos θ + Real.sin θ) = -1 / 3 :=
by
  sorry

end trigonometric_identity_l139_13990


namespace pool_fill_time_l139_13988

theorem pool_fill_time:
  ∀ (A B C D : ℚ),
  (A + B - D = 1 / 6) →
  (A + C - D = 1 / 5) →
  (B + C - D = 1 / 4) →
  (A + B + C - D = 1 / 3) →
  (1 / (A + B + C) = 60 / 23) :=
by intros A B C D h1 h2 h3 h4; sorry

end pool_fill_time_l139_13988


namespace find_sum_f_neg1_f_3_l139_13939

noncomputable def f : ℝ → ℝ := sorry

-- condition: odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f x

-- condition: symmetry around x=1
def symmetric_around_one (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (1 - x) = f (1 + x)

-- condition: specific value at x=1
def value_at_one (f : ℝ → ℝ) : Prop := f 1 = 2

-- Theorem to prove
theorem find_sum_f_neg1_f_3 (h1 : odd_function f) (h2 : symmetric_around_one f) (h3 : value_at_one f) : f (-1) + f 3 = -4 := by
  sorry

end find_sum_f_neg1_f_3_l139_13939


namespace linear_function_in_quadrants_l139_13920

section LinearFunctionQuadrants

variable (m : ℝ)

def passesThroughQuadrants (m : ℝ) : Prop :=
  (m + 1 > 0) ∧ (m - 1 > 0)

theorem linear_function_in_quadrants (h : passesThroughQuadrants m) : m > 1 :=
by sorry

end LinearFunctionQuadrants

end linear_function_in_quadrants_l139_13920


namespace tangent_points_are_on_locus_l139_13970

noncomputable def tangent_points_locus (d : ℝ) : Prop :=
∀ (x y : ℝ), 
((x ≠ 0 ∨ y ≠ 0) ∧ (x-d ≠ 0)) ∧ (y = x) 
→ (y^2 - x*y + d*(x + y) = 0)

theorem tangent_points_are_on_locus (d : ℝ) : 
  tangent_points_locus d :=
by sorry

end tangent_points_are_on_locus_l139_13970


namespace machine_A_sprockets_per_hour_l139_13964

-- Definitions based on the problem conditions
def MachineP_time (A : ℝ) (T : ℝ) : ℝ := T + 10
def MachineQ_rate (A : ℝ) : ℝ := 1.1 * A
def MachineP_sprockets (A : ℝ) (T : ℝ) : ℝ := A * (T + 10)
def MachineQ_sprockets (A : ℝ) (T : ℝ) : ℝ := 1.1 * A * T

-- Lean proof statement to prove that Machine A produces 8 sprockets per hour
theorem machine_A_sprockets_per_hour :
  ∀ A T : ℝ, 
  880 = MachineP_sprockets A T ∧
  880 = MachineQ_sprockets A T →
  A = 8 :=
by
  intros A T h
  have h1 : 880 = MachineP_sprockets A T := h.left
  have h2 : 880 = MachineQ_sprockets A T := h.right
  sorry

end machine_A_sprockets_per_hour_l139_13964


namespace curve_symmetric_origin_l139_13999

theorem curve_symmetric_origin (x y : ℝ) (h : 3*x^2 - 8*x*y + 2*y^2 = 0) :
  3*(-x)^2 - 8*(-x)*(-y) + 2*(-y)^2 = 3*x^2 - 8*x*y + 2*y^2 :=
sorry

end curve_symmetric_origin_l139_13999


namespace susan_ate_candies_l139_13954

theorem susan_ate_candies (candies_tuesday candies_thursday candies_friday candies_left : ℕ) 
  (h_tuesday : candies_tuesday = 3) 
  (h_thursday : candies_thursday = 5) 
  (h_friday : candies_friday = 2) 
  (h_left : candies_left = 4) : candies_tuesday + candies_thursday + candies_friday - candies_left = 6 := by
  sorry

end susan_ate_candies_l139_13954


namespace total_number_of_digits_l139_13966

theorem total_number_of_digits (n S S₅ S₃ : ℕ) (h1 : S = 20 * n) (h2 : S₅ = 5 * 12) (h3 : S₃ = 3 * 33) : n = 8 :=
by
  sorry

end total_number_of_digits_l139_13966


namespace inequality_range_a_l139_13907

open Real

theorem inequality_range_a (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 :=
sorry

end inequality_range_a_l139_13907


namespace problem_solution_l139_13997

-- Definitions based on conditions given in the problem statement
def validExpression (n : ℕ) : ℕ := 
  sorry -- Placeholder for function defining valid expressions

def T (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else validExpression n

def R (n : ℕ) : ℕ := T n % 4

def computeSum (k : ℕ) : ℕ := 
  (List.range k).map R |>.sum

-- Lean theorem statement to be proven
theorem problem_solution : 
  computeSum 1000001 = 320 := 
sorry

end problem_solution_l139_13997


namespace infinite_a_exists_l139_13968

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ), ∀ (k : ℕ), ∃ (a : ℕ), n^6 + 3 * a = (n^2 + 3 * k)^3 := 
sorry

end infinite_a_exists_l139_13968


namespace find_constants_monotonicity_l139_13974

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_constants (a b c : ℝ) 
  (h1 : f' (-2/3) a b = 0)
  (h2 : f' 1 a b = 0) :
  a = -1/2 ∧ b = -2 :=
by sorry

theorem monotonicity (a b c : ℝ)
  (h1 : a = -1/2) 
  (h2 : b = -2) : 
  (∀ x : ℝ, x < -2/3 → f' x a b > 0) ∧ 
  (∀ x : ℝ, -2/3 < x ∧ x < 1 → f' x a b < 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a b > 0) :=
by sorry

end find_constants_monotonicity_l139_13974


namespace hot_dogs_left_over_l139_13929

theorem hot_dogs_left_over : 25197629 % 6 = 5 := 
sorry

end hot_dogs_left_over_l139_13929


namespace club_members_neither_subject_l139_13912

theorem club_members_neither_subject (total members_cs members_bio members_both : ℕ)
  (h_total : total = 150)
  (h_cs : members_cs = 80)
  (h_bio : members_bio = 50)
  (h_both : members_both = 15) :
  total - ((members_cs - members_both) + (members_bio - members_both) + members_both) = 35 := by
  sorry

end club_members_neither_subject_l139_13912


namespace friend_time_to_read_book_l139_13984

-- Define the conditions and variables
def my_reading_time : ℕ := 240 -- 4 hours in minutes
def speed_ratio : ℕ := 2 -- I read at half the speed of my friend

-- Define the variable for my friend's reading time which we need to find
def friend_reading_time : ℕ := my_reading_time / speed_ratio

-- The theorem statement that given the conditions, the friend's reading time is 120 minutes
theorem friend_time_to_read_book : friend_reading_time = 120 := sorry

end friend_time_to_read_book_l139_13984


namespace finished_in_6th_l139_13979

variable (p : ℕ → Prop)
variable (Sana Max Omar Jonah Leila : ℕ)

-- Conditions
def condition1 : Prop := Omar = Jonah - 7
def condition2 : Prop := Sana = Max - 2
def condition3 : Prop := Leila = Jonah + 3
def condition4 : Prop := Max = Omar + 1
def condition5 : Prop := Sana = 4

-- Conclusion
theorem finished_in_6th (h1 : condition1 Omar Jonah)
                         (h2 : condition2 Sana Max)
                         (h3 : condition3 Leila Jonah)
                         (h4 : condition4 Max Omar)
                         (h5 : condition5 Sana) :
  Max = 6 := by
  sorry

end finished_in_6th_l139_13979


namespace min_value_fraction_l139_13976

variable (a b : ℝ)

theorem min_value_fraction (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 2 * b = 1) : 
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_fraction_l139_13976


namespace jeans_price_increase_l139_13994

theorem jeans_price_increase (M R C : ℝ) (hM : M = 100) 
  (hR : R = M * 1.4)
  (hC : C = R * 1.1) : 
  (C - M) / M * 100 = 54 :=
by
  sorry

end jeans_price_increase_l139_13994


namespace compute_x_plus_y_l139_13924

theorem compute_x_plus_y :
    ∃ (x y : ℕ), 4 * y = 7 * 84 ∧ 4 * 63 = 7 * x ∧ x + y = 183 :=
by
  sorry

end compute_x_plus_y_l139_13924


namespace tan_seven_pi_over_four_l139_13991

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by
  sorry

end tan_seven_pi_over_four_l139_13991


namespace profits_equal_l139_13945

-- Define the profit variables
variables (profitA profitB profitC profitD : ℝ)

-- The conditions
def storeA_profit : profitA = 1.2 * profitB := sorry
def storeB_profit : profitB = 1.2 * profitC := sorry
def storeD_profit : profitD = profitA * 0.6 := sorry

-- The statement to be proven
theorem profits_equal : profitC = profitD :=
by sorry

end profits_equal_l139_13945


namespace mod_multiplication_result_l139_13904

theorem mod_multiplication_result :
  ∃ n : ℕ, 507 * 873 ≡ n [MOD 77] ∧ 0 ≤ n ∧ n < 77 ∧ n = 15 := by
  sorry

end mod_multiplication_result_l139_13904


namespace Sharmila_hourly_wage_l139_13906

def Sharmila_hours_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 10
  else if day = "Tuesday" ∨ day = "Thursday" then 8
  else 0

def weekly_total_hours : ℕ :=
  Sharmila_hours_per_day "Monday" + Sharmila_hours_per_day "Tuesday" +
  Sharmila_hours_per_day "Wednesday" + Sharmila_hours_per_day "Thursday" +
  Sharmila_hours_per_day "Friday"

def weekly_earnings : ℤ := 460

def hourly_wage : ℚ :=
  weekly_earnings / weekly_total_hours

theorem Sharmila_hourly_wage :
  hourly_wage = (10 : ℚ) :=
by
  -- proof skipped
  sorry

end Sharmila_hourly_wage_l139_13906


namespace isosceles_triangle_largest_angle_l139_13902

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_iso : A = B) (h_A : C = 50) :
  max A (max B (180 - A - B)) = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l139_13902


namespace computer_price_increase_l139_13953

theorem computer_price_increase (d : ℝ) (h : 2 * d = 585) : d * 1.2 = 351 := by
  sorry

end computer_price_increase_l139_13953


namespace angle_difference_proof_l139_13987

-- Define the angles A and B
def angle_A : ℝ := 65
def angle_B : ℝ := 180 - angle_A

-- Define the difference
def angle_difference : ℝ := angle_B - angle_A

theorem angle_difference_proof : angle_difference = 50 :=
by
  -- The proof goes here
  sorry

end angle_difference_proof_l139_13987


namespace martha_black_butterflies_l139_13965

theorem martha_black_butterflies
    (total_butterflies : ℕ)
    (total_blue_butterflies : ℕ)
    (total_yellow_butterflies : ℕ)
    (total_black_butterflies : ℕ)
    (h1 : total_butterflies = 19)
    (h2 : total_blue_butterflies = 6)
    (h3 : total_blue_butterflies = 2 * total_yellow_butterflies)
    (h4 : total_black_butterflies = total_butterflies - (total_blue_butterflies + total_yellow_butterflies))
    : total_black_butterflies = 10 :=
  sorry

end martha_black_butterflies_l139_13965


namespace first_player_wins_l139_13913

-- Define the polynomial with placeholders
def P (X : ℤ) (a3 a2 a1 a0 : ℤ) : ℤ :=
  X^4 + a3 * X^3 + a2 * X^2 + a1 * X + a0

-- The statement that the first player can always win
theorem first_player_wins :
  ∀ (a3 a2 a1 a0 : ℤ),
    (a0 ≠ 0) → (a1 ≠ 0) → (a2 ≠ 0) → (a3 ≠ 0) →
    ∃ (strategy : ℕ → ℤ),
      (∀ n, strategy n ≠ 0) ∧
      ¬ ∃ (x y : ℤ), x ≠ y ∧ P x (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 ∧ P y (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 :=
by
  sorry

end first_player_wins_l139_13913


namespace milo_eggs_weight_l139_13989

def weight_of_one_egg : ℚ := 1/16
def eggs_per_dozen : ℕ := 12
def dozens_needed : ℕ := 8

theorem milo_eggs_weight :
  (dozens_needed * eggs_per_dozen : ℚ) * weight_of_one_egg = 6 := by sorry

end milo_eggs_weight_l139_13989


namespace tangent_curve_line_a_eq_neg1_l139_13958

theorem tangent_curve_line_a_eq_neg1 (a : ℝ) (x : ℝ) : 
  (∀ (x : ℝ), (e^x + a = x) ∧ (e^x = 1) ) → a = -1 :=
by 
  intro h
  sorry

end tangent_curve_line_a_eq_neg1_l139_13958


namespace area_of_TURS_eq_area_of_PQRS_l139_13977

-- Definition of the rectangle PQRS
structure Rectangle where
  length : ℕ
  width : ℕ
  area : ℕ

-- Definition of the trapezoid TURS
structure Trapezoid where
  base1 : ℕ
  base2 : ℕ
  height : ℕ
  area : ℕ

-- Condition: PQRS is a rectangle whose area is 20 square units
def PQRS : Rectangle := { length := 5, width := 4, area := 20 }

-- Question: Prove the area of TURS equals area of PQRS
theorem area_of_TURS_eq_area_of_PQRS (TURS_area : ℕ) : TURS_area = PQRS.area :=
  sorry

end area_of_TURS_eq_area_of_PQRS_l139_13977


namespace fiona_first_to_toss_eight_l139_13926

theorem fiona_first_to_toss_eight :
  (∃ p : ℚ, p = 49/169 ∧
    (∀ n:ℕ, (7/8:ℚ)^(3*n) * (1/8) = if n = 0 then (49/512) else (49/512) * (343/512)^n)) :=
sorry

end fiona_first_to_toss_eight_l139_13926


namespace rate_per_meter_for_fencing_l139_13948

theorem rate_per_meter_for_fencing
  (w : ℕ) (length : ℕ) (perimeter : ℕ) (cost : ℕ)
  (h1 : length = w + 10)
  (h2 : perimeter = 2 * (length + w))
  (h3 : perimeter = 340)
  (h4 : cost = 2210) : (cost / perimeter : ℝ) = 6.5 := by
  sorry

end rate_per_meter_for_fencing_l139_13948


namespace find_number_eq_36_l139_13975

theorem find_number_eq_36 (n : ℝ) (h : (n / 18) * (n / 72) = 1) : n = 36 :=
sorry

end find_number_eq_36_l139_13975


namespace fourth_person_height_l139_13922

noncomputable def height_of_fourth_person (H : ℕ) : ℕ := 
  let second_person := H + 2
  let third_person := H + 4
  let fourth_person := H + 10
  fourth_person

theorem fourth_person_height {H : ℕ} 
  (cond1 : 2 = 2)
  (cond2 : 6 = 6)
  (average_height : 76 = 76) 
  (height_sum : H + (H + 2) + (H + 4) + (H + 10) = 304) : 
  height_of_fourth_person H = 82 := sorry

end fourth_person_height_l139_13922


namespace tan_phi_l139_13961

theorem tan_phi (φ : ℝ) (h1 : Real.cos (π / 2 + φ) = 2 / 3) (h2 : abs φ < π / 2) : 
  Real.tan φ = -2 * Real.sqrt 5 / 5 := 
by 
  sorry

end tan_phi_l139_13961


namespace solution_set_of_inequality_l139_13967

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 12 < 0 } = { x : ℝ | -4 < x ∧ x < 3 } :=
by
  sorry

end solution_set_of_inequality_l139_13967


namespace find_x_l139_13995

-- Definition of the binary operation
def binary_operation (a b c d : ℤ) : ℤ × ℤ :=
  (a - c, b + d)

-- Definition of our main theorem to be proved
theorem find_x (x y : ℤ) (h : binary_operation x y 2 3 = (4, 5)) : x = 6 :=
  by sorry

end find_x_l139_13995


namespace plane_eq_unique_l139_13942

open Int 

def plane_eq (A B C D x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_eq_unique (x y z : ℤ) (A B C D : ℤ)
  (h₁ : x = 8) 
  (h₂ : y = -6) 
  (h₃ : z = 2) 
  (h₄ : A > 0)
  (h₅ : gcd (|A|) (gcd (|B|) (gcd (|C|) (|D|))) = 1) :
  plane_eq 4 (-3) 1 (-52) x y z :=
by
  sorry

end plane_eq_unique_l139_13942


namespace words_per_hour_after_two_hours_l139_13914

theorem words_per_hour_after_two_hours 
  (total_words : ℕ) (initial_rate : ℕ) (initial_time : ℕ) (start_time_before_deadline : ℕ) 
  (words_written_in_first_phase : ℕ) (remaining_words : ℕ) (remaining_time : ℕ)
  (final_rate_per_hour : ℕ) :
  total_words = 1200 →
  initial_rate = 400 →
  initial_time = 2 →
  start_time_before_deadline = 4 →
  words_written_in_first_phase = initial_rate * initial_time →
  remaining_words = total_words - words_written_in_first_phase →
  remaining_time = start_time_before_deadline - initial_time →
  final_rate_per_hour = remaining_words / remaining_time →
  final_rate_per_hour = 200 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end words_per_hour_after_two_hours_l139_13914


namespace central_angle_is_two_l139_13985

noncomputable def central_angle_of_sector (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) : ℝ :=
  l / r

theorem central_angle_is_two (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) : central_angle_of_sector r l h1 h2 = 2 :=
by
  sorry

end central_angle_is_two_l139_13985


namespace find_integer_mod_condition_l139_13936

theorem find_integer_mod_condition (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 4) (h3 : n ≡ -998 [ZMOD 5]) : n = 2 :=
sorry

end find_integer_mod_condition_l139_13936


namespace intersection_complement_l139_13917

def A := {x : ℝ | -1 < x ∧ x < 6}
def B := {x : ℝ | x^2 < 4}
def complement_R (S : Set ℝ) := {x : ℝ | x ∉ S}

theorem intersection_complement :
  A ∩ (complement_R B) = {x : ℝ | 2 ≤ x ∧ x < 6} := by
sorry

end intersection_complement_l139_13917


namespace sum_of_solutions_l139_13983

theorem sum_of_solutions :
  let a := -48
  let b := 110
  let c := 165
  ( ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) → x1 ≠ x2 → (x1 + x2) = 55 / 24 ) :=
by
  let a := -48
  let b := 110
  let c := 165
  sorry

end sum_of_solutions_l139_13983


namespace hypotenuse_eq_medians_l139_13957

noncomputable def hypotenuse_length_medians (a b : ℝ) (h1 : b^2 + (9 * a^2) / 4 = 48) (h2 : a^2 + (9 * b^2) / 4 = 36) : ℝ :=
  3 * Real.sqrt (336 / 13)

-- definition
theorem hypotenuse_eq_medians {a b : ℝ} (h1 : b^2 + (9 * a^2) / 4 = 48) (h2 : a^2 + (9 * b^2) / 4 = 36) :
    Real.sqrt (9 * (a^2 + b^2)) = 3 * Real.sqrt (336 / 13) :=
sorry

end hypotenuse_eq_medians_l139_13957


namespace cost_price_of_watch_l139_13938

/-
Let's state the problem conditions as functions
C represents the cost price
SP1 represents the selling price at 36% loss
SP2 represents the selling price at 4% gain
-/

def cost_price (C : ℝ) : ℝ := C

def selling_price_loss (C : ℝ) : ℝ := 0.64 * C

def selling_price_gain (C : ℝ) : ℝ := 1.04 * C

def price_difference (C : ℝ) : ℝ := (selling_price_gain C) - (selling_price_loss C)

theorem cost_price_of_watch : ∀ C : ℝ, price_difference C = 140 → C = 350 :=
by
   intro C H
   sorry

end cost_price_of_watch_l139_13938


namespace is_divisible_by_six_l139_13903

/-- A stingy knight keeps gold coins in six chests. Given that he can evenly distribute the coins by opening any
two chests, any three chests, any four chests, or any five chests, prove that the total number of coins can be 
evenly distributed among all six chests. -/
theorem is_divisible_by_six (n : ℕ) 
  (h2 : ∀ (a b : ℕ), a + b = n → (a % 2 = 0 ∧ b % 2 = 0))
  (h3 : ∀ (a b c : ℕ), a + b + c = n → (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0)) 
  (h4 : ∀ (a b c d : ℕ), a + b + c + d = n → (a % 4 = 0 ∧ b % 4 = 0 ∧ c % 4 = 0 ∧ d % 4 = 0))
  (h5 : ∀ (a b c d e : ℕ), a + b + c + d + e = n → (a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0)) :
  n % 6 = 0 :=
sorry

end is_divisible_by_six_l139_13903


namespace savings_account_amount_l139_13969

noncomputable def final_amount : ℝ :=
  let initial_deposit : ℝ := 5000
  let first_quarter_rate : ℝ := 0.01
  let second_quarter_rate : ℝ := 0.0125
  let deposit_end_third_month : ℝ := 1000
  let withdrawal_end_fifth_month : ℝ := 500
  let amount_after_first_quarter := initial_deposit * (1 + first_quarter_rate)
  let amount_before_second_quarter := amount_after_first_quarter + deposit_end_third_month
  let amount_after_second_quarter := amount_before_second_quarter * (1 + second_quarter_rate)
  let final_amount := amount_after_second_quarter - withdrawal_end_fifth_month
  final_amount

theorem savings_account_amount :
  final_amount = 5625.625 :=
by
  sorry

end savings_account_amount_l139_13969


namespace probability_sqrt_lt_7_of_random_two_digit_number_l139_13933

theorem probability_sqrt_lt_7_of_random_two_digit_number : 
  (∃ p : ℚ, (∀ n, 10 ≤ n ∧ n ≤ 99 → n < 49 → ∃ k, k = p) ∧ p = 13 / 30) := 
by
  sorry

end probability_sqrt_lt_7_of_random_two_digit_number_l139_13933


namespace regular_pyramid_cannot_be_hexagonal_l139_13940

theorem regular_pyramid_cannot_be_hexagonal (n : ℕ) (h₁ : n = 6) (base_edge_length slant_height : ℝ) 
  (reg_pyramid : base_edge_length = slant_height) : false :=
by
  sorry

end regular_pyramid_cannot_be_hexagonal_l139_13940


namespace min_area_triangle_l139_13998

theorem min_area_triangle (m n : ℝ) (h : m^2 + n^2 = 1/3) : ∃ S, S = 3 :=
by
  sorry

end min_area_triangle_l139_13998
