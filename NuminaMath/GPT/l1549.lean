import Mathlib

namespace range_of_a_l1549_154970

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (e^x - a)^2 + x^2 - 2 * a * x + a^2 ≤ 1 / 2) ↔ a = 1 / 2 :=
by
  sorry

end range_of_a_l1549_154970


namespace correct_definition_of_regression_independence_l1549_154996

-- Definitions
def regression_analysis (X Y : Type) := ∃ r : X → Y, true -- Placeholder, ideal definition studies correlation
def independence_test (X Y : Type) := ∃ rel : X → Y → Prop, true -- Placeholder, ideal definition examines relationship

-- Theorem statement
theorem correct_definition_of_regression_independence (X Y : Type) :
  (∃ r : X → Y, true) ∧ (∃ rel : X → Y → Prop, true)
  → "Regression analysis studies the correlation between two variables, and independence tests examine whether there is some kind of relationship between two variables" = "C" :=
sorry

end correct_definition_of_regression_independence_l1549_154996


namespace solution_exists_l1549_154947

open Real

theorem solution_exists (x : ℝ) (h1 : x > 9) (h2 : sqrt (x - 3 * sqrt (x - 9)) + 3 = sqrt (x + 3 * sqrt (x - 9)) - 3) : x ≥ 18 :=
sorry

end solution_exists_l1549_154947


namespace contribution_amount_l1549_154999

theorem contribution_amount (x : ℝ) (S : ℝ) :
  (S = 10 * x) ∧ (S = 15 * (x - 100)) → x = 300 :=
by
  sorry

end contribution_amount_l1549_154999


namespace varphi_solution_l1549_154998

noncomputable def varphi (x : ℝ) (m n : ℝ) : ℝ :=
  m * x + n / x

theorem varphi_solution :
  ∃ (m n : ℝ), (varphi 1 m n = 8) ∧ (varphi 16 m n = 16) ∧ (∀ x, varphi x m n = 3 * x + 5 / x) :=
sorry

end varphi_solution_l1549_154998


namespace exists_quadratic_sequence_l1549_154919

theorem exists_quadratic_sequence (b c : ℤ) : ∃ n : ℕ, ∃ (a : ℕ → ℤ), (a 0 = b) ∧ (a n = c) ∧ ∀ i : ℕ, 1 ≤ i → i ≤ n → |a i - a (i - 1)| = i ^ 2 := 
sorry

end exists_quadratic_sequence_l1549_154919


namespace car_return_speed_l1549_154980

noncomputable def round_trip_speed (d : ℝ) (r : ℝ) : ℝ :=
  let travel_time_to_B := d / 75
  let break_time := 1 / 2
  let travel_time_to_A := d / r
  let total_time := travel_time_to_B + travel_time_to_A + break_time
  let total_distance := 2 * d
  total_distance / total_time

theorem car_return_speed :
  let d := 150
  let avg_speed := 50
  round_trip_speed d 42.857 = avg_speed :=
by
  sorry

end car_return_speed_l1549_154980


namespace sufficient_but_not_necessary_condition_l1549_154968

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) : (a - b) * a^2 < 0 → a < b :=
sorry

end sufficient_but_not_necessary_condition_l1549_154968


namespace f_zero_eq_f_expression_alpha_value_l1549_154982

noncomputable def f (ω x : ℝ) : ℝ :=
  3 * Real.sin (ω * x + Real.pi / 6)

theorem f_zero_eq (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  f ω 0 = 3 / 2 :=
by
  sorry

theorem f_expression (ω : ℝ) (hω : ω > 0) (h_period : (2 * Real.pi / ω) = Real.pi / 2) :
  ∀ x : ℝ, f ω x = f 4 x :=
by
  sorry

theorem alpha_value (f_4 : ℝ → ℝ) (α : ℝ) (hα : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h_f4 : ∀ x : ℝ, f_4 x = 3 * Real.sin (4 * x + Real.pi / 6)) (h_fα : f_4 (α / 2) = 3 / 2) :
  α = Real.pi / 3 :=
by
  sorry

end f_zero_eq_f_expression_alpha_value_l1549_154982


namespace find_x_l1549_154946

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : (∀ (a b c d : ℝ), balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 := 
by
  sorry

end find_x_l1549_154946


namespace pyramid_section_rhombus_l1549_154903

structure Pyramid (A B C D : Type) := (point : Type)

def is_parallel (l1 l2 : ℝ) : Prop :=
  ∀ (m n : ℝ), m * l1 = n * l2

def is_parallelogram (K L M N : Type) : Prop :=
  sorry

def is_rhombus (K L M N : Type) : Prop :=
  sorry

noncomputable def side_length_rhombus (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

/-- Prove that the section of pyramid ABCD with a plane parallel to edges AC and BD is a parallelogram,
and under certain conditions, this parallelogram is a rhombus. Find the side of this rhombus given AC = a and BD = b. -/
theorem pyramid_section_rhombus (A B C D K L M N : Type) (a b : ℝ) :
  is_parallel AC BD →
  is_parallelogram K L M N →
  is_rhombus K L M N →
  side_length_rhombus a b = (a * b) / (a + b) :=
by
  sorry

end pyramid_section_rhombus_l1549_154903


namespace percentage_by_which_x_is_less_than_y_l1549_154974

noncomputable def percentageLess (x y : ℝ) : ℝ :=
  ((y - x) / y) * 100

theorem percentage_by_which_x_is_less_than_y :
  ∀ (x y : ℝ),
  y = 125 + 0.10 * 125 →
  x = 123.75 →
  percentageLess x y = 10 :=
by
  intros x y h1 h2
  rw [h1, h2]
  unfold percentageLess
  sorry

end percentage_by_which_x_is_less_than_y_l1549_154974


namespace jeff_bought_from_chad_l1549_154966

/-
  Eric has 4 ninja throwing stars.
  Chad has twice as many ninja throwing stars as Eric.
  Jeff now has 6 ninja throwing stars.
  Together, they have 16 ninja throwing stars.
  How many ninja throwing stars did Jeff buy from Chad?
-/

def eric_stars : ℕ := 4
def chad_stars : ℕ := 2 * eric_stars
def jeff_stars : ℕ := 6
def total_stars : ℕ := 16

theorem jeff_bought_from_chad (bought : ℕ) :
  chad_stars - bought + jeff_stars + eric_stars = total_stars → bought = 2 :=
by
  sorry

end jeff_bought_from_chad_l1549_154966


namespace arccos_half_eq_pi_div_3_l1549_154963

theorem arccos_half_eq_pi_div_3 :
  Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_half_eq_pi_div_3_l1549_154963


namespace brian_holds_breath_for_60_seconds_l1549_154934

-- Definitions based on the problem conditions:
def initial_time : ℕ := 10
def after_first_week (t : ℕ) : ℕ := t * 2
def after_second_week (t : ℕ) : ℕ := t * 2
def after_final_week (t : ℕ) : ℕ := (t * 3) / 2

-- The Lean statement to prove:
theorem brian_holds_breath_for_60_seconds :
  after_final_week (after_second_week (after_first_week initial_time)) = 60 :=
by
  -- Proof steps would go here
  sorry

end brian_holds_breath_for_60_seconds_l1549_154934


namespace fraction_of_area_l1549_154959

def larger_square_side : ℕ := 6
def shaded_square_side : ℕ := 2

def larger_square_area : ℕ := larger_square_side * larger_square_side
def shaded_square_area : ℕ := shaded_square_side * shaded_square_side

theorem fraction_of_area : (shaded_square_area : ℚ) / larger_square_area = 1 / 9 :=
by
  -- proof omitted
  sorry

end fraction_of_area_l1549_154959


namespace find_d_div_a_l1549_154988
noncomputable def quad_to_square_form (x : ℝ) : ℝ :=
  x^2 + 1500 * x + 1800

theorem find_d_div_a : 
  ∃ (a d : ℝ), (∀ x : ℝ, quad_to_square_form x = (x + a)^2 + d) 
  ∧ a = 750 
  ∧ d = -560700 
  ∧ d / a = -560700 / 750 := 
sorry

end find_d_div_a_l1549_154988


namespace total_expenditure_now_l1549_154993

-- Define the conditions in Lean
def original_student_count : ℕ := 100
def additional_students : ℕ := 25
def decrease_in_average_expenditure : ℤ := 10
def increase_in_total_expenditure : ℤ := 500

-- Let's denote the original average expenditure per student as A rupees
variable (A : ℤ)

-- Define the old and new expenditures
def original_total_expenditure := original_student_count * A
def new_average_expenditure := A - decrease_in_average_expenditure
def new_total_expenditure := (original_student_count + additional_students) * new_average_expenditure

-- The theorem to prove
theorem total_expenditure_now :
  new_total_expenditure A - original_total_expenditure A = increase_in_total_expenditure →
  new_total_expenditure A = 7500 :=
by
  sorry

end total_expenditure_now_l1549_154993


namespace expansion_a0_value_l1549_154933

theorem expansion_a0_value :
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), (∀ x : ℝ, (x+1)^5 = a_0 + a_1*(x-1) + a_2*(x-1)^2 + a_3*(x-1)^3 + a_4*(x-1)^4 + a_5*(x-1)^5) ∧ a_0 = 32 :=
  sorry

end expansion_a0_value_l1549_154933


namespace determine_B_l1549_154920

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (h1 : (A ∪ B)ᶜ = {1})
variable (h2 : A ∩ Bᶜ = {3})

theorem determine_B : B = {2, 4, 5} :=
by
  sorry

end determine_B_l1549_154920


namespace total_marbles_l1549_154984

theorem total_marbles (ratio_red_blue_green_yellow : ℕ → ℕ → ℕ → ℕ → Prop) (total : ℕ) :
  (∀ r b g y, ratio_red_blue_green_yellow r b g y ↔ r = 1 ∧ b = 5 ∧ g = 3 ∧ y = 2) →
  (∃ y, y = 20) →
  (total = y * 11 / 2) →
  total = 110 :=
by
  intros ratio_condition yellow_condition total_condition
  sorry

end total_marbles_l1549_154984


namespace parabola_distance_focus_P_l1549_154964

noncomputable def distance_PF : ℝ := sorry

theorem parabola_distance_focus_P : ∀ (P : ℝ × ℝ) (F : ℝ × ℝ),
  P.2^2 = 4 * P.1 ∧ F = (1, 0) ∧ P.1 = 4 → distance_PF = 5 :=
by
  intros P F h
  sorry

end parabola_distance_focus_P_l1549_154964


namespace day_of_week_proof_l1549_154923

/-- 
January 1, 1978, is a Sunday in the Gregorian calendar.
What day of the week is January 1, 2000, in the Gregorian calendar?
-/
def day_of_week_2000 := "Saturday"

theorem day_of_week_proof :
  let initial_year := 1978
  let target_year := 2000
  let initial_weekday := "Sunday"
  let years_between := target_year - initial_year -- 22 years
  let normal_days := years_between * 365 -- Normal days in these years
  let leap_years := 5 -- Number of leap years in the range
  let total_days := normal_days + leap_years -- Total days considering leap years
  let remainder_days := total_days % 7 -- days modulo 7
  initial_weekday = "Sunday" → remainder_days = 6 → 
  day_of_week_2000 = "Saturday" :=
by
  sorry

end day_of_week_proof_l1549_154923


namespace tan_identity_find_sum_l1549_154948

-- Given conditions
def is_geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c

-- Specific problem statements
theorem tan_identity (a b c : ℝ) (A B C : ℝ)
  (h_geometric : is_geometric_sequence a b c)
  (h_cosB : Real.cos B = 3 / 4) :
  1 / Real.tan A + 1 / Real.tan C = 4 / Real.sqrt 7 :=
sorry

theorem find_sum (a b c : ℝ)
  (h_dot_product : a * c * 3 / 4 = 3 / 2) :
  a + c = 3 :=
sorry

end tan_identity_find_sum_l1549_154948


namespace acute_angle_inclination_range_l1549_154908

/-- 
For the line passing through points P(1-a, 1+a) and Q(3, 2a), 
prove that the range of the real number a such that the line has an acute angle of inclination is (-∞, 1) ∪ (1, 4).
-/
theorem acute_angle_inclination_range (a : ℝ) : 
  (a < 1 ∨ (1 < a ∧ a < 4)) ↔ (0 < (a - 1) / (4 - a)) :=
sorry

end acute_angle_inclination_range_l1549_154908


namespace sufficient_but_not_necessary_condition_l1549_154906

variable (a b x y : ℝ)

theorem sufficient_but_not_necessary_condition (ha : a > 0) (hb : b > 0) :
  ((x > a ∧ y > b) → (x + y > a + b ∧ x * y > a * b)) ∧
  ¬((x + y > a + b ∧ x * y > a * b) → (x > a ∧ y > b)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1549_154906


namespace problem_statement_l1549_154973

theorem problem_statement (m : ℂ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2005 = 2006 :=
  sorry

end problem_statement_l1549_154973


namespace intersection_of_sets_l1549_154987

def SetA : Set ℝ := {x | 0 < x ∧ x < 3}
def SetB : Set ℝ := {x | x > 2}
def SetC : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_of_sets :
  SetA ∩ SetB = SetC :=
by
  sorry

end intersection_of_sets_l1549_154987


namespace Rachel_painting_time_l1549_154945

noncomputable def Matt_time : ℕ := 12
noncomputable def Patty_time (Matt_time : ℕ) : ℕ := Matt_time / 3
noncomputable def Rachel_time (Patty_time : ℕ) : ℕ := 5 + 2 * Patty_time

theorem Rachel_painting_time : Rachel_time (Patty_time Matt_time) = 13 := by
  sorry

end Rachel_painting_time_l1549_154945


namespace quadratic_real_solutions_l1549_154905

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) := 
sorry

end quadratic_real_solutions_l1549_154905


namespace service_station_location_l1549_154995

/-- The first exit is at milepost 35. -/
def first_exit_milepost : ℕ := 35

/-- The eighth exit is at milepost 275. -/
def eighth_exit_milepost : ℕ := 275

/-- The expected milepost of the service station built halfway between the first exit and the eighth exit is 155. -/
theorem service_station_location : (first_exit_milepost + (eighth_exit_milepost - first_exit_milepost) / 2) = 155 := by
  sorry

end service_station_location_l1549_154995


namespace remainder_3249_div_82_eq_51_l1549_154938

theorem remainder_3249_div_82_eq_51 : (3249 % 82) = 51 :=
by
  sorry

end remainder_3249_div_82_eq_51_l1549_154938


namespace part1_solution_set_part2_range_of_a_l1549_154935

noncomputable def f (x : ℝ) : ℝ := abs (4 * x - 1) - abs (x + 2)

-- Part 1: Prove the solution set of f(x) < 8 is -9 / 5 < x < 11 / 3
theorem part1_solution_set : {x : ℝ | f x < 8} = {x : ℝ | -9 / 5 < x ∧ x < 11 / 3} :=
sorry

-- Part 2: Prove the range of a such that the inequality has a solution
theorem part2_range_of_a (a : ℝ) : (∃ x : ℝ, f x + 5 * abs (x + 2) < a^2 - 8 * a) ↔ (a < -1 ∨ a > 9) :=
sorry

end part1_solution_set_part2_range_of_a_l1549_154935


namespace probability_jerry_at_four_l1549_154940

theorem probability_jerry_at_four :
  let total_flips := 8
  let coordinate := 4
  let total_possible_outcomes := 2 ^ total_flips
  let favorable_outcomes := Nat.choose total_flips (total_flips / 2 + coordinate / 2)
  let P := favorable_outcomes / total_possible_outcomes
  let a := 7
  let b := 64
  ∃ (a b : ℕ), Nat.gcd a b = 1 ∧ P = a / b ∧ a + b = 71
:= sorry

end probability_jerry_at_four_l1549_154940


namespace batsman_average_after_17th_inning_l1549_154910

theorem batsman_average_after_17th_inning 
    (A : ℕ)  -- assuming A (the average before the 17th inning) is a natural number
    (h₁ : 16 * A + 85 = 17 * (A + 3)) : 
    A + 3 = 37 := by
  sorry

end batsman_average_after_17th_inning_l1549_154910


namespace triangle_area_l1549_154949

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) 
                      (h5 : a + b > c) (h6 : b + c > a) (h7 : c + a > b) :
  ∃ A : ℝ, A = 2 * Real.sqrt 6 ∧
    ∃ (h : 0 ≤ A), A = (Real.sqrt (A * 12 * (12 - a) * (12 - b) * (12 - c))) :=
sorry

end triangle_area_l1549_154949


namespace triangle_value_a_l1549_154972

theorem triangle_value_a (a : ℕ) (h1: a + 2 > 6) (h2: a + 6 > 2) (h3: 2 + 6 > a) : a = 7 :=
sorry

end triangle_value_a_l1549_154972


namespace washingMachineCapacity_l1549_154965

-- Definitions based on the problem's conditions
def numberOfShirts : ℕ := 2
def numberOfSweaters : ℕ := 33
def numberOfLoads : ℕ := 5

-- Statement we need to prove
theorem washingMachineCapacity : 
  (numberOfShirts + numberOfSweaters) / numberOfLoads = 7 := sorry

end washingMachineCapacity_l1549_154965


namespace even_function_m_value_l1549_154960

def f (x m : ℝ) : ℝ := (x - 2) * (x - m)

theorem even_function_m_value (m : ℝ) :
  (∀ x : ℝ, f x m = f (-x) m) → m = -2 := by
  sorry

end even_function_m_value_l1549_154960


namespace sqrt_ab_equals_sqrt_2_l1549_154932

theorem sqrt_ab_equals_sqrt_2 
  (a b : ℝ)
  (h1 : a ^ 2 = 16 / 25)
  (h2 : b ^ 3 = 125 / 8) : 
  Real.sqrt (a * b) = Real.sqrt 2 := 
by 
  -- proof will go here
  sorry

end sqrt_ab_equals_sqrt_2_l1549_154932


namespace cos_alpha_condition_l1549_154921

theorem cos_alpha_condition (k : ℤ) (α : ℝ) :
  (α = 2 * k * Real.pi - Real.pi / 4 -> Real.cos α = Real.sqrt 2 / 2) ∧
  (Real.cos α = Real.sqrt 2 / 2 -> ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 4 ∨ α = 2 * k * Real.pi - Real.pi / 4) :=
by
  sorry

end cos_alpha_condition_l1549_154921


namespace correct_statements_l1549_154981

theorem correct_statements : 
  (∃ n : ℕ, 24 = 4 * n) ∧ (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end correct_statements_l1549_154981


namespace eval_expression_l1549_154983

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end eval_expression_l1549_154983


namespace multiplier_of_reciprocal_l1549_154901

theorem multiplier_of_reciprocal (x m : ℝ) (h1 : x = 7) (h2 : x - 4 = m * (1 / x)) : m = 21 :=
by
  sorry

end multiplier_of_reciprocal_l1549_154901


namespace not_periodic_cos_add_cos_sqrt2_l1549_154928

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.cos (x * Real.sqrt 2)

theorem not_periodic_cos_add_cos_sqrt2 :
  ¬(∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) :=
sorry

end not_periodic_cos_add_cos_sqrt2_l1549_154928


namespace truncated_pyramid_properties_l1549_154941

noncomputable def truncatedPyramidSurfaceArea
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the surface area function

noncomputable def truncatedPyramidVolume
  (a b c : ℝ) (theta m : ℝ) : ℝ :=
sorry -- Definition of the volume function

theorem truncated_pyramid_properties
  (a b c : ℝ) (theta m : ℝ)
  (h₀ : a = 148) 
  (h₁ : b = 156) 
  (h₂ : c = 208) 
  (h₃ : theta = 112.62) 
  (h₄ : m = 27) :
  (truncatedPyramidSurfaceArea a b c theta m = 74352) ∧
  (truncatedPyramidVolume a b c theta m = 395280) :=
by
  sorry -- The actual proof will go here

end truncated_pyramid_properties_l1549_154941


namespace quinn_frogs_caught_l1549_154957

-- Defining the conditions
def Alster_frogs : Nat := 2

def Quinn_frogs (Alster_caught: Nat) : Nat := Alster_caught

def Bret_frogs (Quinn_caught: Nat) : Nat := 3 * Quinn_caught

-- Given that Bret caught 12 frogs, prove the amount Quinn caught
theorem quinn_frogs_caught (Bret_caught: Nat) (h1: Bret_caught = 12) : Quinn_frogs Alster_frogs = 4 :=
by
  sorry

end quinn_frogs_caught_l1549_154957


namespace chord_intersects_inner_circle_probability_l1549_154992

noncomputable def probability_of_chord_intersecting_inner_circle
  (radius_inner : ℝ) (radius_outer : ℝ)
  (chord_probability : ℝ) : Prop :=
  radius_inner = 3 ∧ radius_outer = 5 ∧ chord_probability = 0.205

theorem chord_intersects_inner_circle_probability :
  probability_of_chord_intersecting_inner_circle 3 5 0.205 :=
by {
  sorry
}

end chord_intersects_inner_circle_probability_l1549_154992


namespace train_length_l1549_154997

/-- Given a train that can cross an electric pole in 15 seconds and has a speed of 72 km/h, prove that the length of the train is 300 meters. -/
theorem train_length 
  (time_to_cross_pole : ℝ)
  (train_speed_kmh : ℝ)
  (h1 : time_to_cross_pole = 15)
  (h2 : train_speed_kmh = 72)
  : (train_speed_kmh * 1000 / 3600) * time_to_cross_pole = 300 := 
by
  -- Proof goes here
  sorry

end train_length_l1549_154997


namespace carter_baseball_cards_l1549_154922

theorem carter_baseball_cards (m c : ℕ) (h1 : m = 210) (h2 : m = c + 58) : c = 152 := 
by
  sorry

end carter_baseball_cards_l1549_154922


namespace find_p_q_l1549_154912

theorem find_p_q (p q : ℤ)
  (h : (5 * d^2 - 4 * d + p) * (4 * d^2 + q * d - 5) = 20 * d^4 + 11 * d^3 - 45 * d^2 - 20 * d + 25) :
  p + q = 3 :=
sorry

end find_p_q_l1549_154912


namespace problem_statement_l1549_154927

theorem problem_statement :
  75 * ((4 + 1/3) - (5 + 1/4)) / ((3 + 1/2) + (2 + 1/5)) = -5/31 := 
by
  sorry

end problem_statement_l1549_154927


namespace smallest_k_values_l1549_154967

def cos_squared_eq_one (k : ℕ) : Prop :=
  ∃ n : ℕ, k^2 + 49 = 180 * n

theorem smallest_k_values :
  ∃ (k1 k2 : ℕ), (cos_squared_eq_one k1) ∧ (cos_squared_eq_one k2) ∧
  (∀ k < k1, ¬ cos_squared_eq_one k) ∧ (∀ k < k2, ¬ cos_squared_eq_one k) ∧ 
  k1 = 31 ∧ k2 = 37 :=
by
  sorry

end smallest_k_values_l1549_154967


namespace probability_non_edge_unit_square_l1549_154918

theorem probability_non_edge_unit_square : 
  let total_squares := 100
  let perimeter_squares := 36
  let non_perimeter_squares := total_squares - perimeter_squares
  let probability := (non_perimeter_squares : ℚ) / total_squares
  probability = 16 / 25 :=
by
  sorry

end probability_non_edge_unit_square_l1549_154918


namespace sufficient_not_necessary_condition_l1549_154994

variable (x : ℝ)

theorem sufficient_not_necessary_condition :
  (x > 2 → x > 1) ∧ (¬ (x > 1 → x > 2)) := by
sorry

end sufficient_not_necessary_condition_l1549_154994


namespace leak_drain_time_l1549_154911

noncomputable def pump_rate : ℚ := 1/2
noncomputable def leak_empty_rate : ℚ := 1 / (1 / pump_rate - 5/11)

theorem leak_drain_time :
  let pump_rate := 1/2
  let combined_rate := 5/11
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate = 22 :=
  by
    -- Definition of pump rate
    let pump_rate := 1/2
    -- Definition of combined rate
    let combined_rate := 5/11
    -- Definition of leak rate
    let leak_rate := pump_rate - combined_rate
    -- Calculate leak drain time
    show 1 / leak_rate = 22
    sorry

end leak_drain_time_l1549_154911


namespace second_player_wins_l1549_154978

-- Define the initial condition of the game
def initial_coins : Nat := 2016

-- Define the set of moves a player can make
def valid_moves : Finset Nat := {1, 2, 3}

-- Define the winning condition
def winning_player (coins : Nat) : String :=
  if coins % 4 = 0 then "second player"
  else "first player"

-- The theorem stating that second player has a winning strategy given the initial condition
theorem second_player_wins : winning_player initial_coins = "second player" :=
by
  sorry

end second_player_wins_l1549_154978


namespace least_even_p_l1549_154936

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem least_even_p 
  (p : ℕ) 
  (hp : 2 ∣ p) -- p is an even integer
  (h : is_square (300 * p)) -- 300 * p is the square of an integer
  : p = 3 := 
sorry

end least_even_p_l1549_154936


namespace area_of_one_postcard_is_150_cm2_l1549_154958

/-- Define the conditions of the problem. -/
def perimeter_of_stitched_postcard : ℕ := 70
def vertical_length_of_postcard : ℕ := 15

/-- Definition stating that postcards are attached horizontally and do not overlap. 
    This logically implies that the horizontal length gets doubled and perimeter is 2V + 4H. -/
def attached_horizontally (V H : ℕ) (P : ℕ) : Prop :=
  2 * V + 4 * H = P

/-- Main theorem stating the question and the derived answer,
    proving that the area of one postcard is 150 square centimeters. -/
theorem area_of_one_postcard_is_150_cm2 :
  ∃ (H : ℕ), attached_horizontally vertical_length_of_postcard H perimeter_of_stitched_postcard ∧
  (vertical_length_of_postcard * H = 150) :=
by 
  sorry -- the proof is omitted

end area_of_one_postcard_is_150_cm2_l1549_154958


namespace sum_of_primes_between_20_and_40_l1549_154913

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end sum_of_primes_between_20_and_40_l1549_154913


namespace cyclic_sum_inequality_l1549_154979

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
  sorry

end cyclic_sum_inequality_l1549_154979


namespace range_of_a_l1549_154904

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 3 → true) ∧
  (∀ x : ℝ, (a - 2)*x^2 + (2 * a - 1) * x + 6 > 0 ↔ x = 5 → false) →
  1 < a ∧ a ≤ 7 / 5 :=
by
  sorry

end range_of_a_l1549_154904


namespace three_digit_difference_divisible_by_9_l1549_154929

theorem three_digit_difference_divisible_by_9 :
  ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    (100 * a + 10 * b + c - (a + b + c)) % 9 = 0 :=
by
  intros a b c h
  sorry

end three_digit_difference_divisible_by_9_l1549_154929


namespace division_neg4_by_2_l1549_154916

theorem division_neg4_by_2 : (-4) / 2 = -2 := sorry

end division_neg4_by_2_l1549_154916


namespace sum_X_Y_l1549_154971

-- Define the variables and assumptions
variable (X Y : ℕ)

-- Hypotheses
axiom h1 : Y + 2 = X
axiom h2 : X + 5 = Y

-- Theorem statement
theorem sum_X_Y : X + Y = 12 := by
  sorry

end sum_X_Y_l1549_154971


namespace symmetry_y_axis_B_l1549_154917

def point_A : ℝ × ℝ := (-1, 2)

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-(p.1), p.2)

theorem symmetry_y_axis_B :
  symmetric_point point_A = (1, 2) :=
by
  -- proof is omitted
  sorry

end symmetry_y_axis_B_l1549_154917


namespace Q_has_exactly_one_negative_root_l1549_154969

def Q (x : ℝ) : ℝ := x^7 + 5 * x^5 + 5 * x^4 - 6 * x^3 - 2 * x^2 - 10 * x + 12

theorem Q_has_exactly_one_negative_root :
  ∃! r : ℝ, r < 0 ∧ Q r = 0 := sorry

end Q_has_exactly_one_negative_root_l1549_154969


namespace find_higher_selling_price_l1549_154986

-- Define the constants and initial conditions
def cost_price : ℕ := 200
def selling_price_1 : ℕ := 340
def gain_1 : ℕ := selling_price_1 - cost_price
def new_gain : ℕ := gain_1 + gain_1 * 5 / 100

-- Define the problem statement
theorem find_higher_selling_price : 
  ∀ P : ℕ, P = cost_price + new_gain → P = 347 :=
by
  intro P
  intro h
  sorry

end find_higher_selling_price_l1549_154986


namespace stamp_problem_l1549_154962

/-- Define the context where we have stamps of 7, n, and (n + 2) cents, and 120 cents being the largest
    value that cannot be formed using these stamps -/
theorem stamp_problem (n : ℕ) (h : ∀ k, k > 120 → ∃ a b c, k = 7 * a + n * b + (n + 2) * c) (hn : ¬ ∃ a b c, 120 = 7 * a + n * b + (n + 2) * c) : n = 22 :=
sorry

end stamp_problem_l1549_154962


namespace smallest_positive_integer_solution_l1549_154950

theorem smallest_positive_integer_solution :
  ∃ x : ℕ, 0 < x ∧ 5 * x ≡ 17 [MOD 34] ∧ (∀ y : ℕ, 0 < y ∧ 5 * y ≡ 17 [MOD 34] → x ≤ y) :=
sorry

end smallest_positive_integer_solution_l1549_154950


namespace shirts_per_minute_l1549_154975

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (h1 : total_shirts = 196) (h2 : total_minutes = 28) :
  total_shirts / total_minutes = 7 :=
by
  -- beginning of proof would go here
  sorry

end shirts_per_minute_l1549_154975


namespace factorize_expression_l1549_154943

theorem factorize_expression (a x y : ℝ) :
  a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 :=
by
  sorry

end factorize_expression_l1549_154943


namespace sequence_B_is_arithmetic_l1549_154902

-- Definitions of the sequences
def S_n (n : ℕ) : ℕ := 2*n + 1

-- Theorem statement
theorem sequence_B_is_arithmetic : ∀ n : ℕ, S_n (n + 1) - S_n n = 2 :=
by
  intro n
  sorry

end sequence_B_is_arithmetic_l1549_154902


namespace time_to_build_wall_l1549_154924

theorem time_to_build_wall (t_A t_B t_C : ℝ) 
  (h1 : 1 / t_A + 1 / t_B = 1 / 25)
  (h2 : 1 / t_C = 1 / 35)
  (h3 : 1 / t_A = 1 / t_B + 1 / t_C) : t_B = 87.5 :=
by
  sorry

end time_to_build_wall_l1549_154924


namespace divisibility_l1549_154909

theorem divisibility {n A B k : ℤ} (h_n : n = 1000 * B + A) (h_k : k = A - B) :
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) :=
by
  sorry

end divisibility_l1549_154909


namespace red_packet_grabbing_situations_l1549_154961

-- Definitions based on the conditions
def numberOfPeople := 5
def numberOfPackets := 4
def packets := [2, 2, 3, 5]  -- 2-yuan, 2-yuan, 3-yuan, 5-yuan

-- Main theorem statement
theorem red_packet_grabbing_situations : 
  ∃ situations : ℕ, situations = 60 :=
by
  sorry

end red_packet_grabbing_situations_l1549_154961


namespace positive_slope_of_asymptote_l1549_154952

-- Define the conditions
def is_hyperbola (x y : ℝ) : Prop :=
  abs (Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 5) ^ 2 + (y + 2) ^ 2)) = 3

-- Prove the positive slope of the asymptote of the given hyperbola
theorem positive_slope_of_asymptote :
  (∀ x y : ℝ, is_hyperbola x y) → abs (Real.sqrt 7 / 3) = Real.sqrt 7 / 3 :=
by
  intros h
  -- Proof to be provided (proof steps from the provided solution would be used here usually)
  sorry

end positive_slope_of_asymptote_l1549_154952


namespace find_value_of_a_l1549_154907

theorem find_value_of_a
  (a : ℝ)
  (h : (a + 3) * 2 * (-2 / 3) = -4) :
  a = -3 :=
sorry

end find_value_of_a_l1549_154907


namespace chrysler_building_floors_l1549_154942

theorem chrysler_building_floors (L : ℕ) (Chrysler : ℕ) (h1 : Chrysler = L + 11)
  (h2 : L + Chrysler = 35) : Chrysler = 23 :=
by {
  --- proof would be here
  sorry
}

end chrysler_building_floors_l1549_154942


namespace average_speed_is_correct_l1549_154956

-- Define the conditions
def initial_odometer : ℕ := 2552
def final_odometer : ℕ := 2882
def time_first_day : ℕ := 5
def time_second_day : ℕ := 7

-- Calculate total time and distance
def total_time : ℕ := time_first_day + time_second_day
def total_distance : ℕ := final_odometer - initial_odometer

-- Prove that the average speed is 27.5 miles per hour
theorem average_speed_is_correct : (total_distance : ℚ) / (total_time : ℚ) = 27.5 :=
by
  sorry

end average_speed_is_correct_l1549_154956


namespace base_8_digits_sum_l1549_154900

theorem base_8_digits_sum
    (X Y Z : ℕ)
    (h1 : 1 ≤ X ∧ X < 8)
    (h2 : 1 ≤ Y ∧ Y < 8)
    (h3 : 1 ≤ Z ∧ Z < 8)
    (h4 : X ≠ Y)
    (h5 : Y ≠ Z)
    (h6 : Z ≠ X)
    (h7 : 8^2 * X + 8 * Y + Z + 8^2 * Y + 8 * Z + X + 8^2 * Z + 8 * X + Y = 8^3 * X + 8^2 * X + 8 * X) :
  Y + Z = 7 * X :=
by
  sorry

end base_8_digits_sum_l1549_154900


namespace range_of_a_l1549_154937

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| > a) ↔ a < 2 := 
sorry

end range_of_a_l1549_154937


namespace student_scores_correct_answers_l1549_154990

variable (c w : ℕ)

theorem student_scores_correct_answers :
  (c + w = 60) ∧ (4 * c - w = 130) → c = 38 :=
by
  intro h
  sorry

end student_scores_correct_answers_l1549_154990


namespace part_1_conditions_part_2_min_value_l1549_154954

theorem part_1_conditions
  (a b x : ℝ)
  (h1: 2 * a * x^2 - 8 * x - 3 * a^2 < 0)
  (h2: ∀ x, -1 < x -> x < b)
  : a = 2 ∧ b = 3 := sorry

theorem part_2_min_value
  (a b x y : ℝ)
  (h1: x > 0)
  (h2: y > 0)
  (h3: a = 2)
  (h4: b = 3)
  (h5: (a / x) + (b / y) = 1)
  : ∃ min_val : ℝ, min_val = 3 * x + 2 * y ∧ min_val = 24 := sorry

end part_1_conditions_part_2_min_value_l1549_154954


namespace area_of_isosceles_right_triangle_l1549_154914

def is_isosceles_right_triangle (X Y Z : Type*) : Prop :=
∃ (XY YZ XZ : ℝ), XY = 6.000000000000001 ∧ XY > YZ ∧ YZ = XZ ∧ XY = YZ * Real.sqrt 2

theorem area_of_isosceles_right_triangle
  {X Y Z : Type*}
  (h : is_isosceles_right_triangle X Y Z) :
  ∃ A : ℝ, A = 9.000000000000002 :=
by
  sorry

end area_of_isosceles_right_triangle_l1549_154914


namespace general_equation_of_curve_l1549_154944

theorem general_equation_of_curve
  (t : ℝ) (ht : t > 0)
  (x : ℝ) (hx : x = (Real.sqrt t) - (1 / (Real.sqrt t)))
  (y : ℝ) (hy : y = 3 * (t + 1 / t) + 2) :
  x^2 = (y - 8) / 3 := by
  sorry

end general_equation_of_curve_l1549_154944


namespace workshop_employees_l1549_154915

theorem workshop_employees (x y : ℕ) 
  (H1 : (x + y) - ((1 / 2) * x + (1 / 3) * y + (1 / 3) * x + (1 / 2) * y) = 120)
  (H2 : (1 / 2) * x + (1 / 3) * y = (1 / 7) * ((1 / 3) * x + (1 / 2) * y) + (1 / 3) * x + (1 / 2) * y) : 
  x = 480 ∧ y = 240 := 
by
  sorry

end workshop_employees_l1549_154915


namespace distinct_real_number_sum_and_square_sum_eq_l1549_154955

theorem distinct_real_number_sum_and_square_sum_eq
  (a b c d : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b + c + d = 3)
  (h_square_sum : a^2 + b^2 + c^2 + d^2 = 45) :
  (a^5 / (a - b) / (a - c) / (a - d)) + (b^5 / (b - a) / (b - c) / (b - d)) +
  (c^5 / (c - a) / (c - b) / (c - d)) + (d^5 / (d - a) / (d - b) / (d - c)) = -9 :=
by
  sorry

end distinct_real_number_sum_and_square_sum_eq_l1549_154955


namespace probability_digit_9_in_3_over_11_is_zero_l1549_154989

-- Define the repeating block of the fraction 3/11
def repeating_block_3_over_11 : List ℕ := [2, 7]

-- Define the function to count the occurrences of a digit in a list
def count_occurrences (digit : ℕ) (lst : List ℕ) : ℕ :=
  lst.count digit

-- Define the probability function
def probability_digit_9_in_3_over_11 : ℚ :=
  (count_occurrences 9 repeating_block_3_over_11) / repeating_block_3_over_11.length

-- Theorem statement
theorem probability_digit_9_in_3_over_11_is_zero : 
  probability_digit_9_in_3_over_11 = 0 := 
by 
  sorry

end probability_digit_9_in_3_over_11_is_zero_l1549_154989


namespace sum_of_squares_of_roots_l1549_154925

theorem sum_of_squares_of_roots :
  let a := 5
  let b := -7
  let c := 2
  let x1 := (-b + (b^2 - 4*a*c)^(1/2)) / (2*a)
  let x2 := (-b - (b^2 - 4*a*c)^(1/2)) / (2*a)
  x1^2 + x2^2 = (b^2 - 2*a*c) / a^2 :=
by
  sorry

end sum_of_squares_of_roots_l1549_154925


namespace average_speed_first_part_l1549_154976

noncomputable def speed_of_first_part (v : ℝ) : Prop :=
  let distance_first_part := 124
  let speed_second_part := 60
  let distance_second_part := 250 - distance_first_part
  let total_time := 5.2
  (distance_first_part / v) + (distance_second_part / speed_second_part) = total_time

theorem average_speed_first_part : speed_of_first_part 40 :=
  sorry

end average_speed_first_part_l1549_154976


namespace product_of_roots_l1549_154930

theorem product_of_roots (x : ℝ) (h : x + 16 / x = 12) : (8 : ℝ) * (4 : ℝ) = 32 :=
by
  -- Your proof would go here
  sorry

end product_of_roots_l1549_154930


namespace mean_of_added_numbers_l1549_154926

noncomputable def mean (a : List ℚ) : ℚ :=
  (a.sum) / (a.length)

theorem mean_of_added_numbers 
  (sum_eight_numbers : ℚ)
  (sum_eleven_numbers : ℚ)
  (x y z : ℚ)
  (h_eight : sum_eight_numbers = 8 * 72)
  (h_eleven : sum_eleven_numbers = 11 * 85)
  (h_sum_added : x + y + z = sum_eleven_numbers - sum_eight_numbers) :
  (x + y + z) / 3 = 119 + 2/3 := 
sorry

end mean_of_added_numbers_l1549_154926


namespace total_crayons_l1549_154985

-- Definitions for conditions
def boxes : Nat := 7
def crayons_per_box : Nat := 5

-- Statement that needs to be proved
theorem total_crayons : boxes * crayons_per_box = 35 := by
  sorry

end total_crayons_l1549_154985


namespace count_two_digit_integers_l1549_154931

def two_digit_integers_satisfying_condition : Nat :=
  let candidates := [(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]
  candidates.length

theorem count_two_digit_integers :
  two_digit_integers_satisfying_condition = 8 :=
by
  sorry

end count_two_digit_integers_l1549_154931


namespace john_total_money_after_3_years_l1549_154991

def principal : ℝ := 1000
def rate : ℝ := 0.1
def time : ℝ := 3

/-
  We need to prove that the total money after 3 years is $1300
-/
theorem john_total_money_after_3_years (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal + (principal * rate * time) = 1300 := by
  sorry

end john_total_money_after_3_years_l1549_154991


namespace problem_expression_value_l1549_154953

theorem problem_expression_value {a b c k1 k2 : ℂ} 
  (h_root : ∀ x, x^3 - k1 * x - k2 = 0 → x = a ∨ x = b ∨ x = c) 
  (h_condition : k1 + k2 ≠ 1)
  (h_vieta1 : a + b + c = 0)
  (h_vieta2 : a * b + b * c + c * a = -k1)
  (h_vieta3 : a * b * c = k2) :
  (1 + a)/(1 - a) + (1 + b)/(1 - b) + (1 + c)/(1 - c) = 
  (3 + k1 + 3 * k2)/(1 - k1 - k2) :=
by
  sorry

end problem_expression_value_l1549_154953


namespace maximum_value_of_f_minimum_value_of_f_l1549_154951

-- Define the function f
def f (x y : ℝ) : ℝ := 3 * |x + y| + |4 * y + 9| + |7 * y - 3 * x - 18|

-- Define the condition
def condition (x y : ℝ) : Prop := x^2 + y^2 ≤ 5

-- State the maximum value theorem
theorem maximum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 + 6 * Real.sqrt 5 := sorry

-- State the minimum value theorem
theorem minimum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 - 3 * Real.sqrt 10 := sorry

end maximum_value_of_f_minimum_value_of_f_l1549_154951


namespace false_statement_B_l1549_154977

theorem false_statement_B : ¬ ∀ α β : ℝ, (α < 90) ∧ (β < 90) → (α + β > 90) :=
by
  sorry

end false_statement_B_l1549_154977


namespace ratio_expression_l1549_154939

theorem ratio_expression (A B C : ℚ) (h : A / B = 3 / 1 ∧ B / C = 1 / 6) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 :=
by sorry

end ratio_expression_l1549_154939
