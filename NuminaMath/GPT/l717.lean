import Mathlib

namespace production_increase_percentage_l717_71705

variable (T : ℝ) -- Initial production
variable (T1 T2 T5 : ℝ) -- Productions at different years
variable (x : ℝ) -- Unknown percentage increase for last three years

-- Conditions
def condition1 : Prop := T1 = T * 1.06
def condition2 : Prop := T2 = T1 * 1.08
def condition3 : Prop := T5 = T * (1.1 ^ 5)

-- Statement to prove
theorem production_increase_percentage :
  condition1 T T1 →
  condition2 T1 T2 →
  (T5 = T2 * (1 + x / 100) ^ 3) →
  x = 12.1 :=
by
  sorry

end production_increase_percentage_l717_71705


namespace triangle_in_and_circumcircle_radius_l717_71799

noncomputable def radius_of_incircle (AC : ℝ) (BC : ℝ) (AB : ℝ) (Area : ℝ) (s : ℝ) : ℝ :=
  Area / s

noncomputable def radius_of_circumcircle (AB : ℝ) : ℝ :=
  AB / 2

theorem triangle_in_and_circumcircle_radius :
  ∀ (A B C : ℝ × ℝ) (AC : ℝ) (BC : ℝ) (AB : ℝ)
    (AngleA : ℝ) (AngleC : ℝ),
  AngleC = 90 ∧ AngleA = 60 ∧ AC = 6 ∧
  BC = AC * Real.sqrt 3 ∧ AB = 2 * AC
  → radius_of_incircle AC BC AB (18 * Real.sqrt 3) ((AC + BC + AB) / 2) = 6 * (Real.sqrt 3 - 1) / 13 ∧
    radius_of_circumcircle AB = 6 := by
  intros A B C AC BC AB AngleA AngleC h
  sorry

end triangle_in_and_circumcircle_radius_l717_71799


namespace arctan_sum_eq_pi_div_4_l717_71794

noncomputable def n : ℤ := 27

theorem arctan_sum_eq_pi_div_4 :
  (Real.arctan (1 / 2) + Real.arctan (1 / 4) + Real.arctan (1 / 5) + Real.arctan (1 / n) = Real.pi / 4) :=
sorry

end arctan_sum_eq_pi_div_4_l717_71794


namespace nonoverlapping_unit_squares_in_figure_100_l717_71745

theorem nonoverlapping_unit_squares_in_figure_100 :
  ∃ f : ℕ → ℕ, (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 15 ∧ f 3 = 27) ∧ f 100 = 20203 :=
by
  sorry

end nonoverlapping_unit_squares_in_figure_100_l717_71745


namespace roots_of_equation_l717_71777

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l717_71777


namespace odd_function_value_at_neg_two_l717_71757

noncomputable def f : ℝ → ℝ :=
  λ x => if x > 0 then 2 * x - 3 else - (2 * (-x) - 3)

theorem odd_function_value_at_neg_two :
  (∀ x, f (-x) = -f x) → f (-2) = -1 :=
by
  intro odd_f
  sorry

end odd_function_value_at_neg_two_l717_71757


namespace percent_problem_l717_71784

theorem percent_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 :=
sorry

end percent_problem_l717_71784


namespace expenses_of_5_yuan_l717_71765

-- Define the given condition: income of 5 yuan is +5 yuan
def income (x : Int) : Int := x

-- Define the opposite relationship between income and expenses
def expenses (x : Int) : Int := -income x

-- Proof statement to show that expenses of 5 yuan are -5 yuan, given the above definitions
theorem expenses_of_5_yuan : expenses 5 = -5 := by
  -- The proof is not provided here, so we use sorry to indicate its place
  sorry

end expenses_of_5_yuan_l717_71765


namespace find_a_l717_71736

open Real

def is_chord_length_correct (a : ℝ) : Prop :=
  let x_line := fun t : ℝ => 1 + t
  let y_line := fun t : ℝ => a - t
  let x_circle := fun α : ℝ => 2 + 2 * cos α
  let y_circle := fun α : ℝ => 2 + 2 * sin α
  let distance_from_center := abs (3 - a) / sqrt 2
  let chord_length := 2 * sqrt (4 - distance_from_center ^ 2)
  chord_length = 2 * sqrt 2 

theorem find_a (a : ℝ) : is_chord_length_correct a → a = 1 ∨ a = 5 :=
by
  sorry

end find_a_l717_71736


namespace syllogism_correct_l717_71734

-- Hypotheses for each condition
def OptionA := "The first section, the second section, the third section"
def OptionB := "Major premise, minor premise, conclusion"
def OptionC := "Induction, conjecture, proof"
def OptionD := "Dividing the discussion into three sections"

-- Definition of a syllogism in deductive reasoning
def syllogism_def := "A logical argument that applies deductive reasoning to arrive at a conclusion based on two propositions assumed to be true"

-- Theorem stating that a syllogism corresponds to Option B
theorem syllogism_correct :
  syllogism_def = OptionB :=
by
  sorry

end syllogism_correct_l717_71734


namespace value_of_f_1_plus_g_4_l717_71741

def f (x : Int) : Int := 2 * x - 1
def g (x : Int) : Int := x + 1

theorem value_of_f_1_plus_g_4 : f (1 + g 4) = 11 := by
  sorry

end value_of_f_1_plus_g_4_l717_71741


namespace A_takes_4_hours_l717_71782

variables (A B C : ℝ)

-- Given conditions
axiom h1 : 1 / B + 1 / C = 1 / 2
axiom h2 : 1 / A + 1 / C = 1 / 2
axiom h3 : B = 4

-- What we need to prove: A = 4
theorem A_takes_4_hours :
  A = 4 := by
  sorry

end A_takes_4_hours_l717_71782


namespace solve_equation_l717_71717

theorem solve_equation : 361 + 2 * 19 * 6 + 36 = 625 := by
  sorry

end solve_equation_l717_71717


namespace at_least_one_does_not_land_l717_71776

/-- Proposition stating "A lands within the designated area". -/
def p : Prop := sorry

/-- Proposition stating "B lands within the designated area". -/
def q : Prop := sorry

/-- Negation of proposition p, stating "A does not land within the designated area". -/
def not_p : Prop := ¬p

/-- Negation of proposition q, stating "B does not land within the designated area". -/
def not_q : Prop := ¬q

/-- The proposition "At least one trainee does not land within the designated area" can be expressed as (¬p) ∨ (¬q). -/
theorem at_least_one_does_not_land : (¬p ∨ ¬q) := sorry

end at_least_one_does_not_land_l717_71776


namespace cody_ate_dumplings_l717_71795

theorem cody_ate_dumplings (initial_dumplings remaining_dumplings : ℕ) (h1 : initial_dumplings = 14) (h2 : remaining_dumplings = 7) : initial_dumplings - remaining_dumplings = 7 :=
by
  sorry

end cody_ate_dumplings_l717_71795


namespace geometric_sequence_common_ratio_l717_71787

theorem geometric_sequence_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h1 : a 2 = 1/2)
  (h2 : a 5 = 4)
  (h3 : ∀ n, a n = a 1 * q^(n - 1)) : 
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l717_71787


namespace race_dead_heat_l717_71761

variable (v_B v_A L x : ℝ)

theorem race_dead_heat (h : v_A = 17 / 14 * v_B) : x = 3 / 17 * L :=
by
  sorry

end race_dead_heat_l717_71761


namespace range_of_function_x_l717_71710

theorem range_of_function_x (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := sorry

end range_of_function_x_l717_71710


namespace quadratic_roots_l717_71785

theorem quadratic_roots (a : ℝ) (k c : ℝ) : 
    (∀ x : ℝ, 2 * x^2 + k * x + c = 0 ↔ (x = 7 ∨ x = a)) →
    k = -2 * a - 14 ∧ c = 14 * a :=
by
  sorry

end quadratic_roots_l717_71785


namespace extra_yellow_balls_dispatched_l717_71708

theorem extra_yellow_balls_dispatched : 
  ∀ (W Y E : ℕ), -- Declare natural numbers W, Y, E
  W = Y →      -- Condition that the number of white balls equals the number of yellow balls
  W + Y = 64 → -- Condition that the total number of originally ordered balls is 64
  W / (Y + E) = 8 / 13 → -- The given ratio involving the extra yellow balls
  E = 20 :=               -- Prove that the extra yellow balls E equals 20
by
  intros W Y E h1 h2 h3
  -- Proof mechanism here
  sorry

end extra_yellow_balls_dispatched_l717_71708


namespace value_of_expression_l717_71709

variable (a b : ℝ)

theorem value_of_expression : 
  let x := a + b 
  let y := a - b 
  (x - y) * (x + y) = 4 * a * b := 
by
  sorry

end value_of_expression_l717_71709


namespace student_chose_number_l717_71721

theorem student_chose_number : ∃ x : ℤ, 2 * x - 152 = 102 ∧ x = 127 :=
by
  sorry

end student_chose_number_l717_71721


namespace sam_money_left_l717_71712

-- Assuming the cost per dime and quarter
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Given conditions
def dimes : ℕ := 19
def quarters : ℕ := 6
def cost_per_candy_bar_in_dimes : ℕ := 3
def candy_bars : ℕ := 4
def lollipops : ℕ := 1

-- Calculate the initial money in cents
def initial_money : ℕ := (dimes * dime_value) + (quarters * quarter_value)

-- Calculate the cost of candy bars in cents
def candy_bars_cost : ℕ := candy_bars * cost_per_candy_bar_in_dimes * dime_value

-- Calculate the cost of lollipops in cents
def lollipop_cost : ℕ := lollipops * quarter_value

-- Calculate the total cost of purchases in cents
def total_cost : ℕ := candy_bars_cost + lollipop_cost

-- Calculate the final money left in cents
def final_money : ℕ := initial_money - total_cost

-- Theorem to prove
theorem sam_money_left : final_money = 195 := by
  sorry

end sam_money_left_l717_71712


namespace matrix_problem_l717_71738

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![6, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 8], ![3, -5]]
def RHS : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 2], ![15, -3]]

theorem matrix_problem : 
  2 • A + B = RHS :=
by
  sorry

end matrix_problem_l717_71738


namespace garden_length_to_width_ratio_l717_71719

theorem garden_length_to_width_ratio (area : ℕ) (width : ℕ) (h_area : area = 432) (h_width : width = 12) :
  ∃ length : ℕ, length = area / width ∧ (length / width = 3) := 
by
  sorry

end garden_length_to_width_ratio_l717_71719


namespace prob_red_or_blue_l717_71718

open Nat

noncomputable def total_marbles : Nat := 90
noncomputable def prob_white : (ℚ) := 1 / 6
noncomputable def prob_green : (ℚ) := 1 / 5

theorem prob_red_or_blue :
  let prob_total := 1
  let prob_white_or_green := prob_white + prob_green
  let prob_red_blue := prob_total - prob_white_or_green
  prob_red_blue = 19 / 30 := by
    sorry

end prob_red_or_blue_l717_71718


namespace solve_proof_problem_l717_71770

variables (a b c d : ℝ)

noncomputable def proof_problem : Prop :=
  a = 3 * b ∧ b = 3 * c ∧ c = 5 * d → (a * c) / (b * d) = 15

theorem solve_proof_problem : proof_problem a b c d :=
by
  sorry

end solve_proof_problem_l717_71770


namespace greatest_x_l717_71746

theorem greatest_x (x : ℕ) (h : x > 0 ∧ (x^4 / x^2 : ℚ) < 18) : x ≤ 4 :=
by
  sorry

end greatest_x_l717_71746


namespace setC_not_basis_l717_71716

-- Definitions based on the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e₁ e₂ : V)
variables (v₁ v₂ : V)

-- Assuming e₁ and e₂ are non-collinear
axiom non_collinear : ¬Collinear ℝ {e₁, e₂}

-- The vectors in the set C
def setC_v1 : V := 3 • e₁ - 2 • e₂
def setC_v2 : V := 4 • e₂ - 6 • e₁

-- The proof problem statement
theorem setC_not_basis : Collinear ℝ {setC_v1 e₁ e₂, setC_v2 e₁ e₂} :=
sorry

end setC_not_basis_l717_71716


namespace students_enrolled_in_all_three_l717_71790

variables {total_students at_least_one robotics_students dance_students music_students at_least_two_students all_three_students : ℕ}

-- Given conditions
axiom H1 : total_students = 25
axiom H2 : at_least_one = total_students
axiom H3 : robotics_students = 15
axiom H4 : dance_students = 12
axiom H5 : music_students = 10
axiom H6 : at_least_two_students = 11

-- We need to prove the number of students enrolled in all three workshops is 1
theorem students_enrolled_in_all_three : all_three_students = 1 :=
sorry

end students_enrolled_in_all_three_l717_71790


namespace five_x_plus_four_is_25_over_7_l717_71706

theorem five_x_plus_four_is_25_over_7 (x : ℚ) (h : 5 * x - 8 = 12 * x + 15) : 5 * (x + 4) = 25 / 7 := by
  sorry

end five_x_plus_four_is_25_over_7_l717_71706


namespace race_track_radius_l717_71724

theorem race_track_radius (C_inner : ℝ) (width : ℝ) (r_outer : ℝ) : 
  C_inner = 440 ∧ width = 14 ∧ r_outer = (440 / (2 * Real.pi) + 14) → r_outer = 84 :=
by
  intros
  sorry

end race_track_radius_l717_71724


namespace area_of_perpendicular_triangle_l717_71723

theorem area_of_perpendicular_triangle 
  (S R d : ℝ) (S' : ℝ) -- defining the variables and constants
  (h1 : S > 0) (h2 : R > 0) (h3 : d ≥ 0) :
  S' = (S / 4) * |1 - (d^2 / R^2)| := 
sorry

end area_of_perpendicular_triangle_l717_71723


namespace bullet_speed_difference_l717_71730

theorem bullet_speed_difference
  (horse_speed : ℝ := 20) 
  (bullet_speed : ℝ := 400) : 
  ((bullet_speed + horse_speed) - (bullet_speed - horse_speed) = 40) := by
  sorry

end bullet_speed_difference_l717_71730


namespace fourth_place_points_l717_71752

variables (x : ℕ)

def points_awarded (place : ℕ) : ℕ :=
  if place = 1 then 11
  else if place = 2 then 7
  else if place = 3 then 5
  else if place = 4 then x
  else 0

theorem fourth_place_points:
  (∃ a b c y u : ℕ, a + b + c + y + u = 7 ∧ points_awarded x 1 ^ a * points_awarded x 2 ^ b * points_awarded x 3 ^ c * points_awarded x 4 ^ y * 1 ^ u = 38500) →
  x = 4 :=
sorry

end fourth_place_points_l717_71752


namespace inequality_proof_l717_71726

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom abc_eq_one : a * b * c = 1

theorem inequality_proof :
  (1 + a * b) / (1 + a) + (1 + b * c) / (1 + b) + (1 + c * a) / (1 + c) ≥ 3 :=
by
  sorry

end inequality_proof_l717_71726


namespace mechanical_pencils_and_pens_price_l717_71735

theorem mechanical_pencils_and_pens_price
    (x y : ℝ)
    (h₁ : 7 * x + 6 * y = 46.8)
    (h₂ : 3 * x + 5 * y = 32.2) :
  x = 2.4 ∧ y = 5 :=
sorry

end mechanical_pencils_and_pens_price_l717_71735


namespace work_completion_times_l717_71737

-- Definitions based on conditions
def condition1 (x y : ℝ) : Prop := 2 * (1 / x) + 5 * (1 / y) = 1 / 2
def condition2 (x y : ℝ) : Prop := 3 * (1 / x + 1 / y) = 0.45

-- Main theorem stating the solution
theorem work_completion_times :
  ∃ (x y : ℝ), condition1 x y ∧ condition2 x y ∧ x = 12 ∧ y = 15 := 
sorry

end work_completion_times_l717_71737


namespace value_of_expression_l717_71700

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : 3 * m^2 + 3 * m + 2006 = 2009 :=
by
  sorry

end value_of_expression_l717_71700


namespace expression_bounds_l717_71704

theorem expression_bounds (a b c d : ℝ) (h0a : 0 ≤ a) (h1a : a ≤ 1) (h0b : 0 ≤ b) (h1b : b ≤ 1)
  (h0c : 0 ≤ c) (h1c : c ≤ 1) (h0d : 0 ≤ d) (h1d : d ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^4 + (1 - b^2)^2) +
    Real.sqrt (b^4 + (c^2 - b^2)^2) +
    Real.sqrt (c^4 + (d^2 - c^2)^2) +
    Real.sqrt (d^4 + (a^2 - d^2)^2) ∧
    Real.sqrt (a^4 + (1 - b^2)^2) +
    Real.sqrt (b^4 + (c^2 - b^2)^2) +
    Real.sqrt (c^4 + (d^2 - c^2)^2) +
    Real.sqrt (d^4 + (a^2 - d^2)^2) ≤ 4 :=
by sorry

end expression_bounds_l717_71704


namespace set_representation_equiv_l717_71755

open Nat

theorem set_representation_equiv :
  {x : ℕ | (0 < x) ∧ (x - 3 < 2)} = {1, 2, 3, 4} :=
by
  sorry

end set_representation_equiv_l717_71755


namespace sister_height_on_birthday_l717_71750

theorem sister_height_on_birthday (previous_height : ℝ) (growth_rate : ℝ)
    (h_previous_height : previous_height = 139.65)
    (h_growth_rate : growth_rate = 0.05) :
    previous_height * (1 + growth_rate) = 146.6325 :=
by
  -- Proof omitted
  sorry

end sister_height_on_birthday_l717_71750


namespace tensor_value_l717_71722

variables (h : ℝ)

def tensor (x y : ℝ) : ℝ := x^2 - y^2

theorem tensor_value : tensor h (tensor h h) = h^2 :=
by 
-- Complete proof body not required, 'sorry' is used for omitted proof
sorry

end tensor_value_l717_71722


namespace train_departure_at_10am_l717_71713

noncomputable def train_departure_time (distance travel_rate : ℕ) (arrival_time_chicago : ℕ) (time_difference : ℤ) : ℕ :=
  let travel_time := distance / travel_rate
  let arrival_time_ny := arrival_time_chicago + 1
  arrival_time_ny - travel_time

theorem train_departure_at_10am :
  train_departure_time 480 60 17 1 = 10 :=
by
  -- implementation of the proof will go here
  -- but we skip the proof as per the instructions
  sorry

end train_departure_at_10am_l717_71713


namespace polygon_num_sides_and_exterior_angle_l717_71789

theorem polygon_num_sides_and_exterior_angle 
  (n : ℕ) (x : ℕ) 
  (h : (n - 2) * 180 + x = 1350) 
  (hx : 0 < x ∧ x < 180) 
  : (n = 9) ∧ (x = 90) := 
by 
  sorry

end polygon_num_sides_and_exterior_angle_l717_71789


namespace Jimin_scabs_l717_71739

theorem Jimin_scabs (total_scabs : ℕ) (days_in_week : ℕ) (daily_scabs: ℕ)
  (h₁ : total_scabs = 220) (h₂ : days_in_week = 7) 
  (h₃ : daily_scabs = (total_scabs + days_in_week - 1) / days_in_week) : 
  daily_scabs ≥ 32 := by
  sorry

end Jimin_scabs_l717_71739


namespace allan_balloons_l717_71763

theorem allan_balloons (x : ℕ) : 
  (2 + x) + 1 = 6 → x = 3 :=
by
  intro h
  linarith

end allan_balloons_l717_71763


namespace student_comprehensive_score_l717_71742

def comprehensive_score (t_score i_score d_score : ℕ) (t_ratio i_ratio d_ratio : ℕ) :=
  (t_score * t_ratio + i_score * i_ratio + d_score * d_ratio) / (t_ratio + i_ratio + d_ratio)

theorem student_comprehensive_score :
  comprehensive_score 95 88 90 2 5 3 = 90 :=
by
  -- The proof goes here
  sorry

end student_comprehensive_score_l717_71742


namespace find_m_l717_71788

theorem find_m (θ₁ θ₂ : ℝ) (l : ℝ → ℝ) (m : ℕ) 
  (hθ₁ : θ₁ = Real.pi / 100) 
  (hθ₂ : θ₂ = Real.pi / 75)
  (hl : ∀ x, l x = x / 4) 
  (R : ((ℝ → ℝ) → (ℝ → ℝ)))
  (H_R : ∀ l, R l = (sorry : ℝ → ℝ)) 
  (R_n : ℕ → (ℝ → ℝ) → (ℝ → ℝ)) 
  (H_R1 : R_n 1 l = R l) 
  (H_Rn : ∀ n, R_n (n + 1) l = R (R_n n l)) :
  m = 1500 :=
sorry

end find_m_l717_71788


namespace number_of_8_digit_integers_l717_71725

theorem number_of_8_digit_integers : 
  ∃ n, n = 90000000 ∧ 
    (∀ (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
     d1 ≠ 0 → 0 ≤ d1 ∧ d1 ≤ 9 ∧ 
     0 ≤ d2 ∧ d2 ≤ 9 ∧ 
     0 ≤ d3 ∧ d3 ≤ 9 ∧ 
     0 ≤ d4 ∧ d4 ≤ 9 ∧ 
     0 ≤ d5 ∧ d5 ≤ 9 ∧ 
     0 ≤ d6 ∧ d6 ≤ 9 ∧ 
     0 ≤ d7 ∧ d7 ≤ 9 ∧ 
     0 ≤ d8 ∧ d8 ≤ 9 →
     ∀ count, count = (if d1 ≠ 0 then 9 * 10^7 else 0)) :=
sorry

end number_of_8_digit_integers_l717_71725


namespace find_number_of_raccoons_squirrels_opossums_l717_71772

theorem find_number_of_raccoons_squirrels_opossums
  (R : ℕ)
  (total_animals : ℕ)
  (number_of_squirrels : ℕ := 6 * R)
  (number_of_opossums : ℕ := 2 * R)
  (total : ℕ := R + number_of_squirrels + number_of_opossums) 
  (condition : total_animals = 168)
  (correct_total : total = total_animals) :
  ∃ R : ℕ, R + 6 * R + 2 * R = total_animals :=
by
  sorry

end find_number_of_raccoons_squirrels_opossums_l717_71772


namespace count_squares_within_region_l717_71766

noncomputable def countSquares : Nat := sorry

theorem count_squares_within_region :
  countSquares = 45 :=
sorry

end count_squares_within_region_l717_71766


namespace sum_a_b_c_l717_71748

theorem sum_a_b_c (a b c : ℕ) (h : a = 5 ∧ b = 10 ∧ c = 14) : a + b + c = 29 :=
by
  sorry

end sum_a_b_c_l717_71748


namespace passengers_on_ship_l717_71759

theorem passengers_on_ship :
  (∀ P : ℕ, 
    (P / 12) + (P / 8) + (P / 3) + (P / 6) + 35 = P) → P = 120 :=
by 
  sorry

end passengers_on_ship_l717_71759


namespace business_hours_correct_l717_71773

-- Define the business hours
def start_time : ℕ := 8 * 60 + 30   -- 8:30 in minutes
def end_time : ℕ := 22 * 60 + 30    -- 22:30 in minutes

-- Calculate total business hours in minutes and convert it to hours
def total_business_hours : ℕ := (end_time - start_time) / 60

-- State the business hour condition (which says the total business hour is 15 hours).
def business_hour_claim : ℕ := 15

-- Formulate the statement to prove: the claim that the total business hours are 15 hours is false.
theorem business_hours_correct : total_business_hours ≠ business_hour_claim := by
  sorry

end business_hours_correct_l717_71773


namespace Avery_builds_in_4_hours_l717_71733

variable (A : ℝ) (TomTime : ℝ := 2) (TogetherTime : ℝ := 1) (RemainingTomTime : ℝ := 0.5)

-- Conditions:
axiom Tom_builds_in_2_hours : TomTime = 2
axiom Work_together_for_1_hour : TogetherTime = 1
axiom Tom_finishes_in_0_5_hours : RemainingTomTime = 0.5

-- Question:
theorem Avery_builds_in_4_hours : A = 4 :=
by
  sorry

end Avery_builds_in_4_hours_l717_71733


namespace count_valid_choices_l717_71740

open Nat

def base4_representation (N : ℕ) : ℕ := 
  let a3 := N / 64 % 4
  let a2 := N / 16 % 4
  let a1 := N / 4 % 4
  let a0 := N % 4
  64 * a3 + 16 * a2 + 4 * a1 + a0

def base7_representation (N : ℕ) : ℕ := 
  let b3 := N / 343 % 7
  let b2 := N / 49 % 7
  let b1 := N / 7 % 7
  let b0 := N % 7
  343 * b3 + 49 * b2 + 7 * b1 + b0

def S (N : ℕ) : ℕ := base4_representation N + base7_representation N

def valid_choices (N : ℕ) : Prop := 
  (S N % 100) = (2 * N % 100)

theorem count_valid_choices : 
  ∃ (count : ℕ), count = 20 ∧ ∀ (N : ℕ), (N >= 1000 ∧ N < 10000) → valid_choices N ↔ (count = 20) :=
sorry

end count_valid_choices_l717_71740


namespace sam_new_books_not_signed_l717_71764

noncomputable def num_books_adventure := 13
noncomputable def num_books_mystery := 17
noncomputable def num_books_scifi := 25
noncomputable def num_books_nonfiction := 10
noncomputable def num_books_comics := 5
noncomputable def num_books_total := num_books_adventure + num_books_mystery + num_books_scifi + num_books_nonfiction + num_books_comics

noncomputable def num_books_used := 42
noncomputable def num_books_signed := 10
noncomputable def num_books_borrowed := 3
noncomputable def num_books_lost := 4

noncomputable def num_books_new := num_books_total - num_books_used
noncomputable def num_books_new_not_signed := num_books_new - num_books_signed
noncomputable def num_books_final := num_books_new_not_signed - num_books_lost

theorem sam_new_books_not_signed : num_books_final = 14 :=
by
  sorry

end sam_new_books_not_signed_l717_71764


namespace correct_operations_result_l717_71701

theorem correct_operations_result {n : ℕ} (h₁ : n / 8 - 20 = 12) :
  (n * 8 + 20) = 2068 ∧ 1800 < 2068 ∧ 2068 < 2200 :=
by
  sorry

end correct_operations_result_l717_71701


namespace problem_statement_l717_71768

theorem problem_statement (x y z : ℝ) :
    2 * x > y^2 + z^2 →
    2 * y > x^2 + z^2 →
    2 * z > y^2 + x^2 →
    x * y * z < 1 := by
  sorry

end problem_statement_l717_71768


namespace number_of_women_attended_l717_71729

theorem number_of_women_attended
  (m : ℕ) (w : ℕ)
  (men_dance_women : m = 15)
  (women_dance_men : ∀ i : ℕ, i < 15 → i * 4 = 60)
  (women_condition : w * 3 = 60) :
  w = 20 :=
sorry

end number_of_women_attended_l717_71729


namespace g_five_l717_71732

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_one : g 1 = 2

theorem g_five : g 5 = 10 :=
by sorry

end g_five_l717_71732


namespace cow_cost_calculation_l717_71753

theorem cow_cost_calculation (C cow calf : ℝ) 
  (h1 : cow = 8 * calf) 
  (h2 : cow + calf = 990) : 
  cow = 880 :=
by
  sorry

end cow_cost_calculation_l717_71753


namespace positive_difference_prime_factors_159137_l717_71751

-- Lean 4 Statement Following the Instructions
theorem positive_difference_prime_factors_159137 :
  (159137 = 11 * 17 * 23 * 37) → (37 - 23 = 14) :=
by
  intro h
  sorry -- Proof will be written here

end positive_difference_prime_factors_159137_l717_71751


namespace remainder_of_n_l717_71769

theorem remainder_of_n {n : ℕ} (h1 : n^2 ≡ 4 [MOD 7]) (h2 : n^3 ≡ 6 [MOD 7]): 
  n ≡ 5 [MOD 7] :=
sorry

end remainder_of_n_l717_71769


namespace sum_of_digits_divisible_by_45_l717_71749

theorem sum_of_digits_divisible_by_45 (a b : ℕ) (h1 : b = 0 ∨ b = 5) (h2 : (21 + a + b) % 9 = 0) : a + b = 6 :=
by
  sorry

end sum_of_digits_divisible_by_45_l717_71749


namespace remainders_equal_if_difference_divisible_l717_71775

theorem remainders_equal_if_difference_divisible (a b k : ℤ) (h : k ∣ (a - b)) : 
  a % k = b % k :=
sorry

end remainders_equal_if_difference_divisible_l717_71775


namespace sqrt_square_l717_71756

theorem sqrt_square (x : ℝ) (h_nonneg : 0 ≤ x) : (Real.sqrt x)^2 = x :=
by
  sorry

example : (Real.sqrt 25)^2 = 25 :=
by
  exact sqrt_square 25 (by norm_num)

end sqrt_square_l717_71756


namespace find_salary_l717_71798

variable (S : ℝ)
variable (house_rent_percentage : ℝ) (education_percentage : ℝ) (clothes_percentage : ℝ)
variable (remaining_amount : ℝ)

theorem find_salary (h1 : house_rent_percentage = 0.20)
                    (h2 : education_percentage = 0.10)
                    (h3 : clothes_percentage = 0.10)
                    (h4 : remaining_amount = 1377)
                    (h5 : (1 - clothes_percentage) * (1 - education_percentage) * (1 - house_rent_percentage) * S = remaining_amount) :
                    S = 2125 := 
sorry

end find_salary_l717_71798


namespace cars_gain_one_passenger_each_l717_71767

-- Conditions
def initial_people_per_car : ℕ := 3 -- 2 passengers + 1 driver
def total_cars : ℕ := 20
def total_people_at_end : ℕ := 80

-- Question (equivalent to "answer")
theorem cars_gain_one_passenger_each :
  (total_people_at_end = total_cars * initial_people_per_car + total_cars) →
  total_people_at_end - total_cars * initial_people_per_car = total_cars :=
by sorry

end cars_gain_one_passenger_each_l717_71767


namespace boss_total_amount_l717_71754

def number_of_staff : ℕ := 20
def rate_per_day : ℕ := 100
def number_of_days : ℕ := 30
def petty_cash_amount : ℕ := 1000

theorem boss_total_amount (number_of_staff : ℕ) (rate_per_day : ℕ) (number_of_days : ℕ) (petty_cash_amount : ℕ) :
  let total_allowance_one_staff := rate_per_day * number_of_days
  let total_allowance_all_staff := total_allowance_one_staff * number_of_staff
  total_allowance_all_staff + petty_cash_amount = 61000 := by
  sorry

end boss_total_amount_l717_71754


namespace correct_calc_value_l717_71771

theorem correct_calc_value (x : ℕ) (h : 2 * (3 * x + 14) = 946) : 2 * (x / 3 + 14) = 130 := 
by
  sorry

end correct_calc_value_l717_71771


namespace point_not_on_line_l717_71783

theorem point_not_on_line (m b : ℝ) (h1 : m > 2) (h2 : m * b > 0) : ¬ (b = -2023) :=
by
  sorry

end point_not_on_line_l717_71783


namespace find_smallest_nat_with_remainder_2_l717_71796

noncomputable def smallest_nat_with_remainder_2 : Nat :=
    let x := 26
    if x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] then x
    else 0

theorem find_smallest_nat_with_remainder_2 :
    ∃ x : Nat, x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] ∧ x = smallest_nat_with_remainder_2 :=
    sorry

end find_smallest_nat_with_remainder_2_l717_71796


namespace sum_over_positive_reals_nonnegative_l717_71714

theorem sum_over_positive_reals_nonnegative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (b + c - 2 * a) / (a^2 + b * c) + 
  (c + a - 2 * b) / (b^2 + c * a) + 
  (a + b - 2 * c) / (c^2 + a * b) ≥ 0 :=
sorry

end sum_over_positive_reals_nonnegative_l717_71714


namespace largest_value_of_x_l717_71747

theorem largest_value_of_x : 
  ∃ x, ( (15 * x^2 - 30 * x + 9) / (4 * x - 3) + 6 * x = 7 * x - 2 ) ∧ x = (19 + Real.sqrt 229) / 22 :=
sorry

end largest_value_of_x_l717_71747


namespace mixed_fraction_product_example_l717_71731

theorem mixed_fraction_product_example : 
  ∃ (X Y : ℕ), (5 + 1 / X) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 := 
by
  use 17
  use 8
  simp
  norm_num
  sorry

end mixed_fraction_product_example_l717_71731


namespace solve_inequality_l717_71778

theorem solve_inequality (a : ℝ) (x : ℝ) :
  (a = 0 → x > 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (0 < a → a < 2 → 1 < x → x < 2 / a → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a = 2 → False → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a > 2 → 2 / a < x → x < 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a < 0 → ((x < 2 / a ∨ x > 1) → (ax^2 - (a + 2) * x + 2 < 0))) := sorry

end solve_inequality_l717_71778


namespace no_negative_roots_l717_71780

theorem no_negative_roots (x : ℝ) : 4 * x^4 - 7 * x^3 - 20 * x^2 - 13 * x + 25 ≠ 0 ∨ x ≥ 0 := 
sorry

end no_negative_roots_l717_71780


namespace greatest_third_term_of_arithmetic_sequence_l717_71728

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h₁ : 0 < a)
  (h₂ : 0 < d) (h₃ : 4 * a + 6 * d = 50) : a + 2 * d = 16 :=
by
  -- Using the given condition
  -- 1. 4a + 6d = 50
  -- 2. a and d are in the naturals and greater than 0
  -- We prove that the greatest possible value of the third term (a + 2d)
  -- given these conditions equals 16
  sorry

end greatest_third_term_of_arithmetic_sequence_l717_71728


namespace exists_k_l717_71791

-- Definitions of the conditions
def sequence_def (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a (n+1) = Nat.lcm (a n) (a (n-1)) - Nat.lcm (a (n-1)) (a (n-2))

theorem exists_k (a : ℕ → ℕ) (a₁ a₂ a₃ : ℕ) (h₁ : a 1 = a₁) (h₂ : a 2 = a₂) (h₃ : a 3 = a₃)
  (h_seq : sequence_def a) : ∃ k : ℕ, k ≤ a₃ + 4 ∧ a k = 0 := 
sorry

end exists_k_l717_71791


namespace correct_calculation_l717_71760

-- Definitions for conditions
def cond_A (x y : ℝ) : Prop := 3 * x + 4 * y = 7 * x * y
def cond_B (x : ℝ) : Prop := 5 * x - 2 * x = 3 * x ^ 2
def cond_C (y : ℝ) : Prop := 7 * y ^ 2 - 5 * y ^ 2 = 2
def cond_D (a b : ℝ) : Prop := 6 * a ^ 2 * b - b * a ^ 2 = 5 * a ^ 2 * b

-- Proof statement using conditions
theorem correct_calculation (a b : ℝ) : cond_D a b :=
by
  unfold cond_D
  sorry

end correct_calculation_l717_71760


namespace distinct_real_roots_of_quadratic_find_m_and_other_root_l717_71797

theorem distinct_real_roots_of_quadratic (m : ℝ) (h_neg_m : m < 0) : 
    ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (∀ x, x^2 - 2*x + m = 0 → (x = x₁ ∨ x = x₂))) := 
by 
  sorry

theorem find_m_and_other_root (m : ℝ) (h_neg_m : m < 0) (root_minus_one : ∀ x, x^2 - 2*x + m = 0 → x = -1):
    m = -3 ∧ (∃ x, x^2 - 2*x - 3 = 0 ∧ x = 3) := 
by 
  sorry

end distinct_real_roots_of_quadratic_find_m_and_other_root_l717_71797


namespace periodic_sequences_zero_at_two_l717_71743

variable {R : Type*} [AddGroup R]

def seq_a (a b : ℕ → R) (n : ℕ) : Prop := a (n + 1) = a n + b n
def seq_b (b c : ℕ → R) (n : ℕ) : Prop := b (n + 1) = b n + c n
def seq_c (c d : ℕ → R) (n : ℕ) : Prop := c (n + 1) = c n + d n
def seq_d (d a : ℕ → R) (n : ℕ) : Prop := d (n + 1) = d n + a n

theorem periodic_sequences_zero_at_two
  (a b c d : ℕ → R)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (ha : ∀ n, seq_a a b n)
  (hb : ∀ n, seq_b b c n)
  (hc : ∀ n, seq_c c d n)
  (hd : ∀ n, seq_d d a n)
  (kra : a (k + m) = a m)
  (krb : b (k + m) = b m)
  (krc : c (k + m) = c m)
  (krd : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 := sorry

end periodic_sequences_zero_at_two_l717_71743


namespace area_percentage_change_is_neg_4_percent_l717_71774

noncomputable def percent_change_area (L W : ℝ) : ℝ :=
  let A_initial := L * W
  let A_new := (1.20 * L) * (0.80 * W)
  ((A_new - A_initial) / A_initial) * 100

theorem area_percentage_change_is_neg_4_percent (L W : ℝ) :
  percent_change_area L W = -4 :=
by
  sorry

end area_percentage_change_is_neg_4_percent_l717_71774


namespace proposition_p_is_false_iff_l717_71781

def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 3)

def p (a : ℝ) : Prop := ∃ x : ℝ, f x < a

theorem proposition_p_is_false_iff (a : ℝ) : (¬p a) ↔ (a < 5) :=
by sorry

end proposition_p_is_false_iff_l717_71781


namespace find_a2014_l717_71779

open Nat

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 0 ∧
  (∀ n, a (n + 1) = (a n - 2) / (5 * a n / 4 - 2))

theorem find_a2014 (a : ℕ → ℚ) (h : seq a) : a 2014 = 1 :=
by
  sorry

end find_a2014_l717_71779


namespace other_root_of_quadratic_l717_71793

theorem other_root_of_quadratic (p x : ℝ) (h : 7 * x^2 + p * x - 9 = 0) (root1 : x = -3) : 
  x = 3 / 7 :=
by
  sorry

end other_root_of_quadratic_l717_71793


namespace greatest_integer_third_side_l717_71727

-- Given two sides of a triangle measure 7 cm and 10 cm,
-- we need to prove that the greatest integer number of
-- centimeters that could be the third side is 16 cm.

theorem greatest_integer_third_side (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : 
    ∃ c : ℕ, c < a + b ∧ (∀ d : ℕ, d < a + b → d ≤ c) ∧ c = 16 := 
by
  sorry

end greatest_integer_third_side_l717_71727


namespace binom_10_3_eq_120_l717_71762

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l717_71762


namespace m_1_sufficient_but_not_necessary_l717_71715

def lines_parallel (m : ℝ) : Prop :=
  let l1_slope := -m
  let l2_slope := (2 - 3 * m) / m
  l1_slope = l2_slope

theorem m_1_sufficient_but_not_necessary (m : ℝ) (h₁ : lines_parallel m) : 
  (m = 1) → (∃ m': ℝ, lines_parallel m' ∧ m' ≠ 1) :=
sorry

end m_1_sufficient_but_not_necessary_l717_71715


namespace length_of_room_l717_71707

theorem length_of_room {L : ℝ} (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (h1 : width = 4)
  (h2 : cost_per_sqm = 750)
  (h3 : total_cost = 16500) :
  L = 5.5 ↔ (L * width) * cost_per_sqm = total_cost := 
by
  sorry

end length_of_room_l717_71707


namespace matrix_satisfies_conditions_l717_71786

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def matrix : List (List ℕ) :=
  [[6, 8, 9], [1, 7, 3], [4, 2, 5]]

noncomputable def sum_list (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def valid_matrix (matrix : List (List ℕ)) : Prop :=
  ∀ row_sum col_sum : ℕ, 
    (row_sum ∈ (matrix.map sum_list) ∧ is_prime row_sum) ∧
    (col_sum ∈ (List.transpose matrix).map sum_list ∧ is_prime col_sum)

theorem matrix_satisfies_conditions : valid_matrix matrix :=
by
  sorry

end matrix_satisfies_conditions_l717_71786


namespace inequality_proof_l717_71703

-- Define the main theorem with the conditions
theorem inequality_proof 
  (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧
  ((a = b ∧ b = c ∧ c = d) ↔ (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a)) := 
sorry

end inequality_proof_l717_71703


namespace solve_quadratic_l717_71720

theorem solve_quadratic (x : ℝ) (h_pos : x > 0) (h_eq : 5 * x ^ 2 + 9 * x - 18 = 0) : x = 6 / 5 :=
by
  sorry

end solve_quadratic_l717_71720


namespace prize_behind_door_4_eq_a_l717_71711

theorem prize_behind_door_4_eq_a :
  ∀ (prize : ℕ → ℕ)
    (h_prizes : ∀ i j, 1 ≤ prize i ∧ prize i ≤ 4 ∧ prize i = prize j → i = j)
    (hA1 : prize 1 = 2)
    (hA2 : prize 3 = 3)
    (hB1 : prize 2 = 2)
    (hB2 : prize 3 = 4)
    (hC1 : prize 4 = 2)
    (hC2 : prize 2 = 3)
    (hD1 : prize 4 = 1)
    (hD2 : prize 3 = 3),
    prize 4 = 1 :=
by
  intro prize h_prizes hA1 hA2 hB1 hB2 hC1 hC2 hD1 hD2
  sorry

end prize_behind_door_4_eq_a_l717_71711


namespace transmitted_word_is_PAROHOD_l717_71702

-- Define the binary representation of each letter in the Russian alphabet.
def binary_repr : String → String
| "А" => "00000"
| "Б" => "00001"
| "В" => "00011"
| "Г" => "00111"
| "Д" => "00101"
| "Е" => "00110"
| "Ж" => "01100"
| "З" => "01011"
| "И" => "01001"
| "Й" => "11000"
| "К" => "01010"
| "Л" => "01011"
| "М" => "01101"
| "Н" => "01111"
| "О" => "01100"
| "П" => "01110"
| "Р" => "01010"
| "С" => "01100"
| "Т" => "01001"
| "У" => "01111"
| "Ф" => "11101"
| "Х" => "11011"
| "Ц" => "11100"
| "Ч" => "10111"
| "Ш" => "11110"
| "Щ" => "11110"
| "Ь" => "00010"
| "Ы" => "00011"
| "Ъ" => "00101"
| "Э" => "11100"
| "Ю" => "01111"
| "Я" => "11111"
| _  => "00000" -- default case

-- Define the received scrambled word.
def received_word : List String := ["Э", "А", "В", "Щ", "О", "Щ", "И"]

-- The target transmitted word is "ПАРОХОД" which corresponds to ["П", "А", "Р", "О", "Х", "О", "Д"]
def transmitted_word : List String := ["П", "А", "Р", "О", "Х", "О", "Д"]

-- Lean 4 proof statement to show that the received scrambled word reconstructs to the transmitted word.
theorem transmitted_word_is_PAROHOD (b_repr : String → String)
(received : List String) :
  received = received_word →
  transmitted_word.map b_repr = received.map b_repr → transmitted_word = ["П", "А", "Р", "О", "Х", "О", "Д"] :=
by 
  intros h_received h_repr_eq
  exact sorry

end transmitted_word_is_PAROHOD_l717_71702


namespace cost_equal_at_60_l717_71792

variable (x : ℝ)

def PlanA_cost (x : ℝ) : ℝ := 0.25 * x + 9
def PlanB_cost (x : ℝ) : ℝ := 0.40 * x

theorem cost_equal_at_60 : PlanA_cost x = PlanB_cost x → x = 60 :=
by
  intro h
  sorry

end cost_equal_at_60_l717_71792


namespace hexagon_ratio_identity_l717_71744

theorem hexagon_ratio_identity
  (A B C D E F : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (AB BC CD DE EF FA : ℝ)
  (angle_B angle_D angle_F : ℝ)
  (h1 : AB / BC * CD / DE * EF / FA = 1)
  (h2 : angle_B + angle_D + angle_F = 360) :
  (BC / AC * AE / EF * FD / DB = 1) := sorry

end hexagon_ratio_identity_l717_71744


namespace arlo_stationery_count_l717_71758

theorem arlo_stationery_count (books pens : ℕ) (ratio_books_pens : ℕ × ℕ) (total_books : ℕ)
  (h_ratio : ratio_books_pens = (7, 3)) (h_books : total_books = 280) :
  books + pens = 400 :=
by
  sorry

end arlo_stationery_count_l717_71758
