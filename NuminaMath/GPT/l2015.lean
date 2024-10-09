import Mathlib

namespace range_of_x_l2015_201516

noncomputable def f (x : ℝ) : ℝ := sorry -- Define f to satisfy given conditions later

theorem range_of_x (hf_odd : ∀ x : ℝ, f (-x) = - f x)
                   (hf_inc_mono_neg : ∀ x y : ℝ, x ≤ y → y ≤ 0 → f x ≤ f y)
                   (h_ineq : f 1 + f (Real.log x - 2) < 0) : (0 < x) ∧ (x < 10) :=
by
  sorry

end range_of_x_l2015_201516


namespace car_pedestrian_speed_ratio_l2015_201592

theorem car_pedestrian_speed_ratio
  (L : ℝ) -- Length of the bridge
  (v_p v_c : ℝ) -- Speed of pedestrian and car
  (h1 : (4 / 9) * L / v_p = (5 / 9) * L / v_p + (5 / 9) * L / v_c) -- Initial meet at bridge start
  (h2 : (4 / 9) * L / v_p = (8 / 9) * L / v_c) -- If pedestrian continues to walk
  : v_c / v_p = 9 :=
sorry

end car_pedestrian_speed_ratio_l2015_201592


namespace negation_of_exists_x_quad_eq_zero_l2015_201595

theorem negation_of_exists_x_quad_eq_zero :
  ¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0 ↔ ∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0 :=
by sorry

end negation_of_exists_x_quad_eq_zero_l2015_201595


namespace domain_of_f_univ_l2015_201510

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 1)^(1 / 3) + (9 - x^2)^(1 / 3)

theorem domain_of_f_univ : ∀ x : ℝ, true :=
by
  intro x
  sorry

end domain_of_f_univ_l2015_201510


namespace exists_large_p_l2015_201543

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (Real.pi * x)

theorem exists_large_p (d : ℝ) (h : d > 0) : ∃ p : ℝ, ∀ x : ℝ, |f (x + p) - f x| < d ∧ ∃ M : ℝ, M > 0 ∧ p > M :=
by {
  sorry
}

end exists_large_p_l2015_201543


namespace pentagon_area_l2015_201546

/-- This Lean statement represents the problem of finding the y-coordinate of vertex C
    in a pentagon with given vertex positions and specific area constraint. -/
theorem pentagon_area (y : ℝ) 
  (h_sym : true) -- The pentagon ABCDE has a vertical line of symmetry
  (h_A : (0, 0) = (0, 0)) -- A(0,0)
  (h_B : (0, 5) = (0, 5)) -- B(0, 5)
  (h_C : (3, y) = (3, y)) -- C(3, y)
  (h_D : (6, 5) = (6, 5)) -- D(6, 5)
  (h_E : (6, 0) = (6, 0)) -- E(6, 0)
  (h_area : 50 = 50) -- The total area of the pentagon is 50 square units
  : y = 35 / 3 :=
sorry

end pentagon_area_l2015_201546


namespace combinedHeightOfBuildingsIsCorrect_l2015_201587

-- Define the heights to the top floor of the buildings (in feet)
def empireStateBuildingHeightFeet : Float := 1250
def willisTowerHeightFeet : Float := 1450
def oneWorldTradeCenterHeightFeet : Float := 1368

-- Define the antenna heights of the buildings (in feet)
def empireStateBuildingAntennaFeet : Float := 204
def willisTowerAntennaFeet : Float := 280
def oneWorldTradeCenterAntennaFeet : Float := 408

-- Define the conversion factor from feet to meters
def feetToMeters : Float := 0.3048

-- Calculate the total heights of the buildings in meters
def empireStateBuildingTotalHeightMeters : Float := (empireStateBuildingHeightFeet + empireStateBuildingAntennaFeet) * feetToMeters
def willisTowerTotalHeightMeters : Float := (willisTowerHeightFeet + willisTowerAntennaFeet) * feetToMeters
def oneWorldTradeCenterTotalHeightMeters : Float := (oneWorldTradeCenterHeightFeet + oneWorldTradeCenterAntennaFeet) * feetToMeters

-- Calculate the combined total height of the three buildings in meters
def combinedTotalHeightMeters : Float :=
  empireStateBuildingTotalHeightMeters + willisTowerTotalHeightMeters + oneWorldTradeCenterTotalHeightMeters

-- The statement to prove
theorem combinedHeightOfBuildingsIsCorrect : combinedTotalHeightMeters = 1511.8164 := by
  sorry

end combinedHeightOfBuildingsIsCorrect_l2015_201587


namespace min_colors_for_grid_coloring_l2015_201597

theorem min_colors_for_grid_coloring : ∃c : ℕ, c = 4 ∧ (∀ (color : ℕ × ℕ → ℕ), 
  (∀ i j : ℕ, i < 5 ∧ j < 5 → 
     ((i < 4 → color (i, j) ≠ color (i+1, j+1)) ∧ 
      (j < 4 → color (i, j) ≠ color (i+1, j-1))) ∧ 
     ((i > 0 → color (i, j) ≠ color (i-1, j-1)) ∧ 
      (j > 0 → color (i, j) ≠ color (i-1, j+1)))) → 
  c = 4) :=
sorry

end min_colors_for_grid_coloring_l2015_201597


namespace Q_cannot_be_log_x_l2015_201515

def P : Set ℝ := {y | y ≥ 0}

theorem Q_cannot_be_log_x (Q : Set ℝ) :
  (P ∩ Q = Q) → Q ≠ {y | ∃ x, y = Real.log x} :=
by
  sorry

end Q_cannot_be_log_x_l2015_201515


namespace trapezoid_circle_tangent_ratio_l2015_201529

/-- Given trapezoid EFGH with specified side lengths,
    where EF is parallel to GH, and a circle with
    center Q on EF tangent to FG and HE,
    the ratio EQ : QF is 12 : 37. -/
theorem trapezoid_circle_tangent_ratio :
  ∀ (EF FG GH HE : ℝ) (EQ QF : ℝ),
  EF = 40 → FG = 25 → GH = 12 → HE = 35 →
  ∃ (Q : ℝ) (EQ QF : ℝ),
  EQ + QF = EF ∧ EQ / QF = 12 / 37 ∧ gcd 12 37 = 1 :=
by
  sorry

end trapezoid_circle_tangent_ratio_l2015_201529


namespace stickers_left_after_giving_away_l2015_201563

/-- Willie starts with 36 stickers and gives 7 to Emily. 
    We want to prove that Willie ends up with 29 stickers. -/
theorem stickers_left_after_giving_away (init_stickers : ℕ) (given_away : ℕ) (end_stickers : ℕ) : 
  init_stickers = 36 ∧ given_away = 7 → end_stickers = init_stickers - given_away → end_stickers = 29 :=
by
  intro h
  sorry

end stickers_left_after_giving_away_l2015_201563


namespace base_eight_to_base_ten_l2015_201571

theorem base_eight_to_base_ten (n : ℕ) (h : n = 4 * 8^1 + 7 * 8^0) : n = 39 := by
  sorry

end base_eight_to_base_ten_l2015_201571


namespace probability_of_event_l2015_201556

open Set Real

noncomputable def probability_event_interval (x : ℝ) : Prop :=
  1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3

noncomputable def interval := Icc (0 : ℝ) (3 : ℝ)

noncomputable def event_probability := 1 / 3

theorem probability_of_event :
  ∀ x ∈ interval, probability_event_interval x → (event_probability) = 1 / 3 :=
by
  sorry

end probability_of_event_l2015_201556


namespace find_cashew_kilos_l2015_201565

variables (x : ℕ)

def cashew_cost_per_kilo := 210
def peanut_cost_per_kilo := 130
def total_weight := 5
def peanuts_weight := 2
def avg_price_per_kilo := 178

-- Given conditions
def cashew_total_cost := cashew_cost_per_kilo * x
def peanut_total_cost := peanut_cost_per_kilo * peanuts_weight
def total_price := total_weight * avg_price_per_kilo

theorem find_cashew_kilos (h1 : cashew_total_cost + peanut_total_cost = total_price) : x = 3 :=
by
  sorry

end find_cashew_kilos_l2015_201565


namespace correct_number_of_students_answered_both_l2015_201589

def students_enrolled := 25
def answered_q1_correctly := 22
def answered_q2_correctly := 20
def not_taken_test := 3

def students_answered_both_questions_correctly : Nat :=
  let students_took_test := students_enrolled - not_taken_test
  let b := answered_q2_correctly
  b

theorem correct_number_of_students_answered_both :
  students_answered_both_questions_correctly = answered_q2_correctly :=
by {
  -- this space is for the proof, we are currently not required to provide it
  sorry
}

end correct_number_of_students_answered_both_l2015_201589


namespace number_of_possible_a_values_l2015_201534

-- Define the function f(x)
def f (a x : ℝ) := abs (x + 1) + abs (a * x + 1)

-- Define the condition for the minimum value
def minimum_value_of_f (a : ℝ) := ∃ x : ℝ, f a x = (3 / 2)

-- The proof problem statement
theorem number_of_possible_a_values : 
  (∃ (a1 a2 a3 a4 : ℝ),
    minimum_value_of_f a1 ∧
    minimum_value_of_f a2 ∧
    minimum_value_of_f a3 ∧
    minimum_value_of_f a4 ∧
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :=
sorry

end number_of_possible_a_values_l2015_201534


namespace cos_pi_minus_alpha_l2015_201502

theorem cos_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : 0 < α ∧ α < π / 2) :
  Real.cos (π - α) = -12 / 13 :=
sorry

end cos_pi_minus_alpha_l2015_201502


namespace circle_equation_through_origin_l2015_201593

theorem circle_equation_through_origin (focus : ℝ × ℝ) (radius : ℝ) (x y : ℝ) 
  (h1 : focus = (1, 0)) 
  (h2 : (x - 1)^2 + y^2 = radius^2) : 
  x^2 + y^2 - 2*x = 0 :=
by
  sorry

end circle_equation_through_origin_l2015_201593


namespace num_non_fiction_books_l2015_201532

-- Definitions based on the problem conditions
def num_fiction_configurations : ℕ := 24
def total_configurations : ℕ := 36

-- Non-computable definition for factorial
noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

-- Theorem to prove the number of new non-fiction books
theorem num_non_fiction_books (n : ℕ) :
  num_fiction_configurations * factorial n = total_configurations → n = 2 :=
by
  sorry

end num_non_fiction_books_l2015_201532


namespace sum_ages_l2015_201513

theorem sum_ages (x : ℕ) (h_triple : True) (h_sons_age : ∀ a, a ∈ [16, 16, 16]) (h_beau_age : 42 = 42) :
  3 * (16 - x) = 42 - x → x = 3 := by
  sorry

end sum_ages_l2015_201513


namespace proposition_form_l2015_201550

-- Definitions based on the conditions
def p : Prop := (12 % 4 = 0)
def q : Prop := (12 % 3 = 0)

-- Problem statement to prove
theorem proposition_form : p ∧ q :=
by
  sorry

end proposition_form_l2015_201550


namespace find_parts_per_hour_find_min_A_machines_l2015_201517

-- Conditions
variable (x y : ℕ) -- x is parts per hour by B, y is parts per hour by A

-- Definitions based on conditions
def machineA_speed_relation (x y : ℕ) : Prop :=
  y = x + 2

def time_relation (x y : ℕ) : Prop :=
  80 / y = 60 / x

def min_A_machines (x y : ℕ) (m : ℕ) : Prop :=
  8 * m + 6 * (10 - m) ≥ 70

-- Problem statements
theorem find_parts_per_hour (x y : ℕ) (h1 : machineA_speed_relation x y) (h2 : time_relation x y) :
  x = 6 ∧ y = 8 :=
sorry

theorem find_min_A_machines (m : ℕ) (h1 : machineA_speed_relation 6 8) (h2 : time_relation 6 8) (h3 : min_A_machines 6 8 m) :
  m ≥ 5 :=
sorry

end find_parts_per_hour_find_min_A_machines_l2015_201517


namespace intersection_claim_union_claim_l2015_201554

def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}
def B : Set ℝ := {x | x^2 - 2 * x - 3 ≥ 0}
def U : Set ℝ := Set.univ

-- Claim 1: Prove that A ∩ B = {x | -5 < x ∧ x ≤ -1}
theorem intersection_claim : A ∩ B = {x | -5 < x ∧ x ≤ -1} :=
by
  sorry

-- Claim 2: Prove that A ∪ (U \ B) = {x | -5 < x ∧ x < 3}
theorem union_claim : A ∪ (U \ B) = {x | -5 < x ∧ x < 3} :=
by
  sorry

end intersection_claim_union_claim_l2015_201554


namespace largest_band_members_l2015_201568

theorem largest_band_members
  (p q m : ℕ)
  (h1 : p * q + 3 = m)
  (h2 : (q + 1) * (p + 2) = m)
  (h3 : m < 120) :
  m = 119 :=
sorry

end largest_band_members_l2015_201568


namespace find_f_2010_l2015_201537

noncomputable def f (a b α β : ℝ) (x : ℝ) :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem find_f_2010 {a b α β : ℝ} (h : f a b α β 2009 = 5) : f a b α β 2010 = 3 :=
sorry

end find_f_2010_l2015_201537


namespace graphs_symmetric_y_axis_l2015_201542

theorem graphs_symmetric_y_axis : ∀ (x : ℝ), (-x) ∈ { y | y = 3^(-x) } ↔ x ∈ { y | y = 3^x } :=
by
  intro x
  sorry

end graphs_symmetric_y_axis_l2015_201542


namespace Alan_collected_48_shells_l2015_201511

def Laurie_shells : ℕ := 36
def Ben_shells : ℕ := Laurie_shells / 3
def Alan_shells : ℕ := 4 * Ben_shells

theorem Alan_collected_48_shells :
  Alan_shells = 48 :=
by
  sorry

end Alan_collected_48_shells_l2015_201511


namespace negate_exists_real_l2015_201518

theorem negate_exists_real (h : ¬ ∃ x : ℝ, x^2 - 2 ≤ 0) : ∀ x : ℝ, x^2 - 2 > 0 :=
by
  sorry

end negate_exists_real_l2015_201518


namespace isosceles_trapezoid_ratio_l2015_201567

theorem isosceles_trapezoid_ratio (a b d_E d_G : ℝ) (h1 : a > b)
  (h2 : (1/2) * b * d_G = 3) (h3 : (1/2) * a * d_E = 7)
  (h4 : (1/2) * (a + b) * (d_E + d_G) = 24) :
  (a / b) = 7 / 3 :=
sorry

end isosceles_trapezoid_ratio_l2015_201567


namespace polygon_is_decagon_l2015_201538

-- Definitions based on conditions
def exterior_angles_sum (x : ℕ) : ℝ := 360

def interior_angles_sum (x : ℕ) : ℝ := 4 * exterior_angles_sum x

def interior_sum_formula (n : ℕ) : ℝ := (n - 2) * 180

-- Mathematically equivalent proof problem
theorem polygon_is_decagon (n : ℕ) (h1 : exterior_angles_sum n = 360)
  (h2 : interior_angles_sum n = 4 * exterior_angles_sum n)
  (h3 : interior_sum_formula n = interior_angles_sum n) : n = 10 :=
sorry

end polygon_is_decagon_l2015_201538


namespace distinct_four_digit_odd_numbers_l2015_201573

-- Define the conditions as Lean definitions
def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def valid_first_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

-- The proposition we want to prove
theorem distinct_four_digit_odd_numbers (n : ℕ) :
  (∀ d, d ∈ [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] → is_odd_digit d) →
  valid_first_digit (n / 1000 % 10) →
  1000 ≤ n ∧ n < 10000 →
  n = 500 :=
sorry

end distinct_four_digit_odd_numbers_l2015_201573


namespace vertex_angle_measure_l2015_201590

-- Definitions for Lean Proof
def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (α = γ) ∨ (β = γ)
def exterior_angle (interior exterior : ℝ) : Prop := interior + exterior = 180

-- Conditions from the problem
variables (α β γ : ℝ)
variable (ext_angle : ℝ := 110)

-- Lean 4 statement: The measure of the vertex angle is 70° or 40°
theorem vertex_angle_measure :
  isosceles_triangle α β γ ∧
  (exterior_angle γ ext_angle ∨ exterior_angle α ext_angle ∨ exterior_angle β ext_angle) →
  (γ = 70 ∨ γ = 40) :=
by
  sorry

end vertex_angle_measure_l2015_201590


namespace water_usage_difference_l2015_201525

variable (a b : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (ha_plus_4 : a + 4 ≠ 0)

theorem water_usage_difference :
  b / a - b / (a + 4) = 4 * b / (a * (a + 4)) :=
by
  sorry

end water_usage_difference_l2015_201525


namespace possible_c_value_l2015_201514

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem possible_c_value (a b c : ℝ) 
  (h1 : f (-1) a b c = f (-2) a b c) 
  (h2 : f (-2) a b c = f (-3) a b c) 
  (h3 : 0 ≤ f (-1) a b c) 
  (h4 : f (-1) a b c ≤ 3) : 
  6 ≤ c ∧ c ≤ 9 := sorry

end possible_c_value_l2015_201514


namespace latest_first_pump_time_l2015_201562

theorem latest_first_pump_time 
  (V : ℝ) -- Volume of the pool
  (x y : ℝ) -- Productivity of first and second pumps respectively
  (t : ℝ) -- Time of operation of the first pump until the second pump is turned on
  (h1 : 2*x + 2*y = V/2) -- Condition from 10 AM to 12 PM
  (h2 : 5*x + 5*y = V/2) -- Condition from 12 PM to 5 PM
  (h3 : t*x + 2*x + 2*y = V/2) -- Condition for early morning until 12 PM
  (hx_pos : 0 < x) -- Assume productivity of first pump is positive
  (hy_pos : 0 < y) -- Assume productivity of second pump is positive
  : t ≥ 3 :=
by
  -- The proof goes here...
  sorry

end latest_first_pump_time_l2015_201562


namespace min_value_S_max_value_S_l2015_201570

theorem min_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≥ -512 := 
sorry

theorem max_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288 := 
sorry

end min_value_S_max_value_S_l2015_201570


namespace bet_final_result_l2015_201577

theorem bet_final_result :
  let M₀ := 64
  let final_money := (3 / 2) ^ 3 * (1 / 2) ^ 3 * M₀
  final_money = 27 ∧ M₀ - final_money = 37 :=
by
  sorry

end bet_final_result_l2015_201577


namespace distance_B_amusement_park_l2015_201599

variable (d_A d_B v_A v_B t_A t_B : ℝ)

axiom h1 : v_A = 3
axiom h2 : v_B = 4
axiom h3 : d_B = d_A + 2
axiom h4 : t_A + t_B = 4
axiom h5 : t_A = d_A / v_A
axiom h6 : t_B = d_B / v_B

theorem distance_B_amusement_park:
  d_A / 3 + (d_A + 2) / 4 = 4 → d_B = 8 :=
by
  sorry

end distance_B_amusement_park_l2015_201599


namespace fibonacci_sum_of_squares_l2015_201545

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_of_squares (n : ℕ) (hn : n ≥ 1) :
  (Finset.range n).sum (λ i => (fibonacci (i + 1))^2) = fibonacci n * fibonacci (n + 1) :=
sorry

end fibonacci_sum_of_squares_l2015_201545


namespace line_in_slope_intercept_form_l2015_201581

-- Given the condition
def line_eq (x y : ℝ) : Prop :=
  (2 * (x - 3)) - (y + 4) = 0

-- Prove that the line equation can be expressed as y = 2x - 10.
theorem line_in_slope_intercept_form (x y : ℝ) :
  line_eq x y ↔ y = 2 * x - 10 := 
sorry

end line_in_slope_intercept_form_l2015_201581


namespace b_is_square_of_positive_integer_l2015_201522

theorem b_is_square_of_positive_integer 
  (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (h : b^2 = a^2 + ab + b) : 
  ∃ k : ℕ, b = k^2 := 
by 
  sorry

end b_is_square_of_positive_integer_l2015_201522


namespace joan_total_cost_is_correct_l2015_201596

def year1_home_games := 6
def year1_away_games := 3
def year1_home_playoff_games := 1
def year1_away_playoff_games := 1

def year2_home_games := 2
def year2_away_games := 2
def year2_home_playoff_games := 1
def year2_away_playoff_games := 0

def home_game_ticket := 60
def away_game_ticket := 75
def home_playoff_ticket := 120
def away_playoff_ticket := 100

def friend_home_game_ticket := 45
def friend_away_game_ticket := 75

def home_game_transportation := 25
def away_game_transportation := 50

noncomputable def year1_total_cost : ℕ :=
  (year1_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year1_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year1_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def year2_total_cost : ℕ :=
  (year2_home_games * (home_game_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_games * (away_game_ticket + friend_away_game_ticket + away_game_transportation)) +
  (year2_home_playoff_games * (home_playoff_ticket + friend_home_game_ticket + home_game_transportation)) +
  (year2_away_playoff_games * (away_playoff_ticket + friend_away_game_ticket + away_game_transportation))

noncomputable def total_cost : ℕ := year1_total_cost + year2_total_cost

theorem joan_total_cost_is_correct : total_cost = 2645 := by
  sorry

end joan_total_cost_is_correct_l2015_201596


namespace quotient_is_36_l2015_201557

-- Conditions
def divisor := 85
def remainder := 26
def dividend := 3086

-- The Question and Answer (proof required)
theorem quotient_is_36 (quotient : ℕ) (h : dividend = (divisor * quotient) + remainder) : quotient = 36 := by 
  sorry

end quotient_is_36_l2015_201557


namespace sticks_form_triangle_l2015_201583

theorem sticks_form_triangle:
  (2 + 3 > 4) ∧ (2 + 4 > 3) ∧ (3 + 4 > 2) := by
  sorry

end sticks_form_triangle_l2015_201583


namespace complement_of_A_l2015_201544

open Set

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {3, 4, 5}) :
  (U \ A) = {1, 2, 6} :=
by
  sorry

end complement_of_A_l2015_201544


namespace fraction_to_decimal_l2015_201509

theorem fraction_to_decimal : (3 : ℚ) / 80 = 0.0375 :=
by
  sorry

end fraction_to_decimal_l2015_201509


namespace sin_cos_sum_l2015_201512

open Real

theorem sin_cos_sum : sin (47 : ℝ) * cos (43 : ℝ) + cos (47 : ℝ) * sin (43 : ℝ) = 1 :=
by
  sorry

end sin_cos_sum_l2015_201512


namespace work_finish_in_3_days_l2015_201539

-- Define the respective rates of work
def A_rate := 1/4
def B_rate := 1/14
def C_rate := 1/7

-- Define the duration they start working together
def initial_duration := 2
def after_C_joining := 1 -- time after C joins before A leaves

-- From the third day, consider A leaving the job
theorem work_finish_in_3_days :
  (initial_duration * (A_rate + B_rate)) + 
  (after_C_joining * (A_rate + B_rate + C_rate)) + 
  ((1 : ℝ) - after_C_joining) * (B_rate + C_rate) >= 1 :=
by
  sorry

end work_finish_in_3_days_l2015_201539


namespace find_unique_function_l2015_201582

theorem find_unique_function (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end find_unique_function_l2015_201582


namespace inequality_solution_b_range_l2015_201561

-- Given conditions
variables (a b : ℝ)

def condition1 : Prop := (1 - a < 0) ∧ (a = 3)
def condition2 : Prop := ∀ (x : ℝ), (3 * x^2 + b * x + 3) ≥ 0

-- Assertions to be proved
theorem inequality_solution (a : ℝ) (ha : condition1 a) : 
  ∀ (x : ℝ), (2 * x^2 + (2 - a) * x - a > 0) ↔ (x < -1 ∨ x > 3/2) :=
sorry

theorem b_range (a : ℝ) (hb : condition1 a) : 
  condition2 b ↔ (-6 ≤ b ∧ b ≤ 6) :=
sorry

end inequality_solution_b_range_l2015_201561


namespace factors_2310_l2015_201533

theorem factors_2310 : ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p) ∧ S.card = 5 ∧ (2310 = S.prod id) :=
by
  sorry

end factors_2310_l2015_201533


namespace integer_value_of_a_l2015_201551

theorem integer_value_of_a (a x y z k : ℤ) :
  (x = k) ∧ (y = 4 * k) ∧ (z = 5 * k) ∧ (y = 9 * a^2 - 2 * a - 8) ∧ (z = 10 * a + 2) → a = 5 :=
by 
  sorry

end integer_value_of_a_l2015_201551


namespace problem_solution_l2015_201594

noncomputable def f (x : ℝ) : ℝ := x / (Real.cos x)

variables (x1 x2 x3 : ℝ)

axiom a1 : |x1| < (Real.pi / 2)
axiom a2 : |x2| < (Real.pi / 2)
axiom a3 : |x3| < (Real.pi / 2)

axiom h1 : f x1 + f x2 ≥ 0
axiom h2 : f x2 + f x3 ≥ 0
axiom h3 : f x3 + f x1 ≥ 0

theorem problem_solution : f (x1 + x2 + x3) ≥ 0 := sorry

end problem_solution_l2015_201594


namespace complement_union_l2015_201575

theorem complement_union (U A B complement_U_A : Set Int) (hU : U = {-1, 0, 1, 2}) 
  (hA : A = {-1, 2}) (hB : B = {0, 2}) (hC : complement_U_A = {0, 1}) :
  complement_U_A ∪ B = {0, 1, 2} := by
  sorry

end complement_union_l2015_201575


namespace area_of_shape_l2015_201503

def points := [(0, 1), (1, 2), (3, 2), (4, 1), (2, 0)]

theorem area_of_shape : 
  let I := 6 -- Number of interior points
  let B := 5 -- Number of boundary points
  ∃ (A : ℝ), A = I + B / 2 - 1 ∧ A = 7.5 := 
  by
    use 7.5
    simp
    sorry

end area_of_shape_l2015_201503


namespace subtract_500_from_sum_of_calculations_l2015_201541

theorem subtract_500_from_sum_of_calculations (x : ℕ) (h : 423 - x = 421) : 
  (421 + 423 * x) - 500 = 767 := 
by
  sorry

end subtract_500_from_sum_of_calculations_l2015_201541


namespace quadratic_has_non_real_roots_l2015_201555

theorem quadratic_has_non_real_roots (c : ℝ) (h : c > 16) :
    ∃ (a b : ℂ), (x^2 - 8 * x + c = 0) = (a * a = -1) ∧ (b * b = -1) :=
sorry

end quadratic_has_non_real_roots_l2015_201555


namespace quadratic_solution_value_l2015_201506

open Real

theorem quadratic_solution_value (a b : ℝ) (h1 : 2 + b = -a) (h2 : 2 * b = -6) :
  (2 * a + b)^2023 = -1 :=
sorry

end quadratic_solution_value_l2015_201506


namespace triangle_altitude_l2015_201558

theorem triangle_altitude (b : ℕ) (h : ℕ) (area : ℕ) (h_area : area = 800) (h_base : b = 40) (h_formula : area = (1 / 2) * b * h) : h = 40 :=
by
  sorry

end triangle_altitude_l2015_201558


namespace sum_of_three_integers_l2015_201553

theorem sum_of_three_integers (a b c : ℕ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diff: a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_prod: a * b * c = 5^4) : a + b + c = 131 :=
sorry

end sum_of_three_integers_l2015_201553


namespace remainder_poly_l2015_201585

theorem remainder_poly (x : ℂ) (h : x^5 + x^4 + x^3 + x^2 + x + 1 = 0) :
  (x^75 + x^60 + x^45 + x^30 + x^15 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 :=
by sorry

end remainder_poly_l2015_201585


namespace triangle_area_is_rational_l2015_201584

-- Definition of the area of a triangle given vertices with integer coordinates
def triangle_area (x1 x2 x3 y1 y2 y3 : ℤ) : ℚ :=
0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The theorem stating that the area of a triangle formed by points with integer coordinates is rational
theorem triangle_area_is_rational (x1 x2 x3 y1 y2 y3 : ℤ) :
  ∃ (area : ℚ), area = triangle_area x1 x2 x3 y1 y2 y3 :=
by
  sorry

end triangle_area_is_rational_l2015_201584


namespace compare_P_Q_l2015_201564

-- Define the structure of the number a with 2010 digits of 1
def a := 10^2010 - 1

-- Define P and Q based on a
def P := 24 * a^2
def Q := 24 * a^2 + 4 * a

-- Define the theorem to compare P and Q
theorem compare_P_Q : Q > P := by
  sorry

end compare_P_Q_l2015_201564


namespace range_of_x_satisfying_inequality_l2015_201576

def f (x : ℝ) : ℝ := -- Define the function f (we will leave this definition open for now)
sorry
@[continuity] axiom f_increasing (x y : ℝ) (h : x < y) : f x < f y
axiom f_2_eq_1 : f 2 = 1
axiom f_xy_eq_f_x_add_f_y (x y : ℝ) : f (x * y) = f x + f y

noncomputable def f_4_eq_2 : f 4 = 2 := sorry

theorem range_of_x_satisfying_inequality (x : ℝ) :
  3 < x ∧ x ≤ 4 ↔ f x + f (x - 3) ≤ 2 :=
sorry

end range_of_x_satisfying_inequality_l2015_201576


namespace sugar_percentage_is_7_5_l2015_201572

theorem sugar_percentage_is_7_5 
  (V1 : ℕ := 340)
  (p_water : ℝ := 88/100)
  (p_kola : ℝ := 5/100)
  (p_sugar : ℝ := 7/100)
  (V_sugar_add : ℝ := 3.2)
  (V_water_add : ℝ := 10)
  (V_kola_add : ℝ := 6.8) : 
  (
    (23.8 + 3.2) / (340 + 3.2 + 10 + 6.8) * 100 = 7.5
  ) :=
  by
  sorry

end sugar_percentage_is_7_5_l2015_201572


namespace sum_of_u_and_v_l2015_201527

theorem sum_of_u_and_v (u v : ℤ) (h1 : 1 ≤ v) (h2 : v < u) (h3 : u^2 + v^2 = 500) : u + v = 20 := by
  sorry

end sum_of_u_and_v_l2015_201527


namespace mushrooms_left_l2015_201519

-- Define the initial amount of mushrooms.
def init_mushrooms : ℕ := 15

-- Define the amount of mushrooms eaten.
def eaten_mushrooms : ℕ := 8

-- Define the resulting amount of mushrooms.
def remaining_mushrooms (init : ℕ) (eaten : ℕ) : ℕ := init - eaten

-- The proof statement
theorem mushrooms_left : remaining_mushrooms init_mushrooms eaten_mushrooms = 7 :=
by
    sorry

end mushrooms_left_l2015_201519


namespace superior_sequences_count_l2015_201559

noncomputable def number_of_superior_sequences (n : ℕ) : ℕ :=
  Nat.choose (2 * n + 1) (n + 1) * 2^n

theorem superior_sequences_count (n : ℕ) (h : 2 ≤ n) 
  (x : Fin (n + 1) → ℤ)
  (h1 : ∀ i, 0 ≤ i ∧ i ≤ n → |x i| ≤ n)
  (h2 : ∀ i j, 0 ≤ i ∧ i < j ∧ j ≤ n → x i ≠ x j)
  (h3 : ∀ (i j k : Nat), 0 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n → 
    max (|x k - x i|) (|x k - x j|) = 
    (|x i - x j| + |x j - x k| + |x k - x i|) / 2) :
  number_of_superior_sequences n = Nat.choose (2 * n + 1) (n + 1) * 2^n :=
sorry

end superior_sequences_count_l2015_201559


namespace arithmetic_sequence_common_difference_l2015_201535

theorem arithmetic_sequence_common_difference
  (a₁ d : ℝ)
  (h_a4 : a₁ + 3 * d = -2)
  (h_sum : 10 * a₁ + 45 * d = 65) :
  d = 17 / 3 :=
sorry

end arithmetic_sequence_common_difference_l2015_201535


namespace ribbon_each_box_fraction_l2015_201521

-- Define the conditions
def total_ribbon_used : ℚ := 5 / 12
def number_of_boxes : ℕ := 5
def ribbon_per_box : ℚ := total_ribbon_used / number_of_boxes

-- Statement to be proved
theorem ribbon_each_box_fraction :
  ribbon_per_box = 1 / 12 :=
  sorry

end ribbon_each_box_fraction_l2015_201521


namespace min_a_minus_b_when_ab_eq_156_l2015_201531

theorem min_a_minus_b_when_ab_eq_156 : ∃ a b : ℤ, (a * b = 156 ∧ a - b = -155) :=
by
  sorry

end min_a_minus_b_when_ab_eq_156_l2015_201531


namespace abs_equation_solution_l2015_201566

theorem abs_equation_solution (x : ℝ) (h : |x - 3| = 2 * x + 4) : x = -1 / 3 :=
by
  sorry

end abs_equation_solution_l2015_201566


namespace batches_engine_count_l2015_201586

theorem batches_engine_count (x : ℕ) 
  (h1 : ∀ e, 1/4 * e = 0) -- every batch has engines, no proof needed for this question
  (h2 : 5 * (3/4 : ℚ) * x = 300) : 
  x = 80 := 
sorry

end batches_engine_count_l2015_201586


namespace part1_part2_l2015_201530

noncomputable def probability_A_receives_one_red_envelope : ℚ :=
  sorry

theorem part1 (P_A1 : ℚ) (P_not_A1 : ℚ) (P_A2 : ℚ) (P_not_A2 : ℚ) :
  P_A1 = 1/3 ∧ P_not_A1 = 2/3 ∧ P_A2 = 1/3 ∧ P_not_A2 = 2/3 →
  probability_A_receives_one_red_envelope = 4/9 :=
sorry

noncomputable def probability_B_receives_at_least_10_yuan : ℚ :=
  sorry

theorem part2 (P_B1 : ℚ) (P_not_B1 : ℚ) (P_B2 : ℚ) (P_not_B2 : ℚ) (P_B3 : ℚ) (P_not_B3 : ℚ) :
  P_B1 = 1/3 ∧ P_not_B1 = 2/3 ∧ P_B2 = 1/3 ∧ P_not_B2 = 2/3 ∧ P_B3 = 1/3 ∧ P_not_B3 = 2/3 →
  probability_B_receives_at_least_10_yuan = 11/27 :=
sorry

end part1_part2_l2015_201530


namespace number_of_boys_in_class_l2015_201552

theorem number_of_boys_in_class (B : ℕ) (G : ℕ) (hG : G = 10) (h_combinations : (G * B * (B - 1)) / 2 = 1050) :
    B = 15 :=
by
  sorry

end number_of_boys_in_class_l2015_201552


namespace david_initial_money_l2015_201549

theorem david_initial_money (S X : ℕ) (h1 : S - 800 = 500) (h2 : X = S + 500) : X = 1800 :=
by
  sorry

end david_initial_money_l2015_201549


namespace dans_car_mpg_l2015_201505

noncomputable def milesPerGallon (distance money gas_price : ℝ) : ℝ :=
  distance / (money / gas_price)

theorem dans_car_mpg :
  let gas_price := 4
  let distance := 432
  let money := 54
  milesPerGallon distance money gas_price = 32 :=
by
  simp [milesPerGallon]
  sorry

end dans_car_mpg_l2015_201505


namespace base_circumference_of_cone_l2015_201591

theorem base_circumference_of_cone (r : ℝ) (theta : ℝ) (C : ℝ) 
  (h_radius : r = 6)
  (h_theta : theta = 180)
  (h_C : C = 2 * Real.pi * r) :
  (theta / 360) * C = 6 * Real.pi :=
by
  sorry

end base_circumference_of_cone_l2015_201591


namespace sum_of_variables_l2015_201548

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2 * x + 4 * y - 6 * z + 14 = 0) : x + y + z = 2 :=
sorry

end sum_of_variables_l2015_201548


namespace find_a_l2015_201580

-- Define the binomial coefficient function in Lean
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions and the proof problem statement
theorem find_a (a : ℝ) (h: (-a)^7 * binomial 10 7 = -120) : a = 1 :=
sorry

end find_a_l2015_201580


namespace Anita_should_buy_more_cartons_l2015_201540

def Anita_needs (total_needed : ℕ) : Prop :=
total_needed = 26

def Anita_has (strawberries blueberries : ℕ) : Prop :=
strawberries = 10 ∧ blueberries = 9

def additional_cartons (total_needed strawberries blueberries : ℕ) : ℕ :=
total_needed - (strawberries + blueberries)

theorem Anita_should_buy_more_cartons :
  ∀ (total_needed strawberries blueberries : ℕ),
    Anita_needs total_needed →
    Anita_has strawberries blueberries →
    additional_cartons total_needed strawberries blueberries = 7 :=
by
  intros total_needed strawberries blueberries Hneeds Hhas
  sorry

end Anita_should_buy_more_cartons_l2015_201540


namespace a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l2015_201574

variable (a0 a1 a2 a3 a4 a5 : ℝ)

noncomputable def polynomial (x : ℝ) : ℝ :=
  a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3 + a4 * (1 - x)^4 + a5 * (1 - x)^5

theorem a3_is_neg_10 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a3 = -10 :=
sorry

theorem a1_a3_a5_sum_is_neg_16 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a1 + a3 + a5 = -16 :=
sorry

end a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l2015_201574


namespace jose_share_of_profit_correct_l2015_201504

noncomputable def jose_share_of_profit (total_profit : ℝ) : ℝ :=
  let tom_investment_time := 30000 * 12
  let jose_investment_time := 45000 * 10
  let angela_investment_time := 60000 * 8
  let rebecca_investment_time := 75000 * 6
  let total_investment_time := tom_investment_time + jose_investment_time + angela_investment_time + rebecca_investment_time
  (jose_investment_time / total_investment_time) * total_profit

theorem jose_share_of_profit_correct : 
  ∀ (total_profit : ℝ), total_profit = 72000 -> jose_share_of_profit total_profit = 18620.69 := 
by
  intro total_profit
  sorry

end jose_share_of_profit_correct_l2015_201504


namespace find_ab_l2015_201569

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x - 1) = 7) ∧ (∀ x : ℤ, x^3 + a * x^2 + b * x + 5 % (x + 1) = 9) →
  (a, b) = (3, -2) := 
by
  sorry

end find_ab_l2015_201569


namespace coordinate_minimizes_z_l2015_201528

-- Definitions for conditions
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def equation_holds (x y : ℝ) : Prop := (1 / x) + (1 / (2 * y)) + (3 / (2 * x * y)) = 1

def z_def (x y : ℝ) : ℝ := x * y

-- Statement
theorem coordinate_minimizes_z (x y : ℝ) (h1 : in_first_quadrant x y) (h2 : equation_holds x y) :
    z_def x y = 9 / 2 ∧ (x = 3 ∧ y = 3 / 2) :=
    sorry

end coordinate_minimizes_z_l2015_201528


namespace no_int_solutions_x2_minus_3y2_eq_17_l2015_201598

theorem no_int_solutions_x2_minus_3y2_eq_17 : 
  ∀ (x y : ℤ), (x^2 - 3 * y^2 ≠ 17) := 
by
  intros x y
  sorry

end no_int_solutions_x2_minus_3y2_eq_17_l2015_201598


namespace exponent_solver_l2015_201507

theorem exponent_solver (x : ℕ) : 3^x + 3^x + 3^x + 3^x = 19683 → x = 7 := sorry

end exponent_solver_l2015_201507


namespace obtuse_angle_half_in_first_quadrant_l2015_201578

-- Define α to be an obtuse angle
variable {α : ℝ}

-- The main theorem we want to prove
theorem obtuse_angle_half_in_first_quadrant (h_obtuse : (π / 2) < α ∧ α < π) :
  0 < α / 2 ∧ α / 2 < π / 2 :=
  sorry

end obtuse_angle_half_in_first_quadrant_l2015_201578


namespace every_nat_as_diff_of_same_prime_divisors_l2015_201579

-- Conditions
def prime_divisors (n : ℕ) : ℕ :=
  -- function to count the number of distinct prime divisors of n
  sorry

-- Tuple translation
theorem every_nat_as_diff_of_same_prime_divisors :
  ∀ n : ℕ, ∃ a b : ℕ, n = a - b ∧ prime_divisors a = prime_divisors b := 
by
  sorry

end every_nat_as_diff_of_same_prime_divisors_l2015_201579


namespace installation_rates_l2015_201588

variables (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ)
variables (rate_teamA : ℕ) (rate_teamB : ℕ)

-- Conditions
def conditions : Prop :=
  units_total = 140 ∧
  teamA_units = 80 ∧
  teamB_units = units_total - teamA_units ∧
  team_units_gap = 5 ∧
  rate_teamA = rate_teamB + team_units_gap

-- Question to prove
def solution : Prop :=
  rate_teamB = 15 ∧ rate_teamA = 20

-- Statement of the proof
theorem installation_rates (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ) (rate_teamA : ℕ) (rate_teamB : ℕ) :
  conditions units_total teamA_units teamB_units team_units_gap rate_teamA rate_teamB →
  solution rate_teamA rate_teamB :=
sorry

end installation_rates_l2015_201588


namespace base_of_isosceles_triangle_l2015_201560

theorem base_of_isosceles_triangle (a b side equil_perim iso_perim : ℕ) 
  (h1 : equil_perim = 60)
  (h2 : 3 * side = equil_perim)
  (h3 : iso_perim = 50)
  (h4 : 2 * side + b = iso_perim)
  : b = 10 :=
by
  sorry

end base_of_isosceles_triangle_l2015_201560


namespace stratified_sampling_male_athletes_l2015_201501

theorem stratified_sampling_male_athletes : 
  ∀ (total_males total_females total_to_sample : ℕ), 
    total_males = 20 → 
    total_females = 10 → 
    total_to_sample = 6 → 
    20 * (total_to_sample / (total_males + total_females)) = 4 :=
by
  intros total_males total_females total_to_sample h_males h_females h_sample
  rw [h_males, h_females, h_sample]
  sorry

end stratified_sampling_male_athletes_l2015_201501


namespace miss_hilt_apples_l2015_201524

theorem miss_hilt_apples (h : ℕ) (a_per_hour : ℕ) (total_apples : ℕ) 
    (H1 : a_per_hour = 5) (H2 : total_apples = 15) (H3 : total_apples = h * a_per_hour) : 
  h = 3 :=
by
  sorry

end miss_hilt_apples_l2015_201524


namespace group_purchase_cheaper_l2015_201536

-- Define the initial conditions
def initial_price : ℕ := 10
def bulk_price : ℕ := 7
def delivery_cost : ℕ := 100
def group_size : ℕ := 50

-- Define the costs for individual and group purchases
def individual_cost : ℕ := initial_price
def group_cost : ℕ := bulk_price + (delivery_cost / group_size)

-- Statement to prove: cost per participant in a group purchase is less than cost per participant in individual purchases
theorem group_purchase_cheaper : group_cost < individual_cost := by
  sorry

end group_purchase_cheaper_l2015_201536


namespace final_problem_l2015_201547

def problem1 : Prop :=
  ∃ (x y : ℝ), 10 * x + 20 * y = 3000 ∧ 8 * x + 24 * y = 2800 ∧ x = 200 ∧ y = 50

def problem2 : Prop :=
  ∀ (m : ℕ), 10 ≤ m ∧ m ≤ 12 ∧ 
  200 * m + 50 * (40 - m) ≤ 3800 ∧ 
  (40 - m) ≤ 3 * m →
  (m = 10 ∧ (40 - m) = 30) ∨ 
  (m = 11 ∧ (40 - m) = 29) ∨ 
  (m = 12 ∧ (40 - m) = 28)

theorem final_problem : problem1 ∧ problem2 :=
by
  sorry

end final_problem_l2015_201547


namespace scientific_notation_of_510000000_l2015_201508

theorem scientific_notation_of_510000000 :
  (510000000 : ℝ) = 5.1 * 10^8 := 
sorry

end scientific_notation_of_510000000_l2015_201508


namespace integers_a_b_c_d_arbitrarily_large_l2015_201523

theorem integers_a_b_c_d_arbitrarily_large (n : ℤ) : 
  ∃ (a b c d : ℤ), (a^2 + b^2 + c^2 + d^2 = a * b * c + a * b * d + a * c * d + b * c * d) ∧ 
    min (min a b) (min c d) ≥ n := 
by sorry

end integers_a_b_c_d_arbitrarily_large_l2015_201523


namespace min_sum_of_factors_l2015_201526

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 2310) : a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l2015_201526


namespace yearly_payment_split_evenly_l2015_201520

def monthly_cost : ℤ := 14
def split_cost (cost : ℤ) := cost / 2
def total_yearly_cost (monthly_payment : ℤ) := monthly_payment * 12

theorem yearly_payment_split_evenly (h : split_cost monthly_cost = 7) :
  total_yearly_cost (split_cost monthly_cost) = 84 :=
by
  -- Here we use the hypothesis h which simplifies the proof.
  sorry

end yearly_payment_split_evenly_l2015_201520


namespace solve_system_l2015_201500

theorem solve_system (x y z : ℝ) 
  (h1 : x^3 - y = 6)
  (h2 : y^3 - z = 6)
  (h3 : z^3 - x = 6) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end solve_system_l2015_201500
