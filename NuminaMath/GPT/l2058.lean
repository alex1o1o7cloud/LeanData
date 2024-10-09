import Mathlib

namespace diagonal_ratio_of_squares_l2058_205892

theorem diagonal_ratio_of_squares (P d : ℝ) (h : ∃ s S, 4 * S = 4 * s * 4 ∧ P = 4 * s ∧ d = s * Real.sqrt 2) : 
    (∃ D, D = 4 * d) :=
by
  sorry

end diagonal_ratio_of_squares_l2058_205892


namespace regular_polygon_sides_l2058_205822

theorem regular_polygon_sides (n : ℕ) (h : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l2058_205822


namespace exists_irrationals_floor_neq_l2058_205868

-- Define irrationality of a number
def irrational (x : ℝ) : Prop :=
  ¬ ∃ (r : ℚ), x = r

theorem exists_irrationals_floor_neq :
  ∃ (a b : ℝ), irrational a ∧ irrational b ∧ 1 < a ∧ 1 < b ∧ 
  ∀ (m n : ℕ), ⌊a ^ m⌋ ≠ ⌊b ^ n⌋ :=
by
  sorry

end exists_irrationals_floor_neq_l2058_205868


namespace tina_total_income_is_correct_l2058_205878

-- Definitions based on the conditions
def hourly_wage : ℝ := 18.0
def regular_hours_per_day : ℝ := 8
def overtime_hours_per_day_weekday : ℝ := 2
def double_overtime_hours_per_day_weekend : ℝ := 2

def overtime_rate : ℝ := hourly_wage + 0.5 * hourly_wage
def double_overtime_rate : ℝ := 2 * hourly_wage

def weekday_hours_per_day : ℝ := 10
def weekend_hours_per_day : ℝ := 12

def regular_pay_per_day : ℝ := hourly_wage * regular_hours_per_day
def overtime_pay_per_day_weekday : ℝ := overtime_rate * overtime_hours_per_day_weekday
def double_overtime_pay_per_day_weekend : ℝ := double_overtime_rate * double_overtime_hours_per_day_weekend

def total_weekday_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday
def total_weekend_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday + double_overtime_pay_per_day_weekend

def number_of_weekdays : ℝ := 5
def number_of_weekends : ℝ := 2

def total_weekday_income : ℝ := total_weekday_pay_per_day * number_of_weekdays
def total_weekend_income : ℝ := total_weekend_pay_per_day * number_of_weekends

def total_weekly_income : ℝ := total_weekday_income + total_weekend_income

-- The theorem we need to prove
theorem tina_total_income_is_correct : total_weekly_income = 1530 := by
  sorry

end tina_total_income_is_correct_l2058_205878


namespace sum_of_solutions_eq_eight_l2058_205836

theorem sum_of_solutions_eq_eight : 
  ∀ x : ℝ, (x^2 - 6 * x + 5 = 2 * x - 7) → (∃ a b : ℝ, (a = 6) ∧ (b = 2) ∧ (a + b = 8)) :=
by
  sorry

end sum_of_solutions_eq_eight_l2058_205836


namespace product_of_triangle_areas_not_end_2014_l2058_205809

theorem product_of_triangle_areas_not_end_2014
  (T1 T2 T3 T4 : ℤ)
  (h1 : T1 > 0)
  (h2 : T2 > 0)
  (h3 : T3 > 0)
  (h4 : T4 > 0) :
  (T1 * T2 * T3 * T4) % 10000 ≠ 2014 := by
sorry

end product_of_triangle_areas_not_end_2014_l2058_205809


namespace ball_hits_ground_l2058_205889

theorem ball_hits_ground (t : ℝ) (y : ℝ) : 
  (y = -8 * t^2 - 12 * t + 72) → 
  (y = 0) → 
  t = 3 := 
by
  sorry

end ball_hits_ground_l2058_205889


namespace heather_walked_distance_l2058_205871

theorem heather_walked_distance {H S : ℝ} (hH : H = 5) (hS : S = H + 1) (total_distance : ℝ) (time_delay_stacy : ℝ) (time_heather_meet : ℝ) :
  (total_distance = 30) → (time_delay_stacy = 0.4) → (time_heather_meet = (total_distance - S * time_delay_stacy) / (H + S)) →
  (H * time_heather_meet = 12.55) :=
by
  sorry

end heather_walked_distance_l2058_205871


namespace units_digit_G100_l2058_205824

def G (n : ℕ) := 3 * 2 ^ (2 ^ n) + 2

theorem units_digit_G100 : (G 100) % 10 = 0 :=
by
  sorry

end units_digit_G100_l2058_205824


namespace A_is_sufficient_but_not_necessary_for_D_l2058_205849

variable {A B C D : Prop}

-- Defining the conditions
axiom h1 : A → B
axiom h2 : B ↔ C
axiom h3 : C → D

-- Statement to be proven
theorem A_is_sufficient_but_not_necessary_for_D : (A → D) ∧ ¬(D → A) :=
  by
  sorry

end A_is_sufficient_but_not_necessary_for_D_l2058_205849


namespace tan_inequality_l2058_205818

open Real

theorem tan_inequality {x1 x2 : ℝ} 
  (h1 : 0 < x1 ∧ x1 < π / 2) 
  (h2 : 0 < x2 ∧ x2 < π / 2) 
  (h3 : x1 ≠ x2) : 
  (1 / 2 * (tan x1 + tan x2) > tan ((x1 + x2) / 2)) :=
sorry

end tan_inequality_l2058_205818


namespace rectangle_exists_l2058_205813

theorem rectangle_exists (n : ℕ) (h_n : 0 < n)
  (marked : Finset (Fin n × Fin n))
  (h_marked : marked.card ≥ n * (Real.sqrt n + 0.5)) :
  ∃ (r1 r2 : Fin n) (c1 c2 : Fin n), r1 ≠ r2 ∧ c1 ≠ c2 ∧ 
    ((r1, c1) ∈ marked ∧ (r1, c2) ∈ marked ∧ (r2, c1) ∈ marked ∧ (r2, c2) ∈ marked) :=
  sorry

end rectangle_exists_l2058_205813


namespace unfolded_paper_has_four_symmetrical_holes_l2058_205848

structure Paper :=
  (width : ℤ) (height : ℤ) (hole_x : ℤ) (hole_y : ℤ)

structure Fold :=
  (direction : String) (fold_line : ℤ)

structure UnfoldedPaper :=
  (holes : List (ℤ × ℤ))

-- Define the initial paper, folds, and punching
def initial_paper : Paper := {width := 4, height := 6, hole_x := 2, hole_y := 1}
def folds : List Fold := 
  [{direction := "bottom_to_top", fold_line := initial_paper.height / 2}, 
   {direction := "left_to_right", fold_line := initial_paper.width / 2}]
def punch : (ℤ × ℤ) := (initial_paper.hole_x, initial_paper.hole_y)

-- The theorem to prove the resulting unfolded paper
theorem unfolded_paper_has_four_symmetrical_holes (p : Paper) (fs : List Fold) (punch : ℤ × ℤ) :
  UnfoldedPaper :=
  { holes := [(1, 1), (1, 5), (3, 1), (3, 5)] } -- Four symmetrically placed holes.

end unfolded_paper_has_four_symmetrical_holes_l2058_205848


namespace Deepak_age_l2058_205864

-- Define the current ages of Arun and Deepak
variable (A D : ℕ)

-- Define the conditions
def ratio_condition := A / D = 4 / 3
def future_age_condition := A + 6 = 26

-- Define the proof statement
theorem Deepak_age (h1 : ratio_condition A D) (h2 : future_age_condition A) : D = 15 :=
  sorry

end Deepak_age_l2058_205864


namespace div_rule_2701_is_37_or_73_l2058_205860

theorem div_rule_2701_is_37_or_73 (a b x : ℕ) (h1 : 10 * a + b = x) (h2 : a^2 + b^2 = 58) : 
  (x = 37 ∨ x = 73) ↔ 2701 % x = 0 :=
by
  sorry

end div_rule_2701_is_37_or_73_l2058_205860


namespace construct_line_through_points_l2058_205877

-- Definitions of the conditions
def points_on_sheet (A B : ℝ × ℝ) : Prop := A ≠ B
def tool_constraints (ruler_length compass_max_opening distance_A_B : ℝ) : Prop :=
  distance_A_B > 2 * ruler_length ∧ distance_A_B > 2 * compass_max_opening

-- The main theorem statement
theorem construct_line_through_points (A B : ℝ × ℝ) (ruler_length compass_max_opening : ℝ) 
  (h_points : points_on_sheet A B) 
  (h_constraints : tool_constraints ruler_length compass_max_opening (dist A B)) : 
  ∃ line : ℝ × ℝ → Prop, line A ∧ line B :=
sorry

end construct_line_through_points_l2058_205877


namespace cos_C_equal_two_thirds_l2058_205863

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Define the conditions
def condition1 : a > 0 ∧ b > 0 ∧ c > 0 := sorry
def condition2 : (a / b) + (b / a) = 4 * Real.cos C := sorry
def condition3 : Real.cos (A - B) = 1 / 6 := sorry

-- Statement to prove
theorem cos_C_equal_two_thirds 
  (h1: a > 0 ∧ b > 0 ∧ c > 0) 
  (h2: (a / b) + (b / a) = 4 * Real.cos C) 
  (h3: Real.cos (A - B) = 1 / 6) 
  : Real.cos C = 2 / 3 :=
  sorry

end cos_C_equal_two_thirds_l2058_205863


namespace b_share_of_payment_l2058_205823

def work_fraction (d : ℕ) : ℚ := 1 / d

def total_one_day_work (a_days b_days c_days : ℕ) : ℚ :=
  work_fraction a_days + work_fraction b_days + work_fraction c_days

def share_of_work (b_days : ℕ) (total_work : ℚ) : ℚ :=
  work_fraction b_days / total_work

def share_of_payment (total_payment : ℚ) (work_share : ℚ) : ℚ :=
  total_payment * work_share

theorem b_share_of_payment 
  (a_days b_days c_days : ℕ) (total_payment : ℚ):
  a_days = 6 → b_days = 8 → c_days = 12 → total_payment = 1800 →
  share_of_payment total_payment (share_of_work b_days (total_one_day_work a_days b_days c_days)) = 600 :=
by
  intros ha hb hc hp
  unfold total_one_day_work work_fraction share_of_work share_of_payment
  rw [ha, hb, hc, hp]
  -- Simplify the fractions and the multiplication
  sorry

end b_share_of_payment_l2058_205823


namespace largest_four_digit_perfect_square_l2058_205842

theorem largest_four_digit_perfect_square :
  ∃ (n : ℕ), n = 9261 ∧ (∃ k : ℕ, k * k = n) ∧ ∀ (m : ℕ), m < 10000 → (∃ x, x * x = m) → m ≤ n := 
by 
  sorry

end largest_four_digit_perfect_square_l2058_205842


namespace real_solutions_l2058_205832

theorem real_solutions (x : ℝ) :
  (x ≠ 3 ∧ x ≠ 7) →
  ((x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)) /
  ((x - 3) * (x - 7) * (x - 3)) = 1 →
  x = 3 + Real.sqrt 3 ∨ x = 3 - Real.sqrt 3 ∨ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5 :=
by
  sorry

end real_solutions_l2058_205832


namespace find_range_of_a_l2058_205834

-- Define the operation ⊗ on ℝ: x ⊗ y = x(1 - y)
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the inequality condition for all real numbers x
def inequality_condition (a : ℝ) : Prop :=
  ∀ (x : ℝ), tensor (x - a) (x + 1) < 1

-- State the theorem to prove the range of a
theorem find_range_of_a (a : ℝ) (h : inequality_condition a) : -2 < a ∧ a < 2 :=
  sorry

end find_range_of_a_l2058_205834


namespace am_gm_inequality_l2058_205888

variable (a : ℝ) (h : a > 0) -- Variables and condition

theorem am_gm_inequality (a : ℝ) (h : a > 0) : a + 1 / a ≥ 2 := 
sorry -- Proof is not provided according to instructions.

end am_gm_inequality_l2058_205888


namespace solve_for_y_l2058_205879

-- Given condition
def equation (y : ℚ) := (8 * y^2 + 90 * y + 5) / (3 * y^2 + 4 * y + 49) = 4 * y + 1

-- Prove the resulting polynomial equation
theorem solve_for_y (y : ℚ) (h : equation y) : 12 * y^3 + 11 * y^2 + 110 * y + 44 = 0 :=
sorry

end solve_for_y_l2058_205879


namespace find_m_l2058_205812

theorem find_m :
  ∃ m : ℕ, 264 * 391 % 100 = m ∧ 0 ≤ m ∧ m < 100 ∧ m = 24 :=
by
  sorry

end find_m_l2058_205812


namespace probability_of_triangle_or_circle_l2058_205895

-- Definitions (conditions)
def total_figures : ℕ := 12
def triangles : ℕ := 4
def circles : ℕ := 3
def squares : ℕ := 5
def figures : ℕ := triangles + circles + squares

-- Probability calculation
def probability_triangle_circle := (triangles + circles) / total_figures

-- Theorem statement (problem)
theorem probability_of_triangle_or_circle : probability_triangle_circle = 7 / 12 :=
by
  -- The proof is omitted, insert the proof here when necessary.
  sorry

end probability_of_triangle_or_circle_l2058_205895


namespace students_play_neither_l2058_205859

-- Define the conditions
def total_students : ℕ := 39
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_players : ℕ := 17

-- Define a theorem that states the equivalent proof problem
theorem students_play_neither : 
  total_students - (football_players + long_tennis_players - both_players) = 10 := by
  sorry

end students_play_neither_l2058_205859


namespace max_value_l2058_205885

-- Definitions for conditions
variables {a b : ℝ}
variables (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (1 / b) = 2)

-- Statement of the theorem
theorem max_value : (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 / a) + (1 / b) = 2 ∧ ∀ y : ℝ,
  (1 / y) * ((2 / (y * (3 * y - 1)⁻¹)) + 1) ≤ 25 / 8) :=
sorry

end max_value_l2058_205885


namespace min_width_of_garden_l2058_205810

theorem min_width_of_garden (w : ℝ) (h : 0 < w) (h1 : w * (w + 20) ≥ 120) : w ≥ 4 :=
sorry

end min_width_of_garden_l2058_205810


namespace packaging_combinations_l2058_205838

theorem packaging_combinations :
  let wraps := 10
  let ribbons := 4
  let cards := 5
  let stickers := 6
  wraps * ribbons * cards * stickers = 1200 :=
by
  rfl

end packaging_combinations_l2058_205838


namespace music_store_cellos_l2058_205872

/-- 
A certain music store stocks 600 violas. 
There are 100 cello-viola pairs, such that a cello and a viola were both made with wood from the same tree. 
The probability that the two instruments are made with wood from the same tree is 0.00020833333333333335. 
Prove that the store stocks 800 cellos.
-/
theorem music_store_cellos (V : ℕ) (P : ℕ) (Pr : ℚ) (C : ℕ) 
  (h1 : V = 600) 
  (h2 : P = 100) 
  (h3 : Pr = 0.00020833333333333335) 
  (h4 : Pr = P / (C * V)): C = 800 :=
by
  sorry

end music_store_cellos_l2058_205872


namespace exists_linear_function_second_quadrant_l2058_205800

theorem exists_linear_function_second_quadrant (k b : ℝ) (h1 : k > 0) (h2 : b > 0) :
  ∃ (f : ℝ → ℝ), (∀ x, f x = k * x + b) ∧ (∀ x, x < 0 → f x > 0) :=
by
  -- Prove there exists a linear function of the form f(x) = kx + b with given conditions
  -- Skip the proof for now
  sorry

end exists_linear_function_second_quadrant_l2058_205800


namespace necessary_french_woman_l2058_205876

structure MeetingConditions where
  total_money_women : ℝ
  total_money_men : ℝ
  total_money_french : ℝ
  total_money_russian : ℝ

axiom no_other_representatives : Prop
axiom money_french_vs_russian (conditions : MeetingConditions) : conditions.total_money_french > conditions.total_money_russian
axiom money_women_vs_men (conditions : MeetingConditions) : conditions.total_money_women > conditions.total_money_men

theorem necessary_french_woman (conditions : MeetingConditions) :
  ∃ w_f : ℝ, w_f > 0 ∧ conditions.total_money_french > w_f ∧ w_f + conditions.total_money_men > conditions.total_money_women :=
by
  sorry

end necessary_french_woman_l2058_205876


namespace negation_example_l2058_205835

theorem negation_example :
  (¬ (∃ n : ℕ, n^2 ≥ 2^n)) → (∀ n : ℕ, n^2 < 2^n) :=
by
  sorry

end negation_example_l2058_205835


namespace tickets_sold_second_half_l2058_205841

-- Definitions from conditions
def total_tickets := 9570
def first_half_tickets := 3867

-- Theorem to prove the number of tickets sold in the second half of the season
theorem tickets_sold_second_half : total_tickets - first_half_tickets = 5703 :=
by sorry

end tickets_sold_second_half_l2058_205841


namespace main_theorem_l2058_205819

noncomputable def f : ℝ → ℝ := sorry

axiom h_even : ∀ x : ℝ, f (-x) = f x
axiom h_decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → 
  (x1 < x2 ↔ (f x2 < f x1))

theorem main_theorem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end main_theorem_l2058_205819


namespace find_y_l2058_205806

variable (x y z : ℝ)

theorem find_y
    (h₀ : x + y + z = 150)
    (h₁ : x + 10 = y - 10)
    (h₂ : y - 10 = 3 * z) :
    y = 74.29 :=
by
    sorry

end find_y_l2058_205806


namespace initial_seashells_l2058_205814

-- Definitions for the conditions
def seashells_given_to_Tim : ℕ := 13
def seashells_now : ℕ := 36

-- Proving the number of initially found seashells
theorem initial_seashells : seashells_now + seashells_given_to_Tim = 49 :=
by
  -- we omit the proof steps with sorry
  sorry

end initial_seashells_l2058_205814


namespace proposition_C_l2058_205825

-- Given conditions
variables {a b : ℝ}

-- Proposition C is the correct one
theorem proposition_C (h : a^3 > b^3) : a > b := by
  sorry

end proposition_C_l2058_205825


namespace probability_first_number_greater_l2058_205880

noncomputable def probability_first_greater_second : ℚ :=
  let total_outcomes := 8 * 8
  let favorable_outcomes := 7 + 6 + 5 + 4 + 3 + 2 + 1
  favorable_outcomes / total_outcomes

theorem probability_first_number_greater :
  probability_first_greater_second = 7 / 16 :=
sorry

end probability_first_number_greater_l2058_205880


namespace visited_neither_l2058_205870

def people_total : ℕ := 90
def visited_iceland : ℕ := 55
def visited_norway : ℕ := 33
def visited_both : ℕ := 51

theorem visited_neither :
  people_total - (visited_iceland + visited_norway - visited_both) = 53 := by
  sorry

end visited_neither_l2058_205870


namespace find_y_l2058_205884

theorem find_y (x y : ℤ) (h₁ : x = 4) (h₂ : 3 * x + 2 * y = 30) : y = 9 := 
by
  sorry

end find_y_l2058_205884


namespace store_owner_oil_l2058_205826

noncomputable def liters_of_oil (volume_per_bottle : ℕ) (number_of_bottles : ℕ) : ℕ :=
  (volume_per_bottle * number_of_bottles) / 1000

theorem store_owner_oil : liters_of_oil 200 20 = 4 := by
  sorry

end store_owner_oil_l2058_205826


namespace time_for_B_work_alone_l2058_205855

def work_rate_A : ℚ := 1 / 6
def work_rate_combined : ℚ := 1 / 3
def work_share_C : ℚ := 1 / 8

theorem time_for_B_work_alone : 
  ∃ x : ℚ, (work_rate_A + 1 / x = work_rate_combined - work_share_C) → x = 24 := 
sorry

end time_for_B_work_alone_l2058_205855


namespace lcm_of_two_numbers_l2058_205854

-- Definitions based on the conditions
variable (a b l : ℕ)

-- The conditions from the problem
def hcf_ab : Nat := 9
def prod_ab : Nat := 1800

-- The main statement to prove
theorem lcm_of_two_numbers : Nat.lcm a b = 200 :=
by
  -- Skipping the proof implementation
  sorry

end lcm_of_two_numbers_l2058_205854


namespace find_original_percentage_of_acid_l2058_205886

noncomputable def percentage_of_acid (a w : ℕ) : ℚ :=
  (a : ℚ) / (a + w : ℚ) * 100

theorem find_original_percentage_of_acid (a w : ℕ) 
  (h1 : (a : ℚ) / (a + w + 2 : ℚ) = 1 / 4)
  (h2 : (a + 2 : ℚ) / (a + w + 4 : ℚ) = 2 / 5) : 
  percentage_of_acid a w = 33.33 :=
by 
  sorry

end find_original_percentage_of_acid_l2058_205886


namespace boat_distance_against_water_flow_l2058_205817

variable (a : ℝ) -- speed of the boat in still water

theorem boat_distance_against_water_flow 
  (speed_boat_still_water : ℝ := a)
  (speed_water_flow : ℝ := 3)
  (time_travel : ℝ := 3) :
  (speed_boat_still_water - speed_water_flow) * time_travel = 3 * (a - 3) := 
by
  sorry

end boat_distance_against_water_flow_l2058_205817


namespace james_hours_per_year_l2058_205811

def hours_per_day (trainings_per_day : Nat) (hours_per_training : Nat) : Nat :=
  trainings_per_day * hours_per_training

def days_per_week (total_days : Nat) (rest_days : Nat) : Nat :=
  total_days - rest_days

def hours_per_week (hours_day : Nat) (days_week : Nat) : Nat :=
  hours_day * days_week

def hours_per_year (hours_week : Nat) (weeks_year : Nat) : Nat :=
  hours_week * weeks_year

theorem james_hours_per_year :
  let trainings_per_day := 2
  let hours_per_training := 4
  let total_days_per_week := 7
  let rest_days_per_week := 2
  let weeks_per_year := 52
  hours_per_year 
    (hours_per_week 
      (hours_per_day trainings_per_day hours_per_training) 
      (days_per_week total_days_per_week rest_days_per_week)
    ) weeks_per_year
  = 2080 := by
  sorry

end james_hours_per_year_l2058_205811


namespace largest_n_unique_k_l2058_205807

theorem largest_n_unique_k :
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, (3 : ℚ) / 7 < (n : ℚ) / ((n + k : ℕ) : ℚ) ∧ 
  (n : ℚ) / ((n + k : ℕ) : ℚ) < (8 : ℚ) / 19 → k = 1 := by
sorry

end largest_n_unique_k_l2058_205807


namespace lydia_age_when_planted_l2058_205820

-- Definition of the conditions
def years_to_bear_fruit : ℕ := 7
def lydia_age_when_fruit_bears : ℕ := 11

-- Lean 4 statement to prove Lydia's age when she planted the tree
theorem lydia_age_when_planted (a : ℕ) : a = lydia_age_when_fruit_bears - years_to_bear_fruit :=
by
  have : a = 4 := by sorry
  exact this

end lydia_age_when_planted_l2058_205820


namespace power_function_value_l2058_205850

/-- Given a power function passing through a certain point, find the value at a specific point -/
theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h : f x = x ^ α) 
  (h_passes : f (1/4) = 4) : f 2 = 1/2 :=
sorry

end power_function_value_l2058_205850


namespace lauras_european_stamps_cost_l2058_205805

def stamp_cost (count : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  count * cost_per_stamp

def total_stamps_cost (stamps80 : ℕ) (stamps90 : ℕ) (cost_per_stamp : ℚ) : ℚ :=
  stamp_cost stamps80 cost_per_stamp + stamp_cost stamps90 cost_per_stamp

def european_stamps_cost_80_90 :=
  total_stamps_cost 10 12 0.09 + total_stamps_cost 18 16 0.07

theorem lauras_european_stamps_cost : european_stamps_cost_80_90 = 4.36 :=
by
  sorry

end lauras_european_stamps_cost_l2058_205805


namespace recycling_points_l2058_205893

-- Define the statement
theorem recycling_points : 
  ∀ (C H L I : ℝ) (points_per_six_pounds : ℝ), 
  C = 28 → H = 4.5 → L = 3.25 → I = 8.75 → points_per_six_pounds = 1 / 6 →
  (⌊ C * points_per_six_pounds ⌋ + ⌊ I * points_per_six_pounds ⌋  + ⌊ H * points_per_six_pounds ⌋ + ⌊ L * points_per_six_pounds ⌋ = 5) :=
by
  intros C H L I pps hC hH hL hI hpps
  rw [hC, hH, hL, hI, hpps]
  simp
  sorry

end recycling_points_l2058_205893


namespace smallest_digit_to_correct_l2058_205829

def incorrect_sum : ℕ := 2104
def correct_sum : ℕ := 738 + 625 + 841
def difference : ℕ := correct_sum - incorrect_sum

theorem smallest_digit_to_correct (d : ℕ) (h : difference = 100) :
  d = 6 := 
sorry

end smallest_digit_to_correct_l2058_205829


namespace rice_yield_prediction_l2058_205882

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 5 * x + 250

-- Define the specific condition for x = 80
def fertilizer_amount : ℝ := 80

-- State the theorem for the expected rice yield
theorem rice_yield_prediction : regression_line fertilizer_amount = 650 :=
by
  sorry

end rice_yield_prediction_l2058_205882


namespace find_number_l2058_205866

theorem find_number (x : ℝ) (h : x - (3/5) * x = 62) : x = 155 :=
by
  sorry

end find_number_l2058_205866


namespace total_heads_l2058_205803

def number_of_heads := 1
def number_of_feet_hen := 2
def number_of_feet_cow := 4
def total_feet := 144

theorem total_heads (H C : ℕ) (h_hens : H = 24) (h_feet : number_of_feet_hen * H + number_of_feet_cow * C = total_feet) :
  H + C = 48 :=
sorry

end total_heads_l2058_205803


namespace solve_for_three_times_x_plus_ten_l2058_205830

theorem solve_for_three_times_x_plus_ten (x : ℝ) (h_eq : 5 * x - 7 = 15 * x + 21) : 3 * (x + 10) = 21.6 := by
  sorry

end solve_for_three_times_x_plus_ten_l2058_205830


namespace sequence_monotonic_and_bounded_l2058_205844

theorem sequence_monotonic_and_bounded :
  ∀ (a : ℕ → ℝ), (a 1 = 1 / 2) → (∀ n, a (n + 1) = 1 / 2 + (a n)^2 / 2) →
    (∀ n, a n < 2) ∧ (∀ n, a n < a (n + 1)) :=
by
  sorry

end sequence_monotonic_and_bounded_l2058_205844


namespace unique_solution_for_a_l2058_205874

theorem unique_solution_for_a (a : ℝ) :
  (∃! (x y : ℝ), 
    (x * Real.cos a + y * Real.sin a = 5 * Real.cos a + 2 * Real.sin a) ∧
    (-3 ≤ x + 2 * y ∧ x + 2 * y ≤ 7) ∧
    (-9 ≤ 3 * x - 4 * y ∧ 3 * x - 4 * y ≤ 1)) ↔ 
  (∃ k : ℤ, a = Real.arctan 4 + k * Real.pi ∨ a = -Real.arctan 2 + k * Real.pi) :=
sorry

end unique_solution_for_a_l2058_205874


namespace exterior_angle_of_regular_octagon_l2058_205865

theorem exterior_angle_of_regular_octagon (sum_of_exterior_angles : ℝ) (n_sides : ℕ) (is_regular : n_sides = 8 ∧ sum_of_exterior_angles = 360) :
  sum_of_exterior_angles / n_sides = 45 := by
  sorry

end exterior_angle_of_regular_octagon_l2058_205865


namespace team_E_has_not_played_against_team_B_l2058_205804

-- We begin by defining the teams as an enumeration
inductive Team
| A | B | C | D | E | F

open Team

-- Define the total number of matches each team has played
def matches_played (t : Team) : Nat :=
  match t with
  | A => 5
  | B => 4
  | C => 3
  | D => 2
  | E => 1
  | F => 0 -- Note: we assume F's matches are not provided; this can be adjusted if needed

-- Prove that team E has not played against team B
theorem team_E_has_not_played_against_team_B :
  ∃ t : Team, matches_played B = 4 ∧ matches_played E < matches_played B ∧
  (t = E) :=
by
  sorry

end team_E_has_not_played_against_team_B_l2058_205804


namespace ms_warren_running_time_l2058_205899

theorem ms_warren_running_time 
  (t : ℝ) 
  (ht_total_distance : 6 * t + 2 * 0.5 = 3) : 
  60 * t = 20 := by 
  sorry

end ms_warren_running_time_l2058_205899


namespace syllogistic_reasoning_l2058_205881

theorem syllogistic_reasoning (a b c : Prop) (h1 : b → c) (h2 : a → b) : a → c :=
by sorry

end syllogistic_reasoning_l2058_205881


namespace arithmetic_sequence_count_l2058_205808

theorem arithmetic_sequence_count :
  ∃! (n a d : ℕ), n ≥ 3 ∧ (n * (2 * a + (n - 1) * d) = 2 * 97^2) :=
sorry

end arithmetic_sequence_count_l2058_205808


namespace toby_steps_l2058_205839

theorem toby_steps (sunday tuesday wednesday thursday friday_saturday monday : ℕ) :
    sunday = 9400 →
    tuesday = 8300 →
    wednesday = 9200 →
    thursday = 8900 →
    friday_saturday = 9050 →
    7 * 9000 = 63000 →
    monday = 63000 - (sunday + tuesday + wednesday + thursday + 2 * friday_saturday) → monday = 9100 :=
by
  intros hs ht hw hth hfs htc hnm
  sorry

end toby_steps_l2058_205839


namespace remainder_of_number_divisor_l2058_205891

-- Define the interesting number and the divisor
def number := 2519
def divisor := 9
def expected_remainder := 8

-- State the theorem to prove the remainder condition
theorem remainder_of_number_divisor :
  number % divisor = expected_remainder := by
  sorry

end remainder_of_number_divisor_l2058_205891


namespace value_of_f_l2058_205847

def B : Set ℚ := {x | x ≠ 0 ∧ x ≠ 1}

def f (x : ℚ) : ℝ := sorry

noncomputable def h (x : ℚ) : ℚ :=
  1 / (1 - x)

lemma cyclic_of_h :
  ∀ x ∈ B, h (h (h x)) = x :=
sorry

lemma functional_property (x : ℚ) (hx : x ∈ B) :
  f x + f (h x) = 2 * Real.log (|x|) :=
sorry

theorem value_of_f :
  f 2023 = Real.log 2023 :=
sorry

end value_of_f_l2058_205847


namespace sum_of_b_and_c_base7_l2058_205873

theorem sum_of_b_and_c_base7 (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
(h4 : A < 7) (h5 : B < 7) (h6 : C < 7) 
(h7 : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) 
: B + C = 6 ∨ B + C = 12 := sorry

end sum_of_b_and_c_base7_l2058_205873


namespace profit_per_meter_l2058_205843

theorem profit_per_meter 
  (total_meters : ℕ)
  (cost_price_per_meter : ℝ)
  (total_selling_price : ℝ)
  (h1 : total_meters = 92)
  (h2 : cost_price_per_meter = 83.5)
  (h3 : total_selling_price = 9890) : 
  (total_selling_price - total_meters * cost_price_per_meter) / total_meters = 24.1 :=
by
  sorry

end profit_per_meter_l2058_205843


namespace smallest_k_for_factorial_divisibility_l2058_205816

theorem smallest_k_for_factorial_divisibility : 
  ∃ (k : ℕ), (∀ n : ℕ, n < k → ¬(2040 ∣ n!)) ∧ (2040 ∣ k!) ∧ k = 17 :=
by
  -- We skip the actual proof steps and provide a placeholder for the proof
  sorry

end smallest_k_for_factorial_divisibility_l2058_205816


namespace final_spent_l2058_205890

-- Define all the costs.
def albertoExpenses : ℤ := 2457 + 374 + 520 + 129 + 799
def albertoDiscountExhaust : ℤ := (799 * 5) / 100
def albertoTotalBeforeLoyaltyDiscount : ℤ := albertoExpenses - albertoDiscountExhaust
def albertoLoyaltyDiscount : ℤ := (albertoTotalBeforeLoyaltyDiscount * 7) / 100
def albertoFinal : ℤ := albertoTotalBeforeLoyaltyDiscount - albertoLoyaltyDiscount

def samaraExpenses : ℤ := 25 + 467 + 79 + 175 + 599 + 225
def samaraSalesTax : ℤ := (samaraExpenses * 6) / 100
def samaraFinal : ℤ := samaraExpenses + samaraSalesTax

def difference : ℤ := albertoFinal - samaraFinal

theorem final_spent (h : difference = 2278) : true :=
  sorry

end final_spent_l2058_205890


namespace quadratic_root_relationship_l2058_205828

theorem quadratic_root_relationship
  (m1 m2 : ℝ)
  (x1 x2 x3 x4 : ℝ)
  (h_eq1 : m1 * x1^2 + (1 / 3) * x1 + 1 = 0)
  (h_eq2 : m1 * x2^2 + (1 / 3) * x2 + 1 = 0)
  (h_eq3 : m2 * x3^2 + (1 / 3) * x3 + 1 = 0)
  (h_eq4 : m2 * x4^2 + (1 / 3) * x4 + 1 = 0)
  (h_order : x1 < x3 ∧ x3 < x4 ∧ x4 < x2 ∧ x2 < 0) :
  m2 > m1 ∧ m1 > 0 :=
sorry

end quadratic_root_relationship_l2058_205828


namespace new_difference_l2058_205896

theorem new_difference (x y a : ℝ) (h : x - y = a) : (x + 0.5) - y = a + 0.5 := 
sorry

end new_difference_l2058_205896


namespace math_problem_l2058_205867

theorem math_problem : -5 * (-6) - 2 * (-3 * (-7) + (-8)) = 4 := 
  sorry

end math_problem_l2058_205867


namespace min_shirts_to_save_l2058_205851

theorem min_shirts_to_save (x : ℕ) :
  (75 + 10 * x < if x < 30 then 15 * x else 14 * x) → x = 20 :=
by
  sorry

end min_shirts_to_save_l2058_205851


namespace domain_of_f_l2058_205821

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (-x^2 + 9 * x + 10)) / Real.log (x - 1)

theorem domain_of_f :
  {x : ℝ | -x^2 + 9 * x + 10 ≥ 0 ∧ x - 1 > 0 ∧ Real.log (x - 1) ≠ 0} =
  {x : ℝ | (1 < x ∧ x < 2) ∨ (2 < x ∧ x ≤ 10)} :=
by
  sorry

end domain_of_f_l2058_205821


namespace sum_kml_l2058_205827

theorem sum_kml (k m l : ℤ) (b : ℤ → ℤ)
  (h_seq : ∀ n, ∃ k, b n = k * (Int.floor (Real.sqrt (n + m : ℝ))) + l)
  (h_b1 : b 1 = 2) :
  k + m + l = 3 := by
  sorry

end sum_kml_l2058_205827


namespace seating_arrangement_l2058_205815

def numWaysCableCars (adults children cars capacity : ℕ) : ℕ := 
  sorry 

theorem seating_arrangement :
  numWaysCableCars 4 2 3 3 = 348 :=
by {
  sorry
}

end seating_arrangement_l2058_205815


namespace range_of_a_l2058_205853

def f (a x : ℝ) : ℝ := -x^3 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 1 → -3 * x^2 + a ≥ 0) → a ≥ 3 := 
by
  sorry

end range_of_a_l2058_205853


namespace correct_statements_count_l2058_205833

theorem correct_statements_count :
  (∃ n : ℕ, odd_positive_integer = 4 * n + 1 ∨ odd_positive_integer = 4 * n + 3) ∧
  (∀ k : ℕ, k = 3 * m ∨ k = 3 * m + 1 ∨ k = 3 * m + 2) ∧
  (∀ s : ℕ, odd_positive_integer ^ 2 = 8 * p + 1) ∧
  (∀ t : ℕ, perfect_square = 3 * q ∨ perfect_square = 3 * q + 1) →
  num_correct_statements = 2 :=
by
  sorry

end correct_statements_count_l2058_205833


namespace leah_birds_duration_l2058_205894

-- Define the conditions
def boxes_bought : ℕ := 3
def boxes_existing : ℕ := 5
def parrot_weekly_consumption : ℕ := 100
def cockatiel_weekly_consumption : ℕ := 50
def grams_per_box : ℕ := 225

-- Define the question as a theorem
theorem leah_birds_duration : 
  (boxes_bought + boxes_existing) * grams_per_box / 
  (parrot_weekly_consumption + cockatiel_weekly_consumption) = 12 :=
by
  -- Proof would go here
  sorry

end leah_birds_duration_l2058_205894


namespace space_diagonals_Q_l2058_205869

-- Definitions based on the conditions
def vertices (Q : Type) : ℕ := 30
def edges (Q : Type) : ℕ := 70
def faces (Q : Type) : ℕ := 40
def triangular_faces (Q : Type) : ℕ := 20
def quadrilateral_faces (Q : Type) : ℕ := 15
def pentagon_faces (Q : Type) : ℕ := 5

-- Problem Statement
theorem space_diagonals_Q :
  ∀ (Q : Type),
  vertices Q = 30 →
  edges Q = 70 →
  faces Q = 40 →
  triangular_faces Q = 20 →
  quadrilateral_faces Q = 15 →
  pentagon_faces Q = 5 →
  ∃ d : ℕ, d = 310 := 
by
  -- At this point only the structure of the proof is set up.
  sorry

end space_diagonals_Q_l2058_205869


namespace simplify_expression_l2058_205852

theorem simplify_expression (a b c : ℝ) : a - (a - b + c) = b - c :=
by sorry

end simplify_expression_l2058_205852


namespace necessary_and_sufficient_condition_l2058_205857

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 0 < x → x + (1 / x) > a) ↔ a < 2 :=
sorry

end necessary_and_sufficient_condition_l2058_205857


namespace remainder_a83_l2058_205883

def a_n (n : ℕ) : ℕ := 6^n + 8^n

theorem remainder_a83 (n : ℕ) : 
  a_n 83 % 49 = 35 := sorry

end remainder_a83_l2058_205883


namespace people_visited_on_Sunday_l2058_205801

theorem people_visited_on_Sunday (ticket_price : ℕ) 
                                 (people_per_day_week : ℕ) 
                                 (people_on_Saturday : ℕ) 
                                 (total_revenue : ℕ) 
                                 (days_week : ℕ)
                                 (total_days : ℕ) 
                                 (people_per_day_mf : ℕ) 
                                 (people_on_other_days : ℕ) 
                                 (revenue_other_days : ℕ)
                                 (revenue_Sunday : ℕ)
                                 (people_Sunday : ℕ) :
    ticket_price = 3 →
    people_per_day_week = 100 →
    people_on_Saturday = 200 →
    total_revenue = 3000 →
    days_week = 5 →
    total_days = 7 →
    people_per_day_mf = people_per_day_week * days_week →
    people_on_other_days = people_per_day_mf + people_on_Saturday →
    revenue_other_days = people_on_other_days * ticket_price →
    revenue_Sunday = total_revenue - revenue_other_days →
    people_Sunday = revenue_Sunday / ticket_price →
    people_Sunday = 300 := 
by 
  sorry

end people_visited_on_Sunday_l2058_205801


namespace range_of_a_l2058_205856

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ a = x^2 - x - 1) ↔ -1 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l2058_205856


namespace katy_summer_reading_total_l2058_205887

def katy_books_in_summer (june_books july_books august_books : ℕ) : ℕ := june_books + july_books + august_books

theorem katy_summer_reading_total (june_books : ℕ) (july_books : ℕ) (august_books : ℕ) 
  (h1 : june_books = 8)
  (h2 : july_books = 2 * june_books)
  (h3 : august_books = july_books - 3) :
  katy_books_in_summer june_books july_books august_books = 37 :=
by
  sorry

end katy_summer_reading_total_l2058_205887


namespace gumball_machine_total_gumballs_l2058_205802

/-- A gumball machine has red, green, and blue gumballs. Given the following conditions:
1. The machine has half as many blue gumballs as red gumballs.
2. For each blue gumball, the machine has 4 times as many green gumballs.
3. The machine has 16 red gumballs.
Prove that the total number of gumballs in the machine is 56. -/
theorem gumball_machine_total_gumballs :
  ∀ (red blue green : ℕ),
    (blue = red / 2) →
    (green = blue * 4) →
    red = 16 →
    (red + blue + green = 56) :=
by
  intros red blue green h_blue h_green h_red
  sorry

end gumball_machine_total_gumballs_l2058_205802


namespace nth_equation_l2058_205846

-- Define the product of a list of integers
def prod_list (lst : List ℕ) : ℕ :=
  lst.foldl (· * ·) 1

-- Define the product of first n odd numbers
def prod_odds (n : ℕ) : ℕ :=
  prod_list (List.map (λ i => 2 * i - 1) (List.range n))

-- Define the product of the range from n+1 to 2n
def prod_range (n : ℕ) : ℕ :=
  prod_list (List.range' (n + 1) n)

-- The theorem to prove
theorem nth_equation (n : ℕ) (hn : 0 < n) : prod_range n = 2^n * prod_odds n := 
  sorry

end nth_equation_l2058_205846


namespace star_value_example_l2058_205858

def my_star (a b : ℝ) : ℝ := (a + b)^2 + (a - b)^2

theorem star_value_example : my_star 3 5 = 68 := 
by
  sorry

end star_value_example_l2058_205858


namespace balls_in_jar_l2058_205898

theorem balls_in_jar (total_balls initial_blue_balls balls_after_taking_out : ℕ) (probability_blue : ℚ) :
  initial_blue_balls = 6 →
  balls_after_taking_out = initial_blue_balls - 3 →
  probability_blue = 1 / 5 →
  (balls_after_taking_out : ℚ) / (total_balls - 3 : ℚ) = probability_blue →
  total_balls = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end balls_in_jar_l2058_205898


namespace double_inequality_l2058_205875

variable (a b c : ℝ)

def triangle_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem double_inequality (h : triangle_sides a b c) : 
  3 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 ∧ (a + b + c) ^ 2 < 4 * (a * b + b * c + c * a) :=
by
  sorry

end double_inequality_l2058_205875


namespace planar_figure_area_l2058_205840

noncomputable def side_length : ℝ := 10
noncomputable def area_of_square : ℝ := side_length * side_length
noncomputable def number_of_squares : ℕ := 6
noncomputable def total_area_of_planar_figure : ℝ := number_of_squares * area_of_square

theorem planar_figure_area : total_area_of_planar_figure = 600 :=
by
  sorry

end planar_figure_area_l2058_205840


namespace bake_cookies_l2058_205862

noncomputable def scale_factor (original_cookies target_cookies : ℕ) : ℕ :=
  target_cookies / original_cookies

noncomputable def required_flour (original_flour : ℕ) (scale : ℕ) : ℕ :=
  original_flour * scale

noncomputable def adjusted_sugar (original_sugar : ℕ) (scale : ℕ) (reduction_percent : ℚ) : ℚ :=
  original_sugar * scale * (1 - reduction_percent)

theorem bake_cookies 
  (original_cookies : ℕ)
  (target_cookies : ℕ)
  (original_flour : ℕ)
  (original_sugar : ℕ)
  (reduction_percent : ℚ)
  (h_original_cookies : original_cookies = 40)
  (h_target_cookies : target_cookies = 80)
  (h_original_flour : original_flour = 3)
  (h_original_sugar : original_sugar = 1)
  (h_reduction_percent : reduction_percent = 0.25) :
  required_flour original_flour (scale_factor original_cookies target_cookies) = 6 ∧ 
  adjusted_sugar original_sugar (scale_factor original_cookies target_cookies) reduction_percent = 1.5 := by
    sorry

end bake_cookies_l2058_205862


namespace f_xh_sub_f_x_l2058_205861

def f (x : ℝ) (k : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + k * x - 4

theorem f_xh_sub_f_x (x h : ℝ) (k : ℝ := -5) : 
    f (x + h) k - f x k = h * (6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h - 5) := by
  sorry

end f_xh_sub_f_x_l2058_205861


namespace find_y_in_interval_l2058_205845

theorem find_y_in_interval :
  { y : ℝ | y^2 + 7 * y < 12 } = { y : ℝ | -9 < y ∧ y < 2 } :=
sorry

end find_y_in_interval_l2058_205845


namespace no_such_geometric_sequence_exists_l2058_205837

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ q : ℝ, a (n + 1) = q * a n

noncomputable def satisfies_conditions (a : ℕ → ℝ) : Prop :=
(a 1 + a 6 = 11) ∧
(a 3 * a 4 = 32 / 9) ∧
(∀ n : ℕ, a (n + 1) > a n) ∧
(∃ m : ℕ, m > 4 ∧ (2 * a m^2 = (2 / 3 * a (m - 1) + (a (m + 1) + 4 / 9))))

theorem no_such_geometric_sequence_exists : 
  ¬ ∃ a : ℕ → ℝ, geometric_sequence a ∧ satisfies_conditions a := 
sorry

end no_such_geometric_sequence_exists_l2058_205837


namespace profit_functions_properties_l2058_205831

noncomputable def R (x : ℝ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℝ) : ℝ := 500 * x + 4000
noncomputable def P (x : ℝ) : ℝ := R x - C x
noncomputable def MP (x : ℝ) : ℝ := P (x + 1) - P x

theorem profit_functions_properties :
  (P x = -20 * x^2 + 2500 * x - 4000) ∧ 
  (MP x = -40 * x + 2480) ∧ 
  (∃ x_max₁, ∀ x, P x_max₁ ≥ P x) ∧ 
  (∃ x_max₂, ∀ x, MP x_max₂ ≥ MP x) ∧ 
  P x_max₁ ≠ MP x_max₂ := by
  sorry

end profit_functions_properties_l2058_205831


namespace M_inter_N_l2058_205897

def M : Set ℝ := {x | abs (x - 1) < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

theorem M_inter_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 3} :=
by
  sorry

end M_inter_N_l2058_205897
