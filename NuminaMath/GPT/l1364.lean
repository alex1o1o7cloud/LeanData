import Mathlib

namespace NUMINAMATH_GPT_lcm_inequality_l1364_136423

theorem lcm_inequality (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  Nat.lcm k m * Nat.lcm m n * Nat.lcm n k ≥ Nat.lcm (Nat.lcm k m) n ^ 2 :=
by sorry

end NUMINAMATH_GPT_lcm_inequality_l1364_136423


namespace NUMINAMATH_GPT_tens_digit_of_3_pow_2013_l1364_136438

theorem tens_digit_of_3_pow_2013 : (3^2013 % 100 / 10) % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_3_pow_2013_l1364_136438


namespace NUMINAMATH_GPT_fraction_simplifies_l1364_136422

theorem fraction_simplifies :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_GPT_fraction_simplifies_l1364_136422


namespace NUMINAMATH_GPT_subtracted_amount_l1364_136447

theorem subtracted_amount (A N : ℝ) (h₁ : N = 200) (h₂ : 0.95 * N - A = 178) : A = 12 :=
by
  sorry

end NUMINAMATH_GPT_subtracted_amount_l1364_136447


namespace NUMINAMATH_GPT_angle_perpendicular_coterminal_l1364_136465

theorem angle_perpendicular_coterminal (α β : ℝ) (k : ℤ) 
  (h_perpendicular : ∃ k, β = α + 90 + k * 360 ∨ β = α - 90 + k * 360) : 
  β = α + 90 + k * 360 ∨ β = α - 90 + k * 360 :=
sorry

end NUMINAMATH_GPT_angle_perpendicular_coterminal_l1364_136465


namespace NUMINAMATH_GPT_total_sales_correct_l1364_136435

def normal_sales_per_month : ℕ := 21122
def additional_sales_in_june : ℕ := 3922
def sales_in_june : ℕ := normal_sales_per_month + additional_sales_in_june
def sales_in_july : ℕ := normal_sales_per_month
def total_sales : ℕ := sales_in_june + sales_in_july

theorem total_sales_correct :
  total_sales = 46166 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_sales_correct_l1364_136435


namespace NUMINAMATH_GPT_x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l1364_136459

def y_is_60_percent_greater_than_x (x y : ℝ) : Prop :=
  y = 1.60 * x

def z_is_40_percent_less_than_y (y z : ℝ) : Prop :=
  z = 0.60 * y

theorem x_not_4_17_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x ≠ 0.9583 * z :=
by {
  sorry
}

theorem x_is_8_0032_percent_less_than_z (x y z : ℝ) (h1 : y_is_60_percent_greater_than_x x y) (h2 : z_is_40_percent_less_than_y y z) : 
  x = 0.919968 * z :=
by {
  sorry
}

end NUMINAMATH_GPT_x_not_4_17_percent_less_than_z_x_is_8_0032_percent_less_than_z_l1364_136459


namespace NUMINAMATH_GPT_equation_solution_count_l1364_136419

open Real

theorem equation_solution_count :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (π / 4 * sin x) = cos (π / 4 * cos x)) ∧ s.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_count_l1364_136419


namespace NUMINAMATH_GPT_correct_option_C_l1364_136483

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 5}
def P : Set ℕ := {2, 4}

theorem correct_option_C : 3 ∈ U \ (M ∪ P) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_C_l1364_136483


namespace NUMINAMATH_GPT_option_b_correct_l1364_136432

theorem option_b_correct (a : ℝ) : (a ^ 3) * (a ^ 2) = a ^ 5 := 
by
  sorry

end NUMINAMATH_GPT_option_b_correct_l1364_136432


namespace NUMINAMATH_GPT_base9_to_decimal_l1364_136482

theorem base9_to_decimal : (8 * 9^1 + 5 * 9^0) = 77 := 
by
  sorry

end NUMINAMATH_GPT_base9_to_decimal_l1364_136482


namespace NUMINAMATH_GPT_tom_age_ratio_l1364_136487

variable (T N : ℕ)

theorem tom_age_ratio (h_sum : T = T) (h_relation : T - N = 3 * (T - 3 * N)) : T / N = 4 :=
sorry

end NUMINAMATH_GPT_tom_age_ratio_l1364_136487


namespace NUMINAMATH_GPT_find_D_l1364_136472

-- This representation assumes 'ABCD' represents digits A, B, C, and D forming a four-digit number.
def four_digit_number (A B C D : ℕ) : ℕ :=
  1000 * A + 100 * B + 10 * C + D

theorem find_D (A B C D : ℕ) (h1 : 1000 * A + 100 * B + 10 * C + D 
                            = 2736) (h2: A ≠ B) (h3: A ≠ C) 
  (h4: A ≠ D) (h5: B ≠ C) (h6: B ≠ D) (h7: C ≠ D) : D = 6 := 
sorry

end NUMINAMATH_GPT_find_D_l1364_136472


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l1364_136475

theorem common_ratio_of_geometric_series (a b : ℚ) (h1 : a = 4 / 7) (h2 : b = 16 / 21) :
  b / a = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l1364_136475


namespace NUMINAMATH_GPT_john_naps_70_days_l1364_136445

def total_naps_in_days (naps_per_week nap_duration days_in_week total_days : ℕ) : ℕ :=
  let total_weeks := total_days / days_in_week
  let total_naps := total_weeks * naps_per_week
  total_naps * nap_duration

theorem john_naps_70_days
  (naps_per_week : ℕ)
  (nap_duration : ℕ)
  (days_in_week : ℕ)
  (total_days : ℕ)
  (h_naps_per_week : naps_per_week = 3)
  (h_nap_duration : nap_duration = 2)
  (h_days_in_week : days_in_week = 7)
  (h_total_days : total_days = 70) :
  total_naps_in_days naps_per_week nap_duration days_in_week total_days = 60 :=
by
  rw [h_naps_per_week, h_nap_duration, h_days_in_week, h_total_days]
  sorry

end NUMINAMATH_GPT_john_naps_70_days_l1364_136445


namespace NUMINAMATH_GPT_lunch_break_duration_l1364_136462

theorem lunch_break_duration
  (p h L : ℝ)
  (monday_eq : (9 - L) * (p + h) = 0.4)
  (tuesday_eq : (8 - L) * h = 0.33)
  (wednesday_eq : (12 - L) * p = 0.27) :
  L = 7.0 ∨ L * 60 = 420 :=
by
  sorry

end NUMINAMATH_GPT_lunch_break_duration_l1364_136462


namespace NUMINAMATH_GPT_largest_root_is_1011_l1364_136402

theorem largest_root_is_1011 (a b c d x : ℝ) 
  (h1 : a + d = 2022) 
  (h2 : b + c = 2022) 
  (h3 : a ≠ c) 
  (h4 : (x - a) * (x - b) = (x - c) * (x - d)) : 
  x = 1011 := 
sorry

end NUMINAMATH_GPT_largest_root_is_1011_l1364_136402


namespace NUMINAMATH_GPT_sheets_taken_l1364_136494

noncomputable def remaining_sheets_mean (b c : ℕ) : ℚ :=
  (b * (2 * b + 1) + (100 - 2 * (b + c)) * (2 * (b + c) + 101)) / 2 / (100 - 2 * c)

theorem sheets_taken (b c : ℕ) (h1 : 100 = 2 * 50) 
(h2 : ∀ n, n > 0 → 2 * n = n + n) 
(hmean : remaining_sheets_mean b c = 31) : 
  c = 17 := 
sorry

end NUMINAMATH_GPT_sheets_taken_l1364_136494


namespace NUMINAMATH_GPT_quadratic_radical_type_equivalence_l1364_136497

def is_same_type_as_sqrt2 (x : ℝ) : Prop := ∃ k : ℚ, x = k * (Real.sqrt 2)

theorem quadratic_radical_type_equivalence (A B C D : ℝ) (hA : A = (Real.sqrt 8) / 7)
  (hB : B = Real.sqrt 3) (hC : C = Real.sqrt (1 / 3)) (hD : D = Real.sqrt 12) :
  is_same_type_as_sqrt2 A ∧ ¬ is_same_type_as_sqrt2 B ∧ ¬ is_same_type_as_sqrt2 C ∧ ¬ is_same_type_as_sqrt2 D :=
by
  sorry

end NUMINAMATH_GPT_quadratic_radical_type_equivalence_l1364_136497


namespace NUMINAMATH_GPT_evaluate_expression_l1364_136478

def improper_fraction (n : Int) (a : Int) (b : Int) : Rat :=
  n + (a : Rat) / b

def expression (x : Rat) : Rat :=
  (x * 1.65 - x + (7 / 20) * x) * 47.5 * 0.8 * 2.5

theorem evaluate_expression : 
  expression (improper_fraction 20 94 95) = 1994 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1364_136478


namespace NUMINAMATH_GPT_problem_1_problem_2_l1364_136471

-- Problem 1: Prove that (\frac{1}{5} - \frac{2}{3} - \frac{3}{10}) × (-60) = 46
theorem problem_1 : (1/5 - 2/3 - 3/10) * -60 = 46 := by
  sorry

-- Problem 2: Prove that (-1)^{2024} + 24 ÷ (-2)^3 - 15^2 × (1/15)^2 = -3
theorem problem_2 : (-1)^2024 + 24 / (-2)^3 - 15^2 * (1/15)^2 = -3 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1364_136471


namespace NUMINAMATH_GPT_number_of_roots_l1364_136449

noncomputable def f (a b c x : ℝ) : ℝ := x^3 + 2 * a * x^2 + 2 * b * x + 3 * c

theorem number_of_roots (a b c x₁ x₂ : ℝ) (h_extreme : x₁ ≠ x₂)
    (h_fx1 : f a b c x₁ = x₁) :
    (∃ (r : ℝ), 3 * (f a b c r)^2 + 4 * a * (f a b c r) + 2 * b = 0) :=
sorry

end NUMINAMATH_GPT_number_of_roots_l1364_136449


namespace NUMINAMATH_GPT_initial_floors_l1364_136427

-- Define the conditions given in the problem
def austin_time := 60 -- Time Austin takes in seconds to reach the ground floor
def jake_time := 90 -- Time Jake takes in seconds to reach the ground floor
def jake_steps_per_sec := 3 -- Jake descends 3 steps per second
def steps_per_floor := 30 -- There are 30 steps per floor

-- Define the total number of steps Jake descends
def total_jake_steps := jake_time * jake_steps_per_sec

-- Define the number of floors descended in terms of total steps and steps per floor
def num_floors := total_jake_steps / steps_per_floor

-- Theorem stating the number of floors is 9
theorem initial_floors : num_floors = 9 :=
by 
  -- Provide the basic proof structure
  sorry

end NUMINAMATH_GPT_initial_floors_l1364_136427


namespace NUMINAMATH_GPT_problem_solution_l1364_136408

theorem problem_solution (k m : ℕ) (h1 : 30^k ∣ 929260) (h2 : 20^m ∣ 929260) : (3^k - k^3) + (2^m - m^3) = 2 := 
by sorry

end NUMINAMATH_GPT_problem_solution_l1364_136408


namespace NUMINAMATH_GPT_decorations_count_l1364_136429

/-
Danai is decorating her house for Halloween. She puts 12 plastic skulls all around the house.
She has 4 broomsticks, 1 for each side of the front and back doors to the house.
She puts up 12 spiderwebs around various areas of the house.
Danai puts twice as many pumpkins around the house as she put spiderwebs.
She also places a large cauldron on the dining room table.
Danai has the budget left to buy 20 more decorations and has 10 left to put up.
-/

def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def cauldron := 1
def budget_remaining := 20
def undecorated_items := 10

def initial_decorations := plastic_skulls + broomsticks + spiderwebs + pumpkins + cauldron
def additional_decorations := budget_remaining + undecorated_items
def total_decorations := initial_decorations + additional_decorations

theorem decorations_count : total_decorations = 83 := by
  /- Detailed proof steps -/
  sorry

end NUMINAMATH_GPT_decorations_count_l1364_136429


namespace NUMINAMATH_GPT_circumcircle_radius_is_one_l1364_136450

-- Define the basic setup for the triangle with given sides and angles
variables {A B C : Real} -- Angles of the triangle
variables {a b c : Real} -- Sides of the triangle opposite these angles
variable (triangle_ABC : a = Real.sqrt 3 ∧ (c - 2 * b + 2 * Real.sqrt 3 * Real.cos C = 0)) -- Conditions on the sides

-- Define the circumcircle radius
noncomputable def circumcircle_radius (a b c : Real) (A B C : Real) := a / (2 * (Real.sin A))

-- Statement of the problem to be proven
theorem circumcircle_radius_is_one (h : a = Real.sqrt 3)
  (h1 : c - 2 * b + 2 * Real.sqrt 3 * Real.cos C = 0) :
  circumcircle_radius a b c A B C = 1 :=
sorry

end NUMINAMATH_GPT_circumcircle_radius_is_one_l1364_136450


namespace NUMINAMATH_GPT_village_population_rate_l1364_136433

theorem village_population_rate
    (population_X : ℕ := 68000)
    (population_Y : ℕ := 42000)
    (increase_Y : ℕ := 800)
    (years : ℕ := 13) :
  ∃ R : ℕ, population_X - years * R = population_Y + years * increase_Y ∧ R = 1200 :=
by
  exists 1200
  sorry

end NUMINAMATH_GPT_village_population_rate_l1364_136433


namespace NUMINAMATH_GPT_expected_balls_in_original_position_after_two_transpositions_l1364_136463

-- Define the conditions
def num_balls : ℕ := 10

def probs_ball_unchanged : ℚ :=
  (1 / 50) + (16 / 25)

def expected_unchanged_balls (num_balls : ℕ) (probs_ball_unchanged : ℚ) : ℚ :=
  num_balls * probs_ball_unchanged

-- The theorem stating the expected number of balls in original positions
theorem expected_balls_in_original_position_after_two_transpositions
  (num_balls_eq : num_balls = 10)
  (prob_eq : probs_ball_unchanged = (1 / 50) + (16 / 25)) :
  expected_unchanged_balls num_balls probs_ball_unchanged = 7.2 := 
by
  sorry

end NUMINAMATH_GPT_expected_balls_in_original_position_after_two_transpositions_l1364_136463


namespace NUMINAMATH_GPT_determine_angle_A_max_triangle_area_l1364_136425

-- Conditions: acute triangle with sides opposite to angles A, B, C as a, b, c.
variables {A B C a b c : ℝ}
-- Given condition on angles.
axiom angle_condition : 1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * Real.sin ((B + C) / 2) ^ 2 
-- Circumcircle radius
axiom circumcircle_radius : Real.pi > A ∧ A > 0 

-- Question I: Determine angle A
theorem determine_angle_A : A = Real.pi / 3 :=
by sorry

-- Given radius of the circumcircle
noncomputable def R := 2 * Real.sqrt 3 

-- Maximum area of triangle ABC
theorem max_triangle_area (a b c : ℝ) : ∃ area, area = 9 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_determine_angle_A_max_triangle_area_l1364_136425


namespace NUMINAMATH_GPT_rectangle_length_l1364_136455

theorem rectangle_length (P B L : ℝ) (h1 : P = 600) (h2 : B = 200) (h3 : P = 2 * (L + B)) : L = 100 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l1364_136455


namespace NUMINAMATH_GPT_adult_ticket_cost_l1364_136460

variables (x : ℝ)

-- Conditions
def total_tickets := 510
def senior_tickets := 327
def senior_ticket_cost := 15
def total_receipts := 8748

-- Calculation based on the conditions
def adult_tickets := total_tickets - senior_tickets
def senior_receipts := senior_tickets * senior_ticket_cost
def adult_receipts := total_receipts - senior_receipts

-- Define the problem as an assertion to prove
theorem adult_ticket_cost :
  adult_receipts / adult_tickets = 21 := by
  -- Proof steps will go here, but for now, we'll use sorry.
  sorry

end NUMINAMATH_GPT_adult_ticket_cost_l1364_136460


namespace NUMINAMATH_GPT_asimov_books_l1364_136437

theorem asimov_books (h p : Nat) (condition1 : h + p = 12) (condition2 : 30 * h + 20 * p = 300) : h = 6 := by
  sorry

end NUMINAMATH_GPT_asimov_books_l1364_136437


namespace NUMINAMATH_GPT_perp_line_slope_zero_l1364_136446

theorem perp_line_slope_zero {k : ℝ} (h : ∀ x : ℝ, ∃ y : ℝ, y = k * x + 1 ∧ x = 1 → false) : k = 0 :=
sorry

end NUMINAMATH_GPT_perp_line_slope_zero_l1364_136446


namespace NUMINAMATH_GPT_total_hockey_games_l1364_136477

theorem total_hockey_games (games_per_month : ℕ) (months_in_season : ℕ) 
(h1 : games_per_month = 13) (h2 : months_in_season = 14) : 
games_per_month * months_in_season = 182 := 
by
  -- we can simplify using the given conditions
  sorry

end NUMINAMATH_GPT_total_hockey_games_l1364_136477


namespace NUMINAMATH_GPT_sally_book_pages_l1364_136479

def pages_read_weekdays (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def pages_read_weekends (days: ℕ) (pages_per_day: ℕ): ℕ := days * pages_per_day

def total_pages (weekdays: ℕ) (weekends: ℕ) (pages_weekdays: ℕ) (pages_weekends: ℕ): ℕ :=
  pages_read_weekdays weekdays pages_weekdays + pages_read_weekends weekends pages_weekends

theorem sally_book_pages :
  total_pages 10 4 10 20 = 180 :=
sorry

end NUMINAMATH_GPT_sally_book_pages_l1364_136479


namespace NUMINAMATH_GPT_solutions_exist_iff_l1364_136424

variable (a b : ℝ)

theorem solutions_exist_iff :
  (∃ x y : ℝ, (x^2 + y^2 + xy = a) ∧ (x^2 - y^2 = b)) ↔ (-2 * a ≤ Real.sqrt 3 * b ∧ Real.sqrt 3 * b ≤ 2 * a) :=
sorry

end NUMINAMATH_GPT_solutions_exist_iff_l1364_136424


namespace NUMINAMATH_GPT_regression_value_l1364_136436

theorem regression_value (x : ℝ) (y : ℝ) (h : y = 4.75 * x + 2.57) (hx : x = 28) : y = 135.57 :=
by
  sorry

end NUMINAMATH_GPT_regression_value_l1364_136436


namespace NUMINAMATH_GPT_solve_equation_l1364_136440

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l1364_136440


namespace NUMINAMATH_GPT_distance_between_points_l1364_136434

theorem distance_between_points 
  (v_A v_B : ℝ) 
  (d : ℝ) 
  (h1 : 4 * v_A + 4 * v_B = d)
  (h2 : 3.5 * (v_A + 3) + 3.5 * (v_B + 3) = d) : 
  d = 168 := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_points_l1364_136434


namespace NUMINAMATH_GPT_parents_present_l1364_136411

theorem parents_present (pupils teachers total_people parents : ℕ)
  (h_pupils : pupils = 724)
  (h_teachers : teachers = 744)
  (h_total_people : total_people = 1541) :
  parents = total_people - (pupils + teachers) :=
sorry

end NUMINAMATH_GPT_parents_present_l1364_136411


namespace NUMINAMATH_GPT_greatest_integer_not_exceeding_1000x_l1364_136480

-- Given the conditions of the problem
variables (x : ℝ)
-- Cond 1: Edge length of the cube
def edge_length := 2
-- Cond 2: Point light source is x centimeters above a vertex
-- Cond 3: Shadow area excluding the area beneath the cube is 98 square centimeters
def shadow_area_excluding_cube := 98
-- This is the condition total area of the shadow
def total_shadow_area := shadow_area_excluding_cube + edge_length ^ 2

-- Statement: Prove that the greatest integer not exceeding 1000x is 8100:
theorem greatest_integer_not_exceeding_1000x (h1 : total_shadow_area = 102) : x ≤ 8.1 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_not_exceeding_1000x_l1364_136480


namespace NUMINAMATH_GPT_probability_of_edge_endpoints_in_icosahedron_l1364_136468

theorem probability_of_edge_endpoints_in_icosahedron :
  let vertices := 12
  let edges := 30
  let connections_per_vertex := 5
  (5 / (vertices - 1)) = (5 / 11) := by
  sorry

end NUMINAMATH_GPT_probability_of_edge_endpoints_in_icosahedron_l1364_136468


namespace NUMINAMATH_GPT_angle_relationship_l1364_136495

variables {AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1}
variables {angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ}

-- Define the conditions
def conditions (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 : ℝ) : Prop :=
  AB = A_1B_1 ∧ BC = B_1C_1 ∧ CD = C_1D_1 ∧ DA = D_1A_1 ∧ angleA > angleA1

theorem angle_relationship (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ)
  (h : conditions AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 angleA angleA1) :
  angleB < angleB1 ∧ angleC > angleC1 ∧ angleD < angleD1 :=
by {
  sorry
}

end NUMINAMATH_GPT_angle_relationship_l1364_136495


namespace NUMINAMATH_GPT_factorize_expression_l1364_136420

theorem factorize_expression (m : ℝ) : 3 * m^2 - 12 = 3 * (m + 2) * (m - 2) := 
sorry

end NUMINAMATH_GPT_factorize_expression_l1364_136420


namespace NUMINAMATH_GPT_max_tan_B_l1364_136481

theorem max_tan_B (A B : ℝ) (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) (h : Real.tan (A + B) = 2 * Real.tan A) :
  ∃ B_max, B_max = Real.tan B ∧ B_max ≤ Real.sqrt 2 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_tan_B_l1364_136481


namespace NUMINAMATH_GPT_find_number_l1364_136439

theorem find_number (x : ℤ) (h : 3 * (2 * x + 15) = 75) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1364_136439


namespace NUMINAMATH_GPT_nonpositive_sum_of_products_l1364_136485

theorem nonpositive_sum_of_products {a b c d : ℝ} (h : a + b + c + d = 0) :
  ab + ac + ad + bc + bd + cd ≤ 0 :=
sorry

end NUMINAMATH_GPT_nonpositive_sum_of_products_l1364_136485


namespace NUMINAMATH_GPT_interior_angles_of_n_plus_4_sided_polygon_l1364_136486

theorem interior_angles_of_n_plus_4_sided_polygon (n : ℕ) (hn : 180 * (n - 2) = 1800) : 
  180 * (n + 4 - 2) = 2520 :=
by sorry

end NUMINAMATH_GPT_interior_angles_of_n_plus_4_sided_polygon_l1364_136486


namespace NUMINAMATH_GPT_volume_of_cuboid_l1364_136464

-- Definitions of conditions
def side_length : ℕ := 6
def num_cubes : ℕ := 3
def volume_single_cube (side_length : ℕ) : ℕ := side_length ^ 3

-- The main theorem
theorem volume_of_cuboid : (num_cubes * volume_single_cube side_length) = 648 := by
  sorry

end NUMINAMATH_GPT_volume_of_cuboid_l1364_136464


namespace NUMINAMATH_GPT_find_ten_x_l1364_136406

theorem find_ten_x (x : ℝ) 
  (h : 4^(2*x) + 2^(-x) + 1 = (129 + 8 * Real.sqrt 2) * (4^x + 2^(- x) - 2^x)) : 
  10 * x = 35 := 
sorry

end NUMINAMATH_GPT_find_ten_x_l1364_136406


namespace NUMINAMATH_GPT_div_expression_calc_l1364_136448

theorem div_expression_calc :
  (3752 / (39 * 2) + 5030 / (39 * 10) = 61) :=
by
  sorry -- Proof of the theorem

end NUMINAMATH_GPT_div_expression_calc_l1364_136448


namespace NUMINAMATH_GPT_total_mice_eaten_in_decade_l1364_136492

-- Define the number of weeks in a year
def weeks_in_year (is_leap : Bool) : ℕ := if is_leap then 52 else 52

-- Define the number of mice eaten in the first year
def mice_first_year :
  ℕ := weeks_in_year false / 4

-- Define the number of mice eaten in the second year
def mice_second_year :
  ℕ := weeks_in_year false / 3

-- Define the number of mice eaten per year for years 3 to 10
def mice_per_year :
  ℕ := weeks_in_year false / 2

-- Define the total mice eaten in eight years (years 3 to 10)
def mice_eight_years :
  ℕ := 8 * mice_per_year

-- Define the total mice eaten over a decade
def total_mice_eaten :
  ℕ := mice_first_year + mice_second_year + mice_eight_years

-- Theorem to check if the total number of mice equals 238
theorem total_mice_eaten_in_decade :
  total_mice_eaten = 238 :=
by
  -- Calculation for the total number of mice
  sorry

end NUMINAMATH_GPT_total_mice_eaten_in_decade_l1364_136492


namespace NUMINAMATH_GPT_complement_of_union_l1364_136410

open Set

namespace Proof

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of the union of sets A and B with respect to U
theorem complement_of_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 3}) (hB : B = {3, 5}) : 
  U \ (A ∪ B) = {0, 2, 4} :=
by {
  sorry
}

end Proof

end NUMINAMATH_GPT_complement_of_union_l1364_136410


namespace NUMINAMATH_GPT_lisa_photos_last_weekend_l1364_136466

def photos_of_animals : ℕ := 10
def photos_of_flowers : ℕ := 3 * photos_of_animals
def photos_of_scenery : ℕ := photos_of_flowers - 10
def total_photos_this_week : ℕ := photos_of_animals + photos_of_flowers + photos_of_scenery
def photos_last_weekend : ℕ := total_photos_this_week - 15

theorem lisa_photos_last_weekend : photos_last_weekend = 45 :=
by
  sorry

end NUMINAMATH_GPT_lisa_photos_last_weekend_l1364_136466


namespace NUMINAMATH_GPT_height_of_parallelogram_l1364_136473

theorem height_of_parallelogram (area base height : ℝ) (h1 : area = 240) (h2 : base = 24) : height = 10 :=
by
  sorry

end NUMINAMATH_GPT_height_of_parallelogram_l1364_136473


namespace NUMINAMATH_GPT_third_side_correct_length_longest_side_feasibility_l1364_136414

-- Definitions for part (a)
def adjacent_side_length : ℕ := 40
def total_fencing_length : ℕ := 140

-- Define third side given the conditions
def third_side_length : ℕ :=
  total_fencing_length - (2 * adjacent_side_length)

-- Problem (a)
theorem third_side_correct_length (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  third_side_length = 60 :=
sorry

-- Definitions for part (b)
def longest_side_possible1 : ℕ := 85
def longest_side_possible2 : ℕ := 65

-- Problem (b)
theorem longest_side_feasibility (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  ¬ (longest_side_possible1 = 85 ∧ longest_side_possible2 = 65) :=
sorry

end NUMINAMATH_GPT_third_side_correct_length_longest_side_feasibility_l1364_136414


namespace NUMINAMATH_GPT_area_between_circles_l1364_136452

noncomputable def k_value (θ : ℝ) : ℝ := Real.tan θ

theorem area_between_circles {θ k : ℝ} (h₁ : k = Real.tan θ) (h₂ : θ = 4/3) (h_area : (3 * θ / 2) = 2) :
  k = Real.tan (4/3) :=
sorry

end NUMINAMATH_GPT_area_between_circles_l1364_136452


namespace NUMINAMATH_GPT_shaded_area_percentage_is_100_l1364_136493

-- Definitions and conditions
def square_side := 6
def square_area := square_side * square_side

def rect1_area := 2 * 2
def rect2_area := (5 * 5) - (3 * 3)
def rect3_area := 6 * 6

-- Percentage shaded calculation
def shaded_area := square_area
def percentage_shaded := (shaded_area / square_area) * 100

-- Lean 4 statement for the problem
theorem shaded_area_percentage_is_100 :
  percentage_shaded = 100 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_percentage_is_100_l1364_136493


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1364_136474

theorem solve_eq1 (x : ℝ) :
  3 * x^2 - 11 * x + 9 = 0 ↔ x = (11 + Real.sqrt 13) / 6 ∨ x = (11 - Real.sqrt 13) / 6 :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  5 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1364_136474


namespace NUMINAMATH_GPT_older_brother_catches_up_l1364_136412

-- Define the initial conditions and required functions
def younger_brother_steps_before_chase : ℕ := 10
def time_per_3_steps_older := 1  -- in seconds
def time_per_4_steps_younger := 1  -- in seconds 
def dist_older_in_5_steps : ℕ := 7  -- 7d_younger / 5
def dist_younger_in_7_steps : ℕ := 5
def speed_older : ℕ := 3 * dist_older_in_5_steps / 5  -- steps/second 
def speed_younger : ℕ := 4 * dist_younger_in_7_steps / 7  -- steps/second

theorem older_brother_catches_up : ∃ n : ℕ, n = 150 :=
by sorry  -- final theorem statement with proof omitted

end NUMINAMATH_GPT_older_brother_catches_up_l1364_136412


namespace NUMINAMATH_GPT_cannot_factor_polynomial_l1364_136409

theorem cannot_factor_polynomial (a b c d : ℤ) :
  ¬(x^4 + 3 * x^3 + 6 * x^2 + 9 * x + 12 = (x^2 + a * x + b) * (x^2 + c * x + d)) := 
by {
  sorry
}

end NUMINAMATH_GPT_cannot_factor_polynomial_l1364_136409


namespace NUMINAMATH_GPT_geometric_sequence_S6_l1364_136488

-- We first need to ensure our definitions match the given conditions.
noncomputable def a1 : ℝ := 1 -- root of x^2 - 5x + 4 = 0
noncomputable def a3 : ℝ := 4 -- root of x^2 - 5x + 4 = 0

-- Definition of the geometric sequence
noncomputable def q : ℝ := 2 -- common ratio derived from geometric sequence where a3 = a1 * q^2

-- Definition of the n-th term of the geometric sequence
noncomputable def a (n : ℕ) : ℝ := a1 * q^((n : ℝ) - 1)

-- Definition of the sum of the first n terms of the geometric sequence
noncomputable def S (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)

-- The theorem we want to prove
theorem geometric_sequence_S6 : S 6 = 63 :=
  by sorry

end NUMINAMATH_GPT_geometric_sequence_S6_l1364_136488


namespace NUMINAMATH_GPT_f_2017_eq_one_l1364_136431

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x - β)

-- Given conditions
variables {a b α β : ℝ}
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0)
variable (h_f2016 : f 2016 a α b β = -1)

-- The goal
theorem f_2017_eq_one : f 2017 a α b β = 1 :=
sorry

end NUMINAMATH_GPT_f_2017_eq_one_l1364_136431


namespace NUMINAMATH_GPT_compare_f_values_l1364_136467

noncomputable def f : ℝ → ℝ := sorry

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_monotonically_decreasing_on_nonnegative (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → x1 < x2 → f x2 < f x1

axiom even_property : is_even_function f
axiom decreasing_property : is_monotonically_decreasing_on_nonnegative f

theorem compare_f_values : f 3 < f (-2) ∧ f (-2) < f 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_f_values_l1364_136467


namespace NUMINAMATH_GPT_avg_wx_half_l1364_136456

noncomputable def avg_wx {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) : ℝ :=
(w + x) / 2

theorem avg_wx_half {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) :
  avg_wx h1 h2 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_avg_wx_half_l1364_136456


namespace NUMINAMATH_GPT_circle_radius_condition_l1364_136476

theorem circle_radius_condition (c: ℝ):
  (∃ x y : ℝ, (x^2 + y^2 + 4 * x - 2 * y - 5 * c = 0)) → c > -1 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_condition_l1364_136476


namespace NUMINAMATH_GPT_max_area_100_max_fence_length_l1364_136442

noncomputable def maximum_allowable_area (x y : ℝ) : Prop :=
  40 * x + 2 * 45 * y + 20 * x * y ≤ 3200

theorem max_area_100 (x y S : ℝ) (h : maximum_allowable_area x y) :
  S <= 100 :=
sorry

theorem max_fence_length (x y : ℝ) (h : maximum_allowable_area x y) (h1 : x * y = 100) :
  x = 15 :=
sorry

end NUMINAMATH_GPT_max_area_100_max_fence_length_l1364_136442


namespace NUMINAMATH_GPT_total_circles_l1364_136421

theorem total_circles (n : ℕ) (h1 : ∀ k : ℕ, k = n + 14 → n^2 = (k * (k + 1) / 2)) : 
  n = 35 → n^2 = 1225 :=
by
  sorry

end NUMINAMATH_GPT_total_circles_l1364_136421


namespace NUMINAMATH_GPT_function_increasing_l1364_136418

variable {α : Type*} [LinearOrderedField α]

def is_increasing (f : α → α) : Prop :=
  ∀ x y : α, x < y → f x < f y

theorem function_increasing (f : α → α) (h : ∀ x1 x2 : α, x1 ≠ x2 → x1 * f x1 + x2 * f x2 > x1 * f x2 + x2 * f x1) :
  is_increasing f :=
by
  sorry

end NUMINAMATH_GPT_function_increasing_l1364_136418


namespace NUMINAMATH_GPT_breadth_of_rectangle_l1364_136426

theorem breadth_of_rectangle 
  (Perimeter Length Breadth : ℝ)
  (h_perimeter_eq : Perimeter = 2 * (Length + Breadth))
  (h_given_perimeter : Perimeter = 480)
  (h_given_length : Length = 140) :
  Breadth = 100 := 
by
  sorry

end NUMINAMATH_GPT_breadth_of_rectangle_l1364_136426


namespace NUMINAMATH_GPT_nectar_water_percentage_l1364_136404

-- Definitions as per conditions
def nectar_weight : ℝ := 1.2
def honey_weight : ℝ := 1
def honey_water_ratio : ℝ := 0.4

-- Final statement to prove
theorem nectar_water_percentage : (honey_weight * honey_water_ratio + (nectar_weight - honey_weight)) / nectar_weight = 0.5 := by
  sorry

end NUMINAMATH_GPT_nectar_water_percentage_l1364_136404


namespace NUMINAMATH_GPT_area_of_square_plot_l1364_136491

theorem area_of_square_plot (s : ℕ) (price_per_foot total_cost: ℕ)
  (h_price : price_per_foot = 58)
  (h_total_cost : total_cost = 3944) :
  (s * s = 289) :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_plot_l1364_136491


namespace NUMINAMATH_GPT_number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l1364_136457

def is_consonant (c : Char) : Prop :=
  c = 'B' ∨ c = 'C' ∨ c = 'D' ∨ c = 'F'

def count_5_letter_words_with_at_least_one_consonant : Nat :=
  let total_words := 6 ^ 5
  let vowel_words := 2 ^ 5
  total_words - vowel_words

theorem number_of_5_letter_words_with_at_least_one_consonant_equals_7744 :
  count_5_letter_words_with_at_least_one_consonant = 7744 :=
by
  sorry

end NUMINAMATH_GPT_number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l1364_136457


namespace NUMINAMATH_GPT_keith_and_jason_books_l1364_136496

theorem keith_and_jason_books :
  let K := 20
  let J := 21
  K + J = 41 :=
by
  sorry

end NUMINAMATH_GPT_keith_and_jason_books_l1364_136496


namespace NUMINAMATH_GPT_rick_has_eaten_servings_l1364_136499

theorem rick_has_eaten_servings (calories_per_serving block_servings remaining_calories total_calories servings_eaten : ℝ) 
  (h1 : calories_per_serving = 110) 
  (h2 : block_servings = 16) 
  (h3 : remaining_calories = 1210) 
  (h4 : total_calories = block_servings * calories_per_serving)
  (h5 : servings_eaten = (total_calories - remaining_calories) / calories_per_serving) :
  servings_eaten = 5 :=
by 
  sorry

end NUMINAMATH_GPT_rick_has_eaten_servings_l1364_136499


namespace NUMINAMATH_GPT_police_emergency_number_prime_factor_l1364_136498

theorem police_emergency_number_prime_factor (N : ℕ) (h1 : N % 1000 = 133) : 
  ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ N :=
sorry

end NUMINAMATH_GPT_police_emergency_number_prime_factor_l1364_136498


namespace NUMINAMATH_GPT_trig_identity_l1364_136469

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem trig_identity (x : ℝ) (h : f x = 2 * f' x) : 
  (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 := by
  sorry

end NUMINAMATH_GPT_trig_identity_l1364_136469


namespace NUMINAMATH_GPT_screen_time_morning_l1364_136454

def total_screen_time : ℕ := 120
def evening_screen_time : ℕ := 75
def morning_screen_time : ℕ := 45

theorem screen_time_morning : total_screen_time - evening_screen_time = morning_screen_time := by
  sorry

end NUMINAMATH_GPT_screen_time_morning_l1364_136454


namespace NUMINAMATH_GPT_sum_of_fractions_l1364_136403

theorem sum_of_fractions :
  (3 / 50) + (5 / 500) + (7 / 5000) = 0.0714 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1364_136403


namespace NUMINAMATH_GPT_triangle_side_length_l1364_136400

theorem triangle_side_length (B C : Real) (b c : Real) 
  (h1 : c * Real.cos B = 12) 
  (h2 : b * Real.sin C = 5) 
  (h3 : b * Real.sin B = 5) : 
  c = 13 := 
sorry

end NUMINAMATH_GPT_triangle_side_length_l1364_136400


namespace NUMINAMATH_GPT_coefficient_of_x_in_expansion_l1364_136405

theorem coefficient_of_x_in_expansion : 
  (1 + x) * (x - (2 / x)) ^ 3 = 0 :=
sorry

end NUMINAMATH_GPT_coefficient_of_x_in_expansion_l1364_136405


namespace NUMINAMATH_GPT_inequality_for_positive_numbers_l1364_136430

theorem inequality_for_positive_numbers (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a + b) * (a^4 + b^4) ≥ (a^2 + b^2) * (a^3 + b^3) :=
sorry

end NUMINAMATH_GPT_inequality_for_positive_numbers_l1364_136430


namespace NUMINAMATH_GPT_original_number_of_men_l1364_136415

/--A group of men decided to complete a work in 6 days. 
 However, 4 of them became absent, and the remaining men finished the work in 12 days. 
 Given these conditions, we need to prove that the original number of men was 8. --/
theorem original_number_of_men 
  (x : ℕ) -- original number of men
  (h1 : x * 6 = (x - 4) * 12) -- total work remains the same
  : x = 8 := 
sorry

end NUMINAMATH_GPT_original_number_of_men_l1364_136415


namespace NUMINAMATH_GPT_value_of_expression_l1364_136470

theorem value_of_expression {x y z w : ℝ} (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1364_136470


namespace NUMINAMATH_GPT_second_train_length_l1364_136458

noncomputable def length_of_second_train (speed1_kmph speed2_kmph time_sec length1_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed1_kmph + speed2_kmph) * (1000 / 3600)
  let total_distance := relative_speed_mps * time_sec
  total_distance - length1_m

theorem second_train_length :
  length_of_second_train 60 48 9.99920006399488 140 = 159.9760019198464 :=
by
  sorry

end NUMINAMATH_GPT_second_train_length_l1364_136458


namespace NUMINAMATH_GPT_hotel_towels_l1364_136416

theorem hotel_towels (num_rooms : ℕ) (num_people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : num_rooms = 10) (h2 : num_people_per_room = 3) (h3 : towels_per_person = 2) :
  num_rooms * num_people_per_room * towels_per_person = 60 :=
by
  sorry

end NUMINAMATH_GPT_hotel_towels_l1364_136416


namespace NUMINAMATH_GPT_length_of_one_side_of_regular_octagon_l1364_136441

-- Define the conditions of the problem
def is_regular_octagon (n : ℕ) (P : ℝ) (length_of_side : ℝ) : Prop :=
  n = 8 ∧ P = 72 ∧ length_of_side = P / n

-- State the theorem
theorem length_of_one_side_of_regular_octagon : is_regular_octagon 8 72 9 :=
by
  -- The proof is omitted; only the statement is required
  sorry

end NUMINAMATH_GPT_length_of_one_side_of_regular_octagon_l1364_136441


namespace NUMINAMATH_GPT_angles_sum_132_l1364_136428

theorem angles_sum_132
  (D E F p q : ℝ)
  (hD : D = 38)
  (hE : E = 58)
  (hF : F = 36)
  (five_sided_angle_sum : D + E + (360 - p) + 90 + (126 - q) = 540) : 
  p + q = 132 := 
by
  sorry

end NUMINAMATH_GPT_angles_sum_132_l1364_136428


namespace NUMINAMATH_GPT_nth_inequality_l1364_136401

theorem nth_inequality (n : ℕ) (x : ℝ) (h : x > 0) : x + n^n / x^n ≥ n + 1 := 
sorry

end NUMINAMATH_GPT_nth_inequality_l1364_136401


namespace NUMINAMATH_GPT_geometric_sequence_a6_l1364_136490

theorem geometric_sequence_a6 (a : ℕ → ℕ) (r : ℕ)
  (h₁ : a 1 = 1)
  (h₄ : a 4 = 8)
  (h_geometric : ∀ n, a n = a 1 * r^(n-1)) : 
  a 6 = 32 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l1364_136490


namespace NUMINAMATH_GPT_determine_OP_l1364_136407

theorem determine_OP 
  (a b c d k : ℝ)
  (h1 : k * b ≤ c) 
  (h2 : (A : ℝ) = a)
  (h3 : (B : ℝ) = k * b)
  (h4 : (C : ℝ) = c)
  (h5 : (D : ℝ) = k * d)
  (AP_PD : ∀ (P : ℝ), (a - P) / (P - k * d) = k * (k * b - P) / (P - c))
  :
  ∃ P : ℝ, P = (a * c + k * b * d) / (a + c - k * b + k * d - 1 + k) :=
sorry

end NUMINAMATH_GPT_determine_OP_l1364_136407


namespace NUMINAMATH_GPT_probability_abs_diff_l1364_136444

variables (P : ℕ → ℚ) (m : ℚ)

def is_probability_distribution : Prop :=
  P 1 = m ∧ P 2 = 1/4 ∧ P 3 = 1/4 ∧ P 4 = 1/3 ∧ m + 1/4 + 1/4 + 1/3 = 1

theorem probability_abs_diff (h : is_probability_distribution P m) :
  P 1 + P 3 = 5 / 12 :=
by 
sorry

end NUMINAMATH_GPT_probability_abs_diff_l1364_136444


namespace NUMINAMATH_GPT_daily_rental_cost_l1364_136451

theorem daily_rental_cost (x : ℝ) (total_cost miles : ℝ)
  (cost_per_mile : ℝ) (daily_cost : ℝ) :
  total_cost = daily_cost + cost_per_mile * miles →
  total_cost = 46.12 →
  miles = 214 →
  cost_per_mile = 0.08 →
  daily_cost = 29 :=
by
  sorry

end NUMINAMATH_GPT_daily_rental_cost_l1364_136451


namespace NUMINAMATH_GPT_problem1_problem2_l1364_136489

variable (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
def f (x : ℝ) : ℝ := abs (x - a) + 2 * abs (x + b)

theorem problem1 (h3 : ∃ x, f x = 1) : a + b = 1 := sorry

theorem problem2 (h4 : a + b = 1) (m : ℝ) (h5 : ∀ m, m ≤ 1/a + 2/b)
: m ≤ 3 + 2 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1364_136489


namespace NUMINAMATH_GPT_algorithm_output_l1364_136484

theorem algorithm_output (x y: Int) (h_x: x = -5) (h_y: y = 15) : 
  let x := if x < 0 then y + 3 else x;
  x - y = 3 ∧ x + y = 33 :=
by
  sorry

end NUMINAMATH_GPT_algorithm_output_l1364_136484


namespace NUMINAMATH_GPT_sector_area_correct_l1364_136443

noncomputable def sector_area (r α : ℝ) : ℝ :=
  (1 / 2) * r^2 * α

theorem sector_area_correct :
  sector_area 3 2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_correct_l1364_136443


namespace NUMINAMATH_GPT_cardProblem_l1364_136417

structure InitialState where
  jimmy_cards : ℕ
  bob_cards : ℕ
  sarah_cards : ℕ

structure UpdatedState where
  jimmy_cards_final : ℕ
  sarah_cards_final : ℕ
  sarahs_friends_cards : ℕ

def cardProblemSolved (init : InitialState) (final : UpdatedState) : Prop :=
  let bob_initial := init.bob_cards + 6
  let bob_to_sarah := bob_initial / 3
  let bob_final := bob_initial - bob_to_sarah
  let sarah_initial := init.sarah_cards + bob_to_sarah
  let sarah_friends := sarah_initial / 3
  let sarah_final := sarah_initial - 3 * sarah_friends
  let mary_cards := 2 * 6
  let jimmy_final := init.jimmy_cards - 6 - mary_cards
  let sarah_to_tim := 0 -- since Sarah can't give fractional cards
  (final.jimmy_cards_final = jimmy_final) ∧ 
  (final.sarah_cards_final = sarah_final - sarah_to_tim) ∧ 
  (final.sarahs_friends_cards = sarah_friends)

theorem cardProblem : 
  cardProblemSolved 
    { jimmy_cards := 68, bob_cards := 5, sarah_cards := 7 }
    { jimmy_cards_final := 50, sarah_cards_final := 1, sarahs_friends_cards := 3 } :=
by 
  sorry

end NUMINAMATH_GPT_cardProblem_l1364_136417


namespace NUMINAMATH_GPT_cos_difference_identity_l1364_136461

theorem cos_difference_identity (α β : ℝ) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : Real.sin β = 5 / 13) : Real.cos (α - β) = 63 / 65 := 
by 
  sorry

end NUMINAMATH_GPT_cos_difference_identity_l1364_136461


namespace NUMINAMATH_GPT_tax_calculation_l1364_136453

variable (winnings : ℝ) (processing_fee : ℝ) (take_home : ℝ)
variable (tax_percentage : ℝ)

def given_conditions : Prop :=
  winnings = 50 ∧ processing_fee = 5 ∧ take_home = 35

def to_prove : Prop :=
  tax_percentage = 20

theorem tax_calculation (h : given_conditions winnings processing_fee take_home) : to_prove tax_percentage :=
by
  sorry

end NUMINAMATH_GPT_tax_calculation_l1364_136453


namespace NUMINAMATH_GPT_part1_part2_l1364_136413

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + 2*a*x + 2

theorem part1 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 5 → f x a > 3*a*x) → a < 2*Real.sqrt 2 :=
sorry

theorem part2 (a : ℝ) :
  ∀ x : ℝ,
    ((a = 0) → x > 2) ∧
    ((a > 0) → (x < -1/a ∨ x > 2)) ∧
    ((-1/2 < a ∧ a < 0) → (2 < x ∧ x < -1/a)) ∧
    ((a = -1/2) → false) ∧
    ((a < -1/2) → (-1/a < x ∧ x < 2)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1364_136413
