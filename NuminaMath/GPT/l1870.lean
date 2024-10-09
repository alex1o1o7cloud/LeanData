import Mathlib

namespace trapezoid_diagonal_comparison_l1870_187074

variable {A B C D: Type}
variable (α β : Real) -- Representing angles
variable (AB CD BD AC : Real) -- Representing lengths of sides and diagonals
variable (h : Real) -- Height
variable (A' B' : Real) -- Projections

noncomputable def trapezoid (AB CD: Real) := True -- Trapezoid definition placeholder
noncomputable def angle_relation (α β : Real) := α < β -- Angle relationship

theorem trapezoid_diagonal_comparison
  (trapezoid_ABCD: trapezoid AB CD)
  (angle_relation_ABC_DCB : angle_relation α β)
  : BD > AC :=
sorry

end trapezoid_diagonal_comparison_l1870_187074


namespace find_number_l1870_187085

theorem find_number (x : ℤ) (h : 3 * (2 * x + 15) = 75) : x = 5 :=
by
  sorry

end find_number_l1870_187085


namespace find_A_find_B_l1870_187035

-- First problem: Prove A = 10 given 100A = 35^2 - 15^2
theorem find_A (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) : A = 10 := by
  sorry

-- Second problem: Prove B = 4 given (A-1)^6 = 27^B and A = 10
theorem find_B (B : ℕ) (A : ℕ) (h₁ : 100 * A = 35 ^ 2 - 15 ^ 2) (h₂ : (A - 1) ^ 6 = 27 ^ B) : B = 4 := by
  have A_is_10 : A = 10 := by
    apply find_A
    assumption
  sorry

end find_A_find_B_l1870_187035


namespace nancy_kept_tortilla_chips_l1870_187055

theorem nancy_kept_tortilla_chips (initial_chips : ℕ) (chips_to_brother : ℕ) (chips_to_sister : ℕ) (remaining_chips : ℕ) 
  (h1 : initial_chips = 22) 
  (h2 : chips_to_brother = 7) 
  (h3 : chips_to_sister = 5) 
  (h_total_given : initial_chips - (chips_to_brother + chips_to_sister) = remaining_chips) :
  remaining_chips = 10 :=
sorry

end nancy_kept_tortilla_chips_l1870_187055


namespace lisa_photos_last_weekend_l1870_187077

def photos_of_animals : ℕ := 10
def photos_of_flowers : ℕ := 3 * photos_of_animals
def photos_of_scenery : ℕ := photos_of_flowers - 10
def total_photos_this_week : ℕ := photos_of_animals + photos_of_flowers + photos_of_scenery
def photos_last_weekend : ℕ := total_photos_this_week - 15

theorem lisa_photos_last_weekend : photos_last_weekend = 45 :=
by
  sorry

end lisa_photos_last_weekend_l1870_187077


namespace sqrt_9_eq_3_and_neg3_l1870_187049

theorem sqrt_9_eq_3_and_neg3 : { x : ℝ | x^2 = 9 } = {3, -3} :=
by
  sorry

end sqrt_9_eq_3_and_neg3_l1870_187049


namespace pink_highlighters_count_l1870_187052

-- Definitions for the problem's conditions
def total_highlighters : Nat := 11
def yellow_highlighters : Nat := 2
def blue_highlighters : Nat := 5
def non_pink_highlighters : Nat := yellow_highlighters + blue_highlighters

-- Statement of the problem as a theorem
theorem pink_highlighters_count : total_highlighters - non_pink_highlighters = 4 :=
by
  sorry

end pink_highlighters_count_l1870_187052


namespace find_value_of_expression_l1870_187047

theorem find_value_of_expression (a b c : ℝ) (h : (2*a - 6)^2 + (3*b - 9)^2 + (4*c - 12)^2 = 0) : a + 2*b + 3*c = 18 := 
sorry

end find_value_of_expression_l1870_187047


namespace screen_time_morning_l1870_187081

def total_screen_time : ℕ := 120
def evening_screen_time : ℕ := 75
def morning_screen_time : ℕ := 45

theorem screen_time_morning : total_screen_time - evening_screen_time = morning_screen_time := by
  sorry

end screen_time_morning_l1870_187081


namespace solution_set_of_inequality_l1870_187067

variables {R : Type*} [LinearOrderedField R]

-- Define f as an even function
def even_function (f : R → R) := ∀ x : R, f x = f (-x)

-- Define f as an increasing function on [0, +∞)
def increasing_on_nonneg (f : R → R) := ∀ ⦃x y : R⦄, 0 ≤ x → x ≤ y → f x ≤ f y

-- Define the hypothesis and the theorem
theorem solution_set_of_inequality (f : R → R)
  (h_even : even_function f)
  (h_inc : increasing_on_nonneg f) :
  { x : R | f x > f 1 } = { x : R | x > 1 ∨ x < -1 } :=
by {
  sorry
}

end solution_set_of_inequality_l1870_187067


namespace equal_a_b_l1870_187029

theorem equal_a_b (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_n : 0 < n) 
  (h_eq : (a + b)^n - (a - b)^n = (a / b) * ((a + b)^n + (a - b)^n)) : a = b :=
sorry

end equal_a_b_l1870_187029


namespace ratio_odd_even_divisors_l1870_187010

def sum_of_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of divisors

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of odd divisors

def sum_of_even_divisors (n : ℕ) : ℕ := sorry -- This should be implemented as a function that calculates sum of even divisors

theorem ratio_odd_even_divisors (M : ℕ) (h : M = 36 * 36 * 98 * 210) :
  sum_of_odd_divisors M / sum_of_even_divisors M = 1 / 60 :=
by {
  sorry
}

end ratio_odd_even_divisors_l1870_187010


namespace tory_video_games_l1870_187019

theorem tory_video_games (T J: ℕ) :
    (3 * J + 5 = 11) → (J = T / 3) → T = 6 :=
by
  sorry

end tory_video_games_l1870_187019


namespace distance_between_points_l1870_187078

theorem distance_between_points 
  (v_A v_B : ℝ) 
  (d : ℝ) 
  (h1 : 4 * v_A + 4 * v_B = d)
  (h2 : 3.5 * (v_A + 3) + 3.5 * (v_B + 3) = d) : 
  d = 168 := 
by 
  sorry

end distance_between_points_l1870_187078


namespace adult_ticket_cost_l1870_187091

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

end adult_ticket_cost_l1870_187091


namespace arithmetic_mean_of_distribution_l1870_187004

-- Defining conditions
def stddev : ℝ := 2.3
def value : ℝ := 11.6

-- Proving the mean (μ) is 16.2
theorem arithmetic_mean_of_distribution : ∃ μ : ℝ, μ = 16.2 ∧ value = μ - 2 * stddev :=
by
  use 16.2
  sorry

end arithmetic_mean_of_distribution_l1870_187004


namespace min_value_expression_l1870_187063

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 * b + b^2 * c + c^2 * a = 3) :
  ∃ A : ℝ, A = 3 * Real.sqrt 2 ∧ 
  (A = (Real.sqrt (a^6 + b^4 * c^6) / b) + 
       (Real.sqrt (b^6 + c^4 * a^6) / c) + 
       (Real.sqrt (c^6 + a^4 * b^6) / a)) :=
sorry

end min_value_expression_l1870_187063


namespace expected_balls_in_original_position_after_two_transpositions_l1870_187093

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

end expected_balls_in_original_position_after_two_transpositions_l1870_187093


namespace sum_of_intercepts_l1870_187043

theorem sum_of_intercepts (x y : ℝ) (h : x / 3 - y / 4 = 1) : (x / 3 = 1 ∧ y / (-4) = 1) → 3 + (-4) = -1 :=
by
  sorry

end sum_of_intercepts_l1870_187043


namespace white_rabbit_hop_distance_per_minute_l1870_187041

-- Definitions for given conditions
def brown_hop_per_minute : ℕ := 12
def total_distance_in_5_minutes : ℕ := 135
def brown_distance_in_5_minutes : ℕ := 5 * brown_hop_per_minute

-- The statement we need to prove
theorem white_rabbit_hop_distance_per_minute (W : ℕ) (h1 : brown_hop_per_minute = 12) (h2 : total_distance_in_5_minutes = 135) :
  W = 15 :=
by
  sorry

end white_rabbit_hop_distance_per_minute_l1870_187041


namespace min_cuts_for_30_sided_polygons_l1870_187059

theorem min_cuts_for_30_sided_polygons (n : ℕ) (h : n = 73) : 
  ∃ k : ℕ, (∀ m : ℕ, m < k → (m + 1) ≤ 2 * m - 1972) ∧ (k = 1970) :=
sorry

end min_cuts_for_30_sided_polygons_l1870_187059


namespace angle_perpendicular_coterminal_l1870_187090

theorem angle_perpendicular_coterminal (α β : ℝ) (k : ℤ) 
  (h_perpendicular : ∃ k, β = α + 90 + k * 360 ∨ β = α - 90 + k * 360) : 
  β = α + 90 + k * 360 ∨ β = α - 90 + k * 360 :=
sorry

end angle_perpendicular_coterminal_l1870_187090


namespace john_spending_l1870_187040

theorem john_spending (X : ℝ) 
  (H1 : X * (1 / 4) + X * (1 / 3) + X * (1 / 6) + 6 = X) : 
  X = 24 := 
sorry

end john_spending_l1870_187040


namespace option_b_correct_l1870_187086

theorem option_b_correct (a : ℝ) : (a ^ 3) * (a ^ 2) = a ^ 5 := 
by
  sorry

end option_b_correct_l1870_187086


namespace timeAfter2687Minutes_l1870_187061

-- We define a structure for representing time in hours and minutes.
structure Time :=
  (hour : Nat)
  (minute : Nat)

-- Define the current time
def currentTime : Time := {hour := 7, minute := 0}

-- Define a function that computes the time after adding a given number of minutes to a given time
noncomputable def addMinutes (t : Time) (minutesToAdd : Nat) : Time :=
  let totalMinutes := t.minute + minutesToAdd
  let extraHours := totalMinutes / 60
  let remainingMinutes := totalMinutes % 60
  let totalHours := t.hour + extraHours
  let effectiveHours := totalHours % 24
  {hour := effectiveHours, minute := remainingMinutes}

-- The theorem to state that 2687 minutes after 7:00 a.m. is 3:47 a.m.
theorem timeAfter2687Minutes : addMinutes currentTime 2687 = { hour := 3, minute := 47 } :=
  sorry

end timeAfter2687Minutes_l1870_187061


namespace lcm_inequality_l1870_187096

theorem lcm_inequality (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  Nat.lcm k m * Nat.lcm m n * Nat.lcm n k ≥ Nat.lcm (Nat.lcm k m) n ^ 2 :=
by sorry

end lcm_inequality_l1870_187096


namespace yard_length_is_correct_l1870_187048

-- Definitions based on the conditions
def trees : ℕ := 26
def distance_between_trees : ℕ := 11

-- Theorem stating that the length of the yard is 275 meters
theorem yard_length_is_correct : (trees - 1) * distance_between_trees = 275 :=
by sorry

end yard_length_is_correct_l1870_187048


namespace diff_of_squares_525_475_l1870_187007

theorem diff_of_squares_525_475 : 525^2 - 475^2 = 50000 := by
  sorry

end diff_of_squares_525_475_l1870_187007


namespace circumcircle_radius_is_one_l1870_187098

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

end circumcircle_radius_is_one_l1870_187098


namespace determine_angle_A_max_triangle_area_l1870_187083

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

end determine_angle_A_max_triangle_area_l1870_187083


namespace total_area_of_sheet_l1870_187064

theorem total_area_of_sheet (A B : ℝ) (h1 : A = 4 * B) (h2 : A = B + 2208) : A + B = 3680 :=
by
  sorry

end total_area_of_sheet_l1870_187064


namespace solutions_exist_iff_l1870_187097

variable (a b : ℝ)

theorem solutions_exist_iff :
  (∃ x y : ℝ, (x^2 + y^2 + xy = a) ∧ (x^2 - y^2 = b)) ↔ (-2 * a ≤ Real.sqrt 3 * b ∧ Real.sqrt 3 * b ≤ 2 * a) :=
sorry

end solutions_exist_iff_l1870_187097


namespace quoted_value_stock_l1870_187037

-- Define the conditions
def face_value : ℕ := 100
def dividend_percentage : ℝ := 0.14
def yield_percentage : ℝ := 0.1

-- Define the computed dividend per share
def dividend_per_share : ℝ := dividend_percentage * face_value

-- State the theorem to prove the quoted value
theorem quoted_value_stock : (dividend_per_share / yield_percentage) * 100 = 140 :=
by
  sorry  -- Placeholder for the proof

end quoted_value_stock_l1870_187037


namespace min_value_of_z_l1870_187016

-- Define the conditions as separate hypotheses.
variable (x y : ℝ)

def condition1 : Prop := x - y + 1 ≥ 0
def condition2 : Prop := x + y - 1 ≥ 0
def condition3 : Prop := x ≤ 3

-- Define the objective function.
def z : ℝ := 2 * x - 3 * y

-- State the theorem to prove the minimum value of z given the conditions.
theorem min_value_of_z (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x) :
  ∃ x y, condition1 x y ∧ condition2 x y ∧ condition3 x ∧ z x y = -6 :=
sorry

end min_value_of_z_l1870_187016


namespace student_selection_l1870_187053

theorem student_selection (a b c : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : c = 4) : a + b + c = 12 :=
by {
  sorry
}

end student_selection_l1870_187053


namespace intersection_points_count_l1870_187012

noncomputable def f1 (x : ℝ) : ℝ := abs (3 * x - 2)
noncomputable def f2 (x : ℝ) : ℝ := -abs (2 * x + 5)

theorem intersection_points_count : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f1 x1 = f2 x1 ∧ f1 x2 = f2 x2 ∧ 
    (∀ x : ℝ, f1 x = f2 x → x = x1 ∨ x = x2)) :=
sorry

end intersection_points_count_l1870_187012


namespace triangle_inequality_expression_non_negative_l1870_187031

theorem triangle_inequality_expression_non_negative
  (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end triangle_inequality_expression_non_negative_l1870_187031


namespace proof_problem_l1870_187042

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.sin x + b * x^3 + 4

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * Real.cos x + 3 * b * x^2

theorem proof_problem (a b : ℝ) :
  f 2016 a b + f (-2016) a b + f' 2017 a b - f' (-2017) a b = 8 := by
  sorry

end proof_problem_l1870_187042


namespace solve_equation_l1870_187087

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 := by
  sorry

end solve_equation_l1870_187087


namespace triangle_angle_C_and_area_l1870_187018

theorem triangle_angle_C_and_area (A B C : ℝ) (a b c : ℝ) 
  (h1 : 2 * c * Real.cos B = 2 * a - b)
  (h2 : c = Real.sqrt 3)
  (h3 : b - a = 1) :
  (C = Real.pi / 3) ∧
  (1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2) :=
by
  sorry

end triangle_angle_C_and_area_l1870_187018


namespace rectangle_length_l1870_187082

theorem rectangle_length (P B L : ℝ) (h1 : P = 600) (h2 : B = 200) (h3 : P = 2 * (L + B)) : L = 100 :=
by
  sorry

end rectangle_length_l1870_187082


namespace expression_value_l1870_187046

noncomputable def evaluate_expression : ℝ :=
  Real.logb 2 (3 * 11 + Real.exp (4 - 8)) + 3 * Real.sin (Real.pi^2 - Real.sqrt ((6 * 4) / 3 - 4))

theorem expression_value : evaluate_expression = 3.832 := by
  sorry

end expression_value_l1870_187046


namespace expected_socks_pairs_l1870_187011

noncomputable def expected_socks (n : ℕ) : ℝ :=
2 * n

theorem expected_socks_pairs (n : ℕ) :
  @expected_socks n = 2 * n :=
by
  sorry

end expected_socks_pairs_l1870_187011


namespace find_f_of_7_l1870_187036

-- Defining the conditions in the problem.
variables (f : ℝ → ℝ)
variables (odd_f : ∀ x : ℝ, f (-x) = -f x)
variables (periodic_f : ∀ x : ℝ, f (x + 4) = f x)
variables (f_eqn : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = x + 2)

-- The statement of the problem, to prove f(7) = -3.
theorem find_f_of_7 : f 7 = -3 :=
by
  sorry

end find_f_of_7_l1870_187036


namespace B_join_time_l1870_187039

theorem B_join_time (x : ℕ) (hx : (45000 * 12) / (27000 * (12 - x)) = 2) : x = 2 :=
sorry

end B_join_time_l1870_187039


namespace rectangle_same_color_exists_l1870_187068

theorem rectangle_same_color_exists (grid : Fin 3 → Fin 7 → Bool) : 
  ∃ (r1 r2 c1 c2 : Fin 3), r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid r1 c1 = grid r1 c2 ∧ grid r1 c1 = grid r2 c1 ∧ grid r1 c1 = grid r2 c2 :=
by
  sorry

end rectangle_same_color_exists_l1870_187068


namespace cos_difference_identity_l1870_187092

theorem cos_difference_identity (α β : ℝ) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : Real.sin β = 5 / 13) : Real.cos (α - β) = 63 / 65 := 
by 
  sorry

end cos_difference_identity_l1870_187092


namespace simplify_fraction_l1870_187044

-- Define what it means for a fraction to be in simplest form
def coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

-- Define what it means for a fraction to be reducible
def reducible_fraction (num den : ℕ) : Prop := ∃ d > 1, d ∣ num ∧ d ∣ den

-- Main theorem statement
theorem simplify_fraction 
  (m n : ℕ) (h_coprime : coprime m n) 
  (h_reducible : reducible_fraction (4 * m + 3 * n) (5 * m + 2 * n)) : ∃ d, d = 7 :=
by {
  sorry
}

end simplify_fraction_l1870_187044


namespace find_b_l1870_187002

noncomputable def p (x : ℝ) : ℝ := 3 * x - 8
noncomputable def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

theorem find_b (b : ℝ) : p (q 3 b) = 10 → b = 6 :=
by
  unfold p q
  intro h
  sorry

end find_b_l1870_187002


namespace paul_books_sold_l1870_187051

theorem paul_books_sold:
  ∀ (initial_books friend_books sold_per_day days final_books sold_books: ℝ),
    initial_books = 284.5 →
    friend_books = 63.7 →
    sold_per_day = 16.25 →
    days = 8 →
    final_books = 112.3 →
    sold_books = initial_books - friend_books - final_books →
    sold_books = 108.5 :=
by intros initial_books friend_books sold_per_day days final_books sold_books
   sorry

end paul_books_sold_l1870_187051


namespace volume_of_cuboid_l1870_187094

-- Definitions of conditions
def side_length : ℕ := 6
def num_cubes : ℕ := 3
def volume_single_cube (side_length : ℕ) : ℕ := side_length ^ 3

-- The main theorem
theorem volume_of_cuboid : (num_cubes * volume_single_cube side_length) = 648 := by
  sorry

end volume_of_cuboid_l1870_187094


namespace cans_for_credit_l1870_187023

theorem cans_for_credit (P C R : ℕ) : 
  (3 * P = 2 * C) → (C ≠ 0) → (R ≠ 0) → P * R / C = (P * R / C : ℕ) :=
by
  intros h1 h2 h3
  -- proof required here
  sorry

end cans_for_credit_l1870_187023


namespace max_writers_at_conference_l1870_187005

variables (T E W x : ℕ)

-- Defining the conditions
def conference_conditions (T E W x : ℕ) : Prop :=
  T = 90 ∧ E > 38 ∧ x ≤ 6 ∧ 2 * x + (W + E - x) = T ∧ W = T - E - x

-- Statement to prove the number of writers
theorem max_writers_at_conference : ∃ W, conference_conditions 90 39 W 1 :=
by
  sorry

end max_writers_at_conference_l1870_187005


namespace value_of_a_star_b_l1870_187076

theorem value_of_a_star_b (a b : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  (1 / (a : ℚ) + 1 / (b : ℚ)) = 5 / 12 := by
  sorry

end value_of_a_star_b_l1870_187076


namespace fraction_simplifies_l1870_187089

theorem fraction_simplifies :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end fraction_simplifies_l1870_187089


namespace value_two_sd_below_mean_l1870_187062

theorem value_two_sd_below_mean :
  let mean := 14.5
  let stdev := 1.7
  mean - 2 * stdev = 11.1 :=
by
  sorry

end value_two_sd_below_mean_l1870_187062


namespace decorations_count_l1870_187088

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

end decorations_count_l1870_187088


namespace evaluate_expression_l1870_187034

theorem evaluate_expression : (2014 - 2013) * (2013 - 2012) = 1 := 
by sorry

end evaluate_expression_l1870_187034


namespace good_carrots_total_l1870_187008

-- Define the number of carrots picked by Carol and her mother
def carolCarrots := 29
def motherCarrots := 16

-- Define the number of bad carrots
def badCarrots := 7

-- Define the total number of carrots picked by Carol and her mother
def totalCarrots := carolCarrots + motherCarrots

-- Define the total number of good carrots
def goodCarrots := totalCarrots - badCarrots

-- The theorem to prove that the total number of good carrots is 38
theorem good_carrots_total : goodCarrots = 38 := by
  sorry

end good_carrots_total_l1870_187008


namespace students_neither_l1870_187028

-- Define the conditions
def total_students : ℕ := 60
def students_math : ℕ := 40
def students_physics : ℕ := 35
def students_both : ℕ := 25

-- Define the problem statement
theorem students_neither : total_students - ((students_math - students_both) + (students_physics - students_both) + students_both) = 10 :=
by
  sorry

end students_neither_l1870_187028


namespace player_A_elimination_after_third_round_at_least_one_player_passes_all_l1870_187045

-- Define probabilities for Player A's success in each round
def P_A1 : ℚ := 4 / 5
def P_A2 : ℚ := 3 / 4
def P_A3 : ℚ := 2 / 3

-- Define probabilities for Player B's success in each round
def P_B1 : ℚ := 2 / 3
def P_B2 : ℚ := 2 / 3
def P_B3 : ℚ := 1 / 2

-- Define theorems
theorem player_A_elimination_after_third_round :
  P_A1 * P_A2 * (1 - P_A3) = 1 / 5 := by
  sorry

theorem at_least_one_player_passes_all :
  1 - ((1 - (P_A1 * P_A2 * P_A3)) * (1 - (P_B1 * P_B2 * P_B3))) = 8 / 15 := by
  sorry


end player_A_elimination_after_third_round_at_least_one_player_passes_all_l1870_187045


namespace new_student_bmi_l1870_187001

theorem new_student_bmi 
(average_weight_29 : ℚ)
(average_height_29 : ℚ)
(average_weight_30 : ℚ)
(average_height_30 : ℚ)
(new_student_height : ℚ)
(bmi : ℚ)
(h1 : average_weight_29 = 28)
(h2 : average_height_29 = 1.5)
(h3 : average_weight_30 = 27.5)
(h4 : average_height_30 = 1.5)
(h5 : new_student_height = 1.4)
: bmi = 6.63 := 
sorry

end new_student_bmi_l1870_187001


namespace seating_arrangements_l1870_187073

-- Define the participants
inductive Person : Type
| xiaoMing
| parent1
| parent2
| grandparent1
| grandparent2

open Person

-- Define the function to count seating arrangements
noncomputable def count_seating_arrangements : Nat :=
  let arrangements := [
    -- (Only one parent next to Xiao Ming, parents not next to each other)
    12,
    -- (Only one parent next to Xiao Ming, parents next to each other)
    24,
    -- (Both parents next to Xiao Ming)
    12
  ]
  arrangements.foldr (· + ·) 0

theorem seating_arrangements : count_seating_arrangements = 48 := by
  sorry

end seating_arrangements_l1870_187073


namespace factorize_m_cubed_minus_16m_l1870_187058

theorem factorize_m_cubed_minus_16m (m : ℝ) : m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_m_cubed_minus_16m_l1870_187058


namespace ones_digit_of_sum_of_powers_l1870_187026

theorem ones_digit_of_sum_of_powers :
  (1^2011 + 2^2011 + 3^2011 + 4^2011 + 5^2011 + 6^2011 + 7^2011 + 8^2011 + 9^2011 + 10^2011) % 10 = 5 :=
by
  sorry

end ones_digit_of_sum_of_powers_l1870_187026


namespace polygon_sides_l1870_187038

/-- 
A regular polygon with interior angles of 160 degrees has 18 sides.
-/
theorem polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (interior_angle : ℝ) = 160) : n = 18 := 
by
  have angle_sum : 180 * (n - 2) = 160 * n := 
    by sorry
  have eq_sides : n = 18 := 
    by sorry
  exact eq_sides

end polygon_sides_l1870_187038


namespace determine_original_price_l1870_187075

namespace PriceProblem

variable (x : ℝ)

def final_price (x : ℝ) : ℝ := 0.98175 * x

theorem determine_original_price (h : final_price x = 100) : x = 101.86 :=
by
  sorry

end PriceProblem

end determine_original_price_l1870_187075


namespace initial_ratio_l1870_187033

theorem initial_ratio (partners associates associates_after_hiring : ℕ)
  (h_partners : partners = 20)
  (h_associates_after_hiring : associates_after_hiring = 20 * 34)
  (h_assoc_equation : associates + 50 = associates_after_hiring) :
  (partners : ℚ) / associates = 2 / 63 :=
by
  sorry

end initial_ratio_l1870_187033


namespace village_population_rate_l1870_187099

theorem village_population_rate
    (population_X : ℕ := 68000)
    (population_Y : ℕ := 42000)
    (increase_Y : ℕ := 800)
    (years : ℕ := 13) :
  ∃ R : ℕ, population_X - years * R = population_Y + years * increase_Y ∧ R = 1200 :=
by
  exists 1200
  sorry

end village_population_rate_l1870_187099


namespace jill_spent_on_clothing_l1870_187030

-- Define the total amount spent excluding taxes, T.
variable (T : ℝ)
-- Define the percentage of T Jill spent on clothing, C.
variable (C : ℝ)

-- Define the conditions based on the problem statement.
def jill_tax_conditions : Prop :=
  let food_percent := 0.20
  let other_items_percent := 0.30
  let clothing_tax := 0.04
  let food_tax := 0
  let other_tax := 0.10
  let total_tax := 0.05
  let food_amount := food_percent * T
  let other_items_amount := other_items_percent * T
  let clothing_amount := C * T
  let clothing_tax_amount := clothing_tax * clothing_amount
  let other_tax_amount := other_tax * other_items_amount
  let total_tax_amount := clothing_tax_amount + food_tax * food_amount + other_tax_amount
  C * T + food_percent * T + other_items_percent * T = T ∧
  total_tax_amount / T = total_tax

-- The goal is to prove that C = 0.50.
theorem jill_spent_on_clothing (h : jill_tax_conditions T C) : C = 0.50 :=
by
  sorry

end jill_spent_on_clothing_l1870_187030


namespace rick_savings_ratio_proof_l1870_187069

-- Define the conditions
def erika_savings : ℤ := 155
def cost_of_gift : ℤ := 250
def cost_of_cake : ℤ := 25
def amount_left : ℤ := 5

-- Define the total amount they have together
def total_amount : ℤ := cost_of_gift + cost_of_cake - amount_left

-- Define Rick's savings based on the conditions
def rick_savings : ℤ := total_amount - erika_savings

-- Define the ratio of Rick's savings to the cost of the gift
def rick_gift_ratio : ℚ := rick_savings / cost_of_gift

-- Prove the ratio is 23/50
theorem rick_savings_ratio_proof : rick_gift_ratio = 23 / 50 :=
  by
    have h1 : total_amount = 270 := by sorry
    have h2 : rick_savings = 115 := by sorry
    have h3 : rick_gift_ratio = 23 / 50 := by sorry
    exact h3

end rick_savings_ratio_proof_l1870_187069


namespace probability_of_edge_endpoints_in_icosahedron_l1870_187080

theorem probability_of_edge_endpoints_in_icosahedron :
  let vertices := 12
  let edges := 30
  let connections_per_vertex := 5
  (5 / (vertices - 1)) = (5 / 11) := by
  sorry

end probability_of_edge_endpoints_in_icosahedron_l1870_187080


namespace roots_in_ap_difference_one_l1870_187022

theorem roots_in_ap_difference_one :
  ∀ (r1 r2 r3 : ℝ), 
    64 * r1^3 - 144 * r1^2 + 92 * r1 - 15 = 0 ∧
    64 * r2^3 - 144 * r2^2 + 92 * r2 - 15 = 0 ∧
    64 * r3^3 - 144 * r3^2 + 92 * r3 - 15 = 0 ∧
    (r2 - r1 = r3 - r2) →
    max (max r1 r2) r3 - min (min r1 r2) r3 = 1 := 
by
  intros r1 r2 r3 h
  sorry

end roots_in_ap_difference_one_l1870_187022


namespace cos_alpha_beta_l1870_187006

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x * (sin x) ^ 2 - (1 / 2)

theorem cos_alpha_beta :
  ∀ (α β : ℝ), 
    (0 < α ∧ α < π / 2) →
    (0 < β ∧ β < π / 2) →
    f (α / 2) = sqrt 5 / 5 →
    f (β / 2) = 3 * sqrt 10 / 10 →
    cos (α - β) = sqrt 2 / 2 :=
by
  intros α β hα hβ h1 h2
  sorry

end cos_alpha_beta_l1870_187006


namespace subtracted_amount_l1870_187095

theorem subtracted_amount (A N : ℝ) (h₁ : N = 200) (h₂ : 0.95 * N - A = 178) : A = 12 :=
by
  sorry

end subtracted_amount_l1870_187095


namespace correct_proposition_l1870_187013

theorem correct_proposition (a b : ℝ) (h : |a| < b) : a^2 < b^2 :=
sorry

end correct_proposition_l1870_187013


namespace tens_digit_of_3_pow_2013_l1870_187084

theorem tens_digit_of_3_pow_2013 : (3^2013 % 100 / 10) % 10 = 4 :=
by
  sorry

end tens_digit_of_3_pow_2013_l1870_187084


namespace compute_expression_l1870_187032

-- Definition of the operation "minus the reciprocal of"
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement to prove the given problem
theorem compute_expression :
  ((diamond (diamond 3 4) 5) - (diamond 3 (diamond 4 5))) = -71 / 380 := 
sorry

end compute_expression_l1870_187032


namespace probability_one_head_two_tails_l1870_187017

-- Define an enumeration for Coin with two possible outcomes: heads and tails.
inductive Coin
| heads
| tails

-- Function to count the number of heads in a list of Coin.
def countHeads : List Coin → Nat
| [] => 0
| Coin.heads :: xs => 1 + countHeads xs
| Coin.tails :: xs => countHeads xs

-- Function to calculate the probability of a specific event given the total outcomes.
def probability (specific_events total_outcomes : Nat) : Rat :=
  (specific_events : Rat) / (total_outcomes : Rat)

-- The main theorem
theorem probability_one_head_two_tails : probability 3 8 = (3 / 8 : Rat) :=
sorry

end probability_one_head_two_tails_l1870_187017


namespace at_least_one_not_less_than_two_l1870_187027

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) :=
sorry

end at_least_one_not_less_than_two_l1870_187027


namespace no_savings_if_purchased_together_l1870_187056

def window_price : ℕ := 120

def free_windows (purchased_windows : ℕ) : ℕ :=
  (purchased_windows / 10) * 2

def total_cost (windows_needed : ℕ) : ℕ :=
  (windows_needed - free_windows windows_needed) * window_price

def separate_cost : ℕ :=
  total_cost 9 + total_cost 11 + total_cost 10

def joint_cost : ℕ :=
  total_cost 30

theorem no_savings_if_purchased_together :
  separate_cost = joint_cost :=
by
  -- Proof will be provided here, currently skipped.
  sorry

end no_savings_if_purchased_together_l1870_187056


namespace oblique_prism_volume_l1870_187071

noncomputable def volume_of_oblique_prism 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : ℝ :=
  a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2)

theorem oblique_prism_volume 
  (a b c : ℝ) (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  : volume_of_oblique_prism a b c α β hα hβ = a * b * c / Real.sqrt (1 + (Real.cos α / Real.sin α)^2 + (Real.cos β / Real.sin β)^2) := 
by
  -- Proof will be completed here
  sorry

end oblique_prism_volume_l1870_187071


namespace greatest_possible_y_l1870_187060

theorem greatest_possible_y (x y : ℤ) (h : x * y + 6 * x + 3 * y = 6) : y ≤ 18 :=
sorry

end greatest_possible_y_l1870_187060


namespace circle_intersection_line_l1870_187014

theorem circle_intersection_line (d : ℝ) :
  (∃ (x y : ℝ), (x - 5)^2 + (y + 2)^2 = 49 ∧ (x + 1)^2 + (y - 5)^2 = 25 ∧ x + y = d) ↔ d = 6.5 :=
by
  sorry

end circle_intersection_line_l1870_187014


namespace graph_passes_through_point_l1870_187025

theorem graph_passes_through_point (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : 
    ∃ y : ℝ, y = a^0 + 1 ∧ y = 2 :=
by
  use 2
  simp
  sorry

end graph_passes_through_point_l1870_187025


namespace product_of_five_consecutive_is_divisible_by_sixty_l1870_187020

theorem product_of_five_consecutive_is_divisible_by_sixty (n : ℤ) :
  60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end product_of_five_consecutive_is_divisible_by_sixty_l1870_187020


namespace number_of_terms_in_arithmetic_sequence_l1870_187065

noncomputable def arithmetic_sequence_terms (a d n : ℕ) : Prop :=
  let sum_first_three := 3 * a + 3 * d = 34
  let sum_last_three := 3 * a + 3 * (n - 1) * d = 146
  let sum_all := n * (2 * a + (n - 1) * d) / 2 = 390
  (sum_first_three ∧ sum_last_three ∧ sum_all) → n = 13

theorem number_of_terms_in_arithmetic_sequence (a d n : ℕ) : arithmetic_sequence_terms a d n → n = 13 := 
by
  sorry

end number_of_terms_in_arithmetic_sequence_l1870_187065


namespace subtraction_result_l1870_187024

-- Define the condition as given: x - 46 = 15
def condition (x : ℤ) := x - 46 = 15

-- Define the theorem that gives us the equivalent mathematical statement we want to prove
theorem subtraction_result (x : ℤ) (h : condition x) : x - 29 = 32 :=
by
  -- Here we would include the proof steps, but as per instructions we will use 'sorry' to skip the proof
  sorry

end subtraction_result_l1870_187024


namespace find_m_l1870_187015

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) (h1 : ∀ n, S n = n^2)
  (h2 : S m = (a m + a (m + 1)) / 2)
  (h3 : ∀ n > 1, a n = S n - S (n - 1))
  (h4 : a 1 = 1) :
  m = 2 :=
sorry

end find_m_l1870_187015


namespace compare_f_values_l1870_187079

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

end compare_f_values_l1870_187079


namespace Isabel_reading_pages_l1870_187050

def pages_of_math_homework : ℕ := 2
def problems_per_page : ℕ := 5
def total_problems : ℕ := 30

def math_problems : ℕ := pages_of_math_homework * problems_per_page
def reading_problems : ℕ := total_problems - math_problems

theorem Isabel_reading_pages : (reading_problems / problems_per_page) = 4 :=
by
  sorry

end Isabel_reading_pages_l1870_187050


namespace escalator_length_l1870_187072

theorem escalator_length
  (escalator_speed : ℕ)
  (person_speed : ℕ)
  (time_taken : ℕ)
  (combined_speed : ℕ)
  (condition1 : escalator_speed = 12)
  (condition2 : person_speed = 2)
  (condition3 : time_taken = 14)
  (condition4 : combined_speed = escalator_speed + person_speed)
  (condition5 : combined_speed * time_taken = 196) :
  combined_speed * time_taken = 196 := 
by
  -- the proof would go here
  sorry

end escalator_length_l1870_187072


namespace find_change_l1870_187003

def initial_amount : ℝ := 1.80
def cost_of_candy_bar : ℝ := 0.45
def change : ℝ := 1.35

theorem find_change : initial_amount - cost_of_candy_bar = change :=
by sorry

end find_change_l1870_187003


namespace total_cost_of_puzzles_l1870_187021

-- Definitions for the costs of large and small puzzles
def large_puzzle_cost : ℕ := 15
def small_puzzle_cost : ℕ := 23 - large_puzzle_cost

-- Theorem statement
theorem total_cost_of_puzzles :
  (large_puzzle_cost + 3 * small_puzzle_cost) = 39 :=
by
  -- Placeholder for the proof
  sorry

end total_cost_of_puzzles_l1870_187021


namespace cubes_of_roots_l1870_187066

theorem cubes_of_roots (a b c : ℝ) (h1 : a + b + c = 2) (h2 : ab + ac + bc = 2) (h3 : abc = 3) : 
  a^3 + b^3 + c^3 = 9 :=
by
  sorry

end cubes_of_roots_l1870_187066


namespace correct_number_of_outfits_l1870_187009

-- Define the number of each type of clothing
def num_red_shirts := 4
def num_green_shirts := 4
def num_blue_shirts := 4
def num_pants := 10
def num_red_hats := 6
def num_green_hats := 6
def num_blue_hats := 4

-- Define the total number of outfits that meet the conditions
def total_outfits : ℕ :=
  (num_red_shirts * num_pants * (num_green_hats + num_blue_hats)) +
  (num_green_shirts * num_pants * (num_red_hats + num_blue_hats)) +
  (num_blue_shirts * num_pants * (num_red_hats + num_green_hats))

-- The proof statement asserting that the total number of valid outfits is 1280
theorem correct_number_of_outfits : total_outfits = 1280 := by
  sorry

end correct_number_of_outfits_l1870_187009


namespace could_not_be_diagonal_lengths_l1870_187070

-- Definitions of the diagonal conditions
def diagonal_condition (s : List ℕ) : Prop :=
  match s with
  | [x, y, z] => x^2 + y^2 > z^2 ∧ x^2 + z^2 > y^2 ∧ y^2 + z^2 > x^2
  | _ => false

-- Statement of the problem
theorem could_not_be_diagonal_lengths : 
  ¬ diagonal_condition [5, 6, 8] :=
by 
  sorry

end could_not_be_diagonal_lengths_l1870_187070


namespace cone_slice_ratio_l1870_187054

theorem cone_slice_ratio (h r : ℝ) (hb : h > 0) (hr : r > 0) :
    let V1 := (1/3) * π * (5*r)^2 * (5*h) - (1/3) * π * (4*r)^2 * (4*h)
    let V2 := (1/3) * π * (4*r)^2 * (4*h) - (1/3) * π * (3*r)^2 * (3*h)
    V2 / V1 = 37 / 61 := by {
  sorry
}

end cone_slice_ratio_l1870_187054


namespace classroom_desks_l1870_187000

theorem classroom_desks (N y : ℕ) (h : 16 * y = 21 * N)
  (hN_le: N <= 30 * 16 / 21) (hMultiple: 3 * N % 4 = 0)
  (hy_le: y ≤ 30)
  : y = 21 := by
  sorry

end classroom_desks_l1870_187000


namespace segment_problem_l1870_187057

theorem segment_problem 
  (A C : ℝ) (B D : ℝ) (P Q : ℝ) (x y k : ℝ)
  (hA : A = 0) (hC : C = 0) 
  (hB : B = 6) (hD : D = 9)
  (hx : x = P - A) (hy : y = Q - C) 
  (hxk : x = 3 * k)
  (hxyk : x + y = 12 * k) :
  k = 2 :=
  sorry

end segment_problem_l1870_187057
