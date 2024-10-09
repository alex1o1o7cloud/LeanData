import Mathlib

namespace remainder_n_plus_2023_l1431_143104

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 7 = 3) : (n + 2023) % 7 = 3 :=
by sorry

end remainder_n_plus_2023_l1431_143104


namespace smallest_fraction_numerator_l1431_143182

theorem smallest_fraction_numerator (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : (a * 4) > (b * 3)) : a = 73 :=
  sorry

end smallest_fraction_numerator_l1431_143182


namespace square_area_on_parabola_l1431_143129

theorem square_area_on_parabola (s : ℝ) (h : 0 < s) (hG : (3 + s)^2 - 6 * (3 + s) + 5 = -2 * s) : 
  (2 * s) * (2 * s) = 24 - 8 * Real.sqrt 5 := 
by 
  sorry

end square_area_on_parabola_l1431_143129


namespace Y_tagged_value_l1431_143124

variables (W X Y Z : ℕ)
variables (tag_W : W = 200)
variables (tag_X : X = W / 2)
variables (tag_Z : Z = 400)
variables (total : W + X + Y + Z = 1000)

theorem Y_tagged_value : Y = 300 :=
by sorry

end Y_tagged_value_l1431_143124


namespace find_circle_center_l1431_143112

theorem find_circle_center :
  ∃ (a b : ℝ), a = 1 / 2 ∧ b = 7 / 6 ∧
  (0 - a)^2 + (1 - b)^2 = (1 - a)^2 + (1 - b)^2 ∧
  (1 - a) * 3 = b - 1 :=
by {
  sorry
}

end find_circle_center_l1431_143112


namespace quadratic_intersects_x_axis_at_two_points_l1431_143163

theorem quadratic_intersects_x_axis_at_two_points (k : ℝ) :
  (k < 1 ∧ k ≠ 0) ↔ ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (kx1^2 + 2 * x1 + 1 = 0) ∧ (kx2^2 + 2 * x2 + 1 = 0) := 
by
  sorry

end quadratic_intersects_x_axis_at_two_points_l1431_143163


namespace solve_quadratic_sum_l1431_143189

theorem solve_quadratic_sum (a b : ℕ) (x : ℝ) (h₁ : x^2 + 10 * x = 93)
  (h₂ : x = Real.sqrt a - b) (ha_pos : 0 < a) (hb_pos : 0 < b) : a + b = 123 := by
  sorry

end solve_quadratic_sum_l1431_143189


namespace find_k_l1431_143176

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of vector c depending on k
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- The theorem to be proven
theorem find_k (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) ↔ (k = -10 / 3) :=
by
  sorry

end find_k_l1431_143176


namespace petya_can_restore_numbers_if_and_only_if_odd_l1431_143169

def can_restore_numbers (n : ℕ) : Prop :=
  ∀ (V : Fin n → ℕ) (S : ℕ),
    ∃ f : Fin n → ℕ, 
    (∀ i : Fin n, 
      (V i) = f i ∨ 
      (S = f i)) ↔ (n % 2 = 1)

theorem petya_can_restore_numbers_if_and_only_if_odd (n : ℕ) : can_restore_numbers n ↔ n % 2 = 1 := 
by sorry

end petya_can_restore_numbers_if_and_only_if_odd_l1431_143169


namespace find_other_endpoint_l1431_143146

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ) 
  (h_mid_x : x_m = (x_1 + x_2) / 2)
  (h_mid_y : y_m = (y_1 + y_2) / 2)
  (h_x_m : x_m = 3)
  (h_y_m : y_m = 4)
  (h_x_1 : x_1 = 0)
  (h_y_1 : y_1 = -1) :
  (x_2, y_2) = (6, 9) :=
sorry

end find_other_endpoint_l1431_143146


namespace sequence_sum_S5_l1431_143151

theorem sequence_sum_S5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : S 2 = 4)
  (h2 : ∀ n, a (n + 1) = 2 * S n + 1)
  (h3 : ∀ n, S (n + 1) - S n = a (n + 1)) :
  S 5 = 121 :=
by
  sorry

end sequence_sum_S5_l1431_143151


namespace elizabeth_revenue_per_investment_l1431_143192

theorem elizabeth_revenue_per_investment :
  ∀ (revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth : ℕ),
    revenue_per_investment_banks = 500 →
    total_investments_banks = 8 →
    total_investments_elizabeth = 5 →
    revenue_difference = 500 →
    ((revenue_per_investment_banks * total_investments_banks) + revenue_difference) / total_investments_elizabeth = 900 :=
by
  intros revenue_per_investment_banks revenue_difference total_investments_banks total_investments_elizabeth
  intros h_banks_revenue h_banks_investments h_elizabeth_investments h_revenue_difference
  sorry

end elizabeth_revenue_per_investment_l1431_143192


namespace geometric_seq_sum_l1431_143153

theorem geometric_seq_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1_pos : a 1 > 0)
  (h_a4_7 : a 4 + a 7 = 2)
  (h_a5_6 : a 5 * a 6 = -8) :
  a 1 + a 4 + a 7 + a 10 = -5 := 
sorry

end geometric_seq_sum_l1431_143153


namespace new_ratio_of_partners_to_associates_l1431_143167

theorem new_ratio_of_partners_to_associates
  (partners associates : ℕ)
  (rat_partners_associates : 2 * associates = 63 * partners)
  (partners_count : partners = 18)
  (add_assoc : associates + 45 = 612) :
  (partners:ℚ) / (associates + 45) = 1 / 34 :=
by
  -- Actual proof goes here
  sorry

end new_ratio_of_partners_to_associates_l1431_143167


namespace largest_possible_package_size_l1431_143147

theorem largest_possible_package_size :
  ∃ (p : ℕ), Nat.gcd 60 36 = p ∧ p = 12 :=
by
  use 12
  sorry -- The proof is skipped as per instructions

end largest_possible_package_size_l1431_143147


namespace remainder_of_multiple_l1431_143131

theorem remainder_of_multiple (m k : ℤ) (h1 : m % 5 = 2) (h2 : (2 * k) % 5 = 1) : 
  (k * m) % 5 = 1 := 
sorry

end remainder_of_multiple_l1431_143131


namespace min_paint_steps_l1431_143141

-- Checkered square of size 2021x2021 where all cells initially white.
-- Ivan selects two cells and paints them black.
-- Cells with at least one black neighbor by side are painted black simultaneously each step.

-- Define a function to represent the steps required to paint the square black
noncomputable def min_steps_to_paint_black (n : ℕ) (a b : ℕ × ℕ) : ℕ :=
  sorry -- Placeholder for the actual function definition, as we're focusing on the statement.

-- Define the specific instance of the problem
def square_size := 2021
def initial_cells := ((505, 1010), (1515, 1010))

-- Theorem statement: Proving the minimal number of steps required is 1515
theorem min_paint_steps : min_steps_to_paint_black square_size initial_cells.1 initial_cells.2 = 1515 :=
sorry

end min_paint_steps_l1431_143141


namespace original_price_l1431_143196

theorem original_price (P : ℝ) (h_discount : 0.75 * P = 560): P = 746.68 :=
sorry

end original_price_l1431_143196


namespace intersection_eq_T_l1431_143175

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l1431_143175


namespace basketball_price_l1431_143113

variable (P : ℝ)

def coachA_cost : ℝ := 10 * P
def coachB_baseball_cost : ℝ := 14 * 2.5
def coachB_bat_cost : ℝ := 18
def coachB_total_cost : ℝ := coachB_baseball_cost + coachB_bat_cost
def coachA_excess_cost : ℝ := 237

theorem basketball_price (h : coachA_cost P = coachB_total_cost + coachA_excess_cost) : P = 29 :=
by
  sorry

end basketball_price_l1431_143113


namespace sphere_radius_squared_l1431_143190

theorem sphere_radius_squared (R x y z : ℝ)
  (h1 : 2 * Real.sqrt (R^2 - x^2 - y^2) = 5)
  (h2 : 2 * Real.sqrt (R^2 - x^2 - z^2) = 6)
  (h3 : 2 * Real.sqrt (R^2 - y^2 - z^2) = 7) :
  R^2 = 15 :=
sorry

end sphere_radius_squared_l1431_143190


namespace xiao_ying_performance_l1431_143168

def regular_weight : ℝ := 0.20
def midterm_weight : ℝ := 0.30
def final_weight : ℝ := 0.50

def regular_score : ℝ := 85
def midterm_score : ℝ := 90
def final_score : ℝ := 92

-- Define the function that calculates the weighted average
def semester_performance (rw mw fw rs ms fs : ℝ) : ℝ :=
  rw * rs + mw * ms + fw * fs

-- The theorem that the weighted average of the scores is 90
theorem xiao_ying_performance : semester_performance regular_weight midterm_weight final_weight regular_score midterm_score final_score = 90 := by
  sorry

end xiao_ying_performance_l1431_143168


namespace number_of_men_in_first_group_l1431_143121

-- Condition: Let M be the number of men in the first group
variable (M : ℕ)

-- Condition: M men can complete the work in 20 hours
-- Condition: 15 men can complete the same work in 48 hours
-- We want to prove that if M * 20 = 15 * 48, then M = 36
theorem number_of_men_in_first_group (h : M * 20 = 15 * 48) : M = 36 := by
  sorry

end number_of_men_in_first_group_l1431_143121


namespace mary_keep_warm_hours_l1431_143103

-- Definitions based on the conditions
def sticks_from_chairs (chairs : ℕ) : ℕ := chairs * 6
def sticks_from_tables (tables : ℕ) : ℕ := tables * 9
def sticks_from_stools (stools : ℕ) : ℕ := stools * 2
def sticks_needed_per_hour : ℕ := 5

-- Given counts of furniture
def chairs : ℕ := 18
def tables : ℕ := 6
def stools : ℕ := 4

-- Total number of sticks
def total_sticks : ℕ := (sticks_from_chairs chairs) + (sticks_from_tables tables) + (sticks_from_stools stools)

-- Proving the number of hours Mary can keep warm
theorem mary_keep_warm_hours : total_sticks / sticks_needed_per_hour = 34 := by
  sorry

end mary_keep_warm_hours_l1431_143103


namespace max_sum_of_abc_l1431_143161

theorem max_sum_of_abc (A B C : ℕ) (h1 : A * B * C = 1386) (h2 : A ≠ B) (h3 : A ≠ C) (h4 : B ≠ C) : 
  A + B + C ≤ 88 :=
sorry

end max_sum_of_abc_l1431_143161


namespace divisibility_by_7_l1431_143165

theorem divisibility_by_7 (A X : Nat) (h1 : A < 10) (h2 : X < 10) : (100001 * A + 100010 * X) % 7 = 0 := 
by
  sorry

end divisibility_by_7_l1431_143165


namespace transformation_correct_l1431_143128

variables {x y : ℝ}

theorem transformation_correct (h : x = y) : x - 2 = y - 2 := by
  sorry

end transformation_correct_l1431_143128


namespace lights_on_after_2011_toggles_l1431_143180

-- Definitions for light states and index of lights
inductive Light : Type
| A | B | C | D | E | F | G
deriving DecidableEq

-- Initial light state: function from Light to Bool (true means the light is on)
def initialState : Light → Bool
| Light.A => true
| Light.B => false
| Light.C => true
| Light.D => false
| Light.E => true
| Light.F => false
| Light.G => true

-- Toggling function: toggles the state of a given light
def toggleState (state : Light → Bool) (light : Light) : Light → Bool :=
  fun l => if l = light then ¬ (state l) else state l

-- Toggling sequence: sequentially toggle lights in the given list
def toggleSequence (state : Light → Bool) (seq : List Light) : Light → Bool :=
  seq.foldl toggleState state

-- Toggles the sequence n times
def toggleNTimes (state : Light → Bool) (seq : List Light) (n : Nat) : Light → Bool :=
  let rec aux (state : Light → Bool) (n : Nat) : Light → Bool :=
    if n = 0 then state
    else aux (toggleSequence state seq) (n - 1)
  aux state n

-- Toggling sequence: A, B, C, D, E, F, G
def toggleSeq : List Light := [Light.A, Light.B, Light.C, Light.D, Light.E, Light.F, Light.G]

-- Determine the final state after 2011 toggles
def finalState : Light → Bool := toggleNTimes initialState toggleSeq 2011

-- Proof statement: the state of the lights after 2011 toggles is such that lights A, D, F are on
theorem lights_on_after_2011_toggles :
  finalState Light.A = true ∧
  finalState Light.D = true ∧
  finalState Light.F = true ∧
  finalState Light.B = false ∧
  finalState Light.C = false ∧
  finalState Light.E = false ∧
  finalState Light.G = false :=
by
  sorry

end lights_on_after_2011_toggles_l1431_143180


namespace brianne_savings_in_may_l1431_143118

-- Definitions based on conditions from a)
def initial_savings_jan : ℕ := 20
def multiplier : ℕ := 3
def additional_income : ℕ := 50

-- Savings in successive months
def savings_feb : ℕ := multiplier * initial_savings_jan
def savings_mar : ℕ := multiplier * savings_feb + additional_income
def savings_apr : ℕ := multiplier * savings_mar + additional_income
def savings_may : ℕ := multiplier * savings_apr + additional_income

-- The main theorem to verify
theorem brianne_savings_in_may : savings_may = 2270 :=
sorry

end brianne_savings_in_may_l1431_143118


namespace minimum_value_of_l1431_143117

noncomputable def minimum_value (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem minimum_value_of (x y z : ℝ) (hxyz : x > 0 ∧ y > 0 ∧ z > 0) (h : 1/x + 1/y + 1/z = 9) :
  minimum_value x y z = 1 / 3456 := 
sorry

end minimum_value_of_l1431_143117


namespace isosceles_triangle_vertex_angle_l1431_143115

noncomputable def vertex_angle_of_isosceles (a b : ℝ) : ℝ :=
  if a = b then 40 else 100

theorem isosceles_triangle_vertex_angle (a : ℝ) (interior_angle : ℝ)
  (h_isosceles : a = 40 ∨ a = interior_angle ∧ interior_angle = 40 ∨ interior_angle = 100) :
  vertex_angle_of_isosceles a interior_angle = 40 ∨ vertex_angle_of_isosceles a interior_angle = 100 := 
by
  sorry

end isosceles_triangle_vertex_angle_l1431_143115


namespace percentage_students_camping_trip_l1431_143111

theorem percentage_students_camping_trip 
  (total_students : ℝ)
  (camping_trip_with_more_than_100 : ℝ) 
  (camping_trip_without_more_than_100_ratio : ℝ) :
  camping_trip_with_more_than_100 / (camping_trip_with_more_than_100 / 0.25) = 0.8 :=
by
  sorry

end percentage_students_camping_trip_l1431_143111


namespace solve_equation_l1431_143138

theorem solve_equation :
  ∀ x : ℝ,
  (1 / (x^2 + 12 * x - 9) + 
   1 / (x^2 + 3 * x - 9) + 
   1 / (x^2 - 12 * x - 9) = 0) ↔ 
  (x = 1 ∨ x = -9 ∨ x = 3 ∨ x = -3) := 
by
  sorry

end solve_equation_l1431_143138


namespace find_a_find_b_find_T_l1431_143188

open Real

def S (n : ℕ) : ℝ := 2 * n^2 + n

def a (n : ℕ) : ℝ := if n = 1 then 3 else S n - S (n - 1)

def b (n : ℕ) : ℝ := 2^(n - 1)

def T (n : ℕ) : ℝ := (4 * n - 5) * 2^n + 5

theorem find_a (n : ℕ) (hn : n > 0) : a n = 4 * n - 1 :=
by sorry

theorem find_b (n : ℕ) (hn : n > 0) : b n = 2^(n-1) :=
by sorry

theorem find_T (n : ℕ) (hn : n > 0) (a_def : ∀ n, a n = 4 * n - 1) (b_def : ∀ n, b n = 2^(n-1)) : T n = (4 * n - 5) * 2^n + 5 :=
by sorry

end find_a_find_b_find_T_l1431_143188


namespace trajectory_of_M_l1431_143139

variables {x y : ℝ}

theorem trajectory_of_M (h : y / (x + 2) + y / (x - 2) = 2) (hx : x ≠ 2) (hx' : x ≠ -2) :
  x * y - x^2 + 4 = 0 :=
by sorry

end trajectory_of_M_l1431_143139


namespace fraction_zero_x_value_l1431_143148

theorem fraction_zero_x_value (x : ℝ) (h1 : 2 * x = 0) (h2 : x + 3 ≠ 0) : x = 0 :=
by
  sorry

end fraction_zero_x_value_l1431_143148


namespace smallest_positive_integer_l1431_143166

theorem smallest_positive_integer (m n : ℤ) : ∃ m n : ℤ, 3003 * m + 66666 * n = 3 :=
by
  sorry

end smallest_positive_integer_l1431_143166


namespace integer_solutions_count_l1431_143125

theorem integer_solutions_count :
  (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 15 ∧
    ∀ (pair : ℕ × ℕ), pair ∈ pairs ↔ (∃ x y, pair = (x, y) ∧ (Nat.sqrt x + Nat.sqrt y = 14))) :=
by
  sorry

end integer_solutions_count_l1431_143125


namespace transportation_trucks_l1431_143135

theorem transportation_trucks (boxes : ℕ) (total_weight : ℕ) (box_weight : ℕ) (truck_capacity : ℕ) :
  (total_weight = 10) → (∀ (b : ℕ), b ≤ boxes → box_weight ≤ 1) → (truck_capacity = 3) → 
  ∃ (trucks : ℕ), trucks = 5 :=
by
  sorry

end transportation_trucks_l1431_143135


namespace range_of_m_l1431_143178

def A (x : ℝ) : Prop := 1/2 < x ∧ x < 1

def B (x : ℝ) (m : ℝ) : Prop := x^2 + 2 * x + 1 - m ≤ 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, A x → B x m) → 4 ≤ m := by
  sorry

end range_of_m_l1431_143178


namespace part1_part2_l1431_143119

-- Step 1: Define necessary probabilities
def P_A1 : ℚ := 5 / 6
def P_A2 : ℚ := 2 / 3
def P_B1 : ℚ := 3 / 5
def P_B2 : ℚ := 3 / 4

-- Step 2: Winning event probabilities for both participants
def P_A_wins := P_A1 * P_A2
def P_B_wins := P_B1 * P_B2

-- Step 3: Problem statement: Comparing probabilities
theorem part1 (P_A_wins P_A_wins : ℚ) : P_A_wins > P_B_wins := 
  by sorry

-- Step 4: Complement probabilities for not winning the competition
def P_not_A_wins := 1 - P_A_wins
def P_not_B_wins := 1 - P_B_wins

-- Step 5: Probability at least one wins
def P_at_least_one_wins := 1 - (P_not_A_wins * P_not_B_wins)

-- Step 6: Problem statement: At least one wins
theorem part2 : P_at_least_one_wins = 34 / 45 := 
  by sorry

end part1_part2_l1431_143119


namespace steps_A_l1431_143155

theorem steps_A (t_A t_B : ℝ) (a e t : ℝ) :
  t_A = 3 * t_B →
  t_B = t / 75 →
  a + e * t = 100 →
  75 + e * t = 100 →
  a = 75 :=
by sorry

end steps_A_l1431_143155


namespace initial_points_l1431_143136

theorem initial_points (n : ℕ) (h : 16 * n - 15 = 225) : n = 15 :=
sorry

end initial_points_l1431_143136


namespace fixed_point_through_1_neg2_l1431_143195

noncomputable def fixed_point (a : ℝ) (x : ℝ) : ℝ :=
a^(x - 1) - 3

-- The statement to prove
theorem fixed_point_through_1_neg2 (a : ℝ) (h : a > 0) (h' : a ≠ 1) :
  fixed_point a 1 = -2 :=
by
  unfold fixed_point
  sorry

end fixed_point_through_1_neg2_l1431_143195


namespace num_ordered_pairs_solutions_l1431_143110

theorem num_ordered_pairs_solutions :
  ∃ (n : ℕ), n = 18 ∧
    (∀ (a b : ℝ), (∃ x y : ℤ , a * (x : ℝ) + b * (y : ℝ) = 1 ∧ (x * x + y * y = 50))) :=
sorry

end num_ordered_pairs_solutions_l1431_143110


namespace determine_n_for_11111_base_n_is_perfect_square_l1431_143191

theorem determine_n_for_11111_base_n_is_perfect_square:
  ∃ m : ℤ, m^2 = 3^4 + 3^3 + 3^2 + 3 + 1 :=
by
  sorry

end determine_n_for_11111_base_n_is_perfect_square_l1431_143191


namespace unique_line_through_A_parallel_to_a_l1431_143149

variables {Point Line Plane : Type}
variables {α β : Plane}
variables {a l : Line}
variables {A : Point}

-- Definitions are necessary from conditions in step a)
def parallel_to (a b : Line) : Prop := sorry -- Definition that two lines are parallel
def contains (p : Plane) (x : Point) : Prop := sorry -- Definition that a plane contains a point
def line_parallel_to_plane (a : Line) (p : Plane) : Prop := sorry -- Definition that a line is parallel to a plane

-- Given conditions in the proof problem
variable (a_parallel_α : line_parallel_to_plane a α)
variable (A_in_α : contains α A)

-- Statement to be proven: There is only one line that passes through point A and is parallel to line a, and that line is within plane α.
theorem unique_line_through_A_parallel_to_a : 
  ∃! l : Line, contains α A ∧ parallel_to l a := sorry

end unique_line_through_A_parallel_to_a_l1431_143149


namespace study_time_in_minutes_l1431_143143

theorem study_time_in_minutes :
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  total_hours * 60 = 540 :=
by
  let day1_hours := 2
  let day2_hours := 2 * day1_hours
  let day3_hours := day2_hours - 1
  let total_hours := day1_hours + day2_hours + day3_hours
  sorry

end study_time_in_minutes_l1431_143143


namespace inequality_for_positive_a_b_n_l1431_143127

theorem inequality_for_positive_a_b_n (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry

end inequality_for_positive_a_b_n_l1431_143127


namespace chromosomes_mitosis_late_stage_l1431_143186

/-- A biological cell with 24 chromosomes at the late stage of the second meiotic division. -/
def cell_chromosomes_meiosis_late_stage : ℕ := 24

/-- The number of chromosomes in this organism at the late stage of mitosis is double that at the late stage of the second meiotic division. -/
theorem chromosomes_mitosis_late_stage : cell_chromosomes_meiosis_late_stage * 2 = 48 :=
by
  -- We will add the necessary proof here.
  sorry

end chromosomes_mitosis_late_stage_l1431_143186


namespace avery_egg_cartons_filled_l1431_143137

-- Definitions (conditions identified in step a)
def total_chickens : ℕ := 20
def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12

-- Theorem statement (equivalent to the problem statement)
theorem avery_egg_cartons_filled : (total_chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  -- Proof omitted; sorry used to denote unfinished proof
  sorry

end avery_egg_cartons_filled_l1431_143137


namespace ksyusha_travel_time_l1431_143183

variables (v S : ℝ)

theorem ksyusha_travel_time :
  (2 * S / v) + (S / (2 * v)) = 30 →
  (S / v) + (2 * S / (2 * v)) = 24 :=
by sorry

end ksyusha_travel_time_l1431_143183


namespace athena_total_spent_l1431_143157

-- Define the conditions
def sandwiches_quantity : ℕ := 3
def sandwich_price : ℝ := 3.0
def drinks_quantity : ℕ := 2
def drink_price : ℝ := 2.5

-- Define the calculations
def total_sandwich_cost : ℝ := sandwiches_quantity * sandwich_price
def total_drink_cost : ℝ := drinks_quantity * drink_price

-- Define the total cost
def total_amount_spent : ℝ := total_sandwich_cost + total_drink_cost

-- Prove the total amount spent
theorem athena_total_spent : total_amount_spent = 14 := by
  sorry

end athena_total_spent_l1431_143157


namespace coloring_ways_l1431_143159

-- Definitions of the problem:
def column1 := 1
def column2 := 2
def column3 := 3
def column4 := 4
def column5 := 3
def column6 := 2
def column7 := 1
def total_colors := 3 -- Blue, Yellow, Green

-- Adjacent coloring constraints:
def adjacent_constraints (c1 c2 : ℕ) : Prop := c1 ≠ c2

-- Number of ways to color figure:
theorem coloring_ways : 
  (∃ (n : ℕ), n = 2^5) ∧ 
  n = 32 :=
by 
  sorry

end coloring_ways_l1431_143159


namespace time_for_completion_l1431_143193

noncomputable def efficiency_b : ℕ := 100

noncomputable def efficiency_a := 130

noncomputable def total_work := efficiency_a * 23

noncomputable def combined_efficiency := efficiency_a + efficiency_b

noncomputable def time_taken := total_work / combined_efficiency

theorem time_for_completion (h1 : efficiency_a = 130)
                           (h2 : efficiency_b = 100)
                           (h3 : total_work = 2990)
                           (h4 : combined_efficiency = 230) :
  time_taken = 13 := by
  sorry

end time_for_completion_l1431_143193


namespace Joyce_final_apples_l1431_143199

def initial_apples : ℝ := 350.5
def apples_given_to_larry : ℝ := 218.7
def percentage_given_to_neighbors : ℝ := 0.375
def final_apples : ℝ := 82.375

theorem Joyce_final_apples :
  (initial_apples - apples_given_to_larry - percentage_given_to_neighbors * (initial_apples - apples_given_to_larry)) = final_apples :=
by
  sorry

end Joyce_final_apples_l1431_143199


namespace ratio_PA_AB_l1431_143156

theorem ratio_PA_AB (A B C P : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup P]
  (h1 : ∃ AC CB : ℕ, AC = 2 * CB)
  (h2 : ∃ PA AB : ℕ, PA = 4 * (AB / 5)) :
  PA / AB = 4 / 5 := sorry

end ratio_PA_AB_l1431_143156


namespace total_scarves_l1431_143185

def total_yarns_red : ℕ := 2
def total_yarns_blue : ℕ := 6
def total_yarns_yellow : ℕ := 4
def scarves_per_yarn : ℕ := 3

theorem total_scarves : 
  (total_yarns_red * scarves_per_yarn) + 
  (total_yarns_blue * scarves_per_yarn) + 
  (total_yarns_yellow * scarves_per_yarn) = 36 := 
by
  sorry

end total_scarves_l1431_143185


namespace find_m_l1431_143162

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2 * x - 5 * y + 20 = 0
def l2 (m x y : ℝ) : Prop := m * x + 2 * y - 10 = 0

-- Define the condition of perpendicularity
def lines_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- Proving the value of m given the conditions
theorem find_m (m : ℝ) :
  (∃ x y : ℝ, l1 x y) → (∃ x y : ℝ, l2 m x y) → lines_perpendicular 2 (-5 : ℝ) m 2 → m = 5 :=
sorry

end find_m_l1431_143162


namespace exponential_inequality_l1431_143122

variable (a b : ℝ)

theorem exponential_inequality (h : -1 < a ∧ a < b ∧ b < 1) : Real.exp a < Real.exp b :=
by
  sorry

end exponential_inequality_l1431_143122


namespace series_sum_eq_50_l1431_143172

noncomputable def series_sum (x : ℝ) : ℝ :=
  2 + 6 * x + 10 * x^2 + 14 * x^3 -- This represents the series

theorem series_sum_eq_50 : 
  ∃ x : ℝ, series_sum x = 50 ∧ x = 0.59 :=
by
  sorry

end series_sum_eq_50_l1431_143172


namespace trig_identity_l1431_143194

open Real

theorem trig_identity 
  (θ : ℝ)
  (h : tan (π / 4 + θ) = 3) : 
  sin (2 * θ) - 2 * cos θ ^ 2 = -3 / 4 :=
by
  sorry

end trig_identity_l1431_143194


namespace complex_number_evaluation_l1431_143184

noncomputable def i := Complex.I

theorem complex_number_evaluation :
  (1 - i) * (i * i) / (1 + 2 * i) = (1/5 : ℂ) + (3/5 : ℂ) * i :=
by
  sorry

end complex_number_evaluation_l1431_143184


namespace roots_of_polynomial_l1431_143158

theorem roots_of_polynomial : ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ x = 2 ∨ x = 3 ∨ x = -2 :=
by sorry

end roots_of_polynomial_l1431_143158


namespace Malcom_cards_after_giving_away_half_l1431_143133

def Brandon_cards : ℕ := 20
def Malcom_initial_cards : ℕ := Brandon_cards + 8
def Malcom_remaining_cards : ℕ := Malcom_initial_cards - (Malcom_initial_cards / 2)

theorem Malcom_cards_after_giving_away_half :
  Malcom_remaining_cards = 14 :=
by
  sorry

end Malcom_cards_after_giving_away_half_l1431_143133


namespace find_c_for_circle_radius_5_l1431_143107

theorem find_c_for_circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 4 * x + y^2 + 8 * y + c = 0 
    → x^2 + 4 * x + y^2 + 8 * y = 5^2 - 25) 
  → c = -5 :=
by
  sorry

end find_c_for_circle_radius_5_l1431_143107


namespace correct_calculation_l1431_143108

theorem correct_calculation (x y : ℝ) : -x^2 * y + 3 * x^2 * y = 2 * x^2 * y :=
by
  sorry

end correct_calculation_l1431_143108


namespace sum_of_digits_8_pow_2003_l1431_143132

noncomputable def units_digit (n : ℕ) : ℕ :=
n % 10

noncomputable def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

noncomputable def sum_of_tens_and_units_digits (n : ℕ) : ℕ :=
units_digit n + tens_digit n

theorem sum_of_digits_8_pow_2003 :
  sum_of_tens_and_units_digits (8 ^ 2003) = 2 :=
by
  sorry

end sum_of_digits_8_pow_2003_l1431_143132


namespace demand_change_for_revenue_l1431_143130

theorem demand_change_for_revenue (P D D' : ℝ)
  (h1 : D' = (1.10 * D) / 1.20)
  (h2 : P' = 1.20 * P)
  (h3 : P * D = P' * D') :
  (D' - D) / D * 100 = -8.33 := by
sorry

end demand_change_for_revenue_l1431_143130


namespace running_to_weightlifting_ratio_l1431_143109

-- Definitions for given conditions in the problem
def total_practice_time : ℕ := 120 -- 120 minutes
def shooting_time : ℕ := total_practice_time / 2
def weightlifting_time : ℕ := 20
def running_time : ℕ := shooting_time - weightlifting_time

-- The goal is to prove that the ratio of running time to weightlifting time is 2:1
theorem running_to_weightlifting_ratio : running_time / weightlifting_time = 2 :=
by
  /- use the given problem conditions directly -/
  exact sorry

end running_to_weightlifting_ratio_l1431_143109


namespace total_length_of_wire_l1431_143100

-- Definitions based on conditions
def num_squares : ℕ := 15
def length_of_grid : ℕ := 10
def width_of_grid : ℕ := 5
def height_of_grid : ℕ := 3
def side_length : ℕ := length_of_grid / width_of_grid -- 2 units
def num_horizontal_wires : ℕ := height_of_grid + 1    -- 4 wires
def num_vertical_wires : ℕ := width_of_grid + 1      -- 6 wires
def total_length_horizontal_wires : ℕ := num_horizontal_wires * length_of_grid -- 40 units
def total_length_vertical_wires : ℕ := num_vertical_wires * (height_of_grid * side_length) -- 36 units

-- The theorem to prove the total length of wire needed
theorem total_length_of_wire : total_length_horizontal_wires + total_length_vertical_wires = 76 :=
by
  sorry

end total_length_of_wire_l1431_143100


namespace smallest_number_remainder_problem_l1431_143181

theorem smallest_number_remainder_problem :
  ∃ N : ℕ, (N % 13 = 2) ∧ (N % 15 = 4) ∧ (∀ n : ℕ, (n % 13 = 2 ∧ n % 15 = 4) → n ≥ N) :=
sorry

end smallest_number_remainder_problem_l1431_143181


namespace tournament_rounds_l1431_143144

/-- 
Given a tournament where each participant plays several games with every other participant
and a total of 224 games were played, prove that the number of rounds in the competition is 8.
-/
theorem tournament_rounds (x y : ℕ) (hx : x > 1) (hy : y > 0) (h : x * (x - 1) * y = 448) : y = 8 :=
sorry

end tournament_rounds_l1431_143144


namespace crit_value_expr_l1431_143106

theorem crit_value_expr : 
  ∃ x : ℝ, -4 < x ∧ x < 1 ∧ (x^2 - 2*x + 2) / (2*x - 2) = -1 :=
sorry

end crit_value_expr_l1431_143106


namespace length_of_field_l1431_143126

variable (w : ℕ) (l : ℕ)

def length_field_is_double_width (w l : ℕ) : Prop :=
  l = 2 * w

def pond_area_equals_one_eighth_field_area (w l : ℕ) : Prop :=
  36 = 1 / 8 * (l * w)

theorem length_of_field (w l : ℕ) (h1 : length_field_is_double_width w l) (h2 : pond_area_equals_one_eighth_field_area w l) : l = 24 := 
by
  sorry

end length_of_field_l1431_143126


namespace find_f_5_l1431_143152

-- Definitions from conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x ^ 3 - b * x + 2

-- Stating the theorem
theorem find_f_5 (a b : ℝ) (h : f (-5) a b = 17) : f 5 a b = -13 :=
by
  sorry

end find_f_5_l1431_143152


namespace evaluate_product_l1431_143177

theorem evaluate_product (m : ℕ) (h : m = 3) : (m - 2) * (m - 1) * m * (m + 1) * (m + 2) * (m + 3) = 720 :=
by {
  sorry
}

end evaluate_product_l1431_143177


namespace circles_are_separate_l1431_143140

def circle_center (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circles_are_separate :
  circle_center 0 0 1 x y → 
  circle_center 3 (-4) 3 x' y' →
  dist (0, 0) (3, -4) > 1 + 3 :=
by
  intro h₁ h₂
  sorry

end circles_are_separate_l1431_143140


namespace correct_statement_l1431_143150

variables {α β γ : ℝ → ℝ → ℝ → Prop} -- planes
variables {a b c : ℝ → ℝ → ℝ → Prop} -- lines

def is_parallel (P Q : ℝ → ℝ → ℝ → Prop) : Prop :=
∀ x : ℝ, ∀ y : ℝ, ∀ z : ℝ, (P x y z → Q x y z) ∧ (Q x y z → P x y z)

def is_perpendicular (L : ℝ → ℝ → ℝ → Prop) (P : ℝ → ℝ → ℝ → Prop) : Prop :=
∀ x : ℝ, ∀ y : ℝ, ∀ z : ℝ, L x y z ↔ ¬ P x y z 

theorem correct_statement : 
  (is_perpendicular a α) → 
  (is_parallel b β) → 
  (is_parallel α β) → 
  (is_perpendicular a b) :=
by
  sorry

end correct_statement_l1431_143150


namespace least_xy_value_l1431_143145

theorem least_xy_value (x y : ℕ) (hposx : x > 0) (hposy : y > 0) (h : 1/x + 1/(3*y) = 1/8) :
  xy = 96 :=
by
  sorry

end least_xy_value_l1431_143145


namespace gcd_of_three_l1431_143123

theorem gcd_of_three (a b c : ℕ) (h₁ : a = 9242) (h₂ : b = 13863) (h₃ : c = 34657) :
  Nat.gcd (Nat.gcd a b) c = 1 :=
by
  sorry

end gcd_of_three_l1431_143123


namespace min_value_expression_l1431_143102

open Real

theorem min_value_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 60 := 
  sorry

end min_value_expression_l1431_143102


namespace find_nonzero_c_l1431_143164

def quadratic_has_unique_solution (c b : ℝ) : Prop :=
  (b^4 + (1 - 4 * c) * b^2 + 1 = 0) ∧ (1 - 4 * c)^2 - 4 = 0

theorem find_nonzero_c (c : ℝ) (b : ℝ) (h_nonzero : c ≠ 0) (h_unique_sol : quadratic_has_unique_solution c b) : 
  c = 3 / 4 := 
sorry

end find_nonzero_c_l1431_143164


namespace highest_possible_average_l1431_143197

theorem highest_possible_average (average_score : ℕ) (total_tests : ℕ) (lowest_score : ℕ) 
  (total_marks : ℕ := total_tests * average_score)
  (new_total_tests : ℕ := total_tests - 1)
  (resulting_average : ℚ := (total_marks - lowest_score) / new_total_tests) :
  average_score = 68 ∧ total_tests = 9 ∧ lowest_score = 0 → resulting_average = 76.5 := sorry

end highest_possible_average_l1431_143197


namespace sonya_falls_6_l1431_143154

def number_of_falls_steven : ℕ := 3
def number_of_falls_stephanie : ℕ := number_of_falls_steven + 13
def number_of_falls_sonya : ℕ := (number_of_falls_stephanie / 2) - 2

theorem sonya_falls_6 : number_of_falls_sonya = 6 := 
by
  -- The actual proof is to be filled in here
  sorry

end sonya_falls_6_l1431_143154


namespace selectedParticipants_correct_l1431_143114

-- Define the random number table portion used in the problem
def randomNumTable := [
  [12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57, 12, 10, 14, 21, 88, 26, 49, 81, 76]
]

-- Define the conditions
def totalStudents := 247
def selectedStudentsCount := 4
def startingIndexRow := 4
def startingIndexCol := 9
def startingNumber := randomNumTable[0][8]

-- Define the expected selected participants' numbers
def expectedParticipants := [050, 121, 014, 218]

-- The Lean statement that needs to be proved
theorem selectedParticipants_correct : expectedParticipants = [050, 121, 014, 218] := by
  sorry

end selectedParticipants_correct_l1431_143114


namespace length_of_room_l1431_143171

theorem length_of_room 
  (width : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) 
  (h_width : width = 3.75) 
  (h_total_cost : total_cost = 16500) 
  (h_rate_per_sq_meter : rate_per_sq_meter = 800) : 
  ∃ length : ℝ, length = 5.5 :=
by
  sorry

end length_of_room_l1431_143171


namespace compute_expression_l1431_143179

theorem compute_expression : 12 + 5 * (4 - 9)^2 - 3 = 134 := by
  sorry

end compute_expression_l1431_143179


namespace point_on_line_and_equidistant_l1431_143105

theorem point_on_line_and_equidistant {x y : ℝ} :
  (4 * x + 3 * y = 12) ∧ (x = y) ∧ (x ≥ 0) ∧ (y ≥ 0) ↔ x = 12 / 7 ∧ y = 12 / 7 :=
by
  sorry

end point_on_line_and_equidistant_l1431_143105


namespace n_is_perfect_square_l1431_143142

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k ^ 2

theorem n_is_perfect_square (a b c d : ℤ) (h : a + b + c + d = 0) : 
  is_perfect_square ((ab - cd) * (bc - ad) * (ca - bd)) := 
  sorry

end n_is_perfect_square_l1431_143142


namespace car_travel_first_hour_l1431_143174

-- Define the conditions as variables and the ultimate equality to be proved
theorem car_travel_first_hour (x : ℕ) (h : 12 * x + 132 = 612) : x = 40 :=
by
  -- Proof will be completed here
  sorry

end car_travel_first_hour_l1431_143174


namespace weight_loss_percentage_l1431_143120

theorem weight_loss_percentage (W : ℝ) (hW : W > 0) : 
  let new_weight := 0.89 * W
  let final_weight_with_clothes := new_weight * 1.02
  (W - final_weight_with_clothes) / W * 100 = 9.22 := by
  sorry

end weight_loss_percentage_l1431_143120


namespace stratified_sampling_correct_l1431_143116

-- Define the conditions
def num_freshmen : ℕ := 900
def num_sophomores : ℕ := 1200
def num_seniors : ℕ := 600
def total_sample_size : ℕ := 135
def total_students := num_freshmen + num_sophomores + num_seniors

-- Proportions
def proportion_freshmen := (num_freshmen : ℚ) / total_students
def proportion_sophomores := (num_sophomores : ℚ) / total_students
def proportion_seniors := (num_seniors : ℚ) / total_students

-- Expected samples count
def expected_freshmen_samples := (total_sample_size : ℚ) * proportion_freshmen
def expected_sophomores_samples := (total_sample_size : ℚ) * proportion_sophomores
def expected_seniors_samples := (total_sample_size : ℚ) * proportion_seniors

-- Statement to be proven
theorem stratified_sampling_correct :
  expected_freshmen_samples = (45 : ℚ) ∧
  expected_sophomores_samples = (60 : ℚ) ∧
  expected_seniors_samples = (30 : ℚ) := by
  -- Provide the necessary proof or calculation
  sorry

end stratified_sampling_correct_l1431_143116


namespace final_total_cost_l1431_143187

def initial_spiral_cost : ℝ := 15
def initial_planner_cost : ℝ := 10
def spiral_discount_rate : ℝ := 0.20
def planner_discount_rate : ℝ := 0.15
def num_spirals : ℝ := 4
def num_planners : ℝ := 8
def sales_tax_rate : ℝ := 0.07

theorem final_total_cost :
  let discounted_spiral_cost := initial_spiral_cost * (1 - spiral_discount_rate)
  let discounted_planner_cost := initial_planner_cost * (1 - planner_discount_rate)
  let total_before_tax := num_spirals * discounted_spiral_cost + num_planners * discounted_planner_cost
  let total_tax := total_before_tax * sales_tax_rate
  let total_cost := total_before_tax + total_tax
  total_cost = 124.12 :=
by
  sorry

end final_total_cost_l1431_143187


namespace ratio_of_screws_l1431_143160

def initial_screws : Nat := 8
def total_required_screws : Nat := 4 * 6
def screws_to_buy : Nat := total_required_screws - initial_screws

theorem ratio_of_screws :
  (screws_to_buy : ℚ) / initial_screws = 2 :=
by
  simp [initial_screws, total_required_screws, screws_to_buy]
  sorry

end ratio_of_screws_l1431_143160


namespace farmer_potatoes_initial_l1431_143134

theorem farmer_potatoes_initial (P : ℕ) (h1 : 175 + P - 172 = 80) : P = 77 :=
by {
  sorry
}

end farmer_potatoes_initial_l1431_143134


namespace find_integer_mul_a_l1431_143101

noncomputable def integer_mul_a (a b : ℤ) (n : ℤ) : Prop :=
  n * a * (-8 * b) + a * b = 89 ∧ n < 0 ∧ n * a < 0 ∧ -8 * b < 0

theorem find_integer_mul_a (a b : ℤ) (n : ℤ) (h : integer_mul_a a b n) : n = -11 :=
  sorry

end find_integer_mul_a_l1431_143101


namespace find_prime_and_int_solutions_l1431_143198

-- Define the conditions
def is_solution (p x : ℕ) : Prop :=
  x^(p-1) ∣ (p-1)^x + 1

-- Define the statement to be proven
theorem find_prime_and_int_solutions :
  ∀ p x : ℕ, Prime p → (1 ≤ x ∧ x ≤ 2 * p) →
  (is_solution p x ↔ 
    (p = 2 ∧ (x = 1 ∨ x = 2)) ∨ 
    (p = 3 ∧ (x = 1 ∨ x = 3)) ∨
    (x = 1))
:=
by sorry

end find_prime_and_int_solutions_l1431_143198


namespace problem1_correctness_problem2_correctness_l1431_143173

noncomputable def problem1_solution_1 (x : ℝ) : Prop := x = Real.sqrt 5 - 1
noncomputable def problem1_solution_2 (x : ℝ) : Prop := x = -Real.sqrt 5 - 1
noncomputable def problem2_solution_1 (x : ℝ) : Prop := x = 5
noncomputable def problem2_solution_2 (x : ℝ) : Prop := x = -1 / 3

theorem problem1_correctness (x : ℝ) :
  (x^2 + 2*x - 4 = 0) → (problem1_solution_1 x ∨ problem1_solution_2 x) :=
by sorry

theorem problem2_correctness (x : ℝ) :
  (3 * x * (x - 5) = 5 - x) → (problem2_solution_1 x ∨ problem2_solution_2 x) :=
by sorry

end problem1_correctness_problem2_correctness_l1431_143173


namespace prob_one_side_of_tri_in_decagon_is_half_l1431_143170

noncomputable def probability_one_side_of_tri_in_decagon : ℚ :=
  let num_vertices := 10
  let total_triangles := Nat.choose num_vertices 3
  let favorable_triangles := 10 * 6
  favorable_triangles / total_triangles

theorem prob_one_side_of_tri_in_decagon_is_half :
  probability_one_side_of_tri_in_decagon = 1 / 2 := by
  sorry

end prob_one_side_of_tri_in_decagon_is_half_l1431_143170
