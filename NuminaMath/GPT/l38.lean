import Mathlib

namespace difference_between_place_and_face_value_l38_3867

def numeral : Nat := 856973

def digit_of_interest : Nat := 7

def place_value : Nat := 7 * 10

def face_value : Nat := 7

theorem difference_between_place_and_face_value : place_value - face_value = 63 :=
by
  sorry

end difference_between_place_and_face_value_l38_3867


namespace correct_propositions_count_l38_3806

theorem correct_propositions_count (x y : ℝ) :
  (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0) ∧ -- original proposition
  (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0) ∧ -- converse proposition
  (¬(x ≠ 0 ∨ y ≠ 0) ∨ x^2 + y^2 = 0) ∧ -- negation proposition
  (¬(x^2 + y^2 = 0) ∨ x ≠ 0 ∨ y ≠ 0) -- inverse proposition
  := by
  sorry

end correct_propositions_count_l38_3806


namespace complement_intersection_complement_l38_3844

-- Define the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the statement of the proof problem
theorem complement_intersection_complement:
  (U \ (A ∩ B)) = {1, 4, 6} := by
  sorry

end complement_intersection_complement_l38_3844


namespace Jhon_payment_per_day_l38_3886

theorem Jhon_payment_per_day
  (total_days : ℕ)
  (present_days : ℕ)
  (absent_pay : ℝ)
  (total_pay : ℝ)
  (Jhon_present_days : total_days = 60)
  (Jhon_presence : present_days = 35)
  (Jhon_absent_payment : absent_pay = 3.0)
  (Jhon_total_payment : total_pay = 170) :
  ∃ (P : ℝ), 
    P = 2.71 ∧ 
    total_pay = (present_days * P + (total_days - present_days) * absent_pay) := 
sorry

end Jhon_payment_per_day_l38_3886


namespace maria_cartons_needed_l38_3894

theorem maria_cartons_needed : 
  ∀ (total_needed strawberries blueberries raspberries blackberries : ℕ), 
  total_needed = 36 →
  strawberries = 4 →
  blueberries = 8 →
  raspberries = 3 →
  blackberries = 5 →
  (total_needed - (strawberries + blueberries + raspberries + blackberries) = 16) :=
by
  intros total_needed strawberries blueberries raspberries blackberries ht hs hb hr hb
  -- ... the proof would go here
  sorry

end maria_cartons_needed_l38_3894


namespace max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l38_3808

noncomputable def y (x : ℝ) : ℝ := 3 * x + 4 / x
def max_value (x : ℝ) := y x ≤ -4 * Real.sqrt 3

theorem max_y_value_of_3x_plus_4_div_x (h : x < 0) : max_value x :=
sorry

theorem corresponds_value_of_x (x : ℝ) (h : x = -2 * Real.sqrt 3 / 3) : y x = -4 * Real.sqrt 3 :=
sorry

end max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l38_3808


namespace simplify_expression_l38_3895

theorem simplify_expression : 4 * Real.sqrt (1 / 2) + 3 * Real.sqrt (1 / 3) - Real.sqrt 8 = Real.sqrt 3 := 
by 
  sorry

end simplify_expression_l38_3895


namespace part1_1_part1_2_part1_3_part2_l38_3814

def operation (a b c : ℝ) : Prop := a^c = b

theorem part1_1 : operation 3 81 4 :=
by sorry

theorem part1_2 : operation 4 1 0 :=
by sorry

theorem part1_3 : operation 2 (1 / 4) (-2) :=
by sorry

theorem part2 (x y z : ℝ) (h1 : operation 3 7 x) (h2 : operation 3 8 y) (h3 : operation 3 56 z) : x + y = z :=
by sorry

end part1_1_part1_2_part1_3_part2_l38_3814


namespace age_difference_l38_3866

-- Denote the ages of A, B, and C as a, b, and c respectively.
variables (a b c : ℕ)

-- The given condition
def condition : Prop := a + b = b + c + 12

-- Prove that C is 12 years younger than A.
theorem age_difference (h : condition a b c) : c = a - 12 :=
by {
  -- skip the actual proof here, as instructed
  sorry
}

end age_difference_l38_3866


namespace solve_problem_l38_3899

open Real

noncomputable def problem (x : ℝ) : Prop :=
  (cos (2 * x / 5) - cos (2 * π / 15)) ^ 2 + (sin (2 * x / 3) - sin (4 * π / 9)) ^ 2 = 0

theorem solve_problem : ∀ t : ℤ, problem ((29 * π / 3) + 15 * π * t) :=
by
  intro t
  sorry

end solve_problem_l38_3899


namespace min_matches_to_win_champion_min_total_matches_if_wins_11_l38_3896

-- Define the conditions and problem in Lean 4
def teams := ["A", "B", "C"]
def players_per_team : ℕ := 9
def initial_matches : ℕ := 0

-- The minimum number of matches the champion team must win
theorem min_matches_to_win_champion (H : ∀ t ∈ teams, t ≠ "Champion" → players_per_team = 0) :
  initial_matches + 19 = 19 :=
by
  sorry

-- The minimum total number of matches if the champion team wins 11 matches
theorem min_total_matches_if_wins_11 (wins_by_champion : ℕ := 11) (H : wins_by_champion = 11) :
  initial_matches + wins_by_champion + (players_per_team * 2 - wins_by_champion) + 4 = 24 :=
by
  sorry

end min_matches_to_win_champion_min_total_matches_if_wins_11_l38_3896


namespace seating_arrangement_correct_l38_3881

noncomputable def seatingArrangements (committee : Fin 10) : Nat :=
  Nat.factorial 9

theorem seating_arrangement_correct :
  seatingArrangements committee = 362880 :=
by sorry

end seating_arrangement_correct_l38_3881


namespace days_between_dates_l38_3889

-- Define the starting and ending dates
def start_date : Nat := 1990 * 365 + (19 + 2 * 31 + 28) -- March 19, 1990 (accounting for leap years before the start date)
def end_date : Nat   := 1996 * 365 + (23 + 2 * 31 + 29 + 366 * 2 + 365 * 3) -- March 23, 1996 (accounting for leap years)

-- Define the number of leap years between the dates
def leap_years : Nat := 2 -- 1992 and 1996

-- Total number of days
def total_days : Nat := (end_date - start_date + 1)

theorem days_between_dates : total_days = 2197 :=
by
  sorry

end days_between_dates_l38_3889


namespace triangle_AC_5_sqrt_3_l38_3868

theorem triangle_AC_5_sqrt_3 
  (A B C : ℝ)
  (BC AC : ℝ)
  (h1 : 2 * Real.sin (A - B) + Real.cos (B + C) = 2)
  (h2 : BC = 5) :
  AC = 5 * Real.sqrt 3 :=
  sorry

end triangle_AC_5_sqrt_3_l38_3868


namespace width_of_each_glass_pane_l38_3882

noncomputable def width_of_pane (num_panes : ℕ) (total_area : ℝ) (length_of_pane : ℝ) : ℝ :=
  total_area / num_panes / length_of_pane

theorem width_of_each_glass_pane :
  width_of_pane 8 768 12 = 8 := by
  sorry

end width_of_each_glass_pane_l38_3882


namespace correct_random_error_causes_l38_3834

-- Definitions based on conditions
def is_random_error_cause (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3

-- Theorem: Valid causes of random errors are options (1), (2), and (3)
theorem correct_random_error_causes :
  (is_random_error_cause 1) ∧ (is_random_error_cause 2) ∧ (is_random_error_cause 3) :=
by
  sorry

end correct_random_error_causes_l38_3834


namespace cos_of_complementary_angle_l38_3878

theorem cos_of_complementary_angle (Y Z : ℝ) (h : Y + Z = π / 2) 
  (sin_Y : Real.sin Y = 3 / 5) : Real.cos Z = 3 / 5 := 
  sorry

end cos_of_complementary_angle_l38_3878


namespace complement_union_l38_3827

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {1, 4}

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {2, 4}) (hB : B = {1, 4}) :
  (U \ (A ∪ B)) = {3} :=
by
  simp [hU, hA, hB]
  sorry

end complement_union_l38_3827


namespace symmetric_point_origin_l38_3802

theorem symmetric_point_origin (x y : ℤ) (h : x = -2 ∧ y = 3) :
    (-x, -y) = (2, -3) :=
by
  cases h with
  | intro hx hy =>
  simp only [hx, hy]
  sorry

end symmetric_point_origin_l38_3802


namespace greg_needs_additional_amount_l38_3861

def total_cost : ℤ := 90
def saved_amount : ℤ := 57
def additional_amount_needed : ℤ := total_cost - saved_amount

theorem greg_needs_additional_amount :
  additional_amount_needed = 33 :=
by
  sorry

end greg_needs_additional_amount_l38_3861


namespace semi_minor_axis_l38_3807

theorem semi_minor_axis (a c : ℝ) (h_a : a = 5) (h_c : c = 2) : 
  ∃ b : ℝ, b = Real.sqrt (a^2 - c^2) ∧ b = Real.sqrt 21 :=
by
  use Real.sqrt 21
  sorry

end semi_minor_axis_l38_3807


namespace new_pressure_of_nitrogen_gas_l38_3873

variable (p1 p2 v1 v2 k : ℝ)

theorem new_pressure_of_nitrogen_gas :
  (∀ p v, p * v = k) ∧ (p1 = 8) ∧ (v1 = 3) ∧ (p1 * v1 = k) ∧ (v2 = 7.5) →
  p2 = 3.2 :=
by
  intro h
  sorry

end new_pressure_of_nitrogen_gas_l38_3873


namespace solve1_solve2_solve3_solve4_l38_3835

noncomputable section

-- Problem 1
theorem solve1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := sorry

-- Problem 2
theorem solve2 (x : ℝ) : (x + 1)^2 - 144 = 0 ↔ x = 11 ∨ x = -13 := sorry

-- Problem 3
theorem solve3 (x : ℝ) : 3 * (x - 2)^2 = x * (x - 2) ↔ x = 2 ∨ x = 3 := sorry

-- Problem 4
theorem solve4 (x : ℝ) : x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 29) / 2 ∨ x = (-5 - Real.sqrt 29) / 2 := sorry

end solve1_solve2_solve3_solve4_l38_3835


namespace pens_at_end_l38_3821

-- Define the main variable
variable (x : ℝ)

-- Define the conditions as functions
def initial_pens (x : ℝ) := x
def mike_gives (x : ℝ) := 0.5 * x
def after_mike (x : ℝ) := x + (mike_gives x)
def after_cindy (x : ℝ) := 2 * (after_mike x)
def give_sharon (x : ℝ) := 0.25 * (after_cindy x)

-- Define the final number of pens
def final_pens (x : ℝ) := (after_cindy x) - (give_sharon x)

-- The theorem statement
theorem pens_at_end (x : ℝ) : final_pens x = 2.25 * x :=
by sorry

end pens_at_end_l38_3821


namespace setB_is_correct_l38_3892

def setA : Set ℤ := {-1, 0, 1, 2}
def f (x : ℤ) : ℤ := x^2 - 2*x
def setB : Set ℤ := {y | ∃ x ∈ setA, f x = y}

theorem setB_is_correct : setB = {-1, 0, 3} := by
  sorry

end setB_is_correct_l38_3892


namespace intersection_and_perpendicular_line_l38_3865

theorem intersection_and_perpendicular_line :
  ∃ (x y : ℝ), (3 * x + y - 1 = 0) ∧ (x + 2 * y - 7 = 0) ∧ (2 * x - y + 6 = 0) :=
by
  sorry

end intersection_and_perpendicular_line_l38_3865


namespace inequality_solution_I_inequality_solution_II_l38_3880

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| - |x + 1|

theorem inequality_solution_I (x : ℝ) : f x 1 > 2 ↔ x < -2 / 3 ∨ x > 4 :=
sorry 

noncomputable def g (x a : ℝ) : ℝ := f x a + |x + 1| + x

theorem inequality_solution_II (a : ℝ) : (∀ x, g x a > a ^ 2 - 1 / 2) ↔ (-1 / 2 < a ∧ a < 1) :=
sorry

end inequality_solution_I_inequality_solution_II_l38_3880


namespace penguin_seafood_protein_l38_3869

theorem penguin_seafood_protein
  (digest : ℝ) -- representing 30% 
  (digested : ℝ) -- representing 9 grams 
  (h : digest = 0.30) 
  (h1 : digested = 9) :
  ∃ x : ℝ, digested = digest * x ∧ x = 30 :=
by
  sorry

end penguin_seafood_protein_l38_3869


namespace multiples_of_6_and_8_l38_3830

open Nat

theorem multiples_of_6_and_8 (n m k : ℕ) (h₁ : n = 33) (h₂ : m = 25) (h₃ : k = 8) :
  (n - k) + (m - k) = 42 :=
by
  sorry

end multiples_of_6_and_8_l38_3830


namespace three_digit_number_property_l38_3898

theorem three_digit_number_property :
  (∃ a b c : ℕ, 100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999 ∧ 100 * a + 10 * b + c = (a + b + c)^3) ↔
  (∃ a b c : ℕ, a = 5 ∧ b = 1 ∧ c = 2 ∧ 100 * a + 10 * b + c = 512) := sorry

end three_digit_number_property_l38_3898


namespace total_water_capacity_l38_3809

-- Define the given conditions as constants
def numTrucks : ℕ := 5
def tanksPerTruck : ℕ := 4
def capacityPerTank : ℕ := 200

-- Define the claim as a theorem
theorem total_water_capacity :
  numTrucks * (tanksPerTruck * capacityPerTank) = 4000 :=
by
  sorry

end total_water_capacity_l38_3809


namespace increase_average_by_runs_l38_3811

theorem increase_average_by_runs :
  let total_runs_10_matches : ℕ := 10 * 32
  let runs_scored_next_match : ℕ := 87
  let total_runs_11_matches : ℕ := total_runs_10_matches + runs_scored_next_match
  let new_average_11_matches : ℚ := total_runs_11_matches / 11
  let increased_average : ℚ := 32 + 5
  new_average_11_matches = increased_average :=
by
  sorry

end increase_average_by_runs_l38_3811


namespace f_minus_5_eq_12_l38_3836

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem f_minus_5_eq_12 : f (-5) = 12 := 
by sorry

end f_minus_5_eq_12_l38_3836


namespace susie_rooms_l38_3847

-- Define the conditions
def vacuum_time_per_room : ℕ := 20  -- in minutes
def total_vacuum_time : ℕ := 2 * 60  -- 2 hours in minutes

-- Define the number of rooms in Susie's house
def number_of_rooms (total_time room_time : ℕ) : ℕ := total_time / room_time

-- Prove that Susie has 6 rooms in her house
theorem susie_rooms : number_of_rooms total_vacuum_time vacuum_time_per_room = 6 :=
by
  sorry -- proof goes here

end susie_rooms_l38_3847


namespace parabola_equation_l38_3887

theorem parabola_equation (p : ℝ) (hp : 0 < p) (F : ℝ × ℝ) (Q : ℝ × ℝ) (PQ QF : ℝ)
  (hPQ : PQ = 8 / p) (hQF : QF = 8 / p + p / 2) (hDist : QF = 5 / 4 * PQ) : 
  ∃ x, y^2 = 4 * x :=
by
  sorry

end parabola_equation_l38_3887


namespace find_m_for_all_n_l38_3825

def sum_of_digits (k: ℕ) : ℕ :=
  k.digits 10 |>.sum

def A (k: ℕ) : ℕ :=
  -- Constructing the number A_k as described
  -- This is a placeholder for the actual implementation
  sorry

theorem find_m_for_all_n (n: ℕ) (hn: 0 < n) :
  ∃ m: ℕ, 0 < m ∧ n ∣ A m ∧ n ∣ m ∧ n ∣ sum_of_digits (A m) :=
sorry

end find_m_for_all_n_l38_3825


namespace distance_between_vertices_l38_3888

-- Define the equations of the parabolas
def C_eq (x : ℝ) : ℝ := x^2 + 6 * x + 13
def D_eq (x : ℝ) : ℝ := -x^2 + 2 * x + 8

-- Define the vertices of the parabolas
def vertex_C : (ℝ × ℝ) := (-3, 4)
def vertex_D : (ℝ × ℝ) := (1, 9)

-- Prove that the distance between the vertices is sqrt 41
theorem distance_between_vertices : 
  dist (vertex_C) (vertex_D) = Real.sqrt 41 := 
by
  sorry

end distance_between_vertices_l38_3888


namespace equal_pair_b_l38_3850

def exprA1 := -3^2
def exprA2 := -2^3

def exprB1 := -6^3
def exprB2 := (-6)^3

def exprC1 := -6^2
def exprC2 := (-6)^2

def exprD1 := (-3 * 2)^2
def exprD2 := (-3) * 2^2

theorem equal_pair_b : exprB1 = exprB2 :=
by {
  -- proof steps should go here
  sorry
}

end equal_pair_b_l38_3850


namespace arithmetic_sequence_sum_l38_3879

theorem arithmetic_sequence_sum (a b c : ℤ)
  (h1 : ∃ d : ℤ, a = 3 + d)
  (h2 : ∃ d : ℤ, b = 3 + 2 * d)
  (h3 : ∃ d : ℤ, c = 3 + 3 * d)
  (h4 : 3 + 3 * (c - 3) = 15) : a + b + c = 27 :=
by 
  sorry

end arithmetic_sequence_sum_l38_3879


namespace abs_diff_is_perfect_square_l38_3804

-- Define the conditions
variable (m n : ℤ) (h_odd_m : m % 2 = 1) (h_odd_n : n % 2 = 1)
variable (h_div : (n^2 - 1) ∣ (m^2 + 1 - n^2))

-- Theorem statement
theorem abs_diff_is_perfect_square : ∃ (k : ℤ), (m^2 + 1 - n^2) = k^2 :=
by
  sorry

end abs_diff_is_perfect_square_l38_3804


namespace train_speed_including_stoppages_l38_3859

noncomputable def trainSpeedExcludingStoppages : ℝ := 45
noncomputable def stoppageTimePerHour : ℝ := 20 / 60 -- 20 minutes per hour converted to hours
noncomputable def runningTimePerHour : ℝ := 1 - stoppageTimePerHour

theorem train_speed_including_stoppages (speed : ℝ) (stoppage : ℝ) (running_time : ℝ) : 
  speed = 45 → stoppage = 20 / 60 → running_time = 1 - stoppage → 
  (speed * running_time) / 1 = 30 :=
by sorry

end train_speed_including_stoppages_l38_3859


namespace expression_value_l38_3828

variable (m n : ℝ)

theorem expression_value (h : m - n = 1) : (m - n)^2 - 2 * m + 2 * n = -1 :=
by
  sorry

end expression_value_l38_3828


namespace BoatCrafters_boats_total_l38_3897

theorem BoatCrafters_boats_total
  (n_february: ℕ)
  (h_february: n_february = 5)
  (h_march: 3 * n_february = 15)
  (h_april: 3 * 15 = 45) :
  n_february + 15 + 45 = 65 := 
sorry

end BoatCrafters_boats_total_l38_3897


namespace order_of_numbers_l38_3803

noncomputable def a : ℝ := 60.7
noncomputable def b : ℝ := 0.76
noncomputable def c : ℝ := Real.log 0.76

theorem order_of_numbers : (c < b) ∧ (b < a) :=
by
  have h1 : c = Real.log 0.76 := rfl
  have h2 : b = 0.76 := rfl
  have h3 : a = 60.7 := rfl
  have hc : c < 0 := sorry
  have hb : 0 < b := sorry
  have ha : 1 < a := sorry
  sorry 

end order_of_numbers_l38_3803


namespace number_of_women_in_first_class_l38_3837

-- Definitions for the conditions
def total_passengers : ℕ := 180
def percentage_women : ℝ := 0.65
def percentage_women_first_class : ℝ := 0.15

-- The desired proof statement
theorem number_of_women_in_first_class :
  (round (total_passengers * percentage_women * percentage_women_first_class) = 18) :=
by
  sorry  

end number_of_women_in_first_class_l38_3837


namespace no_integer_solutions_for_mn_squared_eq_1980_l38_3872

theorem no_integer_solutions_for_mn_squared_eq_1980 :
  ¬ ∃ m n : ℤ, m^2 + n^2 = 1980 := 
sorry

end no_integer_solutions_for_mn_squared_eq_1980_l38_3872


namespace find_x_solution_l38_3856

theorem find_x_solution :
  ∃ x, 2 ^ (x / 2) * (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x)) = 6 ∧
       x = 2 * Real.log 1.5 / Real.log 2 := by
  sorry

end find_x_solution_l38_3856


namespace ivy_collectors_edition_dolls_l38_3852

-- Definitions from the conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def collectors_edition_dolls : ℕ := (2 * ivy_dolls) / 3

-- Assertion
theorem ivy_collectors_edition_dolls : collectors_edition_dolls = 20 := by
  sorry

end ivy_collectors_edition_dolls_l38_3852


namespace smallest_angle_of_triangle_l38_3831

noncomputable def smallest_angle (a b : ℝ) (c : ℝ) (h_sum : a + b + c = 180) : ℝ :=
  min a (min b c)

theorem smallest_angle_of_triangle :
  smallest_angle 60 65 (180 - (60 + 65)) (by norm_num) = 55 :=
by
  -- The correct proof steps should be provided for the result
  sorry

end smallest_angle_of_triangle_l38_3831


namespace truck_travel_yards_l38_3855

variables (b t : ℝ)

theorem truck_travel_yards : 
  (2 * (2 * b / 7) / (2 * t)) * 240 / 3 = (80 * b) / (7 * t) :=
by 
  sorry

end truck_travel_yards_l38_3855


namespace value_of_expression_l38_3860

theorem value_of_expression : 2 * 2015 - 2015 = 2015 :=
by
  sorry

end value_of_expression_l38_3860


namespace part1_solution_part2_solution_l38_3851

noncomputable def part1_expr := (1 / (Real.sqrt 5 + 2)) - (Real.sqrt 3 - 1)^0 - Real.sqrt (9 - 4 * Real.sqrt 5)
theorem part1_solution : part1_expr = 2 := by
  sorry

noncomputable def part2_expr := 2 * Real.sqrt 3 * 612 * (7/2)
theorem part2_solution : part2_expr = 5508 * Real.sqrt 3 := by
  sorry

end part1_solution_part2_solution_l38_3851


namespace min_chemistry_teachers_l38_3839

/--
A school has 7 maths teachers, 6 physics teachers, and some chemistry teachers.
Each teacher can teach a maximum of 3 subjects.
The minimum number of teachers required is 6.
Prove that the minimum number of chemistry teachers required is 1.
-/
theorem min_chemistry_teachers (C : ℕ) (math_teachers : ℕ := 7) (physics_teachers : ℕ := 6) 
  (max_subjects_per_teacher : ℕ := 3) (min_teachers_required : ℕ := 6) :
  7 + 6 + C ≤ 6 * 3 → C = 1 := 
by
  sorry

end min_chemistry_teachers_l38_3839


namespace min_board_size_l38_3870

theorem min_board_size (n : ℕ) (total_area : ℕ) (domino_area : ℕ) 
  (h1 : total_area = 2008) 
  (h2 : domino_area = 2) 
  (h3 : ∀ domino_count : ℕ, domino_count = total_area / domino_area → (∃ m : ℕ, (m+1) * (m+1) ≥ domino_count * (2 + 4) → n = m)) :
  n = 77 :=
by
  sorry

end min_board_size_l38_3870


namespace bank_transfer_amount_l38_3819

/-- Paul made two bank transfers. A service charge of 2% was added to each transaction.
The second transaction was reversed without the service charge. His account balance is now $307 if 
it was $400 before he made any transfers. Prove that the amount of the first bank transfer was 
$91.18. -/
theorem bank_transfer_amount (x : ℝ) (initial_balance final_balance : ℝ) (service_charge_rate : ℝ) 
  (second_transaction_reversed : Prop)
  (h_initial : initial_balance = 400)
  (h_final : final_balance = 307)
  (h_charge : service_charge_rate = 0.02)
  (h_reversal : second_transaction_reversed):
  initial_balance - (1 + service_charge_rate) * x = final_balance ↔
  x = 91.18 := 
by
  sorry

end bank_transfer_amount_l38_3819


namespace rectangle_area_l38_3826

theorem rectangle_area (x : ℝ) (h1 : x > 0) (h2 : x * 4 = 28) : x = 7 :=
sorry

end rectangle_area_l38_3826


namespace parabola_vertex_on_x_axis_l38_3801

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ h k : ℝ, y = (x : ℝ)^2 - 12 * x + c ∧
   (h = -12 / 2) ∧
   (k = c - 144 / 4) ∧
   (k = 0)) ↔ c = 36 :=
by
  sorry

end parabola_vertex_on_x_axis_l38_3801


namespace calc_result_l38_3876

-- Define the operation and conditions
def my_op (a b c : ℝ) : ℝ :=
  3 * (a - b - c)^2

theorem calc_result (x y z : ℝ) : 
  my_op ((x - y - z)^2) ((y - x - z)^2) ((z - x - y)^2) = 0 :=
by
  sorry

end calc_result_l38_3876


namespace bad_carrots_count_l38_3810

-- Define the number of carrots each person picked and the number of good carrots
def carol_picked := 29
def mom_picked := 16
def good_carrots := 38

-- Define the total number of carrots picked and the total number of bad carrots
def total_carrots := carol_picked + mom_picked
def bad_carrots := total_carrots - good_carrots

-- State the theorem that the number of bad carrots is 7
theorem bad_carrots_count :
  bad_carrots = 7 :=
by
  sorry

end bad_carrots_count_l38_3810


namespace part1_part2_l38_3885

theorem part1 (a : ℝ) (x : ℝ) (h : a > 0) :
  (|x + 1/a| + |x - a + 1|) ≥ 1 :=
sorry

theorem part2 (a : ℝ) (h1 : a > 0) (h2 : |3 + 1/a| + |3 - a + 1| < 11/2) :
  2 < a ∧ a < (13 + 3 * Real.sqrt 17) / 4 :=
sorry

end part1_part2_l38_3885


namespace difference_fraction_reciprocal_l38_3813

theorem difference_fraction_reciprocal :
  let f := (4 : ℚ) / 5
  let r := (5 : ℚ) / 4
  f - r = 9 / 20 :=
by
  sorry

end difference_fraction_reciprocal_l38_3813


namespace pto_shirts_total_cost_l38_3812

theorem pto_shirts_total_cost :
  let cost_Kindergartners : ℝ := 101 * 5.80
  let cost_FirstGraders : ℝ := 113 * 5.00
  let cost_SecondGraders : ℝ := 107 * 5.60
  let cost_ThirdGraders : ℝ := 108 * 5.25
  cost_Kindergartners + cost_FirstGraders + cost_SecondGraders + cost_ThirdGraders = 2317.00 := by
  sorry

end pto_shirts_total_cost_l38_3812


namespace cylinder_height_and_diameter_l38_3838

/-- The surface area of a sphere is the same as the curved surface area of a right circular cylinder.
    The height and diameter of the cylinder are the same, and the radius of the sphere is 4 cm.
    Prove that the height and diameter of the cylinder are both 8 cm. --/
theorem cylinder_height_and_diameter (r_sphere : ℝ) (r_cylinder h_cylinder : ℝ)
  (h1 : r_sphere = 4)
  (h2 : 4 * π * r_sphere^2 = 2 * π * r_cylinder * h_cylinder)
  (h3 : h_cylinder = 2 * r_cylinder) :
  h_cylinder = 8 ∧ r_cylinder = 4 :=
by
  -- Proof to be completed
  sorry

end cylinder_height_and_diameter_l38_3838


namespace Hari_contribution_l38_3820

theorem Hari_contribution (H : ℕ) (Praveen_capital : ℕ := 3500) (months_Praveen : ℕ := 12) 
                          (months_Hari : ℕ := 7) (profit_ratio_P : ℕ := 2) (profit_ratio_H : ℕ := 3) : 
                          (Praveen_capital * months_Praveen) * profit_ratio_H = (H * months_Hari) * profit_ratio_P → 
                          H = 9000 :=
by
  sorry

end Hari_contribution_l38_3820


namespace xy_value_l38_3853

theorem xy_value (x y : ℝ) (h : x * (x + 2 * y) = x^2 + 10) : x * y = 5 :=
by
  sorry

end xy_value_l38_3853


namespace bacteria_exceeds_day_l38_3863

theorem bacteria_exceeds_day :
  ∃ n : ℕ, 5 * 3^n > 200 ∧ ∀ m : ℕ, (m < n → 5 * 3^m ≤ 200) :=
sorry

end bacteria_exceeds_day_l38_3863


namespace probability_X_greater_than_2_l38_3893

noncomputable def probability_distribution (i : ℕ) : ℝ :=
  if h : 1 ≤ i ∧ i ≤ 4 then i / 10 else 0

theorem probability_X_greater_than_2 :
  (probability_distribution 3 + probability_distribution 4) = 0.7 := by 
  sorry

end probability_X_greater_than_2_l38_3893


namespace sum_of_four_digits_l38_3858

theorem sum_of_four_digits (EH OY AY OH : ℕ) (h1 : EH = 4 * OY) (h2 : AY = 4 * OH) : EH + OY + AY + OH = 150 :=
sorry

end sum_of_four_digits_l38_3858


namespace strawberry_jelly_sales_l38_3805

def jelly_sales (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  raspberry = grape / 3 ∧
  plum = 6

theorem strawberry_jelly_sales {grape strawberry raspberry plum : ℕ}
    (h : jelly_sales grape strawberry raspberry plum) : 
    strawberry = 18 :=
by
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end strawberry_jelly_sales_l38_3805


namespace same_speed_is_4_l38_3823

namespace SpeedProof

theorem same_speed_is_4 (x : ℝ) (h_jack_speed : x^2 - 11 * x - 22 = x - 10) (h_jill_speed : x^2 - 5 * x - 60 = (x - 10) * (x + 6)) :
  x = 14 → (x - 10) = 4 :=
by
  sorry

end SpeedProof

end same_speed_is_4_l38_3823


namespace value_of_a_l38_3832

noncomputable def f (x : ℝ) : ℝ := x^2 + 10
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h₁ : a > 0) (h₂ : f (g a) = 18) :
  a = Real.sqrt (5 + 2 * Real.sqrt 2) ∨ a = Real.sqrt (5 - 2 * Real.sqrt 2) := 
by
  sorry

end value_of_a_l38_3832


namespace smallest_perimeter_l38_3818

theorem smallest_perimeter (m n : ℕ) 
  (h1 : (m - 4) * (n - 4) = 8) 
  (h2 : ∀ k l : ℕ, (k - 4) * (l - 4) = 8 → 2 * k + 2 * l ≥ 2 * m + 2 * n) : 
  (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6) :=
sorry

end smallest_perimeter_l38_3818


namespace crackers_per_box_l38_3829

-- Given conditions
variables (x : ℕ)
variable (darren_boxes : ℕ := 4)
variable (calvin_boxes : ℕ := 2 * darren_boxes - 1)
variable (total_crackers : ℕ := 264)

-- Using the given conditions, create the proof statement to show x = 24
theorem crackers_per_box:
  11 * x = total_crackers → x = 24 :=
by
  sorry

end crackers_per_box_l38_3829


namespace tan_beta_solution_l38_3883

theorem tan_beta_solution
  (α β : ℝ)
  (h₁ : Real.tan α = 2)
  (h₂ : Real.tan (α + β) = -1) :
  Real.tan β = 3 := 
sorry

end tan_beta_solution_l38_3883


namespace math_proof_problem_l38_3871

-- Define constants
def x := 2000000000000
def y := 1111111111111

-- Prove the main statement
theorem math_proof_problem :
  2 * (x - y) = 1777777777778 := 
  by
    sorry

end math_proof_problem_l38_3871


namespace even_and_nonneg_range_l38_3833

theorem even_and_nonneg_range : 
  (∀ x : ℝ, abs x = abs (-x) ∧ (abs x ≥ 0)) ∧ (∀ x : ℝ, x^2 + abs x = ( (-x)^2) + abs (-x) ∧ (x^2 + abs x ≥ 0)) := sorry

end even_and_nonneg_range_l38_3833


namespace battery_charge_to_60_percent_l38_3875

noncomputable def battery_charge_time (initial_charge_percent : ℝ) (initial_time_minutes : ℕ) (additional_time_minutes : ℕ) : ℕ :=
  let rate_per_minute := initial_charge_percent / initial_time_minutes
  let additional_charge_percent := additional_time_minutes * rate_per_minute
  let total_percent := initial_charge_percent + additional_charge_percent
  if total_percent = 60 then
    initial_time_minutes + additional_time_minutes
  else
    sorry

theorem battery_charge_to_60_percent : battery_charge_time 20 60 120 = 180 :=
by
  -- The formal proof will be provided here.
  sorry

end battery_charge_to_60_percent_l38_3875


namespace bruce_mango_purchase_l38_3848

theorem bruce_mango_purchase (m : ℕ) 
  (cost_grapes : 8 * 70 = 560)
  (cost_total : 560 + 55 * m = 1110) : 
  m = 10 :=
by
  sorry

end bruce_mango_purchase_l38_3848


namespace rectangle_area_error_83_percent_l38_3840

theorem rectangle_area_error_83_percent (L W : ℝ) :
  let actual_area := L * W
  let measured_length := 1.14 * L
  let measured_width := 0.95 * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 8.3 := by
  sorry

end rectangle_area_error_83_percent_l38_3840


namespace quadratic_root_sum_m_n_l38_3842

theorem quadratic_root_sum_m_n (m n : ℤ) :
  (∃ x : ℤ, x^2 + m * x + 2 * n = 0 ∧ x = 2) → m + n = -2 :=
by
  sorry

end quadratic_root_sum_m_n_l38_3842


namespace items_left_in_cart_l38_3857

-- Define the initial items in the shopping cart
def initial_items : ℕ := 18

-- Define the items deleted from the shopping cart
def deleted_items : ℕ := 10

-- Theorem statement: Prove the remaining items are 8
theorem items_left_in_cart : initial_items - deleted_items = 8 :=
by
  -- Sorry marks the place where the proof would go.
  sorry

end items_left_in_cart_l38_3857


namespace doubled_dimensions_volume_l38_3864

theorem doubled_dimensions_volume (original_volume : ℝ) (length_factor width_factor height_factor : ℝ) 
  (h : original_volume = 3) 
  (hl : length_factor = 2)
  (hw : width_factor = 2)
  (hh : height_factor = 2) : 
  original_volume * length_factor * width_factor * height_factor = 24 :=
by
  sorry

end doubled_dimensions_volume_l38_3864


namespace necessary_but_not_sufficient_condition_l38_3817

open Set

variable {α : Type*}

def M : Set ℝ := { x | 0 < x ∧ x ≤ 4 }
def N : Set ℝ := { x | 2 ≤ x ∧ x ≤ 3 }

theorem necessary_but_not_sufficient_condition :
  (N ⊆ M) ∧ (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
by
  sorry

end necessary_but_not_sufficient_condition_l38_3817


namespace range_of_a_l38_3874

theorem range_of_a (a : ℝ) (x : ℝ) :
  (x^2 - 4 * a * x + 3 * a^2 < 0 → (x^2 - x - 6 ≤ 0 ∨ x^2 + 2 * x - 8 > 0)) → a < 0 → 
  (a ≤ -4 ∨ -2 / 3 ≤ a ∧ a < 0) :=
by
  sorry

end range_of_a_l38_3874


namespace negation_of_proposition_l38_3845

theorem negation_of_proposition :
  (¬ ∀ (x : ℝ), |x| < 0) ↔ (∃ (x : ℝ), |x| ≥ 0) := 
sorry

end negation_of_proposition_l38_3845


namespace remainder_when_concat_numbers_1_to_54_div_55_l38_3800

def concat_numbers (n : ℕ) : ℕ :=
  let digits x := x.digits 10
  (List.range n).bind digits |> List.reverse |> List.foldl (λ acc x => acc * 10 + x) 0

theorem remainder_when_concat_numbers_1_to_54_div_55 :
  let M := concat_numbers 55
  M % 55 = 44 :=
by
  sorry

end remainder_when_concat_numbers_1_to_54_div_55_l38_3800


namespace num_four_digit_int_with_4_or_5_correct_l38_3841

def num_four_digit_int_with_4_or_5 : ℕ :=
  5416

theorem num_four_digit_int_with_4_or_5_correct (A B : ℕ) (hA : A = 9000) (hB : B = 3584) :
  num_four_digit_int_with_4_or_5 = A - B :=
by
  rw [hA, hB]
  sorry

end num_four_digit_int_with_4_or_5_correct_l38_3841


namespace optimal_pricing_for_max_profit_l38_3891

noncomputable def sales_profit (x : ℝ) : ℝ :=
  -5 * x^3 + 45 * x^2 - 75 * x + 675

theorem optimal_pricing_for_max_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x < 9 ∧ ∀ y : ℝ, 0 ≤ y ∧ y < 9 → sales_profit y ≤ sales_profit 5 ∧ (14 - 5 = 9) :=
by
  sorry

end optimal_pricing_for_max_profit_l38_3891


namespace periodic_odd_fn_calc_l38_3890

theorem periodic_odd_fn_calc :
  ∀ (f : ℝ → ℝ),
  (∀ x, f (x + 2) = f x) ∧ (∀ x, f (-x) = -f x) ∧ (∀ x, 0 < x ∧ x < 1 → f x = 4^x) →
  f (-5 / 2) + f 2 = -2 :=
by
  intros f h
  sorry

end periodic_odd_fn_calc_l38_3890


namespace movie_revenue_multiple_correct_l38_3884

-- Definitions from the conditions
def opening_weekend_revenue : ℝ := 120 * 10^6
def company_share_fraction : ℝ := 0.60
def profit : ℝ := 192 * 10^6
def production_cost : ℝ := 60 * 10^6

-- The statement to prove
theorem movie_revenue_multiple_correct : 
  ∃ M : ℝ, (company_share_fraction * (opening_weekend_revenue * M) - production_cost = profit) ∧ M = 3.5 :=
by
  sorry

end movie_revenue_multiple_correct_l38_3884


namespace interval_of_monotonic_increase_sum_greater_than_2e_l38_3846

noncomputable def f (a x : ℝ) : ℝ := a * x / (Real.log x)

theorem interval_of_monotonic_increase :
  ∀ (x : ℝ), (e < x → f 1 x > f 1 e) := 
sorry

theorem sum_greater_than_2e (x1 x2 : ℝ) (a : ℝ) (h1 : x1 ≠ x2) (hx1 : f 1 x1 = 1) (hx2 : f 1 x2 = 1) :
  x1 + x2 > 2 * Real.exp 1 :=
sorry

end interval_of_monotonic_increase_sum_greater_than_2e_l38_3846


namespace min_value_of_squared_sum_l38_3815

open Real

theorem min_value_of_squared_sum (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
  ∃ m, m = (x^2 + y^2 + z^2) ∧ m = 16 / 3 :=
by
  sorry

end min_value_of_squared_sum_l38_3815


namespace white_surface_area_fraction_l38_3877

theorem white_surface_area_fraction :
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  (white_faces_exposed / surface_area) = (7 / 8) :=
by
  let larger_cube_edge := 4
  let smaller_cube_edge := 1
  let total_smaller_cubes := 64
  let white_cubes := 48
  let black_cubes := 16
  let total_faces := 6
  let black_cubes_per_face := 2
  let surface_area := total_faces * larger_cube_edge^2
  let black_faces_exposed := total_faces * black_cubes_per_face
  let white_faces_exposed := surface_area - black_faces_exposed
  have h_white_fraction : (white_faces_exposed / surface_area) = (7 / 8) := sorry
  exact h_white_fraction

end white_surface_area_fraction_l38_3877


namespace weight_of_pants_l38_3822

def weight_socks := 2
def weight_underwear := 4
def weight_shirt := 5
def weight_shorts := 8
def total_allowed := 50

def weight_total (num_shirts num_shorts num_socks num_underwear : Nat) :=
  num_shirts * weight_shirt + num_shorts * weight_shorts + num_socks * weight_socks + num_underwear * weight_underwear

def items_in_wash := weight_total 2 1 3 4

theorem weight_of_pants :
  let weight_pants := total_allowed - items_in_wash
  weight_pants = 10 :=
by
  sorry

end weight_of_pants_l38_3822


namespace truck_driver_gas_l38_3843

variables (miles_per_gallon distance_to_station gallons_to_add gallons_in_tank total_gallons_needed : ℕ)
variables (current_gas_in_tank : ℕ)
variables (h1 : miles_per_gallon = 3)
variables (h2 : distance_to_station = 90)
variables (h3 : gallons_to_add = 18)

theorem truck_driver_gas :
  current_gas_in_tank = 12 :=
by
  -- Prove that the truck driver already has 12 gallons of gas in his tank,
  -- given the conditions provided.
  sorry

end truck_driver_gas_l38_3843


namespace cone_base_radius_l38_3862

-- Definitions based on conditions
def sphere_radius : ℝ := 1
def cone_height : ℝ := 2

-- Problem statement
theorem cone_base_radius {r : ℝ} 
  (h1 : ∀ x y z : ℝ, (x = sphere_radius ∧ y = sphere_radius ∧ z = sphere_radius) → 
                     (x + y + z = 3 * sphere_radius)) 
  (h2 : ∃ (O O1 O2 O3 : ℝ), (O = 0) ∧ (O1 = 1) ∧ (O2 = 1) ∧ (O3 = 1)) 
  (h3 : ∀ x y z : ℝ, (x + y + z = 3 * sphere_radius) → 
                     (y = z) → (x = z) → y * z + x * z + x * y = 3 * sphere_radius ^ 2)
  (h4 : ∀ h : ℝ, h = cone_height) :
  r = (Real.sqrt 3 / 6) :=
sorry

end cone_base_radius_l38_3862


namespace ball_hits_ground_l38_3854

theorem ball_hits_ground (t : ℚ) : 
  (∃ t ≥ 0, (-4.9 * (t^2 : ℝ) + 5 * t + 10 = 0)) → t = 100 / 49 :=
by
  sorry

end ball_hits_ground_l38_3854


namespace mean_score_l38_3816

theorem mean_score (μ σ : ℝ)
  (h1 : 86 = μ - 7 * σ)
  (h2 : 90 = μ + 3 * σ) : μ = 88.8 := by
  -- Proof steps are not included as per requirements.
  sorry

end mean_score_l38_3816


namespace solve_E_l38_3824

-- Definitions based on the conditions provided
variables {A H S M C O E : ℕ}

-- Given conditions
def algebra_books := A
def geometry_books := H
def history_books := C
def S_algebra_books := S
def M_geometry_books := M
def O_history_books := O
def E_algebra_books := E

-- Prove that E = (AM + AO - SH - SC) / (M + O - H - C) given the conditions
theorem solve_E (h1: A ≠ H) (h2: A ≠ S) (h3: A ≠ M) (h4: A ≠ C) (h5: A ≠ O) (h6: A ≠ E)
                (h7: H ≠ S) (h8: H ≠ M) (h9: H ≠ C) (h10: H ≠ O) (h11: H ≠ E)
                (h12: S ≠ M) (h13: S ≠ C) (h14: S ≠ O) (h15: S ≠ E)
                (h16: M ≠ C) (h17: M ≠ O) (h18: M ≠ E)
                (h19: C ≠ O) (h20: C ≠ E)
                (h21: O ≠ E)
                (pos1: 0 < A) (pos2: 0 < H) (pos3: 0 < S) (pos4: 0 < M) (pos5: 0 < C)
                (pos6: 0 < O) (pos7: 0 < E) :
  E = (A * M + A * O - S * H - S * C) / (M + O - H - C) :=
sorry

end solve_E_l38_3824


namespace remainder_of_3_pow_800_mod_17_l38_3849

theorem remainder_of_3_pow_800_mod_17 : (3^800) % 17 = 1 := by
  sorry

end remainder_of_3_pow_800_mod_17_l38_3849
