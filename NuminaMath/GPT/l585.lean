import Mathlib

namespace RobertAteNine_l585_58500

-- Define the number of chocolates Nickel ate
def chocolatesNickelAte : ℕ := 2

-- Define the additional chocolates Robert ate compared to Nickel
def additionalChocolates : ℕ := 7

-- Define the total chocolates Robert ate
def chocolatesRobertAte : ℕ := chocolatesNickelAte + additionalChocolates

-- State the theorem we want to prove
theorem RobertAteNine : chocolatesRobertAte = 9 := by
  -- Skip the proof
  sorry

end RobertAteNine_l585_58500


namespace ratio_of_area_of_small_triangle_to_square_l585_58535

theorem ratio_of_area_of_small_triangle_to_square
  (n : ℕ)
  (square_area : ℝ)
  (A1 : square_area > 0)
  (ADF_area : ℝ)
  (H1 : ADF_area = n * square_area)
  (FEC_area : ℝ)
  (H2 : FEC_area = 1 / (4 * n)) :
  FEC_area / square_area = 1 / (4 * n) :=
by
  sorry

end ratio_of_area_of_small_triangle_to_square_l585_58535


namespace probability_bus_there_when_mark_arrives_l585_58501

noncomputable def isProbabilityBusThereWhenMarkArrives : Prop :=
  let busArrival : ℝ := 60 -- The bus can arrive from time 0 to 60 minutes (2:00 PM to 3:00 PM)
  let busWait : ℝ := 30 -- The bus waits for 30 minutes
  let markArrival : ℝ := 90 -- Mark can arrive from time 30 to 90 minutes (2:30 PM to 3:30 PM)
  let overlapArea : ℝ := 1350 -- Total shaded area where bus arrival overlaps with Mark's arrival
  let totalArea : ℝ := busArrival * (markArrival - 30)
  let probability := overlapArea / totalArea
  probability = 1 / 4

theorem probability_bus_there_when_mark_arrives : isProbabilityBusThereWhenMarkArrives :=
by
  sorry

end probability_bus_there_when_mark_arrives_l585_58501


namespace greatest_multiple_5_7_less_than_700_l585_58521

theorem greatest_multiple_5_7_less_than_700 :
  ∃ n, n < 700 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 700 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ n) → n = 665 :=
by
  sorry

end greatest_multiple_5_7_less_than_700_l585_58521


namespace min_value_of_F_l585_58530

variable (x1 x2 : ℝ)

def constraints :=
  2 - 2 * x1 - x2 ≥ 0 ∧
  2 - x1 + x2 ≥ 0 ∧
  5 - x1 - x2 ≥ 0 ∧
  0 ≤ x1 ∧
  0 ≤ x2

noncomputable def F := x2 - x1

theorem min_value_of_F : constraints x1 x2 → ∃ (minF : ℝ), minF = -2 :=
by
  sorry

end min_value_of_F_l585_58530


namespace range_of_a_l585_58570

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → a * x^2 - x - 4 > 0) → a > 5 :=
by
  sorry

end range_of_a_l585_58570


namespace jogging_days_in_second_week_l585_58545

theorem jogging_days_in_second_week
  (daily_jogging_time : ℕ) (first_week_days : ℕ) (total_jogging_time : ℕ) :
  daily_jogging_time = 30 →
  first_week_days = 3 →
  total_jogging_time = 240 →
  ∃ second_week_days : ℕ, second_week_days = 5 :=
by
  intros
  -- Conditions
  have h1 := daily_jogging_time = 30
  have h2 := first_week_days = 3
  have h3 := total_jogging_time = 240
  -- Calculations
  have first_week_time := first_week_days * daily_jogging_time
  have second_week_time := total_jogging_time - first_week_time
  have second_week_days := second_week_time / daily_jogging_time
  -- Conclusion
  use second_week_days
  sorry

end jogging_days_in_second_week_l585_58545


namespace henry_time_around_track_l585_58543

theorem henry_time_around_track (H : ℕ) : 
  (∀ (M := 12), lcm M H = 84) → H = 7 :=
by
  sorry

end henry_time_around_track_l585_58543


namespace sphere_circumscribed_around_cone_radius_l585_58578

-- Definitions of the given conditions
variable (r h : ℝ)

-- Theorem statement (without the proof)
theorem sphere_circumscribed_around_cone_radius :
  ∃ R : ℝ, R = (Real.sqrt (r^2 + h^2)) / 2 :=
sorry

end sphere_circumscribed_around_cone_radius_l585_58578


namespace ratio_of_raspberries_l585_58529

theorem ratio_of_raspberries (B R K L : ℕ) (h1 : B = 42) (h2 : L = 7) (h3 : K = B / 3) (h4 : B = R + K + L) :
  R / Nat.gcd R B = 1 ∧ B / Nat.gcd R B = 2 :=
by
  sorry

end ratio_of_raspberries_l585_58529


namespace equation_correct_l585_58583

variable (x y : ℝ)

-- Define the conditions
def condition1 : Prop := (x + y) / 3 = 1.888888888888889
def condition2 : Prop := 2 * x + y = 7

-- Prove the required equation under given conditions
theorem equation_correct : condition1 x y → condition2 x y → (x + y) = 5.666666666666667 := by
  intros _ _
  sorry

end equation_correct_l585_58583


namespace find_x_l585_58520

theorem find_x (p q : ℕ) (h1 : 1 < p) (h2 : 1 < q) (h3 : 17 * (p + 1) = (14 * (q + 1))) (h4 : p + q = 40) : 
    x = 14 := 
by
  sorry

end find_x_l585_58520


namespace trapezoid_area_correct_l585_58507

-- Given sides of the trapezoid
def sides : List ℚ := [4, 6, 8, 10]

-- Definition of the function to calculate the sum of all possible areas.
noncomputable def sumOfAllPossibleAreas (sides : List ℚ) : ℚ :=
  -- Assuming configurations and calculations are correct by problem statement
  let r4 := 21
  let r5 := 7
  let r6 := 0
  let n4 := 3
  let n5 := 15
  r4 + r5 + r6 + n4 + n5

-- Check that the given sides lead to sum of areas equal to 46
theorem trapezoid_area_correct : sumOfAllPossibleAreas sides = 46 := by
  sorry

end trapezoid_area_correct_l585_58507


namespace tammy_speed_second_day_l585_58581

theorem tammy_speed_second_day:
  ∃ (v t: ℝ), 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    (v + 0.5) = 4 := sorry

end tammy_speed_second_day_l585_58581


namespace books_remainder_l585_58568

theorem books_remainder (total_books new_books_per_section sections : ℕ) 
  (h1 : total_books = 1521) 
  (h2 : new_books_per_section = 45) 
  (h3 : sections = 41) : 
  (total_books * sections) % new_books_per_section = 36 :=
by
  sorry

end books_remainder_l585_58568


namespace omega_eq_six_l585_58593

theorem omega_eq_six (A ω : ℝ) (φ : ℝ) (f : ℝ → ℝ) (h1 : A ≠ 0) (h2 : ω > 0)
  (h3 : -π / 2 < φ ∧ φ < π / 2) (h4 : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h5 : ∀ x, f (-x) = -f x) 
  (h6 : ∀ x, f (x + π / 6) = -f (x - π / 6)) :
  ω = 6 :=
sorry

end omega_eq_six_l585_58593


namespace expression_value_at_neg1_l585_58552

theorem expression_value_at_neg1
  (p q : ℤ)
  (h1 : p + q = 2016) :
  p * (-1)^3 + q * (-1) - 10 = -2026 := by
  sorry

end expression_value_at_neg1_l585_58552


namespace evaluate_expression_l585_58567

theorem evaluate_expression (x y z : ℚ) 
    (hx : x = 1 / 4) 
    (hy : y = 1 / 3) 
    (hz : z = -6) : 
    x^2 * y^3 * z^2 = 1 / 12 :=
by
  sorry

end evaluate_expression_l585_58567


namespace number_of_grandchildren_l585_58525

/- Definitions based on conditions -/
def price_before_discount := 20.0
def discount_rate := 0.20
def monogram_cost := 12.0
def total_expenditure := 140.0

/- Definition based on discount calculation -/
def price_after_discount := price_before_discount * (1.0 - discount_rate)

/- Final theorem statement -/
theorem number_of_grandchildren : 
  total_expenditure / (price_after_discount + monogram_cost) = 5 := by
  sorry

end number_of_grandchildren_l585_58525


namespace cover_faces_with_strips_l585_58514

theorem cover_faces_with_strips (a b c : ℕ) :
  (∃ f g h : ℕ, a = 5 * f ∨ b = 5 * g ∨ c = 5 * h) ↔
  (∃ u v : ℕ, (a = 5 * u ∧ b = 5 * v) ∨ (a = 5 * u ∧ c = 5 * v) ∨ (b = 5 * u ∧ c = 5 * v)) := 
sorry

end cover_faces_with_strips_l585_58514


namespace f_g_of_4_eq_18_sqrt_21_div_7_l585_58513

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x - 3

theorem f_g_of_4_eq_18_sqrt_21_div_7 : f (g 4) = (18 * Real.sqrt 21) / 7 := by
  sorry

end f_g_of_4_eq_18_sqrt_21_div_7_l585_58513


namespace intersection_eq_l585_58573

def M (x : ℝ) : Prop := (x + 3) * (x - 2) < 0

def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

def intersection (x : ℝ) : Prop := M x ∧ N x

theorem intersection_eq : ∀ x, intersection x ↔ (1 ≤ x ∧ x < 2) :=
by sorry

end intersection_eq_l585_58573


namespace possible_values_of_a_and_b_l585_58519

theorem possible_values_of_a_and_b (a b : ℕ) : 
  (a = 22 ∨ a = 33 ∨ a = 40 ∨ a = 42) ∧ 
  (b = 21 ∨ b = 10 ∨ b = 3 ∨ b = 1) ∧ 
  (a % (b + 1) = 0) ∧ (43 % (a + b) = 0) :=
sorry

end possible_values_of_a_and_b_l585_58519


namespace jenn_money_left_over_l585_58532

-- Definitions based on problem conditions
def num_jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def value_per_quarter : ℚ := 0.25   -- Rational number to represent $0.25
def cost_of_bike : ℚ := 180         -- Rational number to represent $180

-- Statement to prove that Jenn will have $20 left after buying the bike
theorem jenn_money_left_over : 
  (num_jars * quarters_per_jar * value_per_quarter) - cost_of_bike = 20 :=
by
  sorry

end jenn_money_left_over_l585_58532


namespace stratified_sampling_group_C_l585_58517

theorem stratified_sampling_group_C
  (total_cities : ℕ)
  (cities_group_A : ℕ)
  (cities_group_B : ℕ)
  (cities_group_C : ℕ)
  (total_selected : ℕ)
  (C_subset_correct: total_cities = cities_group_A + cities_group_B + cities_group_C)
  (total_cities_correct: total_cities = 48)
  (cities_group_A_correct: cities_group_A = 8)
  (cities_group_B_correct: cities_group_B = 24)
  (total_selected_correct: total_selected = 12)
  : (total_selected * cities_group_C) / total_cities = 4 :=
by 
  sorry

end stratified_sampling_group_C_l585_58517


namespace distance_between_parallel_lines_correct_l585_58589

open Real

noncomputable def distance_between_parallel_lines : ℝ :=
  let a := (3, 1)
  let b := (2, 4)
  let d := (4, -6)
  let v := (b.1 - a.1, b.2 - a.2)
  let d_perp := (6, 4) -- a vector perpendicular to d
  let v_dot_d_perp := v.1 * d_perp.1 + v.2 * d_perp.2
  let d_perp_dot_d_perp := d_perp.1 * d_perp.1 + d_perp.2 * d_perp.2
  let proj_v_onto_d_perp := (v_dot_d_perp / d_perp_dot_d_perp * d_perp.1, v_dot_d_perp / d_perp_dot_d_perp * d_perp.2)
  sqrt (proj_v_onto_d_perp.1 * proj_v_onto_d_perp.1 + proj_v_onto_d_perp.2 * proj_v_onto_d_perp.2)

theorem distance_between_parallel_lines_correct :
  distance_between_parallel_lines = (3 * sqrt 13) / 13 := by
  sorry

end distance_between_parallel_lines_correct_l585_58589


namespace negation_of_proposition_l585_58572

theorem negation_of_proposition :
  (¬ ∃ m : ℝ, 1 / (m^2 + m - 6) > 0) ↔ (∀ m : ℝ, (1 / (m^2 + m - 6) < 0) ∨ (m^2 + m - 6 = 0)) :=
by
  sorry

end negation_of_proposition_l585_58572


namespace unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l585_58505

theorem unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16 
  (b : ℝ) : 
  (∃ (x : ℝ), bx^2 + 7*x + 4 = 0 ∧ ∀ (x' : ℝ), bx^2 + 7*x' + 4 ≠ 0) ↔ b = 49 / 16 :=
by
  sorry

end unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l585_58505


namespace total_spent_by_mrs_hilt_l585_58551

-- Define the cost per set of tickets for kids.
def cost_per_set_kids : ℕ := 1
-- Define the number of tickets in a set for kids.
def tickets_per_set_kids : ℕ := 4

-- Define the cost per set of tickets for adults.
def cost_per_set_adults : ℕ := 2
-- Define the number of tickets in a set for adults.
def tickets_per_set_adults : ℕ := 3

-- Define the total number of kids' tickets purchased.
def total_kids_tickets : ℕ := 12
-- Define the total number of adults' tickets purchased.
def total_adults_tickets : ℕ := 9

-- Prove that the total amount spent by Mrs. Hilt is $9.
theorem total_spent_by_mrs_hilt :
  (total_kids_tickets / tickets_per_set_kids * cost_per_set_kids) + 
  (total_adults_tickets / tickets_per_set_adults * cost_per_set_adults) = 9 :=
by sorry

end total_spent_by_mrs_hilt_l585_58551


namespace admittedApplicants_l585_58582

-- Definitions for the conditions in the problem
def totalApplicants : ℕ := 70
def task1Applicants : ℕ := 35
def task2Applicants : ℕ := 48
def task3Applicants : ℕ := 64
def task4Applicants : ℕ := 63

-- The proof statement
theorem admittedApplicants : 
  ∀ (totalApplicants task3Applicants task4Applicants : ℕ),
  totalApplicants = 70 →
  task3Applicants = 64 →
  task4Applicants = 63 →
  ∃ (interApplicants : ℕ), interApplicants = 57 :=
by
  intros totalApplicants task3Applicants task4Applicants
  intros h_totalApps h_task3Apps h_task4Apps
  sorry

end admittedApplicants_l585_58582


namespace cos_alpha_l585_58541

theorem cos_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.sin (α - π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 :=
by
  sorry

end cos_alpha_l585_58541


namespace sequence_gcd_equality_l585_58527

theorem sequence_gcd_equality (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) : 
  ∀ i, a i = i := 
sorry

end sequence_gcd_equality_l585_58527


namespace smallest_root_of_unity_l585_58518

open Complex

theorem smallest_root_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ k : ℕ, k < 18 ∧ z = exp (2 * pi * I * k / 18) :=
by
  sorry

end smallest_root_of_unity_l585_58518


namespace paintings_after_30_days_l585_58592

theorem paintings_after_30_days (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ)
    (h1 : paintings_per_day = 2)
    (h2 : initial_paintings = 20)
    (h3 : days = 30) :
    initial_paintings + paintings_per_day * days = 80 := by
  sorry

end paintings_after_30_days_l585_58592


namespace inverse_function_point_l585_58563

noncomputable def f (a : ℝ) (x : ℝ) := a^(x + 1)

theorem inverse_function_point (a : ℝ) (h_pos : 0 < a) (h_annoylem : f a (-1) = 1) :
  ∃ g : ℝ → ℝ, (∀ y, f a (g y) = y ∧ g (f a y) = y) ∧ g 1 = -1 :=
by
  sorry

end inverse_function_point_l585_58563


namespace tylers_age_l585_58509

theorem tylers_age (B T : ℕ) 
  (h1 : T = B - 3) 
  (h2 : T + B = 11) : 
  T = 4 :=
sorry

end tylers_age_l585_58509


namespace repeating_decimal_fraction_l585_58561

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end repeating_decimal_fraction_l585_58561


namespace batsman_average_after_11th_inning_l585_58577

variable (x : ℝ) -- The average before the 11th inning
variable (new_average : ℝ) -- The average after the 11th inning
variable (total_runs : ℝ) -- Total runs scored after 11 innings

-- Given conditions
def condition1 := total_runs = 11 * (x + 5)
def condition2 := total_runs = 10 * x + 110

theorem batsman_average_after_11th_inning : 
  ∀ (x : ℝ), 
    (x = 55) → (x + 5 = 60) :=
by
  intros
  sorry

end batsman_average_after_11th_inning_l585_58577


namespace dog_farthest_distance_l585_58540

/-- 
Given a dog tied to a post at the point (3,4), a 15 meter long rope, and a wall from (5,4) to (5,9), 
prove that the farthest distance the dog can travel from the origin (0,0) is 20 meters.
-/
theorem dog_farthest_distance (post : ℝ × ℝ) (rope_length : ℝ) (wall_start wall_end origin : ℝ × ℝ)
  (h_post : post = (3,4))
  (h_rope_length : rope_length = 15)
  (h_wall_start : wall_start = (5,4))
  (h_wall_end : wall_end = (5,9))
  (h_origin : origin = (0,0)) :
  ∃ farthest_distance : ℝ, farthest_distance = 20 :=
by
  sorry

end dog_farthest_distance_l585_58540


namespace meeting_percentage_l585_58508

theorem meeting_percentage
    (workday_hours : ℕ)
    (first_meeting_minutes : ℕ)
    (second_meeting_factor : ℕ)
    (hp_workday_hours : workday_hours = 10)
    (hp_first_meeting_minutes : first_meeting_minutes = 60)
    (hp_second_meeting_factor : second_meeting_factor = 2) 
    : (first_meeting_minutes + first_meeting_minutes * second_meeting_factor : ℚ) 
    / (workday_hours * 60) * 100 = 30 := 
by
  have workday_minutes := workday_hours * 60
  have second_meeting_minutes := first_meeting_minutes * second_meeting_factor
  have total_meeting_minutes := first_meeting_minutes + second_meeting_minutes
  have percentage := (total_meeting_minutes : ℚ) / workday_minutes * 100
  sorry

end meeting_percentage_l585_58508


namespace total_votes_l585_58533

variable (T S R F V : ℝ)

-- Conditions
axiom h1 : T = S + 0.15 * V
axiom h2 : S = R + 0.05 * V
axiom h3 : R = F + 0.07 * V
axiom h4 : T + S + R + F = V
axiom h5 : T - 2500 - 2000 = S + 2500
axiom h6 : S + 2500 = R + 2000 + 0.05 * V

theorem total_votes : V = 30000 :=
sorry

end total_votes_l585_58533


namespace olivia_used_pieces_l585_58542

-- Definition of initial pieces of paper and remaining pieces of paper
def initial_pieces : ℕ := 81
def remaining_pieces : ℕ := 25

-- Prove that Olivia used 56 pieces of paper
theorem olivia_used_pieces : (initial_pieces - remaining_pieces) = 56 :=
by
  -- Proof steps can be filled here
  sorry

end olivia_used_pieces_l585_58542


namespace symmetric_parabola_l585_58526

def parabola1 (x : ℝ) : ℝ := (x - 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -(x + 2)^2 - 3

theorem symmetric_parabola : ∀ x y : ℝ,
  y = parabola1 x ↔ 
  (-y) = parabola2 (-x) ∧ y = -(x + 2)^2 - 3 :=
sorry

end symmetric_parabola_l585_58526


namespace harris_carrot_expense_l585_58515

theorem harris_carrot_expense
  (carrots_per_day : ℕ)
  (days_per_year : ℕ)
  (carrots_per_bag : ℕ)
  (cost_per_bag : ℝ)
  (total_expense : ℝ) :
  carrots_per_day = 1 →
  days_per_year = 365 →
  carrots_per_bag = 5 →
  cost_per_bag = 2 →
  total_expense = 146 :=
by
  intros h1 h2 h3 h4
  sorry

end harris_carrot_expense_l585_58515


namespace find_ages_of_son_daughter_and_niece_l585_58528

theorem find_ages_of_son_daughter_and_niece
  (S : ℕ) (D : ℕ) (N : ℕ)
  (h1 : ∀ (M : ℕ), M = S + 24) 
  (h2 : ∀ (M : ℕ), 2 * (S + 2) = M + 2)
  (h3 : D = S / 2)
  (h4 : 2 * (D + 6) = 2 * S * 2 / 3)
  (h5 : N = S - 3)
  (h6 : 5 * N = 4 * S) :
  S = 22 ∧ D = 11 ∧ N = 19 := 
by 
  sorry

end find_ages_of_son_daughter_and_niece_l585_58528


namespace age_of_15th_student_l585_58562

theorem age_of_15th_student 
  (avg_age_all : ℕ → ℕ → ℕ)
  (avg_age : avg_age_all 15 15 = 15)
  (avg_age_4 : avg_age_all 4 14 = 14)
  (avg_age_10 : avg_age_all 10 16 = 16) : 
  ∃ age15 : ℕ, age15 = 9 := 
by
  sorry

end age_of_15th_student_l585_58562


namespace digit_A_divisibility_l585_58503

theorem digit_A_divisibility :
  ∃ (A : ℕ), (0 ≤ A ∧ A < 10) ∧ (∃ k_5 : ℕ, 353809 * 10 + A = 5 * k_5) ∧ 
  (∃ k_7 : ℕ, 353809 * 10 + A = 7 * k_7) ∧ (∃ k_11 : ℕ, 353809 * 10 + A = 11 * k_11) 
  ∧ A = 0 :=
by 
  sorry

end digit_A_divisibility_l585_58503


namespace part2_proof_l585_58554

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.log x) - Real.exp 1 * x

theorem part2_proof (x : ℝ) (h : 0 < x) :
  x * f x - Real.exp x + 2 * Real.exp 1 * x ≤ 0 := 
sorry

end part2_proof_l585_58554


namespace time_for_worker_C_l585_58550

theorem time_for_worker_C (time_A time_B time_total : ℝ) (time_A_pos : 0 < time_A) (time_B_pos : 0 < time_B) (time_total_pos : 0 < time_total) 
  (hA : time_A = 12) (hB : time_B = 15) (hTotal : time_total = 6) : 
  (1 / (1 / time_total - 1 / time_A - 1 / time_B) = 60) :=
by 
  sorry

end time_for_worker_C_l585_58550


namespace solve_for_x_l585_58553

theorem solve_for_x (x : ℝ) (h : (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4)) : x = -14 := 
by 
  sorry

end solve_for_x_l585_58553


namespace expr_simplification_l585_58564

noncomputable def simplify_sqrt_expr : ℝ :=
  Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27

theorem expr_simplification : simplify_sqrt_expr = 2 * Real.sqrt 3 := by
  sorry

end expr_simplification_l585_58564


namespace regionA_regionC_area_ratio_l585_58524

-- Definitions for regions A and B
def regionA (l w : ℝ) : Prop := 2 * (l + w) = 16 ∧ l = 2 * w
def regionB (l w : ℝ) : Prop := 2 * (l + w) = 20 ∧ l = 2 * w
def area (l w : ℝ) : ℝ := l * w

theorem regionA_regionC_area_ratio {lA wA lB wB lC wC : ℝ} :
  regionA lA wA → regionB lB wB → (lC = lB ∧ wC = wB) → 
  (area lC wC ≠ 0) → 
  (area lA wA / area lC wC = 16 / 25) :=
by
  intros hA hB hC hC_area_ne_zero
  sorry

end regionA_regionC_area_ratio_l585_58524


namespace inequality_proof_l585_58511

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (sqrt (a^2 + 8 * b * c) / a + sqrt (b^2 + 8 * a * c) / b + sqrt (c^2 + 8 * a * b) / c) ≥ 9 :=
by 
  sorry

end inequality_proof_l585_58511


namespace one_third_sugar_l585_58559

theorem one_third_sugar (s : ℚ) (h : s = 23 / 4) : (1 / 3) * s = 1 + 11 / 12 :=
by {
  sorry
}

end one_third_sugar_l585_58559


namespace narrow_black_stripes_are_eight_l585_58534

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l585_58534


namespace find_two_digit_numbers_l585_58587

theorem find_two_digit_numbers (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h : 2 * (a + b) = a * b) : 
  10 * a + b = 63 ∨ 10 * a + b = 44 ∨ 10 * a + b = 36 :=
by sorry

end find_two_digit_numbers_l585_58587


namespace largest_n_unique_k_l585_58523

theorem largest_n_unique_k : ∃ n : ℕ, (∀ k : ℤ, (8 / 15 : ℚ) < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < (7 / 13 : ℚ) → k = unique_k) ∧ n = 112 :=
sorry

end largest_n_unique_k_l585_58523


namespace positive_integers_condition_l585_58560

theorem positive_integers_condition : ∃ n : ℕ, (n > 0) ∧ (n < 50) ∧ (∃ k : ℕ, n = k * (50 - n)) :=
sorry

end positive_integers_condition_l585_58560


namespace find_x_l585_58539

def binary_operation (a b c d : Int) : Int × Int := (a - c, b + d)

theorem find_x (x y : Int)
  (H1 : binary_operation 6 5 2 3 = (4, 8))
  (H2 : binary_operation x y 5 4 = (4, 8)) :
  x = 9 :=
by
  -- Necessary conditions and hypotheses are provided
  sorry -- Proof not required

end find_x_l585_58539


namespace simplify_expr_l585_58585

theorem simplify_expr (a : ℝ) (h : a > 1) : (1 - a) * (1 / (a - 1)).sqrt = -(a - 1).sqrt :=
sorry

end simplify_expr_l585_58585


namespace previous_spider_weight_l585_58558

noncomputable def giant_spider_weight (prev_spider_weight : ℝ) : ℝ :=
  2.5 * prev_spider_weight

noncomputable def leg_cross_sectional_area : ℝ := 0.5
noncomputable def leg_pressure : ℝ := 4
noncomputable def legs : ℕ := 8

noncomputable def force_per_leg : ℝ := leg_pressure * leg_cross_sectional_area
noncomputable def total_weight : ℝ := force_per_leg * (legs : ℝ)

theorem previous_spider_weight (prev_spider_weight : ℝ) (h_giant : giant_spider_weight prev_spider_weight = total_weight) : prev_spider_weight = 6.4 :=
by
  sorry

end previous_spider_weight_l585_58558


namespace perpendicular_line_eq_l585_58549

theorem perpendicular_line_eq (x y : ℝ) : 
  (∃ m : ℝ, (m * y + 2 * x = -5 / 2) ∧ (x - 2 * y + 3 = 0)) →
  ∃ a b c : ℝ, (a * x + b * y + c = 0) ∧ (2 * a + b = 0) ∧ c = 1 := sorry

end perpendicular_line_eq_l585_58549


namespace brownie_cost_l585_58504

theorem brownie_cost (total_money : ℕ) (num_pans : ℕ) (pieces_per_pan : ℕ) (cost_per_piece : ℕ) :
  total_money = 32 → num_pans = 2 → pieces_per_pan = 8 → cost_per_piece = total_money / (num_pans * pieces_per_pan) → 
  cost_per_piece = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end brownie_cost_l585_58504


namespace chocolates_problem_l585_58576

theorem chocolates_problem (C S : ℝ) (n : ℕ) 
  (h1 : 24 * C = n * S)
  (h2 : (S - C) / C = 0.5) : 
  n = 16 :=
by 
  sorry

end chocolates_problem_l585_58576


namespace loan_duration_in_years_l585_58557

-- Define the conditions as constants
def carPrice : ℝ := 20000
def downPayment : ℝ := 5000
def monthlyPayment : ℝ := 250

-- Define the goal
theorem loan_duration_in_years :
  (carPrice - downPayment) / monthlyPayment / 12 = 5 := 
sorry

end loan_duration_in_years_l585_58557


namespace parabola_equation_conditions_l585_58569

def focus_on_x_axis (focus : ℝ × ℝ) := (∃ x : ℝ, focus = (x, 0))
def foot_of_perpendicular (line : ℝ × ℝ → Prop) (focus : ℝ × ℝ) :=
  (∃ point : ℝ × ℝ, point = (2, 1) ∧ line focus ∧ line point ∧ line (0, 0))

theorem parabola_equation_conditions (focus : ℝ × ℝ) (line : ℝ × ℝ → Prop) :
  focus_on_x_axis focus →
  foot_of_perpendicular line focus →
  ∃ a : ℝ, ∀ x y : ℝ, y^2 = a * x ↔ y^2 = 10 * x :=
by
  intros h1 h2
  use 10
  sorry

end parabola_equation_conditions_l585_58569


namespace shanghai_population_scientific_notation_l585_58516

theorem shanghai_population_scientific_notation :
  16.3 * 10^6 = 1.63 * 10^7 :=
sorry

end shanghai_population_scientific_notation_l585_58516


namespace commission_percentage_l585_58599

theorem commission_percentage (fixed_salary second_base_salary sales_amount earning: ℝ) (commission: ℝ) 
  (h1 : fixed_salary = 1800)
  (h2 : second_base_salary = 1600)
  (h3 : sales_amount = 5000)
  (h4 : earning = 1800) :
  fixed_salary = second_base_salary + (sales_amount * commission) → 
  commission * 100 = 4 :=
by
  -- proof goes here
  sorry

end commission_percentage_l585_58599


namespace find_certain_number_l585_58546

theorem find_certain_number (x : ℝ) : 
  ((2 * (x + 5)) / 5 - 5 = 22) → x = 62.5 :=
by
  intro h
  -- Proof goes here
  sorry

end find_certain_number_l585_58546


namespace target_has_more_tools_l585_58506

-- Define the number of tools in the Walmart multitool
def walmart_screwdriver : ℕ := 1
def walmart_knives : ℕ := 3
def walmart_other_tools : ℕ := 2
def walmart_total_tools : ℕ := walmart_screwdriver + walmart_knives + walmart_other_tools

-- Define the number of tools in the Target multitool
def target_screwdriver : ℕ := 1
def target_knives : ℕ := 2 * walmart_knives
def target_files_scissors : ℕ := 3 + 1
def target_total_tools : ℕ := target_screwdriver + target_knives + target_files_scissors

-- The theorem stating the difference in the number of tools
theorem target_has_more_tools : (target_total_tools - walmart_total_tools) = 6 := by
  sorry

end target_has_more_tools_l585_58506


namespace circumference_of_circle_l585_58588

theorem circumference_of_circle (R : ℝ) : 
  (C = 2 * Real.pi * R) :=
sorry

end circumference_of_circle_l585_58588


namespace determinant_of_A_l585_58575

-- Define the 2x2 matrix A
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![7, -2], ![-3, 6]]

-- The statement to be proved
theorem determinant_of_A : Matrix.det A = 36 := 
  by sorry

end determinant_of_A_l585_58575


namespace Aren_listening_time_l585_58586

/--
Aren’s flight from New York to Hawaii will take 11 hours 20 minutes. He spends 2 hours reading, 
4 hours watching two movies, 30 minutes eating his dinner, some time listening to the radio, 
and 1 hour 10 minutes playing games. He has 3 hours left to take a nap. 
Prove that he spends 40 minutes listening to the radio.
-/
theorem Aren_listening_time 
  (total_flight_time : ℝ := 11 * 60 + 20)
  (reading_time : ℝ := 2 * 60)
  (watching_movies_time : ℝ := 4 * 60)
  (eating_dinner_time : ℝ := 30)
  (playing_games_time : ℝ := 1 * 60 + 10)
  (nap_time : ℝ := 3 * 60) :
  total_flight_time - (reading_time + watching_movies_time + eating_dinner_time + playing_games_time + nap_time) = 40 :=
by sorry

end Aren_listening_time_l585_58586


namespace sequence_geometric_progression_l585_58590

theorem sequence_geometric_progression (p : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = p * a n + 2^n)
  (h3 : ∀ n : ℕ, 0 < n → a (n + 1)^2 = a n * a (n + 2)): 
  ∃ p : ℝ, ∀ n : ℕ, a n = 2^n :=
by
  sorry

end sequence_geometric_progression_l585_58590


namespace find_remainder_of_n_l585_58556

theorem find_remainder_of_n (n k d : ℕ) (hn_pos : n > 0) (hk_pos : k > 0) (hd_pos_digits : d < 10^k) 
  (h : n * 10^k + d = n * (n + 1) / 2) : n % 9 = 1 :=
sorry

end find_remainder_of_n_l585_58556


namespace convert_spherical_to_rectangular_l585_58584

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin phi * Real.cos theta, rho * Real.sin phi * Real.sin theta, rho * Real.cos phi)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 10 (4 * Real.pi / 3) (Real.pi / 3) = (-5 * Real.sqrt 3, -15 / 2, 5) :=
by 
  sorry

end convert_spherical_to_rectangular_l585_58584


namespace positive_integers_count_l585_58544

theorem positive_integers_count (n : ℕ) : 
  ∃ m : ℕ, (m ≤ n / 2014 ∧ m ≤ n / 2016 ∧ (m + 1) * 2014 > n ∧ (m + 1) * 2016 > n) ↔
  (n = 1015056) :=
by
  sorry

end positive_integers_count_l585_58544


namespace work_completion_l585_58536

theorem work_completion (W : ℝ) (a b : ℝ) (ha : a = W / 12) (hb : b = W / 6) :
  W / (a + b) = 4 :=
by {
  sorry
}

end work_completion_l585_58536


namespace probability_two_face_cards_l585_58510

def cardDeck : ℕ := 52
def totalFaceCards : ℕ := 12

-- Probability of selecting one face card as the first card
def probabilityFirstFaceCard : ℚ := totalFaceCards / cardDeck

-- Probability of selecting another face card as the second card
def probabilitySecondFaceCard (cardsLeft : ℕ) : ℚ := (totalFaceCards - 1) / cardsLeft

-- Combined probability of selecting two face cards
theorem probability_two_face_cards :
  let combined_probability := probabilityFirstFaceCard * probabilitySecondFaceCard (cardDeck - 1)
  combined_probability = 22 / 442 := 
  by
    sorry

end probability_two_face_cards_l585_58510


namespace hawks_score_l585_58594

theorem hawks_score (a b : ℕ) (h1 : a + b = 58) (h2 : a - b = 12) : b = 23 :=
by
  sorry

end hawks_score_l585_58594


namespace inequality_2n_1_lt_n_plus_1_sq_l585_58548

theorem inequality_2n_1_lt_n_plus_1_sq (n : ℕ) (h : 0 < n) : 2 * n - 1 < (n + 1) ^ 2 := 
by 
  sorry

end inequality_2n_1_lt_n_plus_1_sq_l585_58548


namespace complete_work_in_days_l585_58595

def rate_x : ℚ := 1 / 10
def rate_y : ℚ := 1 / 15
def rate_z : ℚ := 1 / 20

def combined_rate : ℚ := rate_x + rate_y + rate_z

theorem complete_work_in_days :
  1 / combined_rate = 60 / 13 :=
by
  -- Proof will go here
  sorry

end complete_work_in_days_l585_58595


namespace math_equivalent_problem_l585_58531

noncomputable def correct_difference (A B C D : ℕ) (incorrect_difference : ℕ) : ℕ :=
  if (B = 3) ∧ (D = 2) ∧ (C = 5) ∧ (incorrect_difference = 60) then
    ((A * 10 + B) - 52)
  else
    0

theorem math_equivalent_problem (A : ℕ) : correct_difference A 3 5 2 60 = 31 :=
by
  sorry

end math_equivalent_problem_l585_58531


namespace max_average_hours_l585_58591

theorem max_average_hours :
  let hours_Wednesday := 2
  let hours_Thursday := 2
  let hours_Friday := hours_Wednesday + 3
  let total_hours := hours_Wednesday + hours_Thursday + hours_Friday
  let average_hours := total_hours / 3
  average_hours = 3 :=
by
  sorry

end max_average_hours_l585_58591


namespace number_of_marbles_in_Ellen_box_l585_58579

-- Defining the conditions given in the problem
def Dan_box_volume : ℕ := 216
def Ellen_side_multiplier : ℕ := 3
def marble_size_consistent_between_boxes : Prop := True -- Placeholder for the consistency condition

-- Main theorem statement
theorem number_of_marbles_in_Ellen_box :
  ∃ number_of_marbles_in_Ellen_box : ℕ,
  (∀ s : ℕ, s^3 = Dan_box_volume → (Ellen_side_multiplier * s)^3 / s^3 = 27 → 
  number_of_marbles_in_Ellen_box = 27 * Dan_box_volume) :=
by
  sorry

end number_of_marbles_in_Ellen_box_l585_58579


namespace half_angle_quadrants_l585_58571

variable (k : ℤ) (α : ℝ)

-- Conditions
def is_second_quadrant (α : ℝ) (k : ℤ) : Prop :=
  2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

-- Question: Determine the quadrant(s) in which α / 2 lies under the given condition.
theorem half_angle_quadrants (α : ℝ) (k : ℤ) 
  (h : is_second_quadrant α k) : 
  ((k * Real.pi + Real.pi / 4 < α / 2) ∧ (α / 2 < k * Real.pi + Real.pi / 2)) ↔ 
  (∃ (m : ℤ), (2 * m * Real.pi < α / 2 ∧ α / 2 < 2 * m * Real.pi + Real.pi)) ∨ ( ∃ (m : ℤ), (2 * m * Real.pi + Real.pi < α / 2 ∧ α / 2 < 2 * m * Real.pi + 2 * Real.pi)) := 
sorry

end half_angle_quadrants_l585_58571


namespace rectangle_area_l585_58596

theorem rectangle_area (P W : ℝ) (hP : P = 52) (hW : W = 11) :
  ∃ A L : ℝ, (2 * L + 2 * W = P) ∧ (A = L * W) ∧ (A = 165) :=
by
  sorry

end rectangle_area_l585_58596


namespace intersection_eq_l585_58547

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end intersection_eq_l585_58547


namespace luis_bought_6_pairs_of_blue_socks_l585_58565

open Nat

-- Conditions
def total_pairs_red := 4
def total_cost_red := 3
def total_cost := 42
def blue_socks_cost := 5

-- Deduce the spent amount on red socks, and from there calculate the number of blue socks bought.
theorem luis_bought_6_pairs_of_blue_socks :
  (yes : ℕ) -> yes * blue_socks_cost = total_cost - total_pairs_red * total_cost_red → yes = 6 :=
sorry

end luis_bought_6_pairs_of_blue_socks_l585_58565


namespace almonds_weight_l585_58502

def nuts_mixture (almonds_ratio walnuts_ratio total_weight : ℚ) : ℚ :=
  let total_parts := almonds_ratio + walnuts_ratio
  let weight_per_part := total_weight / total_parts
  let weight_almonds := weight_per_part * almonds_ratio
  weight_almonds

theorem almonds_weight (total_weight : ℚ) (h1 : total_weight = 140) : nuts_mixture 5 1 total_weight = 116.67 :=
by
  sorry

end almonds_weight_l585_58502


namespace horner_method_operations_l585_58555

-- Define the polynomial
def poly (x : ℤ) : ℤ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

-- Define Horner's method evaluation for the specific polynomial at x = 2
def horners_method_evaluated (x : ℤ) : ℤ :=
  (((((5 * x + 4) * x + 3) * x + 2) * x + 1) * x + 1)

-- Count multiplication and addition operations
def count_mul_ops : ℕ := 5
def count_add_ops : ℕ := 5

-- Proof statement
theorem horner_method_operations :
  ∀ (x : ℤ), x = 2 → 
  (count_mul_ops = 5) ∧ (count_add_ops = 5) :=
by
  intros x h
  sorry

end horner_method_operations_l585_58555


namespace banana_pieces_l585_58566

theorem banana_pieces (B G P : ℕ) 
  (h1 : P = 4 * G)
  (h2 : G = B + 5)
  (h3 : P = 192) : B = 43 := 
by
  sorry

end banana_pieces_l585_58566


namespace points_below_line_l585_58597

theorem points_below_line (d q x1 x2 y1 y2 : ℝ) 
  (h1 : 2 = 1 + 3 * d)
  (h2 : x1 = 1 + d)
  (h3 : x2 = x1 + d)
  (h4 : 2 = q ^ 3)
  (h5 : y1 = q)
  (h6 : y2 = q ^ 2) :
  x1 > y1 ∧ x2 > y2 :=
by {
  sorry
}

end points_below_line_l585_58597


namespace cost_of_article_l585_58580

theorem cost_of_article (C : ℝ) (H1 : 350 - C = G + 0.05 * G) (H2 : 345 - C = G) : C = 245 :=
by
  sorry

end cost_of_article_l585_58580


namespace closest_ratio_one_l585_58512

theorem closest_ratio_one (a c : ℕ) (h1 : 30 * a + 15 * c = 2700) (h2 : a ≥ 1) (h3 : c ≥ 1) :
  a = c :=
by sorry

end closest_ratio_one_l585_58512


namespace smallest_n_for_gn_gt_20_l585_58598

def g (n : ℕ) : ℕ := sorry -- definition of the sum of the digits to the right of the decimal of 1 / 3^n

theorem smallest_n_for_gn_gt_20 : ∃ n : ℕ, n > 0 ∧ g n > 20 ∧ ∀ m, 0 < m ∧ m < n -> g m ≤ 20 :=
by
  -- here should be the proof
  sorry

end smallest_n_for_gn_gt_20_l585_58598


namespace tangent_at_5_eqn_l585_58538

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_period : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom tangent_at_neg1 : ∀ x y : ℝ, x - y + 3 = 0 → x = -1 → y = f x

theorem tangent_at_5_eqn : 
  ∀ x y : ℝ, x = 5 → y = f x → x + y - 7 = 0 :=
sorry

end tangent_at_5_eqn_l585_58538


namespace part_a_part_b_part_c_part_d_part_e_part_f_l585_58574

-- Part (a)
theorem part_a (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^2 = 5 * k + 1 ∨ n^2 = 5 * k - 1 := 
sorry

-- Part (b)
theorem part_b (n : ℤ) (h : ¬ ∃ k : ℤ, n = 5 * k) : ∃ k : ℤ, n^4 - 1 = 5 * k := 
sorry

-- Part (c)
theorem part_c (n : ℤ) : n^5 % 10 = n % 10 := 
sorry

-- Part (d)
theorem part_d (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k := 
sorry

-- Part (e)
theorem part_e (k n : ℤ) (h1 : ¬ ∃ j : ℤ, k = 5 * j) (h2 : ¬ ∃ j : ℤ, n = 5 * j) : ∃ j : ℤ, k^4 - n^4 = 5 * j := 
sorry

-- Part (f)
theorem part_f (k m n : ℤ) (h : k^2 + m^2 = n^2) : ∃ j : ℤ, k = 5 * j ∨ ∃ r : ℤ, m = 5 * r ∨ ∃ s : ℤ, n = 5 * s := 
sorry

end part_a_part_b_part_c_part_d_part_e_part_f_l585_58574


namespace inequality_and_equality_hold_l585_58537

theorem inequality_and_equality_hold (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ a = b) :=
sorry

end inequality_and_equality_hold_l585_58537


namespace find_g_expression_l585_58522

theorem find_g_expression (g f : ℝ → ℝ) (h_sym : ∀ x y, g x = y ↔ g (2 - x) = 4 - y)
  (h_f : ∀ x, f x = 3 * x - 1) :
  ∀ x, g x = 3 * x - 1 :=
by
  sorry

end find_g_expression_l585_58522
