import Mathlib

namespace NUMINAMATH_GPT_average_age_of_boys_l69_6965

def boys_age_proportions := (3, 5, 7)
def eldest_boy_age := 21

theorem average_age_of_boys : 
  ∃ (x : ℕ), 7 * x = eldest_boy_age ∧ (3 * x + 5 * x + 7 * x) / 3 = 15 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_boys_l69_6965


namespace NUMINAMATH_GPT_greatest_possible_value_of_a_l69_6977

theorem greatest_possible_value_of_a :
  ∃ a : ℕ, (∀ x : ℤ, x * (x + a) = -12) → a = 13 := by
  sorry

end NUMINAMATH_GPT_greatest_possible_value_of_a_l69_6977


namespace NUMINAMATH_GPT_spending_on_hydrangeas_l69_6902

def lily_spending : ℕ :=
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  let years := end_year - start_year
  cost_per_plant * years

theorem spending_on_hydrangeas : lily_spending = 640 := 
  sorry

end NUMINAMATH_GPT_spending_on_hydrangeas_l69_6902


namespace NUMINAMATH_GPT_cans_needed_eq_l69_6967

axiom Paula_initial_rooms : ℕ
axiom Paula_lost_cans : ℕ
axiom Paula_after_loss_rooms : ℕ
axiom cans_for_25_rooms : ℕ

theorem cans_needed_eq :
  Paula_initial_rooms = 30 →
  Paula_lost_cans = 3 →
  Paula_after_loss_rooms = 25 →
  cans_for_25_rooms = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cans_needed_eq_l69_6967


namespace NUMINAMATH_GPT_probability_of_same_color_balls_l69_6943

-- Definitions of the problem
def total_balls_bag_A := 8 + 4
def total_balls_bag_B := 6 + 6
def white_balls_bag_A := 8
def red_balls_bag_A := 4
def white_balls_bag_B := 6
def red_balls_bag_B := 6

def P (event: Nat -> Bool) (total: Nat) : Nat :=
  let favorable := (List.range total).filter event |>.length
  favorable / total

-- Probability of drawing a white ball from bag A
def P_A := P (λ n => n < white_balls_bag_A) total_balls_bag_A

-- Probability of drawing a red ball from bag A
def P_not_A := P (λ n => n >= white_balls_bag_A && n < total_balls_bag_A) total_balls_bag_A

-- Probability of drawing a white ball from bag B
def P_B := P (λ n => n < white_balls_bag_B) total_balls_bag_B

-- Probability of drawing a red ball from bag B
def P_not_B := P (λ n => n >= white_balls_bag_B && n < total_balls_bag_B) total_balls_bag_B

-- Independence assumption (product rule for independent events)
noncomputable def P_same_color := P_A * P_B + P_not_A * P_not_B

-- Final theorem to prove
theorem probability_of_same_color_balls :
  P_same_color = 1 / 2 := by
    sorry

end NUMINAMATH_GPT_probability_of_same_color_balls_l69_6943


namespace NUMINAMATH_GPT_percentage_basketball_l69_6934

theorem percentage_basketball (total_students : ℕ) (chess_percentage : ℝ) (students_like_chess_basketball : ℕ) 
  (percentage_conversion : ∀ p : ℝ, 0 ≤ p → p / 100 = p) 
  (h_total : total_students = 250) 
  (h_chess : chess_percentage = 10) 
  (h_chess_basketball : students_like_chess_basketball = 125) :
  ∃ (basketball_percentage : ℝ), basketball_percentage = 40 := by
  sorry

end NUMINAMATH_GPT_percentage_basketball_l69_6934


namespace NUMINAMATH_GPT_find_principal_amount_l69_6992

def interest_rate_first_year : ℝ := 0.10
def compounding_periods_first_year : ℕ := 2
def interest_rate_second_year : ℝ := 0.12
def compounding_periods_second_year : ℕ := 4
def diff_interest : ℝ := 12

theorem find_principal_amount (P : ℝ)
  (h1_first : interest_rate_first_year / (compounding_periods_first_year : ℝ) = 0.05)
  (h1_second : interest_rate_second_year / (compounding_periods_second_year : ℝ) = 0.03)
  (compounded_amount : ℝ := P * (1 + 0.05)^(compounding_periods_first_year) * (1 + 0.03)^compounding_periods_second_year)
  (simple_interest : ℝ := P * (interest_rate_first_year + interest_rate_second_year) / 2 * 2)
  (h_diff : compounded_amount - P - simple_interest = diff_interest) : P = 597.01 :=
sorry

end NUMINAMATH_GPT_find_principal_amount_l69_6992


namespace NUMINAMATH_GPT_pie_shop_revenue_l69_6941

noncomputable def revenue_day1 := 5 * 6 * 12 + 6 * 6 * 8 + 7 * 6 * 10
noncomputable def revenue_day2 := 6 * 6 * 15 + 7 * 6 * 10 + 8 * 6 * 14
noncomputable def revenue_day3 := 4 * 6 * 18 + 7 * 6 * 7 + 9 * 6 * 13
noncomputable def total_revenue := revenue_day1 + revenue_day2 + revenue_day3

theorem pie_shop_revenue : total_revenue = 4128 := by
  sorry

end NUMINAMATH_GPT_pie_shop_revenue_l69_6941


namespace NUMINAMATH_GPT_difference_in_balances_l69_6991

/-- Define the parameters for Angela's and Bob's accounts --/
def P_A : ℕ := 5000  -- Angela's principal
def r_A : ℚ := 0.05  -- Angela's annual interest rate
def n_A : ℕ := 2  -- Compounding frequency for Angela
def t : ℕ := 15  -- Time in years

def P_B : ℕ := 7000  -- Bob's principal
def r_B : ℚ := 0.04  -- Bob's annual interest rate

/-- Computing the final amounts for Angela and Bob after 15 years --/
noncomputable def A_A : ℚ := P_A * ((1 + (r_A / n_A)) ^ (n_A * t))  -- Angela's final amount
noncomputable def A_B : ℚ := P_B * (1 + r_B * t)  -- Bob's final amount

/-- Proof statement: The difference in account balances to the nearest dollar --/
theorem difference_in_balances : abs (A_A - A_B) = 726 := by
  sorry

end NUMINAMATH_GPT_difference_in_balances_l69_6991


namespace NUMINAMATH_GPT_censusSurveys_l69_6920

-- Definitions corresponding to the problem conditions
inductive Survey where
  | TVLifespan
  | ManuscriptReview
  | PollutionInvestigation
  | StudentSizeSurvey

open Survey

-- The aim is to identify which surveys are more suitable for a census.
def suitableForCensus (s : Survey) : Prop :=
  match s with
  | TVLifespan => False  -- Lifespan destruction implies sample survey.
  | ManuscriptReview => True  -- Significant and needs high accuracy, thus census.
  | PollutionInvestigation => False  -- Broad scope implies sample survey.
  | StudentSizeSurvey => True  -- Manageable scope makes census appropriate.

-- The theorem to be formalized.
theorem censusSurveys : (suitableForCensus ManuscriptReview) ∧ (suitableForCensus StudentSizeSurvey) :=
  by sorry

end NUMINAMATH_GPT_censusSurveys_l69_6920


namespace NUMINAMATH_GPT_abc_over_ab_bc_ca_l69_6923

variable {a b c : ℝ}

theorem abc_over_ab_bc_ca (h1 : ab / (a + b) = 2)
                          (h2 : bc / (b + c) = 5)
                          (h3 : ca / (c + a) = 7) :
        abc / (ab + bc + ca) = 35 / 44 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_abc_over_ab_bc_ca_l69_6923


namespace NUMINAMATH_GPT_fg_eval_at_3_l69_6921

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_eval_at_3 : f (g 3) = 99 := by
  sorry

end NUMINAMATH_GPT_fg_eval_at_3_l69_6921


namespace NUMINAMATH_GPT_number_of_good_students_is_5_or_7_l69_6982

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_good_students_is_5_or_7_l69_6982


namespace NUMINAMATH_GPT_ben_bonus_leftover_l69_6966

theorem ben_bonus_leftover (b : ℝ) (k h c : ℝ) (bk : k = 1/22 * b) (bh : h = 1/4 * b) (bc : c = 1/8 * b) :
  b - (k + h + c) = 867 :=
by
  sorry

end NUMINAMATH_GPT_ben_bonus_leftover_l69_6966


namespace NUMINAMATH_GPT_inequality_of_products_l69_6924

theorem inequality_of_products
  (a b c d : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hd : 0 < d)
  (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_products_l69_6924


namespace NUMINAMATH_GPT_total_exercise_hours_l69_6985

-- Define the conditions
def Natasha_minutes_per_day : ℕ := 30
def Natasha_days : ℕ := 7
def Esteban_minutes_per_day : ℕ := 10
def Esteban_days : ℕ := 9
def Charlotte_monday_minutes : ℕ := 20
def Charlotte_wednesday_minutes : ℕ := 45
def Charlotte_thursday_minutes : ℕ := 30
def Charlotte_sunday_minutes : ℕ := 60

-- Sum up the minutes for each individual
def Natasha_total_minutes : ℕ := Natasha_minutes_per_day * Natasha_days
def Esteban_total_minutes : ℕ := Esteban_minutes_per_day * Esteban_days
def Charlotte_total_minutes : ℕ := Charlotte_monday_minutes + Charlotte_wednesday_minutes + Charlotte_thursday_minutes + Charlotte_sunday_minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (minutes : ℕ) : ℚ := minutes / 60

-- Calculation of hours for each individual
noncomputable def Natasha_total_hours : ℚ := minutes_to_hours Natasha_total_minutes
noncomputable def Esteban_total_hours : ℚ := minutes_to_hours Esteban_total_minutes
noncomputable def Charlotte_total_hours : ℚ := minutes_to_hours Charlotte_total_minutes

-- Prove total hours of exercise for all three individuals
theorem total_exercise_hours : Natasha_total_hours + Esteban_total_hours + Charlotte_total_hours = 7.5833 := by
  sorry

end NUMINAMATH_GPT_total_exercise_hours_l69_6985


namespace NUMINAMATH_GPT_lying_dwarf_number_is_possible_l69_6917

def dwarfs_sum (a1 a2 a3 a4 a5 a6 a7 : ℕ) : Prop :=
  a2 = a1 ∧
  a3 = a1 + a2 ∧
  a4 = a1 + a2 + a3 ∧
  a5 = a1 + a2 + a3 + a4 ∧
  a6 = a1 + a2 + a3 + a4 + a5 ∧
  a7 = a1 + a2 + a3 + a4 + a5 + a6 ∧
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = 58

theorem lying_dwarf_number_is_possible (a1 a2 a3 a4 a5 a6 a7 : ℕ) :
  dwarfs_sum a1 a2 a3 a4 a5 a6 a7 →
  (a1 = 13 ∨ a1 = 26) :=
sorry

end NUMINAMATH_GPT_lying_dwarf_number_is_possible_l69_6917


namespace NUMINAMATH_GPT_height_is_geometric_mean_of_bases_l69_6937

-- Given conditions
variables (a c m : ℝ)
-- we declare the condition that the given trapezoid is symmetric and tangential
variables (isSymmetricTangentialTrapezoid : Prop)

-- The theorem to be proven
theorem height_is_geometric_mean_of_bases 
(isSymmetricTangentialTrapezoid: isSymmetricTangentialTrapezoid) 
: m = Real.sqrt (a * c) :=
sorry

end NUMINAMATH_GPT_height_is_geometric_mean_of_bases_l69_6937


namespace NUMINAMATH_GPT_cadastral_value_of_land_l69_6989

theorem cadastral_value_of_land (tax_amount : ℝ) (tax_rate : ℝ) (V : ℝ)
    (h1 : tax_amount = 4500)
    (h2 : tax_rate = 0.003) :
    V = 1500000 :=
by
  sorry

end NUMINAMATH_GPT_cadastral_value_of_land_l69_6989


namespace NUMINAMATH_GPT_correct_operation_l69_6945

variable (a : ℝ)

theorem correct_operation :
  (2 * a^2 * a = 2 * a^3) ∧
  ((a + 1)^2 ≠ a^2 + 1) ∧
  ((a^2 / (2 * a)) ≠ 2 * a) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) :=
by
  { sorry }

end NUMINAMATH_GPT_correct_operation_l69_6945


namespace NUMINAMATH_GPT_complement_U_P_l69_6907

def U : Set ℝ := {y | ∃ x > 1, y = Real.log x / Real.log 2}
def P : Set ℝ := {y | ∃ x > 2, y = 1 / x}

theorem complement_U_P :
  (U \ P) = Set.Ici (1 / 2) := 
by
  sorry

end NUMINAMATH_GPT_complement_U_P_l69_6907


namespace NUMINAMATH_GPT_total_amount_shared_l69_6957

theorem total_amount_shared
  (A B C : ℕ)
  (h_ratio : A / 2 = B / 3 ∧ B / 3 = C / 8)
  (h_Ben_share : B = 30) : A + B + C = 130 :=
by
  -- Add placeholder for the proof.
  sorry

end NUMINAMATH_GPT_total_amount_shared_l69_6957


namespace NUMINAMATH_GPT_runners_meet_again_l69_6952

theorem runners_meet_again 
  (v1 v2 v3 v4 v5 : ℕ)
  (h1 : v1 = 32) 
  (h2 : v2 = 40) 
  (h3 : v3 = 48) 
  (h4 : v4 = 56) 
  (h5 : v5 = 64) 
  (h6 : 400 % (v2 - v1) = 0)
  (h7 : 400 % (v3 - v2) = 0)
  (h8 : 400 % (v4 - v3) = 0)
  (h9 : 400 % (v5 - v4) = 0) :
  ∃ t : ℕ, t = 500 :=
by sorry

end NUMINAMATH_GPT_runners_meet_again_l69_6952


namespace NUMINAMATH_GPT_trip_time_difference_l69_6980

theorem trip_time_difference
  (avg_speed : ℝ)
  (dist1 dist2 : ℝ)
  (h_avg_speed : avg_speed = 60)
  (h_dist1 : dist1 = 540)
  (h_dist2 : dist2 = 570) :
  ((dist2 - dist1) / avg_speed) * 60 = 30 := by
  sorry

end NUMINAMATH_GPT_trip_time_difference_l69_6980


namespace NUMINAMATH_GPT_circle_equation_through_points_l69_6960

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end NUMINAMATH_GPT_circle_equation_through_points_l69_6960


namespace NUMINAMATH_GPT_usual_time_of_train_l69_6964

theorem usual_time_of_train (S T : ℝ) (h_speed : S ≠ 0) 
(h_speed_ratio : ∀ (T' : ℝ), T' = T + 3/4 → S * T = (4/5) * S * T' → T = 3) : Prop :=
  T = 3

end NUMINAMATH_GPT_usual_time_of_train_l69_6964


namespace NUMINAMATH_GPT_chinese_medicine_excess_purchased_l69_6901

-- Define the conditions of the problem

def total_plan : ℕ := 1500

def first_half_percentage : ℝ := 0.55
def second_half_percentage : ℝ := 0.65

-- State the theorem to prove the amount purchased in excess
theorem chinese_medicine_excess_purchased :
    first_half_percentage * total_plan + second_half_percentage * total_plan - total_plan = 300 :=
by 
  sorry

end NUMINAMATH_GPT_chinese_medicine_excess_purchased_l69_6901


namespace NUMINAMATH_GPT_sufficient_not_necessary_of_and_false_or_true_l69_6981

variables (p q : Prop)

theorem sufficient_not_necessary_of_and_false_or_true :
  (¬(p ∧ q) → (p ∨ q)) ∧ ((p ∨ q) → ¬(¬(p ∧ q))) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_of_and_false_or_true_l69_6981


namespace NUMINAMATH_GPT_inequality_solution_set_l69_6969

theorem inequality_solution_set (x : ℝ) :
  ∀ x, 
  (x^2 * (x + 1) / (-x^2 - 5 * x + 6) <= 0) ↔ (-6 < x ∧ x <= -1) ∨ (x = 0) ∨ (1 < x) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l69_6969


namespace NUMINAMATH_GPT_complement_intersect_eq_l69_6905

-- Define Universal Set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define Set P
def P : Set ℕ := {2, 3, 4}

-- Define Set Q
def Q : Set ℕ := {1, 2}

-- Complement of P in U
def complement_U_P : Set ℕ := U \ P

-- Goal Statement
theorem complement_intersect_eq {U P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4}) 
  (hP : P = {2, 3, 4}) 
  (hQ : Q = {1, 2}) : 
  (complement_U_P ∩ Q) = {1} := 
by
  sorry

end NUMINAMATH_GPT_complement_intersect_eq_l69_6905


namespace NUMINAMATH_GPT_twentieth_number_l69_6997

-- Defining the conditions and goal
theorem twentieth_number :
  ∃ x : ℕ, x % 8 = 5 ∧ x % 3 = 2 ∧ (∃ n : ℕ, x = 5 + 24 * n) ∧ x = 461 := 
sorry

end NUMINAMATH_GPT_twentieth_number_l69_6997


namespace NUMINAMATH_GPT_attendees_received_all_items_l69_6940

theorem attendees_received_all_items {n : ℕ} (h1 : ∀ k, k ∣ 45 → n % k = 0) (h2 : ∀ k, k ∣ 75 → n % k = 0) (h3 : ∀ k, k ∣ 100 → n % k = 0) (h4 : n = 4500) :
  (4500 / Nat.lcm (Nat.lcm 45 75) 100) = 5 :=
by
  sorry

end NUMINAMATH_GPT_attendees_received_all_items_l69_6940


namespace NUMINAMATH_GPT_remainder_division_l69_6930

theorem remainder_division (G Q1 R1 Q2 : ℕ) (hG : G = 88)
  (h1 : 3815 = G * Q1 + R1) (h2 : 4521 = G * Q2 + 33) : R1 = 31 :=
sorry

end NUMINAMATH_GPT_remainder_division_l69_6930


namespace NUMINAMATH_GPT_find_xyz_l69_6978

open Complex

theorem find_xyz (a b c x y z : ℂ)
(h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0)
(h7 : a = (b + c) / (x - 3)) (h8 : b = (a + c) / (y - 3)) (h9 : c = (a + b) / (z - 3))
(h10 : x * y + x * z + y * z = 10) (h11 : x + y + z = 6) : 
(x * y * z = 15) :=
by
  sorry

end NUMINAMATH_GPT_find_xyz_l69_6978


namespace NUMINAMATH_GPT_salary_decrease_increase_l69_6979

theorem salary_decrease_increase (S : ℝ) (x : ℝ) (h : (S * (1 - x / 100) * (1 + x / 100) = 0.51 * S)) : x = 70 := 
by sorry

end NUMINAMATH_GPT_salary_decrease_increase_l69_6979


namespace NUMINAMATH_GPT_calc_expression_l69_6911

theorem calc_expression :
  5 + 7 * (2 + (1 / 4 : ℝ)) = 20.75 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l69_6911


namespace NUMINAMATH_GPT_students_in_front_l69_6984

theorem students_in_front (total_students : ℕ) (students_behind : ℕ) (students_total : total_students = 25) (behind_Yuna : students_behind = 9) :
  (total_students - (students_behind + 1)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_students_in_front_l69_6984


namespace NUMINAMATH_GPT_radio_show_songs_duration_l69_6909

-- Definitions of the conditions
def hours_per_day := 3
def minutes_per_hour := 60
def talking_segments := 3
def talking_segment_duration := 10
def ad_breaks := 5
def ad_break_duration := 5

-- The main statement translating the conditions and questions to Lean
theorem radio_show_songs_duration :
  (hours_per_day * minutes_per_hour) - (talking_segments * talking_segment_duration + ad_breaks * ad_break_duration) = 125 := by
  sorry

end NUMINAMATH_GPT_radio_show_songs_duration_l69_6909


namespace NUMINAMATH_GPT_johns_trip_distance_is_160_l69_6990

noncomputable def total_distance (y : ℕ) : Prop :=
  y / 2 + 40 + y / 4 = y

theorem johns_trip_distance_is_160 : ∃ y : ℕ, total_distance y ∧ y = 160 :=
by
  use 160
  unfold total_distance
  sorry

end NUMINAMATH_GPT_johns_trip_distance_is_160_l69_6990


namespace NUMINAMATH_GPT_pages_needed_l69_6900

def cards_per_page : ℕ := 3
def new_cards : ℕ := 2
def old_cards : ℕ := 10

theorem pages_needed : (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end NUMINAMATH_GPT_pages_needed_l69_6900


namespace NUMINAMATH_GPT_opposite_sqrt_4_l69_6993

theorem opposite_sqrt_4 : - (Real.sqrt 4) = -2 := sorry

end NUMINAMATH_GPT_opposite_sqrt_4_l69_6993


namespace NUMINAMATH_GPT_sandy_correct_sums_l69_6951

theorem sandy_correct_sums
  (c i : ℕ)
  (h1 : c + i = 30)
  (h2 : 3 * c - 2 * i = 45) :
  c = 21 :=
by
  sorry

end NUMINAMATH_GPT_sandy_correct_sums_l69_6951


namespace NUMINAMATH_GPT_identical_functions_l69_6998

def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := (x^2 - 1)^3^(1/3)

theorem identical_functions : ∀ x : ℝ, f x = g x :=
by
  intro x
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_identical_functions_l69_6998


namespace NUMINAMATH_GPT_correct_proposition_is_D_l69_6975

-- Define the propositions
def propositionA : Prop :=
  (∀ x : ℝ, x^2 = 4 → x = 2 ∨ x = -2) → (∀ x : ℝ, (x ≠ 2 ∨ x ≠ -2) → x^2 ≠ 4)

def propositionB (p : Prop) : Prop :=
  (p → (∀ x : ℝ, x^2 - 2*x + 3 > 0)) → (¬p → (∃ x : ℝ, x^2 - 2*x + 3 < 0))

def propositionC : Prop :=
  ∀ (a b : ℝ) (n : ℕ), a > b → n > 0 → a^n > b^n

def p : Prop := ∀ x : ℝ, x^3 ≥ 0
def q : Prop := ∀ e : ℝ, e > 0 → e < 1
def propositionD := p ∧ q

-- The proof problem
theorem correct_proposition_is_D : propositionD :=
  sorry

end NUMINAMATH_GPT_correct_proposition_is_D_l69_6975


namespace NUMINAMATH_GPT_percentage_increase_in_radius_l69_6995

theorem percentage_increase_in_radius (r R : ℝ) (h : π * R^2 = π * r^2 + 1.25 * (π * r^2)) :
  R = 1.5 * r :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_increase_in_radius_l69_6995


namespace NUMINAMATH_GPT_no_integer_solution_for_150_l69_6962

theorem no_integer_solution_for_150 : ∀ (x : ℤ), x - Int.sqrt x ≠ 150 := 
sorry

end NUMINAMATH_GPT_no_integer_solution_for_150_l69_6962


namespace NUMINAMATH_GPT_dogwood_trees_after_planting_l69_6986

-- Define the number of current dogwood trees and the number to be planted.
def current_dogwood_trees : ℕ := 34
def trees_to_be_planted : ℕ := 49

-- Problem statement to prove the total number of dogwood trees after planting.
theorem dogwood_trees_after_planting : current_dogwood_trees + trees_to_be_planted = 83 := by
  -- A placeholder for proof
  sorry

end NUMINAMATH_GPT_dogwood_trees_after_planting_l69_6986


namespace NUMINAMATH_GPT_brother_books_total_l69_6944

-- Define the conditions
def sarah_paperbacks : ℕ := 6
def sarah_hardbacks : ℕ := 4
def brother_paperbacks : ℕ := sarah_paperbacks / 3
def brother_hardbacks : ℕ := 2 * sarah_hardbacks

-- Define the statement to be proven
theorem brother_books_total : brother_paperbacks + brother_hardbacks = 10 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_brother_books_total_l69_6944


namespace NUMINAMATH_GPT_dorms_and_students_l69_6915

theorem dorms_and_students (x : ℕ) :
  (4 * x + 19) % 6 ≠ 0 → ∃ s : ℕ, (x = 10 ∧ s = 59) ∨ (x = 11 ∧ s = 63) ∨ (x = 12 ∧ s = 67) :=
by
  sorry

end NUMINAMATH_GPT_dorms_and_students_l69_6915


namespace NUMINAMATH_GPT_solve_for_n_l69_6936

theorem solve_for_n (n : ℚ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : 
    n = 3.5 :=
  sorry

end NUMINAMATH_GPT_solve_for_n_l69_6936


namespace NUMINAMATH_GPT_flour_vs_sugar_difference_l69_6955

-- Definitions based on the conditions
def flour_needed : ℕ := 10
def flour_added : ℕ := 7
def sugar_needed : ℕ := 2

-- Define the mathematical statement to prove
theorem flour_vs_sugar_difference :
  (flour_needed - flour_added) - sugar_needed = 1 :=
by
  sorry

end NUMINAMATH_GPT_flour_vs_sugar_difference_l69_6955


namespace NUMINAMATH_GPT_maximize_h_at_1_l69_6931

-- Definitions and conditions
def f (x : ℝ) : ℝ := -2 * x + 2
def g (x : ℝ) : ℝ := -3 * x + 6
def h (x : ℝ) : ℝ := f x * g x

-- The theorem to prove
theorem maximize_h_at_1 : (∀ x : ℝ, h x <= h 1) :=
sorry

end NUMINAMATH_GPT_maximize_h_at_1_l69_6931


namespace NUMINAMATH_GPT_solution_set_of_inequality_system_l69_6927

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_system_l69_6927


namespace NUMINAMATH_GPT_seats_per_bus_l69_6914

theorem seats_per_bus (students buses : ℕ) (h1 : students = 14) (h2 : buses = 7) : students / buses = 2 := by
  sorry

end NUMINAMATH_GPT_seats_per_bus_l69_6914


namespace NUMINAMATH_GPT_Carolyn_wants_to_embroider_l69_6970

theorem Carolyn_wants_to_embroider (s : ℕ) (f : ℕ) (u : ℕ) (g : ℕ) (n_f : ℕ) (t : ℕ) (number_of_unicorns : ℕ) :
  s = 4 ∧ f = 60 ∧ u = 180 ∧ g = 800 ∧ n_f = 50 ∧ t = 1085 ∧ 
  (t * s - (n_f * f) - g) / u = number_of_unicorns ↔ number_of_unicorns = 3 :=
by 
  sorry

end NUMINAMATH_GPT_Carolyn_wants_to_embroider_l69_6970


namespace NUMINAMATH_GPT_no_positive_integers_solution_l69_6928

theorem no_positive_integers_solution (m n : ℕ) (hm : m > 0) (hn : n > 0) : 4 * m * (m + 1) ≠ n * (n + 1) := 
by
  sorry

end NUMINAMATH_GPT_no_positive_integers_solution_l69_6928


namespace NUMINAMATH_GPT_range_of_k_l69_6968

theorem range_of_k (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → y^2 = 2 * x → (∃! (x₀ y₀ : ℝ), y₀ = k * x₀ + 1 ∧ y₀^2 = 2 * x₀)) ↔ 
  (k = 0 ∨ k ≥ 1/2) :=
sorry

end NUMINAMATH_GPT_range_of_k_l69_6968


namespace NUMINAMATH_GPT_solve_for_x_l69_6953

theorem solve_for_x :
  ∀ (x y : ℚ), (3 * x - 4 * y = 8) → (2 * x + 3 * y = 1) → x = 28 / 17 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_solve_for_x_l69_6953


namespace NUMINAMATH_GPT_harmonic_mean_closest_to_six_l69_6974

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_closest_to_six : 
     |harmonic_mean 3 2023 - 6| < 1 :=
sorry

end NUMINAMATH_GPT_harmonic_mean_closest_to_six_l69_6974


namespace NUMINAMATH_GPT_last_number_with_35_zeros_l69_6910

def count_zeros (n : Nat) : Nat :=
  if n = 0 then 1
  else if n < 10 then 0
  else count_zeros (n / 10) + count_zeros (n % 10)

def total_zeros_written (upto : Nat) : Nat :=
  (List.range (upto + 1)).foldl (λ acc n => acc + count_zeros n) 0

theorem last_number_with_35_zeros : ∃ n, total_zeros_written n = 35 ∧ ∀ m, m > n → total_zeros_written m ≠ 35 :=
by
  let x := 204
  have h1 : total_zeros_written x = 35 := sorry
  have h2 : ∀ m, m > x → total_zeros_written m ≠ 35 := sorry
  existsi x
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_last_number_with_35_zeros_l69_6910


namespace NUMINAMATH_GPT_shoes_count_l69_6916

def numberOfShoes (numPairs : Nat) (matchingPairProbability : ℚ) : Nat :=
  let S := numPairs * 2
  if (matchingPairProbability = 1 / (S - 1))
  then S
  else 0

theorem shoes_count 
(numPairs : Nat)
(matchingPairProbability : ℚ)
(hp : numPairs = 9)
(hq : matchingPairProbability = 0.058823529411764705) :
numberOfShoes numPairs matchingPairProbability = 18 := 
by
  -- definition only, the proof is not required
  sorry

end NUMINAMATH_GPT_shoes_count_l69_6916


namespace NUMINAMATH_GPT_maggie_goldfish_fraction_l69_6932

theorem maggie_goldfish_fraction :
  ∀ (x : ℕ), 3*x / 5 + 20 = x → (x / 100 : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_maggie_goldfish_fraction_l69_6932


namespace NUMINAMATH_GPT_find_m_range_l69_6976

noncomputable def p (m : ℝ) : Prop :=
  m < 1 / 3

noncomputable def q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem find_m_range (m : ℝ) :
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (1 / 3 ≤ m ∧ m < 15) :=
by
  sorry

end NUMINAMATH_GPT_find_m_range_l69_6976


namespace NUMINAMATH_GPT_max_value_expression_l69_6918

theorem max_value_expression (x k : ℕ) (h₀ : 0 < x) (h₁ : 0 < k) (y := k * x) : 
  (∀ x k : ℕ, 0 < x → 0 < k → y = k * x → ∃ m : ℝ, m = 2 ∧ 
    ∀ x k : ℕ, 0 < x → 0 < k → y = k * x → (x + y)^2 / (x^2 + y^2) ≤ 2) :=
sorry

end NUMINAMATH_GPT_max_value_expression_l69_6918


namespace NUMINAMATH_GPT_hazel_additional_days_l69_6929

theorem hazel_additional_days (school_year_days : ℕ) (miss_percent : ℝ) (already_missed : ℕ)
  (h1 : school_year_days = 180)
  (h2 : miss_percent = 0.05)
  (h3 : already_missed = 6) :
  (⌊miss_percent * school_year_days⌋ - already_missed) = 3 :=
by
  sorry

end NUMINAMATH_GPT_hazel_additional_days_l69_6929


namespace NUMINAMATH_GPT_plane_through_A_perpendicular_to_BC_l69_6913

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨-3, 6, 4⟩
def B : Point3D := ⟨8, -3, 5⟩
def C : Point3D := ⟨10, -3, 7⟩

-- Define the vector BC
def vectorBC (B C : Point3D) : Point3D :=
  ⟨C.x - B.x, C.y - B.y, C.z - B.z⟩

-- Equation of the plane
def planeEquation (p : Point3D) (n : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - p.x) + n.y * (y - p.y) + n.z * (z - p.z)

theorem plane_through_A_perpendicular_to_BC : 
  planeEquation A (vectorBC B C) x y z = 0 ↔ x + z - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_plane_through_A_perpendicular_to_BC_l69_6913


namespace NUMINAMATH_GPT_trig_identity_example_l69_6904

noncomputable def cos24 := Real.cos (24 * Real.pi / 180)
noncomputable def cos36 := Real.cos (36 * Real.pi / 180)
noncomputable def sin24 := Real.sin (24 * Real.pi / 180)
noncomputable def sin36 := Real.sin (36 * Real.pi / 180)
noncomputable def cos60 := Real.cos (60 * Real.pi / 180)

theorem trig_identity_example :
  cos24 * cos36 - sin24 * sin36 = cos60 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_example_l69_6904


namespace NUMINAMATH_GPT_correct_statement_c_l69_6949

-- Definitions
variables {Point : Type*} {Line Plane : Type*}
variables (l m : Line) (α β : Plane)

-- Conditions
def parallel_planes (α β : Plane) : Prop := sorry  -- α ∥ β
def perpendicular_line_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊥ α
def line_in_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊂ α
def line_perpendicular (l m : Line) : Prop := sorry  -- l ⊥ m

-- Theorem to be proven
theorem correct_statement_c 
  (α β : Plane) (l : Line)
  (h_parallel : parallel_planes α β)
  (h_perpendicular : perpendicular_line_plane l α) :
  ∀ (m : Line), line_in_plane m β → line_perpendicular m l := 
sorry

end NUMINAMATH_GPT_correct_statement_c_l69_6949


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l69_6988

theorem system1_solution (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := sorry

theorem system2_solution (x y : ℝ) (h1 : 3 * x - 5 * y = 9) (h2 : 2 * x + 3 * y = -6) : 
  x = -3 / 19 ∧ y = -36 / 19 := sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l69_6988


namespace NUMINAMATH_GPT_trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l69_6996

open Real

theorem trig_cos2_minus_sin2_eq_neg_sqrt5_div3 (α : ℝ) (hα1 : 0 < α ∧ α < π) (hα2 : sin α + cos α = sqrt 3 / 3) :
  cos α ^ 2 - sin α ^ 2 = - sqrt 5 / 3 := 
  sorry

end NUMINAMATH_GPT_trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l69_6996


namespace NUMINAMATH_GPT_component_unqualified_l69_6973

theorem component_unqualified :
  ∀ (φ : ℝ), (19.98 ≤ φ ∧ φ ≤ 20.02) → ¬(φ = 19.9) → True :=
by
  intro φ
  intro h
  intro h'
  -- skip proof
  sorry

end NUMINAMATH_GPT_component_unqualified_l69_6973


namespace NUMINAMATH_GPT_max_area_of_sector_l69_6959

variable (r l S : ℝ)

theorem max_area_of_sector (h_circumference : 2 * r + l = 8) (h_area : S = (1 / 2) * l * r) : 
  S ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_area_of_sector_l69_6959


namespace NUMINAMATH_GPT_paul_crayons_left_l69_6950

theorem paul_crayons_left (initial_crayons lost_crayons : ℕ) 
  (h_initial : initial_crayons = 253) 
  (h_lost : lost_crayons = 70) : (initial_crayons - lost_crayons) = 183 := 
by
  sorry

end NUMINAMATH_GPT_paul_crayons_left_l69_6950


namespace NUMINAMATH_GPT_abs_eq_case_solution_l69_6947

theorem abs_eq_case_solution :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| := sorry

end NUMINAMATH_GPT_abs_eq_case_solution_l69_6947


namespace NUMINAMATH_GPT_proof_Bill_age_is_24_l69_6933

noncomputable def Bill_is_24 (C : ℝ) (Bill_age : ℝ) (Daniel_age : ℝ) :=
  (Bill_age = 2 * C - 1) ∧ 
  (Daniel_age = C - 4) ∧ 
  (C + Bill_age + Daniel_age = 45) → 
  (Bill_age = 24)

theorem proof_Bill_age_is_24 (C Bill_age Daniel_age : ℝ) : 
  Bill_is_24 C Bill_age Daniel_age :=
by
  sorry

end NUMINAMATH_GPT_proof_Bill_age_is_24_l69_6933


namespace NUMINAMATH_GPT_increase_by_percentage_proof_l69_6956

def initial_number : ℕ := 150
def percentage_increase : ℝ := 0.4
def final_number : ℕ := 210

theorem increase_by_percentage_proof :
  initial_number + (percentage_increase * initial_number) = final_number :=
by
  sorry

end NUMINAMATH_GPT_increase_by_percentage_proof_l69_6956


namespace NUMINAMATH_GPT_find_k_intersecting_lines_l69_6983

theorem find_k_intersecting_lines : 
  ∃ (k : ℚ), (∃ (x y : ℚ), y = 6 * x + 4 ∧ y = -3 * x - 30 ∧ y = 4 * x + k) ∧ k = -32 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_k_intersecting_lines_l69_6983


namespace NUMINAMATH_GPT_sum_polynomials_l69_6922

def p (x : ℝ) : ℝ := 4 * x^2 - 2 * x + 1
def q (x : ℝ) : ℝ := -3 * x^2 + x - 5
def r (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem sum_polynomials (x : ℝ) : p x + q x + r x = 3 * x^2 - 5 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_polynomials_l69_6922


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l69_6971

def A := {x : ℝ | |x - 2| ≤ 1}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}
def C := {x : ℝ | 1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l69_6971


namespace NUMINAMATH_GPT_circle_equation_through_points_l69_6963

theorem circle_equation_through_points 
  (M N : ℝ × ℝ)
  (hM : M = (5, 2))
  (hN : N = (3, 2))
  (hk : ∃ k : ℝ, (M.1 + N.1) / 2 = k ∧ (M.2 + N.2) / 2 = (2 * k - 3))
  : (∃ h : ℝ, ∀ x y: ℝ, (x - 4) ^ 2 + (y - 5) ^ 2 = h) ∧ (∃ r : ℝ, r = 10) := 
sorry

end NUMINAMATH_GPT_circle_equation_through_points_l69_6963


namespace NUMINAMATH_GPT_range_of_m_l69_6919

theorem range_of_m (m : ℝ) (h : (2 - m) * (|m| - 3) < 0) : (-3 < m ∧ m < 2) ∨ (m > 3) :=
sorry

end NUMINAMATH_GPT_range_of_m_l69_6919


namespace NUMINAMATH_GPT_coral_third_week_pages_l69_6994

theorem coral_third_week_pages :
  let total_pages := 600
  let week1_read := total_pages / 2
  let remaining_after_week1 := total_pages - week1_read
  let week2_read := remaining_after_week1 * 0.30
  let remaining_after_week2 := remaining_after_week1 - week2_read
  remaining_after_week2 = 210 :=
by
  sorry

end NUMINAMATH_GPT_coral_third_week_pages_l69_6994


namespace NUMINAMATH_GPT_interval_intersection_l69_6926

theorem interval_intersection (x : ℝ) : 
  (1 < 4 * x ∧ 4 * x < 3) ∧ (2 < 6 * x ∧ 6 * x < 4) ↔ (1 / 3 < x ∧ x < 2 / 3) := 
by 
  sorry

end NUMINAMATH_GPT_interval_intersection_l69_6926


namespace NUMINAMATH_GPT_bus_passing_time_l69_6906

noncomputable def time_for_bus_to_pass (bus_length : ℝ) (bus_speed_kph : ℝ) (man_speed_kph : ℝ) : ℝ :=
  let relative_speed_kph := bus_speed_kph + man_speed_kph
  let relative_speed_mps := (relative_speed_kph * (1000/3600))
  bus_length / relative_speed_mps

theorem bus_passing_time :
  time_for_bus_to_pass 15 40 8 = 1.125 :=
by
  sorry

end NUMINAMATH_GPT_bus_passing_time_l69_6906


namespace NUMINAMATH_GPT_moles_of_NaOH_l69_6925

-- Statement of the problem conditions and desired conclusion
theorem moles_of_NaOH (moles_H2SO4 moles_NaHSO4 : ℕ) (h : moles_H2SO4 = 3) (h_eq : moles_H2SO4 = moles_NaHSO4) : moles_NaHSO4 = 3 := by
  sorry

end NUMINAMATH_GPT_moles_of_NaOH_l69_6925


namespace NUMINAMATH_GPT_good_arrangement_iff_coprime_l69_6972

-- Definitions for the concepts used
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_good_arrangement (n m : ℕ) : Prop :=
  ∃ k₀, ∀ i, (n * k₀ * i) % (m + n) = (i % (m + n))

theorem good_arrangement_iff_coprime (n m : ℕ) : is_good_arrangement n m ↔ is_coprime n m := 
sorry

end NUMINAMATH_GPT_good_arrangement_iff_coprime_l69_6972


namespace NUMINAMATH_GPT_expression_value_l69_6912

theorem expression_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (h₃ : (a^2 * b^2) / (a^4 - 2 * b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l69_6912


namespace NUMINAMATH_GPT_second_solution_sugar_percentage_l69_6939

theorem second_solution_sugar_percentage
  (initial_solution_pct : ℝ)
  (second_solution_pct : ℝ)
  (initial_solution_amount : ℝ)
  (final_solution_pct : ℝ)
  (replaced_fraction : ℝ)
  (final_amount : ℝ) :
  initial_solution_pct = 0.1 →
  final_solution_pct = 0.17 →
  replaced_fraction = 1/4 →
  initial_solution_amount = 100 →
  final_amount = 100 →
  second_solution_pct = 0.38 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_second_solution_sugar_percentage_l69_6939


namespace NUMINAMATH_GPT_Bridget_skittles_after_giving_l69_6946

-- Given conditions
def Bridget_initial_skittles : ℕ := 4
def Henry_skittles : ℕ := 4
def Henry_gives_all_to_Bridget : Prop := True

-- Prove that Bridget will have 8 Skittles in total after Henry gives all of his Skittles to her.
theorem Bridget_skittles_after_giving (h : Henry_gives_all_to_Bridget) :
  Bridget_initial_skittles + Henry_skittles = 8 :=
by
  sorry

end NUMINAMATH_GPT_Bridget_skittles_after_giving_l69_6946


namespace NUMINAMATH_GPT_problem_1_problem_2_l69_6958

-- Define proposition p
def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

-- Define proposition q
def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Define the range of values for a in proposition p
def range_p (a : ℝ) : Prop :=
  a ≤ 1

-- Define set A and set B
def set_A (a : ℝ) : Prop := a ≤ 1
def set_B (a : ℝ) : Prop := a ≥ 1 ∨ a ≤ -2

theorem problem_1 (a : ℝ) (h : proposition_p a) : range_p a := 
sorry

theorem problem_2 (a : ℝ) : 
  (∃ h1 : proposition_p a, set_A a) ∧ (∃ h2 : proposition_q a, set_B a)
  ↔ ¬ ((∃ h1 : proposition_p a, set_B a) ∧ (∃ h2 : proposition_q a, set_A a)) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l69_6958


namespace NUMINAMATH_GPT_range_of_a_l69_6954

theorem range_of_a (a : ℝ) :
  (∃ x, 0 < x ∧ x < 1 ∧ (a^2 * x - 2 * a + 1 = 0)) ↔ (a > 1/2 ∧ a ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l69_6954


namespace NUMINAMATH_GPT_katie_needs_more_sugar_l69_6908

-- Let total_cups be the total cups of sugar required according to the recipe
def total_cups : ℝ := 3

-- Let already_put_in be the cups of sugar Katie has already put in
def already_put_in : ℝ := 0.5

-- Define the amount of sugar Katie still needs to put in
def remaining_cups : ℝ := total_cups - already_put_in 

-- Prove that remaining_cups is 2.5
theorem katie_needs_more_sugar : remaining_cups = 2.5 := 
by 
  -- substitute total_cups and already_put_in
  dsimp [remaining_cups, total_cups, already_put_in]
  -- calculate the difference
  norm_num

end NUMINAMATH_GPT_katie_needs_more_sugar_l69_6908


namespace NUMINAMATH_GPT_susan_vacation_pay_missed_l69_6942

noncomputable def susan_weekly_pay (hours_worked : ℕ) : ℕ :=
  let regular_hours := min 40 hours_worked
  let overtime_hours := max (hours_worked - 40) 0
  15 * regular_hours + 20 * overtime_hours

noncomputable def susan_sunday_pay (num_sundays : ℕ) (hours_per_sunday : ℕ) : ℕ :=
  25 * num_sundays * hours_per_sunday

noncomputable def pay_without_sundays : ℕ :=
  susan_weekly_pay 48
    
noncomputable def total_three_week_pay : ℕ :=
  let weeks_normal_pay := 3 * pay_without_sundays
  let sunday_hours_1 := 1 * 8
  let sunday_hours_2 := 2 * 8
  let sunday_hours_3 := 0 * 8
  let sundays_total_pay := susan_sunday_pay 1 8 + susan_sunday_pay 2 8 + susan_sunday_pay 0 8
  weeks_normal_pay + sundays_total_pay
  
noncomputable def paid_vacation_pay : ℕ :=
  let paid_days := 6
  let paid_weeks_pay := susan_weekly_pay 40 + susan_weekly_pay (paid_days % 5 * 8)
  paid_weeks_pay

theorem susan_vacation_pay_missed :
  let missed_pay := total_three_week_pay - paid_vacation_pay
  missed_pay = 2160 := sorry

end NUMINAMATH_GPT_susan_vacation_pay_missed_l69_6942


namespace NUMINAMATH_GPT_decimal_digits_of_fraction_l69_6987

noncomputable def fraction : ℚ := 987654321 / (2 ^ 30 * 5 ^ 2)

theorem decimal_digits_of_fraction :
  ∃ n ≥ 30, fraction = (987654321 / 10^2) / 2^28 := sorry

end NUMINAMATH_GPT_decimal_digits_of_fraction_l69_6987


namespace NUMINAMATH_GPT_corporate_event_handshakes_l69_6948

def GroupHandshakes (A B C : Nat) (knows_all_A : Nat) (knows_none : Nat) (C_knows_none : Nat) : Nat :=
  -- Handshakes between Group A and Group B
  let handshakes_AB := knows_none * A
  -- Handshakes within Group B
  let handshakes_B := (knows_none * (knows_none - 1)) / 2
  -- Handshakes between Group B and Group C
  let handshakes_BC := B * C_knows_none
  -- Total handshakes
  handshakes_AB + handshakes_B + handshakes_BC

theorem corporate_event_handshakes : GroupHandshakes 15 20 5 5 15 = 430 :=
by
  sorry

end NUMINAMATH_GPT_corporate_event_handshakes_l69_6948


namespace NUMINAMATH_GPT_average_age_union_l69_6935

theorem average_age_union (students_A students_B students_C : ℕ)
  (sumA sumB sumC : ℕ) (avgA avgB avgC avgAB avgAC avgBC : ℚ)
  (hA : avgA = (sumA : ℚ) / students_A)
  (hB : avgB = (sumB : ℚ) / students_B)
  (hC : avgC = (sumC : ℚ) / students_C)
  (hAB : avgAB = (sumA + sumB) / (students_A + students_B))
  (hAC : avgAC = (sumA + sumC) / (students_A + students_C))
  (hBC : avgBC = (sumB + sumC) / (students_B + students_C))
  (h_avgA: avgA = 34)
  (h_avgB: avgB = 25)
  (h_avgC: avgC = 45)
  (h_avgAB: avgAB = 30)
  (h_avgAC: avgAC = 42)
  (h_avgBC: avgBC = 36) :
  (sumA + sumB + sumC : ℚ) / (students_A + students_B + students_C) = 33 := 
  sorry

end NUMINAMATH_GPT_average_age_union_l69_6935


namespace NUMINAMATH_GPT_lime_bottom_means_magenta_top_l69_6938

-- Define the colors as an enumeration for clarity
inductive Color
| Purple : Color
| Cyan : Color
| Magenta : Color
| Lime : Color
| Silver : Color
| Black : Color

open Color

-- Define the function representing the question
def opposite_top_face_given_bottom (bottom : Color) : Color :=
  match bottom with
  | Lime => Magenta
  | _ => Lime  -- For simplicity, we're only handling the Lime case as specified

-- State the theorem
theorem lime_bottom_means_magenta_top : 
  opposite_top_face_given_bottom Lime = Magenta :=
by
  -- This theorem states exactly what we need: if Lime is the bottom face, then Magenta is the top face.
  sorry

end NUMINAMATH_GPT_lime_bottom_means_magenta_top_l69_6938


namespace NUMINAMATH_GPT_price_of_basic_computer_l69_6999

-- Definitions for the prices
variables (C_b P M K C_e : ℝ)

-- Conditions
axiom h1 : C_b + P + M + K = 2500
axiom h2 : C_e + P + M + K = 3100
axiom h3 : P = (3100 / 6)
axiom h4 : M = (3100 / 5)
axiom h5 : K = (3100 / 8)
axiom h6 : C_e = C_b + 600

-- Theorem stating the price of the basic computer
theorem price_of_basic_computer : C_b = 975.83 :=
by {
  sorry
}

end NUMINAMATH_GPT_price_of_basic_computer_l69_6999


namespace NUMINAMATH_GPT_range_of_a_l69_6903

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ 2 * a * x + 4 = 0) ↔ (-2 ≤ a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l69_6903


namespace NUMINAMATH_GPT_complex_power_sum_l69_6961

open Complex

theorem complex_power_sum (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 + Complex.I * Real.sqrt 3 ∨ z^99 + z^100 + z^101 + z^102 + z^103 = 2 - Complex.I * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_complex_power_sum_l69_6961
