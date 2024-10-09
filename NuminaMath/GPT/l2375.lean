import Mathlib

namespace average_speed_l2375_237525

theorem average_speed (x y : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y)
  (total_time : x / 4 + y / 3 + y / 6 + x / 4 = 5) :
  (2 * (x + y)) / 5 = 4 :=
by
  sorry

end average_speed_l2375_237525


namespace negation_of_proposition_l2375_237551

theorem negation_of_proposition (a b : ℝ) : ¬ (a > b ∧ a - 1 > b - 1) ↔ a ≤ b ∨ a - 1 ≤ b - 1 :=
by sorry

end negation_of_proposition_l2375_237551


namespace trigonometric_inequality_equality_conditions_l2375_237578

theorem trigonometric_inequality
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) ≥ 9 :=
sorry

theorem equality_conditions
  (α β : ℝ)
  (hα : α = Real.arctan (Real.sqrt 2))
  (hβ : β = π / 4) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.sin β)^2 * (Real.cos β)^2)) = 9 :=
sorry

end trigonometric_inequality_equality_conditions_l2375_237578


namespace clock_angle_8_30_l2375_237598

theorem clock_angle_8_30 
  (angle_per_hour_mark : ℝ := 30)
  (angle_per_minute_mark : ℝ := 6)
  (hour_hand_angle_8 : ℝ := 8 * angle_per_hour_mark)
  (half_hour_movement : ℝ := 0.5 * angle_per_hour_mark)
  (hour_hand_angle_8_30 : ℝ := hour_hand_angle_8 + half_hour_movement)
  (minute_hand_angle_30 : ℝ := 30 * angle_per_minute_mark) :
  abs (hour_hand_angle_8_30 - minute_hand_angle_30) = 75 :=
by
  sorry

end clock_angle_8_30_l2375_237598


namespace joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l2375_237570

-- Defining the weights of Joe's lifts
variable (J1 J2 : ℕ)

-- Conditions for Joe
def joe_conditions : Prop :=
  (J1 + J2 = 900) ∧ (2 * J1 = J2 + 300)

-- Defining the weights of Mike's lifts
variable (M1 M2 : ℕ)

-- Conditions for Mike  
def mike_conditions : Prop :=
  (M1 + M2 = 1100) ∧ (M2 = M1 + 200)

-- Defining the weights of Lisa's lifts
variable (L1 L2 : ℕ)

-- Conditions for Lisa  
def lisa_conditions : Prop :=
  (L1 + L2 = 1000) ∧ (L1 = 3 * L2)

-- Proof statements
theorem joe_first_lift_is_400 (h : joe_conditions J1 J2) : J1 = 400 :=
by
  sorry

theorem mike_first_lift_is_450 (h : mike_conditions M1 M2) : M1 = 450 :=
by
  sorry

theorem lisa_second_lift_is_250 (h : lisa_conditions L1 L2) : L2 = 250 :=
by
  sorry

end joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l2375_237570


namespace john_votes_l2375_237550

theorem john_votes (J : ℝ) (total_votes : ℝ) (third_candidate_votes : ℝ) (james_votes : ℝ) 
  (h1 : total_votes = 1150) 
  (h2 : third_candidate_votes = J + 150) 
  (h3 : james_votes = 0.70 * (total_votes - J - third_candidate_votes)) 
  (h4 : total_votes = J + james_votes + third_candidate_votes) : 
  J = 500 := 
by 
  rw [h1, h2, h3] at h4 
  sorry

end john_votes_l2375_237550


namespace num_cows_on_farm_l2375_237501

variables (D C S : ℕ)

def total_legs : ℕ := 8 * S + 2 * D + 4 * C
def total_heads : ℕ := D + C + S

theorem num_cows_on_farm
  (h1 : S = 2 * D)
  (h2 : total_legs D C S = 2 * total_heads D C S + 72)
  (h3 : D + C + S ≤ 40) :
  C = 30 :=
sorry

end num_cows_on_farm_l2375_237501


namespace complete_square_l2375_237533

theorem complete_square (x : ℝ) : (x^2 - 4*x + 2 = 0) → ((x - 2)^2 = 2) :=
by
  intro h
  sorry

end complete_square_l2375_237533


namespace find_number_of_dogs_l2375_237512

variables (D P S : ℕ)
theorem find_number_of_dogs (h1 : D = 2 * P) (h2 : P = 2 * S) (h3 : 4 * D + 4 * P + 2 * S = 510) :
  D = 60 := 
sorry

end find_number_of_dogs_l2375_237512


namespace how_many_fewer_girls_l2375_237552

def total_students : ℕ := 27
def girls : ℕ := 11
def boys : ℕ := total_students - girls
def fewer_girls_than_boys : ℕ := boys - girls

theorem how_many_fewer_girls :
  fewer_girls_than_boys = 5 :=
sorry

end how_many_fewer_girls_l2375_237552


namespace value_at_7_5_l2375_237595

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = -f x
axiom interval_condition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = x

theorem value_at_7_5 : f 7.5 = -0.5 := by
  sorry

end value_at_7_5_l2375_237595


namespace susan_mean_l2375_237509

def susan_scores : List ℝ := [87, 90, 95, 98, 100]

theorem susan_mean :
  (susan_scores.sum) / (susan_scores.length) = 94 := by
  sorry

end susan_mean_l2375_237509


namespace books_combination_l2375_237524

theorem books_combination :
  (Nat.choose 15 3) = 455 := 
sorry

end books_combination_l2375_237524


namespace inner_rectangle_length_l2375_237542

def inner_rect_width : ℕ := 2

def second_rect_area (x : ℕ) : ℕ := 6 * (x + 4)

def largest_rect_area (x : ℕ) : ℕ := 10 * (x + 8)

def shaded_area_1 (x : ℕ) : ℕ := second_rect_area x - 2 * x

def shaded_area_2 (x : ℕ) : ℕ := largest_rect_area x - second_rect_area x

def in_arithmetic_progression (a b c : ℕ) : Prop := b - a = c - b

theorem inner_rectangle_length (x : ℕ) :
  in_arithmetic_progression (2 * x) (shaded_area_1 x) (shaded_area_2 x) → x = 4 := by
  intros
  sorry

end inner_rectangle_length_l2375_237542


namespace closure_of_M_is_closed_interval_l2375_237572

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {a | a^2 - 2 * a > 0}

theorem closure_of_M_is_closed_interval :
  closure M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end closure_of_M_is_closed_interval_l2375_237572


namespace solve_equation_l2375_237544

theorem solve_equation :
  (∃ x : ℝ, (x^2 + 3*x + 5) / (x^2 + 5*x + 6) = x + 3) → (x = -1) :=
by
  sorry

end solve_equation_l2375_237544


namespace age_of_15th_student_l2375_237535

theorem age_of_15th_student
  (avg_age_15_students : ℕ)
  (total_students : ℕ)
  (avg_age_5_students : ℕ)
  (students_5 : ℕ)
  (avg_age_9_students : ℕ)
  (students_9 : ℕ)
  (total_age_15_students_eq : avg_age_15_students * total_students = 225)
  (total_age_5_students_eq : avg_age_5_students * students_5 = 70)
  (total_age_9_students_eq : avg_age_9_students * students_9 = 144) :
  (avg_age_15_students * total_students - (avg_age_5_students * students_5 + avg_age_9_students * students_9) = 11) :=
by
  sorry

end age_of_15th_student_l2375_237535


namespace ambulance_ride_cost_l2375_237554

-- Define the conditions as per the given problem.
def totalBill : ℝ := 5000
def medicationPercentage : ℝ := 0.5
def overnightStayPercentage : ℝ := 0.25
def foodCost : ℝ := 175

-- Define the question to be proved.
theorem ambulance_ride_cost :
  let medicationCost := totalBill * medicationPercentage
  let remainingAfterMedication := totalBill - medicationCost
  let overnightStayCost := remainingAfterMedication * overnightStayPercentage
  let remainingAfterOvernight := remainingAfterMedication - overnightStayCost
  let remainingAfterFood := remainingAfterOvernight - foodCost
  remainingAfterFood = 1700 :=
by
  -- Proof can be completed here
  sorry

end ambulance_ride_cost_l2375_237554


namespace most_people_can_attend_on_most_days_l2375_237522

-- Define the days of the week as a type
inductive Day
| Mon | Tues | Wed | Thurs | Fri

open Day

-- Define the availability of each person
def is_available (person : String) (day : Day) : Prop :=
  match person, day with
  | "Anna", Mon => False
  | "Anna", Wed => False
  | "Anna", Fri => False
  | "Bill", Tues => False
  | "Bill", Thurs => False
  | "Bill", Fri => False
  | "Carl", Mon => False
  | "Carl", Tues => False
  | "Carl", Thurs => False
  | "Diana", Wed => False
  | "Diana", Fri => False
  | _, _ => True

-- Prove the result
theorem most_people_can_attend_on_most_days :
  {d : Day | d ∈ [Mon, Tues, Wed]} = {d : Day | ∀p : String, is_available p d → p ∈ ["Bill", "Carl", "Diana"] ∨ p ∉ ["Anna", "Bill"]} :=
sorry

end most_people_can_attend_on_most_days_l2375_237522


namespace largest_n_rational_sqrt_l2375_237596

theorem largest_n_rational_sqrt : ∃ n : ℕ, 
  (∀ k l : ℤ, k = Int.natAbs (Int.sqrt (n - 100)) ∧ l = Int.natAbs (Int.sqrt (n + 100)) → 
  k + l = 100) ∧ 
  (n = 2501) :=
by
  sorry

end largest_n_rational_sqrt_l2375_237596


namespace time_per_step_l2375_237527

def apply_and_dry_time (total_time steps : ℕ) : ℕ :=
  total_time / steps

theorem time_per_step : apply_and_dry_time 120 6 = 20 := by
  -- Proof omitted
  sorry

end time_per_step_l2375_237527


namespace diff_of_cubes_divisible_by_9_l2375_237519

theorem diff_of_cubes_divisible_by_9 (a b : ℤ) : 9 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3) := 
sorry

end diff_of_cubes_divisible_by_9_l2375_237519


namespace arithmetic_sequence_function_positive_l2375_237539

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_function_positive
  {f : ℝ → ℝ} {a : ℕ → ℝ}
  (hf_odd : is_odd f)
  (hf_mono : is_monotonically_increasing f)
  (ha_arith : is_arithmetic_sequence a)
  (ha3_pos : a 3 > 0) : 
  f (a 1) + f (a 3) + f (a 5) > 0 := 
sorry

end arithmetic_sequence_function_positive_l2375_237539


namespace amount_a_put_in_correct_l2375_237507

noncomputable def amount_a_put_in (total_profit managing_fee total_received_by_a profit_remaining: ℝ) : ℝ :=
  let capital_b := 2500
  let a_receives_from_investment := total_received_by_a - managing_fee
  let profit_ratio := a_receives_from_investment / profit_remaining
  profit_ratio * capital_b

theorem amount_a_put_in_correct :
  amount_a_put_in 9600 960 6000 8640 = 3500 :=
by
  dsimp [amount_a_put_in]
  sorry

end amount_a_put_in_correct_l2375_237507


namespace exists_diff_shape_and_color_l2375_237567

variable (Pitcher : Type) 
variable (shape color : Pitcher → Prop)
variable (exists_diff_shape : ∃ (A B : Pitcher), shape A ≠ shape B)
variable (exists_diff_color : ∃ (A B : Pitcher), color A ≠ color B)

theorem exists_diff_shape_and_color : ∃ (A B : Pitcher), shape A ≠ shape B ∧ color A ≠ color B :=
  sorry

end exists_diff_shape_and_color_l2375_237567


namespace words_to_numbers_l2375_237503

def word_to_num (w : String) : Float := sorry

theorem words_to_numbers :
  word_to_num "fifty point zero zero one" = 50.001 ∧
  word_to_num "seventy-five point zero six" = 75.06 :=
by
  sorry

end words_to_numbers_l2375_237503


namespace choose_5_starters_including_twins_l2375_237514

def number_of_ways_choose_starters (total_players : ℕ) (members_in_lineup : ℕ) (twins1 twins2 : (ℕ × ℕ)) : ℕ :=
1834

theorem choose_5_starters_including_twins :
  number_of_ways_choose_starters 18 5 (1, 2) (3, 4) = 1834 :=
sorry

end choose_5_starters_including_twins_l2375_237514


namespace expression_value_l2375_237529

theorem expression_value :
  (2^1006 + 5^1007)^2 - (2^1006 - 5^1007)^2 = 40 * 10^1006 :=
by sorry

end expression_value_l2375_237529


namespace triangle_area_is_sqrt3_over_4_l2375_237566

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem triangle_area_is_sqrt3_over_4
  (a b c A B : ℝ)
  (h1 : A = Real.pi / 3)
  (h2 : b = 2 * a * Real.cos B)
  (h3 : c = 1)
  (h4 : B = Real.pi / 3)
  (h5 : a = 1)
  (h6 : b = 1) :
  area_of_triangle a b c A B (Real.pi - A - B) = Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_is_sqrt3_over_4_l2375_237566


namespace total_pupils_correct_l2375_237573

-- Definitions of the number of girls and boys in each school
def girlsA := 542
def boysA := 387
def girlsB := 713
def boysB := 489
def girlsC := 628
def boysC := 361

-- Total pupils in each school
def pupilsA := girlsA + boysA
def pupilsB := girlsB + boysB
def pupilsC := girlsC + boysC

-- Total pupils across all schools
def total_pupils := pupilsA + pupilsB + pupilsC

-- The proof statement (no proof provided, hence sorry)
theorem total_pupils_correct : total_pupils = 3120 := by sorry

end total_pupils_correct_l2375_237573


namespace cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l2375_237513

-- Defining the conditions
def cos_x_eq_one (x : ℝ) : Prop := Real.cos x = 1
def sin_x_eq_zero (x : ℝ) : Prop := Real.sin x = 0

-- Main theorem statement
theorem cos_x_is_necessary_but_not_sufficient_for_sin_x_zero (x : ℝ) : 
  (∀ x, cos_x_eq_one x → sin_x_eq_zero x) ∧ (∃ x, sin_x_eq_zero x ∧ ¬ cos_x_eq_one x) :=
by 
  sorry

end cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l2375_237513


namespace root_equality_l2375_237528

theorem root_equality (p q : ℝ) (h1 : 1 + p + q = (2 - 2 * q) / p) (h2 : 1 + p + q = (1 - p + q) / q) :
  p + q = 1 :=
sorry

end root_equality_l2375_237528


namespace reading_time_difference_l2375_237553

theorem reading_time_difference (xanthia_speed molly_speed book_length : ℕ)
  (hx : xanthia_speed = 120) (hm : molly_speed = 60) (hb : book_length = 300) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 150 :=
by
  -- We acknowledge the proof here would use the given values
  sorry

end reading_time_difference_l2375_237553


namespace difference_sum_even_odd_1000_l2375_237558

open Nat

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_sum_even_odd_1000 :
  sum_first_n_even 1000 - sum_first_n_odd 1000 = 1000 :=
by
  sorry

end difference_sum_even_odd_1000_l2375_237558


namespace least_number_of_cookies_l2375_237555

theorem least_number_of_cookies :
  ∃ x : ℕ, x % 6 = 4 ∧ x % 5 = 3 ∧ x % 8 = 6 ∧ x % 9 = 7 ∧ x = 208 :=
by
  sorry

end least_number_of_cookies_l2375_237555


namespace minimum_value_l2375_237506

theorem minimum_value (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
    (h_condition : (1 / a) + (1 / b) + (1 / c) = 9) : 
    a^3 * b^2 * c ≥ 64 / 729 :=
sorry

end minimum_value_l2375_237506


namespace Katrina_sold_in_morning_l2375_237576

theorem Katrina_sold_in_morning :
  ∃ M : ℕ, (120 - 57 - 16 - 11) = M := sorry

end Katrina_sold_in_morning_l2375_237576


namespace total_vegetarian_is_33_l2375_237562

-- Definitions of the quantities involved
def only_vegetarian : Nat := 19
def both_vegetarian_non_vegetarian : Nat := 12
def vegan_strictly_vegetarian : Nat := 3
def vegan_non_vegetarian : Nat := 2

-- The total number of people consuming vegetarian dishes
def total_vegetarian_consumers : Nat := only_vegetarian + both_vegetarian_non_vegetarian + vegan_non_vegetarian

-- Prove the number of people consuming vegetarian dishes
theorem total_vegetarian_is_33 :
  total_vegetarian_consumers = 33 :=
sorry

end total_vegetarian_is_33_l2375_237562


namespace cannot_determine_right_triangle_l2375_237538

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def two_angles_complementary (α β : ℝ) : Prop :=
  α + β = 90

def exterior_angle_is_right (γ : ℝ) : Prop :=
  γ = 90

theorem cannot_determine_right_triangle :
  ¬ (∃ (a b c : ℝ), a = 1 ∧ b = 1 ∧ c = 2 ∧ is_right_triangle a b c) :=
by sorry

end cannot_determine_right_triangle_l2375_237538


namespace units_produced_today_l2375_237586

theorem units_produced_today (n : ℕ) (P : ℕ) (T : ℕ) 
  (h1 : n = 14)
  (h2 : P = 60 * n)
  (h3 : (P + T) / (n + 1) = 62) : 
  T = 90 :=
by
  sorry

end units_produced_today_l2375_237586


namespace points_description_l2375_237582

noncomputable def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_description (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x + y = 0) := 
by 
  sorry

end points_description_l2375_237582


namespace downward_parabola_with_symmetry_l2375_237593

-- Define the general form of the problem conditions in Lean
theorem downward_parabola_with_symmetry (k : ℝ) :
  ∃ a : ℝ, a < 0 ∧ ∃ h : ℝ, h = 3 ∧ ∃ k : ℝ, k = k ∧ ∃ (y x : ℝ), y = a * (x - h)^2 + k :=
sorry

end downward_parabola_with_symmetry_l2375_237593


namespace power_calculation_l2375_237564

noncomputable def a : ℕ := 3 ^ 1006
noncomputable def b : ℕ := 7 ^ 1007
noncomputable def lhs : ℕ := (a + b)^2 - (a - b)^2
noncomputable def rhs : ℕ := 42 * (10 ^ 1007)

theorem power_calculation : lhs = rhs := by
  sorry

end power_calculation_l2375_237564


namespace orchestra_ticket_cost_l2375_237511

noncomputable def cost_balcony : ℝ := 8  -- cost of balcony tickets
noncomputable def total_sold : ℝ := 340  -- total tickets sold
noncomputable def total_revenue : ℝ := 3320  -- total revenue
noncomputable def extra_balcony : ℝ := 40  -- extra tickets sold for balcony than orchestra

theorem orchestra_ticket_cost (x y : ℝ) (h1 : x + extra_balcony = total_sold)
    (h2 : y = x + extra_balcony) (h3 : x + y = total_sold)
    (h4 : x + cost_balcony * y = total_revenue) : 
    cost_balcony = 8 → x = 12 :=
by
  sorry

end orchestra_ticket_cost_l2375_237511


namespace range_of_x_minus_y_l2375_237536

variable (x y : ℝ)
variable (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3)

theorem range_of_x_minus_y : -1 < x - y ∧ x - y < 5 := 
by {
  sorry
}

end range_of_x_minus_y_l2375_237536


namespace func_eq_condition_l2375_237565

variable (a : ℝ)

theorem func_eq_condition (f : ℝ → ℝ) :
  (∀ x : ℝ, f (Real.sin x) + a * f (Real.cos x) = Real.cos (2 * x)) ↔ a ∈ (Set.univ \ {1} : Set ℝ) :=
by
  sorry

end func_eq_condition_l2375_237565


namespace bryden_receives_22_50_dollars_l2375_237568

-- Define the face value of a regular quarter
def face_value_regular : ℝ := 0.25

-- Define the number of regular quarters Bryden has
def num_regular_quarters : ℕ := 4

-- Define the face value of the special quarter
def face_value_special : ℝ := face_value_regular * 2

-- The collector pays 15 times the face value for regular quarters
def multiplier : ℝ := 15

-- Calculate the total face value of all quarters
def total_face_value : ℝ := (num_regular_quarters * face_value_regular) + face_value_special

-- Calculate the total amount Bryden will receive
def total_amount_received : ℝ := multiplier * total_face_value

-- Prove that the total amount Bryden will receive is $22.50
theorem bryden_receives_22_50_dollars : total_amount_received = 22.50 :=
by
  sorry

end bryden_receives_22_50_dollars_l2375_237568


namespace projectile_reaches_30m_at_2_seconds_l2375_237563

theorem projectile_reaches_30m_at_2_seconds:
  ∀ t : ℝ, -5 * t^2 + 25 * t = 30 → t = 2 ∨ t = 3 :=
by
  sorry

end projectile_reaches_30m_at_2_seconds_l2375_237563


namespace age_of_sisters_l2375_237546

theorem age_of_sisters (a b : ℕ) (h1 : 10 * a - 9 * b = 89) 
  (h2 : 10 = 10) : a = 17 ∧ b = 9 :=
by sorry

end age_of_sisters_l2375_237546


namespace sum_of_five_consecutive_odd_numbers_l2375_237591

theorem sum_of_five_consecutive_odd_numbers (x : ℤ) : 
  (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 5 * x :=
by
  sorry

end sum_of_five_consecutive_odd_numbers_l2375_237591


namespace find_a_l2375_237577

variable {x : ℝ} {a b : ℝ}

def setA : Set ℝ := {x | Real.log x / Real.log 2 > 1}
def setB (a : ℝ) : Set ℝ := {x | x < a}
def setIntersection (b : ℝ) : Set ℝ := {x | b < x ∧ x < 2 * b + 3}

theorem find_a (h : setA ∩ setB a = setIntersection b) : a = 7 := 
by
  sorry

end find_a_l2375_237577


namespace vertex_of_given_function_l2375_237590

-- Definition of the given quadratic function
def given_function (x : ℝ) : ℝ := 2 * (x - 4) ^ 2 + 5

-- Definition of the vertex coordinates
def vertex_coordinates : ℝ × ℝ := (4, 5)

-- Theorem stating the vertex coordinates of the function
theorem vertex_of_given_function : (0, given_function 4) = vertex_coordinates :=
by 
  -- Placeholder for the proof
  sorry

end vertex_of_given_function_l2375_237590


namespace min_expression_value_l2375_237581

noncomputable def expression (x y : ℝ) : ℝ := 2*x^2 + 2*y^2 - 8*x + 6*y + 25

theorem min_expression_value : ∃ (x y : ℝ), expression x y = 12.5 :=
by
  sorry

end min_expression_value_l2375_237581


namespace solve_for_x_l2375_237549

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 6 * x + 3

theorem solve_for_x (x : ℝ) (h : delta (phi x) = -1) : x = - 3 / 4 :=
by
  sorry

end solve_for_x_l2375_237549


namespace final_value_of_S_is_10_l2375_237517

-- Define the initial value of S
def initial_S : ℕ := 1

-- Define the sequence of I values
def I_values : List ℕ := [1, 3, 5]

-- Define the update operation on S
def update_S (S : ℕ) (I : ℕ) : ℕ := S + I

-- Final value of S after all updates
def final_S : ℕ := (I_values.foldl update_S initial_S)

-- The theorem stating that the final value of S is 10
theorem final_value_of_S_is_10 : final_S = 10 :=
by
  sorry

end final_value_of_S_is_10_l2375_237517


namespace best_model_is_A_l2375_237557

-- Definitions of the models and their R^2 values
def ModelA_R_squared : ℝ := 0.95
def ModelB_R_squared : ℝ := 0.81
def ModelC_R_squared : ℝ := 0.50
def ModelD_R_squared : ℝ := 0.32

-- Definition stating that the best fitting model is the one with the highest R^2 value
def best_fitting_model (R_squared_A R_squared_B R_squared_C R_squared_D: ℝ) : Prop :=
  R_squared_A > R_squared_B ∧ R_squared_A > R_squared_C ∧ R_squared_A > R_squared_D

-- Proof statement
theorem best_model_is_A : best_fitting_model ModelA_R_squared ModelB_R_squared ModelC_R_squared ModelD_R_squared :=
by
  -- Skipping the proof logic
  sorry

end best_model_is_A_l2375_237557


namespace unique_positive_real_solution_l2375_237531

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_real_solution_l2375_237531


namespace possible_rectangle_areas_l2375_237599

def is_valid_pair (a b : ℕ) := 
  a + b = 12 ∧ a > 0 ∧ b > 0

def rectangle_area (a b : ℕ) := a * b

theorem possible_rectangle_areas :
  {area | ∃ (a b : ℕ), is_valid_pair a b ∧ area = rectangle_area a b} 
  = {11, 20, 27, 32, 35, 36} := 
by 
  sorry

end possible_rectangle_areas_l2375_237599


namespace simplify_fraction_l2375_237589

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + 4 * i) = -2 / 17 - (9 / 17) * i :=
by
  sorry

end simplify_fraction_l2375_237589


namespace volume_of_parallelepiped_l2375_237592

theorem volume_of_parallelepiped (x y z : ℝ)
  (h1 : (x^2 + y^2) * z^2 = 13)
  (h2 : (y^2 + z^2) * x^2 = 40)
  (h3 : (x^2 + z^2) * y^2 = 45) :
  x * y * z = 6 :=
by 
  sorry

end volume_of_parallelepiped_l2375_237592


namespace sum_squares_divisible_by_4_iff_even_l2375_237571

theorem sum_squares_divisible_by_4_iff_even (a b c : ℕ) (ha : a % 2 = 0) (hb : b % 2 = 0) (hc : c % 2 = 0) : 
(a^2 + b^2 + c^2) % 4 = 0 ↔ 
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) :=
sorry

end sum_squares_divisible_by_4_iff_even_l2375_237571


namespace problem_statement_l2375_237534

theorem problem_statement (a x : ℝ) (h_linear_eq : (a + 4) * x ^ |a + 3| + 8 = 0) : a^2 + a - 1 = 1 :=
sorry

end problem_statement_l2375_237534


namespace initial_money_l2375_237543

-- Define the conditions
variable (M : ℝ)
variable (h : (1 / 3) * M = 50)

-- Define the theorem to be proved
theorem initial_money : M = 150 := 
by
  sorry

end initial_money_l2375_237543


namespace total_number_of_notes_l2375_237547

theorem total_number_of_notes 
  (total_money : ℕ)
  (fifty_rupees_notes : ℕ)
  (five_hundred_rupees_notes : ℕ)
  (total_money_eq : total_money = 10350)
  (fifty_rupees_notes_eq : fifty_rupees_notes = 117)
  (money_eq : 50 * fifty_rupees_notes + 500 * five_hundred_rupees_notes = total_money) :
  fifty_rupees_notes + five_hundred_rupees_notes = 126 :=
by sorry

end total_number_of_notes_l2375_237547


namespace value_of_y_l2375_237502

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 5) (h2 : x = 7) : y = 54 := by
  sorry

end value_of_y_l2375_237502


namespace simplify_expr1_simplify_expr2_l2375_237526

theorem simplify_expr1 (a b : ℝ) : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 (t : ℝ) : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l2375_237526


namespace kite_area_l2375_237587

theorem kite_area (EF GH : ℝ) (FG EH : ℕ) (h1 : FG * FG + EH * EH = 25) : EF * GH = 12 :=
by
  sorry

end kite_area_l2375_237587


namespace latus_rectum_of_parabola_l2375_237537

theorem latus_rectum_of_parabola (x : ℝ) :
  (∀ x, y = (-1 / 4 : ℝ) * x^2) → y = (-1 / 2 : ℝ) :=
sorry

end latus_rectum_of_parabola_l2375_237537


namespace smallest_value_of_a_b_l2375_237575

theorem smallest_value_of_a_b :
  ∃ (a b : ℤ), (∀ x : ℤ, ((x^2 + a*x + 20) = 0 ∨ (x^2 + 17*x + b) = 0) → x < 0) ∧ a + b = -5 :=
sorry

end smallest_value_of_a_b_l2375_237575


namespace fraction_undefined_l2375_237585

theorem fraction_undefined (x : ℝ) : (x + 1 = 0) ↔ (x = -1) := 
  sorry

end fraction_undefined_l2375_237585


namespace box_volume_l2375_237532

theorem box_volume (L W H : ℝ) (h1 : L * W = 120) (h2 : W * H = 72) (h3 : L * H = 60) : L * W * H = 720 := 
by sorry

end box_volume_l2375_237532


namespace downstream_speed_l2375_237515

-- Define constants based on conditions given 
def V_upstream : ℝ := 30
def V_m : ℝ := 35

-- Define the speed of the stream based on the given conditions and upstream speed
def V_s : ℝ := V_m - V_upstream

-- The downstream speed is the man's speed in still water plus the stream speed
def V_downstream : ℝ := V_m + V_s

-- Theorem to be proved
theorem downstream_speed : V_downstream = 40 :=
by
  -- The actual proof steps are omitted
  sorry

end downstream_speed_l2375_237515


namespace rounding_sum_eq_one_third_probability_l2375_237548

noncomputable def rounding_sum_probability : ℝ :=
  (λ (total : ℝ) => 
    let round := (λ (x : ℝ) => if x < 0.5 then 0 else if x < 1.5 then 1 else if x < 2.5 then 2 else 3)
    let interval := (λ (start : ℝ) (end_ : ℝ) => end_ - start)
    let sum_conditions := [((0.5,1.5), 3), ((1.5,2.5), 2)]
    let total_length := 3

    let valid_intervals := sum_conditions.map (λ p => interval (p.fst.fst) (p.fst.snd))
    let total_valid_interval := List.sum valid_intervals
    total_valid_interval / total_length
  ) 3

theorem rounding_sum_eq_one_third_probability : rounding_sum_probability = 2 / 3 := by sorry

end rounding_sum_eq_one_third_probability_l2375_237548


namespace roots_sum_one_imp_b_eq_neg_a_l2375_237559

theorem roots_sum_one_imp_b_eq_neg_a (a b c : ℝ) (h : a ≠ 0) 
  (hr : ∀ (r s : ℝ), r + s = 1 → (r * s = c / a) → a * (r^2 + (b/a) * r + c/a) = 0) : b = -a :=
sorry

end roots_sum_one_imp_b_eq_neg_a_l2375_237559


namespace last_digit_of_exponents_l2375_237508

theorem last_digit_of_exponents : 
  (∃k, 2011 = 4 * k + 3 ∧ 
         (2^2011 % 10 = 8) ∧ 
         (3^2011 % 10 = 7)) → 
  ((2^2011 + 3^2011) % 10 = 5) := 
by 
  sorry

end last_digit_of_exponents_l2375_237508


namespace no_55_rooms_l2375_237500

theorem no_55_rooms 
  (count_roses count_carnations count_chrysanthemums : ℕ)
  (rooms_with_CC rooms_with_CR rooms_with_HR : ℕ)
  (at_least_one_bouquet_in_each_room: ∀ (room: ℕ), room > 0)
  (total_rooms : ℕ)
  (h_bouquets : count_roses = 30 ∧ count_carnations = 20 ∧ count_chrysanthemums = 10)
  (h_overlap_conditions: rooms_with_CC = 2 ∧ rooms_with_CR = 3 ∧ rooms_with_HR = 4):
  (total_rooms != 55) :=
sorry

end no_55_rooms_l2375_237500


namespace focal_length_is_correct_l2375_237516

def hyperbola_eqn : Prop := (∀ x y : ℝ, (x^2 / 4) - (y^2 / 9) = 1 → True)

noncomputable def focal_length_of_hyperbola : ℝ :=
  2 * Real.sqrt (4 + 9)

theorem focal_length_is_correct : hyperbola_eqn → focal_length_of_hyperbola = 2 * Real.sqrt 13 := by
  intro h
  sorry

end focal_length_is_correct_l2375_237516


namespace triangle_inequality_l2375_237588

theorem triangle_inequality (a b c : ℕ) : 
    a + b > c ∧ a + c > b ∧ b + c > a ↔ 
    (a, b, c) = (2, 3, 4) ∨ (a, b, c) = (3, 4, 7) ∨ (a, b, c) = (4, 6, 2) ∨ (a, b, c) = (7, 10, 2)
    → (a + b > c ∧ a + c > b ∧ b + c > a ↔ (a, b, c) = (2, 3, 4)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a         ↔ (a, b, c) = (3, 4, 7)) ∧
      (a + b = c ∨ a + c = b ∨ b + c = a        ↔ (a, b, c) = (4, 6, 2)) ∧
      (a + b < c ∨ a + c < b ∨ b + c < a        ↔ (a, b, c) = (7, 10, 2)) :=
sorry

end triangle_inequality_l2375_237588


namespace not_hexagonal_pyramid_l2375_237561

-- Definition of the pyramid with slant height, base radius, and height
structure Pyramid where
  r : ℝ  -- Side length of the base equilateral triangle
  h : ℝ  -- Height of the pyramid
  l : ℝ  -- Slant height (lateral edge)
  hypo : h^2 + (r / 2)^2 = l^2

-- The theorem to prove a pyramid with all edges equal cannot be hexagonal
theorem not_hexagonal_pyramid (p : Pyramid) : p.l ≠ p.r :=
sorry

end not_hexagonal_pyramid_l2375_237561


namespace problem_statement_l2375_237505

theorem problem_statement (x y z : ℝ) (hx : x + y + z = 2) (hxy : xy + xz + yz = -9) (hxyz : xyz = 1) :
  (yz / x) + (xz / y) + (xy / z) = 77 := sorry

end problem_statement_l2375_237505


namespace average_difference_correct_l2375_237594

def daily_diff : List ℤ := [15, 0, -15, 25, 5, -5, 10]
def number_of_days : ℤ := 7

theorem average_difference_correct :
  (daily_diff.sum : ℤ) / number_of_days = 5 := by
  sorry

end average_difference_correct_l2375_237594


namespace percy_bound_longer_martha_step_l2375_237583

theorem percy_bound_longer_martha_step (steps_per_gap_martha: ℕ) (bounds_per_gap_percy: ℕ)
  (gaps: ℕ) (total_distance: ℕ) 
  (step_length_martha: ℝ) (bound_length_percy: ℝ) :
  steps_per_gap_martha = 50 →
  bounds_per_gap_percy = 15 →
  gaps = 50 →
  total_distance = 10560 →
  step_length_martha = total_distance / (steps_per_gap_martha * gaps) →
  bound_length_percy = total_distance / (bounds_per_gap_percy * gaps) →
  (bound_length_percy - step_length_martha) = 10 :=
by
  sorry

end percy_bound_longer_martha_step_l2375_237583


namespace cathy_initial_money_l2375_237541

-- Definitions of the conditions
def moneyFromDad : Int := 25
def moneyFromMom : Int := 2 * moneyFromDad
def totalMoneyReceived : Int := moneyFromDad + moneyFromMom
def currentMoney : Int := 87

-- Theorem stating the proof problem
theorem cathy_initial_money (initialMoney : Int) :
  initialMoney + totalMoneyReceived = currentMoney → initialMoney = 12 :=
by
  sorry

end cathy_initial_money_l2375_237541


namespace find_inverse_of_f_at_4_l2375_237510

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2

-- Statement of the problem
theorem find_inverse_of_f_at_4 : ∃ t : ℝ, f t = 4 ∧ t ≤ 1 ∧ t = -1 := by
  sorry

end find_inverse_of_f_at_4_l2375_237510


namespace frog_climb_time_l2375_237556

-- Define the problem as an assertion within Lean.
theorem frog_climb_time 
  (well_depth : ℕ) (climb_up : ℕ) (slide_down : ℕ) (time_per_meter: ℕ) (climb_start_time : ℕ) 
  (time_to_slide_multiplier: ℚ)
  (time_to_second_position: ℕ) 
  (final_distance: ℕ) 
  (total_time: ℕ)
  (h_start : well_depth = 12)
  (h_climb_up: climb_up = 3)
  (h_slide_down : slide_down = 1)
  (h_time_per_meter : time_per_meter = 1)
  (h_time_to_slide_multiplier: time_to_slide_multiplier = 1/3)
  (h_time_to_second_position : climb_start_time = 8 * 60 /\ time_to_second_position = 8 * 60 + 17)
  (h_final_distance : final_distance = 3)
  (h_total_time: total_time = 22) :
  
  ∃ (t: ℕ), 
    t = total_time := 
by
  sorry

end frog_climb_time_l2375_237556


namespace train_passes_bridge_in_128_seconds_l2375_237574

/-- A proof problem regarding a train passing a bridge -/
theorem train_passes_bridge_in_128_seconds 
  (train_length : ℕ) 
  (train_speed_kmh : ℕ) 
  (bridge_length : ℕ) 
  (conversion_factor : ℚ) 
  (time_to_pass : ℚ) :
  train_length = 1200 →
  train_speed_kmh = 90 →
  bridge_length = 2000 →
  conversion_factor = (5 / 18) →
  time_to_pass = (train_length + bridge_length) / (train_speed_kmh * conversion_factor) →
  time_to_pass = 128 := 
by
  -- We are skipping the proof itself
  sorry

end train_passes_bridge_in_128_seconds_l2375_237574


namespace part1_part2_min_part2_max_part3_l2375_237569

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * a / x - 3 * Real.log x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a + 2 * a / (x^2) - 3 / x

theorem part1 (a : ℝ) : f' a 1 = 0 -> a = 1 := sorry

noncomputable def f1 (x : ℝ) : ℝ := x - 2 / x - 3 * Real.log x

noncomputable def f1' (x : ℝ) : ℝ := 1 + 2 / (x^2) - 3 / x

theorem part2_min (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) -> 
    (f1 2 <= f1 x) := sorry

theorem part2_max (h_a : 1 = 1) : 
    ∀ (x : ℝ), (1 ≤ x) ∧ (x ≤ Real.exp 1) ->
    (f1 x <= f1 1) := sorry

theorem part3 (a : ℝ) : 
    (∀ (x : ℝ), x > 0 -> f' a x ≥ 0) -> a ≥ (3 * Real.sqrt 2) / 4 := sorry

end part1_part2_min_part2_max_part3_l2375_237569


namespace words_memorized_l2375_237523

theorem words_memorized (x y z : ℕ) (h1 : x = 4 * (y + z) / 5) (h2 : x + y = 6 * z / 5) (h3 : 100 < x + y + z ∧ x + y + z < 200) : 
  x + y + z = 198 :=
by
  sorry

end words_memorized_l2375_237523


namespace units_digit_is_six_l2375_237521

theorem units_digit_is_six (n : ℤ) (h : (n^2 / 10 % 10) = 7) : (n^2 % 10) = 6 :=
by sorry

end units_digit_is_six_l2375_237521


namespace angle_B_range_l2375_237580

def range_of_angle_B (a b c : ℝ) (A B C : ℝ) : Prop :=
  (0 < B ∧ B ≤ Real.pi / 3)

theorem angle_B_range
  (a b c A B C : ℝ)
  (h1 : b^2 = a * c)
  (h2 : A + B + C = π)
  (h3 : a > 0)
  (h4 : b > 0)
  (h5 : c > 0)
  (h6 : a + b > c)
  (h7 : a + c > b)
  (h8 : b + c > a) :
  range_of_angle_B a b c A B C :=
sorry

end angle_B_range_l2375_237580


namespace find_overlap_length_l2375_237597

-- Define the given conditions
def plank_length : ℝ := 30 -- length of each plank in cm
def number_of_planks : ℕ := 25 -- number of planks
def total_fence_length : ℝ := 690 -- total length of the fence in cm

-- Definition for the overlap length
def overlap_length (y : ℝ) : Prop :=
  total_fence_length = (13 * plank_length) + (12 * (plank_length - 2 * y))

-- Theorem statement to prove the required overlap length
theorem find_overlap_length : ∃ y : ℝ, overlap_length y ∧ y = 2.5 :=
by 
  -- The proof goes here
  sorry

end find_overlap_length_l2375_237597


namespace operation_result_l2375_237518

def operation (a b : Int) : Int :=
  (a + b) * (a - b)

theorem operation_result :
  operation 4 (operation 2 (-1)) = 7 :=
by
  sorry

end operation_result_l2375_237518


namespace sin_cos_pow_eq_l2375_237540

theorem sin_cos_pow_eq (sin cos : ℝ → ℝ) (x : ℝ) (h₀ : sin x + cos x = -1) (n : ℕ) : 
  sin x ^ n + cos x ^ n = (-1) ^ n :=
by
  sorry

end sin_cos_pow_eq_l2375_237540


namespace average_food_per_week_l2375_237545

-- Definitions based on conditions
def food_first_dog := 13
def food_second_dog := 2 * food_first_dog
def food_third_dog := 6
def number_of_dogs := 3

-- Statement of the proof problem
theorem average_food_per_week : 
  (food_first_dog + food_second_dog + food_third_dog) / number_of_dogs = 15 := 
by sorry

end average_food_per_week_l2375_237545


namespace train_length_is_100_meters_l2375_237530

-- Definitions of conditions
def speed_kmh := 40  -- speed in km/hr
def time_s := 9  -- time in seconds

-- Conversion factors
def km_to_m := 1000  -- 1 km = 1000 meters
def hr_to_s := 3600  -- 1 hour = 3600 seconds

-- Converting speed from km/hr to m/s
def speed_ms := (speed_kmh * km_to_m) / hr_to_s

-- The proof that the length of the train is 100 meters
theorem train_length_is_100_meters :
  (speed_ms * time_s) = 100 :=
by
  sorry

-- The Lean statement merely sets up the problem as asked.

end train_length_is_100_meters_l2375_237530


namespace remainder_of_f_x10_mod_f_l2375_237584

def f (x : ℤ) : ℤ := x^4 + x^3 + x^2 + x + 1

theorem remainder_of_f_x10_mod_f (x : ℤ) : (f (x ^ 10)) % (f x) = 5 :=
by
  sorry

end remainder_of_f_x10_mod_f_l2375_237584


namespace xiao_hua_seat_correct_l2375_237560

-- Define the classroom setup
def classroom : Type := ℤ × ℤ

-- Define the total number of rows and columns in the classroom.
def total_rows : ℤ := 7
def total_columns : ℤ := 8

-- Define the position of Xiao Ming's seat.
def xiao_ming_seat : classroom := (3, 7)

-- Define the position of Xiao Hua's seat.
def xiao_hua_seat : classroom := (5, 2)

-- Prove that Xiao Hua's seat is designated as (5, 2)
theorem xiao_hua_seat_correct : xiao_hua_seat = (5, 2) := by
  -- The proof would go here
  sorry

end xiao_hua_seat_correct_l2375_237560


namespace min_value_geq_four_l2375_237504

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_geq_four (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  4 ≤ min_value_expression x y z :=
sorry

end min_value_geq_four_l2375_237504


namespace proof_system_solution_l2375_237520

noncomputable def solve_system : Prop :=
  ∃ x y : ℚ, x + 4 * y = 14 ∧ (x - 3) / 4 - (y - 3) / 3 = 1 / 12 ∧ x = 3 ∧ y = 11 / 4

theorem proof_system_solution : solve_system :=
sorry

end proof_system_solution_l2375_237520


namespace alice_met_tweedledee_l2375_237579

noncomputable def brother_statement (day : ℕ) : Prop :=
  sorry -- Define the exact logical structure of the statement "I am lying today, and my name is Tweedledum" here

theorem alice_met_tweedledee (day : ℕ) : brother_statement day → (∃ (b : String), b = "Tweedledee") :=
by
  sorry -- provide the proof here

end alice_met_tweedledee_l2375_237579
