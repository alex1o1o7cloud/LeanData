import Mathlib

namespace NUMINAMATH_GPT_parabola_y_range_l253_25306

theorem parabola_y_range
  (x y : ℝ)
  (M_on_C : x^2 = 8 * y)
  (F : ℝ × ℝ)
  (F_focus : F = (0, 2))
  (circle_intersects_directrix : F.2 + y > 4) :
  y > 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_y_range_l253_25306


namespace NUMINAMATH_GPT_polynomial_identity_l253_25360

variable (x y : ℝ)

theorem polynomial_identity :
    (x + y^2) * (x - y^2) * (x^2 + y^4) = x^4 - y^8 :=
sorry

end NUMINAMATH_GPT_polynomial_identity_l253_25360


namespace NUMINAMATH_GPT_John_can_lift_now_l253_25300

def originalWeight : ℕ := 135
def trainingIncrease : ℕ := 265
def bracerIncreaseFactor : ℕ := 6

def newWeight : ℕ := originalWeight + trainingIncrease
def bracerIncrease : ℕ := newWeight * bracerIncreaseFactor
def totalWeight : ℕ := newWeight + bracerIncrease

theorem John_can_lift_now :
  totalWeight = 2800 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_John_can_lift_now_l253_25300


namespace NUMINAMATH_GPT_find_A_from_conditions_l253_25365

variable (A B C D : ℕ)
variable (h_distinct : A ≠ B) (h_distinct2 : C ≠ D)
variable (h_positive : A > 0) (h_positive2 : B > 0) (h_positive3 : C > 0) (h_positive4 : D > 0)
variable (h_product1 : A * B = 72)
variable (h_product2 : C * D = 72)
variable (h_condition : A - B = C * D)

theorem find_A_from_conditions :
  A = 3 :=
sorry

end NUMINAMATH_GPT_find_A_from_conditions_l253_25365


namespace NUMINAMATH_GPT_inequality_for_large_exponent_l253_25320

theorem inequality_for_large_exponent (u : ℕ → ℕ) (x : ℕ) (k : ℕ) (hk : k = 100) (hu : u x = 2^x) : 
  2^(2^(x : ℕ)) > 2^(k * x) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_for_large_exponent_l253_25320


namespace NUMINAMATH_GPT_meeting_point_l253_25380

def same_start (x : ℝ) (y : ℝ) : Prop := x = y

def walk_time (x : ℝ) (y : ℝ) (t : ℝ) : Prop := 
  x * t + y * t = 24

def hector_speed (s : ℝ) : ℝ := s

def jane_speed (s : ℝ) : ℝ := 3 * s

theorem meeting_point (s t : ℝ) :
  same_start 0 0 ∧ walk_time (hector_speed s) (jane_speed s) t → t = 6 / s ∧ (6 : ℝ) = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_meeting_point_l253_25380


namespace NUMINAMATH_GPT_salaries_of_a_and_b_l253_25382

theorem salaries_of_a_and_b {x y : ℝ}
  (h1 : x + y = 5000)
  (h2 : 0.05 * x = 0.15 * y) :
  x = 3750 :=
by sorry

end NUMINAMATH_GPT_salaries_of_a_and_b_l253_25382


namespace NUMINAMATH_GPT_basketball_free_throws_l253_25375

theorem basketball_free_throws (total_players : ℕ) (number_captains : ℕ) (players_not_including_one : ℕ) 
  (free_throws_per_captain : ℕ) (total_free_throws : ℕ) 
  (h1 : total_players = 15)
  (h2 : number_captains = 2)
  (h3 : players_not_including_one = total_players - 1)
  (h4 : free_throws_per_captain = players_not_including_one * number_captains)
  (h5 : total_free_throws = free_throws_per_captain)
  : total_free_throws = 28 :=
by
  -- Proof is not required, so we provide sorry to skip it.
  sorry

end NUMINAMATH_GPT_basketball_free_throws_l253_25375


namespace NUMINAMATH_GPT_value_of_fraction_of_power_l253_25387

-- Define the values in the problem
def a : ℝ := 6
def b : ℝ := 30

-- The problem asks us to prove
theorem value_of_fraction_of_power : 
  (1 / 3) * (a ^ b) = 2 * (a ^ (b - 1)) :=
by
  -- Initial Setup
  let c := (1 / 3) * (a ^ b)
  let d := 2 * (a ^ (b - 1))
  -- The main claim
  show c = d
  sorry

end NUMINAMATH_GPT_value_of_fraction_of_power_l253_25387


namespace NUMINAMATH_GPT_percent_decrease_is_20_l253_25336

/-- Define the original price and sale price as constants. -/
def P_original : ℕ := 100
def P_sale : ℕ := 80

/-- Define the formula for percent decrease. -/
def percent_decrease (P_original P_sale : ℕ) : ℕ :=
  ((P_original - P_sale) * 100) / P_original

/-- Prove that the percent decrease is 20%. -/
theorem percent_decrease_is_20 : percent_decrease P_original P_sale = 20 :=
by
  sorry

end NUMINAMATH_GPT_percent_decrease_is_20_l253_25336


namespace NUMINAMATH_GPT_polar_coordinates_of_point_l253_25318

theorem polar_coordinates_of_point :
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  r = 4 ∧ theta = Real.pi / 3 :=
by
  let x := 2
  let y := 2 * Real.sqrt 3
  let r := Real.sqrt (x^2 + y^2)
  let theta := Real.arctan (y / x)
  have h_r : r = 4 := by {
    -- Calculation for r
    sorry
  }
  have h_theta : theta = Real.pi / 3 := by {
    -- Calculation for theta
    sorry
  }
  exact ⟨h_r, h_theta⟩

end NUMINAMATH_GPT_polar_coordinates_of_point_l253_25318


namespace NUMINAMATH_GPT_calories_per_person_l253_25334

theorem calories_per_person 
  (oranges : ℕ)
  (pieces_per_orange : ℕ)
  (people : ℕ)
  (calories_per_orange : ℝ)
  (h_oranges : oranges = 7)
  (h_pieces_per_orange : pieces_per_orange = 12)
  (h_people : people = 6)
  (h_calories_per_orange : calories_per_orange = 80.0) :
  (oranges * pieces_per_orange / people) * (calories_per_orange / pieces_per_orange) = 93.3338 :=
by
  sorry

end NUMINAMATH_GPT_calories_per_person_l253_25334


namespace NUMINAMATH_GPT_rectangle_perimeter_l253_25394

theorem rectangle_perimeter (b : ℕ) (h1 : 3 * b * b = 192) : 2 * ((3 * b) + b) = 64 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l253_25394


namespace NUMINAMATH_GPT_cone_to_cylinder_ratio_l253_25364

theorem cone_to_cylinder_ratio (r : ℝ) (h_cyl : ℝ) (h_cone : ℝ) 
  (V_cyl : ℝ) (V_cone : ℝ) 
  (h_cyl_eq : h_cyl = 18)
  (r_eq : r = 5)
  (h_cone_eq : h_cone = h_cyl / 3)
  (volume_cyl_eq : V_cyl = π * r^2 * h_cyl)
  (volume_cone_eq : V_cone = 1/3 * π * r^2 * h_cone) :
  V_cone / V_cyl = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_cone_to_cylinder_ratio_l253_25364


namespace NUMINAMATH_GPT_order_of_fractions_l253_25392

theorem order_of_fractions (a b c d : ℚ)
  (h₁ : a = 21/14)
  (h₂ : b = 25/18)
  (h₃ : c = 23/16)
  (h₄ : d = 27/19)
  (h₅ : a > b)
  (h₆ : a > c)
  (h₇ : a > d)
  (h₈ : b < c)
  (h₉ : b < d)
  (h₁₀ : c > d) :
  b < d ∧ d < c ∧ c < a := 
sorry

end NUMINAMATH_GPT_order_of_fractions_l253_25392


namespace NUMINAMATH_GPT_min_value_proof_l253_25324

theorem min_value_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x - 2 * y + 3 * z = 0) : 3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_proof_l253_25324


namespace NUMINAMATH_GPT_tom_distance_before_karen_wins_l253_25329

theorem tom_distance_before_karen_wins :
  let speed_Karen := 60
  let speed_Tom := 45
  let delay_Karen := (4 : ℝ) / 60
  let distance_advantage := 4
  let time_to_catch_up := (distance_advantage + speed_Tom * delay_Karen) / (speed_Karen - speed_Tom)
  let distance_Tom := speed_Tom * time_to_catch_up
  distance_Tom = 21 :=
by
  sorry

end NUMINAMATH_GPT_tom_distance_before_karen_wins_l253_25329


namespace NUMINAMATH_GPT_initial_percentage_l253_25398

variable (P : ℝ)

theorem initial_percentage (P : ℝ) 
  (h1 : 0 ≤ P ∧ P ≤ 100)
  (h2 : (7600 * (1 - P / 100) * 0.75) = 5130) :
  P = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_l253_25398


namespace NUMINAMATH_GPT_power_of_point_l253_25314

namespace ChordsIntersect

variables (A B C D P : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]

def AP := 4
def CP := 9

theorem power_of_point (BP DP : ℕ) :
  AP * BP = CP * DP -> (BP / DP) = 9 / 4 :=
by
  sorry

end ChordsIntersect

end NUMINAMATH_GPT_power_of_point_l253_25314


namespace NUMINAMATH_GPT_stack_logs_total_l253_25344

   theorem stack_logs_total (a l d : ℤ) (n : ℕ) (top_logs : ℕ) (h1 : a = 15) (h2 : l = 5) (h3 : d = -2) (h4 : n = ((l - a) / d).natAbs + 1) (h5 : top_logs = 5) : (n / 2 : ℤ) * (a + l) = 60 :=
   by
   sorry
   
end NUMINAMATH_GPT_stack_logs_total_l253_25344


namespace NUMINAMATH_GPT_total_length_of_joined_papers_l253_25385

theorem total_length_of_joined_papers :
  let length_each_sheet := 10 -- in cm
  let number_of_sheets := 20
  let overlap_length := 0.5 -- in cm
  let total_overlapping_connections := number_of_sheets - 1
  let total_length_without_overlap := length_each_sheet * number_of_sheets
  let total_overlap_length := overlap_length * total_overlapping_connections
  let total_length := total_length_without_overlap - total_overlap_length
  total_length = 190.5 :=
by {
    sorry
}

end NUMINAMATH_GPT_total_length_of_joined_papers_l253_25385


namespace NUMINAMATH_GPT_remaining_distance_l253_25340

-- Definitions of the given conditions
def D : ℕ := 500
def daily_alpha : ℕ := 30
def daily_beta : ℕ := 50
def effective_beta : ℕ := daily_beta / 2

-- Proving the theorem with given conditions
theorem remaining_distance (n : ℕ) (h : n = 25) :
  D - daily_alpha * n = 2 * (D - effective_beta * n) :=
by
  sorry

end NUMINAMATH_GPT_remaining_distance_l253_25340


namespace NUMINAMATH_GPT_Ms_Rush_Speed_to_be_on_time_l253_25352

noncomputable def required_speed (d t r : ℝ) :=
  d = 50 * (t + 1/12) ∧ 
  d = 70 * (t - 1/9) →
  r = d / t →
  r = 74

theorem Ms_Rush_Speed_to_be_on_time 
  (d t r : ℝ) 
  (h1 : d = 50 * (t + 1/12)) 
  (h2 : d = 70 * (t - 1/9)) 
  (h3 : r = d / t) : 
  r = 74 :=
sorry

end NUMINAMATH_GPT_Ms_Rush_Speed_to_be_on_time_l253_25352


namespace NUMINAMATH_GPT_calculate_total_difference_in_miles_l253_25315

def miles_bus_a : ℝ := 1.25
def miles_walk_1 : ℝ := 0.35
def miles_bus_b : ℝ := 2.68
def miles_walk_2 : ℝ := 0.47
def miles_bus_c : ℝ := 3.27
def miles_walk_3 : ℝ := 0.21

def total_miles_on_buses : ℝ := miles_bus_a + miles_bus_b + miles_bus_c
def total_miles_walked : ℝ := miles_walk_1 + miles_walk_2 + miles_walk_3
def total_difference_in_miles : ℝ := total_miles_on_buses - total_miles_walked

theorem calculate_total_difference_in_miles :
  total_difference_in_miles = 6.17 := by
  sorry

end NUMINAMATH_GPT_calculate_total_difference_in_miles_l253_25315


namespace NUMINAMATH_GPT_find_initial_speed_l253_25363

-- Definitions for the conditions
def total_distance : ℕ := 800
def time_at_initial_speed : ℕ := 6
def time_at_60_mph : ℕ := 4
def time_at_40_mph : ℕ := 2
def speed_at_60_mph : ℕ := 60
def speed_at_40_mph : ℕ := 40

-- Setting up the equation: total distance covered
def distance_covered (v : ℕ) : ℕ :=
  time_at_initial_speed * v + time_at_60_mph * speed_at_60_mph + time_at_40_mph * speed_at_40_mph

-- Proof problem statement
theorem find_initial_speed : ∃ v : ℕ, distance_covered v = total_distance ∧ v = 80 := by
  existsi 80
  simp [distance_covered, total_distance, time_at_initial_speed, speed_at_60_mph, time_at_40_mph]
  norm_num
  sorry

end NUMINAMATH_GPT_find_initial_speed_l253_25363


namespace NUMINAMATH_GPT_probability_diff_color_balls_l253_25353

theorem probability_diff_color_balls 
  (Box_A_red : ℕ) (Box_A_black : ℕ) (Box_A_white : ℕ) 
  (Box_B_yellow : ℕ) (Box_B_black : ℕ) (Box_B_white : ℕ) 
  (hA : Box_A_red = 3 ∧ Box_A_black = 3 ∧ Box_A_white = 3)
  (hB : Box_B_yellow = 2 ∧ Box_B_black = 2 ∧ Box_B_white = 2) :
  ((Box_A_red * (Box_B_black + Box_B_white + Box_B_yellow))
  + (Box_A_black * (Box_B_yellow + Box_B_white))
  + (Box_A_white * (Box_B_black + Box_B_yellow))) / 
  ((Box_A_red + Box_A_black + Box_A_white) * 
  (Box_B_yellow + Box_B_black + Box_B_white)) = 7 / 9 := 
by
  sorry

end NUMINAMATH_GPT_probability_diff_color_balls_l253_25353


namespace NUMINAMATH_GPT_rectangle_area_l253_25301

theorem rectangle_area (a : ℕ) (w l : ℕ) (h_square_area : a = 36) (h_square_side : w * w = a) (h_rectangle_length : l = 3 * w) : w * l = 108 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_rectangle_area_l253_25301


namespace NUMINAMATH_GPT_zeros_not_adjacent_probability_l253_25302

-- Definitions based on the conditions
def total_arrangements : ℕ := Nat.choose 6 2
def non_adjacent_zero_arrangements : ℕ := Nat.choose 5 2

-- The probability that the 2 zeros are not adjacent
def probability_non_adjacent_zero : ℚ :=
  (non_adjacent_zero_arrangements : ℚ) / (total_arrangements : ℚ)

-- The theorem statement
theorem zeros_not_adjacent_probability :
  probability_non_adjacent_zero = 2 / 3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_zeros_not_adjacent_probability_l253_25302


namespace NUMINAMATH_GPT_nancy_coffee_expense_l253_25348

-- Definitions corresponding to the conditions
def cost_double_espresso : ℝ := 3.00
def cost_iced_coffee : ℝ := 2.50
def days : ℕ := 20

-- The statement of the problem
theorem nancy_coffee_expense :
  (days * (cost_double_espresso + cost_iced_coffee)) = 110.00 := by
  sorry

end NUMINAMATH_GPT_nancy_coffee_expense_l253_25348


namespace NUMINAMATH_GPT_correlational_relationships_l253_25328

-- Definitions of relationships
def learning_attitude_and_academic_performance := "The relationship between a student's learning attitude and their academic performance"
def teacher_quality_and_student_performance := "The relationship between a teacher's teaching quality and students' academic performance"
def student_height_and_academic_performance := "The relationship between a student's height and their academic performance"
def family_economic_conditions_and_performance := "The relationship between family economic conditions and students' academic performance"

-- Definition of a correlational relationship
def correlational_relationship (relation : String) : Prop :=
  relation = learning_attitude_and_academic_performance ∨
  relation = teacher_quality_and_student_performance

-- Problem statement to prove
theorem correlational_relationships :
  correlational_relationship learning_attitude_and_academic_performance ∧ 
  correlational_relationship teacher_quality_and_student_performance :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_correlational_relationships_l253_25328


namespace NUMINAMATH_GPT_right_triangle_properties_l253_25322

theorem right_triangle_properties (a b c : ℝ) (h1 : c = 13) (h2 : a = 5)
  (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 30 ∧ a + b + c = 30 := by
  sorry

end NUMINAMATH_GPT_right_triangle_properties_l253_25322


namespace NUMINAMATH_GPT_probability_at_least_one_girl_l253_25312

theorem probability_at_least_one_girl (total_students boys girls k : ℕ) (h_total: total_students = 5) (h_boys: boys = 3) (h_girls: girls = 2) (h_k: k = 3) : 
  (1 - ((Nat.choose boys k) / (Nat.choose total_students k))) = 9 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_girl_l253_25312


namespace NUMINAMATH_GPT_sequence_inequality_l253_25368

/-- Sequence definition -/
def a (n : ℕ) : ℚ := 
  if n = 0 then 1/2
  else a (n - 1) + (1 / (n:ℚ)^2) * (a (n - 1))^2

theorem sequence_inequality (n : ℕ) : 
  1 - 1 / 2 ^ (n + 1) ≤ a n ∧ a n < 7 / 5 := 
sorry

end NUMINAMATH_GPT_sequence_inequality_l253_25368


namespace NUMINAMATH_GPT_part_1_part_2_l253_25313

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_1 (x : ℝ) : f x ≤ 4 ↔ x ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

theorem part_2 (b : ℝ) (h₁ : b ≠ 0) (x : ℝ) (h₂ : f x ≥ (|2 * b + 1| + |1 - b|) / |b|) : x ≤ -1.5 :=
by sorry

end NUMINAMATH_GPT_part_1_part_2_l253_25313


namespace NUMINAMATH_GPT_ride_count_l253_25304

noncomputable def initial_tickets : ℕ := 287
noncomputable def spent_on_games : ℕ := 134
noncomputable def earned_tickets : ℕ := 32
noncomputable def cost_per_ride : ℕ := 17

theorem ride_count (initial_tickets : ℕ) (spent_on_games : ℕ) (earned_tickets : ℕ) (cost_per_ride : ℕ) : 
  initial_tickets = 287 ∧ spent_on_games = 134 ∧ earned_tickets = 32 ∧ cost_per_ride = 17 → (initial_tickets - spent_on_games + earned_tickets) / cost_per_ride = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ride_count_l253_25304


namespace NUMINAMATH_GPT_max_inscribed_triangle_area_sum_l253_25357

noncomputable def inscribed_triangle_area (a b : ℝ) (h_a : a = 12) (h_b : b = 13) : ℝ :=
  let s := min (a / (Real.sqrt 3 / 2)) (b / (1 / 2))
  (Real.sqrt 3 / 4) * s^2

theorem max_inscribed_triangle_area_sum :
  inscribed_triangle_area 12 13 (by rfl) (by rfl) = 48 * Real.sqrt 3 - 0 :=
by
  sorry

#eval 48 + 3 + 0
-- Expected Result: 51

end NUMINAMATH_GPT_max_inscribed_triangle_area_sum_l253_25357


namespace NUMINAMATH_GPT_abs_a_gt_abs_c_sub_abs_b_l253_25345

theorem abs_a_gt_abs_c_sub_abs_b (a b c : ℝ) (h : |a + c| < b) : |a| > |c| - |b| :=
sorry

end NUMINAMATH_GPT_abs_a_gt_abs_c_sub_abs_b_l253_25345


namespace NUMINAMATH_GPT_removed_number_is_34_l253_25303
open Real

theorem removed_number_is_34 (n : ℕ) (x : ℕ) (h₁ : 946 = (43 * (43 + 1)) / 2) (h₂ : 912 = 43 * (152 / 7)) : x = 34 :=
by
  sorry

end NUMINAMATH_GPT_removed_number_is_34_l253_25303


namespace NUMINAMATH_GPT_sound_speed_temperature_l253_25390

theorem sound_speed_temperature (v : ℝ) (T : ℝ) (h1 : v = 0.4) (h2 : T = 15 * v^2) :
  T = 2.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_sound_speed_temperature_l253_25390


namespace NUMINAMATH_GPT_inequality_solution_l253_25378

theorem inequality_solution (x : ℝ) : (x + 3) / 2 - (5 * x - 1) / 5 ≥ 0 ↔ x ≤ 17 / 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l253_25378


namespace NUMINAMATH_GPT_rectangle_area_l253_25388

/-- Define a rectangle with its length being three times its breadth, and given diagonal length d = 20.
    Prove that the area of the rectangle is 120 square meters. -/
theorem rectangle_area (b : ℝ) (l : ℝ) (d : ℝ) (h1 : l = 3 * b) (h2 : d = 20) (h3 : l^2 + b^2 = d^2) : l * b = 120 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l253_25388


namespace NUMINAMATH_GPT_necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l253_25355

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 2)

-- Define the dot product calculation
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditionally state that x > -3 is necessary for an acute angle
theorem necessary_condition_for_acute_angle (x : ℝ) :
  dot_product vector_a (vector_b x) > 0 → x > -3 := by
  sorry

-- Define the theorem for necessary but not sufficient condition
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x > -3) → (dot_product vector_a (vector_b x) > 0 ∧ x ≠ 4 / 3) := by
  sorry

end NUMINAMATH_GPT_necessary_condition_for_acute_angle_necessary_but_not_sufficient_condition_l253_25355


namespace NUMINAMATH_GPT_sandy_spent_correct_amount_l253_25341

-- Definitions
def shorts_price : ℝ := 13.99
def shirt_price : ℝ := 12.14
def jacket_price : ℝ := 7.43
def shoes_price : ℝ := 8.50
def accessories_price : ℝ := 10.75
def discount_rate : ℝ := 0.10
def coupon_amount : ℝ := 5.00
def tax_rate : ℝ := 0.075

-- Sum of all items before discounts and coupons
def total_before_discount : ℝ :=
  shorts_price + shirt_price + jacket_price + shoes_price + accessories_price

-- Total after applying the discount
def total_after_discount : ℝ :=
  total_before_discount * (1 - discount_rate)

-- Total after applying the coupon
def total_after_coupon : ℝ :=
  total_after_discount - coupon_amount

-- Total after applying the tax
def total_after_tax : ℝ :=
  total_after_coupon * (1 + tax_rate)

-- Theorem assertion that total amount spent is equal to $45.72
theorem sandy_spent_correct_amount : total_after_tax = 45.72 := by
  sorry

end NUMINAMATH_GPT_sandy_spent_correct_amount_l253_25341


namespace NUMINAMATH_GPT_t_shirt_cost_l253_25381

theorem t_shirt_cost
  (marked_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (selling_price : ℝ)
  (cost : ℝ)
  (h1 : marked_price = 240)
  (h2 : discount_rate = 0.20)
  (h3 : profit_rate = 0.20)
  (h4 : selling_price = 0.8 * marked_price)
  (h5 : selling_price = cost + profit_rate * cost)
  : cost = 160 := 
sorry

end NUMINAMATH_GPT_t_shirt_cost_l253_25381


namespace NUMINAMATH_GPT_true_proposition_is_A_l253_25331

-- Define the propositions
def l1 := ∀ (x y : ℝ), x - 2 * y + 3 = 0
def l2 := ∀ (x y : ℝ), 2 * x + y + 3 = 0
def p : Prop := ¬(l1 ∧ l2 ∧ ¬(∃ (x y : ℝ), x - 2 * y + 3 = 0 ∧ 2 * x + y + 3 = 0 ∧ (1 * 2 + (-2) * 1 ≠ 0)))
def q : Prop := ∃ x₀ : ℝ, (0 < x₀) ∧ (x₀ + 2 > Real.exp x₀)

-- The proof problem statement
theorem true_proposition_is_A : (¬p) ∧ q :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_is_A_l253_25331


namespace NUMINAMATH_GPT_regular_polygon_sides_l253_25367

theorem regular_polygon_sides (C : ℕ) (h : (C - 2) * 180 / C = 144) : C = 10 := 
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l253_25367


namespace NUMINAMATH_GPT_correct_operations_l253_25395

theorem correct_operations :
  (∀ {a b : ℝ}, -(-a + b) = a + b → False) ∧
  (∀ {a : ℝ}, 3 * a^3 - 3 * a^2 = a → False) ∧
  (∀ {x : ℝ}, (x^6)^2 = x^8 → False) ∧
  (∀ {z : ℝ}, 1 / (2 / 3 : ℝ)⁻¹ = 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_correct_operations_l253_25395


namespace NUMINAMATH_GPT_can_place_more_domino_domino_placement_possible_l253_25325

theorem can_place_more_domino (total_squares : ℕ := 36) (uncovered_squares : ℕ := 14) : Prop :=
∃ (n : ℕ), (n * 2 + uncovered_squares ≤ total_squares) ∧ (n ≥ 1)

/-- Proof that on a 6x6 chessboard with some 1x2 dominoes placed, if there are 14 uncovered
squares, then at least one more domino can be placed on the board. -/
theorem domino_placement_possible :
  can_place_more_domino := by
  sorry

end NUMINAMATH_GPT_can_place_more_domino_domino_placement_possible_l253_25325


namespace NUMINAMATH_GPT_compute_b_l253_25305

noncomputable def rational_coefficients (a b : ℚ) :=
∃ x : ℚ, (x^3 + a * x^2 + b * x + 15 = 0)

theorem compute_b (a b : ℚ) (h1 : (3 + Real.sqrt 5)∈{root : ℝ | root^3 + a * root^2 + b * root + 15 = 0}) 
(h2 : rational_coefficients a b) : b = -18.5 :=
by
  sorry

end NUMINAMATH_GPT_compute_b_l253_25305


namespace NUMINAMATH_GPT_total_cookies_baked_l253_25396

-- Definitions based on conditions
def pans : ℕ := 5
def cookies_per_pan : ℕ := 8

-- Statement of the theorem to be proven
theorem total_cookies_baked :
  pans * cookies_per_pan = 40 := by
  sorry

end NUMINAMATH_GPT_total_cookies_baked_l253_25396


namespace NUMINAMATH_GPT_distance_between_foci_of_hyperbola_is_correct_l253_25351

noncomputable def distance_between_foci_of_hyperbola : ℝ := 
  let a_sq := 50
  let b_sq := 8
  let c_sq := a_sq + b_sq
  let c := Real.sqrt c_sq
  2 * c

theorem distance_between_foci_of_hyperbola_is_correct :
  distance_between_foci_of_hyperbola = 2 * Real.sqrt 58 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_hyperbola_is_correct_l253_25351


namespace NUMINAMATH_GPT_p_n_div_5_iff_not_mod_4_zero_l253_25358

theorem p_n_div_5_iff_not_mod_4_zero (n : ℕ) (h : 0 < n) : 
  (1 + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_p_n_div_5_iff_not_mod_4_zero_l253_25358


namespace NUMINAMATH_GPT_range_of_a_l253_25338

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a (a : ℝ) :
  (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a p - f a q) / (p - q) > 1)
  ↔ 3 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l253_25338


namespace NUMINAMATH_GPT_find_a2_b2_geom_sequences_unique_c_l253_25319

-- Define the sequences as per the problem statement
def seqs (a b : ℕ → ℝ) :=
  a 1 = 0 ∧ b 1 = 2013 ∧
  ∀ n : ℕ, (1 ≤ n → (2 * a (n+1) = a n + b n)) ∧ (1 ≤ n → (4 * b (n+1) = a n + 3 * b n))

-- (1) Find values of a_2 and b_2
theorem find_a2_b2 {a b : ℕ → ℝ} (h : seqs a b) :
  a 2 = 1006.5 ∧ b 2 = 1509.75 :=
sorry

-- (2) Prove that {a_n - b_n} and {a_n + 2b_n} are geometric sequences
theorem geom_sequences {a b : ℕ → ℝ} (h : seqs a b) :
  ∃ r s : ℝ, (∃ c : ℝ, ∀ n : ℕ, a n - b n = c * r^n) ∧
             (∃ d : ℝ, ∀ n : ℕ, a n + 2 * b n = d * s^n) :=
sorry

-- (3) Prove there is a unique positive integer c such that a_n < c < b_n always holds
theorem unique_c {a b : ℕ → ℝ} (h : seqs a b) :
  ∃! c : ℝ, (0 < c) ∧ (∀ n : ℕ, 1 ≤ n → a n < c ∧ c < b n) :=
sorry

end NUMINAMATH_GPT_find_a2_b2_geom_sequences_unique_c_l253_25319


namespace NUMINAMATH_GPT_metal_waste_l253_25354

theorem metal_waste (a b : ℝ) (h : a < b) :
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * radius^2
  let side_square := a / Real.sqrt 2
  let area_square := side_square^2
  area_rectangle - area_square = a * b - ( a ^ 2 ) / 2 := by
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * (radius ^ 2)
  let side_square := a / Real.sqrt 2
  let area_square := side_square ^ 2
  sorry

end NUMINAMATH_GPT_metal_waste_l253_25354


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l253_25361

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h1 : A = B) (h2 : C = 50) (h3 : A + B + C = 180) : max A C = 80 :=
by 
  -- Define additional facts about the triangle, e.g., A = B = 50, and sum of angles = 180.
  have h4 : A = 50 := sorry
  rw [h4, h2] at h3
  -- Prove the final result using the given conditions.
  sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l253_25361


namespace NUMINAMATH_GPT_sqrt_neg_squared_eq_two_l253_25309

theorem sqrt_neg_squared_eq_two : (-Real.sqrt 2) ^ 2 = 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_neg_squared_eq_two_l253_25309


namespace NUMINAMATH_GPT_tv_interest_rate_zero_l253_25359

theorem tv_interest_rate_zero (price_installment first_installment last_installment : ℕ) 
  (installment_count : ℕ) (total_price : ℕ) : 
  total_price = 60000 ∧  
  price_installment = 1000 ∧ 
  first_installment = price_installment ∧ 
  last_installment = 59000 ∧ 
  installment_count = 20 ∧  
  (20 * price_installment = 20000) ∧
  (total_price - first_installment = 59000) →
  0 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_tv_interest_rate_zero_l253_25359


namespace NUMINAMATH_GPT_largest_integer_with_square_three_digits_base_7_l253_25343

theorem largest_integer_with_square_three_digits_base_7 : 
  ∃ M : ℕ, (7^2 ≤ M^2 ∧ M^2 < 7^3) ∧ ∀ n : ℕ, (7^2 ≤ n^2 ∧ n^2 < 7^3) → n ≤ M := 
sorry

end NUMINAMATH_GPT_largest_integer_with_square_three_digits_base_7_l253_25343


namespace NUMINAMATH_GPT_mod_squares_eq_one_l253_25370

theorem mod_squares_eq_one
  (n : ℕ)
  (h : n = 5)
  (a : ℤ)
  (ha : ∃ b : ℕ, ↑b = a ∧ b * b ≡ 1 [MOD 5]) :
  (a * a) % n = 1 :=
by
  sorry

end NUMINAMATH_GPT_mod_squares_eq_one_l253_25370


namespace NUMINAMATH_GPT_proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l253_25317

structure CubeSymmetry where
  planes : Nat
  axes : Nat
  has_center : Bool

def general_cube_symmetry : CubeSymmetry :=
  { planes := 9, axes := 9, has_center := true }

def case_a : CubeSymmetry :=
  { planes := 4, axes := 1, has_center := false }

def case_b1 : CubeSymmetry :=
  { planes := 5, axes := 3, has_center := true }

def case_b2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

def case_c1 : CubeSymmetry :=
  { planes := 3, axes := 0, has_center := false }

def case_c2 : CubeSymmetry :=
  { planes := 2, axes := 1, has_center := false }

theorem proof_case_a : case_a = { planes := 4, axes := 1, has_center := false } := by
  sorry

theorem proof_case_b1 : case_b1 = { planes := 5, axes := 3, has_center := true } := by
  sorry

theorem proof_case_b2 : case_b2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

theorem proof_case_c1 : case_c1 = { planes := 3, axes := 0, has_center := false } := by
  sorry

theorem proof_case_c2 : case_c2 = { planes := 2, axes := 1, has_center := false } := by
  sorry

end NUMINAMATH_GPT_proof_case_a_proof_case_b1_proof_case_b2_proof_case_c1_proof_case_c2_l253_25317


namespace NUMINAMATH_GPT_probability_heads_tails_4_tosses_l253_25356

-- Define the probabilities of heads and tails
variables (p q : ℝ)

-- Define the conditions
def unfair_coin (p q : ℝ) : Prop :=
  p ≠ q ∧ p + q = 1 ∧ 2 * p * q = 1/2

-- Define the theorem to prove the probability of two heads and two tails
theorem probability_heads_tails_4_tosses 
  (h_unfair : unfair_coin p q) 
  : 6 * (p * q)^2 = 3 / 8 :=
by sorry

end NUMINAMATH_GPT_probability_heads_tails_4_tosses_l253_25356


namespace NUMINAMATH_GPT_initial_number_of_men_l253_25399

def initial_average_age_increased_by_2_years_when_two_women_replace_two_men 
    (M : ℕ) (A men1 men2 women1 women2 : ℕ) : Prop :=
  (men1 = 20) ∧ (men2 = 24) ∧ (women1 = 30) ∧ (women2 = 30) ∧
  ((M * A) + 16 = (M * (A + 2)))

theorem initial_number_of_men (M : ℕ) (A : ℕ) (men1 men2 women1 women2: ℕ):
  initial_average_age_increased_by_2_years_when_two_women_replace_two_men M A men1 men2 women1 women2 → 
  2 * M = 16 → M = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l253_25399


namespace NUMINAMATH_GPT_comb_comb_l253_25362

theorem comb_comb (n1 k1 n2 k2 : ℕ) (h1 : n1 = 10) (h2 : k1 = 3) (h3 : n2 = 8) (h4 : k2 = 4) :
  (Nat.choose n1 k1) * (Nat.choose n2 k2) = 8400 := by
  rw [h1, h2, h3, h4]
  change Nat.choose 10 3 * Nat.choose 8 4 = 8400
  -- Adding the proof steps is not necessary as per instructions
  sorry

end NUMINAMATH_GPT_comb_comb_l253_25362


namespace NUMINAMATH_GPT_nigella_sold_3_houses_l253_25386

noncomputable def houseA_cost : ℝ := 60000
noncomputable def houseB_cost : ℝ := 3 * houseA_cost
noncomputable def houseC_cost : ℝ := 2 * houseA_cost - 110000
noncomputable def commission_rate : ℝ := 0.02

noncomputable def houseA_commission : ℝ := houseA_cost * commission_rate
noncomputable def houseB_commission : ℝ := houseB_cost * commission_rate
noncomputable def houseC_commission : ℝ := houseC_cost * commission_rate

noncomputable def total_commission : ℝ := houseA_commission + houseB_commission + houseC_commission
noncomputable def base_salary : ℝ := 3000
noncomputable def total_earnings : ℝ := base_salary + total_commission

theorem nigella_sold_3_houses 
  (H1 : total_earnings = 8000) 
  (H2 : houseA_cost = 60000) 
  (H3 : houseB_cost = 3 * houseA_cost) 
  (H4 : houseC_cost = 2 * houseA_cost - 110000) 
  (H5 : commission_rate = 0.02) :
  3 = 3 :=
by 
  -- Proof not required
  sorry

end NUMINAMATH_GPT_nigella_sold_3_houses_l253_25386


namespace NUMINAMATH_GPT_percentage_increase_l253_25366

variable (A B y : ℝ)

theorem percentage_increase (h1 : B > A) (h2 : A > 0) :
  B = A + y / 100 * A ↔ y = 100 * (B - A) / A :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l253_25366


namespace NUMINAMATH_GPT_molecular_weight_BaO_is_correct_l253_25391

-- Define the atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of BaO as the sum of atomic weights of Ba and O
def molecular_weight_BaO := atomic_weight_Ba + atomic_weight_O

-- Theorem stating the molecular weight of BaO
theorem molecular_weight_BaO_is_correct : molecular_weight_BaO = 153.33 := by
  -- Proof can be filled in
  sorry

end NUMINAMATH_GPT_molecular_weight_BaO_is_correct_l253_25391


namespace NUMINAMATH_GPT_value_of_f_at_4_l253_25326

noncomputable def f (x : ℝ) (c : ℝ) (d : ℝ) : ℝ :=
  c * x ^ 2 + d * x + 3

theorem value_of_f_at_4 :
  (∃ c d : ℝ, f 1 c d = 3 ∧ f 2 c d = 5) → f 4 1 (-1) = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_at_4_l253_25326


namespace NUMINAMATH_GPT_linear_function_general_form_special_case_linear_function_proof_quadratic_function_general_form_special_case_quadratic_function1_proof_special_case_quadratic_function2_proof_l253_25339

variable {α : Type*} [Ring α]

def linear_function (a b x : α) : α :=
  a * x + b

def special_case_linear_function (a x : α) : α :=
  a * x

def quadratic_function (a b c x : α) : α :=
  a * x^2 + b * x + c

def special_case_quadratic_function1 (a c x : α) : α :=
  a * x^2 + c

def special_case_quadratic_function2 (a x : α) : α :=
  a * x^2

theorem linear_function_general_form (a b x : α) :
  ∃ y, y = linear_function a b x := by
  sorry

theorem special_case_linear_function_proof (a x : α) :
  ∃ y, y = special_case_linear_function a x := by
  sorry

theorem quadratic_function_general_form (a b c x : α) :
  a ≠ 0 → ∃ y, y = quadratic_function a b c x := by
  sorry

theorem special_case_quadratic_function1_proof (a b c x : α) :
  a ≠ 0 → b = 0 → ∃ y, y = special_case_quadratic_function1 a c x := by
  sorry

theorem special_case_quadratic_function2_proof (a b c x : α) :
  a ≠ 0 → b = 0 → c = 0 → ∃ y, y = special_case_quadratic_function2 a x := by
  sorry

end NUMINAMATH_GPT_linear_function_general_form_special_case_linear_function_proof_quadratic_function_general_form_special_case_quadratic_function1_proof_special_case_quadratic_function2_proof_l253_25339


namespace NUMINAMATH_GPT_abs_diff_squares_110_108_l253_25316

theorem abs_diff_squares_110_108 : abs ((110 : ℤ)^2 - (108 : ℤ)^2) = 436 := by
  sorry

end NUMINAMATH_GPT_abs_diff_squares_110_108_l253_25316


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l253_25349

theorem ellipse_foci_coordinates (x y : ℝ) :
  2 * x^2 + 3 * y^2 = 1 →
  (∃ c : ℝ, (c = (Real.sqrt 6) / 6) ∧ ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0))) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l253_25349


namespace NUMINAMATH_GPT_expression_evaluates_to_one_l253_25346

theorem expression_evaluates_to_one :
  (1 / 3)⁻¹ + |1 - Real.sqrt 3| - 2 * Real.sin (Real.pi / 3) + (Real.pi - 2016)^0 - (8:ℝ)^(1/3) = 1 :=
by
  -- step-by-step simplification skipped, as per requirements
  sorry

end NUMINAMATH_GPT_expression_evaluates_to_one_l253_25346


namespace NUMINAMATH_GPT_n_not_composite_l253_25332

theorem n_not_composite
  (n : ℕ) (h1 : n > 1)
  (a : ℕ) (q : ℕ) (hq_prime : Nat.Prime q)
  (hq1 : q ∣ (n - 1))
  (hq2 : q > Nat.sqrt n - 1)
  (hn_div : n ∣ (a^(n-1) - 1))
  (hgcd : Nat.gcd (a^(n-1)/q - 1) n = 1) :
  ¬ Nat.Prime n :=
sorry

end NUMINAMATH_GPT_n_not_composite_l253_25332


namespace NUMINAMATH_GPT_coeff_exists_l253_25369

theorem coeff_exists :
  ∃ (A B C : ℕ), 
    ¬(8 ∣ A) ∧ ¬(8 ∣ B) ∧ ¬(8 ∣ C) ∧ 
    (∀ (n : ℕ), 8 ∣ (A * 5^n + B * 3^(n-1) + C))
    :=
sorry

end NUMINAMATH_GPT_coeff_exists_l253_25369


namespace NUMINAMATH_GPT_find_geo_prog_numbers_l253_25397

noncomputable def geo_prog_numbers (a1 a2 a3 : ℝ) : Prop :=
a1 * a2 * a3 = 27 ∧ a1 + a2 + a3 = 13

theorem find_geo_prog_numbers :
  geo_prog_numbers 1 3 9 ∨ geo_prog_numbers 9 3 1 :=
sorry

end NUMINAMATH_GPT_find_geo_prog_numbers_l253_25397


namespace NUMINAMATH_GPT_graveyard_bones_count_l253_25347

def total_skeletons : ℕ := 20
def half_total (n : ℕ) : ℕ := n / 2
def skeletons_adult_women : ℕ := half_total total_skeletons
def remaining_skeletons : ℕ := total_skeletons - skeletons_adult_women
def even_split (n : ℕ) : ℕ := n / 2
def skeletons_adult_men : ℕ := even_split remaining_skeletons
def skeletons_children : ℕ := even_split remaining_skeletons

def bones_per_woman : ℕ := 20
def bones_per_man : ℕ := bones_per_woman + 5
def bones_per_child : ℕ := bones_per_woman / 2

def total_bones_adult_women : ℕ := skeletons_adult_women * bones_per_woman
def total_bones_adult_men : ℕ := skeletons_adult_men * bones_per_man
def total_bones_children : ℕ := skeletons_children * bones_per_child

def total_bones_in_graveyard : ℕ := total_bones_adult_women + total_bones_adult_men + total_bones_children

theorem graveyard_bones_count : total_bones_in_graveyard = 375 := by
  sorry

end NUMINAMATH_GPT_graveyard_bones_count_l253_25347


namespace NUMINAMATH_GPT_basketball_probability_third_shot_l253_25333

theorem basketball_probability_third_shot
  (p1 : ℚ) (p2_given_made1 : ℚ) (p2_given_missed1 : ℚ) (p3_given_made2 : ℚ) (p3_given_missed2 : ℚ) :
  p1 = 2 / 3 → p2_given_made1 = 2 / 3 → p2_given_missed1 = 1 / 3 → p3_given_made2 = 2 / 3 → p3_given_missed2 = 2 / 3 →
  (p1 * p2_given_made1 * p3_given_made2 + p1 * p2_given_missed1 * p3_given_misseds2 + 
   (1 - p1) * p2_given_made1 * p3_given_made2 + (1 - p1) * p2_given_missed1 * p3_given_missed2) = 14 / 27 :=
by
  sorry

end NUMINAMATH_GPT_basketball_probability_third_shot_l253_25333


namespace NUMINAMATH_GPT_total_bins_l253_25389

-- Definition of the problem conditions
def road_length : ℕ := 400
def placement_interval : ℕ := 20
def bins_per_side : ℕ := (road_length / placement_interval) - 1

-- Statement of the problem
theorem total_bins : 2 * bins_per_side = 38 := by
  sorry

end NUMINAMATH_GPT_total_bins_l253_25389


namespace NUMINAMATH_GPT_only_integer_solution_is_zero_l253_25335

theorem only_integer_solution_is_zero (x y : ℤ) (h : x^4 + y^4 = 3 * x^3 * y) : x = 0 ∧ y = 0 :=
by {
  -- Here we would provide the proof steps.
  sorry
}

end NUMINAMATH_GPT_only_integer_solution_is_zero_l253_25335


namespace NUMINAMATH_GPT_fantasia_max_capacity_reach_l253_25371

def acre_per_person := 1
def land_acres := 40000
def base_population := 500
def population_growth_factor := 4
def years_per_growth_period := 20

def maximum_capacity := land_acres / acre_per_person

def population_at_time (years_from_2000 : ℕ) : ℕ :=
  base_population * population_growth_factor^(years_from_2000 / years_per_growth_period)

theorem fantasia_max_capacity_reach :
  ∃ t : ℕ, t = 60 ∧ population_at_time t = maximum_capacity := by sorry

end NUMINAMATH_GPT_fantasia_max_capacity_reach_l253_25371


namespace NUMINAMATH_GPT_smallest_N_l253_25321

-- Definitions for conditions
variable (a b c : ℕ) (N : ℕ)

-- Define the conditions for the given problem
def valid_block (a b c : ℕ) : Prop :=
  (a - 1) * (b - 1) * (c - 1) = 252

def block_volume (a b c : ℕ) : ℕ := a * b * c

-- The target theorem to be proved
theorem smallest_N (h : valid_block a b c) : N = 224 :=
  sorry

end NUMINAMATH_GPT_smallest_N_l253_25321


namespace NUMINAMATH_GPT_luis_can_make_sum_multiple_of_4_l253_25383

noncomputable def sum_of_dice (dice: List ℕ) : ℕ :=
  dice.sum 

theorem luis_can_make_sum_multiple_of_4 (d1 d2 d3: ℕ) 
  (h1: 1 ≤ d1 ∧ d1 ≤ 6) 
  (h2: 1 ≤ d2 ∧ d2 ≤ 6) 
  (h3: 1 ≤ d3 ∧ d3 ≤ 6) : 
  ∃ (dice: List ℕ), dice.length = 3 ∧ 
  sum_of_dice dice % 4 = 0 := 
by
  sorry

end NUMINAMATH_GPT_luis_can_make_sum_multiple_of_4_l253_25383


namespace NUMINAMATH_GPT_seven_digit_divisible_by_11_l253_25393

theorem seven_digit_divisible_by_11 (m n : ℕ) (h1: 0 ≤ m ∧ m ≤ 9) (h2: 0 ≤ n ∧ n ≤ 9) (h3 : 10 + n - m ≡ 0 [MOD 11])  : m + n = 1 :=
by
  sorry

end NUMINAMATH_GPT_seven_digit_divisible_by_11_l253_25393


namespace NUMINAMATH_GPT_sum_groups_eq_250_l253_25308

-- Definitions for each sum
def sum1 : ℕ := 3 + 13 + 23 + 33 + 43
def sum2 : ℕ := 7 + 17 + 27 + 37 + 47

-- Theorem statement that the sum of these groups is 250
theorem sum_groups_eq_250 : sum1 + sum2 = 250 :=
by sorry

end NUMINAMATH_GPT_sum_groups_eq_250_l253_25308


namespace NUMINAMATH_GPT_exists_n_sum_digits_n3_eq_million_l253_25377

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem exists_n_sum_digits_n3_eq_million :
  ∃ n : ℕ, sum_digits n = 100 ∧ sum_digits (n ^ 3) = 1000000 := sorry

end NUMINAMATH_GPT_exists_n_sum_digits_n3_eq_million_l253_25377


namespace NUMINAMATH_GPT_required_pumps_l253_25327

-- Define the conditions in Lean
variables (x a b n : ℝ)

-- Condition 1: x + 40a = 80b
def condition1 : Prop := x + 40 * a = 2 * 40 * b

-- Condition 2: x + 16a = 64b
def condition2 : Prop := x + 16 * a = 4 * 16 * b

-- Main theorem: Given the conditions, prove that n >= 6 satisfies the remaining requirement
theorem required_pumps (h1 : condition1 x a b) (h2 : condition2 x a b) : n >= 6 :=
by
  sorry

end NUMINAMATH_GPT_required_pumps_l253_25327


namespace NUMINAMATH_GPT_organization_members_count_l253_25310

theorem organization_members_count (num_committees : ℕ) (pair_membership : ℕ → ℕ → ℕ) :
  num_committees = 5 →
  (∀ i j k l : ℕ, i ≠ j → k ≠ l → pair_membership i j = pair_membership k l → i = k ∧ j = l ∨ i = l ∧ j = k) →
  ∃ (num_members : ℕ), num_members = 10 :=
by
  sorry

end NUMINAMATH_GPT_organization_members_count_l253_25310


namespace NUMINAMATH_GPT_gain_percent_calculation_l253_25379

theorem gain_percent_calculation (gain_paise : ℕ) (cost_price_rupees : ℕ) (rupees_to_paise : ℕ)
  (h_gain_paise : gain_paise = 70)
  (h_cost_price_rupees : cost_price_rupees = 70)
  (h_rupees_to_paise : rupees_to_paise = 100) :
  ((gain_paise / rupees_to_paise) / cost_price_rupees) * 100 = 1 :=
by
  -- Placeholder to indicate the need for proof
  sorry

end NUMINAMATH_GPT_gain_percent_calculation_l253_25379


namespace NUMINAMATH_GPT_initial_velocity_is_three_l253_25337

noncomputable def displacement (t : ℝ) : ℝ :=
  3 * t - t^2

theorem initial_velocity_is_three : 
  (deriv displacement 0) = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_velocity_is_three_l253_25337


namespace NUMINAMATH_GPT_fraction_of_time_l253_25373

-- Define the time John takes to clean the entire house
def John_time : ℝ := 6

-- Define the combined time it takes Nick and John to clean the entire house
def combined_time : ℝ := 3.6

-- Given this configuration, we need to prove the fraction result.
theorem fraction_of_time (N : ℝ) (H1 : John_time = 6) (H2 : ∀ N, (1/John_time) + (1/N) = 1/combined_time) :
  (John_time / 2) / N = 1 / 3 := 
by sorry

end NUMINAMATH_GPT_fraction_of_time_l253_25373


namespace NUMINAMATH_GPT_cost_per_dvd_l253_25330

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) (cost_per_dvd : ℝ) :
  total_cost = 4.80 ∧ num_dvds = 4 → cost_per_dvd = 1.20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_per_dvd_l253_25330


namespace NUMINAMATH_GPT_petya_friends_l253_25311

variable (x : ℕ) -- Define x to be a natural number (number of friends)
variable (S : ℕ) -- Define S to be a natural number (total number of stickers Petya has)

-- Conditions from the problem
axiom condition1 : S = 5 * x + 8 -- If Petya gives 5 stickers to each friend, 8 stickers are left
axiom condition2 : S = 6 * x - 11 -- If Petya gives 6 stickers to each friend, he is short 11 stickers

theorem petya_friends : x = 19 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_petya_friends_l253_25311


namespace NUMINAMATH_GPT_Mark_marbles_correct_l253_25384

def Connie_marbles : ℕ := 323
def Juan_marbles : ℕ := Connie_marbles + 175
def Mark_marbles : ℕ := 3 * Juan_marbles

theorem Mark_marbles_correct : Mark_marbles = 1494 := 
by
  sorry

end NUMINAMATH_GPT_Mark_marbles_correct_l253_25384


namespace NUMINAMATH_GPT_cube_edge_adjacency_l253_25376

def is_beautiful (f: Finset ℕ) := 
  ∃ a b c d, f = {a, b, c, d} ∧ a = b + c + d

def cube_is_beautiful (faces: Finset (Finset ℕ)) :=
  ∃ t1 t2 t3, t1 ∈ faces ∧ t2 ∈ faces ∧ t3 ∈ faces ∧
  is_beautiful t1 ∧ is_beautiful t2 ∧ is_beautiful t3

def valid_adjacency (v: ℕ) (n1 n2 n3: ℕ) := 
  v = 6 ∧ ((n1 = 2 ∧ n2 = 3 ∧ n3 = 5) ∨
           (n1 = 2 ∧ n2 = 3 ∧ n3 = 7) ∨
           (n1 = 3 ∧ n2 = 5 ∧ n3 = 7))

theorem cube_edge_adjacency : 
  ∀ faces: Finset (Finset ℕ), 
  ∃ v n1 n2 n3, 
  (v = 6 ∧ (valid_adjacency v n1 n2 n3)) ∧
  cube_is_beautiful faces := 
by
  -- Entails the proof, which is not required here
  sorry

end NUMINAMATH_GPT_cube_edge_adjacency_l253_25376


namespace NUMINAMATH_GPT_perm_prime_count_12345_l253_25372

theorem perm_prime_count_12345 : 
  (∀ x : List ℕ, (x ∈ (List.permutations [1, 2, 3, 4, 5])) → 
    (10^4 * x.head! + 10^3 * x.tail.head! + 10^2 * x.tail.tail.head! + 10 * x.tail.tail.tail.head! + x.tail.tail.tail.tail.head!) % 3 = 0)
  → 
  0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_perm_prime_count_12345_l253_25372


namespace NUMINAMATH_GPT_saved_per_bagel_l253_25342

-- Definitions of the conditions
def bagel_cost_each : ℝ := 3.50
def dozen_cost : ℝ := 38
def bakers_dozen : ℕ := 13
def discount : ℝ := 0.05

-- The conjecture we need to prove
theorem saved_per_bagel : 
  let total_cost_without_discount := dozen_cost + bagel_cost_each
  let discount_amount := discount * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount_amount
  let cost_per_bagel_without_discount := dozen_cost / 12
  let cost_per_bagel_with_discount := total_cost_with_discount / bakers_dozen
  let savings_per_bagel := cost_per_bagel_without_discount - cost_per_bagel_with_discount
  let savings_in_cents := savings_per_bagel * 100
  savings_in_cents = 13.36 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_saved_per_bagel_l253_25342


namespace NUMINAMATH_GPT_gcd_lcm_condition_implies_divisibility_l253_25350

theorem gcd_lcm_condition_implies_divisibility
  (a b : ℤ) (h : Int.gcd a b + Int.lcm a b = a + b) : a ∣ b ∨ b ∣ a := 
sorry

end NUMINAMATH_GPT_gcd_lcm_condition_implies_divisibility_l253_25350


namespace NUMINAMATH_GPT_boys_camp_percentage_l253_25307

theorem boys_camp_percentage (x : ℕ) (total_boys : ℕ) (percent_science : ℕ) (not_science_boys : ℕ) 
    (percent_not_science : ℕ) (h1 : not_science_boys = percent_not_science * (x / 100) * total_boys) 
    (h2 : percent_not_science = 100 - percent_science) (h3 : percent_science = 30) 
    (h4 : not_science_boys = 21) (h5 : total_boys = 150) : x = 20 :=
by 
  sorry

end NUMINAMATH_GPT_boys_camp_percentage_l253_25307


namespace NUMINAMATH_GPT_value_of_a_l253_25323

theorem value_of_a (a : ℝ) (h_neg : a < 0) (h_f : ∀ (x : ℝ), (0 < x ∧ x ≤ 1) → 
  (x + 4 * a / x - a < 0)) : a ≤ -1 / 3 := 
sorry

end NUMINAMATH_GPT_value_of_a_l253_25323


namespace NUMINAMATH_GPT_property_damage_worth_40000_l253_25374

-- Definitions based on conditions in a)
def medical_bills : ℝ := 70000
def insurance_rate : ℝ := 0.80
def carl_payment : ℝ := 22000
def carl_rate : ℝ := 0.20

theorem property_damage_worth_40000 :
  ∃ P : ℝ, P = 40000 ∧ 
    (carl_payment = carl_rate * (P + medical_bills)) :=
by
  sorry

end NUMINAMATH_GPT_property_damage_worth_40000_l253_25374
