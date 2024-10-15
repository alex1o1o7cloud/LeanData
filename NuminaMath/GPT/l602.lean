import Mathlib

namespace NUMINAMATH_GPT_find_four_numbers_l602_60281

theorem find_four_numbers
  (a d : ℕ)
  (h_pos : 0 < a - d ∧ 0 < a ∧ 0 < a + d)
  (h_sum : (a - d) + a + (a + d) = 48)
  (b c : ℕ)
  (h_geo : b = a ∧ c = a + d)
  (last : ℕ)
  (h_last_val : last = 25)
  (h_geometric_seq : (a + d) * (a + d) = b * last)
  : (a - d, a, a + d, last) = (12, 16, 20, 25) := 
  sorry

end NUMINAMATH_GPT_find_four_numbers_l602_60281


namespace NUMINAMATH_GPT_find_ratio_MH_NH_OH_l602_60238

-- Defining the main problem variables.
variable {A B C O H M N : Type} -- A, B, C are points, O is circumcenter, H is orthocenter, M and N are points on other segments
variables (angleA : ℝ) (AB AC : ℝ)
variables (angleBOC angleBHC : ℝ)
variables (BM CN MH NH OH : ℝ)

-- Conditions: Given constraints from the problem.
axiom angle_A_eq_60 : angleA = 60 -- ∠A = 60°
axiom AB_greater_AC : AB > AC -- AB > AC
axiom circumcenter_property : angleBOC = 120 -- ∠BOC = 120°
axiom orthocenter_property : angleBHC = 120 -- ∠BHC = 120°
axiom BM_eq_CN : BM = CN -- BM = CN

-- Statement of the mathematical proof we need to show.
theorem find_ratio_MH_NH_OH : (MH + NH) / OH = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_MH_NH_OH_l602_60238


namespace NUMINAMATH_GPT_sum_f_neg_l602_60286

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_neg {x1 x2 x3 : ℝ}
  (h1 : x1 + x2 > 0)
  (h2 : x2 + x3 > 0)
  (h3 : x3 + x1 > 0) :
  f x1 + f x2 + f x3 < 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_f_neg_l602_60286


namespace NUMINAMATH_GPT_money_left_is_41_l602_60205

-- Define the amounts saved by Tanner in each month
def savings_september : ℕ := 17
def savings_october : ℕ := 48
def savings_november : ℕ := 25

-- Define the amount spent by Tanner on the video game
def spent_video_game : ℕ := 49

-- Total savings after the three months
def total_savings : ℕ := savings_september + savings_october + savings_november

-- Calculate the money left after spending on the video game
def money_left : ℕ := total_savings - spent_video_game

-- The theorem we need to prove
theorem money_left_is_41 : money_left = 41 := by
  sorry

end NUMINAMATH_GPT_money_left_is_41_l602_60205


namespace NUMINAMATH_GPT_original_team_members_l602_60297

theorem original_team_members (m p total_points : ℕ) (h_m : m = 3) (h_p : p = 2) (h_total : total_points = 12) :
  (total_points / p) + m = 9 := by
  sorry

end NUMINAMATH_GPT_original_team_members_l602_60297


namespace NUMINAMATH_GPT_three_g_of_x_l602_60284

noncomputable def g (x : ℝ) : ℝ := 3 / (3 + x)

theorem three_g_of_x (x : ℝ) (h : x > 0) : 3 * g x = 27 / (9 + x) :=
by
  sorry

end NUMINAMATH_GPT_three_g_of_x_l602_60284


namespace NUMINAMATH_GPT_total_cost_mulch_l602_60235

-- Define the conditions
def tons_to_pounds (tons : ℕ) : ℕ := tons * 2000

def price_per_pound : ℝ := 2.5

-- Define the statement to prove
theorem total_cost_mulch (mulch_in_tons : ℕ) (h₁ : mulch_in_tons = 3) : 
  tons_to_pounds mulch_in_tons * price_per_pound = 15000 :=
by
  -- The proof would normally go here.
  sorry

end NUMINAMATH_GPT_total_cost_mulch_l602_60235


namespace NUMINAMATH_GPT_combined_degrees_l602_60263

theorem combined_degrees (S J W : ℕ) (h1 : S = 150) (h2 : J = S - 5) (h3 : W = S - 3) : S + J + W = 442 :=
by
  sorry

end NUMINAMATH_GPT_combined_degrees_l602_60263


namespace NUMINAMATH_GPT_find_m_value_l602_60232

noncomputable def m_value (x : ℤ) (m : ℝ) : Prop :=
  3 * (x + 1) - 2 ≤ 4 * (x - 3) + 1 ∧
  (∃ x, x ≥ 12 ∧ (1 / 2 : ℝ) * x - m = 5)

theorem find_m_value : ∃ m : ℝ, ∀ x : ℤ, m_value x m → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l602_60232


namespace NUMINAMATH_GPT_minimum_group_members_round_table_l602_60222

theorem minimum_group_members_round_table (n : ℕ) (h1 : ∀ (a : ℕ),  a < n) : 5 ≤ n :=
by
  sorry

end NUMINAMATH_GPT_minimum_group_members_round_table_l602_60222


namespace NUMINAMATH_GPT_farmer_shipped_30_boxes_this_week_l602_60278

-- Defining the given conditions
def last_week_boxes : ℕ := 10
def last_week_pomelos : ℕ := 240
def this_week_dozen : ℕ := 60
def pomelos_per_dozen : ℕ := 12

-- Translating conditions into mathematical statements
def pomelos_per_box_last_week : ℕ := last_week_pomelos / last_week_boxes
def this_week_pomelos_total : ℕ := this_week_dozen * pomelos_per_dozen
def boxes_shipped_this_week : ℕ := this_week_pomelos_total / pomelos_per_box_last_week

-- The theorem we prove, that given the conditions, the number of boxes shipped this week is 30.
theorem farmer_shipped_30_boxes_this_week :
  boxes_shipped_this_week = 30 :=
sorry

end NUMINAMATH_GPT_farmer_shipped_30_boxes_this_week_l602_60278


namespace NUMINAMATH_GPT_units_digit_of_factorial_sum_l602_60243

theorem units_digit_of_factorial_sum : 
  (1 + 2 + 6 + 4) % 10 = 3 := sorry

end NUMINAMATH_GPT_units_digit_of_factorial_sum_l602_60243


namespace NUMINAMATH_GPT_chipped_marbles_is_22_l602_60290

def bags : List ℕ := [20, 22, 25, 30, 32, 34, 36]

-- Jane and George take some bags and one bag with chipped marbles is left.
theorem chipped_marbles_is_22
  (h1 : ∃ (jane_bags george_bags : List ℕ) (remaining_bag : ℕ),
    (jane_bags ++ george_bags ++ [remaining_bag] = bags ∧
     jane_bags.length = 3 ∧
     (george_bags.length = 2 ∨ george_bags.length = 3) ∧
     3 * remaining_bag = List.sum jane_bags + List.sum george_bags)) :
  ∃ (c : ℕ), c = 22 := 
sorry

end NUMINAMATH_GPT_chipped_marbles_is_22_l602_60290


namespace NUMINAMATH_GPT_trip_to_market_distance_l602_60260

theorem trip_to_market_distance 
  (school_trip_one_way : ℝ) (school_days_per_week : ℕ) 
  (weekly_total_mileage : ℝ) (round_trips_per_day : ℕ) (market_trip_count : ℕ) :
  (school_trip_one_way = 2.5) →
  (school_days_per_week = 4) →
  (round_trips_per_day = 2) →
  (weekly_total_mileage = 44) →
  (market_trip_count = 1) →
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  market_trip_distance = 2 :=
by
  intros h1 h2 h3 h4 h5
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  sorry

end NUMINAMATH_GPT_trip_to_market_distance_l602_60260


namespace NUMINAMATH_GPT_probability_athlete_A_selected_number_of_males_selected_number_of_females_selected_l602_60239

noncomputable def total_members := 42
noncomputable def boys := 28
noncomputable def girls := 14
noncomputable def selected := 6

theorem probability_athlete_A_selected :
  (selected : ℚ) / total_members = 1 / 7 :=
by sorry

theorem number_of_males_selected :
  (selected * (boys : ℚ)) / total_members = 4 :=
by sorry

theorem number_of_females_selected :
  (selected * (girls : ℚ)) / total_members = 2 :=
by sorry

end NUMINAMATH_GPT_probability_athlete_A_selected_number_of_males_selected_number_of_females_selected_l602_60239


namespace NUMINAMATH_GPT_certain_number_is_l602_60254

theorem certain_number_is (x : ℝ) : 
  x * (-4.5) = 2 * (-4.5) - 36 → x = 10 :=
by
  intro h
  -- proof goes here
  sorry

end NUMINAMATH_GPT_certain_number_is_l602_60254


namespace NUMINAMATH_GPT_total_population_l602_60246

-- Definitions based on given conditions
variables (b g t : ℕ)
variables (h1 : b = 4 * g) (h2 : g = 8 * t)

-- Theorem statement
theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * t :=
by
  sorry

end NUMINAMATH_GPT_total_population_l602_60246


namespace NUMINAMATH_GPT_num_ways_to_assign_grades_l602_60204

-- Define the number of students
def num_students : ℕ := 12

-- Define the number of grades available to each student
def num_grades : ℕ := 4

-- The theorem stating that the total number of ways to assign grades is 4^12
theorem num_ways_to_assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end NUMINAMATH_GPT_num_ways_to_assign_grades_l602_60204


namespace NUMINAMATH_GPT_average_glasses_is_15_l602_60217

variable (S L : ℕ)

-- Conditions:
def box1 := 12 -- One box contains 12 glasses
def box2 := 16 -- Another box contains 16 glasses
def total_glasses := 480 -- Total number of glasses
def diff_L_S := 16 -- There are 16 more larger boxes

-- Equations derived from conditions:
def eq1 : Prop := (12 * S + 16 * L = total_glasses)
def eq2 : Prop := (L = S + diff_L_S)

-- We need to prove that the average number of glasses per box is 15:
def avg_glasses_per_box := total_glasses / (S + L)

-- The statement we need to prove:
theorem average_glasses_is_15 :
  (12 * S + 16 * L = total_glasses) ∧ (L = S + diff_L_S) → avg_glasses_per_box = 15 :=
by
  sorry

end NUMINAMATH_GPT_average_glasses_is_15_l602_60217


namespace NUMINAMATH_GPT_angle_C_max_l602_60250

theorem angle_C_max (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_cond : Real.sin B / Real.sin A = 2 * Real.cos (A + B))
  (h_max_B : B = Real.pi / 3) :
  C = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_max_l602_60250


namespace NUMINAMATH_GPT_trains_clear_time_l602_60257

noncomputable def time_to_clear (length_train1 length_train2 speed_train1 speed_train2 : ℕ) : ℝ :=
  (length_train1 + length_train2) / ((speed_train1 + speed_train2) * 1000 / 3600)

theorem trains_clear_time :
  time_to_clear 121 153 80 65 = 6.803 :=
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_trains_clear_time_l602_60257


namespace NUMINAMATH_GPT_two_trains_clearing_time_l602_60224

noncomputable def length_train1 : ℝ := 100  -- Length of Train 1 in meters
noncomputable def length_train2 : ℝ := 160  -- Length of Train 2 in meters
noncomputable def speed_train1 : ℝ := 42 * 1000 / 3600  -- Speed of Train 1 in m/s
noncomputable def speed_train2 : ℝ := 30 * 1000 / 3600  -- Speed of Train 2 in m/s
noncomputable def total_distance : ℝ := length_train1 + length_train2  -- Total distance to be covered
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2  -- Relative speed

theorem two_trains_clearing_time : total_distance / relative_speed = 13 := by
  sorry

end NUMINAMATH_GPT_two_trains_clearing_time_l602_60224


namespace NUMINAMATH_GPT_absolute_value_inequality_solution_l602_60206

theorem absolute_value_inequality_solution (x : ℝ) : |2*x - 1| < 3 ↔ -1 < x ∧ x < 2 := 
sorry

end NUMINAMATH_GPT_absolute_value_inequality_solution_l602_60206


namespace NUMINAMATH_GPT_maximum_profit_l602_60214

noncomputable def sales_volume (x : ℝ) : ℝ := -10 * x + 1000
noncomputable def profit (x : ℝ) : ℝ := -10 * x^2 + 1300 * x - 30000

theorem maximum_profit : ∀ x : ℝ, 44 ≤ x ∧ x ≤ 46 → profit x ≤ 8640 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_maximum_profit_l602_60214


namespace NUMINAMATH_GPT_correct_equations_l602_60262

theorem correct_equations (m n : ℕ) (h1 : n = 4 * m - 2) (h2 : n = 2 * m + 58) :
  (4 * m - 2 = 2 * m + 58 ∨ (n + 2) / 4 = (n - 58) / 2) :=
by
  sorry

end NUMINAMATH_GPT_correct_equations_l602_60262


namespace NUMINAMATH_GPT_major_axis_length_of_ellipse_l602_60269

-- Definition of the conditions
def line (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 2 = 1
def is_focus (x y m : ℝ) : Prop := line x y ∧ ellipse x y m

theorem major_axis_length_of_ellipse (m : ℝ) (h₀ : m > 0) :
  (∃ (x y : ℝ), is_focus x y m) → 2 * Real.sqrt 6 = 2 * Real.sqrt m :=
sorry

end NUMINAMATH_GPT_major_axis_length_of_ellipse_l602_60269


namespace NUMINAMATH_GPT_ratio_of_areas_l602_60273

theorem ratio_of_areas
  (PQ QR RP : ℝ)
  (PQ_pos : 0 < PQ)
  (QR_pos : 0 < QR)
  (RP_pos : 0 < RP)
  (s t u : ℝ)
  (s_pos : 0 < s)
  (t_pos : 0 < t)
  (u_pos : 0 < u)
  (h1 : s + t + u = 3 / 4)
  (h2 : s^2 + t^2 + u^2 = 1 / 2)
  : (1 - (s * (1 - u) + t * (1 - s) + u * (1 - t))) = 7 / 32 := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l602_60273


namespace NUMINAMATH_GPT_find_r4_l602_60203

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end NUMINAMATH_GPT_find_r4_l602_60203


namespace NUMINAMATH_GPT_diplomats_neither_french_nor_russian_l602_60228

variable (total_diplomats : ℕ)
variable (speak_french : ℕ)
variable (not_speak_russian : ℕ)
variable (speak_both : ℕ)

theorem diplomats_neither_french_nor_russian {total_diplomats speak_french not_speak_russian speak_both : ℕ} 
  (h1 : total_diplomats = 100)
  (h2 : speak_french = 22)
  (h3 : not_speak_russian = 32)
  (h4 : speak_both = 10) :
  ((total_diplomats - (speak_french + (total_diplomats - not_speak_russian) - speak_both)) * 100) / total_diplomats = 20 := 
by
  sorry

end NUMINAMATH_GPT_diplomats_neither_french_nor_russian_l602_60228


namespace NUMINAMATH_GPT_inequality_proof_l602_60248

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l602_60248


namespace NUMINAMATH_GPT_obrien_hats_after_loss_l602_60207

noncomputable def hats_simpson : ℕ := 15

noncomputable def initial_hats_obrien : ℕ := 2 * hats_simpson + 5

theorem obrien_hats_after_loss : initial_hats_obrien - 1 = 34 :=
by
  sorry

end NUMINAMATH_GPT_obrien_hats_after_loss_l602_60207


namespace NUMINAMATH_GPT_half_angle_in_second_quadrant_l602_60272

def quadrant_of_half_alpha (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : Prop :=
  π / 2 < α / 2 ∧ α / 2 < 3 * π / 4

theorem half_angle_in_second_quadrant (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : quadrant_of_half_alpha α hα1 hα2 hcos :=
sorry

end NUMINAMATH_GPT_half_angle_in_second_quadrant_l602_60272


namespace NUMINAMATH_GPT_smallest_part_proportional_division_l602_60210

theorem smallest_part_proportional_division (a b c d total : ℕ) (h : a + b + c + d = total) (sum_equals_360 : 360 = total * 15):
  min (4 * 15) (min (5 * 15) (min (7 * 15) (8 * 15))) = 60 :=
by
  -- Defining the proportions and overall total
  let a := 5
  let b := 7
  let c := 4
  let d := 8
  let total_parts := a + b + c + d

  -- Given that the division is proportional
  let part_value := 360 / total_parts

  -- Assert that the smallest part is equal to the smallest proportion times the value of one part
  let smallest_part := c * part_value
  trivial

end NUMINAMATH_GPT_smallest_part_proportional_division_l602_60210


namespace NUMINAMATH_GPT_Sam_has_seven_watermelons_l602_60288

-- Declare the initial number of watermelons
def initial_watermelons : Nat := 4

-- Declare the additional number of watermelons Sam grew
def more_watermelons : Nat := 3

-- Prove that the total number of watermelons is 7
theorem Sam_has_seven_watermelons : initial_watermelons + more_watermelons = 7 :=
by
  sorry

end NUMINAMATH_GPT_Sam_has_seven_watermelons_l602_60288


namespace NUMINAMATH_GPT_number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l602_60209

-- Definitions for the sets A and B
def A : Set Int := {x | x^2 - 3 * x - 10 <= 0}
def B (m : Int) : Set Int := {x | m - 1 <= x ∧ x <= 2 * m + 1}

-- Proof for the number of non-empty proper subsets of A
theorem number_of_non_empty_proper_subsets_of_A (x : Int) (h : x ∈ A) : 2^(8 : Nat) - 2 = 254 := by
  sorry

-- Proof for the range of m such that A ⊇ B
theorem range_of_m_for_A_superset_B (m : Int) : (∀ x, x ∈ B m → x ∈ A) ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := by
  sorry

end NUMINAMATH_GPT_number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l602_60209


namespace NUMINAMATH_GPT_expense_of_5_yuan_is_minus_5_yuan_l602_60271

def income (x : Int) : Int :=
  x

def expense (x : Int) : Int :=
  -x

theorem expense_of_5_yuan_is_minus_5_yuan : expense 5 = -5 :=
by
  unfold expense
  sorry

end NUMINAMATH_GPT_expense_of_5_yuan_is_minus_5_yuan_l602_60271


namespace NUMINAMATH_GPT_complete_square_transformation_l602_60245

theorem complete_square_transformation : 
  ∀ (x : ℝ), (x^2 - 8 * x + 9 = 0) → ((x - 4)^2 = 7) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_complete_square_transformation_l602_60245


namespace NUMINAMATH_GPT_find_divisor_l602_60219

theorem find_divisor (n : ℕ) (h_n : n = 36) : 
  ∃ D : ℕ, ((n + 10) * 2 / D) - 2 = 44 → D = 2 :=
by
  use 2
  intros h
  sorry

end NUMINAMATH_GPT_find_divisor_l602_60219


namespace NUMINAMATH_GPT_probability_leftmost_blue_off_rightmost_red_on_l602_60270

noncomputable def calculate_probability : ℚ :=
  let total_arrangements := Nat.choose 8 4
  let total_on_choices := Nat.choose 8 4
  let favorable_arrangements := Nat.choose 6 3 * Nat.choose 7 3
  favorable_arrangements / (total_arrangements * total_on_choices)

theorem probability_leftmost_blue_off_rightmost_red_on :
  calculate_probability = 1 / 7 := 
by
  sorry

end NUMINAMATH_GPT_probability_leftmost_blue_off_rightmost_red_on_l602_60270


namespace NUMINAMATH_GPT_container_volume_ratio_l602_60241

theorem container_volume_ratio (V1 V2 : ℚ)
  (h1 : (3 / 5) * V1 = (2 / 3) * V2) :
  V1 / V2 = 10 / 9 :=
by sorry

end NUMINAMATH_GPT_container_volume_ratio_l602_60241


namespace NUMINAMATH_GPT_f_analytical_expression_g_value_l602_60234

noncomputable def f (ω x : ℝ) : ℝ := (1/2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x + Real.pi / 2)

noncomputable def g (ω x : ℝ) : ℝ := f ω (x + Real.pi / 4)

theorem f_analytical_expression (x : ℝ) (hω : ω = 2 ∧ ω > 0) : 
  f 2 x = Real.sin (2 * x - Real.pi / 3) :=
sorry

theorem g_value (α : ℝ) (hω : ω = 2 ∧ ω > 0) (h : g 2 (α / 2) = 4/5) : 
  g 2 (-α) = -7/25 :=
sorry

end NUMINAMATH_GPT_f_analytical_expression_g_value_l602_60234


namespace NUMINAMATH_GPT_sequence_general_formula_l602_60225

theorem sequence_general_formula
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (3 / 2) * (a n) - 3) :
  ∀ n, a n = 3 * (2 : ℝ) ^ n :=
by sorry

end NUMINAMATH_GPT_sequence_general_formula_l602_60225


namespace NUMINAMATH_GPT_maximum_m_value_l602_60298

variable {a b c : ℝ}

noncomputable def maximum_m : ℝ := 9/8

theorem maximum_m_value 
  (h1 : (a - b)^2 + (b - c)^2 + (c - a)^2 ≥ maximum_m * a^2)
  (h2 : b^2 - 4 * a * c ≥ 0) : 
  maximum_m = 9 / 8 :=
sorry

end NUMINAMATH_GPT_maximum_m_value_l602_60298


namespace NUMINAMATH_GPT_blue_balls_needed_l602_60268

-- Conditions
variables (R Y B W : ℝ)
axiom h1 : 2 * R = 5 * B
axiom h2 : 3 * Y = 7 * B
axiom h3 : 9 * B = 6 * W

-- Proof Problem
theorem blue_balls_needed : (3 * R + 4 * Y + 3 * W) = (64 / 3) * B := by
  sorry

end NUMINAMATH_GPT_blue_balls_needed_l602_60268


namespace NUMINAMATH_GPT_digits_property_l602_60253

theorem digits_property (n : ℕ) (h : 100 ≤ n ∧ n < 1000) :
  (∃ (f : ℕ → Prop), ∀ d ∈ [n / 100, (n / 10) % 10, n % 10], f d ∧ (¬ d = 0 ∧ ¬ Nat.Prime d)) ↔ 
  (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ∈ [1, 4, 6, 8, 9]) :=
sorry

end NUMINAMATH_GPT_digits_property_l602_60253


namespace NUMINAMATH_GPT_total_selection_methods_l602_60280

def num_courses_group_A := 3
def num_courses_group_B := 4
def total_courses_selected := 3

theorem total_selection_methods 
  (at_least_one_from_each : num_courses_group_A > 0 ∧ num_courses_group_B > 0)
  (total_courses : total_courses_selected = 3) :
  ∃ N, N = 30 :=
sorry

end NUMINAMATH_GPT_total_selection_methods_l602_60280


namespace NUMINAMATH_GPT_area_of_trapezoid_EFGH_l602_60242

-- Define the vertices of the trapezoid
structure Point where
  x : ℤ
  y : ℤ

def E : Point := ⟨-2, -3⟩
def F : Point := ⟨-2, 2⟩
def G : Point := ⟨4, 5⟩
def H : Point := ⟨4, 0⟩

-- Define the formula for the area of a trapezoid
def trapezoid_area (b1 b2 height : ℤ) : ℤ :=
  (b1 + b2) * height / 2

-- The proof statement
theorem area_of_trapezoid_EFGH : trapezoid_area (F.y - E.y) (G.y - H.y) (G.x - E.x) = 30 := by
  sorry -- proof not required

end NUMINAMATH_GPT_area_of_trapezoid_EFGH_l602_60242


namespace NUMINAMATH_GPT_car_average_speed_l602_60221

noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 24
  let t3 := (D / 3) / 30
  let total_time := t1 + t2 + t3
  D / total_time

theorem car_average_speed :
  average_speed D = 34.2857 := by
  sorry

end NUMINAMATH_GPT_car_average_speed_l602_60221


namespace NUMINAMATH_GPT_compare_powers_l602_60296

theorem compare_powers:
  (2 ^ 2023) * (7 ^ 2023) < (3 ^ 2023) * (5 ^ 2023) :=
  sorry

end NUMINAMATH_GPT_compare_powers_l602_60296


namespace NUMINAMATH_GPT_work_alone_days_l602_60231

theorem work_alone_days (d : ℝ) (p q : ℝ) (h1 : q = 10) (h2 : 2 * (1/d + 1/q) = 0.3) : d = 20 :=
by
  sorry

end NUMINAMATH_GPT_work_alone_days_l602_60231


namespace NUMINAMATH_GPT_intersection_complement_l602_60244

open Set

variable (U : Type) [TopologicalSpace U]

def A : Set ℝ := { x | x ≥ 0 }

def B : Set ℝ := { y | y ≤ 0 }

theorem intersection_complement (U : Type) [TopologicalSpace U] : 
  A ∩ (compl B) = { x | x > 0 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l602_60244


namespace NUMINAMATH_GPT_difference_of_fractions_l602_60227

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h1 : a = 700) (h2 : b = 7) : a - b = 693 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_difference_of_fractions_l602_60227


namespace NUMINAMATH_GPT_system_solutions_l602_60247

noncomputable def f (t : ℝ) : ℝ := 4 * t^2 / (1 + 4 * t^2)

theorem system_solutions (x y z : ℝ) :
  (f x = y ∧ f y = z ∧ f z = x) ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_system_solutions_l602_60247


namespace NUMINAMATH_GPT_min_value_x_squared_y_squared_z_squared_l602_60282

theorem min_value_x_squared_y_squared_z_squared
  (x y z : ℝ)
  (h : x + 2 * y + 3 * z = 6) :
  x^2 + y^2 + z^2 ≥ (18 / 7) :=
sorry

end NUMINAMATH_GPT_min_value_x_squared_y_squared_z_squared_l602_60282


namespace NUMINAMATH_GPT_smallest_n_in_range_l602_60229

theorem smallest_n_in_range (n : ℤ) (h1 : 4 ≤ n ∧ n ≤ 12) (h2 : n ≡ 2 [ZMOD 9]) : n = 11 :=
sorry

end NUMINAMATH_GPT_smallest_n_in_range_l602_60229


namespace NUMINAMATH_GPT_health_risk_probability_l602_60213

theorem health_risk_probability :
  let p := 26
  let q := 57
  p + q = 83 :=
by {
  sorry
}

end NUMINAMATH_GPT_health_risk_probability_l602_60213


namespace NUMINAMATH_GPT_jellybean_probability_l602_60256

theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 6
  let blue_jellybeans := 3
  let white_jellybeans := 6
  let total_chosen := 4
  let total_combinations := Nat.choose total_jellybeans total_chosen
  let red_combinations := Nat.choose red_jellybeans 3
  let non_red_combinations := Nat.choose (blue_jellybeans + white_jellybeans) 1
  let successful_outcomes := red_combinations * non_red_combinations
  let probability := (successful_outcomes : ℚ) / total_combinations
  probability = 4 / 91 :=
by 
  sorry

end NUMINAMATH_GPT_jellybean_probability_l602_60256


namespace NUMINAMATH_GPT_result_more_than_half_l602_60258

theorem result_more_than_half (x : ℕ) (h : x = 4) : (2 * x + 5) - (x / 2) = 11 := by
  sorry

end NUMINAMATH_GPT_result_more_than_half_l602_60258


namespace NUMINAMATH_GPT_cos_sin_eq_l602_60277

theorem cos_sin_eq (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (Real.sin x + 3 * Real.cos x = (2 * Real.sqrt 6 - 3) / 5) ∨
  (Real.sin x + 3 * Real.cos x = -(2 * Real.sqrt 6 + 3) / 5) := 
by
  sorry

end NUMINAMATH_GPT_cos_sin_eq_l602_60277


namespace NUMINAMATH_GPT_AM_GM_Inequality_l602_60220

theorem AM_GM_Inequality 
  (a b c : ℝ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_GPT_AM_GM_Inequality_l602_60220


namespace NUMINAMATH_GPT_min_needed_framing_l602_60275

-- Define the original dimensions of the picture
def original_width_inch : ℕ := 5
def original_height_inch : ℕ := 7

-- Define the factor by which the dimensions are doubled
def doubling_factor : ℕ := 2

-- Define the width of the border
def border_width_inch : ℕ := 3

-- Define the function to calculate the new dimensions after doubling
def new_width_inch : ℕ := original_width_inch * doubling_factor
def new_height_inch : ℕ := original_height_inch * doubling_factor

-- Define the function to calculate dimensions including the border
def total_width_inch : ℕ := new_width_inch + 2 * border_width_inch
def total_height_inch : ℕ := new_height_inch + 2 * border_width_inch

-- Define the function to calculate the perimeter of the picture with border
def perimeter_inch : ℕ := 2 * (total_width_inch + total_height_inch)

-- Conversision from inches to feet (1 foot = 12 inches)
def inch_to_foot_conversion_factor : ℕ := 12

-- Define the function to calculate the minimum linear feet of framing needed
noncomputable def min_linear_feet_of_framing : ℕ := (perimeter_inch + inch_to_foot_conversion_factor - 1) / inch_to_foot_conversion_factor

-- The main theorem statement
theorem min_needed_framing : min_linear_feet_of_framing = 6 := by
  -- Proof construction is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_min_needed_framing_l602_60275


namespace NUMINAMATH_GPT_crayons_lost_or_given_away_l602_60236

theorem crayons_lost_or_given_away (P E L : ℕ) (h1 : P = 479) (h2 : E = 134) (h3 : L = P - E) : L = 345 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_crayons_lost_or_given_away_l602_60236


namespace NUMINAMATH_GPT_top_weight_l602_60264

theorem top_weight (T : ℝ) : 
    (9 * 0.8 + 7 * T = 10.98) → T = 0.54 :=
by 
  intro h
  have H_sum := h
  simp only [mul_add, add_assoc, mul_assoc, mul_comm, add_comm, mul_comm 7] at H_sum
  sorry

end NUMINAMATH_GPT_top_weight_l602_60264


namespace NUMINAMATH_GPT_find_sin_2alpha_l602_60283

theorem find_sin_2alpha (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 4) Real.pi) 
  (h2 : 3 * Real.cos (2 * α) = 4 * Real.sin (Real.pi / 4 - α)) : 
  Real.sin (2 * α) = -1 / 9 :=
sorry

end NUMINAMATH_GPT_find_sin_2alpha_l602_60283


namespace NUMINAMATH_GPT_find_coefficients_l602_60266

theorem find_coefficients (a b : ℚ) (h_a_nonzero : a ≠ 0)
  (h_prod : (3 * b - 2 * a = 0) ∧ (-2 * b + 3 = 0)) : 
  a = 9 / 4 ∧ b = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_coefficients_l602_60266


namespace NUMINAMATH_GPT_wristband_distribution_l602_60251

open Nat 

theorem wristband_distribution (x y : ℕ) 
  (h1 : 2 * x + 2 * y = 460) 
  (h2 : 2 * x = 3 * y) : x = 138 :=
sorry

end NUMINAMATH_GPT_wristband_distribution_l602_60251


namespace NUMINAMATH_GPT_area_of_rectangle_l602_60267

-- Definitions from problem conditions
variable (AB CD x : ℝ)
variable (h1 : AB = 24)
variable (h2 : CD = 60)
variable (h3 : BC = x)
variable (h4 : BF = 2 * x)
variable (h5 : similar (triangle AEB) (triangle FDC))

-- Goal: Prove the area of rectangle BCFE
theorem area_of_rectangle (h1 : AB = 24) (h2 : CD = 60) (x y : ℝ) 
  (h3 : BC = x) (h4 : BF = 2 * x) (h5 : BC * BF = y) : y = 1440 :=
sorry -- proof will be provided here

end NUMINAMATH_GPT_area_of_rectangle_l602_60267


namespace NUMINAMATH_GPT_counterexamples_count_l602_60259

def sum_of_digits (n : Nat) : Nat :=
  -- Function to calculate the sum of digits of n
  sorry

def no_zeros (n : Nat) : Prop :=
  -- Function to check that there are no zeros in the digits of n
  sorry

def is_prime (n : Nat) : Prop :=
  -- Function to check if a number is prime
  sorry

theorem counterexamples_count : 
  ∃ (M : List Nat), 
  (∀ m ∈ M, sum_of_digits m = 5 ∧ no_zeros m) ∧ 
  (∀ m ∈ M, ¬ is_prime m) ∧
  M.length = 9 := 
sorry

end NUMINAMATH_GPT_counterexamples_count_l602_60259


namespace NUMINAMATH_GPT_evaluate_expression_l602_60215

theorem evaluate_expression (m n : ℝ) (h : m - n = 2) :
  (2 * m^2 - 4 * m * n + 2 * n^2 - 1) = 7 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l602_60215


namespace NUMINAMATH_GPT_solution_set_of_xf_gt_0_l602_60218

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_ineq : ∀ x : ℝ, x > 0 → f x < x * (deriv f x)
axiom f_at_one : f 1 = 0

theorem solution_set_of_xf_gt_0 : {x : ℝ | x * f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_xf_gt_0_l602_60218


namespace NUMINAMATH_GPT_find_consecutive_integers_sum_eq_l602_60274

theorem find_consecutive_integers_sum_eq 
    (M : ℤ) : ∃ n k : ℤ, (0 ≤ k ∧ k ≤ 9) ∧ (M = (9 * n + 45 - k)) := 
sorry

end NUMINAMATH_GPT_find_consecutive_integers_sum_eq_l602_60274


namespace NUMINAMATH_GPT_part1_part2_l602_60237

-- Define the absolute value function
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a

-- Given conditions
def condition1 : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6

def condition2 (a : ℝ) : Prop :=
  ∃ t m : ℝ, f (t / 2) a ≤ m - f (-t) a

-- Statements to prove
theorem part1 : ∃ a : ℝ, condition1 ∧ a = 1 := by
  sorry

theorem part2 : ∀ {a : ℝ}, a = 1 → ∃ m : ℝ, m ≥ 3.5 ∧ condition2 a := by
  sorry

end NUMINAMATH_GPT_part1_part2_l602_60237


namespace NUMINAMATH_GPT_larger_exceeds_smaller_by_16_l602_60216

-- Define the smaller number S and the larger number L in terms of the ratio 7:11
def S : ℕ := 28
def L : ℕ := (11 * S) / 7

-- State the theorem that the larger number exceeds the smaller number by 16
theorem larger_exceeds_smaller_by_16 : L - S = 16 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_larger_exceeds_smaller_by_16_l602_60216


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l602_60299

theorem volume_of_rectangular_prism :
  ∃ (a b c : ℝ), (a * b = 54) ∧ (b * c = 56) ∧ (a * c = 60) ∧ (a * b * c = 379) :=
by sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l602_60299


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l602_60294

theorem arithmetic_sequence_a5 {a : ℕ → ℕ} 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 2 + a 8 = 12) : 
  a 5 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l602_60294


namespace NUMINAMATH_GPT_find_a_prove_inequality_l602_60261

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.exp x + 2 * x + a * Real.log x

theorem find_a (a : ℝ) (h : (2 * Real.exp 1 + 2 + a) * (-1 / 2) = -1) : a = -2 * Real.exp 1 :=
by
  sorry

theorem prove_inequality (a : ℝ) (h1 : a = -2 * Real.exp 1) :
    ∀ x : ℝ, x > 0 → f x a > x^2 + 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_prove_inequality_l602_60261


namespace NUMINAMATH_GPT_not_possible_in_five_trips_possible_in_six_trips_l602_60293

def truck_capacity := 2000
def rice_sacks := 150
def corn_sacks := 100
def rice_weight_per_sack := 60
def corn_weight_per_sack := 25

def total_rice_weight := rice_sacks * rice_weight_per_sack
def total_corn_weight := corn_sacks * corn_weight_per_sack
def total_weight := total_rice_weight + total_corn_weight

theorem not_possible_in_five_trips : total_weight > 5 * truck_capacity :=
by
  sorry

theorem possible_in_six_trips : total_weight <= 6 * truck_capacity :=
by
  sorry

#print axioms not_possible_in_five_trips
#print axioms possible_in_six_trips

end NUMINAMATH_GPT_not_possible_in_five_trips_possible_in_six_trips_l602_60293


namespace NUMINAMATH_GPT_grain_to_rice_system_l602_60201

variable (x y : ℕ)

/-- Conversion rate of grain to rice is 3/5. -/
def conversion_rate : ℚ := 3 / 5

/-- Total bucket capacity is 10 dou. -/
def total_capacity : ℕ := 10

/-- Rice obtained after threshing is 7 dou. -/
def rice_obtained : ℕ := 7

/-- The system of equations representing the problem. -/
theorem grain_to_rice_system :
  (x + y = total_capacity) ∧ (conversion_rate * x + y = rice_obtained) := 
sorry

end NUMINAMATH_GPT_grain_to_rice_system_l602_60201


namespace NUMINAMATH_GPT_polynomial_inequality_l602_60291

-- Define P(x) as a polynomial with non-negative coefficients
def isNonNegativePolynomial (P : Polynomial ℝ) : Prop :=
  ∀ i, P.coeff i ≥ 0

-- The main theorem, which states that for any polynomial P with non-negative coefficients,
-- if P(1) * P(1) ≥ 1, then P(x) * P(1/x) ≥ 1 for all positive x.
theorem polynomial_inequality (P : Polynomial ℝ) (hP : isNonNegativePolynomial P) (hP1 : P.eval 1 * P.eval 1 ≥ 1) :
  ∀ x : ℝ, 0 < x → P.eval x * P.eval (1 / x) ≥ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_inequality_l602_60291


namespace NUMINAMATH_GPT_average_expenditure_Feb_to_July_l602_60249

theorem average_expenditure_Feb_to_July (avg_Jan_to_Jun : ℝ) (spend_Jan : ℝ) (spend_July : ℝ) 
    (total_Jan_to_Jun : avg_Jan_to_Jun = 4200) (spend_Jan_eq : spend_Jan = 1200) (spend_July_eq : spend_July = 1500) :
    (4200 * 6 - 1200 + 1500) / 6 = 4250 :=
by
  sorry

end NUMINAMATH_GPT_average_expenditure_Feb_to_July_l602_60249


namespace NUMINAMATH_GPT_population_increase_l602_60289

theorem population_increase (P : ℕ)
  (birth_rate1_per_1000 : ℕ := 25)
  (death_rate1_per_1000 : ℕ := 12)
  (immigration_rate1 : ℕ := 15000)
  (birth_rate2_per_1000 : ℕ := 30)
  (death_rate2_per_1000 : ℕ := 8)
  (immigration_rate2 : ℕ := 30000)
  (pop_increase1_perc : ℤ := 200)
  (pop_increase2_perc : ℤ := 300) :
  (12 * P - P) / P * 100 = 1100 := by
  sorry

end NUMINAMATH_GPT_population_increase_l602_60289


namespace NUMINAMATH_GPT_ladder_base_length_l602_60223

theorem ladder_base_length {a b c : ℕ} (h1 : c = 13) (h2 : b = 12) (h3 : a^2 + b^2 = c^2) :
  a = 5 := 
by 
  sorry

end NUMINAMATH_GPT_ladder_base_length_l602_60223


namespace NUMINAMATH_GPT_customer_paid_amount_l602_60287

def cost_price : Real := 7239.13
def percentage_increase : Real := 0.15
def selling_price := (1 + percentage_increase) * cost_price

theorem customer_paid_amount :
  selling_price = 8325.00 :=
by
  sorry

end NUMINAMATH_GPT_customer_paid_amount_l602_60287


namespace NUMINAMATH_GPT_no_real_solutions_eq_l602_60279

theorem no_real_solutions_eq (x y : ℝ) :
  x^2 + y^2 - 2 * x + 4 * y + 6 ≠ 0 :=
sorry

end NUMINAMATH_GPT_no_real_solutions_eq_l602_60279


namespace NUMINAMATH_GPT_total_first_year_students_400_l602_60240

theorem total_first_year_students_400 (N : ℕ) (A B C : ℕ) 
  (h1 : A = 80) 
  (h2 : B = 100) 
  (h3 : C = 20) 
  (h4 : A * B = C * N) : 
  N = 400 :=
sorry

end NUMINAMATH_GPT_total_first_year_students_400_l602_60240


namespace NUMINAMATH_GPT_john_newspapers_l602_60252

theorem john_newspapers (N : ℕ) (selling_price buying_price total_cost total_revenue : ℝ) 
  (h1 : selling_price = 2)
  (h2 : buying_price = 0.25 * selling_price)
  (h3 : total_cost = N * buying_price)
  (h4 : total_revenue = 0.8 * N * selling_price)
  (h5 : total_revenue - total_cost = 550) :
  N = 500 := 
by 
  -- actual proof here
  sorry

end NUMINAMATH_GPT_john_newspapers_l602_60252


namespace NUMINAMATH_GPT_division_remainder_false_l602_60208

theorem division_remainder_false :
  ¬(1700 / 500 = 17 / 5 ∧ (1700 % 500 = 3 ∧ 17 % 5 = 2)) := by
  sorry

end NUMINAMATH_GPT_division_remainder_false_l602_60208


namespace NUMINAMATH_GPT_reciprocal_inequalities_l602_60255

theorem reciprocal_inequalities (a b c : ℝ)
  (h1 : -1 < a ∧ a < -2/3)
  (h2 : -1/3 < b ∧ b < 0)
  (h3 : 1 < c) :
  1/c < 1/(b - a) ∧ 1/(b - a) < 1/(a * b) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_inequalities_l602_60255


namespace NUMINAMATH_GPT_pump_fills_tank_without_leak_l602_60265

variable (T : ℝ)
-- Condition: The effective rate with the leak is equal to the rate it takes for both to fill the tank.
def effective_rate_with_leak (T : ℝ) : Prop :=
  1 / T - 1 / 21 = 1 / 3.5

-- Conclude: the time it takes the pump to fill the tank without the leak
theorem pump_fills_tank_without_leak : effective_rate_with_leak T → T = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_pump_fills_tank_without_leak_l602_60265


namespace NUMINAMATH_GPT_find_value_l602_60285

theorem find_value : 3 + 2 * (8 - 3) = 13 := by
  sorry

end NUMINAMATH_GPT_find_value_l602_60285


namespace NUMINAMATH_GPT_percentage_increase_is_50_l602_60202

-- Defining the conditions
def new_wage : ℝ := 51
def original_wage : ℝ := 34
def increase : ℝ := new_wage - original_wage

-- Proving the required percentage increase is 50%
theorem percentage_increase_is_50 :
  (increase / original_wage) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_is_50_l602_60202


namespace NUMINAMATH_GPT_nina_money_l602_60212

theorem nina_money (W M : ℕ) (h1 : 6 * W = M) (h2 : 8 * (W - 2) = M) : M = 48 :=
by
  sorry

end NUMINAMATH_GPT_nina_money_l602_60212


namespace NUMINAMATH_GPT_sample_size_correct_l602_60226

-- Definitions derived from conditions in a)
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def sampled_male_employees : ℕ := 18

-- Theorem stating the mathematically equivalent proof problem
theorem sample_size_correct : 
  ∃ (sample_size : ℕ), sample_size = (total_employees * (sampled_male_employees / male_employees)) :=
sorry

end NUMINAMATH_GPT_sample_size_correct_l602_60226


namespace NUMINAMATH_GPT_part_1_part_2_part_3_l602_60200

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (2 * Real.exp x) / (Real.exp x + 1) + k

theorem part_1 (k : ℝ) :
  (∀ x, f x k = -f (-x) k) → k = -1 :=
sorry

theorem part_2 (m : ℝ) :
  (∀ x > 0, (2 * Real.exp x - 1) / (Real.exp x + 1) ≤ m * (Real.exp x - 1) / (Real.exp x + 1)) → 2 ≤ m :=
sorry

noncomputable def g (x : ℝ) : ℝ := (f x (-1) + 1) / (1 - f x (-1))

theorem part_3 (n : ℝ) :
  (∀ a b c : ℝ, 0 < a ∧ a ≤ n → 0 < b ∧ b ≤ n → 0 < c ∧ c ≤ n → (a + b > c ∧ b + c > a ∧ c + a > b) →
   (g a + g b > g c ∧ g b + g c > g a ∧ g c + g a > g b)) → n = 2 * Real.log 2 :=
sorry

end NUMINAMATH_GPT_part_1_part_2_part_3_l602_60200


namespace NUMINAMATH_GPT_monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l602_60233

variables {f : ℝ → ℝ}

-- Definition that f is monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≤ f x2

-- Definition of the derivative being non-negative everywhere
def non_negative_derivative (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ (deriv f) x

theorem monotonically_increasing_implies_non_negative_derivative (f : ℝ → ℝ) :
  monotonically_increasing f → non_negative_derivative f :=
sorry

theorem non_negative_derivative_not_implies_monotonically_increasing (f : ℝ → ℝ) :
  non_negative_derivative f → ¬ monotonically_increasing f :=
sorry

end NUMINAMATH_GPT_monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l602_60233


namespace NUMINAMATH_GPT_area_in_sq_yds_l602_60230

-- Definitions based on conditions
def side_length_ft : ℕ := 9
def sq_ft_per_sq_yd : ℕ := 9

-- Statement to prove
theorem area_in_sq_yds : (side_length_ft * side_length_ft) / sq_ft_per_sq_yd = 9 :=
by
  sorry

end NUMINAMATH_GPT_area_in_sq_yds_l602_60230


namespace NUMINAMATH_GPT_determine_z_l602_60276

theorem determine_z (z : ℝ) (h1 : ∃ x : ℤ, 3 * (x : ℝ) ^ 2 + 19 * (x : ℝ) - 84 = 0 ∧ (x : ℝ) = ⌊z⌋) (h2 : 4 * (z - ⌊z⌋) ^ 2 - 14 * (z - ⌊z⌋) + 6 = 0) : 
  z = -11 :=
  sorry

end NUMINAMATH_GPT_determine_z_l602_60276


namespace NUMINAMATH_GPT_carsProducedInEurope_l602_60292

-- Definitions of the conditions
def carsProducedInNorthAmerica : ℕ := 3884
def totalCarsProduced : ℕ := 6755

-- Theorem statement
theorem carsProducedInEurope : ∃ (carsProducedInEurope : ℕ), totalCarsProduced = carsProducedInNorthAmerica + carsProducedInEurope ∧ carsProducedInEurope = 2871 := by
  sorry

end NUMINAMATH_GPT_carsProducedInEurope_l602_60292


namespace NUMINAMATH_GPT_prob_five_coins_heads_or_one_tail_l602_60211

theorem prob_five_coins_heads_or_one_tail : 
  (∃ (H T : ℚ), H = 1/32 ∧ T = 31/32 ∧ H + T = 1) ↔ 1 = 1 :=
by sorry

end NUMINAMATH_GPT_prob_five_coins_heads_or_one_tail_l602_60211


namespace NUMINAMATH_GPT_min_value_expression_l602_60295

theorem min_value_expression (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : 4 * a + b = 1) :
  (1 / a) + (4 / b) = 16 := sorry

end NUMINAMATH_GPT_min_value_expression_l602_60295
