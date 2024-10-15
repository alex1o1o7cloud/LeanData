import Mathlib

namespace NUMINAMATH_GPT_buses_in_parking_lot_l806_80689

def initial_buses : ℕ := 7
def additional_buses : ℕ := 6
def total_buses : ℕ := initial_buses + additional_buses

theorem buses_in_parking_lot : total_buses = 13 := by
  sorry

end NUMINAMATH_GPT_buses_in_parking_lot_l806_80689


namespace NUMINAMATH_GPT_equidistant_P_AP_BP_CP_DP_l806_80653

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def distance (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2

def A : Point := ⟨10, 0, 0⟩
def B : Point := ⟨0, -6, 0⟩
def C : Point := ⟨0, 0, 8⟩
def D : Point := ⟨0, 0, 0⟩
def P : Point := ⟨5, -3, 4⟩

theorem equidistant_P_AP_BP_CP_DP :
  distance P A = distance P B ∧ distance P B = distance P C ∧ distance P C = distance P D := 
sorry

end NUMINAMATH_GPT_equidistant_P_AP_BP_CP_DP_l806_80653


namespace NUMINAMATH_GPT_tan_neg_3pi_over_4_eq_one_l806_80669

theorem tan_neg_3pi_over_4_eq_one : Real.tan (-3 * Real.pi / 4) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_tan_neg_3pi_over_4_eq_one_l806_80669


namespace NUMINAMATH_GPT_terminal_side_same_line_37_and_neg143_l806_80657

theorem terminal_side_same_line_37_and_neg143 :
  ∃ k : ℤ, (37 : ℝ) + 180 * k = (-143 : ℝ) :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_terminal_side_same_line_37_and_neg143_l806_80657


namespace NUMINAMATH_GPT_base_addition_l806_80682

theorem base_addition (b : ℕ) (h : b > 1) :
  (2 * b^3 + 3 * b^2 + 8 * b + 4) + (3 * b^3 + 4 * b^2 + 1 * b + 7) = 
  1 * b^4 + 0 * b^3 + 2 * b^2 + 0 * b + 1 → b = 10 :=
by
  intro H
  -- skipping the detailed proof steps
  sorry

end NUMINAMATH_GPT_base_addition_l806_80682


namespace NUMINAMATH_GPT_infinitely_many_sum_of_squares_exceptions_l806_80615

-- Define the predicate for a number being expressible as a sum of two squares
def is_sum_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

-- Define the main theorem
theorem infinitely_many_sum_of_squares_exceptions : 
  ∃ f : ℕ → ℕ, (∀ k : ℕ, is_sum_of_squares (f k)) ∧ (∀ k : ℕ, ¬ is_sum_of_squares (f k - 1)) ∧ (∀ k : ℕ, ¬ is_sum_of_squares (f k + 1)) ∧ (∀ k1 k2 : ℕ, k1 ≠ k2 → f k1 ≠ f k2) :=
sorry

end NUMINAMATH_GPT_infinitely_many_sum_of_squares_exceptions_l806_80615


namespace NUMINAMATH_GPT_find_x_l806_80686

structure Vector2D where
  x : ℝ
  y : ℝ

def vecAdd (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

def vecScale (c : ℝ) (v : Vector2D) : Vector2D :=
  ⟨c * v.x, c * v.y⟩

def areParallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

theorem find_x (x : ℝ)
  (a : Vector2D := ⟨1, 2⟩)
  (b : Vector2D := ⟨x, 1⟩)
  (h : areParallel (vecAdd a (vecScale 2 b)) (vecAdd (vecScale 2 a) (vecScale (-2) b))) :
  x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l806_80686


namespace NUMINAMATH_GPT_problem_1_l806_80660

open Set

variable (R : Set ℝ)
variable (A : Set ℝ := { x | 2 * x^2 - 7 * x + 3 ≤ 0 })
variable (B : Set ℝ := { x | x^2 + a < 0 })

theorem problem_1 (a : ℝ) : (a = -4 → (A ∩ B = { x : ℝ | 1 / 2 ≤ x ∧ x < 2 } ∧ A ∪ B = { x : ℝ | -2 < x ∧ x ≤ 3 })) ∧
  ((compl A ∩ B = B) → a ≥ -2) := by
  sorry

end NUMINAMATH_GPT_problem_1_l806_80660


namespace NUMINAMATH_GPT_solve_Cheolsu_weight_l806_80661

def Cheolsu_weight (C M F : ℝ) :=
  (C + M + F) / 3 = M ∧
  C = (2 / 3) * M ∧
  F = 72

theorem solve_Cheolsu_weight {C M F : ℝ} (h : Cheolsu_weight C M F) : C = 36 :=
by
  sorry

end NUMINAMATH_GPT_solve_Cheolsu_weight_l806_80661


namespace NUMINAMATH_GPT_megan_removed_albums_l806_80614

theorem megan_removed_albums :
  ∀ (albums_in_cart : ℕ) (songs_per_album : ℕ) (total_songs_bought : ℕ),
    albums_in_cart = 8 →
    songs_per_album = 7 →
    total_songs_bought = 42 →
    albums_in_cart - (total_songs_bought / songs_per_album) = 2 :=
by
  intros albums_in_cart songs_per_album total_songs_bought h1 h2 h3
  sorry

end NUMINAMATH_GPT_megan_removed_albums_l806_80614


namespace NUMINAMATH_GPT_total_popsicle_sticks_l806_80685

def Gino_popsicle_sticks : ℕ := 63
def My_popsicle_sticks : ℕ := 50
def Nick_popsicle_sticks : ℕ := 82

theorem total_popsicle_sticks : Gino_popsicle_sticks + My_popsicle_sticks + Nick_popsicle_sticks = 195 := by
  sorry

end NUMINAMATH_GPT_total_popsicle_sticks_l806_80685


namespace NUMINAMATH_GPT_minimum_value_of_fraction_l806_80604

theorem minimum_value_of_fraction (x : ℝ) (h : x > 0) : 
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 - 1 ∧ ∀ y, y = (x^2 + x + 3) / (x + 1) -> y ≥ m :=
sorry

end NUMINAMATH_GPT_minimum_value_of_fraction_l806_80604


namespace NUMINAMATH_GPT_x_y_result_l806_80677

noncomputable def x_y_value (x y : ℝ) : ℝ := x + y

theorem x_y_result (x y : ℝ) 
  (h1 : x + Real.cos y = 3009) 
  (h2 : x + 3009 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) : 
  x_y_value x y = 3009 + Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_x_y_result_l806_80677


namespace NUMINAMATH_GPT_climb_stairs_l806_80629

noncomputable def u (n : ℕ) : ℝ :=
  let Φ := (1 + Real.sqrt 5) / 2
  let φ := (1 - Real.sqrt 5) / 2
  let A := (1 + Real.sqrt 5) / (2 * Real.sqrt 5)
  let B := (Real.sqrt 5 - 1) / (2 * Real.sqrt 5)
  A * (Φ ^ n) + B * (φ ^ n)

theorem climb_stairs (n : ℕ) (hn : n ≥ 1) : u n = A * (Φ ^ n) + B * (φ ^ n) := sorry

end NUMINAMATH_GPT_climb_stairs_l806_80629


namespace NUMINAMATH_GPT_right_triangle_ratio_segments_l806_80642

theorem right_triangle_ratio_segments (a b c r s : ℝ) (h : a^2 + b^2 = c^2) (h_drop : r + s = c) (a_to_b_ratio : 2 * b = 5 * a) : r / s = 4 / 25 :=
sorry

end NUMINAMATH_GPT_right_triangle_ratio_segments_l806_80642


namespace NUMINAMATH_GPT_probability_both_in_photo_correct_l806_80624

noncomputable def probability_both_in_photo (lap_time_Emily : ℕ) (lap_time_John : ℕ) (observation_start : ℕ) (observation_end : ℕ) : ℚ := 
  let GCD := Nat.gcd lap_time_Emily lap_time_John
  let cycle_time := lap_time_Emily * lap_time_John / GCD
  let visible_time := 2 * min (lap_time_Emily / 3) (lap_time_John / 3)
  visible_time / cycle_time

theorem probability_both_in_photo_correct : 
  probability_both_in_photo 100 75 900 1200 = 1 / 6 :=
by
  -- Use previous calculations and observations here to construct the proof.
  -- sorry is used to indicate that proof steps are omitted.
  sorry

end NUMINAMATH_GPT_probability_both_in_photo_correct_l806_80624


namespace NUMINAMATH_GPT_ashley_family_spent_30_l806_80696

def cost_of_child_ticket : ℝ := 4.25
def cost_of_adult_ticket : ℝ := cost_of_child_ticket + 3.25
def discount : ℝ := 2.00
def num_adult_tickets : ℕ := 2
def num_child_tickets : ℕ := 4

def total_cost : ℝ := num_adult_tickets * cost_of_adult_ticket + num_child_tickets * cost_of_child_ticket - discount

theorem ashley_family_spent_30 :
  total_cost = 30.00 :=
sorry

end NUMINAMATH_GPT_ashley_family_spent_30_l806_80696


namespace NUMINAMATH_GPT_student_weight_loss_l806_80664

variables (S R L : ℕ)

theorem student_weight_loss :
  S = 75 ∧ S + R = 110 ∧ S - L = 2 * R → L = 5 :=
by
  sorry

end NUMINAMATH_GPT_student_weight_loss_l806_80664


namespace NUMINAMATH_GPT_regular_hexagon_perimeter_is_30_l806_80675

-- Define a regular hexagon with each side length 5 cm
def regular_hexagon_side_length : ℝ := 5

-- Define the perimeter of a regular hexagon
def regular_hexagon_perimeter (side_length : ℝ) : ℝ := 6 * side_length

-- State the theorem about the perimeter of a regular hexagon with side length 5 cm
theorem regular_hexagon_perimeter_is_30 : regular_hexagon_perimeter regular_hexagon_side_length = 30 := 
by 
  sorry

end NUMINAMATH_GPT_regular_hexagon_perimeter_is_30_l806_80675


namespace NUMINAMATH_GPT_dimitri_weekly_calories_l806_80644

-- Define the calories for each type of burger
def calories_burger_a : ℕ := 350
def calories_burger_b : ℕ := 450
def calories_burger_c : ℕ := 550

-- Define the daily consumption of each type of burger
def daily_consumption_a : ℕ := 2
def daily_consumption_b : ℕ := 1
def daily_consumption_c : ℕ := 3

-- Define the duration in days
def duration_in_days : ℕ := 7

-- Define the total number of calories Dimitri consumes in a week
noncomputable def total_weekly_calories : ℕ :=
  (daily_consumption_a * calories_burger_a +
   daily_consumption_b * calories_burger_b +
   daily_consumption_c * calories_burger_c) * duration_in_days

theorem dimitri_weekly_calories : total_weekly_calories = 19600 := 
by 
  sorry

end NUMINAMATH_GPT_dimitri_weekly_calories_l806_80644


namespace NUMINAMATH_GPT_each_person_eats_3_Smores_l806_80688

-- Definitions based on the conditions in (a)
def people := 8
def cost_per_4_Smores := 3
def total_cost := 18

-- The statement we need to prove
theorem each_person_eats_3_Smores (h1 : total_cost = people * (cost_per_4_Smores * 4 / 3)) :
  (total_cost / cost_per_4_Smores) * 4 / people = 3 :=
by
  sorry

end NUMINAMATH_GPT_each_person_eats_3_Smores_l806_80688


namespace NUMINAMATH_GPT_integer_solutions_count_l806_80697

theorem integer_solutions_count :
  ∃ (count : ℤ), (∀ (a : ℤ), 
  (∃ x : ℤ, x^2 + a * x + 8 * a = 0) ↔ count = 8) :=
sorry

end NUMINAMATH_GPT_integer_solutions_count_l806_80697


namespace NUMINAMATH_GPT_race_distance_l806_80674

theorem race_distance {d a b c : ℝ} 
    (h1 : d / a = (d - 25) / b)
    (h2 : d / b = (d - 15) / c)
    (h3 : d / a = (d - 35) / c) :
  d = 75 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_l806_80674


namespace NUMINAMATH_GPT_thompson_class_average_l806_80698

theorem thompson_class_average
  (n : ℕ) (initial_avg : ℚ) (final_avg : ℚ) (bridget_index : ℕ) (first_n_score_sum : ℚ)
  (total_students : ℕ) (final_score_sum : ℚ)
  (h1 : n = 17) -- Number of students initially graded
  (h2 : initial_avg = 76) -- Average score of the first 17 students
  (h3 : final_avg = 78) -- Average score after adding Bridget's test
  (h4 : bridget_index = 18) -- Total number of students
  (h5 : total_students = 18) -- Total number of students
  (h6 : first_n_score_sum = n * initial_avg) -- Total score of the first 17 students
  (h7 : final_score_sum = total_students * final_avg) -- Total score of the 18 students):
  -- Bridget's score
  (bridgets_score : ℚ) :
  bridgets_score = final_score_sum - first_n_score_sum :=
sorry

end NUMINAMATH_GPT_thompson_class_average_l806_80698


namespace NUMINAMATH_GPT_tens_digit_N_pow_20_is_7_hundreds_digit_N_pow_200_is_3_l806_80656

def tens_digit_N_pow_20 (N : ℕ) : Nat :=
if (N % 2 = 0 ∧ N % 10 ≠ 0) then
  if (N % 5 = 1 ∨ N % 5 = 2 ∨ N % 5 = 3 ∨ N % 5 = 4) then
    (N^20 % 100) / 10  -- tens digit of last two digits
  else
    sorry  -- N should be in form of 5k±1 or 5k±2
else
  sorry  -- N not satisfying conditions

def hundreds_digit_N_pow_200 (N : ℕ) : Nat :=
if (N % 2 = 0 ∧ N % 10 ≠ 0) then
  (N^200 % 1000) / 100  -- hundreds digit of the last three digits
else
  sorry  -- N not satisfying conditions

theorem tens_digit_N_pow_20_is_7 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) : 
  tens_digit_N_pow_20 N = 7 := sorry

theorem hundreds_digit_N_pow_200_is_3 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) : 
  hundreds_digit_N_pow_200 N = 3 := sorry

end NUMINAMATH_GPT_tens_digit_N_pow_20_is_7_hundreds_digit_N_pow_200_is_3_l806_80656


namespace NUMINAMATH_GPT_bacteria_reaches_final_in_24_hours_l806_80691

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 200

-- Define the final number of bacteria
def final_bacteria : ℕ := 16200

-- Define the tripling period in hours
def tripling_period : ℕ := 6

-- Define the tripling factor
def tripling_factor : ℕ := 3

-- Define the number of hours needed to reach final number of bacteria
def hours_to_reach_final_bacteria : ℕ := 24

-- Define a function that models the number of bacteria after t hours
def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * tripling_factor^((t / tripling_period))

-- Main statement of the problem: prove that the number of bacteria is 16200 after 24 hours
theorem bacteria_reaches_final_in_24_hours :
  bacteria_after hours_to_reach_final_bacteria = final_bacteria :=
sorry

end NUMINAMATH_GPT_bacteria_reaches_final_in_24_hours_l806_80691


namespace NUMINAMATH_GPT_find_number_l806_80679

theorem find_number (N : ℕ) :
  let sum := 555 + 445
  let difference := 555 - 445
  let divisor := sum
  let quotient := 2 * difference
  let remainder := 70
  N = divisor * quotient + remainder -> N = 220070 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l806_80679


namespace NUMINAMATH_GPT_intersection_of_lines_l806_80671

theorem intersection_of_lines :
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 2 * y = -7 * x - 2 ∧ x = -18 / 17 ∧ y = 46 / 17 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l806_80671


namespace NUMINAMATH_GPT_maxwell_walking_speed_l806_80636

variable (distance : ℕ) (brad_speed : ℕ) (maxwell_time : ℕ) (brad_time : ℕ) (maxwell_speed : ℕ)

-- Given conditions
def conditions := distance = 54 ∧ brad_speed = 6 ∧ maxwell_time = 6 ∧ brad_time = 5

-- Problem statement
theorem maxwell_walking_speed (h : conditions distance brad_speed maxwell_time brad_time) : maxwell_speed = 4 := sorry

end NUMINAMATH_GPT_maxwell_walking_speed_l806_80636


namespace NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l806_80694

theorem product_of_areas_eq_square_of_volume
    (a b c : ℝ)
    (bottom_area : ℝ) (side_area : ℝ) (front_area : ℝ)
    (volume : ℝ)
    (h1 : bottom_area = a * b)
    (h2 : side_area = b * c)
    (h3 : front_area = c * a)
    (h4 : volume = a * b * c) :
    bottom_area * side_area * front_area = volume ^ 2 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_product_of_areas_eq_square_of_volume_l806_80694


namespace NUMINAMATH_GPT_num_males_selected_l806_80650

theorem num_males_selected (total_male total_female total_selected : ℕ)
                           (h_male : total_male = 56)
                           (h_female : total_female = 42)
                           (h_selected : total_selected = 28) :
  (total_male * total_selected) / (total_male + total_female) = 16 := 
by {
  sorry
}

end NUMINAMATH_GPT_num_males_selected_l806_80650


namespace NUMINAMATH_GPT_total_expenditure_l806_80630

-- Define the conditions.
def singers : ℕ := 30
def current_robes : ℕ := 12
def robe_cost : ℕ := 2

-- Define the statement.
theorem total_expenditure (singers current_robes robe_cost : ℕ) : 
  (singers - current_robes) * robe_cost = 36 := by
  sorry

end NUMINAMATH_GPT_total_expenditure_l806_80630


namespace NUMINAMATH_GPT_trader_loses_l806_80687

theorem trader_loses 
  (l_1 l_2 q : ℝ) 
  (h1 : l_1 ≠ l_2) 
  (p_1 p_2 : ℝ) 
  (h2 : p_1 = q * (l_2 / l_1)) 
  (h3 : p_2 = q * (l_1 / l_2)) :
  p_1 + p_2 > 2 * q :=
by {
  sorry
}

end NUMINAMATH_GPT_trader_loses_l806_80687


namespace NUMINAMATH_GPT_PQ_R_exist_l806_80635

theorem PQ_R_exist :
  ∃ P Q R : ℚ, 
    (P = -3/5) ∧ (Q = -1) ∧ (R = 13/5) ∧
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
    (x^2 - 10)/((x - 1)*(x - 4)*(x - 6)) = P/(x - 1) + Q/(x - 4) + R/(x - 6)) :=
by
  sorry

end NUMINAMATH_GPT_PQ_R_exist_l806_80635


namespace NUMINAMATH_GPT_new_fig_sides_l806_80654

def hexagon_side := 1
def triangle_side := 1
def hexagon_sides := 6
def triangle_sides := 3
def joined_sides := 2
def total_initial_sides := hexagon_sides + triangle_sides
def lost_sides := joined_sides * 2
def new_shape_sides := total_initial_sides - lost_sides

theorem new_fig_sides : new_shape_sides = 5 := by
  sorry

end NUMINAMATH_GPT_new_fig_sides_l806_80654


namespace NUMINAMATH_GPT_number_of_men_in_second_group_l806_80639

variable (n m : ℕ)

theorem number_of_men_in_second_group 
  (h1 : 42 * 18 = n)
  (h2 : n = m * 28) : 
  m = 27 := by
  sorry

end NUMINAMATH_GPT_number_of_men_in_second_group_l806_80639


namespace NUMINAMATH_GPT_min_ab_minus_cd_l806_80626

theorem min_ab_minus_cd (a b c d : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9) (h5 : a^2 + b^2 + c^2 + d^2 = 21) : ab - cd ≥ 2 := sorry

end NUMINAMATH_GPT_min_ab_minus_cd_l806_80626


namespace NUMINAMATH_GPT_tan_difference_l806_80632

theorem tan_difference (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h₁ : Real.sin α = 3 / 5) (h₂ : Real.cos β = 12 / 13) : 
    Real.tan (α - β) = 16 / 63 := 
by
  sorry

end NUMINAMATH_GPT_tan_difference_l806_80632


namespace NUMINAMATH_GPT_can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2_l806_80648

-- Question and conditions
def side_length_of_square (A : ℝ) := A = 400
def area_of_rect (A : ℝ) := A = 300
def ratio_of_rect (length width : ℝ) := 3 * width = 2 * length

-- Prove that Li can cut a rectangle with area 300 from the square with area 400
theorem can_cut_rectangle_with_area_300 
  (a : ℝ) (h1 : side_length_of_square a)
  (length width : ℝ)
  (ha : a ^ 2 = 400) (har : length * width = 300) :
  length ≤ a ∧ width ≤ a :=
by
  sorry

-- Prove that Li cannot cut a rectangle with ratio 3:2 from the square
theorem cannot_cut_rectangle_with_ratio_3_2 (a : ℝ)
  (h1 : side_length_of_square a)
  (length width : ℝ)
  (har : area_of_rect (length * width))
  (hratio : ratio_of_rect length width)
  (ha : a ^ 2 = 400) :
  ¬(length ≤ a ∧ width ≤ a) :=
by
  sorry

end NUMINAMATH_GPT_can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2_l806_80648


namespace NUMINAMATH_GPT_find_q_l806_80699

variable (p q : ℝ) (hp : p > 1) (hq : q > 1) (h_cond1 : 1 / p + 1 / q = 1) (h_cond2 : p * q = 9)

theorem find_q : q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_find_q_l806_80699


namespace NUMINAMATH_GPT_total_production_by_june_l806_80607

def initial_production : ℕ := 10

def common_ratio : ℕ := 3

def production_june : ℕ :=
  let a := initial_production
  let r := common_ratio
  a * ((r^6 - 1) / (r - 1))

theorem total_production_by_june : production_june = 3640 :=
by sorry

end NUMINAMATH_GPT_total_production_by_june_l806_80607


namespace NUMINAMATH_GPT_price_of_fruits_l806_80622

theorem price_of_fruits
  (x y : ℝ)
  (h1 : 9 * x + 10 * y = 73.8)
  (h2 : 17 * x + 6 * y = 69.8)
  (hx : x = 2.2)
  (hy : y = 5.4) : 
  9 * 2.2 + 10 * 5.4 = 73.8 ∧ 17 * 2.2 + 6 * 5.4 = 69.8 :=
by
  sorry

end NUMINAMATH_GPT_price_of_fruits_l806_80622


namespace NUMINAMATH_GPT_parabola_directrix_l806_80680

theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) (O D : ℝ × ℝ) :
  A ≠ B →
  O = (0, 0) →
  D = (1, 2) →
  (∃ k, k = ((2:ℝ) - 0) / ((1:ℝ) - 0) ∧ k = 2) →
  (∃ k, k = - 1 / 2) →
  (∀ x y, y^2 = 2 * p * x) →
  p = 5 / 2 →
  O.1 * A.1 + O.2 * A.2 = 0 →
  O.1 * B.1 + O.2 * B.2 = 0 →
  A.1 * B.1 + A.2 * B.2 = 0 →
  (∃ k, (y - 2) = k * (x - 1) ∧ (A.1 * B.1) = 25 ∧ (A.1 + B.1) = 10 + 8 * p) →
  ∃ dir_eq, dir_eq = -5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l806_80680


namespace NUMINAMATH_GPT_admin_in_sample_l806_80641

-- Define the total number of staff members
def total_staff : ℕ := 200

-- Define the number of administrative personnel
def admin_personnel : ℕ := 24

-- Define the sample size taken
def sample_size : ℕ := 50

-- Goal: Prove the number of administrative personnel in the sample
theorem admin_in_sample : 
  (admin_personnel : ℚ) / (total_staff : ℚ) * (sample_size : ℚ) = 6 := 
by
  sorry

end NUMINAMATH_GPT_admin_in_sample_l806_80641


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l806_80618

theorem necessary_and_sufficient_condition (a : ℝ) : (a > 0) ↔ (a + 1 / a ≥ 2) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l806_80618


namespace NUMINAMATH_GPT_sin_y_eq_neg_one_l806_80692

noncomputable def α := Real.arccos (-1 / 5)

theorem sin_y_eq_neg_one (x y z : ℝ) (h1 : x = y - α) (h2 : z = y + α)
  (h3 : (2 + Real.sin x) * (2 + Real.sin z) = (2 + Real.sin y) ^ 2) : Real.sin y = -1 :=
sorry

end NUMINAMATH_GPT_sin_y_eq_neg_one_l806_80692


namespace NUMINAMATH_GPT_find_point_C_coordinates_l806_80617

/-- Given vertices A and B of a triangle, and the centroid G of the triangle, 
prove the coordinates of the third vertex C. 
-/
theorem find_point_C_coordinates : 
  ∀ (x y : ℝ),
  let A := (2, 3)
  let B := (-4, -2)
  let G := (2, -1)
  (2 + -4 + x) / 3 = 2 →
  (3 + -2 + y) / 3 = -1 →
  (x, y) = (8, -4) :=
by
  intro x y A B G h1 h2
  sorry

end NUMINAMATH_GPT_find_point_C_coordinates_l806_80617


namespace NUMINAMATH_GPT_binomial_ratio_l806_80663

theorem binomial_ratio (n : ℕ) (r : ℕ) :
  (Nat.choose n r : ℚ) / (Nat.choose n (r+1) : ℚ) = 1 / 2 →
  (Nat.choose n (r+1) : ℚ) / (Nat.choose n (r+2) : ℚ) = 2 / 3 →
  n = 14 :=
by
  sorry

end NUMINAMATH_GPT_binomial_ratio_l806_80663


namespace NUMINAMATH_GPT_sum_arithmetic_series_eq_499500_l806_80673

theorem sum_arithmetic_series_eq_499500 :
  let a1 := 1
  let an := 999
  let n := 999
  let d := 1
  (n * (a1 + an) / 2) = 499500 := by {
  let a1 := 1
  let an := 999
  let n := 999
  let d := 1
  show (n * (a1 + an) / 2) = 499500
  sorry
}

end NUMINAMATH_GPT_sum_arithmetic_series_eq_499500_l806_80673


namespace NUMINAMATH_GPT_baker_bakes_25_hours_per_week_mon_to_fri_l806_80631

-- Define the conditions
def loaves_per_hour_per_oven := 5
def number_of_ovens := 4
def weekend_baking_hours_per_day := 2
def total_weeks := 3
def total_loaves := 1740

-- Calculate the loaves per hour
def loaves_per_hour := loaves_per_hour_per_oven * number_of_ovens

-- Calculate the weekend baking hours in one week
def weekend_baking_hours_per_week := weekend_baking_hours_per_day * 2

-- Calculate the loaves baked on weekends in one week
def loaves_on_weekends_per_week := loaves_per_hour * weekend_baking_hours_per_week

-- Calculate the total loaves baked on weekends in 3 weeks
def loaves_on_weekends_total := loaves_on_weekends_per_week * total_weeks

-- Calculate the loaves baked from Monday to Friday in 3 weeks
def loaves_on_weekdays_total := total_loaves - loaves_on_weekends_total

-- Calculate the total hours baked from Monday to Friday in 3 weeks
def weekday_baking_hours_total := loaves_on_weekdays_total / loaves_per_hour

-- Calculate the number of hours baked from Monday to Friday in one week
def weekday_baking_hours_per_week := weekday_baking_hours_total / total_weeks

-- Proof statement
theorem baker_bakes_25_hours_per_week_mon_to_fri :
  weekday_baking_hours_per_week = 25 :=
by
  sorry

end NUMINAMATH_GPT_baker_bakes_25_hours_per_week_mon_to_fri_l806_80631


namespace NUMINAMATH_GPT_average_first_14_even_numbers_l806_80672

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun x => 2 * (x + 1))

theorem average_first_14_even_numbers :
  let even_nums := first_n_even_numbers 14
  (even_nums.sum / even_nums.length = 15) :=
by
  sorry

end NUMINAMATH_GPT_average_first_14_even_numbers_l806_80672


namespace NUMINAMATH_GPT_sum_of_arithmetic_progression_l806_80684

theorem sum_of_arithmetic_progression 
  (a d : ℚ) 
  (S : ℕ → ℚ)
  (h_sum_15 : S 15 = 150)
  (h_sum_75 : S 75 = 30)
  (h_arith_sum : ∀ n, S n = (n / 2) * (2 * a + (n - 1) * d)) :
  S 90 = -180 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_progression_l806_80684


namespace NUMINAMATH_GPT_find_non_negative_integer_pairs_l806_80655

theorem find_non_negative_integer_pairs (m n : ℕ) :
  3 * 2^m + 1 = n^2 ↔ (m = 0 ∧ n = 2) ∨ (m = 3 ∧ n = 5) ∨ (m = 4 ∧ n = 7) := by
  sorry

end NUMINAMATH_GPT_find_non_negative_integer_pairs_l806_80655


namespace NUMINAMATH_GPT_area_CDM_l806_80678

noncomputable def AC := 8
noncomputable def BC := 15
noncomputable def AB := 17
noncomputable def M := (AC + BC) / 2
noncomputable def AD := 17
noncomputable def BD := 17

theorem area_CDM (h₁ : AC = 8)
                 (h₂ : BC = 15)
                 (h₃ : AB = 17)
                 (h₄ : AD = 17)
                 (h₅ : BD = 17)
                 : ∃ (m n p : ℕ),
                   m = 121 ∧
                   n = 867 ∧
                   p = 136 ∧
                   m + n + p = 1124 ∧
                   ∃ (area_CDM : ℚ), 
                   area_CDM = (121 * Real.sqrt 867) / 136 :=
by
  sorry

end NUMINAMATH_GPT_area_CDM_l806_80678


namespace NUMINAMATH_GPT_population_in_2050_l806_80643

def population : ℕ → ℕ := sorry

theorem population_in_2050 : population 2050 = 2700 :=
by
  -- sorry statement to skip the proof
  sorry

end NUMINAMATH_GPT_population_in_2050_l806_80643


namespace NUMINAMATH_GPT_simplify_expression_l806_80623

theorem simplify_expression (a : ℤ) : 7 * a - 3 * a = 4 * a :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l806_80623


namespace NUMINAMATH_GPT_cookies_to_milk_l806_80625

theorem cookies_to_milk (milk_quarts : ℕ) (cookies : ℕ) (cups_in_quart : ℕ) 
  (H : milk_quarts = 3) (C : cookies = 24) (Q : cups_in_quart = 4) : 
  ∃ x : ℕ, x = 3 ∧ ∀ y : ℕ, y = 6 → x = (milk_quarts * cups_in_quart * y) / cookies := 
by {
  sorry
}

end NUMINAMATH_GPT_cookies_to_milk_l806_80625


namespace NUMINAMATH_GPT_find_length_y_l806_80652

def length_y (AO OC DO BO BD y : ℝ) : Prop := 
  AO = 3 ∧ OC = 11 ∧ DO = 3 ∧ BO = 6 ∧ BD = 7 ∧ y = 3 * Real.sqrt 91

theorem find_length_y : length_y 3 11 3 6 7 (3 * Real.sqrt 91) :=
by
  sorry

end NUMINAMATH_GPT_find_length_y_l806_80652


namespace NUMINAMATH_GPT_original_price_of_iWatch_l806_80695

theorem original_price_of_iWatch (P : ℝ) (h1 : 800 > 0) (h2 : P > 0)
    (h3 : 680 + 0.90 * P > 0) (h4 : 0.98 * (680 + 0.90 * P) = 931) :
    P = 300 := by
  sorry

end NUMINAMATH_GPT_original_price_of_iWatch_l806_80695


namespace NUMINAMATH_GPT_water_pump_rate_l806_80640

theorem water_pump_rate (hourly_rate : ℕ) (minutes : ℕ) (calculated_gallons : ℕ) : 
  hourly_rate = 600 → minutes = 30 → calculated_gallons = (hourly_rate * (minutes / 60)) → 
  calculated_gallons = 300 :=
by 
  sorry

end NUMINAMATH_GPT_water_pump_rate_l806_80640


namespace NUMINAMATH_GPT_sum_of_mixed_numbers_is_between_18_and_19_l806_80647

theorem sum_of_mixed_numbers_is_between_18_and_19 :
  let a := 2 + 3 / 8;
  let b := 4 + 1 / 3;
  let c := 5 + 2 / 21;
  let d := 6 + 1 / 11;
  18 < a + b + c + d ∧ a + b + c + d < 19 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_mixed_numbers_is_between_18_and_19_l806_80647


namespace NUMINAMATH_GPT_tens_digit_of_desired_number_is_one_l806_80668

def productOfDigits (n : Nat) : Nat :=
  match n / 10, n % 10 with
  | a, b => a * b

def sumOfDigits (n : Nat) : Nat :=
  match n / 10, n % 10 with
  | a, b => a + b

def isDesiredNumber (N : Nat) : Prop :=
  N < 100 ∧ N ≥ 10 ∧ N = (productOfDigits N)^2 + sumOfDigits N

theorem tens_digit_of_desired_number_is_one (N : Nat) (h : isDesiredNumber N) : N / 10 = 1 :=
  sorry

end NUMINAMATH_GPT_tens_digit_of_desired_number_is_one_l806_80668


namespace NUMINAMATH_GPT_no_valid_coloring_l806_80658

theorem no_valid_coloring (colors : Fin 4 → Prop) (board : Fin 5 → Fin 5 → Fin 4) :
  (∀ i j : Fin 5, ∃ c1 c2 c3 : Fin 4, 
    (c1 ≠ c2) ∧ (c2 ≠ c3) ∧ (c1 ≠ c3) ∧ 
    (board i j = c1 ∨ board i j = c2 ∨ board i j = c3)) → False :=
by
  sorry

end NUMINAMATH_GPT_no_valid_coloring_l806_80658


namespace NUMINAMATH_GPT_remainder_8_pow_900_mod_29_l806_80662

theorem remainder_8_pow_900_mod_29 : 8^900 % 29 = 7 :=
by sorry

end NUMINAMATH_GPT_remainder_8_pow_900_mod_29_l806_80662


namespace NUMINAMATH_GPT_time_to_destination_l806_80666

theorem time_to_destination (speed_ratio : ℕ) (mr_harris_time : ℕ) 
  (distance_multiple : ℕ) (h1 : speed_ratio = 3) 
  (h2 : mr_harris_time = 3) 
  (h3 : distance_multiple = 5) : 
  (mr_harris_time / speed_ratio) * distance_multiple = 5 := by
  sorry

end NUMINAMATH_GPT_time_to_destination_l806_80666


namespace NUMINAMATH_GPT_pentagonal_number_formula_l806_80601

def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n + 1)) / 2

theorem pentagonal_number_formula (n : ℕ) :
  pentagonal_number n = (n * (3 * n + 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_pentagonal_number_formula_l806_80601


namespace NUMINAMATH_GPT_sale_price_is_correct_l806_80605

def original_price : ℝ := 100
def percentage_decrease : ℝ := 0.30
def sale_price : ℝ := original_price * (1 - percentage_decrease)

theorem sale_price_is_correct : sale_price = 70 := by
  sorry

end NUMINAMATH_GPT_sale_price_is_correct_l806_80605


namespace NUMINAMATH_GPT_readers_both_l806_80681

-- Define the given conditions
def total_readers : ℕ := 250
def readers_S : ℕ := 180
def readers_L : ℕ := 88

-- Define the proof statement
theorem readers_both : (readers_S + readers_L - total_readers = 18) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_readers_both_l806_80681


namespace NUMINAMATH_GPT_infinitesimal_alpha_as_t_to_zero_l806_80690

open Real

noncomputable def alpha (t : ℝ) : ℝ × ℝ :=
  (t, sin t)

theorem infinitesimal_alpha_as_t_to_zero : 
  ∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, abs t < δ → abs (alpha t).fst + abs (alpha t).snd < ε := by
  sorry

end NUMINAMATH_GPT_infinitesimal_alpha_as_t_to_zero_l806_80690


namespace NUMINAMATH_GPT_alice_pints_wednesday_l806_80646

-- Initial conditions
def pints_sunday : ℕ := 4
def pints_monday : ℕ := 3 * pints_sunday
def pints_tuesday : ℕ := pints_monday / 3
def total_pints_before_return : ℕ := pints_sunday + pints_monday + pints_tuesday
def pints_returned_wednesday : ℕ := pints_tuesday / 2
def pints_wednesday : ℕ := total_pints_before_return - pints_returned_wednesday

-- The proof statement
theorem alice_pints_wednesday : pints_wednesday = 18 :=
by
  sorry

end NUMINAMATH_GPT_alice_pints_wednesday_l806_80646


namespace NUMINAMATH_GPT_gcd_8m_6n_l806_80616

theorem gcd_8m_6n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : Nat.gcd m n = 7) : Nat.gcd (8 * m) (6 * n) = 14 := 
by
  sorry

end NUMINAMATH_GPT_gcd_8m_6n_l806_80616


namespace NUMINAMATH_GPT_sun_salutations_per_year_l806_80627

theorem sun_salutations_per_year :
  let poses_per_day := 5
  let days_per_week := 5
  let weeks_per_year := 52
  poses_per_day * days_per_week * weeks_per_year = 1300 :=
by
  sorry

end NUMINAMATH_GPT_sun_salutations_per_year_l806_80627


namespace NUMINAMATH_GPT_division_of_fractions_l806_80612

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end NUMINAMATH_GPT_division_of_fractions_l806_80612


namespace NUMINAMATH_GPT_power_rule_example_l806_80603

variable {R : Type*} [Ring R] (a b : R)

theorem power_rule_example : (a * b^3) ^ 2 = a^2 * b^6 :=
sorry

end NUMINAMATH_GPT_power_rule_example_l806_80603


namespace NUMINAMATH_GPT_ticket_cost_at_30_years_l806_80611

noncomputable def initial_cost : ℝ := 1000000
noncomputable def halving_period_years : ℕ := 10
noncomputable def halving_factor : ℝ := 0.5

def cost_after_n_years (initial_cost : ℝ) (halving_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_cost * halving_factor ^ (years / period)

theorem ticket_cost_at_30_years (initial_cost halving_factor : ℝ) (years period: ℕ) 
  (h_initial_cost : initial_cost = 1000000)
  (h_halving_factor : halving_factor = 0.5)
  (h_years : years = 30)
  (h_period : period = halving_period_years) : 
  cost_after_n_years initial_cost halving_factor years period = 125000 :=
by 
  sorry

end NUMINAMATH_GPT_ticket_cost_at_30_years_l806_80611


namespace NUMINAMATH_GPT_proof_problem_l806_80610

theorem proof_problem
  (x y : ℚ)
  (h1 : 4 * x + 2 * y = 12)
  (h2 : 2 * x + 4 * y = 16) :
  20 * x^2 + 24 * x * y + 20 * y^2 = 3280 / 9 :=
sorry

end NUMINAMATH_GPT_proof_problem_l806_80610


namespace NUMINAMATH_GPT_fifth_equation_l806_80633

theorem fifth_equation :
  (5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81) := 
by sorry

end NUMINAMATH_GPT_fifth_equation_l806_80633


namespace NUMINAMATH_GPT_value_of_a_is_2_l806_80638

def point_symmetric_x_axis (a b : ℝ) : Prop :=
  (2 * a + b = 1 - 2 * b) ∧ (a - 2 * b = -(-2 * a - b - 1))

theorem value_of_a_is_2 (a b : ℝ) (h : point_symmetric_x_axis a b) : a = 2 :=
by sorry

end NUMINAMATH_GPT_value_of_a_is_2_l806_80638


namespace NUMINAMATH_GPT_grey_eyed_black_haired_students_l806_80676

theorem grey_eyed_black_haired_students (total_students black_haired green_eyed_red_haired grey_eyed : ℕ) 
(h_total : total_students = 60) 
(h_black_haired : black_haired = 35) 
(h_green_eyed_red_haired : green_eyed_red_haired = 20) 
(h_grey_eyed : grey_eyed = 25) : 
grey_eyed - (total_students - black_haired - green_eyed_red_haired) = 20 :=
by
  sorry

end NUMINAMATH_GPT_grey_eyed_black_haired_students_l806_80676


namespace NUMINAMATH_GPT_principal_amount_l806_80667

noncomputable def exponential (r t : ℝ) :=
  Real.exp (r * t)

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  A = 5673981 ∧ r = 0.1125 ∧ t = 7.5 ∧ P = 2438978.57 →
  P = A / exponential r t := 
by
  intros h
  sorry

end NUMINAMATH_GPT_principal_amount_l806_80667


namespace NUMINAMATH_GPT_cubed_expression_l806_80613

theorem cubed_expression (a : ℝ) (h : (a + 1/a)^2 = 4) : a^3 + 1/a^3 = 2 ∨ a^3 + 1/a^3 = -2 :=
sorry

end NUMINAMATH_GPT_cubed_expression_l806_80613


namespace NUMINAMATH_GPT_percentage_of_copper_buttons_l806_80693

-- Definitions for conditions
def total_items : ℕ := 100
def pin_percentage : ℕ := 30
def button_percentage : ℕ := 100 - pin_percentage
def brass_button_percentage : ℕ := 60
def copper_button_percentage : ℕ := 100 - brass_button_percentage

-- Theorem statement proving the question
theorem percentage_of_copper_buttons (h1 : pin_percentage = 30)
  (h2 : button_percentage = total_items - pin_percentage)
  (h3 : brass_button_percentage = 60)
  (h4 : copper_button_percentage = total_items - brass_button_percentage) :
  (button_percentage * copper_button_percentage) / total_items = 28 := 
sorry

end NUMINAMATH_GPT_percentage_of_copper_buttons_l806_80693


namespace NUMINAMATH_GPT_minimize_y_l806_80670

def y (x a b : ℝ) : ℝ := (x-a)^2 * (x-b)^2

theorem minimize_y (a b : ℝ) : ∃ x : ℝ, y x a b = 0 := by
  use a
  sorry

end NUMINAMATH_GPT_minimize_y_l806_80670


namespace NUMINAMATH_GPT_equation_holds_l806_80600

-- Positive integers less than 10
def is_lt_10 (x : ℕ) : Prop := x > 0 ∧ x < 10

theorem equation_holds (a b c : ℕ) (ha : is_lt_10 a) (hb : is_lt_10 b) (hc : is_lt_10 c) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c ↔ b + c = 10 :=
by
  sorry

end NUMINAMATH_GPT_equation_holds_l806_80600


namespace NUMINAMATH_GPT_functional_equation_zero_solution_l806_80608

theorem functional_equation_zero_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_functional_equation_zero_solution_l806_80608


namespace NUMINAMATH_GPT_arccos_sin_eq_pi_div_two_sub_1_72_l806_80651

theorem arccos_sin_eq_pi_div_two_sub_1_72 :
  Real.arccos (Real.sin 8) = Real.pi / 2 - 1.72 :=
sorry

end NUMINAMATH_GPT_arccos_sin_eq_pi_div_two_sub_1_72_l806_80651


namespace NUMINAMATH_GPT_mike_ride_distance_l806_80659

theorem mike_ride_distance (M : ℕ) 
  (cost_Mike : ℝ) 
  (cost_Annie : ℝ) 
  (annies_miles : ℕ := 26) 
  (annies_toll : ℝ := 5) 
  (mile_cost : ℝ := 0.25) 
  (initial_fee : ℝ := 2.5)
  (hc_Mike : cost_Mike = initial_fee + mile_cost * M)
  (hc_Annie : cost_Annie = initial_fee + annies_toll + mile_cost * annies_miles)
  (heq : cost_Mike = cost_Annie) :
  M = 46 := by 
  sorry

end NUMINAMATH_GPT_mike_ride_distance_l806_80659


namespace NUMINAMATH_GPT_total_toothpicks_for_grid_l806_80606

-- Defining the conditions
def grid_height := 30
def grid_width := 15

-- Define the function that calculates the total number of toothpicks
def total_toothpicks (height : ℕ) (width : ℕ) : ℕ :=
  let horizontal_toothpicks := (height + 1) * width
  let vertical_toothpicks := (width + 1) * height
  horizontal_toothpicks + vertical_toothpicks

-- The theorem stating the problem and its answer
theorem total_toothpicks_for_grid : total_toothpicks grid_height grid_width = 945 :=
by {
  -- Here we would write the proof steps. Using sorry for now.
  sorry
}

end NUMINAMATH_GPT_total_toothpicks_for_grid_l806_80606


namespace NUMINAMATH_GPT_total_farm_tax_collected_l806_80628

noncomputable def totalFarmTax (taxPaid: ℝ) (percentage: ℝ) : ℝ := taxPaid / (percentage / 100)

theorem total_farm_tax_collected (taxPaid : ℝ) (percentage : ℝ) (h_taxPaid : taxPaid = 480) (h_percentage : percentage = 16.666666666666668) :
  totalFarmTax taxPaid percentage = 2880 :=
by
  rw [h_taxPaid, h_percentage]
  simp [totalFarmTax]
  norm_num
  sorry

end NUMINAMATH_GPT_total_farm_tax_collected_l806_80628


namespace NUMINAMATH_GPT_first_term_is_sqrt9_l806_80621

noncomputable def geometric_first_term (a r : ℝ) : ℝ :=
by
  have h1 : a * r^2 = 3 := by sorry
  have h2 : a * r^4 = 27 := by sorry
  have h3 : (a * r^4) / (a * r^2) = 27 / 3 := by sorry
  have h4 : r^2 = 9 := by sorry
  have h5 : r = 3 ∨ r = -3 := by sorry
  have h6 : (a * 9) = 3 := by sorry
  have h7 : a = 1/3 := by sorry
  exact a

theorem first_term_is_sqrt9 : geometric_first_term 3 9 = 3 :=
by
  sorry

end NUMINAMATH_GPT_first_term_is_sqrt9_l806_80621


namespace NUMINAMATH_GPT_rainfall_difference_l806_80649

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end NUMINAMATH_GPT_rainfall_difference_l806_80649


namespace NUMINAMATH_GPT_cos_pi_over_3_plus_2alpha_l806_80609

theorem cos_pi_over_3_plus_2alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_over_3_plus_2alpha_l806_80609


namespace NUMINAMATH_GPT_line_through_center_eq_line_bisects_chord_eq_l806_80634

section Geometry

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define the point P
def P := (2, 2)

-- Define when line l passes through the center of the circle
def line_through_center (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define when line l bisects chord AB by point P
def line_bisects_chord (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- Prove the equation of line l passing through the center
theorem line_through_center_eq : 
  (∀ (x y : ℝ), line_through_center x y → circleC x y → (x, y) = (1, 0)) →
  2 * (2:ℝ) - 2 - 2 = 0 := sorry

-- Prove the equation of line l bisects chord AB by point P
theorem line_bisects_chord_eq:
  (∀ (x y : ℝ), line_bisects_chord x y → circleC x y → (2, 2) = P) →
  (2 + 2 * 2 - 6 = 0) := sorry

end Geometry

end NUMINAMATH_GPT_line_through_center_eq_line_bisects_chord_eq_l806_80634


namespace NUMINAMATH_GPT_four_painters_small_room_days_l806_80683

-- Define the constants and conditions
def large_room_days : ℕ := 2
def small_room_factor : ℝ := 0.5
def total_painters : ℕ := 5
def painters_available : ℕ := 4

-- Define the total painter-days needed for the small room
def small_room_painter_days : ℝ := total_painters * (small_room_factor * large_room_days)

-- Define the proof problem statement
theorem four_painters_small_room_days : (small_room_painter_days / painters_available) = 5 / 4 :=
by
  -- Placeholder for the proof: we assume the goal is true for now
  sorry

end NUMINAMATH_GPT_four_painters_small_room_days_l806_80683


namespace NUMINAMATH_GPT_sequence_monotonic_b_gt_neg3_l806_80602

theorem sequence_monotonic_b_gt_neg3 (b : ℝ) :
  (∀ n : ℕ, n > 0 → (n+1)^2 + b*(n+1) > n^2 + b*n) ↔ b > -3 :=
by sorry

end NUMINAMATH_GPT_sequence_monotonic_b_gt_neg3_l806_80602


namespace NUMINAMATH_GPT_units_digit_fraction_mod_10_l806_80619

theorem units_digit_fraction_mod_10 : (30 * 32 * 34 * 36 * 38 * 40) % 2000 % 10 = 2 := by
  sorry

end NUMINAMATH_GPT_units_digit_fraction_mod_10_l806_80619


namespace NUMINAMATH_GPT_sin_sum_bound_l806_80645

theorem sin_sum_bound (x : ℝ) : 
  |(Real.sin x) + (Real.sin (Real.sqrt 2 * x))| < 2 - 1 / (100 * (x^2 + 1)) :=
by sorry

end NUMINAMATH_GPT_sin_sum_bound_l806_80645


namespace NUMINAMATH_GPT_total_problems_l806_80620

theorem total_problems (C : ℕ) (W : ℕ)
  (h1 : C = 20)
  (h2 : 3 * C + 5 * W = 110) : 
  C + W = 30 := by
  sorry

end NUMINAMATH_GPT_total_problems_l806_80620


namespace NUMINAMATH_GPT_red_pairs_l806_80665

theorem red_pairs (total_students green_students red_students total_pairs green_pairs : ℕ) 
  (h1 : total_students = green_students + red_students)
  (h2 : green_students = 67)
  (h3 : red_students = 89)
  (h4 : total_pairs = 78)
  (h5 : green_pairs = 25)
  (h6 : 2 * green_pairs ≤ green_students ∧ 2 * green_pairs ≤ red_students ∧ 2 * green_pairs ≤ 2 * total_pairs) :
  ∃ red_pairs : ℕ, red_pairs = 36 := by
    sorry

end NUMINAMATH_GPT_red_pairs_l806_80665


namespace NUMINAMATH_GPT_angle_bisector_theorem_l806_80637

noncomputable def angle_bisector_length (a b : ℝ) (C : ℝ) (CX : ℝ) : Prop :=
  C = 120 ∧
  CX = (a * b) / (a + b)

theorem angle_bisector_theorem (a b : ℝ) (C : ℝ) (CX : ℝ) :
  angle_bisector_length a b C CX :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_theorem_l806_80637
