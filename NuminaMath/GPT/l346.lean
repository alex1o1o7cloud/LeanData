import Mathlib

namespace NUMINAMATH_GPT_solve_for_y_l346_34698

theorem solve_for_y (x y : ℝ) (h1 : x ^ (2 * y) = 16) (h2 : x = 2) : y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l346_34698


namespace NUMINAMATH_GPT_mango_selling_price_l346_34659

theorem mango_selling_price
  (CP SP_loss SP_profit : ℝ)
  (h1 : SP_loss = 0.8 * CP)
  (h2 : SP_profit = 1.05 * CP)
  (h3 : SP_profit = 6.5625) :
  SP_loss = 5.00 :=
by
  sorry

end NUMINAMATH_GPT_mango_selling_price_l346_34659


namespace NUMINAMATH_GPT_morning_routine_time_l346_34688

section

def time_for_teeth_and_face : ℕ := 3
def time_for_cooking : ℕ := 14
def time_for_reading_while_cooking : ℕ := time_for_cooking - time_for_teeth_and_face
def additional_time_for_reading : ℕ := 1
def total_time_for_reading : ℕ := time_for_reading_while_cooking + additional_time_for_reading
def time_for_eating : ℕ := 6

def total_time_to_school : ℕ := time_for_cooking + time_for_eating

theorem morning_routine_time :
  total_time_to_school = 21 := sorry

end

end NUMINAMATH_GPT_morning_routine_time_l346_34688


namespace NUMINAMATH_GPT_Cally_colored_shirts_l346_34610

theorem Cally_colored_shirts (C : ℕ) (hcally : 10 + 7 + 6 = 23) (hdanny : 6 + 8 + 10 + 6 = 30) (htotal : 23 + 30 + C = 58) : 
  C = 5 := 
by
  sorry

end NUMINAMATH_GPT_Cally_colored_shirts_l346_34610


namespace NUMINAMATH_GPT_geometric_fraction_l346_34618

noncomputable def a_n : ℕ → ℝ := sorry
axiom a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5
axiom geometric_sequence : ∀ n, a_n (n + 1) = a_n n * a_n (n + 1) / a_n (n - 1) 

theorem geometric_fraction (a_n : ℕ → ℝ) (a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5) :
  (a_n 13) / (a_n 9) = 9 :=
sorry

end NUMINAMATH_GPT_geometric_fraction_l346_34618


namespace NUMINAMATH_GPT_gwen_did_not_recycle_2_bags_l346_34695

def points_per_bag : ℕ := 8
def total_bags : ℕ := 4
def points_earned : ℕ := 16

theorem gwen_did_not_recycle_2_bags : total_bags - points_earned / points_per_bag = 2 := by
  sorry

end NUMINAMATH_GPT_gwen_did_not_recycle_2_bags_l346_34695


namespace NUMINAMATH_GPT_total_points_correct_l346_34691

def points_from_two_pointers (t : ℕ) : ℕ := 2 * t
def points_from_three_pointers (th : ℕ) : ℕ := 3 * th
def points_from_free_throws (f : ℕ) : ℕ := f

def total_points (two_points three_points free_throws : ℕ) : ℕ :=
  points_from_two_pointers two_points + points_from_three_pointers three_points + points_from_free_throws free_throws

def sam_points : ℕ := total_points 20 5 10
def alex_points : ℕ := total_points 15 6 8
def jake_points : ℕ := total_points 10 8 5
def lily_points : ℕ := total_points 12 3 16

def game_total_points : ℕ := sam_points + alex_points + jake_points + lily_points

theorem total_points_correct : game_total_points = 219 :=
by
  sorry

end NUMINAMATH_GPT_total_points_correct_l346_34691


namespace NUMINAMATH_GPT_min_value_of_quadratic_l346_34683

theorem min_value_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ x : ℝ, (x = -p / 2) ∧ ∀ y : ℝ, (y^2 + p * y + q) ≥ ((-p/2)^2 + p * (-p/2) + q) :=
sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l346_34683


namespace NUMINAMATH_GPT_other_x_intercept_vertex_symmetric_l346_34616

theorem other_x_intercept_vertex_symmetric (a b c : ℝ)
  (h_vertex : ∀ x y : ℝ, (4, 10) = (x, y) → y = a * x^2 + b * x + c)
  (h_intercept : ∀ x : ℝ, (-1, 0) = (x, 0) → a * x^2 + b * x + c = 0) :
  a * 9^2 + b * 9 + c = 0 :=
sorry

end NUMINAMATH_GPT_other_x_intercept_vertex_symmetric_l346_34616


namespace NUMINAMATH_GPT_inequality_correct_l346_34611

theorem inequality_correct (a b : ℝ) (h : a - |b| > 0) : a + b > 0 :=
sorry

end NUMINAMATH_GPT_inequality_correct_l346_34611


namespace NUMINAMATH_GPT_jihye_wallet_total_l346_34626

-- Declare the amounts
def notes_amount : Nat := 2 * 1000
def coins_amount : Nat := 560

-- Theorem statement asserting the total amount
theorem jihye_wallet_total : notes_amount + coins_amount = 2560 := by
  sorry

end NUMINAMATH_GPT_jihye_wallet_total_l346_34626


namespace NUMINAMATH_GPT_sqrt_expression_eval_l346_34689

theorem sqrt_expression_eval :
  (Real.sqrt 48 / Real.sqrt 3) - (Real.sqrt (1 / 6) * Real.sqrt 12) + Real.sqrt 24 = 4 - Real.sqrt 2 + 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_eval_l346_34689


namespace NUMINAMATH_GPT_average_speed_l346_34677

theorem average_speed (v1 v2 : ℝ) (hv1 : v1 ≠ 0) (hv2 : v2 ≠ 0) : 
  2 / (1 / v1 + 1 / v2) = 2 * v1 * v2 / (v1 + v2) :=
by sorry

end NUMINAMATH_GPT_average_speed_l346_34677


namespace NUMINAMATH_GPT_three_digit_numbers_without_579_l346_34656

def count_valid_digits (exclusions : List Nat) (range : List Nat) : Nat :=
  (range.filter (λ n => n ∉ exclusions)).length

def count_valid_three_digit_numbers : Nat :=
  let hundreds := count_valid_digits [5, 7, 9] [1, 2, 3, 4, 6, 8]
  let tens_units := count_valid_digits [5, 7, 9] [0, 1, 2, 3, 4, 6, 8]
  hundreds * tens_units * tens_units

theorem three_digit_numbers_without_579 : 
  count_valid_three_digit_numbers = 294 :=
by
  unfold count_valid_three_digit_numbers
  /- 
  Here you can add intermediate steps if necessary, 
  but for now we assert the final goal since this is 
  just the problem statement with the proof omitted.
  -/
  sorry

end NUMINAMATH_GPT_three_digit_numbers_without_579_l346_34656


namespace NUMINAMATH_GPT_apples_distribution_l346_34653

theorem apples_distribution (total_apples : ℝ) (apples_per_person : ℝ) (number_of_people : ℝ) 
    (h1 : total_apples = 45) (h2 : apples_per_person = 15.0) : number_of_people = 3 :=
by
  sorry

end NUMINAMATH_GPT_apples_distribution_l346_34653


namespace NUMINAMATH_GPT_average_daily_sales_l346_34676

def pens_sold_day_one : ℕ := 96
def pens_sold_next_days : ℕ := 44
def total_days : ℕ := 13

theorem average_daily_sales : (pens_sold_day_one + 12 * pens_sold_next_days) / total_days = 48 := 
by 
  sorry

end NUMINAMATH_GPT_average_daily_sales_l346_34676


namespace NUMINAMATH_GPT_find_m_l346_34697

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end NUMINAMATH_GPT_find_m_l346_34697


namespace NUMINAMATH_GPT_machine_fill_time_l346_34674

theorem machine_fill_time (filled_cans : ℕ) (time_per_batch : ℕ) (total_cans : ℕ) (expected_time : ℕ)
  (h1 : filled_cans = 150)
  (h2 : time_per_batch = 8)
  (h3 : total_cans = 675)
  (h4 : expected_time = 36) :
  (total_cans / filled_cans) * time_per_batch = expected_time :=
by 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_machine_fill_time_l346_34674


namespace NUMINAMATH_GPT_least_possible_value_of_b_l346_34619

theorem least_possible_value_of_b (a b : ℕ) 
  (ha : ∃ p, (∀ q, p ∣ q ↔ q = 1 ∨ q = p ∨ q = p*p ∨ q = a))
  (hb : ∃ k, (∀ l, k ∣ l ↔ (l = 1 ∨ l = b)))
  (hdiv : a ∣ b) : 
  b = 12 :=
sorry

end NUMINAMATH_GPT_least_possible_value_of_b_l346_34619


namespace NUMINAMATH_GPT_find_four_numbers_l346_34601

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b = 2024) 
  (h2 : a + c = 2026) 
  (h3 : a + d = 2030) 
  (h4 : b + c = 2028) 
  (h5 : b + d = 2032) 
  (h6 : c + d = 2036) : 
  (a = 1011 ∧ b = 1012 ∧ c = 1013 ∧ d = 1015) := 
sorry

end NUMINAMATH_GPT_find_four_numbers_l346_34601


namespace NUMINAMATH_GPT_not_equal_fractions_l346_34644

theorem not_equal_fractions :
  ¬ ((14 / 12 = 7 / 6) ∧
     (1 + 1 / 6 = 7 / 6) ∧
     (21 / 18 = 7 / 6) ∧
     (1 + 2 / 12 = 7 / 6) ∧
     (1 + 1 / 3 = 7 / 6)) :=
by 
  sorry

end NUMINAMATH_GPT_not_equal_fractions_l346_34644


namespace NUMINAMATH_GPT_people_happy_correct_l346_34602

-- Define the size and happiness percentage of an institution.
variables (size : ℕ) (happiness_percentage : ℚ)

-- Assume the size is between 100 and 200.
axiom size_range : 100 ≤ size ∧ size ≤ 200

-- Assume the happiness percentage is between 0.6 and 0.95.
axiom happiness_percentage_range : 0.6 ≤ happiness_percentage ∧ happiness_percentage ≤ 0.95

-- Define the number of people made happy at an institution.
def people_made_happy (size : ℕ) (happiness_percentage : ℚ) : ℚ := 
  size * happiness_percentage

-- Theorem stating that the number of people made happy is as expected.
theorem people_happy_correct : 
  ∀ (size : ℕ) (happiness_percentage : ℚ), 
  100 ≤ size → size ≤ 200 → 
  0.6 ≤ happiness_percentage → happiness_percentage ≤ 0.95 → 
  people_made_happy size happiness_percentage = size * happiness_percentage := 
by 
  intros size happiness_percentage hsize1 hsize2 hperc1 hperc2
  unfold people_made_happy
  sorry

end NUMINAMATH_GPT_people_happy_correct_l346_34602


namespace NUMINAMATH_GPT_problem1_problem2_l346_34607

theorem problem1 : (-5 : ℝ) ^ 0 - (1 / 3) ^ (-2 : ℝ) + (-2 : ℝ) ^ 2 = -4 := 
by
  sorry

variable (a : ℝ)

theorem problem2 : (-3 * a ^ 3) ^ 2 * 2 * a ^ 3 - 8 * a ^ 12 / (2 * a ^ 3) = 14 * a ^ 9 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l346_34607


namespace NUMINAMATH_GPT_intersection_is_correct_l346_34662

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := { x | x > 1/3 }
def setB : Set ℝ := { y | -3 ≤ y ∧ y ≤ 3 }

-- Prove that the intersection of A and B is (1/3, 3]
theorem intersection_is_correct : setA ∩ setB = { x | 1/3 < x ∧ x ≤ 3 } := 
by
  sorry

end NUMINAMATH_GPT_intersection_is_correct_l346_34662


namespace NUMINAMATH_GPT_distinct_arrangements_on_3x3_grid_l346_34606

def is_valid_position (pos : ℤ × ℤ) : Prop :=
  0 ≤ pos.1 ∧ pos.1 < 3 ∧ 0 ≤ pos.2 ∧ pos.2 < 3

def rotations_equiv (pos1 pos2 : ℤ × ℤ) : Prop :=
  pos1 = pos2 ∨ pos1 = (2 - pos2.2, pos2.1) ∨ pos1 = (2 - pos2.1, 2 - pos2.2) ∨ pos1 = (pos2.2, 2 - pos2.1)

def distinct_positions_count (grid_size : ℕ) : ℕ :=
  10  -- given from the problem solution

theorem distinct_arrangements_on_3x3_grid : distinct_positions_count 3 = 10 := sorry

end NUMINAMATH_GPT_distinct_arrangements_on_3x3_grid_l346_34606


namespace NUMINAMATH_GPT_magnitude_BC_eq_sqrt29_l346_34615

noncomputable def A : (ℝ × ℝ) := (2, -1)
noncomputable def C : (ℝ × ℝ) := (0, 2)
noncomputable def AB : (ℝ × ℝ) := (3, 5)

theorem magnitude_BC_eq_sqrt29
    (A : ℝ × ℝ := (2, -1))
    (C : ℝ × ℝ := (0, 2))
    (AB : ℝ × ℝ := (3, 5)) :
    ∃ B : ℝ × ℝ, (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = 29 := 
by
  sorry

end NUMINAMATH_GPT_magnitude_BC_eq_sqrt29_l346_34615


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l346_34670

theorem bus_speed_excluding_stoppages (s_including_stops : ℕ) (stop_time_minutes : ℕ) (s_excluding_stops : ℕ) (v : ℕ) : 
  (s_including_stops = 45) ∧ (stop_time_minutes = 24) ∧ (v = s_including_stops * 5 / 3) → s_excluding_stops = 75 := 
by {
  sorry
}

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l346_34670


namespace NUMINAMATH_GPT_bill_painting_hours_l346_34680

theorem bill_painting_hours (B J : ℝ) (hB : 0 < B) (hJ : 0 < J) : 
  ∃ t : ℝ, t = (B-1)/(B+J) ∧ (t + 1 = (B * (J + 1)) / (B + J)) :=
by
  sorry

end NUMINAMATH_GPT_bill_painting_hours_l346_34680


namespace NUMINAMATH_GPT_find_a1_l346_34668

-- Definitions used in the conditions
variables {a : ℕ → ℝ} -- Sequence a(n)
variable (n : ℕ) -- Number of terms
noncomputable def arithmeticSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

noncomputable def arithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a m + (n - m) * (a 2 - a 1)

theorem find_a1 (h_seq : arithmeticSeq a)
  (h_sum_first_100 : arithmeticSum a 100 = 100)
  (h_sum_last_100 : arithmeticSum (λ i => a (i + 900)) 100 = 1000) :
  a 1 = 101 / 200 :=
  sorry

end NUMINAMATH_GPT_find_a1_l346_34668


namespace NUMINAMATH_GPT_arcs_intersection_l346_34608

theorem arcs_intersection (k : ℕ) : (1 ≤ k ∧ k ≤ 99) ∧ ¬(∃ m : ℕ, k + 1 = 8 * m) ↔ ∃ n l : ℕ, (2 * l + 1) * 100 = (k + 1) * n ∧ n = 100 ∧ k < 100 := by
  sorry

end NUMINAMATH_GPT_arcs_intersection_l346_34608


namespace NUMINAMATH_GPT_zhen_zhen_test_score_l346_34645

theorem zhen_zhen_test_score
  (avg1 avg2 : ℝ) (n m : ℝ)
  (h1 : avg1 = 88)
  (h2 : avg2 = 90)
  (h3 : n = 4)
  (h4 : m = 5) :
  avg2 * m - avg1 * n = 98 :=
by
  -- Given the hypotheses h1, h2, h3, and h4,
  -- we need to show that avg2 * m - avg1 * n = 98.
  sorry

end NUMINAMATH_GPT_zhen_zhen_test_score_l346_34645


namespace NUMINAMATH_GPT_horner_v3_value_correct_l346_34624

def f (x : ℕ) : ℕ :=
  x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_eval (x : ℕ) : ℕ :=
  ((((x + 0) * x + 2) * x + 3) * x + 1) * x + 1

theorem horner_v3_value_correct :
  horner_eval 3 = 36 :=
sorry

end NUMINAMATH_GPT_horner_v3_value_correct_l346_34624


namespace NUMINAMATH_GPT_amy_owes_thirty_l346_34605

variable (A D : ℝ)

theorem amy_owes_thirty
  (total_pledged remaining_owed sally_carl_owe derek_half_amys_owes : ℝ)
  (h1 : total_pledged = 285)
  (h2 : remaining_owed = 400 - total_pledged)
  (h3 : sally_carl_owe = 35 + 35)
  (h4 : derek_half_amys_owes = A / 2)
  (h5 : remaining_owed - sally_carl_owe = 45)
  (h6 : 45 = A + (A / 2)) :
  A = 30 :=
by
  -- Proof steps skipped
  sorry

end NUMINAMATH_GPT_amy_owes_thirty_l346_34605


namespace NUMINAMATH_GPT_range_of_a_l346_34675

noncomputable def proposition_p (a : ℝ) : Prop :=
∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

noncomputable def proposition_q (a : ℝ) : Prop :=
∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) (h : proposition_p a ∧ proposition_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l346_34675


namespace NUMINAMATH_GPT_time_after_2023_hours_l346_34654

theorem time_after_2023_hours (current_time : ℕ) (hours_later : ℕ) (modulus : ℕ) : 
    (current_time = 3) → 
    (hours_later = 2023) → 
    (modulus = 12) → 
    ((current_time + (hours_later % modulus)) % modulus = 2) :=
by 
    intros h1 h2 h3
    rw [h1, h2, h3]
    sorry

end NUMINAMATH_GPT_time_after_2023_hours_l346_34654


namespace NUMINAMATH_GPT_daily_wage_of_a_man_l346_34665

theorem daily_wage_of_a_man (M W : ℝ) 
  (h1 : 24 * M + 16 * W = 11600) 
  (h2 : 12 * M + 37 * W = 11600) : 
  M = 350 :=
by
  sorry

end NUMINAMATH_GPT_daily_wage_of_a_man_l346_34665


namespace NUMINAMATH_GPT_intersection_complement_l346_34613

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

theorem intersection_complement :
  A ∩ (U \ B) = {2} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_complement_l346_34613


namespace NUMINAMATH_GPT_jessa_needs_470_cupcakes_l346_34655

def total_cupcakes_needed (fourth_grade_classes : ℕ) (students_per_fourth_grade_class : ℕ) (pe_class_students : ℕ) (afterschool_clubs : ℕ) (students_per_afterschool_club : ℕ) : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) + pe_class_students + (afterschool_clubs * students_per_afterschool_club)

theorem jessa_needs_470_cupcakes :
  total_cupcakes_needed 8 40 80 2 35 = 470 :=
by
  sorry

end NUMINAMATH_GPT_jessa_needs_470_cupcakes_l346_34655


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l346_34661

theorem lcm_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 9) (h2 : a * b = 1800) : Nat.lcm a b = 200 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l346_34661


namespace NUMINAMATH_GPT_base_area_of_cylinder_l346_34603

variables (S : ℝ) (cylinder : Type)
variables (square_cross_section : cylinder → Prop) (area_square : cylinder → ℝ)
variables (base_area : cylinder → ℝ)

-- Assume that the cylinder has a square cross-section with a given area
axiom cross_section_square : ∀ c : cylinder, square_cross_section c → area_square c = 4 * S

-- Theorem stating the area of the base of the cylinder
theorem base_area_of_cylinder (c : cylinder) (h : square_cross_section c) : base_area c = π * S :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_base_area_of_cylinder_l346_34603


namespace NUMINAMATH_GPT_total_spent_is_140_l346_34631

-- Define the original prices and discounts
def original_price_shoes : ℕ := 50
def original_price_dress : ℕ := 100
def discount_shoes : ℕ := 40
def discount_dress : ℕ := 20

-- Define the number of items purchased
def number_of_shoes : ℕ := 2
def number_of_dresses : ℕ := 1

-- Define the calculation of discounted prices
def discounted_price_shoes (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

def discounted_price_dress (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

-- Define the total cost calculation
def total_cost : ℕ :=
  discounted_price_shoes original_price_shoes discount_shoes number_of_shoes +
  discounted_price_dress original_price_dress discount_dress number_of_dresses

-- The theorem to prove
theorem total_spent_is_140 : total_cost = 140 := by
  sorry

end NUMINAMATH_GPT_total_spent_is_140_l346_34631


namespace NUMINAMATH_GPT_benny_missed_games_l346_34628

theorem benny_missed_games (total_games attended_games missed_games : ℕ)
  (H1 : total_games = 39)
  (H2 : attended_games = 14)
  (H3 : missed_games = total_games - attended_games) :
  missed_games = 25 :=
by
  sorry

end NUMINAMATH_GPT_benny_missed_games_l346_34628


namespace NUMINAMATH_GPT_white_black_ratio_l346_34686

theorem white_black_ratio (W B : ℕ) (h1 : W + B = 78) (h2 : (2 / 3 : ℚ) * (B - W) = 4) : W / B = 6 / 7 := by
  sorry

end NUMINAMATH_GPT_white_black_ratio_l346_34686


namespace NUMINAMATH_GPT_last_number_written_on_sheet_l346_34617

/-- The given problem is to find the last number written on a sheet with specific rules. 
Given:
- The sheet has dimensions of 100 characters in width and 100 characters in height.
- Numbers are written successively with a space between each number.
- If the end of a line is reached, the next number continues at the beginning of the next line.

We need to prove that the last number written on the sheet is 2220.
-/
theorem last_number_written_on_sheet :
  ∃ (n : ℕ), n = 2220 ∧ 
    let width := 100
    let height := 100
    let sheet_size := width * height
    let write_number size occupied_space := occupied_space + size + 1 
    ∃ (numbers : ℕ → ℕ) (space_per_number : ℕ → ℕ),
      ( ∀ i, space_per_number i = if numbers i < 10 then 2 else if numbers i < 100 then 3 else if numbers i < 1000 then 4 else 5 ) ∧
      ∃ (current_space : ℕ), 
        (current_space ≤ sheet_size) ∧
        (∀ i, current_space = write_number (space_per_number i) current_space ) :=
sorry

end NUMINAMATH_GPT_last_number_written_on_sheet_l346_34617


namespace NUMINAMATH_GPT_vector_coordinates_l346_34652

theorem vector_coordinates :
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1) :=
by
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  show (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1)
  sorry

end NUMINAMATH_GPT_vector_coordinates_l346_34652


namespace NUMINAMATH_GPT_weight_of_3_moles_HBrO3_is_386_73_l346_34641

noncomputable def H_weight : ℝ := 1.01
noncomputable def Br_weight : ℝ := 79.90
noncomputable def O_weight : ℝ := 16.00
noncomputable def HBrO3_weight : ℝ := H_weight + Br_weight + 3 * O_weight
noncomputable def weight_of_3_moles_of_HBrO3 : ℝ := 3 * HBrO3_weight

theorem weight_of_3_moles_HBrO3_is_386_73 : weight_of_3_moles_of_HBrO3 = 386.73 := by
  sorry

end NUMINAMATH_GPT_weight_of_3_moles_HBrO3_is_386_73_l346_34641


namespace NUMINAMATH_GPT_find_xyz_sum_l346_34647

theorem find_xyz_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 12)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 37) :
  x * y + y * z + z * x = 20 :=
sorry

end NUMINAMATH_GPT_find_xyz_sum_l346_34647


namespace NUMINAMATH_GPT_banana_cream_pie_correct_slice_l346_34684

def total_students := 45
def strawberry_pie_preference := 15
def pecan_pie_preference := 10
def pumpkin_pie_preference := 9

noncomputable def banana_cream_pie_slice_degrees : ℝ :=
  let remaining_students := total_students - strawberry_pie_preference - pecan_pie_preference - pumpkin_pie_preference
  let students_per_preference := remaining_students / 2
  (students_per_preference / total_students) * 360

theorem banana_cream_pie_correct_slice :
  banana_cream_pie_slice_degrees = 44 := by
  sorry

end NUMINAMATH_GPT_banana_cream_pie_correct_slice_l346_34684


namespace NUMINAMATH_GPT_new_average_age_after_person_leaves_l346_34609

theorem new_average_age_after_person_leaves (avg_age : ℕ) (n : ℕ) (leaving_age : ℕ) (remaining_count : ℕ) :
  ((n * avg_age - leaving_age) / remaining_count) = 33 :=
by
  -- Given conditions
  let avg_age := 30
  let n := 5
  let leaving_age := 18
  let remaining_count := n - 1
  -- Conclusion
  sorry

end NUMINAMATH_GPT_new_average_age_after_person_leaves_l346_34609


namespace NUMINAMATH_GPT_distance_covered_by_wheel_l346_34625

noncomputable def pi_num : ℝ := 3.14159

noncomputable def wheel_diameter : ℝ := 14

noncomputable def number_of_revolutions : ℝ := 33.03002729754322

noncomputable def circumference : ℝ := pi_num * wheel_diameter

noncomputable def calculated_distance : ℝ := circumference * number_of_revolutions

theorem distance_covered_by_wheel : 
  calculated_distance = 1452.996 :=
sorry

end NUMINAMATH_GPT_distance_covered_by_wheel_l346_34625


namespace NUMINAMATH_GPT_TripleApplicationOfF_l346_34620

def f (N : ℝ) : ℝ := 0.7 * N + 2

theorem TripleApplicationOfF :
  f (f (f 40)) = 18.1 :=
  sorry

end NUMINAMATH_GPT_TripleApplicationOfF_l346_34620


namespace NUMINAMATH_GPT_find_zebras_last_year_l346_34658

def zebras_last_year (current : ℕ) (born : ℕ) (died : ℕ) : ℕ :=
  current - born + died

theorem find_zebras_last_year :
  zebras_last_year 725 419 263 = 569 :=
by
  sorry

end NUMINAMATH_GPT_find_zebras_last_year_l346_34658


namespace NUMINAMATH_GPT_parallel_lines_m_eq_minus_seven_l346_34678

theorem parallel_lines_m_eq_minus_seven
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m)
  (l₂ : ∀ x y : ℝ, 2 * x + (5 + m) * y = 8)
  (parallel : ∀ x y : ℝ, (3 + m) * 4 = 2 * (5 + m)) :
  m = -7 :=
sorry

end NUMINAMATH_GPT_parallel_lines_m_eq_minus_seven_l346_34678


namespace NUMINAMATH_GPT_butanoic_acid_molecular_weight_l346_34673

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_weight_butanoic_acid : ℝ :=
  4 * atomic_weight_C + 8 * atomic_weight_H + 2 * atomic_weight_O

theorem butanoic_acid_molecular_weight :
  molecular_weight_butanoic_acid = 88.104 :=
by
  -- proof not required
  sorry

end NUMINAMATH_GPT_butanoic_acid_molecular_weight_l346_34673


namespace NUMINAMATH_GPT_train_length_is_180_l346_34696

noncomputable def train_length (time_seconds : ℕ) (speed_kmh : ℕ) : ℕ := 
  (speed_kmh * 5 / 18) * time_seconds

theorem train_length_is_180 : train_length 9 72 = 180 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_180_l346_34696


namespace NUMINAMATH_GPT_number_of_women_l346_34635

-- Definitions for the given conditions
variables (m w : ℝ)
variable (x : ℝ)

-- Conditions
def cond1 : Prop := 3 * m + 8 * w = 6 * m + 2 * w
def cond2 : Prop := 4 * m + x * w = 0.9285714285714286 * (3 * m + 8 * w)

-- Theorem to prove the number of women in the third group (x)
theorem number_of_women (h1 : cond1 m w) (h2 : cond2 m w x) : x = 5 :=
sorry

end NUMINAMATH_GPT_number_of_women_l346_34635


namespace NUMINAMATH_GPT_maria_drank_8_bottles_l346_34669

def initial_bottles : ℕ := 14
def bought_bottles : ℕ := 45
def remaining_bottles : ℕ := 51

theorem maria_drank_8_bottles :
  let total_bottles := initial_bottles + bought_bottles
  let drank_bottles := total_bottles - remaining_bottles
  drank_bottles = 8 :=
by
  let total_bottles := 14 + 45
  let drank_bottles := total_bottles - 51
  show drank_bottles = 8
  sorry

end NUMINAMATH_GPT_maria_drank_8_bottles_l346_34669


namespace NUMINAMATH_GPT_possible_values_of_a_l346_34632

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem possible_values_of_a 
    (a b c : ℝ) 
    (h1 : f a b c (Real.pi / 2) = 1) 
    (h2 : f a b c Real.pi = 1) 
    (h3 : ∀ x : ℝ, |f a b c x| ≤ 2) :
    4 - 3 * Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l346_34632


namespace NUMINAMATH_GPT_trains_cross_time_l346_34621

noncomputable def time_to_cross_trains : ℝ :=
  let l1 := 220 -- length of the first train in meters
  let s1 := 120 * (5 / 18) -- speed of the first train in meters per second
  let l2 := 280.04 -- length of the second train in meters
  let s2 := 80 * (5 / 18) -- speed of the second train in meters per second
  let relative_speed := s1 + s2 -- relative speed in meters per second
  let total_length := l1 + l2 -- total length to be crossed in meters
  total_length / relative_speed -- time in seconds

theorem trains_cross_time :
  abs (time_to_cross_trains - 9) < 0.01 := -- Allowing a small error to account for approximation
by
  sorry

end NUMINAMATH_GPT_trains_cross_time_l346_34621


namespace NUMINAMATH_GPT_problem_statement_l346_34639

theorem problem_statement (a b c : ℕ) (h1 : a < 12) (h2 : b < 12) (h3 : c < 12) (h4 : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b + c :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l346_34639


namespace NUMINAMATH_GPT_part1_part2_l346_34634

namespace Problem

open Set

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

-- Part (1)
theorem part1 : A ∩ (B ∩ C) = {3} := by 
  sorry

-- Part (2)
theorem part2 : A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0} := by 
  sorry

end Problem

end NUMINAMATH_GPT_part1_part2_l346_34634


namespace NUMINAMATH_GPT_least_number_to_make_divisible_l346_34636

def least_common_multiple (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem least_number_to_make_divisible (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : 
  least_common_multiple a b = 77 → 
  (n % least_common_multiple a b) = 40 →
  c = (least_common_multiple a b - (n % least_common_multiple a b)) →
  c = 37 :=
by
sorry

end NUMINAMATH_GPT_least_number_to_make_divisible_l346_34636


namespace NUMINAMATH_GPT_total_cards_needed_l346_34649

def red_card_credits := 3
def blue_card_credits := 5
def total_credits := 84
def red_cards := 8

theorem total_cards_needed :
  red_card_credits * red_cards + blue_card_credits * (total_credits - red_card_credits * red_cards) / blue_card_credits = 20 := by
  sorry

end NUMINAMATH_GPT_total_cards_needed_l346_34649


namespace NUMINAMATH_GPT_find_prices_max_basketballs_l346_34643

-- Define price of basketballs and soccer balls
def basketball_price : ℕ := 80
def soccer_ball_price : ℕ := 50

-- Define the equations given in the problem
theorem find_prices (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 310)
  (h2 : 5 * x + 2 * y = 500) : 
  x = basketball_price ∧ y = soccer_ball_price :=
sorry

-- Define the maximum number of basketballs given the cost constraints
theorem max_basketballs (m : ℕ)
  (htotal : m + (60 - m) = 60)
  (hcost : 80 * m + 50 * (60 - m) ≤ 4000) : 
  m ≤ 33 :=
sorry

end NUMINAMATH_GPT_find_prices_max_basketballs_l346_34643


namespace NUMINAMATH_GPT_stella_annual_income_after_tax_l346_34622

-- Definitions of the conditions
def base_salary_per_month : ℝ := 3500
def bonuses : List ℝ := [1200, 600, 1500, 900, 1200]
def months_paid : ℝ := 10
def tax_rate : ℝ := 0.05

-- Calculations derived from the conditions
def total_base_salary : ℝ := base_salary_per_month * months_paid
def total_bonuses : ℝ := bonuses.sum
def total_income_before_tax : ℝ := total_base_salary + total_bonuses
def tax_deduction : ℝ := total_income_before_tax * tax_rate
def annual_income_after_tax : ℝ := total_income_before_tax - tax_deduction

-- The theorem to prove
theorem stella_annual_income_after_tax :
  annual_income_after_tax = 38380 := by
  sorry

end NUMINAMATH_GPT_stella_annual_income_after_tax_l346_34622


namespace NUMINAMATH_GPT_find_f_half_l346_34682

variable {α : Type} [DivisionRing α]

theorem find_f_half {f : α → α} (h : ∀ x, f (1 - 2 * x) = 1 / (x^2)) : f (1 / 2) = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_f_half_l346_34682


namespace NUMINAMATH_GPT_sum_max_min_ratio_ellipse_l346_34681

theorem sum_max_min_ratio_ellipse :
  ∃ (a b : ℝ), (∀ (x y : ℝ), 3*x^2 + 2*x*y + 4*y^2 - 18*x - 28*y + 50 = 0 → (y/x = a ∨ y/x = b)) ∧ a + b = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_max_min_ratio_ellipse_l346_34681


namespace NUMINAMATH_GPT_square_side_length_l346_34687

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end NUMINAMATH_GPT_square_side_length_l346_34687


namespace NUMINAMATH_GPT_minimum_x_plus_3y_l346_34657

theorem minimum_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 3 * x + y) : x + 3 * y ≥ 16 :=
sorry

end NUMINAMATH_GPT_minimum_x_plus_3y_l346_34657


namespace NUMINAMATH_GPT_num_different_pairs_l346_34660

theorem num_different_pairs :
  (∃ (A B : Finset ℕ), A ∪ B = {1, 2, 3, 4} ∧ A ≠ B ∧ (A, B) ≠ (B, A)) ∧
  (∃ n : ℕ, n = 81) :=
by
  -- Proof would go here, but it's skipped per instructions
  sorry

end NUMINAMATH_GPT_num_different_pairs_l346_34660


namespace NUMINAMATH_GPT_intersection_A_B_l346_34699

def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem intersection_A_B :
  (setA ∩ setB = {x | 0 ≤ x ∧ x ≤ 2}) :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l346_34699


namespace NUMINAMATH_GPT_reduced_cost_per_meter_l346_34692

theorem reduced_cost_per_meter (original_cost total_cost new_length original_length : ℝ) :
  original_cost = total_cost / original_length →
  new_length = original_length + 4 →
  total_cost = total_cost →
  original_cost - (total_cost / new_length) = 1 :=
by sorry

end NUMINAMATH_GPT_reduced_cost_per_meter_l346_34692


namespace NUMINAMATH_GPT_two_digit_integer_divides_491_remainder_59_l346_34648

theorem two_digit_integer_divides_491_remainder_59 :
  ∃ n Q : ℕ, (n = 10 * x + y) ∧ (0 < x) ∧ (x ≤ 9) ∧ (0 ≤ y) ∧ (y ≤ 9) ∧ (491 = n * Q + 59) ∧ (n = 72) :=
by
  sorry

end NUMINAMATH_GPT_two_digit_integer_divides_491_remainder_59_l346_34648


namespace NUMINAMATH_GPT_union_sets_l346_34612

open Set

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem union_sets :
  A ∪ B = {1, 2, 3, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l346_34612


namespace NUMINAMATH_GPT_discriminant_of_quadratic_eq_l346_34666

-- Define the coefficients of the quadratic equation
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 1

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

-- State the theorem that we want to prove
theorem discriminant_of_quadratic_eq : discriminant a b c = 61 := by
  sorry

end NUMINAMATH_GPT_discriminant_of_quadratic_eq_l346_34666


namespace NUMINAMATH_GPT_unit_prices_purchasing_schemes_maximize_profit_l346_34640

-- Define the conditions and variables
def purchase_price_system (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 240) ∧ (3 * x + 4 * y = 340)

def possible_schemes (a b : ℕ) : Prop :=
  (a + b = 200) ∧ (60 * a + 40 * b ≤ 10440) ∧ (a ≥ 3 * b / 2)

def max_profit (x y : ℝ) (a b : ℕ) : ℝ :=
  (x - 60) * a + (y - 40) * b

-- Prove the unit prices are $60 and $40
theorem unit_prices : ∃ x y, purchase_price_system x y ∧ x = 60 ∧ y = 40 :=
by
  sorry

-- Prove the possible purchasing schemes
theorem purchasing_schemes : ∀ a b, possible_schemes a b → 
  (a = 120 ∧ b = 80 ∨ a = 121 ∧ b = 79 ∨ a = 122 ∧ b = 78) :=
by
  sorry

-- Prove the maximum profit is 3610 with the purchase amounts (122, 78)
theorem maximize_profit :
  ∃ (a b : ℕ), max_profit 80 55 a b = 3610 ∧ purchase_price_system 60 40 ∧ possible_schemes a b ∧ a = 122 ∧ b = 78 :=
by
  sorry

end NUMINAMATH_GPT_unit_prices_purchasing_schemes_maximize_profit_l346_34640


namespace NUMINAMATH_GPT_servings_of_peanut_butter_l346_34637

-- Definitions from conditions
def total_peanut_butter : ℚ := 35 + 4/5
def serving_size : ℚ := 2 + 1/3

-- Theorem to be proved
theorem servings_of_peanut_butter :
  total_peanut_butter / serving_size = 15 + 17/35 := by
  sorry

end NUMINAMATH_GPT_servings_of_peanut_butter_l346_34637


namespace NUMINAMATH_GPT_barber_total_loss_is_120_l346_34663

-- Definitions for the conditions
def haircut_cost : ℕ := 25
def initial_payment_by_customer : ℕ := 50
def flower_shop_change : ℕ := 50
def bakery_change : ℕ := 10
def customer_received_change : ℕ := 25
def counterfeit_50_replacement : ℕ := 50
def counterfeit_10_replacement : ℕ := 10

-- Calculate total loss for the barber
def total_loss : ℕ :=
  let loss_haircut := haircut_cost
  let loss_change_to_customer := customer_received_change
  let loss_given_to_flower_shop := counterfeit_50_replacement
  let loss_given_to_bakery := counterfeit_10_replacement
  let total_loss_before_offset := loss_haircut + loss_change_to_customer + loss_given_to_flower_shop + loss_given_to_bakery
  let real_currency_received := flower_shop_change
  total_loss_before_offset - real_currency_received

-- Proof statement
theorem barber_total_loss_is_120 : total_loss = 120 := by {
  sorry
}

end NUMINAMATH_GPT_barber_total_loss_is_120_l346_34663


namespace NUMINAMATH_GPT_integer_points_count_l346_34651

theorem integer_points_count :
  ∃ (n : ℤ), n = 9 ∧
  ∀ a b : ℝ, (1 < a) → (1 < b) → (ab + a - b - 10 = 0) →
  (a + b = 6) → 
  ∃ (x y : ℤ), (3 * x^2 + 2 * y^2 ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_integer_points_count_l346_34651


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l346_34650

open Real

-- Part (1)
theorem problem_part1 : ∀ x > 0, log x ≤ x - 1 := 
by 
  sorry -- proof goes here


-- Part (2)
theorem problem_part2 : (∀ x > 0, log x ≤ a * x + (a - 1) / x - 1) → 1 ≤ a := 
by 
  sorry -- proof goes here

end NUMINAMATH_GPT_problem_part1_problem_part2_l346_34650


namespace NUMINAMATH_GPT_remi_water_consumption_proof_l346_34630

-- Definitions for the conditions
def daily_consumption (bottle_volume : ℕ) (refills_per_day : ℕ) : ℕ :=
  bottle_volume * refills_per_day

def total_spillage (spill1 : ℕ) (spill2 : ℕ) : ℕ :=
  spill1 + spill2

def total_consumption (daily : ℕ) (days : ℕ) (spill : ℕ) : ℕ :=
  (daily * days) - spill

-- Theorem proving the number of days d
theorem remi_water_consumption_proof (bottle_volume : ℕ) (refills_per_day : ℕ)
  (spill1 spill2 total_water : ℕ) (d : ℕ)
  (h1 : bottle_volume = 20) (h2 : refills_per_day = 3)
  (h3 : spill1 = 5) (h4 : spill2 = 8)
  (h5 : total_water = 407) :
  total_consumption (daily_consumption bottle_volume refills_per_day) d
    (total_spillage spill1 spill2) = total_water → d = 7 := 
by
  -- Assuming the hypotheses to show the equality
  intro h
  have daily := h1 ▸ h2 ▸ 20 * 3 -- ⇒ daily = 60
  have spillage := h3 ▸ h4 ▸ 5 + 8 -- ⇒ spillage = 13
  rw [daily_consumption, total_spillage, h5] at h
  rw [h1, h2, h3, h4] at h -- Substitute conditions in the hypothesis
  sorry -- place a placeholder for the actual proof

end NUMINAMATH_GPT_remi_water_consumption_proof_l346_34630


namespace NUMINAMATH_GPT_tips_fraction_to_salary_l346_34693

theorem tips_fraction_to_salary (S T I : ℝ)
  (h1 : I = S + T)
  (h2 : T / I = 0.6923076923076923) :
  T / S = 2.25 := by
  sorry

end NUMINAMATH_GPT_tips_fraction_to_salary_l346_34693


namespace NUMINAMATH_GPT_find_B_sin_squared_sum_range_l346_34600

-- Define the angles and vectors
variables {A B C : ℝ}
variables (m n : ℝ × ℝ)
variables (α : ℝ)

-- Basic triangle angle sum condition
axiom angle_sum : A + B + C = Real.pi

-- Define vectors as per the problem statement
axiom vector_m : m = (Real.sin B, 1 - Real.cos B)
axiom vector_n : n = (2, 0)

-- The angle between vectors m and n is π/3
axiom angle_between_vectors : α = Real.pi / 3
axiom angle_condition : Real.cos α = (2 * Real.sin B + 0 * (1 - Real.cos B)) / 
                                     (Real.sqrt (Real.sin B ^ 2 + (1 - Real.cos B) ^ 2) * 2)

theorem find_B : B = 2 * Real.pi / 3 := 
sorry

-- Conditions for range of sin^2 A + sin^2 C
axiom range_condition : (0 < A ∧ A < Real.pi / 3) 
                     ∧ (0 < C ∧ C < Real.pi / 3)
                     ∧ (A + C = Real.pi / 3)

theorem sin_squared_sum_range : (Real.sin A) ^ 2 + (Real.sin C) ^ 2 ∈ Set.Ico (1 / 2) 1 := 
sorry

end NUMINAMATH_GPT_find_B_sin_squared_sum_range_l346_34600


namespace NUMINAMATH_GPT_max_area_parallelogram_l346_34672

theorem max_area_parallelogram
    (P : ℝ)
    (a b : ℝ)
    (h1 : P = 60)
    (h2 : a = 3 * b)
    (h3 : P = 2 * a + 2 * b) :
    (a * b ≤ 168.75) :=
by
  -- We prove that given the conditions, the maximum area is 168.75 square units.
  sorry

end NUMINAMATH_GPT_max_area_parallelogram_l346_34672


namespace NUMINAMATH_GPT_even_times_odd_is_even_l346_34664

theorem even_times_odd_is_even {a b : ℤ} (h₁ : ∃ k, a = 2 * k) (h₂ : ∃ j, b = 2 * j + 1) : ∃ m, a * b = 2 * m :=
by
  sorry

end NUMINAMATH_GPT_even_times_odd_is_even_l346_34664


namespace NUMINAMATH_GPT_arithmetic_sequence_a7_l346_34671

noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence_a7 
  (a : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 4 - a 1) / 3)
  (h_a1 : a 1 = 3)
  (h_a4 : a 4 = 5) : 
  a 7 = 7 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a7_l346_34671


namespace NUMINAMATH_GPT_no_such_function_exists_l346_34642

theorem no_such_function_exists (f : ℤ → ℤ) (h : ∀ m n : ℤ, f (m + f n) = f m - n) : false :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l346_34642


namespace NUMINAMATH_GPT_matrix_addition_is_correct_l346_34690

-- Definitions of matrices A and B according to given conditions
def A : Matrix (Fin 4) (Fin 4) ℤ :=  
  ![![ 3,  0,  1,  4],
    ![ 1,  2,  0,  0],
    ![ 5, -3,  2,  1],
    ![ 0,  0, -1,  3]]

def B : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-5, -7,  3,  2],
    ![ 4, -9,  5, -2],
    ![ 8,  2, -3,  0],
    ![ 1,  1, -2, -4]]

-- The expected result matrix from the addition of A and B
def C : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-2, -7,  4,  6],
    ![ 5, -7,  5, -2],
    ![13, -1, -1,  1],
    ![ 1,  1, -3, -1]]

-- The statement that A + B equals C
theorem matrix_addition_is_correct : A + B = C :=
by 
  -- Here we would provide the proof steps.
  sorry

end NUMINAMATH_GPT_matrix_addition_is_correct_l346_34690


namespace NUMINAMATH_GPT_integer_pairs_count_l346_34667

theorem integer_pairs_count : ∃ (pairs : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), (x ≥ y ∧ (x, y) ∈ pairs → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 211))
  ∧ pairs.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_integer_pairs_count_l346_34667


namespace NUMINAMATH_GPT_tv_cost_l346_34633

-- Definitions from the problem conditions
def fraction_on_furniture : ℚ := 3 / 4
def total_savings : ℚ := 1800
def fraction_on_tv : ℚ := 1 - fraction_on_furniture  -- Fraction of savings on TV

-- The proof problem statement
theorem tv_cost : total_savings * fraction_on_tv = 450 := by
  sorry

end NUMINAMATH_GPT_tv_cost_l346_34633


namespace NUMINAMATH_GPT_total_annual_donation_l346_34623

-- Defining the conditions provided in the problem
def monthly_donation : ℕ := 1707
def months_in_year : ℕ := 12

-- Stating the theorem that answers the question
theorem total_annual_donation : monthly_donation * months_in_year = 20484 := 
by
  -- The proof is omitted for brevity
  sorry

end NUMINAMATH_GPT_total_annual_donation_l346_34623


namespace NUMINAMATH_GPT_prime_remainder_30_l346_34638

theorem prime_remainder_30 (p : ℕ) (hp : Nat.Prime p) (hgt : p > 30) (hmod2 : p % 2 ≠ 0) 
(hmod3 : p % 3 ≠ 0) (hmod5 : p % 5 ≠ 0) : 
  ∃ (r : ℕ), r < 30 ∧ (p % 30 = r) ∧ (r = 1 ∨ Nat.Prime r) := 
by
  sorry

end NUMINAMATH_GPT_prime_remainder_30_l346_34638


namespace NUMINAMATH_GPT_seventh_oblong_is_56_l346_34627

def oblong (n : ℕ) : ℕ := n * (n + 1)

theorem seventh_oblong_is_56 : oblong 7 = 56 := by
  sorry

end NUMINAMATH_GPT_seventh_oblong_is_56_l346_34627


namespace NUMINAMATH_GPT_line_through_point_parallel_l346_34646

theorem line_through_point_parallel (x y : ℝ) (h₁ : 2 * 2 + 4 * 3 + x = 0) (h₂ : x = -16) (h₃ : y = 8) :
  2 * x + 4 * y - 3 = 0 → x + 2 * y - 8 = 0 :=
by
  intro h₄
  sorry

end NUMINAMATH_GPT_line_through_point_parallel_l346_34646


namespace NUMINAMATH_GPT_total_teachers_l346_34679

theorem total_teachers (total_individuals sample_size sampled_students : ℕ)
  (H1 : total_individuals = 2400)
  (H2 : sample_size = 160)
  (H3 : sampled_students = 150) :
  ∃ total_teachers, total_teachers * (sample_size / (sample_size - sampled_students)) = 2400 / (sample_size / (sample_size - sampled_students)) ∧ total_teachers = 150 := 
  sorry

end NUMINAMATH_GPT_total_teachers_l346_34679


namespace NUMINAMATH_GPT_equal_numbers_in_sequence_l346_34629

theorem equal_numbers_in_sequence (a : ℕ → ℚ)
  (h : ∀ m n : ℕ, a m + a n = a (m * n)) : 
  ∃ i j : ℕ, i ≠ j ∧ a i = a j :=
sorry

end NUMINAMATH_GPT_equal_numbers_in_sequence_l346_34629


namespace NUMINAMATH_GPT_train_passing_time_correct_l346_34685

-- Definitions of the conditions
def length_of_train : ℕ := 180  -- Length of the train in meters
def speed_of_train_km_hr : ℕ := 54  -- Speed of the train in kilometers per hour

-- Known conversion factors
def km_per_hour_to_m_per_sec (v : ℕ) : ℚ := (v * 1000) / 3600

-- Define the speed of the train in meters per second
def speed_of_train_m_per_sec : ℚ := km_per_hour_to_m_per_sec speed_of_train_km_hr

-- Define the time to pass the oak tree
def time_to_pass_oak_tree (d : ℕ) (v : ℚ) : ℚ := d / v

-- The statement to prove
theorem train_passing_time_correct :
  time_to_pass_oak_tree length_of_train speed_of_train_m_per_sec = 12 := 
by
  sorry

end NUMINAMATH_GPT_train_passing_time_correct_l346_34685


namespace NUMINAMATH_GPT_solve_equation_l346_34614

-- Define the equation as a function of y
def equation (y : ℝ) : ℝ :=
  y^4 - 20 * y + 1

-- State the theorem that y = -1 satisfies the equation.
theorem solve_equation : equation (-1) = 22 := 
  sorry

end NUMINAMATH_GPT_solve_equation_l346_34614


namespace NUMINAMATH_GPT_original_savings_l346_34604

/-- Linda spent 3/4 of her savings on furniture and the rest on a TV costing $210. 
    What were her original savings? -/
theorem original_savings (S : ℝ) (h1 : S * (1/4) = 210) : S = 840 :=
by
  sorry

end NUMINAMATH_GPT_original_savings_l346_34604


namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l346_34694

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l346_34694
