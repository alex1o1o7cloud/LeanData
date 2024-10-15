import Mathlib

namespace NUMINAMATH_GPT_convert_spherical_to_rectangular_l396_39667

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin phi * Real.cos theta, rho * Real.sin phi * Real.sin theta, rho * Real.cos phi)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 10 (4 * Real.pi / 3) (Real.pi / 3) = (-5 * Real.sqrt 3, -15 / 2, 5) :=
by 
  sorry

end NUMINAMATH_GPT_convert_spherical_to_rectangular_l396_39667


namespace NUMINAMATH_GPT_banana_pieces_l396_39674

theorem banana_pieces (B G P : ℕ) 
  (h1 : P = 4 * G)
  (h2 : G = B + 5)
  (h3 : P = 192) : B = 43 := 
by
  sorry

end NUMINAMATH_GPT_banana_pieces_l396_39674


namespace NUMINAMATH_GPT_dog_farthest_distance_l396_39648

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

end NUMINAMATH_GPT_dog_farthest_distance_l396_39648


namespace NUMINAMATH_GPT_complete_work_in_days_l396_39678

def rate_x : ℚ := 1 / 10
def rate_y : ℚ := 1 / 15
def rate_z : ℚ := 1 / 20

def combined_rate : ℚ := rate_x + rate_y + rate_z

theorem complete_work_in_days :
  1 / combined_rate = 60 / 13 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_complete_work_in_days_l396_39678


namespace NUMINAMATH_GPT_possible_values_of_a_and_b_l396_39644

theorem possible_values_of_a_and_b (a b : ℕ) : 
  (a = 22 ∨ a = 33 ∨ a = 40 ∨ a = 42) ∧ 
  (b = 21 ∨ b = 10 ∨ b = 3 ∨ b = 1) ∧ 
  (a % (b + 1) = 0) ∧ (43 % (a + b) = 0) :=
sorry

end NUMINAMATH_GPT_possible_values_of_a_and_b_l396_39644


namespace NUMINAMATH_GPT_biology_class_grades_l396_39675

theorem biology_class_grades (total_students : ℕ)
  (PA PB PC PD : ℕ)
  (h1 : PA = 12 * PB / 10)
  (h2 : PC = PB)
  (h3 : PD = 5 * PB / 10)
  (h4 : PA + PB + PC + PD = total_students) :
  total_students = 40 → PB = 11 := 
by
  sorry

end NUMINAMATH_GPT_biology_class_grades_l396_39675


namespace NUMINAMATH_GPT_find_k_l396_39600

theorem find_k (k : ℝ) (h : (-2)^2 - k * (-2) - 6 = 0) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l396_39600


namespace NUMINAMATH_GPT_harris_carrot_expense_l396_39623

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

end NUMINAMATH_GPT_harris_carrot_expense_l396_39623


namespace NUMINAMATH_GPT_cube_diagonal_length_l396_39693

theorem cube_diagonal_length (V A : ℝ) (hV : V = 384) (hA : A = 384) : 
  ∃ d : ℝ, d = 8 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_cube_diagonal_length_l396_39693


namespace NUMINAMATH_GPT_previous_spider_weight_l396_39659

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

end NUMINAMATH_GPT_previous_spider_weight_l396_39659


namespace NUMINAMATH_GPT_total_spent_by_mrs_hilt_l396_39621

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

end NUMINAMATH_GPT_total_spent_by_mrs_hilt_l396_39621


namespace NUMINAMATH_GPT_f_g_of_4_eq_18_sqrt_21_div_7_l396_39611

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x - 3

theorem f_g_of_4_eq_18_sqrt_21_div_7 : f (g 4) = (18 * Real.sqrt 21) / 7 := by
  sorry

end NUMINAMATH_GPT_f_g_of_4_eq_18_sqrt_21_div_7_l396_39611


namespace NUMINAMATH_GPT_find_x_l396_39637

theorem find_x (p q : ℕ) (h1 : 1 < p) (h2 : 1 < q) (h3 : 17 * (p + 1) = (14 * (q + 1))) (h4 : p + q = 40) : 
    x = 14 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l396_39637


namespace NUMINAMATH_GPT_find_x_l396_39615

def binary_operation (a b c d : Int) : Int × Int := (a - c, b + d)

theorem find_x (x y : Int)
  (H1 : binary_operation 6 5 2 3 = (4, 8))
  (H2 : binary_operation x y 5 4 = (4, 8)) :
  x = 9 :=
by
  -- Necessary conditions and hypotheses are provided
  sorry -- Proof not required

end NUMINAMATH_GPT_find_x_l396_39615


namespace NUMINAMATH_GPT_jogging_days_in_second_week_l396_39616

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

end NUMINAMATH_GPT_jogging_days_in_second_week_l396_39616


namespace NUMINAMATH_GPT_Aren_listening_time_l396_39694

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

end NUMINAMATH_GPT_Aren_listening_time_l396_39694


namespace NUMINAMATH_GPT_area_and_cost_of_path_l396_39683

-- Define the dimensions of the grass field
def length_field : ℝ := 85
def width_field : ℝ := 55

-- Define the width of the path around the field
def width_path : ℝ := 2.5

-- Define the cost per square meter of constructing the path
def cost_per_sqm : ℝ := 2

-- Define new dimensions including the path
def new_length : ℝ := length_field + 2 * width_path
def new_width : ℝ := width_field + 2 * width_path

-- Define the area of the entire field including the path
def area_with_path : ℝ := new_length * new_width

-- Define the area of the grass field without the path
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_with_path - area_field

-- Define the cost of constructing the path
def cost_constructing_path : ℝ := area_path * cost_per_sqm

-- Theorem to prove the area of the path and cost of constructing it
theorem area_and_cost_of_path :
  area_path = 725 ∧ cost_constructing_path = 1450 :=
by
  -- Skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_area_and_cost_of_path_l396_39683


namespace NUMINAMATH_GPT_work_completion_l396_39647

theorem work_completion (W : ℝ) (a b : ℝ) (ha : a = W / 12) (hb : b = W / 6) :
  W / (a + b) = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_work_completion_l396_39647


namespace NUMINAMATH_GPT_relayRaceOrders_l396_39691

def countRelayOrders (s1 s2 s3 s4 : String) : Nat :=
  if s1 = "Laura" then
    (if s2 ≠ "Laura" ∧ s3 ≠ "Laura" ∧ s4 ≠ "Laura" then
      if (s2 = "Alice" ∨ s2 = "Bob" ∨ s2 = "Cindy") ∧ 
         (s3 = "Alice" ∨ s3 = "Bob" ∨ s3 = "Cindy") ∧ 
         (s4 = "Alice" ∨ s4 = "Bob" ∨ s4 = "Cindy") then
        if s2 ≠ s3 ∧ s3 ≠ s4 ∧ s2 ≠ s4 then 6 else 0
      else 0
    else 0)
  else 0

theorem relayRaceOrders : countRelayOrders "Laura" "Alice" "Bob" "Cindy" = 6 := 
by sorry

end NUMINAMATH_GPT_relayRaceOrders_l396_39691


namespace NUMINAMATH_GPT_number_of_grandchildren_l396_39634

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

end NUMINAMATH_GPT_number_of_grandchildren_l396_39634


namespace NUMINAMATH_GPT_time_for_worker_C_l396_39607

theorem time_for_worker_C (time_A time_B time_total : ℝ) (time_A_pos : 0 < time_A) (time_B_pos : 0 < time_B) (time_total_pos : 0 < time_total) 
  (hA : time_A = 12) (hB : time_B = 15) (hTotal : time_total = 6) : 
  (1 / (1 / time_total - 1 / time_A - 1 / time_B) = 60) :=
by 
  sorry

end NUMINAMATH_GPT_time_for_worker_C_l396_39607


namespace NUMINAMATH_GPT_symmetric_parabola_l396_39605

def parabola1 (x : ℝ) : ℝ := (x - 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -(x + 2)^2 - 3

theorem symmetric_parabola : ∀ x y : ℝ,
  y = parabola1 x ↔ 
  (-y) = parabola2 (-x) ∧ y = -(x + 2)^2 - 3 :=
sorry

end NUMINAMATH_GPT_symmetric_parabola_l396_39605


namespace NUMINAMATH_GPT_sequence_gcd_equality_l396_39654

theorem sequence_gcd_equality (a : ℕ → ℕ) 
  (h : ∀ (i j : ℕ), i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) : 
  ∀ i, a i = i := 
sorry

end NUMINAMATH_GPT_sequence_gcd_equality_l396_39654


namespace NUMINAMATH_GPT_brownie_cost_l396_39614

theorem brownie_cost (total_money : ℕ) (num_pans : ℕ) (pieces_per_pan : ℕ) (cost_per_piece : ℕ) :
  total_money = 32 → num_pans = 2 → pieces_per_pan = 8 → cost_per_piece = total_money / (num_pans * pieces_per_pan) → 
  cost_per_piece = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_brownie_cost_l396_39614


namespace NUMINAMATH_GPT_trapezoid_area_correct_l396_39617

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

end NUMINAMATH_GPT_trapezoid_area_correct_l396_39617


namespace NUMINAMATH_GPT_largest_n_unique_k_l396_39650

theorem largest_n_unique_k : ∃ n : ℕ, (∀ k : ℤ, (8 / 15 : ℚ) < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < (7 / 13 : ℚ) → k = unique_k) ∧ n = 112 :=
sorry

end NUMINAMATH_GPT_largest_n_unique_k_l396_39650


namespace NUMINAMATH_GPT_unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l396_39640

theorem unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16 
  (b : ℝ) : 
  (∃ (x : ℝ), bx^2 + 7*x + 4 = 0 ∧ ∀ (x' : ℝ), bx^2 + 7*x' + 4 ≠ 0) ↔ b = 49 / 16 :=
by
  sorry

end NUMINAMATH_GPT_unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l396_39640


namespace NUMINAMATH_GPT_circumference_of_circle_l396_39696

theorem circumference_of_circle (R : ℝ) : 
  (C = 2 * Real.pi * R) :=
sorry

end NUMINAMATH_GPT_circumference_of_circle_l396_39696


namespace NUMINAMATH_GPT_perpendicular_line_eq_l396_39632

theorem perpendicular_line_eq (x y : ℝ) : 
  (∃ m : ℝ, (m * y + 2 * x = -5 / 2) ∧ (x - 2 * y + 3 = 0)) →
  ∃ a b c : ℝ, (a * x + b * y + c = 0) ∧ (2 * a + b = 0) ∧ c = 1 := sorry

end NUMINAMATH_GPT_perpendicular_line_eq_l396_39632


namespace NUMINAMATH_GPT_positive_integers_condition_l396_39680

theorem positive_integers_condition : ∃ n : ℕ, (n > 0) ∧ (n < 50) ∧ (∃ k : ℕ, n = k * (50 - n)) :=
sorry

end NUMINAMATH_GPT_positive_integers_condition_l396_39680


namespace NUMINAMATH_GPT_one_third_sugar_l396_39666

theorem one_third_sugar (s : ℚ) (h : s = 23 / 4) : (1 / 3) * s = 1 + 11 / 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_one_third_sugar_l396_39666


namespace NUMINAMATH_GPT_ratio_of_area_of_small_triangle_to_square_l396_39651

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

end NUMINAMATH_GPT_ratio_of_area_of_small_triangle_to_square_l396_39651


namespace NUMINAMATH_GPT_omega_eq_six_l396_39663

theorem omega_eq_six (A ω : ℝ) (φ : ℝ) (f : ℝ → ℝ) (h1 : A ≠ 0) (h2 : ω > 0)
  (h3 : -π / 2 < φ ∧ φ < π / 2) (h4 : ∀ x, f x = A * Real.sin (ω * x + φ))
  (h5 : ∀ x, f (-x) = -f x) 
  (h6 : ∀ x, f (x + π / 6) = -f (x - π / 6)) :
  ω = 6 :=
sorry

end NUMINAMATH_GPT_omega_eq_six_l396_39663


namespace NUMINAMATH_GPT_inequality_proof_l396_39653

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (sqrt (a^2 + 8 * b * c) / a + sqrt (b^2 + 8 * a * c) / b + sqrt (c^2 + 8 * a * b) / c) ≥ 9 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l396_39653


namespace NUMINAMATH_GPT_henry_time_around_track_l396_39610

theorem henry_time_around_track (H : ℕ) : 
  (∀ (M := 12), lcm M H = 84) → H = 7 :=
by
  sorry

end NUMINAMATH_GPT_henry_time_around_track_l396_39610


namespace NUMINAMATH_GPT_find_remainder_of_n_l396_39638

theorem find_remainder_of_n (n k d : ℕ) (hn_pos : n > 0) (hk_pos : k > 0) (hd_pos_digits : d < 10^k) 
  (h : n * 10^k + d = n * (n + 1) / 2) : n % 9 = 1 :=
sorry

end NUMINAMATH_GPT_find_remainder_of_n_l396_39638


namespace NUMINAMATH_GPT_simplify_expr_l396_39689

theorem simplify_expr (a : ℝ) (h : a > 1) : (1 - a) * (1 / (a - 1)).sqrt = -(a - 1).sqrt :=
sorry

end NUMINAMATH_GPT_simplify_expr_l396_39689


namespace NUMINAMATH_GPT_hawks_score_l396_39664

theorem hawks_score (a b : ℕ) (h1 : a + b = 58) (h2 : a - b = 12) : b = 23 :=
by
  sorry

end NUMINAMATH_GPT_hawks_score_l396_39664


namespace NUMINAMATH_GPT_tangent_at_5_eqn_l396_39646

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_period : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom tangent_at_neg1 : ∀ x y : ℝ, x - y + 3 = 0 → x = -1 → y = f x

theorem tangent_at_5_eqn : 
  ∀ x y : ℝ, x = 5 → y = f x → x + y - 7 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_at_5_eqn_l396_39646


namespace NUMINAMATH_GPT_part2_proof_l396_39620

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.log x) - Real.exp 1 * x

theorem part2_proof (x : ℝ) (h : 0 < x) :
  x * f x - Real.exp x + 2 * Real.exp 1 * x ≤ 0 := 
sorry

end NUMINAMATH_GPT_part2_proof_l396_39620


namespace NUMINAMATH_GPT_find_a_min_value_of_f_l396_39671

theorem find_a (a : ℕ) (h1 : 3 / 2 < 2 + a) (h2 : 1 / 2 ≥ 2 - a) : a = 1 := by
  sorry

theorem min_value_of_f (a x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) : 
    (a = 1) → ∃ m : ℝ, m = 3 ∧ ∀ x : ℝ, |x + a| + |x - 2| ≥ m := by
  sorry

end NUMINAMATH_GPT_find_a_min_value_of_f_l396_39671


namespace NUMINAMATH_GPT_positive_integers_count_l396_39606

theorem positive_integers_count (n : ℕ) : 
  ∃ m : ℕ, (m ≤ n / 2014 ∧ m ≤ n / 2016 ∧ (m + 1) * 2014 > n ∧ (m + 1) * 2016 > n) ↔
  (n = 1015056) :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_count_l396_39606


namespace NUMINAMATH_GPT_total_sales_calculation_l396_39697

def average_price_per_pair : ℝ := 9.8
def number_of_pairs_sold : ℕ := 70
def total_amount : ℝ := 686

theorem total_sales_calculation :
  average_price_per_pair * (number_of_pairs_sold : ℝ) = total_amount :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_sales_calculation_l396_39697


namespace NUMINAMATH_GPT_horner_method_operations_l396_39633

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

end NUMINAMATH_GPT_horner_method_operations_l396_39633


namespace NUMINAMATH_GPT_shanghai_population_scientific_notation_l396_39625

theorem shanghai_population_scientific_notation :
  16.3 * 10^6 = 1.63 * 10^7 :=
sorry

end NUMINAMATH_GPT_shanghai_population_scientific_notation_l396_39625


namespace NUMINAMATH_GPT_sum_of_k_l396_39603

theorem sum_of_k (k : ℕ) :
  ((∃ x, x^2 - 4 * x + 3 = 0 ∧ x^2 - 7 * x + k = 0) →
  (k = 6 ∨ k = 12)) →
  (6 + 12 = 18) :=
by sorry

end NUMINAMATH_GPT_sum_of_k_l396_39603


namespace NUMINAMATH_GPT_find_ages_of_son_daughter_and_niece_l396_39618

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

end NUMINAMATH_GPT_find_ages_of_son_daughter_and_niece_l396_39618


namespace NUMINAMATH_GPT_complex_number_sum_zero_l396_39684

theorem complex_number_sum_zero (a b : ℝ) (i : ℂ) (h : a + b * i = 1 - i) : a + b = 0 := 
by sorry

end NUMINAMATH_GPT_complex_number_sum_zero_l396_39684


namespace NUMINAMATH_GPT_total_votes_l396_39619

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

end NUMINAMATH_GPT_total_votes_l396_39619


namespace NUMINAMATH_GPT_inequality_and_equality_hold_l396_39645

theorem inequality_and_equality_hold (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ a = b) :=
sorry

end NUMINAMATH_GPT_inequality_and_equality_hold_l396_39645


namespace NUMINAMATH_GPT_intersection_eq_l396_39686

def M (x : ℝ) : Prop := (x + 3) * (x - 2) < 0

def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

def intersection (x : ℝ) : Prop := M x ∧ N x

theorem intersection_eq : ∀ x, intersection x ↔ (1 ≤ x ∧ x < 2) :=
by sorry

end NUMINAMATH_GPT_intersection_eq_l396_39686


namespace NUMINAMATH_GPT_triangle_inequality_of_three_l396_39668

theorem triangle_inequality_of_three (x y z : ℝ) :
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := 
sorry

end NUMINAMATH_GPT_triangle_inequality_of_three_l396_39668


namespace NUMINAMATH_GPT_distance_between_parallel_lines_correct_l396_39687

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

end NUMINAMATH_GPT_distance_between_parallel_lines_correct_l396_39687


namespace NUMINAMATH_GPT_parabola_equation_conditions_l396_39699

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

end NUMINAMATH_GPT_parabola_equation_conditions_l396_39699


namespace NUMINAMATH_GPT_range_of_a_l396_39681

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → a * x^2 - x - 4 > 0) → a > 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l396_39681


namespace NUMINAMATH_GPT_intersection_eq_l396_39655

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_eq : A ∩ B = ({-2, 2} : Set ℤ) :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l396_39655


namespace NUMINAMATH_GPT_olivia_used_pieces_l396_39609

-- Definition of initial pieces of paper and remaining pieces of paper
def initial_pieces : ℕ := 81
def remaining_pieces : ℕ := 25

-- Prove that Olivia used 56 pieces of paper
theorem olivia_used_pieces : (initial_pieces - remaining_pieces) = 56 :=
by
  -- Proof steps can be filled here
  sorry

end NUMINAMATH_GPT_olivia_used_pieces_l396_39609


namespace NUMINAMATH_GPT_meeting_percentage_l396_39629

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

end NUMINAMATH_GPT_meeting_percentage_l396_39629


namespace NUMINAMATH_GPT_math_equivalent_problem_l396_39627

noncomputable def correct_difference (A B C D : ℕ) (incorrect_difference : ℕ) : ℕ :=
  if (B = 3) ∧ (D = 2) ∧ (C = 5) ∧ (incorrect_difference = 60) then
    ((A * 10 + B) - 52)
  else
    0

theorem math_equivalent_problem (A : ℕ) : correct_difference A 3 5 2 60 = 31 :=
by
  sorry

end NUMINAMATH_GPT_math_equivalent_problem_l396_39627


namespace NUMINAMATH_GPT_find_certain_number_l396_39656

theorem find_certain_number (x : ℝ) : 
  ((2 * (x + 5)) / 5 - 5 = 22) → x = 62.5 :=
by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_certain_number_l396_39656


namespace NUMINAMATH_GPT_minimum_value_f_l396_39676

theorem minimum_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : (∃ x y : ℝ, (x + y + x * y) ≥ - (9 / 8) ∧ 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :=
sorry

end NUMINAMATH_GPT_minimum_value_f_l396_39676


namespace NUMINAMATH_GPT_jenn_money_left_over_l396_39652

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

end NUMINAMATH_GPT_jenn_money_left_over_l396_39652


namespace NUMINAMATH_GPT_evaluate_expression_l396_39672

theorem evaluate_expression (x y z : ℚ) 
    (hx : x = 1 / 4) 
    (hy : y = 1 / 3) 
    (hz : z = -6) : 
    x^2 * y^3 * z^2 = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l396_39672


namespace NUMINAMATH_GPT_inequality_2n_1_lt_n_plus_1_sq_l396_39622

theorem inequality_2n_1_lt_n_plus_1_sq (n : ℕ) (h : 0 < n) : 2 * n - 1 < (n + 1) ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_2n_1_lt_n_plus_1_sq_l396_39622


namespace NUMINAMATH_GPT_closest_ratio_one_l396_39639

theorem closest_ratio_one (a c : ℕ) (h1 : 30 * a + 15 * c = 2700) (h2 : a ≥ 1) (h3 : c ≥ 1) :
  a = c :=
by sorry

end NUMINAMATH_GPT_closest_ratio_one_l396_39639


namespace NUMINAMATH_GPT_smallest_root_of_unity_l396_39643

open Complex

theorem smallest_root_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ k : ℕ, k < 18 ∧ z = exp (2 * pi * I * k / 18) :=
by
  sorry

end NUMINAMATH_GPT_smallest_root_of_unity_l396_39643


namespace NUMINAMATH_GPT_students_in_school_B_l396_39602

theorem students_in_school_B 
    (A B C : ℕ) 
    (h1 : A + C = 210) 
    (h2 : A = 4 * B) 
    (h3 : C = 3 * B) : 
    B = 30 := 
by 
    sorry

end NUMINAMATH_GPT_students_in_school_B_l396_39602


namespace NUMINAMATH_GPT_luis_bought_6_pairs_of_blue_socks_l396_39673

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

end NUMINAMATH_GPT_luis_bought_6_pairs_of_blue_socks_l396_39673


namespace NUMINAMATH_GPT_max_average_hours_l396_39670

theorem max_average_hours :
  let hours_Wednesday := 2
  let hours_Thursday := 2
  let hours_Friday := hours_Wednesday + 3
  let total_hours := hours_Wednesday + hours_Thursday + hours_Friday
  let average_hours := total_hours / 3
  average_hours = 3 :=
by
  sorry

end NUMINAMATH_GPT_max_average_hours_l396_39670


namespace NUMINAMATH_GPT_regionA_regionC_area_ratio_l396_39624

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

end NUMINAMATH_GPT_regionA_regionC_area_ratio_l396_39624


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l396_39679

theorem repeating_decimal_fraction :
  ∃ (a b : ℕ), (0 ≤ a) ∧ (0 < b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 134) ∧ 
  ((a : ℚ) / b = 35 / 99) :=
by {
  sorry
}

end NUMINAMATH_GPT_repeating_decimal_fraction_l396_39679


namespace NUMINAMATH_GPT_negation_of_proposition_l396_39685

theorem negation_of_proposition :
  (¬ ∃ m : ℝ, 1 / (m^2 + m - 6) > 0) ↔ (∀ m : ℝ, (1 / (m^2 + m - 6) < 0) ∨ (m^2 + m - 6 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l396_39685


namespace NUMINAMATH_GPT_rectangle_area_l396_39658

theorem rectangle_area (P W : ℝ) (hP : P = 52) (hW : W = 11) :
  ∃ A L : ℝ, (2 * L + 2 * W = P) ∧ (A = L * W) ∧ (A = 165) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l396_39658


namespace NUMINAMATH_GPT_ratio_of_raspberries_l396_39642

theorem ratio_of_raspberries (B R K L : ℕ) (h1 : B = 42) (h2 : L = 7) (h3 : K = B / 3) (h4 : B = R + K + L) :
  R / Nat.gcd R B = 1 ∧ B / Nat.gcd R B = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_raspberries_l396_39642


namespace NUMINAMATH_GPT_probability_two_face_cards_l396_39613

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

end NUMINAMATH_GPT_probability_two_face_cards_l396_39613


namespace NUMINAMATH_GPT_cos_alpha_l396_39649

theorem cos_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2) 
  (h2 : Real.sin (α - π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_l396_39649


namespace NUMINAMATH_GPT_greatest_multiple_5_7_less_than_700_l396_39608

theorem greatest_multiple_5_7_less_than_700 :
  ∃ n, n < 700 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ (∀ m, m < 700 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≤ n) → n = 665 :=
by
  sorry

end NUMINAMATH_GPT_greatest_multiple_5_7_less_than_700_l396_39608


namespace NUMINAMATH_GPT_maximum_value_l396_39660

variables (a b c : ℝ)
variables (a_vec b_vec c_vec : EuclideanSpace ℝ (Fin 3))

axiom norm_a : ‖a_vec‖ = 2
axiom norm_b : ‖b_vec‖ = 3
axiom norm_c : ‖c_vec‖ = 4

theorem maximum_value : 
  (‖(a_vec - (3:ℝ) • b_vec)‖^2 + ‖(b_vec - (3:ℝ) • c_vec)‖^2 + ‖(c_vec - (3:ℝ) • a_vec)‖^2) ≤ 377 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_l396_39660


namespace NUMINAMATH_GPT_target_has_more_tools_l396_39641

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

end NUMINAMATH_GPT_target_has_more_tools_l396_39641


namespace NUMINAMATH_GPT_sphere_circumscribed_around_cone_radius_l396_39677

-- Definitions of the given conditions
variable (r h : ℝ)

-- Theorem statement (without the proof)
theorem sphere_circumscribed_around_cone_radius :
  ∃ R : ℝ, R = (Real.sqrt (r^2 + h^2)) / 2 :=
sorry

end NUMINAMATH_GPT_sphere_circumscribed_around_cone_radius_l396_39677


namespace NUMINAMATH_GPT_incorrect_statement_is_C_l396_39662

theorem incorrect_statement_is_C (b h s a x : ℝ) (hb : b > 0) (hh : h > 0) (hs : s > 0) (hx : x < 0) :
  ¬ (9 * s^2 = 4 * (3 * s)^2) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_is_C_l396_39662


namespace NUMINAMATH_GPT_min_value_of_F_l396_39626

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

end NUMINAMATH_GPT_min_value_of_F_l396_39626


namespace NUMINAMATH_GPT_find_other_number_l396_39665

theorem find_other_number (HCF LCM a b : ℕ) (h1 : HCF = 108) (h2 : LCM = 27720) (h3 : a = 216) (h4 : HCF * LCM = a * b) : b = 64 :=
  sorry

end NUMINAMATH_GPT_find_other_number_l396_39665


namespace NUMINAMATH_GPT_tylers_age_l396_39630

theorem tylers_age (B T : ℕ) 
  (h1 : T = B - 3) 
  (h2 : T + B = 11) : 
  T = 4 :=
sorry

end NUMINAMATH_GPT_tylers_age_l396_39630


namespace NUMINAMATH_GPT_P_intersection_complement_Q_l396_39661

-- Define sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }

-- Prove the required intersection
theorem P_intersection_complement_Q : P ∩ (Set.univ \ Q) = { x | 0 ≤ x ∧ x < 2 } :=
by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_P_intersection_complement_Q_l396_39661


namespace NUMINAMATH_GPT_books_remainder_l396_39698

theorem books_remainder (total_books new_books_per_section sections : ℕ) 
  (h1 : total_books = 1521) 
  (h2 : new_books_per_section = 45) 
  (h3 : sections = 41) : 
  (total_books * sections) % new_books_per_section = 36 :=
by
  sorry

end NUMINAMATH_GPT_books_remainder_l396_39698


namespace NUMINAMATH_GPT_narrow_black_stripes_are_eight_l396_39604

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end NUMINAMATH_GPT_narrow_black_stripes_are_eight_l396_39604


namespace NUMINAMATH_GPT_find_g_expression_l396_39628

theorem find_g_expression (g f : ℝ → ℝ) (h_sym : ∀ x y, g x = y ↔ g (2 - x) = 4 - y)
  (h_f : ∀ x, f x = 3 * x - 1) :
  ∀ x, g x = 3 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_find_g_expression_l396_39628


namespace NUMINAMATH_GPT_solve_for_x_l396_39631

theorem solve_for_x (x : ℝ) (h : (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4)) : x = -14 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l396_39631


namespace NUMINAMATH_GPT_part_a_part_b_part_c_part_d_part_e_part_f_l396_39692

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

end NUMINAMATH_GPT_part_a_part_b_part_c_part_d_part_e_part_f_l396_39692


namespace NUMINAMATH_GPT_sequence_geometric_progression_l396_39688

theorem sequence_geometric_progression (p : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = p * a n + 2^n)
  (h3 : ∀ n : ℕ, 0 < n → a (n + 1)^2 = a n * a (n + 2)): 
  ∃ p : ℝ, ∀ n : ℕ, a n = 2^n :=
by
  sorry

end NUMINAMATH_GPT_sequence_geometric_progression_l396_39688


namespace NUMINAMATH_GPT_half_angle_quadrants_l396_39682

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

end NUMINAMATH_GPT_half_angle_quadrants_l396_39682


namespace NUMINAMATH_GPT_expression_value_at_neg1_l396_39636

theorem expression_value_at_neg1
  (p q : ℤ)
  (h1 : p + q = 2016) :
  p * (-1)^3 + q * (-1) - 10 = -2026 := by
  sorry

end NUMINAMATH_GPT_expression_value_at_neg1_l396_39636


namespace NUMINAMATH_GPT_cos_sum_simplified_l396_39669

theorem cos_sum_simplified :
  (Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)) = ((Real.sqrt 13 - 1) / 4) :=
by
  sorry

end NUMINAMATH_GPT_cos_sum_simplified_l396_39669


namespace NUMINAMATH_GPT_smallest_n_for_gn_gt_20_l396_39695

def g (n : ℕ) : ℕ := sorry -- definition of the sum of the digits to the right of the decimal of 1 / 3^n

theorem smallest_n_for_gn_gt_20 : ∃ n : ℕ, n > 0 ∧ g n > 20 ∧ ∀ m, 0 < m ∧ m < n -> g m ≤ 20 :=
by
  -- here should be the proof
  sorry

end NUMINAMATH_GPT_smallest_n_for_gn_gt_20_l396_39695


namespace NUMINAMATH_GPT_equation_correct_l396_39690

variable (x y : ℝ)

-- Define the conditions
def condition1 : Prop := (x + y) / 3 = 1.888888888888889
def condition2 : Prop := 2 * x + y = 7

-- Prove the required equation under given conditions
theorem equation_correct : condition1 x y → condition2 x y → (x + y) = 5.666666666666667 := by
  intros _ _
  sorry

end NUMINAMATH_GPT_equation_correct_l396_39690


namespace NUMINAMATH_GPT_largest_three_digit_number_l396_39601

theorem largest_three_digit_number :
  ∃ (n : ℕ), (n < 1000) ∧ (n % 7 = 1) ∧ (n % 8 = 4) ∧ (∀ (m : ℕ), (m < 1000) ∧ (m % 7 = 1) ∧ (m % 8 = 4) → m ≤ n) :=
sorry

end NUMINAMATH_GPT_largest_three_digit_number_l396_39601


namespace NUMINAMATH_GPT_loan_duration_in_years_l396_39635

-- Define the conditions as constants
def carPrice : ℝ := 20000
def downPayment : ℝ := 5000
def monthlyPayment : ℝ := 250

-- Define the goal
theorem loan_duration_in_years :
  (carPrice - downPayment) / monthlyPayment / 12 = 5 := 
sorry

end NUMINAMATH_GPT_loan_duration_in_years_l396_39635


namespace NUMINAMATH_GPT_stratified_sampling_group_C_l396_39657

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

end NUMINAMATH_GPT_stratified_sampling_group_C_l396_39657


namespace NUMINAMATH_GPT_cover_faces_with_strips_l396_39612

theorem cover_faces_with_strips (a b c : ℕ) :
  (∃ f g h : ℕ, a = 5 * f ∨ b = 5 * g ∨ c = 5 * h) ↔
  (∃ u v : ℕ, (a = 5 * u ∧ b = 5 * v) ∨ (a = 5 * u ∧ c = 5 * v) ∨ (b = 5 * u ∧ c = 5 * v)) := 
sorry

end NUMINAMATH_GPT_cover_faces_with_strips_l396_39612
