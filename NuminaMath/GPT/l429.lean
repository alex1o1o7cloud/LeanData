import Mathlib

namespace unpacked_books_30_l429_42949

theorem unpacked_books_30 :
  let total_books := 1485 * 42
  let books_per_box := 45
  total_books % books_per_box = 30 :=
by
  let total_books := 1485 * 42
  let books_per_box := 45
  have h : total_books % books_per_box = 30 := sorry
  exact h

end unpacked_books_30_l429_42949


namespace paper_clips_collected_l429_42944

theorem paper_clips_collected (boxes paper_clips_per_box total_paper_clips : ℕ) 
  (h1 : boxes = 9) 
  (h2 : paper_clips_per_box = 9) 
  (h3 : total_paper_clips = boxes * paper_clips_per_box) : 
  total_paper_clips = 81 :=
by {
  sorry
}

end paper_clips_collected_l429_42944


namespace intersection_complement_l429_42961

open Set

-- Define sets A and B as provided in the conditions
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the theorem to prove the question is equal to the answer given the conditions
theorem intersection_complement : (A ∩ compl B) = {x | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l429_42961


namespace max_value_of_expression_l429_42991

theorem max_value_of_expression (a b c : ℝ) (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) 
    (h_sum: a + b + c = 3) :
    (ab / (a + b) + ac / (a + c) + bc / (b + c) ≤ 3 / 2) :=
by
  sorry

end max_value_of_expression_l429_42991


namespace determine_c_l429_42931

noncomputable def c_floor : ℤ := -3
noncomputable def c_frac : ℝ := (25 - Real.sqrt 481) / 8

theorem determine_c : c_floor + c_frac = -2.72 := by
  have h1 : 3 * (c_floor : ℝ)^2 + 19 * (c_floor : ℝ) - 63 = 0 := by
    sorry
  have h2 : 4 * c_frac^2 - 25 * c_frac + 9 = 0 := by
    sorry
  sorry

end determine_c_l429_42931


namespace f_value_at_3_l429_42921

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_at_3 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 4 * x^2) : f 3 = -4 / 3 :=
by
  sorry

end f_value_at_3_l429_42921


namespace min_value_at_3_l429_42940

def quadratic_function (x : ℝ) : ℝ :=
  3 * x ^ 2 - 18 * x + 7

theorem min_value_at_3 : ∀ x : ℝ, quadratic_function x ≥ quadratic_function 3 :=
by
  intro x
  sorry

end min_value_at_3_l429_42940


namespace student_correct_answers_l429_42935

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 70) : C = 90 :=
sorry

end student_correct_answers_l429_42935


namespace gcd_digit_bound_l429_42975

theorem gcd_digit_bound (a b : ℕ) (h1 : a < 10^7) (h2 : b < 10^7) (h3 : 10^10 ≤ Nat.lcm a b) :
  Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digit_bound_l429_42975


namespace louise_winning_strategy_2023x2023_l429_42904

theorem louise_winning_strategy_2023x2023 :
  ∀ (n : ℕ), (n % 2 = 1) → (n = 2023) →
  ∃ (strategy : ℕ × ℕ → Prop),
    (∀ turn : ℕ, ∃ (i j : ℕ), i < n ∧ j < n ∧ strategy (i, j)) ∧
    (∃ i j : ℕ, strategy (i, j) ∧ (i = 0 ∧ j = 0)) :=
by
  sorry

end louise_winning_strategy_2023x2023_l429_42904


namespace find_symmetric_L_like_shape_l429_42907

-- Define the L-like shape and its mirror image
def L_like_shape : Type := sorry  -- Placeholder for the actual geometry definition
def mirrored_L_like_shape : Type := sorry  -- Placeholder for the actual mirrored shape

-- Condition: The vertical symmetry function
def symmetric_about_vertical_line (shape1 shape2 : Type) : Prop :=
   sorry  -- Define what it means for shape1 to be symmetric to shape2

-- Given conditions (A to E as L-like shape variations)
def option_A : Type := sorry  -- An inverted L-like shape
def option_B : Type := sorry  -- An upside-down T-like shape
def option_C : Type := mirrored_L_like_shape  -- A mirrored L-like shape
def option_D : Type := sorry  -- A rotated L-like shape by 180 degrees
def option_E : Type := L_like_shape  -- An unchanged L-like shape

-- The theorem statement
theorem find_symmetric_L_like_shape :
  symmetric_about_vertical_line L_like_shape option_C :=
  sorry

end find_symmetric_L_like_shape_l429_42907


namespace find_initial_length_of_cloth_l429_42976

noncomputable def initial_length_of_cloth : ℝ :=
  let work_rate_of_8_men := 36 / 0.75
  work_rate_of_8_men

theorem find_initial_length_of_cloth (L : ℝ) (h1 : (4:ℝ) * 2 = L / ((4:ℝ) / (L / 8)))
    (h2 : (8:ℝ) / L = 36 / 0.75) : L = 48 :=
by
  sorry

end find_initial_length_of_cloth_l429_42976


namespace find_p_l429_42955

theorem find_p (p q : ℚ) (h1 : 5 * p + 3 * q = 10) (h2 : 3 * p + 5 * q = 20) : 
  p = -5 / 8 :=
by
  sorry

end find_p_l429_42955


namespace solve_real_triples_l429_42909

theorem solve_real_triples (a b c : ℝ) :
  (a * (b^2 + c) = c * (c + a * b) ∧
   b * (c^2 + a) = a * (a + b * c) ∧
   c * (a^2 + b) = b * (b + c * a)) ↔ 
  (∃ (x : ℝ), (a = x) ∧ (b = x) ∧ (c = x)) ∨ 
  (b = 0 ∧ c = 0) :=
sorry

end solve_real_triples_l429_42909


namespace longest_piece_length_l429_42977

-- Define the lengths of the ropes
def rope1 : ℕ := 45
def rope2 : ℕ := 75
def rope3 : ℕ := 90

-- Define the greatest common divisor we need to prove
def gcd_of_ropes : ℕ := Nat.gcd rope1 (Nat.gcd rope2 rope3)

-- Goal theorem stating the problem
theorem longest_piece_length : gcd_of_ropes = 15 := by
  sorry

end longest_piece_length_l429_42977


namespace cross_country_hours_l429_42905

-- Definitions based on the conditions from part a)
def total_hours_required : ℕ := 1500
def hours_day_flying : ℕ := 50
def hours_night_flying : ℕ := 9
def goal_months : ℕ := 6
def hours_per_month : ℕ := 220

-- Problem statement: prove she has already completed 1261 hours of cross-country flying
theorem cross_country_hours : 
  (goal_months * hours_per_month) - (hours_day_flying + hours_night_flying) = 1261 := 
by
  -- Proof omitted (using the solution steps)
  sorry

end cross_country_hours_l429_42905


namespace race_distance_l429_42994

variable (speed_cristina speed_nicky head_start time_nicky : ℝ)

theorem race_distance
  (h1 : speed_cristina = 5)
  (h2 : speed_nicky = 3)
  (h3 : head_start = 12)
  (h4 : time_nicky = 30) :
  let time_cristina := time_nicky - head_start
  let distance_nicky := speed_nicky * time_nicky
  let distance_cristina := speed_cristina * time_cristina
  distance_nicky = 90 ∧ distance_cristina = 90 :=
by
  sorry

end race_distance_l429_42994


namespace math_problem_proof_l429_42952

noncomputable def question_to_equivalent_proof_problem : Prop :=
  ∃ (p q r : ℤ), 
    (p + q + r = 0) ∧ 
    (p * q + q * r + r * p = -2023) ∧ 
    (|p| + |q| + |r| = 84)

theorem math_problem_proof : question_to_equivalent_proof_problem := 
  by 
    -- proof goes here
    sorry

end math_problem_proof_l429_42952


namespace wrapping_paper_area_correct_l429_42930

-- Given conditions:
variables (w h : ℝ) -- base length and height of the box

-- Definition of the area of the wrapping paper given the problem's conditions
def wrapping_paper_area (w h : ℝ) : ℝ :=
  2 * (w + h) ^ 2

-- Theorem statement to prove the area of the wrapping paper
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  -- proof to be provided
  sorry

end wrapping_paper_area_correct_l429_42930


namespace probability_black_white_ball_l429_42981

theorem probability_black_white_ball :
  let total_balls := 5
  let black_balls := 3
  let white_balls := 2
  let favorable_outcomes := (Nat.choose 3 1) * (Nat.choose 2 1)
  let total_outcomes := Nat.choose 5 2
  (favorable_outcomes / total_outcomes) = (3 / 5) := 
by
  sorry

end probability_black_white_ball_l429_42981


namespace men_absent_l429_42947

theorem men_absent (x : ℕ) :
  let original_men := 42
  let original_days := 17
  let remaining_days := 21 
  let total_work := original_men * original_days
  let remaining_men_work := (original_men - x) * remaining_days 
  total_work = remaining_men_work →
  x = 8 :=
by
  intros
  let total_work := 42 * 17
  let remaining_men_work := (42 - x) * 21
  have h : total_work = remaining_men_work := ‹total_work = remaining_men_work›
  sorry

end men_absent_l429_42947


namespace Q_subset_P_l429_42957

-- Definitions
def P : Set ℝ := {x : ℝ | x ≥ 0}
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}

-- Statement to prove
theorem Q_subset_P : Q ⊆ P :=
sorry

end Q_subset_P_l429_42957


namespace width_of_house_l429_42913

theorem width_of_house (L P_L P_W A_total : ℝ) (hL : L = 20.5) (hPL : P_L = 6) (hPW : P_W = 4.5) (hAtotal : A_total = 232) :
  ∃ W : ℝ, W = 10 :=
by
  have area_porch : ℝ := P_L * P_W
  have area_house := A_total - area_porch
  use area_house / L
  sorry

end width_of_house_l429_42913


namespace sum_items_l429_42927

theorem sum_items (A B : ℕ) (h1 : A = 585) (h2 : A = B + 249) : A + B = 921 :=
by
  -- Proof step skipped
  sorry

end sum_items_l429_42927


namespace scientific_notation_of_0_0000023_l429_42951

theorem scientific_notation_of_0_0000023 : 
  0.0000023 = 2.3 * 10 ^ (-6) :=
by
  sorry

end scientific_notation_of_0_0000023_l429_42951


namespace non_congruent_triangles_proof_l429_42963

noncomputable def non_congruent_triangles_count : ℕ :=
  let points := [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)]
  9

theorem non_congruent_triangles_proof :
  non_congruent_triangles_count = 9 :=
sorry

end non_congruent_triangles_proof_l429_42963


namespace calc1_calc2_l429_42962

theorem calc1 : (-2) * (-1/8) = 1/4 :=
by
  sorry

theorem calc2 : (-5) / (6/5) = -25/6 :=
by
  sorry

end calc1_calc2_l429_42962


namespace solve_for_k_l429_42911

theorem solve_for_k (x k : ℝ) (h : x = -3) (h_eq : k * (x + 4) - 2 * k - x = 5) : k = -2 :=
by sorry

end solve_for_k_l429_42911


namespace sequence_sum_of_geometric_progressions_l429_42958

theorem sequence_sum_of_geometric_progressions
  (u1 v1 q p : ℝ)
  (h1 : u1 + v1 = 0)
  (h2 : u1 * q + v1 * p = 0) :
  u1 * q^2 + v1 * p^2 = 0 :=
by sorry

end sequence_sum_of_geometric_progressions_l429_42958


namespace Harry_bought_five_packets_of_chili_pepper_l429_42929

noncomputable def price_pumpkin : ℚ := 2.50
noncomputable def price_tomato : ℚ := 1.50
noncomputable def price_chili_pepper : ℚ := 0.90
noncomputable def packets_pumpkin : ℕ := 3
noncomputable def packets_tomato : ℕ := 4
noncomputable def total_spent : ℚ := 18
noncomputable def packets_chili_pepper (p : ℕ) := price_pumpkin * packets_pumpkin + price_tomato * packets_tomato + price_chili_pepper * p = total_spent

theorem Harry_bought_five_packets_of_chili_pepper :
  ∃ p : ℕ, packets_chili_pepper p ∧ p = 5 :=
by 
  sorry

end Harry_bought_five_packets_of_chili_pepper_l429_42929


namespace range_of_a_l429_42914

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

theorem range_of_a (a : ℝ) :
  (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₂ - f a x₁ > x₂ - x₁) →
  a ≥ 4 :=
by
  intro h
  sorry

end range_of_a_l429_42914


namespace probability_divisible_by_256_l429_42943

theorem probability_divisible_by_256 (n : ℕ) (h : 1 ≤ n ∧ n ≤ 1000) :
  ((n * (n + 1) * (n + 2)) % 256 = 0) → (∃ p : ℚ, p = 0.006 ∧ (∃ k : ℕ, k ≤ 1000 ∧ (n = k))) :=
sorry

end probability_divisible_by_256_l429_42943


namespace expected_value_of_winnings_l429_42970

/-- A fair 6-sided die is rolled. If the roll is even, then you win the amount of dollars 
equal to the square of the number you roll. If the roll is odd, you win nothing. 
Prove that the expected value of your winnings is 28/3 dollars. -/
theorem expected_value_of_winnings : 
  (1 / 6) * (2^2 + 4^2 + 6^2) = 28 / 3 := by
sorry

end expected_value_of_winnings_l429_42970


namespace peyton_juice_boxes_needed_l429_42906

def juice_boxes_needed
  (john_juice_per_day : ℕ)
  (samantha_juice_per_day : ℕ)
  (heather_juice_mon_wed : ℕ)
  (heather_juice_tue_thu : ℕ)
  (heather_juice_fri : ℕ)
  (john_weeks : ℕ)
  (samantha_weeks : ℕ)
  (heather_weeks : ℕ)
  : ℕ :=
  let john_juice_per_week := john_juice_per_day * 5
  let samantha_juice_per_week := samantha_juice_per_day * 5
  let heather_juice_per_week := heather_juice_mon_wed * 2 + heather_juice_tue_thu * 2 + heather_juice_fri
  let john_total_juice := john_juice_per_week * john_weeks
  let samantha_total_juice := samantha_juice_per_week * samantha_weeks
  let heather_total_juice := heather_juice_per_week * heather_weeks
  john_total_juice + samantha_total_juice + heather_total_juice

theorem peyton_juice_boxes_needed :
  juice_boxes_needed 2 1 3 2 1 25 20 25 = 625 :=
by
  sorry

end peyton_juice_boxes_needed_l429_42906


namespace range_of_x_l429_42926

open Real

theorem range_of_x (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 / a + 1 / b = 1) :
  a + b ≥ 3 + 2 * sqrt 2 :=
sorry

end range_of_x_l429_42926


namespace bill_miles_sunday_l429_42942

variables (B : ℕ)
def miles_ran_Bill_Saturday := B
def miles_ran_Bill_Sunday := B + 4
def miles_ran_Julia_Sunday := 2 * (B + 4)
def total_miles_ran := miles_ran_Bill_Saturday + miles_ran_Bill_Sunday + miles_ran_Julia_Sunday

theorem bill_miles_sunday (h1 : total_miles_ran B = 32) : 
  miles_ran_Bill_Sunday B = 9 := 
by sorry

end bill_miles_sunday_l429_42942


namespace volume_of_region_l429_42995

theorem volume_of_region : 
  ∀ (x y z : ℝ),
  abs (x + y + z) + abs (x - y + z) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → 
  ∃ V : ℝ, V = 62.5 :=
  sorry

end volume_of_region_l429_42995


namespace camel_height_in_feet_correct_l429_42988

def hare_height_in_inches : ℕ := 14
def multiplication_factor : ℕ := 24
def inches_to_feet_ratio : ℕ := 12

theorem camel_height_in_feet_correct :
  (hare_height_in_inches * multiplication_factor) / inches_to_feet_ratio = 28 := by
  sorry

end camel_height_in_feet_correct_l429_42988


namespace evaluate_expression_l429_42917

theorem evaluate_expression :
  ((3^1 - 2 + 7^3 + 1 : ℚ)⁻¹ * 6) = (2 / 115) := by
  sorry

end evaluate_expression_l429_42917


namespace problem_l429_42974

theorem problem (a b c : ℤ) (h1 : 0 < c) (h2 : c < 90) (h3 : Real.sqrt (9 - 8 * Real.sin (50 * Real.pi / 180)) = a + b * Real.sin (c * Real.pi / 180)) : 
  (a + b) / c = 1 / 2 :=
by
  sorry

end problem_l429_42974


namespace mr_jones_loss_l429_42932

theorem mr_jones_loss :
  ∃ (C_1 C_2 : ℝ), 
    (1.2 = 1.2 * C_1 / 1.2) ∧ 
    (1.2 = 0.8 * C_2) ∧ 
    ((C_1 + C_2) - (2 * 1.2)) = -0.1 :=
by
  sorry

end mr_jones_loss_l429_42932


namespace sum_of_coefficients_l429_42966

theorem sum_of_coefficients (a b c d : ℤ)
  (h1 : a + c = 2)
  (h2 : a * c + b + d = -3)
  (h3 : a * d + b * c = 7)
  (h4 : b * d = -6) :
  a + b + c + d = 7 :=
sorry

end sum_of_coefficients_l429_42966


namespace petStoreHasSixParrots_l429_42979

def petStoreParrotsProof : Prop :=
  let cages := 6.0
  let parakeets := 2.0
  let birds_per_cage := 1.333333333
  let total_birds := cages * birds_per_cage
  let number_of_parrots := total_birds - parakeets
  number_of_parrots = 6.0

theorem petStoreHasSixParrots : petStoreParrotsProof := by
  sorry

end petStoreHasSixParrots_l429_42979


namespace octagon_properties_l429_42922

-- Definitions for a regular octagon inscribed in a circle
def regular_octagon (r : ℝ) := ∀ (a b : ℝ), abs (a - b) = r
def side_length := 5
def inscribed_in_circle (r : ℝ) := ∃ (a b : ℝ), a * a + b * b = r * r

-- Main theorem statement
theorem octagon_properties (r : ℝ) (h : r = side_length) (h1 : regular_octagon r) (h2 : inscribed_in_circle r) :
  let arc_length := (5 * π) / 4
  let area_sector := (25 * π) / 8
  arc_length = (5 * π) / 4 ∧ area_sector = (25 * π) / 8 := by
  sorry

end octagon_properties_l429_42922


namespace intersection_point_of_lines_l429_42982

theorem intersection_point_of_lines
    : ∃ (x y: ℝ), y = 3 * x + 4 ∧ y = - (1 / 3) * x + 5 ∧ x = 3 / 10 ∧ y = 49 / 10 :=
by
  sorry

end intersection_point_of_lines_l429_42982


namespace convex_polygon_angles_eq_nine_l429_42983

theorem convex_polygon_angles_eq_nine (n : ℕ) (a : ℕ → ℝ) (d : ℝ)
  (h1 : a (n - 1) = 180)
  (h2 : ∀ k, a (k + 1) - a k = d)
  (h3 : d = 10) :
  n = 9 :=
by
  sorry

end convex_polygon_angles_eq_nine_l429_42983


namespace xy_sum_value_l429_42908

theorem xy_sum_value (x y : ℝ) (h1 : x + Real.cos y = 1010) (h2 : x + 1010 * Real.sin y = 1009) (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (Real.pi / 2)) :
  x + y = 1010 + (Real.pi / 2) := 
by
  sorry

end xy_sum_value_l429_42908


namespace unique_digit_for_prime_l429_42923

theorem unique_digit_for_prime (B : ℕ) (hB : B < 10) (hprime : Nat.Prime (30420 * 10 + B)) : B = 1 :=
sorry

end unique_digit_for_prime_l429_42923


namespace fraction_of_grid_covered_l429_42987

open Real EuclideanGeometry

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem fraction_of_grid_covered :
  let A := (2, 2)
  let B := (6, 2)
  let C := (4, 5)
  let grid_area := 7 * 7
  let triangle_area := area_of_triangle A B C
  triangle_area / grid_area = 6 / 49 := by
  sorry

end fraction_of_grid_covered_l429_42987


namespace largest_number_is_l429_42956

-- Define the conditions stated in the problem
def sum_of_three_numbers_is_100 (a b c : ℝ) : Prop :=
  a + b + c = 100

def two_larger_numbers_differ_by_8 (b c : ℝ) : Prop :=
  c - b = 8

def two_smaller_numbers_differ_by_5 (a b : ℝ) : Prop :=
  b - a = 5

-- Define the hypothesis
def problem_conditions (a b c : ℝ) : Prop :=
  sum_of_three_numbers_is_100 a b c ∧
  two_larger_numbers_differ_by_8 b c ∧
  two_smaller_numbers_differ_by_5 a b

-- Define the proof problem
theorem largest_number_is (a b c : ℝ) (h : problem_conditions a b c) : 
  c = 121 / 3 :=
sorry

end largest_number_is_l429_42956


namespace compute_expression_l429_42937

theorem compute_expression :
  25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := 
sorry

end compute_expression_l429_42937


namespace toll_for_18_wheel_truck_l429_42934

noncomputable def toll (x : ℕ) : ℝ :=
  2.50 + 0.50 * (x - 2)

theorem toll_for_18_wheel_truck :
  let num_wheels := 18
  let wheels_on_front_axle := 2
  let wheels_per_other_axle := 4
  let num_other_axles := (num_wheels - wheels_on_front_axle) / wheels_per_other_axle
  let total_num_axles := num_other_axles + 1
  toll total_num_axles = 4.00 :=
by
  sorry

end toll_for_18_wheel_truck_l429_42934


namespace min_ab_value_l429_42984

variable (a b : ℝ)

theorem min_ab_value (h1 : a > -1) (h2 : b > -2) (h3 : (a+1) * (b+2) = 16) : a + b ≥ 5 :=
by
  sorry

end min_ab_value_l429_42984


namespace time_to_cover_length_l429_42920

/-- Constants -/
def speed_escalator : ℝ := 10
def length_escalator : ℝ := 112
def speed_person : ℝ := 4

/-- Proof problem -/
theorem time_to_cover_length :
  (length_escalator / (speed_escalator + speed_person)) = 8 := by
  sorry

end time_to_cover_length_l429_42920


namespace strictly_increasing_and_symmetric_l429_42916

open Real

noncomputable def f1 (x : ℝ) : ℝ := x^((1 : ℝ)/2)
noncomputable def f2 (x : ℝ) : ℝ := x^((1 : ℝ)/3)
noncomputable def f3 (x : ℝ) : ℝ := x^((2 : ℝ)/3)
noncomputable def f4 (x : ℝ) : ℝ := x^(-(1 : ℝ)/3)

theorem strictly_increasing_and_symmetric : 
  ∀ f : ℝ → ℝ,
  (f = f2) ↔ 
  ((∀ x : ℝ, 0 < x → f x = x^((1 : ℝ)/3) ∧ f (-x) = -(f x)) ∧ 
   (∀ x y : ℝ, 0 < x ∧ 0 < y → (x < y → f x < f y))) :=
sorry

end strictly_increasing_and_symmetric_l429_42916


namespace intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l429_42945

noncomputable def f (x : ℝ) := x * Real.log (-x)
noncomputable def g (x a : ℝ) := x * f (a * x) - Real.exp (x - 2)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < -1 / Real.exp 1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 / Real.exp 1 < x ∧ x < 0 → deriv f x < 0) ∧
  f (-1 / Real.exp 1) = 1 / Real.exp 1 :=
sorry

theorem number_of_zeros_of_g (a : ℝ) :
  (a > 0 ∨ a = -1 / Real.exp 1 → ∃! x : ℝ, g x a = 0) ∧
  (a < 0 ∧ a ≠ -1 / Real.exp 1 → ∀ x : ℝ, g x a ≠ 0) :=
sorry

end intervals_of_monotonicity_and_extreme_values_number_of_zeros_of_g_l429_42945


namespace prove_inequality_l429_42993

theorem prove_inequality (x : ℝ) (h : 3 * x^2 + x - 8 < 0) : -2 < x ∧ x < 4 / 3 :=
sorry

end prove_inequality_l429_42993


namespace water_cost_function_solve_for_x_and_payments_l429_42972

def water_usage_A (x : ℕ) : ℕ := 5 * x
def water_usage_B (x : ℕ) : ℕ := 3 * x

def water_payment_A (x : ℕ) : ℕ :=
  if water_usage_A x <= 15 then 
    water_usage_A x * 2 
  else 
    15 * 2 + (water_usage_A x - 15) * 3

def water_payment_B (x : ℕ) : ℕ :=
  if water_usage_B x <= 15 then 
    water_usage_B x * 2 
  else 
    15 * 2 + (water_usage_B x - 15) * 3

def total_payment (x : ℕ) : ℕ := water_payment_A x + water_payment_B x

theorem water_cost_function (x : ℕ) : total_payment x =
  if 0 < x ∧ x ≤ 3 then 16 * x
  else if 3 < x ∧ x ≤ 5 then 21 * x - 15
  else if 5 < x then 24 * x - 30
  else 0 := sorry

theorem solve_for_x_and_payments (y : ℕ) : y = 114 → ∃ x, total_payment x = y ∧
  water_usage_A x = 30 ∧ water_payment_A x = 75 ∧
  water_usage_B x = 18 ∧ water_payment_B x = 39 := sorry

end water_cost_function_solve_for_x_and_payments_l429_42972


namespace fraction_cookies_blue_or_green_l429_42953

theorem fraction_cookies_blue_or_green (C : ℕ) (h1 : 1/C = 1/4) (h2 : 0.5555555555555556 = 5/9) :
  (1/4 + (5/9) * (3/4)) = (2/3) :=
by sorry

end fraction_cookies_blue_or_green_l429_42953


namespace value_of_M_l429_42915

theorem value_of_M (x y z M : ℝ) (h1 : x + y + z = 90)
    (h2 : x - 5 = M)
    (h3 : y + 5 = M)
    (h4 : 5 * z = M) :
    M = 450 / 11 :=
by
    sorry

end value_of_M_l429_42915


namespace mean_of_reciprocals_of_first_four_primes_l429_42990

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l429_42990


namespace equal_focal_distances_l429_42936

theorem equal_focal_distances (k : ℝ) (h₁ : k ≠ 0) (h₂ : 16 - k ≠ 0) 
  (h_hyperbola : ∀ x y, (x^2) / (16 - k) - (y^2) / k = 1)
  (h_ellipse : ∀ x y, 9 * x^2 + 25 * y^2 = 225) :
  0 < k ∧ k < 16 :=
sorry

end equal_focal_distances_l429_42936


namespace range_of_a_l429_42959

-- Define an odd function f on ℝ such that f(x) = x^2 for x >= 0
noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then x^2 else -(x^2)

-- Prove the range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc a (a + 2) → f (x - a) ≥ f (3 * x + 1)) →
  a ≤ -5 := sorry

end range_of_a_l429_42959


namespace smallest_n_for_triangle_area_l429_42912

theorem smallest_n_for_triangle_area :
  ∃ n : ℕ, 10 * n^4 - 8 * n^3 - 52 * n^2 + 32 * n - 24 > 10000 ∧ ∀ m : ℕ, 
  (m < n → ¬ (10 * m^4 - 8 * m^3 - 52 * m^2 + 32 * m - 24 > 10000)) :=
sorry

end smallest_n_for_triangle_area_l429_42912


namespace probability_female_wears_glasses_l429_42919

def prob_female_wears_glasses (total_females : ℕ) (females_no_glasses : ℕ) : ℚ :=
  let females_with_glasses := total_females - females_no_glasses
  females_with_glasses / total_females

theorem probability_female_wears_glasses :
  prob_female_wears_glasses 18 8 = 5 / 9 := by
  sorry  -- Proof is skipped

end probability_female_wears_glasses_l429_42919


namespace hexagon_coloring_l429_42971

-- Definitions based on conditions
variable (A B C D E F : ℕ)
variable (color : ℕ → ℕ)
variable (v1 v2 : ℕ)

-- The question is about the number of different colorings
theorem hexagon_coloring (h_distinct : ∀ (x y : ℕ), x ≠ y → color x ≠ color y) 
    (h_colors : ∀ (x : ℕ), x ∈ [A, B, C, D, E, F] → 0 < color x ∧ color x < 5) :
    4 * 3 * 3 * 3 * 3 * 3 = 972 :=
by
  sorry

end hexagon_coloring_l429_42971


namespace simplify_expression_l429_42910

theorem simplify_expression (y : ℝ) : 
  3 * y - 5 * y ^ 2 + 12 - (7 - 3 * y + 5 * y ^ 2) = -10 * y ^ 2 + 6 * y + 5 :=
by 
  sorry

end simplify_expression_l429_42910


namespace range_of_a_l429_42967

noncomputable def interval1 (a : ℝ) : Prop := -2 < a ∧ a <= 1 / 2
noncomputable def interval2 (a : ℝ) : Prop := a >= 2

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x + |x - 2 * a| > 1

theorem range_of_a (a : ℝ) (h1 : ∀ x : ℝ, p a ∨ q a) (h2 : ¬ (∀ x : ℝ, p a ∧ q a)) : 
  interval1 a ∨ interval2 a :=
sorry

end range_of_a_l429_42967


namespace polynomial_roots_sum_l429_42965

noncomputable def roots (p : Polynomial ℚ) : Set ℚ := {r | p.eval r = 0}

theorem polynomial_roots_sum :
  ∀ a b c : ℚ, (a ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (b ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  (c ∈ roots (Polynomial.C 3 - Polynomial.X * Polynomial.C 7 + Polynomial.X^2 * Polynomial.C 8 - Polynomial.X^3)) →
  a ≠ b → b ≠ c → a ≠ c →
  (a + b + c = 8) →
  (a * b + a * c + b * c = 7) →
  (a * b * c = -3) →
  (a / (b * c + 1) + b / (a * c + 1) + c / (a * b + 1) = 17 / 2) := by
    intros a b c ha hb hc hab habc hac sum_nums sum_prods prod_roots
    sorry

#check polynomial_roots_sum

end polynomial_roots_sum_l429_42965


namespace hiring_manager_acceptance_l429_42900

theorem hiring_manager_acceptance {k : ℤ} 
  (avg_age : ℤ) (std_dev : ℤ) (num_accepted_ages : ℤ) 
  (h_avg : avg_age = 20) (h_std_dev : std_dev = 8)
  (h_num_accepted : num_accepted_ages = 17) : 
  (20 + k * 8 - (20 - k * 8) + 1) = 17 → k = 1 :=
by
  intros
  sorry

end hiring_manager_acceptance_l429_42900


namespace intercept_sum_l429_42969

theorem intercept_sum (x y : ℝ) (h : y - 3 = -3 * (x + 2)) :
  (∃ (x_int : ℝ), y = 0 ∧ x_int = -1) ∧ (∃ (y_int : ℝ), x = 0 ∧ y_int = -3) →
  (-1 + (-3) = -4) := by
  sorry

end intercept_sum_l429_42969


namespace scissors_total_l429_42925

theorem scissors_total (original_scissors : ℕ) (added_scissors : ℕ) (total_scissors : ℕ) 
  (h1 : original_scissors = 39)
  (h2 : added_scissors = 13)
  (h3 : total_scissors = original_scissors + added_scissors) : total_scissors = 52 :=
by
  rw [h1, h2] at h3
  exact h3

end scissors_total_l429_42925


namespace small_bottle_sold_percentage_l429_42933

-- Definitions for initial conditions
def small_bottles_initial : ℕ := 6000
def large_bottles_initial : ℕ := 15000
def large_bottle_sold_percentage : ℝ := 0.14
def total_remaining_bottles : ℕ := 18180

-- The statement we need to prove
theorem small_bottle_sold_percentage :
  ∃ k : ℝ, (0 ≤ k ∧ k ≤ 100) ∧
  (small_bottles_initial - (k / 100) * small_bottles_initial + 
   large_bottles_initial - large_bottle_sold_percentage * large_bottles_initial = total_remaining_bottles) ∧
  (k = 12) :=
sorry

end small_bottle_sold_percentage_l429_42933


namespace sin_75_equals_sqrt_1_plus_sin_2_equals_l429_42928

noncomputable def sin_75 : ℝ := Real.sin (75 * Real.pi / 180)
noncomputable def sqrt_1_plus_sin_2 : ℝ := Real.sqrt (1 + Real.sin 2)

theorem sin_75_equals :
  sin_75 = (Real.sqrt 2 + Real.sqrt 6) / 4 := 
sorry

theorem sqrt_1_plus_sin_2_equals :
  sqrt_1_plus_sin_2 = Real.sin 1 + Real.cos 1 := 
sorry

end sin_75_equals_sqrt_1_plus_sin_2_equals_l429_42928


namespace parabola_axis_of_symmetry_l429_42968

theorem parabola_axis_of_symmetry : 
  ∀ (x : ℝ), x = -1 → (∃ y : ℝ, y = -x^2 - 2*x - 3) :=
by
  sorry

end parabola_axis_of_symmetry_l429_42968


namespace range_of_a_for_empty_solution_set_l429_42960

theorem range_of_a_for_empty_solution_set :
  {a : ℝ | ∀ x : ℝ, (a^2 - 9) * x^2 + (a + 3) * x - 1 < 0} = 
  {a : ℝ | -3 ≤ a ∧ a < 9 / 5} :=
sorry

end range_of_a_for_empty_solution_set_l429_42960


namespace part1_part2_part3_l429_42992

-- Definition of the function
def linear_function (m : ℝ) (x : ℝ) : ℝ :=
  (2 * m + 1) * x + m - 3

-- Part 1: If the graph passes through the origin
theorem part1 (h : linear_function m 0 = 0) : m = 3 :=
by {
  sorry
}

-- Part 2: If the graph is parallel to y = 3x - 3
theorem part2 (h : ∀ x, linear_function m x = 3 * x - 3 → 2 * m + 1 = 3) : m = 1 :=
by {
  sorry
}

-- Part 3: If the graph intersects the y-axis below the x-axis
theorem part3 (h_slope : 2 * m + 1 ≠ 0) (h_intercept : m - 3 < 0) : m < 3 ∧ m ≠ -1 / 2 :=
by {
  sorry
}

end part1_part2_part3_l429_42992


namespace next_divisor_of_4_digit_even_number_l429_42902

theorem next_divisor_of_4_digit_even_number (n : ℕ) (h1 : 1000 ≤ n ∧ n < 10000)
  (h2 : n % 2 = 0) (hDiv : n % 221 = 0) :
  ∃ d, d > 221 ∧ d < n ∧ d % 13 = 0 ∧ d % 17 = 0 ∧ d = 442 :=
by
  use 442
  sorry

end next_divisor_of_4_digit_even_number_l429_42902


namespace roots_polynomial_sum_l429_42978

theorem roots_polynomial_sum (p q r s : ℂ)
  (h_roots : (p, q, r, s) ∈ { (p, q, r, s) | (Polynomial.eval p (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval q (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval r (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) ∧
                                      (Polynomial.eval s (Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 10 * Polynomial.X ^ 3 + Polynomial.C 20 * Polynomial.X ^ 2 + Polynomial.C 15 * Polynomial.X + Polynomial.C 6) = 0) })
  (h_sum_two_at_a_time : p*q + p*r + p*s + q*r + q*s + r*s = 20)
  (h_product : p*q*r*s = 6) :
  1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s) = 10 / 3 := by
  sorry

end roots_polynomial_sum_l429_42978


namespace problem_solution_l429_42941

theorem problem_solution (x : ℝ) (h : x ≠ 5) : (x ≥ 8) ↔ ((x + 1) / (x - 5) ≥ 3) :=
sorry

end problem_solution_l429_42941


namespace digit_is_9_if_divisible_by_11_l429_42924

theorem digit_is_9_if_divisible_by_11 (d : ℕ) : 
  (678000 + 9000 + 800 + 90 + d) % 11 = 0 -> d = 9 := by
  sorry

end digit_is_9_if_divisible_by_11_l429_42924


namespace cube_root_simplification_l429_42986

noncomputable def cubeRoot (x : ℝ) : ℝ := x^(1/3)

theorem cube_root_simplification :
  cubeRoot 54880000 = 140 * cubeRoot 20 :=
by
  sorry

end cube_root_simplification_l429_42986


namespace math_problem_l429_42964

theorem math_problem (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^b + 3 = b^a) (h4 : 3 * a^b = b^a + 13) : 
  (a = 2) ∧ (b = 3) :=
sorry

end math_problem_l429_42964


namespace positive_difference_even_odd_sum_l429_42903

noncomputable def sum_first_n_evens (n : ℕ) : ℕ := n * (n + 1)
noncomputable def sum_first_n_odds (n : ℕ) : ℕ := n * n 

theorem positive_difference_even_odd_sum : 
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sum_even_25 - sum_odd_20 = 250 :=
by
  let sum_even_25 := sum_first_n_evens 25
  let sum_odd_20 := sum_first_n_odds 20
  sorry

end positive_difference_even_odd_sum_l429_42903


namespace determine_c_l429_42997

noncomputable def fib (n : ℕ) : ℕ :=
match n with
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem determine_c (c d : ℤ) (h1 : ∃ s : ℂ, s^2 - s - 1 = 0 ∧ (c : ℂ) * s^19 + (d : ℂ) * s^18 + 1 = 0) : 
  c = 1597 :=
by
  sorry

end determine_c_l429_42997


namespace martha_butterflies_total_l429_42998

theorem martha_butterflies_total
  (B : ℕ) (Y : ℕ) (black : ℕ)
  (h1 : B = 4)
  (h2 : Y = B / 2)
  (h3 : black = 5) :
  B + Y + black = 11 :=
by {
  -- skip proof 
  sorry 
}

end martha_butterflies_total_l429_42998


namespace find_Luisa_books_l429_42938

structure Books where
  Maddie : ℕ
  Amy : ℕ
  Amy_and_Luisa : ℕ
  Luisa : ℕ

theorem find_Luisa_books (L M A : ℕ) (hM : M = 15) (hA : A = 6) (hAL : L + A = M + 9) : L = 18 := by
  sorry

end find_Luisa_books_l429_42938


namespace searchlight_reflector_distance_l429_42980

noncomputable def parabola_vertex_distance : Rat :=
  let diameter := 60 -- in cm
  let depth := 40 -- in cm
  let x := 40 -- x-coordinate of the point
  let y := 30 -- y-coordinate of the point
  let p := (y^2) / (2 * x)
  p / 2

theorem searchlight_reflector_distance : parabola_vertex_distance = 45 / 8 := by
  sorry

end searchlight_reflector_distance_l429_42980


namespace length_AC_l429_42989

variable {A B C : Type} [Field A] [Field B] [Field C]

-- Definitions for the problem conditions
noncomputable def length_AB : ℝ := 3
noncomputable def angle_A : ℝ := Real.pi * 120 / 180
noncomputable def area_ABC : ℝ := (15 * Real.sqrt 3) / 4

-- The theorem statement
theorem length_AC (b : ℝ) (h1 : b = length_AB) (h2 : angle_A = Real.pi * 120 / 180) (h3 : area_ABC = (15 * Real.sqrt 3) / 4) : b = 5 :=
sorry

end length_AC_l429_42989


namespace value_of_x_plus_y_div_y_l429_42985

variable (w x y : ℝ)
variable (hx : w / x = 1 / 6)
variable (hy : w / y = 1 / 5)

theorem value_of_x_plus_y_div_y : (x + y) / y = 11 / 5 :=
by
  sorry

end value_of_x_plus_y_div_y_l429_42985


namespace rectangular_field_area_l429_42939

theorem rectangular_field_area (a c : ℝ) (h_a : a = 13) (h_c : c = 17) :
  ∃ b : ℝ, (b = 2 * Real.sqrt 30) ∧ (a * b = 26 * Real.sqrt 30) :=
by
  sorry

end rectangular_field_area_l429_42939


namespace triangle_inequality_l429_42950

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c R : ℝ) : ℝ := a * b * c / (4 * R)
noncomputable def inradius_area (a b c r : ℝ) : ℝ := semiperimeter a b c * r

theorem triangle_inequality (a b c R r : ℝ) (h₁ : a ≤ 1) (h₂ : b ≤ 1) (h₃ : c ≤ 1)
  (h₄ : area a b c R = semiperimeter a b c * r) : 
  semiperimeter a b c * (1 - 2 * R * r) ≥ 1 :=
by 
  -- Proof goes here
  sorry

end triangle_inequality_l429_42950


namespace number_of_pigs_l429_42948

variable (cows pigs : Nat)

theorem number_of_pigs (h1 : 2 * (7 + pigs) = 32) : pigs = 9 := by
  sorry

end number_of_pigs_l429_42948


namespace max_value_arithmetic_sequence_l429_42999

theorem max_value_arithmetic_sequence
  (a : ℕ → ℝ)
  (a1 d : ℝ)
  (h1 : a 1 = a1)
  (h_diff : ∀ n : ℕ, a (n + 1) = a n + d)
  (ha1_pos : a1 > 0)
  (hd_pos : d > 0)
  (h1_2 : a1 + (a1 + d) ≤ 60)
  (h2_3 : (a1 + d) + (a1 + 2 * d) ≤ 100) :
  5 * a1 + (a1 + 4 * d) ≤ 200 :=
sorry

end max_value_arithmetic_sequence_l429_42999


namespace light_flash_fraction_l429_42954

def light_flash_fraction_of_hour (n : ℕ) (t : ℕ) (flashes : ℕ) := 
  (n * t) / (60 * 60)

theorem light_flash_fraction (n : ℕ) (t : ℕ) (flashes : ℕ) (h1 : t = 12) (h2 : flashes = 300) : 
  light_flash_fraction_of_hour n t flashes = 1 := 
by
  sorry

end light_flash_fraction_l429_42954


namespace tom_teaching_years_l429_42918

def years_tom_has_been_teaching (x : ℝ) : Prop :=
  x + (1/2 * x - 5) = 70

theorem tom_teaching_years:
  ∃ x : ℝ, years_tom_has_been_teaching x ∧ x = 50 :=
by
  sorry

end tom_teaching_years_l429_42918


namespace smallest_n_cube_ends_with_2016_l429_42901

theorem smallest_n_cube_ends_with_2016 : ∃ n : ℕ, (n^3 % 10000 = 2016) ∧ (∀ m : ℕ, (m^3 % 10000 = 2016) → n ≤ m) :=
sorry

end smallest_n_cube_ends_with_2016_l429_42901


namespace no_values_less_than_180_l429_42996

/-- Given that w and n are positive integers less than 180 
    such that w % 13 = 2 and n % 8 = 5, 
    prove that there are no such values for w and n. -/
theorem no_values_less_than_180 (w n : ℕ) (hw : w < 180) (hn : n < 180) 
  (h1 : w % 13 = 2) (h2 : n % 8 = 5) : false :=
by
  sorry

end no_values_less_than_180_l429_42996


namespace Harriet_siblings_product_l429_42946

variable (Harry_sisters : Nat)
variable (Harry_brothers : Nat)
variable (Harriet_sisters : Nat)
variable (Harriet_brothers : Nat)

theorem Harriet_siblings_product:
  Harry_sisters = 4 -> 
  Harry_brothers = 6 ->
  Harriet_sisters = Harry_sisters -> 
  Harriet_brothers = Harry_brothers ->
  Harriet_sisters * Harriet_brothers = 24 :=
by
  intro hs hb hhs hhb
  rw [hhs, hhb]
  sorry

end Harriet_siblings_product_l429_42946


namespace tetrahedron_probability_correct_l429_42973

noncomputable def tetrahedron_probability : ℚ :=
  let total_arrangements := 16
  let suitable_arrangements := 2
  suitable_arrangements / total_arrangements

theorem tetrahedron_probability_correct : tetrahedron_probability = 1 / 8 :=
by
  sorry

end tetrahedron_probability_correct_l429_42973
