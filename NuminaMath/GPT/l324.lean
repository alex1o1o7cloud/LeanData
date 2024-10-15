import Mathlib

namespace NUMINAMATH_GPT_chromium_percentage_alloy_l324_32458

theorem chromium_percentage_alloy 
  (w1 w2 w3 w4 : ℝ)
  (p1 p2 p3 p4 : ℝ)
  (h_w1 : w1 = 15)
  (h_w2 : w2 = 30)
  (h_w3 : w3 = 10)
  (h_w4 : w4 = 5)
  (h_p1 : p1 = 0.12)
  (h_p2 : p2 = 0.08)
  (h_p3 : p3 = 0.15)
  (h_p4 : p4 = 0.20) :
  (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4) / (w1 + w2 + w3 + w4) * 100 = 11.17 := 
  sorry

end NUMINAMATH_GPT_chromium_percentage_alloy_l324_32458


namespace NUMINAMATH_GPT_Ricardo_coin_difference_l324_32431

theorem Ricardo_coin_difference (p : ℕ) (h₁ : 1 ≤ p) (h₂ : p ≤ 3029) :
    let max_value := 15150 - 4 * 1
    let min_value := 15150 - 4 * 3029
    max_value - min_value = 12112 := by
  sorry

end NUMINAMATH_GPT_Ricardo_coin_difference_l324_32431


namespace NUMINAMATH_GPT_find_f_expression_l324_32406

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_expression (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  f (x) = (1 / (x - 1)) :=
by sorry

example (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) (hx: f (1 / x) = x / (1 - x)) :
  f x = 1 / (x - 1) :=
find_f_expression x h₀ h₁

end NUMINAMATH_GPT_find_f_expression_l324_32406


namespace NUMINAMATH_GPT_polygon_interior_angle_144_proof_l324_32448

-- Definitions based on the conditions in the problem statement
def interior_angle (n : ℕ) : ℝ := 144
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- The problem statement as a Lean 4 theorem to prove n = 10
theorem polygon_interior_angle_144_proof : ∃ n : ℕ, interior_angle n = 144 ∧ sum_of_interior_angles n = n * 144 → n = 10 := by
  sorry

end NUMINAMATH_GPT_polygon_interior_angle_144_proof_l324_32448


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l324_32461

theorem no_real_roots_of_quadratic (k : ℝ) (h : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0 :=
by sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l324_32461


namespace NUMINAMATH_GPT_numbers_from_five_threes_l324_32472

theorem numbers_from_five_threes :
  (∃ (a b c d e : ℤ), (3*a + 3*b + 3*c + 3*d + 3*e = 11 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 12 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 13 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 14 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 15) ) :=
by
  -- Proof provided by the problem statement steps, using:
  -- 11 = (33/3)
  -- 12 = 3 * 3 + 3 + 3 - 3
  -- 13 = 3 * 3 + 3 + 3/3
  -- 14 = (33 + 3 * 3) / 3
  -- 15 = 3 + 3 + 3 + 3 + 3
  sorry

end NUMINAMATH_GPT_numbers_from_five_threes_l324_32472


namespace NUMINAMATH_GPT_original_perimeter_l324_32426

theorem original_perimeter (a b : ℝ) (h : a / 2 + b / 2 = 129 / 2) : 2 * (a + b) = 258 :=
by
  sorry

end NUMINAMATH_GPT_original_perimeter_l324_32426


namespace NUMINAMATH_GPT_percentage_increase_of_return_trip_l324_32477

noncomputable def speed_increase_percentage (initial_speed avg_speed : ℝ) : ℝ :=
  ((2 * avg_speed * initial_speed) / avg_speed - initial_speed) * 100 / initial_speed

theorem percentage_increase_of_return_trip :
  let initial_speed := 30
  let avg_speed := 34.5
  speed_increase_percentage initial_speed avg_speed = 35.294 :=
  sorry

end NUMINAMATH_GPT_percentage_increase_of_return_trip_l324_32477


namespace NUMINAMATH_GPT_allens_mothers_age_l324_32450

-- Define the conditions
variables (A M S : ℕ) -- Declare variables for ages of Allen, his mother, and his sister

-- Define Allen is 30 years younger than his mother
axiom h1 : A = M - 30

-- Define Allen's sister is 5 years older than him
axiom h2 : S = A + 5

-- Define in 7 years, the sum of their ages will be 110
axiom h3 : (A + 7) + (M + 7) + (S + 7) = 110

-- Define the age difference between Allen's mother and sister is 25 years
axiom h4 : M - S = 25

-- State the theorem: what is the present age of Allen's mother
theorem allens_mothers_age : M = 48 :=
by sorry

end NUMINAMATH_GPT_allens_mothers_age_l324_32450


namespace NUMINAMATH_GPT_calculate_expression_l324_32484

theorem calculate_expression :
  (-0.125)^2022 * 8^2023 = 8 :=
sorry

end NUMINAMATH_GPT_calculate_expression_l324_32484


namespace NUMINAMATH_GPT_determine_x_l324_32470

theorem determine_x (x : Nat) (h1 : x % 9 = 0) (h2 : x^2 > 225) (h3 : x < 30) : x = 18 ∨ x = 27 :=
sorry

end NUMINAMATH_GPT_determine_x_l324_32470


namespace NUMINAMATH_GPT_total_students_l324_32497

variable (T : ℕ)

-- Conditions
def is_girls_percentage (T : ℕ) := 60 / 100 * T
def is_boys_percentage (T : ℕ) := 40 / 100 * T
def boys_not_in_clubs (number_of_boys : ℕ) := 2 / 3 * number_of_boys

theorem total_students (h1 : is_girls_percentage T + is_boys_percentage T = T)
  (h2 : boys_not_in_clubs (is_boys_percentage T) = 40) : T = 150 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l324_32497


namespace NUMINAMATH_GPT_biscuits_more_than_cookies_l324_32451

theorem biscuits_more_than_cookies :
  let morning_butter_cookies := 20
  let morning_biscuits := 40
  let afternoon_butter_cookies := 10
  let afternoon_biscuits := 20
  let total_butter_cookies := morning_butter_cookies + afternoon_butter_cookies
  let total_biscuits := morning_biscuits + afternoon_biscuits
  total_biscuits - total_butter_cookies = 30 :=
by
  sorry

end NUMINAMATH_GPT_biscuits_more_than_cookies_l324_32451


namespace NUMINAMATH_GPT_original_weight_l324_32465

namespace MarbleProblem

def remainingWeightAfterCuts (w : ℝ) : ℝ :=
  w * 0.70 * 0.70 * 0.85

theorem original_weight (w : ℝ) : remainingWeightAfterCuts w = 124.95 → w = 299.94 :=
by
  intros h
  sorry

end MarbleProblem

end NUMINAMATH_GPT_original_weight_l324_32465


namespace NUMINAMATH_GPT_neg_one_to_zero_l324_32468

theorem neg_one_to_zero : (-1 : ℤ)^0 = 1 := 
by 
  -- Expanding expressions and applying the exponent rule for non-zero numbers
  sorry

end NUMINAMATH_GPT_neg_one_to_zero_l324_32468


namespace NUMINAMATH_GPT_cups_per_serving_l324_32466

-- Define the conditions
def total_cups : ℕ := 18
def servings : ℕ := 9

-- State the theorem to prove the answer
theorem cups_per_serving : total_cups / servings = 2 := by
  sorry

end NUMINAMATH_GPT_cups_per_serving_l324_32466


namespace NUMINAMATH_GPT_decrease_in_area_of_equilateral_triangle_l324_32416

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

theorem decrease_in_area_of_equilateral_triangle :
  (equilateral_triangle_area 20 - equilateral_triangle_area 14) = 51 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_decrease_in_area_of_equilateral_triangle_l324_32416


namespace NUMINAMATH_GPT_find_f7_l324_32423

noncomputable def f : ℝ → ℝ := sorry

-- The conditions provided in the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom function_in_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The final proof goal
theorem find_f7 : f 7 = -2 :=
by sorry

end NUMINAMATH_GPT_find_f7_l324_32423


namespace NUMINAMATH_GPT_min_value_fraction_l324_32462

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 19) ∧ (∀ z : ℝ, (z = (x + 15) / Real.sqrt (x - 4)) → z ≥ y) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l324_32462


namespace NUMINAMATH_GPT_simplify_2A_minus_B_value_2A_minus_B_a_eq_neg1_b_eq_2_find_b_independent_of_a_l324_32436

-- Definitions from the conditions
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 + 6 * a * b + 8 * a

-- Part (1): Simplifying 2A - B
theorem simplify_2A_minus_B (a b : ℝ) : 
  2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a := 
by
  sorry

-- Part (2): Finding 2A - B for specific a and b
theorem value_2A_minus_B_a_eq_neg1_b_eq_2 : 
  2 * A (-1) 2 - B (-1) 2 = 52 := 
by 
  sorry

-- Part (3): Finding b for which 2A - B is independent of a
theorem find_b_independent_of_a (a b : ℝ) (h : 2 * A a b - B a b = 6 * b) : 
  b = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_2A_minus_B_value_2A_minus_B_a_eq_neg1_b_eq_2_find_b_independent_of_a_l324_32436


namespace NUMINAMATH_GPT_calculation_result_l324_32449

theorem calculation_result : (18 * 23 - 24 * 17) / 3 + 5 = 7 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l324_32449


namespace NUMINAMATH_GPT_charity_ticket_sales_l324_32444

theorem charity_ticket_sales
  (x y p : ℕ)
  (h1 : x + y = 200)
  (h2 : x * p + y * (p / 2) = 3501)
  (h3 : x = 3 * y) :
  150 * 20 = 3000 :=
by
  sorry

end NUMINAMATH_GPT_charity_ticket_sales_l324_32444


namespace NUMINAMATH_GPT_probability_exactly_three_even_l324_32482

theorem probability_exactly_three_even (p : ℕ → ℚ) (n : ℕ) (k : ℕ) (h : p 20 = 1/2 ∧ n = 5 ∧ k = 3) :
  (∃ C : ℚ, (C = (Nat.choose n k : ℚ)) ∧ (p 20)^n = 1/32) → (C * 1/32 = 5/16) :=
by
  sorry

end NUMINAMATH_GPT_probability_exactly_three_even_l324_32482


namespace NUMINAMATH_GPT_hiker_speeds_l324_32434

theorem hiker_speeds:
  ∃ (d : ℝ), 
  (d > 5) ∧ ((70 / (d - 5)) = (110 / d)) ∧ (d - 5 = 8.75) :=
by
  sorry

end NUMINAMATH_GPT_hiker_speeds_l324_32434


namespace NUMINAMATH_GPT_ab_product_eq_2_l324_32476

theorem ab_product_eq_2 (a b : ℝ) (h : (a + 1)^2 + (b + 2)^2 = 0) : a * b = 2 :=
by sorry

end NUMINAMATH_GPT_ab_product_eq_2_l324_32476


namespace NUMINAMATH_GPT_positive_difference_solutions_l324_32422

theorem positive_difference_solutions (r₁ r₂ : ℝ) (h_r₁ : (r₁^2 - 5 * r₁ - 22) / (r₁ + 4) = 3 * r₁ + 8) (h_r₂ : (r₂^2 - 5 * r₂ - 22) / (r₂ + 4) = 3 * r₂ + 8) (h_r₁_ne : r₁ ≠ -4) (h_r₂_ne : r₂ ≠ -4) :
  |r₁ - r₂| = 3 / 2 := 
sorry


end NUMINAMATH_GPT_positive_difference_solutions_l324_32422


namespace NUMINAMATH_GPT_value_of_x_when_y_equals_8_l324_32403

noncomputable def inverse_variation(cube_root : ℝ → ℝ) (k : ℝ) (x y : ℝ) : Prop :=
  y * (cube_root x) = k

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem value_of_x_when_y_equals_8 : 
  ∃ k : ℝ, (inverse_variation cube_root k 8 2) → 
  (inverse_variation cube_root k (1 / 8) 8) := 
sorry

end NUMINAMATH_GPT_value_of_x_when_y_equals_8_l324_32403


namespace NUMINAMATH_GPT_triangle_side_m_l324_32401

theorem triangle_side_m (a b m : ℝ) (ha : a = 2) (hb : b = 3) (h1 : a + b > m) (h2 : a + m > b) (h3 : b + m > a) :
  (1 < m ∧ m < 5) → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_m_l324_32401


namespace NUMINAMATH_GPT_part1_part2_l324_32439

namespace RationalOp
  -- Define the otimes operation
  def otimes (a b : ℚ) : ℚ := a * b^2 + 2 * a * b + a

  -- Part 1: Prove (-2) ⊗ 4 = -50
  theorem part1 : otimes (-2) 4 = -50 := sorry

  -- Part 2: Given x ⊗ 3 = y ⊗ (-3), prove 8x - 2y + 5 = 5
  theorem part2 (x y : ℚ) (h : otimes x 3 = otimes y (-3)) : 8*x - 2*y + 5 = 5 := sorry
end RationalOp

end NUMINAMATH_GPT_part1_part2_l324_32439


namespace NUMINAMATH_GPT_volume_of_box_l324_32415

theorem volume_of_box (l w h : ℝ) (hlw : l * w = 36) (hwh : w * h = 18) (hlh : l * h = 8) : 
    l * w * h = 72 := 
by 
    sorry

end NUMINAMATH_GPT_volume_of_box_l324_32415


namespace NUMINAMATH_GPT_average_marks_l324_32411

/--
Given:
1. The average marks in physics (P) and mathematics (M) is 90.
2. The average marks in physics (P) and chemistry (C) is 70.
3. The student scored 110 marks in physics (P).

Prove that the average marks the student scored in the 3 subjects (P, C, M) is 70.
-/
theorem average_marks (P C M : ℝ) 
  (h1 : (P + M) / 2 = 90)
  (h2 : (P + C) / 2 = 70)
  (h3 : P = 110) : 
  (P + C + M) / 3 = 70 :=
sorry

end NUMINAMATH_GPT_average_marks_l324_32411


namespace NUMINAMATH_GPT_average_weight_of_all_players_l324_32488

-- Definitions based on conditions
def num_forwards : ℕ := 8
def avg_weight_forwards : ℝ := 75
def num_defensemen : ℕ := 12
def avg_weight_defensemen : ℝ := 82

-- Total number of players
def total_players : ℕ := num_forwards + num_defensemen

-- Values derived from conditions
def total_weight_forwards : ℝ := avg_weight_forwards * num_forwards
def total_weight_defensemen : ℝ := avg_weight_defensemen * num_defensemen
def total_weight : ℝ := total_weight_forwards + total_weight_defensemen

-- Theorem to prove the average weight of all players
theorem average_weight_of_all_players : total_weight / total_players = 79.2 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_of_all_players_l324_32488


namespace NUMINAMATH_GPT_find_number_l324_32417

theorem find_number (x : ℝ) (h : (2 * x - 37 + 25) / 8 = 5) : x = 26 :=
sorry

end NUMINAMATH_GPT_find_number_l324_32417


namespace NUMINAMATH_GPT_correct_calculation_l324_32418

theorem correct_calculation :
  (∀ (x y : ℝ), -3 * x - 3 * x ≠ 0) ∧
  (∀ (x : ℝ), x - 4 * x ≠ -3) ∧
  (∀ (x : ℝ), 2 * x + 3 * x^2 ≠ 5 * x^3) ∧
  (∀ (x y : ℝ), -4 * x * y + 3 * x * y = -x * y) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l324_32418


namespace NUMINAMATH_GPT_combined_average_score_clubs_l324_32446

theorem combined_average_score_clubs
  (nA nB : ℕ) -- Number of members in each club
  (avgA avgB : ℝ) -- Average score of each club
  (hA : nA = 40)
  (hB : nB = 50)
  (hAvgA : avgA = 90)
  (hAvgB : avgB = 81) :
  (nA * avgA + nB * avgB) / (nA + nB) = 85 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_combined_average_score_clubs_l324_32446


namespace NUMINAMATH_GPT_salad_quantity_percentage_difference_l324_32495

noncomputable def Tom_rate := 2/3 -- Tom's rate (lb/min)
noncomputable def Tammy_rate := 3/2 -- Tammy's rate (lb/min)
noncomputable def Total_salad := 65 -- Total salad chopped (lb)
noncomputable def Time_to_chop := Total_salad / (Tom_rate + Tammy_rate) -- Time to chop 65 lb (min)
noncomputable def Tom_chop := Time_to_chop * Tom_rate -- Total chopped by Tom (lb)
noncomputable def Tammy_chop := Time_to_chop * Tammy_rate -- Total chopped by Tammy (lb)
noncomputable def Percent_difference := (Tammy_chop - Tom_chop) / Tom_chop * 100 -- Percent difference

theorem salad_quantity_percentage_difference : Percent_difference = 125 :=
by
  sorry

end NUMINAMATH_GPT_salad_quantity_percentage_difference_l324_32495


namespace NUMINAMATH_GPT_project_scientists_total_l324_32442

def total_scientists (S : ℕ) : Prop :=
  S / 2 + S / 5 + 21 = S

theorem project_scientists_total : ∃ S, total_scientists S ∧ S = 70 :=
by
  existsi 70
  unfold total_scientists
  sorry

end NUMINAMATH_GPT_project_scientists_total_l324_32442


namespace NUMINAMATH_GPT_man_present_age_l324_32479

variable {P : ℝ}

theorem man_present_age (h1 : P = 1.25 * (P - 10)) (h2 : P = (5 / 6) * (P + 10)) : P = 50 :=
  sorry

end NUMINAMATH_GPT_man_present_age_l324_32479


namespace NUMINAMATH_GPT_find_alpha_plus_beta_l324_32475

theorem find_alpha_plus_beta (α β : ℝ)
  (h : ∀ x : ℝ, x ≠ 45 → (x - α) / (x + β) = (x^2 - 90 * x + 1981) / (x^2 + 63 * x - 3420)) :
  α + β = 113 :=
by
  sorry

end NUMINAMATH_GPT_find_alpha_plus_beta_l324_32475


namespace NUMINAMATH_GPT_friday_vs_tuesday_l324_32402

def tuesday_amount : ℝ := 8.5
def wednesday_amount : ℝ := 5.5 * tuesday_amount
def thursday_amount : ℝ := wednesday_amount + 0.10 * wednesday_amount
def friday_amount : ℝ := 0.75 * thursday_amount

theorem friday_vs_tuesday :
  friday_amount - tuesday_amount = 30.06875 :=
sorry

end NUMINAMATH_GPT_friday_vs_tuesday_l324_32402


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l324_32405

/-
Given a geometric sequence {a_n} with common ratio q ≠ -1 and q ≠ 1,
and S_n is the sum of the first n terms of the geometric sequence.
Given S_{12} = 7 S_{4}, prove:
S_{8}/S_{4} = 3
-/

theorem geometric_sequence_ratio {a_n : ℕ → ℝ} (q : ℝ) (h₁ : q ≠ -1) (h₂ : q ≠ 1)
  (S : ℕ → ℝ) (hSn : ∀ n, S n = a_n 0 * (1 - q ^ n) / (1 - q)) (h : S 12 = 7 * S 4) :
  S 8 / S 4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l324_32405


namespace NUMINAMATH_GPT_cube_sum_identity_l324_32435

theorem cube_sum_identity (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by 
 sorry

end NUMINAMATH_GPT_cube_sum_identity_l324_32435


namespace NUMINAMATH_GPT_sam_walked_distance_when_meeting_l324_32469

variable (D_s D_f : ℝ)
variable (t : ℝ)

theorem sam_walked_distance_when_meeting
  (h1 : 55 = D_f + D_s)
  (h2 : D_f = 6 * t)
  (h3 : D_s = 5 * t) :
  D_s = 25 :=
by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_sam_walked_distance_when_meeting_l324_32469


namespace NUMINAMATH_GPT_find_p_q_l324_32493

def vector_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def vector_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_p_q (p q : ℝ)
  (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
  (p, q) = (-29/12, 43/12) :=
by 
  sorry

end NUMINAMATH_GPT_find_p_q_l324_32493


namespace NUMINAMATH_GPT_find_x_l324_32413

theorem find_x (x : ℕ) (h : 220030 = (x + 445) * (2 * (x - 445)) + 30) : x = 555 := 
sorry

end NUMINAMATH_GPT_find_x_l324_32413


namespace NUMINAMATH_GPT_eval_expression_eq_2_l324_32483

theorem eval_expression_eq_2 :
  (10^2 + 11^2 + 12^2 + 13^2 + 14^2) / 365 = 2 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_eq_2_l324_32483


namespace NUMINAMATH_GPT_car_speed_l324_32473

theorem car_speed (v : ℝ) (h₁ : (1/75 * 3600) + 12 = 1/v * 3600) : v = 60 := 
by 
  sorry

end NUMINAMATH_GPT_car_speed_l324_32473


namespace NUMINAMATH_GPT_uncle_ben_parking_probability_l324_32407

theorem uncle_ben_parking_probability :
  let total_spaces := 20
  let cars := 15
  let rv_spaces := 3
  let total_combinations := Nat.choose total_spaces cars
  let non_adjacent_empty_combinations := Nat.choose (total_spaces - rv_spaces) cars
  (1 - (non_adjacent_empty_combinations / total_combinations)) = (232 / 323) := by
  sorry

end NUMINAMATH_GPT_uncle_ben_parking_probability_l324_32407


namespace NUMINAMATH_GPT_find_y_l324_32492

theorem find_y (x y : ℚ) (h1 : x = 153) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 350064) : 
  y = 40 / 3967 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_find_y_l324_32492


namespace NUMINAMATH_GPT_domain_of_fractional_sqrt_function_l324_32489

theorem domain_of_fractional_sqrt_function :
  ∀ x : ℝ, (x + 4 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ∈ (Set.Ici (-4) \ {1})) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_fractional_sqrt_function_l324_32489


namespace NUMINAMATH_GPT_lines_intersection_l324_32452

theorem lines_intersection (a b : ℝ) : 
  (2 : ℝ) = (1/3 : ℝ) * (1 : ℝ) + a →
  (1 : ℝ) = (1/3 : ℝ) * (2 : ℝ) + b →
  a + b = 2 := 
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_lines_intersection_l324_32452


namespace NUMINAMATH_GPT_linear_function_y1_greater_y2_l324_32408

theorem linear_function_y1_greater_y2 :
  ∀ (y_1 y_2 : ℝ), 
    (y_1 = -(-1) + 6) → (y_2 = -(2) + 6) → y_1 > y_2 :=
by
  intros y_1 y_2 h1 h2
  sorry

end NUMINAMATH_GPT_linear_function_y1_greater_y2_l324_32408


namespace NUMINAMATH_GPT_simplify_expression_l324_32421

theorem simplify_expression (x y : ℝ) (h : x - 2 * y = -2) : 9 - 2 * x + 4 * y = 13 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l324_32421


namespace NUMINAMATH_GPT_channel_depth_l324_32447

theorem channel_depth
  (top_width bottom_width area : ℝ)
  (h : ℝ)
  (trapezium_area_formula : area = (1 / 2) * (top_width + bottom_width) * h)
  (top_width_val : top_width = 14)
  (bottom_width_val : bottom_width = 8)
  (area_val : area = 770) :
  h = 70 := 
by
  sorry

end NUMINAMATH_GPT_channel_depth_l324_32447


namespace NUMINAMATH_GPT_accessories_cost_is_200_l324_32438

variable (c_cost a_cost : ℕ)
variable (ps_value ps_sold : ℕ)
variable (john_paid : ℕ)

-- Given Conditions
def computer_cost := 700
def accessories_cost := a_cost
def playstation_value := 400
def playstation_sold := ps_value - (ps_value * 20 / 100)
def john_paid_amount := 580

-- Theorem to be proved
theorem accessories_cost_is_200 :
  ps_value = 400 →
  ps_sold = playstation_sold →
  c_cost = 700 →
  john_paid = 580 →
  john_paid + ps_sold - c_cost = a_cost →
  a_cost = 200 :=
by
  intros
  sorry

end NUMINAMATH_GPT_accessories_cost_is_200_l324_32438


namespace NUMINAMATH_GPT_yellow_chips_are_one_l324_32429

-- Definitions based on conditions
def yellow_chip_points : ℕ := 2
def blue_chip_points : ℕ := 4
def green_chip_points : ℕ := 5

variables (Y B G : ℕ)

-- Given conditions
def point_product_condition : Prop := (yellow_chip_points^Y * blue_chip_points^B * green_chip_points^G = 16000)
def equal_blue_green : Prop := (B = G)

-- Theorem to prove the number of yellow chips
theorem yellow_chips_are_one (Y B G : ℕ) (hprod : point_product_condition Y B G) (heq : equal_blue_green B G) : Y = 1 :=
by {
    sorry -- Proof omitted
}

end NUMINAMATH_GPT_yellow_chips_are_one_l324_32429


namespace NUMINAMATH_GPT_original_decimal_number_l324_32420

theorem original_decimal_number (x : ℝ) (h : x / 100 = x - 1.485) : x = 1.5 := 
by
  sorry

end NUMINAMATH_GPT_original_decimal_number_l324_32420


namespace NUMINAMATH_GPT_regression_equation_correct_l324_32424

-- Defining the given data as constants
def x_data : List ℕ := [1, 2, 3, 4, 5, 6, 7]
def y_data : List ℕ := [891, 888, 351, 220, 200, 138, 112]

def sum_t_y : ℚ := 1586
def avg_t : ℚ := 0.37
def sum_t2_min7_avg_t2 : ℚ := 0.55

-- Defining the target regression equation
def target_regression (x : ℚ) : ℚ := 1000 / x + 30

-- Function to calculate the regression equation from data
noncomputable def calculate_regression (x_data y_data : List ℕ) : (ℚ → ℚ) :=
  let n : ℚ := x_data.length
  let avg_y : ℚ := y_data.sum / n
  let b : ℚ := (sum_t_y - n * avg_t * avg_y) / (sum_t2_min7_avg_t2)
  let a : ℚ := avg_y - b * avg_t
  fun x : ℚ => a + b / x

-- Theorem stating the regression equation matches the target regression equation
theorem regression_equation_correct :
  calculate_regression x_data y_data = target_regression :=
by
  sorry

end NUMINAMATH_GPT_regression_equation_correct_l324_32424


namespace NUMINAMATH_GPT_max_distinct_subsets_l324_32455

def T : Set ℕ := { x | 1 ≤ x ∧ x ≤ 999 }

theorem max_distinct_subsets (k : ℕ) (A : Fin k → Set ℕ) 
  (h : ∀ i j : Fin k, i < j → A i ∪ A j = T) : 
  k ≤ 1000 := 
sorry

end NUMINAMATH_GPT_max_distinct_subsets_l324_32455


namespace NUMINAMATH_GPT_range_of_a_for_local_min_max_l324_32487

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a_for_local_min_max (a e x1 x2 : ℝ) (h_a : 0 < a) (h_a_ne : a ≠ 1) (h_x1_x2 : x1 < x2) 
  (h_min : ∀ x, f a e x > f a e x1) (h_max : ∀ x, f a e x < f a e x2) : 
  (1 / Real.exp 1) < a ∧ a < 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_for_local_min_max_l324_32487


namespace NUMINAMATH_GPT_math_problem_equivalent_l324_32443

-- Given that the problem requires four distinct integers a, b, c, d which are less than 12 and invertible modulo 12.
def coprime_with_12 (x : ℕ) : Prop := Nat.gcd x 12 = 1

theorem math_problem_equivalent 
  (a b c d : ℕ) (ha : coprime_with_12 a) (hb : coprime_with_12 b) 
  (hc : coprime_with_12 c) (hd : coprime_with_12 d) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c)
  (hbd : b ≠ d) (hcd : c ≠ d) :
  ((a * b * c * d) + (a * b * c) + (a * b * d) + (a * c * d) + (b * c * d)) * Nat.gcd (a * b * c * d) 12 = 1 :=
sorry

end NUMINAMATH_GPT_math_problem_equivalent_l324_32443


namespace NUMINAMATH_GPT_determine_marriages_l324_32498

-- Definitions of the items each person bought
variable (a_items b_items c_items : ℕ) -- Number of items bought by wives a, b, and c
variable (A_items B_items C_items : ℕ) -- Number of items bought by husbands A, B, and C

-- Conditions
variable (spend_eq_square_a : a_items * a_items = a_spend) -- Spending equals square of items
variable (spend_eq_square_b : b_items * b_items = b_spend)
variable (spend_eq_square_c : c_items * c_items = c_spend)
variable (spend_eq_square_A : A_items * A_items = A_spend)
variable (spend_eq_square_B : B_items * B_items = B_spend)
variable (spend_eq_square_C : C_items * C_items = C_spend)

variable (A_spend_eq : A_spend = a_spend + 48) -- Husbands spent 48 yuan more than wives
variable (B_spend_eq : B_spend = b_spend + 48)
variable (C_spend_eq : C_spend = c_spend + 48)

variable (A_bought_9_more : A_items = b_items + 9) -- A bought 9 more items than b
variable (B_bought_7_more : B_items = a_items + 7) -- B bought 7 more items than a

-- Theorem statement
theorem determine_marriages (hA : A_items ≥ b_items + 9) (hB : B_items ≥ a_items + 7) :
  (A_spend = A_items * A_items) ∧ (B_spend = B_items * B_items) ∧ (C_spend = C_items * C_items) ∧
  (a_spend = a_items * a_items) ∧ (b_spend = b_items * b_items) ∧ (c_spend = c_items * c_items) →
  (A_spend = a_spend + 48) ∧ (B_spend = b_spend + 48) ∧ (C_spend = c_spend + 48) →
  (A_items = b_items + 9) ∧ (B_items = a_items + 7) →
  (A_items = 13 ∧ c_items = 11) ∧ (B_items = 8 ∧ b_items = 4) ∧ (C_items = 7 ∧ a_items = 1) :=
by
  sorry

end NUMINAMATH_GPT_determine_marriages_l324_32498


namespace NUMINAMATH_GPT_largest_value_b_l324_32432

theorem largest_value_b (b : ℚ) : (3 * b + 7) * (b - 2) = 9 * b -> b = (4 + Real.sqrt 58) / 3 :=
by
  sorry

end NUMINAMATH_GPT_largest_value_b_l324_32432


namespace NUMINAMATH_GPT_combined_money_l324_32464

/-- Tom has a quarter the money of Nataly. Nataly has three times the money of Raquel.
     Sam has twice the money of Nataly. Raquel has $40. Prove that combined they have $430. -/
theorem combined_money : 
  ∀ (T R N S : ℕ), 
    (T = N / 4) ∧ 
    (N = 3 * R) ∧ 
    (S = 2 * N) ∧ 
    (R = 40) → 
    T + R + N + S = 430 := 
by
  sorry

end NUMINAMATH_GPT_combined_money_l324_32464


namespace NUMINAMATH_GPT_ratio_quadrilateral_l324_32459

theorem ratio_quadrilateral
  (ABCD_area : ℝ)
  (h_ABCD : ABCD_area = 40)
  (K L M N : Type)
  (AK KB : ℝ)
  (h_ratio : AK / KB = BL / LC ∧ BL / LC = CM / MD ∧ CM / MD = DN / NA)
  (KLMN_area : ℝ)
  (h_KLMN : KLMN_area = 25) :
  (AK / (AK + KB) = 1 / 4 ∨ AK / (AK + KB) = 3 / 4) :=
sorry

end NUMINAMATH_GPT_ratio_quadrilateral_l324_32459


namespace NUMINAMATH_GPT_ratio_man_to_son_in_two_years_l324_32425

-- Define current ages and the conditions
def son_current_age : ℕ := 24
def man_current_age : ℕ := son_current_age + 26

-- Define ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- State the theorem
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_man_to_son_in_two_years_l324_32425


namespace NUMINAMATH_GPT_lateral_surface_area_of_cylinder_l324_32478

theorem lateral_surface_area_of_cylinder (V : ℝ) (hV : V = 27 * Real.pi) : 
  ∃ (S : ℝ), S = 18 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cylinder_l324_32478


namespace NUMINAMATH_GPT_range_of_m_l324_32454

theorem range_of_m {x m : ℝ} 
  (α : 2 / (x + 1) > 1) 
  (β : m ≤ x ∧ x ≤ 2) 
  (suff_condition : ∀ x, (2 / (x + 1) > 1) → (m ≤ x ∧ x ≤ 2)) :
  m ≤ -1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l324_32454


namespace NUMINAMATH_GPT_shooting_to_practice_ratio_l324_32460

variable (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ)
variable (runningWeightliftingRatio : ℕ)

axiom practiceTime_def : practiceTime = 2 * 60 -- converting 2 hours to minutes
axiom weightliftingTime_def : weightliftingTime = 20
axiom runningWeightliftingRatio_def : runningWeightliftingRatio = 2
axiom runningTime_def : runningTime = runningWeightliftingRatio * weightliftingTime
axiom shootingTime_def : shootingTime = practiceTime - (runningTime + weightliftingTime)

theorem shooting_to_practice_ratio (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ) 
                                   (runningWeightliftingRatio : ℕ) :
  practiceTime = 120 →
  weightliftingTime = 20 →
  runningWeightliftingRatio = 2 →
  runningTime = runningWeightliftingRatio * weightliftingTime →
  shootingTime = practiceTime - (runningTime + weightliftingTime) →
  (shootingTime : ℚ) / practiceTime = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_shooting_to_practice_ratio_l324_32460


namespace NUMINAMATH_GPT_batsman_average_increase_l324_32445

def average_increase (avg_before : ℕ) (runs_12th_inning : ℕ) (avg_after : ℕ) : ℕ :=
  avg_after - avg_before

theorem batsman_average_increase :
  ∀ (avg_before runs_12th_inning avg_after : ℕ),
    (runs_12th_inning = 70) →
    (avg_after = 37) →
    (11 * avg_before + runs_12th_inning = 12 * avg_after) →
    average_increase avg_before runs_12th_inning avg_after = 3 :=
by
  intros avg_before runs_12th_inning avg_after h_runs h_avg_after h_total
  sorry

end NUMINAMATH_GPT_batsman_average_increase_l324_32445


namespace NUMINAMATH_GPT_trajectory_is_line_segment_l324_32409

theorem trajectory_is_line_segment : 
  ∃ (P : ℝ × ℝ) (F1 F2: ℝ × ℝ), 
    F1 = (-3, 0) ∧ F2 = (3, 0) ∧ (|F1.1 - P.1|^2 + |F1.2 - P.2|^2).sqrt + (|F2.1 - P.1|^2 + |F2.2 - P.2|^2).sqrt = 6
  → (P.1 = F1.1 ∨ P.1 = F2.1) ∧ (P.2 = F1.2 ∨ P.2 = F2.2) :=
by sorry

end NUMINAMATH_GPT_trajectory_is_line_segment_l324_32409


namespace NUMINAMATH_GPT_falling_body_time_l324_32430

theorem falling_body_time (g : ℝ) (h_g : g = 9.808) (d : ℝ) (t1 : ℝ) (h_d : d = 49.34) (h_t1 : t1 = 1.3) : 
  ∃ t : ℝ, (1 / 2 * g * (t + t1)^2 - 1 / 2 * g * t^2 = d) → t = 7.088 :=
by 
  use 7.088
  intros h
  sorry

end NUMINAMATH_GPT_falling_body_time_l324_32430


namespace NUMINAMATH_GPT_time_to_cook_rest_of_potatoes_l324_32481

-- Definitions of the conditions
def total_potatoes : ℕ := 12
def already_cooked : ℕ := 6
def minutes_per_potato : ℕ := 6

-- Proof statement
theorem time_to_cook_rest_of_potatoes : (total_potatoes - already_cooked) * minutes_per_potato = 36 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cook_rest_of_potatoes_l324_32481


namespace NUMINAMATH_GPT_find_m_n_l324_32467

-- Define the vectors OA, OB, OC
def vector_oa (m : ℝ) : ℝ × ℝ := (-2, m)
def vector_ob (n : ℝ) : ℝ × ℝ := (n, 1)
def vector_oc : ℝ × ℝ := (5, -1)

-- Define the condition that OA is perpendicular to OB
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Define the condition that points A, B, and C are collinear.
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (A.1 - B.1) * (C.2 - A.2) = k * ((C.1 - A.1) * (A.2 - B.2))

theorem find_m_n (m n : ℝ) :
  collinear (-2, m) (n, 1) (5, -1) ∧ perpendicular (-2, m) (n, 1) → m = 3 ∧ n = 3 / 2 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_m_n_l324_32467


namespace NUMINAMATH_GPT_ellas_quadratic_equation_l324_32428

theorem ellas_quadratic_equation (d e : ℤ) :
  (∀ x : ℤ, |x - 8| = 3 → (x = 11 ∨ x = 5)) →
  (∀ x : ℤ, (x = 11 ∨ x = 5) → x^2 + d * x + e = 0) →
  (d, e) = (-16, 55) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_ellas_quadratic_equation_l324_32428


namespace NUMINAMATH_GPT_dogs_sold_correct_l324_32440

-- Definitions based on conditions
def ratio_cats_to_dogs (cats dogs : ℕ) := 2 * dogs = cats

-- Given conditions
def cats_sold := 16
def dogs_sold := 8

-- The theorem to prove
theorem dogs_sold_correct (h : ratio_cats_to_dogs cats_sold dogs_sold) : dogs_sold = 8 :=
by
  sorry

end NUMINAMATH_GPT_dogs_sold_correct_l324_32440


namespace NUMINAMATH_GPT_correct_calculation_l324_32471

theorem correct_calculation (a x y b : ℝ) :
  (-a - a = 0) = False ∧
  (- (x + y) = -x - y) = True ∧
  (3 * (b - 2 * a) = 3 * b - 2 * a) = False ∧
  (8 * a^4 - 6 * a^2 = 2 * a^2) = False :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l324_32471


namespace NUMINAMATH_GPT_total_new_students_l324_32496

-- Given conditions
def number_of_schools : ℝ := 25.0
def average_students_per_school : ℝ := 9.88

-- Problem statement
theorem total_new_students : number_of_schools * average_students_per_school = 247 :=
by sorry

end NUMINAMATH_GPT_total_new_students_l324_32496


namespace NUMINAMATH_GPT_max_radius_of_circle_touching_graph_l324_32433

theorem max_radius_of_circle_touching_graph :
  ∃ r : ℝ, (∀ (x : ℝ), (x^2 + (x^4 - r)^2 = r^2) → r ≤ (3 * (2:ℝ)^(1/3)) / 4) ∧
           r = (3 * (2:ℝ)^(1/3)) / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_radius_of_circle_touching_graph_l324_32433


namespace NUMINAMATH_GPT_find_new_bottle_caps_l324_32410

theorem find_new_bottle_caps (initial caps_thrown current : ℕ) (h_initial : initial = 69)
  (h_thrown : caps_thrown = 60) (h_current : current = 67) :
  ∃ n, initial - caps_thrown + n = current ∧ n = 58 := by
sorry

end NUMINAMATH_GPT_find_new_bottle_caps_l324_32410


namespace NUMINAMATH_GPT_arithmetic_seq_a12_l324_32456

variable {a : ℕ → ℝ}

theorem arithmetic_seq_a12 :
  (∀ n, ∃ d, a (n + 1) = a n + d)
  ∧ a 5 + a 11 = 30
  ∧ a 4 = 7
  → a 12 = 23 :=
by
  sorry


end NUMINAMATH_GPT_arithmetic_seq_a12_l324_32456


namespace NUMINAMATH_GPT_king_zenobius_more_descendants_l324_32485

-- Conditions
def descendants_paphnutius (p2_descendants p1_descendants: ℕ) := 
  2 + 60 * p2_descendants + 20 * p1_descendants = 142

def descendants_zenobius (z3_descendants z1_descendants : ℕ) := 
  4 + 35 * z3_descendants + 35 * z1_descendants = 144

-- Main statement
theorem king_zenobius_more_descendants:
  ∀ (p2_descendants p1_descendants z3_descendants z1_descendants : ℕ),
    descendants_paphnutius p2_descendants p1_descendants →
    descendants_zenobius z3_descendants z1_descendants →
    144 > 142 :=
by
  intros
  sorry

end NUMINAMATH_GPT_king_zenobius_more_descendants_l324_32485


namespace NUMINAMATH_GPT_min_total_cost_at_n_equals_1_l324_32419

-- Define the conditions and parameters
variables (a : ℕ) -- The total construction area
variables (n : ℕ) -- The number of floors

-- Definitions based on the given problem conditions
def land_expropriation_cost : ℕ := 2388 * a
def construction_cost (n : ℕ) : ℕ :=
  if n = 0 then 0 else if n = 1 then 455 * a else (455 * n * a + 30 * (n-2) * (n-1) / 2 * a)

-- Total cost including land expropriation and construction costs
def total_cost (n : ℕ) : ℕ := land_expropriation_cost a + construction_cost a n

-- The minimum total cost occurs at n = 1
theorem min_total_cost_at_n_equals_1 :
  ∃ n, n = 1 ∧ total_cost a n = 2788 * a :=
by sorry

end NUMINAMATH_GPT_min_total_cost_at_n_equals_1_l324_32419


namespace NUMINAMATH_GPT_sum_of_youngest_and_oldest_cousins_l324_32480

theorem sum_of_youngest_and_oldest_cousins 
  (a1 a2 a3 a4 : ℕ) 
  (h_order : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4) 
  (h_mean : a1 + a2 + a3 + a4 = 36) 
  (h_median : a2 + a3 = 14) : 
  a1 + a4 = 22 :=
by sorry

end NUMINAMATH_GPT_sum_of_youngest_and_oldest_cousins_l324_32480


namespace NUMINAMATH_GPT_measure_of_angle_F_l324_32400

theorem measure_of_angle_F {D E F : ℝ}
  (isosceles : D = E)
  (angle_F_condition : F = D + 40)
  (sum_of_angles : D + E + F = 180) :
  F = 260 / 3 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_F_l324_32400


namespace NUMINAMATH_GPT_free_throws_count_l324_32491

-- Definitions based on the conditions
variables (a b x : ℕ) -- Number of 2-point shots, 3-point shots, and free throws respectively.

-- Condition: Points from two-point shots equal the points from three-point shots
def points_eq : Prop := 2 * a = 3 * b

-- Condition: Number of free throws is twice the number of two-point shots
def free_throws_eq : Prop := x = 2 * a

-- Condition: Total score is adjusted to 78 points
def total_score : Prop := 2 * a + 3 * b + x = 78

-- Proof problem statement
theorem free_throws_count (h1 : points_eq a b) (h2 : free_throws_eq a x) (h3 : total_score a b x) : x = 26 :=
sorry

end NUMINAMATH_GPT_free_throws_count_l324_32491


namespace NUMINAMATH_GPT_sum_of_consecutive_odds_eq_power_l324_32404

theorem sum_of_consecutive_odds_eq_power (n : ℕ) (k : ℕ) (hn : n > 0) (hk : k ≥ 2) :
  ∃ a : ℤ, n * (2 * a + n) = n^k ∧
            (∀ i : ℕ, i < n → 2 * a + 2 * (i : ℤ) + 1 = 2 * a + 1 + 2 * i) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_odds_eq_power_l324_32404


namespace NUMINAMATH_GPT_subset_relation_l324_32453

def P := {x : ℝ | x < 2}
def Q := {y : ℝ | y < 1}

theorem subset_relation : Q ⊆ P := 
by {
  sorry
}

end NUMINAMATH_GPT_subset_relation_l324_32453


namespace NUMINAMATH_GPT_maxValue_a1_l324_32486

variable (a_1 q : ℝ)

def isGeometricSequence (a_1 q : ℝ) : Prop :=
  a_1 ≥ 1 ∧ a_1 * q ≤ 2 ∧ a_1 * q^2 ≥ 3

theorem maxValue_a1 (h : isGeometricSequence a_1 q) : a_1 ≤ 4 / 3 := 
sorry

end NUMINAMATH_GPT_maxValue_a1_l324_32486


namespace NUMINAMATH_GPT_evaporated_water_l324_32474

theorem evaporated_water 
  (E : ℝ)
  (h₁ : 0 < 10) -- initial mass is positive
  (h₂ : 10 * 0.3 + 10 * 0.7 = 3 + 7) -- Solution Y composition check
  (h₃ : (3 + 0.3 * E) / (10 - E + 0.7 * E) = 0.36) -- New solution composition
  : E = 0.9091 := 
sorry

end NUMINAMATH_GPT_evaporated_water_l324_32474


namespace NUMINAMATH_GPT_find_a_l324_32412

theorem find_a (a : ℝ) 
  (h1 : ∀ x y : ℝ, 2*x + y - 2 = 0)
  (h2 : ∀ x y : ℝ, a*x + 4*y + 1 = 0)
  (perpendicular : ∀ (m1 m2 : ℝ), m1 = -2 → m2 = -a/4 → m1 * m2 = -1) :
  a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_l324_32412


namespace NUMINAMATH_GPT_calculate_rectangle_length_l324_32463

theorem calculate_rectangle_length (side_of_square : ℝ) (width_of_rectangle : ℝ)
  (length_of_wire : ℝ) (perimeter_of_rectangle : ℝ) :
  side_of_square = 20 → 
  width_of_rectangle = 14 → 
  length_of_wire = 4 * side_of_square →
  perimeter_of_rectangle = length_of_wire →
  2 * (width_of_rectangle + length_of_rectangle) = perimeter_of_rectangle →
  length_of_rectangle = 26 :=
by
  intros
  sorry

end NUMINAMATH_GPT_calculate_rectangle_length_l324_32463


namespace NUMINAMATH_GPT_total_volume_l324_32490

-- Defining the volumes for different parts as per the conditions.
variables (V_A V_C V_B' V_C' : ℝ)
variables (V : ℝ)

-- The given conditions
axiom V_A_eq_40 : V_A = 40
axiom V_C_eq_300 : V_C = 300
axiom V_B'_eq_360 : V_B' = 360
axiom V_C'_eq_90 : V_C' = 90

-- The proof goal: total volume of the parallelepiped
theorem total_volume (V_A V_C V_B' V_C' : ℝ) 
  (V_A_eq_40 : V_A = 40) (V_C_eq_300 : V_C = 300) 
  (V_B'_eq_360 : V_B' = 360) (V_C'_eq_90 : V_C' = 90) :
  V = V_A + V_C + V_B' + V_C' :=
by
  sorry

end NUMINAMATH_GPT_total_volume_l324_32490


namespace NUMINAMATH_GPT_cost_prices_of_products_l324_32437

-- Define the variables and conditions from the problem
variables (x y : ℝ)

-- Theorem statement
theorem cost_prices_of_products (h1 : 20 * x + 15 * y = 380) (h2 : 15 * x + 10 * y = 280) : 
  x = 16 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cost_prices_of_products_l324_32437


namespace NUMINAMATH_GPT_find_teddy_dogs_l324_32499

-- Definitions from the conditions
def teddy_cats := 8
def ben_dogs (teddy_dogs : ℕ) := teddy_dogs + 9
def dave_cats (teddy_cats : ℕ) := teddy_cats + 13
def dave_dogs (teddy_dogs : ℕ) := teddy_dogs - 5
def total_pets (teddy_dogs teddy_cats : ℕ) := teddy_dogs + teddy_cats + (ben_dogs teddy_dogs) + (dave_dogs teddy_dogs) + (dave_cats teddy_cats)

-- Theorem statement
theorem find_teddy_dogs (teddy_dogs : ℕ) (teddy_cats : ℕ) (hd : total_pets teddy_dogs teddy_cats = 54) :
  teddy_dogs = 7 := sorry

end NUMINAMATH_GPT_find_teddy_dogs_l324_32499


namespace NUMINAMATH_GPT_smallest_int_k_for_64_pow_k_l324_32414

theorem smallest_int_k_for_64_pow_k (k : ℕ) (base : ℕ) (h₁ : k = 7) : 
  64^k > base^20 → base = 4 := by
  sorry

end NUMINAMATH_GPT_smallest_int_k_for_64_pow_k_l324_32414


namespace NUMINAMATH_GPT_num_solutions_triples_l324_32494

theorem num_solutions_triples :
  {n : ℕ // ∃ a b c : ℤ, a^2 - a * (b + c) + b^2 - b * c + c^2 = 1 ∧ n = 10  } :=
  sorry

end NUMINAMATH_GPT_num_solutions_triples_l324_32494


namespace NUMINAMATH_GPT_students_with_certificates_l324_32457

variable (C N : ℕ)

theorem students_with_certificates :
  (C + N = 120) ∧ (C = N + 36) → C = 78 :=
by
  sorry

end NUMINAMATH_GPT_students_with_certificates_l324_32457


namespace NUMINAMATH_GPT_equivalent_proof_problem_l324_32427

-- Define the real numbers x, y, z and the operation ⊗
variables {x y z : ℝ}

def otimes (a b : ℝ) : ℝ := (a - b) ^ 2

theorem equivalent_proof_problem : otimes ((x + z) ^ 2) ((z + y) ^ 2) = (x ^ 2 + 2 * x * z - y ^ 2 - 2 * z * y) ^ 2 :=
by sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l324_32427


namespace NUMINAMATH_GPT_remainder_of_x_l324_32441

theorem remainder_of_x (x : ℕ) 
(H1 : 4 + x ≡ 81 [MOD 16])
(H2 : 6 + x ≡ 16 [MOD 36])
(H3 : 8 + x ≡ 36 [MOD 64]) :
  x ≡ 37 [MOD 48] :=
sorry

end NUMINAMATH_GPT_remainder_of_x_l324_32441
