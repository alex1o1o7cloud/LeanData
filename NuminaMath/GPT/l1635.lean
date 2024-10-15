import Mathlib

namespace NUMINAMATH_GPT_worker_assignment_l1635_163590

theorem worker_assignment (x : ℕ) (y : ℕ) 
  (h1 : x + y = 90)
  (h2 : 2 * 15 * x = 3 * 8 * y) : 
  (x = 40 ∧ y = 50) := by
  sorry

end NUMINAMATH_GPT_worker_assignment_l1635_163590


namespace NUMINAMATH_GPT_triangle_AD_eq_8sqrt2_l1635_163564

/-- Given a triangle ABC where AB = 13, AC = 20, and
    D is the foot of the perpendicular from A to BC,
    with the ratio BD : CD = 3 : 4, prove that AD = 8√2. -/
theorem triangle_AD_eq_8sqrt2 
  (AB AC : ℝ) (BD CD AD : ℝ) 
  (h₁ : AB = 13)
  (h₂ : AC = 20)
  (h₃ : BD / CD = 3 / 4)
  (h₄ : BD^2 = AB^2 - AD^2)
  (h₅ : CD^2 = AC^2 - AD^2) :
  AD = 8 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_AD_eq_8sqrt2_l1635_163564


namespace NUMINAMATH_GPT_solve_eq_f_x_plus_3_l1635_163589

-- Define the function f with its piecewise definition based on the conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 - 3 * x
  else -(x^2 - 3 * (-x))

-- Define the main theorem to find the solution set
theorem solve_eq_f_x_plus_3 (x : ℝ) :
  f x = x + 3 ↔ x = 2 + Real.sqrt 7 ∨ x = -1 ∨ x = -3 :=
by sorry

end NUMINAMATH_GPT_solve_eq_f_x_plus_3_l1635_163589


namespace NUMINAMATH_GPT_brick_length_l1635_163532

theorem brick_length 
  (width : ℝ) (height : ℝ) (num_bricks : ℕ)
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_vol : ℝ) :
  width = 10 →
  height = 7.5 →
  num_bricks = 27000 →
  wall_length = 27 →
  wall_width = 2 →
  wall_height = 0.75 →
  brick_vol = width * height * (20:ℝ) →
  wall_length * wall_width * wall_height * 1000000 = num_bricks * brick_vol :=
by
  intros
  sorry

end NUMINAMATH_GPT_brick_length_l1635_163532


namespace NUMINAMATH_GPT_debby_deletion_l1635_163527

theorem debby_deletion :
  ∀ (zoo_pics museum_pics remaining_pics deleted_pics : ℕ),
    zoo_pics = 24 →
    museum_pics = 12 →
    remaining_pics = 22 →
    deleted_pics = zoo_pics + museum_pics - remaining_pics →
    deleted_pics = 14 :=
by
  intros zoo_pics museum_pics remaining_pics deleted_pics h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_debby_deletion_l1635_163527


namespace NUMINAMATH_GPT_percentage_difference_wages_l1635_163558

variables (W1 W2 : ℝ)
variables (h1 : W1 > 0) (h2 : W2 > 0)
variables (h3 : 0.40 * W2 = 1.60 * 0.20 * W1)

theorem percentage_difference_wages (W1 W2 : ℝ) (h1 : W1 > 0) (h2 : W2 > 0) (h3 : 0.40 * W2 = 1.60 * 0.20 * W1) :
  (W1 - W2) / W1 = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_wages_l1635_163558


namespace NUMINAMATH_GPT_increasing_function_solve_inequality_find_range_l1635_163536

noncomputable def f : ℝ → ℝ := sorry
def a1 := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f x
def a2 := f 1 = 1
def a3 := ∀ m n : ℝ, -1 ≤ m ∧ m ≤ 1 ∧ -1 ≤ n ∧ n ≤ 1 ∧ m + n ≠ 0 → (f m + f n) / (m + n) > 0

-- Statement for question (1)
theorem increasing_function : 
  (∀ x y : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x < f y) :=
by 
  apply sorry

-- Statement for question (2)
theorem solve_inequality (x : ℝ) :
  (f (x^2 - 1) + f (3 - 3*x) < 0 ↔ 1 < x ∧ x ≤ 4/3) :=
by 
  apply sorry

-- Statement for question (3)
theorem find_range (t : ℝ) :
  (∀ x a : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ a ∧ a ≤ 1 → f x ≤ t^2 - 2*a*t + 1) 
  ↔ (2 ≤ t ∨ t ≤ -2 ∨ t = 0) :=
by 
  apply sorry

end NUMINAMATH_GPT_increasing_function_solve_inequality_find_range_l1635_163536


namespace NUMINAMATH_GPT_original_price_of_table_l1635_163515

noncomputable def original_price (sale_price : ℝ) (discount_rate : ℝ) : ℝ :=
  sale_price / (1 - discount_rate)

theorem original_price_of_table
  (d : ℝ) (p' : ℝ) (h_d : d = 0.10) (h_p' : p' = 450) :
  original_price p' d = 500 := by
  rw [h_d, h_p']
  -- Calculating the original price
  show original_price 450 0.10 = 500
  sorry

end NUMINAMATH_GPT_original_price_of_table_l1635_163515


namespace NUMINAMATH_GPT_find_other_number_l1635_163507

theorem find_other_number (m n : ℕ) (H1 : n = 26) 
  (H2 : Nat.lcm n m = 52) (H3 : Nat.gcd n m = 8) : m = 16 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l1635_163507


namespace NUMINAMATH_GPT_percentage_of_students_passed_l1635_163546

def total_students : ℕ := 740
def failed_students : ℕ := 481
def passed_students : ℕ := total_students - failed_students
def pass_percentage : ℚ := (passed_students / total_students) * 100

theorem percentage_of_students_passed : pass_percentage = 35 := by
  sorry

end NUMINAMATH_GPT_percentage_of_students_passed_l1635_163546


namespace NUMINAMATH_GPT_range_of_a_l1635_163518

noncomputable def proposition_p (x : ℝ) : Prop := (4 * x - 3)^2 ≤ 1
noncomputable def proposition_q (x : ℝ) (a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ (∃ x, ¬ proposition_p x) → ¬ (∃ x, ¬ proposition_q x a)) →
  (¬ (¬ (∃ x, ¬ proposition_p x) ∧ ¬ (¬ (∃ x, ¬ proposition_q x a)))) →
  (0 ≤ a ∧ a ≤ 1 / 2) :=
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_range_of_a_l1635_163518


namespace NUMINAMATH_GPT_area_of_region_l1635_163511

-- The problem definition
def condition_1 (z : ℂ) : Prop := 
  0 < z.re / 20 ∧ z.re / 20 < 1 ∧
  0 < z.im / 20 ∧ z.im / 20 < 1 ∧
  0 < (20 / z).re ∧ (20 / z).re < 1 ∧
  0 < (20 / z).im ∧ (20 / z).im < 1

-- The proof statement
theorem area_of_region {z : ℂ} (h : condition_1 z) : 
  ∃ s : ℝ, s = 300 - 50 * Real.pi := sorry

end NUMINAMATH_GPT_area_of_region_l1635_163511


namespace NUMINAMATH_GPT_gcd_relatively_prime_l1635_163593

theorem gcd_relatively_prime (a : ℤ) (m n : ℕ) (h_odd : a % 2 = 1) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_diff : n ≠ m) :
  Int.gcd (a ^ 2^m + 2 ^ 2^m) (a ^ 2^n + 2 ^ 2^n) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_relatively_prime_l1635_163593


namespace NUMINAMATH_GPT_number_of_ways_to_tile_dominos_l1635_163563

-- Define the dimensions of the shapes and the criteria for the tiling problem
def L_shaped_area := 24
def size_of_square := 4
def size_of_rectangles := 2 * 10
def number_of_ways_to_tile := 208

-- Theorem statement
theorem number_of_ways_to_tile_dominos :
  (L_shaped_area = size_of_square + size_of_rectangles) →
  number_of_ways_to_tile = 208 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_number_of_ways_to_tile_dominos_l1635_163563


namespace NUMINAMATH_GPT_midpoint_sum_of_coordinates_l1635_163509

theorem midpoint_sum_of_coordinates : 
  let p1 := (8, 10)
  let p2 := (-4, -10)
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_sum_of_coordinates_l1635_163509


namespace NUMINAMATH_GPT_seats_taken_correct_l1635_163594

-- Define the conditions
def rows := 40
def chairs_per_row := 20
def unoccupied_seats := 10

-- Define the total number of seats
def total_seats := rows * chairs_per_row

-- Define the number of seats taken
def seats_taken := total_seats - unoccupied_seats

-- Statement of our math proof problem
theorem seats_taken_correct : seats_taken = 790 := by
  sorry

end NUMINAMATH_GPT_seats_taken_correct_l1635_163594


namespace NUMINAMATH_GPT_smallest_number_divisible_1_through_12_and_15_l1635_163528

theorem smallest_number_divisible_1_through_12_and_15 :
  ∃ n, (∀ i, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ∧ 15 ∣ n ∧ n = 27720 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_divisible_1_through_12_and_15_l1635_163528


namespace NUMINAMATH_GPT_skew_lines_angle_range_l1635_163549

theorem skew_lines_angle_range (θ : ℝ) (h_skew : θ > 0 ∧ θ ≤ 90) :
  0 < θ ∧ θ ≤ 90 :=
sorry

end NUMINAMATH_GPT_skew_lines_angle_range_l1635_163549


namespace NUMINAMATH_GPT_Maria_height_in_meters_l1635_163598

theorem Maria_height_in_meters :
  let inch_to_cm := 2.54
  let cm_to_m := 0.01
  let height_in_inch := 54
  let height_in_cm := height_in_inch * inch_to_cm
  let height_in_m := height_in_cm * cm_to_m
  let rounded_height_in_m := Float.round (height_in_m * 1000) / 1000
  rounded_height_in_m = 1.372 := 
by
  sorry

end NUMINAMATH_GPT_Maria_height_in_meters_l1635_163598


namespace NUMINAMATH_GPT_max_floor_l1635_163508

theorem max_floor (x : ℝ) (h : ⌊(x + 4) / 10⌋ = 5) : ⌊(6 * x) / 5⌋ = 67 :=
  sorry

end NUMINAMATH_GPT_max_floor_l1635_163508


namespace NUMINAMATH_GPT_shopkeeper_marked_price_l1635_163585

theorem shopkeeper_marked_price 
  (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : C = 0.75 * S)
  (h3 : S = 0.85 * M) :
  M = 1.17647 * L :=
sorry

end NUMINAMATH_GPT_shopkeeper_marked_price_l1635_163585


namespace NUMINAMATH_GPT_sample_average_l1635_163525

theorem sample_average (x : ℝ) 
  (h1 : (1 + 3 + 2 + 5 + x) / 5 = 3) : x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_sample_average_l1635_163525


namespace NUMINAMATH_GPT_expectation_of_binomial_l1635_163588

noncomputable def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p

theorem expectation_of_binomial :
  binomial_expectation 6 (1/3) = 2 :=
by
  sorry

end NUMINAMATH_GPT_expectation_of_binomial_l1635_163588


namespace NUMINAMATH_GPT_amateur_definition_l1635_163539
-- Import necessary libraries

-- Define the meaning of "amateur" and state that it is "amateurish" or "non-professional"
def meaning_of_amateur : String :=
  "amateurish or non-professional"

-- The main statement asserting that the meaning of "amateur" is indeed "amateurish" or "non-professional"
theorem amateur_definition : meaning_of_amateur = "amateurish or non-professional" :=
by
  -- The proof is trivial and assumed to be correct
  sorry

end NUMINAMATH_GPT_amateur_definition_l1635_163539


namespace NUMINAMATH_GPT_rectangle_enclosing_ways_l1635_163576

/-- Given five horizontal lines and five vertical lines, the total number of ways to choose four lines (two horizontal, two vertical) such that they form a rectangle is 100 --/
theorem rectangle_enclosing_ways : 
  let horizontal_lines := [1, 2, 3, 4, 5]
  let vertical_lines := [1, 2, 3, 4, 5]
  let ways_horizontal := Nat.choose 5 2
  let ways_vertical := Nat.choose 5 2
  ways_horizontal * ways_vertical = 100 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_enclosing_ways_l1635_163576


namespace NUMINAMATH_GPT_handshakes_at_convention_l1635_163516

theorem handshakes_at_convention (num_gremlins : ℕ) (num_imps : ℕ) 
  (H_gremlins_shake : num_gremlins = 25) (H_imps_shake_gremlins : num_imps = 20) : 
  let handshakes_among_gremlins := num_gremlins * (num_gremlins - 1) / 2
  let handshakes_between_imps_and_gremlins := num_imps * num_gremlins
  let total_handshakes := handshakes_among_gremlins + handshakes_between_imps_and_gremlins
  total_handshakes = 800 := 
by 
  sorry

end NUMINAMATH_GPT_handshakes_at_convention_l1635_163516


namespace NUMINAMATH_GPT_acute_triangle_condition_l1635_163555

theorem acute_triangle_condition (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 0) (h3 : B > 0) (h4 : C > 0)
    (h5 : A + B > 90) (h6 : B + C > 90) (h7 : C + A > 90) : A < 90 ∧ B < 90 ∧ C < 90 :=
sorry

end NUMINAMATH_GPT_acute_triangle_condition_l1635_163555


namespace NUMINAMATH_GPT_second_solution_percentage_l1635_163559

theorem second_solution_percentage (P : ℝ) : 
  (28 * 0.30 + 12 * P = 40 * 0.45) → P = 0.8 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_second_solution_percentage_l1635_163559


namespace NUMINAMATH_GPT_intersection_A_B_l1635_163505

-- Definitions of the sets A and B
def set_A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 7 }

-- Theorem statement to prove the intersection
theorem intersection_A_B : set_A ∩ set_B = { x | 3 ≤ x ∧ x < 7 } := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1635_163505


namespace NUMINAMATH_GPT_parabola_focus_distance_x_l1635_163574

theorem parabola_focus_distance_x (x y : ℝ) :
  y^2 = 4 * x ∧ y^2 = 4 * (x^2 + 5^2) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_x_l1635_163574


namespace NUMINAMATH_GPT_sufficiency_condition_l1635_163500

-- Definitions of p and q
def p (a b : ℝ) : Prop := a > |b|
def q (a b : ℝ) : Prop := a^2 > b^2

-- Main theorem statement
theorem sufficiency_condition (a b : ℝ) : (p a b → q a b) ∧ (¬(q a b → p a b)) := 
by
  sorry

end NUMINAMATH_GPT_sufficiency_condition_l1635_163500


namespace NUMINAMATH_GPT_pyramid_certain_height_l1635_163560

noncomputable def certain_height (h : ℝ) : Prop :=
  let height := h + 20
  let width := height + 234
  (height + width = 1274) → h = 1000 / 3

theorem pyramid_certain_height (h : ℝ) : certain_height h :=
by
  let height := h + 20
  let width := height + 234
  have h_eq : (height + width = 1274) → h = 1000 / 3 := sorry
  exact h_eq

end NUMINAMATH_GPT_pyramid_certain_height_l1635_163560


namespace NUMINAMATH_GPT_width_of_boxes_l1635_163529

theorem width_of_boxes
  (total_volume : ℝ)
  (total_payment : ℝ)
  (cost_per_box : ℝ)
  (h1 : total_volume = 1.08 * 10^6)
  (h2 : total_payment = 120)
  (h3 : cost_per_box = 0.2) :
  (∃ w : ℝ, w = (total_volume / (total_payment / cost_per_box))^(1/3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_width_of_boxes_l1635_163529


namespace NUMINAMATH_GPT_rectangle_width_l1635_163544

theorem rectangle_width (L W : ℝ) (h1 : 2 * (L + W) = 16) (h2 : W = L + 2) : W = 5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l1635_163544


namespace NUMINAMATH_GPT_TriangleInscribedAngle_l1635_163561

theorem TriangleInscribedAngle
  (x : ℝ)
  (arc_PQ : ℝ := x + 100)
  (arc_QR : ℝ := 2 * x + 50)
  (arc_RP : ℝ := 3 * x - 40)
  (angle_sum_eq_360 : arc_PQ + arc_QR + arc_RP = 360) :
  ∃ angle_PQR : ℝ, angle_PQR = 70.84 := 
sorry

end NUMINAMATH_GPT_TriangleInscribedAngle_l1635_163561


namespace NUMINAMATH_GPT_find_k_l1635_163562

theorem find_k (angle_BAC : ℝ) (angle_D : ℝ)
  (h1 : 0 < angle_BAC ∧ angle_BAC < π)
  (h2 : 0 < angle_D ∧ angle_D < π)
  (h3 : (π - angle_BAC) / 2 = 3 * angle_D) :
  angle_BAC = (5 / 11) * π :=
by sorry

end NUMINAMATH_GPT_find_k_l1635_163562


namespace NUMINAMATH_GPT_cost_of_article_l1635_163592

theorem cost_of_article 
    (C G : ℝ) 
    (h1 : 340 = C + G) 
    (h2 : 350 = C + G + 0.05 * G) 
    : C = 140 :=
by
    -- We do not need to provide the proof; 'sorry' is sufficient.
    sorry

end NUMINAMATH_GPT_cost_of_article_l1635_163592


namespace NUMINAMATH_GPT_mask_production_l1635_163548

theorem mask_production (M : ℕ) (h : 16 * M = 48000) : M = 3000 :=
by
  sorry

end NUMINAMATH_GPT_mask_production_l1635_163548


namespace NUMINAMATH_GPT_production_today_l1635_163568

-- Conditions
def average_daily_production_past_n_days (P : ℕ) (n : ℕ) := P = n * 50
def new_average_daily_production (P : ℕ) (T : ℕ) (new_n : ℕ) := (P + T) / new_n = 55

-- Values from conditions
def n := 11
def P := 11 * 50

-- Mathematically equivalent proof problem
theorem production_today :
  ∃ (T : ℕ), average_daily_production_past_n_days P n ∧ new_average_daily_production P T 12 → T = 110 :=
by
  sorry

end NUMINAMATH_GPT_production_today_l1635_163568


namespace NUMINAMATH_GPT_spaceship_speed_conversion_l1635_163553

theorem spaceship_speed_conversion (speed_km_per_sec : ℕ) (seconds_in_hour : ℕ) (correct_speed_km_per_hour : ℕ) :
  speed_km_per_sec = 12 →
  seconds_in_hour = 3600 →
  correct_speed_km_per_hour = 43200 →
  speed_km_per_sec * seconds_in_hour = correct_speed_km_per_hour := by
  sorry

end NUMINAMATH_GPT_spaceship_speed_conversion_l1635_163553


namespace NUMINAMATH_GPT_age_problem_l1635_163571

variables (a b c : ℕ)

theorem age_problem (h₁ : a = b + 2) (h₂ : b = 2 * c) (h₃ : a + b + c = 27) : b = 10 :=
by {
  -- Interactive proof steps can go here.
  sorry
}

end NUMINAMATH_GPT_age_problem_l1635_163571


namespace NUMINAMATH_GPT_converse_x_gt_y_then_x_gt_abs_y_is_true_l1635_163523

theorem converse_x_gt_y_then_x_gt_abs_y_is_true :
  (∀ x y : ℝ, (x > y) → (x > |y|)) → (∀ x y : ℝ, (x > |y|) → (x > y)) :=
by
  sorry

end NUMINAMATH_GPT_converse_x_gt_y_then_x_gt_abs_y_is_true_l1635_163523


namespace NUMINAMATH_GPT_average_of_rest_l1635_163510

theorem average_of_rest 
  (total_students : ℕ)
  (marks_5_students : ℕ)
  (marks_3_students : ℕ)
  (marks_others : ℕ)
  (average_class : ℚ)
  (remaining_students : ℕ)
  (expected_average : ℚ) 
  (h1 : total_students = 27) 
  (h2 : marks_5_students = 5 * 95) 
  (h3 : marks_3_students = 3 * 0) 
  (h4 : average_class = 49.25925925925926) 
  (h5 : remaining_students = 27 - 5 - 3) 
  (h6 : (marks_5_students + marks_3_students + marks_others) = total_students * average_class)
  : marks_others / remaining_students = expected_average :=
sorry

end NUMINAMATH_GPT_average_of_rest_l1635_163510


namespace NUMINAMATH_GPT_same_solution_sets_l1635_163573

theorem same_solution_sets (a : ℝ) :
  (∀ x : ℝ, 3 * x - 5 < a ↔ 2 * x < 4) → a = 1 := 
by
  sorry

end NUMINAMATH_GPT_same_solution_sets_l1635_163573


namespace NUMINAMATH_GPT_pancake_fundraiser_l1635_163535

-- Define the constants and conditions
def cost_per_stack_of_pancakes : ℕ := 4
def cost_per_slice_of_bacon : ℕ := 2
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90
def total_raised : ℕ := 420

-- Define a theorem that states what we want to prove
theorem pancake_fundraiser : 
  (stacks_sold * cost_per_stack_of_pancakes + slices_sold * cost_per_slice_of_bacon) = total_raised :=
by
  sorry -- We place a sorry here to skip the proof, as instructed.

end NUMINAMATH_GPT_pancake_fundraiser_l1635_163535


namespace NUMINAMATH_GPT_p_satisfies_conditions_l1635_163596

noncomputable def p (x : ℕ) : ℕ := sorry

theorem p_satisfies_conditions (h_monic : p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5) : 
  p 6 = 126 := sorry

end NUMINAMATH_GPT_p_satisfies_conditions_l1635_163596


namespace NUMINAMATH_GPT_trig_identity_l1635_163581

theorem trig_identity (α : ℝ) (h : Real.tan α = 1/3) :
  Real.cos α ^ 2 + Real.cos (Real.pi / 2 + 2 * α) = 3 / 10 := 
sorry

end NUMINAMATH_GPT_trig_identity_l1635_163581


namespace NUMINAMATH_GPT_domain_of_sqrt_ln_l1635_163519

def domain_function (x : ℝ) : Prop := x - 1 ≥ 0 ∧ 2 - x > 0

theorem domain_of_sqrt_ln (x : ℝ) : domain_function x ↔ 1 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_ln_l1635_163519


namespace NUMINAMATH_GPT_johns_total_cost_l1635_163566

variable (C_s C_d : ℝ)

theorem johns_total_cost (h_s : C_s = 20) (h_d : C_d = 0.5 * C_s) : C_s + C_d = 30 := by
  sorry

end NUMINAMATH_GPT_johns_total_cost_l1635_163566


namespace NUMINAMATH_GPT_geom_seq_general_term_arith_seq_sum_l1635_163584

theorem geom_seq_general_term (q : ℕ → ℕ) (a_1 a_2 a_3 : ℕ) (h1 : a_1 = 2)
  (h2 : (a_1 + a_3) / 2 = a_2 + 1) (h3 : a_2 = q 2) (h4 : a_3 = q 3)
  (g : ℕ → ℕ) (Sn : ℕ → ℕ) (gen_term : ∀ n, q n = 2^n) (sum_term : ∀ n, Sn n = 2^(n+1) - 2) :
  q n = g n :=
sorry

theorem arith_seq_sum (a_1 a_2 a_4 : ℕ) (b : ℕ → ℕ) (Tn : ℕ → ℕ) (h1 : a_1 = 2)
  (h2 : a_2 = 4) (h3 : a_4 = 16) (h4 : b 2 = a_1) (h5 : b 8 = a_2 + a_4)
  (gen_term : ∀ n, b n = 1 + 3 * (n - 1)) (sum_term : ∀ n, Tn n = (3 * n^2 - n) / 2) :
  Tn n = (3 * n^2 - 1) / 2 :=
sorry

end NUMINAMATH_GPT_geom_seq_general_term_arith_seq_sum_l1635_163584


namespace NUMINAMATH_GPT_total_profit_correct_l1635_163569

noncomputable def total_profit (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : ℝ := Tp

theorem total_profit_correct (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : 
  total_profit Cp Cq Cr Tp h1 h2 hR = 4650 :=
sorry

end NUMINAMATH_GPT_total_profit_correct_l1635_163569


namespace NUMINAMATH_GPT_residue_of_neg_1235_mod_29_l1635_163595

theorem residue_of_neg_1235_mod_29 : 
  ∃ r, 0 ≤ r ∧ r < 29 ∧ (-1235) % 29 = r ∧ r = 12 :=
by
  sorry

end NUMINAMATH_GPT_residue_of_neg_1235_mod_29_l1635_163595


namespace NUMINAMATH_GPT_farm_transaction_difference_l1635_163572

theorem farm_transaction_difference
  (x : ℕ)
  (h_initial : 6 * x - 15 > 0) -- Ensure initial horses are enough to sell 15
  (h_ratio_initial : 6 * x = x * 6)
  (h_ratio_final : (6 * x - 15) = 3 * (x + 15)) :
  (6 * x - 15) - (x + 15) = 70 :=
by
  sorry

end NUMINAMATH_GPT_farm_transaction_difference_l1635_163572


namespace NUMINAMATH_GPT_remainder_range_l1635_163543

theorem remainder_range (x y z a b c d e : ℕ)
(h1 : x % 211 = a) (h2 : y % 211 = b) (h3 : z % 211 = c)
(h4 : x % 251 = c) (h5 : y % 251 = d) (h6 : z % 251 = e)
(h7 : a < 211) (h8 : b < 211) (h9 : c < 211)
(h10 : c < 251) (h11 : d < 251) (h12 : e < 251) :
0 ≤ (2 * x - y + 3 * z + 47) % (211 * 251) ∧
(2 * x - y + 3 * z + 47) % (211 * 251) < (211 * 251) :=
by
  sorry

end NUMINAMATH_GPT_remainder_range_l1635_163543


namespace NUMINAMATH_GPT_apples_in_box_at_first_l1635_163541

noncomputable def initial_apples (X : ℕ) : Prop :=
  (X / 2 - 25 = 6)

theorem apples_in_box_at_first (X : ℕ) : initial_apples X ↔ X = 62 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_box_at_first_l1635_163541


namespace NUMINAMATH_GPT_smallest_term_l1635_163526

theorem smallest_term (a1 d : ℕ) (h_a1 : a1 = 7) (h_d : d = 7) :
  ∃ n : ℕ, (a1 + (n - 1) * d) > 150 ∧ (a1 + (n - 1) * d) % 5 = 0 ∧
  (∀ m : ℕ, (a1 + (m - 1) * d) > 150 ∧ (a1 + (m - 1) * d) % 5 = 0 → (a1 + (m - 1) * d) ≥ (a1 + (n - 1) * d)) → a1 + (n - 1) * d = 175 :=
by
  -- We need to prove given the conditions.
  sorry

end NUMINAMATH_GPT_smallest_term_l1635_163526


namespace NUMINAMATH_GPT_non_prime_in_sequence_l1635_163550

theorem non_prime_in_sequence : ∃ n : ℕ, ¬Prime (41 + n * (n - 1)) :=
by {
  use 41,
  sorry
}

end NUMINAMATH_GPT_non_prime_in_sequence_l1635_163550


namespace NUMINAMATH_GPT_det_matrixE_l1635_163556

def matrixE : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

theorem det_matrixE : (matrixE.det) = 25 := by
  sorry

end NUMINAMATH_GPT_det_matrixE_l1635_163556


namespace NUMINAMATH_GPT_find_q_l1635_163587

noncomputable def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h₁ : -p / 3 = q) (h₂ : q = 1 + p + q + 5) (h₃ : d = 5) : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l1635_163587


namespace NUMINAMATH_GPT_floor_add_self_eq_20_5_iff_l1635_163586

theorem floor_add_self_eq_20_5_iff (s : ℝ) : (⌊s⌋₊ : ℝ) + s = 20.5 ↔ s = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_floor_add_self_eq_20_5_iff_l1635_163586


namespace NUMINAMATH_GPT_find_p_l1635_163542

theorem find_p (p : ℕ) (h : 81^6 = 3^p) : p = 24 :=
sorry

end NUMINAMATH_GPT_find_p_l1635_163542


namespace NUMINAMATH_GPT_adding_books_multiplying_books_l1635_163557

-- Define the conditions
def num_books_first_shelf : ℕ := 4
def num_books_second_shelf : ℕ := 5
def num_books_third_shelf : ℕ := 6

-- Define the first question and prove its correctness
theorem adding_books :
  num_books_first_shelf + num_books_second_shelf + num_books_third_shelf = 15 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

-- Define the second question and prove its correctness
theorem multiplying_books :
  num_books_first_shelf * num_books_second_shelf * num_books_third_shelf = 120 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

end NUMINAMATH_GPT_adding_books_multiplying_books_l1635_163557


namespace NUMINAMATH_GPT_complex_number_solution_l1635_163503

variable (z : ℂ)
variable (i : ℂ)

theorem complex_number_solution (h : (1 - i)^2 / z = 1 + i) (hi : i^2 = -1) : z = -1 - i :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l1635_163503


namespace NUMINAMATH_GPT_smallest_possible_product_l1635_163554

def digits : Set ℕ := {2, 4, 5, 8}

def is_valid_pair (a b : ℤ) : Prop :=
  let (d1, d2, d3, d4) := (a / 10, a % 10, b / 10, b % 10)
  {d1.toNat, d2.toNat, d3.toNat, d4.toNat} ⊆ digits ∧ {d1.toNat, d2.toNat, d3.toNat, d4.toNat} = digits

def smallest_product : ℤ :=
  1200

theorem smallest_possible_product :
  ∀ (a b : ℤ), is_valid_pair a b → a * b ≥ smallest_product :=
by
  intro a b h
  sorry

end NUMINAMATH_GPT_smallest_possible_product_l1635_163554


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l1635_163520

theorem solve_quadratic_inequality :
  ∀ x : ℝ, ((x - 1) * (x - 3) < 0) ↔ (1 < x ∧ x < 3) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l1635_163520


namespace NUMINAMATH_GPT_total_volume_of_all_cubes_l1635_163533

def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

def total_volume (count : ℕ) (side_length : ℕ) : ℕ := count * (cube_volume side_length)

theorem total_volume_of_all_cubes :
  total_volume 4 3 + total_volume 3 4 = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_all_cubes_l1635_163533


namespace NUMINAMATH_GPT_frac_y_over_x_plus_y_eq_one_third_l1635_163504

theorem frac_y_over_x_plus_y_eq_one_third (x y : ℝ) (h : y / x = 1 / 2) : y / (x + y) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_frac_y_over_x_plus_y_eq_one_third_l1635_163504


namespace NUMINAMATH_GPT_problem_a51_l1635_163501

-- Definitions of given conditions
variable {a : ℕ → ℤ}
variable (h1 : ∀ n : ℕ, a (n + 2) - 2 * a (n + 1) + a n = 16)
variable (h2 : a 63 = 10)
variable (h3 : a 89 = 10)

-- Proof problem statement
theorem problem_a51 :
  a 51 = 3658 :=
by
  sorry

end NUMINAMATH_GPT_problem_a51_l1635_163501


namespace NUMINAMATH_GPT_proposition_holds_for_odd_numbers_l1635_163591

variable (P : ℕ → Prop)

theorem proposition_holds_for_odd_numbers 
  (h1 : P 1)
  (h_ind : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) :
  ∀ n : ℕ, n % 2 = 1 → P n :=
by
  sorry

end NUMINAMATH_GPT_proposition_holds_for_odd_numbers_l1635_163591


namespace NUMINAMATH_GPT_inequality_proof_l1635_163517

noncomputable def inequality (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : Prop :=
  (a * b) / (c - 1) + (b * c) / (a - 1) + (c * a) / (b - 1) >= 12

theorem inequality_proof (a b c : ℝ) (ha: a > 1) (hb: b > 1) (hc: c > 1) : inequality a b c ha hb hc :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1635_163517


namespace NUMINAMATH_GPT_tea_consumption_eq1_tea_consumption_eq2_l1635_163551

theorem tea_consumption_eq1 (k : ℝ) (w_sunday t_sunday w_wednesday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : w_wednesday = 4) : 
  t_wednesday = 6 := 
  by sorry

theorem tea_consumption_eq2 (k : ℝ) (w_sunday t_sunday t_thursday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : t_thursday = 2) : 
  w_thursday = 12 := 
  by sorry

end NUMINAMATH_GPT_tea_consumption_eq1_tea_consumption_eq2_l1635_163551


namespace NUMINAMATH_GPT_negation_of_diagonals_equal_l1635_163506

-- Define a rectangle type and a function for the diagonals being equal
structure Rectangle :=
  (a b c d : ℝ) -- Assuming rectangle sides

-- Assume a function that checks if the diagonals of a given rectangle are equal
def diagonals_are_equal (r : Rectangle) : Prop :=
  sorry -- The actual function definition is omitted for this context

-- The proof problem
theorem negation_of_diagonals_equal :
  ¬ (∀ r : Rectangle, diagonals_are_equal r) ↔ (∃ r : Rectangle, ¬ diagonals_are_equal r) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_diagonals_equal_l1635_163506


namespace NUMINAMATH_GPT_sqrt_range_l1635_163545

theorem sqrt_range (x : ℝ) (hx : 0 ≤ x - 1) : 1 ≤ x :=
by sorry

end NUMINAMATH_GPT_sqrt_range_l1635_163545


namespace NUMINAMATH_GPT_proof1_l1635_163583

def prob1 : Prop :=
  (1 : ℝ) * (Real.sqrt 45 + Real.sqrt 18) - (Real.sqrt 8 - Real.sqrt 125) = 8 * Real.sqrt 5 + Real.sqrt 2

theorem proof1 : prob1 :=
by
  sorry

end NUMINAMATH_GPT_proof1_l1635_163583


namespace NUMINAMATH_GPT_angle_part_a_angle_part_b_l1635_163578

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1^2 + a.2^2)

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((dot_product a b) / (magnitude a * magnitude b))

theorem angle_part_a :
  angle_between_vectors (4, 0) (2, -2) = Real.arccos (Real.sqrt 2 / 2) :=
by
  sorry

theorem angle_part_b :
  angle_between_vectors (5, -3) (3, 5) = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_angle_part_a_angle_part_b_l1635_163578


namespace NUMINAMATH_GPT_minimum_value_l1635_163540

def f (x a : ℝ) : ℝ := x^3 - a*x^2 - a^2*x
def f_prime (x a : ℝ) : ℝ := 3*x^2 - 2*a*x - a^2

theorem minimum_value (a : ℝ) (hf_prime : f_prime 1 a = 0) (ha : a = -3) : ∃ x : ℝ, f x a = -5 := 
sorry

end NUMINAMATH_GPT_minimum_value_l1635_163540


namespace NUMINAMATH_GPT_probability_hare_claims_not_hare_then_not_rabbit_l1635_163512

noncomputable def probability_hare_given_claims : ℚ := (27 / 59)

theorem probability_hare_claims_not_hare_then_not_rabbit
  (population : ℚ) (hares : ℚ) (rabbits : ℚ)
  (belief_hare_not_hare : ℚ) (belief_hare_not_rabbit : ℚ)
  (belief_rabbit_not_hare : ℚ) (belief_rabbit_not_rabbit : ℚ) :
  population = 1 ∧ hares = 1/2 ∧ rabbits = 1/2 ∧
  belief_hare_not_hare = 1/4 ∧ belief_hare_not_rabbit = 3/4 ∧
  belief_rabbit_not_hare = 2/3 ∧ belief_rabbit_not_rabbit = 1/3 →
  (27 / 59) = probability_hare_given_claims :=
sorry

end NUMINAMATH_GPT_probability_hare_claims_not_hare_then_not_rabbit_l1635_163512


namespace NUMINAMATH_GPT_percentage_pine_cones_on_roof_l1635_163565

theorem percentage_pine_cones_on_roof 
  (num_trees : Nat) 
  (pine_cones_per_tree : Nat) 
  (pine_cone_weight_oz : Nat) 
  (total_pine_cone_weight_on_roof_oz : Nat) 
  : num_trees = 8 ∧ pine_cones_per_tree = 200 ∧ pine_cone_weight_oz = 4 ∧ total_pine_cone_weight_on_roof_oz = 1920 →
    (total_pine_cone_weight_on_roof_oz / pine_cone_weight_oz) / (num_trees * pine_cones_per_tree) * 100 = 30 := 
by
  sorry

end NUMINAMATH_GPT_percentage_pine_cones_on_roof_l1635_163565


namespace NUMINAMATH_GPT_regular_18gon_lines_rotational_symmetry_sum_l1635_163575

def L : ℕ := 18
def R : ℕ := 20

theorem regular_18gon_lines_rotational_symmetry_sum : L + R = 38 :=
by 
  sorry

end NUMINAMATH_GPT_regular_18gon_lines_rotational_symmetry_sum_l1635_163575


namespace NUMINAMATH_GPT_smallest_constant_N_l1635_163514

theorem smallest_constant_N (a : ℝ) (ha : a > 0) : 
  let b := a
  let c := a
  (a = b ∧ b = c) → (a^2 + b^2 + c^2) / (a + b + c) > (0 : ℝ) := 
by
  -- Assuming the proof steps are written here
  sorry

end NUMINAMATH_GPT_smallest_constant_N_l1635_163514


namespace NUMINAMATH_GPT_perimeter_of_AF1B_l1635_163577

noncomputable def ellipse_perimeter (a b x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (2 * a)

theorem perimeter_of_AF1B (h : (6:ℝ) = 6) :
  ellipse_perimeter 6 4 0 0 6 0 = 24 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_AF1B_l1635_163577


namespace NUMINAMATH_GPT_rate_per_kg_for_fruits_l1635_163580

-- Definitions and conditions
def total_cost (rate_per_kg : ℝ) : ℝ := 8 * rate_per_kg + 9 * rate_per_kg

def total_paid : ℝ := 1190

theorem rate_per_kg_for_fruits : ∃ R : ℝ, total_cost R = total_paid ∧ R = 70 :=
by
  sorry

end NUMINAMATH_GPT_rate_per_kg_for_fruits_l1635_163580


namespace NUMINAMATH_GPT_intersection_with_complement_N_l1635_163521

open Set Real

def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def N : Set ℝ := {x | 0 < x ∧ x < 2}
def complement_N : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

theorem intersection_with_complement_N : M ∩ complement_N = Ico 2 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_with_complement_N_l1635_163521


namespace NUMINAMATH_GPT_find_angle_A_l1635_163524

theorem find_angle_A (A B C a b c : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : B = (A + C) / 2)
  (h3 : 2 * b ^ 2 = 3 * a * c) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l1635_163524


namespace NUMINAMATH_GPT_efficiency_of_worker_p_more_than_q_l1635_163513

noncomputable def worker_p_rate : ℚ := 1 / 22
noncomputable def combined_rate : ℚ := 1 / 12

theorem efficiency_of_worker_p_more_than_q
  (W_p : ℚ) (W_q : ℚ)
  (h1 : W_p = worker_p_rate)
  (h2 : W_p + W_q = combined_rate) : (W_p / W_q) = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_efficiency_of_worker_p_more_than_q_l1635_163513


namespace NUMINAMATH_GPT_dot_product_ABC_l1635_163567

-- Defining vectors as pairs of real numbers
def vector := (ℝ × ℝ)

-- Defining the vectors AB and AC
def AB : vector := (1, 0)
def AC : vector := (-2, 3)

-- Definition of vector subtraction
def vector_sub (v1 v2 : vector) : vector := (v1.1 - v2.1, v1.2 - v2.2)

-- Definition of dot product
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define vector BC using the given vectors AB and AC
def BC : vector := vector_sub AC AB

-- The theorem stating the desired dot product result
theorem dot_product_ABC : dot_product AB BC = -3 := by
  sorry

end NUMINAMATH_GPT_dot_product_ABC_l1635_163567


namespace NUMINAMATH_GPT_light_bulb_arrangement_l1635_163538

theorem light_bulb_arrangement :
  let B := 6
  let R := 7
  let W := 9
  let total_arrangements := Nat.choose (B + R) B * Nat.choose (B + R + 1) W
  total_arrangements = 3435432 :=
by
  sorry

end NUMINAMATH_GPT_light_bulb_arrangement_l1635_163538


namespace NUMINAMATH_GPT_pencils_added_l1635_163570

theorem pencils_added (initial_pencils total_pencils Mike_pencils : ℕ) 
    (h1 : initial_pencils = 41) 
    (h2 : total_pencils = 71) 
    (h3 : total_pencils = initial_pencils + Mike_pencils) :
    Mike_pencils = 30 := by
  sorry

end NUMINAMATH_GPT_pencils_added_l1635_163570


namespace NUMINAMATH_GPT_roots_of_quadratic_discriminant_positive_l1635_163537

theorem roots_of_quadratic_discriminant_positive {a b c : ℝ} (h : b^2 - 4 * a * c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_roots_of_quadratic_discriminant_positive_l1635_163537


namespace NUMINAMATH_GPT_part1_part2_l1635_163579

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 2)

theorem part1 : 
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry 

noncomputable def g (x : ℝ) : ℝ := f x - x^2 + x

theorem part2 (m : ℝ) : 
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
sorry 

end NUMINAMATH_GPT_part1_part2_l1635_163579


namespace NUMINAMATH_GPT_cubic_polynomial_roots_value_l1635_163552

theorem cubic_polynomial_roots_value
  (a b c d : ℝ) 
  (h_cond : a ≠ 0 ∧ d ≠ 0)
  (h_equiv : (a * (1/2)^3 + b * (1/2)^2 + c * (1/2) + d) + (a * (-1/2)^3 + b * (-1/2)^2 + c * (-1/2) + d) = 1000 * d)
  (h_roots : ∃ (x1 x2 x3 : ℝ), a * x1^3 + b * x1^2 + c * x1 + d = 0 ∧ a * x2^3 + b * x2^2 + c * x2 + d = 0 ∧ a * x3^3 + b * x3^2 + c * x3 + d = 0) 
  : (∃ (x1 x2 x3 : ℝ), (1 / (x1 * x2) + 1 / (x2 * x3) + 1 / (x1 * x3) = 1996)) :=
by
  sorry

end NUMINAMATH_GPT_cubic_polynomial_roots_value_l1635_163552


namespace NUMINAMATH_GPT_problem_1_problem_2_l1635_163597

variable {c : ℝ}

def p (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → c ^ x₁ > c ^ x₂

def q (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, (1 / 2) < x₁ ∧ x₁ < x₂ → (x₁ ^ 2 - 2 * c * x₁ + 1) < (x₂ ^ 2 - 2 * c * x₂ + 1)

theorem problem_1 (hc : 0 < c) (hcn1 : c ≠ 1) (hp : p c) (hnq_false : ¬ ¬ q c) : 0 < c ∧ c ≤ 1 / 2 :=
by
  sorry

theorem problem_2 (hc : 0 < c) (hcn1 : c ≠ 1) (hpq_false : ¬ (p c ∧ q c)) (hp_or_q : p c ∨ q c) : 1 / 2 < c ∧ c < 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1635_163597


namespace NUMINAMATH_GPT_find_k_in_geometric_sequence_l1635_163547

theorem find_k_in_geometric_sequence (c k : ℝ) (h1_nonzero : c ≠ 0)
  (S : ℕ → ℝ) (a : ℕ → ℝ) (h2 : ∀ n, a (n + 1) = c * a n)
  (h3 : ∀ n, S n = 3^n + k)
  (h4 : a 1 = 3 + k)
  (h5 : a 2 = S 2 - S 1)
  (h6 : a 3 = S 3 - S 2) : k = -1 := by
  sorry

end NUMINAMATH_GPT_find_k_in_geometric_sequence_l1635_163547


namespace NUMINAMATH_GPT_mom_has_enough_money_l1635_163582

def original_price : ℝ := 268
def discount_rate : ℝ := 0.2
def money_brought : ℝ := 230
def discounted_price := original_price * (1 - discount_rate)

theorem mom_has_enough_money : money_brought ≥ discounted_price := by
  sorry

end NUMINAMATH_GPT_mom_has_enough_money_l1635_163582


namespace NUMINAMATH_GPT_quadratic_monotonic_range_l1635_163534

theorem quadratic_monotonic_range {a : ℝ} :
  (∀ x1 x2 : ℝ, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → (x1^2 - 2*a*x1 + 1) ≤ (x2^2 - 2*a*x2 + 1) ∨ (x1^2 - 2*a*x1 + 1) ≥ (x2^2 - 2*a*x2 + 1)) → (a ≤ 2 ∨ a ≥ 3) := 
sorry

end NUMINAMATH_GPT_quadratic_monotonic_range_l1635_163534


namespace NUMINAMATH_GPT_total_crayons_is_12_l1635_163522

-- Definitions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Goal to prove
theorem total_crayons_is_12 : initial_crayons + added_crayons = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_crayons_is_12_l1635_163522


namespace NUMINAMATH_GPT_truck_loading_time_l1635_163599

theorem truck_loading_time (h1_rate h2_rate h3_rate : ℝ)
  (h1 : h1_rate = 1 / 5) (h2 : h2_rate = 1 / 4) (h3 : h3_rate = 1 / 6) :
  (1 / (h1_rate + h2_rate + h3_rate)) = 60 / 37 :=
by simp [h1, h2, h3]; sorry

end NUMINAMATH_GPT_truck_loading_time_l1635_163599


namespace NUMINAMATH_GPT_least_n_for_multiple_of_8_l1635_163502

def is_positive_integer (n : ℕ) : Prop := n > 0

def is_multiple_of_8 (k : ℕ) : Prop := ∃ m : ℕ, k = 8 * m

theorem least_n_for_multiple_of_8 :
  ∀ n : ℕ, (is_positive_integer n → is_multiple_of_8 (Nat.factorial n)) → n ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_least_n_for_multiple_of_8_l1635_163502


namespace NUMINAMATH_GPT_carA_catches_up_with_carB_at_150_km_l1635_163531

-- Definitions representing the problem's conditions
variable (t_A t_B v_A v_B : ℝ)
variable (distance_A_B : ℝ := 300)
variable (time_diff_start : ℝ := 1)
variable (time_diff_end : ℝ := 1)

-- Assumptions representing the problem's conditions
axiom speed_carA : v_A = distance_A_B / t_A
axiom speed_carB : v_B = distance_A_B / (t_A + 2)
axiom time_relation : t_B = t_A + 2
axiom time_diff_starting : t_A = t_B - 2

-- The statement to be proven: car A catches up with car B 150 km from city B
theorem carA_catches_up_with_carB_at_150_km :
  ∃ t₀ : ℝ, v_A * t₀ = v_B * (t₀ + time_diff_start) ∧ (distance_A_B - v_A * t₀ = 150) :=
sorry

end NUMINAMATH_GPT_carA_catches_up_with_carB_at_150_km_l1635_163531


namespace NUMINAMATH_GPT_minimum_positive_temperature_announcement_l1635_163530

-- Problem conditions translated into Lean
def num_interactions (x : ℕ) : ℕ := x * (x - 1)
def total_interactions := 132
def total_positive := 78
def total_negative := 54
def positive_temperature_count (x y : ℕ) : ℕ := y * (y - 1)
def negative_temperature_count (x y : ℕ) : ℕ := (x - y) * (x - 1 - y)
def minimum_positive_temperature (x y : ℕ) := 
  x = 12 → 
  total_interactions = total_positive + total_negative →
  total_positive + total_negative = num_interactions x →
  total_positive = positive_temperature_count x y →
  sorry -- proof goes here

theorem minimum_positive_temperature_announcement : ∃ y, 
  minimum_positive_temperature 12 y ∧ y = 3 :=
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_minimum_positive_temperature_announcement_l1635_163530
