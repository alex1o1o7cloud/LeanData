import Mathlib

namespace NUMINAMATH_GPT_solution_set_of_inequality_l444_44410

theorem solution_set_of_inequality (x : ℝ) : (|x - 3| < 1) → (2 < x ∧ x < 4) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l444_44410


namespace NUMINAMATH_GPT_find_m_l444_44483

def vector_perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) : vector_perpendicular (3, 1) (m, -3) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l444_44483


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l444_44498

theorem number_of_men_in_first_group (M : ℕ) : (M * 15 = 25 * 18) → M = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_men_in_first_group_l444_44498


namespace NUMINAMATH_GPT_Sues_necklace_total_beads_l444_44443

theorem Sues_necklace_total_beads 
  (purple_beads : ℕ)
  (blue_beads : ℕ)
  (green_beads : ℕ)
  (h1 : purple_beads = 7)
  (h2 : blue_beads = 2 * purple_beads)
  (h3 : green_beads = blue_beads + 11) :
  purple_beads + blue_beads + green_beads = 46 :=
by
  sorry

end NUMINAMATH_GPT_Sues_necklace_total_beads_l444_44443


namespace NUMINAMATH_GPT_find_second_divisor_l444_44481

theorem find_second_divisor (k : ℕ) (d : ℕ) 
  (h1 : k % 5 = 2)
  (h2 : k < 42)
  (h3 : k % 7 = 3)
  (h4 : k % d = 5) : d = 12 := 
sorry

end NUMINAMATH_GPT_find_second_divisor_l444_44481


namespace NUMINAMATH_GPT_range_of_b_for_monotonic_function_l444_44471

theorem range_of_b_for_monotonic_function :
  (∀ x : ℝ, (x^2 + 2 * b * x + b + 2) ≥ 0) ↔ (-1 ≤ b ∧ b ≤ 2) :=
by sorry

end NUMINAMATH_GPT_range_of_b_for_monotonic_function_l444_44471


namespace NUMINAMATH_GPT_hundredth_odd_positive_integer_equals_199_even_integer_following_199_equals_200_l444_44429

theorem hundredth_odd_positive_integer_equals_199 : (2 * 100 - 1 = 199) :=
by {
  sorry
}

theorem even_integer_following_199_equals_200 : (199 + 1 = 200) :=
by {
  sorry
}

end NUMINAMATH_GPT_hundredth_odd_positive_integer_equals_199_even_integer_following_199_equals_200_l444_44429


namespace NUMINAMATH_GPT_multiple_of_27_l444_44411

theorem multiple_of_27 (x y z : ℤ) 
  (h1 : (2 * x + 5 * y + 11 * z) = 4 * (x + y + z)) 
  (h2 : (2 * x + 20 * y + 110 * z) = 6 * (2 * x + 5 * y + 11 * z)) :
  ∃ k : ℤ, x + y + z = 27 * k :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_27_l444_44411


namespace NUMINAMATH_GPT_reciprocal_of_neg_two_l444_44401

theorem reciprocal_of_neg_two : 1 / (-2) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_two_l444_44401


namespace NUMINAMATH_GPT_sin_and_tan_alpha_in_second_quadrant_expression_value_for_given_tan_l444_44450

theorem sin_and_tan_alpha_in_second_quadrant 
  (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (hcos : Real.cos α = -8 / 17) :
  Real.sin α = 15 / 17 ∧ Real.tan α = -15 / 8 := 
  sorry

theorem expression_value_for_given_tan 
  (α : ℝ) (htan : Real.tan α = 2) :
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 := 
  sorry

end NUMINAMATH_GPT_sin_and_tan_alpha_in_second_quadrant_expression_value_for_given_tan_l444_44450


namespace NUMINAMATH_GPT_sequences_zero_at_2_l444_44489

theorem sequences_zero_at_2
  (a b c d : ℕ → ℝ)
  (h1 : ∀ n, a (n+1) = a n + b n)
  (h2 : ∀ n, b (n+1) = b n + c n)
  (h3 : ∀ n, c (n+1) = c n + d n)
  (h4 : ∀ n, d (n+1) = d n + a n)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (h5 : a (k + m) = a m)
  (h6 : b (k + m) = b m)
  (h7 : c (k + m) = c m)
  (h8 : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 :=
by sorry

end NUMINAMATH_GPT_sequences_zero_at_2_l444_44489


namespace NUMINAMATH_GPT_oliver_first_coupon_redeem_on_friday_l444_44440

-- Definitions of conditions in the problem
def has_coupons (n : ℕ) := n = 8
def uses_coupon_every_9_days (days : ℕ) := days = 9
def is_closed_on_monday (day : ℕ) := day % 7 = 1  -- Assuming 1 represents Monday
def does_not_redeem_on_closed_day (redemption_days : List ℕ) :=
  ∀ day ∈ redemption_days, day % 7 ≠ 1

-- Main theorem statement
theorem oliver_first_coupon_redeem_on_friday : 
  ∃ (first_redeem_day: ℕ), 
  has_coupons 8 ∧ uses_coupon_every_9_days 9 ∧
  is_closed_on_monday 1 ∧ 
  does_not_redeem_on_closed_day [first_redeem_day, first_redeem_day + 9, first_redeem_day + 18, first_redeem_day + 27, first_redeem_day + 36, first_redeem_day + 45, first_redeem_day + 54, first_redeem_day + 63] ∧ 
  first_redeem_day % 7 = 5 := sorry

end NUMINAMATH_GPT_oliver_first_coupon_redeem_on_friday_l444_44440


namespace NUMINAMATH_GPT_total_kayaks_built_by_April_l444_44446

theorem total_kayaks_built_by_April
    (a : Nat := 9) (r : Nat := 3) (n : Nat := 4) :
    let S := a * (r ^ n - 1) / (r - 1)
    S = 360 := by
  sorry

end NUMINAMATH_GPT_total_kayaks_built_by_April_l444_44446


namespace NUMINAMATH_GPT_emily_sixth_score_needed_l444_44436

def emily_test_scores : List ℕ := [88, 92, 85, 90, 97]

def needed_sixth_score (scores : List ℕ) (target_mean : ℕ) : ℕ :=
  let current_sum := scores.sum
  let total_sum_needed := target_mean * (scores.length + 1)
  total_sum_needed - current_sum

theorem emily_sixth_score_needed :
  needed_sixth_score emily_test_scores 91 = 94 := by
  sorry

end NUMINAMATH_GPT_emily_sixth_score_needed_l444_44436


namespace NUMINAMATH_GPT_Mike_height_l444_44457

theorem Mike_height (h_mark: 5 * 12 + 3 = 63) (h_mark_mike:  63 + 10 = 73) (h_foot: 12 = 12)
: 73 / 12 = 6 ∧ 73 % 12 = 1 := 
sorry

end NUMINAMATH_GPT_Mike_height_l444_44457


namespace NUMINAMATH_GPT_correct_relation_l444_44448

def satisfies_relation : Prop :=
  (∀ x y, (x = 0 ∧ y = 200) ∨ (x = 1 ∧ y = 170) ∨ (x = 2 ∧ y = 120) ∨ (x = 3 ∧ y = 50) ∨ (x = 4 ∧ y = 0) →
  y = 200 - 10 * x - 10 * x^2) 

theorem correct_relation : satisfies_relation :=
sorry

end NUMINAMATH_GPT_correct_relation_l444_44448


namespace NUMINAMATH_GPT_prove_billy_age_l444_44462

-- Define B and J as real numbers representing the ages of Billy and Joe respectively
variables (B J : ℝ)

-- State the conditions
def billy_triple_of_joe : Prop := B = 3 * J
def sum_of_ages : Prop := B + J = 63

-- State the proposition to prove
def billy_age_proof : Prop := B = 47.25

-- Main theorem combining the conditions and the proof statement
theorem prove_billy_age (h1 : billy_triple_of_joe B J) (h2 : sum_of_ages B J) : billy_age_proof B :=
by
  sorry

end NUMINAMATH_GPT_prove_billy_age_l444_44462


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l444_44409

theorem cone_lateral_surface_area (r V: ℝ) (h : ℝ) (l : ℝ) (L: ℝ):
  r = 3 →
  V = 12 * Real.pi →
  V = (1 / 3) * Real.pi * r^2 * h →
  l = Real.sqrt (r^2 + h^2) →
  L = Real.pi * r * l →
  L = 15 * Real.pi :=
by
  intros hr hv hV hl hL
  rw [hr, hv] at hV
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l444_44409


namespace NUMINAMATH_GPT_evaluate_expression_l444_44420

theorem evaluate_expression : 40 + 5 * 12 / (180 / 3) = 41 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l444_44420


namespace NUMINAMATH_GPT_findYears_l444_44495

def totalInterest (n : ℕ) : ℕ :=
  24 * n + 70 * n

theorem findYears (n : ℕ) : totalInterest n = 350 → n = 4 := 
sorry

end NUMINAMATH_GPT_findYears_l444_44495


namespace NUMINAMATH_GPT_measure_of_one_exterior_angle_l444_44482

theorem measure_of_one_exterior_angle (n : ℕ) (h : n > 2) : 
  n > 2 → ∃ (angle : ℝ), angle = 360 / n :=
by 
  sorry

end NUMINAMATH_GPT_measure_of_one_exterior_angle_l444_44482


namespace NUMINAMATH_GPT_symmetric_inverse_sum_l444_44422

theorem symmetric_inverse_sum {f g : ℝ → ℝ} (h₁ : ∀ x, f (-x - 2) = -f (x)) (h₂ : ∀ y, g (f y) = y) (h₃ : ∀ y, f (g y) = y) (x₁ x₂ : ℝ) (h₄ : x₁ + x₂ = 0) : 
  g x₁ + g x₂ = -2 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_inverse_sum_l444_44422


namespace NUMINAMATH_GPT_c_sq_minus_a_sq_divisible_by_48_l444_44493

theorem c_sq_minus_a_sq_divisible_by_48
  (a b c : ℤ) (h_ac : a < c) (h_eq : a^2 + c^2 = 2 * b^2) : 48 ∣ (c^2 - a^2) := 
  sorry

end NUMINAMATH_GPT_c_sq_minus_a_sq_divisible_by_48_l444_44493


namespace NUMINAMATH_GPT_total_ladybugs_and_ants_l444_44435

def num_leaves : ℕ := 84
def ladybugs_per_leaf : ℕ := 139
def ants_per_leaf : ℕ := 97

def total_ladybugs := ladybugs_per_leaf * num_leaves
def total_ants := ants_per_leaf * num_leaves
def total_insects := total_ladybugs + total_ants

theorem total_ladybugs_and_ants : total_insects = 19824 := by
  sorry

end NUMINAMATH_GPT_total_ladybugs_and_ants_l444_44435


namespace NUMINAMATH_GPT_books_before_grant_correct_l444_44447

-- Definitions based on the given conditions
def books_purchased : ℕ := 2647
def total_books_now : ℕ := 8582

-- Definition and the proof statement
def books_before_grant : ℕ := 5935

-- Proof statement: The number of books before the grant plus the books purchased equals the total books now
theorem books_before_grant_correct :
  books_before_grant + books_purchased = total_books_now :=
by
  -- Predictably, no need to complete proof, 'sorry' is used.
  sorry

end NUMINAMATH_GPT_books_before_grant_correct_l444_44447


namespace NUMINAMATH_GPT_problem_statement_l444_44442

theorem problem_statement (p q : Prop) :
  ¬(p ∧ q) ∧ ¬¬p → ¬q := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l444_44442


namespace NUMINAMATH_GPT_min_square_side_length_l444_44415

theorem min_square_side_length 
  (table_length : ℕ) (table_breadth : ℕ) (cube_side : ℕ) (num_tables : ℕ)
  (cond1 : table_length = 12)
  (cond2 : table_breadth = 16)
  (cond3 : cube_side = 4)
  (cond4 : num_tables = 4) :
  (2 * table_length + 2 * table_breadth) = 56 := 
by
  sorry

end NUMINAMATH_GPT_min_square_side_length_l444_44415


namespace NUMINAMATH_GPT_man_walking_rate_is_12_l444_44475

theorem man_walking_rate_is_12 (M : ℝ) (woman_speed : ℝ) (time_waiting : ℝ) (catch_up_time : ℝ) 
  (woman_speed_eq : woman_speed = 12) (time_waiting_eq : time_waiting = 1 / 6) 
  (catch_up_time_eq : catch_up_time = 1 / 6): 
  (M * catch_up_time = woman_speed * time_waiting) → M = 12 := by
  intro h
  rw [woman_speed_eq, time_waiting_eq, catch_up_time_eq] at h
  sorry

end NUMINAMATH_GPT_man_walking_rate_is_12_l444_44475


namespace NUMINAMATH_GPT_prime_sum_remainder_l444_44468

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end NUMINAMATH_GPT_prime_sum_remainder_l444_44468


namespace NUMINAMATH_GPT_simplify_expression_l444_44453

theorem simplify_expression :
  let a := 7
  let b := 11
  let c := 19
  (49 * (1 / 11 - 1 / 19) + 121 * (1 / 19 - 1 / 7) + 361 * (1 / 7 - 1 / 11)) /
  (7 * (1 / 11 - 1 / 19) + 11 * (1 / 19 - 1 / 7) + 19 * (1 / 7 - 1 / 11)) = 37 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l444_44453


namespace NUMINAMATH_GPT_find_x2_x1_add_x3_l444_44469

-- Definition of the polynomial
def polynomial (x : ℝ) : ℝ := (10*x^3 - 210*x^2 + 3)

-- Statement including conditions and the question we need to prove
theorem find_x2_x1_add_x3 :
  ∃ x₁ x₂ x₃ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ 
    polynomial x₁ = 0 ∧ 
    polynomial x₂ = 0 ∧ 
    polynomial x₃ = 0 ∧ 
    x₂ * (x₁ + x₃) = 21 :=
by sorry

end NUMINAMATH_GPT_find_x2_x1_add_x3_l444_44469


namespace NUMINAMATH_GPT_triangle_right_angle_and_m_values_l444_44430

open Real

-- Definitions and conditions
def line_AB (x y : ℝ) : Prop := 3 * x - 2 * y + 6 = 0
def line_AC (x y : ℝ) : Prop := 2 * x + 3 * y - 22 = 0
def line_BC (x y m : ℝ) : Prop := 3 * x + 4 * y - m = 0

-- Prove the shape and value of m when the height from BC is 1
theorem triangle_right_angle_and_m_values :
  (∃ (x y : ℝ), line_AB x y ∧ line_AC x y ∧ line_AB x y ∧ (-3/2) ≠ (2/3)) ∧
  (∀ x y, line_AB x y → line_AC x y → 3 * x + 4 * y - 25 = 0 ∨ 3 * x + 4 * y - 35 = 0) := 
sorry

end NUMINAMATH_GPT_triangle_right_angle_and_m_values_l444_44430


namespace NUMINAMATH_GPT_largest_systematic_sample_l444_44473

theorem largest_systematic_sample {n_products interval start second_smallest max_sample : ℕ} 
  (h1 : n_products = 300) 
  (h2 : start = 2) 
  (h3 : second_smallest = 17) 
  (h4 : interval = second_smallest - start) 
  (h5 : n_products % interval = 0) 
  (h6 : max_sample = start + (interval * ((n_products / interval) - 1))) : 
  max_sample = 287 := 
by
  -- This is where the proof would go if required.
  sorry

end NUMINAMATH_GPT_largest_systematic_sample_l444_44473


namespace NUMINAMATH_GPT_polynomial_function_correct_l444_44427

theorem polynomial_function_correct :
  ∀ (f : ℝ → ℝ),
  (∀ (x : ℝ), f (x^2 + 1) = x^4 + 5 * x^2 + 3) →
  ∀ (x : ℝ), f (x^2 - 1) = x^4 + x^2 - 3 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_function_correct_l444_44427


namespace NUMINAMATH_GPT_units_digit_of_p_is_6_l444_44472

-- Given conditions
variable (p : ℕ)
variable (h1 : p % 2 = 0)                -- p is a positive even integer
variable (h2 : (p^3 % 10) - (p^2 % 10) = 0)  -- The units digit of p^3 minus the units digit of p^2 is 0
variable (h3 : (p + 2) % 10 = 8)         -- The units digit of p + 2 is 8

-- Prove the units digit of p is 6
theorem units_digit_of_p_is_6 : p % 10 = 6 :=
sorry

end NUMINAMATH_GPT_units_digit_of_p_is_6_l444_44472


namespace NUMINAMATH_GPT_point_on_angle_bisector_l444_44426

theorem point_on_angle_bisector (a : ℝ) 
  (h : (2 : ℝ) * a + (3 : ℝ) = a) : a = -3 :=
sorry

end NUMINAMATH_GPT_point_on_angle_bisector_l444_44426


namespace NUMINAMATH_GPT_problem1_correct_problem2_correct_l444_44407

-- Definition for Problem 1
def problem1 (a b c d : ℚ) : ℚ :=
  (a - b + c) * d

-- Statement for Problem 1
theorem problem1_correct : problem1 (1/6) (5/7) (2/3) (-42) = -5 :=
by
  sorry

-- Definitions for Problem 2
def problem2 (a b c d : ℚ) : ℚ :=
  (-a^2 + b^2 * c - d^2 / |d|)

-- Statement for Problem 2
theorem problem2_correct : problem2 (-2) (-3) (-2/3) 4 = -14 :=
by
  sorry

end NUMINAMATH_GPT_problem1_correct_problem2_correct_l444_44407


namespace NUMINAMATH_GPT_boxes_per_hand_l444_44455

theorem boxes_per_hand (total_people : ℕ) (total_boxes : ℕ) (boxes_per_person : ℕ) (hands_per_person : ℕ) 
  (h1: total_people = 10) (h2: total_boxes = 20) (h3: boxes_per_person = total_boxes / total_people) 
  (h4: hands_per_person = 2) : boxes_per_person / hands_per_person = 1 := 
by
  sorry

end NUMINAMATH_GPT_boxes_per_hand_l444_44455


namespace NUMINAMATH_GPT_find_a8_l444_44433

def seq (a : Nat → Int) := a 1 = -1 ∧ ∀ n, a (n + 1) = a n - 3

theorem find_a8 (a : Nat → Int) (h : seq a) : a 8 = -22 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a8_l444_44433


namespace NUMINAMATH_GPT_length_of_train_l444_44434

-- Definitions based on the conditions in the problem
def time_to_cross_signal_pole : ℝ := 18
def time_to_cross_platform : ℝ := 54
def length_of_platform : ℝ := 600.0000000000001

-- Prove that the length of the train is 300.00000000000005 meters
theorem length_of_train
    (L V : ℝ)
    (h1 : L = V * time_to_cross_signal_pole)
    (h2 : L + length_of_platform = V * time_to_cross_platform) :
    L = 300.00000000000005 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l444_44434


namespace NUMINAMATH_GPT_incorrect_statement_C_l444_44414

theorem incorrect_statement_C :
  (∀ (b h : ℝ), b > 0 → h > 0 → 2 * (b * h) = (2 * b) * h) ∧
  (∀ (r h : ℝ), r > 0 → h > 0 → 2 * (π * r^2 * h) = π * r^2 * (2 * h)) ∧
  (∀ (a : ℝ), a > 0 → 4 * (a^3) ≠ (2 * a)^3) ∧
  (∀ (a b : ℚ), b ≠ 0 → a / (2 * b) ≠ (a / 2) / b) ∧
  (∀ (x : ℝ), x < 0 → 2 * x < x) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_C_l444_44414


namespace NUMINAMATH_GPT_value_of_k_l444_44466

-- Define the conditions of the quartic equation and the product of two roots
variable (a b c d k : ℝ)
variable (hx : (Polynomial.X ^ 4 - 18 * Polynomial.X ^ 3 + k * Polynomial.X ^ 2 + 200 * Polynomial.X - 1984).rootSet ℝ = {a, b, c, d})
variable (hprod_ab : a * b = -32)

-- The statement to prove: the value of k is 86
theorem value_of_k :
  k = 86 :=
by sorry

end NUMINAMATH_GPT_value_of_k_l444_44466


namespace NUMINAMATH_GPT_trapezium_division_l444_44428

theorem trapezium_division (h : ℝ) (m n : ℕ) (h_pos : 0 < h) 
  (areas_equal : 4 / (3 * ↑m) = 7 / (6 * ↑n)) :
  m + n = 15 := by
  sorry

end NUMINAMATH_GPT_trapezium_division_l444_44428


namespace NUMINAMATH_GPT_ab_value_l444_44437

theorem ab_value (a b : ℝ) (h1 : a + b = 7) (h2 : a^3 + b^3 = 91) : a * b = 12 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l444_44437


namespace NUMINAMATH_GPT_projection_plane_right_angle_l444_44485

-- Given conditions and definitions
def is_right_angle (α β : ℝ) : Prop := α = 90 ∧ β = 90
def is_parallel_to_side (plane : ℝ → ℝ → Prop) (side : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, plane x y ↔ a * x + b * y = c ∧ ∃ d e : ℝ, ∀ x y : ℝ, side x y ↔ d * x + e * y = 90

theorem projection_plane_right_angle (plane : ℝ → ℝ → Prop) (side1 side2 : ℝ → ℝ → Prop) :
  is_right_angle (90 : ℝ) (90 : ℝ) →
  (is_parallel_to_side plane side1 ∨ is_parallel_to_side plane side2) →
  ∃ α β : ℝ, is_right_angle α β :=
by 
  sorry

end NUMINAMATH_GPT_projection_plane_right_angle_l444_44485


namespace NUMINAMATH_GPT_range_of_m_l444_44461

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 > 0
def B (x : ℝ) (m : ℝ) : Prop := 2 * m - 1 ≤ x ∧ x ≤ m + 3
def subset (B A : ℝ → Prop) : Prop := ∀ x, B x → A x

theorem range_of_m (m : ℝ) : (∀ x, B x m → A x) ↔ (m < -4 ∨ m > 2) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l444_44461


namespace NUMINAMATH_GPT_point_A_on_x_axis_l444_44400

def point_A : ℝ × ℝ := (-2, 0)

theorem point_A_on_x_axis : point_A.snd = 0 :=
by
  unfold point_A
  sorry

end NUMINAMATH_GPT_point_A_on_x_axis_l444_44400


namespace NUMINAMATH_GPT_close_time_for_pipe_b_l444_44432

-- Define entities and rates
def rate_fill (A_rate B_rate : ℝ) (t_fill t_empty t_fill_target t_close : ℝ) : Prop :=
  A_rate = 1 / t_fill ∧
  B_rate = 1 / t_empty ∧
  t_fill_target = 30 ∧
  A_rate * (t_close + (t_fill_target - t_close)) - B_rate * t_close = 1

-- Declare the theorem statement
theorem close_time_for_pipe_b (A_rate B_rate t_fill_target t_fill t_empty t_close: ℝ) :
   rate_fill A_rate B_rate t_fill t_empty t_fill_target t_close → t_close = 26.25 :=
by have h1 : A_rate = 1 / 15 := by sorry
   have h2 : B_rate = 1 / 24 := by sorry
   have h3 : t_fill_target = 30 := by sorry
   sorry

end NUMINAMATH_GPT_close_time_for_pipe_b_l444_44432


namespace NUMINAMATH_GPT_find_m_probability_l444_44418

theorem find_m_probability (m : ℝ) (ξ : ℕ → ℝ) :
  (ξ 1 = m * (2/3)) ∧ (ξ 2 = m * (2/3)^2) ∧ (ξ 3 = m * (2/3)^3) ∧ 
  (ξ 1 + ξ 2 + ξ 3 = 1) → 
  m = 27 / 38 := 
sorry

end NUMINAMATH_GPT_find_m_probability_l444_44418


namespace NUMINAMATH_GPT_angle_A_in_quadrilateral_l444_44480

noncomputable def degree_measure_A (A B C D : ℝ) := A

theorem angle_A_in_quadrilateral 
  (A B C D : ℝ)
  (hA : A = 3 * B)
  (hC : A = 4 * C)
  (hD : A = 6 * D)
  (sum_angles : A + B + C + D = 360) :
  degree_measure_A A B C D = 206 :=
by
  sorry

end NUMINAMATH_GPT_angle_A_in_quadrilateral_l444_44480


namespace NUMINAMATH_GPT_no_nonconstant_poly_prime_for_all_l444_44405

open Polynomial

theorem no_nonconstant_poly_prime_for_all (f : Polynomial ℤ) (h : ∀ n : ℕ, Prime (f.eval (n : ℤ))) :
  ∃ c : ℤ, f = Polynomial.C c :=
sorry

end NUMINAMATH_GPT_no_nonconstant_poly_prime_for_all_l444_44405


namespace NUMINAMATH_GPT_percent_gain_is_5_333_l444_44464

noncomputable def calculate_percent_gain (total_sheep : ℕ) 
                                         (sold_sheep : ℕ) 
                                         (price_paid_sheep : ℕ) 
                                         (sold_remaining_sheep : ℕ)
                                         (remaining_sheep : ℕ) 
                                         (total_cost : ℝ) 
                                         (initial_revenue : ℝ) 
                                         (remaining_revenue : ℝ) : ℝ :=
  (remaining_revenue + initial_revenue - total_cost) / total_cost * 100

theorem percent_gain_is_5_333
  (x : ℝ)
  (total_sheep : ℕ := 800)
  (sold_sheep : ℕ := 750)
  (price_paid_sheep : ℕ := 790)
  (remaining_sheep : ℕ := 50)
  (total_cost : ℝ := (800 : ℝ) * x)
  (initial_revenue : ℝ := (790 : ℝ) * x)
  (remaining_revenue : ℝ := (50 : ℝ) * ((790 : ℝ) * x / 750)) :
  calculate_percent_gain total_sheep sold_sheep price_paid_sheep remaining_sheep 50 total_cost initial_revenue remaining_revenue = 5.333 := by
  sorry

end NUMINAMATH_GPT_percent_gain_is_5_333_l444_44464


namespace NUMINAMATH_GPT_cars_meet_time_l444_44406

-- Define the initial conditions as Lean definitions
def distance_car1 (t : ℝ) : ℝ := 15 * t
def distance_car2 (t : ℝ) : ℝ := 20 * t
def total_distance : ℝ := 105

-- Define the proposition we want to prove
theorem cars_meet_time : ∃ (t : ℝ), distance_car1 t + distance_car2 t = total_distance ∧ t = 3 :=
by
  sorry

end NUMINAMATH_GPT_cars_meet_time_l444_44406


namespace NUMINAMATH_GPT_find_f_zero_l444_44459

theorem find_f_zero (f : ℝ → ℝ) (h : ∀ x, f ((x + 1) / (x - 1)) = x^2 + 3) : f 0 = 4 :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_find_f_zero_l444_44459


namespace NUMINAMATH_GPT_museum_wings_paintings_l444_44487

theorem museum_wings_paintings (P A : ℕ) (h1: P + A = 8) (h2: P = 1 + 2) : P = 3 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_museum_wings_paintings_l444_44487


namespace NUMINAMATH_GPT_cost_price_of_article_l444_44441

theorem cost_price_of_article :
  ∃ (CP : ℝ), (616 = 1.10 * (1.17 * CP)) → CP = 478.77 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l444_44441


namespace NUMINAMATH_GPT_suff_but_not_nec_l444_44496

-- Definition of proposition p
def p (m : ℝ) : Prop := m = -1

-- Definition of proposition q
def q (m : ℝ) : Prop := 
  let line1 := fun (x y : ℝ) => x - y = 0
  let line2 := fun (x y : ℝ) => x + (m^2) * y = 0
  ∀ (x1 y1 x2 y2 : ℝ), line1 x1 y1 → line2 x2 y2 → (x1 = x2 → y1 = -y2)

-- The proof problem
theorem suff_but_not_nec (m : ℝ) : p m → q m ∧ (q m → m = -1 ∨ m = 1) :=
sorry

end NUMINAMATH_GPT_suff_but_not_nec_l444_44496


namespace NUMINAMATH_GPT_solution_set_of_x_sq_gt_x_l444_44456

theorem solution_set_of_x_sq_gt_x :
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} := 
sorry

end NUMINAMATH_GPT_solution_set_of_x_sq_gt_x_l444_44456


namespace NUMINAMATH_GPT_no_real_solutions_l444_44423

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (4 * x^3 + 3 * x^2 + x + 2) / (x - 2) = 4 * x^2 + 5 :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l444_44423


namespace NUMINAMATH_GPT_trajectory_equation_l444_44497

theorem trajectory_equation (m x y : ℝ) (a b : ℝ × ℝ)
  (ha : a = (m * x, y + 1))
  (hb : b = (x, y - 1))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  m * x^2 + y^2 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_equation_l444_44497


namespace NUMINAMATH_GPT_a_runs_4_times_faster_than_b_l444_44424

theorem a_runs_4_times_faster_than_b (v_A v_B : ℝ) (k : ℝ) 
    (h1 : v_A = k * v_B) 
    (h2 : 92 / v_A = 23 / v_B) : 
    k = 4 := 
sorry

end NUMINAMATH_GPT_a_runs_4_times_faster_than_b_l444_44424


namespace NUMINAMATH_GPT_calculate_distance_l444_44402

def velocity (t : ℝ) : ℝ := 3 * t^2 + t

theorem calculate_distance : ∫ t in (0 : ℝ)..(4 : ℝ), velocity t = 72 := 
by
  sorry

end NUMINAMATH_GPT_calculate_distance_l444_44402


namespace NUMINAMATH_GPT_find_common_ratio_and_difference_l444_44431

theorem find_common_ratio_and_difference (q d : ℤ) 
  (h1 : q^3 = 1 + 7 * d) 
  (h2 : 1 + q + q^2 + q^3 = 1 + 7 * d + 21) : 
  (q = 4 ∧ d = 9) ∨ (q = -5 ∧ d = -18) :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_and_difference_l444_44431


namespace NUMINAMATH_GPT_proof_l444_44454

-- Define the expression
def expr : ℕ :=
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128)

-- Define the conjectured result
def result : ℕ := 5^128 - 4^128

-- Assert their equality
theorem proof : expr = result :=
by
    sorry

end NUMINAMATH_GPT_proof_l444_44454


namespace NUMINAMATH_GPT_total_chairs_calculation_l444_44439

theorem total_chairs_calculation
  (chairs_per_trip : ℕ)
  (trips_per_student : ℕ)
  (total_students : ℕ)
  (h1 : chairs_per_trip = 5)
  (h2 : trips_per_student = 10)
  (h3 : total_students = 5) :
  total_students * (chairs_per_trip * trips_per_student) = 250 :=
by
  sorry

end NUMINAMATH_GPT_total_chairs_calculation_l444_44439


namespace NUMINAMATH_GPT_unique_sequence_l444_44403

theorem unique_sequence (n : ℕ) (h : 1 < n)
  (x : Fin (n-1) → ℕ)
  (h_pos : ∀ i, 0 < x i)
  (h_incr : ∀ i j, i < j → x i < x j)
  (h_symm : ∀ i : Fin (n-1), x i + x ⟨n - 2 - i.val, sorry⟩ = 2 * n)
  (h_sum : ∀ i j : Fin (n-1), x i + x j < 2 * n → ∃ k : Fin (n-1), x i + x j = x k) :
  ∀ i : Fin (n-1), x i = 2 * (i + 1) :=
by
  sorry

end NUMINAMATH_GPT_unique_sequence_l444_44403


namespace NUMINAMATH_GPT_train_usual_time_l444_44416

theorem train_usual_time (T : ℝ) (h1 : T > 0) : 
  (4 / 5 : ℝ) * (T + 1/2) = T :=
by 
  sorry

end NUMINAMATH_GPT_train_usual_time_l444_44416


namespace NUMINAMATH_GPT_ukuleles_and_violins_l444_44452

theorem ukuleles_and_violins (U V : ℕ) : 
  (4 * U + 6 * 4 + 4 * V = 40) → (U + V = 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ukuleles_and_violins_l444_44452


namespace NUMINAMATH_GPT_train_speed_approx_kmph_l444_44490

noncomputable def length_of_train : ℝ := 150
noncomputable def time_to_cross_pole : ℝ := 4.425875438161669

theorem train_speed_approx_kmph :
  (length_of_train / time_to_cross_pole) * 3.6 = 122.03 :=
by sorry

end NUMINAMATH_GPT_train_speed_approx_kmph_l444_44490


namespace NUMINAMATH_GPT_hockey_players_l444_44494

theorem hockey_players (n : ℕ) (h1 : n < 30) (h2 : n % 2 = 0) (h3 : n % 4 = 0) (h4 : n % 7 = 0) :
  (n / 4 = 7) :=
by
  sorry

end NUMINAMATH_GPT_hockey_players_l444_44494


namespace NUMINAMATH_GPT_brenda_spay_cats_l444_44479

theorem brenda_spay_cats (c d : ℕ) (h1 : c + d = 21) (h2 : d = 2 * c) : c = 7 :=
sorry

end NUMINAMATH_GPT_brenda_spay_cats_l444_44479


namespace NUMINAMATH_GPT_tricycle_count_l444_44474

theorem tricycle_count
    (total_children : ℕ) (total_wheels : ℕ) (walking_children : ℕ)
    (h1 : total_children - walking_children = 8)
    (h2 : 2 * (total_children - walking_children - (total_wheels - 16) / 3) + 3 * ((total_wheels - 16) / 3) = total_wheels) :
    (total_wheels - 16) / 3 = 8 :=
by
    intros
    sorry

end NUMINAMATH_GPT_tricycle_count_l444_44474


namespace NUMINAMATH_GPT_describe_difference_of_squares_l444_44477

def description_of_a_squared_minus_b_squared : Prop :=
  ∃ (a b : ℝ), (a^2 - b^2) = (a^2 - b^2)

theorem describe_difference_of_squares :
  description_of_a_squared_minus_b_squared :=
by sorry

end NUMINAMATH_GPT_describe_difference_of_squares_l444_44477


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l444_44408

noncomputable def a_n (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

noncomputable def S_n (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (a1 d : ℕ) (h1 : a_n a1 d 3 = 8) (h2 : S_n a1 d 6 = 54) : d = 2 :=
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l444_44408


namespace NUMINAMATH_GPT_identify_triangle_centers_l444_44470

variable (P : Fin 7 → Type)
variable (I O H L G N K : Type)
variable (P1 P2 P3 P4 P5 P6 P7 : Type)
variable (cond : (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H))

theorem identify_triangle_centers :
  (P 1 = K) ∧ (P 2 = O) ∧ (P 3 = L) ∧ (P 4 = I) ∧ (P 5 = N) ∧ (P 6 = G) ∧ (P 7 = H) :=
by sorry

end NUMINAMATH_GPT_identify_triangle_centers_l444_44470


namespace NUMINAMATH_GPT_range_of_x_l444_44438

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the relevant conditions
axiom decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) < 0
axiom symmetry : ∀ x : ℝ, f (1 - x) = -f (1 + x)
axiom f_one : f 1 = -1

-- Define the statement to be proved
theorem range_of_x : ∀ x : ℝ, -1 ≤ f (0.5 * x - 1) ∧ f (0.5 * x - 1) ≤ 1 → 0 ≤ x ∧ x ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_x_l444_44438


namespace NUMINAMATH_GPT_shooting_accuracy_l444_44499

theorem shooting_accuracy 
  (P_A : ℚ) 
  (P_AB : ℚ) 
  (h1 : P_A = 9 / 10) 
  (h2 : P_AB = 1 / 2) 
  : P_AB / P_A = 5 / 9 := 
by
  sorry

end NUMINAMATH_GPT_shooting_accuracy_l444_44499


namespace NUMINAMATH_GPT_max_sin_x_value_l444_44484

theorem max_sin_x_value (x y z : ℝ) (h1 : Real.sin x = Real.cos y) (h2 : Real.sin y = Real.cos z) (h3 : Real.sin z = Real.cos x) : Real.sin x ≤ Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_sin_x_value_l444_44484


namespace NUMINAMATH_GPT_larger_of_two_numbers_l444_44419

theorem larger_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 8) : max x y = 29 :=
by
  sorry

end NUMINAMATH_GPT_larger_of_two_numbers_l444_44419


namespace NUMINAMATH_GPT_markup_percentage_l444_44449

-- Define the purchase price and the gross profit
def purchase_price : ℝ := 54
def gross_profit : ℝ := 18

-- Define the sale price after discount
def sale_discount : ℝ := 0.8

-- Given that the sale price after the discount is purchase_price + gross_profit
theorem markup_percentage (M : ℝ) (SP : ℝ) : 
  SP = purchase_price * (1 + M / 100) → -- selling price as function of markup
  (SP * sale_discount = purchase_price + gross_profit) → -- sale price after 20% discount
  M = 66.67 := 
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_markup_percentage_l444_44449


namespace NUMINAMATH_GPT_solve_for_m_l444_44458

theorem solve_for_m (x y m : ℝ) 
  (h1 : 2 * x + y = 3 * m) 
  (h2 : x - 4 * y = -2 * m)
  (h3 : y + 2 * m = 1 + x) :
  m = 3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_m_l444_44458


namespace NUMINAMATH_GPT_total_bones_in_graveyard_l444_44417

def total_skeletons : ℕ := 20

def adult_women : ℕ := total_skeletons / 2
def adult_men : ℕ := (total_skeletons - adult_women) / 2
def children : ℕ := (total_skeletons - adult_women) / 2

def bones_adult_woman : ℕ := 20
def bones_adult_man : ℕ := bones_adult_woman + 5
def bones_child : ℕ := bones_adult_woman / 2

def bones_graveyard : ℕ :=
  (adult_women * bones_adult_woman) +
  (adult_men * bones_adult_man) +
  (children * bones_child)

theorem total_bones_in_graveyard :
  bones_graveyard = 375 :=
sorry

end NUMINAMATH_GPT_total_bones_in_graveyard_l444_44417


namespace NUMINAMATH_GPT_loss_percentage_is_nine_percent_l444_44486

theorem loss_percentage_is_nine_percent
    (C S : ℝ)
    (h1 : 15 * C = 20 * S)
    (discount_rate : ℝ := 0.10)
    (tax_rate : ℝ := 0.08) :
    (((0.9 * C) - (1.08 * S)) / C) * 100 = 9 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_is_nine_percent_l444_44486


namespace NUMINAMATH_GPT_total_bottles_l444_44460

theorem total_bottles (n : ℕ) (h1 : ∃ one_third two_third: ℕ, one_third = n / 3 ∧ two_third = 2 * (n / 3) ∧ 3 * one_third = n)
    (h2 : 25 ≤ n)
    (h3 : ∃ damage1 damage2 damage_diff : ℕ, damage1 = 25 * 160 ∧ damage2 = (n / 3) * 160 + ((2 * (n / 3) - 25) * 130) ∧ damage1 - damage2 = 660) :
    n = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_bottles_l444_44460


namespace NUMINAMATH_GPT_number_of_rectangular_arrays_of_chairs_l444_44492

/-- 
Given a classroom that contains 45 chairs, prove that 
the number of rectangular arrays of chairs that can be made such that 
each row contains at least 3 chairs and each column contains at least 3 chairs is 4.
-/
theorem number_of_rectangular_arrays_of_chairs : 
  ∃ (n : ℕ), n = 4 ∧ 
    ∀ (a b : ℕ), (a * b = 45) → 
      (a ≥ 3) → (b ≥ 3) → 
      (n = 4) := 
sorry

end NUMINAMATH_GPT_number_of_rectangular_arrays_of_chairs_l444_44492


namespace NUMINAMATH_GPT_math_problem_solution_l444_44421

theorem math_problem_solution : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^y - 1 = y^x ∧ 2*x^y = y^x + 5 ∧ x = 2 ∧ y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_solution_l444_44421


namespace NUMINAMATH_GPT_product_of_roots_l444_44467

-- Define the coefficients of the cubic equation
def a : ℝ := 2
def d : ℝ := 12

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := a * x^3 - 3 * x^2 - 8 * x + d

-- Prove the product of the roots is -6 using Vieta's formulas
theorem product_of_roots : -d / a = -6 := by
  sorry

end NUMINAMATH_GPT_product_of_roots_l444_44467


namespace NUMINAMATH_GPT_sale_decrease_by_20_percent_l444_44478

theorem sale_decrease_by_20_percent (P Q : ℝ)
  (h1 : P > 0) (h2 : Q > 0)
  (price_increased : ∀ P', P' = 1.30 * P)
  (revenue_increase : ∀ R, R = P * Q → ∀ R', R' = 1.04 * R)
  (new_revenue : ∀ P' Q' R', P' = 1.30 * P → Q' = Q * (1 - x / 100) → R' = P' * Q' → R' = 1.04 * (P * Q)) :
  1 - (20 / 100) = 0.8 :=
by sorry

end NUMINAMATH_GPT_sale_decrease_by_20_percent_l444_44478


namespace NUMINAMATH_GPT_problem_1_part_1_problem_1_part_2_l444_44425

-- Define the function f
def f (x a : ℝ) := |x - a| + 3 * x

-- The first problem statement - Part (Ⅰ)
theorem problem_1_part_1 (x : ℝ) : { x | x ≥ 3 ∨ x ≤ -1 } = { x | f x 1 ≥ 3 * x + 2 } :=
by {
  sorry
}

-- The second problem statement - Part (Ⅱ)
theorem problem_1_part_2 : { x | x ≤ -1 } = { x | f x 2 ≤ 0 } :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_1_part_1_problem_1_part_2_l444_44425


namespace NUMINAMATH_GPT_sin_theta_value_l444_44404

theorem sin_theta_value {θ : ℝ} (h₁ : 9 * (Real.tan θ)^2 = 4 * Real.cos θ) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sin_theta_value_l444_44404


namespace NUMINAMATH_GPT_proof_goal_l444_44488

noncomputable def proof_problem (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a^2 + b^2 + c^2 = 1) : Prop :=
  (1 / a) + (1 / b) + (1 / c) > 4

theorem proof_goal (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a^2 + b^2 + c^2 = 1) : 
  (1 / a) + (1 / b) + (1 / c) > 4 :=
sorry

end NUMINAMATH_GPT_proof_goal_l444_44488


namespace NUMINAMATH_GPT_apples_distribution_count_l444_44476

theorem apples_distribution_count : 
  ∃ (count : ℕ), count = 249 ∧ 
  (∃ (a b c : ℕ), a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3 ∧ a ≤ 20) →
  (a' + 3 + b' + 3 + c' + 3 = 30 ∧ a' + b' + c' = 21) → 
  (∃ (a' b' c' : ℕ), a' + b' + c' = 21 ∧ a' ≤ 17) :=
by
  sorry

end NUMINAMATH_GPT_apples_distribution_count_l444_44476


namespace NUMINAMATH_GPT_false_p_and_q_l444_44463

variable {a : ℝ} 

def p (a : ℝ) := 3 * a / 2 ≤ 1
def q (a : ℝ) := 0 < 2 * a - 1 ∧ 2 * a - 1 < 1

theorem false_p_and_q (a : ℝ) :
  ¬ (p a ∧ q a) ↔ (a ≤ (1 : ℝ) / 2 ∨ a > (2 : ℝ) / 3) :=
by
  sorry

end NUMINAMATH_GPT_false_p_and_q_l444_44463


namespace NUMINAMATH_GPT_quadratic_points_relation_l444_44412

theorem quadratic_points_relation (h y1 y2 y3 : ℝ) :
  (∀ x, x = -1/2 → y1 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 1 → y2 = -(x-2) ^ 2 + h) ∧
  (∀ x, x = 2 → y3 = -(x-2) ^ 2 + h) →
  y1 < y2 ∧ y2 < y3 :=
by
  -- The required proof is omitted
  sorry

end NUMINAMATH_GPT_quadratic_points_relation_l444_44412


namespace NUMINAMATH_GPT_total_animal_eyes_l444_44465

def num_snakes := 18
def num_alligators := 10
def eyes_per_snake := 2
def eyes_per_alligator := 2

theorem total_animal_eyes : 
  (num_snakes * eyes_per_snake) + (num_alligators * eyes_per_alligator) = 56 :=
by 
  sorry

end NUMINAMATH_GPT_total_animal_eyes_l444_44465


namespace NUMINAMATH_GPT_alpha_beta_square_eq_eight_l444_44445

theorem alpha_beta_square_eq_eight (α β : ℝ) 
  (hα : α^2 = 2*α + 1) 
  (hβ : β^2 = 2*β + 1) 
  (h_distinct : α ≠ β) : 
  (α - β)^2 = 8 := 
sorry

end NUMINAMATH_GPT_alpha_beta_square_eq_eight_l444_44445


namespace NUMINAMATH_GPT_shopping_center_expense_l444_44444

theorem shopping_center_expense
    (films_count : ℕ := 9)
    (films_original_price : ℝ := 7)
    (film_discount : ℝ := 2)
    (books_full_price : ℝ := 10)
    (books_count : ℕ := 5)
    (books_discount_rate : ℝ := 0.25)
    (cd_price : ℝ := 4.50)
    (cd_count : ℕ := 6)
    (tax_rate : ℝ := 0.06)
    (total_amount_spent : ℝ := 109.18) :
    let films_total := films_count * (films_original_price - film_discount)
    let remaining_books := books_count - 1
    let discounted_books_total := remaining_books * (books_full_price * (1 - books_discount_rate))
    let books_total := books_full_price + discounted_books_total
    let cds_paid_count := cd_count - (cd_count / 3)
    let cds_total := cds_paid_count * cd_price
    let total_before_tax := films_total + books_total + cds_total
    let tax := total_before_tax * tax_rate
    let total_with_tax := total_before_tax + tax
    total_with_tax = total_amount_spent :=
by
  sorry

end NUMINAMATH_GPT_shopping_center_expense_l444_44444


namespace NUMINAMATH_GPT_any_positive_integer_can_be_expressed_l444_44413

theorem any_positive_integer_can_be_expressed 
  (N : ℕ) (hN : 0 < N) : 
  ∃ (p q u v : ℤ), N = p * q + u * v ∧ (u - v = 2 * (p - q)) := 
sorry

end NUMINAMATH_GPT_any_positive_integer_can_be_expressed_l444_44413


namespace NUMINAMATH_GPT_different_sets_l444_44491

theorem different_sets (a b c : ℤ) (h1 : 0 < a) (h2 : a < c - 1) (h3 : 1 < b) (h4 : b < c)
  (rk : ∀ (k : ℤ), 0 ≤ k ∧ k ≤ a → ∃ (r : ℤ), 0 ≤ r ∧ r < c ∧ k * b % c = r) :
  {r | ∃ k, 0 ≤ k ∧ k ≤ a ∧ r = k * b % c} ≠ {k | 0 ≤ k ∧ k ≤ a} :=
sorry

end NUMINAMATH_GPT_different_sets_l444_44491


namespace NUMINAMATH_GPT_math_problem_l444_44451

theorem math_problem (a b c d e : ℤ) (x : ℤ) (hx : x > 196)
  (h1 : a + b = 183) (h2 : a + c = 186) (h3 : d + e = x) (h4 : c + e = 196)
  (h5 : 183 < 186) (h6 : 186 < 187) (h7 : 187 < 190) (h8 : 190 < 191) (h9 : 191 < 192)
  (h10 : 192 < 193) (h11 : 193 < 194) (h12 : 194 < 196) (h13 : 196 < x) :
  (a = 91 ∧ b = 92 ∧ c = 95 ∧ d = 99 ∧ e = 101 ∧ x = 200) ∧ (∃ y, y = 10 * x + 3 ∧ y = 2003) :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l444_44451
