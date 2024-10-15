import Mathlib

namespace NUMINAMATH_GPT_gcd_of_XY_is_6_l966_96626

theorem gcd_of_XY_is_6 (X Y : ℕ) (h1 : Nat.lcm X Y = 180)
  (h2 : X * 6 = Y * 5) : Nat.gcd X Y = 6 :=
sorry

end NUMINAMATH_GPT_gcd_of_XY_is_6_l966_96626


namespace NUMINAMATH_GPT_problem_solution_l966_96601

theorem problem_solution (a b c : ℝ)
  (h₁ : 10 = (6 / 100) * a)
  (h₂ : 6 = (10 / 100) * b)
  (h₃ : c = b / a) : c = 0.36 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l966_96601


namespace NUMINAMATH_GPT_g_interval_l966_96639

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem g_interval (a b c : ℝ) (ha : 0 < a) (hb: 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
sorry

end NUMINAMATH_GPT_g_interval_l966_96639


namespace NUMINAMATH_GPT_shirt_cost_l966_96693

def cost_of_jeans_and_shirts (J S : ℝ) : Prop := (3 * J + 2 * S = 69) ∧ (2 * J + 3 * S = 81)

theorem shirt_cost (J S : ℝ) (h : cost_of_jeans_and_shirts J S) : S = 21 :=
by {
  sorry
}

end NUMINAMATH_GPT_shirt_cost_l966_96693


namespace NUMINAMATH_GPT_geom_seq_a_sum_first_n_terms_l966_96635

noncomputable def a (n : ℕ) : ℕ := 2^(n + 1)

def b (n : ℕ) : ℕ := 3 * (n + 1) - 2

def a_b_product (n : ℕ) : ℕ := (3 * (n + 1) - 2) * 2^(n + 1)

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => a_b_product k)

theorem geom_seq_a (n : ℕ) : a (n + 1) = 2 * a n :=
by sorry

theorem sum_first_n_terms (n : ℕ) : S n = 10 + (3 * n - 5) * 2^(n + 1) :=
by sorry

end NUMINAMATH_GPT_geom_seq_a_sum_first_n_terms_l966_96635


namespace NUMINAMATH_GPT_price_of_turban_l966_96647

theorem price_of_turban (T : ℝ) (h1 : ∀ (T : ℝ), 3 / 4 * (90 + T) = 40 + T) : T = 110 :=
by
  sorry

end NUMINAMATH_GPT_price_of_turban_l966_96647


namespace NUMINAMATH_GPT_correctFractions_equivalence_l966_96675

def correctFractions: List (ℕ × ℕ) := [(26, 65), (16, 64), (19, 95), (49, 98)]

def isValidCancellation (num den: ℕ): Prop :=
  ∃ n₁ n₂ n₃ d₁ d₂ d₃: ℕ, 
    num = 10 * n₁ + n₂ ∧
    den = 10 * d₁ + d₂ ∧
    ((n₁ = d₁ ∧ n₂ = d₂) ∨ (n₁ = d₃ ∧ n₃ = d₂)) ∧
    n₁ ≠ 0 ∧ n₂ ≠ 0 ∧ d₁ ≠ 0 ∧ d₂ ≠ 0

theorem correctFractions_equivalence : 
  ∀ (frac : ℕ × ℕ), frac ∈ correctFractions → 
    ∃ a b: ℕ, correctFractions = [(a, b)] ∧ 
      isValidCancellation a b := sorry

end NUMINAMATH_GPT_correctFractions_equivalence_l966_96675


namespace NUMINAMATH_GPT_hyperbola_range_m_l966_96683

-- Define the condition that the equation represents a hyperbola
def isHyperbola (m : ℝ) : Prop := (2 + m) * (m + 1) < 0

-- The theorem stating the range of m given the condition
theorem hyperbola_range_m (m : ℝ) : isHyperbola m → -2 < m ∧ m < -1 := by
  sorry

end NUMINAMATH_GPT_hyperbola_range_m_l966_96683


namespace NUMINAMATH_GPT_base9_subtraction_multiple_of_seven_l966_96682

theorem base9_subtraction_multiple_of_seven (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 9) 
(h2 : (3 * 9^6 + 1 * 9^5 + 5 * 9^4 + 4 * 9^3 + 6 * 9^2 + 7 * 9^1 + 2 * 9^0) - b % 7 = 0) : b = 0 :=
sorry

end NUMINAMATH_GPT_base9_subtraction_multiple_of_seven_l966_96682


namespace NUMINAMATH_GPT_legs_per_bee_l966_96649

def number_of_bees : ℕ := 8
def total_legs : ℕ := 48

theorem legs_per_bee : (total_legs / number_of_bees) = 6 := by
  sorry

end NUMINAMATH_GPT_legs_per_bee_l966_96649


namespace NUMINAMATH_GPT_smallest_angle_in_right_triangle_l966_96615

noncomputable def is_consecutive_primes (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ ∀ r, Nat.Prime r → p < r → r < q → False

theorem smallest_angle_in_right_triangle : ∃ p : ℕ, ∃ q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ p + q = 90 ∧ is_consecutive_primes p q ∧ p = 43 :=
by
  sorry

end NUMINAMATH_GPT_smallest_angle_in_right_triangle_l966_96615


namespace NUMINAMATH_GPT_prob_less_than_8_prob_at_least_7_l966_96687

def prob_9_or_above : ℝ := 0.56
def prob_8 : ℝ := 0.22
def prob_7 : ℝ := 0.12

theorem prob_less_than_8 : prob_7 + (1 - prob_9_or_above - prob_8) = 0.22 := 
sorry

theorem prob_at_least_7 : prob_9_or_above + prob_8 + prob_7 = 0.9 := 
sorry

end NUMINAMATH_GPT_prob_less_than_8_prob_at_least_7_l966_96687


namespace NUMINAMATH_GPT_value_of_expression_l966_96604

theorem value_of_expression (a : ℝ) (h : a = 1/2) : 
  (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l966_96604


namespace NUMINAMATH_GPT_remainder_317_l966_96658

theorem remainder_317 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 16])
  (h2 : 6 + y ≡ 8 [ZMOD 81])
  (h3 : 8 + y ≡ 49 [ZMOD 625]) :
  y ≡ 317 [ZMOD 360] := 
sorry

end NUMINAMATH_GPT_remainder_317_l966_96658


namespace NUMINAMATH_GPT_valid_passwords_count_l966_96660

-- Define the total number of unrestricted passwords (each digit can be 0-9)
def total_passwords := 10^5

-- Define the number of restricted passwords (those starting with the sequence 8,3,2)
def restricted_passwords := 10^2

-- State the main theorem to be proved
theorem valid_passwords_count : total_passwords - restricted_passwords = 99900 := by
  sorry

end NUMINAMATH_GPT_valid_passwords_count_l966_96660


namespace NUMINAMATH_GPT_negation_all_dogs_playful_l966_96622

variable {α : Type} (dog playful : α → Prop)

theorem negation_all_dogs_playful :
  (¬ ∀ x, dog x → playful x) ↔ (∃ x, dog x ∧ ¬ playful x) :=
by sorry

end NUMINAMATH_GPT_negation_all_dogs_playful_l966_96622


namespace NUMINAMATH_GPT_alex_has_more_pens_than_jane_l966_96688

-- Definitions based on the conditions
def starting_pens_alex : ℕ := 4
def pens_jane_after_month : ℕ := 16

-- Alex's pen count after each week
def pens_alex_after_week (w : ℕ) : ℕ :=
  starting_pens_alex * 2 ^ w

-- Proof statement
theorem alex_has_more_pens_than_jane :
  pens_alex_after_week 4 - pens_jane_after_month = 16 := by
  sorry

end NUMINAMATH_GPT_alex_has_more_pens_than_jane_l966_96688


namespace NUMINAMATH_GPT_three_digit_integers_sum_to_7_l966_96690

theorem three_digit_integers_sum_to_7 : 
  ∃ n : ℕ, n = 28 ∧ (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ a + b + c = 7) :=
sorry

end NUMINAMATH_GPT_three_digit_integers_sum_to_7_l966_96690


namespace NUMINAMATH_GPT_complement_computation_l966_96637

open Set

theorem complement_computation (U A : Set ℕ) :
  U = {1, 2, 3, 4, 5, 6, 7} → A = {2, 4, 5} →
  U \ A = {1, 3, 6, 7} :=
by
  intros hU hA
  rw [hU, hA]
  ext
  simp
  sorry

end NUMINAMATH_GPT_complement_computation_l966_96637


namespace NUMINAMATH_GPT_temperature_rise_result_l966_96699

def initial_temperature : ℤ := -2
def rise : ℤ := 3

theorem temperature_rise_result : initial_temperature + rise = 1 := 
by 
  sorry

end NUMINAMATH_GPT_temperature_rise_result_l966_96699


namespace NUMINAMATH_GPT_horizontal_length_of_monitor_l966_96618

def monitor_diagonal := 32
def aspect_ratio_horizontal := 16
def aspect_ratio_height := 9

theorem horizontal_length_of_monitor :
  ∃ (horizontal_length : ℝ), horizontal_length = 512 / Real.sqrt 337 := by
  sorry

end NUMINAMATH_GPT_horizontal_length_of_monitor_l966_96618


namespace NUMINAMATH_GPT_at_least_one_worker_must_wait_l966_96621

/-- 
Given five workers who collectively have a salary of 1500 rubles, 
and each tape recorder costs 320 rubles, we need to prove that 
at least one worker will not be able to buy a tape recorder immediately. 
-/
theorem at_least_one_worker_must_wait 
  (num_workers : ℕ) 
  (total_salary : ℕ) 
  (tape_recorder_cost : ℕ) 
  (h_workers : num_workers = 5) 
  (h_salary : total_salary = 1500) 
  (h_cost : tape_recorder_cost = 320) :
  ∀ (tape_recorders_required : ℕ), 
    tape_recorders_required = num_workers → total_salary < tape_recorder_cost * tape_recorders_required → ∃ (k : ℕ), 1 ≤ k ∧ k ≤ num_workers ∧ total_salary < k * tape_recorder_cost :=
by 
  intros tape_recorders_required h_required h_insufficient
  sorry

end NUMINAMATH_GPT_at_least_one_worker_must_wait_l966_96621


namespace NUMINAMATH_GPT_find_y_l966_96698

theorem find_y (a b y : ℝ) (h1 : s = (3 * a) ^ (2 * b)) (h2 : s = 5 * (a ^ b) * (y ^ b))
  (h3 : 0 < a) (h4 : 0 < b) : 
  y = 9 * a / 5 := by
  sorry

end NUMINAMATH_GPT_find_y_l966_96698


namespace NUMINAMATH_GPT_cucumbers_for_20_apples_l966_96666

-- Definitions for all conditions
def apples := ℕ
def bananas := ℕ
def cucumbers := ℕ

def cost_equivalence_apples_bananas (a b : ℕ) : Prop := 10 * a = 5 * b
def cost_equivalence_bananas_cucumbers (b c : ℕ) : Prop := 3 * b = 4 * c

-- Main theorem statement
theorem cucumbers_for_20_apples :
  ∀ (a b c : ℕ),
    cost_equivalence_apples_bananas a b →
    cost_equivalence_bananas_cucumbers b c →
    ∃ k : ℕ, k = 13 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cucumbers_for_20_apples_l966_96666


namespace NUMINAMATH_GPT_find_d_minus_c_l966_96630

variable (c d : ℝ)

def rotate180 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (2 * cx - x, 2 * cy - y)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

def transformations (q : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_eq_x (rotate180 q (2, 3))

theorem find_d_minus_c :
  transformations (c, d) = (1, -4) → d - c = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_d_minus_c_l966_96630


namespace NUMINAMATH_GPT_J_3_3_4_l966_96661

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_3_4 : J 3 (3 / 4) 4 = 259 / 48 := 
by {
    -- We would normally include proof steps here, but according to the instruction, we use 'sorry'.
    sorry
}

end NUMINAMATH_GPT_J_3_3_4_l966_96661


namespace NUMINAMATH_GPT_larger_number_is_50_l966_96617

variable (a b : ℕ)
-- Conditions given in the problem
axiom cond1 : 4 * b = 5 * a
axiom cond2 : b - a = 10

-- The proof statement
theorem larger_number_is_50 : b = 50 :=
sorry

end NUMINAMATH_GPT_larger_number_is_50_l966_96617


namespace NUMINAMATH_GPT_mean_three_numbers_l966_96697

open BigOperators

theorem mean_three_numbers (a b c : ℝ) (s : Finset ℝ) (h₀ : s.card = 20)
  (h₁ : (∑ x in s, x) / 20 = 45) 
  (h₂ : (∑ x in s ∪ {a, b, c}, x) / 23 = 50) : 
  (a + b + c) / 3 = 250 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_three_numbers_l966_96697


namespace NUMINAMATH_GPT_intersection_A_B_l966_96671

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | x < -1 ∨ x > 4}

theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 4 < x ∧ x < 7} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l966_96671


namespace NUMINAMATH_GPT_salary_reduction_l966_96653

theorem salary_reduction (S : ℝ) (R : ℝ) :
  ((S - (R / 100 * S)) * 1.25 = S) → (R = 20) :=
by
  sorry

end NUMINAMATH_GPT_salary_reduction_l966_96653


namespace NUMINAMATH_GPT_highest_mean_possible_l966_96677

def max_arithmetic_mean (g : Matrix (Fin 3) (Fin 3) ℕ) : ℚ := 
  let mean (a b c d : ℕ) : ℚ := (a + b + c + d : ℚ) / 4
  let circles := [
    mean (g 0 0) (g 0 1) (g 1 0) (g 1 1),
    mean (g 0 1) (g 0 2) (g 1 1) (g 1 2),
    mean (g 1 0) (g 1 1) (g 2 0) (g 2 1),
    mean (g 1 1) (g 1 2) (g 2 1) (g 2 2)
  ]
  (circles.sum / 4)

theorem highest_mean_possible :
  ∃ g : Matrix (Fin 3) (Fin 3) ℕ, 
  (∀ i j, 1 ≤ g i j ∧ g i j ≤ 9) ∧ 
  max_arithmetic_mean g = 6.125 :=
by
  sorry

end NUMINAMATH_GPT_highest_mean_possible_l966_96677


namespace NUMINAMATH_GPT_fraction_irreducible_l966_96694

theorem fraction_irreducible (n : ℕ) : Nat.gcd (12 * n + 1) (30 * n + 1) = 1 :=
sorry

end NUMINAMATH_GPT_fraction_irreducible_l966_96694


namespace NUMINAMATH_GPT_calculate_hidden_dots_l966_96670

def sum_faces_of_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice : ℕ := 4
def total_sum_of_dots : ℕ := number_of_dice * sum_faces_of_die

def visible_faces : List (ℕ × String) :=
  [(1, "red"), (1, "none"), (2, "none"), (2, "blue"),
   (3, "none"), (4, "none"), (5, "none"), (6, "none")]

def adjust_face_value (value : ℕ) (color : String) : ℕ :=
  match color with
  | "red" => 2 * value
  | "blue" => 2 * value
  | _ => value

def visible_sum : ℕ :=
  visible_faces.foldl (fun acc (face) => acc + adjust_face_value face.1 face.2) 0

theorem calculate_hidden_dots :
  (total_sum_of_dots - visible_sum) = 57 :=
sorry

end NUMINAMATH_GPT_calculate_hidden_dots_l966_96670


namespace NUMINAMATH_GPT_combined_work_time_l966_96607

def Worker_A_time : ℝ := 10
def Worker_B_time : ℝ := 15

theorem combined_work_time :
  (1 / Worker_A_time + 1 / Worker_B_time)⁻¹ = 6 := by
  sorry

end NUMINAMATH_GPT_combined_work_time_l966_96607


namespace NUMINAMATH_GPT_higher_profit_percentage_l966_96655

theorem higher_profit_percentage (P : ℝ) :
  (P / 100 * 800 = 144) ↔ (P = 18) :=
by
  sorry

end NUMINAMATH_GPT_higher_profit_percentage_l966_96655


namespace NUMINAMATH_GPT_ab_equals_4_l966_96646

theorem ab_equals_4 (a b : ℝ) (h_pos : a > 0 ∧ b > 0)
  (h_area : (1/2) * (12 / a) * (8 / b) = 12) : a * b = 4 :=
by
  sorry

end NUMINAMATH_GPT_ab_equals_4_l966_96646


namespace NUMINAMATH_GPT_actual_average_height_is_correct_l966_96642

-- Definitions based on given conditions
def number_of_students : ℕ := 20
def incorrect_average_height : ℝ := 175.0
def incorrect_height_of_student : ℝ := 151.0
def actual_height_of_student : ℝ := 136.0

-- Prove that the actual average height is 174.25 cm
theorem actual_average_height_is_correct :
  (incorrect_average_height * number_of_students - (incorrect_height_of_student - actual_height_of_student)) / number_of_students = 174.25 :=
sorry

end NUMINAMATH_GPT_actual_average_height_is_correct_l966_96642


namespace NUMINAMATH_GPT_coffee_ratio_is_one_to_five_l966_96643

-- Given conditions
def thermos_capacity : ℕ := 20 -- capacity in ounces
def times_filled_per_day : ℕ := 2
def school_days_per_week : ℕ := 5
def new_weekly_coffee_consumption : ℕ := 40 -- in ounces

-- Definitions based on the conditions
def old_daily_coffee_consumption := thermos_capacity * times_filled_per_day
def old_weekly_coffee_consumption := old_daily_coffee_consumption * school_days_per_week

-- Theorem: The ratio of the new weekly coffee consumption to the old weekly coffee consumption is 1:5
theorem coffee_ratio_is_one_to_five : 
  new_weekly_coffee_consumption / old_weekly_coffee_consumption = 1 / 5 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_coffee_ratio_is_one_to_five_l966_96643


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l966_96692

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 7) :
  ∃ (c : ℝ), (a = b ∧ 7 = c ∨ a = c ∧ 7 = b) ∧ a + b + c = 17 :=
by
  use 17
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l966_96692


namespace NUMINAMATH_GPT_rational_values_of_expressions_l966_96633

theorem rational_values_of_expressions {x : ℚ} :
  (∃ a : ℚ, x / (x^2 + x + 1) = a) → (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) :=
by
  sorry

end NUMINAMATH_GPT_rational_values_of_expressions_l966_96633


namespace NUMINAMATH_GPT_range_of_x_l966_96631

noncomputable def is_valid_x (x : ℝ) : Prop :=
  x ≥ 0 ∧ x ≠ 4

theorem range_of_x (x : ℝ) : 
  is_valid_x x ↔ x ≥ 0 ∧ x ≠ 4 :=
by sorry

end NUMINAMATH_GPT_range_of_x_l966_96631


namespace NUMINAMATH_GPT_find_a_from_conditions_l966_96602

noncomputable def f (x b : ℤ) : ℤ := 4 * x + b

theorem find_a_from_conditions (b a : ℤ) (h1 : a = f (-4) b) (h2 : -4 = f a b) : a = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_from_conditions_l966_96602


namespace NUMINAMATH_GPT_area_of_triangle_COD_l966_96673

theorem area_of_triangle_COD (x p : ℕ) (hx : 0 < x) (hx' : x < 12) (hp : 0 < p) :
  (∃ A : ℚ, A = (x * p : ℚ) / 2) :=
sorry

end NUMINAMATH_GPT_area_of_triangle_COD_l966_96673


namespace NUMINAMATH_GPT_not_monotonic_on_interval_l966_96657

noncomputable def f (x : ℝ) : ℝ := (x^2 / 2) - Real.log x

theorem not_monotonic_on_interval (m : ℝ) : 
  (∃ x y : ℝ, m < x ∧ x < m + 1/2 ∧ m < y ∧ y < m + 1/2 ∧ (x ≠ y) ∧ f x ≠ f y ) ↔ (1/2 < m ∧ m < 1) :=
sorry

end NUMINAMATH_GPT_not_monotonic_on_interval_l966_96657


namespace NUMINAMATH_GPT_trigonometric_identity_solution_l966_96659

open Real

noncomputable def x_sol1 (k : ℤ) : ℝ := (π / 2) * (4 * k - 1)
noncomputable def x_sol2 (l : ℤ) : ℝ := (π / 3) * (6 * l + 1)
noncomputable def x_sol2_neg (l : ℤ) : ℝ := (π / 3) * (6 * l - 1)

theorem trigonometric_identity_solution (x : ℝ) :
    (3 * sin (x / 2) ^ 2 * cos (3 * π / 2 + x / 2) +
    3 * sin (x / 2) ^ 2 * cos (x / 2) -
    sin (x / 2) * cos (x / 2) ^ 2 =
    sin (π / 2 + x / 2) ^ 2 * cos (x / 2)) →
    (∃ k : ℤ, x = x_sol1 k) ∨
    (∃ l : ℤ, x = x_sol2 l ∨ x = x_sol2_neg l) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_solution_l966_96659


namespace NUMINAMATH_GPT_reading_schedule_l966_96603

-- Definitions of reading speeds and conditions
def total_pages := 910
def alice_speed := 30  -- seconds per page
def bob_speed := 60    -- seconds per page
def chandra_speed := 45  -- seconds per page

-- Mathematical problem statement
theorem reading_schedule :
  ∃ (x y : ℕ), 
    (x < y) ∧ 
    (y ≤ total_pages) ∧ 
    (30 * x = 45 * (y - x) ∧ 45 * (y - x) = 60 * (total_pages - y)) ∧ 
    x = 420 ∧ 
    y = 700 :=
  sorry

end NUMINAMATH_GPT_reading_schedule_l966_96603


namespace NUMINAMATH_GPT_smallest_n_common_factor_l966_96679

theorem smallest_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ (∀ d : ℕ, d > 1 → d ∣ (11 * n - 4) → d ∣ (8 * n - 5)) ∧ n = 15 :=
by {
  -- Define the conditions as given in the problem
  sorry
}

end NUMINAMATH_GPT_smallest_n_common_factor_l966_96679


namespace NUMINAMATH_GPT_part1_part2_l966_96628

-- Statement for Part 1
theorem part1 : 
  ∃ (a : Fin 8 → ℕ), 
    (∀ i : Fin 8, 1 ≤ a i ∧ a i ≤ 8) ∧ 
    (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ 
    (∀ i : Fin 8, (a i + a (i + 1) + a (i + 2)) > 11) := sorry

-- Statement for Part 2
theorem part2 : 
  ¬ ∃ (a : Fin 8 → ℕ), 
    (∀ i : Fin 8, 1 ≤ a i ∧ a i ≤ 8) ∧ 
    (∀ i j : Fin 8, i ≠ j → a i ≠ a j) ∧ 
    (∀ i : Fin 8, (a i + a (i + 1) + a (i + 2)) > 13) := sorry

end NUMINAMATH_GPT_part1_part2_l966_96628


namespace NUMINAMATH_GPT_average_weight_of_whole_class_l966_96652

theorem average_weight_of_whole_class (n_a n_b : ℕ) (w_a w_b : ℕ) (avg_w_a avg_w_b : ℕ)
  (h_a : n_a = 36) (h_b : n_b = 24) (h_avg_a : avg_w_a = 30) (h_avg_b : avg_w_b = 30) :
  ((n_a * avg_w_a + n_b * avg_w_b) / (n_a + n_b) = 30) := 
by
  sorry

end NUMINAMATH_GPT_average_weight_of_whole_class_l966_96652


namespace NUMINAMATH_GPT_age_difference_l966_96610

theorem age_difference (a b : ℕ) (ha : a < 10) (hb : b < 10)
  (h1 : 10 * a + b + 10 = 3 * (10 * b + a + 10)) :
  10 * a + b - (10 * b + a) = 54 :=
by sorry

end NUMINAMATH_GPT_age_difference_l966_96610


namespace NUMINAMATH_GPT_number_of_recipes_needed_l966_96672

def numStudents : ℕ := 150
def avgCookiesPerStudent : ℕ := 3
def cookiesPerRecipe : ℕ := 18
def attendanceDrop : ℝ := 0.40

theorem number_of_recipes_needed (n : ℕ) (c : ℕ) (r : ℕ) (d : ℝ) : 
  n = numStudents →
  c = avgCookiesPerStudent →
  r = cookiesPerRecipe →
  d = attendanceDrop →
  ∃ (recipes : ℕ), recipes = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_recipes_needed_l966_96672


namespace NUMINAMATH_GPT_hemisphere_surface_area_l966_96620

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 225 * π) : 2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end NUMINAMATH_GPT_hemisphere_surface_area_l966_96620


namespace NUMINAMATH_GPT_optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l966_96616
-- Import necessary libraries

-- Define each of the conditions as Lean definitions
def OptionA (a b c : ℝ) : Prop := a = 1.5 ∧ b = 2 ∧ c = 3
def OptionB (a b c : ℝ) : Prop := a = 7 ∧ b = 24 ∧ c = 25
def OptionC (a b c : ℝ) : Prop := ∃ k : ℕ, a = (3 : ℝ)*k ∧ b = (4 : ℝ)*k ∧ c = (5 : ℝ)*k
def OptionD (a b c : ℝ) : Prop := a = 9 ∧ b = 12 ∧ c = 15

-- Define the Pythagorean theorem predicate
def Pythagorean (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- State the theorem to prove Option A cannot form a right triangle
theorem optionA_not_right_triangle : ¬ Pythagorean 1.5 2 3 := by sorry

-- State the remaining options can form a right triangle
theorem optionB_right_triangle : Pythagorean 7 24 25 := by sorry
theorem optionC_right_triangle (k : ℕ) : Pythagorean (3 * k) (4 * k) (5 * k) := by sorry
theorem optionD_right_triangle : Pythagorean 9 12 15 := by sorry

end NUMINAMATH_GPT_optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l966_96616


namespace NUMINAMATH_GPT_find_y_l966_96638

theorem find_y (x y z : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 8) (h4 : z = 2) : y = 3 :=
    sorry

end NUMINAMATH_GPT_find_y_l966_96638


namespace NUMINAMATH_GPT_number_of_space_diagonals_l966_96668

theorem number_of_space_diagonals
  (V E F T Q : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 42)
  (hT : T = 30)
  (hQ : Q = 12):
  (V * (V - 1) / 2 - E - 2 * Q) = 341 :=
by
  sorry

end NUMINAMATH_GPT_number_of_space_diagonals_l966_96668


namespace NUMINAMATH_GPT_matching_pair_probability_l966_96645

theorem matching_pair_probability :
  let gray_socks := 12
  let white_socks := 10
  let black_socks := 6
  let total_socks := gray_socks + white_socks + black_socks
  let total_ways := total_socks.choose 2
  let gray_matching := gray_socks.choose 2
  let white_matching := white_socks.choose 2
  let black_matching := black_socks.choose 2
  let matching_ways := gray_matching + white_matching + black_matching
  let probability := matching_ways / total_ways
  probability = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_matching_pair_probability_l966_96645


namespace NUMINAMATH_GPT_increase_average_l966_96678

variable (total_runs : ℕ) (innings : ℕ) (average : ℕ) (new_runs : ℕ) (x : ℕ)

theorem increase_average (h1 : innings = 10) 
                         (h2 : average = 30) 
                         (h3 : total_runs = average * innings) 
                         (h4 : new_runs = 74) 
                         (h5 : total_runs + new_runs = (average + x) * (innings + 1)) :
    x = 4 := 
sorry

end NUMINAMATH_GPT_increase_average_l966_96678


namespace NUMINAMATH_GPT_purchase_price_eq_360_l966_96656

theorem purchase_price_eq_360 (P : ℝ) (M : ℝ) (H1 : M = 30) (H2 : M = 0.05 * P + 12) : P = 360 :=
by
  sorry

end NUMINAMATH_GPT_purchase_price_eq_360_l966_96656


namespace NUMINAMATH_GPT_largest_integer_y_l966_96664

theorem largest_integer_y (y : ℤ) : (y / 4 + 3 / 7 : ℝ) < 9 / 4 → y ≤ 7 := by
  intros h
  sorry -- Proof needed

end NUMINAMATH_GPT_largest_integer_y_l966_96664


namespace NUMINAMATH_GPT_number_of_B_students_l966_96644

-- Conditions
def prob_A (prob_B : ℝ) := 0.6 * prob_B
def prob_C (prob_B : ℝ) := 1.6 * prob_B
def prob_D (prob_B : ℝ) := 0.3 * prob_B

-- Total students
def total_students : ℝ := 50

-- Main theorem statement
theorem number_of_B_students (x : ℝ) (h1 : prob_A x + x + prob_C x + prob_D x = total_students) :
  x = 14 :=
  by
-- Proof skipped
  sorry

end NUMINAMATH_GPT_number_of_B_students_l966_96644


namespace NUMINAMATH_GPT_third_side_length_l966_96686

noncomputable def calc_third_side (a b : ℕ) (hypotenuse : Bool) : ℝ :=
if hypotenuse then
  Real.sqrt (a^2 + b^2)
else
  Real.sqrt (abs (a^2 - b^2))

theorem third_side_length (a b : ℕ) (h_right_triangle : (a = 8 ∧ b = 15)) :
  calc_third_side a b true = 17 ∨ calc_third_side 15 8 false = Real.sqrt 161 :=
by {
  sorry
}

end NUMINAMATH_GPT_third_side_length_l966_96686


namespace NUMINAMATH_GPT_quadratic_real_root_iff_b_range_l966_96606

open Real

theorem quadratic_real_root_iff_b_range (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_root_iff_b_range_l966_96606


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l966_96640

-- Define the condition on a
def condition (a : ℝ) : Prop := a > 0

-- Define the quadratic inequality
def quadratic_inequality (a : ℝ) : Prop := a^2 + a ≥ 0

-- The proof statement that "a > 0" is a sufficient but not necessary condition for "a^2 + a ≥ 0"
theorem sufficient_not_necessary_condition (a : ℝ) : condition a → quadratic_inequality a :=
by
    intro ha
    -- [The remaining part of the proof is skipped.]
    sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l966_96640


namespace NUMINAMATH_GPT_power_function_properties_l966_96612

def power_function (f : ℝ → ℝ) (x : ℝ) (a : ℝ) : Prop :=
  f x = x ^ a

theorem power_function_properties :
  ∃ (f : ℝ → ℝ) (a : ℝ), power_function f 2 a ∧ f 2 = 1/2 ∧ 
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → 0 < x2 → 
    (f x1 + f x2) / 2 > f ((x1 + x2) / 2)) :=
sorry

end NUMINAMATH_GPT_power_function_properties_l966_96612


namespace NUMINAMATH_GPT_binary_to_decimal_101101_l966_96600

def binary_to_decimal (digits : List ℕ) : ℕ :=
  digits.foldr (λ (digit : ℕ) (acc : ℕ × ℕ) => (acc.1 + digit * 2 ^ acc.2, acc.2 + 1)) (0, 0) |>.1

theorem binary_to_decimal_101101 : binary_to_decimal [1, 0, 1, 1, 0, 1] = 45 :=
by
  -- Proof is needed but here we use sorry as placeholder.
  sorry

end NUMINAMATH_GPT_binary_to_decimal_101101_l966_96600


namespace NUMINAMATH_GPT_Yura_catches_up_in_five_minutes_l966_96627

-- Define the speeds and distances
variables (v_Lena v_Yura d_Lena d_Yura : ℝ)
-- Assume v_Yura = 2 * v_Lena (Yura is twice as fast)
axiom h1 : v_Yura = 2 * v_Lena 
-- Assume Lena walks for 5 minutes before Yura starts
axiom h2 : d_Lena = v_Lena * 5
-- Assume they walk at constant speeds
noncomputable def t_to_catch_up := 10 / 2 -- time Yura takes to catch up Lena

-- Define the proof problem
theorem Yura_catches_up_in_five_minutes :
    t_to_catch_up = 5 :=
by
    sorry

end NUMINAMATH_GPT_Yura_catches_up_in_five_minutes_l966_96627


namespace NUMINAMATH_GPT_arrangement_valid_l966_96665

def unique_digits (a b c d e f : Nat) : Prop :=
  (a = 4) ∧ (b = 1) ∧ (c = 2) ∧ (d = 5) ∧ (e = 6) ∧ (f = 3)

def sum_15 (x y z : Nat) : Prop :=
  x + y + z = 15

theorem arrangement_valid :
  ∃ a b c d e f : Nat, unique_digits a b c d e f ∧
  sum_15 a d e ∧
  sum_15 d b f ∧
  sum_15 f e c ∧
  sum_15 a b c ∧
  sum_15 a e f ∧
  sum_15 b d c :=
sorry

end NUMINAMATH_GPT_arrangement_valid_l966_96665


namespace NUMINAMATH_GPT_product_of_repeating_decimal_l966_96632

theorem product_of_repeating_decimal 
  (t : ℚ) 
  (h : t = 456 / 999) : 
  8 * t = 1216 / 333 :=
by
  sorry

end NUMINAMATH_GPT_product_of_repeating_decimal_l966_96632


namespace NUMINAMATH_GPT_coordinates_of_P_l966_96667

variable (a : ℝ)

def y_coord (a : ℝ) : ℝ :=
  3 * a + 9

def x_coord (a : ℝ) : ℝ :=
  4 - a

theorem coordinates_of_P :
  (∃ a : ℝ, y_coord a = 0) → ∃ a : ℝ, (x_coord a, y_coord a) = (7, 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l966_96667


namespace NUMINAMATH_GPT_rate_of_current_l966_96609

theorem rate_of_current : 
  ∀ (v c : ℝ), v = 3.3 → (∀ d: ℝ, d > 0 → (d / (v - c) = 2 * (d / (v + c))) → c = 1.1) :=
by
  intros v c hv h
  sorry

end NUMINAMATH_GPT_rate_of_current_l966_96609


namespace NUMINAMATH_GPT_triangle_side_length_c_l966_96654

theorem triangle_side_length_c
  (a b A B C : ℝ)
  (ha : a = Real.sqrt 3)
  (hb : b = 1)
  (hA : A = 2 * B)
  (hAngleSum : A + B + C = Real.pi) :
  ∃ c : ℝ, c = 2 := 
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_c_l966_96654


namespace NUMINAMATH_GPT_sasha_quarters_max_l966_96636

/-- Sasha has \$4.80 in U.S. coins. She has four times as many dimes as she has nickels 
and the same number of quarters as nickels. Prove that the greatest number 
of quarters she could have is 6. -/
theorem sasha_quarters_max (q n d : ℝ) (h1 : 0.25 * q + 0.05 * n + 0.1 * d = 4.80)
  (h2 : n = q) (h3 : d = 4 * n) : q = 6 := 
sorry

end NUMINAMATH_GPT_sasha_quarters_max_l966_96636


namespace NUMINAMATH_GPT_statement_a_statement_b_statement_c_l966_96681

theorem statement_a (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  0 ≤ a ∧ a ≤ 4 := sorry

theorem statement_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -1 ≤ b ∧ b ≤ 3 := sorry

theorem statement_c (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -2 ≤ 3 * a - 2 * b ∧ 3 * a - 2 * b ≤ 10 := sorry

end NUMINAMATH_GPT_statement_a_statement_b_statement_c_l966_96681


namespace NUMINAMATH_GPT_proof_problem_l966_96674

variables (Books : Type) (Available : Books -> Prop)

def all_books_available : Prop := ∀ b : Books, Available b
def some_books_not_available : Prop := ∃ b : Books, ¬ Available b
def not_all_books_available : Prop := ¬ all_books_available Books Available

theorem proof_problem (h : ¬ all_books_available Books Available) : 
  some_books_not_available Books Available ∧ not_all_books_available Books Available :=
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l966_96674


namespace NUMINAMATH_GPT_vector_at_t5_l966_96641

theorem vector_at_t5 (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) 
  (h1 : (a, b) = (2, 5)) 
  (h2 : (a + 3 * c, b + 3 * d) = (8, -7)) :
  (a + 5 * c, b + 5 * d) = (10, -11) :=
by
  sorry

end NUMINAMATH_GPT_vector_at_t5_l966_96641


namespace NUMINAMATH_GPT_total_blue_balloons_l966_96669

def joan_blue_balloons : ℕ := 60
def melanie_blue_balloons : ℕ := 85
def alex_blue_balloons : ℕ := 37
def gary_blue_balloons : ℕ := 48

theorem total_blue_balloons :
  joan_blue_balloons + melanie_blue_balloons + alex_blue_balloons + gary_blue_balloons = 230 :=
by simp [joan_blue_balloons, melanie_blue_balloons, alex_blue_balloons, gary_blue_balloons]

end NUMINAMATH_GPT_total_blue_balloons_l966_96669


namespace NUMINAMATH_GPT_math_question_l966_96676

def set_medians_equal (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x3 + x4) / 2 = (x3 + x4) / 2

def set_ranges_inequality (x1 x2 x3 x4 x5 x6 : ℝ) : Prop :=
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  (x6 - x1) ≥ (x5 - x2)

theorem math_question (x1 x2 x3 x4 x5 x6 : ℝ) :
  (x1 < x2) ∧ (x2 < x3) ∧ (x3 < x4) ∧ (x4 < x5) ∧ (x5 < x6) →
  set_medians_equal x1 x2 x3 x4 x5 x6 ∧
  set_ranges_inequality x1 x2 x3 x4 x5 x6 :=
by
  sorry

end NUMINAMATH_GPT_math_question_l966_96676


namespace NUMINAMATH_GPT_doris_needs_weeks_l966_96689

noncomputable def average_weeks_to_cover_expenses (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) : ℝ := 
  let weekday_income := weekly_babysit_hours * 20
  let saturday_income := saturday_hours * (if weekly_babysit_hours > 15 then 15 else 20)
  let teaching_income := 100
  let total_weekly_income := weekday_income + saturday_income + teaching_income
  let monthly_income_before_tax := total_weekly_income * 4
  let monthly_income_after_tax := monthly_income_before_tax * 0.85
  monthly_income_after_tax / 4 / 1200

theorem doris_needs_weeks (weekly_babysit_hours: ℝ) (saturday_hours: ℝ) :
  1200 ≤ (average_weeks_to_cover_expenses weekly_babysit_hours saturday_hours) * 4 * 1200 :=
  by
    sorry

end NUMINAMATH_GPT_doris_needs_weeks_l966_96689


namespace NUMINAMATH_GPT_monomial_completes_square_l966_96648

variable (x : ℝ)

theorem monomial_completes_square :
  ∃ (m : ℝ), ∀ (x : ℝ), ∃ (a b : ℝ), (16 * x^2 + 1 + m) = (a * x + b)^2 :=
sorry

end NUMINAMATH_GPT_monomial_completes_square_l966_96648


namespace NUMINAMATH_GPT_angie_pretzels_l966_96684

theorem angie_pretzels (Barry_Shelly: ℕ) (Shelly_Angie: ℕ) :
  (Barry_Shelly = 12 / 2) → (Shelly_Angie = 3 * Barry_Shelly) → (Barry_Shelly = 6) → (Shelly_Angie = 18) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_angie_pretzels_l966_96684


namespace NUMINAMATH_GPT_carpet_length_l966_96680

-- Define the conditions as hypotheses
def width_of_carpet : ℝ := 4
def area_of_living_room : ℝ := 60

-- Formalize the corresponding proof problem
theorem carpet_length (h : 60 = width_of_carpet * length) : length = 15 :=
sorry

end NUMINAMATH_GPT_carpet_length_l966_96680


namespace NUMINAMATH_GPT_smaller_triangle_perimeter_l966_96613

theorem smaller_triangle_perimeter (p : ℕ) (p1 : ℕ) (p2 : ℕ) (p3 : ℕ) 
  (h₀ : p = 11)
  (h₁ : p1 = 5)
  (h₂ : p2 = 7)
  (h₃ : p3 = 9) : 
  p1 + p2 + p3 - p = 10 := by
  sorry

end NUMINAMATH_GPT_smaller_triangle_perimeter_l966_96613


namespace NUMINAMATH_GPT_sequence_bounds_l966_96629

theorem sequence_bounds (n : ℕ) (hpos : 0 < n) :
  ∃ (a : ℕ → ℝ), (a 0 = 1/2) ∧
  (∀ k < n, a (k + 1) = a k + (1/n) * (a k)^2) ∧
  (1 - 1 / n < a n ∧ a n < 1) :=
sorry

end NUMINAMATH_GPT_sequence_bounds_l966_96629


namespace NUMINAMATH_GPT_Sara_spent_on_each_movie_ticket_l966_96650

def Sara_spent_on_each_movie_ticket_correct : Prop :=
  let T := 36.78
  let R := 1.59
  let B := 13.95
  (T - R - B) / 2 = 10.62

theorem Sara_spent_on_each_movie_ticket : 
  Sara_spent_on_each_movie_ticket_correct :=
by
  sorry

end NUMINAMATH_GPT_Sara_spent_on_each_movie_ticket_l966_96650


namespace NUMINAMATH_GPT_reading_homework_pages_eq_three_l966_96696

-- Define the conditions
def pages_of_math_homework : ℕ := 7
def difference : ℕ := 4

-- Define what we need to prove
theorem reading_homework_pages_eq_three (x : ℕ) (h : x + difference = pages_of_math_homework) : x = 3 := by
  sorry

end NUMINAMATH_GPT_reading_homework_pages_eq_three_l966_96696


namespace NUMINAMATH_GPT_second_cart_travel_distance_l966_96691

-- Given definitions:
def first_cart_first_term : ℕ := 6
def first_cart_common_difference : ℕ := 8
def second_cart_first_term : ℕ := 7
def second_cart_common_difference : ℕ := 9

-- Given times:
def time_first_cart : ℕ := 35
def time_second_cart : ℕ := 33

-- Arithmetic series sum formula
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

-- Total distance traveled by the second cart
noncomputable def distance_second_cart : ℕ :=
  arithmetic_series_sum second_cart_first_term second_cart_common_difference time_second_cart

-- Theorem to prove the distance traveled by the second cart
theorem second_cart_travel_distance : distance_second_cart = 4983 :=
  sorry

end NUMINAMATH_GPT_second_cart_travel_distance_l966_96691


namespace NUMINAMATH_GPT_compute_complex_power_l966_96662

theorem compute_complex_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 :=
by
  sorry

end NUMINAMATH_GPT_compute_complex_power_l966_96662


namespace NUMINAMATH_GPT_smallest_solution_l966_96608

theorem smallest_solution :
  (∃ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  ∀ y : ℝ, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y) →
  x = 4 - Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_smallest_solution_l966_96608


namespace NUMINAMATH_GPT_average_weight_decrease_l966_96623

theorem average_weight_decrease 
  (A1 : ℝ) (new_person_weight : ℝ) (num_initial : ℕ) (num_total : ℕ) 
  (hA1 : A1 = 55) (hnew_person_weight : new_person_weight = 50) 
  (hnum_initial : num_initial = 20) (hnum_total : num_total = 21) :
  A1 - ((A1 * num_initial + new_person_weight) / num_total) = 0.24 :=
by
  rw [hA1, hnew_person_weight, hnum_initial, hnum_total]
  -- Further proof steps would go here
  sorry

end NUMINAMATH_GPT_average_weight_decrease_l966_96623


namespace NUMINAMATH_GPT_madeline_refills_l966_96651

theorem madeline_refills :
  let total_water := 100
  let bottle_capacity := 12
  let remaining_to_drink := 16
  let already_drank := total_water - remaining_to_drink
  let initial_refills := already_drank / bottle_capacity
  let refills := initial_refills + 1
  refills = 8 :=
by
  sorry

end NUMINAMATH_GPT_madeline_refills_l966_96651


namespace NUMINAMATH_GPT_minimum_bailing_rate_l966_96614

-- Conditions
def distance_to_shore : ℝ := 2 -- miles
def rowing_speed : ℝ := 3 -- miles per hour
def water_intake_rate : ℝ := 15 -- gallons per minute
def max_water_capacity : ℝ := 50 -- gallons

-- Result to prove
theorem minimum_bailing_rate (r : ℝ) : 
  (distance_to_shore / rowing_speed * 60 * water_intake_rate - distance_to_shore / rowing_speed * 60 * r) ≤ max_water_capacity →
  r ≥ 13.75 :=
by
  sorry

end NUMINAMATH_GPT_minimum_bailing_rate_l966_96614


namespace NUMINAMATH_GPT_last_two_digits_of_17_pow_17_l966_96611

theorem last_two_digits_of_17_pow_17 : (17 ^ 17) % 100 = 77 := 
by sorry

end NUMINAMATH_GPT_last_two_digits_of_17_pow_17_l966_96611


namespace NUMINAMATH_GPT_journey_speed_condition_l966_96634

theorem journey_speed_condition (v : ℝ) :
  (10 : ℝ) = 112 / v + 112 / 24 → (224 / 2 = 112) → v = 21 := by
  intros
  apply sorry

end NUMINAMATH_GPT_journey_speed_condition_l966_96634


namespace NUMINAMATH_GPT_find_a_plus_b_l966_96605

theorem find_a_plus_b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) :
  a + b = 82 / 7 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l966_96605


namespace NUMINAMATH_GPT_server_multiplications_in_half_hour_l966_96663

theorem server_multiplications_in_half_hour : 
  let rate := 5000
  let seconds_in_half_hour := 1800
  rate * seconds_in_half_hour = 9000000 := by
  sorry

end NUMINAMATH_GPT_server_multiplications_in_half_hour_l966_96663


namespace NUMINAMATH_GPT_ratio_of_side_length_to_radius_l966_96685

theorem ratio_of_side_length_to_radius (r s : ℝ) (c d : ℝ) 
  (h1 : s = 2 * r)
  (h2 : s^2 = (c / d) * (s^2 - π * r^2)) : 
  (s / r) = (Real.sqrt (c * π) / Real.sqrt (d - c)) := by
  sorry

end NUMINAMATH_GPT_ratio_of_side_length_to_radius_l966_96685


namespace NUMINAMATH_GPT_probability_first_queen_second_diamond_l966_96625

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_first_queen_second_diamond_l966_96625


namespace NUMINAMATH_GPT_children_multiple_of_four_l966_96695

theorem children_multiple_of_four (C : ℕ) 
  (h_event : ∃ (A : ℕ) (T : ℕ), A = 12 ∧ T = 4 ∧ 12 % T = 0 ∧ C % T = 0) : ∃ k : ℕ, C = 4 * k :=
by
  obtain ⟨A, T, hA, hT, hA_div, hC_div⟩ := h_event
  rw [hA, hT] at *
  sorry

end NUMINAMATH_GPT_children_multiple_of_four_l966_96695


namespace NUMINAMATH_GPT_probability_of_winning_exactly_once_l966_96619

-- Define the probability of player A winning a match
def prob_win_A (p : ℝ) : Prop := (1 - p) ^ 3 = 1 - 63 / 64

-- Define the binomial probability for exactly one win in three matches
def binomial_prob (p : ℝ) : ℝ := 3 * p * (1 - p) ^ 2

theorem probability_of_winning_exactly_once (p : ℝ) (h : prob_win_A p) : binomial_prob p = 9 / 64 :=
sorry

end NUMINAMATH_GPT_probability_of_winning_exactly_once_l966_96619


namespace NUMINAMATH_GPT_abigail_collected_43_l966_96624

noncomputable def cans_needed : ℕ := 100
noncomputable def collected_by_alyssa : ℕ := 30
noncomputable def more_to_collect : ℕ := 27
noncomputable def collected_by_abigail : ℕ := cans_needed - (collected_by_alyssa + more_to_collect)

theorem abigail_collected_43 : collected_by_abigail = 43 := by
  sorry

end NUMINAMATH_GPT_abigail_collected_43_l966_96624
