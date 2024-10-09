import Mathlib

namespace consumption_increase_percentage_l485_48575

theorem consumption_increase_percentage (T C : ℝ) (T_pos : 0 < T) (C_pos : 0 < C) :
  (0.7 * (1 + x / 100) * T * C = 0.84 * T * C) → x = 20 :=
by sorry

end consumption_increase_percentage_l485_48575


namespace more_blue_blocks_than_red_l485_48518

theorem more_blue_blocks_than_red 
  (red_blocks : ℕ) 
  (yellow_blocks : ℕ) 
  (blue_blocks : ℕ) 
  (total_blocks : ℕ) 
  (h_red : red_blocks = 18) 
  (h_yellow : yellow_blocks = red_blocks + 7) 
  (h_total : total_blocks = red_blocks + yellow_blocks + blue_blocks) 
  (h_total_given : total_blocks = 75) :
  blue_blocks - red_blocks = 14 :=
by sorry

end more_blue_blocks_than_red_l485_48518


namespace zane_total_payment_l485_48599

open Real

noncomputable def shirt1_price := 50.0
noncomputable def shirt2_price := 50.0
noncomputable def discount1 := 0.4 * shirt1_price
noncomputable def discount2 := 0.3 * shirt2_price
noncomputable def price1_after_discount := shirt1_price - discount1
noncomputable def price2_after_discount := shirt2_price - discount2
noncomputable def total_before_tax := price1_after_discount + price2_after_discount
noncomputable def sales_tax := 0.08 * total_before_tax
noncomputable def total_cost := total_before_tax + sales_tax

-- We want to prove:
theorem zane_total_payment : total_cost = 70.20 := by sorry

end zane_total_payment_l485_48599


namespace points_on_opposite_sides_of_line_l485_48559

theorem points_on_opposite_sides_of_line (a : ℝ) :
  let A := (3, 1)
  let B := (-4, 6)
  (3 * A.1 - 2 * A.2 + a) * (3 * B.1 - 2 * B.2 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  let A := (3, 1)
  let B := (-4, 6)
  have hA : 3 * A.1 - 2 * A.2 + a = 7 + a := by sorry
  have hB : 3 * B.1 - 2 * B.2 + a = -24 + a := by sorry
  exact sorry

end points_on_opposite_sides_of_line_l485_48559


namespace sallys_dad_nickels_l485_48557

theorem sallys_dad_nickels :
  ∀ (initial_nickels mother's_nickels total_nickels nickels_from_dad : ℕ), 
    initial_nickels = 7 → 
    mother's_nickels = 2 →
    total_nickels = 18 →
    total_nickels = initial_nickels + mother's_nickels + nickels_from_dad →
    nickels_from_dad = 9 :=
by
  intros initial_nickels mother's_nickels total_nickels nickels_from_dad
  intros h1 h2 h3 h4
  sorry

end sallys_dad_nickels_l485_48557


namespace number_of_girls_l485_48504

variable (G : ℕ) -- Number of girls in the school
axiom boys_count : G + 807 = 841 -- Given condition

theorem number_of_girls : G = 34 :=
by
  sorry

end number_of_girls_l485_48504


namespace product_three_consecutive_integers_divisible_by_six_l485_48567

theorem product_three_consecutive_integers_divisible_by_six
  (n : ℕ) (h_pos : 0 < n) : ∃ k : ℕ, (n - 1) * n * (n + 1) = 6 * k :=
by sorry

end product_three_consecutive_integers_divisible_by_six_l485_48567


namespace stratified_sampling_A_l485_48595

theorem stratified_sampling_A (A B C total_units : ℕ) (propA : A = 400) (propB : B = 300) (propC : C = 200) (units : total_units = 90) :
  let total_families := A + B + C
  let nA := (A * total_units) / total_families
  nA = 40 :=
by
  -- prove the theorem here
  sorry

end stratified_sampling_A_l485_48595


namespace abs_ineq_subs_ineq_l485_48565

-- Problem 1
theorem abs_ineq (x : ℝ) : -2 ≤ x ∧ x ≤ 2 ↔ |x - 1| + |x + 1| ≤ 4 := 
sorry

-- Problem 2
theorem subs_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a + b + c := 
sorry

end abs_ineq_subs_ineq_l485_48565


namespace smallest_integer_solution_l485_48578

theorem smallest_integer_solution (x : ℤ) : 
  (∃ y : ℤ, (y > 20 / 21 ∧ (y = ↑x ∧ (x = 1)))) → (x = 1) :=
by
  sorry

end smallest_integer_solution_l485_48578


namespace inscribed_squares_ratio_l485_48530

theorem inscribed_squares_ratio (a b : ℝ) (h_triangle : 5^2 + 12^2 = 13^2)
    (h_square1 : a = 25 / 37) (h_square2 : b = 10) :
    a / b = 25 / 370 :=
by 
  sorry

end inscribed_squares_ratio_l485_48530


namespace melons_count_l485_48558

theorem melons_count (w_apples_total w_apple w_2apples w_watermelons w_total w_melons : ℕ) :
  w_apples_total = 4500 →
  9 * w_apple = w_apples_total →
  2 * w_apple = w_2apples →
  5 * 1050 = w_watermelons →
  w_total = w_2apples + w_melons →
  w_total = w_watermelons →
  w_melons / 850 = 5 :=
by
  sorry

end melons_count_l485_48558


namespace speed_ratio_l485_48535

theorem speed_ratio (L tA tB : ℝ) (R : ℝ) (h1: A_speed = R * B_speed) 
  (h2: head_start = 0.35 * L) (h3: finish_margin = 0.25 * L)
  (h4: A_distance = L + head_start) (h5: B_distance = L)
  (h6: A_finish = A_distance / A_speed)
  (h7: B_finish = B_distance / B_speed)
  (h8: B_finish_time = A_finish + finish_margin / B_speed)
  : R = 1.08 :=
by
  sorry

end speed_ratio_l485_48535


namespace linear_function_product_neg_l485_48566

theorem linear_function_product_neg (a1 b1 a2 b2 : ℝ) (hP : b1 = -3 * a1 + 4) (hQ : b2 = -3 * a2 + 4) :
  (a1 - a2) * (b1 - b2) < 0 :=
by
  sorry

end linear_function_product_neg_l485_48566


namespace inequality_square_l485_48584

theorem inequality_square (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end inequality_square_l485_48584


namespace ratio_345_iff_arithmetic_sequence_l485_48540

-- Define the variables and the context
variables (a b c : ℕ) -- assuming non-negative integers for simplicity
variable (k : ℕ) -- scaling factor for the 3:4:5 ratio
variable (d : ℕ) -- common difference in the arithmetic sequence

-- Conditions given
def isRightAngledTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧ a < b ∧ b < c

def is345Ratio (a b c : ℕ) : Prop :=
  ∃ k, a = 3 * k ∧ b = 4 * k ∧ c = 5 * k

def formsArithmeticSequence (a b c : ℕ) : Prop :=
  ∃ d, b = a + d ∧ c = b + d 

-- The statement to prove: sufficiency and necessity
theorem ratio_345_iff_arithmetic_sequence 
  (h_triangle : isRightAngledTriangle a b c) :
  (is345Ratio a b c ↔ formsArithmeticSequence a b c) :=
sorry

end ratio_345_iff_arithmetic_sequence_l485_48540


namespace remaining_files_calc_l485_48524

-- Definitions based on given conditions
def music_files : ℕ := 27
def video_files : ℕ := 42
def deleted_files : ℕ := 11

-- Theorem statement to prove the number of remaining files
theorem remaining_files_calc : music_files + video_files - deleted_files = 58 := by
  sorry

end remaining_files_calc_l485_48524


namespace div_equal_octagons_l485_48588

-- Definitions based on the conditions
def squareArea (n : ℕ) := n * n
def isDivisor (m n : ℕ) := n % m = 0

-- Main statement
theorem div_equal_octagons (n : ℕ) (hn : n = 8) :
  (2 ∣ squareArea n) ∨ (4 ∣ squareArea n) ∨ (8 ∣ squareArea n) ∨ (16 ∣ squareArea n) :=
by
  -- We shall show the divisibility aspect later.
  sorry

end div_equal_octagons_l485_48588


namespace fisherman_daily_earnings_l485_48569

theorem fisherman_daily_earnings :
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 :=
by
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  show red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52
  sorry

end fisherman_daily_earnings_l485_48569


namespace probability_at_least_3_l485_48555

noncomputable def probability_hitting_at_least_3_of_4 (p : ℝ) (n : ℕ) : ℝ :=
  let p3 := (Nat.choose n 3) * (p^3) * ((1 - p)^(n - 3))
  let p4 := (Nat.choose n 4) * (p^4)
  p3 + p4

theorem probability_at_least_3 (h : probability_hitting_at_least_3_of_4 0.8 4 = 0.8192) : 
   True :=
by trivial

end probability_at_least_3_l485_48555


namespace polynomial_has_real_root_l485_48549

theorem polynomial_has_real_root (a b : ℝ) :
  ∃ x : ℝ, x^3 + a * x + b = 0 :=
sorry

end polynomial_has_real_root_l485_48549


namespace find_a_l485_48525

theorem find_a (a : ℝ) (h : ∀ x y : ℝ, ax + y - 4 = 0 → x + (a + 3/2) * y + 2 = 0 → True) : a = 1/2 :=
sorry

end find_a_l485_48525


namespace initial_money_l485_48547

theorem initial_money {M : ℝ} (h : (M - 10) - (M - 10) / 4 = 15) : M = 30 :=
sorry

end initial_money_l485_48547


namespace find_start_time_l485_48536

def time_first_train_started 
  (distance_pq : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (start_time_train2 : ℝ) 
  (meeting_time : ℝ) 
  (T : ℝ) : ℝ :=
  T

theorem find_start_time 
  (distance_pq : ℝ := 200)
  (speed_train1 : ℝ := 20)
  (speed_train2 : ℝ := 25)
  (start_time_train2 : ℝ := 8)
  (meeting_time : ℝ := 12) 
  : time_first_train_started distance_pq speed_train1 speed_train2 start_time_train2 meeting_time 7 = 7 :=
by
  sorry

end find_start_time_l485_48536


namespace math_proof_problem_l485_48563

noncomputable def proof_problem (c d : ℝ) : Prop :=
  (∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  ∧ (∃ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3)
  ∧ 100 * c + d = 141
  
theorem math_proof_problem (c d : ℝ) 
  (h1 : ∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  (h2 : ∀ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3) :
  100 * c + d = 141 := 
sorry

end math_proof_problem_l485_48563


namespace problem_arithmetic_sequence_l485_48520

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arithmetic_sequence (a : ℕ → ℝ) (d a2 a8 : ℝ) :
  arithmetic_sequence a d →
  (a 2 + a 3 + a 4 + a 5 + a 6 = 450) →
  (a 1 + a 7 = 2 * a 4) →
  (a 2 + a 6 = 2 * a 4) →
  (a 2 + a 8 = 180) :=
by
  sorry

end problem_arithmetic_sequence_l485_48520


namespace intersection_M_N_l485_48574

def M (x : ℝ) : Prop := 2 - x > 0
def N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3

theorem intersection_M_N:
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l485_48574


namespace striped_turtles_adult_percentage_l485_48531

noncomputable def percentage_of_adult_striped_turtles (total_turtles : ℕ) (female_percentage : ℝ) (stripes_per_male : ℕ) (baby_stripes : ℕ) : ℝ :=
  let total_male := total_turtles * (1 - female_percentage)
  let total_striped_male := total_male / stripes_per_male
  let adult_striped_males := total_striped_male - baby_stripes
  (adult_striped_males / total_striped_male) * 100

theorem striped_turtles_adult_percentage :
  percentage_of_adult_striped_turtles 100 0.60 4 4 = 60 := 
  by
  -- proof omitted
  sorry

end striped_turtles_adult_percentage_l485_48531


namespace positive_difference_l485_48587

-- Define the conditions given in the problem
def conditions (x y : ℝ) : Prop :=
  x + y = 40 ∧ 3 * y - 4 * x = 20

-- The theorem to prove
theorem positive_difference (x y : ℝ) (h : conditions x y) : abs (y - x) = 11.42 :=
by
  sorry -- proof omitted

end positive_difference_l485_48587


namespace room_width_is_7_l485_48542

-- Define the conditions of the problem
def room_length : ℝ := 10
def room_height : ℝ := 5
def door_width : ℝ := 1
def door_height : ℝ := 3
def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5
def cost_per_sq_meter : ℝ := 3
def total_cost : ℝ := 474

-- Define the total cost to be painted
def total_area_painted (width : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height) + 2 * (width * room_height)
  let door_area := 2 * (door_width * door_height)
  let window_area := (window1_width * window1_height) + 2 * (window2_width * window2_height)
  wall_area - door_area - window_area

def cost_equation (width : ℝ) : Prop :=
  (total_cost / cost_per_sq_meter) = total_area_painted width

-- Prove that the width required to satisfy the painting cost equation is 7 meters
theorem room_width_is_7 : ∃ w : ℝ, cost_equation w ∧ w = 7 :=
by
  sorry

end room_width_is_7_l485_48542


namespace first_term_of_geometric_series_l485_48553

theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) :
  r = 1 / 4 → S = 20 → S = a / (1 - r) → a = 15 :=
by
  intro hr hS hsum
  sorry

end first_term_of_geometric_series_l485_48553


namespace coefficient_of_xy6_eq_one_l485_48528

theorem coefficient_of_xy6_eq_one (a : ℚ) (h : (7 : ℚ) * a = 1) : a = 1 / 7 :=
by sorry

end coefficient_of_xy6_eq_one_l485_48528


namespace least_total_acorns_l485_48529

theorem least_total_acorns :
  ∃ a₁ a₂ a₃ : ℕ,
    (∀ k : ℕ, (∃ a₁ a₂ a₃ : ℕ,
      (2 * a₁ / 3 + a₁ % 3 / 3 + a₂ + a₃ / 9) % 6 = 4 * k ∧
      (a₁ / 6 + a₂ / 3 + a₃ / 3 + 8 * a₃ / 18) % 6 = 3 * k ∧
      (a₁ / 6 + 5 * a₂ / 6 + a₃ / 9) % 6 = 2 * k) → k = 630) ∧
    (a₁ + a₂ + a₃) = 630 :=
sorry

end least_total_acorns_l485_48529


namespace relationship_between_problems_geometry_problem_count_steve_questions_l485_48539

variable (x y W A G : ℕ)

def word_problems (x : ℕ) : ℕ := x / 2
def addition_and_subtraction_problems (x : ℕ) : ℕ := x / 3
def geometry_problems (x W A : ℕ) : ℕ := x - W - A

theorem relationship_between_problems :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x ∧
  G = geometry_problems x W A →
  W + A + G = x :=
by
  sorry

theorem geometry_problem_count :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x →
  G = geometry_problems x W A →
  G = x / 6 :=
by
  sorry

theorem steve_questions :
  y = x / 2 - 4 :=
by
  sorry

end relationship_between_problems_geometry_problem_count_steve_questions_l485_48539


namespace total_questions_on_test_l485_48591

theorem total_questions_on_test :
  ∀ (correct incorrect score : ℕ),
  (score = correct - 2 * incorrect) →
  (score = 76) →
  (correct = 92) →
  (correct + incorrect = 100) :=
by
  intros correct incorrect score grading_system score_eq correct_eq
  sorry

end total_questions_on_test_l485_48591


namespace exists_three_irrationals_l485_48537

theorem exists_three_irrationals
    (x1 x2 x3 : ℝ)
    (h1 : ¬ ∃ q : ℚ, x1 = q)
    (h2 : ¬ ∃ q : ℚ, x2 = q)
    (h3 : ¬ ∃ q : ℚ, x3 = q)
    (sum_integer : ∃ n : ℤ, x1 + x2 + x3 = n)
    (sum_reciprocals_integer : ∃ m : ℤ, (1/x1) + (1/x2) + (1/x3) = m) :
  true :=
sorry

end exists_three_irrationals_l485_48537


namespace probability_three_defective_before_two_good_correct_l485_48500

noncomputable def probability_three_defective_before_two_good 
  (total_items : ℕ) 
  (good_items : ℕ) 
  (defective_items : ℕ) 
  (sequence_length : ℕ) : ℚ := 
  -- We will skip the proof part and just acknowledge the result as mentioned
  (1 / 55 : ℚ)

theorem probability_three_defective_before_two_good_correct :
  probability_three_defective_before_two_good 12 9 3 5 = 1 / 55 := 
by sorry

end probability_three_defective_before_two_good_correct_l485_48500


namespace number_of_students_passed_l485_48572

theorem number_of_students_passed (total_students : ℕ) (failure_frequency : ℝ) (h1 : total_students = 1000) (h2 : failure_frequency = 0.4) : 
  (total_students - (total_students * failure_frequency)) = 600 :=
by
  sorry

end number_of_students_passed_l485_48572


namespace deductive_reasoning_correct_l485_48596

theorem deductive_reasoning_correct :
  (∀ (s : ℕ), s = 3 ↔
    (s == 1 → DeductiveReasoningGeneralToSpecific ∧
     s == 2 → alwaysCorrect ∧
     s == 3 → InFormOfSyllogism ∧
     s == 4 → ConclusionDependsOnPremisesAndForm)) :=
sorry

end deductive_reasoning_correct_l485_48596


namespace relationship_of_magnitudes_l485_48548

noncomputable def is_ordered (x : ℝ) (A B C : ℝ) : Prop :=
  0 < x ∧ x < Real.pi / 4 ∧
  A = Real.cos (x ^ Real.sin (x ^ Real.sin x)) ∧
  B = Real.sin (x ^ Real.cos (x ^ Real.sin x)) ∧
  C = Real.cos (x ^ Real.sin (x * (x ^ Real.cos x))) ∧
  B < A ∧ A < C

theorem relationship_of_magnitudes (x A B C : ℝ) : 
  is_ordered x A B C := 
sorry

end relationship_of_magnitudes_l485_48548


namespace symmetric_point_correct_l485_48538

def point : Type := ℝ × ℝ × ℝ

def symmetric_with_respect_to_y_axis (A : point) : point :=
  let (x, y, z) := A
  (-x, y, z)

def A : point := (-4, 8, 6)

theorem symmetric_point_correct :
  symmetric_with_respect_to_y_axis A = (4, 8, 6) := by
  sorry

end symmetric_point_correct_l485_48538


namespace first_box_oranges_l485_48501

theorem first_box_oranges (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) = 120) : x = 11 :=
sorry

end first_box_oranges_l485_48501


namespace find_dimensions_l485_48516

theorem find_dimensions (x y : ℝ) 
  (h1 : 90 = (2 * x + y) * (2 * y))
  (h2 : x * y = 10) : x = 2 ∧ y = 5 :=
by
  sorry

end find_dimensions_l485_48516


namespace sum_division_l485_48511

theorem sum_division (x y z : ℝ) (total_share_y : ℝ) 
  (Hx : x = 1) 
  (Hy : y = 0.45) 
  (Hz : z = 0.30) 
  (share_y : total_share_y = 36) 
  : (x + y + z) * (total_share_y / y) = 140 := by
  sorry

end sum_division_l485_48511


namespace power_addition_proof_l485_48517

theorem power_addition_proof :
  (-2) ^ 48 + 3 ^ (4 ^ 3 + 5 ^ 2 - 7 ^ 2) = 2 ^ 48 + 3 ^ 40 := 
by
  sorry

end power_addition_proof_l485_48517


namespace largest_five_digit_divisible_by_97_l485_48510

theorem largest_five_digit_divisible_by_97 :
  ∃ n, (99999 - n % 97) = 99930 ∧ n % 97 = 0 ∧ 10000 ≤ n ∧ n ≤ 99999 :=
by
  sorry

end largest_five_digit_divisible_by_97_l485_48510


namespace log_expression_simplification_l485_48581

open Real

noncomputable def log_expr (a b c d x y z : ℝ) : ℝ :=
  log (a^2 / b) + log (b^2 / c) + log (c^2 / d) - log (a^2 * y * z / (d^2 * x))

theorem log_expression_simplification (a b c d x y z : ℝ) (h : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : d ≠ 0) (h5 : x ≠ 0) (h6 : y ≠ 0) (h7 : z ≠ 0) :
  log_expr a b c d x y z = log (bdx / yz) :=
by
  -- Proof goes here
  sorry

end log_expression_simplification_l485_48581


namespace kx2_kx_1_pos_l485_48597

theorem kx2_kx_1_pos (k : ℝ) : (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end kx2_kx_1_pos_l485_48597


namespace sqrt_of_9_l485_48586

theorem sqrt_of_9 : Real.sqrt 9 = 3 := 
by 
  sorry

end sqrt_of_9_l485_48586


namespace one_thirds_in_fraction_l485_48502

theorem one_thirds_in_fraction : (11 / 5) / (1 / 3) = 33 / 5 := by
  sorry

end one_thirds_in_fraction_l485_48502


namespace rubiks_cube_repeats_l485_48598

theorem rubiks_cube_repeats (num_positions : ℕ) (H : num_positions = 43252003274489856000) 
  (moves : ℕ → ℕ) : 
  ∃ n, ∃ m, (∀ P, moves n = moves m → P = moves 0) :=
by
  sorry

end rubiks_cube_repeats_l485_48598


namespace fraction_not_equal_l485_48554

theorem fraction_not_equal : ¬ (7 / 5 = 1 + 4 / 20) :=
by
  -- We'll use simplification to demonstrate the inequality
  sorry

end fraction_not_equal_l485_48554


namespace area_of_given_polygon_l485_48562

def point := (ℝ × ℝ)

def vertices : List point := [(0,0), (5,0), (5,2), (3,2), (3,3), (2,3), (2,2), (0,2), (0,0)]

def polygon_area (vertices : List point) : ℝ := 
  -- Function to compute the area of the given polygon
  -- Implementation of the area computation is assumed to be correct
  sorry

theorem area_of_given_polygon : polygon_area vertices = 11 :=
sorry

end area_of_given_polygon_l485_48562


namespace regular_price_of_fish_l485_48506

theorem regular_price_of_fish (discounted_price_per_quarter_pound : ℝ)
  (discount : ℝ) (hp1 : discounted_price_per_quarter_pound = 2) (hp2 : discount = 0.4) :
  ∃ x : ℝ, x = (40 / 3) :=
by
  sorry

end regular_price_of_fish_l485_48506


namespace pollution_control_l485_48515

theorem pollution_control (x y : ℕ) (h1 : x - y = 5) (h2 : 2 * x + 3 * y = 45) : x = 12 ∧ y = 7 :=
by
  sorry

end pollution_control_l485_48515


namespace opposite_2024_eq_neg_2024_l485_48560

def opposite (n : ℤ) : ℤ := -n

theorem opposite_2024_eq_neg_2024 : opposite 2024 = -2024 :=
by
  sorry

end opposite_2024_eq_neg_2024_l485_48560


namespace obtain_angle_10_30_l485_48583

theorem obtain_angle_10_30 (a : ℕ) (h : 100 + a = 135) : a = 35 := 
by sorry

end obtain_angle_10_30_l485_48583


namespace necessary_but_not_sufficient_l485_48594

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 0 = 0) →
  (∀ x : ℝ, f (-x) = -f x) →
  ¬∀ f' : ℝ → ℝ, (f' 0 = 0 → ∀ y : ℝ, f' (-y) = -f' y)
:= by
  sorry

end necessary_but_not_sufficient_l485_48594


namespace find_f_inv_128_l485_48585

open Function

theorem find_f_inv_128 (f : ℕ → ℕ) 
  (h₀ : f 5 = 2) 
  (h₁ : ∀ x, f (2 * x) = 2 * f x) : 
  f⁻¹' {128} = {320} :=
by
  sorry

end find_f_inv_128_l485_48585


namespace min_value_of_f_solve_inequality_l485_48573

noncomputable def f (x : ℝ) : ℝ := abs (x - 5/2) + abs (x - 1/2)

theorem min_value_of_f : (∀ x : ℝ, f x ≥ 2) ∧ (∃ x : ℝ, f x = 2) := by
  sorry

theorem solve_inequality (x : ℝ) : (f x ≤ x + 4) ↔ (-1/3 ≤ x ∧ x ≤ 7) := by
  sorry

end min_value_of_f_solve_inequality_l485_48573


namespace smaller_of_x_y_l485_48579

theorem smaller_of_x_y (x y a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : x * y = c) (h6 : x^2 - b * x + a * y = 0) : min x y = c / a :=
by sorry

end smaller_of_x_y_l485_48579


namespace trig_identity_l485_48527

open Real

theorem trig_identity (α : ℝ) (hα : α > -π ∧ α < -π/2) :
  (sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α))) = - 2 / tan α :=
by
  sorry

end trig_identity_l485_48527


namespace problem_statement_l485_48577

theorem problem_statement (a : ℕ → ℝ)
  (h_recur : ∀ n, n ≥ 1 → a (n + 1) = a (n - 1) / (1 + n * a (n - 1) * a n))
  (h_initial_0 : a 0 = 1)
  (h_initial_1 : a 1 = 1) :
  1 / (a 190 * a 200) = 19901 :=
by
  sorry

end problem_statement_l485_48577


namespace geom_seq_sum_l485_48544

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : a 0 + a 1 = 3 / 4)
  (h4 : a 2 + a 3 + a 4 + a 5 = 15) :
  a 6 + a 7 + a 8 = 112 := by
  sorry

end geom_seq_sum_l485_48544


namespace number_of_bricks_l485_48590

theorem number_of_bricks (b1_hours b2_hours combined_hours: ℝ) (reduction_rate: ℝ) (x: ℝ):
  b1_hours = 12 ∧ 
  b2_hours = 15 ∧ 
  combined_hours = 6 ∧ 
  reduction_rate = 15 ∧ 
  (combined_hours * ((x / b1_hours) + (x / b2_hours) - reduction_rate) = x) → 
  x = 1800 :=
by
  sorry

end number_of_bricks_l485_48590


namespace billy_sleep_total_l485_48526

def billy_sleep : Prop :=
  let first_night := 6
  let second_night := first_night + 2
  let third_night := second_night / 2
  let fourth_night := third_night * 3
  first_night + second_night + third_night + fourth_night = 30

theorem billy_sleep_total : billy_sleep := by
  sorry

end billy_sleep_total_l485_48526


namespace ap_contains_sixth_power_l485_48541

theorem ap_contains_sixth_power (a d : ℕ) (i j x y : ℕ) 
  (h_positive : ∀ n, a + n * d > 0) 
  (h_square : a + i * d = x^2) 
  (h_cube : a + j * d = y^3) :
  ∃ k z : ℕ, a + k * d = z^6 := 
  sorry

end ap_contains_sixth_power_l485_48541


namespace find_y_l485_48546

theorem find_y (x y : ℤ) (h1 : x^2 - 3 * x + 7 = y + 3) (h2 : x = -5) : y = 44 := by
  sorry

end find_y_l485_48546


namespace girls_select_same_colored_marble_l485_48550

def probability_same_color (total_white total_black girls boys : ℕ) : ℚ :=
  let prob_white := (total_white * (total_white - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  let prob_black := (total_black * (total_black - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  prob_white + prob_black

theorem girls_select_same_colored_marble :
  probability_same_color 2 2 2 2 = 1 / 3 :=
by
  sorry

end girls_select_same_colored_marble_l485_48550


namespace checker_move_10_cells_checker_move_11_cells_l485_48552

noncomputable def F : ℕ → Nat 
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem checker_move_10_cells : F 10 = 89 := by
  sorry

theorem checker_move_11_cells : F 11 = 144 := by
  sorry

end checker_move_10_cells_checker_move_11_cells_l485_48552


namespace average_speed_ratio_l485_48533

theorem average_speed_ratio
  (time_eddy : ℕ)
  (time_freddy : ℕ)
  (distance_ab : ℕ)
  (distance_ac : ℕ)
  (h1 : time_eddy = 3)
  (h2 : time_freddy = 4)
  (h3 : distance_ab = 570)
  (h4 : distance_ac = 300) :
  (distance_ab / time_eddy) / (distance_ac / time_freddy) = 38 / 15 := 
by
  sorry

end average_speed_ratio_l485_48533


namespace company_employee_percentage_l485_48512

theorem company_employee_percentage (M : ℝ)
  (h1 : 0.20 * M + 0.40 * (1 - M) = 0.31000000000000007) :
  M = 0.45 :=
sorry

end company_employee_percentage_l485_48512


namespace geo_seq_sum_condition_l485_48532

noncomputable def geometric_seq (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^n

noncomputable def sum_geo_seq_3 (a : ℝ) (q : ℝ) : ℝ :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

noncomputable def sum_geo_seq_6 (a : ℝ) (q : ℝ) : ℝ :=
  sum_geo_seq_3 a q + geometric_seq a q 3 + geometric_seq a q 4 + geometric_seq a q 5

theorem geo_seq_sum_condition {a q S₃ S₆ : ℝ} (h_sum_eq : S₆ = 9 * S₃)
  (h_S₃_def : S₃ = sum_geo_seq_3 a q)
  (h_S₆_def : S₆ = sum_geo_seq_6 a q) :
  q = 2 :=
by
  sorry

end geo_seq_sum_condition_l485_48532


namespace arithmetic_sequence_properties_l485_48522

variables {a : ℕ → ℤ} {S T : ℕ → ℤ}

theorem arithmetic_sequence_properties 
  (h₁ : a 2 = 11)
  (h₂ : S 10 = 40)
  (h₃ : ∀ n, S n = n * a 1 + (n * (n - 1)) / 2 * (a 2 - a 1)) -- Sum of first n terms of arithmetic sequence
  (h₄ : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1)) -- General term formula of arithmetic sequence
  : (∀ n, a n = -2 * n + 15) ∧
    ( (∀ n, 1 ≤ n ∧ n ≤ 7 → T n = -n^2 + 14 * n) ∧ 
      (∀ n, n ≥ 8 → T n = n^2 - 14 * n + 98)) :=
by
sorry

end arithmetic_sequence_properties_l485_48522


namespace city_rentals_cost_per_mile_l485_48576

theorem city_rentals_cost_per_mile (x : ℝ)
  (h₁ : 38.95 + 150 * x = 41.95 + 150 * 0.29) :
  x = 0.31 :=
by sorry

end city_rentals_cost_per_mile_l485_48576


namespace birgit_numbers_sum_l485_48508

theorem birgit_numbers_sum (a b c d : ℕ) 
  (h1 : a + b + c = 415) 
  (h2 : a + b + d = 442) 
  (h3 : a + c + d = 396) 
  (h4 : b + c + d = 325) : 
  a + b + c + d = 526 :=
by
  sorry

end birgit_numbers_sum_l485_48508


namespace three_times_sum_first_35_odd_l485_48513

/-- 
The sum of the first n odd numbers --/
def sum_first_n_odd (n : ℕ) : ℕ := n * n

/-- Given that 69 is the 35th odd number --/
theorem three_times_sum_first_35_odd : 3 * sum_first_n_odd 35 = 3675 := by
  sorry

end three_times_sum_first_35_odd_l485_48513


namespace train_speed_kmh_l485_48571

theorem train_speed_kmh 
  (L_train : ℝ) (L_bridge : ℝ) (time : ℝ)
  (h_train : L_train = 460)
  (h_bridge : L_bridge = 140)
  (h_time : time = 48) : 
  (L_train + L_bridge) / time * 3.6 = 45 := 
by
  -- Definitions and conditions
  have h_total_dist : L_train + L_bridge = 600 := by sorry
  have h_speed_mps : (L_train + L_bridge) / time = 600 / 48 := by sorry
  have h_speed_mps_simplified : 600 / 48 = 12.5 := by sorry
  have h_speed_kmh : 12.5 * 3.6 = 45 := by sorry
  sorry

end train_speed_kmh_l485_48571


namespace smallest_q_for_5_in_range_l485_48556

theorem smallest_q_for_5_in_range : ∃ q, (q = 9) ∧ (∃ x, (x^2 - 4 * x + q = 5)) := 
by 
  sorry

end smallest_q_for_5_in_range_l485_48556


namespace heating_rate_l485_48505

/-- 
 Andy is making fudge. He needs to raise the temperature of the candy mixture from 60 degrees to 240 degrees. 
 Then, he needs to cool it down to 170 degrees. The candy heats at a certain rate and cools at a rate of 7 degrees/minute.
 It takes 46 minutes for the candy to be done. Prove that the heating rate is 5 degrees per minute.
-/
theorem heating_rate (initial_temp heating_temp cooling_temp : ℝ) (cooling_rate total_time : ℝ) 
  (h1 : initial_temp = 60) (h2 : heating_temp = 240) (h3 : cooling_temp = 170) 
  (h4 : cooling_rate = 7) (h5 : total_time = 46) : 
  ∃ (H : ℝ), H = 5 :=
by 
  -- We declare here that the rate H exists and is 5 degrees per minute.
  let H : ℝ := 5
  existsi H
  sorry

end heating_rate_l485_48505


namespace symmetric_point_in_third_quadrant_l485_48509

-- Define a structure for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to find the symmetric point about the y-axis
def symmetric_about_y (P : Point) : Point :=
  Point.mk (-P.x) P.y

-- Define the original point P
def P : Point := { x := 3, y := -2 }

-- Define the symmetric point P' about the y-axis
def P' : Point := symmetric_about_y P

-- Define a condition to determine if a point is in the third quadrant
def is_in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- The theorem stating that the symmetric point of P about the y-axis is in the third quadrant
theorem symmetric_point_in_third_quadrant : is_in_third_quadrant P' :=
  by
  sorry

end symmetric_point_in_third_quadrant_l485_48509


namespace candy_difference_l485_48570

-- Defining the conditions as Lean hypotheses
variable (R K B M : ℕ)

-- Given conditions
axiom h1 : K = 4
axiom h2 : B = M - 6
axiom h3 : M = R + 2
axiom h4 : K = B + 2

-- Prove that Robert gets 2 more pieces of candy than Kate
theorem candy_difference : R - K = 2 :=
by {
  sorry
}

end candy_difference_l485_48570


namespace sandy_tokens_ratio_l485_48589

theorem sandy_tokens_ratio :
  ∀ (total_tokens : ℕ) (num_siblings : ℕ) (difference : ℕ),
  total_tokens = 1000000 →
  num_siblings = 4 →
  difference = 375000 →
  ∃ (sandy_tokens : ℕ),
  sandy_tokens = (total_tokens - (num_siblings * ((total_tokens - difference) / (num_siblings + 1)))) ∧
  sandy_tokens / total_tokens = 1 / 2 :=
by 
  intros total_tokens num_siblings difference h1 h2 h3
  sorry

end sandy_tokens_ratio_l485_48589


namespace solution_l485_48519

noncomputable def problem (x : ℕ) : Prop :=
  2 ^ 28 = 4 ^ x  -- Simplified form of the condition given

theorem solution : problem 14 :=
by
  sorry

end solution_l485_48519


namespace triangle_inscribed_and_arcs_l485_48514

theorem triangle_inscribed_and_arcs
  (PQ QR PR : ℝ) (X Y Z : ℝ)
  (QY XZ QX YZ PX RY : ℝ)
  (H1 : PQ = 26)
  (H2 : QR = 28) 
  (H3 : PR = 27)
  (H4 : QY = XZ)
  (H5 : QX = YZ)
  (H6 : PX = RY)
  (H7 : RY = PX + 1)
  (H8 : XZ = QX + 1)
  (H9 : QY = YZ + 2) :
  QX = 29 / 2 :=
by
  sorry

end triangle_inscribed_and_arcs_l485_48514


namespace max_alpha_l485_48582

theorem max_alpha (A B C : ℝ) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hSum : A + B + C = π)
  (hmin : ∀ alpha, alpha = min (2 * A - B) (min (3 * B - 2 * C) (π / 2 - A))) :
  ∃ alpha, alpha = 2 * π / 9 := 
sorry

end max_alpha_l485_48582


namespace cube_volume_l485_48545

/-- Given the perimeter of one face of a cube, proving the volume of the cube -/

theorem cube_volume (h : ∀ (s : ℝ), 4 * s = 28) : (∃ (v : ℝ), v = (7 : ℝ) ^ 3) :=
by
  sorry

end cube_volume_l485_48545


namespace find_number_l485_48580

theorem find_number (x : ℝ) (h : 20 / x = 0.8) : x = 25 := 
by
  sorry

end find_number_l485_48580


namespace max_amount_paul_received_l485_48593

theorem max_amount_paul_received :
  ∃ (numBplus numA numAplus : ℕ),
  (numBplus + numA + numAplus = 10) ∧ 
  (numAplus ≥ 2 → 
    let BplusReward := 5;
    let AReward := 2 * BplusReward;
    let AplusReward := 15;
    let Total := numAplus * AplusReward + numA * (2 * AReward) + numBplus * (2 * BplusReward);
    Total = 190
  ) :=
sorry

end max_amount_paul_received_l485_48593


namespace jordyn_total_payment_l485_48534

theorem jordyn_total_payment :
  let price_cherries := 5
  let price_olives := 7
  let price_grapes := 11
  let num_cherries := 50
  let num_olives := 75
  let num_grapes := 25
  let discount_cherries := 0.12
  let discount_olives := 0.08
  let discount_grapes := 0.15
  let sales_tax := 0.05
  let service_charge := 0.02
  let total_cherries := num_cherries * price_cherries
  let total_olives := num_olives * price_olives
  let total_grapes := num_grapes * price_grapes
  let discounted_cherries := total_cherries * (1 - discount_cherries)
  let discounted_olives := total_olives * (1 - discount_olives)
  let discounted_grapes := total_grapes * (1 - discount_grapes)
  let subtotal := discounted_cherries + discounted_olives + discounted_grapes
  let taxed_amount := subtotal * (1 + sales_tax)
  let final_amount := taxed_amount * (1 + service_charge)
  final_amount = 1002.32 :=
by
  sorry

end jordyn_total_payment_l485_48534


namespace diff_of_squares_l485_48551

theorem diff_of_squares (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
by
  sorry

end diff_of_squares_l485_48551


namespace cylinder_lateral_surface_area_l485_48568

theorem cylinder_lateral_surface_area (r l : ℝ) (A : ℝ) (h_r : r = 1) (h_l : l = 2) : A = 4 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l485_48568


namespace problem_293_l485_48592

theorem problem_293 (s : ℝ) (R' : ℝ) (rectangle1 : ℝ) (circle1 : ℝ) 
  (condition1 : s = 4) 
  (condition2 : rectangle1 = 2 * 4) 
  (condition3 : circle1 = Real.pi * 1^2) 
  (condition4 : R' = s^2 - (rectangle1 + circle1)) 
  (fraction_form : ∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n) : 
  (∃ m n : ℕ, gcd m n = 1 ∧ R' = m / n ∧ m + n = 293) := 
sorry

end problem_293_l485_48592


namespace inhabitable_fraction_of_mars_surface_l485_48523

theorem inhabitable_fraction_of_mars_surface :
  (3 / 5 : ℚ) * (2 / 3) = (2 / 5) :=
by
  sorry

end inhabitable_fraction_of_mars_surface_l485_48523


namespace ratio_third_first_l485_48507

theorem ratio_third_first (A B C : ℕ) (h1 : A + B + C = 110) (h2 : A = 2 * B) (h3 : B = 30) :
  C / A = 1 / 3 :=
by
  sorry

end ratio_third_first_l485_48507


namespace marathons_total_distance_l485_48503

theorem marathons_total_distance :
  ∀ (m y : ℕ),
  (26 + 385 / 1760 : ℕ) = 26 ∧ 385 % 1760 = 385 →
  15 * 26 + 15 * 385 / 1760 = m + 495 / 1760 ∧
  15 * 385 % 1760 = 495 →
  0 ≤ 495 ∧ 495 < 1760 →
  y = 495 := by
  intros
  sorry

end marathons_total_distance_l485_48503


namespace money_problem_solution_l485_48561

theorem money_problem_solution (a b : ℝ) (h1 : 7 * a + b < 100) (h2 : 4 * a - b = 40) (h3 : b = 0.5 * a) : 
  a = 80 / 7 ∧ b = 40 / 7 :=
by
  sorry

end money_problem_solution_l485_48561


namespace train_speed_proof_l485_48521

noncomputable def speedOfTrain (lengthOfTrain : ℝ) (timeToCross : ℝ) (speedOfMan : ℝ) : ℝ :=
  let man_speed_m_per_s := speedOfMan * 1000 / 3600
  let relative_speed := lengthOfTrain / timeToCross
  let train_speed_m_per_s := relative_speed + man_speed_m_per_s
  train_speed_m_per_s * 3600 / 1000

theorem train_speed_proof :
  speedOfTrain 100 5.999520038396929 3 = 63 := by
  sorry

end train_speed_proof_l485_48521


namespace rope_cut_into_pieces_l485_48564

theorem rope_cut_into_pieces (length_of_rope_cm : ℕ) (num_equal_pieces : ℕ) (length_equal_piece_mm : ℕ) (length_remaining_piece_mm : ℕ) 
  (h1 : length_of_rope_cm = 1165) (h2 : num_equal_pieces = 150) (h3 : length_equal_piece_mm = 75) (h4 : length_remaining_piece_mm = 100) :
  (num_equal_pieces * length_equal_piece_mm + (11650 - num_equal_pieces * length_equal_piece_mm) / length_remaining_piece_mm = 154) :=
by
  sorry

end rope_cut_into_pieces_l485_48564


namespace inequality_proof_l485_48543

variables (a b c : ℝ)

theorem inequality_proof
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (cond : a^2 + b^2 + c^2 + ab + bc + ca ≤ 2) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 := 
sorry

end inequality_proof_l485_48543
