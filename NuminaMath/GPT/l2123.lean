import Mathlib

namespace NUMINAMATH_GPT_fraction_simplification_l2123_212324

theorem fraction_simplification : (3^2040 + 3^2038) / (3^2040 - 3^2038) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l2123_212324


namespace NUMINAMATH_GPT_initial_deadline_l2123_212303

theorem initial_deadline (W : ℕ) (R : ℕ) (D : ℕ) :
    100 * 25 * 8 = (1/3 : ℚ) * W →
    (2/3 : ℚ) * W = 160 * R * 10 →
    D = 25 + R →
    D = 50 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_initial_deadline_l2123_212303


namespace NUMINAMATH_GPT_number_of_men_in_first_group_l2123_212331

-- Define the conditions as hypotheses in Lean
def work_completed_in_25_days (x : ℕ) : Prop := x * 25 * (1 : ℚ) / (25 * x) = (1 : ℚ)
def twenty_men_complete_in_15_days : Prop := 20 * 15 * (1 : ℚ) / 15 = (1 : ℚ)

-- Define the theorem to prove the number of men in the first group
theorem number_of_men_in_first_group (x : ℕ) (h1 : work_completed_in_25_days x)
  (h2 : twenty_men_complete_in_15_days) : x = 20 :=
  sorry

end NUMINAMATH_GPT_number_of_men_in_first_group_l2123_212331


namespace NUMINAMATH_GPT_word_identification_l2123_212366

theorem word_identification (word : String) :
  ( ( (word = "бал" ∨ word = "баллы")
    ∧ (∃ sport : String, sport = "figure skating" ∨ sport = "rhythmic gymnastics"))
    ∧ (∃ year : Nat, year = 2015 ∧ word = "пенсионные баллы") ) → 
  word = "баллы" :=
by
  sorry

end NUMINAMATH_GPT_word_identification_l2123_212366


namespace NUMINAMATH_GPT_cards_eaten_by_hippopotamus_l2123_212369

-- Defining the initial and remaining number of cards
def initial_cards : ℕ := 72
def remaining_cards : ℕ := 11

-- Statement of the proof problem
theorem cards_eaten_by_hippopotamus (initial_cards remaining_cards : ℕ) : initial_cards - remaining_cards = 61 :=
by
  sorry

end NUMINAMATH_GPT_cards_eaten_by_hippopotamus_l2123_212369


namespace NUMINAMATH_GPT_solve_inequality_system_l2123_212326

-- Define the inequalities as conditions.
def cond1 (x : ℝ) := 2 * x + 1 < 3 * x - 2
def cond2 (x : ℝ) := 3 * (x - 2) - x ≤ 4

-- Formulate the theorem to prove that these conditions give the solution 3 < x ≤ 5.
theorem solve_inequality_system (x : ℝ) : cond1 x ∧ cond2 x ↔ 3 < x ∧ x ≤ 5 := 
sorry

end NUMINAMATH_GPT_solve_inequality_system_l2123_212326


namespace NUMINAMATH_GPT_ryan_time_learning_l2123_212314

variable (t : ℕ) (c : ℕ)

/-- Ryan spends a total of 3 hours on both languages every day. Assume further that he spends 1 hour on learning Chinese every day, and you need to find how many hours he spends on learning English. --/
theorem ryan_time_learning (h_total : t = 3) (h_chinese : c = 1) : (t - c) = 2 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ryan_time_learning_l2123_212314


namespace NUMINAMATH_GPT_hunting_season_fraction_l2123_212337

noncomputable def fraction_of_year_hunting_season (hunting_times_per_month : ℕ) 
    (deers_per_hunt : ℕ) (weight_per_deer : ℕ) (fraction_kept : ℚ) 
    (total_weight_kept : ℕ) : ℚ :=
  let total_yearly_weight := total_weight_kept * 2
  let weight_per_hunt := deers_per_hunt * weight_per_deer
  let total_hunts_per_year := total_yearly_weight / weight_per_hunt
  let total_months_hunting := total_hunts_per_year / hunting_times_per_month
  let fraction_of_year := total_months_hunting / 12
  fraction_of_year

theorem hunting_season_fraction : 
  fraction_of_year_hunting_season 6 2 600 (1 / 2 : ℚ) 10800 = 1 / 4 := 
by
  simp [fraction_of_year_hunting_season]
  sorry

end NUMINAMATH_GPT_hunting_season_fraction_l2123_212337


namespace NUMINAMATH_GPT_find_constants_l2123_212302

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x + 1

noncomputable def f_inv (x a b c : ℝ) : ℝ :=
  ( (x - a + Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3) +
  ( (x - a - Real.sqrt (x^2 - b * x + c)) / 2 )^(1/3)

theorem find_constants (a b c : ℝ) (h1 : f_inv (1:ℝ) a b c = 0)
  (ha : a = 1) (hb : b = 2) (hc : c = 5) : a + 10 * b + 100 * c = 521 :=
by
  rw [ha, hb, hc]
  norm_num

end NUMINAMATH_GPT_find_constants_l2123_212302


namespace NUMINAMATH_GPT_find_x0_l2123_212319

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem find_x0 (x0 : ℝ) (h : deriv f x0 = 0) : x0 = Real.exp 1 :=
by 
  sorry

end NUMINAMATH_GPT_find_x0_l2123_212319


namespace NUMINAMATH_GPT_length_of_ST_l2123_212357

theorem length_of_ST (LM MN NL: ℝ) (LR : ℝ) (LT TR LS SR: ℝ) 
  (h1: LM = 8) (h2: MN = 10) (h3: NL = 6) (h4: LR = 6) 
  (h5: LT = 8 / 3) (h6: TR = 10 / 3) (h7: LS = 9 / 4) (h8: SR = 15 / 4) :
  LS - LT = -5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_length_of_ST_l2123_212357


namespace NUMINAMATH_GPT_range_of_m_l2123_212358

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ -4 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l2123_212358


namespace NUMINAMATH_GPT_D_72_value_l2123_212351

-- Define D(n) as described
def D (n : ℕ) : ℕ := 
  sorry -- Placeholder for the actual function definition

-- Theorem statement
theorem D_72_value : D 72 = 97 :=
by sorry

end NUMINAMATH_GPT_D_72_value_l2123_212351


namespace NUMINAMATH_GPT_average_of_multiples_l2123_212352

theorem average_of_multiples :
  let sum_of_first_7_multiples_of_9 := 9 + 18 + 27 + 36 + 45 + 54 + 63
  let sum_of_first_5_multiples_of_11 := 11 + 22 + 33 + 44 + 55
  let sum_of_first_3_negative_multiples_of_13 := -13 + -26 + -39
  let total_sum := sum_of_first_7_multiples_of_9 + sum_of_first_5_multiples_of_11 + sum_of_first_3_negative_multiples_of_13
  let average := total_sum / 3
  average = 113 :=
by
  sorry

end NUMINAMATH_GPT_average_of_multiples_l2123_212352


namespace NUMINAMATH_GPT_base_conversion_subtraction_l2123_212305

theorem base_conversion_subtraction :
  let n1_base9 := 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  let n2_base7 := 1 * 7^2 + 6 * 7^1 + 5 * 7^0
  n1_base9 - n2_base7 = 169 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_subtraction_l2123_212305


namespace NUMINAMATH_GPT_lcm_12_18_l2123_212383

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_GPT_lcm_12_18_l2123_212383


namespace NUMINAMATH_GPT_cone_lateral_area_l2123_212320

theorem cone_lateral_area (r l S: ℝ) (h1: r = 1 / 2) (h2: l = 1) (h3: S = π * r * l) : 
  S = π / 2 :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_area_l2123_212320


namespace NUMINAMATH_GPT_minimum_points_to_guarantee_highest_score_l2123_212350

theorem minimum_points_to_guarantee_highest_score :
  ∃ (score1 score2 score3 : ℕ), 
   (score1 = 7 ∨ score1 = 4 ∨ score1 = 2) ∧ (score2 = 7 ∨ score2 = 4 ∨ score2 = 2) ∧
   (score3 = 7 ∨ score3 = 4 ∨ score3 = 2) ∧ 
   (∀ (score4 : ℕ), 
     (score4 = 7 ∨ score4 = 4 ∨ score4 = 2) → 
     (score1 + score2 + score3 + score4 < 25)) → 
  score1 + score2 + score3 + 7 ≥ 25 :=
   sorry

end NUMINAMATH_GPT_minimum_points_to_guarantee_highest_score_l2123_212350


namespace NUMINAMATH_GPT_complement_A_B_correct_l2123_212327

open Set

-- Given sets A and B
def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

-- Define the complement of B with respect to A
def complement_A_B : Set ℕ := A \ B

-- Statement to prove
theorem complement_A_B_correct : complement_A_B = {0, 2, 6, 10} :=
  by sorry

end NUMINAMATH_GPT_complement_A_B_correct_l2123_212327


namespace NUMINAMATH_GPT_maximum_value_x2_add_3xy_add_y2_l2123_212340

-- Define the conditions
variables {x y : ℝ}

-- State the theorem
theorem maximum_value_x2_add_3xy_add_y2 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : 3 * x^2 - 2 * x * y + 5 * y^2 = 12) :
  ∃ e f g h : ℕ,
    x^2 + 3 * x * y + y^2 = (1144 + 204 * Real.sqrt 15) / 91 ∧ e + f + g + h = 1454 :=
sorry

end NUMINAMATH_GPT_maximum_value_x2_add_3xy_add_y2_l2123_212340


namespace NUMINAMATH_GPT_solve_for_n_l2123_212316

theorem solve_for_n (n : ℝ) (h : 1 / (2 * n) + 1 / (4 * n) = 3 / 12) : n = 3 :=
sorry

end NUMINAMATH_GPT_solve_for_n_l2123_212316


namespace NUMINAMATH_GPT_find_x_squared_l2123_212308

variable (a b x p q : ℝ)

theorem find_x_squared (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : q ≠ p) (h4 : (a^2 + x^2) / (b^2 + x^2) = p / q) : 
  x^2 = (b^2 * p - a^2 * q) / (q - p) := 
by 
  sorry

end NUMINAMATH_GPT_find_x_squared_l2123_212308


namespace NUMINAMATH_GPT_max_contestants_l2123_212306

theorem max_contestants (n : ℕ) (h1 : n = 55) (h2 : ∀ (i j : ℕ), i < j → j < n → (j - i) % 5 ≠ 4) : ∃(k : ℕ), k = 30 := 
  sorry

end NUMINAMATH_GPT_max_contestants_l2123_212306


namespace NUMINAMATH_GPT_red_pencils_in_box_l2123_212325

theorem red_pencils_in_box (B R G : ℕ) 
  (h1 : B + R + G = 20)
  (h2 : B = 6 * G)
  (h3 : R < B) : R = 6 := by
  sorry

end NUMINAMATH_GPT_red_pencils_in_box_l2123_212325


namespace NUMINAMATH_GPT_tangent_line_at_pi_over_4_l2123_212345

noncomputable def tangent_eq (x y : ℝ) : Prop :=
  y = 2 * x * Real.tan x

noncomputable def tangent_line_eq (x y : ℝ) : Prop :=
  (2 + Real.pi) * x - y - (Real.pi^2 / 4) = 0

theorem tangent_line_at_pi_over_4 :
  tangent_eq (Real.pi / 4) (Real.pi / 2) →
  tangent_line_eq (Real.pi / 4) (Real.pi / 2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_pi_over_4_l2123_212345


namespace NUMINAMATH_GPT_inequality_part1_inequality_part2_l2123_212353

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end NUMINAMATH_GPT_inequality_part1_inequality_part2_l2123_212353


namespace NUMINAMATH_GPT_Carter_cards_l2123_212346

variable (C : ℕ) -- Let C be the number of baseball cards Carter has.

-- Condition 1: Marcus has 210 baseball cards.
def Marcus_cards : ℕ := 210

-- Condition 2: Marcus has 58 more cards than Carter.
def Marcus_has_more (C : ℕ) : Prop := Marcus_cards = C + 58

theorem Carter_cards (C : ℕ) (h : Marcus_has_more C) : C = 152 :=
by
  -- Expand the condition
  unfold Marcus_has_more at h
  -- Simplify the given equation
  rw [Marcus_cards] at h
  -- Solve for C
  linarith

end NUMINAMATH_GPT_Carter_cards_l2123_212346


namespace NUMINAMATH_GPT_find_income_l2123_212355

noncomputable def income_expenditure_proof : Prop := 
  ∃ (x : ℕ), (5 * x - 4 * x = 3600) ∧ (5 * x = 18000)

theorem find_income : income_expenditure_proof :=
  sorry

end NUMINAMATH_GPT_find_income_l2123_212355


namespace NUMINAMATH_GPT_largest_integer_among_four_l2123_212392

theorem largest_integer_among_four 
  (x y z w : ℤ)
  (h1 : x + y + z = 234)
  (h2 : x + y + w = 255)
  (h3 : x + z + w = 271)
  (h4 : y + z + w = 198) :
  max x (max y (max z w)) = 121 := 
by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_largest_integer_among_four_l2123_212392


namespace NUMINAMATH_GPT_remainder_sum_mod_13_l2123_212301

theorem remainder_sum_mod_13 (a b c d : ℕ) 
(h₁ : a % 13 = 3) (h₂ : b % 13 = 5) (h₃ : c % 13 = 7) (h₄ : d % 13 = 9) : 
  (a + b + c + d) % 13 = 11 :=
by sorry

end NUMINAMATH_GPT_remainder_sum_mod_13_l2123_212301


namespace NUMINAMATH_GPT_storks_initially_l2123_212311

-- Definitions for conditions
variable (S : ℕ) -- initial number of storks
variable (B : ℕ) -- initial number of birds

theorem storks_initially (h1 : B = 2) (h2 : S = B + 3 + 1) : S = 6 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_storks_initially_l2123_212311


namespace NUMINAMATH_GPT_find_x_l2123_212362

-- Definitions of vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Definition of parallel vectors
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem statement
theorem find_x (x : ℝ) (h_parallel : parallel a (b x)) : x = 6 :=
sorry

end NUMINAMATH_GPT_find_x_l2123_212362


namespace NUMINAMATH_GPT_units_digit_of_first_four_composite_numbers_l2123_212344

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_first_four_composite_numbers_l2123_212344


namespace NUMINAMATH_GPT_books_per_day_l2123_212378

-- Define the condition: Mrs. Hilt reads 15 books in 3 days.
def reads_books_in_days (total_books : ℕ) (days : ℕ) : Prop :=
  total_books = 15 ∧ days = 3

-- Define the theorem to prove that Mrs. Hilt reads 5 books per day.
theorem books_per_day (total_books : ℕ) (days : ℕ) (h : reads_books_in_days total_books days) : total_books / days = 5 :=
by
  -- Stub proof
  sorry

end NUMINAMATH_GPT_books_per_day_l2123_212378


namespace NUMINAMATH_GPT_greatest_second_term_arithmetic_sequence_l2123_212309

theorem greatest_second_term_arithmetic_sequence:
  ∃ a d : ℕ, (a > 0) ∧ (d > 0) ∧ (2 * a + 3 * d = 29) ∧ (4 * a + 6 * d = 58) ∧ (((a + d : ℤ) / 3 : ℤ) = 10) :=
sorry

end NUMINAMATH_GPT_greatest_second_term_arithmetic_sequence_l2123_212309


namespace NUMINAMATH_GPT_three_pow_sub_cube_eq_two_l2123_212313

theorem three_pow_sub_cube_eq_two (k : ℕ) (h : 30^k ∣ 929260) : 3^k - k^3 = 2 := 
sorry

end NUMINAMATH_GPT_three_pow_sub_cube_eq_two_l2123_212313


namespace NUMINAMATH_GPT_quadratic_real_roots_range_k_l2123_212335

-- Define the quadratic function
def quadratic_eq (k x : ℝ) : ℝ := k * x^2 - 6 * x + 9

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the conditions for the quadratic equation to have distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ discriminant a b c > 0

theorem quadratic_real_roots_range_k (k : ℝ) :
  has_two_distinct_real_roots k (-6) 9 ↔ k < 1 ∧ k ≠ 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_k_l2123_212335


namespace NUMINAMATH_GPT_simplify_neg_x_mul_3_minus_x_l2123_212381

theorem simplify_neg_x_mul_3_minus_x (x : ℝ) : -x * (3 - x) = -3 * x + x^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_neg_x_mul_3_minus_x_l2123_212381


namespace NUMINAMATH_GPT_sum_of_consecutive_numbers_with_lcm_168_l2123_212341

theorem sum_of_consecutive_numbers_with_lcm_168 (n : ℕ) (h_lcm : Nat.lcm (Nat.lcm n (n + 1)) (n + 2) = 168) : n + (n + 1) + (n + 2) = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_numbers_with_lcm_168_l2123_212341


namespace NUMINAMATH_GPT_g_four_times_of_three_l2123_212356

noncomputable def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 4 * x - 1

theorem g_four_times_of_three :
  g (g (g (g 3))) = 3 := by
  sorry

end NUMINAMATH_GPT_g_four_times_of_three_l2123_212356


namespace NUMINAMATH_GPT_trader_sells_cloth_l2123_212396

variable (x : ℝ) (SP_total : ℝ := 6900) (profit_per_meter : ℝ := 20) (CP_per_meter : ℝ := 66.25)

theorem trader_sells_cloth : SP_total = x * (CP_per_meter + profit_per_meter) → x = 80 :=
by
  intro h
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_trader_sells_cloth_l2123_212396


namespace NUMINAMATH_GPT_sum_of_pqrstu_l2123_212329

theorem sum_of_pqrstu (p q r s t : ℤ) (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -72) 
  (hpqrs : p ≠ q) (hnpr : p ≠ r) (hnps : p ≠ s) (hnpt : p ≠ t) (hnqr : q ≠ r) 
  (hnqs : q ≠ s) (hnqt : q ≠ t) (hnrs : r ≠ s) (hnrt : r ≠ t) (hnst : s ≠ t) : 
  p + q + r + s + t = 25 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_pqrstu_l2123_212329


namespace NUMINAMATH_GPT_root_constraints_between_zero_and_twoR_l2123_212343

variable (R l a : ℝ)
variable (hR : R > 0) (hl : l > 0) (ha_nonzero : a ≠ 0)

theorem root_constraints_between_zero_and_twoR :
  ∀ (x : ℝ), (2 * R * x^2 - (l^2 + 4 * a * R) * x + 2 * R * a^2 = 0) →
  (0 < x ∧ x < 2 * R) ↔
  (a > 0 ∧ a < 2 * R ∧ l^2 < (2 * R - a)^2) ∨
  (a < 0 ∧ -2 * R < a ∧ l^2 < (2 * R - a)^2) :=
sorry

end NUMINAMATH_GPT_root_constraints_between_zero_and_twoR_l2123_212343


namespace NUMINAMATH_GPT_course_count_l2123_212397

theorem course_count (n1 n2 : ℕ) (sum_x1 sum_x2 : ℕ) :
  (n1 = 6) →
  (sum_x1 = n1 * 100) →
  (sum_x2 = n2 * 50) →
  ((sum_x1 + sum_x2) / (n1 + n2) = 77) →
  n2 = 5 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_course_count_l2123_212397


namespace NUMINAMATH_GPT_yeast_counting_procedure_l2123_212394

def yeast_counting_conditions (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool) : Prop :=
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true

theorem yeast_counting_procedure :
  ∀ (counting_method : String) (shake_test_tube_needed : Bool) (dilution_needed : Bool),
  yeast_counting_conditions counting_method shake_test_tube_needed dilution_needed →
  counting_method = "Sampling inspection" ∧ 
  shake_test_tube_needed = true ∧ 
  dilution_needed = true :=
by
  intros counting_method shake_test_tube_needed dilution_needed h_condition
  exact h_condition

end NUMINAMATH_GPT_yeast_counting_procedure_l2123_212394


namespace NUMINAMATH_GPT_f_0_eq_0_l2123_212395

-- Define a function f with the given condition
def f (x : ℤ) : ℤ := if x = 0 then 0
                     else (x-1)^2 + 2*(x-1) + 1

-- State the theorem
theorem f_0_eq_0 : f 0 = 0 :=
by sorry

end NUMINAMATH_GPT_f_0_eq_0_l2123_212395


namespace NUMINAMATH_GPT_percentage_of_men_in_company_l2123_212368

theorem percentage_of_men_in_company 
  (M W : ℝ) 
  (h1 : 0.60 * M + 0.35 * W = 50) 
  (h2 : M + W = 100) : 
  M = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_men_in_company_l2123_212368


namespace NUMINAMATH_GPT_angle_PQR_correct_l2123_212390

-- Define the points and angles
variables {R P Q S : Type*}
variables (angle_RSQ angle_QSP angle_RQS angle_PQS : ℝ)

-- Define the conditions
def condition1 : Prop := true  -- RSP is a straight line implicitly means angle_RSQ + angle_QSP = 180
def condition2 : Prop := angle_QSP = 70
def condition3 (RS SQ : Type*) : Prop := true  -- Triangle RSQ is isosceles with RS = SQ
def condition4 (PS SQ : Type*) : Prop := true  -- Triangle PSQ is isosceles with PS = SQ

-- Define the isosceles triangle properties
def angle_RSQ_def : ℝ := 180 - angle_QSP
def angle_RQS_def : ℝ := 0.5 * (180 - angle_RSQ)
def angle_PQS_def : ℝ := 0.5 * (180 - angle_QSP)

-- Prove the main statement
theorem angle_PQR_correct : 
  (angle_RSQ = 110) →
  (angle_RQS = 35) →
  (angle_PQS = 55) →
  (angle_PQR : ℝ) = angle_PQS + angle_RQS :=
sorry

end NUMINAMATH_GPT_angle_PQR_correct_l2123_212390


namespace NUMINAMATH_GPT_work_completion_l2123_212315

theorem work_completion (W : ℕ) (a_rate b_rate combined_rate : ℕ) 
  (h1: combined_rate = W/8) 
  (h2: a_rate = W/12) 
  (h3: combined_rate = a_rate + b_rate) 
  : combined_rate = W/8 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_l2123_212315


namespace NUMINAMATH_GPT_factorize_a_cubed_minus_25a_l2123_212310

variable {a : ℝ}

theorem factorize_a_cubed_minus_25a (a : ℝ) : a^3 - 25 * a = a * (a + 5) * (a - 5) := 
by sorry

end NUMINAMATH_GPT_factorize_a_cubed_minus_25a_l2123_212310


namespace NUMINAMATH_GPT_problem_statement_l2123_212354

def y_and (y : ℤ) : ℤ := 9 - y
def and_y (y : ℤ) : ℤ := y - 9

theorem problem_statement : and_y (y_and 15) = -15 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2123_212354


namespace NUMINAMATH_GPT_fg_at_3_equals_97_l2123_212330

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_at_3_equals_97 : f (g 3) = 97 := by
  sorry

end NUMINAMATH_GPT_fg_at_3_equals_97_l2123_212330


namespace NUMINAMATH_GPT_complex_problem_l2123_212339

theorem complex_problem (a b : ℝ) (h : (⟨a, 3⟩ : ℂ) + ⟨2, -1⟩ = ⟨5, b⟩) : a * b = 6 := by
  sorry

end NUMINAMATH_GPT_complex_problem_l2123_212339


namespace NUMINAMATH_GPT_slope_of_line_l2123_212338

theorem slope_of_line (x y : ℝ) (h : x / 4 + y / 3 = 1) : ∀ m : ℝ, (y = m * x + 3) → m = -3/4 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_l2123_212338


namespace NUMINAMATH_GPT_road_unrepaired_is_42_percent_statement_is_false_l2123_212312

def road_length : ℝ := 1
def phase1_completion : ℝ := 0.40
def phase2_remaining_factor : ℝ := 0.30

def remaining_road (road : ℝ) (phase1 : ℝ) (phase2_factor : ℝ) : ℝ :=
  road - phase1 - (road - phase1) * phase2_factor

theorem road_unrepaired_is_42_percent (road_length : ℝ) (phase1_completion : ℝ) (phase2_remaining_factor : ℝ) :
  remaining_road road_length phase1_completion phase2_remaining_factor = 0.42 :=
by
  sorry

theorem statement_is_false : ¬(remaining_road road_length phase1_completion phase2_remaining_factor = 0.30) :=
by
  sorry

end NUMINAMATH_GPT_road_unrepaired_is_42_percent_statement_is_false_l2123_212312


namespace NUMINAMATH_GPT_exists_linear_eq_solution_x_2_l2123_212348

theorem exists_linear_eq_solution_x_2 : ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x : ℝ, a * x + b = 0 ↔ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_exists_linear_eq_solution_x_2_l2123_212348


namespace NUMINAMATH_GPT_workers_in_workshop_l2123_212375

theorem workers_in_workshop (W : ℕ) (h1 : W ≤ 100) (h2 : W % 3 = 0) (h3 : W % 25 = 0)
  : W = 75 ∧ W / 3 = 25 ∧ W * 8 / 100 = 6 :=
by
  sorry

end NUMINAMATH_GPT_workers_in_workshop_l2123_212375


namespace NUMINAMATH_GPT_bottle_caps_per_friend_l2123_212377

-- The context where Catherine has 18 bottle caps
def bottle_caps : Nat := 18

-- Catherine distributes these bottle caps among 6 friends
def number_of_friends : Nat := 6

-- We need to prove that each friend gets 3 bottle caps
theorem bottle_caps_per_friend : bottle_caps / number_of_friends = 3 :=
by sorry

end NUMINAMATH_GPT_bottle_caps_per_friend_l2123_212377


namespace NUMINAMATH_GPT_trapezoid_area_correct_l2123_212328

noncomputable def trapezoid_area (x : ℝ) : ℝ :=
  let base1 := 3 * x
  let base2 := 5 * x + 2
  (base1 + base2) / 2 * x

theorem trapezoid_area_correct (x : ℝ) : trapezoid_area x = 4 * x^2 + x :=
  by
  sorry

end NUMINAMATH_GPT_trapezoid_area_correct_l2123_212328


namespace NUMINAMATH_GPT_sum_of_adjacent_to_7_l2123_212398

/-- Define the divisors of 245, excluding 1 -/
def divisors245 : Set ℕ := {5, 7, 35, 49, 245}

/-- Define the adjacency condition to ensure every pair of adjacent integers has a common factor greater than 1 -/
def adjacency_condition (a b : ℕ) : Prop := (a ≠ b) ∨ (Nat.gcd a b > 1)

/-- Prove the sum of the two integers adjacent to 7 in the given condition is 294. -/
theorem sum_of_adjacent_to_7 (d1 d2 : ℕ) (h1 : d1 ∈ divisors245) (h2 : d2 ∈ divisors245) 
    (adj1 : adjacency_condition 7 d1) (adj2 : adjacency_condition 7 d2) : 
    d1 + d2 = 294 := 
sorry

end NUMINAMATH_GPT_sum_of_adjacent_to_7_l2123_212398


namespace NUMINAMATH_GPT_equations_have_different_graphs_l2123_212347

theorem equations_have_different_graphs :
  (∃ (x : ℝ), ∀ (y₁ y₂ y₃ : ℝ),
    (y₁ = x - 2) ∧
    (y₂ = (x^2 - 4) / (x + 2) ∧ x ≠ -2) ∧
    (y₃ = (x^2 - 4) / (x + 2) ∧ x ≠ -2 ∨ (x = -2 ∧ ∀ y₃ : ℝ, (x+2) * y₃ = x^2 - 4)))
  → (∃ y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∨ y₁ ≠ y₃ ∨ y₂ ≠ y₃) := sorry

end NUMINAMATH_GPT_equations_have_different_graphs_l2123_212347


namespace NUMINAMATH_GPT_maximize_revenue_l2123_212370

noncomputable def revenue (p : ℝ) : ℝ :=
p * (145 - 7 * p)

theorem maximize_revenue : ∃ p : ℕ, p ≤ 30 ∧ p = 10 ∧ ∀ q ≤ 30, revenue (q : ℝ) ≤ revenue 10 :=
by
  sorry

end NUMINAMATH_GPT_maximize_revenue_l2123_212370


namespace NUMINAMATH_GPT_transport_cost_correct_l2123_212371

-- Defining the weights of the sensor unit and communication module in grams
def weight_sensor_grams : ℕ := 500
def weight_comm_module_grams : ℕ := 1500

-- Defining the transport cost per kilogram
def cost_per_kg_sensor : ℕ := 25000
def cost_per_kg_comm_module : ℕ := 20000

-- Converting weights to kilograms
def weight_sensor_kg : ℚ := weight_sensor_grams / 1000
def weight_comm_module_kg : ℚ := weight_comm_module_grams / 1000

-- Calculating the transport costs
def cost_sensor : ℚ := weight_sensor_kg * cost_per_kg_sensor
def cost_comm_module : ℚ := weight_comm_module_kg * cost_per_kg_comm_module

-- Total cost of transporting both units
def total_cost : ℚ := cost_sensor + cost_comm_module

-- Proving that the total cost is $42500
theorem transport_cost_correct : total_cost = 42500 := by
  sorry

end NUMINAMATH_GPT_transport_cost_correct_l2123_212371


namespace NUMINAMATH_GPT_chord_length_of_tangent_circle_l2123_212380

theorem chord_length_of_tangent_circle
  (area_of_ring : ℝ)
  (diameter_large_circle : ℝ)
  (h1 : area_of_ring = (50 / 3) * Real.pi)
  (h2 : diameter_large_circle = 10) :
  ∃ (length_of_chord : ℝ), length_of_chord = (10 * Real.sqrt 6) / 3 := by
  sorry

end NUMINAMATH_GPT_chord_length_of_tangent_circle_l2123_212380


namespace NUMINAMATH_GPT_greatest_possible_difference_l2123_212323

theorem greatest_possible_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) : 
  ∃ d, d = y - x ∧ d = 6 := 
by
  sorry

end NUMINAMATH_GPT_greatest_possible_difference_l2123_212323


namespace NUMINAMATH_GPT_chord_eq_l2123_212386

/-- 
If a chord of the ellipse x^2 / 36 + y^2 / 9 = 1 is bisected by the point (4,2),
then the equation of the line on which this chord lies is x + 2y - 8 = 0.
-/
theorem chord_eq {x y : ℝ} (H : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 / 36 + A.2 ^ 2 / 9 = 1) ∧ 
  (B.1 ^ 2 / 36 + B.2 ^ 2 / 9 = 1) ∧ 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4, 2)) :
  x + 2 * y = 8 :=
sorry

end NUMINAMATH_GPT_chord_eq_l2123_212386


namespace NUMINAMATH_GPT_brick_length_proof_l2123_212389

-- Definitions based on conditions
def courtyard_length_m : ℝ := 18
def courtyard_width_m : ℝ := 16
def brick_width_cm : ℝ := 10
def total_bricks : ℝ := 14400

-- Conversion factors
def sqm_to_sqcm (area_sqm : ℝ) : ℝ := area_sqm * 10000
def courtyard_area_cm2 : ℝ := sqm_to_sqcm (courtyard_length_m * courtyard_width_m)

-- The proof statement
theorem brick_length_proof :
  (∀ (L : ℝ), courtyard_area_cm2 = total_bricks * (L * brick_width_cm)) → 
  (∃ (L : ℝ), L = 20) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_brick_length_proof_l2123_212389


namespace NUMINAMATH_GPT_hash_hash_hash_72_eq_12_5_l2123_212300

def hash (N : ℝ) : ℝ := 0.5 * N + 2

theorem hash_hash_hash_72_eq_12_5 : hash (hash (hash 72)) = 12.5 := 
by
  sorry

end NUMINAMATH_GPT_hash_hash_hash_72_eq_12_5_l2123_212300


namespace NUMINAMATH_GPT_find_constant_c_l2123_212393

theorem find_constant_c (c : ℝ) :
  (∀ x y : ℝ, x + y = c ∧ y - (2 + 5) / 2 = x - (8 + 11) / 2) →
  (c = 13) :=
by
  sorry

end NUMINAMATH_GPT_find_constant_c_l2123_212393


namespace NUMINAMATH_GPT_cubic_yard_to_cubic_meter_and_liters_l2123_212385

theorem cubic_yard_to_cubic_meter_and_liters :
  (1 : ℝ) * (0.9144 : ℝ)^3 = 0.764554 ∧ 0.764554 * 1000 = 764.554 :=
by
  sorry

end NUMINAMATH_GPT_cubic_yard_to_cubic_meter_and_liters_l2123_212385


namespace NUMINAMATH_GPT_perfect_square_condition_l2123_212387

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_condition_l2123_212387


namespace NUMINAMATH_GPT_value_of_expression_l2123_212365

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2006 = 2007 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l2123_212365


namespace NUMINAMATH_GPT_reflection_coefficient_l2123_212367

theorem reflection_coefficient (I_0 : ℝ) (I_4 : ℝ) (k : ℝ) 
  (h1 : I_4 = I_0 * (1 - k)^4) 
  (h2 : I_4 = I_0 / 256) : 
  k = 0.75 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_reflection_coefficient_l2123_212367


namespace NUMINAMATH_GPT_shortest_distance_between_circles_l2123_212334

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1) ^ 2 + (p₂.2 - p₁.2) ^ 2)

theorem shortest_distance_between_circles :
  let c₁ := (4, -3)
  let r₁ := 4
  let c₂ := (-5, 1)
  let r₂ := 1
  distance c₁ c₂ - (r₁ + r₂) = Real.sqrt 97 - 5 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_between_circles_l2123_212334


namespace NUMINAMATH_GPT_greatest_brownies_produced_l2123_212317

theorem greatest_brownies_produced (p side_length a b brownies : ℕ) :
  (4 * side_length = p) →
  (p = 40) →
  (brownies = side_length * side_length) →
  ((side_length - a - 2) * (side_length - b - 2) = 2 * (2 * (side_length - a) + 2 * (side_length - b) - 4)) →
  (a = 4) →
  (b = 4) →
  brownies = 100 :=
by
  intros h_perimeter h_perimeter_value h_brownies h_eq h_a h_b
  sorry

end NUMINAMATH_GPT_greatest_brownies_produced_l2123_212317


namespace NUMINAMATH_GPT_value_of_k_l2123_212379

theorem value_of_k :
  (∀ x : ℝ, x ^ 2 - x - 2 > 0 → 2 * x ^ 2 + (5 + 2 * k) * x + 5 * k < 0 → x = -2) ↔ -3 ≤ k ∧ k < 2 :=
sorry

end NUMINAMATH_GPT_value_of_k_l2123_212379


namespace NUMINAMATH_GPT_find_starting_number_l2123_212374

theorem find_starting_number (num_even_ints: ℕ) (end_num: ℕ) (h_num: num_even_ints = 35) (h_end: end_num = 95) : 
  ∃ start_num: ℕ, start_num = 24 ∧ (∀ n: ℕ, (start_num + 2 * n ≤ end_num ∧ n < num_even_ints)) := by
  sorry

end NUMINAMATH_GPT_find_starting_number_l2123_212374


namespace NUMINAMATH_GPT_largest_divisor_of_product_of_consecutive_evens_l2123_212321

theorem largest_divisor_of_product_of_consecutive_evens (n : ℤ) : 
  ∃ d, d = 8 ∧ ∀ n, d ∣ (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_product_of_consecutive_evens_l2123_212321


namespace NUMINAMATH_GPT_b_plus_c_is_square_l2123_212359

-- Given the conditions:
variables (a b c : ℕ)
variable (h1 : a > 0 ∧ b > 0 ∧ c > 0)  -- Condition 1: Positive integers
variable (h2 : Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1)  -- Condition 2: Pairwise relatively prime
variable (h3 : a % 2 = 1 ∧ c % 2 = 1)  -- Condition 3: a and c are odd
variable (h4 : a^2 + b^2 = c^2)  -- Condition 4: Pythagorean triple equation

-- Prove that b + c is the square of an integer
theorem b_plus_c_is_square : ∃ k : ℕ, b + c = k^2 :=
by
  sorry

end NUMINAMATH_GPT_b_plus_c_is_square_l2123_212359


namespace NUMINAMATH_GPT_find_w_l2123_212304

theorem find_w (a w : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * w) : w = 49 :=
by
  sorry

end NUMINAMATH_GPT_find_w_l2123_212304


namespace NUMINAMATH_GPT_find_j_l2123_212349

theorem find_j (n j : ℕ) (h_n_pos : n > 0) (h_j_pos : j > 0) (h_rem : n % j = 28) (h_div : n / j = 142 ∧ (↑n / ↑j : ℝ) = 142.07) : j = 400 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_j_l2123_212349


namespace NUMINAMATH_GPT_grain_remaining_l2123_212388

def originalGrain : ℕ := 50870
def spilledGrain : ℕ := 49952
def remainingGrain : ℕ := 918

theorem grain_remaining : originalGrain - spilledGrain = remainingGrain := by
  -- calculations are omitted in the theorem statement
  sorry

end NUMINAMATH_GPT_grain_remaining_l2123_212388


namespace NUMINAMATH_GPT_find_dividend_l2123_212364

variable (Divisor Quotient Remainder Dividend : ℕ)
variable (h₁ : Divisor = 15)
variable (h₂ : Quotient = 8)
variable (h₃ : Remainder = 5)

theorem find_dividend : Dividend = 125 ↔ Dividend = Divisor * Quotient + Remainder := by
  sorry

end NUMINAMATH_GPT_find_dividend_l2123_212364


namespace NUMINAMATH_GPT_right_triangle_leg_lengths_l2123_212399

theorem right_triangle_leg_lengths (a b c : ℕ) (h : a ^ 2 + b ^ 2 = c ^ 2) (h1: c = 17) (h2: a + (c - b) = 17) (h3: b + (c - a) = 17) : a = 8 ∧ b = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_right_triangle_leg_lengths_l2123_212399


namespace NUMINAMATH_GPT_solve_equation_l2123_212361

open Real

noncomputable def verify_solution (x : ℝ) : Prop :=
  1 / ((x - 3) * (x - 4)) +
  1 / ((x - 4) * (x - 5)) +
  1 / ((x - 5) * (x - 6)) = 1 / 8

theorem solve_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6) :
  verify_solution x ↔ (x = (9 + sqrt 57) / 2 ∨ x = (9 - sqrt 57) / 2) := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2123_212361


namespace NUMINAMATH_GPT_intersection_complement_l2123_212322

open Set

variable (U : Set ℕ) (P Q : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (hP : P = {1, 2, 3, 4, 5})
variable (hQ : Q = {3, 4, 5, 6, 7})

theorem intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l2123_212322


namespace NUMINAMATH_GPT_fillets_per_fish_l2123_212336

-- Definitions for the conditions
def fish_caught_per_day := 2
def days := 30
def total_fish_caught : Nat := fish_caught_per_day * days
def total_fish_fillets := 120

-- The proof problem statement
theorem fillets_per_fish (h1 : total_fish_caught = 60) (h2 : total_fish_fillets = 120) : 
  (total_fish_fillets / total_fish_caught) = 2 := sorry

end NUMINAMATH_GPT_fillets_per_fish_l2123_212336


namespace NUMINAMATH_GPT_second_number_is_650_l2123_212318

theorem second_number_is_650 (x : ℝ) (h1 : 0.20 * 1600 = 0.20 * x + 190) : x = 650 :=
by sorry

end NUMINAMATH_GPT_second_number_is_650_l2123_212318


namespace NUMINAMATH_GPT_integer_roots_of_polynomial_l2123_212373

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = 2 ∨ x = -3 ∨ x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_integer_roots_of_polynomial_l2123_212373


namespace NUMINAMATH_GPT_Joey_study_time_l2123_212363

theorem Joey_study_time :
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96 := by
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  show (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96
  -- define study times
  let weekday_hours_per_week := weekday_hours_per_night * nights_per_week
  let weekend_hours_per_week := weekend_hours_per_day * days_per_weekend
  -- sum times per week
  let total_hours_per_week := weekday_hours_per_week + weekend_hours_per_week
  -- multiply by weeks until exam
  let total_study_time := total_hours_per_week * weeks_until_exam
  have h : total_study_time = 96 := by sorry
  exact h

end NUMINAMATH_GPT_Joey_study_time_l2123_212363


namespace NUMINAMATH_GPT_g_h_2_eq_2175_l2123_212360

def g (x : ℝ) : ℝ := 2 * x^2 - 3
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem g_h_2_eq_2175 : g (h 2) = 2175 := by
  sorry

end NUMINAMATH_GPT_g_h_2_eq_2175_l2123_212360


namespace NUMINAMATH_GPT_original_bill_l2123_212333

theorem original_bill (m : ℝ) (h1 : 10 * (m / 10) = m)
                      (h2 : 9 * ((m - 10) / 10 + 3) = m - 10) :
  m = 180 :=
  sorry

end NUMINAMATH_GPT_original_bill_l2123_212333


namespace NUMINAMATH_GPT_fair_hair_women_percentage_l2123_212307

-- Definitions based on conditions
def total_employees (E : ℝ) := E
def women_with_fair_hair (E : ℝ) := 0.28 * E
def fair_hair_employees (E : ℝ) := 0.70 * E

-- Theorem to prove
theorem fair_hair_women_percentage (E : ℝ) (hE : E > 0) :
  (women_with_fair_hair E) / (fair_hair_employees E) * 100 = 40 :=
by 
  -- Sorry denotes the proof is omitted
  sorry

end NUMINAMATH_GPT_fair_hair_women_percentage_l2123_212307


namespace NUMINAMATH_GPT_correct_statement_l2123_212391

theorem correct_statement (x : ℝ) : 
  (∃ y : ℝ, y ≠ 0 ∧ y * x = 1 → x = 1 ∨ x = -1 ∨ x = 0) → false ∧
  (∃ y : ℝ, -y = y → y = 0 ∨ y = 1) → false ∧
  (abs x = x → x ≥ 0) → (x ^ 2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l2123_212391


namespace NUMINAMATH_GPT_smallest_t_circle_sin_l2123_212384

theorem smallest_t_circle_sin (t : ℝ) (h0 : 0 ≤ t) (h : ∀ θ, 0 ≤ θ ∧ θ ≤ t → ∃ k : ℤ, θ = (π/2 + 2 * π * k) ∨ θ = (3 * π / 2 + 2 * π * k)) : t = π :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_t_circle_sin_l2123_212384


namespace NUMINAMATH_GPT_find_a_l2123_212376

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x + a

theorem find_a (a : ℝ) :
  (∃! x : ℝ, f x a = 0) → (a = -2 ∨ a = 2) :=
sorry

end NUMINAMATH_GPT_find_a_l2123_212376


namespace NUMINAMATH_GPT_total_time_for_seven_flights_l2123_212382

theorem total_time_for_seven_flights :
  let a := 15
  let d := 8
  let n := 7
  let l := a + (n - 1) * d
  let S_n := n * (a + l) / 2
  S_n = 273 :=
by
  sorry

end NUMINAMATH_GPT_total_time_for_seven_flights_l2123_212382


namespace NUMINAMATH_GPT_solve_for_y_in_terms_of_x_l2123_212332

theorem solve_for_y_in_terms_of_x (x y : ℝ) (h : x - 2 = y + 3 * x) : y = -2 * x - 2 :=
sorry

end NUMINAMATH_GPT_solve_for_y_in_terms_of_x_l2123_212332


namespace NUMINAMATH_GPT_number_of_arrangements_l2123_212342

theorem number_of_arrangements (teams : Finset ℕ) (sites : Finset ℕ) :
  (∀ team, team ∈ teams → (team ∈ sites)) ∧ ((Finset.card sites = 3) ∧ (Finset.card teams = 6)) ∧ 
  (∃ (a b c : ℕ), a + b + c = 6 ∧ a >= 2 ∧ b >= 1 ∧ c >= 1) →
  ∃ (n : ℕ), n = 360 :=
sorry

end NUMINAMATH_GPT_number_of_arrangements_l2123_212342


namespace NUMINAMATH_GPT_f_value_at_3_l2123_212372

theorem f_value_at_3 (a b : ℝ) (h : (a * (-3)^3 - b * (-3) + 2 = -1)) : a * (3)^3 - b * 3 + 2 = 5 :=
sorry

end NUMINAMATH_GPT_f_value_at_3_l2123_212372
