import Mathlib

namespace NUMINAMATH_GPT_amount_paid_correct_l733_73370

def initial_debt : ℕ := 100
def hourly_wage : ℕ := 15
def hours_worked : ℕ := 4
def amount_paid_before_work : ℕ := initial_debt - (hourly_wage * hours_worked)

theorem amount_paid_correct : amount_paid_before_work = 40 := by
  sorry

end NUMINAMATH_GPT_amount_paid_correct_l733_73370


namespace NUMINAMATH_GPT_option_B_is_perfect_square_option_C_is_perfect_square_option_E_is_perfect_square_l733_73384

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Definitions of the given options as natural numbers
def A := 3^3 * 4^4 * 5^5
def B := 3^4 * 4^5 * 5^6
def C := 3^6 * 4^4 * 5^6
def D := 3^5 * 4^6 * 5^5
def E := 3^6 * 4^6 * 5^4

-- Lean statements for each option being a perfect square
theorem option_B_is_perfect_square : is_perfect_square B := sorry
theorem option_C_is_perfect_square : is_perfect_square C := sorry
theorem option_E_is_perfect_square : is_perfect_square E := sorry

end NUMINAMATH_GPT_option_B_is_perfect_square_option_C_is_perfect_square_option_E_is_perfect_square_l733_73384


namespace NUMINAMATH_GPT_prove_statement_II_l733_73368

variable (digit : ℕ)

def statement_I : Prop := (digit = 2)
def statement_II : Prop := (digit ≠ 3)
def statement_III : Prop := (digit = 5)
def statement_IV : Prop := (digit ≠ 6)

/- The main proposition that three statements are true and one is false. -/
def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop :=
  (s1 ∧ s2 ∧ s3 ∧ ¬s4) ∨ (s1 ∧ s2 ∧ ¬s3 ∧ s4) ∨ 
  (s1 ∧ ¬s2 ∧ s3 ∧ s4) ∨ (¬s1 ∧ s2 ∧ s3 ∧ s4)

theorem prove_statement_II : 
  (three_true_one_false (statement_I digit) (statement_II digit) (statement_III digit) (statement_IV digit)) → 
  statement_II digit :=
sorry

end NUMINAMATH_GPT_prove_statement_II_l733_73368


namespace NUMINAMATH_GPT_combined_tickets_l733_73300

-- Definitions from the conditions
def dave_spent : Nat := 43
def dave_left : Nat := 55
def alex_spent : Nat := 65
def alex_left : Nat := 42

-- Theorem to prove that the combined starting tickets of Dave and Alex is 205
theorem combined_tickets : dave_spent + dave_left + alex_spent + alex_left = 205 := 
by
  sorry

end NUMINAMATH_GPT_combined_tickets_l733_73300


namespace NUMINAMATH_GPT_parabola_above_line_l733_73352

variable (a b c : ℝ) (h : (b - c)^2 - 4 * a * c < 0)

theorem parabola_above_line : (b - c)^2 - 4 * a * c < 0 → (b - c)^2 - 4 * c * (a + b) < 0 :=
by sorry

end NUMINAMATH_GPT_parabola_above_line_l733_73352


namespace NUMINAMATH_GPT_total_marbles_l733_73330

theorem total_marbles (jars clay_pots total_marbles jars_marbles pots_marbles : ℕ)
  (h1 : jars = 16)
  (h2 : jars = 2 * clay_pots)
  (h3 : jars_marbles = 5)
  (h4 : pots_marbles = 3 * jars_marbles)
  (h5 : total_marbles = jars * jars_marbles + clay_pots * pots_marbles) :
  total_marbles = 200 := by
  sorry

end NUMINAMATH_GPT_total_marbles_l733_73330


namespace NUMINAMATH_GPT_factor_polynomial_l733_73326

-- Define the polynomial expression
def polynomial (x : ℝ) : ℝ := 60 * x + 45 + 9 * x ^ 2

-- Define the factored form of the polynomial
def factored_form (x : ℝ) : ℝ := 3 * (3 * x + 5) * (x + 3)

-- The statement of the problem to prove equivalence of the forms
theorem factor_polynomial : ∀ x : ℝ, polynomial x = factored_form x :=
by
  -- The actual proof is omitted and replaced by sorry
  sorry

end NUMINAMATH_GPT_factor_polynomial_l733_73326


namespace NUMINAMATH_GPT_zoey_holidays_in_a_year_l733_73346

-- Definitions based on the conditions
def holidays_per_month := 2
def months_in_year := 12

-- Lean statement representing the proof problem
theorem zoey_holidays_in_a_year : (holidays_per_month * months_in_year) = 24 :=
by sorry

end NUMINAMATH_GPT_zoey_holidays_in_a_year_l733_73346


namespace NUMINAMATH_GPT_convert_to_canonical_form_l733_73374

def quadratic_eqn (x y : ℝ) : ℝ :=
  8 * x^2 + 4 * x * y + 5 * y^2 - 56 * x - 32 * y + 80

def canonical_form (x2 y2 : ℝ) : Prop :=
  (x2^2 / 4) + (y2^2 / 9) = 1

theorem convert_to_canonical_form (x y : ℝ) :
  quadratic_eqn x y = 0 → ∃ (x2 y2 : ℝ), canonical_form x2 y2 :=
sorry

end NUMINAMATH_GPT_convert_to_canonical_form_l733_73374


namespace NUMINAMATH_GPT_remainder_3_pow_1000_mod_7_l733_73348

theorem remainder_3_pow_1000_mod_7 : 3 ^ 1000 % 7 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_1000_mod_7_l733_73348


namespace NUMINAMATH_GPT_find_radius_of_smaller_circles_l733_73314

noncomputable def smaller_circle_radius (r : ℝ) : Prop :=
  ∃ sin72 : ℝ, sin72 = Real.sin (72 * Real.pi / 180) ∧
  r = (2 * sin72) / (1 - sin72)

theorem find_radius_of_smaller_circles (r : ℝ) :
  (smaller_circle_radius r) ↔
  r = (2 * Real.sin (72 * Real.pi / 180)) / (1 - Real.sin (72 * Real.pi / 180)) :=
by
  sorry

end NUMINAMATH_GPT_find_radius_of_smaller_circles_l733_73314


namespace NUMINAMATH_GPT_certain_number_l733_73306

theorem certain_number (x certain_number : ℕ) (h1 : x = 3327) (h2 : 9873 + x = certain_number) : 
  certain_number = 13200 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_l733_73306


namespace NUMINAMATH_GPT_max_n_for_Sn_neg_l733_73329

noncomputable def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_n_for_Sn_neg (a : ℕ → ℝ) (h1 : ∀ n : ℕ, (n + 1) * Sn n a < n * Sn (n + 1) a)
  (h2 : a 8 / a 7 < -1) :
  ∀ n : ℕ, S_13 < 0 ∧ S_14 > 0 →
  ∀ m : ℕ, m > 13 → Sn m a ≥ 0 :=
sorry

end NUMINAMATH_GPT_max_n_for_Sn_neg_l733_73329


namespace NUMINAMATH_GPT_satisify_absolute_value_inequality_l733_73308

theorem satisify_absolute_value_inequality :
  ∃ (t : Finset ℤ), t.card = 2 ∧ ∀ y ∈ t, |7 * y + 4| ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_satisify_absolute_value_inequality_l733_73308


namespace NUMINAMATH_GPT_fraction_of_shaded_circle_l733_73321

theorem fraction_of_shaded_circle (total_regions shaded_regions : ℕ) (h1 : total_regions = 4) (h2 : shaded_regions = 1) :
  shaded_regions / total_regions = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_fraction_of_shaded_circle_l733_73321


namespace NUMINAMATH_GPT_solve_n_is_2_l733_73375

noncomputable def problem_statement (n : ℕ) : Prop :=
  ∃ m : ℕ, 9 * n^2 + 5 * n - 26 = m * (m + 1)

theorem solve_n_is_2 : problem_statement 2 :=
  sorry

end NUMINAMATH_GPT_solve_n_is_2_l733_73375


namespace NUMINAMATH_GPT_angle_in_second_quadrant_l733_73324

open Real

-- Define the fourth quadrant condition
def isFourthQuadrant (α : ℝ) (k : ℤ) : Prop :=
  2 * k * π - π / 2 < α ∧ α < 2 * k * π

-- Define the second quadrant condition
def isSecondQuadrant (β : ℝ) (k : ℤ) : Prop :=
  2 * k * π + π / 2 < β ∧ β < 2 * k * π + π

-- The main theorem to prove
theorem angle_in_second_quadrant (α : ℝ) (k : ℤ) :
  isFourthQuadrant α k → isSecondQuadrant (π + α) k :=
sorry

end NUMINAMATH_GPT_angle_in_second_quadrant_l733_73324


namespace NUMINAMATH_GPT_bald_eagle_pairs_l733_73355

theorem bald_eagle_pairs (n_1963 : ℕ) (increase : ℕ) (h1 : n_1963 = 417) (h2 : increase = 6649) :
  (n_1963 + increase = 7066) :=
by
  sorry

end NUMINAMATH_GPT_bald_eagle_pairs_l733_73355


namespace NUMINAMATH_GPT_lcm_924_660_eq_4620_l733_73389

theorem lcm_924_660_eq_4620 : Nat.lcm 924 660 = 4620 := 
by
  sorry

end NUMINAMATH_GPT_lcm_924_660_eq_4620_l733_73389


namespace NUMINAMATH_GPT_valid_triangle_inequality_l733_73305

theorem valid_triangle_inequality (a : ℝ) 
  (h1 : 4 + 6 > a) 
  (h2 : 4 + a > 6) 
  (h3 : 6 + a > 4) : 
  a = 5 :=
sorry

end NUMINAMATH_GPT_valid_triangle_inequality_l733_73305


namespace NUMINAMATH_GPT_arithmetic_mean_of_two_digit_multiples_of_9_l733_73399

theorem arithmetic_mean_of_two_digit_multiples_of_9 :
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  M = 58.5 :=
by
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_two_digit_multiples_of_9_l733_73399


namespace NUMINAMATH_GPT_simplify_expression_l733_73361

variable (a b c : ℝ)

theorem simplify_expression :
  (-32 * a^4 * b^5 * c) / ((-2 * a * b)^3) * (-3 / 4 * a * c) = -3 * a^2 * b^2 * c^2 :=
  by
    sorry

end NUMINAMATH_GPT_simplify_expression_l733_73361


namespace NUMINAMATH_GPT_shortest_distance_parabola_line_l733_73350

theorem shortest_distance_parabola_line :
  ∃ (P Q : ℝ × ℝ), P.2 = P.1^2 - 6 * P.1 + 15 ∧ Q.2 = 2 * Q.1 - 7 ∧
  ∀ (p q : ℝ × ℝ), p.2 = p.1^2 - 6 * p.1 + 15 → q.2 = 2 * q.1 - 7 → 
  dist p q ≥ dist P Q :=
sorry

end NUMINAMATH_GPT_shortest_distance_parabola_line_l733_73350


namespace NUMINAMATH_GPT_numPythagoreanTriples_l733_73338

def isPythagoreanTriple (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ x^2 + y^2 = z^2

theorem numPythagoreanTriples (n : ℕ) : ∃! T : (ℕ × ℕ × ℕ) → Prop, 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (T (2^(n+1))) :=
sorry

end NUMINAMATH_GPT_numPythagoreanTriples_l733_73338


namespace NUMINAMATH_GPT_pradeep_marks_l733_73396

-- Conditions as definitions
def passing_percentage : ℝ := 0.35
def max_marks : ℕ := 600
def fail_difference : ℕ := 25

def passing_marks (total_marks : ℕ) (percentage : ℝ) : ℝ :=
  percentage * total_marks

def obtained_marks (passing_marks : ℝ) (difference : ℕ) : ℝ :=
  passing_marks - difference

-- Theorem statement
theorem pradeep_marks : obtained_marks (passing_marks max_marks passing_percentage) fail_difference = 185 := by
  sorry

end NUMINAMATH_GPT_pradeep_marks_l733_73396


namespace NUMINAMATH_GPT_consecutive_numbers_count_l733_73332

theorem consecutive_numbers_count (n : ℕ) 
(avg : ℝ) 
(largest : ℕ) 
(h_avg : avg = 20) 
(h_largest : largest = 23) 
(h_eq : (largest + (largest - (n - 1))) / 2 = avg) : 
n = 7 := 
by 
  sorry

end NUMINAMATH_GPT_consecutive_numbers_count_l733_73332


namespace NUMINAMATH_GPT_Jungkook_red_balls_count_l733_73309

-- Define the conditions
def red_balls_per_box : ℕ := 3
def boxes_Jungkook_has : ℕ := 2

-- Statement to prove
theorem Jungkook_red_balls_count : red_balls_per_box * boxes_Jungkook_has = 6 :=
by sorry

end NUMINAMATH_GPT_Jungkook_red_balls_count_l733_73309


namespace NUMINAMATH_GPT_mail_total_correct_l733_73394

def Monday_mail : ℕ := 65
def Tuesday_mail : ℕ := Monday_mail + 10
def Wednesday_mail : ℕ := Tuesday_mail - 5
def Thursday_mail : ℕ := Wednesday_mail + 15
def total_mail : ℕ := Monday_mail + Tuesday_mail + Wednesday_mail + Thursday_mail

theorem mail_total_correct : total_mail = 295 := by
  sorry

end NUMINAMATH_GPT_mail_total_correct_l733_73394


namespace NUMINAMATH_GPT_find_tire_price_l733_73335

def regular_price_of_tire (x : ℝ) : Prop :=
  3 * x + 0.75 * x = 270

theorem find_tire_price (x : ℝ) (h1 : regular_price_of_tire x) : x = 72 :=
by
  sorry

end NUMINAMATH_GPT_find_tire_price_l733_73335


namespace NUMINAMATH_GPT_negation_of_p_l733_73362

theorem negation_of_p (p : Prop) :
  (¬ (∀ (a : ℝ), a ≥ 0 → a^4 + a^2 ≥ 0)) ↔ (∃ (a : ℝ), a ≥ 0 ∧ a^4 + a^2 < 0) := 
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l733_73362


namespace NUMINAMATH_GPT_function_is_one_l733_73397

noncomputable def f : ℝ → ℝ := sorry

theorem function_is_one (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x*y) + f (x*z) ≥ 1 + f (x) * f (y*z))
  : ∀ x : ℝ, f x = 1 :=
sorry

end NUMINAMATH_GPT_function_is_one_l733_73397


namespace NUMINAMATH_GPT_broccoli_area_l733_73395

/--
A farmer grows broccoli in a square-shaped farm. This year, he produced 2601 broccoli,
which is 101 more than last year. The shape of the area used for growing the broccoli 
has remained square in both years. Assuming each broccoli takes up an equal amount of 
area, prove that each broccoli takes up 1 square unit of area.
-/
theorem broccoli_area (x y : ℕ) 
  (h1 : y^2 = x^2 + 101) 
  (h2 : y^2 = 2601) : 
  1 = 1 := 
sorry

end NUMINAMATH_GPT_broccoli_area_l733_73395


namespace NUMINAMATH_GPT_arithmetic_seq_problem_l733_73360

theorem arithmetic_seq_problem (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_problem_l733_73360


namespace NUMINAMATH_GPT_andrew_eggs_count_l733_73359

def cost_of_toast (num_toasts : ℕ) : ℕ :=
  num_toasts * 1

def cost_of_eggs (num_eggs : ℕ) : ℕ :=
  num_eggs * 3

def total_cost (num_toasts : ℕ) (num_eggs : ℕ) : ℕ :=
  cost_of_toast num_toasts + cost_of_eggs num_eggs

theorem andrew_eggs_count (E : ℕ) (H1 : total_cost 2 2 = 8)
                       (H2 : total_cost 1 E + 8 = 15) : E = 2 := by
  sorry

end NUMINAMATH_GPT_andrew_eggs_count_l733_73359


namespace NUMINAMATH_GPT_quad_common_root_l733_73356

theorem quad_common_root (a b c d : ℝ) :
  (∃ α : ℝ, α^2 + a * α + b = 0 ∧ α^2 + c * α + d = 0) ↔ (a * d - b * c) * (c - a) = (b - d)^2 ∧ (a ≠ c) := 
sorry

end NUMINAMATH_GPT_quad_common_root_l733_73356


namespace NUMINAMATH_GPT_ab_value_l733_73337

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 240) :
  a * b = 255 :=
sorry

end NUMINAMATH_GPT_ab_value_l733_73337


namespace NUMINAMATH_GPT_friends_count_l733_73372

-- Define the conditions
def num_kids : ℕ := 2
def shonda_present : Prop := True  -- Shonda is present, we may just incorporate it as part of count for clarity
def num_adults : ℕ := 7
def num_baskets : ℕ := 15
def eggs_per_basket : ℕ := 12
def eggs_per_person : ℕ := 9

-- Define the total number of eggs
def total_eggs : ℕ := num_baskets * eggs_per_basket

-- Define the total number of people
def total_people : ℕ := total_eggs / eggs_per_person

-- Define the number of known people (Shonda, her kids, and the other adults)
def known_people : ℕ := num_kids + 1 + num_adults  -- 1 represents Shonda

-- Define the number of friends
def num_friends : ℕ := total_people - known_people

-- The theorem we need to prove
theorem friends_count : num_friends = 10 :=
by
  sorry

end NUMINAMATH_GPT_friends_count_l733_73372


namespace NUMINAMATH_GPT_Jake_has_one_more_balloon_than_Allan_l733_73398

-- Defining the given values
def A : ℕ := 6
def J_initial : ℕ := 3
def J_buy : ℕ := 4
def J_total : ℕ := J_initial + J_buy

-- The theorem statement
theorem Jake_has_one_more_balloon_than_Allan : J_total - A = 1 := 
by
  sorry -- proof goes here

end NUMINAMATH_GPT_Jake_has_one_more_balloon_than_Allan_l733_73398


namespace NUMINAMATH_GPT_sequence_inequality_l733_73304

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

noncomputable def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) - b n = b 1 - b 0

theorem sequence_inequality
  (ha : ∀ n, 0 < a n)
  (hg : is_geometric a q)
  (ha6_eq_b7 : a 6 = b 7)
  (hb : is_arithmetic b) :
  a 3 + a 9 ≥ b 4 + b 10 :=
by
  sorry

end NUMINAMATH_GPT_sequence_inequality_l733_73304


namespace NUMINAMATH_GPT_packs_in_each_set_l733_73385

variable (cost_per_set cost_per_pack total_savings : ℝ)
variable (x : ℕ)

-- Objecting conditions
axiom cost_set : cost_per_set = 2.5
axiom cost_pack : cost_per_pack = 1.3
axiom savings : total_savings = 1

-- Main proof problem
theorem packs_in_each_set :
  10 * x * cost_per_pack = 10 * cost_per_set + total_savings → x = 2 :=
by
  -- sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_packs_in_each_set_l733_73385


namespace NUMINAMATH_GPT_paul_has_5point86_left_l733_73381

noncomputable def paulLeftMoney : ℝ := 15 - (2 + (3 - 0.1*3) + 2*2 + 0.05 * (2 + (3 - 0.1*3) + 2*2))

theorem paul_has_5point86_left :
  paulLeftMoney = 5.86 :=
by
  sorry

end NUMINAMATH_GPT_paul_has_5point86_left_l733_73381


namespace NUMINAMATH_GPT_lines_are_skew_iff_l733_73379

def line1 (s : ℝ) (b : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3 * s, 3 + 4 * s, b + 5 * s)

def line2 (v : ℝ) : ℝ × ℝ × ℝ :=
  (5 + 6 * v, 2 + 3 * v, 1 + 2 * v)

def lines_intersect (s v b : ℝ) : Prop :=
  line1 s b = line2 v

theorem lines_are_skew_iff (b : ℝ) : ¬ (∃ s v, lines_intersect s v b) ↔ b ≠ 9 :=
by
  sorry

end NUMINAMATH_GPT_lines_are_skew_iff_l733_73379


namespace NUMINAMATH_GPT_andrew_stickers_now_l733_73364

-- Defining the conditions
def total_stickers : Nat := 1500
def ratio_susan : Nat := 1
def ratio_andrew : Nat := 1
def ratio_sam : Nat := 3
def total_ratio : Nat := ratio_susan + ratio_andrew + ratio_sam
def part : Nat := total_stickers / total_ratio
def susan_share : Nat := ratio_susan * part
def andrew_share_initial : Nat := ratio_andrew * part
def sam_share : Nat := ratio_sam * part
def sam_to_andrew : Nat := (2 * sam_share) / 3

-- Andrew's final stickers count
def andrew_share_final : Nat :=
  andrew_share_initial + sam_to_andrew

-- The theorem to prove
theorem andrew_stickers_now : andrew_share_final = 900 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_andrew_stickers_now_l733_73364


namespace NUMINAMATH_GPT_avg_weight_a_b_l733_73315

theorem avg_weight_a_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 60)
  (h2 : (B + C) / 2 = 50)
  (h3 : B = 60) :
  (A + B) / 2 = 70 := 
sorry

end NUMINAMATH_GPT_avg_weight_a_b_l733_73315


namespace NUMINAMATH_GPT_cone_csa_l733_73392

theorem cone_csa (r l : ℝ) (h_r : r = 8) (h_l : l = 18) : 
  (Real.pi * r * l) = 144 * Real.pi :=
by 
  rw [h_r, h_l]
  norm_num
  sorry

end NUMINAMATH_GPT_cone_csa_l733_73392


namespace NUMINAMATH_GPT_faith_work_days_per_week_l733_73390

theorem faith_work_days_per_week 
  (hourly_wage : ℝ)
  (normal_hours_per_day : ℝ)
  (overtime_hours_per_day : ℝ)
  (weekly_earnings : ℝ)
  (overtime_rate_multiplier : ℝ) :
  hourly_wage = 13.50 → 
  normal_hours_per_day = 8 → 
  overtime_hours_per_day = 2 → 
  weekly_earnings = 675 →
  overtime_rate_multiplier = 1.5 →
  ∀ days_per_week : ℝ, days_per_week = 5 :=
sorry

end NUMINAMATH_GPT_faith_work_days_per_week_l733_73390


namespace NUMINAMATH_GPT_domain_of_h_l733_73325

noncomputable def h (x : ℝ) : ℝ := (x^4 - 5 * x + 6) / (|x - 4| + |x + 2| - 1)

theorem domain_of_h : ∀ x : ℝ, |x - 4| + |x + 2| - 1 ≠ 0 := by
  intro x
  sorry

end NUMINAMATH_GPT_domain_of_h_l733_73325


namespace NUMINAMATH_GPT_log_sum_nine_l733_73340

-- Define that {a_n} is a geometric sequence and satisfies the given conditions.
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a n = a 1 * r ^ (n - 1)

-- Given conditions
axiom a_pos (a : ℕ → ℝ) : (∀ n, a n > 0)      -- All terms are positive
axiom a2a8_eq_4 (a : ℕ → ℝ) : a 2 * a 8 = 4    -- a₂a₈ = 4

theorem log_sum_nine (a : ℕ → ℝ) 
  (geo_seq : geometric_sequence a) 
  (pos : ∀ n, a n > 0)
  (eq4 : a 2 * a 8 = 4) :
  (Real.logb 2 (a 1) + Real.logb 2 (a 2) + Real.logb 2 (a 3) + Real.logb 2 (a 4)
  + Real.logb 2 (a 5) + Real.logb 2 (a 6) + Real.logb 2 (a 7) + Real.logb 2 (a 8)
  + Real.logb 2 (a 9)) = 9 :=
by
  sorry

end NUMINAMATH_GPT_log_sum_nine_l733_73340


namespace NUMINAMATH_GPT_vanya_correct_answers_l733_73344

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end NUMINAMATH_GPT_vanya_correct_answers_l733_73344


namespace NUMINAMATH_GPT_quadratic_solution_property_l733_73336

theorem quadratic_solution_property (p q : ℝ)
  (h : ∀ x, 2 * x^2 + 8 * x - 42 = 0 → x = p ∨ x = q) :
  (p - q + 2) ^ 2 = 144 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_property_l733_73336


namespace NUMINAMATH_GPT_average_speed_l733_73307

-- Define the conditions as constants and theorems
def distance1 : ℝ := 240
def distance2 : ℝ := 420
def time_diff : ℝ := 3

theorem average_speed : ∃ v t : ℝ, distance1 = v * t ∧ distance2 = v * (t + time_diff) → v = 60 := 
by
  sorry

end NUMINAMATH_GPT_average_speed_l733_73307


namespace NUMINAMATH_GPT_compare_solutions_l733_73343

variables (p q r s : ℝ)
variables (hp : p ≠ 0) (hr : r ≠ 0)

theorem compare_solutions :
  ((-q / p) > (-s / r)) ↔ (s * r > q * p) :=
by sorry

end NUMINAMATH_GPT_compare_solutions_l733_73343


namespace NUMINAMATH_GPT_jamal_books_remaining_l733_73342

variable (initial_books : ℕ := 51)
variable (history_books : ℕ := 12)
variable (fiction_books : ℕ := 19)
variable (children_books : ℕ := 8)
variable (misplaced_books : ℕ := 4)

theorem jamal_books_remaining : 
  initial_books - history_books - fiction_books - children_books + misplaced_books = 16 := by
  sorry

end NUMINAMATH_GPT_jamal_books_remaining_l733_73342


namespace NUMINAMATH_GPT_men_dropped_out_l733_73366

theorem men_dropped_out (x : ℕ) : 
  (∀ (days_half days_full men men_remaining : ℕ),
    days_half = 15 ∧ days_full = 25 ∧ men = 5 ∧ men_remaining = men - x ∧ 
    (men * (2 * days_half)) = ((men_remaining) * days_full)) -> x = 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_men_dropped_out_l733_73366


namespace NUMINAMATH_GPT_gcd_36_60_l733_73383

theorem gcd_36_60 : Int.gcd 36 60 = 12 := by
  sorry

end NUMINAMATH_GPT_gcd_36_60_l733_73383


namespace NUMINAMATH_GPT_eval_expression_l733_73365

theorem eval_expression : 15 * 30 + 45 * 15 - 15 * 10 = 975 :=
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l733_73365


namespace NUMINAMATH_GPT_weekly_earnings_l733_73328

theorem weekly_earnings :
  let hours_Monday := 2
  let minutes_Tuesday := 75
  let start_Thursday := (15, 10) -- 3:10 PM in (hour, minute) format
  let end_Thursday := (17, 45) -- 5:45 PM in (hour, minute) format
  let minutes_Saturday := 45

  let pay_rate_weekday := 4 -- \$4 per hour
  let pay_rate_weekend := 5 -- \$5 per hour

  -- Convert time to hours
  let hours_Tuesday := minutes_Tuesday / 60.0
  let Thursday_work_minutes := (end_Thursday.1 * 60 + end_Thursday.2) - (start_Thursday.1 * 60 + start_Thursday.2)
  let hours_Thursday := Thursday_work_minutes / 60.0
  let hours_Saturday := minutes_Saturday / 60.0

  -- Calculate earnings
  let earnings_Monday := hours_Monday * pay_rate_weekday
  let earnings_Tuesday := hours_Tuesday * pay_rate_weekday
  let earnings_Thursday := hours_Thursday * pay_rate_weekday
  let earnings_Saturday := hours_Saturday * pay_rate_weekend

  -- Total earnings
  let total_earnings := earnings_Monday + earnings_Tuesday + earnings_Thursday + earnings_Saturday

  total_earnings = 27.08 := by sorry

end NUMINAMATH_GPT_weekly_earnings_l733_73328


namespace NUMINAMATH_GPT_inverse_function_property_l733_73358

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a ^ x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_property (a : ℝ) (h : g a 2 = 4) : f a 2 = 1 := by
  have g_inverse_f : g a (f a 2) = 2 := by sorry
  have a_value : a = 2 := by sorry
  rw [a_value]
  sorry

end NUMINAMATH_GPT_inverse_function_property_l733_73358


namespace NUMINAMATH_GPT_percentage_decrease_in_area_l733_73339

noncomputable def original_radius (r : ℝ) : ℝ := r
noncomputable def new_radius (r : ℝ) : ℝ := 0.5 * r
noncomputable def original_area (r : ℝ) : ℝ := Real.pi * r ^ 2
noncomputable def new_area (r : ℝ) : ℝ := Real.pi * (0.5 * r) ^ 2

theorem percentage_decrease_in_area (r : ℝ) (hr : 0 ≤ r) :
  ((original_area r - new_area r) / original_area r) * 100 = 75 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_in_area_l733_73339


namespace NUMINAMATH_GPT_remaining_volume_of_cube_l733_73345

theorem remaining_volume_of_cube :
  let s := 6
  let r := 3
  let h := 6
  let V_cube := s^3
  let V_cylinder := Real.pi * (r^2) * h
  V_cube - V_cylinder = 216 - 54 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_remaining_volume_of_cube_l733_73345


namespace NUMINAMATH_GPT_certain_amount_l733_73322

theorem certain_amount (n : ℤ) (x : ℤ) : n = 5 ∧ 7 * n - 15 = 2 * n + x → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_certain_amount_l733_73322


namespace NUMINAMATH_GPT_ice_cream_cost_l733_73341

variable {x F M : ℤ}

theorem ice_cream_cost (h1 : F = x - 7) (h2 : M = x - 1) (h3 : F + M < x) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_cost_l733_73341


namespace NUMINAMATH_GPT_elizabeth_net_profit_l733_73316

theorem elizabeth_net_profit :
  let cost_per_bag := 3.00
  let num_bags := 20
  let price_first_15_bags := 6.00
  let price_last_5_bags := 4.00
  let total_cost := cost_per_bag * num_bags
  let revenue_first_15 := 15 * price_first_15_bags
  let revenue_last_5 := 5 * price_last_5_bags
  let total_revenue := revenue_first_15 + revenue_last_5
  let net_profit := total_revenue - total_cost
  net_profit = 50.00 :=
by
  sorry

end NUMINAMATH_GPT_elizabeth_net_profit_l733_73316


namespace NUMINAMATH_GPT_area_of_triangle_PQR_l733_73310

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 5 }

-- Define the lines using their slopes and the point P
def line1 (x : ℝ) : ℝ := -x + 7
def line2 (x : ℝ) : ℝ := -2 * x + 9

-- Definitions of points Q and R, which are the x-intercepts
def Q : Point := { x := 7, y := 0 }
def R : Point := { x := 4.5, y := 0 }

-- Theorem statement
theorem area_of_triangle_PQR : 
  let base := 7 - 4.5
  let height := 5
  (1 / 2) * base * height = 6.25 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_PQR_l733_73310


namespace NUMINAMATH_GPT_correct_calculation_l733_73357

theorem correct_calculation (x : ℤ) (h1 : x + 65 = 125) : x + 95 = 155 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l733_73357


namespace NUMINAMATH_GPT_group_scores_analysis_l733_73373

def group1_scores : List ℕ := [92, 90, 91, 96, 96]
def group2_scores : List ℕ := [92, 96, 90, 95, 92]

def median (l : List ℕ) : ℕ := sorry
def mode (l : List ℕ) : ℕ := sorry
def mean (l : List ℕ) : ℕ := sorry
def variance (l : List ℕ) : ℕ := sorry

theorem group_scores_analysis :
  median group2_scores = 92 ∧
  mode group1_scores = 96 ∧
  mean group2_scores = 93 ∧
  variance group1_scores = 64 / 10 ∧
  variance group2_scores = 48 / 10 ∧
  variance group2_scores < variance group1_scores :=
by
  sorry

end NUMINAMATH_GPT_group_scores_analysis_l733_73373


namespace NUMINAMATH_GPT_not_enough_pharmacies_l733_73351

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end NUMINAMATH_GPT_not_enough_pharmacies_l733_73351


namespace NUMINAMATH_GPT_percentage_students_with_same_grade_l733_73313

def total_students : ℕ := 50
def students_with_same_grade : ℕ := 3 + 6 + 8 + 2 + 1

theorem percentage_students_with_same_grade :
  (students_with_same_grade / total_students : ℚ) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_students_with_same_grade_l733_73313


namespace NUMINAMATH_GPT_value_of_a_minus_b_l733_73319

theorem value_of_a_minus_b (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 13) (h3 : a * b > 0) : a - b = -10 ∨ a - b = 10 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l733_73319


namespace NUMINAMATH_GPT_ratio_area_rectangle_triangle_l733_73347

noncomputable def area_rectangle (L W : ℝ) : ℝ :=
  L * W

noncomputable def area_triangle (L W : ℝ) : ℝ :=
  (1 / 2) * L * W

theorem ratio_area_rectangle_triangle (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  area_rectangle L W / area_triangle L W = 2 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end NUMINAMATH_GPT_ratio_area_rectangle_triangle_l733_73347


namespace NUMINAMATH_GPT_scout_troop_profit_l733_73311

-- Defining the basic conditions as Lean definitions
def num_bars : ℕ := 1500
def cost_rate : ℚ := 3 / 4 -- rate in dollars per bar
def sell_rate : ℚ := 2 / 3 -- rate in dollars per bar

-- Calculate total cost, total revenue, and profit
def total_cost : ℚ := num_bars * cost_rate
def total_revenue : ℚ := num_bars * sell_rate
def profit : ℚ := total_revenue - total_cost

-- The final theorem to be proved
theorem scout_troop_profit : profit = -125 := by
  sorry

end NUMINAMATH_GPT_scout_troop_profit_l733_73311


namespace NUMINAMATH_GPT_second_train_catches_first_l733_73380

-- Define the starting times and speeds
def t1_start_time := 14 -- 2:00 pm in 24-hour format
def t1_speed := 70 -- km/h
def t2_start_time := 15 -- 3:00 pm in 24-hour format
def t2_speed := 80 -- km/h

-- Define the time at which the second train catches the first train
def catch_time := 22 -- 10:00 pm in 24-hour format

theorem second_train_catches_first :
  ∃ t : ℕ, t = catch_time ∧
    t1_speed * ((t - t1_start_time) + 1) = t2_speed * (t - t2_start_time) := by
  sorry

end NUMINAMATH_GPT_second_train_catches_first_l733_73380


namespace NUMINAMATH_GPT_find_x_plus_y_l733_73303

theorem find_x_plus_y (x y : ℝ) 
  (h1 : x + Real.cos y = 1004)
  (h2 : x + 1004 * Real.sin y = 1003)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 1003 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l733_73303


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l733_73334

def p (x : ℤ) : ℤ := x^5 + x^3 + x + 3

theorem remainder_when_divided_by_x_minus_2 :
  p 2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l733_73334


namespace NUMINAMATH_GPT_determine_gizmos_l733_73386

theorem determine_gizmos (g d : ℝ)
  (h1 : 80 * (g * 160 + d * 240) = 80)
  (h2 : 100 * (3 * g * 900 + 3 * d * 600) = 100)
  (h3 : 70 * (5 * g * n + 5 * d * 1050) = 70 * 5 * (g + d) ) :
  n = 70 := sorry

end NUMINAMATH_GPT_determine_gizmos_l733_73386


namespace NUMINAMATH_GPT_problem_statement_l733_73371

theorem problem_statement (a n : ℕ) (h1 : 1 ≤ a) (h2 : n = 1) : ∃ m : ℤ, ((a + 1)^n - a^n) = m * n := by
  sorry

end NUMINAMATH_GPT_problem_statement_l733_73371


namespace NUMINAMATH_GPT_no_entangled_two_digit_numbers_l733_73302

theorem no_entangled_two_digit_numbers :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9 → 10 * a + b ≠ 2 * (a + b ^ 3) :=
by
  intros a b h
  rcases h with ⟨ha1, ha9, hb9⟩
  sorry

end NUMINAMATH_GPT_no_entangled_two_digit_numbers_l733_73302


namespace NUMINAMATH_GPT_original_cost_price_l733_73317

theorem original_cost_price (C : ℝ) : 
  (0.89 * C * 1.20 = 54000) → C = 50561.80 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_price_l733_73317


namespace NUMINAMATH_GPT_gate_distance_probability_correct_l733_73378

-- Define the number of gates
def num_gates : ℕ := 15

-- Define the distance between adjacent gates
def distance_between_gates : ℕ := 80

-- Define the maximum distance Dave can walk
def max_distance : ℕ := 320

-- Define the function that calculates the probability
def calculate_probability (num_gates : ℕ) (distance_between_gates : ℕ) (max_distance : ℕ) : ℚ :=
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs :=
    2 * (4 + 5 + 6 + 7) + 7 * 8
  valid_pairs / total_pairs

-- Assert the relevant result and stated answer
theorem gate_distance_probability_correct :
  let m := 10
  let n := 21
  let probability := calculate_probability num_gates distance_between_gates max_distance
  m + n = 31 ∧ probability = (10 / 21 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_gate_distance_probability_correct_l733_73378


namespace NUMINAMATH_GPT_cashier_adjustment_l733_73353

-- Define the conditions
variables {y : ℝ}

-- Error calculation given the conditions
def half_dollar_error (y : ℝ) : ℝ := 0.50 * y
def five_dollar_error (y : ℝ) : ℝ := 5 * y
def total_error (y : ℝ) : ℝ := half_dollar_error y + five_dollar_error y

-- Theorem statement
theorem cashier_adjustment (y : ℝ) : total_error y = 5.50 * y :=
sorry

end NUMINAMATH_GPT_cashier_adjustment_l733_73353


namespace NUMINAMATH_GPT_speed_of_current_l733_73333

theorem speed_of_current (c r : ℝ) 
  (h1 : 12 = (c - r) * 6) 
  (h2 : 12 = (c + r) * 0.75) : 
  r = 7 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_current_l733_73333


namespace NUMINAMATH_GPT_find_positive_integer_N_l733_73367

theorem find_positive_integer_N (N : ℕ) (h₁ : 33^2 * 55^2 = 15^2 * N^2) : N = 121 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_positive_integer_N_l733_73367


namespace NUMINAMATH_GPT_value_of_expression_l733_73387

theorem value_of_expression (V E F t h : ℕ) (H T : ℕ) 
  (h1 : V - E + F = 2)
  (h2 : F = 42)
  (h3 : T = 3)
  (h4 : H = 2)
  (h5 : t + h = 42)
  (h6 : E = (3 * t + 6 * h) / 2) :
  100 * H + 10 * T + V = 328 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l733_73387


namespace NUMINAMATH_GPT_sum_of_x_and_reciprocal_eq_3_5_l733_73327

theorem sum_of_x_and_reciprocal_eq_3_5
    (x : ℝ)
    (h : x^2 + (1 / x^2) = 10.25) :
    x + (1 / x) = 3.5 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_reciprocal_eq_3_5_l733_73327


namespace NUMINAMATH_GPT_determine_machines_in_first_group_l733_73363

noncomputable def machines_in_first_group (x r : ℝ) : Prop :=
  (x * r * 6 = 1) ∧ (12 * r * 4 = 1)

theorem determine_machines_in_first_group (x r : ℝ) (h : machines_in_first_group x r) :
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_determine_machines_in_first_group_l733_73363


namespace NUMINAMATH_GPT_subtracted_value_from_numbers_l733_73382

theorem subtracted_value_from_numbers (A B C D E X : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 5)
  (h2 : ((A - X) + (B - X) + (C - X) + (D - X) + E) / 5 = 3.4) :
  X = 2 :=
by
  sorry

end NUMINAMATH_GPT_subtracted_value_from_numbers_l733_73382


namespace NUMINAMATH_GPT_smallest_solution_x4_minus_40x2_plus_400_eq_zero_l733_73323

theorem smallest_solution_x4_minus_40x2_plus_400_eq_zero :
  ∃ x : ℝ, (x^4 - 40 * x^2 + 400 = 0) ∧ (∀ y : ℝ, (y^4 - 40 * y^2 + 400 = 0) → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_solution_x4_minus_40x2_plus_400_eq_zero_l733_73323


namespace NUMINAMATH_GPT_sarahs_packages_l733_73320

def num_cupcakes_before : ℕ := 60
def num_cupcakes_ate : ℕ := 22
def cupcakes_per_package : ℕ := 10

theorem sarahs_packages : (num_cupcakes_before - num_cupcakes_ate) / cupcakes_per_package = 3 :=
by
  sorry

end NUMINAMATH_GPT_sarahs_packages_l733_73320


namespace NUMINAMATH_GPT_fermat_little_theorem_l733_73301

theorem fermat_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℕ) : a^p ≡ a [MOD p] :=
sorry

end NUMINAMATH_GPT_fermat_little_theorem_l733_73301


namespace NUMINAMATH_GPT_Lisa_favorite_number_l733_73391

theorem Lisa_favorite_number (a b : ℕ) (h : 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) :
  (10 * a + b)^2 = (a + b)^3 → 10 * a + b = 27 := by
  intro h_eq
  sorry

end NUMINAMATH_GPT_Lisa_favorite_number_l733_73391


namespace NUMINAMATH_GPT_recurring_decimal_division_l733_73312

noncomputable def recurring_decimal_fraction : ℚ :=
  let frac_81 := (81 : ℚ) / 99
  let frac_36 := (36 : ℚ) / 99
  frac_81 / frac_36

theorem recurring_decimal_division :
  recurring_decimal_fraction = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_recurring_decimal_division_l733_73312


namespace NUMINAMATH_GPT_cube_side_length_l733_73393

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end NUMINAMATH_GPT_cube_side_length_l733_73393


namespace NUMINAMATH_GPT_smallest_five_digit_divisible_by_53_and_3_l733_73354

/-- The smallest five-digit positive integer divisible by 53 and 3 is 10062 -/
theorem smallest_five_digit_divisible_by_53_and_3 : ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 ∧ n % 3 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 53 = 0 ∧ m % 3 = 0 → n ≤ m ∧ n = 10062 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_divisible_by_53_and_3_l733_73354


namespace NUMINAMATH_GPT_percentage_increase_l733_73331

theorem percentage_increase (P : ℝ) (x : ℝ) 
(h1 : 1.17 * P = 0.90 * P * (1 + x / 100)) : x = 33.33 :=
by sorry

end NUMINAMATH_GPT_percentage_increase_l733_73331


namespace NUMINAMATH_GPT_opera_house_rows_l733_73318

variable (R : ℕ)
variable (SeatsPerRow : ℕ)
variable (TicketPrice : ℕ)
variable (TotalEarnings : ℕ)
variable (SeatsTakenPercent : ℝ)

-- Conditions
axiom num_seats_per_row : SeatsPerRow = 10
axiom ticket_price : TicketPrice = 10
axiom total_earnings : TotalEarnings = 12000
axiom seats_taken_percent : SeatsTakenPercent = 0.8

-- Main theorem statement
theorem opera_house_rows
  (h1 : SeatsPerRow = 10)
  (h2 : TicketPrice = 10)
  (h3 : TotalEarnings = 12000)
  (h4 : SeatsTakenPercent = 0.8) :
  R = 150 :=
sorry

end NUMINAMATH_GPT_opera_house_rows_l733_73318


namespace NUMINAMATH_GPT_pictures_vertically_l733_73369

def total_pictures := 30
def haphazard_pictures := 5
def horizontal_pictures := total_pictures / 2

theorem pictures_vertically : total_pictures - (horizontal_pictures + haphazard_pictures) = 10 := by
  sorry

end NUMINAMATH_GPT_pictures_vertically_l733_73369


namespace NUMINAMATH_GPT_additional_people_needed_l733_73349

theorem additional_people_needed
  (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ)
  (h_initial : initial_people * initial_time = 24)
  (h_time : new_time = 2)
  (h_initial_people : initial_people = 8)
  (h_initial_time : initial_time = 3) :
  (24 / new_time) - initial_people = 4 :=
by
  sorry

end NUMINAMATH_GPT_additional_people_needed_l733_73349


namespace NUMINAMATH_GPT_real_roots_m_range_find_value_of_m_l733_73388

-- Part 1: Prove the discriminant condition for real roots
theorem real_roots_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - (2 * m + 3) * x + m^2 + 2 = 0) ↔ m ≥ -1/12 := 
sorry

-- Part 2: Prove the value of m given the condition on roots
theorem find_value_of_m (m : ℝ) (x1 x2 : ℝ) 
  (h : x1^2 + x2^2 = 3 * x1 * x2 - 14)
  (h_roots : x^2 - (2 * m + 3) * x + m^2 + 2 = 0 → (x = x1 ∨ x = x2)) :
  m = 13 := 
sorry

end NUMINAMATH_GPT_real_roots_m_range_find_value_of_m_l733_73388


namespace NUMINAMATH_GPT_eq1_solution_eq2_solution_l733_73376

theorem eq1_solution (x : ℝ) : (x = 3 + 2 * Real.sqrt 2 ∨ x = 3 - 2 * Real.sqrt 2) ↔ (x^2 - 6 * x + 1 = 0) :=
by
  sorry

theorem eq2_solution (x : ℝ) : (x = 1 ∨ x = -5 / 2) ↔ (2 * x^2 + 3 * x - 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_eq1_solution_eq2_solution_l733_73376


namespace NUMINAMATH_GPT_prob_rain_both_days_correct_l733_73377

-- Definitions according to the conditions
def prob_rain_Saturday : ℝ := 0.4
def prob_rain_Sunday : ℝ := 0.3
def cond_prob_rain_Sunday_given_Saturday : ℝ := 0.5

-- Target probability to prove
def prob_rain_both_days : ℝ := prob_rain_Saturday * cond_prob_rain_Sunday_given_Saturday

-- Theorem statement
theorem prob_rain_both_days_correct : prob_rain_both_days = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_prob_rain_both_days_correct_l733_73377
