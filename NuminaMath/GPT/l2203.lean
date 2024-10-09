import Mathlib

namespace find_ab_l2203_220350

theorem find_ab (a b : ℤ) :
  (∀ x : ℤ, a * (x - 3) + b * (3 * x + 1) = 5 * (x + 1)) →
  a = -1 ∧ b = 2 :=
by
  sorry

end find_ab_l2203_220350


namespace problem_proof_l2203_220332

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 2

theorem problem_proof : f (1 + g 3) = 32 := by
  sorry

end problem_proof_l2203_220332


namespace find_line_equation_l2203_220395

theorem find_line_equation :
  ∃ (a b c : ℝ), (a * -5 + b * -1 = c) ∧ (a * 1 + b * 1 = c + 2) ∧ (b ≠ 0) ∧ (a * 2 + b = 0) → (∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = -5) :=
by
  sorry

end find_line_equation_l2203_220395


namespace n_cubed_minus_9n_plus_27_not_div_by_81_l2203_220351

theorem n_cubed_minus_9n_plus_27_not_div_by_81 (n : ℤ) : ¬ 81 ∣ (n^3 - 9 * n + 27) :=
sorry

end n_cubed_minus_9n_plus_27_not_div_by_81_l2203_220351


namespace factor_expression_l2203_220311

theorem factor_expression (x : ℝ) : 12 * x^2 - 6 * x = 6 * x * (2 * x - 1) :=
by
sorry

end factor_expression_l2203_220311


namespace unique_solution_abs_eq_l2203_220387

theorem unique_solution_abs_eq (x : ℝ) : (|x - 9| = |x + 3| + 2) ↔ x = 2 :=
by
  sorry

end unique_solution_abs_eq_l2203_220387


namespace cover_condition_l2203_220323

theorem cover_condition (n : ℕ) :
  (∃ (f : ℕ) (h1 : f = n^2), f % 2 = 0) ↔ (n % 2 = 0) := 
sorry

end cover_condition_l2203_220323


namespace simplify_expression_l2203_220301

variable (a : ℝ)

theorem simplify_expression :
    5 * a^2 - (a^2 - 2 * (a^2 - 3 * a)) = 6 * a^2 - 6 * a := by
  sorry

end simplify_expression_l2203_220301


namespace sum_of_first_15_terms_l2203_220359

-- Given an arithmetic sequence {a_n} such that a_4 + a_6 + a_8 + a_10 + a_12 = 40
-- we need to prove that the sum of the first 15 terms is 120

theorem sum_of_first_15_terms 
  (a_4 a_6 a_8 a_10 a_12 : ℤ)
  (h1 : a_4 + a_6 + a_8 + a_10 + a_12 = 40)
  (a1 d : ℤ)
  (h2 : a_4 = a1 + 3*d)
  (h3 : a_6 = a1 + 5*d)
  (h4 : a_8 = a1 + 7*d)
  (h5 : a_10 = a1 + 9*d)
  (h6 : a_12 = a1 + 11*d) :
  (15 * (a1 + 7*d) = 120) :=
by
  sorry

end sum_of_first_15_terms_l2203_220359


namespace option_d_is_deductive_reasoning_l2203_220335

-- Define the conditions of the problem
def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ c q : ℤ, c * q ≠ 0 ∧ ∀ n : ℕ, a n = c * q ^ n

-- Define the specific sequence {-2^n}
def a (n : ℕ) : ℤ := -2^n

-- State the proof problem
theorem option_d_is_deductive_reasoning :
  is_geometric_sequence a :=
sorry

end option_d_is_deductive_reasoning_l2203_220335


namespace remainder_when_divided_by_13_l2203_220383

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) : (N = 39 * k + 17) → (N % 13 = 4) := by
  sorry

end remainder_when_divided_by_13_l2203_220383


namespace range_of_a_l2203_220381

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, a ≤ x ∧ (x : ℝ) < 2 → x = -1 ∨ x = 0 ∨ x = 1) ↔ (-2 < a ∧ a ≤ -1) :=
by
  sorry

end range_of_a_l2203_220381


namespace speed_difference_between_lucy_and_sam_l2203_220340

noncomputable def average_speed (distance : ℚ) (time_minutes : ℚ) : ℚ :=
  distance / (time_minutes / 60)

theorem speed_difference_between_lucy_and_sam :
  let distance := 6
  let lucy_time := 15
  let sam_time := 45
  let lucy_speed := average_speed distance lucy_time
  let sam_speed := average_speed distance sam_time
  (lucy_speed - sam_speed) = 16 :=
by
  sorry

end speed_difference_between_lucy_and_sam_l2203_220340


namespace instantaneous_velocity_at_t2_l2203_220326

def displacement (t : ℝ) : ℝ := 14 * t - t ^ 2

theorem instantaneous_velocity_at_t2 : (deriv displacement 2) = 10 := by
  sorry

end instantaneous_velocity_at_t2_l2203_220326


namespace asha_savings_l2203_220330

theorem asha_savings (brother father mother granny spending remaining total borrowed_gifted savings : ℤ) 
  (h1 : brother = 20)
  (h2 : father = 40)
  (h3 : mother = 30)
  (h4 : granny = 70)
  (h5 : spending = 3 * total / 4)
  (h6 : remaining = 65)
  (h7 : remaining = total - spending)
  (h8 : total = brother + father + mother + granny + savings)
  (h9 : borrowed_gifted = brother + father + mother + granny) :
  savings = 100 := by
    sorry

end asha_savings_l2203_220330


namespace angle_bisector_proportion_l2203_220321

theorem angle_bisector_proportion
  (p q r : ℝ)
  (u v : ℝ)
  (h1 : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < u ∧ 0 < v)
  (h2 : u + v = p)
  (h3 : u * q = v * r) :
  u / p = r / (r + q) :=
sorry

end angle_bisector_proportion_l2203_220321


namespace line_through_point_parallel_to_given_l2203_220331

open Real

theorem line_through_point_parallel_to_given (x y : ℝ) :
  (∃ (m : ℝ), (y - 0 = m * (x - 1)) ∧ x - 2*y - 1 = 0) ↔
  (x = 1 ∧ y = 0 ∧ ∃ l, x - 2*y - l = 0) :=
by sorry

end line_through_point_parallel_to_given_l2203_220331


namespace foci_distance_of_hyperbola_l2203_220366

theorem foci_distance_of_hyperbola : 
  let a_squared := 32
  let b_squared := 8
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  2 * c = 4 * Real.sqrt 10 :=
by
  -- Definitions based on conditions
  let a_squared := 32
  let b_squared := 8
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  
  -- Proof outline here (using sorry to skip proof details)
  sorry

end foci_distance_of_hyperbola_l2203_220366


namespace negation_of_existential_l2203_220305

theorem negation_of_existential : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 > 0)) ↔ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 ≤ 0) := 
by 
  sorry

end negation_of_existential_l2203_220305


namespace ab_eq_neg_two_l2203_220382

theorem ab_eq_neg_two (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : a * b^a = -2 :=
by
  sorry

end ab_eq_neg_two_l2203_220382


namespace baseball_card_value_decrease_l2203_220388

theorem baseball_card_value_decrease (initial_value : ℝ) :
  (1 - 0.70 * 0.90) * 100 = 37 := 
by sorry

end baseball_card_value_decrease_l2203_220388


namespace average_weight_of_whole_class_l2203_220304

theorem average_weight_of_whole_class :
  ∀ (n_a n_b : ℕ) (w_avg_a w_avg_b : ℝ),
    n_a = 60 →
    n_b = 70 →
    w_avg_a = 60 →
    w_avg_b = 80 →
    (n_a * w_avg_a + n_b * w_avg_b) / (n_a + n_b) = 70.77 :=
by
  intros n_a n_b w_avg_a w_avg_b h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end average_weight_of_whole_class_l2203_220304


namespace product_of_abcd_l2203_220369

noncomputable def a (c : ℚ) : ℚ := 33 * c + 16
noncomputable def b (c : ℚ) : ℚ := 8 * c + 4
noncomputable def d (c : ℚ) : ℚ := c + 1

theorem product_of_abcd :
  (2 * a c + 3 * b c + 5 * c + 8 * d c = 45) →
  (4 * (d c + c) = b c) →
  (4 * (b c) + c = a c) →
  (c + 1 = d c) →
  a c * b c * c * d c = ((1511 : ℚ) / 103) * ((332 : ℚ) / 103) * (-(7 : ℚ) / 103) * ((96 : ℚ) / 103) :=
by
  intros
  sorry

end product_of_abcd_l2203_220369


namespace shaded_area_ratio_l2203_220371

noncomputable def ratio_of_shaded_area_to_circle_area (AB r : ℝ) : ℝ :=
  let AC := r
  let CB := 2 * r
  let radius_semicircle_AB := 3 * r / 2
  let area_semicircle_AB := (1 / 2) * (Real.pi * (radius_semicircle_AB ^ 2))
  let radius_semicircle_AC := r / 2
  let area_semicircle_AC := (1 / 2) * (Real.pi * (radius_semicircle_AC ^ 2))
  let radius_semicircle_CB := r
  let area_semicircle_CB := (1 / 2) * (Real.pi * (radius_semicircle_CB ^ 2))
  let total_area_semicircles := area_semicircle_AB + area_semicircle_AC + area_semicircle_CB
  let non_overlapping_area_semicircle_AB := area_semicircle_AB - (area_semicircle_AC + area_semicircle_CB)
  let shaded_area := non_overlapping_area_semicircle_AB
  let area_circle_CD := Real.pi * (r ^ 2)
  shaded_area / area_circle_CD

theorem shaded_area_ratio (AB r : ℝ) : ratio_of_shaded_area_to_circle_area AB r = 1 / 4 :=
by
  sorry

end shaded_area_ratio_l2203_220371


namespace prime_intersect_even_l2203_220374

-- Definitions for prime numbers and even numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Sets P and Q
def P : Set ℕ := { n | is_prime n }
def Q : Set ℕ := { n | is_even n }

-- Proof statement
theorem prime_intersect_even : P ∩ Q = {2} :=
by
  sorry

end prime_intersect_even_l2203_220374


namespace power_function_passing_through_point_l2203_220385

theorem power_function_passing_through_point :
  ∃ (α : ℝ), (2:ℝ)^α = 4 := by
  sorry

end power_function_passing_through_point_l2203_220385


namespace irrationals_among_examples_l2203_220314

theorem irrationals_among_examples :
  ¬ ∃ (r : ℚ), r = π ∧
  (∃ (a b : ℚ), a * a = 4) ∧
  (∃ (r : ℚ), r = 0) ∧
  (∃ (r : ℚ), r = -22 / 7) := 
sorry

end irrationals_among_examples_l2203_220314


namespace average_grade_of_female_students_is_92_l2203_220338

noncomputable def female_average_grade 
  (overall_avg : ℝ) (male_avg : ℝ) (num_males : ℕ) (num_females : ℕ) : ℝ :=
  let total_students := num_males + num_females
  let total_score := total_students * overall_avg
  let male_total_score := num_males * male_avg
  let female_total_score := total_score - male_total_score
  female_total_score / num_females

theorem average_grade_of_female_students_is_92 :
  female_average_grade 90 83 8 28 = 92 := 
by
  -- Proof steps to be completed
  sorry

end average_grade_of_female_students_is_92_l2203_220338


namespace total_revenue_is_correct_l2203_220320

-- Joan decided to sell all of her old books.
-- She had 33 books in total.
-- She sold 15 books at $4 each.
-- She sold 6 books at $7 each.
-- The rest of the books were sold at $10 each.
-- We need to prove that the total revenue is $222.

def totalBooks := 33
def booksAt4 := 15
def priceAt4 := 4
def booksAt7 := 6
def priceAt7 := 7
def priceAt10 := 10
def remainingBooks := totalBooks - (booksAt4 + booksAt7)
def revenueAt4 := booksAt4 * priceAt4
def revenueAt7 := booksAt7 * priceAt7
def revenueAt10 := remainingBooks * priceAt10
def totalRevenue := revenueAt4 + revenueAt7 + revenueAt10

theorem total_revenue_is_correct : totalRevenue = 222 := by
  sorry

end total_revenue_is_correct_l2203_220320


namespace part_a_l2203_220386

theorem part_a (c : ℤ) : (∃ x : ℤ, x + (x / 2) = c) ↔ (c % 3 ≠ 2) :=
sorry

end part_a_l2203_220386


namespace simplify_fraction_l2203_220322

-- Define factorial (or use the existing factorial definition if available in Mathlib)
def fact : ℕ → ℕ 
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Problem statement
theorem simplify_fraction :
  (5 * fact 7 + 35 * fact 6) / fact 8 = 5 / 4 := by
  sorry

end simplify_fraction_l2203_220322


namespace lines_intersect_l2203_220362

-- Condition definitions
def line1 (t : ℝ) : ℝ × ℝ :=
  ⟨2 + t * -1, 3 + t * 5⟩

def line2 (u : ℝ) : ℝ × ℝ :=
  ⟨u * -1, 7 + u * 4⟩

-- Theorem statement
theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (6, -17) :=
by
  sorry

end lines_intersect_l2203_220362


namespace four_digit_integers_correct_five_digit_integers_correct_l2203_220318

-- Definition for the four-digit integers problem
def num_four_digit_integers := ∃ digits : Finset (Fin 5), 4 * 24 = 96

theorem four_digit_integers_correct : num_four_digit_integers := 
by
  sorry

-- Definition for the five-digit integers problem without repetition and greater than 21000
def num_five_digit_integers := ∃ digits : Finset (Fin 5), 48 + 18 = 66

theorem five_digit_integers_correct : num_five_digit_integers := 
by
  sorry

end four_digit_integers_correct_five_digit_integers_correct_l2203_220318


namespace passed_boys_avg_marks_l2203_220342

theorem passed_boys_avg_marks (total_boys : ℕ) (avg_marks_all_boys : ℕ) (avg_marks_failed_boys : ℕ) (passed_boys : ℕ) 
  (h1 : total_boys = 120)
  (h2 : avg_marks_all_boys = 35)
  (h3 : avg_marks_failed_boys = 15)
  (h4 : passed_boys = 100) : 
  (39 = (35 * 120 - 15 * (total_boys - passed_boys)) / passed_boys) :=
  sorry

end passed_boys_avg_marks_l2203_220342


namespace integer_roots_of_quadratic_eq_are_neg3_and_neg7_l2203_220378

theorem integer_roots_of_quadratic_eq_are_neg3_and_neg7 :
  {k : ℤ | ∃ x : ℤ, k * x^2 - 2 * (3 * k - 1) * x + 9 * k - 1 = 0} = {-3, -7} :=
by
  sorry

end integer_roots_of_quadratic_eq_are_neg3_and_neg7_l2203_220378


namespace cube_painting_problem_l2203_220379

theorem cube_painting_problem (n : ℕ) (hn : n > 0) :
  (6 * n^2 = (6 * n^3) / 3) ↔ n = 3 :=
by sorry

end cube_painting_problem_l2203_220379


namespace find_a10_l2203_220360

def arith_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

variables (a : ℕ → ℚ) (d : ℚ)

-- Conditions
def condition1 := a 4 + a 11 = 16  -- translates to a_5 + a_12 = 16
def condition2 := a 6 = 1  -- translates to a_7 = 1
def condition3 := arith_seq a d  -- a is an arithmetic sequence with common difference d

-- The main theorem
theorem find_a10 : condition1 a ∧ condition2 a ∧ condition3 a d → a 9 = 15 := sorry

end find_a10_l2203_220360


namespace guests_did_not_respond_l2203_220356

theorem guests_did_not_respond (n : ℕ) (p_yes p_no : ℝ) (hn : n = 200)
    (hp_yes : p_yes = 0.83) (hp_no : p_no = 0.09) : 
    n - (n * p_yes + n * p_no) = 16 :=
by sorry

end guests_did_not_respond_l2203_220356


namespace find_m_of_cos_alpha_l2203_220308

theorem find_m_of_cos_alpha (m : ℝ) (h₁ : (2 * Real.sqrt 5) / 5 = m / Real.sqrt (m ^ 2 + 1)) (h₂ : m > 0) : m = 2 :=
sorry

end find_m_of_cos_alpha_l2203_220308


namespace ab_cd_zero_l2203_220306

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1)
  (h3 : a * c + b * d = 0) : 
  a * b + c * d = 0 := 
by sorry

end ab_cd_zero_l2203_220306


namespace average_of_remaining_numbers_l2203_220317

theorem average_of_remaining_numbers (S : ℕ) 
  (h₁ : S = 85 * 10) 
  (S' : ℕ) 
  (h₂ : S' = S - 70 - 76) : 
  S' / 8 = 88 := 
sorry

end average_of_remaining_numbers_l2203_220317


namespace value_of_y_l2203_220333

theorem value_of_y (y : ℕ) (hy : (1 / 8) * 2^36 = 8^y) : y = 11 :=
by
  sorry

end value_of_y_l2203_220333


namespace sum_of_squares_of_consecutive_integers_l2203_220327

theorem sum_of_squares_of_consecutive_integers :
  ∃ x : ℕ, x * (x + 1) * (x + 2) = 12 * (x + (x + 1) + (x + 2)) ∧ (x^2 + (x + 1)^2 + (x + 2)^2 = 77) :=
by
  sorry

end sum_of_squares_of_consecutive_integers_l2203_220327


namespace bob_speed_l2203_220368

theorem bob_speed (v : ℝ) : (∀ v_a : ℝ, v_a > 120 → 30 / v_a < 30 / v - 0.5) → v = 40 :=
by
  sorry

end bob_speed_l2203_220368


namespace xiaohong_home_to_school_distance_l2203_220376

noncomputable def driving_distance : ℝ := 1000
noncomputable def total_travel_time : ℝ := 22.5
noncomputable def walking_speed : ℝ := 80
noncomputable def biking_time : ℝ := 40
noncomputable def biking_speed_offset : ℝ := 800

theorem xiaohong_home_to_school_distance (d : ℝ) (v_d : ℝ) :
    let t_w := (d - driving_distance) / walking_speed
    let t_d := driving_distance / v_d
    let v_b := v_d - biking_speed_offset
    (t_d + t_w = total_travel_time)
    → (d / v_b = biking_time)
    → d = 2720 :=
by
  sorry

end xiaohong_home_to_school_distance_l2203_220376


namespace prime_large_factor_l2203_220370

theorem prime_large_factor (p : ℕ) (hp : Nat.Prime p) (hp_ge_3 : p ≥ 3) (x : ℕ) (hx_large : ∃ N, x ≥ N) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ (p + 3) / 2 ∧ (∃ q : ℕ, Nat.Prime q ∧ q > p ∧ q ∣ (x + i)) := by
  sorry

end prime_large_factor_l2203_220370


namespace range_of_a_l2203_220310

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 :=
sorry

end range_of_a_l2203_220310


namespace FindDotsOnFaces_l2203_220352

-- Define the structure of a die with specific dot distribution
structure Die where
  three_dots_face : ℕ
  two_dots_faces : ℕ
  one_dot_faces : ℕ

-- Define the problem scenario of 7 identical dice forming 'П' shape
noncomputable def SevenIdenticalDiceFormP (A B C : ℕ) : Prop :=
  ∃ (d : Die), 
    d.three_dots_face = 3 ∧
    d.two_dots_faces = 2 ∧
    d.one_dot_faces = 1 ∧
    (d.three_dots_face + d.two_dots_faces + d.one_dot_faces = 6) ∧
    (A = 2) ∧
    (B = 2) ∧
    (C = 3) 

-- State the theorem to prove A = 2, B = 2, C = 3 given the conditions
theorem FindDotsOnFaces (A B C : ℕ) (h : SevenIdenticalDiceFormP A B C) : A = 2 ∧ B = 2 ∧ C = 3 :=
  by sorry

end FindDotsOnFaces_l2203_220352


namespace certain_multiple_l2203_220354

theorem certain_multiple (n m : ℤ) (h : n = 5) (eq : 7 * n - 15 = m * n + 10) : m = 2 :=
by
  sorry

end certain_multiple_l2203_220354


namespace ones_digit_of_first_in_sequence_l2203_220325

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
  
def in_arithmetic_sequence (a d : ℕ) (n : ℕ) : Prop :=
  ∃ k, a = k * d + n

theorem ones_digit_of_first_in_sequence {p q r s t : ℕ}
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (ht : is_prime t)
  (hseq : in_arithmetic_sequence p 10 q ∧ 
          in_arithmetic_sequence q 10 r ∧
          in_arithmetic_sequence r 10 s ∧
          in_arithmetic_sequence s 10 t)
  (hincr : p < q ∧ q < r ∧ r < s ∧ s < t)
  (hstart : p > 5) :
  p % 10 = 1 := sorry

end ones_digit_of_first_in_sequence_l2203_220325


namespace exponent_division_l2203_220398

-- We need to reformulate the given condition into Lean definitions
def twenty_seven_is_three_cubed : Prop := 27 = 3^3

-- Using the condition to state the problem
theorem exponent_division (h : twenty_seven_is_three_cubed) : 
  3^15 / 27^3 = 729 :=
by
  sorry

end exponent_division_l2203_220398


namespace remainder_problem_l2203_220329

theorem remainder_problem (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 :=
by
  sorry

end remainder_problem_l2203_220329


namespace y_range_l2203_220344

variable (a b : ℝ)
variable (h₀ : 0 < a) (h₁ : 0 < b)

theorem y_range (x : ℝ) (y : ℝ) (h₂ : y = (a * Real.sin x + b) / (a * Real.sin x - b)) : 
  y ≥ (a - b) / (a + b) ∨ y ≤ (a + b) / (a - b) :=
sorry

end y_range_l2203_220344


namespace decagon_adjacent_vertices_probability_l2203_220324

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l2203_220324


namespace find_g2_l2203_220393

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x : ℝ) (hx : x ≠ 0) : 4 * g x - 3 * g (1 / x) = x^2

theorem find_g2 : g 2 = 67 / 28 :=
by {
  sorry
}

end find_g2_l2203_220393


namespace painting_time_l2203_220365

noncomputable def work_rate (t : ℕ) : ℚ := 1 / t

theorem painting_time (shawn_time karen_time alex_time total_work_rate : ℚ)
  (h_shawn : shawn_time = 18)
  (h_karen : karen_time = 12)
  (h_alex : alex_time = 15) :
  total_work_rate = 1 / (shawn_time + karen_time + alex_time) :=
by
  sorry

end painting_time_l2203_220365


namespace bryan_total_books_magazines_l2203_220380

-- Conditions as definitions
def novels : ℕ := 90
def comics : ℕ := 160
def rooms : ℕ := 12
def x := (3 / 4 : ℚ) * novels
def y := (6 / 5 : ℚ) * comics
def z := (1 / 2 : ℚ) * rooms

-- Calculations based on conditions
def books_per_shelf := 27 * x
def magazines_per_shelf := 80 * y
def total_shelves := 23 * z
def total_books := books_per_shelf * total_shelves
def total_magazines := magazines_per_shelf * total_shelves
def grand_total := total_books + total_magazines

-- Theorem to prove
theorem bryan_total_books_magazines :
  grand_total = 2371275 := by
  sorry

end bryan_total_books_magazines_l2203_220380


namespace sufficient_but_not_necessary_l2203_220355

variable {a b : ℝ}

theorem sufficient_but_not_necessary (ha : a > 0) (hb : b > 0) : 
  (ab > 1) → (a + b > 2) ∧ ¬ (a + b > 2 → ab > 1) :=
by
  sorry

end sufficient_but_not_necessary_l2203_220355


namespace sum_mod_9_l2203_220315

theorem sum_mod_9 (h1 : 34125 % 9 = 1) (h2 : 34126 % 9 = 2) (h3 : 34127 % 9 = 3)
                  (h4 : 34128 % 9 = 4) (h5 : 34129 % 9 = 5) (h6 : 34130 % 9 = 6)
                  (h7 : 34131 % 9 = 7) :
  (34125 + 34126 + 34127 + 34128 + 34129 + 34130 + 34131) % 9 = 1 :=
by
  sorry

end sum_mod_9_l2203_220315


namespace point_M_coordinates_l2203_220375

/- Define the conditions -/

def isInFourthQuadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distanceToXAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.2 = d

def distanceToYAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.1 = d

/- Write the Lean theorem statement -/

theorem point_M_coordinates :
  ∀ (M : ℝ × ℝ), isInFourthQuadrant M ∧ distanceToXAxis M 3 ∧ distanceToYAxis M 4 → M = (4, -3) :=
by
  intro M
  sorry

end point_M_coordinates_l2203_220375


namespace train_crosses_platform_in_26_seconds_l2203_220373

def km_per_hr_to_m_per_s (km_per_hr : ℕ) : ℕ :=
  km_per_hr * 5 / 18

def train_crossing_time
  (train_speed_km_per_hr : ℕ)
  (train_length_m : ℕ)
  (platform_length_m : ℕ) : ℕ :=
  let total_distance_m := train_length_m + platform_length_m
  let train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr
  total_distance_m / train_speed_m_per_s

theorem train_crosses_platform_in_26_seconds :
  train_crossing_time 72 300 220 = 26 :=
by
  sorry

end train_crosses_platform_in_26_seconds_l2203_220373


namespace xy_sum_proof_l2203_220348

-- Define the given list of numbers
def original_list := [201, 202, 204, 205, 206, 209, 209, 210, 212]

-- Define the target new average and sum of numbers
def target_average : ℕ := 207
def sum_xy : ℕ := 417

-- Calculate the original sum
def original_sum : ℕ := 201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212

-- The new total sum calculation with x and y included
def new_total_sum := original_sum + sum_xy

-- Number of elements in the new list
def new_num_elements : ℕ := 11

-- Target new sum based on the new average and number of elements
def target_new_sum := target_average * new_num_elements

theorem xy_sum_proof : new_total_sum = target_new_sum := by
  sorry

end xy_sum_proof_l2203_220348


namespace weight_of_second_new_player_l2203_220390

theorem weight_of_second_new_player
  (number_of_original_players : ℕ)
  (average_weight_of_original_players : ℝ)
  (weight_of_first_new_player : ℝ)
  (new_average_weight : ℝ)
  (total_number_of_players : ℕ)
  (total_weight_of_9_players : ℝ)
  (combined_weight_of_original_and_first_new : ℝ)
  (weight_of_second_new_player : ℝ)
  (h1 : number_of_original_players = 7)
  (h2 : average_weight_of_original_players = 103)
  (h3 : weight_of_first_new_player = 110)
  (h4 : new_average_weight = 99)
  (h5 : total_number_of_players = 9)
  (h6 : total_weight_of_9_players = total_number_of_players * new_average_weight)
  (h7 : combined_weight_of_original_and_first_new = number_of_original_players * average_weight_of_original_players + weight_of_first_new_player)
  (h8 : total_weight_of_9_players - combined_weight_of_original_and_first_new = weight_of_second_new_player) :
  weight_of_second_new_player = 60 :=
by
  sorry

end weight_of_second_new_player_l2203_220390


namespace total_team_players_l2203_220389

-- Conditions
def team_percent_boys : ℚ := 0.6
def team_percent_girls := 1 - team_percent_boys
def junior_girls_count : ℕ := 10
def total_girls := junior_girls_count * 2
def girl_percentage_as_decimal := team_percent_girls

-- Problem
theorem total_team_players : (total_girls : ℚ) / girl_percentage_as_decimal = 50 := 
by 
    sorry

end total_team_players_l2203_220389


namespace average_speed_l2203_220309

theorem average_speed (speed1 speed2: ℝ) (time1 time2: ℝ) (h1: speed1 = 90) (h2: speed2 = 40) (h3: time1 = 1) (h4: time2 = 1) :
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 65 := by
  sorry

end average_speed_l2203_220309


namespace sum_of_cubes_l2203_220347

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l2203_220347


namespace solution_inequality_l2203_220358

theorem solution_inequality (θ x : ℝ)
  (h : |x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) : 
  -1 ≤ x ∧ x ≤ -Real.cos (2 * θ) :=
sorry

end solution_inequality_l2203_220358


namespace number_is_76_l2203_220363

theorem number_is_76 (x : ℝ) (h : (3 / 4) * x = x - 19) : x = 76 :=
sorry

end number_is_76_l2203_220363


namespace files_missing_is_15_l2203_220337

def total_files : ℕ := 60
def morning_files : ℕ := total_files / 2
def afternoon_files : ℕ := 15
def organized_files : ℕ := morning_files + afternoon_files
def missing_files : ℕ := total_files - organized_files

theorem files_missing_is_15 : missing_files = 15 :=
  sorry

end files_missing_is_15_l2203_220337


namespace minimum_f_value_g_ge_f_implies_a_ge_4_l2203_220341

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3

theorem minimum_f_value : (∃ x : ℝ, f x = 2 / Real.exp 1) :=
  sorry

theorem g_ge_f_implies_a_ge_4 (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f x ≤ g x a) → a ≥ 4 :=
  sorry

end minimum_f_value_g_ge_f_implies_a_ge_4_l2203_220341


namespace count_multiples_of_13_three_digit_l2203_220303

-- Definitions based on the conditions in the problem
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13 * k

-- Statement of the proof problem
theorem count_multiples_of_13_three_digit :
  ∃ (count : ℕ), count = (76 - 8 + 1) :=
sorry

end count_multiples_of_13_three_digit_l2203_220303


namespace length_of_train_l2203_220349

theorem length_of_train
  (L : ℝ) 
  (h1 : ∀ S, S = L / 8)
  (h2 : L + 267 = (L / 8) * 20) :
  L = 178 :=
sorry

end length_of_train_l2203_220349


namespace ratio_of_cost_to_selling_price_l2203_220307

-- Define the conditions in Lean
variable (C S : ℝ) -- C is the cost price per pencil, S is the selling price per pencil
variable (h : 90 * C - 40 * S = 90 * S)

-- Define the statement to be proved
theorem ratio_of_cost_to_selling_price (C S : ℝ) (h : 90 * C - 40 * S = 90 * S) : (90 * C) / (90 * S) = 13 :=
by
  sorry

end ratio_of_cost_to_selling_price_l2203_220307


namespace div_neg_21_by_3_l2203_220372

theorem div_neg_21_by_3 : (-21 : ℤ) / 3 = -7 :=
by sorry

end div_neg_21_by_3_l2203_220372


namespace quadratic_equation_root_form_l2203_220328

theorem quadratic_equation_root_form
  (a b c : ℤ) (m n p : ℤ)
  (ha : a = 3)
  (hb : b = -4)
  (hc : c = -7)
  (h_discriminant : b^2 - 4 * a * c = n)
  (hgcd_mn : Int.gcd m n = 1)
  (hgcd_mp : Int.gcd m p = 1)
  (hgcd_np : Int.gcd n p = 1) :
  n = 100 :=
by
  sorry

end quadratic_equation_root_form_l2203_220328


namespace relationship_between_a_and_b_l2203_220367

-- Definitions based on the conditions
def point1_lies_on_line (a : ℝ) : Prop := a = (2/3 : ℝ) * (-1 : ℝ) - 3
def point2_lies_on_line (b : ℝ) : Prop := b = (2/3 : ℝ) * (1/2 : ℝ) - 3

-- The main theorem to prove the relationship between a and b
theorem relationship_between_a_and_b (a b : ℝ) 
  (h1 : point1_lies_on_line a)
  (h2 : point2_lies_on_line b) : a < b :=
by
  -- Skipping the actual proof. Including sorry to indicate it's not provided.
  sorry

end relationship_between_a_and_b_l2203_220367


namespace necessary_but_not_sufficient_condition_l2203_220357

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  ((1 / a < 1 ↔ a < 0 ∨ a > 1) ∧ ¬(1 / a < 1 → a ≤ 0 ∨ a ≤ 1)) := 
by sorry

end necessary_but_not_sufficient_condition_l2203_220357


namespace employee_total_weekly_pay_l2203_220364

-- Define the conditions
def hours_per_day_first_3_days : ℕ := 6
def hours_per_day_last_2_days : ℕ := 2 * hours_per_day_first_3_days
def first_40_hours_pay_rate : ℕ := 30
def overtime_multiplier : ℕ := 3 / 2 -- 50% more pay, i.e., 1.5 times

-- Functions to compute total hours worked and total pay
def hours_first_3_days (d : ℕ) : ℕ := d * hours_per_day_first_3_days
def hours_last_2_days (d : ℕ) : ℕ := d * hours_per_day_last_2_days
def total_hours_worked : ℕ := (hours_first_3_days 3) + (hours_last_2_days 2)
def regular_hours : ℕ := min 40 total_hours_worked
def overtime_hours : ℕ := total_hours_worked - regular_hours
def regular_pay : ℕ := regular_hours * first_40_hours_pay_rate
def overtime_pay_rate : ℕ := first_40_hours_pay_rate + (first_40_hours_pay_rate / 2) -- 50% more
def overtime_pay : ℕ := overtime_hours * overtime_pay_rate
def total_pay : ℕ := regular_pay + overtime_pay

-- The statement to be proved
theorem employee_total_weekly_pay : total_pay = 1290 := by
  sorry

end employee_total_weekly_pay_l2203_220364


namespace simplify_expression_l2203_220377

theorem simplify_expression :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 10) / 4) = 13 := by
  sorry

end simplify_expression_l2203_220377


namespace gretchen_flavors_l2203_220399

/-- 
Gretchen's local ice cream shop offers 100 different flavors. She tried a quarter of the flavors 2 years ago and double that amount last year. Prove how many more flavors she needs to try this year to have tried all 100 flavors.
-/
theorem gretchen_flavors (F T2 T1 T R : ℕ) (h1 : F = 100)
  (h2 : T2 = F / 4)
  (h3 : T1 = 2 * T2)
  (h4 : T = T2 + T1)
  (h5 : R = F - T) : R = 25 :=
sorry

end gretchen_flavors_l2203_220399


namespace batsman_average_after_25th_innings_l2203_220345

theorem batsman_average_after_25th_innings (A : ℝ) (h_pre_avg : (25 * (A + 3)) = (24 * A + 80))
  : A + 3 = 8 := 
by
  sorry

end batsman_average_after_25th_innings_l2203_220345


namespace cube_sum_l2203_220334

theorem cube_sum (x y : ℝ) (h1 : x * y = 15) (h2 : x + y = 11) : x^3 + y^3 = 836 := 
by
  sorry

end cube_sum_l2203_220334


namespace option_c_opposites_l2203_220302

theorem option_c_opposites : -|3| = -3 ∧ 3 = 3 → ( ∃ x y : ℝ, x = -3 ∧ y = 3 ∧ x = -y) :=
by
  sorry

end option_c_opposites_l2203_220302


namespace reciprocal_equality_l2203_220353

theorem reciprocal_equality (a b : ℝ) (h1 : 1 / a = -8) (h2 : 1 / -b = 8) : a = b :=
sorry

end reciprocal_equality_l2203_220353


namespace normal_level_shortage_l2203_220392

theorem normal_level_shortage
  (T : ℝ) (Normal_level : ℝ)
  (h1 : 0.75 * T = 30)
  (h2 : 30 = 2 * Normal_level) :
  T - Normal_level = 25 := 
by
  sorry

end normal_level_shortage_l2203_220392


namespace smallest_n_square_area_l2203_220361

theorem smallest_n_square_area (n : ℕ) (n_positive : 0 < n) : ∃ k : ℕ, 14 * n = k^2 ↔ n = 14 := 
sorry

end smallest_n_square_area_l2203_220361


namespace quadratic_sum_roots_twice_difference_l2203_220339

theorem quadratic_sum_roots_twice_difference
  (a b c x₁ x₂ : ℝ)
  (h_eq : a * x₁^2 + b * x₁ + c = 0)
  (h_eq2 : a * x₂^2 + b * x₂ + c = 0)
  (h_sum_twice_diff: x₁ + x₂ = 2 * (x₁ - x₂)) :
  3 * b^2 = 16 * a * c :=
sorry

end quadratic_sum_roots_twice_difference_l2203_220339


namespace greatest_fraction_l2203_220300

theorem greatest_fraction 
  (w x y z : ℕ)
  (hw : w > 0)
  (h_ordering : w < x ∧ x < y ∧ y < z) :
  (x + y + z) / (w + x + y) > (w + x + y) / (x + y + z) ∧
  (x + y + z) / (w + x + y) > (w + y + z) / (x + w + z) ∧
  (x + y + z) / (w + x + y) > (x + w + z) / (w + y + z) ∧
  (x + y + z) / (w + x + y) > (y + z + w) / (x + y + z) :=
sorry

end greatest_fraction_l2203_220300


namespace tetrahedron_sphere_relations_l2203_220391

theorem tetrahedron_sphere_relations 
  (ρ ρ1 ρ2 ρ3 ρ4 m1 m2 m3 m4 : ℝ)
  (hρ_pos : ρ > 0)
  (hρ1_pos : ρ1 > 0)
  (hρ2_pos : ρ2 > 0)
  (hρ3_pos : ρ3 > 0)
  (hρ4_pos : ρ4 > 0)
  (hm1_pos : m1 > 0)
  (hm2_pos : m2 > 0)
  (hm3_pos : m3 > 0)
  (hm4_pos : m4 > 0) : 
  (2 / ρ = 1 / ρ1 + 1 / ρ2 + 1 / ρ3 + 1 / ρ4) ∧
  (1 / ρ = 1 / m1 + 1 / m2 + 1 / m3 + 1 / m4) ∧
  ( 1 / ρ1 = -1 / m1 + 1 / m2 + 1 / m3 + 1 / m4 ) := sorry

end tetrahedron_sphere_relations_l2203_220391


namespace value_of_1_minus_a_l2203_220384

theorem value_of_1_minus_a (a : ℤ) (h : a = -(-6)) : 1 - a = -5 := 
by 
  sorry

end value_of_1_minus_a_l2203_220384


namespace twenty_yuan_banknotes_count_l2203_220397

theorem twenty_yuan_banknotes_count (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
                                    (total_banknotes : x + y + z = 24)
                                    (total_amount : 10 * x + 20 * y + 50 * z = 1000) :
                                    y = 4 := 
sorry

end twenty_yuan_banknotes_count_l2203_220397


namespace card_2015_in_box_3_l2203_220346

-- Define the pattern function for placing cards
def card_placement (n : ℕ) : ℕ :=
  let cycle_length := 12
  let cycle_pos := (n - 1) % cycle_length + 1
  if cycle_pos ≤ 7 then cycle_pos
  else 14 - cycle_pos

-- Define the theorem to prove the position of the 2015th card
theorem card_2015_in_box_3 : card_placement 2015 = 3 := by
  -- sorry is used to skip the proof
  sorry

end card_2015_in_box_3_l2203_220346


namespace bread_remaining_is_26_85_l2203_220316

noncomputable def bread_leftover (jimin_cm : ℕ) (taehyung_m original_length : ℝ) : ℝ :=
  original_length - (jimin_cm / 100 + taehyung_m)

theorem bread_remaining_is_26_85 :
  bread_leftover 150 1.65 30 = 26.85 :=
by
  sorry

end bread_remaining_is_26_85_l2203_220316


namespace min_f_value_f_achieves_min_l2203_220319

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x ^ 2 + 1) + (x * (x + 3)) / (x ^ 2 + 2) + (3 * (x + 1)) / (x * (x ^ 2 + 2))

theorem min_f_value (x : ℝ) (hx : x > 0) : f x ≥ 3 :=
sorry

theorem f_achieves_min (x : ℝ) (hx : x > 0) : ∃ x, f x = 3 :=
sorry

end min_f_value_f_achieves_min_l2203_220319


namespace arithmetic_sequence_terms_count_l2203_220343

theorem arithmetic_sequence_terms_count :
  ∃ n : ℕ, ∀ a d l, 
    a = 13 → 
    d = 3 → 
    l = 73 → 
    l = a + (n - 1) * d ∧ n = 21 :=
by
  sorry

end arithmetic_sequence_terms_count_l2203_220343


namespace total_weight_of_nuts_l2203_220396

theorem total_weight_of_nuts:
  let almonds := 0.14
  let pecans := 0.38
  let walnuts := 0.22
  let cashews := 0.47
  let pistachios := 0.29
  almonds + pecans + walnuts + cashews + pistachios = 1.50 :=
by
  sorry

end total_weight_of_nuts_l2203_220396


namespace perpendicular_line_sum_l2203_220336

theorem perpendicular_line_sum (a b c : ℝ) 
  (h1 : -a / 4 * 2 / 5 = -1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * c + b = 0) : 
  a + b + c = -4 :=
sorry

end perpendicular_line_sum_l2203_220336


namespace problem_equivalent_l2203_220394

theorem problem_equivalent (a c : ℕ) (h : (3 * 100 + a * 10 + 7) + 214 = 5 * 100 + c * 10 + 1) (h5c1_div3 : (5 + c + 1) % 3 = 0) : a + c = 4 :=
sorry

end problem_equivalent_l2203_220394


namespace symmetrical_character_is_C_l2203_220312

-- Definitions of the characters and the concept of symmetry
def is_symmetrical (char: Char): Prop := 
  match char with
  | '中' => True
  | _ => False

-- The options given in the problem
def optionA := '爱'
def optionB := '我'
def optionC := '中'
def optionD := '国'

-- The problem statement: Prove that among the given options, the symmetrical character is 中.
theorem symmetrical_character_is_C : (is_symmetrical optionA = False) ∧ (is_symmetrical optionB = False) ∧ (is_symmetrical optionC = True) ∧ (is_symmetrical optionD = False) :=
by
  sorry

end symmetrical_character_is_C_l2203_220312


namespace geometric_Sn_over_n_sum_first_n_terms_l2203_220313

-- The first problem statement translation to Lean 4
theorem geometric_Sn_over_n (a S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n+1) = (n + 2) * S n) :
  ∃ r : ℕ, (r = 2 ∧ ∃ b : ℕ, b = 1 ∧ 
    ∀ n : ℕ, 0 < n → (S (n + 1)) / (n + 1) = r * (S n) / n) := 
sorry

-- The second problem statement translation to Lean 4
theorem sum_first_n_terms (a S : ℕ → ℕ) (T : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 2) * S n)
  (h3 : ∀ n : ℕ, S n = n * 2^(n - 1)) :
  ∀ n : ℕ, T n = (n - 1) * 2^n + 1 :=
sorry

end geometric_Sn_over_n_sum_first_n_terms_l2203_220313
