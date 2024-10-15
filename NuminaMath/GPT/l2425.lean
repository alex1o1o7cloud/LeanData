import Mathlib

namespace NUMINAMATH_GPT_point_outside_circle_l2425_242540

theorem point_outside_circle (a b : ℝ) (h_intersect : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a*x + b*y = 1) : a^2 + b^2 > 1 :=
by
  sorry

end NUMINAMATH_GPT_point_outside_circle_l2425_242540


namespace NUMINAMATH_GPT_excircle_tangent_segment_length_l2425_242563

theorem excircle_tangent_segment_length (A B C M : ℝ) 
  (h1 : A + B + C = 1) 
  (h2 : M = (1 / 2)) : 
  M = 1 / 2 := 
  by
    -- This is where the proof would go
    sorry

end NUMINAMATH_GPT_excircle_tangent_segment_length_l2425_242563


namespace NUMINAMATH_GPT_goods_train_speed_l2425_242503

theorem goods_train_speed 
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_to_cross : ℕ)
  (h_train : length_train = 270)
  (h_platform : length_platform = 250)
  (h_time : time_to_cross = 26) : 
  (length_train + length_platform) / time_to_cross = 20 := 
by
  sorry

end NUMINAMATH_GPT_goods_train_speed_l2425_242503


namespace NUMINAMATH_GPT_find_a_decreasing_l2425_242585

-- Define the given function
def f (a x : ℝ) : ℝ := (x - 1) ^ 2 + 2 * a * x + 1

-- State the condition
def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y ≤ f x

-- State the proposition
theorem find_a_decreasing :
  ∀ a : ℝ, is_decreasing_on (f a) (Set.Iio 4) → a ≤ -3 :=
by
  intro a
  intro h
  sorry

end NUMINAMATH_GPT_find_a_decreasing_l2425_242585


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_even_numbers_l2425_242518

theorem sum_of_squares_of_consecutive_even_numbers :
  ∃ (x : ℤ), x + (x + 2) + (x + 4) + (x + 6) = 36 → (x ^ 2 + (x + 2) ^ 2 + (x + 4) ^ 2 + (x + 6) ^ 2 = 344) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_even_numbers_l2425_242518


namespace NUMINAMATH_GPT_arithmetic_sequence_k_l2425_242599

theorem arithmetic_sequence_k (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (ha : ∀ n, S (n + 1) = S n + a (n + 1))
  (hS3_S8 : S 3 = S 8) 
  (hS7_Sk : ∃ k, S 7 = S k)
  : ∃ k, k = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_k_l2425_242599


namespace NUMINAMATH_GPT_unique_positive_integers_abc_l2425_242595

def coprime (a b : ℕ) := Nat.gcd a b = 1

def allPrimeDivisorsNotCongruentTo1Mod7 (n : ℕ) := 
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p % 7 ≠ 1

theorem unique_positive_integers_abc :
  ∀ a b c : ℕ,
    (1 ≤ a) →
    (1 ≤ b) →
    (1 ≤ c) →
    coprime a b →
    coprime b c →
    coprime c a →
    (a * a + b) ∣ (b * b + c) →
    (b * b + c) ∣ (c * c + a) →
    allPrimeDivisorsNotCongruentTo1Mod7 (a * a + b) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
  sorry

end NUMINAMATH_GPT_unique_positive_integers_abc_l2425_242595


namespace NUMINAMATH_GPT_hoseok_has_least_papers_l2425_242574

-- Definitions based on the conditions
def pieces_jungkook : ℕ := 10
def pieces_hoseok : ℕ := 7
def pieces_seokjin : ℕ := pieces_jungkook - 2

-- Theorem stating Hoseok has the least pieces of colored paper
theorem hoseok_has_least_papers : pieces_hoseok < pieces_jungkook ∧ pieces_hoseok < pieces_seokjin := by 
  sorry

end NUMINAMATH_GPT_hoseok_has_least_papers_l2425_242574


namespace NUMINAMATH_GPT_sports_club_total_members_l2425_242597

theorem sports_club_total_members :
  ∀ (B T Both Neither Total : ℕ),
    B = 17 → T = 19 → Both = 10 → Neither = 2 → Total = B + T - Both + Neither → Total = 28 :=
by
  intros B T Both Neither Total hB hT hBoth hNeither hTotal
  rw [hB, hT, hBoth, hNeither] at hTotal
  exact hTotal

end NUMINAMATH_GPT_sports_club_total_members_l2425_242597


namespace NUMINAMATH_GPT_smallest_whole_number_larger_than_triangle_perimeter_l2425_242591

theorem smallest_whole_number_larger_than_triangle_perimeter
  (s : ℝ) (h1 : 5 + 19 > s) (h2 : 5 + s > 19) (h3 : 19 + s > 5) :
  ∃ P : ℝ, P = 5 + 19 + s ∧ P < 48 ∧ ∀ n : ℤ, n > P → n = 48 :=
by
  sorry

end NUMINAMATH_GPT_smallest_whole_number_larger_than_triangle_perimeter_l2425_242591


namespace NUMINAMATH_GPT_find_m_and_equation_of_l2_l2425_242551

theorem find_m_and_equation_of_l2 (a : ℝ) (M: ℝ × ℝ) (m : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (hM : M = (-5, 1)) 
  (hl1 : ∀ {x y : ℝ}, 2 * x - y + 2 = 0) 
  (hl : ∀ {x y : ℝ}, x + y + m = 0) 
  (hl2 : ∀ {x y : ℝ}, (∃ p : ℝ × ℝ, p = M → x - 2 * y + 7 = 0)) : 
  m = -5 ∧ ∀ {x y : ℝ}, x - 2 * y + 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_m_and_equation_of_l2_l2425_242551


namespace NUMINAMATH_GPT_annual_interest_rate_l2425_242580

-- Define the initial conditions
def P : ℝ := 5600
def A : ℝ := 6384
def t : ℝ := 2
def n : ℝ := 1

-- The theorem statement:
theorem annual_interest_rate : ∃ (r : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ r = 0.067 :=
by 
  sorry -- proof goes here

end NUMINAMATH_GPT_annual_interest_rate_l2425_242580


namespace NUMINAMATH_GPT_problem_expression_value_l2425_242508

theorem problem_expression_value :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 : ℤ) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 : ℤ) = 6608 :=
by
  sorry

end NUMINAMATH_GPT_problem_expression_value_l2425_242508


namespace NUMINAMATH_GPT_missing_digit_divisibility_by_nine_l2425_242543

theorem missing_digit_divisibility_by_nine (x : ℕ) (h : 0 ≤ x ∧ x < 10) :
  9 ∣ (3 + 5 + 2 + 4 + x) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_missing_digit_divisibility_by_nine_l2425_242543


namespace NUMINAMATH_GPT_inequality_holds_l2425_242501

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def even_function : Prop := ∀ x : ℝ, f x = f (-x)
def decreasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f y ≤ f x

-- Proof goal
theorem inequality_holds (h_even : even_function f) (h_decreasing : decreasing_on_pos f) : 
  f (-3/4) ≥ f (a^2 - a + 1) := 
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l2425_242501


namespace NUMINAMATH_GPT_part1_part2_l2425_242519

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Condition 2: ∀ a b ∈ ℝ, (a + b ≠ 0) → (f(a) + f(b))/(a + b) > 0
def positiveQuotient (f : ℝ → ℝ) : Prop :=
  ∀ a b, a + b ≠ 0 → (f a + f b) / (a + b) > 0

-- Sub-problem (1): For any a, b ∈ ℝ, a > b ⟹ f(a) > f(b)
theorem part1 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) (a b : ℝ) (h : a > b) : f a > f b :=
  sorry

-- Sub-problem (2): If f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x ∈ [0, ∞), then k < 1
theorem part2 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) :
  (∀ x : ℝ, 0 ≤ x → f (9^x - 2 * 3^x) + f (2 * 9^x - k) > 0) → k < 1 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l2425_242519


namespace NUMINAMATH_GPT_max_ab_value_l2425_242512

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * a * x - b^2 + 12 ≤ 0 → x = a) : ab = 6 := by
  sorry

end NUMINAMATH_GPT_max_ab_value_l2425_242512


namespace NUMINAMATH_GPT_focus_of_parabola_l2425_242523

theorem focus_of_parabola (x y : ℝ) (h : x^2 = 16 * y) : (0, 4) = (0, 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_focus_of_parabola_l2425_242523


namespace NUMINAMATH_GPT_parabola_directrix_distance_l2425_242570

theorem parabola_directrix_distance (m : ℝ) (h : |1 / (4 * m)| = 2) : m = 1/8 ∨ m = -1/8 :=
by { sorry }

end NUMINAMATH_GPT_parabola_directrix_distance_l2425_242570


namespace NUMINAMATH_GPT_right_triangle_similarity_l2425_242500

theorem right_triangle_similarity (y : ℝ) (h : 12 / y = 9 / 7) : y = 9.33 := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_similarity_l2425_242500


namespace NUMINAMATH_GPT_dimension_tolerance_l2425_242558

theorem dimension_tolerance (base_dim : ℝ) (pos_tolerance : ℝ) (neg_tolerance : ℝ) 
  (max_dim : ℝ) (min_dim : ℝ) 
  (h_base : base_dim = 7) 
  (h_pos_tolerance : pos_tolerance = 0.05) 
  (h_neg_tolerance : neg_tolerance = 0.02) 
  (h_max_dim : max_dim = base_dim + pos_tolerance) 
  (h_min_dim : min_dim = base_dim - neg_tolerance) :
  max_dim = 7.05 ∧ min_dim = 6.98 :=
by
  sorry

end NUMINAMATH_GPT_dimension_tolerance_l2425_242558


namespace NUMINAMATH_GPT_inverse_of_square_positive_is_negative_l2425_242560

variable {x : ℝ}

-- Original proposition: ∀ x, x < 0 → x^2 > 0
def original_proposition : Prop :=
  ∀ x : ℝ, x < 0 → x^2 > 0

-- Inverse proposition to be proven: ∀ x, x^2 > 0 → x < 0
def inverse_proposition (x : ℝ) : Prop :=
  x^2 > 0 → x < 0

theorem inverse_of_square_positive_is_negative :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ x : ℝ, x^2 > 0 → x < 0) :=
  sorry

end NUMINAMATH_GPT_inverse_of_square_positive_is_negative_l2425_242560


namespace NUMINAMATH_GPT_new_person_weight_l2425_242547

theorem new_person_weight (average_increase : ℝ) (num_persons : ℕ) (replaced_weight : ℝ) (new_weight : ℝ) 
  (h1 : num_persons = 10) 
  (h2 : average_increase = 3.2) 
  (h3 : replaced_weight = 65) : 
  new_weight = 97 :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l2425_242547


namespace NUMINAMATH_GPT_females_with_advanced_degrees_eq_90_l2425_242567

-- define the given constants
def total_employees : ℕ := 360
def total_females : ℕ := 220
def total_males : ℕ := 140
def advanced_degrees : ℕ := 140
def college_degrees : ℕ := 160
def vocational_training : ℕ := 60
def males_with_college_only : ℕ := 55
def females_with_vocational_training : ℕ := 25

-- define the main theorem to prove the number of females with advanced degrees
theorem females_with_advanced_degrees_eq_90 :
  ∃ (females_with_advanced_degrees : ℕ), females_with_advanced_degrees = 90 :=
by
  sorry

end NUMINAMATH_GPT_females_with_advanced_degrees_eq_90_l2425_242567


namespace NUMINAMATH_GPT_quadratic_equation_solution_l2425_242598

-- Define the problem statement and the conditions: the equation being quadratic.
theorem quadratic_equation_solution (m : ℤ) :
  (∃ (a : ℤ), a ≠ 0 ∧ (a*x^2 - x - 2 = 0)) →
  m = -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_solution_l2425_242598


namespace NUMINAMATH_GPT_minimum_common_perimeter_l2425_242562

noncomputable def is_integer (x: ℝ) : Prop := ∃ (n: ℤ), x = n

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ is_triangle a b c

theorem minimum_common_perimeter :
  ∃ (a b : ℝ),
    is_integer a ∧ is_integer b ∧
    4 * a = 5 * b - 18 ∧
    is_isosceles_triangle a a (2 * a - 12) ∧
    is_isosceles_triangle b b (3 * b - 30) ∧
    (2 * a + (2 * a - 12) = 2 * b + (3 * b - 30)) ∧
    (2 * a + (2 * a - 12) = 228) := sorry

end NUMINAMATH_GPT_minimum_common_perimeter_l2425_242562


namespace NUMINAMATH_GPT_binom_prod_l2425_242568

theorem binom_prod : (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 := by
  sorry

end NUMINAMATH_GPT_binom_prod_l2425_242568


namespace NUMINAMATH_GPT_a_share_calculation_l2425_242589

noncomputable def investment_a : ℕ := 15000
noncomputable def investment_b : ℕ := 21000
noncomputable def investment_c : ℕ := 27000
noncomputable def total_investment : ℕ := investment_a + investment_b + investment_c -- 63000
noncomputable def b_share : ℕ := 1540
noncomputable def total_profit : ℕ := 4620  -- from the solution steps

theorem a_share_calculation :
  (investment_a * total_profit) / total_investment = 1100 := 
by
  sorry

end NUMINAMATH_GPT_a_share_calculation_l2425_242589


namespace NUMINAMATH_GPT_total_questions_on_test_l2425_242507

/-- A teacher grades students' tests by subtracting twice the number of incorrect responses
    from the number of correct responses. Given that a student received a score of 64
    and answered 88 questions correctly, prove that the total number of questions on the test is 100. -/
theorem total_questions_on_test (score correct_responses : ℕ) (grading_system : ℕ → ℕ → ℕ)
  (h1 : score = grading_system correct_responses (88 - 2 * 12))
  (h2 : correct_responses = 88)
  (h3 : score = 64) : correct_responses + (88 - 2 * 12) = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_questions_on_test_l2425_242507


namespace NUMINAMATH_GPT_songs_per_album_l2425_242522

theorem songs_per_album (C P : ℕ) (h1 : 4 * C + 5 * P = 72) (h2 : C = P) : C = 8 :=
by
  sorry

end NUMINAMATH_GPT_songs_per_album_l2425_242522


namespace NUMINAMATH_GPT_distinct_real_roots_implies_positive_l2425_242578

theorem distinct_real_roots_implies_positive (k : ℝ) (x1 x2 : ℝ) (h_distinct : x1 ≠ x2) 
  (h_root1 : x1^2 + 2*x1 - k = 0) 
  (h_root2 : x2^2 + 2*x2 - k = 0) : 
  x1^2 + x2^2 - 2 > 0 := 
sorry

end NUMINAMATH_GPT_distinct_real_roots_implies_positive_l2425_242578


namespace NUMINAMATH_GPT_min_positive_numbers_l2425_242572

theorem min_positive_numbers (n : ℕ) (numbers : ℕ → ℤ) 
  (h_length : n = 103) 
  (h_consecutive : ∀ i : ℕ, i < n → (∃ (p1 p2 : ℕ), p1 < 5 ∧ p2 < 5 ∧ p1 ≠ p2 ∧ numbers (i + p1) > 0 ∧ numbers (i + p2) > 0)) :
  ∃ (min_positive : ℕ), min_positive = 42 :=
by
  sorry

end NUMINAMATH_GPT_min_positive_numbers_l2425_242572


namespace NUMINAMATH_GPT_no_three_distinct_integers_solving_polynomial_l2425_242526

theorem no_three_distinct_integers_solving_polynomial (p : ℤ → ℤ) (hp : ∀ x, ∃ k : ℕ, p x = k • x + p 0) :
  ∀ a b c : ℤ, a ≠ b → b ≠ c → c ≠ a → p a = b → p b = c → p c = a → false :=
by
  intros a b c hab hbc hca hpa_hp pb_pc_pc
  sorry

end NUMINAMATH_GPT_no_three_distinct_integers_solving_polynomial_l2425_242526


namespace NUMINAMATH_GPT_marciaHairLengthProof_l2425_242528

noncomputable def marciaHairLengthAtEndOfSchoolYear : Float :=
  let L0 := 24.0                           -- initial length
  let L1 := L0 - 0.3 * L0                  -- length after September cut
  let L2 := L1 + 3.0 * 1.5                 -- length after three months of growth (Sept - Dec)
  let L3 := L2 - 0.2 * L2                  -- length after January cut
  let L4 := L3 + 5.0 * 1.8                 -- length after five months of growth (Jan - May)
  let L5 := L4 - 4.0                       -- length after June cut
  L5

theorem marciaHairLengthProof : marciaHairLengthAtEndOfSchoolYear = 22.04 :=
by
  sorry

end NUMINAMATH_GPT_marciaHairLengthProof_l2425_242528


namespace NUMINAMATH_GPT_g_of_36_l2425_242564

theorem g_of_36 (g : ℕ → ℕ)
  (h1 : ∀ n, g (n + 1) > g n)
  (h2 : ∀ m n, g (m * n) = g m * g n)
  (h3 : ∀ m n, m ≠ n ∧ m ^ n = n ^ m → (g m = n ∨ g n = m))
  (h4 : ∀ n, g (n ^ 2) = g n * n) :
  g 36 = 36 :=
  sorry

end NUMINAMATH_GPT_g_of_36_l2425_242564


namespace NUMINAMATH_GPT_find_n_in_range_l2425_242529

theorem find_n_in_range :
  ∃ n : ℕ, n > 1 ∧ 
           n % 3 = 2 ∧ 
           n % 5 = 2 ∧ 
           n % 7 = 2 ∧ 
           101 ≤ n ∧ n ≤ 134 :=
by sorry

end NUMINAMATH_GPT_find_n_in_range_l2425_242529


namespace NUMINAMATH_GPT_sixth_power_sum_l2425_242546

/-- Given:
     (1) a + b = 1
     (2) a^2 + b^2 = 3
     (3) a^3 + b^3 = 4
     (4) a^4 + b^4 = 7
     (5) a^5 + b^5 = 11
    Prove:
     a^6 + b^6 = 18 -/
theorem sixth_power_sum (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end NUMINAMATH_GPT_sixth_power_sum_l2425_242546


namespace NUMINAMATH_GPT_percentage_of_men_35_l2425_242539

theorem percentage_of_men_35 (M W : ℝ) (hm1 : M + W = 100) 
  (hm2 : 0.6 * M + 0.2923 * W = 40)
  (hw : W = 100 - M) : 
  M = 35 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_men_35_l2425_242539


namespace NUMINAMATH_GPT_smallest_element_in_M_l2425_242515

def f : ℝ → ℝ := sorry
axiom f1 (x y : ℝ) (h1 : x ≥ 1) (h2 : y = 3 * x) : f y = 3 * f x
axiom f2 (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) : f x = 1 - abs (x - 2)
axiom f99_value : f 99 = 18

theorem smallest_element_in_M : ∃ x : ℝ, x = 45 ∧ f x = 18 := by
  -- proof will be provided later
  sorry

end NUMINAMATH_GPT_smallest_element_in_M_l2425_242515


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l2425_242527

open Real

theorem parabola_focus_coordinates (x y : ℝ) (h : y^2 = 6 * x) : (x, y) = (3 / 2, 0) :=
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l2425_242527


namespace NUMINAMATH_GPT_find_x_l2425_242569

theorem find_x (U : Set ℕ) (A B : Set ℕ) (x : ℕ) 
  (hU : U = Set.univ)
  (hA : A = {1, 4, x})
  (hB : B = {1, x ^ 2})
  (h : compl A ⊂ compl B) :
  x = 0 ∨ x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l2425_242569


namespace NUMINAMATH_GPT_rory_more_jellybeans_l2425_242544

-- Definitions based on the conditions
def G : ℕ := 15 -- Gigi has 15 jellybeans
def LorelaiConsumed (R G : ℕ) : ℕ := 3 * (R + G) -- Lorelai has already eaten three times the total number of jellybeans

theorem rory_more_jellybeans {R : ℕ} (h1 : LorelaiConsumed R G = 180) : (R - G) = 30 :=
  by
    -- we can skip the proof here with sorry, as we are only interested in the statement for now
    sorry

end NUMINAMATH_GPT_rory_more_jellybeans_l2425_242544


namespace NUMINAMATH_GPT_value_of_a_l2425_242566

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2425_242566


namespace NUMINAMATH_GPT_area_of_triangle_PQS_l2425_242587

-- Define a structure to capture the conditions of the trapezoid and its properties.
structure Trapezoid (P Q R S : Type) :=
(area : ℝ)
(PQ : ℝ)
(RS : ℝ)
(area_PQS : ℝ)
(condition1 : area = 18)
(condition2 : RS = 3 * PQ)

-- Here's the theorem we want to prove, stating the conclusion based on the given conditions.
theorem area_of_triangle_PQS {P Q R S : Type} (T : Trapezoid P Q R S) : T.area_PQS = 4.5 :=
by
  -- Proof will go here, but for now we use sorry.
  sorry

end NUMINAMATH_GPT_area_of_triangle_PQS_l2425_242587


namespace NUMINAMATH_GPT_basketball_game_score_l2425_242513

theorem basketball_game_score 
  (a r b d : ℕ)
  (H1 : a = b)
  (H2 : a + a * r + a * r^2 = b + (b + d) + (b + 2 * d))
  (H3 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (H4 : r = 3)
  (H5 : a = 3)
  (H6 : d = 10)
  (H7 : a * (1 + r) = 12)
  (H8 : b * (1 + 3 + (b + d)) = 16) :
  a + a * r + b + (b + d) = 28 :=
by simp [H4, H5, H6, H7, H8]; linarith

end NUMINAMATH_GPT_basketball_game_score_l2425_242513


namespace NUMINAMATH_GPT_conic_sections_of_equation_l2425_242549

noncomputable def is_parabola (s : Set (ℝ × ℝ)) : Prop :=
∃ a b c : ℝ, ∀ x y : ℝ, (x, y) ∈ s ↔ y ≠ 0 ∧ y = a * x^3 + b * x + c

theorem conic_sections_of_equation :
  let eq := { p : ℝ × ℝ | p.2^6 - 9 * p.1^6 = 3 * p.2^3 - 1 }
  (is_parabola eq1) → (is_parabola eq2) → (eq = eq1 ∪ eq2) :=
by sorry

end NUMINAMATH_GPT_conic_sections_of_equation_l2425_242549


namespace NUMINAMATH_GPT_find_factor_l2425_242510

theorem find_factor (x f : ℕ) (h1 : x = 15) (h2 : (2 * x + 5) * f = 105) : f = 3 :=
sorry

end NUMINAMATH_GPT_find_factor_l2425_242510


namespace NUMINAMATH_GPT_normal_distribution_test_l2425_242505

noncomputable def normal_distribution_at_least_90 : Prop :=
  let μ := 78
  let σ := 4
  -- Given reference data
  let p_within_3_sigma := 0.9974
  -- Calculate P(X >= 90)
  let p_at_least_90 := (1 - p_within_3_sigma) / 2
  -- The expected answer 0.13% ⇒ 0.0013
  p_at_least_90 = 0.0013

theorem normal_distribution_test :
  normal_distribution_at_least_90 :=
by
  sorry

end NUMINAMATH_GPT_normal_distribution_test_l2425_242505


namespace NUMINAMATH_GPT_box_volume_max_l2425_242550

noncomputable def volume (a x : ℝ) : ℝ :=
  (a - 2 * x) ^ 2 * x

theorem box_volume_max (a : ℝ) (h : 0 < a) :
  ∃ x, 0 < x ∧ x < a / 2 ∧ volume a x = volume a (a / 6) ∧ volume a (a / 6) = (2 * a^3) / 27 :=
by
  sorry

end NUMINAMATH_GPT_box_volume_max_l2425_242550


namespace NUMINAMATH_GPT_Jordana_current_age_is_80_l2425_242517

-- Given conditions
def current_age_Jennifer := 20  -- since Jennifer will be 30 in ten years
def current_age_Jordana := 80  -- since the problem states we need to verify this

-- Prove that Jordana's current age is 80 years old given the conditions
theorem Jordana_current_age_is_80:
  (current_age_Jennifer + 10 = 30) →
  (current_age_Jordana + 10 = 3 * 30) →
  current_age_Jordana = 80 :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_Jordana_current_age_is_80_l2425_242517


namespace NUMINAMATH_GPT_reciprocal_proof_l2425_242534

theorem reciprocal_proof :
  (-2) * (-(1 / 2)) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_reciprocal_proof_l2425_242534


namespace NUMINAMATH_GPT_distinct_naturals_and_power_of_prime_l2425_242577

theorem distinct_naturals_and_power_of_prime (a b : ℕ) (p k : ℕ) (h1 : a ≠ b) (h2 : a^2 + b ∣ b^2 + a) (h3 : ∃ (p : ℕ) (k : ℕ), b^2 + a = p^k) : (a = 2 ∧ b = 5) ∨ (a = 5 ∧ b = 2) :=
sorry

end NUMINAMATH_GPT_distinct_naturals_and_power_of_prime_l2425_242577


namespace NUMINAMATH_GPT_triangle_side_length_l2425_242504

theorem triangle_side_length 
  (A : ℝ) (a m n : ℝ) 
  (hA : A = 60) 
  (h1 : m + n = 7) 
  (h2 : m * n = 11) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l2425_242504


namespace NUMINAMATH_GPT_arithmetic_sequence_m_l2425_242593

theorem arithmetic_sequence_m (m : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, a n = 2 * n - 1) →
  (∀ n, S n = n * (2 * n - 1) / 2) →
  S m = (a m + a (m + 1)) / 2 →
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_m_l2425_242593


namespace NUMINAMATH_GPT_probability_all_quitters_same_tribe_l2425_242533

-- Definitions of the problem conditions
def total_contestants : ℕ := 20
def tribe_size : ℕ := 10
def quitters : ℕ := 3

-- Definition of the binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem probability_all_quitters_same_tribe :
  (choose tribe_size quitters + choose tribe_size quitters) * 
  (total_contestants.choose quitters) = 240 
  ∧ ((choose tribe_size quitters + choose tribe_size quitters) / (total_contestants.choose quitters)) = 20 / 95 :=
by
  sorry

end NUMINAMATH_GPT_probability_all_quitters_same_tribe_l2425_242533


namespace NUMINAMATH_GPT_meat_division_l2425_242552

theorem meat_division (w1 w2 meat : ℕ) (h1 : w1 = 645) (h2 : w2 = 237) (h3 : meat = 1000) :
  ∃ (m1 m2 : ℕ), m1 = 296 ∧ m2 = 704 ∧ w1 + m1 = w2 + m2 := by
  sorry

end NUMINAMATH_GPT_meat_division_l2425_242552


namespace NUMINAMATH_GPT_number_of_3_letter_words_with_at_least_one_A_l2425_242576

theorem number_of_3_letter_words_with_at_least_one_A :
  let all_words := 5^3
  let no_A_words := 4^3
  all_words - no_A_words = 61 :=
by
  sorry

end NUMINAMATH_GPT_number_of_3_letter_words_with_at_least_one_A_l2425_242576


namespace NUMINAMATH_GPT_maxvalue_on_ellipse_l2425_242596

open Real

noncomputable def max_x_plus_y : ℝ := 343 / 88

theorem maxvalue_on_ellipse (x y : ℝ) :
  (x^2 + 3 * x * y + 2 * y^2 - 14 * x - 21 * y + 49 = 0) →
  x + y ≤ max_x_plus_y := 
sorry

end NUMINAMATH_GPT_maxvalue_on_ellipse_l2425_242596


namespace NUMINAMATH_GPT_total_routes_A_to_B_l2425_242555

-- Define the conditions
def routes_A_to_C : ℕ := 4
def routes_C_to_B : ℕ := 2

-- Statement to prove
theorem total_routes_A_to_B : (routes_A_to_C * routes_C_to_B = 8) :=
by
  -- Omitting the proof, but stating that there is a total of 8 routes from A to B
  sorry

end NUMINAMATH_GPT_total_routes_A_to_B_l2425_242555


namespace NUMINAMATH_GPT_symmetry_center_2tan_2x_sub_pi_div_4_l2425_242592

theorem symmetry_center_2tan_2x_sub_pi_div_4 (k : ℤ) :
  ∃ (x : ℝ), 2 * (x) - π / 4 = k * π / 2 ∧ x = k * π / 4 + π / 8 :=
by
  sorry

end NUMINAMATH_GPT_symmetry_center_2tan_2x_sub_pi_div_4_l2425_242592


namespace NUMINAMATH_GPT_diameter_increase_l2425_242581

theorem diameter_increase (A A' D D' : ℝ)
  (hA_increase: A' = 4 * A)
  (hA: A = π * (D / 2)^2)
  (hA': A' = π * (D' / 2)^2) :
  D' = 2 * D :=
by 
  sorry

end NUMINAMATH_GPT_diameter_increase_l2425_242581


namespace NUMINAMATH_GPT_leak_empties_cistern_in_24_hours_l2425_242588

noncomputable def cistern_fill_rate_without_leak : ℝ := 1 / 8
noncomputable def cistern_fill_rate_with_leak : ℝ := 1 / 12

theorem leak_empties_cistern_in_24_hours :
  (1 / (cistern_fill_rate_without_leak - cistern_fill_rate_with_leak)) = 24 :=
by
  sorry

end NUMINAMATH_GPT_leak_empties_cistern_in_24_hours_l2425_242588


namespace NUMINAMATH_GPT_jenny_sold_192_packs_l2425_242545

-- Define the conditions
def boxes_sold : ℝ := 24.0
def packs_per_box : ℝ := 8.0

-- The total number of packs sold
def total_packs_sold : ℝ := boxes_sold * packs_per_box

-- Proof statement that total packs sold equals 192.0
theorem jenny_sold_192_packs : total_packs_sold = 192.0 :=
by
  sorry

end NUMINAMATH_GPT_jenny_sold_192_packs_l2425_242545


namespace NUMINAMATH_GPT_rectangular_garden_width_l2425_242571

variable (w : ℕ)

/-- The length of a rectangular garden is three times its width.
Given that the area of the rectangular garden is 768 square meters,
prove that the width of the garden is 16 meters. -/
theorem rectangular_garden_width
  (h1 : 768 = w * (3 * w)) :
  w = 16 := by
  sorry

end NUMINAMATH_GPT_rectangular_garden_width_l2425_242571


namespace NUMINAMATH_GPT_units_digit_7_pow_75_plus_6_l2425_242584

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_7_pow_75_plus_6 : units_digit (7 ^ 75 + 6) = 9 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_75_plus_6_l2425_242584


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2425_242506

variables (a_n b_n : ℕ → ℚ)
variables (S_n T_n : ℕ → ℚ)
variable (n : ℕ)

axiom sum_a_terms : ∀ n : ℕ, S_n n = n / 2 * (a_n 1 + a_n n)
axiom sum_b_terms : ∀ n : ℕ, T_n n = n / 2 * (b_n 1 + b_n n)
axiom given_fraction : ∀ n : ℕ, n > 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)

theorem arithmetic_sequence_problem : 
  (a_n 10) / (b_n 3 + b_n 18) + (a_n 11) / (b_n 6 + b_n 15) = 41 / 78 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2425_242506


namespace NUMINAMATH_GPT_quadrant_of_alpha_l2425_242583

theorem quadrant_of_alpha (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end NUMINAMATH_GPT_quadrant_of_alpha_l2425_242583


namespace NUMINAMATH_GPT_first_discount_is_10_l2425_242531

def list_price : ℝ := 70
def final_price : ℝ := 59.85
def second_discount : ℝ := 0.05

theorem first_discount_is_10 :
  ∃ (x : ℝ), list_price * (1 - x/100) * (1 - second_discount) = final_price ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_is_10_l2425_242531


namespace NUMINAMATH_GPT_exp_decreasing_range_l2425_242579

theorem exp_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (a-2) ^ x < (a-2) ^ (x - 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_exp_decreasing_range_l2425_242579


namespace NUMINAMATH_GPT_white_pawn_on_white_square_l2425_242542

theorem white_pawn_on_white_square (w b N_b N_w : ℕ) (h1 : w > b) (h2 : N_b < N_w) : ∃ k : ℕ, k > 0 :=
by 
  -- Let's assume a contradiction
  -- The proof steps would be written here
  sorry

end NUMINAMATH_GPT_white_pawn_on_white_square_l2425_242542


namespace NUMINAMATH_GPT_max_value_of_m_l2425_242590

theorem max_value_of_m (x m : ℝ) (h1 : x^2 - 4*x - 5 > 0) (h2 : x^2 - 2*x + 1 - m^2 > 0) (hm : m > 0) 
(hsuff : ∀ (x : ℝ), (x < -1 ∨ x > 5) → (x > m + 1 ∨ x < 1 - m)) : m ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_m_l2425_242590


namespace NUMINAMATH_GPT_gardener_area_l2425_242557

-- The definition considers the placement of gardeners and the condition for attending flowers.
noncomputable def grid_assignment (gardener_position: (ℕ × ℕ)) (flower_position: (ℕ × ℕ)) : List (ℕ × ℕ) :=
  sorry

-- A theorem that states the equivalent proof.
theorem gardener_area (gardener_position: (ℕ × ℕ)) :
  ∀ flower_position: (ℕ × ℕ), (∃ g1 g2 g3, g1 ∈ grid_assignment gardener_position flower_position ∧
                                            g2 ∈ grid_assignment gardener_position flower_position ∧
                                            g3 ∈ grid_assignment gardener_position flower_position) →
  (gardener_position = g1 ∨ gardener_position = g2 ∨ gardener_position = g3) → true :=
by
  sorry

end NUMINAMATH_GPT_gardener_area_l2425_242557


namespace NUMINAMATH_GPT_redistribute_oil_l2425_242502

def total_boxes (trucks1 trucks2 boxes1 boxes2 : Nat) :=
  (trucks1 * boxes1) + (trucks2 * boxes2)

def total_containers (boxes containers_per_box : Nat) :=
  boxes * containers_per_box

def containers_per_truck (total_containers trucks : Nat) :=
  total_containers / trucks

theorem redistribute_oil :
  ∀ (trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks : Nat),
  trucks1 = 7 →
  trucks2 = 5 →
  boxes1 = 20 →
  boxes2 = 12 →
  containers_per_box = 8 →
  total_trucks = 10 →
  containers_per_truck (total_containers (total_boxes trucks1 trucks2 boxes1 boxes2) containers_per_box) total_trucks = 160 :=
by
  intros trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks
  intros h_trucks1 h_trucks2 h_boxes1 h_boxes2 h_containers_per_box h_total_trucks
  sorry

end NUMINAMATH_GPT_redistribute_oil_l2425_242502


namespace NUMINAMATH_GPT_total_volume_of_removed_pyramids_l2425_242541

noncomputable def volume_of_removed_pyramids (edge_length : ℝ) : ℝ :=
  8 * (1 / 3 * (1 / 2 * (edge_length / 4) * (edge_length / 4)) * (edge_length / 4) / 6)

theorem total_volume_of_removed_pyramids :
  volume_of_removed_pyramids 1 = 1 / 48 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_removed_pyramids_l2425_242541


namespace NUMINAMATH_GPT_problem1_f_x_linear_problem2_f_x_l2425_242556

-- Problem 1 statement: Prove f(x) = 2x + 7 given conditions
theorem problem1_f_x_linear (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = 2 * x + 7)
  (h2 : ∀ x, 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) : 
  ∀ x, f x = 2 * x + 7 :=
by sorry

-- Problem 2 statement: Prove f(x) = 2x - 1/x given conditions
theorem problem2_f_x (f : ℝ → ℝ) 
  (h1 : ∀ x, 2 * f x + f (1 / x) = 3 * x) : 
  ∀ x, f x = 2 * x - 1 / x :=
by sorry

end NUMINAMATH_GPT_problem1_f_x_linear_problem2_f_x_l2425_242556


namespace NUMINAMATH_GPT_net_effect_transactions_l2425_242554

theorem net_effect_transactions {a o : ℝ} (h1 : 3 * a / 4 = 15000) (h2 : 5 * o / 4 = 15000) :
  a + o - (2 * 15000) = 2000 :=
by
  sorry

end NUMINAMATH_GPT_net_effect_transactions_l2425_242554


namespace NUMINAMATH_GPT_prime_count_at_least_two_l2425_242532

theorem prime_count_at_least_two :
  ∃ (n1 n2 : ℕ), n1 ≥ 2 ∧ n2 ≥ 2 ∧ (n1 ≠ n2) ∧ Prime (n1^3 + n1^2 + 1) ∧ Prime (n2^3 + n2^2 + 1) := 
by
  sorry

end NUMINAMATH_GPT_prime_count_at_least_two_l2425_242532


namespace NUMINAMATH_GPT_find_p_a_l2425_242520

variables (p : ℕ → ℝ) (a b : ℕ)

-- Given conditions
axiom p_b : p b = 0.5
axiom p_b_given_a : p b / p a = 0.2 
axiom p_a_inter_b : p a * p b = 0.36

-- Problem statement
theorem find_p_a : p a = 1.8 :=
by
  sorry

end NUMINAMATH_GPT_find_p_a_l2425_242520


namespace NUMINAMATH_GPT_total_games_equal_684_l2425_242573

-- Define the number of players
def n : Nat := 19

-- Define the formula to calculate the total number of games played
def total_games (n : Nat) : Nat := n * (n - 1) * 2

-- The proposition asserting the total number of games equals 684
theorem total_games_equal_684 : total_games n = 684 :=
by
  sorry

end NUMINAMATH_GPT_total_games_equal_684_l2425_242573


namespace NUMINAMATH_GPT_area_of_inscribed_rectangle_l2425_242536

variable (b h x : ℝ)

def is_isosceles_triangle (b h : ℝ) : Prop :=
  b > 0 ∧ h > 0

def is_inscribed_rectangle (b h x : ℝ) : Prop :=
  x > 0 ∧ x < h 

theorem area_of_inscribed_rectangle (h_pos : is_isosceles_triangle b h) 
                                    (rect_pos : is_inscribed_rectangle b h x) : 
                                    ∃ A : ℝ, A = (b / (2 * h)) * x ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_rectangle_l2425_242536


namespace NUMINAMATH_GPT_stamps_difference_l2425_242559

theorem stamps_difference (x : ℕ) (h1: 5 * x / 3 * x = 5 / 3)
(h2: (5 * x - 12) / (3 * x + 12) = 4 / 3) : 
(5 * x - 12) - (3 * x + 12) = 32 := by
sorry

end NUMINAMATH_GPT_stamps_difference_l2425_242559


namespace NUMINAMATH_GPT_hyperbola_line_intersection_unique_l2425_242537

theorem hyperbola_line_intersection_unique :
  ∀ (x y : ℝ), (x^2 / 9 - y^2 = 1) ∧ (y = 1/3 * (x + 1)) → ∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_line_intersection_unique_l2425_242537


namespace NUMINAMATH_GPT_diff_of_squares_535_465_l2425_242582

theorem diff_of_squares_535_465 : (535^2 - 465^2) = 70000 :=
sorry

end NUMINAMATH_GPT_diff_of_squares_535_465_l2425_242582


namespace NUMINAMATH_GPT_find_a_2016_l2425_242594

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (n + 1) / n * a n

theorem find_a_2016 (a : ℕ → ℝ) (h : seq a) : a 2016 = 4032 :=
by
  sorry

end NUMINAMATH_GPT_find_a_2016_l2425_242594


namespace NUMINAMATH_GPT_father_three_times_marika_in_year_l2425_242516

-- Define the given conditions as constants.
def marika_age_2004 : ℕ := 8
def father_age_2004 : ℕ := 32

-- Define the proof goal.
theorem father_three_times_marika_in_year :
  ∃ (x : ℕ), father_age_2004 + x = 3 * (marika_age_2004 + x) → 2004 + x = 2008 := 
by {
  sorry
}

end NUMINAMATH_GPT_father_three_times_marika_in_year_l2425_242516


namespace NUMINAMATH_GPT_negation_of_universal_l2425_242514

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ¬ (∀ x : ℤ, x^3 < 1) ↔ ∃ x : ℤ, x^3 ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_l2425_242514


namespace NUMINAMATH_GPT_inversely_proportional_find_p_l2425_242511

theorem inversely_proportional_find_p (p q : ℕ) (h1 : p * 8 = 160) (h2 : q = 10) : p * q = 160 → p = 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_inversely_proportional_find_p_l2425_242511


namespace NUMINAMATH_GPT_percentage_by_which_x_more_than_y_l2425_242535

theorem percentage_by_which_x_more_than_y
    (x y z : ℝ)
    (h1 : y = 1.20 * z)
    (h2 : z = 150)
    (h3 : x + y + z = 555) :
    ((x - y) / y) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_by_which_x_more_than_y_l2425_242535


namespace NUMINAMATH_GPT_coordinates_of_point_B_l2425_242538

theorem coordinates_of_point_B (A B : ℝ × ℝ) (AB : ℝ) :
  A = (-1, 2) ∧ B.1 = -1 ∧ AB = 3 ∧ (B.2 = 5 ∨ B.2 = -1) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_B_l2425_242538


namespace NUMINAMATH_GPT_train_crosses_bridge_in_time_l2425_242521

noncomputable def length_of_train : ℝ := 125
noncomputable def length_of_bridge : ℝ := 250.03
noncomputable def speed_of_train_kmh : ℝ := 45

noncomputable def speed_of_train_ms : ℝ := (speed_of_train_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_time :
  time_to_cross_bridge = 30.0024 :=
  sorry

end NUMINAMATH_GPT_train_crosses_bridge_in_time_l2425_242521


namespace NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l2425_242548

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := by
  sorry

end NUMINAMATH_GPT_arcsin_one_eq_pi_div_two_l2425_242548


namespace NUMINAMATH_GPT_triangle_exists_l2425_242553

theorem triangle_exists (x : ℕ) (hx : x > 0) :
  (3 * x + 10 > x * x) ∧ (x * x + 10 > 3 * x) ∧ (x * x + 3 * x > 10) ↔ (x = 3 ∨ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_triangle_exists_l2425_242553


namespace NUMINAMATH_GPT_max_x2y_l2425_242575

noncomputable def maximum_value_x_squared_y (x y : ℝ) : ℝ :=
  if x ∈ Set.Ici 0 ∧ y ∈ Set.Ici 0 ∧ x^3 + y^3 + 3*x*y = 1 then x^2 * y else 0

theorem max_x2y (x y : ℝ) (h1 : x ∈ Set.Ici 0) (h2 : y ∈ Set.Ici 0) (h3 : x^3 + y^3 + 3*x*y = 1) :
  maximum_value_x_squared_y x y = 4 / 27 :=
sorry

end NUMINAMATH_GPT_max_x2y_l2425_242575


namespace NUMINAMATH_GPT_problem_l2425_242565

/-- Given a number d > 7, 
    digits A and B in base d such that the equation AB_d + AA_d = 172_d holds, 
    we want to prove that A_d - B_d = 5. --/

theorem problem (d A B : ℕ) (hd : 7 < d)
  (hAB : d * A + B + d * A + A = 1 * d^2 + 7 * d + 2) : A - B = 5 :=
sorry

end NUMINAMATH_GPT_problem_l2425_242565


namespace NUMINAMATH_GPT_kaleb_clothing_problem_l2425_242509

theorem kaleb_clothing_problem 
  (initial_clothing : ℕ) 
  (one_load : ℕ) 
  (remaining_loads : ℕ) : 
  initial_clothing = 39 → one_load = 19 → remaining_loads = 5 → (initial_clothing - one_load) / remaining_loads = 4 :=
sorry

end NUMINAMATH_GPT_kaleb_clothing_problem_l2425_242509


namespace NUMINAMATH_GPT_power_multiplication_l2425_242530

theorem power_multiplication :
  2^4 * 5^4 = 10000 := 
by
  sorry

end NUMINAMATH_GPT_power_multiplication_l2425_242530


namespace NUMINAMATH_GPT_price_per_pie_l2425_242525

-- Define the relevant variables and conditions
def cost_pumpkin_pie : ℕ := 3
def num_pumpkin_pies : ℕ := 10
def cost_cherry_pie : ℕ := 5
def num_cherry_pies : ℕ := 12
def desired_profit : ℕ := 20

-- Total production and profit calculation
def total_cost : ℕ := (cost_pumpkin_pie * num_pumpkin_pies) + (cost_cherry_pie * num_cherry_pies)
def total_earnings_needed : ℕ := total_cost + desired_profit
def total_pies : ℕ := num_pumpkin_pies + num_cherry_pies

-- Proposition to prove that the price per pie should be $5
theorem price_per_pie : (total_earnings_needed / total_pies) = 5 := by
  sorry

end NUMINAMATH_GPT_price_per_pie_l2425_242525


namespace NUMINAMATH_GPT_gcd_654327_543216_is_1_l2425_242524

-- Define the gcd function and relevant numbers
def gcd_problem : Prop :=
  gcd 654327 543216 = 1

-- The statement of the theorem, with a placeholder for the proof
theorem gcd_654327_543216_is_1 : gcd_problem :=
by {
  -- actual proof will go here
  sorry
}

end NUMINAMATH_GPT_gcd_654327_543216_is_1_l2425_242524


namespace NUMINAMATH_GPT_brick_height_l2425_242561

theorem brick_height (H : ℝ) 
    (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
    (brick_length : ℝ) (brick_width : ℝ) (num_bricks : ℝ)
    (volume_wall: wall_length = 900 ∧ wall_width = 500 ∧ wall_height = 1850)
    (volume_brick: brick_length = 21 ∧ brick_width = 10)
    (num_bricks_value: num_bricks = 4955.357142857142) :
    (H = 0.8) :=
by {
  sorry
}

end NUMINAMATH_GPT_brick_height_l2425_242561


namespace NUMINAMATH_GPT_reduced_price_is_25_l2425_242586

noncomputable def original_price (P : ℝ) := P
noncomputable def reduced_price (P : ℝ) := P * 0.85
noncomputable def amount_of_wheat_original (P : ℝ) := 500 / P
noncomputable def amount_of_wheat_reduced (P : ℝ) := 500 / (P * 0.85)

theorem reduced_price_is_25 : 
  ∃ (P : ℝ), reduced_price P = 25 ∧ (amount_of_wheat_reduced P = amount_of_wheat_original P + 3) :=
sorry

end NUMINAMATH_GPT_reduced_price_is_25_l2425_242586
