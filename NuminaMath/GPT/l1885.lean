import Mathlib

namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l1885_188562

theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) :
  (h = 5 * Real.sqrt 2) →
  (A = 12.5) →
  ∃ (leg : ℝ), (leg = 5) ∧ (A = 1 / 2 * leg^2) := by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l1885_188562


namespace NUMINAMATH_GPT_sin_330_eq_neg_one_half_l1885_188560

theorem sin_330_eq_neg_one_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_330_eq_neg_one_half_l1885_188560


namespace NUMINAMATH_GPT_lcm_36_125_l1885_188512

-- Define the prime factorizations
def factorization_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
def factorization_125 : List (ℕ × ℕ) := [(5, 3)]

-- Least common multiple definition
noncomputable def my_lcm (a b : ℕ) : ℕ :=
  a * b / (Nat.gcd a b)

-- Theorem to prove
theorem lcm_36_125 : my_lcm 36 125 = 4500 :=
by
  sorry

end NUMINAMATH_GPT_lcm_36_125_l1885_188512


namespace NUMINAMATH_GPT_maria_age_l1885_188587

variable (M J : Nat)

theorem maria_age (h1 : J = M + 12) (h2 : M + J = 40) : M = 14 := by
  sorry

end NUMINAMATH_GPT_maria_age_l1885_188587


namespace NUMINAMATH_GPT_train_people_count_l1885_188513

theorem train_people_count :
  let initial := 48
  let after_first_stop := initial - 13 + 5
  let after_second_stop := after_first_stop - 9 + 10 - 2
  let after_third_stop := after_second_stop - 7 + 4 - 3
  let after_fourth_stop := after_third_stop - 16 + 7 - 5
  let after_fifth_stop := after_fourth_stop - 8 + 15
  after_fifth_stop = 26 := sorry

end NUMINAMATH_GPT_train_people_count_l1885_188513


namespace NUMINAMATH_GPT_integer_solutions_are_zero_l1885_188535

-- Definitions for integers and the given equation
def satisfies_equation (a b : ℤ) : Prop :=
  a^2 * b^2 = a^2 + b^2

-- The main statement to prove
theorem integer_solutions_are_zero :
  ∀ (a b : ℤ), satisfies_equation a b → (a = 0 ∧ b = 0) :=
sorry

end NUMINAMATH_GPT_integer_solutions_are_zero_l1885_188535


namespace NUMINAMATH_GPT_part1_part2_l1885_188565

def P (a : ℝ) := ∀ x : ℝ, x^2 - a * x + a + 5 / 4 > 0
def Q (a : ℝ) := 4 * a + 7 ≠ 0 ∧ a - 3 ≠ 0 ∧ (4 * a + 7) * (a - 3) < 0

theorem part1 (h : Q a) : -7 / 4 < a ∧ a < 3 := sorry

theorem part2 (h : (P a ∨ Q a) ∧ ¬(P a ∧ Q a)) :
  (-7 / 4 < a ∧ a ≤ -1) ∨ (3 ≤ a ∧ a < 5) := sorry

end NUMINAMATH_GPT_part1_part2_l1885_188565


namespace NUMINAMATH_GPT_track_length_l1885_188504

theorem track_length (L : ℕ)
  (h1 : ∃ B S : ℕ, B = 120 ∧ (L - B) = S ∧ (S + 200) - B = (L + 80) - B)
  (h2 : L + 80 = 440 - L) : L = 180 := 
  by
    sorry

end NUMINAMATH_GPT_track_length_l1885_188504


namespace NUMINAMATH_GPT_sum_of_circumferences_eq_28pi_l1885_188531

theorem sum_of_circumferences_eq_28pi (R r : ℝ) (h1 : r = (1:ℝ)/3 * R) (h2 : R - r = 7) : 
  2 * Real.pi * R + 2 * Real.pi * r = 28 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sum_of_circumferences_eq_28pi_l1885_188531


namespace NUMINAMATH_GPT_divisibility_by_120_l1885_188569

theorem divisibility_by_120 (n : ℕ) : 120 ∣ (n^7 - n^3) :=
sorry

end NUMINAMATH_GPT_divisibility_by_120_l1885_188569


namespace NUMINAMATH_GPT_candidate_final_score_l1885_188572

/- Given conditions -/
def interview_score : ℤ := 80
def written_test_score : ℤ := 90
def interview_weight : ℤ := 3
def written_test_weight : ℤ := 2

/- Final score computation -/
noncomputable def final_score : ℤ :=
  (interview_score * interview_weight + written_test_score * written_test_weight) / (interview_weight + written_test_weight)

theorem candidate_final_score : final_score = 84 := 
by
  sorry

end NUMINAMATH_GPT_candidate_final_score_l1885_188572


namespace NUMINAMATH_GPT_complex_number_solution_l1885_188529

def imaginary_unit : ℂ := Complex.I -- defining the imaginary unit

theorem complex_number_solution (z : ℂ) (h : z / (z - imaginary_unit) = imaginary_unit) :
  z = (1 / 2 : ℂ) + (1 / 2 : ℂ) * imaginary_unit :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l1885_188529


namespace NUMINAMATH_GPT_book_price_distribution_l1885_188584

theorem book_price_distribution :
  ∃ (x y z: ℤ), 
  x + y + z = 109 ∧
  (34 * x + 27.5 * y + 17.5 * z : ℝ) = 2845 ∧
  (x - y : ℤ).natAbs ≤ 2 ∧ (y - z).natAbs ≤ 2 := 
sorry

end NUMINAMATH_GPT_book_price_distribution_l1885_188584


namespace NUMINAMATH_GPT_weight_of_one_pencil_l1885_188514

theorem weight_of_one_pencil (total_weight : ℝ) (num_pencils : ℕ) (H : total_weight = 141.5) (H' : num_pencils = 5) : (total_weight / num_pencils) = 28.3 :=
by sorry

end NUMINAMATH_GPT_weight_of_one_pencil_l1885_188514


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l1885_188542

theorem hcf_of_two_numbers (H : ℕ) 
(lcm_def : lcm a b = H * 13 * 14) 
(h : a = 280 ∨ b = 280) 
(is_factor_h : H ∣ 280) : 
H = 5 :=
sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l1885_188542


namespace NUMINAMATH_GPT_max_necklaces_with_beads_l1885_188596

noncomputable def necklace_problem : Prop :=
  ∃ (necklaces : ℕ),
    let green_beads := 200
    let white_beads := 100
    let orange_beads := 50
    let beads_per_pattern_green := 3
    let beads_per_pattern_white := 1
    let beads_per_pattern_orange := 1
    necklaces = orange_beads ∧
    green_beads / beads_per_pattern_green >= necklaces ∧
    white_beads / beads_per_pattern_white >= necklaces ∧
    orange_beads / beads_per_pattern_orange >= necklaces

theorem max_necklaces_with_beads : necklace_problem :=
  sorry

end NUMINAMATH_GPT_max_necklaces_with_beads_l1885_188596


namespace NUMINAMATH_GPT_find_n_satisfying_conditions_l1885_188519

noncomputable def exists_set_satisfying_conditions (n : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧
  (∀ x ∈ S, x < 2^(n-1)) ∧
  ∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ B → A ≠ ∅ → B ≠ ∅ → A.sum id ≠ B.sum id

theorem find_n_satisfying_conditions : ∀ n : ℕ, (n ≥ 4) ↔ exists_set_satisfying_conditions n :=
sorry

end NUMINAMATH_GPT_find_n_satisfying_conditions_l1885_188519


namespace NUMINAMATH_GPT_cos4_x_minus_sin4_x_l1885_188559

theorem cos4_x_minus_sin4_x (x : ℝ) (h : x = π / 12) : (Real.cos x) ^ 4 - (Real.sin x) ^ 4 = (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_cos4_x_minus_sin4_x_l1885_188559


namespace NUMINAMATH_GPT_largest_number_is_B_l1885_188550

noncomputable def numA : ℝ := 7.196533
noncomputable def numB : ℝ := 7.19655555555555555555555555555555555555 -- 7.196\overline{5}
noncomputable def numC : ℝ := 7.1965656565656565656565656565656565 -- 7.19\overline{65}
noncomputable def numD : ℝ := 7.196596596596596596596596596596596 -- 7.1\overline{965}
noncomputable def numE : ℝ := 7.196519651965196519651965196519651 -- 7.\overline{1965}

theorem largest_number_is_B : 
  numB > numA ∧ numB > numC ∧ numB > numD ∧ numB > numE :=
by
  sorry

end NUMINAMATH_GPT_largest_number_is_B_l1885_188550


namespace NUMINAMATH_GPT_undefined_denominator_values_l1885_188500

theorem undefined_denominator_values (a : ℝ) : a = 3 ∨ a = -3 ↔ ∃ b : ℝ, (a - b) * (a + b) = 0 := by
  sorry

end NUMINAMATH_GPT_undefined_denominator_values_l1885_188500


namespace NUMINAMATH_GPT_abc_le_one_eighth_l1885_188558

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_abc_le_one_eighth_l1885_188558


namespace NUMINAMATH_GPT_half_vectorAB_is_2_1_l1885_188526

def point := ℝ × ℝ -- Define a point as a pair of real numbers
def vector := ℝ × ℝ -- Define a vector as a pair of real numbers

def A : point := (-1, 0) -- Define point A
def B : point := (3, 2) -- Define point B

noncomputable def vectorAB : vector := (B.1 - A.1, B.2 - A.2) -- Define vector AB as B - A

noncomputable def half_vectorAB : vector := (1 / 2 * vectorAB.1, 1 / 2 * vectorAB.2) -- Define half of vector AB

theorem half_vectorAB_is_2_1 : half_vectorAB = (2, 1) := by
  -- Sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_half_vectorAB_is_2_1_l1885_188526


namespace NUMINAMATH_GPT_compare_magnitudes_l1885_188540

noncomputable
def f (x : ℝ) : ℝ := Real.cos (Real.cos x)

noncomputable
def g (x : ℝ) : ℝ := Real.sin (Real.sin x)

theorem compare_magnitudes : ∀ x : ℝ, f x > g x :=
by
  sorry

end NUMINAMATH_GPT_compare_magnitudes_l1885_188540


namespace NUMINAMATH_GPT_union_of_A_and_B_l1885_188520

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1885_188520


namespace NUMINAMATH_GPT_infinite_values_prime_divisor_l1885_188592

noncomputable def largestPrimeDivisor (n : ℕ) : ℕ :=
  sorry

theorem infinite_values_prime_divisor :
  ∃ᶠ n in at_top, largestPrimeDivisor (n^2 + n + 1) = largestPrimeDivisor ((n+1)^2 + (n+1) + 1) :=
sorry

end NUMINAMATH_GPT_infinite_values_prime_divisor_l1885_188592


namespace NUMINAMATH_GPT_even_function_phi_l1885_188554

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

noncomputable def f' (x φ : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + φ)

noncomputable def y (x φ : ℝ) : ℝ := f x φ + f' x φ

def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

theorem even_function_phi :
  (∀ x : ℝ, y x φ = y (-x) φ) → ∃ k : ℤ, φ = -Real.pi / 3 + k * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_even_function_phi_l1885_188554


namespace NUMINAMATH_GPT_final_price_after_adjustments_l1885_188571

theorem final_price_after_adjustments (p : ℝ) :
  let increased_price := p * 1.30
  let discounted_price := increased_price * 0.75
  let final_price := discounted_price * 1.10
  final_price = 1.0725 * p :=
by
  sorry

end NUMINAMATH_GPT_final_price_after_adjustments_l1885_188571


namespace NUMINAMATH_GPT_smallest_integer_in_set_l1885_188548

theorem smallest_integer_in_set : 
  ∀ (n : ℤ), (n + 6 < 2 * (n + 3)) → n ≥ 1 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_integer_in_set_l1885_188548


namespace NUMINAMATH_GPT_min_a1_value_l1885_188532

theorem min_a1_value (a : ℕ → ℝ) :
  (∀ n > 1, a n = 9 * a (n-1) - 2 * n) →
  (∀ n, a n > 0) →
  (∀ x, (∀ n > 1, a n = 9 * a (n-1) - 2 * n) → (∀ n, a n > 0) → x ≥ a 1) →
  a 1 = 499.25 / 648 :=
sorry

end NUMINAMATH_GPT_min_a1_value_l1885_188532


namespace NUMINAMATH_GPT_no_real_roots_implies_negative_l1885_188507

theorem no_real_roots_implies_negative (m : ℝ) : (¬ ∃ x : ℝ, x^2 = m) → m < 0 :=
sorry

end NUMINAMATH_GPT_no_real_roots_implies_negative_l1885_188507


namespace NUMINAMATH_GPT_most_stable_scores_l1885_188522

structure StudentScores :=
  (average : ℝ)
  (variance : ℝ)

def studentA : StudentScores := { average := 132, variance := 38 }
def studentB : StudentScores := { average := 132, variance := 10 }
def studentC : StudentScores := { average := 132, variance := 26 }

theorem most_stable_scores :
  studentB.variance < studentA.variance ∧ studentB.variance < studentC.variance :=
by 
  sorry

end NUMINAMATH_GPT_most_stable_scores_l1885_188522


namespace NUMINAMATH_GPT_train_length_l1885_188534

theorem train_length (time : ℝ) (speed_in_kmph : ℝ) (speed_in_mps : ℝ) (length_of_train : ℝ) :
  (time = 6) →
  (speed_in_kmph = 96) →
  (speed_in_mps = speed_in_kmph * (5 / 18)) →
  length_of_train = speed_in_mps * time →
  length_of_train = 480 := by
  sorry

end NUMINAMATH_GPT_train_length_l1885_188534


namespace NUMINAMATH_GPT_quadratic_real_roots_condition_l1885_188505

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0) → m ≤ 1/4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_condition_l1885_188505


namespace NUMINAMATH_GPT_determine_n_l1885_188527

theorem determine_n (n : ℕ) (h : n ≥ 2)
    (condition : ∀ i j : ℕ, i ≤ n → j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) :
    ∃ k : ℕ, k ≥ 2 ∧ n = 2^k - 2 := 
sorry

end NUMINAMATH_GPT_determine_n_l1885_188527


namespace NUMINAMATH_GPT_reflect_over_y_axis_matrix_l1885_188524

theorem reflect_over_y_axis_matrix :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![![ -1, 0], ![0, 1]] :=
  -- Proof
  sorry

end NUMINAMATH_GPT_reflect_over_y_axis_matrix_l1885_188524


namespace NUMINAMATH_GPT_angle_sum_in_hexagon_l1885_188574

theorem angle_sum_in_hexagon (P Q R s t : ℝ) 
    (hP: P = 40) (hQ: Q = 88) (hR: R = 30)
    (hex_sum: 6 * 180 - 720 = 0): 
    s + t = 312 :=
by
  have hex_interior_sum: 6 * 180 - 720 = 0 := hex_sum
  sorry

end NUMINAMATH_GPT_angle_sum_in_hexagon_l1885_188574


namespace NUMINAMATH_GPT_polynomial_remainder_l1885_188530

theorem polynomial_remainder (x : ℂ) : (x^1500) % (x^3 - 1) = 1 := 
sorry

end NUMINAMATH_GPT_polynomial_remainder_l1885_188530


namespace NUMINAMATH_GPT_solve_box_dimensions_l1885_188598

theorem solve_box_dimensions (m n r : ℕ) (h1 : m ≤ n) (h2 : n ≤ r) (h3 : m ≥ 1) (h4 : n ≥ 1) (h5 : r ≥ 1) :
  let k₀ := (m - 2) * (n - 2) * (r - 2)
  let k₁ := 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2))
  let k₂ := 4 * ((m - 2) + (n - 2) + (r - 2))
  (k₀ + k₂ - k₁ = 1985) ↔ ((m = 5 ∧ n = 7 ∧ r = 663) ∨ 
                            (m = 5 ∧ n = 5 ∧ r = 1981) ∨
                            (m = 3 ∧ n = 3 ∧ r = 1981) ∨
                            (m = 1 ∧ n = 7 ∧ r = 399) ∨
                            (m = 1 ∧ n = 3 ∧ r = 1987)) :=
sorry

end NUMINAMATH_GPT_solve_box_dimensions_l1885_188598


namespace NUMINAMATH_GPT_rectangle_dimensions_l1885_188517

theorem rectangle_dimensions (w l : ℚ) (h1 : 2 * l + 2 * w = 2 * l * w) (h2 : l = 3 * w) :
  w = 4 / 3 ∧ l = 4 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1885_188517


namespace NUMINAMATH_GPT_find_number_l1885_188586

theorem find_number (x : ℤ) (n : ℤ) (h1 : x = 88320) (h2 : x + 1315 + n - 1569 = 11901) : n = -75165 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l1885_188586


namespace NUMINAMATH_GPT_sin_neg_seven_pi_over_three_correct_l1885_188503

noncomputable def sin_neg_seven_pi_over_three : Prop :=
  (Real.sin (-7 * Real.pi / 3) = - (Real.sqrt 3 / 2))

theorem sin_neg_seven_pi_over_three_correct : sin_neg_seven_pi_over_three := 
by
  sorry

end NUMINAMATH_GPT_sin_neg_seven_pi_over_three_correct_l1885_188503


namespace NUMINAMATH_GPT_tens_digit_of_8_pow_2023_l1885_188566

theorem tens_digit_of_8_pow_2023 :
    ∃ d, 0 ≤ d ∧ d < 10 ∧ (8^2023 % 100) / 10 = d ∧ d = 1 :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_of_8_pow_2023_l1885_188566


namespace NUMINAMATH_GPT_arrange_balls_l1885_188533

/-- Given 4 yellow balls and 3 red balls, we want to prove that there are 35 different ways to arrange these balls in a row. -/
theorem arrange_balls : (Nat.choose 7 4) = 35 := by
  sorry

end NUMINAMATH_GPT_arrange_balls_l1885_188533


namespace NUMINAMATH_GPT_sum_a_b_l1885_188525

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem sum_a_b (a b : ℝ) 
  (H : ∀ x, 2 < x ∧ x < 3 → otimes (x - a) (x - b) > 0) : a + b = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_a_b_l1885_188525


namespace NUMINAMATH_GPT_R_depends_on_d_and_n_l1885_188564

-- Define the given properties of the arithmetic progression sums
def s1 (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2
def s3 (a d n : ℕ) : ℕ := (3 * n * (2 * a + (3 * n - 1) * d)) / 2
def s5 (a d n : ℕ) : ℕ := (5 * n * (2 * a + (5 * n - 1) * d)) / 2

-- Define R in terms of s1, s3, and s5
def R (a d n : ℕ) : ℕ := s5 a d n - s3 a d n - s1 a d n

-- The main theorem to prove the statement about R's dependency
theorem R_depends_on_d_and_n (a d n : ℕ) : R a d n = 7 * d * n^2 := by 
  sorry

end NUMINAMATH_GPT_R_depends_on_d_and_n_l1885_188564


namespace NUMINAMATH_GPT_find_a_plus_b_l1885_188510

theorem find_a_plus_b (a b : ℝ) : (3 = 1/3 * 1 + a) → (1 = 1/3 * 3 + b) → a + b = 8/3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1885_188510


namespace NUMINAMATH_GPT_fraction_second_year_not_third_year_l1885_188555

theorem fraction_second_year_not_third_year (N T S : ℕ) (hN : N = 100) (hT : T = N / 2) (hS : S = N * 3 / 10) :
  (S / (N - T) : ℚ) = 3 / 5 :=
by
  rw [hN, hT, hS]
  norm_num
  sorry

end NUMINAMATH_GPT_fraction_second_year_not_third_year_l1885_188555


namespace NUMINAMATH_GPT_S10_value_l1885_188521

def sequence_sum (n : ℕ) : ℕ :=
  (2^(n+1)) - 2 - n

theorem S10_value : sequence_sum 10 = 2036 := by
  sorry

end NUMINAMATH_GPT_S10_value_l1885_188521


namespace NUMINAMATH_GPT_triangle_is_isosceles_range_of_expression_l1885_188578

variable {a b c A B C : ℝ}
variable (triangle_ABC : 0 < A ∧ A < π ∧ 0 < B ∧ B < π)
variable (opposite_sides : a = 1 ∧ b = 1 ∧ c = 1)
variable (cos_condition : a * Real.cos B = b * Real.cos A)

theorem triangle_is_isosceles (h : a * Real.cos B = b * Real.cos A) : A = B := sorry

theorem range_of_expression 
  (h1 : 0 < A ∧ A < π/2) 
  (h2 : a * Real.cos B = b * Real.cos A) : 
  -3/2 < Real.sin (2 * A + π / 6) - 2 * Real.cos B ^ 2 ∧ Real.sin (2 * A + π / 6) - 2 * Real.cos B ^ 2 < 0 := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_range_of_expression_l1885_188578


namespace NUMINAMATH_GPT_range_of_a_l1885_188570

-- Define the inequality condition
def inequality (x a : ℝ) : Prop :=
  2 * x^2 + a * x - a^2 > 0

-- State the main problem
theorem range_of_a (a: ℝ) : 
  inequality 2 a -> (-2 < a) ∧ (a < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1885_188570


namespace NUMINAMATH_GPT_geom_seq_m_equals_11_l1885_188528

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  ∀ (n : ℕ), a n = a1 * q ^ n

theorem geom_seq_m_equals_11 {a : ℕ → ℝ} {q : ℝ} (hq : q ≠ 1) 
  (h : geometric_sequence a 1 q) : 
  a 11 = a 1 * a 2 * a 3 * a 4 * a 5 := 
by sorry

end NUMINAMATH_GPT_geom_seq_m_equals_11_l1885_188528


namespace NUMINAMATH_GPT_clock_hand_overlaps_in_24_hours_l1885_188577

-- Define the number of revolutions of the hour hand in 24 hours.
def hour_hand_revolutions_24_hours : ℕ := 2

-- Define the number of revolutions of the minute hand in 24 hours.
def minute_hand_revolutions_24_hours : ℕ := 24

-- Define the number of overlaps as a constant.
def number_of_overlaps (hour_rev : ℕ) (minute_rev : ℕ) : ℕ :=
  minute_rev - hour_rev

-- The theorem we want to prove:
theorem clock_hand_overlaps_in_24_hours :
  number_of_overlaps hour_hand_revolutions_24_hours minute_hand_revolutions_24_hours = 22 :=
sorry

end NUMINAMATH_GPT_clock_hand_overlaps_in_24_hours_l1885_188577


namespace NUMINAMATH_GPT_fourth_and_fifth_suppliers_cars_equal_l1885_188579

-- Define the conditions
def total_cars : ℕ := 5650000
def cars_supplier_1 : ℕ := 1000000
def cars_supplier_2 : ℕ := cars_supplier_1 + 500000
def cars_supplier_3 : ℕ := cars_supplier_1 + cars_supplier_2
def cars_distributed_first_three : ℕ := cars_supplier_1 + cars_supplier_2 + cars_supplier_3
def cars_remaining : ℕ := total_cars - cars_distributed_first_three

-- Theorem stating the question and answer
theorem fourth_and_fifth_suppliers_cars_equal 
  : (cars_remaining / 2) = 325000 := by
  sorry

end NUMINAMATH_GPT_fourth_and_fifth_suppliers_cars_equal_l1885_188579


namespace NUMINAMATH_GPT_greatest_integer_x_l1885_188556

theorem greatest_integer_x (x : ℤ) (h : (5 : ℚ) / 8 > (x : ℚ) / 17) : x ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_x_l1885_188556


namespace NUMINAMATH_GPT_like_terms_exponent_l1885_188543

theorem like_terms_exponent (x y : ℝ) (n : ℕ) : 
  (∀ (a b : ℝ), a * x ^ 3 * y ^ (n - 1) = b * x ^ 3 * y ^ 1 → n = 2) :=
by
  sorry

end NUMINAMATH_GPT_like_terms_exponent_l1885_188543


namespace NUMINAMATH_GPT_unique_pair_exists_for_each_n_l1885_188599

theorem unique_pair_exists_for_each_n (n : ℕ) (h : n > 0) : 
  ∃! (a b : ℕ), a > 0 ∧ b > 0 ∧ n = (a + b - 1) * (a + b - 2) / 2 + a :=
sorry

end NUMINAMATH_GPT_unique_pair_exists_for_each_n_l1885_188599


namespace NUMINAMATH_GPT_length_of_first_two_CDs_l1885_188575

theorem length_of_first_two_CDs
  (x : ℝ)
  (h1 : x + x + 2 * x = 6) :
  x = 1.5 := 
sorry

end NUMINAMATH_GPT_length_of_first_two_CDs_l1885_188575


namespace NUMINAMATH_GPT_kaleb_can_buy_toys_l1885_188545

def kaleb_initial_money : ℕ := 12
def money_spent_on_game : ℕ := 8
def money_saved : ℕ := 2
def toy_cost : ℕ := 2

theorem kaleb_can_buy_toys :
  (kaleb_initial_money - money_spent_on_game - money_saved) / toy_cost = 1 :=
by
  sorry

end NUMINAMATH_GPT_kaleb_can_buy_toys_l1885_188545


namespace NUMINAMATH_GPT_smallest_b_l1885_188502

theorem smallest_b (b: ℕ) (h1: b > 3) (h2: ∃ n: ℕ, n^3 = 2 * b + 3) : b = 12 :=
sorry

end NUMINAMATH_GPT_smallest_b_l1885_188502


namespace NUMINAMATH_GPT_tank_volume_ratio_l1885_188585

theorem tank_volume_ratio (A B : ℝ) 
    (h : (3 / 4) * A = (5 / 8) * B) : A / B = 6 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_tank_volume_ratio_l1885_188585


namespace NUMINAMATH_GPT_pow_15_1234_mod_19_l1885_188551

theorem pow_15_1234_mod_19 : (15^1234) % 19 = 6 := 
by sorry

end NUMINAMATH_GPT_pow_15_1234_mod_19_l1885_188551


namespace NUMINAMATH_GPT_total_job_applications_l1885_188541

theorem total_job_applications (apps_in_state : ℕ) (apps_other_states : ℕ) 
  (h1 : apps_in_state = 200)
  (h2 : apps_other_states = 2 * apps_in_state) :
  apps_in_state + apps_other_states = 600 :=
by
  sorry

end NUMINAMATH_GPT_total_job_applications_l1885_188541


namespace NUMINAMATH_GPT_prob_A_not_losing_prob_A_not_winning_l1885_188591

-- Definitions based on the conditions
def prob_winning : ℝ := 0.41
def prob_tie : ℝ := 0.27

-- The probability of A not losing
def prob_not_losing : ℝ := prob_winning + prob_tie

-- The probability of A not winning
def prob_not_winning : ℝ := 1 - prob_winning

-- Proof problems
theorem prob_A_not_losing : prob_not_losing = 0.68 := by
  sorry

theorem prob_A_not_winning : prob_not_winning = 0.59 := by
  sorry

end NUMINAMATH_GPT_prob_A_not_losing_prob_A_not_winning_l1885_188591


namespace NUMINAMATH_GPT_pens_exceed_500_on_saturday_l1885_188561

theorem pens_exceed_500_on_saturday :
  ∃ k : ℕ, (5 * 3 ^ k > 500) ∧ k = 6 :=
by 
  sorry   -- Skipping the actual proof here

end NUMINAMATH_GPT_pens_exceed_500_on_saturday_l1885_188561


namespace NUMINAMATH_GPT_range_of_f_l1885_188568

noncomputable def f (x : ℝ) : ℝ := 2^(2*x) + 2^(x+1) + 3

theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y > 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1885_188568


namespace NUMINAMATH_GPT_speed_of_water_current_l1885_188539

theorem speed_of_water_current (v : ℝ) 
  (swimmer_speed_still_water : ℝ := 4) 
  (distance : ℝ := 3) 
  (time : ℝ := 1.5)
  (effective_speed_against_current : ℝ := swimmer_speed_still_water - v) :
  effective_speed_against_current = distance / time → v = 2 := 
by
  -- Proof
  sorry

end NUMINAMATH_GPT_speed_of_water_current_l1885_188539


namespace NUMINAMATH_GPT_solve_for_nabla_l1885_188536

theorem solve_for_nabla (nabla : ℤ) (h : 4 * (-3) = nabla + 3) : nabla = -15 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_nabla_l1885_188536


namespace NUMINAMATH_GPT_password_probability_l1885_188537

theorem password_probability 
  (password : Fin 6 → Fin 10) 
  (attempts : ℕ) 
  (correct_digit : Fin 10) 
  (probability_first_try : ℚ := 1 / 10)
  (probability_second_try : ℚ := (9 / 10) * (1 / 9)) : 
  ((password 5 = correct_digit) ∧ attempts ≤ 2) →
  (probability_first_try + probability_second_try = 1 / 5) :=
sorry

end NUMINAMATH_GPT_password_probability_l1885_188537


namespace NUMINAMATH_GPT_pipe_filling_time_l1885_188557

-- Definitions for the conditions
variables (A : ℝ) (h : 1 / A - 1 / 24 = 1 / 12)

-- The statement of the problem
theorem pipe_filling_time : A = 8 :=
by
  sorry

end NUMINAMATH_GPT_pipe_filling_time_l1885_188557


namespace NUMINAMATH_GPT_length_O_D1_l1885_188582

-- Definitions for the setup of the cube and its faces, the center of the sphere, and the intersecting circles
def O : Point := sorry -- Center of the sphere and cube
def radius : ℝ := 10 -- Radius of the sphere

-- Intersection circles with given radii on specific faces of the cube
def r_ADA1D1 : ℝ := 1 -- Radius of the intersection circle on face ADA1D1
def r_A1B1C1D1 : ℝ := 1 -- Radius of the intersection circle on face A1B1C1D1
def r_CDD1C1 : ℝ := 3 -- Radius of the intersection circle on face CDD1C1

-- Distances derived from the problem
def OX1_sq : ℝ := radius^2 - r_ADA1D1^2
def OX2_sq : ℝ := radius^2 - r_A1B1C1D1^2
def OX_sq : ℝ := radius^2 - r_CDD1C1^2

-- To simplify, replace OX1, OX2, and OX with their squared values directly
def OX1_sq_calc : ℝ := 99
def OX2_sq_calc : ℝ := 99
def OX_sq_calc : ℝ := 91

theorem length_O_D1 : (OX1_sq_calc + OX2_sq_calc + OX_sq_calc) = 289 ↔ OD1 = 17 := by
  sorry

end NUMINAMATH_GPT_length_O_D1_l1885_188582


namespace NUMINAMATH_GPT_sum_of_midpoints_l1885_188595

theorem sum_of_midpoints 
  (a b c d e f : ℝ)
  (h1 : a + b + c = 15)
  (h2 : d + e + f = 15) :
  ((a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15) ∧ 
  ((d + e) / 2 + (d + f) / 2 + (e + f) / 2 = 15) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_l1885_188595


namespace NUMINAMATH_GPT_compare_fractions_l1885_188563

theorem compare_fractions : - (1 + 3 / 5) < -1.5 := 
by
  sorry

end NUMINAMATH_GPT_compare_fractions_l1885_188563


namespace NUMINAMATH_GPT_hyperbola_properties_l1885_188567

open Real

def is_asymptote (y x : ℝ) : Prop :=
  y = (1/2) * x ∨ y = -(1/2) * x

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_properties :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) →
  ∀ (a b c : ℝ), 
  (a = 2) →
  (b = 1) →
  (c = sqrt (a^2 + b^2)) →
  (∀ y x : ℝ, (is_asymptote y x)) ∧ (eccentricity a (sqrt (a^2 + b^2)) = sqrt 5 / 2) :=
by
  intros x y h a b c ha hb hc
  sorry

end NUMINAMATH_GPT_hyperbola_properties_l1885_188567


namespace NUMINAMATH_GPT_appetizer_cost_per_person_l1885_188516

theorem appetizer_cost_per_person
    (cost_per_bag: ℕ)
    (num_bags: ℕ)
    (cost_creme_fraiche: ℕ)
    (cost_caviar: ℕ)
    (num_people: ℕ)
    (h1: cost_per_bag = 1)
    (h2: num_bags = 3)
    (h3: cost_creme_fraiche = 5)
    (h4: cost_caviar = 73)
    (h5: num_people = 3):
    (cost_per_bag * num_bags + cost_creme_fraiche + cost_caviar) / num_people = 27 := 
  by
    sorry

end NUMINAMATH_GPT_appetizer_cost_per_person_l1885_188516


namespace NUMINAMATH_GPT_largest_consecutive_odd_integers_sum_255_l1885_188501

theorem largest_consecutive_odd_integers_sum_255 : 
  ∃ (n : ℤ), (n + (n + 2) + (n + 4) + (n + 6) + (n + 8) = 255) ∧ (n + 8 = 55) :=
by
  sorry

end NUMINAMATH_GPT_largest_consecutive_odd_integers_sum_255_l1885_188501


namespace NUMINAMATH_GPT_comparison_of_large_exponents_l1885_188594

theorem comparison_of_large_exponents : 2^1997 > 5^850 := sorry

end NUMINAMATH_GPT_comparison_of_large_exponents_l1885_188594


namespace NUMINAMATH_GPT_solve_for_x_l1885_188518

theorem solve_for_x : (2 / 5 : ℚ) - (1 / 7) = 1 / (35 / 9) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1885_188518


namespace NUMINAMATH_GPT_min_distance_eq_3_l1885_188549

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (Real.pi / 3 * x + Real.pi / 4)

theorem min_distance_eq_3 (x₁ x₂ : ℝ) 
  (h₁ : f x₁ ≤ f x) (h₂ : f x ≤ f x₂) 
  (x : ℝ) :
  |x₁ - x₂| = 3 :=
by
  -- Sorry placeholder for proof.
  sorry

end NUMINAMATH_GPT_min_distance_eq_3_l1885_188549


namespace NUMINAMATH_GPT_m_squared_minus_n_squared_plus_one_is_perfect_square_l1885_188552

theorem m_squared_minus_n_squared_plus_one_is_perfect_square (m n : ℤ)
  (hm : m % 2 = 1) (hn : n % 2 = 1)
  (h : m^2 - n^2 + 1 ∣ n^2 - 1) :
  ∃ k : ℤ, k^2 = m^2 - n^2 + 1 :=
sorry

end NUMINAMATH_GPT_m_squared_minus_n_squared_plus_one_is_perfect_square_l1885_188552


namespace NUMINAMATH_GPT_quadrilateral_segments_l1885_188580

theorem quadrilateral_segments {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a + b + c + d = 2) (h6 : 1/4 < a) (h7 : a < 1) (h8 : 1/4 < b) (h9 : b < 1)
  (h10 : 1/4 < c) (h11 : c < 1) (h12 : 1/4 < d) (h13 : d < 1) : 
  (a + b > d) ∧ (a + c > d) ∧ (a + d > c) ∧ (b + c > d) ∧ 
  (b + d > c) ∧ (c + d > a) ∧ (a + b + c > d) ∧ (a + b + d > c) ∧
  (a + c + d > b) ∧ (b + c + d > a) :=
sorry

end NUMINAMATH_GPT_quadrilateral_segments_l1885_188580


namespace NUMINAMATH_GPT_find_n_l1885_188583

theorem find_n (n : ℝ) (h1 : (n ≠ 0)) (h2 : ∃ (n' : ℝ), n = n' ∧ -n' = -9 / n') (h3 : ∀ x : ℝ, x > 0 → -n * x < 0) : n = 3 :=
sorry

end NUMINAMATH_GPT_find_n_l1885_188583


namespace NUMINAMATH_GPT_mixture_contains_pecans_l1885_188538

theorem mixture_contains_pecans 
  (price_per_cashew_per_pound : ℝ)
  (cashews_weight : ℝ)
  (price_per_mixture_per_pound : ℝ)
  (price_of_cashews : ℝ)
  (mixture_weight : ℝ)
  (pecans_weight : ℝ)
  (price_per_pecan_per_pound : ℝ)
  (pecans_price : ℝ)
  (total_cost_of_mixture : ℝ)
  
  (h1 : price_per_cashew_per_pound = 3.50) 
  (h2 : cashews_weight = 2)
  (h3 : price_per_mixture_per_pound = 4.34) 
  (h4 : pecans_weight = 1.33333333333)
  (h5 : price_per_pecan_per_pound = 5.60)
  
  (h6 : price_of_cashews = cashews_weight * price_per_cashew_per_pound)
  (h7 : mixture_weight = cashews_weight + pecans_weight)
  (h8 : pecans_price = pecans_weight * price_per_pecan_per_pound)
  (h9 : total_cost_of_mixture = price_of_cashews + pecans_price)

  (h10 : price_per_mixture_per_pound = total_cost_of_mixture / mixture_weight)
  
  : pecans_weight = 1.33333333333 :=
sorry

end NUMINAMATH_GPT_mixture_contains_pecans_l1885_188538


namespace NUMINAMATH_GPT_find_value_of_M_l1885_188511

theorem find_value_of_M (a b M : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = M) (h4 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y ≤ (M^2) / 4) (h5 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y = 2) :
  M = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_M_l1885_188511


namespace NUMINAMATH_GPT_range_of_f_l1885_188546

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^4 + 6 * x^2 + 9

-- Define the domain as [0, ∞)
def domain (x : ℝ) : Prop := x ≥ 0

-- State the theorem which asserts the range of f(x) is [9, ∞)
theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, domain x ∧ f x = y) ↔ y ≥ 9 := by
  sorry

end NUMINAMATH_GPT_range_of_f_l1885_188546


namespace NUMINAMATH_GPT_roundness_1000000_l1885_188589

-- Definitions based on the conditions in the problem
def prime_factors (n : ℕ) : List (ℕ × ℕ) :=
  if n = 1 then []
  else [(2, 6), (5, 6)] -- Example specifically for 1,000,000

def roundness (n : ℕ) : ℕ :=
  (prime_factors n).map Prod.snd |>.sum

-- The main theorem
theorem roundness_1000000 : roundness 1000000 = 12 := by
  sorry

end NUMINAMATH_GPT_roundness_1000000_l1885_188589


namespace NUMINAMATH_GPT_complex_div_i_l1885_188593

open Complex

theorem complex_div_i (z : ℂ) (hz : z = -2 - i) : z / i = -1 + 2 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_div_i_l1885_188593


namespace NUMINAMATH_GPT_convince_jury_l1885_188515

-- Define predicates for being a criminal, normal man, guilty, or a knight
def Criminal : Prop := sorry
def NormalMan : Prop := sorry
def Guilty : Prop := sorry
def Knight : Prop := sorry

-- Define your status
variable (you : Prop)

-- Assumptions as per given conditions
axiom criminal_not_normal_man : Criminal → ¬NormalMan
axiom you_not_guilty : ¬Guilty
axiom you_not_knight : ¬Knight

-- The statement to prove
theorem convince_jury : ¬Guilty ∧ ¬Knight := by
  exact And.intro you_not_guilty you_not_knight

end NUMINAMATH_GPT_convince_jury_l1885_188515


namespace NUMINAMATH_GPT_work_together_days_l1885_188544

theorem work_together_days
  (a_days : ℝ) (ha : a_days = 18)
  (b_days : ℝ) (hb : b_days = 30)
  (c_days : ℝ) (hc : c_days = 45)
  (combined_days : ℝ) :
  (combined_days = 1 / ((1 / a_days) + (1 / b_days) + (1 / c_days))) → combined_days = 9 := 
by
  sorry

end NUMINAMATH_GPT_work_together_days_l1885_188544


namespace NUMINAMATH_GPT_exists_k_simplifies_expression_to_5x_squared_l1885_188547

theorem exists_k_simplifies_expression_to_5x_squared :
  ∃ k : ℝ, (∀ x : ℝ, (x - k * x) * (2 * x - k * x) - 3 * x * (2 * x - k * x) = 5 * x^2) :=
by
  sorry

end NUMINAMATH_GPT_exists_k_simplifies_expression_to_5x_squared_l1885_188547


namespace NUMINAMATH_GPT_polygon_properties_l1885_188588

theorem polygon_properties
  (n : ℕ)
  (h_exterior_angle : 360 / 20 = n)
  (h_n_sides : n = 18) :
  (180 * (n - 2) = 2880) ∧ (n * (n - 3) / 2 = 135) :=
by
  sorry

end NUMINAMATH_GPT_polygon_properties_l1885_188588


namespace NUMINAMATH_GPT_john_taller_than_lena_l1885_188506

-- Define the heights of John, Lena, and Rebeca.
variables (J L R : ℕ)

-- Given conditions:
-- 1. John has a height of 152 cm
axiom john_height : J = 152

-- 2. John is 6 cm shorter than Rebeca
axiom john_shorter_rebeca : J = R - 6

-- 3. The height of Lena and Rebeca together is 295 cm
axiom lena_rebeca_together : L + R = 295

-- Prove that John is 15 cm taller than Lena
theorem john_taller_than_lena : (J - L) = 15 := by
  sorry

end NUMINAMATH_GPT_john_taller_than_lena_l1885_188506


namespace NUMINAMATH_GPT_cubic_diff_l1885_188576

theorem cubic_diff (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 40) : a^3 - b^3 = 208 :=
by
  sorry

end NUMINAMATH_GPT_cubic_diff_l1885_188576


namespace NUMINAMATH_GPT_find_a_equals_two_l1885_188553

noncomputable def a := ((7 + 4 * Real.sqrt 3) ^ (1 / 2) - (7 - 4 * Real.sqrt 3) ^ (1 / 2)) / Real.sqrt 3

theorem find_a_equals_two : a = 2 := 
sorry

end NUMINAMATH_GPT_find_a_equals_two_l1885_188553


namespace NUMINAMATH_GPT_min_b_for_factorization_l1885_188573

theorem min_b_for_factorization : 
  ∃ b : ℕ, (∀ p q : ℤ, (p + q = b) ∧ (p * q = 1764) → x^2 + b * x + 1764 = (x + p) * (x + q)) 
  ∧ b = 84 :=
sorry

end NUMINAMATH_GPT_min_b_for_factorization_l1885_188573


namespace NUMINAMATH_GPT_total_distance_craig_walked_l1885_188509

theorem total_distance_craig_walked :
  0.2 + 0.7 = 0.9 :=
by sorry

end NUMINAMATH_GPT_total_distance_craig_walked_l1885_188509


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1885_188597

theorem quadratic_two_distinct_real_roots : 
  ∀ x : ℝ, ∃ a b c : ℝ, (∀ x : ℝ, (x+1)*(x-1) = 2*x + 3 → x^2 - 2*x - 4 = 0) ∧ 
  (a = 1) ∧ (b = -2) ∧ (c = -4) ∧ (b^2 - 4*a*c > 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1885_188597


namespace NUMINAMATH_GPT_james_jail_time_l1885_188581

-- Definitions based on the conditions
def arson_sentence := 6
def arson_count := 2
def total_arson_sentence := arson_sentence * arson_count

def explosives_sentence := 2 * total_arson_sentence
def terrorism_sentence := 20

-- Total sentence calculation
def total_jail_time := total_arson_sentence + explosives_sentence + terrorism_sentence

-- The theorem we want to prove
theorem james_jail_time : total_jail_time = 56 := by
  sorry

end NUMINAMATH_GPT_james_jail_time_l1885_188581


namespace NUMINAMATH_GPT_cost_of_20_pounds_of_bananas_l1885_188590

noncomputable def cost_of_bananas (rate : ℝ) (amount : ℝ) : ℝ :=
rate * amount / 4

theorem cost_of_20_pounds_of_bananas :
  cost_of_bananas 6 20 = 30 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_20_pounds_of_bananas_l1885_188590


namespace NUMINAMATH_GPT_function_monotonically_increasing_on_interval_l1885_188523

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem function_monotonically_increasing_on_interval (e : ℝ) (h_e_pos : 0 < e) (h_ln_e_pos : 0 < Real.log e) :
  ∀ x : ℝ, e < x → 0 < Real.log x - 1 := 
sorry

end NUMINAMATH_GPT_function_monotonically_increasing_on_interval_l1885_188523


namespace NUMINAMATH_GPT_problem_solution_l1885_188508

noncomputable def f (x : ℝ) := 2 * Real.sin x + x^3 + 1

theorem problem_solution (a : ℝ) (h : f a = 3) : f (-a) = -1 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1885_188508
