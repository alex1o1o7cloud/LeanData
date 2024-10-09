import Mathlib

namespace painted_surface_area_of_pyramid_l47_4718

/--
Given 19 unit cubes arranged in a 4-layer pyramid-like structure, where:
- The top layer has 1 cube,
- The second layer has 3 cubes,
- The third layer has 5 cubes,
- The bottom layer has 10 cubes,

Prove that the total painted surface area is 43 square meters.
-/
theorem painted_surface_area_of_pyramid :
  let layer1 := 1 -- top layer
  let layer2 := 3 -- second layer
  let layer3 := 5 -- third layer
  let layer4 := 10 -- bottom layer
  let total_cubes := layer1 + layer2 + layer3 + layer4
  let top_faces := layer1 * 1 + layer2 * 1 + layer3 * 1 + layer4 * 1
  let side_faces_layer1 := layer1 * 5
  let side_faces_layer2 := layer2 * 3
  let side_faces_layer3 := layer3 * 2
  let side_faces := side_faces_layer1 + side_faces_layer2 + side_faces_layer3
  let total_surface_area := top_faces + side_faces
  total_cubes = 19 → total_surface_area = 43 :=
by
  intros
  sorry

end painted_surface_area_of_pyramid_l47_4718


namespace triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l47_4731

def triangle (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b

def right_triangle (a b c : ℝ) : Prop := 
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem triangle_condition_A (a b c : ℝ) (h : triangle a b c) : 
  b^2 = (a + c) * (c - a) → right_triangle a c b := 
sorry

theorem triangle_condition_B (A B C : ℝ) (h : A + B + C = 180) : 
  A = B + C → 90 = A :=
sorry

theorem triangle_condition_C (A B C : ℝ) (h : A + B + C = 180) : 
  3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → 
  ¬ (right_triangle A B C) :=
sorry

theorem triangle_condition_D : 
  right_triangle 6 8 10 := 
sorry

theorem problem_solution (a b c : ℝ) (A B C : ℝ) (hABC : triangle a b c) : 
  (b^2 = (a + c) * (c - a) → right_triangle a c b) ∧
  ((A + B + C = 180) ∧ (A = B + C) → 90 = A) ∧
  (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C → ¬ right_triangle a b c) ∧
  (right_triangle 6 8 10) → 
  ∃ (cond : Prop), cond = (3 * (A / 12) = A ∧ 4 * (A / 12) = B ∧ 5 * (A / 12) = C) := 
sorry

end triangle_condition_A_triangle_condition_B_triangle_condition_C_triangle_condition_D_problem_solution_l47_4731


namespace fraction_taken_out_is_one_sixth_l47_4797

-- Define the conditions
def original_cards : ℕ := 43
def cards_added_by_Sasha : ℕ := 48
def cards_left_after_Karen_took_out : ℕ := 83

-- Calculate the total number of cards initially after Sasha added hers
def total_cards_after_Sasha : ℕ := original_cards + cards_added_by_Sasha

-- Calculate the number of cards Karen took out
def cards_taken_out_by_Karen : ℕ := total_cards_after_Sasha - cards_left_after_Karen_took_out

-- Define the fraction of the cards Sasha added that Karen took out
def fraction_taken_out : ℚ := cards_taken_out_by_Karen / cards_added_by_Sasha

-- Proof statement: Fraction of the cards Sasha added that Karen took out is 1/6
theorem fraction_taken_out_is_one_sixth : fraction_taken_out = 1 / 6 :=
by
    -- Sorry is a placeholder for the proof, which is not required.
    sorry

end fraction_taken_out_is_one_sixth_l47_4797


namespace prime_gt_10_exists_m_n_l47_4734

theorem prime_gt_10_exists_m_n (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_10 : p > 10) :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ m + n < p ∧ p ∣ (5^m * 7^n - 1) :=
by
  sorry

end prime_gt_10_exists_m_n_l47_4734


namespace find_p_l47_4729

theorem find_p (p q : ℚ) (h1 : 3 * p + 4 * q = 15) (h2 : 4 * p + 3 * q = 18) : p = 27 / 7 :=
by
  sorry

end find_p_l47_4729


namespace probability_of_two_co_presidents_l47_4747

noncomputable section

def binomial (n k : ℕ) : ℕ :=
  if h : n ≥ k then Nat.choose n k else 0

def club_prob (n : ℕ) : ℚ :=
  (binomial (n-2) 2 : ℚ) / (binomial n 4 : ℚ)

def total_probability : ℚ :=
  (1/4 : ℚ) * (club_prob 6 + club_prob 8 + club_prob 9 + club_prob 10)

theorem probability_of_two_co_presidents : total_probability = 0.2286 := by
  -- We expect this to be true based on the given solution
  sorry

end probability_of_two_co_presidents_l47_4747


namespace non_zero_real_positive_integer_l47_4786

theorem non_zero_real_positive_integer (x : ℝ) (h : x ≠ 0) : 
  (∃ k : ℤ, k > 0 ∧ (x - |x-1|) / x = k) ↔ x = 1 := 
sorry

end non_zero_real_positive_integer_l47_4786


namespace find_y_intercept_l47_4796

theorem find_y_intercept (x1 y1 x2 y2 : ℝ) (h₁ : (x1, y1) = (2, -2)) (h₂ : (x2, y2) = (6, 6)) : 
  ∃ b : ℝ, (∀ x : ℝ, y = 2 * x + b) ∧ b = -6 :=
by
  sorry

end find_y_intercept_l47_4796


namespace trig_expr_eval_sin_minus_cos_l47_4784

-- Problem 1: Evaluation of trigonometric expression
theorem trig_expr_eval : 
    (Real.sin (-π / 2) + 3 * Real.cos 0 - 2 * Real.tan (3 * π / 4) - 4 * Real.cos (5 * π / 3)) = 2 :=
by 
    sorry

-- Problem 2: Given tangent value and angle constraints, find sine minus cosine
theorem sin_minus_cos {θ : ℝ} 
    (h1 : Real.tan θ = 4 / 3)
    (h2 : 0 < θ)
    (h3 : θ < π / 2) : 
    (Real.sin θ - Real.cos θ) = 1 / 5 :=
by 
    sorry

end trig_expr_eval_sin_minus_cos_l47_4784


namespace problem_D_l47_4768

-- Define the lines m and n, and planes α and β
variables (m n : Type) (α β : Type)

-- Define the parallel and perpendicular relations
variables (parallel : Type → Type → Prop) (perpendicular : Type → Type → Prop)

-- Assume the conditions of problem D
variables (h1 : perpendicular m α) (h2 : parallel n β) (h3 : parallel α β)

-- The proof problem statement: Prove that under these assumptions, m is perpendicular to n
theorem problem_D : perpendicular m n :=
sorry

end problem_D_l47_4768


namespace geometric_sequence_t_value_l47_4795

theorem geometric_sequence_t_value (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t * 5^n - 2) → 
  (∀ n ≥ 1, a (n + 1) = S (n + 1) - S n) → 
  (a 1 ≠ 0) → -- Ensure the sequence is non-trivial.
  (∀ n, a (n + 1) / a n = 5) → 
  t = 5 := 
by 
  intros h1 h2 h3 h4
  sorry

end geometric_sequence_t_value_l47_4795


namespace sequence_term_l47_4735

theorem sequence_term (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, (n + 1) * a n = 2 * n * a (n + 1)) : 
  ∀ n : ℕ, a n = n / 2^(n - 1) :=
by
  sorry

end sequence_term_l47_4735


namespace positive_integers_are_N_star_l47_4787

def Q := { x : ℚ | true } -- The set of rational numbers
def N := { x : ℕ | true } -- The set of natural numbers
def N_star := { x : ℕ | x > 0 } -- The set of positive integers
def Z := { x : ℤ | true } -- The set of integers

theorem positive_integers_are_N_star : 
  ∀ x : ℕ, (x ∈ N_star) ↔ (x > 0) := 
sorry

end positive_integers_are_N_star_l47_4787


namespace ratio_problem_l47_4744

theorem ratio_problem
  (a b c d : ℝ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 49) :
  d / a = 1 / 122.5 :=
by {
  -- Proof steps would go here
  sorry
}

end ratio_problem_l47_4744


namespace largest_n_l47_4757

theorem largest_n (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  ∃ n : ℕ, n > 0 ∧ n = 10 ∧ n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 12 := 
sorry

end largest_n_l47_4757


namespace find_coordinates_of_C_l47_4707

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := { x := 4, y := -1, z := 2 }
def B : Point := { x := 2, y := -3, z := 0 }

def satisfies_condition (C : Point) : Prop :=
  (C.x - B.x, C.y - B.y, C.z - B.z) = (2 * (A.x - C.x), 2 * (A.y - C.y), 2 * (A.z - C.z))

theorem find_coordinates_of_C (C : Point) (h : satisfies_condition C) : C = { x := 10/3, y := -5/3, z := 4/3 } :=
  sorry -- Proof is omitted as requested

end find_coordinates_of_C_l47_4707


namespace find_the_number_l47_4782

theorem find_the_number :
  ∃ X : ℝ, (66.2 = (6.620000000000001 / 100) * X) ∧ X = 1000 :=
by
  sorry

end find_the_number_l47_4782


namespace shirley_eggs_start_l47_4733

theorem shirley_eggs_start (eggs_end : ℕ) (eggs_bought : ℕ) (eggs_start : ℕ) (h_end : eggs_end = 106) (h_bought : eggs_bought = 8) :
  eggs_start = eggs_end - eggs_bought → eggs_start = 98 :=
by
  intros h_start
  rw [h_end, h_bought] at h_start
  exact h_start

end shirley_eggs_start_l47_4733


namespace sin_x_solution_l47_4755

theorem sin_x_solution (A B C x : ℝ) (h : A * Real.cos x + B * Real.sin x = C) :
  ∃ (u v : ℝ),  -- We assert the existence of u and v such that 
    Real.sin x = (A * C + B * u) / (A^2 + B^2) ∨ 
    Real.sin x = (A * C - B * v) / (A^2 + B^2) :=
sorry

end sin_x_solution_l47_4755


namespace toy_store_restock_l47_4760

theorem toy_store_restock 
  (initial_games : ℕ) (games_sold : ℕ) (after_restock_games : ℕ) 
  (initial_games_condition : initial_games = 95)
  (games_sold_condition : games_sold = 68)
  (after_restock_games_condition : after_restock_games = 74) :
  after_restock_games - (initial_games - games_sold) = 47 :=
by {
  sorry
}

end toy_store_restock_l47_4760


namespace smallest_two_digit_integer_l47_4740

theorem smallest_two_digit_integer (n a b : ℕ) (h1 : n = 10 * a + b) (h2 : 2 * n = 10 * b + a + 5) (h3 : 1 ≤ a) (h4 : a ≤ 9) (h5 : 0 ≤ b) (h6 : b ≤ 9) : n = 69 := 
by 
  sorry

end smallest_two_digit_integer_l47_4740


namespace simplify_expression_l47_4794

-- Define the problem and its conditions
theorem simplify_expression :
  (81 * 10^12) / (9 * 10^4) = 900000000 :=
by
  sorry  -- Proof placeholder

end simplify_expression_l47_4794


namespace twenty_four_times_ninety_nine_l47_4738

theorem twenty_four_times_ninety_nine : 24 * 99 = 2376 :=
by sorry

end twenty_four_times_ninety_nine_l47_4738


namespace find_possible_values_for_P_l47_4743

theorem find_possible_values_for_P (x y P : ℕ) (h1 : x < y) :
  P = (x^3 - y) / (1 + x * y) → (P = 0 ∨ P ≥ 2) :=
by
  sorry

end find_possible_values_for_P_l47_4743


namespace gcd_of_polynomial_l47_4723

theorem gcd_of_polynomial (x : ℕ) (hx : 32515 ∣ x) :
    Nat.gcd ((3 * x + 5) * (5 * x + 3) * (11 * x + 7) * (x + 17)) x = 35 :=
sorry

end gcd_of_polynomial_l47_4723


namespace apollonius_circle_equation_l47_4769

theorem apollonius_circle_equation (x y : ℝ) (A B : ℝ × ℝ) (hA : A = (2, 0)) (hB : B = (8, 0))
  (h : dist (x, y) A / dist (x, y) B = 1 / 2) : x^2 + y^2 = 16 := 
sorry

end apollonius_circle_equation_l47_4769


namespace approximate_number_of_fish_in_pond_l47_4739

theorem approximate_number_of_fish_in_pond :
  ∃ N : ℕ, N = 800 ∧
  (40 : ℕ) / N = (2 : ℕ) / (40 : ℕ) := 
sorry

end approximate_number_of_fish_in_pond_l47_4739


namespace find_m_minus_n_l47_4717

theorem find_m_minus_n (m n : ℝ) (h1 : -5 + 1 = m) (h2 : -5 * 1 = n) : m - n = 1 :=
sorry

end find_m_minus_n_l47_4717


namespace infinite_series_sum_l47_4709

theorem infinite_series_sum : (∑' n : ℕ, if n % 3 = 0 then 1 / (3 * 2^(((n - n % 3) / 3) + 1)) 
                                 else if n % 3 = 1 then -1 / (6 * 2^(((n - n % 3) / 3)))
                                 else -1 / (12 * 2^(((n - n % 3) / 3)))) = 1 / 72 :=
by
  sorry

end infinite_series_sum_l47_4709


namespace tom_splitting_slices_l47_4767

theorem tom_splitting_slices :
  ∃ S : ℕ, (∃ t, t = 3/8 * S) → 
          (∃ u, u = 1/2 * (S - t)) → 
          (∃ v, v = u + t) → 
          (v = 5) → 
          (S / 2 = 8) :=
sorry

end tom_splitting_slices_l47_4767


namespace candidate1_fails_by_l47_4704

-- Define the total marks (T), passing marks (P), percentage marks (perc1 and perc2), and the extra marks.
def T : ℝ := 600
def P : ℝ := 160
def perc1 : ℝ := 0.20
def perc2 : ℝ := 0.30
def extra_marks : ℝ := 20

-- Define the marks obtained by the candidates.
def marks_candidate1 : ℝ := perc1 * T
def marks_candidate2 : ℝ := perc2 * T

-- The theorem stating the number of marks by which the first candidate fails.
theorem candidate1_fails_by (h_pass: perc2 * T = P + extra_marks) : P - marks_candidate1 = 40 :=
by
  -- The proof would go here.
  sorry

end candidate1_fails_by_l47_4704


namespace digits_in_number_l47_4721

def four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def contains_digits (n : ℕ) (d1 d2 d3 : ℕ) : Prop :=
  (n / 1000 = d1 ∨ n / 100 % 10 = d1 ∨ n / 10 % 10 = d1 ∨ n % 10 = d1) ∧
  (n / 1000 = d2 ∨ n / 100 % 10 = d2 ∨ n / 10 % 10 = d2 ∨ n % 10 = d2) ∧
  (n / 1000 = d3 ∨ n / 100 % 10 = d3 ∨ n / 10 % 10 = d3 ∨ n % 10 = d3)

def exactly_two_statements_true (s1 s2 s3 : Prop) : Prop :=
  (s1 ∧ s2 ∧ ¬s3) ∨ (s1 ∧ ¬s2 ∧ s3) ∨ (¬s1 ∧ s2 ∧ s3)

theorem digits_in_number (n : ℕ) 
  (h1 : four_digit_number n)
  (h2 : contains_digits n 1 4 5 ∨ contains_digits n 1 5 9 ∨ contains_digits n 7 8 9)
  (h3 : exactly_two_statements_true (contains_digits n 1 4 5) (contains_digits n 1 5 9) (contains_digits n 7 8 9)) :
  contains_digits n 1 4 5 ∧ contains_digits n 1 5 9 :=
sorry

end digits_in_number_l47_4721


namespace range_of_m_three_zeros_l47_4728

noncomputable def f (x m : ℝ) : ℝ :=
if h : x < 0 then -x + m else x^2 - 1

theorem range_of_m_three_zeros (h : 0 < m) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (f x1 m) m - 1 = 0 ∧ f (f x2 m) m - 1 = 0 ∧ f (f x3 m) m - 1 = 0) ↔ (0 < m ∧ m < 1) :=
by
  sorry

end range_of_m_three_zeros_l47_4728


namespace fraction_of_3_4_is_4_27_l47_4792

theorem fraction_of_3_4_is_4_27 (a b : ℚ) (h1 : a = 3/4) (h2 : b = 1/9) :
  b / a = 4 / 27 :=
by
  sorry

end fraction_of_3_4_is_4_27_l47_4792


namespace center_of_circle_tangent_to_parallel_lines_l47_4750

-- Define the line equations
def line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 40
def line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -20
def line3 (x y : ℝ) : Prop := x - 2 * y = 0

-- The proof problem
theorem center_of_circle_tangent_to_parallel_lines
  (x y : ℝ)
  (h1 : line1 x y → false)
  (h2 : line2 x y → false)
  (h3 : line3 x y) :
  x = 10 ∧ y = 5 := by
  sorry

end center_of_circle_tangent_to_parallel_lines_l47_4750


namespace CarlyWorkedOnElevenDogs_l47_4774

-- Given conditions
def CarlyTrimmedNails : ℕ := 164
def DogsWithThreeLegs : ℕ := 3
def NailsPerPaw : ℕ := 4
def PawsPerThreeLeggedDog : ℕ := 3
def PawsPerFourLeggedDog : ℕ := 4

-- Deduction steps
def TotalPawsWorkedOn := CarlyTrimmedNails / NailsPerPaw
def PawsOnThreeLeggedDogs := DogsWithThreeLegs * PawsPerThreeLeggedDog
def PawsOnFourLeggedDogs := TotalPawsWorkedOn - PawsOnThreeLeggedDogs
def CountFourLeggedDogs := PawsOnFourLeggedDogs / PawsPerFourLeggedDog

-- Total dogs Carly worked on
def TotalDogsCarlyWorkedOn := CountFourLeggedDogs + DogsWithThreeLegs

-- The statement we need to prove
theorem CarlyWorkedOnElevenDogs : TotalDogsCarlyWorkedOn = 11 := by
  sorry

end CarlyWorkedOnElevenDogs_l47_4774


namespace work_completion_time_l47_4726

theorem work_completion_time (a b c : ℕ) (ha : a = 36) (hb : b = 18) (hc : c = 6) : (1 / (1 / a + 1 / b + 1 / c) = 4) := by
  sorry

end work_completion_time_l47_4726


namespace andy_time_difference_l47_4748

def time_dawn : ℕ := 20
def time_andy : ℕ := 46
def double_time_dawn : ℕ := 2 * time_dawn

theorem andy_time_difference :
  time_andy - double_time_dawn = 6 := by
  sorry

end andy_time_difference_l47_4748


namespace number_of_students_preferring_dogs_l47_4705

-- Define the conditions
def total_students : ℕ := 30
def dogs_video_games_chocolate_percentage : ℚ := 0.50
def dogs_movies_vanilla_percentage : ℚ := 0.10
def cats_video_games_chocolate_percentage : ℚ := 0.20
def cats_movies_vanilla_percentage : ℚ := 0.15

-- Define the target statement to prove
theorem number_of_students_preferring_dogs : 
  (dogs_video_games_chocolate_percentage + dogs_movies_vanilla_percentage) * total_students = 18 :=
by
  sorry

end number_of_students_preferring_dogs_l47_4705


namespace find_certain_number_l47_4764

theorem find_certain_number :
  ∃ C, ∃ A B, (A + B = 15) ∧ (A = 7) ∧ (C * B = 5 * A - 11) ∧ (C = 3) :=
by
  sorry

end find_certain_number_l47_4764


namespace pentagon_area_l47_4789

theorem pentagon_area 
  (PQ QR RS ST TP : ℝ) 
  (angle_TPQ angle_PQR : ℝ) 
  (hPQ : PQ = 8) 
  (hQR : QR = 2) 
  (hRS : RS = 13) 
  (hST : ST = 13) 
  (hTP : TP = 8) 
  (hangle_TPQ : angle_TPQ = 90) 
  (hangle_PQR : angle_PQR = 90) : 
  PQ * QR + (1 / 2) * (TP - QR) * PQ + (1 / 2) * 10 * 12 = 100 := 
by
  sorry

end pentagon_area_l47_4789


namespace cole_round_trip_time_l47_4781

/-- Prove that the total round trip time is 2 hours given the conditions -/
theorem cole_round_trip_time :
  ∀ (speed_to_work : ℝ) (speed_back_home : ℝ) (time_to_work_min : ℝ),
  speed_to_work = 50 → speed_back_home = 110 → time_to_work_min = 82.5 →
  ((time_to_work_min / 60) * speed_to_work + (time_to_work_min * speed_to_work / speed_back_home) / 60) = 2 :=
by
  intros
  sorry

end cole_round_trip_time_l47_4781


namespace remainder_of_m_l47_4798

theorem remainder_of_m (m : ℕ) (h₁ : m ^ 3 % 7 = 6) (h₂ : m ^ 4 % 7 = 4) : m % 7 = 3 := 
sorry

end remainder_of_m_l47_4798


namespace sum_polynomial_coefficients_l47_4712

theorem sum_polynomial_coefficients :
  let a := 1
  let a_sum := -2
  (2009 * a + a_sum) = 2007 :=
by
  sorry

end sum_polynomial_coefficients_l47_4712


namespace employee_y_payment_l47_4719

theorem employee_y_payment (X Y : ℝ) (h1 : X + Y = 590) (h2 : X = 1.2 * Y) : Y = 268.18 := by
  sorry

end employee_y_payment_l47_4719


namespace probability_exactly_one_win_l47_4713

theorem probability_exactly_one_win :
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  P_exactly_one_win = 8 / 15 :=
by
  let P_win_Jp := 2 / 3
  let P_win_Us := 2 / 5
  let P_exactly_one_win := P_win_Jp * (1 - P_win_Us) + (1 - P_win_Jp) * P_win_Us
  have h1 : P_exactly_one_win = 8 / 15 := sorry
  exact h1

end probability_exactly_one_win_l47_4713


namespace one_in_M_l47_4714

def N := { x : ℕ | true } -- Define the natural numbers ℕ

def M : Set ℕ := { x ∈ N | 1 / (x - 2) ≤ 0 }

theorem one_in_M : 1 ∈ M :=
  sorry

end one_in_M_l47_4714


namespace cone_volume_l47_4770

theorem cone_volume (r l: ℝ) (h: ℝ) (hr : r = 1) (hl : l = 2) (hh : h = Real.sqrt (l^2 - r^2)) : 
  (1 / 3) * Real.pi * r^2 * h = (Real.sqrt 3 * Real.pi) / 3 :=
by 
  sorry

end cone_volume_l47_4770


namespace problem_a_problem_b_l47_4754

-- Define the conditions for problem (a):
variable (x y z : ℝ)
variable (h_xyz : x * y * z = 1)

theorem problem_a (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) :
  (x^2 / (x - 1)^2) + (y^2 / (y - 1)^2) + (z^2 / (z - 1)^2) ≥ 1 :=
sorry

-- Define the conditions for problem (b):
variable (a b c : ℚ)

theorem problem_b (h_abc : a * b * c = 1) :
  ∃ (x y z : ℚ), x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧ (x * y * z = 1) ∧ 
  (x^2 / (x - 1)^2 + y^2 / (y - 1)^2 + z^2 / (z - 1)^2 = 1) :=
sorry

end problem_a_problem_b_l47_4754


namespace cows_count_l47_4779

theorem cows_count (D C : ℕ) (h1 : 2 * (D + C) + 32 = 2 * D + 4 * C) : C = 16 :=
by
  sorry

end cows_count_l47_4779


namespace jane_doe_total_investment_mutual_funds_l47_4701

theorem jane_doe_total_investment_mutual_funds :
  ∀ (c m : ℝ) (total_investment : ℝ),
  total_investment = 250000 → m = 3 * c → c + m = total_investment → m = 187500 :=
by
  intros c m total_investment h_total h_relation h_sum
  sorry

end jane_doe_total_investment_mutual_funds_l47_4701


namespace each_album_contains_correct_pictures_l47_4730

def pictures_in_each_album (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat) :=
  (pictures_per_album_phone + pictures_per_album_camera)

theorem each_album_contains_correct_pictures (pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera : Nat)
  (h1 : pictures_phone = 80)
  (h2 : pictures_camera = 40)
  (h3 : albums = 10)
  (h4 : pictures_per_album_phone = 8)
  (h5 : pictures_per_album_camera = 4)
  : pictures_in_each_album pictures_phone pictures_camera albums pictures_per_album_phone pictures_per_album_camera = 12 := by
  sorry

end each_album_contains_correct_pictures_l47_4730


namespace total_votes_cast_l47_4706

theorem total_votes_cast (S : ℝ) (x : ℝ) (h1 : S = 120) (h2 : S = 0.72 * x - 0.28 * x) : x = 273 := by
  sorry

end total_votes_cast_l47_4706


namespace sum_of_digits_of_m_l47_4785

-- Define the logarithms and intermediate expressions
noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem sum_of_digits_of_m :
  ∃ m : ℕ, log_b 3 (log_b 81 m) = log_b 9 (log_b 9 m) ∧ sum_of_digits m = 10 := 
by
  sorry

end sum_of_digits_of_m_l47_4785


namespace twentieth_term_arithmetic_sequence_eq_neg49_l47_4736

-- Definitions based on the conditions
def a1 : ℤ := 8
def d : ℤ := 5 - 8
def a (n : ℕ) : ℤ := a1 + (n - 1) * d

-- The proof statement
theorem twentieth_term_arithmetic_sequence_eq_neg49 : a 20 = -49 :=
by 
  -- Proof will be inserted here
  sorry

end twentieth_term_arithmetic_sequence_eq_neg49_l47_4736


namespace maximize_area_playground_l47_4777

noncomputable def maxAreaPlayground : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area_playground : ∀ (l w : ℝ),
  (2 * l + 2 * w = 400) ∧ (l ≥ 100) ∧ (w ≥ 60) → l * w ≤ maxAreaPlayground :=
by
  intros l w h
  sorry

end maximize_area_playground_l47_4777


namespace original_faculty_number_l47_4778

theorem original_faculty_number (x : ℝ) (h : 0.85 * x = 195) : x = 229 := by
  sorry

end original_faculty_number_l47_4778


namespace increase_in_circumference_by_2_cm_l47_4772

noncomputable def radius_increase_by_two (r : ℝ) : ℝ := r + 2
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem increase_in_circumference_by_2_cm (r : ℝ) : 
    circumference (radius_increase_by_two r) - circumference r = 12.56 :=
by sorry

end increase_in_circumference_by_2_cm_l47_4772


namespace pieces_of_gum_l47_4776

variable (initial_gum total_gum given_gum : ℕ)

theorem pieces_of_gum (h1 : given_gum = 16) (h2 : total_gum = 54) : initial_gum = 38 :=
by
  sorry

end pieces_of_gum_l47_4776


namespace proj_v_w_l47_4741

noncomputable def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2
  let w_dot_w := dot_product w w
  let v_dot_w := dot_product v w
  let scalar := v_dot_w / w_dot_w
  (scalar * w.1, scalar * w.2)

theorem proj_v_w :
  let v := (4, -3)
  let w := (12, 5)
  proj v w = (396 / 169, 165 / 169) :=
by
  sorry

end proj_v_w_l47_4741


namespace sum_of_numbers_l47_4703

def contains_digit (n : Nat) (d : Nat) : Prop := 
  (n / 100 = d) ∨ (n % 100 / 10 = d) ∨ (n % 10 = d)

variables {A B C : Nat}

-- Given conditions
axiom three_digit_number : A ≥ 100 ∧ A < 1000
axiom two_digit_numbers : B ≥ 10 ∧ B < 100 ∧ C ≥ 10 ∧ C < 100
axiom sum_with_sevens : contains_digit A 7 ∨ contains_digit B 7 ∨ contains_digit C 7 → A + B + C = 208
axiom sum_with_threes : contains_digit B 3 ∧ contains_digit C 3 ∧ B + C = 76

-- Main theorem to be proved
theorem sum_of_numbers : A + B + C = 247 :=
sorry

end sum_of_numbers_l47_4703


namespace geometric_a1_value_l47_4756

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ (n - 1)

theorem geometric_a1_value (a3 a5 : ℝ) (q : ℝ) : 
  a3 = geometric_sequence a1 q 3 →
  a5 = geometric_sequence a1 q 5 →
  a1 = 2 :=
by
  sorry

end geometric_a1_value_l47_4756


namespace polygon_interior_angles_sum_l47_4753

theorem polygon_interior_angles_sum {n : ℕ} 
  (h1 : ∀ (k : ℕ), k > 2 → (360 = k * 40)) :
  180 * (9 - 2) = 1260 :=
by
  sorry

end polygon_interior_angles_sum_l47_4753


namespace total_loss_l47_4724

theorem total_loss (P : ℝ) (A : ℝ) (L : ℝ) (h1 : A = (1/9) * P) (h2 : 603 = (P / (A + P)) * L) : 
  L = 670 :=
by
  sorry

end total_loss_l47_4724


namespace problem_statement_l47_4700

open Function

theorem problem_statement :
  ∃ g : ℝ → ℝ, 
    (g 1 = 2) ∧ 
    (∀ (x y : ℝ), g (x^2 - y^2) = (x - y) * (g x + g y)) ∧ 
    (g 3 = 6) := 
by
  sorry

end problem_statement_l47_4700


namespace symmetric_axis_of_quadratic_l47_4766

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := (x - 3) * (x + 5)

-- Prove that the symmetric axis of the quadratic function is the line x = -1
theorem symmetric_axis_of_quadratic : ∀ (x : ℝ), quadratic_function x = (x - 3) * (x + 5) → x = -1 :=
by
  intro x h
  sorry

end symmetric_axis_of_quadratic_l47_4766


namespace calculate_radius_l47_4751

noncomputable def radius_of_wheel (D : ℝ) (N : ℕ) (π : ℝ) : ℝ :=
  D / (2 * π * N)

theorem calculate_radius : 
  radius_of_wheel 4224 3000 Real.pi = 0.224 :=
by
  sorry

end calculate_radius_l47_4751


namespace min_handshakes_l47_4742

theorem min_handshakes (n : ℕ) (h1 : n = 25) 
  (h2 : ∀ (p : ℕ), p < n → ∃ q r : ℕ, q ≠ r ∧ q < n ∧ r < n ∧ q ≠ p ∧ r ≠ p) 
  (h3 : ∃ a b c : ℕ, a < n ∧ b < n ∧ c < n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬(∃ d : ℕ, (d = a ∨ d = b ∨ d = c) ∧ (¬(a = d ∨ b = d ∨ c = d)) ∧ d < n)) :
  ∃ m : ℕ, m = 28 :=
by
  sorry

end min_handshakes_l47_4742


namespace part1_inequality_part2_min_value_l47_4746

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  4^x + m * 2^x

theorem part1_inequality (x : ℝ) : f x (-3) > 4 → x > 2 :=
  sorry

theorem part2_min_value (h : (∀ x : ℝ, f x m + f (-x) m ≥ -4)) : m = -3 :=
  sorry

end part1_inequality_part2_min_value_l47_4746


namespace value_of_m_l47_4788

theorem value_of_m (m : ℝ) : (∀ x : ℝ, (x^2 + 2 * m * x + m > 3 / 16)) ↔ (1 / 4 < m ∧ m < 3 / 4) :=
by sorry

end value_of_m_l47_4788


namespace proof_complement_union_l47_4725

-- Definition of the universal set U
def U : Finset ℕ := {0, 1, 2, 3, 4}

-- Definition of the subset A
def A : Finset ℕ := {0, 3, 4}

-- Definition of the subset B
def B : Finset ℕ := {1, 3}

-- Definition of the complement of A in U
def complement_A : Finset ℕ := U \ A

-- Definition of the union of the complement of A and B
def union_complement_A_B : Finset ℕ := complement_A ∪ B

-- Statement of the theorem
theorem proof_complement_union :
  union_complement_A_B = {1, 2, 3} :=
sorry

end proof_complement_union_l47_4725


namespace meeting_time_l47_4790

theorem meeting_time (x : ℝ) :
  (1/6) * x + (1/4) * (x - 1) = 1 :=
sorry

end meeting_time_l47_4790


namespace minimum_roots_in_interval_l47_4791

noncomputable def g : ℝ → ℝ := sorry

lemma symmetry_condition_1 (x : ℝ) : g (3 + x) = g (3 - x) := sorry
lemma symmetry_condition_2 (x : ℝ) : g (8 + x) = g (8 - x) := sorry
lemma initial_condition : g 1 = 0 := sorry

theorem minimum_roots_in_interval : 
  ∃ k, ∀ x, -1000 ≤ x ∧ x ≤ 1000 → g x = 0 ∧ 
  (2 * k) = 286 := sorry

end minimum_roots_in_interval_l47_4791


namespace wheel_radius_increase_proof_l47_4720

noncomputable def radius_increase (orig_distance odometer_distance : ℝ) (orig_radius : ℝ) : ℝ :=
  let orig_circumference := 2 * Real.pi * orig_radius
  let distance_per_rotation := orig_circumference / 63360
  let num_rotations_orig := orig_distance / distance_per_rotation
  let num_rotations_new := odometer_distance / distance_per_rotation
  let new_distance := orig_distance
  let new_radius := (new_distance / num_rotations_new) * 63360 / (2 * Real.pi)
  new_radius - orig_radius

theorem wheel_radius_increase_proof :
  radius_increase 600 580 16 = 0.42 :=
by 
  -- The proof is skipped.
  sorry

end wheel_radius_increase_proof_l47_4720


namespace minimum_throws_for_repetition_of_sum_l47_4761

/-- To ensure that the same sum is rolled twice when throwing four fair six-sided dice,
you must throw the dice at least 22 times. -/
theorem minimum_throws_for_repetition_of_sum :
  ∀ (throws : ℕ), (∀ (sum : ℕ), 4 ≤ sum ∧ sum ≤ 24 → ∃ (count : ℕ), count ≤ 21 ∧ sum = count + 4) → throws ≥ 22 :=
by
  sorry

end minimum_throws_for_repetition_of_sum_l47_4761


namespace trace_bag_weight_is_two_l47_4762

-- Given the conditions in the problem
def weight_gordon_bag₁ : ℕ := 3
def weight_gordon_bag₂ : ℕ := 7
def num_traces_bag : ℕ := 5

-- Total weight of Gordon's bags is 10
def total_weight_gordon := weight_gordon_bag₁ + weight_gordon_bag₂

-- Trace's bags weight
def total_weight_trace := total_weight_gordon

-- All conditions must imply this equation is true
theorem trace_bag_weight_is_two :
  (num_traces_bag * 2 = total_weight_trace) → (2 = 2) :=
  by
    sorry

end trace_bag_weight_is_two_l47_4762


namespace olivia_total_pieces_l47_4758

def initial_pieces_folder1 : ℕ := 152
def initial_pieces_folder2 : ℕ := 98
def used_pieces_folder1 : ℕ := 78
def used_pieces_folder2 : ℕ := 42

def remaining_pieces_folder1 : ℕ :=
  initial_pieces_folder1 - used_pieces_folder1

def remaining_pieces_folder2 : ℕ :=
  initial_pieces_folder2 - used_pieces_folder2

def total_remaining_pieces : ℕ :=
  remaining_pieces_folder1 + remaining_pieces_folder2

theorem olivia_total_pieces : total_remaining_pieces = 130 :=
  by sorry

end olivia_total_pieces_l47_4758


namespace sequence_has_both_max_and_min_l47_4732

noncomputable def a_n (n : ℕ) : ℝ :=
  (n + 1) * ((-10 / 11) ^ n)

theorem sequence_has_both_max_and_min :
  ∃ (max min : ℝ) (N M : ℕ), 
    (∀ n : ℕ, a_n n ≤ max) ∧ (∀ n : ℕ, min ≤ a_n n) ∧ 
    (a_n N = max) ∧ (a_n M = min) := 
sorry

end sequence_has_both_max_and_min_l47_4732


namespace max_value_l47_4702

open Real

theorem max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 75) : 
  xy * (75 - 2 * x - 5 * y) ≤ 1562.5 := 
sorry

end max_value_l47_4702


namespace acute_triangle_probability_l47_4715

open Finset

noncomputable def isAcuteTriangleProb (n : ℕ) : Prop :=
  ∃ k : ℕ, (n = 2 * k ∧ (3 * (k - 2)) / (2 * (2 * k - 1)) = 93 / 125) ∨ (n = 2 * k + 1 ∧ (3 * (k - 1)) / (2 * (2 * k - 1)) = 93 / 125)

theorem acute_triangle_probability (n : ℕ) : isAcuteTriangleProb n → n = 376 ∨ n = 127 :=
by
  sorry

end acute_triangle_probability_l47_4715


namespace coffee_ratio_l47_4793

/-- Define the conditions -/
def initial_coffees_per_day := 4
def initial_price_per_coffee := 2
def price_increase_percentage := 50 / 100
def savings_per_day := 2

/-- Define the price calculations -/
def new_price_per_coffee := initial_price_per_coffee + (initial_price_per_coffee * price_increase_percentage)
def initial_daily_cost := initial_coffees_per_day * initial_price_per_coffee
def new_daily_cost := initial_daily_cost - savings_per_day
def new_coffees_per_day := new_daily_cost / new_price_per_coffee

/-- Prove the ratio -/
theorem coffee_ratio : (new_coffees_per_day / initial_coffees_per_day) = (1 : ℝ) / (2 : ℝ) :=
  by sorry

end coffee_ratio_l47_4793


namespace gcd_78_36_l47_4722

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := 
by
  sorry

end gcd_78_36_l47_4722


namespace joan_initial_balloons_l47_4711

-- Definitions using conditions from a)
def initial_balloons (lost : ℕ) (current : ℕ) : ℕ := lost + current

-- Statement of our equivalent math proof problem
theorem joan_initial_balloons : initial_balloons 2 7 = 9 := 
by
  -- Proof skipped using sorry
  sorry

end joan_initial_balloons_l47_4711


namespace deanna_wins_l47_4759

theorem deanna_wins (A B C D : ℕ) (total_games : ℕ) (total_wins : ℕ) (A_wins : A = 5) (B_wins : B = 2)
  (C_wins : C = 1) (total_games_def : total_games = 6) (total_wins_def : total_wins = 12)
  (total_wins_eq : A + B + C + D = total_wins) : D = 4 :=
by
  sorry

end deanna_wins_l47_4759


namespace smallest_d0_l47_4727

theorem smallest_d0 (r : ℕ) (hr : r ≥ 3) : ∃ d₀, d₀ = 2^(r - 2) ∧ (7^d₀ ≡ 1 [MOD 2^r]) :=
by
  sorry

end smallest_d0_l47_4727


namespace mass_percentage_Al_in_Al2CO33_l47_4783
-- Importing the required libraries

-- Define the necessary constants for molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def molar_mass_Al2CO33 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_C + 9 * molar_mass_O
def mass_Al_in_Al2CO33 : ℝ := 2 * molar_mass_Al

-- Define the main theorem to prove the mass percentage of Al in Al2(CO3)3
theorem mass_percentage_Al_in_Al2CO33 :
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  simp [molar_mass_Al, molar_mass_C, molar_mass_O, molar_mass_Al2CO33, mass_Al_in_Al2CO33]
  -- Calculation result based on given molar masses
  sorry

end mass_percentage_Al_in_Al2CO33_l47_4783


namespace sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l47_4799

-- Defining the terms and the theorem
theorem sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one :
  (Real.sqrt 5 - 2) ^ 2023 * (Real.sqrt 5 + 2) ^ 2023 = 1 := 
by
  sorry

end sqrt5_minus_2_power_2023_mul_sqrt5_plus_2_power_2023_eq_one_l47_4799


namespace odd_periodic_function_l47_4775

variable {f : ℝ → ℝ}

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic_function (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

-- Problem statement
theorem odd_periodic_function (h_odd : odd_function f)
  (h_period : periodic_function f) (h_half : f 0.5 = 1) : f 7.5 = -1 :=
sorry

end odd_periodic_function_l47_4775


namespace contractor_laborers_l47_4737

theorem contractor_laborers (x : ℕ) (h : 9 * x = 15 * (x - 6)) : x = 15 :=
by
  sorry

end contractor_laborers_l47_4737


namespace systematic_sampling_fourth_group_l47_4708

theorem systematic_sampling_fourth_group (n m k g2 g4 : ℕ) (h_class_size : n = 72)
  (h_sample_size : m = 6) (h_k : k = n / m) (h_group2 : g2 = 16) (h_group4 : g4 = g2 + 2 * k) :
  g4 = 40 := by
  sorry

end systematic_sampling_fourth_group_l47_4708


namespace west_1000_move_l47_4752

def eastMovement (d : Int) := d  -- east movement positive
def westMovement (d : Int) := -d -- west movement negative

theorem west_1000_move : westMovement 1000 = -1000 :=
  by
    sorry

end west_1000_move_l47_4752


namespace cameron_list_count_l47_4771

-- Definitions
def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = k * b
def is_perfect_square (n : ℕ) : Prop := ∃ m, n = m * m
def is_perfect_cube (n : ℕ) : Prop := ∃ m, n = m * m * m

-- The main statement
theorem cameron_list_count :
  let smallest_square := 25
  let smallest_cube := 125
  (∀ n : ℕ, is_multiple_of n 25 → smallest_square ≤ n → n ≤ smallest_cube) →
  ∃ count : ℕ, count = 5 :=
by 
  sorry

end cameron_list_count_l47_4771


namespace find_smaller_number_l47_4780

theorem find_smaller_number
  (x y : ℝ) (m : ℝ)
  (h1 : x - y = 9) 
  (h2 : x + y = 46)
  (h3 : x = m * y) : 
  min x y = 18.5 :=
by 
  sorry

end find_smaller_number_l47_4780


namespace general_term_formula_sum_first_n_terms_l47_4773

theorem general_term_formula :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) := 
  sorry

theorem sum_first_n_terms :
  ∀ (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, a n > 0) →
  a 1 = 1 / 2 →
  (∀ n, (a (n + 1))^2 = a n^2 + 2 * ↑n) →
  (∀ n, a n = n - 1 / 2) →
  (∀ n, b n = 1 / (a n * a (n + 1))) →
  (∀ n, S n = 2 * (1 - 1 / (2 * n + 1))) →
  (S n = 4 * n / (2 * n + 1)) :=
  sorry

end general_term_formula_sum_first_n_terms_l47_4773


namespace mike_can_buy_nine_games_l47_4763

noncomputable def mike_dollars (initial_dollars : ℕ) (spent_dollars : ℕ) (game_cost : ℕ) : ℕ :=
  (initial_dollars - spent_dollars) / game_cost

theorem mike_can_buy_nine_games : mike_dollars 69 24 5 = 9 := by
  sorry

end mike_can_buy_nine_games_l47_4763


namespace inner_circle_radius_is_sqrt_2_l47_4749

noncomputable def radius_of_inner_circle (side_length : ℝ) : ℝ :=
  let semicircle_radius := side_length / 4
  let distance_from_center_to_semicircle_center :=
    Real.sqrt ((side_length / 2) ^ 2 + (side_length / 2) ^ 2)
  let inner_circle_radius := (distance_from_center_to_semicircle_center - semicircle_radius)
  inner_circle_radius

theorem inner_circle_radius_is_sqrt_2 (side_length : ℝ) (h: side_length = 4) : 
  radius_of_inner_circle side_length = Real.sqrt 2 :=
by
  sorry

end inner_circle_radius_is_sqrt_2_l47_4749


namespace matrix_vec_addition_l47_4765

def matrix := (Fin 2 → Fin 2 → ℤ)
def vector := Fin 2 → ℤ

def m : matrix := ![![4, -2], ![6, 5]]
def v1 : vector := ![-2, 3]
def v2 : vector := ![1, -1]

def matrix_vec_mul (m : matrix) (v : vector) : vector :=
  ![m 0 0 * v 0 + m 0 1 * v 1,
    m 1 0 * v 0 + m 1 1 * v 1]

def vec_add (v1 v2 : vector) : vector :=
  ![v1 0 + v2 0, v1 1 + v2 1]

theorem matrix_vec_addition :
  vec_add (matrix_vec_mul m v1) v2 = ![-13, 2] :=
by
  sorry

end matrix_vec_addition_l47_4765


namespace n_product_expression_l47_4716

theorem n_product_expression (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1)^2 :=
sorry

end n_product_expression_l47_4716


namespace find_w_squared_l47_4710

theorem find_w_squared (w : ℝ) :
  (w + 15)^2 = (4 * w + 9) * (3 * w + 6) →
  w^2 = ((-21 + Real.sqrt 7965) / 22)^2 ∨ 
        w^2 = ((-21 - Real.sqrt 7965) / 22)^2 :=
by sorry

end find_w_squared_l47_4710


namespace complete_the_square_k_l47_4745

theorem complete_the_square_k (x : ℝ) : ∃ k : ℝ, (∃ a h: ℝ, (a = 1) ∧ (h = 7 / 2) ∧ (x^2 - 7*x = a * (x - h)^2 + k)) → k = -49 / 4 :=
by
  sorry

end complete_the_square_k_l47_4745
