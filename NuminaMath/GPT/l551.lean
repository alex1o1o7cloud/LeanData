import Mathlib

namespace NUMINAMATH_GPT_correct_answer_is_ln_abs_l551_55101

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, (0 < x ∧ x < y) → f x ≤ f y

theorem correct_answer_is_ln_abs :
  is_even_function (fun x => Real.log (abs x)) ∧ is_monotonically_increasing_on_pos (fun x => Real.log (abs x)) ∧
  ¬ is_even_function (fun x => x^3) ∧
  ¬ is_monotonically_increasing_on_pos (fun x => Real.cos x) :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_is_ln_abs_l551_55101


namespace NUMINAMATH_GPT_can_vasya_obtain_400_mercedes_l551_55189

-- Define the types for the cars
inductive Car : Type
| Zh : Car
| V : Car
| M : Car

-- Define the initial conditions as exchange constraints
def exchange1 (Zh V M : ℕ) : Prop :=
  3 * Zh = V + M

def exchange2 (V Zh M : ℕ) : Prop :=
  3 * V = 2 * Zh + M

-- Define the initial number of Zhiguli cars Vasya has.
def initial_Zh : ℕ := 700

-- Define the target number of Mercedes cars Vasya wants.
def target_M : ℕ := 400

-- The proof goal: Vasya cannot exchange to get exactly 400 Mercedes cars.
theorem can_vasya_obtain_400_mercedes (Zh V M : ℕ) (h1 : exchange1 Zh V M) (h2 : exchange2 V Zh M) :
  initial_Zh = 700 → target_M = 400 → (Zh ≠ 0 ∨ V ≠ 0 ∨ M ≠ 400) := sorry

end NUMINAMATH_GPT_can_vasya_obtain_400_mercedes_l551_55189


namespace NUMINAMATH_GPT_expression_positive_intervals_l551_55119
open Real

theorem expression_positive_intervals (x : ℝ) :
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := by
  sorry

end NUMINAMATH_GPT_expression_positive_intervals_l551_55119


namespace NUMINAMATH_GPT_Jenny_total_wins_l551_55180

theorem Jenny_total_wins :
  let games_against_mark := 10
  let mark_wins := 1
  let mark_losses := games_against_mark - mark_wins
  let games_against_jill := 2 * games_against_mark
  let jill_wins := (75 / 100) * games_against_jill
  let jenny_wins_against_jill := games_against_jill - jill_wins
  mark_losses + jenny_wins_against_jill = 14 :=
by
  sorry

end NUMINAMATH_GPT_Jenny_total_wins_l551_55180


namespace NUMINAMATH_GPT_brenda_age_problem_l551_55146

variable (A B J : Nat)

theorem brenda_age_problem
  (h1 : A = 4 * B) 
  (h2 : J = B + 9) 
  (h3 : A = J) : 
  B = 3 := 
by 
  sorry

end NUMINAMATH_GPT_brenda_age_problem_l551_55146


namespace NUMINAMATH_GPT_find_vector_BC_l551_55162

structure Point2D where
  x : ℝ
  y : ℝ

def A : Point2D := ⟨0, 1⟩
def B : Point2D := ⟨3, 2⟩
def AC : Point2D := ⟨-4, -3⟩

def vector_add (p1 p2 : Point2D) : Point2D := ⟨p1.x + p2.x, p1.y + p2.y⟩
def vector_sub (p1 p2 : Point2D) : Point2D := ⟨p1.x - p2.x, p1.y - p2.y⟩

def C : Point2D := vector_add A AC
def BC : Point2D := vector_sub C B

theorem find_vector_BC : BC = ⟨-7, -4⟩ := by
  sorry

end NUMINAMATH_GPT_find_vector_BC_l551_55162


namespace NUMINAMATH_GPT_michael_number_l551_55138

theorem michael_number (m : ℕ) (h1 : m % 75 = 0) (h2 : m % 40 = 0) (h3 : 1000 < m) (h4 : m < 3000) :
  m = 1800 ∨ m = 2400 ∨ m = 3000 :=
sorry

end NUMINAMATH_GPT_michael_number_l551_55138


namespace NUMINAMATH_GPT_tyler_meal_combinations_is_720_l551_55198

-- Required imports for permutations and combinations
open Nat
open BigOperators

-- Assumptions based on the problem conditions
def meat_options  := 4
def veg_options := 4
def dessert_options := 5
def bread_options := 3

-- Using combinations and permutations for calculations
def comb(n k : ℕ) := Nat.choose n k
def perm(n k : ℕ) := n.factorial / (n - k).factorial

-- Number of ways to choose meals
def meal_combinations : ℕ :=
  meat_options * (comb veg_options 2) * dessert_options * (perm bread_options 2)

theorem tyler_meal_combinations_is_720 : meal_combinations = 720 := by
  -- We provide proof later; for now, put sorry to skip
  sorry

end NUMINAMATH_GPT_tyler_meal_combinations_is_720_l551_55198


namespace NUMINAMATH_GPT_foreign_students_next_semester_l551_55179

theorem foreign_students_next_semester (total_students : ℕ) (percent_foreign : ℝ) (new_foreign_students : ℕ) 
  (h_total : total_students = 1800) (h_percent : percent_foreign = 0.30) (h_new : new_foreign_students = 200) : 
  (0.30 * 1800 + 200 : ℝ) = 740 := by
  sorry

end NUMINAMATH_GPT_foreign_students_next_semester_l551_55179


namespace NUMINAMATH_GPT_alice_sold_20_pears_l551_55112

variables (S P C : ℝ)

theorem alice_sold_20_pears (h1 : C = 1.20 * P)
  (h2 : P = 0.50 * S)
  (h3 : S + P + C = 42) : S = 20 :=
by {
  -- mark the proof as incomplete with sorry
  sorry
}

end NUMINAMATH_GPT_alice_sold_20_pears_l551_55112


namespace NUMINAMATH_GPT_tickets_sold_at_door_l551_55139

theorem tickets_sold_at_door :
  ∃ D : ℕ, ∃ A : ℕ, A + D = 800 ∧ (1450 * A + 2200 * D = 166400) ∧ D = 672 :=
by
  sorry

end NUMINAMATH_GPT_tickets_sold_at_door_l551_55139


namespace NUMINAMATH_GPT_student_test_score_l551_55158

variable (C I : ℕ)

theorem student_test_score  
  (h1 : C + I = 100)
  (h2 : C - 2 * I = 64) :
  C = 88 :=
by
  -- Proof steps should go here
  sorry

end NUMINAMATH_GPT_student_test_score_l551_55158


namespace NUMINAMATH_GPT_solve_for_x_l551_55175

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 3029) : x = 200.4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l551_55175


namespace NUMINAMATH_GPT_replace_with_30_digit_nat_number_l551_55106

noncomputable def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000

theorem replace_with_30_digit_nat_number (a : Fin 10 → ℕ) (h : ∀ i, is_three_digit (a i)) :
  ∃ b : ℕ, (b < 10^30 ∧ ∃ x : ℤ, (a 9) * x^9 + (a 8) * x^8 + (a 7) * x^7 + (a 6) * x^6 + (a 5) * x^5 + 
           (a 4) * x^4 + (a 3) * x^3 + (a 2) * x^2 + (a 1) * x + (a 0) = b) :=
by
  sorry

end NUMINAMATH_GPT_replace_with_30_digit_nat_number_l551_55106


namespace NUMINAMATH_GPT_gcd_le_sqrt_sum_l551_55183

theorem gcd_le_sqrt_sum {a b : ℕ} (h : ∃ k : ℕ, (a + 1) / b + (b + 1) / a = k) :
  ↑(Nat.gcd a b) ≤ Real.sqrt (a + b) := sorry

end NUMINAMATH_GPT_gcd_le_sqrt_sum_l551_55183


namespace NUMINAMATH_GPT_part1_part2_l551_55110

-- Define the function, assumptions, and the proof for the first part
theorem part1 (m : ℝ) (x : ℝ) :
  (∀ x > 1, -m * (0 * x + 1) * Real.log x + x - 0 ≥ 0) →
  m ≤ Real.exp 1 := sorry

-- Define the function, assumptions, and the proof for the second part
theorem part2 (x : ℝ) :
  (∀ x > 0, (x - 1) * (-(x + 1) * Real.log x + x - 1) ≤ 0) := sorry

end NUMINAMATH_GPT_part1_part2_l551_55110


namespace NUMINAMATH_GPT_sum_of_cubes_of_real_roots_eq_11_l551_55159

-- Define the polynomial f(x) = x^3 - 2x^2 - x + 1
def poly (x : ℝ) : ℝ := x^3 - 2 * x^2 - x + 1

-- State that the polynomial has exactly three real roots
axiom three_real_roots : ∃ (x1 x2 x3 : ℝ), poly x1 = 0 ∧ poly x2 = 0 ∧ poly x3 = 0

-- Prove that the sum of the cubes of the real roots is 11
theorem sum_of_cubes_of_real_roots_eq_11 (x1 x2 x3 : ℝ)
  (hx1 : poly x1 = 0) (hx2 : poly x2 = 0) (hx3 : poly x3 = 0) : 
  x1^3 + x2^3 + x3^3 = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_real_roots_eq_11_l551_55159


namespace NUMINAMATH_GPT_sum_of_three_squares_l551_55117

variable (t s : ℝ)

-- Given equations
axiom h1 : 3 * t + 2 * s = 27
axiom h2 : 2 * t + 3 * s = 25

-- What we aim to prove
theorem sum_of_three_squares : 3 * s = 63 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_squares_l551_55117


namespace NUMINAMATH_GPT_range_of_m_l551_55131

theorem range_of_m (x y m : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) 
  (h3 : x < 0) (h4 : y < 0) : m < -2 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l551_55131


namespace NUMINAMATH_GPT_simplified_t_l551_55125

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem simplified_t (t : ℝ) (h : t = 1 / (3 - cuberoot 3)) : t = (3 + cuberoot 3) / 6 :=
by
  sorry

end NUMINAMATH_GPT_simplified_t_l551_55125


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l551_55172

-- Given sets A and B
def A : Set ℤ := { -1, 0, 1, 2 }
def B : Set ℤ := { 0, 2, 3 }

-- Prove that the intersection of A and B is {0, 2}
theorem intersection_of_A_and_B : A ∩ B = {0, 2} := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l551_55172


namespace NUMINAMATH_GPT_jeremy_oranges_l551_55192

theorem jeremy_oranges (M : ℕ) (h : M + 3 * M + 70 = 470) : M = 100 := 
by
  sorry

end NUMINAMATH_GPT_jeremy_oranges_l551_55192


namespace NUMINAMATH_GPT_polynomial_roots_l551_55149

-- Problem statement: prove that the roots of the given polynomial are {-1, 3, 3}
theorem polynomial_roots : 
  (λ x => x^3 - 5 * x^2 + 3 * x + 9) = (λ x => (x + 1) * (x - 3) ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l551_55149


namespace NUMINAMATH_GPT_find_w_l551_55166

noncomputable def line_p(t : ℝ) : (ℝ × ℝ) := (2 + 3 * t, 5 + 2 * t)
noncomputable def line_q(u : ℝ) : (ℝ × ℝ) := (-3 + 3 * u, 7 + 2 * u)

def vector_DC(t u : ℝ) : ℝ × ℝ := ((2 + 3 * t) - (-3 + 3 * u), (5 + 2 * t) - (7 + 2 * u))

def w_condition (w1 w2 : ℝ) : Prop := w1 + w2 = 3

theorem find_w (t u : ℝ) :
  ∃ w1 w2 : ℝ, 
    w_condition w1 w2 ∧ 
    (∃ k : ℝ, 
      sorry -- This is a placeholder for the projection calculation
    )
    :=
  sorry -- This is a placeholder for the final proof

end NUMINAMATH_GPT_find_w_l551_55166


namespace NUMINAMATH_GPT_find_x_for_slope_l551_55174

theorem find_x_for_slope (x : ℝ) (h : (2 - 5) / (x - (-3)) = -1 / 4) : x = 9 :=
by 
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_find_x_for_slope_l551_55174


namespace NUMINAMATH_GPT_distinct_c_values_l551_55182

theorem distinct_c_values (c r s t : ℂ) 
  (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_unity : ∃ ω : ℂ, ω^3 = 1 ∧ r = 1 ∧ s = ω ∧ t = ω^2)
  (h_eq : ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - c * r) * (z - c * s) * (z - c * t)) :
  ∃ (c_vals : Finset ℂ), c_vals.card = 3 ∧ ∀ (c' : ℂ), c' ∈ c_vals → c'^3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_distinct_c_values_l551_55182


namespace NUMINAMATH_GPT_evaluate_expression_l551_55163

theorem evaluate_expression (a b c : ℝ) : 
  (a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) → 
  (6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 2.4) :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l551_55163


namespace NUMINAMATH_GPT_candies_left_to_share_l551_55168

def initial_candies : Nat := 100
def sibling_count : Nat := 3
def candies_per_sibling : Nat := 10
def candies_Josh_eats : Nat := 16

theorem candies_left_to_share :
  let candies_given_to_siblings := sibling_count * candies_per_sibling;
  let candies_after_siblings := initial_candies - candies_given_to_siblings;
  let candies_given_to_friend := candies_after_siblings / 2;
  let candies_after_friend := candies_after_siblings - candies_given_to_friend;
  let candies_after_Josh := candies_after_friend - candies_Josh_eats;
  candies_after_Josh = 19 :=
by
  sorry

end NUMINAMATH_GPT_candies_left_to_share_l551_55168


namespace NUMINAMATH_GPT_min_value_a1_l551_55153

noncomputable def is_geometric_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, ∀ n : ℕ, seq (n + 1) = r * seq n

theorem min_value_a1 (a1 a2 : ℕ) (seq : ℕ → ℕ)
  (h1 : is_geometric_sequence seq)
  (h2 : ∀ n : ℕ, seq n > 0)
  (h3 : seq 20 + seq 21 = 20^21) :
  ∃ a b : ℕ, a1 = 2^a * 5^b ∧ a + b = 24 :=
sorry

end NUMINAMATH_GPT_min_value_a1_l551_55153


namespace NUMINAMATH_GPT_longest_diagonal_length_l551_55150

-- Define the conditions
variables {a b : ℝ} (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3)

-- Define the target to prove
theorem longest_diagonal_length (a b : ℝ) (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3) :
    a = 15 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_longest_diagonal_length_l551_55150


namespace NUMINAMATH_GPT_pure_imaginary_condition_l551_55121

-- Define the problem
theorem pure_imaginary_condition (θ : ℝ) :
  (∀ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi) →
  ∀ z : ℂ, z = (Complex.cos θ - Complex.sin θ * Complex.I) * (1 + Complex.I) →
  ∃ k : ℤ, θ = (3 * Real.pi / 4) + k * Real.pi → 
  (Complex.re z = 0 ∧ Complex.im z ≠ 0) :=
  sorry

end NUMINAMATH_GPT_pure_imaginary_condition_l551_55121


namespace NUMINAMATH_GPT_votes_cast_l551_55104

theorem votes_cast (candidate_percentage : ℝ) (vote_difference : ℝ) (total_votes : ℝ) 
  (h1 : candidate_percentage = 0.30) 
  (h2 : vote_difference = 1760) 
  (h3 : total_votes = vote_difference / (1 - 2 * candidate_percentage)) 
  : total_votes = 4400 := by
  sorry

end NUMINAMATH_GPT_votes_cast_l551_55104


namespace NUMINAMATH_GPT_problem_statement_l551_55115

variables (a b : ℝ)

-- Conditions: The lines \(x = \frac{1}{3}y + a\) and \(y = \frac{1}{3}x + b\) intersect at \((3, 1)\).
def lines_intersect_at (a b : ℝ) : Prop :=
  (3 = (1/3) * 1 + a) ∧ (1 = (1/3) * 3 + b)

-- Goal: Prove that \(a + b = \frac{8}{3}\)
theorem problem_statement (H : lines_intersect_at a b) : a + b = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l551_55115


namespace NUMINAMATH_GPT_trigonometric_identity_l551_55123

theorem trigonometric_identity
  (x : ℝ)
  (h_tan : Real.tan x = -1/2) :
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l551_55123


namespace NUMINAMATH_GPT_stellar_hospital_multiple_births_l551_55122

/-- At Stellar Hospital, in a particular year, the multiple-birth statistics were such that sets of twins, triplets, and quintuplets accounted for 1200 of the babies born. 
There were twice as many sets of triplets as sets of quintuplets, and there were twice as many sets of twins as sets of triplets.
Determine how many of these 1200 babies were in sets of quintuplets. -/
theorem stellar_hospital_multiple_births 
    (a b c : ℕ)
    (h1 : b = 2 * c)
    (h2 : a = 2 * b)
    (h3 : 2 * a + 3 * b + 5 * c = 1200) :
    5 * c = 316 :=
by sorry

end NUMINAMATH_GPT_stellar_hospital_multiple_births_l551_55122


namespace NUMINAMATH_GPT_cricket_innings_l551_55100

theorem cricket_innings (n : ℕ) 
  (avg_run_inn : n * 36 = n * 36)  -- average runs is 36 (initially true for any n)
  (increase_avg_by_4 : (36 * n + 120) / (n + 1) = 40) : 
  n = 20 := 
sorry

end NUMINAMATH_GPT_cricket_innings_l551_55100


namespace NUMINAMATH_GPT_sqrt_D_always_irrational_l551_55145

-- Definitions for consecutive even integers and D
def is_consecutive_even (p q : ℤ) : Prop :=
  ∃ k : ℤ, p = 2 * k ∧ q = 2 * k + 2

def D (p q : ℤ) : ℤ :=
  p^2 + q^2 + p * q^2

-- The main statement to prove
theorem sqrt_D_always_irrational (p q : ℤ) (h : is_consecutive_even p q) :
  ¬ ∃ r : ℤ, r * r = D p q :=
sorry

end NUMINAMATH_GPT_sqrt_D_always_irrational_l551_55145


namespace NUMINAMATH_GPT_total_pears_sold_l551_55185

theorem total_pears_sold (sold_morning : ℕ) (sold_afternoon : ℕ) (h_morning : sold_morning = 120) (h_afternoon : sold_afternoon = 240) :
  sold_morning + sold_afternoon = 360 :=
by
  sorry

end NUMINAMATH_GPT_total_pears_sold_l551_55185


namespace NUMINAMATH_GPT_manny_has_more_10_bills_than_mandy_l551_55113

theorem manny_has_more_10_bills_than_mandy :
  let mandy_bills_20 := 3
  let manny_bills_50 := 2
  let mandy_total_money := 20 * mandy_bills_20
  let manny_total_money := 50 * manny_bills_50
  let mandy_10_bills := mandy_total_money / 10
  let manny_10_bills := manny_total_money / 10
  mandy_10_bills < manny_10_bills →
  manny_10_bills - mandy_10_bills = 4 := sorry

end NUMINAMATH_GPT_manny_has_more_10_bills_than_mandy_l551_55113


namespace NUMINAMATH_GPT_intersection_A_B_l551_55191

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {1, 3} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l551_55191


namespace NUMINAMATH_GPT_trig_identity_l551_55134

theorem trig_identity (α : ℝ) :
  4.10 * (Real.cos (45 * Real.pi / 180 - α)) ^ 2 
  - (Real.cos (60 * Real.pi / 180 + α)) ^ 2 
  - Real.cos (75 * Real.pi / 180) * Real.sin (75 * Real.pi / 180 - 2 * α) 
  = Real.sin (2 * α) := 
sorry

end NUMINAMATH_GPT_trig_identity_l551_55134


namespace NUMINAMATH_GPT_score_below_mean_l551_55193

theorem score_below_mean :
  ∃ (σ : ℝ), (74 - 2 * σ = 58) ∧ (98 - 74 = 3 * σ) :=
sorry

end NUMINAMATH_GPT_score_below_mean_l551_55193


namespace NUMINAMATH_GPT_nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l551_55108

theorem nat_no_solution_x3_plus_5y_eq_y3_plus_5x (x y : ℕ) (h₁ : x ≠ y) : 
  x^3 + 5 * y ≠ y^3 + 5 * x :=
sorry

theorem positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5 * y = y^3 + 5 * x :=
sorry

end NUMINAMATH_GPT_nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l551_55108


namespace NUMINAMATH_GPT_largest_possible_green_socks_l551_55143

/--
A box contains a mixture of green socks and yellow socks, with at most 2023 socks in total.
The probability of randomly pulling out two socks of the same color is exactly 1/3.
What is the largest possible number of green socks in the box? 
-/
theorem largest_possible_green_socks (g y : ℤ) (t : ℕ) (h : t ≤ 2023) 
  (prob_condition : (g * (g - 1) + y * (y - 1) = t * (t - 1) / 3)) : 
  g ≤ 990 :=
sorry

end NUMINAMATH_GPT_largest_possible_green_socks_l551_55143


namespace NUMINAMATH_GPT_dave_apps_problem_l551_55126

theorem dave_apps_problem 
  (initial_apps : ℕ)
  (added_apps : ℕ)
  (final_apps : ℕ)
  (total_apps := initial_apps + added_apps)
  (deleted_apps := total_apps - final_apps) :
  initial_apps = 21 →
  added_apps = 89 →
  final_apps = 24 →
  (added_apps - deleted_apps = 3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_dave_apps_problem_l551_55126


namespace NUMINAMATH_GPT_decimal_to_vulgar_fraction_l551_55141

theorem decimal_to_vulgar_fraction :
  ∃ (n d : ℕ), (0.34 : ℝ) = (n : ℝ) / (d : ℝ) ∧ n = 17 :=
by
  sorry

end NUMINAMATH_GPT_decimal_to_vulgar_fraction_l551_55141


namespace NUMINAMATH_GPT_apples_for_pies_l551_55130

-- Define the conditions
def apples_per_pie : ℝ := 4.0
def number_of_pies : ℝ := 126.0

-- Define the expected answer
def number_of_apples : ℝ := number_of_pies * apples_per_pie

-- State the theorem to prove the question == answer given the conditions
theorem apples_for_pies : number_of_apples = 504 :=
by
  -- This is where the proof would go. Currently skipped.
  sorry

end NUMINAMATH_GPT_apples_for_pies_l551_55130


namespace NUMINAMATH_GPT_zack_traveled_countries_l551_55197

theorem zack_traveled_countries 
  (a : ℕ) (g : ℕ) (j : ℕ) (p : ℕ) (z : ℕ)
  (ha : a = 30)
  (hg : g = (3 / 5) * a)
  (hj : j = (1 / 3) * g)
  (hp : p = (4 / 3) * j)
  (hz : z = (5 / 2) * p) :
  z = 20 := 
sorry

end NUMINAMATH_GPT_zack_traveled_countries_l551_55197


namespace NUMINAMATH_GPT_switches_assembled_are_correct_l551_55147

-- Definitions based on conditions
def total_payment : ℕ := 4700
def first_worker_payment : ℕ := 2000
def second_worker_per_switch_time_min : ℕ := 4
def third_worker_less_payment : ℕ := 300
def overtime_hours : ℕ := 5
def total_minutes (hours : ℕ) : ℕ := hours * 60

-- Function to calculate total switches assembled
noncomputable def total_switches_assembled :=
  let second_worker_payment := (total_payment - first_worker_payment + third_worker_less_payment) / 2
  let third_worker_payment := second_worker_payment - third_worker_less_payment
  let rate_per_switch := second_worker_payment / (total_minutes overtime_hours / second_worker_per_switch_time_min)
  let first_worker_switches := first_worker_payment / rate_per_switch
  let second_worker_switches := total_minutes overtime_hours / second_worker_per_switch_time_min
  let third_worker_switches := third_worker_payment / rate_per_switch
  first_worker_switches + second_worker_switches + third_worker_switches

-- Lean 4 statement to prove the problem
theorem switches_assembled_are_correct : 
  total_switches_assembled = 235 := by
  sorry

end NUMINAMATH_GPT_switches_assembled_are_correct_l551_55147


namespace NUMINAMATH_GPT_quadratic_has_real_roots_find_pos_m_l551_55127

-- Proof problem 1:
theorem quadratic_has_real_roots (m : ℝ) : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x^2 - 4 * m * x + 3 * m^2 = 0 :=
by
  sorry

-- Proof problem 2:
theorem find_pos_m (m x1 x2 : ℝ) (hm : x1 > x2) (h_diff : x1 - x2 = 2)
  (h_roots : ∀ m, (x^2 - 4*m*x + 3*m^2 = 0)) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_find_pos_m_l551_55127


namespace NUMINAMATH_GPT_quadrilateral_AD_length_l551_55136

noncomputable def length_AD (AB BC CD : ℝ) (angleB angleC : ℝ) : ℝ :=
  let AE := AB + BC * Real.cos angleC
  let CE := BC * Real.sin angleC
  let DE := CD - CE
  Real.sqrt (AE^2 + DE^2)

theorem quadrilateral_AD_length :
  let AB := 7
  let BC := 10
  let CD := 24
  let angleB := Real.pi / 2 -- 90 degrees in radians
  let angleC := Real.pi / 3 -- 60 degrees in radians
  length_AD AB BC CD angleB angleC = Real.sqrt (795 - 240 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_AD_length_l551_55136


namespace NUMINAMATH_GPT_length_of_second_offset_l551_55173

theorem length_of_second_offset 
  (d : ℝ) (offset1 : ℝ) (area : ℝ) (offset2 : ℝ) 
  (h1 : d = 40)
  (h2 : offset1 = 9)
  (h3 : area = 300) :
  offset2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_offset_l551_55173


namespace NUMINAMATH_GPT_crayons_divided_equally_l551_55111

theorem crayons_divided_equally (total_crayons : ℕ) (number_of_people : ℕ) (crayons_per_person : ℕ) 
  (h1 : total_crayons = 24) (h2 : number_of_people = 3) : 
  crayons_per_person = total_crayons / number_of_people → crayons_per_person = 8 :=
by
  intro h
  rw [h1, h2] at h
  have : 24 / 3 = 8 := by norm_num
  rw [this] at h
  exact h

end NUMINAMATH_GPT_crayons_divided_equally_l551_55111


namespace NUMINAMATH_GPT_prices_of_books_book_purchasing_plans_l551_55176

-- Define the conditions
def cost_eq1 (x y : ℕ): Prop := 20 * x + 40 * y = 1520
def cost_eq2 (x y : ℕ): Prop := 20 * x - 20 * y = 440
def plan_conditions (x y : ℕ): Prop := (20 + y - x = 20) ∧ (x + y + 20 ≥ 72) ∧ (40 * x + 18 * (y + 20) ≤ 2000)

-- Prove price of each book
theorem prices_of_books : 
  ∃ (x y : ℕ), cost_eq1 x y ∧ cost_eq2 x y ∧ x = 40 ∧ y = 18 :=
by {
  sorry
}

-- Prove possible book purchasing plans
theorem book_purchasing_plans : 
  ∃ (x : ℕ), plan_conditions x (x + 20) ∧ 
  (x = 26 ∧ x + 20 = 46 ∨ 
   x = 27 ∧ x + 20 = 47 ∨ 
   x = 28 ∧ x + 20 = 48) :=
by {
  sorry
}

end NUMINAMATH_GPT_prices_of_books_book_purchasing_plans_l551_55176


namespace NUMINAMATH_GPT_range_of_m_l551_55165

-- Definitions of propositions p and q
def p (m : ℝ) : Prop := 
  (2 * m - 3)^2 - 4 > 0

def q (m : ℝ) : Prop := 
  2 * m > 3

-- Theorem statement
theorem range_of_m (m : ℝ) : ¬ (p m ∧ q m) ∧ (p m ∨ q m) ↔ (m < 1 / 2 ∨ 3 / 2 < m ∧ m ≤ 5 / 2) :=
  sorry

end NUMINAMATH_GPT_range_of_m_l551_55165


namespace NUMINAMATH_GPT_transfer_equation_correct_l551_55142

theorem transfer_equation_correct (x : ℕ) :
  46 + x = 3 * (30 - x) := 
sorry

end NUMINAMATH_GPT_transfer_equation_correct_l551_55142


namespace NUMINAMATH_GPT_towel_percentage_decrease_l551_55140

theorem towel_percentage_decrease (L B : ℝ) (hL: L > 0) (hB: B > 0) :
  let OriginalArea := L * B
  let NewLength := 0.8 * L
  let NewBreadth := 0.8 * B
  let NewArea := NewLength * NewBreadth
  let PercentageDecrease := ((OriginalArea - NewArea) / OriginalArea) * 100
  PercentageDecrease = 36 :=
by
  sorry

end NUMINAMATH_GPT_towel_percentage_decrease_l551_55140


namespace NUMINAMATH_GPT_find_a_value_l551_55184

theorem find_a_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (max (a^1) (a^2) + min (a^1) (a^2)) = 12) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l551_55184


namespace NUMINAMATH_GPT_sin_315_eq_neg_sqrt2_over_2_l551_55103

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_315_eq_neg_sqrt2_over_2_l551_55103


namespace NUMINAMATH_GPT_find_train_speed_l551_55157

variable (L V : ℝ)

-- Conditions
def condition1 := V = L / 10
def condition2 := V = (L + 600) / 30

-- Theorem statement
theorem find_train_speed (h1 : condition1 L V) (h2 : condition2 L V) : V = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_train_speed_l551_55157


namespace NUMINAMATH_GPT_nora_muffin_price_l551_55178

theorem nora_muffin_price
  (cases : ℕ)
  (packs_per_case : ℕ)
  (muffins_per_pack : ℕ)
  (total_money : ℕ)
  (total_cases : ℕ)
  (h1 : total_money = 120)
  (h2 : packs_per_case = 3)
  (h3 : muffins_per_pack = 4)
  (h4 : total_cases = 5) :
  (total_money / (total_cases * packs_per_case * muffins_per_pack) = 2) :=
by
  sorry

end NUMINAMATH_GPT_nora_muffin_price_l551_55178


namespace NUMINAMATH_GPT_fraction_of_innocent_cases_l551_55194

-- Definitions based on the given conditions
def total_cases : ℕ := 17
def dismissed_cases : ℕ := 2
def delayed_cases : ℕ := 1
def guilty_cases : ℕ := 4

-- The remaining cases after dismissals
def remaining_cases : ℕ := total_cases - dismissed_cases

-- The remaining cases that are not innocent
def non_innocent_cases : ℕ := delayed_cases + guilty_cases

-- The innocent cases
def innocent_cases : ℕ := remaining_cases - non_innocent_cases

-- The fraction of the remaining cases that were ruled innocent
def fraction_innocent : Rat := innocent_cases / remaining_cases

-- The theorem we want to prove
theorem fraction_of_innocent_cases :
  fraction_innocent = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_of_innocent_cases_l551_55194


namespace NUMINAMATH_GPT_violet_balloons_count_l551_55196

-- Define the initial number of violet balloons
def initial_violet_balloons := 7

-- Define the number of violet balloons Jason lost
def lost_violet_balloons := 3

-- Define the remaining violet balloons after losing some
def remaining_violet_balloons := initial_violet_balloons - lost_violet_balloons

-- Prove that the remaining violet balloons is equal to 4
theorem violet_balloons_count : remaining_violet_balloons = 4 :=
by
  sorry

end NUMINAMATH_GPT_violet_balloons_count_l551_55196


namespace NUMINAMATH_GPT_cori_age_proof_l551_55133

theorem cori_age_proof:
  ∃ (x : ℕ), (3 + x = (1 / 3) * (19 + x)) ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_cori_age_proof_l551_55133


namespace NUMINAMATH_GPT_not_always_true_inequality_l551_55148

theorem not_always_true_inequality (x : ℝ) (hx : x > 0) : 2^x ≤ x^2 := sorry

end NUMINAMATH_GPT_not_always_true_inequality_l551_55148


namespace NUMINAMATH_GPT_find_missing_number_l551_55152

theorem find_missing_number (x : ℕ) (h1 : (1 + 22 + 23 + 24 + x + 26 + 27 + 2) = 8 * 20) : x = 35 :=
  sorry

end NUMINAMATH_GPT_find_missing_number_l551_55152


namespace NUMINAMATH_GPT_composite_has_at_least_three_factors_l551_55186

-- Definition of composite number in terms of its factors
def is_composite (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Theorem stating that a composite number has at least 3 factors
theorem composite_has_at_least_three_factors (n : ℕ) (h : is_composite n) : 
  (∃ f1 f2 f3, f1 ∣ n ∧ f2 ∣ n ∧ f3 ∣ n ∧ f1 ≠ 1 ∧ f1 ≠ n ∧ f2 ≠ 1 ∧ f2 ≠ n ∧ f3 ≠ 1 ∧ f3 ≠ n ∧ f1 ≠ f2 ∧ f2 ≠ f3) := 
sorry

end NUMINAMATH_GPT_composite_has_at_least_three_factors_l551_55186


namespace NUMINAMATH_GPT_monotonic_decreasing_fx_l551_55124

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem monotonic_decreasing_fx : ∀ (x : ℝ), (0 < x) ∧ (x < (1 / exp 1)) → deriv f x < 0 := 
by
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_fx_l551_55124


namespace NUMINAMATH_GPT_inequality_solution_l551_55171

theorem inequality_solution (x : ℝ) :
  (∃ x, 2 < x ∧ x < 3) ↔ ∃ x, (x-2)*(x-3)/(x^2 + 1) < 0 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l551_55171


namespace NUMINAMATH_GPT_expression_value_l551_55155

theorem expression_value (a b : ℕ) (h₁ : a = 37) (h₂ : b = 12) : 
  (a + b)^2 - (a^2 + b^2) = 888 := by
  sorry

end NUMINAMATH_GPT_expression_value_l551_55155


namespace NUMINAMATH_GPT_answered_both_l551_55144

variables (A B : Type)
variables {test_takers : Type}

-- Defining the conditions
def pa : ℝ := 0.80  -- 80% answered first question correctly
def pb : ℝ := 0.75  -- 75% answered second question correctly
def pnone : ℝ := 0.05 -- 5% answered neither question correctly

-- Formal problem statement
theorem answered_both (test_takers: Type) : 
  (pa + pb - (1 - pnone)) = 0.60 :=
by
  sorry

end NUMINAMATH_GPT_answered_both_l551_55144


namespace NUMINAMATH_GPT_sqrt_mult_pow_l551_55161

theorem sqrt_mult_pow (a : ℝ) (h_nonneg : 0 ≤ a) : (a^(2/3) * a^(1/5)) = a^(13/15) := by
  sorry

end NUMINAMATH_GPT_sqrt_mult_pow_l551_55161


namespace NUMINAMATH_GPT_calculate_total_people_l551_55154

-- Definitions given in the problem
def cost_per_adult_meal := 3
def num_kids := 7
def total_cost := 15

-- The target property to prove
theorem calculate_total_people : 
  (total_cost / cost_per_adult_meal) + num_kids = 12 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_total_people_l551_55154


namespace NUMINAMATH_GPT_pipe_B_fill_time_l551_55116

variable (A B C : ℝ)
variable (fill_time : ℝ := 16)
variable (total_tank : ℝ := 1)

-- Conditions
axiom condition1 : A + B + C = (1 / fill_time)
axiom condition2 : A = 2 * B
axiom condition3 : B = 2 * C

-- Prove that B alone will take 56 hours to fill the tank
theorem pipe_B_fill_time : B = (1 / 56) :=
by sorry

end NUMINAMATH_GPT_pipe_B_fill_time_l551_55116


namespace NUMINAMATH_GPT_magic_square_sum_l551_55102

theorem magic_square_sum (a b c d e f S : ℕ) 
  (h1 : 30 + b + 22 = S) 
  (h2 : 19 + c + d = S) 
  (h3 : a + 28 + f = S)
  (h4 : 30 + 19 + a = S)
  (h5 : b + c + 28 = S)
  (h6 : 22 + d + f = S)
  (h7 : 30 + c + f = S)
  (h8 : 22 + c + a = S)
  (h9 : e = b) :
  d + e = 54 := 
by 
  sorry

end NUMINAMATH_GPT_magic_square_sum_l551_55102


namespace NUMINAMATH_GPT_determine_k_for_linear_dependence_l551_55195

theorem determine_k_for_linear_dependence :
  ∃ k : ℝ, (∀ (a1 a2 : ℝ), a1 ≠ 0 ∧ a2 ≠ 0 → 
  a1 • (⟨1, 2, 3⟩ : ℝ × ℝ × ℝ) + a2 • (⟨4, k, 6⟩ : ℝ × ℝ × ℝ) = (⟨0, 0, 0⟩ : ℝ × ℝ × ℝ)) → k = 8 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_for_linear_dependence_l551_55195


namespace NUMINAMATH_GPT_tom_filled_balloons_l551_55177

theorem tom_filled_balloons :
  ∀ (Tom Luke Anthony : ℕ), 
    (Tom = 3 * Luke) →
    (Luke = Anthony / 4) →
    (Anthony = 44) →
    (Tom = 33) :=
by
  intros Tom Luke Anthony hTom hLuke hAnthony
  sorry

end NUMINAMATH_GPT_tom_filled_balloons_l551_55177


namespace NUMINAMATH_GPT_profit_calculation_l551_55129

def totalProfit (totalMoney part1 interest1 interest2 time : ℕ) : ℕ :=
  let part2 := totalMoney - part1
  let interestFromPart1 := part1 * interest1 / 100 * time
  let interestFromPart2 := part2 * interest2 / 100 * time
  interestFromPart1 + interestFromPart2

theorem profit_calculation : 
  totalProfit 80000 70000 10 20 1 = 9000 :=
  by 
    -- Rather than providing a full proof, we insert 'sorry' as per the instruction.
    sorry

end NUMINAMATH_GPT_profit_calculation_l551_55129


namespace NUMINAMATH_GPT_simplify_radical_expression_l551_55120

noncomputable def simpl_radical_form (q : ℝ) : ℝ :=
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (2 * q^3)

theorem simplify_radical_expression (q : ℝ) :
  simpl_radical_form q = 3 * q^3 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_simplify_radical_expression_l551_55120


namespace NUMINAMATH_GPT_smallest_integer_greater_than_100_with_gcd_24_eq_4_l551_55151

theorem smallest_integer_greater_than_100_with_gcd_24_eq_4 :
  ∃ x : ℤ, x > 100 ∧ x % 24 = 4 ∧ (∀ y : ℤ, y > 100 ∧ y % 24 = 4 → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_integer_greater_than_100_with_gcd_24_eq_4_l551_55151


namespace NUMINAMATH_GPT_smallest_perimeter_l551_55132

noncomputable def smallest_possible_perimeter : ℕ :=
  let n := 3
  n + (n + 1) + (n + 2)

theorem smallest_perimeter (n : ℕ) (h : n > 2) (ineq1 : n + (n + 1) > (n + 2)) 
  (ineq2 : n + (n + 2) > (n + 1)) (ineq3 : (n + 1) + (n + 2) > n) : 
  smallest_possible_perimeter = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_perimeter_l551_55132


namespace NUMINAMATH_GPT_Sue_necklace_total_beads_l551_55181

theorem Sue_necklace_total_beads :
  ∃ (purple blue green red total : ℕ),
  purple = 7 ∧
  blue = 2 * purple ∧
  green = blue + 11 ∧
  (red : ℕ) = green / 2 ∧
  total = purple + blue + green + red ∧
  total % 2 = 0 ∧
  total = 58 := by
    sorry

end NUMINAMATH_GPT_Sue_necklace_total_beads_l551_55181


namespace NUMINAMATH_GPT_decreasing_line_implies_m_half_l551_55190

theorem decreasing_line_implies_m_half (m b : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * m - 1) * x₁ + b > (2 * m - 1) * x₂ + b) → m < 1 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_decreasing_line_implies_m_half_l551_55190


namespace NUMINAMATH_GPT_find_certain_number_l551_55135

theorem find_certain_number (x : ℝ) (h : ((x^4) * 3.456789)^10 = 10^20) : x = 10 :=
sorry

end NUMINAMATH_GPT_find_certain_number_l551_55135


namespace NUMINAMATH_GPT_basketball_holes_l551_55128

theorem basketball_holes (soccer_balls total_basketballs soccer_balls_with_hole balls_without_holes basketballs_without_holes: ℕ) 
  (h1: soccer_balls = 40) 
  (h2: total_basketballs = 15)
  (h3: soccer_balls_with_hole = 30) 
  (h4: balls_without_holes = 18) 
  (h5: basketballs_without_holes = 8) 
  : (total_basketballs - basketballs_without_holes = 7) := 
by
  sorry

end NUMINAMATH_GPT_basketball_holes_l551_55128


namespace NUMINAMATH_GPT_gcd_90_270_l551_55109

theorem gcd_90_270 : Int.gcd 90 270 = 90 :=
by
  sorry

end NUMINAMATH_GPT_gcd_90_270_l551_55109


namespace NUMINAMATH_GPT_kindergarten_solution_l551_55114

def kindergarten_cards (x y z t : ℕ) : Prop :=
  (x + y = 20) ∧ (z + t = 30) ∧ (y + z = 40) → (x + t = 10)

theorem kindergarten_solution : ∃ (x y z t : ℕ), kindergarten_cards x y z t :=
by {
  sorry
}

end NUMINAMATH_GPT_kindergarten_solution_l551_55114


namespace NUMINAMATH_GPT_algebraic_expression_value_l551_55156

theorem algebraic_expression_value (a b : ℝ) (h : ∃ x : ℝ, x = 2 ∧ 3 * (a - x) = 2 * (b * x - 4)) :
  9 * a^2 - 24 * a * b + 16 * b^2 + 25 = 29 :=
by sorry

end NUMINAMATH_GPT_algebraic_expression_value_l551_55156


namespace NUMINAMATH_GPT_worksheets_graded_l551_55170

theorem worksheets_graded (w : ℕ) (h1 : ∀ (n : ℕ), n = 3) (h2 : ∀ (n : ℕ), n = 15) (h3 : ∀ (p : ℕ), p = 24)  :
  w = 7 :=
sorry

end NUMINAMATH_GPT_worksheets_graded_l551_55170


namespace NUMINAMATH_GPT_interest_rate_of_second_part_l551_55187

theorem interest_rate_of_second_part 
  (total_sum : ℝ) (P2 : ℝ) (interest1_rate : ℝ) 
  (time1 : ℝ) (time2 : ℝ) (interest2_value : ℝ) : 
  (total_sum = 2704) → 
  (P2 = 1664) → 
  (interest1_rate = 0.03) → 
  (time1 = 8) → 
  (interest2_value = interest1_rate * (total_sum - P2) * time1) → 
  (time2 = 3) → 
  1664 * r * time2 = interest2_value → 
  r = 0.05 := 
by sorry

end NUMINAMATH_GPT_interest_rate_of_second_part_l551_55187


namespace NUMINAMATH_GPT_intersection_distance_eq_l551_55199

theorem intersection_distance_eq (p q : ℕ) (h1 : p = 88) (h2 : q = 9) :
  p - q = 79 :=
by
  sorry

end NUMINAMATH_GPT_intersection_distance_eq_l551_55199


namespace NUMINAMATH_GPT_math_problem_l551_55164

theorem math_problem (a b : ℝ) (h1 : 4 + a = 5 - b) (h2 : 5 + b = 8 + a) : 4 - a = 3 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l551_55164


namespace NUMINAMATH_GPT_identify_quadratic_equation_l551_55169

-- Definitions of the equations
def eqA : Prop := ∀ x : ℝ, x^2 + 1/x^2 = 4
def eqB : Prop := ∀ (a b x : ℝ), a*x^2 + b*x - 3 = 0
def eqC : Prop := ∀ x : ℝ, (x - 1)*(x + 2) = 1
def eqD : Prop := ∀ (x y : ℝ), 3*x^2 - 2*x*y - 5*y^2 = 0

-- Definition that identifies whether a given equation is a quadratic equation in one variable
def isQuadraticInOneVariable (eq : Prop) : Prop := 
  ∃ (a b c : ℝ) (a0 : a ≠ 0), ∀ x : ℝ, eq = (a * x^2 + b * x + c = 0)

theorem identify_quadratic_equation :
  isQuadraticInOneVariable eqC :=
by
  sorry

end NUMINAMATH_GPT_identify_quadratic_equation_l551_55169


namespace NUMINAMATH_GPT_paint_left_for_solar_system_l551_55160

-- Definitions for the paint used
def Mary's_paint := 3
def Mike's_paint := Mary's_paint + 2
def Lucy's_paint := 4

-- Total original amount of paint
def original_paint := 25

-- Total paint used by Mary, Mike, and Lucy
def total_paint_used := Mary's_paint + Mike's_paint + Lucy's_paint

-- Theorem stating the amount of paint left for the solar system
theorem paint_left_for_solar_system : (original_paint - total_paint_used) = 13 :=
by
  sorry

end NUMINAMATH_GPT_paint_left_for_solar_system_l551_55160


namespace NUMINAMATH_GPT_total_cost_proof_l551_55167

def uber_cost : ℤ := 22
def lyft_cost : ℤ := uber_cost - 3
def taxi_cost : ℤ := lyft_cost - 4
def tip : ℤ := (taxi_cost * 20) / 100
def total_cost : ℤ := taxi_cost + tip

theorem total_cost_proof :
  total_cost = 18 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_proof_l551_55167


namespace NUMINAMATH_GPT_number_of_citroens_submerged_is_zero_l551_55188

-- Definitions based on the conditions
variables (x y : ℕ) -- Define x as the number of Citroen and y as the number of Renault submerged
variables (r p c vr vp : ℕ) -- Define r as the number of Renault, p as the number of Peugeot, c as the number of Citroën

-- Given conditions translated
-- Condition 1: There were twice as many Renault cars as there were Peugeot cars
def condition1 (r p : ℕ) : Prop := r = 2 * p
-- Condition 2: There were twice as many Peugeot cars as there were Citroens
def condition2 (p c : ℕ) : Prop := p = 2 * c
-- Condition 3: As many Citroens as Renaults were submerged in the water
def condition3 (x y : ℕ) : Prop := y = x
-- Condition 4: Three times as many Renaults were in the water as there were Peugeots
def condition4 (r y : ℕ) : Prop := r = 3 * y
-- Condition 5: As many Peugeots visible in the water as there were Citroens
def condition5 (vp c : ℕ) : Prop := vp = c

-- The question to prove: The number of Citroen cars submerged is 0
theorem number_of_citroens_submerged_is_zero
  (h1 : condition1 r p) 
  (h2 : condition2 p c)
  (h3 : condition3 x y)
  (h4 : condition4 r y)
  (h5 : condition5 vp c) :
  x = 0 :=
sorry

end NUMINAMATH_GPT_number_of_citroens_submerged_is_zero_l551_55188


namespace NUMINAMATH_GPT_denomination_calculation_l551_55137

variables (total_money rs_50_count total_count rs_50_value remaining_count remaining_amount remaining_denomination_value : ℕ)

theorem denomination_calculation 
  (h1 : total_money = 10350)
  (h2 : rs_50_count = 97)
  (h3 : total_count = 108)
  (h4 : rs_50_value = 50)
  (h5 : remaining_count = total_count - rs_50_count)
  (h6 : remaining_amount = total_money - rs_50_count * rs_50_value)
  (h7 : remaining_denomination_value = remaining_amount / remaining_count) :
  remaining_denomination_value = 500 := 
sorry

end NUMINAMATH_GPT_denomination_calculation_l551_55137


namespace NUMINAMATH_GPT_minimum_value_is_16_l551_55118

noncomputable def minimum_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) : ℝ :=
  (x^3 / (y - 1) + y^3 / (x - 1))

theorem minimum_value_is_16 (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  minimum_value_expression x y hx hy ≥ 16 :=
sorry

end NUMINAMATH_GPT_minimum_value_is_16_l551_55118


namespace NUMINAMATH_GPT_fractions_sum_to_one_l551_55107

theorem fractions_sum_to_one :
  ∃ (a b c : ℕ), (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ ((a, b, c) = (2, 3, 6) ∨ (a, b, c) = (2, 6, 3) ∨ (a, b, c) = (3, 2, 6) ∨ (a, b, c) = (3, 6, 2) ∨ (a, b, c) = (6, 2, 3) ∨ (a, b, c) = (6, 3, 2)) :=
by
  sorry

end NUMINAMATH_GPT_fractions_sum_to_one_l551_55107


namespace NUMINAMATH_GPT_real_condition_complex_condition_pure_imaginary_condition_l551_55105

-- Definitions for our conditions
def is_real (z : ℂ) : Prop := z.im = 0
def is_complex (z : ℂ) : Prop := z.im ≠ 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- The given complex number definition
def z (m : ℝ) : ℂ := { re := m^2 + m, im := m^2 - 1 }

-- Prove that for z to be a real number, m must be ±1
theorem real_condition (m : ℝ) : is_real (z m) ↔ m = 1 ∨ m = -1 := 
sorry

-- Prove that for z to be a complex number, m must not be ±1 
theorem complex_condition (m : ℝ) : is_complex (z m) ↔ m ≠ 1 ∧ m ≠ -1 := 
sorry 

-- Prove that for z to be a pure imaginary number, m must be 0
theorem pure_imaginary_condition (m : ℝ) : is_pure_imaginary (z m) ↔ m = 0 := 
sorry 

end NUMINAMATH_GPT_real_condition_complex_condition_pure_imaginary_condition_l551_55105
