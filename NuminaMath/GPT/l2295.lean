import Mathlib

namespace NUMINAMATH_GPT_pizza_non_crust_percentage_l2295_229525

theorem pizza_non_crust_percentage (total_weight crust_weight : ℕ) (h₁ : total_weight = 200) (h₂ : crust_weight = 50) :
  (total_weight - crust_weight) * 100 / total_weight = 75 :=
by
  sorry

end NUMINAMATH_GPT_pizza_non_crust_percentage_l2295_229525


namespace NUMINAMATH_GPT_incorrect_pair_l2295_229565

def roots_of_polynomial (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

theorem incorrect_pair : ¬ ∃ x : ℝ, (y = x - 1 ∧ y = x + 1 ∧ roots_of_polynomial x) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_pair_l2295_229565


namespace NUMINAMATH_GPT_find_length_AD_l2295_229514

noncomputable def length_AD (AB AC BC : ℝ) (is_equal_AB_AC : AB = AC) (BD DC : ℝ) (D_midpoint : BD = DC) : ℝ :=
  let BE := BC / 2
  let AE := Real.sqrt (AB ^ 2 - BE ^ 2)
  AE

theorem find_length_AD (AB AC BC BD DC : ℝ) (is_equal_AB_AC : AB = AC) (D_midpoint : BD = DC) (H1 : AB = 26) (H2 : AC = 26) (H3 : BC = 24) (H4 : BD = 12) (H5 : DC = 12) :
  length_AD AB AC BC is_equal_AB_AC BD DC D_midpoint = 2 * Real.sqrt 133 :=
by
  -- the steps of the proof would go here
  sorry

end NUMINAMATH_GPT_find_length_AD_l2295_229514


namespace NUMINAMATH_GPT_remainder_theorem_div_l2295_229513

noncomputable
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 5

theorem remainder_theorem_div (A B C : ℝ) (h : p A B C 2 = 13) : p A B C (-2) = 13 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_theorem_div_l2295_229513


namespace NUMINAMATH_GPT_cos_half_pi_minus_2alpha_l2295_229583

open Real

theorem cos_half_pi_minus_2alpha (α : ℝ) (h : sin α - cos α = 1 / 3) : cos (π / 2 - 2 * α) = 8 / 9 :=
sorry

end NUMINAMATH_GPT_cos_half_pi_minus_2alpha_l2295_229583


namespace NUMINAMATH_GPT_test_question_total_l2295_229544

theorem test_question_total
  (total_points : ℕ)
  (points_2q : ℕ)
  (points_4q : ℕ)
  (num_2q : ℕ)
  (num_4q : ℕ)
  (H1 : total_points = 100)
  (H2 : points_2q = 2)
  (H3 : points_4q = 4)
  (H4 : num_2q = 30)
  (H5 : total_points = num_2q * points_2q + num_4q * points_4q) :
  num_2q + num_4q = 40 := 
sorry

end NUMINAMATH_GPT_test_question_total_l2295_229544


namespace NUMINAMATH_GPT_proof_problem_l2295_229532

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem proof_problem (x1 x2 : ℝ) (h₁ : x1 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₂ : x2 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) 
                                (h₃ : f x1 + f x2 > 0) : 
  x1 + x2 > 0 :=
sorry

end NUMINAMATH_GPT_proof_problem_l2295_229532


namespace NUMINAMATH_GPT_football_team_total_progress_l2295_229591

theorem football_team_total_progress :
  let play1 := -5
  let play2 := 13
  let play3 := -2 * play1
  let play4 := play3 / 2
  play1 + play2 + play3 + play4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_football_team_total_progress_l2295_229591


namespace NUMINAMATH_GPT_possible_double_roots_l2295_229543

theorem possible_double_roots (b₃ b₂ b₁ : ℤ) (s : ℤ) :
  s^2 ∣ 50 →
  (Polynomial.eval s (Polynomial.C 50 + Polynomial.C b₁ * Polynomial.X + Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₃ * Polynomial.X^3 + Polynomial.X^4) = 0) →
  (Polynomial.eval s (Polynomial.derivative (Polynomial.C 50 + Polynomial.C b₁ * Polynomial.X + Polynomial.C b₂ * Polynomial.X^2 + Polynomial.C b₃ * Polynomial.X^3 + Polynomial.X^4)) = 0) →
  s = 1 ∨ s = -1 ∨ s = 5 ∨ s = -5 :=
by
  sorry

end NUMINAMATH_GPT_possible_double_roots_l2295_229543


namespace NUMINAMATH_GPT_at_least_one_not_greater_than_minus_four_l2295_229586

theorem at_least_one_not_greater_than_minus_four {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + 4 / b ≤ -4 ∨ b + 4 / c ≤ -4 ∨ c + 4 / a ≤ -4 :=
sorry

end NUMINAMATH_GPT_at_least_one_not_greater_than_minus_four_l2295_229586


namespace NUMINAMATH_GPT_total_pizzas_two_days_l2295_229550

-- Declare variables representing the number of pizzas made by Craig and Heather
variables (c1 c2 h1 h2 : ℕ)

-- Given conditions as definitions
def condition_1 : Prop := h1 = 4 * c1
def condition_2 : Prop := h2 = c2 - 20
def condition_3 : Prop := c1 = 40
def condition_4 : Prop := c2 = c1 + 60

-- Prove that the total number of pizzas made in two days equals 380
theorem total_pizzas_two_days (c1 c2 h1 h2 : ℕ)
  (h_cond1 : condition_1 c1 h1) (h_cond2 : condition_2 h2 c2)
  (h_cond3 : condition_3 c1) (h_cond4 : condition_4 c1 c2) :
  (h1 + c1) + (h2 + c2) = 380 :=
by
  -- As we don't need to provide the proof here, just insert sorry
  sorry

end NUMINAMATH_GPT_total_pizzas_two_days_l2295_229550


namespace NUMINAMATH_GPT_birds_meeting_distance_l2295_229522

theorem birds_meeting_distance 
  (D : ℝ) (S1 : ℝ) (S2 : ℝ) (t : ℝ)
  (H1 : D = 45)
  (H2 : S1 = 6)
  (H3 : S2 = 2.5)
  (H4 : t = D / (S1 + S2)) :
  S1 * t = 31.76 :=
by
  sorry

end NUMINAMATH_GPT_birds_meeting_distance_l2295_229522


namespace NUMINAMATH_GPT_Leroy_min_bail_rate_l2295_229551

noncomputable def min_bailing_rate
    (distance_to_shore : ℝ)
    (leak_rate : ℝ)
    (max_tolerable_water : ℝ)
    (rowing_speed : ℝ)
    : ℝ :=
  let time_to_shore := distance_to_shore / rowing_speed * 60
  let total_water_intake := leak_rate * time_to_shore
  let required_bailing := total_water_intake - max_tolerable_water
  required_bailing / time_to_shore

theorem Leroy_min_bail_rate
    (distance_to_shore : ℝ := 2)
    (leak_rate : ℝ := 15)
    (max_tolerable_water : ℝ := 60)
    (rowing_speed : ℝ := 4)
    : min_bailing_rate 2 15 60 4 = 13 := 
by
  simp [min_bailing_rate]
  sorry

end NUMINAMATH_GPT_Leroy_min_bail_rate_l2295_229551


namespace NUMINAMATH_GPT_tangent_line_at_b_l2295_229537

theorem tangent_line_at_b (b : ℝ) : (∃ x : ℝ, (4*x^3 = 4) ∧ (4*x + b = x^4 - 1)) ↔ (b = -4) := 
by 
  sorry

end NUMINAMATH_GPT_tangent_line_at_b_l2295_229537


namespace NUMINAMATH_GPT_fraction_simplification_l2295_229556

/-- Given x and y, under the conditions x ≠ 3y and x ≠ -3y, 
we want to prove that (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y). -/
theorem fraction_simplification (x y : ℝ) (h1 : x ≠ 3 * y) (h2 : x ≠ -3 * y) :
  (2 * x) / (x ^ 2 - 9 * y ^ 2) - 1 / (x - 3 * y) = 1 / (x + 3 * y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l2295_229556


namespace NUMINAMATH_GPT_minimum_positive_difference_contains_amounts_of_numbers_on_strips_l2295_229566

theorem minimum_positive_difference_contains_amounts_of_numbers_on_strips (a b c d e f : ℕ) 
  (h1 : a + f = 7) (h2 : b + e = 7) (h3 : c + d = 7) :
  ∃ (min_diff : ℕ), min_diff = 1 :=
by {
  -- The problem guarantees the minimum difference given the conditions.
  sorry
}

end NUMINAMATH_GPT_minimum_positive_difference_contains_amounts_of_numbers_on_strips_l2295_229566


namespace NUMINAMATH_GPT_two_f_x_eq_8_over_4_plus_x_l2295_229576

variable (f : ℝ → ℝ)
variable (x : ℝ)
variables (hx : 0 < x)
variable (h : ∀ x, 0 < x → f (2 * x) = 2 / (2 + x))

theorem two_f_x_eq_8_over_4_plus_x : 2 * f x = 8 / (4 + x) :=
by sorry

end NUMINAMATH_GPT_two_f_x_eq_8_over_4_plus_x_l2295_229576


namespace NUMINAMATH_GPT_Leah_lost_11_dollars_l2295_229594

-- Define the conditions
def LeahEarned : ℕ := 28
def MilkshakeCost : ℕ := LeahEarned / 7
def RemainingAfterMilkshake : ℕ := LeahEarned - MilkshakeCost
def Savings : ℕ := RemainingAfterMilkshake / 2
def WalletAfterSavings : ℕ := RemainingAfterMilkshake - Savings
def WalletAfterDog : ℕ := 1

-- Define the theorem to prove Leah's loss
theorem Leah_lost_11_dollars : WalletAfterSavings - WalletAfterDog = 11 := 
by 
  sorry

end NUMINAMATH_GPT_Leah_lost_11_dollars_l2295_229594


namespace NUMINAMATH_GPT_find_x_l2295_229510

variable (x y : ℚ)

-- Condition
def condition : Prop :=
  (x / (x - 2)) = ((y^3 + 3 * y - 2) / (y^3 + 3 * y - 5))

-- Assertion to prove
theorem find_x (h : condition x y) : x = ((2 * y^3 + 6 * y - 4) / 3) :=
sorry

end NUMINAMATH_GPT_find_x_l2295_229510


namespace NUMINAMATH_GPT_polynomial_integer_root_l2295_229521

theorem polynomial_integer_root (b : ℤ) :
  (∃ x : ℤ, x^3 + 5 * x^2 + b * x + 9 = 0) ↔ b = -127 ∨ b = -74 ∨ b = -27 ∨ b = -24 ∨ b = -15 ∨ b = -13 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_integer_root_l2295_229521


namespace NUMINAMATH_GPT_difference_of_squares_expression_l2295_229504

theorem difference_of_squares_expression
  (x y : ℝ) :
  (x + 2 * y) * (x - 2 * y) = x^2 - (2 * y)^2 :=
by sorry

end NUMINAMATH_GPT_difference_of_squares_expression_l2295_229504


namespace NUMINAMATH_GPT_train_speed_l2295_229538

theorem train_speed (L : ℝ) (T : ℝ) (V_m : ℝ) (V_t : ℝ) : (L = 500) → (T = 29.997600191984642) → (V_m = 5 / 6) → (V_t = (L / T) + V_m) → (V_t * 3.6 = 63) :=
by
  intros hL hT hVm hVt
  simp at hL hT hVm hVt
  sorry

end NUMINAMATH_GPT_train_speed_l2295_229538


namespace NUMINAMATH_GPT_common_area_of_equilateral_triangles_in_unit_square_l2295_229582

theorem common_area_of_equilateral_triangles_in_unit_square
  (unit_square_side_length : ℝ)
  (triangle_side_length : ℝ)
  (common_area : ℝ)
  (h_unit_square : unit_square_side_length = 1)
  (h_triangle_side : triangle_side_length = 1) :
  common_area = -1 :=
by
  sorry

end NUMINAMATH_GPT_common_area_of_equilateral_triangles_in_unit_square_l2295_229582


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2295_229531

theorem quadratic_inequality_solution :
  { x : ℝ | x^2 + 7*x + 6 < 0 } = { x : ℝ | -6 < x ∧ x < -1 } :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2295_229531


namespace NUMINAMATH_GPT_scientific_notation_500_billion_l2295_229539

theorem scientific_notation_500_billion :
  ∃ (a : ℝ), 500000000000 = a * 10 ^ 10 ∧ 1 ≤ a ∧ a < 10 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_500_billion_l2295_229539


namespace NUMINAMATH_GPT_three_pos_reals_inequality_l2295_229549

open Real

theorem three_pos_reals_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : a + b + c = a^2 + b^2 + c^2) :
  ((a^2) / (a^2 + b * c) + (b^2) / (b^2 + c * a) + (c^2) / (c^2 + a * b)) ≥ (a + b + c) / 2 :=
by
  sorry

end NUMINAMATH_GPT_three_pos_reals_inequality_l2295_229549


namespace NUMINAMATH_GPT_evaluate_f_l2295_229527

def f (x : ℝ) : ℝ := sorry  -- Placeholder function definition

theorem evaluate_f :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x + 5/2) = -1 / f x) ∧
  (∀ x : ℝ, x ∈ [-5/2, 0] → f x = x * (x + 5/2))
  → f 2016 = 3/2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_l2295_229527


namespace NUMINAMATH_GPT_min_visible_pairs_l2295_229540

-- Define the problem conditions
def bird_circle_flock (P : ℕ) : Prop :=
  P = 155

def mutual_visibility_condition (θ : ℝ) : Prop :=
  θ ≤ 10

-- Define the minimum number of mutually visible pairs
def min_mutual_visible_pairs (P_pairs : ℕ) : Prop :=
  P_pairs = 270

-- The main theorem statement
theorem min_visible_pairs (n : ℕ) (θ : ℝ) (P_pairs : ℕ)
  (H1 : bird_circle_flock n)
  (H2 : mutual_visibility_condition θ) :
  min_mutual_visible_pairs P_pairs :=
by
  sorry

end NUMINAMATH_GPT_min_visible_pairs_l2295_229540


namespace NUMINAMATH_GPT_find_a_value_l2295_229585

-- Problem statement
theorem find_a_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 + a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^(10)) → a = 3 :=
by sorry

end NUMINAMATH_GPT_find_a_value_l2295_229585


namespace NUMINAMATH_GPT_xy_gt_1_necessary_but_not_sufficient_l2295_229534

-- To define the conditions and prove the necessary and sufficient conditions.

variable (x y : ℝ)

-- The main statement to prove once conditions are defined.
theorem xy_gt_1_necessary_but_not_sufficient : 
  (x > 1 ∧ y > 1 → x * y > 1) ∧ ¬ (x * y > 1 → x > 1 ∧ y > 1) := 
by 
  sorry

end NUMINAMATH_GPT_xy_gt_1_necessary_but_not_sufficient_l2295_229534


namespace NUMINAMATH_GPT_find_totally_damaged_cartons_l2295_229528

def jarsPerCarton : ℕ := 20
def initialCartons : ℕ := 50
def reducedCartons : ℕ := 30
def damagedJarsPerCarton : ℕ := 3
def damagedCartons : ℕ := 5
def totalGoodJars : ℕ := 565

theorem find_totally_damaged_cartons :
  (initialCartons * jarsPerCarton - ((initialCartons - reducedCartons) * jarsPerCarton + damagedJarsPerCarton * damagedCartons - totalGoodJars)) / jarsPerCarton = 1 := by
  sorry

end NUMINAMATH_GPT_find_totally_damaged_cartons_l2295_229528


namespace NUMINAMATH_GPT_trig_identity_l2295_229596

theorem trig_identity (α m : ℝ) (h : Real.tan α = m) :
  (Real.sin (π / 4 + α))^2 - (Real.sin (π / 6 - α))^2 - Real.cos (5 * π / 12) * Real.sin (5 * π / 12 - 2 * α) = 2 * m / (1 + m^2) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l2295_229596


namespace NUMINAMATH_GPT_tan_subtraction_inequality_l2295_229569

theorem tan_subtraction_inequality (x y : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (h : Real.tan x = 3 * Real.tan y) : 
  x - y ≤ π / 6 ∧ (x - y = π / 6 ↔ (x = π / 3 ∧ y = π / 6)) := 
sorry

end NUMINAMATH_GPT_tan_subtraction_inequality_l2295_229569


namespace NUMINAMATH_GPT_find_N_l2295_229500

noncomputable def sum_of_sequence : ℤ :=
  985 + 987 + 989 + 991 + 993 + 995 + 997 + 999

theorem find_N : ∃ (N : ℤ), 8000 - N = sum_of_sequence ∧ N = 64 := by
  use 64
  -- The actual proof steps will go here
  sorry

end NUMINAMATH_GPT_find_N_l2295_229500


namespace NUMINAMATH_GPT_difference_is_correct_l2295_229597

-- Definition of the given numbers
def numbers : List ℕ := [44, 16, 2, 77, 241]

-- Define the sum of the numbers
def sum_numbers := numbers.sum

-- Define the average of the numbers
def average := sum_numbers / numbers.length

-- Define the difference between sum and average
def difference := sum_numbers - average

-- The theorem we need to prove
theorem difference_is_correct : difference = 304 := by
  sorry

end NUMINAMATH_GPT_difference_is_correct_l2295_229597


namespace NUMINAMATH_GPT_transform_polynomial_l2295_229545

theorem transform_polynomial (x y : ℝ) (h1 : y = x + 1 / x) (h2 : x^4 - x^3 - 6 * x^2 - x + 1 = 0) :
  x^2 * (y^2 - y - 6) = 0 := 
  sorry

end NUMINAMATH_GPT_transform_polynomial_l2295_229545


namespace NUMINAMATH_GPT_negation_is_false_l2295_229501

-- Define even numbers
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the original proposition P
def P (a b : ℕ) : Prop := even a ∧ even b → even (a + b)

-- The negation of the proposition P
def notP (a b : ℕ) : Prop := ¬(even a ∧ even b → even (a + b))

-- The theorem to prove
theorem negation_is_false : ∀ a b : ℕ, ¬notP a b :=
by
  sorry

end NUMINAMATH_GPT_negation_is_false_l2295_229501


namespace NUMINAMATH_GPT_perpendicular_vectors_l2295_229529

theorem perpendicular_vectors (x : ℝ) : (2 * x + 3 = 0) → (x = -3 / 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_l2295_229529


namespace NUMINAMATH_GPT_infinite_six_consecutive_epsilon_squarish_l2295_229558

def is_epsilon_squarish (ε : ℝ) (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 < a ∧ a < b ∧ b < (1 + ε) * a ∧ n = a * b

theorem infinite_six_consecutive_epsilon_squarish (ε : ℝ) (hε : 0 < ε) : 
  ∃ (N : ℕ), ∃ (n : ℕ), N ≤ n ∧
  (is_epsilon_squarish ε n) ∧ 
  (is_epsilon_squarish ε (n + 1)) ∧ 
  (is_epsilon_squarish ε (n + 2)) ∧ 
  (is_epsilon_squarish ε (n + 3)) ∧ 
  (is_epsilon_squarish ε (n + 4)) ∧ 
  (is_epsilon_squarish ε (n + 5)) :=
  sorry

end NUMINAMATH_GPT_infinite_six_consecutive_epsilon_squarish_l2295_229558


namespace NUMINAMATH_GPT_which_is_negative_l2295_229574

theorem which_is_negative
    (A : ℤ := 2023)
    (B : ℤ := -2023)
    (C : ℚ := 1/2023)
    (D : ℤ := 0) :
    B < 0 :=
by
  sorry

end NUMINAMATH_GPT_which_is_negative_l2295_229574


namespace NUMINAMATH_GPT_value_of_a_l2295_229503

theorem value_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2295_229503


namespace NUMINAMATH_GPT_find_inverse_l2295_229505

noncomputable def f (x : ℝ) := (x^7 - 1) / 5

theorem find_inverse :
  (f⁻¹ (-1 / 80) = (15 / 16)^(1 / 7)) :=
sorry

end NUMINAMATH_GPT_find_inverse_l2295_229505


namespace NUMINAMATH_GPT_geom_seq_sum_l2295_229575

variable {a : ℕ → ℝ}

theorem geom_seq_sum (h : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 ∨ a 5 + a 7 = -6 := by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_l2295_229575


namespace NUMINAMATH_GPT_smaller_investment_value_l2295_229581

theorem smaller_investment_value :
  ∃ (x : ℝ), 0.07 * x + 0.27 * 1500 = 0.22 * (x + 1500) ∧ x = 500 :=
by
  sorry

end NUMINAMATH_GPT_smaller_investment_value_l2295_229581


namespace NUMINAMATH_GPT_evaluate_expression_l2295_229554

theorem evaluate_expression : (2^3002 * 3^3004) / (6^3003) = (3 / 2) := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2295_229554


namespace NUMINAMATH_GPT_one_odd_one_even_l2295_229560

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem one_odd_one_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prime : is_prime a) (h_eq : a^2 + b^2 = c^2) : 
(is_odd b ∧ is_even c) ∨ (is_even b ∧ is_odd c) :=
sorry

end NUMINAMATH_GPT_one_odd_one_even_l2295_229560


namespace NUMINAMATH_GPT_Vasya_birthday_on_Thursday_l2295_229562

-- Define the days of the week enumeration
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define the conditions
def Sunday_is_the_day_after_tomorrow (today : DayOfWeek) : Prop :=
  match today with
  | Friday => true
  | _      => false

def Vasya_birthday_not_on_Sunday (birthday : DayOfWeek) : Prop :=
  birthday ≠ Sunday

def Today_is_next_day_after_birthday (today birthday : DayOfWeek) : Prop :=
  match (today, birthday) with
  | (Friday, Thursday) => true
  | _                  => false

-- The main theorem to prove Vasya's birthday was on Thursday
theorem Vasya_birthday_on_Thursday (birthday today : DayOfWeek) 
  (H1: Sunday_is_the_day_after_tomorrow today) 
  (H2: Vasya_birthday_not_on_Sunday birthday) 
  (H3: Today_is_next_day_after_birthday today birthday) : 
  birthday = Thursday :=
by
  sorry

end NUMINAMATH_GPT_Vasya_birthday_on_Thursday_l2295_229562


namespace NUMINAMATH_GPT_inequality_solution_l2295_229573

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Iio (-3/4) ∪ Set.Ioc 4 5 ∪ Set.Ioi 5) ↔ 
  (x+2) ≠ 0 ∧ (x-2) ≠ 0 ∧ (4 * (x^2 - 1) * (x-2) - (x+2) * (7 * x - 6)) / (4 * (x+2) * (x-2)) ≥ 0 := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2295_229573


namespace NUMINAMATH_GPT_area_ratio_l2295_229588

-- Definitions for the conditions in the problem
variables (PQ QR RP : ℝ) (p q r : ℝ)

-- Conditions
def pq_condition := PQ = 18
def qr_condition := QR = 24
def rp_condition := RP = 30
def pqr_sum := p + q + r = 3 / 4
def pqr_squaresum := p^2 + q^2 + r^2 = 1 / 2

-- Goal statement that the area ratio of triangles XYZ to PQR is 23/32
theorem area_ratio (h1 : PQ = 18) (h2 : QR = 24) (h3 : RP = 30) 
  (h4 : p + q + r = 3 / 4) (h5 : p^2 + q^2 + r^2 = 1 / 2) : 
  ∃ (m n : ℕ), (m + n = 55) ∧ (m / n = 23 / 32) := 
sorry

end NUMINAMATH_GPT_area_ratio_l2295_229588


namespace NUMINAMATH_GPT_range_of_B_l2295_229526

theorem range_of_B (A : ℝ × ℝ) (hA : A = (1, 2)) (h : 2 * A.1 - B * A.2 + 3 ≥ 0) : B ≤ 2.5 :=
by sorry

end NUMINAMATH_GPT_range_of_B_l2295_229526


namespace NUMINAMATH_GPT_intersection_complement_eq_l2295_229552

def U : Set Int := { -2, -1, 0, 1, 2, 3 }
def M : Set Int := { 0, 1, 2 }
def N : Set Int := { 0, 1, 2, 3 }

noncomputable def C_U (A : Set Int) := U \ A

theorem intersection_complement_eq :
  (C_U M ∩ N) = {3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l2295_229552


namespace NUMINAMATH_GPT_solve_for_k_l2295_229557

theorem solve_for_k : ∃ k : ℤ, 2^5 - 10 = 5^2 + k ∧ k = -3 := by
  sorry

end NUMINAMATH_GPT_solve_for_k_l2295_229557


namespace NUMINAMATH_GPT_closest_ratio_to_one_l2295_229511

theorem closest_ratio_to_one (a c : ℕ) (h1 : 2 * a + c = 130) (h2 : a ≥ 1) (h3 : c ≥ 1) : 
  a = 43 ∧ c = 44 :=
by {
    sorry 
}

end NUMINAMATH_GPT_closest_ratio_to_one_l2295_229511


namespace NUMINAMATH_GPT_ellipse_equation_l2295_229516

theorem ellipse_equation
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  (eccentricity : ℝ)
  (eccentricity_eq : eccentricity = (Real.sqrt 3 / 3))
  (perimeter_triangle : ℝ)
  (perimeter_eq : perimeter_triangle = 4 * Real.sqrt 3) :
  a = Real.sqrt 3 ∧ b = Real.sqrt 2 ∧ (a > b) ∧ (eccentricity = 1 / Real.sqrt 3) →
  (∀ x y : ℝ, (x^2 / 3 + y^2 / 2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l2295_229516


namespace NUMINAMATH_GPT_gray_region_area_l2295_229508

noncomputable def area_of_gray_region (length width : ℝ) (angle_deg : ℝ) : ℝ :=
  if (length = 55 ∧ width = 44 ∧ angle_deg = 45) then 10 else 0

theorem gray_region_area :
  area_of_gray_region 55 44 45 = 10 :=
by sorry

end NUMINAMATH_GPT_gray_region_area_l2295_229508


namespace NUMINAMATH_GPT_extended_pattern_ratio_l2295_229533

noncomputable def original_black_tiles : ℕ := 12
noncomputable def original_white_tiles : ℕ := 24
noncomputable def original_total_tiles : ℕ := 36
noncomputable def extended_total_tiles : ℕ := 64
noncomputable def border_black_tiles : ℕ := 24 /- The new border adds 24 black tiles -/
noncomputable def extended_black_tiles : ℕ := 36
noncomputable def extended_white_tiles := original_white_tiles

theorem extended_pattern_ratio :
  (extended_black_tiles : ℚ) / extended_white_tiles = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_extended_pattern_ratio_l2295_229533


namespace NUMINAMATH_GPT_parabola_equation_l2295_229559

theorem parabola_equation (d : ℝ) (p : ℝ) (x y : ℝ) (h1 : d = 2) (h2 : y = 2) (h3 : x = 1) :
  y^2 = 4 * x :=
sorry

end NUMINAMATH_GPT_parabola_equation_l2295_229559


namespace NUMINAMATH_GPT_polynomial_roots_l2295_229595

theorem polynomial_roots (d e : ℤ) :
  (∀ r, r^2 - 2 * r - 1 = 0 → r^5 - d * r - e = 0) ↔ (d = 29 ∧ e = 12) := by
  sorry

end NUMINAMATH_GPT_polynomial_roots_l2295_229595


namespace NUMINAMATH_GPT_condition_a_condition_b_condition_c_l2295_229542

-- Definitions for conditions
variable {ι : Type*} (f₁ f₂ f₃ f₄ : ι → ℝ) (x : ι)

-- First part: Condition to prove second equation is a consequence of first
theorem condition_a :
  (∀ x, f₁ x * f₄ x = f₂ x * f₃ x) →
  ((f₂ x ≠ 0) ∧ (f₂ x + f₄ x ≠ 0)) →
  (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) :=
sorry

-- Second part: Condition to prove first equation is a consequence of second
theorem condition_b :
  (∀ x, f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) →
  ((f₄ x ≠ 0) ∧ (f₂ x ≠ 0)) →
  (f₁ x * f₄ x = f₂ x * f₃ x) :=
sorry

-- Third part: Condition for equivalence of the equations
theorem condition_c :
  (∀ x, (f₁ x * f₄ x = f₂ x * f₃ x) ∧ (x ∉ {x | f₂ x + f₄ x = 0})) ↔
  (∀ x, (f₁ x * (f₂ x + f₄ x) = f₂ x * (f₁ x + f₃ x)) ∧ (x ∉ {x | f₄ x = 0})) :=
sorry

end NUMINAMATH_GPT_condition_a_condition_b_condition_c_l2295_229542


namespace NUMINAMATH_GPT_rational_solutions_for_k_l2295_229598

theorem rational_solutions_for_k :
  ∀ (k : ℕ), k > 0 → 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end NUMINAMATH_GPT_rational_solutions_for_k_l2295_229598


namespace NUMINAMATH_GPT_fraction_by_foot_l2295_229524

theorem fraction_by_foot (D distance_by_bus distance_by_car distance_by_foot : ℕ) (h1 : D = 24) 
  (h2 : distance_by_bus = D / 4) (h3 : distance_by_car = 6) 
  (h4 : distance_by_foot = D - (distance_by_bus + distance_by_car)) : 
  (distance_by_foot : ℚ) / D = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_by_foot_l2295_229524


namespace NUMINAMATH_GPT_find_coordinates_of_symmetric_point_l2295_229570

def point_on_parabola (A : ℝ × ℝ) : Prop :=
  A.2 = (A.1 - 1)^2 + 2

def symmetric_with_respect_to_axis (A A' : ℝ × ℝ) : Prop :=
  A'.1 = 2 * 1 - A.1 ∧ A'.2 = A.2

def correct_coordinates_of_A' (A' : ℝ × ℝ) : Prop :=
  A' = (3, 6)

theorem find_coordinates_of_symmetric_point (A A' : ℝ × ℝ)
  (hA : A = (-1, 6))
  (h_parabola : point_on_parabola A)
  (h_symmetric : symmetric_with_respect_to_axis A A') :
  correct_coordinates_of_A' A' :=
sorry

end NUMINAMATH_GPT_find_coordinates_of_symmetric_point_l2295_229570


namespace NUMINAMATH_GPT_nina_shoe_payment_l2295_229568

theorem nina_shoe_payment :
  let first_pair_original := 22
  let first_pair_discount := 0.10 * first_pair_original
  let first_pair_discounted := first_pair_original - first_pair_discount
  let first_pair_tax := 0.05 * first_pair_discounted
  let first_pair_final := first_pair_discounted + first_pair_tax

  let second_pair_original := first_pair_original * 1.50
  let second_pair_discount := 0.15 * second_pair_original
  let second_pair_discounted := second_pair_original - second_pair_discount
  let second_pair_tax := 0.07 * second_pair_discounted
  let second_pair_final := second_pair_discounted + second_pair_tax

  let total_payment := first_pair_final + second_pair_final
  total_payment = 50.80 :=
by 
  sorry

end NUMINAMATH_GPT_nina_shoe_payment_l2295_229568


namespace NUMINAMATH_GPT_tangent_line_at_point_l2295_229589

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = (2 * x - 1)^3) (h_point : (x, y) = (1, 1)) :
  ∃ m b : ℝ, y = m * x + b ∧ m = 6 ∧ b = -5 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_point_l2295_229589


namespace NUMINAMATH_GPT_neg_p_neither_sufficient_nor_necessary_l2295_229563

-- Definitions of p and q as described
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

-- Proving that ¬p is neither a sufficient nor necessary condition for q
theorem neg_p_neither_sufficient_nor_necessary (x : ℝ) : 
  ( ¬ p x → q x ) = false ∧ ( q x → ¬ p x ) = false := by
  sorry

end NUMINAMATH_GPT_neg_p_neither_sufficient_nor_necessary_l2295_229563


namespace NUMINAMATH_GPT_domain_of_function_l2295_229561

-- Definitions of the conditions

def sqrt_condition (x : ℝ) : Prop := -x^2 - 3*x + 4 ≥ 0
def log_condition (x : ℝ) : Prop := x + 1 > 0 ∧ x + 1 ≠ 1

-- Statement of the problem

theorem domain_of_function :
  {x : ℝ | sqrt_condition x ∧ log_condition x} = { x | -1 < x ∧ x < 0 ∨ 0 < x ∧ x ≤ 1 } :=
sorry

end NUMINAMATH_GPT_domain_of_function_l2295_229561


namespace NUMINAMATH_GPT_plain_chips_count_l2295_229587

theorem plain_chips_count (total_chips : ℕ) (BBQ_chips : ℕ)
  (hyp1 : total_chips = 9) (hyp2 : BBQ_chips = 5)
  (hyp3 : (5 * 4 / (2 * 1) : ℚ) / ((9 * 8 * 7) / (3 * 2 * 1)) = 0.11904761904761904) :
  total_chips - BBQ_chips = 4 := by
sorry

end NUMINAMATH_GPT_plain_chips_count_l2295_229587


namespace NUMINAMATH_GPT_angles_satisfy_system_l2295_229541

theorem angles_satisfy_system (k : ℤ) : 
  let x := Real.pi / 3 + k * Real.pi
  let y := k * Real.pi
  x - y = Real.pi / 3 ∧ Real.tan x - Real.tan y = Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_angles_satisfy_system_l2295_229541


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_value_l2295_229509

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)
  (h_cond : (a 5)^2 - a 3 - a 7 = 0) 
  : a 5 = 2 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_value_l2295_229509


namespace NUMINAMATH_GPT_cos_arcsin_l2295_229572

theorem cos_arcsin (h : 8^2 + 15^2 = 17^2) : Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
by
  have opp_hyp_pos : 0 ≤ 8 / 17 := by norm_num
  have hyp_pos : 0 < 17 := by norm_num
  have opp_le_hyp : 8 / 17 ≤ 1 := by norm_num
  sorry

end NUMINAMATH_GPT_cos_arcsin_l2295_229572


namespace NUMINAMATH_GPT_jerry_feathers_left_l2295_229592

def hawk_feathers : ℕ := 37
def eagle_feathers : ℝ := 17.5 * hawk_feathers
def total_feathers : ℝ := hawk_feathers + eagle_feathers
def feathers_to_sister : ℝ := 0.45 * total_feathers
def remaining_feathers_after_sister : ℝ := total_feathers - feathers_to_sister
def feathers_sold : ℝ := 0.85 * remaining_feathers_after_sister
def final_remaining_feathers : ℝ := remaining_feathers_after_sister - feathers_sold

theorem jerry_feathers_left : ⌊final_remaining_feathers⌋₊ = 56 := by
  sorry

end NUMINAMATH_GPT_jerry_feathers_left_l2295_229592


namespace NUMINAMATH_GPT_intersection_A_B_l2295_229555

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_A_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2295_229555


namespace NUMINAMATH_GPT_problem1_problem2_l2295_229530

-- Problem 1: Prove that the minimum value of f(x) is at least m for all x ∈ ℝ when k = 0
theorem problem1 (f : ℝ → ℝ) (m : ℝ) (h : ∀ x : ℝ, f x = Real.exp x - x) : m ≤ 1 := 
sorry

-- Problem 2: Prove that there exists exactly one zero of f(x) in the interval (k, 2k) when k > 1
theorem problem2 (f : ℝ → ℝ) (k : ℝ) (hk : k > 1) (h : ∀ x : ℝ, f x = Real.exp (x - k) - x) :
  ∃! (x : ℝ), x ∈ Set.Ioo k (2 * k) ∧ f x = 0 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l2295_229530


namespace NUMINAMATH_GPT_mass_percentage_H_correct_l2295_229579

noncomputable def mass_percentage_H_in_CaH2 : ℝ :=
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_H : ℝ := 1.01
  let molar_mass_CaH2 : ℝ := molar_mass_Ca + 2 * molar_mass_H
  (2 * molar_mass_H / molar_mass_CaH2) * 100

theorem mass_percentage_H_correct :
  |mass_percentage_H_in_CaH2 - 4.80| < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_H_correct_l2295_229579


namespace NUMINAMATH_GPT_sum_of_coefficients_l2295_229520

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 : ℤ) (a_eq : (1 - 2 * (0:ℤ)) ^ 7 = a)
  (hx_eq : ∀ (x : ℤ), (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2295_229520


namespace NUMINAMATH_GPT_incorrect_step_l2295_229599

-- Given conditions
variables {a b : ℝ} (hab : a < b)

-- Proof statement of the incorrect step ③
theorem incorrect_step : ¬ (2 * (a - b) ^ 2 < (a - b) ^ 2) :=
by sorry

end NUMINAMATH_GPT_incorrect_step_l2295_229599


namespace NUMINAMATH_GPT_mod_division_l2295_229507

theorem mod_division (N : ℕ) (h₁ : N = 5 * 2 + 0) : N % 4 = 2 :=
by sorry

end NUMINAMATH_GPT_mod_division_l2295_229507


namespace NUMINAMATH_GPT_part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l2295_229564

-- Conditions
def quadratic (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2 * a * x + 2 * a
def point_A (a : ℝ) : ℝ × ℝ := (-1, quadratic a (-1))
def point_B (a : ℝ) : ℝ × ℝ := (3, quadratic a 3)
def line_EF (a : ℝ) : ℝ × ℝ × ℝ × ℝ := ((a - 1), -1, (2 * a + 3), -1)

-- Statements based on solution
theorem part_1 (a : ℝ) :
  (quadratic a (-1)) = -1 := sorry

theorem part_2_max_min (a : ℝ) : 
  a = 1 → 
  (∀ x, -2 ≤ x ∧ x ≤ 3 → 
    (quadratic 1 1 = 3 ∧ 
     quadratic 1 (-2) = -6 ∧ 
     quadratic 1 3 = -1)) := sorry

theorem part_3_length_AC (a : ℝ) (h : a > -1) :
  abs ((2 * a + 1) - (-1)) = abs ((2 * a + 2)) := sorry

theorem part_4_range_a (a : ℝ) : 
  quadratic a (a-1) = -1 ∧ quadratic a (2 * a + 3) = -1 → 
  a ∈ ({-2, -1} ∪ {b : ℝ | b ≥ 0}) := sorry

end NUMINAMATH_GPT_part_1_part_2_max_min_part_3_length_AC_part_4_range_a_l2295_229564


namespace NUMINAMATH_GPT_pocket_knife_worth_40_l2295_229506

def value_of_pocket_knife (x : ℕ) (p : ℕ) (R : ℕ) : Prop :=
  p = 10 * x ∧
  R = 10 * x^2 ∧
  (∃ num_100_bills : ℕ, 2 * num_100_bills * 100 + 40 = R)

theorem pocket_knife_worth_40 (x : ℕ) (p : ℕ) (R : ℕ) :
  value_of_pocket_knife x p R → (∃ knife_value : ℕ, knife_value = 40) :=
by
  sorry

end NUMINAMATH_GPT_pocket_knife_worth_40_l2295_229506


namespace NUMINAMATH_GPT_residue_mod_17_l2295_229593

theorem residue_mod_17 : (230 * 15 - 20 * 9 + 5) % 17 = 0 :=
  by
  sorry

end NUMINAMATH_GPT_residue_mod_17_l2295_229593


namespace NUMINAMATH_GPT_bruce_eggs_lost_l2295_229518

theorem bruce_eggs_lost :
  ∀ (initial_eggs remaining_eggs eggs_lost : ℕ), 
  initial_eggs = 75 → remaining_eggs = 5 →
  eggs_lost = initial_eggs - remaining_eggs →
  eggs_lost = 70 :=
by
  intros initial_eggs remaining_eggs eggs_lost h_initial h_remaining h_loss
  sorry

end NUMINAMATH_GPT_bruce_eggs_lost_l2295_229518


namespace NUMINAMATH_GPT_triangle_area_is_18_l2295_229571

noncomputable def area_of_triangle (y_8 y_2_2x y_2_minus_2x : ℝ) : ℝ :=
  let intersect1 : ℝ × ℝ := (3, 8)
  let intersect2 : ℝ × ℝ := (-3, 8)
  let intersect3 : ℝ × ℝ := (0, 2)
  let base := 3 - -3
  let height := 8 - 2
  (1 / 2 ) * base * height

theorem triangle_area_is_18 : 
  area_of_triangle (8) (2 + 2 * x) (2 - 2 * x) = 18 := 
  by
    sorry

end NUMINAMATH_GPT_triangle_area_is_18_l2295_229571


namespace NUMINAMATH_GPT_num_small_triangles_l2295_229512

-- Define the lengths of the legs of the large and small triangles
variables (a h b k : ℕ)

-- Define the areas of the large and small triangles
def area_large_triangle (a h : ℕ) : ℕ := (a * h) / 2
def area_small_triangle (b k : ℕ) : ℕ := (b * k) / 2

-- Define the main theorem
theorem num_small_triangles (ha : a = 6) (hh : h = 4) (hb : b = 2) (hk : k = 1) :
  (area_large_triangle a h) / (area_small_triangle b k) = 12 :=
by
  sorry

end NUMINAMATH_GPT_num_small_triangles_l2295_229512


namespace NUMINAMATH_GPT_sales_discount_l2295_229584

theorem sales_discount
  (P N : ℝ)  -- original price and number of items sold
  (H1 : (1 - D / 100) * 1.3 = 1.17) -- condition when discount D is applied
  (D : ℝ)  -- sales discount percentage
  : D = 10 := by
  sorry

end NUMINAMATH_GPT_sales_discount_l2295_229584


namespace NUMINAMATH_GPT_homework_checked_on_friday_l2295_229517

theorem homework_checked_on_friday
  (prob_no_check : ℚ := 1/2)
  (prob_check_on_friday_given_check : ℚ := 1/5)
  (prob_a : ℚ := 3/5)
  : 1/3 = prob_check_on_friday_given_check / prob_a :=
by
  sorry

end NUMINAMATH_GPT_homework_checked_on_friday_l2295_229517


namespace NUMINAMATH_GPT_monster_perimeter_l2295_229546

theorem monster_perimeter (r : ℝ) (theta : ℝ) (h₁ : r = 2) (h₂ : theta = 90 * π / 180) :
  2 * r + (3 / 4) * (2 * π * r) = 3 * π + 4 := by
  -- Sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_monster_perimeter_l2295_229546


namespace NUMINAMATH_GPT_find_ending_number_divisible_by_eleven_l2295_229502

theorem find_ending_number_divisible_by_eleven (start n end_num : ℕ) (h1 : start = 29) (h2 : n = 5) (h3 : ∀ k : ℕ, ∃ m : ℕ, m = start + k * 11) : end_num = 77 :=
sorry

end NUMINAMATH_GPT_find_ending_number_divisible_by_eleven_l2295_229502


namespace NUMINAMATH_GPT_first_part_is_7613_l2295_229580

theorem first_part_is_7613 :
  ∃ (n : ℕ), ∃ (d : ℕ), d = 3 ∧ (761 * 10 + d) * 1000 + 829 = n ∧ (n % 9 = 0) ∧ (761 * 10 + d = 7613) := 
by
  sorry

end NUMINAMATH_GPT_first_part_is_7613_l2295_229580


namespace NUMINAMATH_GPT_remainder_of_exponentiated_sum_modulo_seven_l2295_229535

theorem remainder_of_exponentiated_sum_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_of_exponentiated_sum_modulo_seven_l2295_229535


namespace NUMINAMATH_GPT_cost_of_each_entree_l2295_229536

def cost_of_appetizer : ℝ := 10
def number_of_entrees : ℝ := 4
def tip_percentage : ℝ := 0.20
def total_spent : ℝ := 108

theorem cost_of_each_entree :
  ∃ E : ℝ, total_spent = cost_of_appetizer + number_of_entrees * E + tip_percentage * (cost_of_appetizer + number_of_entrees * E) ∧ E = 20 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_entree_l2295_229536


namespace NUMINAMATH_GPT_fraction_simplification_addition_l2295_229590

theorem fraction_simplification_addition :
  (∃ a b : ℕ, 0.4375 = (a : ℚ) / b ∧ Nat.gcd a b = 1 ∧ a + b = 23) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_addition_l2295_229590


namespace NUMINAMATH_GPT_percent_fewer_than_50000_is_75_l2295_229567

-- Define the given conditions as hypotheses
variables {P_1 P_2 P_3 P_4 : ℝ}
variable (h1 : P_1 = 0.35)
variable (h2 : P_2 = 0.40)
variable (h3 : P_3 = 0.15)
variable (h4 : P_4 = 0.10)

-- Define the percentage of counties with fewer than 50,000 residents
def percent_fewer_than_50000 (P_1 P_2 : ℝ) : ℝ :=
  P_1 + P_2

-- The theorem statement we need to prove
theorem percent_fewer_than_50000_is_75 (h1 : P_1 = 0.35) (h2 : P_2 = 0.40) :
  percent_fewer_than_50000 P_1 P_2 = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_percent_fewer_than_50000_is_75_l2295_229567


namespace NUMINAMATH_GPT_cherries_eaten_l2295_229577

-- Define the number of cherries Oliver had initially
def initial_cherries : ℕ := 16

-- Define the number of cherries Oliver had left after eating some
def left_cherries : ℕ := 6

-- Prove that the difference between the initial and left cherries is 10
theorem cherries_eaten : initial_cherries - left_cherries = 10 := by
  sorry

end NUMINAMATH_GPT_cherries_eaten_l2295_229577


namespace NUMINAMATH_GPT_find_P_coordinates_l2295_229519

-- Define points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (4, -3)

-- Define the theorem
theorem find_P_coordinates :
  ∃ P : ℝ × ℝ, P = (8, -15) ∧ (P.1 - A.1, P.2 - A.2) = (3 * (B.1 - A.1), 3 * (B.2 - A.2)) :=
sorry

end NUMINAMATH_GPT_find_P_coordinates_l2295_229519


namespace NUMINAMATH_GPT_cubicsum_l2295_229523

theorem cubicsum (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : a^3 + b^3 = 1008 := 
by 
  sorry

end NUMINAMATH_GPT_cubicsum_l2295_229523


namespace NUMINAMATH_GPT_parrots_count_l2295_229553

theorem parrots_count (p r : ℕ) : 2 * p + 4 * r = 26 → p + r = 10 → p = 7 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_parrots_count_l2295_229553


namespace NUMINAMATH_GPT_pow_eq_of_pow_sub_eq_l2295_229548

theorem pow_eq_of_pow_sub_eq (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^(m-n) = 2) : a^n = 3 := 
by
  sorry

end NUMINAMATH_GPT_pow_eq_of_pow_sub_eq_l2295_229548


namespace NUMINAMATH_GPT_number_of_cats_l2295_229547

theorem number_of_cats (c d : ℕ) (h1 : c = 20 + d) (h2 : c + d = 60) : c = 40 :=
sorry

end NUMINAMATH_GPT_number_of_cats_l2295_229547


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2295_229515

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (2^x + 1)
noncomputable def f_inv (x : ℝ) : ℝ := Real.logb 2 (2^x - 1)

theorem problem_1 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x = m + f x) ↔ 
  m ∈ (Set.Icc (Real.logb 2 (1/3)) (Real.logb 2 (3/5))) :=
sorry

theorem problem_2 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (3/5))) :=
sorry

theorem problem_3 : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (1/3))) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2295_229515


namespace NUMINAMATH_GPT_minimum_height_for_surface_area_geq_120_l2295_229578

noncomputable def box_surface_area (x : ℝ) : ℝ :=
  6 * x^2 + 20 * x

theorem minimum_height_for_surface_area_geq_120 :
  ∃ (x : ℝ), (x ≥ 0) ∧ (box_surface_area x ≥ 120) ∧ (x + 5 = 9) := by
  sorry

end NUMINAMATH_GPT_minimum_height_for_surface_area_geq_120_l2295_229578
