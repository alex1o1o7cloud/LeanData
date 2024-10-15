import Mathlib

namespace NUMINAMATH_GPT_greater_number_l1724_172430

theorem greater_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 12) : a = 26 :=
by
  have h3 : 2 * a = 52 := by linarith
  have h4 : a = 26 := by linarith
  exact h4

end NUMINAMATH_GPT_greater_number_l1724_172430


namespace NUMINAMATH_GPT_simplify_fraction_l1724_172406

theorem simplify_fraction (x : ℚ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1724_172406


namespace NUMINAMATH_GPT_dice_probability_l1724_172445

theorem dice_probability :
  let one_digit_prob := 9 / 20
  let two_digit_prob := 11 / 20
  let number_of_dice := 5
  ∃ p : ℚ,
    (number_of_dice.choose 2) * (one_digit_prob ^ 2) * (two_digit_prob ^ 3) = p ∧
    p = 107811 / 320000 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_l1724_172445


namespace NUMINAMATH_GPT_gcd_442872_312750_l1724_172488

theorem gcd_442872_312750 : Nat.gcd 442872 312750 = 18 :=
by
  sorry

end NUMINAMATH_GPT_gcd_442872_312750_l1724_172488


namespace NUMINAMATH_GPT_neg_triangle_obtuse_angle_l1724_172434

theorem neg_triangle_obtuse_angle : 
  (¬ ∀ (A B C : ℝ), A + B + C = π → max (max A B) C < π/2) ↔ (∃ (A B C : ℝ), A + B + C = π ∧ min (min A B) C > π/2) :=
by
  sorry

end NUMINAMATH_GPT_neg_triangle_obtuse_angle_l1724_172434


namespace NUMINAMATH_GPT_scientific_notation_of_384000_l1724_172415

theorem scientific_notation_of_384000 :
  (384000 : ℝ) = 3.84 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_384000_l1724_172415


namespace NUMINAMATH_GPT_total_meals_per_week_l1724_172496

-- Definitions for the conditions
def first_restaurant_meals := 20
def second_restaurant_meals := 40
def third_restaurant_meals := 50
def days_in_week := 7

-- The theorem for the total meals per week
theorem total_meals_per_week : 
  (first_restaurant_meals + second_restaurant_meals + third_restaurant_meals) * days_in_week = 770 := 
by
  sorry

end NUMINAMATH_GPT_total_meals_per_week_l1724_172496


namespace NUMINAMATH_GPT_product_of_three_greater_than_two_or_four_of_others_l1724_172444

theorem product_of_three_greater_than_two_or_four_of_others 
  (x : Fin 10 → ℕ) 
  (h_unique : ∀ i j : Fin 10, i ≠ j → x i ≠ x j) 
  (h_positive : ∀ i : Fin 10, 0 < x i) : 
  ∃ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ a b : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ b ≠ i ∧ b ≠ j ∧ b ≠ k → 
      x i * x j * x k > x a * x b) ∨ 
    (∀ a b c d : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ 
      b ≠ i ∧ b ≠ j ∧ b ≠ k ∧ 
      c ≠ i ∧ c ≠ j ∧ c ≠ k ∧ 
      d ≠ i ∧ d ≠ j ∧ d ≠ k → 
      x i * x j * x k > x a * x b * x c * x d) := sorry

end NUMINAMATH_GPT_product_of_three_greater_than_two_or_four_of_others_l1724_172444


namespace NUMINAMATH_GPT_find_f_expression_find_f_range_l1724_172420

noncomputable def y (t x : ℝ) : ℝ := 1 - 2 * t - 2 * t * x + 2 * x ^ 2

noncomputable def f (t : ℝ) : ℝ := 
  if t < -2 then 3 
  else if t > 2 then -4 * t + 3 
  else -t ^ 2 / 2 - 2 * t + 1

theorem find_f_expression (t : ℝ) : 
  f t = if t < -2 then 3 else 
          if t > 2 then -4 * t + 3 
          else - t ^ 2 / 2 - 2 * t + 1 :=
sorry

theorem find_f_range (t : ℝ) (ht : -2 ≤ t ∧ t ≤ 0) : 
  1 ≤ f t ∧ f t ≤ 3 := 
sorry

end NUMINAMATH_GPT_find_f_expression_find_f_range_l1724_172420


namespace NUMINAMATH_GPT_general_term_formula_l1724_172475

noncomputable def a (n : ℕ) : ℝ := 1 / (Real.sqrt n)

theorem general_term_formula :
  ∀ (n : ℕ), a n = 1 / Real.sqrt n :=
by
  intros
  rfl

end NUMINAMATH_GPT_general_term_formula_l1724_172475


namespace NUMINAMATH_GPT_range_of_k_l1724_172452

variable (k : ℝ)
def f (x : ℝ) : ℝ := k * x + 1
def g (x : ℝ) : ℝ := x^2 - 1

theorem range_of_k (h : ∀ x : ℝ, f k x > 0 ∨ g x > 0) : k ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) := 
sorry

end NUMINAMATH_GPT_range_of_k_l1724_172452


namespace NUMINAMATH_GPT_father_l1724_172495

-- Define the variables
variables (F S : ℕ)

-- Define the conditions
def condition1 : Prop := F = 4 * S
def condition2 : Prop := F + 20 = 2 * (S + 20)
def condition3 : Prop := S = 10

-- Statement of the problem
theorem father's_age (h1 : condition1 F S) (h2 : condition2 F S) (h3 : condition3 S) : F = 40 :=
by sorry

end NUMINAMATH_GPT_father_l1724_172495


namespace NUMINAMATH_GPT_problem_1_problem_2_0_lt_a_lt_1_problem_2_a_gt_1_l1724_172438

noncomputable def f (a x : ℝ) := a^(3 * x + 1)
noncomputable def g (a x : ℝ) := (1 / a)^(5 * x - 2)

variables {a x : ℝ}

theorem problem_1 (h : 0 < a ∧ a < 1) : f a x < 1 ↔ x > -1/3 :=
sorry

theorem problem_2_0_lt_a_lt_1 (h : 0 < a ∧ a < 1) : f a x ≥ g a x ↔ x ≤ 1 / 8 :=
sorry

theorem problem_2_a_gt_1 (h : a > 1) : f a x ≥ g a x ↔ x ≥ 1 / 8 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_0_lt_a_lt_1_problem_2_a_gt_1_l1724_172438


namespace NUMINAMATH_GPT_distance_covered_l1724_172442

-- Define the rate and time as constants
def rate : ℝ := 4 -- 4 miles per hour
def time : ℝ := 2 -- 2 hours

-- Theorem statement: Verify the distance covered
theorem distance_covered : rate * time = 8 := 
by
  sorry

end NUMINAMATH_GPT_distance_covered_l1724_172442


namespace NUMINAMATH_GPT_rectangle_area_l1724_172451

theorem rectangle_area (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 4) :
  l * w = 8 / 9 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1724_172451


namespace NUMINAMATH_GPT_total_volume_correct_l1724_172483

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2812

-- Define the target volume
def total_volume_of_water : ℕ := 11248

-- The theorem to be proved
theorem total_volume_correct : volume_of_hemisphere * number_of_hemispheres = total_volume_of_water :=
by
  sorry

end NUMINAMATH_GPT_total_volume_correct_l1724_172483


namespace NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l1724_172411

-- For the first quadratic equation: 3x^2 = 6x
theorem solve_quadratic1 (x : ℝ) (h : 3 * x^2 = 6 * x) : x = 0 ∨ x = 2 :=
sorry

-- For the second quadratic equation: x^2 - 6x + 5 = 0
theorem solve_quadratic2 (x : ℝ) (h : x^2 - 6 * x + 5 = 0) : x = 5 ∨ x = 1 :=
sorry

end NUMINAMATH_GPT_solve_quadratic1_solve_quadratic2_l1724_172411


namespace NUMINAMATH_GPT_find_n_divides_2_pow_2000_l1724_172423

theorem find_n_divides_2_pow_2000 (n : ℕ) (h₁ : n > 2) :
  (1 + n + n * (n - 1) / 2 + n * (n - 1) * (n - 2) / 6) ∣ (2 ^ 2000) →
  n = 3 ∨ n = 7 ∨ n = 23 :=
sorry

end NUMINAMATH_GPT_find_n_divides_2_pow_2000_l1724_172423


namespace NUMINAMATH_GPT_sophomores_bought_15_more_markers_l1724_172401

theorem sophomores_bought_15_more_markers (f_cost s_cost marker_cost : ℕ) (hf: f_cost = 267) (hs: s_cost = 312) (hm: marker_cost = 3) : 
  (s_cost / marker_cost) - (f_cost / marker_cost) = 15 :=
by
  sorry

end NUMINAMATH_GPT_sophomores_bought_15_more_markers_l1724_172401


namespace NUMINAMATH_GPT_root_value_cond_l1724_172404

theorem root_value_cond (p q : ℝ) (h₁ : ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = q) (h₂ : q ≠ 0) : p + q = -1 := 
sorry

end NUMINAMATH_GPT_root_value_cond_l1724_172404


namespace NUMINAMATH_GPT_nth_equation_l1724_172463

theorem nth_equation (n : ℕ) (h : n > 0) : (1 / n) * ((n^2 + 2 * n) / (n + 1)) - (1 / (n + 1)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_l1724_172463


namespace NUMINAMATH_GPT_opposite_neg_two_l1724_172416

def opposite (x : Int) : Int := -x

theorem opposite_neg_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_GPT_opposite_neg_two_l1724_172416


namespace NUMINAMATH_GPT_number_of_squares_in_figure_100_l1724_172454

theorem number_of_squares_in_figure_100 :
  ∃ (a b c : ℤ), (c = 1) ∧ (a + b + c = 7) ∧ (4 * a + 2 * b + c = 19) ∧ (3 * 100^2 + 3 * 100 + 1 = 30301) :=
sorry

end NUMINAMATH_GPT_number_of_squares_in_figure_100_l1724_172454


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1724_172427

theorem value_of_a_plus_b (a b : ℝ) : (|a - 1| + (b + 3)^2 = 0) → (a + b = -2) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1724_172427


namespace NUMINAMATH_GPT_price_of_two_identical_filters_l1724_172449

def price_of_individual_filters (x : ℝ) : Prop :=
  let total_individual := 2 * 14.05 + 19.50 + 2 * x
  total_individual = 87.50 / 0.92

theorem price_of_two_identical_filters
  (h1 : price_of_individual_filters 23.76) :
  23.76 * 2 + 28.10 + 19.50 = 87.50 / 0.92 :=
by sorry

end NUMINAMATH_GPT_price_of_two_identical_filters_l1724_172449


namespace NUMINAMATH_GPT_ratio_of_ages_l1724_172460

theorem ratio_of_ages (age_saras age_kul : ℕ) (h_saras : age_saras = 33) (h_kul : age_kul = 22) : 
  age_saras / Nat.gcd age_saras age_kul = 3 ∧ age_kul / Nat.gcd age_saras age_kul = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1724_172460


namespace NUMINAMATH_GPT_a_c3_b3_equiv_zero_l1724_172492

-- Definitions based on conditions
def cubic_eq_has_geom_progression_roots (a b c : ℝ) :=
  ∃ d q : ℝ, d ≠ 0 ∧ q ≠ 0 ∧ d + d * q + d * q^2 = -a ∧
    d^2 * q * (1 + q + q^2) = b ∧
    d^3 * q^3 = -c

-- Main theorem to prove
theorem a_c3_b3_equiv_zero (a b c : ℝ) :
  cubic_eq_has_geom_progression_roots a b c → a^3 * c - b^3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_a_c3_b3_equiv_zero_l1724_172492


namespace NUMINAMATH_GPT_smallest_n_for_sum_or_difference_divisible_l1724_172494

theorem smallest_n_for_sum_or_difference_divisible (n : ℕ) :
  (∃ n : ℕ, ∀ (S : Finset ℤ), S.card = n → (∃ (x y : ℤ) (h₁ : x ≠ y), ((x + y) % 1991 = 0) ∨ ((x - y) % 1991 = 0))) ↔ n = 997 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_sum_or_difference_divisible_l1724_172494


namespace NUMINAMATH_GPT_find_g_5_l1724_172421

-- Define the function g and the condition it satisfies
variable {g : ℝ → ℝ}
variable (hg : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x)

-- The proof goal
theorem find_g_5 : g 5 = 206 / 35 :=
by
  -- To be proven using the given condition hg
  sorry

end NUMINAMATH_GPT_find_g_5_l1724_172421


namespace NUMINAMATH_GPT_find_number_l1724_172425

noncomputable def N := 953.87

theorem find_number (h : (0.47 * N - 0.36 * 1412) + 65 = 5) : N = 953.87 := sorry

end NUMINAMATH_GPT_find_number_l1724_172425


namespace NUMINAMATH_GPT_equation_conditions_l1724_172468

theorem equation_conditions (m n : ℤ) (h1 : m ≠ 1) (h2 : n = 1) :
  ∃ x : ℤ, (m - 1) * x = 3 ↔ m = -2 ∨ m = 0 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end NUMINAMATH_GPT_equation_conditions_l1724_172468


namespace NUMINAMATH_GPT_flowers_per_basket_l1724_172412

-- Definitions derived from the conditions
def initial_flowers : ℕ := 10
def grown_flowers : ℕ := 20
def dead_flowers : ℕ := 10
def baskets : ℕ := 5

-- Theorem stating the equivalence of the problem to its solution
theorem flowers_per_basket :
  (initial_flowers + grown_flowers - dead_flowers) / baskets = 4 :=
by
  sorry

end NUMINAMATH_GPT_flowers_per_basket_l1724_172412


namespace NUMINAMATH_GPT_scientific_notation_to_decimal_l1724_172472

theorem scientific_notation_to_decimal :
  5.2 * 10^(-5) = 0.000052 :=
sorry

end NUMINAMATH_GPT_scientific_notation_to_decimal_l1724_172472


namespace NUMINAMATH_GPT_integer_type_l1724_172417

theorem integer_type (f : ℕ) (h : f = 14) (x : ℕ) (hx : 3150 * f = x * x) : f > 0 :=
by
  sorry

end NUMINAMATH_GPT_integer_type_l1724_172417


namespace NUMINAMATH_GPT_percentage_loss_is_correct_l1724_172410

-- Define the cost price and selling price
def cost_price : ℕ := 2000
def selling_price : ℕ := 1800

-- Define the calculation of loss and percentage loss
def loss (cp sp : ℕ) := cp - sp
def percentage_loss (loss cp : ℕ) := (loss * 100) / cp

-- The goal is to prove that the percentage loss is 10%
theorem percentage_loss_is_correct : percentage_loss (loss cost_price selling_price) cost_price = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_loss_is_correct_l1724_172410


namespace NUMINAMATH_GPT_question_mark_value_l1724_172469

theorem question_mark_value :
  ∀ (x : ℕ), ( ( (5568: ℝ) / (x: ℝ) )^(1/3: ℝ) + ( (72: ℝ) * (2: ℝ) )^(1/2: ℝ) = (256: ℝ)^(1/2: ℝ) ) → x = 87 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_question_mark_value_l1724_172469


namespace NUMINAMATH_GPT_solution_set_of_xf_x_gt_0_l1724_172407

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = - f x
axiom h2 : f 2 = 0
axiom h3 : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x < 0

theorem solution_set_of_xf_x_gt_0 :
  {x : ℝ | x * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_of_xf_x_gt_0_l1724_172407


namespace NUMINAMATH_GPT_systematic_sampling_40th_number_l1724_172474

open Nat

theorem systematic_sampling_40th_number (N n : ℕ) (sample_size_eq : n = 50) (total_students_eq : N = 1000) (k_def : k = N / n) (first_number : ℕ) (first_number_eq : first_number = 15) : 
  first_number + k * 39 = 795 := by
  sorry

end NUMINAMATH_GPT_systematic_sampling_40th_number_l1724_172474


namespace NUMINAMATH_GPT_width_of_rectangle_l1724_172424

-- Define the side length of the square and the length of the rectangle.
def side_length_square : ℝ := 12
def length_rectangle : ℝ := 18

-- Calculate the perimeter of the square.
def perimeter_square : ℝ := 4 * side_length_square

-- This definition represents the perimeter of the rectangle made from the same wire.
def perimeter_rectangle : ℝ := perimeter_square

-- Show that the width of the rectangle is 6 cm.
theorem width_of_rectangle : ∃ W : ℝ, 2 * (length_rectangle + W) = perimeter_rectangle ∧ W = 6 :=
by
  use 6
  simp [length_rectangle, perimeter_rectangle, side_length_square]
  norm_num
  sorry

end NUMINAMATH_GPT_width_of_rectangle_l1724_172424


namespace NUMINAMATH_GPT_houses_with_garage_l1724_172479

theorem houses_with_garage (P GP N : ℕ) (hP : P = 40) (hGP : GP = 35) (hN : N = 10) 
    (total_houses : P + GP - GP + N = 65) : 
    P + 65 - P - GP + GP - N = 50 :=
by
  sorry

end NUMINAMATH_GPT_houses_with_garage_l1724_172479


namespace NUMINAMATH_GPT_andrew_age_l1724_172409

variable (a g s : ℝ)

theorem andrew_age :
  g = 10 * a ∧ g - s = a + 45 ∧ s = 5 → a = 50 / 9 := by
  sorry

end NUMINAMATH_GPT_andrew_age_l1724_172409


namespace NUMINAMATH_GPT_fraction_equality_l1724_172413

theorem fraction_equality :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 59 / 61 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l1724_172413


namespace NUMINAMATH_GPT_gcd_expression_infinite_composite_pairs_exists_l1724_172433

-- Part (a)
theorem gcd_expression (n : ℕ) (a : ℕ) (b : ℕ) (hn : n > 0) (ha : a > 0) (hb : b > 0) :
  Nat.gcd (n^a + 1) (n^b + 1) ≤ n^(Nat.gcd a b) + 1 :=
by
  sorry

-- Part (b)
theorem infinite_composite_pairs_exists (n : ℕ) (hn : n > 0) :
  ∃ (pairs : ℕ × ℕ → Prop), (∀ a b, pairs (a, b) → a > 1 ∧ b > 1 ∧ ∃ d, d > 1 ∧ a = d ∧ b = dn) ∧
  (∀ a b, pairs (a, b) → Nat.gcd (n^a + 1) (n^b + 1) = n^(Nat.gcd a b) + 1) ∧
  (∀ x y, x > 1 → y > 1 → x ∣ y ∨ y ∣ x → ¬pairs (x, y)) :=
by
  sorry

end NUMINAMATH_GPT_gcd_expression_infinite_composite_pairs_exists_l1724_172433


namespace NUMINAMATH_GPT_fraction_of_clerical_staff_is_one_third_l1724_172486

-- Defining the conditions
variables (employees clerical_f clerical employees_reduced employees_remaining : ℝ)

def company_conditions (employees clerical_f clerical employees_reduced employees_remaining : ℝ) : Prop :=
  employees = 3600 ∧
  clerical = 3600 * clerical_f ∧
  employees_reduced = clerical * (2 / 3) ∧
  employees_remaining = employees - clerical * (1 / 3) ∧
  employees_reduced = 0.25 * employees_remaining

-- The statement to prove the fraction of clerical employees given the conditions
theorem fraction_of_clerical_staff_is_one_third
  (hc : company_conditions employees clerical_f clerical employees_reduced employees_remaining) :
  clerical_f = 1 / 3 :=
sorry

end NUMINAMATH_GPT_fraction_of_clerical_staff_is_one_third_l1724_172486


namespace NUMINAMATH_GPT_votes_for_veggies_l1724_172477

theorem votes_for_veggies (T M V : ℕ) (hT : T = 672) (hM : M = 335) (hV : V = T - M) : V = 337 := 
by
  rw [hT, hM] at hV
  simp at hV
  exact hV

end NUMINAMATH_GPT_votes_for_veggies_l1724_172477


namespace NUMINAMATH_GPT_custom_op_two_neg_four_l1724_172466

-- Define the binary operation *
def custom_op (x y : ℚ) : ℚ := (x * y) / (x + y)

-- Proposition stating 2 * (-4) = 4 using the custom operation
theorem custom_op_two_neg_four : custom_op 2 (-4) = 4 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_two_neg_four_l1724_172466


namespace NUMINAMATH_GPT_sufficiency_not_necessity_l1724_172461

theorem sufficiency_not_necessity (x y : ℝ) :
  (x > 3 ∧ y > 3) → (x + y > 6 ∧ x * y > 9) ∧ (¬ (x + y > 6 ∧ x * y > 9 → x > 3 ∧ y > 3)) :=
by
  sorry

end NUMINAMATH_GPT_sufficiency_not_necessity_l1724_172461


namespace NUMINAMATH_GPT_rate_of_current_in_river_l1724_172403

theorem rate_of_current_in_river (b c : ℝ) (h1 : 4 * (b + c) = 24) (h2 : 6 * (b - c) = 24) : c = 1 := by
  sorry

end NUMINAMATH_GPT_rate_of_current_in_river_l1724_172403


namespace NUMINAMATH_GPT_problem_equivalence_of_angles_l1724_172429

noncomputable def ctg (x : ℝ) : ℝ := 1 / (Real.tan x)

theorem problem_equivalence_of_angles
  (a b c t S ω : ℝ)
  (hS : S = Real.sqrt ((a^2 + b^2 + c^2)^2 + (4 * t)^2))
  (h1 : ctg ω = (a^2 + b^2 + c^2) / (4 * t))
  (h2 : Real.cos ω = (a^2 + b^2 + c^2) / S)
  (h3 : Real.sin ω = (4 * t) / S) :
  True :=
sorry

end NUMINAMATH_GPT_problem_equivalence_of_angles_l1724_172429


namespace NUMINAMATH_GPT_calculate_shot_cost_l1724_172458

theorem calculate_shot_cost :
  let num_pregnant_dogs := 3
  let puppies_per_dog := 4
  let shots_per_puppy := 2
  let cost_per_shot := 5
  let total_puppies := num_pregnant_dogs * puppies_per_dog
  let total_shots := total_puppies * shots_per_puppy
  let total_cost := total_shots * cost_per_shot
  total_cost = 120 :=
by
  sorry

end NUMINAMATH_GPT_calculate_shot_cost_l1724_172458


namespace NUMINAMATH_GPT_smallest_c_for_defined_expression_l1724_172455

theorem smallest_c_for_defined_expression :
  ∃ (c : ℤ), (∀ x : ℝ, x^2 + (c : ℝ) * x + 15 ≠ 0) ∧
             (∀ k : ℤ, (∀ x : ℝ, x^2 + (k : ℝ) * x + 15 ≠ 0) → c ≤ k) ∧
             c = -7 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_c_for_defined_expression_l1724_172455


namespace NUMINAMATH_GPT_intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l1724_172498

noncomputable def h (a x : ℝ) : ℝ := a * x^3 - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x

noncomputable def f (a x : ℝ) : ℝ := h a x + 3 * x * g x
noncomputable def F (a x : ℝ) : ℝ := (a - (1/3)) * x^3 + (1/2) * x^2 * g a - h a x - 1

theorem intervals_of_monotonicity (a : ℝ) (ha : f a 1 = -1) :
  ((a = 0) → (∀ x : ℝ, (0 < x ∧ x < Real.exp (-1) → f 0 x < f 0 x + 3 * x * g x)) ∧
    (Real.exp (-1) < x ∧ 0 < x → f 0 x + 3 * x * g x > f 0 x)) := sorry

theorem m_in_terms_of_x0 (a x0 m : ℝ) (ha : a > Real.exp (10 / 3))
  (tangent_line : ∀ y, y - ( -(1 / 3) * x0^3 + (1 / 2) * x0^2 * g a) = 
    (-(x0^2) + x0 * g a) * (x - x0)) :
  m = (2 / 3) * x0^3 - (1 + (1 / 2) * g a) * x0^2 + x0 * g a := sorry

theorem at_least_two_tangents (a m : ℝ) (ha : a > Real.exp (10 / 3))
  (at_least_two : ∃ x0 y, x0 ≠ y ∧ F a x0 = m ∧ F a y = m) :
  m = 4 / 3 := sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l1724_172498


namespace NUMINAMATH_GPT_relationship_not_true_l1724_172405

theorem relationship_not_true (a b : ℕ) :
  (b = a + 5 ∨ b = a + 15 ∨ b = a + 29) → ¬(a = b - 9) :=
by
  sorry

end NUMINAMATH_GPT_relationship_not_true_l1724_172405


namespace NUMINAMATH_GPT_standing_arrangements_l1724_172465

theorem standing_arrangements : ∃ (arrangements : ℕ), arrangements = 2 :=
by
  -- Given that Jia, Yi, Bing, and Ding are four distinct people standing in a row
  -- We need to prove that there are exactly 2 different ways for them to stand such that Jia is not at the far left and Yi is not at the far right
  sorry

end NUMINAMATH_GPT_standing_arrangements_l1724_172465


namespace NUMINAMATH_GPT_remaining_soup_feeds_20_adults_l1724_172470

theorem remaining_soup_feeds_20_adults (cans_of_soup : ℕ) (feed_4_adults : ℕ) (feed_7_children : ℕ) (initial_cans : ℕ) (children_fed : ℕ)
    (h1 : feed_4_adults = 4)
    (h2 : feed_7_children = 7)
    (h3 : initial_cans = 8)
    (h4 : children_fed = 21) : 
    (initial_cans - (children_fed / feed_7_children)) * feed_4_adults = 20 :=
by
  sorry

end NUMINAMATH_GPT_remaining_soup_feeds_20_adults_l1724_172470


namespace NUMINAMATH_GPT_living_room_area_l1724_172448

theorem living_room_area (L W : ℝ) (percent_covered : ℝ) (expected_area : ℝ) 
  (hL : L = 6.5) (hW : W = 12) (hpercent : percent_covered = 0.85) 
  (hexpected_area : expected_area = 91.76) : 
  (L * W / percent_covered = expected_area) :=
by
  sorry  -- The proof is omitted.

end NUMINAMATH_GPT_living_room_area_l1724_172448


namespace NUMINAMATH_GPT_inverse_of_composed_function_l1724_172441

theorem inverse_of_composed_function :
  let f (x : ℝ) := 4 * x + 5
  let g (x : ℝ) := 3 * x - 4
  let k (x : ℝ) := f (g x)
  ∀ y : ℝ, k ( (y + 11) / 12 ) = y :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_composed_function_l1724_172441


namespace NUMINAMATH_GPT_inverse_sum_is_minus_two_l1724_172491

variable (f : ℝ → ℝ)
variable (h_injective : Function.Injective f)
variable (h_surjective : Function.Surjective f)
variable (h_eq : ∀ x : ℝ, f (x + 1) + f (-x - 3) = 2)

theorem inverse_sum_is_minus_two (x : ℝ) : f⁻¹ (2009 - x) + f⁻¹ (x - 2007) = -2 := 
  sorry

end NUMINAMATH_GPT_inverse_sum_is_minus_two_l1724_172491


namespace NUMINAMATH_GPT_price_of_feed_corn_l1724_172487

theorem price_of_feed_corn :
  ∀ (num_sheep : ℕ) (num_cows : ℕ) (grass_per_cow : ℕ) (grass_per_sheep : ℕ)
    (feed_corn_duration_cow : ℕ) (feed_corn_duration_sheep : ℕ)
    (total_grass : ℕ) (total_expenditure : ℕ) (months_in_year : ℕ),
  num_sheep = 8 →
  num_cows = 5 →
  grass_per_cow = 2 →
  grass_per_sheep = 1 →
  feed_corn_duration_cow = 1 →
  feed_corn_duration_sheep = 2 →
  total_grass = 144 →
  total_expenditure = 360 →
  months_in_year = 12 →
  ((total_expenditure : ℝ) / (((num_cows * feed_corn_duration_cow * 4) + (num_sheep * (4 / feed_corn_duration_sheep))) : ℝ)) = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_price_of_feed_corn_l1724_172487


namespace NUMINAMATH_GPT_cyclists_meet_at_start_point_l1724_172459

-- Conditions from the problem
def cyclist1_speed : ℝ := 7 -- speed of the first cyclist in m/s
def cyclist2_speed : ℝ := 8 -- speed of the second cyclist in m/s
def circumference : ℝ := 600 -- circumference of the circular track in meters

-- Relative speed when cyclists move in opposite directions
def relative_speed := cyclist1_speed + cyclist2_speed

-- Prove that they meet at the starting point after 40 seconds
theorem cyclists_meet_at_start_point :
  (circumference / relative_speed) = 40 := by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_cyclists_meet_at_start_point_l1724_172459


namespace NUMINAMATH_GPT_solve_inequality_l1724_172476

theorem solve_inequality (x : ℝ) : 1 + 2 * (x - 1) ≤ 3 → x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1724_172476


namespace NUMINAMATH_GPT_range_of_m_l1724_172471

noncomputable def proposition_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 + m * x + 1 ≥ 0

noncomputable def proposition_q (m : ℝ) : Prop :=
∀ x : ℝ, (8 * x + 4 * (m - 1)) ≥ 0

def conditions (m : ℝ) : Prop :=
(proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m)

theorem range_of_m (m : ℝ) : 
  conditions m → ( -2 ≤ m ∧ m < 1 ) ∨ m > 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_m_l1724_172471


namespace NUMINAMATH_GPT_sum_of_number_and_reverse_l1724_172446

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_reverse_l1724_172446


namespace NUMINAMATH_GPT_taxi_fare_miles_l1724_172443

theorem taxi_fare_miles (total_spent : ℝ) (tip : ℝ) (base_fare : ℝ) (additional_fare_rate : ℝ) (base_mile : ℝ) (additional_mile_unit : ℝ) (x : ℝ) :
  (total_spent = 15) →
  (tip = 3) →
  (base_fare = 3) →
  (additional_fare_rate = 0.25) →
  (base_mile = 0.5) →
  (additional_mile_unit = 0.1) →
  (x = base_mile + (total_spent - tip - base_fare) / (additional_fare_rate / additional_mile_unit)) →
  x = 4.1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_taxi_fare_miles_l1724_172443


namespace NUMINAMATH_GPT_intersecting_to_quadrilateral_l1724_172426

-- Define the geometric solids
inductive GeometricSolid
| cone : GeometricSolid
| sphere : GeometricSolid
| cylinder : GeometricSolid

-- Define a function that checks if intersecting a given solid with a plane can produce a quadrilateral
def can_intersect_to_quadrilateral (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.cone => false
  | GeometricSolid.sphere => false
  | GeometricSolid.cylinder => true

-- State the theorem
theorem intersecting_to_quadrilateral (solid : GeometricSolid) :
  can_intersect_to_quadrilateral solid ↔ solid = GeometricSolid.cylinder :=
sorry

end NUMINAMATH_GPT_intersecting_to_quadrilateral_l1724_172426


namespace NUMINAMATH_GPT_maximum_sum_l1724_172402

theorem maximum_sum (a b c d : ℕ) (h₀ : a < b ∧ b < c ∧ c < d)
  (h₁ : (c + d) + (a + b + c) = 2017) : a + b + c + d ≤ 806 :=
sorry

end NUMINAMATH_GPT_maximum_sum_l1724_172402


namespace NUMINAMATH_GPT_train_pass_time_l1724_172453

noncomputable def train_length : ℕ := 360
noncomputable def platform_length : ℕ := 140
noncomputable def train_speed_kmh : ℕ := 45

noncomputable def convert_speed_to_mps (speed_kmh : ℕ) : ℚ := 
  (speed_kmh * 1000) / 3600

noncomputable def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

noncomputable def time_to_pass (distance : ℕ) (speed_mps : ℚ) : ℚ :=
  distance / speed_mps

theorem train_pass_time 
  (train_len : ℕ) 
  (platform_len : ℕ) 
  (speed_kmh : ℕ) : 
  time_to_pass (total_distance train_len platform_len) (convert_speed_to_mps speed_kmh) = 40 := 
by 
  sorry

end NUMINAMATH_GPT_train_pass_time_l1724_172453


namespace NUMINAMATH_GPT_tetrahedron_volume_le_one_l1724_172485

open Real

noncomputable def volume_tetrahedron (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  let (x0, y0, z0) := A
  let (x1, y1, z1) := B
  let (x2, y2, z2) := C
  let (x3, y3, z3) := D
  abs ((x1 - x0) * ((y2 - y0) * (z3 - z0) - (y3 - y0) * (z2 - z0)) -
       (x2 - x0) * ((y1 - y0) * (z3 - z0) - (y3 - y0) * (z1 - z0)) +
       (x3 - x0) * ((y1 - y0) * (z2 - z0) - (y2 - y0) * (z1 - z0))) / 6

theorem tetrahedron_volume_le_one (A B C D : ℝ × ℝ × ℝ)
  (h1 : dist A B ≤ 2) (h2 : dist A C ≤ 2) (h3 : dist A D ≤ 2)
  (h4 : dist B C ≤ 2) (h5 : dist B D ≤ 2) (h6 : dist C D ≤ 2) :
  volume_tetrahedron A B C D ≤ 1 := by
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_le_one_l1724_172485


namespace NUMINAMATH_GPT_root_exists_in_interval_l1724_172419

def f (x : ℝ) : ℝ := 2 * x + x - 2

theorem root_exists_in_interval :
  (∃ x ∈ (Set.Ioo 0 1), f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_root_exists_in_interval_l1724_172419


namespace NUMINAMATH_GPT_square_completion_l1724_172456

theorem square_completion (a : ℝ) (h : a^2 + 2 * a - 2 = 0) : (a + 1)^2 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_square_completion_l1724_172456


namespace NUMINAMATH_GPT_count_households_in_apartment_l1724_172467

noncomputable def total_households 
  (houses_left : ℕ)
  (houses_right : ℕ)
  (floors_above : ℕ)
  (floors_below : ℕ) 
  (households_per_house : ℕ) : ℕ :=
(houses_left + houses_right) * (floors_above + floors_below) * households_per_house

theorem count_households_in_apartment : 
  ∀ (houses_left houses_right floors_above floors_below households_per_house : ℕ),
  houses_left = 1 →
  houses_right = 6 →
  floors_above = 1 →
  floors_below = 3 →
  households_per_house = 3 →
  total_households houses_left houses_right floors_above floors_below households_per_house = 105 :=
by
  intros houses_left houses_right floors_above floors_below households_per_house hl hr fa fb hh
  rw [hl, hr, fa, fb, hh]
  unfold total_households
  norm_num
  sorry

end NUMINAMATH_GPT_count_households_in_apartment_l1724_172467


namespace NUMINAMATH_GPT_solve_for_x_l1724_172497

theorem solve_for_x (x : ℝ) : 9 * x^2 - 4 = 0 → (x = 2/3 ∨ x = -2/3) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1724_172497


namespace NUMINAMATH_GPT_meiosis_fertilization_stability_l1724_172437

def maintains_chromosome_stability (x : String) : Prop :=
  x = "Meiosis and Fertilization"

theorem meiosis_fertilization_stability :
  maintains_chromosome_stability "Meiosis and Fertilization" :=
by
  sorry

end NUMINAMATH_GPT_meiosis_fertilization_stability_l1724_172437


namespace NUMINAMATH_GPT_right_triangle_legs_l1724_172481

theorem right_triangle_legs (a b c : ℝ) 
  (h : ℝ) 
  (h_h : h = 12) 
  (h_perimeter : a + b + c = 60) 
  (h1 : a^2 + b^2 = c^2) 
  (h_altitude : h = a * b / c) :
  (a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_legs_l1724_172481


namespace NUMINAMATH_GPT_distinct_x_intercepts_l1724_172489

theorem distinct_x_intercepts : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, (x - 5) * (x ^ 2 + 3 * x + 2) = 0) ∧ s.card = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_distinct_x_intercepts_l1724_172489


namespace NUMINAMATH_GPT_max_knights_l1724_172457

/-- 
On an island with knights who always tell the truth and liars who always lie,
100 islanders seated around a round table where:
  - 50 of them say "both my neighbors are liars,"
  - The other 50 say "among my neighbors, there is exactly one liar."
Prove that the maximum number of knights at the table is 67.
-/
theorem max_knights (K L : ℕ) (h1 : K + L = 100) (h2 : ∃ k, k ≤ 25 ∧ K = 2 * k + (100 - 3 * k) / 2) : K = 67 :=
sorry

end NUMINAMATH_GPT_max_knights_l1724_172457


namespace NUMINAMATH_GPT_measure_4_minutes_with_hourglasses_l1724_172432

/-- Prove that it is possible to measure exactly 4 minutes using hourglasses of 9 minutes and 7 minutes and the minimum total time required is 18 minutes -/
theorem measure_4_minutes_with_hourglasses : 
  ∃ (a b : ℕ), (9 * a - 7 * b = 4) ∧ (a + b) * 1 ≤ 2 ∧ (a * 9 ≤ 18 ∧ b * 7 <= 18) :=
by {
  sorry
}

end NUMINAMATH_GPT_measure_4_minutes_with_hourglasses_l1724_172432


namespace NUMINAMATH_GPT_era_slices_burger_l1724_172436

theorem era_slices_burger (slices_per_burger : ℕ) (h : 5 * slices_per_burger = 10) : slices_per_burger = 2 :=
by 
  sorry

end NUMINAMATH_GPT_era_slices_burger_l1724_172436


namespace NUMINAMATH_GPT_deans_height_l1724_172447

theorem deans_height
  (D : ℕ) 
  (h1 : 10 * D = D + 81) : 
  D = 9 := sorry

end NUMINAMATH_GPT_deans_height_l1724_172447


namespace NUMINAMATH_GPT_division_of_product_l1724_172493

theorem division_of_product :
  (1.6 * 0.5) / 1 = 0.8 :=
sorry

end NUMINAMATH_GPT_division_of_product_l1724_172493


namespace NUMINAMATH_GPT_original_number_is_two_thirds_l1724_172482

theorem original_number_is_two_thirds (x : ℚ) (h : 1 + (1 / x) = 5 / 2) : x = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_two_thirds_l1724_172482


namespace NUMINAMATH_GPT_possible_values_of_x_l1724_172480

theorem possible_values_of_x (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) 
    (h1 : x + 1 / z = 15) (h2 : z + 1 / x = 9 / 20) :
    x = (15 + 5 * Real.sqrt 11) / 2 ∨ x = (15 - 5 * Real.sqrt 11) / 2 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_x_l1724_172480


namespace NUMINAMATH_GPT_lassis_from_mangoes_l1724_172490

theorem lassis_from_mangoes (L M : ℕ) (h : 2 * L = 11 * M) : 12 * L = 66 :=
by sorry

end NUMINAMATH_GPT_lassis_from_mangoes_l1724_172490


namespace NUMINAMATH_GPT_f_1997_leq_666_l1724_172400

noncomputable def f : ℕ+ → ℕ := sorry

axiom f_mn_inequality : ∀ (m n : ℕ+), f (m + n) ≥ f m + f n
axiom f_two : f 2 = 0
axiom f_three_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

theorem f_1997_leq_666 : f 1997 ≤ 666 := sorry

end NUMINAMATH_GPT_f_1997_leq_666_l1724_172400


namespace NUMINAMATH_GPT_f_sub_f_neg_l1724_172439

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + 7 * x

-- State the theorem
theorem f_sub_f_neg : f 3 - f (-3) = 582 :=
by
  -- Definitions and calculations for the proof
  -- (You can complete this part in later proof development)
  sorry

end NUMINAMATH_GPT_f_sub_f_neg_l1724_172439


namespace NUMINAMATH_GPT_value_of_x_l1724_172408

theorem value_of_x (x : ℕ) (h : x + (10 * x + x) = 12) : x = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l1724_172408


namespace NUMINAMATH_GPT_henry_money_l1724_172450

-- Define the conditions
def initial : ℕ := 11
def birthday : ℕ := 18
def spent : ℕ := 10

-- Define the final amount
def final_amount : ℕ := initial + birthday - spent

-- State the theorem
theorem henry_money : final_amount = 19 := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_henry_money_l1724_172450


namespace NUMINAMATH_GPT_final_bicycle_price_is_225_l1724_172473

noncomputable def final_selling_price (cp_A : ℝ) (profit_A : ℝ) (profit_B : ℝ) : ℝ :=
  let sp_B := cp_A * (1 + profit_A / 100)
  let sp_C := sp_B * (1 + profit_B / 100)
  sp_C

theorem final_bicycle_price_is_225 :
  final_selling_price 114.94 35 45 = 224.99505 :=
by
  sorry

end NUMINAMATH_GPT_final_bicycle_price_is_225_l1724_172473


namespace NUMINAMATH_GPT_fibonacci_polynomial_property_l1724_172435

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Define the polynomial P(x) of degree 990
noncomputable def P : ℕ → ℕ :=
  sorry  -- To be defined as a polynomial with specified properties

-- Statement of the problem (theorem)
theorem fibonacci_polynomial_property (P : ℕ → ℕ) (hP : ∀ k, 992 ≤ k → k ≤ 1982 → P k = fibonacci k) :
  P 1983 = fibonacci 1983 - 1 :=
sorry  -- Proof omitted

end NUMINAMATH_GPT_fibonacci_polynomial_property_l1724_172435


namespace NUMINAMATH_GPT_max_ratio_of_three_digit_to_sum_l1724_172422

theorem max_ratio_of_three_digit_to_sum (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9)
  (hc : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c) / (a + b + c) ≤ 100 :=
by sorry

end NUMINAMATH_GPT_max_ratio_of_three_digit_to_sum_l1724_172422


namespace NUMINAMATH_GPT_boy_age_is_10_l1724_172418

-- Define the boy's current age as a variable
def boy_current_age := 10

-- Define a condition based on the boy's statement
def boy_statement_condition (x : ℕ) : Prop :=
  x = 2 * (x - 5)

-- The main theorem stating equivalence of the boy's current age to 10 given the condition
theorem boy_age_is_10 (x : ℕ) (h : boy_statement_condition x) : x = boy_current_age := by
  sorry

end NUMINAMATH_GPT_boy_age_is_10_l1724_172418


namespace NUMINAMATH_GPT_case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l1724_172428

theorem case_one_ellipses_foci_xaxis :
  ∀ (a : ℝ) (e : ℝ), a = 6 ∧ e = 2 / 3 → (∃ (b : ℝ), (b^2 = (a^2 - (e * a)^2) ∧ (a > 0) → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)) ∨ (y^2 / a^2 + x^2 / b^2 = 1)))) :=
by
  sorry

theorem case_two_ellipses_foci_exact :
  ∀ (F1 F2 : ℝ × ℝ), F1 = (-4,0) ∧ F2 = (4,0) ∧ ∀ P : ℝ × ℝ, ((dist P F1) + (dist P F2) = 10) →
  ∃ (a : ℝ) (b : ℝ), a = 5 ∧ b^2 = a^2 - 4^2 → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1))) :=
by
  sorry

end NUMINAMATH_GPT_case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l1724_172428


namespace NUMINAMATH_GPT_sandy_savings_percentage_l1724_172431

theorem sandy_savings_percentage
  (S : ℝ) -- Sandy's salary last year
  (H1 : 0.10 * S = saved_last_year) -- Last year, Sandy saved 10% of her salary.
  (H2 : 1.10 * S = salary_this_year) -- This year, Sandy made 10% more than last year.
  (H3 : 0.15 * salary_this_year = saved_this_year) -- This year, Sandy saved 15% of her salary.
  : (saved_this_year / saved_last_year) * 100 = 165 := 
by 
  sorry

end NUMINAMATH_GPT_sandy_savings_percentage_l1724_172431


namespace NUMINAMATH_GPT_statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l1724_172440

-- Statement A
theorem statement_A_incorrect (a b c d : ℝ) (ha : a < b) (hc : c < d) : ¬ (a * c < b * d) := by
  sorry

-- Statement B
theorem statement_B_correct (a b : ℝ) (ha : -2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) : -1 < a / b ∧ a / b < 3 := by
  sorry

-- Statement C
theorem statement_C_incorrect (m : ℝ) : ¬ (∀ x > 0, x / 2 + 2 / x ≥ m) ∧ (m ≤ 1) := by
  sorry

-- Statement D
theorem statement_D_incorrect : ∃ x : ℝ, (x^2 + 2) + 1 / (x^2 + 2) ≠ 2 := by
  sorry

end NUMINAMATH_GPT_statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l1724_172440


namespace NUMINAMATH_GPT_paul_and_lisa_total_dollars_l1724_172484

def total_dollars_of_paul_and_lisa (paul_dol : ℚ) (lisa_dol : ℚ) : ℚ :=
  paul_dol + lisa_dol

theorem paul_and_lisa_total_dollars (paul_dol := (5 / 6 : ℚ)) (lisa_dol := (2 / 5 : ℚ)) :
  total_dollars_of_paul_and_lisa paul_dol lisa_dol = (123 / 100 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_paul_and_lisa_total_dollars_l1724_172484


namespace NUMINAMATH_GPT_carolyn_practice_time_l1724_172414

theorem carolyn_practice_time :
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  monthly_minutes_total = 1920 :=
by
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  sorry

end NUMINAMATH_GPT_carolyn_practice_time_l1724_172414


namespace NUMINAMATH_GPT_negation_exists_l1724_172478

theorem negation_exists:
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_l1724_172478


namespace NUMINAMATH_GPT_routeY_is_quicker_l1724_172499

noncomputable def timeRouteX : ℝ := 
  8 / 40 

noncomputable def timeRouteY1 : ℝ := 
  6.5 / 50 

noncomputable def timeRouteY2 : ℝ := 
  0.5 / 10

noncomputable def timeRouteY : ℝ := 
  timeRouteY1 + timeRouteY2  

noncomputable def timeDifference : ℝ := 
  (timeRouteX - timeRouteY) * 60 

theorem routeY_is_quicker : 
  timeDifference = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_routeY_is_quicker_l1724_172499


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1724_172462

theorem arithmetic_sequence_sum :
  ∃ x y d : ℕ,
    d = 6
    ∧ x = 3 + d * (3 - 1)
    ∧ y = x + d
    ∧ y + d = 39
    ∧ x + y = 60 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1724_172462


namespace NUMINAMATH_GPT_matrix_power_2023_correct_l1724_172464

noncomputable def matrix_power_2023 : Matrix (Fin 2) (Fin 2) ℤ :=
  let A := !![1, 0; 2, 1]  -- Define the matrix
  A^2023

theorem matrix_power_2023_correct :
  matrix_power_2023 = !![1, 0; 4046, 1] := by
  sorry

end NUMINAMATH_GPT_matrix_power_2023_correct_l1724_172464
