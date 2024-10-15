import Mathlib

namespace NUMINAMATH_GPT_inequality_holds_l1892_189276

theorem inequality_holds (a b : ℝ) : 
  a^2 + a * b + b^2 ≥ 3 * (a + b - 1) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1892_189276


namespace NUMINAMATH_GPT_largest_root_polynomial_intersection_l1892_189266

/-
Given a polynomial P(x) = x^6 - 15x^5 + 74x^4 - 130x^3 + a * x^2 + b * x
and a line L(x) = c * x - 24,
such that P(x) stays above L(x) except at three distinct values of x where they intersect,
and one of the intersections is a root of triple multiplicity.
Prove that the largest value of x for which P(x) = L(x) is 6.
-/
theorem largest_root_polynomial_intersection (a b c : ℝ) (P L : ℝ → ℝ) (x : ℝ) :
  P x = x^6 - 15*x^5 + 74*x^4 - 130*x^3 + a*x^2 + b*x →
  L x = c*x - 24 →
  (∀ x, P x ≥ L x) ∨ (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ P x1 = L x1 ∧ P x2 = L x2 ∧ P x3 = L x3 ∧
  (∃ x0 : ℝ, x1 = x0 ∧ x2 = x0 ∧ x3 = x0 ∧ ∃ k : ℕ, k = 3)) →
  x = 6 :=
sorry

end NUMINAMATH_GPT_largest_root_polynomial_intersection_l1892_189266


namespace NUMINAMATH_GPT_circle_condition_l1892_189241

theorem circle_condition (k : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 4 * k + 1 = 0) → (k < 1) :=
by
  sorry

end NUMINAMATH_GPT_circle_condition_l1892_189241


namespace NUMINAMATH_GPT_hazel_walked_distance_l1892_189254

theorem hazel_walked_distance
  (first_hour_distance : ℕ)
  (second_hour_distance : ℕ)
  (h1 : first_hour_distance = 2)
  (h2 : second_hour_distance = 2 * first_hour_distance) :
  (first_hour_distance + second_hour_distance = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_hazel_walked_distance_l1892_189254


namespace NUMINAMATH_GPT_prime_factorization_sum_l1892_189234

theorem prime_factorization_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : 13 * x^7 = 17 * y^11) : 
  a * e + b * f = 18 :=
by
  -- Let a and b be prime factors of x
  let a : ℕ := 17 -- prime factor found in the solution
  let e : ℕ := 1 -- exponent found for 17
  let b : ℕ := 0 -- no second prime factor
  let f : ℕ := 0 -- corresponding exponent

  sorry

end NUMINAMATH_GPT_prime_factorization_sum_l1892_189234


namespace NUMINAMATH_GPT_complement_of_intersection_l1892_189260

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {1, 2, 3}
def intersection : Set ℕ := M ∩ N
def complement : Set ℕ := U \ intersection

theorem complement_of_intersection (U M N : Set ℕ) :
  U = {0, 1, 2, 3} →
  M = {0, 1, 2} →
  N = {1, 2, 3} →
  (U \ (M ∩ N)) = {0, 3} := by
  intro hU hM hN
  simp [hU, hM, hN]
  sorry

end NUMINAMATH_GPT_complement_of_intersection_l1892_189260


namespace NUMINAMATH_GPT_shapes_fit_exactly_l1892_189267

-- Conditions: Shapes are drawn on a piece of paper and folded along a central bold line
def shapes_drawn_on_paper := true
def paper_folded_along_central_line := true

-- Define the main proof problem
theorem shapes_fit_exactly : shapes_drawn_on_paper ∧ paper_folded_along_central_line → 
  number_of_shapes_fitting_exactly_on_top = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_shapes_fit_exactly_l1892_189267


namespace NUMINAMATH_GPT_abs_gt_two_l1892_189240

theorem abs_gt_two (x : ℝ) : |x| > 2 → x > 2 ∨ x < -2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_abs_gt_two_l1892_189240


namespace NUMINAMATH_GPT_jennifer_money_left_l1892_189246

variable (initial_amount : ℝ) (spent_sandwich_rate : ℝ) (spent_museum_rate : ℝ) (spent_book_rate : ℝ)

def money_left := initial_amount - (spent_sandwich_rate * initial_amount + spent_museum_rate * initial_amount + spent_book_rate * initial_amount)

theorem jennifer_money_left (h_initial : initial_amount = 150)
  (h_sandwich_rate : spent_sandwich_rate = 1/5)
  (h_museum_rate : spent_museum_rate = 1/6)
  (h_book_rate : spent_book_rate = 1/2) :
  money_left initial_amount spent_sandwich_rate spent_museum_rate spent_book_rate = 20 :=
by
  sorry

end NUMINAMATH_GPT_jennifer_money_left_l1892_189246


namespace NUMINAMATH_GPT_proposition_holds_for_all_positive_odd_numbers_l1892_189259

theorem proposition_holds_for_all_positive_odd_numbers
  (P : ℕ → Prop)
  (h1 : P 1)
  (h2 : ∀ k, k ≥ 1 → P k → P (k + 2)) :
  ∀ n, n % 2 = 1 → n ≥ 1 → P n :=
by
  sorry

end NUMINAMATH_GPT_proposition_holds_for_all_positive_odd_numbers_l1892_189259


namespace NUMINAMATH_GPT_kevin_eggs_l1892_189252

theorem kevin_eggs : 
  ∀ (bonnie george cheryl kevin : ℕ),
  bonnie = 13 → 
  george = 9 → 
  cheryl = 56 → 
  cheryl = bonnie + george + kevin + 29 →
  kevin = 5 :=
by
  intros bonnie george cheryl kevin h_bonnie h_george h_cheryl h_eqn
  subst h_bonnie
  subst h_george
  subst h_cheryl
  simp at h_eqn
  sorry

end NUMINAMATH_GPT_kevin_eggs_l1892_189252


namespace NUMINAMATH_GPT_polygon_triangle_division_l1892_189220

theorem polygon_triangle_division (n k : ℕ) (h : k * 3 = n * 3 - 6) : k ≥ n - 2 := sorry

end NUMINAMATH_GPT_polygon_triangle_division_l1892_189220


namespace NUMINAMATH_GPT_percentage_of_total_spent_on_other_items_l1892_189257

-- Definitions for the given problem conditions

def total_amount (a : ℝ) := a
def spent_on_clothing (a : ℝ) := 0.50 * a
def spent_on_food (a : ℝ) := 0.10 * a
def spent_on_other_items (a x_clothing x_food : ℝ) := a - x_clothing - x_food
def tax_on_clothing (x_clothing : ℝ) := 0.04 * x_clothing
def tax_on_food := 0
def tax_on_other_items (x_other_items : ℝ) := 0.08 * x_other_items
def total_tax (a : ℝ) := 0.052 * a

-- The theorem we need to prove
theorem percentage_of_total_spent_on_other_items (a x_clothing x_food x_other_items : ℝ)
    (h1 : x_clothing = spent_on_clothing a)
    (h2 : x_food = spent_on_food a)
    (h3 : x_other_items = spent_on_other_items a x_clothing x_food)
    (h4 : tax_on_clothing x_clothing + tax_on_food + tax_on_other_items x_other_items = total_tax a) :
    0.40 * a = x_other_items :=
sorry

end NUMINAMATH_GPT_percentage_of_total_spent_on_other_items_l1892_189257


namespace NUMINAMATH_GPT_sum_of_integers_l1892_189258

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1892_189258


namespace NUMINAMATH_GPT_solve_system_eq_l1892_189200

theorem solve_system_eq (x y z t : ℕ) : 
  ((x^2 + t^2) * (z^2 + y^2) = 50) ↔
    (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
    (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
    (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
    (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
by 
  sorry

end NUMINAMATH_GPT_solve_system_eq_l1892_189200


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1892_189250

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - (y^2 / 9) = 1) → (y = 3 * x ∨ y = -3 * x) :=
by
  -- conditions and theorem to prove
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1892_189250


namespace NUMINAMATH_GPT_linear_function_m_l1892_189223

theorem linear_function_m (m : ℤ) (h₁ : |m| = 1) (h₂ : m + 1 ≠ 0) : m = 1 := by
  sorry

end NUMINAMATH_GPT_linear_function_m_l1892_189223


namespace NUMINAMATH_GPT_barbara_weekly_allowance_l1892_189291

theorem barbara_weekly_allowance (W C S : ℕ) (H : W = 100) (A : S = 20) (N : C = 16) :
  (W - S) / C = 5 :=
by
  -- definitions to match conditions
  have W_def : W = 100 := H
  have S_def : S = 20 := A
  have C_def : C = 16 := N
  sorry

end NUMINAMATH_GPT_barbara_weekly_allowance_l1892_189291


namespace NUMINAMATH_GPT_expand_product_l1892_189274

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by 
  sorry

end NUMINAMATH_GPT_expand_product_l1892_189274


namespace NUMINAMATH_GPT_min_a_value_l1892_189217

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end NUMINAMATH_GPT_min_a_value_l1892_189217


namespace NUMINAMATH_GPT_integer_combination_zero_l1892_189296

theorem integer_combination_zero (a b c : ℤ) (h : a * Real.sqrt 2 + b * Real.sqrt 3 + c = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end NUMINAMATH_GPT_integer_combination_zero_l1892_189296


namespace NUMINAMATH_GPT_arithmetic_seq_50th_term_l1892_189205

def arithmetic_seq_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem arithmetic_seq_50th_term : 
  arithmetic_seq_nth_term 3 7 50 = 346 :=
by
  -- Intentionally left as sorry
  sorry

end NUMINAMATH_GPT_arithmetic_seq_50th_term_l1892_189205


namespace NUMINAMATH_GPT_find_number_of_tails_l1892_189273

-- Definitions based on conditions
variables (T H : ℕ)
axiom total_coins : T + H = 1250
axiom heads_more_than_tails : H = T + 124

-- The goal is to prove T = 563
theorem find_number_of_tails : T = 563 :=
sorry

end NUMINAMATH_GPT_find_number_of_tails_l1892_189273


namespace NUMINAMATH_GPT_harriet_forward_speed_proof_l1892_189248

def harriet_forward_time : ℝ := 3 -- forward time in hours
def harriet_return_speed : ℝ := 150 -- return speed in km/h
def harriet_total_time : ℝ := 5 -- total trip time in hours

noncomputable def harriet_forward_speed : ℝ :=
  let distance := harriet_return_speed * (harriet_total_time - harriet_forward_time)
  distance / harriet_forward_time

theorem harriet_forward_speed_proof : harriet_forward_speed = 100 := by
  sorry

end NUMINAMATH_GPT_harriet_forward_speed_proof_l1892_189248


namespace NUMINAMATH_GPT_differentiable_increasing_necessary_but_not_sufficient_l1892_189237

variable {f : ℝ → ℝ}

theorem differentiable_increasing_necessary_but_not_sufficient (h_diff : ∀ x : ℝ, DifferentiableAt ℝ f x) :
  (∀ x : ℝ, 0 < deriv f x) → ∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) ∧ ¬ (∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) → ∀ x : ℝ, 0 < deriv f x) := 
sorry

end NUMINAMATH_GPT_differentiable_increasing_necessary_but_not_sufficient_l1892_189237


namespace NUMINAMATH_GPT_problem1_problem2_l1892_189207

theorem problem1 : ( (2 / 3 - 1 / 4 - 5 / 6) * 12 = -5 ) :=
by sorry

theorem problem2 : ( (-3)^2 * 2 + 4 * (-3) - 28 / (7 / 4) = -10 ) :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1892_189207


namespace NUMINAMATH_GPT_triangle_ratio_l1892_189227

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ)
  (hA : A = 2 * Real.pi / 3)
  (h_a : a = Real.sqrt 3 * c)
  (h_angle_sum : A + B + C = Real.pi)
  (h_law_of_sines : a / Real.sin A = c / Real.sin C) :
  b / c = 1 :=
sorry

end NUMINAMATH_GPT_triangle_ratio_l1892_189227


namespace NUMINAMATH_GPT_exist_ints_a_b_l1892_189212

theorem exist_ints_a_b (n : ℕ) : (∃ a b : ℤ, (n : ℤ) + a^2 = b^2) ↔ ¬ n % 4 = 2 := 
by
  sorry

end NUMINAMATH_GPT_exist_ints_a_b_l1892_189212


namespace NUMINAMATH_GPT_five_coins_not_155_l1892_189272

def coin_values : List ℕ := [5, 25, 50]

def can_sum_to (n : ℕ) (count : ℕ) : Prop :=
  ∃ (a b c : ℕ), a + b + c = count ∧ a * 5 + b * 25 + c * 50 = n

theorem five_coins_not_155 : ¬ can_sum_to 155 5 :=
  sorry

end NUMINAMATH_GPT_five_coins_not_155_l1892_189272


namespace NUMINAMATH_GPT_alphabet_letter_count_l1892_189255

def sequence_count : Nat :=
  let total_sequences := 2^7
  let sequences_per_letter := 1 + 7 -- 1 correct sequence + 7 single-bit alterations
  total_sequences / sequences_per_letter

theorem alphabet_letter_count : sequence_count = 16 :=
  by
    -- Proof placeholder
    sorry

end NUMINAMATH_GPT_alphabet_letter_count_l1892_189255


namespace NUMINAMATH_GPT_car_initial_speed_l1892_189282

theorem car_initial_speed (s t : ℝ) (h₁ : t = 15 * s^2) (h₂ : t = 3) :
  s = (Real.sqrt 2) / 5 :=
by
  sorry

end NUMINAMATH_GPT_car_initial_speed_l1892_189282


namespace NUMINAMATH_GPT_least_possible_b_l1892_189231

open Nat

/-- 
  Given conditions:
  a and b are consecutive Fibonacci numbers with a > b,
  and their sum is 100 degrees.
  We need to prove that the least possible value of b is 21 degrees.
-/
theorem least_possible_b (a b : ℕ) (h1 : fib a = fib (b + 1))
  (h2 : a > b) (h3 : a + b = 100) : b = 21 :=
sorry

end NUMINAMATH_GPT_least_possible_b_l1892_189231


namespace NUMINAMATH_GPT_original_gain_percentage_is_5_l1892_189279

def costPrice : ℝ := 200
def newCostPrice : ℝ := costPrice * 0.95
def desiredProfitRatio : ℝ := 0.10
def newSellingPrice : ℝ := newCostPrice * (1 + desiredProfitRatio)
def originalSellingPrice : ℝ := newSellingPrice + 1

theorem original_gain_percentage_is_5 :
  ((originalSellingPrice - costPrice) / costPrice) * 100 = 5 :=
by 
  sorry

end NUMINAMATH_GPT_original_gain_percentage_is_5_l1892_189279


namespace NUMINAMATH_GPT_number_of_puppies_with_4_spots_is_3_l1892_189206

noncomputable def total_puppies : Nat := 10
noncomputable def puppies_with_5_spots : Nat := 6
noncomputable def puppies_with_2_spots : Nat := 1
noncomputable def puppies_with_4_spots : Nat := total_puppies - puppies_with_5_spots - puppies_with_2_spots

theorem number_of_puppies_with_4_spots_is_3 :
  puppies_with_4_spots = 3 := 
sorry

end NUMINAMATH_GPT_number_of_puppies_with_4_spots_is_3_l1892_189206


namespace NUMINAMATH_GPT_first_year_with_sum_15_l1892_189232

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10)

theorem first_year_with_sum_15 : ∃ y > 2100, sum_of_digits y = 15 :=
  sorry

end NUMINAMATH_GPT_first_year_with_sum_15_l1892_189232


namespace NUMINAMATH_GPT_find_Y_l1892_189253

theorem find_Y :
  ∃ Y : ℤ, (19 + Y / 151) * 151 = 2912 ∧ Y = 43 :=
by
  use 43
  sorry

end NUMINAMATH_GPT_find_Y_l1892_189253


namespace NUMINAMATH_GPT_dog_catches_fox_at_distance_l1892_189219

def initial_distance : ℝ := 30
def dog_leap_distance : ℝ := 2
def fox_leap_distance : ℝ := 1
def dog_leaps_per_time_unit : ℝ := 2
def fox_leaps_per_time_unit : ℝ := 3

noncomputable def dog_speed : ℝ := dog_leaps_per_time_unit * dog_leap_distance
noncomputable def fox_speed : ℝ := fox_leaps_per_time_unit * fox_leap_distance
noncomputable def relative_speed : ℝ := dog_speed - fox_speed
noncomputable def time_to_catch := initial_distance / relative_speed
noncomputable def distance_dog_runs := time_to_catch * dog_speed

theorem dog_catches_fox_at_distance :
  distance_dog_runs = 120 :=
  by sorry

end NUMINAMATH_GPT_dog_catches_fox_at_distance_l1892_189219


namespace NUMINAMATH_GPT_dice_sum_probability_l1892_189284

theorem dice_sum_probability
  (a b c d : ℕ)
  (cond1 : 1 ≤ a ∧ a ≤ 6)
  (cond2 : 1 ≤ b ∧ b ≤ 6)
  (cond3 : 1 ≤ c ∧ c ≤ 6)
  (cond4 : 1 ≤ d ∧ d ≤ 6)
  (sum_cond : a + b + c + d = 5) :
  (∃ p, p = 1 / 324) :=
sorry

end NUMINAMATH_GPT_dice_sum_probability_l1892_189284


namespace NUMINAMATH_GPT_ladybugs_with_spots_l1892_189285

theorem ladybugs_with_spots (total_ladybugs : ℕ) (ladybugs_without_spots : ℕ) : total_ladybugs = 67082 ∧ ladybugs_without_spots = 54912 → total_ladybugs - ladybugs_without_spots = 12170 := by
  sorry

end NUMINAMATH_GPT_ladybugs_with_spots_l1892_189285


namespace NUMINAMATH_GPT_remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l1892_189249

-- Definitions of angle types
def obtuse_angle (θ : ℝ) := θ > 90 ∧ θ < 180
def right_angle (θ : ℝ) := θ = 90
def acute_angle (θ : ℝ) := θ > 0 ∧ θ < 90
def straight_angle (θ : ℝ) := θ = 180

-- Proposition 1: Remaining angle when an obtuse angle is cut by a right angle is acute
theorem remaining_angle_obtuse_cut_by_right_is_acute (θ : ℝ) (φ : ℝ) 
    (h1 : obtuse_angle θ) (h2 : right_angle φ) : acute_angle (θ - φ) :=
  sorry

-- Proposition 2: Remaining angle when a straight angle is cut by an acute angle is obtuse
theorem remaining_angle_straight_cut_by_acute_is_obtuse (α : ℝ) (β : ℝ) 
    (h1 : straight_angle α) (h2 : acute_angle β) : obtuse_angle (α - β) :=
  sorry

end NUMINAMATH_GPT_remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l1892_189249


namespace NUMINAMATH_GPT_no_yarn_earnings_l1892_189290

noncomputable def yarn_cost : Prop :=
  let monday_yards := 20
  let tuesday_yards := 2 * monday_yards
  let wednesday_yards := (1 / 4) * tuesday_yards
  let total_yards := monday_yards + tuesday_yards + wednesday_yards
  let fabric_cost_per_yard := 2
  let total_fabric_earnings := total_yards * fabric_cost_per_yard
  let total_earnings := 140
  total_fabric_earnings = total_earnings

theorem no_yarn_earnings:
  yarn_cost :=
sorry

end NUMINAMATH_GPT_no_yarn_earnings_l1892_189290


namespace NUMINAMATH_GPT_solid_has_identical_views_is_sphere_or_cube_l1892_189265

-- Define the conditions for orthographic projections being identical
def identical_views_in_orthographic_projections (solid : Type) : Prop :=
  sorry -- Assume the logic for checking identical orthographic projections is defined

-- Define the types for sphere and cube
structure Sphere : Type := 
  (radius : ℝ)

structure Cube : Type := 
  (side_length : ℝ)

-- The main statement to prove
theorem solid_has_identical_views_is_sphere_or_cube (solid : Type) 
  (h : identical_views_in_orthographic_projections solid) : 
  solid = Sphere ∨ solid = Cube :=
by 
  sorry -- The detailed proof is omitted

end NUMINAMATH_GPT_solid_has_identical_views_is_sphere_or_cube_l1892_189265


namespace NUMINAMATH_GPT_tan_double_angle_sum_l1892_189288

theorem tan_double_angle_sum (α : ℝ) (h : Real.tan α = 3 / 2) :
  Real.tan (2 * α + Real.pi / 4) = -7 / 17 := 
sorry

end NUMINAMATH_GPT_tan_double_angle_sum_l1892_189288


namespace NUMINAMATH_GPT_inequality_problem_l1892_189293

theorem inequality_problem (a b c : ℝ) (h : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_problem_l1892_189293


namespace NUMINAMATH_GPT_general_formula_sum_of_b_l1892_189215

variable {a : ℕ → ℕ} (b : ℕ → ℕ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n+2) = q * a (n+1)

def initial_conditions (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 = 9 ∧ a 2 + a 3 = 18

theorem general_formula (q : ℕ) (h1 : is_geometric_sequence a q) (h2 : initial_conditions a) :
  a n = 3 * 2^(n - 1) :=
sorry

def b_n (a : ℕ → ℕ) (n : ℕ) : ℕ := a n + 2 * n

def sum_b (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem sum_of_b (h1 : ∀ m : ℕ, b m = a m + 2 * m) (h2 : initial_conditions a) :
  sum_b b n = 3 * 2^n + n * (n + 1) - 3 :=
sorry

end NUMINAMATH_GPT_general_formula_sum_of_b_l1892_189215


namespace NUMINAMATH_GPT_possible_values_of_a_l1892_189275

theorem possible_values_of_a :
  ∃ (a b c : ℤ), ∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c) → a = 3 ∨ a = 7 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l1892_189275


namespace NUMINAMATH_GPT_complement_union_and_complement_intersect_l1892_189245

-- Definitions of sets according to the problem conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

-- The correct answers derived in the solution
def complement_union_A_B : Set ℝ := { x | x ≤ 2 ∨ 10 ≤ x }
def complement_A_intersect_B : Set ℝ := { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) }

-- Statement of the mathematically equivalent proof problem
theorem complement_union_and_complement_intersect:
  (Set.compl (A ∪ B) = complement_union_A_B) ∧ 
  ((Set.compl A) ∩ B = complement_A_intersect_B) :=
  by 
    sorry

end NUMINAMATH_GPT_complement_union_and_complement_intersect_l1892_189245


namespace NUMINAMATH_GPT_intersection_point_l1892_189271

variable (x y : ℚ)

theorem intersection_point :
  (8 * x - 5 * y = 40) ∧ (10 * x + 2 * y = 14) → 
  (x = 25 / 11) ∧ (y = 48 / 11) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l1892_189271


namespace NUMINAMATH_GPT_circumcircle_diameter_l1892_189262

-- Given that the perimeter of triangle ABC is equal to 3 times the sum of the sines of its angles
-- and the Law of Sines holds for this triangle, we need to prove the diameter of the circumcircle is 3.
theorem circumcircle_diameter (a b c : ℝ) (A B C : ℝ) (R : ℝ)
  (h_perimeter : a + b + c = 3 * (Real.sin A + Real.sin B + Real.sin C))
  (h_law_of_sines : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R) :
  2 * R = 3 := 
by
  sorry

end NUMINAMATH_GPT_circumcircle_diameter_l1892_189262


namespace NUMINAMATH_GPT_simplify_expression_l1892_189270

theorem simplify_expression : 4 * (12 / 9) * (36 / -45) = -12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1892_189270


namespace NUMINAMATH_GPT_percentage_milk_in_B_l1892_189226

theorem percentage_milk_in_B :
  ∀ (A B C : ℕ),
  A = 1200 → B + C = A → B + 150 = C - 150 →
  (B:ℝ) / (A:ℝ) * 100 = 37.5 :=
by
  intros A B C hA hBC hE
  sorry

end NUMINAMATH_GPT_percentage_milk_in_B_l1892_189226


namespace NUMINAMATH_GPT_not_p_or_not_q_implies_p_and_q_and_p_or_q_l1892_189221

variable (p q : Prop)

theorem not_p_or_not_q_implies_p_and_q_and_p_or_q (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
sorry

end NUMINAMATH_GPT_not_p_or_not_q_implies_p_and_q_and_p_or_q_l1892_189221


namespace NUMINAMATH_GPT_find_K_l1892_189289

theorem find_K 
  (Z K : ℤ) 
  (hZ_range : 1000 < Z ∧ Z < 2000)
  (hZ_eq : Z = K^4)
  (hK_pos : K > 0) :
  K = 6 :=
by {
  sorry -- Proof to be filled in
}

end NUMINAMATH_GPT_find_K_l1892_189289


namespace NUMINAMATH_GPT_number_of_possible_x_values_l1892_189280
noncomputable def triangle_sides_possible_values (x : ℕ) : Prop :=
  27 < x ∧ x < 63

theorem number_of_possible_x_values : 
  ∃ n, n = (62 - 28 + 1) ∧ ( ∀ x : ℕ, triangle_sides_possible_values x ↔ 28 ≤ x ∧ x ≤ 62) :=
sorry

end NUMINAMATH_GPT_number_of_possible_x_values_l1892_189280


namespace NUMINAMATH_GPT_equation_solutions_l1892_189235

theorem equation_solutions
  (a : ℝ) :
  (∃ x : ℝ, (1 < a ∧ a < 2) ∧ (x = (1 - a) / a ∨ x = -1)) ∨
  (a = 2 ∧ (∃ x : ℝ, x = -1 ∨ x = -1/2)) ∨
  (a > 2 ∧ (∃ x : ℝ, x = (1 - a) / a ∨ x = -1 ∨ x = 1 - a)) ∨
  (0 ≤ a ∧ a ≤ 1 ∧ (∃ x : ℝ, x = -1)) ∨
  (a < 0 ∧ (∃ x : ℝ, x = (1 - a) / a ∨ x = -1)) := sorry

end NUMINAMATH_GPT_equation_solutions_l1892_189235


namespace NUMINAMATH_GPT_minimum_value_of_polynomial_l1892_189214

def polynomial (x : ℝ) : ℝ := (12 - x) * (10 - x) * (12 + x) * (10 + x)

theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial x = -484 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_polynomial_l1892_189214


namespace NUMINAMATH_GPT_max_value_l1892_189244

noncomputable def f (x y : ℝ) : ℝ := 8 * x ^ 2 + 9 * x * y + 18 * y ^ 2 + 2 * x + 3 * y
noncomputable def g (x y : ℝ) : Prop := 4 * x ^ 2 + 9 * y ^ 2 = 8

theorem max_value : ∃ x y : ℝ, g x y ∧ f x y = 26 :=
by
  sorry

end NUMINAMATH_GPT_max_value_l1892_189244


namespace NUMINAMATH_GPT_two_x_equals_y_l1892_189256

theorem two_x_equals_y (x y : ℝ) (h1 : (x + y) / 3 = 1) (h2 : x + 2 * y = 5) : 2 * x = y := 
by
  sorry

end NUMINAMATH_GPT_two_x_equals_y_l1892_189256


namespace NUMINAMATH_GPT_correct_calculation_l1892_189251

theorem correct_calculation (a b : ℝ) : 
  ¬(a * a^3 = a^3) ∧ ¬((a^2)^3 = a^5) ∧ (-a^2 * b)^2 = a^4 * b^2 ∧ ¬(a^3 / a = a^3) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_calculation_l1892_189251


namespace NUMINAMATH_GPT_range_of_a_l1892_189287

noncomputable def range_a (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ (x^3 * Real.exp (y / x) = a * y^3)

theorem range_of_a (a : ℝ) : range_a a → a ≥ Real.exp 3 / 27 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1892_189287


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1892_189230

variable (α β : Real)

theorem trigonometric_identity_proof :
  4.28 * Real.sin (β / 2 - Real.pi / 2) ^ 2 - Real.cos (α - 3 * Real.pi / 2) ^ 2 = 
  Real.cos (α + β) * Real.cos (α - β) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1892_189230


namespace NUMINAMATH_GPT_roots_of_quadratic_implies_values_l1892_189238

theorem roots_of_quadratic_implies_values (a b : ℝ) :
  (∃ x : ℝ, x^2 + 2 * (1 + a) * x + (3 * a^2 + 4 * a * b + 4 * b^2 + 2) = 0) →
  a = 1 ∧ b = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_implies_values_l1892_189238


namespace NUMINAMATH_GPT_sales_tax_difference_l1892_189203

theorem sales_tax_difference (rate1 rate2 : ℝ) (price : ℝ) (h1 : rate1 = 0.075) (h2 : rate2 = 0.0625) (hprice : price = 50) : 
  rate1 * price - rate2 * price = 0.625 :=
by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l1892_189203


namespace NUMINAMATH_GPT_frac_subtraction_l1892_189297

theorem frac_subtraction : (18 / 42) - (3 / 8) = (3 / 56) := by
  -- Conditions
  have h1 : 18 / 42 = 3 / 7 := by sorry
  have h2 : 3 / 7 = 24 / 56 := by sorry
  have h3 : 3 / 8 = 21 / 56 := by sorry
  -- Proof using the conditions
  sorry

end NUMINAMATH_GPT_frac_subtraction_l1892_189297


namespace NUMINAMATH_GPT_complex_number_addition_identity_l1892_189261

-- Definitions of the conditions
def imaginary_unit (i : ℂ) := i^2 = -1

def complex_fraction_decomposition (a b : ℝ) (i : ℂ) := 
  (1 + i) / (1 - i) = a + b * i

-- The statement of the problem
theorem complex_number_addition_identity :
  ∃ (a b : ℝ) (i : ℂ), imaginary_unit i ∧ complex_fraction_decomposition a b i ∧ (a + b = 1) :=
sorry

end NUMINAMATH_GPT_complex_number_addition_identity_l1892_189261


namespace NUMINAMATH_GPT_D_times_C_eq_l1892_189208

-- Define the matrices C and D
variable (C D : Matrix (Fin 2) (Fin 2) ℚ)

-- Add the conditions
axiom h1 : C * D = C + D
axiom h2 : C * D = ![![15/2, 9/2], ![-6/2, 12/2]]

-- Define the goal
theorem D_times_C_eq : D * C = ![![15/2, 9/2], ![-6/2, 12/2]] :=
sorry

end NUMINAMATH_GPT_D_times_C_eq_l1892_189208


namespace NUMINAMATH_GPT_sum_of_possible_values_of_a_l1892_189211

theorem sum_of_possible_values_of_a :
  (∀ r s : ℤ, r + s = a ∧ r * s = 3 * a) → ∃ a : ℤ, (a = 12) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_possible_values_of_a_l1892_189211


namespace NUMINAMATH_GPT_stock_price_after_two_years_l1892_189213

def initial_price : ℝ := 120

def first_year_increase (p : ℝ) : ℝ := p * 2

def second_year_decrease (p : ℝ) : ℝ := p * 0.30

def final_price (initial : ℝ) : ℝ :=
  let after_first_year := first_year_increase initial
  after_first_year - second_year_decrease after_first_year

theorem stock_price_after_two_years : final_price initial_price = 168 :=
by
  sorry

end NUMINAMATH_GPT_stock_price_after_two_years_l1892_189213


namespace NUMINAMATH_GPT_difference_in_tiles_l1892_189295

-- Definition of side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Theorem stating the difference in tiles between the 10th and 9th squares
theorem difference_in_tiles : (side_length 10) ^ 2 - (side_length 9) ^ 2 = 19 := 
by {
  sorry
}

end NUMINAMATH_GPT_difference_in_tiles_l1892_189295


namespace NUMINAMATH_GPT_intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l1892_189263

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }

def B : Set ℝ := { x | -4 < x ∧ x < 0 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -4 < x ∧ x ≤ -3 } :=
by sorry

theorem union_of_A_and_B :
  A ∪ B = { x | x < 0 ∨ x ≥ 1 } :=
by sorry

theorem complement_of_A_with_respect_to_U :
  U \ A = { x | -3 < x ∧ x < 1 } :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l1892_189263


namespace NUMINAMATH_GPT_total_amount_l1892_189216

theorem total_amount (x y z : ℝ) (hx : y = 45 / 0.45)
  (hy : z = (45 / 0.45) * 0.30)
  (hx_total : y = 45) :
  x + y + z = 175 :=
by
  -- Proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_total_amount_l1892_189216


namespace NUMINAMATH_GPT_selling_price_correct_l1892_189222

noncomputable def discount1 (price : ℝ) : ℝ := price * 0.85
noncomputable def discount2 (price : ℝ) : ℝ := price * 0.90
noncomputable def discount3 (price : ℝ) : ℝ := price * 0.95

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  discount3 (discount2 (discount1 initial_price))

theorem selling_price_correct : final_price 3600 = 2616.30 := by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l1892_189222


namespace NUMINAMATH_GPT_circle_equation_standard_l1892_189283

def center : ℝ × ℝ := (-1, 1)
def radius : ℝ := 2

theorem circle_equation_standard:
  (∀ x y : ℝ, ((x + 1)^2 + (y - 1)^2 = 4) ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2)) :=
by 
  intros x y
  rw [center, radius]
  simp
  sorry

end NUMINAMATH_GPT_circle_equation_standard_l1892_189283


namespace NUMINAMATH_GPT_floor_neg_seven_over_four_l1892_189224

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end NUMINAMATH_GPT_floor_neg_seven_over_four_l1892_189224


namespace NUMINAMATH_GPT_minimum_value_expr_l1892_189202

noncomputable def expr (x y z : ℝ) : ℝ :=
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) +
  (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2)))

theorem minimum_value_expr : ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) →
  expr x y z ≥ 2 :=
by sorry

end NUMINAMATH_GPT_minimum_value_expr_l1892_189202


namespace NUMINAMATH_GPT_find_y_l1892_189294

theorem find_y (y : ℕ) : y = (12 ^ 3 * 6 ^ 4) / 432 → y = 5184 :=
by
  intro h
  rw [h]
  sorry

end NUMINAMATH_GPT_find_y_l1892_189294


namespace NUMINAMATH_GPT_cost_of_each_pair_of_socks_eq_2_l1892_189204

-- Definitions and conditions
def cost_of_shoes : ℤ := 74
def cost_of_bag : ℤ := 42
def paid_amount : ℤ := 118
def discount_rate : ℚ := 0.10

-- Given the conditions
def total_cost (x : ℚ) : ℚ := cost_of_shoes + 2 * x + cost_of_bag
def discount (x : ℚ) : ℚ := if total_cost x > 100 then discount_rate * (total_cost x - 100) else 0
def total_cost_after_discount (x : ℚ) : ℚ := total_cost x - discount x

-- Theorem to prove
theorem cost_of_each_pair_of_socks_eq_2 : 
  ∃ x : ℚ, total_cost_after_discount x = paid_amount ∧ 2 * x = 4 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_pair_of_socks_eq_2_l1892_189204


namespace NUMINAMATH_GPT_food_for_elephants_l1892_189277

theorem food_for_elephants (t : ℕ) : 
  (∀ (food_per_day : ℕ), (12 * food_per_day) * 1 = (1000 * food_per_day) * 600) →
  (∀ (food_per_day : ℕ), (t * food_per_day) * 1 = (100 * food_per_day) * d) →
  d = 500 * t :=
by
  sorry

end NUMINAMATH_GPT_food_for_elephants_l1892_189277


namespace NUMINAMATH_GPT_height_of_prism_l1892_189233

-- Definitions based on conditions
def Volume : ℝ := 120
def edge1 : ℝ := 3
def edge2 : ℝ := 4
def BaseArea : ℝ := edge1 * edge2

-- Define the problem statement
theorem height_of_prism (h : ℝ) : (BaseArea * h / 2 = Volume) → (h = 20) :=
by
  intro h_value
  have Volume_equiv : h = 2 * Volume / BaseArea := sorry
  sorry

end NUMINAMATH_GPT_height_of_prism_l1892_189233


namespace NUMINAMATH_GPT_calculate_N_l1892_189236

theorem calculate_N (h : (25 / 100) * N = (55 / 100) * 3010) : N = 6622 :=
by
  sorry

end NUMINAMATH_GPT_calculate_N_l1892_189236


namespace NUMINAMATH_GPT_problem_statement_l1892_189298

theorem problem_statement (x y z : ℝ) 
  (h1 : x + y + z = 6) 
  (h2 : x * y + y * z + z * x = 11) 
  (h3 : x * y * z = 6) : 
  x / (y * z) + y / (z * x) + z / (x * y) = 7 / 3 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1892_189298


namespace NUMINAMATH_GPT_model_x_computers_used_l1892_189299

theorem model_x_computers_used
    (x_rate : ℝ)
    (y_rate : ℝ)
    (combined_rate : ℝ)
    (num_computers : ℝ) :
    x_rate = 1 / 72 →
    y_rate = 1 / 36 →
    combined_rate = num_computers * (x_rate + y_rate) →
    combined_rate = 1 →
    num_computers = 24 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_model_x_computers_used_l1892_189299


namespace NUMINAMATH_GPT_proof_f_g_l1892_189228

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 1
def g (x : ℝ) : ℝ := 2*x + 3

theorem proof_f_g (x : ℝ) : f (g 2) - g (f 2) = 258 :=
by
  sorry

end NUMINAMATH_GPT_proof_f_g_l1892_189228


namespace NUMINAMATH_GPT_tan_ratio_l1892_189229

theorem tan_ratio (p q : ℝ) 
  (h1: Real.sin (p + q) = 5 / 8)
  (h2: Real.sin (p - q) = 3 / 8) : Real.tan p / Real.tan q = 4 := 
by
  sorry

end NUMINAMATH_GPT_tan_ratio_l1892_189229


namespace NUMINAMATH_GPT_parabola_min_distance_a_l1892_189243

noncomputable def directrix_distance (P : Real × Real) (a : Real) : Real :=
abs (P.2 + 1 / (4 * a))

noncomputable def distance (P Q : Real × Real) : Real :=
Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem parabola_min_distance_a (a : Real) :
  (∀ (P : Real × Real), P.2 = a * P.1^2 → 
    distance P (2, 0) + directrix_distance P a = Real.sqrt 5) ↔ 
    a = 1 / 4 ∨ a = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_min_distance_a_l1892_189243


namespace NUMINAMATH_GPT_total_journey_distance_l1892_189292

theorem total_journey_distance (D : ℝ)
  (h1 : (D / 2) / 21 + (D / 2) / 24 = 25) : D = 560 := by
  sorry

end NUMINAMATH_GPT_total_journey_distance_l1892_189292


namespace NUMINAMATH_GPT_jenna_more_than_four_times_martha_l1892_189218

noncomputable def problems : ℝ := 20
noncomputable def martha_problems : ℝ := 2
noncomputable def angela_problems : ℝ := 9
noncomputable def jenna_problems : ℝ := 6  -- We calculated J = 6 from the conditions
noncomputable def mark_problems : ℝ := jenna_problems / 2

theorem jenna_more_than_four_times_martha :
  (jenna_problems - 4 * martha_problems = 2) :=
by
  sorry

end NUMINAMATH_GPT_jenna_more_than_four_times_martha_l1892_189218


namespace NUMINAMATH_GPT_gcd_50421_35343_l1892_189264

theorem gcd_50421_35343 : Int.gcd 50421 35343 = 23 := by
  sorry

end NUMINAMATH_GPT_gcd_50421_35343_l1892_189264


namespace NUMINAMATH_GPT_scorpion_millipedes_needed_l1892_189201

theorem scorpion_millipedes_needed 
  (total_segments_required : ℕ)
  (eaten_millipede_1_segments : ℕ)
  (eaten_millipede_2_segments : ℕ)
  (segments_per_millipede : ℕ)
  (n_millipedes_needed : ℕ)
  (total_segments : total_segments_required = 800) 
  (segments_1 : eaten_millipede_1_segments = 60)
  (segments_2 : eaten_millipede_2_segments = 2 * 60)
  (needed_segments_calculation : 800 - (60 + 2 * (2 * 60)) = n_millipedes_needed * 50) 
  : n_millipedes_needed = 10 :=
by
  sorry

end NUMINAMATH_GPT_scorpion_millipedes_needed_l1892_189201


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1892_189247

theorem solution_set_of_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + 3| ≥ a) ↔ a ≤ 5 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1892_189247


namespace NUMINAMATH_GPT_probability_painted_faces_l1892_189286

theorem probability_painted_faces (total_cubes : ℕ) (corner_cubes : ℕ) (no_painted_face_cubes : ℕ) (successful_outcomes : ℕ) (total_outcomes : ℕ) 
  (probability : ℚ) : 
  total_cubes = 125 ∧ corner_cubes = 8 ∧ no_painted_face_cubes = 27 ∧ successful_outcomes = 216 ∧ total_outcomes = 7750 ∧ 
  probability = 72 / 2583 :=
by
  sorry

end NUMINAMATH_GPT_probability_painted_faces_l1892_189286


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1892_189268

theorem arithmetic_sequence_common_difference (a d : ℕ) (n : ℕ) :
  a = 5 →
  (a + (n - 1) * d = 50) →
  (n * (a + (a + (n - 1) * d)) / 2 = 275) →
  d = 5 := 
by
  intros ha ha_n hs_n
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1892_189268


namespace NUMINAMATH_GPT_sin_cos_alpha_l1892_189210

open Real

theorem sin_cos_alpha (α : ℝ) (h1 : sin (2 * α) = -sqrt 2 / 2) (h2 : α ∈ Set.Ioc (3 * π / 2) (2 * π)) :
  sin α + cos α = sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_sin_cos_alpha_l1892_189210


namespace NUMINAMATH_GPT_slope_of_CD_l1892_189239

-- Given circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 15 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 48 = 0

-- Define the line whose slope needs to be found
def line (x y : ℝ) : Prop := 22*x - 12*y - 33 = 0

-- State the proof problem
theorem slope_of_CD : ∀ x y : ℝ, circle1 x y → circle2 x y → line x y ∧ (∃ m : ℝ, m = 11/6) :=
by sorry

end NUMINAMATH_GPT_slope_of_CD_l1892_189239


namespace NUMINAMATH_GPT_roman_remy_gallons_l1892_189278

theorem roman_remy_gallons (R : ℕ) (Remy_uses : 3 * R + 1 = 25) :
  R + (3 * R + 1) = 33 :=
by
  sorry

end NUMINAMATH_GPT_roman_remy_gallons_l1892_189278


namespace NUMINAMATH_GPT_factor_expression_l1892_189209

theorem factor_expression (x : ℝ) : 2 * x * (x + 3) + (x + 3) = (2 * x + 1) * (x + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1892_189209


namespace NUMINAMATH_GPT_geometric_sequence_product_l1892_189269

-- Defining a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Given data
def a := fun n => (4 : ℝ) * (2 : ℝ)^(n-4)

-- Main proof problem
theorem geometric_sequence_product (a : ℕ → ℝ) (h : is_geometric_sequence a) (h₁ : a 4 = 4) :
  a 2 * a 6 = 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1892_189269


namespace NUMINAMATH_GPT_suresh_investment_correct_l1892_189225

noncomputable def suresh_investment
  (ramesh_investment : ℝ)
  (total_profit : ℝ)
  (ramesh_profit_share : ℝ)
  : ℝ := sorry

theorem suresh_investment_correct
  (ramesh_investment : ℝ := 40000)
  (total_profit : ℝ := 19000)
  (ramesh_profit_share : ℝ := 11875)
  : suresh_investment ramesh_investment total_profit ramesh_profit_share = 24000 := sorry

end NUMINAMATH_GPT_suresh_investment_correct_l1892_189225


namespace NUMINAMATH_GPT_area_triangle_CIN_l1892_189242

variables (A B C D M N I : Type*)

-- Definitions and assumptions
-- ABCD is a square
def is_square (ABCD : Type*) (side : ℝ) : Prop := sorry
-- M is the midpoint of AB
def midpoint_AB (M A B : Type*) : Prop := sorry
-- N is the midpoint of BC
def midpoint_BC (N B C : Type*) : Prop := sorry
-- Lines CM and DN intersect at I
def lines_intersect_at (C M D N I : Type*) : Prop := sorry

-- Goal
theorem area_triangle_CIN (ABCD : Type*) (side : ℝ) (M N C I : Type*) 
  (h1 : is_square ABCD side)
  (h2 : midpoint_AB M A B)
  (h3 : midpoint_BC N B C)
  (h4 : lines_intersect_at C M D N I) :
  sorry := sorry

end NUMINAMATH_GPT_area_triangle_CIN_l1892_189242


namespace NUMINAMATH_GPT_weight_box_plate_cups_l1892_189281

theorem weight_box_plate_cups (b p c : ℝ) 
  (h₁ : b + 20 * p + 30 * c = 4.8)
  (h₂ : b + 40 * p + 50 * c = 8.4) : 
  b + 10 * p + 20 * c = 3 :=
sorry

end NUMINAMATH_GPT_weight_box_plate_cups_l1892_189281
