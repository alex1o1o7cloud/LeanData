import Mathlib

namespace NUMINAMATH_GPT_larger_sign_diameter_l2394_239455

theorem larger_sign_diameter (d k : ℝ) 
  (h1 : ∀ d, d > 0) 
  (h2 : ∀ k, (π * (k * d / 2)^2 = 49 * π * (d / 2)^2)) : 
  k = 7 :=
by
sorry

end NUMINAMATH_GPT_larger_sign_diameter_l2394_239455


namespace NUMINAMATH_GPT_tan_inequality_l2394_239474

variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := Real.tan x

theorem tan_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < π / 2) (h3 : 0 < x2) (h4 : x2 < π / 2) (h5 : x1 ≠ x2) :
  (1/2) * (f x1 + f x2) > f ((x1 + x2) / 2) :=
  sorry

end NUMINAMATH_GPT_tan_inequality_l2394_239474


namespace NUMINAMATH_GPT_range_of_a_l2394_239436

theorem range_of_a (a : ℝ) : (1 < a) → 
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ 
  (1 / (x1 + 2) = a * |x1| ∧ 1 / (x2 + 2) = a * |x2| ∧ 1 / (x3 + 2) = a * |x3|) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2394_239436


namespace NUMINAMATH_GPT_part1_l2394_239411

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : 
  f x 1 ≥ 1 :=
sorry

end NUMINAMATH_GPT_part1_l2394_239411


namespace NUMINAMATH_GPT_square_area_proof_l2394_239489

theorem square_area_proof
  (x s : ℝ)
  (h1 : x^2 = 3 * s)
  (h2 : 4 * x = s^2) :
  x^2 = 6 :=
  sorry

end NUMINAMATH_GPT_square_area_proof_l2394_239489


namespace NUMINAMATH_GPT_find_p_l2394_239497

noncomputable def binomial_parameter (n : ℕ) (p : ℚ) (E : ℚ) (D : ℚ) : Prop :=
  E = n * p ∧ D = n * p * (1 - p)

theorem find_p (n : ℕ) (p : ℚ) 
  (hE : n * p = 50)
  (hD : n * p * (1 - p) = 30)
  : p = 2 / 5 :=
sorry

end NUMINAMATH_GPT_find_p_l2394_239497


namespace NUMINAMATH_GPT_difference_of_quarters_l2394_239401

variables (n d q : ℕ)

theorem difference_of_quarters :
  (n + d + q = 150) ∧ (5 * n + 10 * d + 25 * q = 1425) →
  (∃ qmin qmax : ℕ, q = qmax - qmin ∧ qmax - qmin = 30) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_quarters_l2394_239401


namespace NUMINAMATH_GPT_exists_square_all_invisible_l2394_239493

open Nat

theorem exists_square_all_invisible (n : ℕ) : 
  ∃ a b : ℤ, ∀ i j : ℕ, i < n → j < n → gcd (a + i) (b + j) > 1 := 
sorry

end NUMINAMATH_GPT_exists_square_all_invisible_l2394_239493


namespace NUMINAMATH_GPT_rectangle_original_length_doubles_area_l2394_239491

-- Let L and W denote the length and width of a rectangle respectively
-- Given the condition: (L + 2)W = 2LW
-- We need to prove that L = 2

theorem rectangle_original_length_doubles_area (L W : ℝ) (h : (L + 2) * W = 2 * L * W) : L = 2 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_original_length_doubles_area_l2394_239491


namespace NUMINAMATH_GPT_minimum_value_m_sq_plus_n_sq_l2394_239418

theorem minimum_value_m_sq_plus_n_sq :
  ∃ (m n : ℝ), (m ≠ 0) ∧ (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 4 ∧ (m * x^2 + (2 * n + 1) * x - m - 2) = 0) ∧
  (m^2 + n^2) = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_m_sq_plus_n_sq_l2394_239418


namespace NUMINAMATH_GPT_problem_proof_l2394_239405

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- The conditions
def even_function_on_ℝ := ∀ x : ℝ, f x = f (-x)
def f_at_0_is_2 := f 0 = 2
def odd_after_translation := ∀ x : ℝ, f (x - 1) = -f (-x - 1)

-- Prove the required condition
theorem problem_proof (h1 : even_function_on_ℝ f) (h2 : f_at_0_is_2 f) (h3 : odd_after_translation f) :
    f 1 + f 3 + f 5 + f 7 + f 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l2394_239405


namespace NUMINAMATH_GPT_roses_and_orchids_difference_l2394_239469

theorem roses_and_orchids_difference :
  let roses_now := 11
  let orchids_now := 20
  orchids_now - roses_now = 9 := 
by
  sorry

end NUMINAMATH_GPT_roses_and_orchids_difference_l2394_239469


namespace NUMINAMATH_GPT_num_values_f100_eq_0_l2394_239423

def f0 (x : ℝ) : ℝ := x + |x - 100| - |x + 100|

def fn : ℕ → ℝ → ℝ
| 0, x   => f0 x
| (n+1), x => |fn n x| - 1

theorem num_values_f100_eq_0 : ∃ (xs : Finset ℝ), ∀ x ∈ xs, fn 100 x = 0 ∧ xs.card = 301 :=
by
  sorry

end NUMINAMATH_GPT_num_values_f100_eq_0_l2394_239423


namespace NUMINAMATH_GPT_persimmons_in_Jungkook_house_l2394_239485

-- Define the number of boxes and the number of persimmons per box
def num_boxes : ℕ := 4
def persimmons_per_box : ℕ := 5

-- Define the total number of persimmons calculation
def total_persimmons (boxes : ℕ) (per_box : ℕ) : ℕ := boxes * per_box

-- The main theorem statement proving the total number of persimmons
theorem persimmons_in_Jungkook_house : total_persimmons num_boxes persimmons_per_box = 20 := 
by 
  -- We should prove this, but we use 'sorry' to skip proof in this example.
  sorry

end NUMINAMATH_GPT_persimmons_in_Jungkook_house_l2394_239485


namespace NUMINAMATH_GPT_tank_capacity_l2394_239437

theorem tank_capacity (C : ℝ) :
  (3/4 * C - 0.4 * (3/4 * C) + 0.3 * (3/4 * C - 0.4 * (3/4 * C))) = 4680 → C = 8000 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l2394_239437


namespace NUMINAMATH_GPT_number_of_cards_above_1999_l2394_239451

def numberOfCardsAbove1999 (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if numberOfCardsAbove1999 (n-1) = n-2 then 1
  else numberOfCardsAbove1999 (n-1) + 2

theorem number_of_cards_above_1999 : numberOfCardsAbove1999 2000 = 927 := by
  sorry

end NUMINAMATH_GPT_number_of_cards_above_1999_l2394_239451


namespace NUMINAMATH_GPT_function_zero_solution_l2394_239402

def floor (x : ℝ) : ℤ := sorry -- Define floor function properly.

theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = (-1) ^ (floor y) * f x + (-1) ^ (floor x) * f y) →
  (∀ x : ℝ, f x = 0) := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_function_zero_solution_l2394_239402


namespace NUMINAMATH_GPT_total_cost_l2394_239479

noncomputable def C1 : ℝ := 990 / 1.10
noncomputable def C2 : ℝ := 990 / 0.90

theorem total_cost (SP : ℝ) (profit_rate loss_rate : ℝ) : SP = 990 ∧ profit_rate = 0.10 ∧ loss_rate = 0.10 →
  C1 + C2 = 2000 :=
by
  intro h
  -- Show the sum of C1 and C2 equals 2000
  sorry

end NUMINAMATH_GPT_total_cost_l2394_239479


namespace NUMINAMATH_GPT_nth_term_arithmetic_seq_l2394_239449

theorem nth_term_arithmetic_seq (a b n t count : ℕ) (h1 : count = 25) (h2 : a = 3) (h3 : b = 75) (h4 : n = 8) :
    t = a + (n - 1) * ((b - a) / (count - 1)) → t = 24 :=
by
  intros
  sorry

end NUMINAMATH_GPT_nth_term_arithmetic_seq_l2394_239449


namespace NUMINAMATH_GPT_sale_price_tea_correct_l2394_239443

noncomputable def sale_price_of_mixed_tea (weight1 weight2 price1 price2 profit_percentage : ℝ) : ℝ :=
let total_cost := weight1 * price1 + weight2 * price2
let total_weight := weight1 + weight2
let cost_price_per_kg := total_cost / total_weight
let profit_per_kg := profit_percentage * cost_price_per_kg
let sale_price_per_kg := cost_price_per_kg + profit_per_kg
sale_price_per_kg

theorem sale_price_tea_correct :
  sale_price_of_mixed_tea 80 20 15 20 0.20 = 19.2 :=
  by
  sorry

end NUMINAMATH_GPT_sale_price_tea_correct_l2394_239443


namespace NUMINAMATH_GPT_percentage_of_sikh_boys_l2394_239470

theorem percentage_of_sikh_boys (total_boys muslim_percentage hindu_percentage other_boys : ℕ) 
  (h₁ : total_boys = 300) 
  (h₂ : muslim_percentage = 44) 
  (h₃ : hindu_percentage = 28) 
  (h₄ : other_boys = 54) : 
  (10 : ℝ) = 
  (((total_boys - (muslim_percentage * total_boys / 100 + hindu_percentage * total_boys / 100 + other_boys)) * 100) / total_boys : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_sikh_boys_l2394_239470


namespace NUMINAMATH_GPT_quadratic_solution_l2394_239440

theorem quadratic_solution : 
  ∀ x : ℝ, (x^2 - 2*x - 3 = 0) ↔ (x = 3 ∨ x = -1) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_solution_l2394_239440


namespace NUMINAMATH_GPT_oranges_for_juice_l2394_239429

-- Define conditions
def total_oranges : ℝ := 7 -- in million tons
def export_percentage : ℝ := 0.25
def juice_percentage : ℝ := 0.60

-- Define the mathematical problem
theorem oranges_for_juice : 
  (total_oranges * (1 - export_percentage) * juice_percentage) = 3.2 :=
by
  sorry

end NUMINAMATH_GPT_oranges_for_juice_l2394_239429


namespace NUMINAMATH_GPT_coin_grid_probability_l2394_239490

/--
A square grid is given where the edge length of each smallest square is 6 cm.
A hard coin with a diameter of 2 cm is thrown onto this grid.
Prove that the probability that the coin, after landing, will have a common point with the grid lines is 5/9.
-/
theorem coin_grid_probability :
  let square_edge_cm := 6
  let coin_diameter_cm := 2
  let coin_radius_cm := coin_diameter_cm / 2
  let grid_center_edge_cm := square_edge_cm - coin_diameter_cm
  let non_intersect_area_ratio := (grid_center_edge_cm ^ 2) / (square_edge_cm ^ 2)
  1 - non_intersect_area_ratio = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_coin_grid_probability_l2394_239490


namespace NUMINAMATH_GPT_ratio_of_large_rooms_l2394_239477

-- Definitions for the problem conditions
def total_classrooms : ℕ := 15
def total_students : ℕ := 400
def desks_in_large_room : ℕ := 30
def desks_in_small_room : ℕ := 25

-- Define x as the number of large (30-desk) rooms and y as the number of small (25-desk) rooms
variables (x y : ℕ)

-- Two conditions provided by the problem
def classrooms_condition := x + y = total_classrooms
def students_condition := desks_in_large_room * x + desks_in_small_room * y = total_students

-- Our main theorem to prove
theorem ratio_of_large_rooms :
  classrooms_condition x y →
  students_condition x y →
  (x : ℚ) / (total_classrooms : ℚ) = 1 / 3 :=
by
-- Here we would have our proof, but we leave it as "sorry" since the task only requires the statement.
sorry

end NUMINAMATH_GPT_ratio_of_large_rooms_l2394_239477


namespace NUMINAMATH_GPT_popsicle_sticks_l2394_239426

theorem popsicle_sticks (total_sticks : ℕ) (gino_sticks : ℕ) (my_sticks : ℕ) 
  (h1 : total_sticks = 113) (h2 : gino_sticks = 63) (h3 : total_sticks = gino_sticks + my_sticks) : 
  my_sticks = 50 :=
  sorry

end NUMINAMATH_GPT_popsicle_sticks_l2394_239426


namespace NUMINAMATH_GPT_find_number_l2394_239495

theorem find_number (n x : ℝ) (hx : x = 0.8999999999999999) (h : n / x = 0.01) : n = 0.008999999999999999 := by
  sorry

end NUMINAMATH_GPT_find_number_l2394_239495


namespace NUMINAMATH_GPT_part1_part2_l2394_239430

def A (x : ℝ) : Prop := x < -2 ∨ x > 3
def B (a : ℝ) (x : ℝ) : Prop := 1 - a < x ∧ x < a + 3

theorem part1 (x : ℝ) : (¬A x ∨ B 1 x) ↔ -2 ≤ x ∧ x < 4 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, ¬(A x ∧ B a x)) ↔ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2394_239430


namespace NUMINAMATH_GPT_invalid_votes_l2394_239415

theorem invalid_votes (W L total_polls : ℕ) 
  (h1 : total_polls = 90830) 
  (h2 : L = 9 * W / 11) 
  (h3 : W = L + 9000)
  (h4 : 100 * (W + L) = 90000) : 
  total_polls - (W + L) = 830 := 
sorry

end NUMINAMATH_GPT_invalid_votes_l2394_239415


namespace NUMINAMATH_GPT_range_of_dot_product_l2394_239484

theorem range_of_dot_product
  (a b : ℝ)
  (h: ∃ (A B : ℝ × ℝ), (A ≠ B) ∧ ∃ m n : ℝ, A = (m, n) ∧ B = (-m, -n) ∧ m^2 + (n^2 / 9) = 1)
  : ∃ r : Set ℝ, r = (Set.Icc 41 49) :=
  sorry

end NUMINAMATH_GPT_range_of_dot_product_l2394_239484


namespace NUMINAMATH_GPT_abc_is_772_l2394_239480

noncomputable def find_abc (a b c : ℝ) : ℝ :=
if h₁ : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * (b + c) = 160 ∧ b * (c + a) = 168 ∧ c * (a + b) = 180
then 772 else 0

theorem abc_is_772 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
(h₄ : a * (b + c) = 160) (h₅ : b * (c + a) = 168) (h₆ : c * (a + b) = 180) :
  find_abc a b c = 772 := by
  sorry

end NUMINAMATH_GPT_abc_is_772_l2394_239480


namespace NUMINAMATH_GPT_scale_model_height_l2394_239435

theorem scale_model_height (real_height : ℕ) (scale_ratio : ℕ) (h_real : real_height = 1454) (h_scale : scale_ratio = 50) : 
⌊(real_height : ℝ) / scale_ratio + 0.5⌋ = 29 :=
by
  rw [h_real, h_scale]
  norm_num
  sorry

end NUMINAMATH_GPT_scale_model_height_l2394_239435


namespace NUMINAMATH_GPT_margin_expression_l2394_239417

variable (n : ℕ) (C S M : ℝ)

theorem margin_expression (H1 : M = (1 / n) * C) (H2 : C = S - M) : 
  M = (1 / (n + 1)) * S := 
by
  sorry

end NUMINAMATH_GPT_margin_expression_l2394_239417


namespace NUMINAMATH_GPT_fractional_expression_simplification_l2394_239463

theorem fractional_expression_simplification (x : ℕ) (h : x - 3 < 0) : 
  (2 * x) / (x^2 - 1) - 1 / (x - 1) = 1 / 3 :=
by {
  -- Typical proof steps would go here, adhering to the natural conditions.
  sorry
}

end NUMINAMATH_GPT_fractional_expression_simplification_l2394_239463


namespace NUMINAMATH_GPT_ratio_john_maya_age_l2394_239412

theorem ratio_john_maya_age :
  ∀ (john drew maya peter jacob : ℕ),
  -- Conditions:
  john = 30 ∧
  drew = maya + 5 ∧
  peter = drew + 4 ∧
  jacob = 11 ∧
  jacob + 2 = (peter + 2) / 2 →
  -- Conclusion:
  john / gcd john maya = 2 ∧ maya / gcd john maya = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_john_maya_age_l2394_239412


namespace NUMINAMATH_GPT_frog_jump_l2394_239450

def coprime (p q : ℕ) : Prop := Nat.gcd p q = 1

theorem frog_jump (p q : ℕ) (h_coprime : coprime p q) :
  ∀ d : ℕ, d < p + q → (∃ m n : ℤ, m ≠ n ∧ (m - n = d ∨ n - m = d)) :=
by
  sorry

end NUMINAMATH_GPT_frog_jump_l2394_239450


namespace NUMINAMATH_GPT_no_number_exists_decreasing_by_removing_digit_l2394_239461

theorem no_number_exists_decreasing_by_removing_digit :
  ¬ ∃ (x y n : ℕ), x * 10^n + y = 58 * y :=
by
  sorry

end NUMINAMATH_GPT_no_number_exists_decreasing_by_removing_digit_l2394_239461


namespace NUMINAMATH_GPT_father_l2394_239421

theorem father's_age_at_middle_son_birth (a b c F : ℕ) 
  (h1 : a = b + c) 
  (h2 : F * a * b * c = 27090) : 
  F - b = 34 :=
by sorry

end NUMINAMATH_GPT_father_l2394_239421


namespace NUMINAMATH_GPT_andres_possibilities_10_dollars_l2394_239487

theorem andres_possibilities_10_dollars : 
  (∃ (num_1_coins num_2_coins num_5_bills : ℕ),
    num_1_coins + 2 * num_2_coins + 5 * num_5_bills = 10) → 
  ∃ (ways : ℕ), ways = 10 :=
by
  -- The proof can be provided here, but we'll use sorry to skip it in this template.
  sorry

end NUMINAMATH_GPT_andres_possibilities_10_dollars_l2394_239487


namespace NUMINAMATH_GPT_helen_made_56_pies_l2394_239472

theorem helen_made_56_pies (pinky_pies total_pies : ℕ) (h_pinky : pinky_pies = 147) (h_total : total_pies = 203) :
  (total_pies - pinky_pies) = 56 :=
by
  sorry

end NUMINAMATH_GPT_helen_made_56_pies_l2394_239472


namespace NUMINAMATH_GPT_total_number_of_shirts_l2394_239486

variable (total_cost : ℕ) (num_15_dollar_shirts : ℕ) (cost_15_dollar_shirts : ℕ) 
          (cost_remaining_shirts : ℕ) (num_remaining_shirts : ℕ) 

theorem total_number_of_shirts :
  total_cost = 85 →
  num_15_dollar_shirts = 3 →
  cost_15_dollar_shirts = 15 →
  cost_remaining_shirts = 20 →
  (num_remaining_shirts * cost_remaining_shirts) + (num_15_dollar_shirts * cost_15_dollar_shirts) = total_cost →
  num_15_dollar_shirts + num_remaining_shirts = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_number_of_shirts_l2394_239486


namespace NUMINAMATH_GPT_max_a_plus_b_min_a_squared_plus_b_squared_l2394_239447

theorem max_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 - a - b + a * b = 1) :
  a + b ≤ 2 := 
sorry

theorem min_a_squared_plus_b_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 - a - b + a * b = 1) :
  2 ≤ a^2 + b^2 := 
sorry

end NUMINAMATH_GPT_max_a_plus_b_min_a_squared_plus_b_squared_l2394_239447


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2394_239419

theorem solution_set_of_inequality
  (a b : ℝ)
  (x y : ℝ)
  (h1 : a * (-2) + b = 3)
  (h2 : a * (-1) + b = 2)
  :  -x + 1 < 0 ↔ x > 1 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2394_239419


namespace NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_div_2_l2394_239427

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_div_2_l2394_239427


namespace NUMINAMATH_GPT_area_of_trapezoid_l2394_239439

-- Definitions of geometric properties and conditions
def is_perpendicular (a b c : ℝ) : Prop := a + b = 90 -- representing ∠ABC = 90°
def tangent_length (bc ad : ℝ) (O : ℝ) : Prop := bc * ad = O -- representing BC tangent to O with diameter AD
def is_diameter (ad r : ℝ) : Prop := ad = 2 * r -- AD being the diameter of the circle with radius r

-- Given conditions in the problem
variables (AB BC CD AD r O : ℝ) (h1 : is_perpendicular AB BC 90) (h2 : is_perpendicular BC CD 90)
          (h3 : tangent_length BC AD O) (h4 : is_diameter AD r) (h5 : BC = 2 * CD)
          (h6 : AB = 9) (h7 : CD = 3)

-- Statement to prove the area is 36
theorem area_of_trapezoid : (AB + CD) * CD = 36 := by
  sorry

end NUMINAMATH_GPT_area_of_trapezoid_l2394_239439


namespace NUMINAMATH_GPT_seq_a8_value_l2394_239425

theorem seq_a8_value 
  (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n < a (n + 1)) 
  (h2 : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n) 
  (a7_eq : a 7 = 120) 
  : a 8 = 194 :=
sorry

end NUMINAMATH_GPT_seq_a8_value_l2394_239425


namespace NUMINAMATH_GPT_election_total_votes_l2394_239404

theorem election_total_votes (V : ℝ)
  (h_majority : ∃ O, 0.84 * V = O + 476)
  (h_total_votes : ∀ O, V = 0.84 * V + O) :
  V = 700 :=
sorry

end NUMINAMATH_GPT_election_total_votes_l2394_239404


namespace NUMINAMATH_GPT_radius_of_O2_l2394_239476

theorem radius_of_O2 (r_O1 r_dist r_O2 : ℝ) 
  (h1 : r_O1 = 3) 
  (h2 : r_dist = 7) 
  (h3 : (r_dist = r_O1 + r_O2 ∨ r_dist = |r_O2 - r_O1|)) :
  r_O2 = 4 ∨ r_O2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_O2_l2394_239476


namespace NUMINAMATH_GPT_range_of_a_l2394_239488

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a-1)*x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2394_239488


namespace NUMINAMATH_GPT_negation_equivalence_l2394_239453

-- Define the original proposition P
def proposition_P : Prop := ∀ x : ℝ, 0 ≤ x → x^3 + 2 * x ≥ 0

-- Define the negation of the proposition P
def negation_P : Prop := ∃ x : ℝ, 0 ≤ x ∧ x^3 + 2 * x < 0

-- The statement to be proven
theorem negation_equivalence : ¬ proposition_P ↔ negation_P := 
by sorry

end NUMINAMATH_GPT_negation_equivalence_l2394_239453


namespace NUMINAMATH_GPT_max_sum_of_factors_of_48_l2394_239496

theorem max_sum_of_factors_of_48 : ∃ (heartsuit clubsuit : ℕ), heartsuit * clubsuit = 48 ∧ heartsuit + clubsuit = 49 :=
by
  -- We insert sorry here to skip the actual proof construction.
  sorry

end NUMINAMATH_GPT_max_sum_of_factors_of_48_l2394_239496


namespace NUMINAMATH_GPT_Kolya_Homework_Problem_l2394_239413

-- Given conditions as definitions
def squaresToDigits (x : ℕ) (a b : ℕ) : Prop := x^2 = 10 * a + b
def doubledToDigits (x : ℕ) (a b : ℕ) : Prop := 2 * x = 10 * b + a

-- The main theorem statement
theorem Kolya_Homework_Problem :
  ∃ (x a b : ℕ), squaresToDigits x a b ∧ doubledToDigits x a b ∧ x = 9 ∧ x^2 = 81 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_Kolya_Homework_Problem_l2394_239413


namespace NUMINAMATH_GPT_cassandra_collected_pennies_l2394_239422

theorem cassandra_collected_pennies 
(C : ℕ) 
(h1 : ∀ J : ℕ,  J = C - 276) 
(h2 : ∀ J : ℕ, C + J = 9724) 
: C = 5000 := 
by
  sorry

end NUMINAMATH_GPT_cassandra_collected_pennies_l2394_239422


namespace NUMINAMATH_GPT_faith_change_l2394_239462

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def given_amount : ℕ := 20 * 2 + 3
def total_cost := flour_cost + cake_stand_cost
def change := given_amount - total_cost

theorem faith_change : change = 10 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_faith_change_l2394_239462


namespace NUMINAMATH_GPT_tangent_line_at_origin_local_minimum_at_zero_number_of_zeros_l2394_239448

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (x - 1) - 1 / 2 * Real.exp a * x^2

theorem tangent_line_at_origin (a : ℝ) (h : a < 0) : 
  let f₀ := f 0 a
  ∃ c : ℝ, (∀ x : ℝ,  f₀ + c * x = -1) := sorry

theorem local_minimum_at_zero (a : ℝ) (h : a < 0) :
  ∀ x : ℝ, f 0 a ≤ f x a := sorry

theorem number_of_zeros (a : ℝ) (h : a < 0) :
  ∃! x : ℝ, f x a = 0 := sorry

end NUMINAMATH_GPT_tangent_line_at_origin_local_minimum_at_zero_number_of_zeros_l2394_239448


namespace NUMINAMATH_GPT_number_of_routes_from_A_to_L_is_6_l2394_239494

def A_to_B_or_E : Prop := True
def B_to_A_or_C_or_F : Prop := True
def C_to_B_or_D_or_G : Prop := True
def D_to_C_or_H : Prop := True
def E_to_A_or_F_or_I : Prop := True
def F_to_B_or_E_or_G_or_J : Prop := True
def G_to_C_or_F_or_H_or_K : Prop := True
def H_to_D_or_G_or_L : Prop := True
def I_to_E_or_J : Prop := True
def J_to_F_or_I_or_K : Prop := True
def K_to_G_or_J_or_L : Prop := True
def L_from_H_or_K : Prop := True

theorem number_of_routes_from_A_to_L_is_6 
  (h1 : A_to_B_or_E)
  (h2 : B_to_A_or_C_or_F)
  (h3 : C_to_B_or_D_or_G)
  (h4 : D_to_C_or_H)
  (h5 : E_to_A_or_F_or_I)
  (h6 : F_to_B_or_E_or_G_or_J)
  (h7 : G_to_C_or_F_or_H_or_K)
  (h8 : H_to_D_or_G_or_L)
  (h9 : I_to_E_or_J)
  (h10 : J_to_F_or_I_or_K)
  (h11 : K_to_G_or_J_or_L)
  (h12 : L_from_H_or_K) : 
  6 = 6 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_routes_from_A_to_L_is_6_l2394_239494


namespace NUMINAMATH_GPT_standing_next_to_boris_l2394_239424

structure Person :=
  (name : String)

def Arkady := Person.mk "Arkady"
def Boris := Person.mk "Boris"
def Vera := Person.mk "Vera"
def Galya := Person.mk "Galya"
def Danya := Person.mk "Danya"
def Egor := Person.mk "Egor"

def next_to (p1 p2 : Person) : Prop := sorry
def opposite (p1 p2 : Person) : Prop := sorry

axiom condition1 : next_to Danya Vera
axiom condition2 : next_to Danya Egor
axiom condition3 : opposite Galya Egor
axiom condition4 : ¬ next_to Arkady Galya

theorem standing_next_to_boris : next_to Boris Arkady ∧ next_to Boris Galya := 
by {
  sorry
}

end NUMINAMATH_GPT_standing_next_to_boris_l2394_239424


namespace NUMINAMATH_GPT_max_band_members_l2394_239410

theorem max_band_members (r x m : ℕ) (h1 : m < 150) (h2 : r * x + 3 = m) (h3 : (r - 3) * (x + 2) = m) : m = 147 := by
  sorry

end NUMINAMATH_GPT_max_band_members_l2394_239410


namespace NUMINAMATH_GPT_gcd_of_products_l2394_239431

theorem gcd_of_products (a b a' b' d d' : ℕ) (h1 : Nat.gcd a b = d) (h2 : Nat.gcd a' b' = d') (ha : 0 < a) (hb : 0 < b) (ha' : 0 < a') (hb' : 0 < b') :
  Nat.gcd (Nat.gcd (aa') (ab')) (Nat.gcd (ba') (bb')) = d * d' := 
sorry

end NUMINAMATH_GPT_gcd_of_products_l2394_239431


namespace NUMINAMATH_GPT_bisection_method_next_interval_l2394_239483

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x0 := (a + b) / 2
  (f a * f x0 < 0) ∨ (f x0 * f b < 0) →
  (x0 = 2.5) →
  f 2 * f 2.5 < 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_bisection_method_next_interval_l2394_239483


namespace NUMINAMATH_GPT_problem_l2394_239420

open Real

noncomputable def f (x : ℝ) : ℝ := exp (2 * x) + 2 * cos x - 4

theorem problem (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * π) : 
  ∀ a b : ℝ, (0 ≤ a ∧ a ≤ 2 * π) → (0 ≤ b ∧ b ≤ 2 * π) → a ≤ b → f a ≤ f b := 
sorry

end NUMINAMATH_GPT_problem_l2394_239420


namespace NUMINAMATH_GPT_certain_number_exists_l2394_239428

theorem certain_number_exists
  (N : ℕ) 
  (hN : ∀ x, x < N → x % 2 = 1 → ∃ k m, k = 5 * m ∧ x = k ∧ m % 2 = 1) :
  N = 76 := by
  sorry

end NUMINAMATH_GPT_certain_number_exists_l2394_239428


namespace NUMINAMATH_GPT_integer_part_of_shortest_distance_l2394_239465

def cone_slant_height := 21
def cone_radius := 14
def ant_position := cone_slant_height / 2
def angle_opposite := 240
def cos_angle_opposite := -1 / 2

noncomputable def shortest_distance := 
  Real.sqrt ((ant_position ^ 2) + (ant_position ^ 2) + (2 * ant_position ^ 2 * cos_angle_opposite))

theorem integer_part_of_shortest_distance : Int.floor shortest_distance = 18 :=
by
  /- Proof steps go here -/
  sorry

end NUMINAMATH_GPT_integer_part_of_shortest_distance_l2394_239465


namespace NUMINAMATH_GPT_less_money_than_Bob_l2394_239471

noncomputable def Jennas_money (P: ℝ) : ℝ := 2 * P
noncomputable def Phils_money (B: ℝ) : ℝ := B / 3
noncomputable def Bobs_money : ℝ := 60
noncomputable def Johns_money (P: ℝ) : ℝ := P + 0.35 * P
noncomputable def average (x y: ℝ) : ℝ := (x + y) / 2

theorem less_money_than_Bob :
  ∀ (P Q J B : ℝ),
    P = Phils_money B →
    J = Jennas_money P →
    Q = Johns_money P →
    B = Bobs_money →
    average J Q = B - 0.25 * B →
    B - J = 20
  :=
by
  intros P Q J B hP hJ hQ hB h_avg
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_less_money_than_Bob_l2394_239471


namespace NUMINAMATH_GPT_prove_a_eq_b_l2394_239464

theorem prove_a_eq_b 
    (a b : ℕ) 
    (h_pos : a > 0 ∧ b > 0) 
    (h_multiple : ∃ k : ℤ, a^2 + a * b + 1 = k * (b^2 + b * a + 1)) : 
    a = b := 
sorry

end NUMINAMATH_GPT_prove_a_eq_b_l2394_239464


namespace NUMINAMATH_GPT_frac_multiplication_l2394_239434

theorem frac_multiplication : 
    ((2/3:ℚ)^4 * (1/5) * (3/4) = 4/135) :=
by
  sorry

end NUMINAMATH_GPT_frac_multiplication_l2394_239434


namespace NUMINAMATH_GPT_ground_beef_per_package_l2394_239459

-- Declare the given conditions and the expected result.
theorem ground_beef_per_package (num_people : ℕ) (weight_per_burger : ℕ) (total_packages : ℕ) 
    (h1 : num_people = 10) 
    (h2 : weight_per_burger = 2) 
    (h3 : total_packages = 4) : 
    (num_people * weight_per_burger) / total_packages = 5 := 
by 
  sorry

end NUMINAMATH_GPT_ground_beef_per_package_l2394_239459


namespace NUMINAMATH_GPT_floor_S_value_l2394_239441

theorem floor_S_value
  (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (h_ab_squared : a^2 + b^2 = 1458)
  (h_cd_squared : c^2 + d^2 = 1458)
  (h_ac_product : a * c = 1156)
  (h_bd_product : b * d = 1156) :
  (⌊a + b + c + d⌋ = 77) := 
sorry

end NUMINAMATH_GPT_floor_S_value_l2394_239441


namespace NUMINAMATH_GPT_forty_percent_more_than_seventyfive_by_fifty_l2394_239452

def number : ℝ := 312.5

theorem forty_percent_more_than_seventyfive_by_fifty 
    (x : ℝ) 
    (h : 0.40 * x = 0.75 * 100 + 50) : 
    x = number :=
by
  sorry

end NUMINAMATH_GPT_forty_percent_more_than_seventyfive_by_fifty_l2394_239452


namespace NUMINAMATH_GPT_radius_of_larger_ball_l2394_239475

theorem radius_of_larger_ball :
  let V_small : ℝ := (4/3) * Real.pi * (2^3)
  let V_total : ℝ := 6 * V_small
  let R : ℝ := (48:ℝ)^(1/3)
  V_total = (4/3) * Real.pi * (R^3) := 
  by
  sorry

end NUMINAMATH_GPT_radius_of_larger_ball_l2394_239475


namespace NUMINAMATH_GPT_flag_count_l2394_239492

-- Definitions based on the conditions
def colors : ℕ := 3
def stripes : ℕ := 3

-- The main statement
theorem flag_count : colors ^ stripes = 27 :=
by
  sorry

end NUMINAMATH_GPT_flag_count_l2394_239492


namespace NUMINAMATH_GPT_total_points_always_odd_l2394_239458

theorem total_points_always_odd (n : ℕ) (h : n ≥ 1) :
  ∀ k : ℕ, ∃ m : ℕ, m = (2 ^ k * (n + 1) - 1) ∧ m % 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_total_points_always_odd_l2394_239458


namespace NUMINAMATH_GPT_evaluate_expression_l2394_239407

theorem evaluate_expression (x y : ℝ) (P Q : ℝ) 
  (hP : P = x^2 + y^2) 
  (hQ : Q = x - y) : 
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = 4 * x * y / ((x^2 + y^2)^2 - (x - y)^2) :=
by 
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2394_239407


namespace NUMINAMATH_GPT_solve_for_x_l2394_239454

theorem solve_for_x : ∀ x : ℚ, 2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53 → x = 3 / 4 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_solve_for_x_l2394_239454


namespace NUMINAMATH_GPT_flowers_given_l2394_239498

theorem flowers_given (initial_flowers total_flowers flowers_given : ℝ)
  (h1 : initial_flowers = 67)
  (h2 : total_flowers = 157)
  (h3 : total_flowers = initial_flowers + flowers_given) :
  flowers_given = 90 :=
sorry

end NUMINAMATH_GPT_flowers_given_l2394_239498


namespace NUMINAMATH_GPT_not_possible_linear_poly_conditions_l2394_239414

theorem not_possible_linear_poly_conditions (a b : ℝ):
    ¬ (abs (b - 1) < 1 ∧ abs (a + b - 3) < 1 ∧ abs (2 * a + b - 9) < 1) := 
by
    sorry

end NUMINAMATH_GPT_not_possible_linear_poly_conditions_l2394_239414


namespace NUMINAMATH_GPT_loss_percentage_second_venture_l2394_239482

theorem loss_percentage_second_venture 
  (investment_total : ℝ)
  (investment_each : ℝ)
  (profit_percentage_first_venture : ℝ)
  (total_return_percentage : ℝ)
  (L : ℝ) 
  (H1 : investment_total = 25000) 
  (H2 : investment_each = 16250)
  (H3 : profit_percentage_first_venture = 0.15)
  (H4 : total_return_percentage = 0.08)
  (H5 : (investment_total * total_return_percentage) = ((investment_each * profit_percentage_first_venture) - (investment_each * L))) :
  L = 0.0269 := 
by
  sorry

end NUMINAMATH_GPT_loss_percentage_second_venture_l2394_239482


namespace NUMINAMATH_GPT_athlete_last_finish_l2394_239432

theorem athlete_last_finish (v1 v2 v3 : ℝ) (h1 : v1 > v2) (h2 : v2 > v3) :
  let T1 := 1 / v1 + 2 / v2 
  let T2 := 1 / v2 + 2 / v3
  let T3 := 1 / v3 + 2 / v1
  T2 > T1 ∧ T2 > T3 :=
by
  sorry

end NUMINAMATH_GPT_athlete_last_finish_l2394_239432


namespace NUMINAMATH_GPT_rate_2nd_and_3rd_hours_equals_10_l2394_239445

-- Define the conditions as given in the problem
def total_gallons_after_5_hours := 34 
def rate_1st_hour := 8 
def rate_4th_hour := 14 
def water_lost_5th_hour := 8 

-- Problem statement: Prove the rate during 2nd and 3rd hours is 10 gallons/hour
theorem rate_2nd_and_3rd_hours_equals_10 (R : ℕ) :
  total_gallons_after_5_hours = rate_1st_hour + 2 * R + rate_4th_hour - water_lost_5th_hour →
  R = 10 :=
by sorry

end NUMINAMATH_GPT_rate_2nd_and_3rd_hours_equals_10_l2394_239445


namespace NUMINAMATH_GPT_find_savings_l2394_239457

def income : ℕ := 15000
def expenditure (I : ℕ) : ℕ := 4 * I / 5
def savings (I E : ℕ) : ℕ := I - E

theorem find_savings : savings income (expenditure income) = 3000 := 
by
  sorry

end NUMINAMATH_GPT_find_savings_l2394_239457


namespace NUMINAMATH_GPT_same_sign_m_minus_n_opposite_sign_m_plus_n_l2394_239473

-- Definitions and Conditions
noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry

axiom abs_m_eq_4 : |m| = 4
axiom abs_n_eq_3 : |n| = 3

-- Part 1: Prove m - n when m and n have the same sign
theorem same_sign_m_minus_n :
  (m > 0 ∧ n > 0) ∨ (m < 0 ∧ n < 0) → (m - n = 1 ∨ m - n = -1) :=
by
  sorry

-- Part 2: Prove m + n when m and n have opposite signs
theorem opposite_sign_m_plus_n :
  (m > 0 ∧ n < 0) ∨ (m < 0 ∧ n > 0) → (m + n = 1 ∨ m + n = -1) :=
by
  sorry

end NUMINAMATH_GPT_same_sign_m_minus_n_opposite_sign_m_plus_n_l2394_239473


namespace NUMINAMATH_GPT_gumballs_difference_l2394_239442

theorem gumballs_difference :
  ∃ (x_min x_max : ℕ), 
    19 ≤ (16 + 12 + x_min) / 3 ∧ (16 + 12 + x_min) / 3 ≤ 25 ∧
    19 ≤ (16 + 12 + x_max) / 3 ∧ (16 + 12 + x_max) / 3 ≤ 25 ∧
    (x_max - x_min = 18) :=
by
  sorry

end NUMINAMATH_GPT_gumballs_difference_l2394_239442


namespace NUMINAMATH_GPT_parallel_lines_l2394_239481

-- Definitions of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y = 8

-- Definition of parallel lines: slopes are equal and the lines are not identical
def slopes_equal (m : ℝ) : Prop := -(3 + m) / 4 = -2 / (5 + m)
def not_identical_lines (m : ℝ) : Prop := l1 m ≠ l2 m

-- Theorem stating the given conditions
theorem parallel_lines (m : ℝ) (x y : ℝ) : slopes_equal m → not_identical_lines m → m = -7 := by
  sorry

end NUMINAMATH_GPT_parallel_lines_l2394_239481


namespace NUMINAMATH_GPT_Carolina_mailed_five_letters_l2394_239446

-- Definitions translating the given conditions into Lean
def cost_of_mail (cost_letters cost_packages : ℝ) (num_letters num_packages : ℕ) : ℝ :=
  cost_letters * num_letters + cost_packages * num_packages

-- The main theorem to prove the desired answer
theorem Carolina_mailed_five_letters (P L : ℕ)
  (h1 : L = P + 2)
  (h2 : cost_of_mail 0.37 0.88 L P = 4.49) :
  L = 5 := 
sorry

end NUMINAMATH_GPT_Carolina_mailed_five_letters_l2394_239446


namespace NUMINAMATH_GPT_hunter_movies_count_l2394_239460

theorem hunter_movies_count (H : ℕ) 
  (dalton_movies : ℕ := 7)
  (alex_movies : ℕ := 15)
  (together_movies : ℕ := 2)
  (total_movies : ℕ := 30)
  (all_different_movies : dalton_movies + alex_movies - together_movies + H = total_movies) :
  H = 8 :=
by
  -- The mathematical proof will go here
  sorry

end NUMINAMATH_GPT_hunter_movies_count_l2394_239460


namespace NUMINAMATH_GPT_exists_function_l2394_239416

theorem exists_function {n : ℕ} (hn : n ≥ 3) (S : Finset ℤ) (hS : S.card = n) :
  ∃ f : Fin (n) → S, 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ i j k : Fin n, i < j ∧ j < k → 2 * (f j : ℤ) ≠ (f i : ℤ) + (f k : ℤ)) :=
by
  sorry

end NUMINAMATH_GPT_exists_function_l2394_239416


namespace NUMINAMATH_GPT_fido_area_reach_l2394_239444

theorem fido_area_reach (r : ℝ) (s : ℝ) (a b : ℕ) (h1 : r = s * (1 / (Real.tan (Real.pi / 8))))
  (h2 : a = 2) (h3 : b = 8)
  (h_fraction : (Real.pi * r ^ 2) / (2 * (1 + Real.sqrt 2) * (r ^ 2 * (Real.tan (Real.pi / 8)) ^ 2)) = (Real.pi * Real.sqrt a) / b) :
  a * b = 16 := by
  sorry

end NUMINAMATH_GPT_fido_area_reach_l2394_239444


namespace NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_7_divided_9_l2394_239467

theorem largest_integer_less_than_100_with_remainder_7_divided_9 :
  ∃ x : ℕ, (∀ m : ℤ, x = 9 * m + 7 → 9 * m + 7 < 100) ∧ x = 97 :=
sorry

end NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_7_divided_9_l2394_239467


namespace NUMINAMATH_GPT_find_candies_l2394_239466

variable (e : ℝ)

-- Given conditions
def candies_sum (e : ℝ) : ℝ := e + 4 * e + 16 * e + 96 * e

theorem find_candies (h : candies_sum e = 876) : e = 7.5 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_find_candies_l2394_239466


namespace NUMINAMATH_GPT_friends_pay_6_22_l2394_239409

noncomputable def cost_per_friend : ℕ :=
  let hamburgers := 5 * 3
  let fries := 4 * 120 / 100
  let soda := 5 * 50 / 100
  let spaghetti := 270 / 100
  let milkshakes := 3 * 250 / 100
  let nuggets := 2 * 350 / 100
  let total_bill := hamburgers + fries + soda + spaghetti + milkshakes + nuggets
  let discount := total_bill * 10 / 100
  let discounted_bill := total_bill - discount
  let birthday_friend := discounted_bill * 30 / 100
  let remaining_amount := discounted_bill - birthday_friend
  remaining_amount / 4

theorem friends_pay_6_22 : cost_per_friend = 622 / 100 :=
by
  sorry

end NUMINAMATH_GPT_friends_pay_6_22_l2394_239409


namespace NUMINAMATH_GPT_least_xy_value_l2394_239468

theorem least_xy_value {x y : ℕ} (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/8) : x * y = 96 :=
  sorry

end NUMINAMATH_GPT_least_xy_value_l2394_239468


namespace NUMINAMATH_GPT_number_of_blueberries_l2394_239400

def total_berries : ℕ := 42
def raspberries : ℕ := total_berries / 2
def blackberries : ℕ := total_berries / 3
def blueberries : ℕ := total_berries - (raspberries + blackberries)

theorem number_of_blueberries :
  blueberries = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_blueberries_l2394_239400


namespace NUMINAMATH_GPT_find_points_and_min_ordinate_l2394_239478

noncomputable def pi : Real := Real.pi
noncomputable def sin : Real → Real := Real.sin
noncomputable def cos : Real → Real := Real.cos

def within_square (x y : Real) : Prop :=
  -pi ≤ x ∧ x ≤ pi ∧ 0 ≤ y ∧ y ≤ 2 * pi

def satisfies_system (x y : Real) : Prop :=
  sin x + sin y = sin 2 ∧ cos x + cos y = cos 2

theorem find_points_and_min_ordinate :
  ∃ (points : List (Real × Real)), 
    (∀ (p : Real × Real), p ∈ points → within_square p.1 p.2 ∧ satisfies_system p.1 p.2) ∧
    points.length = 2 ∧
    ∃ (min_point : Real × Real), min_point ∈ points ∧ ∀ (p : Real × Real), p ∈ points → min_point.2 ≤ p.2 ∧ min_point = (2 + Real.pi / 3, 2 - Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_points_and_min_ordinate_l2394_239478


namespace NUMINAMATH_GPT_overall_loss_is_450_l2394_239438

noncomputable def total_worth_stock : ℝ := 22499.999999999996

noncomputable def selling_price_20_percent_stock (W : ℝ) : ℝ :=
    0.20 * W * 1.10

noncomputable def selling_price_80_percent_stock (W : ℝ) : ℝ :=
    0.80 * W * 0.95

noncomputable def total_selling_price (W : ℝ) : ℝ :=
    selling_price_20_percent_stock W + selling_price_80_percent_stock W

noncomputable def overall_loss (W : ℝ) : ℝ :=
    W - total_selling_price W

theorem overall_loss_is_450 :
  overall_loss total_worth_stock = 450 := by
  sorry

end NUMINAMATH_GPT_overall_loss_is_450_l2394_239438


namespace NUMINAMATH_GPT_cos_angle_plus_pi_over_two_l2394_239433

theorem cos_angle_plus_pi_over_two (α : ℝ) (h1 : Real.cos α = 1 / 5) (h2 : α ∈ Set.Icc (-2 * Real.pi) (-3 * Real.pi / 2) ∪ Set.Icc (0) (Real.pi / 2)) :
  Real.cos (α + Real.pi / 2) = 2 * Real.sqrt 6 / 5 :=
sorry

end NUMINAMATH_GPT_cos_angle_plus_pi_over_two_l2394_239433


namespace NUMINAMATH_GPT_rowing_upstream_distance_l2394_239499

theorem rowing_upstream_distance (b s d : ℝ) (h_stream_speed : s = 5)
    (h_downstream_distance : 60 = (b + s) * 3)
    (h_upstream_time : d = (b - s) * 3) : 
    d = 30 := by
  have h_b : b = 15 := by
    linarith [h_downstream_distance, h_stream_speed]
  rw [h_b, h_stream_speed] at h_upstream_time
  linarith [h_upstream_time]

end NUMINAMATH_GPT_rowing_upstream_distance_l2394_239499


namespace NUMINAMATH_GPT_benedict_house_size_l2394_239403

variable (K B : ℕ)

theorem benedict_house_size
    (h1 : K = 4 * B + 600)
    (h2 : K = 10000) : B = 2350 := by
sorry

end NUMINAMATH_GPT_benedict_house_size_l2394_239403


namespace NUMINAMATH_GPT_raghu_investment_is_2200_l2394_239456

noncomputable def RaghuInvestment : ℝ := 
  let R := 2200
  let T := 0.9 * R
  let V := 1.1 * T
  if R + T + V = 6358 then R else 0

theorem raghu_investment_is_2200 :
  RaghuInvestment = 2200 := by
  sorry

end NUMINAMATH_GPT_raghu_investment_is_2200_l2394_239456


namespace NUMINAMATH_GPT_simplify_expression_l2394_239406

theorem simplify_expression : 
  (2 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) = 
  Real.sqrt 6 + 2 * Real.sqrt 2 - Real.sqrt 14 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l2394_239406


namespace NUMINAMATH_GPT_advance_agency_fees_eq_8280_l2394_239408

-- Conditions
variables (Commission GivenFees Incentive AdvanceAgencyFees : ℝ)
-- Given values
variables (h_comm : Commission = 25000) 
          (h_given : GivenFees = 18500) 
          (h_incent : Incentive = 1780)

-- The problem statement to prove
theorem advance_agency_fees_eq_8280 
    (h_comm : Commission = 25000) 
    (h_given : GivenFees = 18500) 
    (h_incent : Incentive = 1780)
    : AdvanceAgencyFees = 26780 - GivenFees :=
by
  sorry

end NUMINAMATH_GPT_advance_agency_fees_eq_8280_l2394_239408
