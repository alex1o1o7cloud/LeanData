import Mathlib

namespace ott_fractional_part_l1702_170248

theorem ott_fractional_part (x : ℝ) :
  let moe_initial := 6 * x
  let loki_initial := 5 * x
  let nick_initial := 4 * x
  let ott_initial := 1
  
  let moe_given := (x : ℝ)
  let loki_given := (x : ℝ)
  let nick_given := (x : ℝ)
  
  let ott_returned_each := (1 / 10) * x
  
  let moe_effective := moe_given - ott_returned_each
  let loki_effective := loki_given - ott_returned_each
  let nick_effective := nick_given - ott_returned_each
  
  let ott_received := moe_effective + loki_effective + nick_effective
  let ott_final_money := ott_initial + ott_received
  
  let total_money_original := moe_initial + loki_initial + nick_initial + ott_initial
  let fraction_ott_final := ott_final_money / total_money_original
  
  ott_final_money / total_money_original = (10 + 27 * x) / (150 * x + 10) :=
by
  sorry

end ott_fractional_part_l1702_170248


namespace moles_of_NH4Cl_l1702_170268

-- Define what is meant by "mole" and the substances NH3, HCl, and NH4Cl
def NH3 : Type := ℕ -- Use ℕ to represent moles
def HCl : Type := ℕ
def NH4Cl : Type := ℕ

-- Define the stoichiometry of the reaction
def reaction (n_NH3 n_HCl : ℕ) : ℕ :=
n_NH3 + n_HCl

-- Lean 4 statement: given 1 mole of NH3 and 1 mole of HCl, prove the reaction produces 1 mole of NH4Cl
theorem moles_of_NH4Cl (n_NH3 n_HCl : ℕ) (h1 : n_NH3 = 1) (h2 : n_HCl = 1) : 
  reaction n_NH3 n_HCl = 1 :=
by
  sorry

end moles_of_NH4Cl_l1702_170268


namespace felix_can_lift_150_l1702_170235

-- Define the weights of Felix and his brother.
variables (F B : ℤ)

-- Given conditions
-- Felix's brother can lift three times his weight off the ground, and this amount is 600 pounds.
def brother_lift (B : ℤ) : Prop := 3 * B = 600
-- Felix's brother weighs twice as much as Felix.
def brother_weight (B F : ℤ) : Prop := B = 2 * F
-- Felix can lift off the ground 1.5 times his weight.
def felix_lift (F : ℤ) : ℤ := 3 * F / 2 -- Note: 1.5F can be represented as 3F/2 in Lean for integer operations.

-- Goal: Prove that Felix can lift 150 pounds.
theorem felix_can_lift_150 (F B : ℤ) (h1 : brother_lift B) (h2 : brother_weight B F) : felix_lift F = 150 := by
  dsimp [brother_lift, brother_weight, felix_lift] at *
  sorry

end felix_can_lift_150_l1702_170235


namespace comparison_l1702_170217

noncomputable def a : ℝ := 7 / 9
noncomputable def b : ℝ := 0.7 * Real.exp 0.1
noncomputable def c : ℝ := Real.cos (2 / 3)

theorem comparison : c > a ∧ a > b :=
by
  -- c > a proof
  have h1 : c > a := sorry
  -- a > b proof
  have h2 : a > b := sorry
  exact ⟨h1, h2⟩

end comparison_l1702_170217


namespace intersect_curves_l1702_170264

theorem intersect_curves (R : ℝ) (hR : R > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = R^2 ∧ x - y - 2 = 0) ↔ R ≥ Real.sqrt 2 :=
sorry

end intersect_curves_l1702_170264


namespace Maddie_bought_palettes_l1702_170220

-- Defining constants and conditions as per the problem statement.
def cost_per_palette : ℝ := 15
def number_of_lipsticks : ℝ := 4
def cost_per_lipstick : ℝ := 2.50
def number_of_hair_boxes : ℝ := 3
def cost_per_hair_box : ℝ := 4
def total_paid : ℝ := 67

-- Defining the condition which we need to prove for number of makeup palettes bought.
theorem Maddie_bought_palettes (P : ℝ) :
  (number_of_lipsticks * cost_per_lipstick) +
  (number_of_hair_boxes * cost_per_hair_box) +
  (cost_per_palette * P) = total_paid →
  P = 3 :=
sorry

end Maddie_bought_palettes_l1702_170220


namespace same_number_of_friends_l1702_170253

theorem same_number_of_friends (n : ℕ) (friends : Fin n → Fin n) :
  (∃ i j : Fin n, i ≠ j ∧ friends i = friends j) :=
by
  -- The proof is omitted.
  sorry

end same_number_of_friends_l1702_170253


namespace televisions_sold_this_black_friday_l1702_170283

theorem televisions_sold_this_black_friday 
  (T : ℕ) 
  (h1 : ∀ (n : ℕ), n = 3 → (T + (50 * n) = 477)) 
  : T = 327 := 
sorry

end televisions_sold_this_black_friday_l1702_170283


namespace min_r_minus_p_l1702_170279

theorem min_r_minus_p : ∃ (p q r : ℕ), p * q * r = 362880 ∧ p < q ∧ q < r ∧ (∀ p' q' r' : ℕ, (p' * q' * r' = 362880 ∧ p' < q' ∧ q' < r') → r - p ≤ r' - p') ∧ r - p = 39 :=
by
  sorry

end min_r_minus_p_l1702_170279


namespace amount_spent_l1702_170269

-- Definitions
def initial_amount : ℕ := 54
def amount_left : ℕ := 29

-- Proof statement
theorem amount_spent : initial_amount - amount_left = 25 :=
by
  sorry

end amount_spent_l1702_170269


namespace triangle_side_lengths_inequality_l1702_170232

theorem triangle_side_lengths_inequality
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end triangle_side_lengths_inequality_l1702_170232


namespace fractions_equal_l1702_170221

theorem fractions_equal (a b c d : ℚ) (h1 : a = 2/7) (h2 : b = 3) (h3 : c = 3/7) (h4 : d = 2) :
  a * b = c * d := 
sorry

end fractions_equal_l1702_170221


namespace probability_of_successful_meeting_l1702_170240

noncomputable def meeting_probability : ℚ := 7 / 64

theorem probability_of_successful_meeting :
  (∃ x y z : ℝ,
     0 ≤ x ∧ x ≤ 2 ∧
     0 ≤ y ∧ y ≤ 2 ∧
     0 ≤ z ∧ z ≤ 2 ∧
     abs (x - z) ≤ 0.75 ∧
     abs (y - z) ≤ 1.5 ∧
     z ≥ x ∧
     z ≥ y) →
  meeting_probability = 7 / 64 := by
  sorry

end probability_of_successful_meeting_l1702_170240


namespace opposite_of_2023_is_neg_2023_l1702_170236

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l1702_170236


namespace main_expr_equals_target_l1702_170214

-- Define the improper fractions for the mixed numbers:
def mixed_to_improper (a b : ℕ) (c : ℕ) : ℚ := (a * b + c) / b

noncomputable def mixed_1 := mixed_to_improper 5 7 2
noncomputable def mixed_2 := mixed_to_improper 3 4 3
noncomputable def mixed_3 := mixed_to_improper 4 6 1
noncomputable def mixed_4 := mixed_to_improper 2 5 1

-- Define the main expression
noncomputable def main_expr := 47 * (mixed_1 - mixed_2) / (mixed_3 + mixed_4)

-- Define the target result converted to an improper fraction
noncomputable def target_result : ℚ := (11 * 99 + 13) / 99

-- The theorem to be proved: main_expr == target_result
theorem main_expr_equals_target : main_expr = target_result :=
by sorry

end main_expr_equals_target_l1702_170214


namespace max_value_fraction_l1702_170276

theorem max_value_fraction : ∀ x : ℝ, 
  (∃ x : ℝ, max (1 + (16 / (4 * x^2 + 8 * x + 5))) = 17) :=
by
  sorry

end max_value_fraction_l1702_170276


namespace chord_intersection_eq_l1702_170223

theorem chord_intersection_eq (x y : ℝ) (r : ℝ) : 
  (x + 1)^2 + y^2 = r^2 → 
  (x - 4)^2 + (y - 1)^2 = 4 → 
  (x = 4) → 
  (y = 1) → 
  (r^2 = 26) → (5 * x + y - 19 = 0) :=
by
  sorry

end chord_intersection_eq_l1702_170223


namespace no_roots_impl_a_neg_l1702_170228

theorem no_roots_impl_a_neg {a : ℝ} : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x - 1/x + a ≠ 0) → a < 0 :=
sorry

end no_roots_impl_a_neg_l1702_170228


namespace find_f_50_l1702_170260

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x * y
axiom f_20 : f 20 = 10

theorem find_f_50 : f 50 = 25 :=
by
  sorry

end find_f_50_l1702_170260


namespace find_a_l1702_170224

theorem find_a (a : ℤ) (h : |a + 1| = 3) : a = 2 ∨ a = -4 :=
sorry

end find_a_l1702_170224


namespace smallest_n_geometric_seq_l1702_170281

noncomputable def geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

noncomputable def S_n (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r ^ n) / (1 - r)

theorem smallest_n_geometric_seq :
  (∃ n : ℕ, S_n (1/9) 3 n > 2018) ∧ ∀ m : ℕ, m < 10 → S_n (1/9) 3 m ≤ 2018 :=
by
  sorry

end smallest_n_geometric_seq_l1702_170281


namespace tangential_difference_l1702_170231

noncomputable def tan_alpha_minus_beta (α β : ℝ) : ℝ :=
  Real.tan (α - β)

theorem tangential_difference 
  {α β : ℝ}
  (h : 3 / (2 + Real.sin (2 * α)) + 2021 / (2 + Real.sin β) = 2024) : 
  tan_alpha_minus_beta α β = 1 := 
sorry

end tangential_difference_l1702_170231


namespace find_x_approx_l1702_170222

theorem find_x_approx :
  ∀ (x : ℝ), 3639 + 11.95 - x^2 = 3054 → abs (x - 24.43) < 0.01 :=
by
  intro x
  sorry

end find_x_approx_l1702_170222


namespace number_multiply_increase_l1702_170207

theorem number_multiply_increase (x : ℕ) (h : 25 * x = 25 + 375) : x = 16 := by
  sorry

end number_multiply_increase_l1702_170207


namespace no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l1702_170230

theorem no_20_digit_number_starting_with_11111111111_is_a_perfect_square :
  ¬ ∃ (n : ℤ), (10^19 ≤ n ∧ n < 10^20 ∧ (11111111111 * 10^9 ≤ n ∧ n < 11111111112 * 10^9) ∧ (∃ k : ℤ, n = k^2)) :=
by
  sorry

end no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l1702_170230


namespace garden_ratio_l1702_170267

-- Define the given conditions
def garden_length : ℕ := 100
def garden_perimeter : ℕ := 300

-- Problem statement: Prove the ratio of the length to the width is 2:1
theorem garden_ratio : 
  ∃ (W L : ℕ), 
    L = garden_length ∧ 
    2 * L + 2 * W = garden_perimeter ∧ 
    L / W = 2 :=
by 
  sorry

end garden_ratio_l1702_170267


namespace find_triples_l1702_170270

theorem find_triples (x n p : ℕ) (hp : Nat.Prime p) 
  (hx_pos : x > 0) (hn_pos : n > 0) : 
  x^3 + 3 * x + 14 = 2 * p^n → (x = 1 ∧ n = 2 ∧ p = 3) ∨ (x = 3 ∧ n = 2 ∧ p = 5) :=
by 
  sorry

end find_triples_l1702_170270


namespace minimum_value_of_f_div_f_l1702_170238

noncomputable def quadratic_function_min_value (a b c : ℝ) (h : 0 < b) (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) : ℝ :=
  (a + b + c) / b

theorem minimum_value_of_f_div_f' (a b c : ℝ) (h : 0 < b)
  (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) :
  quadratic_function_min_value a b c h h₀ h₁ h₂ = 2 :=
sorry

end minimum_value_of_f_div_f_l1702_170238


namespace geometric_sequence_a9_l1702_170284

open Nat

theorem geometric_sequence_a9 (a : ℕ → ℝ) (h1 : a 3 = 20) (h2 : a 6 = 5) 
  (h_geometric : ∀ m n, a ((m + n) / 2) ^ 2 = a m * a n) : 
  a 9 = 5 / 4 := 
by
  sorry

end geometric_sequence_a9_l1702_170284


namespace final_amount_H2O_l1702_170272

theorem final_amount_H2O (main_reaction : ∀ (Li3N H2O LiOH NH3 : ℕ), Li3N + 3 * H2O = 3 * LiOH + NH3)
  (side_reaction : ∀ (Li3N LiOH Li2O NH4OH : ℕ), Li3N + LiOH = Li2O + NH4OH)
  (temperature : ℕ) (pressure : ℕ)
  (percentage : ℝ) (init_moles_LiOH : ℕ) (init_moles_Li3N : ℕ)
  (H2O_req_main : ℝ) (H2O_req_side : ℝ) :
  400 = temperature →
  2 = pressure →
  0.05 = percentage →
  9 = init_moles_LiOH →
  3 = init_moles_Li3N →
  H2O_req_main = init_moles_Li3N * 3 →
  H2O_req_side = init_moles_LiOH * percentage →
  H2O_req_main + H2O_req_side = 9.45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end final_amount_H2O_l1702_170272


namespace tank_depth_l1702_170225

open Real

theorem tank_depth :
  ∃ d : ℝ, (0.75 * (2 * 25 * d + 2 * 12 * d + 25 * 12) = 558) ∧ d = 6 :=
sorry

end tank_depth_l1702_170225


namespace base_number_equals_2_l1702_170234

theorem base_number_equals_2 (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^26) (h2 : n = 25) : x = 2 :=
by
  sorry

end base_number_equals_2_l1702_170234


namespace calories_burned_per_week_l1702_170286

-- Definitions of the conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℝ := 1.5
def calories_per_min : ℝ := 7
def minutes_per_hour : ℝ := 60

-- Theorem stating the proof problem
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * minutes_per_hour) * calories_per_min) = 1890 := by
  sorry

end calories_burned_per_week_l1702_170286


namespace value_of_expression_l1702_170226

theorem value_of_expression (n m : ℤ) (h : m = 2 * n^2 + n + 1) : 8 * n^2 - 4 * m + 4 * n - 3 = -7 := by
  sorry

end value_of_expression_l1702_170226


namespace find_k_l1702_170201

theorem find_k 
  (h : ∀ x, 2 * x ^ 2 + 14 * x + k = 0 → x = ((-14 + Real.sqrt 10) / 4) ∨ x = ((-14 - Real.sqrt 10) / 4)) :
  k = 93 / 4 :=
sorry

end find_k_l1702_170201


namespace math_problem_l1702_170250

theorem math_problem : 33333 * 33334 = 1111122222 := 
by sorry

end math_problem_l1702_170250


namespace square_area_l1702_170233

theorem square_area (x : ℝ) (s1 s2 area : ℝ) 
  (h1 : s1 = 5 * x - 21) 
  (h2 : s2 = 36 - 4 * x) 
  (hs : s1 = s2)
  (ha : area = s1 * s1) : 
  area = 113.4225 := 
by
  -- Proof goes here
  sorry

end square_area_l1702_170233


namespace similar_triangle_shortest_side_l1702_170295

theorem similar_triangle_shortest_side 
  (a₁ : ℝ) (b₁ : ℝ) (c₁ : ℝ) (c₂ : ℝ) (k : ℝ)
  (h₁ : a₁ = 15) 
  (h₂ : c₁ = 39) 
  (h₃ : c₂ = 117) 
  (h₄ : k = c₂ / c₁) 
  (h₅ : k = 3) 
  (h₆ : a₂ = a₁ * k) :
  a₂ = 45 := 
by {
  sorry -- proof is not required
}

end similar_triangle_shortest_side_l1702_170295


namespace minimize_shelves_books_l1702_170216

theorem minimize_shelves_books : 
  ∀ (n : ℕ),
    (n > 0 ∧ 130 % n = 0 ∧ 195 % n = 0) → 
    (n ≤ 65) := sorry

end minimize_shelves_books_l1702_170216


namespace question1_question2_case1_question2_case2_question2_case3_l1702_170237

def f (x a : ℝ) : ℝ := x^2 + (1 - a) * x - a

theorem question1 (x : ℝ) (h : (-1 < x) ∧ (x < 3)) : f x 3 < 0 := sorry

theorem question2_case1 (x : ℝ) : f x (-1) > 0 ↔ x ≠ -1 := sorry

theorem question2_case2 (x a : ℝ) (h : a > -1) : f x a > 0 ↔ (x < -1 ∨ x > a) := sorry

theorem question2_case3 (x a : ℝ) (h : a < -1) : f x a > 0 ↔ (x < a ∨ x > -1) := sorry

end question1_question2_case1_question2_case2_question2_case3_l1702_170237


namespace magnet_cost_is_three_l1702_170252

noncomputable def stuffed_animal_cost : ℕ := 6
noncomputable def combined_stuffed_animals_cost : ℕ := 2 * stuffed_animal_cost
noncomputable def magnet_cost : ℕ := combined_stuffed_animals_cost / 4

theorem magnet_cost_is_three : magnet_cost = 3 :=
by
  sorry

end magnet_cost_is_three_l1702_170252


namespace probability_three_one_l1702_170200

-- Definitions based on the conditions
def total_balls : ℕ := 18
def black_balls : ℕ := 10
def white_balls : ℕ := 8
def drawn_balls : ℕ := 4

-- Defining the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the total number of ways to draw 4 balls from 18
def total_ways_to_draw : ℕ := binom total_balls drawn_balls

-- Definition of the number of favorable ways to draw 3 black and 1 white ball
def favorable_black_white : ℕ := binom black_balls 3 * binom white_balls 1

-- Definition of the number of favorable ways to draw 1 black and 3 white balls
def favorable_white_black : ℕ := binom black_balls 1 * binom white_balls 3

-- Total favorable outcomes
def total_favorable_ways : ℕ := favorable_black_white + favorable_white_black

-- The probability of drawing 3 one color and 1 other color
def probability : ℚ := total_favorable_ways / total_ways_to_draw

-- Prove that the probability is 19/38
theorem probability_three_one :
  probability = 19 / 38 :=
sorry

end probability_three_one_l1702_170200


namespace Bianca_pictures_distribution_l1702_170202

theorem Bianca_pictures_distribution 
(pictures_total : ℕ) 
(pictures_in_one_album : ℕ) 
(albums_remaining : ℕ) 
(h1 : pictures_total = 33)
(h2 : pictures_in_one_album = 27)
(h3 : albums_remaining = 3)
: (pictures_total - pictures_in_one_album) / albums_remaining = 2 := 
by 
  sorry

end Bianca_pictures_distribution_l1702_170202


namespace slope_of_intersection_line_is_one_l1702_170242

open Real

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 2 * y + 4 = 0

-- The statement to prove that the slope of the line through the intersection points is 1
theorem slope_of_intersection_line_is_one :
  ∃ m : ℝ, (∀ x y : ℝ, circle1 x y → circle2 x y → (y = m * x + b)) ∧ m = 1 :=
by
  sorry

end slope_of_intersection_line_is_one_l1702_170242


namespace two_subsets_count_l1702_170255

-- Definitions from the problem conditions
def S : Set (Fin 5) := {0, 1, 2, 3, 4}

-- Main statement
theorem two_subsets_count : 
  (∃ A B : Set (Fin 5), A ∪ B = S ∧ A ∩ B = {a, b} ∧ A ≠ B) → 
  (number_of_ways = 40) :=
sorry

end two_subsets_count_l1702_170255


namespace factorize_expression_l1702_170285

theorem factorize_expression (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  sorry

end factorize_expression_l1702_170285


namespace C_share_l1702_170292

-- Definitions based on conditions
def total_sum : ℝ := 164
def ratio_B : ℝ := 0.65
def ratio_C : ℝ := 0.40

-- Statement of the proof problem
theorem C_share : (ratio_C * (total_sum / (1 + ratio_B + ratio_C))) = 32 :=
by
  sorry

end C_share_l1702_170292


namespace sin_A_equals_4_over_5_l1702_170275

variables {A B C : ℝ}
-- Given a right triangle ABC with angle B = 90 degrees
def right_triangle (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (B = 90)

-- We are given 3 * sin(A) = 4 * cos(A)
def given_condition (A : ℝ) : Prop :=
  3 * Real.sin A = 4 * Real.cos A

-- We need to prove that sin(A) = 4/5
theorem sin_A_equals_4_over_5 (A B C : ℝ) 
  (h1 : right_triangle B 90 C)
  (h2 : given_condition A) : 
  Real.sin A = 4 / 5 :=
by
  sorry

end sin_A_equals_4_over_5_l1702_170275


namespace sum_of_powers_l1702_170271

-- Here is the statement in Lean 4
theorem sum_of_powers (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  (ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68) = (ω^2 - 1) / (ω^4 - 1) :=
sorry -- Proof is omitted as per instructions.

end sum_of_powers_l1702_170271


namespace distance_from_unselected_vertex_l1702_170297

-- Define the problem statement
theorem distance_from_unselected_vertex
  (base length : ℝ) (area : ℝ) (h : ℝ) 
  (h_area : area = (base * h) / 2) 
  (h_base : base = 8) 
  (h_area_given : area = 24) : 
  h = 6 :=
by
  -- The proof here is skipped
  sorry

end distance_from_unselected_vertex_l1702_170297


namespace range_of_a_l1702_170289

theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 4) (h3 : a > b) (h4 : b > c) :
  a ∈ Set.Ioo (2 / 3) 2 :=
sorry

end range_of_a_l1702_170289


namespace hire_applicant_A_l1702_170290

-- Define the test scores for applicants A and B
def education_A := 7
def experience_A := 8
def attitude_A := 9

def education_B := 10
def experience_B := 7
def attitude_B := 8

-- Define the weights for the test items
def weight_education := 1 / 6
def weight_experience := 2 / 6
def weight_attitude := 3 / 6

-- Define the final scores
def final_score_A := education_A * weight_education + experience_A * weight_experience + attitude_A * weight_attitude
def final_score_B := education_B * weight_education + experience_B * weight_experience + attitude_B * weight_attitude

-- Prove that Applicant A is hired because their final score is higher
theorem hire_applicant_A : final_score_A > final_score_B :=
by sorry

end hire_applicant_A_l1702_170290


namespace prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l1702_170273

theorem prop1_converse (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b := sorry

theorem prop1_inverse (a b c : ℝ) (h : a ≤ b) : a * c^2 ≤ b * c^2 := sorry

theorem prop1_contrapositive (a b c : ℝ) (h : a * c^2 ≤ b * c^2) : a ≤ b := sorry

theorem prop2_converse (a b c : ℝ) (f : ℝ → ℝ) (h : ∃x, f x = 0) : b^2 - 4 * a * c < 0 := sorry

theorem prop2_inverse (a b c : ℝ) (f : ℝ → ℝ) (h : b^2 - 4 * a * c ≥ 0) : ¬∃x, f x = 0 := sorry

theorem prop2_contrapositive (a b c : ℝ) (f : ℝ → ℝ) (h : ¬∃x, f x = 0) : b^2 - 4 * a * c ≥ 0 := sorry

end prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l1702_170273


namespace max_quadratic_value_l1702_170227

def quadratic (x : ℝ) : ℝ :=
  -2 * x^2 + 4 * x + 3

theorem max_quadratic_value : ∃ x : ℝ, ∀ y : ℝ, quadratic x = y → y ≤ 5 ∧ (∀ z : ℝ, quadratic z ≤ y) := 
by
  sorry

end max_quadratic_value_l1702_170227


namespace math_problem_l1702_170219

variable (a b c m : ℝ)

-- Quadratic equation: y = ax^2 + bx + c
def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Opens downward
axiom a_neg : a < 0
-- Passes through A(1, 0)
axiom passes_A : quadratic a b c 1 = 0
-- Passes through B(m, 0) with -2 < m < -1
axiom passes_B : quadratic a b c m = 0
axiom m_range : -2 < m ∧ m < -1

-- Prove the conclusions
theorem math_problem : b < 0 ∧ (a + b + c = 0) ∧ (a * (m+1) - b + c > 0) ∧ ¬(4 * a * c - b^2 > 4 * a) :=
by
  sorry

end math_problem_l1702_170219


namespace intersection_eq_l1702_170241

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_eq : A ∩ B = {1, 3} :=
by
  sorry

end intersection_eq_l1702_170241


namespace find_n_l1702_170211

-- Defining the parameters and conditions
def large_block_positions (n : ℕ) : ℕ := 199 * n + 110 * (n - 1)

-- Theorem statement
theorem find_n (h : large_block_positions n = 2362) : n = 8 :=
sorry

end find_n_l1702_170211


namespace binary_to_decimal_101101_l1702_170215

theorem binary_to_decimal_101101 : 
  (1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5) = 45 := 
by 
  sorry

end binary_to_decimal_101101_l1702_170215


namespace value_of_a_l1702_170206

theorem value_of_a (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x - 1 = 0) ∧ (∀ x y : ℝ, a * x^2 - 2 * x - 1 = 0 ∧ a * y^2 - 2 * y - 1 = 0 → x = y) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end value_of_a_l1702_170206


namespace find_f_5_l1702_170282

-- Define the function f satisfying the given conditions
noncomputable def f : ℝ → ℝ :=
sorry

-- Assert the conditions as hypotheses
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x + f y
axiom f_zero : f 0 = 2

-- State the theorem we need to prove
theorem find_f_5 : f 5 = 1 :=
sorry

end find_f_5_l1702_170282


namespace smallest_positive_debt_l1702_170277

theorem smallest_positive_debt :
  ∃ (p g : ℤ), 25 = 250 * p + 175 * g :=
by
  sorry

end smallest_positive_debt_l1702_170277


namespace juvy_chives_l1702_170258

-- Definitions based on the problem conditions
def total_rows : Nat := 20
def plants_per_row : Nat := 10
def parsley_rows : Nat := 3
def rosemary_rows : Nat := 2
def chive_rows : Nat := total_rows - (parsley_rows + rosemary_rows)

-- The statement we want to prove
theorem juvy_chives : chive_rows * plants_per_row = 150 := by
  sorry

end juvy_chives_l1702_170258


namespace find_b_l1702_170208

theorem find_b (b n : ℝ) (h_neg : b < 0) :
  (∀ x, x^2 + b * x + 1 / 4 = (x + n)^2 + 1 / 18) → b = - (Real.sqrt 7) / 3 :=
by
  sorry

end find_b_l1702_170208


namespace more_stable_performance_l1702_170218

theorem more_stable_performance (s_A_sq s_B_sq : ℝ) (hA : s_A_sq = 0.25) (hB : s_B_sq = 0.12) : s_A_sq > s_B_sq :=
by
  rw [hA, hB]
  sorry

end more_stable_performance_l1702_170218


namespace jim_age_in_2_years_l1702_170239

theorem jim_age_in_2_years (c1 : ∀ t : ℕ, t = 37) (c2 : ∀ j : ℕ, j = 27) : ∀ j2 : ℕ, j2 = 29 :=
by
  sorry

end jim_age_in_2_years_l1702_170239


namespace cone_lateral_area_and_sector_area_l1702_170261

theorem cone_lateral_area_and_sector_area 
  (slant_height : ℝ) 
  (height : ℝ) 
  (r : ℝ) 
  (h_slant : slant_height = 1) 
  (h_height : height = 0.8) 
  (h_r : r = Real.sqrt (slant_height^2 - height^2)) :
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) ∧
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) :=
by
  sorry

end cone_lateral_area_and_sector_area_l1702_170261


namespace susans_coins_worth_l1702_170246

theorem susans_coins_worth :
  ∃ n d : ℕ, n + d = 40 ∧ (5 * n + 10 * d) = 230 ∧ (10 * n + 5 * d) = 370 :=
sorry

end susans_coins_worth_l1702_170246


namespace equivalent_polar_coordinates_l1702_170278

-- Definitions of given conditions and the problem statement
def polar_point_neg (r : ℝ) (θ : ℝ) : Prop := r = -3 ∧ θ = 5 * Real.pi / 6
def polar_point_pos (r : ℝ) (θ : ℝ) : Prop := r = 3 ∧ θ = 11 * Real.pi / 6
def angle_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

theorem equivalent_polar_coordinates :
  ∃ (r θ : ℝ), polar_point_neg r θ → polar_point_pos 3 (11 * Real.pi / 6) ∧ angle_range (11 * Real.pi / 6) :=
by
  sorry

end equivalent_polar_coordinates_l1702_170278


namespace problem_statement_l1702_170213

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := 2⁻¹
noncomputable def c : ℝ := Real.log 6 / Real.log 5

theorem problem_statement : b < a ∧ a < c := by
  sorry

end problem_statement_l1702_170213


namespace determinant_of_sum_l1702_170256

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 6], ![2, 3]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 1], ![1, 0]]

theorem determinant_of_sum : (A + B).det = -3 := 
by 
  sorry

end determinant_of_sum_l1702_170256


namespace peanuts_in_box_l1702_170204

theorem peanuts_in_box (original_peanuts added_peanuts total_peanuts : ℕ) (h1 : original_peanuts = 10) (h2 : added_peanuts = 8) (h3 : total_peanuts = original_peanuts + added_peanuts) : total_peanuts = 18 := 
by {
  sorry
}

end peanuts_in_box_l1702_170204


namespace mrs_santiago_more_roses_l1702_170205

theorem mrs_santiago_more_roses :
  58 - 24 = 34 :=
by 
  sorry

end mrs_santiago_more_roses_l1702_170205


namespace total_area_pool_and_deck_l1702_170245

theorem total_area_pool_and_deck (pool_length pool_width deck_width : ℕ) 
  (h1 : pool_length = 12) 
  (h2 : pool_width = 10) 
  (h3 : deck_width = 4) : 
  (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width) = 360 := 
by sorry

end total_area_pool_and_deck_l1702_170245


namespace number_of_planes_l1702_170229

-- Definitions based on the conditions
def Line (space: Type) := space → space → Prop

variables {space: Type} [MetricSpace space]

-- Given conditions
variable (l1 l2 l3 : Line space)
variable (intersects : ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p)

-- The theorem stating the conclusion
theorem number_of_planes (h: ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p) :
  (1 = 1 ∨ 1 = 2 ∨ 1 = 3) ∨ (2 = 1 ∨ 2 = 2 ∨ 2 = 3) ∨ (3 = 1 ∨ 3 = 2 ∨ 3 = 3) := 
sorry

end number_of_planes_l1702_170229


namespace minimize_y_l1702_170299

noncomputable def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + 3 * x + 5

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) → x = (2 * a + 2 * b - 3) / 4 := by
  sorry

end minimize_y_l1702_170299


namespace arithmetic_sequence_third_term_l1702_170298

theorem arithmetic_sequence_third_term (b y : ℝ) 
  (h1 : 2 * b + y + 2 = 10) 
  (h2 : b + y + 2 = b + y + 2) : 
  8 - b = 6 := 
by 
  sorry

end arithmetic_sequence_third_term_l1702_170298


namespace most_likely_units_digit_l1702_170293

theorem most_likely_units_digit :
  ∃ m n : Fin 11, ∀ (M N : Fin 11), (∃ k : Nat, k * 11 + M + N = m + n) → 
    (m + n) % 10 = 0 :=
by
  sorry

end most_likely_units_digit_l1702_170293


namespace other_person_age_l1702_170263

variable {x : ℕ} -- age of the other person
variable {y : ℕ} -- Marco's age

-- Conditions given in the problem.
axiom marco_age : y = 2 * x + 1
axiom sum_ages : x + y = 37

-- Goal: Prove that the age of the other person is 12.
theorem other_person_age : x = 12 :=
by
  -- Proof is skipped
  sorry

end other_person_age_l1702_170263


namespace george_total_coins_l1702_170209

-- We'll state the problem as proving the total number of coins George has.
variable (num_nickels num_dimes : ℕ)
variable (value_of_coins : ℝ := 2.60)
variable (value_of_nickels : ℝ := 0.05 * num_nickels)
variable (value_of_dimes : ℝ := 0.10 * num_dimes)

theorem george_total_coins :
  num_nickels = 4 → 
  value_of_coins = value_of_nickels + value_of_dimes → 
  num_nickels + num_dimes = 28 := 
by
  sorry

end george_total_coins_l1702_170209


namespace aaron_age_l1702_170247

variable (A : ℕ)
variable (henry_sister_age : ℕ)
variable (henry_age : ℕ)
variable (combined_age : ℕ)

theorem aaron_age (h1 : henry_sister_age = 3 * A)
                 (h2 : henry_age = 4 * henry_sister_age)
                 (h3 : combined_age = henry_sister_age + henry_age)
                 (h4 : combined_age = 240) : A = 16 := by
  sorry

end aaron_age_l1702_170247


namespace second_term_of_series_l1702_170262

noncomputable def geometric_series_second_term (a r S : ℝ) := r * a

theorem second_term_of_series (a r : ℝ) (S : ℝ) (hr : r = 1/4) (hs : S = 16) 
  (hS_formula : S = a / (1 - r)) : geometric_series_second_term a r S = 3 :=
by
  -- Definitions are in place, applying algebraic manipulation steps here would follow
  sorry

end second_term_of_series_l1702_170262


namespace students_count_l1702_170265

theorem students_count (initial: ℕ) (left: ℕ) (new: ℕ) (result: ℕ) 
  (h1: initial = 31)
  (h2: left = 5)
  (h3: new = 11)
  (h4: result = initial - left + new) : result = 37 := by
  sorry

end students_count_l1702_170265


namespace minimum_arc_length_of_curve_and_line_l1702_170266

-- Definition of the curve C and the line x = π/4
def curve (x y α : ℝ) : Prop :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line (x : ℝ) : Prop :=
  x = Real.pi / 4

-- Statement of the proof problem: the minimum value of d as α varies
theorem minimum_arc_length_of_curve_and_line : 
  (∀ α : ℝ, ∃ d : ℝ, (∃ y : ℝ, curve (Real.pi / 4) y α) → 
    (d = Real.pi / 2)) :=
sorry

end minimum_arc_length_of_curve_and_line_l1702_170266


namespace principal_amount_l1702_170212

variable (P : ℝ)

/-- Prove the principal amount P given that the simple interest at 4% for 5 years is Rs. 2400 less than the principal --/
theorem principal_amount : 
  (4/100) * P * 5 = P - 2400 → 
  P = 3000 := 
by 
  sorry

end principal_amount_l1702_170212


namespace equal_angles_proof_l1702_170257

/-- Proof Problem: After how many minutes will the hour and minute hands form equal angles with their positions at 12 o'clock? -/
noncomputable def equal_angle_time (x : ℝ) : Prop :=
  -- Defining the conditions for the problem
  let minute_hand_speed := 6 -- degrees per minute
  let hour_hand_speed := 0.5 -- degrees per minute
  let total_degrees := 360 * x -- total degrees of minute hand till time x
  let hour_hand_degrees := 30 * (x / 60) -- total degrees of hour hand till time x

  -- Equation for equal angles formed with respect to 12 o'clock
  30 * (x / 60) = 360 - 360 * (x / 60)

theorem equal_angles_proof :
  ∃ (x : ℝ), equal_angle_time x ∧ x = 55 + 5/13 :=
sorry

end equal_angles_proof_l1702_170257


namespace union_A_B_intersection_complements_l1702_170254
open Set

noncomputable def A : Set ℤ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B : Set ℤ := {x | x^2 - x - 2 = 0}
def U : Set ℤ := {x | abs x ≤ 3}

theorem union_A_B :
  A ∪ B = { -1, 2, 3 } :=
by sorry

theorem intersection_complements :
  (U \ A) ∩ (U \ B) = { -3, -2, 0, 1 } :=
by sorry

end union_A_B_intersection_complements_l1702_170254


namespace canoe_problem_l1702_170294

-- Definitions:
variables (P_L P_R : ℝ)

-- Conditions:
def conditions := 
  (P_L = P_R) ∧ -- Condition that the probabilities for left and right oars working are the same
  (0 ≤ P_L) ∧ (P_L ≤ 1) ∧ -- Probability values must be between 0 and 1
  (1 - (1 - P_L) * (1 - P_R) = 0.84) -- Given the rowing probability is 0.84

-- Theorem that P_L = 0.6 given the conditions:
theorem canoe_problem : conditions P_L P_R → P_L = 0.6 :=
by
  sorry

end canoe_problem_l1702_170294


namespace integral_cos_plus_one_l1702_170203

theorem integral_cos_plus_one :
  ∫ x in - (Real.pi / 2).. (Real.pi / 2), (1 + Real.cos x) = Real.pi + 2 :=
by
  sorry

end integral_cos_plus_one_l1702_170203


namespace sack_flour_cost_l1702_170280

theorem sack_flour_cost
  (x y : ℝ) 
  (h1 : 10 * x + 800 = 108 * y)
  (h2 : 4 * x - 800 = 36 * y) : x = 1600 := by
  -- Add your proof here
  sorry

end sack_flour_cost_l1702_170280


namespace final_speed_train_l1702_170287

theorem final_speed_train
  (u : ℝ) (a : ℝ) (t : ℕ) :
  u = 0 → a = 1 → t = 20 → u + a * t = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end final_speed_train_l1702_170287


namespace james_added_8_fish_l1702_170296

theorem james_added_8_fish
  (initial_fish : ℕ := 60)
  (fish_eaten_per_day : ℕ := 2)
  (total_days_with_worm : ℕ := 21)
  (fish_remaining_when_discovered : ℕ := 26) :
  ∃ (additional_fish : ℕ), additional_fish = 8 :=
by
  let total_fish_eaten := total_days_with_worm * fish_eaten_per_day
  let fish_remaining_without_addition := initial_fish - total_fish_eaten
  let additional_fish := fish_remaining_when_discovered - fish_remaining_without_addition
  exact ⟨additional_fish, sorry⟩

end james_added_8_fish_l1702_170296


namespace find_subtracted_number_l1702_170251

theorem find_subtracted_number 
  (a : ℕ) (b : ℕ) (g : ℕ) (n : ℕ) 
  (h1 : a = 2) 
  (h2 : b = 3 * a) 
  (h3 : g = 2 * b - n) 
  (h4 : g = 8) : n = 4 :=
by 
  sorry

end find_subtracted_number_l1702_170251


namespace probability_same_color_is_correct_l1702_170243

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l1702_170243


namespace function_identity_l1702_170288

theorem function_identity (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, f m + f n ∣ m + n) : ∀ m : ℕ+, f m = m := by
  sorry

end function_identity_l1702_170288


namespace arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l1702_170210

-- Definitions based on conditions
def performances : Nat := 8
def singing : Nat := 2
def dance : Nat := 3
def variety : Nat := 3

-- Problem 1: Prove arrangement with a singing program at the beginning and end
theorem arrange_singing_begin_end : 1440 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 2: Prove arrangement with singing programs not adjacent
theorem arrange_singing_not_adjacent : 30240 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 3: Prove arrangement with singing programs adjacent and dance not adjacent
theorem arrange_singing_adjacent_dance_not_adjacent : 2880 = sorry :=
by
  -- proof goes here
  sorry

end arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l1702_170210


namespace find_a4_l1702_170274

noncomputable def quadratic_eq (t : ℝ) := t^2 - 36 * t + 288 = 0

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∃ a1 : ℝ, a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def condition1 (a : ℕ → ℝ) := a 1 + a 2 = -1
def condition2 (a : ℕ → ℝ) := a 1 - a 3 = -3

theorem find_a4 :
  ∃ (a : ℕ → ℝ) (q : ℝ), quadratic_eq q ∧ geometric_sequence a q ∧ condition1 a ∧ condition2 a ∧ a 4 = -8 :=
by
  sorry

end find_a4_l1702_170274


namespace sufficient_but_not_necessary_condition_l1702_170244

variable {a : ℝ}

theorem sufficient_but_not_necessary_condition :
  (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (a ≥ 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1702_170244


namespace first_car_made_earlier_l1702_170249

def year_first_car : ℕ := 1970
def year_third_car : ℕ := 2000
def diff_third_second : ℕ := 20

theorem first_car_made_earlier : (year_third_car - diff_third_second) - year_first_car = 10 := by
  sorry

end first_car_made_earlier_l1702_170249


namespace money_last_duration_l1702_170259

-- Defining the conditions
def money_from_mowing : ℕ := 14
def money_from_weed_eating : ℕ := 26
def money_spent_per_week : ℕ := 5

-- Theorem statement to prove Mike's money will last 8 weeks
theorem money_last_duration : (money_from_mowing + money_from_weed_eating) / money_spent_per_week = 8 := by
  sorry

end money_last_duration_l1702_170259


namespace kyler_games_won_l1702_170291

theorem kyler_games_won (peter_wins peter_losses emma_wins emma_losses kyler_losses : ℕ)
  (h_peter : peter_wins = 5)
  (h_peter_losses : peter_losses = 4)
  (h_emma : emma_wins = 2)
  (h_emma_losses : emma_losses = 5)
  (h_kyler_losses : kyler_losses = 4) : ∃ kyler_wins : ℕ, kyler_wins = 2 :=
by {
  sorry
}

end kyler_games_won_l1702_170291
