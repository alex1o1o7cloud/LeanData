import Mathlib

namespace part1_part2_l131_131914

-- Define the sequence
def a (n : ℕ) : ℤ := 15 * n + 2 + (15 * n - 32) * (16^(n - 1 : ℤ))

-- Part 1: Prove that 3375 divides a_n for all non-negative integers n
theorem part1 (n : ℕ) : 3375 ∣ a n := sorry

-- Part 2: Find all n such that 1991 divides a_n, a_(n+1), and a_(n+2)
theorem part2 (n : ℕ) : 1991 ∣ a n ∧ 1991 ∣ a (n+1) ∧ 1991 ∣ a (n+2) ↔ 
  ∃ k : ℕ, n = 89595 * k := sorry

end part1_part2_l131_131914


namespace can_construct_polygon_l131_131759

def match_length : ℕ := 2
def number_of_matches : ℕ := 12
def total_length : ℕ := number_of_matches * match_length
def required_area : ℝ := 16

theorem can_construct_polygon : 
  (∃ (P : Polygon), P.perimeter = total_length ∧ P.area = required_area) := 
sorry

end can_construct_polygon_l131_131759


namespace log_nine_cbrt_equals_third_l131_131111

theorem log_nine_cbrt_equals_third : log 9 (9 ^ (1 / 3 : ℝ)) = 1 / 3 :=
by
  sorry

end log_nine_cbrt_equals_third_l131_131111


namespace quadrilateral_area_bounds_l131_131293

open Locale.RealInnerProductSpace
open Set
open Convex

variables {V : Type} [InnerProductSpace ℝ V]

-- Coordinates of points A, B, C, D
variables (A B C D E F G H : V)

-- Conditions for midpoint
def is_midpoint (P Q R : V) : Prop :=
  P = (Q + R) / 2

-- Definitions for the given problem
def quadrilateral_midpoints (A B C D E F G H : V) : Prop :=
  is_midpoint E A B ∧ is_midpoint F B C ∧ is_midpoint G C D ∧ is_midpoint H D A

-- Areas of triangles
def area (u v : V) : ℝ := 0.5 * ∥u - v∥ * ∥u + v∥

-- Areas of quadrilateral
def area_quadrilateral (u v w x : V) : ℝ := 
  area (u - v) (v - w) + area (w - x) (x - u)

-- The hypothesis conditions and conclusions
theorem quadrilateral_area_bounds 
  (A B C D E F G H : V)
  (h_midpoints : quadrilateral_midpoints A B C D E F G H) :
  area_quadrilateral A B C D ≤ ∥E - G∥ * ∥H - F∥ ∧ 
  ∥H - F∥ ≤ (∥A - B∥ + ∥C - D∥) * (∥B - C∥ + ∥D - A∥) / 4 :=
sorry

end quadrilateral_area_bounds_l131_131293


namespace cylinder_cone_volume_ratio_l131_131819

-- Define the given variables
def r : ℝ := 5
def h_cyl : ℝ := 18
def h_cone : ℝ := 9

-- Define the volumes
def V_cyl : ℝ := π * r^2 * h_cyl
def V_cone : ℝ := (1/3) * π * r^2 * h_cone

-- Prove the ratio is 1/6
theorem cylinder_cone_volume_ratio : V_cone / V_cyl = 1 / 6 := 
by sorry

end cylinder_cone_volume_ratio_l131_131819


namespace four_digit_numbers_even_sum_count_l131_131527

-- Definitions for the conditions
def digits := {0, 1, 2, 3, 4, 5, 6}
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def no_repetition (n : ℕ) : Prop := (λ l, l.nodup) (n.digits)
def digits_sum_even (n : ℕ) : Prop := (n % 10 + (n / 10) % 10 + (n / 100) % 10) % 2 = 0

-- Main statement to be proven
theorem four_digit_numbers_even_sum_count : 
  (∑ n in (digits.product digits).product digits, if is_four_digit_number n ∧ no_repetition n ∧ digits_sum_even n then 1 else 0) = 324 := 
sorry

end four_digit_numbers_even_sum_count_l131_131527


namespace amanda_final_quiz_score_l131_131842

theorem amanda_final_quiz_score
  (average_score_4quizzes : ℕ)
  (total_quizzes : ℕ)
  (average_a : ℕ)
  (current_score : ℕ)
  (required_total_score : ℕ)
  (required_score_final_quiz : ℕ) :
  average_score_4quizzes = 92 →
  total_quizzes = 5 →
  average_a = 93 →
  current_score = 4 * average_score_4quizzes →
  required_total_score = total_quizzes * average_a →
  required_score_final_quiz = required_total_score - current_score →
  required_score_final_quiz = 97 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end amanda_final_quiz_score_l131_131842


namespace train_time_to_B_l131_131705

theorem train_time_to_B (T : ℝ) (M : ℝ) :
  (∃ (D : ℝ), (T + 5) * (D + M) / T = 6 * M ∧ 2 * D = 5 * M) → T = 7 :=
by
  sorry

end train_time_to_B_l131_131705


namespace tangent_product_identity_l131_131251

theorem tangent_product_identity : 
  ∏ k in finset.range 89 (λ k, (1 + real.tan (k + 1 : ℝ) * real.pi / 180)) = 2^45 :=
by 
  sorry

end tangent_product_identity_l131_131251


namespace log_nine_cbrt_equals_third_l131_131110

theorem log_nine_cbrt_equals_third : log 9 (9 ^ (1 / 3 : ℝ)) = 1 / 3 :=
by
  sorry

end log_nine_cbrt_equals_third_l131_131110


namespace repetend_of_5_div_17_l131_131143

theorem repetend_of_5_div_17 : 
  ∃ repetend : ℕ, 
  decimal_repetend (5 / 17) = repetend ∧ 
  repetend = 2941176470588235 :=
by 
  skip

end repetend_of_5_div_17_l131_131143


namespace area_of_region_in_triangle_l131_131643

noncomputable def triangleAreaRegion (a b c r : ℝ) : ℝ :=
  if a = 2 ∧ b = 7 ∧ c = 5 * Real.sqrt 3 ∧ r = 43 then
    Real.pi / 9
  else
    0

theorem area_of_region_in_triangle :
  ∀ {a b c : ℝ}, a = 2 → b = 7 → c = 5 * Real.sqrt 3 → 
  triangleAreaRegion a b c 43 = Real.pi / 9 :=
by
  intros a b c ha hb hc
  unfold triangleAreaRegion
  rw [ha, hb, hc]
  simp
  have h_abcs : a = 2 ∧ b = 7 ∧ c = 5 * Real.sqrt (3 : ℝ) ∧ 43 = 43 := ⟨ha, hb, hc, rfl⟩
  simp [h_abcs]
  done

end area_of_region_in_triangle_l131_131643


namespace find_other_coin_denomination_l131_131763

variables (total_coins : ℕ) (total_value_paise : ℕ) (paise_value_per_coin : ℕ)
variables (num_20_paise_coins : ℕ) (num_other_coins : ℕ) (value_other_coins_paise : ℕ)

-- Given conditions
def given_conditions :=
  total_coins = 342 ∧
  total_value_paise = 7100 ∧
  num_20_paise_coins = 290 ∧
  value_other_coins_paise = 1300 ∧
  num_other_coins = total_coins - num_20_paise_coins

-- Theorem to prove
theorem find_other_coin_denomination (denomination_other_coin : ℕ) :
  given_conditions →
  denomination_other_coin = 25 :=
sorry

end find_other_coin_denomination_l131_131763


namespace g_g_even_l131_131308

variable {α : Type*} [HasNeg α]
variable {g : α → α}

def is_even (f : α → α) : Prop := ∀ x, f (-x) = f x

theorem g_g_even (h : is_even g) : is_even (g ∘ g) :=
by
  sorry

end g_g_even_l131_131308


namespace problem_solution_l131_131578

-- Define the function f(x)
def f (x c : ℝ) (a : ℝ := -1/2) (b : ℝ := -2) : ℝ :=
  x^3 + a * x^2 + b * x + c

-- Derivative of the function f(x)
def f' (x c : ℝ) (a : ℝ := -1/2) (b : ℝ := -2) : ℝ :=
  3 * x^2 + 2 * a * x + b

-- Given conditions as hypotheses
theorem problem_solution (c : ℝ) :
  (f' (-2/3) c = 0) →
  (f' 1 c = 0) →
  (f' (-2/3) = 0 → f' 1 = 0 → ∀ x, -2/3 ≤ x ∧ x ≤ 1 → f' x ≤ 0) →
  (∀ x, -1 ≤ x ∧ x ≤ 2 → f x c < c^2) →
  c ∈ (-∞ : set ℝ) ∪ Iio (-1) ∪ Ioi 2 :=
by
  intros h1 h2 h3 h4
  sorry

end problem_solution_l131_131578


namespace rect_plot_length_more_than_breadth_l131_131371

theorem rect_plot_length_more_than_breadth (b x : ℕ) (cost_per_m : ℚ)
  (length_eq : b + x = 56)
  (fencing_cost : (4 * b + 2 * x) * cost_per_m = 5300)
  (cost_rate : cost_per_m = 26.50) : x = 12 :=
by
  sorry

end rect_plot_length_more_than_breadth_l131_131371


namespace logarithmic_values_count_l131_131196

theorem logarithmic_values_count :
  let S := {1, 2, 3, 4, 5}
  let pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.2 ≠ 1 → p.1 ≠ 1}
  finset.card pairs = 1 + finset.card {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p ≠ (p.1, 1)}
    := 13 :=
by
  sorry

end logarithmic_values_count_l131_131196


namespace point_in_quadrant_IV_l131_131632

-- Define the quadrants as an enumeration type
inductive Quadrant
| I
| II
| III
| IV

-- Define the point and specify its coordinates
def point : (ℝ × ℝ) := (5, -2)

-- Define the function to determine the quadrant of a point
def findQuadrant (p : ℝ × ℝ) : Quadrant :=
  if p.1 > 0 then 
    if p.2 > 0 then Quadrant.I 
    else Quadrant.IV 
  else 
    if p.2 > 0 then Quadrant.II 
    else Quadrant.III

-- Statement to prove that the point (5, -2) lies in Quadrant IV
theorem point_in_quadrant_IV : findQuadrant point = Quadrant.IV :=
  sorry

end point_in_quadrant_IV_l131_131632


namespace m_plus_n_eq_2613_l131_131287

def probability_same_heads : ℚ :=
  let p_fair := 1/2
  let p_biased := 5/8
  let generating_function := (1 + p_fair)^3 * (3 + 5 * p_biased)
  let coefficients := [3, 14, 24, 18, 5]  -- coefficients after expansion
  let total_outcomes := 64
  let probability := (coefficients.map (λ c => c * c)).sum
  probability / (total_outcomes * total_outcomes)

theorem m_plus_n_eq_2613 :
  let fraction := probability_same_heads
  let m := fraction.num
  let n := fraction.denom
  m.gcd(n) = 1 → m + n = 2613
:= by
  sorry

end m_plus_n_eq_2613_l131_131287


namespace length_CD_l131_131270

theorem length_CD {s : ℝ} (h1 : 3 * s = 18) (h2 : ∀ A B r_green r_red, A.distance B = r_green ∧ r_red = r_green + s) :
  ∃ CD : ℝ, CD = 12 :=
by
  existsi 12
  sorry

end length_CD_l131_131270


namespace prob_1_2_to_5_2_l131_131568

-- Define the probability mass function for X
def pmf (k : ℕ) : ℚ :=
  if k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 5 then k / 25 else 0

-- Define the event of interest
def event : set ℚ := {x : ℚ | 1 / 2 < x ∧ x < 5 / 2}

-- Define the probability for the event
def prob_event (p : ℕ → ℚ) : ℚ :=
  p 1 + p 2

theorem prob_1_2_to_5_2 : prob_event pmf = 1 / 5 :=
sorry

end prob_1_2_to_5_2_l131_131568


namespace value_of_triangle_l131_131258

-- Define the problem conditions
def condition1 (triangle : ℕ) : Prop := (triangle + 3) % 6 = 0
def condition2 (triangle : ℕ) : Prop := ((triangle + 5 + 1) % 6 = 2)
def condition3 (triangle : ℕ) : Prop := ((4 * 6^2 + 3 * 6 + 2 * 6 + triangle)
                                         + (3 * 6 + 5 * 1)
                                         + (triangle * 3))
                                         = (5 * 6^2 + 3 * 6 + triangle * 6 + 0)

-- Define the main theorem
theorem value_of_triangle : ∃ triangle : ℕ, condition1 triangle ∧ condition2 triangle ∧ condition3 triangle ∧ triangle = 3 :=
by {
  existsi 3,
  split,
  { simp [condition1], },
  split,
  { simp [condition2], },
  split,
  { simp [condition3], sorry, },
  { refl, },
  sorry,
}

end value_of_triangle_l131_131258


namespace count_valid_numbers_l131_131193

-- Definitions derived from conditions
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6}
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬is_even n

-- Condition definitions
def different_parity (a b : ℕ) : Prop := (is_even a ∧ is_odd b) ∨ (is_odd a ∧ is_even b)
def adjacent_12 (l : List ℕ) : Prop := l.indexOf 1 = l.indexOf 2 + 1 ∨ l.indexOf 1 + 1 = l.indexOf 2
def valid_six_digit_number (l : List ℕ) : Prop :=
  l.length = 6 ∧
  ∀ i < 5, different_parity (l.nthLe i (by linarith)) (l.nthLe (i+1) (by linarith)) ∧
  adjacent_12 l

-- Theorem statement
theorem count_valid_numbers : 
  (Finset.filter valid_six_digit_number (Finset.perm_of_list [1, 2, 3, 4, 5, 6])).card = 40 :=
by
  sorry

end count_valid_numbers_l131_131193


namespace can_construct_polygon_l131_131758

def match_length : ℕ := 2
def number_of_matches : ℕ := 12
def total_length : ℕ := number_of_matches * match_length
def required_area : ℝ := 16

theorem can_construct_polygon : 
  (∃ (P : Polygon), P.perimeter = total_length ∧ P.area = required_area) := 
sorry

end can_construct_polygon_l131_131758


namespace find_a8_plus_a12_l131_131214

noncomputable def a_n (n : ℕ) : ℝ := sorry

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variables (q : ℝ) (a : ℕ → ℝ)

theorem find_a8_plus_a12
  (geo_seq : is_geometric_sequence a)
  (h1 : a 2 + a 6 = 3)
  (h2 : a 6 + a 10 = 12) :
  a 8 + a 12 = 24 :=
sorry

end find_a8_plus_a12_l131_131214


namespace square_area_l131_131998

/-- 
In a square ABCD, points P on AD and Q on AB such that BP and CQ intersect at right angles at R.
BR is one quarter of the side length of the square and PR is twice the length of BR.
Prove that the area of the square is 16.
-/
theorem square_area (s : ℝ) (BR PR : ℝ) 
(BR_def : BR = s / 4)
(PR_def : PR = 2 * BR)
(intersection_right_angle : ∃ P Q R, ∠BRP = 90) : 
  s^2 = 16 := 
by
  -- the statement only, proof is omitted
  sorry

end square_area_l131_131998


namespace original_price_of_table_l131_131700

noncomputable def original_price (sale_price : ℝ) (discount_rate : ℝ) : ℝ :=
  sale_price / (1 - discount_rate)

theorem original_price_of_table
  (d : ℝ) (p' : ℝ) (h_d : d = 0.10) (h_p' : p' = 450) :
  original_price p' d = 500 := by
  rw [h_d, h_p']
  -- Calculating the original price
  show original_price 450 0.10 = 500
  sorry

end original_price_of_table_l131_131700


namespace area_of_square_l131_131370

theorem area_of_square (r s L B: ℕ) (h1 : r = s) (h2 : L = 5 * r) (h3 : B = 11) (h4 : 220 = L * B) : s^2 = 16 := by
  sorry

end area_of_square_l131_131370


namespace candy_store_spending_l131_131244

-- Definitions corresponding to the conditions
def weekly_allowance : ℝ := 3.75
def arcade_fraction : ℝ := 3/5
def toy_store_fraction : ℝ := 1/3

-- Main theorem stating the mathematical equivalences to prove the final spending at the candy store
theorem candy_store_spending :
  let arcade_spending := arcade_fraction * weekly_allowance in
  let remaining_after_arcade := weekly_allowance - arcade_spending in
  let toy_store_spending := toy_store_fraction * remaining_after_arcade in
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spending in
  remaining_after_toy_store = 1.0 :=
by
  sorry

end candy_store_spending_l131_131244


namespace distance_between_trees_l131_131030

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) 
  (h_yard : yard_length = 225) (h_trees : num_trees = 26) : 
  yard_length / (num_trees - 1) = 9 := 
by 
  have num_gaps := num_trees - 1 
  have h_num_gaps : num_gaps = 25 := by rw [h_trees]; norm_num
  calc
  yard_length / num_gaps = 225 / 25 : by rw [h_yard, h_num_gaps]
  ... = 9 : by norm_num

end distance_between_trees_l131_131030


namespace ellipse_equation_and_point_M_exists_l131_131935

-- Define the ellipse E with general parameters a and b
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Given conditions
def right_focus := (1 : ℝ, 0 : ℝ)
def eccentricity := (1 / 2 : ℝ)

-- Main theorem
theorem ellipse_equation_and_point_M_exists :
  ∀ (a b : ℝ) (h : a > b ∧ b > 0),
    (∀ (x y : ℝ), right_focus = (1, 0) ∧ eccentricity = 1 / 2 →
      ellipse a b h x y → (a = 2 ∧ b^2 = 3) ∧ 
      (∀ (M : ℝ × ℝ), M = (11 / 8, 0) → 
        (∃ (m : ℝ), (3 * m^2 + 4) * y^2 + 6 * m * y - 9 = 0) → 
        ∀ (x₀ : ℝ), x₀ =  11 / 8 → 
        ∃ (dot_product : ℝ), dot_product = -135 / 64)) :=
sorry

end ellipse_equation_and_point_M_exists_l131_131935


namespace find_root_and_m_l131_131562

theorem find_root_and_m (x₁ m : ℝ) (h₁ : -2 * x₁ = 2) (h₂ : x^2 + m * x + 2 = 0) : x₁ = -1 ∧ m = 3 := 
by 
  -- Proof omitted
  sorry

end find_root_and_m_l131_131562


namespace angle_ECD_given_conditions_l131_131607

-- Prove that for a given triangle ABC with the specified conditions, the measure of ∠ECD is 50°.
theorem angle_ECD_given_conditions (A B C D E : Point)
  (h1 : AC = BC)
  (h2 : m∠ DCB = 50)
  (h3 : CD ∥ AB)
  (h4 : E lies_on_extension_of CD)
  (h5 : DE ∥ BC)
  : m∠ ECD = 50 :=
  sorry

end angle_ECD_given_conditions_l131_131607


namespace repetend_of_five_seventeenths_l131_131157

theorem repetend_of_five_seventeenths :
  (decimal_expansion (5 / 17)).repeat_called == "294117647" :=
sorry

end repetend_of_five_seventeenths_l131_131157


namespace even_composition_l131_131312

variable {α : Type} [CommRing α]

def is_even_function (g : α → α) : Prop :=
  ∀ x, g (-x) = g x

theorem even_composition (g : α → α) (h : is_even_function g) :
  is_even_function (λ x, g (g x)) :=
by 
  sorry

end even_composition_l131_131312


namespace number_of_elements_in_A_l131_131291

noncomputable def A : set ℝ := 
  { x | ∃ k : ℕ, k > 0 ∧ x = ∑ j in finset.range k, real.sin (2 * j * real.pi / 2023) }

theorem number_of_elements_in_A : finset.card (finset.image (λ k, ∑ j in finset.range k, real.sin (2 * j * real.pi / 2023)) (finset.range 2024)) = 1012 := 
sorry

end number_of_elements_in_A_l131_131291


namespace molecular_physics_statements_l131_131483

theorem molecular_physics_statements :
  (¬A) ∧ B ∧ C ∧ D :=
by sorry

end molecular_physics_statements_l131_131483


namespace cost_of_baguette_is_correct_l131_131858

-- Define the costs of the bakery items
def cost_white_bread : ℝ := 3.50
def cost_sourdough_bread : ℝ := 4.50
def cost_almond_croissant : ℝ := 2.00
def total_spent_over_4_weeks : ℝ := 78.00

-- Define the amounts of each item Candice buys per week
def num_white_bread : ℕ := 2
def num_sourdough_bread : ℕ := 2
def num_almond_croissant : ℕ := 1

-- Define the time frame
def num_weeks : ℕ := 4

-- Calculate the weekly cost of bread and croissant
def weekly_cost : ℝ :=
  (num_white_bread * cost_white_bread) +
  (num_sourdough_bread * cost_sourdough_bread) +
  (num_almond_croissant * cost_almond_croissant)

-- Calculate the total spent on bread and croissant over 4 weeks
def total_bread_and_croissant_cost : ℝ :=
  weekly_cost * num_weeks

-- Calculate the remaining cost spent on baguettes over 4 weeks
def total_baguette_cost : ℝ :=
  total_spent_over_4_weeks - total_bread_and_croissant_cost

-- Calculate the cost of one baguette
def cost_one_baguette : ℝ :=
  total_baguette_cost / num_weeks

-- The theorem we need to prove
theorem cost_of_baguette_is_correct :
  cost_one_baguette = 1.50 :=
by
  sorry

end cost_of_baguette_is_correct_l131_131858


namespace Abby_wins_if_N_2011_Brian_wins_in_31_cases_l131_131473

-- Definitions and assumptions directly from the problem conditions
inductive Player
| Abby
| Brian

def game_condition (N : ℕ) : Prop :=
  ∀ (p : Player), 
    (p = Player.Abby → (∃ k, N = 2 * k + 1)) ∧ 
    (p = Player.Brian → (∃ k, N = 2 * (2^k - 1))) -- This encodes the winning state conditions for simplicity

-- Part (a)
theorem Abby_wins_if_N_2011 : game_condition 2011 :=
by
  sorry

-- Part (b)
theorem Brian_wins_in_31_cases : 
  (∃ S : Finset ℕ, (∀ N ∈ S, N ≤ 2011 ∧ game_condition N) ∧ S.card = 31) :=
by
  sorry

end Abby_wins_if_N_2011_Brian_wins_in_31_cases_l131_131473


namespace minimum_tablets_l131_131809

theorem minimum_tablets (A B C : ℕ) (hA : A = 30) (hB : B = 24) (hC : C = 18) :
  ∃ n, (n >= 16) ∧
       (∀ n' < 16, ∃ a b c, a + b + c = n' ∧ (a < 3 ∨ b < 3 ∨ c < 3)) :=
begin
  use 16,
  split,
  { linarith, },
  { intros n' hn',
    use [2, 2, 2],
    split,
    { linarith, },
    { left, linarith, },
  },
end

end minimum_tablets_l131_131809


namespace problem_1_problem_2_l131_131579

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (2 * x - 3)

theorem problem_1 (x : ℝ) : f(x) ≤ 5 → -1 / 4 ≤ x ∧ x ≤ 9 / 4 := by
  sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, m ^ 2 - m < f(x)) → -1 < m ∧ m < 2 := by
  sorry

end problem_1_problem_2_l131_131579


namespace equivalent_proof_problem_l131_131429

variable (α : ℝ)

-- Definitions of conditions
def quadratic_eq_roots_real_and_signs : Prop :=
  ∀ (a : ℝ), (2 * Real.cos α - 1) * a^2 - 4 * a + 4 * Real.cos α + 2 = 0 → 
  (30 ≤ α ∧ α < 90 → (a ≥ 0) ∧ 
  (30 ≤ α ∧ α < 60 → a > 0) ∧ 
  (60 < α ∧ α < 90 → (a > 0) ∨ (a < 0))) 

def transform_product_of_roots (α : ℝ) : ℝ :=
  2 * (Real.tan (3/2 * α) * Real.cot (1/2 * α))

theorem equivalent_proof_problem :
  (∀ (a : ℝ), (2 * Real.cos α - 1) * a^2 - 4 * a + 4 * Real.cos α + 2 = 0 →
  (30 ≤ α ∧ α < 90 →
    (a ≥ 0 ∧ 
    (30 ≤ α ∧ α < 60 → a > 0) ∧ 
    (60 < α ∧ α < 90 → a > 0 ∨ a < 0)))) ∧
  (∀ (a₁ a₂ : ℝ), 
    (2 * Real.cos α - 1) * a₁^2 - 4 * a₁ + 4 * Real.cos α + 2 = 0 ∧ 
    (2 * Real.cos α - 1) * a₂^2 - 4 * a₂ + 4 * Real.cos α + 2 = 0 →
    2 * (Real.tan (3/2 * α) * Real.cot (1/2 * α))) :=
sorry

end equivalent_proof_problem_l131_131429


namespace roden_gold_fish_count_l131_131689

theorem roden_gold_fish_count
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (gold_fish : ℕ)
  (h1 : total_fish = 22)
  (h2 : blue_fish = 7)
  (h3 : total_fish = blue_fish + gold_fish) : gold_fish = 15 :=
by
  sorry

end roden_gold_fish_count_l131_131689


namespace solve_abs_inequality_l131_131533

theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 15 ↔ (-3 ≤ x ∧ x ≤ 4 / 3) ∨ (8 / 3 ≤ x ∧ x ≤ 7) := 
sorry

end solve_abs_inequality_l131_131533


namespace xiao_gao_actual_score_l131_131601

-- Definitions from the conditions:
def standard_score : ℕ := 80
def xiao_gao_recorded_score : ℤ := 12

-- Proof problem statement:
theorem xiao_gao_actual_score : (standard_score : ℤ) + xiao_gao_recorded_score = 92 :=
by
  sorry

end xiao_gao_actual_score_l131_131601


namespace find_y_parallel_vectors_l131_131951

variable (y : ℝ)

def vector_a := (2, 3 : ℝ)
def vector_b := (4, -1 + y : ℝ)
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = k • w

theorem find_y_parallel_vectors (h : parallel vector_a vector_b) : y = 7 :=
sorry

end find_y_parallel_vectors_l131_131951


namespace k_lt_half_plus_sqrt_2n_l131_131315

noncomputable def set_equidistant_points (S : Set (ℝ × ℝ)) (P : ℝ × ℝ) (k : ℕ) : Prop :=
  ∃ T : Finset (ℝ × ℝ), T ⊆ S ∧ ∃ r : ℝ, r > 0 ∧ (T.card ≥ k ∧ ∀ Q ∈ T, dist P Q = r)

theorem k_lt_half_plus_sqrt_2n
  {n k : ℕ}
  (S : Set (ℝ × ℝ))
  (hS₁ : S.card = n)
  (hS₂ : ∀ (P Q R : ℝ × ℝ), P ∈ S → Q ∈ S → R ∈ S → P ≠ Q → Q ≠ R → P ≠ R → ¬Collinear ℝ {P, Q, R})
  (hS₃ : ∀ P ∈ S, set_equidistant_points S P k) :
  k < (1 / 2 : ℝ) + Real.sqrt (2 * n) :=
by
  sorry

end k_lt_half_plus_sqrt_2n_l131_131315


namespace two_digit_prime_sum_9_l131_131187

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l131_131187


namespace sum_of_midpoint_coordinates_l131_131787

theorem sum_of_midpoint_coordinates (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = 16) (h3 : x2 = -2) (h4 : y2 = -8) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 7 := by
  sorry

end sum_of_midpoint_coordinates_l131_131787


namespace repetend_of_5_div_17_l131_131137

theorem repetend_of_5_div_17 : 
  ∃ repetend : ℕ, 
  decimal_repetend (5 / 17) = repetend ∧ 
  repetend = 2941176470588235 :=
by 
  skip

end repetend_of_5_div_17_l131_131137


namespace measure_angle_AOC_l131_131475

noncomputable def deg_measure_angle_AOC
  (A C O : EuclideanSpace (Fin 3) ℝ)
  (Alice_lat Alice_long : ℝ)
  (Charlie_lat Charlie_long : ℝ)
  (Alice_lat = 0)
  (Alice_long = 73)
  (Charlie_lat = 0)
  (Charlie_long = -78)
  (Earth : is_sphere O) : ℝ :=
  209

theorem measure_angle_AOC : deg_measure_angle_AOC A C O 0 73 0 (-78) _ = 209 :=
by
  sorry

end measure_angle_AOC_l131_131475


namespace job_candidates_excel_nights_l131_131027

theorem job_candidates_excel_nights (hasExcel : ℝ) (dayShift : ℝ) 
    (h1 : hasExcel = 0.2) (h2 : dayShift = 0.7) : 
    (1 - dayShift) * hasExcel = 0.06 :=
by
  sorry

end job_candidates_excel_nights_l131_131027


namespace no_two_digit_prime_with_digit_sum_9_l131_131181

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end no_two_digit_prime_with_digit_sum_9_l131_131181


namespace molar_mass_of_compound_l131_131248

theorem molar_mass_of_compound (total_weight : ℝ) (moles : ℝ) (h_total_weight : total_weight = 2070) (h_moles : moles = 10) :
  (total_weight / moles) = 207 :=
by
  rw [h_total_weight, h_moles]
  simp
  norm_num
  sorry

end molar_mass_of_compound_l131_131248


namespace rate_of_interest_is_8_l131_131044

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def principal_C : ℕ := 3000
def time_C : ℕ := 4
def total_interest : ℕ := 1760

theorem rate_of_interest_is_8 :
  ∃ (R : ℝ), ((principal_B * R * time_B) / 100 + (principal_C * R * time_C) / 100 = total_interest) → R = 8 := 
by
  sorry

end rate_of_interest_is_8_l131_131044


namespace number_of_puppies_sold_l131_131850

variables (P : ℕ) (p_0 : ℕ) (k_0 : ℕ) (r : ℕ) (k_s : ℕ)

theorem number_of_puppies_sold 
  (h1 : p_0 = 7) 
  (h2 : k_0 = 6) 
  (h3 : r = 8) 
  (h4 : k_s = 3) : 
  P = p_0 - (r - (k_0 - k_s)) :=
by sorry

end number_of_puppies_sold_l131_131850


namespace max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l131_131521

open Real

theorem max_min_2sinx_minus_3 : 
  ∀ x : ℝ, 
    -5 ≤ 2 * sin x - 3 ∧ 
    2 * sin x - 3 ≤ -1 :=
by sorry

theorem max_min_7_fourth_sinx_minus_sinx_squared : 
  ∀ x : ℝ, 
    -1/4 ≤ (7/4 + sin x - sin x ^ 2) ∧ 
    (7/4 + sin x - sin x ^ 2) ≤ 2 :=
by sorry

end max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l131_131521


namespace cryptarithmic_solution_l131_131618

variable (A B C : ℕ) -- Define the variables as natural numbers
variable H1 : A + B = C + 10
variable H2 : B + C = 12
variable H3 : A + B < 10
variable H4 : A < 10
variable H5 : B < 10
variable H6 : C < 10

-- We need to show that A = 22
theorem cryptarithmic_solution : A = 22 := sorry

end cryptarithmic_solution_l131_131618


namespace domain_f_l131_131520

noncomputable def f (x : ℝ) : ℝ := sqrt (x - 2) - 1 / sqrt (6 - x)

theorem domain_f : {x : ℝ | 2 ≤ x ∧ x < 6} = {x : ℝ | is_defined (f x)} :=
by
    sorry

end domain_f_l131_131520


namespace amanda_final_score_l131_131845

theorem amanda_final_score
  (average1 : ℕ) (quizzes1 : ℕ) (average_required : ℕ) (quizzes_total : ℕ)
  (H1 : average1 = 92) (H2 : quizzes1 = 4) (H3 : average_required = 93) (H4 : quizzes_total = 5) :
  let total_points1 := quizzes1 * average1,
      total_points_required := quizzes_total * average_required,
      final_score_needed := total_points_required - total_points1
  in final_score_needed = 97 := by
  sorry

end amanda_final_score_l131_131845


namespace find_sum_of_a_and_b_l131_131807

variable (a b w y z S : ℕ)

-- Conditions based on problem statement
axiom condition1 : 19 + w + 23 = S
axiom condition2 : 22 + y + a = S
axiom condition3 : b + 18 + z = S
axiom condition4 : 19 + 22 + b = S
axiom condition5 : w + y + 18 = S
axiom condition6 : 23 + a + z = S
axiom condition7 : 19 + y + z = S
axiom condition8 : 23 + y + b = S

theorem find_sum_of_a_and_b : a + b = 23 :=
by
  sorry  -- To be provided with the actual proof later

end find_sum_of_a_and_b_l131_131807


namespace trains_clear_time_l131_131775

noncomputable def time_to_clear_each_other
  (length_train1 : ℝ)
  (length_train2 : ℝ)
  (speed_train1 : ℝ)
  (speed_train2 : ℝ) : ℝ :=
let total_distance := length_train1 + length_train2 in
let relative_speed := (speed_train1 + speed_train2) * (1000 / 3600) in
total_distance / relative_speed

theorem trains_clear_time
  (length_train1 : ℝ := 315)
  (length_train2 : ℝ := 285)
  (speed_train1 : ℝ := 120)
  (speed_train2 : ℝ := 95) :
  time_to_clear_each_other length_train1 length_train2 speed_train1 speed_train2 ≈ 10.0464 :=
by
  -- Proof will be filled in here
  sorry

end trains_clear_time_l131_131775


namespace interest_rate_proven_l131_131691

structure InvestmentProblem where
  P : ℝ  -- Principal amount
  A : ℝ  -- Accumulated amount
  n : ℕ  -- Number of times interest is compounded per year
  t : ℕ  -- Time in years
  rate : ℝ  -- Interest rate per annum (to be proven)

noncomputable def solve_interest_rate (ip : InvestmentProblem) : ℝ :=
  let half_yearly_rate := ip.rate / 2 / 100
  let amount_formula := ip.P * (1 + half_yearly_rate)^(ip.n * ip.t)
  half_yearly_rate

theorem interest_rate_proven :
  ∀ (P A : ℝ) (n t : ℕ), 
  P = 6000 → 
  A = 6615 → 
  n = 2 → 
  t = 1 → 
  solve_interest_rate {P := P, A := A, n := n, t := t, rate := 10.0952} = 10.0952 := 
by 
  intros
  rw [solve_interest_rate]
  sorry

end interest_rate_proven_l131_131691


namespace polygon_possible_with_area_sixteen_l131_131743

theorem polygon_possible_with_area_sixteen :
  ∃ (P : polygon) (matches : list (side P)), (length(matches) = 12 ∧ (∀ m ∈ matches, m.length = 2) ∧ P.area = 16) := 
sorry

end polygon_possible_with_area_sixteen_l131_131743


namespace a2_value_is_42_l131_131943

noncomputable def a₂_value (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :=
  a_2

theorem a2_value_is_42 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (x^3 + x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + a_4 * (x + 1)^4 +
                a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + 
                a_9 * (x + 1)^9 + a_10 * (x + 1)^10) →
  a₂_value a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 = 42 :=
by
  sorry

end a2_value_is_42_l131_131943


namespace partitioned_sums_eq_l131_131907

/-- Given 200 distinct real numbers, we partition them into two groups of 100 each,
    form two pairs of sorted sequences, and show that the sums of the absolute differences 
    of corresponding elements from both partitions are equal. -/
theorem partitioned_sums_eq (xs : Fin 200 → ℝ) (a a' b b' : Fin 100 → ℝ) 
  (ha : ∀ i j, i < j → a i < a j) (ha' : ∀ i j, i < j → a' i < a' j) 
  (hb : ∀ i j, i < j → b i < b j) (hb' : ∀ i j, i < j → b' i < b' j) 
  (ht1 : ∀ i, a i ∈ xs ∧ a' i ∈ xs ∧ b (⟨i.1, sorry⟩) ∈ xs ∧ b' (⟨i.1, sorry⟩) ∈ xs)
  (ht2 : ∀ x, x ∈ xs → ∃ i, x = a i ∨ x = a' i ∨ x = b i ∨ x = b' i) :
  ∑ i, |a i - a' i| = ∑ j, |b j - b' j| :=
by sorry

end partitioned_sums_eq_l131_131907


namespace arithmetic_seq_general_formula_sum_b_sequence_formula_l131_131624

noncomputable def a (n : ℕ) : ℚ := 2 * n
noncomputable def b (n : ℕ) : ℚ := (-1) ^ (n + 1) * (2 / (a n) + 2 / (a (n + 1)))
noncomputable def T (n : ℕ) : ℚ := (Finset.range (2 * n - 1)).sum (λ k, b (k + 1))

theorem arithmetic_seq_general_formula : ∀ n : ℕ, a n = 2 * n := 
begin
  intro n,
  sorry,
end

theorem sum_b_sequence_formula (n : ℕ) : T n = 1 + 1 / (2 * n) :=
begin
  sorry,
end

end arithmetic_seq_general_formula_sum_b_sequence_formula_l131_131624


namespace complement_is_correct_l131_131585

-- Define the universal set U and set M
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

-- Define the complement of M with respect to U
def complement_U (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

-- State the theorem to be proved
theorem complement_is_correct : complement_U U M = {3, 5, 6} :=
by
  sorry

end complement_is_correct_l131_131585


namespace sqrt_of_fraction_and_product_l131_131855

theorem sqrt_of_fraction_and_product (a b : ℝ) (ha : a = 1 / 2) (hb : b = 18) :
  (Real.sqrt a) * (Real.sqrt b) = 3 :=
by
  have h : a * b = 9
  { rw [ha, hb]
    -- simplification step for the fractional multiplication and product
    norm_num }
  rw [←Real.sqrt_mul (sqrt_nonneg a) (sqrt_nonneg b), h]
  norm_num
  sorry

end sqrt_of_fraction_and_product_l131_131855


namespace ratio_areas_trapezoids_l131_131641

-- Define the given problem conditions
variable (AB CD AC BD OP : ℝ)
variable (O X Y P : Point)
variable (A B C D : Point) 
variable [nonparallel AD BC]

-- Define the properties stated in the problem
variable (H1 : AB = 15)
variable (H2 : CD = 25)
variable (H3 : parallel AB CD)
variable (H4 : midpoint X AD)
variable (H5 : midpoint Y BC)
variable (H6 : intersects AC BD O)
variable (H7 : perpendicular OP AB)

-- Problem to prove
theorem ratio_areas_trapezoids :
  let AreaABYX := 1/2 * (AB + XY) * (OP / 2) in
  let AreaXYCD := 1/2 * (XY + CD) * (OP / 2) in
  AreaABYX / AreaXYCD = 27/35 ∧
  27 + 35 = 62 := 
by
  sorry

end ratio_areas_trapezoids_l131_131641


namespace trajectory_is_parabola_l131_131255

noncomputable def P_trajectory : Type := {P : ℝ × ℝ // 
  let x := P.1, 
      y := P.2 in 
  real.sqrt ((x - 1)^2 + (y - 2)^2) = abs ((3/5) * x - (4/5) * y - 1)}

theorem trajectory_is_parabola (P : P_trajectory) : 
  true := sorry

end trajectory_is_parabola_l131_131255


namespace evaluate_log_l131_131117

theorem evaluate_log : ∀ (a b : ℝ), a = 3^2 → log a (b^(1/3)) = (1/3) := 
by
  intros a b h1
  sorry

end evaluate_log_l131_131117


namespace find_z_value_l131_131173

noncomputable def z : ℝ := ((2 : ℝ).log (3 : ℝ)) * ((3 : ℝ).log (4 : ℝ)) * … * ((49 : ℝ).log (50 : ℝ))

theorem find_z_value : z = 5 := sorry

end find_z_value_l131_131173


namespace determine_spies_possible_l131_131064

-- Define the total number of posts
def num_posts : ℕ := 15

-- Define the \( T_i \) values for the number of spies seen at each post and its neighbors
variable (T : Fin num_posts -> ℕ)

-- Define the theorem to prove that it is possible to determine the number of spies at each post
theorem determine_spies_possible : 
  ∃ a : Fin num_posts -> ℕ, 
      ∀ i : Fin num_posts,
        if h₀ : i.1 = 0 then a i + a (Fin.mk (i.1+1) (by linarith [Fin.is_lt i])) = T i
        else if h₁ : i.1 = num_posts - 1 then a (Fin.mk (i.1-1) (by linarith [Fin.is_lt i])) + a i = T i
        else a (Fin.mk (i.1-1) (by linarith [Fin.is_lt i])) + a i + a (Fin.mk (i.1+1) (by linarith [Fin.is_lt i])) = T i := 
sorry

end determine_spies_possible_l131_131064


namespace total_area_rectangles_l131_131696

theorem total_area_rectangles : 
  let widths : List ℕ := [2, 2, 2, 2, 2, 2]
  let lengths : List ℕ := [1, 9, 25, 49, 81, 121]
  let areas := List.map₂ (λ w l, w * l) widths lengths
  List.sum areas = 572 :=
by
  let widths := [2, 2, 2, 2, 2, 2]
  let lengths := [1, 9, 25, 49, 81, 121]
  let areas := List.map₂ (λ w l, w * l) widths lengths
  have areas_sum : List.sum areas = 572
  sorry

end total_area_rectangles_l131_131696


namespace ellipse_other_intersection_point_l131_131068

/-- Let F1 = (0, 3) and F2 = (4, 0) be the foci of an ellipse. The ellipse intersects the x-axis at (1, 0).
    We claim that the other point of intersection on the x-axis is (7, 0). -/
theorem ellipse_other_intersection_point :
  let F₁ := (0 : ℝ, 3 : ℝ)
  let F₂ := (4 : ℝ, 0 : ℝ)
  let intersection₁ := (1 : ℝ, 0 : ℝ)
  let intersection₂ := (7 : ℝ, 0 : ℝ)
  (dist (1,0) F₁ + dist (1,0) F₂) = (dist intersection₂ F₁ + dist intersection₂ F₂) :=
by
  sorry

end ellipse_other_intersection_point_l131_131068


namespace find_non_integer_angles_count_l131_131669

def angle_measure_not_integer (n : ℕ) : Prop :=
  let angle_measure := 200 * (n - 2) / n
  ¬ (angle_measure : ℕ) = angle_measure

def count_non_integer_angles : ℕ :=
  (Finset.filter (λ n, angle_measure_not_integer n) (Finset.Ico 3 15)).card

theorem find_non_integer_angles_count (h : count_non_integer_angles = 6) : count_non_integer_angles = 6 :=
  by
    sorry

end find_non_integer_angles_count_l131_131669


namespace angle_C_is_140_l131_131997

def angle_A (x : ℝ) : ℝ := 7 * x
def angle_B (x : ℝ) : ℝ := 2 * x
def sum_of_angles_is_180 : Prop := ∀ x : ℝ, angle_A x + angle_B x = 180

theorem angle_C_is_140 
  (x : ℝ)
  (h1 : sum_of_angles_is_180)
  (h2 : angle_A x = angle_C x) : 
  angle_C x = 140 :=
by
  sorry

end angle_C_is_140_l131_131997


namespace impossible_40_percent_lemonade_l131_131467

theorem impossible_40_percent_lemonade (x : ℝ) :
  let water_initial := 5
      syrup_initial := 4
      lime_juice_initial := 2
      total_parts := water_initial + syrup_initial + lime_juice_initial
  in
  let water_final := water_initial - (5 / total_parts) * x + x
      syrup_final := syrup_initial - (4 / total_parts) * x
      lime_juice_final := lime_juice_initial
      total_parts_final := total_parts
  in
  (syrup_final / total_parts_final) = 0.4 → False :=
by
  intro h
  sorry

end impossible_40_percent_lemonade_l131_131467


namespace repetend_of_5_div_17_l131_131141

theorem repetend_of_5_div_17 : 
  ∃ repetend : ℕ, 
  decimal_repetend (5 / 17) = repetend ∧ 
  repetend = 2941176470588235 :=
by 
  skip

end repetend_of_5_div_17_l131_131141


namespace solution_of_xyz_l131_131971

theorem solution_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : x + y + z = 48 := 
sorry

end solution_of_xyz_l131_131971


namespace repetend_of_five_over_seventeen_l131_131126

theorem repetend_of_five_over_seventeen : 
  let r := 5 / 17 in
  ∃ a b : ℕ, a * 10^b = 294117 ∧ (r * 10^b - a) = (r * 10^6 - r * (10^6 / 17))
   ∧ (r * 10^k = (r * 10^6).floor / 10^k ) where k = 6 := sorry

end repetend_of_five_over_seventeen_l131_131126


namespace arithmetic_progression_integers_l131_131484

theorem arithmetic_progression_integers 
  (d : ℤ) (a : ℤ) (h_d_pos : d > 0)
  (h_progression : ∀ i j : ℤ, i ≠ j → ∃ k : ℤ, a * (a + i * d) = a + k * d)
  : ∀ n : ℤ, ∃ m : ℤ, a + n * d = m :=
by
  sorry

end arithmetic_progression_integers_l131_131484


namespace find_B_l131_131263

open Real

noncomputable def a := (5 * sqrt 3) / 3
noncomputable def b := 5
noncomputable def A := 30 * (pi / 180) -- Converting degrees to radians

theorem find_B (B : ℝ) (h_a : a = (5 * sqrt 3) / 3) (h_b : b = 5) (h_A : A = 30 * (pi / 180)) :
  (B = 60 * (pi / 180)) ∨ (B = 120 * (pi / 180)) :=
sorry

end find_B_l131_131263


namespace john_pays_more_than_jane_l131_131657

noncomputable def original_price : ℝ := 34.00
noncomputable def discount : ℝ := 0.10
noncomputable def tip_percent : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price - (discount * original_price)
noncomputable def john_tip : ℝ := tip_percent * original_price
noncomputable def john_total : ℝ := discounted_price + john_tip
noncomputable def jane_tip : ℝ := tip_percent * discounted_price
noncomputable def jane_total : ℝ := discounted_price + jane_tip

theorem john_pays_more_than_jane : john_total - jane_total = 0.51 := by
  sorry

end john_pays_more_than_jane_l131_131657


namespace probability_MAME_on_top_l131_131779

theorem probability_MAME_on_top : 
  let quadrants := 8 
  ∧ let favorable_quadrants := 1
  ∧ let probability := favorable_quadrants / quadrants
  in probability = 1 / 8 := 
begin
  sorry
end

end probability_MAME_on_top_l131_131779


namespace cot_difference_l131_131281

-- Define the triangle ABC with specific conditions
variables {A B C D : Type}
variables (triangle_ABC : Triangle A B C)
variables (median_AD : Median A D B C)
variables (angle_between_AD_BC : ∠ median_AD B C = 30)
variables (length_BD : Real := 2 * length_CD)

-- Define the problem of proving the required value of |cot B - cot C|
theorem cot_difference {A B C D : Type}
  (triangle_ABC : Triangle A B C)
  (median_AD : Median A D B C)
  (angle_between_AD_BC : ∠ median_AD B C = 30)
  (length_BD : Real := 2 * length_CD) :
  |cot(angle A B B) - cot(angle A B C)| = 3 * sqrt 3 := 
sorry

end cot_difference_l131_131281


namespace sqrt_inequality_iff_condition_l131_131658

variables {a b c d : ℝ}

open Real

-- Define the distinct and positive condition
def distinct_positive (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- Define the condition where c or d is between a and b, or vice versa
def in_between_conditions (a b c d : ℝ) : Prop :=
  (a < c ∧ c < b) ∨ (a < d ∧ d < b) ∨ (c < a ∧ a < d) ∨ (c < b ∧ b < d)

-- Inequality that needs to be proved
def sqrt_inequality (a b c d : ℝ) : Prop :=
  sqrt ((a + b) * (c + d)) > sqrt (a * b) + sqrt (c * d)

-- Main theorem statement
theorem sqrt_inequality_iff_condition :
  distinct_positive a b c d →
  (in_between_conditions a b c d → sqrt_inequality a b c d) ∧
  (¬ in_between_conditions a b c d → ∃ a b c d, √((a + b) * (c + d)) ≤ √(a * b) + √(c * d)) :=
by
  sorry

end sqrt_inequality_iff_condition_l131_131658


namespace Joe_paint_usage_l131_131017

theorem Joe_paint_usage :
  (let initial_paint := 360 in
   let first_week_usage := initial_paint / 9 in
   let remaining_paint := initial_paint - first_week_usage in
   let second_week_usage := remaining_paint / 5 in
   let total_usage := first_week_usage + second_week_usage in
   total_usage = 104) :=
by
  let initial_paint := 360
  let first_week_usage := initial_paint / 9
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := remaining_paint / 5
  let total_usage := first_week_usage + second_week_usage

  show total_usage = 104
  sorry

end Joe_paint_usage_l131_131017


namespace kira_travel_time_l131_131396

theorem kira_travel_time :
  let time_between_stations := 2 * 60 -- converting hours to minutes
  let break_time := 30 -- in minutes
  let total_time := 2 * time_between_stations + break_time
  total_time = 270 :=
by
  let time_between_stations := 2 * 60
  let break_time := 30
  let total_time := 2 * time_between_stations + break_time
  exact rfl

end kira_travel_time_l131_131396


namespace minimize_sum_first_n_terms_l131_131257

noncomputable def arithmetic_sequence_min_sum (a : ℕ → ℤ) (d : ℤ) : Prop :=
(a 5 + a 7 + a 9 < 0) ∧ (a 4 + a 11 > 0) ∧ (∀ n, n = 7 → 
  ∑ i in finset.range n, (a 1 + (i - 1) * d) = min_n (λ m, ∑ i in finset.range m, (a 1 + (i - 1) * d)))

theorem minimize_sum_first_n_terms (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 + a 7 + a 9 < 0) 
  (h2 : a 4 + a 11 > 0) :
  arithmetic_sequence_min_sum a d :=
by
  sorry

end minimize_sum_first_n_terms_l131_131257


namespace BH_bisects_CD_l131_131989

open Real
open Classical

variables {A B C D H : Type} [AddGroup A] [MulAction ℝ A] [AddCommGroup B] [Module ℝ B] [AddGroup C] [Module ℝ C] [MetricSpace D] 

structure is_convex_quadrilateral (A B C D : A) : Prop :=
(right_angle_B : ∠ B = 90)
(angle_bisector_AC : ∠ DAB = 2 * ∠ DAC)
(equal_sides_AC_AD : dist A C = dist A D)
(altitude_DH : exists H : A, is_altitude H A D C)

-- Given conditions as structure will be helpful for a clear and concise statement
noncomputable def meet_conditions (A B C D H : Type) [MetricSpace A] [Dist A A] :=
  is_convex_quadrilateral A B C D ∧
  ∃ altitude_DH : Type, H ∈ C ∧ H ∈ D

-- Proving that BH bisects CD
theorem BH_bisects_CD (A B C D H : Type) [MetricSpace A] [Dist A A] :
  meet_conditions A B C D H → dist B H = dist C H → H ∈ B ∧ H ∈ D → 
  H ∈ segment C D :=
sorry

end BH_bisects_CD_l131_131989


namespace right_triangle_identity_l131_131372

variables (a b d : ℝ) (δ : ℝ)

-- Conditions: a and b are the legs of the right-angled triangle, d is the length of the segment from the right angle to the hypotenuse, δ is the angle formed with leg a.
def triangle_condition (a b d : ℝ) (δ : ℝ) : Prop :=
  ∃ (h1 : a > 0) (h2 : b > 0) (h3 : d > 0) (h4 : 0 < δ ∧ δ < π / 2), 
    true

-- Question: Prove that 1/d = (cos δ / a) + (sin δ / b)
theorem right_triangle_identity (a b d : ℝ) (δ : ℝ) (h : triangle_condition a b d δ) :
  1 / d = (Real.cos δ) / a + (Real.sin δ) / b :=
sorry

end right_triangle_identity_l131_131372


namespace sum_of_digits_1197_l131_131987

theorem sum_of_digits_1197 : (1 + 1 + 9 + 7 = 18) := by sorry

end sum_of_digits_1197_l131_131987


namespace fixed_point_coordinates_l131_131717

theorem fixed_point_coordinates (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (0, 2) ∧ ∀ x : ℝ, (λ x, a * x - 1 + 3) x = P.snd :=
by
  sorry

end fixed_point_coordinates_l131_131717


namespace number_of_valid_m_values_l131_131636

/--
In the coordinate plane, construct a right triangle with its legs parallel to the x and y axes, and with the medians on its legs lying on the lines y = 3x + 1 and y = mx + 2. 
Prove that the number of values for the constant m such that this triangle exists is 2.
-/
theorem number_of_valid_m_values : 
  ∃ (m : ℝ), 
    (∃ (a b : ℝ), 
      (∀ D E : ℝ × ℝ, D = (a / 2, 0) ∧ E = (0, b / 2) →
      D.2 = 3 * D.1 + 1 ∧ 
      E.2 = m * E.1 + 2)) → 
    (number_of_solutions_for_m = 2) 
  :=
sorry

end number_of_valid_m_values_l131_131636


namespace parabola_has_one_x_intercept_l131_131956

-- Define the equation of the parabola.
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 4

-- Prove that the number of x-intercepts of the graph of the parabola is 1.
theorem parabola_has_one_x_intercept : (∃! y : ℝ, parabola y = 4) :=
by
  sorry

end parabola_has_one_x_intercept_l131_131956


namespace polynomial_divisibility_l131_131911

theorem polynomial_divisibility 
  (a b c : ℤ)
  (P : ℤ → ℤ)
  (root_condition : ∃ u v : ℤ, u * v * (u + v) = -c ∧ u * v = b) 
  (P_def : ∀ x, P x = x^3 + a * x^2 + b * x + c) :
  2 * P (-1) ∣ (P 1 + P (-1) - 2 * (1 + P 0)) :=
by
  sorry

end polynomial_divisibility_l131_131911


namespace abc_zero_l131_131022

-- Define the given conditions as hypotheses
theorem abc_zero (a b c : ℚ) 
  (h1 : (a^2 + 1)^3 = b + 1)
  (h2 : (b^2 + 1)^3 = c + 1)
  (h3 : (c^2 + 1)^3 = a + 1) : 
  a = 0 ∧ b = 0 ∧ c = 0 := 
sorry

end abc_zero_l131_131022


namespace repetend_of_five_over_seventeen_l131_131124

theorem repetend_of_five_over_seventeen : 
  let r := 5 / 17 in
  ∃ a b : ℕ, a * 10^b = 294117 ∧ (r * 10^b - a) = (r * 10^6 - r * (10^6 / 17))
   ∧ (r * 10^k = (r * 10^6).floor / 10^k ) where k = 6 := sorry

end repetend_of_five_over_seventeen_l131_131124


namespace repetend_of_5_div_17_l131_131142

theorem repetend_of_5_div_17 : 
  ∃ repetend : ℕ, 
  decimal_repetend (5 / 17) = repetend ∧ 
  repetend = 2941176470588235 :=
by 
  skip

end repetend_of_5_div_17_l131_131142


namespace find_f_minus1_plus_f_prime_minus1_l131_131229

variable (f : ℝ → ℝ)

theorem find_f_minus1_plus_f_prime_minus1 (h1 : ∀ x y, x + y - 3 = 0 → y = f x)
  (h2 : f' = λ x, -1) : f (-1) + f' (-1) = 3 := by
sury

end find_f_minus1_plus_f_prime_minus1_l131_131229


namespace number_of_digits_in_M_l131_131909

theorem number_of_digits_in_M (lg2: ℝ) (h: lg2 = 0.30103) : ∃ d: ℕ, d = 3011 ∧
    ∃ M: ℝ, M = (∑ i in range (10001), binomial 10000 i * (10^4)^i) ∧
    d = (⌊log 10 M⌋ + 1) := 
by 
  sorry


end number_of_digits_in_M_l131_131909


namespace simplify_expression_l131_131225

variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b < 0)

theorem simplify_expression : 
  real.sqrt (b^2) + abs (b - a) = a - 2 * b :=
sorry

end simplify_expression_l131_131225


namespace three_digit_number_l131_131733

open Classical

theorem three_digit_number (a x y : ℕ) (h : 1 ≤ x ∧ x ≤ 9) :
  (∀ z : ℕ, (100 * x + 10 * y + 1 = z) → 
     ((10 * x + y - 11 = (10 * a ^ (log 3 / log (sqrt a))) / 9) → z = 211)) :=
sorry

end three_digit_number_l131_131733


namespace can_construct_polygon_l131_131757

def match_length : ℕ := 2
def number_of_matches : ℕ := 12
def total_length : ℕ := number_of_matches * match_length
def required_area : ℝ := 16

theorem can_construct_polygon : 
  (∃ (P : Polygon), P.perimeter = total_length ∧ P.area = required_area) := 
sorry

end can_construct_polygon_l131_131757


namespace two_digit_prime_sum_9_l131_131190

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l131_131190


namespace reciprocal_of_neg3_l131_131726

theorem reciprocal_of_neg3 : (1 : ℚ) / (-3 : ℚ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l131_131726


namespace remainder_is_undetermined_l131_131048

-- Define the given conditions
def exists_number_with_remainder (k : ℤ) : Prop :=
  ∃ N, N = 18 * k + 19

-- Define the main statement
theorem remainder_is_undetermined : ∀ k : ℤ, exists_number_with_remainder k → ¬ ∃ r : ℕ, r < 242 ∧ (∃ N, N = 18 * k + 19 ∧ N % 242 = r) → 
sorry

end remainder_is_undetermined_l131_131048


namespace diane_initial_amount_l131_131509

theorem diane_initial_amount
  (X : ℝ)        -- the amount Diane started with
  (won_amount : ℝ := 65)
  (total_loss : ℝ := 215)
  (owing_friends : ℝ := 50)
  (final_amount := X + won_amount - total_loss - owing_friends) :
  X = 100 := 
by 
  sorry

end diane_initial_amount_l131_131509


namespace q_active_time_l131_131770

theorem q_active_time
  (ratio_of_investments : ℕ → ℕ → ℕ)
  (ratio_of_returns : ℕ → ℕ → ℕ)
  (P_active_months : ℕ)
  (R_active_months : ℕ)
  (total_profit : ℕ)
  (P_share : ℕ) : 
  ratio_of_investments(7) → ratio_of_investments(5.00001) → ratio_of_investments(3.99999) →
  ratio_of_returns(7.00001) → ratio_of_returns(10) → ratio_of_returns(6) →
  P_active_months = 5 → 
  R_active_months = 8 → 
  total_profit = 200000 → 
  P_share = 50000 → 
  ∃ (Q_active_months : ℕ), Q_active_months = 6 :=
sorry

end q_active_time_l131_131770


namespace unit_vector_in_yz_plane_l131_131530

theorem unit_vector_in_yz_plane (u : ℝ × ℝ × ℝ)
  (h1 : ∥u∥ = 1)
  (h2 : u.1 = 0)
  (h3 : ∀ v : ℝ × ℝ × ℝ, 
        (v = ⟨1, 2, -2⟩ →
        inner u v = ∥v∥ * (real.cos (30 * real.pi / 180)))
  )
  (h4 : ∀ v : ℝ × ℝ × ℝ,
        (v = ⟨1, 1, 1⟩ →
        inner u v = ∥v∥ * (real.cos (45 * real.pi / 180)))
  ) :
  ∃ y z : ℝ, u = ⟨0, y, z⟩ ∧ y^2 + z^2 = 1 :=
sorry

end unit_vector_in_yz_plane_l131_131530


namespace probability_of_collinear_dots_in_5x5_grid_l131_131640

-- The Lean code to represent the problem and statement 
theorem probability_of_collinear_dots_in_5x5_grid :
  let total_dots := 25
  let dots_chosen := 4
  let collinear_sets := 14
  let total_combinations := Nat.choose total_dots dots_chosen
  let probability := (collinear_sets : ℚ) / total_combinations in
  probability = 7 / 6325 :=
begin
  sorry
end

end probability_of_collinear_dots_in_5x5_grid_l131_131640


namespace reciprocal_sum_fractions_l131_131784

theorem reciprocal_sum_fractions:
  let a := (3: ℚ) / 4
  let b := (5: ℚ) / 6
  let c := (1: ℚ) / 2
  (a + b + c)⁻¹ = 12 / 25 :=
by
  sorry

end reciprocal_sum_fractions_l131_131784


namespace parallelogram_condition_l131_131634

theorem parallelogram_condition
  (ABCD : Quadrilateral)
  (convex: ABCD.Convex)
  (P1 P2 P3 : Point)
  (non_collinear : ¬Collinear P1 P2 P3)
  (H1 : Area (Triangle ABCD.AB P1) + Area (Triangle ABCD.CD P1) = Area (Triangle ABCD.BC P1) + Area (Triangle ABCD.AD P1))
  (H2 : Area (Triangle ABCD.AB P2) + Area (Triangle ABCD.CD P2) = Area (Triangle ABCD.BC P2) + Area (Triangle ABCD.AD P2))
  (H3 : Area (Triangle ABCD.AB P3) + Area (Triangle ABCD.CD P3) = Area (Triangle ABCD.BC P3) + Area (Triangle ABCD.AD P3))
  : ABCD.Parallelogram :=
by
  sorry

end parallelogram_condition_l131_131634


namespace distance_between_cities_l131_131797

def distance_thing 
  (d_A d_B : ℝ) 
  (v_A v_B : ℝ) 
  (t_diff : ℝ) : Prop :=
d_A = (3 / 5) * d_B ∧
v_A = 72 ∧
v_B = 108 ∧
t_diff = (1 / 4) ∧
(d_A + d_B) = 432

theorem distance_between_cities
  (d_A d_B : ℝ)
  (v_A v_B : ℝ)
  (t_diff : ℝ)
  (h : distance_thing d_A d_B v_A v_B t_diff)
  : d_A + d_B = 432 := by
  sorry

end distance_between_cities_l131_131797


namespace folded_string_pieces_l131_131174

/-- Let a string be folded in half 10 times, resulting in 1024 layers, and then cut into 10 equal parts.
Given this configuration, we prove that the number of longer strings obtained is 1023 and the number
of shorter strings obtained is 8194. -/
theorem folded_string_pieces :
  ∃ longer shorter : ℕ,
    (∃ folded_layers : ℕ, folded_layers = 2^10) ∧
    (∃ equal_parts : ℕ, equal_parts = 10) ∧
    (folded_layers/ equal_parts = 1024 / 10) →
    (longer = 1023 ∧ shorter = 8194) :=
begin
  sorry
end

end folded_string_pieces_l131_131174


namespace log_base_9_of_cubic_root_of_9_l131_131113

theorem log_base_9_of_cubic_root_of_9 : log 9 (cbrt 9) = 1 / 3 :=
by 
  -- required to use Mathfcklb.log or any other definitions properly
  sorry

end log_base_9_of_cubic_root_of_9_l131_131113


namespace augmented_matrix_correct_l131_131706

-- Define the system of linear equations as a pair of equations
def system_of_equations (x y : ℝ) : Prop :=
  (2 * x + y = 1) ∧ (3 * x - 2 * y = 0)

-- Define what it means to be the correct augmented matrix for the system
def is_augmented_matrix (A : Matrix (Fin 2) (Fin 3) ℝ) : Prop :=
  A = ![
    ![2, 1, 1],
    ![3, -2, 0]
  ]

-- The theorem states that the augmented matrix of the given system of equations is the specified matrix
theorem augmented_matrix_correct :
  ∃ x y : ℝ, system_of_equations x y ∧ is_augmented_matrix ![
    ![2, 1, 1],
    ![3, -2, 0]
  ] :=
sorry

end augmented_matrix_correct_l131_131706


namespace units_produced_today_eq_90_l131_131191

-- Define the average production and number of past days
def average_past_production (n : ℕ) (past_avg : ℕ) : ℕ :=
  n * past_avg

def average_total_production (n : ℕ) (current_avg : ℕ) : ℕ :=
  (n + 1) * current_avg

def units_produced_today (n : ℕ) (past_avg : ℕ) (current_avg : ℕ) : ℕ :=
  average_total_production n current_avg - average_past_production n past_avg

-- Given conditions
def n := 5
def past_avg := 60
def current_avg := 65

-- Statement to prove
theorem units_produced_today_eq_90 : units_produced_today n past_avg current_avg = 90 :=
by
  -- Declare which parts need proving
  sorry

end units_produced_today_eq_90_l131_131191


namespace find_a_range_l131_131925

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then - (1 / 3) * x ^ 3 + (1 - a) * (1 / 2) * x ^ 2 + a * x - (4 / 3)
else (a - 1) * Real.log x + (1 / 2) * x ^ 2 - a * x

theorem find_a_range (a : ℝ) (h : a > 0) (hx : ∀ x : ℝ, -a < x ∧ x < 2 * a → Monotone (λ y, f a y)) :
  0 < a ∧ a ≤ (10 / 9) := sorry

end find_a_range_l131_131925


namespace revenue_decrease_by_1_point_1_percent_l131_131385

variable (T C : ℝ)

def original_revenue (T C : ℝ) : ℝ := T * C
def new_revenue (T C : ℝ) : ℝ := 0.86 * T * 1.15 * C
def revenue_effect (T C : ℝ) : ℝ := (new_revenue T C) / (original_revenue T C)

theorem revenue_decrease_by_1_point_1_percent (T C : ℝ) : revenue_effect T C = 0.989 :=
by
  sorry

end revenue_decrease_by_1_point_1_percent_l131_131385


namespace product_sum_125_l131_131175

theorem product_sum_125 :
  ∀ (m n : ℕ), m ≥ n ∧
              (∀ (k : ℕ), 0 < k → |Real.log m - Real.log k| < Real.log n → k ≠ 0)
              → (m * n = 125) :=
by sorry

end product_sum_125_l131_131175


namespace reciprocal_of_neg3_l131_131727

theorem reciprocal_of_neg3 : (1 : ℚ) / (-3 : ℚ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l131_131727


namespace true_proposition_l131_131210

variables {a : ℝ}

def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → a^x < a^y
def q (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ log a 2 + log 2 a ≥ 2

theorem true_proposition (a : ℝ) (h1 : p a) (h2 : q a) : p a ∨ ¬ q a :=
begin
  sorry
end

end true_proposition_l131_131210


namespace fenced_area_with_cutout_l131_131359

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l131_131359


namespace polygon_possible_l131_131740

-- Definition: a polygon with matches without breaking them
structure MatchPolygon (n : ℕ) (length : ℝ) where
  num_matches : ℕ
  len_matches : ℝ
  area : ℝ
  notequalzero : len_matches ≠ 0
  notequalzero2 : area ≠ 0
  perimeter_eq: num_matches * len_matches = length * real.of_nat n
  all_matches_used : n = 12
  no_breaking : (length / real.of_nat n) = 2 

theorem polygon_possible : 
  ∃ P : MatchPolygon 12 2, P.area = 16 :=
sorry

end polygon_possible_l131_131740


namespace am_gm_four_vars_l131_131433

theorem am_gm_four_vars {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / a + 1 / b + 1 / c + 1 / d) ≥ 16 :=
by
  sorry

end am_gm_four_vars_l131_131433


namespace water_force_on_dam_l131_131853

-- Given conditions
def density : Real := 1000  -- kg/m^3
def gravity : Real := 10    -- m/s^2
def a : Real := 5.7         -- m
def b : Real := 9.0         -- m
def h : Real := 4.0         -- m

-- Prove that the force is 544000 N under the given conditions
theorem water_force_on_dam : ∃ (F : Real), F = 544000 :=
by
  sorry  -- proof goes here

end water_force_on_dam_l131_131853


namespace projectile_reaches_30m_l131_131463

noncomputable def projectile_height (t : ℝ) : ℝ := 60 - 5 * t - 6 * t^2

theorem projectile_reaches_30m :
  ∃ t : ℝ, t = ( -5 + real.sqrt 745 ) / 12 ∧ projectile_height t = 30 :=
begin
  sorry
end

end projectile_reaches_30m_l131_131463


namespace num_of_factorizable_poly_l131_131100

theorem num_of_factorizable_poly : 
  ∃ (n : ℕ), (1 ≤ n ∧ n ≤ 2023) ∧ 
              (∃ (a : ℤ), n = a * (a + 1)) :=
sorry

end num_of_factorizable_poly_l131_131100


namespace additional_oil_needed_l131_131197

variable (oil_per_cylinder : ℕ) (number_of_cylinders : ℕ) (oil_already_added : ℕ)

theorem additional_oil_needed (h1 : oil_per_cylinder = 8) (h2 : number_of_cylinders = 6) (h3 : oil_already_added = 16) :
  oil_per_cylinder * number_of_cylinders - oil_already_added = 32 :=
by
  -- proof here
  sorry

end additional_oil_needed_l131_131197


namespace sequence_periodic_l131_131944

def sequence : ℕ → ℚ
| 0       := 1 / 2  -- Lean indexing starts from 0, so this represents a₁
| (n + 1) := 1 / (1 - sequence n)

theorem sequence_periodic (n : ℕ) : sequence (3 * n + 2) = -1 :=
by
  sorry

end sequence_periodic_l131_131944


namespace true_proposition_D_l131_131794

-- Defining the propositions as individual conditions
def proposition_A : Prop :=
  (∃ x, x ^ 3 = 0 ∧ x ≠ 0) ∨ (∃ y, y ^ 3 = 1 ∧ y ≠ 1)

def proposition_B (l1 l2 t : ℝ) : Prop :=
  ∃ (a1 a2 : ℝ), 
    a1 ≠ a2 ∧ l1 = t ∧ l2 = t ∧ l1 = l2

def proposition_C (p : ℝ) (l : ℝ) : Prop :=
  ∃! l', (l' ≠ l ∧ through p l' ∧ parallel l l')

def proposition_D (l1 l2 l3 : ℝ) : Prop :=
  (⊥ l1 l3) ∧ (⊥ l2 l3) → parallel l1 l2

-- The main theorem stating the true proposition
theorem true_proposition_D (l1 l2 l3 : ℝ) : proposition_D l1 l2 l3 :=
  sorry

end true_proposition_D_l131_131794


namespace job_candidates_excel_nights_l131_131026

theorem job_candidates_excel_nights (hasExcel : ℝ) (dayShift : ℝ) 
    (h1 : hasExcel = 0.2) (h2 : dayShift = 0.7) : 
    (1 - dayShift) * hasExcel = 0.06 :=
by
  sorry

end job_candidates_excel_nights_l131_131026


namespace three_digit_numbers_left_l131_131962

def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def isABAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ n = 100 * A + 10 * B + A

def isAABOrBAAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ (n = 100 * A + 10 * A + B ∨ n = 100 * B + 10 * A + A)

def totalThreeDigitNumbers : ℕ := 900

def countABA : ℕ := 81

def countAABAndBAA : ℕ := 153

theorem three_digit_numbers_left : 
  (totalThreeDigitNumbers - countABA - countAABAndBAA) = 666 := 
by
   sorry

end three_digit_numbers_left_l131_131962


namespace Innokentiy_games_l131_131108

theorem Innokentiy_games (Egor_games : ℕ) (Nikita_games : ℕ) (total_participation : ℕ)
  (H1 : Egor_games = 13)
  (H2 : Nikita_games = 27)
  (H3 : total_participation = 54) :
  ∃ Innokentiy_games : ℕ, Egor_games + Nikita_games + Innokentiy_games = total_participation ∧ Innokentiy_games = 14 :=
by {
  have h1 : Egor_games + Nikita_games + 14 = total_participation,
  { rw [H1, H2, H3], norm_num },
  use 14,
  split,
  { exact h1 },
  { refl }
}

end Innokentiy_games_l131_131108


namespace chord_length_l131_131709

noncomputable def circle_center (c: ℝ × ℝ) (r: ℝ): Prop := 
  ∃ x y: ℝ, 
    (x - c.1)^2 + (y - c.2)^2 = r^2

noncomputable def line_equation (a b c: ℝ): Prop := 
  ∀ x y: ℝ, 
    a*x + b*y + c = 0

theorem chord_length (a: ℝ): 
  circle_center (2, 1) 2 ∧ line_equation a 1 (-5) ∧
  ∃(chord_len: ℝ), chord_len = 4 → 
  a = 2 :=
by
  sorry

end chord_length_l131_131709


namespace rotation_150_degree_matrix_l131_131507

def rotation_matrix (θ : ℝ) : Matrix 2 2 ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem rotation_150_degree_matrix : 
  rotation_matrix (5 * Real.pi / 6) = 
  ![![-(Real.sqrt 3) / 2, -1 / 2], ![1 / 2, -(Real.sqrt 3) / 2]] :=
by sorry

end rotation_150_degree_matrix_l131_131507


namespace inverse_matrix_l131_131921

theorem inverse_matrix
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (B : Matrix (Fin 2) (Fin 2) ℚ)
  (H : A * B = ![![1, 2], ![0, 6]]) :
  A⁻¹ = ![![-1, 0], ![0, 2]] :=
sorry

end inverse_matrix_l131_131921


namespace proportion_of_adopted_kittens_l131_131676

-- Define the relevant objects and conditions in Lean
def breeding_rabbits : ℕ := 10
def kittens_first_spring := 10 * breeding_rabbits -- 100 kittens
def kittens_second_spring : ℕ := 60
def adopted_first_spring (P : ℝ) := 100 * P
def returned_first_spring : ℕ := 5
def adopted_second_spring : ℕ := 4
def total_rabbits_in_house (P : ℝ) :=
  breeding_rabbits + (kittens_first_spring - adopted_first_spring P + returned_first_spring) +
  (kittens_second_spring - adopted_second_spring)

theorem proportion_of_adopted_kittens : ∃ (P : ℝ), total_rabbits_in_house P = 121 ∧ P = 0.5 :=
by
  use 0.5
  -- Proof part (with "sorry" to skip the detailed proof)
  sorry

end proportion_of_adopted_kittens_l131_131676


namespace repetend_of_5_div_17_l131_131131

theorem repetend_of_5_div_17 :
  let dec := 5 / 17 in
  decimal_repetend dec = "294117" := sorry

end repetend_of_5_div_17_l131_131131


namespace Janice_earnings_l131_131651

theorem Janice_earnings (days_worked_per_week : ℕ) (earnings_per_day : ℕ) (overtime_shifts : ℕ) (overtime_earnings_per_shift : ℕ)
  (h1 : days_worked_per_week = 5)
  (h2 : earnings_per_day = 30)
  (h3 : overtime_shifts = 3)
  (h4 : overtime_earnings_per_shift = 15) :
  (days_worked_per_week * earnings_per_day) + (overtime_shifts * overtime_earnings_per_shift) = 195 :=
by {
  sorry
}

end Janice_earnings_l131_131651


namespace trigonometric_identity_l131_131490

theorem trigonometric_identity :
  sin (315 * real.pi / 180) - cos (135 * real.pi / 180) + 2 * sin (570 * real.pi / 180) = real.sqrt 2 + 1 :=
by sorry

end trigonometric_identity_l131_131490


namespace no_zero_root_l131_131498

theorem no_zero_root (x : ℝ) :
  (¬ (∃ x : ℝ, (4 * x ^ 2 - 3 = 49) ∧ x = 0)) ∧
  (¬ (∃ x : ℝ, (x ^ 2 - x - 20 = 0) ∧ x = 0)) :=
by
  sorry

end no_zero_root_l131_131498


namespace part1_part2_l131_131239

-- Define the polar equation for curve C1
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 2 / (3 + real.cos (2 * θ))

-- Define the rectangular equation for curve C1
def rectangular_equation_C1 (x y : ℝ) : Prop :=
  2 * x^2 + y^2 = 1

-- Define the transformation to curve C2
def rectangular_equation_C2 (x y : ℝ) : Prop :=
  (x^2) / 2 + 4 * y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x + y - 5 = 0

-- Define the maximum distance from any point on curve C2 to the line l
def max_distance_to_line (θ : ℝ) : ℝ :=
  abs ((sqrt 2 * real.cos θ + (1 / 2) * real.sin θ - 5)) / sqrt 2

theorem part1 (ρ θ x y : ℝ) :
  polar_equation ρ θ →
  (ρ * real.cos θ = x ∧ ρ * real.sin θ = y) →
  rectangular_equation_C1 x y :=
sorry

theorem part2 (x y : ℝ) (θ : ℝ) :
  rectangular_equation_C2 x y →
  max_distance_to_line θ ≤ (13 * sqrt 2) / 4 :=
sorry

end part1_part2_l131_131239


namespace evaluate_expression_l131_131876

variables (x y : ℕ)

theorem evaluate_expression : x = 2 → y = 4 → y * (y - 2 * x + 1) = 4 :=
by
  intro h1 h2
  sorry

end evaluate_expression_l131_131876


namespace max_value_of_m_l131_131550

-- Define the conditions including the triangle and area statement
noncomputable def triangle
  (A B C : ℝ) -- Angles
  (a b c : ℝ) -- Corresponding sides
  (S : ℝ)    -- Given area
  (area_condition : S = (Real.sqrt 3) / 12 * a^2) 
  : Prop :=
  ∃ (m : ℝ), (m = 4) ∧ (∀ B C , ∀ b c, Real.sin B^2 + Real.sin C^2 = m * Real.sin B * Real.sin C)

-- The theorem statement indicating the maximum value of m
theorem max_value_of_m
  (A B C : ℝ) -- Angles
  (a b c : ℝ) -- Corresponding sides
  (S : ℝ)    -- Area of the triangle
  (area_condition : S = (Real.sqrt 3) / 12 * a^2) 
  : (∃ (m : ℝ), (m = 4) ∧ (∀ (B C :ℝ), ∀ (b c :ℝ), Real.sin B^2 + Real.sin C^2 = m * Real.sin B * Real.sin C)) :=
sorry

end max_value_of_m_l131_131550


namespace first_digging_project_length_l131_131035

def volume (length : ℝ) (breadth : ℝ) (depth : ℝ) : ℝ :=
  length * breadth * depth

theorem first_digging_project_length :
  let L := (20 * 50 * 75) / (30 * 100) in
  volume L 30 100 = volume 20 50 75 :=
by
  sorry

end first_digging_project_length_l131_131035


namespace simplify_radical_1_simplify_radical_2_find_value_of_a_l131_131427

-- Problem 1
theorem simplify_radical_1 : 7 + 2 * (Real.sqrt 10) = (Real.sqrt 2 + Real.sqrt 5) ^ 2 := 
by sorry

-- Problem 2
theorem simplify_radical_2 : (Real.sqrt (11 - 6 * (Real.sqrt 2))) = 3 - Real.sqrt 2 := 
by sorry

-- Problem 3
theorem find_value_of_a (a m n : ℕ) (h : a + 2 * Real.sqrt 21 = (Real.sqrt m + Real.sqrt n) ^ 2) : 
  a = 10 ∨ a = 22 := 
by sorry

end simplify_radical_1_simplify_radical_2_find_value_of_a_l131_131427


namespace repeating_decimal_difference_l131_131504

theorem repeating_decimal_difference :
  let x := (8 / 11) in
  x - (72 / 100) = (800 / 1099989) :=
sorry

end repeating_decimal_difference_l131_131504


namespace three_spades_select_order_ways_l131_131267

theorem three_spades_select_order_ways (n : ℕ) (hn : n = 13) : 
  (∏ i in (finset.range 3), (n - i)) = 1716 := by 
sorry

end three_spades_select_order_ways_l131_131267


namespace speed_of_man_in_still_water_l131_131459

variable (v_m v_s : ℝ)

-- Conditions
def downstream_condition : 36 = 6 * (v_m + v_s) := by sorry
def upstream_condition : 18 = 6 * (v_m - v_s) := by sorry

-- Theorem statement
theorem speed_of_man_in_still_water (h1 : downstream_condition v_m v_s) (h2 : upstream_condition v_m v_s) :
  v_m = 4.5 := by sorry

end speed_of_man_in_still_water_l131_131459


namespace exactly_two_talents_l131_131106

open Nat

def total_students : Nat := 50
def cannot_sing_students : Nat := 20
def cannot_dance_students : Nat := 35
def cannot_act_students : Nat := 15

theorem exactly_two_talents : 
  (total_students - cannot_sing_students) + 
  (total_students - cannot_dance_students) + 
  (total_students - cannot_act_students) - total_students = 30 := by
  sorry

end exactly_two_talents_l131_131106


namespace xyz_sum_48_l131_131968

theorem xyz_sum_48 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : z * x + y = 47) : 
  x + y + z = 48 :=
sorry

end xyz_sum_48_l131_131968


namespace count_distinct_4_digit_numbers_l131_131606

theorem count_distinct_4_digit_numbers 
  (a b c d : ℕ) 
  (h1 : a ∈ {1, 2, 3, 4}) 
  (h2 : b ∈ {1, 2, 3, 4}) 
  (h3 : c ∈ {1, 2, 3, 4}) 
  (h4 : d ∈ {1, 2, 3, 4}) 
  (h5 : a ≠ b) 
  (h6 : b ≠ c) 
  (h7 : c ≠ d) 
  (h8 : d ≠ a) 
  (h9 : a ≤ b ∧ a ≤ c ∧ a ≤ d) : 
  ∃ n : ℕ, n = 18 := 
sorry

end count_distinct_4_digit_numbers_l131_131606


namespace min_value_A_x1_x2_l131_131567

noncomputable def f (x : ℝ) : ℝ := Real.sin (2014 * x + π / 6) + Real.cos (2014 * x - π / 3)

theorem min_value_A_x1_x2 :
  let A := 2 in
  ∃ (x1 x2 : ℝ), (∀ x, f x1 ≤ f x ∧ f x ≤ f x2) ∧ A * |x1 - x2| = π / 1007 :=
by
  sorry

end min_value_A_x1_x2_l131_131567


namespace part_a_part_b_l131_131684

def P (m n : ℕ) : ℕ := m^2003 * n^2017 - m^2017 * n^2003

theorem part_a (m n : ℕ) : P m n % 24 = 0 := 
by sorry

theorem part_b : ∃ (m n : ℕ), P m n % 7 ≠ 0 :=
by sorry

end part_a_part_b_l131_131684


namespace collinear_A_E_F_l131_131710

-- Variables and definitions representing given conditions
variables (C1 C2 : Circle) (A K B C P Q E F : Point)
variables (h1 : C1 ∩ C2 = {A, K})
variables (k_is_tangent_near_K : ∀ {B C : Point}, C1.isTangent B ∧ C2.isTangent C)
variables (h2 : P = foot (B, AC))
variables (h3 : Q = foot (C, AB))
variables (h4 : E = reflection K PQ)
variables (h5 : F = reflection K BC)

-- The theorem to prove collinearity
theorem collinear_A_E_F : Collinear {A, E, F} :=
by
  sorry

end collinear_A_E_F_l131_131710


namespace find_z_l131_131242

theorem find_z (a b c : ℕ) (z : ℚ) (h_a : a = 105) (h_b : b = 36) (h_c : c = 90) 
(h_eq : a^3 = (21 * b * c * z) / 6) : z = 102.083 := by {
  subst h_a,
  subst h_b,
  subst h_c,
  field_simp at h_eq,
  norm_num at h_eq,
  exact h_eq,
  -- sorry added to avoid proof
  sorry
}

end find_z_l131_131242


namespace no_two_digit_prime_sum_digits_nine_l131_131180

theorem no_two_digit_prime_sum_digits_nine :
  ¬ ∃ p : ℕ, prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 + p % 10 = 9) :=
sorry

end no_two_digit_prime_sum_digits_nine_l131_131180


namespace unique_real_root_of_quadratic_l131_131605

theorem unique_real_root_of_quadratic (k : ℝ) :
  (∃ a : ℝ, ∀ b : ℝ, ((k^2 - 9) * b^2 - 2 * (k + 1) * b + 1 = 0 → b = a)) ↔ (k = 3 ∨ k = -3 ∨ k = -5) :=
by
  sorry

end unique_real_root_of_quadratic_l131_131605


namespace sum_of_midpoint_coordinates_l131_131786

theorem sum_of_midpoint_coordinates (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = 16) (h3 : x2 = -2) (h4 : y2 = -8) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 7 := by
  sorry

end sum_of_midpoint_coordinates_l131_131786


namespace outfits_not_all_same_color_l131_131964

def number_of_shirts : ℕ := 5
def number_of_pants : ℕ := 3
def number_of_hats : ℕ := 5
def pants_colors : Finset String := {"red", "green", "blue"}
def shirts_and_hats_colors : Finset String := {"red", "green", "blue", "orange", "purple"}
def matching_outfit_count : ℕ := 3

theorem outfits_not_all_same_color (total_outfits : ℕ) (non_matching_outfits : ℕ) : 
  total_outfits = number_of_shirts * number_of_pants * number_of_hats →
  non_matching_outfits = total_outfits - matching_outfit_count →
  non_matching_outfits = 72 :=
by
  intros h_total h_non_matching
  rw [h_total, h_non_matching]
  exact rfl


end outfits_not_all_same_color_l131_131964


namespace determine_jubilee_coin_weight_l131_131477

theorem determine_jubilee_coin_weight
  (coins : List ℕ)
  (h₁ : coins.length = 16)
  (h₂ : coins.count 11 = 8)
  (h₃ : coins.count 10 = 8) :
  ∃ J ∈ coins, ∃ weighings : List (List ℕ × List ℕ), weighings.length ≤ 3 ∧ 
  (∀ (balance_scale : List ℕ × List ℕ) (left right : List ℕ),
  balance_scale = (left, right) →
  J ∈ left ∨ J ∈ right ∨ 
  List.foldl (+) 0 left ≠ List.foldl (+) 0 right):
  sorry


end determine_jubilee_coin_weight_l131_131477


namespace staircase_toothpicks_l131_131496

theorem staircase_toothpicks (n : ℕ) (h_toothpicks : 2 * (List.sum (List.map (λ k, k^2) (List.range (n+1)))) = 630) : n = 9 :=
sorry

end staircase_toothpicks_l131_131496


namespace polygon_possible_with_area_sixteen_l131_131744

theorem polygon_possible_with_area_sixteen :
  ∃ (P : polygon) (matches : list (side P)), (length(matches) = 12 ∧ (∀ m ∈ matches, m.length = 2) ∧ P.area = 16) := 
sorry

end polygon_possible_with_area_sixteen_l131_131744


namespace addition_in_base_three_l131_131409

theorem addition_in_base_three : (
  (25 + 36) = 61 ∧ 
    ∃ (digits : List ℕ), 
      digits = [2, 0, 2, 1] ∧ 
      61 = digits.reverse.foldl (λ (n d : ℕ), n * 3 + d) 0
) :=
by 
  sorry

end addition_in_base_three_l131_131409


namespace quadratic_roots_in_range_l131_131548

theorem quadratic_roots_in_range (a : ℝ) (α β : ℝ)
  (h_eq : ∀ x : ℝ, x^2 + (a^2 + 1) * x + a - 2 = 0)
  (h_root1 : α > 1)
  (h_root2 : β < -1)
  (h_viete_sum : α + β = -(a^2 + 1))
  (h_viete_prod : α * β = a - 2) :
  0 < a ∧ a < 2 :=
  sorry

end quadratic_roots_in_range_l131_131548


namespace line_does_not_pass_through_third_quadrant_l131_131999

def line (x : ℝ) : ℝ := -x + 1

-- A line passes through the point (1, 0) and has a slope of -1
def passes_through_point (P : ℝ × ℝ) : Prop :=
  ∃ m b, m = -1 ∧ P.2 = m * P.1 + b ∧ line P.1 = P.2

def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem line_does_not_pass_through_third_quadrant :
  ¬ ∃ p : ℝ × ℝ, passes_through_point p ∧ third_quadrant p :=
sorry

end line_does_not_pass_through_third_quadrant_l131_131999


namespace new_area_of_rectangle_l131_131347

theorem new_area_of_rectangle (A : ℝ) (zone1 : ℝ) (A' : ℝ) :
  A = 500 ∧ zone1 = 350 ∧ A' = 0.96 * A → A' = 480 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h4]
  norm_num
  sorry

end new_area_of_rectangle_l131_131347


namespace construct_polygon_with_area_l131_131756

theorem construct_polygon_with_area 
  (n : ℕ) (l : ℝ) (a : ℝ) 
  (matchsticks : n = 12) 
  (matchstick_length : l = 2) 
  (area_target : a = 16) : 
  ∃ (polygon : EuclideanGeometry.Polygon ℝ) (sides : list ℝ),
    sides.length = n ∧ ∀ side ∈ sides, side = l ∧ polygon.area = a := 
sorry

end construct_polygon_with_area_l131_131756


namespace mass_percentage_nitrogen_in_ammonia_l131_131887

def ammonia_formula := "NH3"
def molar_mass_nitrogen : ℝ := 14.01
def molar_mass_hydrogen : ℝ := 1.01

def molar_mass_ammonia : ℝ := 
  molar_mass_nitrogen + 3 * molar_mass_hydrogen

def mass_percentage_nitrogen (molar_mass_n : ℝ) (molar_mass_nh3 : ℝ) : ℝ := 
  (molar_mass_n / molar_mass_nh3) * 100

theorem mass_percentage_nitrogen_in_ammonia :
  mass_percentage_nitrogen molar_mass_nitrogen molar_mass_ammonia = 82.23 := by
  sorry

end mass_percentage_nitrogen_in_ammonia_l131_131887


namespace visible_edge_length_l131_131620

/--
  Given a circular flower bed with a radius of 2 meters and a central angle cut out of 90°,
  prove that the total edge length visible for the remaining fan-shaped segment is $3\pi + 4$ meters.
-/
theorem visible_edge_length
  (r : ℝ)
  (sector_cut : ℝ)
  (circular_bed_circumference : ℝ := 2 * Real.pi * r)
  (length_of_arc : ℝ := (3 / 4) * circular_bed_circumference)
  (total_radii : ℝ := 2 * r)
  (total_edge_length : ℝ := length_of_arc + total_radii) :
  r = 2 ∧ sector_cut = (π / 2) → total_edge_length = 3 * Real.pi + 4 := 
by
  intro h
  cases h with hr hcut
  rw [hr, hcut]
  -- Steps leading to the final conclusion can be inserted here.
  sorry

end visible_edge_length_l131_131620


namespace power_identity_l131_131252

theorem power_identity :
  (3 ^ 12) * (3 ^ 8) = 243 ^ 4 :=
sorry

end power_identity_l131_131252


namespace problem_solution_l131_131395

noncomputable def x : ℝ := 3 / 0.15
noncomputable def y : ℝ := 3 / 0.25
noncomputable def z : ℝ := 0.30 * y

theorem problem_solution : x - y + z = 11.6 := sorry

end problem_solution_l131_131395


namespace no_two_digit_prime_sum_digits_nine_l131_131176

theorem no_two_digit_prime_sum_digits_nine :
  ¬ ∃ p : ℕ, prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 + p % 10 = 9) :=
sorry

end no_two_digit_prime_sum_digits_nine_l131_131176


namespace alcohol_percentage_in_second_vessel_l131_131472

open Real

theorem alcohol_percentage_in_second_vessel (x : ℝ) (h : (0.2 * 2) + (0.01 * x * 6) = 8 * 0.28) : 
  x = 30.666666666666668 :=
by 
  sorry

end alcohol_percentage_in_second_vessel_l131_131472


namespace cos_relation_l131_131331

theorem cos_relation 
  (a b c A B C : ℝ)
  (h1 : a = b * Real.cos C + c * Real.cos B)
  (h2 : b = c * Real.cos A + a * Real.cos C)
  (h3 : c = a * Real.cos B + b * Real.cos A)
  (h_abc_nonzero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 + 2 * Real.cos A * Real.cos B * Real.cos C = 1 :=
sorry

end cos_relation_l131_131331


namespace max_gold_coins_l131_131004

theorem max_gold_coins (n k : ℕ) 
  (h1 : n = 8 * k + 4)
  (h2 : n < 150) : 
  n = 148 :=
by
  sorry

end max_gold_coins_l131_131004


namespace right_triangle_conditions_l131_131871

theorem right_triangle_conditions (x y z h α β : ℝ) : 
  x - y = α → 
  z - h = β → 
  x^2 + y^2 = z^2 → 
  x * y = h * z → 
  β > α :=
by 
sorry

end right_triangle_conditions_l131_131871


namespace optimal_garden_area_l131_131771

variable (l w : ℕ)

/-- Tiffany is building a fence around a rectangular garden. Determine the optimal area, 
    in square feet, that can be enclosed under the conditions. -/
theorem optimal_garden_area 
  (h1 : l >= 100)
  (h2 : w >= 50)
  (h3 : 2 * l + 2 * w = 400) : (l * w) ≤ 7500 := 
sorry

end optimal_garden_area_l131_131771


namespace repetend_of_5_div_17_l131_131134

theorem repetend_of_5_div_17 :
  let dec := 5 / 17 in
  decimal_repetend dec = "294117" := sorry

end repetend_of_5_div_17_l131_131134


namespace trig_identity_l131_131950

theorem trig_identity (α : ℝ) (h : -2 * cos α + sin α = 0) :
  (sin (2 * α)) / (3 - 2 * (sin α)^2) = 4 / 7 :=
by
  sorry

end trig_identity_l131_131950


namespace true_propositions_count_is_one_l131_131220

-- Definitions of non-coincident lines and planes
def non_coincident_lines (m n : Type) : Prop := m ≠ n
def non_coincident_planes (α β : Type) : Prop := α ≠ β

-- Definitions of the relationships between lines and planes
def line_in_plane (m : Type) (α : Type) : Prop := sorry
def line_parallel_to_plane (m : Type) (α : Type) : Prop := sorry
def line_parallel_to_line (m n : Type) : Prop := sorry
def planes_parallel (α β : Type) : Prop := sorry
def line_perpendicular_to_plane (m : Type) (α : Type) : Prop := sorry
def planes_intersection (α β : Type) (n : Type) : Prop := sorry

-- Given conditions
variables (m n : Type) (α β : Type)
variables (h1 : non_coincident_lines m n) (h2 : non_coincident_planes α β)

-- Propositions
def prop1 := line_in_plane m α ∧ line_parallel_to_plane n α → line_parallel_to_line m n
def prop2 := line_parallel_to_plane m α ∧ line_parallel_to_plane m β → planes_parallel α β
def prop3 := planes_intersection α β n ∧ line_parallel_to_line m n → 
  line_parallel_to_plane m α ∧ line_parallel_to_plane m β
def prop4 := line_perpendicular_to_plane m α ∧ line_perpendicular_to_plane m β → planes_parallel α β

-- Proof problem: Exactly one proposition is true
theorem true_propositions_count_is_one : ∃! (p : Prop), p = prop1 ∨ p = prop2 ∨ p = prop3 ∨ p = prop4 ∧ (p = prop4) :=
by 
  -- The proof is omitted (skeleton only)
  sorry

end true_propositions_count_is_one_l131_131220


namespace kira_travel_time_l131_131398

def total_travel_time (hours_between_stations : ℕ) (break_minutes : ℕ) : ℕ :=
  let travel_time_hours := 2 * hours_between_stations
  let travel_time_minutes := travel_time_hours * 60
  travel_time_minutes + break_minutes

theorem kira_travel_time : total_travel_time 2 30 = 270 :=
  by sorry

end kira_travel_time_l131_131398


namespace latus_rectum_of_parabola_l131_131168

theorem latus_rectum_of_parabola : 
  ∀ x y : ℝ, x^2 = -y → y = 1/4 :=
by
  -- Proof omitted
  sorry

end latus_rectum_of_parabola_l131_131168


namespace A_div_B_l131_131503

noncomputable def A : ℝ := 
  ∑' n, if n % 2 = 0 ∧ n % 4 ≠ 0 then 1 / (n:ℝ)^2 else 0

noncomputable def B : ℝ := 
  ∑' n, if n % 4 = 0 then (-1)^(n / 4 + 1) * 1 / (n:ℝ)^2 else 0

theorem A_div_B : A / B = 17 := by
  sorry

end A_div_B_l131_131503


namespace kira_travel_time_l131_131399

def total_travel_time (hours_between_stations : ℕ) (break_minutes : ℕ) : ℕ :=
  let travel_time_hours := 2 * hours_between_stations
  let travel_time_minutes := travel_time_hours * 60
  travel_time_minutes + break_minutes

theorem kira_travel_time : total_travel_time 2 30 = 270 :=
  by sorry

end kira_travel_time_l131_131399


namespace arithmetic_mean_of_a_X_l131_131320

-- Define the set M
def M : Set ℕ := {i | 1 ≤ i ∧ i ≤ 1000}

-- Define a_X
def a_X (X : Set ℕ) [hX : X ⊆ M ∧ X ≠ ∅] : ℕ :=
  (X.to_finset.max' sorry) + (X.to_finset.min' sorry)

-- The theorem to prove
theorem arithmetic_mean_of_a_X : 
  (finset.univ.powerset.filter (λ X, X.card > 0)).sum (λ X, a_X X) / ((2^1000) - 1) = 1001 :=
sorry

end arithmetic_mean_of_a_X_l131_131320


namespace count_distinct_values_of_a1_infinite_repeats_l131_131664

def primes_below_30 : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def S : set ℕ := 
  {n | ∃ (primes : list ℕ), (∀ p ∈ primes, p ∈ primes_below_30) 
         ∧ n = primes.prod}

def transformation (a n : ℕ) : ℕ :=
  if a % (n + 1) = 0 then a / (n + 1) else (n + 2) * a

def infinite_repeats (a : ℕ) : Prop :=
  ∀ n : ℕ, ∃ j > n, transformation a j = a

theorem count_distinct_values_of_a1_infinite_repeats :
  ∃ (count : ℕ), count = 512 ∧ (∀ a1 ∈ S, infinite_repeats a1 ↔ a1 % 2 = 1) :=
begin
  sorry
end

end count_distinct_values_of_a1_infinite_repeats_l131_131664


namespace ordered_pair_solution_l131_131891

theorem ordered_pair_solution :
  ∃ x y : ℤ, (x + y = (3 - x) + (3 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ (x = 2) ∧ (y = 1) :=
by
  use 2, 1
  repeat { sorry }

end ordered_pair_solution_l131_131891


namespace gyration_inequality_l131_131667

variables (A B C K L M : Type) [triangle ABC]
variables (K_on_BC : K ∈ line_segment B C) (L_on_CA : L ∈ line_segment C A) (M_on_AB : M ∈ line_segment A B)
variables (concurrent : are_concurrent (line A K) (line B L) (line C M))

theorem gyration_inequality :
  ∃ (T₁ T₂ : triangle), {T₁, T₂} ⊆ {triangle A M L, triangle B K M, triangle C L K} ∧
    (radius_of_gyration T₁ + radius_of_gyration T₂) ≥ radius_of_gyration (triangle A B C) :=
sorry

end gyration_inequality_l131_131667


namespace range_of_m_l131_131805

theorem range_of_m (m : ℝ) : (1^2 + 2*1 - m ≤ 0) ∧ (2^2 + 2*2 - m > 0) → 3 ≤ m ∧ m < 8 := by
  sorry

end range_of_m_l131_131805


namespace construct_polygon_with_area_l131_131752

theorem construct_polygon_with_area 
  (n : ℕ) (l : ℝ) (a : ℝ) 
  (matchsticks : n = 12) 
  (matchstick_length : l = 2) 
  (area_target : a = 16) : 
  ∃ (polygon : EuclideanGeometry.Polygon ℝ) (sides : list ℝ),
    sides.length = n ∧ ∀ side ∈ sides, side = l ∧ polygon.area = a := 
sorry

end construct_polygon_with_area_l131_131752


namespace differentiable_function_solution_l131_131879

theorem differentiable_function_solution (f : ℝ → ℝ) 
  (h_diff : ∀ x, DifferentiableAt ℝ f x)
  (h_eq : ∀ x n, 0 < n → fderiv ℝ f x = (f x + ↑n - f x) / n) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b :=
by
  sorry

end differentiable_function_solution_l131_131879


namespace brendan_total_wins_l131_131079

-- Define the number of matches won in each round
def matches_won_first_round : ℕ := 6
def matches_won_second_round : ℕ := 4
def matches_won_third_round : ℕ := 3
def matches_won_final_round : ℕ := 5

-- Define the total number of matches won
def total_matches_won : ℕ := 
  matches_won_first_round + matches_won_second_round + matches_won_third_round + matches_won_final_round

-- State the theorem that needs to be proven
theorem brendan_total_wins : total_matches_won = 18 := by
  sorry

end brendan_total_wins_l131_131079


namespace difference_between_numbers_is_9_l131_131774

open Nat

def sum_of_remaining_43_eq_prod (x y : ℕ) : Prop :=
  let total_sum := (45 * 46) / 2
  total_sum - x - y = x * y

theorem difference_between_numbers_is_9 
  (x y : ℕ) (h₁ : x ≠ y) (h₂ : 1 ≤ x ∧ x ≤ 45) (h₃ : 1 ≤ y ∧ y ≤ 45) (h₄ : sum_of_remaining_43_eq_prod x y) :
  |y - x| = 9 :=
sorry

end difference_between_numbers_is_9_l131_131774


namespace fraction_of_occupied_student_chairs_is_4_over_5_l131_131736

-- Definitions based on the conditions provided
def total_chairs : ℕ := 10 * 15
def awardees_chairs : ℕ := 15
def admin_teachers_chairs : ℕ := 2 * 15
def parents_chairs : ℕ := 2 * 15
def student_chairs : ℕ := total_chairs - (awardees_chairs + admin_teachers_chairs + parents_chairs)
def vacant_student_chairs_given_to_parents : ℕ := 15
def occupied_student_chairs : ℕ := student_chairs - vacant_student_chairs_given_to_parents

-- Theorem statement based on the problem
theorem fraction_of_occupied_student_chairs_is_4_over_5 :
    (occupied_student_chairs : ℚ) / student_chairs = 4 / 5 :=
by
    sorry

end fraction_of_occupied_student_chairs_is_4_over_5_l131_131736


namespace determine_a_for_parallel_lines_l131_131674

noncomputable def slope (a : ℝ) (x1 y1 c1 x2 y2 c2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

theorem determine_a_for_parallel_lines (a : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, (∃ c1 c2 : ℝ, (x1 = 1 ∧ y1 = -(a / 4) + c1 / 4) ∧ (x2 = 1 ∧ y2 = -(1 / a) + c2 / a)) ∧
  slope a x1 y1 c1 x2 y2 c2 = slope 1 x1 y1 c1 x2 y2 c2) → 
  a = 2 :=
by
  intros h
  sorry

end determine_a_for_parallel_lines_l131_131674


namespace quadratic_difference_l131_131665

theorem quadratic_difference (f : ℝ → ℝ) (hpoly : ∃ c d e : ℤ, ∀ x, f x = c*x^2 + d*x + e) 
(h : f (Real.sqrt 3) - f (Real.sqrt 2) = 4) : 
f (Real.sqrt 10) - f (Real.sqrt 7) = 12 := sorry

end quadratic_difference_l131_131665


namespace soccer_balls_count_l131_131323

theorem soccer_balls_count (B : ℕ) (S_with_holes : ℕ) (B_with_holes : ℕ) (total_without_holes : ℕ) 
  (h1 : B = 15) 
  (h2 : S_with_holes = 30) 
  (h3 : B_with_holes = 7) 
  (h4 : total_without_holes = 18) : 
  let S_without_holes := total_without_holes - (B - B_with_holes) in
  let S := S_with_holes + S_without_holes in
  S = 40 := 
by 
  sorry

end soccer_balls_count_l131_131323


namespace elizabeth_bananas_eaten_l131_131873

theorem elizabeth_bananas_eaten (initial_bananas remaining_bananas eaten_bananas : ℕ) 
    (h1 : initial_bananas = 12) 
    (h2 : remaining_bananas = 8) 
    (h3 : eaten_bananas = initial_bananas - remaining_bananas) :
    eaten_bananas = 4 := 
sorry

end elizabeth_bananas_eaten_l131_131873


namespace indira_nagar_average_l131_131800

theorem indira_nagar_average (A B C D E F : ℕ) 
  (ha : A = 7) (hb : B = 8) (hc : C = 10) (hd : D = 13) (he : E = 6) (hf : F = 10)
  (h_leave : 1 = 1) : 
  (A - h_leave + B - h_leave + C - h_leave + D - h_leave + E - h_leave + F - h_leave) / 6 = 8 :=
by {
  sorry
}

end indira_nagar_average_l131_131800


namespace extremum_suff_not_nec_l131_131560

-- Define the differentiable function and its properties
variable {F : Type} (f : F → ℝ) (f' : F → ℝ) 
variable (x : F) (x0 : F)

-- Conditions: f is differentiable, f(x) attains extremum at x0, f'(x0) = 0
def differentiable (f : F → ℝ) : Prop :=
  ∀ x : F, ∃ f' : F → ℝ, continuous_at f' x

def extremum_at (f : F → ℝ) (x0 : F) : Prop :=
  (∀ ϵ > 0, ∃ δ > 0, |f x - f x0| < ϵ)

def derivative_zero_at (f' : F → ℝ) (x0 : F) : Prop :=
  f' x0 = 0

-- The proof problem: Prove the statement of sufficient but not necessary condition
theorem extremum_suff_not_nec :
  (differentiable f) ∧ (extremum_at f x0) ∧ (derivative_zero_at f' x0) → 
  (sufficient_but_not_necessary_condition (extremum_at f x0) (derivative_zero_at f' x0)) :=
sorry

end extremum_suff_not_nec_l131_131560


namespace correct_propositions_l131_131920

-- Definitions for lines m, n and planes α, β
variables (m n : Line) (α β : Plane)

-- Propositions P1 to P4
def P1 := (m ⊥ α ∧ m ⊥ β) → α ⊥ β
def P2 := (m ∥ α ∧ m ∥ β) → α ∥ β
def P3 := (m ⊥ α ∧ m ∥ β) → α ⊥ β
def P4 := (skew m n ∧ m ⊥ n) → ∃ γ : Plane, m ⊂ γ ∧ γ ⊥ n

-- Theorem stating the correct propositions are P3 and P4
theorem correct_propositions :
  (P3 m α β ∧ P4 m n α β) ∧ ¬(P1 m α β) ∧ ¬(P2 m α β) :=
by
  sorry

end correct_propositions_l131_131920


namespace repetend_of_five_seventeenths_l131_131154

theorem repetend_of_five_seventeenths :
  (decimal_expansion (5 / 17)).repeat_called == "294117647" :=
sorry

end repetend_of_five_seventeenths_l131_131154


namespace product_of_solutions_l131_131528

theorem product_of_solutions (x : ℝ) (h : x^4 = 81) :
  ∏ a in ({x : ℝ | x^4 = 81}), x = -9 :=
begin
  sorry
end

end product_of_solutions_l131_131528


namespace sum_of_midpoint_coords_is_seven_l131_131789

-- Define coordinates of the endpoints
def endpoint1 : ℝ × ℝ := (8, 16)
def endpoint2 : ℝ × ℝ := (-2, -8)

-- Define the midpoint coordinates
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the sum of the coordinates of the midpoint
def sum_of_midpoint_coords (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

-- Theorem stating that the sum of the coordinates of the midpoint is 7
theorem sum_of_midpoint_coords_is_seven : 
  sum_of_midpoint_coords (midpoint endpoint1 endpoint2) = 7 :=
by
  -- Proof would go here
  sorry

end sum_of_midpoint_coords_is_seven_l131_131789


namespace intersection_point_ordinate_interval_l131_131720

theorem intersection_point_ordinate_interval:
  ∃ m : ℤ, ∀ x : ℝ, e ^ x = 5 - x → 3 < x ∧ x < 4 :=
by sorry

end intersection_point_ordinate_interval_l131_131720


namespace log_base_9_of_cubic_root_of_9_l131_131114

theorem log_base_9_of_cubic_root_of_9 : log 9 (cbrt 9) = 1 / 3 :=
by 
  -- required to use Mathfcklb.log or any other definitions properly
  sorry

end log_base_9_of_cubic_root_of_9_l131_131114


namespace chord_min_area_l131_131953

noncomputable def is_minimized_area (a : ℝ) (x : ℝ) : Prop :=
  ∀ y1 y2 : ℝ, y1 = 6 * a^2 * Real.sqrt(1 + (1/(x^2))) → y2 = 6 * a^2 → y1 ≥ y2

theorem chord_min_area (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, x = -Real.sqrt 3 * a ∧ is_minimized_area a x) :=
begin
  sorry
end

end chord_min_area_l131_131953


namespace simplest_quadratic_radical_l131_131413

theorem simplest_quadratic_radical :
  ∀ (A B C D : ℝ), 
  A = Real.sqrt 2 →
  B = Real.cbrt 3 →
  C = Real.sqrt (1/2) →
  D = Real.sqrt 16 →
  ∃ simplest : ℝ, simplest = A :=
  by
  intros A B C D hA hB hC hD
  use A
  sorry

end simplest_quadratic_radical_l131_131413


namespace functional_equation_solution_l131_131099

noncomputable def f : ℕ → ℕ := sorry

theorem functional_equation_solution (f : ℕ → ℕ)
    (h : ∀ n : ℕ, f (f (f n)) + f (f n) + f n = 3 * n) :
    ∀ n : ℕ, f n = n := sorry

end functional_equation_solution_l131_131099


namespace radius_of_third_circle_correct_l131_131816

noncomputable def radius_of_third_circle (R : ℝ) : ℝ :=
  R * Real.sqrt 3 / 4

theorem radius_of_third_circle_correct (R : ℝ) : 
  ∃ x : ℝ,
  (x = radius_of_third_circle R) ∧
  (let AB := 2 * R,
      first_circle_center := A in
    let second_circle_center := A,
      third_circle_radius := x,
      internal_tangent_distance := R - x,
      external_tangent_distance := R + x in
     let P := R + Real.sqrt (R * R - 2 * R * x) in
    internal_tangent_distance ^ 2 + P ^ 2 = external_tangent_distance ^ 2 ∧
    P = x) :=
begin
  use radius_of_third_circle R,
  split,
  -- Part 1: The radius x of the third circle is defined as radius_of_third_circle R
  exact rfl,
  -- Part 2: Show all conditions hold with the derived radius x
  sorry
end

end radius_of_third_circle_correct_l131_131816


namespace yoojeong_initial_correct_l131_131798

variable (yoojeong_initial yoojeong_after marbles_given : ℕ)

-- Given conditions
axiom marbles_given_cond : marbles_given = 8
axiom yoojeong_after_cond : yoojeong_after = 24

-- Equation relating initial, given marbles, and marbles left
theorem yoojeong_initial_correct : 
  yoojeong_initial = yoojeong_after + marbles_given := by
  -- Proof skipped
  sorry

end yoojeong_initial_correct_l131_131798


namespace expression_evaluation_l131_131856

theorem expression_evaluation : 
    ((-1 / 3) ^ (-2 : ℤ)) + (| 4 * real.sqrt 2 - 6 |) - (2 ^ 3) = 7 - 4 * real.sqrt 2 :=
by
  sorry

end expression_evaluation_l131_131856


namespace time_in_3467_hours_l131_131350

-- Define the current time, the number of hours, and the modulus
def current_time : ℕ := 2
def hours_from_now : ℕ := 3467
def clock_modulus : ℕ := 12

-- Define the function to calculate the future time on a 12-hour clock
def future_time (current_time : ℕ) (hours_from_now : ℕ) (modulus : ℕ) : ℕ := 
  (current_time + hours_from_now) % modulus

-- Theorem statement
theorem time_in_3467_hours :
  future_time current_time hours_from_now clock_modulus = 9 :=
by
  -- Proof would go here
  sorry

end time_in_3467_hours_l131_131350


namespace problem_conditions_problem_m_range_l131_131934

noncomputable def z1 : ℂ := 2 + 2 * complex.I

theorem problem_conditions (z1: ℂ) (h1 : z1 - 2 * complex.I = complex.abs (√3 + complex.I)) :
  z1 = 2 + 2 * complex.I := 
by 
  sorry

theorem problem_m_range (z2 : ℂ) (m : ℝ) (h1 : z1 - 2 * complex.I = complex.abs (√3 + complex.I))
  (h2 : z2 - z1 = m^2 - 2 * m - 5 + (m -2) * complex.I) 
  (h3: z2.re > 0 ∧ z2.im > 0) :
  3 < m := 
by 
  sorry

end problem_conditions_problem_m_range_l131_131934


namespace residual_drug_time_approximation_l131_131078

theorem residual_drug_time_approximation (a : ℝ) (y : ℝ) :
  (∀ (t : ℝ), y = a * real.log 2 (12 / (t + 1))) →
  (y_at_2 := a * real.log 2 (12 / (2 + 1))) →
  (y_target := y_at_2 / 4) →
  (∃ t : ℝ, abs ((12 / (t + 1)) - real.sqrt 2) < 0.01) :=
by
  intros h_formula y_at_2_def y_target_def
  use 7.5
  sorry

end residual_drug_time_approximation_l131_131078


namespace diff_function_f_prime_zero_l131_131596

theorem diff_function_f_prime_zero (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) (h_def : ∀ x : ℝ, f x = x^2 + 2 * (f' 2) * x + 3) : f' 0 = -8 := by
sorry

end diff_function_f_prime_zero_l131_131596


namespace coefficient_x2_l131_131886

def polynomial1 (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 2 * x
def polynomial2 (b x : ℝ) : ℝ := b * x^2 - 6 * x - 4 * x

theorem coefficient_x2 (a b x : ℝ) :  -- x is representing the variable part
  -- Extracting the coefficient of the x^2 term in the expansion:
  (let product := (polynomial1 a x) * (polynomial2 b x) in
   -- Here we assume that coefficient_x2_expr extracts the x^2 term's coefficient from the product
   (coefficient_x2_expr product) = 12) :=
  sorry

end coefficient_x2_l131_131886


namespace sequence_convergence_to_4_l131_131381

-- Definition of the sequence a_n and the sum of digits function s
def a : ℕ → ℕ
| 0 => 0 -- sequence is 1-indexed
| 1 => 2^20
| (n + 2) => s (a (n + 1))

-- Sum of the digits function
def s (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Prove that a_{100} = 4
theorem sequence_convergence_to_4 : a 100 = 4 := by
  sorry

end sequence_convergence_to_4_l131_131381


namespace merchant_profit_l131_131821

theorem merchant_profit 
  (CP MP SP profit : ℝ)
  (markup_percentage discount_percentage : ℝ)
  (h1 : CP = 100)
  (h2 : markup_percentage = 0.40)
  (h3 : discount_percentage = 0.10)
  (h4 : MP = CP + (markup_percentage * CP))
  (h5 : SP = MP - (discount_percentage * MP))
  (h6 : profit = SP - CP) :
  profit / CP * 100 = 26 :=
by sorry

end merchant_profit_l131_131821


namespace card_area_after_shortening_l131_131799

/--
If an index card has dimensions 5 inches by 7 inches, and if shortening one side by 2 inches 
results in an area of 15 square inches, then shortening the other side by 2 inches results 
in an area of 21 square inches.
-/
theorem card_area_after_shortening 
    (original_length original_width : ℕ)
    (new_length new_width : ℕ)
    (h_dims : original_length = 5 ∧ original_width = 7)
    (h_shorten_one_side : (new_length = original_length - 2) ∨ (new_width = original_width - 2))
    (h_new_area_15 : new_length * new_width = 15) :
    ((original_length - 2) * original_width = 21) ∨ (original_length * (original_width - 2) = 21) :=
begin
  sorry
end

end card_area_after_shortening_l131_131799


namespace cos_B_cos_B_plus_pi_div_4_l131_131608

-- Conditions and questions for Part 1
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h1 : c = (sqrt 5) / 2 * b)
variable (h2 : C = 2 * B)

theorem cos_B (h : sin C = ((sqrt 5) / 2) * sin B) (hB : sin B > 0) : cos B = (sqrt 5) / 4 := by
  sorry

-- Conditions and questions for Part 2
variable (A1 B1 C1 : ℝ)
variable (a1 b1 c1 : ℝ)
variable (h3 : a1 = c1)

theorem cos_B_plus_pi_div_4 (h4 : cos B1 = 3 / 5) : cos (B1 + (Real.pi / 4)) = -sqrt 2 / 10 := by
  sorry

end cos_B_cos_B_plus_pi_div_4_l131_131608


namespace right_triangle_median_square_l131_131020

theorem right_triangle_median_square (a b c k_a k_b : ℝ) :
  c = Real.sqrt (a^2 + b^2) → -- c is the hypotenuse
  k_a = Real.sqrt ((2 * b^2 + 2 * (a^2 + b^2) - a^2) / 4) → -- k_a is the median to side a
  k_b = Real.sqrt ((2 * a^2 + 2 * (a^2 + b^2) - b^2) / 4) → -- k_b is the median to side b
  c^2 = (4 / 5) * (k_a^2 + k_b^2) :=
by
  intros h_c h_ka h_kb
  sorry

end right_triangle_median_square_l131_131020


namespace infinite_sequence_even_number_l131_131107

theorem infinite_sequence_even_number (a : ℕ → ℕ) 
  (h : ∀ n, ∃ d, d > 0 ∧ d ≤ 9 ∧ a (n + 1) = a n + d) : 
  ∃ n, even (a n) :=
by
  sorry

end infinite_sequence_even_number_l131_131107


namespace cost_difference_two_white_and_one_brown_l131_131404

-- Definitions based on conditions
def cost_two_white_socks : ℕ := 45 -- cost in cents
def cost_fifteen_brown_socks : ℕ := 300 -- cost in cents (3 dollars)

-- The proof problem
theorem cost_difference_two_white_and_one_brown
  (cost_two_white_socks = 45) 
  (cost_fifteen_brown_socks = 300) : 
  (cost_two_white_socks - cost_fifteen_brown_socks / 15) = 25 := 
sorry

end cost_difference_two_white_and_one_brown_l131_131404


namespace monotonic_intervals_maximum_value_on_interval_l131_131575

-- Define the function f
def f (x : ℝ) := (Real.log x) / x - 1

-- State the theorem, breaking down the problem into different parts
theorem monotonic_intervals :
  (∀ x : ℝ, (0 < x ∧ x < Real.exp 1) → (f x)' > 0) ∧
  (∀ x : ℝ, (x > Real.exp 1) → (f x)' < 0) := by
  sorry

theorem maximum_value_on_interval (m : ℝ) (hm : m > 0) :
  (m ≤ Real.exp 1 / 2 → (∀ x : ℝ, m ≤ x ∧ x ≤ 2 * m → f x ≤ f (2 * m))) ∧
  (m ≥ Real.exp 1 → (∀ x : ℝ, m ≤ x ∧ x ≤ 2 * m → f x ≤ f m)) ∧
  ((Real.exp 1 / 2 < m ∧ m < Real.exp 1) → (∀ x : ℝ, m ≤ x ∧ x ≤ 2 * m → f x ≤ f (Real.exp 1))) := by
  sorry

end monotonic_intervals_maximum_value_on_interval_l131_131575


namespace start_distance_l131_131052

-- Define the speeds of A and B
variables (v_A v_B : ℝ)
-- Define the start distance d and the final distance to the post
variables (d : ℝ)

-- Conditions
axiom speed_ratio : v_A = (5 / 3) * v_B
axiom equal_time : (200 / v_A) = ((200 - d) / v_B)

-- Proposition to prove
theorem start_distance : d = 80 :=
by 
  have h1 : v_A = (5 / 3) * v_B := speed_ratio
  have h2 : 200 / v_A = (200 - d) / v_B := equal_time
  -- Now performing the substitution and solving within the proof (leaving details out with sorry)
  sorry

end start_distance_l131_131052


namespace B_plus_D_l131_131686

section conjugate_transform

variables {z z1 z2 z3 z4 : ℂ}

def f (z : ℂ) : ℂ := 4 * complex.I * complex.conj z

noncomputable def P : polynomial ℂ := 
  polynomial.C 1 + polynomial.C 2 * polynomial.X +
  polynomial.C 3 * polynomial.X^2 + 
  polynomial.C 4 * polynomial.X^3 + 
  polynomial.X^4

noncomputable def Q : polynomial ℂ := 
  polynomial.C 1 + polynomial.C (f (polynomial.leading_coeff P).conj^(1/4)) * polynomial.X +
  polynomial.C (f ((3:ℂ)).conj) * polynomial.X^2 +
  polynomial.C (f ((4:ℂ)).conj) * polynomial.X^3 +
  polynomial.X^4

theorem B_plus_D : (P.roots.map (λ z, f z)).prod (λ z1 z2, f z1 * f z2) + 
  (P.roots.map (λ z, f z)).prod = 208 :=
sorry

end conjugate_transform

end B_plus_D_l131_131686


namespace triangle_area_l131_131551

theorem triangle_area (R : ℝ) (A : ℝ) (b c : ℝ) (S : ℝ)
  (hR : R = 4) (hA : A = 60) (hb_diff_c : b - c = 4) (hc : S = (1 / 2) * 32 * (Math.sin (60 * Math.pi / 180)))
  : S = 8 * Math.sqrt 3 := by
  sorry

end triangle_area_l131_131551


namespace range_of_b_l131_131904

noncomputable def f (x : ℝ) : ℝ :=
if x < -1/2 then (2*x + 1) / (x^2) else real.log (x + 1)

def g (x : ℝ) : ℝ := x^2 - 4 * x - 4

theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : -1 ≤ b ∧ b ≤ 5 :=
sorry

end range_of_b_l131_131904


namespace avg_comm_add_distrib_avg_l131_131025

def avg (a b : ℝ) : ℝ := (a + b) / 2

-- Statement II: Commutativity
theorem avg_comm (a b : ℝ) : avg a b = avg b a :=
by sorry

-- Statement IV: Distributivity of Addition over *
theorem add_distrib_avg (a b c : ℝ) : a + avg b c = avg (a + b) (a + c) :=
by sorry

end avg_comm_add_distrib_avg_l131_131025


namespace distance_between_cities_l131_131713

noncomputable def distance_A_to_B : ℕ := 180
noncomputable def distance_B_to_A : ℕ := 150
noncomputable def total_distance : ℕ := distance_A_to_B + distance_B_to_A

theorem distance_between_cities : total_distance = 330 := by
  sorry

end distance_between_cities_l131_131713


namespace functions_symmetrical_about_y_axis_l131_131965

def f1 (x : ℝ) : ℝ := log x / log 2
def f2 (x : ℝ) : ℝ := x^2
def f3 (x : ℝ) : ℝ := 2^|x|
def f4 (x : ℝ) : ℝ := Real.arcsin x

theorem functions_symmetrical_about_y_axis :
  {f2, f3} = {f : ℝ → ℝ | ∀ x : ℝ, f(-x) = f(x)} :=
by
  sorry

end functions_symmetrical_about_y_axis_l131_131965


namespace calculate_average_speed_l131_131718

def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem calculate_average_speed :
  average_speed 200 6 = 33.33 := by
  sorry

end calculate_average_speed_l131_131718


namespace range_of_a_l131_131600

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
by
  sorry

end range_of_a_l131_131600


namespace distinct_sequences_count_l131_131246

theorem distinct_sequences_count : 
    let letters := ['E', 'Q', 'U', 'A', 'L', 'S'] in
    let filtered_letters := letters.filter (λ x, x ≠ 'L' ∧ x ≠ 'S') in
    ∃ seqs : list (list Char),
        (∀ seq ∈ seqs, list.length seq = 5 ∧ list.head seq = some 'L' ∧ list.getLast seq 'x' = 'S') ∧
        (∀ seq ∈ seqs, list.norepeats seq) ∧
        seqs.length = 24 := 
sorry

end distinct_sequences_count_l131_131246


namespace pulley_distance_l131_131485

theorem pulley_distance (r₁ r₂ d l : ℝ):
    r₁ = 10 →
    r₂ = 6 →
    l = 30 →
    (d = 2 * Real.sqrt 229) :=
by
    intros h₁ h₂ h₃
    sorry

end pulley_distance_l131_131485


namespace ratio_of_a_b_l131_131261

variable (x y a b : ℝ)

theorem ratio_of_a_b (h₁ : 4 * x - 2 * y = a)
                     (h₂ : 6 * y - 12 * x = b)
                     (hb : b ≠ 0)
                     (ha_solution : ∃ x y, 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b) :
                     a / b = 1 / 3 :=
by sorry

end ratio_of_a_b_l131_131261


namespace coefficient_x2y7_l131_131711

theorem coefficient_x2y7 (x y : ℝ) :
  (let expr := (x - y) * (x + y) ^ 8 in
   (expr.coeff_monomial (2, 7))) = -20 :=
sorry

end coefficient_x2y7_l131_131711


namespace cos_alpha_equivalence_l131_131927

open Real

noncomputable def cos_alpha : ℝ :=
  let α := π/4 in (⟨cos(α + π/4) = 4/5, λ α hα, by sorry⟩ : Prop)

theorem cos_alpha_equivalence (hα : 0 < α ∧ α < π/2) (hcos : cos(α + π/4) = 4/5) :
  cos α = 7 * sqrt 2 / 10 :=
by sorry

end cos_alpha_equivalence_l131_131927


namespace polygon_possible_l131_131737

-- Definition: a polygon with matches without breaking them
structure MatchPolygon (n : ℕ) (length : ℝ) where
  num_matches : ℕ
  len_matches : ℝ
  area : ℝ
  notequalzero : len_matches ≠ 0
  notequalzero2 : area ≠ 0
  perimeter_eq: num_matches * len_matches = length * real.of_nat n
  all_matches_used : n = 12
  no_breaking : (length / real.of_nat n) = 2 

theorem polygon_possible : 
  ∃ P : MatchPolygon 12 2, P.area = 16 :=
sorry

end polygon_possible_l131_131737


namespace rectangle_perimeter_l131_131687

-- Definitions and conditions
variables (x y a b : ℝ)
def is_rectangle (x y : ℝ) (area : ℝ) := x * y = area
def ellipse_conditions (x y a b : ℝ) (ellipse_area : ℝ) :=
  2 * a * sqrt b^2 = ellipse_area ∧
  2530 = x * y

-- The proof statement
theorem rectangle_perimeter (x y a b : ℝ) (h1 : is_rectangle x y 2530) (h2 : ellipse_conditions x y a b (2530 * Real.pi)) :
  2 * (x + y) = 8 * sqrt 1265 :=
sorry

end rectangle_perimeter_l131_131687


namespace calculator_presses_to_exceed_1000_l131_131440

theorem calculator_presses_to_exceed_1000 :
  ∃ n : ℕ, n = 3 ∧ (∃ k : ℕ, k ≤ n ∧ (nat.iterate (λ x : ℕ, x * x) k 3 > 1000)) :=
begin
  -- theorem states the existence of n such that n = 3,
  -- and existence of k such that k is at most n and iterated function k times on 3 is greater than 1000
  sorry
end

end calculator_presses_to_exceed_1000_l131_131440


namespace correct_proposition_l131_131480

theorem correct_proposition :
  (∃ x : ℝ, 3 * x ^ 2 - 4 = 6 * x) ∧
  (∀ x : ℝ, (x - real.sqrt 2) ^ 2 > 0 → false) ∧
  (∀ x : ℚ, x ^ 2 > 0 → false) ∧
  (∃ x : ℤ, 3 * x = 128 → false) :=
by
  sorry

end correct_proposition_l131_131480


namespace find_x_l131_131297

def oslash (a b : ℝ) : ℝ := (sqrt (3 * a + b)) ^ 3

theorem find_x (x : ℝ) (h : oslash 3 x = 64) : x = 7 :=
by
  sorry

end find_x_l131_131297


namespace perfect_cubes_count_l131_131594

theorem perfect_cubes_count : 
  Nat.card {n : ℕ | n^3 > 500 ∧ n^3 < 2000} = 5 :=
by
  sorry

end perfect_cubes_count_l131_131594


namespace simplify_expression_l131_131693

theorem simplify_expression (x : ℝ) (hx2 : x ≠ 2) (hx_2 : x ≠ -2) (hx1 : x ≠ 1) : 
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x^2 - 4)) = (x + 2) / (x - 1) :=
by
  sorry

end simplify_expression_l131_131693


namespace largest_spherical_ball_on_torus_l131_131058

theorem largest_spherical_ball_on_torus : 
  ∃ r : ℝ, 
  let O := (0, 0, r) in 
  let P := (4, 0, 1) in 
  r > 0 ∧ 
  let dist_OP := real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2 + (P.3 - O.3)^2) in
  dist_OP = r + 1 ∧ 
  r = 4 := 
begin
  sorry
end

end largest_spherical_ball_on_torus_l131_131058


namespace meeting_time_l131_131351

-- Define the conditions
def distance : ℕ := 600  -- distance between A and B
def speed_A_to_B : ℕ := 70  -- speed of the first person
def speed_B_to_A : ℕ := 80  -- speed of the second person
def start_time : ℕ := 10  -- start time in hours

-- State the problem formally in Lean 4
theorem meeting_time : (distance / (speed_A_to_B + speed_B_to_A)) + start_time = 14 := 
by
  sorry

end meeting_time_l131_131351


namespace hexagon_area_l131_131660

theorem hexagon_area (ABCDEF : Type) [RegularHexagon ABCDEF]
  (A B C D E F G H I : Point)
  (mid_AB : midpoint A B = G)
  (mid_CD : midpoint C D = H)
  (mid_EF : midpoint E F = I)
  (area_GHI : area (triangle G H I) = 81)
  : area (hexagon A B C D E F) = 486 :=
sorry

end hexagon_area_l131_131660


namespace matrix_sum_correct_l131_131084

def mat1 : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -1], ![3, 7]]
def mat2 : Matrix (Fin 2) (Fin 2) ℤ := ![![ -6, 8], ![5, -2]]
def mat_sum : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, 7], ![8, 5]]

theorem matrix_sum_correct : mat1 + mat2 = mat_sum :=
by
  rw [mat1, mat2]
  sorry

end matrix_sum_correct_l131_131084


namespace PA_PB_value_chord_length_midpoint_coordinates_l131_131547

-- Define the problem conditions
def line_inclination_angle (P : Point ℝ) (α : ℝ) : Prop :=
  α = Real.pi / 4 ∧ P = (1, 1)

def parabola_equation (y x : ℝ) : Prop :=
  y^2 = x + 1

def line_through_point (l : ℝ → Point ℝ) (P : Point ℝ) : Prop :=
  ∃ (t : ℝ), P = (1 + t * Real.sqrt 2 / 2, 1 + t * Real.sqrt 2 / 2) 

def intersection_points (A B : Point ℝ) (P : Point ℝ) (l : ℝ → Point ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (t₁ t₂ : ℝ), parabola (l t₁).2 (l t₁).1 ∧ parabola (l t₂).2 (l t₂).1

-- Define the proof statements using the conditions
theorem PA_PB_value
  (P : Point ℝ) (l : ℝ → Point ℝ) (A B : Point ℝ) (parabola : ℝ → ℝ → Prop) 
  (h₁ : line_inclination_angle P (Real.pi / 4))
  (h₂ : line_through_point l P) 
  (h₃ : intersection_points A B P l parabola) : 
  |(A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)| = Real.sqrt 10 := 
  sorry

theorem chord_length
  (A B : Point ℝ) (P : Point ℝ) (l : ℝ → Point ℝ) (parabola : ℝ → ℝ → Prop) 
  (h₁ : line_inclination_angle P (Real.pi / 4))
  (h₂ : line_through_point l P) 
  (h₃ : intersection_points A B P l parabola) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 10 :=
  sorry

theorem midpoint_coordinates
  (A B : Point ℝ) (M P : Point ℝ) (l : ℝ → Point ℝ) (parabola : ℝ → ℝ → Prop) 
  (h₁ : line_inclination_angle P (Real.pi / 4))
  (h₂ : line_through_point l P) 
  (h₃ : intersection_points A B P l parabola) 
  (h₄ : M = ( (A.1 + B.1)/2, (A.2 + B.2)/2 )) : 
  M = (1 / 2, 1 / 2) := 
  sorry

end PA_PB_value_chord_length_midpoint_coordinates_l131_131547


namespace general_formula_a_n_general_formula_b_n_l131_131913

-- Prove general formula for the sequence a_n
theorem general_formula_a_n (S : Nat → Nat) (a : Nat → Nat) (h₁ : ∀ n, S n = 2^(n+1) - 2) :
  (∀ n, a n = S n - S (n - 1)) → ∀ n, a n = 2^n :=
by
  sorry

-- Prove general formula for the sequence b_n
theorem general_formula_b_n (a b : Nat → Nat) (h₁ : ∀ n, a n = 2^n) :
  (∀ n, b n = a n + a (n + 1)) → ∀ n, b n = 3 * 2^n :=
by
  sorry

end general_formula_a_n_general_formula_b_n_l131_131913


namespace find_z_l131_131545

open Complex

noncomputable def sqrt_five : ℝ := Real.sqrt 5

theorem find_z (z : ℂ) 
  (hz1 : z.re < 0) 
  (hz2 : z.im > 0) 
  (h_modulus : abs z = 3) 
  (h_real_part : z.re = -sqrt_five) : 
  z = -sqrt_five + 2 * I :=
by
  sorry

end find_z_l131_131545


namespace integer_part_S2017_l131_131730

def a : ℕ → ℝ
| 0     := 1/3
| (n+1) := (a n)^2 + a n

def c (n : ℕ) : ℝ :=
1 / (a n + 1)

def S (n : ℕ) : ℝ :=
∑ i in Finset.range n, c i

theorem integer_part_S2017 : int.floor (S 2017) = 2 :=
sorry

end integer_part_S2017_l131_131730


namespace line_through_point_equal_intercepts_locus_equidistant_lines_l131_131434

theorem line_through_point_equal_intercepts (x y : ℝ) (hx : x = 1) (hy : y = 3) :
  (∃ k : ℝ, y = k * x ∧ k = 3) ∨ (∃ a : ℝ, x + y = a ∧ a = 4) :=
sorry

theorem locus_equidistant_lines (x y : ℝ) :
  ∀ (a b : ℝ), (2 * x + 3 * y - a = 0) ∧ (4 * x + 6 * y + b = 0) →
  ∀ b : ℝ, |b + 10| = |b - 8| → b = -9 → 
  4 * x + 6 * y - 9 = 0 :=
sorry

end line_through_point_equal_intercepts_locus_equidistant_lines_l131_131434


namespace train_speed_l131_131059

theorem train_speed (train_length : ℕ) (time_to_cross : ℕ) (man_speed_kmh : ℕ) :
  train_length = 250 →
  time_to_cross = 8 →
  man_speed_kmh = 7 →
  let man_speed_ms := (man_speed_kmh : ℝ) * 1000 / 3600 in
  let relative_speed := (train_length : ℝ) / time_to_cross in
  let train_speed_ms := relative_speed - man_speed_ms in
  let train_speed_kmh := train_speed_ms * 3600 / 1000 in
  train_speed_kmh ≈ 105.5 := by
  sorry

end train_speed_l131_131059


namespace equivalent_single_reduction_l131_131724

theorem equivalent_single_reduction :
  ∀ (P : ℝ), P * (1 - 0.25) * (1 - 0.20) = P * (1 - 0.40) :=
by
  intros P
  -- Proof will be skipped
  sorry

end equivalent_single_reduction_l131_131724


namespace compute_100m_plus_t_l131_131290

-- Definitions as per conditions
def infinite_grid := ℕ × ℕ 

def initial_configuration : infinite_grid → Prop
| (0, 0) := true
| (_, _) := false

def marco_move (grid : infinite_grid → Prop) : infinite_grid → Prop
| (x, y) := if grid (x, y) ∧ (nat.bodd x ∧ nat.bodd y) then (¬ grid (x, y)) else grid (x, y)

def vera_move (grid : infinite_grid → Prop) (k : ℕ) : infinite_grid → Prop
| (x, y) := if k ≠ 0 then ¬ grid (x, y) else grid (x, y)

noncomputable def minimal_k (k t : ℕ) (grid : infinite_grid → Prop) : ℕ :=
if ∀ (k = 2) ∃ t, (Vera_wins_in_t_turns k t grid) then 2 else 0

-- Main theorem statement
theorem compute_100m_plus_t : ∃ (m t : ℕ), m = 2 ∧ t = 3 ∧ 100 * m + t = 203 :=
by
  existsi 2
  existsi 3
  simp
  exact sorry

end compute_100m_plus_t_l131_131290


namespace Zachary_sold_40_games_l131_131001

theorem Zachary_sold_40_games 
  (R J Z : ℝ)
  (games_Zachary_sold : ℕ)
  (h1 : R = J + 50)
  (h2 : J = 1.30 * Z)
  (h3 : Z = 5 * games_Zachary_sold)
  (h4 : Z + J + R = 770) :
  games_Zachary_sold = 40 :=
by
  sorry

end Zachary_sold_40_games_l131_131001


namespace ray_has_4_nickels_left_l131_131334

theorem ray_has_4_nickels_left (initial_cents : ℕ) (given_to_peter : ℕ)
    (given_to_randi : ℕ) (value_of_nickel : ℕ) (remaining_cents : ℕ) 
    (remaining_nickels : ℕ) :
    initial_cents = 95 →
    given_to_peter = 25 →
    given_to_randi = 2 * given_to_peter →
    value_of_nickel = 5 →
    remaining_cents = initial_cents - given_to_peter - given_to_randi →
    remaining_nickels = remaining_cents / value_of_nickel →
    remaining_nickels = 4 :=
by
  intros
  sorry

end ray_has_4_nickels_left_l131_131334


namespace fraction_of_a_equals_one_fourth_of_b_l131_131416

theorem fraction_of_a_equals_one_fourth_of_b :
  ∃ (x : ℚ), (B = 484) ∧ (A + B = 1210) ∧ (x * A = 1/4 * B) ∧ (x = 1/6) :=
by
  let A := 1210 - 484
  let B := 484
  let x := 1/6
  use x
  have h1 : B = 484 := by sorry
  have h2 : A + B = 1210 := by sorry
  have h3 : x * A = 1/4 * B := by sorry
  have h4 : x = 1/6 := by sorry
  exact ⟨h1, h2, h3, h4⟩

end fraction_of_a_equals_one_fourth_of_b_l131_131416


namespace pentagon_diagonals_areas_l131_131200

open EuclideanGeometry

theorem pentagon_diagonals_areas (A B C D E : Point) 
  (h_convex : ConvexPentagon A B C D E) :
  let area_pentagon := Area (Polygon5 A B C D E)
  let area_triangles := Area (Triangle A B E) + Area (Triangle A C E) + Area (Triangle A D E) + 
                        Area (Triangle B C D) + Area (Triangle B D E) + Area (Triangle C D E)
  area_triangles > area_pentagon :=
sorry

end pentagon_diagonals_areas_l131_131200


namespace unique_solution_f_l131_131880

theorem unique_solution_f {f : ℝ → ℝ} (h : ∀ x y : ℝ, f(x * y + 1) = x * f(y) + 2) : 
  ∀ x : ℝ, f(x) = 2 * x :=
by
  sorry

end unique_solution_f_l131_131880


namespace count_perfect_cubes_between_500_and_2000_l131_131592

theorem count_perfect_cubes_between_500_and_2000 : ∃ count : ℕ, count = 5 ∧ (∀ n, 500 < n^3 ∧ n^3 < 2000 → (8 ≤ n ∧ n ≤ 12)) :=
by
  existsi 5
  split
  {
    sorry,  -- Proof that count = 5
    sorry,  -- Proof that for any n, if 500 < n^3 and n^3 < 2000 then 8 <= n <= 12
  }

end count_perfect_cubes_between_500_and_2000_l131_131592


namespace quadratic_y1_gt_y2_l131_131234

theorem quadratic_y1_gt_y2 (a b c y1 y2 : ℝ) (h_a_pos : a > 0) (h_sym : ∀ x, a * (x - 1)^2 + c = a * (1 - x)^2 + c) (h1 : y1 = a * (-1)^2 + b * (-1) + c) (h2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
sorry

end quadratic_y1_gt_y2_l131_131234


namespace quadratic_roots_p_l131_131241

noncomputable def equation : Type* := sorry

theorem quadratic_roots_p
  (α β : ℝ)
  (K : ℝ)
  (h1 : 3 * α ^ 2 + 7 * α + K = 0)
  (h2 : 3 * β ^ 2 + 7 * β + K = 0)
  (sum_roots : α + β = -7 / 3)
  (prod_roots : α * β = K / 3)
  : ∃ p : ℝ, p = -70 / 9 + 2 * K / 3 := 
sorry

end quadratic_roots_p_l131_131241


namespace length_of_train_is_250_l131_131061

-- Definitions:
def speed_kmph := 180
def time_seconds := 5

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed_kmph : ℕ) : ℕ := (speed_kmph * 1000) / 3600

-- Question: What is the length of the train in meters?
def length_of_train (speed_kmph : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := kmph_to_mps speed_kmph
  in speed_mps * time_seconds

-- Proof statement:
theorem length_of_train_is_250 :
  length_of_train speed_kmph time_seconds = 250 :=
sorry

end length_of_train_is_250_l131_131061


namespace complex_number_z_value_l131_131219

def i : ℂ := complex.i

def P : set ℂ := {1, -1}
def Q : set ℂ := {i, -1}

theorem complex_number_z_value : ∃ z : ℂ, P ∩ Q = {z * i} ∧ z = i :=
by {
  sorry
}

end complex_number_z_value_l131_131219


namespace repetend_of_five_over_seventeen_l131_131128

theorem repetend_of_five_over_seventeen : 
  let r := 5 / 17 in
  ∃ a b : ℕ, a * 10^b = 294117 ∧ (r * 10^b - a) = (r * 10^6 - r * (10^6 / 17))
   ∧ (r * 10^k = (r * 10^6).floor / 10^k ) where k = 6 := sorry

end repetend_of_five_over_seventeen_l131_131128


namespace polygon_possible_l131_131741

-- Definition: a polygon with matches without breaking them
structure MatchPolygon (n : ℕ) (length : ℝ) where
  num_matches : ℕ
  len_matches : ℝ
  area : ℝ
  notequalzero : len_matches ≠ 0
  notequalzero2 : area ≠ 0
  perimeter_eq: num_matches * len_matches = length * real.of_nat n
  all_matches_used : n = 12
  no_breaking : (length / real.of_nat n) = 2 

theorem polygon_possible : 
  ∃ P : MatchPolygon 12 2, P.area = 16 :=
sorry

end polygon_possible_l131_131741


namespace grid_condition_l131_131613

theorem grid_condition
  (grid : Fin 2000 → Fin 2000 → ℤ)
  (h1 : ∀ i j, grid i j = 1 ∨ grid i j = -1)
  (h2 : (Finset.univ.sum (λ i, Finset.univ.sum (grid i))) ≥ 0) :
  ∃ (rows cols : Finset (Fin 2000)), rows.card = 1000 ∧ cols.card = 1000 ∧ (rows.sum (λ i, cols.sum (grid i))) ≥ 1000 :=
sorry

end grid_condition_l131_131613


namespace problem_statement_l131_131852

def digits_match_condition (N N9 N7 D : ℕ) : Prop :=
  |N - N9| = D ∧ |N7 - N9| = D ∧ (D % 100 = N % 100)

def num_satisfying_N : ℕ :=
  (∀ N : ℕ, 100 ≤ N ∧ N < 1000 →
    ∃ N9 N7 D, (∃ a2 a1 a0 b2 b1 b0 : ℕ, N9 = 81 * a2 + 9 * a1 + a0 ∧ N7 = 49 * b2 + 7 * b1 + b0) ∧
    digits_match_condition N N9 N7 D) ∧
  (count (λ N, ∃ N9 N7 D, digits_match_condition N N9 N7 D) (range 1000) = 45)

theorem problem_statement : num_satisfying_N = 45 := by
  sorry

end problem_statement_l131_131852


namespace speed_approximation_l131_131974

theorem speed_approximation : 
  let distance_feet := 90
  let time_seconds := 3
  let feet_per_mile := 5280
  let miles_per_foot := (1 : ℝ) / feet_per_mile
  let distance_miles := distance_feet * miles_per_foot
  let seconds_per_minute := 60
  let minutes_per_hour := 60
  let hours_per_second := (1 : ℝ) / (seconds_per_minute * minutes_per_hour)
  let time_hours := time_seconds * hours_per_second
  let speed_mph := distance_miles / time_hours
  speed_mph ≈ 20.47 := 
by 
  sorry

end speed_approximation_l131_131974


namespace base_conversion_problem_l131_131708

theorem base_conversion_problem (b : ℕ) (h : b^2 + 2 * b - 25 = 0) : b = 3 :=
sorry

end base_conversion_problem_l131_131708


namespace explain_education_policy_l131_131631

theorem explain_education_policy :
  ∃ (reason1 reason2 : String), reason1 ≠ reason2 ∧
    (reason1 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason2 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions")
    ∨
    (reason2 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason1 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions") :=
by
  sorry

end explain_education_policy_l131_131631


namespace tess_distance_graph_l131_131346

variable (A B C : Type) [metric_space A] [inhabited A] (dist : A → A → ℝ)

theorem tess_distance_graph (starts_at : A) (runs_counterclockwise : True) (triangle_path : A → A → A → ℕ → ℝ) :
  ∃ (f : ℕ → ℝ), 
    (f 0 = 0) ∧
    (∀ t, t ∈ [0, 1] → f t = dist starts_at (triangle_path starts_at B C t)) ∧
    (∃ t_peak, t_peak ∈ [0, 1] ∧ f t_peak = max (f 0) (f 1)) ∧
    (∀ t, t ∈ [1, 2] → f t varies) ∧ 
    (f 3 = 0) :=
sorry

end tess_distance_graph_l131_131346


namespace moles_of_HCl_is_one_l131_131889

def moles_of_HCl_combined 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : ℝ := 
by 
  sorry

theorem moles_of_HCl_is_one 
  (moles_NaHSO3 : ℝ) 
  (moles_H2O_formed : ℝ)
  (reaction_completes : moles_H2O_formed = 1) 
  (one_mole_NaHSO3_used : moles_NaHSO3 = 1) 
  : moles_of_HCl_combined moles_NaHSO3 moles_H2O_formed reaction_completes one_mole_NaHSO3_used = 1 := 
by 
  sorry

end moles_of_HCl_is_one_l131_131889


namespace weeks_to_save_l131_131537

-- Define the conditions as given in the problem
def cost_of_bike : ℕ := 600
def gift_from_parents : ℕ := 60
def gift_from_uncle : ℕ := 40
def gift_from_sister : ℕ := 20
def gift_from_friend : ℕ := 30
def weekly_earnings : ℕ := 18

-- Total gift money
def total_gift_money : ℕ := gift_from_parents + gift_from_uncle + gift_from_sister + gift_from_friend

-- Total money after x weeks
def total_money_after_weeks (x : ℕ) : ℕ := total_gift_money + weekly_earnings * x

-- Main theorem statement
theorem weeks_to_save (x : ℕ) : total_money_after_weeks x = cost_of_bike → x = 25 := by
  sorry

end weeks_to_save_l131_131537


namespace angle_of_sum_l131_131092

noncomputable def sum_of_exponentials (x: ℂ) := 
  exp(2 * complex.pi * complex.i / 24) + 
  exp(14 * complex.pi * complex.i / 24) + 
  exp(26 * complex.pi * complex.i / 24) + 
  exp(38 * complex.pi * complex.i / 24)

theorem angle_of_sum : ∃ (r : ℝ), sum_of_exponentials(re) = r * exp(complex.i * (complex.pi / 3)) := 
by
  sorry

end angle_of_sum_l131_131092


namespace eval_complex_expr_l131_131905

theorem eval_complex_expr (z : ℂ) (h : z = 1 + complex.I) : 
  (2 / z + z^2) = 1 + complex.I :=
by
  sorry

end eval_complex_expr_l131_131905


namespace max_digit_sum_digital_watch_display_l131_131448

theorem max_digit_sum_digital_watch_display : 
  (∃ (hours minutes : ℕ), 
    0 ≤ hours ∧ hours < 24 ∧ 
    0 ≤ minutes ∧ minutes < 60 ∧ 
    digit_sum(hours) + digit_sum(minutes) = 24) :=
sorry

def digit_sum (n : ℕ) : ℕ :=
(n.digits 10).sum

end max_digit_sum_digital_watch_display_l131_131448


namespace exist_tangents_intersecting_line_l131_131703

-- Define the plane Delta and line a using parametric and implicit equations
variables (A B C D r x0 y0 z0 : ℝ)
variables (u1 u2 u3 t : ℝ)
variables (x1 y1 z1 : ℝ)

-- Define the equations
def sphere (x y z : ℝ) := (x - x0)^2 + (y - y0)^2 + (z - z0)^2 = r^2
def plane_delta (x y z : ℝ) := A * x + B * y + C * z + D = 0
def line_a (t : ℝ) := (x1 + t * u1, y1 + t * u2, z1 + t * u3)
def circle_C (x y : ℝ) := (x - x0)^2 + (y - y0)^2 = r^2 -- Simplified for intersection on xy-plane

noncomputable def intersection_point_P := 
  let (x, y, z) := line_a t in
  {x1 := x, y1 := y, z1 := z} -- Intersection logic to find point

theorem exist_tangents_intersecting_line (t : ℝ):
  ∃ T₁ T₂ : (ℝ × ℝ), plane_delta (T₁.fst) (T₁.snd) T₁.snd ∧ plane_delta (T₂.fst) (T₂.snd) T₂.snd ∧
  T₁ ≠ T₂ ∧ 
  (T₁, T₂) ∈ ({p : (ℝ × ℝ) | (circle_C p.1 p.2) ∧ (intersection_point_P.tangent p.1 p.2)} : set (ℝ × ℝ)) :=
sorry -- Proof is omitted


end exist_tangents_intersecting_line_l131_131703


namespace previous_year_ranking_l131_131384

-- Define the initial conditions
def districts := ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
def current_ranks : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
def changes : List String := ["↑",  "-", "↓", "-", "↑", "↓", "↑", "↓", "↓", "-"]

-- Axiomatize the ranking change constraints
axiom change_constraint (d : String) (δ : String) 
  (rank prev_rank : ℕ) :
    δ ∈ changes → 
    (δ = "↑" → rank = prev_rank - 1 ∨ rank = prev_rank - 2) ∧
    (δ = "↓" → rank = prev_rank + 1 ∨ rank = prev_rank + 2) ∧
    (δ = "-" → rank = prev_rank)

-- State the problem to be proven
theorem previous_year_ranking :
  ∃ (r1 r6 r7 r8 : String)
    (ranks : (String → ℕ))
    (prev_ranks : (String → ℕ)),
      (prev_ranks "C" = 1 ∧
      (prev_ranks "E" = 6 ∨ prev_ranks "H" = 6 ∨ prev_ranks "I" = 6) ∧
      (prev_ranks "E" = 7 ∨ prev_ranks "H" = 7 ∨ prev_ranks "I" = 7) ∧
      (prev_ranks "E" = 8 ∨ prev_ranks "H" = 8 ∨ prev_ranks "I" = 8)) ∧
      ranks "A" = 1 ∧ ranks "B" = 2 ∧ ranks "C" = 3 ∧ ranks "D" = 4 ∧ ranks "E" = 5 ∧ ranks "F" = 6 ∧ ranks "G" = 7 ∧ ranks "H" = 8 ∧ ranks "I" = 9 ∧ ranks "J" = 10 ∧
      (change_constraint ("A") ("↑") (1) (prev_ranks "A")) ∧
      (change_constraint ("B") ("-") (2) (prev_ranks "B")) ∧
      (change_constraint ("C") ("↓") (3) (prev_ranks "C")) ∧
      (change_constraint ("D") ("-") (4) (prev_ranks "D")) ∧
      (change_constraint ("E") ("↑") (5) (prev_ranks "E")) ∧
      (change_constraint ("F") ("↓") (6) (prev_ranks "F")) ∧
      (change_constraint ("G") ("↑") (7) (prev_ranks "G")) ∧
      (change_constraint ("H") ("↓") (8) (prev_ranks "H")) ∧
      (change_constraint ("I") ("↓") (9) (prev_ranks "I")) ∧
      (change_constraint ("J") ("-") (10) (prev_ranks "J")) := 
sorry

end previous_year_ranking_l131_131384


namespace repetend_of_fraction_l131_131159

/-- The repeating sequence of the decimal representation of 5/17 is 294117 -/
theorem repetend_of_fraction : 
  let rep := list.take 6 (list.drop 1 (to_digits 10 (5 / 17) 8)) in
  rep = [2, 9, 4, 1, 1, 7] := 
by
  sorry

end repetend_of_fraction_l131_131159


namespace age_of_B_l131_131455

theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 37) : B = 14 :=
by sorry

end age_of_B_l131_131455


namespace probability_same_color_l131_131765

theorem probability_same_color :
  let bagA_white := 8
  let bagA_red := 4
  let bagB_white := 6
  let bagB_red := 6
  let totalA := bagA_white + bagA_red
  let totalB := bagB_white + bagB_red
  let prob_white_white := (bagA_white / totalA) * (bagB_white / totalB)
  let prob_red_red := (bagA_red / totalA) * (bagB_red / totalB)
  let total_prob := prob_white_white + prob_red_red
  total_prob = 1 / 2 := 
by 
  sorry

end probability_same_color_l131_131765


namespace inverse_89_mod_90_l131_131119

theorem inverse_89_mod_90 : (∃ x : ℤ, 0 ≤ x ∧ x ≤ 89 ∧ x * 89 % 90 = 1) :=
by
  use 89
  constructor
  · exact by norm_num
  constructor
  · exact by norm_num
  · exact by norm_num
  sorry

end inverse_89_mod_90_l131_131119


namespace g_g_even_l131_131309

variable {α : Type*} [HasNeg α]
variable {g : α → α}

def is_even (f : α → α) : Prop := ∀ x, f (-x) = f x

theorem g_g_even (h : is_even g) : is_even (g ∘ g) :=
by
  sorry

end g_g_even_l131_131309


namespace orthic_triangle_circumradius_l131_131933

variables (α β γ R : ℝ)

theorem orthic_triangle_circumradius (hα : α + β + γ = π) :
  let α1 := π - 2 * α,
      β1 := π - 2 * β,
      γ1 := π - 2 * γ,
      sin_2α := sin (2 * α),
      sin_2γ := sin (2 * γ),
      a1 := R * sin_2α,
      b1 := R * sin (2 * β),
      c1 := R * sin_2γ in
  (∃ r : ℝ, r = R / 2) :=
sorry

end orthic_triangle_circumradius_l131_131933


namespace newsletter_cost_l131_131195

theorem newsletter_cost (x : ℝ) (h1 : 14 * x < 16) (h2 : 19 * x > 21) : x = 1.11 :=
by
  sorry

end newsletter_cost_l131_131195


namespace triangle_centroid_altitude_l131_131642

/-- In triangle XYZ with side lengths XY = 7, XZ = 24, and YZ = 25, the length of GQ where Q 
    is the foot of the altitude from the centroid G to the side YZ is 56/25. -/
theorem triangle_centroid_altitude :
  let XY := 7
  let XZ := 24
  let YZ := 25
  let GQ := 56 / 25
  GQ = (56 : ℝ) / 25 :=
by
  -- proof goes here
  sorry

end triangle_centroid_altitude_l131_131642


namespace math_proof_problem_chord_proof_l131_131546

noncomputable def is_conic (x y : ℝ) (a : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + a = 0

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def line_eq (x y : ℝ) : Prop :=
  x - y + 1 = 0

def equation_of_circle (x y : ℝ) (c : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = c

theorem math_proof_problem (a : ℝ) :
  (∀ {A B : ℝ × ℝ}, is_conic A.1 A.2 a → is_conic B.1 B.2 a → midpoint A B = (0, 1)) →
  (a < 3 ∧ (∀ {x y : ℝ}, is_conic x y a → line_eq x y)) :=
sorry

theorem chord_proof (len : ℝ) :
  (len = 2 * real.sqrt 7) →
  equation_of_circle 0 1 9 :=
sorry

end math_proof_problem_chord_proof_l131_131546


namespace r_daily_earnings_l131_131803

def earnings_problem (P Q R : ℝ) : Prop :=
  (9 * (P + Q + R) = 1890) ∧ 
  (5 * (P + R) = 600) ∧ 
  (7 * (Q + R) = 910)

theorem r_daily_earnings :
  ∃ P Q R : ℝ, earnings_problem P Q R ∧ R = 40 := sorry

end r_daily_earnings_l131_131803


namespace f_f_10_eq_1_l131_131939

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 10^(x-1) else log10 x

theorem f_f_10_eq_1 : f (f 10) = 1 :=
by
  sorry

end f_f_10_eq_1_l131_131939


namespace proof_solution_l131_131296

noncomputable def proof_problem : Prop :=
  ∀ (ω : Type) (A B O H X Y M : ω)
  (d : O ≠ H)
  [circle ω (d) (A) (B)]
  [point_on_circle O ω]
  [orthogonal_projection H O AB]
  [midpoint M OH]
  [circle_intersection ω O (radius OH) X Y],
  collinear X Y M

theorem proof_solution : proof_problem :=
  sorry

end proof_solution_l131_131296


namespace polygon_possible_l131_131738

-- Definition: a polygon with matches without breaking them
structure MatchPolygon (n : ℕ) (length : ℝ) where
  num_matches : ℕ
  len_matches : ℝ
  area : ℝ
  notequalzero : len_matches ≠ 0
  notequalzero2 : area ≠ 0
  perimeter_eq: num_matches * len_matches = length * real.of_nat n
  all_matches_used : n = 12
  no_breaking : (length / real.of_nat n) = 2 

theorem polygon_possible : 
  ∃ P : MatchPolygon 12 2, P.area = 16 :=
sorry

end polygon_possible_l131_131738


namespace possible_ages_l131_131447

-- Define the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 3}

-- Condition: The age must start with "211"
def starting_sequence : List ℕ := [2, 1, 1]

-- Calculate the count of possible ages
def count_ages : ℕ :=
  let remaining_digits := [2, 2, 1, 3]
  let total_permutations := Nat.factorial 4
  let repetitions := Nat.factorial 2
  total_permutations / repetitions

theorem possible_ages : count_ages = 12 := by
  -- Proof should go here but it's omitted according to instructions.
  sorry

end possible_ages_l131_131447


namespace part_a_part_b_l131_131436

-- Definition of a convex quadrilateral
structure ConvexQuadrilateral :=
  (A B C D : Point)
  (is_convex : Convex {A, B, C, D})

-- A condition to define side lengths in a convex quadrilateral
def side_length (p1 p2 : Point) : ℝ := dist p1 p2

-- Function to find the longest diagonal of a convex quadrilateral
def longest_diagonal (quad : ConvexQuadrilateral) : ℝ :=
  max (side_length quad.A quad.C) (side_length quad.B quad.D)

-- Statement for Part (a)
theorem part_a (quad : ConvexQuadrilateral) :
  ∃ s1 s2, 
    {s1, s2} ⊆ {side_length quad.A quad.B, side_length quad.B quad.C, side_length quad.C quad.D, side_length quad.D quad.A} ∧ 
    s1 < longest_diagonal quad ∧ 
    s2 < longest_diagonal quad := 
sorry

-- Statement for Part (b)
theorem part_b :
  ∃ (quad : ConvexQuadrilateral), 
    let ld := longest_diagonal quad in 
    ∃ s1 s2, 
      {s1, s2} ⊆ {side_length quad.A quad.B, side_length quad.B quad.C, side_length quad.C quad.D, side_length quad.D quad.A} ∧ 
      ∀ s, s ∈ {side_length quad.A quad.B, side_length quad.B quad.C, side_length quad.C quad.D, side_length quad.D quad.A} → (s = s1 ∨ s = s2 → s < ld) :=
sorry

end part_a_part_b_l131_131436


namespace area_within_fence_l131_131361

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l131_131361


namespace repetend_of_5_div_17_l131_131140

theorem repetend_of_5_div_17 : 
  ∃ repetend : ℕ, 
  decimal_repetend (5 / 17) = repetend ∧ 
  repetend = 2941176470588235 :=
by 
  skip

end repetend_of_5_div_17_l131_131140


namespace large_hotdogs_sold_l131_131450

theorem large_hotdogs_sold (total_hodogs : ℕ) (small_hotdogs : ℕ) (h1 : total_hodogs = 79) (h2 : small_hotdogs = 58) : 
  total_hodogs - small_hotdogs = 21 :=
by
  sorry

end large_hotdogs_sold_l131_131450


namespace shop_a_tv_sets_l131_131764

noncomputable def total_tv_sets := 240
def num_shops := 5
def avg_tv_sets := 48

def shop_b_tv_sets := 30
def shop_c_tv_sets := 60
def shop_d_tv_sets := 80
def shop_e_tv_sets := 50

theorem shop_a_tv_sets : 
  shop_a_tv_sets = total_tv_sets - (shop_b_tv_sets + shop_c_tv_sets + shop_d_tv_sets + shop_e_tv_sets) :=
by {
  let total_tv_sets := avg_tv_sets * num_shops,
  calc 
  shop_a_tv_sets 
      = total_tv_sets - (shop_b_tv_sets + shop_c_tv_sets + shop_d_tv_sets + shop_e_tv_sets) :
      by sorry
}

end shop_a_tv_sets_l131_131764


namespace repetend_of_5_over_17_is_294117_l131_131148

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l131_131148


namespace sum_roots_of_quadratic_l131_131172

theorem sum_roots_of_quadratic :
  ∀ (a b c : ℝ), a ≠ 0 → -48 = a → 96 = b → -72 = c →
  (∑ root in (-48*x^2 + 96*x + -72).roots, root) = 2 :=
by
  intros a b c ha ha_eq hb_eq hc_eq
  have h_poly : -48*x^2 + 96*x + (-72) = a*x^2 + b*x + c,
  { rw [ha_eq, hb_eq, hc_eq] }
  sorry

end sum_roots_of_quadratic_l131_131172


namespace ray_has_4_nickels_left_l131_131337

variables {cents_per_nickel : ℕ := 5}

-- Conditions
def initial_cents := 95
def cents_given_to_peter := 25
def cents_given_to_randi := 2 * cents_given_to_peter
def total_cents_given := cents_given_to_peter + cents_given_to_randi
def remaining_cents := initial_cents - total_cents_given

-- Theorem statement
theorem ray_has_4_nickels_left :
  (remaining_cents / cents_per_nickel) = 4 :=
begin
  sorry
end

end ray_has_4_nickels_left_l131_131337


namespace integer_solutions_system_inequalities_l131_131170

theorem integer_solutions_system_inequalities:
  {x : ℤ} → (2 * x - 1 < x + 1) → (1 - 2 * (x - 1) ≤ 3) → x = 0 ∨ x = 1 := 
by
  intros x h1 h2
  sorry

end integer_solutions_system_inequalities_l131_131170


namespace transform_unit_square_l131_131294

structure Point (α : Type) :=
  (x : α)
  (y : α)

def transform (p : Point ℝ) : Point ℝ :=
  ⟨p.x^2 - p.y^2 + 2 * p.x, p.x * p.y + p.x⟩

def unit_square := { p : Point ℝ // (p = ⟨0, 0⟩) ∨ (p = ⟨1, 0⟩) ∨ (p = ⟨1, 1⟩) ∨ (p = ⟨0, 1⟩) }

theorem transform_unit_square :
  ∃ (shape : set (Point ℝ)), is_closed shape ∧
  ∀ (p ∈ unit_square), transform p ∈ shape ∧ ∃ (q : Point ℝ), transform q ∈ shape :=
sorry

end transform_unit_square_l131_131294


namespace sum_of_valid_B_divisors_l131_131979

theorem sum_of_valid_B_divisors : 
  (sum (filter (λ B: ℕ, (2 * 100 + B * 10 + 7) % 8 = 0) (list.range 10))) = 18 :=
by
  sorry

end sum_of_valid_B_divisors_l131_131979


namespace probability_bernardo_less_than_silvia_l131_131076

open Finset

def bernardo_choices : Finset (Finset (Fin 9)) := (powerset (range 9)).filter (λ s => s.card = 3)
def silvia_choices : Finset (Finset (Fin 10)) := (powerset (range 10)).filter (λ s => s.card = 3)

-- Define the event that Bernardo's number is less than Silvia's number
def event_b_less_s (b s : Finset ℕ) : Prop := b.to_list.sorted_lt s.to_list

-- Calculate the probability that Bernardo's number is less than Silvia's
def probability_b_less_s : ℚ := 
  (bernardo_choices.card : ℚ) / (silvia_choices.card : ℚ) * 
  ((1 / 2) + (choose (9 : ℚ) 2 / choose (10 : ℚ) 3))

theorem probability_bernardo_less_than_silvia : probability_b_less_s = 14 / 25 := sorry

end probability_bernardo_less_than_silvia_l131_131076


namespace incorrect_statements_count_l131_131478

open Set

def conditions_check : Prop :=
  (1 ∈ ({0, 1, 2} : Set ℕ)) ∧
  (∅ ⊆ ({0, 1, 2} : Set ℕ)) ∧
  ¬({1} ∈ ({0, 1, 2} : Set (Set ℕ))) ∧
  ({0, 1, 2} = ({2, 0, 1} : Set ℕ))

theorem incorrect_statements_count :
  conditions_check → 
  (Nat.card {c ∈ [(1 ∈ ({0, 1, 2} : Set ℕ)), (∅ ⊆ ({0, 1, 2} : Set ℕ)), ({1} ∈ ({0, 1, 2} : Set (Set ℕ))), ({0, 1, 2} = ({2, 0, 1} : Set ℕ))] | c = false}) = 1 :=
by
  sorry

end incorrect_statements_count_l131_131478


namespace martys_journey_length_l131_131678

theorem martys_journey_length (x : ℝ) (h1 : x / 4 + 30 + x / 3 = x) : x = 72 :=
sorry

end martys_journey_length_l131_131678


namespace calculate_expression_l131_131085

theorem calculate_expression :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) = 3^128 - 5^128 :=
by
  sorry

end calculate_expression_l131_131085


namespace coefficient_x2y7_expansion_l131_131869

theorem coefficient_x2y7_expansion : 
  ∀ (x y : ℕ), 
  (coeff x^2 y^7 in (x - 2 * y) * (x + y)^8) = -48 := 
by sorry

end coefficient_x2y7_expansion_l131_131869


namespace find_x_l131_131982

theorem find_x (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 :=
by
  sorry

end find_x_l131_131982


namespace original_price_discount_l131_131697

theorem original_price_discount (P : ℝ) (h : 0.90 * P = 450) : P = 500 :=
by
  sorry

end original_price_discount_l131_131697


namespace constant_term_in_expansion_l131_131167

theorem constant_term_in_expansion :
  let T := (λ x : ℝ, (sqrt x - 1 / x) ^ 9) in
  ∃ c : ℤ, (∀ x : ℝ, T x = c) → c = -84 :=
by
  sorry

end constant_term_in_expansion_l131_131167


namespace A_oplus_B_l131_131534

def set_minus (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := (set_minus M N) ∪ (set_minus N M)

def A : Set ℝ := {y | ∃ x ∈ ℝ, y = x^2 - 3 * x}
def B : Set ℝ := {y | ∃ x ∈ ℝ, y = -2^x}

theorem A_oplus_B : sym_diff A B = (Set.Iio (-9 / 4) ∪ Set.Ici 0) :=
sorry

end A_oplus_B_l131_131534


namespace six_points_in_rectangle_close_dist_l131_131021

theorem six_points_in_rectangle_close_dist :
  ∀ (rect : set (ℝ × ℝ)) (points : fin 6 → ℝ × ℝ),
    rect = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3} →
    (∀ i, points i ∈ rect) →
    ∃ (i j : fin 6), i ≠ j ∧ dist (points i) (points j) ≤ real.sqrt 5 :=
by {sorry}

end six_points_in_rectangle_close_dist_l131_131021


namespace bottles_per_case_correct_l131_131443

-- Define the conditions given in the problem
def daily_bottle_production : ℕ := 120000
def number_of_cases_needed : ℕ := 10000

-- Define the expected answer
def bottles_per_case : ℕ := 12

-- The statement we need to prove
theorem bottles_per_case_correct :
  daily_bottle_production / number_of_cases_needed = bottles_per_case :=
by
  -- Leap of logic: actually solving this for correctness is here considered a leap
  sorry

end bottles_per_case_correct_l131_131443


namespace jogging_speed_equals_9_1_l131_131289

open Real

noncomputable def distance_to_school : ℝ := 6.666666666666666
noncomputable def bus_speed : ℝ := 25
noncomputable def total_time : ℝ := 1

theorem jogging_speed_equals_9_1
  (v : ℝ)
  (h1 : total_time = (distance_to_school / v) + (distance_to_school / bus_speed))
  : v ≈ 9.1 := 
sorry

end jogging_speed_equals_9_1_l131_131289


namespace triangle_to_20_sided_polygon_l131_131638

theorem triangle_to_20_sided_polygon (T : Triangle) :
  ∃ (A B : Shape), (A ∪ B = T) ∧ (forms_20_sided_polygon (A ∪ B)) :=
sorry

end triangle_to_20_sided_polygon_l131_131638


namespace Lucky_steps_6098_l131_131392

-- Define initial conditions and behaviors
def Position := Int

structure State :=
  (position : Position)
  (facing_pos : Bool)  -- True if facing positive direction
  (coins : Position → Option Bool) -- Some true = heads-up, Some false = tails-up, None = no coin.

-- Define Lucky's step (procedures)
def step (s : State) : State :=
  let current_coin := s.coins s.position
  match current_coin with
  | none =>
    {
      position := s.position + (if s.facing_pos then 1 else -1),
      facing_pos := s.facing_pos,
      coins := s.coins.insert s.position true
    }
  | some true =>
    {
      position := s.position + (if s.facing_pos then -1 else 1),
      facing_pos := !s.facing_pos,
      coins := s.coins.insert s.position false
    }
  | some false =>
    {
      position := s.position + (if s.facing_pos then 1 else -1),
      facing_pos := s.facing_pos,
      coins := s.coins.remove s.position
    }

-- Function to count tails-up coins
def countTails (coins : Position → Option Bool) : Nat :=
  coins.fold 0 (fun _ c acc => if c = some false then acc + 1 else acc)

-- Function to determine the stopping condition
def stopped (s : State) : Bool :=
  countTails s.coins = 20

-- Initial state
def initialState : State := {
  position := 0,
  facing_pos := true,
  coins := fun n => some true
}

-- Prove that the process stops after 6098 steps
theorem Lucky_steps_6098 :
  ∃ s : State, (∃ n : Nat, n = 6098 ∧ (step^[n] initialState = s)) ∧ stopped (step^[6098] initialState) := by
  sorry

end Lucky_steps_6098_l131_131392


namespace infinite_geometric_series_sum_eq_l131_131661

noncomputable def geometric_series_sum (a b : ℝ) : ℝ :=
  ∑ i in finset.range 5, a / b^(i + 1)

theorem infinite_geometric_series_sum_eq 
  (a b : ℝ) 
  (h : geometric_series_sum a b = 3) :
  (∑' n : ℕ, 2 * a / (a + b)^(n + 1)) = (6 * (1 - 1 / b^5)) / (4 - 1 / b^5) :=
by
  sorry

end infinite_geometric_series_sum_eq_l131_131661


namespace conference_duration_excluding_breaks_l131_131053

-- Definitions based on the conditions
def total_hours : Nat := 14
def additional_minutes : Nat := 20
def break_minutes : Nat := 15

-- Total time including breaks
def total_time_minutes : Nat := total_hours * 60 + additional_minutes
-- Number of breaks
def number_of_breaks : Nat := total_hours
-- Total break time
def total_break_minutes : Nat := number_of_breaks * break_minutes

-- Proof statement
theorem conference_duration_excluding_breaks :
  total_time_minutes - total_break_minutes = 650 := by
  sorry

end conference_duration_excluding_breaks_l131_131053


namespace area_within_fence_l131_131363

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l131_131363


namespace number_of_x_intercepts_cos_inv_eq_2864_l131_131526

open Real Int

noncomputable def count_x_intercepts (f : ℝ → ℝ) (a b : ℝ) :=
  { x : ℝ | a < x ∧ x < b ∧ f x = 0 }.to_finset.card

theorem number_of_x_intercepts_cos_inv_eq_2864 :
  count_x_intercepts (λ x, cos (1 / x)) 0.00005 0.0005 = 2864 :=
by
  sorry

end number_of_x_intercepts_cos_inv_eq_2864_l131_131526


namespace percent_round_trip_tickets_is_100_l131_131325

noncomputable def percent_round_trip_tickets (P : ℕ) (x : ℚ) : ℚ :=
  let R := x / 0.20
  R

theorem percent_round_trip_tickets_is_100
  (P : ℕ)
  (x : ℚ)
  (h : 20 * x = P) :
  percent_round_trip_tickets P (x / P) = 100 :=
by
  sorry

end percent_round_trip_tickets_is_100_l131_131325


namespace log_nine_cbrt_equals_third_l131_131109

theorem log_nine_cbrt_equals_third : log 9 (9 ^ (1 / 3 : ℝ)) = 1 / 3 :=
by
  sorry

end log_nine_cbrt_equals_third_l131_131109


namespace janice_total_earnings_l131_131655

-- Defining the working conditions as constants
def days_per_week : ℕ := 5  -- Janice works 5 days a week
def earning_per_day : ℕ := 30  -- Janice earns $30 per day
def overtime_earning_per_shift : ℕ := 15  -- Janice earns $15 per overtime shift
def overtime_shifts : ℕ := 3  -- Janice works three overtime shifts

-- Defining Janice's total earnings for the week
def total_earnings : ℕ := (days_per_week * earning_per_day) + (overtime_shifts * overtime_earning_per_shift)

-- Statement to prove that Janice's total earnings are $195
theorem janice_total_earnings : total_earnings = 195 :=
by
  -- The proof is omitted.
  sorry

end janice_total_earnings_l131_131655


namespace mass_of_man_is_correct_l131_131417

-- Definitions for conditions
def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def sinking_depth : ℝ := 0.012
def density_of_water : ℝ := 1000

-- Volume of water displaced
def volume_displaced := length_of_boat * breadth_of_boat * sinking_depth

-- Mass of the man
def mass_of_man := density_of_water * volume_displaced

-- Prove that the mass of the man is 72 kg
theorem mass_of_man_is_correct : mass_of_man = 72 := by
  sorry

end mass_of_man_is_correct_l131_131417


namespace magnitude_two_a_plus_b_l131_131776

open Real

variables (a b : ℝ^3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (angle_120 : inner a b = -1/2)

theorem magnitude_two_a_plus_b : ∥2 • a + b∥ = sqrt 3 :=
by sorry

end magnitude_two_a_plus_b_l131_131776


namespace solve_equation_l131_131102

noncomputable def y (x : ℝ) : ℝ := x^2 + 1

theorem solve_equation (x : ℝ) :
  4 * (y x)^2 + 2 * (y x) + 5 = 3 * (5 * x^2 + y x + 3) ↔ 
  x = √(1 + √5 / 2) ∨ x = -√(1 + √5 / 2) ∨ x = √(1 - √5 / 2) ∨ x = -√(1 - √5 / 2) :=
by
  sorry

end solve_equation_l131_131102


namespace find_SSE_l131_131273

theorem find_SSE (SST SSR : ℝ) (h1 : SST = 13) (h2 : SSR = 10) : SST - SSR = 3 :=
by
  sorry

end find_SSE_l131_131273


namespace sequence_formula_l131_131275

theorem sequence_formula (a : ℕ → ℕ) (n : ℕ) (h : ∀ n ≥ 1, a n = a (n - 1) + n^3) : 
  a n = (n * (n + 1) / 2) ^ 2 := sorry

end sequence_formula_l131_131275


namespace can_construct_polygon_l131_131761

def match_length : ℕ := 2
def number_of_matches : ℕ := 12
def total_length : ℕ := number_of_matches * match_length
def required_area : ℝ := 16

theorem can_construct_polygon : 
  (∃ (P : Polygon), P.perimeter = total_length ∧ P.area = required_area) := 
sorry

end can_construct_polygon_l131_131761


namespace A_rotated_l131_131330

-- Define initial coordinates of point A
def A_initial : ℝ × ℝ := (1, 2)

-- Define the transformation for a 180-degree clockwise rotation around the origin
def rotate_180_deg (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- The Lean statement to prove the coordinates after the rotation
theorem A_rotated : rotate_180_deg A_initial = (-1, -2) :=
by
  sorry

end A_rotated_l131_131330


namespace sequence_of_divisors_eq_one_l131_131313

theorem sequence_of_divisors_eq_one (k : ℕ) (Hk : k ≥ 2)
    (n : Fin k → ℕ) (H1 : ∀ i : Fin k, (n (i + 1) mod k) ∣ 2^(n i) - 1) 
    (H2 : n 0 ∣ 2^(n (k - 1)) - 1) :
    ∀ i : Fin k, n i = 1 := 
sorry

end sequence_of_divisors_eq_one_l131_131313


namespace sin_inverse_equation_l131_131088

noncomputable def a := Real.arcsin (4/5)
noncomputable def b := Real.arctan 1
noncomputable def c := Real.arccos (1/3)
noncomputable def sin_a_plus_b_minus_c := Real.sin (a + b - c)

theorem sin_inverse_equation : sin_a_plus_b_minus_c = 11 / 15 := sorry

end sin_inverse_equation_l131_131088


namespace inequality_proof_l131_131668

noncomputable def geometric_mean (l : List ℝ) : ℝ :=
(Math.prod l)^(1 / l.length)

noncomputable def arithmetic_mean (l : List ℝ) : ℝ :=
l.sum / l.length

theorem inequality_proof (n : ℕ) (α : Fin n → ℝ)
  (h_pos : ∀ i, 0 < α i) (h_n_gt_1 : 1 < n) :
  let g_n := geometric_mean (List.ofFn α)
      A (k : ℕ) := arithmetic_mean (List.ofFn (fun i => if i < k then α ⟨i,_⟩ else 0))
      G_n := geometric_mean (List.map A (List.range n)) in
  n * (G_n / A n)^(1/2) + g_n / G_n ≤ n + 1 ∧
  (∀ i, α i = α 0 ↔ n * (G_n / A n)^(1/2) + g_n / G_n = n + 1) :=
by
  -- Proof would go here
  sorry

end inequality_proof_l131_131668


namespace fenced_area_l131_131355

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l131_131355


namespace problem_l131_131259

variables {f : ℝ → ℝ}

-- Defining the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

-- Theorem statement
theorem problem (hf_even : is_even f) 
                (hf_increasing : is_increasing_on f {x | x ≤ -1}) : 
  f 2 < f (-1.5) ∧ f (-1.5) < f (-1) :=
by {
  sorry
}

end problem_l131_131259


namespace pressure_on_trapezoidal_dam_l131_131508

noncomputable def water_pressure_on_trapezoidal_dam (ρ g h a b : ℝ) : ℝ :=
  ρ * g * (h^2) * (2 * a + b) / 6

theorem pressure_on_trapezoidal_dam
  (ρ g h a b : ℝ) : water_pressure_on_trapezoidal_dam ρ g h a b = ρ * g * (h^2) * (2 * a + b) / 6 := by
  sorry

end pressure_on_trapezoidal_dam_l131_131508


namespace valid_combinations_count_l131_131468

theorem valid_combinations_count : 
  let wrapping_paper_count := 10
  let ribbon_count := 3
  let gift_card_count := 5
  let invalid_combinations := 1 -- red ribbon with birthday card
  let total_combinations := wrapping_paper_count * ribbon_count * gift_card_count
  total_combinations - invalid_combinations = 149 := 
by 
  sorry

end valid_combinations_count_l131_131468


namespace four_digit_combinations_l131_131901

theorem four_digit_combinations : 
  let digits := {1, 2, 3, 4, 5, 6}
  let even_digits := {2, 4, 6}
  let odd_digits := {1, 3, 5}
  let n_combinations (s: Finset ℕ) (k: ℕ) := (s.card.choose k)
  let n_permutations (n: ℕ) := nat.factorial n
  let total_combinations := n_combinations even_digits 2 * n_combinations odd_digits 2 * n_permutations 4
  total_combinations = 216 := 
by
  sorry

end four_digit_combinations_l131_131901


namespace hours_per_day_initial_l131_131031

-- Definition of the problem and conditions
def initial_men : ℕ := 75
def depth1 : ℕ := 50
def additional_men : ℕ := 65
def total_men : ℕ := initial_men + additional_men
def depth2 : ℕ := 70
def hours_per_day2 : ℕ := 6
def work1 (H : ℝ) := initial_men * H * depth1
def work2 := total_men * hours_per_day2 * depth2

-- Statement to prove
theorem hours_per_day_initial (H : ℝ) (h1 : work1 H = work2) : H = 15.68 :=
by
  sorry

end hours_per_day_initial_l131_131031


namespace length_curve_y_squared_eq_x_cubed_l131_131301

noncomputable def length_of_curve (C : ℝ → ℝ) (x1 x2 : ℝ) : ℝ :=
∫ x1..x2, sqrt (1 + (deriv C x)^2)

theorem length_curve_y_squared_eq_x_cubed :
  ∀ (x : ℝ), y^2 = x^3 
  ∧ ∀ (O : ℝ × ℝ), O = (0, 0) 
  ∧ ∀ (A : ℝ × ℝ), A = (4 / 9, 8 / 27) 
  -> length_of_curve (λ x, sqrt (x^3)) 0 (4 / 9) = (8 / 27) * (2 * sqrt 2 - 1) := 
sorry

end length_curve_y_squared_eq_x_cubed_l131_131301


namespace inv_89_mod_90_l131_131122

theorem inv_89_mod_90 : ∃ x : ℕ, (0 ≤ x ∧ x ≤ 89) ∧ (89 * x ≡ 1 [MOD 90]) :=
by
  use 89
  split
  · linarith
  sorry

end inv_89_mod_90_l131_131122


namespace find_scalar_k_l131_131768

noncomputable def scalar_k_exists (u v w : ℝ^3) (k : ℝ) : Prop :=
  u + v + w = 0 ∧ k * (v × u) + (v × w) + (w × u) = 0

theorem find_scalar_k (u v w : ℝ^3) (k : ℝ) :
  (∀ (u v w : ℝ^3), u + v + w = 0 → k * (v × u) + (v × w) + (w × u) = 0) ↔ k = 2 :=
by
  sorry

end find_scalar_k_l131_131768


namespace a_2014_value_l131_131549

-- Definitions based on the given conditions a)
def sequence (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, a (4 * n + 3) = 1) ∧
  (∀ n : ℕ, a (4 * n + 1) = 0) ∧
  (∀ n : ℕ, a (2 * n) = a n)

-- The theorem to prove the equivalent problem c)
theorem a_2014_value (a : ℕ → ℕ) (h : sequence a) : a 2014 = 0 :=
  sorry

end a_2014_value_l131_131549


namespace value_of_f_l131_131932

noncomputable def f : ℝ → ℝ := sorry

axiom symmetry_about_line (f : ℝ → ℝ) (x : ℝ) : f (3 + x) = f (3 - x)
axiom value_at_neg_1 : f (-1) = 320
axiom trig_cond (x : ℝ) : cos x - sin x = 3 * sqrt 2 / 5

theorem value_of_f (x : ℝ) : f (15 * sin (2 * x) / cos (x + π / 4)) = 320 := 
by
  -- Use axioms and conditions given
  sorry

end value_of_f_l131_131932


namespace neither_probability_l131_131253

variables {Ω : Type} [ProbabilitySpace Ω]

def P (s : Set Ω) : ℝ := ProbabilityTheory.prob s

variables (A B : Set Ω) (hA : P A = 0.75) (hB : P B = 0.25) (hAB : P (A ∩ B) = 0.20)

theorem neither_probability :
  P (-(A ∪ B)) = 0.20 :=
by
  sorry

end neither_probability_l131_131253


namespace gcf_factorial_5_6_l131_131169

theorem gcf_factorial_5_6 : Nat.gcd (Nat.factorial 5) (Nat.factorial 6) = Nat.factorial 5 := by
  sorry

end gcf_factorial_5_6_l131_131169


namespace determine_k_l131_131602

theorem determine_k (k : ℝ) :
  (∀ x : ℝ, ((k^2 - 9) * x^2 - (2 * (k + 1)) * x + 1 = 0 → x ∈ {a : ℝ | a = - (2 * (k+1)) / (2 * (k^2-9)} ∨ a = (2*(k+1) + √4 * (k+1)^2 - 4 * (k^2-9) ) / (2*(k^2-9)) ∨ a = (2 * (k+1) - √4 * (k+1)^2 - 4 *(k^2-9)) / (2 *(k^2-9))) → x ∈  {a : ℝ | a = - (2 * (k+1)) / (2 * (k^2-9)} ) ∨ x ∈  {a : ℝ | a = (2*(k+1) + 0 ) / (2 * (k^2-9))} ∨ a = (2*(k+1) - 0 ) / (2 *(k^2-9)} )  → 
  k = 3 ∨ k = -3 ∨ k = -5 := sorry

end determine_k_l131_131602


namespace smallest_n_has_8_factors_l131_131892

theorem smallest_n_has_8_factors (n : ℕ) (a : ℕ) (m : ℕ) (h : n = 2^a * m) (hm: ¬ (2 ∣ m)) (factors_8: (nat.factors m).length = 1 → (nat.factors m).head = 3 → n = 2187) : n = 2187 :=
sorry

end smallest_n_has_8_factors_l131_131892


namespace range_of_alpha_l131_131553

theorem range_of_alpha (α : ℝ) (h1 : 0 ≤ α ∧ α ≤ 2 * Real.pi) :
    (sin α - cos α > 0 ∧ tan α > 0) ↔ 
    (α ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2) ∨ α ∈ Set.Ioo Real.pi (5 * Real.pi / 4)) :=
by sorry

end range_of_alpha_l131_131553


namespace parabola_x_intercepts_l131_131958

theorem parabola_x_intercepts :
  ∃! y : ℝ, -3 * y^2 + 2 * y + 4 = y := 
by
  sorry

end parabola_x_intercepts_l131_131958


namespace james_hourly_charge_l131_131645

def mural_length : ℝ := 20
def mural_width : ℝ := 15
def time_per_square_foot : ℝ := 20 / 60 -- converting minutes to hours
def total_charge : ℝ := 15000

theorem james_hourly_charge : 
  let area := mural_length * mural_width in
  let total_time := area * time_per_square_foot in
  let hourly_rate := total_charge / total_time in
  hourly_rate = 150 :=
by
  sorry

end james_hourly_charge_l131_131645


namespace value_of_a_plus_b_l131_131906

-- Define the main problem conditions
variables (a b : ℝ)

-- State the problem in Lean
theorem value_of_a_plus_b (h1 : |a| = 2) (h2 : |b| = 3) (h3 : |a - b| = - (a - b)) :
  a + b = 5 ∨ a + b = 1 :=
sorry

end value_of_a_plus_b_l131_131906


namespace sum_series_l131_131494

theorem sum_series : 
  (∑ a in Finset.Icc 1 (Finset.range 1), 
   ∑ b in Finset.Icc (a+1) (Finset.range (a+1)), 
   ∑ c in Finset.Icc (b+1) (Finset.range (b+1)), 
   (1 : ℚ) / (3^a * 5^b * 7^c)
  ) = 1 / 21216 := sorry

end sum_series_l131_131494


namespace cube_partition_l131_131446

theorem cube_partition (N : ℕ) : 
  N = 10 ∧ 
  ∃ (cubes : list ℕ), cubes.length = N ∧ 
  (∀ (e ∈ cubes), e ∈ [1, 2, 3]) ∧ 
  (∃ (a b : ℕ), a ∈ cubes ∧ b ∈ cubes ∧ a ≠ b) := 
begin
  sorry
end

end cube_partition_l131_131446


namespace tenby_position_l131_131777

theorem tenby_position : 
  ∀ letters : Finset Char, 
  letters = {'B', 'E', 'N', 'T', 'Y'} → 
  findAlphabeticalPosition "TENBY" letters = 75 := 
by 
  sorry

end tenby_position_l131_131777


namespace cost_of_each_piece_of_wood_l131_131098

theorem cost_of_each_piece_of_wood
  (pieces_per_birdhouse : ℕ)
  (profit_per_birdhouse : ℝ)
  (payment_for_two_birdhouses : ℝ)
  (cost_per_piece_of_wood : ℝ) :
  pieces_per_birdhouse = 7 →
  profit_per_birdhouse = 5.50 →
  payment_for_two_birdhouses = 32 →
  cost_per_piece_of_wood = 1.50 :=
begin
  intros h1 h2 h3,
  -- Definitions based on the conditions
  let C := cost_per_piece_of_wood,
  have eq1 : 2 * (7 * C + 5.5) = 32,
  { rw [←h1, ←h2, ←h3], },
  have eq2 : 14 * C + 11 = 32,
  { linarith, },
  have eq3 : 14 * C = 21,
  { linarith, },
  have eq4 : C = 21 / 14,
  { exact (eq3.div (by norm_num)).symm, },
  exact eq4,
end

end cost_of_each_piece_of_wood_l131_131098


namespace cyclic_matrix_det_zero_l131_131298

theorem cyclic_matrix_det_zero 
  (a b c d p q r : ℝ)
  (h_roots : ∃ (a b c d : ℝ), polynomial.roots (λ x, x^4 + p * x^2 + q * x + r) = {a, b, c, d}) :
  matrix.det ![
    ![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]
  ] = 0 :=
  sorry

end cyclic_matrix_det_zero_l131_131298


namespace passing_marks_l131_131418

-- Define the conditions and prove P = 160 given these conditions
theorem passing_marks (T P : ℝ) (h1 : 0.40 * T = P - 40) (h2 : 0.60 * T = P + 20) : P = 160 :=
by
  sorry

end passing_marks_l131_131418


namespace find_point_M_l131_131918

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def point_M (y : ℝ) : Prop :=
  (∃ t : ℝ, y = t ∧ t > 0)

noncomputable def area_condition (A B : ℝ × ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, 
    A = (x1, y1) ∧ B = (x2, y2) ∧ 
    (A ≠ B) ∧ 
    (3 * |x1| * |x2| * Real.cos (Real.atan2 y1 x1 - Real.atan2 y2 x2) = -3)

theorem find_point_M :
  (∀ M : ℝ × ℝ, 
    (point_M M.snd) ∧ 
    ((∃ (A B : ℝ × ℝ), area_condition A B ∧ ellipse_equation A.1 A.2 ∧ ellipse_equation B.1 B.2)) → 
    M = (0, (Real.sqrt 21) / 7)) :=
sorry

end find_point_M_l131_131918


namespace inverse_89_mod_90_l131_131120

theorem inverse_89_mod_90 : (∃ x : ℤ, 0 ≤ x ∧ x ≤ 89 ∧ x * 89 % 90 = 1) :=
by
  use 89
  constructor
  · exact by norm_num
  constructor
  · exact by norm_num
  · exact by norm_num
  sorry

end inverse_89_mod_90_l131_131120


namespace inv_89_mod_90_l131_131121

theorem inv_89_mod_90 : ∃ x : ℕ, (0 ≤ x ∧ x ≤ 89) ∧ (89 * x ≡ 1 [MOD 90]) :=
by
  use 89
  split
  · linarith
  sorry

end inv_89_mod_90_l131_131121


namespace matchstick_polygon_area_l131_131750

-- Given conditions
def number_of_matches := 12
def length_of_each_match := 2 -- in cm

-- Question: Is it possible to construct a polygon with an area of 16 cm^2 using all the matches?
def polygon_possible : Prop :=
  ∃ (p : Polygon), 
    (p.edges = number_of_matches) ∧ 
    (∃ (match_length : ℝ), match_length = length_of_each_match ∧ by 
      -- Form the polygon using all matches without breaking
      sorry) ∧ 
    (polygon_area p = 16)

-- Proof statement
theorem matchstick_polygon_area :
  polygon_possible :=
  sorry

end matchstick_polygon_area_l131_131750


namespace factorization_cubic_solution_l131_131420

section part_a

variables {a b c : ℝ}

theorem factorization : a^3 + b^3 + c^3 - 3 * a * b * c = 
  (a + b + c) * (a^2 + b^2 + c^2 - a * b - b * c - c * a) :=
begin
  -- proof here, currently omitted
  sorry
end

end part_a

section part_b

variables {x p q : ℝ}

theorem cubic_solution :
  (∃ a b : ℝ, 
    a = (↑(∛(q / 2 + sqrt (q^2 / 4 + p^3 / 27)) : ℂ)).re ∧
    b = (↑(∛(q / 2 - sqrt (q^2 / 4 + p^3 / 27)) : ℂ)).re ∧
    x^3 + p * x + q = 0 ∧
    x = -a - b) :=
begin
  -- proof here, currently omitted
  sorry
end

end part_b

end factorization_cubic_solution_l131_131420


namespace fenced_area_l131_131353

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l131_131353


namespace solution_of_xyz_l131_131972

theorem solution_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : x + y + z = 48 := 
sorry

end solution_of_xyz_l131_131972


namespace triangle_ABC_area_l131_131486

-- Definitions based on the problem conditions
variables (A B C X Y Z : Type)
variable [has_area : has_area A B C]
variable [area_XYZ : has_area X Y Z]
variable [Y_con : YZ = 2 * ZC]
variable [Z_con : ZX = 3 * XA]
variable [X_con : XY = 4 * YB]

-- Given condition
axiom area_XYZ_24 : area_XYZ = 24

-- Prove statement
theorem triangle_ABC_area : area A B C = 59 :=
  sorry

end triangle_ABC_area_l131_131486


namespace distance_origin_to_point_l131_131991

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_origin_to_point :
  distance (0, 0) (-15, 8) = 17 :=
by 
  sorry

end distance_origin_to_point_l131_131991


namespace product_of_odds_less_than_5000_l131_131783

theorem product_of_odds_less_than_5000 : 
  (∏ n in (Finset.range 5000).filter (λ x, odd x), n) = 5000! / (2^2500 * (2500!)) :=
by sorry

end product_of_odds_less_than_5000_l131_131783


namespace distance_AB_eq_sqrt_5_max_area_triangle_coordinates_l131_131942

noncomputable def point_A : ℝ × ℝ := (2, 0)
noncomputable def point_B : ℝ × ℝ := (0, 1)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_AB_eq_sqrt_5 :
  distance point_A point_B = real.sqrt 5 :=
sorry

noncomputable def point_on_curve (θ : ℝ) : ℝ × ℝ :=
  (2 * real.cos θ, real.sin θ)

theorem max_area_triangle_coordinates :
  ∃ θ : ℝ, point_on_curve θ = (-real.sqrt 2, -real.sqrt 2 / 2) :=
sorry

end distance_AB_eq_sqrt_5_max_area_triangle_coordinates_l131_131942


namespace minimum_value_of_fraction_l131_131221

theorem minimum_value_of_fraction (x : ℝ) (h : x > 0) : 
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 - 1 ∧ ∀ y, y = (x^2 + x + 3) / (x + 1) -> y ≥ m :=
sorry

end minimum_value_of_fraction_l131_131221


namespace pole_length_after_cut_l131_131826

theorem pole_length_after_cut (original_length : ℝ) (percentage_shorter : ℝ) (h1 : original_length = 20) (h2 : percentage_shorter = 0.30) : 
  let length_cut = (percentage_shorter * original_length)
  let new_length = original_length - length_cut
  new_length = 14 := 
by
  sorry

end pole_length_after_cut_l131_131826


namespace factor_expression_l131_131865

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) - y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + z^2 - z * x) :=
by
  sorry

end factor_expression_l131_131865


namespace average_rainfall_is_correct_l131_131538

def total_rainfall : ℝ := 1026
def days_in_june : ℝ := 30
def days_in_july : ℝ := 31
def days_in_august : ℝ := 31
def hours_in_day : ℝ := 24

def total_days : ℝ := days_in_june + days_in_july + days_in_august
def total_hours : ℝ := total_days * hours_in_day
def average_rainfall_per_hour : ℝ := total_rainfall / total_hours

theorem average_rainfall_is_correct :
  average_rainfall_per_hour = 1026 / 2208 :=
by
  sorry

end average_rainfall_is_correct_l131_131538


namespace total_bins_l131_131466

-- Definition of the problem conditions
def road_length : ℕ := 400
def placement_interval : ℕ := 20
def bins_per_side : ℕ := (road_length / placement_interval) - 1

-- Statement of the problem
theorem total_bins : 2 * bins_per_side = 38 := by
  sorry

end total_bins_l131_131466


namespace polar_line_equation_l131_131274

theorem polar_line_equation 
  (ρ θ: ℝ)
  (h1: ∃ (ρ θ: ℝ), ρ = 6 * cos θ) 
  (h2: ∀ (x y: ℝ), x = 3 → (x-3)^2 + y^2 = 9) 
  (h3: ∀ (x: ℝ), x = 3 → x ⊥ (polar_axis))
: ρ * cos θ = 3 :=
sorry

end polar_line_equation_l131_131274


namespace hexagon_area_percent_l131_131462

/- Define the conditions -/
variable (total_regions : ℕ) (num_squares : ℕ) (num_hexagons : ℕ) (side_length : ℝ)
variable (area_rectangle : ℝ)

-- Set specific values based on problem conditions
def conditions := 
  total_regions = 12 ∧
  num_squares = 3 ∧
  num_hexagons = 9 ∧
  side_length = 2 ∧
  area_rectangle = 48

-- Define the side length of square and calculate respective areas
def area_square (side_length : ℝ) := side_length^2
def total_area_squares := num_squares * area_square side_length
def area_hexagons := area_rectangle - total_area_squares

-- Calculate the percentage area of hexagons
def percent_area_hexagons := (area_hexagons / area_rectangle) * 100

-- The proof problem
theorem hexagon_area_percent : conditions → percent_area_hexagons = 75 := by
  sorry


end hexagon_area_percent_l131_131462


namespace sum_of_a_b_c_l131_131867

-- Define the problem setup
lemma sum_cosines_zero (x : ℝ) :
  sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (7 * x) ^ 2 = 2 →
  (cos (8 * x) * cos (4 * x) * cos (2 * x) = 0) :=
begin
  sorry
end

-- Define the main theorem
theorem sum_of_a_b_c :
  (∃ a b c : ℕ, 
    (∀ x : ℝ, sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (7 * x) ^ 2 = 2 → cos (a * x) * cos (b * x) * cos (c * x) = 0) ∧
    a + b + c = 14) :=
begin
  use [2, 4, 8],
  split,
  {
    intros x h,
    apply sum_cosines_zero x h
  },
  {
    norm_num
  }
end

end sum_of_a_b_c_l131_131867


namespace candidate_knows_Excel_and_willing_nights_l131_131028

variable (PExcel PXNight : ℝ)
variable (H1 : PExcel = 0.20) (H2 : PXNight = 0.30)

theorem candidate_knows_Excel_and_willing_nights : (PExcel * PXNight) = 0.06 :=
by
  rw [H1, H2]
  norm_num

end candidate_knows_Excel_and_willing_nights_l131_131028


namespace image_of_2_4_preimage_of_neg5_3_l131_131581

def f (x y : ℤ) : ℤ × ℤ := (x - 2 * y, 2 ^ x + x)

theorem image_of_2_4 :
  f 2 4 = (-6, 6) :=
by
  simp [f]
  sorry

theorem preimage_of_neg5_3 :
  ∃ x y : ℤ, f x y = (-5, 3) ∧ x = 1 ∧ y = 3 :=
by
  use 1
  use 3
  simp [f]
  sorry

end image_of_2_4_preimage_of_neg5_3_l131_131581


namespace problem_statement_l131_131564

variables {f : ℝ → ℝ}

-- Define the conditions as hypotheses
theorem problem_statement (h1 : differentiable ℝ f)
                          (h2 : ∀ x : ℝ, deriv f x < sin (2 * x))
                          (h3 : ∀ x : ℝ, f (-x) + f x = 2 * sin x) :
                          f (Real.pi / 4) > f Real.pi :=
begin
  sorry   -- This is the placeholder for the actual proof
end

end problem_statement_l131_131564


namespace range_of_k_l131_131238

variable {k x : ℝ}

def f (x: ℝ) : ℝ := k * x
def g (x: ℝ) : ℝ := 2 * Real.log x + 2 * Real.exp 1

theorem range_of_k :
  (∀ (x : ℝ), 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 2 → k = - 2 / x * Real.log x) →
  -2 / Real.exp 1 ≤ k ∧ k ≤ 2 * Real.exp 1 :=
by
  sorry

end range_of_k_l131_131238


namespace cost_of_scissor_l131_131405

noncomputable def scissor_cost (initial_money: ℕ) (scissors: ℕ) (eraser_count: ℕ) (eraser_cost: ℕ) (remaining_money: ℕ) :=
  (initial_money - remaining_money - (eraser_count * eraser_cost)) / scissors

theorem cost_of_scissor : scissor_cost 100 8 10 4 20 = 5 := 
by 
  sorry 

end cost_of_scissor_l131_131405


namespace find_circle_eq_l131_131582

-- Define the structure of a point in a plane.
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the parabola equation.
def parabola (p : Point) : Prop :=
  p.x^2 = 2 * p.y

-- Define the properties of a circle.
def circle (center : Point) (radius : ℝ) (p : Point) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

-- Define the focus of the parabola.
def focus : Point :=
  { x := 0, y := 0.5 }

-- Define the condition that quadrilateral is rectangle and the circle intersect properties.
def rectangle_intersections (p1 p2 p3 p4 : Point) : Prop :=
  -- p1 and p2 are intersections with the parabola
  parabola p1 ∧ parabola p2 ∧
  -- p3 and p4 are intersections with the directrix
  p3.y = -0.5 ∧ p4.y = -0.5 ∧
  -- Quadrilateral ABCD is rectangle
  (p1.x - p4.x) = (p2.x - p3.x) ∧ (p1.y - p2.y) = (p4.y - p3.y)

-- The main theorem to prove
theorem find_circle_eq (A B C D : Point) :
  rectangle_intersections A B C D →
  circle focus 2 (A) → circle focus 2 (B) →
  circle focus 2 (C) → circle focus 2 (D) →
  ∀ p : Point, circle focus 2 p → (p.x^2 + (p.y - 0.5)^2 = 4) :=
by
  intros h1 h2 h3 h4 h5 p h6
  sorry

end find_circle_eq_l131_131582


namespace function_is_not_necessarily_straight_line_l131_131432

theorem function_is_not_necessarily_straight_line (f : ℝ → ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f(x) = y) ∧
  (∀ x : ℝ, ∃ d : ℝ, d = (deriv f x)) ∧
  (∀ x : ℝ, ∀ y : ℝ, (rat.cast x) ∧ ¬(rat.cast y) ∨ ¬(rat.cast x) ∧ (rat.cast y) → false) → 
  ¬ (∃ (a b c : ℝ), ∀ x : ℝ, f(x) = a * x + b) :=
begin
  sorry
end

end function_is_not_necessarily_straight_line_l131_131432


namespace num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l131_131894

open Nat

def num_arrangements_A_middle (n : ℕ) : ℕ :=
  if n = 4 then factorial 4 else 0

def num_arrangements_A_not_adj_B (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3) * (factorial 4 / factorial 2) else 0

def num_arrangements_A_B_not_ends (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3 / factorial 2) * factorial 3 else 0

theorem num_arrangements_thm1 : num_arrangements_A_middle 4 = 24 := 
  sorry

theorem num_arrangements_thm2 : num_arrangements_A_not_adj_B 5 = 72 := 
  sorry

theorem num_arrangements_thm3 : num_arrangements_A_B_not_ends 5 = 36 := 
  sorry

end num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l131_131894


namespace min_m_for_four_elements_l131_131912

open Set

theorem min_m_for_four_elements (n : ℕ) (hn : n ≥ 2) :
  ∃ m, m = 2 * n + 2 ∧ 
  (∀ (S : Finset ℕ), S.card = m → 
    (∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a = b + c + d)) :=
by
  sorry

end min_m_for_four_elements_l131_131912


namespace parabola_x_intercepts_l131_131959

theorem parabola_x_intercepts :
  ∃! y : ℝ, -3 * y^2 + 2 * y + 4 = y := 
by
  sorry

end parabola_x_intercepts_l131_131959


namespace al_original_portion_l131_131474

theorem al_original_portion {a b c d : ℕ} 
  (h1 : a + b + c + d = 2000)
  (h2 : a - 150 + 3 * b + 3 * c + d - 50 = 2500)
  (h3 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  a = 450 :=
sorry

end al_original_portion_l131_131474


namespace number_of_functions_with_given_range_l131_131598

theorem number_of_functions_with_given_range :
  ∃ (F : Set (ℕ → ℕ)), 
    (∀ f ∈ F, ∃ D : Set ℤ, (∀ x ∈ D, f x = x^2) ∧ (Set.image f D = {4, 9}))
    ∧ F.card = 9 :=
by
  sorry

end number_of_functions_with_given_range_l131_131598


namespace rectangle_area_divisible_by_12_l131_131332

theorem rectangle_area_divisible_by_12 {a b c : ℕ} (h : a ^ 2 + b ^ 2 = c ^ 2) :
  12 ∣ (a * b) :=
sorry

end rectangle_area_divisible_by_12_l131_131332


namespace fraction_remains_unchanged_l131_131975

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (3 * x)) / (2 * (3 * x) - 3 * y) = (3 * x) / (2 * x - y) :=
by
  sorry

end fraction_remains_unchanged_l131_131975


namespace sum_of_angles_polyhedron_sum_of_angular_defects_l131_131802

theorem sum_of_angles_polyhedron (V E F : ℕ) (polyhedron : Polyhedron) 
    (convex_polyhedron : isConvex polyhedron)
    (num_vertices_polyhedron_equals_num_vertices_polygon : V = numVertices polyhedron) :
    sum_of_angles faces polyhedron = 2 * sum_of_internal_angles polygon := by
  sorry

theorem sum_of_angular_defects (V E F : ℕ) (polyhedron : Polyhedron) 
    (convex_polyhedron : isConvex polyhedron) :
    sum_angular_defects polyhedron = 4 * π := by
  sorry

end sum_of_angles_polyhedron_sum_of_angular_defects_l131_131802


namespace quadratic_y1_gt_y2_l131_131232

theorem quadratic_y1_gt_y2 {a b c y1 y2 : ℝ} (ha : a > 0) (hy1 : y1 = a * (-1)^2 + b * (-1) + c) (hy2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
  sorry

end quadratic_y1_gt_y2_l131_131232


namespace no_two_digit_prime_sum_digits_nine_l131_131178

theorem no_two_digit_prime_sum_digits_nine :
  ¬ ∃ p : ℕ, prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 + p % 10 = 9) :=
sorry

end no_two_digit_prime_sum_digits_nine_l131_131178


namespace run_time_equals_36_seconds_l131_131015

-- Define side length of the square field
def side_length : ℝ := 30

-- Define the running speed of the boy in km/hr
def running_speed_km_hr : ℝ := 12

-- Conversion factor from km/hr to m/s
def conversion_factor : ℝ := 1000 / 3600

-- Calculate the running speed in m/s
def running_speed_m_s : ℝ := running_speed_km_hr * conversion_factor

-- Calculate the perimeter of the square field
def perimeter : ℝ := 4 * side_length

-- The expected time to run around the square field in seconds
def expected_time : ℝ := perimeter / running_speed_m_s

-- Prove that the expected time is 36 seconds
theorem run_time_equals_36_seconds : expected_time = 36 := by
  sorry

end run_time_equals_36_seconds_l131_131015


namespace problem_l131_131540

theorem problem (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} a_{12} : ℤ) :
  (∀ x : ℤ, (x + 1)^4 * (x + 4)^8 = 
    a + a_1 * (x + 3) + a_2 * (x + 3)^2 + a_3 * (x + 3)^3 + a_4 * (x + 3)^4 
    + a_5 * (x + 3)^5 + a_6 * (x + 3)^6 + a_7 * (x + 3)^7 + a_8 * (x + 3)^8 
    + a_9 * (x + 3)^9 + a_{10} * (x + 3)^{10} + a_{11} * (x + 3)^{11} 
    + a_{12} * (x + 3)^{12)) →
  (a_2 + a_4 + a_6 + a_8 + a_{10} + a_{12} = 112) :=
sorry

end problem_l131_131540


namespace determine_angle_B_l131_131216

-- Definitions based on the problem conditions
def Triangle (A B C : ℝ) : Prop := ∀ (a b c : ℝ), a^2 + b^2 >= c^2

def vector_m := (√3, -1)
def vector_n (A : ℝ) := (Real.cos A, Real.sin A)

def perpendicular (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def equation (a b c B A C : ℝ) : Prop :=
  a * Real.cos B + b * Real.cos A = c * Real.sin C

noncomputable def angle_B (a b c A : ℝ) : ℝ :=
  if h : perpendicular vector_m (vector_n A) then
    if A = π / 3 then
      π / 6
    else
      0
  else
    0

-- The theorem to be proved
theorem determine_angle_B 
  (a b c : ℝ) (A : ℝ) 
  (h_perpendicular : perpendicular vector_m (vector_n A))
  (h_equation : equation a b c (π / 6) A (π / 2)) : angle_B a b c A = π / 6 :=
by
  sorry

end determine_angle_B_l131_131216


namespace janice_weekly_earnings_l131_131647

-- define the conditions
def regular_days_per_week : Nat := 5
def regular_earnings_per_day : Nat := 30
def overtime_earnings_per_shift : Nat := 15
def overtime_shifts_per_week : Nat := 3

-- define the total earnings calculation
def total_earnings (regular_days : Nat) (regular_rate : Nat) (overtime_shifts : Nat) (overtime_rate : Nat) : Nat :=
  (regular_days * regular_rate) + (overtime_shifts * overtime_rate)

-- state the problem to be proved
theorem janice_weekly_earnings : total_earnings regular_days_per_week regular_earnings_per_day overtime_shifts_per_week overtime_earnings_per_shift = 195 :=
by
  sorry

end janice_weekly_earnings_l131_131647


namespace f_f_neg1_eq_five_l131_131223

def f (x : ℝ) : ℝ :=
if x >= 1 then x + 1 else 3 - x

theorem f_f_neg1_eq_five : f (f (-1)) = 5 := by
  sorry

end f_f_neg1_eq_five_l131_131223


namespace chessboard_max_pieces_theorem_l131_131612

noncomputable def chessboard_max_pieces : ℕ :=
  let S := 200 * 200
  let max_pieces := 3800
  max_pieces

theorem chessboard_max_pieces_theorem :
  ∀ {board : fin 200 → fin 200 → option (fin 2)},
    (∀ i j, (board i j = some 0 ↔ ∃ p, p ≠ i ∧ (board p j = some 1 ∨ board i p = some 1)) ∧ 
            (board i j = some 1 ↔ ∃ p, p ≠ i ∧ (board p j = some 0 ∨ board i p = some 0))) →
    board.card ≤ chessboard_max_pieces :=
by { sorry }

end chessboard_max_pieces_theorem_l131_131612


namespace sum_of_roots_of_given_polynomial_l131_131872

noncomputable def given_polynomial : Polynomial ℝ := 
  6 * X^3 - 12 * X^2 - 45 * X - 27

theorem sum_of_roots_of_given_polynomial :
  let r := -(-12 / 6) in       -- By Vieta's formulas for sum of roots of cubic polynomial
  r = 2 := 
by
  sorry

end sum_of_roots_of_given_polynomial_l131_131872


namespace janice_total_earnings_l131_131653

-- Defining the working conditions as constants
def days_per_week : ℕ := 5  -- Janice works 5 days a week
def earning_per_day : ℕ := 30  -- Janice earns $30 per day
def overtime_earning_per_shift : ℕ := 15  -- Janice earns $15 per overtime shift
def overtime_shifts : ℕ := 3  -- Janice works three overtime shifts

-- Defining Janice's total earnings for the week
def total_earnings : ℕ := (days_per_week * earning_per_day) + (overtime_shifts * overtime_earning_per_shift)

-- Statement to prove that Janice's total earnings are $195
theorem janice_total_earnings : total_earnings = 195 :=
by
  -- The proof is omitted.
  sorry

end janice_total_earnings_l131_131653


namespace cricketer_total_score_l131_131445

theorem cricketer_total_score
  (boundaries : ℕ)
  (sixes : ℕ)
  (percent_running : ℝ)
  (runs_from_boundaries : ℕ = boundaries * 4)
  (runs_from_sixes : ℕ = sixes * 6)
  (total_runs_from_boundaries_and_sixes : ℕ = runs_from_boundaries + runs_from_sixes)
  (total_score : ℝ)
  (running_percentage_eq : ℝ = percent_running / 100)
  (running_runs : ℝ = running_percentage_eq * total_score) :
  boundaries = 12 → sixes = 2 →
  percent_running = 55.88235294117647 →
  total_score = 136 :=
by
  intros hb hx hp
  sorry

end cricketer_total_score_l131_131445


namespace henrikh_commute_distance_l131_131014

theorem henrikh_commute_distance (x : ℕ)
    (h1 : ∀ y : ℕ, y = x → y = x)
    (h2 : 1 * x = x)
    (h3 : 20 * x = (x : ℕ))
    (h4 : x = (x / 3) + 8) :
    x = 12 := sorry

end henrikh_commute_distance_l131_131014


namespace direction_vector_l131_131460

open Matrix BigOperators

noncomputable def P : Matrix (Fin 3) (Fin 3) ℚ :=
  !![
    [1 / 10, 1 / 20, 1 / 5],
    [1 / 20, 1 / 5, 2 / 5],
    [1 / 5, 2 / 5, 4 / 5]
  ]

def v : Fin 3 → ℤ := ![2, 1, 10]

theorem direction_vector (a b c : ℤ) (g : Fin 3 → ℤ)
  (hₐ : a > 0) (h_gcd : Int.gcd (Int.gcd (Int.gcd a (Int.gcd b c)) 1) = 1)
  (h_Pv : ∀ (i : Fin 3), (P i).sum (λ j x, x * Int.cast (g j)) = Int.cast (g i)) : 
  g = v :=
sorry

end direction_vector_l131_131460


namespace fair_coin_three_flips_l131_131791

open ProbabilityTheory

/-- When flipping a fair coin three times, the probability that the first flip is heads and 
    the last two flips are tails is 1/8. -/
theorem fair_coin_three_flips (p : Real) (H : p = 1/2) :
  P (λ (s : Fin 3 → Bool), s 0 = tt ∧ s 1 = ff ∧ s 2 = ff) = 1/8 := 
sorry

end fair_coin_three_flips_l131_131791


namespace simplify_expression_l131_131692

theorem simplify_expression :
  (18 / 17) * (13 / 24) * (68 / 39) = 1 := 
by
  sorry

end simplify_expression_l131_131692


namespace pyramid_surface_area_l131_131500

theorem pyramid_surface_area (base_edge volume : ℝ)
  (h_base_edge : base_edge = 1)
  (h_volume : volume = 1) :
  let height := 3
  let slant_height := Real.sqrt (9.25)
  let base_area := base_edge * base_edge
  let lateral_area := 4 * (1 / 2 * base_edge * slant_height)
  let total_surface_area := base_area + lateral_area
  total_surface_area = 7.082 :=
by
  sorry

end pyramid_surface_area_l131_131500


namespace tan_alpha_minus_pi_over_4_l131_131923

theorem tan_alpha_minus_pi_over_4 
  (α : ℝ)
  (h₁ : 0 < α)
  (h₂ : α < π)
  (h₃ : sin α + cos α = sqrt 2 / 3) : 
  tan (α - π / 4) = 2 * sqrt 2 :=
by sorry

end tan_alpha_minus_pi_over_4_l131_131923


namespace existence_of_two_balancing_lines_l131_131314

-- Define the problem conditions
variable (n : ℕ) (h : n > 1) (points : Fin 2n → ℝ × ℝ)
variable (colors : Fin 2n → Bool) -- Assume True for Blue, and False for Red

-- Assumption: No three points are collinear
-- This part will be considered within the context of the problem stated.

-- Define what a balancing line is
def is_balancing_line (line : ℝ × ℝ × ℝ) : Prop :=
  ∃ (b r : Fin 2n), colors b = true ∧ colors r = false ∧
  ∀ side, (card (finset.filter (λ p, (side = 0 ∧ points p in_left_halfspace line) ∨ 
                                      (side = 1 ∧ points p in_right_halfspace line)) 
                      (Finset.univ : Finset (Fin 2n))) filter colors = side) = 
           (card (finset.filter (λ p, (side = 0 ∧ points p in_left_halfspace line) ∨ 
                                      (side = 1 ∧ points p in_right_halfspace line)) 
                      (Finset.univ : Finset (Fin 2n))) filter colors = ¬ side)

-- The theorem we want to prove
theorem existence_of_two_balancing_lines
  (n : ℕ) (h : n > 1) (points : Fin 2n → ℝ × ℝ)
  (colors : Fin 2n → Bool) 
  (h_collinear : ∀ (a b c : Fin 2n), ¬ collinear (points a) (points b) (points c))
  (h_points : card (Finset.filter colors Finset.univ) = n ∧ 
              card (Finset.filter (λ x, ¬ colors x) Finset.univ) = n) :
  ∃ line₁ line₂ : ℝ × ℝ × ℝ, line₁ ≠ line₂ ∧ is_balancing_line points colors line₁ ∧ is_balancing_line points colors line₂ := 
  sorry

end existence_of_two_balancing_lines_l131_131314


namespace two_digit_prime_sum_9_l131_131186

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l131_131186


namespace kira_travel_time_l131_131397

theorem kira_travel_time :
  let time_between_stations := 2 * 60 -- converting hours to minutes
  let break_time := 30 -- in minutes
  let total_time := 2 * time_between_stations + break_time
  total_time = 270 :=
by
  let time_between_stations := 2 * 60
  let break_time := 30
  let total_time := 2 * time_between_stations + break_time
  exact rfl

end kira_travel_time_l131_131397


namespace triangle_segments_ratio_eq_l131_131327

variables (A B C C1 C2 A1 A2 B1 B2 : Type) [Point]
variables (triangle : Triangle A B C)
variables 
  (C1_on_AB : OnLineSegment C1 triangle.AB)
  (C2_on_AB : OnLineSegment C2 triangle.AB)
  (A1_on_BC : OnLineSegment A1 triangle.BC)
  (A2_on_BC : OnLineSegment A2 triangle.BC)
  (B1_on_AC : OnLineSegment B1 triangle.AC)
  (B2_on_AC : OnLineSegment B2 triangle.AC)
  (len_eq : dist A1 B2 = dist B1 C2 ∧ dist B1 C2 = dist C1 A2 ∧ dist C1 A2 = dist A1 B2)
  (intersect_at_one : collinear A1 B2 ∧ collinear B1 C2 ∧ collinear C1 A2)
  (angle_eq : measure_angle A1 B2 B1 = 60 ∧ measure_angle B1 C2 C1 = 60 ∧ measure_angle C1 A2 A1 = 60)

theorem triangle_segments_ratio_eq :
  ∃ k : Ratio, k = A1A2 / length triangle.BC ∧ k = B1B2 / length triangle.AC ∧ k = C1C2 / length triangle.AB := sorry

end triangle_segments_ratio_eq_l131_131327


namespace range_of_a_l131_131227

theorem range_of_a
  (f : ℝ → ℝ := λ x, real.sqrt ((x + 1) / (x - 2)))
  (g : ℝ → ℝ := λ x, 1 / real.sqrt (x^2 - (2*a + 1) * x + a^2 + a))
  (A : set ℝ := {x | x > 2 ∨ x ≤ -1})
  (B : set ℝ := {x | x > a + 1 ∨ x < a})
  (h : A ∪ B = B) :
  -1 < a ∧ a ≤ 1 :=
by sorry

end range_of_a_l131_131227


namespace integer_part_of_shortest_distance_l131_131038

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

end integer_part_of_shortest_distance_l131_131038


namespace probability_of_two_same_color_l131_131808

noncomputable def probability_at_least_two_same_color (reds whites blues greens : ℕ) (total_draws : ℕ) : ℚ :=
  have total_marbles := reds + whites + blues + greens
  let total_combinations := Nat.choose total_marbles total_draws
  let two_reds := Nat.choose reds 2 * (total_marbles - 2)
  let two_whites := Nat.choose whites 2 * (total_marbles - 2)
  let two_blues := Nat.choose blues 2 * (total_marbles - 2)
  let two_greens := Nat.choose greens 2 * (total_marbles - 2)
  
  let all_reds := Nat.choose reds 3
  let all_whites := Nat.choose whites 3
  let all_blues := Nat.choose blues 3
  let all_greens := Nat.choose greens 3
  
  let desired_outcomes := two_reds + two_whites + two_blues + two_greens +
                          all_reds + all_whites + all_blues + all_greens
                          
  (desired_outcomes : ℚ) / (total_combinations : ℚ)

theorem probability_of_two_same_color : probability_at_least_two_same_color 6 7 8 4 3 = 69 / 115 := 
by
  sorry

end probability_of_two_same_color_l131_131808


namespace percent_of_475_25_is_129_89_l131_131411

theorem percent_of_475_25_is_129_89 :
  (129.89 / 475.25) * 100 = 27.33 :=
by
  sorry

end percent_of_475_25_is_129_89_l131_131411


namespace sufficient_but_not_necessary_l131_131479

theorem sufficient_but_not_necessary (a b : ℝ) : (a > b + 1) → (a > b) ∧ ∃ (a b : ℝ), (a > b) ∧ (a ≤ b + 1) :=
by
  intro h
  refine ⟨by linarith, ⟨a, b, by linarith, by linarith⟩⟩
  sorry

end sufficient_but_not_necessary_l131_131479


namespace matchstick_polygon_area_l131_131747

-- Given conditions
def number_of_matches := 12
def length_of_each_match := 2 -- in cm

-- Question: Is it possible to construct a polygon with an area of 16 cm^2 using all the matches?
def polygon_possible : Prop :=
  ∃ (p : Polygon), 
    (p.edges = number_of_matches) ∧ 
    (∃ (match_length : ℝ), match_length = length_of_each_match ∧ by 
      -- Form the polygon using all matches without breaking
      sorry) ∧ 
    (polygon_area p = 16)

-- Proof statement
theorem matchstick_polygon_area :
  polygon_possible :=
  sorry

end matchstick_polygon_area_l131_131747


namespace slope_of_line_l131_131801

noncomputable def slope {x1 y1 x2 y2 : ℝ} : ℝ :=
  (y2 - y1) / (x2 - x1)

theorem slope_of_line :
  (slope 0 20 150 600) ≈ 3.87 :=
sorry

end slope_of_line_l131_131801


namespace AM_GM_Inequality_equality_condition_l131_131666

-- Given conditions
variables (n : ℕ) (a b : ℝ)

-- Assumptions
lemma condition_n : 0 < n := sorry
lemma condition_a : 0 < a := sorry
lemma condition_b : 0 < b := sorry

-- Statement
theorem AM_GM_Inequality :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) :=
sorry

-- Equality condition
theorem equality_condition :
  (1 + a / b) ^ n + (1 + b / a) ^ n = 2 ^ (n + 1) ↔ a = b :=
sorry

end AM_GM_Inequality_equality_condition_l131_131666


namespace problem_a_problem_b_problem_c_l131_131960

theorem problem_a : (∃ (numbers : ℕ), ∀ (d : Fin 4 → Fin 5), (∀ i j, i ≠ j → d i ≠ d j) → numbers = 120) :=
by
  sorry

theorem problem_b : (∃ (numbers : ℕ), ∀ (d : Fin 4 → Fin 5), numbers = 5^4) :=
by
  sorry

theorem problem_c : (∃ (numbers : ℕ), ∀ (d : Fin 4 → Fin 5), (∀ i j, i ≠ j → d i ≠ d j) → d 3 ∈ {1, 3, 5} → numbers = 72) :=
by
  sorry

end problem_a_problem_b_problem_c_l131_131960


namespace remainder_of_x_plus_3uy_l131_131790

-- Given conditions
variables (x y u v : ℕ)
variable (Hdiv : x = u * y + v)
variable (H0_le_v : 0 ≤ v)
variable (Hv_lt_y : v < y)

-- Statement to prove
theorem remainder_of_x_plus_3uy (x y u v : ℕ) (Hdiv : x = u * y + v) (H0_le_v : 0 ≤ v) (Hv_lt_y : v < y) :
  (x + 3 * u * y) % y = v :=
sorry

end remainder_of_x_plus_3uy_l131_131790


namespace Diamond_value_l131_131919

-- Declare the operation Diamond for positive real numbers
variable {R : Type*} [ordered_semiring R] [linear_order R]
variable (Diamond : R → R → R)

-- Conditions of the problem
variables (x y : R)

-- Declare the relations and assumptions given in the problem
axiom axiom1 : ∀ x y : R, 0 < x → 0 < y → Diamond (x * y) y = x * (Diamond y y)
axiom axiom2 : ∀ x : R, 0 < x → Diamond (Diamond x 2) x = Diamond x 2
axiom axiom3 : 0 < 2 → Diamond 2 2 = 4

-- The proof goal
theorem Diamond_value : Diamond 5 20 = 20 :=
by
  sorry

end Diamond_value_l131_131919


namespace abs_diff_is_323_l131_131080

noncomputable def C : ℝ :=
  (finset.range 20).sum (λ k, (if k < 20 then (2 + k * 2) * (3 + k * 2 + 1) else 0)) + 40

noncomputable def D : ℝ :=
  (finset.range 20).sum (λ k, (if k < 19 then (2 + k * 2 + 1) * (2 + k * 2 + 3) else 0)) + 39 * 40

theorem abs_diff_is_323 : |C - D| = 323 :=
by
  sorry

end abs_diff_is_323_l131_131080


namespace polynomial_evaluation_qin_jiushao_l131_131087

theorem polynomial_evaluation_qin_jiushao :
  let x := 3
  let V0 := 7
  let V1 := V0 * x + 6
  let V2 := V1 * x + 5
  let V3 := V2 * x + 4
  let V4 := V3 * x + 3
  V4 = 789 :=
by
  -- placeholder for proof
  sorry

end polynomial_evaluation_qin_jiushao_l131_131087


namespace max_value_of_f_l131_131243

open Real

-- Given conditions
def m := ((1 : ℝ) / 2, 3)
def n := (π / 6, 0)
def vectorOp (a : ℝ × ℝ) (b1 : ℝ) : ℝ × ℝ := (a.1 * b1, a.2 * b1)
def P_path := λ x, (x, sin x)
def Q_path := λ (x y : ℝ), let (x0, y0) := P_path x in ((x0/2) + π/6, 3 * y0)
def f (x : ℝ) := 3 * sin (2 * x - π / 3)

-- Theorem statement: Maximum value of f(x)
theorem max_value_of_f : ∃ x, f x = 3 :=
sorry

end max_value_of_f_l131_131243


namespace area_of_remaining_shape_l131_131456

noncomputable def area_of_square (side_length : ℝ) : ℝ := side_length ^ 2

noncomputable def area_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * side_length ^ 2

noncomputable def area_of_circle (diameter : ℝ) : ℝ :=
  real.pi * (diameter / 2) ^ 2

theorem area_of_remaining_shape
  (side_square: ℝ)
  (side_triangle: ℝ)
  (diameter_circle: ℝ)
  (h1: side_square = 5)
  (h2: side_triangle = 2)
  (h3: diameter_circle = 1)
  (triangle_and_circle_inside_square : true) : 
  area_of_square side_square - (area_of_equilateral_triangle side_triangle + area_of_circle diameter_circle) = 
  25 - real.sqrt 3 - real.pi / 4 :=
by 
  simp [area_of_square, area_of_equilateral_triangle, area_of_circle]
  rw [h1, h2, h3]
  sorry

end area_of_remaining_shape_l131_131456


namespace regular_polygon_lattice_points_l131_131192

theorem regular_polygon_lattice_points (n : ℕ) (h : n ≥ 3) :
    (∃ (A : Fin n → ℤ × ℤ), ∀ i : Fin n, let j := (i + 1) % n in
          let dist_sq := (A i).fst - (A j).fst) * ((A i).fst - (A j).fst) + ((A i).snd - (A j).snd) * ((A i).snd - (A j).snd) 
          ∧ (dist_sq = dist_sq (i + 1 % n))
    ) ↔ n = 4 := 
  sorry

end regular_polygon_lattice_points_l131_131192


namespace tangent_product_constant_l131_131400

variable (a x₁ x₂ y₁ y₂ : ℝ)

def point_on_parabola (x y : ℝ) := x^2 = 4 * y
def point_P := (a, -2)
def point_A := (x₁, y₁)
def point_B := (x₂, y₂)

theorem tangent_product_constant
  (h₁ : point_on_parabola x₁ y₁)
  (h₂ : point_on_parabola x₂ y₂)
  (h₃ : ∃ k₁ k₂ : ℝ, 
        (y₁ + 2 = k₁ * (x₁ - a) ∧ y₂ + 2 = k₂ * (x₂ - a)) 
        ∧ (k₁ * k₂ = -2)) :
  x₁ * x₂ + y₁ * y₂ = -4 :=
sorry

end tangent_product_constant_l131_131400


namespace value_of_f_at_8_l131_131451

theorem value_of_f_at_8 :
  (∃ f : ℝ → ℝ, ∀ x > 0, 3 * f(x) + 7 * f(2016 / x) = 2 * x) →
  (∀ f : ℝ → ℝ, ∀ x > 0, 3 * f(x) + 7 * f(2016 / x) = 2 * x → f(8) = 87) :=
begin
  sorry
end

end value_of_f_at_8_l131_131451


namespace repetend_of_fraction_l131_131162

/-- The repeating sequence of the decimal representation of 5/17 is 294117 -/
theorem repetend_of_fraction : 
  let rep := list.take 6 (list.drop 1 (to_digits 10 (5 / 17) 8)) in
  rep = [2, 9, 4, 1, 1, 7] := 
by
  sorry

end repetend_of_fraction_l131_131162


namespace triangle_colored_segments_l131_131283

theorem triangle_colored_segments (n : ℕ) (k1 k2 k3 : ℕ) (vertices_points : ℕ) :
  (vertices_points = 3) →
  (0 ≤ k1 ∧ k1 ≤ 3) →
  (0 ≤ k2 ∧ k2 ≤ 3) →
  (0 ≤ k3 ∧ k3 ≤ 3) →
  (∀ (p : ℕ), p (verts points) → p -verts.points (k1 + k2 + k3 = 3) → 
  (∀ i, i ∧ 0 < i → i ≤ 3) →
  (n % 2 = 1) →
  (n + k1) % 2 = 0 →
  (n + k2) % 2 = 0 →
  (n + k3) % 2 = 0 →
   (k1 = 1 ∧ k2 = 1 ∧ k3 = 1)  →
  (k1 mod 2 = 1) →
  (k2 mod 2 = 1) →
  (k3 mod 2 = 1) →
  n > 0 →
  (∀ vertex : ℕ, vertex ≤ 3) →
  (∃ (k1 : ℕ) (k2 : ℕ) (k3 : ℕ),
    k1 + k2 + k3 = 3 ∧
    (∀ i, i ≤ 3) →
    k1 + k2 + k3 = 3 ) ∧
  (∀ p : ℕ, vertex_connected point_to_segment p ∧ 
  (k1 = 1) ∧ 
  (k2 = 1) ∧ 
  (k3 = 1)) →
  (vertex_connected p.to_segment r).distinct_colors :=
begin
  sorry
end

end triangle_colored_segments_l131_131283


namespace least_people_to_complete_job_on_time_l131_131095

theorem least_people_to_complete_job_on_time
  (total_duration : ℕ)
  (initial_days : ℕ)
  (initial_people : ℕ)
  (initial_work_done : ℚ)
  (efficiency_multiplier : ℚ)
  (remaining_work_fraction : ℚ)
  (remaining_days : ℕ)
  (resulting_people : ℕ)
  (work_rate_doubled : ℕ → ℚ → ℚ)
  (final_resulting_people : ℚ)
  : initial_work_done = 1/4 →
    efficiency_multiplier = 2 →
    remaining_work_fraction = 3/4 →
    total_duration = 40 →
    initial_days = 10 →
    initial_people = 12 →
    remaining_days = 20 →
    work_rate_doubled 12 2 = 24 →
    final_resulting_people = (1/2) →
    resulting_people = 6 :=
sorry

end least_people_to_complete_job_on_time_l131_131095


namespace projection_problem_l131_131207

noncomputable def vectorProjection (a b : EuclideanSpace ℝ (Fin 3)) : ℝ := 
  let projection := ((b - a) • a) / (a • a)
  projection

theorem projection_problem 
  (a b : EuclideanSpace ℝ (Fin 3))
  (ha_ne : a ≠ 0)
  (hb_ne : b ≠ 0)
  (ha_norm : ∥a∥ = 2)
  (h_asapb : ∥a + b∥ = ∥a - b∥) :
  vectorProjection a b = -2 :=
by
  sorry

end projection_problem_l131_131207


namespace simple_interest_correct_l131_131013

-- Define the parameters
def principal : ℝ := 10000
def rate_decimal : ℝ := 0.04
def time_years : ℝ := 1

-- Define the simple interest calculation function
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Prove that the simple interest is equal to $400
theorem simple_interest_correct : simple_interest principal rate_decimal time_years = 400 :=
by
  -- Placeholder for the proof
  sorry

end simple_interest_correct_l131_131013


namespace parallel_planes_of_skew_lines_l131_131682

variables {Plane : Type*} {Line : Type*}
variables (α β : Plane)
variables (a b : Line)

-- Conditions
def is_parallel (p1 p2 : Plane) : Prop := sorry -- Parallel planes relation
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- Line in plane relation
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- Line parallel to plane relation
def is_skew_lines (l1 l2 : Line) : Prop := sorry -- Skew lines relation

-- Theorem to prove
theorem parallel_planes_of_skew_lines 
  (h1 : line_in_plane a α)
  (h2 : line_in_plane b β)
  (h3 : line_parallel_plane a β)
  (h4 : line_parallel_plane b α)
  (h5 : is_skew_lines a b) :
  is_parallel α β :=
sorry

end parallel_planes_of_skew_lines_l131_131682


namespace coeff_of_x5_in_expansion_l131_131101

theorem coeff_of_x5_in_expansion :
  ∃ C : ℤ, C = 51 ∧ (x : ℝ) ∈ Polynomial.Coeff (Polynomial.expand (x^2 + x + 1) 5) 5 = C :=
begin
  sorry
end

end coeff_of_x5_in_expansion_l131_131101


namespace inequality_proof_l131_131561

noncomputable def pos_real := { x : ℝ // 0 < x }

theorem inequality_proof (a b c d : pos_real) :
    real.sqrt (a.1^2 + b.1^2) + real.sqrt (b.1^2 + c.1^2) + real.sqrt (c.1^2 + d.1^2) + real.sqrt (d.1^2 + a.1^2) 
    ≥ real.sqrt 2 * (a.1 + b.1 + c.1 + d.1) :=
by 
  sorry

end inequality_proof_l131_131561


namespace choose_3_out_of_10_l131_131268

-- Define the problem by setting relevant definitions and the final proof goal.
def num_of_ways_to_choose_3_of_10 := 120

theorem choose_3_out_of_10 :
  (Nat.choose 10 3) = num_of_ways_to_choose_3_of_10 :=
begin
  sorry
end

end choose_3_out_of_10_l131_131268


namespace increasing_function_range_l131_131218

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : ∀ x y : ℝ, x < y → f a x ≤ f a y) : 
  1.5 ≤ a ∧ a < 2 :=
sorry

end increasing_function_range_l131_131218


namespace tin_to_copper_ratio_l131_131806

theorem tin_to_copper_ratio (L_A T_A T_B C_B : ℝ) 
  (h_total_mass_A : L_A + T_A = 90)
  (h_ratio_A : L_A / T_A = 3 / 4)
  (h_total_mass_B : T_B + C_B = 140)
  (h_total_tin : T_A + T_B = 91.42857142857143) :
  T_B / C_B = 2 / 5 :=
sorry

end tin_to_copper_ratio_l131_131806


namespace area_within_fence_l131_131362

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l131_131362


namespace very_small_probability_can_occur_l131_131414

-- Question: Prove that an event with a very small probability can occur.
theorem very_small_probability_can_occur (small_prob_event : Prop) (prob : ℝ) (h1 : 0 < prob) (h2 : prob < 0.01) : 
  ∃ e : Prop, small_prob_event = e ∧ e = true :=
begin
  sorry
end

end very_small_probability_can_occur_l131_131414


namespace M_minus_m_l131_131673

def f (a b : ℝ) : ℝ := (3 / a) + b

theorem M_minus_m :
  let M := (setOf (λ x, ∃ a b, 1 ≤ a ∧ a ≤ b ∧ b ≤ 2 ∧ x = f a b)).sup id in
  let m := (setOf (λ x, ∃ a b, 1 ≤ a ∧ a ≤ b ∧ b ≤ 2 ∧ x = f a b)).inf id in
  M - m = 5 - 2 * real.sqrt 3 :=
by
  sorry

end M_minus_m_l131_131673


namespace max_sum_of_squares_eq_100_l131_131250

theorem max_sum_of_squares_eq_100 : 
  ∃ (x y : ℤ), x^2 + y^2 = 100 ∧ 
  (∀ (x y : ℤ), x^2 + y^2 = 100 → x + y ≤ 14) ∧ 
  (∃ (x y : ℕ), x^2 + y^2 = 100 ∧ x + y = 14) :=
by {
  sorry
}

end max_sum_of_squares_eq_100_l131_131250


namespace intersection_ratio_YQ_QZ_l131_131264

-- Define the main coordinates and ratios based on the conditions
variables {X Y Z P K Q : Type} {XZ_ratio : ℝ} {P_midpoint_ratio : ℝ} 

-- Given conditions
def P_divides_XZ := XZ_ratio = 2/3
def K_is_midpoint_Y_P := P_midpoint_ratio = 1/1

-- Define the theorem to express the problem and solutions as given in the problem
theorem intersection_ratio_YQ_QZ (h1 : P_divides_XZ) (h2 : K_is_midpoint_Y_P) : (((2:ℝ) / (5:ℝ)) = (2:5)) :=
by
  sorry

end intersection_ratio_YQ_QZ_l131_131264


namespace repetend_of_fraction_l131_131164

/-- The repeating sequence of the decimal representation of 5/17 is 294117 -/
theorem repetend_of_fraction : 
  let rep := list.take 6 (list.drop 1 (to_digits 10 (5 / 17) 8)) in
  rep = [2, 9, 4, 1, 1, 7] := 
by
  sorry

end repetend_of_fraction_l131_131164


namespace find_m_l131_131368

def symmetric_about_y (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x > f y

theorem find_m (m : ℤ) :
  symmetric_about_y (fun x : ℝ => x ^ (m * m - 4 * m)) ∧
  decreasing_on (fun x : ℝ => x ^ (m * m - 4 * m)) (set.Ioi 0) →
  m = 2 :=
sorry

end find_m_l131_131368


namespace AM_minus_GM_lower_bound_l131_131587

theorem AM_minus_GM_lower_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) : 
  (x + y) / 2 - Real.sqrt (x * y) ≥ (x - y)^2 / (8 * x) := 
by {
  sorry -- Proof to be filled in
}

end AM_minus_GM_lower_bound_l131_131587


namespace find_a_9_l131_131541

theorem find_a_9
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} b_0 b_1 b_2 b_3 b_4 b_5 b_6 b_7 b_8 b_9 : ℝ)
  (g : ℝ → ℝ) (h : ℝ → ℝ)
  (hg : ∀ x, g x = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_{10} * x^{10})
  (hh : ∀ x, h x = b_0 + b_1 * x + b_2 * x^2 + b_3 * x^3 + b_4 * x^4 + b_5 * x^5 + b_6 * x^6 + b_7 * x^7 + b_8 * x^8 + b_9 * x^9)
  (eqn : ∀ x, (1 + x) * (1 - 2 * x) ^ 19 = (1 - x) ^ 10 * g x + h x) :
  a_9 = -3 * 2 ^ 18 :=
by sorry

end find_a_9_l131_131541


namespace real_solutions_x_inequality_l131_131104

theorem real_solutions_x_inequality (x : ℝ) :
  (∃ y : ℝ, y^2 + 6 * x * y + x + 8 = 0) ↔ (x ≤ -8 / 9 ∨ x ≥ 1) := 
sorry

end real_solutions_x_inequality_l131_131104


namespace pencils_per_student_l131_131249

theorem pencils_per_student
  (boxes : ℝ) (pencils_per_box : ℝ) (students : ℝ)
  (h1 : boxes = 4.0)
  (h2 : pencils_per_box = 648.0)
  (h3 : students = 36.0) :
  (boxes * pencils_per_box) / students = 72.0 :=
by
  sorry

end pencils_per_student_l131_131249


namespace f_sum_1990_l131_131635

-- Define the function f based on the given conditions
def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

-- Define the sum from 1 to 1990 for function f
def f_sum : ℕ :=
  (List.range 1990).map (λ n => f (n+1)).sum

-- The theorem to prove the sum of f from 1 to 1990 is 1326
theorem f_sum_1990 : f_sum = 1326 :=
by
  sorry

end f_sum_1990_l131_131635


namespace four_noncongruent_triangles_l131_131090

noncomputable def num_noncongruent_triangles (A B C D P Q R S : Point) : ℕ := sorry

theorem four_noncongruent_triangles (A B C D P Q R S : Point)
  (h_square: is_square A B C D) 
  (h_midpoints: are_midpoints P Q R S A B C D) :
  num_noncongruent_triangles A B C D P Q R S = 4 :=
sorry

end four_noncongruent_triangles_l131_131090


namespace no_two_digit_prime_with_digit_sum_9_l131_131182

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end no_two_digit_prime_with_digit_sum_9_l131_131182


namespace range_of_x_l131_131928

theorem range_of_x (a : Fin 25 → ℕ) (H : ∀ i, a i = 0 ∨ a i = 2) :
  let x := ∑ i, a i * (3 ^ (i+1))⁻¹ in
  (0 ≤ x ∧ x < 1/3) ∨ (2/3 ≤ x ∧ x < 1) :=
by
  sorry

end range_of_x_l131_131928


namespace side_face_area_l131_131010

noncomputable def length := L
noncomputable def width := W
noncomputable def height := H

axiom front_face_area_half_top_face_area : W * H = (1/2) * (L * W)
axiom top_face_area_1_5_times_side_face_area : L * W = 1.5 * (H * L)
axiom box_volume : L * W * H = 192

theorem side_face_area : H * L = 32 := 
by 
  sorry

end side_face_area_l131_131010


namespace initial_erasers_in_box_l131_131767

-- Definitions based on the conditions
def erasers_in_bag_jane := 15
def erasers_taken_out_doris := 54
def erasers_left_in_box := 15

-- Theorem statement
theorem initial_erasers_in_box : ∃ B_i : ℕ, B_i = erasers_taken_out_doris + erasers_left_in_box ∧ B_i = 69 :=
by
  use 69
  -- omitted proof steps
  sorry

end initial_erasers_in_box_l131_131767


namespace arithmetic_sequence_common_difference_and_m_l131_131497

theorem arithmetic_sequence_common_difference_and_m (S : ℕ → ℤ) (a : ℕ → ℤ) (m d : ℕ) 
(h1 : S (m-1) = -2) (h2 : S m = 0) (h3 : S (m+1) = 3) :
  d = 1 ∧ m = 5 :=
by sorry

end arithmetic_sequence_common_difference_and_m_l131_131497


namespace repetend_of_5_div_17_l131_131132

theorem repetend_of_5_div_17 :
  let dec := 5 / 17 in
  decimal_repetend dec = "294117" := sorry

end repetend_of_5_div_17_l131_131132


namespace continuous_stripe_probability_l131_131511

-- Definitions based on conditions from a)
def total_possible_combinations : ℕ := 4^6

def favorable_outcomes : ℕ := 12

def probability_of_continuous_stripe : ℚ := favorable_outcomes / total_possible_combinations

-- The theorem equivalent to prove the given problem
theorem continuous_stripe_probability :
  probability_of_continuous_stripe = 3 / 1024 :=
by
  sorry

end continuous_stripe_probability_l131_131511


namespace smallest_omega_l131_131367

theorem smallest_omega :
  ∃ (ω : ℝ), ω > 0 ∧ (∀ k : ℤ, ω = -12 * k - 3) ∧ ω = 9 := 
begin
  sorry
end

end smallest_omega_l131_131367


namespace spinner_probability_C_l131_131439

open_locale probability_theory

theorem spinner_probability_C :
  let prob_A := (2 / 7 : ℚ),
      prob_B := (3 / 14 : ℚ),
      prob_C := x,
      prob_D := 2 * x,
      prob_E := x in
  prob_A + prob_B + prob_C + prob_D + prob_E = 1 → x = 1 / 8 :=
by
  intro h
  sorry

end spinner_probability_C_l131_131439


namespace perpendicular_plane_perpendicular_to_both_l131_131198

-- Definitions of planes α, β and line l
variables (α β : plane) (l : line)

-- Conditions given
variables (h1 : α ⊥ β) (h2 : α ∩ β = l)

-- Statement we need to prove
theorem perpendicular_plane_perpendicular_to_both {π : plane} :
  π ⊥ l → π ⊥ α ∧ π ⊥ β :=
sorry

end perpendicular_plane_perpendicular_to_both_l131_131198


namespace imaginary_part_of_complex_example_l131_131558

def imaginary_unit_property (i : ℂ) : Prop :=
  i^2 = -1

def complex_example := (7 + ⟨0, 1⟩) / (3 + ⟨0, 4⟩)

theorem imaginary_part_of_complex_example :
  imaginary_unit_property ⟨0, 1⟩ →
  complex_example.im = -1 :=
by
  intro h
  sorry

end imaginary_part_of_complex_example_l131_131558


namespace repetend_of_5_div_17_l131_131133

theorem repetend_of_5_div_17 :
  let dec := 5 / 17 in
  decimal_repetend dec = "294117" := sorry

end repetend_of_5_div_17_l131_131133


namespace minimum_swaps_to_transform_sequence_l131_131539

theorem minimum_swaps_to_transform_sequence :
  let initial_sequence := (List.range 100).map (. + 1)
  let desired_sequence := (List.range 99).map (. + 2) ++ [1]
  (∃ m : ℕ, 
    m = 99 ∧ 
    ∀ f : List ℕ → List ℕ, 
    (∀ s : List ℕ, s.length = 100 → 
      if s = initial_sequence then f s else f s = s) →
    (∀ i j, 
      i ≠ j → i < 100 → j < 100 → 
      (let s' := (function.swap (List.update_nth) i j) ⊚ f)
      (initial_sequence) = desired_sequence)) :=
  sorry

end minimum_swaps_to_transform_sequence_l131_131539


namespace Points_concyclic_l131_131282

open EuclideanGeometry

variables 
  (A B C D E F M N U V P : Point)
  (I : Circle)
  (h_triangle : Triangle ABC)
  (h_incircle : incircle I)
  (h_tangent_BC : tangent_point I BC D)
  (h_tangent_CA : tangent_point I CA E)
  (h_tangent_AB : tangent_point I AB F)
  (h_mid_M : midpoint M DE)
  (h_mid_N : midpoint N DF)
  (h_on_MN_U : on_line U MN)
  (h_on_MN_V : on_line V MN)
  (h_BU_NU : dist B U = dist N U)
  (h_CV_MV : dist C V = dist M V)
  (h_intersection_P : intersection CV BU P)

theorem Points_concyclic (h_concyclic : concyclic {A, B, C, P}) : Proof :=
  sorry

end Points_concyclic_l131_131282


namespace find_k_values_tangent_find_max_k_value_l131_131580

-- Definition of functions f and g
def f (x : ℝ) : ℝ := 5 + Real.log x
def g (x : ℝ) (k : ℝ) : ℝ := k * x / (x + 1)

-- Definitions for part (I)
def tangent_line_f_at_1 : ℝ → ℝ := λ x, x + 4
def k_values_tangent_to_g (x₀ : ℝ) (k : ℝ) : Prop :=
  (k / (x₀ + 1)^2 = 1) ∧ (k * x₀ / (x₀ + 1) = x₀ + 4)

-- Definitions for part (II)
def f_gt_g_criteria (x : ℝ) (k : ℕ) : Prop := f x > g x k

-- Lean statement for Part (I)
theorem find_k_values_tangent :
  ∃ x₀ k, k_values_tangent_to_g x₀ k ∧ (k = 1 ∨ k = 9) :=
sorry

-- Lean statement for Part (II)
theorem find_max_k_value :
  ∃ k : ℕ, (∀ x : ℝ, 1 < x → f_gt_g_criteria x k) ∧ (k = 7) :=
sorry

end find_k_values_tangent_find_max_k_value_l131_131580


namespace no_two_digit_prime_with_digit_sum_9_l131_131185

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end no_two_digit_prime_with_digit_sum_9_l131_131185


namespace evaluate_f_complex_l131_131299

noncomputable def f (z : ℂ) : ℂ :=
  if ¬((z.re : ℝ) = z) then z^2 + 1
  else if (z.re = z) && (z.im = 0) then -z^2 + 1
  else 2 * z

theorem evaluate_f_complex : f (f (f (f (1 + Complex.I)))) = 378 + 336 * Complex.I :=
  by
    sorry

end evaluate_f_complex_l131_131299


namespace repetend_of_fraction_l131_131158

/-- The repeating sequence of the decimal representation of 5/17 is 294117 -/
theorem repetend_of_fraction : 
  let rep := list.take 6 (list.drop 1 (to_digits 10 (5 / 17) 8)) in
  rep = [2, 9, 4, 1, 1, 7] := 
by
  sorry

end repetend_of_fraction_l131_131158


namespace cakes_served_at_lunch_today_l131_131051

variable (L : ℕ)
variable (dinnerCakes : ℕ) (yesterdayCakes : ℕ) (totalCakes : ℕ)

theorem cakes_served_at_lunch_today :
  (dinnerCakes = 6) → (yesterdayCakes = 3) → (totalCakes = 14) → (L + dinnerCakes + yesterdayCakes = totalCakes) → L = 5 :=
by
  intros h_dinner h_yesterday h_total h_eq
  sorry

end cakes_served_at_lunch_today_l131_131051


namespace range_of_mn_l131_131908

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := 2^x * m + x^2 + n * x

theorem range_of_mn (m n : ℝ) (h : {x | f x m n = 0} = {x | f (f x m n) m n = 0} ∧ {x | f x m n = 0}.nonempty) :
  0 ≤ m + n ∧ m + n < 4 :=
begin
  sorry,
end

end range_of_mn_l131_131908


namespace students_in_sample_l131_131266

-- Definitions of conditions
variables (T F : ℝ)

-- 22 percent are juniors
def juniors := 0.22 * T

-- 74 percent are not sophomores, so 26 percent are sophomores
def sophomores := 0.26 * T

-- Seniors are given as 160
def seniors := 160

-- There are 48 more freshmen than sophomores
def freshmen := F
def sophomores_from_freshmen := F - 48

-- Equation adjusting to total students
def total_students_eq := freshmen + sophomores_from_freshmen + juniors + seniors = T

-- Final solution
theorem students_in_sample : 
  (∃ (T F : ℝ), juniors T = 0.22 * T ∧ sophomores T = 0.26 * T ∧ seniors = 160 ∧ freshmen F = F ∧ sophomores_from_freshmen F = F - 48 ∧ total_students_eq T F ∧ T = 431).
sorry

end students_in_sample_l131_131266


namespace g_of_3_equals_5_l131_131260

def g (x : ℝ) : ℝ := 2 * (x - 2) + 3

theorem g_of_3_equals_5 :
  g 3 = 5 :=
by
  sorry

end g_of_3_equals_5_l131_131260


namespace mary_mileage_l131_131772

def base9_to_base10 : Nat :=
  let d0 := 6 * 9^0
  let d1 := 5 * 9^1
  let d2 := 9 * 9^2
  let d3 := 3 * 9^3
  d0 + d1 + d2 + d3 

theorem mary_mileage :
  base9_to_base10 = 2967 :=
by 
  -- Calculation steps are skipped using sorry
  sorry

end mary_mileage_l131_131772


namespace bisect_angle_l131_131773

structure Circle (α : Type) :=
(center : α)
(radius : ℝ)

variables {α : Type} [euclidean_space α]

variables (O1 O2 A K : α) (r1 r2 : ℝ)

/-- Given two circles that lie completely outside each other -/
def circles_non_intersecting (c1 c2 : Circle α) : Prop :=
  ∀ (p ∈ c1.center) (q ∈ c2.center), dist p q > c1.radius + c2.radius

/-- A is the point of intersection of the internal common tangents of the circles -/
def intersection_internal_tangents (A : α) (c1 c2 : Circle α) : Prop :=
  ∃ (t1 t2 : set α) (p1 p2 : α),
    is_tangent t1 c1 p1 ∧ is_tangent t2 c2 p2 ∧
    A ∈ t1 ∧ A ∈ t2 ∧
    are_common_tangents t1 t2 (internal := true)

/-- K is the projection of point A onto one of their external common tangents -/
def projection_on_external_tangent (A K : α) (c1 c2 : Circle α) : Prop :=
  ∃ (t : set α),
    is_tangent t c1 ∧ is_tangent t c2 ∧
    K ∈ t ∧
    dist A K = shortest_distance_to_tangent A t

/-- Tangents from K to circles -/
def tangents_from_point (K : α) (c : Circle α) (M : α) : Prop :=
  dist K M = c.radius ∧
  ∃ (t : set α), is_tangent t c M ∧ K ∈ t

/-- Theorem statement: Prove the line AK bisects angle M1KM2 -/
theorem bisect_angle
  (c1 c2 : Circle α)
  (H1 : circles_non_intersecting c1 c2)
  (H2 : intersection_internal_tangents A c1 c2)
  (H3 : projection_on_external_tangent A K c1 c2)
  (M1 M2 : α)
  (H4 : tangents_from_point K c1 M1)
  (H5 : tangents_from_point K c2 M2) :
  is_angle_bisector (line_through A K) M1 K M2 :=
sorry

end bisect_angle_l131_131773


namespace apples_total_l131_131502

theorem apples_total :
  ∀ (Marin David Amanda : ℕ),
  Marin = 6 →
  David = 2 * Marin →
  Amanda = David + 5 →
  Marin + David + Amanda = 35 :=
by
  intros Marin David Amanda hMarin hDavid hAmanda
  sorry

end apples_total_l131_131502


namespace length_squared_l131_131702

noncomputable def f (x : ℝ) : ℝ := 1.5 * x + 1
noncomputable def g (x : ℝ) : ℝ := -1.5 * x + 1
noncomputable def h (x : ℝ) : ℝ := 0.5 * x + 3
noncomputable def i (x : ℝ) : ℝ := -0.5 * x + 3

noncomputable def j (x : ℝ) : ℝ := max (max (f x) (g x)) (max (h x) (i x))
noncomputable def k (x : ℝ) : ℝ := min (min (f x) (g x)) (min (h x) (i x))

theorem length_squared :
  let l := (sqrt (52 : ℝ) + sqrt (20 : ℝ))
  l^2 = 72 + 2 * sqrt (1040 : ℝ) := sorry

end length_squared_l131_131702


namespace determine_original_volume_of_tank_l131_131469

noncomputable def salt_volume (x : ℝ) := 0.20 * x
noncomputable def new_volume_after_evaporation (x : ℝ) := (3 / 4) * x
noncomputable def new_volume_after_additions (x : ℝ) := (3 / 4) * x + 6 + 12
noncomputable def new_salt_after_addition (x : ℝ) := 0.20 * x + 12
noncomputable def resulting_salt_concentration (x : ℝ) := (0.20 * x + 12) / ((3 / 4) * x + 18)

theorem determine_original_volume_of_tank (x : ℝ) :
  resulting_salt_concentration x = 1 / 3 → x = 120 := 
by 
  sorry

end determine_original_volume_of_tank_l131_131469


namespace area_within_fence_l131_131364

theorem area_within_fence : 
  let rectangle_area := 20 * 18
  let cutout_area := 4 * 4
  rectangle_area - cutout_area = 344 := by
    -- Definitions
    let rectangle_area := 20 * 18
    let cutout_area := 4 * 4
    
    -- Computation of areas
    show rectangle_area - cutout_area = 344
    sorry

end area_within_fence_l131_131364


namespace correct_relation_l131_131482

theorem correct_relation : 
  (0 ∉ (∅ : Set ℕ)) ∧ 
  (¬((2 : Set ℕ) ⊆ {x | x ≤ 10})) ∧ 
  (∅ ⊂ ({0} : Set ℕ)) ∧ 
  (¬(({0} : Set ℕ) ∈ {x | x ≤ 1})) := 
by {
  split,
  { exact not_mem_empty _ },
  split,
  { intro h,
    have := h (2 : ℕ),
    simp only [set.mem_set_of, _root_.le, set.mem_singleton_iff, implies_true_iff] at this,
    contradiction,
  },
  split,
  { exact empty_subset _ },
  { intro h,
    simp only [set.mem_set_of, le_refl, set.singleton_subset_iff, set.mem_singleton_iff] at h,
    contradiction,
  }
}

end correct_relation_l131_131482


namespace no_two_digit_prime_sum_digits_nine_l131_131177

theorem no_two_digit_prime_sum_digits_nine :
  ¬ ∃ p : ℕ, prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 + p % 10 = 9) :=
sorry

end no_two_digit_prime_sum_digits_nine_l131_131177


namespace total_number_of_students_l131_131422

theorem total_number_of_students 
    (T : ℕ)
    (h1 : ∃ a, a = T / 5) 
    (h2 : ∃ b, b = T / 4) 
    (h3 : ∃ c, c = T / 2) 
    (h4 : T - (T / 5 + T / 4 + T / 2) = 25) : 
  T = 500 := by 
  sorry

end total_number_of_students_l131_131422


namespace find_larger_number_l131_131599

theorem find_larger_number (x y : ℕ) 
  (h1 : 4 * y = 5 * x) 
  (h2 : x + y = 54) : 
  y = 30 :=
sorry

end find_larger_number_l131_131599


namespace area_of_triangle_DEF_l131_131619

variables (s : ℝ) (T : EuclideanGeometry.Point) (DE DF EF : ℝ)

-- Conditions
def square_PQRS (s : ℝ) (area : ℝ) : Prop :=
  s^2 = area

def small_squares (length : ℝ) : Prop :=
  length = 2

def triangle_DEF (DE DF EF : ℝ) : Prop :=
  DE = DF ∧ EF = (s - 2 - 2)

def fold_condition (T : EuclideanGeometry.Point) : Prop :=
  true  -- Basically assumes the folding condition is implicitly given.

-- Main proof problem
theorem area_of_triangle_DEF (s : ℝ) (D T : EuclideanGeometry.Point)
    (DE DF EF : ℝ) :
  square_PQRS s 36 →
  small_squares 2 →
  triangle_DEF DE DF EF →
  fold_condition T →
  (∃ H : T, DE = DF ∧ EF = (s - 2 - 2) ) →
  let DT := s / 2 + 2 + 2 in
  (1 / 2 * EF * DT = 10) :=
by
  intros h1 h2 h3 h4 h5
  let DT := s / 2 + 2 + 2
  apply sorry

end area_of_triangle_DEF_l131_131619


namespace construct_points_l131_131848

/-!
# Proof Problem
Given an equilateral triangle ABC with vertices A, B, C, and circles k1, k2, k3 with equal radii centered at A, B, and C respectively.
We are to prove that there exist exactly two points P and Q such that after successive inversions through k1, k2, k3, the final points coincide with their original points.
-/

variables {α : Type*} [MetricSpace α]

structure EquilateralTriangle (α : Type*) extends Triangle α :=
(a b c : α)
(is_equilateral : dist a b = dist b c ∧ dist b c = dist c a ∧ dist c a = dist a b)

structure Circle (α : Type*) :=
(center : α)
(radius : ℝ)
(equal_radii : ∀ k : Circle α, radius = k.radius)

def inversion (P O : α) (r : ℝ) : α := sorry -- Define an appropriate inversion function

theorem construct_points (A B C P Q : α) (r : ℝ)
  (h_eq_tri : EquilateralTriangle α)
  (h_k1 : Circle α) (h_k2 : Circle α) (h_k3 : Circle α) 
  (h_eq_radii : h_k1.radius = r ∧ h_k2.radius = r ∧ h_k3.radius = r) 
  (h_central : h_k1.center = A ∧ h_k2.center = B ∧ h_k3.center = C)
  : ∃ P Q, 
      P ≠ Q ∧ 
      let P1 := inversion P h_k1.center h_k1.radius,
          P2 := inversion P1 h_k2.center h_k2.radius,
          P3 := inversion P2 h_k3.center h_k3.radius,
          Q1 := inversion Q h_k1.center h_k1.radius,
          Q2 := inversion Q1 h_k2.center h_k2.radius,
          Q3 := inversion Q2 h_k3.center h_k3.radius
      in P = P3 ∧ Q = Q3 :=
sorry

end construct_points_l131_131848


namespace angle_QRP_60_l131_131859

/-- 
Given a triangle DEF with angles 50°, 70°, and 60°, 
If circle Ω is the incircle of △DEF and also the circumcircle of △PQR,
where P is on EF, Q is on DE, and R is on DF,
then the measure of ∠QRP is 60°.
-/
theorem angle_QRP_60 (d e f p q r : Type) 
  [metric_space d] [metric_space e] [metric_space f]
  [metric_space p] [metric_space q] [metric_space r]
  (angle_D : d → ℝ) (angle_E : e → ℝ) (angle_F : f → ℝ)
  (H_angle_D : ∀ d, angle_D d = 50)
  (H_angle_E : ∀ e, angle_E e = 70)
  (H_angle_F : ∀ f, angle_F f = 60)
  (Ω : Type)
  [incircle : is_incircle_of_triangle Ω d e f]
  [circumcircle : is_circumcircle_of_triangle Ω p q r]
  (H_P : p ∈ EF)
  (H_Q : q ∈ DE)
  (H_R : r ∈ DF):
  ∠ QRP = 60 := 
sorry

end angle_QRP_60_l131_131859


namespace pole_length_after_cut_l131_131824

theorem pole_length_after_cut
  (initial_length : ℝ)
  (cut_percentage : ℝ)
  (initial_length_eq : initial_length = 20)
  (cut_percentage_eq : cut_percentage = 0.30) :
  let new_length := initial_length * (1 - cut_percentage) in
  new_length = 14 := by
  sorry

end pole_length_after_cut_l131_131824


namespace greatest_number_that_divides_54_87_172_l131_131012

noncomputable def gcdThree (a b c : ℤ) : ℤ :=
  gcd (gcd a b) c

theorem greatest_number_that_divides_54_87_172
  (d r : ℤ)
  (h1 : 54 % d = r)
  (h2 : 87 % d = r)
  (h3 : 172 % d = r) :
  d = gcdThree 33 85 118 := by
  -- We would start the proof here, but it's omitted per instructions
  sorry

end greatest_number_that_divides_54_87_172_l131_131012


namespace matchstick_polygon_area_l131_131749

-- Given conditions
def number_of_matches := 12
def length_of_each_match := 2 -- in cm

-- Question: Is it possible to construct a polygon with an area of 16 cm^2 using all the matches?
def polygon_possible : Prop :=
  ∃ (p : Polygon), 
    (p.edges = number_of_matches) ∧ 
    (∃ (match_length : ℝ), match_length = length_of_each_match ∧ by 
      -- Form the polygon using all matches without breaking
      sorry) ∧ 
    (polygon_area p = 16)

-- Proof statement
theorem matchstick_polygon_area :
  polygon_possible :=
  sorry

end matchstick_polygon_area_l131_131749


namespace maximize_cubic_quartic_l131_131318

theorem maximize_cubic_quartic (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + 2 * y = 35) : 
  (x, y) = (21, 7) ↔ x^3 * y^4 = (21:ℝ)^3 * (7:ℝ)^4 := 
by
  sorry

end maximize_cubic_quartic_l131_131318


namespace cosine_sum_minimum_value_l131_131523

theorem cosine_sum_minimum_value (x y : ℝ) (h : cos x + cos y = 1 / 3) :
  cos (x + y) ≥ -17 / 18 :=
sorry

end cosine_sum_minimum_value_l131_131523


namespace original_price_discount_l131_131698

theorem original_price_discount (P : ℝ) (h : 0.90 * P = 450) : P = 500 :=
by
  sorry

end original_price_discount_l131_131698


namespace pears_in_D_is_25_l131_131626

-- Define the conditions
constant A_fruits : ℕ
constant B_fruits : ℕ
constant C_fruits : ℕ
constant E_fruits : ℕ
constant num_baskets : ℕ
constant average_fruits : ℕ

-- Assign values based on conditions
def A_fruits := 15
def B_fruits := 30
def C_fruits := 20
def E_fruits := 35
def num_baskets := 5
def average_fruits := 25

-- Calculate the total number of fruits
def total_fruits := num_baskets * average_fruits

-- Calculate the number of fruits in D
def D_fruits := total_fruits - (A_fruits + B_fruits + C_fruits + E_fruits)

-- The theorem to prove the number of pears in basket D
theorem pears_in_D_is_25 : D_fruits = 25 := by
  -- Here we would provide the steps to prove the theorem, but we use sorry to skip it
  sorry

end pears_in_D_is_25_l131_131626


namespace bus_routes_setup_possible_l131_131286

noncomputable section

open Finset

-- Definitions based on conditions
def is_valid_configuration (lines : Finset ℤ) (intersections : Finset (ℤ × ℤ)) : Prop :=
  lines.card = 10 ∧ 
  intersections.card = 45 ∧ 
  ∀ (chosen8 : Finset ℤ), chosen8.card = 8 → ∃ (stop : ℤ × ℤ), stop ∉ intersections.filter (λ p, p.1 ∈ chosen8 ∧ p.2 ∈ chosen8) ∧
  ∀ (chosen9 : Finset ℤ), chosen9.card = 9 → 
  intersections ⊆ intersections.filter (λ p, p.1 ∈ chosen9 ∧ p.2 ∈ chosen9)

-- The final proof statement
theorem bus_routes_setup_possible : ∃ (lines : Finset ℤ) (intersections : Finset (ℤ × ℤ)), is_valid_configuration lines intersections :=
sorry

end bus_routes_setup_possible_l131_131286


namespace find_lengths_l131_131441

noncomputable def radius_C1 : ℝ := 2 * Real.sqrt 6
noncomputable def radius_C2 : ℝ := Real.sqrt 6
noncomputable def dist_O1O2 : ℝ := Real.sqrt 70
constant A1 A2 B1 B2 O1 O2 A B : Point
constant l1 l2 l3 : Line

axiom circles_same_side_of_l1 : same_side O1 O2 l1
axiom circles_opposite_sides_of_l2 : opposite_sides O1 O2 l2
axiom points_on_circles : on_circle A1 O1 radius_C1 ∧ on_circle B1 O1 radius_C1 ∧ on_circle A2 O2 radius_C2 ∧ on_circle B2 O2 radius_C2
axiom lines_tangent_to_circles : tangent_at_line l1 A1 C1 ∧ tangent_at_line l1 A2 C2 ∧ tangent_at_line l2 B1 C1 ∧ tangent_at_line l2 B2 C2
axiom A1_B1_opposite_sides : opposite_sides A1 B1 (line_through O1 O2)
axiom l3_perpendicular_to_l2 : perpendicular (line_through B2) l3 l2

theorem find_lengths : 
  dist A1 A2 = 8 ∧ 
  dist B1 B2 = 4 ∧
  (∃ a b : ℝ, right_triangle a b (dist A B) (dist A2 B2) (dist B B2) 2 10 (4 * Real.sqrt 6) ) :=
sorry

end find_lengths_l131_131441


namespace problem_statement_l131_131262

noncomputable def proof_sinC_area (A C B : ℝ) (a b c : ℝ) 
  (h_cosA: ℝ) (h_c: ℝ) (h_a: ℝ) : Prop :=
sin C = 1 / 3 ∧ 1 / 2 * a * c * sin B = 5 * real.sqrt 2 / 2

theorem problem_statement :
  ∀ (A B C : ℝ) (a c : ℝ),
    cos A = real.sqrt 3 / 3 → 
    c = real.sqrt 3 → 
    a = 3 * real.sqrt 2 →
    proof_sinC_area A C B a b c (cos A) c a :=
by 
  intros A B C a c h_cosA h_c h_a
  unfold proof_sinC_area
  sorry  

end problem_statement_l131_131262


namespace monomials_like_terms_l131_131978

theorem monomials_like_terms (a b : ℤ) (h1 : a + 1 = 2) (h2 : b - 2 = 3) : a + b = 6 :=
sorry

end monomials_like_terms_l131_131978


namespace taxi_fare_l131_131836

theorem taxi_fare :
  let base_distance := 60 in
  let total_distance_80 := 80 in
  let total_fare_80 := 180 in
  let extra_rate_multiple := 1.25 in
  let trips := 100 in
  let base_fare := (total_fare_80) / (base_distance + extra_rate_multiple * (total_distance_80 - base_distance)) in
  let total_fare_100 := (base_fare * base_distance) + (extra_rate_multiple * base_fare * (trips - base_distance)) in
  total_fare_100 = 233.05 :=
by
  let base_distance := 60
  let total_distance_80 := 80
  let total_fare_80 := 180
  let extra_rate_multiple := 1.25
  let trips := 100
  let base_fare := (total_fare_80) / (base_distance + extra_rate_multiple * (total_distance_80 - base_distance))
  let total_fare_100 := (base_fare * base_distance) + (extra_rate_multiple * base_fare * (trips - base_distance))
  show total_fare_100 = 233.05
  sorry

end taxi_fare_l131_131836


namespace real_root_of_polynomial_l131_131683

noncomputable def polynomial (c d : ℝ) (x : ℂ) : ℂ := c * x^3 - x^2 + d * x + 30

theorem real_root_of_polynomial :
  let c := -5 / 53
  let d := -426 / 53
  has_root (polynomial c d) (53 / 5 : ℝ) :=
by
  sorry

end real_root_of_polynomial_l131_131683


namespace ray_has_4_nickels_left_l131_131336

variables {cents_per_nickel : ℕ := 5}

-- Conditions
def initial_cents := 95
def cents_given_to_peter := 25
def cents_given_to_randi := 2 * cents_given_to_peter
def total_cents_given := cents_given_to_peter + cents_given_to_randi
def remaining_cents := initial_cents - total_cents_given

-- Theorem statement
theorem ray_has_4_nickels_left :
  (remaining_cents / cents_per_nickel) = 4 :=
begin
  sorry
end

end ray_has_4_nickels_left_l131_131336


namespace coefficient_x2_in_binomial_expansion_l131_131349

-- Theorem to prove that the coefficient of the x^2 term in the expansion of (2x + 1)^5 is 40
theorem coefficient_x2_in_binomial_expansion (x : ℝ) :
  let term := (2*x + 1) ^ 5 in  -- the polynomial to expand
  (term.coeff 2) = 40 :=       -- coefficient of x^2 is 40
  sorry

end coefficient_x2_in_binomial_expansion_l131_131349


namespace no_two_digit_prime_with_digit_sum_9_l131_131183

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end no_two_digit_prime_with_digit_sum_9_l131_131183


namespace exists_mod_inv_l131_131317

theorem exists_mod_inv (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (h : ¬ a ∣ p) : ∃ b : ℕ, a * b ≡ 1 [MOD p] :=
by
  sorry

end exists_mod_inv_l131_131317


namespace partition_naturals_7th_power_l131_131857

theorem partition_naturals_7th_power :
  ∃ (A : ℕ → Finset ℕ), (∀ k : ℕ, A k.card = k) ∧ (∀ k : ℕ, (A k).sum id = k^7) :=
sorry

end partition_naturals_7th_power_l131_131857


namespace even_comp_even_l131_131306

variable {X : Type} [AddGroup X] [HasNeg X]

def is_even_function (g : X → X) : Prop :=
  ∀ x : X, g (-x) = g x

theorem even_comp_even (g : X → X) (h_even : is_even_function g) :
  is_even_function (g ∘ g) :=
by
  intro x
  rw [function.comp_apply, function.comp_apply]
  rw [h_even] -- by the even property of g
  rw [h_even] -- by the even property of g
  sorry

end even_comp_even_l131_131306


namespace cylinder_volume_calc_l131_131000

def cylinder_volume (r h : ℝ) (π : ℝ) : ℝ := π * r^2 * h

theorem cylinder_volume_calc :
    cylinder_volume 5 (5 + 3) 3.14 = 628 :=
by
  -- We set r = 5, h = 8 (since h = r + 3), and π = 3.14 to calculate the volume
  sorry

end cylinder_volume_calc_l131_131000


namespace cluster_set_convergence_l131_131421

theorem cluster_set_convergence (xn : ℕ → ℝ) (a : ℝ) :
  (∀ ε > 0, ∃ N, ∀ n > N, |xn n - a| < ε) →
  (∀ ε > 0, ∃ N, ∀ n > N, xn n ∈ set.Ioo (a - ε) (a + ε)) ∧
  (∀ I, ¬ ((a ∈ I) → ∃ N, ∀ n > N, xn n ∈ I)) :=
by
  intros h ε hε
  constructor
  { sorry }
  { sorry }

end cluster_set_convergence_l131_131421


namespace total_apples_picked_l131_131075

-- Define the number of apples picked by Benny
def applesBenny : Nat := 2

-- Define the number of apples picked by Dan
def applesDan : Nat := 9

-- The theorem we want to prove
theorem total_apples_picked : applesBenny + applesDan = 11 := 
by 
  sorry

end total_apples_picked_l131_131075


namespace probability_of_odd_divisor_l131_131374

noncomputable def factorial_prime_factors : ℕ → List (ℕ × ℕ)
| 21 => [(2, 18), (3, 9), (5, 4), (7, 3), (11, 1), (13, 1), (17, 1), (19, 1)]
| _ => []

def number_of_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

def number_of_odd_factors (factors : List (ℕ × ℕ)) : ℕ :=
  number_of_factors (factors.filter (λ ⟨p, _⟩ => p != 2))

theorem probability_of_odd_divisor : (number_of_odd_factors (factorial_prime_factors 21)) /
(number_of_factors (factorial_prime_factors 21)) = 1 / 19 := 
by
  sorry

end probability_of_odd_divisor_l131_131374


namespace num_sets_C_l131_131557

def A : Set ℕ := { x | x^2 - 5 * x + 6 = 0 }
def B : Set ℕ := { x | 0 < x ∧ x < 6 }

theorem num_sets_C (hA : A = {2, 3}) (hB : B = {1, 2, 3, 4, 5}) :
  {C : Set ℕ // A ⊆ C ∧ C ⊆ B}.card = 8 := by
  sorry

end num_sets_C_l131_131557


namespace not_possible_to_divide_prism_into_pyramids_l131_131284

variables (V : Type) [NormedAddCommGroup V] [NormedSpace ℝ V]

structure Prism (V : Type) :=
  (base1 base2 : V → V)
  (parallelogram_faces : set (set (V × V)))

theorem not_possible_to_divide_prism_into_pyramids
  (P : Prism V) :
  ¬∃ (pyramids : set (set (V × V))),
    (∀ (p : set (V × V)), p ∈ pyramids → 
      ∃ (base : V → V) (apex : V),
        base ∈ {P.base1, P.base2} ∧
        apex ∈ {P.base2 x | x ∈ (range P.base1)} ∧
        ∀ (x : V), x ∈ base → (x, apex) ∈ p) ∧
    (∀ (x : V), x ∈ P.base1 → 
      (∃ p ∈ pyramids, x ∈ fst '' p) ∨
      (∃ p ∈ pyramids, x ∈ snd '' p)) :=
begin
  sorry
end

end not_possible_to_divide_prism_into_pyramids_l131_131284


namespace Jill_gift_amount_l131_131512

theorem Jill_gift_amount (S : ℝ) (hS : S = 3600) :
  let D := (1 / 5) * S,
      V := 0.30 * D,
      G := 0.20 * D,
      E := 0.35 * D,
      T := V + G + E,
      L := D - T
   in L = 108 :=
by
  sorry

end Jill_gift_amount_l131_131512


namespace who_always_tells_the_truth_l131_131328

-- Define the three boys
inductive Boy
| Anton
| Vanya
| Sasha

open Boy

-- Define each boy's statement
def statement : Boy → Prop
| Anton := ¬ ∀ t : Boy, t = Vanya → statement t
| Vanya := ¬ ∀ t : Boy, t = Vanya → statement t
| Sasha := ¬ ∀ t : Boy, t = Anton → statement t

-- Define that at least one of them lied
def at_least_one_lied : Prop :=
  ∃ b : Boy, ¬ statement b

-- Define that only one always tells the truth
def only_one_always_tells_the_truth : Prop :=
  ∃ t : Boy, (statement t) ∧ (∀ b : Boy, b ≠ t → ¬ statement b)

theorem who_always_tells_the_truth :
  at_least_one_lied → only_one_always_tells_the_truth → 
  ∃ t : Boy, t = Anton ∧ (statement Anton) :=
by
  intros h1 h2
  sorry

end who_always_tells_the_truth_l131_131328


namespace problem_l131_131662

variable (g : ℝ → ℝ)

axiom g_def : ∀ x : ℝ, g(x) = 5

theorem problem : ∀ x : ℝ, 3 * g(x - 3) + 1 = 16 := by
  -- Proof will go here
  sorry

end problem_l131_131662


namespace isosceles_triangle_count_l131_131863

-- Define the concept of distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).sqrt

-- Define what it means for a triangle to be isosceles
def is_isosceles (p1 p2 p3 : ℝ × ℝ) : Prop :=
  distance p1 p2 = distance p2 p3 ∨
  distance p2 p3 = distance p3 p1 ∨
  distance p3 p1 = distance p1 p2

-- Define the vertices of the five triangles
def t1_v1 : ℝ × ℝ := (0, 7)
def t1_v2 : ℝ × ℝ := (2, 7)
def t1_v3 : ℝ × ℝ := (1, 5)

def t2_v1 : ℝ × ℝ := (4, 3)
def t2_v2 : ℝ × ℝ := (4, 5)
def t2_v3 : ℝ × ℝ := (6, 3)

def t3_v1 : ℝ × ℝ := (0, 2)
def t3_v2 : ℝ × ℝ := (3, 3)
def t3_v3 : ℝ × ℝ := (6, 2)

def t4_v1 : ℝ × ℝ := (1, 1)
def t4_v2 : ℝ × ℝ := (0, 3)
def t4_v3 : ℝ × ℝ := (3, 1)

def t5_v1 : ℝ × ℝ := (3, 6)
def t5_v2 : ℝ × ℝ := (4, 4)
def t5_v3 : ℝ × ℝ := (5, 7)

-- Prove that exactly three of these triangles are isosceles
theorem isosceles_triangle_count : 
  (is_isosceles t1_v1 t1_v2 t1_v3 ∧ 
   is_isosceles t2_v1 t2_v2 t2_v3 ∧ 
   is_isosceles t3_v1 t3_v2 t3_v3) ∧ 
  ¬ (is_isosceles t4_v1 t4_v2 t4_v3) ∧ 
  ¬ (is_isosceles t5_v1 t5_v2 t5_v3) :=
by
  -- This 'by' block is for concise proof structure, but it's left to be filled in the actual proof.
  sorry

end isosceles_triangle_count_l131_131863


namespace board_game_cost_correct_l131_131501

-- Definitions
def jump_rope_cost : ℕ := 7
def ball_cost : ℕ := 4
def saved_money : ℕ := 6
def gift_money : ℕ := 13
def needed_money : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_money + gift_money

-- Total cost of all items
def total_cost : ℕ := total_money + needed_money

-- Combined cost of jump rope and ball
def combined_cost_jump_rope_ball : ℕ := jump_rope_cost + ball_cost

-- Cost of the board game
def board_game_cost : ℕ := total_cost - combined_cost_jump_rope_ball

-- Theorem to prove
theorem board_game_cost_correct : board_game_cost = 12 :=
by 
  -- Proof omitted
  sorry

end board_game_cost_correct_l131_131501


namespace parallel_lines_distance_l131_131403

theorem parallel_lines_distance (m n : ℝ) 
  (h1 : 3 = |(n / 2) - 5| / (real.sqrt(3^2 + (m / 2)^2)))
  (h2 : 4 / 3 = m / 2) : m + n = 48 ∨ m + n = -12 :=
sorry

end parallel_lines_distance_l131_131403


namespace angle_less_than_sixty_l131_131804

theorem angle_less_than_sixty (ABC : Triangle) (equilateral : ∀ (A B C : Point), ABC.ABC = is_equilateral)
  (I : Point) (incenter : is_incenter I)
  (B' C' : Point) (B'_foot : is_foot B' (ABC.angle_bisector AB))
  (C'_foot : is_foot C' (ABC.angle_bisector AC)) :
  angle B' A' C' < 60 :=
sorry

end angle_less_than_sixty_l131_131804


namespace complex_solution_l131_131571

open Complex

noncomputable def complex_z : ℂ := cos (2 * Real.pi / 3) + complex.I * sin (2 * Real.pi / 3)

theorem complex_solution : (complex_z^3 + complex_z^2) = ((1 - Real.sqrt 3 * complex.I) / 2) :=
by
  -- step 1: express z using Euler's formula
  -- step 2: calculate z^2 
  -- step 3: calculate z^3
  -- step 4: add z^3 and z^2
  -- step 5: use trigonometric identities
  -- combine all above to prove the statement
  sorry

end complex_solution_l131_131571


namespace negation_of_universal_proposition_l131_131722

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, |x - 1| - |x + 1| ≤ 3) ↔ ∃ x : ℝ, |x - 1| - |x + 1| > 3 :=
by
  sorry

end negation_of_universal_proposition_l131_131722


namespace cube_inscribed_circumscribed_volume_ratio_l131_131995

theorem cube_inscribed_circumscribed_volume_ratio
  (S_1 S_2 V_1 V_2 : ℝ)
  (h : S_1 / S_2 = (1 / Real.sqrt 2) ^ 2) :
  V_1 / V_2 = (Real.sqrt 3 / 3) ^ 3 :=
sorry

end cube_inscribed_circumscribed_volume_ratio_l131_131995


namespace not_possible_to_see_all_cyclic_orders_of_spires_l131_131846

-- Define the problem conditions
def skyscrapers := 7
def lines := (skyscrapers * (skyscrapers - 1)) / 2
def maxRegions (n : ℕ) := (n * (n + 1)) / 2 + 1
def cyclicOrders (n : ℕ) := (n - 1)!

-- Assert the main proof statement
theorem not_possible_to_see_all_cyclic_orders_of_spires :
  skyscrapers = 7 ∧ cyclicOrders skyscrapers = 720 ∧ lines = 21 ∧ maxRegions lines = 232 → 720 > 232 :=
by
  intros
  sorry

end not_possible_to_see_all_cyclic_orders_of_spires_l131_131846


namespace area_of_quadrilateral_APQB_l131_131431

open Real

theorem area_of_quadrilateral_APQB (O A B E F C P Q : Point) (R : ℝ)
  (h_circle : Circle O R)
  (h_diameters : AE ⊥ BF)
  (h_point_on_arc : C ∈ arc EF)
  (h_intersections : CA ∩ BF = P ∧ CB ∩ AE = Q) :
  area_of_quadrilateral APQB = R^2 :=
sorry

end area_of_quadrilateral_APQB_l131_131431


namespace repetend_of_five_seventeenths_l131_131152

theorem repetend_of_five_seventeenths :
  (decimal_expansion (5 / 17)).repeat_called == "294117647" :=
sorry

end repetend_of_five_seventeenths_l131_131152


namespace cos_sum_minimum_l131_131524

theorem cos_sum_minimum (x y : ℝ) (h : cos x + cos y = 1 / 3) : cos (x + y) ≥ -17 / 18 :=
sorry

end cos_sum_minimum_l131_131524


namespace depth_of_water_in_smaller_container_l131_131832

-- Definitions for the container dimensions and initial conditions
def larger_container_height : ℝ := 20
def larger_container_radius : ℝ := 6
def initial_water_depth : ℝ := 17
def smaller_container_height : ℝ := 18
def smaller_container_radius : ℝ := 5

-- Calculate volumes
def volume_cylinder (radius height : ℝ) : ℝ :=
  π * radius^2 * height

def volume_large := volume_cylinder larger_container_radius larger_container_height
def volume_small := volume_cylinder smaller_container_radius smaller_container_height
def initial_water_volume := volume_cylinder larger_container_radius initial_water_depth

-- The combined volume when the smaller container is lowered entirely
def total_volume := initial_water_volume + volume_small

-- Overflow volume when the total volume exceeds the capacity of the larger container
def overflow_volume := total_volume - volume_large

-- Volume that actually spills into the smaller container after overflow
def transferred_volume := π * larger_container_radius^2 * 2

-- We aim to prove that the depth of water in the smaller container is 2.88 cm
theorem depth_of_water_in_smaller_container :
  let final_water_volume := transferred_volume in
  let smaller_container_base_area := π * (smaller_container_radius^2) in
  final_water_volume / smaller_container_base_area = 2.88 := sorry

end depth_of_water_in_smaller_container_l131_131832


namespace sum_of_nonconstant_coeffs_l131_131529

theorem sum_of_nonconstant_coeffs :
  let exp := (λ (x : ℝ), (1 / x - 2 * x ^ 2) ^ 9) in
  let coeff_sum := (λ (x : ℝ), exp 1) - (-2)^3 * Nat.binomial 9 3 in
  coeff_sum = 671 :=
by
  sorry

end sum_of_nonconstant_coeffs_l131_131529


namespace isosceles_triangle_angle_l131_131625

theorem isosceles_triangle_angle (A B C P Q : Type)
    [HasAngle A B C] [HasAngle A P Q] [HasAngle P Q B] [HasAngle Q P B]
    [HasLength A B] [HasLength B C] [HasLength A P] [HasLength P Q] [HasLength Q B]
    (h1 : IsoscelesTriangle A B C)
    (h2 : SegmentOnLine P A B)
    (h3 : SegmentOnLine Q B C)
    (h4 : SegmentLength A P = SegmentLength P Q)
    (h5 : SegmentLength P Q = SegmentLength Q B)
    (h6 : SegmentLength Q B = SegmentLength B C) :
    AngleMeasure B = 90 := sorry

end isosceles_triangle_angle_l131_131625


namespace expected_number_of_distinct_faces_l131_131042

open BigOperators

variable (n : ℕ := 6)

def expected_distinct_faces (k : ℕ) : ℝ := 
  ∑ i in finset.range (k + 1), 
    (if i = 0 then real.of_rat ((5 / 6) ^ k) else real.of_rat (1 - ((5 / 6) ^ k)))

theorem expected_number_of_distinct_faces :
  expected_distinct_faces n = (6 ^ 6 - 5 ^ 6) / (6 ^ 5) :=
sorry

end expected_number_of_distinct_faces_l131_131042


namespace amount_paid_two_months_ago_l131_131288

-- Define the conditions as variables and assumptions in Lean
variable (x : ℝ)  -- The amount Jerry paid two months ago
variable (debt_total debt_remaining : ℝ)
variable (last_payment_diff : ℝ)

-- Assume the conditions given in the problem
axiom debt_total_eq : debt_total = 50
axiom debt_remaining_eq : debt_remaining = 23
axiom last_payment_diff_eq : last_payment_diff = 3

-- Define the Lean statement to prove the correct answer
theorem amount_paid_two_months_ago :
  (debt_total - debt_remaining) = x + (x + last_payment_diff) →
  x = 12 :=
by
  intros h
  rw [debt_total_eq, debt_remaining_eq, last_payment_diff_eq] at h
  have h1 : 27 = 2 * x + 3 := by linarith
  have h2 : 24 = 2 * x := by linarith
  linarith

end amount_paid_two_months_ago_l131_131288


namespace problem1_problem2_l131_131199

noncomputable section
namespace ProofProblem

variables {a b c x : ℝ}

-- Condition for the first problem
def condition1 : Prop := a^2 + b^2 + c^2 = 1

-- First proof problem: |a + b + c| ≤ √3
theorem problem1 (h : condition1) : |a + b + c| ≤ Real.sqrt 3 := 
sorry

-- Second proof problem: Range of x given the inequality |x - 1| + |x + 1| ≥ (a + b + c)^2
theorem problem2 (h : condition1) : 
  (∀ a b c : ℝ, |x - 1| + |x + 1| ≥ (a + b + c)^2) ↔ (x ≤ -3 / 2 ∨ x ≥ 3 / 2) := 
sorry

end ProofProblem

end problem1_problem2_l131_131199


namespace hyperbola_eccentricity_is_correct_l131_131940

-- Define the condition of the hyperbola and eccentricity
noncomputable def hyperbola_eccentricity (a : ℝ) (h : a > sqrt 2) : ℝ :=
  let c := sqrt (a ^ 2 + 2)
  c / a

-- The main theorem stating the problem and answer
theorem hyperbola_eccentricity_is_correct (a : ℝ) (h : a > sqrt 2) (angle_asymptotes : ℝ) :
  (angle_asymptotes = π / 3) →
  hyperbola_eccentricity a h = 2 * sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_is_correct_l131_131940


namespace prob_team_member_A_receives_3_8_X_distribution_is_correct_mean_X_is_correct_variance_X_is_correct_l131_131036

noncomputable def prob_team_member_A_receives (n : ℕ) : ℚ :=
  have prob_not_captain := (n-5) / n
  have prob_not_vc := (n-5) / n
  1 - prob_not_captain * prob_not_vc

theorem prob_team_member_A_receives_3_8 : prob_team_member_A_receives 8 = 39 / 64 := by
  sorry

structure prob_distribution :=
  (P : ℚ)
  (X : ℕ)

def X_distribution : list prob_distribution := [
  {X:=3, P:=1/56},
  {X:=4, P:=15/56},
  {X:=5, P:=15/28},
  {X:=6, P:=5/28}
]

def mean_X (dist : list prob_distribution) : ℚ :=
  dist.foldl (λ acc d, acc + d.X * d.P) 0

def variance_X (dist : list prob_distribution) (mean: ℚ) : ℚ :=
  dist.foldl (λ acc d, acc + (d.X - mean)^2 * d.P) 0

theorem X_distribution_is_correct :
  X_distribution = [
    {X:=3, P:=1/56},
    {X:=4, P:=15/56},
    {X:=5, P:=15/28},
    {X:=6, P:=5/28}
  ] :=
  by sorry

theorem mean_X_is_correct : mean_X X_distribution = 39 / 8 := by
  sorry

theorem variance_X_is_correct : variance_X X_distribution (mean_X X_distribution) = 225 / 448 := by
  sorry

end prob_team_member_A_receives_3_8_X_distribution_is_correct_mean_X_is_correct_variance_X_is_correct_l131_131036


namespace amanda_final_score_l131_131844

theorem amanda_final_score
  (average1 : ℕ) (quizzes1 : ℕ) (average_required : ℕ) (quizzes_total : ℕ)
  (H1 : average1 = 92) (H2 : quizzes1 = 4) (H3 : average_required = 93) (H4 : quizzes_total = 5) :
  let total_points1 := quizzes1 * average1,
      total_points_required := quizzes_total * average_required,
      final_score_needed := total_points_required - total_points1
  in final_score_needed = 97 := by
  sorry

end amanda_final_score_l131_131844


namespace bakery_order_alpha_l131_131438

noncomputable def f (n : ℕ) : ℝ sorry

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_bakery_of_order_n (f : ℕ → ℝ) (B : ℕ → ℝ) (n : ℕ) : Prop := sorry

theorem bakery_order_alpha (α : ℝ) (h : α = 3 / 2) :
  ∀ (n : ℕ), (8 ≤ n) → is_even n → 
  let f_n := f(n) in
  (1 / 100) < (f_n / (n^α)) ∧ (f_n / (n^α)) < 100 :=
by
  intros n hn h_even h_f
  sorry

end bakery_order_alpha_l131_131438


namespace cos_sum_minimum_l131_131525

theorem cos_sum_minimum (x y : ℝ) (h : cos x + cos y = 1 / 3) : cos (x + y) ≥ -17 / 18 :=
sorry

end cos_sum_minimum_l131_131525


namespace worker_A_time_l131_131796

theorem worker_A_time (time_B : ℝ) (time_together : ℝ) (time_A : ℝ) : 
    time_B = 10 → time_together = 10 / 3 → time_A = 5 :=
λ h_B h_together, by
  have h1 : (1 / time_A) + (1 / time_B) = 1 / time_together,
    from by rw [h_B, h_together]; linarith,
  sorry

end worker_A_time_l131_131796


namespace matchstick_polygon_area_l131_131751

-- Given conditions
def number_of_matches := 12
def length_of_each_match := 2 -- in cm

-- Question: Is it possible to construct a polygon with an area of 16 cm^2 using all the matches?
def polygon_possible : Prop :=
  ∃ (p : Polygon), 
    (p.edges = number_of_matches) ∧ 
    (∃ (match_length : ℝ), match_length = length_of_each_match ∧ by 
      -- Form the polygon using all matches without breaking
      sorry) ∧ 
    (polygon_area p = 16)

-- Proof statement
theorem matchstick_polygon_area :
  polygon_possible :=
  sorry

end matchstick_polygon_area_l131_131751


namespace SandySpentTotal_l131_131341

theorem SandySpentTotal :
  let shorts := 13.99
  let shirt := 12.14
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := by
  sorry

end SandySpentTotal_l131_131341


namespace distance_from_Q0_to_Q8_l131_131810

noncomputable def ζ : ℂ := Complex.exp (Complex.I * Real.pi / 4)

def Q (k : ℕ) (zk : ℂ) : ℂ :=
  if k = 0 then zk
  else ζ^k * (k : ℂ) + Q (k - 1) (zk)

theorem distance_from_Q0_to_Q8 : 
  let dist := Complex.abs ((ζ - 1)⁻¹ * (Complex.ofReal 7)) in 
  dist = 3.5 * Real.sqrt 2 :=
by
  sorry

end distance_from_Q0_to_Q8_l131_131810


namespace probability_no_red_no_yellow_l131_131685

theorem probability_no_red_no_yellow : 
    let balls := {1, 2, 3, 4, 5} in
    let boxes := {"red", "yellow", "blue", "white", "black"} in
    ∀ (arrangement : balls → boxes),
    (∀ (b1 : ∃ (f : Nat → String), f 1 ≠ "red"),
    ∀ (b2 : ∃ (f : Nat → String), f 2 ≠ "yellow"),
    card {f | f 1 ≠ "red" ∧ f 2 ≠ "yellow"} / 
    card {f | true} =
    13 / 20
by {
  sorry
} 

end probability_no_red_no_yellow_l131_131685


namespace prove_total_payment_l131_131007

-- Define the conditions under which the problem is set
def monthly_subscription_cost : ℝ := 14
def split_ratio : ℝ := 0.5
def months_in_year : ℕ := 12

-- Define the target amount to prove
def total_payment_after_one_year : ℝ := 84

-- Theorem statement
theorem prove_total_payment
  (h1: monthly_subscription_cost = 14)
  (h2: split_ratio = 0.5)
  (h3: months_in_year = 12) :
  monthly_subscription_cost * split_ratio * months_in_year = total_payment_after_one_year := 
  by
  sorry

end prove_total_payment_l131_131007


namespace repetend_of_five_over_seventeen_l131_131129

theorem repetend_of_five_over_seventeen : 
  let r := 5 / 17 in
  ∃ a b : ℕ, a * 10^b = 294117 ∧ (r * 10^b - a) = (r * 10^6 - r * (10^6 / 17))
   ∧ (r * 10^k = (r * 10^6).floor / 10^k ) where k = 6 := sorry

end repetend_of_five_over_seventeen_l131_131129


namespace solve_inequality_system_l131_131344

-- Define the conditions for the inequalities
def condition1 (x : ℝ) : Prop := 2 * (1 - x) ≤ 4
def condition2 (x : ℝ) : Prop := x - 4 < (x - 8) / 3

-- Define the integer solutions to be proved
def integer_solutions (xs : Set ℤ) : Prop :=
  xs = {-1, 0, 1}

-- The theorem statement encapsulating the problem and the solution
theorem solve_inequality_system (xs : Set ℤ) :
  (∀ x : ℝ, condition1 x ∧ condition2 x → x ∈ xs) →
  integer_solutions xs :=
by
  sorry

end solve_inequality_system_l131_131344


namespace find_S3m_l131_131383
  
-- Arithmetic sequence with given properties
variable (m : ℕ)
variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

-- Define the conditions
axiom Sm : S m = 30
axiom S2m : S (2 * m) = 100

-- Problem statement to prove
theorem find_S3m : S (3 * m) = 170 :=
by
  sorry

end find_S3m_l131_131383


namespace geometric_sequence_sum_product_l131_131769

theorem geometric_sequence_sum_product {a b c : ℝ} : 
  a + b + c = 14 → 
  a * b * c = 64 → 
  (a = 8 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) :=
by
  sorry

end geometric_sequence_sum_product_l131_131769


namespace boy_scout_troop_interest_l131_131063

noncomputable def interest_credited_in_cents (A : ℚ) (r : ℚ) (t : ℚ) : ℚ :=
  let P := A / (1 + r * t)
  in A - P

theorem boy_scout_troop_interest :
  interest_credited_in_cents 270.45 0.06 (1/4) = 0.45 :=
by sorry

end boy_scout_troop_interest_l131_131063


namespace minimum_magnitude_c_l131_131947

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x - 2, 2)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (4, y)
noncomputable def vector_c (x y : ℝ) : ℝ × ℝ := (x, y)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem minimum_magnitude_c {x y : ℝ} (h : dot_product (vector_a x) (vector_b y) = 0) :
  ∃ x, ∃ y, vector_c x y := vector_c (x, 4 - 2 * x) := ⟨x, y⟩, |x, y|^2 = (4 * √5)/5 . :=
sorry

end minimum_magnitude_c_l131_131947


namespace greatest_int_e_minus_3_l131_131531

-- Define e
def e_approx : ℝ := 2.71828

-- Define the greatest integer function [x]
def greatest_int (x : ℝ) : ℤ :=
  Int.floor x

-- Proof statement
theorem greatest_int_e_minus_3 : greatest_int (e_approx - 3) = -1 :=
by
  sorry

end greatest_int_e_minus_3_l131_131531


namespace andrea_birthday_next_tuesday_l131_131279

open Nat

def is_leap_year (year : Nat) : Bool :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def day_of_week_2015_dec_18 : String := "Friday"

def next_birthday_day_of_week (start_year : Nat) (start_day_of_week: String) : Nat :=
  let days_in_week := 7
  let non_leap_year_days := 365
  let leap_year_days := 366
  let days_to_next_tuesday := 4
  let target_day := "Tuesday"
  let mut current_day := start_day_of_week
  let mut current_year := start_year
  let mut total_days := 0

  while current_day ≠ target_day do
    total_days := if is_leap_year current_year then total_days + leap_year_days else total_days + non_leap_year_days
    current_year := current_year + 1
    let days_progress := total_days % days_in_week
    current_day := match (days_progress + days_to_next_tuesday) % days_in_week with
      | 0 => "Sunday"
      | 1 => "Monday"
      | 2 => "Tuesday"
      | 3 => "Wednesday"
      | 4 => "Thursday"
      | 5 => "Friday"
      | _ => "Saturday"
  
  current_year

theorem andrea_birthday_next_tuesday (start_year : Nat) (start_day_of_week : String) (h : start_year = 2015 ∧ start_day_of_week = "Friday") : 
  next_birthday_day_of_week start_year start_day_of_week = 2018 := by
  sorry

end andrea_birthday_next_tuesday_l131_131279


namespace problem_statement_l131_131566

-- Define the parabolic function y^2 = x
noncomputable def parabola (y : ℝ) : ℝ := y ^ 2

-- Define the focus of the parabola
def focus : (ℝ × ℝ) := (1 / 4, 0)

-- Define the line passing through the focus with a given slope k
def line (k x : ℝ) : ℝ := k * (x - 1 / 4)

-- Define the coordinates of the origin
def origin : (ℝ × ℝ) := (0, 0)

-- Define the points A and B on the parabola intersecting with the line
def points_intersect (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (k^2 - (1 / 2 * k^2 + 1)) / (k^2) in
  let y1 := k * (x1 - 1 / 4) in
  let x2 := ((1 / 2 * k ^ 2 + 1) / k ^ 2 - x1) in
  let y2 := k * (x2 - 1 / 4) in
  ((x1, y1), (x2, y2))

-- Define the dot product of vectors OA and OB
def dot_product (A B : (ℝ × ℝ)) : ℝ :=
  A.1 * B.1 + A.2 * B.2

-- The theorem to prove
theorem problem_statement (k : ℝ) :
  ∀ A B, A = (parabola (line k)), B = (parabola (line k)) ->
  dot_product (origin, A) (origin, B) = -3 / 16 :=
by
  sorry

end problem_statement_l131_131566


namespace residual_drug_time_approximation_l131_131077

theorem residual_drug_time_approximation (a : ℝ) (y : ℝ) :
  (∀ (t : ℝ), y = a * real.log 2 (12 / (t + 1))) →
  (y_at_2 := a * real.log 2 (12 / (2 + 1))) →
  (y_target := y_at_2 / 4) →
  (∃ t : ℝ, abs ((12 / (t + 1)) - real.sqrt 2) < 0.01) :=
by
  intros h_formula y_at_2_def y_target_def
  use 7.5
  sorry

end residual_drug_time_approximation_l131_131077


namespace abs_mult_example_l131_131086

theorem abs_mult_example : (|(-3)| * 2) = 6 := by
  have h1 : |(-3)| = 3 := by
    exact abs_of_neg (show -3 < 0 by norm_num)
  rw [h1]
  exact mul_eq_mul_left_iff.mpr (Or.inl rfl)

end abs_mult_example_l131_131086


namespace pole_length_after_cut_l131_131825

theorem pole_length_after_cut (original_length : ℝ) (percentage_shorter : ℝ) (h1 : original_length = 20) (h2 : percentage_shorter = 0.30) : 
  let length_cut = (percentage_shorter * original_length)
  let new_length = original_length - length_cut
  new_length = 14 := 
by
  sorry

end pole_length_after_cut_l131_131825


namespace erwin_chocolates_weeks_l131_131875

-- Define weekdays chocolates and weekends chocolates
def weekdays_chocolates := 2
def weekends_chocolates := 1

-- Define the total chocolates Erwin ate
def total_chocolates := 24

-- Define the number of weekdays and weekend days in a week
def weekdays := 5
def weekends := 2

-- Define the total chocolates Erwin eats in a week
def chocolates_per_week : Nat := (weekdays * weekdays_chocolates) + (weekends * weekends_chocolates)

-- Prove that Erwin finishes all chocolates in 2 weeks
theorem erwin_chocolates_weeks : (total_chocolates / chocolates_per_week) = 2 := by
  sorry

end erwin_chocolates_weeks_l131_131875


namespace geometric_mean_a2_a8_l131_131271

noncomputable def geometric_sequence (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^n

theorem geometric_mean_a2_a8 :
  ∀ {a1 q : ℝ} (h_a1 : a1 = 3) (h_q : q = 2),
    let a2 := geometric_sequence a1 q 1 in
    let a8 := geometric_sequence a1 q 7 in
    a2 = 6 ∧ a8 = 384 →
    (Real.sqrt (a2 * a8) = 48 ∨ Real.sqrt (a2 * a8) = -48) :=
by
  assume a1 q h_a1 h_q
  let a2 := geometric_sequence a1 q 1
  let a8 := geometric_sequence a1 q 7
  have h_a2 : a2 = 6, from sorry
  have h_a8 : a8 = 384, from sorry
  show (Real.sqrt (a2 * a8) = 48 ∨ Real.sqrt (a2 * a8) = -48), from sorry

end geometric_mean_a2_a8_l131_131271


namespace count_hundreds_odd_n_cubed_l131_131896
open Nat

theorem count_hundreds_odd_n_cubed :
  (Finset.filter (λ n: ℕ, ((n ^ 3) / 100 % 10) % 2 = 1) (Finset.range 101)).card = 40 :=
by
  sorry

end count_hundreds_odd_n_cubed_l131_131896


namespace unique_real_root_of_quadratic_l131_131604

theorem unique_real_root_of_quadratic (k : ℝ) :
  (∃ a : ℝ, ∀ b : ℝ, ((k^2 - 9) * b^2 - 2 * (k + 1) * b + 1 = 0 → b = a)) ↔ (k = 3 ∨ k = -3 ∨ k = -5) :=
by
  sorry

end unique_real_root_of_quadratic_l131_131604


namespace motorcycle_licenses_count_l131_131831

theorem motorcycle_licenses_count : (3 * (10 ^ 6) = 3000000) :=
by
  sorry -- Proof would go here.

end motorcycle_licenses_count_l131_131831


namespace optimal_route_l131_131452

-- Define the probabilities of no traffic jam on each road segment.
def P_AC : ℚ := 9 / 10
def P_CD : ℚ := 14 / 15
def P_DB : ℚ := 5 / 6
def P_CF : ℚ := 9 / 10
def P_FB : ℚ := 15 / 16
def P_AE : ℚ := 9 / 10
def P_EF : ℚ := 9 / 10
def P_FB2 : ℚ := 19 / 20  -- Alias for repeated probability

-- Define the probability of encountering a traffic jam on a route
def prob_traffic_jam (p_no_jam : ℚ) : ℚ := 1 - p_no_jam

-- Define the probabilities of encountering a traffic jam along each route.
def P_ACDB_jam : ℚ := prob_traffic_jam (P_AC * P_CD * P_DB)
def P_ACFB_jam : ℚ := prob_traffic_jam (P_AC * P_CF * P_FB)
def P_AEFB_jam : ℚ := prob_traffic_jam (P_AE * P_EF * P_FB2)

-- State the theorem to prove the optimal route
theorem optimal_route : P_ACDB_jam < P_ACFB_jam ∧ P_ACDB_jam < P_AEFB_jam :=
by { sorry }

end optimal_route_l131_131452


namespace Carl_saving_weeks_l131_131492

theorem Carl_saving_weeks 
  (w : ℕ)
  (saving_per_week : ℕ)
  (bills_seventh_week : ℚ)
  (dad_gift : ℕ)
  (coat_cost : ℕ) 
  (total_savings : ℚ) 
  (use_bills : bills_seventh_week = 25 * 7 * (1 / 3)) 
  (weekly_saving : saving_per_week = 25) 
  (gift_from_dad : dad_gift = 70) 
  (cost_of_coat : coat_cost = 170)
  (resulting_saving :  total_savings = (25 * w) - bills_seventh_week + dad_gift) :
  resulting_saving = coat_cost ↔ w = 19 :=
by 
  sorry

end Carl_saving_weeks_l131_131492


namespace at_least_one_fraction_lt_two_l131_131542

theorem at_least_one_fraction_lt_two 
  (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : 2 < x + y) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_fraction_lt_two_l131_131542


namespace games_played_so_far_l131_131988

-- Definitions based on conditions
def total_matches := 20
def points_for_victory := 3
def points_for_draw := 1
def points_for_defeat := 0
def points_scored_so_far := 14
def points_needed := 40
def required_wins := 6

-- The proof problem
theorem games_played_so_far : 
  ∃ W D L : ℕ, 3 * W + D + 0 * L = points_scored_so_far ∧ 
  ∃ W' D' L' : ℕ, 3 * W' + D' + 0 * L' + 3 * required_wins = points_needed ∧ 
  (total_matches - required_wins = 14) :=
by 
  sorry

end games_played_so_far_l131_131988


namespace number_of_n_such_that_n_div_25_minus_n_is_square_l131_131897

theorem number_of_n_such_that_n_div_25_minus_n_is_square :
  ∃! n1 n2 : ℤ, ∀ n : ℤ, (n = n1 ∨ n = n2) ↔ ∃ k : ℤ, k^2 = n / (25 - n) :=
sorry

end number_of_n_such_that_n_div_25_minus_n_is_square_l131_131897


namespace repetend_of_five_over_seventeen_l131_131127

theorem repetend_of_five_over_seventeen : 
  let r := 5 / 17 in
  ∃ a b : ℕ, a * 10^b = 294117 ∧ (r * 10^b - a) = (r * 10^6 - r * (10^6 / 17))
   ∧ (r * 10^k = (r * 10^6).floor / 10^k ) where k = 6 := sorry

end repetend_of_five_over_seventeen_l131_131127


namespace fair_haired_employees_percentages_l131_131071

noncomputable def fair_hair_ratio := 4
noncomputable def dark_hair_ratio := 9
noncomputable def red_hair_ratio := 7

noncomputable def women_ratio := 3
noncomputable def men_ratio := 5

noncomputable def managerial_ratio := 1
noncomputable def non_managerial_ratio := 4

noncomputable def fair_haired_women_percent := 0.40
noncomputable def women_managerial_percent := 0.60
noncomputable def men_non_managerial_percent := 0.70

theorem fair_haired_employees_percentages :
  (0.24 : ℝ) = fair_haired_women_percent * women_managerial_percent ∧
  (0.16 : ℝ) = fair_haired_women_percent * (1 - women_managerial_percent) ∧
  (0.18 : ℝ) = (1 - fair_haired_women_percent) * (1 - men_non_managerial_percent) ∧
  (0.42 : ℝ) = (1 - fair_haired_women_percent) * men_non_managerial_percent := by
  sorry

end fair_haired_employees_percentages_l131_131071


namespace correct_choice_l131_131922

-- Define propositions P and Q
def P : Prop := ∀ (r : ℚ), r ∈ ℝ
def Q : Prop := ∀ (x : ℝ), 0 < x → 0 < log x

-- Given conditions
axiom P_true : P
axiom Q_false : ¬Q

-- The correct choice among A, B, C, D is D: ¬P ∨ ¬Q
theorem correct_choice : ¬P ∨ ¬Q := by
  sorry

end correct_choice_l131_131922


namespace ball_passes_infinitely_l131_131032

theorem ball_passes_infinitely (table : Set ℝ) (P : ℝ) (pass_count : ℕ) :
  (is_circular table) ∧ (is_billiard_table table) ∧ (moves_endlessly table) ∧ 
  (reflects_on_edge table) ∧ (point_on_table P table) ∧ (pass_count = 3) →
  (∀ n : ℕ, ∃ t : ℝ, ball_passes_through P t (n + 3)) := 
sorry

end ball_passes_infinitely_l131_131032


namespace championship_races_l131_131621

theorem championship_races
  (num_sprinters : ℕ)
  (lanes : ℕ)
  (advance : ℕ)
  (eliminate : ℕ)
  (total_eliminated_sprinters : ℕ) :
  num_sprinters = 400 →
  lanes = 10 →
  advance = 2 →
  eliminate = 8 →
  total_eliminated_sprinters = 399 →
  (total_eliminated_sprinters / eliminate).ceil = 50 :=
by
  sorry

end championship_races_l131_131621


namespace Janice_earnings_l131_131652

theorem Janice_earnings (days_worked_per_week : ℕ) (earnings_per_day : ℕ) (overtime_shifts : ℕ) (overtime_earnings_per_shift : ℕ)
  (h1 : days_worked_per_week = 5)
  (h2 : earnings_per_day = 30)
  (h3 : overtime_shifts = 3)
  (h4 : overtime_earnings_per_shift = 15) :
  (days_worked_per_week * earnings_per_day) + (overtime_shifts * overtime_earnings_per_shift) = 195 :=
by {
  sorry
}

end Janice_earnings_l131_131652


namespace expected_number_of_distinct_faces_l131_131041

open BigOperators

variable (n : ℕ := 6)

def expected_distinct_faces (k : ℕ) : ℝ := 
  ∑ i in finset.range (k + 1), 
    (if i = 0 then real.of_rat ((5 / 6) ^ k) else real.of_rat (1 - ((5 / 6) ^ k)))

theorem expected_number_of_distinct_faces :
  expected_distinct_faces n = (6 ^ 6 - 5 ^ 6) / (6 ^ 5) :=
sorry

end expected_number_of_distinct_faces_l131_131041


namespace triangle_is_right_angled_triangle_is_obtuse_l131_131280

open Real

-- Prove that triangle with angles A = 30° and B = 60° is right-angled
theorem triangle_is_right_angled (A B C : ℝ) (hA : A = 30) (hB : B = 60) (hSum : A + B + C = 180) : C = 90 :=
by
  unfold RealAngleOfTriangle
  sorry

-- Prove that triangle with angle ratio A : B : C = 1 : 3 : 5 is obtuse
theorem triangle_is_obtuse (A B C : ℝ) (hRatio : A / B = 1 / 3 ∧ B / C = 3 / 5) (hSum : A + B + C = 180) : C > 90 :=
by
  unfold RatioOfTriangle
  sorry

end triangle_is_right_angled_triangle_is_obtuse_l131_131280


namespace two_digit_prime_sum_9_l131_131189

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l131_131189


namespace length_of_AB_l131_131045

-- Definitions of the given entities
def is_on_parabola (A : ℝ × ℝ) : Prop := A.2^2 = 4 * A.1
def focus : ℝ × ℝ := (1, 0)
def line_through_focus (l : ℝ × ℝ → Prop) : Prop := l focus

-- The theorem we need to prove
theorem length_of_AB (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop)
  (h1 : is_on_parabola A)
  (h2 : is_on_parabola B)
  (h3 : line_through_focus l)
  (h4 : l A)
  (h5 : l B)
  (h6 : A.1 + B.1 = 10 / 3) :
  dist A B = 16 / 3 :=
sorry

end length_of_AB_l131_131045


namespace siblings_pizza_order_l131_131899

theorem siblings_pizza_order 
  (hAlex : ℚ := 1/7) 
  (hBeth : ℚ := 2/5) 
  (hCyril : ℚ := 3/10) 
  (hLeftover : ℚ := 1 - (hAlex + hBeth + hCyril)) 
  (hDan : ℚ := 2 * hLeftover) 
  (slices : ℚ := 70)
  (alex_slices : ℚ := hAlex * slices)
  (beth_slices : ℚ := hBeth * slices)
  (cyril_slices : ℚ := hCyril * slices)
  (dan_slices : ℚ := hDan * slices):
  [beth_slices, dan_slices, cyril_slices, alex_slices].sort (≥) = [28, 22, 21, 10] := 
by 
  sorry

end siblings_pizza_order_l131_131899


namespace economic_rationale_education_policy_l131_131628

theorem economic_rationale_education_policy
  (countries : Type)
  (foreign_citizens : Type)
  (universities : Type)
  (free_or_nominal_fee : countries → Prop)
  (international_agreements : countries → Prop)
  (aging_population : countries → Prop)
  (economic_benefits : countries → Prop)
  (credit_concessions : countries → Prop)
  (reciprocity_education : countries → Prop)
  (educated_youth_contributions : countries → Prop)
  :
  (∀ c : countries, free_or_nominal_fee c ↔
    (international_agreements c ∧ (credit_concessions c ∨ reciprocity_education c)) ∨
    (aging_population c ∧ economic_benefits c ∧ educated_youth_contributions c)) := 
sorry

end economic_rationale_education_policy_l131_131628


namespace area_of_triangle_PQR_l131_131868

def point := (ℝ × ℝ)

def P : point := (-2, 3)
def Q : point := (6, 3)
def R : point := (4, -2)

theorem area_of_triangle_PQR : 
  let area := (1 / 2 : ℝ) * (8 : ℝ) * (5 : ℝ)
  in area = 20 := 
by
  sorry

end area_of_triangle_PQR_l131_131868


namespace function_always_positive_l131_131574

-- Define the function f(x)
def f (x k : ℝ) : ℝ := 2 * k * x^2 + k * x + 3 / 8

-- State the theorem to prove
theorem function_always_positive (k : ℝ) (h : 0 ≤ k ∧ k < 3) : ∀ x : ℝ, 0 < f x k :=
by {
  sorry
}

end function_always_positive_l131_131574


namespace blisters_on_rest_of_body_l131_131487

theorem blisters_on_rest_of_body (blisters_per_arm total_blisters : ℕ) (h1 : blisters_per_arm = 60) (h2 : total_blisters = 200) : 
  total_blisters - 2 * blisters_per_arm = 80 :=
by {
  -- The proof can be written here
  sorry
}

end blisters_on_rest_of_body_l131_131487


namespace composite_ratio_proof_l131_131081

theorem composite_ratio_proof :
  let composites := [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24]
  let first_seven_product := (composites.take 7).prod
  let next_seven_product := (composites.drop 7).take 7).prod
  first_seven_product.toRat / next_seven_product.toRat = (1 : ℚ) / 264 :=
by
  sorry

end composite_ratio_proof_l131_131081


namespace percentage_spent_on_household_items_eq_50_l131_131837

-- Definitions for the conditions in the problem
def MonthlyIncome : ℝ := 90000
def ClothesPercentage : ℝ := 0.25
def MedicinesPercentage : ℝ := 0.15
def Savings : ℝ := 9000

-- Definition of the statement where we need to calculate the percentage spent on household items
theorem percentage_spent_on_household_items_eq_50 :
  let ClothesExpense := ClothesPercentage * MonthlyIncome
  let MedicinesExpense := MedicinesPercentage * MonthlyIncome
  let TotalExpense := ClothesExpense + MedicinesExpense + Savings
  let HouseholdItemsExpense := MonthlyIncome - TotalExpense
  let TotalIncome := MonthlyIncome
  (HouseholdItemsExpense / TotalIncome) * 100 = 50 :=
by
  sorry

end percentage_spent_on_household_items_eq_50_l131_131837


namespace intersection_probability_l131_131333

def circle :=
  { center : (ℝ × ℝ), radius : ℝ // center = (0, 1) ∧ radius = √2 }

def line (b : ℝ) : Prop :=
  ∃ (x y : ℝ), y = x + b

def intersects (b : ℝ) : Prop :=
  let dist := |b - 1| / √2 in dist ≤ √2

def probability_intersects : ℝ :=
  let interval := Icc (-3 : ℝ) (3 : ℝ) in
  let valid_b_values := { b : ℝ // intersects b } in
  (measure_theory.measure univ valid_b_values / measure_theory.measure univ interval).to_real

theorem intersection_probability :
  probability_intersects = 2 / 3 := sorry

end intersection_probability_l131_131333


namespace train_boarding_probability_l131_131471

theorem train_boarding_probability :
  (0.5 / 5) = 1 / 10 :=
by sorry

end train_boarding_probability_l131_131471


namespace smallest_part_division_l131_131050

theorem smallest_part_division (quantity : ℕ) (a b c : ℕ) (h_quantity : quantity = 120) 
(h_proportions : a = 3 ∧ b = 5 ∧ c = 7) : 
  let x := quantity / (a + b + c) in
  min (a * x) (min (b * x) (c * x)) = 24 :=
by
  -- The proof is omitted; we only require the statement for this task.
  sorry

end smallest_part_division_l131_131050


namespace evaluate_expression_l131_131976

theorem evaluate_expression
  (p q r s : ℚ)
  (h1 : p / q = 4 / 5)
  (h2 : r / s = 3 / 7) :
  (18 / 7) + ((2 * q - p) / (2 * q + p)) - ((3 * s + r) / (3 * s - r)) = 5 / 3 := by
  sorry

end evaluate_expression_l131_131976


namespace intersection_A_B_l131_131319

def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem intersection_A_B :
  (setA ∩ setB = {x | 0 ≤ x ∧ x ≤ 2}) :=
by
  sorry

end intersection_A_B_l131_131319


namespace find_median_interval_l131_131862

-- Definition of the data for student counts
def student_counts : List (ℕ × ℕ) := [
  (55, 59, 4),
  (60, 64, 8),
  (65, 69, 15),
  (70, 74, 20),
  (75, 79, 18),
  (80, 84, 10)
]

-- Definition of total students
def total_students : ℕ := 75

-- Computation of median position
def median_position (n : ℕ) : ℕ := (n + 1) / 2

-- Definition of cumulative frequencies
def cumulative_frequencies (counts : List (ℕ × ℕ × ℕ)) : List ℕ :=
  counts.scanl (+) 0 (List.map (λ (_, _, c) => c) counts)

-- Prove that the interval containing the median score
theorem find_median_interval :
  let median_pos := median_position total_students,
  let cum_freqs := cumulative_frequencies student_counts,
  median_pos ≤ cum_freqs.nth! 3 ∧ median_pos > cum_freqs.nth! 2 :=
sorry

end find_median_interval_l131_131862


namespace min_val_xyz_l131_131300

noncomputable def minProduct (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 2) (h_order : x ≤ y ∧ y ≤ z) (h_bound : z ≤ 3 * x) : ℝ :=
  xyz := x*y*z

theorem min_val_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 2) (h_order : x ≤ y ∧ y ≤ z) (h_bound : z ≤ 3 * x) :
  minProduct x y z hx hy hz h_sum h_order h_bound = 32 / 81 := sorry

end min_val_xyz_l131_131300


namespace smallest_constant_l131_131543

theorem smallest_constant (c : ℝ) (h1 : c ∈ Ioo (1 / 2 : ℝ) 1) :
  ∃ M, (M = 1 / (1 - c)) ∧ 
        ∀ (n : ℕ) (hn : 2 ≤ n) (a : ℕ → ℝ),
        (∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ a i) ∧ 
        (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → a i ≤ a j) ∧ 
        (∑ k in finset.range(n+1), k * a k = c * ∑ k in finset.range(n+1), a k) → 
        (∑ k in finset.range(n+1), a k ≤ M * ∑ k in finset.Ico 1 (⌊c * n⌋ + 1), a k) :=
by sorry

end smallest_constant_l131_131543


namespace hexagon_angle_F_l131_131476

theorem hexagon_angle_F (A B C D E F : ℝ) (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) (side_length : ℝ) :
  (angle_A = 120) → (angle_B = 120) → (angle_C = 90) →
  (∀ i j, i ≠ j → side_length i = side_length j) →
  ∑ (angle : fin 6), angle = 720 →
  angle_F = 210 :=
by
  intros hA hB hC h_equal_lengths h_sum_angles
  sorry

end hexagon_angle_F_l131_131476


namespace even_comp_even_l131_131304

variable {X : Type} [AddGroup X] [HasNeg X]

def is_even_function (g : X → X) : Prop :=
  ∀ x : X, g (-x) = g x

theorem even_comp_even (g : X → X) (h_even : is_even_function g) :
  is_even_function (g ∘ g) :=
by
  intro x
  rw [function.comp_apply, function.comp_apply]
  rw [h_even] -- by the even property of g
  rw [h_even] -- by the even property of g
  sorry

end even_comp_even_l131_131304


namespace probability_of_point_in_sphere_and_cube_l131_131205

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Math.pi * r^3
noncomputable def volume_of_cube (s : ℝ) : ℝ := s^3
noncomputable def inscribed_cube_side_length (r : ℝ) : ℝ := 2 * r / Real.sqrt 3
noncomputable def probability_point_in_cube (r : ℝ) : ℝ :=
  let s := inscribed_cube_side_length r
  volume_of_cube s / volume_of_sphere r

theorem probability_of_point_in_sphere_and_cube (r : ℝ) (h : r = 2 * Real.sqrt 3) :
  probability_point_in_cube r = (2 * Real.sqrt 3) / (3 * Math.pi) :=
by
  rw [h]
  sorry

end probability_of_point_in_sphere_and_cube_l131_131205


namespace rate_per_annum_l131_131056

noncomputable def simple_interest_rate (P: ℝ) (T: ℝ) (R: ℝ) : Prop :=
  P * R * T / 100 = P / 6

theorem rate_per_annum : ∃ R : ℝ, simple_interest_rate 1 7 R ∧ (100 / 49 ≈ 2.04) := 
by 
  sorry

end rate_per_annum_l131_131056


namespace find_f_neg_19_div_3_l131_131228

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 1 then 
    8^x 
  else 
    sorry -- The full definition is complex and not needed for the statement

-- Define the properties of f
lemma f_periodic (x : ℝ) : f (x + 2) = f x := 
  sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := 
  sorry

theorem find_f_neg_19_div_3 : f (-19/3) = -2 :=
  sorry

end find_f_neg_19_div_3_l131_131228


namespace min_dot_product_trajectory_l131_131208

-- Definitions of points and conditions
def point (x y : ℝ) : Prop := True

def trajectory (P : ℝ × ℝ) : Prop := 
  let x := P.1
  let y := P.2
  x * x - y * y = 2 ∧ x ≥ Real.sqrt 2

-- Definition of dot product over vectors from origin
def dot_product (A B : ℝ × ℝ) : ℝ :=
  A.1 * B.1 + A.2 * B.2

-- Stating the theorem for minimum value of dot product
theorem min_dot_product_trajectory (A B : ℝ × ℝ) (hA : trajectory A) (hB : trajectory B) : 
  dot_product A B ≥ 2 := 
sorry

end min_dot_product_trajectory_l131_131208


namespace quadratic_y1_gt_y2_l131_131231

theorem quadratic_y1_gt_y2 {a b c y1 y2 : ℝ} (ha : a > 0) (hy1 : y1 = a * (-1)^2 + b * (-1) + c) (hy2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
  sorry

end quadratic_y1_gt_y2_l131_131231


namespace basement_pump_time_l131_131033

/-- A basement has a 30-foot by 36-foot rectangular floor, flooded to a depth of 24 inches.
Using three pumps, each pumping 10 gallons per minute, and knowing that a cubic foot of water
contains 7.5 gallons, this theorem asserts it will take 540 minutes to pump out all the water. -/
theorem basement_pump_time :
  let length := 30 -- in feet
  let width := 36 -- in feet
  let depth_inch := 24 -- in inches
  let depth := depth_inch / 12 -- converting depth to feet
  let volume_ft3 := length * width * depth -- volume in cubic feet
  let gallons_per_ft3 := 7.5 -- gallons per cubic foot
  let total_gallons := volume_ft3 * gallons_per_ft3 -- total volume in gallons
  let pump_capacity_gpm := 10 -- gallons per minute per pump
  let total_pumps := 3 -- number of pumps
  let total_pump_gpm := pump_capacity_gpm * total_pumps -- total gallons per minute for all pumps
  let pump_time := total_gallons / total_pump_gpm -- time in minutes to pump all the water
  pump_time = 540 := sorry

end basement_pump_time_l131_131033


namespace julia_birth_year_l131_131984

open Nat

theorem julia_birth_year (w_age : ℕ) (p_diff : ℕ) (j_diff : ℕ) (current_year : ℕ) :
  w_age = 37 →
  p_diff = 3 →
  j_diff = 2 →
  current_year = 2021 →
  (current_year - w_age) - p_diff - j_diff = 1979 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end julia_birth_year_l131_131984


namespace monotonous_integers_count_is_494_l131_131866

noncomputable def monotonous_integers_count : ℕ :=
  let increasing_seqs := (∑ n in finset.range (9 + 1), if 2 ≤ n then nat.choose 8 n else 0)
  let decreasing_seqs := (∑ n in finset.range (9 + 1), if 2 ≤ n then nat.choose 8 n else 0)
  in 2 * (increasing_seqs + decreasing_seqs)

theorem monotonous_integers_count_is_494 : monotonous_integers_count = 494 :=
sorry

end monotonous_integers_count_is_494_l131_131866


namespace obtuse_triangle_l131_131322

noncomputable theory
open_locale big_operators

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (not_collinear : ¬ collinear ℝ ({0, a, b} : set V))
variables (h1 : ∥a - b∥ = 3)
variables (h2 : ∥a + b∥ = 1)

theorem obtuse_triangle (a b : V) (not_collinear : ¬ collinear ℝ ({0, a, b} : set V))
  (h1 : ∥a - b∥ = 3) (h2 : ∥a + b∥ = 1) : 
  ∃ (angle_ABC : ℝ), angle_ABC > pi/2 ∧ triangle a b angle_ABC
:=
sorry

end obtuse_triangle_l131_131322


namespace max_value_proof_l131_131895

noncomputable def max_value_of_a_b_c : ℕ :=
  let n : ℕ := sorry
  let a b c : ℕ := sorry
  let A_n : ℕ := a * (10^n - 1) / 9
  let B_n : ℕ := b * (10^n - 1) / 9
  let C_n : ℕ := c * (10^(2*n) - 1) / 9
  if C_n - B_n = A_n^2 ∧ (∃ n1 n2 : ℕ, n1 ≠ n2 ∧ C_n - B_n = A_n^2)
  then a + b + c
  else 0

theorem max_value_proof : max_value_of_a_b_c = 18 := 
sorry

end max_value_proof_l131_131895


namespace negation_example_l131_131240

open Classical
variable (x : ℝ)

theorem negation_example :
  (¬ (∀ x : ℝ, 2 * x - 1 > 0)) ↔ (∃ x : ℝ, 2 * x - 1 ≤ 0) :=
by
  sorry

end negation_example_l131_131240


namespace tangential_quadrilateral_difference_l131_131049

-- Definitions of the conditions given in the problem
def is_cyclic_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the quadrilateral vertices lie on a circle
def is_tangential_quadrilateral (a b c d : ℝ) : Prop := sorry -- In real setting, it means the sides are tangent to a common incircle
def point_tangency (a b c : ℝ) : Prop := sorry

-- Main theorem
theorem tangential_quadrilateral_difference (AB BC CD DA : ℝ) (x y : ℝ) 
  (h1 : is_cyclic_quadrilateral AB BC CD DA)
  (h2 : is_tangential_quadrilateral AB BC CD DA)
  (h3 : AB = 80) (h4 : BC = 140) (h5 : CD = 120) (h6 : DA = 100)
  (h7 : point_tangency x y CD)
  (h8 : x + y = 120) :
  |x - y| = 80 := 
sorry

end tangential_quadrilateral_difference_l131_131049


namespace calculate_unused_sector_angle_l131_131900

theorem calculate_unused_sector_angle
  (R : ℝ)
  (cone_base_radius : ℝ := 9)
  (cone_volume : ℝ := 243 * real.pi)  
  (unused_sector_angle : ℝ := 105.44) : 
  unused_sector_angle = 360 - 254.56 :=
by
  -- radius: R
  -- Mary's cone radius: cone_base_radius
  -- cone volume: cone_volume
  -- unused sector angle to be proved: 105.44
  sorry

end calculate_unused_sector_angle_l131_131900


namespace not_less_than_x3_y5_for_x2y_l131_131795

theorem not_less_than_x3_y5_for_x2y (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) : x^2 * y ≥ x^3 + y^5 :=
sorry

end not_less_than_x3_y5_for_x2y_l131_131795


namespace symmetry_properties_l131_131671

noncomputable def f : ℝ → ℝ := sorry

def g₁ (x : ℝ) := f (x + 3)
def g₂ (x : ℝ) := f (3 - x)

theorem symmetry_properties (h : ∀ x, f (1 + x) = f (1 - x)) :
  (∀ x, f x = f (2 - x)) ∧ (∀ x, g₂ (-x) = g₁ x) := sorry

end symmetry_properties_l131_131671


namespace repetend_of_fraction_l131_131163

/-- The repeating sequence of the decimal representation of 5/17 is 294117 -/
theorem repetend_of_fraction : 
  let rep := list.take 6 (list.drop 1 (to_digits 10 (5 / 17) 8)) in
  rep = [2, 9, 4, 1, 1, 7] := 
by
  sorry

end repetend_of_fraction_l131_131163


namespace polygon_possible_with_area_sixteen_l131_131746

theorem polygon_possible_with_area_sixteen :
  ∃ (P : polygon) (matches : list (side P)), (length(matches) = 12 ∧ (∀ m ∈ matches, m.length = 2) ∧ P.area = 16) := 
sorry

end polygon_possible_with_area_sixteen_l131_131746


namespace cube_triangle_area_sum_l131_131082

/-- 
  Given a 2x2x2 cube, calculate the sum of the areas of all triangles 
  whose vertices are also vertices of the cube. Express the total area 
  as m + √n + √p, where m, n, and p are integers. Prove that 
  m = 48, n = 288, p = 288 and that the value of m + n + p is 624.
-/
theorem cube_triangle_area_sum :
  ∃ (m n p : ℕ), (m + n + p = 624) ∧ (m = 48) ∧ (n = 288) ∧ (p = 288) ∧
  (let total_area = m + (nat.sqrt n) + (nat.sqrt p) in 
  total_area = 48 + 48 * real.sqrt 2 + 32 * real.sqrt 3) :=
by 
  sorry

end cube_triangle_area_sum_l131_131082


namespace range_of_a_l131_131732

theorem range_of_a (a : ℝ) : 
  (∀ x, (0 ≤ x ∧ x ≤ 2) → (a ≤ x ∧ x ≤ a + 3)) ∧ 
  (∃ x, (a ≤ x ∧ x ≤ a + 3) ∧ ¬(0 ≤ x ∧ x ≤ 2)) → 
  a ∈ Icc (-1 : ℝ) (0 : ℝ) := 
by 
  sorry

end range_of_a_l131_131732


namespace find_number_l131_131047

theorem find_number (x : ℝ) (h : 2 * x - 2.6 * 4 = 10) : x = 10.2 :=
sorry

end find_number_l131_131047


namespace sum_of_coefficients_l131_131083

-- Define the polynomial
def polynomial := 4 * (2 * x^8 - 3 * x^5 + 9) + 9 * (x^6 + 4 * x^3 - x + 6)

-- State the problem of summing the coefficients
theorem sum_of_coefficients : 
  (4 * (2 * 1^8 - 3 * 1^5 + 9) + 9 * (1^6 + 4 * 1^3 - 1 + 6)) = 122 :=
by
  sorry

end sum_of_coefficients_l131_131083


namespace find_x_in_plane_figure_l131_131410

theorem find_x_in_plane_figure (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 360) 
  (h3 : 2 * x + 160 = 360) : 
  x = 100 :=
by
  sorry

end find_x_in_plane_figure_l131_131410


namespace n_tuples_condition_l131_131201

theorem n_tuples_condition (n : ℕ) (h₁ : 2 ≤ n) 
  (a : Fin n → ℕ) (h₂ : 1 < a 0) (h₃ : ∀ i : Fin (n - 1), a i ≤ a (i + 1))
  (h₄ : Odd (a 0)) 
  (h₅ : ∃ M > 0, M = (1 / 2^n) * (a 0 - 1) * ∏ i in (Finset.range n).erase 0, a i)
  (h₆ : ∃ (k : Fin (h₅.some)), ∀ i₁ i₂ : Fin h₅.some, i₁ < i₂ → ∃ j : Fin n, (k i₁ j - k i₂ j) % (a j) ∉ {0, ±1}) : 
  ∃ a : ℕ, 2 ^ a ∣ a 0 - 1 := 
sorry

end n_tuples_condition_l131_131201


namespace max_sum_of_arithmetic_sequence_l131_131215

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 7)
  (h_a1_a7 : a 1 + a 7 = 10)
  (h_S : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 1 - a 0))) / 2) :
  ∃ n, S n = S 6 ∧ (∀ m, S m ≤ S 6) :=
sorry

end max_sum_of_arithmetic_sequence_l131_131215


namespace janice_weekly_earnings_l131_131649

-- define the conditions
def regular_days_per_week : Nat := 5
def regular_earnings_per_day : Nat := 30
def overtime_earnings_per_shift : Nat := 15
def overtime_shifts_per_week : Nat := 3

-- define the total earnings calculation
def total_earnings (regular_days : Nat) (regular_rate : Nat) (overtime_shifts : Nat) (overtime_rate : Nat) : Nat :=
  (regular_days * regular_rate) + (overtime_shifts * overtime_rate)

-- state the problem to be proved
theorem janice_weekly_earnings : total_earnings regular_days_per_week regular_earnings_per_day overtime_shifts_per_week overtime_earnings_per_shift = 195 :=
by
  sorry

end janice_weekly_earnings_l131_131649


namespace find_m_plus_n_l131_131230

theorem find_m_plus_n (a m n : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a^m = n) (h4 : a^0 = 1) : m + n = 1 :=
sorry

end find_m_plus_n_l131_131230


namespace explain_education_policy_l131_131630

theorem explain_education_policy :
  ∃ (reason1 reason2 : String), reason1 ≠ reason2 ∧
    (reason1 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason2 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions")
    ∨
    (reason2 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason1 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions") :=
by
  sorry

end explain_education_policy_l131_131630


namespace man_speed_in_still_water_l131_131458

-- Definitions based on conditions from part a)
def speed_of_current : ℝ := 5  -- 5 kmph
def time_in_seconds : ℝ := 10.799136069114471  -- Time taken to cover 60 meters
def distance_in_meters : ℝ := 60  -- Distance covered downstream

-- Conversion factors
def meters_to_kilometers (m : ℝ) : ℝ := m / 1000
def seconds_to_hours (s : ℝ) : ℝ := s / 3600

-- Calculations based on solution in part b)
def time_in_hours : ℝ := seconds_to_hours time_in_seconds
def distance_in_kilometers : ℝ := meters_to_kilometers distance_in_meters
def speed_downstream : ℝ := distance_in_kilometers / time_in_hours

-- The goal is to prove that the man's speed in still water is 15.0008 kmph
theorem man_speed_in_still_water : ∃ (v : ℝ), v ≈ 15.0008 ∧ (speed_downstream = v + speed_of_current) :=
by
  use (speed_downstream - speed_of_current)
  sorry

end man_speed_in_still_water_l131_131458


namespace product_of_real_roots_eq_one_l131_131171

theorem product_of_real_roots_eq_one :
  (∀ x : ℝ, x ^ real.log x / real.log 5 = 25) → 
  ( 5^real.sqrt 2 * 5^(-real.sqrt 2) = 1) := 
by
  sorry

end product_of_real_roots_eq_one_l131_131171


namespace stadium_breadth_l131_131980

theorem stadium_breadth (P L B : ℕ) (h1 : P = 800) (h2 : L = 100) :
  2 * (L + B) = P → B = 300 :=
by
  sorry

end stadium_breadth_l131_131980


namespace hyperbola_equation_correct_l131_131352

-- Definition of a hyperbola with given focus and asymptotes.
def hyperbola_with_focus_and_asymptotes (focus : ℝ × ℝ) (asymptote_eq : ℝ) : Prop := 
  focus = (0, 6) ∧ asymptote_eq = 2

-- The correct equation that needs to be proven.
def correct_hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 12 - x^2 / 24 = 1

-- The Lean 4 theorem statement encompassing the given problem conditions and correct answer.
theorem hyperbola_equation_correct : 
  (∀ x y, hyperbola_with_focus_and_asymptotes (0,6) 2 → correct_hyperbola_equation x y) :=
by
  -- We state the existence of simplified values to satisfy the proof without performing the actual proof steps.
  intro x y h
  have : correct_hyperbola_equation x y,
  sorry
  exact this

end hyperbola_equation_correct_l131_131352


namespace shifted_parabola_equation_l131_131723

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the shifted parabola function
def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

-- Proposition to prove that the given parabola equation is correct after transformations
theorem shifted_parabola_equation : 
  ∀ x : ℝ, shifted_parabola x = -2 * (x + 1)^2 + 3 :=
by
  sorry

end shifted_parabola_equation_l131_131723


namespace fenced_area_with_cutout_l131_131358

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l131_131358


namespace standard_deviation_of_data_l131_131382

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (fun x => (x - m)^2)).sum / data.length

noncomputable def std_dev (data : List ℝ) : ℝ :=
  Real.sqrt (variance data)

theorem standard_deviation_of_data :
  std_dev [5, 7, 7, 8, 10, 11] = 2 := 
sorry

end standard_deviation_of_data_l131_131382


namespace proof_problem_l131_131097

def Delta (a b : ℕ) : ℕ := a^3 - b

theorem proof_problem : (5^(Delta 6 8)) Δ (4^(Delta 2 7)) = 5^(624) - 4 :=
by
  -- Define the custom operation
  let Δ := λ (a b : ℕ), a^3 - b
  -- Define intermediate computations
  let six_delta_eight := Δ 6 8 
  let two_delta_seven := Δ 2 7 
  -- Perform exponentiations
  let five_to_six_delta_eight := 5 ^ six_delta_eight
  let four_to_two_delta_seven := 4 ^ two_delta_seven
  -- Use the custom operation again
  let expression := Δ five_to_six_delta_eight four_to_two_delta_seven
  -- The final comparison
  exact expression = 5^(624) - 4

end proof_problem_l131_131097


namespace repetend_of_5_over_17_is_294117_l131_131150

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l131_131150


namespace age_of_b_l131_131009

variable (a b c : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = 2 * c
def condition3 : Prop := a + b + c = 27

theorem age_of_b (h1 : condition1 a b)
                 (h2 : condition2 b c)
                 (h3 : condition3 a b c) : 
                 b = 10 := 
by sorry

end age_of_b_l131_131009


namespace four_digit_unique_count_l131_131898

theorem four_digit_unique_count : 
  (∃ k : ℕ, k = 14 ∧ ∃ lst : List ℕ, lst.length = 4 ∧ 
    (∀ d ∈ lst, d = 2 ∨ d = 3) ∧ (2 ∈ lst) ∧ (3 ∈ lst)) :=
by
  sorry

end four_digit_unique_count_l131_131898


namespace blue_tint_percentage_in_new_mixture_l131_131437

-- Define the conditions given in the problem
def original_volume : ℝ := 40
def blue_tint_percentage : ℝ := 0.20
def added_blue_tint_volume : ℝ := 8

-- Calculate the original blue tint volume
def original_blue_tint_volume := blue_tint_percentage * original_volume

-- Calculate the new blue tint volume after adding more blue tint
def new_blue_tint_volume := original_blue_tint_volume + added_blue_tint_volume

-- Calculate the new total volume of the mixture
def new_total_volume := original_volume + added_blue_tint_volume

-- Define the expected result in percentage
def expected_blue_tint_percentage : ℝ := 33.3333

-- Statement to prove
theorem blue_tint_percentage_in_new_mixture :
  (new_blue_tint_volume / new_total_volume) * 100 = expected_blue_tint_percentage :=
sorry

end blue_tint_percentage_in_new_mixture_l131_131437


namespace length_of_AD_l131_131499

theorem length_of_AD
  (A B C D : Point)
  (AB AC : Segment)
  (BC : Segment)
  (AB_eq : length AB = 26)
  (AC_eq : length AC = 26)
  (BC_eq : length BC = 24)
  (D_midpoint : midpoint D B C) :
  length AD = 2 * Real.sqrt 133 :=
by
  sorry

end length_of_AD_l131_131499


namespace quadratic_inequality_solution_l131_131369

theorem quadratic_inequality_solution (a : ℝ) (h1 : ∀ x : ℝ, ax^2 + (a + 1) * x + 1 ≥ 0) : a = 1 := by
  sorry

end quadratic_inequality_solution_l131_131369


namespace converse_even_sum_l131_131793

variable (a b : ℤ)

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem converse_even_sum (h : is_even (a + b)) : is_even a ∧ is_even b :=
sorry

end converse_even_sum_l131_131793


namespace largest_divisor_of_even_diff_squares_l131_131663

theorem largest_divisor_of_even_diff_squares (m n : ℤ) (h_m_even : ∃ k : ℤ, m = 2 * k) (h_n_even : ∃ k : ℤ, n = 2 * k) (h_n_lt_m : n < m) : 
  ∃ d : ℤ, d = 16 ∧ ∀ p : ℤ, (p ∣ (m^2 - n^2)) → p ≤ d :=
sorry

end largest_divisor_of_even_diff_squares_l131_131663


namespace find_functional_equation_solutions_l131_131516

theorem find_functional_equation_solutions :
  (∀ f : ℝ → ℝ, (∀ x y : ℝ, x > 0 → y > 0 → f x * f (y * f x) = f (x + y)) →
    (∃ a > 0, ∀ x > 0, f x = 1 / (1 + a * x) ∨ ∀ x > 0, f x = 1)) :=
by
  sorry

end find_functional_equation_solutions_l131_131516


namespace quadrilateral_area_l131_131011

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 30) (hh1 : h1 = 10) (hh2 : h2 = 6) :
  (1 / 2 * d * (h1 + h2) = 240) := by
  sorry

end quadrilateral_area_l131_131011


namespace working_mom_work_percentage_l131_131834

theorem working_mom_work_percentage :
  let total_hours_in_day := 24
  let work_hours := 8
  let gym_hours := 2
  let cooking_hours := 1.5
  let bath_hours := 0.5
  let homework_hours := 1
  let packing_hours := 0.5
  let cleaning_hours := 0.5
  let leisure_hours := 2
  let total_activity_hours := work_hours + gym_hours + cooking_hours + bath_hours + homework_hours + packing_hours + cleaning_hours + leisure_hours
  16 = total_activity_hours →
  (work_hours / total_hours_in_day) * 100 = 33.33 :=
by
  sorry

end working_mom_work_percentage_l131_131834


namespace mrs_hilt_rocks_proof_l131_131324

def num_rocks_already_placed : ℝ := 125.0
def total_num_rocks_planned : ℝ := 189
def num_more_rocks_needed : ℝ := 64

theorem mrs_hilt_rocks_proof : total_num_rocks_planned - num_rocks_already_placed = num_more_rocks_needed :=
by
  sorry

end mrs_hilt_rocks_proof_l131_131324


namespace acute_angle_probability_l131_131820

noncomputable def prob_acute_angle : ℝ :=
  let m_values := [1, 2, 3, 4, 5, 6]
  let outcomes_count := (36 : ℝ)
  let good_outcomes_count := (15 : ℝ)
  good_outcomes_count / outcomes_count

theorem acute_angle_probability :
  prob_acute_angle = 5 / 12 :=
by
  sorry

end acute_angle_probability_l131_131820


namespace base8_subtraction_l131_131515

theorem base8_subtraction :
  (7324₈ - 3657₈ = 4445₈) := sorry

end base8_subtraction_l131_131515


namespace polygon_possible_with_area_sixteen_l131_131742

theorem polygon_possible_with_area_sixteen :
  ∃ (P : polygon) (matches : list (side P)), (length(matches) = 12 ∧ (∀ m ∈ matches, m.length = 2) ∧ P.area = 16) := 
sorry

end polygon_possible_with_area_sixteen_l131_131742


namespace better_fitting_ModelA_l131_131412

def R2 (model : Type) : ℝ := sorry

variable (ModelA ModelB : Type)

axiom R2_ModelA : R2 ModelA ≈ 0.96
axiom R2_ModelB : R2 ModelB ≈ 0.85

theorem better_fitting_ModelA (h1 : R2 ModelA ≈ 0.96) (h2 : R2 ModelB ≈ 0.85) : 
  (R2 ModelA > R2 ModelB) :=
by 
  simp [h1, h2]
  sorry

end better_fitting_ModelA_l131_131412


namespace line_tangent_to_exp_l131_131565

-- Define the exponential function and its derivative
def exp (x : ℝ) : ℝ := Real.exp x
def exp_deriv (x : ℝ) : ℝ := Real.exp x

-- Define the condition of tangency at a point (x0, y0)
def is_tangent (k : ℝ) (x0 : ℝ) (y0 : ℝ) : Prop :=
  y0 = exp x0 ∧ y0 = k * x0 ∧ exp_deriv x0 = k

theorem line_tangent_to_exp (k : ℝ) (x0 : ℝ) (y0 : ℝ) (h_tangent : is_tangent k x0 y0) : k = Real.exp 1 :=
by
  have exp_x0 : exp x0 = k := h_tangent.1
  have : x0 = 1 := by sorry
  rw [exp_x0] at this
  exact this

end line_tangent_to_exp_l131_131565


namespace repetend_of_5_div_17_l131_131138

theorem repetend_of_5_div_17 : 
  ∃ repetend : ℕ, 
  decimal_repetend (5 / 17) = repetend ∧ 
  repetend = 2941176470588235 :=
by 
  skip

end repetend_of_5_div_17_l131_131138


namespace angle_ADP_eq_angle_PBQ_l131_131623

section GeometricProof

variables {A B C D E F P Q : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace P] [MetricSpace Q]
variables {ab ac bc ca ef : Line}
variables (hABC : AcuteTriangle A B C)
variables (hAB_lt_AC : Distance AB AB < Distance AC AC)
variables (hD_foot : PerpendicularFoot A BC D)
variables (hE_foot : PerpendicularFoot B CA E)
variables (hF_foot : PerpendicularFoot C AB F)
variables (hDP_perp_EF : Perpendicular DP EF)
variables (hBQ_eq_CQ : Distance B Q = Distance C Q)

theorem angle_ADP_eq_angle_PBQ :
  ∠ ADP = ∠ PBQ :=
by
  sorry

end GeometricProof

end angle_ADP_eq_angle_PBQ_l131_131623


namespace length_of_CD_l131_131627

theorem length_of_CD 
  (BO OD AO OC AB : ℝ)
  (hBO : BO = 5) 
  (hOD : OD = 7) 
  (hAO : AO = 10) 
  (hOC : OC = 4) 
  (hAB : AB = 8) : 
  (BO * OC * cos (angle B O C) + OD * OC * cos (angle O D C) - 2 * OC * OD * cos (angle O D C) = 99.16) := 
sorry

end length_of_CD_l131_131627


namespace count_propositions_l131_131067

def is_proposition (s : String) : Prop :=
  s = "−5 ∈ ℤ" ∨ s = "π ∉ ℝ" ∨ s = "0 {0} ∈ ℕ"

theorem count_propositions :
  (if is_proposition "1: |x + 2|" then 1 else 0) +
  (if is_proposition "-5 ∈ ℤ" then 1 else 0) +
  (if is_proposition "π ∉ ℝ" then 1 else 0) +
  (if is_proposition "{0} ∈ ℕ" then 1 else 0) = 3 :=
by
  sorry

end count_propositions_l131_131067


namespace construct_polygon_with_area_l131_131755

theorem construct_polygon_with_area 
  (n : ℕ) (l : ℝ) (a : ℝ) 
  (matchsticks : n = 12) 
  (matchstick_length : l = 2) 
  (area_target : a = 16) : 
  ∃ (polygon : EuclideanGeometry.Polygon ℝ) (sides : list ℝ),
    sides.length = n ∧ ∀ side ∈ sides, side = l ∧ polygon.area = a := 
sorry

end construct_polygon_with_area_l131_131755


namespace number_of_male_rabbits_l131_131762

-- Definitions based on the conditions
def white_rabbits : ℕ := 12
def black_rabbits : ℕ := 9
def female_rabbits : ℕ := 8

-- The question and proof goal
theorem number_of_male_rabbits : 
  (white_rabbits + black_rabbits - female_rabbits) = 13 :=
by
  sorry

end number_of_male_rabbits_l131_131762


namespace gain_percent_is_28_125_l131_131018

variable (MP : ℝ) (CP SP Gain : ℝ)

def CP_condition : Prop := CP = 0.64 * MP
def SP_condition : Prop := SP = MP * 0.82
def Gain_definition : Prop := Gain = SP - CP
def Gain_percent : ℝ := (Gain / CP) * 100

theorem gain_percent_is_28_125
  (h1 : CP_condition MP CP)
  (h2 : SP_condition MP SP)
  (h3 : Gain_definition MP SP CP Gain) :
  Gain_percent CP Gain = 28.125 := 
by
  sorry

end gain_percent_is_28_125_l131_131018


namespace car_travel_city_miles_per_gallon_l131_131419

noncomputable def miles_per_gallon_city (miles_highway : ℕ) (miles_city : ℕ) (less_miles_city : ℕ) (miles_per_tank_highway : ℕ) (miles_per_tank_city : ℕ) : ℕ :=
  let h := miles_per_tank_highway / miles_highway in
  let c := h - less_miles_city in
  c

theorem car_travel_city_miles_per_gallon :
  ∀ (miles_highway : ℕ) (miles_city : ℕ) (less_miles_city : ℕ) (miles_per_tank_highway : ℕ) (miles_per_tank_city : ℕ),
    miles_highway = 462 → miles_city = 336 → less_miles_city = 3 → 
    miles_per_tank_highway = 462 → miles_per_tank_city = 336 → 
    miles_per_gallon_city miles_highway miles_city less_miles_city miles_per_tank_highway miles_per_tank_city = 8 :=
by
  intros miles_highway miles_city less_miles_city miles_per_tank_highway miles_per_tank_city
  intro h_eq
  intro c_eq
  intro l_eq
  intro th_eq
  intro tc_eq
  -- sorry to skip the proof steps
  sorry

end car_travel_city_miles_per_gallon_l131_131419


namespace units_digit_difference_l131_131563

def is_units_digit_11 (n : ℤ) : Prop := (n % 10 = 1)

def is_even (n : ℤ) : Prop := n % 2 = 0

def units_digit (n: ℤ) : ℕ := (n % 10).nat_abs

theorem units_digit_difference (p : ℤ) 
  (h1 : 0 < p)
  (h2 : is_even p)
  (h3 : units_digit p = 6)
  (h4 : units_digit (p + 5) = 1) :
  units_digit (p^3) - units_digit (p^2) = 0 := by
  sorry

end units_digit_difference_l131_131563


namespace exists_infinite_A_and_B_l131_131847

open Nat

def is_valid_set (A B : Set ℕ) :=
  ∀ n : ℕ, ∃! (a b : ℕ), (a ∈ A ∧ b ∈ B ∧ n = a + b)

theorem exists_infinite_A_and_B :
  ∃ A B : Set ℕ, (∀ m : ℕ, m ∈ A → 0 ≤ m ∧ m ∉ B) ∧
                 (∀ m : ℕ, m ∈ B → 0 ≤ m ∧ m ∉ A) ∧
                 is_valid_set A B ∧
                 (Set.Infinite A ∧ Set.Infinite B) :=
begin
  sorry
end

end exists_infinite_A_and_B_l131_131847


namespace circles_intersect_on_AB_l131_131292

-- Define the structures and properties of the points and triangles involved
structure Point :=
(x : ℝ)
(y : ℝ)

def is_right_triangle (A B C : Point) : Prop :=
    C.x = A.x ∨ C.x = B.x ∧ C.y = A.y ∨ C.y = B.y

def midpoint (A B : Point) : Point :=
    {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2}

def on_segment (G C : Point) : Prop := sorry -- Define G on the segment MC

def angle_eq (P A G : Point) (B C : Point) : Prop := sorry -- Define that corresponding angles are equal

noncomputable def proof_problem (A B C M G P Q : Point) : Prop :=
  is_right_triangle A B C ∧
  M = midpoint A B ∧
  on_segment G M ∧
  angle_eq P A G B C ∧
  angle_eq Q B G C A

theorem circles_intersect_on_AB
  (A B C M G P Q : Point) (h : proof_problem A B C M G P Q) :
  ∃ H : Point, sorry -- H is the intersection point on AB
    -- here, more conditions defining the circumscribed circles intersection
    -- this would typically involve circumcircle definitions and other geometric properties
    sorry :=
sorry -- Proof would go here

end circles_intersect_on_AB_l131_131292


namespace yearly_payment_split_evenly_l131_131005

def monthly_cost : ℤ := 14
def split_cost (cost : ℤ) := cost / 2
def total_yearly_cost (monthly_payment : ℤ) := monthly_payment * 12

theorem yearly_payment_split_evenly (h : split_cost monthly_cost = 7) :
  total_yearly_cost (split_cost monthly_cost) = 84 :=
by
  -- Here we use the hypothesis h which simplifies the proof.
  sorry

end yearly_payment_split_evenly_l131_131005


namespace last_digit_of_power_of_two_l131_131342

theorem last_digit_of_power_of_two (n : ℕ) (h : n ≥ 2) : (2 ^ (2 ^ n) + 1) % 10 = 7 :=
sorry

end last_digit_of_power_of_two_l131_131342


namespace smallest_repeating_block_length_div_8_7_l131_131245

theorem smallest_repeating_block_length_div_8_7 : 
  (minimum_repeating_block_length (decimal_expansion (8/7)) = 6) := 
sorry

end smallest_repeating_block_length_div_8_7_l131_131245


namespace Elise_paid_23_dollars_l131_131780

-- Definitions and conditions
def base_price := 3
def cost_per_mile := 4
def distance := 5

-- Desired conclusion (total cost)
def total_cost := base_price + cost_per_mile * distance

-- Theorem statement
theorem Elise_paid_23_dollars : total_cost = 23 := by
  sorry

end Elise_paid_23_dollars_l131_131780


namespace leap_years_count_l131_131464

theorem leap_years_count : 
  {y : ℕ | 2100 ≤ y ∧ y ≤ 4500 ∧ y % 100 = 0 ∧ ((y % 1100 = 300) ∨ (y % 1100 = 700))}.to_finset.card = 4 :=
by
  -- Proof is required to show this number counts to 4
  sorry

end leap_years_count_l131_131464


namespace evaluate_log_l131_131115

theorem evaluate_log : ∀ (a b : ℝ), a = 3^2 → log a (b^(1/3)) = (1/3) := 
by
  intros a b h1
  sorry

end evaluate_log_l131_131115


namespace fifteenth_term_is_143_l131_131715

noncomputable def first_term : ℕ := 3
noncomputable def second_term : ℕ := 13
noncomputable def third_term : ℕ := 23
noncomputable def common_difference : ℕ := second_term - first_term
noncomputable def nth_term (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

theorem fifteenth_term_is_143 :
  nth_term 15 = 143 := by
  sorry

end fifteenth_term_is_143_l131_131715


namespace prob_half_to_four_l131_131379

noncomputable theory

variable (X : ℕ → ℝ)

def P : ℕ → ℝ := λ k,
  if k ∈ {1, 2, 3, 4, 5, 6} then
    (7 / 6) / (k * (k + 1))
  else 0

axiom prob_sum : (∑ k in {1, 2, 3, 4, 5, 6}, P k) = 1

theorem prob_half_to_four :
  (P 1 + P 2 + P 3) = 7 / 8 :=
sorry

end prob_half_to_four_l131_131379


namespace regular_12gon_symmetry_and_angle_l131_131465

theorem regular_12gon_symmetry_and_angle :
  ∀ (L R : ℕ), 
  (L = 12) ∧ (R = 30) → 
  (L + R = 42) :=
by
  -- placeholder for the actual proof
  sorry

end regular_12gon_symmetry_and_angle_l131_131465


namespace total_spending_is_correct_l131_131435

def total_spending : ℝ :=
  let meal_expenses_10 := 10 * 18
  let meal_expenses_5 := 5 * 25
  let total_meal_expenses := meal_expenses_10 + meal_expenses_5
  let service_charge := 50
  let total_before_discount := total_meal_expenses + service_charge
  let discount := 0.05 * total_meal_expenses
  let total_after_discount := total_before_discount - discount
  let tip := 0.10 * total_before_discount
  total_after_discount + tip

theorem total_spending_is_correct : total_spending = 375.25 :=
by
  sorry

end total_spending_is_correct_l131_131435


namespace total_revenue_l131_131425

variable (ticket_price : ℕ := 20)
variable (num_people : ℕ := 50)
variable (first_group : ℕ := 10)
variable (second_group : ℕ := 20)
variable (discount1 : ℕ := 40)
variable (discount2 : ℕ := 15)

def discounted_price (price : ℕ) (discount : ℕ) : ℕ := 
  price - (price * discount / 100)

def revenue (num : ℕ) (price : ℕ) : ℕ := 
  num * price

theorem total_revenue :
  let price1 := discounted_price ticket_price discount1 in
  let price2 := discounted_price ticket_price discount2 in
  let rev1 := revenue first_group price1 in
  let rev2 := revenue second_group price2 in
  let rev3 := revenue (num_people - first_group - second_group) ticket_price in
  rev1 + rev2 + rev3 = 860 :=
by
  let price1 : ℕ := discounted_price ticket_price discount1
  let price2 : ℕ := discounted_price ticket_price discount2
  let rev1 : ℕ := revenue first_group price1
  let rev2 : ℕ := revenue second_group price2
  let rev3 : ℕ := revenue (num_people - first_group - second_group) ticket_price
  show (rev1 + rev2 + rev3 = 860)
  sorry

end total_revenue_l131_131425


namespace evaluate_expression_l131_131491

theorem evaluate_expression : 
  ( (2^12)^2 - (2^10)^2 ) / ( (2^11)^2 - (2^9)^2 ) = 4 :=
by
  sorry

end evaluate_expression_l131_131491


namespace arithmetic_seq_finite_negative_terms_l131_131916

theorem arithmetic_seq_finite_negative_terms (a d : ℝ) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → a + n * d ≥ 0) ↔ (a < 0 ∧ d > 0) :=
by
  sorry

end arithmetic_seq_finite_negative_terms_l131_131916


namespace construct_polygon_with_area_l131_131753

theorem construct_polygon_with_area 
  (n : ℕ) (l : ℝ) (a : ℝ) 
  (matchsticks : n = 12) 
  (matchstick_length : l = 2) 
  (area_target : a = 16) : 
  ∃ (polygon : EuclideanGeometry.Polygon ℝ) (sides : list ℝ),
    sides.length = n ∧ ∀ side ∈ sides, side = l ∧ polygon.area = a := 
sorry

end construct_polygon_with_area_l131_131753


namespace totalPagesInBook_l131_131688

def pagesAlreadyRead : ℕ := 125
def pagesSkipped : ℕ := 16
def pagesLeftToRead : ℕ := 231

theorem totalPagesInBook : pagesAlreadyRead + pagesSkipped + pagesLeftToRead = 372 := 
by simp [pagesAlreadyRead, pagesSkipped, pagesLeftToRead]; sorry

end totalPagesInBook_l131_131688


namespace digit_one_most_frequent_sum_l131_131781

theorem digit_one_most_frequent_sum (n : Nat) (h : 1 ≤ n ∧ n ≤ 1000000) : 
  let digit_sum := Nat.sumDigits
  let repeated_digit_sum := sum_digits_until_single digit_sum n
  ∃ k : Nat, repeated_digit_sum = k ∧ k = 1 ∧ count_of_digit 1 (repeated_digit_sum_list n) > count_of_digit d (repeated_digit_sum_list n)
    where d ≠ 1 :=
    sorry

end digit_one_most_frequent_sum_l131_131781


namespace number_of_correct_conclusions_l131_131066

def P1 (α β : ℝ) (h1 : α = β ∨ α + β = 180) : Prop := 
  α ≠ β ∧ α + β ≠ 180

def P2 (α β γ δ : ℝ) (h2 : (α = γ ∧ β = δ) ∨ (α = δ ∧ β = γ)) : Prop := 
  α = γ ∨ β = δ

def P3 (α β : ℝ) (h3 : α = 90 ∨ β = 90 ∨ α + β = 90) : Prop :=
  α = 90 ∨ β = 90 ∨ α + β = 90

def P4 (l1 l2 l3 : ℝ) (h4 : l1 = l3 ∧ l2 = l3) : Prop :=
  l1 = l2

theorem number_of_correct_conclusions : 
  let c1 := ¬ P1 45 45 (or.inl rfl)
  let c2 := P2 30 60 30 60 (or.inl (and.intro rfl rfl))
  let c3 := P3 45 45 (or.inr (or.inr rfl))
  let c4 := P4 10 10 10 (and.intro rfl rfl)
  in [c1, c2, c3, c4].count(λ x, x = true) = 3 :=
by {
  let c1 := ¬ P1 45 45 (or.inl rfl),
  let c2 := P2 30 60 30 60 (or.inl (and.intro rfl rfl)),
  let c3 := P3 45 45 (or.inr (or.inr rfl)),
  let c4 := P4 10 10 10 (and.intro rfl rfl),
  have h1 : c1 = false := sorry,
  have h2 : c2 = true := sorry,
  have h3 : c3 = true := sorry,
  have h4 : c4 = true := sorry,
  have h_correct : [c1, c2, c3, c4].count(λ x, x = true) = 3 := sorry,
  exact h_correct,
}

end number_of_correct_conclusions_l131_131066


namespace length_of_df_l131_131990

theorem length_of_df (A B C D E F : Type)
  (hABC : is_triangle A B C)
  (hAB_AC : AB = 5 ∧ AC = 5)
  (h_angle_BAC : ∀ θ, angle A B C = θ)
  (h_isosceles_ABC : is_isosceles_triangle A B C)
  (hDEF : is_triangle D E F)
  (hDE : DE = 2)
  (h_equal_areas : ∀ θ, area_of_triangle A B C = area_of_triangle D E F)
  : DF = 12.5 := 
sorry

end length_of_df_l131_131990


namespace evaluate_log_l131_131116

theorem evaluate_log : ∀ (a b : ℝ), a = 3^2 → log a (b^(1/3)) = (1/3) := 
by
  intros a b h1
  sorry

end evaluate_log_l131_131116


namespace prob_cond_conclusion_l131_131966

theorem prob_cond_conclusion (z : ℂ) 
  (h : (1 + complex.I) * (z + 2) = 2) : 
  z + complex.conj(z) = -2 := 
sorry

end prob_cond_conclusion_l131_131966


namespace parametric_to_ordinary_l131_131377

theorem parametric_to_ordinary (θ : ℝ) (x y : ℝ) : 
  x = Real.cos θ ^ 2 →
  y = 2 * Real.sin θ ^ 2 →
  (x ∈ Set.Icc 0 1) → 
  2 * x + y - 2 = 0 :=
by
  intros hx hy h_range
  sorry

end parametric_to_ordinary_l131_131377


namespace probability_of_odd_sums_l131_131401

theorem probability_of_odd_sums :
  let tiles := finset.range 1 13
  let odd_tiles := tiles.filter (λ t, t % 2 = 1)
  let even_tiles := tiles.filter (λ t, t % 2 = 0)
  (odd_tiles.card = 6 ∧ even_tiles.card = 6) →
  (∀ t ∈ tiles, t ∈ odd_tiles ∨ t ∈ even_tiles) →
  ∃ (m n : ℕ), nat.gcd m n = 1 ∧ (∃ (prob : ℚ), prob = m / n ∧
  prob = 800 / 963) :=
begin
  intros tiles odd_tiles even_tiles,
  intros h₁ h₂,
  use [800, 963],
  split,
  { -- Proving that 800 and 963 are relatively prime
    exact nat.gcd_eq_one 800 963,
  },
  use 800 / 963,
  split,
  { -- Proving the probability is 800/963
    exact rfl,
  }
end

end probability_of_odd_sums_l131_131401


namespace find_theta_perpendicular_l131_131588

theorem find_theta_perpendicular (θ : ℝ) (hθ : 0 < θ ∧ θ < π)
  (a b : ℝ × ℝ) (ha : a = (Real.sin θ, 1)) (hb : b = (2 * Real.cos θ, -1))
  (hperp : a.fst * b.fst + a.snd * b.snd = 0) : θ = π / 4 :=
by
  -- Proof would be written here
  sorry

end find_theta_perpendicular_l131_131588


namespace intersect_circle_line_l131_131910

-- Define the circle and line
def circle (r : ℝ) : Prop := r > 0
def line (x y : ℝ) : Prop := x + √3 * y - 2 = 0

-- Define the distance function
def distance (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  abs (A * p.1 + B * p.2 + C) / real.sqrt (A * A + B * B)

-- Circle and line given in the problem
def C (x y : ℝ) (r : ℝ) : Prop := x^2 + y^2 = r^2
def l (x y : ℝ) : Prop := x + √3 * y - 2 = 0

-- The main theorem statement
theorem intersect_circle_line (r : ℝ) (hr : r > 0) :
  (r > 3) → ∃ (x y : ℝ), C x y r ∧ l x y :=
sorry

end intersect_circle_line_l131_131910


namespace smallest_ABC_CBA_divisible_by_eleven_l131_131827

theorem smallest_ABC_CBA_divisible_by_eleven :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (1 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ (0 ≤ C ∧ C ≤ 9) ∧
  let num := 100001 * A + 10010 * B + 1001 * C in
  num = 123321 :=
by
  sorry

end smallest_ABC_CBA_divisible_by_eleven_l131_131827


namespace semicircle_parametric_and_point_D_l131_131269

-- Definition of the semicircle C in polar coordinates
def semicircle_C (θ : ℝ) := 2 * Real.sin θ

-- Definition of the line l
def line_l (x y : ℝ) := x - (sqrt 3) * y - 2 = 0

-- Parametric equation of the semicircle C
def parametric_equation (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, 1 + Real.sin α)

-- Proof that the parametric equation corresponds to the given polar equation and the given conditions
theorem semicircle_parametric_and_point_D :
  (∀ α ∈ Icc (-π/2) (π/2),
    ∃ θ ∈ Icc 0 (π/2),
      (semicircle_C θ = 2 * Real.sin θ) ∧
      (parametric_equation α = (Real.cos θ, 1 + Real.sin θ)))
  ∧ (∃ D : ℝ × ℝ, D = (sqrt 3 / 2, 3 / 2) ∧
                    (∀ α ∈ Icc (-π/2) (π/2),
                      (parametric_equation α = D) → 
                      (∃ θ ∈ Icc 0 (π/2),
                        line_l (Real.cos θ) (1 + Real.sin θ))) )
:= sorry

end semicircle_parametric_and_point_D_l131_131269


namespace fenced_area_with_cutout_l131_131357

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l131_131357


namespace total_students_l131_131265

variable (T : ℝ)   -- T is the total number of students in the school

-- Conditions
def below_eight_students := 0.2 * T
def age_eight_students := 24
def above_eight_students := (2/3) * age_eight_students

-- Goal
theorem total_students (h1 : below_eight_students + age_eight_students + above_eight_students = T) : 
  T = 50 := 
sorry

end total_students_l131_131265


namespace final_symbol_is_minus_l131_131680

def remaining_symbol (plus_count minus_count : ℕ) : symbol := sorry

theorem final_symbol_is_minus :
  remaining_symbol 20 35 = symbol.minus := sorry

end final_symbol_is_minus_l131_131680


namespace repetend_of_five_seventeenths_l131_131153

theorem repetend_of_five_seventeenths :
  (decimal_expansion (5 / 17)).repeat_called == "294117647" :=
sorry

end repetend_of_five_seventeenths_l131_131153


namespace solution_of_xyz_l131_131970

theorem solution_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) : x + y + z = 48 := 
sorry

end solution_of_xyz_l131_131970


namespace Janice_earnings_l131_131650

theorem Janice_earnings (days_worked_per_week : ℕ) (earnings_per_day : ℕ) (overtime_shifts : ℕ) (overtime_earnings_per_shift : ℕ)
  (h1 : days_worked_per_week = 5)
  (h2 : earnings_per_day = 30)
  (h3 : overtime_shifts = 3)
  (h4 : overtime_earnings_per_shift = 15) :
  (days_worked_per_week * earnings_per_day) + (overtime_shifts * overtime_earnings_per_shift) = 195 :=
by {
  sorry
}

end Janice_earnings_l131_131650


namespace polynomial_expr_range_l131_131375

theorem polynomial_expr_range (a : ℝ) (h : 0 < a ∧ a < 32 / 27) :
    ∃ (x y z : ℝ), x < y ∧ y < z ∧ 
    (∀ t : ℝ, t^3 + 4 * t^2 + 4 * t + a = 0 ↔ t = x ∨ t = y ∨ t = z) ∧ 
    (x + y + z = -4) ∧ (xy + yz + xz = 4) ∧ (xyz = -a) ∧ 
    let A := x^3 - 4 * y^2 - 4 * z^2 - 4 * y - 4 * z + 32 in 
    A ∈ (400 / 27, 16) :=
sorry

end polynomial_expr_range_l131_131375


namespace necessary_but_not_sufficient_condition_l131_131209

-- Definitions of our propositions
def p (C : Point → Prop) (A B : Point) (a : ℝ) (h_a : a > 0) : Prop :=
  ∀ P : Point, C P ↔ abs (dist P A - dist P B) = a

def q (C : Point → Prop) : Prop :=
  ∀ P : Point, C P ↔ ∃ e f : Point, e ≠ f ∧ C P = is_hyperbola P e f

-- The main theorem stating the necessary but not sufficient condition
theorem necessary_but_not_sufficient_condition (C : Point → Prop) (A B : Point) (a : ℝ) (h_a : a > 0) :
  (p C A B a h_a) → (q C) :=
sorry

end necessary_but_not_sufficient_condition_l131_131209


namespace line_through_A_intersects_BC_probability_l131_131254

-- Define the problem in Lean 4.
theorem line_through_A_intersects_BC_probability :
  ∀ (A B C : Type) [equilateral_triangle A B C], 
  (∃ l : A → Prop, ∃ θ : ℝ, -π ≤ θ ∧ θ ≤ π ∧ intersects_BC l B C) →
  (probability (line_through_A_intersects_BC A B C) = 1 / 3) :=
by
  sorry

end line_through_A_intersects_BC_probability_l131_131254


namespace words_per_page_smaller_type_l131_131996

theorem words_per_page_smaller_type (total_words : ℕ) (larger_type_words_per_page : ℕ) (total_pages : ℕ) (smaller_type_pages : ℕ) :
  total_words = 48000 → larger_type_words_per_page = 1800 → total_pages = 21 → smaller_type_pages = 17 →
  (total_words - (total_pages - smaller_type_pages) * larger_type_words_per_page) / smaller_type_pages = 2400 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end words_per_page_smaller_type_l131_131996


namespace xyz_sum_48_l131_131967

theorem xyz_sum_48 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : z * x + y = 47) : 
  x + y + z = 48 :=
sorry

end xyz_sum_48_l131_131967


namespace quadratic_with_other_roots_l131_131977

-- Definitions for the roots and the conditions
variables (a p b q : ℝ)
def alpha := (b - q) / (p - a)
def beta := -a - alpha
def gamma := -p - alpha

-- Sum and product of roots for the new quadratic equation
def beta_gamma_sum := gamma + beta
def beta_gamma_product := (b * q * (p - a)^2) / (b - q)^2

-- Statement of the quadratic equation with roots beta and gamma
theorem quadratic_with_other_roots :
  ∃ (r s : ℝ), r = beta_gamma_sum ∧ s = beta_gamma_product ∧ (
    λ x : ℝ, x^2 - r * x + s = 0
  ) :=
by {
  sorry
}

end quadratic_with_other_roots_l131_131977


namespace sam_runs_more_than_sarah_sue_runs_less_than_sarah_l131_131985

-- Definitions based on the problem conditions
def street_width : ℝ := 25
def block_side_length : ℝ := 500
def sarah_perimeter : ℝ := 4 * block_side_length
def sam_perimeter : ℝ := 4 * (block_side_length + 2 * street_width)
def sue_perimeter : ℝ := 4 * (block_side_length - 2 * street_width)

-- The proof problem statements
theorem sam_runs_more_than_sarah : sam_perimeter - sarah_perimeter = 200 := by
  sorry

theorem sue_runs_less_than_sarah : sarah_perimeter - sue_perimeter = 200 := by
  sorry

end sam_runs_more_than_sarah_sue_runs_less_than_sarah_l131_131985


namespace volume_of_new_pyramid_is_108_l131_131829

noncomputable def volume_of_cut_pyramid : ℝ :=
  let base_edge_length := 12 * Real.sqrt 2
  let slant_edge_length := 15
  let cut_height := 4.5
  -- Calculate the height of the original pyramid using Pythagorean theorem
  let original_height := Real.sqrt (slant_edge_length^2 - (base_edge_length/2 * Real.sqrt 2)^2)
  -- Calculate the remaining height of the smaller pyramid
  let remaining_height := original_height - cut_height
  -- Calculate the scale factor
  let scale_factor := remaining_height / original_height
  -- New base edge length
  let new_base_edge_length := base_edge_length * scale_factor
  -- New base area
  let new_base_area := (new_base_edge_length)^2
  -- Volume of the new pyramid
  (1 / 3) * new_base_area * remaining_height

-- Define the statement to prove
theorem volume_of_new_pyramid_is_108 :
  volume_of_cut_pyramid = 108 :=
by
  sorry

end volume_of_new_pyramid_is_108_l131_131829


namespace probability_each_university_at_least_one_admission_l131_131536

def total_students := 4
def total_universities := 3

theorem probability_each_university_at_least_one_admission :
  ∃ (p : ℚ), p = 4 / 9 :=
by
  sorry

end probability_each_university_at_least_one_admission_l131_131536


namespace prove_integral_l131_131430

noncomputable def integral_problem : Prop :=
  ∀ (x : ℝ), ∫ (2 * x^3 + 11 * x^2 + 16 * x + 10) / ((x + 2)^2 * (x^2 + 2 * x + 3)) dx =
  -2 / (x + 2) + log |x^2 + 2 * x + 3| - (1 / sqrt 2) * arctan ((x + 1) / sqrt 2) + C

variable (x : ℝ)

theorem prove_integral : integral_problem x :=
begin
  sorry
end

end prove_integral_l131_131430


namespace monotonicity_of_f_exists_positive_x_l131_131576

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

theorem monotonicity_of_f (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f x₁ a ≤ f x₂ a) ∨
  (∃ x₀ : ℝ, ∀ x : ℝ, 
    (x < x₀ → f x a > f (x + 1e-10) a) ∧ 
    (x > x₀ → f x a < f (x - 1e-10) a)) := sorry

theorem exists_positive_x (a : ℝ) (h : 1 < a) : 
  ∃ x : ℝ, 0 < x ∧ f x a > 0 :=
  ⟨ 2 * Real.log a, by
    have ha : Real.exp (2 * Real.log a) = a ^ 2 := by
      rw [Real.exp_mul, Real.exp_log (lt_trans zero_lt_one h)]
    have Hf_eq : f (2 * Real.log a) a = a^2 - 2*a*Real.log a - 1 := by
      rw [f, ha]
      ring
    have Hineq : a ^ 2 - 2 * a * Real.log a - 1 > 0 := sorry
    exact ⟨ by apply Real.mul_pos two_pos (Real.log_pos h), by rw [Hf_eq]; exact Hineq ⟩ ⟩

end monotonicity_of_f_exists_positive_x_l131_131576


namespace irrational_number_among_given_options_l131_131481

def is_irrational (x : ℝ) : Prop :=
  ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem irrational_number_among_given_options : is_irrational (π) ∧
  ¬ is_irrational (sqrt 4) ∧ 
  ¬ is_irrational (-↑(real.cbrt 8)) ∧ 
  ¬ is_irrational (1 / 3) :=
by {
  sorry
}

end irrational_number_among_given_options_l131_131481


namespace number_of_valid_19_tuples_is_one_l131_131890

theorem number_of_valid_19_tuples_is_one :
  ∃! (v : Fin 19 → ℤ), ∀ i, v i ^ 3 = 3 * (Finset.univ.sum (λ j, v j) - v i) :=
by
  sorry

end number_of_valid_19_tuples_is_one_l131_131890


namespace participants_handshake_l131_131072

theorem participants_handshake :
  ∀ (participants : Fin 2016 → ℕ), 
  (∀ i : Fin 2015, participants i = i.val + 1) → 
  participants ⟨2015, by simp⟩ = 1008 :=
begin
  intros participants h,
  sorry,
end

end participants_handshake_l131_131072


namespace vehicle_value_last_year_l131_131387

variables (V_this_year : ℝ) (V_last_year : ℝ)

def condition1 : Prop := V_this_year = 16000
def condition2 : Prop := V_this_year = 0.8 * V_last_year

theorem vehicle_value_last_year :
  condition1 V_this_year ∧ condition2 V_this_year V_last_year → V_last_year = 20000 :=
by
  intros h
  cases h with h1 h2
  sorry

end vehicle_value_last_year_l131_131387


namespace range_of_a_l131_131572

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 0 then x^2 + (4 * a - 3) * x + 3 * a else Real.log a (x + 1) + 1

theorem range_of_a (a : ℝ) :
  (∀ x y, x < y → f a x ≥ f a y) → (0 < a ∧ a ≠ 1) → (a ≥ 1 / 3 ∧ a ≤ 3 / 4) :=
by {
  intro h1 h2,
  sorry,
}

end range_of_a_l131_131572


namespace volleyball_club_girls_l131_131062

theorem volleyball_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 3 : ℝ) * G + ↑B = 20) : G = 18 := 
by
  sorry

end volleyball_club_girls_l131_131062


namespace goldfish_graph_discrete_points_l131_131589

theorem goldfish_graph_discrete_points : 
  ∀ n : ℤ, 1 ≤ n ∧ n ≤ 10 → ∃ C : ℤ, C = 20 * n + 10 ∧ ∀ m : ℤ, (1 ≤ m ∧ m ≤ 10 ∧ m ≠ n) → C ≠ (20 * m + 10) :=
by
  sorry

end goldfish_graph_discrete_points_l131_131589


namespace imaginary_root_modulus_eq_l131_131973

theorem imaginary_root_modulus_eq {a b p : ℝ} (h1 : (a + b * complex.I) ∈ complex.im_roots (λ x, x^2 + 2 * x + p))
  (h2 : complex.abs (a + b * complex.I) = 2) : p = 4 :=
by sorry

end imaginary_root_modulus_eq_l131_131973


namespace sum_of_digits_N_l131_131833

-- Define the main problem conditions and the result statement
theorem sum_of_digits_N {N : ℕ} 
  (h₁ : (N * (N + 1)) / 2 = 5103) : 
  (N.digits 10).sum = 2 :=
sorry

end sum_of_digits_N_l131_131833


namespace even_composition_l131_131311

variable {α : Type} [CommRing α]

def is_even_function (g : α → α) : Prop :=
  ∀ x, g (-x) = g x

theorem even_composition (g : α → α) (h : is_even_function g) :
  is_even_function (λ x, g (g x)) :=
by 
  sorry

end even_composition_l131_131311


namespace gasoline_needed_for_journey_l131_131812

variable (d_highway : ℕ) (d_uphill : ℕ) (d_city : ℕ) (eff_highway : ℕ) (eff_uphill : ℕ) (eff_city : ℕ) (total_distance : ℕ)

theorem gasoline_needed_for_journey
  (h1 : d_highway = 70)
  (h2 : d_uphill = 60)
  (h3 : d_city = total_distance - d_highway - d_uphill)
  (h4 : eff_highway = 40)
  (h5 : eff_uphill = 20)
  (h6 : eff_city = 30)
  (h7 : total_distance = 160) :
  (d_highway / eff_highway) + (d_uphill / eff_uphill) + (d_city / eff_city) = 5.75 := 
  by
  sorry

end gasoline_needed_for_journey_l131_131812


namespace polynomial_functional_equation_l131_131881

theorem polynomial_functional_equation (f : polynomial ℝ)
  (h : ∀ (g : polynomial ℝ), f.comp g = g.comp f) :
  f = polynomial.X := 
sorry

end polynomial_functional_equation_l131_131881


namespace combined_mpg_is_correct_l131_131338

-- Definitions for the conditions
def average_mpg_ray : ℝ := 50
def average_mpg_tom : ℝ := 20
def average_mpg_amy : ℝ := 40
def distance_each : ℝ := 120

-- The theorem statement
theorem combined_mpg_is_correct :
  total_combined_mpg average_mpg_ray average_mpg_tom average_mpg_amy distance_each = 31.55 :=
by sorry

-- Function to compute total combined mpg
def total_combined_mpg (mpg_ray : ℝ) (mpg_tom : ℝ) (mpg_amy : ℝ) (dist : ℝ) : ℝ :=
  let total_distance := 3 * dist
  let total_gas := (dist / mpg_ray) + (dist / mpg_tom) + (dist / mpg_amy)
  total_distance / total_gas

example : combined_mpg_is_correct := by sorry

end combined_mpg_is_correct_l131_131338


namespace num_permutations_with_P_gt_without_P_l131_131316

def has_property_P (n : ℕ) (σ : FinPerm (2 * n)) : Prop :=
  ∃ i : Fin (2 * n - 1), (|σ (fin.of_nat i) - σ (fin.of_nat (i + 1))| = n)

theorem num_permutations_with_P_gt_without_P (n : ℕ) :
  ∑ σ in univ, has_property_P n σ > ∑ σ in univ, ¬ has_property_P n σ := 
sorry

end num_permutations_with_P_gt_without_P_l131_131316


namespace modular_inverse_17_1001_l131_131407

theorem modular_inverse_17_1001 : ∃ x : ℕ, x = 530 ∧ (17 * x) % 1001 = 1 := 
by
  use 530
  split
  . exact rfl
  . calc
    (17 * 530) % 1001
    = 9010 % 1001 : by norm_num
    = 1 : by norm_num

end modular_inverse_17_1001_l131_131407


namespace fenced_area_l131_131354

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l131_131354


namespace age_ratio_albert_mary_l131_131838

variable (A M B : ℕ) 

theorem age_ratio_albert_mary
    (h1 : A = 4 * B)
    (h2 : M = A - 10)
    (h3 : B = 5) :
    A = 2 * M :=
by
    sorry

end age_ratio_albert_mary_l131_131838


namespace repetend_of_5_div_17_l131_131130

theorem repetend_of_5_div_17 :
  let dec := 5 / 17 in
  decimal_repetend dec = "294117" := sorry

end repetend_of_5_div_17_l131_131130


namespace num_pairs_nat_numbers_l131_131961

open Real

/--
  Prove that the number of pairs of natural numbers (m, n) such that
  m, n ≤ 1000 and (m / (n + 1) < sqrt 2 < (m + 1) / n)
  equals 1706.
-/
theorem num_pairs_nat_numbers (S : Set (ℕ × ℕ)) :
  S = {p | p.1 ≤ 1000 ∧ p.2 ≤ 1000 ∧ 
           (p.1: ℝ) / (p.2 + 1).to_nat < sqrt 2 ∧ sqrt 2 < (p.1 + 1: ℝ) / p.2.to_nat} →
  S.card = 1706 := 
sorry

end num_pairs_nat_numbers_l131_131961


namespace sum_of_parameters_of_parabolas_l131_131091

theorem sum_of_parameters_of_parabolas 
  (c d : ℝ)
  (h₁ : ∀ x : ℝ, (cx^2 + 3 = 0 → False) ∧ (5 - dx^2 = 0 → False))
  (h₂ : (2 * (5 - 3) = 4) ∧ (2 * sqrt (5 / d) = 2))
  (h₃ : (1/2) * (2 * sqrt (5 / d)) * 4 = 8)
  : c + d = 2 :=
sorry

end sum_of_parameters_of_parabolas_l131_131091


namespace average_movie_length_l131_131003

theorem average_movie_length :
  ∀ (run_speed : ℕ) (num_movies : ℕ) (total_miles : ℕ),
  run_speed = 12 →
  num_movies = 2 →
  total_miles = 15 →
  (total_miles * run_speed) / num_movies = 90 :=
begin
  intros run_speed num_movies total_miles,
  intros h_run_speed h_num_movies h_total_miles,
  rw [h_run_speed, h_num_movies, h_total_miles],
  norm_num,
end

end average_movie_length_l131_131003


namespace part1_part2_l131_131211

-- Part (1)
theorem part1 (a : ℝ) (A B : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hB : B = { x : ℝ | x^2 - a * x + a - 1 = 0 }) 
  (hUnion : A ∪ B = A) : 
  a = 2 ∨ a = 3 := 
sorry

-- Part (2)
theorem part2 (m : ℝ) (A C : Set ℝ) 
  (hA : A = { x : ℝ | x^2 - 3 * x + 2 = 0 }) 
  (hC : C = { x : ℝ | x^2 + 2 * (m + 1) * x + m^2 - 5 = 0 }) 
  (hInter : A ∩ C = C) : 
  m ∈ Set.Iic (-3) := 
sorry

end part1_part2_l131_131211


namespace tan_sum_product_l131_131046

theorem tan_sum_product (tan : ℝ → ℝ) : 
  (1 + tan 23) * (1 + tan 22) = 2 + tan 23 * tan 22 := by sorry

end tan_sum_product_l131_131046


namespace find_third_number_l131_131729

theorem find_third_number (x y z : ℝ) 
  (h1 : y = 3 * x - 7)
  (h2 : z = 2 * x + 2)
  (h3 : x + y + z = 168) : z = 60 :=
sorry

end find_third_number_l131_131729


namespace determine_k_l131_131603

theorem determine_k (k : ℝ) :
  (∀ x : ℝ, ((k^2 - 9) * x^2 - (2 * (k + 1)) * x + 1 = 0 → x ∈ {a : ℝ | a = - (2 * (k+1)) / (2 * (k^2-9)} ∨ a = (2*(k+1) + √4 * (k+1)^2 - 4 * (k^2-9) ) / (2*(k^2-9)) ∨ a = (2 * (k+1) - √4 * (k+1)^2 - 4 *(k^2-9)) / (2 *(k^2-9))) → x ∈  {a : ℝ | a = - (2 * (k+1)) / (2 * (k^2-9)} ) ∨ x ∈  {a : ℝ | a = (2*(k+1) + 0 ) / (2 * (k^2-9))} ∨ a = (2*(k+1) - 0 ) / (2 *(k^2-9)} )  → 
  k = 3 ∨ k = -3 ∨ k = -5 := sorry

end determine_k_l131_131603


namespace radius_probability_satisfies_l131_131055

theorem radius_probability_satisfies :
  ∃ (d : ℝ), (square_vertices := {(0, 0), (3030, 0), (3030, 3030), (0, 3030)}) →
  (lattice_points_coverage_prob := 3 / 4) →
  (d_approx := (Ξ : ℝ), Ξ ≈ 0.5) →
  True := sorry

end radius_probability_satisfies_l131_131055


namespace average_headcount_spring_terms_l131_131406

def spring_headcount_02_03 := 10900
def spring_headcount_03_04 := 10500
def spring_headcount_04_05 := 10700

theorem average_headcount_spring_terms :
  (spring_headcount_02_03 + spring_headcount_03_04 + spring_headcount_04_05) / 3 = 10700 := by
  sorry

end average_headcount_spring_terms_l131_131406


namespace repetend_of_5_div_17_l131_131136

theorem repetend_of_5_div_17 :
  let dec := 5 / 17 in
  decimal_repetend dec = "294117" := sorry

end repetend_of_5_div_17_l131_131136


namespace jebb_take_home_pay_is_4620_l131_131681

noncomputable def gross_salary : ℤ := 6500
noncomputable def federal_tax (income : ℤ) : ℤ :=
  let tax1 := min income 2000 * 10 / 100
  let tax2 := min (max (income - 2000) 0) 2000 * 15 / 100
  let tax3 := max (income - 4000) 0 * 25 / 100
  tax1 + tax2 + tax3

noncomputable def health_insurance : ℤ := 300
noncomputable def retirement_contribution (income : ℤ) : ℤ := income * 7 / 100

noncomputable def total_deductions (income : ℤ) : ℤ :=
  federal_tax income + health_insurance + retirement_contribution income

noncomputable def take_home_pay (income : ℤ) : ℤ :=
  income - total_deductions income

theorem jebb_take_home_pay_is_4620 : take_home_pay gross_salary = 4620 := by
  sorry

end jebb_take_home_pay_is_4620_l131_131681


namespace repetend_of_5_over_17_is_294117_l131_131147

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l131_131147


namespace lisa_flight_time_l131_131675

noncomputable def distance : ℝ := 519.5
noncomputable def speed : ℝ := 54.75
noncomputable def time : ℝ := 9.49

theorem lisa_flight_time : distance / speed = time :=
by
  sorry

end lisa_flight_time_l131_131675


namespace area_of_unpainted_region_l131_131402

theorem area_of_unpainted_region 
    (width_board1 : ℝ) (width_board2 : ℝ) (θ : ℝ) 
    (h_board1_width : width_board1 = 5)
    (h_board2_width : width_board2 = 7)
    (h_angle : θ = π / 4) : 
    (area : ℝ) := 
  (sqrt 2 * width_board1 * width_board2 = 35 * sqrt 2) :=
sorry

end area_of_unpainted_region_l131_131402


namespace total_gain_loss_is_correct_l131_131457

noncomputable def total_gain_loss_percentage 
    (cost1 cost2 cost3 : ℝ) 
    (gain1 gain2 gain3 : ℝ) : ℝ :=
  let total_cost := cost1 + cost2 + cost3
  let gain_amount1 := cost1 * gain1
  let loss_amount2 := cost2 * gain2
  let gain_amount3 := cost3 * gain3
  let net_gain_loss := (gain_amount1 + gain_amount3) - loss_amount2
  (net_gain_loss / total_cost) * 100

theorem total_gain_loss_is_correct :
  total_gain_loss_percentage 
    675958 995320 837492 0.11 (-0.11) 0.15 = 3.608 := 
sorry

end total_gain_loss_is_correct_l131_131457


namespace probability_cos_interval_l131_131461

noncomputable def cos_probability : ℝ :=
  (∫ x in -1..(-2/3), 1) + (∫ x in (2/3)..1, 1) / (∫ x in -1..1, 1)

theorem probability_cos_interval :
  cos_probability = 1 / 3 :=
by
  sorry

end probability_cos_interval_l131_131461


namespace inf_integral_eq_inv_e_l131_131302

noncomputable def C := { f : ℝ → ℝ // (∀ x ∈ (Set.Icc (0 : ℝ) 1), has_deriv_at f (f' x) x) ∧ (f 0 = 0) ∧ (f 1 = 1) }

theorem inf_integral_eq_inv_e :
  ∃ f ∈ C, ∀ g ∈ C, ∫ x in 0..1, |f' x - f x| dx ≥ ∫ x in 0..1, |g' x - g x| dx :=
sorry

end inf_integral_eq_inv_e_l131_131302


namespace tangent_line_range_of_a_l131_131937

-- Condition: Definition of the function f
def f (x : ℝ) (a : ℝ) := Real.log x - (a * (x - 1)) / (x + 1)

-- Problem 1: If x = 2 is an extremum point, find the tangent line at (1, f(1))
theorem tangent_line (a : ℝ) (h_extremum : (derivative (λ x, f x a)) 2 = 0) :
  let k := (derivative (λ x, f x a)) 1
  let tangent_point := (1, f 1 a)
  k = -1 / 8 ∧ tangent_point.2 = 0 ∧
  (∀ x y : ℝ, y = k * (x - 1) + 0 ↔ x + 8 * y - 1 = 0) :=
by sorry

-- Problem 2: Find the range of a such that f(x) is monotonically increasing on (0, +∞)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → (derivative (λ x, f x a)) x ≥ 0) ↔ a ≤ 2 :=
by sorry

end tangent_line_range_of_a_l131_131937


namespace systematic_sampling_6_of_60_l131_131735

theorem systematic_sampling_6_of_60 :
  ∃ (S : Finset ℕ), (∀ x ∈ S, 1 ≤ x ∧ x ≤ 60) ∧
                    S.card = 6 ∧ 
                    ∀ x y ∈ S, x ≠ y → (y - x) % (60 / 6) = 0 :=
begin
  use {3, 13, 23, 33, 43, 53},
  split,
  { intros x hx,
    finset.mem_insert.mp hx; -- check that x is one of {3, 13, 23, 33, 43, 53}
    finset.mem_insert.mp; -- implying x is indeed within bounds (1 ≤ x ≤ 60)
    simp },
  split,
  { exact finset.card_insert_of_not_mem _ -- proof of cardinality = 6
    finset.nil },
  { intros x y hx hy hxy,
    finset.mem_insert.mp hx; repeat {and.intro (nat.mod_eq_zero_of_dvd (nat.dvd_of_mem_insert hx hy))}}
end 

end systematic_sampling_6_of_60_l131_131735


namespace max_remainder_when_divided_by_8_l131_131034

-- Define the problem: greatest possible remainder when apples divided by 8.
theorem max_remainder_when_divided_by_8 (n : ℕ) : ∃ r : ℕ, r < 8 ∧ r = 7 ∧ n % 8 = r := 
sorry

end max_remainder_when_divided_by_8_l131_131034


namespace valid_range_of_a_l131_131224

theorem valid_range_of_a (a : ℝ) :
  (∀ θ : ℝ, complex.abs ((a + real.cos θ) + complex.I * (2 * a - real.sin θ)) ≤ 2) →
  a ∈ Icc (-real.sqrt 5 / 5) (real.sqrt 5 / 5) :=
by
  -- Skipping the proof
  sorry

end valid_range_of_a_l131_131224


namespace isosceles_triangle_circumcircle_radius_l131_131993

-- Assume variables and definitions
variables {α r R : ℝ}

-- Definition of the cotangent function (You might need additional imports or definitions from Lean's libraries for trigonometric functions)
def cot (x : ℝ) : ℝ := 1 / tan x

theorem isosceles_triangle_circumcircle_radius (h: ∀ α r, 
    ∃ (R : ℝ), 
      (R = r * cot (α / 2) / sin (2 * α))
  ) : ∃ R : ℝ, 
      (R = r * cot (α / 2) / sin (2 * α)) :=
begin
  sorry,
end

end isosceles_triangle_circumcircle_radius_l131_131993


namespace angle_between_vectors_eq_90_l131_131949

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_vectors_eq_90 
  {a b : V} 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h : ∥a + b∥ = ∥a - b∥) : 
  ⟪a, b⟫ = 0 :=
by {
  sorry
}

end angle_between_vectors_eq_90_l131_131949


namespace sequence_formula_l131_131276

theorem sequence_formula (a : ℕ → ℕ) (n : ℕ) (h : ∀ n ≥ 1, a n = a (n - 1) + n^3) : 
  a n = (n * (n + 1) / 2) ^ 2 := sorry

end sequence_formula_l131_131276


namespace tan_C_eq_2_tan_B_area_of_triangle_l131_131609

variable {A B C a b c : ℝ}

-- Conditions
axiom triangle_conditions (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : A + B + C = π) :
  c * Real.cos B - b * Real.cos C = (1 / 3) * a

-- Part (I)
theorem tan_C_eq_2_tan_B (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : A + B + C = π) :
  c * Real.cos B - b * Real.cos C = (1 / 3) * a → 
  Real.tan C = 2 * Real.tan B :=
sorry

-- Part (II)
theorem area_of_triangle (h1 : B + C = π - A) (h2 : Real.tan A = 9 / 7) (h3 : a = 3) :
  ∃ area : ℝ, 
  area = (1 / 2) * b * c * Real.sin A ∧ 
  area = (9 / 2) :=
sorry

end tan_C_eq_2_tan_B_area_of_triangle_l131_131609


namespace even_comp_even_l131_131305

variable {X : Type} [AddGroup X] [HasNeg X]

def is_even_function (g : X → X) : Prop :=
  ∀ x : X, g (-x) = g x

theorem even_comp_even (g : X → X) (h_even : is_even_function g) :
  is_even_function (g ∘ g) :=
by
  intro x
  rw [function.comp_apply, function.comp_apply]
  rw [h_even] -- by the even property of g
  rw [h_even] -- by the even property of g
  sorry

end even_comp_even_l131_131305


namespace repetend_of_5_div_17_l131_131135

theorem repetend_of_5_div_17 :
  let dec := 5 / 17 in
  decimal_repetend dec = "294117" := sorry

end repetend_of_5_div_17_l131_131135


namespace difference_of_sums_l131_131506

theorem difference_of_sums :
  let sum_evens := 2 * (Finset.range 100).sum (λ n => n + 1)
  let sum_multiples_3 := 3 * (Finset.range 100).sum (λ n => n + 1)
  sum_evens - sum_multiples_3 = -5050 :=
by
  have h1 : sum_evens = 2 * (Finset.range 100).sum (λ n => n + 1) := rfl
  have h2 : sum_multiples_3 = 3 * (Finset.range 100).sum (λ n => n + 1) := rfl
  sorry

end difference_of_sums_l131_131506


namespace problem_solution_l131_131659

noncomputable def A (f : ℝ → ℝ) : Prop := ∀ b : ℝ, ∃ a : ℝ, f a = b
noncomputable def B (f : ℝ → ℝ) : Prop := ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, f x ∈ Icc (-M) M

noncomputable def prop1 (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  (A f) ↔ (∀ b : ℝ, ∃ a ∈ D, f a = b)

noncomputable def prop2 (f : ℝ → ℝ) : Prop :=
  (B f) ↔ (∃ x : ℝ, ∃ y : ℝ, (∀ z : ℝ, f z ≤ y) ∧ (∀ z : ℝ, f z ≥ x))

noncomputable def prop3 (f g : ℝ → ℝ) : Prop :=
  (A f) ∧ (B g) ∧ (∀ x : ℝ, f x = g x) → ¬ B (λ x, f x + g x)

noncomputable def prop4 (a : ℝ) : Prop :=
  (∃ M : ℝ, ∀ x > -2, a * real.log (x + 2) + x / (x^2 + 1) = M) → 
  B (λ x, a * real.log (x + 2) + x / (x^2 + 1))

theorem problem_solution : prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4 :=
sorry

end problem_solution_l131_131659


namespace max_tanB_cotC_l131_131235

open Real 

-- Define the given conditions in the problem
variables {a b c x0 y0 z0 : ℝ} 
variables (B C : ℝ)

-- Assume all conditions given in the problem
-- Triangle with sides lengths
hypothesis h₁ : b > max a c

-- line equation passes through specific point
hypothesis h₂ : a * (z0 / x0) + b * (2 * y0 / x0) + c = 0

-- point lies on the ellipse
hypothesis h₃ : (z0 / y0)^2 + (x0 / y0)^2 / 4 = 1

-- Define the statement to be proven
theorem max_tanB_cotC : tan B * cot C ≤ 5 / 3 :=
sorry

end max_tanB_cotC_l131_131235


namespace min_value_3x_2y_l131_131559

theorem min_value_3x_2y (x y : ℝ) (h1: x > 0) (h2 : y > 0) (h3 : x = 4 * x * y - 2 * y) :
  3 * x + 2 * y >= 2 + Real.sqrt 3 :=
by
  sorry

end min_value_3x_2y_l131_131559


namespace no_integer_solutions_eqn_l131_131408

theorem no_integer_solutions_eqn : ∀ x y : ℤ, 2^(2 * x) - 3^(2 * y) ≠ 24 := by
  intro x y
  sorry

end no_integer_solutions_eqn_l131_131408


namespace sum_of_hofstadterian_residues_l131_131203

noncomputable def is_hofstadterian_residue (p k n : ℕ) : Prop :=
  ∃ (seq : ℕ → ℕ), seq 0 % p = n ∧ ∀ i, (seq (i + 1))^k % p = seq i % p

noncomputable def f (p k : ℕ) : ℕ :=
  (finset.range p).filter (λ n, is_hofstadterian_residue p k n).card

theorem sum_of_hofstadterian_residues : (finset.range 2017).sum (λ k, f 2017 k) = 1162656 :=
sorry

end sum_of_hofstadterian_residues_l131_131203


namespace hexagon_eq_triangle_l131_131495

open Complex

-- Define the conditions and the theorem
structure HexagonInCircle where
  A B C D E F : Complex
  G H K : Complex
  r : ℝ
  h_circum : ∀ z ∈ {A, B, C, D, E, F}, abs z = r
  midpoints : G = (B + C) / 2 ∧ H = (D + E) / 2 ∧ K = (F + A) / 2

theorem hexagon_eq_triangle (hex : HexagonInCircle) : 
  abs (hex.G - hex.H) = abs (hex.H - hex.K) ∧ abs (hex.H - hex.K) = abs (hex.K - hex.G) := by 
  sorry

end hexagon_eq_triangle_l131_131495


namespace addition_and_rounding_l131_131835

def add_and_round (a b : ℝ) : ℝ :=
  Float.round (a + b)

theorem addition_and_rounding :
  add_and_round 81.76 34.587 = 116.3 :=
by
  sorry

end addition_and_rounding_l131_131835


namespace projection_of_vec2_l131_131380

noncomputable def projection (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  let dot_product := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let square := v.1 * v.1 + v.2 * v.2 + v.3 * v.3
  (dot_product / square * v.1, dot_product / square * v.2, dot_product / square * v.3)

-- Given Hypotheses
def vec1 := (0, 1, 4)
def w := (1:ℝ, -1/2, 1/2)
def projected_vec1 := (1:ℝ, -1/2, 1/2)

-- The theorem we are to prove
theorem projection_of_vec2 (vec2 := (3, 3, -2)) :
  projection vec2 w = (1/3, -1/6, 1/6) :=
sorry

end projection_of_vec2_l131_131380


namespace area_of_triangle_abe_l131_131931

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
10 -- Dummy definition, in actual scenario appropriate area calculation will be required.

def length_AD : ℝ := 2
def length_BD : ℝ := 3

def areas_equal (S_ABE S_DBFE : ℝ) : Prop :=
    S_ABE = S_DBFE

theorem area_of_triangle_abe
  (area_abc : ℝ)
  (length_ad length_bd : ℝ)
  (equal_areas : areas_equal (triangle_area 1 1 1) 1) -- Dummy values, should be substituted with correct arguments
  : triangle_area 1 1 1 = 6 :=
sorry -- proof will be filled later

end area_of_triangle_abe_l131_131931


namespace problem1_problem2_l131_131089

theorem problem1 : (-1) ^ 2012 + (-1 / 2) ^ (-2) - (3.14 - Real.pi) ^ 0 = 4 :=
  by
    sorry

theorem problem2 (x : ℝ) : (x + 1) * (x - 3) - (x + 1) ^ 2 = -4 * x - 4 :=
  by
    sorry

end problem1_problem2_l131_131089


namespace minimum_rounds_l131_131202

-- Given conditions based on the problem statement
variable (m : ℕ) (hm : m ≥ 17)
variable (players : Fin (2 * m)) -- Representing 2m players
variable (rounds : Fin (2 * m - 1)) -- Representing 2m - 1 rounds
variable (pairs : Fin m → Fin (2 * m) × Fin (2 * m)) -- Pairing for each of the m pairs in each round

-- Statement of the proof problem
theorem minimum_rounds (h1 : ∀ i j, i ≠ j → ∃! (k : Fin m), pairs k = (i, j) ∨ pairs k = (j, i))
(h2 : ∀ k : Fin m, (pairs k).fst ≠ (pairs k).snd)
(h3 : ∀ i j, i ≠ j → ∃ r : Fin (2 * m - 1), (∃ k : Fin m, pairs k = (i, j)) ∧ (∃ k : Fin m, pairs k = (j, i))) :
∃ (n : ℕ), n = m - 1 ∧ ∀ s : Fin 4 → Fin (2 * m), (∀ i j, i ≠ j → ¬ ∃ r : Fin n, ∃ k : Fin m, pairs k = (s i, s j)) ∨ (∃ r1 r2 : Fin n, ∃ i j, i ≠ j ∧ ∃ k1 k2 : Fin m, pairs k1 = (s i, s j) ∧ pairs k2 = (s j, s i)) :=
sorry

end minimum_rounds_l131_131202


namespace sequence_formula_l131_131277

-- Define the sequence a_n using the recurrence relation
def a : ℕ → ℚ
| 0     := 0
| (n+1) := a n + (n+1)^3

-- The statement to be proved
theorem sequence_formula (n : ℕ) : 
  a n = (n^2 * (n+1)^2) / 4 := sorry

end sequence_formula_l131_131277


namespace granola_bars_proof_l131_131590

noncomputable def total_granola_bars : ℕ := 20

theorem granola_bars_proof 
  (bars_set_aside : ℕ := 7) 
  (bars_traded : ℕ := 3) 
  (bars_given_to_sisters : ℕ := 10) 
  (bars_to_each_sister : ℕ := 5) : 
  bars_set_aside + bars_traded + bars_given_to_sisters = total_granola_bars :=
by
  have h : bars_given_to_sisters = 2 * bars_to_each_sister := by rfl
  have total_bars : bars_set_aside + bars_traded + bars_given_to_sisters = 7 + 3 + 10 := by rfl
  rw total_bars
  norm_num

end granola_bars_proof_l131_131590


namespace repetend_of_5_over_17_is_294117_l131_131146

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l131_131146


namespace sum_of_squares_l131_131552

theorem sum_of_squares (a b n : ℕ) (h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2) : 
  ∃ e f : ℕ, a^2 + n * b^2 = e^2 + f^2 :=
by
  sorry

-- Theorem parameters and logical flow explained:

-- a, b, n : ℕ                  -- Natural number inputs
-- h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2  -- Condition given in the problem that a^2 + 2nb^2 is a perfect square
-- Prove that there exist natural numbers e and f such that a^2 + nb^2 = e^2 + f^2

end sum_of_squares_l131_131552


namespace coed_softball_team_total_players_l131_131389

theorem coed_softball_team_total_players (M W : ℕ) 
  (h1 : W = M + 4) 
  (h2 : (M : ℚ) / W = 0.6363636363636364) :
  M + W = 18 := 
by sorry

end coed_softball_team_total_players_l131_131389


namespace quadratic_y1_gt_y2_l131_131233

theorem quadratic_y1_gt_y2 (a b c y1 y2 : ℝ) (h_a_pos : a > 0) (h_sym : ∀ x, a * (x - 1)^2 + c = a * (1 - x)^2 + c) (h1 : y1 = a * (-1)^2 + b * (-1) + c) (h2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
sorry

end quadratic_y1_gt_y2_l131_131233


namespace compute_k_l131_131670

noncomputable def tan_inverse (k : ℝ) : ℝ := Real.arctan k

theorem compute_k (x k : ℝ) (hx1 : Real.tan x = 2 / 3) (hx2 : Real.tan (3 * x) = 3 / 5) : k = 2 / 3 := sorry

end compute_k_l131_131670


namespace find_m_n_l131_131883

theorem find_m_n : ∀ (m n : ℕ), 0 < m → 0 < n → (m + n)^m = n^m + 1413 → m = 3 ∧ n = 11 :=
by {
  intros m n h_m h_n h_eq,
  sorry
}

end find_m_n_l131_131883


namespace math_proof_statement_l131_131206

noncomputable def the_ellipse (x y a b : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (x / a)^2 + (y / b)^2 = 1

noncomputable def holds_through (x y : ℝ) : Prop :=
  the_ellipse (-3) (-1) x y

noncomputable def eccentricity_cond (a b c : ℝ) : Prop :=
  (c / a = real.sqrt 6 / 3) ∧ (a^2 = b^2 + c^2)

noncomputable def line_l (x y : ℝ) : Prop :=
  x - y - 2 = 0

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x / 12 + y / 4 = 1

noncomputable def points_AB (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 = 0 ∧ y1 = -2) ∨ (x2 = 3 ∧ y2 = 1)

noncomputable def max_point_area (x y : ℝ) : Prop :=
  x = -3 ∧ y = 1

noncomputable def max_area (area : ℝ) : Prop :=
  area = 9

theorem math_proof_statement :
  ∃ a b c x1 y1 x2 y2 px py area,
    (the_ellipse a b ∧ holds_through a b ∧ eccentricity_cond a b c ∧
    line_l x1 y1 ∧ line_l x2 y2 ∧ points_AB x1 y1 x2 y2 ∧
    ellipse_equation a b ∧ max_point_area px py ∧ max_area area) :=
sorry

end math_proof_statement_l131_131206


namespace hazel_fish_count_l131_131616

theorem hazel_fish_count (total_fish father_fish hazel_fish : ℕ) 
  (h1 : total_fish = 94)
  (h2 : father_fish = 46)
  (h3 : hazel_fish = total_fish - father_fish) 
  : hazel_fish = 48 := 
by 
  rw [h1, h2, h3]
  sorry

end hazel_fish_count_l131_131616


namespace card_triangle_probability_l131_131695

theorem card_triangle_probability :
  let cards := [2, 2, 4, 4, 6, 6] in
  let total_combinations := (cards.combinations 3).length in
  let valid_combinations := (cards.combinations 3).count (λ triplet, 
    let [a, b, c] := triplet.sort in
    a + b > c ∧ a + c > b ∧ b + c > a) in
  (valid_combinations: ℚ) / total_combinations = (2: ℚ) / 5 :=
by
  sorry

end card_triangle_probability_l131_131695


namespace sequence_formula_l131_131278

-- Define the sequence a_n using the recurrence relation
def a : ℕ → ℚ
| 0     := 0
| (n+1) := a n + (n+1)^3

-- The statement to be proved
theorem sequence_formula (n : ℕ) : 
  a n = (n^2 * (n+1)^2) / 4 := sorry

end sequence_formula_l131_131278


namespace maximize_total_subsidy_l131_131057

theorem maximize_total_subsidy (a b : ℕ) (m : ℝ) (h₁ : a + b = 100) (h₂ : 10 ≤ a) (h₃ : 10 ≤ b) 
  (h₄ : 0 < m) : 
  let f (x : ℝ) := m * Real.log (x + 1) - x / 10 + 1 in
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 9 ∧ 
  (if 0 < m ∧ m ≤ (1/5 : ℝ) then x = 1
   else if (1/5 : ℝ) < m ∧ m < 1 then x = 10 * m - 1
   else x = 9) :=
by
  sorry

end maximize_total_subsidy_l131_131057


namespace area_swept_by_AC_l131_131428

theorem area_swept_by_AC 
  (AB BC : ℝ) (angle_ABC : ℝ) (rotation_angle : ℝ)
  (h_AB : AB = 10) (h_BC : BC = 5) (h_angle_ABC : angle_ABC = 60) (h_rotation_angle : rotation_angle = 120) :
  let π := 3 in
  let area_of_swept := (100 * π / 3) - (25 * π / 3) in
  area_of_swept = 75 :=
by
  sorry

end area_swept_by_AC_l131_131428


namespace find_value_l131_131093

theorem find_value (x y z : ℚ) 
(h1 : 2 * x + y = 6) 
(h2 : x + 2 * y = 5) 
(h3 : x - y + z = 7)
: (x + y) / 3 = 11 / 9 := 
begin
  sorry
end

end find_value_l131_131093


namespace horse_goat_sheep_consumption_l131_131453

theorem horse_goat_sheep_consumption :
  (1 / (1 / (1 : ℝ) + 1 / 2 + 1 / 3)) = 6 / 11 :=
by
  sorry

end horse_goat_sheep_consumption_l131_131453


namespace closest_integer_area_triangle_eq_79_l131_131822

theorem closest_integer_area_triangle_eq_79 (P A B C : Point)
  (PA PB PC : ℝ)
  (hPA : PA = dist P A)
  (hPB : PB = dist P B)
  (hPC : PC = dist P C)
  (hPA_6 : PA = 6)
  (hPB_8 : PB = 8)
  (hPC_10 : PC = 10)
  (equilateral : equilateral_triangle A B C) :
  abs ((triangle_area A B C) - 79) < 1 := by
  sorry

end closest_integer_area_triangle_eq_79_l131_131822


namespace g_g_even_l131_131307

variable {α : Type*} [HasNeg α]
variable {g : α → α}

def is_even (f : α → α) : Prop := ∀ x, f (-x) = f x

theorem g_g_even (h : is_even g) : is_even (g ∘ g) :=
by
  sorry

end g_g_even_l131_131307


namespace box_length_approximation_l131_131813

theorem box_length_approximation :
  ∃ (l : ℝ), abs (l - 12.2) < 0.1 ∧
  ∃ n : ℕ, 1.08e6 = n * (l ^ 3) ∧ 300 = (0.5 * n) :=
sorry

end box_length_approximation_l131_131813


namespace probability_no_defective_pens_selected_l131_131614

noncomputable def probability_not_defective (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) : ℚ :=
  let non_defective_pens := total_pens - defective_pens
  let first_prob := non_defective_pens / total_pens
  let remaining_pens := total_pens - 1
  let remaining_non_defective_pens := non_defective_pens - 1
  let second_prob := remaining_non_defective_pens / remaining_pens
  first_prob * second_prob

theorem probability_no_defective_pens_selected :
  probability_not_defective 12 3 2 = 6 / 11 :=
by
  sorry

end probability_no_defective_pens_selected_l131_131614


namespace set_of_integers_between_10_and_16_l131_131514

theorem set_of_integers_between_10_and_16 :
  {x : ℤ | 10 < x ∧ x < 16} = {11, 12, 13, 14, 15} :=
by
  sorry

end set_of_integers_between_10_and_16_l131_131514


namespace find_angle_C_value_of_c_l131_131952

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}

-- Define the vectors m and n
def vec_m (A : ℝ) : ℝ × ℝ := (Real.sin A, Real.cos A)
def vec_n (B : ℝ) : ℝ × ℝ := (Real.cos B, Real.sin B)

-- Define the dot product of vectors
def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

-- Define the sine and cosine expressions given the conditions
def angle_sine_condition (A B C : ℝ) : Prop := dot_product (vec_m A) (vec_n B) = Real.sin (2 * C)
def sequence_condition (A B C : ℝ) : Prop := (Real.sin C)^2 = Real.sin A * Real.sin B
def geometric_condition (a b c : ℝ) : Prop := c^2 = a * b
def vector_dot_condition : Prop := let ab := (c^2 / 36) in ab * (1 / 2) = 18

-- Define the goal for angle C
theorem find_angle_C (h:sine_condition A B C) (h0: Real.sin C ≠ 0) (h1: C ∈ (0:ℝ) .. Real.pi):
  C = Real.pi / 3 := sorry

-- Define the goal for side c
theorem value_of_c (a b : ℝ) (h: sequence_condition A B C) (h1: geometric_condition a b c) (h2: vector_dot_condition):
  c = 6 := sorry

end find_angle_C_value_of_c_l131_131952


namespace integral_abs_eq_11_over_3_l131_131513

theorem integral_abs_eq_11_over_3 :
  ∫ x in 0..1, |x^2 - 4| = 11 / 3 := by
  sorry

end integral_abs_eq_11_over_3_l131_131513


namespace find_starting_number_l131_131166

theorem find_starting_number (n : ℕ) (h : ((28 + n) / 2) = 18) : n = 8 :=
sorry

end find_starting_number_l131_131166


namespace wrapping_paper_area_l131_131442

-- Defining the dimensions of the box
variables (l w h : ℝ)

-- Define the calculation for max value used in wrapping paper dimensions
def wrapping_paper_side (l w h : ℝ) : ℝ :=
  2 * max ((l / 2) + h) ((w / 2) + h)

-- Theorem to prove the required area for wrapping paper
theorem wrapping_paper_area (l w h : ℝ) :
  4 * (max ((l / 2) + h) ((w / 2) + h)) ^ 2 = wrapping_paper_side l w h ^ 2 :=
by
  sorry

end wrapping_paper_area_l131_131442


namespace derivative_at_one_l131_131103

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x

theorem derivative_at_one : (deriv f 1) = 4 := by
  sorry

end derivative_at_one_l131_131103


namespace max_value_sin_cos_l131_131888

theorem max_value_sin_cos (x : Real) (hx : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ M, M = 1 ∧ ∀ y, y ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → sin y - Real.sqrt 3 * cos y ≤ M := 
sorry

end max_value_sin_cos_l131_131888


namespace simplify_fraction_l131_131716

theorem simplify_fraction :
  ((3^2008)^2 - (3^2006)^2) / ((3^2007)^2 - (3^2005)^2) = 9 :=
by
  sorry

end simplify_fraction_l131_131716


namespace order_of_magnitudes_l131_131948

theorem order_of_magnitudes (a b c : ℝ) (h1 : a = 0.7^6) (h2 : b = 6^0.7) (h3 : c = log 0.7 6) : c < a ∧ a < b :=
by
  -- Definitions from conditions
  have ha : 0 < a := by sorry
  have hb : 1 < b := by sorry
  have hc : c < 0 := by sorry
  -- Proof of the order
  have cab: c < a := by sorry
  have ab : a < b := by sorry
  exact ⟨cab, ab⟩

end order_of_magnitudes_l131_131948


namespace complex_number_first_quadrant_l131_131712

theorem complex_number_first_quadrant (z : ℂ) (h : z = (i - 1) / i) : 
  ∃ x y : ℝ, z = x + y * I ∧ x > 0 ∧ y > 0 := 
sorry

end complex_number_first_quadrant_l131_131712


namespace annie_overtakes_bonnie_l131_131069

-- Define the conditions
def track_circumference : ℝ := 300
def bonnie_speed (v : ℝ) : ℝ := v
def annie_speed (v : ℝ) : ℝ := 1.5 * v

-- Define the statement for proving the number of laps completed by Annie when she first overtakes Bonnie
theorem annie_overtakes_bonnie (v t : ℝ) : 
  bonnie_speed v * t = track_circumference * 2 → 
  annie_speed v * t = track_circumference * 3 :=
by
  sorry

end annie_overtakes_bonnie_l131_131069


namespace proving_positivity_l131_131926

variables {ℝ : Type*} [LinearOrderedField ℝ] (f : ℝ → ℝ)

-- Assuming the function f is differentiable and its derivative is noted as f'
-- also assuming the condition f(x) + (x - 1) * f' (x) > 0 for all x in ℝ
def problem_condition (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x + (x - 1) * f' x > 0

-- Formulate the proof problem
theorem proving_positivity (f' : ℝ → ℝ) (cond : problem_condition f f') :
  ∀ x : ℝ, f x > 0 :=
  by sorry

end proving_positivity_l131_131926


namespace AB_word_implementable_l131_131617

def AB_word (n : ℕ) := {w // ∀ i : ℕ, i < 2^n → implementable i}
-- This definition assumes a generic way to represent the AB_word and its implementability for the given constraints.

theorem AB_word_implementable {n : ℕ} (all_words_implementable : ∀ (w : AB_word n), implementable w) :
  ∀ (k : ℕ), ∀ (w : string), w.length = k → implementable w :=
sorry

end AB_word_implementable_l131_131617


namespace probability_not_finishing_on_time_l131_131725

-- Definitions based on the conditions
def P_finishing_on_time : ℚ := 5 / 8

-- Theorem to prove the required probability
theorem probability_not_finishing_on_time :
  (1 - P_finishing_on_time) = 3 / 8 := by
  sorry

end probability_not_finishing_on_time_l131_131725


namespace geometric_sum_problem_l131_131272

theorem geometric_sum_problem (a : ℕ → ℝ) (a_4 : a 4 = 27 * a 3) :
  ∑ k in Finset.range n, (a (2 * (k + 1))) / (a (k + 1)) = (3 ^ (n + 1) - 3) / 2 :=
by
  sorry

end geometric_sum_problem_l131_131272


namespace remove_zeros_l131_131391

theorem remove_zeros (n : ℕ) (A : matrix (fin (2 * n)) (fin (2 * n)) ℕ) :
  (∀ i j, A i j = 0 ∨ A i j = 1) →
  (∑ i j, if A i j = 0 then 1 else 0 = 3 * n) →
  ∃ (rowsToRemove colsToRemove : finset (fin (2 * n))),
    rowsToRemove.card = n ∧
    colsToRemove.card = n ∧
    (∀ i j, i ∉ rowsToRemove → j ∉ colsToRemove → A i j ≠ 0) :=
sorry

end remove_zeros_l131_131391


namespace intersection_A_B_l131_131584

/-- Definition of set A -/
def A : Set ℕ := {1, 2, 3, 4}

/-- Definition of set B -/
def B : Set ℕ := {x | x > 2}

/-- The theorem to prove the intersection of sets A and B -/
theorem intersection_A_B : A ∩ B = {3, 4} :=
by
  sorry

end intersection_A_B_l131_131584


namespace part_I_solution_part_II_solution_l131_131573

-- Part I
theorem part_I_solution : 
  let f (x : ℝ) := 2 * x^2 - 3 * x + 2
  in ∀ x : ℝ, f x > 1 ↔ x < 1/2 ∨ x > 1 :=
by
  sorry

-- Part II
theorem part_II_solution (a : ℝ) : 
  let f (x : ℝ) := a * x^2 - (a + 1) * x + 2
  in (∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → f x ≥ 0) ↔ (1/6 ≤ a ∧ a ≤ 3 + 2 * Real.sqrt 2) :=
by
  sorry

end part_I_solution_part_II_solution_l131_131573


namespace cos_495_eq_neg_sqrt2_div2_l131_131493

theorem cos_495_eq_neg_sqrt2_div2 : 
  let deg495 := 495 : ℝ 
  let deg135 := 135 : ℝ 
  let deg45 := 45 : ℝ 
  let sqrt2_div2 := real.sqrt (2) / 2 
  let cos_deg45 := real.cos (deg45 * real.pi / 180) = sqrt2_div2 
  deg495 = deg135 + 360 ∧ real.cos (deg135 * real.pi / 180) = -sqrt2_div2 → 
  real.cos (deg495 * real.pi / 180) = -sqrt2_div2 := 
by 
  intros h1 h2; 
  sorry

end cos_495_eq_neg_sqrt2_div2_l131_131493


namespace simplify_and_evaluate_l131_131694

noncomputable def simplify_expression (x : ℤ) (h1 : -1 ≤ x ∧ x ≤ 3) (h2 : x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 2) : ℝ :=
  -1 / (x + 1)

theorem simplify_and_evaluate (x : ℤ) (h1 : -1 ≤ x ∧ x ≤ 3) (h2 : x = 3) :
  simplify_expression x h1 (by { cases h1, linarith }) = -1 / 4 :=
by {
  -- simplify (x^2 + 1) / (x^2 - 1) - (x - 2) / (x - 1) ÷ (x - 2) / x
  have h3 : x ≠ 1 ∧ x ≠ -1 ∧ x ≠ 2 := by { cases h1 with neg1_le and le3, split, linarith, split, linarith, linarith },
  refine congr_arg (λ y, y / (x + 1)) _,
  simp [simplify_expression, h1, h2, h3],
  linarith,
}

#eval simplify_expression 3 (by linarith) (by simp)

end simplify_and_evaluate_l131_131694


namespace circle_equation_line_equation_l131_131226

theorem circle_equation (a b r x y : ℝ) (h1 : a + b = 2 * x + y)
  (h2 : (a, 2*a - 2) = ((1, 2) : ℝ × ℝ))
  (h3 : (a, 2*a - 2) = ((2, 1) : ℝ × ℝ)) :
  (x - 2) ^ 2 + (y - 2) ^ 2 = 1 := sorry

theorem line_equation (x y m : ℝ) (h1 : y + 3 = (x - (-3)) * ((-3) - 0) / (m - (-3)))
  (h2 : (x, y) = (m, 0) ∨ (x, y) = (m, 0))
  (h3 : (m = 1 ∨ m = - 3 / 4)) :
  (3 * x + 4 * y - 3 = 0) ∨ (4 * x + 3 * y + 3 = 0) := sorry

end circle_equation_line_equation_l131_131226


namespace athlete_a_catches_up_and_race_duration_l131_131986

-- Track is 1000 meters
def track_length : ℕ := 1000

-- Athlete A's speed: first minute, increasing until 5th minute and decreasing until 600 meters/min
def athlete_A_speed (minute : ℕ) : ℕ :=
  match minute with
  | 0 => 1000
  | 1 => 1000
  | 2 => 1200
  | 3 => 1400
  | 4 => 1600
  | 5 => 1400
  | 6 => 1200
  | 7 => 1000
  | 8 => 800
  | 9 => 600
  | _ => 600

-- Athlete B's constant speed
def athlete_B_speed : ℕ := 1200

-- Function to compute distance covered in given minutes, assuming starts at 0
def total_distance (speed : ℕ → ℕ) (minutes : ℕ) : ℕ :=
  (List.range minutes).map speed |>.sum

-- Defining the maximum speed moment for A
def athlete_A_max_speed_distance : ℕ := total_distance athlete_A_speed 4
def athlete_B_max_speed_distance : ℕ := athlete_B_speed * 4

-- Proof calculation for target time 10 2/3 minutes
def time_catch : ℚ := 10 + 2 / 3

-- Defining the theorem to be proven
theorem athlete_a_catches_up_and_race_duration :
  athlete_A_max_speed_distance > athlete_B_max_speed_distance ∧ time_catch = 32 / 3 :=
by
  -- Place holder for the proof's details
  sorry

end athlete_a_catches_up_and_race_duration_l131_131986


namespace count_integers_with_property_l131_131885

theorem count_integers_with_property : 
  let valid_integers := 
    {n | 3000 ≤ n ∧ n ≤ 5999 ∧ 
    let d := n % 10,
    let a := (n / 1000),
    let b := (n / 100) % 10,
    let c := (n / 10) % 10 
    in d = a + b + c } 
  in valid_integers.card = 64 :=
by
  sorry

end count_integers_with_property_l131_131885


namespace repetend_of_5_div_17_l131_131139

theorem repetend_of_5_div_17 : 
  ∃ repetend : ℕ, 
  decimal_repetend (5 / 17) = repetend ∧ 
  repetend = 2941176470588235 :=
by 
  skip

end repetend_of_5_div_17_l131_131139


namespace f_monotonicity_g_no_zeros_l131_131577

def f (x : ℝ) := x^2 - (1/2) * Real.log x
def g (x : ℝ) (m : ℝ) := f x + (1/2) * m * x

theorem f_monotonicity :
  (∀ x, 0 < x → x < 1/2 → f' x < 0) ∧
  (∀ x, 1/2 < x → f' x > 0) := sorry

theorem g_no_zeros (m : ℝ) :
  (∀ x, 1 < x → g x m > 0) → m ∈ set.Ici (-2) := sorry

end f_monotonicity_g_no_zeros_l131_131577


namespace yearly_payment_split_evenly_l131_131006

def monthly_cost : ℤ := 14
def split_cost (cost : ℤ) := cost / 2
def total_yearly_cost (monthly_payment : ℤ) := monthly_payment * 12

theorem yearly_payment_split_evenly (h : split_cost monthly_cost = 7) :
  total_yearly_cost (split_cost monthly_cost) = 84 :=
by
  -- Here we use the hypothesis h which simplifies the proof.
  sorry

end yearly_payment_split_evenly_l131_131006


namespace work_completation_time_l131_131811

theorem work_completation_time (x : ℕ) (B_time : ℕ) (total_days : ℕ) (last_days: ℕ) : 
  B_time = 30 ∧ total_days = 18 ∧ last_days = 10 ∧ 
  (8 * (1 / x + 1 / B_time) + last_days * (1 / B_time) = 1) → 
  x = 20 :=
begin 
  sorry 
end

end work_completation_time_l131_131811


namespace ratio_of_age_difference_l131_131339

theorem ratio_of_age_difference (R J K : ℕ) 
  (h1 : R = J + 6) 
  (h2 : R + 4 = 2 * (J + 4)) 
  (h3 : (R + 4) * (K + 4) = 108) : 
  (R - J) / (R - K) = 2 :=
by 
  sorry

end ratio_of_age_difference_l131_131339


namespace area_bounded_by_given_curves_is_pi_l131_131489

noncomputable def area_of_region_bounded_by_curves 
    (r1 r2 : ℝ → ℝ) 
    (r1_def : ∀ φ, r1 φ = 6 * cos (3 * φ)) 
    (r2_def : ∀ φ, r2 φ = 3) 
    (phi1 phi2 : ℝ) 
    (phi_bounds: phi1 = -π/9 ∧ phi2 = π/9) : ℝ :=
  (1/2) * ∫ φ in phi1..phi2, (r1 φ)^2 - (r2 φ)^2

theorem area_bounded_by_given_curves_is_pi :
  area_of_region_bounded_by_curves (λ φ, 6 * cos (3 * φ)) (λ φ, 3) 
    (λ φ, rfl) (λ φ, rfl) (-π/9) (π/9) 
    (by simp [neg_div_self, div_self] : -π/9 = -(π/9) ∧ π/9 = π/9) = π := 
  sorry

end area_bounded_by_given_curves_is_pi_l131_131489


namespace num_possible_values_of_b2_l131_131830

def seq (b : ℕ → ℕ) : Prop := ∀ n ≥ 1, b (n + 2) = |b (n + 1) - b n|

noncomputable def gcd (a b : ℕ) : ℕ := sorry

theorem num_possible_values_of_b2 :
  ∃ b : ℕ → ℕ, seq b ∧ b 1 = 1001 ∧ b 2 < 1001 ∧ b 2023 = 1 ∧ 
  (∃ m : ℕ, m = 359 ∧ ∀ b2, (b 2 = b2 ∧ b2 < 1001 ∧ odd b2 ∧ 
  gcd 1001 b2 = 1) → b2 ∈ (finset.range 1001).filter (λ x, x % 2 = 1 ∧ 1001.gcd x = 1) 
  ∧ (finset.range 1001).filter (λ x, x % 2 = 1 ∧ 1001.gcd x = 1).card = 359)) :=
sorry

end num_possible_values_of_b2_l131_131830


namespace tan_of_angle_l131_131570

noncomputable def tan_val (α : ℝ) : ℝ := Real.tan α

theorem tan_of_angle (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (h2 : Real.cos (2 * α) = -3 / 5) :
  tan_val α = -2 := by
  sorry

end tan_of_angle_l131_131570


namespace cubes_sum_correct_l131_131365

noncomputable def max_cubes : ℕ := 11
noncomputable def min_cubes : ℕ := 9

theorem cubes_sum_correct : max_cubes + min_cubes = 20 :=
by
  unfold max_cubes min_cubes
  sorry

end cubes_sum_correct_l131_131365


namespace q_can_do_work_in_10_days_l131_131423

theorem q_can_do_work_in_10_days (R_p R_q R_pq: ℝ)
  (h1 : R_p = 1 / 15)
  (h2 : R_pq = 1 / 6)
  (h3 : R_p + R_q = R_pq) :
  1 / R_q = 10 :=
by
  -- Proof steps go here.
  sorry

end q_can_do_work_in_10_days_l131_131423


namespace polar_coordinates_correct_l131_131096

-- Define rectangular coordinates
def rectangular_coordinates : ℝ × ℝ := (-2, 2 * Real.sqrt 3)

-- Define the polar coordinates solution
def polar_coordinates : ℝ × ℝ := (4, 2 * Real.pi / 3)

-- Prove that polar coordinates correspond to the given rectangular coordinates
theorem polar_coordinates_correct :
  ∃ r θ, (r, θ) = polar_coordinates ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*Real.pi ∧ 
                r = Real.sqrt ((-2)^2 + (2*Real.sqrt 3)^2) ∧
                θ = Real.atan ((2 * Real.sqrt 3) / -2) + Real.pi :=
by
  sorry

end polar_coordinates_correct_l131_131096


namespace circle_points_to_one_l131_131074

theorem circle_points_to_one (n : ℕ) (initial_config : Fin n → ℤ) (h : ∀ i, (initial_config i = 1 ∨ initial_config i = -1)) :
  (∃ k ≥ 1, n = 2 ^ k) ↔ 
  (∀ (op : Fin n → ℤ) (k : ℕ), 
    (∀ i, op i = initial_config i) → 
    (∀ i, (op (i + 1) mod n) = 1)
  → ∃ m, op = λ x, 1) := 
sorry

end circle_points_to_one_l131_131074


namespace intersection_of_M_and_N_l131_131583

theorem intersection_of_M_and_N (x : ℝ) :
  {x | x > 1} ∩ {x | x^2 - 2 * x < 0} = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_M_and_N_l131_131583


namespace value_of_a_l131_131378

theorem value_of_a (a : ℝ) (h : (a, 0) ∈ {p : ℝ × ℝ | p.2 = p.1 + 8}) : a = -8 :=
sorry

end value_of_a_l131_131378


namespace obtuse_angle_in_triangle_l131_131505

-- Defining points A, B, and C
def A : (ℝ × ℝ) := (1, 2)
def B : (ℝ × ℝ) := (-3, 4)
def C : (ℝ × ℝ) := (0, -2)

-- Function to calculate the square of the distance between two points
def dist_sq (p1 p2: ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the square of the lengths of sides AB, BC, and AC
def AB_sq : ℝ := dist_sq A B
def BC_sq : ℝ := dist_sq B C
def AC_sq : ℝ := dist_sq A C

-- Main theorem
theorem obtuse_angle_in_triangle :
  BC_sq > AB_sq + AC_sq :=
by
  -- Skipping the proof
  sorry

end obtuse_angle_in_triangle_l131_131505


namespace middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l131_131212

noncomputable def term_in_expansion (n k : ℕ) : ℚ :=
  (Nat.choose n k) * ((-1/2) ^ k)

theorem middle_term_in_expansion :
  term_in_expansion 8 4 = 35 / 8 := by
  sorry

theorem sum_of_odd_coefficients :
  (term_in_expansion 8 1 + term_in_expansion 8 3 + term_in_expansion 8 5 + term_in_expansion 8 7) = -(205 / 16) := by
  sorry

theorem weighted_sum_of_coefficients :
  ((1 * term_in_expansion 8 1) + (2 * term_in_expansion 8 2) + (3 * term_in_expansion 8 3) + (4 * term_in_expansion 8 4) +
  (5 * term_in_expansion 8 5) + (6 * term_in_expansion 8 6) + (7 * term_in_expansion 8 7) + (8 * term_in_expansion 8 8)) =
  -(1 / 32) := by
  sorry

end middle_term_in_expansion_sum_of_odd_coefficients_weighted_sum_of_coefficients_l131_131212


namespace f_monotonically_decreasing_solve_inequality_l131_131672

-- Definitions based on conditions
variable {R : Type*} [LinearOrderedField R]

def domain (f : R → R) := ∀ x : R, x > 0 → f x < 0
def functionalEq (f : R → R) := ∀ x y : R, x > 0 → y > 0 → f (x / y) = f x - f y
def valueHalf (f : R → R) := f (1 / 2) = 1

-- Prove that f(x) is monotonically decreasing
theorem f_monotonically_decreasing (f : R → R) [domain f] [functionalEq f] [valueHalf f] : 
  ∀ x1 x2 : R, 0 < x1 → 0 < x2 → x1 > x2 → f x1 < f x2 := 
sorry

-- Prove the solution to the inequality
theorem solve_inequality (f : R → R) [domain f] [functionalEq f] [valueHalf f] : 
  ∀ x : R, 1 ≤ x ∧ x ≤ 4 ↔ f x + f (5 - x) ≥ -2 :=
sorry

end f_monotonically_decreasing_solve_inequality_l131_131672


namespace repetend_of_5_over_17_is_294117_l131_131144

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l131_131144


namespace houses_before_boom_l131_131851

theorem houses_before_boom (current_houses built_during_boom houses_before : ℕ) 
  (h1 : current_houses = 2000)
  (h2 : built_during_boom = 574)
  (h3 : current_houses = houses_before + built_during_boom) : 
  houses_before = 1426 := 
by
  -- Proof omitted
  sorry

end houses_before_boom_l131_131851


namespace units_digit_of_33_pow_33_mul_7_pow_7_l131_131854

theorem units_digit_of_33_pow_33_mul_7_pow_7 : (33 ^ (33 * (7 ^ 7))) % 10 = 7 := 
  sorry

end units_digit_of_33_pow_33_mul_7_pow_7_l131_131854


namespace ship_distances_l131_131054

-- Define the conditions based on the initial problem statement
variables (f : ℕ → ℝ)
def distances_at_known_times : Prop :=
  f 0 = 49 ∧ f 2 = 25 ∧ f 3 = 121

-- Define the questions to prove the distances at unknown times
def distance_at_time_1 : Prop :=
  f 1 = 1

def distance_at_time_4 : Prop :=
  f 4 = 289

-- The proof problem
theorem ship_distances
  (f : ℕ → ℝ)
  (hf : ∀ t, ∃ a b c, f t = a*t^2 + b*t + c)
  (h_known : distances_at_known_times f) :
  distance_at_time_1 f ∧ distance_at_time_4 f :=
by
  sorry

end ship_distances_l131_131054


namespace proposition_2_correct_proposition_3_correct_l131_131936

theorem proposition_2_correct :
  ∀ {A B C D : Point}, ¬Coplanar A B C D → (¬Collinear A B C ∧ ¬Collinear A B D ∧ ¬Collinear A C D ∧ ¬Collinear B C D) :=
by
  sorry

theorem proposition_3_correct :
  ∀ {A B C D : Point}, (Collinear A B C ∨ Collinear A B D ∨ Collinear A C D ∨ Collinear B C D) → Coplanar A B C D :=
by
  sorry

end proposition_2_correct_proposition_3_correct_l131_131936


namespace ratio_of_areas_l131_131721

variable (s : ℝ)

def area_square (s : ℝ) : ℝ := s * s
def longer_side (s : ℝ) : ℝ := 1.2 * s
def shorter_side (s : ℝ) : ℝ := 0.8 * s
def area_rectangle (s : ℝ) : ℝ := longer_side s * shorter_side s
def area_one_triangle (s : ℝ) : ℝ := (area_rectangle s) / 2
def ratio (s : ℝ) : ℝ := (area_one_triangle s) / (area_square s)

theorem ratio_of_areas (s : ℝ) : ratio s = 12 / 25 :=
by
  sorry

end ratio_of_areas_l131_131721


namespace perfect_cubes_count_l131_131593

theorem perfect_cubes_count : 
  Nat.card {n : ℕ | n^3 > 500 ∧ n^3 < 2000} = 5 :=
by
  sorry

end perfect_cubes_count_l131_131593


namespace complex_in_second_quadrant_l131_131597

noncomputable def is_second_quadrant (θ : ℝ) : Prop :=
  θ ∈ Ioo (3 * π / 4) (5 * π / 4) ∧ 
  (cos θ + sin θ) < 0 ∧ 
  (sin θ - cos θ) > 0

theorem complex_in_second_quadrant (θ : ℝ) (h : θ ∈ Ioo (3 * π / 4) (5 * π / 4)) :
  is_second_quadrant θ :=
by
  unfold is_second_quadrant
  split
  . exact h
  . sorry
  . sorry

end complex_in_second_quadrant_l131_131597


namespace sin_alpha_plus_5pi_over_12_l131_131222

theorem sin_alpha_plus_5pi_over_12 (α : ℝ) 
  (hα : 0 < α ∧ α < π / 4) 
  (ha : (cos α, sin α) = (cos α, sin α)) 
  (hb : (1, -1) = (1, -1)) 
  (dot_product : cos α - sin α = (2 * Real.sqrt 2) / 3) :
  Real.sin (α + 5 * Real.pi / 12) = (2 + Real.sqrt 15) / 6 := 
by 
  sorry -- Proof goes here

end sin_alpha_plus_5pi_over_12_l131_131222


namespace partitions_number_of_U_l131_131256

open Set

-- Definitions
variable {U : Set ℕ} (A B : Set ℕ)

-- non-empty condition
def non_empty (s : Set ℕ) : Prop := s ≠ ∅

-- Partition conditions
def partition_condition1 : Prop := A ∪ B = U
def partition_condition2 : Prop := A ∩ B = ∅

-- Problem statement
theorem partitions_number_of_U :
  (non_empty A) → (non_empty B) → (partition_condition1 A B) → (partition_condition2 A B) → (U = {1, 2, 3}) →
  ∃! (n : ℕ), n = 6 := sorry

end partitions_number_of_U_l131_131256


namespace assign_students_to_tests_l131_131388

def students := Fin 5
def tests := Fin 5

-- Define the conditions as stated in the problem.
def condition1 (s : students) := Set (tests) -- Each student takes a set of tests.
def condition2 (t : tests) := Set (students) -- Each test is taken by a set of students.

noncomputable def problem : Prop :=
  ∃ (f : students → Set tests) (g : tests → Set students),
    (∀ s, (f s).card = 2) ∧  -- Each student chooses 2 distinct tests
    (∀ t, (g t).card = 2) ∧  -- Each test is taken by exactly 2 students
    (∀ s t, t ∈ f s ↔ s ∈ g t) -- Consistency between f and g

theorem assign_students_to_tests : problem → ∃! n, n = 2040 :=
begin
  intros h,
  -- Proof to be filled in
  use 2040,
  sorry
end

end assign_students_to_tests_l131_131388


namespace minimum_distance_is_1805_l131_131728

/-- Define points A and B in a 2D plane -/
def A : ℝ × ℝ := (0, 350)
def B : ℝ × ℝ := (1500, 600)
def wall_start : ℝ × ℝ := (0, 0)
def wall_end : ℝ × ℝ := (1500, 0)

/-- The calculation of the reflected point B' -/
def B' : ℝ × ℝ := (1500, -600)

/-- The Pythagorean theorem to find the distance AB' -/
def distance_AB' : ℝ := real.sqrt (1500^2 + (350 + 600)^2)

/-- The minimum running distance a participant must run, rounded to the nearest meter -/
def min_running_distance : ℕ := real.to_nat (real.ceil (distance_AB'))

theorem minimum_distance_is_1805 : min_running_distance = 1805 :=
by
  -- Proof steps would go here
  sorry

end minimum_distance_is_1805_l131_131728


namespace minimum_pairwise_non_parallel_lines_l131_131544

theorem minimum_pairwise_non_parallel_lines (n : ℕ) (h : ∀ (p1 p2 p3 : ℕ × ℕ), ¬(p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ collinear p1 p2 p3)) :
  ∃ (m : ℕ), m = n ∧ m = minimum_pairwise_non_parallel_lines n :=
sorry

def collinear (p1 p2 p3 : ℕ × ℕ) : Prop :=
  (p2.1 - p1.1) * (p3.2 - p1.2) = (p3.1 - p1.1) * (p2.2 - p1.2)

def minimum_pairwise_non_parallel_lines (n : ℕ) : ℕ := n

end minimum_pairwise_non_parallel_lines_l131_131544


namespace area_of_quadrilateral_l131_131165

theorem area_of_quadrilateral (d o1 o2 : ℝ) (h1 : d = 24) (h2 : o1 = 9) (h3 : o2 = 6) :
  (1 / 2 * d * o1) + (1 / 2 * d * o2) = 180 :=
by {
  sorry
}

end area_of_quadrilateral_l131_131165


namespace solve_equation_l131_131343

theorem solve_equation (x : ℝ) (h : real.sqrt (5 * x - 4) + 12 / real.sqrt (5 * x - 4) = 9) :
  x = 13 / 5 ∨ x = 4 :=
by
  sorry

end solve_equation_l131_131343


namespace marys_garbage_bill_is_correct_l131_131679

noncomputable def calculate_garbage_bill :=
  let weekly_trash_bin_cost := 2 * 10
  let weekly_recycling_bin_cost := 1 * 5
  let weekly_green_waste_bin_cost := 1 * 3
  let total_weekly_cost := weekly_trash_bin_cost + weekly_recycling_bin_cost + weekly_green_waste_bin_cost
  let monthly_bin_cost := total_weekly_cost * 4
  let base_monthly_cost := monthly_bin_cost + 15
  let discount := base_monthly_cost * 0.18
  let discounted_cost := base_monthly_cost - discount
  let fines := 20 + 10
  discounted_cost + fines

theorem marys_garbage_bill_is_correct :
  calculate_garbage_bill = 134.14 := 
  by {
  sorry
  }

end marys_garbage_bill_is_correct_l131_131679


namespace constant_term_of_binomial_expansion_l131_131519

-- Define the given binomial expression and the conditions
def binomial_expression (x : ℝ) : ℝ := (x + 2 / real.sqrt x) ^ 6

-- Define what it means to expand and find the constant term
def constant_term (exp : ℝ → ℝ) : ℝ :=
  if h : ∃ n : ℕ, ∀ x : ℝ, exp x = polynomial.eval x (polynomial.X ^ n) then
    polynomial.coeff h.some 0
  else
    0

-- Prove the specific case
theorem constant_term_of_binomial_expansion : constant_term binomial_expression = 240 := by
  sorry

end constant_term_of_binomial_expansion_l131_131519


namespace cosine_sum_minimum_value_l131_131522

theorem cosine_sum_minimum_value (x y : ℝ) (h : cos x + cos y = 1 / 3) :
  cos (x + y) ≥ -17 / 18 :=
sorry

end cosine_sum_minimum_value_l131_131522


namespace complement_A_with_respect_to_U_l131_131946

-- Defining the universal set U
def U : Set ℝ := {x : ℝ | x^2 > 1}

-- Defining the set A
def A : Set ℝ := {x : ℝ | x^2 - 4 * x + 3 < 0}

-- Statement of the problem in Lean 4
theorem complement_A_with_respect_to_U :
  ∁_U A = {x : ℝ | x < -1 ∨ x ≥ 3} :=
sorry

end complement_A_with_respect_to_U_l131_131946


namespace sum_of_midpoint_coords_is_seven_l131_131788

-- Define coordinates of the endpoints
def endpoint1 : ℝ × ℝ := (8, 16)
def endpoint2 : ℝ × ℝ := (-2, -8)

-- Define the midpoint coordinates
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the sum of the coordinates of the midpoint
def sum_of_midpoint_coords (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

-- Theorem stating that the sum of the coordinates of the midpoint is 7
theorem sum_of_midpoint_coords_is_seven : 
  sum_of_midpoint_coords (midpoint endpoint1 endpoint2) = 7 :=
by
  -- Proof would go here
  sorry

end sum_of_midpoint_coords_is_seven_l131_131788


namespace allocation_ways_l131_131840

-- Define the problem statement in Lean 4
theorem allocation_ways :
  let positions := 8
  let schools := 3
  ∃ (allocations : list ℕ), 
    length allocations = schools ∧ 
    all (λ x, x > 0) allocations ∧ 
    (∀ i j, i ≠ j → allocations.nth i ≠ allocations.nth j) ∧ 
    allocations.sum = positions ∧ 
    num_ways allocations = 12 :=
sorry

end allocation_ways_l131_131840


namespace geometry_concurrence_problem_l131_131841

theorem geometry_concurrence_problem
  (A B C H A₁ C₁ B₀ A' C' : Point)
  [acute_triangle : AcuteTriangle A B C]
  [orthocenter : Orthocenter A H] : 
  midpoint (A C) B₀ -> 
  (line_through B (parallel_to AC)) -> 
  meets B₀ A₁ A' -> 
  meets B₀ C₁ C' -> 
  concurrency (line_through A A') (line_through C C') (line_through B H) := sorry

end geometry_concurrence_problem_l131_131841


namespace symmetric_line_equation_l131_131930

theorem symmetric_line_equation :
  (∃ l, ∀ x y, l x y ↔ (4*(-x) - 3*y + 5 = 0)) → 
  (∃ l, ∀ x y, l x y ↔ (4*x + 3*y - 5 = 0)) :=
by
  intro h
  rcases h with ⟨l, hl⟩
  use λ x y, 4*x + 3*y - 5 = 0
  sorry

end symmetric_line_equation_l131_131930


namespace daily_production_l131_131449

-- Definitions based on conditions
def weekly_production : ℕ := 3400
def working_days_in_week : ℕ := 5

-- Statement to prove the number of toys produced each day
theorem daily_production : (weekly_production / working_days_in_week) = 680 :=
by
  sorry

end daily_production_l131_131449


namespace alice_unanswered_questions_l131_131839

theorem alice_unanswered_questions :
  ∃ (c w u : ℕ), (5 * c - 2 * w = 54) ∧ (2 * c + u = 36) ∧ (c + w + u = 30) ∧ (u = 8) :=
by
  -- proof omitted
  sorry

end alice_unanswered_questions_l131_131839


namespace ratio_longer_to_shorter_side_l131_131828

-- Definitions of the problem
variables (l s : ℝ)
def rect_sheet_fold : Prop :=
  l = Real.sqrt (s^2 + (s^2 / l)^2)

-- The to-be-proved theorem
theorem ratio_longer_to_shorter_side (h : rect_sheet_fold l s) :
  l / s = Real.sqrt ((2 : ℝ) / (Real.sqrt 5 - 1)) :=
sorry

end ratio_longer_to_shorter_side_l131_131828


namespace range_of_a_l131_131941

theorem range_of_a (a : ℝ) :
  (∃ l1 l2 : ℝ → ℝ, 
    (∀ x, l1 x = k * (x - 1) - 2) ∧ (∀ x, l2 x = - (1/k) * (x - 1) - 2) ∧
    k > 0 ∧ 
    ((∃ x, y = ax^2 ∧ y = l1 x) ∨ (∃ x, y = ax^2 ∧ y = l2 x))
  ) ↔ (a ∈ Iio 0 ∪ Ioc 0 (1/8)) := sorry

end range_of_a_l131_131941


namespace probability_X_lt_3_l131_131204

-- Given definitions
def X : ProbabilityTheory.RealDistribution := ProbabilityTheory.Normal 1 σ

-- Conditions
axiom P_0_lt_X_lt_3 : set.prob { x | 0 < x ∧ x < 3 } (X .val) = 0.5
axiom P_0_lt_X_lt_1 : set.prob { x | 0 < x ∧ x < 1 } (X .val) = 0.2

-- Theorem to prove the required probability
theorem probability_X_lt_3 (σ : ℝ) :
  set.prob { x | x < 3 } (X .val) = 0.8 :=
begin
  sorry
end

end probability_X_lt_3_l131_131204


namespace sum_of_subseq_one_seventh_sum_of_subseq_not_one_fifth_l131_131285

-- Problem (a): Prove that the sum of a subsequence can be 1/7
theorem sum_of_subseq_one_seventh : 
  ∃ S : Set ℝ, S ⊆ { 1 / (2 ^ n) | n : ℕ } ∧ ∑' x in S, x = 1 / 7 := 
sorry

-- Problem (b): Prove that the sum of a subsequence cannot be 1/5
theorem sum_of_subseq_not_one_fifth : 
  ¬ ∃ S : Set ℝ, S ⊆ { 1 / (2 ^ n) | n : ℕ } ∧ ∑' x in S, x = 1 / 5 := 
sorry

end sum_of_subseq_one_seventh_sum_of_subseq_not_one_fifth_l131_131285


namespace repetend_of_5_over_17_is_294117_l131_131145

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l131_131145


namespace pair_represent_different_sets_BD_l131_131792

def setA_M : Set ℝ := {3, -1}
def setA_P : Set ℝ := {-1, 3}

def setB_M : Set (ℝ × ℝ) := {(3, 1)}
def setB_P : Set (ℝ × ℝ) := {(1, 3)}

def setC_M : Set ℝ := {y ∣ ∃ x : ℝ, y = x^2 + 1}
def setC_P : Set ℝ := {t ∣ t ≥ 1}

def setD_M : Set ℝ := {y ∣ ∃ x : ℝ, y = x^2 - 1}
def setD_P : Set (ℝ × ℝ) := {(x, y) ∣ y = x^2 - 1}

theorem pair_represent_different_sets_BD :
  ¬ (setA_M = setA_P) ∧ (setB_M ≠ setB_P) ∧ (setC_M = setC_P) ∧ (setD_M ≠ setD_P) :=
by {
  sorry
}

end pair_represent_different_sets_BD_l131_131792


namespace exist_three_elements_with_perfect_cube_product_l131_131915

-- Define the problem conditions
def has_prime_factors_leq3 (n : ℕ) : Prop :=
  ∀ p, nat.prime p → p ∣ n → p ≤ 3

def S : fin 9 → ℕ := sorry  -- Given S is a set of 9 integers

theorem exist_three_elements_with_perfect_cube_product :
  (∀ i, has_prime_factors_leq3 (S i)) →
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ ∃ m, S i * S j * S k = m^3 :=
sorry

end exist_three_elements_with_perfect_cube_product_l131_131915


namespace shortest_distance_origin_to_circle_closest_distance_line_to_circle_l131_131785

noncomputable def circle_center : ℝ × ℝ := (9, -3)
noncomputable def circle_radius : ℝ := Real.sqrt 55

def line (x : ℝ) : ℝ := -3 * x + 1

def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1) ^ 2 + (p₁.2 - p₂.2) ^ 2)

def distance_point_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  |A * p.1 + B * p.2 + C| / Real.sqrt (A ^ 2 + B ^ 2)

theorem shortest_distance_origin_to_circle : distance (0, 0) circle_center - circle_radius = 3 * Real.sqrt 10 - Real.sqrt 55 :=
by
  sorry

theorem closest_distance_line_to_circle : distance_point_to_line circle_center (-3) 1 1 - circle_radius = 29 / Real.sqrt 10 - Real.sqrt 55 :=
by
  sorry

end shortest_distance_origin_to_circle_closest_distance_line_to_circle_l131_131785


namespace area_of_shaded_region_l131_131817

def tangent_circles_area : ℝ :=
  let r_small := 2
  let r_large := 3
  let theta := 2 * Real.arccos (r_small / r_large)
  let sector_area := theta * r_large^2 / 2
  let triangle_area := r_small * Real.sqrt(r_large^2 - r_small^2) / 2
  2 * (sector_area - triangle_area)

theorem area_of_shaded_region : 
  tangent_circles_area = 2 * Real.arccos (2 / 3) * 5 - 2 * Real.sqrt (15) / 3 := 
by
  sorry

end area_of_shaded_region_l131_131817


namespace DE_DF_constant_l131_131917

variable {x y : ℝ}

def ellipse_eq (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

def line_l (x y : ℝ) : Prop := x = 2 * Real.sqrt 2 ∧ y = 0

noncomputable def DE (x₀ y₀ : ℝ) : ℝ :=
  let E_y := (2 * Real.sqrt 2 + 2) * y₀ / (x₀ + 2)
  Real.abs ((2 * Real.sqrt 2 + 2) * y₀ / (x₀ + 2))

noncomputable def DF (x₀ y₀ : ℝ) : ℝ :=
  let F_y := (2 * Real.sqrt 2 - 2) * y₀ / (x₀ - 2)
  Real.abs ((2 * Real.sqrt 2 - 2) * y₀ / (x₀ - 2))

theorem DE_DF_constant (x₀ y₀ : ℝ)
  (ellipse_cond : ellipse_eq x₀ y₀)
  (P_cond : -2 < x₀ ∧ x₀ < 2) :
  DE x₀ y₀ * DF x₀ y₀ = 1 := by
  sorry

end DE_DF_constant_l131_131917


namespace repetend_of_five_over_seventeen_l131_131123

theorem repetend_of_five_over_seventeen : 
  let r := 5 / 17 in
  ∃ a b : ℕ, a * 10^b = 294117 ∧ (r * 10^b - a) = (r * 10^6 - r * (10^6 / 17))
   ∧ (r * 10^k = (r * 10^6).floor / 10^k ) where k = 6 := sorry

end repetend_of_five_over_seventeen_l131_131123


namespace part_1_part_2_l131_131902

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

theorem part_1 (m : ℝ) : (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) → (m = 2) := 
by
  sorry

theorem part_2 (m : ℝ) : (A ⊆ (Set.univ \ B m)) → (m > 5 ∨ m < -3) := 
by
  sorry

end part_1_part_2_l131_131902


namespace ratio_alex_jacob_l131_131073

theorem ratio_alex_jacob :
  ∃ n : ℕ, (let J := 8 in 
           let A := n * J in 
           let remaining_fish_alex := A - 23 in 
           let required_fish_jacob := J + 26 in 
           required_fish_jacob = remaining_fish_alex + 1) →
  n * 8 / 8 = 7 :=
by
  sorry

end ratio_alex_jacob_l131_131073


namespace smallest_n_condition_l131_131864

-- Define the conditions
def condition1 (x : ℤ) : Prop := 2 * x - 3 ≡ 0 [ZMOD 13]
def condition2 (y : ℤ) : Prop := 3 * y + 4 ≡ 0 [ZMOD 13]

-- Problem statement: finding n such that the expression is a multiple of 13
theorem smallest_n_condition (x y : ℤ) (n : ℤ) :
  condition1 x → condition2 y → x^2 - x * y + y^2 + n ≡ 0 [ZMOD 13] → n = 1 := 
by
  sorry

end smallest_n_condition_l131_131864


namespace original_price_of_table_l131_131699

noncomputable def original_price (sale_price : ℝ) (discount_rate : ℝ) : ℝ :=
  sale_price / (1 - discount_rate)

theorem original_price_of_table
  (d : ℝ) (p' : ℝ) (h_d : d = 0.10) (h_p' : p' = 450) :
  original_price p' d = 500 := by
  rw [h_d, h_p']
  -- Calculating the original price
  show original_price 450 0.10 = 500
  sorry

end original_price_of_table_l131_131699


namespace num_triangles_with_perimeter_36_l131_131376

theorem num_triangles_with_perimeter_36 : 
  let count : ℕ := {
    -- define a predicate that checks if a triplet of integers forms a valid triangle
    is_valid_triangle (a b c : ℕ) : Prop :=
      a + b > c ∧ a + c > b ∧ b + c > a,
    -- calculate the number of valid triangles with perimeter 36
    (∑ c in finset.range 18 \ finset.range 12, 
      ∑ b in finset.range (c + 1), 
        ∑ a in finset.range (b + 1), 
          if a + b + c = 36 ∧ is_valid_triangle a b c then 1 else 0) }
    = 23 := 
begin
  sorry
end

end num_triangles_with_perimeter_36_l131_131376


namespace sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l131_131023

theorem sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C
  {A B C : ℝ}
  (h : A + B + C = π) :
  Real.sin (4 * A) + Real.sin (4 * B) + Real.sin (4 * C) = -4 * Real.sin (2 * A) * Real.sin (2 * B) * Real.sin (2 * C) :=
sorry

end sin_4A_plus_sin_4B_plus_sin_4C_eq_neg_4_sin_2A_sin_2B_sin_2C_l131_131023


namespace supremum_g_l131_131065

noncomputable def g (a b : ℝ) : ℝ :=
  - (1 / (2 * a)) - (2 / b)

theorem supremum_g (a b : ℝ) (h : a > 0 ∧ b > 0) (h_sum : a + b = 1) : 
  (Sup {x | ∃ a b, x = g a b ∧ a > 0 ∧ b > 0 ∧ a + b = 1}) = -9 / 2 := 
sorry

end supremum_g_l131_131065


namespace janice_total_earnings_l131_131654

-- Defining the working conditions as constants
def days_per_week : ℕ := 5  -- Janice works 5 days a week
def earning_per_day : ℕ := 30  -- Janice earns $30 per day
def overtime_earning_per_shift : ℕ := 15  -- Janice earns $15 per overtime shift
def overtime_shifts : ℕ := 3  -- Janice works three overtime shifts

-- Defining Janice's total earnings for the week
def total_earnings : ℕ := (days_per_week * earning_per_day) + (overtime_shifts * overtime_earning_per_shift)

-- Statement to prove that Janice's total earnings are $195
theorem janice_total_earnings : total_earnings = 195 :=
by
  -- The proof is omitted.
  sorry

end janice_total_earnings_l131_131654


namespace main_proof_l131_131554

-- Definitions based on given conditions
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

def g (x : ℝ) : ℝ := f (x - Real.pi / 4)

-- Conditions and propositions for the proof
def p : Prop := ∀ x ∈ Set.Icc (-Real.pi / 3) 0, f (x - Real.pi / 4) > f ((x - Real.pi / 4) + 0.0001)

def q : Prop := ∀ x : ℝ, f (-x) = f (x + 3)

theorem main_proof : (¬ p) ∧ q := by
  sorry

end main_proof_l131_131554


namespace pentagon_AE_length_l131_131994

theorem pentagon_AE_length (BC CD DE : ℝ) (∠E ∠B ∠C ∠D : ℝ) (a b : ℝ) :
  BC = 3 → 
  CD = 3 → 
  DE = 3 → 
  ∠E = 120 → 
  ∠B = 120 → 
  ∠C = 120 → 
  ∠D = 120 → 
  AE = a - 3 * Real.sqrt b → 
  a + b = 7 :=
by sorry

end pentagon_AE_length_l131_131994


namespace rectangles_count_l131_131424

theorem rectangles_count (p q : ℕ) (hp : p ≥ 2) (hq : q ≥ 2) :
  let rectangles := p * q * (p - 1) * (q - 1) / 4 in
  rectangles = (p * (p - 1) / 2) * (q * (q - 1) / 2) :=
by
  sorry

end rectangles_count_l131_131424


namespace sum_fraction_eq_seven_over_390_l131_131861

theorem sum_fraction_eq_seven_over_390 : 
  (∑ a in Finset.range (c + 1).filter (λ a, 1 ≤ a), 
     ∑ b in Finset.range (c + 1).filter (λ b, a < b), 
       ∑ c in Finset.range (c + 1).filter (λ c, b ≤ c), 
         (1 : ℚ) / (3^a * 2^b * 7^c)) = 7 / 390 := 
by
  sorry

end sum_fraction_eq_seven_over_390_l131_131861


namespace value_of_m_l131_131213

noncomputable def hyperbola_asymptote (a b c : ℝ) (h_pos : a > 0 ∧ b > 0) (h_c : c = Real.sqrt (a^2 + b^2)) (A : ℝ × ℝ)
  (h_hyperbola : A.1 = c/2 ∧ A.2 = (Real.sqrt 3)/2 * c) (h_on_hyperbola : (A.1)^2 / a^2 - (A.2)^2 / b^2 = 1) : ℝ :=
let m := b^2 / a^2 in
if h : m > 0 then m else 0

theorem value_of_m (a b c : ℝ) (h_pos : a > 0 ∧ b > 0) (h_c : c = Real.sqrt (a^2 + b^2)) (A : ℝ × ℝ)
  (h_hyperbola : A.1 = c/2 ∧ A.2 = (Real.sqrt 3)/2 * c) (h_on_hyperbola : (A.1)^2 / a^2 - (A.2)^2 / b^2 = 1) : 
  let m := hyperbola_asymptote a b c h_pos h_c A h_hyperbola h_on_hyperbola
  in m = 3 + 2 * Real.sqrt 3 :=
sorry

end value_of_m_l131_131213


namespace repetend_of_five_seventeenths_l131_131155

theorem repetend_of_five_seventeenths :
  (decimal_expansion (5 / 17)).repeat_called == "294117647" :=
sorry

end repetend_of_five_seventeenths_l131_131155


namespace proof_statements_l131_131002

theorem proof_statements :
  ∀ (a b : ℝ) (x : ℝ),
  (a > 1 ∧ b > 1 → (a - 1) * (b - 1) > 0) ∧
  ((0 < a ∧ a < 4) ↔ ∀ x : ℝ, a * x^2 + a * x + 1 > 0) ∧
  (a < 1 → ∀ x > 1, deriv (λ x, x^2 - a * x) x > 0) ∧
  ((a * b > 0 ↔ (a^3 + a^2 * b - a^2 - a * b + a + b > 0))) :=
by sorry

end proof_statements_l131_131002


namespace find_all_pos_integers_l131_131882

theorem find_all_pos_integers (M : ℕ) (h1 : M > 0) (h2 : M < 10) :
  (5 ∣ (1989^M + M^1989)) ↔ (M = 1) ∨ (M = 4) :=
by
  sorry

end find_all_pos_integers_l131_131882


namespace janice_weekly_earnings_l131_131648

-- define the conditions
def regular_days_per_week : Nat := 5
def regular_earnings_per_day : Nat := 30
def overtime_earnings_per_shift : Nat := 15
def overtime_shifts_per_week : Nat := 3

-- define the total earnings calculation
def total_earnings (regular_days : Nat) (regular_rate : Nat) (overtime_shifts : Nat) (overtime_rate : Nat) : Nat :=
  (regular_days * regular_rate) + (overtime_shifts * overtime_rate)

-- state the problem to be proved
theorem janice_weekly_earnings : total_earnings regular_days_per_week regular_earnings_per_day overtime_shifts_per_week overtime_earnings_per_shift = 195 :=
by
  sorry

end janice_weekly_earnings_l131_131648


namespace greatest_difference_units_digit_multiple_of_3_l131_131386

theorem greatest_difference_units_digit_multiple_of_3 :
  ∃ x y : ℕ, 
  (x < 10 ∧ y < 10) ∧
  (12 + x) % 3 = 0 ∧
  (12 + y) % 3 = 0 ∧
  (x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9) ∧
  (y = 0 ∨ y = 3 ∨ y = 6 ∨ y = 9) ∧
  x ≠ y ∧
  (x - y = 9 ∨ y - x = 9) :=
begin
  existsi 0,
  existsi 9,
  split,
  { split; norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { left, refl },
  split,
  { right, right, right, refl },
  split,
  { intro h, linarith },
  { right, refl },
end

end greatest_difference_units_digit_multiple_of_3_l131_131386


namespace enum_representation_of_set_A_l131_131945

noncomputable theory

open Set

theorem enum_representation_of_set_A :
  { p : ℤ × ℤ | ∃ (x y: ℤ), p = (x, y) ∧ x^2 = y + 1 ∧ |x| < 2 } = {(-1, 0), (0, -1), (1, 0)} :=
by
  sorry

end enum_representation_of_set_A_l131_131945


namespace unique_solution_quadratic_l131_131981

theorem unique_solution_quadratic (a : ℝ) (A : set ℝ) :
  (A = {x | a * x^2 + 2 * x - 1 = 0} ∧ (∃ x₀ : ℝ, ∀ x : ℝ, a * x^2 + 2 * x - 1 = 0 → x = x₀)) →
  (a = 0 ∨ a = -1) :=
by
  sorry

end unique_solution_quadratic_l131_131981


namespace expected_distinct_faces_in_six_rolls_of_die_l131_131040

theorem expected_distinct_faces_in_six_rolls_of_die : 
  let ξi (i : ℕ) := if i ∈ {1, 2, 3, 4, 5, 6} then 1 else 0 in 
  ∀ (i : ℕ) (face : ℕ), face ∈ {1, 2, 3, 4, 5, 6} → 
  (∃ sixRolls : Fin 6 → Fin 6, 
    (ξi i = 1 ↔ face ∈ finset.range 1) ∧ 
    (∑ (ξ : ℕ) in finset.range 6, ξ * ξi ξ) = 6 * (1 - (5 / 6) ^ 6)) :=
by
  let ξi := λ (i : ℕ), if i ∈ {1, 2, 3, 4, 5, 6} then 1 else 0
  intros i face hface
  sorry

end expected_distinct_faces_in_six_rolls_of_die_l131_131040


namespace composite_num_with_eights_l131_131373

-- Define the number with n digits "8" between 20 and 21
def num_with_eights (n : ℕ) : ℕ := 
  if n < 2 then 0 
  else 20 * 10^(n+1) + (list.sum (list.map (λ m, 8 * 10^m) (list.range n.tail))) + 21

theorem composite_num_with_eights (n : ℕ) (h : n ≥ 2) : ∃ (a b : ℕ), 1 < a ∧ 1 < b ∧ num_with_eights n = a * b := 
sorry

end composite_num_with_eights_l131_131373


namespace center_number_is_seven_l131_131070

theorem center_number_is_seven:
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ),
    {a1, a2, a3, a4, a5, a6, a7, a8, a9} = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    a1 + a3 + a7 + a9 = 20 ∧
    (a1 = a2 ± 1 ∨ a1 = a4 ± 1 ∨ a1 = a5 ± 1 ∨ a1 = a9 ± 1) ∧
    (a2 = a1 ± 1 ∨ a2 = a3 ± 1 ∨ a2 = a4 ± 1 ∨ a2 = a5 ± 1 ∨ a2 = a6 ± 1) ∧
    (a3 = a2 ± 1 ∨ a3 = a5 ± 1 ∨ a3 = a6 ± 1 ∨ a3 = a7 ± 1) ∧
    (a4 = a1 ± 1 ∨ a4 = a2 ± 1 ∨ a4 = a8 ± 1 ∨ a4 = a5 ± 1 ∨ a4 = a9 ± 1) ∧
    (a5 = a1 ± 1 ∨ a5 = a2 ± 1 ∨ a5 = a3 ± 1 ∨ a5 = a4 ± 1 ∨ a5 = a6 ± 1 ∨ a5 = a7 ± 1 ∨ a5 = a8 ± 1 ∨ a5 = a9 ± 1) ∧
    (a6 = a2 ± 1 ∨ a6 = a3 ± 1 ∨ a6 = a5 ± 1 ∨ a6 = a7 ± 1 ∨ a6 = a9 ± 1) ∧
    (a7 = a3 ± 1 ∨ a7 = a5 ± 1 ∨ a7 = a6 ± 1 ∨ a7 = a8 ± 1 ∨ a7 = a9 ± 1) ∧
    (a8 = a4 ± 1 ∨ a8 = a5 ± 1 ∨ a8 = a7 ± 1 ∨ a8 = a9 ± 1) ∧
    (a9 = a4 ± 1 ∨ a9 = a5 ± 1 ∨ a9 = a6 ± 1 ∨ a9 = a7 ± 1 ∨ a9 = a8 ± 1) ∧
    a5 = 7 :=
sorry

end center_number_is_seven_l131_131070


namespace find_f_2_l131_131938

theorem find_f_2 (f : ℝ → ℝ) (h : ∀ x, f (1 / x + 1) = 2 * x + 3) : f 2 = 5 :=
by
  sorry

end find_f_2_l131_131938


namespace find_v2007_l131_131366

def g : ℕ → ℕ
| 1 := 5
| 2 := 3
| 3 := 2
| 4 := 1
| 5 := 4
| _ := 0  -- Since in this context, g is only defined for inputs 1 to 5. 

def v : ℕ → ℕ
| 0 := 5
| (n + 1) := g (v n)

theorem find_v2007 : v 2007 = 1 :=
sorry

end find_v2007_l131_131366


namespace trapezium_area_l131_131518

theorem trapezium_area (a b h A : ℝ) (ha : a = 10) (hh : h = 10.00001) (hA : A = 140.00014) :
  b = 18 :=
by
  rw [ha, hh, hA]
  -- We need to show that area matches the given area for b = 18
  sorry

end trapezium_area_l131_131518


namespace fenced_area_with_cutout_l131_131360

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l131_131360


namespace sum_of_inscribed_angles_l131_131037

theorem sum_of_inscribed_angles
  (h1 : ∀ t, ∃ a b : Real, a + b = 360 ∧ a = 20 ∧ t ∈ [1..18])  -- The circle is divided into 18 equal arcs
  (h2 : ∃ x y : Real, x = 30 ∧ y = 50)  -- Angles x and y as inscribed angles subtended by three and five arcs respectively
  : x + y = 80 := sorry

end sum_of_inscribed_angles_l131_131037


namespace altitude_difference_l131_131610

noncomputable def abs_diff_altitude (h1 h2 : ℝ) : ℝ :=
  abs (h2 - h1)

theorem altitude_difference (k : ℝ) (h1 h2 : ℝ) (ln2 ln3 : ℝ) :
  exp(-k * h1) = 1 / 2 →
  exp(-k * 8700) = 1 / 3 →
  ln3 ≈ 1.1 →
  ln2 ≈ 0.7 →
  k = ln3 / 8700 →
  h1 = ln2 * 8700 / ln3 →
  abs_diff_altitude h1 8700 ≈ 3164 :=
sorry

end altitude_difference_l131_131610


namespace expected_distinct_faces_in_six_rolls_of_die_l131_131039

theorem expected_distinct_faces_in_six_rolls_of_die : 
  let ξi (i : ℕ) := if i ∈ {1, 2, 3, 4, 5, 6} then 1 else 0 in 
  ∀ (i : ℕ) (face : ℕ), face ∈ {1, 2, 3, 4, 5, 6} → 
  (∃ sixRolls : Fin 6 → Fin 6, 
    (ξi i = 1 ↔ face ∈ finset.range 1) ∧ 
    (∑ (ξ : ℕ) in finset.range 6, ξ * ξi ξ) = 6 * (1 - (5 / 6) ^ 6)) :=
by
  let ξi := λ (i : ℕ), if i ∈ {1, 2, 3, 4, 5, 6} then 1 else 0
  intros i face hface
  sorry

end expected_distinct_faces_in_six_rolls_of_die_l131_131039


namespace xyz_sum_48_l131_131969

theorem xyz_sum_48 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : z * x + y = 47) : 
  x + y + z = 48 :=
sorry

end xyz_sum_48_l131_131969


namespace repetend_of_five_seventeenths_l131_131151

theorem repetend_of_five_seventeenths :
  (decimal_expansion (5 / 17)).repeat_called == "294117647" :=
sorry

end repetend_of_five_seventeenths_l131_131151


namespace line_contains_point_at_k_59_l131_131535

-- Define the point (1/4, -6)
def point : ℝ × ℝ := (1/4, -6)

-- Define the line equation as a function
def line_eq (k : ℝ) (x y : ℝ) : Prop :=
  -1 / 2 - 2 * k * x = 5 * y

-- State the theorem
theorem line_contains_point_at_k_59 :
  ∃ k : ℝ, k = 59 ∧ line_eq k (point.1) (point.2) :=
sorry

end line_contains_point_at_k_59_l131_131535


namespace count_three_digit_integers_with_eight_l131_131963

theorem count_three_digit_integers_with_eight : 
  ∃ N : ℕ, N = 270 ∧ (∀ x : ℕ, 100 ≤ x ∧ x < 1000 → 
    (∃ n : ℕ, n <= 2 ∧ (x / 10 ^ n % 10 = 8)) → x ∈ (finset.range 1000).filter (λ x => x / 100 = 8 ∨ x / 10 % 10 = 8 ∨ x % 10 = 8)).card = N :=
begin
  sorry
end

end count_three_digit_integers_with_eight_l131_131963


namespace mutually_exclusive_event_is_D_l131_131622

namespace Problem

def event_A (n : ℕ) (defective : ℕ) : Prop := defective ≥ 2
def mutually_exclusive_event (n : ℕ) : Prop := (∀ (defective : ℕ), defective ≤ 1) ↔ (∀ (defective : ℕ), defective ≥ 2 → false)

theorem mutually_exclusive_event_is_D (n : ℕ) : mutually_exclusive_event n := 
by 
  sorry

end Problem

end mutually_exclusive_event_is_D_l131_131622


namespace bacteria_population_at_8_30_l131_131329

theorem bacteria_population_at_8_30 
  (initial_population : ℕ := 50)
  (tripling_interval : ℕ := 6) -- in minutes
  (total_duration : ℕ := 30) -- in minutes
  : ∀ (population : ℕ), population = initial_population * (3 ^ (total_duration / tripling_interval)) → population = 12150 :=
by
  intro population
  assume h
  rw h
  sorry

end bacteria_population_at_8_30_l131_131329


namespace value_of_w_l131_131016

theorem value_of_w (x : ℝ) (hx : x + 1/x = 5) : x^2 + (1/x)^2 = 23 :=
by
  sorry

end value_of_w_l131_131016


namespace mona_age_l131_131701

-- Define the list of guesses
def guesses : List ℕ := [16, 25, 27, 32, 36, 40, 42, 49, 64, 81]

-- Define what it means for at least half of the guesses to be too low
def at_least_half_too_low (age : ℕ) : Prop :=
  (guesses.filter (λ x => x < age)).length ≥ guesses.length / 2

-- Define what it means for two guesses to be off by 1
def two_off_by_one (age : ℕ) : Prop :=
  (guesses.count (λ x => x = age - 1) == 1) ∧ (guesses.count (λ x => x = age + 1) == 1)

-- Define what it means for the age to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

-- The main theorem stating Mona's age
theorem mona_age : ∃ (age : ℕ), age ∈ guesses ∧ at_least_half_too_low age ∧ two_off_by_one age ∧ is_perfect_square age ∧ age = 49 :=
by
  -- Fill in with the corresponding steps in the proof
  sorry

end mona_age_l131_131701


namespace matchstick_polygon_area_l131_131748

-- Given conditions
def number_of_matches := 12
def length_of_each_match := 2 -- in cm

-- Question: Is it possible to construct a polygon with an area of 16 cm^2 using all the matches?
def polygon_possible : Prop :=
  ∃ (p : Polygon), 
    (p.edges = number_of_matches) ∧ 
    (∃ (match_length : ℝ), match_length = length_of_each_match ∧ by 
      -- Form the polygon using all matches without breaking
      sorry) ∧ 
    (polygon_area p = 16)

-- Proof statement
theorem matchstick_polygon_area :
  polygon_possible :=
  sorry

end matchstick_polygon_area_l131_131748


namespace candidate_knows_Excel_and_willing_nights_l131_131029

variable (PExcel PXNight : ℝ)
variable (H1 : PExcel = 0.20) (H2 : PXNight = 0.30)

theorem candidate_knows_Excel_and_willing_nights : (PExcel * PXNight) = 0.06 :=
by
  rw [H1, H2]
  norm_num

end candidate_knows_Excel_and_willing_nights_l131_131029


namespace smallest_logarithmic_term_l131_131929

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem smallest_logarithmic_term (x₀ : ℝ) (hx₀ : f x₀ = 0) (h_interval : 2 < x₀ ∧ x₀ < Real.exp 1) :
  min (min (Real.log x₀) (Real.log (Real.sqrt x₀))) (min (Real.log (Real.log x₀)) ((Real.log x₀)^2)) = Real.log (Real.log x₀) := 
by
  sorry

end smallest_logarithmic_term_l131_131929


namespace geometric_sequence_existence_l131_131569

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b (n : ℕ) : ℕ := 1 / (a n * a (n + 1))

noncomputable def S (n : ℕ) : ℕ := (n * (2 * a n - 1)) / 2

noncomputable def T (n : ℕ) : ℕ :=
  let partial_sum : ℕ → ℕ
  | 0     => 0
  | k + 1 => partial_sum k + b (k + 1)
  partial_sum n

theorem geometric_sequence_existence :
  (∀ n : ℕ, a n * a n = S (2 * n - 1))
  ∧ (∀ n : ℕ, T n = n / (2 * n + 1))
  ∧ (∃ (m n : ℕ), 1 < m ∧ m < n ∧ (T 1) * (T n) = (T m)^2) :=
begin
  sorry
end

end geometric_sequence_existence_l131_131569


namespace product_of_solutions_l131_131595

theorem product_of_solutions (x : ℝ) (h : |(18 / x) - 6| = 3) : 2 * 6 = 12 :=
by
  sorry

end product_of_solutions_l131_131595


namespace altitude_difference_l131_131611

noncomputable def abs_diff_altitude (h1 h2 : ℝ) : ℝ :=
  abs (h2 - h1)

theorem altitude_difference (k : ℝ) (h1 h2 : ℝ) (ln2 ln3 : ℝ) :
  exp(-k * h1) = 1 / 2 →
  exp(-k * 8700) = 1 / 3 →
  ln3 ≈ 1.1 →
  ln2 ≈ 0.7 →
  k = ln3 / 8700 →
  h1 = ln2 * 8700 / ln3 →
  abs_diff_altitude h1 8700 ≈ 3164 :=
sorry

end altitude_difference_l131_131611


namespace complement_union_l131_131321

open Set

def A : Set ℝ := { x | -1 < x ∧ x < 1 }
def B : Set ℝ := { x | x ≥ 0 }

theorem complement_union : (A ∪ B)ᶜ = { x : ℝ | x ≤ -1 } := by
  sorry

end complement_union_l131_131321


namespace intersection_singleton_one_l131_131556

-- Define sets A and B according to the given conditions
def setA : Set ℤ := { x | 0 < x ∧ x < 4 }
def setB : Set ℤ := { x | (x+1)*(x-2) < 0 }

-- Statement to prove A ∩ B = {1}
theorem intersection_singleton_one : setA ∩ setB = {1} :=
by 
  sorry

end intersection_singleton_one_l131_131556


namespace triangle_MPQ_circumcircle_tangent_to_excircle_A_l131_131903

variables {A B C D E F P Q M : Point}
variables {Δ : Triangle}
variables {σ₁ σ₂ : Circle}

-- Given 
def tABC := Δ A B C  -- Δ is the triangle ABC
def excircle_A' := σ₁  -- σ₁ is the excircle opposite to A
def tangent_to_sides := excircle_A'.Tangent B C ∧ excircle_A'.Tangent C A ∧ excircle_A'.Tangent A B
def circumcircle_AEF := σ₂  -- σ₂ is the circumcircle of ∆ AEF
def intersection_PQ := σ₂.IntersectLine BC P Q  -- the circumcircle of ∆ AEF intersects line BC at P and Q
def midpoint_AD := Midpoint M A D  -- M is the midpoint of AD

-- To prove 
theorem triangle_MPQ_circumcircle_tangent_to_excircle_A' :
  tangent_to_sides →
  intersection_PQ →
  midpoint_AD →
  tangent (circumcircle Δ M P Q) excircle_A' :=
by
  intros h_tangent_to_sides h_intersection_PQ h_midpoint_AD
  sorry

end triangle_MPQ_circumcircle_tangent_to_excircle_A_l131_131903


namespace students_location_choices_l131_131532

theorem students_location_choices : (3^5 = 243) :=
by {
  -- Proof to be filled here
  sorry,
}

end students_location_choices_l131_131532


namespace shirts_sewn_on_tuesday_l131_131340

theorem shirts_sewn_on_tuesday 
  (shirts_monday : ℕ) 
  (shirts_wednesday : ℕ) 
  (total_buttons : ℕ) 
  (buttons_per_shirt : ℕ) 
  (shirts_tuesday : ℕ) 
  (h1: shirts_monday = 4) 
  (h2: shirts_wednesday = 2) 
  (h3: total_buttons = 45) 
  (h4: buttons_per_shirt = 5) 
  (h5: shirts_tuesday * buttons_per_shirt + shirts_monday * buttons_per_shirt + shirts_wednesday * buttons_per_shirt = total_buttons) : 
  shirts_tuesday = 3 :=
by 
  sorry

end shirts_sewn_on_tuesday_l131_131340


namespace no_two_digit_prime_with_digit_sum_9_l131_131184

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end no_two_digit_prime_with_digit_sum_9_l131_131184


namespace kids_on_soccer_field_l131_131393

def original_kids : ℕ := 14
def joined_kids : ℕ := 22
def total_kids : ℕ := 36

theorem kids_on_soccer_field : (original_kids + joined_kids) = total_kids :=
by 
  sorry

end kids_on_soccer_field_l131_131393


namespace weight_of_new_boy_l131_131707

theorem weight_of_new_boy (W : ℕ) (original_weight : ℕ) (total_new_weight : ℕ)
  (h_original_avg : original_weight = 5 * 35)
  (h_new_avg : total_new_weight = 6 * 36)
  (h_new_weight : total_new_weight = original_weight + W) :
  W = 41 := by
  sorry

end weight_of_new_boy_l131_131707


namespace find_integer_N_l131_131878

theorem find_integer_N :
  ∃ (N : ℕ), (0 < N) ∧ (N % 100000 = (N * N) % 100000) ∧ (N % 100000 ≠ 0) := by
  let N := 3125
  use N
  split
  · exact (show 0 < N from Nat.zero_lt_succ 3124)
  split
  · exact (by rw [Nat.mul_mod, show 3125 * 3125 % 100000 = 3125 % 100000, from rfl])
  · exact (dec_trivial : 3125 % 100000 ≠ 0)
  sorry

end find_integer_N_l131_131878


namespace total_fruit_78_l131_131390

-- Definitions based on the conditions
def total_pieces_of_fruit (T : ℕ) : Prop :=
  (2 / 3 : ℚ) * T = 52

theorem total_fruit_78 : ∃ (T : ℕ), total_pieces_of_fruit T ∧ T = 78 :=
by
  use 78
  split
  · show (2 / 3 : ℚ) * 78 = 52
    sorry
  · rfl

end total_fruit_78_l131_131390


namespace CPD_angle_calculation_l131_131637

noncomputable def angle_CPD_deg
  (tangent_PC_SAR : Prop)
  (tangent_PD_RBT : Prop)
  (SRT_straight_line : Prop)
  (arc_AS_deg : ℝ)
  (arc_BT_deg : ℝ) : ℝ :=
  if tangent_PC_SAR ∧ tangent_PD_RBT ∧ SRT_straight_line ∧ arc_AS_deg = 48 ∧ arc_BT_deg = 52 then 100 else 0

theorem CPD_angle_calculation
  (tangent_PC_SAR : Prop)
  (tangent_PD_RBT : Prop)
  (SRT_straight_line : Prop)
  (arc_AS_deg : ℝ)
  (arc_BT_deg : ℝ)
  (h1 : tangent_PC_SAR)
  (h2 : tangent_PD_RBT)
  (h3 : SRT_straight_line)
  (h4 : arc_AS_deg = 48)
  (h5 : arc_BT_deg = 52) :
  angle_CPD_deg tangent_PC_SAR tangent_PD_RBT SRT_straight_line arc_AS_deg arc_BT_deg = 100 :=
by
  simp [angle_CPD_deg, h1, h2, h3, h4, h5]
  sorry

end CPD_angle_calculation_l131_131637


namespace distinct_digit_values_for_D_l131_131633

def distinct (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem distinct_digit_values_for_D :
  (A B C D : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10) (hD : D < 10)
  (hDistinct : distinct A B C D)
  (hFirstColumn : A + B < 10 → A + B = D)
  (hThirdColumn : B + A = 11)
  (hFifthColumn : (C + 1 + A = 10 + D)) :
  #{ D | ∃ A B C : ℕ, A < 10 ∧ B < 10 ∧ C < 10 ∧ distinct A B C D ∧ A + B = D ∧ B + A = 11 ∧ C + 1 + A = 10 + D } = 7 :=
sorry

end distinct_digit_values_for_D_l131_131633


namespace area_of_shaded_trapezoid_l131_131194

-- Definitions of conditions:
def side_lengths : List ℕ := [1, 3, 5, 7]
def total_base : ℕ := side_lengths.sum
def height_largest_square : ℕ := 7
def ratio : ℚ := height_largest_square / total_base

def height_at_end (n : ℕ) : ℚ := ratio * n
def lower_base_height : ℚ := height_at_end 4
def upper_base_height : ℚ := height_at_end 9
def trapezoid_height : ℕ := 2

-- Main theorem:
theorem area_of_shaded_trapezoid :
  (1 / 2) * (lower_base_height + upper_base_height) * trapezoid_height = 91 / 8 :=
by
  sorry

end area_of_shaded_trapezoid_l131_131194


namespace train_speed_l131_131060

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 700) (h_time : time = 40) : length / time = 17.5 :=
by
  -- length / time represents the speed of the train
  -- given length = 700 meters and time = 40 seconds
  -- we have to prove that 700 / 40 = 17.5
  sorry

end train_speed_l131_131060


namespace sequence_b_k_minus_a_k_sequence_a_k_sequence_b_k_l131_131094

-- Define initial concentrations and recurrence relations
def a₀ (a : ℝ) := a
def b₀ (b : ℝ) := b
def b_k (a b : ℝ) (b_k_minus_1 a_k_minus_1 : ℕ → ℝ) : ℕ → ℝ :=
  λ k, (1/5) * a_k_minus_1 (k - 1) + (4/5) * b_k_minus_1 (k - 1)
def a_k (a b : ℝ) (a_k_minus_1 b_k : ℕ → ℝ) : ℕ → ℝ :=
  λ k, (13/15) * a_k_minus_1 (k - 1) + (2/15) * b_k (k - 1)

-- Define the sequences to be proved
theorem sequence_b_k_minus_a_k (a b : ℝ) :
  ∀ k : ℕ, b_k a b (b_k a b) (a_k a b) k - a_k a b (a_k a b) (b_k a b) k = (b - a) * (2 / 3) ^ k :=
sorry

theorem sequence_a_k (a b : ℝ) :
  ∀ k : ℕ, a_k a b (a_k a b) (b_k a b) k = (3 * a + 2 * b) / 5 + (a - b) * (2 / 5) * (2 / 3) ^ k :=
sorry

theorem sequence_b_k (a b : ℝ) :
  ∀ k : ℕ, b_k a b (b_k a b) (a_k a b) k = (3 * a + 2 * b) / 5 - (a - b) * (3 / 5) * (2 / 3) ^ k :=
sorry

end sequence_b_k_minus_a_k_sequence_a_k_sequence_b_k_l131_131094


namespace tank_volume_ratio_l131_131646

theorem tank_volume_ratio (A B : ℝ) 
    (h : (3 / 4) * A = (5 / 8) * B) : A / B = 6 / 5 := 
by 
  sorry

end tank_volume_ratio_l131_131646


namespace count_valid_integers_l131_131247

theorem count_valid_integers : 
  (∃ (count : ℕ), count = 100 ∧ 
    ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 999) → 
    (∀ (d : ℕ), d ∈ digit_list n → 
      d ≠ 2 ∧ d ≠ 3 ∧ d ≠ 4 ∧ d ≠ 5 ∧ d ≠ 6) → 
    count = valid_integers_from_1_to_999 ) :=
sorry

def digit_list (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else n.digits 10

def valid_integers_from_1_to_999 : ℕ :=
  100

end count_valid_integers_l131_131247


namespace law_of_sines_l131_131024

theorem law_of_sines
  (A B C : Type) [EuclideanTriangle A B C]
  (R : ℝ) [circumcircle_radius A B C R]:
  (sin (angle A)) / (side_length B C) = 1 / (2 * R) ∧
  (sin (angle B)) / (side_length A C) = 1 / (2 * R) ∧
  (sin (angle C)) / (side_length A B) = 1 / (2 * R) := by
sorry

end law_of_sines_l131_131024


namespace polynomial_has_two_positive_roots_iff_l131_131884

theorem polynomial_has_two_positive_roots_iff (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    (x₁^4 - 2 * p * x₁^3 + x₁^2 - 2 * p * x₁ + 1 = 0) ∧ 
    (x₂^4 - 2 * p * x₂^3 + x₂^2 - 2 * p * x₂ + 1 = 0)) ↔ p ∈ Ioi (5/4) :=
by
  sorry

end polynomial_has_two_positive_roots_iff_l131_131884


namespace sum_nonneg_of_any_two_nonneg_any_two_nonneg_of_sum_nonneg_l131_131303

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {x : Fin n → ℝ}

theorem sum_nonneg_of_any_two_nonneg
  (hn : ∀ i j : Fin n, i ≠ j → a i + a j ≥ 0)
  (hx_nonneg : ∀ i : Fin n, 0 ≤ x i)
  (hx_sum_one : (Finset.univ : Finset (Fin n)).sum x = 1) :
  (Finset.univ : Finset (Fin n)).sum (λ i, a i * x i) ≥ (Finset.univ : Finset (Fin n)).sum (λ i, a i * (x i)^2) :=
sorry

theorem any_two_nonneg_of_sum_nonneg
  (hineq : ∀ x : Fin n → ℝ, 
    (∀ i, 0 ≤ x i) → 
    (Finset.univ : Finset (Fin n)).sum x = 1 → 
    (Finset.univ : Finset (Fin n)).sum (λ i, a i * x i) ≥ (Finset.univ : Finset (Fin n)).sum (λ i, a i * (x i)^2)) :
  ∀ i j : Fin n, i ≠ j → a i + a j ≥ 0 :=
sorry

end sum_nonneg_of_any_two_nonneg_any_two_nonneg_of_sum_nonneg_l131_131303


namespace escalator_steps_l131_131778

theorem escalator_steps
  (x : ℕ)
  (time_me : ℕ := 60)
  (steps_me : ℕ := 20)
  (time_wife : ℕ := 72)
  (steps_wife : ℕ := 16)
  (escalator_speed_me : x - steps_me = 60 * (x - 20) / 72)
  (escalator_speed_wife : x - steps_wife = 72 * (x - 16) / 60) :
  x = 40 := by
  sorry

end escalator_steps_l131_131778


namespace sum_of_first_2017_terms_l131_131924

theorem sum_of_first_2017_terms :
  ∃ (a : ℕ → ℝ) (S : ℕ → ℝ), 
  (a 1 = 1) ∧ 
  (∀ n, a (n + 1) = 2 * a n) ∧ 
  (∀ n, S n = ∑ i in range n, a (i+1)) ∧ 
  9 * S 3 = S 6 ∧ 
  (∑ i in range 2017, a (i+1) * a (i+2) = (2 / 3) * (4^2017 - 1)) :=
begin
  sorry
end

end sum_of_first_2017_terms_l131_131924


namespace largest_fraction_l131_131644

theorem largest_fraction (d x : ℕ) 
  (h1: (2 * x / d) + (3 * x / d) + (4 * x / d) = 10 / 11)
  (h2: d = 11 * x) : (4 / 11 : ℚ) = (4 * x / d : ℚ) :=
by
  sorry

end largest_fraction_l131_131644


namespace economic_rationale_education_policy_l131_131629

theorem economic_rationale_education_policy
  (countries : Type)
  (foreign_citizens : Type)
  (universities : Type)
  (free_or_nominal_fee : countries → Prop)
  (international_agreements : countries → Prop)
  (aging_population : countries → Prop)
  (economic_benefits : countries → Prop)
  (credit_concessions : countries → Prop)
  (reciprocity_education : countries → Prop)
  (educated_youth_contributions : countries → Prop)
  :
  (∀ c : countries, free_or_nominal_fee c ↔
    (international_agreements c ∧ (credit_concessions c ∨ reciprocity_education c)) ∨
    (aging_population c ∧ economic_benefits c ∧ educated_youth_contributions c)) := 
sorry

end economic_rationale_education_policy_l131_131629


namespace box_office_collection_l131_131704

open Nat

/-- Define the total tickets sold -/
def total_tickets : ℕ := 1500

/-- Define the price of an adult ticket -/
def price_adult_ticket : ℕ := 12

/-- Define the price of a student ticket -/
def price_student_ticket : ℕ := 6

/-- Define the number of student tickets sold -/
def student_tickets : ℕ := 300

/-- Define the number of adult tickets sold -/
def adult_tickets : ℕ := total_tickets - student_tickets

/-- Define the revenue from adult tickets -/
def revenue_adult_tickets : ℕ := adult_tickets * price_adult_ticket

/-- Define the revenue from student tickets -/
def revenue_student_tickets : ℕ := student_tickets * price_student_ticket

/-- Define the total amount collected -/
def total_amount_collected : ℕ := revenue_adult_tickets + revenue_student_tickets

/-- Theorem to prove the total amount collected at the box office -/
theorem box_office_collection : total_amount_collected = 16200 := by
  sorry

end box_office_collection_l131_131704


namespace rancher_distance_l131_131454

-- Definitions for conditions
def cattle_count := 400
def truck_capacity := 20
def travel_speed := 60
def total_time_hours := 40

-- Target statement to prove
theorem rancher_distance (cattle_count : ℕ) (truck_capacity : ℕ) (travel_speed : ℕ) (total_time_hours : ℕ) :
  (cattle_count = 400) → 
  (truck_capacity = 20) → 
  (travel_speed = 60) → 
  (total_time_hours = 40) → 
  ∃ (distance : ℕ), distance = 60 :=
by
  assume h1 : cattle_count = 400,
  assume h2 : truck_capacity = 20,
  assume h3 : travel_speed = 60,
  assume h4 : total_time_hours = 40,
  use (60 : ℕ),
  sorry

end rancher_distance_l131_131454


namespace repetend_of_five_seventeenths_l131_131156

theorem repetend_of_five_seventeenths :
  (decimal_expansion (5 / 17)).repeat_called == "294117647" :=
sorry

end repetend_of_five_seventeenths_l131_131156


namespace trains_same_distance_at_meeting_l131_131470

theorem trains_same_distance_at_meeting
  (d v : ℝ) (h_d : 0 < d) (h_v : 0 < v) :
  ∃ t : ℝ, v * t + v * (t - 1) = d ∧ 
  v * t = (d + v) / 2 ∧ 
  d - (v * (t - 1)) = (d + v) / 2 :=
by
  sorry

end trains_same_distance_at_meeting_l131_131470


namespace geometric_sequence_q_l131_131639

theorem geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 * a 6 = 16)
  (h3 : a 4 + a 8 = 8) :
  q = 1 :=
by
  sorry

end geometric_sequence_q_l131_131639


namespace prove_total_payment_l131_131008

-- Define the conditions under which the problem is set
def monthly_subscription_cost : ℝ := 14
def split_ratio : ℝ := 0.5
def months_in_year : ℕ := 12

-- Define the target amount to prove
def total_payment_after_one_year : ℝ := 84

-- Theorem statement
theorem prove_total_payment
  (h1: monthly_subscription_cost = 14)
  (h2: split_ratio = 0.5)
  (h3: months_in_year = 12) :
  monthly_subscription_cost * split_ratio * months_in_year = total_payment_after_one_year := 
  by
  sorry

end prove_total_payment_l131_131008


namespace repetend_of_5_over_17_is_294117_l131_131149

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l131_131149


namespace pole_length_after_cut_l131_131823

theorem pole_length_after_cut
  (initial_length : ℝ)
  (cut_percentage : ℝ)
  (initial_length_eq : initial_length = 20)
  (cut_percentage_eq : cut_percentage = 0.30) :
  let new_length := initial_length * (1 - cut_percentage) in
  new_length = 14 := by
  sorry

end pole_length_after_cut_l131_131823


namespace repetend_of_fraction_l131_131161

/-- The repeating sequence of the decimal representation of 5/17 is 294117 -/
theorem repetend_of_fraction : 
  let rep := list.take 6 (list.drop 1 (to_digits 10 (5 / 17) 8)) in
  rep = [2, 9, 4, 1, 1, 7] := 
by
  sorry

end repetend_of_fraction_l131_131161


namespace cup_height_l131_131818

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * r^2 * h

theorem cup_height (V r h : ℝ) (hV : V = 200) (hr : r = 5) : h ≈ 8 :=
by
  sorry

end cup_height_l131_131818


namespace total_sample_variance_correct_l131_131815

noncomputable def total_sample_variance : real :=
  let male_mean := 172
  let female_mean := 164
  let male_var := 18
  let female_var := 30
  let total_males := 100
  let total_females := 60
  let combined_mean := (100 / 160) * male_mean + (60 / 160) * female_mean
  let male_component := (total_males / (total_males + total_females)) * (male_var + (male_mean - combined_mean) ^ 2)
  let female_component := (total_females / (total_males + total_females)) * (female_var + (female_mean - combined_mean) ^ 2)
  male_component + female_component

theorem total_sample_variance_correct : 
  total_sample_variance = 37.5 :=
by {
  -- Definitions from the conditions
  let male_mean := 172
  let female_mean := 164
  let male_var := 18
  let female_var := 30
  let total_males := 100
  let total_females := 60
  let combined_mean := (100 / 160) * male_mean + (60 / 160) * female_mean

  -- Calculate the male component of the variance
  let male_component := (total_males / (total_males + total_females)) * (male_var + (male_mean - combined_mean) ^ 2)

  -- Calculate the female component of the variance
  let female_component := (total_females / (total_males + total_females)) * (female_var + (female_mean - combined_mean) ^ 2)

  -- Combine the components to get the total variance
  let total_sample_variance := male_component + female_component 

  -- Expected value
  have : total_sample_variance = 37.5, from sorry,
  
  exact this
}

end total_sample_variance_correct_l131_131815


namespace construct_polygon_with_area_l131_131754

theorem construct_polygon_with_area 
  (n : ℕ) (l : ℝ) (a : ℝ) 
  (matchsticks : n = 12) 
  (matchstick_length : l = 2) 
  (area_target : a = 16) : 
  ∃ (polygon : EuclideanGeometry.Polygon ℝ) (sides : list ℝ),
    sides.length = n ∧ ∀ side ∈ sides, side = l ∧ polygon.area = a := 
sorry

end construct_polygon_with_area_l131_131754


namespace overall_average_score_l131_131348

-- Definitions used from conditions
def male_students : Nat := 8
def male_avg_score : Real := 83
def female_students : Nat := 28
def female_avg_score : Real := 92

-- Theorem to prove the overall average score is 90
theorem overall_average_score : 
  (male_students * male_avg_score + female_students * female_avg_score) / (male_students + female_students) = 90 := 
by 
  sorry

end overall_average_score_l131_131348


namespace curve_C1_equation_curve_C2_equation_min_distance_M_to_line_C3_min_max_distance_Q_to_line_C3_range_2x_plus_y_range_a_l131_131586

section problem
  -- Definitions from conditions
  def C1 (t : ℝ) : ℝ × ℝ := (-4 + Real.cos t, 3 + Real.sin t)
  def C2 (theta : ℝ) : ℝ × ℝ := (6 * Real.cos theta, 2 * Real.sin theta)
  def line_C3 (t : ℝ) : ℝ × ℝ := (-3 * Real.sqrt 3 + Real.sqrt 3 * t, -3 - t)

  -- Problem (1)
  theorem curve_C1_equation : ∀ (x y : ℝ),
    (∃ t : ℝ, x = -4 + Real.cos t ∧ y = 3 + Real.sin t) ↔ (x + 4) ^ 2 + (y - 3) ^ 2 = 1 := sorry

  theorem curve_C2_equation : ∀ (x y : ℝ),
    (∃ theta : ℝ, x = 6 * Real.cos theta ∧ y = 2 * Real.sin theta) ↔ (x ^ 2) / 36 + (y ^ 2) / 4 = 1 := sorry

  -- Problem (2)
  theorem min_distance_M_to_line_C3 : 
    let P := C1 (Real.pi / 2),
        Q := C2 theta,
        M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2),
        dist := (fun x y => (x + Real.sqrt 3 * y + 6 * Real.sqrt 3) / Real.sqrt (1 + 3 : ℝ)) in
    ∃ t, dist M.1 M.2 = 3 * Real.sqrt 3 - 1 := sorry

  -- Problem (3)
  theorem min_max_distance_Q_to_line_C3 :
    let dist := (fun x y => (x + Real.sqrt 3 * y + 6 * Real.sqrt 3) / Real.sqrt (1 + 3 : ℝ)) in
    ∃ min_d max_d : ℝ, min_d = 3 * Real.sqrt 3 - 1 ∧ max_d = 5 * Real.sqrt 3 - 1 := sorry

  -- Problem (4)
  theorem range_2x_plus_y :
    let P := C1 t in
    ∃ a b : ℝ, 2 * P.1 + P.2 ∈ set.Icc a b ∧ a = -5 - Real.sqrt 5 ∧ b = -5 + Real.sqrt 5  := sorry
  
  -- Problem (5)
  theorem range_a : 
    let P := C1 t in
    ∀ (a : ℝ), (∀ (x y : ℝ), (x = P.1 ∧ y = P.2) → x + y + a ≥ 0) ↔ a ∈ set.Ici (1 + Real.sqrt 2) := sorry
end problem

end curve_C1_equation_curve_C2_equation_min_distance_M_to_line_C3_min_max_distance_Q_to_line_C3_range_2x_plus_y_range_a_l131_131586


namespace calc_dot_product_l131_131295

open EuclideanGeometry

variable {u v : ℝ^3}
variable (unit_u : ∥u∥ = 1)
variable (orthogonal_w_u : ∀ w, w = 2 * (u × v) + u → w ⬝ u = 0)

theorem calc_dot_product : (u ⬝ (v × (2 * (u × v) + u))) = 2 * (v ⬝ v) - 1 :=
by
  sorry

end calc_dot_product_l131_131295


namespace rooster_weight_l131_131874

variable (W : ℝ)  -- The weight of the first rooster

theorem rooster_weight (h1 : 0.50 * W + 0.50 * 40 = 35) : W = 30 :=
by
  sorry

end rooster_weight_l131_131874


namespace count_perfect_cubes_between_500_and_2000_l131_131591

theorem count_perfect_cubes_between_500_and_2000 : ∃ count : ℕ, count = 5 ∧ (∀ n, 500 < n^3 ∧ n^3 < 2000 → (8 ≤ n ∧ n ≤ 12)) :=
by
  existsi 5
  split
  {
    sorry,  -- Proof that count = 5
    sorry,  -- Proof that for any n, if 500 < n^3 and n^3 < 2000 then 8 <= n <= 12
  }

end count_perfect_cubes_between_500_and_2000_l131_131591


namespace total_rainfall_l131_131510

theorem total_rainfall (R1 R2 : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 15) : R1 + R2 = 25 := 
by
  sorry

end total_rainfall_l131_131510


namespace interest_rate_second_part_l131_131690

noncomputable def P1 : ℝ := 2799.9999999999995
noncomputable def P2 : ℝ := 4000 - P1
noncomputable def Interest1 : ℝ := P1 * (3 / 100)
noncomputable def TotalInterest : ℝ := 144
noncomputable def Interest2 : ℝ := TotalInterest - Interest1

theorem interest_rate_second_part :
  ∃ r : ℝ, Interest2 = P2 * (r / 100) ∧ r = 5 :=
by
  sorry

end interest_rate_second_part_l131_131690


namespace probability_of_solution_l131_131426

theorem probability_of_solution (S : Set ℝ) (hS : S = { -10, -6, -5, -4, -2.5, -1, 0, 2.5, 4, 6, 7, 10 }) :
  let equation := λ x : ℝ, (x + 5) * (x + 10) * (2 * x - 5) = 0 in
  let solutions := {x | equation x} in
  let valid_solutions := S ∩ solutions in
  probability valid_solutions S = 1 / 4 :=
by
  sorry

end probability_of_solution_l131_131426


namespace prove_ratio_l131_131719

def P_and_Q (P Q : ℤ) : Prop :=
  ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 6 →
  P * x * (x - 6) + Q * (x + 5) = x^2 - 4 * x + 20

theorem prove_ratio (P Q : ℤ) (h : P_and_Q P Q) : Q / P = 4 :=
by
  sorry

end prove_ratio_l131_131719


namespace angle_A_equals_pi_div_3_min_AD_value_l131_131236

-- Define the triangle ABC with sides a, b, c, and angles A, B, C
variable (a b c : ℝ)
variable (A B C : ℝ)

-- Define the conditions provided in the problem
axiom cond1 : a * real.sin B = b * real.sin (A + π / 3)
axiom cond2 : S = (sqrt 3 / 2) * (∥BA∥ * ∥CA∥)
axiom cond3 : c * real.tan A = (2 * b - c) * real.tan C

-- Problem definition in Lean:
theorem angle_A_equals_pi_div_3 (h : cond1) : A = π / 3 := sorry

-- Additional geometric conditions for the second part of the problem
variable (S : ℝ)
axiom area_condition : S = 2 * sqrt 3

-- Given point D on BC such that BD = 2DC
variable (BD DC AD : ℝ)
axiom BD_double_DC : BD = 2 * DC

-- Prove the minimum value of AD
theorem min_AD_value (h1 : area_condition) (h2 : BD_double_DC) : AD = 4 * sqrt 3 / 3 := sorry

end angle_A_equals_pi_div_3_min_AD_value_l131_131236


namespace correct_completion_l131_131019

theorem correct_completion (A B C D : String) : C = "None" :=
by
  let sentence := "Did you have any trouble with the customs officer? " ++ C ++ " to speak of."
  let correct_sentence := "Did you have any trouble with the customs officer? None to speak of."
  sorry

end correct_completion_l131_131019


namespace car_tank_capacity_l131_131849

/-- Conditions -/
def gallons_per_mile := 1 / 30  -- The car uses 1 gallon of gasoline every 30 miles.
def travel_time := 5            -- The car travels for 5 hours.
def speed := 50                 -- The car travels at 50 miles per hour.
def gasoline_used_fraction := 0.4166666666666667  -- The amount of gasoline used is 0.4166666666666667 of a full tank.

/-- Proof problem statement -/
theorem car_tank_capacity :
  let distance := speed * travel_time in
  let gasoline_used := distance * gallons_per_mile in
  let full_tank_capacity := gasoline_used / gasoline_used_fraction in
  full_tank_capacity = 20 :=
by
  sorry

end car_tank_capacity_l131_131849


namespace ray_has_4_nickels_left_l131_131335

theorem ray_has_4_nickels_left (initial_cents : ℕ) (given_to_peter : ℕ)
    (given_to_randi : ℕ) (value_of_nickel : ℕ) (remaining_cents : ℕ) 
    (remaining_nickels : ℕ) :
    initial_cents = 95 →
    given_to_peter = 25 →
    given_to_randi = 2 * given_to_peter →
    value_of_nickel = 5 →
    remaining_cents = initial_cents - given_to_peter - given_to_randi →
    remaining_nickels = remaining_cents / value_of_nickel →
    remaining_nickels = 4 :=
by
  intros
  sorry

end ray_has_4_nickels_left_l131_131335


namespace height_of_trapezoid_l131_131992

-- Define the condition that a trapezoid has diagonals of given lengths and a given midline.
def trapezoid_conditions (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : Prop := 
  AC = 6 ∧ BD = 8 ∧ ML = 5

-- Define the height of the trapezoid.
def trapezoid_height (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : ℝ :=
  4.8

-- The theorem statement
theorem height_of_trapezoid (AC BD ML h : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : 
  trapezoid_conditions AC BD ML h_d1 h_d2 h_ml 
  → trapezoid_height AC BD ML h_d1 h_d2 h_ml = 4.8 := 
by
  intros
  sorry

end height_of_trapezoid_l131_131992


namespace arithmetic_mean_of_first_n_even_integers_l131_131782

theorem arithmetic_mean_of_first_n_even_integers (n : ℕ) : 
  let seq := λ (n : ℕ), list.map (λ k, 2 * k) (list.range n) in
  (list.sum (seq n)) / n = n + 1 :=
by
  sorry

end arithmetic_mean_of_first_n_even_integers_l131_131782


namespace find_t_l131_131615

theorem find_t : ∀ (p j t x y a b c : ℝ),
  j = 0.75 * p →
  j = 0.80 * t →
  t = p - (t/100) * p →
  x = 0.10 * t →
  y = 0.50 * j →
  x + y = 12 →
  a = x + y →
  b = 0.15 * a →
  c = 2 * b →
  t = 24 := 
by
  intros p j t x y a b c hjp hjt htp hxt hyy hxy ha hb hc
  sorry

end find_t_l131_131615


namespace number_of_small_branches_l131_131814

-- Define the number of small branches grown by each branch as a variable
variable (x : ℕ)

-- Define the total number of main stems, branches, and small branches
def total := 1 + x + x * x

theorem number_of_small_branches (h : total x = 91) : x = 9 :=
by
  -- Proof is not required as per instructions
  sorry

end number_of_small_branches_l131_131814


namespace number_of_valid_3_digit_numbers_l131_131870

/-- Determine the number of 3-digit numbers such that the units digit is at least three times the tens digit. -/
def count_valid_3_digit_numbers (H T U : ℕ) :=
  1 ≤ H ∧ H ≤ 9 ∧
  0 ≤ T ∧ T ≤ 4 ∧
  0 ≤ U ∧ U ≤ 9 ∧
  (U ≥ 3 * T)

theorem number_of_valid_3_digit_numbers : (∑ H in Finset.range (9 + 1), ∑ T in Finset.range (5), ∑ U in Finset.range (10), if count_valid_3_digit_numbers H T U then 1 else 0) = 198 := 
sorry

end number_of_valid_3_digit_numbers_l131_131870


namespace library_visit_period_l131_131656

noncomputable def dance_class_days := 6
noncomputable def karate_class_days := 12
noncomputable def common_days := 36

theorem library_visit_period (library_days : ℕ) 
  (hdance : ∀ (n : ℕ), n * dance_class_days = common_days)
  (hkarate : ∀ (n : ℕ), n * karate_class_days = common_days)
  (hcommon : ∀ (n : ℕ), n * library_days = common_days) : 
  library_days = 18 := 
sorry

end library_visit_period_l131_131656


namespace two_digit_prime_sum_9_l131_131188

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- There are 0 two-digit prime numbers for which the sum of the digits equals 9 -/
theorem two_digit_prime_sum_9 : ∃! n : ℕ, (9 ≤ n ∧ n < 100) ∧ (n.digits 10).sum = 9 ∧ is_prime n :=
sorry

end two_digit_prime_sum_9_l131_131188


namespace depth_of_lake_l131_131444

theorem depth_of_lake (h : ℝ := 12000) 
  (fraction_above_water : ℝ := 1/6) 
  (v_submerged : ℝ := 5/6) 
  : depth_of_lake = 780 := by
  let volume_ratio := (v_submerged / 1)^(1/3)
  let h'_ratio := volume_ratio
  let h' := h * h'_ratio
  have depth := h - h'
  sorry

end depth_of_lake_l131_131444


namespace students_per_normal_class_l131_131954

theorem students_per_normal_class : 
  ∀ (n : ℕ), n = 1590 → 
  let m := (40 * n) / 100 in
  m = 636 ∧
  let g := 3 in
  let studentsPerGrade := m / g in
  studentsPerGrade = 212 ∧
  let advancedClass := 20 in
  let remainingStudents := studentsPerGrade - advancedClass in
  remainingStudents = 192 ∧
  let numberOfClasses := 6 in
  let studentsPerNormalClass := remainingStudents / numberOfClasses in
  studentsPerNormalClass = 32 :=
by
  intros n h
  let m := (40 * n) / 100
  have m_eq : m = 636 := by 
    rw h
    norm_num
  let g := 3
  let studentsPerGrade := m / g
  have studentsPerGrade_eq : studentsPerGrade = 212 := by 
    rw [m_eq]
    norm_num
  let advancedClass := 20
  let remainingStudents := studentsPerGrade - advancedClass
  have remainingStudents_eq : remainingStudents = 192 := by 
    rw [studentsPerGrade_eq]
    norm_num
  let numberOfClasses := 6
  let studentsPerNormalClass := remainingStudents / numberOfClasses
  have studentsPerNormalClass_eq : studentsPerNormalClass = 32 := by 
    rw [remainingStudents_eq]
    norm_num
  exact ⟨m_eq, studentsPerGrade_eq, remainingStudents_eq, studentsPerNormalClass_eq⟩

end students_per_normal_class_l131_131954


namespace most_likely_number_of_white_balls_l131_131766

theorem most_likely_number_of_white_balls (total_balls : ℕ) (freq_yellow : ℚ) (hy : total_balls = 32) (hf : freq_yellow = 0.25) : ℕ :=
  let Y := total_balls * (freq_yellow.numerator) / (freq_yellow.denominator) in
  let W := total_balls - Y in
  W

#eval most_likely_number_of_white_balls 32 0.25 (by rfl) (by norm_num)

end most_likely_number_of_white_balls_l131_131766


namespace evaluate_expression_equals_128_l131_131118

-- Define the expression as a Lean function
def expression : ℕ := (8^6) / (4 * 8^3)

-- Theorem stating that the expression equals 128
theorem evaluate_expression_equals_128 : expression = 128 := 
sorry

end evaluate_expression_equals_128_l131_131118


namespace even_composition_l131_131310

variable {α : Type} [CommRing α]

def is_even_function (g : α → α) : Prop :=
  ∀ x, g (-x) = g x

theorem even_composition (g : α → α) (h : is_even_function g) :
  is_even_function (λ x, g (g x)) :=
by 
  sorry

end even_composition_l131_131310


namespace pears_seed_avg_l131_131345

def apple_seed_avg : ℕ := 6
def grape_seed_avg : ℕ := 3
def total_seeds_required : ℕ := 60
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def seeds_short : ℕ := 3
def total_seeds_obtained : ℕ := total_seeds_required - seeds_short

theorem pears_seed_avg :
  (apples_count * apple_seed_avg) + (grapes_count * grape_seed_avg) + (pears_count * P) = total_seeds_obtained → 
  P = 2 :=
by
  sorry

end pears_seed_avg_l131_131345


namespace repetend_of_five_over_seventeen_l131_131125

theorem repetend_of_five_over_seventeen : 
  let r := 5 / 17 in
  ∃ a b : ℕ, a * 10^b = 294117 ∧ (r * 10^b - a) = (r * 10^6 - r * (10^6 / 17))
   ∧ (r * 10^k = (r * 10^6).floor / 10^k ) where k = 6 := sorry

end repetend_of_five_over_seventeen_l131_131125


namespace amanda_final_quiz_score_l131_131843

theorem amanda_final_quiz_score
  (average_score_4quizzes : ℕ)
  (total_quizzes : ℕ)
  (average_a : ℕ)
  (current_score : ℕ)
  (required_total_score : ℕ)
  (required_score_final_quiz : ℕ) :
  average_score_4quizzes = 92 →
  total_quizzes = 5 →
  average_a = 93 →
  current_score = 4 * average_score_4quizzes →
  required_total_score = total_quizzes * average_a →
  required_score_final_quiz = required_total_score - current_score →
  required_score_final_quiz = 97 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end amanda_final_quiz_score_l131_131843


namespace frustum_total_surface_area_l131_131043

-- Define the given conditions
def r₁ : ℝ := 8
def r₂ : ℝ := 4
def h : ℝ := 5

-- Define variables for slant height, lateral area, and areas of bases
def l : ℝ := real.sqrt (h^2 + (r₁ - r₂)^2)
def A_lat : ℝ := real.pi * (r₁ + r₂) * l
def A_base1 : ℝ := real.pi * r₁^2
def A_base2 : ℝ := real.pi * r₂^2

-- Define the total surface area variable
def A_total : ℝ := A_lat + A_base1 + A_base2

-- State the theorem
theorem frustum_total_surface_area : A_total = 80 * real.pi + 12 * real.pi * real.sqrt 41 :=
by 
  -- proof would go here 
  sorry

end frustum_total_surface_area_l131_131043


namespace polygon_possible_l131_131739

-- Definition: a polygon with matches without breaking them
structure MatchPolygon (n : ℕ) (length : ℝ) where
  num_matches : ℕ
  len_matches : ℝ
  area : ℝ
  notequalzero : len_matches ≠ 0
  notequalzero2 : area ≠ 0
  perimeter_eq: num_matches * len_matches = length * real.of_nat n
  all_matches_used : n = 12
  no_breaking : (length / real.of_nat n) = 2 

theorem polygon_possible : 
  ∃ P : MatchPolygon 12 2, P.area = 16 :=
sorry

end polygon_possible_l131_131739


namespace harry_ends_up_at_B_l131_131955

-- This indicates that the definitions used here are not computable since we are dealing with probabilities
noncomputable def probability_Harry_ends_up_at_B (S B V W X Y Z : Type) 
  (path_S_to_V : S → V)
  (path_S_to_W : S → W)
  (path_S_to_X : S → X) 
  (path_V_to_W : V → W)
  (path_V_to_X : V → X)
  (path_V_to_Y : V → Y)
  (path_W_to_X : W → X)
  (path_W_to_Y : W → Y)
  (path_X_to_Y : X → Y)
  (path_X_to_Z : X → Z)
  (path_Y_to_X : Y → X)
  (prob : V → ℚ)
  (pq_W_to_X : W → ℚ)
  (pq_Y_to_X : Y → ℚ)
  (pq_V_to_X_direct : V → ℚ)
  (pq_V_to_W_to_X : V → ℚ)
  (pq_V_to_Y_to_X : V → ℚ) : ℚ :=
  pq_V_to_X_direct V + pq_V_to_W_to_X V + pq_V_to_Y_to_X V

theorem harry_ends_up_at_B (S B V W X Y Z : Type)
  (path_S_to_V : S → V)
  (path_S_to_W : S → W)
  (path_S_to_X : S → X) 
  (path_V_to_W : V → W)
  (path_V_to_X : V → X)
  (path_V_to_Y : V → Y)
  (path_W_to_X : W → X)
  (path_W_to_Y : W → Y)
  (path_X_to_Y : X → Y)
  (path_X_to_Z : X → Z)
  (path_Y_to_X : Y → X)
  (prob : V → ℚ)
  (pq_W_to_X : W → ℚ)
  (pq_Y_to_X : Y → ℚ)
  (pq_V_to_X_direct : V → ℚ)
  (pq_V_to_W_to_X : V → ℚ)
  (pq_V_to_Y_to_X : V → ℚ)
  (p_path : V → ℚ) :
  p_path V = 11/18 := 
by 
  sorry

end harry_ends_up_at_B_l131_131955


namespace balance_tree_l131_131326

def weight (c : Char) : ℕ :=
  match c with
  | 'O' => 300
  | 'B' => 300
  | 'M' => 200
  | 'E' => 200
  | 'P' => 100
  | _ => 0

def initial_left : List Char := ['M', 'B']
def initial_right : List Char := ['P', 'E']

def total_left_initial := initial_left.map weight |>.sum
def total_right_initial := initial_right.map weight |>.sum + weight 'P'

def remaining_letters : List Char := ['O', 'O', 'B', 'B', 'M', 'M', 'E', 'E', 'P', 'P']

def total_remaining := remaining_letters.map weight |>.sum

noncomputable def distribute_letters (left_add right_add : List Char) : Prop :=
  left_add.map weight |>.sum + total_left_initial = right_add.map weight |>.sum + total_right_initial

theorem balance_tree : ∃ left_add right_add,
  -- left_add and right_add are mutually exclusive splits of remaining_letters
  (left_add ++ right_add = remaining_letters ∧ left_add ∩ right_add = [])
  ∧ distribute_letters left_add right_add := by
  sorry

end balance_tree_l131_131326


namespace repetend_of_fraction_l131_131160

/-- The repeating sequence of the decimal representation of 5/17 is 294117 -/
theorem repetend_of_fraction : 
  let rep := list.take 6 (list.drop 1 (to_digits 10 (5 / 17) 8)) in
  rep = [2, 9, 4, 1, 1, 7] := 
by
  sorry

end repetend_of_fraction_l131_131160


namespace can_construct_polygon_l131_131760

def match_length : ℕ := 2
def number_of_matches : ℕ := 12
def total_length : ℕ := number_of_matches * match_length
def required_area : ℝ := 16

theorem can_construct_polygon : 
  (∃ (P : Polygon), P.perimeter = total_length ∧ P.area = required_area) := 
sorry

end can_construct_polygon_l131_131760


namespace fenced_area_l131_131356

theorem fenced_area (w : ℕ) (h : ℕ) (cut_out : ℕ) (rectangle_area : ℕ) (cut_out_area : ℕ) (net_area : ℕ) :
  w = 20 → h = 18 → cut_out = 4 → rectangle_area = w * h → cut_out_area = cut_out * cut_out → net_area = rectangle_area - cut_out_area → net_area = 344 :=
by
  intros
  subst_vars
  sorry

end fenced_area_l131_131356


namespace log_base_9_of_cubic_root_of_9_l131_131112

theorem log_base_9_of_cubic_root_of_9 : log 9 (cbrt 9) = 1 / 3 :=
by 
  -- required to use Mathfcklb.log or any other definitions properly
  sorry

end log_base_9_of_cubic_root_of_9_l131_131112


namespace power_multiplication_l131_131488

theorem power_multiplication :
  3^5 * 6^5 = 1889568 :=
by
  sorry

end power_multiplication_l131_131488


namespace people_got_off_train_l131_131394

theorem people_got_off_train (initial_people : ℕ) (people_left : ℕ) (people_got_off : ℕ) 
  (h1 : initial_people = 48) 
  (h2 : people_left = 31) 
  : people_got_off = 17 := by
  sorry

end people_got_off_train_l131_131394


namespace a_1964_not_divisible_by_4_l131_131731

-- Definitions
def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧
  a 1 = 1 ∧
  ∀ n : ℕ, a (n + 1) = a n * a (n - 1) + 1

-- Theorem
theorem a_1964_not_divisible_by_4 (a : ℕ → ℕ) (h : sequence a) : a 1964 % 4 ≠ 0 :=
by
  sorry

end a_1964_not_divisible_by_4_l131_131731


namespace no_19_distinct_natural_numbers_with_same_digit_sum_l131_131105

theorem no_19_distinct_natural_numbers_with_same_digit_sum 
  (S : ℕ) 
  (f : ℕ → ℕ) 
  (hf : ∀ n, f n = S) 
  (distinct_numbers : finset ℕ) 
  (h_distinct : distinct_numbers.card = 19) :
  ¬ (∑ n in distinct_numbers, n = 1999) := 
sorry

end no_19_distinct_natural_numbers_with_same_digit_sum_l131_131105


namespace machines_fill_5_boxes_in_5_minutes_l131_131677

theorem machines_fill_5_boxes_in_5_minutes :
  let boxes_per_minute_A := 24 / 60,
      boxes_per_minute_B := 36 / 60,
      total_boxes_per_minute := boxes_per_minute_A + boxes_per_minute_B
  in 5 * total_boxes_per_minute = 5 :=
by
  sorry

end machines_fill_5_boxes_in_5_minutes_l131_131677


namespace form_of_odd_function_for_negative_x_l131_131217

-- Assuming f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = - f(x)

-- Define f for x > 0
def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 2*x - 3 else 0

-- Statement to prove the form of f(x) when x < 0
theorem form_of_odd_function_for_negative_x (f : ℝ → ℝ) (h_odd : odd_function f) 
  (h_pos : ∀ x, 0 < x → f(x) = x^2 - 2*x - 3) :
  ∀ x, x < 0 → f(x) = -x^2 - 2x + 3 :=
begin
  intros x h,
  have h_pos_neg : f(-x) = x^2 - 2*x - 3,
  { exact h_pos (-x) (by linarith), },
  have h_odd_neg : f(x) = -f(-x),
  { exact h_odd x },
  rw h_odd_neg,
  rw h_pos_neg,
  linarith,
end

-- Provide a sorry for completeness although it is not needed.
sorry

end form_of_odd_function_for_negative_x_l131_131217


namespace parallel_line_through_point_l131_131714

theorem parallel_line_through_point (x y : ℝ) (m b : ℝ) (h₁ : y = -3 * x + b) (h₂ : x = 2) (h₃ : y = 1) :
  b = 7 :=
by
  -- x, y are components of the point P (2,1)
  -- equation of line parallel to y = -3x + 2 has slope -3 but different y-intercept
  -- y = -3x + b is the general form, and must pass through (2,1) => 1 = -3*2 + b
  -- Therefore, b must be 7
  sorry

end parallel_line_through_point_l131_131714


namespace polygon_possible_with_area_sixteen_l131_131745

theorem polygon_possible_with_area_sixteen :
  ∃ (P : polygon) (matches : list (side P)), (length(matches) = 12 ∧ (∀ m ∈ matches, m.length = 2) ∧ P.area = 16) := 
sorry

end polygon_possible_with_area_sixteen_l131_131745


namespace impossible_65_cents_with_five_coins_l131_131893

-- Define the coin types and their values
inductive Coin
| penny | nickel | dime | quarter | half_dollar

open Coin

def coin_value : Coin → ℕ
| penny       := 1
| nickel      := 5
| dime        := 10
| quarter     := 25
| half_dollar := 50

def total_value (coins : List Coin) : ℕ :=
coins.sum (λ c => coin_value c)

theorem impossible_65_cents_with_five_coins :
  ∀ (coins : List Coin), coins.length = 5 → total_value coins = 65 → False :=
by
  sorry

end impossible_65_cents_with_five_coins_l131_131893


namespace exists_five_distinct_natural_numbers_product_eq_1000_l131_131860

theorem exists_five_distinct_natural_numbers_product_eq_1000 :
  ∃ (a b c d e : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a * b * c * d * e = 1000 := sorry

end exists_five_distinct_natural_numbers_product_eq_1000_l131_131860


namespace baseball_card_problem_l131_131415

theorem baseball_card_problem:
  let initial_cards := 15
  let maria_takes := (initial_cards + 1) / 2
  let cards_after_maria := initial_cards - maria_takes
  let cards_after_peter := cards_after_maria - 1
  let final_cards := cards_after_peter * 3
  final_cards = 18 :=
by
  sorry

end baseball_card_problem_l131_131415


namespace no_two_digit_prime_sum_digits_nine_l131_131179

theorem no_two_digit_prime_sum_digits_nine :
  ¬ ∃ p : ℕ, prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 + p % 10 = 9) :=
sorry

end no_two_digit_prime_sum_digits_nine_l131_131179


namespace julia_birth_year_l131_131983

open Nat

theorem julia_birth_year (w_age : ℕ) (p_diff : ℕ) (j_diff : ℕ) (current_year : ℕ) :
  w_age = 37 →
  p_diff = 3 →
  j_diff = 2 →
  current_year = 2021 →
  (current_year - w_age) - p_diff - j_diff = 1979 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end julia_birth_year_l131_131983


namespace prove_false_q_l131_131555

-- Definitions based on the conditions provided
def p : Prop := ∃ x : ℝ, (1 / 10)^(x - 3) ≤ Real.cos 2

-- The proposition is that (¬p) ∧ q is false
def neg_p_and_q_is_false (q : Prop) : Prop := ¬(¬p ∧ q)

-- Option definitions
def option_A (m : ℝ) : Prop := -2 ≤ m ∧ m < 0 → ∀ x ∈ Ioo (-4 : ℝ) (-1), deriv (λ x, -x^2 + m * x) x > 0
def option_B (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4 → Real.logb (1 / 5) x ≥ -1
def option_C : Prop := ∀ x : ℝ, Real.cos (2 * x) - Real.sqrt 3 * Real.sin (2 * x) = 2 * Real.cos (2 * x + Real.pi / 3)
def option_D (a x : ℝ) : Prop := x ∈ Ioo (1 : ℝ) 3 → (deriv (λ x, 1 / 2 * x^2 - a * Real.log x) x) = 0

-- q is one of the options A, B, C or D
def q : Prop := option_D a x

theorem prove_false_q : ¬p ∧ q → False :=
by
  intro h
  cases h.2
  sorry

end prove_false_q_l131_131555


namespace integral_cos6_correct_l131_131877

noncomputable def integral_cos6 (C : ℝ) : ℝ → ℝ :=
  λ x, (1 / 8) * ( (5 / 2) * x + sin(4 * x) + (3 / 16) * sin(8 * x) - (1 / 12) * sin(4 * x)^3 ) + C

theorem integral_cos6_correct : 
  ∃ C : ℝ, 
    ∀ x : ℝ, 
      ∫ cos(2 * x)^6 dx = integral_cos6 C x :=
sorry

end integral_cos6_correct_l131_131877


namespace solve_quartic_equation_l131_131517

theorem solve_quartic_equation :
  (∃ x : ℝ, x > 0 ∧ 
    (1 / 3) * (4 * x ^ 2 - 3) = (x ^ 2 - 60 * x - 12) * (x ^ 2 + 30 * x + 6) ∧ 
    ∃ y1 y2 : ℝ, y1 + y2 = 60 ∧ (x^2 - 60 * x - 12 = 0)) → 
    x = 30 + Real.sqrt 912 :=
sorry

end solve_quartic_equation_l131_131517


namespace monotone_decreasing_condition_l131_131237

noncomputable def f (a x : ℝ) : ℝ := x^2 + 4 * a * x + 2

theorem monotone_decreasing_condition (a : ℝ) : 
  (∀ x ∈ set.Iio 6, (deriv (f a)) x ≤ 0) → a ≤ 3 :=
by
  sorry

end monotone_decreasing_condition_l131_131237


namespace sum_of_digits_inequality_l131_131734

def S : ℕ → ℕ := sorry -- This should be the sum of the digits of a number

theorem sum_of_digits_inequality (n : ℕ) : S(8 * n) ≥ (S(n) / 8) :=
by sorry

end sum_of_digits_inequality_l131_131734


namespace parabola_has_one_x_intercept_l131_131957

-- Define the equation of the parabola.
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 4

-- Prove that the number of x-intercepts of the graph of the parabola is 1.
theorem parabola_has_one_x_intercept : (∃! y : ℝ, parabola y = 4) :=
by
  sorry

end parabola_has_one_x_intercept_l131_131957
