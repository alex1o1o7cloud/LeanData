import Mathlib

namespace find_N_l2412_241270

noncomputable def N : ℕ := 1156

-- Condition 1: N is a perfect square
axiom N_perfect_square : ∃ n : ℕ, N = n^2

-- Condition 2: All digits of N are less than 7
axiom N_digits_less_than_7 : ∀ d, d ∈ [1, 1, 5, 6] → d < 7

-- Condition 3: Adding 3 to each digit yields another perfect square
axiom N_plus_3_perfect_square : ∃ m : ℕ, (m^2 = 1156 + 3333)

theorem find_N : N = 1156 :=
by
  -- Proof goes here
  sorry

end find_N_l2412_241270


namespace max_abs_sum_eq_two_l2412_241274

theorem max_abs_sum_eq_two (x y : ℝ) (h : x^2 + y^2 = 2) : |x| + |y| ≤ 2 :=
by
  sorry

end max_abs_sum_eq_two_l2412_241274


namespace inequality_solution_l2412_241299

theorem inequality_solution (x : ℝ) :
  (∀ y : ℝ, (0 < y) → (4 * (x^2 * y^2 + x * y^3 + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y + y)) ↔ (1 < x) :=
by
  sorry

end inequality_solution_l2412_241299


namespace gigi_remaining_batches_l2412_241266

variable (f b1 tf remaining_batches : ℕ)
variable (f_pos : 0 < f)
variable (batches_nonneg : 0 ≤ b1)
variable (t_f_pos : 0 < tf)
variable (h_f : f = 2)
variable (h_b1 : b1 = 3)
variable (h_tf : tf = 20)

theorem gigi_remaining_batches (h : remaining_batches = (tf - (f * b1)) / f) : remaining_batches = 7 := by
  sorry

end gigi_remaining_batches_l2412_241266


namespace inscribed_quadrilateral_circle_eq_radius_l2412_241252

noncomputable def inscribed_circle_condition (AB CD AD BC : ℝ) : Prop :=
  AB + CD = AD + BC

noncomputable def equal_radius_condition (r₁ r₂ r₃ r₄ : ℝ) : Prop :=
  r₁ = r₃ ∨ r₄ = r₂

theorem inscribed_quadrilateral_circle_eq_radius 
  (AB CD AD BC r₁ r₂ r₃ r₄ : ℝ)
  (h_inscribed_circle: inscribed_circle_condition AB CD AD BC)
  (h_four_circles: ∀ i, (i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4) → ∃ (r : ℝ), r = rᵢ): 
  equal_radius_condition r₁ r₂ r₃ r₄ :=
by {
  sorry
}

end inscribed_quadrilateral_circle_eq_radius_l2412_241252


namespace leg_ratio_of_right_triangle_l2412_241236

theorem leg_ratio_of_right_triangle (a b c m : ℝ) (h1 : a ≤ b)
  (h2 : a * b = c * m) (h3 : c^2 = a^2 + b^2) (h4 : a^2 + m^2 = b^2) :
  (a / b) = Real.sqrt ((-1 + Real.sqrt 5) / 2) :=
sorry

end leg_ratio_of_right_triangle_l2412_241236


namespace probability_of_hitting_target_at_least_once_l2412_241204

-- Define the constant probability of hitting the target in a single shot
def p_hit : ℚ := 2 / 3

-- Define the probability of missing the target in a single shot
def p_miss := 1 - p_hit

-- Define the probability of missing the target in all 3 shots
def p_miss_all_3 := p_miss ^ 3

-- Define the probability of hitting the target at least once in 3 shots
def p_hit_at_least_once := 1 - p_miss_all_3

-- Provide the theorem stating the solution
theorem probability_of_hitting_target_at_least_once :
  p_hit_at_least_once = 26 / 27 :=
by
  -- sorry is used to indicate the theorem needs to be proved
  sorry

end probability_of_hitting_target_at_least_once_l2412_241204


namespace abs_ineq_solution_l2412_241283

theorem abs_ineq_solution (x : ℝ) :
  (|x - 2| + |x + 1| < 4) ↔ (x ∈ Set.Ioo (-7 / 2) (-1) ∪ Set.Ico (-1) (5 / 2)) := by
  sorry

end abs_ineq_solution_l2412_241283


namespace find_inequality_solution_l2412_241264

theorem find_inequality_solution :
  {x : ℝ | (x + 1) / (x - 2) + (x + 3) / (2 * x + 1) ≤ 2}
  = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 1 ∨ 2 ≤ x ∧ x ≤ 9} :=
by
  -- The proof steps are omitted.
  sorry

end find_inequality_solution_l2412_241264


namespace parabola_focus_l2412_241207

noncomputable def parabola_focus_coordinates (a : ℝ) : ℝ × ℝ :=
  if a ≠ 0 then (0, 1 / (4 * a)) else (0, 0)

theorem parabola_focus {x y : ℝ} (a : ℝ) (h : a = 2) (h_eq : y = a * x^2) :
  parabola_focus_coordinates a = (0, 1 / 8) :=
by sorry

end parabola_focus_l2412_241207


namespace negation_of_p_l2412_241248

-- Defining the proposition 'p'
def p : Prop := ∃ x : ℝ, x^3 > x

-- Stating the theorem
theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x^3 ≤ x :=
by
  sorry

end negation_of_p_l2412_241248


namespace vector_parallel_and_on_line_l2412_241250

noncomputable def is_point_on_line (x y t : ℝ) : Prop :=
  x = 5 * t + 3 ∧ y = 2 * t + 4

noncomputable def is_parallel (a b c d : ℝ) : Prop :=
  ∃ k : ℝ, a = k * c ∧ b = k * d

theorem vector_parallel_and_on_line :
  ∃ (a b t : ℝ), 
      (a = (5 * t + 3) - 1) ∧ (b = (2 * t + 4) - 1) ∧ 
      is_parallel a b 3 2 ∧ is_point_on_line (5 * t + 3) (2 * t + 4) t := 
by
  use (33 / 4), (11 / 2), (5 / 4)
  sorry

end vector_parallel_and_on_line_l2412_241250


namespace squares_with_equal_black_and_white_cells_l2412_241278

open Nat

/-- Given a specific coloring of cells in a 5x5 grid, prove that there are
exactly 16 squares that have an equal number of black and white cells. --/
theorem squares_with_equal_black_and_white_cells :
  let gridSize := 5
  let number_of_squares_with_equal_black_and_white_cells := 16
  true := sorry

end squares_with_equal_black_and_white_cells_l2412_241278


namespace sum_in_base_b_l2412_241261

-- Definitions needed to articulate the problem
def base_b_value (n : ℕ) (b : ℕ) : ℕ :=
  match n with
  | 12 => b + 2
  | 15 => b + 5
  | 16 => b + 6
  | 3146 => 3 * b^3 + 1 * b^2 + 4 * b + 6
  | _  => 0

def s_in_base_b (b : ℕ) : ℕ :=
  base_b_value 12 b + base_b_value 15 b + base_b_value 16 b

theorem sum_in_base_b (b : ℕ) (h : (base_b_value 12 b) * (base_b_value 15 b) * (base_b_value 16 b) = base_b_value 3146 b) :
  s_in_base_b b = 44 := by
  sorry

end sum_in_base_b_l2412_241261


namespace function_properties_l2412_241219

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (2^(x + 1))

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end function_properties_l2412_241219


namespace longest_side_length_of_quadrilateral_l2412_241218

-- Define the system of inequalities
def inFeasibleRegion (x y : ℝ) : Prop :=
  (x + 2 * y ≤ 4) ∧
  (3 * x + y ≥ 3) ∧
  (x ≥ 0) ∧
  (y ≥ 0)

-- The goal is to prove that the longest side length is 5
theorem longest_side_length_of_quadrilateral :
  ∃ a b c d : (ℝ × ℝ), inFeasibleRegion a.1 a.2 ∧
                  inFeasibleRegion b.1 b.2 ∧
                  inFeasibleRegion c.1 c.2 ∧
                  inFeasibleRegion d.1 d.2 ∧
                  -- For each side, specify the length condition (Euclidean distance)
                  max (dist a b) (max (dist b c) (max (dist c d) (dist d a))) = 5 :=
by sorry

end longest_side_length_of_quadrilateral_l2412_241218


namespace segments_count_l2412_241269

/--
Given two concentric circles, with chords of the larger circle that are tangent to the smaller circle,
if each chord subtends an angle of 80 degrees at the center, then the number of such segments 
drawn before returning to the starting point is 18.
-/
theorem segments_count (angle_ABC : ℝ) (circumference_angle_sum : ℝ → ℝ) (n m : ℕ) :
  angle_ABC = 80 → 
  circumference_angle_sum angle_ABC = 360 → 
  100 * n = 360 * m → 
  5 * n = 18 * m →
  n = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end segments_count_l2412_241269


namespace total_number_of_coins_l2412_241275

variable (nickels dimes total_value : ℝ)
variable (total_nickels : ℕ)

def value_of_nickel : ℝ := 0.05
def value_of_dime : ℝ := 0.10

theorem total_number_of_coins :
  total_value = 3.50 → total_nickels = 30 → total_value = total_nickels * value_of_nickel + dimes * value_of_dime → 
  total_nickels + dimes = 50 :=
by
  intros h_total_value h_total_nickels h_value_equation
  sorry

end total_number_of_coins_l2412_241275


namespace probability_of_both_types_probability_distribution_and_expectation_of_X_l2412_241255

-- Definitions
def total_zongzi : ℕ := 8
def red_bean_paste_zongzi : ℕ := 2
def date_zongzi : ℕ := 6
def selected_zongzi : ℕ := 3

-- Part 1: The probability of selecting both red bean paste and date zongzi
theorem probability_of_both_types :
  let total_combinations := Nat.choose total_zongzi selected_zongzi
  let one_red_two_date := Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2
  let two_red_one_date := Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1
  (one_red_two_date + two_red_one_date) / total_combinations = 9 / 14 :=
by sorry

-- Part 2: The probability distribution and expectation of X
theorem probability_distribution_and_expectation_of_X :
  let P_X_0 := (Nat.choose red_bean_paste_zongzi 0 * Nat.choose date_zongzi 3) / Nat.choose total_zongzi selected_zongzi
  let P_X_1 := (Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2) / Nat.choose total_zongzi selected_zongzi
  let P_X_2 := (Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1) / Nat.choose total_zongzi selected_zongzi
  P_X_0 = 5 / 14 ∧ P_X_1 = 15 / 28 ∧ P_X_2 = 3 / 28 ∧
  (0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 = 3 / 4) :=
by sorry

end probability_of_both_types_probability_distribution_and_expectation_of_X_l2412_241255


namespace largest_convex_ngon_with_integer_tangents_l2412_241229

-- Definitions of conditions and the statement
def isConvex (n : ℕ) : Prop := n ≥ 3 -- Condition 1: n is at least 3
def isConvexPolygon (n : ℕ) : Prop := isConvex n -- Condition 2: the polygon is convex
def tanInteriorAnglesAreIntegers (n : ℕ) : Prop := true -- Placeholder for Condition 3

-- Statement to prove
theorem largest_convex_ngon_with_integer_tangents : 
  ∀ n : ℕ, isConvexPolygon n → tanInteriorAnglesAreIntegers n → n ≤ 8 :=
by
  intros n h_convex h_tangents
  sorry

end largest_convex_ngon_with_integer_tangents_l2412_241229


namespace false_converse_of_vertical_angles_l2412_241291

theorem false_converse_of_vertical_angles
  (P : Prop) (Q : Prop) (V : ∀ {A B C D : Type}, (A = B ∧ C = D) → P) (C1 : P → Q) :
  ¬ (Q → P) :=
sorry

end false_converse_of_vertical_angles_l2412_241291


namespace intersection_point_l2412_241247

def parametric_line (t : ℝ) : ℝ × ℝ × ℝ :=
  (-1 - 2 * t, 0, -1 + 3 * t)

def plane (x y z : ℝ) : Prop := x + 4 * y + 13 * z - 23 = 0

theorem intersection_point :
  ∃ t : ℝ, plane (-1 - 2 * t) 0 (-1 + 3 * t) ∧ parametric_line t = (-3, 0, 2) :=
by
  sorry

end intersection_point_l2412_241247


namespace sum_series_evaluation_l2412_241257

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (if k = 0 then 0 else (2 * k) / (4 : ℝ) ^ k)

theorem sum_series_evaluation : sum_series = 8 / 9 := by
  sorry

end sum_series_evaluation_l2412_241257


namespace max_gold_coins_l2412_241244

theorem max_gold_coins (k : ℤ) (h1 : ∃ k : ℤ, 15 * k + 3 < 120) : 
  ∃ n : ℤ, n = 15 * k + 3 ∧ n < 120 ∧ n = 108 :=
by
  sorry

end max_gold_coins_l2412_241244


namespace cost_price_computer_table_l2412_241292

theorem cost_price_computer_table (C S : ℝ) (hS1 : S = 1.25 * C) (hS2 : S = 1000) : C = 800 :=
by
  sorry

end cost_price_computer_table_l2412_241292


namespace searchlight_revolutions_l2412_241263

theorem searchlight_revolutions (p : ℝ) (r : ℝ) (t : ℝ) 
  (h1 : p = 0.6666666666666667) 
  (h2 : t = 10) 
  (h3 : p = (60 / r - t) / (60 / r)) : 
  r = 2 :=
by sorry

end searchlight_revolutions_l2412_241263


namespace sum_of_smallest_and_largest_prime_l2412_241276

def primes_between (a b : ℕ) : List ℕ := List.filter Nat.Prime (List.range' a (b - a + 1))

def smallest_prime_in_range (a b : ℕ) : ℕ :=
  match primes_between a b with
  | [] => 0
  | h::t => h

def largest_prime_in_range (a b : ℕ) : ℕ :=
  match List.reverse (primes_between a b) with
  | [] => 0
  | h::t => h

theorem sum_of_smallest_and_largest_prime : smallest_prime_in_range 1 50 + largest_prime_in_range 1 50 = 49 := 
by
  -- Let the Lean prover take over from here
  sorry

end sum_of_smallest_and_largest_prime_l2412_241276


namespace max_min_sum_difference_l2412_241281

-- The statement that we need to prove
theorem max_min_sum_difference : 
  ∃ (max_sum min_sum: ℕ), (∀ (RST UVW XYZ : ℕ),
   -- Constraints for Max's and Minnie's sums respectively
   (RST = 100 * 9 + 10 * 6 + 3 ∧ UVW = 100 * 8 + 10 * 5 + 2 ∧ XYZ = 100 * 7 + 10 * 4 + 1 → max_sum = 2556) ∧ 
   (RST = 100 * 1 + 10 * 0 + 6 ∧ UVW = 100 * 2 + 10 * 4 + 7 ∧ XYZ = 100 * 3 + 10 * 5 + 8 → min_sum = 711)) → 
    max_sum - min_sum = 1845 :=
by
  sorry

end max_min_sum_difference_l2412_241281


namespace pet_store_cats_left_l2412_241215

theorem pet_store_cats_left :
  let initial_siamese := 13.5
  let initial_house := 5.25
  let added_cats := 10.75
  let discount := 0.5
  let initial_total := initial_siamese + initial_house
  let new_total := initial_total + added_cats
  let final_total := new_total - discount
  final_total = 29 :=
by sorry

end pet_store_cats_left_l2412_241215


namespace shane_gum_left_l2412_241210

def elyse_initial_gum : ℕ := 100
def half (x : ℕ) := x / 2
def rick_gum : ℕ := half elyse_initial_gum
def shane_initial_gum : ℕ := half rick_gum
def chewed_gum : ℕ := 11

theorem shane_gum_left : shane_initial_gum - chewed_gum = 14 := by
  sorry

end shane_gum_left_l2412_241210


namespace intersection_M_N_l2412_241237

def M : Set ℝ := { x | x > 1 }
def N : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem intersection_M_N :
  M ∩ N = { x | 1 < x ∧ x ≤ 2 } := 
sorry

end intersection_M_N_l2412_241237


namespace one_eq_a_l2412_241296

theorem one_eq_a (x y z a : ℝ) (h₁: x + y + z = a) (h₂: 1/x + 1/y + 1/z = 1/a) :
  x = a ∨ y = a ∨ z = a :=
  sorry

end one_eq_a_l2412_241296


namespace solve_equation_l2412_241284

theorem solve_equation : ∃ x : ℤ, (x - 15) / 3 = (3 * x + 11) / 8 ∧ x = -153 := 
by
  use -153
  sorry

end solve_equation_l2412_241284


namespace tetris_blocks_form_square_l2412_241297

-- Definitions of Tetris blocks types
inductive TetrisBlock
| A | B | C | D | E | F | G

open TetrisBlock

-- Definition of a block's ability to form a square
def canFormSquare (block: TetrisBlock) : Prop :=
  block = A ∨ block = B ∨ block = C ∨ block = D ∨ block = G

-- The main theorem statement
theorem tetris_blocks_form_square : ∀ (block : TetrisBlock), canFormSquare block → block = A ∨ block = B ∨ block = C ∨ block = D ∨ block = G := 
by
  intros block h
  exact h

end tetris_blocks_form_square_l2412_241297


namespace base_number_in_exponent_l2412_241221

theorem base_number_in_exponent (x : ℝ) (k : ℕ) (h₁ : k = 8) (h₂ : 64^k > x^22) : 
  x = 2^(24/11) :=
sorry

end base_number_in_exponent_l2412_241221


namespace find_person_age_l2412_241206

theorem find_person_age : ∃ x : ℕ, 4 * (x + 4) - 4 * (x - 4) = x ∧ x = 32 := by
  sorry

end find_person_age_l2412_241206


namespace division_value_l2412_241285

theorem division_value (n x : ℝ) (h₀ : n = 4.5) (h₁ : (n / x) * 12 = 9) : x = 6 :=
by
  sorry

end division_value_l2412_241285


namespace mohan_least_cookies_l2412_241282

theorem mohan_least_cookies :
  ∃ b : ℕ, 
    b % 6 = 5 ∧
    b % 8 = 3 ∧
    b % 9 = 6 ∧
    b = 59 :=
by
  sorry

end mohan_least_cookies_l2412_241282


namespace volume_of_given_tetrahedron_l2412_241267

noncomputable def volume_of_tetrahedron (radius : ℝ) (total_length : ℝ) : ℝ := 
  let R := radius
  let L := total_length
  let a := (2 * Real.sqrt 33) / 3
  let V := (a^3 * Real.sqrt 2) / 12
  V

theorem volume_of_given_tetrahedron :
  volume_of_tetrahedron (Real.sqrt 22 / 2) (8 * Real.pi) = 48 := 
  sorry

end volume_of_given_tetrahedron_l2412_241267


namespace tournament_trio_l2412_241272

theorem tournament_trio
  (n : ℕ)
  (h_n : n ≥ 3)
  (match_result : Fin n → Fin n → Prop)
  (h1 : ∀ i j : Fin n, i ≠ j → (match_result i j ∨ match_result j i))
  (h2 : ∀ i : Fin n, ∃ j : Fin n, match_result i j)
:
  ∃ (A B C : Fin n), match_result A B ∧ match_result B C ∧ match_result C A :=
by
  sorry

end tournament_trio_l2412_241272


namespace unique_triangle_exists_l2412_241228

theorem unique_triangle_exists : 
  (¬ (∀ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = 3 → a + b > c)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 30 → ∃ (C : ℝ), C > 0)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 100 → ∃ (C : ℝ), C > 0)) ∧
  (∀ (b c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45 → ∃! (a c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45) :=
by sorry

end unique_triangle_exists_l2412_241228


namespace positive_difference_of_two_numbers_l2412_241262

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l2412_241262


namespace hair_growth_l2412_241259

-- Define the length of Isabella's hair initially and the growth
def initial_length : ℕ := 18
def growth : ℕ := 4

-- Define the final length of the hair after growth
def final_length (initial_length : ℕ) (growth : ℕ) : ℕ := initial_length + growth

-- State the theorem that the final length is 22 inches
theorem hair_growth : final_length initial_length growth = 22 := 
by
  sorry

end hair_growth_l2412_241259


namespace spherical_to_rectangular_conversion_l2412_241265

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_conversion :
  spherical_to_rectangular 4 (Real.pi / 2) (Real.pi / 4) = (0, 2 * Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_conversion_l2412_241265


namespace unique_triangled_pair_l2412_241232

theorem unique_triangled_pair (a b x y : ℝ) (h : ∀ a b : ℝ, (a, b) = (a * x + b * y, a * y + b * x)) : (x, y) = (1, 0) :=
by sorry

end unique_triangled_pair_l2412_241232


namespace number_of_paths_to_spell_BINGO_l2412_241249

theorem number_of_paths_to_spell_BINGO : 
  ∃ (paths : ℕ), paths = 36 :=
by
  sorry

end number_of_paths_to_spell_BINGO_l2412_241249


namespace evaluate_expression_l2412_241217

theorem evaluate_expression : (3 / (1 - (2 / 5))) = 5 := by
  sorry

end evaluate_expression_l2412_241217


namespace proposition_true_l2412_241234

theorem proposition_true (a b : ℝ) (h1 : 0 > a) (h2 : a > b) : (1/a) < (1/b) := 
sorry

end proposition_true_l2412_241234


namespace polyhedron_space_diagonals_l2412_241288

theorem polyhedron_space_diagonals (V E F T Q P : ℕ) (hV : V = 30) (hE : E = 70) (hF : F = 42)
                                    (hT : T = 26) (hQ : Q = 12) (hP : P = 4) : 
  ∃ D : ℕ, D = 321 :=
by
  have total_pairs := (30 * 29) / 2
  have triangular_face_diagonals := 0
  have quadrilateral_face_diagonals := 12 * 2
  have pentagon_face_diagonals := 4 * 5
  have total_face_diagonals := triangular_face_diagonals + quadrilateral_face_diagonals + pentagon_face_diagonals
  have total_edges_and_diagonals := total_pairs - 70 - total_face_diagonals
  use total_edges_and_diagonals
  sorry

end polyhedron_space_diagonals_l2412_241288


namespace unique_positive_b_for_one_solution_l2412_241227

theorem unique_positive_b_for_one_solution
  (a : ℝ) (c : ℝ) :
  a = 3 →
  (∃! (b : ℝ), b > 0 ∧ (3 * (b + (1 / b)))^2 - 4 * c = 0 ) →
  c = 9 :=
by
  intros ha h
  -- Proceed to show that c must be 9
  sorry

end unique_positive_b_for_one_solution_l2412_241227


namespace part1_simplification_part2_inequality_l2412_241200

-- Part 1: Prove the simplification of the algebraic expression
theorem part1_simplification (x : ℝ) (h₁ : x ≠ 3):
  (2 * x + 4) / (x^2 - 6 * x + 9) / ((2 * x - 1) / (x - 3) - 1) = 2 / (x - 3) :=
sorry

-- Part 2: Prove the solution set for the inequality system
theorem part2_inequality (x : ℝ) :
  (5 * x - 2 > 3 * (x + 1)) → (1/2 * x - 1 ≥ 7 - 3/2 * x) → x ≥ 4 :=
sorry

end part1_simplification_part2_inequality_l2412_241200


namespace digit_Q_is_0_l2412_241251

theorem digit_Q_is_0 (M N P Q : ℕ) (hM : M < 10) (hN : N < 10) (hP : P < 10) (hQ : Q < 10) 
  (add_eq : 10 * M + N + 10 * P + M = 10 * Q + N) 
  (sub_eq : 10 * M + N - (10 * P + M) = N) : Q = 0 := 
by
  sorry

end digit_Q_is_0_l2412_241251


namespace hotel_accommodation_l2412_241287

theorem hotel_accommodation :
  ∃ (arrangements : ℕ), arrangements = 27 :=
by
  -- problem statement
  let triple_room := 1
  let double_room := 1
  let single_room := 1
  let adults := 3
  let children := 2
  
  -- use the given conditions and properties of combinations to calculate arrangements
  sorry

end hotel_accommodation_l2412_241287


namespace total_annual_salary_excluding_turban_l2412_241298

-- Let X be the total amount of money Gopi gives as salary for one year, excluding the turban.
variable (X : ℝ)

-- Condition: The servant leaves after 9 months and receives Rs. 60 plus the turban.
variable (received_money : ℝ)
variable (turban_price : ℝ)

-- Condition values:
axiom received_money_condition : received_money = 60
axiom turban_price_condition : turban_price = 30

-- Question: Prove that X equals 90.
theorem total_annual_salary_excluding_turban :
  3/4 * (X + turban_price) = 90 :=
sorry

end total_annual_salary_excluding_turban_l2412_241298


namespace probability_of_lamps_arrangement_l2412_241294

noncomputable def probability_lava_lamps : ℚ :=
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_turn_on := 4
  let left_red_on := 1
  let right_blue_off := 1
  let ways_to_choose_positions := Nat.choose total_lamps red_lamps
  let ways_to_choose_turn_on := Nat.choose total_lamps total_turn_on
  let remaining_positions := total_lamps - left_red_on - right_blue_off
  let remaining_red_lamps := red_lamps - left_red_on
  let remaining_turn_on := total_turn_on - left_red_on
  let arrangements_of_remaining_red := Nat.choose remaining_positions remaining_red_lamps
  let arrangements_of_turn_on :=
    Nat.choose (remaining_positions - right_blue_off) remaining_turn_on
  -- The probability calculation
  (arrangements_of_remaining_red * arrangements_of_turn_on : ℚ) / 
    (ways_to_choose_positions * ways_to_choose_turn_on)

theorem probability_of_lamps_arrangement :
    probability_lava_lamps = 4 / 49 :=
by
  sorry

end probability_of_lamps_arrangement_l2412_241294


namespace min_value_seq_ratio_l2412_241208

-- Define the sequence {a_n} based on the given recurrence relation and initial condition
def seq (n : ℕ) : ℕ := 
  if n = 0 then 0 -- Handling the case when n is 0, though sequence starts from n=1
  else n^2 - n + 15

-- Prove the minimum value of (a_n / n) is 27/4
theorem min_value_seq_ratio : 
  ∃ n : ℕ, n > 0 ∧ seq n / n = 27 / 4 :=
by
  sorry

end min_value_seq_ratio_l2412_241208


namespace total_toes_on_bus_l2412_241293

-- Definitions based on conditions
def toes_per_hand_hoopit : Nat := 3
def hands_per_hoopit : Nat := 4
def number_of_hoopits_on_bus : Nat := 7

def toes_per_hand_neglart : Nat := 2
def hands_per_neglart : Nat := 5
def number_of_neglarts_on_bus : Nat := 8

-- We need to prove that the total number of toes on the bus is 164
theorem total_toes_on_bus :
  (toes_per_hand_hoopit * hands_per_hoopit * number_of_hoopits_on_bus) +
  (toes_per_hand_neglart * hands_per_neglart * number_of_neglarts_on_bus) = 164 :=
by sorry

end total_toes_on_bus_l2412_241293


namespace shirts_per_minute_l2412_241226

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (shirts_per_min : ℕ) 
  (h : total_shirts = 12 ∧ total_minutes = 6) :
  shirts_per_min = 2 :=
sorry

end shirts_per_minute_l2412_241226


namespace positive_perfect_squares_multiples_of_36_lt_10_pow_8_l2412_241256

theorem positive_perfect_squares_multiples_of_36_lt_10_pow_8 :
  ∃ (count : ℕ), count = 1666 ∧ 
    (∀ n : ℕ, (n * n < 10^8 → n % 6 = 0) ↔ (n < 10^4 ∧ n % 6 = 0)) :=
sorry

end positive_perfect_squares_multiples_of_36_lt_10_pow_8_l2412_241256


namespace carSpeedIs52mpg_l2412_241202

noncomputable def carSpeed (fuelConsumptionKMPL : ℕ) -- 32 kilometers per liter
                           (gallonToLiter : ℝ)        -- 1 gallon = 3.8 liters
                           (fuelDecreaseGallons : ℝ)  -- 3.9 gallons
                           (timeHours : ℝ)            -- 5.7 hours
                           (kmToMiles : ℝ)            -- 1 mile = 1.6 kilometers
                           : ℝ :=
  let totalLiters := fuelDecreaseGallons * gallonToLiter
  let totalKilometers := totalLiters * fuelConsumptionKMPL
  let totalMiles := totalKilometers / kmToMiles
  totalMiles / timeHours

theorem carSpeedIs52mpg : carSpeed 32 3.8 3.9 5.7 1.6 = 52 := sorry

end carSpeedIs52mpg_l2412_241202


namespace factorize_expression_l2412_241213

variable (x y : ℝ)

theorem factorize_expression : xy^2 + 6*xy + 9*x = x*(y + 3)^2 := by
  sorry

end factorize_expression_l2412_241213


namespace rectangle_circle_area_ratio_l2412_241277

theorem rectangle_circle_area_ratio (w r : ℝ) (h1 : 2 * 2 * w + 2 * w = 2 * pi * r) :
  ((2 * w) * w) / (pi * r^2) = 2 * pi / 9 :=
by
  sorry

end rectangle_circle_area_ratio_l2412_241277


namespace chess_tournament_rounds_needed_l2412_241290

theorem chess_tournament_rounds_needed
  (num_players : ℕ)
  (num_games_per_round : ℕ)
  (H1 : num_players = 20)
  (H2 : num_games_per_round = 10) :
  (num_players * (num_players - 1)) / num_games_per_round = 38 :=
by
  sorry

end chess_tournament_rounds_needed_l2412_241290


namespace roots_equality_l2412_241205

noncomputable def problem_statement (α β γ δ p q : ℝ) : Prop :=
(α - γ) * (β - δ) * (α + δ) * (β + γ) = 4 * (2 * p - 3 * q) ^ 2

theorem roots_equality (α β γ δ p q : ℝ)
  (h₁ : ∀ x, x^2 - 2 * p * x + 3 = 0 → (x = α ∨ x = β))
  (h₂ : ∀ x, x^2 - 3 * q * x + 4 = 0 → (x = γ ∨ x = δ)) :
  problem_statement α β γ δ p q :=
sorry

end roots_equality_l2412_241205


namespace find_k_l2412_241239

theorem find_k (x₁ x₂ k : ℝ) (h1 : x₁ * x₁ - 6 * x₁ + k = 0) (h2 : x₂ * x₂ - 6 * x₂ + k = 0) (h3 : (1 / x₁) + (1 / x₂) = 3) :
  k = 2 :=
by
  sorry

end find_k_l2412_241239


namespace average_cookies_l2412_241289

theorem average_cookies (cookie_counts : List ℕ) (h : cookie_counts = [8, 10, 12, 15, 16, 17, 20]) :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 14 := by
    -- Proof goes here
  sorry

end average_cookies_l2412_241289


namespace real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l2412_241212

def z (m : ℝ) : ℂ := (m^2 - 5 * m + 6 : ℝ) + (m^2 - 3 * m : ℝ) * Complex.I

theorem real_part_0_or_3 (m : ℝ) : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) := sorry

theorem complex_part_not_0_or_3 (m : ℝ) : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) := sorry

theorem purely_imaginary_at_2 (m : ℝ) : (m^2 - 5 * m + 6 = 0) ∧ (m^2 - 3 * m ≠ 0) ↔ (m = 2) := sorry

theorem no_second_quadrant (m : ℝ) : ¬(m^2 - 5 * m + 6 < 0 ∧ m^2 - 3 * m > 0) := sorry

end real_part_0_or_3_complex_part_not_0_or_3_purely_imaginary_at_2_no_second_quadrant_l2412_241212


namespace students_accounting_majors_l2412_241230

theorem students_accounting_majors (p q r s : ℕ) 
  (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) (h5 : p * q * r * s = 1365) : p = 3 := 
by 
  sorry

end students_accounting_majors_l2412_241230


namespace trajectory_eq_l2412_241231

theorem trajectory_eq (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = 6 → x^2 + y^2 + 2 * x + 2 * y - 3 = 0 → 
    ∃ p q : ℝ, p = a + 1 ∧ q = b + 1 ∧ (p * x + q * y = (a^2 + b^2 - 3)/2)) →
  a^2 + b^2 + 2 * a + 2 * b + 1 = 0 :=
by
  intros h
  sorry

end trajectory_eq_l2412_241231


namespace tangent_intersects_x_axis_l2412_241240

theorem tangent_intersects_x_axis (x0 x1 : ℝ) (hx : ∀ x : ℝ, x1 = x0 - 1) :
  x1 - x0 = -1 :=
by
  sorry

end tangent_intersects_x_axis_l2412_241240


namespace number_of_purchasing_schemes_l2412_241258

def total_cost (a : Nat) (b : Nat) : Nat := 8 * a + 10 * b

def valid_schemes : List (Nat × Nat) :=
  [(4, 4), (4, 5), (4, 6), (4, 7),
   (5, 4), (5, 5), (5, 6),
   (6, 4), (6, 5),
   (7, 4)]

theorem number_of_purchasing_schemes : valid_schemes.length = 9 := sorry

end number_of_purchasing_schemes_l2412_241258


namespace total_cost_of_trick_decks_l2412_241254

theorem total_cost_of_trick_decks (cost_per_deck: ℕ) (victor_decks: ℕ) (friend_decks: ℕ) (total_spent: ℕ) : 
  cost_per_deck = 8 → victor_decks = 6 → friend_decks = 2 → total_spent = cost_per_deck * victor_decks + cost_per_deck * friend_decks → total_spent = 64 :=
by 
  sorry

end total_cost_of_trick_decks_l2412_241254


namespace quadratic_function_value_at_18_l2412_241220

noncomputable def p (d e f x : ℝ) : ℝ := d*x^2 + e*x + f

theorem quadratic_function_value_at_18
  (d e f : ℝ)
  (h_sym : ∀ x1 x2 : ℝ, p d e f 6 = p d e f 12)
  (h_max : ∀ x : ℝ, x = 10 → ∃ p_max : ℝ, ∀ y : ℝ, p d e f x ≤ p_max)
  (h_p0 : p d e f 0 = -1) : 
  p d e f 18 = -1 := 
sorry

end quadratic_function_value_at_18_l2412_241220


namespace sqrt_multiplication_and_subtraction_l2412_241222

theorem sqrt_multiplication_and_subtraction :
  (Real.sqrt 21 * Real.sqrt 7 - Real.sqrt 3) = 6 * Real.sqrt 3 := 
by
  sorry

end sqrt_multiplication_and_subtraction_l2412_241222


namespace friend_spent_13_50_l2412_241295

noncomputable def amount_you_spent : ℝ := 
  let x := (22 - 5) / 2
  x

noncomputable def amount_friend_spent (x : ℝ) : ℝ := 
  x + 5

theorem friend_spent_13_50 :
  ∃ x : ℝ, (x + (x + 5) = 22) ∧ (x + 5 = 13.5) :=
by
  sorry

end friend_spent_13_50_l2412_241295


namespace even_function_f_l2412_241253

-- Problem statement:
-- Given that f is an even function and that for x < 0, f(x) = x^2 - 1/x,
-- prove that f(1) = 2.

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - 1/x else 0

theorem even_function_f {f : ℝ → ℝ} (h_even : ∀ x, f x = f (-x))
  (h_neg_def : ∀ x, x < 0 → f x = x^2 - 1/x) : f 1 = 2 :=
by
  -- Proof body (to be completed)
  sorry

end even_function_f_l2412_241253


namespace domain_of_reciprocal_shifted_function_l2412_241273

def domain_of_function (x : ℝ) : Prop :=
  x ≠ 1

theorem domain_of_reciprocal_shifted_function : 
  ∀ x : ℝ, (∃ y : ℝ, y = 1 / (x - 1)) ↔ domain_of_function x :=
by 
  sorry

end domain_of_reciprocal_shifted_function_l2412_241273


namespace remitted_amount_is_correct_l2412_241201

-- Define the constants and conditions of the problem
def total_sales : ℝ := 32500
def commission_rate1 : ℝ := 0.05
def commission_limit : ℝ := 10000
def commission_rate2 : ℝ := 0.04

-- Define the function to calculate the remitted amount
def remitted_amount (total_sales commission_rate1 commission_limit commission_rate2 : ℝ) : ℝ :=
  let commission1 := commission_rate1 * commission_limit
  let remaining_sales := total_sales - commission_limit
  let commission2 := commission_rate2 * remaining_sales
  total_sales - (commission1 + commission2)

-- Lean statement to prove the remitted amount
theorem remitted_amount_is_correct :
  remitted_amount total_sales commission_rate1 commission_limit commission_rate2 = 31100 :=
by
  sorry

end remitted_amount_is_correct_l2412_241201


namespace sams_speed_l2412_241214

theorem sams_speed (lucas_speed : ℝ) (maya_factor : ℝ) (relationship_factor : ℝ) 
  (h_lucas : lucas_speed = 5)
  (h_maya : maya_factor = 4 / 5)
  (h_relationship : relationship_factor = 9 / 8) :
  (5 / relationship_factor) = 40 / 9 :=
by
  sorry

end sams_speed_l2412_241214


namespace number_of_cats_l2412_241223

theorem number_of_cats 
  (n k : ℕ)
  (h1 : n * k = 999919)
  (h2 : k > n) :
  n = 991 :=
sorry

end number_of_cats_l2412_241223


namespace find_number_l2412_241245

-- Define the necessary variables and constants
variables (N : ℝ) (h1 : (5 / 4) * N = (4 / 5) * N + 18)

-- State the problem as a theorem to be proved
theorem find_number : N = 40 :=
by
  sorry

end find_number_l2412_241245


namespace find_tuition_l2412_241268

def tuition_problem (T : ℝ) : Prop :=
  75 = T + (T - 15)

theorem find_tuition (T : ℝ) (h : tuition_problem T) : T = 45 :=
by
  sorry

end find_tuition_l2412_241268


namespace customers_left_l2412_241216

theorem customers_left (x : ℕ) 
  (h1 : 47 - x + 20 = 26) : 
  x = 41 :=
sorry

end customers_left_l2412_241216


namespace find_A_l2412_241271

def clubsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

theorem find_A (A : ℝ) : clubsuit A 6 = 31 → A = 10.5 :=
by
  intro h
  sorry

end find_A_l2412_241271


namespace cost_per_page_first_time_l2412_241280

-- Definitions based on conditions
variables (num_pages : ℕ) (rev_once_pages : ℕ) (rev_twice_pages : ℕ)
variables (rev_cost : ℕ) (total_cost : ℕ)
variables (first_time_cost : ℕ)

-- Conditions
axiom h1 : num_pages = 100
axiom h2 : rev_once_pages = 35
axiom h3 : rev_twice_pages = 15
axiom h4 : rev_cost = 4
axiom h5 : total_cost = 860

-- Proof statement: Prove that the cost per page for the first time a page is typed is $6
theorem cost_per_page_first_time : first_time_cost = 6 :=
sorry

end cost_per_page_first_time_l2412_241280


namespace maria_travel_fraction_l2412_241279

theorem maria_travel_fraction (x : ℝ) (total_distance : ℝ)
  (h1 : ∀ d1 d2, d1 + d2 = total_distance)
  (h2 : total_distance = 360)
  (h3 : ∃ d1 d2 d3, d1 = 360 * x ∧ d2 = (1 / 4) * (360 - 360 * x) ∧ d3 = 135)
  (h4 : d1 + d2 + d3 = total_distance)
  : x = 1 / 2 :=
by
  sorry

end maria_travel_fraction_l2412_241279


namespace average_percentage_decrease_l2412_241233

-- Given definitions
def original_price : ℝ := 10000
def final_price : ℝ := 6400
def num_reductions : ℕ := 2

-- The goal is to prove the average percentage decrease per reduction
theorem average_percentage_decrease (x : ℝ) (h : (original_price * (1 - x)^num_reductions = final_price)) : x = 0.2 :=
sorry

end average_percentage_decrease_l2412_241233


namespace tom_total_payment_l2412_241225

theorem tom_total_payment :
  let apples_cost := 8 * 70
  let mangoes_cost := 9 * 55
  let oranges_cost := 5 * 40
  let bananas_cost := 12 * 30
  let grapes_cost := 7 * 45
  let cherries_cost := 4 * 80
  apples_cost + mangoes_cost + oranges_cost + bananas_cost + grapes_cost + cherries_cost = 2250 :=
by
  sorry

end tom_total_payment_l2412_241225


namespace textbook_weight_l2412_241211

theorem textbook_weight
  (w : ℝ)
  (bookcase_limit : ℝ := 80)
  (hardcover_books : ℕ := 70)
  (hardcover_weight_per_book : ℝ := 0.5)
  (textbooks : ℕ := 30)
  (knick_knacks : ℕ := 3)
  (knick_knack_weight : ℝ := 6)
  (over_limit : ℝ := 33)
  (total_items_weight : ℝ := bookcase_limit + over_limit)
  (hardcover_total_weight : ℝ := hardcover_books * hardcover_weight_per_book)
  (knick_knack_total_weight : ℝ := knick_knacks * knick_knack_weight)
  (remaining_weight : ℝ := total_items_weight - (hardcover_total_weight + knick_knack_total_weight)) :
  remaining_weight = textbooks * 2 :=
by
  sorry

end textbook_weight_l2412_241211


namespace product_of_roots_l2412_241238

-- Define the quadratic function in terms of a, b, c
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the conditions
variables (a b c y : ℝ)

-- Given conditions from the problem
def condition_1 := ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2
def condition_2 := quadratic a b c y = 0
def condition_3 := quadratic a b c (4 * y) = 0

-- The statement to be proved
theorem product_of_roots (a b c y : ℝ) 
  (h1: ∀ x, quadratic a b c x = 0 → ∃ x1 x2, x = x1 ∨ x = x2)
  (h2: quadratic a b c y = 0) 
  (h3: quadratic a b c (4 * y) = 0) :
  ∃ x1 x2, (quadratic a b c x = 0 → (x1 = y ∧ x2 = 4 * y) ∨ (x1 = 4 * y ∧ x2 = y)) ∧ x1 * x2 = 4 * y^2 :=
by
  sorry

end product_of_roots_l2412_241238


namespace tailor_buttons_l2412_241224

theorem tailor_buttons (G : ℕ) (yellow_buttons : ℕ) (blue_buttons : ℕ) 
(h1 : yellow_buttons = G + 10) (h2 : blue_buttons = G - 5) 
(h3 : G + yellow_buttons + blue_buttons = 275) : G = 90 :=
sorry

end tailor_buttons_l2412_241224


namespace determine_b_l2412_241260

theorem determine_b (b : ℝ) : (∀ x1 x2 : ℝ, x1^2 - x2^2 = 7 → x1 * x2 = 12 → x1 + x2 = b) → (b = 7 ∨ b = -7) := 
by {
  -- Proof needs to be provided
  sorry
}

end determine_b_l2412_241260


namespace eval_dollar_expr_l2412_241246

noncomputable def dollar (k : ℝ) (a b : ℝ) := k * (a - b) ^ 2

theorem eval_dollar_expr (x y : ℝ) : dollar 3 ((2 * x - 3 * y) ^ 2) ((3 * y - 2 * x) ^ 2) = 0 :=
by sorry

end eval_dollar_expr_l2412_241246


namespace complement_of_A_relative_to_U_l2412_241242

def U := { x : ℝ | x < 3 }
def A := { x : ℝ | x < 1 }

def complement_U_A := { x : ℝ | 1 ≤ x ∧ x < 3 }

theorem complement_of_A_relative_to_U : (complement_U_A = { x : ℝ | x ∈ U ∧ x ∉ A }) :=
by
  sorry

end complement_of_A_relative_to_U_l2412_241242


namespace strength_training_sessions_l2412_241203

-- Define the problem conditions
def strength_training_hours (x : ℕ) : ℝ := x * 1
def boxing_training_hours : ℝ := 4 * 1.5
def total_training_hours : ℝ := 9

-- Prove how many times a week does Kat do strength training
theorem strength_training_sessions : ∃ x : ℕ, strength_training_hours x + boxing_training_hours = total_training_hours ∧ x = 3 := 
by {
  sorry
}

end strength_training_sessions_l2412_241203


namespace no_such_integers_l2412_241235

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem no_such_integers (a b c d k : ℤ) (h : k > 1) :
  (a + b * omega + c * omega^2 + d * omega^3)^k ≠ 1 + omega :=
sorry

end no_such_integers_l2412_241235


namespace algebraic_expression_value_l2412_241286

theorem algebraic_expression_value (p q : ℝ) 
  (h : p * 3^3 + q * 3 + 1 = 2015) : 
  p * (-3)^3 + q * (-3) + 1 = -2013 :=
by 
  sorry

end algebraic_expression_value_l2412_241286


namespace side_error_percentage_l2412_241209

theorem side_error_percentage (S S' : ℝ) (h1: S' = S * Real.sqrt 1.0609) : 
  (S' / S - 1) * 100 = 3 :=
by
  sorry

end side_error_percentage_l2412_241209


namespace trisect_54_degree_angle_l2412_241243

theorem trisect_54_degree_angle :
  ∃ (a1 a2 : ℝ), a1 = 18 ∧ a2 = 36 ∧ a1 + a2 + a2 = 54 :=
by sorry

end trisect_54_degree_angle_l2412_241243


namespace total_mangoes_calculation_l2412_241241

-- Define conditions as constants
def boxes : ℕ := 36
def dozen_to_mangoes : ℕ := 12
def dozens_per_box : ℕ := 10

-- Define the expected correct answer for the total mangoes
def expected_total_mangoes : ℕ := 4320

-- Lean statement to prove
theorem total_mangoes_calculation :
  dozens_per_box * dozen_to_mangoes * boxes = expected_total_mangoes :=
by sorry

end total_mangoes_calculation_l2412_241241
