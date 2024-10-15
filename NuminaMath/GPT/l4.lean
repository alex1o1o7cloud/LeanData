import Mathlib

namespace NUMINAMATH_GPT_area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l4_454

-- Define the side lengths of squares A, B, and C
def side_length_A (s : ℝ) : ℝ := s
def side_length_B (s : ℝ) : ℝ := 2 * s
def side_length_C (s : ℝ) : ℝ := 3.6 * s

-- Define the areas of squares A, B, and C
def area_A (s : ℝ) : ℝ := (side_length_A s) ^ 2
def area_B (s : ℝ) : ℝ := (side_length_B s) ^ 2
def area_C (s : ℝ) : ℝ := (side_length_C s) ^ 2

-- Define the sum of areas of squares A and B
def sum_area_A_B (s : ℝ) : ℝ := area_A s + area_B s

-- Prove that the area of square C is 159.2% greater than the sum of areas of squares A and B
theorem area_C_greater_than_sum_area_A_B_by_159_point_2_percent (s : ℝ) : 
  ((area_C s - sum_area_A_B s) / (sum_area_A_B s)) * 100 = 159.2 := 
sorry

end NUMINAMATH_GPT_area_C_greater_than_sum_area_A_B_by_159_point_2_percent_l4_454


namespace NUMINAMATH_GPT_sin_2B_sin_A_sin_C_eq_neg_7_over_8_l4_446

theorem sin_2B_sin_A_sin_C_eq_neg_7_over_8
    (A B C : ℝ)
    (a b c : ℝ)
    (h1 : (2 * a + c) * Real.cos B + b * Real.cos C = 0)
    (h2 : 1/2 * a * c * Real.sin B = 15 * Real.sqrt 3)
    (h3 : a + b + c = 30) :
    (2 * Real.sin B * Real.cos B) / (Real.sin A + Real.sin C) = -7/8 := 
sorry

end NUMINAMATH_GPT_sin_2B_sin_A_sin_C_eq_neg_7_over_8_l4_446


namespace NUMINAMATH_GPT_negation_of_at_least_three_is_at_most_two_l4_472

theorem negation_of_at_least_three_is_at_most_two :
  (¬ (∀ n : ℕ, n ≥ 3)) ↔ (∃ n : ℕ, n ≤ 2) :=
sorry

end NUMINAMATH_GPT_negation_of_at_least_three_is_at_most_two_l4_472


namespace NUMINAMATH_GPT_nested_composition_l4_440

def g (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem nested_composition : g (g (g (g (g (g 2))))) = 2 := by
  sorry

end NUMINAMATH_GPT_nested_composition_l4_440


namespace NUMINAMATH_GPT_find_brown_mms_second_bag_l4_447

variable (x : ℕ)

-- Definitions based on the conditions
def BrownMmsFirstBag := 9
def BrownMmsThirdBag := 8
def BrownMmsFourthBag := 8
def BrownMmsFifthBag := 3
def AveBrownMmsPerBag := 8
def NumBags := 5

-- Condition specifying the average brown M&Ms per bag
axiom average_condition : AveBrownMmsPerBag = (BrownMmsFirstBag + x + BrownMmsThirdBag + BrownMmsFourthBag + BrownMmsFifthBag) / NumBags

-- Prove the number of brown M&Ms in the second bag
theorem find_brown_mms_second_bag : x = 12 := by
  sorry

end NUMINAMATH_GPT_find_brown_mms_second_bag_l4_447


namespace NUMINAMATH_GPT_inscribed_angle_sum_l4_462

theorem inscribed_angle_sum : 
  let arcs := 24 
  let arc_to_angle (n : ℕ) := 360 / arcs * n / 2 
  (arc_to_angle 4 + arc_to_angle 6 = 75) :=
by
  sorry

end NUMINAMATH_GPT_inscribed_angle_sum_l4_462


namespace NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l4_445

-- Define the point through which the line passes
def point : ℝ × ℝ := (3, -2)

-- Define the property of having equal absolute intercepts
def has_equal_absolute_intercepts (a b : ℝ) : Prop :=
  |a| = |b|

-- Define the general form of a line equation
def line_eq (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Main theorem: Any line passing through (3, -2) with equal absolute intercepts satisfies the given equations
theorem line_through_point_with_equal_intercepts (a b : ℝ) :
  has_equal_absolute_intercepts a b
  → line_eq 2 3 0 3 (-2)
  ∨ line_eq 1 1 (-1) 3 (-2)
  ∨ line_eq 1 (-1) (-5) 3 (-2) :=
by {
  sorry
}

end NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l4_445


namespace NUMINAMATH_GPT_container_could_be_emptied_l4_419

theorem container_could_be_emptied (a b c : ℕ) (h : 0 ≤ a ∧ a ≤ b ∧ b ≤ c) :
  ∃ (a' b' c' : ℕ), (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
  (∀ x y z : ℕ, (a, b, c) = (x, y, z) → (a', b', c') = (y + y, z - y, x - y)) :=
sorry

end NUMINAMATH_GPT_container_could_be_emptied_l4_419


namespace NUMINAMATH_GPT_price_difference_l4_408

/-- Given an original price, two successive price increases, and special deal prices for a fixed number of items, 
    calculate the difference between the final retail price and the average special deal price. -/
theorem price_difference
  (original_price : ℝ) (first_increase_percent: ℝ) (second_increase_percent: ℝ)
  (special_deal_percent_1: ℝ) (num_items_1: ℕ) (special_deal_percent_2: ℝ) (num_items_2: ℕ)
  (final_retail_price : ℝ) (average_special_deal_price : ℝ) :
  original_price = 50 →
  first_increase_percent = 0.30 →
  second_increase_percent = 0.15 →
  special_deal_percent_1 = 0.70 →
  num_items_1 = 50 →
  special_deal_percent_2 = 0.85 →
  num_items_2 = 100 →
  final_retail_price = original_price * (1 + first_increase_percent) * (1 + second_increase_percent) →
  average_special_deal_price = 
    (num_items_1 * (special_deal_percent_1 * final_retail_price) + 
    num_items_2 * (special_deal_percent_2 * final_retail_price)) / 
    (num_items_1 + num_items_2) →
  final_retail_price - average_special_deal_price = 14.95 :=
by
  intros
  sorry

end NUMINAMATH_GPT_price_difference_l4_408


namespace NUMINAMATH_GPT_impossible_pawn_placement_l4_468

theorem impossible_pawn_placement :
  ¬(∃ a b c : ℕ, a + b + c = 50 ∧ 
  ∀ (x y z : ℕ), 2 * a ≤ x ∧ x ≤ 2 * b ∧ 2 * b ≤ y ∧ y ≤ 2 * c ∧ 2 * c ≤ z ∧ z ≤ 2 * a) := sorry

end NUMINAMATH_GPT_impossible_pawn_placement_l4_468


namespace NUMINAMATH_GPT_check_correct_l4_479

-- Given the conditions
variable (x y : ℕ) (H1 : 10 ≤ x ∧ x ≤ 81) (H2 : y = x + 18)

-- Rewrite the problem and correct answer for verification in Lean
theorem check_correct (Hx : 10 ≤ x ∧ x ≤ 81) (Hy : y = x + 18) : 
  y = 2 * x ↔ x = 18 := 
by
  sorry

end NUMINAMATH_GPT_check_correct_l4_479


namespace NUMINAMATH_GPT_cartons_in_load_l4_456

theorem cartons_in_load 
  (crate_weight : ℕ)
  (carton_weight : ℕ)
  (num_crates : ℕ)
  (total_load_weight : ℕ)
  (h1 : crate_weight = 4)
  (h2 : carton_weight = 3)
  (h3 : num_crates = 12)
  (h4 : total_load_weight = 96) :
  ∃ C : ℕ, num_crates * crate_weight + C * carton_weight = total_load_weight ∧ C = 16 := 
by 
  sorry

end NUMINAMATH_GPT_cartons_in_load_l4_456


namespace NUMINAMATH_GPT_shelter_total_cats_l4_473

theorem shelter_total_cats (total_adult_cats num_female_cats num_litters avg_kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 150) 
  (h2 : num_female_cats = 2 * total_adult_cats / 3)
  (h3 : num_litters = 2 * num_female_cats / 3)
  (h4 : avg_kittens_per_litter = 5):
  total_adult_cats + num_litters * avg_kittens_per_litter = 480 :=
by
  sorry

end NUMINAMATH_GPT_shelter_total_cats_l4_473


namespace NUMINAMATH_GPT_percentage_difference_is_20_l4_441

/-
Barry can reach apples that are 5 feet high.
Larry is 5 feet tall.
When Barry stands on Larry's shoulders, they can reach 9 feet high.
-/
def Barry_height : ℝ := 5
def Larry_height : ℝ := 5
def Combined_height : ℝ := 9

/-
Prove the percentage difference between Larry's full height and his shoulder height is 20%.
-/
theorem percentage_difference_is_20 :
  ((Larry_height - (Combined_height - Barry_height)) / Larry_height) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_is_20_l4_441


namespace NUMINAMATH_GPT_least_xy_l4_487

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end NUMINAMATH_GPT_least_xy_l4_487


namespace NUMINAMATH_GPT_domain_of_f_l4_414

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2

theorem domain_of_f :
  {x : ℝ | x + 1 > 0} = {x : ℝ | x > -1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l4_414


namespace NUMINAMATH_GPT_ways_to_turn_off_lights_l4_429

-- Define the problem conditions
def streetlights := 12
def can_turn_off := 3
def not_turn_off_at_ends := true
def not_adjacent := true

-- The theorem to be proved
theorem ways_to_turn_off_lights : 
  ∃ n, 
  streetlights = 12 ∧ 
  can_turn_off = 3 ∧ 
  not_turn_off_at_ends ∧ 
  not_adjacent ∧ 
  n = 56 :=
by 
  sorry

end NUMINAMATH_GPT_ways_to_turn_off_lights_l4_429


namespace NUMINAMATH_GPT_value_of_expression_l4_484

theorem value_of_expression (x : ℝ) (h : x + 3 = 10) : 5 * x + 15 = 50 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l4_484


namespace NUMINAMATH_GPT_find_b_age_l4_405

theorem find_b_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 :=
sorry

end NUMINAMATH_GPT_find_b_age_l4_405


namespace NUMINAMATH_GPT_parallel_vectors_l4_453

def a : (ℝ × ℝ) := (1, -2)
def b (x : ℝ) : (ℝ × ℝ) := (-2, x)

theorem parallel_vectors (x : ℝ) (h : 1 / -2 = -2 / x) : x = 4 := by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l4_453


namespace NUMINAMATH_GPT_least_possible_value_d_l4_469

theorem least_possible_value_d 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (hxy : x < y)
  (hyz : y < z)
  (hyx_gt_five : y - x > 5) : 
  z - x = 9 :=
sorry

end NUMINAMATH_GPT_least_possible_value_d_l4_469


namespace NUMINAMATH_GPT_children_on_ferris_wheel_l4_416

theorem children_on_ferris_wheel (x : ℕ) (h : 5 * x + 3 * 5 + 8 * 2 * 5 = 110) : x = 3 :=
sorry

end NUMINAMATH_GPT_children_on_ferris_wheel_l4_416


namespace NUMINAMATH_GPT_find_f2_l4_417

theorem find_f2 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 2 * x ^ 2) :
  f 2 = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l4_417


namespace NUMINAMATH_GPT_wrapping_paper_l4_481

theorem wrapping_paper (total_used_per_roll : ℚ) (number_of_presents : ℕ) (fraction_used : ℚ) (fraction_left : ℚ) 
  (h1 : total_used_per_roll = 2 / 5) 
  (h2 : number_of_presents = 5) 
  (h3 : fraction_used = total_used_per_roll / number_of_presents) 
  (h4 : fraction_left = 1 - total_used_per_roll) : 
  fraction_used = 2 / 25 ∧ fraction_left = 3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_wrapping_paper_l4_481


namespace NUMINAMATH_GPT_exists_positive_n_l4_435

theorem exists_positive_n {k : ℕ} (h_k : 0 < k) {m : ℕ} (h_m : m % 2 = 1) :
  ∃ n : ℕ, 0 < n ∧ (n^n - m) % 2^k = 0 := 
sorry

end NUMINAMATH_GPT_exists_positive_n_l4_435


namespace NUMINAMATH_GPT_common_root_value_l4_443

theorem common_root_value (p : ℝ) (hp : p > 0) : 
  (∃ x : ℝ, 3 * x ^ 2 - 4 * p * x + 9 = 0 ∧ x ^ 2 - 2 * p * x + 5 = 0) ↔ p = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_common_root_value_l4_443


namespace NUMINAMATH_GPT_investment_ratio_l4_425

theorem investment_ratio (P Q : ℝ) (h1 : (P * 5) / (Q * 9) = 7 / 9) : P / Q = 7 / 5 :=
by sorry

end NUMINAMATH_GPT_investment_ratio_l4_425


namespace NUMINAMATH_GPT_equal_roots_polynomial_l4_433

open ComplexConjugate

theorem equal_roots_polynomial (k : ℚ) :
  (3 : ℚ) * x^2 - k * x + 2 * x + (12 : ℚ) = 0 → 
  (b : ℚ) ^ 2 - 4 * (3 : ℚ) * (12 : ℚ) = 0 ↔ k = -10 ∨ k = 14 :=
by
  sorry

end NUMINAMATH_GPT_equal_roots_polynomial_l4_433


namespace NUMINAMATH_GPT_profit_percent_l4_407

theorem profit_percent (marked_price : ℝ) (num_bought : ℝ) (num_payed_price : ℝ) (discount_percent : ℝ) : 
  num_bought = 56 → 
  num_payed_price = 46 → 
  discount_percent = 0.01 →
  marked_price = 1 →
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 20.52 :=
by 
  intro hnum_bought hnum_payed_price hdiscount_percent hmarked_price 
  let cost_price := num_payed_price
  let selling_price_per_pen := marked_price * (1 - discount_percent)
  let total_selling_price := num_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  sorry

end NUMINAMATH_GPT_profit_percent_l4_407


namespace NUMINAMATH_GPT_E_runs_is_20_l4_483

-- Definitions of runs scored by each batsman as multiples of 4
def a := 28
def e := 20
def d := e + 12
def b := d + e
def c := 107 - b
def total_runs := a + b + c + d + e

-- Adding conditions
axiom A_max: a > b ∧ a > c ∧ a > d ∧ a > e
axiom runs_multiple_of_4: ∀ (x : ℕ), x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e → x % 4 = 0
axiom average_runs: total_runs = 180
axiom d_condition: d = e + 12
axiom e_condition: e = a - 8
axiom b_condition: b = d + e
axiom bc_condition: b + c = 107

theorem E_runs_is_20 : e = 20 := by
  sorry

end NUMINAMATH_GPT_E_runs_is_20_l4_483


namespace NUMINAMATH_GPT_max_area_of_fencing_l4_485

theorem max_area_of_fencing (P : ℕ) (hP : P = 150) 
  (x y : ℕ) (h1 : x + y = P / 2) : (x * y) ≤ 1406 :=
sorry

end NUMINAMATH_GPT_max_area_of_fencing_l4_485


namespace NUMINAMATH_GPT_value_is_sqrt_5_over_3_l4_451

noncomputable def findValue (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) : ℝ :=
  (x + y) / (x - y)

theorem value_is_sqrt_5_over_3 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x / y + y / x = 8) :
  findValue x y h1 h2 h3 = Real.sqrt (5 / 3) :=
sorry

end NUMINAMATH_GPT_value_is_sqrt_5_over_3_l4_451


namespace NUMINAMATH_GPT_ad_plus_bc_eq_pm_one_l4_474

theorem ad_plus_bc_eq_pm_one
  (a b c d : ℤ)
  (h1 : ∃ n : ℤ, n = ad + bc ∧ n ∣ a ∧ n ∣ b ∧ n ∣ c ∧ n ∣ d) :
  ad + bc = 1 ∨ ad + bc = -1 := 
sorry

end NUMINAMATH_GPT_ad_plus_bc_eq_pm_one_l4_474


namespace NUMINAMATH_GPT_find_m_repeated_root_l4_460

theorem find_m_repeated_root (m : ℝ) :
  (∃ x : ℝ, (x - 1) ≠ 0 ∧ (m - 1) - x = 0) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_repeated_root_l4_460


namespace NUMINAMATH_GPT_find_a1_range_a1_l4_478

variables (a_1 : ℤ) (d : ℤ := -1) (S : ℕ → ℤ)

-- Definition of sum of first n terms in an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- Definition of nth term in an arithmetic sequence
def arithmetic_nth_term (n : ℕ) : ℤ := a_1 + (n - 1) * d

-- Given conditions for the problems
axiom S_def : ∀ n, S n = arithmetic_sum a_1 d n

-- Problem 1: Proving a1 = 1 given S_5 = -5
theorem find_a1 (h : S 5 = -5) : a_1 = 1 :=
by
  sorry

-- Problem 2: Proving range of a1 given S_n ≤ a_n for any positive integer n
theorem range_a1 (h : ∀ n : ℕ, n > 0 → S n ≤ arithmetic_nth_term a_1 d n) : a_1 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_range_a1_l4_478


namespace NUMINAMATH_GPT_determine_specialty_l4_459

variables 
  (Peter_is_mathematician Sergey_is_physicist Roman_is_physicist : Prop)
  (Peter_is_chemist Sergey_is_mathematician Roman_is_chemist : Prop)

-- Conditions
axiom cond1 : Peter_is_mathematician → ¬ Sergey_is_physicist
axiom cond2 : ¬ Roman_is_physicist → Peter_is_mathematician
axiom cond3 : ¬ Sergey_is_mathematician → Roman_is_chemist

theorem determine_specialty 
  (h1 : ¬ Roman_is_physicist)
: Peter_is_chemist ∧ Sergey_is_mathematician ∧ Roman_is_physicist := 
by sorry

end NUMINAMATH_GPT_determine_specialty_l4_459


namespace NUMINAMATH_GPT_cindy_correct_answer_l4_412

theorem cindy_correct_answer (x : ℝ) (h : (x - 10) / 5 = 50) : (x - 5) / 10 = 25.5 :=
sorry

end NUMINAMATH_GPT_cindy_correct_answer_l4_412


namespace NUMINAMATH_GPT_integer_solutions_to_cube_sum_eq_2_pow_30_l4_406

theorem integer_solutions_to_cube_sum_eq_2_pow_30 (x y : ℤ) :
  x^3 + y^3 = 2^30 → (x = 0 ∧ y = 2^10) ∨ (x = 2^10 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_to_cube_sum_eq_2_pow_30_l4_406


namespace NUMINAMATH_GPT_rationalize_denominator_l4_432

theorem rationalize_denominator :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) := by sorry

end NUMINAMATH_GPT_rationalize_denominator_l4_432


namespace NUMINAMATH_GPT_number_of_crayons_given_to_friends_l4_444

def totalCrayonsLostOrGivenAway := 229
def crayonsLost := 16
def crayonsGivenToFriends := totalCrayonsLostOrGivenAway - crayonsLost

theorem number_of_crayons_given_to_friends :
  crayonsGivenToFriends = 213 :=
by
  sorry

end NUMINAMATH_GPT_number_of_crayons_given_to_friends_l4_444


namespace NUMINAMATH_GPT_find_recip_sum_of_shifted_roots_l4_455

noncomputable def reciprocal_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) : ℝ :=
  1 / (α + 2) + 1 / (β + 2) + 1 / (γ + 2)

theorem find_recip_sum_of_shifted_roots (α β γ : ℝ) (hαβγ : Polynomial.roots (Polynomial.C α * Polynomial.C β * Polynomial.C γ + Polynomial.X ^ 3 - 2 * Polynomial.X ^ 2 - Polynomial.X + Polynomial.C 2) = {α, β, γ}) :
  reciprocal_sum_of_shifted_roots α β γ hαβγ = -19 / 14 :=
  sorry

end NUMINAMATH_GPT_find_recip_sum_of_shifted_roots_l4_455


namespace NUMINAMATH_GPT_stairs_climbed_l4_494

theorem stairs_climbed (s v r : ℕ) 
  (h_s: s = 318) 
  (h_v: v = 18 + s / 2) 
  (h_r: r = 2 * v) 
  : s + v + r = 849 :=
by {
  sorry
}

end NUMINAMATH_GPT_stairs_climbed_l4_494


namespace NUMINAMATH_GPT_smallest_n_for_symmetry_property_l4_482

-- Define the setup for the problem
def has_required_symmetry (n : ℕ) : Prop :=
∀ (S : Finset (Fin n)), S.card = 5 →
∃ (l : Fin n → Fin n), (∀ v ∈ S, l v ≠ v) ∧ (∀ v ∈ S, l v ∉ S)

-- The main lemma we are proving
theorem smallest_n_for_symmetry_property : ∃ n : ℕ, (∀ m < n, ¬ has_required_symmetry m) ∧ has_required_symmetry 14 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_symmetry_property_l4_482


namespace NUMINAMATH_GPT_find_lisa_speed_l4_480

theorem find_lisa_speed (Distance : ℕ) (Time : ℕ) (h1 : Distance = 256) (h2 : Time = 8) : Distance / Time = 32 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_lisa_speed_l4_480


namespace NUMINAMATH_GPT_sum_of_roots_quadratic_eq_l4_467

theorem sum_of_roots_quadratic_eq (x₁ x₂ : ℝ) (h : x₁^2 + 2 * x₁ - 4 = 0 ∧ x₂^2 + 2 * x₂ - 4 = 0) : 
  x₁ + x₂ = -2 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_quadratic_eq_l4_467


namespace NUMINAMATH_GPT_arithmetic_mean_12_24_36_48_l4_464

theorem arithmetic_mean_12_24_36_48 : (12 + 24 + 36 + 48) / 4 = 30 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_12_24_36_48_l4_464


namespace NUMINAMATH_GPT_find_c_l4_492

-- Define the polynomial f(x)
def f (c : ℚ) (x : ℚ) : ℚ := 2 * c * x^3 + 14 * x^2 - 6 * c * x + 25

-- State the problem in Lean 4
theorem find_c (c : ℚ) : (∀ x : ℚ, f c x = 0 ↔ x = (-5)) → c = 75 / 44 := 
by sorry

end NUMINAMATH_GPT_find_c_l4_492


namespace NUMINAMATH_GPT_number_of_integer_solutions_l4_493

theorem number_of_integer_solutions (x : ℤ) :
  (∃ n : ℤ, n^2 = x^4 + 8*x^3 + 18*x^2 + 8*x + 36) ↔ x = -1 :=
sorry

end NUMINAMATH_GPT_number_of_integer_solutions_l4_493


namespace NUMINAMATH_GPT_total_volume_of_four_boxes_l4_463

-- Define the edge length of the cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_one_cube := edge_length ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 4

-- The total volume of the four cubes
def total_volume := number_of_cubes * volume_of_one_cube

-- Statement to prove that the total volume equals 500 cubic feet
theorem total_volume_of_four_boxes :
  total_volume = 500 :=
sorry

end NUMINAMATH_GPT_total_volume_of_four_boxes_l4_463


namespace NUMINAMATH_GPT_range_of_m_l4_495

noncomputable def proposition_p (x m : ℝ) := (x - m) ^ 2 > 3 * (x - m)
noncomputable def proposition_q (x : ℝ) := x ^ 2 + 3 * x - 4 < 0

theorem range_of_m (m : ℝ) : 
  (∀ x, proposition_p x m → proposition_q x) → 
  (1 ≤ m ∨ m ≤ -7) :=
sorry

end NUMINAMATH_GPT_range_of_m_l4_495


namespace NUMINAMATH_GPT_mn_sum_value_l4_448

-- Definition of the problem conditions
def faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_consecutive (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) ∨
  (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3) ∨
  (a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨
  (a = 5 ∧ b = 6) ∨ (a = 6 ∧ b = 5) ∨
  (a = 6 ∧ b = 7) ∨ (a = 7 ∧ b = 6) ∨
  (a = 7 ∧ b = 8) ∨ (a = 8 ∧ b = 7) ∨
  (a = 8 ∧ b = 9) ∨ (a = 9 ∧ b = 8) ∨
  (a = 9 ∧ b = 1) ∨ (a = 1 ∧ b = 9)

noncomputable def m_n_sum : ℕ :=
  let total_permutations := 5040
  let valid_permutations := 60
  let probability := valid_permutations / total_permutations
  let m := 1
  let n := total_permutations / valid_permutations
  m + n

theorem mn_sum_value : m_n_sum = 85 :=
  sorry

end NUMINAMATH_GPT_mn_sum_value_l4_448


namespace NUMINAMATH_GPT_slope_ge_one_sum_pq_eq_17_l4_400

noncomputable def Q_prob_satisfaction : ℚ := 1/16

theorem slope_ge_one_sum_pq_eq_17 :
  let p := 1
  let q := 16
  p + q = 17 := by
  sorry

end NUMINAMATH_GPT_slope_ge_one_sum_pq_eq_17_l4_400


namespace NUMINAMATH_GPT_min_value_expr_l4_431

variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b = 1)

theorem min_value_expr : (1 / (2 * a)) + (1 / (2 * b)) + (8 / (a + b)) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l4_431


namespace NUMINAMATH_GPT_factorization_correctness_l4_411

theorem factorization_correctness :
  ∀ x : ℝ, x^2 - 2*x + 1 = (x - 1)^2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_factorization_correctness_l4_411


namespace NUMINAMATH_GPT_members_who_play_both_l4_499

theorem members_who_play_both (N B T Neither : ℕ) (hN : N = 30) (hB : B = 16) (hT : T = 19) (hNeither : Neither = 2) : 
  B + T - (N - Neither) = 7 :=
by
  sorry

end NUMINAMATH_GPT_members_who_play_both_l4_499


namespace NUMINAMATH_GPT_patrol_streets_in_one_hour_l4_476

-- Definitions of the given conditions
def streets_patrolled_by_A := 36
def hours_by_A := 4
def rate_A := streets_patrolled_by_A / hours_by_A

def streets_patrolled_by_B := 55
def hours_by_B := 5
def rate_B := streets_patrolled_by_B / hours_by_B

def streets_patrolled_by_C := 42
def hours_by_C := 6
def rate_C := streets_patrolled_by_C / hours_by_C

-- Proof statement 
theorem patrol_streets_in_one_hour : rate_A + rate_B + rate_C = 27 := by
  sorry

end NUMINAMATH_GPT_patrol_streets_in_one_hour_l4_476


namespace NUMINAMATH_GPT_animath_interns_pigeonhole_l4_434

theorem animath_interns_pigeonhole (n : ℕ) (knows : Fin n → Finset (Fin n)) :
  ∃ (i j : Fin n), i ≠ j ∧ (knows i).card = (knows j).card :=
by
  sorry

end NUMINAMATH_GPT_animath_interns_pigeonhole_l4_434


namespace NUMINAMATH_GPT_total_prep_time_l4_488

-- Definitions:
def jack_time_to_put_shoes_on : ℕ := 4
def additional_time_per_toddler : ℕ := 3
def number_of_toddlers : ℕ := 2

-- Total time calculation
def total_time : ℕ :=
  let time_per_toddler := jack_time_to_put_shoes_on + additional_time_per_toddler
  let total_toddler_time := time_per_toddler * number_of_toddlers
  total_toddler_time + jack_time_to_put_shoes_on

-- Theorem:
theorem total_prep_time :
  total_time = 18 :=
sorry

end NUMINAMATH_GPT_total_prep_time_l4_488


namespace NUMINAMATH_GPT_eleven_power_2023_mod_50_l4_409

theorem eleven_power_2023_mod_50 :
  11^2023 % 50 = 31 :=
by
  sorry

end NUMINAMATH_GPT_eleven_power_2023_mod_50_l4_409


namespace NUMINAMATH_GPT_more_than_four_numbers_make_polynomial_prime_l4_438

def polynomial (n : ℕ) : ℤ := n^3 - 10 * n^2 + 31 * n - 17

def is_prime (k : ℤ) : Prop :=
  k > 1 ∧ ∀ m : ℤ, m > 1 ∧ m < k → ¬ (m ∣ k)

theorem more_than_four_numbers_make_polynomial_prime :
  (∃ n1 n2 n3 n4 n5 : ℕ, 
    n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧ n5 > 0 ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ n1 ≠ n5 ∧
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ n2 ≠ n5 ∧ 
    n3 ≠ n4 ∧ n3 ≠ n5 ∧ 
    n4 ≠ n5 ∧ 
    is_prime (polynomial n1) ∧
    is_prime (polynomial n2) ∧
    is_prime (polynomial n3) ∧
    is_prime (polynomial n4) ∧
    is_prime (polynomial n5)) :=
sorry

end NUMINAMATH_GPT_more_than_four_numbers_make_polynomial_prime_l4_438


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l4_465

def f (x : ℝ) : ℝ := x^5 + 2 * x^3 + x^2 + 4

theorem remainder_when_divided_by_x_minus_2 : f 2 = 56 :=
by
  -- Proof steps will go here.
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l4_465


namespace NUMINAMATH_GPT_jack_evening_emails_l4_477

theorem jack_evening_emails
  (emails_afternoon : ℕ := 3)
  (emails_morning : ℕ := 6)
  (emails_total : ℕ := 10) :
  emails_total - emails_afternoon - emails_morning = 1 :=
by
  sorry

end NUMINAMATH_GPT_jack_evening_emails_l4_477


namespace NUMINAMATH_GPT_range_of_a_l4_458

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2 * a)^x < (3 - 2 * a)^y

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : (1 ≤ a ∧ a < 2) ∨ (a ≤ -2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l4_458


namespace NUMINAMATH_GPT_find_original_number_l4_489

variable (x : ℕ)

theorem find_original_number (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l4_489


namespace NUMINAMATH_GPT_chess_tournament_max_N_l4_486

theorem chess_tournament_max_N :
  ∃ (N : ℕ), N = 120 ∧
  ∀ (S T : Finset ℕ), S.card = 15 ∧ T.card = 15 ∧
  (∀ s ∈ S, ∀ t ∈ T, (s, t) ∈ (S.product T)) ∧
  (∀ s, ∃! t, (s, t) ∈ (S.product T)) → 
  ∃ (ways_one_game : ℕ), ways_one_game = N ∧ ways_one_game = 120 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_max_N_l4_486


namespace NUMINAMATH_GPT_equivalence_negation_l4_497

-- Define irrational numbers
def is_irrational (x : ℝ) : Prop :=
  ¬ (∃ q : ℚ, x = q)

-- Define rational numbers
def is_rational (x : ℝ) : Prop :=
  ∃ q : ℚ, x = q

-- Original proposition: There exists an irrational number whose square is rational
def original_proposition : Prop :=
  ∃ x : ℝ, is_irrational x ∧ is_rational (x * x)

-- Negation of the original proposition
def negation_of_proposition : Prop :=
  ∀ x : ℝ, is_irrational x → ¬is_rational (x * x)

-- Proof statement that the negation of the original proposition is equivalent to "Every irrational number has a square that is not rational"
theorem equivalence_negation :
  (¬ original_proposition) ↔ negation_of_proposition :=
sorry

end NUMINAMATH_GPT_equivalence_negation_l4_497


namespace NUMINAMATH_GPT_find_xyz_l4_437

open Complex

theorem find_xyz (a b c x y z : ℂ)
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : x ≠ 0) 
  (h5 : y ≠ 0) 
  (h6 : z ≠ 0)
  (ha : a = (b + c) / (x + 1))
  (hb : b = (a + c) / (y + 1))
  (hc : c = (a + b) / (z + 1))
  (hxy_z_1 : x * y + x * z + y * z = 9)
  (hxy_z_2 : x + y + z = 5) :
  x * y * z = 13 := 
sorry

end NUMINAMATH_GPT_find_xyz_l4_437


namespace NUMINAMATH_GPT_find_making_lines_parallel_l4_430

theorem find_making_lines_parallel (m : ℝ) : 
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2 
  (line1_slope = line2_slope) ↔ (m = 1) := 
by
  -- definitions
  intros
  let line1_slope := -1 / (1 + m)
  let line2_slope := -m / 2
  -- equation for slopes to be equal
  have slope_equation : line1_slope = line2_slope ↔ (m = 1)
  sorry

  exact slope_equation

end NUMINAMATH_GPT_find_making_lines_parallel_l4_430


namespace NUMINAMATH_GPT_simplification_evaluation_l4_418

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  ( (2 * x - 6) / (x - 2) ) / ( (5 / (x - 2)) - (x + 2) ) = Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_GPT_simplification_evaluation_l4_418


namespace NUMINAMATH_GPT_abs_sum_eq_abs_add_iff_ab_gt_zero_l4_421

theorem abs_sum_eq_abs_add_iff_ab_gt_zero (a b : ℝ) :
  (|a + b| = |a| + |b|) → (a = 0 ∧ b = 0 ∨ ab > 0) :=
sorry

end NUMINAMATH_GPT_abs_sum_eq_abs_add_iff_ab_gt_zero_l4_421


namespace NUMINAMATH_GPT_real_solutions_l4_427

noncomputable def solveEquation (x : ℝ) : Prop :=
  (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10

theorem real_solutions :
  {x : ℝ | solveEquation x} = {x : ℝ | x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15} :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_l4_427


namespace NUMINAMATH_GPT_movie_marathon_duration_l4_470

theorem movie_marathon_duration :
  let first_movie := 2
  let second_movie := first_movie + 0.5 * first_movie
  let combined_time := first_movie + second_movie
  let third_movie := combined_time - 1
  first_movie + second_movie + third_movie = 9 := by
  sorry

end NUMINAMATH_GPT_movie_marathon_duration_l4_470


namespace NUMINAMATH_GPT_team_points_difference_l4_452

   -- Definitions for points of each member
   def Max_points : ℝ := 7
   def Dulce_points : ℝ := 5
   def Val_points : ℝ := 4 * (Max_points + Dulce_points)
   def Sarah_points : ℝ := 2 * Dulce_points
   def Steve_points : ℝ := 2.5 * (Max_points + Val_points)

   -- Definition for total points of their team
   def their_team_points : ℝ := Max_points + Dulce_points + Val_points + Sarah_points + Steve_points

   -- Definition for total points of the opponents' team
   def opponents_team_points : ℝ := 200

   -- The main theorem to prove
   theorem team_points_difference : their_team_points - opponents_team_points = 7.5 := by
     sorry
   
end NUMINAMATH_GPT_team_points_difference_l4_452


namespace NUMINAMATH_GPT_fifth_friend_payment_l4_402

/-- 
Five friends bought a piece of furniture for $120.
The first friend paid one third of the sum of the amounts paid by the other four;
the second friend paid one fourth of the sum of the amounts paid by the other four;
the third friend paid one fifth of the sum of the amounts paid by the other four;
and the fourth friend paid one sixth of the sum of the amounts paid by the other four.
Prove that the fifth friend paid $41.33.
-/
theorem fifth_friend_payment :
  ∀ (a b c d e : ℝ),
    a = 1/3 * (b + c + d + e) →
    b = 1/4 * (a + c + d + e) →
    c = 1/5 * (a + b + d + e) →
    d = 1/6 * (a + b + c + e) →
    a + b + c + d + e = 120 →
    e = 41.33 :=
by
  intros a b c d e ha hb hc hd he_sum
  sorry

end NUMINAMATH_GPT_fifth_friend_payment_l4_402


namespace NUMINAMATH_GPT_buns_distribution_not_equal_for_all_cases_l4_490

theorem buns_distribution_not_equal_for_all_cases :
  ∀ (initial_buns : Fin 30 → ℕ),
  (∃ (p : ℕ → Fin 30 → Fin 30), 
    (∀ t, 
      (∀ i, 
        (initial_buns (p t i) = initial_buns i ∨ 
         initial_buns (p t i) = initial_buns i + 2 ∨ 
         initial_buns (p t i) = initial_buns i - 2))) → 
    ¬ ∀ n : Fin 30, initial_buns n = 2) := 
sorry

end NUMINAMATH_GPT_buns_distribution_not_equal_for_all_cases_l4_490


namespace NUMINAMATH_GPT_parabola_directrix_equation_l4_403

theorem parabola_directrix_equation :
  ∀ (x y : ℝ),
  y = -4 * x^2 - 16 * x + 1 →
  ∃ d : ℝ, d = 273 / 16 ∧ y = d :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_equation_l4_403


namespace NUMINAMATH_GPT_divide_5000_among_x_and_y_l4_471

theorem divide_5000_among_x_and_y (total_amount : ℝ) (ratio_x : ℝ) (ratio_y : ℝ) (parts : ℝ) :
  total_amount = 5000 → ratio_x = 2 → ratio_y = 8 → parts = ratio_x + ratio_y → 
  (total_amount / parts) * ratio_x = 1000 := 
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_divide_5000_among_x_and_y_l4_471


namespace NUMINAMATH_GPT_find_a_l4_461

theorem find_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a * b - a - b = 4) : a = 6 :=
sorry

end NUMINAMATH_GPT_find_a_l4_461


namespace NUMINAMATH_GPT_probability_two_tails_after_two_heads_l4_424

noncomputable def fair_coin_probability : ℚ :=
  -- Given conditions:
  let p_head := (1 : ℚ) / 2
  let p_tail := (1 : ℚ) / 2

  -- Define the probability Q as stated in the problem
  let Q := ((1 : ℚ) / 4) / (1 - (1 : ℚ) / 4)

  -- Calculate the probability of starting with sequence "HTH"
  let p_HTH := p_head * p_tail * p_head

  -- Calculate the final probability
  p_HTH * Q

theorem probability_two_tails_after_two_heads :
  fair_coin_probability = (1 : ℚ) / 24 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_tails_after_two_heads_l4_424


namespace NUMINAMATH_GPT_find_n_value_l4_436

theorem find_n_value (m n k : ℝ) (h1 : n = k / m) (h2 : m = k / 2) (h3 : k ≠ 0): n = 2 :=
sorry

end NUMINAMATH_GPT_find_n_value_l4_436


namespace NUMINAMATH_GPT_trees_per_day_l4_410

def blocks_per_tree := 3
def total_blocks := 30
def days := 5

theorem trees_per_day : (total_blocks / days) / blocks_per_tree = 2 := by
  sorry

end NUMINAMATH_GPT_trees_per_day_l4_410


namespace NUMINAMATH_GPT_bacteria_growth_time_l4_423
-- Import necessary library

-- Define the conditions
def initial_bacteria_count : ℕ := 100
def final_bacteria_count : ℕ := 102400
def multiplication_factor : ℕ := 4
def multiplication_period_hours : ℕ := 6

-- Define the proof problem
theorem bacteria_growth_time :
  ∃ t : ℕ, t * multiplication_period_hours = 30 ∧ initial_bacteria_count * multiplication_factor^t = final_bacteria_count :=
by
  sorry

end NUMINAMATH_GPT_bacteria_growth_time_l4_423


namespace NUMINAMATH_GPT_number_of_routes_l4_404

variable {City : Type}
variable (A B C D E : City)
variable (AB_N AB_S AD AE BC BD CD DE : City → City → Prop)
  
theorem number_of_routes 
  (hAB_N : AB_N A B) (hAB_S : AB_S A B)
  (hAD : AD A D) (hAE : AE A E)
  (hBC : BC B C) (hBD : BD B D)
  (hCD : CD C D) (hDE : DE D E) :
  ∃ r : ℕ, r = 16 := 
sorry

end NUMINAMATH_GPT_number_of_routes_l4_404


namespace NUMINAMATH_GPT_total_gallons_l4_450

-- Definitions from conditions
def num_vans : ℕ := 6
def standard_capacity : ℕ := 8000
def reduced_capacity : ℕ := standard_capacity - (30 * standard_capacity / 100)
def increased_capacity : ℕ := standard_capacity + (50 * standard_capacity / 100)

-- Total number of specific types of vans
def num_standard_vans : ℕ := 2
def num_reduced_vans : ℕ := 1
def num_increased_vans : ℕ := num_vans - num_standard_vans - num_reduced_vans

-- The proof goal
theorem total_gallons : 
  (num_standard_vans * standard_capacity) + 
  (num_reduced_vans * reduced_capacity) + 
  (num_increased_vans * increased_capacity) = 
  57600 := 
by
  -- The necessary proof can be filled here
  sorry

end NUMINAMATH_GPT_total_gallons_l4_450


namespace NUMINAMATH_GPT_range_of_m_range_of_x_l4_420

variable {a b m : ℝ}

-- Given conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom sum_eq_one : a + b = 1

-- Problem (I): Prove range of m
theorem range_of_m (h : ab ≤ m) : m ≥ 1 / 4 := by
  sorry

variable {x : ℝ}

-- Problem (II): Prove range of x
theorem range_of_x (h : 4 / a + 1 / b ≥ |2 * x - 1| - |x + 2|) : -2 ≤ x ∧ x ≤ 6 := by
  sorry

end NUMINAMATH_GPT_range_of_m_range_of_x_l4_420


namespace NUMINAMATH_GPT_shortest_chord_length_l4_439

theorem shortest_chord_length 
  (C : ℝ → ℝ → Prop) 
  (l : ℝ → ℝ → ℝ → Prop) 
  (radius : ℝ) 
  (center_x center_y : ℝ) 
  (cx cy : ℝ) 
  (m : ℝ) :
  (∀ x y, C x y ↔ (x - 1)^2 + (y - 2)^2 = 25) →
  (∀ x y m, l x y m ↔ (2*m+1)*x + (m+1)*y - 7*m - 4 = 0) →
  center_x = 1 →
  center_y = 2 →
  radius = 5 →
  cx = 3 →
  cy = 1 →
  ∃ shortest_chord_length : ℝ, shortest_chord_length = 4 * Real.sqrt 5 := sorry

end NUMINAMATH_GPT_shortest_chord_length_l4_439


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l4_491

theorem tan_alpha_plus_pi_over_4 
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 :=
sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l4_491


namespace NUMINAMATH_GPT_value_of_x_l4_496

theorem value_of_x {x y z w v : ℝ} 
  (h1 : y * x = 3)
  (h2 : z = 3)
  (h3 : w = z * y)
  (h4 : v = w * z)
  (h5 : v = 18)
  (h6 : w = 6) :
  x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l4_496


namespace NUMINAMATH_GPT_vector_coordinates_l4_466

-- Define the given vectors.
def a : (ℝ × ℝ) := (1, 1)
def b : (ℝ × ℝ) := (1, -1)

-- Define the proof goal.
theorem vector_coordinates :
  -2 • a - b = (-3, -1) :=
by
  sorry -- Proof not required.

end NUMINAMATH_GPT_vector_coordinates_l4_466


namespace NUMINAMATH_GPT_most_likely_units_digit_sum_is_zero_l4_449

theorem most_likely_units_digit_sum_is_zero :
  ∃ (units_digit : ℕ), 
  (∀ m n : ℕ, (1 ≤ m ∧ m ≤ 9) ∧ (1 ≤ n ∧ n ≤ 9) → 
    units_digit = (m + n) % 10) ∧ 
  units_digit = 0 :=
sorry

end NUMINAMATH_GPT_most_likely_units_digit_sum_is_zero_l4_449


namespace NUMINAMATH_GPT_granger_total_payment_proof_l4_428

-- Conditions
def cost_per_can_spam := 3
def cost_per_jar_peanut_butter := 5
def cost_per_loaf_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Calculation
def total_cost_spam := quantity_spam * cost_per_can_spam
def total_cost_peanut_butter := quantity_peanut_butter * cost_per_jar_peanut_butter
def total_cost_bread := quantity_bread * cost_per_loaf_bread

-- Total amount paid
def total_amount_paid := total_cost_spam + total_cost_peanut_butter + total_cost_bread

-- Theorem to be proven
theorem granger_total_payment_proof : total_amount_paid = 59 :=
by
  sorry

end NUMINAMATH_GPT_granger_total_payment_proof_l4_428


namespace NUMINAMATH_GPT_max_value_expression_l4_475

theorem max_value_expression (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 * (a + b))) +
   Real.sqrt (Real.sqrt (b^2 * (b + c))) +
   Real.sqrt (Real.sqrt (c^2 * (c + d))) +
   Real.sqrt (Real.sqrt (d^2 * (d + a)))) ≤ 4 * Real.sqrt (Real.sqrt 2) := by
  sorry

end NUMINAMATH_GPT_max_value_expression_l4_475


namespace NUMINAMATH_GPT_fruit_salad_mixture_l4_401

theorem fruit_salad_mixture :
  ∃ (A P G : ℝ), A / P = 12 / 8 ∧ A / G = 12 / 7 ∧ P / G = 8 / 7 ∧ A = G + 10 ∧ A + P + G = 54 :=
by
  sorry

end NUMINAMATH_GPT_fruit_salad_mixture_l4_401


namespace NUMINAMATH_GPT_find_number_l4_413

theorem find_number (x : ℝ) (h : x / 0.07 = 700) : x = 49 :=
sorry

end NUMINAMATH_GPT_find_number_l4_413


namespace NUMINAMATH_GPT_value_of_f_neg6_l4_457

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = -f x

theorem value_of_f_neg6 : f (-6) = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_neg6_l4_457


namespace NUMINAMATH_GPT_all_numbers_appear_on_diagonal_l4_415

theorem all_numbers_appear_on_diagonal 
  (n : ℕ) 
  (h_odd : n % 2 = 1)
  (A : Matrix (Fin n) (Fin n) (Fin n.succ))
  (h_elements : ∀ i j, 1 ≤ A i j ∧ A i j ≤ n) 
  (h_unique_row : ∀ i k, ∃! j, A i j = k)
  (h_unique_col : ∀ j k, ∃! i, A i j = k)
  (h_symmetric : ∀ i j, A i j = A j i)
  : ∀ k, 1 ≤ k ∧ k ≤ n → ∃ i, A i i = k := 
by {
  sorry
}

end NUMINAMATH_GPT_all_numbers_appear_on_diagonal_l4_415


namespace NUMINAMATH_GPT_roots_abs_less_than_one_l4_498

theorem roots_abs_less_than_one {a b : ℝ} 
    (h : |a| + |b| < 1) 
    (x1 x2 : ℝ) 
    (h_roots : x1 * x1 + a * x1 + b = 0) 
    (h_roots' : x2 * x2 + a * x2 + b = 0) 
    : |x1| < 1 ∧ |x2| < 1 := 
sorry

end NUMINAMATH_GPT_roots_abs_less_than_one_l4_498


namespace NUMINAMATH_GPT_monotonicity_and_range_of_m_l4_422

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (1 - a) / 2 * x ^ 2 + a * x - Real.log x

theorem monotonicity_and_range_of_m (a m : ℝ) (h₀ : 2 < a) (h₁ : a < 3)
  (h₂ : ∀ (x1 x2 : ℝ), 1 ≤ x1 ∧ x1 ≤ 2 → 1 ≤ x2 ∧ x2 ≤ 2 -> ma + Real.log 2 > |f x1 a - f x2 a|):
  m ≥ 0 :=
sorry

end NUMINAMATH_GPT_monotonicity_and_range_of_m_l4_422


namespace NUMINAMATH_GPT_calc_first_term_l4_442

theorem calc_first_term (a d : ℚ)
    (h1 : 15 * (2 * a + 29 * d) = 300)
    (h2 : 20 * (2 * a + 99 * d) = 2200) :
    a = -121 / 14 :=
by
  -- We can add the sorry placeholder here as we are not providing the complete proof steps
  sorry

end NUMINAMATH_GPT_calc_first_term_l4_442


namespace NUMINAMATH_GPT_central_angle_of_regular_hexagon_l4_426

theorem central_angle_of_regular_hexagon :
  ∀ (total_angle : ℝ) (sides : ℝ), total_angle = 360 → sides = 6 → total_angle / sides = 60 :=
by
  intros total_angle sides h_total_angle h_sides
  rw [h_total_angle, h_sides]
  norm_num

end NUMINAMATH_GPT_central_angle_of_regular_hexagon_l4_426
