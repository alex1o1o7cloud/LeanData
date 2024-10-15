import Mathlib

namespace NUMINAMATH_GPT_total_stickers_l1258_125895

theorem total_stickers :
  (20.0 : ℝ) + (26.0 : ℝ) + (20.0 : ℝ) + (6.0 : ℝ) + (58.0 : ℝ) = 130.0 := by
  sorry

end NUMINAMATH_GPT_total_stickers_l1258_125895


namespace NUMINAMATH_GPT_bananas_to_oranges_equivalence_l1258_125890

theorem bananas_to_oranges_equivalence :
  (3 / 4 : ℚ) * 16 = 12 ->
  (2 / 5 : ℚ) * 10 = 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_bananas_to_oranges_equivalence_l1258_125890


namespace NUMINAMATH_GPT_baseball_card_difference_l1258_125885

theorem baseball_card_difference (marcus_cards carter_cards : ℕ) (h1 : marcus_cards = 210) (h2 : carter_cards = 152) : marcus_cards - carter_cards = 58 :=
by {
    --skip the proof
    sorry
}

end NUMINAMATH_GPT_baseball_card_difference_l1258_125885


namespace NUMINAMATH_GPT_number_neither_9_nice_nor_10_nice_500_l1258_125864

def is_k_nice (N k : ℕ) : Prop := ∃ a : ℕ, a > 0 ∧ (∃ m : ℕ, N = (k * m) + 1)

def count_k_nice (N k : ℕ) : ℕ :=
  (N - 1) / k + 1

def count_neither_9_nice_nor_10_nice (N : ℕ) : ℕ :=
  let count_9_nice := count_k_nice N 9
  let count_10_nice := count_k_nice N 10
  let lcm_9_10 := 90  -- lcm of 9 and 10
  let count_both := count_k_nice N lcm_9_10
  N - (count_9_nice + count_10_nice - count_both)

theorem number_neither_9_nice_nor_10_nice_500 : count_neither_9_nice_nor_10_nice 500 = 400 :=
  sorry

end NUMINAMATH_GPT_number_neither_9_nice_nor_10_nice_500_l1258_125864


namespace NUMINAMATH_GPT_evaluate_expression_l1258_125877

theorem evaluate_expression : (2 * (-1) + 3) * (2 * (-1) - 3) - ((-1) - 1) * ((-1) + 5) = 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1258_125877


namespace NUMINAMATH_GPT_eq_proof_l1258_125852

noncomputable def S_even : ℚ := 28
noncomputable def S_odd : ℚ := 24

theorem eq_proof : ( (S_even / S_odd - S_odd / S_even) * 2 ) = (13 / 21) :=
by
  sorry

end NUMINAMATH_GPT_eq_proof_l1258_125852


namespace NUMINAMATH_GPT_alyssa_kittens_l1258_125839

theorem alyssa_kittens (original_kittens given_away: ℕ) (h1: original_kittens = 8) (h2: given_away = 4) :
  original_kittens - given_away = 4 :=
by
  sorry

end NUMINAMATH_GPT_alyssa_kittens_l1258_125839


namespace NUMINAMATH_GPT_min_box_height_l1258_125897

noncomputable def height_of_box (x : ℝ) := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem min_box_height (x h : ℝ) (h₁ : h = height_of_box x) (h₂ : surface_area x ≥ 130) : h ≥ 25 / 3 :=
by sorry

end NUMINAMATH_GPT_min_box_height_l1258_125897


namespace NUMINAMATH_GPT_sugar_for_cake_l1258_125810

-- Definitions of given values
def sugar_for_frosting : ℝ := 0.6
def total_sugar_required : ℝ := 0.8

-- Proof statement
theorem sugar_for_cake : (total_sugar_required - sugar_for_frosting) = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_sugar_for_cake_l1258_125810


namespace NUMINAMATH_GPT_largest_cannot_be_sum_of_two_composites_l1258_125887

def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

def cannot_be_sum_of_two_composites (n : ℕ) : Prop :=
  ∀ a b : ℕ, is_composite a → is_composite b → a + b ≠ n

theorem largest_cannot_be_sum_of_two_composites :
  ∀ n, n > 11 → ¬ cannot_be_sum_of_two_composites n := 
by {
  sorry
}

end NUMINAMATH_GPT_largest_cannot_be_sum_of_two_composites_l1258_125887


namespace NUMINAMATH_GPT_number_subtracted_l1258_125836

-- Define the variables x and y
variable (x y : ℝ)

-- Define the conditions
def condition1 := 6 * x - y = 102
def condition2 := x = 40

-- Define the theorem to prove
theorem number_subtracted (h1 : condition1 x y) (h2 : condition2 x) : y = 138 :=
sorry

end NUMINAMATH_GPT_number_subtracted_l1258_125836


namespace NUMINAMATH_GPT_not_divisible_1998_minus_1_by_1000_minus_1_l1258_125835

theorem not_divisible_1998_minus_1_by_1000_minus_1 (m : ℕ) : ¬ (1000^m - 1 ∣ 1998^m - 1) :=
sorry

end NUMINAMATH_GPT_not_divisible_1998_minus_1_by_1000_minus_1_l1258_125835


namespace NUMINAMATH_GPT_unique_solution_l1258_125827

theorem unique_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : x * y + y * z + z * x = 12) (eq2 : x * y * z = 2 + x + y + z) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by 
  sorry

end NUMINAMATH_GPT_unique_solution_l1258_125827


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1258_125805

theorem problem_part1 (k m : ℝ) :
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ k ≠ 3)) →
  k = -3 :=
sorry

theorem problem_part2 (k m : ℝ) :
  ((∃ x1 x2 : ℝ, 
     ((|k|-3) * x1^2 - (k-3) * x1 + 2*m + 1 = 0) ∧
     (3 * x2 - 2 = 4 - 5 * x2 + 2 * x2) ∧
     x1 = -x2) →
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ x = -1)) →
  (k = -3 ∧ m = 5/2)) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1258_125805


namespace NUMINAMATH_GPT_inheritance_amount_l1258_125866

def federalTax (x : ℝ) : ℝ := 0.25 * x
def remainingAfterFederalTax (x : ℝ) : ℝ := x - federalTax x
def stateTax (x : ℝ) : ℝ := 0.15 * remainingAfterFederalTax x
def totalTaxes (x : ℝ) : ℝ := federalTax x + stateTax x

theorem inheritance_amount (x : ℝ) (h : totalTaxes x = 15000) : x = 41379 :=
by
  sorry

end NUMINAMATH_GPT_inheritance_amount_l1258_125866


namespace NUMINAMATH_GPT_parabola_hyperbola_intersection_l1258_125869

open Real

theorem parabola_hyperbola_intersection (p : ℝ) (hp : p > 0)
  (h_hyperbola : ∀ x y, (x^2 / 4 - y^2 = 1) → (y = 2*x ∨ y = -2*x))
  (h_parabola_directrix : ∀ y, (x^2 = 2 * p * y) → (x = -p/2)) 
  (h_area_triangle : (1/2) * (p/2) * (2 * p) = 1) :
  p = sqrt 2 := sorry

end NUMINAMATH_GPT_parabola_hyperbola_intersection_l1258_125869


namespace NUMINAMATH_GPT_xiao_dong_actual_jump_distance_l1258_125811

-- Conditions are defined here
def standard_jump_distance : ℝ := 4.00
def xiao_dong_recorded_result : ℝ := -0.32

-- Here we structure our problem
theorem xiao_dong_actual_jump_distance :
  standard_jump_distance + xiao_dong_recorded_result = 3.68 :=
by
  sorry

end NUMINAMATH_GPT_xiao_dong_actual_jump_distance_l1258_125811


namespace NUMINAMATH_GPT_probability_cond_satisfied_l1258_125830

-- Define the floor and log conditions
def cond1 (x : ℝ) : Prop := ⌊Real.log x / Real.log 2 + 1⌋ = ⌊Real.log x / Real.log 2⌋
def cond2 (x : ℝ) : Prop := ⌊Real.log (2 * x) / Real.log 2 + 1⌋ = ⌊Real.log (2 * x) / Real.log 2⌋
def valid_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Main theorem stating the proof problem
theorem probability_cond_satisfied : 
  (∀ (x : ℝ), valid_interval x → cond1 x → cond2 x → x ∈ Set.Icc (0.25:ℝ) 0.5) → 
  (0.5 - 0.25) / 1 = 1 / 4 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_probability_cond_satisfied_l1258_125830


namespace NUMINAMATH_GPT_checkerboard_black_squares_l1258_125813

theorem checkerboard_black_squares (n : ℕ) (hn : n = 33) :
  let black_squares : ℕ := (n * n + 1) / 2
  black_squares = 545 :=
by
  sorry

end NUMINAMATH_GPT_checkerboard_black_squares_l1258_125813


namespace NUMINAMATH_GPT_probability_XOXOX_l1258_125891

theorem probability_XOXOX (n_X n_O n_total : ℕ) (h_total : n_X + n_O = n_total)
  (h_X : n_X = 3) (h_O : n_O = 2) (h_total' : n_total = 5) :
  (1 / ↑(Nat.choose n_total n_X)) = (1 / 10) :=
by
  sorry

end NUMINAMATH_GPT_probability_XOXOX_l1258_125891


namespace NUMINAMATH_GPT_locus_of_midpoint_of_chord_l1258_125804

theorem locus_of_midpoint_of_chord 
  (A B C : ℝ) (h_arith_seq : A - 2 * B + C = 0) 
  (h_passing_through : ∀ t : ℝ,  t*A + -2*B + C = 0) :
  ∀ (x y : ℝ), 
    (Ax + By + C = 0) → 
    (h_on_parabola : y = -2 * x ^ 2) 
    → y + 1 = -(2 * x - 1) ^ 2 :=
sorry

end NUMINAMATH_GPT_locus_of_midpoint_of_chord_l1258_125804


namespace NUMINAMATH_GPT_square_side_length_l1258_125842

theorem square_side_length (A : ℝ) (s : ℝ) (h : A = s^2) (hA : A = 144) : s = 12 :=
by 
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_square_side_length_l1258_125842


namespace NUMINAMATH_GPT_tom_profit_calculation_l1258_125847

theorem tom_profit_calculation :
  let flour_needed := 500
  let flour_per_bag := 50
  let flour_bag_cost := 20
  let salt_needed := 10
  let salt_cost_per_pound := 0.2
  let promotion_cost := 1000
  let tickets_sold := 500
  let ticket_price := 20

  let flour_bags := flour_needed / flour_per_bag
  let cost_flour := flour_bags * flour_bag_cost
  let cost_salt := salt_needed * salt_cost_per_pound
  let total_expenses := cost_flour + cost_salt + promotion_cost
  let total_revenue := tickets_sold * ticket_price

  let profit := total_revenue - total_expenses

  profit = 8798 := by
  sorry

end NUMINAMATH_GPT_tom_profit_calculation_l1258_125847


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l1258_125822

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l1258_125822


namespace NUMINAMATH_GPT_complement_union_l1258_125843

open Set

def S : Set ℝ := { x | x > -2 }
def T : Set ℝ := { x | x^2 + 3*x - 4 ≤ 0 }

theorem complement_union :
  (compl S) ∪ T = { x : ℝ | x ≤ 1 } :=
sorry

end NUMINAMATH_GPT_complement_union_l1258_125843


namespace NUMINAMATH_GPT_count_birds_l1258_125825

theorem count_birds (b m c : ℕ) (h1 : b + m + c = 300) (h2 : 2 * b + 4 * m + 3 * c = 708) : b = 192 := 
sorry

end NUMINAMATH_GPT_count_birds_l1258_125825


namespace NUMINAMATH_GPT_second_number_is_twenty_two_l1258_125828

theorem second_number_is_twenty_two (x y : ℕ) 
  (h1 : x + y = 33) 
  (h2 : y = 2 * x) : 
  y = 22 :=
by
  sorry

end NUMINAMATH_GPT_second_number_is_twenty_two_l1258_125828


namespace NUMINAMATH_GPT_problem_1_problem_2_l1258_125838
-- Import the entire Mathlib library.

-- Problem (1)
theorem problem_1 (x y : ℝ) (h1 : |x - 3 * y| < 1 / 2) (h2 : |x + 2 * y| < 1 / 6) : |x| < 3 / 10 :=
sorry

-- Problem (2)
theorem problem_2 (x y : ℝ) : x^4 + 16 * y^4 ≥ 2 * x^3 * y + 8 * x * y^3 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1258_125838


namespace NUMINAMATH_GPT_set_diff_M_N_l1258_125815

def set_diff {α : Type} (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

def M : Set ℝ := {x | |x + 1| ≤ 2}

def N : Set ℝ := {x | ∃ α : ℝ, x = |Real.sin α| }

theorem set_diff_M_N :
  set_diff M N = {x | -3 ≤ x ∧ x < 0} :=
by
  sorry

end NUMINAMATH_GPT_set_diff_M_N_l1258_125815


namespace NUMINAMATH_GPT_original_price_of_car_l1258_125823

-- Let P be the original price of the car
variable (P : ℝ)

-- Condition: The car's value is reduced by 30%
-- Condition: The car's current value is $2800, which means 70% of the original price
def car_current_value_reduced (P : ℝ) : Prop :=
  0.70 * P = 2800

-- Theorem: Prove that the original price of the car is $4000
theorem original_price_of_car (P : ℝ) (h : car_current_value_reduced P) : P = 4000 := by
  sorry

end NUMINAMATH_GPT_original_price_of_car_l1258_125823


namespace NUMINAMATH_GPT_find_B_age_l1258_125867

variable (a b c : ℕ)

def problem_conditions : Prop :=
  a = b + 2 ∧ b = 2 * c ∧ a + b + c = 22

theorem find_B_age (h : problem_conditions a b c) : b = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_B_age_l1258_125867


namespace NUMINAMATH_GPT_total_spending_l1258_125848

-- Conditions used as definitions
def price_pants : ℝ := 110.00
def discount_pants : ℝ := 0.30
def number_of_pants : ℕ := 4

def price_socks : ℝ := 60.00
def discount_socks : ℝ := 0.30
def number_of_socks : ℕ := 2

-- Lean 4 statement to prove the total spending
theorem total_spending :
  (number_of_pants : ℝ) * (price_pants * (1 - discount_pants)) +
  (number_of_socks : ℝ) * (price_socks * (1 - discount_socks)) = 392.00 :=
by
  sorry

end NUMINAMATH_GPT_total_spending_l1258_125848


namespace NUMINAMATH_GPT_isabel_piggy_bank_l1258_125831

theorem isabel_piggy_bank:
  ∀ (initial_amount spent_on_toy spent_on_book remaining_amount : ℕ),
  initial_amount = 204 →
  spent_on_toy = initial_amount / 2 →
  remaining_amount = initial_amount - spent_on_toy →
  spent_on_book = remaining_amount / 2 →
  remaining_amount - spent_on_book = 51 :=
by
  sorry

end NUMINAMATH_GPT_isabel_piggy_bank_l1258_125831


namespace NUMINAMATH_GPT_trig_identity_l1258_125889

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : 
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1258_125889


namespace NUMINAMATH_GPT_fraction_bad_teams_leq_l1258_125863

variable (teams total_teams : ℕ) (b : ℝ)

-- Given conditions
variable (cond₁ : total_teams = 18)
variable (cond₂ : teams = total_teams / 2)
variable (cond₃ : ∀ (rb_teams : ℕ), rb_teams ≠ 10 → rb_teams ≤ teams)

theorem fraction_bad_teams_leq (H : 18 * b ≤ teams) : b ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_fraction_bad_teams_leq_l1258_125863


namespace NUMINAMATH_GPT_james_points_l1258_125808

theorem james_points (x : ℕ) :
  13 * 3 + 20 * x = 79 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_james_points_l1258_125808


namespace NUMINAMATH_GPT_number_of_lattice_points_l1258_125857

theorem number_of_lattice_points (A B : ℝ) (h : B - A = 10) :
  ∃ n, n = 10 ∨ n = 11 :=
sorry

end NUMINAMATH_GPT_number_of_lattice_points_l1258_125857


namespace NUMINAMATH_GPT_domino_cover_grid_l1258_125872

-- Definitions representing the conditions:
def isPositive (n : ℕ) : Prop := n > 0
def divides (a b : ℕ) : Prop := ∃ k, b = k * a
def canCoverWithDominos (n k : ℕ) : Prop := ∀ i j, (i < n) → (j < n) → (∃ r, i = r * k ∨ j = r * k)

-- The hypothesis: n and k are positive integers
axiom n : ℕ
axiom k : ℕ
axiom n_positive : isPositive n
axiom k_positive : isPositive k

-- The main theorem
theorem domino_cover_grid (n k : ℕ) (n_positive : isPositive n) (k_positive : isPositive k) :
  canCoverWithDominos n k ↔ divides k n := by
  sorry

end NUMINAMATH_GPT_domino_cover_grid_l1258_125872


namespace NUMINAMATH_GPT_find_triples_l1258_125876

theorem find_triples (x p n : ℕ) (hp : Nat.Prime p) :
  2 * x * (x + 5) = p^n + 3 * (x - 1) →
  (x = 2 ∧ p = 5 ∧ n = 2) ∨ (x = 0 ∧ p = 3 ∧ n = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_triples_l1258_125876


namespace NUMINAMATH_GPT_major_axis_length_l1258_125888

theorem major_axis_length (r : ℝ) (minor_axis : ℝ) (major_axis : ℝ) 
  (h1 : r = 2) 
  (h2 : minor_axis = 2 * r) 
  (h3 : major_axis = minor_axis + 0.75 * minor_axis) : 
  major_axis = 7 := 
by 
  sorry

end NUMINAMATH_GPT_major_axis_length_l1258_125888


namespace NUMINAMATH_GPT_smallest_positive_solution_l1258_125884

theorem smallest_positive_solution :
  ∃ x : ℝ, x > 0 ∧ (x ^ 4 - 50 * x ^ 2 + 576 = 0) ∧ (∀ y : ℝ, y > 0 ∧ y ^ 4 - 50 * y ^ 2 + 576 = 0 → x ≤ y) ∧ x = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_solution_l1258_125884


namespace NUMINAMATH_GPT_shaded_region_area_l1258_125894

section

-- Define points and shapes
structure point := (x : ℝ) (y : ℝ)
def square_side_length : ℝ := 40
def square_area : ℝ := square_side_length * square_side_length

-- Points defining the square and triangles within it
def point_O : point := ⟨0, 0⟩
def point_A : point := ⟨15, 0⟩
def point_B : point := ⟨40, 25⟩
def point_C : point := ⟨40, 40⟩
def point_D1 : point := ⟨25, 40⟩
def point_E : point := ⟨0, 15⟩

-- Function to calculate the area of a triangle given base and height
def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

-- Areas of individual triangles
def triangle1_area : ℝ := triangle_area 15 15
def triangle2_area : ℝ := triangle_area 25 25
def triangle3_area : ℝ := triangle_area 15 15

-- Total area of the triangles
def total_triangles_area : ℝ := triangle1_area + triangle2_area + triangle3_area

-- Shaded area calculation
def shaded_area : ℝ := square_area - total_triangles_area

-- Statement of the theorem to be proven
theorem shaded_region_area : shaded_area = 1062.5 := by sorry

end

end NUMINAMATH_GPT_shaded_region_area_l1258_125894


namespace NUMINAMATH_GPT_total_chocolate_bars_in_large_box_l1258_125801

-- Define the given conditions
def small_boxes : ℕ := 16
def chocolate_bars_per_box : ℕ := 25

-- State the proof problem
theorem total_chocolate_bars_in_large_box :
  small_boxes * chocolate_bars_per_box = 400 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_total_chocolate_bars_in_large_box_l1258_125801


namespace NUMINAMATH_GPT_sphere_radius_is_five_l1258_125821

theorem sphere_radius_is_five
    (π : ℝ)
    (r r_cylinder h : ℝ)
    (A_sphere A_cylinder : ℝ)
    (h1 : A_sphere = 4 * π * r ^ 2)
    (h2 : A_cylinder = 2 * π * r_cylinder * h)
    (h3 : h = 10)
    (h4 : r_cylinder = 5)
    (h5 : A_sphere = A_cylinder) :
    r = 5 :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_is_five_l1258_125821


namespace NUMINAMATH_GPT_solve_x_l1258_125845

theorem solve_x (x : ℝ) (h : (x / 3) / 5 = 5 / (x / 3)) : x = 15 ∨ x = -15 :=
by sorry

end NUMINAMATH_GPT_solve_x_l1258_125845


namespace NUMINAMATH_GPT_twice_x_minus_three_lt_zero_l1258_125814

theorem twice_x_minus_three_lt_zero (x : ℝ) : (2 * x - 3 < 0) ↔ (2 * x < 3) :=
by
  sorry

end NUMINAMATH_GPT_twice_x_minus_three_lt_zero_l1258_125814


namespace NUMINAMATH_GPT_sum_zero_inv_sum_zero_a_plus_d_zero_l1258_125850

theorem sum_zero_inv_sum_zero_a_plus_d_zero 
  (a b c d : ℝ) (h1 : a ≤ b ∧ b ≤ c ∧ c ≤ d) 
  (h2 : a + b + c + d = 0) 
  (h3 : 1/a + 1/b + 1/c + 1/d = 0) :
  a + d = 0 := 
  sorry

end NUMINAMATH_GPT_sum_zero_inv_sum_zero_a_plus_d_zero_l1258_125850


namespace NUMINAMATH_GPT_train_passing_time_correct_l1258_125854

noncomputable def train_passing_time (L1 L2 : ℕ) (S1 S2 : ℕ) : ℝ :=
  let S1_mps := S1 * (1000 / 3600)
  let S2_mps := S2 * (1000 / 3600)
  let relative_speed := S1_mps + S2_mps
  let total_length := L1 + L2
  total_length / relative_speed

theorem train_passing_time_correct :
  train_passing_time 105 140 45 36 = 10.89 := by
  sorry

end NUMINAMATH_GPT_train_passing_time_correct_l1258_125854


namespace NUMINAMATH_GPT_problem_l1258_125803

def op (x y : ℝ) : ℝ := x^2 - y

theorem problem (h : ℝ) : op h (op h h) = h :=
by
  sorry

end NUMINAMATH_GPT_problem_l1258_125803


namespace NUMINAMATH_GPT_train_length_l1258_125829

theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 300)
  (h2 : time = 33) : (speed * 1000 / 3600) * time = 2750 := by
  sorry

end NUMINAMATH_GPT_train_length_l1258_125829


namespace NUMINAMATH_GPT_sum_difference_l1258_125879

def arithmetic_series_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def set_A_sum : ℕ :=
  arithmetic_series_sum 42 2 25

def set_B_sum : ℕ :=
  arithmetic_series_sum 62 2 25

theorem sum_difference :
  set_B_sum - set_A_sum = 500 :=
by
  sorry

end NUMINAMATH_GPT_sum_difference_l1258_125879


namespace NUMINAMATH_GPT_cuboid_second_edge_l1258_125896

variable (x : ℝ)

theorem cuboid_second_edge (h1 : 4 * x * 6 = 96) : x = 4 := by
  sorry

end NUMINAMATH_GPT_cuboid_second_edge_l1258_125896


namespace NUMINAMATH_GPT_hyperbola_eccentricity_is_4_l1258_125826

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
: ℝ := c / a

theorem hyperbola_eccentricity_is_4 (a b c : ℝ)
  (h_eq1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eq2 : ∀ y : ℝ, y^2 = 16 * (4 : ℝ))
  (h_focus : c = 4)
  (h_c2 : c^2 = a^2 + b^2)
  (h_bc : b^2 = a^2 * (c^2 / a^2 - 1))
: hyperbola_eccentricity a b c h_eq1 h_eq2 h_focus = 4 := by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_is_4_l1258_125826


namespace NUMINAMATH_GPT_find_C_l1258_125837

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 300) 
  (h2 : A + C = 200) 
  (h3 : B + C = 350) : 
  C = 250 := 
  by sorry

end NUMINAMATH_GPT_find_C_l1258_125837


namespace NUMINAMATH_GPT_inequality_proof_l1258_125860

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 9 * y + 3 * z) * (x + 4 * y + 2 * z) * (2 * x + 12 * y + 9 * z) ≥ 1029 * x * y * z :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1258_125860


namespace NUMINAMATH_GPT_log_problem_l1258_125818

open Real

noncomputable def lg (x : ℝ) := log x / log 10

theorem log_problem :
  lg 2 ^ 2 + lg 2 * lg 5 + lg 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_log_problem_l1258_125818


namespace NUMINAMATH_GPT_problem_l1258_125875

universe u

def U : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {x ∈ U | x ≠ 0} -- Placeholder, B itself is a generic subset of U
def A : Set ℕ := {x ∈ U | x = 3 ∨ x = 5 ∨ x = 9}

noncomputable def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

axiom h1 : A ∩ B = {3, 5}
axiom h2 : A ∩ C_U B = {9}

theorem problem : A = {3, 5, 9} :=
by
  sorry

end NUMINAMATH_GPT_problem_l1258_125875


namespace NUMINAMATH_GPT_problem1_problem2_l1258_125861

-- Definitions for Problem 1
def cond1 (x t : ℝ) : Prop := |2 * x + t| - t ≤ 8
def sol_set1 (x : ℝ) : Prop := -5 ≤ x ∧ x ≤ 4

theorem problem1 {t : ℝ} : (∀ x, cond1 x t → sol_set1 x) → t = 1 :=
sorry

-- Definitions for Problem 2
def cond2 (x y z : ℝ) : Prop := x^2 + (1 / 4) * y^2 + (1 / 9) * z^2 = 2

theorem problem2 {x y z : ℝ} : cond2 x y z → x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1258_125861


namespace NUMINAMATH_GPT_susan_age_is_11_l1258_125820

theorem susan_age_is_11 (S A : ℕ) 
  (h1 : A = S + 5) 
  (h2 : A + S = 27) : 
  S = 11 := 
by 
  sorry

end NUMINAMATH_GPT_susan_age_is_11_l1258_125820


namespace NUMINAMATH_GPT_dinitrogen_monoxide_molecular_weight_l1258_125800

def atomic_weight_N : Real := 14.01
def atomic_weight_O : Real := 16.00

def chemical_formula_N2O_weight : Real :=
  (2 * atomic_weight_N) + (1 * atomic_weight_O)

theorem dinitrogen_monoxide_molecular_weight :
  chemical_formula_N2O_weight = 44.02 :=
by
  sorry

end NUMINAMATH_GPT_dinitrogen_monoxide_molecular_weight_l1258_125800


namespace NUMINAMATH_GPT_roots_quadratic_expression_l1258_125802

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) 
  (sum_roots : m + n = -2) (product_roots : m * n = -5) : m^2 + m * n + 3 * m + n = -2 :=
sorry

end NUMINAMATH_GPT_roots_quadratic_expression_l1258_125802


namespace NUMINAMATH_GPT_miles_to_add_per_week_l1258_125840

theorem miles_to_add_per_week :
  ∀ (initial_miles_per_week : ℝ) 
    (percentage_increase : ℝ) 
    (total_days : ℕ) 
    (days_in_week : ℕ),
    initial_miles_per_week = 100 →
    percentage_increase = 0.2 →
    total_days = 280 →
    days_in_week = 7 →
    ((initial_miles_per_week * (1 + percentage_increase)) - initial_miles_per_week) / (total_days / days_in_week) = 3 :=
by
  intros initial_miles_per_week percentage_increase total_days days_in_week
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_miles_to_add_per_week_l1258_125840


namespace NUMINAMATH_GPT_find_fourth_number_l1258_125874

variables (A B C D E F : ℝ)

theorem find_fourth_number
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) :
  D = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_fourth_number_l1258_125874


namespace NUMINAMATH_GPT_river_flow_volume_l1258_125841

theorem river_flow_volume (depth width : ℝ) (flow_rate_kmph : ℝ) :
  depth = 3 → width = 36 → flow_rate_kmph = 2 → 
  (depth * width) * (flow_rate_kmph * 1000 / 60) = 3599.64 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_river_flow_volume_l1258_125841


namespace NUMINAMATH_GPT_sum_of_third_terms_arithmetic_progressions_l1258_125833

theorem sum_of_third_terms_arithmetic_progressions
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∃ d1 : ℕ, ∀ n : ℕ, a (n + 1) = a 1 + n * d1)
  (h2 : ∃ d2 : ℕ, ∀ n : ℕ, b (n + 1) = b 1 + n * d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 5 + b 5 = 35) :
  a 3 + b 3 = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_third_terms_arithmetic_progressions_l1258_125833


namespace NUMINAMATH_GPT_hyperbola_equation_l1258_125882

theorem hyperbola_equation (h : ∃ (x y : ℝ), y = 1 / 2 * x) (p : (2, 2) ∈ {p : ℝ × ℝ | ((p.snd)^2 / 3) - ((p.fst)^2 / 12) = 1}) :
  ∀ (x y : ℝ), (y^2 / 3 - x^2 / 12 = 1) ↔ (∃ (a b : ℝ), y = a * x ∧ b * y = x ^ 2) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l1258_125882


namespace NUMINAMATH_GPT_simplify_expression_l1258_125812

theorem simplify_expression (a b c : ℝ) (h : a + b + c = 3) : 
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) →
  (1 / (b^2 + c^2 - 3 * a^2) + 1 / (a^2 + c^2 - 3 * b^2) + 1 / (a^2 + b^2 - 3 * c^2) = -3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_simplify_expression_l1258_125812


namespace NUMINAMATH_GPT_rectangle_area_l1258_125862

variable (l w : ℕ)

-- Conditions
def length_eq_four_times_width (l w : ℕ) : Prop := l = 4 * w
def perimeter_eq_200 (l w : ℕ) : Prop := 2 * l + 2 * w = 200

-- Question: Prove the area is 1600 square centimeters
theorem rectangle_area (h1 : length_eq_four_times_width l w) (h2 : perimeter_eq_200 l w) :
  l * w = 1600 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1258_125862


namespace NUMINAMATH_GPT_find_a_l1258_125853

theorem find_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {1, 2})
  (hB : B = {a, a^2 + 1})
  (hUnion : A ∪ B = {0, 1, 2}) :
  a = 0 :=
sorry

end NUMINAMATH_GPT_find_a_l1258_125853


namespace NUMINAMATH_GPT_circles_tangent_area_l1258_125859

noncomputable def triangle_area (r1 r2 r3 : ℝ) := 
  let d1 := r1 + r2
  let d2 := r2 + r3
  let d3 := r1 + r3
  let s := (d1 + d2 + d3) / 2
  (s * (s - d1) * (s - d2) * (s - d3)).sqrt

theorem circles_tangent_area :
  let r1 := 5
  let r2 := 12
  let r3 := 13
  let area := triangle_area r1 r2 r3 / (4 * (r1 + r2 + r3)).sqrt
  area = 120 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_circles_tangent_area_l1258_125859


namespace NUMINAMATH_GPT_investment_amount_l1258_125883

theorem investment_amount (A_investment B_investment total_profit A_share : ℝ)
  (hA_investment : A_investment = 100)
  (hB_investment_months : B_investment > 0)
  (h_total_profit : total_profit = 100)
  (h_A_share : A_share = 50)
  (h_conditions : A_share / total_profit = (A_investment * 12) / ((A_investment * 12) + (B_investment * 6))) :
  B_investment = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_investment_amount_l1258_125883


namespace NUMINAMATH_GPT_problem_1_problem_2_l1258_125849

noncomputable def f (x a : ℝ) : ℝ := abs (x + a) + abs (x - 2)

-- (1) Prove that, given f(x) and a = -3, the solution set for f(x) ≥ 3 is (-∞, 1] ∪ [4, +∞)
theorem problem_1 (x : ℝ) : 
  (∃ (a : ℝ), a = -3 ∧ f x a ≥ 3) ↔ (x ≤ 1 ∨ x ≥ 4) :=
sorry

-- (2) Prove that for f(x) to be ≥ 3 for all x, the range of a is a ≥ 1 or a ≤ -5
theorem problem_2 : 
  (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≥ 1 ∨ a ≤ -5) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1258_125849


namespace NUMINAMATH_GPT_team_a_games_played_l1258_125878

theorem team_a_games_played (a b: ℕ) (hA_wins : 3 * a = 4 * wins_A)
(hB_wins : 2 * b = 3 * wins_B)
(hB_more_wins : wins_B = wins_A + 8)
(hB_more_loss : b - wins_B = a - wins_A + 8) :
  a = 192 := 
by
  sorry

end NUMINAMATH_GPT_team_a_games_played_l1258_125878


namespace NUMINAMATH_GPT_maximum_delta_value_l1258_125865

-- Definition of the sequence a 
def a (n : ℕ) : ℕ := 1 + n^3

-- Definition of δ_n as the gcd of consecutive terms in the sequence a
def delta (n : ℕ) : ℕ := Nat.gcd (a (n + 1)) (a n)

-- Main theorem statement
theorem maximum_delta_value : ∃ n, delta n = 7 :=
by
  -- Insert the proof later
  sorry

end NUMINAMATH_GPT_maximum_delta_value_l1258_125865


namespace NUMINAMATH_GPT_largest_stickers_per_page_l1258_125855

theorem largest_stickers_per_page :
  Nat.gcd (Nat.gcd 1050 1260) 945 = 105 := 
sorry

end NUMINAMATH_GPT_largest_stickers_per_page_l1258_125855


namespace NUMINAMATH_GPT_k_positive_first_third_quadrants_l1258_125873

theorem k_positive_first_third_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, (x > 0 → k*x > 0) ∧ (x < 0 → k*x < 0)) → k > 0 :=
by
  sorry

end NUMINAMATH_GPT_k_positive_first_third_quadrants_l1258_125873


namespace NUMINAMATH_GPT_ratio_x_y_l1258_125806

-- Definitions based on conditions
variables (a b c x y : ℝ) 

-- Conditions
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2)
def a_b_ratio (a b : ℝ) := (a / b = 2 / 5)
def segments_ratio (a b c x y : ℝ) := (x = a^2 / c) ∧ (y = b^2 / c)
def perpendicular_division (x y a b : ℝ) := ((a^2 / x) = c) ∧ ((b^2 / y) = c)

-- The proof statement we need
theorem ratio_x_y : 
  ∀ (a b c x y : ℝ),
    right_triangle a b c → 
    a_b_ratio a b → 
    segments_ratio a b c x y → 
    (x / y = 4 / 25) :=
by sorry

end NUMINAMATH_GPT_ratio_x_y_l1258_125806


namespace NUMINAMATH_GPT_sum_of_square_roots_of_consecutive_odd_numbers_l1258_125846

theorem sum_of_square_roots_of_consecutive_odd_numbers :
  (Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9)) = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_square_roots_of_consecutive_odd_numbers_l1258_125846


namespace NUMINAMATH_GPT_angle_B_lt_90_l1258_125851

theorem angle_B_lt_90 {a b c : ℝ} (h_arith : b = (a + c) / 2) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  ∃ (A B C : ℝ), B < 90 :=
sorry

end NUMINAMATH_GPT_angle_B_lt_90_l1258_125851


namespace NUMINAMATH_GPT_square_side_to_diagonal_ratio_l1258_125817

theorem square_side_to_diagonal_ratio (s : ℝ) : 
  s / (s * Real.sqrt 2) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_square_side_to_diagonal_ratio_l1258_125817


namespace NUMINAMATH_GPT_cone_volume_l1258_125809

theorem cone_volume (l : ℝ) (circumference : ℝ) (radius : ℝ) (height : ℝ) (volume : ℝ) 
  (h1 : l = 8) 
  (h2 : circumference = 6 * Real.pi) 
  (h3 : radius = circumference / (2 * Real.pi))
  (h4 : height = Real.sqrt (l^2 - radius^2)) 
  (h5 : volume = (1 / 3) * Real.pi * radius^2 * height) :
  volume = 3 * Real.sqrt 55 * Real.pi := 
  by 
    sorry

end NUMINAMATH_GPT_cone_volume_l1258_125809


namespace NUMINAMATH_GPT_spent_on_video_game_l1258_125886

def saved_September : ℕ := 30
def saved_October : ℕ := 49
def saved_November : ℕ := 46
def money_left : ℕ := 67
def total_saved := saved_September + saved_October + saved_November

theorem spent_on_video_game : total_saved - money_left = 58 := by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_spent_on_video_game_l1258_125886


namespace NUMINAMATH_GPT_line_through_points_l1258_125816

-- Define the conditions and the required proof statement
theorem line_through_points (x1 y1 z1 x2 y2 z2 x y z m n p : ℝ) :
  (∃ m n p, (x-x1) / m = (y-y1) / n ∧ (y-y1) / n = (z-z1) / p) → 
  (x-x1) / (x2 - x1) = (y-y1) / (y2 - y1) ∧ 
  (y-y1) / (y2 - y1) = (z-z1) / (z2 - z1) :=
sorry

end NUMINAMATH_GPT_line_through_points_l1258_125816


namespace NUMINAMATH_GPT_resized_height_l1258_125832

-- Define original dimensions
def original_width : ℝ := 4.5
def original_height : ℝ := 3

-- Define new width
def new_width : ℝ := 13.5

-- Define new height to be proven
def new_height : ℝ := 9

-- Theorem statement
theorem resized_height :
  (new_width / original_width) * original_height = new_height :=
by
  -- The statement that equates the new height calculated proportionately to 9
  sorry

end NUMINAMATH_GPT_resized_height_l1258_125832


namespace NUMINAMATH_GPT_sphere_cube_volume_ratio_l1258_125856

theorem sphere_cube_volume_ratio (d a : ℝ) (h_d : d = 12) (h_a : a = 6) :
  let r := d / 2
  let V_sphere := (4 / 3) * π * r^3
  let V_cube := a^3
  V_sphere / V_cube = (4 * π) / 3 :=
by
  sorry

end NUMINAMATH_GPT_sphere_cube_volume_ratio_l1258_125856


namespace NUMINAMATH_GPT_sum_of_x_y_l1258_125807

theorem sum_of_x_y :
  ∀ (x y : ℚ), (1 / x + 1 / y = 4) → (1 / x - 1 / y = -8) → x + y = -1 / 3 := 
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_x_y_l1258_125807


namespace NUMINAMATH_GPT_expression_of_y_l1258_125824

theorem expression_of_y (x y : ℝ) (h : x - y / 2 = 1) : y = 2 * x - 2 :=
sorry

end NUMINAMATH_GPT_expression_of_y_l1258_125824


namespace NUMINAMATH_GPT_minimum_value_l1258_125881

theorem minimum_value (x : ℝ) (h : x > -3) : 2 * x + (1 / (x + 3)) ≥ 2 * Real.sqrt 2 - 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1258_125881


namespace NUMINAMATH_GPT_Sadie_l1258_125892

theorem Sadie's_homework_problems (T : ℝ) 
  (h1 : 0.40 * T = A) 
  (h2 : 0.5 * A = 28) 
  : T = 140 := 
by
  sorry

end NUMINAMATH_GPT_Sadie_l1258_125892


namespace NUMINAMATH_GPT_matrix_eq_l1258_125899

open Matrix

def matA : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 3], ![4, 2]]
def matI : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem matrix_eq (A : Matrix (Fin 2) (Fin 2) ℤ)
  (hA : A = ![![1, 3], ![4, 2]]) :
  A ^ 7 = 9936 * A ^ 2 + 12400 * 1 :=
  by
    sorry

end NUMINAMATH_GPT_matrix_eq_l1258_125899


namespace NUMINAMATH_GPT_no_solution_eqn_l1258_125844

theorem no_solution_eqn : ∀ x : ℝ, x ≠ -11 ∧ x ≠ -8 ∧ x ≠ -12 ∧ x ≠ -7 →
  ¬ (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_no_solution_eqn_l1258_125844


namespace NUMINAMATH_GPT_sum_of_quarter_circle_arcs_l1258_125819

-- Define the main variables and problem statement.
variable (D : ℝ) -- Diameter of the original circle.
variable (n : ℕ) (hn : 0 < n) -- Number of parts (positive integer).

-- Define a theorem stating that the sum of quarter-circle arcs is greater than D, but less than (pi D / 2) as n tends to infinity.
theorem sum_of_quarter_circle_arcs (hn : 0 < n) :
  D < (π * D) / 4 ∧ (π * D) / 4 < (π * D) / 2 :=
by
  sorry -- Proof of the theorem goes here.

end NUMINAMATH_GPT_sum_of_quarter_circle_arcs_l1258_125819


namespace NUMINAMATH_GPT_ratio_men_to_women_l1258_125858

theorem ratio_men_to_women
  (W M : ℕ)      -- W is the number of women, M is the number of men
  (avg_height_all : ℕ) (avg_height_female : ℕ) (avg_height_male : ℕ)
  (h1 : avg_height_all = 180)
  (h2 : avg_height_female = 170)
  (h3 : avg_height_male = 182) 
  (h_avg : (170 * W + 182 * M) / (W + M) = 180) :
  M = 5 * W :=
by
  sorry

end NUMINAMATH_GPT_ratio_men_to_women_l1258_125858


namespace NUMINAMATH_GPT_B_value_l1258_125880

theorem B_value (A B : Nat) (hA : A < 10) (hB : B < 10) (h_div99 : (100000 * A + 10000 + 1000 * 5 + 100 * B + 90 + 4) % 99 = 0) :
  B = 3 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_B_value_l1258_125880


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1258_125898

theorem quadratic_inequality_solution 
  (a b : ℝ) 
  (h1 : (∀ x : ℝ, x^2 + a * x + b > 0 → (x < 3 ∨ x > 1))) :
  ∀ x : ℝ, a * x + b < 0 → x > 3 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1258_125898


namespace NUMINAMATH_GPT_maximum_value_of_expression_l1258_125893

theorem maximum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b = 1) :
  (1 / (a + 9 * b) + 1 / (9 * a + b)) ≤ 5 / 24 := sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l1258_125893


namespace NUMINAMATH_GPT_average_shifted_data_is_7_l1258_125834

variable (x1 x2 x3 : ℝ)

theorem average_shifted_data_is_7 (h : (x1 + x2 + x3) / 3 = 5) : 
  ((x1 + 2) + (x2 + 2) + (x3 + 2)) / 3 = 7 :=
by
  sorry

end NUMINAMATH_GPT_average_shifted_data_is_7_l1258_125834


namespace NUMINAMATH_GPT_cos_240_degree_l1258_125870

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_cos_240_degree_l1258_125870


namespace NUMINAMATH_GPT_complement_of_M_in_U_l1258_125868

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l1258_125868


namespace NUMINAMATH_GPT_find_sum_of_cubes_l1258_125871

noncomputable def roots_of_polynomial := 
  ∃ a b c : ℝ, 
    (6 * a^3 + 500 * a + 1001 = 0) ∧ 
    (6 * b^3 + 500 * b + 1001 = 0) ∧ 
    (6 * c^3 + 500 * c + 1001 = 0)

theorem find_sum_of_cubes (a b c : ℝ) 
  (h : roots_of_polynomial) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 500.5 := 
sorry

end NUMINAMATH_GPT_find_sum_of_cubes_l1258_125871
