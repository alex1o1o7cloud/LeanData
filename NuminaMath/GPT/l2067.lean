import Mathlib

namespace NUMINAMATH_GPT_books_per_shelf_l2067_206722

theorem books_per_shelf (total_distance : ℕ) (total_shelves : ℕ) (one_way_distance : ℕ) 
  (h1 : total_distance = 3200) (h2 : total_shelves = 4) (h3 : one_way_distance = total_distance / 2) 
  (h4 : one_way_distance = 1600) :
  ∀ books_per_shelf : ℕ, books_per_shelf = one_way_distance / total_shelves := 
by
  sorry

end NUMINAMATH_GPT_books_per_shelf_l2067_206722


namespace NUMINAMATH_GPT_planting_flowers_cost_l2067_206708

theorem planting_flowers_cost 
  (flower_cost : ℕ) (clay_cost : ℕ) (soil_cost : ℕ)
  (h₁ : flower_cost = 9)
  (h₂ : clay_cost = flower_cost + 20)
  (h₃ : soil_cost = flower_cost - 2) :
  flower_cost + clay_cost + soil_cost = 45 :=
sorry

end NUMINAMATH_GPT_planting_flowers_cost_l2067_206708


namespace NUMINAMATH_GPT_rectangle_length_l2067_206777

theorem rectangle_length (P W : ℝ) (hP : P = 30) (hW : W = 10) :
  ∃ (L : ℝ), 2 * (L + W) = P ∧ L = 5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l2067_206777


namespace NUMINAMATH_GPT_percentage_return_l2067_206779

theorem percentage_return (income investment : ℝ) (h_income : income = 680) (h_investment : investment = 8160) :
  (income / investment) * 100 = 8.33 :=
by
  rw [h_income, h_investment]
  -- The rest of the proof is omitted.
  sorry

end NUMINAMATH_GPT_percentage_return_l2067_206779


namespace NUMINAMATH_GPT_sum_of_first_50_primes_is_5356_l2067_206718

open Nat

-- Define the first 50 prime numbers
def first_50_primes : List Nat := 
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 
   83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 
   179, 181, 191, 193, 197, 199, 211, 223, 227, 229]

-- Calculate their sum
def sum_first_50_primes : Nat := List.foldr (Nat.add) 0 first_50_primes

-- Now we state the theorem we want to prove
theorem sum_of_first_50_primes_is_5356 : 
  sum_first_50_primes = 5356 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_sum_of_first_50_primes_is_5356_l2067_206718


namespace NUMINAMATH_GPT_dampening_factor_l2067_206745

theorem dampening_factor (s r : ℝ) 
  (h1 : s / (1 - r) = 16) 
  (h2 : s * r / (1 - r^2) = -6) :
  r = -3 / 11 := 
sorry

end NUMINAMATH_GPT_dampening_factor_l2067_206745


namespace NUMINAMATH_GPT_intersection_of_sets_l2067_206713

def setA : Set ℝ := {x | (x - 2) / x ≤ 0}
def setB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def setC : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets : setA ∩ setB = setC :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l2067_206713


namespace NUMINAMATH_GPT_arithmetical_puzzle_l2067_206758

theorem arithmetical_puzzle (S I X T W E N : ℕ) 
  (h1 : S = 1) 
  (h2 : N % 2 = 0) 
  (h3 : (1 * 100 + I * 10 + X) * 3 = T * 1000 + W * 100 + E * 10 + N) 
  (h4 : ∀ (a b c d e f : ℕ), 
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f) :
  T = 5 := sorry

end NUMINAMATH_GPT_arithmetical_puzzle_l2067_206758


namespace NUMINAMATH_GPT_supplementary_angle_l2067_206789

theorem supplementary_angle {α β : ℝ} (angle_supplementary : α + β = 180) (angle_1_eq : α = 80) : β = 100 :=
by
  sorry

end NUMINAMATH_GPT_supplementary_angle_l2067_206789


namespace NUMINAMATH_GPT_solution_set_for_inequality_l2067_206736

def f (x : ℝ) : ℝ := x^3 + x

theorem solution_set_for_inequality {a : ℝ} (h : -2 < a ∧ a < 2) :
  f a + f (a^2 - 2) < 0 ↔ -2 < a ∧ a < 0 ∨ 0 < a ∧ a < 1 := sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l2067_206736


namespace NUMINAMATH_GPT_true_converses_count_l2067_206740

-- Definitions according to the conditions
def parallel_lines (L1 L2 : Prop) : Prop := L1 ↔ L2
def congruent_triangles (T1 T2 : Prop) : Prop := T1 ↔ T2
def vertical_angles (A1 A2 : Prop) : Prop := A1 = A2
def squares_equal (m n : ℝ) : Prop := m = n → (m^2 = n^2)

-- Propositions with their converses
def converse_parallel (L1 L2 : Prop) : Prop := parallel_lines L1 L2 → parallel_lines L2 L1
def converse_congruent (T1 T2 : Prop) : Prop := congruent_triangles T1 T2 → congruent_triangles T2 T1
def converse_vertical (A1 A2 : Prop) : Prop := vertical_angles A1 A2 → vertical_angles A2 A1
def converse_squares (m n : ℝ) : Prop := (m^2 = n^2) → (m = n)

-- Proving the number of true converses
theorem true_converses_count : 
  (∃ L1 L2, converse_parallel L1 L2) →
  (∃ T1 T2, ¬converse_congruent T1 T2) →
  (∃ A1 A2, converse_vertical A1 A2) →
  (∃ m n : ℝ, ¬converse_squares m n) →
  (2 = 2) := by
  intros _ _ _ _
  sorry

end NUMINAMATH_GPT_true_converses_count_l2067_206740


namespace NUMINAMATH_GPT_molecular_weight_K2Cr2O7_l2067_206775

/--
K2Cr2O7 consists of:
- 2 K atoms
- 2 Cr atoms
- 7 O atoms

Atomic weights:
- K: 39.10 g/mol
- Cr: 52.00 g/mol
- O: 16.00 g/mol

We need to prove that the molecular weight of 4 moles of K2Cr2O7 is 1176.80 g/mol.
-/
theorem molecular_weight_K2Cr2O7 :
  let weight_K := 39.10
  let weight_Cr := 52.00
  let weight_O := 16.00
  let mol_weight_K2Cr2O7 := (2 * weight_K) + (2 * weight_Cr) + (7 * weight_O)
  (4 * mol_weight_K2Cr2O7) = 1176.80 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_K2Cr2O7_l2067_206775


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2067_206766

def a (n : ℕ) : ℕ := 3 * (2 ^ (n - 1))

theorem geometric_sequence_sum :
  a 1 = 3 → a 4 = 24 → (a 3 + a 4 + a 5) = 84 :=
by
  intros h1 h4
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2067_206766


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l2067_206762

-- Proof Problem 1
theorem problem1 : -12 - (-18) + (-7) = -1 := 
by {
  sorry
}

-- Proof Problem 2
theorem problem2 : ((4 / 7) - (1 / 9) + (2 / 21)) * (-63) = -35 := 
by {
  sorry
}

-- Proof Problem 3
theorem problem3 : ((-4) ^ 2) / 2 + 9 * (-1 / 3) - abs (3 - 4) = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem1_problem2_problem3_l2067_206762


namespace NUMINAMATH_GPT_original_number_l2067_206781

theorem original_number (x : ℝ) (hx : 1000 * x = 9 * (1 / x)) : 
  x = 3 * (Real.sqrt 10) / 100 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l2067_206781


namespace NUMINAMATH_GPT_equation_has_roots_l2067_206748

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end NUMINAMATH_GPT_equation_has_roots_l2067_206748


namespace NUMINAMATH_GPT_hari_contribution_l2067_206790

theorem hari_contribution 
    (P_investment : ℕ) (P_time : ℕ) (H_time : ℕ) (profit_ratio : ℚ)
    (investment_ratio : P_investment * P_time / (Hari_contribution * H_time) = profit_ratio) :
    Hari_contribution = 10080 :=
by
    have P_investment := 3920
    have P_time := 12
    have H_time := 7
    have profit_ratio := (2 : ℚ) / 3
    sorry

end NUMINAMATH_GPT_hari_contribution_l2067_206790


namespace NUMINAMATH_GPT_ratio_area_of_rectangle_to_square_l2067_206787

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_ratio_area_of_rectangle_to_square_l2067_206787


namespace NUMINAMATH_GPT_alex_cell_phone_cost_l2067_206796

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.1
def extra_min_cost_per_minute : ℝ := 0.15
def text_messages_sent : ℕ := 150
def hours_talked : ℝ := 32
def included_hours : ℝ := 25

theorem alex_cell_phone_cost : base_cost 
  + (text_messages_sent * text_cost_per_message)
  + ((hours_talked - included_hours) * 60 * extra_min_cost_per_minute) = 98 := by
  sorry

end NUMINAMATH_GPT_alex_cell_phone_cost_l2067_206796


namespace NUMINAMATH_GPT_johnny_hourly_wage_l2067_206786

-- Definitions based on conditions
def hours_worked : ℕ := 6
def total_earnings : ℝ := 28.5

-- Theorem statement
theorem johnny_hourly_wage : total_earnings / hours_worked = 4.75 :=
by
  sorry

end NUMINAMATH_GPT_johnny_hourly_wage_l2067_206786


namespace NUMINAMATH_GPT_solve_problem_l2067_206765

open Nat

theorem solve_problem :
  ∃ (n p : ℕ), p.Prime ∧ n > 0 ∧ ∃ k : ℤ, p^2 + 7^n = k^2 ∧ (n, p) = (1, 3) := 
by
  sorry

end NUMINAMATH_GPT_solve_problem_l2067_206765


namespace NUMINAMATH_GPT_ratio_angela_jacob_l2067_206754

-- Definitions for the conditions
def deans_insects := 30
def jacobs_insects := 5 * deans_insects
def angelas_insects := 75

-- The proof statement proving the ratio
theorem ratio_angela_jacob : angelas_insects / jacobs_insects = 1 / 2 :=
by
  -- Sorry is used here to indicate that the proof is skipped
  sorry

end NUMINAMATH_GPT_ratio_angela_jacob_l2067_206754


namespace NUMINAMATH_GPT_shirts_sold_l2067_206771

theorem shirts_sold (initial final : ℕ) (h : initial = 49) (h1 : final = 28) : initial - final = 21 :=
sorry

end NUMINAMATH_GPT_shirts_sold_l2067_206771


namespace NUMINAMATH_GPT_base_equivalence_l2067_206752

theorem base_equivalence : 
  ∀ (b : ℕ), (b^3 + 3*b^2 + 4)^2 = 9*b^4 + 9*b^3 + 2*b^2 + 2*b + 5 ↔ b = 10 := 
by
  sorry

end NUMINAMATH_GPT_base_equivalence_l2067_206752


namespace NUMINAMATH_GPT_longest_side_of_triangle_l2067_206788

theorem longest_side_of_triangle :
  ∀ (A B C a b : ℝ),
    B = 2 * π / 3 →
    C = π / 6 →
    a = 5 →
    A = π - B - C →
    (b / (Real.sin B) = a / (Real.sin A)) →
    b = 5 * Real.sqrt 3 :=
by
  intros A B C a b hB hC ha hA h_sine_ratio
  sorry

end NUMINAMATH_GPT_longest_side_of_triangle_l2067_206788


namespace NUMINAMATH_GPT_domain_of_expression_l2067_206759

theorem domain_of_expression (x : ℝ) 
  (h1 : 3 * x - 6 ≥ 0) 
  (h2 : 7 - 2 * x ≥ 0) 
  (h3 : 7 - 2 * x > 0) : 
  2 ≤ x ∧ x < 7 / 2 := by
sorry

end NUMINAMATH_GPT_domain_of_expression_l2067_206759


namespace NUMINAMATH_GPT_fraction_of_original_water_after_four_replacements_l2067_206733

-- Define the initial condition and process
def initial_water_volume : ℚ := 10
def initial_alcohol_volume : ℚ := 10
def initial_total_volume : ℚ := initial_water_volume + initial_alcohol_volume

def fraction_remaining_after_removal (fraction_remaining : ℚ) : ℚ :=
  fraction_remaining * (initial_total_volume - 5) / initial_total_volume

-- Define the function counting the iterations process
def fraction_after_replacements (n : ℕ) (fraction_remaining : ℚ) : ℚ :=
  Nat.iterate fraction_remaining_after_removal n fraction_remaining

-- We have 4 replacements, start with 1 (because initially half of tank is water, 
-- fraction is 1 means we start with all original water)
def fraction_of_original_water_remaining : ℚ := (fraction_after_replacements 4 1)

-- Our goal in proof form
theorem fraction_of_original_water_after_four_replacements :
  fraction_of_original_water_remaining = (81 / 256) := by
  sorry

end NUMINAMATH_GPT_fraction_of_original_water_after_four_replacements_l2067_206733


namespace NUMINAMATH_GPT_domain_of_lg_abs_x_minus_1_l2067_206719

theorem domain_of_lg_abs_x_minus_1 (x : ℝ) : 
  (|x| - 1 > 0) ↔ (x < -1 ∨ x > 1) := 
by
  sorry

end NUMINAMATH_GPT_domain_of_lg_abs_x_minus_1_l2067_206719


namespace NUMINAMATH_GPT_fraction_cubed_sum_l2067_206744

theorem fraction_cubed_sum (x y : ℤ) (h1 : x = 3) (h2 : y = 4) :
  (x^3 + 3 * y^3) / 7 = 31 + 3 / 7 := by
  sorry

end NUMINAMATH_GPT_fraction_cubed_sum_l2067_206744


namespace NUMINAMATH_GPT_correct_calculation_l2067_206747

variable {a b : ℝ}

theorem correct_calculation : 
  (2 * a^3 + 2 * a ≠ 2 * a^4) ∧
  ((a - 2 * b)^2 ≠ a^2 - 4 * b^2) ∧
  (-5 * (2 * a - b) ≠ -10 * a - 5 * b) ∧
  ((-2 * a^2 * b)^3 = -8 * a^6 * b^3) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2067_206747


namespace NUMINAMATH_GPT_mary_more_than_marco_l2067_206717

def marco_initial : ℕ := 24
def mary_initial : ℕ := 15
def half_marco : ℕ := marco_initial / 2
def mary_after_give : ℕ := mary_initial + half_marco
def mary_after_spend : ℕ := mary_after_give - 5
def marco_final : ℕ := marco_initial - half_marco

theorem mary_more_than_marco :
  mary_after_spend - marco_final = 10 := by
  sorry

end NUMINAMATH_GPT_mary_more_than_marco_l2067_206717


namespace NUMINAMATH_GPT_probability_without_replacement_probability_with_replacement_l2067_206742

-- Definition for without replacement context
def without_replacement_total_outcomes : ℕ := 6
def without_replacement_favorable_outcomes : ℕ := 3
def without_replacement_prob : ℚ :=
  without_replacement_favorable_outcomes / without_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers without replacement is 1/2
theorem probability_without_replacement : 
  without_replacement_prob = 1 / 2 := by
  sorry

-- Definition for with replacement context
def with_replacement_total_outcomes : ℕ := 16
def with_replacement_favorable_outcomes : ℕ := 3
def with_replacement_prob : ℚ :=
  with_replacement_favorable_outcomes / with_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers with replacement is 3/16
theorem probability_with_replacement : 
  with_replacement_prob = 3 / 16 := by
  sorry

end NUMINAMATH_GPT_probability_without_replacement_probability_with_replacement_l2067_206742


namespace NUMINAMATH_GPT_max_ab_condition_max_ab_value_l2067_206795

theorem max_ab_condition (a b : ℝ) (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : ab ≤ 1 / 4 :=
sorry

theorem max_ab_value (a b : ℝ) (h1 : a + b = 1) (h2 : a = b) : ab = 1 / 4 :=
sorry

end NUMINAMATH_GPT_max_ab_condition_max_ab_value_l2067_206795


namespace NUMINAMATH_GPT_jordan_buys_rice_l2067_206785

variables (r l : ℝ)

theorem jordan_buys_rice
  (price_rice : ℝ := 1.20)
  (price_lentils : ℝ := 0.60)
  (total_pounds : ℝ := 30)
  (total_cost : ℝ := 27.00)
  (eq1 : r + l = total_pounds)
  (eq2 : price_rice * r + price_lentils * l = total_cost) :
  r = 15.0 :=
by
  sorry

end NUMINAMATH_GPT_jordan_buys_rice_l2067_206785


namespace NUMINAMATH_GPT_min_value_sequence_l2067_206791

theorem min_value_sequence (a : ℕ → ℕ) (h1 : a 2 = 102) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) - a n = 4 * n) : 
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (a m) / m ≥ 26) :=
sorry

end NUMINAMATH_GPT_min_value_sequence_l2067_206791


namespace NUMINAMATH_GPT_unique_cube_coloring_l2067_206712

-- Definition of vertices at the bottom of the cube with specific colors
inductive Color 
| Red | Green | Blue | Purple

open Color

def bottom_colors : Fin 4 → Color
| 0 => Red
| 1 => Green
| 2 => Blue
| 3 => Purple

-- Definition of the property that ensures each face of the cube has different colored corners
def all_faces_different_colors (top_colors : Fin 4 → Color) : Prop :=
  (top_colors 0 ≠ Red) ∧ (top_colors 0 ≠ Green) ∧ (top_colors 0 ≠ Blue) ∧
  (top_colors 1 ≠ Green) ∧ (top_colors 1 ≠ Blue) ∧ (top_colors 1 ≠ Purple) ∧
  (top_colors 2 ≠ Red) ∧ (top_colors 2 ≠ Blue) ∧ (top_colors 2 ≠ Purple) ∧
  (top_colors 3 ≠ Red) ∧ (top_colors 3 ≠ Green) ∧ (top_colors 3 ≠ Purple)

-- Prove there is exactly one way to achieve this coloring of the top corners
theorem unique_cube_coloring : ∃! (top_colors : Fin 4 → Color), all_faces_different_colors top_colors :=
sorry

end NUMINAMATH_GPT_unique_cube_coloring_l2067_206712


namespace NUMINAMATH_GPT_determine_a_b_l2067_206726

-- Step d) The Lean 4 statement for the transformed problem
theorem determine_a_b (a b : ℝ) (h : ∀ t : ℝ, (t^2 + t + 1) * 1^2 - 2 * (a + t)^2 * 1 + t^2 + 3 * a * t + b = 0) : 
  a = 1 ∧ b = 1 := 
sorry

end NUMINAMATH_GPT_determine_a_b_l2067_206726


namespace NUMINAMATH_GPT_arithmetic_sequence_a7_l2067_206724

theorem arithmetic_sequence_a7 (S_13 : ℕ → ℕ → ℕ) (n : ℕ) (a7 : ℕ) (h1: S_13 13 52 = 52) (h2: S_13 13 a7 = 13 * a7):
  a7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a7_l2067_206724


namespace NUMINAMATH_GPT_new_person_weight_l2067_206769

theorem new_person_weight (avg_weight_increase : ℝ) (old_weight new_weight : ℝ) (n : ℕ)
    (weight_increase_per_person : avg_weight_increase = 3.5)
    (number_of_persons : n = 8)
    (replaced_person_weight : old_weight = 62) :
    new_weight = 90 :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l2067_206769


namespace NUMINAMATH_GPT_problem_l2067_206768

variable {x y : ℝ}

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 10) : (x - y)^2 = 41 := 
by
  sorry

end NUMINAMATH_GPT_problem_l2067_206768


namespace NUMINAMATH_GPT_tangent_line_through_point_l2067_206729

theorem tangent_line_through_point (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) : 
  (∃ k : ℝ, 15 * x - 8 * y - 13 = 0) ∨ x = 3 := sorry

end NUMINAMATH_GPT_tangent_line_through_point_l2067_206729


namespace NUMINAMATH_GPT_probability_of_two_jacob_one_isaac_l2067_206757

-- Definition of the problem conditions
def jacob_letters := 5
def isaac_letters := 5
def total_cards := 12
def cards_drawn := 3

-- Combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probability calculation
def probability_two_jacob_one_isaac : ℚ :=
  (C jacob_letters 2 * C isaac_letters 1 : ℚ) / (C total_cards cards_drawn : ℚ)

-- The statement of the problem
theorem probability_of_two_jacob_one_isaac :
  probability_two_jacob_one_isaac = 5 / 22 :=
  by sorry

end NUMINAMATH_GPT_probability_of_two_jacob_one_isaac_l2067_206757


namespace NUMINAMATH_GPT_minimum_a_inequality_l2067_206714

variable {x y : ℝ}

/-- The inequality (x + y) * (1/x + a/y) ≥ 9 holds for any positive real numbers x and y 
     if and only if a ≥ 4.  -/
theorem minimum_a_inequality (a : ℝ) (h : ∀ (x y : ℝ), 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  a ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_a_inequality_l2067_206714


namespace NUMINAMATH_GPT_negation_of_exists_l2067_206760

theorem negation_of_exists (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x < 0) ↔ ¬ (∀ x : ℝ, x^2 + 2 * x >= 0) :=
sorry

end NUMINAMATH_GPT_negation_of_exists_l2067_206760


namespace NUMINAMATH_GPT_find_x_ineq_solution_l2067_206710

open Set

theorem find_x_ineq_solution :
  {x : ℝ | (x - 2) / (x - 4) ≥ 3} = Ioc 4 5 := 
sorry

end NUMINAMATH_GPT_find_x_ineq_solution_l2067_206710


namespace NUMINAMATH_GPT_work_together_10_days_l2067_206702

noncomputable def rate_A (W : ℝ) : ℝ := W / 20
noncomputable def rate_B (W : ℝ) : ℝ := W / 20

theorem work_together_10_days (W : ℝ) (hW : W > 0) :
  let A := rate_A W
  let B := rate_B W
  let combined_rate := A + B
  W / combined_rate = 10 :=
by
  sorry

end NUMINAMATH_GPT_work_together_10_days_l2067_206702


namespace NUMINAMATH_GPT_arithmetic_mean_of_pq_is_10_l2067_206794

variable (p q r : ℝ)

theorem arithmetic_mean_of_pq_is_10
  (H_mean_qr : (q + r) / 2 = 20)
  (H_r_minus_p : r - p = 20) :
  (p + q) / 2 = 10 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_pq_is_10_l2067_206794


namespace NUMINAMATH_GPT_infinite_primes_dividing_S_l2067_206767

noncomputable def infinite_set_of_pos_integers (S : Set ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ m ∈ S) ∧ ∀ n : ℕ, n ∈ S → n > 0

def set_of_sums (S : Set ℕ) : Set ℕ :=
  {t | ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ t = x + y}

noncomputable def finitely_many_primes_condition (S : Set ℕ) (T : Set ℕ) : Prop :=
  {p : ℕ | Prime p ∧ p % 4 = 1 ∧ (∃ t ∈ T, p ∣ t)}.Finite

theorem infinite_primes_dividing_S (S : Set ℕ) (T := set_of_sums S)
  (hS : infinite_set_of_pos_integers S)
  (hT : finitely_many_primes_condition S T) :
  {p : ℕ | Prime p ∧ ∃ s ∈ S, p ∣ s}.Infinite := 
sorry

end NUMINAMATH_GPT_infinite_primes_dividing_S_l2067_206767


namespace NUMINAMATH_GPT_sarahs_total_problems_l2067_206784

def math_pages : ℕ := 4
def reading_pages : ℕ := 6
def science_pages : ℕ := 5
def math_problems_per_page : ℕ := 4
def reading_problems_per_page : ℕ := 4
def science_problems_per_page : ℕ := 6

def total_math_problems : ℕ := math_pages * math_problems_per_page
def total_reading_problems : ℕ := reading_pages * reading_problems_per_page
def total_science_problems : ℕ := science_pages * science_problems_per_page

def total_problems : ℕ := total_math_problems + total_reading_problems + total_science_problems

theorem sarahs_total_problems :
  total_problems = 70 :=
by
  -- proof will be inserted here
  sorry

end NUMINAMATH_GPT_sarahs_total_problems_l2067_206784


namespace NUMINAMATH_GPT_marigolds_sold_second_day_l2067_206743

theorem marigolds_sold_second_day (x : ℕ) (h1 : 14 ≤ x)
  (h2 : 2 * x + 14 + x = 89) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_marigolds_sold_second_day_l2067_206743


namespace NUMINAMATH_GPT_sum_of_midpoints_l2067_206731

theorem sum_of_midpoints {a b c : ℝ} (h : a + b + c = 10) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_l2067_206731


namespace NUMINAMATH_GPT_balls_into_boxes_l2067_206799

noncomputable def countDistributions : ℕ :=
  sorry

theorem balls_into_boxes :
  countDistributions = 8 :=
  sorry

end NUMINAMATH_GPT_balls_into_boxes_l2067_206799


namespace NUMINAMATH_GPT_tan_half_angle_sin_cos_expression_l2067_206770

-- Proof Problem 1: If α is an angle in the third quadrant and sin α = -5/13, then tan (α / 2) = -5.
theorem tan_half_angle (α : ℝ) (h1 : Real.sin α = -5/13) (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -5 := 
by 
  sorry

-- Proof Problem 2: If tan α = 2, then sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5.
theorem sin_cos_expression (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_tan_half_angle_sin_cos_expression_l2067_206770


namespace NUMINAMATH_GPT_induction_proof_l2067_206707

open Nat

noncomputable def S (n : ℕ) : ℚ :=
  match n with
  | 0     => 0
  | (n+1) => S n + 1 / ((n+1) * (n+2))

theorem induction_proof : ∀ n : ℕ, S n = n / (n + 1) := by
  intro n
  induction n with
  | zero => 
    -- Base case: S(1) = 1/2
    sorry
  | succ n ih =>
    -- Induction step: Assume S(n) = n / (n + 1), prove S(n+1) = (n+1) / (n+2)
    sorry

end NUMINAMATH_GPT_induction_proof_l2067_206707


namespace NUMINAMATH_GPT_f_odd_function_l2067_206738

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (a b : ℝ) : f (a + b) = f a + f b

theorem f_odd_function : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  sorry

end NUMINAMATH_GPT_f_odd_function_l2067_206738


namespace NUMINAMATH_GPT_no_three_times_age_ago_l2067_206727

theorem no_three_times_age_ago (F D : ℕ) (h₁ : F = 40) (h₂ : D = 40) (h₃ : F = 2 * D) :
  ¬ ∃ x, F - x = 3 * (D - x) :=
by
  sorry

end NUMINAMATH_GPT_no_three_times_age_ago_l2067_206727


namespace NUMINAMATH_GPT_ricardo_coin_difference_l2067_206793

theorem ricardo_coin_difference (p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ 2299) :
  (11500 - 4 * p) - (11500 - 4 * (2300 - p)) = 9192 :=
by
  sorry

end NUMINAMATH_GPT_ricardo_coin_difference_l2067_206793


namespace NUMINAMATH_GPT_first_year_with_sum_of_digits_10_after_2020_l2067_206792

theorem first_year_with_sum_of_digits_10_after_2020 :
  ∃ (y : ℕ), y > 2020 ∧ (y.digits 10).sum = 10 ∧ ∀ (z : ℕ), (z > 2020 ∧ (z.digits 10).sum = 10) → y ≤ z :=
sorry

end NUMINAMATH_GPT_first_year_with_sum_of_digits_10_after_2020_l2067_206792


namespace NUMINAMATH_GPT_product_of_positive_integer_solutions_l2067_206721

theorem product_of_positive_integer_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (n^2 - 47 * n + 660 = p) → False :=
by
  -- Placeholder for proof, based on the problem conditions.
  sorry

end NUMINAMATH_GPT_product_of_positive_integer_solutions_l2067_206721


namespace NUMINAMATH_GPT_smallest_k_condition_exists_l2067_206723

theorem smallest_k_condition_exists (k : ℕ) :
    k > 1 ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 3 = 1) → k = 313 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_condition_exists_l2067_206723


namespace NUMINAMATH_GPT_total_people_going_to_zoo_l2067_206739

def cars : ℝ := 3.0
def people_per_car : ℝ := 63.0

theorem total_people_going_to_zoo : cars * people_per_car = 189.0 :=
by 
  sorry

end NUMINAMATH_GPT_total_people_going_to_zoo_l2067_206739


namespace NUMINAMATH_GPT_amount_paid_is_200_l2067_206711

-- Definitions of the costs and change received
def cost_of_pants := 140
def cost_of_shirt := 43
def cost_of_tie := 15
def change_received := 2

-- Total cost calculation
def total_cost := cost_of_pants + cost_of_shirt + cost_of_tie

-- Lean proof statement
theorem amount_paid_is_200 : total_cost + change_received = 200 := by
  -- Definitions ensure the total cost and change received are used directly from conditions
  sorry

end NUMINAMATH_GPT_amount_paid_is_200_l2067_206711


namespace NUMINAMATH_GPT_part1_part2_l2067_206732

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x - 1| + |2 * x - a|

theorem part1 (x : ℝ) : (f x 2 < 2) ↔ (1/4 < x ∧ x < 5/4) := by
  sorry
  
theorem part2 (a : ℝ) (hx : ∀ x : ℝ, f x a ≥ 3 * a + 2) :
  (-3/2 ≤ a ∧ a ≤ -1/4) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2067_206732


namespace NUMINAMATH_GPT_compute_expr_l2067_206741

theorem compute_expr :
  ((π - 3.14)^0 + (-0.125)^2008 * 8^2008) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_compute_expr_l2067_206741


namespace NUMINAMATH_GPT_find_w_l2067_206780

variables (w x y z : ℕ)

-- conditions
def condition1 : Prop := x = w / 2
def condition2 : Prop := y = w + x
def condition3 : Prop := z = 400
def condition4 : Prop := w + x + y + z = 1000

-- problem to prove
theorem find_w (h1 : condition1 w x) (h2 : condition2 w x y) (h3 : condition3 z) (h4 : condition4 w x y z) : w = 200 :=
by sorry

end NUMINAMATH_GPT_find_w_l2067_206780


namespace NUMINAMATH_GPT_decreasing_interval_l2067_206737

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 2

theorem decreasing_interval : ∀ x : ℝ, (-2 < x ∧ x < 0) → (deriv f x < 0) := 
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_l2067_206737


namespace NUMINAMATH_GPT_weak_multiple_l2067_206730

def is_weak (a b n : ℕ) : Prop :=
  ∀ (x y : ℕ), n ≠ a * x + b * y

theorem weak_multiple (a b n : ℕ) (h_coprime : Nat.gcd a b = 1) (h_weak : is_weak a b n) (h_bound : n < a * b / 6) : 
  ∃ k ≥ 2, is_weak a b (k * n) :=
by
  sorry

end NUMINAMATH_GPT_weak_multiple_l2067_206730


namespace NUMINAMATH_GPT_find_a_l2067_206756

open Complex

theorem find_a (a : ℝ) (h : (2 + Complex.I * a) / (1 + Complex.I * Real.sqrt 2) = -Complex.I * Real.sqrt 2) :
  a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l2067_206756


namespace NUMINAMATH_GPT_keys_missing_l2067_206797

theorem keys_missing (vowels := 5) (consonants := 21)
  (missing_consonants := consonants / 7) (missing_vowels := 2) :
  missing_consonants + missing_vowels = 5 := by
  sorry

end NUMINAMATH_GPT_keys_missing_l2067_206797


namespace NUMINAMATH_GPT_quadratic_relationship_l2067_206763

theorem quadratic_relationship :
  ∀ (x z : ℕ), (x = 1 ∧ z = 5) ∨ (x = 2 ∧ z = 12) ∨ (x = 3 ∧ z = 23) ∨ (x = 4 ∧ z = 38) ∨ (x = 5 ∧ z = 57) →
  z = 2 * x^2 + x + 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_relationship_l2067_206763


namespace NUMINAMATH_GPT_find_down_payment_l2067_206783

noncomputable def purchasePrice : ℝ := 118
noncomputable def monthlyPayment : ℝ := 10
noncomputable def numberOfMonths : ℝ := 12
noncomputable def interestRate : ℝ := 0.15254237288135593
noncomputable def totalPayments : ℝ := numberOfMonths * monthlyPayment -- total amount paid through installments
noncomputable def interestPaid : ℝ := purchasePrice * interestRate -- total interest paid
noncomputable def totalPaid : ℝ := purchasePrice + interestPaid -- total amount paid including interest

theorem find_down_payment : ∃ D : ℝ, D + totalPayments = totalPaid ∧ D = 16 :=
by sorry

end NUMINAMATH_GPT_find_down_payment_l2067_206783


namespace NUMINAMATH_GPT_ratio_of_side_lengths_sum_l2067_206749

theorem ratio_of_side_lengths_sum (a b c : ℕ) (ha : a = 4) (hb : b = 15) (hc : c = 25) :
  a + b + c = 44 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_side_lengths_sum_l2067_206749


namespace NUMINAMATH_GPT_angle_BAC_l2067_206701

theorem angle_BAC
  (elevation_angle_B_from_A : ℝ)
  (depression_angle_C_from_A : ℝ)
  (h₁ : elevation_angle_B_from_A = 60)
  (h₂ : depression_angle_C_from_A = 70) :
  elevation_angle_B_from_A + depression_angle_C_from_A = 130 :=
by
  sorry

end NUMINAMATH_GPT_angle_BAC_l2067_206701


namespace NUMINAMATH_GPT_power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l2067_206709

-- Definitions and conditions
def is_power_of_2 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

def is_sum_of_two_or_more_consecutive_naturals (n : ℕ) : Prop :=
  ∃ (a k : ℕ), k ≥ 2 ∧ n = (k * a) + (k * (k - 1)) / 2

-- Proofs to be stated
theorem power_of_2_not_sum_of_consecutive (n : ℕ) (h : is_power_of_2 n) : ¬ is_sum_of_two_or_more_consecutive_naturals n :=
by
    sorry

theorem not_power_of_2_is_sum_of_consecutive (M : ℕ) (h : ¬ is_power_of_2 M) : is_sum_of_two_or_more_consecutive_naturals M :=
by
    sorry

end NUMINAMATH_GPT_power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l2067_206709


namespace NUMINAMATH_GPT_problem1_arithmetic_sequence_problem2_geometric_sequence_l2067_206735

-- Problem (1)
variable (S : Nat → Int)
variable (a : Nat → Int)

axiom S10_eq_50 : S 10 = 50
axiom S20_eq_300 : S 20 = 300
axiom S_def : (∀ n : Nat, n > 0 → S n = n * a 1 + (n * (n-1) / 2) * (a 2 - a 1))

theorem problem1_arithmetic_sequence (n : Nat) : a n = 2 * n - 6 := sorry

-- Problem (2)
variable (a : Nat → Int)

axiom S3_eq_a2_plus_10a1 : S 3 = a 2 + 10 * a 1
axiom a5_eq_81 : a 5 = 81
axiom positive_terms : ∀ n, a n > 0

theorem problem2_geometric_sequence (n : Nat) : S n = (3 ^ n - 1) / 2 := sorry

end NUMINAMATH_GPT_problem1_arithmetic_sequence_problem2_geometric_sequence_l2067_206735


namespace NUMINAMATH_GPT_bernoulli_inequality_gt_bernoulli_inequality_lt_l2067_206750

theorem bernoulli_inequality_gt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : x > 1 ∨ x < 0) : (1 + h)^x > 1 + h * x := sorry

theorem bernoulli_inequality_lt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : 0 < x) (hx3 : x < 1) : (1 + h)^x < 1 + h * x := sorry

end NUMINAMATH_GPT_bernoulli_inequality_gt_bernoulli_inequality_lt_l2067_206750


namespace NUMINAMATH_GPT_radius_first_field_l2067_206773

theorem radius_first_field (r_2 : ℝ) (h_r2 : r_2 = 10) (h_area : ∃ A_2, ∃ A_1, A_1 = 0.09 * A_2 ∧ A_2 = π * r_2^2) : ∃ r_1 : ℝ, r_1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_first_field_l2067_206773


namespace NUMINAMATH_GPT_sin_beta_value_l2067_206753

open Real

theorem sin_beta_value (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h1 : sin α = 5 / 13) 
  (h2 : cos (α + β) = -4 / 5) : 
  sin β = 56 / 65 := 
sorry

end NUMINAMATH_GPT_sin_beta_value_l2067_206753


namespace NUMINAMATH_GPT_simplify_fraction_l2067_206725

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2067_206725


namespace NUMINAMATH_GPT_how_many_integers_satisfy_l2067_206734

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end NUMINAMATH_GPT_how_many_integers_satisfy_l2067_206734


namespace NUMINAMATH_GPT_cos_product_triangle_l2067_206703

theorem cos_product_triangle (A B C : ℝ) (h : A + B + C = π) (hA : A > 0) (hB : B > 0) (hC : C > 0) : 
  Real.cos A * Real.cos B * Real.cos C ≤ 1 / 8 := 
sorry

end NUMINAMATH_GPT_cos_product_triangle_l2067_206703


namespace NUMINAMATH_GPT_combined_height_is_9_l2067_206706

def barrys_reach : ℝ := 5 -- Barry can reach apples that are 5 feet high

def larrys_full_height : ℝ := 5 -- Larry's full height is 5 feet

def larrys_shoulder_height : ℝ := larrys_full_height * 0.8 -- Larry's shoulder height is 20% less than his full height

def combined_reach (b_reach : ℝ) (l_shoulder : ℝ) : ℝ := b_reach + l_shoulder

theorem combined_height_is_9 : combined_reach barrys_reach larrys_shoulder_height = 9 := by
  sorry

end NUMINAMATH_GPT_combined_height_is_9_l2067_206706


namespace NUMINAMATH_GPT_vertex_on_x_axis_l2067_206715

theorem vertex_on_x_axis (d : ℝ) : 
  (∃ x : ℝ, x^2 - 6 * x + d = 0) ↔ d = 9 :=
by
  sorry

end NUMINAMATH_GPT_vertex_on_x_axis_l2067_206715


namespace NUMINAMATH_GPT_problem_inequality_l2067_206716

open Real

theorem problem_inequality 
  (p q r x y theta: ℝ) :
  p * x ^ (q - y) + q * x ^ (r - y) + r * x ^ (y - theta)  ≥ p + q + r :=
sorry

end NUMINAMATH_GPT_problem_inequality_l2067_206716


namespace NUMINAMATH_GPT_complex_number_condition_l2067_206746

theorem complex_number_condition (b : ℝ) :
  (2 + b) / 5 = (2 * b - 1) / 5 → b = 3 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_condition_l2067_206746


namespace NUMINAMATH_GPT_probability_cs_majors_consecutive_l2067_206774

def total_ways_to_choose_5_out_of_12 : ℕ :=
  Nat.choose 12 5

def number_of_ways_cs_majors_consecutive : ℕ :=
  12

theorem probability_cs_majors_consecutive :
  (number_of_ways_cs_majors_consecutive : ℚ) / (total_ways_to_choose_5_out_of_12 : ℚ) = 1 / 66 := by
  sorry

end NUMINAMATH_GPT_probability_cs_majors_consecutive_l2067_206774


namespace NUMINAMATH_GPT_plane_intersects_unit_cubes_l2067_206720

def unitCubeCount (side_length : ℕ) : ℕ :=
  side_length ^ 3

def intersectionCount (num_unitCubes : ℕ) (side_length : ℕ) : ℕ :=
  if side_length = 4 then 32 else 0 -- intersection count only applies for side_length = 4

theorem plane_intersects_unit_cubes
  (side_length : ℕ)
  (num_unitCubes : ℕ)
  (cubeArrangement : num_unitCubes = unitCubeCount side_length)
  (planeCondition : True) -- the plane is perpendicular to the diagonal and bisects it
  : intersectionCount num_unitCubes side_length = 32 := by
  sorry

end NUMINAMATH_GPT_plane_intersects_unit_cubes_l2067_206720


namespace NUMINAMATH_GPT_average_age_of_cricket_team_l2067_206778

theorem average_age_of_cricket_team :
  let captain_age := 28
  let ages_sum := 28 + (28 + 4) + (28 - 2) + (28 + 6)
  let remaining_players := 15 - 4
  let total_sum := ages_sum + remaining_players * (A - 1)
  let total_players := 15
  total_sum / total_players = 27.25 := 
by 
  sorry

end NUMINAMATH_GPT_average_age_of_cricket_team_l2067_206778


namespace NUMINAMATH_GPT_Nancy_antacid_consumption_l2067_206751

theorem Nancy_antacid_consumption :
  let antacids_per_month : ℕ :=
    let antacids_per_day_indian := 3
    let antacids_per_day_mexican := 2
    let antacids_per_day_other := 1
    let days_indian_per_week := 3
    let days_mexican_per_week := 2
    let days_total_per_week := 7
    let weeks_per_month := 4

    let antacids_per_week_indian := antacids_per_day_indian * days_indian_per_week
    let antacids_per_week_mexican := antacids_per_day_mexican * days_mexican_per_week
    let days_other_per_week := days_total_per_week - days_indian_per_week - days_mexican_per_week
    let antacids_per_week_other := antacids_per_day_other * days_other_per_week

    let antacids_per_week_total := antacids_per_week_indian + antacids_per_week_mexican + antacids_per_week_other

    antacids_per_week_total * weeks_per_month
    
  antacids_per_month = 60 := sorry

end NUMINAMATH_GPT_Nancy_antacid_consumption_l2067_206751


namespace NUMINAMATH_GPT_inequality_solution_set_l2067_206700

theorem inequality_solution_set (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l2067_206700


namespace NUMINAMATH_GPT_Vitya_catches_mother_l2067_206761

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end NUMINAMATH_GPT_Vitya_catches_mother_l2067_206761


namespace NUMINAMATH_GPT_boy_usual_time_l2067_206704

theorem boy_usual_time (R T : ℝ) (h : R * T = (7 / 6) * R * (T - 2)) : T = 14 :=
by
  sorry

end NUMINAMATH_GPT_boy_usual_time_l2067_206704


namespace NUMINAMATH_GPT_retail_women_in_LA_l2067_206764

/-
Los Angeles has 6 million people living in it. If half the population is women 
and 1/3 of the women work in retail, how many women work in retail in Los Angeles?
-/

theorem retail_women_in_LA 
  (total_population : ℕ)
  (half_population_women : total_population / 2 = women_population)
  (third_women_retail : women_population / 3 = retail_women)
  : total_population = 6000000 → retail_women = 1000000 :=
by
  sorry

end NUMINAMATH_GPT_retail_women_in_LA_l2067_206764


namespace NUMINAMATH_GPT_sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l2067_206755

-- Definitions
def a (t : ℤ) := 4 * t
def b (t : ℤ) := 3 - 2 * t - t^2
def c (t : ℤ) := 3 + 2 * t - t^2

-- Theorem for sum of squares
theorem sum_of_squares_twice_square (t : ℤ) : 
  a t ^ 2 + b t ^ 2 + c t ^ 2 = 2 * ((3 + t^2) ^ 2) :=
by 
  sorry

-- Theorem for sum of fourth powers
theorem sum_of_fourth_powers_twice_fourth_power (t : ℤ) : 
  a t ^ 4 + b t ^ 4 + c t ^ 4 = 2 * ((3 + t^2) ^ 4) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l2067_206755


namespace NUMINAMATH_GPT_fraction_of_5100_l2067_206772

theorem fraction_of_5100 (x : ℝ) (h : ((3 / 4) * x * (2 / 5) * 5100 = 765.0000000000001)) : x = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_5100_l2067_206772


namespace NUMINAMATH_GPT_three_distinct_real_roots_l2067_206705

theorem three_distinct_real_roots 
  (c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1*x1 + 6*x1 + c)*(x1*x1 + 6*x1 + c) = 0 ∧ 
    (x2*x2 + 6*x2 + c)*(x2*x2 + 6*x2 + c) = 0 ∧ 
    (x3*x3 + 6*x3 + c)*(x3*x3 + 6*x3 + c) = 0) 
  ↔ c = (11 - Real.sqrt 13) / 2 :=
by sorry

end NUMINAMATH_GPT_three_distinct_real_roots_l2067_206705


namespace NUMINAMATH_GPT_least_number_to_subtract_l2067_206782

theorem least_number_to_subtract (x : ℕ) (h : 5026 % 5 = x) : x = 1 :=
by sorry

end NUMINAMATH_GPT_least_number_to_subtract_l2067_206782


namespace NUMINAMATH_GPT_carnations_count_l2067_206776

-- Define the conditions 
def vase_capacity : Nat := 9
def number_of_vases : Nat := 3
def number_of_roses : Nat := 23
def total_flowers : Nat := number_of_vases * vase_capacity

-- Define the number of carnations
def number_of_carnations : Nat := total_flowers - number_of_roses

-- Assertion that should be proved
theorem carnations_count : number_of_carnations = 4 := by
  sorry

end NUMINAMATH_GPT_carnations_count_l2067_206776


namespace NUMINAMATH_GPT_principal_sum_l2067_206728

theorem principal_sum (A1 A2 : ℝ) (I P : ℝ) 
  (hA1 : A1 = 1717) 
  (hA2 : A2 = 1734) 
  (hI : I = A2 - A1)
  (h_simple_interest : A1 = P + I) : P = 1700 :=
by
  sorry

end NUMINAMATH_GPT_principal_sum_l2067_206728


namespace NUMINAMATH_GPT_total_pigs_correct_l2067_206798

def initial_pigs : Float := 64.0
def incoming_pigs : Float := 86.0
def total_pigs : Float := 150.0

theorem total_pigs_correct : initial_pigs + incoming_pigs = total_pigs := by 
  sorry

end NUMINAMATH_GPT_total_pigs_correct_l2067_206798
