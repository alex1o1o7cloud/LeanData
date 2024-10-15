import Mathlib

namespace NUMINAMATH_GPT_line_l_passes_fixed_point_line_l_perpendicular_value_a_l44_4414

variable (a : ℝ)

def line_l (a : ℝ) : ℝ × ℝ → Prop :=
  λ p => (a + 1) * p.1 + p.2 + 2 - a = 0

def perpendicular_line : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - 3 * p.2 + 4 = 0

theorem line_l_passes_fixed_point :
  line_l a (1, -3) :=
by
  sorry

theorem line_l_perpendicular_value_a (a : ℝ) :
  (∀ p : ℝ × ℝ, perpendicular_line p → line_l a p) → 
  a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_line_l_passes_fixed_point_line_l_perpendicular_value_a_l44_4414


namespace NUMINAMATH_GPT_max_piles_660_stones_l44_4455

-- Define the conditions in Lean
def initial_stones := 660

def valid_pile_sizes (piles : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → a ≤ b → b < 2 * a

-- Define the goal statement in Lean
theorem max_piles_660_stones :
  ∃ (piles : List ℕ), (piles.length = 30) ∧ (piles.sum = initial_stones) ∧ valid_pile_sizes piles :=
sorry

end NUMINAMATH_GPT_max_piles_660_stones_l44_4455


namespace NUMINAMATH_GPT_fat_content_whole_milk_l44_4433

open Real

theorem fat_content_whole_milk :
  ∃ (s w : ℝ), 0 < s ∧ 0 < w ∧
  3 / 100 = 0.75 * s / 100 ∧
  s / 100 = 0.8 * w / 100 ∧
  w = 5 :=
by
  sorry

end NUMINAMATH_GPT_fat_content_whole_milk_l44_4433


namespace NUMINAMATH_GPT_coefficient_and_degree_of_monomial_l44_4472

variable (x y : ℝ)

def monomial : ℝ := -2 * x * y^3

theorem coefficient_and_degree_of_monomial :
  ( ∃ c : ℝ, ∃ d : ℤ, monomial x y = c * x * y^d ∧ c = -2 ∧ d = 4 ) :=
by
  sorry

end NUMINAMATH_GPT_coefficient_and_degree_of_monomial_l44_4472


namespace NUMINAMATH_GPT_draw_at_least_one_even_ball_l44_4463

theorem draw_at_least_one_even_ball:
  -- Let the total number of ordered draws of 4 balls from 15 balls
  let total_draws := 15 * 14 * 13 * 12
  -- Let the total number of ordered draws of 4 balls where all balls are odd (balls 1, 3, ..., 15)
  let odd_draws := 8 * 7 * 6 * 5
  -- The number of valid draws containing at least one even ball
  total_draws - odd_draws = 31080 :=
by
  sorry

end NUMINAMATH_GPT_draw_at_least_one_even_ball_l44_4463


namespace NUMINAMATH_GPT_effective_simple_interest_rate_proof_l44_4460

noncomputable def effective_simple_interest_rate : ℝ :=
  let P := 1
  let r1 := 0.10 / 2 -- Half-yearly rate for year 1
  let t1 := 2 -- number of compounding periods semi-annual
  let A1 := P * (1 + r1) ^ t1

  let r2 := 0.12 / 2 -- Half-yearly rate for year 2
  let t2 := 2
  let A2 := A1 * (1 + r2) ^ t2

  let r3 := 0.14 / 2 -- Half-yearly rate for year 3
  let t3 := 2
  let A3 := A2 * (1 + r3) ^ t3

  let r4 := 0.16 / 2 -- Half-yearly rate for year 4
  let t4 := 2
  let A4 := A3 * (1 + r4) ^ t4

  let CI := 993
  let P_actual := CI / (A4 - P)
  let effective_simple_interest := (CI / P_actual) * 100
  effective_simple_interest

theorem effective_simple_interest_rate_proof :
  effective_simple_interest_rate = 65.48 := by
  sorry

end NUMINAMATH_GPT_effective_simple_interest_rate_proof_l44_4460


namespace NUMINAMATH_GPT_worst_ranking_l44_4420

theorem worst_ranking (teams : Fin 25 → Nat) (A : Fin 25)
  (round_robin : ∀ i j, i ≠ j → teams i + teams j ≤ 4)
  (most_goals : ∀ i, i ≠ A → teams A > teams i)
  (fewest_goals : ∀ i, i ≠ A → teams i > teams A) :
  ∃ ranking : Fin 25 → Fin 25, ranking A = 24 :=
by
  sorry

end NUMINAMATH_GPT_worst_ranking_l44_4420


namespace NUMINAMATH_GPT_find_integer_k_l44_4427

theorem find_integer_k (k x : ℤ) (h : (k^2 - 1) * x^2 - 6 * (3 * k - 1) * x + 72 = 0) (hx : x > 0) :
  k = 1 ∨ k = 2 ∨ k = 3 :=
sorry

end NUMINAMATH_GPT_find_integer_k_l44_4427


namespace NUMINAMATH_GPT_selection_methods_l44_4440

/-- Type definition for the workers -/
inductive Worker
  | PliersOnly  : Worker
  | CarOnly     : Worker
  | Both        : Worker

/-- Conditions -/
def num_workers : ℕ := 11
def num_pliers_only : ℕ := 5
def num_car_only : ℕ := 4
def num_both : ℕ := 2
def pliers_needed : ℕ := 4
def car_needed : ℕ := 4

/-- Main statement -/
theorem selection_methods : 
  (num_pliers_only + num_car_only + num_both = num_workers) → 
  (num_pliers_only = 5) → 
  (num_car_only = 4) → 
  (num_both = 2) → 
  (pliers_needed = 4) → 
  (car_needed = 4) → 
  ∃ n : ℕ, n = 185 := 
by 
  sorry -- Proof Skipped

end NUMINAMATH_GPT_selection_methods_l44_4440


namespace NUMINAMATH_GPT_find_higher_percentage_l44_4493

-- Definitions based on conditions
def principal : ℕ := 8400
def time : ℕ := 2
def rate_0 : ℕ := 10
def delta_interest : ℕ := 840

-- The proof statement
theorem find_higher_percentage (r : ℕ) :
  (principal * rate_0 * time / 100 + delta_interest = principal * r * time / 100) →
  r = 15 :=
by sorry

end NUMINAMATH_GPT_find_higher_percentage_l44_4493


namespace NUMINAMATH_GPT_audrey_not_dreaming_fraction_l44_4406

theorem audrey_not_dreaming_fraction :
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  cycle1_not_dreaming + cycle2_not_dreaming + cycle3_not_dreaming + cycle4_not_dreaming = 227 / 84 :=
by
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  sorry

end NUMINAMATH_GPT_audrey_not_dreaming_fraction_l44_4406


namespace NUMINAMATH_GPT_consecutive_numbers_probability_l44_4498

theorem consecutive_numbers_probability :
  let total_ways := Nat.choose 20 5
  let non_consecutive_ways := Nat.choose 16 5
  let probability_of_non_consecutive := (non_consecutive_ways : ℚ) / (total_ways : ℚ)
  let probability_of_consecutive := 1 - probability_of_non_consecutive
  probability_of_consecutive = 232 / 323 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_numbers_probability_l44_4498


namespace NUMINAMATH_GPT_tan_of_angle_through_point_l44_4411

theorem tan_of_angle_through_point (α : ℝ) (hα : ∃ x y : ℝ, (x = 1) ∧ (y = 2) ∧ (y/x = (Real.sin α) / (Real.cos α))) :
  Real.tan α = 2 :=
sorry

end NUMINAMATH_GPT_tan_of_angle_through_point_l44_4411


namespace NUMINAMATH_GPT_lines_region_division_l44_4483

theorem lines_region_division (f : ℕ → ℕ) (k : ℕ) (h : k ≥ 2) : 
  (∀ m, f m = m * (m + 1) / 2 + 1) → f (k + 1) = f k + (k + 1) :=
by
  intro h_f
  have h_base : f 1 = 2 := by sorry
  have h_ih : ∀ n, n ≥ 2 → f (n + 1) = f n + (n + 1) := by sorry
  exact h_ih k h

end NUMINAMATH_GPT_lines_region_division_l44_4483


namespace NUMINAMATH_GPT_exception_to_roots_l44_4431

theorem exception_to_roots (x : ℝ) :
    ¬ (∃ x₀, (x₀ ∈ ({x | x = x} ∩ {x | x = x - 2}))) :=
by sorry

end NUMINAMATH_GPT_exception_to_roots_l44_4431


namespace NUMINAMATH_GPT_number_of_interior_diagonals_of_dodecahedron_l44_4417

-- Definitions based on conditions
def dodecahedron_vertices := 20
def faces_per_vertex := 3
def vertices_per_face := 5
def shared_edges_per_vertex := faces_per_vertex
def total_faces := 12
def total_vertices := 20

-- Property of the dodecahedron
def potential_diagonals_per_vertex := dodecahedron_vertices - 1 - shared_edges_per_vertex - (vertices_per_face - 1)
def total_potential_diagonals := potential_diagonals_per_vertex * total_vertices

-- Proof statement:
theorem number_of_interior_diagonals_of_dodecahedron :
  total_potential_diagonals / 2 = 90 :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_number_of_interior_diagonals_of_dodecahedron_l44_4417


namespace NUMINAMATH_GPT_sum_of_interior_angles_divisible_by_360_l44_4479

theorem sum_of_interior_angles_divisible_by_360
  (n : ℕ)
  (h : n > 0) :
  ∃ k : ℤ, ((2 * n - 2) * 180) = 360 * k :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_divisible_by_360_l44_4479


namespace NUMINAMATH_GPT_total_money_shared_l44_4474

-- Let us define the conditions
def ratio (a b c : ℕ) : Prop := ∃ k : ℕ, (2 * k = a) ∧ (3 * k = b) ∧ (8 * k = c)

def olivia_share := 30

-- Our goal is to prove the total amount of money shared
theorem total_money_shared (a b c : ℕ) (h_ratio : ratio a b c) (h_olivia : a = olivia_share) :
    a + b + c = 195 :=
by
  sorry

end NUMINAMATH_GPT_total_money_shared_l44_4474


namespace NUMINAMATH_GPT_intersection_M_N_l44_4435

def M := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }
def N := { y : ℝ | y > 0 }

theorem intersection_M_N : (M ∩ N) = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l44_4435


namespace NUMINAMATH_GPT_solve_for_square_l44_4475

theorem solve_for_square (x : ℝ) 
  (h : 10 + 9 + 8 * 7 / x + 6 - 5 * 4 - 3 * 2 = 1) : 
  x = 28 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_square_l44_4475


namespace NUMINAMATH_GPT_colbert_materials_needed_l44_4412

def wooden_planks_needed (total_needed quarter_in_stock : ℕ) : ℕ :=
  let total_purchased := total_needed - quarter_in_stock / 4
  (total_purchased + 7) / 8 -- ceil division by 8

def iron_nails_needed (total_needed thirty_percent_provided : ℕ) : ℕ :=
  let total_purchased := total_needed - total_needed * thirty_percent_provided / 100
  (total_purchased + 24) / 25 -- ceil division by 25

def fabric_needed (total_needed third_provided : ℚ) : ℚ :=
  total_needed - total_needed / third_provided

def metal_brackets_needed (total_needed in_stock multiple : ℕ) : ℕ :=
  let total_purchased := total_needed - in_stock
  (total_purchased + multiple - 1) / multiple * multiple -- ceil to next multiple of 5

theorem colbert_materials_needed :
  wooden_planks_needed 250 62 = 24 ∧
  iron_nails_needed 500 30 = 14 ∧
  fabric_needed 10 3 = 6.67 ∧
  metal_brackets_needed 40 10 5 = 30 :=
by sorry

end NUMINAMATH_GPT_colbert_materials_needed_l44_4412


namespace NUMINAMATH_GPT_proof_problem_l44_4450

-- Define the problem:
def problem := ∀ (a : Fin 100 → ℝ), 
  (∀ i j, i ≠ j → a i ≠ a j) →  -- All numbers are distinct
  ∃ i : Fin 100, a i + a (⟨i.val + 3, sorry⟩) > a (⟨i.val + 1, sorry⟩) + a (⟨i.val + 2, sorry⟩)
-- Summarize: there exists four consecutive points on the circle such that 
-- the sum of the numbers at the ends is greater than the sum of the numbers in the middle.

theorem proof_problem : problem := sorry

end NUMINAMATH_GPT_proof_problem_l44_4450


namespace NUMINAMATH_GPT_circle_symmetric_point_l44_4488

theorem circle_symmetric_point (a b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + a * x - 2 * y + b = 0 → x = 2 ∧ y = 1) ∧
  (∀ x y : ℝ, (x, y) ∈ { (px, py) | px = 2 ∧ py = 1 ∨ x + y - 1 = 0 } → x^2 + y^2 + a * x - 2 * y + b = 0) →
  a = 0 ∧ b = -3 := 
by {
  sorry
}

end NUMINAMATH_GPT_circle_symmetric_point_l44_4488


namespace NUMINAMATH_GPT_first_term_of_geometric_series_l44_4446

theorem first_term_of_geometric_series (r a S : ℚ) (h_common_ratio : r = -1/5) (h_sum : S = 16) :
  a = 96 / 5 :=
by
  sorry

end NUMINAMATH_GPT_first_term_of_geometric_series_l44_4446


namespace NUMINAMATH_GPT_tea_to_cheese_ratio_l44_4408

-- Definitions based on conditions
def total_cost : ℝ := 21
def tea_cost : ℝ := 10
def butter_to_cheese_ratio : ℝ := 0.8
def bread_to_butter_ratio : ℝ := 0.5

-- Main theorem statement
theorem tea_to_cheese_ratio (B C Br : ℝ) (hBr : Br = B * bread_to_butter_ratio) (hB : B = butter_to_cheese_ratio * C) (hTotal : B + Br + C + tea_cost = total_cost) :
  10 / C = 2 :=
  sorry

end NUMINAMATH_GPT_tea_to_cheese_ratio_l44_4408


namespace NUMINAMATH_GPT_dot_product_vec_a_vec_b_l44_4439

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem dot_product_vec_a_vec_b : dot_product vec_a vec_b = 1 := by
  sorry

end NUMINAMATH_GPT_dot_product_vec_a_vec_b_l44_4439


namespace NUMINAMATH_GPT_man_walking_rate_l44_4443

theorem man_walking_rate (x : ℝ) 
  (woman_rate : ℝ := 15)
  (woman_time_after_passing : ℝ := 2 / 60)
  (man_time_to_catch_up : ℝ := 4 / 60)
  (distance_woman : ℝ := woman_rate * woman_time_after_passing)
  (distance_man : ℝ := x * man_time_to_catch_up)
  (h : distance_man = distance_woman) :
  x = 7.5 :=
sorry

end NUMINAMATH_GPT_man_walking_rate_l44_4443


namespace NUMINAMATH_GPT_bob_calories_consumed_l44_4495

/-- Bob eats half of the pizza with 8 slices, each slice being 300 calories.
   Prove that Bob eats 1200 calories. -/
theorem bob_calories_consumed (total_slices : ℕ) (calories_per_slice : ℕ) (half_slices : ℕ) (calories_consumed : ℕ) 
  (h1 : total_slices = 8)
  (h2 : calories_per_slice = 300)
  (h3 : half_slices = total_slices / 2)
  (h4 : calories_consumed = half_slices * calories_per_slice) 
  : calories_consumed = 1200 := 
sorry

end NUMINAMATH_GPT_bob_calories_consumed_l44_4495


namespace NUMINAMATH_GPT_unique_elements_set_l44_4432

theorem unique_elements_set (x : ℝ) : x ≠ 3 ∧ x ≠ -1 ∧ x ≠ 0 ↔ 3 ≠ x ∧ x ≠ (x ^ 2 - 2 * x) ∧ (x ^ 2 - 2 * x) ≠ 3 := by
  sorry

end NUMINAMATH_GPT_unique_elements_set_l44_4432


namespace NUMINAMATH_GPT_average_of_remaining_numbers_l44_4464

theorem average_of_remaining_numbers 
  (S S' : ℝ)
  (h1 : S / 12 = 90)
  (h2 : S' = S - 80 - 82) :
  S' / 10 = 91.8 :=
sorry

end NUMINAMATH_GPT_average_of_remaining_numbers_l44_4464


namespace NUMINAMATH_GPT_expand_polynomial_l44_4452

theorem expand_polynomial :
  (2 * t^2 - 3 * t + 2) * (-3 * t^2 + t - 5) = -6 * t^4 + 11 * t^3 - 19 * t^2 + 17 * t - 10 :=
by sorry

end NUMINAMATH_GPT_expand_polynomial_l44_4452


namespace NUMINAMATH_GPT_bracelet_price_l44_4458

theorem bracelet_price 
  (B : ℝ) -- price of each bracelet
  (H1 : B > 0) 
  (H2 : 3 * B + 2 * 10 + 20 = 100 - 15) : 
  B = 15 :=
by
  sorry

end NUMINAMATH_GPT_bracelet_price_l44_4458


namespace NUMINAMATH_GPT_no_solution_l44_4447

noncomputable def problem_statement : Prop :=
  ∀ x : ℝ, ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x)))

theorem no_solution : problem_statement :=
by
  intro x
  have h₁ : ¬(85 + x = 3.5 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  have h₂ : ¬(55 + x = 2 * (15 + x)) -> ¬((85 + x = 3.5 * (15 + x)) ∧ (55 + x = 2 * (15 + x))) := sorry
  exact sorry

end NUMINAMATH_GPT_no_solution_l44_4447


namespace NUMINAMATH_GPT_range_of_m_l44_4451

variable {R : Type*} [LinearOrderedField R]

def discriminant (a b c : R) := b * b - 4 * a * c

theorem range_of_m (m : R) : (∀ x : R, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l44_4451


namespace NUMINAMATH_GPT_number_of_terms_in_ap_is_eight_l44_4416

theorem number_of_terms_in_ap_is_eight
  (n : ℕ) (a d : ℝ)
  (even : n % 2 = 0)
  (sum_odd : (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 24)
  (sum_even : (n / 2 : ℝ) * (2 * a + n * d) = 30)
  (last_exceeds_first : (n - 1) * d = 10.5) :
  n = 8 :=
by sorry

end NUMINAMATH_GPT_number_of_terms_in_ap_is_eight_l44_4416


namespace NUMINAMATH_GPT_largest_number_is_40_l44_4497

theorem largest_number_is_40 
    (a b c : ℕ) 
    (h1 : a ≠ b)
    (h2 : b ≠ c)
    (h3 : a ≠ c)
    (h4 : a + b + c = 100)
    (h5 : c - b = 8)
    (h6 : b - a = 4) : c = 40 :=
sorry

end NUMINAMATH_GPT_largest_number_is_40_l44_4497


namespace NUMINAMATH_GPT_angle_C_length_CD_area_range_l44_4423

-- 1. Prove C = π / 3 given (2a - b)cos C = c cos B
theorem angle_C (a b c : ℝ) (A B C : ℝ) (h : (2 * a - b) * Real.cos C = c * Real.cos B) : 
  C = Real.pi / 3 := sorry

-- 2. Prove the length of CD is 6√3 / 5 given a = 2, b = 3, and CD is the angle bisector of angle C
theorem length_CD (a b x : ℝ) (C D : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : x = (6 * Real.sqrt 3) / 5) : 
  x = (6 * Real.sqrt 3) / 5 := sorry

-- 3. Prove the range of values for the area of acute triangle ABC is (8√3 / 3, 4√3] given a cos B + b cos A = 4
theorem area_range (a b : ℝ) (A B C : ℝ) (S : Set ℝ) (h1 : a * Real.cos B + b * Real.cos A = 4) 
  (h2 : S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3)) : 
  S = Set.Ioc (8 * Real.sqrt 3 / 3) (4 * Real.sqrt 3) := sorry

end NUMINAMATH_GPT_angle_C_length_CD_area_range_l44_4423


namespace NUMINAMATH_GPT_fourier_series_decomposition_l44_4401

open Real

noncomputable def f : ℝ → ℝ :=
  λ x => if (x < 0) then -1 else (if (0 < x) then 1/2 else 0)

theorem fourier_series_decomposition :
    ∀ x, -π ≤ x ∧ x ≤ π →
         f x = -1/4 + (3/π) * ∑' k, (sin ((2*k+1)*x)) / (2*k+1) :=
by
  sorry

end NUMINAMATH_GPT_fourier_series_decomposition_l44_4401


namespace NUMINAMATH_GPT_sum_eighth_row_l44_4421

-- Definitions based on the conditions
def sum_of_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

axiom sum_fifth_row : sum_of_interior_numbers 5 = 14
axiom sum_sixth_row : sum_of_interior_numbers 6 = 30

-- The proof problem statement
theorem sum_eighth_row : sum_of_interior_numbers 8 = 126 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_eighth_row_l44_4421


namespace NUMINAMATH_GPT_prob_at_least_one_2_in_two_8_sided_dice_l44_4449

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end NUMINAMATH_GPT_prob_at_least_one_2_in_two_8_sided_dice_l44_4449


namespace NUMINAMATH_GPT_milk_transfer_equal_l44_4496

theorem milk_transfer_equal (A B C x : ℕ) (hA : A = 1200) (hB : B = A - 750) (hC : C = A - B) (h_eq : B + x = C - x) :
  x = 150 :=
by
  sorry

end NUMINAMATH_GPT_milk_transfer_equal_l44_4496


namespace NUMINAMATH_GPT_amount_saved_percent_l44_4456

variable (S : ℝ)

theorem amount_saved_percent :
  (0.165 * S) / (0.10 * S) * 100 = 165 := sorry

end NUMINAMATH_GPT_amount_saved_percent_l44_4456


namespace NUMINAMATH_GPT_sock_combination_count_l44_4419

noncomputable def numSockCombinations : Nat :=
  let striped := 4
  let solid := 4
  let checkered := 4
  let striped_and_solid := striped * solid
  let striped_and_checkered := striped * checkered
  striped_and_solid + striped_and_checkered

theorem sock_combination_count :
  numSockCombinations = 32 :=
by
  unfold numSockCombinations
  sorry

end NUMINAMATH_GPT_sock_combination_count_l44_4419


namespace NUMINAMATH_GPT_max_balls_in_cube_l44_4404

noncomputable def volume_of_cube : ℝ := (5 : ℝ)^3

noncomputable def volume_of_ball : ℝ := (4 / 3) * Real.pi * (1 : ℝ)^3

theorem max_balls_in_cube (c_length : ℝ) (b_radius : ℝ) (h1 : c_length = 5)
  (h2 : b_radius = 1) : 
  ⌊volume_of_cube / volume_of_ball⌋ = 29 := 
by
  sorry

end NUMINAMATH_GPT_max_balls_in_cube_l44_4404


namespace NUMINAMATH_GPT_remainder_2_pow_2015_mod_20_l44_4426

/-- 
  Given that powers of 2 modulo 20 follow a repeating cycle every 4 terms:
  2, 4, 8, 16, 12
  
  Prove that the remainder when \(2^{2015}\) is divided by 20 is 8.
-/
theorem remainder_2_pow_2015_mod_20 : (2 ^ 2015) % 20 = 8 :=
by
  -- The proof is to be filled in.
  sorry

end NUMINAMATH_GPT_remainder_2_pow_2015_mod_20_l44_4426


namespace NUMINAMATH_GPT_cube_diagonal_length_l44_4490

theorem cube_diagonal_length
  (side_length : ℝ)
  (h_side_length : side_length = 15) :
  ∃ d : ℝ, d = side_length * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_cube_diagonal_length_l44_4490


namespace NUMINAMATH_GPT_value_of_a_squared_plus_2a_l44_4499

theorem value_of_a_squared_plus_2a (a x : ℝ) (h1 : x = -5) (h2 : 2 * x + 8 = x / 5 - a) : a^2 + 2 * a = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_a_squared_plus_2a_l44_4499


namespace NUMINAMATH_GPT_ladder_length_difference_l44_4462

theorem ladder_length_difference :
  ∀ (flights : ℕ) (flight_height rope ladder_total_height : ℕ),
    flights = 3 →
    flight_height = 10 →
    rope = (flights * flight_height) / 2 →
    ladder_total_height = 70 →
    ladder_total_height - (flights * flight_height + rope) = 25 →
    ladder_total_height - (flights * flight_height) - rope = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ladder_length_difference_l44_4462


namespace NUMINAMATH_GPT_primes_solution_l44_4469

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_solution (p : ℕ) (hp : is_prime p) :
  is_prime (p^2 + 2007 * p - 1) ↔ p = 3 :=
by
  sorry

end NUMINAMATH_GPT_primes_solution_l44_4469


namespace NUMINAMATH_GPT_transform_quadratic_to_squared_form_l44_4491

theorem transform_quadratic_to_squared_form :
  ∀ x : ℝ, 2 * x^2 - 3 * x + 1 = 0 → (x - 3 / 4)^2 = 1 / 16 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_transform_quadratic_to_squared_form_l44_4491


namespace NUMINAMATH_GPT_find_second_number_l44_4465

theorem find_second_number
  (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x = (3 : ℚ) / 4 * y)
  (h3 : z = (7 : ℚ) / 5 * y) :
  y = 800 / 21 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l44_4465


namespace NUMINAMATH_GPT_six_digit_number_divisible_by_eleven_l44_4429

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def reverse_digits (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a

def concatenate_reverse (a b c : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a

theorem six_digit_number_divisible_by_eleven (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
  (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : 0 ≤ c) (h₆ : c ≤ 9) :
  11 ∣ concatenate_reverse a b c :=
by
  sorry

end NUMINAMATH_GPT_six_digit_number_divisible_by_eleven_l44_4429


namespace NUMINAMATH_GPT_student_A_more_stable_l44_4477

-- Define the variances for students A and B
def variance_A : ℝ := 0.05
def variance_B : ℝ := 0.06

-- The theorem to prove that student A has more stable performance
theorem student_A_more_stable : variance_A < variance_B :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_student_A_more_stable_l44_4477


namespace NUMINAMATH_GPT_linoleum_cut_rearrange_l44_4409

def linoleum : Type := sorry -- placeholder for the specific type of the linoleum piece

def A : linoleum := sorry -- define piece A
def B : linoleum := sorry -- define piece B

def cut_and_rearrange (L : linoleum) (A B : linoleum) : Prop :=
  -- Define the proposition that pieces A and B can be rearranged into an 8x8 square
  sorry

theorem linoleum_cut_rearrange (L : linoleum) (A B : linoleum) :
  cut_and_rearrange L A B :=
sorry

end NUMINAMATH_GPT_linoleum_cut_rearrange_l44_4409


namespace NUMINAMATH_GPT_bananas_per_box_l44_4480

def total_bananas : ℕ := 40
def number_of_boxes : ℕ := 10

theorem bananas_per_box : total_bananas / number_of_boxes = 4 := by
  sorry

end NUMINAMATH_GPT_bananas_per_box_l44_4480


namespace NUMINAMATH_GPT_sum_x_y_z_l44_4494

theorem sum_x_y_z (a b : ℝ) (x y z : ℕ) 
  (h_a : a^2 = 16 / 44) 
  (h_b : b^2 = (2 + Real.sqrt 5)^2 / 11) 
  (h_a_neg : a < 0) 
  (h_b_pos : b > 0) 
  (h_expr : (a + b)^3 = x * Real.sqrt y / z) : 
  x + y + z = 181 := 
sorry

end NUMINAMATH_GPT_sum_x_y_z_l44_4494


namespace NUMINAMATH_GPT_problem4_l44_4407

theorem problem4 (a : ℝ) : (a-1)^2 = a^3 - 2*a + 1 ↔ a = 0 ∨ a = 1 := 
by sorry

end NUMINAMATH_GPT_problem4_l44_4407


namespace NUMINAMATH_GPT_max_abs_diff_f_l44_4489

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f (k : ℝ) (h₁ : -3 ≤ k) (h₂ : k ≤ -1) (x₁ x₂ : ℝ) (h₃ : k ≤ x₁) (h₄ : x₁ ≤ k + 2) (h₅ : k ≤ x₂) (h₆ : x₂ ≤ k + 2) :
  |f x₁ - f x₂| ≤ 4 * Real.exp 1 := sorry

end NUMINAMATH_GPT_max_abs_diff_f_l44_4489


namespace NUMINAMATH_GPT_donna_card_shop_hourly_wage_correct_l44_4405

noncomputable def donna_hourly_wage_at_card_shop : ℝ := 
  let total_earnings := 305.0
  let earnings_dog_walking := 2 * 10.0 * 5
  let earnings_babysitting := 4 * 10.0
  let earnings_card_shop := total_earnings - (earnings_dog_walking + earnings_babysitting)
  let hours_card_shop := 5 * 2
  earnings_card_shop / hours_card_shop

theorem donna_card_shop_hourly_wage_correct : donna_hourly_wage_at_card_shop = 16.50 :=
by 
  -- Skipping proof steps for the implementation
  sorry

end NUMINAMATH_GPT_donna_card_shop_hourly_wage_correct_l44_4405


namespace NUMINAMATH_GPT_find_k_values_l44_4413

theorem find_k_values (k : ℝ) : 
  ((2 * 1 + 3 * k = 0) ∨
   (1 * 2 + (3 - k) * 3 = 0) ∨
   (1 * 1 + (3 - k) * k = 0)) →
   (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 3)/2 ∨ k = (3 - Real.sqrt 3)/2) := 
by
  sorry

end NUMINAMATH_GPT_find_k_values_l44_4413


namespace NUMINAMATH_GPT_cos_ratio_l44_4486

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (angle_A angle_B angle_C : ℝ)
variable (bc_coeff : 2 * c = 3 * b)
variable (sin_coeff : Real.sin angle_A = 2 * Real.sin angle_B)

theorem cos_ratio :
  (2 * c = 3 * b) →
  (Real.sin angle_A = 2 * Real.sin angle_B) →
  let cos_A := (b^2 + c^2 - a^2) / (2 * b * c)
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  (Real.cos angle_A / Real.cos angle_B = -2 / 7) :=
by
  intros bc_coeff sin_coeff
  sorry

end NUMINAMATH_GPT_cos_ratio_l44_4486


namespace NUMINAMATH_GPT_largest_perimeter_regular_polygons_l44_4485

theorem largest_perimeter_regular_polygons :
  ∃ (p q r : ℕ), 
    (p ≥ 3 ∧ q ≥ 3 ∧ r >= 3) ∧
    (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧
    (180 * (p - 2)/p + 180 * (q - 2)/q + 180 * (r - 2)/r = 360) ∧
    ((p + q + r - 6) = 9) :=
sorry

end NUMINAMATH_GPT_largest_perimeter_regular_polygons_l44_4485


namespace NUMINAMATH_GPT_find_x_l44_4478

theorem find_x (x : ℝ) (h : (2 * x) / 16 = 25) : x = 200 :=
sorry

end NUMINAMATH_GPT_find_x_l44_4478


namespace NUMINAMATH_GPT_chef_makes_10_cakes_l44_4481

def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

theorem chef_makes_10_cakes :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 := by
  sorry

end NUMINAMATH_GPT_chef_makes_10_cakes_l44_4481


namespace NUMINAMATH_GPT_water_consumption_per_hour_l44_4445

theorem water_consumption_per_hour 
  (W : ℝ) 
  (initial_water : ℝ := 20) 
  (initial_food : ℝ := 10) 
  (initial_gear : ℝ := 20) 
  (food_consumption_rate : ℝ := 1 / 3) 
  (hours : ℝ := 6) 
  (remaining_weight : ℝ := 34)
  (initial_weight := initial_water + initial_food + initial_gear)
  (consumed_water := W * hours)
  (consumed_food := food_consumption_rate * W * hours)
  (consumed_weight := consumed_water + consumed_food)
  (final_equation := initial_weight - consumed_weight)
  (correct_answer := 2) :
  final_equation = remaining_weight → W = correct_answer := 
by 
  sorry

end NUMINAMATH_GPT_water_consumption_per_hour_l44_4445


namespace NUMINAMATH_GPT_initial_volume_of_mixture_l44_4428

theorem initial_volume_of_mixture (M W : ℕ) (h1 : 2 * M = 3 * W) (h2 : 4 * M = 3 * (W + 46)) : M + W = 115 := 
sorry

end NUMINAMATH_GPT_initial_volume_of_mixture_l44_4428


namespace NUMINAMATH_GPT_gandalf_reachability_l44_4418

theorem gandalf_reachability :
  ∀ (k : ℕ), ∃ (s : ℕ → ℕ) (m : ℕ), (s 0 = 1) ∧ (s m = k) ∧ (∀ i < m, s (i + 1) = 2 * s i ∨ s (i + 1) = 3 * s i + 1) := 
by
  sorry

end NUMINAMATH_GPT_gandalf_reachability_l44_4418


namespace NUMINAMATH_GPT_consecutive_integers_sum_l44_4461

theorem consecutive_integers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 504) : n + n+1 + n+2 = 24 :=
sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l44_4461


namespace NUMINAMATH_GPT_triangle_perimeter_l44_4467

/-- Given the lengths of two sides of a triangle are 1 and 4,
    and the length of the third side is an integer, 
    prove that the perimeter of the triangle is 9 -/
theorem triangle_perimeter
  (a b : ℕ)
  (c : ℤ)
  (h₁ : a = 1)
  (h₂ : b = 4)
  (h₃ : 3 < c ∧ c < 5) :
  a + b + c = 9 :=
by sorry

end NUMINAMATH_GPT_triangle_perimeter_l44_4467


namespace NUMINAMATH_GPT_cone_height_l44_4422

theorem cone_height (S h H Vcone Vcylinder : ℝ)
  (hcylinder_height : H = 9)
  (hvolumes : Vcone = Vcylinder)
  (hbase_areas : S = S)
  (hV_cone : Vcone = (1 / 3) * S * h)
  (hV_cylinder : Vcylinder = S * H) : h = 27 :=
by
  -- sorry is used here to indicate missing proof steps which are predefined as unnecessary
  sorry

end NUMINAMATH_GPT_cone_height_l44_4422


namespace NUMINAMATH_GPT_interest_rate_first_year_l44_4402

theorem interest_rate_first_year (R : ℚ)
  (principal : ℚ := 7000)
  (final_amount : ℚ := 7644)
  (time_period_first_year : ℚ := 1)
  (time_period_second_year : ℚ := 1)
  (rate_second_year : ℚ := 5) :
  principal + (principal * R * time_period_first_year / 100) + 
  ((principal + (principal * R * time_period_first_year / 100)) * rate_second_year * time_period_second_year / 100) = final_amount →
  R = 4 := 
by {
  sorry
}

end NUMINAMATH_GPT_interest_rate_first_year_l44_4402


namespace NUMINAMATH_GPT_vector_expression_evaluation_l44_4400

theorem vector_expression_evaluation (θ : ℝ) :
  let a := (2 * Real.cos θ, Real.sin θ)
  let b := (1, -6)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (2 * Real.cos θ + Real.sin θ) / (Real.cos θ + 3 * Real.sin θ) = 7 / 6 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_vector_expression_evaluation_l44_4400


namespace NUMINAMATH_GPT_find_original_number_l44_4441

theorem find_original_number :
  ∃ x : ℚ, (5 * (3 * x + 15) = 245) ∧ x = 34 / 3 := by
  sorry

end NUMINAMATH_GPT_find_original_number_l44_4441


namespace NUMINAMATH_GPT_subset_zero_in_A_l44_4437

def A := { x : ℝ | x > -1 }

theorem subset_zero_in_A : {0} ⊆ A :=
by sorry

end NUMINAMATH_GPT_subset_zero_in_A_l44_4437


namespace NUMINAMATH_GPT_mean_value_of_pentagon_interior_angles_l44_4487

theorem mean_value_of_pentagon_interior_angles :
  let n := 5
  let sum_of_interior_angles := (n - 2) * 180
  let mean_value := sum_of_interior_angles / n
  mean_value = 108 :=
by
  sorry

end NUMINAMATH_GPT_mean_value_of_pentagon_interior_angles_l44_4487


namespace NUMINAMATH_GPT_find_some_number_l44_4470

theorem find_some_number : 
  ∃ (some_number : ℝ), (∃ (n : ℝ), n = 54 ∧ (n / some_number) * (n / 162) = 1) → some_number = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_some_number_l44_4470


namespace NUMINAMATH_GPT_problem_l44_4484

theorem problem (x y z : ℝ) 
  (h1 : (x - 4)^2 + (y - 3)^2 + (z - 2)^2 = 0)
  (h2 : 3 * x + 2 * y - z = 12) :
  x + y + z = 9 := 
  sorry

end NUMINAMATH_GPT_problem_l44_4484


namespace NUMINAMATH_GPT_complete_residue_system_l44_4410

theorem complete_residue_system {m n : ℕ} {a : ℕ → ℕ} {b : ℕ → ℕ}
  (h₁ : ∀ i j, 1 ≤ i → i ≤ m → 1 ≤ j → j ≤ n → (a i) * (b j) % (m * n) ≠ (a i) * (b j)) :
  (∀ i₁ i₂, 1 ≤ i₁ → i₁ ≤ m → 1 ≤ i₂ → i₂ ≤ m → i₁ ≠ i₂ → (a i₁ % m ≠ a i₂ % m)) ∧ 
  (∀ j₁ j₂, 1 ≤ j₁ → j₁ ≤ n → 1 ≤ j₂ → j₂ ≤ n → j₁ ≠ j₂ → (b j₁ % n ≠ b j₂ % n)) := sorry

end NUMINAMATH_GPT_complete_residue_system_l44_4410


namespace NUMINAMATH_GPT_sharmila_hourly_wage_l44_4453

-- Sharmila works 10 hours per day on Monday, Wednesday, and Friday.
def hours_worked_mwf : ℕ := 3 * 10

-- Sharmila works 8 hours per day on Tuesday and Thursday.
def hours_worked_tt : ℕ := 2 * 8

-- Total hours worked in a week.
def total_hours_worked : ℕ := hours_worked_mwf + hours_worked_tt

-- Sharmila earns $460 per week.
def weekly_earnings : ℕ := 460

-- Calculate and prove her hourly wage is $10 per hour.
theorem sharmila_hourly_wage : (weekly_earnings / total_hours_worked) = 10 :=
by sorry

end NUMINAMATH_GPT_sharmila_hourly_wage_l44_4453


namespace NUMINAMATH_GPT_probability_distribution_correct_l44_4442

noncomputable def X_possible_scores : Set ℤ := {-90, -30, 30, 90}

def prob_correct : ℚ := 0.8
def prob_incorrect : ℚ := 1 - prob_correct

def P_X_neg90 : ℚ := prob_incorrect ^ 3
def P_X_neg30 : ℚ := 3 * prob_correct * prob_incorrect ^ 2
def P_X_30 : ℚ := 3 * prob_correct ^ 2 * prob_incorrect
def P_X_90 : ℚ := prob_correct ^ 3

def P_advance : ℚ := P_X_30 + P_X_90

theorem probability_distribution_correct :
  (P_X_neg90 = (1/125) ∧ P_X_neg30 = (12/125) ∧ P_X_30 = (48/125) ∧ P_X_90 = (64/125)) ∧ 
  P_advance = (112/125) := 
by
  sorry

end NUMINAMATH_GPT_probability_distribution_correct_l44_4442


namespace NUMINAMATH_GPT_number_of_cars_on_street_l44_4444

-- Definitions based on conditions
def cars_equally_spaced (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

def distance_between_first_and_last_car (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 242

def distance_between_cars (n : ℕ) : Prop :=
  ∃ d : ℝ, d = 5.5

-- Given all conditions, prove n = 45
theorem number_of_cars_on_street (n : ℕ) :
  cars_equally_spaced n →
  distance_between_first_and_last_car n →
  distance_between_cars n →
  n = 45 :=
sorry

end NUMINAMATH_GPT_number_of_cars_on_street_l44_4444


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l44_4473

theorem sufficient_but_not_necessary (a : ℝ) : (a = 1 → a^2 = 1) ∧ ¬(a^2 = 1 → a = 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l44_4473


namespace NUMINAMATH_GPT_vertex_of_parabola_l44_4492

theorem vertex_of_parabola : 
  ∀ (x y : ℝ), (y = -x^2 + 3) → (0, 3) ∈ {(h, k) | ∃ (a : ℝ), y = a * (x - h)^2 + k} :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l44_4492


namespace NUMINAMATH_GPT_positive_difference_of_solutions_is_14_l44_4468

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 5 * x + 15 = x + 55

-- Define the positive difference between solutions of the quadratic equation
def positive_difference (a b : ℝ) : ℝ := |a - b|

-- State the theorem
theorem positive_difference_of_solutions_is_14 : 
  ∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ positive_difference a b = 14 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_solutions_is_14_l44_4468


namespace NUMINAMATH_GPT_art_of_passing_through_walls_l44_4438

theorem art_of_passing_through_walls (n : ℕ) :
  (2 * Real.sqrt (2 / 3) = Real.sqrt (2 * (2 / 3))) ∧
  (3 * Real.sqrt (3 / 8) = Real.sqrt (3 * (3 / 8))) ∧
  (4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15))) ∧
  (5 * Real.sqrt (5 / 24) = Real.sqrt (5 * (5 / 24))) →
  8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n)) →
  n = 63 :=
by
  sorry

end NUMINAMATH_GPT_art_of_passing_through_walls_l44_4438


namespace NUMINAMATH_GPT_range_of_k_l44_4403

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x > 0 → (k+4) * x < 0) → k < -4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l44_4403


namespace NUMINAMATH_GPT_squares_difference_l44_4482

theorem squares_difference (n : ℕ) (h : n > 0) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := 
by 
  sorry

end NUMINAMATH_GPT_squares_difference_l44_4482


namespace NUMINAMATH_GPT_number_of_division_games_l44_4424

theorem number_of_division_games (N M : ℕ) (h1 : N > 2 * M) (h2 : M > 5) (h3 : 4 * N + 5 * M = 100) :
  4 * N = 60 :=
by
  sorry

end NUMINAMATH_GPT_number_of_division_games_l44_4424


namespace NUMINAMATH_GPT_total_veg_eaters_l44_4457

def people_eat_only_veg : ℕ := 16
def people_eat_only_nonveg : ℕ := 9
def people_eat_both_veg_and_nonveg : ℕ := 12

theorem total_veg_eaters : people_eat_only_veg + people_eat_both_veg_and_nonveg = 28 := 
by
  sorry

end NUMINAMATH_GPT_total_veg_eaters_l44_4457


namespace NUMINAMATH_GPT_solve_fractional_equation_l44_4471

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) : 
  (2 * x) / (x - 1) = x / (3 * (x - 1)) + 1 ↔ x = -3 / 2 :=
by sorry

end NUMINAMATH_GPT_solve_fractional_equation_l44_4471


namespace NUMINAMATH_GPT_find_a_l44_4425

theorem find_a (a : ℝ) (extreme_at_neg_2 : ∀ x : ℝ, (3 * a * x^2 + 2 * x) = 0 → x = -2) :
    a = 1 / 3 :=
sorry

end NUMINAMATH_GPT_find_a_l44_4425


namespace NUMINAMATH_GPT_inequality_solution_set_l44_4459

theorem inequality_solution_set :
  { x : ℝ | (10 * x^2 + 20 * x - 68) / ((2 * x - 3) * (x + 4) * (x - 2)) < 3 } =
  { x : ℝ | (-4 < x ∧ x < -2) ∨ (-1 / 3 < x ∧ x < 3 / 2) } :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l44_4459


namespace NUMINAMATH_GPT_round_trip_time_correct_l44_4430

variables (river_current_speed boat_speed_still_water distance_upstream_distance : ℕ)

def upstream_speed := boat_speed_still_water - river_current_speed
def downstream_speed := boat_speed_still_water + river_current_speed

def time_upstream := distance_upstream_distance / upstream_speed
def time_downstream := distance_upstream_distance / downstream_speed

def round_trip_time := time_upstream + time_downstream

theorem round_trip_time_correct :
  river_current_speed = 10 →
  boat_speed_still_water = 50 →
  distance_upstream_distance = 120 →
  round_trip_time river_current_speed boat_speed_still_water distance_upstream_distance = 5 :=
by
  intros rc bs d
  sorry

end NUMINAMATH_GPT_round_trip_time_correct_l44_4430


namespace NUMINAMATH_GPT_range_of_4a_minus_2b_l44_4415

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b)
  (h2 : a - b ≤ 2)
  (h3 : 2 ≤ a + b)
  (h4 : a + b ≤ 4) : 
  5 ≤ 4 * a - 2 * b ∧ 4 * a - 2 * b ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_range_of_4a_minus_2b_l44_4415


namespace NUMINAMATH_GPT_heartsuit_calc_l44_4448

def heartsuit (u v : ℝ) : ℝ := (u + 2*v) * (u - v)

theorem heartsuit_calc : heartsuit 2 (heartsuit 3 4) = -260 := by
  sorry

end NUMINAMATH_GPT_heartsuit_calc_l44_4448


namespace NUMINAMATH_GPT_LindasTrip_l44_4434

theorem LindasTrip (x : ℝ) :
    (1 / 4) * x + 30 + (1 / 6) * x = x →
    x = 360 / 7 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_LindasTrip_l44_4434


namespace NUMINAMATH_GPT_calculate_expression_l44_4454

theorem calculate_expression :
  500 * 996 * 0.0996 * 20 + 5000 = 997016 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l44_4454


namespace NUMINAMATH_GPT_compound_interest_principal_l44_4476

theorem compound_interest_principal 
    (CI : Real)
    (r : Real)
    (n : Nat)
    (t : Nat)
    (A : Real)
    (P : Real) :
  CI = 945.0000000000009 →
  r = 0.10 →
  n = 1 →
  t = 2 →
  A = P * (1 + r / n) ^ (n * t) →
  CI = A - P →
  P = 4500.0000000000045 :=
by intros
   sorry

end NUMINAMATH_GPT_compound_interest_principal_l44_4476


namespace NUMINAMATH_GPT_three_digit_numbers_satisfy_condition_l44_4436

theorem three_digit_numbers_satisfy_condition : 
  ∃ (x y z : ℕ), 
    1 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 ∧ 
    0 ≤ z ∧ z ≤ 9 ∧ 
    x + y + z = (10 * x + y) - (10 * y + z) ∧ 
    (100 * x + 10 * y + z = 209 ∨ 
     100 * x + 10 * y + z = 428 ∨ 
     100 * x + 10 * y + z = 647 ∨ 
     100 * x + 10 * y + z = 866 ∨ 
     100 * x + 10 * y + z = 214 ∨ 
     100 * x + 10 * y + z = 433 ∨ 
     100 * x + 10 * y + z = 652 ∨ 
     100 * x + 10 * y + z = 871) := sorry

end NUMINAMATH_GPT_three_digit_numbers_satisfy_condition_l44_4436


namespace NUMINAMATH_GPT_M_value_l44_4466

noncomputable def x : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)

noncomputable def y : ℝ := Real.sqrt (4 - 2 * Real.sqrt 3)

noncomputable def M : ℝ := x - y

theorem M_value :
  M = (5 / 2) * Real.sqrt 2 - Real.sqrt 3 + (3 / 2) :=
sorry

end NUMINAMATH_GPT_M_value_l44_4466
