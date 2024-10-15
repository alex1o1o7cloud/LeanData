import Mathlib

namespace NUMINAMATH_GPT_value_of_2_68_times_0_74_l55_5575

theorem value_of_2_68_times_0_74 : 
  (268 * 74 = 19732) → (2.68 * 0.74 = 1.9732) :=
by intro h1; sorry

end NUMINAMATH_GPT_value_of_2_68_times_0_74_l55_5575


namespace NUMINAMATH_GPT_fertilizer_production_l55_5527

theorem fertilizer_production (daily_production : ℕ) (days : ℕ) (total_production : ℕ) 
  (h1 : daily_production = 105) 
  (h2 : days = 24) 
  (h3 : total_production = daily_production * days) : 
  total_production = 2520 := 
  by 
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_fertilizer_production_l55_5527


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l55_5503

-- Problem 1
theorem problem1 : (-20 + 3 - (-5) - 7 : Int) = -19 := sorry

-- Problem 2
theorem problem2 : (-2.4 - 3.7 - 4.6 + 5.7 : Real) = -5 := sorry

-- Problem 3
theorem problem3 : (-0.25 + ((-3 / 7) * (4 / 5)) : Real) = (-83 / 140) := sorry

-- Problem 4
theorem problem4 : ((-1 / 2) * (-8) + (-6)^2 : Real) = 40 := sorry

-- Problem 5
theorem problem5 : ((-1 / 12 - 1 / 36 + 1 / 6) * (-36) : Real) = -2 := sorry

-- Problem 6
theorem problem6 : (-1^4 + (-2) + (-1 / 3) - abs (-9) : Real) = -37 / 3 := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_l55_5503


namespace NUMINAMATH_GPT_circles_ordered_by_radius_l55_5529

def circle_radii_ordered (rA rB rC : ℝ) : Prop :=
  rA < rC ∧ rC < rB

theorem circles_ordered_by_radius :
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  circle_radii_ordered rA rB rC :=
by
  intros
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  show circle_radii_ordered rA rB rC
  sorry

end NUMINAMATH_GPT_circles_ordered_by_radius_l55_5529


namespace NUMINAMATH_GPT_apples_per_friend_l55_5538

def Benny_apples : Nat := 5
def Dan_apples : Nat := 2 * Benny_apples
def Total_apples : Nat := Benny_apples + Dan_apples
def Number_of_friends : Nat := 3

theorem apples_per_friend : Total_apples / Number_of_friends = 5 := by
  sorry

end NUMINAMATH_GPT_apples_per_friend_l55_5538


namespace NUMINAMATH_GPT_total_amount_l55_5524

theorem total_amount (x : ℝ) (hC : 2 * x = 70) :
  let B_share := 1.25 * x
  let C_share := 2 * x
  let D_share := 0.7 * x
  let E_share := 0.5 * x
  let A_share := x
  B_share + C_share + D_share + E_share + A_share = 190.75 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_l55_5524


namespace NUMINAMATH_GPT_similar_triangle_shortest_side_l55_5547

theorem similar_triangle_shortest_side (a b c: ℝ) (d e f: ℝ) :
  a = 21 ∧ b = 20 ∧ c = 29 ∧ d = 87 ∧ c^2 = a^2 + b^2 ∧ d / c = 3 → e = 60 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangle_shortest_side_l55_5547


namespace NUMINAMATH_GPT_find_g_of_one_fifth_l55_5570

variable {g : ℝ → ℝ}

theorem find_g_of_one_fifth (h₀ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1)
    (h₁ : g 0 = 0)
    (h₂ : ∀ {x y}, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y)
    (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x)
    (h₄ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2) :
  g (1 / 5) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_g_of_one_fifth_l55_5570


namespace NUMINAMATH_GPT_prism_width_l55_5577

/-- A rectangular prism with dimensions l, w, h such that the diagonal length is 13
    and given l = 3 and h = 12, has width w = 4. -/
theorem prism_width (w : ℕ) 
  (h : ℕ) (l : ℕ) 
  (diag_len : ℕ) 
  (hl : l = 3) 
  (hh : h = 12) 
  (hd : diag_len = 13) 
  (h_diag : diag_len = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := 
  sorry

end NUMINAMATH_GPT_prism_width_l55_5577


namespace NUMINAMATH_GPT_nate_total_time_l55_5578

/-- Definitions for the conditions -/
def sectionG : ℕ := 18 * 12
def sectionH : ℕ := 25 * 10
def sectionI : ℕ := 17 * 11
def sectionJ : ℕ := 20 * 9
def sectionK : ℕ := 15 * 13

def speedGH : ℕ := 8
def speedIJ : ℕ := 10
def speedK : ℕ := 6

/-- Compute the time spent in each section, rounding up where necessary -/
def timeG : ℕ := (sectionG + speedGH - 1) / speedGH
def timeH : ℕ := (sectionH + speedGH - 1) / speedGH
def timeI : ℕ := (sectionI + speedIJ - 1) / speedIJ
def timeJ : ℕ := (sectionJ + speedIJ - 1) / speedIJ
def timeK : ℕ := (sectionK + speedK - 1) / speedK

/-- Compute the total time spent -/
def totalTime : ℕ := timeG + timeH + timeI + timeJ + timeK

/-- The proof statement -/
theorem nate_total_time : totalTime = 129 := by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_nate_total_time_l55_5578


namespace NUMINAMATH_GPT_central_angle_of_sector_l55_5519

theorem central_angle_of_sector (r A θ : ℝ) (hr : r = 2) (hA : A = 4) :
  θ = 2 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l55_5519


namespace NUMINAMATH_GPT_projection_of_vec_c_onto_vec_b_l55_5599

def vec (x y : ℝ) : Prod ℝ ℝ := (x, y)

noncomputable def projection_of_c_onto_b := 
  let a := vec 2 3
  let b := vec (-4) 7
  let c := vec (-2) (-3)
  let dot_product_c_b := (-2) * (-4) + (-3) * 7
  let magnitude_b := Real.sqrt ((-4)^2 + 7^2)
  dot_product_c_b / magnitude_b
  
theorem projection_of_vec_c_onto_vec_b : 
  let a := vec 2 3
  let b := vec (-4) 7
  let c := vec (-2) (-3)
  let projection := projection_of_c_onto_b
  a + c = vec 0 0 ->
  projection = - Real.sqrt 65 / 5 := by
    sorry

end NUMINAMATH_GPT_projection_of_vec_c_onto_vec_b_l55_5599


namespace NUMINAMATH_GPT_values_of_x_l55_5594

theorem values_of_x (x : ℕ) (h : Nat.choose 18 x = Nat.choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_values_of_x_l55_5594


namespace NUMINAMATH_GPT_prob_not_same_city_l55_5523

def prob_A_city_A : ℝ := 0.6
def prob_B_city_A : ℝ := 0.3

theorem prob_not_same_city :
  (prob_A_city_A * (1 - prob_B_city_A) + (1 - prob_A_city_A) * prob_B_city_A) = 0.54 :=
by 
  -- This is just a placeholder to indicate that the proof is skipped
  sorry

end NUMINAMATH_GPT_prob_not_same_city_l55_5523


namespace NUMINAMATH_GPT_charts_per_associate_professor_l55_5550

theorem charts_per_associate_professor (A B C : ℕ) 
  (h1 : A + B = 6) 
  (h2 : 2 * A + B = 10) 
  (h3 : C * A + 2 * B = 8) : 
  C = 1 :=
by
  sorry

end NUMINAMATH_GPT_charts_per_associate_professor_l55_5550


namespace NUMINAMATH_GPT_min_rice_pounds_l55_5553

variable {o r : ℝ}

theorem min_rice_pounds (h1 : o ≥ 8 + r / 3) (h2 : o ≤ 2 * r) : r ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_rice_pounds_l55_5553


namespace NUMINAMATH_GPT_distance_to_y_axis_parabola_midpoint_l55_5557

noncomputable def distance_from_midpoint_to_y_axis (x1 x2 : ℝ) : ℝ :=
  (x1 + x2) / 2

theorem distance_to_y_axis_parabola_midpoint :
  ∀ (x1 y1 x2 y2 : ℝ), y1^2 = x1 → y2^2 = x2 → 
  abs (x1 + 1 / 4) + abs (x2 + 1 / 4) = 3 →
  abs (distance_from_midpoint_to_y_axis x1 x2) = 5 / 4 :=
by
  intros x1 y1 x2 y2 h1 h2 h3
  sorry

end NUMINAMATH_GPT_distance_to_y_axis_parabola_midpoint_l55_5557


namespace NUMINAMATH_GPT_b_share_in_profit_l55_5546

theorem b_share_in_profit (A B C : ℝ) (p : ℝ := 4400) (x : ℝ)
  (h1 : A = 3 * B)
  (h2 : B = (2 / 3) * C)
  (h3 : C = x) :
  B / (A + B + C) * p = 800 :=
by
  sorry

end NUMINAMATH_GPT_b_share_in_profit_l55_5546


namespace NUMINAMATH_GPT_change_in_mean_l55_5563

theorem change_in_mean {a b c d : ℝ} 
  (h1 : (a + b + c + d) / 4 = 10)
  (h2 : (b + c + d) / 3 = 11)
  (h3 : (a + c + d) / 3 = 12)
  (h4 : (a + b + d) / 3 = 13) : 
  ((a + b + c) / 3) = 4 := by 
  sorry

end NUMINAMATH_GPT_change_in_mean_l55_5563


namespace NUMINAMATH_GPT_expression_for_B_A_greater_than_B_l55_5525

-- Define the polynomials A and B
def A (x : ℝ) := 3 * x^2 - 2 * x + 1
def B (x : ℝ) := 2 * x^2 - x - 3

-- Prove that the given expression for B validates the equation A + B = 5x^2 - 4x - 2.
theorem expression_for_B (x : ℝ) : A x + 2 * x^2 - x - 3 = 5 * x^2 - 4 * x - 2 :=
by {
  sorry
}

-- Prove that A is always greater than B for all values of x.
theorem A_greater_than_B (x : ℝ) : A x > B x :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_for_B_A_greater_than_B_l55_5525


namespace NUMINAMATH_GPT_find_M_l55_5592

theorem find_M (M : ℤ) (h1 : 22 < M) (h2 : M < 24) : M = 23 := by
  sorry

end NUMINAMATH_GPT_find_M_l55_5592


namespace NUMINAMATH_GPT_sum_of_last_two_digits_l55_5584

theorem sum_of_last_two_digits (x y : ℕ) : 
  x = 8 → y = 12 → (x^25 + y^25) % 100 = 0 := 
by
  intros hx hy
  sorry

end NUMINAMATH_GPT_sum_of_last_two_digits_l55_5584


namespace NUMINAMATH_GPT_circle_radius_l55_5593

theorem circle_radius (x y : ℝ) : x^2 - 10 * x + y^2 + 4 * y + 13 = 0 → (x - 5)^2 + (y + 2)^2 = 4^2 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l55_5593


namespace NUMINAMATH_GPT_find_square_digit_l55_5512

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_digits_31_42_7s (s : ℕ) : ℕ :=
  3 + 1 + 4 + 2 + 7 + s

-- The main theorem to prove
theorem find_square_digit (d : ℕ) (h0 : is_even d) (h1 : (sum_digits_31_42_7s d) % 3 = 0) : d = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_square_digit_l55_5512


namespace NUMINAMATH_GPT_probability_second_roll_twice_first_l55_5511

theorem probability_second_roll_twice_first :
  let outcomes := [(1, 2), (2, 4), (3, 6)]
  let total_outcomes := 36
  let favorable_outcomes := 3
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_second_roll_twice_first_l55_5511


namespace NUMINAMATH_GPT_sum_of_b_for_one_solution_l55_5591

theorem sum_of_b_for_one_solution :
  let A := 3
  let C := 12
  ∀ b : ℝ, ((b + 5)^2 - 4 * A * C = 0) → (b = 7 ∨ b = -17) → (7 + (-17)) = -10 :=
by
  intro A C b
  sorry

end NUMINAMATH_GPT_sum_of_b_for_one_solution_l55_5591


namespace NUMINAMATH_GPT_find_50th_term_arithmetic_sequence_l55_5596

theorem find_50th_term_arithmetic_sequence :
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  a₅₀ = 346 :=
by
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  show a₅₀ = 346
  sorry

end NUMINAMATH_GPT_find_50th_term_arithmetic_sequence_l55_5596


namespace NUMINAMATH_GPT_ellipse_x_intercepts_l55_5537

noncomputable def distances_sum (x : ℝ) (y : ℝ) (f₁ f₂ : ℝ × ℝ) :=
  (Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2)) + (Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2))

def is_on_ellipse (x y : ℝ) : Prop := 
  distances_sum x y (0, 3) (4, 0) = 7

theorem ellipse_x_intercepts 
  (h₀ : is_on_ellipse 0 0) 
  (hx_intercept : ∀ x : ℝ, is_on_ellipse x 0 → x = 0 ∨ x = 20 / 7) :
  ∀ x : ℝ, is_on_ellipse x 0 ↔ x = 0 ∨ x = 20 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_x_intercepts_l55_5537


namespace NUMINAMATH_GPT_medicine_price_after_discount_l55_5560

theorem medicine_price_after_discount :
  ∀ (price : ℝ) (discount : ℝ), price = 120 → discount = 0.3 → 
  (price - price * discount) = 84 :=
by
  intros price discount h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_medicine_price_after_discount_l55_5560


namespace NUMINAMATH_GPT_george_coin_distribution_l55_5588

theorem george_coin_distribution (a b c : ℕ) (h₁ : a = 1050) (h₂ : b = 1260) (h₃ : c = 210) :
  Nat.gcd (Nat.gcd a b) c = 210 :=
by
  sorry

end NUMINAMATH_GPT_george_coin_distribution_l55_5588


namespace NUMINAMATH_GPT_notebook_cost_l55_5573

theorem notebook_cost
  (initial_amount : ℝ)
  (notebook_count : ℕ)
  (pen_count : ℕ)
  (pen_cost : ℝ)
  (remaining_amount : ℝ)
  (total_spent : ℝ)
  (notebook_cost : ℝ) :
  initial_amount = 15 →
  notebook_count = 2 →
  pen_count = 2 →
  pen_cost = 1.5 →
  remaining_amount = 4 →
  total_spent = initial_amount - remaining_amount →
  total_spent = notebook_count * notebook_cost + pen_count * pen_cost →
  notebook_cost = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_notebook_cost_l55_5573


namespace NUMINAMATH_GPT_mass_percentage_ba_in_bao_l55_5597

-- Define the constants needed in the problem
def molarMassBa : ℝ := 137.33
def molarMassO : ℝ := 16.00

-- Calculate the molar mass of BaO
def molarMassBaO : ℝ := molarMassBa + molarMassO

-- Express the problem as a Lean theorem for proof
theorem mass_percentage_ba_in_bao : 
  (molarMassBa / molarMassBaO) * 100 = 89.55 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_ba_in_bao_l55_5597


namespace NUMINAMATH_GPT_worst_is_father_l55_5576

-- Definitions for players
inductive Player
| father
| sister
| daughter
| son
deriving DecidableEq

open Player

def opposite_sex (p1 p2 : Player) : Bool :=
match p1, p2 with
| father, sister => true
| father, daughter => true
| sister, father => true
| daughter, father => true
| son, sister => true
| son, daughter => true
| daughter, son => true
| sister, son => true
| _, _ => false 

-- Problem conditions
variables (worst best : Player)
variable (twins : Player → Player)
variable (worst_best_twins : twins worst = best)
variable (worst_twin_conditions : opposite_sex (twins worst) best)

-- Goal: Prove that the worst player is the father
theorem worst_is_father : worst = Player.father := by
  sorry

end NUMINAMATH_GPT_worst_is_father_l55_5576


namespace NUMINAMATH_GPT_line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l55_5541

-- Define the points A, B and P
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the functions and theorems for the problem
theorem line_through_P_parallel_to_AB :
  ∃ k b : ℝ, ∀ x y : ℝ, ((y = k * x + b) ↔ (x + 2 * y - 8 = 0)) :=
sorry

theorem circumcircle_of_triangle_OAB :
  ∃ cx cy r : ℝ, (cx, cy) = (2, 1) ∧ r^2 = 5 ∧ ∀ x y : ℝ, ((x - cx)^2 + (y - cy)^2 = r^2) ↔ ((x - 2)^2 + (y - 1)^2 = 5) :=
sorry

end NUMINAMATH_GPT_line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l55_5541


namespace NUMINAMATH_GPT_center_digit_is_two_l55_5548

theorem center_digit_is_two :
  ∃ (a b : ℕ), (a^2 < 1000 ∧ b^2 < 1000 ∧ (a^2 ≠ b^2) ∧
  (∀ d, d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] → d ∈ [2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10] → d ∈ [2, 3, 4, 5, 6])) ∧
  (∀ d, (d ∈ [2, 3, 4, 5, 6]) → (d ∈ [a^2 / 100, (a^2 / 10) % 10, a^2 % 10] ∨ d ∈ [b^2 / 100, (b^2 / 10) % 10, b^2 % 10])) ∧
  2 = (a^2 / 10) % 10 ∨ 2 = (b^2 / 10) % 10 :=
sorry -- no proof needed, just the statement

end NUMINAMATH_GPT_center_digit_is_two_l55_5548


namespace NUMINAMATH_GPT_molecular_weight_correct_l55_5574

-- Define the atomic weights
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01

-- Define the number of atoms of each element
def num_atoms_K : ℕ := 2
def num_atoms_Br : ℕ := 2
def num_atoms_O : ℕ := 4
def num_atoms_H : ℕ := 3
def num_atoms_N : ℕ := 1

-- Calculate the molecular weight
def molecular_weight : ℝ :=
  num_atoms_K * atomic_weight_K +
  num_atoms_Br * atomic_weight_Br +
  num_atoms_O * atomic_weight_O +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N

-- Define the expected molecular weight
def expected_molecular_weight : ℝ := 319.04

-- The theorem stating that the calculated molecular weight matches the expected molecular weight
theorem molecular_weight_correct : molecular_weight = expected_molecular_weight :=
  by
  sorry -- Proof is skipped

end NUMINAMATH_GPT_molecular_weight_correct_l55_5574


namespace NUMINAMATH_GPT_stratified_sampling_counts_l55_5528

-- Defining the given conditions
def num_elderly : ℕ := 27
def num_middle_aged : ℕ := 54
def num_young : ℕ := 81
def total_sample : ℕ := 42

-- Proving the required stratified sample counts
theorem stratified_sampling_counts :
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  elderly_count = 7 ∧ middle_aged_count = 14 ∧ young_count = 21 :=
by 
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  have h1 : elderly_count = 7 := by sorry
  have h2 : middle_aged_count = 14 := by sorry
  have h3 : young_count = 21 := by sorry
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_stratified_sampling_counts_l55_5528


namespace NUMINAMATH_GPT_total_area_correct_l55_5587

-- Define the given conditions
def dust_covered_area : ℕ := 64535
def untouched_area : ℕ := 522

-- Define the total area of prairie by summing covered and untouched areas
def total_prairie_area : ℕ := dust_covered_area + untouched_area

-- State the theorem we need to prove
theorem total_area_correct : total_prairie_area = 65057 := by
  sorry

end NUMINAMATH_GPT_total_area_correct_l55_5587


namespace NUMINAMATH_GPT_tan_alpha_plus_cot_alpha_l55_5556

theorem tan_alpha_plus_cot_alpha (α : Real) (h : Real.sin (2 * α) = 3 / 4) : 
  Real.tan α + 1 / Real.tan α = 8 / 3 :=
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_cot_alpha_l55_5556


namespace NUMINAMATH_GPT_sarah_bought_new_shirts_l55_5540

-- Define the given conditions
def original_shirts : ℕ := 9
def total_shirts : ℕ := 17

-- The proof statement: Prove that the number of new shirts is 8
theorem sarah_bought_new_shirts : total_shirts - original_shirts = 8 := by
  sorry

end NUMINAMATH_GPT_sarah_bought_new_shirts_l55_5540


namespace NUMINAMATH_GPT_inequality_proof_l55_5526

variables (a b c d : ℝ)

theorem inequality_proof (h1 : a > b) (h2 : c = d) : a + c > b + d :=
sorry

end NUMINAMATH_GPT_inequality_proof_l55_5526


namespace NUMINAMATH_GPT_positive_correlation_not_proportional_l55_5565

/-- Two quantities x and y depend on each other, and when one increases, the other also increases.
    This general relationship is denoted as a function g such that for any x₁, x₂,
    if x₁ < x₂ then g(x₁) < g(x₂). This implies a positive correlation but not necessarily proportionality. 
    We will prove that this does not imply a proportional relationship (y = kx). -/
theorem positive_correlation_not_proportional (g : ℝ → ℝ) 
(h_increasing: ∀ x₁ x₂ : ℝ, x₁ < x₂ → g x₁ < g x₂) :
¬ ∃ k : ℝ, ∀ x : ℝ, g x = k * x :=
sorry

end NUMINAMATH_GPT_positive_correlation_not_proportional_l55_5565


namespace NUMINAMATH_GPT_solve_system_of_inequalities_l55_5549

theorem solve_system_of_inequalities (x : ℝ) :
  4*x^2 - 27*x + 18 > 0 ∧ x^2 + 4*x + 4 > 0 ↔ (x < 3/4 ∨ x > 6) ∧ x ≠ -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_inequalities_l55_5549


namespace NUMINAMATH_GPT_salon_customers_l55_5501

theorem salon_customers (C : ℕ) (H : C * 2 + 5 = 33) : C = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_salon_customers_l55_5501


namespace NUMINAMATH_GPT_side_length_square_correct_l55_5564

noncomputable def side_length_square (time_seconds : ℕ) (speed_kmph : ℕ) : ℕ := sorry

theorem side_length_square_correct (time_seconds : ℕ) (speed_kmph : ℕ) (h_time : time_seconds = 24) 
  (h_speed : speed_kmph = 12) : side_length_square time_seconds speed_kmph = 20 :=
sorry

end NUMINAMATH_GPT_side_length_square_correct_l55_5564


namespace NUMINAMATH_GPT_yola_past_weight_l55_5504

variable (W Y Y_past : ℕ)

-- Conditions
def condition1 : Prop := W = Y + 30
def condition2 : Prop := W = Y_past + 80
def condition3 : Prop := Y = 220

-- Theorem statement
theorem yola_past_weight : condition1 W Y → condition2 W Y_past → condition3 Y → Y_past = 170 :=
by
  intros h_condition1 h_condition2 h_condition3
  -- Placeholder for the proof, not required in the solution
  sorry

end NUMINAMATH_GPT_yola_past_weight_l55_5504


namespace NUMINAMATH_GPT_bad_carrots_l55_5590

-- Conditions
def carrots_picked_by_vanessa := 17
def carrots_picked_by_mom := 14
def good_carrots := 24
def total_carrots := carrots_picked_by_vanessa + carrots_picked_by_mom

-- Question and Proof
theorem bad_carrots :
  total_carrots - good_carrots = 7 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_bad_carrots_l55_5590


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l55_5539

noncomputable def log_base := (1 / 4 : ℝ)

def quad_expression (x : ℝ) : ℝ := -x^2 + 2*x + 3

def is_defined (x : ℝ) : Prop := quad_expression x > 0

theorem monotonic_increasing_interval : ∀ (x : ℝ), 
  is_defined x → 
  ∃ (a b : ℝ), 1 < a ∧ a ≤ x ∧ x < b ∧ b < 3 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l55_5539


namespace NUMINAMATH_GPT_min_value_2xy_minus_2x_minus_y_l55_5559

theorem min_value_2xy_minus_2x_minus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 2/y = 1) :
  2 * x * y - 2 * x - y ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_2xy_minus_2x_minus_y_l55_5559


namespace NUMINAMATH_GPT_cube_edge_length_and_volume_l55_5567

variable (edge_length : ℕ)

def cube_edge_total_length (edge_length : ℕ) : ℕ := edge_length * 12
def cube_volume (edge_length : ℕ) : ℕ := edge_length * edge_length * edge_length

theorem cube_edge_length_and_volume (h : cube_edge_total_length edge_length = 96) :
  edge_length = 8 ∧ cube_volume edge_length = 512 :=
by
  sorry

end NUMINAMATH_GPT_cube_edge_length_and_volume_l55_5567


namespace NUMINAMATH_GPT_perpendicular_length_GH_from_centroid_l55_5568

theorem perpendicular_length_GH_from_centroid
  (A B C D E F G : ℝ)
  -- Conditions for distances from vertices to the line RS
  (hAD : AD = 12)
  (hBE : BE = 12)
  (hCF : CF = 18)
  -- Define the coordinates based on the vertical distances to line RS
  (yA : A = 12)
  (yB : B = 12)
  (yC : C = 18)
  -- Define the centroid G of triangle ABC based on the average of the y-coordinates
  (yG : G = (A + B + C) / 3)
  : G = 14 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_length_GH_from_centroid_l55_5568


namespace NUMINAMATH_GPT_lotion_cost_l55_5502

variable (shampoo_conditioner_cost lotion_total_spend: ℝ)
variable (num_lotions num_lotions_cost_target: ℕ)
variable (free_shipping_threshold additional_spend_needed: ℝ)

noncomputable def cost_of_each_lotion := lotion_total_spend / num_lotions

theorem lotion_cost
    (h1 : shampoo_conditioner_cost = 10)
    (h2 : num_lotions = 3)
    (h3 : additional_spend_needed = 12)
    (h4 : free_shipping_threshold = 50)
    (h5 : (shampoo_conditioner_cost * 2) + additional_spend_needed + lotion_total_spend = free_shipping_threshold) :
    cost_of_each_lotion = 10 :=
by
  sorry

end NUMINAMATH_GPT_lotion_cost_l55_5502


namespace NUMINAMATH_GPT_driver_total_miles_per_week_l55_5552

theorem driver_total_miles_per_week :
  let distance_monday_to_saturday := (30 * 3 + 25 * 4 + 40 * 2) * 6
  let distance_sunday := 35 * (5 - 1)
  distance_monday_to_saturday + distance_sunday = 1760 := by
  sorry

end NUMINAMATH_GPT_driver_total_miles_per_week_l55_5552


namespace NUMINAMATH_GPT_inequality_solution_set_l55_5515

theorem inequality_solution_set (x : ℝ) :
  (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 := by
sorry

end NUMINAMATH_GPT_inequality_solution_set_l55_5515


namespace NUMINAMATH_GPT_paco_initial_cookies_l55_5535

theorem paco_initial_cookies (cookies_ate : ℕ) (cookies_left : ℕ) (cookies_initial : ℕ) 
  (h1 : cookies_ate = 15) (h2 : cookies_left = 78) :
  cookies_initial = cookies_ate + cookies_left → cookies_initial = 93 :=
by
  sorry

end NUMINAMATH_GPT_paco_initial_cookies_l55_5535


namespace NUMINAMATH_GPT_bus_commutes_three_times_a_week_l55_5582

-- Define the commuting times
def bike_time := 30
def bus_time := bike_time + 10
def friend_time := bike_time * (1 - (2/3))
def total_weekly_time := 160

-- Define the number of times taking the bus as a variable
variable (b : ℕ)

-- The equation for total commuting time
def commuting_time_eq := bike_time + bus_time * b + friend_time = total_weekly_time

-- The proof statement: b should be equal to 3
theorem bus_commutes_three_times_a_week (h : commuting_time_eq b) : b = 3 := sorry

end NUMINAMATH_GPT_bus_commutes_three_times_a_week_l55_5582


namespace NUMINAMATH_GPT_brian_books_chapters_l55_5536

variable (x : ℕ)

theorem brian_books_chapters (h1 : 1 ≤ x) (h2 : 20 + 2 * x + (20 + 2 * x) / 2 = 75) : x = 15 :=
sorry

end NUMINAMATH_GPT_brian_books_chapters_l55_5536


namespace NUMINAMATH_GPT_john_annual_profit_l55_5509

namespace JohnProfit

def number_of_people_subletting := 3
def rent_per_person_per_month := 400
def john_rent_per_month := 900
def months_in_year := 12

theorem john_annual_profit 
  (h1 : number_of_people_subletting = 3)
  (h2 : rent_per_person_per_month = 400)
  (h3 : john_rent_per_month = 900)
  (h4 : months_in_year = 12) : 
  (number_of_people_subletting * rent_per_person_per_month - john_rent_per_month) * months_in_year = 3600 :=
by
  sorry

end JohnProfit

end NUMINAMATH_GPT_john_annual_profit_l55_5509


namespace NUMINAMATH_GPT_frank_hamburger_goal_l55_5506

theorem frank_hamburger_goal:
  let price_per_hamburger := 5
  let group1_hamburgers := 2 * 4
  let group2_hamburgers := 2 * 2
  let current_hamburgers := group1_hamburgers + group2_hamburgers
  let extra_hamburgers_needed := 4
  let total_hamburgers := current_hamburgers + extra_hamburgers_needed
  price_per_hamburger * total_hamburgers = 80 :=
by
  sorry

end NUMINAMATH_GPT_frank_hamburger_goal_l55_5506


namespace NUMINAMATH_GPT_max_three_digit_sum_l55_5505

theorem max_three_digit_sum :
  ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ 101 * A + 11 * B + 11 * C = 986 := 
sorry

end NUMINAMATH_GPT_max_three_digit_sum_l55_5505


namespace NUMINAMATH_GPT_cricket_innings_l55_5555

theorem cricket_innings (n : ℕ) 
  (average_run : ℕ := 40) 
  (next_innings_run : ℕ := 84) 
  (new_average_run : ℕ := 44) :
  (40 * n + 84) / (n + 1) = 44 ↔ n = 10 := 
by
  sorry

end NUMINAMATH_GPT_cricket_innings_l55_5555


namespace NUMINAMATH_GPT_positive_numbers_inequality_l55_5516

theorem positive_numbers_inequality
  (x y z : ℝ)
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x * y + y * z + z * x = 6) :
  (1 / (2 * Real.sqrt 2 + x^2 * (y + z)) + 
   1 / (2 * Real.sqrt 2 + y^2 * (x + z)) + 
   1 / (2 * Real.sqrt 2 + z^2 * (x + y))) <= 
  (1 / (x * y * z)) :=
by
  sorry

end NUMINAMATH_GPT_positive_numbers_inequality_l55_5516


namespace NUMINAMATH_GPT_sum_of_7_and_2_terms_l55_5572

open Nat

variable {α : Type*} [Field α]

-- Definitions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = d
  
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∀ m n k : ℕ, m < n → n < k → a n * a n = a m * a k
  
def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n * (a 0 + a n)) / 2

-- Given Conditions
variable (a : ℕ → α) 
variable (d : α)

-- Checked. Arithmetic sequence with non-zero common difference
axiom h1 : is_arithmetic_sequence a d

-- Known values provided in the problem statement
axiom h2 : a 1 = 6

-- Terms forming a geometric sequence
axiom h3 : is_geometric_sequence a

-- The goal is to find the sum of the first 7 terms and the first 2 terms
theorem sum_of_7_and_2_terms : sum_first_n_terms a 7 + sum_first_n_terms a 2 = 80 := 
by {
  -- Proof will be here
  sorry
}

end NUMINAMATH_GPT_sum_of_7_and_2_terms_l55_5572


namespace NUMINAMATH_GPT_cube_face_min_sum_l55_5520

open Set

theorem cube_face_min_sum (S : Finset ℕ)
  (hS : S = {1, 2, 3, 4, 5, 6, 7, 8})
  (h_faces_sum : ∀ a b c d : ℕ, a ∈ S → b ∈ S → c ∈ S → d ∈ S → 
                    (a + b + c >= 10) ∨ 
                    (a + b + d >= 10) ∨ 
                    (a + c + d >= 10) ∨ 
                    (b + c + d >= 10)) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 16 :=
sorry

end NUMINAMATH_GPT_cube_face_min_sum_l55_5520


namespace NUMINAMATH_GPT_smallest_sum_of_four_consecutive_primes_divisible_by_five_l55_5521

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_sum_of_four_consecutive_primes_divisible_by_five :
  ∃ (a b c d : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    a < b ∧ b < c ∧ c < d ∧
    b = a + 2 ∧ c = b + 4 ∧ d = c + 2 ∧
    (a + b + c + d) % 5 = 0 ∧ (a + b + c + d = 60) := sorry

end NUMINAMATH_GPT_smallest_sum_of_four_consecutive_primes_divisible_by_five_l55_5521


namespace NUMINAMATH_GPT_pages_allocation_correct_l55_5508

-- Define times per page for Alice, Bob, and Chandra
def t_A := 40
def t_B := 60
def t_C := 48

-- Define pages read by Alice, Bob, and Chandra
def pages_A := 295
def pages_B := 197
def pages_C := 420

-- Total pages in the novel
def total_pages := 912

-- Calculate the total time each one spends reading
def total_time_A := t_A * pages_A
def total_time_B := t_B * pages_B
def total_time_C := t_C * pages_C

-- Theorem: Prove the correct allocation of pages
theorem pages_allocation_correct : 
  total_pages = pages_A + pages_B + pages_C ∧
  total_time_A = total_time_B ∧
  total_time_B = total_time_C :=
by 
  -- Place end of proof here 
  sorry

end NUMINAMATH_GPT_pages_allocation_correct_l55_5508


namespace NUMINAMATH_GPT_cara_neighbors_l55_5500

theorem cara_neighbors (friends : Finset Person) (mark : Person) (cara : Person) (h_mark : mark ∈ friends) (h_len : friends.card = 8) :
  ∃ pairs : Finset (Person × Person), pairs.card = 6 ∧
    ∀ (p : Person × Person), p ∈ pairs → p.1 = mark ∨ p.2 = mark :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_cara_neighbors_l55_5500


namespace NUMINAMATH_GPT_total_days_spent_on_island_l55_5598

noncomputable def first_expedition_weeks := 3
noncomputable def second_expedition_weeks := first_expedition_weeks + 2
noncomputable def last_expedition_weeks := 2 * second_expedition_weeks
noncomputable def total_weeks := first_expedition_weeks + second_expedition_weeks + last_expedition_weeks
noncomputable def total_days := 7 * total_weeks

theorem total_days_spent_on_island : total_days = 126 := by
  sorry

end NUMINAMATH_GPT_total_days_spent_on_island_l55_5598


namespace NUMINAMATH_GPT_quadratic_root_other_l55_5589

theorem quadratic_root_other (a : ℝ) (h : (3 : ℝ)*3 - 2*3 + a = 0) : 
  ∃ (b : ℝ), b = -1 ∧ (b : ℝ)*b - 2*b + a = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_other_l55_5589


namespace NUMINAMATH_GPT_stable_number_divisible_by_11_l55_5583

/-- Definition of a stable number as a three-digit number (cen, ten, uni) where
    each digit is non-zero, and the sum of any two digits is greater than the remaining digit.
-/
def is_stable_number (cen ten uni : ℕ) : Prop :=
cen ≠ 0 ∧ ten ≠ 0 ∧ uni ≠ 0 ∧
(cen + ten > uni) ∧ (cen + uni > ten) ∧ (ten + uni > cen)

/-- Function F defined for a stable number (cen ten uni). -/
def F (cen ten uni : ℕ) : ℕ := 10 * ten + cen + uni

/-- Function Q defined for a stable number (cen ten uni). -/
def Q (cen ten uni : ℕ) : ℕ := 10 * cen + ten + uni

/-- Statement to prove: Given a stable number s = 100a + 101b + 30 where 1 ≤ a ≤ 5 and 1 ≤ b ≤ 4,
    the expression 5 * F(s) + 2 * Q(s) is divisible by 11.
-/
theorem stable_number_divisible_by_11 (a b cen ten uni : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 5)
  (h_b : 1 ≤ b ∧ b ≤ 4)
  (h_s : 100 * a + 101 * b + 30 = 100 * cen + 10 * ten + uni)
  (h_stable : is_stable_number cen ten uni) :
  (5 * F cen ten uni + 2 * Q cen ten uni) % 11 = 0 :=
sorry

end NUMINAMATH_GPT_stable_number_divisible_by_11_l55_5583


namespace NUMINAMATH_GPT_find_f_2017_l55_5554

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x
axiom f_neg1 : f (-1) = -3

theorem find_f_2017 : f 2017 = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_f_2017_l55_5554


namespace NUMINAMATH_GPT_remaining_distance_is_one_l55_5544

def total_distance_to_grandma : ℕ := 78
def initial_distance_traveled : ℕ := 35
def bakery_detour : ℕ := 7
def pie_distance : ℕ := 18
def gift_detour : ℕ := 3
def next_travel_distance : ℕ := 12
def scenic_detour : ℕ := 2

def total_distance_traveled : ℕ :=
  initial_distance_traveled + bakery_detour + pie_distance + gift_detour + next_travel_distance + scenic_detour

theorem remaining_distance_is_one :
  total_distance_to_grandma - total_distance_traveled = 1 := by
  sorry

end NUMINAMATH_GPT_remaining_distance_is_one_l55_5544


namespace NUMINAMATH_GPT_dichromate_molecular_weight_l55_5510

theorem dichromate_molecular_weight :
  let atomic_weight_Cr := 52.00
  let atomic_weight_O := 16.00
  let dichromate_num_Cr := 2
  let dichromate_num_O := 7
  (dichromate_num_Cr * atomic_weight_Cr + dichromate_num_O * atomic_weight_O) = 216.00 :=
by
  sorry

end NUMINAMATH_GPT_dichromate_molecular_weight_l55_5510


namespace NUMINAMATH_GPT_m_gt_n_l55_5566

variable (m n : ℝ)

-- Definition of points A and B lying on the line y = -2x + 1
def point_A_on_line : Prop := m = -2 * (-1) + 1
def point_B_on_line : Prop := n = -2 * 3 + 1

-- Theorem stating that m > n given the conditions
theorem m_gt_n (hA : point_A_on_line m) (hB : point_B_on_line n) : m > n :=
by
  -- To avoid the proof part, which we skip as per instructions
  sorry

end NUMINAMATH_GPT_m_gt_n_l55_5566


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l55_5522

theorem quadratic_has_two_distinct_real_roots :
  ∃ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 6 ∧ a*x^2 + b*x + c = 0 → (b^2 - 4*a*c) > 0 := 
sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l55_5522


namespace NUMINAMATH_GPT_cost_of_staying_23_days_l55_5517

def hostel_cost (days: ℕ) : ℝ :=
  if days ≤ 7 then
    days * 18
  else
    7 * 18 + (days - 7) * 14

theorem cost_of_staying_23_days : hostel_cost 23 = 350 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_staying_23_days_l55_5517


namespace NUMINAMATH_GPT_option_a_correct_option_c_correct_option_d_correct_l55_5514

theorem option_a_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (1 / a > 1 / b) :=
sorry

theorem option_c_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (Real.sqrt (-a) > Real.sqrt (-b)) :=
sorry

theorem option_d_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (|a| > -b) :=
sorry

end NUMINAMATH_GPT_option_a_correct_option_c_correct_option_d_correct_l55_5514


namespace NUMINAMATH_GPT_smallest_c_l55_5543

theorem smallest_c {a b c : ℤ} (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c)
  (h4 : a^2 = c * b) : c = 4 :=
by
  -- We state the theorem here without proof. 
  -- The actual proof steps are omitted and replaced by sorry.
  sorry

end NUMINAMATH_GPT_smallest_c_l55_5543


namespace NUMINAMATH_GPT_intersection_correct_l55_5569

-- Define sets M and N
def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log (2 * x + 1) > 0}

-- Define the intersection of M and N
def M_intersect_N := {x : ℝ | 0 < x ∧ x < 1}

-- Prove that M_intersect_N is the correct intersection
theorem intersection_correct : M ∩ N = M_intersect_N :=
by
  sorry

end NUMINAMATH_GPT_intersection_correct_l55_5569


namespace NUMINAMATH_GPT_problem_statement_l55_5558

theorem problem_statement (a b : ℝ) (h : a + b = 1) : 
  ((∀ (a b : ℝ), a + b = 1 → ab ≤ 1/4) ∧ 
   (∀ (a b : ℝ), ¬(ab ≤ 1/4) → ¬(a + b = 1)) ∧ 
   ¬(∀ (a b : ℝ), ab ≤ 1/4 → a + b = 1) ∧ 
   ¬(∀ (a b : ℝ), ¬(a + b = 1) → ¬(ab ≤ 1/4))) := 
sorry

end NUMINAMATH_GPT_problem_statement_l55_5558


namespace NUMINAMATH_GPT_krystiana_earnings_l55_5561

def earning_building1_first_floor : ℝ := 5 * 15 * 0.8
def earning_building1_second_floor : ℝ := 6 * 25 * 0.75
def earning_building1_third_floor : ℝ := 9 * 30 * 0.5
def earning_building1_fourth_floor : ℝ := 4 * 60 * 0.85
def earnings_building1 : ℝ := earning_building1_first_floor + earning_building1_second_floor + earning_building1_third_floor + earning_building1_fourth_floor

def earning_building2_first_floor : ℝ := 7 * 20 * 0.9
def earning_building2_second_floor : ℝ := (25 + 30 + 35 + 40 + 45 + 50 + 55 + 60) * 0.7
def earning_building2_third_floor : ℝ := 6 * 60 * 0.6
def earnings_building2 : ℝ := earning_building2_first_floor + earning_building2_second_floor + earning_building2_third_floor

def total_earnings : ℝ := earnings_building1 + earnings_building2

theorem krystiana_earnings : total_earnings = 1091.5 := by
  sorry

end NUMINAMATH_GPT_krystiana_earnings_l55_5561


namespace NUMINAMATH_GPT_length_of_segment_BD_is_sqrt_3_l55_5580

open Real

-- Define the triangle ABC and the point D according to the problem conditions
def triangle_ABC (A B C : ℝ × ℝ) :=
  B.1 = 0 ∧ B.2 = 0 ∧
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = 3 ∧
  (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = 7 ∧
  C.2 = 0 ∧ (A.1 - C.1) ^ 2 + A.2 ^ 2 = 10

def point_D (A B C D : ℝ × ℝ) :=
  ∃ BD DC : ℝ, BD + DC = sqrt 7 ∧
  BD / DC = sqrt 3 / sqrt 7 ∧
  D.1 = BD / sqrt 7 ∧ D.2 = 0

-- The theorem to prove
theorem length_of_segment_BD_is_sqrt_3 (A B C D : ℝ × ℝ)
  (h₁ : triangle_ABC A B C)
  (h₂ : point_D A B C D) :
  (sqrt ((D.1 - B.1) ^ 2 + (D.2 - B.2) ^ 2)) = sqrt 3 :=
sorry

end NUMINAMATH_GPT_length_of_segment_BD_is_sqrt_3_l55_5580


namespace NUMINAMATH_GPT_remainder_of_sum_of_primes_mod_eighth_prime_l55_5545

def sum_first_seven_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13 + 17

def eighth_prime : ℕ := 19

theorem remainder_of_sum_of_primes_mod_eighth_prime : sum_first_seven_primes % eighth_prime = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_primes_mod_eighth_prime_l55_5545


namespace NUMINAMATH_GPT_num_pairs_sold_l55_5551

theorem num_pairs_sold : 
  let total_amount : ℤ := 735
  let avg_price : ℝ := 9.8
  let num_pairs : ℝ := total_amount / avg_price
  num_pairs = 75 :=
by
  let total_amount : ℤ := 735
  let avg_price : ℝ := 9.8
  let num_pairs : ℝ := total_amount / avg_price
  exact sorry

end NUMINAMATH_GPT_num_pairs_sold_l55_5551


namespace NUMINAMATH_GPT_Winnie_keeps_lollipops_l55_5530

-- Definitions based on the conditions provided
def total_lollipops : ℕ := 60 + 135 + 5 + 250
def number_of_friends : ℕ := 12

-- The theorem statement we need to prove
theorem Winnie_keeps_lollipops : total_lollipops % number_of_friends = 6 :=
by
  -- proof omitted as instructed
  sorry

end NUMINAMATH_GPT_Winnie_keeps_lollipops_l55_5530


namespace NUMINAMATH_GPT_inequality_abc_l55_5585

open Real

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / sqrt (a^2 + 8 * b * c)) + (b / sqrt (b^2 + 8 * c * a)) + (c / sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end NUMINAMATH_GPT_inequality_abc_l55_5585


namespace NUMINAMATH_GPT_union_of_sets_l55_5542

theorem union_of_sets (A B : Set α) : A ∪ B = { x | x ∈ A ∨ x ∈ B } :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l55_5542


namespace NUMINAMATH_GPT_sum_of_roots_eq_three_l55_5534

theorem sum_of_roots_eq_three {a b : ℝ} (h₁ : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (11 - a) = 0)
  (h₂ : a * 2^3 + (a + 3 * b) * 2^2 + (b - 4 * a) * 2 + (11 - a) = 0)
  (h₃ : a * 4^3 + (a + 3 * b) * 4^2 + (b - 4 * a) * 4 + (11 - a) = 0) :
  (-3) + 2 + 4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_three_l55_5534


namespace NUMINAMATH_GPT_blue_balls_initial_count_l55_5507

theorem blue_balls_initial_count (B : ℕ)
  (h1 : 15 - 3 = 12)
  (h2 : (B - 3) / 12 = 1 / 3) :
  B = 7 :=
sorry

end NUMINAMATH_GPT_blue_balls_initial_count_l55_5507


namespace NUMINAMATH_GPT_fraction_of_repeating_decimal_l55_5533

-- Definitions corresponding to the problem conditions
def a : ℚ := 56 / 100
def r : ℚ := 1 / 100
def infinite_geom_sum (a r : ℚ) := a / (1 - r)

-- The statement we need to prove
theorem fraction_of_repeating_decimal :
  infinite_geom_sum a r = 56 / 99 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_repeating_decimal_l55_5533


namespace NUMINAMATH_GPT_slope_positive_if_and_only_if_l55_5532

/-- Given points A(2, 1) and B(1, m^2), the slope of the line passing through them is positive,
if and only if m is in the range -1 < m < 1. -/
theorem slope_positive_if_and_only_if
  (m : ℝ) : 1 - m^2 > 0 ↔ -1 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_slope_positive_if_and_only_if_l55_5532


namespace NUMINAMATH_GPT_sum_of_squares_l55_5571

theorem sum_of_squares (b j s : ℕ) 
  (h1 : 3 * b + 2 * j + 4 * s = 70)
  (h2 : 4 * b + 3 * j + 2 * s = 88) : 
  b^2 + j^2 + s^2 = 405 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_l55_5571


namespace NUMINAMATH_GPT_mod_inverse_13_997_l55_5531

-- The theorem statement
theorem mod_inverse_13_997 : ∃ x : ℕ, 0 ≤ x ∧ x < 997 ∧ (13 * x) % 997 = 1 ∧ x = 767 := 
by
  sorry

end NUMINAMATH_GPT_mod_inverse_13_997_l55_5531


namespace NUMINAMATH_GPT_exists_decreasing_lcm_sequence_l55_5581

theorem exists_decreasing_lcm_sequence :
  ∃ (a : Fin 100 → ℕ), 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i : Fin 99, Nat.lcm (a i) (a (i + 1)) > Nat.lcm (a (i + 1)) (a (i + 2))) :=
sorry

end NUMINAMATH_GPT_exists_decreasing_lcm_sequence_l55_5581


namespace NUMINAMATH_GPT_smaller_two_digit_product_is_34_l55_5518

theorem smaller_two_digit_product_is_34 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 5082) : min a b = 34 :=
by
  sorry

end NUMINAMATH_GPT_smaller_two_digit_product_is_34_l55_5518


namespace NUMINAMATH_GPT_area_ratio_of_circles_l55_5513

theorem area_ratio_of_circles (R_C R_D : ℝ) (hL : (60.0 / 360.0) * 2.0 * Real.pi * R_C = (40.0 / 360.0) * 2.0 * Real.pi * R_D) : 
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4.0 / 9.0 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_circles_l55_5513


namespace NUMINAMATH_GPT_parabola_vertex_coordinates_l55_5595

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), y = (x - 2)^2 ∧ (x, y) = (2, 0) :=
sorry

end NUMINAMATH_GPT_parabola_vertex_coordinates_l55_5595


namespace NUMINAMATH_GPT_molecularWeight_correct_l55_5562

noncomputable def molecularWeight (nC nH nO nN: ℤ) 
    (wC wH wO wN : ℚ) : ℚ := nC * wC + nH * wH + nO * wO + nN * wN

theorem molecularWeight_correct : 
    molecularWeight 5 12 3 1 12.01 1.008 16.00 14.01 = 134.156 := by
  sorry

end NUMINAMATH_GPT_molecularWeight_correct_l55_5562


namespace NUMINAMATH_GPT_baron_not_lying_l55_5579

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem baron_not_lying : 
  ∃ a b : ℕ, 
  (a ≠ b ∧ a ≥ 10^9 ∧ a < 10^10 ∧ b ≥ 10^9 ∧ b < 10^10 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a + sum_of_digits (a * a) = b + sum_of_digits (b * b))) :=
  sorry

end NUMINAMATH_GPT_baron_not_lying_l55_5579


namespace NUMINAMATH_GPT_new_number_formed_l55_5586

theorem new_number_formed (h t u : ℕ) (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) :
  let original_number := 100 * h + 10 * t + u
  let new_number := 2000 + 10 * original_number
  new_number = 1000 * (h + 2) + 100 * t + 10 * u :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_new_number_formed_l55_5586
