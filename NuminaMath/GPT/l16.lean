import Mathlib

namespace NUMINAMATH_GPT_fraction_red_after_tripling_l16_1628

-- Define the initial conditions
def initial_fraction_blue : ℚ := 4 / 7
def initial_fraction_red : ℚ := 1 - initial_fraction_blue
def triple_red_fraction (initial_red : ℚ) : ℚ := 3 * initial_red

-- Theorem statement
theorem fraction_red_after_tripling :
  let x := 1 -- Any number since it will cancel out
  let initial_red_marble := initial_fraction_red * x
  let total_marble := x
  let new_red_marble := triple_red_fraction initial_red_marble
  let new_total_marble := initial_fraction_blue * x + new_red_marble
  (new_red_marble / new_total_marble) = 9 / 13 :=
by
  sorry

end NUMINAMATH_GPT_fraction_red_after_tripling_l16_1628


namespace NUMINAMATH_GPT_Robert_ate_10_chocolates_l16_1678

def chocolates_eaten_by_Nickel : Nat := 5
def difference_between_Robert_and_Nickel : Nat := 5
def chocolates_eaten_by_Robert := chocolates_eaten_by_Nickel + difference_between_Robert_and_Nickel

theorem Robert_ate_10_chocolates : chocolates_eaten_by_Robert = 10 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Robert_ate_10_chocolates_l16_1678


namespace NUMINAMATH_GPT_smallest_number_of_set_s_l16_1616

theorem smallest_number_of_set_s : 
  ∀ (s : Set ℕ),
    (∃ n : ℕ, s = {k | ∃ m : ℕ, k = 5 * (m+n) ∧ m < 45}) ∧ 
    (275 ∈ s) → 
      (∃ min_elem : ℕ, min_elem ∈ s ∧ min_elem = 55) 
  :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_set_s_l16_1616


namespace NUMINAMATH_GPT_color_set_no_arith_prog_same_color_l16_1645

def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 1987}

def colors : Fin 4 := sorry  -- Color indexing set (0, 1, 2, 3)

def valid_coloring (c : ℕ → Fin 4) : Prop :=
  ∀ (a d : ℕ) (h₁ : a ∈ M) (h₂ : d ≠ 0) (h₃ : ∀ k, a + k * d ∈ M ∧ k < 10), 
  ¬ ∀ k, c (a + k * d) = c a

theorem color_set_no_arith_prog_same_color :
  ∃ (c : ℕ → Fin 4), valid_coloring c :=
sorry

end NUMINAMATH_GPT_color_set_no_arith_prog_same_color_l16_1645


namespace NUMINAMATH_GPT_age_of_oldest_child_l16_1636

theorem age_of_oldest_child
  (a b c d : ℕ)
  (h1 : a = 6)
  (h2 : b = 8)
  (h3 : c = 10)
  (h4 : (a + b + c + d) / 4 = 9) :
  d = 12 :=
sorry

end NUMINAMATH_GPT_age_of_oldest_child_l16_1636


namespace NUMINAMATH_GPT_no_real_solution_l16_1667

theorem no_real_solution : ∀ x : ℝ, ¬ ((2*x - 3*x + 7)^2 + 4 = -|2*x|) :=
by
  intro x
  have h1 : (2*x - 3*x + 7)^2 + 4 ≥ 4 := by
    sorry
  have h2 : -|2*x| ≤ 0 := by
    sorry
  -- The main contradiction follows from comparing h1 and h2
  sorry

end NUMINAMATH_GPT_no_real_solution_l16_1667


namespace NUMINAMATH_GPT_initial_persons_count_is_eight_l16_1691

noncomputable def number_of_persons_initially 
  (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : ℝ := 
  (new_weight - old_weight) / avg_increase

theorem initial_persons_count_is_eight 
  (avg_increase : ℝ := 2.5) (old_weight : ℝ := 60) (new_weight : ℝ := 80) : 
  number_of_persons_initially avg_increase old_weight new_weight = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_persons_count_is_eight_l16_1691


namespace NUMINAMATH_GPT_exists_sequence_satisfying_conditions_l16_1659

def F : ℕ → ℕ := sorry

theorem exists_sequence_satisfying_conditions :
  (∀ n, ∃ k, F k = n) ∧ 
  (∀ n, ∃ m > n, F m = n) ∧ 
  (∀ n ≥ 2, F (F (n ^ 163)) = F (F n) + F (F 361)) :=
sorry

end NUMINAMATH_GPT_exists_sequence_satisfying_conditions_l16_1659


namespace NUMINAMATH_GPT_tuning_day_method_pi_l16_1653

variable (x : ℝ)

-- Initial bounds and approximations
def initial_bounds (π : ℝ) := 31 / 10 < π ∧ π < 49 / 15

-- Definition of the "Tuning Day Method"
def tuning_day_method (a b c d : ℕ) (a' b' : ℝ) := a' = (b + d) / (a + c)

theorem tuning_day_method_pi :
  ∀ π : ℝ, initial_bounds π →
  (31 / 10 < π ∧ π < 16 / 5) ∧ 
  (47 / 15 < π ∧ π < 63 / 20) ∧
  (47 / 15 < π ∧ π < 22 / 7) →
  22 / 7 = 22 / 7 :=
by
  sorry

end NUMINAMATH_GPT_tuning_day_method_pi_l16_1653


namespace NUMINAMATH_GPT_quadratic_real_root_m_l16_1661

theorem quadratic_real_root_m (m : ℝ) (h : 4 - 4 * m ≥ 0) : m = 0 ∨ m = 2 ∨ m = 4 ∨ m = 6 ↔ m = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_root_m_l16_1661


namespace NUMINAMATH_GPT_midpoint_coordinates_l16_1650

theorem midpoint_coordinates (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 2) (hy1 : y1 = 9) (hx2 : x2 = 8) (hy2 : y2 = 3) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx, my) = (5, 6) :=
by
  rw [hx1, hy1, hx2, hy2]
  sorry

end NUMINAMATH_GPT_midpoint_coordinates_l16_1650


namespace NUMINAMATH_GPT_find_english_score_l16_1693

-- Define the scores
def M : ℕ := 82
def K : ℕ := M + 5
variable (E : ℕ)

-- The average score condition
axiom avg_condition : (K + E + M) / 3 = 89

-- Our goal is to prove that E = 98
theorem find_english_score : E = 98 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_find_english_score_l16_1693


namespace NUMINAMATH_GPT_proof_problem_l16_1658

variable {a b : ℝ}

theorem proof_problem (h₁ : a < b) (h₂ : b < 0) : (b/a) + (a/b) > 2 :=
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l16_1658


namespace NUMINAMATH_GPT_total_spending_l16_1697

-- Define the condition of spending for each day
def friday_spending : ℝ := 20
def saturday_spending : ℝ := 2 * friday_spending
def sunday_spending : ℝ := 3 * friday_spending

-- Define the statement to be proven
theorem total_spending : friday_spending + saturday_spending + sunday_spending = 120 :=
by
  -- Provide conditions and calculations here (if needed)
  sorry

end NUMINAMATH_GPT_total_spending_l16_1697


namespace NUMINAMATH_GPT_problem1_problem2_l16_1639

-- For Problem (1)
theorem problem1 (x : ℝ) : 2 * x - 3 > x + 1 → x > 4 := 
by sorry

-- For Problem (2)
theorem problem2 (a b : ℝ) (h : a^2 + 3 * a * b = 5) : (a + b) * (a + 2 * b) - 2 * b^2 = 5 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l16_1639


namespace NUMINAMATH_GPT_factorization_analysis_l16_1623

variable (a b c : ℝ)

theorem factorization_analysis : a^2 - 2 * a * b + b^2 - c^2 = (a - b + c) * (a - b - c) := 
sorry

end NUMINAMATH_GPT_factorization_analysis_l16_1623


namespace NUMINAMATH_GPT_contrapositive_l16_1625

variable {α : Type} (M : α → Prop) (a b : α)

theorem contrapositive (h : (M a → ¬ M b)) : (M b → ¬ M a) := 
by
  sorry

end NUMINAMATH_GPT_contrapositive_l16_1625


namespace NUMINAMATH_GPT_pencils_placed_by_Joan_l16_1643

variable (initial_pencils : ℕ)
variable (total_pencils : ℕ)

theorem pencils_placed_by_Joan 
  (h1 : initial_pencils = 33) 
  (h2 : total_pencils = 60)
  : total_pencils - initial_pencils = 27 := 
by
  sorry

end NUMINAMATH_GPT_pencils_placed_by_Joan_l16_1643


namespace NUMINAMATH_GPT_find_2a_minus_3b_l16_1688

theorem find_2a_minus_3b
  (a b : ℝ)
  (h1 : a * 2 - b * 1 = 4)
  (h2 : a * 2 + b * 1 = 2) :
  2 * a - 3 * b = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_2a_minus_3b_l16_1688


namespace NUMINAMATH_GPT_number_of_children_l16_1654

-- Definition of the conditions
def pencils_per_child : ℕ := 2
def total_pencils : ℕ := 30

-- Theorem statement
theorem number_of_children (n : ℕ) (h1 : pencils_per_child = 2) (h2 : total_pencils = 30) :
  n = total_pencils / pencils_per_child :=
by
  have h : n = 30 / 2 := sorry
  exact h

end NUMINAMATH_GPT_number_of_children_l16_1654


namespace NUMINAMATH_GPT_neg_prop_p_l16_1611

theorem neg_prop_p :
  (¬ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_prop_p_l16_1611


namespace NUMINAMATH_GPT_hexagon_bc_de_eq_14_l16_1698

theorem hexagon_bc_de_eq_14
  (α β γ δ ε ζ : ℝ)
  (angle_cond : α = β ∧ β = γ ∧ γ = δ ∧ δ = ε ∧ ε = ζ)
  (AB BC CD DE EF FA : ℝ)
  (sum_AB_BC : AB + BC = 11)
  (diff_FA_CD : FA - CD = 3)
  : BC + DE = 14 := sorry

end NUMINAMATH_GPT_hexagon_bc_de_eq_14_l16_1698


namespace NUMINAMATH_GPT_polar_to_rectangular_l16_1676

theorem polar_to_rectangular :
  ∀ (r θ : ℝ), r = 3 * Real.sqrt 2 → θ = (3 * Real.pi) / 4 → 
  (r * Real.cos θ, r * Real.sin θ) = (-3, 3) :=
by
  intro r θ hr hθ
  rw [hr, hθ]
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l16_1676


namespace NUMINAMATH_GPT_arccos_cos_three_l16_1615

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := 
  sorry

end NUMINAMATH_GPT_arccos_cos_three_l16_1615


namespace NUMINAMATH_GPT_arithmetic_series_sum_l16_1638

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 56
  let n := 19
  let pairs_sum := (n-1) / 2 * (-3)
  let single_term := 56
  2 - 5 + 8 - 11 + 14 - 17 + 20 - 23 + 26 - 29 + 32 - 35 + 38 - 41 + 44 - 47 + 50 - 53 + 56 = 29 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l16_1638


namespace NUMINAMATH_GPT_solve_for_k_l16_1646

theorem solve_for_k : {k : ℕ | ∀ x : ℝ, (x^2 - 1)^(2*k) + (x^2 + 2*x)^(2*k) + (2*x + 1)^(2*k) = 2*(1 + x + x^2)^(2*k)} = {1, 2} :=
sorry

end NUMINAMATH_GPT_solve_for_k_l16_1646


namespace NUMINAMATH_GPT_weight_order_l16_1626

variable {P Q R S T : ℕ}

theorem weight_order
    (h1 : Q + S = 1200)
    (h2 : R + T = 2100)
    (h3 : Q + T = 800)
    (h4 : Q + R = 900)
    (h5 : P + T = 700)
    (hP : P < 1000)
    (hQ : Q < 1000)
    (hR : R < 1000)
    (hS : S < 1000)
    (hT : T < 1000) :
  S > R ∧ R > T ∧ T > Q ∧ Q > P :=
sorry

end NUMINAMATH_GPT_weight_order_l16_1626


namespace NUMINAMATH_GPT_compute_pounds_of_cotton_l16_1633

theorem compute_pounds_of_cotton (x : ℝ) :
  (5 * 30 + 10 * x = 640) → (x = 49) := by
  intro h
  sorry

end NUMINAMATH_GPT_compute_pounds_of_cotton_l16_1633


namespace NUMINAMATH_GPT_projectiles_meet_in_90_minutes_l16_1652

theorem projectiles_meet_in_90_minutes
  (d : ℝ) (v1 : ℝ) (v2 : ℝ) (time_in_minutes : ℝ)
  (h_d : d = 1455)
  (h_v1 : v1 = 470)
  (h_v2 : v2 = 500)
  (h_time : time_in_minutes = 90) :
  d / (v1 + v2) * 60 = time_in_minutes :=
by
  sorry

end NUMINAMATH_GPT_projectiles_meet_in_90_minutes_l16_1652


namespace NUMINAMATH_GPT_solve_inequality_l16_1669

noncomputable def inequality_solution : Set ℝ :=
  { x | x^2 / (x + 2) ≥ 3 / (x - 2) + 7 / 4 }

theorem solve_inequality :
  inequality_solution = { x | -2 < x ∧ x < 2 } ∪ { x | 3 ≤ x } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l16_1669


namespace NUMINAMATH_GPT_prism_height_relation_l16_1609

theorem prism_height_relation (a b c h : ℝ) 
  (h_perp : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_height : 0 < h) 
  (h_right_angles : true) :
  1 / h^2 = 1 / a^2 + 1 / b^2 + 1 / c^2 :=
by 
  sorry 

end NUMINAMATH_GPT_prism_height_relation_l16_1609


namespace NUMINAMATH_GPT_number_of_routes_l16_1655

structure RailwayStation :=
  (A B C D E F G H I J K L M : ℕ)

def initialize_station : RailwayStation :=
  ⟨1, 1, 1, 1, 2, 2, 3, 3, 3, 6, 9, 9, 18⟩

theorem number_of_routes (station : RailwayStation) : station.M = 18 :=
  by sorry

end NUMINAMATH_GPT_number_of_routes_l16_1655


namespace NUMINAMATH_GPT_crate_minimum_dimension_l16_1680

theorem crate_minimum_dimension (a : ℕ) (h1 : a ≥ 12) :
  min a (min 8 12) = 8 :=
by
  sorry

end NUMINAMATH_GPT_crate_minimum_dimension_l16_1680


namespace NUMINAMATH_GPT_calculate_total_money_l16_1604

noncomputable def cost_per_gumdrop : ℕ := 4
noncomputable def number_of_gumdrops : ℕ := 20
noncomputable def total_money : ℕ := 80

theorem calculate_total_money : 
  cost_per_gumdrop * number_of_gumdrops = total_money := 
by
  sorry

end NUMINAMATH_GPT_calculate_total_money_l16_1604


namespace NUMINAMATH_GPT_number_of_arrangements_is_48_l16_1657

noncomputable def number_of_arrangements (students : List String) (boy_not_at_ends : String) (adjacent_girls : List String) : Nat :=
  sorry

theorem number_of_arrangements_is_48 : number_of_arrangements ["A", "B1", "B2", "G1", "G2", "G3"] "B1" ["G1", "G2", "G3"] = 48 :=
by
  sorry

end NUMINAMATH_GPT_number_of_arrangements_is_48_l16_1657


namespace NUMINAMATH_GPT_sum_of_altitudes_of_triangle_l16_1607

open Real

noncomputable def sum_of_altitudes (a b c : ℝ) : ℝ :=
  let inter_x := -c / a
  let inter_y := -c / b
  let vertex1 := (inter_x, 0)
  let vertex2 := (0, inter_y)
  let vertex3 := (0, 0)
  let area_triangle := (1 / 2) * abs (inter_x * inter_y)
  let altitude_x := abs inter_x
  let altitude_y := abs inter_y
  let altitude_line := abs c / sqrt (a ^ 2 + b ^ 2)
  altitude_x + altitude_y + altitude_line

theorem sum_of_altitudes_of_triangle :
  sum_of_altitudes 15 6 90 = 21 + 10 * sqrt (1 / 29) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_altitudes_of_triangle_l16_1607


namespace NUMINAMATH_GPT_markup_percentage_l16_1630

-- Definitions coming from conditions
variables (C : ℝ) (M : ℝ) (S : ℝ)
-- Markup formula
def markup_formula : Prop := M = 0.10 * C
-- Selling price formula
def selling_price_formula : Prop := S = C + M

-- Given the conditions, we need to prove that the markup is 9.09% of the selling price
theorem markup_percentage (h1 : markup_formula C M) (h2 : selling_price_formula C M S) :
  (M / S) * 100 = 9.09 :=
sorry

end NUMINAMATH_GPT_markup_percentage_l16_1630


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l16_1648

theorem arithmetic_sequence_common_difference
  (a_n : ℕ → ℤ) (h_arithmetic : ∀ n, (a_n (n + 1) = a_n n + d)) 
  (h_sum1 : a_n 1 + a_n 3 + a_n 5 = 105)
  (h_sum2 : a_n 2 + a_n 4 + a_n 6 = 99) : 
  d = -2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l16_1648


namespace NUMINAMATH_GPT_product_of_fractions_l16_1649

theorem product_of_fractions :
  (1 / 5) * (3 / 7) = 3 / 35 :=
sorry

end NUMINAMATH_GPT_product_of_fractions_l16_1649


namespace NUMINAMATH_GPT_measure_of_unknown_angle_in_hexagon_l16_1642

theorem measure_of_unknown_angle_in_hexagon :
  let a1 := 135
  let a2 := 105
  let a3 := 87
  let a4 := 120
  let a5 := 78
  let total_internal_angles := 180 * (6 - 2)
  let known_sum := a1 + a2 + a3 + a4 + a5
  let Q := total_internal_angles - known_sum
  Q = 195 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_unknown_angle_in_hexagon_l16_1642


namespace NUMINAMATH_GPT_find_a9_l16_1686

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
def a_n : ℕ → ℝ := sorry   -- The sequence itself is unknown initially.

axiom a3 : a_n 3 = 5
axiom a4_a8 : a_n 4 + a_n 8 = 22

theorem find_a9 : a_n 9 = 41 :=
by
  sorry

end NUMINAMATH_GPT_find_a9_l16_1686


namespace NUMINAMATH_GPT_responses_needed_750_l16_1666

section Responses
  variable (q_min : ℕ) (response_rate : ℝ)

  def responses_needed : ℝ := response_rate * q_min

  theorem responses_needed_750 (h1 : q_min = 1250) (h2 : response_rate = 0.60) : responses_needed q_min response_rate = 750 :=
  by
    simp [responses_needed, h1, h2]
    sorry
end Responses

end NUMINAMATH_GPT_responses_needed_750_l16_1666


namespace NUMINAMATH_GPT_range_of_independent_variable_l16_1624

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y : ℝ, y = 2 * x / (x - 1)) ↔ x ≠ 1 :=
by sorry

end NUMINAMATH_GPT_range_of_independent_variable_l16_1624


namespace NUMINAMATH_GPT_angles_in_interval_l16_1601

open Real

theorem angles_in_interval
    (θ : ℝ)
    (hθ : 0 ≤ θ ∧ θ ≤ 2 * π)
    (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 * sin θ - x * (2 - x) + (2 - x)^2 * cos θ > 0) :
  π / 12 < θ ∧ θ < 5 * π / 12 :=
by
  sorry

end NUMINAMATH_GPT_angles_in_interval_l16_1601


namespace NUMINAMATH_GPT_expand_expression_l16_1627

theorem expand_expression (x : ℝ) : 3 * (x - 6) * (x - 7) = 3 * x^2 - 39 * x + 126 := by
  sorry

end NUMINAMATH_GPT_expand_expression_l16_1627


namespace NUMINAMATH_GPT_maximize_S_l16_1694

noncomputable def a (n: ℕ) : ℝ := 24 - 2 * n

noncomputable def S (n: ℕ) : ℝ := -n^2 + 23 * n

theorem maximize_S (n : ℕ) : 
  (n = 11 ∨ n = 12) → ∀ m : ℕ, m ≠ 11 ∧ m ≠ 12 → S m ≤ S n :=
sorry

end NUMINAMATH_GPT_maximize_S_l16_1694


namespace NUMINAMATH_GPT_total_pieces_of_tomatoes_l16_1687

namespace FarmerTomatoes

variables (rows plants_per_row yield_per_plant : ℕ)

def total_plants (rows plants_per_row : ℕ) := rows * plants_per_row

def total_tomatoes (total_plants yield_per_plant : ℕ) := total_plants * yield_per_plant

theorem total_pieces_of_tomatoes 
  (hrows : rows = 30)
  (hplants_per_row : plants_per_row = 10)
  (hyield_per_plant : yield_per_plant = 20) :
  total_tomatoes (total_plants rows plants_per_row) yield_per_plant = 6000 :=
by
  rw [hrows, hplants_per_row, hyield_per_plant]
  unfold total_plants total_tomatoes
  norm_num
  done

end FarmerTomatoes

end NUMINAMATH_GPT_total_pieces_of_tomatoes_l16_1687


namespace NUMINAMATH_GPT_miles_run_by_harriet_l16_1622

def miles_run_by_all_runners := 285
def miles_run_by_katarina := 51
def miles_run_by_adriana := 74
def miles_run_by_tomas_tyler_harriet (total_run: ℝ) := (total_run - (miles_run_by_katarina + miles_run_by_adriana))

theorem miles_run_by_harriet : (miles_run_by_tomas_tyler_harriet miles_run_by_all_runners) / 3 = 53.33 := by
  sorry

end NUMINAMATH_GPT_miles_run_by_harriet_l16_1622


namespace NUMINAMATH_GPT_sales_percentage_l16_1673

theorem sales_percentage (pens_sales pencils_sales notebooks_sales : ℕ) 
  (h1 : pens_sales = 25)
  (h2 : pencils_sales = 20)
  (h3 : notebooks_sales = 30) :
  100 - (pens_sales + pencils_sales + notebooks_sales) = 25 :=
by
  sorry

end NUMINAMATH_GPT_sales_percentage_l16_1673


namespace NUMINAMATH_GPT_smallest_possible_n_l16_1685

theorem smallest_possible_n (n : ℕ) (h_pos: n > 0)
  (h_int: (1/3 : ℚ) + 1/4 + 1/9 + 1/n = (1:ℚ)) : 
  n = 18 :=
sorry

end NUMINAMATH_GPT_smallest_possible_n_l16_1685


namespace NUMINAMATH_GPT_Peter_finishes_all_tasks_at_5_30_PM_l16_1620

-- Definitions representing the initial conditions
def start_time : ℕ := 9 * 60 -- 9:00 AM in minutes
def third_task_completion_time : ℕ := 11 * 60 + 30 -- 11:30 AM in minutes
def task_durations : List ℕ :=
  [30, 30, 60, 120, 240] -- Durations of the 5 tasks in minutes
  
-- Statement for the proof problem
theorem Peter_finishes_all_tasks_at_5_30_PM :
  let total_duration := task_durations.sum 
  let finish_time := start_time + total_duration
  finish_time = 17 * 60 + 30 := -- 5:30 PM in minutes
  sorry

end NUMINAMATH_GPT_Peter_finishes_all_tasks_at_5_30_PM_l16_1620


namespace NUMINAMATH_GPT_intersection_of_lines_l16_1665

theorem intersection_of_lines : ∃ (x y : ℚ), y = -3 * x + 1 ∧ y + 5 = 15 * x - 2 ∧ x = 1 / 3 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l16_1665


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l16_1682

theorem arithmetic_geometric_sequence (a1 d : ℝ) (h1 : a1 = 1) (h2 : d ≠ 0) (h_geom : (a1 + d) ^ 2 = a1 * (a1 + 4 * d)) :
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l16_1682


namespace NUMINAMATH_GPT_num_solutions_20_l16_1683

-- Define the number of integer solutions function
def num_solutions (n : ℕ) : ℕ := 4 * n

-- Given conditions
axiom h1 : num_solutions 1 = 4
axiom h2 : num_solutions 2 = 8

-- Theorem to prove the number of solutions for |x| + |y| = 20 is 80
theorem num_solutions_20 : num_solutions 20 = 80 :=
by sorry

end NUMINAMATH_GPT_num_solutions_20_l16_1683


namespace NUMINAMATH_GPT_min_value_proof_l16_1619

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a^2 + 5 * a + 2) * (b^2 + 5 * b + 2) * (c^2 + 5 * c + 2) / (a * b * c)

theorem min_value_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min_value a b c ≥ 343 :=
sorry

end NUMINAMATH_GPT_min_value_proof_l16_1619


namespace NUMINAMATH_GPT_equivalent_problem_l16_1634

def f (x : ℤ) : ℤ := 9 - x

def g (x : ℤ) : ℤ := x - 9

theorem equivalent_problem : g (f 15) = -15 := sorry

end NUMINAMATH_GPT_equivalent_problem_l16_1634


namespace NUMINAMATH_GPT_Yoongi_class_students_l16_1695

theorem Yoongi_class_students (Total_a Total_b Total_ab : ℕ)
  (h1 : Total_a = 18)
  (h2 : Total_b = 24)
  (h3 : Total_ab = 7)
  (h4 : Total_a + Total_b - Total_ab = 35) : 
  Total_a + Total_b - Total_ab = 35 :=
sorry

end NUMINAMATH_GPT_Yoongi_class_students_l16_1695


namespace NUMINAMATH_GPT_cube_root_of_5_irrational_l16_1617

theorem cube_root_of_5_irrational : ¬ ∃ (a b : ℚ), (b ≠ 0) ∧ (a / b)^3 = 5 := 
by
  sorry

end NUMINAMATH_GPT_cube_root_of_5_irrational_l16_1617


namespace NUMINAMATH_GPT_complex_cube_root_identity_l16_1631

theorem complex_cube_root_identity (a b c : ℂ) (ω : ℂ)
  (h1 : ω^3 = 1)
  (h2 : 1 + ω + ω^2 = 0) :
  (a + b * ω + c * ω^2) * (a + b * ω^2 + c * ω) = a^2 + b^2 + c^2 - ab - ac - bc :=
by
  sorry

end NUMINAMATH_GPT_complex_cube_root_identity_l16_1631


namespace NUMINAMATH_GPT_fraction_unchanged_l16_1664

-- Define the digit rotation
def rotate (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => d  -- for completeness, though we assume d only takes {0, 1, 6, 8, 9}

-- Define the condition for a fraction to be unchanged when flipped
def unchanged_when_flipped (numerator denominator : ℕ) : Prop :=
  let rotated_numerator := rotate numerator
  let rotated_denominator := rotate denominator
  rotated_numerator * denominator = rotated_denominator * numerator

-- Define the specific fraction 6/9
def specific_fraction_6_9 : Prop :=
  unchanged_when_flipped 6 9 ∧ 6 < 9

-- Theorem stating 6/9 is unchanged when its digits are flipped and it's a valid fraction
theorem fraction_unchanged : specific_fraction_6_9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_unchanged_l16_1664


namespace NUMINAMATH_GPT_combined_percentage_tennis_is_31_l16_1603

-- Define the number of students at North High School
def students_north : ℕ := 1800

-- Define the number of students at South Elementary School
def students_south : ℕ := 2200

-- Define the percentage of students who prefer tennis at North High School
def percentage_tennis_north : ℚ := 25/100

-- Define the percentage of students who prefer tennis at South Elementary School
def percentage_tennis_south : ℚ := 35/100

-- Calculate the number of students who prefer tennis at North High School
def tennis_students_north : ℚ := students_north * percentage_tennis_north

-- Calculate the number of students who prefer tennis at South Elementary School
def tennis_students_south : ℚ := students_south * percentage_tennis_south

-- Calculate the total number of students who prefer tennis in both schools
def total_tennis_students : ℚ := tennis_students_north + tennis_students_south

-- Calculate the total number of students in both schools
def total_students : ℚ := students_north + students_south

-- Calculate the combined percentage of students who prefer tennis
def combined_percentage_tennis : ℚ := (total_tennis_students / total_students) * 100

-- Main statement to prove
theorem combined_percentage_tennis_is_31 :
  round combined_percentage_tennis = 31 := by sorry

end NUMINAMATH_GPT_combined_percentage_tennis_is_31_l16_1603


namespace NUMINAMATH_GPT_revenue_difference_l16_1621

theorem revenue_difference {x z : ℕ} (hx : 10 ≤ x ∧ x ≤ 96) (hz : z = x + 3) :
  1000 * z + 10 * x - (1000 * x + 10 * z) = 2920 :=
by
  sorry

end NUMINAMATH_GPT_revenue_difference_l16_1621


namespace NUMINAMATH_GPT_hall_width_length_ratio_l16_1699

theorem hall_width_length_ratio 
  (w l : ℝ) 
  (h1 : w * l = 128) 
  (h2 : l - w = 8) : 
  w / l = 1 / 2 := 
by sorry

end NUMINAMATH_GPT_hall_width_length_ratio_l16_1699


namespace NUMINAMATH_GPT_basketball_game_half_points_l16_1641

noncomputable def eagles_geometric_sequence (a r : ℕ) (n : ℕ) : ℕ :=
  a * r ^ n

noncomputable def lions_arithmetic_sequence (b d : ℕ) (n : ℕ) : ℕ :=
  b + n * d

noncomputable def total_first_half_points (a r b d : ℕ) : ℕ :=
  eagles_geometric_sequence a r 0 + eagles_geometric_sequence a r 1 +
  lions_arithmetic_sequence b d 0 + lions_arithmetic_sequence b d 1

theorem basketball_game_half_points (a r b d : ℕ) (h1 : a + a * r = b + (b + d)) (h2 : a + a * r + a * r^2 + a * r^3 = b + (b + d) + (b + 2*d) + (b + 3*d)) :
  total_first_half_points a r b d = 8 :=
by sorry

end NUMINAMATH_GPT_basketball_game_half_points_l16_1641


namespace NUMINAMATH_GPT_total_children_on_bus_after_stop_l16_1675

theorem total_children_on_bus_after_stop (initial : ℕ) (additional : ℕ) (total : ℕ) 
  (h1 : initial = 18) (h2 : additional = 7) : total = 25 :=
by sorry

end NUMINAMATH_GPT_total_children_on_bus_after_stop_l16_1675


namespace NUMINAMATH_GPT_find_f_2015_l16_1612

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2015
  (h1 : ∀ x, f (-x) = -f x) -- f is an odd function
  (h2 : ∀ x, f (x + 2) = -f x) -- f(x+2) = -f(x)
  (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) -- f(x) = 2x^2 for x in (0, 2)
  : f 2015 = -2 :=
sorry

end NUMINAMATH_GPT_find_f_2015_l16_1612


namespace NUMINAMATH_GPT_books_bought_l16_1670

theorem books_bought (math_price : ℕ) (hist_price : ℕ) (total_cost : ℕ) (math_books : ℕ) (hist_books : ℕ) 
  (H : math_price = 4) (H1 : hist_price = 5) (H2 : total_cost = 396) (H3 : math_books = 54) 
  (H4 : math_books * math_price + hist_books * hist_price = total_cost) :
  math_books + hist_books = 90 :=
by sorry

end NUMINAMATH_GPT_books_bought_l16_1670


namespace NUMINAMATH_GPT_floor_S_proof_l16_1651

noncomputable def floor_S (a b c d: ℝ) : ℝ :=
⌊a + b + c + d⌋

theorem floor_S_proof (a b c d : ℝ)
  (h1 : a ^ 2 + 2 * b ^ 2 = 2016)
  (h2 : c ^ 2 + 2 * d ^ 2 = 2016)
  (h3 : a * c = 1024)
  (h4 : b * d = 1024)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : floor_S a b c d = 129 := 
sorry

end NUMINAMATH_GPT_floor_S_proof_l16_1651


namespace NUMINAMATH_GPT_ratio_of_sides_product_of_areas_and_segments_l16_1674

variable (S S' S'' : ℝ) (a a' : ℝ)

-- Given condition
axiom proportion_condition : S / S'' = a / a'

-- Proofs that need to be verified
theorem ratio_of_sides (S S' : ℝ) (a a' : ℝ) (h : S / S'' = a / a') :
  S / a = S' / a' :=
sorry

theorem product_of_areas_and_segments (S S' : ℝ) (a a' : ℝ) (h: S / S'' = a / a') :
  S * a' = S' * a :=
sorry

end NUMINAMATH_GPT_ratio_of_sides_product_of_areas_and_segments_l16_1674


namespace NUMINAMATH_GPT_fire_fighting_max_saved_houses_l16_1662

noncomputable def max_houses_saved (n c : ℕ) : ℕ :=
  n^2 + c^2 - n * c - c

theorem fire_fighting_max_saved_houses (n c : ℕ) (h : c ≤ n / 2) :
    ∃ k, k = max_houses_saved n c :=
    sorry

end NUMINAMATH_GPT_fire_fighting_max_saved_houses_l16_1662


namespace NUMINAMATH_GPT_find_a_l16_1679

-- Define the given conditions
def parabola_eq (a b c y : ℝ) : ℝ := a * y^2 + b * y + c
def vertex : (ℝ × ℝ) := (3, -1)
def point_on_parabola : (ℝ × ℝ) := (7, 3)

-- Define the theorem to be proved
theorem find_a (a b c : ℝ) (h_eqn : ∀ y, parabola_eq a b c y = x)
  (h_vertex : parabola_eq a b c (-vertex.snd) = vertex.fst)
  (h_point : parabola_eq a b c (point_on_parabola.snd) = point_on_parabola.fst) :
  a = 1 / 4 := 
sorry

end NUMINAMATH_GPT_find_a_l16_1679


namespace NUMINAMATH_GPT_flight_duration_sum_l16_1681

theorem flight_duration_sum (h m : ℕ) (h_hours : h = 11) (m_minutes : m = 45) (time_limit : 0 < m ∧ m < 60) :
  h + m = 56 :=
by
  sorry

end NUMINAMATH_GPT_flight_duration_sum_l16_1681


namespace NUMINAMATH_GPT_proof_correct_judgments_l16_1672

def terms_are_like (t1 t2 : Expr) : Prop := sorry -- Define like terms
def is_polynomial (p : Expr) : Prop := sorry -- Define polynomial
def is_quadratic_trinomial (p : Expr) : Prop := sorry -- Define quadratic trinomial
def constant_term (p : Expr) : Expr := sorry -- Define extraction of constant term

theorem proof_correct_judgments :
  let t1 := (2 * Real.pi * (a ^ 2) * b)
  let t2 := ((1 / 3) * (a ^ 2) * b)
  let p1 := (5 * a + 4 * b - 1)
  let p2 := (x - 2 * x * y + y)
  let p3 := ((x + y) / 4)
  let p4 := (x / 2 + 1)
  let p5 := (a / 4)
  terms_are_like t1 t2 ∧ 
  constant_term p1 = 1 = False ∧
  is_quadratic_trinomial p2 ∧
  is_polynomial p3 ∧ is_polynomial p4 ∧ is_polynomial p5
  → ("①③④" = "C") :=
by
  sorry

end NUMINAMATH_GPT_proof_correct_judgments_l16_1672


namespace NUMINAMATH_GPT_remainder_of_3x_minus_2y_mod_30_l16_1663

theorem remainder_of_3x_minus_2y_mod_30
  (p q : ℤ) (x y : ℤ)
  (hx : x = 60 * p + 53)
  (hy : y = 45 * q + 28) :
  (3 * x - 2 * y) % 30 = 13 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_of_3x_minus_2y_mod_30_l16_1663


namespace NUMINAMATH_GPT_prime_square_mod_12_l16_1640

theorem prime_square_mod_12 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) : 
  (p ^ 2) % 12 = 1 :=
sorry

end NUMINAMATH_GPT_prime_square_mod_12_l16_1640


namespace NUMINAMATH_GPT_symmetric_function_l16_1600

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def symmetric_about_axis (f : ℤ → ℤ) (axis : ℤ) : Prop :=
  ∀ x : ℤ, f (axis - x) = f (axis + x)

theorem symmetric_function (a : ℕ → ℤ) (d : ℤ) (f : ℤ → ℤ) (a1 a2 : ℤ) (axis : ℤ) :
  (∀ x, f x = |x - a1| + |x - a2|) →
  arithmetic_sequence a d →
  d ≠ 0 →
  axis = (a1 + a2) / 2 →
  symmetric_about_axis f axis :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_symmetric_function_l16_1600


namespace NUMINAMATH_GPT_rotated_angle_l16_1608

theorem rotated_angle (angle_ACB_initial : ℝ) (rotation_angle : ℝ) (h1 : angle_ACB_initial = 60) (h2 : rotation_angle = 630) : 
  ∃ (angle_ACB_new : ℝ), angle_ACB_new = 30 :=
by
  -- Define the effective rotation
  let effective_rotation := rotation_angle % 360 -- Modulo operation
  
  -- Calculate the new angle
  let angle_new := angle_ACB_initial + effective_rotation
  
  -- Ensure the angle is acute by converting if needed
  let acute_angle_new := if angle_new > 180 then 360 - angle_new else angle_new
  
  -- The acute angle should be 30 degrees
  use acute_angle_new
  have : acute_angle_new = 30 := sorry
  exact this

end NUMINAMATH_GPT_rotated_angle_l16_1608


namespace NUMINAMATH_GPT_Rajesh_Spend_Salary_on_Food_l16_1671

theorem Rajesh_Spend_Salary_on_Food
    (monthly_salary : ℝ)
    (percentage_medicines : ℝ)
    (savings_percentage : ℝ)
    (savings : ℝ) :
    monthly_salary = 15000 ∧
    percentage_medicines = 0.20 ∧
    savings_percentage = 0.60 ∧
    savings = 4320 →
    (32 : ℝ) = ((monthly_salary * percentage_medicines + monthly_salary * (1 - (percentage_medicines + savings_percentage))) / monthly_salary) * 100 :=
by
  sorry

end NUMINAMATH_GPT_Rajesh_Spend_Salary_on_Food_l16_1671


namespace NUMINAMATH_GPT_division_problem_l16_1610

theorem division_problem : 240 / (12 + 14 * 2) = 6 := by
  sorry

end NUMINAMATH_GPT_division_problem_l16_1610


namespace NUMINAMATH_GPT_other_candidate_valid_votes_l16_1677

noncomputable def validVotes (totalVotes invalidPct : ℝ) : ℝ :=
  totalVotes * (1 - invalidPct)

noncomputable def otherCandidateVotes (validVotes oneCandidatePct : ℝ) : ℝ :=
  validVotes * (1 - oneCandidatePct)

theorem other_candidate_valid_votes :
  let totalVotes := 7500
  let invalidPct := 0.20
  let oneCandidatePct := 0.55
  validVotes totalVotes invalidPct = 6000 ∧
  otherCandidateVotes (validVotes totalVotes invalidPct) oneCandidatePct = 2700 :=
by
  sorry

end NUMINAMATH_GPT_other_candidate_valid_votes_l16_1677


namespace NUMINAMATH_GPT_central_angle_of_sector_l16_1644

theorem central_angle_of_sector (R θ l : ℝ) (h1 : 2 * R + l = π * R) : θ = π - 2 := 
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l16_1644


namespace NUMINAMATH_GPT_reaches_school_early_l16_1637

theorem reaches_school_early (R : ℝ) (T : ℝ) (F : ℝ) (T' : ℝ)
    (h₁ : F = (6/5) * R)
    (h₂ : T = 24)
    (h₃ : R * T = F * T')
    : T - T' = 4 := by
  -- All the given conditions are set; fill in the below placeholder with the proof.
  sorry

end NUMINAMATH_GPT_reaches_school_early_l16_1637


namespace NUMINAMATH_GPT_cost_of_bananas_l16_1656

theorem cost_of_bananas (A B : ℝ) (n : ℝ) (Tcost: ℝ) (Acost: ℝ): 
  (A * n + B = Tcost) → (A * (1 / 2 * n) + B = Acost) → (Tcost = 7) → (Acost = 5) → B = 3 :=
by
  intros hTony hArnold hTcost hAcost
  sorry

end NUMINAMATH_GPT_cost_of_bananas_l16_1656


namespace NUMINAMATH_GPT_weekly_cost_l16_1614

def cost_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 7
def number_of_bodyguards : ℕ := 2

theorem weekly_cost :
  (cost_per_hour * hours_per_day * number_of_bodyguards * days_per_week) = 2240 := by
  sorry

end NUMINAMATH_GPT_weekly_cost_l16_1614


namespace NUMINAMATH_GPT_add_in_base6_l16_1660

def add_base6 (a b : ℕ) : ℕ := (a + b) % 6 + (((a + b) / 6) * 10)

theorem add_in_base6 (x y : ℕ) (h1 : x = 5) (h2 : y = 23) : add_base6 x y = 32 :=
by
  rw [h1, h2]
  -- Explanation: here add_base6 interprets numbers as base 6 and then performs addition,
  -- taking care of the base conversion automatically. This avoids directly involving steps of the given solution.
  sorry

end NUMINAMATH_GPT_add_in_base6_l16_1660


namespace NUMINAMATH_GPT_periodic_symmetry_mono_f_l16_1692

-- Let f be a function from ℝ to ℝ.
variable (f : ℝ → ℝ)

-- f has the domain of ℝ.
-- f(x) = f(x + 6) for all x ∈ ℝ.
axiom periodic_f : ∀ x : ℝ, f x = f (x + 6)

-- f is monotonically decreasing in (0, 3).
axiom mono_f : ∀ ⦃x y : ℝ⦄, 0 < x → x < y → y < 3 → f y < f x

-- The graph of f is symmetric about the line x = 3.
axiom symmetry_f : ∀ x : ℝ, f x = f (6 - x)

-- Prove that f(3.5) < f(1.5) < f(6.5).
theorem periodic_symmetry_mono_f : f 3.5 < f 1.5 ∧ f 1.5 < f 6.5 :=
sorry

end NUMINAMATH_GPT_periodic_symmetry_mono_f_l16_1692


namespace NUMINAMATH_GPT_range_of_a_l16_1635

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 4^x - (a + 3) * 2^x + 1 = 0) → a ≥ -1 := sorry

end NUMINAMATH_GPT_range_of_a_l16_1635


namespace NUMINAMATH_GPT_part1_tangent_line_at_x2_part2_inequality_l16_1629

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x + Real.exp 2 - 7

theorem part1_tangent_line_at_x2 (a : ℝ) (h_a : a = 2) :
  ∃ m b : ℝ, (∀ x : ℝ, f x a = m * x + b) ∧ m = Real.exp 2 - 2 ∧ b = -(2 * Real.exp 2 - 7) := by
  sorry

theorem part2_inequality (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x a ≥ (7 / 4) * x^2) → a ≤ Real.exp 2 - 7 := by
  sorry

end NUMINAMATH_GPT_part1_tangent_line_at_x2_part2_inequality_l16_1629


namespace NUMINAMATH_GPT_tanya_erasers_l16_1605

theorem tanya_erasers (H R TR T : ℕ) 
  (h1 : H = 2 * R) 
  (h2 : R = TR / 2 - 3) 
  (h3 : H = 4) 
  (h4 : TR = T / 2) : 
  T = 20 := 
by 
  sorry

end NUMINAMATH_GPT_tanya_erasers_l16_1605


namespace NUMINAMATH_GPT_infinite_coprime_pairs_with_divisibility_l16_1618

theorem infinite_coprime_pairs_with_divisibility :
  ∃ (A : ℕ → ℕ) (B : ℕ → ℕ), (∀ n, gcd (A n) (B n) = 1) ∧
    ∀ n, (A n ∣ (B n)^2 - 5) ∧ (B n ∣ (A n)^2 - 5) :=
sorry

end NUMINAMATH_GPT_infinite_coprime_pairs_with_divisibility_l16_1618


namespace NUMINAMATH_GPT_sin_30_eq_half_l16_1613

theorem sin_30_eq_half : Real.sin (π / 6) = 1 / 2 := 
  sorry

end NUMINAMATH_GPT_sin_30_eq_half_l16_1613


namespace NUMINAMATH_GPT_triangle_angle_A_l16_1606

theorem triangle_angle_A (AC BC : ℝ) (angle_B : ℝ) (h_AC : AC = Real.sqrt 2) (h_BC : BC = 1) (h_angle_B : angle_B = 45) :
  ∃ (angle_A : ℝ), angle_A = 30 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_A_l16_1606


namespace NUMINAMATH_GPT_parents_without_full_time_jobs_l16_1647

theorem parents_without_full_time_jobs
  {total_parents mothers fathers : ℕ}
  (h_total_parents : total_parents = 100)
  (h_mothers_percentage : mothers = 60)
  (h_fathers_percentage : fathers = 40)
  (h_mothers_full_time : ℕ)
  (h_fathers_full_time : ℕ)
  (h_mothers_ratio : h_mothers_full_time = (5 * mothers) / 6)
  (h_fathers_ratio : h_fathers_full_time = (3 * fathers) / 4) :
  ((total_parents - (h_mothers_full_time + h_fathers_full_time)) * 100 / total_parents = 20) := sorry

end NUMINAMATH_GPT_parents_without_full_time_jobs_l16_1647


namespace NUMINAMATH_GPT_quadratic_roots_transform_l16_1690

theorem quadratic_roots_transform {p q : ℝ} (h1 : 3 * p^2 + 5 * p - 7 = 0) (h2 : 3 * q^2 + 5 * q - 7 = 0) : (p - 2) * (q - 2) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_transform_l16_1690


namespace NUMINAMATH_GPT_fitted_bowling_ball_volume_l16_1668

theorem fitted_bowling_ball_volume :
  let r_bowl := 20 -- radius of the bowling ball in cm
  let r_hole1 := 1 -- radius of the first hole in cm
  let r_hole2 := 2 -- radius of the second hole in cm
  let r_hole3 := 2 -- radius of the third hole in cm
  let depth := 10 -- depth of each hole in cm
  let V_bowl := (4/3) * Real.pi * r_bowl^3
  let V_hole1 := Real.pi * r_hole1^2 * depth
  let V_hole2 := Real.pi * r_hole2^2 * depth
  let V_hole3 := Real.pi * r_hole3^2 * depth
  let V_holes := V_hole1 + V_hole2 + V_hole3
  let V_fitted := V_bowl - V_holes
  V_fitted = (31710 / 3) * Real.pi :=
by sorry

end NUMINAMATH_GPT_fitted_bowling_ball_volume_l16_1668


namespace NUMINAMATH_GPT_joan_apples_after_giving_l16_1696

-- Definitions of the conditions
def initial_apples : ℕ := 43
def given_away_apples : ℕ := 27

-- Statement to prove
theorem joan_apples_after_giving : (initial_apples - given_away_apples = 16) :=
by sorry

end NUMINAMATH_GPT_joan_apples_after_giving_l16_1696


namespace NUMINAMATH_GPT_b_work_time_l16_1602

theorem b_work_time (W : ℝ) (days_A days_combined : ℝ)
  (hA : W / days_A = W / 16)
  (h_combined : W / days_combined = W / (16 / 3)) :
  ∃ days_B, days_B = 8 :=
by
  sorry

end NUMINAMATH_GPT_b_work_time_l16_1602


namespace NUMINAMATH_GPT_probability_red_or_yellow_l16_1684

-- Definitions and conditions
def p_green : ℝ := 0.25
def p_blue : ℝ := 0.35
def total_probability := 1
def p_red_and_yellow := total_probability - (p_green + p_blue)

-- Theorem statement
theorem probability_red_or_yellow :
  p_red_and_yellow = 0.40 :=
by
  -- Here we would prove that the combined probability of selecting either a red or yellow jelly bean is 0.40, given the conditions.
  sorry

end NUMINAMATH_GPT_probability_red_or_yellow_l16_1684


namespace NUMINAMATH_GPT_james_milk_left_l16_1689

@[simp] def ounces_in_gallon : ℕ := 128
@[simp] def gallons_james_has : ℕ := 3
@[simp] def ounces_drank : ℕ := 13

theorem james_milk_left :
  (gallons_james_has * ounces_in_gallon - ounces_drank) = 371 :=
by
  sorry

end NUMINAMATH_GPT_james_milk_left_l16_1689


namespace NUMINAMATH_GPT_range_f_pos_l16_1632

noncomputable def f : ℝ → ℝ := sorry
axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom increasing_f : ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f x ≤ f y
axiom f_at_neg_one : f (-1) = 0

theorem range_f_pos : {x : ℝ | f x > 0} = Set.Ioo (-1) 1 := 
by
  sorry

end NUMINAMATH_GPT_range_f_pos_l16_1632
