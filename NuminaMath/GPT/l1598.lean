import Mathlib

namespace NUMINAMATH_GPT_volume_of_cube_l1598_159848

-- Define the conditions
def surface_area (a : ℝ) : ℝ := 6 * a^2
def side_length (a : ℝ) (SA : ℝ) : Prop := SA = 6 * a^2
def volume (a : ℝ) : ℝ := a^3

-- State the theorem
theorem volume_of_cube (a : ℝ) (SA : surface_area a = 150) : volume a = 125 := 
sorry

end NUMINAMATH_GPT_volume_of_cube_l1598_159848


namespace NUMINAMATH_GPT_employees_in_january_l1598_159828

theorem employees_in_january (E : ℝ) (h : 500 = 1.15 * E) : E = 500 / 1.15 :=
by
  sorry

end NUMINAMATH_GPT_employees_in_january_l1598_159828


namespace NUMINAMATH_GPT_actual_plot_area_l1598_159870

noncomputable def area_of_triangle_in_acres : Real :=
  let base_cm : Real := 8
  let height_cm : Real := 5
  let area_cm2 : Real := 0.5 * base_cm * height_cm
  let conversion_factor_cm2_to_km2 : Real := 25
  let area_km2 : Real := area_cm2 * conversion_factor_cm2_to_km2
  let conversion_factor_km2_to_acres : Real := 247.1
  area_km2 * conversion_factor_km2_to_acres

theorem actual_plot_area :
  area_of_triangle_in_acres = 123550 :=
by
  sorry

end NUMINAMATH_GPT_actual_plot_area_l1598_159870


namespace NUMINAMATH_GPT_numberOfBookshelves_l1598_159801

-- Define the conditions as hypotheses
def numBooks : ℕ := 23
def numMagazines : ℕ := 61
def totalItems : ℕ := 2436

-- Define the number of items per bookshelf
def itemsPerBookshelf : ℕ := numBooks + numMagazines

-- State the theorem to be proven
theorem numberOfBookshelves (bookshelves : ℕ) :
  itemsPerBookshelf * bookshelves = totalItems → 
  bookshelves = 29 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_numberOfBookshelves_l1598_159801


namespace NUMINAMATH_GPT_smallest_m_for_reflection_l1598_159833

noncomputable def theta : Real := Real.arctan (1 / 3)
noncomputable def pi_8 : Real := Real.pi / 8
noncomputable def pi_12 : Real := Real.pi / 12
noncomputable def pi_4 : Real := Real.pi / 4
noncomputable def pi_6 : Real := Real.pi / 6

/-- The smallest positive integer m such that R^(m)(l) = l
where the transformation R(l) is described as:
l is reflected in l1 (angle pi/8), then the resulting line is
reflected in l2 (angle pi/12) -/
theorem smallest_m_for_reflection :
  ∃ (m : ℕ), m > 0 ∧ ∀ (k : ℤ), m = 12 * k + 12 := by
sorry

end NUMINAMATH_GPT_smallest_m_for_reflection_l1598_159833


namespace NUMINAMATH_GPT_point_on_line_iff_l1598_159847

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given points O, A, B, and X in a vector space V, prove that X lies on the line AB if and only if
there exists a scalar t such that the position vector of X is a linear combination of the position vectors
of A and B with respect to O. -/
theorem point_on_line_iff (O A B X : V) :
  (∃ t : ℝ, X - O = t • (A - O) + (1 - t) • (B - O)) ↔ (∃ t : ℝ, ∃ (t : ℝ), X - O = (1 - t) • (A - O) + t • (B - O)) :=
sorry

end NUMINAMATH_GPT_point_on_line_iff_l1598_159847


namespace NUMINAMATH_GPT_sum_m_n_l1598_159800

-- Declare the namespaces and definitions for the problem
namespace DelegateProblem

-- Condition: total number of delegates
def total_delegates : Nat := 12

-- Condition: number of delegates from each country
def delegates_per_country : Nat := 4

-- Computation of m and n such that their sum is 452
-- This follows from the problem statement and the solution provided
def m : Nat := 221
def n : Nat := 231

-- Theorem statement in Lean for proving m + n = 452
theorem sum_m_n : m + n = 452 := by
  -- Algebraic proof omitted
  sorry

end DelegateProblem

end NUMINAMATH_GPT_sum_m_n_l1598_159800


namespace NUMINAMATH_GPT_supplement_of_supplement_of_58_l1598_159823

theorem supplement_of_supplement_of_58 (α : ℝ) (h : α = 58) : 180 - (180 - α) = 58 :=
by
  sorry

end NUMINAMATH_GPT_supplement_of_supplement_of_58_l1598_159823


namespace NUMINAMATH_GPT_find_k_l1598_159827

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 1 / x + 5
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^3 - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → k = - 485 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1598_159827


namespace NUMINAMATH_GPT_Joe_team_wins_eq_1_l1598_159803

-- Definition for the points a team gets for winning a game.
def points_per_win := 3
-- Definition for the points a team gets for a tie game.
def points_per_tie := 1

-- Given conditions
def Joe_team_draws := 3
def first_place_wins := 2
def first_place_ties := 2
def points_difference := 2

def first_place_points := (first_place_wins * points_per_win) + (first_place_ties * points_per_tie)

def Joe_team_total_points := first_place_points - points_difference
def Joe_team_points_from_ties := Joe_team_draws * points_per_tie
def Joe_team_points_from_wins := Joe_team_total_points - Joe_team_points_from_ties

-- To prove: number of games Joe's team won
theorem Joe_team_wins_eq_1 : (Joe_team_points_from_wins / points_per_win) = 1 :=
by
  sorry

end NUMINAMATH_GPT_Joe_team_wins_eq_1_l1598_159803


namespace NUMINAMATH_GPT_local_minimum_at_two_l1598_159873

def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem local_minimum_at_two : ∃ a : ℝ, a = 2 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, (0 < |x - a| ∧ |x - a| < δ) → f x > f a :=
by sorry

end NUMINAMATH_GPT_local_minimum_at_two_l1598_159873


namespace NUMINAMATH_GPT_karen_total_cost_l1598_159832

noncomputable def calculate_total_cost (burger_price sandwich_price smoothie_price : ℝ) (num_smoothies : ℕ)
  (discount_rate tax_rate : ℝ) (order_time : ℕ) : ℝ :=
  let total_cost_before_discount := burger_price + sandwich_price + (num_smoothies * smoothie_price)
  let discount := if total_cost_before_discount > 15 ∧ order_time ≥ 1400 ∧ order_time ≤ 1600 then total_cost_before_discount * discount_rate else 0
  let reduced_price := total_cost_before_discount - discount
  let tax := reduced_price * tax_rate
  reduced_price + tax

theorem karen_total_cost :
  calculate_total_cost 5.75 4.50 4.25 2 0.20 0.12 1545 = 16.80 :=
by
  sorry

end NUMINAMATH_GPT_karen_total_cost_l1598_159832


namespace NUMINAMATH_GPT_range_of_x_satisfying_inequality_l1598_159812

noncomputable def f : ℝ → ℝ := sorry -- f is some even and monotonically increasing function

theorem range_of_x_satisfying_inequality :
  (∀ x, f (-x) = f x) ∧ (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) → {x : ℝ | f x < f 1} = {x : ℝ | -1 < x ∧ x < 1} :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_x_satisfying_inequality_l1598_159812


namespace NUMINAMATH_GPT_speed_on_way_home_l1598_159845

theorem speed_on_way_home (d : ℝ) (v_up : ℝ) (v_avg : ℝ) (v_home : ℝ) 
  (h1 : v_up = 110) 
  (h2 : v_avg = 91)
  (h3 : 91 = (2 * d) / (d / 110 + d / v_home)) : 
  v_home = 10010 / 129 := 
sorry

end NUMINAMATH_GPT_speed_on_way_home_l1598_159845


namespace NUMINAMATH_GPT_Alejandra_overall_score_l1598_159876

theorem Alejandra_overall_score :
  let score1 := (60/100 : ℝ) * 20
  let score2 := (75/100 : ℝ) * 30
  let score3 := (85/100 : ℝ) * 40
  let total_score := score1 + score2 + score3
  let total_questions := 90
  let overall_percentage := (total_score / total_questions) * 100
  round overall_percentage = 77 :=
by
  sorry

end NUMINAMATH_GPT_Alejandra_overall_score_l1598_159876


namespace NUMINAMATH_GPT_percentage_of_water_in_juice_l1598_159899

-- Define the initial condition for tomato puree water percentage
def puree_water_percentage : ℝ := 0.20

-- Define the volume of tomato puree produced from tomato juice
def volume_puree : ℝ := 3.75

-- Define the volume of tomato juice used to produce the puree
def volume_juice : ℝ := 30

-- Given conditions and definitions, prove the percentage of water in tomato juice
theorem percentage_of_water_in_juice :
  ((volume_juice - (volume_puree - puree_water_percentage * volume_puree)) / volume_juice) * 100 = 90 :=
by sorry

end NUMINAMATH_GPT_percentage_of_water_in_juice_l1598_159899


namespace NUMINAMATH_GPT_neg_one_power_zero_l1598_159854

theorem neg_one_power_zero : (-1: ℤ)^0 = 1 := 
sorry

end NUMINAMATH_GPT_neg_one_power_zero_l1598_159854


namespace NUMINAMATH_GPT_arithmetic_seq_formula_l1598_159809

variable (a : ℕ → ℤ)

-- Given conditions
axiom h1 : a 1 + a 2 + a 3 = 0
axiom h2 : a 4 + a 5 + a 6 = 18

-- Goal: general formula for the arithmetic sequence
theorem arithmetic_seq_formula (n : ℕ) : a n = 2 * n - 4 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_formula_l1598_159809


namespace NUMINAMATH_GPT_total_cans_collected_l1598_159830

theorem total_cans_collected :
  let cans_in_first_bag := 5
  let cans_in_second_bag := 7
  let cans_in_third_bag := 12
  let cans_in_fourth_bag := 4
  let cans_in_fifth_bag := 8
  let cans_in_sixth_bag := 10
  let cans_in_seventh_bag := 15
  let cans_in_eighth_bag := 6
  let cans_in_ninth_bag := 5
  let cans_in_tenth_bag := 13
  let total_cans := cans_in_first_bag + cans_in_second_bag + cans_in_third_bag + cans_in_fourth_bag + cans_in_fifth_bag + cans_in_sixth_bag + cans_in_seventh_bag + cans_in_eighth_bag + cans_in_ninth_bag + cans_in_tenth_bag
  total_cans = 85 :=
by
  sorry

end NUMINAMATH_GPT_total_cans_collected_l1598_159830


namespace NUMINAMATH_GPT_greatest_possible_value_of_q_minus_r_l1598_159835

theorem greatest_possible_value_of_q_minus_r :
  ∃ q r : ℕ, 0 < q ∧ 0 < r ∧ 852 = 21 * q + r ∧ q - r = 28 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_greatest_possible_value_of_q_minus_r_l1598_159835


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1598_159822

theorem hyperbola_asymptotes (x y : ℝ) (h : y^2 / 16 - x^2 / 9 = (1 : ℝ)) :
  ∃ (m : ℝ), (m = 4 / 3) ∨ (m = -4 / 3) :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1598_159822


namespace NUMINAMATH_GPT_system_solution_l1598_159878

theorem system_solution (x y z : ℚ) 
  (h1 : x + y + x * y = 19) 
  (h2 : y + z + y * z = 11) 
  (h3 : z + x + z * x = 14) :
    (x = 4 ∧ y = 3 ∧ z = 2) ∨ (x = -6 ∧ y = -5 ∧ z = -4) :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l1598_159878


namespace NUMINAMATH_GPT_score_difference_l1598_159810

theorem score_difference 
  (x y z w : ℝ)
  (h1 : x = 2 + (y + z + w) / 3)
  (h2 : y = (x + z + w) / 3 - 3)
  (h3 : z = 3 + (x + y + w) / 3) :
  (x + y + z) / 3 - w = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_score_difference_l1598_159810


namespace NUMINAMATH_GPT_find_multiplier_l1598_159853

variable {a b : ℝ} 

theorem find_multiplier (h1 : 3 * a = x * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 4 = b / 3) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_multiplier_l1598_159853


namespace NUMINAMATH_GPT_vanAubel_theorem_l1598_159805

variables (A B C O A1 B1 C1 : Type)
variables (CA1 A1B CB1 B1A CO OC1 : ℝ)

-- Given Conditions
axiom condition1 : CB1 / B1A = 1
axiom condition2 : CO / OC1 = 2

-- Van Aubel's theorem statement
theorem vanAubel_theorem : (CO / OC1) = (CA1 / A1B) + (CB1 / B1A) := by
  sorry

end NUMINAMATH_GPT_vanAubel_theorem_l1598_159805


namespace NUMINAMATH_GPT_remainder_a52_div_52_l1598_159816

def a_n (n : ℕ) : ℕ := 
  (List.range (n + 1)).foldl (λ acc x => acc * 10 ^ (Nat.digits 10 x).length + x) 0

theorem remainder_a52_div_52 : (a_n 52) % 52 = 28 := 
  by
  sorry

end NUMINAMATH_GPT_remainder_a52_div_52_l1598_159816


namespace NUMINAMATH_GPT_michael_students_l1598_159857

theorem michael_students (M N : ℕ) (h1 : M = 5 * N) (h2 : M + N + 300 = 3500) : M = 2667 := 
by 
  -- This to be filled later
  sorry

end NUMINAMATH_GPT_michael_students_l1598_159857


namespace NUMINAMATH_GPT_multiples_sum_squared_l1598_159890

theorem multiples_sum_squared :
  let a := 4
  let b := 4
  ((a + b)^2) = 64 :=
by
  sorry

end NUMINAMATH_GPT_multiples_sum_squared_l1598_159890


namespace NUMINAMATH_GPT_total_paper_clips_l1598_159867

/-
Given:
- The number of cartons: c = 3
- The number of boxes: b = 4
- The number of bags: p = 2
- The number of paper clips in each carton: paper_clips_per_carton = 300
- The number of paper clips in each box: paper_clips_per_box = 550
- The number of paper clips in each bag: paper_clips_per_bag = 1200

Prove that the total number of paper clips is 5500.
-/

theorem total_paper_clips :
  let c := 3
  let paper_clips_per_carton := 300
  let b := 4
  let paper_clips_per_box := 550
  let p := 2
  let paper_clips_per_bag := 1200
  (c * paper_clips_per_carton + b * paper_clips_per_box + p * paper_clips_per_bag) = 5500 :=
by
  sorry

end NUMINAMATH_GPT_total_paper_clips_l1598_159867


namespace NUMINAMATH_GPT_largest_of_five_consecutive_integers_with_product_15120_eq_9_l1598_159883

theorem largest_of_five_consecutive_integers_with_product_15120_eq_9 :
  ∃ n : ℕ, (n + 0) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120 ∧ n + 4 = 9 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_five_consecutive_integers_with_product_15120_eq_9_l1598_159883


namespace NUMINAMATH_GPT_simplify_logarithmic_expression_l1598_159815

theorem simplify_logarithmic_expression :
  (1 / (Real.logb 12 3 + 1) + 1 / (Real.logb 8 2 + 1) + 1 / (Real.logb 18 9 + 1) = 1) :=
sorry

end NUMINAMATH_GPT_simplify_logarithmic_expression_l1598_159815


namespace NUMINAMATH_GPT_shelter_cats_incoming_l1598_159896

theorem shelter_cats_incoming (x : ℕ) (h : x + x / 2 - 3 + 5 - 1 = 19) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_shelter_cats_incoming_l1598_159896


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l1598_159846

theorem arithmetic_sequence_general_term
    (a : ℕ → ℤ)
    (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_mean_26 : (a 2 + a 6) / 2 = 5)
    (h_mean_37 : (a 3 + a 7) / 2 = 7) :
    ∀ n, a n = 2 * n - 3 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l1598_159846


namespace NUMINAMATH_GPT_fencing_required_l1598_159895

theorem fencing_required (L W : ℝ) (hL : L = 20) (hArea : L * W = 60) : (L + 2 * W) = 26 := 
by
  sorry

end NUMINAMATH_GPT_fencing_required_l1598_159895


namespace NUMINAMATH_GPT_locus_of_point_P_l1598_159872

noncomputable def ellipse_locus
  (r : ℝ) (u v : ℝ) : Prop :=
  ∃ x1 y1 : ℝ,
    (x1^2 + y1^2 = r^2) ∧ (u - x1)^2 + v^2 = y1^2

theorem locus_of_point_P {r u v : ℝ} :
  (ellipse_locus r u v) ↔ ((u^2 / (2 * r^2)) + (v^2 / r^2) ≤ 1) :=
by sorry

end NUMINAMATH_GPT_locus_of_point_P_l1598_159872


namespace NUMINAMATH_GPT_intersect_point_l1598_159897

-- Definitions as per conditions
def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b
def f_inv (x : ℝ) (a : ℝ) : ℝ := a -- We define inverse as per given (4, a)

-- Variables for the conditions
variables (a b : ℤ)

-- Theorems to prove the conditions match the answers
theorem intersect_point : ∃ a b : ℤ, f 4 b = a ∧ f_inv 4 a = 4 ∧ a = 4 := by
  sorry

end NUMINAMATH_GPT_intersect_point_l1598_159897


namespace NUMINAMATH_GPT_pats_stick_length_correct_l1598_159885

noncomputable def jane_stick_length : ℕ := 22
noncomputable def sarah_stick_length : ℕ := jane_stick_length + 24
noncomputable def uncovered_pats_stick : ℕ := sarah_stick_length / 2
noncomputable def covered_pats_stick : ℕ := 7
noncomputable def total_pats_stick : ℕ := uncovered_pats_stick + covered_pats_stick

theorem pats_stick_length_correct : total_pats_stick = 30 := by
  sorry

end NUMINAMATH_GPT_pats_stick_length_correct_l1598_159885


namespace NUMINAMATH_GPT_adam_total_spending_l1598_159886

def first_laptop_cost : ℤ := 500
def second_laptop_cost : ℤ := 3 * first_laptop_cost
def total_cost : ℤ := first_laptop_cost + second_laptop_cost

theorem adam_total_spending : total_cost = 2000 := by
  sorry

end NUMINAMATH_GPT_adam_total_spending_l1598_159886


namespace NUMINAMATH_GPT_outer_circle_radius_l1598_159891

theorem outer_circle_radius (C_inner : ℝ) (w : ℝ) (r_outer : ℝ) (h1 : C_inner = 440) (h2 : w = 14) :
  r_outer = (440 / (2 * Real.pi)) + 14 :=
by 
  have h_r_inner : r_outer = (440 / (2 * Real.pi)) + 14 := by sorry
  exact h_r_inner

end NUMINAMATH_GPT_outer_circle_radius_l1598_159891


namespace NUMINAMATH_GPT_percentage_of_x_eq_y_l1598_159817

theorem percentage_of_x_eq_y
  (x y : ℝ) 
  (h : 0.60 * (x - y) = 0.20 * (x + y)) :
  y = 0.50 * x := 
sorry

end NUMINAMATH_GPT_percentage_of_x_eq_y_l1598_159817


namespace NUMINAMATH_GPT_circle_equation_from_parabola_l1598_159865

theorem circle_equation_from_parabola :
  let F := (2, 0)
  let A := (2, 4)
  let B := (2, -4)
  let diameter := 8
  let center := F
  let radius_squared := diameter^2 / 4
  (x - center.1)^2 + y^2 = radius_squared :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_from_parabola_l1598_159865


namespace NUMINAMATH_GPT_triangular_number_is_perfect_square_l1598_159826

def is_triangular_number (T : ℕ) : Prop :=
∃ n : ℕ, T = n * (n + 1) / 2

def is_perfect_square (T : ℕ) : Prop :=
∃ y : ℕ, T = y * y

theorem triangular_number_is_perfect_square:
  ∀ (x_k : ℕ), 
    ((∃ n y : ℕ, (2 * n + 1)^2 - 8 * y^2 = 1 ∧ T_n = n * (n + 1) / 2 ∧ T_n = x_k^2 - 1 / 8) →
    (is_triangular_number T_n → is_perfect_square T_n)) :=
by
  sorry

end NUMINAMATH_GPT_triangular_number_is_perfect_square_l1598_159826


namespace NUMINAMATH_GPT_actual_revenue_percentage_of_projected_l1598_159831

theorem actual_revenue_percentage_of_projected (R : ℝ) (hR : R > 0) :
  (0.75 * R) / (1.2 * R) * 100 = 62.5 := 
by
  sorry

end NUMINAMATH_GPT_actual_revenue_percentage_of_projected_l1598_159831


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1598_159861

variable (a : ℝ)
variable (b : ℝ)

theorem simplify_and_evaluate (h : b = -1/3) : (a + b)^2 - a * (2 * b + a) = 1/9 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1598_159861


namespace NUMINAMATH_GPT_simplest_common_denominator_l1598_159811

theorem simplest_common_denominator (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (d : ℤ), d = x^2 * y^2 ∧ ∀ (a b : ℤ), 
    (∃ (k : ℤ), a = k * (x^2 * y)) ∧ (∃ (m : ℤ), b = m * (x * y^2)) → d = lcm a b :=
by
  sorry

end NUMINAMATH_GPT_simplest_common_denominator_l1598_159811


namespace NUMINAMATH_GPT_largest_number_is_C_l1598_159819

theorem largest_number_is_C (A B C D E : ℝ) 
  (hA : A = 0.989) 
  (hB : B = 0.9098) 
  (hC : C = 0.9899) 
  (hD : D = 0.9009) 
  (hE : E = 0.9809) : 
  C > A ∧ C > B ∧ C > D ∧ C > E := 
by 
  sorry

end NUMINAMATH_GPT_largest_number_is_C_l1598_159819


namespace NUMINAMATH_GPT_product_of_x_y_l1598_159889

theorem product_of_x_y (x y : ℝ) (h1 : -3 * x + 4 * y = 28) (h2 : 3 * x - 2 * y = 8) : x * y = 264 :=
by
  sorry

end NUMINAMATH_GPT_product_of_x_y_l1598_159889


namespace NUMINAMATH_GPT_equality_of_expressions_l1598_159850

theorem equality_of_expressions (a b c : ℝ) (h : a = b + c + 2) : 
  a + b * c = (a + b) * (a + c) ↔ a = 0 ∨ a = 1 :=
by sorry

end NUMINAMATH_GPT_equality_of_expressions_l1598_159850


namespace NUMINAMATH_GPT_determinant_scaling_l1598_159838

variable (p q r s : ℝ)

theorem determinant_scaling 
  (h : Matrix.det ![![p, q], ![r, s]] = 3) : 
  Matrix.det ![![2 * p, 2 * p + 5 * q], ![2 * r, 2 * r + 5 * s]] = 30 :=
by 
  sorry

end NUMINAMATH_GPT_determinant_scaling_l1598_159838


namespace NUMINAMATH_GPT_scrap_cookie_radius_l1598_159843

theorem scrap_cookie_radius (r: ℝ) (r_cookies: ℝ) (A_scrap: ℝ) (r_large: ℝ) (A_large: ℝ) (A_total_small: ℝ):
  r_cookies = 1.5 ∧
  r_large = r_cookies + 2 * r_cookies ∧
  A_large = π * r_large^2 ∧
  A_total_small = 8 * (π * r_cookies^2) ∧
  A_scrap = A_large - A_total_small ∧
  A_scrap = π * r^2
  → r = r_cookies
  :=
by
  intro h
  rcases h with ⟨hcookies, hrlarge, halarge, hatotalsmall, hascrap, hpi⟩
  sorry

end NUMINAMATH_GPT_scrap_cookie_radius_l1598_159843


namespace NUMINAMATH_GPT_find_pair_not_satisfying_equation_l1598_159888

theorem find_pair_not_satisfying_equation :
  ¬ (187 * 314 - 104 * 565 = 41) :=
by
  sorry

end NUMINAMATH_GPT_find_pair_not_satisfying_equation_l1598_159888


namespace NUMINAMATH_GPT_temperature_range_l1598_159887

-- Define the problem conditions
def highest_temp := 26
def lowest_temp := 12

-- The theorem stating the range of temperature change
theorem temperature_range : ∀ t : ℝ, lowest_temp ≤ t ∧ t ≤ highest_temp :=
by sorry

end NUMINAMATH_GPT_temperature_range_l1598_159887


namespace NUMINAMATH_GPT_total_games_played_l1598_159842

-- Define the function for combinations
def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Given conditions
def teams : ℕ := 20
def games_per_pair : ℕ := 10

-- Proposition stating the target result
theorem total_games_played : 
  (combination teams 2 * games_per_pair) = 1900 :=
by
  sorry

end NUMINAMATH_GPT_total_games_played_l1598_159842


namespace NUMINAMATH_GPT_raccoon_hid_nuts_l1598_159844

theorem raccoon_hid_nuts :
  ∃ (r p : ℕ), r + p = 25 ∧ (p = r - 3) ∧ 5 * r = 6 * p ∧ 5 * r = 70 :=
by
  sorry

end NUMINAMATH_GPT_raccoon_hid_nuts_l1598_159844


namespace NUMINAMATH_GPT_lola_dora_allowance_l1598_159893

variable (total_cost deck_cost sticker_cost sticker_count packs_each : ℕ)
variable (allowance : ℕ)

theorem lola_dora_allowance 
  (h1 : deck_cost = 10)
  (h2 : sticker_cost = 2)
  (h3 : packs_each = 2)
  (h4 : sticker_count = 2 * packs_each)
  (h5 : total_cost = deck_cost + sticker_count * sticker_cost)
  (h6 : total_cost = 18) :
  allowance = 9 :=
sorry

end NUMINAMATH_GPT_lola_dora_allowance_l1598_159893


namespace NUMINAMATH_GPT_triangular_region_area_l1598_159856

noncomputable def area_of_triangle (f g h : ℝ → ℝ) : ℝ :=
  let (x1, y1) := (-3, f (-3))
  let (x2, y2) := (7/3, g (7/3))
  let (x3, y3) := (15/11, f (15/11))
  let base := abs (x2 - x1)
  let height := abs (y3 - 2)
  (1/2) * base * height

theorem triangular_region_area :
  let f x := (2/3) * x + 4
  let g x := -3 * x + 9
  let h x := (2 : ℝ)
  area_of_triangle f g h = 256/33 :=  -- Given conditions
by
  sorry  -- Proof to be supplied

end NUMINAMATH_GPT_triangular_region_area_l1598_159856


namespace NUMINAMATH_GPT_second_part_of_ratio_l1598_159863

-- Define the conditions
def ratio_percent := 20
def first_part := 4

-- Define the proof statement using the conditions
theorem second_part_of_ratio (ratio_percent : ℕ) (first_part : ℕ) : 
  ∃ second_part : ℕ, (first_part * 100) = ratio_percent * second_part :=
by
  -- Let the second part be 20 and verify the condition
  use 20
  -- Clear the proof (details are not required)
  sorry

end NUMINAMATH_GPT_second_part_of_ratio_l1598_159863


namespace NUMINAMATH_GPT_ken_house_distance_condition_l1598_159813

noncomputable def ken_distance_to_dawn : ℕ := 4 -- This is the correct answer

theorem ken_house_distance_condition (K M : ℕ) (h1 : K = 2 * M) (h2 : K + M + M + K = 12) :
  K = ken_distance_to_dawn :=
  by
  sorry

end NUMINAMATH_GPT_ken_house_distance_condition_l1598_159813


namespace NUMINAMATH_GPT_ratio_of_sum_and_difference_l1598_159818

theorem ratio_of_sum_and_difference (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (x + y) / (x - y) = x / y) : x / y = 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_sum_and_difference_l1598_159818


namespace NUMINAMATH_GPT_bus_arrival_time_at_first_station_l1598_159875

noncomputable def time_to_first_station (start_time end_time first_station_to_work: ℕ) : ℕ :=
  (end_time - start_time) - first_station_to_work

theorem bus_arrival_time_at_first_station :
  time_to_first_station 360 540 140 = 40 :=
by
  -- provide the proof here, which has been omitted per the instructions
  sorry

end NUMINAMATH_GPT_bus_arrival_time_at_first_station_l1598_159875


namespace NUMINAMATH_GPT_composite_product_quotient_l1598_159825

def first_seven_composite := [4, 6, 8, 9, 10, 12, 14]
def next_eight_composite := [15, 16, 18, 20, 21, 22, 24, 25]

noncomputable def product {α : Type*} [Monoid α] (l : List α) : α :=
  l.foldl (· * ·) 1

theorem composite_product_quotient : 
  (product first_seven_composite : ℚ) / (product next_eight_composite : ℚ) = 1 / 2475 := 
by 
  sorry

end NUMINAMATH_GPT_composite_product_quotient_l1598_159825


namespace NUMINAMATH_GPT_xiao_li_hits_bullseye_14_times_l1598_159869

theorem xiao_li_hits_bullseye_14_times
  (initial_rifle_bullets : ℕ := 10)
  (initial_pistol_bullets : ℕ := 14)
  (reward_per_bullseye_rifle : ℕ := 2)
  (reward_per_bullseye_pistol : ℕ := 4)
  (xiao_wang_bullseyes : ℕ := 30)
  (total_bullets : ℕ := initial_rifle_bullets + xiao_wang_bullseyes * reward_per_bullseye_rifle) :
  ∃ (xiao_li_bullseyes : ℕ), total_bullets = initial_pistol_bullets + xiao_li_bullseyes * reward_per_bullseye_pistol ∧ xiao_li_bullseyes = 14 :=
by sorry

end NUMINAMATH_GPT_xiao_li_hits_bullseye_14_times_l1598_159869


namespace NUMINAMATH_GPT_comm_ring_of_center_condition_l1598_159858

variable {R : Type*} [Ring R]

def in_center (x : R) : Prop := ∀ y : R, (x * y = y * x)

def is_commutative (R : Type*) [Ring R] : Prop := ∀ a b : R, a * b = b * a

theorem comm_ring_of_center_condition (h : ∀ x : R, in_center (x^2 - x)) : is_commutative R :=
sorry

end NUMINAMATH_GPT_comm_ring_of_center_condition_l1598_159858


namespace NUMINAMATH_GPT_range_of_x_l1598_159806

theorem range_of_x (x : ℝ) (h : |2 * x + 1| + |2 * x - 5| = 6) : -1 / 2 ≤ x ∧ x ≤ 5 / 2 := by
  sorry

end NUMINAMATH_GPT_range_of_x_l1598_159806


namespace NUMINAMATH_GPT_simplify_tan_expression_simplify_complex_expression_l1598_159829

-- Problem 1
theorem simplify_tan_expression (α : ℝ) (hα : 0 < α ∧ α < 2 * π ∧ α > 3 / 2 * π) : 
  (Real.tan α + Real.sqrt ((1 / (Real.cos α)^2) - 1) + 2 * (Real.sin α)^2 + 2 * (Real.cos α)^2 = 2) :=
sorry

-- Problem 2
theorem simplify_complex_expression (α : ℝ) (hα : 0 < α ∧ α < 2 * π ∧ α > 3 / 2 * π) : 
  (Real.sin (α + π) * Real.tan (π - α) * Real.cos (2 * π - α) / (Real.sin (π - α) * Real.sin (π / 2 + α)) + Real.cos (5 * π / 2) = - Real.cos α) :=
sorry

end NUMINAMATH_GPT_simplify_tan_expression_simplify_complex_expression_l1598_159829


namespace NUMINAMATH_GPT_probability_unit_square_not_touch_central_2x2_square_l1598_159855

-- Given a 6x6 checkerboard with a marked 2x2 square at the center,
-- prove that the probability of choosing a unit square that does not touch
-- the marked 2x2 square is 2/3.

theorem probability_unit_square_not_touch_central_2x2_square : 
    let total_squares := 36
    let touching_squares := 12
    let squares_not_touching := total_squares - touching_squares
    (squares_not_touching : ℚ) / (total_squares : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_unit_square_not_touch_central_2x2_square_l1598_159855


namespace NUMINAMATH_GPT_twice_not_square_l1598_159852

theorem twice_not_square (m : ℝ) : 2 * m ≠ m * m := by
  sorry

end NUMINAMATH_GPT_twice_not_square_l1598_159852


namespace NUMINAMATH_GPT_distinct_lines_isosceles_not_equilateral_l1598_159820

-- Define a structure for an isosceles triangle that is not equilateral
structure IsoscelesButNotEquilateralTriangle :=
  (a b c : ℕ)    -- sides of the triangle
  (h₁ : a = b)   -- two equal sides
  (h₂ : a ≠ c)   -- not equilateral (not all three sides are equal)

-- Define that the number of distinct lines representing altitudes, medians, and interior angle bisectors is 5
theorem distinct_lines_isosceles_not_equilateral (T : IsoscelesButNotEquilateralTriangle) : 
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end NUMINAMATH_GPT_distinct_lines_isosceles_not_equilateral_l1598_159820


namespace NUMINAMATH_GPT_angle_between_vectors_l1598_159892

def vector (α : Type) [Field α] := (α × α)

theorem angle_between_vectors
    (a : vector ℝ)
    (b : vector ℝ)
    (ha : a = (4, 0))
    (hb : b = (-1, Real.sqrt 3)) :
  let dot_product (v w : vector ℝ) : ℝ := (v.1 * w.1 + v.2 * w.2)
  let norm (v : vector ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
  let cos_theta := dot_product a b / (norm a * norm b)
  ∀ theta, Real.cos theta = cos_theta → theta = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_vectors_l1598_159892


namespace NUMINAMATH_GPT_custom_op_3_7_l1598_159881

-- Define the custom operation (a # b)
def custom_op (a b : ℕ) : ℕ := a * b - b + b^2

-- State the theorem that proves the result
theorem custom_op_3_7 : custom_op 3 7 = 63 := by
  sorry

end NUMINAMATH_GPT_custom_op_3_7_l1598_159881


namespace NUMINAMATH_GPT_find_m_independent_quadratic_term_l1598_159877

def quadratic_poly (m : ℝ) (x : ℝ) : ℝ :=
  -3 * x^2 + m * x^2 - x + 3

theorem find_m_independent_quadratic_term (m : ℝ) :
  (∀ x, quadratic_poly m x = -x + 3) → m = 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_independent_quadratic_term_l1598_159877


namespace NUMINAMATH_GPT_profit_per_meter_is_15_l1598_159874

def sellingPrice (meters : ℕ) : ℕ := 
    if meters = 85 then 8500 else 0

def costPricePerMeter : ℕ := 85

def totalCostPrice (meters : ℕ) : ℕ := 
    meters * costPricePerMeter

def totalProfit (meters : ℕ) (sellingPrice : ℕ) (costPrice : ℕ) : ℕ := 
    sellingPrice - costPrice

def profitPerMeter (profit : ℕ) (meters : ℕ) : ℕ := 
    profit / meters

theorem profit_per_meter_is_15 : profitPerMeter (totalProfit 85 (sellingPrice 85) (totalCostPrice 85)) 85 = 15 := 
by sorry

end NUMINAMATH_GPT_profit_per_meter_is_15_l1598_159874


namespace NUMINAMATH_GPT_A_work_days_l1598_159862

theorem A_work_days (x : ℝ) (h1 : 1 / 15 + 1 / x = 1 / 8.571428571428571) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_A_work_days_l1598_159862


namespace NUMINAMATH_GPT_smallest_k_for_inequality_l1598_159894

theorem smallest_k_for_inequality :
  ∃ k : ℕ, (∀ m : ℕ, m < k → 64^m ≤ 7) ∧ 64^k > 7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_for_inequality_l1598_159894


namespace NUMINAMATH_GPT_find_series_sum_l1598_159841

noncomputable def series_sum (s : ℝ) : ℝ := ∑' n : ℕ, (n+1) * s^(4*n + 3)

theorem find_series_sum (s : ℝ) (h : s^4 - s - 1/2 = 0) : series_sum s = -4 := by
  sorry

end NUMINAMATH_GPT_find_series_sum_l1598_159841


namespace NUMINAMATH_GPT_coloringBooks_shelves_l1598_159837

variables (initialStock soldBooks shelves : ℕ)

-- Given conditions
def initialBooks : initialStock = 87 := sorry
def booksSold : soldBooks = 33 := sorry
def numberOfShelves : shelves = 9 := sorry

-- Number of coloring books per shelf
def coloringBooksPerShelf (remainingBooksResult : ℕ) (booksPerShelfResult : ℕ) : Prop :=
  remainingBooksResult = initialStock - soldBooks ∧ booksPerShelfResult = remainingBooksResult / shelves

-- Prove the number of coloring books per shelf is 6
theorem coloringBooks_shelves (remainingBooksResult booksPerShelfResult : ℕ) : 
  coloringBooksPerShelf initialStock soldBooks shelves remainingBooksResult booksPerShelfResult →
  booksPerShelfResult = 6 :=
sorry

end NUMINAMATH_GPT_coloringBooks_shelves_l1598_159837


namespace NUMINAMATH_GPT_decreasing_interval_0_pi_over_4_l1598_159898

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (x + φ)

theorem decreasing_interval_0_pi_over_4 (φ : ℝ) (hφ1 : 0 < |φ| ∧ |φ| < Real.pi / 2)
  (hodd : ∀ x : ℝ, f (x + Real.pi / 4) φ = -f (-x + Real.pi / 4) φ) :
  ∀ x : ℝ, 0 < x ∧ x < Real.pi / 4 → f x φ > f (x + 1e-6) φ :=
by sorry

end NUMINAMATH_GPT_decreasing_interval_0_pi_over_4_l1598_159898


namespace NUMINAMATH_GPT_plan_y_more_cost_effective_l1598_159849

theorem plan_y_more_cost_effective (m : Nat) : 2500 + 7 * m < 15 * m → 313 ≤ m :=
by
  intro h
  sorry

end NUMINAMATH_GPT_plan_y_more_cost_effective_l1598_159849


namespace NUMINAMATH_GPT_sequence_monotonically_increasing_l1598_159824

noncomputable def a (n : ℕ) : ℝ := (n - 1 : ℝ) / (n + 1 : ℝ)

theorem sequence_monotonically_increasing : ∀ n : ℕ, a (n + 1) > a n :=
by
  sorry

end NUMINAMATH_GPT_sequence_monotonically_increasing_l1598_159824


namespace NUMINAMATH_GPT_problem_l1598_159879

theorem problem (x y : ℕ) (hy : y > 3) (h : x^2 + y^4 = 2 * ((x-6)^2 + (y+1)^2)) : x^2 + y^4 = 1994 := by
  sorry

end NUMINAMATH_GPT_problem_l1598_159879


namespace NUMINAMATH_GPT_prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l1598_159871

def person_A_hits : ℚ := 1 / 2
def person_B_hits : ℚ := 1 / 3

def person_A_misses : ℚ := 1 - person_A_hits
def person_B_misses : ℚ := 1 - person_B_hits

def exactly_one_hits : ℚ := (person_A_hits * person_B_misses) + (person_B_hits * person_A_misses)
def at_least_one_hits : ℚ := 1 - (person_A_misses * person_B_misses)

theorem prob_exactly_one_hits_is_one_half : exactly_one_hits = 1 / 2 := sorry

theorem prob_at_least_one_hits_is_two_thirds : at_least_one_hits = 2 / 3 := sorry

end NUMINAMATH_GPT_prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l1598_159871


namespace NUMINAMATH_GPT_penny_canoe_l1598_159808

theorem penny_canoe (P : ℕ)
  (h1 : 140 * (2/3 : ℚ) * P + 35 = 595) : P = 6 :=
sorry

end NUMINAMATH_GPT_penny_canoe_l1598_159808


namespace NUMINAMATH_GPT_find_a6_l1598_159864

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {a₁ : ℝ}

/-- The sequence is a geometric sequence -/
axiom geom_seq (n : ℕ) : a n = a₁ * q ^ (n - 1)

/-- The sum of the first three terms is 168 -/
axiom sum_of_first_three_terms : a₁ + a₁ * q + a₁ * q ^ 2 = 168

/-- The difference between the 2nd and the 5th terms is 42 -/
axiom difference_a2_a5 : a₁ * q - a₁ * q ^ 4 = 42

theorem find_a6 : a 6 = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_a6_l1598_159864


namespace NUMINAMATH_GPT_weight_loss_percentage_l1598_159884

variables (W : ℝ) (x : ℝ)

def weight_loss_challenge :=
  W - W * x / 100 + W * 2 / 100 = W * 86.7 / 100

theorem weight_loss_percentage (h : weight_loss_challenge W x) : x = 15.3 :=
by sorry

end NUMINAMATH_GPT_weight_loss_percentage_l1598_159884


namespace NUMINAMATH_GPT_christina_age_half_in_five_years_l1598_159868

theorem christina_age_half_in_five_years (C Y : ℕ) 
  (h1 : C + 5 = Y / 2)
  (h2 : 21 = 3 * C / 5) :
  Y = 80 :=
sorry

end NUMINAMATH_GPT_christina_age_half_in_five_years_l1598_159868


namespace NUMINAMATH_GPT_shaded_area_l1598_159834

theorem shaded_area (r : ℝ) (α : ℝ) (β : ℝ) (h1 : r = 4) (h2 : α = 1/2) :
  β = 64 - 16 * Real.pi := by sorry

end NUMINAMATH_GPT_shaded_area_l1598_159834


namespace NUMINAMATH_GPT_angle_bisector_length_l1598_159804

variable (a b : ℝ) (α l : ℝ)

theorem angle_bisector_length (ha : 0 < a) (hb : 0 < b) (hα : 0 < α) (hl : l = (2 * a * b * Real.cos (α / 2)) / (a + b)) :
  l = (2 * a * b * Real.cos (α / 2)) / (a + b) := by
  -- problem assumptions
  have h1 : a > 0 := ha
  have h2 : b > 0 := hb
  have h3 : α > 0 := hα
  -- conclusion
  exact hl

end NUMINAMATH_GPT_angle_bisector_length_l1598_159804


namespace NUMINAMATH_GPT_problem_l1598_159882

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Conditions
def condition1 : a + b = 1 := sorry
def condition2 : a^2 + b^2 = 3 := sorry
def condition3 : a^3 + b^3 = 4 := sorry
def condition4 : a^4 + b^4 = 7 := sorry

-- Question and proof
theorem problem : a^10 + b^10 = 123 :=
by
  have h1 : a + b = 1 := condition1
  have h2 : a^2 + b^2 = 3 := condition2
  have h3 : a^3 + b^3 = 4 := condition3
  have h4 : a^4 + b^4 = 7 := condition4
  sorry

end NUMINAMATH_GPT_problem_l1598_159882


namespace NUMINAMATH_GPT_square_perimeter_l1598_159814

-- Define the area of the square
def square_area := 720

-- Define the side length of the square
noncomputable def side_length := Real.sqrt square_area

-- Define the perimeter of the square
noncomputable def perimeter := 4 * side_length

-- Statement: Prove that the perimeter is 48 * sqrt(5)
theorem square_perimeter : perimeter = 48 * Real.sqrt 5 :=
by
  -- The proof is omitted as instructed
  sorry

end NUMINAMATH_GPT_square_perimeter_l1598_159814


namespace NUMINAMATH_GPT_time_difference_for_x_miles_l1598_159851

def time_old_shoes (n : Nat) : Int := 10 * n
def time_new_shoes (n : Nat) : Int := 13 * n
def time_difference_for_5_miles : Int := time_new_shoes 5 - time_old_shoes 5

theorem time_difference_for_x_miles (x : Nat) (h : time_difference_for_5_miles = 15) : 
  time_new_shoes x - time_old_shoes x = 3 * x := 
by
  sorry

end NUMINAMATH_GPT_time_difference_for_x_miles_l1598_159851


namespace NUMINAMATH_GPT_mostSuitableForComprehensiveSurvey_l1598_159836

-- Definitions of conditions
def optionA := "Understanding the sleep time of middle school students nationwide"
def optionB := "Understanding the water quality of a river"
def optionC := "Surveying the vision of all classmates"
def optionD := "Surveying the number of fish in a pond"

-- Define the notion of being the most suitable option for a comprehensive survey
def isSuitableForComprehensiveSurvey (option : String) := option = optionC

-- The theorem statement
theorem mostSuitableForComprehensiveSurvey : isSuitableForComprehensiveSurvey optionC := by
  -- This is the Lean 4 statement where we accept the hypotheses
  -- and conclude the theorem. Proof is omitted with "sorry".
  sorry

end NUMINAMATH_GPT_mostSuitableForComprehensiveSurvey_l1598_159836


namespace NUMINAMATH_GPT_simplify_division_l1598_159821

noncomputable def a := 5 * 10 ^ 10
noncomputable def b := 2 * 10 ^ 4 * 10 ^ 2

theorem simplify_division : a / b = 25000 := by
  sorry

end NUMINAMATH_GPT_simplify_division_l1598_159821


namespace NUMINAMATH_GPT_student_count_incorrect_l1598_159860

theorem student_count_incorrect :
  ∀ k : ℕ, 2012 ≠ 18 + 17 * k :=
by
  intro k
  sorry

end NUMINAMATH_GPT_student_count_incorrect_l1598_159860


namespace NUMINAMATH_GPT_exponential_function_decreasing_l1598_159802

theorem exponential_function_decreasing {a : ℝ} 
  (h : ∀ x y : ℝ, x > y → (a-1)^x < (a-1)^y) : 1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_GPT_exponential_function_decreasing_l1598_159802


namespace NUMINAMATH_GPT_find_divisor_l1598_159859

theorem find_divisor (x : ℝ) (h : 1152 / x - 189 = 3) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1598_159859


namespace NUMINAMATH_GPT_final_sale_price_is_correct_l1598_159839

-- Define the required conditions
def original_price : ℝ := 1200.00
def first_discount_rate : ℝ := 0.10
def second_discount_rate : ℝ := 0.20
def final_discount_rate : ℝ := 0.05

-- Define the expression to calculate the sale price after the discounts
def first_discount_price := original_price * (1 - first_discount_rate)
def second_discount_price := first_discount_price * (1 - second_discount_rate)
def final_sale_price := second_discount_price * (1 - final_discount_rate)

-- Prove that the final sale price equals $820.80
theorem final_sale_price_is_correct : final_sale_price = 820.80 := by
  sorry

end NUMINAMATH_GPT_final_sale_price_is_correct_l1598_159839


namespace NUMINAMATH_GPT_unique_solution_for_k_l1598_159807

theorem unique_solution_for_k : 
  ∃! k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, (x + 3) / (k * x - 2) = x ↔ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_for_k_l1598_159807


namespace NUMINAMATH_GPT_nine_op_ten_l1598_159840

def op (A B : ℕ) : ℚ := (1 : ℚ) / (A * B) + (1 : ℚ) / ((A + 1) * (B + 2))

theorem nine_op_ten : op 9 10 = 7 / 360 := by
  sorry

end NUMINAMATH_GPT_nine_op_ten_l1598_159840


namespace NUMINAMATH_GPT_number_of_dogs_per_box_l1598_159866

-- Definition of the problem
def num_boxes : ℕ := 7
def total_dogs : ℕ := 28

-- Statement of the theorem to prove
theorem number_of_dogs_per_box (x : ℕ) (h : num_boxes * x = total_dogs) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dogs_per_box_l1598_159866


namespace NUMINAMATH_GPT_solve_for_y_l1598_159880

theorem solve_for_y (y : ℝ) (h : y ≠ 2) :
  (7 * y / (y - 2) - 4 / (y - 2) = 3 / (y - 2)) → y = 1 :=
by
  intro h_eq
  sorry

end NUMINAMATH_GPT_solve_for_y_l1598_159880
