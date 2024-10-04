import Mathlib
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Real
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Graph
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Trig
import Mathlib.Data.Set.Basic
import Mathlib.Data.Time.Clock
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.NumberTheory.ZsigersTheorem
import Mathlib.Probability.Basic
import Mathlib.Statistics.Regression.Linear
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Euclidean.EuclideanSpace
import ProbabilityTheory

namespace product_of_four_consecutive_integers_divisible_by_12_l417_417014

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417014


namespace projectile_height_time_l417_417695

-- Define constants and the height function
def a : ℝ := -4.9
def b : ℝ := 29.75
def c : ℝ := -35
def y (t : ℝ) : ℝ := a * t^2 + b * t

-- Problem statement
theorem projectile_height_time (h : y t = 35) : ∃ t : ℝ, 0 < t ∧ abs (t - 1.598) < 0.001 := by
  -- Placeholder for actual proof
  sorry

end projectile_height_time_l417_417695


namespace total_employees_l417_417691

theorem total_employees (N : ℕ) (N_S : ℕ) : 
  (N_S = N - 40) ∧ 
  (0.75 * N_S) ∧ 
  (0.65 * N_S) ∧ 
  ((25 / 100) * N_S + (65 / 100) * N_S = 0.90 * N_S) ∧ 
  (0.90 * N_S = (5 / 6) * N) ∧ 
  (N_S = N - 40) → 
  N = 540 := by
  sorry

end total_employees_l417_417691


namespace euclid_unique_students_l417_417847

theorem euclid_unique_students :
  let pythagoras_students := 15
  let lovelace_students := 12
  let euler_students := 10
  let overlap := 3
  let total_students := pythagoras_students + lovelace_students + euler_students - overlap
  total_students = 34 :=
by
  let pythagoras_students := 15
  let lovelace_students := 12
  let euler_students := 10
  let overlap := 3
  let total_students := pythagoras_students + lovelace_students + euler_students - overlap
  show total_students = 34 from sorry

end euclid_unique_students_l417_417847


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417012

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417012


namespace solve_for_y_l417_417945

theorem solve_for_y (x y : ℝ) (h : 2 * x - 3 * y = 4) : y = (2 * x - 4) / 3 :=
sorry

end solve_for_y_l417_417945


namespace total_employees_l417_417692

theorem total_employees (N : ℕ) (N_S : ℕ) : 
  (N_S = N - 40) ∧ 
  (0.75 * N_S) ∧ 
  (0.65 * N_S) ∧ 
  ((25 / 100) * N_S + (65 / 100) * N_S = 0.90 * N_S) ∧ 
  (0.90 * N_S = (5 / 6) * N) ∧ 
  (N_S = N - 40) → 
  N = 540 := by
  sorry

end total_employees_l417_417692


namespace socks_ratio_l417_417345

theorem socks_ratio (b : ℕ) (x : ℕ) (h : 1.6 * (15 * x + b * x) = 3 * b * x + 5 * x) :
  5 / b = 7 / 19 :=
by
  sorry

end socks_ratio_l417_417345


namespace average_age_of_others_when_youngest_was_born_l417_417072

noncomputable def average_age_when_youngest_was_born (total_people : ℕ) (average_age : ℕ) (youngest_age : ℕ) : ℚ :=
  let total_age := total_people * average_age
  let age_without_youngest := total_age - youngest_age
  age_without_youngest / (total_people - 1)

theorem average_age_of_others_when_youngest_was_born :
  average_age_when_youngest_was_born 7 30 7 = 33.833 :=
by
  sorry

end average_age_of_others_when_youngest_was_born_l417_417072


namespace locus_of_P_l417_417986

-- Defining given points and lines
structure Point :=
(x : ℝ) (y : ℝ)

def A : Point := ⟨0, 4 / 3⟩
def B : Point := ⟨-1, 0⟩
def C : Point := ⟨1, 0⟩

-- Definition of distance from a point to a line
def dist_to_line (P : Point) (a b c : ℝ) : ℝ :=
  abs (a * P.x + b * P.y + c) / real.sqrt (a^2 + b^2)

-- Defining the lines AB, AC, and BC
def line_AB (P : Point) : ℝ := 4 * P.x - 3 * P.y + 4
def line_AC (P : Point) : ℝ := 4 * P.x + 3 * P.y - 4
def line_BC (P : Point) : ℝ := P.y

-- Distances from point P to the lines
def d1 (P : Point) : ℝ := dist_to_line P 4 (-3) 4
def d2 (P : Point) : ℝ := dist_to_line P 4 3 (-4)
def d3 (P : Point) : ℝ := dist_to_line P 0 1 0

-- Given geometric mean condition
def geom_mean_condition (P : Point) : Prop :=
  d3 P = real.sqrt ((d1 P) * (d2 P))

-- Equation of locus
def equation1 (P : Point) : Prop :=
  8 * P.x^2 - 17 * P.y^2 + 12 * P.y - 8 = 0

def equation2 (P : Point) : Prop :=
  2 * P.x^2 + 2 * P.y^2 + 3 * P.y - 2 = 0

-- Final proof statement
theorem locus_of_P (P : Point) (h : geom_mean_condition P) :
  equation1 P ∨ equation2 P :=
sorry

end locus_of_P_l417_417986


namespace fold_points_area_proof_l417_417159

-- Define the Basic Structure of Triangle and Points
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (D E F : Point)
  (DE : ℝ)
  (DF : ℝ)
  (angle_E : ℝ)

def P_within_triangle (T : Triangle) (P : Point) : Prop :=
  -- Note: Assuming some definition that checks if point P is within triangle T
  true -- This should be replaced by the actual definition

def fold_points_area (T : Triangle) (P : Point) : ℝ :=
  156 * Real.pi - 144 * Real.sqrt 3

-- Lean Statement
theorem fold_points_area_proof (T : Triangle) (P : Point)
  (h_DE : T.DE = 24)
  (h_DF : T.DF = 48)
  (h_angle_E : T.angle_E = Real.pi / 2)
  (h_P_within : P_within_triangle T P) :
  fold_points_area T P = (156 * Real.pi - 144 * Real.sqrt 3) := sorry

end fold_points_area_proof_l417_417159


namespace area_of_first_square_l417_417428

-- Definitions derived from the problem
def perimeter_of_square_B : Nat := 12
def probability_not_within_B : Float := 0.64

-- The side length and area calculation for square B based on the perimeter
def side_length_of_square_B : Nat := perimeter_of_square_B / 4
def area_of_square_B : Nat := side_length_of_square_B ^ 2

-- The main theorem we need to prove
theorem area_of_first_square : Nat :=
  let area_of_square_B : Nat := 9 -- since side length of square B is 3 cm
  let probability_within_B : Float := 0.36 -- since 1 - probability_not_within_B = 0.36
  let area_first_square : Float := area_of_square_B / probability_within_B
  by
    have h : area_first_square = 25.0 := by sorry
    exact h

end area_of_first_square_l417_417428


namespace molecular_weights_correct_l417_417459

-- Define atomic weights
def atomic_weight_Al : Float := 26.98
def atomic_weight_Cl : Float := 35.45
def atomic_weight_K : Float := 39.10

-- Define molecular weight calculations
def molecular_weight_AlCl3 : Float :=
  atomic_weight_Al + 3 * atomic_weight_Cl

def molecular_weight_KCl : Float :=
  atomic_weight_K + atomic_weight_Cl

-- Theorem statement to prove
theorem molecular_weights_correct :
  molecular_weight_AlCl3 = 133.33 ∧ molecular_weight_KCl = 74.55 :=
by
  -- This is where we would normally prove the equivalence
  sorry

end molecular_weights_correct_l417_417459


namespace magnitude_c_l417_417530

open Real

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Conditions

-- a and b are unit vectors
axiom unit_a : ∥a∥ = 1
axiom unit_b : ∥b∥ = 1

-- The angle between a and b is 60 degrees, hence their dot product is 1/2
axiom dot_ab : inner a b = 1/2

-- Question: Find the magnitude of c = 3a + b
def vec_c := 3 • a + b

-- Expected Result: |c| = √13
theorem magnitude_c : ∥vec_c∥ = sqrt 13 :=
by
  -- Proof is not included, adding sorry as a placeholder
  sorry

end magnitude_c_l417_417530


namespace eccentricity_of_ellipse_l417_417923

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt (1 - (b^2) / (a^2)))

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b > 0) (h2 : Real.sqrt 3 * a = 2 * b) :
  eccentricity a b = 1 / 2 :=
by
  sorry

noncomputable def ellipse_eqn (a b c : ℝ) (h1 : a > b > 0) (h2 : b = Real.sqrt 3 / 2 * a) (h3 : c = 1 / 2 * a)
  (h4 : let ey := line_through_points (-c,0) (c * 3/4, 3/4*a) in tangent_circle_tangent_to_xaxis_and_line ey (4,2)) :
  ∀ x y, (x^2) / (4 * c^2) + (y^2) / (3 * c^2) = 1 ↔ (x^2) / 16 + (y^2) / 12 = 1 :=
by
  sorry

end eccentricity_of_ellipse_l417_417923


namespace greatest_value_NNM_l417_417491

theorem greatest_value_NNM :
  ∃ (M : ℕ), (M * M % 10 = M) ∧ (∃ (MM : ℕ), MM = 11 * M ∧ (MM * M = 396)) :=
by
  sorry

end greatest_value_NNM_l417_417491


namespace flea_treatment_problem_l417_417818

/-- One flea treatment halves the flea population.
    After four treatments, the dog has 14 fleas remaining.
    The number of additional fleas before treatments compared to after four treatments is 210. -/
theorem flea_treatment_problem :
  ∃ (initial_fleas : ℕ), ((initial_fleas / 2 / 2 / 2 / 2) = 14) ∧ (initial_fleas - 14 = 210) :=
begin
  sorry,
end

end flea_treatment_problem_l417_417818


namespace min_value_3x_plus_4y_l417_417969

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 28 :=
sorry

end min_value_3x_plus_4y_l417_417969


namespace carol_optimal_choice_l417_417840

/-- Define the uniform random variables for Alice and Bob --/
noncomputable def alice : ℝ → ℝ := λ x, if (0 ≤ x ∧ x ≤ 1) then x else 0
noncomputable def bob : ℝ → ℝ := λ y, if (1/3 ≤ y ∧ y ≤ 2/3) then y else 0

/-- Define the win condition for Carol --/
def carol_wins (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

/-- Prove the winning value for Carol --/
theorem carol_optimal_choice : ∃ (c : ℝ), c = 13 / 24 ∧ ∀ (c' : ℝ), carol_wins (alice c') (bob c') c → c ≥ c' := 
begin
  sorry
end

end carol_optimal_choice_l417_417840


namespace sin_18_cos_36_eq_quarter_l417_417392

theorem sin_18_cos_36_eq_quarter : Real.sin (Real.pi / 10) * Real.cos (Real.pi / 5) = 1 / 4 :=
by
  sorry

end sin_18_cos_36_eq_quarter_l417_417392


namespace minimum_value_S15_minus_S10_l417_417920

theorem minimum_value_S15_minus_S10 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom_seq : ∀ n, S (n + 1) = S n * a (n + 1))
  (h_pos_terms : ∀ n, a n > 0)
  (h_arith_seq : S 10 - 2 * S 5 = 3)
  (h_geom_sub_seq : (S 10 - S 5) * (S 10 - S 5) = S 5 * (S 15 - S 10)) :
  ∃ m, m = 12 ∧ (S 15 - S 10) ≥ m := sorry

end minimum_value_S15_minus_S10_l417_417920


namespace empty_one_vessel_possible_l417_417340

theorem empty_one_vessel_possible (a b c : ℕ) :
  (∃ (x y z : ℕ), (x + y + z = a + b + c) ∧ (x = 0 ∨ y = 0 ∨ z = 0)) :=
begin
  -- Problem statement to be proven here.
  sorry
end

end empty_one_vessel_possible_l417_417340


namespace side_length_of_inscribed_square_l417_417798

theorem side_length_of_inscribed_square
  (S1 S2 S3 : ℝ)
  (hS1 : S1 = 1) (hS2 : S2 = 3) (hS3 : S3 = 1) :
  ∃ (x : ℝ), S1 = 1 ∧ S2 = 3 ∧ S3 = 1 ∧ x = 2 := 
by
  sorry

end side_length_of_inscribed_square_l417_417798


namespace flea_treatment_problem_l417_417819

/-- One flea treatment halves the flea population.
    After four treatments, the dog has 14 fleas remaining.
    The number of additional fleas before treatments compared to after four treatments is 210. -/
theorem flea_treatment_problem :
  ∃ (initial_fleas : ℕ), ((initial_fleas / 2 / 2 / 2 / 2) = 14) ∧ (initial_fleas - 14 = 210) :=
begin
  sorry,
end

end flea_treatment_problem_l417_417819


namespace nero_speed_l417_417606

theorem nero_speed (jerome_speed : ℝ) (jerome_time : ℝ) (nero_time : ℝ) :
  jerome_speed = 4 → jerome_time = 6 → nero_time = 3 → 
  ∃ nero_speed : ℝ, nero_speed = 8 :=
by
  intros h1 h2 h3
  use jerome_speed * jerome_time / nero_time
  rw [h1, h2, h3]
  norm_num
  sorry

end nero_speed_l417_417606


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417020

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417020


namespace side_CD_line_eq_l417_417539

-- Definitions based on the given conditions
def center_of_parallelogram := (0, 3) : ℝ × ℝ
def side_AB_line_eq : ℝ → ℝ → Prop := λ x y, 3 * x + 4 * y - 2 = 0

-- The proof problem statement
theorem side_CD_line_eq (x y : ℝ) :
  center_of_parallelogram = (0, 3) →
  side_AB_line_eq x y →
  (∃ x y, 3 * x + 4 * y - 22 = 0) :=
begin
  intro h1,
  intro h2,
  use (-x),
  use (6 - y),
  sorry
end

end side_CD_line_eq_l417_417539


namespace complex_number_in_second_quadrant_l417_417991

theorem complex_number_in_second_quadrant :
  let z : ℂ := 5 * complex.I / (2 - complex.I)
  (z.re < 0) ∧ (0 < z.im) :=
by
  sorry

end complex_number_in_second_quadrant_l417_417991


namespace complement_U_A_l417_417949

-- Definitions based on conditions
def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

-- Statement of the problem
theorem complement_U_A :
  (U \ A) = {4} :=
by
  sorry

end complement_U_A_l417_417949


namespace avg_ABC_l417_417619

variables (A B C : Set ℕ) -- Sets of people
variables (a b c : ℕ) -- Numbers of people in sets A, B, and C respectively
variables (sum_A sum_B sum_C : ℕ) -- Sums of the ages of people in sets A, B, and C respectively

-- Given conditions
axiom avg_A : sum_A / a = 30
axiom avg_B : sum_B / b = 20
axiom avg_C : sum_C / c = 45

axiom avg_AB : (sum_A + sum_B) / (a + b) = 25
axiom avg_AC : (sum_A + sum_C) / (a + c) = 40
axiom avg_BC : (sum_B + sum_C) / (b + c) = 32

theorem avg_ABC : (sum_A + sum_B + sum_C) / (a + b + c) = 35 :=
by
  sorry

end avg_ABC_l417_417619


namespace total_kids_played_l417_417615

def kids_played_week (monday tuesday wednesday thursday: ℕ): ℕ :=
  let friday := thursday + (thursday * 20 / 100)
  let saturday := friday - (friday * 30 / 100)
  let sunday := 2 * monday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem total_kids_played : 
  kids_played_week 15 18 25 30 = 180 :=
by
  sorry

end total_kids_played_l417_417615


namespace eval_expr_at_a_eq_3_l417_417127

theorem eval_expr_at_a_eq_3 :
  let a := 3
  in (\frac{3 * a⁻¹ + \frac{2 * a⁻¹}{3}}{2 * a} : ℚ) = 11 / 54 :=
by
  sorry

end eval_expr_at_a_eq_3_l417_417127


namespace paper_tape_information_count_l417_417447

theorem paper_tape_information_count (n : ℕ) (h : n = 8) : 
  let positionChoices := 2 in 
  (positionChoices ^ n) = 256 := by 
  sorry

end paper_tape_information_count_l417_417447


namespace prob_at_least_one_2_in_two_8_sided_dice_l417_417359

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end prob_at_least_one_2_in_two_8_sided_dice_l417_417359


namespace complex_in_fourth_quadrant_l417_417626

noncomputable def complex_quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "first quadrant"
  else if z.re < 0 ∧ z.im > 0 then "second quadrant"
  else if z.re < 0 ∧ z.im < 0 then "third quadrant"
  else if z.re > 0 ∧ z.im < 0 then "fourth quadrant"
  else "on an axis"

theorem complex_in_fourth_quadrant : complex_quadrant ((2 - complex.I) / (1 + complex.I)) = "fourth quadrant" :=
sorry

end complex_in_fourth_quadrant_l417_417626


namespace exists_infinitely_many_coprime_pairs_l417_417873

theorem exists_infinitely_many_coprime_pairs :
  ∃^∞ (a b : ℕ), a ≠ b ∧ Nat.Coprime a b ∧ (a * b ∣ a^2 + b^2 + 1) :=
sorry

end exists_infinitely_many_coprime_pairs_l417_417873


namespace constant_term_polynomial_l417_417721

theorem constant_term_polynomial :
  (∃ n : ℕ, (∑ i in finset.range n.succ, (2-1+1)^i) = 64) → 
  (∃ c : ℕ, c = 200) :=
by
  sorry

end constant_term_polynomial_l417_417721


namespace height_diff_greater_l417_417249

theorem height_diff_greater (A B C H : Type) [Field A] [Field B] [Field C] [Field H]
  (triangle_ABC : A ≠ B → B ≠ C → A ≠ C → true)
  (AC_gt_AB : A → B → C → A ≠ B → A ≠ C → (norm (C - A)) > (norm (B - A)))
  (height_AH : A → H → B → C → true) :
  ∃ H, (norm (C - H) - norm (B - H)) > (norm (C - A) - norm (B - A)) :=
by
  sorry

end height_diff_greater_l417_417249


namespace cards_probability_comparison_l417_417228

noncomputable def probability_case_a : ℚ :=
  (Nat.choose 13 10) * (Nat.choose 39 3) / Nat.choose 52 13

noncomputable def probability_case_b : ℚ :=
  4 ^ 13 / Nat.choose 52 13

theorem cards_probability_comparison :
  probability_case_b > probability_case_a :=
  sorry

end cards_probability_comparison_l417_417228


namespace possible_values_count_l417_417941

theorem possible_values_count :
  ∃ (φ : ℝ), (∀ φ, (abs φ < 2016 * π) → (f (x - π / 6) = g x) ∧ (∀ φ, f x = sin (2 * x + φ)))
  → (set.count (λ φ, ∃ k : ℤ, φ = 2 * k * π + π / 3 ∧ abs φ < 2016 * π) = 2016) :=
sorry

end possible_values_count_l417_417941


namespace ivans_profit_l417_417610

def price_meat_per_kg : ℕ := 500
def kg_meat_sold : ℕ := 100
def price_eggs_per_dozen : ℕ := 50
def eggs_sold : ℕ := 20000
def annual_expenses : ℕ := 100000

def revenue_meat : ℕ := kg_meat_sold * price_meat_per_kg
def revenue_eggs : ℕ := eggs_sold * (price_eggs_per_dozen / 10)
def total_revenue : ℕ := revenue_meat + revenue_eggs

def profit : ℕ := total_revenue - annual_expenses

theorem ivans_profit : profit = 50000 := by
  sorry

end ivans_profit_l417_417610


namespace rhombus_diagonals_perpendicular_l417_417196

theorem rhombus_diagonals_perpendicular (Q : Type) [quadrilateral Q] :
  (∀ (r : rhombus Q), perpendicular_diagonals r) ∧
  (∃ (q : quadrilateral Q), perpendicular_diagonals q → ¬ (is_rhombus q)) ∧
  (∀ (q : quadrilateral Q), ¬perpendicular_diagonals q → ¬is_rhombus q) ∧
  (∀ (q : quadrilateral Q), ¬is_rhombus q → ¬perpendicular_diagonals q) :=
by
  sorry

end rhombus_diagonals_perpendicular_l417_417196


namespace sum_reciprocal_roots_l417_417264

noncomputable def poly : Polynomial ℂ :=
  Polynomial.sum (Finset.range 500.filter_even) (λ n, Polynomial.X^n) - 500

theorem sum_reciprocal_roots :
  let roots := poly.roots in
  roots.length = 500 →
  ∑ n in Finset.range 500, 2 / (roots.get n?. - 1) = 2 :=
by
  intros
  sorry

end sum_reciprocal_roots_l417_417264


namespace percentage_increase_costume_l417_417256

theorem percentage_increase_costume (deposit_fraction : ℝ) (last_year_cost : ℝ) (remaining_payment : ℝ) (this_year_cost : ℝ) (percentage_increase : ℝ) :
  deposit_fraction = 0.10 →
  last_year_cost = 250 →
  remaining_payment = 315 →
  this_year_cost = remaining_payment / (1 - deposit_fraction) →
  percentage_increase = ((this_year_cost - last_year_cost) / last_year_cost) * 100 →
  percentage_increase = 40 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end percentage_increase_costume_l417_417256


namespace base_85_solution_l417_417867

theorem base_85_solution (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 16) :
  (352936524 - b) % 17 = 0 ↔ b = 4 :=
by
  sorry

end base_85_solution_l417_417867


namespace avg_daily_production_l417_417065

theorem avg_daily_production (x y : ℕ) (h1 : x + y = 350) (h2 : 2 * x - y = 250) : x = 200 ∧ y = 150 := 
by
  sorry

end avg_daily_production_l417_417065


namespace exists_k0_for_all_k_greater_p_sqrt_sum_square_leq_l417_417671

theorem exists_k0_for_all_k_greater_p_sqrt_sum_square_leq :
  ∀ (p : ℕ), p ≥ 1 → ∃ k0 : ℕ, ∀ k : ℕ, k > k0 →
  (∑ (i : ℕ) in finset.range (p+1), real.sqrt (k + i)) ^ 2 ≤ (1 / 2) * (p+1)^2 * (2*k + p) - 1 :=
by
  intros
  sorry

end exists_k0_for_all_k_greater_p_sqrt_sum_square_leq_l417_417671


namespace max_acute_angles_in_non_convex_n_gon_l417_417043

theorem max_acute_angles_in_non_convex_n_gon (n : ℕ) (h_n : 3 ≤ n) :
  ∃ k : ℕ, k ≤ (2 * n) / 3 + 1 ∧
  (∀ θ : ℕ, θ < 90 → θ ∈ set.Ico 0 k → acute θ) ∧ 
  ∑ θ in (set.range k), (θ) = (n - 2) * 180 :=
begin
  sorry
end

def acute (θ : ℕ) : Prop :=
  θ < 90

end max_acute_angles_in_non_convex_n_gon_l417_417043


namespace hyperbola_focal_length_l417_417175

-- Define the given conditions
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- Prove that the focal length is 4 given the conditions
theorem hyperbola_focal_length (m : ℝ) (hm : m > 0) :
  (∀ x y, hyperbola_eq x y m) →
  (∀ x y, asymptote_eq x y m) →
  (2 * sqrt (m + 1) = 4) :=
by
  intros; sorry

end hyperbola_focal_length_l417_417175


namespace negation_of_p_l417_417547

variable (x : ℝ)

def proposition_p : Prop := ∀ x : ℝ, 2^x > 0

theorem negation_of_p : ¬ proposition_p ↔ ∃ x : ℝ, 2^x ≤ 0 := by
  sorry

end negation_of_p_l417_417547


namespace max_acute_angles_non_convex_ngon_l417_417045

def max_acute_angles (n : ℕ) : ℕ :=
  ⌊(2 * n / 3 : ℚ)⌋ + 1

theorem max_acute_angles_non_convex_ngon (n : ℕ) (h1 : n ≥ 3) (h2 : (∀ k : ℕ, k < n → k * 90 < (n-2) * 180 - (n-k) * 360)) 
  (h3 : (∀ k : ℕ, k < n → ∀ m : ℕ, m = n - k → m * 180 > m * 180)) : 
  ∀ k : ℕ, k ≤ max_acute_angles n :=
sorry

end max_acute_angles_non_convex_ngon_l417_417045


namespace square_1033_l417_417856

theorem square_1033 :
  let a := 1000
  let b := 33
  (a + b)^2 = a^2 + 2 * a * b + b^2 → 
  1033^2 = 1067089 :=
by
  let a := 1000
  let b := 33
  have h : (a + b)^2 = a^2 + 2 * a * b + b^2 := sorry
  calc 
    1033^2 = (1000 + 33)^2 : by rw [←(a + b)]
    ... = a^2 + 2 * a * b + b^2 : by rw [h]
    ... = 1000000 + 66000 + 1089 : by sorry
    ... = 1067089 : by sorry

end square_1033_l417_417856


namespace arithmetic_sequence_sum_l417_417237

open_locale big_operators

/-- In an arithmetic sequence {a_n}, a_3 + a_9 = 27 - a_6, prove that the sum of the first 11 terms is 99. -/
theorem arithmetic_sequence_sum {a : ℕ → ℤ} (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_eq : a 3 + a 9 = 27 - a 6) : 
  (∑ i in finset.range 11, a i) = 99 := 
sorry

end arithmetic_sequence_sum_l417_417237


namespace switch_connections_l417_417746

theorem switch_connections (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 :=
by
  rw [h1, h2]
  norm_num
  sorry

end switch_connections_l417_417746


namespace base_five_product_correct_l417_417775

-- Declare that we will be working with noncomputable necessary items
noncomputable def base_five_product : ℕ :=
let num1 := 1 * 5^3 + 3 * 5^2 + 2 * 5^1 + 4 * 5^0 in
let num2 := 2 * 5^1 + 3 * 5^0 in
let product := num1 * num2 in
let p0 := product % 5 in
let p1 := (product / 5) % 5 in
let p2 := (product / 5^2) % 5 in
let p3 := (product / 5^3) % 5 in
let p4 := (product / 5^4) % 5 in
p4 * 10000 + p3 * 1000 + p2 * 100 + p1 * 10 + p0

theorem base_five_product_correct : base_five_product = 42112 := by
  sorry

end base_five_product_correct_l417_417775


namespace no_partition_of_six_consecutive_numbers_product_equal_l417_417601

theorem no_partition_of_six_consecutive_numbers_product_equal (n : ℕ) :
  ¬ ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range (n+6) ∧ 
    A ∩ B = ∅ ∧ 
    A.prod id = B.prod id :=
by
  sorry

end no_partition_of_six_consecutive_numbers_product_equal_l417_417601


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417025

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417025


namespace locus_of_equidistant_points_l417_417777

variables {P O₁ O₂ : Type*}
variables [metric_space P]

-- Define the centers O1 and O2 of the circles
variables (O₁ O₂ : P)
-- Define the radii of the circles
variables (R r : ℝ)
-- Define a point P which is equidistant from the circumferences of the two circles
variable (P : P)

-- Define the distances from point P to the circumferences of the circles
def distance_to_larger_circle := dist P O₁ - R
def distance_to_smaller_circle := dist P O₂ - r

-- Equidistance condition using absolute values
def equidistant_to_circumferences :=
  abs (dist P O₁ - R) = abs (dist P O₂ - r)

-- Goal: The locus of point P is an ellipse described by the conditions
theorem locus_of_equidistant_points :
  equidistant_to_circumferences P O₁ O₂ R r →
  ∃ (a : ℝ), a = (R + r) ∨ a = (R - r) ∧
    ∀ P : P, dist P O₁ + dist P O₂ = a :=
sorry

end locus_of_equidistant_points_l417_417777


namespace probability_of_sum_greater_than_one_third_l417_417667

open Set

noncomputable def probability_sum_greater_than_one_third : ℝ :=
  let S : Set (ℝ × ℝ) := { p | 0 < p.1 ∧ p.1 < 1 ∧ 0 < p.2 ∧ p.2 < 1 ∧ p.1 + p.2 > 1/3 }
  let interval := Ioo (0 : ℝ) 1
  let measure := Volume.intervalLebesgueMeasure
  (measure (prod interval interval) S) / (measure (prod interval interval))

theorem probability_of_sum_greater_than_one_third :
  probability_sum_greater_than_one_third = 17 / 18 :=
sorry

end probability_of_sum_greater_than_one_third_l417_417667


namespace class_average_is_86_l417_417815

def class_average (total_students : ℕ) (boys_percentage : ℝ) (boys_score : ℝ) (girls_score : ℝ) : ℝ :=
  let boys := total_students * boys_percentage
  let girls := total_students * (1 - boys_percentage)
  let total_boys_score := boys * boys_score
  let total_girls_score := girls * girls_score
  (total_boys_score + total_girls_score) / total_students

theorem class_average_is_86 :
  class_average 100 0.4 0.8 0.9 = 0.86 :=
by
  simp [class_average]
  -- Prove the equality 40 * 0.8 + 60 * 0.9 / 100 = 86%
  sorry

end class_average_is_86_l417_417815


namespace largest_number_in_6x6_interesting_board_no_interesting_7x7_board_l417_417809

/-- A board is called interesting if:
 1. In all unit squares below the main diagonal, the number 0 is written;
 2. Positive integers are written in all other unit squares;
 3. The sums of all n rows and the sums of all n columns are the numbers 1, 2, ..., 2n (not necessarily in that order).
-/
structure InterestingBoard (n : ℕ) :=
  (values : fin n → fin n → ℕ)
  (below_diagonal_zero : ∀ i j, i > j → values i j = 0)
  (positive_elsewhere : ∀ i j, i ≤ j → values i j > 0)
  (row_column_sums : multiset (fin (2 * n)) → Prop)

/-- A proof that the largest number that can appear in a 6 × 6 interesting board is 12. -/
theorem largest_number_in_6x6_interesting_board :
  ∃ B : InterestingBoard 6, ∀ i j, B.values i j ≤ 12 ∧ (∃ i j, B.values i j = 12) := 
sorry

/-- A proof that there is no interesting board of dimensions 7 × 7. -/
theorem no_interesting_7x7_board : ¬ (∃ B : InterestingBoard 7, true) := 
sorry

end largest_number_in_6x6_interesting_board_no_interesting_7x7_board_l417_417809


namespace original_weight_before_processing_l417_417054

variable (original_weight : ℝ)

def processed_weight (lost_percentage : ℝ) (original_weight : ℝ) : ℝ :=
  original_weight * (1 - lost_percentage / 100)

theorem original_weight_before_processing :
  processed_weight 35 original_weight = 580 → original_weight = 892 :=
by
  intro h
  sorry

end original_weight_before_processing_l417_417054


namespace graduates_job_choice_preference_l417_417874

theorem graduates_job_choice_preference (total_resumes : ℕ)
  (pos_A : ℕ) (pos_B : ℕ)
  (resumes_A : ℕ → ℕ → ℕ)
  (resumes_B : ℕ → ℕ → ℕ)
  (k1 : ℝ) (k2 : ℝ)
  (p_critical : ℝ → ℝ)
  (h_total_resumes : total_resumes = 100)
  (h_pos_A : pos_A = 2)
  (h_pos_B : pos_B = 3)
  (h_resumes_A : ∀ h1 h2, resumes_A h1 h2)
  (h_resumes_B : ∀ h1 h2, resumes_B h1 h2)
  (h_k1 : k1 ≈ 1.010)
  (h_k2 : k2 ≈ 9.090)
  (h_p_critical : ∀ p, p_critical p)
  : (k1 < k2) := sorry

end graduates_job_choice_preference_l417_417874


namespace greatest_distance_between_A_and_B_l417_417591

def A : set ℂ := {z | z^4 = 16}
def B : set ℂ := {z | z^3 - 8*z^2 - 16*z + 64 = 0}

noncomputable def greatest_distance : ℝ := 
  supr (λ a : ℂ, supr (λ b : ℂ, complex.abs (a - b))) 

theorem greatest_distance_between_A_and_B :
  greatest_distance (A, B) = 10 * real.sqrt(1 + 0.64 * real.sqrt(2)) :=
sorry

end greatest_distance_between_A_and_B_l417_417591


namespace range_gauss_function_l417_417506

noncomputable def gauss_function (x : ℝ) : ℤ := ⌊x⌋

def f (x : ℝ) : ℝ := 2 - (3 / (2^x + 1))

theorem range_gauss_function :
  (∀ x : ℝ, gauss_function (f x) ∈ {-1, 0, 1}) :=
by
  sorry

end range_gauss_function_l417_417506


namespace ratio_expression_l417_417207

-- Given conditions: X : Y : Z = 3 : 2 : 6
def ratio (X Y Z : ℚ) : Prop := X / Y = 3 / 2 ∧ Y / Z = 2 / 6

-- The expression to be evaluated
def expr (X Y Z : ℚ) : ℚ := (4 * X + 3 * Y) / (5 * Z - 2 * X)

-- The proof problem itself
theorem ratio_expression (X Y Z : ℚ) (h : ratio X Y Z) : expr X Y Z = 3 / 4 := by
  sorry

end ratio_expression_l417_417207


namespace time_after_3250_minutes_final_answer_is_A_l417_417375

open Nat

/-- 
  What time is it 3250 minutes after 3:00 AM on January 1, 2020?
-/
noncomputable def minutes_from (start : posix_time) (minutes : ℕ) : posix_time :=
  posix_time.add_seconds start (minutes * 60)

/-- 
  The start time is January 1, 2020 at 3:00 AM 
-/
def start_time : posix_time := 
  posix_time.mk 1577865600 -- Timestamp for January 1, 2020 00:00:00 UTC 
  + 3 * 60 * 60           -- Adding 3 hours in seconds

/-- 
  The expected time after 3250 minutes
-/
def expected_time : posix_time := 
  start_time + (3250 * 60) -- Converting 3250 minutes to seconds

/--
  Prove that 3250 minutes after 3:00 AM on January 1, 2020 is January 3, 2020 at 9:10 AM.
-/
theorem time_after_3250_minutes : minutes_from start_time 3250 = expected_time :=
by
  sorry

theorem final_answer_is_A : 
  minutes_from start_time 3250 == expected_time :=
by
  rw [time_after_3250_minutes]
  sorry

end time_after_3250_minutes_final_answer_is_A_l417_417375


namespace books_sold_correct_l417_417707

-- Define the given conditions
def total_books : ℕ := 9900
def fraction_left : ℚ := 4/6

-- Define how many books are left
def books_left (total: ℕ) (fraction: ℚ) : ℕ := (fraction * total).natAbs

-- Define the number of books sold
def books_sold (total: ℕ) (left: ℕ) : ℕ := total - left

-- State the goal
theorem books_sold_correct : books_sold total_books (books_left total_books fraction_left) = 3300 := 
by
  sorry

end books_sold_correct_l417_417707


namespace range_of_a_l417_417545

noncomputable def f (x a : ℝ) : ℝ := x^3 - a*x^2 + 4

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 3 < a :=
by
  sorry

end range_of_a_l417_417545


namespace books_sold_l417_417709

theorem books_sold (total_books : ℕ) (fraction_left : ℚ)
  (h1 : total_books = 9900)
  (h2 : fraction_left = 4/6) :
  (total_books - (total_books * fraction_left).natAbs) = 3300 :=
by sorry

end books_sold_l417_417709


namespace units_digit_of_n_squared_plus_two_to_the_n_is_one_l417_417628

theorem units_digit_of_n_squared_plus_two_to_the_n_is_one :
  let n := 1234^2 + 2^1234 in
  (n^2 + 2^n) % 10 = 1 := by
  let n := 1234^2 + 2^1234
  sorry

end units_digit_of_n_squared_plus_two_to_the_n_is_one_l417_417628


namespace jack_leftover_money_l417_417251

theorem jack_leftover_money :
  let saved_money_base8 : ℕ := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 : ℕ := 1200
  saved_money_base8 - ticket_cost_base10 = 847 :=
by
  let saved_money_base8 := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 := 1200
  show saved_money_base8 - ticket_cost_base10 = 847
  sorry

end jack_leftover_money_l417_417251


namespace find_focal_length_of_hyperbola_l417_417180

def focal_length_of_hyperbola (m : ℝ) (h_m_gt_zero : m > 0) (h_asymptote : sqrt 3 * x + m * y = 0) : ℝ :=
4

theorem find_focal_length_of_hyperbola (m : ℝ) (h_m_gt_zero : m > 0) (h_asymptote : sqrt 3 * x + m * y = 0) :
  focal_length_of_hyperbola m h_m_gt_zero h_asymptote = 4 :=
sorry

end find_focal_length_of_hyperbola_l417_417180


namespace exists_k_l417_417952

open Set Real

variables {A B : Set ℝ} -- Sets on the circumference of the circle
variables {m : ℕ} [h_m : Fact (0 < m)] -- m is a positive natural number

-- Length function l(M) denotes the total length of all the arcs in M
def l (M : Set ℝ) : ℝ := sorry -- This would be a proper def in a full proof

-- Define the rotation
def rotate (M : Set ℝ) (θ : ℝ) : Set ℝ := sorry -- This would rotate set M by θ

-- Define the sets A^j
def A_j (j : ℕ) : Set ℝ := rotate A (j * π / m)

-- Statement of the problem
theorem exists_k (hA : Finite A) (hB : Finite B)
    (hB_length : ∀ b ∈ B, arcLength b = π / m) :
  ∃ k ∈ {1, 2, ..., 2 * m}, l (A_j k ∩ B) ≥ (1 / (2 * π)) * l A * l B :=
by smathlib
endลาดลุน

end exists_k_l417_417952


namespace required_connections_l417_417748

theorem required_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 := by
  sorry

end required_connections_l417_417748


namespace sufficient_but_not_necessary_l417_417154

noncomputable def z1 (m : ℝ) : ℂ := complex.mk (m^2 + m + 1) (m^2 + m - 4)
def z2 : ℂ := complex.mk 3 (-2)

theorem sufficient_but_not_necessary (m : ℝ) : (z1 m = z2) ↔ (m = 1 ∨ m = -2) := by
  sorry

end sufficient_but_not_necessary_l417_417154


namespace farmer_apples_l417_417696

theorem farmer_apples : 127 - 39 = 88 := by
  -- Skipping proof details
  sorry

end farmer_apples_l417_417696


namespace units_digit_probability_l417_417418

def m_set := {2, 4, 6, 8}
def n_set := {i ∈ Set.range 2022 | i ≥ 2001}

noncomputable def favorable_outcomes : ℕ :=
  (Set.count {n ∈ n_set | 2^n % 10 = 4} m_set) +
  (Set.count {n ∈ n_set | 4^n % 10 = 4} m_set) +
  (Set.count {n ∈ n_set | 8^n % 10 = 4} m_set)

noncomputable def total_outcomes : ℕ :=
  Set.card m_set * Set.card n_set

noncomputable def probability_favorable : ℚ :=
  favorable_outcomes / total_outcomes

theorem units_digit_probability :
  probability_favorable = 13 / 42 :=
sorry

end units_digit_probability_l417_417418


namespace combination_sum_l417_417917

open Nat

theorem combination_sum (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m ≤ n) :
  (∑ i in range (k+1), Nat.choose k i * Nat.choose n (m - i)) = Nat.choose (n + k) m :=
by
  sorry

end combination_sum_l417_417917


namespace ivan_sergeyevich_profit_l417_417611

def revenue_from_meat (meat_sold price_per_kg : ℝ) : ℝ :=
  meat_sold * price_per_kg

def revenue_from_eggs (eggs_sold price_per_dozen : ℝ) : ℝ :=
  eggs_sold * (price_per_dozen / 12)

def total_revenue (meat_revenue egg_revenue : ℝ) : ℝ :=
  meat_revenue + egg_revenue

def profit (total_revenue expenses : ℝ) : ℝ :=
  total_revenue - expenses

-- Given conditions
def meat_sold := 100
def price_per_kg := 500
def eggs_sold := 20000
def price_per_dozen := 50
def expenses := 100000

theorem ivan_sergeyevich_profit : 
  profit (total_revenue (revenue_from_meat meat_sold price_per_kg) (revenue_from_eggs eggs_sold price_per_dozen)) expenses = 50000 :=
by sorry

end ivan_sergeyevich_profit_l417_417611


namespace find_remainder_l417_417046

def mod_condition : Prop :=
  (764251 % 31 = 5) ∧
  (1095223 % 31 = 6) ∧
  (1487719 % 31 = 1) ∧
  (263311 % 31 = 0) ∧
  (12097 % 31 = 25) ∧
  (16817 % 31 = 26) ∧
  (23431 % 31 = 0) ∧
  (305643 % 31 = 20)

theorem find_remainder (h : mod_condition) : 
  ((764251 * 1095223 * 1487719 + 263311) * (12097 * 16817 * 23431 - 305643)) % 31 = 8 := 
by
  sorry

end find_remainder_l417_417046


namespace each_monkey_gets_bananas_l417_417825

-- Define the conditions
def total_monkeys : ℕ := 12
def total_piles : ℕ := 10
def first_piles : ℕ := 6
def first_pile_hands : ℕ := 9
def first_hand_bananas : ℕ := 14
def remaining_piles : ℕ := total_piles - first_piles
def remaining_pile_hands : ℕ := 12
def remaining_hand_bananas : ℕ := 9

-- Define the number of bananas in each type of pile
def bananas_in_first_piles : ℕ := first_piles * first_pile_hands * first_hand_bananas
def bananas_in_remaining_piles : ℕ := remaining_piles * remaining_pile_hands * remaining_hand_bananas
def total_bananas : ℕ := bananas_in_first_piles + bananas_in_remaining_piles

-- Define the main theorem to be proved
theorem each_monkey_gets_bananas : total_bananas / total_monkeys = 99 := by
  sorry

end each_monkey_gets_bananas_l417_417825


namespace fleas_initial_minus_final_l417_417820

theorem fleas_initial_minus_final (F : ℕ) (h : F / 16 = 14) :
  F - 14 = 210 :=
sorry

end fleas_initial_minus_final_l417_417820


namespace production_days_l417_417794

theorem production_days (n : ℕ) 
    (h1 : 70 * n + 90 = 75 * (n + 1)) : n = 3 := 
sorry

end production_days_l417_417794


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417038

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417038


namespace number_of_members_l417_417838

noncomputable def club_members (n O N : ℕ) : Prop :=
  (3 * n = O - N) ∧ (O - N = 15)

theorem number_of_members (n O N : ℕ) (h : club_members n O N) : n = 5 :=
  by
    sorry

end number_of_members_l417_417838


namespace smallest_union_l417_417668

theorem smallest_union (A B : Finset α) [Fintype α] (hA : A.card = 30) (hB : B.card = 25) :
  (A ∪ B).card ≥ 30 :=
by
  have h : (A ∪ B).card = (A.card + B.card - (A ∩ B).card) := Finset.card_union_add_card_inter
  sorry

end smallest_union_l417_417668


namespace smallest_n_equal_triangles_l417_417710

theorem smallest_n_equal_triangles (n : ℕ) : ∃ O : Point, ∃ vertices : Fin n → Point, 
  convex_polygon vertices ∧ (∀ i j, vertices i ≠ vertices j) → divides_into_equal_triangles O vertices →
  n = 5 := by
sorry

end smallest_n_equal_triangles_l417_417710


namespace ellipse_focus_coordinates_l417_417455

noncomputable theory

def major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((0, 0), (8, 0))
def minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ) := ((4, 3), (4, -3))

theorem ellipse_focus_coordinates :
  let center := ((fst (fst major_axis_endpoints) + fst (snd major_axis_endpoints)) / 2, (snd (fst major_axis_endpoints) + snd (snd major_axis_endpoints)) / 2) in
  let a := (fst (snd major_axis_endpoints) - fst (fst major_axis_endpoints)) / 2 in
  let b := (snd (fst minor_axis_endpoints) - snd (snd minor_axis_endpoints)) / 2 in
  let c := Real.sqrt (a ^ 2 - b ^ 2) in
  (fst center + c, snd center) = (4 + Real.sqrt 7, 0) :=
by
  sorry

end ellipse_focus_coordinates_l417_417455


namespace pentagon_triangles_count_l417_417125

theorem pentagon_triangles_count (n : ℕ) (h_n : n = 5) : 
  ∃ T, T = 35 ∧ 
  all_diagonals_drawn (regular_polygon n) → count_triangles (regular_polygon n) T := 
by
  sorry

end pentagon_triangles_count_l417_417125


namespace gerald_remaining_pfennigs_l417_417507

-- Definitions of Gerald's initial money and the costs of items
def farthings : Nat := 54
def groats : Nat := 8
def florins : Nat := 17
def meat_pie_cost : Nat := 120
def sausage_roll_cost : Nat := 75

-- Conversion rates
def farthings_to_pfennigs (f : Nat) : Nat := f / 6
def groats_to_pfennigs (g : Nat) : Nat := g * 4
def florins_to_pfennigs (f : Nat) : Nat := f * 40

-- Total pfennigs Gerald has
def total_pfennigs : Nat :=
  farthings_to_pfennigs farthings + groats_to_pfennigs groats + florins_to_pfennigs florins

-- Total cost of both items
def total_cost : Nat := meat_pie_cost + sausage_roll_cost

-- Gerald's remaining pfennigs after purchase
def remaining_pfennigs : Nat := total_pfennigs - total_cost

theorem gerald_remaining_pfennigs :
  remaining_pfennigs = 526 :=
by
  sorry

end gerald_remaining_pfennigs_l417_417507


namespace geometric_sequence_iff_arithmetic_sequence_l417_417524

/-
  Suppose that {a_n} is an infinite geometric sequence with common ratio q, where q^2 ≠ 1.
  Also suppose that {b_n} is a sequence of positive natural numbers (ℕ).
  Prove that {a_{b_n}} forms a geometric sequence if and only if {b_n} forms an arithmetic sequence.
-/

theorem geometric_sequence_iff_arithmetic_sequence
  (a : ℕ → ℕ) (b : ℕ → ℕ) (q : ℝ)
  (h_geom_a : ∃ a1, ∀ n, a n = a1 * q ^ (n - 1))
  (h_q_squared_ne_one : q^2 ≠ 1)
  (h_bn_positive : ∀ n, 0 < b n) :
  (∃ a1, ∃ q', ∀ n, a (b n) = a1 * q' ^ n) ↔ (∃ d, ∀ n, b (n + 1) - b n = d) := 
sorry

end geometric_sequence_iff_arithmetic_sequence_l417_417524


namespace average_marks_l417_417686

theorem average_marks (D I T : ℕ) 
  (hD : D = 90)
  (hI : I = (3 * D) / 5)
  (hT : T = 2 * I) : 
  (D + I + T) / 3 = 84 :=
by
  sorry

end average_marks_l417_417686


namespace additional_cars_can_fit_l417_417585

-- Given definitions and conditions
variable (total_cars : Nat)
variable (levels : Nat)
variable (parked_cars : Nat)

-- Specific conditions for our problem
def total_cars := 425
def levels := 5
def parked_cars := 23

-- Goal statement
theorem additional_cars_can_fit : (total_cars / levels) - parked_cars = 62 := by
  sorry

end additional_cars_can_fit_l417_417585


namespace vasya_tolya_badges_l417_417768

theorem vasya_tolya_badges (x y : ℤ)
    (h1 : y = x + 5) -- Vasya initially had 5 more badges than Tolya
    (h2 : (y - (6 * (y / 25) / 25) + (4 * x) / 25) = (x - (4 * x) / 5 + 6 * (y / 25) / 5 - 1)) : -- equation balancing after exchange
    x = 45 ∧ y = 50 := 
sorry

end vasya_tolya_badges_l417_417768


namespace integral_problem_solution_l417_417119

noncomputable def integral_problem : ℝ :=
  ∫ x in -2..2, (Real.sin x + 2)

theorem integral_problem_solution :
  integral_problem = 8 :=
by 
  sorry

end integral_problem_solution_l417_417119


namespace find_AN_squared_l417_417245

noncomputable def semicircle_condition_1 (M N A B : ℝ) (MB BN AN : ℝ) : Prop :=
MB / BN = 2 / 3

noncomputable def semicircle_condition_2 : Prop :=
MN = 10

theorem find_AN_squared (M N A B : ℝ) (MB BN AN : ℝ) 
  (cond1 : semicircle_condition_1 M N A B MB BN AN)
  (cond2 : semicircle_condition_2) :
  AN^2 = 36 :=
sorry

end find_AN_squared_l417_417245


namespace contradiction_proof_l417_417363

theorem contradiction_proof (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  -- Negate the proposition:
  -- Assuming that neither a nor b is divisible by 5.
  have h1 : ¬ (5 ∣ a) := sorry
  have h2 : ¬ (5 ∣ b) := sorry
  -- We need to achieve a contradiction here. 
  -- Proof will proceed by contradiction, showing that h1 and h2 can't both hold true given h.
  contradiction
  

end contradiction_proof_l417_417363


namespace correct_system_l417_417989

variable (x y : ℝ)

def equation1 := y - x = 4.5
def equation2 := x - (1/2) * y = 1

theorem correct_system :
  (equation1 x y) ∧ (equation2 x y) ↔
  (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
begin
  sorry
end

end correct_system_l417_417989


namespace parameterization_of_line_l417_417319

theorem parameterization_of_line (t : ℝ) :
  let x := 10 * t + 13 in
  let y := 20 * t - 14 in
  y = 2 * x - 40 := by
sorry

end parameterization_of_line_l417_417319


namespace masha_ate_10_pies_l417_417283

-- Define the conditions as Lean definitions
def total_pies : ℕ := 40
def bear_speed_multiple : ℕ := 3
def total_raspberries : ℕ := R (assume R is positive integer)
def half_raspberries : ℕ := total_raspberries / 2

-- Define the rates
def masha_rate : ℕ := r
def bear_rate : ℕ := bear_speed_multiple * masha_rate

-- Define the time variables for eating raspberries and pies
def masha_time_raspberries : ℕ := t1
def bear_time_raspberries : ℕ := masha_time_raspberries / bear_speed_multiple
def masha_time_pies : ℕ := t2
def bear_time_pies : ℕ := t2

-- Define the amount of pies eaten
def masha_pies := masha_rate * masha_time_pies
def bear_pies := bear_rate * bear_time_pies

-- Lean Theorem: Prove that Masha ate 10 pies
theorem masha_ate_10_pies (t1 t2 r : ℕ) (h : total_pies = masha_pies + bear_pies) (bear_rate_eq : bear_rate = bear_speed_multiple * masha_rate) : 
  masha_pies = 10 :=
by
  sorry

end masha_ate_10_pies_l417_417283


namespace average_marks_l417_417683

-- Definitions
def Tatuya_score (Ivanna_score : ℕ) : ℕ := 2 * Ivanna_score
def Ivanna_score (Dorothy_score : ℕ) : ℕ := (3 * Dorothy_score) / 5
def Dorothy_score : ℕ := 90

-- Theorem statement
theorem average_marks :
  let Dorothy_score := Dorothy_score in
  let Ivanna_score := Ivanna_score Dorothy_score in
  let Tatuya_score := Tatuya_score Ivanna_score in
  (Dorothy_score + Ivanna_score + Tatuya_score) / 3 = 84 :=
by 
  -- Proof goes here
  sorry

end average_marks_l417_417683


namespace predicted_weight_approx_l417_417149

/-- Given the regression equation y = 0.849x - 85.712 and a student height of 172 cm,
    the predicted weight is approximately 60.316 kg. -/
theorem predicted_weight_approx (h_eq : ∀ (x : Real), 0.849 * x - 85.712) (height_172 : 172) : 
  (h_eq 172) ≈ 60.316 := 
sorry

end predicted_weight_approx_l417_417149


namespace second_term_binomial_expansion_l417_417718

theorem second_term_binomial_expansion :
  let expr := (x - 2 / x) ^ 6 in
  (binomial_expansion expr).second_term = -12 * x ^ 4 := 
sorry

end second_term_binomial_expansion_l417_417718


namespace area_triangle_3_6_l417_417796

/-
Problem: Prove that the area of a triangle with base 3 meters and height 6 meters is 9 square meters.
Definitions: 
- base: The base of the triangle is 3 meters.
- height: The height of the triangle is 6 meters.
Conditions: 
- The area of a triangle formula.
Correct Answer: 9 square meters.
-/

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

theorem area_triangle_3_6 : area_of_triangle 3 6 = 9 := by
  sorry

end area_triangle_3_6_l417_417796


namespace sum_of_roots_eq_3_l417_417326

-- Define the polynomial function
def polynomial (x : ℝ) : ℝ := 3 * x ^ 3 - 9 * x ^ 2 - 72 * x - 18

-- The theorem that states the sum of the roots is 3
theorem sum_of_roots_eq_3 : (∑ x in (multiset.roots polynomial), x) = 3 :=
by
    sorry

end sum_of_roots_eq_3_l417_417326


namespace sufficient_but_not_necessary_mul_pos_of_pos_of_pos_l417_417915

theorem sufficient_but_not_necessary (a b : ℝ) (h₁ : a > 2) (h₂ : b > 2) : (a * b > 4) ∧ ¬(∀ a b, (a * b > 4) → (a > 2 ∧ b > 2)) :=
by {
  split,
  { -- Prove sufficient condition a > 2 ∧ b > 2 -> a * b > 4
    apply mul_pos_of_pos_of_pos; assumption,
  },
  { -- Provide counterexample to show it's not a necessary condition
    intro h,
    specialize h 2 3,
    have h_counterexample : ¬ (2 > 2 ∧ 3 > 2),
    { simp, },
    exact h_counterexample (h (by linarith)),
  }
}

-- This adds sorry to indicate missing proofs in other parts
theorem mul_pos_of_pos_of_pos (a b : ℝ) (ha : a > 0) (hb : b > 0) : a * b > 0 :=
sorry

end sufficient_but_not_necessary_mul_pos_of_pos_of_pos_l417_417915


namespace sequence_term_4th_l417_417551

theorem sequence_term_4th (a_n : ℕ → ℝ) (h : ∀ n, a_n n = 2 / (n^2 + n)) :
  ∃ n, a_n n = 1 / 10 ∧ n = 4 :=
by
  sorry

end sequence_term_4th_l417_417551


namespace proof_of_polar_coordinates_M_proof_of_cartesian_equation_of_circle_proof_of_sum_of_reciprocals_of_distances_l417_417223

noncomputable def polar_coordinates_of_M (t : ℝ) : Prop :=
  let x := (Real.sqrt 2) / 2 * t
  let y := 2 + (Real.sqrt 2) / 2 * t
  t = -Real.sqrt 2 → x = -1 ∧ y = 1 ∧
  (Real.sqrt (x^2 + y^2) = Real.sqrt 2) ∧ 
  (Real.arctan (y / x) = 3 * Real.pi / 4)

noncomputable def cartesian_equation_of_circle (ρ : ℝ) : Prop :=
  (ρ = 4) → (ρ^2 = 16) ∧ (∀ (x y: ℝ), x^2 + y^2 = 16)

noncomputable def sum_of_reciprocals_of_distances (P A B : ℝ × ℝ) : Prop :=
  let l_x t := (Real.sqrt 2) / 2 * t
  let l_y t := 2 + (Real.sqrt 2) / 2 * t
  ∃ t₁ t₂ : ℝ, 
    (l_x t₁)^2 + (l_y t₁)^2 = 16 ∧ 
    (l_x t₂)^2 + (l_y t₂)^2 = 16 ∧ 
    P = (0, 2) → 
    t₁ + t₂ = -2 * Real.sqrt 2 ∧ t₁ * t₂ = -12 ∧
    (1 / (Real.abs t₁) + 1 / (Real.abs t₂) = Real.sqrt 14 / 6)

theorem proof_of_polar_coordinates_M :
  polar_coordinates_of_M (-Real.sqrt 2) :=
sorry

theorem proof_of_cartesian_equation_of_circle :
  cartesian_equation_of_circle 4 :=
sorry

theorem proof_of_sum_of_reciprocals_of_distances :
  sum_of_reciprocals_of_distances (0, 2) (0, 2) (0, 2) :=
sorry

end proof_of_polar_coordinates_M_proof_of_cartesian_equation_of_circle_proof_of_sum_of_reciprocals_of_distances_l417_417223


namespace sequence_properties_l417_417857

-- Given definitions
def is_sequence_positive (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → 0 < a n

def sum_of_first_n (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), 0 < n → (S n) = (finset.range n).sum (λ k, a (k + 1))

def satisfies_equation (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → (S n) ^ 2 - (n ^ 2 + n - 3) * S n - 3 * (n ^ 2 + n) = 0

-- The Lean statement we need:
theorem sequence_properties 
  (a S : ℕ → ℕ)
  (hpos : is_sequence_positive a)
  (hsum : sum_of_first_n a S)
  (heqn : satisfies_equation S) :
  a 1 = 2 ∧ (∀ n : ℕ, 0 < n → a n = 2 * n) ∧ 
  ∀ n : ℕ, 0 < n → (finset.range n).sum (λ k, 1 / ((a (k + 1)) * (a (k + 1) + 1))) < 1 / 3 := 
begin
  sorry
end

end sequence_properties_l417_417857


namespace sin_150_equals_half_l417_417854

theorem sin_150_equals_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by 
  sorry

end sin_150_equals_half_l417_417854


namespace convex_quad_inscribed_l417_417817

theorem convex_quad_inscribed (A B C D O : Point) (h : IsInscribed ABCD O) (hR : radius O = 1)
    : 0 < |dist A B + dist B C + dist C D + dist D A - dist A C - dist B D| ∧ 
      |dist A B + dist B C + dist C D + dist D A - dist A C - dist B D| < 2 := 
sorry

end convex_quad_inscribed_l417_417817


namespace perpendicular_vectors_l417_417954

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (1, 0)

-- Definition of dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem perpendicular_vectors
  (λ : ℝ)
  (h : dot_product (a.1 - b.1, a.2 - b.2)
                   (2 * a.1 + λ * b.1, 2 * a.2 + λ * b.2) = 0) :
  λ = 3 :=
sorry

end perpendicular_vectors_l417_417954


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417033

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417033


namespace hyperbola_focal_length_l417_417176

-- Define the given conditions
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- Prove that the focal length is 4 given the conditions
theorem hyperbola_focal_length (m : ℝ) (hm : m > 0) :
  (∀ x y, hyperbola_eq x y m) →
  (∀ x y, asymptote_eq x y m) →
  (2 * sqrt (m + 1) = 4) :=
by
  intros; sorry

end hyperbola_focal_length_l417_417176


namespace sum_of_coefficients_of_y_terms_l417_417880

theorem sum_of_coefficients_of_y_terms :
  let expr := (4 * x + 3 * y + 2) * (2 * x + 5 * y + 3) in
  let terms_with_y := [26 * x * y, 15 * y^2, 19 * y] in
  (terms_with_y.sum (λ term, term.coeff)) = 60 :=
by
  sorry

end sum_of_coefficients_of_y_terms_l417_417880


namespace median_of_dataset_l417_417164

theorem median_of_dataset
  (x : ℝ)
  (h_avg : (6 + 2 + 8 + x + 7) / 5 = 6) :
  let dataset := [6, 2, 8, x, 7] in 
  let sorted := dataset.sorted in 
  sorted.nth_le 2 sorry = 7 :=
by
  sorry

end median_of_dataset_l417_417164


namespace arrange_colored_houses_l417_417981

-- Define the problem conditions
def pass_before (house1 house2 : Nat) : Prop := house1 < house2
def not_adjacent (house1 house2 : Nat) : Prop := abs (house1 - house2) ≠ 1

-- Define the houses' indices
inductive House : Type
| green : House
| purple : House
| pink : House
| black : House
| white : House

-- State the theorem
theorem arrange_colored_houses :
    ∃ (h : House → Nat), 
        pass_before (h House.green) (h House.purple) ∧
        pass_before (h House.black) (h House.white) ∧
        not_adjacent (h House.green) (h House.pink) ∧
        (∃ l : List House, l.length = 5 ∧ l.nodup ∧ ∀ x ∈ l, True) ∧ -- List contains all houses without duplication
        (l.sort Nat.lt = [h House.green, h House.purple, h House.pink, h House.black, h House.white] →
            ∀ l, l.nodup) →
    True ∧
    6 = 6 :=
by
  sorry

end arrange_colored_houses_l417_417981


namespace exists_k_between_S2023_S2024_l417_417634

noncomputable def S (n : ℕ) : ℕ := ∑ i in (Finset.range n).map Nat.primeList, id

theorem exists_k_between_S2023_S2024 :
  ∃ k : ℕ, S 2023 < k^2 ∧ k^2 < S 2024 := by
  sorry

end exists_k_between_S2023_S2024_l417_417634


namespace a_values_contained_in_set_l417_417197

-- Definitions of sets A and B
def A := {-2, 1}
def B (a : ℝ) := {x : ℝ | a * x = 2}

-- Proof problem statement
theorem a_values_contained_in_set :
  {a : ℝ | B(a) ⊆ A} = {-1, 0, 2} :=
sorry

end a_values_contained_in_set_l417_417197


namespace probability_sum_equals_40_l417_417461

theorem probability_sum_equals_40 : 
  let primes := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37},
      pairs := {(3, 37), (11, 29), (17, 23)} in
  primes.card = 12 ∧ pairs.card = 3 →
  (finset.card (finset.powerset_len 2 primes.to_finset).filter (λ p, p.sum = 40)) = 3 →
  (3 / (finset.card (finset.powerset_len 2 primes.to_finset)) : ℝ) = 1 / 22 :=
by
  intros,
  sorry

end probability_sum_equals_40_l417_417461


namespace consequence_of_implication_l417_417394

-- Define the conditions
variable (A B : Prop)

-- State the theorem to prove
theorem consequence_of_implication (h : B → A) : A → B := 
  sorry

end consequence_of_implication_l417_417394


namespace arithmetic_sequence_third_term_l417_417698

theorem arithmetic_sequence_third_term :
  ∀ (a d : ℤ), (a + 4 * d = 2) ∧ (a + 5 * d = 5) → (a + 2 * d = -4) :=
by sorry

end arithmetic_sequence_third_term_l417_417698


namespace correct_option_is_b_l417_417805

-- Definitions
def options : List (String × String) :=
  [("A", ("a", "不填")), ("B", ("a", "the")), ("C", ("the", "a")), ("D", ("不填", "a"))]

def sentence : String :=
  "Gottlieb Daimler, \_\_\_\_\_\_ German engineer, is normally believed to have invented \_\_\_\_\_\_ car."

-- Theorem statement
theorem correct_option_is_b : ∃ opt, opt ∈ options ∧ opt = ("B", ("a", "the")) :=
by
  use ("B", ("a", "the"))
  split
  · simp [options]
  · refl

end correct_option_is_b_l417_417805


namespace division_dividend_l417_417583

/-- In a division sum, the quotient is 40, the divisor is 72, and the remainder is 64. We need to prove that the dividend is 2944. -/
theorem division_dividend : 
  let Q := 40
  let D := 72
  let R := 64
  (D * Q + R = 2944) :=
by
  sorry

end division_dividend_l417_417583


namespace relationship_y1_y2_y3_l417_417963

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = c) →
  (y2 = -3 + c) →
  (y3 = 5 + c) →
  y2 < y1 ∧ y1 < y3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  split
  · linarith
  · linarith

end relationship_y1_y2_y3_l417_417963


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417016

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417016


namespace repeating_decimal_identity_l417_417049

theorem repeating_decimal_identity (abc : ℕ) (h1 : 78 * (1 + abc / 999) = 78 * (1 + abc * 0.001) + 0.75) (h2 : 0.001 * abc = abc / 999):
  abc = 960 := by
  sorry

end repeating_decimal_identity_l417_417049


namespace original_number_of_candies_l417_417438

theorem original_number_of_candies (x : ℝ) (h₀ : x * (0.7 ^ 3) = 40) : x = 117 :=
by 
  sorry

end original_number_of_candies_l417_417438


namespace linear_function_passing_through_points_A_and_B_l417_417415

theorem linear_function_passing_through_points_A_and_B :
  ∃ k b : ℝ, (∀ x y : ℝ, (y = k * x + b → ((x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 5)))) ∧ ∀ x : ℝ, y = k * x + b :=
begin
  have eq1 : 3 * k + b = 4, from sorry,
  have eq2 : 4 * k + b = 5, from sorry,
  have k_eq : k = 1, by sorry,
  have b_eq : b = 1, by sorry,
  use [k_eq, b_eq],
  sorry -- rest of the proof
end

end linear_function_passing_through_points_A_and_B_l417_417415


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417035

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417035


namespace minimum_shift_is_pi_over_3_l417_417575

noncomputable def minimum_positive_shift 
  (m : ℝ) : ℝ :=
  if h : ∃ k : ℤ, -2 * m + π / 6 = π / 2 + k * π 
  then if m > 0 
    then m 
    else 0 
  else 0

theorem minimum_shift_is_pi_over_3 
  (m : ℝ) 
  (cond : ∃ k : ℤ, -2 * m + π / 6 = π / 2 + k * π) 
  (m_pos : m > 0) : 
  minimum_positive_shift m = π / 3 :=
sorry

end minimum_shift_is_pi_over_3_l417_417575


namespace max_min_ratio_max_min_diff_max_min_square_l417_417926

-- Given the relation x^2 + y^2 - 4x + 1 = 0

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 1 = 0

theorem max_min_ratio (x y : ℝ) (h : circle_eq x y) :
  ∃ k_max k_min : ℝ, k_max = sqrt 3 ∧ k_min = -sqrt 3 ∧
                     k_max = max (y / x) ∧ k_min = min (y / x) :=
begin
  sorry
end

theorem max_min_diff (x y : ℝ) (h : circle_eq x y) :
  ∃ t_max t_min : ℝ, t_max = 2 + sqrt 6 ∧ t_min = -2 - sqrt 6 ∧
                     t_max = max (y - x) ∧ t_min = min (y - x) :=
begin
  sorry
end

theorem max_min_square (x y : ℝ) (h : circle_eq x y) :
  ∃ s_max s_min : ℝ, s_max = 7 + 4 * sqrt 3 ∧ s_min = 7 - 4 * sqrt 3 ∧
                     s_max = max (x^2 + y^2) ∧ s_min = min (x^2 + y^2) :=
begin
  sorry
end

end max_min_ratio_max_min_diff_max_min_square_l417_417926


namespace construct_segments_l417_417097

-- Define a triangle as a structure
structure Triangle (α : Type*) :=
  (A B C : α)

variable {α : Type*} [EuclideanGeometry α]

-- Define conditions and conclusions as Lean statements
theorem construct_segments (T : Triangle α) (base_par : T.A ≠ T.B) :
  ∃ X Y X₁ Y₁ : α, parallel (line[XY]) (line[AB])
                   ∧ XY = 2 * AX
                   ∧ XY = 2 * BY
                   ∧ parallel (line[X₁Y₁]) (line[AB])
                   ∧ X₁Y₁ = 2 * AX₁
                   ∧ X₁Y₁ = 2 * BY₁ :=
  sorry

end construct_segments_l417_417097


namespace volume_is_correct_l417_417496

def condition1 (x y z : ℝ) : Prop := abs (x + 2 * y + 3 * z) + abs (x + 2 * y - 3 * z) ≤ 18
def condition2 (x y z : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0
def region (x y z : ℝ) : Prop := condition1 x y z ∧ condition2 x y z

noncomputable def volume_of_region : ℝ :=
  60.75 -- the result obtained from the calculation steps

theorem volume_is_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 60.75 :=
by
  sorry

end volume_is_correct_l417_417496


namespace parallel_lines_a_l417_417555

theorem parallel_lines_a (a : ℝ) (x y : ℝ)
  (h1 : x + 2 * a * y - 1 = 0)
  (h2 : (a + 1) * x - a * y = 0)
  (h_parallel : ∀ (l1 l2 : ℝ → ℝ → Prop), l1 x y ∧ l2 x y → l1 = l2) :
  a = -3 / 2 ∨ a = 0 :=
sorry

end parallel_lines_a_l417_417555


namespace coin_stack_correct_l417_417235

-- Setup a noncomputable definition and the overall theorem statement.
noncomputable def coinStackSolution : Prop :=
  ∃ (p n d q : ℕ), 
    (1.65 * p + 1.85 * n + 1.25 * d + 1.80 * q = 15) ∧ 
    (p + 5 * n + 10 * d + 25 * q = 85) ∧ 
    (p = 4) ∧ (n = 1) ∧ (d = 1) ∧ (q = 3)

theorem coin_stack_correct : coinStackSolution :=
by
  unfold coinStackSolution
  existsi (4, 1, 1, 3)
  -- Add the necessary equations and their proofs (left as sorry for simplicity)
  simp
  sorry -- Equations proof placeholder

end coin_stack_correct_l417_417235


namespace problem_l417_417863

noncomputable def f : ℝ → ℝ := sorry
axiom h_even : ∀ x : ℝ, f (-x) = f x
axiom h_deriv_neg : ∀ x : ℝ, x ≥ 0 → f' x < 0

theorem problem : f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end problem_l417_417863


namespace no_perfect_square_solution_l417_417641

theorem no_perfect_square_solution (n : ℕ) (x : ℕ) (hx : x < 10^n) :
  ¬ (∀ y, 0 ≤ y ∧ y ≤ 9 → ∃ z : ℤ, ∃ k : ℤ, 10^(n+1) * z + 10 * x + y = k^2) :=
sorry

end no_perfect_square_solution_l417_417641


namespace units_digit_3968_805_l417_417472

theorem units_digit_3968_805 : 
  (3968 ^ 805) % 10 = 8 := 
by
  -- Proof goes here
  sorry

end units_digit_3968_805_l417_417472


namespace problem_conditions_triangle_condition_l417_417379

variables (A B C D M P Q N : Type) [LinearOrderedField A] [LinearOrderedField B]
  [LinearOrderedField C] [LinearOrderedField D]
  [OrderedAddCommGroup M] [LinearOrderedField M]
  [Auto : LinearOrderedField N]

-- Defining the midpoint, parallel lines and the intersection
noncomputable def midpoint (P Q : A) : A = sorry

def parallel (L1 L2 : A) : Prop := sorry

noncomputable def intersection (L1 L2 : A) : A = sorry

-- Defining the coordinates and variables
variables (a : A) (M := (midpoint (C) (D))) (P := (midpoint (M) (A))) (Q := (midpoint (M) (B)))
  (N := (intersection (D, P) (C, Q)))

theorem problem_conditions : 
  (ABCD : Type) (BC_parallel_AD : parallel BC AD) (MidM : M = midpoint C D) 
  (MidP : P = midpoint M A) (MidQ : Q = midpoint M B) 
  (IntersectionN : N = intersection (D, P) (C, Q)) :
  \(\pass_if \frac{1}{3} \leq \frac{|AD|}{|BC|} \leq 3\)

-- Mathematical condition equivalent to the problem statement.
theorem triangle_condition :
  \all (A B M N : Type) \exists (yN : A), yN ≤ 2  ⋀ -yN ≤ 2 - a^2  ⋀ -yN \geq -(2 - a^2) ⟹ 
  let (D A) := A a in ((xN = (2 - a^2)) ⋀ yN = -a) 
  iff 
  (1 / 3 ≤ (Distance_m' AD BC) ≤ 3) :=
begin
  sorry
end

end problem_conditions_triangle_condition_l417_417379


namespace relationship_among_a_b_c_l417_417565

noncomputable def a : ℝ := (1 / 2) ^ 10
noncomputable def b : ℝ := (1 / 5) ^ (-1 / 2)
noncomputable def c : ℝ := log (1 / 5) 10

theorem relationship_among_a_b_c :
  b > a ∧ a > c := by
  sorry

end relationship_among_a_b_c_l417_417565


namespace find_value_l417_417957

-- Defining the known conditions
def number : ℕ := 20
def half (n : ℕ) : ℕ := n / 2
def value_added (V : ℕ) : Prop := half number + V = 17

-- Proving that the value added to half the number is 7
theorem find_value : value_added 7 :=
by
  -- providing the proof for the theorem
  -- skipping the proof steps with sorry
  sorry

end find_value_l417_417957


namespace badges_exchange_l417_417757

theorem badges_exchange (Vasya_initial Tolya_initial : ℕ) 
    (h1 : Vasya_initial = Tolya_initial + 5)
    (h2 : Vasya_initial - 0.24 * Vasya_initial + 0.20 * Tolya_initial = Tolya_initial - 0.20 * Tolya_initial + 0.24 * Vasya_initial - 1) 
    : Vasya_initial = 50 ∧ Tolya_initial = 45 :=
by sorry

end badges_exchange_l417_417757


namespace max_tan_theta_of_parabola_l417_417158

theorem max_tan_theta_of_parabola (p : ℝ) (h : p > 0) 
  (A B : ℝ × ℝ) (O : ℝ × ℝ := (0, 0))
  (theta : ℝ) (h_theta : theta ≠ π / 2)
  (S : ℝ := m * tan theta)
  (m : ℝ := -p^2 / 2) :
  ∃ (tan_theta_maximum : ℝ), tan_theta_maximum = -2 * real.sqrt 2 := 
by {
  sorry
}

end max_tan_theta_of_parabola_l417_417158


namespace probability_of_at_least_one_diamond_or_joker_l417_417232

theorem probability_of_at_least_one_diamond_or_joker (total_cards diamonds jokers : ℕ) :
  total_cards = 60 →
  diamonds = 15 →
  jokers = 6 →
  let favorable := diamonds + jokers in
  let prob_not_favorable := (total_cards - favorable) / total_cards.to_rat in
  let prob_none_favorable := prob_not_favorable * prob_not_favorable in
  1 - prob_none_favorable = 231 / 400 :=
begin
  intros h_total_cards h_diamonds h_jokers favorable prob_not_favorable prob_none_favorable,
  sorry
end

end probability_of_at_least_one_diamond_or_joker_l417_417232


namespace triangle_area_l417_417485

theorem triangle_area : 
  let line1 := λ x, 2 * x + 3,
      line2 := λ x, (9 - x) / 2,
      y_intercept_line1 := line1 0,
      y_intercept_line2 := line2 0,
      intersection_x := 1,
      intersection_y := line1 intersection_x,
      base := y_intercept_line2 - y_intercept_line1,
      height := intersection_x in
  y_intercept_line1 = 3 ∧ 
  y_intercept_line2 = 9 / 2 ∧ 
  intersection_x = 1 ∧ 
  intersection_y = 5 ∧ 
  base = 3 / 2 ∧ 
  height = 1 →
  (1 / 2) * base * height = 3 / 4 :=
by
  intros
  sorry

end triangle_area_l417_417485


namespace find_x2000_l417_417520

-- Define the sequence and its properties
variable {α : Type} [AddGroup α]

-- Given conditions
def condition1 (x : ℕ → α) : Prop :=
∀ n, x n + x (n + 1) + x (n + 2) = 20

def condition2 (x : ℕ → α) : Prop :=
x 4 = 9

def condition3 (x : ℕ → α) : Prop :=
x 12 = 7

-- Final Lean statement to prove that x 2000 = 4
theorem find_x2000 (x : ℕ → α) (h1 : condition1 x) (h2 : condition2 x) (h3 : condition3 x) : x 2000 = 4 :=
by
  sorry

end find_x2000_l417_417520


namespace h_domain_l417_417866

def h (x : ℝ) : ℝ := (2 * x^3 - 5 * x + 4) / ((x - 2)^2 + abs (x + 2))

theorem h_domain : ∀ x : ℝ, (x - 2)^2 + abs (x + 2) ≠ 0 :=
by 
  intro x
  sorry

end h_domain_l417_417866


namespace four_consecutive_product_divisible_by_12_l417_417001

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l417_417001


namespace min_colors_for_annulus_l417_417699

theorem min_colors_for_annulus (n : ℕ) (H : n = 14) : 
  ∃ (k : ℕ), k = 4 ∧ (∀ coloring : fin n → fin k, ∀ (i j : fin n), adjacent i j → coloring i ≠ coloring j)  := 
sorry

end min_colors_for_annulus_l417_417699


namespace area_tris_double_l417_417802

variables (A B C P D E F : EuclideanGeometry.Point ℝ) 
variables (PA PB PC BC CA AB : EuclideanGeometry.Line ℝ)
variables [hABC : EquilateralTriangle A B C] [hCircum : OnCircumcircle P A B C]

def intersects (l1 l2 : EuclideanGeometry.Line ℝ) (P : EuclideanGeometry.Point ℝ) : Prop := 
  l1.contains P ∧ l2.contains P

axiom intersects_PA_BC : intersects PA BC D
axiom intersects_PB_CA : intersects PB CA E
axiom intersects_PC_AB : intersects PC AB F

theorem area_tris_double : 
  EuclideanGeometry.Area (EuclideanGeometry.Triangle D E F) = 
  2 * EuclideanGeometry.Area (EuclideanGeometry.Triangle A B C) :=
begin
  sorry
end

end area_tris_double_l417_417802


namespace positive_solution_bounds_l417_417247

-- Define the quadratic function and the conditions
def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

variables {a b c m : ℝ} (x : ℝ)
-- Assume a ≠ 0 and the condition for m
def nonzero_a := a ≠ 0
def m_condition := 1 < m ∧ m < 1.5

-- Values from the table
def y_values (x : ℝ) : Prop :=
  (x = -2 → quadratic_function a b c x = m - 4.5) ∧
  (x = -1 → quadratic_function a b c x = m - 2) ∧
  (x = 0 → quadratic_function a b c x = m - 0.5) ∧
  (x = 1 → quadratic_function a b c x = m) ∧
  (x = 2 → quadratic_function a b c x = m - 0.5) ∧
  (x = 3 → quadratic_function a b c x = m - 2) ∧
  (x = 4 → quadratic_function a b c x = m - 4.5)

-- The proof statement
theorem positive_solution_bounds (a_ne_zero : nonzero_a) (m_cond : m_condition) (y_vals : y_values x) :
  ∃ x₁ : ℝ, 2 < x₁ ∧ x₁ < 3 ∧ quadratic_function a b c x₁ = 0 :=
sorry

end positive_solution_bounds_l417_417247


namespace acute_triangle_of_parallelepiped_section_l417_417294

theorem acute_triangle_of_parallelepiped_section (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (∀ A B C : ℝ, (A, B, C) = (x, y, z) → 
  (∀ θ : ℝ, (0 < θ) ∧ (θ < π/2) ⟹ 
  (triangle_formed_by_section_is_acute (A, B, C)))) :=
sorry

end acute_triangle_of_parallelepiped_section_l417_417294


namespace pears_more_than_apples_l417_417331

theorem pears_more_than_apples (red_apples green_apples pears : ℕ) (h1 : red_apples = 15) (h2 : green_apples = 8) (h3 : pears = 32) : (pears - (red_apples + green_apples) = 9) :=
by
  sorry

end pears_more_than_apples_l417_417331


namespace sum_of_factors_of_1000_l417_417712

-- Define what it means for an integer to not contain the digit '0'
def no_zero_digits (n : ℕ) : Prop :=
∀ c ∈ (n.digits 10), c ≠ 0

-- Define the problem statement
theorem sum_of_factors_of_1000 :
  ∃ (a b : ℕ), a * b = 1000 ∧ no_zero_digits a ∧ no_zero_digits b ∧ (a + b = 133) :=
sorry

end sum_of_factors_of_1000_l417_417712


namespace nero_speed_l417_417605

theorem nero_speed (jerome_speed : ℝ) (jerome_time : ℝ) (nero_time : ℝ) :
  jerome_speed = 4 → jerome_time = 6 → nero_time = 3 → 
  ∃ nero_speed : ℝ, nero_speed = 8 :=
by
  intros h1 h2 h3
  use jerome_speed * jerome_time / nero_time
  rw [h1, h2, h3]
  norm_num
  sorry

end nero_speed_l417_417605


namespace angle_Oa_B_Oc_eq_angle_A_I_C_l417_417060

-- Definitions and setup as provided by the problem statement
variables (A B C A₁ C₁ I A₂ C₂ Oₐ O₀ : Type)
variables [inner_product_space ℝ Type]

/-- Defining the given properties -/
def angle_bisector (X Y Z : Type) : Prop := sorry -- Placeholder definition for angle bisector
def parallel (X Y Z W : Type) : Prop := sorry -- Placeholder definition for parallel lines
def circumcenter (X Y Z : Type) : Type := sorry -- Placeholder definition for circumcenter

-- Hypotheses based on given conditions
variables (h1 : angle_bisector A A₁ C)
variables (h2 : angle_bisector C C₁ A)
variables (h3 : parallel A C A₂ B)
variables (h4 : parallel A C C₂ B)
variables (h5 : circumcenter A C₁ C₂ = Oₐ)
variables (h6 : circumcenter C A₁ A₂ = O₀)

-- Angle relationships
def angle (X Y Z: Type) : ℝ := sorry -- Placeholder definition for angle function

theorem angle_Oa_B_Oc_eq_angle_A_I_C :
  angle Oₐ B O₀ = angle A I C :=
sorry

end angle_Oa_B_Oc_eq_angle_A_I_C_l417_417060


namespace calculate_rectangle_length_l417_417285

theorem calculate_rectangle_length (side_of_square : ℝ) (width_of_rectangle : ℝ)
  (length_of_wire : ℝ) (perimeter_of_rectangle : ℝ) :
  side_of_square = 20 → 
  width_of_rectangle = 14 → 
  length_of_wire = 4 * side_of_square →
  perimeter_of_rectangle = length_of_wire →
  2 * (width_of_rectangle + length_of_rectangle) = perimeter_of_rectangle →
  length_of_rectangle = 26 :=
by
  intros
  sorry

end calculate_rectangle_length_l417_417285


namespace min_rice_pounds_l417_417257

variable {o r : ℝ}

theorem min_rice_pounds (h1 : o ≥ 8 + r / 3) (h2 : o ≤ 2 * r) : r ≥ 5 :=
sorry

end min_rice_pounds_l417_417257


namespace hyperbola_focal_length_l417_417177

theorem hyperbola_focal_length (m : ℝ) (h_pos : 0 < m)
  (h_asymptote : ∀ x y : ℝ, (√3 * x + m * y = 0) ↔ (x = 0 ∧ y = 0)) :
  2 * sqrt (3 + 1) = 4 := by
  sorry

end hyperbola_focal_length_l417_417177


namespace probability_at_least_one_2_on_8_sided_dice_l417_417349

theorem probability_at_least_one_2_on_8_sided_dice :
  (∃ (d1 d2 : Fin 8), d1 = 1 ∨ d2 = 1) → (15 / 64) = (15 / 64) := by
  intro h
  sorry

end probability_at_least_one_2_on_8_sided_dice_l417_417349


namespace determine_f_l417_417546

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom f_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem determine_f (x : ℝ) : f x = x + 1 := by
  sorry

end determine_f_l417_417546


namespace equation_has_exactly_one_solution_l417_417503

theorem equation_has_exactly_one_solution (m : ℝ) : 
  (m ∈ { -1 } ∪ Set.Ioo (-1/2 : ℝ) (1/0) ) ↔ ∃ (x : ℝ), 2 * Real.sqrt (1 - m * (x + 2)) = x + 4 :=
sorry

end equation_has_exactly_one_solution_l417_417503


namespace wifi_cost_per_hour_l417_417456

-- Define the conditions as hypotheses
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def hourly_income : ℝ := 12
def trip_duration : ℝ := 3
def total_expenses : ℝ := ticket_cost + snacks_cost + headphones_cost
def total_earnings : ℝ := hourly_income * trip_duration

-- Translate the proof problem to Lean 4 statement
theorem wifi_cost_per_hour: 
  (total_earnings - total_expenses) / trip_duration = 2 :=
by sorry

end wifi_cost_per_hour_l417_417456


namespace f_2017_eq_11_div_24_l417_417509

def f (x : ℝ) : ℝ :=
  if x > 0 then f (x - 5)
  else 2^x + ∫ t in 0..(π / 6), cos (3 * t)

theorem f_2017_eq_11_div_24 : f 2017 = 11 / 24 := by
  sorry

end f_2017_eq_11_div_24_l417_417509


namespace part_II_part_III_l417_417525

-- You might define the sequence {a_n} as a function from ℕ to ℝ
def seq (a : ℕ → ℝ) := 
(∀ n : ℕ, n ≤ 4 → a n / n ≤ a 4 / 4) ∧
(∀ n : ℕ, n > 4 → a n = max (finset.image (λ i, a i + a (n - i)) (finset.range n)))

-- Part (Ⅱ)
theorem part_II (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n > 0 → a n / n ≤ a 4 / 4 := sorry

-- Part (Ⅲ)
theorem part_III (a : ℕ → ℝ) (h : seq a) : ∃ N : ℕ, ∀ n : ℕ, n > N → a n = a 4 + a (n - 4) := sorry

end part_II_part_III_l417_417525


namespace divisibility_by_11_l417_417296

theorem divisibility_by_11 (N : ℕ) (a : ℕ → ℕ) (n : ℕ)
  (hN : N = ∑ i in finset.range (n + 1), a i * 10^i) :
  (11 ∣ N) ↔ (11 ∣ ∑ i in finset.range (n + 1), (-1)^i * a i) :=
  sorry

end divisibility_by_11_l417_417296


namespace prob_at_least_one_2_in_two_8_sided_dice_l417_417360

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end prob_at_least_one_2_in_two_8_sided_dice_l417_417360


namespace range_of_a_l417_417804

noncomputable def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

noncomputable def q (a : ℝ) : Prop :=
  a < 1 ∧ a ≠ 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) :
  (1 ≤ a ∧ a < 2) ∨ a ≤ -2 ∨ a = 0 :=
by sorry

end range_of_a_l417_417804


namespace hens_ratio_l417_417651

theorem hens_ratio
  (total_chickens : ℕ)
  (fraction_roosters : ℚ)
  (chickens_not_laying : ℕ)
  (h : total_chickens = 80)
  (fr : fraction_roosters = 1/4)
  (cnl : chickens_not_laying = 35) :
  (total_chickens * (1 - fraction_roosters) - chickens_not_laying) / (total_chickens * (1 - fraction_roosters)) = 5 / 12 :=
by
  sorry

end hens_ratio_l417_417651


namespace total_chocolate_pieces_l417_417401

def total_chocolates (boxes : ℕ) (per_box : ℕ) : ℕ :=
  boxes * per_box

theorem total_chocolate_pieces :
  total_chocolates 6 500 = 3000 :=
by
  sorry

end total_chocolate_pieces_l417_417401


namespace powers_of_3_not_9_count_l417_417960

theorem powers_of_3_not_9_count : 
  (count (n : ℕ) (0 < n ∧ n < 500000 ∧ ∃ k : ℕ, n = 3^k ∧ ¬ ∃ j : ℕ, n = 9^j)) = 6 := 
sorry

end powers_of_3_not_9_count_l417_417960


namespace identify_power_functions_l417_417781

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (α : ℝ), ∀ (x : ℝ), f x = x^α

def fA (x : ℝ) : ℝ := 2 * x^(1/2)
def fB (x : ℝ) : ℝ := x^3
def fC (x : ℝ) : ℝ := (x + 1)^2
def fD (x : ℝ) : ℝ := x^(-1)

theorem identify_power_functions :
  (is_power_function fA = false) ∧
  (is_power_function fB = true) ∧
  (is_power_function fC = false) ∧
  (is_power_function fD = true) :=
by
  sorry

end identify_power_functions_l417_417781


namespace negative_reciprocal_0_l417_417218

theorem negative_reciprocal_0.125 : ∃ x : ℝ, (0.125 * x = -1) ∧ (x = -8) :=
by
  -- Definition of a reciprocal
  let reciprocal (a : ℝ) := 1 / a
  
  -- Define the conditions in the problem
  assume h1 : 0.125 * reciprocal 0.125 = 1,
  
  -- Given number reciprocal
  let r := reciprocal 0.125,
  
  -- Negative reciprocal, we multiply reciprocal by -1
  let nr := -1 * r,
  
  -- Now we assert that nr is equal to -8
  have h2 : nr = -8 := sorry,
  
  -- Converting the have statement to an exist statement for Lean's satisfaction
  exact ⟨nr, h1, h2⟩

end negative_reciprocal_0_l417_417218


namespace circle_eq_l417_417489

theorem circle_eq (x y : ℝ) (h₀ : ∃ t : ℝ, t < 0 ∧ (x, y) = (-t, ½ * t^2))
  (h₁ : circle_tangent_to_axis : Prop)
  (h₂ : circle_tangent_to_y_axis : Prop) :
  (x + 1)^2 + (y - ½)^2 = 1 := by
  sorry

end circle_eq_l417_417489


namespace find_x_l417_417273

-- Definition of the binary operation
def binary_operation (a b c d : ℤ) : ℤ × ℤ :=
  (a - c, b + d)

-- Definition of our main theorem to be proved
theorem find_x (x y : ℤ) (h : binary_operation x y 2 3 = (4, 5)) : x = 6 :=
  by sorry

end find_x_l417_417273


namespace total_length_correct_l417_417096

def segment_lengths_Figure1 : List ℕ := [10, 3, 1, 1, 5, 7]

def removed_segments : List ℕ := [3, 1, 1, 5]

def remaining_segments_Figure2 : List ℕ := [10, (3 + 1 + 1), 7, 1]

def total_length_Figure2 : ℕ := remaining_segments_Figure2.sum

theorem total_length_correct :
  total_length_Figure2 = 23 :=
by
  sorry

end total_length_correct_l417_417096


namespace vector_collinearity_magnitude_sqrt2_l417_417151

noncomputable def vector_magnitude (v : ℝ×ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem vector_collinearity_magnitude_sqrt2 (λ : ℝ) (h_collinear : 8 * (2 * λ + 1) = 24) :
  vector_magnitude (1, λ) = Real.sqrt 2 :=
by
  have : λ = 1 := by nlinarith [h_collinear]
  rw [this]
  unfold vector_magnitude
  norm_num

end vector_collinearity_magnitude_sqrt2_l417_417151


namespace last_two_digits_g_2021_l417_417469

noncomputable def g (x : ℕ) : ℝ := Nat.sqrt (4^x + Nat.sqrt (4^(x+1) + Nat.sqrt (4^(x+2) + ...)))

theorem last_two_digits_g_2021 : (2^2021 + 1) % 100 = 53 :=
by
  sorry

end last_two_digits_g_2021_l417_417469


namespace total_pieces_in_boxes_l417_417400

theorem total_pieces_in_boxes (num_boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ)
    (h1 : num_boxes = 6) (h2 : pieces_per_box = 500) :
    num_boxes * pieces_per_box = total_pieces → total_pieces = 3000 :=
by
  intro h
  rw [h1, h2] at h
  rw h
  rfl

end total_pieces_in_boxes_l417_417400


namespace badge_counts_l417_417764

variable (x : ℝ) -- Number of badges Tolya initially had
variable (y : ℝ) -- Number of badges Vasya initially had 

-- Conditions
axiom initial_more_badges : y = x + 5
axiom exchange_badges_vasya : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y
axiom exchange_badges_tolya : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y

-- Proof goal
theorem badge_counts (x : ℝ) (y : ℝ) 
  (h1 : y = x + 5) 
  (h2 : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y) 
  (h3 : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y) : 
  x = 45 ∧ y = 50 :=
by
  sorry

end badge_counts_l417_417764


namespace rattlesnakes_count_l417_417332

theorem rattlesnakes_count (total_snakes : ℕ) (boa_constrictors pythons rattlesnakes : ℕ)
  (h1 : total_snakes = 200)
  (h2 : boa_constrictors = 40)
  (h3 : pythons = 3 * boa_constrictors)
  (h4 : total_snakes = boa_constrictors + pythons + rattlesnakes) :
  rattlesnakes = 40 :=
by
  sorry

end rattlesnakes_count_l417_417332


namespace hyperbola_focal_length_l417_417179

theorem hyperbola_focal_length (m : ℝ) (h_pos : 0 < m)
  (h_asymptote : ∀ x y : ℝ, (√3 * x + m * y = 0) ↔ (x = 0 ∧ y = 0)) :
  2 * sqrt (3 + 1) = 4 := by
  sorry

end hyperbola_focal_length_l417_417179


namespace complex_conjugate_of_solution_l417_417308

theorem complex_conjugate_of_solution (z : ℂ) (h : (z - 3) * (2 - I) = 5) :
  conj(z) = 5 - I :=
sorry

end complex_conjugate_of_solution_l417_417308


namespace badge_exchange_proof_l417_417769

-- Definitions based on the conditions
def initial_badges_Tolya : ℝ := 45
def initial_badges_Vasya : ℝ := 50

def tollya_exchange_badges (badges_Tolya : ℝ) : ℝ := 0.2 * badges_Tolya
def tollya_receive_badges (badges_Vasya : ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_exchange_badges (badges_Vasya: ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_receive_badges (badges_Tolya: ℝ) : ℝ := 0.2 * badges_Tolya

-- Vasya ended up with one badge less than Tolya after the exchange
theorem badge_exchange_proof 
  (tolya_initial badges_Tolya_initial : ℝ)
  (badges_Vasya_initial: ℝ)
  (tollya_initial_has_24: tollya_receive_badges badges_Vasya_initial)
  (vasya_initial_has_20: vasya_receive_badges badges_Tolya_initial):
  (tollya_initial = initial_badges_Tolya) ∧ (vasya_initial = initial_badges_Vasya) :=
sorry

end badge_exchange_proof_l417_417769


namespace sum_first_11_terms_of_inv_a_l417_417552

variable {a : ℕ → ℕ} {n : ℕ}

-- Defining the sequence given the conditions
def seq_condition1 : Prop := (a 1 = 1)
def seq_condition2 : Prop := ∀ n : ℕ, (1 < n) → (a n - a (n - 1) = n)

-- Define the function for the sum of the first 11 terms of 1/an
def sum_inv_a (a : ℕ → ℕ) : ℚ :=
  (1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 + 1 / a 6 + 1 / a 7 + 1 / a 8 + 1 / a 9 + 1 / a 10 + 1 / a 11)

-- Main theorem stating the proof of the given problem
theorem sum_first_11_terms_of_inv_a :
  seq_condition1 ∧ seq_condition2 → sum_inv_a a = 11 / 6 :=
by
  sorry

end sum_first_11_terms_of_inv_a_l417_417552


namespace max_even_numbers_l417_417463

theorem max_even_numbers 
  (a : Fin 100 → ℕ) 
  (h : ∃ (count_even : ℕ), count_even = 32 ∧
    ∀ i, by_cases (a 0 * a 1 * a 2 * ... * a ⟨i.1, Nat.lt_of_le_of_lt (Nat.zero_le i.1) i.2⟩ = a 0 * a 1 * ... * a ⟨i.1.pred, sorry⟩ + a i.1.succ * ... * a ⟨99, sorry⟩) then b : (Fin i.1.pred) 
    count_even = 32) : 
  ∃ n, n ≤ 33 ∧ (∀ i, a i % 2 = 0 → n)

end max_even_numbers_l417_417463


namespace minimum_oxygen_cylinders_l417_417126

theorem minimum_oxygen_cylinders (max_carry : ℕ) (consume_per_day : ℕ) (camps : ℕ) : 
  max_carry = 3 ∧ consume_per_day = 1 ∧ camps = 4 → 
  minimum_cylinders max_carry consume_per_day camps = 81 := by 
  sorry

end minimum_oxygen_cylinders_l417_417126


namespace simplify_expression_correct_l417_417674

noncomputable def simplify_expression : ℝ :=
  2 * Real.sqrt (3 + Real.sqrt (5 - Real.sqrt (13 + Real.sqrt (48))))

theorem simplify_expression_correct : simplify_expression = (Real.sqrt 6) + (Real.sqrt 2) :=
  sorry

end simplify_expression_correct_l417_417674


namespace pages_per_donut_l417_417255

def pages_written (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ) : ℕ :=
  let donuts := total_calories / calories_per_donut
  total_pages / donuts

theorem pages_per_donut (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ): 
  total_pages = 12 → calories_per_donut = 150 → total_calories = 900 → pages_written total_pages calories_per_donut total_calories = 2 := by
  intros
  sorry

end pages_per_donut_l417_417255


namespace parallelogram_area_correct_l417_417921

-- Define the conditions where the angle is in degrees and the sides are given in inches.
variables (α : ℝ) (a b : ℝ)
-- Define the angle in degrees and the sides in inches based on the given problem.
def angle_A := 100
def side_AB := 15
def side_AD := 14

-- Convert the angle to radians for trigonometric calculations.
def angle_A_rad := angle_A * (Real.pi / 180)

-- Calculate the height using the sine of the angle (80 degrees in this context).
def height_DX := side_AD * Real.sin (Real.pi * 80 / 180)

-- Finally, calculate and assert the area.
noncomputable def parallelogram_area := side_AB * height_DX

-- Statement that the area of the parallelogram is approximately 206.808 square inches.
theorem parallelogram_area_correct :
  parallelogram_area = 206.808 :=
by
  sorry

end parallelogram_area_correct_l417_417921


namespace find_n_geq_13_l417_417884

-- Define the condition using max and min functions
def condition {n : ℕ} (a : Fin n → ℝ) : Prop :=
  Finset.max (Finset.univ.image a) ≤ n * Finset.min (Finset.univ.image a)

-- Define the property of three numbers being side lengths of an acute triangle
def is_acute_triangle (x y z : ℝ) : Prop :=
  x^2 + y^2 > z^2 ∧ y^2 + z^2 > x^2 ∧ z^2 + x^2 > y^2

-- Define the main theorem
theorem find_n_geq_13 (n : ℕ) (h : n ≥ 3) :
  (∀ (a : Fin n → ℝ), condition a → 
  (∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ is_acute_triangle (a i) (a j) (a k))) ↔ n ≥ 13 :=
sorry

end find_n_geq_13_l417_417884


namespace part_a_part_b_l417_417055

-- Definition of Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

-- Part (a)
theorem part_a (m : ℕ) (hm : m ≥ 2) : 
  ∃ (n1 n2 n3 n4 : ℕ), 
  10^(m-1) ≤ fib n1 ∧ fib n1 < 10^m ∧
  10^(m-1) ≤ fib n2 ∧ fib n2 < 10^m ∧
  10^(m-1) ≤ fib n3 ∧ fib n3 < 10^m ∧
  10^(m-1) ≤ fib n4 ∧ fib n4 < 10^m ∧
  ∀ n, 10^(m-1) ≤ fib n → fib n < 10^m → n ∈ {n1, n2, n3, n4, n4+1} := sorry

-- Part (b)
theorem part_b (n : ℕ) : 
  fib (5*n + 2) ≥ 10^n := sorry

end part_a_part_b_l417_417055


namespace manfred_average_paycheck_l417_417795

def average_paycheck : ℕ → ℕ → ℕ → ℕ := fun total_paychecks first_paychecks_value num_first_paychecks =>
  let remaining_paychecks_value := first_paychecks_value + 20
  let total_payment := (num_first_paychecks * first_paychecks_value) + ((total_paychecks - num_first_paychecks) * remaining_paychecks_value)
  let average_payment := total_payment / total_paychecks
  average_payment

theorem manfred_average_paycheck :
  average_paycheck 26 750 6 = 765 := by
  sorry

end manfred_average_paycheck_l417_417795


namespace time_to_pick_72_peas_l417_417848

theorem time_to_pick_72_peas :
  (∀ t : ℕ, t = 56 / 7) → (72 / 8 = 9) :=
begin
  intro t_rate,
  have rate_of_picking := t_rate 8,
  rw rate_of_picking,
  have time := 72 / 8,
  exact time,
end

end time_to_pick_72_peas_l417_417848


namespace mike_nina_meet_l417_417648

-- Defining the necessary setup 

/-- Probability that Mike and Nina meet at the cafe -/
def prob_meet_cafe : ℝ := 8 / 9

theorem mike_nina_meet (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x + 1/3 ≥ y ∧ x ≤ y + 1/3 → (1 - (2 * 1/18)) = 8 / 9 :=
by
  sorry

end mike_nina_meet_l417_417648


namespace find_c_l417_417570

noncomputable def condition1 (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

noncomputable def condition2 (c : ℝ) : Prop :=
  6 * 15 * c = 1

theorem find_c (c : ℝ) (h1 : condition1 6 15 c) (h2 : condition2 c) : c = 11 := 
by
  sorry

end find_c_l417_417570


namespace domain_g_is_1_to_2_l417_417936

variable (f : ℝ → ℝ)

-- Define the domain of f
def dom_f := (0, 4)

-- Define the function g
def g (x : ℝ) := f(x + 2) / sqrt(x - 1)

theorem domain_g_is_1_to_2 :
  (∀ x, x ∈ Ioo 0 4 → f(x).is_defined) → 
  (∀ x, x ∈ Ioo 1 2 → g(x).is_defined) := by
  sorry

end domain_g_is_1_to_2_l417_417936


namespace point_on_same_side_l417_417841

theorem point_on_same_side (p1 p2 p3 p4 : ℝ × ℝ) :
  let line_eq (p : ℝ × ℝ) := p.1 + p.2 - 1
  line_eq (1, 2) > 0 → 
  line_eq p1 <= 0 ∧
  line_eq p2 <= 0 ∧
  line_eq p3 > 0 ∧
  line_eq p4 <= 0 →
  p3 = (-1, 3) :=
by {
  intros,
  sorry
}

end point_on_same_side_l417_417841


namespace least_n_for_multiple_of_8_factorial_l417_417214

theorem least_n_for_multiple_of_8_factorial (n : ℕ) (h : ∃ k : ℕ, k + 3 ≤ n ∧ 8 ∣ factorial n) :
  4 ≤ n :=
by sorry

end least_n_for_multiple_of_8_factorial_l417_417214


namespace infinite_series_evaluation_l417_417128

theorem infinite_series_evaluation : 
  (∑' n : ℕ, (n ^ 2 + 2 * n - 2) / ((n + 3)! : ℝ)) = 1 / 3 := 
by sorry

end infinite_series_evaluation_l417_417128


namespace ratio_of_deaf_to_blind_students_l417_417090

theorem ratio_of_deaf_to_blind_students (total_students deaf_students : ℕ) 
  (h1 : total_students = 240) 
  (h2 : deaf_students = 180) : 
  let blind_students := total_students - deaf_students 
  in (deaf_students : ℚ) / (blind_students : ℚ) = 3 :=
by 
  let blind_students := total_students - deaf_students
  have h_blind : blind_students = 240 - 180 := by rw [h1, h2]
  have h_ratio : (deaf_students : ℚ) / (blind_students : ℚ) = 180 / 60 := by rw [h2, h_blind]
  rw [rat.div_eq_div_iff, rat.eq_iff_mul_eq_mul] at h_ratio
  exact h_ratio

end ratio_of_deaf_to_blind_students_l417_417090


namespace percentage_gain_is_10_l417_417077

-- Define the given conditions
variables {CP SP_loss SP_gain : ℝ}
variable (H1 : 0.9 * CP = 720)
variable (H2 : SP_gain = 880)

-- Define the required percentage gain, let's call it PG
def percentage_gain := ((SP_gain - CP) / CP) * 100

-- The theorem to be proved
theorem percentage_gain_is_10 : percentage_gain = 10 :=
by
  sorry

end percentage_gain_is_10_l417_417077


namespace gcd_all_abc_plus_cba_l417_417976

noncomputable def gcd_of_abc_cba (a : ℕ) (b : ℕ := 2 * a) (c : ℕ := 3 * a) : ℕ :=
  let abc := 64 * a + 8 * b + c
  let cba := 64 * c + 8 * b + a
  Nat.gcd (abc + cba) 300

theorem gcd_all_abc_plus_cba (a : ℕ) : gcd_of_abc_cba a = 300 :=
  sorry

end gcd_all_abc_plus_cba_l417_417976


namespace chips_probability_l417_417073

-- Definitions for the chips and their counts
def tan_chips := 4
def pink_chips := 3
def violet_chips := 5
def green_chips := 2
def total_chips := tan_chips + pink_chips + violet_chips + green_chips

-- Factorial function for combinations
def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Definition of factorial leveraging standard library
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Expected probability calculation
def probability : ℚ := 
  (factorial 4 * factorial 3 * factorial 5 * factorial 5) / (factorial 14)

-- Define an expected result based on given problem statement
def expected_probability := (1 : ℚ) / 42000

-- The theorem to be proved
theorem chips_probability : probability = expected_probability := by
  sorry

end chips_probability_l417_417073


namespace largest_85_digit_number_satisfies_property_l417_417141

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def product_of_digits (n : ℕ) : ℕ :=
  n.digits.prod

def largest_85_digit_number := 8322111111111111111111111111111111111111111111111111111111111111111111111111111

theorem largest_85_digit_number_satisfies_property :
  10^84 ≤ largest_85_digit_number ∧ largest_85_digit_number < 10^85 ∧
  sum_of_digits largest_85_digit_number = product_of_digits largest_85_digit_number :=
by
  sorry

end largest_85_digit_number_satisfies_property_l417_417141


namespace max_acute_angles_in_non_convex_n_gon_l417_417042

theorem max_acute_angles_in_non_convex_n_gon (n : ℕ) (h_n : 3 ≤ n) :
  ∃ k : ℕ, k ≤ (2 * n) / 3 + 1 ∧
  (∀ θ : ℕ, θ < 90 → θ ∈ set.Ico 0 k → acute θ) ∧ 
  ∑ θ in (set.range k), (θ) = (n - 2) * 180 :=
begin
  sorry
end

def acute (θ : ℕ) : Prop :=
  θ < 90

end max_acute_angles_in_non_convex_n_gon_l417_417042


namespace no_n_divisible_by_1955_l417_417872

theorem no_n_divisible_by_1955 : ¬ ∃ n : ℤ, 1955 ∣ (n^2 + n + 1) := by
  have prime_factors : 1955 = 5 * 17 * 23 := by sorry
  intros n h
  have div_5 : 5 ∣ (n^2 + n + 1) := by sorry
  have remainders_mod_5 : ∀ k : ℤ, (k % 5) = 0 ∨ (k % 5) = 1 ∨ (k % 5) = 2 ∨ (k % 5) = 3 ∨ (k % 5) = 4 := by sorry
  have not_mod_5_zero : ∀ k : ℤ, (k^2 + k + 1) % 5 ≠ 0 := by sorry
  have contradiction : false := by sorry
  exact contradiction

end no_n_divisible_by_1955_l417_417872


namespace smallest_positive_debt_resolvable_l417_417362

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 250

/-- The value of a lamb in dollars -/
def lamb_value : ℕ := 150

/-- Given a debt D that can be expressed in the form of 250s + 150l for integers s and l,
prove that the smallest positive amount of D is 50 dollars -/
theorem smallest_positive_debt_resolvable : 
  ∃ (s l : ℤ), sheep_value * s + lamb_value * l = 50 :=
sorry

end smallest_positive_debt_resolvable_l417_417362


namespace parking_lot_cars_l417_417588

theorem parking_lot_cars (total_capacity : ℕ) (num_levels : ℕ) (parked_cars : ℕ) 
  (h1 : total_capacity = 425) (h2 : num_levels = 5) (h3 : parked_cars = 23) : 
  (total_capacity / num_levels) - parked_cars = 62 :=
by
  sorry

end parking_lot_cars_l417_417588


namespace sum_of_mean_median_mode_is_133_12_l417_417373

def numbers : List ℚ := [1, 4, 4, 2, 2, 2, 5, 3, 4, 5, 5, 6]

noncomputable def mean (l : List ℚ) := l.sum / l.length

noncomputable def median (l : List ℚ) :=
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 0 then
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2
  else
    sorted.get (sorted.length / 2)

def mode (l : List ℚ) :=
  let counts := l.foldl (λ m x => m.insert x (m.findD x 0 + 1)) (RBMap.empty ℚ ℕ compare)
  let maxCount := counts.fold 0 (λ _ count maxCount => max maxCount count)
  let modes := counts.fold [] (λ key count acc => if count = maxCount then key :: acc else acc)
  modes.head

noncomputable def sum_mean_median_mode : ℚ :=
  mean numbers + median numbers + mode numbers

theorem sum_of_mean_median_mode_is_133_12 :
  sum_mean_median_mode = 133 / 12 := 
  by
  sorry

end sum_of_mean_median_mode_is_133_12_l417_417373


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417041

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417041


namespace difference_between_number_and_its_fraction_l417_417488

theorem difference_between_number_and_its_fraction (x : ℕ) (h : x = 150) :
  x - (3 * x / 5) = 60 :=
by 
  -- Assumption: the number is 150
  rw h
  -- Calculate the difference between 150 and its 3/5
  sorry

end difference_between_number_and_its_fraction_l417_417488


namespace sum_of_xyz_l417_417211

theorem sum_of_xyz (x y z : ℝ) (h : (x - 3)^2 + (y - 4)^2 + (z - 5)^2 = 0) : x + y + z = 12 :=
sorry

end sum_of_xyz_l417_417211


namespace solve_for_y_l417_417112

theorem solve_for_y (y : ℤ) (h : 3^(y + 1) + 3^(y + 1) + 3^(y + 1) = 2187) : y = 5 := 
by
  sorry

end solve_for_y_l417_417112


namespace intersection_question_l417_417927

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2 * a - 1}

theorem intersection_question : M ∩ N = {1} :=
by sorry

end intersection_question_l417_417927


namespace real_estate_commission_l417_417791

theorem real_estate_commission (commission_rate commission selling_price : ℝ) 
  (h1 : commission_rate = 0.06) 
  (h2 : commission = 8880) : 
  selling_price = 148000 :=
by
  sorry

end real_estate_commission_l417_417791


namespace right_triangle_AD_square_DE_square_AE_square_l417_417595

theorem right_triangle_AD_square_DE_square_AE_square (A B C D E : ℝ) (a : ℝ) 
  (hA : A = 0) (hABC_right_angle : ∠ A = 90)
  (hBC : BC = a)
  (hD : D = B + a/3)
  (hE : E = B + 2 * a/3) :
  (AD^2 + DE^2 + AE^2 = 2/3 * a^2) :=
sorry

end right_triangle_AD_square_DE_square_AE_square_l417_417595


namespace sequence_solution_l417_417550

theorem sequence_solution (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, (2*n - 1) * a (n + 1) = (2*n + 1) * a n) : 
∀ n : ℕ, a n = 2 * n - 1 := 
by
  sorry

end sequence_solution_l417_417550


namespace sum_of_n_satisfying_condition_l417_417780

open Nat

theorem sum_of_n_satisfying_condition : 
  let n_vals := [n | n <- [1..499], lcm n 100 = gcd n 100 + 450]
  (∑ x in n_vals.to_finset, x) = 250 := 
by {
  sorry,
}

end sum_of_n_satisfying_condition_l417_417780


namespace tangent_line_to_curve_at_Mpi_l417_417314

noncomputable def tangent_line_eq_at_point (x : ℝ) (y : ℝ) : Prop :=
  y = (Real.sin x) / x

theorem tangent_line_to_curve_at_Mpi :
  (∀ x y, tangent_line_eq_at_point x y →
    (∃ (m : ℝ), m = -1 / π) →
    (∀ x1 y1 (hx : x1 = π) (hy : y1 = 0), x + π * y - π = 0)) :=
by
  sorry

end tangent_line_to_curve_at_Mpi_l417_417314


namespace cone_shaped_jar_price_l417_417414

noncomputable def priceConeShapedJar (d_cyl h_cyl price_cyl d_cone h_cone : ℝ) : ℝ :=
let r_cyl := d_cyl / 2
let vol_cyl := π * r_cyl ^ 2 * h_cyl
let price_per_cubic_inch := price_cyl / vol_cyl
let r_cone := d_cone / 2
let vol_cone := (1 / 3) * π * r_cone ^ 2 * h_cone
vol_cone * price_per_cubic_inch

theorem cone_shaped_jar_price :
  priceConeShapedJar 4 5 1.20 8 10 = 3.20 :=
by
  sorry

end cone_shaped_jar_price_l417_417414


namespace distance_between_trees_l417_417071

theorem distance_between_trees (yard_length : ℕ) (number_of_trees : ℕ) (number_of_gaps : ℕ)
  (h1 : yard_length = 400) (h2 : number_of_trees = 26) (h3 : number_of_gaps = number_of_trees - 1) :
  yard_length / number_of_gaps = 16 := by
  sorry

end distance_between_trees_l417_417071


namespace nth_permutation_2013_eq_3546127_l417_417540

-- Given the digits 1 through 7, there are 7! = 5040 permutations.
-- We want to prove that the 2013th permutation in ascending order is 3546127.

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def nth_permutation (n : ℕ) (digits : List ℕ) : List ℕ :=
  sorry

theorem nth_permutation_2013_eq_3546127 :
  nth_permutation 2013 digits = [3, 5, 4, 6, 1, 2, 7] :=
sorry

end nth_permutation_2013_eq_3546127_l417_417540


namespace first_quarter_profit_l417_417833

def profits := ℕ
def q3_profit : profits := 3000
def q4_profit : profits := 2000
def annual_profit : profits := 8000

theorem first_quarter_profit :
  (annual_profit - (q3_profit + q4_profit) = 3000) :=
by
  sorry

end first_quarter_profit_l417_417833


namespace midpoint_parallelogram_l417_417984

variables {A B C D E K Y Y1 X X1 : Type}

def is_midpoint {V : Type} [AddCommGroup V] [Vector N] (a b p : V) : Prop :=
  p = (a + b) / 2

def is_parallelogram {V : Type} [AddCommGroup V] [Vector N] (p1 p2 p3 p4 : V) : Prop :=
  (p2 - p1) + (p4 - p3) = 0 ∧ (p3 - p2) + (p1 - p4) = 0

variables {V : Type} [AddCommGroup V] [VectorSpace ℝ V]

theorem midpoint_parallelogram
  {A B C D : V}
  (E : V) (K : V)
  (hE : is_midpoint A B E)
  (hK : is_midpoint C D K) :
  ∃ Y Y1 X X1 :V, 
    is_midpoint A K Y ∧
    is_midpoint C E Y1 ∧
    is_midpoint B K X1 ∧
    is_midpoint D E X ∧
    is_parallelogram Y Y1 X X1 :=
begin
  sorry
end

end midpoint_parallelogram_l417_417984


namespace number_of_coloring_methods_l417_417364

-- Definition of the problem in Lean 4

def num_colors : ℕ := 6
def stations : Finset (String) := {"P", "A", "B", "C", "D", "E"}
-- Function that assigns a color to each station
def color_assignment := stations → Fin num_colors

def adjacent_stations : List (String × String) := [("P", "A"), ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]

/-- Two stations have different colors if they are adjacent. --/
def adjacent_different_colors (f : color_assignment) : Prop :=
  ∀ (s1 s2 : String), (s1, s2) ∈ adjacent_stations → f s1 ≠ f s2

/-- The total number of valid colorings --/
def num_painting_methods : ℕ := 6120

theorem number_of_coloring_methods :
  (∃ f : color_assignment, adjacent_different_colors f) →  ∃ cnt : ℕ, cnt = num_painting_methods :=
sorry

end number_of_coloring_methods_l417_417364


namespace exists_hamiltonian_cycle_with_edges_removed_l417_417535

-- Define the vertices as numbers from 0 to 1023, which represents 1024 cities.
def vertices : Finset ℕ := Finset.range 1024

-- Define the edges for the Q_10 hypercube graph.
def is_edge (m n : ℕ) : Prop :=
  ∃ k : ℕ, 0 ≤ k ∧ k < 10 ∧ Nat.test_bit m k ≠ Nat.test_bit n k ∧ ∀ j, j ≠ k → Nat.test_bit m j = Nat.test_bit n j

-- Define the graph with 8 edges removed
def Q10_with_edges_removed : Graph ℕ := 
  -- Initial graph
  { vertices := vertices,
    is_edge := λ m n, is_edge m n ∧ ¬Set.finite ⟨m, n⟩}

-- Problem definition: Proving existence of a Hamiltonian cycle
theorem exists_hamiltonian_cycle_with_edges_removed (E_removed : Set (ℕ × ℕ)) (hE_removed : E_removed.finite ∧ E_removed.card = 8) :
  ∃ (cycle : List ℕ), ∀ (v ∈ vertices), v ∈ cycle ∧ ∀ i < 1024, is_edge (cycle.get i) (cycle.get ((i + 1) % 1024)) :=
sorry

end exists_hamiltonian_cycle_with_edges_removed_l417_417535


namespace f_10_equals_1_l417_417701

noncomputable def f : ℝ → ℝ 
| x => sorry 

axiom odd_f_x_minus_1 : ∀ x : ℝ, f (-x-1) = -f (x-1)
axiom even_f_x_plus_1 : ∀ x : ℝ, f (-x+1) = f (x+1)
axiom f_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = 2^x

theorem f_10_equals_1 : f 10 = 1 :=
by
  sorry -- The actual proof goes here.

end f_10_equals_1_l417_417701


namespace profit_percentage_is_60_l417_417103

-- Define the conditions
def selling_price : ℝ := 600
def cost_price : ℝ := 375

-- Calculate profit percentage
def profit_percentage : ℝ := ((selling_price - cost_price) / cost_price) * 100

-- The theorem stating the profit percentage
theorem profit_percentage_is_60 : profit_percentage = 60 := by
  -- skipping the actual proof for brevity
  sorry

end profit_percentage_is_60_l417_417103


namespace original_team_members_l417_417095

theorem original_team_members (x : ℕ) (h1 : x - 2 members didn't show up) (h2 : (x - 2) * 4 = 20) : x = 7 :=
by
  sorry

end original_team_members_l417_417095


namespace t_le_s_l417_417911

theorem t_le_s (a b : ℝ) (t s : ℝ) (h1 : t = a + 2 * b) (h2 : s = a + b^2 + 1) : t ≤ s :=
by
  sorry

end t_le_s_l417_417911


namespace farmer_apples_l417_417697

theorem farmer_apples : 127 - 39 = 88 := by
  -- Skipping proof details
  sorry

end farmer_apples_l417_417697


namespace carpet_needed_in_sq_yards_l417_417089

theorem carpet_needed_in_sq_yards :
  let length := 15
  let width := 10
  let area_sq_feet := length * width
  let conversion_factor := 9
  let area_sq_yards := area_sq_feet / conversion_factor
  area_sq_yards = 16.67 := by
  sorry

end carpet_needed_in_sq_yards_l417_417089


namespace maximum_value_problem_l417_417913

theorem maximum_value_problem (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a^2 - b * c) * (b^2 - c * a) * (c^2 - a * b) ≤ 1 / 8 :=
sorry

end maximum_value_problem_l417_417913


namespace jason_loan_payoff_months_l417_417254

-- Define the conditions
def car_cost : ℝ := 32000
def down_payment : ℝ := 8000
def monthly_payment : ℝ := 525
def interest_rate : ℝ := 0.05

-- Define the required total loan amount and monthly interest
def loan_amount : ℝ := car_cost - down_payment
def monthly_interest : ℝ := interest_rate * monthly_payment
def principal_payment : ℝ := monthly_payment - monthly_interest

-- Define the target number of months
def target_months : ℝ := loan_amount / principal_payment

-- Theorem to prove the number of months needed to pay off the loan is 49
theorem jason_loan_payoff_months : ⌈target_months⌉ = 49 :=
by
  sorry

end jason_loan_payoff_months_l417_417254


namespace Alex_Needs_31_Gallons_l417_417098

/-
Question: How many gallons of paint does Alex need to buy to paint the curved surfaces of 20 right, cylindrical pillars?
Conditions:
1. Alex will paint 20 right, cylindrical pillars.
2. Only the curved surfaces of the pillars will be painted.
3. Each pillar has a height of 15 feet.
4. Each pillar has a diameter of 8 feet.
5. One gallon of paint covers 250 square feet.
6. Paint is only sold in full gallons.

Mathematically Equivalent Proof Problem: Prove that Alex must buy 31 gallons of paint to cover the curved surfaces of 20 right, cylindrical pillars, each with a height of 15 feet and a diameter of 8 feet, given that one gallon of paint covers 250 square feet and paint is sold only in full gallons.
-/
theorem Alex_Needs_31_Gallons :
  ∀ (n : ℕ) (h d : ℝ) (coverage : ℝ),
  n = 20 → h = 15 → d = 8 → coverage = 250 →
  ⌈((n : ℝ) * (real.pi * (d / 2) * h) / coverage)⌉ = 31 :=
by
  intros n h d coverage hn hh hd hcov
  sorry

end Alex_Needs_31_Gallons_l417_417098


namespace not_p_and_q_implies_a_range_l417_417548

theorem not_p_and_q_implies_a_range (a : ℝ) :
  (¬ (∀ x ∈ set.Icc (0 : ℝ) 1, a ≥ real.exp x) ∨ ¬ (∃ x0 : ℝ, x0 ^ 2 + 4 * x0 + a = 0)) →
  (a < real.exp 1 ∨ a > 4) :=
by
  sorry

end not_p_and_q_implies_a_range_l417_417548


namespace badges_exchange_l417_417758

theorem badges_exchange (Vasya_initial Tolya_initial : ℕ) 
    (h1 : Vasya_initial = Tolya_initial + 5)
    (h2 : Vasya_initial - 0.24 * Vasya_initial + 0.20 * Tolya_initial = Tolya_initial - 0.20 * Tolya_initial + 0.24 * Vasya_initial - 1) 
    : Vasya_initial = 50 ∧ Tolya_initial = 45 :=
by sorry

end badges_exchange_l417_417758


namespace water_required_to_react_l417_417138

def molecules (C O H: Type) [Mul C O] [Mul O H] [Mul H C] := 
  (C * O * 2) → (H * O * 3)

theorem water_required_to_react (CO2 H2CO3 : ℕ) : 
  CO2 = 2 ∧ H2CO3 = 2 → ∃ H2O: ℕ, H2O = 2 :=
by
  intro h
  cases h with hCO2 hH2CO3
  use 2
  exact sorry

end water_required_to_react_l417_417138


namespace parallel_necessary_not_sufficient_for_orthogonal_l417_417896

-- Definitions of vector orthogonality and parallelism in Lean
variables {V : Type*} [inner_product_space ℝ V]

def orthogonal (a b : V) : Prop := ⟪a, b⟫ = 0
def parallel (a b : V) : Prop := ∃ k : ℝ, k ≠ 0 ∧ a = k • b

-- Non-zero vectors a and b
variables (a b : V)
hypothesis (a_nonzero : a ≠ 0)
hypothesis (b_nonzero : b ≠ 0)

-- The main statement of the theorem
theorem parallel_necessary_not_sufficient_for_orthogonal :
  parallel a b → orthogonal a b → (parallel a b ∧ ¬orthogonal a b) := sorry

end parallel_necessary_not_sufficient_for_orthogonal_l417_417896


namespace probability_at_least_one_two_l417_417353

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end probability_at_least_one_two_l417_417353


namespace badge_exchange_proof_l417_417771

-- Definitions based on the conditions
def initial_badges_Tolya : ℝ := 45
def initial_badges_Vasya : ℝ := 50

def tollya_exchange_badges (badges_Tolya : ℝ) : ℝ := 0.2 * badges_Tolya
def tollya_receive_badges (badges_Vasya : ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_exchange_badges (badges_Vasya: ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_receive_badges (badges_Tolya: ℝ) : ℝ := 0.2 * badges_Tolya

-- Vasya ended up with one badge less than Tolya after the exchange
theorem badge_exchange_proof 
  (tolya_initial badges_Tolya_initial : ℝ)
  (badges_Vasya_initial: ℝ)
  (tollya_initial_has_24: tollya_receive_badges badges_Vasya_initial)
  (vasya_initial_has_20: vasya_receive_badges badges_Tolya_initial):
  (tollya_initial = initial_badges_Tolya) ∧ (vasya_initial = initial_badges_Vasya) :=
sorry

end badge_exchange_proof_l417_417771


namespace estimate_fish_population_l417_417074

theorem estimate_fish_population :
  ∀ (initial_tagged: ℕ) (august_sample: ℕ) (tagged_in_august: ℕ) (leaving_rate: ℝ) (new_rate: ℝ),
  initial_tagged = 50 →
  august_sample = 80 →
  tagged_in_august = 4 →
  leaving_rate = 0.30 →
  new_rate = 0.45 →
  ∃ (april_population : ℕ),
  april_population = 550 :=
by
  intros initial_tagged august_sample tagged_in_august leaving_rate new_rate
  intros h_initial_tagged h_august_sample h_tagged_in_august h_leaving_rate h_new_rate
  existsi 550
  sorry

end estimate_fish_population_l417_417074


namespace find_c_from_minimum_value_l417_417454

-- Define the function y = c sec(dx)
def my_function (c d x : ℝ) : ℝ := c * (1 / real.cos (d * x))

-- Define the minimum value from the graph where y > 0
axiom minimum_value_condition : ∃ c d x : ℝ, c > 0 ∧ d > 0 ∧ my_function c d x = 3 ∧ my_function c d x > 0

-- Theorem: If the minimum value of y = c sec(dx) for y > 0 is 3, then c = 3
theorem find_c_from_minimum_value : ∀ c d x : ℝ, c > 0 ∧ d > 0 → my_function c d x = 3 ∧ my_function c d x > 0 → c = 3 :=
by
  sorry

end find_c_from_minimum_value_l417_417454


namespace town_population_in_5_years_l417_417321

def population_after_n_years (P₀ : ℕ) (n : ℕ) : ℕ :=
  let rec growth_rate (year : ℕ) : ℚ :=
    10 + 2 * (year - 1)
  let rec population (P : ℚ) (year : ℕ) : ℚ :=
    match year with
    | 0 => P
    | _ => let new_population = P * (1 + growth_rate(year) / 100)
           population new_population (year - 1)
  population P₀ n

theorem town_population_in_5_years :
  population_after_n_years 300 5 ≈ 576.734912 :=
sorry

end town_population_in_5_years_l417_417321


namespace field_area_in_acres_l417_417236

theorem field_area_in_acres :
  ∃ (length width diagonal : ℕ), 
    length = 30 ∧ 
    (width + diagonal = 50) ∧ 
    (30 * 30 + width * width = diagonal * diagonal) ∧ 
    (length * width / 240 = 2) :=
begin
  sorry
end

end field_area_in_acres_l417_417236


namespace length_PQ_correct_l417_417248

noncomputable def triangle_length_PQ 
  (A B C P Q D : Type) 
  (angle_B angle_C : ℝ) 
  (BC AP AQ : ℝ) 
  (perpendicular_AD_BC : Prop) 
  (dihedral_angle_ADB_ADC : ℝ) : ℝ :=
PQ

theorem length_PQ_correct :
  ∀ (A B C P Q D : Type)
    (angle_B angle_C : ℝ) 
    (BC AP AQ : ℝ) 
    (perpendicular_AD_BC : Prop) 
    (dihedral_angle_ADB_ADC : ℝ),
  angle_B = 30 ∧ angle_C = 30 ∧ BC = 2 * real.sqrt 3 ∧ AP = 1 ∧ AQ = real.sqrt 2 ∧ perpendicular_AD_BC ∧ dihedral_angle_ADB_ADC = 60 →
  triangle_length_PQ A B C P Q D angle_B angle_C BC AP AQ perpendicular_AD_BC dihedral_angle_ADB_ADC =
  (real.sqrt (12 - 5 * real.sqrt 2)) / 2 :=
begin
  sorry
end

end length_PQ_correct_l417_417248


namespace gino_gave_away_l417_417508

theorem gino_gave_away (initial_sticks given_away left_sticks : ℝ) 
  (h1 : initial_sticks = 63.0) (h2 : left_sticks = 13.0) 
  (h3 : left_sticks = initial_sticks - given_away) : 
  given_away = 50.0 :=
by
  sorry

end gino_gave_away_l417_417508


namespace liams_bills_cost_l417_417279

/-- Let's define the conditions given in the problem. --/
noncomputable theory

def monthly_savings : ℤ := 500
def savings_duration_years : ℤ := 2
def total_savings : ℤ := monthly_savings * 12 * savings_duration_years
def remaining_after_bills : ℤ := 8500

/-- Now prove that Liam's bills cost $3,500. --/
theorem liams_bills_cost :
  total_savings - remaining_after_bills = 3500 :=
by
  sorry

end liams_bills_cost_l417_417279


namespace preimages_of_f_l417_417194

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem preimages_of_f (k : ℝ) : (∃ x₁ x₂ : ℝ, f x₁ = k ∧ f x₂ = k ∧ x₁ ≠ x₂) ↔ k < 1 := by
  sorry

end preimages_of_f_l417_417194


namespace parking_lot_cars_l417_417587

theorem parking_lot_cars (total_capacity : ℕ) (num_levels : ℕ) (parked_cars : ℕ) 
  (h1 : total_capacity = 425) (h2 : num_levels = 5) (h3 : parked_cars = 23) : 
  (total_capacity / num_levels) - parked_cars = 62 :=
by
  sorry

end parking_lot_cars_l417_417587


namespace part_I_part_II_l417_417517

section problem

-- Given conditions
def line_eq (ρ θ : ℝ) : Prop := ρ * (cos θ - sin θ) = 1
def curve_c (α : ℝ) : ℝ × ℝ := (2 * cos α, 2 + 2 * sin α)

-- Part (I)
def midpoint (M : ℝ × ℝ) : ℝ × ℝ := (M.fst / 2, M.snd / 2)

theorem part_I (α : ℝ) :
  let P := midpoint (curve_c α) in
  (P.fst)^2 + (P.snd - 1)^2 = 1 :=
sorry

-- Part (II)
def line_eq_std (x y : ℝ) : Prop := x - y = 1
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Minimum distance from the circle's center to the line, minus radius
theorem part_II :
  let d := abs(0 - 2 - 1) / sqrt 2 - 2 in
  d = (3 * sqrt 2) / 2 - 2 :=
sorry

end problem

end part_I_part_II_l417_417517


namespace eq_exactly_three_distinct_real_roots_l417_417574

theorem eq_exactly_three_distinct_real_roots (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = |x^2 + a * x| - 4) ∧ f =0 ∧ (∃! x₀, f x₀ = 0)) → (a = 4 ∨ a = -4) :=
begin
  sorry
end

end eq_exactly_three_distinct_real_roots_l417_417574


namespace tan_pi_over_12_eq_l417_417494

theorem tan_pi_over_12_eq : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 :=
by
  sorry

end tan_pi_over_12_eq_l417_417494


namespace T_100_gt_T_99_l417_417895

-- Definition: T(n) denotes the number of ways to place n objects of weights 1, 2, ..., n on a balance such that the sum of the weights in each pan is the same.
def T (n : ℕ) : ℕ := sorry

-- Theorem we need to prove
theorem T_100_gt_T_99 : T 100 > T 99 := 
sorry

end T_100_gt_T_99_l417_417895


namespace solve_for_2a_plus_b_l417_417961

theorem solve_for_2a_plus_b : 
  ∀ (a b : ℝ), 2 * a + 3 * b = 18 ∧ 4 * a + 5 * b = 31 → 2 * a + b = 8 :=
by
  intros a b h
  cases h with h1 h2
  sorry

end solve_for_2a_plus_b_l417_417961


namespace remainder_2n_div_9_l417_417382

theorem remainder_2n_div_9 (n : ℤ) (h : n % 18 = 10) : (2 * n) % 9 = 2 := 
sorry

end remainder_2n_div_9_l417_417382


namespace always_true_inequality_l417_417051

theorem always_true_inequality (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by
  sorry

end always_true_inequality_l417_417051


namespace domain_implies_range_of_a_range_implies_range_of_a_l417_417068

theorem domain_implies_range_of_a (a : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + 1 > 0) → 1 < a := by
  sorry

theorem range_implies_range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, log 2 (ax^2 + 2*x + 1) = y) → 0 ≤ a ∧ a ≤ 1 := by
  sorry

end domain_implies_range_of_a_range_implies_range_of_a_l417_417068


namespace arithmetic_sequence_value_l417_417165

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_value (h : a 3 + a 5 + a 11 + a 13 = 80) : a 8 = 20 :=
sorry

end arithmetic_sequence_value_l417_417165


namespace height_of_fourth_tree_l417_417334

theorem height_of_fourth_tree
  (h_A h_B h_C : ℝ)
  (a b c : ℝ)
  (ha: h_A = a)
  (hb: h_B = b)
  (hc: h_C = c) :
  ∃ d : ℝ, d = sqrt (a^2 + c^2 - b^2) :=
  sorry

end height_of_fourth_tree_l417_417334


namespace f_eq_2fac_n_n1_l417_417325

def a_seq : ℕ → ℚ 
| 0 := 0 
| 1 := 1 
| n := (1/2 * n * a_seq (n-1)) + (1/2 * n * (n-1) * a_seq (n-2)) + (-1)^n * (1 - (n/2))

def f (n : ℕ) : ℚ := 
  a_seq n + ∑ i in finset.range (n-1), (i+1) * nat.choose n i * a_seq (n - i - 1)

theorem f_eq_2fac_n_n1 (n : ℕ) : 
  (f n) = 2 * nat.factorial n - (n + 1) := 
by 
  sorry

end f_eq_2fac_n_n1_l417_417325


namespace no_natural_with_all_permutations_perfect_squares_l417_417663

def digits_are_non_zero (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d ≠ 0

def permutations_are_perfect_squares (n : ℕ) : Prop :=
  ∀ m ∈ (List.permutations (n.digits 10)), (fromDigits m) ∈ (List.range (⌊(10^((m.length + 1) / 2))⌋ + 1)).map (λ x, x^2)

theorem no_natural_with_all_permutations_perfect_squares (n : ℕ) (h1 : 10 ≤ n)
  (h2 : digits_are_non_zero n) : ¬ permutations_are_perfect_squares n :=
sorry

end no_natural_with_all_permutations_perfect_squares_l417_417663


namespace circle_C2_equation_line_l_equation_l417_417514

-- Proof problem 1: Finding the equation of C2
theorem circle_C2_equation (C1_center_x C1_center_y : ℝ) (A_x A_y : ℝ) 
  (C2_center_x : ℝ) (C1_radius : ℝ) :
  C1_center_x = 6 ∧ C1_center_y = 7 ∧ C1_radius = 5 →
  A_x = 2 ∧ A_y = 4 →
  C2_center_x = 6 →
  (∀ y : ℝ, ((y - C1_center_y = C1_radius + (C1_radius + (y - C1_center_y)))) →
    (x - C2_center_x)^2 + (y - C2_center_y)^2 = 1) :=
sorry

-- Proof problem 2: Finding the equation of the line l
theorem line_l_equation (O_x O_y A_x A_y : ℝ) 
  (C1_center_x C1_center_y : ℝ) 
  (A_BC_dist : ℝ) :
  O_x = 0 ∧ O_y = 0 →
  A_x = 2 ∧ A_y = 4 →
  C1_center_x = 6 ∧ C1_center_y = 7 →
  A_BC_dist = 2 * (25^(1 / 2)) →
  ((2 : ℝ)*x - y + 5 = 0 ∨ (2 : ℝ)*x - y - 15 = 0) :=
sorry

end circle_C2_equation_line_l_equation_l417_417514


namespace probability_at_least_one_two_l417_417352

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end probability_at_least_one_two_l417_417352


namespace hyperbola_eccentricity_eq_sqrt5_l417_417931

-- Definitions corresponding to problem conditions
variables (a b c : ℝ) (h : a > 0) (k : b > 0)
-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
-- Definition of the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = (b / a) * x
-- Definition of the foci (F1 and F2)
def F1 : ℝ × ℝ := (-c, 0)
def F2 : ℝ × ℝ := (c, 0)
-- Definition of the point P
def P : ℝ × ℝ := (2*a^2 / c - c, 2*a*b / c)

-- The proof that we need to formulate in Lean
theorem hyperbola_eccentricity_eq_sqrt5 
  (h_hyperbola : ∀ x y, hyperbola x y)
  (h_symm : ∃ P : ℝ × ℝ, P ∈ { Q | hyperbola Q.fst Q.snd ∧ symmetry_line Q.fst Q.snd }) :
  ∃ e, e = real.sqrt 5 := sorry

end hyperbola_eccentricity_eq_sqrt5_l417_417931


namespace problem_l417_417191

def f : ℝ → ℝ :=
  λ x, if x ≤ 1 then 2^x else f (x - 1)

theorem problem :
  f (Real.log 3 / Real.log 2 + 2016) = 3 / 2 :=
sorry

end problem_l417_417191


namespace complex_magnitude_l417_417934

open Complex

noncomputable def complexZ : ℂ := sorry -- Definition of complex number z

theorem complex_magnitude (z : ℂ) (h : (1 + 2 * Complex.I) * z = -3 + 4 * Complex.I) : abs z = Real.sqrt 5 :=
sorry

end complex_magnitude_l417_417934


namespace a_n_monotonic_increasing_a_n_tends_to_zero_l417_417715

noncomputable def p : ℕ → (ℝ → ℝ) 
| 1 := λ x, 1 + x
| 2 := λ x, 1 + 2 * x
| (2*n + 1) := λ x, p (2*n) x + (n + 1) * x * p (2*n - 1) x
| (2*n + 2) := λ x, p (2*n + 1) x + (n + 1) * x * p (2*n) x

noncomputable def a (n : ℕ) : ℝ := sorry -- definition of largest real root of p_n(x)

theorem a_n_monotonic_increasing (n : ℕ) : 
  a (n + 1) > a n := sorry
  
theorem a_n_tends_to_zero (ε : ℝ) (hε : ε > 0) : 
  ∃ (N : ℕ), ∀ n ≥ N, abs (a n) < ε := sorry

end a_n_monotonic_increasing_a_n_tends_to_zero_l417_417715


namespace find_positive_integers_y_l417_417635

noncomputable def lcm_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

theorem find_positive_integers_y (k : ℕ) (a : Fin (k + 1) → ℕ) (y : ℕ) (hk : 1 ≤ k)
    (h0 : a 0 ∣ y)
    (h1 : ∀ i : Fin (k + 1), a 0 + a i ∣ y + a i) :
    ∃ n : ℕ, y = n * lcm_list (List.ofFn fun i ↦ a 0 + a i.val) + a 0 := sorry

end find_positive_integers_y_l417_417635


namespace find_focal_length_of_hyperbola_l417_417181

def focal_length_of_hyperbola (m : ℝ) (h_m_gt_zero : m > 0) (h_asymptote : sqrt 3 * x + m * y = 0) : ℝ :=
4

theorem find_focal_length_of_hyperbola (m : ℝ) (h_m_gt_zero : m > 0) (h_asymptote : sqrt 3 * x + m * y = 0) :
  focal_length_of_hyperbola m h_m_gt_zero h_asymptote = 4 :=
sorry

end find_focal_length_of_hyperbola_l417_417181


namespace max_value_f_l417_417320

def f (x : ℝ) : ℝ :=
  Real.sqrt (x^4 - 3 * x^2 - 6 * x + 13) - Real.sqrt (x^4 - x^2 + 1)

theorem max_value_f : ∃ x : ℝ, f x = Real.sqrt 10 :=
by
  sorry

end max_value_f_l417_417320


namespace number_of_positive_integer_divisors_of_N_l417_417499

noncomputable def f (a : Fin 10 → ℕ) : ℕ :=
  (∏ i in Finset.range 10, Nat.choose (20 - (Finset.range i).sum (λ j, a j)) (a i)) *
  (∑ i in Finset.range 10, Nat.choose (18 + i + 1) 19 * a i)

def SatisfiesCondition (a : Fin 10 → ℕ) : Prop :=
  (Finset.univ.sum a = 20)

def N : ℕ :=
  (Finset.univ.filter SatisfiesCondition).sum (λ a, f a) / (Finset.univ.filter SatisfiesCondition).card

theorem number_of_positive_integer_divisors_of_N : Nat.divisors_count N = 462 :=
sorry

end number_of_positive_integer_divisors_of_N_l417_417499


namespace sum_sin_4k_tan_pq_l417_417536

theorem sum_sin_4k_tan_pq (p q : ℕ) (hpq : Int.gcd p q = 1) (h_angle : p < q * 90) :
  (\sum k in Finset.range 40, Real.sin (4 * (k + 1) * Real.pi / 180)) = Real.tan (p * Real.pi / (q * 180)) →
  p + q = 19 :=
sorry

end sum_sin_4k_tan_pq_l417_417536


namespace max_f_l417_417803

theorem max_f (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ x : ℝ, (-1 < x) →  ∀ y : ℝ, (y > -1) → ((1 + y)^a - a*y ≤ 1) :=
sorry

end max_f_l417_417803


namespace average_food_per_week_l417_417843

-- Definitions based on conditions
def food_first_dog := 13
def food_second_dog := 2 * food_first_dog
def food_third_dog := 6
def number_of_dogs := 3

-- Statement of the proof problem
theorem average_food_per_week : 
  (food_first_dog + food_second_dog + food_third_dog) / number_of_dogs = 15 := 
by sorry

end average_food_per_week_l417_417843


namespace eggs_removed_l417_417467

theorem eggs_removed (initial remaining : ℕ) (h1 : initial = 27) (h2 : remaining = 20) : initial - remaining = 7 :=
by
  sorry

end eggs_removed_l417_417467


namespace trapezoid_proof_l417_417598

theorem trapezoid_proof
  (EF GH EH FH : ℝ)
  (p q : ℤ)
  (FG GH : ℝ)
  (R Q : ℝ)
  (H_perp : EH ⊥ FH)
  (Q_mid : 2 * Q = FH)
  (RQ : RQ = 15)
  (EF_parallel : EF ∥ GH)
  (FG_eq : FG = 39)
  (GH_eq : GH = 39)
  (FH_calc : FH = 60)
  : EH = 2 * sqrt q ∧ p + q = 623 :=
by
  have H_perp_condition : EH^2 + FH^2 = EF^2 := sorry
  have FH_calc_condition : FH = 60 := sorry
  have lengths_rel_condition : EF = FH + FG = 78 := sorry
  have p_q_condition : q = 621 ∧ p = 2 := sorry
  use [p_q_condition],
  sorry

end trapezoid_proof_l417_417598


namespace angle_bisector_inequality_l417_417063

variables {A B C D : Type} [metric_space A] [has_dist A]
variables (R : ℝ) (AD AB AC : ℝ)
variable (angle_BAC : ℝ)
variable (circumcircle : metric.sphere A R)

-- Define AD as the angle bisector of ∠BAC
def angle_bisector (AD : ℝ) (angle_BAC : ℝ) : Prop :=
  AD = 2 * R * Real.sin(angle_BAC / 2)

-- Defining the condition that D lies on the circumcircle of triangle ABC
def on_circumcircle (D : A) (circumcircle : metric.sphere A R) : Prop :=
  D ∈ circumcircle

-- The main theorem to be proved
theorem angle_bisector_inequality
  (H1 : angle_bisector AD angle_BAC)
  (H2 : on_circumcircle D circumcircle) :
  2 * AD > AB + AC :=
sorry

end angle_bisector_inequality_l417_417063


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417028

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417028


namespace time_to_pick_72_peas_l417_417850

theorem time_to_pick_72_peas :
  (∀ t : ℕ, t = 56 / 7) → (72 / 8 = 9) :=
begin
  intro t_rate,
  have rate_of_picking := t_rate 8,
  rw rate_of_picking,
  have time := 72 / 8,
  exact time,
end

end time_to_pick_72_peas_l417_417850


namespace find_m_l417_417944

theorem find_m (m : ℝ) : 
  let A := (2, 2 * real.sqrt 2)
  let B := (1 / 2, -real.sqrt 2)
  let M := (-1, m)
  let MA := (3, 2 * real.sqrt 2 - m)
  let MB := (3 / 2, -real.sqrt 2 - m)
  MA.1 * MB.1 + MA.2 * MB.2 = 0 →
  m = real.sqrt 2 / 2 :=
by
  sorry

end find_m_l417_417944


namespace train_crosses_bridge_in_30_seconds_l417_417093

-- Definitions
def length_train : ℝ := 156 -- in meters
def speed_train_kmh : ℝ := 45 -- in kilometers per hour
def length_bridge : ℝ := 219.03 -- in meters

-- Auxiliary definitions and constants
def total_distance : ℝ := length_train + length_bridge
def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600) -- convert km/h to m/s
def time_to_cross : ℝ := total_distance / speed_train_ms

-- Theorem stating that the time to cross the bridge is 30 seconds
theorem train_crosses_bridge_in_30_seconds : time_to_cross = 30 := by
  -- Proof goes here
  sorry

end train_crosses_bridge_in_30_seconds_l417_417093


namespace general_term_correct_l417_417519

open Nat

def sum_seq (n : Nat) : Nat := 2^n + 3

def general_term : Nat → Nat
| 1 => 5
| n + 2 => 2^(n + 1)

theorem general_term_correct (n : Nat) : 
  sum_seq n = ∑ k in range (n + 1), general_term (k + 1) := 
  sorry

end general_term_correct_l417_417519


namespace average_rate_of_discount_l417_417091

theorem average_rate_of_discount :
  let marked_price1 := 240
  let sale_price1 := 120
  let marked_price2 := 360
  let sale_price2 := 270
  let marked_price3 := 480
  let sale_price3 := 384

  let discount_rate := λ marked_price sale_price : ℝ, (marked_price - sale_price) / marked_price * 100
  let first_bag_discount := discount_rate marked_price1 sale_price1
  let second_bag_discount := discount_rate marked_price2 sale_price2
  let third_bag_discount := discount_rate marked_price3 sale_price3

  let average_discount := (first_bag_discount + second_bag_discount + third_bag_discount) / 3

  average_discount = 31.67 := by
  sorry

end average_rate_of_discount_l417_417091


namespace probability_at_least_one_2_on_8_sided_dice_l417_417348

theorem probability_at_least_one_2_on_8_sided_dice :
  (∃ (d1 d2 : Fin 8), d1 = 1 ∨ d2 = 1) → (15 / 64) = (15 / 64) := by
  intro h
  sorry

end probability_at_least_one_2_on_8_sided_dice_l417_417348


namespace simplify_and_evaluate_expr_l417_417298

noncomputable def a : ℝ := Real.sqrt 2 - 2

noncomputable def expr (a : ℝ) : ℝ := (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1))

theorem simplify_and_evaluate_expr :
  expr (Real.sqrt 2 - 2) = Real.sqrt 2 / 2 :=
by sorry

end simplify_and_evaluate_expr_l417_417298


namespace num_exclusive_multiples_4_6_less_151_l417_417204

def numMultiplesExclusive (n : ℕ) (a b : ℕ) : ℕ :=
  let lcm_ab := Nat.lcm a b
  (n-1) / a - (n-1) / lcm_ab + (n-1) / b - (n-1) / lcm_ab

theorem num_exclusive_multiples_4_6_less_151 : 
  numMultiplesExclusive 151 4 6 = 38 := 
by 
  sorry

end num_exclusive_multiples_4_6_less_151_l417_417204


namespace john_leftover_earnings_l417_417258

/-- John's expenditure problem -/
theorem john_leftover_earnings (E : ℝ) (rent_percent dishwasher_percent_less_than_rent : ℝ) :
    rent_percent = 0.40 → 
    dishwasher_percent_less_than_rent = 0.30 →
    let dishwasher_percent := dishwasher_percent_less_than_rent * rent_percent in
    let total_spent_percent := rent_percent + dishwasher_percent in
    let leftover_percent := 1 - total_spent_percent in
    leftover_percent = 0.48 := 
by
  intros h1 h2
  let dishwasher_percent := dishwasher_percent_less_than_rent * rent_percent
  let total_spent_percent := rent_percent + dishwasher_percent
  let leftover_percent := 1 - total_spent_percent
  have h3 : dishwasher_percent = 0.12 := by sorry
  have h4 : total_spent_percent = 0.52 := by sorry
  have h5 : leftover_percent = 1 - 0.52 := by sorry
  exact eq.trans h5 (by norm_num)

end john_leftover_earnings_l417_417258


namespace mandy_reading_books_l417_417280

theorem mandy_reading_books (starting_age certain_multiple current_age : ℕ) :
  starting_age = 6 →
  certain_multiple * starting_age = 30 →                   -- Mandy's age when reading 40-page books
  current_age = certain_multiple * starting_age + 8 →     -- Presently
  4 * (3 * (5 * 8)) = 480 → (30 / 6 = 5) ∧ (6:ℝ ≠ 0) :=  -- Equivalent steps to reach the ratio
begin
  intros h1 h2 h3 h4,
  split,
  {
    calc
      30 / 6 = 5 : by norm_num
  },
  {
    exact (ne_of_lt (by norm_num : (0 : ℝ) < 6)).symm
  },
  sorry -- For further proof details if needed.
end

end mandy_reading_books_l417_417280


namespace first_term_of_polynomial_l417_417657

theorem first_term_of_polynomial (p : ℝ → ℝ) (h : ∀ x, p x = x^2 - x^3) : true :=
by
  have hx0 : p 0 = 0 := by { rw h, ring },
  have hx1 : p 1 = 0 := by { rw h, ring },
  sorry

end first_term_of_polynomial_l417_417657


namespace constant_term_in_expansion_l417_417890

theorem constant_term_in_expansion : 
  (let expr := (sqrt 2 * x - (1 / x ^ 2)) ^ 3 
   in (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → expr = c)) ↔ c = -6 :=
by sorry

end constant_term_in_expansion_l417_417890


namespace find_angle_APB_l417_417592

open Real

noncomputable def angle_APB
    (PA_tangent_SAR: Bool) 
    (PB_tangent_RBT: Bool) 
    (SRT_straight: Bool)
    (arc_AS: ℝ)
    (arc_BT: ℝ) 
    : ℝ := 
  sorry

theorem find_angle_APB 
    (PA_tangent_SAR: Bool) 
    (PB_tangent_RBT: Bool) 
    (SRT_straight: Bool) 
    (arc_AS: ℝ := 45)
    (arc_BT: ℝ := 30) 
    :
    arc_AS = 45 → 
    arc_BT = 30 → 
    angle_APB PA_tangent_SAR PB_tangent_RBT SRT_straight arc_AS arc_BT = 75 :=
  sorry

end find_angle_APB_l417_417592


namespace midpoint_PQ_at_M_l417_417341

open Function

variables {K : Type*} [LinearOrderedField K]

variable (c : Circle K) (A B C D E F M P Q : Point K) (h_midpoint: ∃ M, midpoint A B M)
          (h_chord_CD: ∃ CD, chord CD c) (h_chord_EF: ∃ EF, chord EF c)
          (h_intersection_CF: intersection CF AB P)
          (h_intersection_ED: intersection ED AB Q)

theorem midpoint_PQ_at_M :
  PM = MQ :=
begin
  sorry
end

end midpoint_PQ_at_M_l417_417341


namespace horizon_intersects_sun_l417_417452

-- Given problem's conditions:
variable (circle : Type) (line : Type)
variable (intersect : line → circle → Prop)
variable (horizonSunRelation : circle → line → Prop) -- This is our condition that says the horizon and the sun are in some relation.

-- Define the problem conditions:
axiom small_half_of_sun_visible : ∀ (c : circle) (l : line), horizonSunRelation c l → intersect l c
axiom sun_is_red_but_not_bright : ∀ (c : circle), true  -- This condition is a placeholder as it's not used in the proof logically.

-- The theorem to prove:
theorem horizon_intersects_sun (c : circle) (l : line) (h : horizonSunRelation c l) : intersect l c :=
by
  exact small_half_of_sun_visible c l h

end horizon_intersects_sun_l417_417452


namespace badges_exchange_l417_417759

theorem badges_exchange (Vasya_initial Tolya_initial : ℕ) 
    (h1 : Vasya_initial = Tolya_initial + 5)
    (h2 : Vasya_initial - 0.24 * Vasya_initial + 0.20 * Tolya_initial = Tolya_initial - 0.20 * Tolya_initial + 0.24 * Vasya_initial - 1) 
    : Vasya_initial = 50 ∧ Tolya_initial = 45 :=
by sorry

end badges_exchange_l417_417759


namespace books_sold_correct_l417_417706

-- Define the given conditions
def total_books : ℕ := 9900
def fraction_left : ℚ := 4/6

-- Define how many books are left
def books_left (total: ℕ) (fraction: ℚ) : ℕ := (fraction * total).natAbs

-- Define the number of books sold
def books_sold (total: ℕ) (left: ℕ) : ℕ := total - left

-- State the goal
theorem books_sold_correct : books_sold total_books (books_left total_books fraction_left) = 3300 := 
by
  sorry

end books_sold_correct_l417_417706


namespace shortest_player_height_l417_417724

theorem shortest_player_height :
  ∀ (tallest_height difference : ℝ), 
    tallest_height = 77.75 ∧ difference = 9.5 → 
    tallest_height - difference = 68.25 :=
by
  intros tallest_height difference h
  cases h
  sorry

end shortest_player_height_l417_417724


namespace find_r_l417_417962

theorem find_r (k r : ℝ) 
  (h1 : 7 = k * 3^r) 
  (h2 : 49 = k * 9^r) : 
  r = Real.log 7 / Real.log 3 :=
by
  sorry

end find_r_l417_417962


namespace angle_between_sum_and_difference_perp_condition_l417_417955

-- (1) Definitions of vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 4)

-- (1) Prove the angle between a + b and a - b is 3π / 4
theorem angle_between_sum_and_difference:
  let u := (a.1 + b.1, a.2 + b.2)
  let v := (a.1 - b.1, a.2 - b.2)
  real.angle u v = 3 * real.pi / 4 := by
  sorry

-- (2) Prove that if a ⊥ (a + λb), then λ = -1
theorem perp_condition (λ : ℝ) :
  (a.1 * (a.1 + λ * b.1) + a.2 * (a.2 + λ * b.2) = 0) → λ = -1 := by
  sorry

end angle_between_sum_and_difference_perp_condition_l417_417955


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417013

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417013


namespace parallel_line_through_point_l417_417200

theorem parallel_line_through_point (line1 line2 : ℝ → ℝ → Prop) (P : ℝ × ℝ)
  (h1 : ∀ x1 y1 x2 y2, line1 x1 y1 ↔ line2 x2 y2)
  (h2 : line1 P.1 P.2) :
  ∃ line3, (∀ x y, line3 x y ↔ (α * x + β = y)) ∧ ∀ x1 y1 x2 y2, line1 x1 y1 ∧ line3 x2 y2 → α * x1 + β = y2 ∧ α * x2 + β = y1 :=
sorry

end parallel_line_through_point_l417_417200


namespace senior_ticket_cost_is_13_l417_417653

theorem senior_ticket_cost_is_13
    (adult_ticket_cost : ℕ)
    (child_ticket_cost : ℕ)
    (senior_ticket_cost : ℕ)
    (total_cost : ℕ)
    (num_adults : ℕ)
    (num_children : ℕ)
    (num_senior_citizens : ℕ)
    (age_child1 : ℕ)
    (age_child2 : ℕ)
    (age_child3 : ℕ) :
    adult_ticket_cost = 11 → 
    child_ticket_cost = 8 →
    total_cost = 64 →
    num_adults = 2 →
    num_children = 2 → -- children with discount tickets
    num_senior_citizens = 2 →
    age_child1 = 7 → 
    age_child2 = 10 → 
    age_child3 = 14 → -- this child does not get discount
    senior_ticket_cost * num_senior_citizens = total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) →
    senior_ticket_cost = 13 :=
by
  intros
  sorry

end senior_ticket_cost_is_13_l417_417653


namespace hexagon_max_distance_l417_417202

theorem hexagon_max_distance (a : ℝ) (h : ∀ (x y : ℝ), (x - a)^2 + (y - a)^2 = (1 - a)^2) : 
  let S := {p : ℝ × ℝ | ∃ k : fin 6, p = (cos (k * (2 * Real.pi / 6)), sin (k * (2 * Real.pi / 6)))}
  in ∃ (p1 p2 : ℝ × ℝ), p1 ∈ S ∧ p2 ∈ S ∧ dist p1 p2 = 2 := 
sorry

end hexagon_max_distance_l417_417202


namespace radius_of_set_of_points_l417_417161

theorem radius_of_set_of_points
  (a b k : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : k^2 > 2 * (a^2 / 3) + 2 * (b^2 / 3)) :
  ∃ (c : ℝ), c = sqrt ((k^2 - (2 * a^2 / 3) - (2 * b^2 / 3)) / 3) :=
by
  sorry

end radius_of_set_of_points_l417_417161


namespace proof_problem_l417_417933

variable {a_n : ℕ → ℤ}
variable {b_n : ℕ → ℤ}
variable {c_n : ℕ → ℤ}
variable {T_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Conditions

-- 1. The common difference d of the arithmetic sequence {a_n} is greater than 0
def common_difference_positive (d : ℤ) : Prop :=
  d > 0

-- 2. a_2 and a_5 are the two roots of the equation x^2 - 12x + 27 = 0
def roots_of_quadratic (a2 a5 : ℤ) : Prop :=
  a2^2 - 12 * a2 + 27 = 0 ∧ a5^2 - 12 * a5 + 27 = 0

-- 3. The sum of the first n terms of the sequence {b_n} is S_n, and it is given that S_n = (3 / 2)(b_n - 1)
def sum_of_b_n (S_n b_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = 3/2 * (b_n n - 1)

-- Define the sequences to display further characteristics

-- 1. Find the general formula for the sequences {a_n} and {b_n}
def general_formula_a (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2 * n - 1

def general_formula_b (b : ℕ → ℤ) : Prop :=
  ∀ n, b n = 3 ^ n

-- 2. Check if c_n = a_n * b_n and find the sum T_n
def c_n_equals_a_n_times_b_n (a b : ℕ → ℤ) (c : ℕ → ℤ) : Prop :=
  ∀ n, c n = a n * b n

def sum_T_n (T c : ℕ → ℤ) : Prop :=
  ∀ n, T n = 3 + (n - 1) * 3^(n + 1)

theorem proof_problem 
  (d : ℤ)
  (a2 a5 : ℤ)
  (S_n b_n : ℕ → ℤ)
  (a_n b_n c_n T_n : ℕ → ℤ) :
  common_difference_positive d ∧
  roots_of_quadratic a2 a5 ∧ 
  sum_of_b_n S_n b_n ∧ 
  general_formula_a a_n ∧ 
  general_formula_b b_n ∧ 
  c_n_equals_a_n_times_b_n a_n b_n c_n ∧ 
  sum_T_n T_n c_n :=
sorry

end proof_problem_l417_417933


namespace normal_dist_within_one_std_dev_l417_417147

-- Given a normal distribution function
def normal_dist (mu sigma : ℝ) (x : ℝ) : ℝ := 
  exp (-(x - mu)^2 / (2 * sigma^2)) / (sigma * sqrt (2 * π))

-- Given that X follows a normal distribution N(0,1)
axiom X_normal_0_1 : ∀ a b : ℝ, P (a < X ∧ X ≤ b) = ∫ x in a..b, normal_dist 0 1 x

-- The goal to prove
theorem normal_dist_within_one_std_dev :
  ∫ x in -1..1, normal_dist 0 1 x = 0.6826 :=
sorry

end normal_dist_within_one_std_dev_l417_417147


namespace emily_caught_total_weight_l417_417476

theorem emily_caught_total_weight :
  let trout_count := 4
  let catfish_count := 3
  let bluegills_count := 5
  let sunfish_count := 6
  let bass_count := 2
  let trout_weight := 2.3
  let catfish_weight := 1.5
  let bluegills_weight := 2.5
  let sunfish_weight := 1.75
  let bass_weight := 3.8
  let total_weight := 
    (trout_count * trout_weight) + 
    (catfish_count * catfish_weight) + 
    (bluegills_count * bluegills_weight) + 
    (sunfish_count * sunfish_weight) + 
    (bass_count * bass_weight)
  in total_weight = 44.3 := by
  sorry

end emily_caught_total_weight_l417_417476


namespace sequence_positive_integer_l417_417121

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ ∀ n, 1 ≤ n → a (n + 3) = (a (n + 1))^2 + (a (n + 2))^2 - 2 / a n

theorem sequence_positive_integer (a : ℕ → ℤ) (h_seq : sequence a) :
  ∀ n, 1 ≤ n → a n > 0 := 
by
  sorry

end sequence_positive_integer_l417_417121


namespace company_employees_l417_417694

noncomputable def num_employees
  (N_S : ℕ)
  (uses_vk : ℕ)
  (uses_both : ℕ)
  (uses_only_odno : ℕ)
  (prop_odno : ℚ)
  (no_social : ℕ)
  (total_prop_odno : ℚ) :
  Prop :=
  (no_social = 40) ∧
  (uses_vk = (3 / 4) * N_S) ∧
  (uses_both = (13 / 20) * N_S) ∧
  (uses_only_odno = N_S - uses_vk + uses_both) ∧
  (0.9 * N_S = (5 / 6) * prop_odno) ∧
  (N_S = total_prop_odno - no_social) ∧
  (prop_odno = total_prop_odno)

theorem company_employees :
  ∃ (N : ℕ), ∀ (N_S : ℕ) (uses_vk : ℕ) (uses_both : ℕ) (uses_only_odno : ℕ) (prop_odno : ℚ) (no_social : ℕ) (total_prop_odno : ℚ),
  (num_employees N_S uses_vk uses_both uses_only_odno prop_odno no_social prop_odno) → N = 540 :=
begin
  sorry
end

end company_employees_l417_417694


namespace locus_of_M_l417_417659

-- Definitions for points and collinearity
variables {α : Type} [MetricSpace α] -- Points exist in a metric space
variables (A B C M : α)
variables (k : ℝ)

-- Collinearity and condition for B being between A and C
def collinear (P Q R : α) : Prop := ∃ (l : AffineSubspace ℝ α), P ∈ l ∧ Q ∈ l ∧ R ∈ l
def between (P Q R : α) : Prop := ∃ (d1 d2 : ℝ), d1 > 0 ∧ d2 > 0 ∧ dist P Q = d1 ∧ dist Q R = d2 ∧ dist P R = d1 + d2

-- Given conditions
axiom hA : collinear A B C
axiom hB : between A B C

-- Mathematical equivalent proof statement
theorem locus_of_M (H : cot (angle A M B) + cot (angle B M C) = k) 
  : ∃ d : ℝ, (d = (k * dist A B * dist B C) / (dist A B + dist B C)) ∧
    locus M = { P : α | dist P B = d } :=
sorry

end locus_of_M_l417_417659


namespace exists_parallelogram_marked_cells_l417_417983

theorem exists_parallelogram_marked_cells (n : ℕ) (marked : Finset (Fin n × Fin n)) (h_marked : marked.card = 2 * n) :
  ∃ (a b c d : Fin n × Fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
  ((a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2)) :=
sorry

end exists_parallelogram_marked_cells_l417_417983


namespace total_disks_in_bag_l417_417129

/-- Given that the number of blue disks b, yellow disks y, and green disks g are in the ratio 3:7:8,
    and there are 30 more green disks than blue disks (g = b + 30),
    prove that the total number of disks is 108. -/
theorem total_disks_in_bag (b y g : ℕ) (h1 : 3 * y = 7 * b) (h2 : 8 * y = 7 * g) (h3 : g = b + 30) :
  b + y + g = 108 := by
  sorry

end total_disks_in_bag_l417_417129


namespace definite_integral_equal_value_l417_417388

noncomputable theory

open Real

theorem definite_integral_equal_value :
  ∫ x in 0..π / 3, (sin x / cos x)^2 / (4 + 3 * cos (2 * x)) = 
  sqrt 3 - sqrt 7 * arctan (sqrt (3 / 7)) :=
by
  sorry

end definite_integral_equal_value_l417_417388


namespace cos_arctan_eq_x_find_x_squared_l417_417335

theorem cos_arctan_eq_x_find_x_squared :
  ∃ x : ℝ, 0 < x ∧ cos (arctan x) = x ∧ x^2 = (-1 + sqrt 5) / 2 :=
by
  sorry

end cos_arctan_eq_x_find_x_squared_l417_417335


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417037

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417037


namespace part1_part2_l417_417556

-- Define the vectors and the conditions
variables (λ : ℝ) (α β : ℝ)
def OA := (λ * cos α, λ * sin α)
def OB := (-sin β, cos β)
def OA_dot_OB := λ * (sin (α - β))
def magnitude_OA := abs λ
def magnitude_OB := 1

-- Part 1: Given α - β = π / 6 and λ < 0, prove the angle θ between vectors OA and OB is 2π / 3
theorem part1 (h1 : α - β = π / 6) (hλ : λ < 0) : 
  let θ := acos (-1/2) in θ = 2 * π / 3 := sorry

-- Part 2: Given |AB| ≥ 2|OB| for any real numbers α, β, find the range of λ
theorem part2 (h2 : ∀ (α β : ℝ), 
  let AB := (OB.1 - OA.1, OB.2 - OA.2) in
  (AB.1^2 + AB.2^2) ^ (1/2) ≥ 2)
  : λ ≤ -3 ∨ λ ≥ 3 := sorry

end part1_part2_l417_417556


namespace find_sin_double_angle_l417_417665

-- Define the basic geometric constructions and conditions
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def is_isosceles_right_triangle (a b : ℝ) : Prop :=
  a = b ∧ is_right_triangle a a (a * Real.sqrt 2)

def triangle_perimeter (a b c : ℝ) : ℝ :=
  a + b + c

-- Given conditions
def cond1 : Prop := is_isosceles_right_triangle 2 2
def cond2 : Prop := ∀ AE EC CA, 
  is_right_triangle (2 * Real.sqrt 2) EC CA ∧ 
  triangle_perimeter 2 2 (2 * Real.sqrt 2) = triangle_perimeter (2 * Real.sqrt 2) EC CA

-- Prove the trigonometric identity
theorem find_sin_double_angle (a b c : ℝ) (h1 : is_isosceles_right_triangle 2 2) 
  (h2 : ∀ AE EC CA, is_right_triangle (2 * Real.sqrt 2) EC CA ∧ 
  triangle_perimeter 2 2 (2 * Real.sqrt 2) = triangle_perimeter (2 * Real.sqrt 2) EC CA) :
  Real.sin (2 * (Real.arcsin (1 / Real.sqrt 2))) = Real.sqrt 2 / 2 :=
by
  sorry

end find_sin_double_angle_l417_417665


namespace expression_value_l417_417566

theorem expression_value (x y z : ℕ) (hx : x = 5) (hy : y = 4) (hz : z = 3) :
  ( (1 / (y : ℚ)) + (1 / (z : ℚ))) / (1 / (x : ℚ)) = 35 / 12 := by
  sorry

end expression_value_l417_417566


namespace count_lattice_points_in_square_l417_417892

def is_within_square (x y : ℤ) : Prop := 
  abs x + abs y ≤ 5

def lattice_points_in_square := 
  {p : ℤ × ℤ // is_within_square p.1 p.2}

theorem count_lattice_points_in_square : 
  fintype.card lattice_points_in_square = 61 :=
sorry

end count_lattice_points_in_square_l417_417892


namespace pieces_of_gum_l417_417344

variable (initial_gum total_gum given_gum : ℕ)

theorem pieces_of_gum (h1 : given_gum = 16) (h2 : total_gum = 54) : initial_gum = 38 :=
by
  sorry

end pieces_of_gum_l417_417344


namespace probability_all_appear_before_1_l417_417822

theorem probability_all_appear_before_1 :
  let p := (1 : ℚ) / 720 in
  ∀ n : ℕ, 
  ∀ rolls : list ℕ, 
  ((∀ i ∈ rolls, i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 ∨ i = 6 ∨ i = 1) ∧
   (∃ i, rolls.nth i = some 1) ∧ 
   (list.countp (λ x, x = 1) rolls = 1) ∧
   list.filter (λ x, x ≠ 1) rolls = list.nodup [2, 4, 6, 3, 5]) →
  (1 : ℚ) / (6 ^ n) * 
  ∑ k in finset.range (n-5), 
  (stirling2 k 5) * (5.fact) = p :=
by 
  sorry

end probability_all_appear_before_1_l417_417822


namespace sin_double_angle_l417_417150

theorem sin_double_angle (α : ℝ) (h1 : cos (α + π / 2) = 4 / 5) (h2 : α ∈ Set.Ioo (3 * π / 2) (2 * π)) : 
  sin (2 * α) = -24 / 25 := 
sorry

end sin_double_angle_l417_417150


namespace tangent_intersection_on_line_l417_417621

noncomputable section

open Real Geometry

/-- Let Γ₁ and Γ₂ be two distinct circles with centers O₁ and O₂. An external common tangent touches 
them at A and C respectively, while an internal common tangent intersects them at B and D respectively.
Show that the point of intersection of (AB) and (CD) lies on (O₁ O₂). -/
theorem tangent_intersection_on_line 
  (Γ₁ Γ₂ : Circle) (O₁ O₂ A B C D : Point)
  (h1 : Center(Γ₁) = O₁) (h2 : Center(Γ₂) = O₂) 
  (h3 : external_tangent Γ₁ Γ₂ A C) (h4 : internal_tangent Γ₁ Γ₂ B D) : 
  ∃ (P : Point), (line_through A B).is_on P ∧ (line_through C D).is_on P ∧ (line_through O₁ O₂).is_on P := 
sorry

end tangent_intersection_on_line_l417_417621


namespace sum_with_integers_bounds_property_l417_417166

theorem sum_with_integers_bounds_property (n : ℕ) (a : ℕ → ℝ) 
    (h1 : ∀ i, 1 ≤ i → i ≤ n → 0 ≤ a i )
    (h2 : ∀ i, 2 ≤ i → i ≤ n → a (i - 1) ≤ a i ∧ a i ≤ 2 * a (i - 1)) :
    ∃ (b : ℕ → ℝ), (∀ i, 1 ≤ i → i ≤ n → (b i = 1 ∨ b i = -1)) ∧ 
    (0 ≤ ∑ k in Finset.range (n + 1), b k * a k) ∧ 
    (∑ k in Finset.range (n + 1), b k * a k ≤ a 1) := 
sorry

end sum_with_integers_bounds_property_l417_417166


namespace octahedron_side_length_l417_417407

theorem octahedron_side_length (cube_side_length : ℝ)
  (h_cube : cube_side_length = 2)
  (h_vertices :
     ∃ (Q1 Q2 Q3 Q4 Q1' Q2' Q3' Q4' : ℝ × ℝ × ℝ),
     Q1 = (0, 0, 0) ∧ Q1' = (2, 2, 2) ∧
     Q2 = (2, 0, 0) ∧ Q3 = (0, 2, 0) ∧ Q4 = (0, 0, 2) ∧
     ∀ i, 1 ≤ i ∧ i ≤ 4 →
     (Q1, Q2, Q3, Q4, Q1', Q2', Q3', Q4')) :
  ∃ (side_length : ℝ),
  side_length = sqrt(8 / 9) := sorry

end octahedron_side_length_l417_417407


namespace min_distance_origin_to_intersections_l417_417142

theorem min_distance_origin_to_intersections (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hline : (1 : ℝ)/a + 4/b = 1) :
  |(0 : ℝ) - a| + |(0 : ℝ) - b| = 9 :=
sorry

end min_distance_origin_to_intersections_l417_417142


namespace problem_constant_term_problem_sum_of_coefficients_l417_417186

theorem problem_constant_term :
  ∃ n, 2^n = 512 ∧ 
       (∃ r, n = 2 * r ∧ r = 3 ∧ (2^r * Nat.choose 9 r = 672)) :=
by
  sorry

theorem problem_sum_of_coefficients :
  ∃ n, 2^n = 512 ∧ 
       (∀ x : ℝ, (sqrt x + 2 / x)^n = (1 + 2) ^ 9) :=
by
  sorry

end problem_constant_term_problem_sum_of_coefficients_l417_417186


namespace paths_to_spell_math_l417_417229

def grid : List (List Char) := [
    ['X', 'A', 'X'],
    ['A', 'M', 'T'],
    ['X', 'T', 'H']
]

def is_adjacent (pos1 pos2 : (Int × Int)) : Bool :=
    match pos1, pos2 with
    | (x1, y1), (x2, y2) =>
        (abs (x1 - x2) = 1 ∧ y1 = y2) ∨
        (abs (y1 - y2) = 1 ∧ x1 = x2)

def find_char_positions (g : List (List Char)) (c : Char) : List (Int × Int) :=
    g.foldl (fun acc row y => 
        row.foldl (fun acc' elem x => if elem = c then (x, y) :: acc' else acc') acc) []

theorem paths_to_spell_math : 
    (let start_positions := find_char_positions grid 'M';
     let a_positions := find_char_positions grid 'A';
     let t_positions := find_char_positions grid 'T';
     let h_positions := find_char_positions grid 'H' in
    (∃ m_pos ∈ start_positions, ∃ a_pos ∈ a_positions, ∃ t_pos ∈ t_positions, ∃ h_pos ∈ h_positions,
     is_adjacent m_pos a_pos ∧ is_adjacent a_pos t_pos ∧ is_adjacent t_pos h_pos) = 2) :=
sorry

end paths_to_spell_math_l417_417229


namespace find_special_primes_l417_417483

theorem find_special_primes :
  ∃ p : ℕ, prime p ∧
    (∃! (x y : ℕ), 0 ≤ x ∧ x ≤ p ∧ 0 ≤ y ∧ y ≤ p ∧ y^2 ≡ x^3 - x [MOD p]) ↔ 
  (p = 2 ∨ (p % 4 = 3)) :=
sorry

end find_special_primes_l417_417483


namespace sum_of_slope_and_intercept_half_area_bisector_l417_417754

-- Definitions of points A, B, and C
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (9, 0)

-- Midpoint M of line segment AC
def M : ℝ × ℝ := (4.5, 5)

-- Slope of the line through B and M
def slope_BM := (M.2 - B.2) / (M.1 - B.1)

-- y-intercept of the line passing through B and M, assuming y = slope_BM * x + c
def y_intercept_BM := B.2 - slope_BM * B.1

-- Sum of the slope and y-intercept
def sum_slope_y_intercept := slope_BM + y_intercept_BM

-- Now we state the theorem we want to prove.
theorem sum_of_slope_and_intercept_half_area_bisector :
  sum_slope_y_intercept = -20 / 3 :=
by
  sorry

end sum_of_slope_and_intercept_half_area_bisector_l417_417754


namespace devin_biked_more_l417_417233

def cyra_distance := 77
def cyra_time := 7
def cyra_speed := cyra_distance / cyra_time
def devin_speed := cyra_speed + 3
def marathon_time := 7
def devin_distance := devin_speed * marathon_time
def distance_difference := devin_distance - cyra_distance

theorem devin_biked_more : distance_difference = 21 := 
  by
    sorry

end devin_biked_more_l417_417233


namespace minimum_value_l417_417994

noncomputable theory

def AB : ℝ := 2
def AC_dot_BD (AC BD : E) : ℝ := -3
variables (a b c : ℝ) [InnerProductSpace ℝ E]

def AD : ℝ := a
def BC : ℝ := b
def CD : ℝ := c

theorem minimum_value (ac bd : E) :
  (ac ∙ bd = -3) ∧ (CD = (a^2 + b^2 + 2).sqrt) → 
  ( ∀ (a b c : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c,
  ∃ (min_val : ℝ), min_val = 2 ∧ 
  ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → 
  (a*x = AB / x) ∧ (b*y = AC_dot_BD ac bd) → 
  (c*y = (a^2 + b^2 + 2).sqrt) → 
  (min_val = min (a^2 + b^2 + 2) / (a*b + 1))) :=
sorry

end minimum_value_l417_417994


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417023

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417023


namespace probability_X_eq_2_l417_417396

theorem probability_X_eq_2 :
  let n_r := 9
  let n_w := 3
  let total_balls := n_r + n_w
  let n_draws := 4
  let X_eq := 2
  let total_ways := (Finset.choose total_balls n_draws)
  let red_ways := (Finset.choose n_r X_eq)
  let white_ways := (Finset.choose n_w (n_draws - X_eq))
  (red_ways * white_ways) / total_ways = (Finset.choose 9 2 * Finset.choose 3 2) / Finset.choose 12 4 := by
    sorry

end probability_X_eq_2_l417_417396


namespace range_combined_set_l417_417383

/-- 
  Prove that the range of the new set formed by combining set X and set Y 
  is 93, given:
  - X is the set of all two-digit primes
  - Y is the set of all positive multiples of 4 less than 100
-/
theorem range_combined_set : 
  let X := {p ∈ Finset.Icc 10 99 | Nat.Prime p}
  let Y := {k * 4 | k ∈ Finset.Icc 1 24}
  let Z := X ∪ Y
  Finset.range Z.card = 93 := 
by
  sorry

end range_combined_set_l417_417383


namespace switches_connections_l417_417742

theorem switches_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 :=
by
  sorry

end switches_connections_l417_417742


namespace domain_of_f_x_squared_l417_417312

theorem domain_of_f_x_squared (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = x) →
  (∀ x, -real.sqrt 2 ≤ x ∧ x ≤ real.sqrt 2 → ∃ y, f (x^2) = y) := by
  intro h
  sorry

end domain_of_f_x_squared_l417_417312


namespace bananas_per_monkey_l417_417824

-- Define the given conditions
def total_monkeys : ℕ := 12
def piles_with_9hands : ℕ := 6
def hands_per_pile_9hands : ℕ := 9
def bananas_per_hand_9hands : ℕ := 14
def piles_with_12hands : ℕ := 4
def hands_per_pile_12hands : ℕ := 12
def bananas_per_hand_12hands : ℕ := 9

-- Calculate the total number of bananas from each type of pile
def total_bananas_9hands : ℕ := piles_with_9hands * hands_per_pile_9hands * bananas_per_hand_9hands
def total_bananas_12hands : ℕ := piles_with_12hands * hands_per_pile_12hands * bananas_per_hand_12hands

-- Sum the total number of bananas
def total_bananas : ℕ := total_bananas_9hands + total_bananas_12hands

-- Prove that each monkey gets 99 bananas
theorem bananas_per_monkey : total_bananas / total_monkeys = 99 := by
  sorry

end bananas_per_monkey_l417_417824


namespace symmetric_points_l417_417988

def symmetric_about_y_axis (A B : ℝ × ℝ) : Prop :=
  A.2 = B.2 ∧ A.1 = -B.1

theorem symmetric_points (m n : ℝ) (h₁ : symmetric_about_y_axis (m, 3) (4, n)) : 
  (m + n) ^ 2015 = -1 :=
by 
  have h₂ : n = 3 := by sorry,
  have h₃ : m = -4 := by sorry,
  rw [h₂, h₃],
  simp,
  exact dec_trivial

end symmetric_points_l417_417988


namespace infinite_prime_factors_of_sequence_l417_417498

def sequence_of_positive_integers (a : ℕ → ℕ) :=
  ∀ i, a i > 0

def distinct_divisors_property (a : ℕ → ℕ) :=
  ∀ i j : ℕ, i ≠ j → ¬ (a i ∣ a j)

def prime_factors_of_sequence (a : ℕ → ℕ) : set ℕ :=
  { p | ∃ i, Nat.Prime p ∧ p ∣ a i }

theorem infinite_prime_factors_of_sequence (a : ℕ → ℕ)
    (hseq : sequence_of_positive_integers a)
    (hdiv : distinct_divisors_property a) :
    set.infinite (prime_factors_of_sequence a) := 
  sorry

end infinite_prime_factors_of_sequence_l417_417498


namespace vector_CD_l417_417558

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b c : V)
variable (h1 : B - A = a)
variable (h2 : B - C = b)
variable (h3 : D - A = c)

theorem vector_CD :
  D - C = -a + b + c :=
by
  -- Proof omitted
  sorry

end vector_CD_l417_417558


namespace motion_equation_l417_417302

-- Define acceleration function
def a (t : ℝ) : ℝ := 6 * t - 2

-- Define initial conditions
def v0 : ℝ := 1 -- velocity at t = 0
def s0 : ℝ := 0 -- distance at t = 0

-- Theorem statement for motion equation
theorem motion_equation : 
  (∃ v s : ℝ → ℝ, 
    (∀ t, v t = ∫ x in 0..t, a x + v0) ∧ 
    (∀ t, s t = ∫ x in 0..t, v x + s0) ∧ 
    (∀ t, s t = t^3 - t^2 + t)) :=
  by 
    sorry

end motion_equation_l417_417302


namespace shortest_player_height_l417_417723

-- let h_tall be the height of the tallest player
-- let h_short be the height of the shortest player
-- let diff be the height difference between the tallest and the shortest player

variable (h_tall h_short diff : ℝ)

-- conditions given in the problem
axiom tall_player_height : h_tall = 77.75
axiom height_difference : diff = 9.5
axiom height_relationship : h_tall = h_short + diff

-- the statement we need to prove
theorem shortest_player_height : h_short = 68.25 := by
  sorry

end shortest_player_height_l417_417723


namespace monotonic_decreasing_interval_of_my_function_l417_417711

def is_monotonic_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f y ≤ f x

noncomputable def my_function : ℝ → ℝ := λ x, 2 + 3 * Real.sin x

theorem monotonic_decreasing_interval_of_my_function :
  ∃ (k : ℤ), is_monotonic_decreasing my_function (set.Icc (↑(Real.pi) / 2 + 2 * ↑k * Real.pi) (3 * (Real.pi) / 2 + 2 * ↑k * Real.pi)) :=
sorry

end monotonic_decreasing_interval_of_my_function_l417_417711


namespace find_tan_alpha_l417_417956

variables {α : Type*} [real  : Real α]

def vector_parallel (a b : Real × Real) : Prop :=
  ∃ k : Real, a = (k * b.1, k * b.2)

theorem find_tan_alpha (α : Real) (ha : vector_parallel (3, 4) (Real.sin α, Real.cos α)) : Real.tan α = 3 / 4 :=
by
  sorry

end find_tan_alpha_l417_417956


namespace correlation_index_l417_417597

-- Define the conditions given in the problem
def height_explains_weight_variation : Prop :=
  ∃ R : ℝ, R^2 = 0.64

-- State the main conjecture (actual proof omitted for simplicity)
theorem correlation_index (R : ℝ) (h : height_explains_weight_variation) : R^2 = 0.64 := by
  sorry

end correlation_index_l417_417597


namespace sequence_number_pair_l417_417938

theorem sequence_number_pair (n m : ℕ) (h : m ≤ n) : (m, n - m + 1) = (m, n - m + 1) :=
by sorry

end sequence_number_pair_l417_417938


namespace minimize_f_minimize_f_exact_l417_417048

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 14 * x - 20

-- State the theorem that x = -7 minimizes the function f(x)
theorem minimize_f : ∀ x : ℝ, f x ≥ f (-7) :=
by
  intro x
  unfold f
  sorry

-- An alternative statement could include the exact condition for the minimum value
theorem minimize_f_exact : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ x = -7 :=
by
  use -7
  intro y
  unfold f
  sorry

end minimize_f_minimize_f_exact_l417_417048


namespace four_statements_are_false_l417_417365

-- We define the statements as boolean variables
def statement_1 := "On this card, exactly one statement is false."
def statement_2 := "On this card, exactly two statements are false."
def statement_3 := "On this card, exactly three statements are false."
def statement_4 := "On this card, exactly four statements are false."
def statement_5 := "On this card, exactly five statements are false."

theorem four_statements_are_false :
  ∃ n : ℕ, (n = 4) ∧
  (¬ (statement_1 ∧ statement_2 ∧ statement_3 ∧ statement_4 ∧ statement_5)) :=
by
  sorry

end four_statements_are_false_l417_417365


namespace distribution_plans_of_teachers_l417_417106

/-- Given 4 unique teachers and 3 unique schools, show that the number of ways to distribute the 
teachers such that each school has at least one teacher is 36. --/

theorem distribution_plans_of_teachers : 
  let teachers : Finset ℕ := {1, 2, 3, 4},
      schools : Finset ℕ := {1, 2, 3}
  in 
  ∃ (distribution : Finset (Finset ℕ × Finset ℕ × Finset ℕ)),
    (∀ (d ∈ distribution), d.1 ∪ d.2 ∪ d.3 = teachers) ∧
    (∀ (d ∈ distribution), d.1 ∩ d.2 = ∅ ∧ d.1 ∩ d.3 = ∅ ∧ d.2 ∩ d.3 = ∅) ∧
    (∀ (d ∈ distribution), d.1 ≠ ∅ ∧ d.2 ≠ ∅ ∧ d.3 ≠ ∅) ∧
    (distribution.card = 36) :=
by
  sorry

end distribution_plans_of_teachers_l417_417106


namespace total_number_of_applications_l417_417117

def in_state_apps := 200
def out_state_apps := 2 * in_state_apps
def total_apps := in_state_apps + out_state_apps

theorem total_number_of_applications : total_apps = 600 := by
  sorry

end total_number_of_applications_l417_417117


namespace division_result_l417_417393

theorem division_result : (108 * 3 - (108 + 92)) / (92 * 7 - (45 * 3)) = 124 / 509 := 
by
  sorry

end division_result_l417_417393


namespace donna_smallest_n_l417_417473

theorem donna_smallest_n (n : ℕ) : 15 * n - 1 % 6 = 0 ↔ n % 6 = 5 := sorry

end donna_smallest_n_l417_417473


namespace probability_alternating_colors_l417_417398

theorem probability_alternating_colors :
  let balls := ([true, true, true, true, true, true, false, false, false, false, false] : List Bool) in
  let initial_prob := 6 / 11 in
  let second_prob := 5 / 10 in
  let third_prob := 4 / 9 in
  let fourth_prob := 4 / 8 in
  (initial_prob * second_prob * third_prob * fourth_prob) = 2 / 33 :=
by
  -- Step 1: Calculate individual probabilities
  have h_initial_prob : initial_prob = 6 / 11 := rfl
  have h_second_prob : second_prob = 5 / 10 := by norm_num
  have h_third_prob : third_prob = 4 / 9 := rfl
  have h_fourth_prob : fourth_prob = 4 / 8 := by norm_num
  
  -- Step 2: Calculate product of probabilities
  let total_prob := initial_prob * second_prob * third_prob * fourth_prob
  have h_total_prob : total_prob = 6 / 11 * 5 / 10 * 4 / 9 * 4 / 8 := rfl

  -- Step 3: Verify final probability
  have final_result : total_prob = 2 / 33
  {
    -- Combine and simplify the probabilities
    calc
      total_prob = (6 * 5 * 4 * 4) / (11 * 10 * 9 * 8) : by
        { rw [h_initial_prob, h_second_prob, h_third_prob, h_fourth_prob], norm_num }
      ... = 2 / 33 : by norm_num
  }

  -- Proof completion
  exact final_result

end probability_alternating_colors_l417_417398


namespace translation_complex_number_l417_417431

theorem translation_complex_number :
  let w := -2 + 4*complex.I - (1 + 3*complex.I)
  (3 + 7*complex.I) + w = 8*complex.I :=
by
  let w := -2 + 4*complex.I - (1 + 3*complex.I)
  show (3 + 7*complex.I) + w = 8*complex.I
  sorry

end translation_complex_number_l417_417431


namespace optionC_is_correct_l417_417052

theorem optionC_is_correct (x : ℝ) : (x^2)^3 = x^6 :=
by sorry

end optionC_is_correct_l417_417052


namespace percentage_decrease_in_y_when_x_doubles_l417_417679

variable {k x y : ℝ}
variable (h_pos_x : 0 < x) (h_pos_y : 0 < y)
variable (inverse_proportional : x * y = k)

theorem percentage_decrease_in_y_when_x_doubles :
  (x' = 2 * x) →
  (y' = y / 2) →
  (100 * (y - y') / y) = 50 :=
by
  intro h1 h2
  simp [h1, h2]
  sorry

end percentage_decrease_in_y_when_x_doubles_l417_417679


namespace sum_of_integers_l417_417310

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 15) (h3 : x * y = 56) : x + y = Real.sqrt 449 :=
by
  sorry

end sum_of_integers_l417_417310


namespace particle_final_position_l417_417419

noncomputable def particle_transform (z : ℂ) : ℂ :=
  2 * (complex.exp (complex.I * real.pi / 6)) * z + 8

noncomputable def particle_position_after_n_moves (n : ℕ) : ℂ :=
  nat.rec_on n (3 : ℂ) (λ _ z_n, particle_transform z_n)

theorem particle_final_position :
  particle_position_after_n_moves 120 =
  2^122 - 8 + 4 * real.sqrt 3 * (2^120 - 1) :=
sorry

end particle_final_position_l417_417419


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417030

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417030


namespace number_of_sheep_l417_417333

theorem number_of_sheep (legs animals : ℕ) (h1 : legs = 60) (h2 : animals = 20)
  (chickens sheep : ℕ) (hc : chickens + sheep = animals) (hl : 2 * chickens + 4 * sheep = legs) :
  sheep = 10 :=
sorry

end number_of_sheep_l417_417333


namespace f_is_odd_function_f_is_increasing_f_range_of_m_l417_417862

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_interval : ∀ x, x ∈ Ioo (-1 : ℝ) 1 → f x ∈ ℝ
axiom f_equation : ∀ x y, x ∈ Ioo (-1 : ℝ) 1 → y ∈ Ioo (-1 : ℝ) 1 → f ((x + y) / (1 + x * y)) = f x + f y
axiom f_positive : ∀ x, 0 < x → x < 1 → 0 < f x
axiom f_half : f (1 / 2) = 1 / 2

-- (1) Prove that f is an odd function on the interval (-1,1).
theorem f_is_odd_function : ∀ x, x ∈ Ioo (-1 : ℝ) 1 → f (-x) = -f x := sorry

-- (2) Prove that f is an increasing function on the interval (-1,1).
theorem f_is_increasing : ∀ x₁ x₂, x₁ ∈ Ioo (-1 : ℝ) 1 → x₂ ∈ Ioo (-1 : ℝ) 1 → x₁ < x₂ → f x₁ < f x₂ := sorry

-- (3) Prove the range of m given the condition.
theorem f_range_of_m (m : ℝ) : (∀ x ∈ Icc (-4/5) (4/5), ∀ a ∈ Icc (-1 : ℝ) 1, f x < m^2 - 2 * a * m + 1) → m < -2 ∨ 2 < m := sorry

end f_is_odd_function_f_is_increasing_f_range_of_m_l417_417862


namespace union_sets_inter_sets_complement_union_l417_417948

-- Definitions of sets A and B
def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | x < 3 }

-- Question 1: Prove A ∪ B = ℝ
theorem union_sets : A ∪ B = { x : ℝ | true } := by
  sorry

-- Question 2: Prove A ∩ B = { x | 2 ≤ x ∧ x < 3 }
theorem inter_sets : A ∩ B = { x : ℝ | 2 ≤ x ∧ x < 3 } := by
  sorry

-- Complement of sets A and B
def C_R (S : set ℝ) := { x : ℝ | x ∉ S }
def C_R_A := C_R A
def C_R_B := C_R B

-- Question 3: Prove (C_R A) ∪ (C_R B) = { x | x < 2 ∨ x ≥ 3 }
theorem complement_union : C_R_A ∪ C_R_B = { x : ℝ | x < 2 ∨ x ≥ 3 } := by
  sorry

end union_sets_inter_sets_complement_union_l417_417948


namespace false_statement_quadrilateral_diagonals_l417_417786

theorem false_statement_quadrilateral_diagonals :
  ¬ (∀ (Q : Type) [quad : quadrilateral Q] (d1 d2 : diagonal Q), 
      perpendicular d1 d2 ∧ equal_length d1 d2 → is_square Q) := sorry

end false_statement_quadrilateral_diagonals_l417_417786


namespace bm_bisects_angle_tbc_l417_417616

theorem bm_bisects_angle_tbc
  (ABC : Type)
  [linear_ordered_field ABC]
  (A B C D K M N T : ABC)
  (h_triangle : ∃ (a b c : ABC), a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_acute : is_acute_triangle A B C)
  (h_ab_gt_ac : dist A B > dist A C)
  (h_D : foot_point A B C D)
  (h_K : intersection AD (angle_bisector B) K)
  (h_M : foot_perpendicular B C K M)
  (h_N : intersection BM AK N)
  (h_T : intersection AC (//(DM)) (line_through N) T) :
  internal_bisector (angle T B C) BM :=
sorry

end bm_bisects_angle_tbc_l417_417616


namespace average_marks_l417_417684

-- Definitions
def Tatuya_score (Ivanna_score : ℕ) : ℕ := 2 * Ivanna_score
def Ivanna_score (Dorothy_score : ℕ) : ℕ := (3 * Dorothy_score) / 5
def Dorothy_score : ℕ := 90

-- Theorem statement
theorem average_marks :
  let Dorothy_score := Dorothy_score in
  let Ivanna_score := Ivanna_score Dorothy_score in
  let Tatuya_score := Tatuya_score Ivanna_score in
  (Dorothy_score + Ivanna_score + Tatuya_score) / 3 = 84 :=
by 
  -- Proof goes here
  sorry

end average_marks_l417_417684


namespace number_of_molecules_correct_l417_417387

-- Define Avogadro's number
def avogadros_number : ℝ := 6.022 * 10^23

-- Define the given number of molecules
def given_number_of_molecules : ℝ := 3 * 10^26

-- State the problem
theorem number_of_molecules_correct :
  (number_of_molecules = given_number_of_molecules) :=
by
  sorry

end number_of_molecules_correct_l417_417387


namespace vasya_tolya_badges_l417_417766

theorem vasya_tolya_badges (x y : ℤ)
    (h1 : y = x + 5) -- Vasya initially had 5 more badges than Tolya
    (h2 : (y - (6 * (y / 25) / 25) + (4 * x) / 25) = (x - (4 * x) / 5 + 6 * (y / 25) / 5 - 1)) : -- equation balancing after exchange
    x = 45 ∧ y = 50 := 
sorry

end vasya_tolya_badges_l417_417766


namespace value_of_function_at_2_l417_417109

theorem value_of_function_at_2 (q : ℝ → ℝ) : q 2 = 5 :=
by
  -- Condition: The point (2, 5) lies on the graph of q
  have point_on_graph : q 2 = 5 := sorry
  exact point_on_graph

end value_of_function_at_2_l417_417109


namespace find_k_l417_417859

theorem find_k (k : ℝ) (α β : ℝ) 
  (h1 : α + β = -k) 
  (h2 : α * β = 12) 
  (h3 : α + 7 + β + 7 = k) : 
  k = -7 :=
sorry

end find_k_l417_417859


namespace Diana_additional_video_game_time_l417_417871

theorem Diana_additional_video_game_time 
    (original_reward_per_hour : ℕ := 30)
    (raise_percentage : ℕ := 20)
    (hours_read : ℕ := 12)
    (minutes_per_hour : ℕ := 60) :
    let raise := (raise_percentage * original_reward_per_hour) / 100
    let new_reward_per_hour := original_reward_per_hour + raise
    let total_time_after_raise := new_reward_per_hour * hours_read
    let total_time_before_raise := original_reward_per_hour * hours_read
    let additional_minutes := total_time_after_raise - total_time_before_raise
    additional_minutes = 72 :=
by sorry

end Diana_additional_video_game_time_l417_417871


namespace concurrency_of_A_I1_B_I2_C_I3_l417_417714

theorem concurrency_of_A_I1_B_I2_C_I3
  (A B C D E F H I1 I2 I3 : Point)
  (h_orthocenter : orthocenter H A B C)
  (h_altitudes : is_altitude AD ∧ is_altitude BE ∧ is_altitude CF)
  (h_incenters : incenter I1 E H F ∧ incenter I2 F H D ∧ incenter I3 D H E) :
  concurrent (line_through A I1) (line_through B I2) (line_through C I3) :=
begin
  sorry
end

end concurrency_of_A_I1_B_I2_C_I3_l417_417714


namespace polygon_sides_l417_417221

theorem polygon_sides (sum_of_angles : ℕ) (h : sum_of_angles = 1440) :
  ∃ n : ℕ, (n - 2) * 180 = 1440 ∧ n = 10 :=
by
  use 10
  split
  . exact h
  . refl

end polygon_sides_l417_417221


namespace semicircle_radius_l417_417716

noncomputable def pi_approx := 3.14159

def perimeter (r : ℝ) : ℝ := pi * r + 2 * r

theorem semicircle_radius (p : ℝ) (h : p = 32.90619298297467) : 
  ∃ r : ℝ, r = 32.90619298297467 / (pi_approx + 2) := by
  sorry

end semicircle_radius_l417_417716


namespace total_rain_duration_l417_417999

theorem total_rain_duration:
  let first_day_duration := 10
  let second_day_duration := first_day_duration + 2
  let third_day_duration := 2 * second_day_duration
  first_day_duration + second_day_duration + third_day_duration = 46 :=
by
  sorry

end total_rain_duration_l417_417999


namespace exists_solutions_for_equation_l417_417600

theorem exists_solutions_for_equation (a : ℝ) :
  (∃ x : ℝ, (x - a)^2 = (if x > 0 then 1 else -1) + a + 1) ↔ 
  (a ∈ set.Icc (-1 : ℝ) 0 ∪ set.Icc (1 : ℝ) 2 ∨ (0 : ℝ) < a ∧ a < 1 ∨ a > 2) :=
by sorry

end exists_solutions_for_equation_l417_417600


namespace mod_remainder_l417_417893

-- Define the product of all odd numbers from 1 to 2005
noncomputable def N : ℕ := ∏ i in multiset.filter (odd) (multiset.range 2006), i

-- State the main theorem
theorem mod_remainder (hN : N = 1 * 3 * 5 * ... * 2003 * 2005):
  N % 1000 = 375 :=
sorry

end mod_remainder_l417_417893


namespace fixed_point_coordinates_l417_417500

def passes_through_fixed_point (t x y : ℝ) : Prop :=
  (t + 1) * x - (2 * t + 5) * y - 6 = 0

theorem fixed_point_coordinates :
  ∃ (x y : ℝ), (∀ t : ℝ, passes_through_fixed_point t x y) ∧ x = -4 ∧ y = -2 :=
by
  use [-4, -2]
  simp [passes_through_fixed_point]
  intro t
  linarith

end fixed_point_coordinates_l417_417500


namespace pencils_undefined_l417_417733

-- Definitions for the conditions given in the problem
def initial_crayons : Nat := 41
def added_crayons : Nat := 12
def total_crayons : Nat := 53

-- Theorem stating the problem's required proof
theorem pencils_undefined (initial_crayons : Nat) (added_crayons : Nat) (total_crayons : Nat) : Prop :=
  initial_crayons = 41 ∧ added_crayons = 12 ∧ total_crayons = 53 → 
  ∃ (pencils : Nat), true
-- Since the number of pencils is unknown and no direct information is given, we represent it as an existential statement that pencils exist in some quantity, but we cannot determine their exact number based on given information.

end pencils_undefined_l417_417733


namespace parallel_to_l3_through_P_perpendicular_to_l3_through_P_l417_417924

-- Define the lines l1, l2, and l3
def l1 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def l3 (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the intersection point P
def P := (1, 1)

-- Define the parallel line equation to l3 passing through P
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Define the perpendicular line equation to l3 passing through P
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Prove the parallel line through P is 2x + y - 3 = 0
theorem parallel_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (parallel_line 1 1) := 
by 
  sorry

-- Prove the perpendicular line through P is x - 2y + 1 = 0
theorem perpendicular_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (perpendicular_line 1 1) := 
by 
  sorry

end parallel_to_l3_through_P_perpendicular_to_l3_through_P_l417_417924


namespace find_g_26_l417_417316

-- Defining the function g satisfying the given conditions
axiom g : ℝ → ℝ
axiom g_property : ∀ x, g(x + g(x)) = 5 * g(x)
axiom g_at_1 : g 1 = 5

theorem find_g_26 : g 26 = 125 := 
by {
  sorry
}

end find_g_26_l417_417316


namespace option_a_option_b_option_c_l417_417564

variable {a b : ℝ}

theorem option_a (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  ab ≤ 1 := sorry

theorem option_b (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  sqrt a + sqrt b ≤ 2 := sorry

theorem option_c (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) : 
  a^2 + b^2 ≥ 2 := sorry

end option_a_option_b_option_c_l417_417564


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417027

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417027


namespace consecutive_tree_distance_l417_417808

theorem consecutive_tree_distance (yard_length : ℕ) (num_trees : ℕ) (distance : ℚ)
  (h1 : yard_length = 520) 
  (h2 : num_trees = 40) :
  distance = yard_length / (num_trees - 1) :=
by
  -- Proof steps would go here
  sorry

end consecutive_tree_distance_l417_417808


namespace remainder_7_pow_135_mod_8_l417_417372

theorem remainder_7_pow_135_mod_8 : 7^135 % 8 = 7 :=
calc
  7^135 % 8 = (7^2)^67 * 7 % 8 : by sorry
        ... = 1^67 * 7 % 8 : by sorry
        ... = 1 * 7 % 8 : by sorry
        ... = 7 : by sorry

end remainder_7_pow_135_mod_8_l417_417372


namespace coeff_x2y2_in_expansion_l417_417865

theorem coeff_x2y2_in_expansion (x y : ℝ) :
  ((1+x)^3 * (1+y)^4).coeff (2, 2) = 18 := 
sorry

end coeff_x2y2_in_expansion_l417_417865


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417031

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417031


namespace butterfat_mixture_l417_417380

theorem butterfat_mixture (x : ℝ) :
  (0.10 * x + 0.30 * 8 = 0.20 * (x + 8)) → x = 8 :=
by
  intro h
  sorry

end butterfat_mixture_l417_417380


namespace functional_solutions_l417_417480

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x * f y + y * f x = (x + y) * (f x) * (f y)

theorem functional_solutions (f : ℝ → ℝ) (h : functional_equation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∃ (a : ℝ), ∀ x : ℝ, (x ≠ 0 → f x = 1) ∧ (x = 0 → f x = a)) :=
  sorry

end functional_solutions_l417_417480


namespace ways_to_divide_candies_l417_417789

def count_ways_to_divide_8_candies : Prop :=
  ∃ (Yuna Hoseok : ℕ), Yuna + Hoseok = 8 ∧ Yuna ≥ 1 ∧ Hoseok ≥ 1

theorem ways_to_divide_candies : (finset.card {p : ℕ × ℕ | p.1 + p.2 = 8 ∧ p.1 ≥ 1 ∧ p.2 ≥ 1} = 7) :=
by {
  sorry
}

end ways_to_divide_candies_l417_417789


namespace no_hamiltonian_circuit_rhombic_dodecahedron_l417_417410

-- We define the graph of a rhombic dodecahedron.
def rhombic_dodecahedron : SimpleGraph ℕ := sorry

-- We state the theorem: the rhombic dodecahedron has no Hamiltonian circuit.
theorem no_hamiltonian_circuit_rhombic_dodecahedron :
  ¬(∃ p : List ℕ, rhombic_dodecahedron.IsHamiltonianCircuit p) :=
sorry

end no_hamiltonian_circuit_rhombic_dodecahedron_l417_417410


namespace ivan_sergeyevich_profit_l417_417612

def revenue_from_meat (meat_sold price_per_kg : ℝ) : ℝ :=
  meat_sold * price_per_kg

def revenue_from_eggs (eggs_sold price_per_dozen : ℝ) : ℝ :=
  eggs_sold * (price_per_dozen / 12)

def total_revenue (meat_revenue egg_revenue : ℝ) : ℝ :=
  meat_revenue + egg_revenue

def profit (total_revenue expenses : ℝ) : ℝ :=
  total_revenue - expenses

-- Given conditions
def meat_sold := 100
def price_per_kg := 500
def eggs_sold := 20000
def price_per_dozen := 50
def expenses := 100000

theorem ivan_sergeyevich_profit : 
  profit (total_revenue (revenue_from_meat meat_sold price_per_kg) (revenue_from_eggs eggs_sold price_per_dozen)) expenses = 50000 :=
by sorry

end ivan_sergeyevich_profit_l417_417612


namespace scientific_notation_of_0_0000205_l417_417479

noncomputable def scientific_notation (n : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_0_0000205 :
  scientific_notation 0.0000205 = (2.05, -5) :=
sorry

end scientific_notation_of_0_0000205_l417_417479


namespace remainder_of_3_pow_21_mod_11_l417_417779

theorem remainder_of_3_pow_21_mod_11 : (3^21 % 11) = 3 := 
by {
  sorry
}

end remainder_of_3_pow_21_mod_11_l417_417779


namespace circle_area_in_square_centimeters_l417_417371

theorem circle_area_in_square_centimeters (d_meters : ℤ) (h : d_meters = 8) :
  ∃ (A : ℤ), A = 160000 * Real.pi ∧ 
  A = π * (d_meters / 2) ^ 2 * 10000 :=
by
  sorry

end circle_area_in_square_centimeters_l417_417371


namespace arithmetic_mean_of_integers_minus5_to_5_l417_417368

def arithmetic_mean (s : Finset ℤ) : ℚ :=
  (s.sum id : ℚ) / s.card

theorem arithmetic_mean_of_integers_minus5_to_5 :
  let S := (Finset.range 11).image (λ n, n - 5) in
  arithmetic_mean S = 0 :=
by
  -- The proof will go here
  sorry

end arithmetic_mean_of_integers_minus5_to_5_l417_417368


namespace num_distinct_lines_l417_417959

-- Define a 3D point in the grid
structure Point3D where
  i : ℕ
  j : ℕ
  k : ℕ
  h_i : 1 ≤ i ∧ i ≤ 5
  h_j : 1 ≤ j ∧ j ≤ 5
  h_k : 1 ≤ k ∧ k ≤ 5

-- Define a line in the grid by specifying a starting point and a direction
structure Line3D where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D
  p5 : Point3D
  collinear : ∀ (a b : ℤ), p2 = ⟨p1.i + a, p1.j + b, p1.k + c, _⟩ ∧ 
                          p3 = ⟨p1.i + 2*a, p1.j + 2*b, p1.k + 2*c, _⟩ ∧ 
                          p4 = ⟨p1.i + 3*a, p1.j + 3*b, p1.k + 3*c, _⟩ ∧ 
                          p5 = ⟨p1.i + 4*a, p1.j + 4*b, p1.k + 4*c, _⟩

-- Main theorem statement
theorem num_distinct_lines : ∃ n : ℕ, n = 150 ∧ 
  ∀ (lines : set Line3D), lines.finite ∧ lines.card = n := sorry

end num_distinct_lines_l417_417959


namespace linear_polynomial_split_quadratic_polynomial_split_l417_417466

theorem linear_polynomial_split (a b : ℝ) :
  (∫ x in -1..-1/2, a * x + b) + (∫ x in 1/2..1, a * x + b) =
  (∫ x in -1/2..1/2, a * x + b) :=
sorry

theorem quadratic_polynomial_split (a b c : ℝ) :
  ∑ I in [{(-1, -3/4)}, {(-1/4, 0)}, {(1/4, 3/4)}],
    (∫ x in I.1..I.2, a * x^2 + b * x + c) =
  ∑ I in [{(-3/4, -1/4)}, {(0, 1/4)}, {(3/4, 1)}],
    (∫ x in I.1..I.2, a * x^2 + b * x + c) :=
sorry

end linear_polynomial_split_quadratic_polynomial_split_l417_417466


namespace total_number_of_applications_l417_417116

def in_state_apps := 200
def out_state_apps := 2 * in_state_apps
def total_apps := in_state_apps + out_state_apps

theorem total_number_of_applications : total_apps = 600 := by
  sorry

end total_number_of_applications_l417_417116


namespace combined_mpg_l417_417442

theorem combined_mpg (miles_alice : ℕ) (mpg_alice : ℕ) (miles_bob : ℕ) (mpg_bob : ℕ) :
  miles_alice = 120 ∧ mpg_alice = 30 ∧ miles_bob = 180 ∧ mpg_bob = 20 →
  (miles_alice + miles_bob) / ((miles_alice / mpg_alice) + (miles_bob / mpg_bob)) = 300 / 13 :=
by
  intros h
  sorry

end combined_mpg_l417_417442


namespace graph_of_equation_l417_417050

theorem graph_of_equation {(x y: ℝ) : (x - y)^2 = x^2 - y^2 - 2 * x * y} : ∃ c: ℝ, ∀ x: ℝ, y = c * x := 
by
  sorry

end graph_of_equation_l417_417050


namespace increasing_range_of_a_l417_417184

noncomputable def is_increasing (f : ℝ → ℝ) (a b : ℝ) := ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem increasing_range_of_a (a : ℝ) :
  (∀ x : ℝ, x < 3 → -Math.log 2 * 2^(-x^2 + a*x + 1) * (-2*x + a) > 0) ↔ 6 < a :=
by
  sorry

end increasing_range_of_a_l417_417184


namespace goblins_pixies_handshake_l417_417735

theorem goblins_pixies_handshake :
  let goblins := 30,
      pixies := 10,
      goblins_handshakes := goblins * (goblins - 1) / 2,
      goblin_pixy_handshakes := pixies * goblins
  in goblins_handshakes + goblin_pixy_handshakes = 735 :=
by
  let goblins := 30
  let pixies := 10
  let goblins_handshakes := goblins * (goblins - 1) / 2
  let goblin_pixy_handshakes := pixies * goblins
  calc
    goblins_handshakes + goblin_pixy_handshakes = 30 * 29 / 2 + 10 * 30 := by rfl
    ... = 435 + 300 := by norm_num
    ... = 735 := by norm_num

end goblins_pixies_handshake_l417_417735


namespace quadrilateral_area_l417_417297
  
theorem quadrilateral_area
  (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB : real) (BC : real) (CD : real) (DA : real)
  (AC : real)
  (h1 : AB * AB + BC * BC = AC * AC)
  (h2 : DA * DA + CD * CD = AC * AC)
  (h3 : AC = 5)
  (h4 : ∃ (AB : ℕ) (BC : ℕ) (AD : ℕ) (CD : ℕ), 
       AB * AB + BC * BC = 25 ∧ AD * AD + CD * CD = 25 ∧ AB ≠ AD ∧ BC ≠ CD) :
  ∃ (area : real), area = 12 :=
by
  sorry

end quadrilateral_area_l417_417297


namespace truncated_pyramid_cyclic_faces_l417_417521

theorem truncated_pyramid_cyclic_faces (P : TruncatedTriangularPyramid) 
  (h1 : P.lateral_face_1.is_cyclic) (h2 : P.lateral_face_2.is_cyclic) : 
  P.lateral_face_3.is_cyclic :=
sorry

end truncated_pyramid_cyclic_faces_l417_417521


namespace quadrilateral_area_l417_417484

theorem quadrilateral_area (d : ℝ) (h1 : ℝ) (h2 : ℝ) :
    d = 10 → h1 = 7 → h2 = 3 → (d * h1 / 2 + d * h2 / 2) = 50 := 
begin
  intros hd hh1 hh2,
  rw [hd, hh1, hh2],
  norm_num,
end

end quadrilateral_area_l417_417484


namespace M_correct_complement_M_inter_N_correct_l417_417553

-- Defining the sets based on the conditions provided in the problem.
def M : set ℝ := {x | ∃ y, y = real.log ((x - 1) / x)}
def N : set ℝ := {y | ∃ x, y = x^2 + 2 * x + 2}

-- Define the complement of M in real numbers.
def complement_M : set ℝ := {x | x ∉ M}

-- Statements to prove
theorem M_correct : M = set.Iio 0 ∪ set.Ioi 1 := 
sorry

theorem complement_M_inter_N_correct : (set.compl M) ∩ N = {1} :=
sorry

end M_correct_complement_M_inter_N_correct_l417_417553


namespace tan_alpha_gt_tan_beta_l417_417208

open Real

theorem tan_alpha_gt_tan_beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) (h3 : sin α > sin β) : tan α > tan β :=
sorry

end tan_alpha_gt_tan_beta_l417_417208


namespace trailing_zeros_4018_l417_417734

theorem trailing_zeros_4018 :
  let x := 10^2009 - 1
  in ∃ R, R = 4018 ∧ (x * x + x) = 10^4018 - 10^2009 :=
sorry

end trailing_zeros_4018_l417_417734


namespace sugar_spilled_l417_417289

-- Define the initial amount of sugar and the amount left
def initial_sugar : ℝ := 9.8
def remaining_sugar : ℝ := 4.6

-- State the problem as a theorem
theorem sugar_spilled :
  initial_sugar - remaining_sugar = 5.2 := 
sorry

end sugar_spilled_l417_417289


namespace average_marks_l417_417681

-- Definitions
def tat_score (i_score : ℕ) : ℕ := 2 * i_score
def iva_score (d_score : ℕ) : ℕ := (3 / 5 : ℝ) * d_score

theorem average_marks :
  let D : ℕ := 90
  let I : ℕ := iva_score D
  let T : ℕ := tat_score I
  (D + I + T) / 3 = 84 :=
by {
  -- This is where the proof would go.
  -- Exact math proof steps are omitted as per the instructions.
  sorry
}

end average_marks_l417_417681


namespace savings_together_vs_separate_l417_417834

def price_per_window : ℕ := 100

def free_windows_per_5_purchased : ℕ := 2

def daves_windows_needed : ℕ := 10

def dougs_windows_needed : ℕ := 11

def total_windows_needed : ℕ := daves_windows_needed + dougs_windows_needed

-- Cost calculation for Dave's windows with the offer
def daves_cost_with_offer : ℕ := 8 * price_per_window

-- Cost calculation for Doug's windows with the offer
def dougs_cost_with_offer : ℕ := 9 * price_per_window

-- Total cost calculation if purchased separately with the offer
def total_cost_separately_with_offer : ℕ := daves_cost_with_offer + dougs_cost_with_offer

-- Total cost calculation if purchased together with the offer
def total_cost_together_with_offer : ℕ := 17 * price_per_window

-- Calculate additional savings if Dave and Doug purchase together rather than separately
def additional_savings_together_vs_separate := 
  total_cost_separately_with_offer - total_cost_together_with_offer = 0

theorem savings_together_vs_separate : additional_savings_together_vs_separate := by
  sorry

end savings_together_vs_separate_l417_417834


namespace percentage_of_difference_l417_417970

variable {P x y : ℝ}

-- Given conditions
def condition1 := P * (x - y) = 0.30 * (x + y)
def condition2 := y = 0.25 * x

-- The theorem statement to prove
theorem percentage_of_difference (h1 : condition1) (h2 : condition2) : P = 0.5 :=
sorry

end percentage_of_difference_l417_417970


namespace half_angle_fourth_quadrant_l417_417173

theorem half_angle_fourth_quadrant
  (α : ℝ)
  (hα : π < α ∧ α < 3 * π / 2)
  (h_sin : |sin (α / 2)| = -sin (α / 2)) :
  π < α / 2 ∧ α / 2 < 3 * π / 2 :=
by
  sorry

end half_angle_fourth_quadrant_l417_417173


namespace train_crosses_bridge_in_12_4_seconds_l417_417560

noncomputable def train_crossing_bridge_time (length_train : ℝ) (speed_train_kmph : ℝ) (length_bridge : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (1000 / 3600)
  let total_distance := length_train + length_bridge
  total_distance / speed_train_mps

theorem train_crosses_bridge_in_12_4_seconds :
  train_crossing_bridge_time 110 72 138 = 12.4 :=
by
  sorry

end train_crosses_bridge_in_12_4_seconds_l417_417560


namespace repayment_days_least_integer_l417_417282

theorem repayment_days_least_integer:
  ∀ (x : ℤ), (20 + 2 * x ≥ 60) → (x ≥ 20) :=
by
  intro x
  intro h
  sorry

end repayment_days_least_integer_l417_417282


namespace segment_length_l417_417676

noncomputable def point_mid (x y : ℝ) : ℝ := (x + y) / 2

noncomputable def distance (x y : ℝ) : ℝ := real.sqrt ((x - y)^2)

theorem segment_length 
    (square_pqrs_exists : ∃ PQRS : Type, is_square PQRS)
    (coplanar : is_coplanar PQRS)
    (pq_length : distance P Q = 10)
    (midpoint_n : point_mid P Q = N) :
    distance N C = 5 * real.sqrt 10 :=
by
  sorry

end segment_length_l417_417676


namespace rank_of_student_l417_417875

-- Given conditions
def mu : ℝ := 98
def sigma : ℝ := 10
def num_students : ℝ := 9450
def score : ℝ := 108

-- Question and expected answer
def expected_rank : ℝ := 1502

-- The theorem to state
theorem rank_of_student :
  ∀ (score : ℝ), (score = 108) → approx_rank score = 1502 := sorry


end rank_of_student_l417_417875


namespace carpet_needed_l417_417087

def room_length : ℕ := 15
def room_width : ℕ := 10
def ft2_to_yd2 : ℕ := 9

theorem carpet_needed :
  (room_length * room_width / ft2_to_yd2).ceil = 17 :=
by
  sorry

end carpet_needed_l417_417087


namespace xy_inequality_l417_417293

theorem xy_inequality (x y θ : ℝ) 
    (h : (x * Real.cos θ + y * Real.sin θ)^2 + x * Real.sin θ - y * Real.cos θ = 1) : 
    x^2 + y^2 ≥ 3/4 :=
sorry

end xy_inequality_l417_417293


namespace find_k_l417_417450

-- Define the height function.
def height (k : ℝ) (t : ℝ) : ℝ :=
  -k * (t - 3)^2 + 150

-- Given the conditions, define the specific scenario.
theorem find_k (k : ℝ) :
  height k 5 = 94 → k = 14 :=
by
  intro h_eq
  sorry

end find_k_l417_417450


namespace equal_powers_eq_a_b_l417_417571

theorem equal_powers_eq_a_b 
  (a b : ℝ) 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b)
  (h_exp_eq : a^b = b^a)
  (h_a_lt_1 : a < 1) : 
  a = b :=
sorry

end equal_powers_eq_a_b_l417_417571


namespace polynomial_divisibility_by_120_l417_417881

theorem polynomial_divisibility_by_120 (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end polynomial_divisibility_by_120_l417_417881


namespace sum_of_products_sum_of_products_correct_l417_417481

theorem sum_of_products (x y : ℤ) (h : x * y = 20 - 3 * x + y): (x, y) = (2, 14) ∨ (x, y) = (0, -20) ∨ (x, y) = (18, -2) ∨ (x, y) = (-16, -4) :=
sorry

-- The main theorem to be proved:
theorem sum_of_products_correct : 
  let pairs := [(2, 14), (0, -20), (18, -2), (-16, -4)] in 
  let products := List.map (λ p, p.1 * p.2) pairs in 
  List.sum products = 56 :=
by {
  have h_pairs : 
    (x, y) = (2, 14) ∨ (x, y) = (0, -20) ∨ (x, y) = (18, -2) ∨ (x, y) = (-16, -4) → 
    ∃ a b : ℤ, (a, b) ∈ pairs ∧ a * b = x * y := 
    by { intro h, cases h; simp },
   sorry
}

end sum_of_products_sum_of_products_correct_l417_417481


namespace find_ad_zero_l417_417642

variables (a b c d : ℝ) 

noncomputable def m (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem find_ad_zero (h_not_both_zero : ¬(c = 0 ∧ d = 0))
    (h_def : ∀ x : ℝ, c ≠ 0 → x ≠ -d/c)
    (h_solution_exists : ∃ x : ℝ, x = m (m x) ∧ x ≠ m x) :
    a + d = 0 := 
sorry

end find_ad_zero_l417_417642


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417010

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417010


namespace range_of_ratio_l417_417122

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_ratio (h1 : ∀ x : ℝ, x > 0 → 9 * f(x) < x * (deriv f x) ∧ x * (deriv f x) < 10 * f(x))
                       (h2 : ∀ x : ℝ, x > 0 → f(x) > 0) :
  2^9 < (f 2) / (f 1) ∧ (f 2) / (f 1) < 2^10 := by
  sorry

end range_of_ratio_l417_417122


namespace complement_of_M_l417_417951

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}

theorem complement_of_M :
  (U \ M) = {x | x < 1} :=
by
  sorry

end complement_of_M_l417_417951


namespace tan_simplification_l417_417914

theorem tan_simplification 
  (θ : ℝ) 
  (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / (Real.cos θ) - (Real.cos θ) / (1 + Real.sin θ) = 0 := 
by 
  sorry

end tan_simplification_l417_417914


namespace proper_subset_count_eq_three_l417_417205

-- Definition of proper subset
def proper_subset (M : Set ℕ) (S : Set ℕ) : Prop :=
  M ⊆ S ∧ M ≠ S

-- Definition of the set {1, 2}
def S : Set ℕ := {1, 2}

-- The main theorem statement
theorem proper_subset_count_eq_three : 
  (finset.filter (λ M, proper_subset M S) (finset.powerset S)).card = 3 :=
by sorry

end proper_subset_count_eq_three_l417_417205


namespace problem_1_problem_3_problem_4_l417_417274

-- Definition of the function f(x)
def f (x : ℝ) (b c : ℝ) : ℝ := (|x| * x) + (b * x) + c

-- Prove that when b > 0, f(x) is monotonically increasing on ℝ
theorem problem_1 (b c : ℝ) (h : b > 0) : 
  ∀ x y : ℝ, x < y → f x b c < f y b c :=
sorry

-- Prove that the graph of f(x) is symmetric about the point (0, c) when b = 0
theorem problem_3 (b c : ℝ) (h : b = 0) :
  ∀ x : ℝ, f x b c = f (-x) b c :=
sorry

-- Prove that when b < 0, f(x) = 0 can have three real roots
theorem problem_4 (b c : ℝ) (h : b < 0) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0 :=
sorry

end problem_1_problem_3_problem_4_l417_417274


namespace problem_statement_l417_417752

noncomputable def correlation_coefficient (data : List (ℝ × ℝ)) : ℝ :=
let xs := data.map Prod.fst
let ys := data.map Prod.snd
let mean_x := (List.sum xs) / (xs.length : ℝ)
let mean_y := (List.sum ys) / (ys.length : ℝ)
let covariance := List.sum (data.map (λ (p : ℝ × ℝ), (p.1 - mean_x) * (p.2 - mean_y)))
let variance_x := List.sum (xs.map (λ x, (x - mean_x)^2))
let variance_y := List.sum (ys.map (λ y, (y - mean_y)^2))
covariance / (Real.sqrt variance_x * Real.sqrt variance_y)

noncomputable def regression_slope (data : List (ℝ × ℝ)) : ℝ :=
let xs := data.map Prod.fst
let ys := data.map Prod.snd
let mean_x := (List.sum xs) / (xs.length : ℝ)
let mean_y := (List.sum ys) / (ys.length : ℝ)
let numerator := List.sum (data.map (λ (p : ℝ × ℝ), (p.1 - mean_x) * (p.2 - mean_y)))
let denominator := List.sum (xs.map (λ x, (x - mean_x)^2))
numerator / denominator

noncomputable def regression_intercept (data : List (ℝ × ℝ)) : ℝ :=
let mean_x := (List.sum (data.map Prod.fst)) / (data.length : ℝ)
let mean_y := (List.sum (data.map Prod.snd)) / (data.length : ℝ)
mean_y - regression_slope data * mean_x

def predict_y (data : List (ℝ × ℝ)) (x : ℝ) : ℝ :=
regression_slope data * x + regression_intercept data

theorem problem_statement :
  let data := [(2, 300), (4, 400), (5, 400), (6, 400), (8, 500)] in
  correlation_coefficient data ≈ 0.95 ∧ predict_y data 15 = 700 :=
by { sorry }

end problem_statement_l417_417752


namespace greatest_p_meets_conditions_l417_417412

-- Define a four-digit number and its reversal being divisible by 63 and another condition of divisibility
def is_divisible_by (n m : ℕ) : Prop :=
  m % n = 0

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ a d => a * 10 + d) 0

def is_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def p := 9507

-- The main theorem we aim to prove.
theorem greatest_p_meets_conditions (p q : ℕ) 
  (h1 : is_four_digit p) 
  (h2 : is_four_digit q) 
  (h3 : reverse_digits p = q) 
  (h4 : is_divisible_by 63 p) 
  (h5 : is_divisible_by 63 q) 
  (h6 : is_divisible_by 9 p) : 
  p = 9507 :=
sorry

end greatest_p_meets_conditions_l417_417412


namespace part1_trajectory_of_M_part2_find_t_l417_417167

noncomputable def trajectory_of_M : ℝ → ℝ := λ x, real.sqrt (3 * x)

theorem part1_trajectory_of_M (b a x y : ℝ) (H : (6 * a - b^2 = 0)) (M_on_PQ : y^2 = 3 * x) : 
  ∃ C, trajectory_of_M x^2 = C :=
by {
  sorry
}
  
theorem part2_find_t (k t : ℝ) (H1 : 4 * (real.sqrt ((9/4) + ((9/(4 * k^2))))) = (real.sqrt 3) * (1 + k^2) * real.sqrt ((9 / k^4) - (12 / k^2))) (H2 : t = ((3/(2 * k^2)) - 1)) : 
  t = 9 / 2 :=
by {
  sorry
}

end part1_trajectory_of_M_part2_find_t_l417_417167


namespace true_propositions_count_l417_417947

theorem true_propositions_count (a : ℝ) (h : a > 2 → a^2 > 4) :
  (a^2 > 4 → a > 2 ∨ ¬(a^2 > 4 ∨ a ≤ 2)) → 
  (¬(a > 2) → ¬(a^2 > 4) ∨ ¬(¬(a > 2) → ¬(a^2 > 4))) → 
  1 :=
by sorry

end true_propositions_count_l417_417947


namespace vasya_tolya_badges_l417_417765

theorem vasya_tolya_badges (x y : ℤ)
    (h1 : y = x + 5) -- Vasya initially had 5 more badges than Tolya
    (h2 : (y - (6 * (y / 25) / 25) + (4 * x) / 25) = (x - (4 * x) / 5 + 6 * (y / 25) / 5 - 1)) : -- equation balancing after exchange
    x = 45 ∧ y = 50 := 
sorry

end vasya_tolya_badges_l417_417765


namespace weight_of_empty_carton_l417_417726

theorem weight_of_empty_carton
    (half_full_carton_weight : ℕ)
    (full_carton_weight : ℕ)
    (h1 : half_full_carton_weight = 5)
    (h2 : full_carton_weight = 8) :
  full_carton_weight - 2 * (full_carton_weight - half_full_carton_weight) = 2 :=
by
  sorry

end weight_of_empty_carton_l417_417726


namespace average_marks_l417_417688

theorem average_marks (D I T : ℕ) 
  (hD : D = 90)
  (hI : I = (3 * D) / 5)
  (hT : T = 2 * I) : 
  (D + I + T) / 3 = 84 :=
by
  sorry

end average_marks_l417_417688


namespace find_sin_2_angle_BAD_l417_417602

noncomputable def isosceles_right_triangle (A B C : Type*) [InnerProductSpace ℝ Type*] (AB_length BC_length : ℝ) :=
  AB_length = 2 ∧ BC_length = 2 ∧ ∠ BCA = π / 2

noncomputable def right_triangle_with_equal_area (A C D : Type*) [InnerProductSpace ℝ Type*]
  (AC_length : ℝ) (ACD_area : ℝ) (ACD_right_angle : ∠ ACD) :=
  AC_length = 2 * real.sqrt 2 ∧ ACD_area = 2 ∧ ACD_right_angle = π / 2

theorem find_sin_2_angle_BAD (A B C D : Type*) [InnerProductSpace ℝ Type*]
  (h1 : isosceles_right_triangle A B C 2 2)
  (h2 : right_triangle_with_equal_area A C D (2 * real.sqrt 2) 2 (π / 2)):
  sin (2 * (angle A B D)) = 9 / 25 :=
begin
  sorry
end

end find_sin_2_angle_BAD_l417_417602


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417029

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417029


namespace range_a_l417_417190

noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ set.Icc (0 : ℝ) (1 / 2) then 
    - (1 / 2) * x + 1 / 4 
  else if x ∈ set.Ioc (1 / 2) 1 then 
    2 * x^2 / (x + 2) 
  else 
    0 -- value outside [0,1] is 0, not our concern

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.cos (π * x / 2) + 5 - 2 * a

theorem range_a (a : ℝ) (h : a > 0) : 
  (∀ x1 ∈ set.Icc (0 : ℝ) 1, ∃ x2 ∈ set.Icc (0 : ℝ) 1, f x1 = g a x2) ↔ 
  (5 / 2 ≤ a ∧ a ≤ 13 / 3) :=
sorry

end range_a_l417_417190


namespace largest_and_smallest_terms_among_first_30_l417_417603

noncomputable def a_n (n : ℕ) : ℝ :=
  (n - real.sqrt 98) / (n - real.sqrt 99)

theorem largest_and_smallest_terms_among_first_30 :
  let max_n := 10
  let min_n := 9
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 → a_n n ≤ a_n max_n) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 → a_n min_n ≤ a_n n) :=
begin
  sorry
end

end largest_and_smallest_terms_among_first_30_l417_417603


namespace minimum_tasks_for_18_points_l417_417404

def task_count (points : ℕ) : ℕ :=
  if points <= 9 then
    (points / 3) * 1
  else if points <= 15 then
    3 + (points - 9 + 2) / 3 * 2
  else
    3 + 4 + (points - 15 + 2) / 3 * 3

theorem minimum_tasks_for_18_points : task_count 18 = 10 := by
  sorry

end minimum_tasks_for_18_points_l417_417404


namespace baking_trays_used_l417_417451

-- Let T be the number of baking trays Anna used.
variable (T : ℕ)

-- Condition: Each tray has 20 cupcakes.
def cupcakes_per_tray : ℕ := 20

-- Condition: Each cupcake was sold for $2.
def cupcake_price : ℕ := 2

-- Condition: Only 3/5 of the cupcakes were sold.
def fraction_sold : ℚ := 3 / 5

-- Condition: Anna earned $96 from sold cupcakes.
def earnings : ℕ := 96

-- Derived expressions:
def total_cupcakes (T : ℕ) : ℕ := cupcakes_per_tray * T

def sold_cupcakes (T : ℕ) : ℚ := fraction_sold * total_cupcakes T

def total_earnings (T : ℕ) : ℚ := cupcake_price * sold_cupcakes T

-- The statement to be proved: Given the conditions, the number of trays T must be 4.
theorem baking_trays_used (h : total_earnings T = earnings) : T = 4 := by
  sorry

end baking_trays_used_l417_417451


namespace inequality_not_always_hold_l417_417964

theorem inequality_not_always_hold (a b : ℝ) (h : a > -b) : ¬ (∀ a b : ℝ, a > -b → (1 / a + 1 / b > 0)) :=
by
  intro h2
  have h3 := h2 a b h
  sorry

end inequality_not_always_hold_l417_417964


namespace range_of_k_l417_417541

theorem range_of_k (k : ℝ) 
  (h1 : ∀ x y : ℝ, (x^2 / (k-3) + y^2 / (2-k) = 1) → (k-3 < 0) ∧ (2-k > 0)) : 
  k < 2 := by
  sorry

end range_of_k_l417_417541


namespace coefficient_x5_in_expansion_l417_417244

theorem coefficient_x5_in_expansion : 
  ∀ (x : ℝ), (∃ c : ℝ, (∀ y, (3 - y)^7 = (∑ i in finset.range 8, c * y^i) )) → 
  ((-189 : ℝ) ∈ (λ (s:finset (ℕ × ℝ × ℕ)), s.filter (λ (u:ℕ × ℝ × ℕ), u.1 = 7 ∧ u.3 = 5))) :=
begin
  sorry
end

end coefficient_x5_in_expansion_l417_417244


namespace weekly_rental_cost_is_10_l417_417085

-- Define the weekly rental cost as a variable W
variable (W : ℕ)

-- Define the conditions given in the problem
def monthly_rental_cost := 35
def yearly_cost_monthly := 12 * monthly_rental_cost
def savings := 100
def yearly_cost_weekly := yearly_cost_monthly + savings
def weeks_in_year := 52

-- The formal statement we want to prove
theorem weekly_rental_cost_is_10 :
  (weeks_in_year * W = yearly_cost_weekly) → W = 10 :=
begin
  -- skipping proof
  sorry
end

end weekly_rental_cost_is_10_l417_417085


namespace cube_angle_60_degrees_l417_417240

-- Definitions of the vertices and properties of the cube ABCD-A1B1C1D1
variables (A B C D A1 B1 C1 D1 : Type) [IsCube A B C D A1 B1 C1 D1]

-- Statement that we want to prove
theorem cube_angle_60_degrees : angle A1 C1 B1 C = 60 :=
sorry

end cube_angle_60_degrees_l417_417240


namespace line_equation_through_intersection_and_y_intercept_l417_417932

theorem line_equation_through_intersection_and_y_intercept :
  ∃ (a b c : ℝ), (∀ x y : ℝ, 3 * x + 2 * y - 5 = 0 ∧ 2 * x + 3 * y - 5 = 0 → 6 * x - y - 5 = 0) ∧
                   (c = -5) :=
begin
  sorry
end

end line_equation_through_intersection_and_y_intercept_l417_417932


namespace max_b_c_plus_four_over_a_l417_417185

theorem max_b_c_plus_four_over_a (a b c : ℝ) (ha : a < 0)
  (h_quad : ∀ x : ℝ, -1 < x ∧ x < 2 → (a * x^2 + b * x + c) > 0) : 
  b - c + 4 / a ≤ -4 :=
sorry

end max_b_c_plus_four_over_a_l417_417185


namespace radius_of_circumcircle_of_triangle_l417_417533

theorem radius_of_circumcircle_of_triangle (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧ (∃ (R : ℝ), R = 2.5) :=
by {
  sorry
}

end radius_of_circumcircle_of_triangle_l417_417533


namespace michael_class_choosing_l417_417286

open Nat

theorem michael_class_choosing :
  (choose 6 3) * (choose 4 2) + (choose 6 4) * (choose 4 1) + (choose 6 5) = 186 := 
by
  sorry

end michael_class_choosing_l417_417286


namespace parabola_has_zero_y_intercepts_l417_417123

lemma parabola_no_y_intercepts :
  (∀ y : ℝ, 3 * y^2 - 2 * y + 1 ≠ 0) :=
by
  intro y
  have h_disc : (-2)^2 - 4 * 3 * 1 < 0 := by linarith
  rw [←sub_eq_zero] at * 
  intro h_eq
  have : (3 * y^2 - 2 * y + 1) = 0 := by assumption
  have lt_zero_disc := by {
    have := calc
      (3 : ℝ) > 0               : by norm_num,
      (3 : ℝ) * (y^2) ≥ 0       : mul_nonneg_of_pos_of_sq_zero,
      (3 : ℝ) * (y^2) - (2 : ℝ) * y + 1 ≥ 0            : by linarith [calc_usage_here]
  }
  contradiction

theorem parabola_has_zero_y_intercepts : 
  ¬( ∃ y : ℝ, 3 * y^2 - 2 * y + 1 = 0) :=
by
  intro h
  apply parabola_no_y_intercepts
  cases h with y hy
  apply hy

end parabola_has_zero_y_intercepts_l417_417123


namespace circle_center_radius_sum_l417_417271

-- Definition of the circle equation
def circle_eq (x y : ℝ) : Prop :=
  x^2 + 14 * y + 63 = - (y^2 + 12 * x)

-- Definition of the center (a, b) and radius r
def center_and_radius (x y : ℝ) : Prop :=
  ∃ a b r : ℝ, (x + 6) ^ 2 + (y + 7) ^ 2 = 22 ∧ a = -6 ∧ b = -7 ∧ r = real.sqrt 22

-- Proof goal
theorem circle_center_radius_sum :
  ∃ a b r : ℝ, circle_eq x y ∧ center_and_radius x y ∧ a + b + r = -13 + real.sqrt 22 :=
by
  sorry

end circle_center_radius_sum_l417_417271


namespace digit_202_of_fraction_5_13_l417_417367

theorem digit_202_of_fraction_5_13 : 
  let s := "384615".to_list
  (s.cycle.drop 201).head = '6' := by
sorry

end digit_202_of_fraction_5_13_l417_417367


namespace constant_term_expansion_equals_10_l417_417220

theorem constant_term_expansion_equals_10 (n : ℕ) (h₁ : (x^3 + 1/x^2)^n = 32) :
  ∃ C, C = ((5 choose 3) : ℕ) ∧ C = 10 := 
sorry

end constant_term_expansion_equals_10_l417_417220


namespace M_final_value_is_4_l417_417291

noncomputable def final_value_of_M : ℤ :=
  let M := 1 in
  let M1 := M + 1 in
  let M2 := M1 + 2 in
  M2

theorem M_final_value_is_4 : final_value_of_M = 4 :=
by
  -- Proof steps go here, but we will use sorry for now.
  sorry

end M_final_value_is_4_l417_417291


namespace max_acute_angles_non_convex_ngon_l417_417044

def max_acute_angles (n : ℕ) : ℕ :=
  ⌊(2 * n / 3 : ℚ)⌋ + 1

theorem max_acute_angles_non_convex_ngon (n : ℕ) (h1 : n ≥ 3) (h2 : (∀ k : ℕ, k < n → k * 90 < (n-2) * 180 - (n-k) * 360)) 
  (h3 : (∀ k : ℕ, k < n → ∀ m : ℕ, m = n - k → m * 180 > m * 180)) : 
  ∀ k : ℕ, k ≤ max_acute_angles n :=
sorry

end max_acute_angles_non_convex_ngon_l417_417044


namespace diana_additional_game_time_l417_417868

theorem diana_additional_game_time :
  ∀ (reading_hours : ℕ) (minutes_per_hour : ℕ) (raise_percent : ℕ),
    reading_hours = 12 → minutes_per_hour = 30 → raise_percent = 20 →
    (reading_hours * (minutes_per_hour * raise_percent / 100)) = 72 :=
by
  intros reading_hours minutes_per_hour raise_percent h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end diana_additional_game_time_l417_417868


namespace find_pairs_satisfying_mn_divisibility_l417_417886

theorem find_pairs_satisfying_mn_divisibility :
  ∀ (m n : ℕ), 0 < m → 0 < n →
  (mn : ℕ) = m * n →
  (3^m + 1) % m * n = 0 → (3^n + 1) % m * n = 0 →
  (m, n) = (1, 1) ∨ (m, n) = (1, 2) ∨ (m, n) = (1, 4) ∨ (m, n) = (2, 1) ∨ (m, n) = (4, 1) :=
begin
  intros m n hm hn hmn hdivm hdivn,
  -- Add proofs here 
  sorry
end

end find_pairs_satisfying_mn_divisibility_l417_417886


namespace probability_at_least_one_two_l417_417351

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end probability_at_least_one_two_l417_417351


namespace angle_turned_by_hour_hand_l417_417704

-- Define the problem conditions
def hours_in_clock : ℕ := 12
def angle_between_adjacent_hours : ℤ := 30 -- in degrees
def divisions_between_hours : ℕ := 5
def angle_per_division : ℤ := 6 -- in degrees

-- The main proof statement
theorem angle_turned_by_hour_hand :
  let number_of_divisions := (5 : ℤ) + 5 * (3 / 4 : ℤ) in
  let calculated_angle := angle_per_division * number_of_divisions in
  calculated_angle = 52 * 60 + 30 :=
by
  sorry

end angle_turned_by_hour_hand_l417_417704


namespace not_possible_odd_white_black_columns_l417_417395

theorem not_possible_odd_white_black_columns :
  ∀ (cube : ℕ → ℕ → ℕ → Prop),
  (∃ (white black : ℕ), white = 14 ∧ black = 13 ∧ (∀ x y z, cube x y z → (cube x y z = white ∨ cube x y z = black))
    ∧ ∀ (i j k : ℕ), i < 3 ∧ j < 3 ∧ k < 3 
    →(count_odd (λ (x y z : ℕ), cube x y z = white) + count_odd (λ (x y z : ℕ), cube x y z = black) = 9)) → 
  false :=
by
sorrry

end not_possible_odd_white_black_columns_l417_417395


namespace parking_garage_floors_l417_417253

theorem parking_garage_floors 
  (total_time : ℕ)
  (time_per_floor : ℕ)
  (gate_time : ℕ)
  (every_n_floors : ℕ) 
  (F : ℕ) 
  (h1 : total_time = 1440)
  (h2 : time_per_floor = 80)
  (h3 : gate_time = 120)
  (h4 : every_n_floors = 3)
  :
  F = 13 :=
by
  have total_id_time : ℕ := gate_time * ((F - 1) / every_n_floors)
  have total_drive_time : ℕ := time_per_floor * (F - 1)
  have total_time_calc : ℕ := total_drive_time + total_id_time
  have h5 := total_time_calc = total_time
  -- Now we simplify the algebraic equation given the problem conditions
  sorry

end parking_garage_floors_l417_417253


namespace find_a4_from_roots_of_quadratic_l417_417593

variable {a : ℕ → ℝ}

-- Definition of the geometric sequence property
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = a n * (a 1 / a 0)

-- Conditions
variables (a3 a5 : ℝ) (h1 : a3 * a5 = 64) (h2 : a3 + a5 = 34)
variable (ha : is_geometric_sequence a)

-- Define the specific sequence values
noncomputable def a_3 : ℝ := a 3
noncomputable def a_5 : ℝ := a 5
noncomputable def a_4 : ℝ := a 4

-- Proof Statement
theorem find_a4_from_roots_of_quadratic :
  a_4 = 8 ∨ a_4 = -8 :=
sorry

end find_a4_from_roots_of_quadratic_l417_417593


namespace problem1_problem2_problem3_l417_417930

-- Definitions and conditions given in the problem
def isPermutationOf (l1 l2 : List ℕ) : Prop :=
  l1.length = l2.length ∧ l1.perm l2

def sequence_b (a : List ℕ) (n : ℕ) : List ℕ :=
  a.map (λ x => n + 1 - x)

def sequence_c (n : ℕ) : List ℕ :=
  List.range n |>.reverse.map (λ x => x + 1)

def S_n (c : List ℕ) : ℕ :=
  (List.range c.length).sum (λ i => (i + 1) * c.getI i)

-- Problem statements
theorem problem1 (n : ℕ) (h : n % 2 = 0) (a : List ℕ) (ha : a.length = n) (hp : isPermutationOf a (List.range n.map (λ x => x + 1))) :
  ¬ (∀ k, 1 ≤ k ∧ k ≤ n → a.getI (k - 1) = sequence_b a n |>.getI (k - 1)) :=
sorry

theorem problem2 (n : ℕ) :
  S_n (sequence_c n) = (1/6)*n*(n+1)*(n+2) :=
sorry

theorem problem3 (n : ℕ) (a : List ℕ) (ha : a.length = n) (hb : ∀ k, 1 ≤ k ∧ k ≤ n → sequence_b a n.getI (k - 1) = b.getI (k - 1)) :
  (List.range n).sum (λ i => (i + 1) * sequence_b a n.getI i) ≤ (1/6)*n*(n+1)*(2n+1) ∧
  (List.range n).sum (λ i => (i + 1) * a.getI i)>= (S_n (sequence_c n)) :=
sorry

end problem1_problem2_problem3_l417_417930


namespace range_of_g_l417_417105

-- Define the function g(x)
def g (x : ℝ) : ℝ :=
  (Real.cos x) ^ 4 + (Real.sin x) ^ 2

theorem range_of_g : ∀ x : ℝ, g x ∈ set.Icc (3 / 4) 1 := by
  sorry

end range_of_g_l417_417105


namespace negative_real_roots_q_iter2016_l417_417640

def q (x : ℝ) : ℝ := 2 * x^2 + 2 * x - 1

def q_iter (n : ℕ) (x : ℝ) : ℝ :=
  Nat.recOn n x (λ _ ih, q ih)

theorem negative_real_roots_q_iter2016 :
  ∃ count : ℕ, count = (2^2017 + 1) / 3 ∧ ∀ x : ℝ, q_iter 2016 x < 0 ↔ x < 0 :=
sorry

end negative_real_roots_q_iter2016_l417_417640


namespace tangent_line_eq_bounded_area_l417_417985

-- Given two parabolas and a tangent line, and a positive constant a
variables (a : ℝ)
variables (y1 y2 l : ℝ → ℝ)

-- Conditions:
def parabola1 := ∀ (x : ℝ), y1 x = x^2 + a * x
def parabola2 := ∀ (x : ℝ), y2 x = x^2 - 2 * a * x
def tangent_line := ∀ (x : ℝ), l x = - (a / 2) * x - (9 * a^2 / 16)
def a_positive := a > 0

-- Proof goals:
theorem tangent_line_eq : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∀ x, (y1 x = l x ∨ y2 x = l x) :=
sorry

theorem bounded_area : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∫ (x : ℝ) in (-3 * a / 4)..(3 * a / 4), (y1 x - l x) + (y2 x - l x) = 9 * a^3 / 8 :=
sorry

end tangent_line_eq_bounded_area_l417_417985


namespace min_a_l417_417719

noncomputable def a : ℕ → ℝ
| 1 := 1 / 3
| (n + 1) := if n = 1 then (1 / 3) * (1 / 3) else a n * a 1

def S_n (n : ℕ) : ℝ := (List.range n).sum (λ i, a (i + 1))

theorem min_a (a : ℝ) : (∀ n, S_n n < a) → a ≥ 1 / 2 := sorry

end min_a_l417_417719


namespace bananas_per_monkey_l417_417823

-- Define the given conditions
def total_monkeys : ℕ := 12
def piles_with_9hands : ℕ := 6
def hands_per_pile_9hands : ℕ := 9
def bananas_per_hand_9hands : ℕ := 14
def piles_with_12hands : ℕ := 4
def hands_per_pile_12hands : ℕ := 12
def bananas_per_hand_12hands : ℕ := 9

-- Calculate the total number of bananas from each type of pile
def total_bananas_9hands : ℕ := piles_with_9hands * hands_per_pile_9hands * bananas_per_hand_9hands
def total_bananas_12hands : ℕ := piles_with_12hands * hands_per_pile_12hands * bananas_per_hand_12hands

-- Sum the total number of bananas
def total_bananas : ℕ := total_bananas_9hands + total_bananas_12hands

-- Prove that each monkey gets 99 bananas
theorem bananas_per_monkey : total_bananas / total_monkeys = 99 := by
  sorry

end bananas_per_monkey_l417_417823


namespace expand_polynomial_l417_417111

theorem expand_polynomial (x : ℂ) : 
  (1 + x^4) * (1 - x^5) * (1 + x^7) = 1 + x^4 - x^5 + x^7 + x^11 - x^9 - x^12 - x^16 := 
sorry

end expand_polynomial_l417_417111


namespace sequence_product_integer_l417_417061

theorem sequence_product_integer (a b : ℤ) (a_pos : 0 < a) (b_pos : 0 < b) :
  ∀ (m n : ℕ), 0 < m → 0 < n →
  let x : ℕ → ℤ := λ n, nat.rec_on n 0 (λ _ ih, nat.rec_on ih 1 (λ _ ih', a * ih + b * ih'))
  in (x (m + 1) * x (m + 2) * ... * x (m + n)) / (x 1 * x 2 * ... * x n) ∈ ℤ :=
by
  assume a b a_pos b_pos m n m_pos n_pos
  let x : ℕ → ℤ := λ n, nat.rec_on n 0 (λ _ ih, nat.rec_on ih 1 (λ _ ih', a * ih + b * ih'))
  sorry

end sequence_product_integer_l417_417061


namespace probability_at_least_one_two_l417_417355

theorem probability_at_least_one_two (dice_fair : ∀ i, 1 ≤ i ∧ i ≤ 8) (dice_count : 2):
  ∃ probability, probability = 15 / 64 := 
by
  sorry

end probability_at_least_one_two_l417_417355


namespace divisors_sum_l417_417276

theorem divisors_sum (d : Fin 27 → ℕ) (h : ∀ i, d i ∣ 2014^2) :
  (\sum i : Fin 27, 1 / (d i + 2014)) = 27 / 2014 := by
  sorry

end divisors_sum_l417_417276


namespace number_of_elements_l417_417261

def A : Set ℤ := {x | 2 ≤ 2^(2 - x) ∧ 2^(2 - x) < 8}
def B : Set ℝ := {x | abs (Real.log x / Real.log 2) > 1}

theorem number_of_elements {X : Type*} [has_mem ℤ X] [has_mem ℝ X]:
  subset A (coe : ℤ → X) ∧ subset B (coe : ℝ → X) 
  → fintype.card (A ∩ (akkbar : Set ℝ \ B)) = 2 := 
by sorry

end number_of_elements_l417_417261


namespace unshaded_triangles_area_l417_417424

theorem unshaded_triangles_area 
  (octagon_side_length : ℝ) 
  (octagon_sides : ∃ n: ℕ, n = 8)
  (side_length_2cm : octagon_side_length = 2) 
  (shaded_rectangles : ∃ PQTU RSVW : set (ℝ × ℝ), true) 
  (unshaded_triangles : ∃ n: ℕ, n = 4) 
  : 
  ∃ area : ℝ, area = 4 :=
  sorry

end unshaded_triangles_area_l417_417424


namespace exists_unique_a_for_arithmetic_sequence_l417_417265

variable {R : Type*} [LinearOrderedField R]

def f (a x : R) : R := (x-2) * exp x - a * (x^2 - 2 * x) + 2

theorem exists_unique_a_for_arithmetic_sequence
  (a : R) (h_a_lt_half_e : a < exp 1 / 2)
  (h_three_zeros : ∃ x1 x2 x3 : R, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)
  (h_neg_product : ∃ x1 x3 : R, f a x1 = 0 ∧ f a x3 = 0 ∧ x1 * x3 < 0) :
  ∃! (a : R),
    (∃ x1 x2 x3 : R, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
      (x1 + x3) / 2 = x2) ∧
    a = exp ((x1+x3)/2) / ((x1+x3)/2) + 2 / ((x1+x3)/2 * ((x1+x3)/2 - 2)) :=
sorry

end exists_unique_a_for_arithmetic_sequence_l417_417265


namespace icosahedron_dodecahedron_same_sphere_l417_417449

theorem icosahedron_dodecahedron_same_sphere
    (S : Sphere)
    (I D : Polyhedron)
    (OC : Point) 
    (A : Point)
    (B C D : Point)
    (inscribed_in_S_I : inscribed S I)
    (inscribed_in_S_D : inscribed S D)
    (center_S : center S = OC)
    (vertex_D : vertex D = A)
    (center_of_face_D : center_of_face D = B ∨ center_of_face D = C ∨ center_of_face D = D) :
    circumscribe OC I = circumscribe OC D :=
by
  sorry

end icosahedron_dodecahedron_same_sphere_l417_417449


namespace find_A_l417_417336

theorem find_A : ∃ A : ℕ, 691 - (600 + A * 10 + 7) = 4 ∧ A = 8 := by
  sorry

end find_A_l417_417336


namespace each_monkey_gets_bananas_l417_417826

-- Define the conditions
def total_monkeys : ℕ := 12
def total_piles : ℕ := 10
def first_piles : ℕ := 6
def first_pile_hands : ℕ := 9
def first_hand_bananas : ℕ := 14
def remaining_piles : ℕ := total_piles - first_piles
def remaining_pile_hands : ℕ := 12
def remaining_hand_bananas : ℕ := 9

-- Define the number of bananas in each type of pile
def bananas_in_first_piles : ℕ := first_piles * first_pile_hands * first_hand_bananas
def bananas_in_remaining_piles : ℕ := remaining_piles * remaining_pile_hands * remaining_hand_bananas
def total_bananas : ℕ := bananas_in_first_piles + bananas_in_remaining_piles

-- Define the main theorem to be proved
theorem each_monkey_gets_bananas : total_bananas / total_monkeys = 99 := by
  sorry

end each_monkey_gets_bananas_l417_417826


namespace areas_of_regions_l417_417080

theorem areas_of_regions (A B C : ℝ)
  (h_triangle : is_right_triangle 20 21 29)
  (h_circle : is_circumscribed 20 21 29)
  (h_conditions: A + B + 210 = C) : 
  A + B + 210 = C :=
sorry

end areas_of_regions_l417_417080


namespace eval_expr_l417_417067

theorem eval_expr : 4 * (-3) + 60 / (-15) = -16 := by
  sorry

end eval_expr_l417_417067


namespace triangle_with_incircle_radius_one_has_sides_5_4_3_l417_417136

variable {a b c : ℕ} (h1 : a ≥ b ∧ b ≥ c)
variable (h2 : ∃ (a b c : ℕ), (a + b + c) / 2 * 1 = (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_with_incircle_radius_one_has_sides_5_4_3 :
  a = 5 ∧ b = 4 ∧ c = 3 :=
by
    sorry

end triangle_with_incircle_radius_one_has_sides_5_4_3_l417_417136


namespace reliable_length_l417_417284

def measurements : List ℝ := [1.991, 1.995, 1.996, 1.993, 1.999, 1.995, 1.997, 1.994, 1.995, 1.930]

theorem reliable_length (h : ∀ m ∈ measurements, (m * 1000).round / 1000 = m) : 
  ∃ l, (∃ k, l ∈ measurements ∧ l = 1.995 ∧ k = measurements.count 1.995) :=
by
  sorry

end reliable_length_l417_417284


namespace find_S5_l417_417919

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → ℝ) : (ℕ → ℝ) :=
  λ n, finset.sum (finset.range n) a

-- Conditions
axiom a_n_gt_zero : ∀ n, a n > 0
axiom geom_seq : geometric_sequence a q
axiom prod_a2_a4 : a 2 * a 4 = 16
axiom sum_S3 : sum_of_first_n_terms a 3 = 7
axiom q_pos : q > 0

theorem find_S5 : sum_of_first_n_terms a 5 = 31 :=
sorry

end find_S5_l417_417919


namespace false_statement_B_l417_417784

theorem false_statement_B :
  ∃ (Q : Type) [quadrilateral Q], quadrilateral.has_perpendicular_and_equal_diagonals Q ∧ ¬ quadrilateral.is_square Q := by
  sorry

end false_statement_B_l417_417784


namespace arrangements_of_4_volunteers_2_elderly_not_adjacent_l417_417832

def arrangements_volunteers_elderly (v e : ℕ) (cond : e ≠ 0) : ℕ :=
  let slots := v + 1
  Nat.factorial v * ((slots.choose e) * Nat.factorial e - (slots-1).choose(e-1) * Nat.factorial e)

theorem arrangements_of_4_volunteers_2_elderly_not_adjacent :
  arrangements_volunteers_elderly 4 2 (by decide) = 480 :=
by
  sorry

end arrangements_of_4_volunteers_2_elderly_not_adjacent_l417_417832


namespace gcd_of_12547_23791_l417_417139

theorem gcd_of_12547_23791 : Nat.gcd 12547 23791 = 1 :=
by
  sorry

end gcd_of_12547_23791_l417_417139


namespace micheal_completes_alone_in_25_days_l417_417647

theorem micheal_completes_alone_in_25_days :
  (∀ (W : ℕ) (M A : ℕ), 
    (1 / M + 1 / A = 1 / 20) → 
    (18 / 20) + (10 / A) = 1 →
    A = 100) →
  M = 25 :=
by
  intros W M A hRate hCompleted hAdam
  have hMichaelRate : 1 / M = 1 / 25, from sorry,
  show M = 25, from sorry

end micheal_completes_alone_in_25_days_l417_417647


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417011

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417011


namespace password_lock_probability_l417_417421

theorem password_lock_probability :
  (∃ (A B C : ℕ), 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 
   (∃ password : ℕ × ℕ × ℕ, password = (A, B, C) ∧ 
   (∃ chosen_middle_digit : ℕ, 0 <= chosen_middle_digit ∧ chosen_middle_digit <= 9 ∧
   chosen_middle_digit = B))) →
  (1/10 : ℝ) = 0.1 :=
by 
  sorry

end password_lock_probability_l417_417421


namespace total_winter_clothing_l417_417288

def num_scarves (boxes : ℕ) (scarves_per_box : ℕ) : ℕ := boxes * scarves_per_box
def num_mittens (boxes : ℕ) (mittens_per_box : ℕ) : ℕ := boxes * mittens_per_box
def num_hats (boxes : ℕ) (hats_per_box : ℕ) : ℕ := boxes * hats_per_box
def num_jackets (boxes : ℕ) (jackets_per_box : ℕ) : ℕ := boxes * jackets_per_box

theorem total_winter_clothing :
    num_scarves 4 8 + num_mittens 3 6 + num_hats 2 5 + num_jackets 1 3 = 63 :=
by
  -- The proof will use the given definitions and calculate the total
  sorry

end total_winter_clothing_l417_417288


namespace triangle_area_lt_sqrt3_div3_l417_417295

theorem triangle_area_lt_sqrt3_div3
  (A B C : Type) [euclidean_space A B C]
  (lA lB lC : ℝ) -- lengths of angle bisectors from vertices A, B, and C
  (h1 : lA < 1)
  (h2 : lB < 1)
  (h3 : lC < 1) :
  area (triangle A B C) < (Real.sqrt 3 / 3) :=
sorry

end triangle_area_lt_sqrt3_div3_l417_417295


namespace solution_y_eq_2_l417_417567

theorem solution_y_eq_2 (y : ℝ) (h_pos : y > 0) (h_eq : y^6 = 64) : y = 2 :=
sorry

end solution_y_eq_2_l417_417567


namespace time_after_2004_hours_l417_417329

variable (h : ℕ) 

-- Current time is represented as an integer from 0 to 11 (9 o'clock).
def current_time : ℕ := 9

-- 12-hour clock cycles every 12 hours.
def cycle : ℕ := 12

-- Time after 2004 hours.
def hours_after : ℕ := 2004

-- Proof statement
theorem time_after_2004_hours (h : ℕ) :
  (current_time + hours_after) % cycle = current_time := 
sorry

end time_after_2004_hours_l417_417329


namespace triangle_side_ratio_l417_417633

theorem triangle_side_ratio 
  (A B C : Point)
  (O I : Point)
  (hO : circumcenter O A B C)
  (hI : incenter I A B C)
  (hAngleAIO : angle A I O = 90)
  (hAngleCIO : angle C I O = 45) :
  side_ratio A B C = 3:4:5 :=
sorry

end triangle_side_ratio_l417_417633


namespace num_of_prime_sums_eq_six_l417_417322

def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_sequence_sum (n : ℕ) : ℕ :=
if n = 1 then 2
else Nat.recOn n 5 (λ k s, s + Nat.succ (2 * k))

theorem num_of_prime_sums_eq_six : 
  (∃ (f : ℕ → ℕ), (∀ n, f n = prime_sequence_sum n) ∧ 
  (f 1 = 2) ∧ 
  (f 2 = 5) ∧
  (f 3 = 10) ∧
  (f 4 = 17) ∧
  (f 5 = 28) ∧
  (f 6 = 41) ∧
  (f 7 = 58) ∧
  (f 8 = 77) ∧
  (f 9 = 100) ∧
  (f 10 = 129) ∧
  (f 11 = 160) ∧
  (f 12 = 197) ∧
  (f 13 = 238) ∧
  (f 14 = 281) ∧
  (f 15 = 328)) ∧ 
  ((is_prime (prime_sequence_sum 1)) +
  (is_prime (prime_sequence_sum 2)) +
  (is_prime (prime_sequence_sum 4)) +
  (is_prime (prime_sequence_sum 6)) +
  (is_prime (prime_sequence_sum 12)) +
  (is_prime (prime_sequence_sum 14))) = 6 :=
by sorry

end num_of_prime_sums_eq_six_l417_417322


namespace prime_triples_perfect_square_l417_417137

theorem prime_triples_perfect_square (p q r : ℕ) (hp : prime p) (hq : prime q) (hr : prime r) :
  ∃ k : ℕ, p^q + p^r = k^2 ↔ 
  (p, q, r) = (2, 5, 2) ∨ (p, q, r) = (2, 2, 5) ∨ (p, q, r) = (2, 3, 3) ∨ (p, q, r) = (3, 3, 2) ∨ 
  ∃ n : ℕ, (p, q, r) = (2, 2*n+1, 2*n+1) :=
sorry

end prime_triples_perfect_square_l417_417137


namespace coefficient_x_squared_in_binomial_expansion_l417_417910

-- Define the integral condition
def integral_x_squared : ℝ := ∫ x in -1..1, x^2

-- Prove that for n derived from the integral, the coefficient of x^2 in (x - 2/x)^n is 60
theorem coefficient_x_squared_in_binomial_expansion : integral_x_squared = 2 / 3 → 
  let n := 6 in  -- Correcting based on assumed step correction
  ∀ n = 6, 
    (∀ r : ℕ, r = 2 → binomial 6 r * (-2)^r = 60) :=
by
  admit

end coefficient_x_squared_in_binomial_expansion_l417_417910


namespace sequence_non_positive_l417_417163

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h₀ : a 0 = 0) 
  (hₙ : a n = 0)
  (h : ∀ k, 1 ≤ k ∧ k < n → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) : 
  ∀ k, k ≤ n → a k ≤ 0 :=
by
  sorry

end sequence_non_positive_l417_417163


namespace quadratic_root_k_eq_one_l417_417502

theorem quadratic_root_k_eq_one
  (k : ℝ)
  (h₀ : (k + 3) ≠ 0)
  (h₁ : ∃ x : ℝ, (x = 0) ∧ ((k + 3) * x^2 + 5 * x + k^2 + 2 * k - 3 = 0)) :
  k = 1 :=
by
  sorry

end quadratic_root_k_eq_one_l417_417502


namespace probability_sum_greater_than_8_l417_417290

open Set

def all_outcomes : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} }

def favorable_outcomes : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ x + y > 8 }

def probability_of_event : ℚ :=
  (favorable_outcomes.toFinset.card : ℚ) / (all_outcomes.toFinset.card : ℚ)

theorem probability_sum_greater_than_8 :
  probability_of_event = 5 / 18 :=
sorry

end probability_sum_greater_than_8_l417_417290


namespace shopkeeper_loss_l417_417427

-- Define the initial conditions
def total_apples : ℝ := 120
def sold_40_percent : ℝ := 0.40 * total_apples
def sold_30_percent : ℝ := 0.30 * total_apples
def sold_20_percent : ℝ := 0.20 * total_apples
def unsold_percent : ℝ := 0.10 * total_apples

def profit_40_percent : ℝ := 0.25 * sold_40_percent
def profit_30_percent : ℝ := 0.35 * sold_30_percent
def profit_20_percent : ℝ := 0.20 * sold_20_percent

def total_profit : ℝ := profit_40_percent + profit_30_percent + profit_20_percent
def additional_expenses : ℝ := 20
def cost_unsold_apples : ℝ := unsold_percent * 1  -- assuming $1 per kg cost

def effective_profit : ℝ := total_profit - cost_unsold_apples - additional_expenses

def total_cost_price_sold_apples : ℝ := sold_40_percent + sold_30_percent + sold_20_percent

def effective_percentage_loss : ℝ := (effective_profit / total_cost_price_sold_apples) * 100

-- Define the theorem to prove the effective percentage loss
theorem shopkeeper_loss : effective_percentage_loss ≈ -2.407 : ℝ :=
by sorry

end shopkeeper_loss_l417_417427


namespace length_NG_l417_417300

theorem length_NG (E F G H N : Point) (EFGH_square : Square E F G H)
  (diameter_EF : Length E F = 10) (N_midpoint_arc_EF : MidpointArc N E F) :
  Length N G = 5 * Real.sqrt 10 := 
sorry

end length_NG_l417_417300


namespace vasya_tolya_badges_l417_417767

theorem vasya_tolya_badges (x y : ℤ)
    (h1 : y = x + 5) -- Vasya initially had 5 more badges than Tolya
    (h2 : (y - (6 * (y / 25) / 25) + (4 * x) / 25) = (x - (4 * x) / 5 + 6 * (y / 25) / 5 - 1)) : -- equation balancing after exchange
    x = 45 ∧ y = 50 := 
sorry

end vasya_tolya_badges_l417_417767


namespace new_average_daily_production_l417_417899

theorem new_average_daily_production 
  (n : ℕ) 
  (avg_past_n_days : ℕ) 
  (today_production : ℕ)
  (new_avg_production : ℕ)
  (hn : n = 5) 
  (havg : avg_past_n_days = 60) 
  (htoday : today_production = 90) 
  (hnew_avg : new_avg_production = 65)
  : (n + 1 = 6) ∧ ((n * 60 + today_production) = 390) ∧ (390 / 6 = 65) :=
by
  sorry

end new_average_daily_production_l417_417899


namespace unique_factorization_l417_417639

theorem unique_factorization (m : ℕ) (h1 : m > 0) (h2 : m % 4 = 2) :
  ∃! (ab : ℕ × ℕ), 
  let a := ab.1 
  let b := ab.2 
  a > 0 ∧ b > 0 ∧ m = a * b ∧ 0 < a - b ∧ a - b < Real.sqrt(5 + 4 * Real.sqrt(4 * m + 1)) :=
sorry

end unique_factorization_l417_417639


namespace win_sector_area_l417_417812

theorem win_sector_area (r : ℝ) (P_win : ℝ) (h_r : r = 5) (h_P_win : P_win = 2 / 5) : 
  ∃ A_WIN : ℝ, A_WIN = 10 * π :=
by
  -- given conditions
  have h_total_area : 25 * π = π * r^2,
  -- A_WIN sector is proportionate to P_win
  have h_ratio : P_win = A_WIN / (π * r^2),
  -- solve for A_WIN
  sorry

end win_sector_area_l417_417812


namespace probability_four_white_balls_l417_417075

-- Definitions from the problem conditions
def whiteBalls : ℕ := 7
def blackBalls : ℕ := 8
def totalBalls : ℕ := whiteBalls + blackBalls
def ballsToDraw : ℕ := 5
def whiteBallsToDraw : ℕ := 4
def blackBallsToDraw : ℕ := 1

-- Lean statement for the probability question
theorem probability_four_white_balls :
  (Nat.choose totalBalls ballsToDraw = 3003) →
  (Nat.choose whiteBalls whiteBallsToDraw = 35) →
  (Nat.choose blackBalls blackBallsToDraw = 8) →
  (280 : ℚ / 3003) =
    (Nat.choose whiteBalls whiteBallsToDraw * Nat.choose blackBalls blackBallsToDraw : ℚ)
    / (Nat.choose totalBalls ballsToDraw : ℚ) :=
by
  intros
  sorry

end probability_four_white_balls_l417_417075


namespace required_connections_l417_417749

theorem required_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 := by
  sorry

end required_connections_l417_417749


namespace ratio_of_areas_l417_417953

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) : ℝ := 
  let s := semi_perimeter a b c
  in real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem ratio_of_areas (a b c d e f : ℝ) (h₁ : a = 18) (h₂ : b = 18) (h₃ : c = 12)
  (h₄ : d = 18) (h₅ : e = 18) (h₆ : f = 30) :
  area a b c / area d e f = 102 / 149.25 := by
  rw [h₁, h₂, h₃, h₄, h₅, h₆]
  -- calculations will be done here --
  sorry

end ratio_of_areas_l417_417953


namespace birds_flew_up_l417_417070

theorem birds_flew_up (original_birds new_total_birds : ℕ) (h1 : original_birds = 231) (h2 : new_total_birds = 312) : new_total_birds - original_birds = 81 := by
  rw [h1, h2]
  norm_num

end birds_flew_up_l417_417070


namespace sum_of_squares_interval_l417_417259

theorem sum_of_squares_interval (A : ℝ) (hA : 0 < A) :
  ∃ (S : set ℝ), S = {s | ∃ (x : ℕ → ℝ), (∀ n, 0 < x n) ∧ (∑' n, x n = A) ∧ (s = ∑' n, (x n)^2) } ∧ S = set.Ioo 0 (A^2) :=
sorry

end sum_of_squares_interval_l417_417259


namespace minimize_AP_BP_CP_squared_l417_417493

noncomputable def minimize_distance (α β : ℝ^3) : ℝ^3 :=
  (1 / 3) • (α + β)

theorem minimize_AP_BP_CP_squared (α β : ℝ^3) :
  ∃ p : ℝ^3, p = (1 / 3) • (α + β) ∧
  ∀ q : ℝ^3, 
    let AP := q - α in
    let BP := q - β in
    let CP := q in
    (AP.norm_sq + BP.norm_sq + CP.norm_sq) ≥ 
    let p := (1 / 3) • (α + β) in
    let AP := p - α in
    let BP := p - β in
    let CP := p in
    (AP.norm_sq + BP.norm_sq + CP.norm_sq) := sorry

end minimize_AP_BP_CP_squared_l417_417493


namespace switches_connections_l417_417743

theorem switches_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 :=
by
  sorry

end switches_connections_l417_417743


namespace bucket_fill_turns_l417_417458

-- Definitions
variables (Q : ℚ) (capacityP capacityQ capacityR drumCapacity turns : ℚ)

-- Conditions
def capacities : Prop :=
  capacityP = 3 * capacityQ ∧
  capacityR = (1/2) * capacityQ ∧
  drumCapacity = 80 * capacityP

theorem bucket_fill_turns (h : capacities Q capacityP capacityQ capacityR drumCapacity) : 
  let combined_capacity := 3 * capacityQ + capacityQ + (1/2) * capacityQ in
  turns = drumCapacity / combined_capacity → 
  turns = 54 := 
by 
  intros _ 
  sorry

end bucket_fill_turns_l417_417458


namespace overall_percentage_supporting_increased_funding_l417_417830

-- Definitions for the conditions
def percent_of_men_supporting (percent_men_supporting : ℕ := 60) : ℕ := percent_men_supporting
def percent_of_women_supporting (percent_women_supporting : ℕ := 80) : ℕ := percent_women_supporting
def number_of_men_surveyed (men_surveyed : ℕ := 100) : ℕ := men_surveyed
def number_of_women_surveyed (women_surveyed : ℕ := 900) : ℕ := women_surveyed

-- Theorem: the overall percent of people surveyed who supported increased funding is 78%
theorem overall_percentage_supporting_increased_funding : 
  (percent_of_men_supporting * number_of_men_surveyed + percent_of_women_supporting * number_of_women_surveyed) / 
  (number_of_men_surveyed + number_of_women_surveyed) = 78 := 
sorry

end overall_percentage_supporting_increased_funding_l417_417830


namespace intersection_points_of_y_eq_const_with_y_eq_3x_minus_x_cubed_l417_417217

theorem intersection_points_of_y_eq_const_with_y_eq_3x_minus_x_cubed 
(m : ℝ) : 
  (∃ three_distinct_points : set ℝ, your_exists_explanation_here) ↔ (-2 < m ∧ m < 2) := by
sorry

end intersection_points_of_y_eq_const_with_y_eq_3x_minus_x_cubed_l417_417217


namespace time_to_pick_sugar_snap_peas_l417_417852

theorem time_to_pick_sugar_snap_peas (pea_count1 pea_count2 : ℕ) (time1 : ℕ) :
  (pea_count1 = 56 ∧ time1 = 7 ∧ pea_count2 = 72) →
  let rate := pea_count1 / time1 in
  (pea_count2 / rate = 9) :=
by
  intros h
  let ⟨h1, h2, h3⟩ := h
  sorry

end time_to_pick_sugar_snap_peas_l417_417852


namespace solution_set_of_inequality_l417_417515

-- Assuming f is a differentiable function on ℝ with certain properties
variable (f : ℝ → ℝ)
variable [differentiable ℝ f]

-- Given conditions:
--   1. f(0) = 2018
--   2. ∀ x ∈ ℝ, f(x) > f''(x)
theorem solution_set_of_inequality (h₀ : f 0 = 2018) 
                                   (h₁ : ∀ x : ℝ, f x > (deriv (deriv f)) x) :
                                   {x : ℝ | f x < 2018 * Real.exp x} = Ioi 0 :=
sorry

end solution_set_of_inequality_l417_417515


namespace exactly_one_true_proposition_l417_417627

-- Definitions of lines and planes
variables {m n : Line} {α β γ : Plane}

-- Propositions as conditions
def prop1 : Prop := m ⊆ α ∧ n ∥ α → m ∥ n
def prop2 : Prop := α ∥ β ∧ β ∥ γ ∧ m ⊥ α → m ⊥ γ
def prop3 : Prop := α ∩ β = n ∧ m ∥ n → m ∥ α ∧ m ∥ β
def prop4 : Prop := α ⊥ γ ∧ β ⊥ γ → α ∥ β

-- Theorem stating there is only one true proposition among the four
theorem exactly_one_true_proposition : 
  (¬ prop1 ∧ prop2 ∧ ¬ prop3 ∧ ¬ prop4) → 
  (prop2 ∧ ¬(prop1 ∨ prop3 ∨ prop4)) :=
begin
  intros h,
  sorry
end

end exactly_one_true_proposition_l417_417627


namespace exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l417_417544

variable (m : ℝ)
def f (x : ℝ) : ℝ := x^2 + m*x + 1

theorem exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2 :
  (∃ x0 : ℝ, x0 > 0 ∧ f m x0 < 0) → m < -2 := by
  sorry

end exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l417_417544


namespace sum_of_numbers_l417_417323

theorem sum_of_numbers (x : ℝ) (h : x^2 + (2 * x)^2 + (4 * x)^2 = 4725) : 
  x + 2 * x + 4 * x = 105 := 
sorry

end sum_of_numbers_l417_417323


namespace quadratic_function_positive_l417_417465

theorem quadratic_function_positive (a m : ℝ) (h : a > 0) (h_fm : (m^2 + m + a) < 0) : (m + 1)^2 + (m + 1) + a > 0 :=
by sorry

end quadratic_function_positive_l417_417465


namespace boat_distance_l417_417720

noncomputable def distance_covered (speed_still_water : ℝ) (speed_current : ℝ) (time_seconds : ℝ) : ℝ :=
  let effective_speed := (speed_still_water + speed_current) * 1000 / 3600 -- convert kmph to m/s
  effective_speed * time_seconds

theorem boat_distance (speed_still_water : ℝ) (speed_current : ℝ) (time_seconds : ℝ) :
  speed_still_water = 20 ∧ speed_current = 3 ∧ time_seconds = 9.390553103577801 →
  distance_covered speed_still_water speed_current time_seconds ≈ 60 :=
by
  intros h
  obtain ⟨h1, h2, h3⟩ := h
  have H : distance_covered speed_still_water speed_current time_seconds = 6.388888888888889 * 9.390553103577801 := rfl
  rw [h1, h2, h3] at H
  sorry -- calculation steps are skipped

end boat_distance_l417_417720


namespace range_of_s_l417_417148

noncomputable def s (n : ℕ) : ℕ := 
  if n = 1 then 0 
  else (multiset.to_finset (nat.factors n)).sum (λ p, (p * p) * (multiset.count p (nat.factors n)))

theorem range_of_s :
  {m : ℕ | ∃ n : ℕ, n > 1 ∧ ¬ nat.prime n ∧ s n = m} = {k | k ≥ 8 ∧ k % 2 = 0} :=
sorry

end range_of_s_l417_417148


namespace max_AC_not_RS_l417_417230

theorem max_AC_not_RS (TotalCars NoACCars MinRS MaxACnotRS : ℕ)
  (h1 : TotalCars = 100)
  (h2 : NoACCars = 49)
  (h3 : MinRS >= 51)
  (h4 : (TotalCars - NoACCars) - MinRS = MaxACnotRS)
  : MaxACnotRS = 0 :=
by
  sorry

end max_AC_not_RS_l417_417230


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l417_417299

-- (x+2)^2 = 2x+4 has solutions x = 0 and x = -2
theorem solve_eq1 (x : ℝ) : (x+2)^2 = 2x+4 ↔ x = 0 ∨ x = -2 := by
  sorry

-- x^2 - 2x - 5 = 0 has solutions x = 1 + sqrt(6) and x = 1 - sqrt(6)
theorem solve_eq2 (x : ℝ) : x^2 - 2x - 5 = 0 ↔ x = 1 + Real.sqrt 6 ∨ x = 1 - Real.sqrt 6 := by
  sorry

-- x^2 - 5x - 6 = 0 has solutions x = -1 and x = 6
theorem solve_eq3 (x : ℝ) : x^2 - 5x - 6 = 0 ↔ x = -1 ∨ x = 6 := by
  sorry

-- (x+3)^2 = (1-2x)^2 has solutions x = -2/3 and x = 4
theorem solve_eq4 (x : ℝ) : (x+3)^2 = (1-2x)^2 ↔ x = -2/3 ∨ x = 4 := by
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l417_417299


namespace biscuits_per_dog_l417_417652

-- Define constants for conditions
def total_biscuits : ℕ := 6
def number_of_dogs : ℕ := 2

-- Define the statement to prove
theorem biscuits_per_dog : total_biscuits / number_of_dogs = 3 := by
  -- Calculation here
  sorry

end biscuits_per_dog_l417_417652


namespace find_b_value_l417_417670

-- Definition of the problem conditions and the statement to prove
theorem find_b_value (b : ℝ) :
  let total_area := 7.0,
      region_area := total_area / 2,
      line_eq := (λ x : ℝ, (5 / (5 - b)) * (x - b)),
      triangle_base := 5 - b,
      triangle_height := 5,
      triangle_area := (1 / 2) * triangle_base * triangle_height
  in triangle_area = region_area ->
     b = (18 / 5) :=
by
  intros total_area region_area line_eq triangle_base triangle_height triangle_area h
  sorry

end find_b_value_l417_417670


namespace marshmallow_challenge_l417_417559

noncomputable def haley := 8
noncomputable def michael := 3 * haley
noncomputable def brandon := (1 / 2) * michael
noncomputable def sofia := 2 * (haley + brandon)
noncomputable def total := haley + michael + brandon + sofia

theorem marshmallow_challenge : total = 84 :=
by
  sorry

end marshmallow_challenge_l417_417559


namespace complement_correct_l417_417620

-- Define the universal set U
def U : Set ℤ := {x | -2 < x ∧ x ≤ 3}

-- Define the set A
def A : Set ℤ := {3}

-- Define the complement of A with respect to U
def complement_U_A : Set ℤ := {x | x ∈ U ∧ x ∉ A}

theorem complement_correct : complement_U_A = { -1, 0, 1, 2 } :=
by
  sorry

end complement_correct_l417_417620


namespace min_max_abs_expr_l417_417882

theorem min_max_abs_expr : 
  (minimize (λ y : ℝ, maximize (λ x : ℝ, abs (x^2 - x * y / 3)) (0 ≤ x ∧ x ≤ 1)) (y ∈ ℝ)) = 3 / 4 :=
sorry

end min_max_abs_expr_l417_417882


namespace solution_set_of_inequality_l417_417083

theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (odd_f : ∀ x, f (-x) = -f x)
  (deriv_f_lt_f : ∀ x, deriv f x < f x)
  (func_cond : ∀ x, f x * f (x + 3) = -1)
  (f_at_2015 : f 2015 = - real.exp 1) : 
  {x : ℝ | f x < real.exp x} = {0} ∪ set.Ioi 1 :=
by
  sorry

end solution_set_of_inequality_l417_417083


namespace equilateral_triangle_area_l417_417448

theorem equilateral_triangle_area (p : ℝ) (h : p > 0) : 
  ∃ (A : ℝ), A = (sqrt 3 / 4) * p^2 :=
by 
  sorry

end equilateral_triangle_area_l417_417448


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417034

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417034


namespace total_routes_from_Mathville_to_Algebratown_l417_417062

def routes (start end : String) (edges : List (String × String)) : Nat :=
  List.foldl (λ acc (start, finish) =>
    if (start = "M" ∧ finish = "A") ∨ 
       (start = "M" ∧ finish = "B") ∨ 
       (start = "M" ∧ finish = "C") ∨ 
       (start = "M" ∧ finish = "D") ∨ 
       (start = "B" ∧ finish = "E") ∨ 
       (start = "C" ∧ finish = "D") ∨ 
       (start = "C" ∧ finish = "F") ∨ 
       (start = "D" ∧ finish = "F") ∨ 
       (start = "D" ∧ finish = "E") ∨ 
       (start = "E" ∧ finish = "A") ∨ 
       (start = "F" ∧ finish = "A") 
    then acc + 1 else acc)
  0 edges

theorem total_routes_from_Mathville_to_Algebratown :
  routes "M" "A" [("M", "C"), ("M", "B"), ("M", "D"), ("B", "E"), ("C", "D"), ("C", "F"), 
                    ("D", "F"), ("D", "E"), ("E", "A"), ("F", "A")] = 8 := 
sorry

end total_routes_from_Mathville_to_Algebratown_l417_417062


namespace hyperbola_focal_length_l417_417174

-- Define the given conditions
def hyperbola_eq (x y m : ℝ) := (x^2 / m) - y^2 = 1
def asymptote_eq (x y m : ℝ) := sqrt 3 * x + m * y = 0

-- Prove that the focal length is 4 given the conditions
theorem hyperbola_focal_length (m : ℝ) (hm : m > 0) :
  (∀ x y, hyperbola_eq x y m) →
  (∀ x y, asymptote_eq x y m) →
  (2 * sqrt (m + 1) = 4) :=
by
  intros; sorry

end hyperbola_focal_length_l417_417174


namespace nonagon_arithmetic_mean_property_l417_417995

def is_equilateral_triangle (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  (j = (i + 3) % 9) ∧ (k = (i + 6) % 9)

def is_arithmetic_mean (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  A j = (A i + A k) / 2

theorem nonagon_arithmetic_mean_property :
  ∀ (A : Fin 9 → ℤ),
    (∀ i, A i = 2016 + i) →
    (∀ i j k : Fin 9, is_equilateral_triangle A i j k → is_arithmetic_mean A i j k) :=
by
  intros
  sorry

end nonagon_arithmetic_mean_property_l417_417995


namespace geom_prog_common_ratio_l417_417224

theorem geom_prog_common_ratio {a r : ℝ} 
  (h1 : ∀ (n : ℕ), a * r^(n-1) - a * r^n = 1/2 * (a * r^n + a * r^(n+1))) :
  r = (real.sqrt 17 - 3) / 2 := 
begin
  sorry
end

end geom_prog_common_ratio_l417_417224


namespace smallest_integer_10403_l417_417047

/-- 
The smallest positive integer that is neither prime nor a perfect square and has no prime factors less than 100 is 10403.
-/
theorem smallest_integer_10403 :
  ∃ n : ℕ, 
  n > 0 ∧ 
  ¬ nat.prime n ∧ 
  ¬ ∃ m : ℕ, m * m = n ∧ 
  (∀ p : ℕ, nat.prime p → p ∣ n → 100 < p) ∧ 
  n = 10403 ∧ 
  ∀ m : ℕ, m > 0 → 
  ¬ nat.prime m → 
  ¬ ∃ k : ℕ, k * k = m → 
  (∀ p : ℕ, nat.prime p → p ∣ m → 100 < p) → 
  m ≥ n :=
sorry

end smallest_integer_10403_l417_417047


namespace max_cookies_eaten_l417_417842

def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem max_cookies_eaten 
  (total_cookies : ℕ)
  (andy_cookies : ℕ)
  (alexa_cookies : ℕ)
  (hx : andy_cookies + alexa_cookies = total_cookies)
  (hp : ∃ p : ℕ, prime p ∧ alexa_cookies = p * andy_cookies)
  (htotal : total_cookies = 30) :
  andy_cookies = 10 :=
  sorry

end max_cookies_eaten_l417_417842


namespace probability_odd_80_heads_l417_417104

noncomputable def coin_toss_probability_odd (n : ℕ) (p : ℝ) : ℝ :=
  (1 / 2) * (1 - (1 / 3^n))

theorem probability_odd_80_heads :
  coin_toss_probability_odd 80 (3 / 4) = (1 / 2) * (1 - 1 / 3^80) :=
by
  sorry

end probability_odd_80_heads_l417_417104


namespace solve_inequality_l417_417675

theorem solve_inequality (x : ℝ) :
  (11/9 < x ∧ x < 7/3) ↔ (| (3*x - 2) / (2*x - 3) | > 3) :=
begin
  sorry
end

end solve_inequality_l417_417675


namespace div_by_m_impl_div_by_m_factorial_exists_poly_div_by_m_l417_417423

-- Definition of integer polynomial
structure Polynomial (R : Type _) [CommRing R] := 
(coeff : ℕ → R)
(degree : ℕ)

-- Convert polynomial to function
def eval {R : Type _} [CommRing R] (p : Polynomial R) (x : R) : R := 
  ∑ i in Finset.range (p.degree + 1), p.coeff i * x ^ i

-- Prove first statement
theorem div_by_m_impl_div_by_m_factorial 
  (p : Polynomial ℤ) (m : ℤ) 
  (h : ∀ x : ℤ, m ∣ eval p x) : 
  m ∣ (Nat.factorial p.degree) * p.coeff p.degree := 
sorry

-- Prove second statement
theorem exists_poly_div_by_m 
  (a0 k m : ℤ) 
  (h : m ∣ (Nat.factorial (Int.toNat k) * a0)) : 
  ∃ p : Polynomial ℤ,  
  p.coeff p.degree = a0 ∧ 
  k = p.degree ∧ 
  ∀ x : ℤ, m ∣ eval p x := 
sorry

end div_by_m_impl_div_by_m_factorial_exists_poly_div_by_m_l417_417423


namespace password_lock_probability_l417_417420

theorem password_lock_probability :
  (∃ (A B C : ℕ), 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 
   (∃ password : ℕ × ℕ × ℕ, password = (A, B, C) ∧ 
   (∃ chosen_middle_digit : ℕ, 0 <= chosen_middle_digit ∧ chosen_middle_digit <= 9 ∧
   chosen_middle_digit = B))) →
  (1/10 : ℝ) = 0.1 :=
by 
  sorry

end password_lock_probability_l417_417420


namespace michael_sarah_games_count_l417_417860

/-- 
Given: 
- Twelve players, including Michael and Sarah.
- Each day at recess, the twelve players split into two four-square games, each with six players.
- Over the course of a semester, each possible match-up of six players occurs once.

Prove:
- The number of times Michael and Sarah play in the same game is 210.
-/
theorem michael_sarah_games_count :
  let players := 12 in
  let michael_sarah := 2 in
  let other_players := players - michael_sarah in
  let player_combinations := Nat.choose 12 6 in
  let games_with_sarah := player_combinations / 2 in
  let games_michael_sarah := (games_with_sarah * 5) / 11 in
  games_michael_sarah = 210 := 
by 
  sorry

end michael_sarah_games_count_l417_417860


namespace trajectory_no_intersection_distance_AB_l417_417157

variable (M : Type) [MetricSpace M]

-- Point M on the plane
variable (M : ℝ × ℝ)

-- Given conditions
def condition1 (M : ℝ × ℝ) : Prop := 
  (Real.sqrt ((M.1 - 8)^2 + M.2^2) = 2 * Real.sqrt ((M.1 - 2)^2 + M.2^2))

-- 1. Proving the trajectory C of M
theorem trajectory (M : ℝ × ℝ) (h : condition1 M) : M.1^2 + M.2^2 = 16 :=
by
  sorry

-- 2. Range of values for k such that y = kx - 5 does not intersect trajectory C
theorem no_intersection (k : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 16 → y ≠ k * x - 5) ↔ (-3 / 4 < k ∧ k < 3 / 4) :=
by
  sorry

-- 3. Distance between intersection points A and B of given circles
def intersection_condition (x y : ℝ) : Prop :=
  (x^2 + y^2 = 16) ∧ (x^2 + y^2 - 8 * x - 8 * y + 16 = 0)

theorem distance_AB (A B : ℝ × ℝ) (hA : intersection_condition A.1 A.2) (hB : intersection_condition B.1 B.2) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 2 :=
by
  sorry

end trajectory_no_intersection_distance_AB_l417_417157


namespace apothem_plane_angle_sine_l417_417801

-- Given definitions
variables (a : ℝ)
def centroid_of_equilateral_triangle (a : ℝ) : ℝ := a * sqrt 3 / 3
def midpoint_of_side (a : ℝ) : ℝ := a * sqrt 3 / 6
def length_apothem (a : ℝ) : ℝ := sqrt (a^2 + (a * sqrt 3 / 6)^2)
def perpendicular_distance (a : ℝ) : ℝ := a * sqrt 3 / 4
def length_KD (a : ℝ) : ℝ := (a * sqrt 3 / 4) / 2

-- Given pyramid and angle conditions
variables (apothem_angle : ℝ)
def sin_phi (a : ℝ) : ℝ := (a * sqrt 3 / 8) / (a * sqrt 13 / 2 / sqrt 3)

-- Goal to prove
theorem apothem_plane_angle_sine : 
  sin_phi a = (3 * sqrt 3) / 13 :=
by
  sorry

end apothem_plane_angle_sine_l417_417801


namespace triangle_area_calc_l417_417386

def perimeter : ℝ := 48
def inradius : ℝ := 2.5
def area (p r : ℝ) : ℝ := r * p / 2

theorem triangle_area_calc : area perimeter inradius = 60 := by
  sorry

end triangle_area_calc_l417_417386


namespace decreasing_function_implies_inequality_l417_417317

theorem decreasing_function_implies_inequality (k b : ℝ) (h : ∀ x : ℝ, (2 * k + 1) * x + b = (2 * k + 1) * x + b) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k + 1) * x1 + b > (2 * k + 1) * x2 + b) → k < -1/2 :=
by sorry

end decreasing_function_implies_inequality_l417_417317


namespace sunzi_oranges_l417_417066

theorem sunzi_oranges :
  ∃ (a : ℕ), ( 5 * a + 10 * 3 = 60 ) ∧ ( ∀ n, n = 0 → a = 6 ) :=
by
  sorry

end sunzi_oranges_l417_417066


namespace switch_connections_l417_417747

theorem switch_connections (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 :=
by
  rw [h1, h2]
  norm_num
  sorry

end switch_connections_l417_417747


namespace reflection_property_l417_417664

variable {V : Type*} [InnerProductSpace ℝ V]

def reflection (a b : V) : V :=
  2 * (b ⬝ (a, b) / (b ⬝ b)) - a

theorem reflection_property (a b : V) :
  reflection a b = 2 * b * (inner a b / inner b b) - a :=
sorry

end reflection_property_l417_417664


namespace f_is_odd_max_min_f_l417_417267

noncomputable theory

variable (f : ℝ → ℝ)

-- Conditions
def C1 := ∀ x y : ℝ, f (x + y) = f x + f y
def C2 := ∀ x : ℝ, x > 0 → f x < 0
def C3 := f 1 = -2

-- Proof statements
theorem f_is_odd {f : ℝ → ℝ} (h1 : C1 f) (h2 : C2 f) (h3 : C3 f) : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem max_min_f {f : ℝ → ℝ} (h1 : C1 f) (h2 : C2 f) (h3 : C3 f) : 
  ∃ max min : ℝ, max = f (-3) ∧ min = f 3 ∧ max = 6 ∧ min = -6 :=
sorry

end f_is_odd_max_min_f_l417_417267


namespace certain_number_eq_l417_417811

theorem certain_number_eq :
  ∃ y : ℝ, y + (y * 4) = 48 ∧ y = 9.6 :=
by
  sorry

end certain_number_eq_l417_417811


namespace negative_number_in_options_l417_417100

def options : List ℝ := [-1, 0, 2, Real.sqrt 2]

theorem negative_number_in_options : ∃ (x : ℝ), x ∈ options ∧ x < 0 := by
  sorry

end negative_number_in_options_l417_417100


namespace find_teacher_age_l417_417385

noncomputable def age_of_teacher (avg_age_students : ℕ) (num_students : ℕ) 
                                (avg_age_inclusive : ℕ) (num_people_inclusive : ℕ) : ℕ :=
  let total_age_students := num_students * avg_age_students
  let total_age_inclusive := num_people_inclusive * avg_age_inclusive
  total_age_inclusive - total_age_students

theorem find_teacher_age : age_of_teacher 15 10 16 11 = 26 := 
by 
  sorry

end find_teacher_age_l417_417385


namespace area_ABC_is_24sqrt3_l417_417922

noncomputable def area_ABC : ℝ :=
  let A : Fin 3 → ℝ := ![0, 0]
  let B : Fin 3 → ℝ := ![8 * Real.sqrt 3, 0]
  let C : Fin 3 → ℝ := ![4 * Real.sqrt 3, 12]
  1/2 * Real.abs (B 0 * C 1 - C 0 * B 1)

theorem area_ABC_is_24sqrt3 : 
  ∀ (A B C : Fin 3 → ℝ), 
    (A 0 = 0 ∧ A 1 = 0) ∧ 
    (B 0 = 8 * Real.sqrt 3 ∧ B 1 = 0) ∧ 
    (C 0 = 4 * Real.sqrt 3 ∧ C 1 = 12) →
    area_ABC = 24 * Real.sqrt 3 :=
by
  sorry

end area_ABC_is_24sqrt3_l417_417922


namespace add_complex_eq_result_l417_417374

-- Definitions

def complex_add (a b : ℂ) : ℂ := complex.add a b
def complex_sub (a b : ℂ) : ℂ := complex.sub a b
constant i : ℂ
axiom i_squared : i ^ 2 = -1

-- Problem statement
theorem add_complex_eq_result : complex_add (6 - 3 * i) (-7 + 12 * i) = -1 + 9 * i := by
  sorry

end add_complex_eq_result_l417_417374


namespace closest_whole_number_to_shaded_area_l417_417831

theorem closest_whole_number_to_shaded_area :
  let rectangle_area := 4 * 6
  let circle_radius := 2 / 2
  let circle_area := Real.pi * circle_radius^2
  let shaded_area := rectangle_area - circle_area
  20 <= shaded_area ∧ shaded_area < 21.5 := 
by
  let rectangle_area := 4 * 6
  let circle_radius := 2 / 2
  let circle_area := Real.pi * circle_radius^2
  let shaded_area := rectangle_area - circle_area
  have h : 20.86 < 21.5, by norm_num
  have h1 : 20.86 >= 20, by norm_num
  split
  exact h1
  exact h

end closest_whole_number_to_shaded_area_l417_417831


namespace books_sold_l417_417708

theorem books_sold (total_books : ℕ) (fraction_left : ℚ)
  (h1 : total_books = 9900)
  (h2 : fraction_left = 4/6) :
  (total_books - (total_books * fraction_left).natAbs) = 3300 :=
by sorry

end books_sold_l417_417708


namespace tangent_line_eq_l417_417429

noncomputable def equation_of_tangent_line (x y : ℝ) : Prop := 
  ∃ k : ℝ, (y = k * (x - 2) + 2) ∧ 2 * x + y - 6 = 0

theorem tangent_line_eq :
  ∀ (x y : ℝ), 
    (y = 2 / (x - 1)) ∧ (∃ (a b : ℝ), (a, b) = (1, 4)) ->
    equation_of_tangent_line x y :=
by
  sorry

end tangent_line_eq_l417_417429


namespace selection_methods_count_l417_417504

theorem selection_methods_count
  (multiple_choice_questions : ℕ)
  (fill_in_the_blank_questions : ℕ)
  (h1 : multiple_choice_questions = 9)
  (h2 : fill_in_the_blank_questions = 3) :
  multiple_choice_questions + fill_in_the_blank_questions = 12 := by
  sorry

end selection_methods_count_l417_417504


namespace minimum_m_plus_n_l417_417188

theorem minimum_m_plus_n
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_ellipse : 1 / m + 4 / n = 1) :
  m + n = 9 :=
sorry

end minimum_m_plus_n_l417_417188


namespace magnitude_of_a_angle_between_a_and_b_l417_417925

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V] 

-- Definitions based on the conditions
variables (a b : V)
variables (θ : ℝ)
variables (non_zero_a : a ≠ 0) (non_zero_b : b ≠ 0)
variables (norm_b_eq : ∥b∥ = real.sqrt 2)
variables (dot_product_eq : inner (a - b) (a + b) = 1 / 4)
variable (a_dot_b : inner a b = 3 / 2)

-- Part I: Prove |a| = 3/2
theorem magnitude_of_a : ∥a∥ = 3 / 2 :=
by sorry

-- Part II: Prove the angle θ between a and b is 45 degrees
theorem angle_between_a_and_b : θ = real.pi / 4 :=
by sorry

end magnitude_of_a_angle_between_a_and_b_l417_417925


namespace angle_ADY_equals_angle_ADZ_l417_417434

-- Define variables and conditions
variables (A B C K Y Z : Type) [acute_triangle A B C]
variable (D : Line) (AD_altitude : altitude D A B C)
hypothesis (K_on_AD : K ∈ AD_altitude)
variable (BK : Line) (BK_intersect_AC_at_Y : BK ∩ LineAC = Y)
variable (CK : Line) (CK_intersect_AB_at_Z : CK ∩ LineAB = Z)

-- Proof goal
theorem angle_ADY_equals_angle_ADZ :
  ∀ (A B C K Y Z : Type) 
    [acute_triangle A B C] (D : Line) (AD_altitude : altitude D A B C)
    (K_on_AD : K ∈ AD_altitude)
    (BK : Line) (BK_intersect_AC_at_Y : BK ∩ LineAC = Y)
    (CK : Line) (CK_intersect_AB_at_Z : CK ∩ LineAB = Z),
  ∠ADY = ∠ADZ :=
sorry

end angle_ADY_equals_angle_ADZ_l417_417434


namespace distance_to_origin_is_sqrt_2_l417_417239

def complex_number : ℂ := (2 * complex.I) / (1 - complex.I)

theorem distance_to_origin_is_sqrt_2 : complex.abs complex_number = Real.sqrt 2 := by
  sorry

end distance_to_origin_is_sqrt_2_l417_417239


namespace stratified_sampling_grade_10_l417_417717

theorem stratified_sampling_grade_10 (x y z x_s : ℕ)
  (h1 : x = 2 * z)
  (h2 : y = 2 * z)
  (h3 : 45 * x / (x + y + z) = x_s) :
  x_s = 18 := by
  sorry

end stratified_sampling_grade_10_l417_417717


namespace second_number_l417_417490

theorem second_number (x : ℕ) (h1 : ∃ k : ℕ, 1428 = 129 * k + 9)
  (h2 : ∃ m : ℕ, x = 129 * m + 13) (h_gcd : ∀ (d : ℕ), d ∣ (1428 - 9 : ℕ) ∧ d ∣ (x - 13 : ℕ) → d ≤ 129) :
  x = 1561 :=
by
  sorry

end second_number_l417_417490


namespace function_range_l417_417543

variable {a c : ℝ}
def f (x : ℝ) : ℝ := a * x^2 - c

theorem function_range 
  (h1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
  (h2 : -1 ≤ f 2 ∧ f 2 ≤ 5)
  (ha_pos : 0 < a) : 
  -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end function_range_l417_417543


namespace correct_area_of_triangle_l417_417889

def Point := (ℝ × ℝ)

def triangle_area (A B C : Point) : ℝ :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  let (x3, y3) := C in
  0.5 * (abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)))

theorem correct_area_of_triangle :
  let A : Point := (4, -7)
  let B : Point := (-1, 3)
  let C : Point := (2, -8)
  triangle_area A B C = 12.5 :=
by
  sorry

end correct_area_of_triangle_l417_417889


namespace company_employees_l417_417693

noncomputable def num_employees
  (N_S : ℕ)
  (uses_vk : ℕ)
  (uses_both : ℕ)
  (uses_only_odno : ℕ)
  (prop_odno : ℚ)
  (no_social : ℕ)
  (total_prop_odno : ℚ) :
  Prop :=
  (no_social = 40) ∧
  (uses_vk = (3 / 4) * N_S) ∧
  (uses_both = (13 / 20) * N_S) ∧
  (uses_only_odno = N_S - uses_vk + uses_both) ∧
  (0.9 * N_S = (5 / 6) * prop_odno) ∧
  (N_S = total_prop_odno - no_social) ∧
  (prop_odno = total_prop_odno)

theorem company_employees :
  ∃ (N : ℕ), ∀ (N_S : ℕ) (uses_vk : ℕ) (uses_both : ℕ) (uses_only_odno : ℕ) (prop_odno : ℚ) (no_social : ℕ) (total_prop_odno : ℚ),
  (num_employees N_S uses_vk uses_both uses_only_odno prop_odno no_social prop_odno) → N = 540 :=
begin
  sorry
end

end company_employees_l417_417693


namespace a8_value_l417_417162

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else sequence (n - 1) + 2 ^ (n - 1)

theorem a8_value : sequence 8 = 255 := by
  sorry

end a8_value_l417_417162


namespace net_price_change_l417_417974

theorem net_price_change (P : ℝ) : 
  let decreased_price := P * 0.85
  let increased_price := decreased_price * 1.10
  increased_price - P = -0.065 * P := by
  let decreased_price := P * 0.85
  let increased_price := decreased_price * 1.10
  have h1 : decreased_price = P * 0.85 := rfl
  have h2 : increased_price = decreased_price * 1.10 := rfl
  have h3 : increased_price = P * 0.935 := by rw [h1, h2]
  have h4 : increased_price - P = 0.935 * P - P := by rw [h3]
  have h5 : 0.935 * P - P = -0.065 * P := by ring
  rw [h4, h5]
  done

end net_price_change_l417_417974


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417007

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417007


namespace intersection_A_B_l417_417950

open Set

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 1}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := 
by {
  sorry
}

end intersection_A_B_l417_417950


namespace diana_additional_game_time_l417_417869

theorem diana_additional_game_time :
  ∀ (reading_hours : ℕ) (minutes_per_hour : ℕ) (raise_percent : ℕ),
    reading_hours = 12 → minutes_per_hour = 30 → raise_percent = 20 →
    (reading_hours * (minutes_per_hour * raise_percent / 100)) = 72 :=
by
  intros reading_hours minutes_per_hour raise_percent h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end diana_additional_game_time_l417_417869


namespace alice_wins_game_l417_417439

theorem alice_wins_game (m : ℕ) (h : m > 5) : 
  ∃ (strategy : ℕ → ℝ), (∀ n, 
  -- conditions for the strategy
  (
    (∀ i j, i ≠ j → abs (strategy i - strategy j) ≥ 1.5) ∧ 
    (∀ k, strategy k ∈ set.Icc 0 m)
  ) →
  -- strategy results in Alice winning
  (n % 2 = 0 → player (strategy n) = Alice) ∧ 
  (n % 2 = 1 → player (strategy n) = Bob)) :=
sorry

end alice_wins_game_l417_417439


namespace vector_parallel_solution_l417_417557

theorem vector_parallel_solution (x : ℝ) 
    (a : ℝ × ℝ := (2, 1))
    (b : ℝ × ℝ := (x, 2))
    (h : (a.1 + 3 * b.1, a.2 + 3 * b.2) ∥ (a.1 - b.1, a.2 - b.2)) :
  x = 4 :=
by
  sorry

end vector_parallel_solution_l417_417557


namespace hyperbola_focus_distance_l417_417193

theorem hyperbola_focus_distance:
  let b := sqrt 5
  let hyperbola_asymptote (x y : ℝ) := sqrt 5 * x - 2 * y = 0
  let parabola_focus : ℝ × ℝ := (3, 0)
  let hyperbola_focus : ℝ × ℝ := parabola_focus
  (dist_from_point_to_line hyperbola_focus hyperbola_asymptote) = sqrt 5 :=
by
  sorry

end hyperbola_focus_distance_l417_417193


namespace min_separable_covering_l417_417526

open Set Finset

noncomputable theory

def separable_covering (n : ℕ) (F : Finset (Finset (Fin n))) : Prop :=
  (∀ x y : Fin n, x ≠ y → ∃ A ∈ F, (x ∈ A ∧ y ∉ A) ∨ (x ∉ A ∧ y ∈ A)) ∧
  (∀ x : Fin n, ∃ A ∈ F, x ∈ A)

theorem min_separable_covering (n : ℕ) (h : 2 ≤ n) :
  ∃ t : ℕ, (∃ F : Finset (Finset (Fin n)), separable_covering n F ∧ F.card = t) ∧ t = 1 + ⌊Real.log2 (n + 1)⌋ :=
sorry

end min_separable_covering_l417_417526


namespace max_distinct_reduced_quadratic_equations_l417_417778

theorem max_distinct_reduced_quadratic_equations 
  (quadratic_eq : ℕ → ℝ → ℝ → Prop) 
  (quadratic : ∀ n a b, quadratic_eq n a b → (a ≠ b))
  (common_root : ∀ n1 n2 m1 m2 a b, quadratic_eq n1 a b → quadratic_eq n2 a b → b = a)
  (no_five_common_root : ∀ a b, (∀ n, n < 5 → quadratic_eq n a b) → false) :
  ∃ N, ∀ M, M > N → ¬ (∀ n m a b, n < M → quadratic_eq n a b) :=
by sorry

end max_distinct_reduced_quadratic_equations_l417_417778


namespace average_difference_l417_417689

theorem average_difference (F1 L1 F2 L2 : ℤ) (H1 : F1 = 200) (H2 : L1 = 400) (H3 : F2 = 100) (H4 : L2 = 200) :
  (F1 + L1) / 2 - (F2 + L2) / 2 = 150 := 
by 
  sorry

end average_difference_l417_417689


namespace negative_expressions_l417_417212

theorem negative_expressions (x : ℝ) (hx : x < 0) :
  (x / (3 * |x|) < 0) ∧ (-x^3 < 0) ∧ (-x⁻² < 0) := by
  sorry

end negative_expressions_l417_417212


namespace expected_pieces_ice_cream_l417_417897

open ProbabilityTheory

noncomputable def iceCreamCutting (diameter : Real) (cube : Set (ℝ × ℝ × ℝ))
  (planes : Set (ℝ × ℝ × ℝ)) : Real := sorry

theorem expected_pieces_ice_cream :
  let diameter := 2
  let cube := { p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ 0 ≤ p.3 ∧ p.3 ≤ 1 }
  ∃ planes : Set (ℝ × ℝ × ℝ),
    integrate (λ x, iceCreamCutting diameter cube planes) (uniformSpace { p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ 0 ≤ p.3 ∧ p.3 ≤ 1 }) = 7 + (13 * Real.pi) / 3 := 
sorry

end expected_pieces_ice_cream_l417_417897


namespace bus_speed_excluding_stoppages_l417_417130

theorem bus_speed_excluding_stoppages (v : ℝ) (effective_speed : ℝ) (stoppage_minutes_per_hour : ℝ) :
  effective_speed = 43 → stoppage_minutes_per_hour = 8.4 → v = 50 :=
by
  intro h_effective_speed h_stoppage_minutes
  have h_stoppage_hours : stoppage_minutes_per_hour / 60 = 0.14 := by sorry
  have h_moving_time : (1 - stoppage_minutes_per_hour / 60) = 0.86 := by sorry
  have h_ratio : effective_speed / v = (1 - stoppage_minutes_per_hour / 60) := by sorry
  have h_solution : v = 43 / (0.86) := by sorry
  exact h_solution

end bus_speed_excluding_stoppages_l417_417130


namespace carpet_needed_in_sq_yards_l417_417088

theorem carpet_needed_in_sq_yards :
  let length := 15
  let width := 10
  let area_sq_feet := length * width
  let conversion_factor := 9
  let area_sq_yards := area_sq_feet / conversion_factor
  area_sq_yards = 16.67 := by
  sorry

end carpet_needed_in_sq_yards_l417_417088


namespace breadth_of_rectangular_plot_l417_417797

theorem breadth_of_rectangular_plot :
  ∃ b : ℝ, (∃ l : ℝ, l = 3 * b ∧ l * b = 867) ∧ b = 17 :=
by
  sorry

end breadth_of_rectangular_plot_l417_417797


namespace problem_statement_l417_417377

def contrapositive {P Q : Prop} (h : P → Q) : ¬Q → ¬P :=
by sorry

def sufficient_but_not_necessary (P Q : Prop) : (P → Q) ∧ ¬(Q → P) :=
by sorry

def proposition_C (p q : Prop) : ¬(p ∧ q) → (¬p ∨ ¬q) :=
by sorry

def negate_exists (P : ℝ → Prop) : (∃ x : ℝ, P x) → ¬(∀ x : ℝ, ¬P x) :=
by sorry

theorem problem_statement : 
¬ (∀ (P Q : Prop), ¬(P ∧ Q) → (¬P ∨ ¬Q)) :=
by sorry

end problem_statement_l417_417377


namespace male_adults_count_l417_417437

-- Define the main problem context
noncomputable def family_reunion := 
  ∀ (male_adults : ℕ), 
  let female_adults := 50 + male_adults in 
  let total_adults := male_adults + female_adults in 
  let children := 2 * total_adults in 
  total_adults + children = 750 → male_adults = 100

theorem male_adults_count : family_reunion :=
by {
  sorry
}

end male_adults_count_l417_417437


namespace find_first_offset_l417_417888

theorem find_first_offset {area diagonal offset₁ offset₂ : ℝ}
  (h_area : area = 150)
  (h_diagonal : diagonal = 20)
  (h_offset₂ : offset₂ = 6) :
  2 * area = diagonal * (offset₁ + offset₂) → offset₁ = 9 := by
  sorry

end find_first_offset_l417_417888


namespace shortest_player_height_l417_417725

theorem shortest_player_height :
  ∀ (tallest_height difference : ℝ), 
    tallest_height = 77.75 ∧ difference = 9.5 → 
    tallest_height - difference = 68.25 :=
by
  intros tallest_height difference h
  cases h
  sorry

end shortest_player_height_l417_417725


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417040

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417040


namespace find_m_if_power_function_l417_417315

theorem find_m_if_power_function : 
  ∀ (m : ℝ), (∃ f : ℝ → ℝ, f = λ x, (m^2 - m - 1) * x^m ∧ (∃ α : ℝ, ∀ x : ℝ, f x = x^α)) → m^2 - m - 1 = 1 :=
by
  intro m
  intro h
  -- The proof will follow here.
  sorry

end find_m_if_power_function_l417_417315


namespace all_parallelepipeds_are_cubes_l417_417406

-- Defining the variables and conditions
variables (n : ℕ) (s : ℝ)
variables (a b c : Fin n → ℝ)

-- The main theorem
theorem all_parallelepipeds_are_cubes
  (h : (π * √3 * s^3) / 2 = ∑ i in Finset.univ, (π * (a i ^ 2 + b i ^ 2 + c i ^ 2) ^ (3 / 2)) / 6) :
  ∀ i, a i = b i ∧ b i = c i :=
sorry

end all_parallelepipeds_are_cubes_l417_417406


namespace odot_subtraction_l417_417470

-- Define the new operation
def odot (a b : ℚ) : ℚ := (a^3) / (b^2)

-- State the theorem
theorem odot_subtraction :
  ((odot (odot 2 4) 6) - (odot 2 (odot 4 6)) = -81 / 32) :=
by
  sorry

end odot_subtraction_l417_417470


namespace pure_imaginary_factors_real_solutions_l417_417887

theorem pure_imaginary_factors_real_solutions :
  {x : ℝ | let z := (complex.mk x 1) in
              let z1 := (complex.mk (x + 1) 1) in
              let z2 := (complex.mk (x + 2) 1) in
              (z * z1 * z2).re = 0} = {-3, -1, 1} :=
by
  sorry

end pure_imaginary_factors_real_solutions_l417_417887


namespace required_connections_l417_417750

theorem required_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 := by
  sorry

end required_connections_l417_417750


namespace maria_profit_disks_l417_417281

theorem maria_profit_disks (cost_price_per_5 : ℝ) (sell_price_per_4 : ℝ) (desired_profit : ℝ) : 
  (cost_price_per_5 = 6) → (sell_price_per_4 = 8) → (desired_profit = 120) →
  (150 : ℝ) = desired_profit / ((sell_price_per_4 / 4) - (cost_price_per_5 / 5)) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end maria_profit_disks_l417_417281


namespace probability_at_least_one_2_on_8_sided_dice_l417_417346

theorem probability_at_least_one_2_on_8_sided_dice :
  (∃ (d1 d2 : Fin 8), d1 = 1 ∨ d2 = 1) → (15 / 64) = (15 / 64) := by
  intro h
  sorry

end probability_at_least_one_2_on_8_sided_dice_l417_417346


namespace num_triples_solutions_l417_417143

def sign (a : ℝ) : ℝ :=
  if a > 0 then 1 else if a < 0 then -1 else 0

theorem num_triples_solutions : ∃ n : ℕ, n = 3 ∧
  ∃ x y z : ℝ, x = 2023 - 2024 * sign (y + z) ∧
              y = 2023 - 2024 * sign (x + z) ∧
              z = 2023 - 2024 * sign (x + y) := sorry

end num_triples_solutions_l417_417143


namespace round_5738291_4982_to_nearest_integer_l417_417666

theorem round_5738291_4982_to_nearest_integer :
  Real.round 5738291.4982 = 5738291 :=
by
  sorry

end round_5738291_4982_to_nearest_integer_l417_417666


namespace find_angle_BXY_l417_417241

-- Defining the setup of the problem
variables {A B C D E F G H X Y : Type}
variables (angle_AXE angle_CYX : ℝ)
variable (parallel_AB_CD : Prop)
variable (lines_intersect : Prop)

-- Conditions encoded
def problem_conditions : Prop :=
  parallel_AB_CD ∧ lines_intersect ∧ angle_AXE = 4 * angle_CYX - 120

-- Correct answer encoded
def problem_solution : Prop :=
  angle_AXE = 40

-- The theorem to prove
theorem find_angle_BXY (h : problem_conditions) : problem_solution :=
by sorry

end find_angle_BXY_l417_417241


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417017

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417017


namespace lower_bound_of_range_of_expression_l417_417501

theorem lower_bound_of_range_of_expression :
  ∃ L, (∀ n : ℤ, L < 4*n + 7 → 4*n + 7 < 100) ∧
  (∃! n_min n_max : ℤ, 4*n_min + 7 = L ∧ 4*n_max + 7 = 99 ∧ (n_max - n_min + 1 = 25)) :=
sorry

end lower_bound_of_range_of_expression_l417_417501


namespace sum_of_roots_eq_neg_two_l417_417201

noncomputable def polynomial_roots_sum (a b : ℝ) : ℝ :=
  let p1 := polynomial.X^2 + polynomial.C a * polynomial.X + polynomial.C b
  let p2 := polynomial.X^2 + polynomial.C b * polynomial.X + polynomial.C a
  (p1 * p2).roots.sum

theorem sum_of_roots_eq_neg_two (a b : ℝ) :
  (a^2 - 4*b > 0) → 
  (b^2 - 4*a > 0) → 
  (∃ x y z, (polynomial_roots_sum a b) = x + y + z ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) → 
  polynomial_roots_sum a b = -2 := 
sorry

end sum_of_roots_eq_neg_two_l417_417201


namespace distinct_sequences_count_l417_417561
-- Broad import to bring in the entirety of the necessary library

-- Statement of the proof problem
theorem distinct_sequences_count : 
  let letters := ['T','R','I','A','N','G','L','E']
  ∃ (seq : List Char), 
  seq.length = 5 ∧
  seq.head = 'T' ∧ 
  seq.last ≠ 'E' ∧ 
  (seq.get? 1 = some 'G' ∨ seq.get? 2 = some 'G') ∧ 
  (seq.erase 'T').erase '?' ≠ [] ∧  -- ensure letters in the resulting list never run out
  seq.erase 'G' ≠ [] → -- ensure G is in the valid positions
  seq.erase 'E' ≠ [] → -- ensure E is not in the last position
  seq.erase '? < 8! - 7! - (3 * 5!) = 96 -- counting distinct valid sequences
sorry

end distinct_sequences_count_l417_417561


namespace log_one_half_eight_l417_417879

noncomputable def log_base_half (x : ℝ) : ℝ := log x / log (1/2)

theorem log_one_half_eight : log_base_half 8 = -3 := by
  unfold log_base_half
  sorry

end log_one_half_eight_l417_417879


namespace jake_planting_charge_l417_417613

def mowing_time : ℕ := 1
def mowing_pay : ℕ := 15
def planting_time : ℕ := 2
def desired_rate : ℕ := 20

theorem jake_planting_charge : mowing_time = 1 ∧ mowing_pay = 15 ∧ planting_time = 2 ∧ desired_rate = 20 → 
  let total_planting_charge := desired_rate * planting_time + (desired_rate - mowing_pay)
  total_planting_charge = 45 :=
by
  intros
  simp
  sorry

end jake_planting_charge_l417_417613


namespace valid_pairs_l417_417482

open Nat

theorem valid_pairs (m n : ℕ) (h1 : m ∣ 2^(euler_totient n) + 1) (h2 : n ∣ 2^(euler_totient m) + 1) :
  (m, n) = (1, 1) ∨ (m, n) = (1, 3) ∨ (m, n) = (3, 1) :=
by
  -- the proof can be provided here
  sorry

end valid_pairs_l417_417482


namespace sum_of_a_and_b_l417_417646

noncomputable def a : ℕ :=
sorry

noncomputable def b : ℕ :=
sorry

theorem sum_of_a_and_b :
  (100 ≤ a ∧ a ≤ 999) ∧ (1000 ≤ b ∧ b ≤ 9999) ∧ (10000 * a + b = 7 * a * b) ->
  a + b = 1458 :=
by
  sorry

end sum_of_a_and_b_l417_417646


namespace white_longer_than_blue_l417_417132

noncomputable def whiteLineInches : ℝ := 7.666666666666667
noncomputable def blueLineInches : ℝ := 3.3333333333333335
noncomputable def inchToCm : ℝ := 2.54
noncomputable def cmToMm : ℝ := 10

theorem white_longer_than_blue :
  let whiteLineCm := whiteLineInches * inchToCm
  let blueLineCm := blueLineInches * inchToCm
  let differenceCm := whiteLineCm - blueLineCm
  let differenceMm := differenceCm * cmToMm
  differenceMm = 110.05555555555553 := by
  sorry

end white_longer_than_blue_l417_417132


namespace sum_f_eq_zero_l417_417625

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 2 * x - x^2 else
    if x < 0 then -f (-x) else -f (x - 2)

theorem sum_f_eq_zero : 
  (∑ i in Finset.range 2016, f i) = 0 :=
by
  sorry

end sum_f_eq_zero_l417_417625


namespace smallest_t_for_full_ellipse_l417_417703

theorem smallest_t_for_full_ellipse : 
  ∃ t : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t) → r θ = 2 * Math.cos θ) ∧ 
           (∀ θ : ℝ, (r θ = (2 * Math.cos θ)) ↔ 
           (r θ = (2 * Math.cos (θ + π)))) :=
begin
  use π,
  -- Proof goes here
  sorry
end

end smallest_t_for_full_ellipse_l417_417703


namespace probability_at_least_one_two_l417_417356

theorem probability_at_least_one_two (dice_fair : ∀ i, 1 ≤ i ∧ i ≤ 8) (dice_count : 2):
  ∃ probability, probability = 15 / 64 := 
by
  sorry

end probability_at_least_one_two_l417_417356


namespace coefficient_x_in_binomial_expansion_l417_417590

theorem coefficient_x_in_binomial_expansion : 
  let T (k : ℕ) := (Nat.choose 3 k) * 2^k * x^(3-2*k)
  in (x + 2/x)^3 = ∑ k in finset.range(4), T k →
  coefficient 1 (∑ k in finset.range(4), T k) = 6 :=
begin
  sorry
end

end coefficient_x_in_binomial_expansion_l417_417590


namespace volume_inequality_holds_l417_417378

def volume (x : ℕ) : ℤ :=
  (x^2 - 16) * (x^3 + 25)

theorem volume_inequality_holds :
  ∃ (n : ℕ), n = 1 ∧ ∃ x : ℕ, volume x < 1000 ∧ (x - 4) > 0 :=
by
  sorry

end volume_inequality_holds_l417_417378


namespace find_original_number_l417_417369

-- Let x be the original number
def original_number (x : ℤ) : Prop := (x + 10) % 26 = 0

theorem find_original_number : ∃ x : ℤ, original_number x ∧ x = 16 :=
by {
  use 16,
  unfold original_number,
  norm_num,
  exact eq.refl 0
}

end find_original_number_l417_417369


namespace valid_license_plates_count_l417_417433

def validLicensePlates : Nat :=
  26 * 26 * 26 * 10 * 9 * 8

theorem valid_license_plates_count :
  validLicensePlates = 15818400 :=
by
  sorry

end valid_license_plates_count_l417_417433


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417008

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417008


namespace line_equation_final_equation_l417_417516

theorem line_equation (k : ℝ) : 
  (∀ x y, y = k * (x - 1) + 1 ↔ 
  ∀ x y, y = k * ((x + 2) - 1) + 1 - 1) → 
  k = 1 / 2 :=
by
  sorry

theorem final_equation : 
  ∃ k : ℝ, k = 1 / 2 ∧ (∀ x y, y = k * (x - 1) + 1) → 
  ∀ x y, x - 2 * y + 1 = 0 :=
by
  sorry

end line_equation_final_equation_l417_417516


namespace probability_more_twos_than_fives_correct_l417_417966

noncomputable def probability_more_twos_than_fives : ℚ :=
  -- Assumptions:
  -- - Five fair six-sided dice
  -- - Each die roll is independent and uniformly distributed over {1, 2, 3, 4, 5, 6}
  let total_outcomes := 6^5 in  -- Total number of outcomes when 5 dice are rolled
  let equal_twos_and_fives := 1024 + 1280 + 120 in  -- Outcomes with equal number of 2's and 5's calculated in solution
  let probability_of_equal_twos_and_fives := equal_twos_and_fives / total_outcomes in
  1 / 2 * (1 - probability_of_equal_twos_and_fives)

theorem probability_more_twos_than_fives_correct : 
  probability_more_twos_than_fives = 2676 / 7776 :=
by
  sorry

end probability_more_twos_than_fives_correct_l417_417966


namespace fire_does_not_spread_with_seven_squares_fire_spreads_with_eight_squares_l417_417669

open Classical

noncomputable theory

def Chessboard := Fin (8 * 8) → Prop

def neighbors (f : Chessboard) (n : Fin 8 × Fin 8) : Finset (Fin 8 × Fin 8) :=
  Finset.filter (λ p, |p.1 - n.1| + |p.2 - n.2| = 1) (Finset.univ : Finset (Fin 8 × Fin 8))

def on_fire (f : Chessboard) (n : Fin 8 × Fin 8) : Prop :=
  neighbors f n |>.to_list.filter (λ p, f p).length ≥ 2

def fire_spreads_everywhere (f : Chessboard) : Prop :=
  ∀ (n : Fin 8 × Fin 8), f n = true

def seven_initial_squares_on_fire (f : Chessboard) : Prop :=
  finset.filter (λ n, f n) finset.univ.to_list.length = 7

def eight_initial_squares_on_fire (f : Chessboard) : Prop :=
  finset.filter (λ n, f n) finset.univ.to_list.length = 8

theorem fire_does_not_spread_with_seven_squares (f : Chessboard) :
  seven_initial_squares_on_fire f → ¬ fire_spreads_everywhere f := by sorry

theorem fire_spreads_with_eight_squares (f : Chessboard) :
  eight_initial_squares_on_fire f → fire_spreads_everywhere f := by sorry

end fire_does_not_spread_with_seven_squares_fire_spreads_with_eight_squares_l417_417669


namespace sum_of_distances_to_faces_eq_height_l417_417155

noncomputable def tetrahedron_distance_sum (a : ℝ) : ℝ := (sqrt 6 / 3) * a

theorem sum_of_distances_to_faces_eq_height (a : ℝ) (P : EuclideanGeometry.Point ℝ 3) (T : EuclideanGeometry.RegularTetrahedron ℝ) :
  EuclideanGeometry.PointInsideTetrahedron P T →
  let m := EuclideanGeometry.DistanceFromPointToFace P T.face1
    n := EuclideanGeometry.DistanceFromPointToFace P T.face2
    p := EuclideanGeometry.DistanceFromPointToFace P T.face3
    q := EuclideanGeometry.DistanceFromPointToFace P T.face4 in
  m + n + p + q = tetrahedron_distance_sum a := by
  sorry

end sum_of_distances_to_faces_eq_height_l417_417155


namespace probability_at_least_one_two_l417_417350

def num_dice := 2
def sides_dice := 8
def total_outcomes := sides_dice ^ num_dice
def num_non_favorable_outcomes := (sides_dice - 1) ^ num_dice
def num_favorable_outcomes := total_outcomes - num_non_favorable_outcomes
def probability_favorable_outcomes := (15 : ℚ) / (64 : ℚ)

theorem probability_at_least_one_two :
  probability_favorable_outcomes = 15 / 64 :=
sorry

end probability_at_least_one_two_l417_417350


namespace no_solution_system_of_equations_l417_417124

theorem no_solution_system_of_equations :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 6) ∧ (4 * x - 6 * y = 8) ∧ (5 * x - 5 * y = 15) :=
by {
  sorry
}

end no_solution_system_of_equations_l417_417124


namespace proof_subset_l417_417573

def set_A := {x : ℝ | x ≥ 0}

theorem proof_subset (B : Set ℝ) (h : set_A ∪ B = B) : set_A ⊆ B := 
by
  sorry

end proof_subset_l417_417573


namespace ticket_price_increase_l417_417835

noncomputable def y (x : ℕ) : ℝ :=
  if x ≤ 100 then
    30 * x - 50 * Real.sqrt x - 500
  else
    30 * x - 50 * Real.sqrt x - 700

theorem ticket_price_increase (m : ℝ) : 
  m * 20 - 50 * Real.sqrt 20 - 500 ≥ 0 → m ≥ 37 := sorry

end ticket_price_increase_l417_417835


namespace total_new_emails_l417_417252

-- Define the conditions
def receive_new_emails (n : ℕ) : ℕ := 20 * n

def final_batch_emails : ℕ := 10

-- Define the theorem to be proved
theorem total_new_emails (n : ℕ) : receive_new_emails(n) + final_batch_emails = 20 * n + 10 :=
by
  sorry

end total_new_emails_l417_417252


namespace tan_half_sum_angles_l417_417629

theorem tan_half_sum_angles (p q : ℝ) 
(h1 : cos p + cos q = 1/3)
(h2 : sin p + sin q = 5/13) :
tan ((p + q) / 2) = 15 / 13 := 
sorry

end tan_half_sum_angles_l417_417629


namespace switch_connections_l417_417744

theorem switch_connections (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 :=
by
  rw [h1, h2]
  norm_num
  sorry

end switch_connections_l417_417744


namespace unique_prime_satisfying_condition_l417_417134

theorem unique_prime_satisfying_condition :
  ∃! p : ℕ, Prime p ∧ (∀ q : ℕ, Prime q ∧ q < p → ∀ k r : ℕ, p = k * q + r ∧ 0 ≤ r ∧ r < q → ∀ a : ℕ, a > 1 → ¬ a^2 ∣ r) ∧ p = 13 :=
sorry

end unique_prime_satisfying_condition_l417_417134


namespace find_alpha_l417_417677

variable (α β k : ℝ)
variable (h1 : α = k * β)
variable (h2 : α = 8)
variable (h3 : β = 2)

theorem find_alpha (β : ℝ) (hα_inverse : α = k * β) (h_initial1 : α = 8) (h_initial2 : β = 2) :
  ∃ k, α = 8 ∧ β = 2 → (α = 20 ∧ β = 5) :=
sorry

end find_alpha_l417_417677


namespace sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l417_417486

theorem sum_of_two_terms_is_term_iff_a_is_multiple_of_d
    (a d : ℤ) 
    (n k : ℕ) 
    (h : ∀ (p : ℕ), a + d * n + (a + d * k) = a + d * p)
    : ∃ m : ℤ, a = d * m :=
sorry

end sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l417_417486


namespace number_of_spiders_l417_417250

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 32) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 4 := by
  sorry

end number_of_spiders_l417_417250


namespace caroline_wins_with_21_l417_417581

-- Conditions provided in the problem:
def anthony_hand : List ℕ := [6, 3, 10] -- Adjusted the 12 to 10 as mentioned in the solution
def leo_score : ℕ := 28
def winning_score : ℕ := 21

-- Question translated to a Lean statement:
theorem caroline_wins_with_21 (c_score : ℕ) 
  (a_total : ∑ i in anthony_hand, i = 19) -- Anthony's corrected total
  (l_score : leo_score = 28)
  (w_score : ∀ hand, hand = winning_score → hand <= 21)
  (w_cond : ∑ i in anthony_hand, i < winning_score) 
  : c_score = winning_score :=
sorry

end caroline_wins_with_21_l417_417581


namespace grid1_cannot_become_zero_grid_grid2_cannot_become_zero_grid_grid3_can_become_zero_grid_l417_417799

-- Define the initial grids for each case
def grid1 := [
  [0, 1, 1, 0], 
  [1, 1, 0, 1], 
  [0, 0, 1, 1], 
  [0, 0, 1, 1]
]

def grid2 := [
  [0, 1, 0, 0], 
  [1, 1, 0, 1], 
  [0, 0, 0, 1], 
  [1, 0, 1, 1]
]

def grid3 := [
  [0, 0, 0, 0], 
  [1, 1, 0, 0], 
  [0, 1, 0, 1], 
  [1, 0, 0, 1]
]

-- Define the theorem statements
theorem grid1_cannot_become_zero_grid : ¬ (∃ steps, perform_steps grid1 steps = zero_grid) :=
sorry

theorem grid2_cannot_become_zero_grid : ¬ (∃ steps, perform_steps grid2 steps = zero_grid) :=
sorry

theorem grid3_can_become_zero_grid : ∃ steps, perform_steps grid3 steps = zero_grid :=
sorry

end grid1_cannot_become_zero_grid_grid2_cannot_become_zero_grid_grid3_can_become_zero_grid_l417_417799


namespace walnut_tree_total_count_l417_417729

theorem walnut_tree_total_count (current_trees new_trees : ℕ) (h_current : current_trees = 22) (h_new : new_trees = 33) : 
  current_trees + new_trees = 55 :=
by 
  rw [h_current, h_new]
  exact rfl

end walnut_tree_total_count_l417_417729


namespace christmas_not_one_seventh_l417_417836

theorem christmas_not_one_seventh
    (is_common_year : ∀ y : ℕ, ¬ (y % 4 = 0 → (y % 100 ≠ 0) ∨ (y % 400 ≠ 0)))
    (is_leap_year : ∀ y : ℕ, (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0))
    (days_in_leap_year : ∀ y : ℕ, (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0) → 366)
    (days_in_common_year : ∀ y : ℕ, ¬((y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)) → 365)
    (total_days_in_400_years : 400 * 365 + (400 / 4 - 400 / 100 + 400 / 400) = 146100 ∨ 365 * 303 + 366 * 97 = 146097 → 146097 = 0 [MOD 7])
    (N : ℕ) :
    (N / 400) ≠ (1 / 7) :=
sorry

end christmas_not_one_seventh_l417_417836


namespace equation_of_ellipse_C_max_area_triangle_AOB_l417_417187

open Real

section Example

variables {a b: ℝ} (k m: ℝ)
def ellipse_C : Prop := (a > b) ∧ (a^2 = 4) ∧ (a^2 - b^2 = 1)
def ellipse_Omega_major_axis : Prop := (2 * a = 4)
def ellipse_Omega_minor_axis : Prop := (2 * b = 2)

theorem equation_of_ellipse_C : ellipse_C a b → ∀ {x y: ℝ}, 
  (x^2 / 4 + y^2 / 3 = 1) :=
by
  intros hC x y
  cases hC with h1 h2,
  exact sorry

theorem max_area_triangle_AOB 
  (C_eq : ∀ {x y: ℝ}, (x^2 / 4 + y^2 / 3 = 1))
  (l_eq : ∀ {x y: ℝ}, y = k * x + m)
  (condition: ∀ {x1 y1 x2 y2: ℝ}, 
    y1 = k * x1 + m ∧ y2 = k * x2 + m ∧ 
    2*k = (y1 / (x1 + 1) + y2 / (x2 + 1)) ∧ 
    x1 + x2 + 2 = 0) :
  ∃ max_area, max_area = sqrt 3 :=
by
  have h1 := C_eq,
  have h2 := l_eq,
  exact sorry

end Example

end equation_of_ellipse_C_max_area_triangle_AOB_l417_417187


namespace tax_rate_computation_l417_417430

-- Define the inputs
def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 134.4

-- Define the derived taxable amount
def taxable_amount : ℝ := total_value - non_taxable_amount

-- Define the expected tax rate
def expected_tax_rate : ℝ := 0.12

-- State the theorem
theorem tax_rate_computation : 
  (tax_paid / taxable_amount * 100) = expected_tax_rate * 100 := 
by
  sorry

end tax_rate_computation_l417_417430


namespace ordered_pair_is_l417_417992

open Real

noncomputable def vector_problem : Prop :=
  let OA : ℝ := 2
  let OB : ℝ := 3
  let OC : ℝ := 2 * sqrt 5
  let tan_AOC : ℝ := 2
  let angle_BOC_deg : ℝ := 60
  let cos_AOC := 1 / sqrt 5
  let sin_AOC := 2 / sqrt 5
  let cos_BOC := 1 / 2
  let sin_BOC := sqrt 3 / 2
  let p := (2 * sqrt 5 - 3) / 3
  let q := (2 - 2 * sqrt 5) / 5
  sqrt((p * OA + q * OB)^2 + (p * sin_AOC * OC + q * sin_BOC * OB)^2) = OC

theorem ordered_pair_is : vector_problem :=
  by
  -- Proof steps would go here
  sorry

end ordered_pair_is_l417_417992


namespace head_start_ratio_l417_417793

variable (Va Vb L H : ℕ)

-- Conditions
def speed_relation : Prop := Va = (4 * Vb) / 3

-- The head start fraction that makes A and B finish the race at the same time given the speed relation
theorem head_start_ratio (Va Vb L H : ℕ)
  (h1 : speed_relation Va Vb)
  (h2 : L > 0) : (H = L / 4) :=
sorry

end head_start_ratio_l417_417793


namespace number_of_special_permutations_l417_417617

theorem number_of_special_permutations : 
  let a := [a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11, a_12],
      perm_a := list.perm a [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      desc_a1_a2_a3_a4 := a_1 > a_2 ∧ a_2 > a_3 ∧ a_3 > a_4,
      asc_a4_rest := a_4 < a_5 ∧ a_5 < a_6 ∧ a_6 < a_7 ∧ a_7 < a_8 ∧ a_8 < a_9 ∧ a_9 < a_10 ∧ a_10 < a_11 ∧ a_11 < a_12
  in
  perm_a ∧ desc_a1_a2_a3_a4 ∧ asc_a4_rest → (number_of_special_permutations = 165) :=
by 
  sorry

end number_of_special_permutations_l417_417617


namespace network_connections_l417_417736

theorem network_connections (n : ℕ) (k : ℕ) :
  n = 30 → k = 4 → (n * k) / 2 = 60 := by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end network_connections_l417_417736


namespace triangle_right_l417_417580

theorem triangle_right (A B C : ℝ) (h : (sin A + sin B) * (sin A - sin B) = (sin C)^2) :
  A + B + C = π ∧ (A = π / 2 ∨ B = π / 2 ∨ C = π / 2) :=
sorry

end triangle_right_l417_417580


namespace badge_counts_l417_417761

variable (x : ℝ) -- Number of badges Tolya initially had
variable (y : ℝ) -- Number of badges Vasya initially had 

-- Conditions
axiom initial_more_badges : y = x + 5
axiom exchange_badges_vasya : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y
axiom exchange_badges_tolya : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y

-- Proof goal
theorem badge_counts (x : ℝ) (y : ℝ) 
  (h1 : y = x + 5) 
  (h2 : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y) 
  (h3 : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y) : 
  x = 45 ∧ y = 50 :=
by
  sorry

end badge_counts_l417_417761


namespace percentage_red_toys_l417_417226

-- Step d: Lean 4 statement
theorem percentage_red_toys (T R: ℕ) (h1: T / 2 = 100)
                            (h2: R = 0.10 * T + 60)
                            (h3: 0.10 * T ∈ ℕ) :
  (R / T) * 100 = 40 :=
by
  sorry

end percentage_red_toys_l417_417226


namespace eval_f_l417_417538

-- Definitions based on conditions in a)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def f (x : ℝ) : ℝ := if x > 0 then x^2 - (1 / x) else x^2 + (1 / x)

-- Lean statement
theorem eval_f'_at_neg_1 :
  is_even_function f →
  f (1 : ℝ) = 1 - (1 / 1) →
  f (-1) = 1 + (1 / 1) →
  (∃ x, f x = 1) →
  deriv f (-1) = -3 :=
by
  intros
  sorry

end eval_f_l417_417538


namespace total_chocolate_pieces_l417_417402

def total_chocolates (boxes : ℕ) (per_box : ℕ) : ℕ :=
  boxes * per_box

theorem total_chocolate_pieces :
  total_chocolates 6 500 = 3000 :=
by
  sorry

end total_chocolate_pieces_l417_417402


namespace max_sum_of_digits_l417_417278

theorem max_sum_of_digits (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C) (h5 : B ≠ D) 
  (h6 : C ≠ D) (h7: A ∈ {0,1,2,3,4,5,6,7,8,9}) (h8: B ∈ {0,1,2,3,4,5,6,7,8,9}) 
  (h9: C ∈ {0,1,2,3,4,5,6,7,8,9}) (h10: D ∈ {0,1,2,3,4,5,6,7,8,9}) (h11 : (C + D) % 2 = 0) 
  (h12 : (A + B) % (C + D) = 0) :
  A + B ≤ 16 := 
begin
  sorry
end

end max_sum_of_digits_l417_417278


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417004

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417004


namespace problem_conditions_l417_417082

variables {R : Type*} [LinearOrderedField R]

-- Define the conditions as assumptions
variable (f : R → R)
variable (a x : R)

-- The three goals to prove
theorem problem_conditions (h1 : ∀ x y : R, f (x + y) = f x + f y) (h2 : ∀ x : R, x < 0 → f x > 0) (ha : a > 0) :
  (f 0 = 0) ∧ 
  (∀ x : R, f (-x) = -f x) ∧ 
  ((0 < a ∧ a < Real.sqrt 2 → ∃ x : R, x ∈ Set.Ioo a (2 / a)) ∧
   (a = Real.sqrt 2 → Set.Ioo a (2 / a) = ∅) ∧
   (a > Real.sqrt 2 → ∃ x : R, x ∈ Set.Ioo (2 / a) a)) :=
begin
  sorry  -- The proof
end

end problem_conditions_l417_417082


namespace MariaDriveDistanceAfterSecondStop_l417_417474

theorem MariaDriveDistanceAfterSecondStop :
  ∀ (totalDistance : ℕ) (fraction1 : ℚ) (fraction2 : ℚ),
  totalDistance = 560 →
  fraction1 = (1/2 : ℚ) →
  fraction2 = (1/4 : ℚ) →
  let firstStop := totalDistance * fraction1,
      remainingDistanceAfterFirstStop := totalDistance - firstStop,
      secondStop := remainingDistanceAfterFirstStop * fraction2,
      remainingDistanceAfterSecondStop := remainingDistanceAfterFirstStop - secondStop
  in remainingDistanceAfterSecondStop = 210 :=
by
  -- Proof goes here
  sorry

end MariaDriveDistanceAfterSecondStop_l417_417474


namespace height_comparison_l417_417755

variable (r1 h1 r2 h2 : ℝ)

-- Define the conditions
def condition_1 : Prop := π * r1^2 * h1 = π * r2^2 * h2
def condition_2 : Prop := r2 = 1.2 * r1

-- Prove the relationship between h1 and h2
theorem height_comparison
  (h_cond1 : condition_1 r1 h1 r2 h2)
  (h_cond2 : condition_2 r1 r2) :
  h1 = 1.44 * h2 :=
by
  sorry

end height_comparison_l417_417755


namespace quadrilateral_area_l417_417057

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 40) (hh1 : h1 = 11) (hh2 : h2 = 9) :
  (d * (h1 + h2)) / 2 = 400 :=
by
  rw [hd, hh1, hh2]
  norm_num
  sorry

end quadrilateral_area_l417_417057


namespace not_continuous_at_origin_l417_417996

def f (x y : ℝ) : ℝ :=
if x^2 + y^4 ≠ 0 then - (x * y^2) / (y^2 + y^4) else 0

theorem not_continuous_at_origin :
  ¬ContinuousAt (f) (0, 0) :=
sorry

end not_continuous_at_origin_l417_417996


namespace network_connections_l417_417739

theorem network_connections (n : ℕ) (k : ℕ) :
  n = 30 → k = 4 → (n * k) / 2 = 60 := by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end network_connections_l417_417739


namespace unique_function_satisfying_conditions_l417_417135

theorem unique_function_satisfying_conditions :
  ∀ f : ℚ → ℚ, (f 1 = 2) → (∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) → (∀ x : ℚ, f x = x + 1) :=
by
  intro f h1 hCond
  sorry

end unique_function_satisfying_conditions_l417_417135


namespace total_job_applications_l417_417114

theorem total_job_applications (apps_in_state : ℕ) (apps_other_states : ℕ) 
  (h1 : apps_in_state = 200)
  (h2 : apps_other_states = 2 * apps_in_state) :
  apps_in_state + apps_other_states = 600 :=
by
  sorry

end total_job_applications_l417_417114


namespace avg_height_and_variance_correct_l417_417225

noncomputable def avg_height_and_variance
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_avg_height : ℕ)
  (boys_variance : ℕ)
  (girls_avg_height : ℕ)
  (girls_variance : ℕ) : (ℕ × ℕ) := 
  let total_students := 300
  let boys := 180
  let girls := 120
  let boys_avg_height := 170
  let boys_variance := 14
  let girls_avg_height := 160
  let girls_variance := 24
  let avg_height := (boys * boys_avg_height + girls * girls_avg_height) / total_students 
  let variance := (boys * (boys_variance + (boys_avg_height - avg_height) ^ 2) 
                    + girls * (girls_variance + (girls_avg_height - avg_height) ^ 2)) / total_students
  (avg_height, variance)

theorem avg_height_and_variance_correct:
   avg_height_and_variance 300 180 120 170 14 160 24 = (166, 42) := 
  by {
    sorry
  }

end avg_height_and_variance_correct_l417_417225


namespace classroom_desks_l417_417990

theorem classroom_desks (N y : ℕ) (h : 16 * y = 21 * N)
  (hN_le: N <= 30 * 16 / 21) (hMultiple: 3 * N % 4 = 0)
  (hy_le: y ≤ 30)
  : y = 21 := by
  sorry

end classroom_desks_l417_417990


namespace bead_selection_non_zero_power_of_two_l417_417728

open Finset

def bead : Type := Fin 20
def color : Type := Fin 10
def box : Type := Fin 10

variable (f : bead → box)
variable (g : bead → color)

theorem bead_selection_non_zero_power_of_two :
  (∀ S : Finset box, S.card = 10 → ∃ T : Finset bead, T.card = 10 ∧ ∀ (b ∈ T), f b ∈ S ∧ ∀ c : color, (∃ b ∈ T, g b = c)) →
  ∃ n : ℕ, (∃ m : ℕ, m > 0 ∧ n = 2 ^ m) :=
by sorry

end bead_selection_non_zero_power_of_two_l417_417728


namespace final_mixture_ratio_l417_417277

variable (x y : ℝ)

-- Define the conditions as given in the problem
def mp := 5 * x
def wp := 4 * x
def mq := 2 * y
def wq := 7 * y

-- Define the final mixture constraints
def P_milk := (mp + mq) / (mp + wp + mq + wq)
def P_water := (wp + wq) / (mp + wp + mq + wq)

-- Statement to prove
theorem final_mixture_ratio (h : P_milk = P_water) : x = 5 * y := by
  sorry

end final_mixture_ratio_l417_417277


namespace min_ab_min_a_plus_b_l417_417905

theorem min_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : ab >= 8 :=
sorry

theorem min_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : a + b >= 3 + 2 * Real.sqrt 2 :=
sorry

end min_ab_min_a_plus_b_l417_417905


namespace setup_nim_value_zero_l417_417462

-- Define the game and its conditions

-- Function to compute nim-value for a given number of bricks
def nim_value (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => 1
  | 6 => 2
  | 7 => 3
  | _ => 0  -- not relevant for this problem 

-- The main statement we need to prove
theorem setup_nim_value_zero : nim_value 7 ⊕ nim_value 2 ⊕ nim_value 1 = 0 := by
  -- Proof would go here
  sorry

end setup_nim_value_zero_l417_417462


namespace half_of_4_pow_2022_is_2_pow_4043_l417_417366

theorem half_of_4_pow_2022_is_2_pow_4043 :
  (4 ^ 2022) / 2 = 2 ^ 4043 :=
by sorry

end half_of_4_pow_2022_is_2_pow_4043_l417_417366


namespace vector_at_t_neg3_l417_417518

theorem vector_at_t_neg3 :
  let a := (2, 3)
  let b := (12, -37)
  let d := ((b.1 - a.1) / 5, (b.2 - a.2) / 5)
  let line_param (t : ℝ) := (a.1 + t * d.1, a.2 + t * d.2)
  line_param (-3) = (-4, 27) := by
  -- Proof goes here
  sorry

end vector_at_t_neg3_l417_417518


namespace Nero_speed_is_8_l417_417608

-- Defining the conditions
def Jerome_time := 6 -- in hours
def Nero_time := 3 -- in hours
def Jerome_speed := 4 -- in miles per hour

-- Calculation step
def Distance := Jerome_speed * Jerome_time

-- The theorem we need to prove (Nero's speed)
theorem Nero_speed_is_8 :
  (Distance / Nero_time) = 8 := by
  sorry

end Nero_speed_is_8_l417_417608


namespace age_of_25th_student_l417_417305

theorem age_of_25th_student (avg_age_25 : ℕ → ℕ → ℚ)
  (h_total_avg : avg_age_25 25 16 = 16)
  (h_7_avg : avg_age_25 7 15 = 15)
  (h_12_avg : avg_age_25 12 16 = 16)
  (h_5_avg : avg_age_25 5 18 = 18) : 
  ∃ age_25 : ℕ, age_25 = 13 :=
by {
  -- Defining total ages based on given conditions
  have total_age_7 := 7 * 15,
  have total_age_12 := 12 * 16,
  have total_age_5 := 5 * 18,
  
  -- Sum of ages of first 24 students
  have total_age_24 : ℕ := total_age_7 + total_age_12 + total_age_5,
  
  -- Overall total age of 25 students
  have total_age_25 : ℕ := 25 * 16,
  
  -- Age of the 25th student
  have age_25 := total_age_25 - total_age_24,
  
  -- Assert that age_25 equals 13
  use age_25,
  
  exact rfl,
}

end age_of_25th_student_l417_417305


namespace coeff_x3_of_p_cube_l417_417569

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^4 - 4*x^2 + 3*x + 1

-- Statement to prove
theorem coeff_x3_of_p_cube (x : ℝ) : polynomial.coeff ((polynomial.C (p x))^3) 3 = 27 :=
sorry

end coeff_x3_of_p_cube_l417_417569


namespace detergent_for_9_pounds_l417_417287

-- Define the given condition.
def detergent_per_pound : ℕ := 2

-- Define the total weight of clothes
def weight_of_clothes : ℕ := 9

-- Define the result of the detergent used.
def detergent_used (d : ℕ) (w : ℕ) : ℕ := d * w

-- Prove that the detergent used to wash 9 pounds of clothes is 18 ounces
theorem detergent_for_9_pounds :
  detergent_used detergent_per_pound weight_of_clothes = 18 := 
sorry

end detergent_for_9_pounds_l417_417287


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417036

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417036


namespace find_f_a_l417_417152

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 0 then 4 * Real.log (2 : ℝ) (-x) / (Real.log 2)
else abs (x^2 + a * x)

theorem find_f_a (a : ℝ) (h : a ≠ 0) (h_f : f a (f a (-Real.sqrt 2)) = 4) : f a a = 8 :=
by
  have ha : a = -4 := sorry,
  rw ha,
  exact sorry

end find_f_a_l417_417152


namespace class_average_is_86_l417_417813

variable {BoyPercentage GirlPercentage BoyScore GirlScore ClassAverage : ℝ}

axiom h1 : BoyPercentage = 0.4
axiom h2 : GirlPercentage = 1 - BoyPercentage
axiom h3 : BoyScore = 80
axiom h4 : GirlScore = 90
axiom h5 : ClassAverage = BoyPercentage * BoyScore + GirlPercentage * GirlScore

theorem class_average_is_86 :
  ClassAverage = 86 :=
by
  have hGirlPercentage := calc
    GirlPercentage = 1 - BoyPercentage : by rw [h2]
  show ClassAverage = 86 from sorry

end class_average_is_86_l417_417813


namespace limit_of_a_n_plus_S_n_l417_417596

noncomputable theory

-- Define the sequence a_n and the sum S_n.
def a₁ : ℕ → ℝ := λ n, if n = 1 then 1 else 0  -- Initial value for the sequence
def S : ℕ → ℝ
| 0     := 0
| (n+1) := S n + if n = 0 then 1 else (a₁ (n+1))

-- Define a recursive relationship for the sequence.
def a : ℕ → ℝ
| 0     := 1
| (n+1) := 3 * S n - 4  -- Recursive relationship

-- The theorem statement to prove.
theorem limit_of_a_n_plus_S_n : (∀ n, a₁ n = 1 ∨ a₁ n = 0) ∧
  (∀ n ≥ 1, S (n-1) - 8, a₁ n, 8 * S n - 4 are_in_arithmetic_progression) → 
  (∃ l, tendsto (λ n, a₁ n + S n) at_top (𝓝 l) ∧ l = 4 / 3) :=
by sorry

end limit_of_a_n_plus_S_n_l417_417596


namespace distinct_numbers_constructible_l417_417064

theorem distinct_numbers_constructible {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  ∃ (S : set ℕ), S.card = (k - 1) * (n - k + 1) ∧
  (∀ a ∈ S, ∃ f : k → ℕ, (length f = n ∧ eval_expr f = a)) ∧
  (∀ a ∈ S, ¬ ∃ f : k → ℕ, length f = n - 1 ∧ eval_expr f = a) :=
sorry

definition eval_expr (f : ℕ → ℕ) : ℕ :=
sorry

end distinct_numbers_constructible_l417_417064


namespace time_n_kilometer_l417_417078

-- Define the conditions in Lean
noncomputable theory

def speed (n : ℕ) (k : ℝ) : ℝ := k / (n - 1)^2

axiom speed_at_third : speed 3 (4/3) = 1 / 3

theorem time_n_kilometer (n : ℕ) (h : n ≥ 3) : 
  ∃ t : ℝ, t = 3 * (n - 1)^2 / 4 :=
sorry

end time_n_kilometer_l417_417078


namespace badge_counts_l417_417762

variable (x : ℝ) -- Number of badges Tolya initially had
variable (y : ℝ) -- Number of badges Vasya initially had 

-- Conditions
axiom initial_more_badges : y = x + 5
axiom exchange_badges_vasya : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y
axiom exchange_badges_tolya : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y

-- Proof goal
theorem badge_counts (x : ℝ) (y : ℝ) 
  (h1 : y = x + 5) 
  (h2 : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y) 
  (h3 : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y) : 
  x = 45 ∧ y = 50 :=
by
  sorry

end badge_counts_l417_417762


namespace average_targets_hit_average_targets_hit_ge_half_l417_417658

theorem average_targets_hit (n : ℕ) : 
  let p := 1 - (1 - (1 : ℚ) / n)^n in 
  n * p = n * (1 - (1 - (1 : ℚ) / n)^n) :=
sorry

theorem average_targets_hit_ge_half (n : ℕ) (hn : n > 0) : 
  let E := n * (1 - (1 - (1 : ℚ) / n)^n) in 
  E >= n / 2 :=
sorry

end average_targets_hit_average_targets_hit_ge_half_l417_417658


namespace pentagon_circle_area_ratio_l417_417403

theorem pentagon_circle_area_ratio
    (s : ℝ)
    (O : Type)
    (r : ℝ)
    (tan_54 : ℝ)
    (h₁ : ∀ P Q : O, O ∈ circle → tangent P ∧ tangent Q)
    (h₂ : center_of_circle O ∧ center_of_pentagon A B)
    (h₃ : equidistant O B C ∧ equidistant O C D)
    (h₄ : tan_54 = Real.tan (54 * (Real.pi / 180))) :
    (π * (s / (2 * tan_54))^2) / s^2 = π / (4 * tan_54^2) :=
by
    sorry

end pentagon_circle_area_ratio_l417_417403


namespace class_gpa_l417_417702

theorem class_gpa (n : ℕ) (hn : n > 0) (gpa1 : ℝ := 30) (gpa2 : ℝ := 33) : 
    (gpa1 * (n:ℝ) + gpa2 * (2 * n : ℝ)) / (3 * n : ℝ) = 32 :=
by
  sorry

end class_gpa_l417_417702


namespace total_amount_paid_correct_l417_417958

-- Definitions of quantities and rates
def quantity_grapes := 3
def rate_grapes := 70
def quantity_mangoes := 9
def rate_mangoes := 55

-- Total amount calculation
def total_amount_paid := quantity_grapes * rate_grapes + quantity_mangoes * rate_mangoes

-- Theorem to prove total amount paid is 705
theorem total_amount_paid_correct : total_amount_paid = 705 :=
by
  sorry

end total_amount_paid_correct_l417_417958


namespace network_connections_l417_417737

theorem network_connections (n : ℕ) (k : ℕ) :
  n = 30 → k = 4 → (n * k) / 2 = 60 := by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end network_connections_l417_417737


namespace part1_part2_l417_417942

def f (x : ℝ) := |x - 2| - |x - 5|

theorem part1 : ∀ x : ℝ, -3 ≤ f x ∧ f x ≤ 3 := by
  intro x
  -- Proof is omitted
  sorry

theorem part2 : {x : ℝ | f x ≥ x^2 - 8*x + 14} = {x : ℝ | 3 ≤ x ∧ x ≤ 4 + Real.sqrt 5} := by
  -- Proof is omitted
  sorry

end part1_part2_l417_417942


namespace lattice_polygon_enclosure_l417_417594

-- Define what it means to be a lattice point
def is_lattice_point (P : ℤ × ℤ) : Prop :=
  true

-- Define a lattice polygon as a set of lattice points
def is_lattice_polygon (Γ : set (ℤ × ℤ)) : Prop :=
  ∀ P ∈ Γ, is_lattice_point P

-- Define what it means to be convex
def is_convex (Γ : set (ℤ × ℤ)) : Prop :=
  ∀ (P Q R : ℤ × ℤ), P ∈ Γ → Q ∈ Γ → R ∈ Γ →
  (∃ a b c : ℝ, a + b + c = 1 ∧ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧
                 a • (P.1, P.2) + b • (Q.1, Q.2) + c • (R.1, R.2) ∈ Γ)

-- We state the actual theorem we want to prove
theorem lattice_polygon_enclosure (Γ : set (ℤ × ℤ)) (h₁ : is_lattice_polygon Γ) (h₂ : is_convex Γ) :
  ∃ Γ' : set (ℤ × ℤ), is_lattice_polygon Γ' ∧ is_convex Γ' ∧
  (∀ P ∈ Γ, P ∈ Γ') ∧ (∃ T ∈ Γ', T ∉ Γ ∧ ∀ V ∈ Γ', V ∉ Γ → V = T) :=
by sorry

end lattice_polygon_enclosure_l417_417594


namespace traveler_payment_strategy_l417_417432

-- Definition of the chain and the problem conditions
def chain : Type := list ℕ

-- Given Conditions
def initial_chain : chain := [1, 2, 3, 4, 5, 6, 7]
def cuts_allowed : ℕ := 1
def days : ℕ := 7

-- Question restated as a Theorem: The traveler can pay the hotel owner for 7 days with only one cut
theorem traveler_payment_strategy (c : chain) (cuts : ℕ) (d : ℕ) :
  c = initial_chain ∧ cuts = cuts_allowed ∧ d = days →
  ∃ (strategy : list chain), (∀ n, 1 ≤ n ∧ n ≤ 7 → strategy = some_strategy) ∧ (cuts = 1) :=
  sorry

end traveler_payment_strategy_l417_417432


namespace badge_counts_l417_417763

variable (x : ℝ) -- Number of badges Tolya initially had
variable (y : ℝ) -- Number of badges Vasya initially had 

-- Conditions
axiom initial_more_badges : y = x + 5
axiom exchange_badges_vasya : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y
axiom exchange_badges_tolya : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y

-- Proof goal
theorem badge_counts (x : ℝ) (y : ℝ) 
  (h1 : y = x + 5) 
  (h2 : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y) 
  (h3 : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y) : 
  x = 45 ∧ y = 50 :=
by
  sorry

end badge_counts_l417_417763


namespace determine_constants_l417_417471

theorem determine_constants (P Q R : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-2 * x^2 + 5 * x - 7) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) ↔
    (P = 7 ∧ Q = -9 ∧ R = 5) :=
by
  sorry

end determine_constants_l417_417471


namespace max_balls_in_cube_l417_417370

theorem max_balls_in_cube 
  (radius : ℝ) (side_length : ℝ) 
  (ball_volume : ℝ := (4 / 3) * Real.pi * (radius^3)) 
  (cube_volume : ℝ := side_length^3) 
  (max_balls : ℝ := cube_volume / ball_volume) :
  radius = 3 ∧ side_length = 8 → Int.floor max_balls = 4 := 
by
  intro h
  rw [h.left, h.right]
  -- further proof would use numerical evaluation
  sorry

end max_balls_in_cube_l417_417370


namespace time_to_pick_sugar_snap_peas_l417_417851

theorem time_to_pick_sugar_snap_peas (pea_count1 pea_count2 : ℕ) (time1 : ℕ) :
  (pea_count1 = 56 ∧ time1 = 7 ∧ pea_count2 = 72) →
  let rate := pea_count1 / time1 in
  (pea_count2 / rate = 9) :=
by
  intros h
  let ⟨h1, h2, h3⟩ := h
  sorry

end time_to_pick_sugar_snap_peas_l417_417851


namespace find_YW_l417_417599

/-- Given a triangle XYZ with XZ = YZ = 10, and a point W on XY such that XW = 5 and CW = 6, 
    find that YW is approximately 6.32. -/
theorem find_YW (X Y Z W : Type) (d_XZ d_YZ d_XW d_CW : ℝ) 
    (h_XZ: d_XZ = 10) (h_YZ : d_YZ = 10) (h_XW : d_XW = 5) (h_CW : d_CW = 6) :
    ∃ (d_YW : ℝ), d_YW ≈ 6.32 := 
sorry

end find_YW_l417_417599


namespace badge_exchange_proof_l417_417770

-- Definitions based on the conditions
def initial_badges_Tolya : ℝ := 45
def initial_badges_Vasya : ℝ := 50

def tollya_exchange_badges (badges_Tolya : ℝ) : ℝ := 0.2 * badges_Tolya
def tollya_receive_badges (badges_Vasya : ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_exchange_badges (badges_Vasya: ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_receive_badges (badges_Tolya: ℝ) : ℝ := 0.2 * badges_Tolya

-- Vasya ended up with one badge less than Tolya after the exchange
theorem badge_exchange_proof 
  (tolya_initial badges_Tolya_initial : ℝ)
  (badges_Vasya_initial: ℝ)
  (tollya_initial_has_24: tollya_receive_badges badges_Vasya_initial)
  (vasya_initial_has_20: vasya_receive_badges badges_Tolya_initial):
  (tollya_initial = initial_badges_Tolya) ∧ (vasya_initial = initial_badges_Vasya) :=
sorry

end badge_exchange_proof_l417_417770


namespace range_of_a_l417_417183

open Real

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_a (f_mono : ∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2)
  (h_ex : ∃ x : ℝ, f (| x + 1 |) ≤ f (log 2 a - | x + 2 |)) :
  a ≥ 2 :=
by
  sorry

end range_of_a_l417_417183


namespace maximize_profit_l417_417405

def cups_sold (p : ℝ) : ℝ :=
  150 - 4 * p

def revenue (p : ℝ) : ℝ :=
  p * cups_sold p

def cost : ℝ :=
  200

def profit (p : ℝ) : ℝ :=
  revenue p - cost

theorem maximize_profit (p : ℝ) (h : p ≤ 30) : p = 19 → profit p = 1206.25 :=
by
  sorry

end maximize_profit_l417_417405


namespace numerical_form_correct_l417_417787

theorem numerical_form_correct : 
  "one billion two hundred two million five" = 1,202,000,005 :=
by sorry

end numerical_form_correct_l417_417787


namespace smallest_n_1000_l417_417678

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def count_factors_of_five (n : ℕ) : ℕ :=
n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 + n / 15625 + n / 78125 + n / 390625 + n / 1953125 + n / 9765625

theorem smallest_n_1000 (a b c d : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (hpos_d : 0 < d) (hsum : a + b + c + d = 4014) :
  let n := count_factors_of_five a + count_factors_of_five b + count_factors_of_five c + count_factors_of_five d in
  n = 1000 :=
sorry

end smallest_n_1000_l417_417678


namespace tetrahedron_volume_distance_relation_l417_417422

theorem tetrahedron_volume_distance_relation
  (V : ℝ) (S1 S2 S3 S4 : ℝ) (H1 H2 H3 H4 : ℝ)
  (k : ℝ)
  (hS : (S1 / 1) = k) (hS2 : (S2 / 2) = k) (hS3 : (S3 / 3) = k) (hS4 : (S4 / 4) = k) :
  H1 + 2 * H2 + 3 * H3 + 4 * H4 = 3 * V / k :=
sorry

end tetrahedron_volume_distance_relation_l417_417422


namespace probability_even_sum_or_product_div_by_5_l417_417579

theorem probability_even_sum_or_product_div_by_5 :
  let S := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
      pairs := {p : ℕ × ℕ // p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 ≠ p.2}
      even_sum := λ x : ℕ × ℕ, (x.1 + x.2) % 2 = 0
      div_by_5 := λ x : ℕ × ℕ, (x.1 * x.2) % 5 = 0
      favorable_pairs := {p : ℕ × ℕ // (even_sum p ∨ div_by_5 p)}
      total_pairs := pairs.to_finset.card
      favorable_count := favorable_pairs.to_finset.card
  in favorable_count / total_pairs = 16 / 45 :=
 by sorry

end probability_even_sum_or_product_div_by_5_l417_417579


namespace sum_of_squares_of_rates_is_correct_l417_417475

noncomputable def biking_jogging_swimming_rates_sum_of_squares : ℕ :=
  let b := 15
  let j := 8
  let s := 5
  b^2 + j^2 + s^2

theorem sum_of_squares_of_rates_is_correct :
  (∃ b j s : ℕ, 2 * b + 3 * j + 4 * s = 74 ∧ 2 * j + 3 * s + 4 * b = 91 ∧ b^2 + j^2 + s^2 = biking_jogging_swimming_rates_sum_of_squares) :=
begin
  use 15,
  use 8,
  use 5,
  split,
  { norm_num },
  split,
  { norm_num },
  norm_num,
end

#eval biking_jogging_swimming_rates_sum_of_squares

example : biking_jogging_swimming_rates_sum_of_squares = 314 := by norm_num

end sum_of_squares_of_rates_is_correct_l417_417475


namespace even_func_sol_set_l417_417624

theorem even_func_sol_set (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) (h_f2 : f 2 = 0) :
  { x : ℝ | f (x + 1) > 0 } = set.Ioo (-1) 1 :=
by
  sorry

end even_func_sol_set_l417_417624


namespace quotient_base4_division_l417_417131

def base4_to_decimal (n : ℕ) (digs : List ℕ) : ℕ :=
  digs.enum_from 0 |>.sum (λ ⟨i, d⟩ => d * n ^ i)

def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n == 0 then [0] else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n == 0 then acc else loop (n / 4) ((n % 4) :: acc)
    loop n []

theorem quotient_base4_division :
  let n1 := base4_to_decimal 4 [3, 1, 0, 2]  -- 2013_4
  let n2 := base4_to_decimal 4 [3, 1]      -- 13_4
  let q := n1 / n2 in
  (decimal_to_base4 q = [1, 3]) :=  -- 13_4
by {
  sorry
}

end quotient_base4_division_l417_417131


namespace complex_calculation_l417_417537

variable (a b c : ℂ)

noncomputable def condition1 : Prop := a^2 + a * b + b^2 = 1 + complex.i
noncomputable def condition2 : Prop := b^2 + b * c + c^2 = -2
noncomputable def condition3 : Prop := c^2 + c * a + a^2 = 1

theorem complex_calculation (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) :
  (a * b + b * c + c * a) ^ 2 = (-11 - 4 * complex.i) / 3 :=
sorry

end complex_calculation_l417_417537


namespace hyperbola_eccentricity_l417_417156

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (P Q F₁ F₂ : ℝ × ℝ)
  (H1 : ∃ x y : ℝ, P = (x, y) ∧ Q = (-x, -y) ∧ (x / a)^2 - (y / b)^2 = 1)
  (H2 : F₁ = (-(c 0), 0) ∧ F₂ = (c 0, 0) ∧ c 0 = sqrt (a^2 + b^2))
  (H3 : dist P F₂ = dist Q F₂)
  (H4 : dist P F₁ = 2 * dist Q F₁)
  (H5 : dist (0, 0) P = dist (0, 0) Q)
  : eccentricity = sqrt 5 :=
sorry

end hyperbola_eccentricity_l417_417156


namespace find_focal_length_of_hyperbola_l417_417182

def focal_length_of_hyperbola (m : ℝ) (h_m_gt_zero : m > 0) (h_asymptote : sqrt 3 * x + m * y = 0) : ℝ :=
4

theorem find_focal_length_of_hyperbola (m : ℝ) (h_m_gt_zero : m > 0) (h_asymptote : sqrt 3 * x + m * y = 0) :
  focal_length_of_hyperbola m h_m_gt_zero h_asymptote = 4 :=
sorry

end find_focal_length_of_hyperbola_l417_417182


namespace algebraic_expression_value_l417_417578

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x + 7 = 6) : 4 * x^2 + 8 * x - 5 = -9 :=
by
  sorry

end algebraic_expression_value_l417_417578


namespace number_of_white_circles_l417_417092

-- Define the repeating pattern and the length of the sequence
def pattern := [true, true, true, false, false]  -- true represents a white circle, false represents a black circle
def sequence_length := 102

-- Prove that the number of white circles in the first 102 circles is 62
theorem number_of_white_circles : list.count true (list.take sequence_length (list.repeat pattern (sequence_length / pattern.length))) + list.count true (list.take (sequence_length % pattern.length) pattern) = 62 := 
by
sorTry

end number_of_white_circles_l417_417092


namespace cos_theta_value_l417_417903

theorem cos_theta_value (θ : ℝ) 
  (sin_cos_sum : sin θ + cos θ = 7 / 5) 
  (tan_gt_one : tan θ > 1) : 
  cos θ = 3 / 5 := 
by 
  sorry -- skipping the proof

end cos_theta_value_l417_417903


namespace right_triangle_median_l417_417160

noncomputable def median_to_hypotenuse_length (a b : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (a^2 + b^2)
  hypotenuse / 2

theorem right_triangle_median
  (a b : ℝ) (h_a : a = 3) (h_b : b = 4) :
  median_to_hypotenuse_length a b = 2.5 :=
by
  sorry

end right_triangle_median_l417_417160


namespace last_remaining_marble_is_black_l417_417774

theorem last_remaining_marble_is_black 
  (n_black : ℕ) (n_white : ℕ) (initial_black : n_black = 2013) (initial_white : n_white = 2014)
  (add_white : ℕ → ℕ)
  (draw_two_marbles : (ℕ × ℕ) → (ℕ × ℕ))
  (remaining_one_marble : ℕ = 1) :
  ∃ final_color : string, final_color = "black" :=
by 
  sorry

end last_remaining_marble_is_black_l417_417774


namespace average_marks_l417_417685

-- Definitions
def Tatuya_score (Ivanna_score : ℕ) : ℕ := 2 * Ivanna_score
def Ivanna_score (Dorothy_score : ℕ) : ℕ := (3 * Dorothy_score) / 5
def Dorothy_score : ℕ := 90

-- Theorem statement
theorem average_marks :
  let Dorothy_score := Dorothy_score in
  let Ivanna_score := Ivanna_score Dorothy_score in
  let Tatuya_score := Tatuya_score Ivanna_score in
  (Dorothy_score + Ivanna_score + Tatuya_score) / 3 = 84 :=
by 
  -- Proof goes here
  sorry

end average_marks_l417_417685


namespace positive_solution_unique_m_l417_417972

theorem positive_solution_unique_m (m : ℝ) : ¬ (4 < m ∧ m < 2) :=
by
  sorry

end positive_solution_unique_m_l417_417972


namespace probability_even_product_l417_417529

noncomputable def A : Set ℕ := {1, 2, 3, 4}
noncomputable def B : Set ℕ := {5, 6, 7, 8}

def count_even_products (a : Set ℕ) (b : Set ℕ) : ℕ :=
  (a.toFinset.product b.toFinset).count (λ p, (p.1 * p.2) % 2 = 0)

theorem probability_even_product : (count_even_products A B : ℚ) / (A.toFinset.card * B.toFinset.card) = 3 / 4 := 
sorry

end probability_even_product_l417_417529


namespace forest_length_l417_417411

noncomputable def length_of_forest (L : ℝ) : Prop :=
  let area := L * 6
  let total_trees := 600 * area
  let trees_per_logger_per_month := 6 * 30
  let total_trees_cut_down := 8 * trees_per_logger_per_month * 10
  total_trees = total_trees_cut_down

theorem forest_length : length_of_forest 40 :=
by
  let L : ℝ := 40
  let area := L * 6
  let total_trees := 600 * area
  let trees_per_logger_per_month := 6 * 30
  let total_trees_cut_down := 8 * trees_per_logger_per_month * 10
  show 600 * L * 6 = total_trees_cut_down
  sorry

end forest_length_l417_417411


namespace find_A_minus_B_l417_417262

theorem find_A_minus_B :
  let A := (4 - abs (-12))
  let B := (-6 + 5)
  A - B = -7 :=
by
  let A := 4 - abs (-12)
  let B := -6 + 5
  have hA : A = -8 := by
    calc 
      A = 4 - abs (-12) : rfl
      ... = 4 - 12 : by simp
      ... = -8 : by linarith
  have hB : B = -1 := by
    calc 
      B = -6 + 5 : rfl
      ... = -1 : by linarith
  calc 
    A - B = -8 - (-1) : by rw [hA, hB]
    ... = -8 + 1 : by linarith
    ... = -7 : by linarith

end find_A_minus_B_l417_417262


namespace lottery_blanks_l417_417584

theorem lottery_blanks (P B : ℕ) (h₁ : P = 10) (h₂ : (P : ℝ) / (P + B) = 0.2857142857142857) : B = 25 := 
by
  sorry

end lottery_blanks_l417_417584


namespace slope_CD_is_one_l417_417987

theorem slope_CD_is_one
  (k : ℝ) (x1 x2 : ℝ) 
  (h_k_pos : 0 < k) 
  (h_x1_pos : 0 < x1) 
  (h_x2_pos : 0 < x2)
  (h_A : k * x1 = real.exp (x1 - 1)) 
  (h_B : k * x2 = real.exp (x2 - 1)) 
  :
  (real.log x2 - real.log x1) / (x2 - x1) = 1 := 
sorry

end slope_CD_is_one_l417_417987


namespace yarn_total_length_l417_417318

/-- The green yarn is 156 cm long, the red yarn is 8 cm more than three times the green yarn,
    prove that the total length of the two pieces of yarn is 632 cm. --/
theorem yarn_total_length : 
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  green_yarn + red_yarn = 632 :=
by
  let green_yarn := 156
  let red_yarn := 3 * green_yarn + 8
  sorry

end yarn_total_length_l417_417318


namespace finding_b_l417_417140

def equation_expr_correct (a b c : ℕ) (N : ℕ) : Prop :=
  a > 1 ∧ b > 1 ∧ c > 1 ∧ N ≠ 1 →
  (N^(1/a) * N^(1/(a * b)) * N^(2/(a * b * c)) = N^(17/24)) →

theorem finding_b (a b c N : ℕ) (h : equation_expr_correct a b c N) : b = 4 :=
  sorry

end finding_b_l417_417140


namespace line_eq_l417_417313

theorem line_eq (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_eq : 1 / a + 9 / b = 1) (h_min_interp : a + b = 16) : 
  ∃ l : ℝ × ℝ → ℝ, ∀ x y : ℝ, l (x, y) = 3 * x + y - 12 :=
by
  sorry

end line_eq_l417_417313


namespace tub_drain_time_l417_417381

theorem tub_drain_time (t : ℝ) (p q : ℝ) (h1 : t = 4) (h2 : p = 5 / 7) (h3 : q = 2 / 7) :
  q * t / p = 1.6 := by
  sorry

end tub_drain_time_l417_417381


namespace factor_theorem_example_l417_417883

theorem factor_theorem_example (t : ℚ) : (4 * t^3 + 6 * t^2 + 11 * t - 6 = 0) ↔ (t = 1/2) :=
by sorry

end factor_theorem_example_l417_417883


namespace net_effect_on_sale_l417_417059

theorem net_effect_on_sale (P Q : ℝ) :
  let new_price := 0.65 * P
  let new_quantity := 1.8 * Q
  let original_revenue := P * Q
  let new_revenue := new_price * new_quantity
  new_revenue - original_revenue = 0.17 * original_revenue :=
by
  sorry

end net_effect_on_sale_l417_417059


namespace proof_problem_l417_417855

noncomputable def a : ℝ := 3.54
noncomputable def b : ℝ := 1.32
noncomputable def result : ℝ := (a - b) * 2

theorem proof_problem : result = 4.44 := by
  sorry

end proof_problem_l417_417855


namespace minimal_egg_moves_l417_417206

-- Define the conditions as a Lean structure or predicate
def symmetric_moves (initial_eggs : ℕ) (rows : ℕ) (columns : ℕ) : Prop :=
  initial_eggs = 15 ∧ rows = 3 ∧ columns = 5

-- Define the statement of minimal extra moves needed
theorem minimal_egg_moves : symmetric_moves 15 3 5 → ∃ moves : ℕ, moves = 11 :=
by
  intro h
  use 11
  sorry

end minimal_egg_moves_l417_417206


namespace walnut_tree_total_count_l417_417730

theorem walnut_tree_total_count (current_trees new_trees : ℕ) (h_current : current_trees = 22) (h_new : new_trees = 33) : 
  current_trees + new_trees = 55 :=
by 
  rw [h_current, h_new]
  exact rfl

end walnut_tree_total_count_l417_417730


namespace chewbacca_pack_size_l417_417118

/-- Given Chewbacca has 20 pieces of cherry gum and 30 pieces of grape gum,
if losing one pack of cherry gum keeps the ratio of cherry to grape gum the same
as when finding 5 packs of grape gum, determine the number of pieces x in each 
complete pack of gum. We show that x = 14. -/
theorem chewbacca_pack_size :
  ∃ (x : ℕ), (20 - x) * (30 + 5 * x) = 20 * 30 ∧ ∀ (y : ℕ), (20 - y) * (30 + 5 * y) = 600 → y = 14 :=
by
  sorry

end chewbacca_pack_size_l417_417118


namespace part1_part2_l417_417554

noncomputable def dot_product (x : ℝ) : ℝ :=
  (Real.cos (3 / 2 * x) * Real.cos (1 / 2 * x)) + (Real.sin (3 / 2 * x) * Real.sin (1 / 2 * x))

noncomputable def magnitude_sum (x : ℝ) : ℝ :=
  Real.sqrt ((Real.cos (3 / 2 * x))^2 + (Real.sin (3 / 2 * x))^2 + 
             (Real.cos (1 / 2 * x))^2 + (Real.sin (1 / 2 * x))^2 + 
             2 * ((Real.cos (3 / 2 * x) * Real.cos (1 / 2 * x)) + (Real.sin (3 / 2 * x) * Real.sin (1 / 2 * x))))

def max_f_x (m x : ℝ) : ℝ :=
  let t := Real.cos (x / 2)
  in -2 * t^2 + 2 * m * t - Real.cos x

theorem part1 (x : ℝ) (h₁ : x = π / 4) : 
  dot_product x = Real.sqrt 2 / 2 ∧ magnitude_sum x = Real.sqrt (2 + Real.sqrt 2) :=
sorry

theorem part2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π) (m : ℝ) : 
  max_f_x m x ≤ 
    if m > 2 then 2 * m - 3
    else if 0 ≤ m ∧ m ≤ 2 then m^2 / 2 - 1
    else -1 :=
sorry

end part1_part2_l417_417554


namespace line_pq_equation_M_4_2_fixed_point_pq_l417_417234

section geometry

variable {x y t: ℝ}

def circle (x y: ℝ) := x^2 + y^2 = 4
def line_l (x: ℝ) := x = 4
def intersection_with_x_axis (x: ℝ) := y = 0

noncomputable def point_A1 := (-2, 0)
noncomputable def point_A2 := (2, 0)
def point_M (t: ℝ) := (4, t)

noncomputable def point_P (t: ℝ) := (fun (x y: ℝ) => 
  ((y * (36 + t^2) = 24 * t) ∧ (x * (36 + t^2) = 72 - 2 * t^2))) sorry
noncomputable def point_Q (t: ℝ) := (fun (x y: ℝ) =>
  ((y * (4 + t^2) = -8 * t) ∧ (x * (4 + t^2) = 2 * t^2 - 8))) sorry

theorem line_pq_equation_M_4_2:
  let M := (4, 2) in 
  ∃ a b c : ℝ, 
    (a * (point_P 2).1 + b * (point_P 2).2 + c = 0) ∧
    (a * (point_Q 2).1 + b * (point_Q 2).2 + c = 0) ∧
    (a = 2 ∧ b = -1 ∧ c = -2) :=
begin
  sorry
end

theorem fixed_point_pq: 
  ∀ t: ℝ, 
  t ≠ sqrt 3 ∧ t ≠ -sqrt 3 → 
  let M := (4, t) in 
  ∃ a b c : ℝ, 
    (a * (point_P t).1 + b * (point_P t).2 + c = 0) ∧
    (a * (point_Q t).1 + b * (point_Q t).2 + c = 0) ∧
    (a = 8 * t / (12 - t^2) ∧ b = -1 ∧ c = 0) ∧
    ((1, 0) = (1, 0)) :=
begin
  have fixed_point := (1, 0),
  intros,
  sorry
end

end geometry

end line_pq_equation_M_4_2_fixed_point_pq_l417_417234


namespace complex_expression_l417_417238

theorem complex_expression (z : ℂ) (h : z = -1 + I) : z * conj(z) + z = 1 + I :=
by
  simp [h]
  sorry

end complex_expression_l417_417238


namespace prob_at_least_one_2_in_two_8_sided_dice_l417_417358

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end prob_at_least_one_2_in_two_8_sided_dice_l417_417358


namespace angle_C_proof_l417_417582

variable (B : ℝ) (adjacentAngle : ℝ) (verticalAngle : ℝ) (C : ℝ)

-- Given conditions
axiom angle_B : B = 120
axiom adjacent_angle : adjacentAngle = 180 - B
axiom vertical_angle : verticalAngle = 30

-- Proof problem statement
theorem angle_C_proof : B = 120 → adjacentAngle = 60 → verticalAngle = 30 → C = 90 := by
  intros hB hadj hvert
  have h1 : adjacentAngle = 180 - B := by
    simp [hB, hadj]

  have h2 : verticalAngle = 30 := by
    simp [hvert]
  
  have h_sum : 180 = verticalAngle + adjacentAngle + C := by sorry
  
  have hC : C = 180 - verticalAngle - adjacentAngle := by sorry

  simp [hB, hadj, hvert] at hC
  exact hC

end angle_C_proof_l417_417582


namespace total_profit_eq_five_thousand_l417_417053

-- Define the investments and profit share
variables (A B C P : ℝ)
variable hA : A = 5000
variable hB : B = 15000
variable hC : C = 30000
variable hPC : P = 3000

-- Define the simplified ratio
def ratio := 1 / 10 * (A + 3 * B + 6 * C)

-- Prove the total profit
theorem total_profit_eq_five_thousand : ratio = 5000 :=
by
  sorry

end total_profit_eq_five_thousand_l417_417053


namespace find_n_for_sum_2006_l417_417549

def seq_a (n : ℕ) : ℕ :=
if n = 1 then
  1599
else
  let k := Nat.sqrt n in
  if k ^ 2 < n ∧ n < (k + 1) ^ 2 then
    (k + 1) / 2 + k / 2 + 1
  else 0  -- This doesn't happen based on the problem conditions, here for completeness

theorem find_n_for_sum_2006 : ∃ n : ℕ, ∑ i in Finset.range n.succ, seq_a i = 2006 := sorry

end find_n_for_sum_2006_l417_417549


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417006

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417006


namespace find_n_l417_417973

theorem find_n
  (k m n : ℝ)
  (h_tangent : ∀ x y, y = k * x + 2 → y = x^3 + m * x + n → y = 4 ∧ x = 1) 
  (h_point : (1 : ℝ), 4) :
  n = 4 := by
  sorry

end find_n_l417_417973


namespace hyperbola_focal_length_l417_417178

theorem hyperbola_focal_length (m : ℝ) (h_pos : 0 < m)
  (h_asymptote : ∀ x y : ℝ, (√3 * x + m * y = 0) ↔ (x = 0 ∧ y = 0)) :
  2 * sqrt (3 + 1) = 4 := by
  sorry

end hyperbola_focal_length_l417_417178


namespace power_modulo_problem_l417_417120

-- Using noncomputable context for the convenience of larger computations
noncomputable theory
open_locale big_operators

theorem power_modulo_problem :
  (45 ^ 1234 - 25 ^ 1234) % 7 = 5 :=
by
  -- Definitions and simplifications according to given conditions
  have h1 : 45 % 7 = 3 := by norm_num,
  have h2 : 25 % 7 = 4 := by norm_num,
  -- Reduce the given exponential expressions modulo 7 using the properties and cycling
  have h3 : 3 ^ 1234 % 7 = 2 := by
    -- Calculation for powers of 3 modulo 7
    sorry,
  have h4 : 4 ^ 1234 % 7 = 4 := by
    -- Calculation for powers of 4 modulo 7
    sorry,
  -- Combine all the results to prove the desired congruence
  calc
    (45 ^ 1234 - 25 ^ 1234) % 7 
        = (3 ^ 1234 - 4 ^ 1234) % 7 : 
      by rw [←h1, ←h2, ←pow_mod]
        ... = (2 - 4) % 7 : 
      by rw [h3, h4]
        ... = ( -2 % 7) :
      by norm_num
        ... = 5 : 
      by norm_num

end power_modulo_problem_l417_417120


namespace prove_parabola_l417_417644

structure Parabola :=
  (focus : ℝ × ℝ)
  (intersects : (ℝ × ℝ) × (ℝ × ℝ))

noncomputable def parabola_proof (C : Parabola) (M : ℝ × ℝ) (A B: ℝ × ℝ) 
  (x_A x_B : ℝ) (k : ℝ) (F : ℝ × ℝ) : Prop :=
  let y_A := k * (x_A + 1) in
  let y_B := k * (x_B + 1) in
  C = ⟨F, (A, B)⟩ ∧
  M = (-1, 0) ∧
  x_B = 2 * x_A + 1 / 2 ∧
  |B.1 - F.1| = 2 * |A.1 - F.1| ∧
  x_A * x_B = 1 ∧
  y_A^2 = 2 * x_A ∧
  y_B^2 = 2 * x_B ∧
  x_B + 2 * x_A = (sqrt 33) / 2

-- Statement to prove
theorem prove_parabola : 
  ∃ (C : Parabola) (M : ℝ × ℝ) (A B: ℝ × ℝ) (x_A x_B : ℝ) (k : ℝ) (F : ℝ × ℝ),
    parabola_proof C M A B x_A x_B k F :=
begin
  sorry
end

end prove_parabola_l417_417644


namespace sum_abs_eq_pos_or_neg_three_l417_417912

theorem sum_abs_eq_pos_or_neg_three (x y : Real) (h1 : abs x = 1) (h2 : abs y = 2) (h3 : x * y > 0) :
    x + y = 3 ∨ x + y = -3 :=
by
  sorry

end sum_abs_eq_pos_or_neg_three_l417_417912


namespace divisible_by_2022_or_prime_sum_l417_417327

theorem divisible_by_2022_or_prime_sum (k : ℕ) 
  (prime : nat.prime ((k % 2022) + ((k + 1) % 2022) + ((k + 2) % 2022))) :
  (k % 2022 = 0) ∨ ((k + 1) % 2022 = 0) ∨ ((k + 2) % 2022 = 0) :=
sorry

end divisible_by_2022_or_prime_sum_l417_417327


namespace tire_radius_increase_is_038_l417_417446

noncomputable def original_radius : ℝ := 18
noncomputable def original_circumference : ℝ := 2 * Real.pi * original_radius
noncomputable def distance_per_rotation_orig : ℝ := original_circumference / 63360
noncomputable def odometer_reading_orig : ℝ := 550
noncomputable def rotations_orig : ℝ := odometer_reading_orig / distance_per_rotation_orig

noncomputable def odometer_reading_new : ℝ := 530
noncomputable def real_distance : ℝ := 550

noncomputable def new_rotations : ℝ := real_distance / distance_per_rotation_orig
noncomputable def new_radius : ℝ := (real_distance * distance_per_rotation_orig * 63360) / (2 * Real.pi * odometer_reading_new)

noncomputable def radius_increase : ℝ := new_radius - original_radius

theorem tire_radius_increase_is_038 :
  radius_increase ≈ 0.38 :=
by
  sorry

end tire_radius_increase_is_038_l417_417446


namespace largest_irrational_among_options_l417_417102

theorem largest_irrational_among_options :
  ∃ x, x = π ∧ (x = π ∨ x = sqrt 5) ∧ (forall y, (y = sqrt 5 ∨ y = π) -> y ≤ x) :=
by 
  sorry

end largest_irrational_among_options_l417_417102


namespace circle_properties_l417_417079

theorem circle_properties (C : ℝ) (hC : C = 36) :
  let r := C / (2 * Real.pi),
      d := 2 * r,
      A := Real.pi * r^2
  in r = 18 / Real.pi ∧ d = 36 / Real.pi ∧ A = 324 / Real.pi :=
by
  sorry

end circle_properties_l417_417079


namespace complex_exponentiation_problem_l417_417935

theorem complex_exponentiation_problem (z : ℂ) (h : z^2 + z + 1 = 0) : z^2010 + z^2009 + 1 = 0 :=
sorry

end complex_exponentiation_problem_l417_417935


namespace num_specific_ray_partitional_points_l417_417263

noncomputable def ray_partitional_points (R : set (ℝ × ℝ)) (n : ℕ) : set (ℝ × ℝ) :=
  {X | ∃ rays :
    fin n → ℝ × ℝ, (∀ r i, r ∉ ∂R) ∧ (⋂ i, X + (rays i) > (R / n))}

theorem num_specific_ray_partitional_points : 
  let R : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1} in
  let eighty_ray_points := ray_partitional_points R 80 in
  let fifty_ray_points := ray_partitional_points R 50 in
  (eighty_ray_points \ fifty_ray_points).card = 1505 := 
by sorry

end num_specific_ray_partitional_points_l417_417263


namespace sum_third_row_l417_417654

-- Definitions for the grid size and the number range
def grid_size := 17
def num_elements := grid_size * grid_size

-- Definition of the central point
def center := (grid_size / 2, grid_size / 2)

-- Spiral generation function (assuming a proper definition in the real proof)
def spiral_grid (n : ℕ) : matrix (fin grid_size) (fin grid_size) ℕ := sorry

-- Definition to determine the third row from the top
def third_row (m : matrix (fin grid_size) (fin grid_size) ℕ) : list ℕ :=
  (list.fin_range grid_size).map (λ j, m (3 % grid_size) j)

-- Definitions to find the smallest and largest numbers in the list
def smallest (l : list ℕ) : ℕ := l.minimum (by apply_instance)
def largest (l : list ℕ) : ℕ := l.maximum (by apply_instance)

-- Definition of the sum of the smallest and largest
def sum_smallest_largest (l : list ℕ) : ℕ := smallest l + largest l

-- Theorem to prove the sum in the third row
theorem sum_third_row : 
  ∀ n ≤ num_elements,
  sum_smallest_largest (third_row (spiral_grid n)) = 544 :=
by intro n h; sorry

end sum_third_row_l417_417654


namespace Diana_additional_video_game_time_l417_417870

theorem Diana_additional_video_game_time 
    (original_reward_per_hour : ℕ := 30)
    (raise_percentage : ℕ := 20)
    (hours_read : ℕ := 12)
    (minutes_per_hour : ℕ := 60) :
    let raise := (raise_percentage * original_reward_per_hour) / 100
    let new_reward_per_hour := original_reward_per_hour + raise
    let total_time_after_raise := new_reward_per_hour * hours_read
    let total_time_before_raise := original_reward_per_hour * hours_read
    let additional_minutes := total_time_after_raise - total_time_before_raise
    additional_minutes = 72 :=
by sorry

end Diana_additional_video_game_time_l417_417870


namespace math_proof_problem_l417_417939

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x - 1)^3 - a * x - b + 1

-- We will define the conditions of the problem
def cond_a := ∀ b : ℝ, (∀ x : ℝ, (f x 3 b = 0) → x ∈ set.Icc (-4 : ℝ) 0)
def cond_b := ∀ a b : ℝ, (f (2 - x) a b = 3 - f x a b) → a + b = -1
def cond_c := ∀ g : ℝ → ℝ, (∀ m : ℝ, (-5 < m ∧ m < -4) → g(x) = f x - 3*x + a*x + b)
def cond_d := ∀ a b x₀ x₁, (f' x₀ = 0 ∧ f x₀ a b = f x₁ a b ∧ x₀ ≠ x₁) → x₁ + 2*x₀ = 3

theorem math_proof_problem :
  (cond_a ∧ cond_c ∧ cond_d) :=
begin
  sorry
end

end math_proof_problem_l417_417939


namespace cauchy_schwarz_min_of_sum_eq_two_l417_417636

theorem cauchy_schwarz_min_of_sum_eq_two :
  ∀ (b : Fin 10 → ℝ), (∀ i, 0 < b i) → (∑ i, b i = 2) → (∑ i, 1 / b i ≥ 50) :=
by
  -- Define variables for the problem based on the conditions
  intro b b_pos b_sum
  -- The proof goes here, but it is omitted  
  sorry

end cauchy_schwarz_min_of_sum_eq_two_l417_417636


namespace jordan_score_points_l417_417977

theorem jordan_score_points (x y : ℕ) 
  (h1 : x + y = 40)
  (h2 : 1.2 * x + 1.0 * y = Jordan's point)
  (h3 : Jordan's point = 44) : 
  1.2 * x + (1.0 * (40 - x)) = 44
  := sorry

end jordan_score_points_l417_417977


namespace tangent_condition_sum_f_l417_417542

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

theorem tangent_condition (a : ℝ) (h : f a 1 = f a 1) (m : ℝ) : 
    (3 * a + 1 = (7 - (f a 1)) / 2) := 
    sorry

theorem sum_f (a : ℝ) (h : a = 3/7) : 
    f a (-4) + f a (-3) + f a (-2) + f a (-1) + f a 0 + 
    f a 1 + f a 2 + f a 3 + f a 4 = 9 := 
    sorry

end tangent_condition_sum_f_l417_417542


namespace badge_exchange_proof_l417_417772

-- Definitions based on the conditions
def initial_badges_Tolya : ℝ := 45
def initial_badges_Vasya : ℝ := 50

def tollya_exchange_badges (badges_Tolya : ℝ) : ℝ := 0.2 * badges_Tolya
def tollya_receive_badges (badges_Vasya : ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_exchange_badges (badges_Vasya: ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_receive_badges (badges_Tolya: ℝ) : ℝ := 0.2 * badges_Tolya

-- Vasya ended up with one badge less than Tolya after the exchange
theorem badge_exchange_proof 
  (tolya_initial badges_Tolya_initial : ℝ)
  (badges_Vasya_initial: ℝ)
  (tollya_initial_has_24: tollya_receive_badges badges_Vasya_initial)
  (vasya_initial_has_20: vasya_receive_badges badges_Tolya_initial):
  (tollya_initial = initial_badges_Tolya) ∧ (vasya_initial = initial_badges_Vasya) :=
sorry

end badge_exchange_proof_l417_417772


namespace washing_time_l417_417649

-- Define the conditions
def Mona_rate : ℝ := 1 / 3
def Jane_rate : ℝ := 1 / 4
def combined_rate : ℝ := Mona_rate + Jane_rate

-- The time t includes 1 hour of coaching plus the time they work together
theorem washing_time (t : ℝ) :
  (combined_rate * (t - 1) = 1) → t = 19 / 7 :=
by
  assume h : combined_rate * (t - 1) = 1
  have cr : combined_rate = 7 / 12 := by sorry
  sorry

end washing_time_l417_417649


namespace max_product_sum_2020_l417_417513

theorem max_product_sum_2020 (n : ℕ) (x : Fin n → ℕ) (h_sum : (Finset.univ.sum (λ i, x i)) = 2020) :
  ∃ k : ℕ, ∃ y : Fin k → ℕ, (Finset.univ.sum (λ i, y i) = 2020) ∧ (Finset.univ.prod (λ i, y i) = 2^2 * 3^672) :=
begin
  -- The proof goes here
  sorry
end

end max_product_sum_2020_l417_417513


namespace smallest_possible_range_l417_417408

theorem smallest_possible_range (x : Fin 7 → ℝ)
  (mean_condition : 15 = (∑ i, x i) / 7)
  (median_condition : x 3 = 18) -- Note: x(3) is the 4th element when indexed from 0
  : ∃ (r : ℝ), r = ∃ y : Fin 7 → ℝ, (∑ i, y i) / 7 = 15 ∧ y 3 = 18 ∧ (∀ i j, i ≤ j → y i ≤ y j) ∧ 
  ((x 6 - x 0) = r) ∧ r = 3 :=
begin
  sorry
end

end smallest_possible_range_l417_417408


namespace solve_a1_solve_a2_l417_417227

noncomputable def initial_volume := 1  -- in m^3
noncomputable def initial_pressure := 10^5  -- in Pa
noncomputable def initial_temperature := 300  -- in K

theorem solve_a1 (a1 : ℝ) : a1 = -10^5 :=
  sorry

theorem solve_a2 (a2 : ℝ) : a2 = -1.4 * 10^5 :=
  sorry

end solve_a1_solve_a2_l417_417227


namespace average_marks_l417_417682

-- Definitions
def tat_score (i_score : ℕ) : ℕ := 2 * i_score
def iva_score (d_score : ℕ) : ℕ := (3 / 5 : ℝ) * d_score

theorem average_marks :
  let D : ℕ := 90
  let I : ℕ := iva_score D
  let T : ℕ := tat_score I
  (D + I + T) / 3 = 84 :=
by {
  -- This is where the proof would go.
  -- Exact math proof steps are omitted as per the instructions.
  sorry
}

end average_marks_l417_417682


namespace average_marks_l417_417680

-- Definitions
def tat_score (i_score : ℕ) : ℕ := 2 * i_score
def iva_score (d_score : ℕ) : ℕ := (3 / 5 : ℝ) * d_score

theorem average_marks :
  let D : ℕ := 90
  let I : ℕ := iva_score D
  let T : ℕ := tat_score I
  (D + I + T) / 3 = 84 :=
by {
  -- This is where the proof would go.
  -- Exact math proof steps are omitted as per the instructions.
  sorry
}

end average_marks_l417_417680


namespace bottle_caps_sum_l417_417655

theorem bottle_caps_sum : 
  let starting_caps := 91
  let found_caps := 88
  starting_caps + found_caps = 179 :=
by
  sorry

end bottle_caps_sum_l417_417655


namespace condition_sufficient_but_not_necessary_l417_417904
noncomputable def sufficient_but_not_necessary (a b : ℝ) : Prop :=
∀ (a b : ℝ), a < 0 → -1 < b ∧ b < 0 → a + a * b < 0

-- Define the theorem stating the proof problem
theorem condition_sufficient_but_not_necessary (a b : ℝ) :
  (a < 0 ∧ -1 < b ∧ b < 0 → a + a * b < 0) ∧ 
  (a + a * b < 0 → a < 0 ∧ 1 + b > 0 ∨ a > 0 ∧ 1 + b < 0) :=
sorry

end condition_sufficient_but_not_necessary_l417_417904


namespace find_initial_tomatoes_l417_417409

-- Define the initial number of tomatoes
def initial_tomatoes (T : ℕ) : Prop :=
  T + 77 - 172 = 80

-- Theorem statement to prove the initial number of tomatoes is 175
theorem find_initial_tomatoes : ∃ T : ℕ, initial_tomatoes T ∧ T = 175 :=
sorry

end find_initial_tomatoes_l417_417409


namespace average_age_union_l417_417391

-- Definitions of our sets and their properties
variables {Person : Type} (A B C : finset Person)
variables [fintype Person]
variables (age : Person → ℝ)

-- Conditions from the problem
def average_age (S : finset Person) : ℝ :=
  (S.sum (λ p, age p)) / S.card

-- Given Conditions
axiom h1 : disjoint A B
axiom h2 : disjoint B C
axiom h3 : disjoint A C
axiom h4 : average_age age A = 34
axiom h5 : average_age age B = 26
axiom h6 : average_age age C = 48
axiom h7 : average_age age (A ∪ B) = 31
axiom h8 : average_age age (A ∪ C) = 42.5
axiom h9 : average_age age (B ∪ C) = 39

-- The proof problem
theorem average_age_union : average_age age (A ∪ B ∪ C) = 34 :=
sorry

end average_age_union_l417_417391


namespace infinite_divisors_l417_417672

theorem infinite_divisors (a : ℕ) : ∃ (a : ℕ) (a_seq : ℕ → ℕ), (∀ n : ℕ, (a_seq n)^2 ∣ 2^(a_seq n) + 3^(a_seq n)) :=
by
  sorry

end infinite_divisors_l417_417672


namespace sum_mod_inv_l417_417460

theorem sum_mod_inv :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶ + 3⁻⁷) % 17 = 6 :=
by sorry

end sum_mod_inv_l417_417460


namespace probability_at_least_one_two_l417_417357

theorem probability_at_least_one_two (dice_fair : ∀ i, 1 ≤ i ∧ i ≤ 8) (dice_count : 2):
  ∃ probability, probability = 15 / 64 := 
by
  sorry

end probability_at_least_one_two_l417_417357


namespace determine_p1_eq_24_l417_417630

def is_factor_of (p q : Polynomial ℤ) : Prop :=
  ∃ r : Polynomial ℤ, q = p * r

theorem determine_p1_eq_24 (b c : ℤ) :
  let p := Polynomial.C c + Polynomial.X * Polynomial.C b + Polynomial.X ^ 2 in
  is_factor_of p (Polynomial.X ^ 4 + 8 * Polynomial.X ^ 2 + 49) ∧
  is_factor_of p (2 * Polynomial.X ^ 4 + 5 * Polynomial.X ^ 2 + 32 * Polynomial.X + 8) →
  p.eval 1 = 24 :=
by
  sorry

end determine_p1_eq_24_l417_417630


namespace switches_connections_l417_417740

theorem switches_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 :=
by
  sorry

end switches_connections_l417_417740


namespace class_average_is_86_l417_417814

variable {BoyPercentage GirlPercentage BoyScore GirlScore ClassAverage : ℝ}

axiom h1 : BoyPercentage = 0.4
axiom h2 : GirlPercentage = 1 - BoyPercentage
axiom h3 : BoyScore = 80
axiom h4 : GirlScore = 90
axiom h5 : ClassAverage = BoyPercentage * BoyScore + GirlPercentage * GirlScore

theorem class_average_is_86 :
  ClassAverage = 86 :=
by
  have hGirlPercentage := calc
    GirlPercentage = 1 - BoyPercentage : by rw [h2]
  show ClassAverage = 86 from sorry

end class_average_is_86_l417_417814


namespace largest_additional_plates_l417_417975

theorem largest_additional_plates
  (initial_first_set_size : ℕ)
  (initial_second_set_size : ℕ)
  (initial_third_set_size : ℕ)
  (new_letters : ℕ)
  (constraint : 1 ≤ initial_second_set_size + 1 ∧ 1 ≤ initial_third_set_size + 1)
  (initial_combinations : ℕ)
  (final_combinations1 : ℕ)
  (final_combinations2 : ℕ)
  (additional_combinations : ℕ) :
  initial_first_set_size = 5 →
  initial_second_set_size = 3 →
  initial_third_set_size = 4 →
  new_letters = 4 →
  initial_combinations = initial_first_set_size * initial_second_set_size * initial_third_set_size →
  final_combinations1 = initial_first_set_size * (initial_second_set_size + 2) * (initial_third_set_size + 2) →
  final_combinations2 = (initial_first_set_size + 1) * (initial_second_set_size + 2) * (initial_third_set_size + 1) →
  additional_combinations = max (final_combinations1 - initial_combinations) (final_combinations2 - initial_combinations) →
  additional_combinations = 90 :=
by sorry

end largest_additional_plates_l417_417975


namespace number_of_favored_last_digits_l417_417878

def is_divisible_by_3_and_5 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 5 = 0

def possible_last_digits (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

theorem number_of_favored_last_digits : 
  (∃ (d : ℕ), is_divisible_by_3_and_5 d ∧ d % 10 ∈ {0, 5}) →
  (Finset.card ((Finset.filter possible_last_digits (Finset.range 10))) = 2) :=
by 
  sorry

end number_of_favored_last_digits_l417_417878


namespace length_of_string_for_circle_l417_417307

theorem length_of_string_for_circle (A : ℝ) (pi_approx : ℝ) (extra_length : ℝ) (hA : A = 616) (hpi : pi_approx = 22 / 7) (hextra : extra_length = 5) :
  ∃ (length : ℝ), length = 93 :=
by {
  sorry
}

end length_of_string_for_circle_l417_417307


namespace contemplation_duration_is_correct_l417_417338

def start_hour : Real := 9
def start_x : Real := 16.36 / 60 -- in hours
def end_hour : Real := 10
def end_y : Real := (60 - (60 / 5.5)) / 60 -- in hours
def contemplation_duration : Real := end_hour - (start_hour + start_x ) - ( 1 - end_y )

theorem contemplation_duration_is_correct :
  contemplation_duration = (32 + 45 / 60) / 60 := by
  sorry

end contemplation_duration_is_correct_l417_417338


namespace iterative_multiplication_result_l417_417384

theorem iterative_multiplication_result :
  let t := 1 in
  (t * 2 * 3 * 4 * 5 = 120) :=
by
  sorry

end iterative_multiplication_result_l417_417384


namespace some_Zims_are_not_Cims_l417_417199

variables (Zim Bim Cim : Type) 
variable all_Zims_are_Bims : ∀ z : Zim, Bim
variable some_Bims_are_not_Cims : ∃ b : Bim, ¬ ∀ (b' : Bim), Cim

theorem some_Zims_are_not_Cims : ∃ z : Zim, ¬ ∀ (z' : Zim), Cim :=
by
  sorry

end some_Zims_are_not_Cims_l417_417199


namespace max_value_of_N_l417_417146

def I_k (k : Nat) : Nat :=
  10^(k + 1) + 32

def N (k : Nat) : Nat :=
  (Nat.factors (I_k k)).count 2

theorem max_value_of_N :
  ∃ k : Nat, N k = 6 ∧ (∀ m : Nat, N m ≤ 6) :=
by
  sorry

end max_value_of_N_l417_417146


namespace product_of_lengths_equal_l417_417918

variables {n : ℕ} 
variables {A : ℕ → ℝ × ℝ} 
variables {B D : ℕ → ℝ × ℝ} 
variables {h : ℕ → ℝ}

noncomputable theory

def convex_polygon (A : ℕ → ℝ × ℝ) : Prop :=
  ∀ i, (0 ≤ i < n) → 
  ∃ B_i D_i : ℝ × ℝ, B i = B_i ∧ D i = D_i ∧ 
  let j := (i + 1) % n in 
  ∃ C_i : ℝ × ℝ, A i = (• C_i •)

def intersect_at_O (A : ℕ → ℝ × ℝ) (C : ℕ → ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  ∀ i, (0 ≤ i < n) → line_through (A i) (C i) = O

theorem product_of_lengths_equal :
  convex_polygon A ∧ 
  (∀ i, (0 ≤ i < n) → ∃ C_i, parallelogram (A i) (B i) (C i) (D i)) →
  intersect_at_O A (λ i, B i) (0, 0) →
  ∏ i in finset.range n, (distance (A i) (B i)) = ∏ i in finset.range n, (distance (A i) (D i)) :=
sorry

end product_of_lengths_equal_l417_417918


namespace required_connections_l417_417751

theorem required_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 := by
  sorry

end required_connections_l417_417751


namespace find_n_l417_417568

noncomputable def condition (n : ℕ) : Prop :=
  (1/5)^n * (1/4)^18 = 1 / (2 * 10^35)

theorem find_n (n : ℕ) (h : condition n) : n = 35 :=
by
  sorry

end find_n_l417_417568


namespace range_of_m_l417_417222

theorem range_of_m (a b m : ℝ) 
  (h_vertex_on_line : ∀ a, ∃ b, a^2 + b = ma - (a^2 + b) - 2m + 1)
  (h_common_point : ∃ x, -x^2 + 2ax + b = x^2) :
  m ≥ 5/2 ∨ m ≤ 3/2 :=
sorry

end range_of_m_l417_417222


namespace expected_second_ace_position_l417_417084

noncomputable def expected_position_of_second_ace (n : ℕ) : ℝ :=
((n + 1) : ℝ) / 2

theorem expected_second_ace_position (n : ℕ) (h : 2 < n) :
  expected_position_of_second_ace n = (n + 1) / 2 := by
sorry

end expected_second_ace_position_l417_417084


namespace max_fn_limit_l417_417891

noncomputable theory

open Classical

theorem max_fn_limit : 
  ∀ (n : ℕ), (∃ f_n : (fin n → ℝ) → ℝ, 
  (∀ x : fin n → ℝ, (∀ i, 0 ≤ x i) → f_n x = 
    ∑ i in finset.range n, x i / (1 + ∑ j in finset.range n, x j)^2) 
  ∧ ∃ m : ℝ, ∀ x : fin n → ℝ, (∀ i, 0 ≤ x i) → f_n x ≤ m 
  ∧ m = 1
  ∧ ∃ L : ℝ, (∀ m : ℝ, (∃ n : ℕ, ∀ i : fin n, 0 ≤ x i → f_n x ≤ m) → L = 1)) :=
sorry

end max_fn_limit_l417_417891


namespace problem_part1_problem_part2_l417_417275

noncomputable def f (x a : ℝ) := |x - a| + x

theorem problem_part1 (a : ℝ) (h_a : a = 1) :
  {x : ℝ | f x a ≥ x + 2} = {x : ℝ | x ≥ 3} ∪ {x : ℝ | x ≤ -1} :=
by 
  simp [h_a, f]
  sorry

theorem problem_part2 (a : ℝ) (h_solution : {x : ℝ | f x a ≤ 3 * x} = {x : ℝ | x ≥ 2}) :
  a = 6 :=
by
  simp [f] at h_solution
  sorry

end problem_part1_problem_part2_l417_417275


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417005

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417005


namespace length_error_probability_l417_417998

theorem length_error_probability
  (μ σ : ℝ)
  (X : ℝ → ℝ)
  (h_norm_dist : ∀ x : ℝ, X x = (Real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2)) / (σ * Real.sqrt (2 * Real.pi))))
  (h_max_density : X 0 = 1 / (3 * Real.sqrt (2 * Real.pi)))
  (P : Set ℝ → ℝ)
  (h_prop1 : P {x | μ - σ < x ∧ x < μ + σ} = 0.6826)
  (h_prop2 : P {x | μ - 2 * σ < x ∧ x < μ + 2 * σ} = 0.9544) :
  P {x | 3 < x ∧ x < 6} = 0.1359 :=
sorry

end length_error_probability_l417_417998


namespace vector_dot_product_zero_l417_417782

theorem vector_dot_product_zero {a b : ℝ^3} :
  ∥a + b∥ = ∥a - b∥ → a ⬝ b = 0 :=
by 
sorry

end vector_dot_product_zero_l417_417782


namespace number_and_reverse_sum_is_99_l417_417309

noncomputable def sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) 
(h3 : abs ((10 * a + b) - (10 * b + a)) = 3 * abs (a - b)) (h4 : a ≠ b) : ℕ :=
(10 * a + b) + (10 * b + a)

theorem number_and_reverse_sum_is_99 {a b : ℕ} (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
(h3 : abs ((10 * a + b) - (10 * b + a)) = 3 * abs (a - b)) (h4 : a ≠ b) :
sum_of_number_and_reverse a b h1 h2 h3 h4 = 99 :=
sorry

end number_and_reverse_sum_is_99_l417_417309


namespace VivianMailApril_l417_417773

variable (piecesMailApril piecesMailMay piecesMailJune piecesMailJuly piecesMailAugust : ℕ)

-- Conditions
def condition_double_monthly (a b : ℕ) : Prop := b = 2 * a

axiom May : piecesMailMay = 10
axiom June : piecesMailJune = 20
axiom July : piecesMailJuly = 40
axiom August : piecesMailAugust = 80

axiom patternMay : condition_double_monthly piecesMailApril piecesMailMay
axiom patternJune : condition_double_monthly piecesMailMay piecesMailJune
axiom patternJuly : condition_double_monthly piecesMailJune piecesMailJuly
axiom patternAugust : condition_double_monthly piecesMailJuly piecesMailAugust

-- Statement to prove
theorem VivianMailApril :
  piecesMailApril = 5 :=
by
  sorry

end VivianMailApril_l417_417773


namespace time_to_pick_sugar_snap_peas_l417_417853

theorem time_to_pick_sugar_snap_peas (pea_count1 pea_count2 : ℕ) (time1 : ℕ) :
  (pea_count1 = 56 ∧ time1 = 7 ∧ pea_count2 = 72) →
  let rate := pea_count1 / time1 in
  (pea_count2 / rate = 9) :=
by
  intros h
  let ⟨h1, h2, h3⟩ := h
  sorry

end time_to_pick_sugar_snap_peas_l417_417853


namespace seating_arrangements_l417_417339

-- Definitions for the conditions
variables (n k : ℕ)

-- Total seats in a row
def total_seats := 8

-- Number of people to be seated
def people := 3

-- Effective seats available considering the constraints
def effective_seats := total_seats - 2 * people

-- Number of spaces for seating calculation
def spaces := effective_seats + 1

-- Prove the number of seating arrangements equals 24
theorem seating_arrangements : effective_seats = 6 → spaces = 4 → nat.fact 4 / nat.fact (4 - 3) = 24 :=
by
  intros h1 h2
  rw h1
  rw h2
  simp
  sorry

end seating_arrangements_l417_417339


namespace books_selection_l417_417562

theorem books_selection 
  (num_mystery : ℕ)
  (num_fantasy : ℕ)
  (num_biographies : ℕ)
  (Hmystery : num_mystery = 5)
  (Hfantasy : num_fantasy = 4)
  (Hbiographies : num_biographies = 6) :
  (num_mystery * num_fantasy * num_biographies = 120) :=
by
  -- Proof goes here
  sorry

end books_selection_l417_417562


namespace bisect_A_N_A_l417_417800

-- Define the points and necessary conditions
variables {A B C O N D A' : Type} [affine_space ℝ Type]

-- Defining the reflection concept
def reflection (P Q R : Type) : Type := sorry

-- Circumcenter definition
def is_circumcenter (O A B C : Type) : Prop := sorry

-- Nine-point circle center definition
def is_nine_point_center (N A B C : Type) : Prop := sorry

-- Intersect definition
def intersects (l m : Type) (P : Type) : Prop := sorry

-- Extend definition
def extend (P Q : Type) : Type := sorry

-- Bisection definition
def bisects (L P Q : Type) : Prop := sorry

-- Lean statement
theorem bisect_A_N_A'_D (h_circ : is_circumcenter O A B C) 
  (h_nine : is_nine_point_center N A B C)
  (h_reflect : A' = reflection A B C)
  (h_intersect : intersects (extend A O) (circumcircle B O C) D)
  : bisects (line_through A N) A' D :=
sorry

end bisect_A_N_A_l417_417800


namespace radius_of_circle_S_l417_417306

theorem radius_of_circle_S (a b : ℕ) (h₁ : a = 5) (h₂ : b = 12) :
  let c := Int.sqrt (a^2 + b^2)
      r := (a + b - c) / 2
      OC := r * Real.sqrt 2
  in (OC - r = r * Real.sqrt 2 - 2) ∨ (OC + r = r * Real.sqrt 2 + 2) := 
by
  sorry

end radius_of_circle_S_l417_417306


namespace integer_exponent_condition_l417_417898

theorem integer_exponent_condition : 
  (∃ (n : ℤ), (((n : ℂ) + complex.I) ^ 6).im = 0) → 
  ∃ (n : ℤ), ((n + complex.I) ^ 6).re ∈ ℤ :=
by
  sorry

end integer_exponent_condition_l417_417898


namespace angle_bisectors_isosceles_trapezoid_l417_417846

theorem angle_bisectors_isosceles_trapezoid
  (ABCD KLGH : Type)
  [quadrilateral ABCD] [quadrilateral KLGH]
  (A B C D K L G H : Point)
  (angle_A angle_B angle_C angle_D : Angle)
  (h_eq : angle_A = angle_C)
  (angle_bisectors : (∃ (da dg : line), da.is_angle_bisector angle_A ∧ dg.is_angle_bisector angle_D)
                    ∧ (∃ (dc dh : line), dc.is_angle_bisector angle_C ∧ dh.is_angle_bisector angle_D))
  (K L G H : Point)
  (formed_by_bisectors : (∃ (K' L' G' H': Point), KLGH = quadrilateral K' L' G' H' →
                          K' ∈ intersection ((da ∩ dc) ∧  L' ∈ intersection ((da ∩ dh))
                          ∧  G' ∈ intersection ((dg ∩ dc)) ∧ H' ∈ intersection ((dg ∩ dh))))
: is_isosceles_trapezoid K L G H :=
sorry

end angle_bisectors_isosceles_trapezoid_l417_417846


namespace nat_divisors_5n_l417_417417

theorem nat_divisors_5n (n : ℕ) (h : nat.tau (36 * n^2) = 51) : nat.tau (5 * n) = 16 :=
by
  sorry

end nat_divisors_5n_l417_417417


namespace coeff_x_neg_3_binom_expansion_l417_417243

theorem coeff_x_neg_3_binom_expansion :
  let gen_term (n r : ℕ) := binom n r * (-1)^r * 4^(n-r) * x^(2*(n-r) - r)
  ∃ n r : ℕ, n = 6 ∧ r = 5 ∧ gen_term n r = -24 :=
by
  sorry

end coeff_x_neg_3_binom_expansion_l417_417243


namespace probability_three_primes_in_ten_consecutive_l417_417444

def primes : Finset ℕ := Finset.filter Nat.prime (Finset.range 51)

-- Define if a sequence of 10 consecutive numbers contains exactly 3 primes
def has_exactly_three_primes (s : Finset ℕ) : Prop := (s.filter Nat.prime).card = 3

-- Compute the number of 10-number segments that have exactly 3 primes
noncomputable def valid_segments_count : ℕ :=
  Finset.card (Finset.filter (λ s, has_exactly_three_primes (s.attach)) (Finset.range 41))

-- Total number of possible 10-number segments from 1 to 50
def total_segments_count : ℕ := 41

-- Probability computed as a fraction
noncomputable def probability_exactly_three_primes : ℚ :=
  (valid_segments_count : ℚ) / (total_segments_count : ℚ)

-- The theorem statement
theorem probability_three_primes_in_ten_consecutive :
  probability_exactly_three_primes = 22 / 41 :=
sorry

end probability_three_primes_in_ten_consecutive_l417_417444


namespace total_triangles_l417_417858

-- Definitions for the problem
def Vertex : Type := ℝ×ℝ
def Triangle (A B C : Vertex) : Prop :=
  ∃ T : set Vertex, T = {A, B, C} ∧ T.card = 3

variables (A B C M N L G : Vertex)

-- Condition definitions
def equilateral_triangle (A B C : Vertex) : Prop :=
  dist(A, B) = dist(B, C) ∧ dist(B, C) = dist(C, A)

def midpoint (X Y M : Vertex) : Prop :=
  2 * M = X + Y

def medians_concurrent (A B C G : Vertex) : Prop :=
  ∃ M N L, midpoint A B M ∧ midpoint B C N ∧ midpoint C A L ∧
  (2/3) * M + (1/3) * G = A ∧
  (2/3) * N + (1/3) * G = B ∧
  (2/3) * L + (1/3) * G = C

-- Theorem statement: given the conditions, the total number of triangles is 13
theorem total_triangles (h₁: equilateral_triangle A B C)
                        (h₂: midpoint A C M)
                        (h₃: midpoint B C N)
                        (h₄: midpoint A B L)
                        (h₅: medians_concurrent A B C G) :
  ∃ total, total = 13 := sorry

end total_triangles_l417_417858


namespace acute_angle_intersection_l417_417390

section
variables (A B C D E S : Type) [RegularPentagon A B C D E]
variables (AC BD : Type) [Diagonal AC] [Diagonal BD]
variables [Intersection AC BD S]

theorem acute_angle_intersection (A B C D E S : Type) [RegularPentagon A B C D E] 
  [Diagonal AC] [Diagonal BD] [Intersection AC BD S] : 
  angle C S D = 72 :=
sorry
end

end acute_angle_intersection_l417_417390


namespace proof_problem_l417_417195

-- Define the parametric equation of line l
def parametric_line (m t : ℝ) : ℝ × ℝ :=
  (m + (Real.sqrt 2)/2 * t, (Real.sqrt 2)/2 * t)

-- Define the rectangular coordinate equation of curve C
def curve_C (x y : ℝ) : Prop :=
  (x^2)/12 + (y^2)/4 = 1

-- Define the left focus of the curve C
def left_focus (x y : ℝ) : Prop :=
  x = -2 * Real.sqrt 2 ∧ y = 0

-- Define the condition that the focus is on the line l
def focus_on_line (m t : ℝ) : Prop :=
  left_focus (m + (Real.sqrt 2)/2 * t) ((Real.sqrt 2)/2 * t)

-- Define the value to be proven
def value_AF_BF (t1 t2 : ℝ) : ℝ :=
  (abs (t1 - t2)) / (abs (t1 * t2))

-- The main theorem combining the conditions and results
theorem proof_problem :
  ∃ m t1 t2 : ℝ, 
    (focus_on_line m t1) ∧
    (curve_C (-2 * Real.sqrt 2 + (Real.sqrt 2)/2 * t1) ((Real.sqrt 2)/2 * t1)) ∧
    (curve_C (-2 * Real.sqrt 2 + (Real.sqrt 2)/2 * t2) ((Real.sqrt 2)/2 * t2)) ∧
    m = -2 * Real.sqrt 2 ∧
    t1 = 1 - Real.sqrt 3 ∧
    t2 = 1 + Real.sqrt 3 ∧
    value_AF_BF t1 t2 = Real.sqrt 3 :=
sorry

end proof_problem_l417_417195


namespace prove_incorrect_proposition_l417_417443

-- Define the propositions, axioms and conditions
def propA (p q : Prop) : Prop := p → ¬q → (p ∨ ¬q)
def propB (a b : ℕ) : Prop := a + b ≠ 7 → (a ≠ 2 ∨ b ≠ 5)
def propC_neg := ¬(x^2 - x = 0 → x = 0 ∨ x = 1) → (x^2 - x ≠ 0 → x ≠ 0 ∧ x ≠ 1)
def propD_neg (p : Prop) := (∀ x > 0, p x)

-- Define the equivalence proof problem
theorem prove_incorrect_proposition :
  let x, y, z, w : Prop := sorry in       -- Let variables as placeholders for propositions
  let A : propA x y := sorry in
  let B : propB 2 5 := sorry in
  let C : propC_neg := sorry in
  let D : propD_neg (λ x : ℕ, sorry) := sorry in
  ¬C := by sorry -- Prove without proof

end prove_incorrect_proposition_l417_417443


namespace fleas_initial_minus_final_l417_417821

theorem fleas_initial_minus_final (F : ℕ) (h : F / 16 = 14) :
  F - 14 = 210 :=
sorry

end fleas_initial_minus_final_l417_417821


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417022

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417022


namespace problem_statement_l417_417101

-- Create the necessary variables
variables {α : Type*} [OrderedField α] {a b : α}

theorem problem_statement (h : a > b) : a^3 > b^3 :=
sorry

end problem_statement_l417_417101


namespace exists_permutation_multiple_of_7_l417_417268

def digits : List ℕ := [1, 3, 7, 9]

def is_multiple_of_7 (n : ℕ) : Prop := n % 7 = 0

theorem exists_permutation_multiple_of_7 :
  ∃ (perm : List ℕ), perm.permutations.any (λ p, is_multiple_of_7 (List.foldl (λ acc d, acc * 10 + d) 0 p)) := 
sorry

end exists_permutation_multiple_of_7_l417_417268


namespace number_of_white_squares_l417_417416

theorem number_of_white_squares (n : ℕ) (hn : n = 20) : 
  let N := 3 * n in
  (N % 2 = 0) → (N / 2 = 30) :=
by
  intro h
  rw hn
  sorry

end number_of_white_squares_l417_417416


namespace equation_of_ellipse_C1_point_Q_on_line_l417_417169

-- Given Conditions
def F1 := (0, 1 : ℝ)
def F2 := (0, -1 : ℝ)
def P := (1, 3 : ℝ)
def M := (-2*sqrt(6) / 3, 2 / 3 : ℝ)
def λ : ℝ := sorry -- Any non-zero, non plus/minus one value

-- The ellipse with given foci and point on it
def ellipse_C1 (x y : ℝ) : Prop := (y^2 / 4 + x^2 / 3 = 1)

-- The parabola with F1 as a focus
def parabola_C2 (x y : ℝ) : Prop := (x^2 = 4 * y)

-- Prove the equation of the ellipse
theorem equation_of_ellipse_C1 :
  ellipse_C1 M.1 M.2 :=
by sorry

-- Prove that point Q always lies on the line x + 3y = 3
theorem point_Q_on_line (Q : ℝ × ℝ) (A B : ℝ × ℝ) (hA_on_circle : A.1^2 + A.2^2 = 3) (hB_on_circle : B.1^2 + B.2^2 = 3) (hAP : ℝ × ℝ) (hPB : ℝ × ℝ) :
  (∀ Q (hAP = -λ * hPB) (hAQ = λ * hQB), Q.1 + 3 * Q.2 = 3) :=
by sorry

end equation_of_ellipse_C1_point_Q_on_line_l417_417169


namespace students_selected_juice_l417_417453

def fraction_of_students_choosing_juice (students_selected_juice_ratio students_selected_soda_ratio : ℚ) : ℚ :=
  students_selected_juice_ratio / students_selected_soda_ratio

def num_students_selecting (students_selected_soda : ℕ) (fraction_juice : ℚ) : ℚ :=
  fraction_juice * students_selected_soda

theorem students_selected_juice (students_selected_soda : ℕ) : students_selected_soda = 120 ∧
    (fraction_of_students_choosing_juice 0.15 0.75) = 1/5 →
    num_students_selecting students_selected_soda (fraction_of_students_choosing_juice 0.15 0.75) = 24 :=
by
  intros h
  sorry

end students_selected_juice_l417_417453


namespace switch_connections_l417_417745

theorem switch_connections (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 :=
by
  rw [h1, h2]
  norm_num
  sorry

end switch_connections_l417_417745


namespace not_consecutive_1_2_3_4_l417_417713

def f (x : ℤ) : ℤ :=
  if even x then 0 else 1

def a (n : ℕ) : ℕ
| 0     := 1
| 1     := 9
| 2     := 8
| 3     := 1
| (n+4) := (a n + a (n+1) + a (n+2) + a (n+3)) % 10

def b (n : ℕ) : ℤ := f (a n)

theorem not_consecutive_1_2_3_4 : ¬(∃ n, a n = 1 ∧ a (n+1) = 2 ∧ a (n+2) = 3 ∧ a (n+3) = 4) :=
sorry

end not_consecutive_1_2_3_4_l417_417713


namespace unknown_rate_of_blankets_l417_417790

theorem unknown_rate_of_blankets :
  ∀ (x : ℕ),
  (3 * 100 + 5 * 150 + 2 * x) / 10 = 160 → x = 275 :=
by { intros, sorry }

end unknown_rate_of_blankets_l417_417790


namespace tan_960_eq_sqrt_3_l417_417330

theorem tan_960_eq_sqrt_3 : Real.tan (960 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

end tan_960_eq_sqrt_3_l417_417330


namespace constant_c_for_local_maximum_l417_417215

theorem constant_c_for_local_maximum (c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x * (x - c) ^ 2) (h2 : ∃ δ > 0, ∀ x, |x - 2| < δ → f x ≤ f 2) : c = 6 :=
sorry

end constant_c_for_local_maximum_l417_417215


namespace additional_cars_can_fit_l417_417586

-- Given definitions and conditions
variable (total_cars : Nat)
variable (levels : Nat)
variable (parked_cars : Nat)

-- Specific conditions for our problem
def total_cars := 425
def levels := 5
def parked_cars := 23

-- Goal statement
theorem additional_cars_can_fit : (total_cars / levels) - parked_cars = 62 := by
  sorry

end additional_cars_can_fit_l417_417586


namespace average_marks_l417_417687

theorem average_marks (D I T : ℕ) 
  (hD : D = 90)
  (hI : I = (3 * D) / 5)
  (hT : T = 2 * I) : 
  (D + I + T) / 3 = 84 :=
by
  sorry

end average_marks_l417_417687


namespace Coe_speed_theorem_l417_417967

-- Define the conditions
def Teena_speed : ℝ := 55
def initial_distance_behind : ℝ := 7.5
def time_hours : ℝ := 1.5
def distance_ahead : ℝ := 15

-- Define Coe's speed
def Coe_speed := 50

-- State the theorem
theorem Coe_speed_theorem : 
  let distance_Teena_covers := Teena_speed * time_hours
  let total_relative_distance := distance_Teena_covers + initial_distance_behind
  let distance_Coe_covers := total_relative_distance - distance_ahead
  let computed_Coe_speed := distance_Coe_covers / time_hours
  computed_Coe_speed = Coe_speed :=
by sorry

end Coe_speed_theorem_l417_417967


namespace problem_part_1_problem_part_2_l417_417929

noncomputable def sequence_geometric (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) : Prop :=
  S n + n = 2 * a n

theorem problem_part_1 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : ∀ n, sequence_geometric S a n) :
  ∃ r, ∀ n, a (n + 1) + 1 = r * (a n + 1) ∧ r = 2 :=
sorry

noncomputable def b_sequence (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  2^n / (a n * a (n + 1))

noncomputable def T (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (finset.range n).sum b

theorem problem_part_2 (S : ℕ → ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (h : ∀ n, sequence_geometric S a n) :
  b = b_sequence a → T = λ n, (finset.range n).sum b → T n < 13 / 14 → n = 2 :=
sorry

end problem_part_1_problem_part_2_l417_417929


namespace ratio_of_salt_to_flour_l417_417342

def ratio_number_teaspoons_salt_to_cups_flour (water flour total : ℕ) : ℕ × ℕ :=
  let salt := total - (water + flour) in
  (salt, flour)

theorem ratio_of_salt_to_flour
  (water flour total : ℕ)
  (h_w : water = 10)
  (h_f : flour = 16)
  (h_t : total = 34) :
  ratio_number_teaspoons_salt_to_cups_flour water flour total = (1, 2) :=
by
  sorry

end ratio_of_salt_to_flour_l417_417342


namespace remainder_when_x_plus_3uy_divided_by_y_eq_v_l417_417301

theorem remainder_when_x_plus_3uy_divided_by_y_eq_v
  (x y u v : ℕ) (h_pos_y : 0 < y) (h_division_algo : x = u * y + v) (h_remainder : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end remainder_when_x_plus_3uy_divided_by_y_eq_v_l417_417301


namespace vehicle_wax_initial_amount_l417_417604

theorem vehicle_wax_initial_amount
  (wax_car wax_suv wax_spilled wax_left original_amount : ℕ)
  (h_wax_car : wax_car = 3)
  (h_wax_suv : wax_suv = 4)
  (h_wax_spilled : wax_spilled = 2)
  (h_wax_left : wax_left = 2)
  (h_total_wax_used : wax_car + wax_suv = 7)
  (h_wax_before_waxing : wax_car + wax_suv + wax_spilled = 9) :
  original_amount = 11 := by
  sorry

end vehicle_wax_initial_amount_l417_417604


namespace compare_abc_l417_417906

noncomputable def a : ℝ := 3 ^ 0.8
noncomputable def b : ℝ := 0.8 ^ 3
noncomputable def c : ℝ := Real.log 0.8 / Real.log 3

theorem compare_abc : a > b ∧ b > c :=
by
  sorry

end compare_abc_l417_417906


namespace classroom_not_1_hectare_l417_417303

def hectare_in_sq_meters : ℕ := 10000
def classroom_area_approx : ℕ := 60

theorem classroom_not_1_hectare : ¬ (classroom_area_approx = hectare_in_sq_meters) :=
by 
  sorry

end classroom_not_1_hectare_l417_417303


namespace constant_c_square_of_binomial_l417_417968

theorem constant_c_square_of_binomial (c : ℝ) (h : ∃ d : ℝ, (3*x + d)^2 = 9*x^2 - 18*x + c) : c = 9 :=
sorry

end constant_c_square_of_binomial_l417_417968


namespace proof_problem_l417_417523

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

theorem proof_problem (h_even : even_function f)
                      (h_period : ∀ x, f (x + 2) = -f x)
                      (h_incr : increasing_on f (-2) 0) :
                      periodic_function f 4 ∧ symmetric_about f 2 :=
by { sorry }

end proof_problem_l417_417523


namespace shortest_player_height_l417_417722

-- let h_tall be the height of the tallest player
-- let h_short be the height of the shortest player
-- let diff be the height difference between the tallest and the shortest player

variable (h_tall h_short diff : ℝ)

-- conditions given in the problem
axiom tall_player_height : h_tall = 77.75
axiom height_difference : diff = 9.5
axiom height_relationship : h_tall = h_short + diff

-- the statement we need to prove
theorem shortest_player_height : h_short = 68.25 := by
  sorry

end shortest_player_height_l417_417722


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417039

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417039


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417024

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417024


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417021

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417021


namespace number_of_solutions_l417_417172

theorem number_of_solutions (n : ℕ) (h : 0 < 14 * 21 * n) :
  ∃ m : ℕ, m = n^3 - n ∧ ∀ x : ℝ, 1 ≤ x ∧ x < n →
  (x^3 - ⌊x^3⌋ = (x - ⌊x⌋)^3 ↔ x ∈ {x | 1 ≤ x ∧ x < n}) :=
by
  sorry

end number_of_solutions_l417_417172


namespace max_value_f_1_max_value_f_2_max_value_f_3_l417_417943
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x

theorem max_value_f_1 (m : ℝ) (h : m ≤ 1 / Real.exp 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ 1 - m * Real.exp 1 :=
sorry

theorem max_value_f_2 (m : ℝ) (h1 : 1 / Real.exp 1 < m) (h2 : m < 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -Real.log m - 1 :=
sorry

theorem max_value_f_3 (m : ℝ) (h : m ≥ 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -m :=
sorry

end max_value_f_1_max_value_f_2_max_value_f_3_l417_417943


namespace part1_part2_l417_417528

-- Define the conditions
def proposition_A (t : ℝ) : Prop :=
  ∃ (y x : ℝ), (5 - t > 0) ∧ (t - 1 > 0) ∧ (5 - t > t - 1) ∧ 
  (e = √((5 - t) - (t - 1)) / √(5 - t)) ∧ e = √6 / 3

def proposition_B (t : ℝ) : Prop :=
  t^2 - 3 * t - 4 < 0

-- Prove these
theorem part1 (t : ℝ) : proposition_A t → t = 2 :=
  sorry

theorem part2 (t : ℝ) : proposition_A t → proposition_B t ∧ (proposition_A t → proposition_B t) :=
  sorry

end part1_part2_l417_417528


namespace sum_of_money_l417_417727

-- Conditions
def mass_record_coin_kg : ℝ := 100  -- 100 kg
def mass_one_pound_coin_g : ℝ := 10  -- 10 g

-- Conversion factor
def kg_to_g : ℝ := 1000

-- Question: Prove the sum of money in £1 coins that weighs the same as the record-breaking coin is £10,000.
theorem sum_of_money 
  (mass_record_coin_g := mass_record_coin_kg * kg_to_g)
  (number_of_coins := mass_record_coin_g / mass_one_pound_coin_g) 
  (sum_of_money := number_of_coins) : 
  sum_of_money = 10000 :=
  sorry

end sum_of_money_l417_417727


namespace numbers_not_divisible_by_5_or_7_l417_417069

theorem numbers_not_divisible_by_5_or_7 (n : ℕ) (h : n = 2006) : 
  let count_div_5 := ((∑ i in Finset.range n.succ, if i % 5 = 0 then 1 else 0) : ℕ)
  let count_div_7 := ((∑ i in Finset.range n.succ, if i % 7 = 0 then 1 else 0) : ℕ)
  let count_div_35 := ((∑ i in Finset.range n.succ, if i % 35 = 0 then 1 else 0) : ℕ)
  let count_div_5_or_7 := count_div_5 + count_div_7 - count_div_35
  let total := n
  let count_neither := total - count_div_5_or_7
  count_neither = 1376 :=
by
  sorry

end numbers_not_divisible_by_5_or_7_l417_417069


namespace four_consecutive_product_divisible_by_12_l417_417000

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l417_417000


namespace sin_C_of_arith_prog_angles_l417_417937

theorem sin_C_of_arith_prog_angles (A B C a b : ℝ) (h_abc : A + B + C = Real.pi)
  (h_arith_prog : 2 * B = A + C) (h_a : a = Real.sqrt 2) (h_b : b = Real.sqrt 3) :
  Real.sin C = (Real.sqrt 2 + Real.sqrt 6) / 4 :=
sorry

end sin_C_of_arith_prog_angles_l417_417937


namespace part_a_l417_417806

theorem part_a (n x : ℕ) (h1 : n > 0) (h2 : x > 0) (h3 : 2^n + 1 = x^2) : (n = 3 ∧ x = 3) :=
begin
  sorry
end

end part_a_l417_417806


namespace sqrt_sum_difference_product_l417_417324

open Real

theorem sqrt_sum_difference_product :
  (sqrt 3 + sqrt 2) * (sqrt 3 - sqrt 2) = 1 := by
  sorry

end sqrt_sum_difference_product_l417_417324


namespace range_of_x_l417_417464

def f (x : ℝ) := -x^2 + 6 / (2 + |x|)

theorem range_of_x (x : ℝ) : f (2*x - 3) < f 1 ↔ x ∈ Iio 1 ∪ Ioi 2 :=
by
  sorry

end range_of_x_l417_417464


namespace firm_sold_quantity_l417_417107

def totalCost (Q : ℕ) : ℕ := 4 * Q

theorem firm_sold_quantity (P Q : ℕ) (TC : ℕ) : P = 4 → TC = 12 → totalCost Q = TC → Q = 3 := 
by
  intros hp htc hqtc
  rw [hp] at hqtc
  rw [← totalCost] at hqtc
  have ht : 4 * Q = 12 := by assumption
  sorry

end firm_sold_quantity_l417_417107


namespace problem_solution_l417_417572

-- Define the problem conditions and state the theorem
variable (a b : ℝ)
variable (h1 : a^2 - 4 * a + 3 = 0)
variable (h2 : b^2 - 4 * b + 3 = 0)
variable (h3 : a ≠ b)

theorem problem_solution : (a+1)*(b+1) = 8 := by
  sorry

end problem_solution_l417_417572


namespace third_dog_average_daily_miles_l417_417457

/-- Bingo has three dogs. On average, they walk a total of 100 miles a week.

    The first dog walks an average of 2 miles a day.

    The second dog walks 1 mile if it is an odd day of the month and 3 miles if it is an even day of the month.

    Considering a 30-day month, the goal is to find the average daily miles of the third dog. -/
theorem third_dog_average_daily_miles :
  let total_dogs := 3
  let weekly_total_miles := 100
  let first_dog_daily_miles := 2
  let second_dog_odd_day_miles := 1
  let second_dog_even_day_miles := 3
  let days_in_month := 30
  let odd_days_in_month := 15
  let even_days_in_month := 15
  let weeks_in_month := days_in_month / 7
  let first_dog_monthly_miles := days_in_month * first_dog_daily_miles
  let second_dog_monthly_miles := (second_dog_odd_day_miles * odd_days_in_month) + (second_dog_even_day_miles * even_days_in_month)
  let third_dog_monthly_miles := (weekly_total_miles * weeks_in_month) - (first_dog_monthly_miles + second_dog_monthly_miles)
  let third_dog_daily_miles := third_dog_monthly_miles / days_in_month
  third_dog_daily_miles = 10.33 :=
by
  sorry

end third_dog_average_daily_miles_l417_417457


namespace a_n_prime_l417_417260

def a (n : ℕ) : ℕ := |n * (n + 1) - 19|

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

theorem a_n_prime (n : ℕ) (h1 : ∀ k < n, Nat.gcd (a n) (a k) = 1) (h2 : n ≠ 4) : 
  is_prime (a n) := by
sorry

end a_n_prime_l417_417260


namespace A_investment_l417_417827

theorem A_investment (B C profit A_profit : ℕ) (hB : B = 4200) (hC : C = 10500) (hprofit : profit = 12200) (hA_profit : A_profit = 3660) :
  ∃ x : ℕ, x / (x + B + C) = A_profit / profit → x = 6725 :=
by
  use (3660 * 15700) / 8560
  split
  { -- Prove the ratio holds
    sorry },
  { -- Prove the value equals 6725
    reflexivity }

end A_investment_l417_417827


namespace series_sum_l417_417113

open BigOperators

theorem series_sum :
  (∑ n in Finset.range 99, (1 : ℝ) / ((n + 1) * (n + 2))) = 99 / 100 :=
by
  sorry

end series_sum_l417_417113


namespace staffing_arrangements_l417_417505

-- Definitions based on problem conditions
def male_teachers := 5
def female_teachers := 4
def total_teachers := 9
def selected_teachers := 3

-- Calculate the permutations
def perm (n k : ℕ) : ℕ := nat.factorial n / nat.factorial (n - k)

-- Total permutations without gender constraint
def P_9_3 := perm total_teachers selected_teachers

-- Invalid permutations (all male or all female)
def P_5_3 := perm male_teachers selected_teachers
def P_4_3 := perm female_teachers selected_teachers

-- Valid permutations with at least one male and one female
def valid_permutations := P_9_3 - (P_5_3 + P_4_3)

-- The proof
theorem staffing_arrangements : valid_permutations = 420 := by
  sorry

end staffing_arrangements_l417_417505


namespace unique_membership_card_designs_fraction_division_result_l417_417081

noncomputable def M : ℕ :=
  120 + 360 + 120

theorem unique_membership_card_designs : M = 600 :=
by
  -- The specific computations:
  have step1 : 120 = 120 := rfl
  have step2 : 360 = 360 := rfl
  have step3 : 120 = 120 := rfl

  -- Therefore, M = 120 + 360 + 120 = 600:
  show M = 600,
  calc M = 120 + 360 + 120 := sorry
     ... = 600 := by sorry

theorem fraction_division_result : M / 6 = 100 :=
by
  -- Given unique membership card designs theorem:
  have hM : M = 600 := by sorry

  -- Substituting and performing division:
  show M / 6 = 100,
  calc
    M / 6 = 600 / 6 := by rw [hM]
        ... = 100    := by norm_num

#eval fraction_division_result -- This will give the result of the computation.

end unique_membership_card_designs_fraction_division_result_l417_417081


namespace radius_of_circumcircle_of_triangle_l417_417534

theorem radius_of_circumcircle_of_triangle (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧ (∃ (R : ℝ), R = 2.5) :=
by {
  sorry
}

end radius_of_circumcircle_of_triangle_l417_417534


namespace probability_at_least_one_two_l417_417354

theorem probability_at_least_one_two (dice_fair : ∀ i, 1 ≤ i ∧ i ≤ 8) (dice_count : 2):
  ∃ probability, probability = 15 / 64 := 
by
  sorry

end probability_at_least_one_two_l417_417354


namespace percentage_sum_l417_417979

noncomputable def womenWithRedHairBelow30 : ℝ := 0.07
noncomputable def menWithDarkHair30OrOlder : ℝ := 0.13

theorem percentage_sum :
  womenWithRedHairBelow30 + menWithDarkHair30OrOlder = 0.20 := by
  sorry -- Proof is omitted

end percentage_sum_l417_417979


namespace childrens_cookbook_cost_l417_417877

theorem childrens_cookbook_cost :
  ∀ (d db s a cb : ℕ), d = 5 → db = 11 → s = 19 → a = 2 →
  s + a = 21 ∧ d + db = 16 → cb = 21 - 16 := 
by
  intros d db s a cb hd hdb hs ha htotal,
  rw [hd, hdb, hs, ha] at htotal,
  exact htotal.2.trans (htotal.1.symm ▸ rfl)

end childrens_cookbook_cost_l417_417877


namespace total_expenditure_of_8_people_l417_417807

theorem total_expenditure_of_8_people :
  ∀ (e : Fin 8 → ℝ), 
    (∀ i, i < 7 → e i = 10) → 
    e 7 = (1 / 8 * (∑ i, e i)) + 7 → 
    ∑ i, e i = 88 :=
by
  intro e h7 he8
  sorry

end total_expenditure_of_8_people_l417_417807


namespace circle_properties_l417_417690

noncomputable def pi : Real := 3.14
variable (C : Real) (diameter : Real) (radius : Real) (area : Real)

theorem circle_properties (h₀ : C = 31.4) :
  radius = C / (2 * pi) ∧
  diameter = 2 * radius ∧
  area = pi * radius^2 ∧
  radius = 5 ∧
  diameter = 10 ∧
  area = 78.5 :=
by
  sorry

end circle_properties_l417_417690


namespace harmonic_sum_lt_80_l417_417660

def digit9 (k : ℕ) : Prop := 
  ∃ d ∈ (Nat.digits 10 k), d = 9

def harmonic_sum_without_digit9 (n : ℕ) : ℝ := 
  ∑ k in Finset.range (n + 1), if digit9 k then 0 else 1 / k

theorem harmonic_sum_lt_80 (n : ℕ) : harmonic_sum_without_digit9 n < 80 := 
  sorry

end harmonic_sum_lt_80_l417_417660


namespace problem1_problem2_l417_417623

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^x - a^(-2)) / (2^x + 1)

instance : Inhabited ℝ := ⟨0⟩

theorem problem1 (a : ℝ) (h_odd : ∀ x : ℝ, f(a, x) = -f(a, -x)) :
  a = 1 → 
  let f' := λ x, (2^x - 1) / (2^x + 1) in
  let F := λ x, f'(x) + 2^x - 4 / (2^x + 1) - 1 in
  F 1 = 0 :=
begin
  intro ha,
  subst ha,
  dsimp [f'],
  dsimp [F],
  sorry
end

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := 2 * Real.log 2 ((1 + x) / k)

theorem problem2 (k : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1 / 2) (2 / 3) → 
    let f_inv := λ x, Real.log 2 ((1 + x) / (1 - x)) in
    let g_val := g k x in
    f_inv x ≤ g_val) → 
  0 < k ∧ k ≤ Real.sqrt (5 / 9) :=
begin
  intros h,
  sorry
end

end problem1_problem2_l417_417623


namespace ratio_of_areas_l417_417980

noncomputable def regular_dodecagon (vertices : Fin 12 → ℝ × ℝ) : Prop :=
  ∃ (p : ℝ × ℝ) (r : ℝ), r > 0 ∧
    ∀ i : Fin 12, vertices i = (p.1 + r * cos (2 * i * π / 12), p.2 + r * sin (2 * i * π / 12))

def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
  ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

def area_ratio (A B C D E F G H I J K L M N : ℝ × ℝ) : ℚ :=
  2 / 3

theorem ratio_of_areas {vertices : Fin 12 → ℝ × ℝ}
    (h : regular_dodecagon vertices)
    (M : ℝ × ℝ)
    (N : ℝ × ℝ) :
  midpoint (vertices 2) (vertices 3) = M →
  midpoint (vertices 6) (vertices 7) = N →
  area_ratio (vertices 0) (vertices 1) (vertices 2) M (vertices 4) (vertices 5)
    (vertices 6) (vertices 7) (vertices 8) (vertices 9) (vertices 10) (vertices 11) M N = (2 / 3 : ℚ) :=
sorry

end ratio_of_areas_l417_417980


namespace total_job_applications_l417_417115

theorem total_job_applications (apps_in_state : ℕ) (apps_other_states : ℕ) 
  (h1 : apps_in_state = 200)
  (h2 : apps_other_states = 2 * apps_in_state) :
  apps_in_state + apps_other_states = 600 :=
by
  sorry

end total_job_applications_l417_417115


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417026

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417026


namespace part1_part2_l417_417192

noncomputable def f (a x : ℝ) := Real.log x (logBase a)
noncomputable def g (x : ℝ) := - (4 / x)
noncomputable def F (a x m : ℝ) := f a (x^2 - m * x - m)

theorem part1 (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) 
  (H₁ : ∀ x ∈ Icc (1:ℝ) 2, f a x ≤ 2) (H₂ : ∃ x ∈ Icc (1:ℝ) 2, f a x = 2) 
  (H₃ : ∀ x ∈ Icc (1:ℝ) 2, g x ≤ -2) (H₄ : ∃ x ∈ Icc (1:ℝ) 2, g x = -2) : 
  a = Real.sqrt 2 := 
sorry

theorem part2 (a : ℝ) (m : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1)
  (H₁ : ∀ x ∈ Icc (1:ℝ) 2, f a x ≤ 2) (H₂ : ∃ x ∈ Icc (1:ℝ) 2, f a x = 2) 
  (H₃ : ∀ x ∈ Icc (1:ℝ) 2, g x ≤ -2) (H₄ : ∃ x ∈ Icc (1:ℝ) 2, g x = -2) 
  (H₅ : ∀ x ∈ Iio (1 - Real.sqrt 3), F a x m ≤ F a 0 m) : 
  2 - 2 * Real.sqrt 3 ≤ m ∧ m ≤ 2 := 
sorry

end part1_part2_l417_417192


namespace find_f_value_l417_417907

theorem find_f_value (f : ℕ → ℕ) (h : ∀ x, f(2 * x + 1) = x^2) : f(5) = 4 :=
by
  sorry

end find_f_value_l417_417907


namespace cosine_angle_zero_l417_417487

structure Point (α : Type) :=
  (x : α)
  (y : α)
  (z : α)

def vector (p1 p2 : Point ℝ) : Point ℝ :=
  Point.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)

def dot_product (v1 v2 : Point ℝ) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point ℝ) : ℝ :=
  real.sqrt (v.x^2 + v.y^2 + v.z^2)

theorem cosine_angle_zero 
  (A B C : Point ℝ)
  (hA : A = Point.mk (-2) 1 1)
  (hB : B = Point.mk 2 3 (-2))
  (hC : C = Point.mk 0 0 3) : 
    let AB := vector A B,
        AC := vector A C in 
        dot_product AB AC = 0 :=
sorry

end cosine_angle_zero_l417_417487


namespace martha_cards_l417_417645

theorem martha_cards (start_cards : ℕ) : start_cards + 76 = 79 → start_cards = 3 :=
by
  sorry

end martha_cards_l417_417645


namespace num_solutions_f2002_eq_half_x_l417_417512

def f1 (x : ℝ) : ℝ := 1 - 2 * x

def f (fn : ℝ → ℝ) (x : ℝ) : ℝ := fn (fn x)

def f_seq (n : ℕ) : (ℝ → ℝ)
| 0 => λ x, x
| 1 => f1
| (n+2) => f (f_seq (n+1))

theorem num_solutions_f2002_eq_half_x :
  let fn := f_seq 2002
  ∃ (n : ℕ), n = 2^2002 ∧ ∀ x, fn x = (1/2) * x → true :=
by {
  let fn := f_seq 2002,
  scope
    intro x,
    sorry
}

end num_solutions_f2002_eq_half_x_l417_417512


namespace network_connections_l417_417738

theorem network_connections (n : ℕ) (k : ℕ) :
  n = 30 → k = 4 → (n * k) / 2 = 60 := by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end network_connections_l417_417738


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417009

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417009


namespace difference_of_squares_l417_417632

theorem difference_of_squares (S : Set ℤ) (h1 : S = { n | ∃ k : ℕ, 0 ≤ k ∧ k < 250 ∧ n = 264 + 12 * k }) :
  ∃ K : ℤ, K = 10504848 ∧
    let a := 264,
    let b := 264 + 12 * 249
    in K = b^2 - a^2 :=
by {
  sorry
}

end difference_of_squares_l417_417632


namespace sum_of_digits_largest_number_l417_417788

theorem sum_of_digits_largest_number : 
  ∃ (a b c : ℕ), 
    ({a, b, c} ⊆ {4, 5, 6, 7}) ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (765 = a * 100 + b * 10 + c ∨ 765 = a * 100 + c * 10 + b ∨ 765 = b * 100 + a * 10 + c ∨ 
     765 = b * 100 + c * 10 + a ∨ 765 = c * 100 + a * 10 + b ∨ 765 = c * 100 + b * 10 + a) →
    (a + b + c = 18) :=
by
  sorry

end sum_of_digits_largest_number_l417_417788


namespace total_dots_not_visible_proof_l417_417145

def total_dots_on_one_die : ℕ := 21

def total_dots_on_five_dice : ℕ := 5 * total_dots_on_one_die

def visible_numbers : List ℕ := [1, 2, 3, 1, 4, 5, 6, 2]

def sum_visible_numbers : ℕ := visible_numbers.sum

def total_dots_not_visible (total : ℕ) (visible_sum : ℕ) : ℕ :=
  total - visible_sum

theorem total_dots_not_visible_proof :
  total_dots_not_visible total_dots_on_five_dice sum_visible_numbers = 81 :=
by
  sorry

end total_dots_not_visible_proof_l417_417145


namespace part1_part2_l417_417198

variables {x a : ℝ}

def set_A (a : ℝ) : set ℝ := { x | (x - 6) * (x - 2 * a - 5) > 0 }
def set_B (a : ℝ) : set ℝ := { x | (a^2 + 2 - x) * (2 * a - x) < 0 }

theorem part1 : 
  (A : set ℝ) (B : set ℝ) (A_inter_B : set ℝ): 
  (A = { x | (x - 6) * (x - 15) > 0 }) 
  ∧ (B = { x | (27 - x) * (10 - x) < 0 }) 
  ∧ (A_inter_B = { x | 15 < x ∧ x < 27 }) 
  ∧ (A ∩ B = A_inter_B) := 
sorry

theorem part2 : 
  (A : set ℝ) (B : set ℝ) : 
  (a > 1/2) 
  ∧ (A = { x | (x - 6) * (x - (2 * a + 5)) > 0 }) 
  ∧ (B = { x | (a^2 + 2 - x) * (2 * a - x) < 0 }) 
  ∧ (B ⊆ A) 
  ∧ (1/2 < a ∧ a <= 2) := 
sorry

end part1_part2_l417_417198


namespace find_range_of_x_l417_417511

noncomputable def f : ℝ → ℝ := sorry -- We assume the existence of f

theorem find_range_of_x : 
  (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f(x₁) ≤ f(x₂)) ∧ -- f(x) is monotonically increasing
  (∀ y : ℝ, f(f⁻¹(y)) = y ∧ f⁻¹(f(y)) = y) ∧ -- f and f⁻¹ are inverse functions
  (f (-4 + 1) = 0 ∧ f (2 + 1) = 3) ∧ -- Graph of y = f(x+1) passes through (-4, 0) and (2, 3)
  (∀ x : ℝ, |f⁻¹(x + 1)| ≤ 3) -- |f⁻¹(x+1)| ≤ 3
  → (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2) := -- Prove that the range of x is -1 ≤ x ≤ 2
sorry

end find_range_of_x_l417_417511


namespace greatest_consecutive_integers_sum_36_l417_417776

-- Definition of the sum of N consecutive integers starting from a
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Problem statement in Lean 4
theorem greatest_consecutive_integers_sum_36 (N : ℤ) (h : sum_consecutive_integers (-35) 72 = 36) : N = 72 := by
  sorry

end greatest_consecutive_integers_sum_36_l417_417776


namespace boots_selection_ways_l417_417901

theorem boots_selection_ways:
  let num_pairs := 5 in
  let total_boots := 2 * num_pairs in
  let selected_boots := 4 in
  let ways_to_choose_pair := Nat.choose num_pairs 1 in
  let ways_to_choose_two_individual_boots := Nat.choose (num_pairs - 1) 2 in
  let ways_to_choose_from_each_pair := 2 ^ 2 in
  ways_to_choose_pair * ways_to_choose_two_individual_boots * ways_to_choose_from_each_pair = 120 := 
by
  sorry

end boots_selection_ways_l417_417901


namespace sample_size_is_correct_l417_417425

-- Define the school and selection conditions
def total_classes := 40
def students_per_class := 50

-- Given condition
def selected_students := 150

-- Theorem statement
theorem sample_size_is_correct : selected_students = 150 := 
by 
  sorry

end sample_size_is_correct_l417_417425


namespace Nero_speed_is_8_l417_417607

-- Defining the conditions
def Jerome_time := 6 -- in hours
def Nero_time := 3 -- in hours
def Jerome_speed := 4 -- in miles per hour

-- Calculation step
def Distance := Jerome_speed * Jerome_time

-- The theorem we need to prove (Nero's speed)
theorem Nero_speed_is_8 :
  (Distance / Nero_time) = 8 := by
  sorry

end Nero_speed_is_8_l417_417607


namespace rest_days_in_first_800_days_l417_417441

theorem rest_days_in_first_800_days :
  let alice_cycle := [false, false, false, false, true, true] in
  let bob_cycle := [false, false, false, false, false, true] in
  let common_rest_days (n : ℕ) : ℕ := 
    list.countp (λ i, alice_cycle[i % 6] ∧ bob_cycle[i % 6]) (list.fin_range n) in
  common_rest_days 800 = 133 :=
by
  sorry

end rest_days_in_first_800_days_l417_417441


namespace sum_arithmetic_sequence_l417_417522

theorem sum_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith_seq : ∀ n, S (n + 1) - S n = a n)
  (h_S2 : S 2 = 4) 
  (h_S4 : S 4 = 16) 
: a 5 + a 6 = 20 :=
sorry

end sum_arithmetic_sequence_l417_417522


namespace increase_per_size_l417_417828

noncomputable def size_difference := 17 - 8
noncomputable def length_15 := 5.9
noncomputable def increase_in_length (x : ℝ) := x

theorem increase_per_size (x : ℝ) :
  ((length_15 + 2 * increase_in_length x) = (5.9 - 7 * increase_in_length x) * 1.4) →
  (increase_in_length x = 0.2) :=
by
  intro h
  sorry

end increase_per_size_l417_417828


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417019

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417019


namespace h_at_0_l417_417638

noncomputable def h (x : ℝ) : ℝ := sorry -- the actual polynomial
-- Conditions for h(x)
axiom h_cond1 : h (-2) = -4
axiom h_cond2 : h (1) = -1
axiom h_cond3 : h (-3) = -9
axiom h_cond4 : h (3) = -9
axiom h_cond5 : h (5) = -25

-- Statement of the proof problem
theorem h_at_0 : h (0) = -90 := sorry

end h_at_0_l417_417638


namespace prob_at_least_one_2_in_two_8_sided_dice_l417_417361

/-- Probability of getting at least one 2 when rolling two 8-sided dice -/
theorem prob_at_least_one_2_in_two_8_sided_dice : 
  let total_outcomes := 64
  let favorable_outcomes := 15
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 15 / 64 :=
by
  sorry

end prob_at_least_one_2_in_two_8_sided_dice_l417_417361


namespace volunteer_group_selection_l417_417343

theorem volunteer_group_selection :
  let M := 4  -- Number of male teachers
  let F := 5  -- Number of female teachers
  let G := 3  -- Total number of teachers in the group
  -- Calculate the number of ways to select 2 male teachers and 1 female teacher
  let ways1 := (Nat.choose M 2) * (Nat.choose F 1)
  -- Calculate the number of ways to select 1 male teacher and 2 female teachers
  let ways2 := (Nat.choose M 1) * (Nat.choose F 2)
  -- The total number of ways to form the group
  ways1 + ways2 = 70 := by sorry

end volunteer_group_selection_l417_417343


namespace inequality_for_f_and_g_l417_417270

noncomputable def f (x : ℝ) : ℝ := ∑ i in finset.range(1009), (1 / (x - 2 * i))
noncomputable def g (x : ℝ) : ℝ := ∑ i in finset.range(1009), (1 / (x - (2 * i + 1)))

theorem inequality_for_f_and_g (x : ℝ) (h0 : 0 < x) (h1 : x < 2018) (h2 : ¬ is_integer x) :
  |f x - g x| > 2 :=
by sorry

end inequality_for_f_and_g_l417_417270


namespace real_value_of_a_l417_417219

theorem real_value_of_a (a : ℝ) (i : ℂ) (H1 : i = complex.I) (H2 : (2 + a * i)^2 ∈ ℝ) : a = 0 :=
sorry

end real_value_of_a_l417_417219


namespace problem_1_problem_2_l417_417993

noncomputable def seq (n : ℕ) : ℝ :=
  match n with
  | 1 => 4
  | n + 2 => Real.sqrt (2 * seq (n + 1) + 3)

theorem problem_1 (n : ℕ) (h : n ≥ 2) :
  abs (seq n - 3) ≤ (2 / 3) * abs (seq (n - 1) - 3) :=
sorry

theorem problem_2 (n : ℕ) :
  3 - (2 / 3) ^ (n - 1) ≤ seq n ∧ seq n ≤ 3 + (2 / 3) ^ (n - 1) :=
sorry

end problem_1_problem_2_l417_417993


namespace tom_and_jerry_drank_80_ounces_l417_417753

theorem tom_and_jerry_drank_80_ounces
    (T J : ℝ) 
    (initial_T : T = 40)
    (initial_J : J = 2 * T)
    (T_drank J_drank : ℝ)
    (T_remaining J_remaining : ℝ)
    (T_after_pour J_after_pour : ℝ)
    (T_final J_final : ℝ)
    (H1 : T_drank = (2 / 3) * T)
    (H2 : J_drank = (2 / 3) * J)
    (H3 : T_remaining = T - T_drank)
    (H4 : J_remaining = J - J_drank)
    (H5 : T_after_pour = T_remaining + (1 / 4) * J_remaining)
    (H6 : J_after_pour = J_remaining - (1 / 4) * J_remaining)
    (H7 : T_final = T_after_pour - 5)
    (H8 : J_final = J_after_pour + 5)
    (H9 : T_final = J_final + 4)
    : T_drank + J_drank = 80 :=
by
  sorry

end tom_and_jerry_drank_80_ounces_l417_417753


namespace greatest_divisor_of_product_of_consecutive_integers_l417_417032

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l417_417032


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417018

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417018


namespace class_average_is_86_l417_417816

def class_average (total_students : ℕ) (boys_percentage : ℝ) (boys_score : ℝ) (girls_score : ℝ) : ℝ :=
  let boys := total_students * boys_percentage
  let girls := total_students * (1 - boys_percentage)
  let total_boys_score := boys * boys_score
  let total_girls_score := girls * girls_score
  (total_boys_score + total_girls_score) / total_students

theorem class_average_is_86 :
  class_average 100 0.4 0.8 0.9 = 0.86 :=
by
  simp [class_average]
  -- Prove the equality 40 * 0.8 + 60 * 0.9 / 100 = 86%
  sorry

end class_average_is_86_l417_417816


namespace find_C1_equation_find_min_distance_l417_417589

-- Define the parameterized equations for lines l1 and l2
def param_line_l1 (t : ℝ) (k : ℝ) : (ℝ × ℝ) :=
  (t - (real.sqrt 3), k * t)

def param_line_l2 (m : ℝ) (k : ℝ) : (ℝ × ℝ) :=
  ((real.sqrt 3) - m, m / (3 * k))

-- Define the Cartesian equation of curve C1
def cartesian_eq_C1 (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

-- Define the Cartesian equation of line C2
def cartesian_eq_C2 (x y : ℝ) : Prop :=
  x + y - 8 = 0

-- Function for the intersection point P of l1 and l2 parameterized by k
def intersection_point (t m k : ℝ) : (ℝ × ℝ) :=
  let (x1, y1) := param_line_l1 t k in
  let (x2, y2) := param_line_l2 m k in
  if x1 = x2 ∧ y1 = y2 then (x1, y1) else (0,0)  -- Assuming unique intersection

-- Lean statement for part 1
theorem find_C1_equation (x y t m k : ℝ) (h1 : (x, y) = param_line_l1 t k)
  (h2 : (x, y) = param_line_l2 m k) : cartesian_eq_C1 x y :=
by
  sorry

-- Function to find the minimum distance between point Q on C1 and line C2
def min_distance (α : ℝ) : ℝ :=
  let x := (real.sqrt 3) * real.cos α in
  let y := real.sin α in
  abs (x + y - 8) / (real.sqrt 2)

-- Lean statement for part 2
theorem find_min_distance (α : ℝ) (d : ℝ) (h : d = min_distance α) : d = 3 * real.sqrt 2 :=
by
  sorry

end find_C1_equation_find_min_distance_l417_417589


namespace find_f_3_l417_417908

open Function

-- Given condition as a definition
def f (x : ℝ) : ℝ := sorry

-- Main theorem statement
theorem find_f_3 : (∀ x : ℝ, f (2 * x + 1) = x^2 - 2 * x) → (f 3 = -1) :=
by
  intro h,
  -- proof is to be done
  sorry

end find_f_3_l417_417908


namespace handshake_count_l417_417231

theorem handshake_count (n : ℕ) (m : ℕ) (couples : ℕ) (people : ℕ) 
  (h1 : couples = 15) 
  (h2 : people = 2 * couples)
  (h3 : people = 30)
  (h4 : n = couples) 
  (h5 : m = people / 2)
  (h6 : ∀ i : ℕ, i < m → ∀ j : ℕ, j < m → i ≠ j → i * j + i ≠ n 
    ∧ j * i + j ≠ n) 
  : n * (n - 1) / 2 + (2 * n - 2) * n = 315 :=
by
  sorry

end handshake_count_l417_417231


namespace find_B_l417_417110

-- Define the function and constants
def y (A B C D E F G H x : ℝ) : ℝ :=
  A * Real.sin (B * x + C) + D + E * Real.cos (F * x + G) + H

-- Given conditions
axiom A : ℝ := 3
axiom E : ℝ := 2
axiom F : ℝ
axiom B : ℝ
axiom C : ℝ
axiom D : ℝ
axiom G : ℝ
axiom H : ℝ

-- Additional conditions
axiom A_neq_E : A ≠ E
axiom F_eq_B : F = B
axiom six_periods : True -- Placeholder for the condition "graph covers six periods from 0 to 2π"

-- Proof statement
theorem find_B : B = 6 := by
  sorry

end find_B_l417_417110


namespace racecourse_length_l417_417792

variable (v : ℝ) -- Speed of B
variable (d : ℝ) -- Length of the racecourse

-- Conditions
def faster_A : Prop := 2 * v
def start_advantage : ℝ := 42

-- Proof Problem
theorem racecourse_length
  (h_faster : 2 * v) -- A runs twice as fast as B
  (h_advantage : d + start_advantage = 2 * d) -- A gives B a start of 42 meters and both reach the finish line at the same time
  : d = 42 :=
sorry

end racecourse_length_l417_417792


namespace odd_prime_divides_discriminant_l417_417272

theorem odd_prime_divides_discriminant (a b c p : ℤ) (hp : Nat.Prime p ∧ p % 2 = 1)
  (hpsquares : ∀ x : ℤ, (∃ k : ℤ, f (x + k) = a * (x + k) ^ 2 + b * (x + k) + c ∧ is_square (f (x + k))) ∧ (2 * p - 1 > k)) :
  p ∣ (b ^ 2 - 4 * a * c) := sorry

end odd_prime_divides_discriminant_l417_417272


namespace total_walnut_trees_in_park_l417_417731

-- Define initial number of walnut trees in the park
def initial_walnut_trees : ℕ := 22

-- Define number of walnut trees planted by workers
def planted_walnut_trees : ℕ := 33

-- Prove the total number of walnut trees in the park
theorem total_walnut_trees_in_park : initial_walnut_trees + planted_walnut_trees = 55 := by
  sorry

end total_walnut_trees_in_park_l417_417731


namespace middle_car_is_Tom_l417_417982

def positions : Type := ℕ
def person : Type := ℕ

variables (Kim Tom Bob Dom Sam : person)
variables (car1 car2 car3 car4 car5 : positions)

-- Kim sat in the first car
axiom h1 : (Kim = car1)

-- Tom sat directly behind Bob
axiom h2 : (Tom = Bob + 1)

-- Dom sat in one of the cars behind Tom
axiom h3 : (Dom > Tom)

-- At least one person sat between Sam and Dom
axiom h4 : ∀ p, (p < Dom → p ≠ Sam)

-- The person who sat in the middle car
theorem middle_car_is_Tom : Tom = car3 := 
sorry

end middle_car_is_Tom_l417_417982


namespace evaluate_expression_l417_417673

theorem evaluate_expression (a : ℝ) (h : a = 2) : 
    (a / (a^2 - 1) - 1 / (a^2 - 1)) = 1 / 3 := by
  sorry

end evaluate_expression_l417_417673


namespace find_set_of_integers_l417_417643

theorem find_set_of_integers : 
  {x : ℤ | ¬(x^2 - x ≥ 6) ∧ x ∈ ℤ} = {x : ℤ | -2 < x ∧ x < 3} := 
by
  sorry

end find_set_of_integers_l417_417643


namespace minimum_value_of_f_l417_417216

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) * (x^2 - 4 * x + 3)

theorem minimum_value_of_f : ∃ m : ℝ, m = -16 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use -16
  sorry

end minimum_value_of_f_l417_417216


namespace shaded_area_of_five_circles_l417_417242

theorem shaded_area_of_five_circles (
  radius : ℝ
  (radius_eq : radius = 5)
) : 
  let quarter_circle_area := (1/4) * Real.pi * radius^2
  let triangle_area := (1/2) * radius^2
  let one_segment_area := quarter_circle_area - triangle_area
  let total_shaded_area := 10 * one_segment_area
  total_shaded_area = 62.5 * Real.pi - 125
:= 
by
  sorry

end shaded_area_of_five_circles_l417_417242


namespace elements_starting_with_1_l417_417864

noncomputable def T : Set ℕ := {x | ∃ k : ℕ, k ≤ 1000 ∧ x = 3^k}

theorem elements_starting_with_1 : 
  (∃ digits : ℕ, digits = 477 ∧ 
  ∃ n : ℕ, n = 1000 ∧ 
  ∀ x ∈ T, (3^n).digits = digits) → 
  ∀ t ∈ T, t < 3^1000 → count (λ n : ℕ, n < 10 * 10^digits ∧ (n : ℕ).digits = 477) {n ∈ T | (n.to_digits! 10).head = 1} = 524 := 
by
  sorry

end elements_starting_with_1_l417_417864


namespace value_of_algebraic_expression_l417_417971

noncomputable def quadratic_expression (m : ℝ) : ℝ :=
  3 * m * (2 * m - 3) - 1

theorem value_of_algebraic_expression (m : ℝ) (h : 2 * m^2 - 3 * m - 1 = 0) : quadratic_expression m = 2 :=
by {
  sorry
}

end value_of_algebraic_expression_l417_417971


namespace range_of_m_l417_417577

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → x^(m-1) > y^(m-1)) → m < 1 :=
by
  sorry

end range_of_m_l417_417577


namespace contribution_amount_l417_417426

-- Definitions based on conditions
variable (x : ℝ)

-- Total amount needed
def total_needed := 200

-- Contributions from different families
def contribution_two_families := 2 * x
def contribution_eight_families := 8 * 10 -- 80
def contribution_ten_families := 10 * 5 -- 50
def total_contribution := contribution_two_families + contribution_eight_families + contribution_ten_families

-- Amount raised so far given they need 30 more to reach the target
def raised_so_far := total_needed - 30 -- 170

-- Statement to prove
theorem contribution_amount :
  total_contribution x = raised_so_far →
  x = 20 := by 
  sorry

end contribution_amount_l417_417426


namespace decreased_price_correct_l417_417436

def actual_cost : ℝ := 250
def percentage_decrease : ℝ := 0.2

theorem decreased_price_correct : actual_cost - (percentage_decrease * actual_cost) = 200 :=
by
  sorry

end decreased_price_correct_l417_417436


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417003

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417003


namespace false_statement_l417_417700

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x

def p : Prop := ∃ x0 : ℝ, f x0 = -1
def q : Prop := ∀ x : ℝ, f (2 * Real.pi + x) = f x

theorem false_statement : ¬ (p ∧ q) := sorry

end false_statement_l417_417700


namespace b_2006_equals_4_l417_417266

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 1 then 5 else
  if n = 2 then 4 else
  sequence (n - 1) / sequence (n - 2)

theorem b_2006_equals_4 :
  sequence 2006 = 4 :=
sorry

end b_2006_equals_4_l417_417266


namespace max_area_triangle_PAB_l417_417527

noncomputable def point := (ℝ × ℝ)

def on_circle (k : ℝ) (p : point) : Prop :=
  p.1 ^ 2 + p.2 ^ 2 + k * p.1 = 0

def symmetric_with_respect_to (p1 p2 : point) (l : ℝ) : Prop :=
  p1.1 = 2 * l - p2.1 ∧ p1.2 = p2.2

def line (x y c : ℝ) : point → Prop :=
  λ p, p.1 * x + p.2 * y + c = 0

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def area_triangle (A B P : point) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2) + P.1 * (A.2 - B.2))

theorem max_area_triangle_PAB :
  ∀ (A B P M N : point) (k : ℝ),
    A = (-2, 0) → B = (0, 2) →
    on_circle k M → on_circle k N →
    symmetric_with_respect_to M N 1 →
    on_circle k P →
    ∃ area : ℝ, area_triangle A B P = area ∧ area = 3 + real.sqrt 2 :=
by
  sorry

end max_area_triangle_PAB_l417_417527


namespace arun_working_days_l417_417844

theorem arun_working_days (A T : ℝ) 
  (h1 : A + T = 1/10) 
  (h2 : A = 1/18) : 
  (1 / A) = 18 :=
by
  -- Proof will be skipped
  sorry

end arun_working_days_l417_417844


namespace inequality_problems_l417_417153

theorem inequality_problems
  (m n l : ℝ)
  (h1 : m > n)
  (h2 : n > l) :
  (m + 1/m > n + 1/n) ∧ (m + 1/n > n + 1/m) :=
by
  sorry

end inequality_problems_l417_417153


namespace actual_distance_between_city_centers_l417_417311

-- Define the conditions
def map_distance_cm : ℝ := 45
def scale_cm_to_km : ℝ := 10

-- Define the proof statement
theorem actual_distance_between_city_centers
  (md : ℝ := map_distance_cm)
  (scale : ℝ := scale_cm_to_km) :
  md * scale = 450 :=
by
  sorry

end actual_distance_between_city_centers_l417_417311


namespace students_disliked_menu_l417_417837

theorem students_disliked_menu (total_students liked_students : ℕ) (h1 : total_students = 400) (h2 : liked_students = 235) : total_students - liked_students = 165 :=
by 
  sorry

end students_disliked_menu_l417_417837


namespace painter_total_amount_l417_417656

noncomputable def total_amount_collected : ℝ :=
let east_addresses := list.range 25 |>.map (λ n, 5 + n * 6) in
let west_addresses := list.range 25 |>.map (λ n, 2 + n * 6) in
let digit_cost (n : ℕ) := 1.5 * (real.log10 (n + 1.to_real) + 1).floor.to_real in
east_addresses.sum digit_cost + west_addresses.sum digit_cost

theorem painter_total_amount : total_amount_collected = 171 :=
sorry

end painter_total_amount_l417_417656


namespace vector_equality_l417_417170

open_locale classical

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (A B C D O : V)

def midpoint (D B C : V) : Prop := 2 • D = B + C

theorem vector_equality 
  (mid_D: midpoint D B C) 
  (h : 2 • O + B + C = 0) : 
  O - A = D := 
begin 
  sorry 
end

end vector_equality_l417_417170


namespace correct_statement_about_K2_l417_417376

theorem correct_statement_about_K2
  (hA : ¬(∀ independent_problems, K2 is suitable for testing relatedness))
  (hC : ∀ X Y : categorical_variables, (small (K2 X Y) → low correlation X Y) ∧ ¬(small (K2 X Y) → uncorrelated X Y))
  (hD : ¬(∀ n a b c d : ℕ, K2_calculation = (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))))
  (statement_B : ∀ X Y : events, larger (K2 X Y) → greater_correlation X Y) :
  true
by 
  sorry

end correct_statement_about_K2_l417_417376


namespace find_B_l417_417099

-- Let f(z) be the polynomial z^4 - 6z^3 + Az^2 + Bz + 9 with all positive integer roots.
def polynomial (A B z : ℤ) := z^4 - 6 * z^3 + A * z^2 + B * z + 9

-- Theorem to be proven: Given the conditions, B equals -13.
theorem find_B (A B : ℤ) (h : ∀ z, z ∈ {1, 2} → polynomial A B z = 0) : B = -13 :=
by
  sorry

end find_B_l417_417099


namespace triangle_side_lengths_triangle_circumradius_l417_417532

theorem triangle_side_lengths (a b c : ℝ) (h : a^2 + b^2 + c^2 - 6 * a - 8 * b - 10 * c + 50 = 0) :
  a = 3 ∧ b = 4 ∧ c = 5 :=
by sorry

theorem triangle_circumradius (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let R := c / 2 in R = 2.5 :=
by sorry

end triangle_side_lengths_triangle_circumradius_l417_417532


namespace sphere_and_cylinder_volumes_l417_417328

noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * π * r^2
noncomputable def sphere_volume (r : ℝ) : ℝ := (4/3) * π * r^3
noncomputable def cylinder_volume (r : ℝ) (h : ℝ) : ℝ := π * r^2 * h

theorem sphere_and_cylinder_volumes (r : ℝ) (h : ℝ)
  (h1 : sphere_surface_area r = 144 * π)
  (h2 : h = 2 * r) :
  sphere_volume r = 288 * π ∧ cylinder_volume r h = 432 * π :=
by
  sorry

end sphere_and_cylinder_volumes_l417_417328


namespace translate_point_l417_417094

noncomputable def translate (z w : ℂ) : ℂ := z + w

theorem translate_point (z1 z2 z3 : ℂ) (w : ℂ) :
  translate (1 - 3 * complex.i) w = 5 + 2 * complex.i →
  translate (3 - 4 * complex.i) w = 7 + complex.i :=
sorry

end translate_point_l417_417094


namespace sum_of_transformed_roots_l417_417213

theorem sum_of_transformed_roots (α β γ : ℂ) (h₁ : α^3 - α + 1 = 0) (h₂ : β^3 - β + 1 = 0) (h₃ : γ^3 - γ + 1 = 0) :
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
by
  sorry

end sum_of_transformed_roots_l417_417213


namespace points_concyclic_l417_417845

theorem points_concyclic
  {P A B C : Type} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (PA PB PC : ℝ) 
  (α β : ℝ) 
  (h1 : angle P A B = α)
  (h2 : angle C P B = β)
  (h3 : α + β < 180)
  (h4 : PB * sin (α + β) = PC * sin α + PA * sin β) :
  (concyclic P A B C) :=
by
  sorry

end points_concyclic_l417_417845


namespace regular_pentagon_diagonal_ratio_l417_417661

theorem regular_pentagon_diagonal_ratio (ABCDE : Type) [regular_pentagon ABCDE] 
  (s : ℝ) :
  ∃ d : ℝ, d / s = (Real.sqrt 5 + 1) / 2 :=
sorry

end regular_pentagon_diagonal_ratio_l417_417661


namespace total_pieces_in_boxes_l417_417399

theorem total_pieces_in_boxes (num_boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ)
    (h1 : num_boxes = 6) (h2 : pieces_per_box = 500) :
    num_boxes * pieces_per_box = total_pieces → total_pieces = 3000 :=
by
  intro h
  rw [h1, h2] at h
  rw h
  rfl

end total_pieces_in_boxes_l417_417399


namespace birds_initially_l417_417337

-- Definitions of the conditions
def initial_birds (B : Nat) := B
def initial_storks := 4
def additional_storks := 6
def total := 13

-- The theorem we need to prove
theorem birds_initially (B : Nat) (h : initial_birds B + initial_storks + additional_storks = total) : initial_birds B = 3 :=
by
  -- The proof can go here
  sorry

end birds_initially_l417_417337


namespace probability_at_least_one_2_on_8_sided_dice_l417_417347

theorem probability_at_least_one_2_on_8_sided_dice :
  (∃ (d1 d2 : Fin 8), d1 = 1 ∨ d2 = 1) → (15 / 64) = (15 / 64) := by
  intro h
  sorry

end probability_at_least_one_2_on_8_sided_dice_l417_417347


namespace probability_twice_as_many_sons_l417_417650
open ProbabilityTheory

/-- Define a probability space where each child's gender is equally likely (0.5) to be male or female. -/
noncomputable def child_gender : MeasureTheory.ProbabilityMeasure (Fin 2 → Bool) := 
  MeasureTheory.ProbabilityMeasure.pi (Fin 2 → Bool) (λ _, MeasureTheory.Measure.dirac ff ⊔ MeasureTheory.Measure.dirac tt)

/-- Define the event that Mr. Lee has at least twice as many sons as daughters given 8 children. -/
def at_least_twice_sons_as_daughters (children : Fin 8 → Bool) : Prop :=
  let number_of_sons := Finset.card (Finset.filter (λ i, children i = tt) Finset.univ)
  let number_of_daughters := 8 - number_of_sons
  2 * number_of_daughters ≤ number_of_sons

/-- The probability that Mr. Lee has at least twice as many sons as daughters given the conditions. -/
theorem probability_twice_as_many_sons :
  MeasureTheory.Measure.to_prob_measure child_gender (set_of at_least_twice_sons_as_daughters) = 37 / 256 :=
sorry

end probability_twice_as_many_sons_l417_417650


namespace sum_even_sum_odd_l417_417940

noncomputable def f (x : ℕ) : ℝ := 4^x

def b (n : ℕ) : ℝ :=
  if Even n then 1 / n else n + 2

-- Definition and condition for the sequence \(a_n\)
def a (n : ℕ) : ℝ := (n + 2) / 2

-- Prove \(S_n\) for even \(n\)
theorem sum_even (n : ℕ) (h : Even n) : 
  let S (n : ℕ) := ∑ i in Finset.range n, b (i+1) / a (i+1)
  S n = n + 0.5 - 1 / (n + 2) := sorry

-- Prove \(S_n\) for odd \(n\)
theorem sum_odd (n : ℕ) (h : ¬ Even n) : 
  let S (n : ℕ) := ∑ i in Finset.range n, b (i+1) / a (i+1)
  S n = n + 1.5 - 1 / (n + 1) := sorry

end sum_even_sum_odd_l417_417940


namespace sum_of_possible_values_of_x_l417_417563

theorem sum_of_possible_values_of_x :
  ∀ x : ℝ, (x + 2) * (x - 3) = 20 → ∃ s, s = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l417_417563


namespace alice_favorite_number_l417_417839

theorem alice_favorite_number :
  ∃ (n : ℕ), 50 < n ∧ n < 100 ∧ n % 11 = 0 ∧ n % 2 ≠ 0 ∧ (n / 10 + n % 10) % 5 = 0 ∧ n = 55 :=
by
  sorry

end alice_favorite_number_l417_417839


namespace perimeter_of_region_proof_l417_417304

noncomputable def perimeter_of_region (total_area : ℕ) (num_squares : ℕ) (arrangement : String) : ℕ :=
  if total_area = 512 ∧ num_squares = 8 ∧ arrangement = "vertical rectangle" then 160 else 0

theorem perimeter_of_region_proof :
  perimeter_of_region 512 8 "vertical rectangle" = 160 :=
by
  sorry

end perimeter_of_region_proof_l417_417304


namespace false_statement_quadrilateral_diagonals_l417_417785

theorem false_statement_quadrilateral_diagonals :
  ¬ (∀ (Q : Type) [quad : quadrilateral Q] (d1 d2 : diagonal Q), 
      perpendicular d1 d2 ∧ equal_length d1 d2 → is_square Q) := sorry

end false_statement_quadrilateral_diagonals_l417_417785


namespace coins_remainder_l417_417076

theorem coins_remainder 
  (n : ℕ)
  (h₁ : n % 8 = 6)
  (h₂ : n % 7 = 2)
  (h₃ : n = 30) :
  n % 9 = 3 :=
sorry

end coins_remainder_l417_417076


namespace max_abs_P_attained_in_interval_l417_417829

theorem max_abs_P_attained_in_interval (P : ℝ → ℝ) (n : ℕ) (h_n : 3 ≤ n) 
  (roots : Fin n → ℝ) (sorted_roots : ∀ i j, i < j → roots i < roots j)
  (increasing_diff : ∀ i : Fin (n - 1), roots (⟨i+1, Nat.add_lt_of_lt_sub (i.is_lt)⟩) - roots i <
                                       roots (⟨i+2, Nat.add_lt_of_lt_add 2 (i.is_lt)⟩) - roots (⟨i+1, Nat.add_lt_of_lt_add 1 (i.is_lt)⟩))
  (P_eq_prod : ∀ x, P x = (Finset.univ.prod (λ k : Fin n, (x - roots k)))) :
  ∃ c ∈ Ico (roots (⟨n-1, sorry⟩)) (roots (⟨n, sorry⟩)), 
    ∀ x ∈ Icc (roots ⟨0, sorry⟩) (roots (⟨n, sorry⟩)), |P x| ≤ |P c| :=
sorry

end max_abs_P_attained_in_interval_l417_417829


namespace time_to_pick_72_peas_l417_417849

theorem time_to_pick_72_peas :
  (∀ t : ℕ, t = 56 / 7) → (72 / 8 = 9) :=
begin
  intro t_rate,
  have rate_of_picking := t_rate 8,
  rw rate_of_picking,
  have time := 72 / 8,
  exact time,
end

end time_to_pick_72_peas_l417_417849


namespace clock_angle_9_30_l417_417108

theorem clock_angle_9_30 : 
  let hour_hand_pos := 9.5 
  let minute_hand_pos := 6 
  let degrees_per_division := 30 
  let divisions_apart := hour_hand_pos - minute_hand_pos
  let angle := divisions_apart * degrees_per_division
  angle = 105 :=
by
  sorry

end clock_angle_9_30_l417_417108


namespace trigonometric_series_closed_form_l417_417478

theorem trigonometric_series_closed_form (x : ℝ) :
  (∑ i in Finset.range (2011 + 1), 
    sin^2 (2^i * x) * ∏ j in Finset.range i, cos^2 (2^j * x)) 
    = 1 - sin^2 (2^(2011 + 1) * x) / (4^(2011 + 1) * sin^2 x) :=
by
  sorry

end trigonometric_series_closed_form_l417_417478


namespace heptagon_perimeter_sum_l417_417413

/-- Definition of the points forming the heptagon -/
def points : List (ℝ × ℝ) := [(0,1), (1,3), (3,3), (4,2), (4,1), (3,0), (1,0), (0,1)]

/-- Definition to calculate the Euclidean distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem heptagon_perimeter_sum :
  let perimeter := (distance (0,1) (1,3)) + (distance (1,3) (3,3)) +
                   (distance (3,3) (4,2)) + (distance (4,2) (4,1)) +
                   (distance (4,1) (3,0)) + (distance (3,0) (1,0)) +
                   (distance (1,0) (0,1))
  ∃ a b c d e : ℤ, (perimeter = a + b * Real.sqrt 2 + c * Real.sqrt 3 + d * Real.sqrt 5 + e * Real.sqrt 10) ∧ 
                  (a + b + c + d + e = 9) :=
by
  sorry

end heptagon_perimeter_sum_l417_417413


namespace investment_equivalence_l417_417614

theorem investment_equivalence :
  ∃ P2 : ℝ,
    let P1 := 80804 in
    let r1 := 0.02 in
    let n1 := 2 in
    let r2 := 0.04 in
    let n2 := 4 in
    let t := 1 in
    let A1 := P1 * (1 + r1 / n1)^(n1 * t) in
    let A2 := P2 * (1 + r2 / n2)^(n2 * t) in
    A1 = A2 ∧ P2 = 79200 := 
begin 
  sorry 
end

end investment_equivalence_l417_417614


namespace min_value_log_expression_l417_417269

theorem min_value_log_expression (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h_geom_mean : z^2 = x * y) :
  ∃m, m = 9 / 8 ∧ (∀ (x y z : ℝ), 1 < x → 1 < y → 1 < z → z^2 = x * y →
  m ≤ (log 10 z / (4 * log 10 x) + log 10 z / (log 10 y))) :=
sorry

end min_value_log_expression_l417_417269


namespace part_one_part_two_l417_417622

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ (sin_C sin_A_minus_B sin_B_minus_A : ℝ), 
    sin_C * sin (A - B) = sin B * sin (C - A)

theorem part_one (a b c A B C: ℝ) 
  (h1: triangle_ABC a b c A B C)
  : 2 * a^2 = b^2 + c^2 :=
  sorry

theorem part_two {a b c A: ℝ} (h1: a = 5) (h2: cos A = 25 / 31) 
  : ∀ (B C : ℝ), triangle_ABC 5 b c A B C → a + b + c = 14 :=
  sorry

end part_one_part_two_l417_417622


namespace polynomial_coeff_sum_l417_417928

noncomputable def givenPolynomialExpansion  (x : ℝ) (a : ℕ → ℝ) : ℝ :=
  (1 - 2 * x)^10 - (a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + 
                   a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10)

theorem polynomial_coeff_sum (a : ℕ → ℝ) (h : ∀ x : ℝ, givenPolynomialExpansion x a = 0) :
  a 1 + a 2 / 2 + a 3 / 2^2 + a 4 / 2^3 + a 5 / 2^4 + a 6 / 2^5 + 
  a 7 / 2^6 + a 8 / 2^7 + a 9 / 2^8 + a 10 / 2^9 = -2 :=
begin
  sorry
end

end polynomial_coeff_sum_l417_417928


namespace first_problem_second_problem_l417_417144

-- Definitions for the first ellipse proof
def first_ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36
def second_ellipse_passes_through (x y : ℝ) : Prop := x = 3 ∧ y = -2

-- Proof problems
theorem first_problem (x y : ℝ):
  first_ellipse 3 (-2) ∧ second_ellipse_passes_through 3 (-2) ∧ 
  (∃ x y, 
    (first_ellipse x y ∧ second_ellipse_passes_through 3 (-2)) 
  → 
  ∀ x y, (x^2 / 15) + (y^2 / 10) = 1) :=
sorry

-- Definitions for the second ellipse proof
def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (c a : ℝ) : ℝ := c / a
def given_perimeter (f1 f2 p1 p2 : ℝ) : Prop := f1 + f2 + p1 + p2 = 16

theorem second_problem (a b c : ℝ):
  ellipse 0 0 a b ∧ eccentricity c a = (Real.sqrt 2) / 2 ∧ 
  given_perimeter 2 2 sqrt 2 sqrt 2 ∧ 
  (∃ a b, a = 4 ∧ b = 2 * sqrt 2 ∧ c = 2)
  → 
  ∀ x y, (x^2 / 16) + (y^2 / 8) = 1 :=
sorry

end first_problem_second_problem_l417_417144


namespace days_Dakota_spent_in_hospital_l417_417861

-- Definitions of the conditions
def bed_charge_per_day := 900
def specialist_charge_per_hour := 250
def specialist_time_per_visitation := 0.25 -- in hours
def number_of_specialists := 2
def ambulance_charge := 1800
def total_bill := 4625

-- Define the main problem as calculating the number of days
def calculate_days_in_hospital (bed_charge_per_day : ℕ)
                               (specialist_charge_per_hour : ℕ)
                               (specialist_time_per_visitation : ℝ)
                               (number_of_specialists : ℕ)
                               (ambulance_charge : ℕ)
                               (total_bill : ℕ) : ℕ :=
  let specialist_cost := specialist_charge_per_hour * specialist_time_per_visitation * number_of_specialists
  let bed_cost := total_bill - ambulance_charge - specialist_cost
  let days := bed_cost / bed_charge_per_day
  let fractional_day := (bed_cost % bed_charge_per_day) > 0 -- Check if there is a remainder
  days + if fractional_day then 1 else 0

-- The theorem stating the number of days Dakota spent in the hospital
theorem days_Dakota_spent_in_hospital : calculate_days_in_hospital bed_charge_per_day specialist_charge_per_hour specialist_time_per_visitation number_of_specialists ambulance_charge total_bill = 3 :=
by sorry

end days_Dakota_spent_in_hospital_l417_417861


namespace debra_probability_l417_417468

theorem debra_probability :
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  (p_THTHT * P) = 1 / 96 :=
by
  -- Definitions of p_tail, p_head, p_THTHT, and P
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  -- Placeholder for proof computation
  sorry

end debra_probability_l417_417468


namespace cross_product_self_zero_l417_417209

open Matrix

def u : ℝ^3 := ![u1, u2, u3]
def v : ℝ^3 := ![v1, v2, v3]

axiom huv : u × v = ![7, -3, 6]

theorem cross_product_self_zero : 
  (u + 2 • v) × (u + 2 • v) = ![0, 0, 0] := 
by
  -- The proof itself is omitted, the statement is the main goal.
  sorry

end cross_product_self_zero_l417_417209


namespace problem_statement_l417_417397

def number_of_combinations (n k : ℕ) : ℕ := Nat.choose n k

def successful_outcomes : ℕ :=
  (number_of_combinations 3 1) * (number_of_combinations 5 1) * (number_of_combinations 4 5) +
  (number_of_combinations 3 2) * (number_of_combinations 4 5)

def total_outcomes : ℕ := number_of_combinations 12 7

def probability_at_least_75_cents : ℚ :=
  successful_outcomes / total_outcomes

theorem problem_statement : probability_at_least_75_cents = 3 / 22 := by
  sorry

end problem_statement_l417_417397


namespace original_cost_of_remaining_shirt_l417_417497

theorem original_cost_of_remaining_shirt 
  (total_original_cost : ℝ) 
  (shirts_on_discount : ℕ) 
  (original_cost_per_discounted_shirt : ℝ) 
  (discount : ℝ) 
  (current_total_cost : ℝ) : 
  total_original_cost = 100 → 
  shirts_on_discount = 3 → 
  original_cost_per_discounted_shirt = 25 → 
  discount = 0.4 → 
  current_total_cost = 85 → 
  ∃ (remaining_shirts : ℕ) (original_cost_per_remaining_shirt : ℝ), 
    remaining_shirts = 2 ∧ 
    original_cost_per_remaining_shirt = 12.5 :=
by 
  sorry

end original_cost_of_remaining_shirt_l417_417497


namespace hcf_of_two_numbers_l417_417058
-- Importing the entire Mathlib library for mathematical functions

-- Define the two numbers and the conditions given in the problem
variables (x y : ℕ)

-- State the conditions as hypotheses
def conditions (h1 : x + y = 45) (h2 : Nat.lcm x y = 120) (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Prop :=
  True

-- State the theorem we want to prove
theorem hcf_of_two_numbers (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Nat.gcd x y = 1 :=
  sorry

end hcf_of_two_numbers_l417_417058


namespace triangle_side_lengths_triangle_circumradius_l417_417531

theorem triangle_side_lengths (a b c : ℝ) (h : a^2 + b^2 + c^2 - 6 * a - 8 * b - 10 * c + 50 = 0) :
  a = 3 ∧ b = 4 ∧ c = 5 :=
by sorry

theorem triangle_circumradius (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let R := c / 2 in R = 2.5 :=
by sorry

end triangle_side_lengths_triangle_circumradius_l417_417531


namespace simplification_of_expression_l417_417495

open Real

variables {α β : ℝ}

theorem simplification_of_expression (hα : cos α ≠ 0) (hβ : cos β ≠ 0) :
  (cos (α⁻¹) + cos (β⁻¹)) / (tan α * cos (β⁻¹) + tan β * cos (α⁻¹)) = cot ((α + β) / 2) :=
by {
  -- Proof will go here
  sorry
}

end simplification_of_expression_l417_417495


namespace product_x_y_l417_417965

theorem product_x_y {x y : ℝ} (h1 : -3 * x + 4 * y = 28) (h2 : 3 * x - 2 * y = 8) : 
  x * y ≈ 264 :=
sorry

end product_x_y_l417_417965


namespace math_equivalence_problem_l417_417756

theorem math_equivalence_problem :
  (2^2 + 92 * 3^2) * (4^2 + 92 * 5^2) = 1388^2 + 92 * 2^2 :=
by
  sorry

end math_equivalence_problem_l417_417756


namespace num_female_managers_l417_417056

-- Definitions based on the conditions
def total_employees : ℕ := 250
def female_employees : ℕ := 90
def total_managers : ℕ := 40
def male_associates : ℕ := 160

-- Proof statement that computes the number of female managers
theorem num_female_managers : 
  (total_managers - (total_employees - female_employees - male_associates)) = 40 := 
by 
  sorry

end num_female_managers_l417_417056


namespace find_integers_in_range_l417_417885

def is_digit (n : ℕ) : Prop := n > 0 ∧ n < 10

def F (a : Fin 9 → ℕ) : ℚ := ∑ i, ((i : ℚ) + 1) / (a i)

theorem find_integers_in_range : 
  ∀ n : ℚ, 5 ≤ n → n ≤ 45 → ∃ a : Fin 9 → ℕ, (∀ i, is_digit (a i)) ∧ F a = n :=
by
  sorry

end find_integers_in_range_l417_417885


namespace odd_sequence_property_l417_417246

def seq_a (n : ℕ) : ℕ :=
  if h : n > 0 then
    match (n : ℕ).sqrt, h with
    | k, _ => if k % 2 = 1 then k else k + 1
  else
    0

noncomputable def a_n_formula (n b c d : ℕ) : ℕ :=
  b * Int.floor (Real.sqrt (n + c)) + d

theorem odd_sequence_property (b c d : ℕ) :
  (∀ (n : ℕ), seq_a n = a_n_formula n b c d) →
  b + c + d = 2 :=
begin
  assume h,
  -- Proof goes here
  sorry
end

end odd_sequence_property_l417_417246


namespace probability_calculation_l417_417445

noncomputable def probability_at_least_seven_at_least_three_times : ℚ :=
  let p := 1 / 4
  let q := 3 / 4
  (4 * p^3 * q) + (p^4)

theorem probability_calculation :
  probability_at_least_seven_at_least_three_times = 13 / 256 :=
by sorry

end probability_calculation_l417_417445


namespace difference_mean_median_is_4_l417_417978

-- Define variables and conditions
def total_students : ℕ := 40
def score_distribution : List (ℕ × ℝ) := [(6, 60), (14, 75), (4, 82), (8, 88), (8, 92)]

-- Calculate the median score
def median_score (dist : List (ℕ × ℝ)) : ℝ :=
  let sorted_scores := dist.flat_map (λ (n, s), List.replicate n s)
  let len := sorted_scores.length
  if len % 2 = 0 then
    let mid1 := sorted_scores.get! (len / 2 - 1)
    let mid2 := sorted_scores.get! (len / 2)
    (mid1 + mid2) / 2
  else
    sorted_scores.get! (len / 2)

-- Calculate the mean score
def mean_score (dist : List (ℕ × ℝ)) : ℝ :=
  let total_score := dist.foldl (λ acc (n, s), acc + n * s) 0
  total_score / total_students

-- Main theorem statement
theorem difference_mean_median_is_4 :
  let mean := mean_score score_distribution
  let median := median_score score_distribution
  abs (mean - median) = 4 :=
by
  sorry

end difference_mean_median_is_4_l417_417978


namespace ivans_profit_l417_417609

def price_meat_per_kg : ℕ := 500
def kg_meat_sold : ℕ := 100
def price_eggs_per_dozen : ℕ := 50
def eggs_sold : ℕ := 20000
def annual_expenses : ℕ := 100000

def revenue_meat : ℕ := kg_meat_sold * price_meat_per_kg
def revenue_eggs : ℕ := eggs_sold * (price_eggs_per_dozen / 10)
def total_revenue : ℕ := revenue_meat + revenue_eggs

def profit : ℕ := total_revenue - annual_expenses

theorem ivans_profit : profit = 50000 := by
  sorry

end ivans_profit_l417_417609


namespace switches_connections_l417_417741

theorem switches_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 :=
by
  sorry

end switches_connections_l417_417741


namespace calculate_f_prime_two_l417_417916

-- Define the function based on given conditions
def f (x : ℝ) : ℝ := x^3 + 2 * x * f' 1 
-- Define its derivative 
def f' (x : ℝ) : ℝ := 3 * x^2 - 6 
-- Prove that f'(2) = 6 given the initial condition
theorem calculate_f_prime_two : f' 2 = 6 :=
by 
  -- Proof steps would go here
  sorry

end calculate_f_prime_two_l417_417916


namespace g_inverse_f_11_eq_sqrt7_or_negsqrt7_l417_417909

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def finv : ℝ → ℝ := sorry
noncomputable def ginv : ℝ → ℝ := sorry

axiom f_inv_g (x : ℝ) : finv (g x) = 2 * x^2 - 3
axiom g_has_inverse : function.has_inverse g
axiom ginv_is_inverse : ∀ y, ginv (g y) = y

theorem g_inverse_f_11_eq_sqrt7_or_negsqrt7 :
  ginv (f 11) = sqrt 7 ∨ ginv (f 11) = -sqrt 7 :=
by sorry

end g_inverse_f_11_eq_sqrt7_or_negsqrt7_l417_417909


namespace integer_solution_count_eq_four_l417_417203

def equation_solution_count (x : ℤ) : Prop :=
  (x - 2)^(25 - x^2) = 1

theorem integer_solution_count_eq_four :
  {x : ℤ | equation_solution_count x}.finite.to_finset.card = 4 :=
by sorry

end integer_solution_count_eq_four_l417_417203


namespace sequence_v_n_l417_417618

theorem sequence_v_n (v : ℕ → ℝ)
  (h_recurr : ∀ n, v (n+2) = 3 * v (n+1) - v n)
  (h_init1 : v 3 = 16)
  (h_init2 : v 6 = 211) : 
  v 5 = 81.125 :=
sorry

end sequence_v_n_l417_417618


namespace second_solution_l417_417171

theorem second_solution (hx : 40 * (4/5)^2 - 69 * (4/5) + 24 = 0) : 40 * (3/8)^2 - 69 * (3/8) + 24 = 0 :=
begin
  sorry
end

end second_solution_l417_417171


namespace difference_in_money_in_nickels_l417_417440

-- Define the given conditions
def alice_quarters (p : ℕ) : ℕ := 3 * p + 2
def bob_quarters (p : ℕ) : ℕ := 2 * p + 8

-- Define the difference in their money in nickels
def difference_in_nickels (p : ℕ) : ℕ := 5 * (p - 6)

-- The proof problem statement
theorem difference_in_money_in_nickels (p : ℕ) : 
  (5 * (alice_quarters p - bob_quarters p)) = difference_in_nickels p :=
by 
  sorry

end difference_in_money_in_nickels_l417_417440


namespace carpet_needed_l417_417086

def room_length : ℕ := 15
def room_width : ℕ := 10
def ft2_to_yd2 : ℕ := 9

theorem carpet_needed :
  (room_length * room_width / ft2_to_yd2).ceil = 17 :=
by
  sorry

end carpet_needed_l417_417086


namespace number_of_repeating_decimals_l417_417894

-- Define the condition that a number has only prime factors 2 and 5.
def has_only_prime_factors_2_and_5 (k : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → p ∣ k → p = 2 ∨ p = 5

-- Define the condition that a number has a prime factor other than 2 and 5.
def has_prime_factor_other_than_2_and_5 (k : ℕ) : Prop :=
  ∃ p : ℕ, p.prime ∧ p ∣ k ∧ p ≠ 2 ∧ p ≠ 5

-- Define the range
def range_1_to_100 : finset ℕ := finset.Icc 1 100

-- Define the condition when n/(n+1) is a repeating decimal.
def is_repeating_decimal (n : ℕ) : Prop :=
  has_prime_factor_other_than_2_and_5 (n + 1)

-- Count the number of such n in the given range.
def count_repeating_decimals : ℕ :=
  (range_1_to_100.filter is_repeating_decimal).card

theorem number_of_repeating_decimals :
  count_repeating_decimals = 86 :=
sorry

end number_of_repeating_decimals_l417_417894


namespace B_inv_cube_inverse_l417_417510

def B_inv : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![1, 4]]

theorem B_inv_cube_inverse :
  (B_inv ^ 3)⁻¹ = ![![7, -70], ![35, 42]] :=
by
  sorry

end B_inv_cube_inverse_l417_417510


namespace prove_equivalent_expression_l417_417637

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 * x - 7
def g (x : ℝ) : ℝ := x / 2 + 3

-- The statement we need to prove
theorem prove_equivalent_expression (x : ℝ) : f (g x) - g (f x) = 8 :=
by
  sorry

end prove_equivalent_expression_l417_417637


namespace dot_product_to_linear_form_l417_417662

noncomputable def proof_problem (r a : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := a.1
  let B := a.2
  let C := -m
  (r.1 * a.1 + r.2 * a.2 = m) → (A * r.1 + B * r.2 + C = 0)

-- The theorem statement
theorem dot_product_to_linear_form (r a : ℝ × ℝ) (m : ℝ) :
  proof_problem r a m :=
sorry

end dot_product_to_linear_form_l417_417662


namespace two_fixed_points_only_l417_417997

open Classical

noncomputable def exists_two_fixed_points_and_only_two (A B : Point) (body : set Point) : Prop :=
  A ∈ body ∧ B ∈ body ∧ 
  (∀ p ∈ body, p = A ∨ p = B)

theorem two_fixed_points_only (A B : Point) (body : set Point) :
  exists (rotating_body: set Point), 
    exists_two_fixed_points_and_only_two A B rotating_body :=
sorry

end two_fixed_points_only_l417_417997


namespace n_gon_partition_l417_417133

-- Define a function to determine if an n-gon can be partitioned as required
noncomputable def canBePartitioned (n : ℕ) (h : n ≥ 3) : Prop :=
  n ≠ 4 ∧ n ≥ 3

theorem n_gon_partition (n : ℕ) (h : n ≥ 3) : canBePartitioned n h ↔ (n = 3 ∨ n ≥ 5) :=
by sorry

end n_gon_partition_l417_417133


namespace calories_in_250g_l417_417876

-- Definitions of calorie content per 100 grams
def calories_per_100g_orange := 45
def calories_per_100g_honey := 304
def calories_per_100g_water := 0

-- Definitions of quantities in the beverage
def grams_orange := 150
def grams_honey := 50
def grams_water := 300

-- Definitions of total quantities
def total_grams := grams_orange + grams_honey + grams_water
def total_calories := (calories_per_100g_orange * grams_orange / 100) +
                      (calories_per_100g_honey * grams_honey / 100) +
                      (calories_per_100g_water * grams_water / 100)

-- Caloric density calculation
def caloric_density := total_calories / total_grams

-- Target proof statement
theorem calories_in_250g (total_calories : ℝ) (total_grams : ℝ) 
  (caloric_density : ℝ) : 
  250 * (total_calories / total_grams) = 110 := by
  -- calculate calories in 250 grams based on caloric density
  calc
    250 * (total_calories / total_grams) = 250 * caloric_density : by rw [caloric_density]
                                           = 250 * (219.5 / 500) : by sorry -- intermediate calculations here
                                           = 110 : by sorry

end calories_in_250g_l417_417876


namespace apples_difference_l417_417435

theorem apples_difference (A J : ℕ) (hA : A = 9) (hJ : J = 10) : J - A = 1 := 
by
  -- We include the definitions and conditions here
  rw [hA, hJ]
  -- sorry is added to skip detailed steps
  sorry

end apples_difference_l417_417435


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417015

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417015


namespace imaginary_part_of_fraction_l417_417705

open Complex

theorem imaginary_part_of_fraction : 
  ∀ (re₁ im₁ re₂ im₂ : ℝ),
  (re₁ = 2) → (im₁ = 1) → (re₂ = 1) → (im₂ = -2) →
  Im ((Complex.mk re₁ im₁) / (Complex.mk re₂ im₂)) = 1 := 
by
  intros re₁ im₁ re₂ im₂ h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  simp
  sorry

end imaginary_part_of_fraction_l417_417705


namespace false_statement_B_l417_417783

theorem false_statement_B :
  ∃ (Q : Type) [quadrilateral Q], quadrilateral.has_perpendicular_and_equal_diagonals Q ∧ ¬ quadrilateral.is_square Q := by
  sorry

end false_statement_B_l417_417783


namespace minimum_value_f_of_a_eq_one_third_l417_417576

noncomputable def f (a : ℝ) (x : ℝ) := a^(x-2) - 2 * a

theorem minimum_value_f_of_a_eq_one_third :
  ∀ (a : ℝ), 0 < a ∧ a ≠ 1 ∧ f a 2 = 1 - 2 * a → ∃ x ∈ set.Icc (0 : ℝ) 3, f (1/3) x = -1/3 := 
by 
  sorry

end minimum_value_f_of_a_eq_one_third_l417_417576


namespace breadth_of_boat_l417_417810

theorem breadth_of_boat
  (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (ρ : ℝ) (B : ℝ)
  (hL : L = 3)
  (hh : h = 0.01)
  (hm : m = 60)
  (hg : g = 9.81)
  (hρ : ρ = 1000) :
  B = 2 := by
  sorry

end breadth_of_boat_l417_417810


namespace recurring_decimals_sum_l417_417477

theorem recurring_decimals_sum :
  (0.5 + 0.1 - 0.3 + 0.6) = (1 : ℝ) := 
by {
  have h₀ : (0.5 : ℝ) = 5 / 9 := sorry,
  have h₁ : (0.1 : ℝ) = 1 / 9 := sorry,
  have h₂ : (0.3 : ℝ) = 1 / 3 := sorry,
  have h₃ : (0.6 : ℝ) = 2 / 3 := sorry,

  -- Prove the final equality using the established conversions
  sorry,
}

end recurring_decimals_sum_l417_417477


namespace product_xyz_l417_417210

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 5) : 
  x * y * z = 1 / 9 := 
by
  sorry

end product_xyz_l417_417210


namespace number_of_intersections_l417_417492

theorem number_of_intersections (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x = 4) → (x = 4 ∧ y = 0) :=
by {
  sorry
}

end number_of_intersections_l417_417492


namespace total_walnut_trees_in_park_l417_417732

-- Define initial number of walnut trees in the park
def initial_walnut_trees : ℕ := 22

-- Define number of walnut trees planted by workers
def planted_walnut_trees : ℕ := 33

-- Prove the total number of walnut trees in the park
theorem total_walnut_trees_in_park : initial_walnut_trees + planted_walnut_trees = 55 := by
  sorry

end total_walnut_trees_in_park_l417_417732


namespace angle_between_2a_plus_b_and_b_l417_417946

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V) (θ : ℝ)

axiom angle_between_vectors : real_inner (a) (b) = |a| * |b| * real.cos (θ)
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2
axiom angle_constraint : θ = real.pi / 3

theorem angle_between_2a_plus_b_and_b :
  ∠ (2 • a + b) b = real.pi / 6 :=
sorry

end angle_between_2a_plus_b_and_b_l417_417946


namespace max_modulus_complex_expression_l417_417631

theorem max_modulus_complex_expression (z : ℂ) (h : |z| = 2) : 
  ∃ x : ℝ, x = 16 * real.sqrt 2 ∧ ∀ z : ℂ, |z| = 2 → |(z - 2)^2 * (z + 2)| ≤ x :=
by
  sorry

end max_modulus_complex_expression_l417_417631


namespace badges_exchange_l417_417760

theorem badges_exchange (Vasya_initial Tolya_initial : ℕ) 
    (h1 : Vasya_initial = Tolya_initial + 5)
    (h2 : Vasya_initial - 0.24 * Vasya_initial + 0.20 * Tolya_initial = Tolya_initial - 0.20 * Tolya_initial + 0.24 * Vasya_initial - 1) 
    : Vasya_initial = 50 ∧ Tolya_initial = 45 :=
by sorry

end badges_exchange_l417_417760


namespace range_a_mul_b_sub_three_half_l417_417902

theorem range_a_mul_b_sub_three_half (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) (h4 : b = (1 + Real.sqrt 5) / 2 * a) :
  (∃ l u : ℝ, ∀ f, l ≤ f ∧ f < u ↔ f = a * (b - 3 / 2)) :=
sorry

end range_a_mul_b_sub_three_half_l417_417902


namespace bisect_perimeter_and_area_l417_417292

theorem bisect_perimeter_and_area (a b c r x y: ℝ) (O : ℝ × ℝ)
  (h1 : ∀ (P Q : ℝ × ℝ), P ≠ Q → ∃! l, ∃ t₁ t₂, P = l t₁ ∧ Q = l t₂) -- Line l passing through O
  (h2 : x + y = (a + b + c) / 2) -- Given condition: x + y = (a + b + c) / 2
: 1 / 2 * r * (x + y) = 1 / 2 * 1 / 2 * r * (a + b + c) :=
sorry -- The proof is omitted

end bisect_perimeter_and_area_l417_417292


namespace quadratic_expression_l417_417168

theorem quadratic_expression (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 6) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 98.08 := 
by sorry

end quadratic_expression_l417_417168


namespace ratio_surface_areas_l417_417900

def surface_area_of_tetrahedron (a : ℝ) : ℝ := 
  (sqrt 3) * a^2

def surface_area_of_cube (s : ℝ) : ℝ := 
  6 * s^2

theorem ratio_surface_areas (s : ℝ) (hs : s > 0) 
  (h_tetra : True) : 
  (surface_area_of_cube s / surface_area_of_tetrahedron (sqrt 2 * s)) = sqrt 3 :=
by
  sorry

end ratio_surface_areas_l417_417900


namespace positive_solution_l417_417189

variable {x y z : ℝ}

theorem positive_solution (h1 : x * y = 8 - 2 * x - 3 * y)
    (h2 : y * z = 8 - 4 * y - 2 * z)
    (h3 : x * z = 40 - 5 * x - 3 * z) :
    x = 10 := by
  sorry

end positive_solution_l417_417189


namespace seq_arithmetic_progression_l417_417389

theorem seq_arithmetic_progression (a : ℕ → ℝ) 
  (h : ∀ m n : ℕ, abs (a m + a n - a (m + n)) ≤ 1 / (m + n)) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d :=
begin
  sorry,
end

end seq_arithmetic_progression_l417_417389


namespace product_of_four_consecutive_integers_divisible_by_12_l417_417002

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l417_417002
