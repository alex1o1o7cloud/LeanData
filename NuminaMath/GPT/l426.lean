import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.GCDMonoid
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.LinearEquiv
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Ramsey
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Log
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace min_colors_three_l426_426183

-- We define the context
def is_valid (lines : Set Line) : Prop :=
  lines.card = 2015 ∧
  (∀ l1 l2 ∈ lines, l1 ≠ l2 → ¬(parallel l1 l2)) ∧
  (∀ l1 l2 l3 ∈ lines, l1 ≠ l2 → l2 ≠ l3 → l1 ≠ l3 → 
      ¬(concurrent l1 l2 l3))

def intersection_points (lines : Set Line) : Set Point :=
  { p : Point | ∃ l1 l2 ∈ lines, l1 ≠ l2 ∧ intersect l1 l2 p }

def valid_coloring (lines : Set Line) (color : Point → ℕ) : Prop :=
  ∀ l ∈ lines, ∀ p q ∈ intersection_points {l}, 
    p ≠ q ∧ ∀ r, (intersect_segment l p q r → r ∈ intersection_points {l}) → color p ≠ color q

-- Prove that for any valid set of lines, a valid coloring with 3 colors exists
theorem min_colors_three (lines : Set Line) 
  (h_valid : is_valid lines) : 
  ∃ color : Point → ℕ, 
  (∀ p ∈ intersection_points lines, color p ∈ {1, 2, 3}) ∧ 
  valid_coloring lines color :=
sorry

end min_colors_three_l426_426183


namespace total_students_l426_426645

-- Definitions based on problem conditions
def H := 36
def S := 32
def union_H_S := 59
def history_not_statistics := 27

-- The proof statement
theorem total_students : H + S - (H - history_not_statistics) = union_H_S :=
by sorry

end total_students_l426_426645


namespace hyperbola_equation_l426_426608

-- Fixed points F_1 and F_2
def F1 : ℝ × ℝ := (5, 0)
def F2 : ℝ × ℝ := (-5, 0)

-- Condition: The absolute value of the difference in distances from P to F1 and F2 is 6
def distance_condition (P : ℝ × ℝ) : Prop :=
  abs ((dist P F1) - (dist P F2)) = 6

theorem hyperbola_equation : 
  ∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ ∀ (x y : ℝ), distance_condition (x, y) → 
  (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1 :=
by
  -- We state the conditions and result derived from them
  sorry

end hyperbola_equation_l426_426608


namespace total_carrots_l426_426717

def sally_carrots : ℕ := 6
def fred_carrots : ℕ := 4
def mary_carrots : ℕ := 10

theorem total_carrots : sally_carrots + fred_carrots + mary_carrots = 20 := by
  sorry

end total_carrots_l426_426717


namespace volume_of_solid_l426_426097

theorem volume_of_solid : 
  ∀ (s : ℝ) (h : 2 : ℝ) (A : ℝ), 
  (s * Real.sqrt 2 = 4 * Real.sqrt 2) → 
  (s^2 = 16) → 
  (h = 2) → 
  (A = s^2) → 
  ∃ V : ℝ, V = (1/3) * A * h ↔ V = 32/3 := 
by 
  intros s h A h1 h2 h3 h4 
  use (1/3) * A * h 
  sorry

end volume_of_solid_l426_426097


namespace find_b_wage_days_l426_426473

-- Define the problem conditions and parameters
variables (S A B : ℚ)
variable x : ℚ

-- State the conditions
axiom h1 : S = 21 * A
axiom h2 : S = 12 * (A + B)
axiom h3 : S = x * B

-- State the proof goal
theorem find_b_wage_days : x = 28 :=
by sorry

end find_b_wage_days_l426_426473


namespace average_percent_decrease_is_35_percent_l426_426307

-- Given conditions
def last_week_small_price_per_pack := 7 / 3
def this_week_small_price_per_pack := 5 / 4
def last_week_large_price_per_pack := 8 / 2
def this_week_large_price_per_pack := 9 / 3

-- Calculate percent decrease for small packs
def small_pack_percent_decrease := ((last_week_small_price_per_pack - this_week_small_price_per_pack) / last_week_small_price_per_pack) * 100

-- Calculate percent decrease for large packs
def large_pack_percent_decrease := ((last_week_large_price_per_pack - this_week_large_price_per_pack) / last_week_large_price_per_pack) * 100

-- Calculate average percent decrease
def average_percent_decrease := (small_pack_percent_decrease + large_pack_percent_decrease) / 2

theorem average_percent_decrease_is_35_percent : average_percent_decrease = 35 := by
  sorry

end average_percent_decrease_is_35_percent_l426_426307


namespace angleA_eq_2angleB_iff_AC_eq_2MD_l426_426660

variables (A B C M D : Type)
variables [Triangle A B C]
variables [Midpoint M A B]
variables [Foot D C A B]

theorem angleA_eq_2angleB_iff_AC_eq_2MD :
  (angle A = 2 * angle B) ↔ (distance A C = 2 * distance M D) := by
  sorry

end angleA_eq_2angleB_iff_AC_eq_2MD_l426_426660


namespace distance_from_center_to_plane_l426_426265

theorem distance_from_center_to_plane (r_sphere : ℝ) (area_circle : ℝ) 
  (h_r_sphere : r_sphere = 2) (h_area_circle : area_circle = π) : 
  ∃ d : ℝ, d = sqrt 3 := 
by 
  -- Setting the radius of the circle derived from the area (π * r^2 = π).
  let r_circle := 1
  -- Based on the Pythagorean theorem in the context of the sphere's radius and the circle's radius.
  have h_distance : sqrt (r_sphere ^ 2 - r_circle ^ 2) = sqrt 3,
  sorry
  use sqrt 3
  exact h_distance

end distance_from_center_to_plane_l426_426265


namespace sum_cot_inv_series_l426_426539

noncomputable def cot_inv (x : ℝ) := real.arccot x

theorem sum_cot_inv_series :
  (∀ (m : ℝ), 0 < cot_inv m ∧ cot_inv m ≤ real.pi / 2) →
  ∑' n : ℕ, cot_inv (n^2 + n + 1) = real.pi / 2 :=
by 
  intro h
  sorry

end sum_cot_inv_series_l426_426539


namespace junior_score_l426_426478

variables (n : ℕ) (j s : ℕ)
  (num_juniors num_seniors : ℕ)
  (average_class_score average_senior_score : ℕ)

-- Given conditions
def cond1 := num_juniors = 0.1 * n
def cond2 := num_seniors = 0.9 * n
def cond3 := average_class_score = 84
def cond4 := average_senior_score = 83

-- Total scores
def total_class_score := n * average_class_score
def total_senior_score := num_seniors * average_senior_score
def total_junior_score := j * num_juniors

-- Assert total score consistency
def cons_total_score := total_class_score = total_senior_score + total_junior_score

-- Proof statement
theorem junior_score :
  cond1 →
  cond2 →
  cond3 →
  cond4 →
  cons_total_score →
  j = 93 :=
by
  sorry

end junior_score_l426_426478


namespace find_square_root_l426_426580

theorem find_square_root (x y : ℝ) (h_cond : (sqrt (2 * x + y - 2)) * ((x - y + 3)^2) < 0) : 
  (sqrt (x^2 + y) = 5 / 3) ∨ (sqrt (x^2 + y) = -5 / 3) := by
  sorry

end find_square_root_l426_426580


namespace find_c_l426_426641

theorem find_c {A B C : ℝ} (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) 
(h3 : a * Real.sin A + b * Real.sin B - c * Real.sin C = (6 * Real.sqrt 7 / 7) * a * Real.sin B * Real.sin C) :
  c = 2 :=
sorry

end find_c_l426_426641


namespace x_n_increasing_x_n_limit_l426_426022

noncomputable def Q : ℕ → (ℝ → ℝ)
| 1 := λ x, 1 + x
| 2 := λ x, 1 + 2 * x
| (2 * m + 1) := λ x, Q (2 * m) x + (m + 1) * x * Q (2 * m - 1) x
| (2 * m + 2) := λ x, Q (2 * m + 1) x + (m + 1) * x * Q (2 * m) x

noncomputable def x_n (n : ℕ) : ℝ := 
  Real.root (Q n) -- Assuming Real.root gives the largest real root of the polynomial

theorem x_n_increasing (n : ℕ) : 
  x_n n < x_n (n + 1) :=
sorry

theorem x_n_limit : 
  Tendsto (x_n) atTop (nhds 0) :=
sorry

end x_n_increasing_x_n_limit_l426_426022


namespace weight_calcium_acetate_from_acetic_acid_l426_426616

-- Definitions for the balanced reaction
def balanced_reaction : ℕ → ℕ → ℕ := λ n_acetic_acid n_calcium_hydroxide, 
    2 * n_acetic_acid = n_calcium_hydroxide * 1

-- Molar masses
def molar_mass_calcium_acetate : ℚ := 
    40.08 + 4 * 12.01 + 6 * 1.008 + 4 * 16.00

-- Calculation of calcium acetate produced
def calc_calcium_acetate (moles_acetic_acid : ℚ) : ℚ := 
    (moles_acetic_acid / 2)

-- Weight of the product
def weight_product (moles_acetate : ℚ) : ℚ := 
    moles_acetate * molar_mass_calcium_acetate

theorem weight_calcium_acetate_from_acetic_acid 
    (moles_acetic_acid : ℚ)
    (balanced : balanced_reaction 7 1) :
    weight_product (calc_calcium_acetate 7) = 553.588 :=
by
    sorry

end weight_calcium_acetate_from_acetic_acid_l426_426616


namespace function_at_neg_one_zero_l426_426589

-- Define the function f with the given conditions
variable {f : ℝ → ℝ}

-- Declare the conditions as hypotheses
def domain_condition : ∀ x : ℝ, true := by sorry
def non_zero_condition : ∃ x : ℝ, f x ≠ 0 := by sorry
def even_function_condition : ∀ x : ℝ, f (x + 2) = f (2 - x) := by sorry
def odd_function_condition : ∀ x : ℝ, f (1 - 2 * x) = -f (2 * x + 1) := by sorry

-- The main theorem to be proved
theorem function_at_neg_one_zero :
  f (-1) = 0 :=
by
  -- Use the conditions to derive the result
  sorry

end function_at_neg_one_zero_l426_426589


namespace typing_speed_ratio_l426_426811

-- Defining the conditions for the problem
def typing_speeds (T M : ℝ) : Prop :=
  (T + M = 12) ∧ (T + 1.25 * M = 14)

-- Stating the theorem with conditions and the expected result
theorem typing_speed_ratio (T M : ℝ) (h : typing_speeds T M) : M / T = 2 :=
by
  cases h
  sorry

end typing_speed_ratio_l426_426811


namespace product_ab_zero_l426_426582

variable {a b : ℝ}

theorem product_ab_zero (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
  sorry

end product_ab_zero_l426_426582


namespace total_weight_of_mixture_l426_426081

theorem total_weight_of_mixture 
  (almonds_parts : ℕ)
  (walnuts_parts : ℕ)
  (almonds_weight : ℕ)
  (total_parts := almonds_parts + walnuts_parts)
  (weight_per_part := almonds_weight / almonds_parts)
  (total_weight := weight_per_part * total_parts) :
  almonds_parts = 5 →
  walnuts_parts = 2 →
  almonds_weight = 200 →
  total_weight = 280 := 
by
  intros h_almonds_parts h_walnuts_parts h_almonds_weight
  rw [←h_almonds_parts, ←h_walnuts_parts, ←h_almonds_weight, Nat.add_comm]
  norm_num
  sorry

end total_weight_of_mixture_l426_426081


namespace incorrect_propositions_l426_426060

-- Definitions of propositions
def proposition_A := ∀ (u v w : V), ∀ (a b c : ℝ), a * u + b * v + c * w = 0 → (∃ k : ℝ, k = 0)
def proposition_B := ∀ (a : ℝ), (∀ x > 0, (ln x + x^2 - a * x) >= (ln x + x^2 - ax)) ↔ a ≤ 2 * sqrt 2
def proposition_C {V : Type*} [add_comm_group V] [module ℝ V] (O A B C P : V) (h : P = 2 • A - 2 • B + C) :=
  ∃ (k l m n : ℝ), k * O + l * A + m * B + n * C = 0
def proposition_D := ∀ (x1 : ℝ), (∃ (x2 : ℝ), x2 > 0 ∧ (exp x1 - exp 1 = log x2 + 1)) → x1 - x2 ≤ 1 - (1 / exp 1)

-- Theorem to prove the incorrectness of A and D
theorem incorrect_propositions : ¬ proposition_A ∧ ¬ proposition_D :=
by
  sorry

end incorrect_propositions_l426_426060


namespace solution_l426_426623

-- Define the condition of an odd function, which means f(x) == -f(-x)
def is_odd_function (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x ∈ Icc a b, f x = -f (-x)

-- Specifying the interval of definition
def interval := set.Icc (-5 : ℝ) (5 : ℝ)

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : is_odd_function f (-5) 5) (h_lt : f 3 < f 1)

-- The theorem we need to prove
theorem solution : f (-1) < f (-3) :=
by sorry

end solution_l426_426623


namespace rainfall_difference_l426_426784

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end rainfall_difference_l426_426784


namespace profit_difference_l426_426702

theorem profit_difference (num_records : ℕ) (sammy_price : ℕ) (bryan_price_interested : ℕ) (bryan_price_not_interested : ℕ)
  (h1 : num_records = 200) (h2 : sammy_price = 4) (h3 : bryan_price_interested = 6) (h4 : bryan_price_not_interested = 1) :
  let total_sammy := num_records * sammy_price,
      num_records_half := num_records / 2,
      total_bryan := (num_records_half * bryan_price_interested) + (num_records_half * bryan_price_not_interested),
      difference := total_sammy - total_bryan
  in difference = 100 :=
by
  sorry

end profit_difference_l426_426702


namespace range_of_a_l426_426583

open Real

variable (f g : ℝ → ℝ)
variable (a : ℝ) (x₀ : ℝ)
variable h₀ : f(-x₀) = -f(x₀)
variable h₁ : g(-x₀) = g(x₀)
variable h₂ : f(x₀) + g(x₀) = (1 / 2)^x₀

theorem range_of_a :
  (∃ x₀ ∈ Icc (1 / 2 : ℝ) 1, a * f x₀ + g (2 * x₀) = 0) →
  2 * Real.sqrt 2 ≤ a ∧ a ≤ (5 / 2) * Real.sqrt 2 :=
  by
  sorry

end range_of_a_l426_426583


namespace max_balls_in_cube_l426_426805

-- Main definition
def radius : ℝ := 3
def side_length : ℝ := 10

-- Volumes
def volume_cube (side_length : ℝ) : ℝ := side_length ^ 3
def volume_ball (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

-- Proving the maximum number of balls
theorem max_balls_in_cube (r s : ℝ) (h_r : r = 3) (h_s : s = 10) :
  ⌊(volume_cube s) / (volume_ball r)⌋ = 8 := by
  -- Reiterate the conditions given
  rw [h_r, h_s]
  -- Continue with specific assumptions or steps
  sorry

end max_balls_in_cube_l426_426805


namespace jessica_mother_age_l426_426298

theorem jessica_mother_age
  (mother_age_when_died : ℕ)
  (jessica_age_when_died : ℕ)
  (jessica_current_age : ℕ)
  (years_since_mother_died : ℕ)
  (half_age_condition : jessica_age_when_died = mother_age_when_died / 2)
  (current_age_condition : jessica_current_age = 40)
  (years_since_death_condition : years_since_mother_died = 10)
  (age_at_death_condition : jessica_age_when_died = jessica_current_age - years_since_mother_died) :
  mother_age_when_died + years_since_mother_died = 70 :=
by {
  sorry
}

end jessica_mother_age_l426_426298


namespace parabola_expression_l426_426167

theorem parabola_expression:
  (∀ x : ℝ, y = a * (x + 3) * (x - 1)) →
  a * (0 + 3) * (0 - 1) = 2 →
  a = -2 / 3 →
  (∀ x : ℝ, y = -2 / 3 * x^2 - 4 / 3 * x + 2) :=
by
  sorry

end parabola_expression_l426_426167


namespace ratio_of_altitudes_l426_426289

theorem ratio_of_altitudes (A B C D E F H : Type)
  [triangle A B C]
  (hBC : BC = 6)
  (hAC : AC = 4 * Real.sqrt 2)
  (hAngleC : angle A C B = 45)
  [altitude AD from A to BC]
  [altitude BE from B to AC]
  [altitude CF from C to AB]
  (hH : orthocenter A B C = H)
  : (AH / HD) = 1 := by
  sorry

end ratio_of_altitudes_l426_426289


namespace total_screens_sold_is_45000_l426_426122

-- Define the number of screens sold in each month based on X
variables (X : ℕ)

-- Conditions given in the problem
def screens_in_January := X
def screens_in_February := 2 * X
def screens_in_March := (screens_in_January X + screens_in_February X) / 2
def screens_in_April := min (2 * screens_in_March X) 20000

-- Given that April sales were 18000
axiom apr_sales_18000 : screens_in_April X = 18000

-- Total sales is the sum of sales from January to April
def total_sales := screens_in_January X + screens_in_February X + screens_in_March X + 18000

-- Prove that total sales is 45000
theorem total_screens_sold_is_45000 : total_sales X = 45000 :=
by sorry

end total_screens_sold_is_45000_l426_426122


namespace factor_x4_minus_81_l426_426924

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intros x
  sorry

end factor_x4_minus_81_l426_426924


namespace abs_ineq_solution_l426_426725

theorem abs_ineq_solution (x : ℝ) :
  (|x - 2| + |x + 1| < 4) ↔ (x ∈ Set.Ioo (-7 / 2) (-1) ∪ Set.Ico (-1) (5 / 2)) := by
  sorry

end abs_ineq_solution_l426_426725


namespace junior_score_l426_426477

variables (n : ℕ) (j s : ℕ)
  (num_juniors num_seniors : ℕ)
  (average_class_score average_senior_score : ℕ)

-- Given conditions
def cond1 := num_juniors = 0.1 * n
def cond2 := num_seniors = 0.9 * n
def cond3 := average_class_score = 84
def cond4 := average_senior_score = 83

-- Total scores
def total_class_score := n * average_class_score
def total_senior_score := num_seniors * average_senior_score
def total_junior_score := j * num_juniors

-- Assert total score consistency
def cons_total_score := total_class_score = total_senior_score + total_junior_score

-- Proof statement
theorem junior_score :
  cond1 →
  cond2 →
  cond3 →
  cond4 →
  cons_total_score →
  j = 93 :=
by
  sorry

end junior_score_l426_426477


namespace greg_age_is_18_l426_426906

def diana_age : ℕ := 15
def eduardo_age (c : ℕ) : ℕ := 2 * c
def chad_age (c : ℕ) : ℕ := c
def faye_age (c : ℕ) : ℕ := c - 1
def greg_age (c : ℕ) : ℕ := 2 * (c - 1)
def diana_relation (c : ℕ) : Prop := 15 = (2 * c) - 5

theorem greg_age_is_18 (c : ℕ) (h : diana_relation c) :
  greg_age c = 18 :=
by
  sorry

end greg_age_is_18_l426_426906


namespace roots_of_quadratic_l426_426767

theorem roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = -2 ∧ c = -6) :
  let Δ := b^2 - 4 * a * c in
  Δ > 0 :=
by
  obtain ⟨ha, hb, hc⟩ := h_eq
  have Δ_def : Δ = b^2 - 4 * a * c := rfl
  rw [ha, hb, hc] at Δ_def
  sorry

end roots_of_quadratic_l426_426767


namespace semi_circle_perimeter_correct_l426_426387

noncomputable def semi_circle_perimeter (r : ℝ) : ℝ :=
  π * r + 2 * r

theorem semi_circle_perimeter_correct :
  semi_circle_perimeter 28.006886134680677 = 144.06214538256845 :=
by
  sorry

end semi_circle_perimeter_correct_l426_426387


namespace fill_time_with_leakage_l426_426261

def tap_fill_rate : ℚ := 1/12
def leak_rate : ℚ := 1/36
def effective_fill_rate : ℚ := tap_fill_rate - leak_rate

theorem fill_time_with_leakage : 1 / effective_fill_rate = 18 := by
  -- self-contained definitions from conditions
  have h1 : tap_fill_rate = 1/12 := rfl
  have h2 : leak_rate = 1/36 := rfl
  have h3 : effective_fill_rate = tap_fill_rate - leak_rate := rfl

  -- calculation
  calc
    1 / effective_fill_rate
      = 1 / (tap_fill_rate - leak_rate) : by rw [h3]
  ... = 1 / (1/12 - 1/36) : by rw [h1, h2]
  ... = 1 / (3/36 - 1/36) : by norm_num
  ... = 1 / (2/36) : by norm_num
  ... = 1 / (1/18) : by norm_num
  ... = 18 : by norm_num

-- Placeholder for other statements
sorry

end fill_time_with_leakage_l426_426261


namespace min_rubles_reaching_50_points_l426_426043

-- Define conditions and prove the required rubles amount
def min_rubles_needed : ℕ := 11

theorem min_rubles_reaching_50_points (points : ℕ) (rubles : ℕ) : points = 50 ∧ rubles = min_rubles_needed → rubles = 11 :=
by
  intro h
  sorry

end min_rubles_reaching_50_points_l426_426043


namespace larger_number_of_two_l426_426411

theorem larger_number_of_two (x y : ℝ) (h1 : x - y = 3) (h2 : x + y = 29) (h3 : x * y > 200) : x = 16 :=
by sorry

end larger_number_of_two_l426_426411


namespace carol_packs_l426_426891

theorem carol_packs (invitations_per_pack total_invitations packs_bought : ℕ) 
  (h1 : invitations_per_pack = 9)
  (h2 : total_invitations = 45) 
  (h3 : packs_bought = total_invitations / invitations_per_pack) : 
  packs_bought = 5 :=
by 
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end carol_packs_l426_426891


namespace xiao_guang_guesses_48_xiao_guang_expected_guesses_l426_426279

/- 
   Proving that Xiao Guang needs exactly 3 guesses to find the target number 48 
   using the middle number strategy.
-/
theorem xiao_guang_guesses_48 : 
    ∃ (n : ℕ), n = 3 ∧ (∀ target (xiao_guess : ℕ → ℕ), target = 48 → xiao_guess n = target) := 
by
  sorry

/-
   Proving the expected number of guesses to be 6.055 for Xiao Guang
   when the target number is randomly picked between 1 and 127.
-/
theorem xiao_guang_expected_guesses : 
    ∃ (avg_guesses : ℝ), avg_guesses = 6.055 ∧ (∀ target : ℕ, target ∈ set.range (λ x, x + 1)..127 → ∃ (n : ℕ), xiao_guess target = n ∧ (n : ℝ) = avg_guesses) := 
by
  sorry

end xiao_guang_guesses_48_xiao_guang_expected_guesses_l426_426279


namespace consecutive_natural_numbers_sum_l426_426394

theorem consecutive_natural_numbers_sum :
  (∃ (n : ℕ), 0 < n → n ≤ 4 ∧ (n-1) + n + (n+1) ≤ 12) → 
  (∃ n_sets : ℕ, n_sets = 4) :=
by
  sorry

end consecutive_natural_numbers_sum_l426_426394


namespace Sarah_total_weeds_l426_426719

noncomputable def Tuesday_weeds : ℕ := 25
noncomputable def Wednesday_weeds : ℕ := 3 * Tuesday_weeds
noncomputable def Thursday_weeds : ℕ := (1 / 5) * Tuesday_weeds
noncomputable def Friday_weeds : ℕ := (3 / 4) * Tuesday_weeds - 10

noncomputable def Total_weeds : ℕ := Tuesday_weeds + Wednesday_weeds + Thursday_weeds + Friday_weeds

theorem Sarah_total_weeds : Total_weeds = 113 := by
  sorry

end Sarah_total_weeds_l426_426719


namespace sixteen_is_sixtyfour_percent_l426_426354

theorem sixteen_is_sixtyfour_percent (x : ℝ) (h : 16 / x = 64 / 100) : x = 25 :=
by sorry

end sixteen_is_sixtyfour_percent_l426_426354


namespace expected_deliveries_tomorrow_l426_426842

/-- 
A courier received 80 packages yesterday and twice as many today. On average, 90% of the packages are successfully delivered each day.
Considering the overall success rate, the expected number of packages to be delivered tomorrow is 144.
-/
theorem expected_deliveries_tomorrow : 
  let yesterday_packages := 80
  let today_packages := 2 * yesterday_packages
  let success_rate := 0.90
  today_packages * success_rate = 144 := 
by
  let yesterday_packages := 80
  let today_packages := 2 * yesterday_packages
  let success_rate := 0.90
  show today_packages * success_rate = 144 from sorry

end expected_deliveries_tomorrow_l426_426842


namespace problem_statement_l426_426649

-- Definitions based on the problem conditions
def unit_equilateral_triangle_area : ℝ := (sqrt 3) / 4

def smaller_triangle_area : ℝ := (sqrt 3) / 16

def total_area_smaller_triangles : ℝ := 4 * smaller_triangle_area

def area_R : ℝ := unit_equilateral_triangle_area + total_area_smaller_triangles

-- The dimensions of the smallest surrounding rectangle S
def width_S : ℝ := 1.5
def height_S : ℝ := 1.5

def area_S : ℝ := width_S * height_S

-- Prove the area inside S but outside R
def area_inside_S_outside_R : ℝ := area_S - area_R

theorem problem_statement : area_inside_S_outside_R = 1.384 :=
by
  -- The proof is omitted (use sorry to skip proof)
  sorry

end problem_statement_l426_426649


namespace tangent_triangle_perimeter_l426_426185

theorem tangent_triangle_perimeter :
  ∀ (O M A B C K L : Point) (MA MB MK ML : Line),
  (circle O 1) ∧
  tangent MA (circle O 1) ∧
  tangent MB (circle O 1) ∧
  tangent MK (circle O 1) ∧
  tangent ML (circle O 1) ∧
  MA ⊥ MB ∧
  K = intersection MK ML ∧
  L = second_intersection MK ML ∧
  C ∈ arc A B ∧
  K ∈ tangent_through C (circle O 1) ∧
  L ∈ tangent_through C (circle O 1) →
  perimeter_triangle K L M = 2 :=
by {
  sorry
}

end tangent_triangle_perimeter_l426_426185


namespace point_not_on_graph_l426_426435

theorem point_not_on_graph (x y : ℝ): ¬ (x = -1 ∧ y = 1 ∧ y = x / (x + 1)) :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  have : x + 1 = 0 := by rw [h1, add_zero]
  rw [this] at h4
  have : 0 ≠ 0 := by rw [h4] -- because divisor becomes zero, it's invalid
  contradiction

end point_not_on_graph_l426_426435


namespace gas_cost_per_gallon_l426_426138

-- Define the conditions as Lean definitions
def miles_per_gallon : ℕ := 32
def total_miles : ℕ := 336
def total_cost : ℕ := 42

-- Prove the cost of gas per gallon, which is $4 per gallon
theorem gas_cost_per_gallon : total_cost / (total_miles / miles_per_gallon) = 4 :=
by
  sorry

end gas_cost_per_gallon_l426_426138


namespace sum_of_reciprocals_of_squares_l426_426065

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 11) :
  (1 / (a:ℚ)^2) + (1 / (b:ℚ)^2) = 122 / 121 :=
sorry

end sum_of_reciprocals_of_squares_l426_426065


namespace prove_inequality_equality_cases_l426_426020

noncomputable theory

variable {x y z : ℝ}

theorem prove_inequality (h : x^2 + y^2 + z^2 = 3) :
  x^3 - (y^2 + yz + z^2) * x + y^2 * z + y * z^2 ≤ 3 * real.sqrt 3 := sorry

theorem equality_cases (h : x^2 + y^2 + z^2 = 3) :
  (x^3 - (y^2 + yz + z^2) * x + y^2 * z + y * z^2 = 3 * real.sqrt 3) →
  (x = real.sqrt 3 ∧ y = 0 ∧ z = 0) ∨
  (x = -real.sqrt 3 ∧ y = 0 ∧ z = 0) ∨
  (x = real.sqrt 3 / 3 ∧ y = -2 * real.sqrt 3 / 3 ∧ z = -2 * real.sqrt 3 / 3) ∨
  (x = -real.sqrt 3 / 3 ∧ y = 2 * real.sqrt 3 / 3 ∧ z = 2 * real.sqrt 3 / 3) := sorry

end prove_inequality_equality_cases_l426_426020


namespace fraction_problem_l426_426252

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end fraction_problem_l426_426252


namespace correct_operation_l426_426061

theorem correct_operation (a : ℝ) :
  (a^5)^2 = a^10 :=
by sorry

end correct_operation_l426_426061


namespace sum_interior_seventh_row_l426_426663

/-- Condition: the sum of the interior numbers in the fourth row of Pascal's Triangle is 6 -/
def sum_interior_fourth_row : ℕ := 6

/-- Condition: the sum of the interior numbers in the fifth row of Pascal's Triangle is 14 -/
def sum_interior_fifth_row : ℕ := 14

/-- Problem: What is the sum of the interior numbers of the seventh row of Pascal's Triangle? -/
theorem sum_interior_seventh_row : 
  (∀ n : ℕ, sum_interior_nth_row n = 2^(n-1) - 2) →
  sum_interior_nth_row 4 = sum_interior_fourth_row →
  sum_interior_nth_row 5 = sum_interior_fifth_row →
  sum_interior_nth_row 7 = 62 :=
by
  sorry

end sum_interior_seventh_row_l426_426663


namespace different_color_chips_probability_l426_426400

theorem different_color_chips_probability :
  let blue := 4
  let red := 3
  let yellow := 2
  let green := 5
  let orange := 6
  let total := blue + red + yellow + green + orange
  let prob_blue := blue / total
  let prob_blue_not_blue := blue / total * (total - blue) / total
  let prob_red := red / total
  let prob_red_not_red := red / total * (total - red) / total
  let prob_yellow := yellow / total
  let prob_yellow_not_yellow := yellow / total * (total - yellow) / total
  let prob_green := green / total
  let prob_green_not_green := green / total * (total - green) / total
  let prob_orange := orange / total
  let prob_orange_not_orange := orange / total * (total - orange) / total
  (prob_blue_not_blue + prob_red_not_red + prob_yellow_not_yellow + prob_green_not_green + prob_orange_not_orange) = 31 / 40 :=
by
  let blue := 4
  let red := 3
  let yellow := 2
  let green := 5
  let orange := 6
  let total := blue + red + yellow + green + orange
  have prob_blue := (blue : ℝ) / total
  have prob_blue_not_blue := prob_blue * (total - blue) / total
  have prob_red := (red : ℝ) / total
  have prob_red_not_red := prob_red * (total - red) / total
  have prob_yellow := (yellow : ℝ) / total
  have prob_yellow_not_yellow := prob_yellow * (total - yellow) / total
  have prob_green := (green : ℝ) / total
  have prob_green_not_green := prob_green * (total - green) / total
  have prob_orange := (orange : ℝ) / total
  have prob_orange_not_orange := prob_orange * (total - orange) / total
  have probability := prob_blue_not_blue + prob_red_not_red + prob_yellow_not_yellow + prob_green_not_green + prob_orange_not_orange
  show probability = 31 / 40 from sorry

end different_color_chips_probability_l426_426400


namespace line_XY_through_midpoint_K_l426_426287

-- Define the quadrilateral ABCD with right angles at vertices A and C
structure Quadrilateral (V : Type) :=
  (A B C D : V)
  (angle_A : is_right_angle A B C)
  (angle_C : is_right_angle C D A)

-- Define a circle with specified diameter
structure Circle_diameter (V : Type) :=
  (P Q : V)
  (diameter : P ≠ Q)
  (boundary_circle : ∀ R, R ∈ Circle_diameter P Q → dist P R * dist Q R = dist P Q / 2 * dist P Q / 2)

-- Define the circles on AB and CD diameters
variables {V : Type} [Metrics V] (AB_CD_circles : Π (quadrilateral : Quadrilateral V), Circle_diameter V × Circle_diameter V)

-- Define midpoints
def midpoint {V : Type} [Metrics V] (P Q : V) : V := (P + Q) / 2

-- The question. Prove that the line XY passes through the midpoint K of diagonal AC.
theorem line_XY_through_midpoint_K
  {V : Type} [Metrics V]
  (quadrilateral : Quadrilateral V)
  (circles : Circle_diameter V × Circle_diameter V)
  (intersect_XY : ∃ X Y : V, (X ∈ Circle_diameter.boundary_circle circles.1) ∧ (X ∈ Circle_diameter.boundary_circle circles.2) ∧ (Y ∈ Circle_diameter.boundary_circle circles.1) ∧ (Y ∈ Circle_diameter.boundary_circle circles.2)) :
  let K := midpoint quadrilateral.A quadrilateral.C in
  (∃ K ∈ line X Y, K = midpointquar A C) sorry

end line_XY_through_midpoint_K_l426_426287


namespace find_point_C_l426_426708

variables {A B D C : ℝ × ℝ} 

-- Given conditions:
def point_A := (10, 4)
def point_B := (1, -5)
def point_D := (0, -2)
def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  let dist := λ (p q : ℝ × ℝ), ((p.1 - q.1)^2 + (p.2 - q.2)^2) in
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B 

-- Main goal: Prove that C must be (-1, 1)
theorem find_point_C :
  point_A = A ∧ point_B = B ∧ point_D = D ∧
  (equilateral_triangle A B C) ∧
  (D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) →
  C = (-1, 1) :=
by {
  sorry
}

end find_point_C_l426_426708


namespace skittles_students_division_l426_426032

theorem skittles_students_division (n : ℕ) (h1 : 27 % 3 = 0) (h2 : 27 / 3 = n) : n = 9 := by
  sorry

end skittles_students_division_l426_426032


namespace percentage_discount_proof_l426_426500

noncomputable def ticket_price : ℝ := 25
noncomputable def price_to_pay : ℝ := 18.75
noncomputable def discount_amount : ℝ := ticket_price - price_to_pay
noncomputable def percentage_discount : ℝ := (discount_amount / ticket_price) * 100

theorem percentage_discount_proof : percentage_discount = 25 := by
  sorry

end percentage_discount_proof_l426_426500


namespace bob_calories_l426_426876

-- conditions
def slices : ℕ := 8
def half_slices (slices : ℕ) : ℕ := slices / 2
def calories_per_slice : ℕ := 300
def total_calories (half_slices : ℕ) (calories_per_slice : ℕ) : ℕ := half_slices * calories_per_slice

-- proof problem
theorem bob_calories : total_calories (half_slices slices) calories_per_slice = 1200 := by
  sorry

end bob_calories_l426_426876


namespace midpoint_BC_l426_426322

-- Given conditions
variables {A B C Y Z P X M : Type}
variables [incircle : Circumcircle B C Y Z Γ]
variables [h1 : Y ∈ Γ] [h2 : Z ∈ Γ] [h3 : Y ∈ Line A B] [h4 : Z ∈ Line A C]
variables [intersection_P : (BZ ∩ CY) = {P}]
variables [intersection_X : (AP ∩ BC) = {X}]
variables [intersection_M : (Circumcircle X Y Z ∩ BC) = {X, M}] (hM : M ≠ X)

-- Goal
theorem midpoint_BC (hM : M ∈ Circumcircle X Y Z) : dist B M = dist M C := 
begin
  sorry,
end

end midpoint_BC_l426_426322


namespace circle_area_l426_426802

noncomputable def circle_area_above_line (x y : ℝ) : Prop :=
  x^2 - 8 * x + y^2 - 18 * y + 61 = 0

theorem circle_area (A : ℝ) (H : ∀ (x y : ℝ), circle_area_above_line x y → y >= 4) :
  A = π :=
by
  sorry

end circle_area_l426_426802


namespace open_cave_iff_l426_426446

theorem open_cave_iff (N : ℕ) : (∃ (f : ℕ → ℕ) (steps : ℕ), ∀ i < N, ∀ j, f(steps * i + j) % N = f(steps * (i+1) + j) % N) ↔ ∃ k : ℕ, N = 2^k :=
sorry

end open_cave_iff_l426_426446


namespace part_a_part_b_l426_426764

variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (c : ℝ)

-- Condition: 1 = a_0
axiom a0_eq_1 : a 0 = 1

-- Condition: a is non-decreasing
axiom a_non_decreasing : ∀ n, a n ≤ a (n + 1)

-- Definition of b_n
noncomputable def b (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (1 - (a k / a (k + 1))) / Real.sqrt (a (k + 1))

-- Statement for Part (a)
theorem part_a (n : ℕ) : 0 ≤ b n ∧ b n < 2 := sorry

-- Statement for Part (b)
theorem part_b (c : ℝ) (h : 0 ≤ c ∧ c < 2) :
  ∃ a : ℕ → ℝ, (∀ n, 1 = a 0 ∧ (∀ m, a m ≤ a (m + 1))) ∧ (∀ n, b n > c) := sorry

end part_a_part_b_l426_426764


namespace sum_floor_log2_eq_16397_l426_426537

def floor_log2_sum (n : ℕ) := ∑ i in Finset.range (n+1), Nat.floor (Real.log i / Real.log 2)

theorem sum_floor_log2_eq_16397 : floor_log2_sum 2048 = 16397 :=
by
  simp [floor_log2_sum]
  sorry

end sum_floor_log2_eq_16397_l426_426537


namespace factor_x4_minus_81_l426_426928

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
sorry

end factor_x4_minus_81_l426_426928


namespace frac_pow_multiplication_l426_426494

theorem frac_pow_multiplication :
  ( (3 / 5)^5 * (4 / 7)^(-2) * (1 / 3)^4 = 11877 / 4050000 ) :=
by 
  -- Lean code to prove the statement
  sorry

end frac_pow_multiplication_l426_426494


namespace sin_of_angle_l426_426988

theorem sin_of_angle (α : ℝ) (x y : ℝ) (h1 : x = -3) (h2 : y = -4) (r : ℝ) (hr : r = Real.sqrt (x^2 + y^2)) : 
  Real.sin α = -4 / r := 
by
  -- Definitions
  let y := -4
  let x := -3
  let r := Real.sqrt (x^2 + y^2)
  -- Proof
  sorry

end sin_of_angle_l426_426988


namespace merchant_markup_l426_426458

-- Define conditions as expressions in Lean
def cost_price := 100
def selling_price := 120  -- Selling price after 20% profit on cost price
def discount := 0.20  -- 20% discount
def final_selling_price := selling_price  -- Selling price after discount is equal to $120

-- Define the marked price calculated from conditions
def marked_price := final_selling_price / (1 - discount)

-- Define the markup percentage
def markup_percentage := ((marked_price - cost_price) / cost_price) * 100

-- State the problem for proof
theorem merchant_markup : markup_percentage = 50 :=
by
  -- Proof skipped
  sorry

end merchant_markup_l426_426458


namespace spherical_to_rectangular_coordinates_l426_426513

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ),
  ρ = 5 → θ = π / 6 → φ = π / 3 →
  let x := ρ * (Real.sin φ * Real.cos θ)
  let y := ρ * (Real.sin φ * Real.sin θ)
  let z := ρ * Real.cos φ
  x = 15 / 4 ∧ y = 5 * Real.sqrt 3 / 4 ∧ z = 2.5 :=
by
  intros ρ θ φ hρ hθ hφ
  sorry

end spherical_to_rectangular_coordinates_l426_426513


namespace quadratic_has_at_most_two_solutions_l426_426341

theorem quadratic_has_at_most_two_solutions {a b c : ℝ} (h : a ≠ 0) :
  ¬ ∃ x₁ x₂ x₃ : ℝ, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ (a * x₃^2 + b * x₃ + c = 0) ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ :=
begin
  sorry
end

end quadratic_has_at_most_two_solutions_l426_426341


namespace angle_A_area_triangle_l426_426640

-- The first problem: Proving angle A
theorem angle_A (a b c : ℝ) (A C : ℝ) 
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C) : 
  A = Real.pi / 3 :=
by sorry

-- The second problem: Finding the area of triangle ABC
theorem area_triangle (a b c : ℝ) (A : ℝ)
  (h1 : a = 3)
  (h2 : b = 2 * c)
  (h3 : A = Real.pi / 3) :
  0.5 * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 :=
by sorry

end angle_A_area_triangle_l426_426640


namespace interior_angle_regular_octagon_exterior_angle_regular_octagon_l426_426421

-- Definitions
def sumInteriorAngles (n : ℕ) : ℕ := 180 * (n - 2)
def oneInteriorAngle (n : ℕ) (sumInterior : ℕ) : ℕ := sumInterior / n
def sumExteriorAngles : ℕ := 360
def oneExteriorAngle (n : ℕ) (sumExterior : ℕ) : ℕ := sumExterior / n

-- Theorem statements
theorem interior_angle_regular_octagon : oneInteriorAngle 8 (sumInteriorAngles 8) = 135 := by sorry

theorem exterior_angle_regular_octagon : oneExteriorAngle 8 sumExteriorAngles = 45 := by sorry

end interior_angle_regular_octagon_exterior_angle_regular_octagon_l426_426421


namespace tan_sum_identity_l426_426199

theorem tan_sum_identity (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.tan α + (1 / Real.tan α) = 3 :=
by
  sorry

end tan_sum_identity_l426_426199


namespace child_tickets_count_l426_426408

theorem child_tickets_count (A C : ℕ) (h1 : A + C = 21) (h2 : 5.50 * A + 3.50 * C = 83.50) : C = 16 :=
by
  sorry

end child_tickets_count_l426_426408


namespace hyperbola_eccentricity_sqrt_5_l426_426209

theorem hyperbola_eccentricity_sqrt_5
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1))
  (h2 : ∀ x : ℝ, (y = x^2 + 1))
  (h3 : ∃ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1) ∧ (y = x^2 + 1) ∧ 
        ∀ x' y' : ℝ, ((x'^2 / a^2) - (y'^2 / b^2) = 1) ∧ (y' = x^2 + 1) → (x, y) = (x', y')) :
  (∃ e : ℝ, e = sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_sqrt_5_l426_426209


namespace positive_correlation_l426_426561

variables {X Y : Type} [LinearOrder X] [LinearOrder Y]

def scatter_plot (points : list (X × Y)) : Prop :=
  ∀ (x1 x2 : X) (y1 y2 : Y), (x1 ≤ x2 ∧ y1 ≤ y2) → points = (x1, y1) :: (x2, y2) :: points.drop 2

theorem positive_correlation (points : list (X × Y)) :
  scatter_plot points →
  (∀ (x1 x2 : X) (y1 y2 : Y), (x1 ≤ x2 ∧ y1 ≤ y2) → 
    ∃ (corr_type : string), corr_type = "positive correlation") :=
by
  intros h_plot x1 x2 y1 y2 h
  use ("positive correlation")
  sorry

end positive_correlation_l426_426561


namespace probability_is_correct_l426_426363

-- Define the values of the coins
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def half_dollar := 50

-- Define the value function
def coin_value (counts : List Bool) : ℕ := 
  (if counts.head! then penny else 0) + 
  (if counts.get! 1 then nickel else 0) + 
  (if counts.get! 2 then dime else 0) + 
  (if counts.get! 3 then quarter else 0) + 
  (if counts.get! 4 then half_dollar else 0)

-- Define possible outcomes
def outcomes := List.replicate 32 (List.replicate 5 bool.default)

-- Probability computation
def probability_at_least_40_cents := outcomes.count (fun outcome => coin_value outcome ≥ 40) / outcomes.length

theorem probability_is_correct :
  probability_at_least_40_cents = 9 / 16 := by
  sorry

end probability_is_correct_l426_426363


namespace determine_f_of_2_over_3_l426_426685

-- Define the functions under the given conditions
def g (x : ℝ) : ℝ := 2 - 3 * x^2
def f (u : ℝ) : ℝ := u / (2 * (u + 3) / 3)

-- Define the theorem that states the problem
theorem determine_f_of_2_over_3 : f (g (2/3)) = 3/4 := by
  sorry

end determine_f_of_2_over_3_l426_426685


namespace taxi_overtakes_bus_in_3_hours_l426_426476

-- Definitions for the conditions
def taxi_speed : ℝ := 60
def bus_speed : ℝ := taxi_speed - 30
def head_start_time : ℝ := 3

-- Statement to be proved
theorem taxi_overtakes_bus_in_3_hours : 
  ∃ t : ℝ, (taxi_speed * t = bus_speed * (t + head_start_time)) ∧ t = 3 :=
begin
  use 3,
  split,
  {
    -- Proving that at t = 3, the equation holds
    have h1 : 60 * 3 = 30 * (3 + 3), by norm_num,
    exact h1,
  },
  {
    -- By definition of t, we have t = 3
    refl,
  }
end

end taxi_overtakes_bus_in_3_hours_l426_426476


namespace circumcircle_eqn_l426_426607

theorem circumcircle_eqn (A B C : ℝ × ℝ) (hA : A = (1, 0)) (hB : B = (0, real.sqrt 3)) (hC : C = (2, real.sqrt 3)) :
  ∃ D E F : ℝ, D = -2 ∧ E = - (4 * real.sqrt 3) / 3 ∧ F = 1 ∧ 
  ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ 
    (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = real.sqrt 3) ∨ (x = 2 ∧ y = real.sqrt 3) := 
sorry

end circumcircle_eqn_l426_426607


namespace always_real_roots_triangle_perimeter_l426_426598

-- Define the discriminant of the quadratic equation
def discriminant_quadratic (k : ℝ) : ℝ := (3*k + 1)^2 - 4*(2*k^2 + 2*k)

-- Problem statement for the discriminant
theorem always_real_roots (k : ℝ) : discriminant_quadratic k ≥ 0 := 
by
  sorry

-- Define the sides of triangle with given equation roots
def triangle_sides (k : ℝ) : ℝ × ℝ × ℝ := (6, 2*k, k + 1)

-- Define the perimeter given the sides of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Conditions of the problem
def is_isosceles (a b c : ℝ) : Prop := a = 6 ∧ (b = 2 * b) ∨ (c = k + 1)

-- Problem statement for the perimeter
theorem triangle_perimeter (k : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (h : is_isosceles a b c) :
  perimeter a b c = 16 ∨ perimeter a b c = 22 := 
by
  sorry

end always_real_roots_triangle_perimeter_l426_426598


namespace circle_line_distance_l426_426739

/-- 
  The distance from the center of the circle (x + 1)^2 + y^2 = 2 to the line y = x + 3 is √2.
-/
theorem circle_line_distance : 
  let center := (-1 : ℝ, 0 : ℝ)
  let A := 1
  let B := -1
  let C := 3
which {
  let distance := \frac{|A * center.1 + B * center.2 + C|}{\sqrt{A^2 + B^2}}
  distance = √2
} :=
by
  sorry

end circle_line_distance_l426_426739


namespace bug_visits_tiles_l426_426465

theorem bug_visits_tiles :
  let width : ℕ := 11
  let length : ℕ := 19
  width + length - Nat.gcd width length = 29 :=
by
  let width : ℕ := 11
  let length : ℕ := 19
  have h1 : Nat.gcd width length = 1 := by
    sorry
  calc
    width + length - Nat.gcd width length
      = 11 + 19 - Nat.gcd 11 19 : by rfl
  ... = 11 + 19 - 1 : by rw [h1]
  ... = 29 : by norm_num

end bug_visits_tiles_l426_426465


namespace correct_number_of_statements_l426_426966

noncomputable def PABCDEF : Type := sorry -- Define the hexagonal pyramid P-ABCDEF
noncomputable def base : Type := sorry -- Define the base hexagon ABCDEF
noncomputable def planeABC : Type := sorry -- Define the plane ABC containing the base

-- Define necessary conditions
axiom regular_base : regular_hexagon base
axiom PA_perpendicular_to_planeABC : ∀ (P A : Type), is_perpendicular_to_plane P A planeABC

-- Define propositions to be checked
axiom statement1 : ∀ (C D P A F : Type), is_parallel_to_plane (line_segment C D) (plane P A F)
axiom statement2 : ∀ (D F P A F : Type), is_perpendicular_to_plane (line_segment D F) (plane P A F)
axiom statement3 : ∀ (C F P A B : Type), is_parallel_to_plane (line_segment C F) (plane P A B)
axiom statement4 : ∀ (C F P A D : Type), is_parallel_to_plane (line_segment C F) (plane P A D)

-- Prove that exactly three out of these four statements are true
theorem correct_number_of_statements : 
    (statement1 ∨ statement2 ∨ statement3 ∨ statement4) ∧ ¬(statement1 ∧ statement2 ∧ statement3 ∧ statement4) :=
    sorry

end correct_number_of_statements_l426_426966


namespace intersect_on_semicircle_l426_426694

-- Define the problem as a Lean theorem
theorem intersect_on_semicircle
  (ABC : Triangle) (C : Point)
  (H : Point) (D : Point) (P : Point) (Q : Point) (ω : Semicircle)
  (angle_ACB : ∠ A C B = 90°)
  (altitude_CH : IsAltitude C H ABC)
  (D_inside_CBH : D ∈ Triangle CBH)
  (CH_bisects_AD : IsAngleBisector (line CH) (segment AD))
  (P_intersection : P = line.intersection (line BD) (line CH))
  (ω_semicircle : IsSemicircle ω (segment BD))
  (Q_tangent : TangentAt ω Q (line PQ))
: ∃ T ∈ ω, T ∈ line CQ ∧ T ∈ line AD :=
sorry

end intersect_on_semicircle_l426_426694


namespace total_number_of_trees_on_both_sides_of_road_l426_426453

theorem total_number_of_trees_on_both_sides_of_road
  (road_length : ℕ)
  (original_distance : ℕ)
  (additional_trees : ℕ) :
  road_length = 7200 →
  original_distance = 120 →
  additional_trees = 5 →
  (let intervals := road_length / original_distance in
   let trees_per_interval := 1 + additional_trees in
   let total_trees_one_side := intervals * trees_per_interval + 1 in
   total_trees_one_side * 2 = 722) :=
begin
  intros h1 h2 h3,
  let intervals := road_length / original_distance,
  let trees_per_interval := 1 + additional_trees,
  let total_trees_one_side := intervals * trees_per_interval + 1,
  have h_intervals : intervals = 60, sorry,
  have h_trees_per_interval : trees_per_interval = 6, sorry,
  have h_total_trees_one_side : total_trees_one_side = 361, sorry,
  rw [h1, h2, h3, h_intervals, h_trees_per_interval, h_total_trees_one_side],
  exact rfl,
end

end total_number_of_trees_on_both_sides_of_road_l426_426453


namespace width_of_room_l426_426382

theorem width_of_room (length room_area cost paving_rate : ℝ) 
  (H_length : length = 5.5) 
  (H_cost : cost = 17600)
  (H_paving_rate : paving_rate = 800)
  (H_area : room_area = cost / paving_rate) :
  room_area = length * 4 :=
by
  -- sorry to skip proof
  sorry

end width_of_room_l426_426382


namespace train_speed_l426_426038

theorem train_speed (length : ℝ) (time : ℝ) (length = 100) (time = 4.499640028797696) : 
  let distance := length * 2 in
  let relative_speed := distance / time in
  let train_speed_mps := relative_speed / 2 in
  let train_speed_kmph := train_speed_mps * 3.6 in
  train_speed_kmph ≈ 79.96762 := by
  sorry

end train_speed_l426_426038


namespace last_two_digits_factorial_squares_sum_l426_426415

theorem last_two_digits_factorial_squares_sum : 
  (∑ i in {1, 2, 3, 4, 5}.to_finset, (nat.factorial i)^2) % 100 = 17 :=
by
  sorry

end last_two_digits_factorial_squares_sum_l426_426415


namespace factor_x4_minus_81_l426_426923

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intros x
  sorry

end factor_x4_minus_81_l426_426923


namespace geometric_sequence_sum_l426_426506

theorem geometric_sequence_sum (a : ℕ → ℝ) (b : ℕ → ℝ) :
  -- Conditions
  (∀ n, a n = (1 / (3 ^ n))) ∧
  -- Definitions of a, b_n (use of logarithm base 3 and sum)
  (∀ n, b n = ∑ i in finset.range n, real.log 3 (a i)) ∧
  -- Given conditions
  (2 * a 1 + 3 * a 2 = 1) ∧
  ((a 3) ^ 2 = 9 * (a 2) * (a 6))
  -- Statement to prove
  → (∑ i in finset.range n, (1 / (b i))) = - (2 * n / (n + 1)) :=
sorry

end geometric_sequence_sum_l426_426506


namespace evaluate_f_l426_426992

def f (x : ℝ) : ℝ :=
if x > 0 then 2 * x - 1 else 1 - 2 * x

theorem evaluate_f : f 1 + f (-1) = 4 := by
  sorry

end evaluate_f_l426_426992


namespace cookies_taken_in_four_days_l426_426401

-- Define the initial conditions
def initial_cookies : ℕ := 70
def remaining_cookies : ℕ := 28
def days_in_week : ℕ := 7
def days_of_interest : ℕ := 4

-- Define the total cookies taken out in a week
def cookies_taken_week := initial_cookies - remaining_cookies

-- Define the cookies taken out each day
def cookies_taken_per_day := cookies_taken_week / days_in_week

-- Final statement to show the number of cookies taken out in four days
theorem cookies_taken_in_four_days : cookies_taken_per_day * days_of_interest = 24 := by
  sorry -- The proof steps will be here.

end cookies_taken_in_four_days_l426_426401


namespace find_side_length_of_square_l426_426037

variable (a : ℝ)

theorem find_side_length_of_square (h1 : a - 3 > 0)
                                   (h2 : 3 * a + 5 * (a - 3) = 57) :
  a = 9 := 
by
  sorry

end find_side_length_of_square_l426_426037


namespace soap_bubble_radius_l426_426099

theorem soap_bubble_radius (V_hemisphere : ℝ) (V_hemisphere_eq : V_hemisphere = 36 * Real.pi) : 
  ∃ (R : ℝ), (4 / 3) * Real.pi * R^3 = 36 * Real.pi ∧ R = 3 :=
by
  -- Introduce the variable for radius of the hemisphere
  set r := Real.cbrt 54 with hr_eq,
  -- The volume of the hemisphere should be  (2/3) * π * r^3
  have V_hemisphere_formula : (2 / 3) * Real.pi * r^3 = V_hemisphere, by sorry,
  -- Given V_hemisphere is 36 * π
  rw V_hemisphere_eq at V_hemisphere_formula,
  -- so  (2/3) * π * r^3 = 36 * π
  -- Divide both sides by π and simplify
  have r3_eq_54 : r^3 = 54, by sorry,
  -- From there, determine r
  have r_val : r = Real.cbrt 54, by sorry,
  -- Considering the original spherical bubble, its volume is  (4/3) * π * R^3
  use Real.cbrt 27,
  split,
  { -- Prove that (4 / 3) * π * (Real.cbrt 27)^3 = 36 * π
    rw [Real.cbrt_pow, ← Real.mul_self_sqrt (4 / 3 * (3 * 3 * 3)), mul_assoc, ← mul_assoc (4 / 3), ← mul_assoc (4 / 3)] at *,
    sorry
  },
  { -- Finally, note that Real.cbrt 27 = 3
    sorry
  }

end soap_bubble_radius_l426_426099


namespace min_rubles_reaching_50_points_l426_426044

-- Define conditions and prove the required rubles amount
def min_rubles_needed : ℕ := 11

theorem min_rubles_reaching_50_points (points : ℕ) (rubles : ℕ) : points = 50 ∧ rubles = min_rubles_needed → rubles = 11 :=
by
  intro h
  sorry

end min_rubles_reaching_50_points_l426_426044


namespace area_of_triangle_ABF_l426_426358

-- Define the vertices of the square
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (0, 2)

-- Define the point G
def G : ℝ × ℝ := (1, 0)

-- Define the point F as the intersection of BD and AE
def F : ℝ × ℝ := (1, 1)

-- Define the area calculation function for a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

theorem area_of_triangle_ABF : triangle_area A B F = 1 := by
  sorry

end area_of_triangle_ABF_l426_426358


namespace books_on_wednesday_l426_426057

def books_on_friday := 80
def books_returned_on_friday := 7
def books_checked_out_on_thursday := 5
def books_returned_on_thursday := 23
def books_checked_out_on_wednesday := 43

theorem books_on_wednesday (wednesday_books : ℕ) :
  wednesday_books = 98 :=
by
  have friday_before_return := books_on_friday - books_returned_on_friday
  have thursday_books := friday_before_return + books_checked_out_on_thursday - books_returned_on_thursday
  have wednesday_books := thursday_books + books_checked_out_on_wednesday
  exact wednesday_books = 98

end books_on_wednesday_l426_426057


namespace harmonic_mean_of_3_6_12_l426_426497

-- Defining the harmonic mean function
def harmonic_mean (a b c : ℕ) : ℚ := 
  3 / ((1 / (a : ℚ)) + (1 / (b : ℚ)) + (1 / (c : ℚ)))

-- Stating the theorem
theorem harmonic_mean_of_3_6_12 : harmonic_mean 3 6 12 = 36 / 7 :=
by
  sorry

end harmonic_mean_of_3_6_12_l426_426497


namespace rainfall_difference_l426_426788

noncomputable def r₁ : ℝ := 26
noncomputable def r₂ : ℝ := 34
noncomputable def r₃ : ℝ := r₂ - 12
noncomputable def avg : ℝ := 140

theorem rainfall_difference : (avg - (r₁ + r₂ + r₃)) = 58 := 
by
  sorry

end rainfall_difference_l426_426788


namespace factor_x4_minus_81_l426_426931

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
sorry

end factor_x4_minus_81_l426_426931


namespace convert_angle_degrees_to_radians_l426_426512

theorem convert_angle_degrees_to_radians :
  ∃ (k : ℤ) (α : ℝ), -1125 * (Real.pi / 180) = 2 * k * Real.pi + α ∧ 0 ≤ α ∧ α < 2 * Real.pi ∧ (-8 * Real.pi + 7 * Real.pi / 4) = 2 * k * Real.pi + α :=
by {
  sorry
}

end convert_angle_degrees_to_radians_l426_426512


namespace primary_school_sampling_l426_426644

theorem primary_school_sampling:
  let middle_school_students := 10900
  let primary_school_students := 11000
  let sample_size := 243
  let total_students := middle_school_students + primary_school_students
  let sampling_ratio := (sample_size : ℚ) / total_students
  let sampled_primary := (sampling_ratio * primary_school_students).toNat
  sampled_primary = 122 := 
by
  sorry

end primary_school_sampling_l426_426644


namespace village_Y_initial_population_l426_426800

def population_X := 76000
def decrease_rate_X := 1200
def increase_rate_Y := 800
def years := 17

def population_X_after_17_years := population_X - decrease_rate_X * years
def population_Y_after_17_years (P : Nat) := P + increase_rate_Y * years

theorem village_Y_initial_population (P : Nat) (h : population_Y_after_17_years P = population_X_after_17_years) : P = 42000 :=
by
  sorry

end village_Y_initial_population_l426_426800


namespace min_distance_midpoint_to_line_l426_426595

-- Define the parametric equations of curve C
def curve_C (φ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos φ, -Real.sqrt 3 + 2 * Real.sin φ)

-- Define the Cartesian coordinates of point P
def point_P : ℝ × ℝ := (3, Real.sqrt 3)

-- Define the ordinary equation of the line l
def line_l (x y : ℝ) : ℝ := 4 * x + 3 * y + 1

-- Define the coordinates of the midpoint M of PQ
def midpoint_M (φ : ℝ) : ℝ × ℝ :=
  (3 / 2 + Real.cos φ, Real.sin φ)

-- Function to compute the distance from M to the line l
def distance_to_line (φ : ℝ) : ℝ :=
  abs (4 * (3 / 2 + Real.cos φ) + 3 * Real.sin φ + 1) / 5

-- Theorem stating the minimum distance is 2/5
theorem min_distance_midpoint_to_line : (∀ φ : ℝ, distance_to_line φ) = 2 / 5 :=
by
  sorry

end min_distance_midpoint_to_line_l426_426595


namespace unattainable_y_l426_426597

theorem unattainable_y (x : ℚ) (h : x ≠ -5 / 4) : y = -3 / 4 → ¬ ∃ x, y = (2 - 3 * x) / (4 * x + 5) :=
by
  intro hy
  rw hy
  have h1 : 4 * y + 3 = 0 := by {
    show 4 * (-3 / 4) + 3 = 0
    calc
      4 * (-3 / 4) + 3
        = -3 + 3 : by ring
    ... = 0 : by norm_num
  }
  rw h1
  -- y = -3 / 4 makes the denominator zero, contradiction
  }

#check unattainable_y

end unattainable_y_l426_426597


namespace one_time_product_cost_l426_426856

def variable_cost : ℝ := 8.25
def selling_price : ℝ := 21.75
def books_to_sell : ℕ := 4180

theorem one_time_product_cost :
  let C := (selling_price - variable_cost) * books_to_sell in
  C = 56430 := by
  sorry

end one_time_product_cost_l426_426856


namespace remainder_product_mod_eq_l426_426254

theorem remainder_product_mod_eq (n : ℤ) :
  ((12 - 2 * n) * (n + 5)) % 11 = (-2 * n^2 + 2 * n + 5) % 11 := by
  sorry

end remainder_product_mod_eq_l426_426254


namespace no_nontrivial_sum_periodic_functions_l426_426085

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

def is_nontrivial_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  periodic f p ∧ ∃ x y, x ≠ y ∧ f x ≠ f y

theorem no_nontrivial_sum_periodic_functions (g h : ℝ → ℝ) :
  is_nontrivial_periodic_function g 1 →
  is_nontrivial_periodic_function h π →
  ¬ ∃ T > 0, ∀ x, (g + h) (x + T) = (g + h) x :=
sorry

end no_nontrivial_sum_periodic_functions_l426_426085


namespace equilateral_triangle_pigeonhole_l426_426572

/-- Given an equilateral triangle with side length 2, there exist two points whose distance from each other is at most 1, if there are five points inside the triangle. -/
theorem equilateral_triangle_pigeonhole 
  (points : Fin 5 → EuclideanSpace ℝ 2) 
  (h_points: ∀ i, ∀ j, i ≠ j → points i ≠ points j)
  (h_triangle : ∀ p, IsInEquilateralTriangle p 2) :
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 1 := 
sorry

end equilateral_triangle_pigeonhole_l426_426572


namespace response_rate_increase_l426_426475

noncomputable def original_survey_customers : ℕ := 90
noncomputable def original_survey_responses : ℕ := 7
noncomputable def redesigned_survey_customers : ℕ := 63
noncomputable def redesigned_survey_responses : ℕ := 9

def response_rate (customers responses : ℕ) : ℚ :=
  (responses : ℚ) / customers

def percentage_increase (old_rate new_rate : ℚ) : ℚ :=
  ((new_rate - old_rate) / old_rate) * 100

theorem response_rate_increase :
  let original_rate := response_rate original_survey_customers original_survey_responses,
      redesigned_rate := response_rate redesigned_survey_customers redesigned_survey_responses,
      increase := percentage_increase original_rate redesigned_rate
  in increase ≈ 83.68 :=
by 
  sorry

end response_rate_increase_l426_426475


namespace smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l426_426996

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x

theorem smallest_positive_period_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

theorem range_of_f : ∀ y : ℝ, y ∈ Set.range f ↔ -1 ≤ y ∧ y ≤ 1 := by
  sorry

theorem intervals_monotonically_increasing : 
  ∀ k : ℤ, 
  ∀ x : ℝ, 
  (2 * k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + 5 * Real.pi / 6) → 
  (f (x + Real.pi / 6) - f x) ≥ 0 := by
  sorry

end smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l426_426996


namespace a_beats_b_by_7_seconds_l426_426642

/-
  Given:
  1. A's time to finish the race is 28 seconds (tA = 28).
  2. The race distance is 280 meters (d = 280).
  3. A beats B by 56 meters (dA - dB = 56).
  
  Prove:
  A beats B by 7 seconds (tB - tA = 7).
-/

theorem a_beats_b_by_7_seconds 
  (tA : ℕ) (d : ℕ) (speedA : ℕ) (dB : ℕ) (tB : ℕ) 
  (h1 : tA = 28) 
  (h2 : d = 280) 
  (h3 : d - dB = 56) 
  (h4 : speedA = d / tA) 
  (h5 : dB = speedA * tA) 
  (h6 : tB = d / speedA) :
  tB - tA = 7 := 
sorry

end a_beats_b_by_7_seconds_l426_426642


namespace factor_x4_minus_81_l426_426926

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intros x
  sorry

end factor_x4_minus_81_l426_426926


namespace value_of_72_a_in_terms_of_m_and_n_l426_426563

theorem value_of_72_a_in_terms_of_m_and_n (a m n : ℝ) (hm : 2^a = m) (hn : 3^a = n) :
  72^a = m^3 * n^2 :=
by sorry

end value_of_72_a_in_terms_of_m_and_n_l426_426563


namespace pq_square_solution_l426_426687

theorem pq_square_solution :
  let p q : ℚ := if h : ∃ x : ℚ, 6 * x^2 - 7 * x - 20 = 0 then Classical.choose h else 0
  in (p - q) ^ 2 = 529 / 36 :=
by
  sorry

end pq_square_solution_l426_426687


namespace sufficient_but_not_necessary_l426_426824

theorem sufficient_but_not_necessary (a b : ℝ) (h : a * b ≠ 0) : 
  (¬ (a = 0)) ∧ ¬ ((a ≠ 0) → (a * b ≠ 0)) :=
by {
  -- The proof will be constructed here and is omitted as per the instructions
  sorry
}

end sufficient_but_not_necessary_l426_426824


namespace DN_eq_CM_l426_426278

theorem DN_eq_CM (A B C D M P N : Point)
  (h1 : IsRectangle A B C D)
  (h2 : OnSide M C D)
  (h3 : Perpendicular C (LineThrough M B))
  (h4 : Perpendicular D (LineThrough M A))
  (h5 : P = Intersection (PerpendicularLineFrom C (LineThrough M B)) (PerpendicularLineFrom D (LineThrough M A)))
  (h6 : N = FootOfPerpendicularFrom P (LineThrough C D)) :
  Distance D N = Distance C M := sorry

end DN_eq_CM_l426_426278


namespace P_ne_77_for_integers_l426_426345

def P (x y : ℤ) : ℤ :=
  x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_ne_77_for_integers (x y : ℤ) : P x y ≠ 77 :=
by
  sorry

end P_ne_77_for_integers_l426_426345


namespace bacteria_doubling_l426_426368

theorem bacteria_doubling :
  ∀ (n : ℕ),
    (∀ k, k = 3 * (nat.log 256) → 256 = nat.pow 2 8) →
    51,200 = 200 * 2 ^ (n / 3) →
    51,200 = 51,200 →
    200 = 200 →
    3 * 8 = n :=
begin
  sorry
end

end bacteria_doubling_l426_426368


namespace a8_value_l426_426969

def sequence_sum (n : ℕ) : ℕ := 2^n - 1

def nth_term (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  S n - S (n - 1)

theorem a8_value : nth_term sequence_sum 8 = 128 :=
by
  -- Proof goes here
  sorry

end a8_value_l426_426969


namespace data_analysis_proof_l426_426656

noncomputable def mode_of_list (l : List ℕ) : ℕ :=
  l.groupBy id
   .maxBy (λ g, g.length)
   .headI

noncomputable def fraction_less_than (n : ℕ) (l : List ℕ) : ℚ :=
  (l.count (λ x => x < n) : ℕ) / (l.length : ℕ)

noncomputable def first_quartile (l : List ℕ) : ℕ :=
  let sorted := l.sorted
  sorted.take (sorted.length / 2) |>.nth!(sorted.length / 4)

noncomputable def median_within_first_quartile (l : List ℕ) : ℕ :=
  let q1 := first_quartile l
  let within_q1 := l.filter (λ x => x <= q1)
  let sorted_within_q1 := within_q1.sorted
  if sorted_within_q1.length % 2 == 1 then
    sorted_within_q1.nth!(sorted_within_q1.length / 2)
  else
    (sorted_within_q1.nth!(sorted_within_q1.length / 2 - 1)
      + sorted_within_q1.nth!(sorted_within_q1.length / 2)) / 2

theorem data_analysis_proof :
  let l := [3, 3, 4, 4, 5, 5, 5, 5, 7, 11, 21]
  mode_of_list l = 5 ∧
  fraction_less_than 5 l = (4 : ℚ) / 11 ∧
  first_quartile l = 4 ∧
  median_within_first_quartile l = 4 :=
by sorry

end data_analysis_proof_l426_426656


namespace checkerboard_squares_count_l426_426452

theorem checkerboard_squares_count :
  let checkerboard : List (List Bool) :=
    List.replicate 10 (List.cycle [true, false]) ++
    List.replicate 10 (List.cycle [false, true])

  ∃ checkerboard : List (List Bool), 
    (∀ i j, checkerboard i j = 
      if (i + j) % 2 = 0 then true else false) → 
  ∑ n in [4, 5, 6, 7, 8, 9, 10], 
    (10 - n + 1)^2 
    = 140 :=
begin
  sorry
end

end checkerboard_squares_count_l426_426452


namespace common_leading_digit_of_powers_l426_426388

theorem common_leading_digit_of_powers (n : ℕ) (h1 : (2 ^ n).digits ≠ [] ∧ (5 ^ n).digits ≠ []) (h2 : (2 ^ n).digits.head = (5 ^ n).digits.head) :
  (2 ^ n).digits.head = 3 :=
by
  sorry

end common_leading_digit_of_powers_l426_426388


namespace total_drivers_proof_l426_426027

-- Define the conditions
def community_A_total_drivers : ℕ := 96
def community_A_sampled_drivers : ℕ := 12
def community_A_sampled_drivers_B : ℕ := 21
def community_C_sampled_drivers : ℕ := 25
def community_D_sampled_drivers : ℕ := 43

-- Define the total number of sampled drivers
def total_sampled_drivers : ℕ := 
  community_A_sampled_drivers + 
  community_A_sampled_drivers_B + 
  community_C_sampled_drivers +
  community_D_sampled_drivers

-- Define the sampling fraction for community A
def sampling_fraction_A : ℚ := 
  community_A_sampled_drivers / community_A_total_drivers.toRat

-- The main problem statement (proof problem)
theorem total_drivers_proof : 
  (total_sampled_drivers.toRat / sampling_fraction_A) = (808 : ℚ) :=
by
  -- This is where the proof would go, but is left as sorry
  sorry

end total_drivers_proof_l426_426027


namespace sum_of_segments_eq_side_l426_426700

variable {A B C D L M N E F: Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable [Point L] [Point M] [Point N] [Point E] [Point F]
variable [IsRectangle A B C D]
variable [On AB L]
variable [On AD M]
variable [On BC N]
variable [Ray L M]
variable [Ray L N]
variable [AngleDivideThree L A L B]
variable [CircleWithDiameter M N]
variable [IntersectionPoints E F Circle]

theorem sum_of_segments_eq_side (h_rectangle: IsRectangle A B C D)
                                (h_on_ab: On AB L)
                                (h_on_ad: On AD M)
                                (h_on_bc: On BC N)
                                (h_ray_lm: Ray L M)
                                (h_ray_ln: Ray L N)
                                (h_angle_divide: AngleDivideThree L A L B)
                                (h_circle_mn: CircleWithDiameter M N)
                                (h_intersection_points: IntersectionPoints E F Circle) :
                                segment_length E L + segment_length L F = segment_length A B :=
sorry

end sum_of_segments_eq_side_l426_426700


namespace find_ordered_pair_l426_426011

-- Definitions
def parameterized_line (u : ℝ) (m v : ℝ) : ℝ × ℝ :=
  (2 + u * m, v + u * 8)

def line_equation (x y : ℝ) : Prop :=
  y = -x + 3

-- Theorem statement
theorem find_ordered_pair : ∃ (v m : ℝ), 
  (∀ (u : ℝ), line_equation (fst (parameterized_line u m v)) (snd (parameterized_line u m v))) ∧
  v = 1 ∧ m = -8 :=
by
  sorry

end find_ordered_pair_l426_426011


namespace unique_nat_with_six_divisors_and_sum_of_3500_l426_426699

noncomputable def num_divisors (n : ℕ) : ℕ :=
  (multiset.range (n + 1)).countp (λ d, d > 0 ∧ n % d = 0)

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  (multiset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).sum

theorem unique_nat_with_six_divisors_and_sum_of_3500 (M : ℕ) 
  (h1 : num_divisors M = 6) 
  (h2 : sum_divisors M = 3500) : 
  M = 1996 :=
sorry

end unique_nat_with_six_divisors_and_sum_of_3500_l426_426699


namespace dice_sum_to_24_prob_l426_426953

theorem dice_sum_to_24_prob : 
  let events := { (x, y, z, w) : ℕ × ℕ × ℕ × ℕ | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ z ∈ {1, 2, 3, 4, 5, 6} ∧ w ∈ {1, 2, 3, 4, 5, 6} }
  let event_sum_24 := { (x, y, z, w) ∈ events | x + y + z + w = 24 }
  (finset.card event_sum_24.to_finset) / (finset.card events.to_finset) = 1 / 1296 :=
begin
  sorry
end

end dice_sum_to_24_prob_l426_426953


namespace subsets_count_eq_three_pow_n_l426_426447

theorem subsets_count_eq_three_pow_n (n : ℕ) (C : Finset (Fin n)) :
  let A B : Finset (Fin n),
  ∃ (A B : Finset (Fin n)), (A ∩ B = ∅) ∧ (A ⊆ B) → (card (C.powerset) = 3^n) :=
by
  sorry

end subsets_count_eq_three_pow_n_l426_426447


namespace sum_floor_log2_eq_16397_l426_426538

def floor_log2_sum (n : ℕ) := ∑ i in Finset.range (n+1), Nat.floor (Real.log i / Real.log 2)

theorem sum_floor_log2_eq_16397 : floor_log2_sum 2048 = 16397 :=
by
  simp [floor_log2_sum]
  sorry

end sum_floor_log2_eq_16397_l426_426538


namespace jack_baseball_cards_l426_426665

variable (F B : ℕ)

-- Conditions
def total_cards (F B : ℕ) : Prop := (F + B = 125)
def baseball_cards_equation (F B : ℕ) : Prop := (B = 3 * F + 5)

-- Proof problem: prove B = 95 given the conditions
theorem jack_baseball_cards : total_cards F B ∧ baseball_cards_equation F B → B = 95 :=
by
  intros h
  cases h with h_tot h_eqn
  sorry

end jack_baseball_cards_l426_426665


namespace octahedral_dice_sum_16_probability_l426_426376

theorem octahedral_dice_sum_16_probability : 
  let outcomes := (1:8) × (1:8)
  let successful_outcome := (8, 8) in
  (∑ outcome in outcomes, if outcome.1 + outcome.2 = 16 then 1 else 0) / (8 * 8) = 1 / 64 :=
sorry

end octahedral_dice_sum_16_probability_l426_426376


namespace value_of_a_l426_426990

theorem value_of_a (a : ℝ) (h : ((complex.I * a + 2) * complex.I).re = -((complex.I * a + 2) * complex.I).im) : a = 2 :=
sorry

end value_of_a_l426_426990


namespace total_carrots_l426_426716

theorem total_carrots (sally_carrots fred_carrots mary_carrots : ℕ)
  (h_sally : sally_carrots = 6)
  (h_fred : fred_carrots = 4)
  (h_mary : mary_carrots = 10) :
  sally_carrots + fred_carrots + mary_carrots = 20 := 
by sorry

end total_carrots_l426_426716


namespace find_n_l426_426169

theorem find_n : ∃ n : ℤ, (0 ≤ n ∧ n ≤ 9) ∧ n ≡ -5678 [MOD 10] ∧ n = 2 :=
by {
  use 2,
  split,
  { split,
    { norm_num }, 
    { norm_num }, 
  },
  split,
  { exact Int.mod_eq_of_lt (by norm_num) (by norm_num) },
  { refl }
}

end find_n_l426_426169


namespace scientific_notation_570_million_l426_426729

theorem scientific_notation_570_million:
  (570 * 10^6 : ℝ) = (5.7 * 10^8 : ℝ) :=
sorry

end scientific_notation_570_million_l426_426729


namespace no_partition_exists_l426_426664

open Finset

theorem no_partition_exists :
  ¬ ∃ (G : Finset (Finset ℕ)),
    (∀ g ∈ G, ∃ a b c, g = {a, b, c} ∧ (a + b = c ∨ a + c = b ∨ b + c = a)) ∧
    G.card = 11 ∧
    (∀ g ∈ G, g.card = 3) ∧
    (⋃₀ G) = (finset.range 34).erase 0 :=  
sorry

end no_partition_exists_l426_426664


namespace calculate_expression_l426_426888

theorem calculate_expression :
  (121^2 - 110^2 + 11) / 10 = 255.2 := 
sorry

end calculate_expression_l426_426888


namespace duration_of_period_l426_426848

variable (t : ℝ)

theorem duration_of_period:
  (2800 * 0.185 * t - 2800 * 0.15 * t = 294) ↔ (t = 3) :=
by
  sorry

end duration_of_period_l426_426848


namespace index_of_50th_term_neg_l426_426515

-- Given conditions
def sequence_b (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), (Real.sin k + Real.cos k)

-- Statement
theorem index_of_50th_term_neg : 
  ∃ n : ℕ, (∃ m : ℕ, m = 50 ∧ sequence_b n < 0) ∧
           n = 314 := 
sorry

end index_of_50th_term_neg_l426_426515


namespace average_score_last_3_matches_l426_426732

theorem average_score_last_3_matches
  (avg_12 : ℕ → ℕ → ℕ)
  (avg_5 : ℕ → ℕ → ℕ)
  (avg_4 : ℕ → ℕ → ℕ)
  (runs12 : avg_12 12 62)
  (runs5 : avg_5 5 52)
  (runs4 : avg_4 4 58) :
  avg_3 3 84 :=
by
  sorry

end average_score_last_3_matches_l426_426732


namespace at_least_two_equal_l426_426555

-- Define the problem
theorem at_least_two_equal (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x^2 / y) + (y^2 / z) + (z^2 / x) = (x^2 / z) + (y^2 / x) + (z^2 / y)) :
  x = y ∨ y = z ∨ z = x := 
by 
  sorry

end at_least_two_equal_l426_426555


namespace push_mower_cuts_one_acre_per_hour_l426_426296

noncomputable def acres_per_hour_push_mower : ℕ :=
  let total_acres := 8
  let fraction_riding := 3 / 4
  let riding_mower_rate := 2
  let mowing_hours := 5
  let acres_riding := fraction_riding * total_acres
  let time_riding_mower := acres_riding / riding_mower_rate
  let remaining_hours := mowing_hours - time_riding_mower
  let remaining_acres := total_acres - acres_riding
  remaining_acres / remaining_hours

theorem push_mower_cuts_one_acre_per_hour :
  acres_per_hour_push_mower = 1 := 
by 
  -- Detailed proof steps would go here.
  sorry

end push_mower_cuts_one_acre_per_hour_l426_426296


namespace simplify_and_substitute_l426_426721

def simplify_expr (a : ℝ) : ℝ :=
  (a+1 - (5+2*a)/(a+1)) / ((a^2 + 4*a + 4)/(a+1))

theorem simplify_and_substitute 
  (a : ℝ)
  (h1 : a ≠ -1)
  (h2 : a ≠ -2) :
  simplify_expr (-3) = 5 := 
by 
  sorry

end simplify_and_substitute_l426_426721


namespace probability_score_odd_l426_426505

-- Define the radii of the circles
def outer_radius : ℝ := 8
def inner_radius : ℝ := 4

-- Define the point values for regions in both circles
def points_inner : List ℕ := [3, 4, 4]
def points_outer : List ℕ := [4, 3, 3]

-- Define the areas of regions based on radius
def area_inner_region : ℝ := π * inner_radius^2 / 3
def area_outer_region : ℝ := π * (outer_radius^2 - inner_radius^2) / 3

-- Define the probability of hitting each type of region
def prob_odd_region : ℝ :=
  (2 * area_outer_region + area_inner_region) / (π * outer_radius^2)

def prob_even_region : ℝ :=
  (2 * area_inner_region + area_outer_region) / (π * outer_radius^2)

-- Probability that the score is odd when two darts are thrown
def prob_odd_score : ℝ :=
  2 * prob_odd_region * prob_even_region

theorem probability_score_odd :
  prob_odd_score = 35 / 72 :=
by
  -- proof goes here
  sorry

end probability_score_odd_l426_426505


namespace larger_solution_quadratic_l426_426170

theorem larger_solution_quadratic :
  ∃ x : ℝ, x^2 - 13 * x + 30 = 0 ∧ (∀ y : ℝ, y^2 - 13 * y + 30 = 0 → y ≤ x) ∧ x = 10 := 
by
  sorry

end larger_solution_quadratic_l426_426170


namespace batting_score_difference_l426_426734

variable (total_runs total_runs_excl HL high_score diff low_score : ℕ)

noncomputable def difference_between_highest_and_lowest_score (total_runs : ℕ) (total_runs_excl : ℕ) (high_score : ℕ) : ℕ :=
  let HL := total_runs - total_runs_excl
  let low_score := HL - high_score
  high_score - low_score

theorem batting_score_difference :
  let total_runs := 60 * 46 in
  let total_runs_excl := 58 * 44 in
  let high_score := 174 in
  difference_between_highest_and_lowest_score total_runs total_runs_excl high_score = 140 :=
by
  sorry

end batting_score_difference_l426_426734


namespace flowers_per_vase_l426_426483

theorem flowers_per_vase (carnations roses vases total_flowers flowers_per_vase : ℕ)
  (h1 : carnations = 7)
  (h2 : roses = 47)
  (h3 : vases = 9)
  (h4 : total_flowers = carnations + roses)
  (h5 : flowers_per_vase = total_flowers / vases):
  flowers_per_vase = 6 := 
by {
  sorry
}

end flowers_per_vase_l426_426483


namespace proof_problem_l426_426205

-- Define α in degrees
def α_deg : ℝ := 1680

-- Convert α to radians
def α_rad : ℝ := α_deg * (Real.pi / 180)

-- Define β and k
def k : ℤ := 4
def β : ℝ := (4 * Real.pi / 3)

-- Define θ as coterminal to α and in the interval (-4π, -2π)
def θ : ℝ := -8 * Real.pi / 3

-- Define the interval condition for θ
def interval_condition (θ : ℝ) := -4 * Real.pi < θ ∧ θ < -2 * Real.pi 

-- Proof statement
theorem proof_problem : 
  α_rad = (k * 2 * Real.pi + β) ∧ interval_condition θ ∧
  ∃ k' : ℤ, θ = 2 * k' * Real.pi + β :=
by {
  unfold α_rad k β θ interval_condition,
  -- The actual proof content would go here
  sorry
}

end proof_problem_l426_426205


namespace find_h_l426_426639

def quadratic_expr : ℝ → ℝ := λ x, 3 * x^2 + 9 * x + 20

theorem find_h : ∃ (a k : ℝ) (h : ℝ), 
  (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 :=
sorry

end find_h_l426_426639


namespace interior_angle_regular_octagon_exterior_angle_regular_octagon_l426_426420

-- Definitions
def sumInteriorAngles (n : ℕ) : ℕ := 180 * (n - 2)
def oneInteriorAngle (n : ℕ) (sumInterior : ℕ) : ℕ := sumInterior / n
def sumExteriorAngles : ℕ := 360
def oneExteriorAngle (n : ℕ) (sumExterior : ℕ) : ℕ := sumExterior / n

-- Theorem statements
theorem interior_angle_regular_octagon : oneInteriorAngle 8 (sumInteriorAngles 8) = 135 := by sorry

theorem exterior_angle_regular_octagon : oneExteriorAngle 8 sumExteriorAngles = 45 := by sorry

end interior_angle_regular_octagon_exterior_angle_regular_octagon_l426_426420


namespace round_to_nearest_hundredth_l426_426349

theorem round_to_nearest_hundredth (x : ℝ) (h : x = 23.7495) : Real.round (x * 100) / 100 = 23.75 :=
by
  rw [h]
  sorry

end round_to_nearest_hundredth_l426_426349


namespace quadratic_has_two_distinct_real_roots_l426_426770

theorem quadratic_has_two_distinct_real_roots :
  ∀ x : ℝ, ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (x^2 - 2 * x - 6 = 0 ∧ x = r1 ∨ x = r2) :=
by sorry

end quadratic_has_two_distinct_real_roots_l426_426770


namespace tiffany_lives_after_game_l426_426073

/-- Tiffany's initial number of lives -/
def initial_lives : ℕ := 43

/-- Lives Tiffany loses in the hard part of the game -/
def lost_lives : ℕ := 14

/-- Lives Tiffany gains in the next level -/
def gained_lives : ℕ := 27

/-- Calculate the total lives Tiffany has after losing and gaining lives -/
def total_lives : ℕ := (initial_lives - lost_lives) + gained_lives

-- Prove that the total number of lives Tiffany has is 56
theorem tiffany_lives_after_game : total_lives = 56 := by
  -- This is where the proof would go
  sorry

end tiffany_lives_after_game_l426_426073


namespace range_of_f_l426_426424

def f (x : ℝ) : ℝ := 1 / (x * x)

theorem range_of_f : Set.range f = {y : ℝ | 0 < y} := 
by
  sorry

end range_of_f_l426_426424


namespace volume_of_cone_formed_from_half_sector_l426_426846

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * Mathlib.pi * r^2 * h

theorem volume_of_cone_formed_from_half_sector (r : ℝ) (hr : r = 6)
  (circumference : ℝ) (hcirc : circumference = 6 * Mathlib.pi)
  (base_radius : ℝ) (hbase_radius : 2 * Mathlib.pi * base_radius = circumference)
  (slant_height : ℝ) (hslant_height : slant_height = r)
  (height : ℝ) (hheight : height^2 + base_radius^2 = slant_height^2) :
  volume_of_cone base_radius height = 9 * Mathlib.pi * Real.sqrt 3 := sorry

end volume_of_cone_formed_from_half_sector_l426_426846


namespace sum_even_fourth_powers_lt_500_l426_426426

/-- 
  The sum of all even positive integers less than 500 that are 
  fourth powers of integers is 272.
--/
theorem sum_even_fourth_powers_lt_500 : 
  (∑ n in finset.filter (λ x, even x) (finset.filter (λ x, x < 500) 
    (finset.image (λ n, n^4) (finset.range 500))), id) = 272 := 
  sorry

end sum_even_fourth_powers_lt_500_l426_426426


namespace gaussian_solutions_count_l426_426364

noncomputable def solve_gaussian (x : ℝ) : ℕ :=
  if h : x^2 = 2 * (⌊x⌋ : ℝ) + 1 then 
    1 
  else
    0

theorem gaussian_solutions_count :
  ∀ x : ℝ, solve_gaussian x = 2 :=
sorry

end gaussian_solutions_count_l426_426364


namespace matrix_product_is_zero_l426_426502

variable {a b c d : ℝ}

def A : Matrix (Fin 4) (Fin 4) ℝ := ![
  ![0, d, -c, b],
  ![-d, 0, b, -a],
  ![c, -b, 0, a],
  ![-b, a, -a, 0]
]

def B : Matrix (Fin 4) (Fin 4) ℝ := ![
  ![a^2, ab, ac, ad],
  ![ab, b^2, bc, bd],
  ![ac, bc, c^2, cd],
  ![ad, bd, cd, d^2]
]

def zero_matrix : Matrix (Fin 4) (Fin 4) ℝ := ![
  ![0, 0, 0, 0],
  ![0, 0, 0, 0],
  ![0, 0, 0, 0],
  ![0, 0, 0, 0]
]

theorem matrix_product_is_zero : A ⬝ B = zero_matrix := by
  sorry

end matrix_product_is_zero_l426_426502


namespace even_function_condition_l426_426962

-- Definitions
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

-- The problem statement:
theorem even_function_condition (a : ℝ) :
    is_even_function (λ x : ℝ, x^2 + a*x + 1) ↔ a = 0 :=
sorry

end even_function_condition_l426_426962


namespace number_of_arrangements_eq_combinations_and_subtraction_l426_426498

theorem number_of_arrangements_eq_combinations_and_subtraction :
  (Finset.card (Finset.powersetLen 2 (Finset.range 6) ∪ Finset.powersetLen 3 (Finset.range 6) ∪ 
                Finset.powersetLen 4 (Finset.range 6) ∪ Finset.powersetLen 5 (Finset.range 6) ∪ 
                Finset.powersetLen 6 (Finset.range 6)) = 
                (Nat.choose 6 2 + Nat.choose 6 3 + Nat.choose 6 4 + Nat.choose 6 5 + Nat.choose 6 6) )
  ∧ (Finset.card (Finset.powersetLen 2 (Finset.range 6) ∪ Finset.powersetLen 3 (Finset.range 6) ∪ 
                  Finset.powersetLen 4 (Finset.range 6) ∪ Finset.powersetLen 5 (Finset.range 6) ∪ 
                  Finset.powersetLen 6 (Finset.range 6))  = (2^6 - 7)) :=
begin
  sorry,
end

end number_of_arrangements_eq_combinations_and_subtraction_l426_426498


namespace smallest_positive_solution_to_congruence_l426_426050

theorem smallest_positive_solution_to_congruence :
  ∃ x : ℕ, 5 * x ≡ 14 [MOD 33] ∧ x = 28 := 
by 
  sorry

end smallest_positive_solution_to_congruence_l426_426050


namespace sin_B_value_height_on_BC_l426_426661

-- Given a triangle ABC with sides a, b and angle A
variables {a b c : ℝ} {A B C : ℝ}

-- Condition definitions
def triangle_ABC_conditions : Prop :=
  a = 7 ∧ b = 8 ∧ A = π / 3

-- Theorem 1: Proving sin B given the conditions
theorem sin_B_value (h : triangle_ABC_conditions) : sin B = 4 * real.sqrt 3 / 7 :=
by sorry

-- Condition for obtuse triangle
def is_obtuse_triangle (h : triangle_ABC_conditions) : Prop :=
  B > π / 2

-- Theorem 2: Finding the height on side BC in an obtuse triangle
theorem height_on_BC (h1 : triangle_ABC_conditions) (h2 : is_obtuse_triangle h1) : 
  (sqrt (c^2 - (c * cos A)^2)) = 12 * real.sqrt 3 / 7 :=
by sorry

end sin_B_value_height_on_BC_l426_426661


namespace jana_walk_distance_l426_426666

theorem jana_walk_distance :
  (1 / 20 * 15 : ℝ) = 0.8 :=
by sorry

end jana_walk_distance_l426_426666


namespace maria_miles_after_second_stop_l426_426522

theorem maria_miles_after_second_stop (total_distance : ℕ)
    (h1 : total_distance = 360)
    (distance_first_stop : ℕ)
    (h2 : distance_first_stop = total_distance / 2)
    (remaining_distance_after_first_stop : ℕ)
    (h3 : remaining_distance_after_first_stop = total_distance - distance_first_stop)
    (distance_second_stop : ℕ)
    (h4 : distance_second_stop = remaining_distance_after_first_stop / 4)
    (remaining_distance_after_second_stop : ℕ)
    (h5 : remaining_distance_after_second_stop = remaining_distance_after_first_stop - distance_second_stop) :
    remaining_distance_after_second_stop = 135 := by
  sorry

end maria_miles_after_second_stop_l426_426522


namespace f_sum_l426_426552

theorem f_sum (f : ℕ → ℝ)
  (h_f : ∀ n : ℕ, 0 < n → f(n) = Real.log n^2 / Real.log 2002) :
  f 11 + f 13 + f 14 = 2 :=
sorry

end f_sum_l426_426552


namespace rainfall_difference_l426_426785

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end rainfall_difference_l426_426785


namespace hall_volume_l426_426437

theorem hall_volume (length breadth : ℝ) (h : ℝ)
  (h_length : length = 15) (h_breadth : breadth = 12)
  (h_area : 2 * (length * breadth) = 2 * (breadth * h) + 2 * (length * h)) :
  length * breadth * h = 8004 := 
by
  -- Proof not required
  sorry

end hall_volume_l426_426437


namespace ducks_and_geese_meeting_l426_426280

theorem ducks_and_geese_meeting:
  ∀ x : ℕ, ( ∀ ducks_speed : ℚ, ducks_speed = (1/7) ) → 
         ( ∀ geese_speed : ℚ, geese_speed = (1/9) ) → 
         (ducks_speed * x + geese_speed * x = 1) :=
by
  sorry

end ducks_and_geese_meeting_l426_426280


namespace max_pies_without_ingredients_l426_426669

theorem max_pies_without_ingredients 
  (total_pies : ℕ)
  (strawberry_ratio : ℚ)
  (banana_ratio : ℚ)
  (kiwifruit_ratio : ℚ)
  (coconut_ratio : ℚ)
  (h1 : total_pies = 48)
  (h2 : strawberry_ratio = 5 / 8)
  (h3 : banana_ratio = 3 / 4)
  (h4 : kiwifruit_ratio = 2 / 3)
  (h5 : coconut_ratio = 1 / 4) : 
  ∃ (n : ℕ), n = 12 := 
by 
  use 12
  sorry

end max_pies_without_ingredients_l426_426669


namespace function_characterization_l426_426160
noncomputable def f : ℕ → ℕ := sorry

theorem function_characterization (h : ∀ m n : ℕ, m^2 + f n ∣ m * f m + n) : 
  ∀ n : ℕ, f n = n :=
by
  intro n
  sorry

end function_characterization_l426_426160


namespace acute_triangle_inequality_l426_426342

theorem acute_triangle_inequality
  (a b c la lb lc : ℝ) 
  (h₀ : 0 < a) 
  (h₁ : 0 < b) 
  (h₂ : 0 < c)
  (h3 : 0 < la)
  (h4 : 0 < lb)
  (h5 : 0 < lc)
  (h6 : a^2 + b^2 > c^2)
  (h7 : a^2 + c^2 > b^2)
  (h8 : b^2 + c^2 > a^2)
  (h_la : la = √(b * c * (1 - a^2 / (b + c)^2)))
  (h_lb : lb = √(a * c * (1 - b^2 / (a + c)^2)))
  (h_lc : lc = √(a * b * (1 - c^2 / (a + b)^2))) : 
  1/la + 1/lb + 1/lc ≤ √2 * (1/a + 1/b + 1/c) := 
by 
  sorry

end acute_triangle_inequality_l426_426342


namespace range_of_k_l426_426267

noncomputable def quadratic_has_real_roots (k : ℝ): Prop :=
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≥ -1 :=
by
  sorry

end range_of_k_l426_426267


namespace fraction_of_fish_taken_out_on_day_5_l426_426667

-- Define the initial conditions
def initial_fish := 6
def doubling (n : ℕ) := 2^n

-- Define the fish count at each day
def fish_on_day_2 := initial_fish * doubling 1
def fish_on_day_3 := (initial_fish * doubling 2) - ((initial_fish * doubling 2) / 3)

def fish_on_day_4 := fish_on_day_3 * doubling 1
def fish_on_day_5 (f : ℚ) := fish_on_day_4 * doubling 1 - (fish_on_day_4 * doubling 1 * f)
def fish_on_day_6 (f : ℚ) := fish_on_day_5 f * doubling 1
def fish_on_day_7 (f : ℚ) := fish_on_day_6 f * doubling 1 + 15

-- Define the total fish on the seventh day
def total_fish_on_day_7 := 207

-- Formulate the theorem
theorem fraction_of_fish_taken_out_on_day_5 : 
  ∃ f : ℚ, fish_on_day_7 f = total_fish_on_day_7 → f = 1/4 :=
begin
  sorry,
end

end fraction_of_fish_taken_out_on_day_5_l426_426667


namespace parabola_equation_l426_426460

theorem parabola_equation {p : ℝ} (hp : p > 0) :
    (∃ M N : ℝ × ℝ,
        M.1 ^ 2 - M.2 ^ 2 = 6 ∧
        N.1 ^ 2 - N.2 ^ 2 = 6 ∧
        M.2 = N.2 = p / 2 ∧
        (M.1, M.2) ≠ (N.1, N.2) ∧
        let F := (0, p / 2) in
        (M.1 - F.1) ^ 2 + (M.2 - F.2) ^ 2 = (N.1 - F.1) ^ 2 + (N.2 - F.2) ^ 2 ∧
        (M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2) = 0) →
  ∃ c : ℝ, x^2 = 4 * c * y :=
by
  sorry

end parabola_equation_l426_426460


namespace roots_of_quadratic_l426_426766

theorem roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = -2 ∧ c = -6) :
  let Δ := b^2 - 4 * a * c in
  Δ > 0 :=
by
  obtain ⟨ha, hb, hc⟩ := h_eq
  have Δ_def : Δ = b^2 - 4 * a * c := rfl
  rw [ha, hb, hc] at Δ_def
  sorry

end roots_of_quadratic_l426_426766


namespace parity_of_T2021_T2022_T2023_l426_426470

def sequence (T : ℕ → ℕ) : Prop :=
  T 0 = 0 ∧ T 1 = 0 ∧ T 2 = 2 ∧ ∀ n ≥ 3, T n = T (n-1) + T (n-2) + T (n-3)

theorem parity_of_T2021_T2022_T2023 (T : ℕ → ℕ) (hT : sequence T) :
  (T 2021 % 2 = 0) ∧ (T 2022 % 2 = 0) ∧ (T 2023 % 2 = 0) :=
by 
  sorry

end parity_of_T2021_T2022_T2023_l426_426470


namespace rational_solution_exists_l426_426308

theorem rational_solution_exists (a b c : ℤ) (x₀ y₀ z₀ : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h₁ : a * x₀^2 + b * y₀^2 + c * z₀^2 = 0) (h₂ : x₀ ≠ 0 ∨ y₀ ≠ 0 ∨ z₀ ≠ 0) : 
  ∃ (x y z : ℚ), a * x^2 + b * y^2 + c * z^2 = 1 := 
sorry

end rational_solution_exists_l426_426308


namespace rationality_proof_l426_426434

def is_rational (x : ℚ) := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

def sqrt_4_rational : Prop := is_rational (real.sqrt 4)

def cube_root_0_5_irrational : Prop := ¬is_rational (real.cbrt 0.5)

def fourth_root_0_0625_rational : Prop := is_rational (real.root 4 0.0625)

theorem rationality_proof : sqrt_4_rational ∧ fourth_root_0_0625_rational ∧ cube_root_0_5_irrational :=
by
  sorry

end rationality_proof_l426_426434


namespace population_increase_rate_l426_426269

theorem population_increase_rate (p_increase : ℕ) (time_minutes : ℕ) (seconds_per_minute : ℕ) : 
  p_increase = 20 → time_minutes = 10 → seconds_per_minute = 60 → 
  (time_minutes * seconds_per_minute) / p_increase = 30 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  refl

end population_increase_rate_l426_426269


namespace connecting_M_midpoint_ON_perpendicular_OK_l426_426659

variable (K L M N O P Q : Type)
variable [trapezoid K L M N]
variable (h1 : LM = 1/2 * KN)
variable (h2 : KL = OL)

-- Assume additional variables indicating geometrical relationships
variable [is_midpoint P O N]
variable [is_midpoint Q K O]

-- Proposition to be proven
theorem connecting_M_midpoint_ON_perpendicular_OK 
  (h3 : ∀ P Q, MP ⊥ KO) : 
  MP ⊥ KO :=
by
  sorry

end connecting_M_midpoint_ON_perpendicular_OK_l426_426659


namespace factorize_x4_minus_81_l426_426936

theorem factorize_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factorize_x4_minus_81_l426_426936


namespace hot_dogs_served_today_l426_426467

theorem hot_dogs_served_today : 9 + 2 = 11 :=
by
  sorry

end hot_dogs_served_today_l426_426467


namespace trapezoid_triangle_area_l426_426658

-- Definitions of the problem, considering only the given conditions
variables {A B C D E : Type} [points_are_in_space : Geometry]
constants (AD DC BE : ℝ) (AD_perpendicular_DC BE_parallel_AD : Prop)
constants (AD_sector : AD = 3) (AB_sector : AB = 3) (DC_sector : DC = 6)
constants (E_on_DC : E ∈ [DC])

-- The theorem to prove
theorem trapezoid_triangle_area (h1 : AD_perpendicular_DC) (h2 : BE_parallel_AD) : 
  area_triangle B E C = 4.5 :=
sorry -- proof to be filled in by a theorem prover or manually

end trapezoid_triangle_area_l426_426658


namespace correct_equation_is_x2_sub_10x_add_9_l426_426909

-- Define the roots found by Student A and Student B
def roots_A := (8, 2)
def roots_B := (-9, -1)

-- Define the incorrect equation by student A from given roots
def equation_A (x : ℝ) := x^2 - 10 * x + 16

-- Define the incorrect equation by student B from given roots
def equation_B (x : ℝ) := x^2 + 10 * x + 9

-- Define the correct quadratic equation
def correct_quadratic_equation (x : ℝ) := x^2 - 10 * x + 9

-- Theorem stating that the correct quadratic equation balances the errors of both students
theorem correct_equation_is_x2_sub_10x_add_9 :
  ∃ (eq_correct : ℝ → ℝ), 
    eq_correct = correct_quadratic_equation :=
by
  -- proof will go here
  sorry

end correct_equation_is_x2_sub_10x_add_9_l426_426909


namespace paint_for_smaller_statues_l426_426629

variables (n_statues : ℕ) (h_large h_small : ℝ) (p_large : ℝ)

-- Conditions
def similar_statues := n_statues = 800 ∧ h_large = 8 ∧ h_small = 2 ∧ p_large = 2

-- Theorem statement (prove 100 pints of paint are required)
theorem paint_for_smaller_statues (H : similar_statues n_statues h_large h_small p_large) :
  let paint_needed := (p_large / (h_large^2))*(h_small^2) * n_statues 
  in paint_needed = 100 :=
by
  sorry

end paint_for_smaller_statues_l426_426629


namespace calculate_myOp_l426_426139

-- Define the operation
def myOp (x y : ℝ) : ℝ := x^3 - y

-- Given condition for h as a real number
variable (h : ℝ)

-- The theorem we need to prove
theorem calculate_myOp : myOp (2 * h) (myOp (2 * h) (2 * h)) = 2 * h := by
  sorry

end calculate_myOp_l426_426139


namespace factor_x4_minus_81_l426_426920

theorem factor_x4_minus_81 : ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intro x
  sorry

end factor_x4_minus_81_l426_426920


namespace total_pages_l426_426301

def Johnny_word_count : ℕ := 195
def Madeline_word_count : ℕ := 2 * Johnny_word_count
def Timothy_word_count : ℕ := Madeline_word_count + 50
def Samantha_word_count : ℕ := 3 * Madeline_word_count
def Ryan_word_count : ℕ := Johnny_word_count + 100
def Words_per_page : ℕ := 235

def pages_needed (words : ℕ) : ℕ :=
  if words % Words_per_page = 0 then words / Words_per_page else words / Words_per_page + 1

theorem total_pages :
  pages_needed Johnny_word_count +
  pages_needed Madeline_word_count +
  pages_needed Timothy_word_count +
  pages_needed Samantha_word_count +
  pages_needed Ryan_word_count = 12 :=
  by sorry

end total_pages_l426_426301


namespace cylinder_sphere_ratio_l426_426568

theorem cylinder_sphere_ratio (r R : ℝ) (h : 8 * r^2 = 4 * R^2) : R / r = Real.sqrt 2 :=
by
  sorry

end cylinder_sphere_ratio_l426_426568


namespace evaluation_expression_l426_426778

theorem evaluation_expression : 
  20 * (10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5))) = 192.6 := 
by
  sorry

end evaluation_expression_l426_426778


namespace monotonically_decreasing_interval_l426_426001

def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

theorem monotonically_decreasing_interval :
  ∀ x : ℝ, x < 1 → f' x < 0 :=
by
  -- placeholder for the explicit proof
  sorry

-- helper lemma to express derivative of f
lemma derivative_of_f (x : ℝ) : Deriv f x = 2 * x - 2 :=
by
  -- placeholder for the explicit proof
  sorry

-- demonstrating that the function is decreasing for values less than 1
lemma f_is_decreasing_before_1 :
  ∀ (x : ℝ), x < 1 → Deriv f x < 0 :=
by
  intros x hx
  rw derivative_of_f
  linarith

end monotonically_decreasing_interval_l426_426001


namespace distance_traveled_downstream_l426_426774

-- Define the conditions
variable (c w : ℝ)
def boat_speed_in_still_water := 26 -- km/hr
def current_speed := c             -- km/hr
def wind_speed := w                -- km/hr
def wind_efficiency := 0.10
def time_in_hours := 20 / 60       -- converting 20 minutes to hours

-- Define the total effective speed
def total_effective_speed := boat_speed_in_still_water + current_speed + wind_efficiency * wind_speed

-- The proof problem statement
theorem distance_traveled_downstream :
  (total_effective_speed * time_in_hours) = (26 / 3 + c / 3 + (0.10 * w) / 3) := by
  sorry

end distance_traveled_downstream_l426_426774


namespace h_at_2_l426_426309

def f (x : ℝ) : ℝ := 2 * x^2 + 2 * x + 5
noncomputable def g (x : ℝ) : ℝ := Real.exp (Real.sqrt (f x)) - 2
def h (x : ℝ) : ℝ := f (g x)

theorem h_at_2 : h 2 = 2 * Real.exp (2 * Real.sqrt 17) - 6 * Real.exp (Real.sqrt 17) + 13 := by
  sorry

end h_at_2_l426_426309


namespace probability_of_waiting_time_l426_426091

theorem probability_of_waiting_time (total_duration favorable_duration : ℕ) 
  (h_total : total_duration = 60) (h_favorable : favorable_duration = 15) : 
  favorable_duration / total_duration = 1 / 4 := 
by 
  -- Using the provided conditions
  rw [h_total, h_favorable]
  -- Simplifying the division
  norm_num

-- Adding sorry to skip the proof

end probability_of_waiting_time_l426_426091


namespace factorize_x4_minus_81_l426_426935

theorem factorize_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factorize_x4_minus_81_l426_426935


namespace train_pass_time_l426_426106

-- Definitions of the conditions
def length_of_train : ℝ := 400 -- in meters
def speed_of_train : ℝ := 120 -- in kmph
def speed_of_man : ℝ := 20 -- in kmph

-- Helper function to convert kmph to m/s
def kmph_to_mps (kmph : ℝ) : ℝ := kmph * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_of_train - speed_of_man)

-- Proof problem statement
theorem train_pass_time : length_of_train / relative_speed_mps ≈ 14.4 := sorry

end train_pass_time_l426_426106


namespace manuscript_typing_cost_l426_426066

-- Defining the conditions as per our problem
def first_time_typing_rate : ℕ := 5 -- $5 per page for first-time typing
def revision_rate : ℕ := 3 -- $3 per page per revision

def num_pages : ℕ := 100 -- total number of pages
def revised_once : ℕ := 30 -- number of pages revised once
def revised_twice : ℕ := 20 -- number of pages revised twice
def no_revision := num_pages - (revised_once + revised_twice) -- pages with no revisions

-- Defining the cost function to calculate the total cost of typing
noncomputable def total_typing_cost : ℕ :=
  (num_pages * first_time_typing_rate) + (revised_once * revision_rate) + (revised_twice * revision_rate * 2)

-- Lean theorem statement to prove the total cost is $710
theorem manuscript_typing_cost :
  total_typing_cost = 710 := by
  sorry

end manuscript_typing_cost_l426_426066


namespace thief_run_distance_before_overtake_l426_426104

-- Define the constants for the problem.
def d₀ : ℝ := 500  -- Initial distance in meters
def vₜₕᵢₑf : ℝ := 12 * 1000 / 3600  -- Speed of the thief in m/s
def vₚₒₗᵢcₑ : ℝ := 15 * 1000 / 3600  -- Speed of the policeman in m/s
def relative_speed : ℝ := vₚₒₗᵢcₑ - vₜₕᵢₑf  -- Relative speed in m/s

theorem thief_run_distance_before_overtake :
  let time_to_overtake := d₀ / relative_speed
  let dₜₕᵢₑf := vₜₕᵢₑf * time_to_overtake
  dₜₕᵢₑf = 2000 := by
  sorry

end thief_run_distance_before_overtake_l426_426104


namespace no_real_solutions_for_b_l426_426518

theorem no_real_solutions_for_b :
    ∀ (b : ℝ), ¬(b^2 - b + 1 = 0) := 
by
  intro b
  have : (b^2 - b + 1) = 0 → (b^2 - b + 1 ≥ 0), from sorry
  have discriminant_neg : (b^2 - b + 1 < 0), from sorry
  exact λ h, lt_irrefl 0 (lt_of_lt_of_le discriminant_neg (this h))

end no_real_solutions_for_b_l426_426518


namespace tangent_line_eq_l426_426396

theorem tangent_line_eq (m : ℝ) (hP : ∃ m, (1:ℝ)^2 + (1:ℝ)^2 - 4*(1:ℝ) + m*(1:ℝ) = 0) :
  ∃ (A B C : ℝ), A + B + C = 0 ∧
  (x y : ℝ) (H : x = 1 ∧ y = 1), A*x + B*y + C = A*(1:ℝ) + B*(1:ℝ) + C = 0,
  x - 2*y + 1 = 0 :=
sorry

end tangent_line_eq_l426_426396


namespace makennas_garden_larger_l426_426303

noncomputable def effective_area 
(outer_length : ℕ) (outer_width : ℕ) (pathway_width : ℕ) : ℕ :=
let effective_length := outer_length - 2 * pathway_width in
let effective_width := outer_width - 2 * pathway_width in
effective_length * effective_width

theorem makennas_garden_larger
(karl_length : ℕ) (karl_width : ℕ) (karl_pathway_width : ℕ)
(makenna_length : ℕ) (makenna_width : ℕ) (makenna_pathway_width : ℕ)
(h_karl : karl_length = 30) (h_karl_w : karl_width = 50) (h_karl_pw : karl_pathway_width = 2)
(h_makenna : makenna_length = 35) (h_makenna_w : makenna_width = 55) (h_makenna_pw : makenna_pathway_width = 3) :
  effective_area makenna_length makenna_width makenna_pathway_width = 
  effective_area karl_length karl_width karl_pathway_width + 225 := 
by
  sorry

end makennas_garden_larger_l426_426303


namespace cos_diff_identity_sin_sum_identity_l426_426710

theorem cos_diff_identity : cos (2 * Real.pi / 5) - cos (4 * Real.pi / 5) = Real.sqrt 5 / 2 := sorry

theorem sin_sum_identity : sin (2 * Real.pi / 7) + sin (4 * Real.pi / 7) - sin (6 * Real.pi / 7) = Real.sqrt 7 / 2 := sorry

end cos_diff_identity_sin_sum_identity_l426_426710


namespace revenue_maximization_l426_426820

/-- Lean statement for the given problem -/
theorem revenue_maximization (initial_price : ℝ) (initial_sales : ℝ) (delta_price : ℝ) (price_increase_effect : ℝ) 
  (price : ℝ) (revenue : ℝ) (optimal_price : ℝ) : 
  initial_price = 600 → 
  initial_sales = 300 → 
  delta_price = 7 → 
  price_increase_effect = 3 → 
  price = 600 + delta_price → 
  revenue = price * (initial_sales - (price_increase_effect / delta_price) * delta_price) → 
  optimal_price = 650 → 
  (initial_price * initial_sales < revenue) ∧ 
  price = optimal_price := 
begin
  intros h1 h2 h3 h4 h5 h6 h7,
  sorry
end

end revenue_maximization_l426_426820


namespace rectangular_prism_triangle_area_l426_426776

theorem rectangular_prism_triangle_area :
  ∃ m a n : ℕ, (m + a * Real.sqrt n = sum_of_areas_of_triangles 1 2 3) ∧ (m + n + a = 49) :=
sorry

noncomputable def sum_of_areas_of_triangles (x y z : ℕ) : ℝ :=
  -- This function will compute sum of areas of all triangles
  sorry

end rectangular_prism_triangle_area_l426_426776


namespace probability_of_winning_prize_l426_426487

theorem probability_of_winning_prize :
  (∃ (cards : Fin 3 → ℕ) (bottles : Fin 5 → Fin 3),
    let no_prize := 3^5 - ((choose 3 2) * 2^5 - 3)
    in (1 - (no_prize / 3^5)) = 50 / 81) :=
sorry

end probability_of_winning_prize_l426_426487


namespace no_tiling_possible_l426_426853

theorem no_tiling_possible (rect : ℕ × ℕ) 
  (initial_tiles : list (ℕ × ℕ)) 
  (tiled_initially : ∀ t ∈ initial_tiles, t = (1, 4) ∨ t = (2, 2)) 
  (one_2x2_replaced : ∃ (r : ℕ × ℕ), r ∈ initial_tiles ∧ r = (2, 2) ∧ (initial_tiles.erase r).append [(1, 4)] = new_tiles) 
  : ¬ ∀ t ∈ new_tiles, t = (1, 4) ∨ t = (2, 2) → tileable rect new_tiles :=
by
  sorry

end no_tiling_possible_l426_426853


namespace problem1_problem2_l426_426974

def setA (m : ℝ) : set ℝ := {x | x^2 - 2 * m * x + m^2 - 1 < 0}

-- (1) if m = 2, then A = (1, 3)
theorem problem1 : setA 2 = {x : ℝ | 1 < x ∧ x < 3} := 
sorry

-- (2) if 1 ∈ A and 3 ∉ A, then 0 < m < 2
theorem problem2 (m : ℝ) (h1 : 1 ∈ setA m) (h2 : 3 ∉ setA m) : 0 < m ∧ m < 2 := 
sorry

end problem1_problem2_l426_426974


namespace exists_y_average_abs_deviation_equals_half_l426_426176

theorem exists_y_average_abs_deviation_equals_half 
  (n : ℕ) (X : fin n → ℝ) (hX : ∀ i, 0 ≤ X i ∧ X i ≤ 1) : 
  ∃ y ∈ Icc 0 1, (1 / n) * ∑ i, |y - X i| = 1 / 2 :=
begin
  sorry
end

end exists_y_average_abs_deviation_equals_half_l426_426176


namespace graph_through_point_l426_426635

variable {α β : Type*}

noncomputable def has_inverse_function (f : α → β) : Prop :=
  ∃ g : β → α, ∀ x y, f (g y) = y ∧ g (f x) = x

theorem graph_through_point
  {f : ℝ → ℝ} 
  (hf_inv : has_inverse_function f)
  (h_passes_through : (2, 1) ∈ (λ x, (2 * x - f x)) '' univ) :
  (3, -4) ∈ (λ x, f⁻¹' x - 2 * x) '' univ :=
  sorry

end graph_through_point_l426_426635


namespace sum_ab_equals_five_l426_426987

-- Definitions for conditions
variables {a b : ℝ}

-- Assumption that establishes the solution set for the quadratic inequality
axiom quadratic_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ x^2 + b * x - a < 0

-- Statement to be proved
theorem sum_ab_equals_five : a + b = 5 :=
sorry

end sum_ab_equals_five_l426_426987


namespace abs_difference_evaluation_l426_426444

   -- Define the absolute value function
   def abs (x : ℝ) : ℝ := if x >= 0 then x else -x

   theorem abs_difference_evaluation : abs (9 - 4) - abs (12 - 14) = 3 := by
     sorry
   
end abs_difference_evaluation_l426_426444


namespace lottery_win_probability_l426_426012

theorem lottery_win_probability :
  let MegaBall_prob := 1 / 30
  let WinnerBall_prob := 1 / Nat.choose 50 5
  let BonusBall_prob := 1 / 15
  let Total_prob := MegaBall_prob * WinnerBall_prob * BonusBall_prob
  Total_prob = 1 / 953658000 :=
by
  sorry

end lottery_win_probability_l426_426012


namespace derivative_of_f_l426_426166

def f (x : ℝ) : ℝ := ln (2 * x ^ 2 - 4)

theorem derivative_of_f : (deriv f) = λ x, (2 * x / (x ^ 2 - 2)) := by
  sorry

end derivative_of_f_l426_426166


namespace graphs_differ_l426_426432

theorem graphs_differ (x : ℝ) :
  (∀ (y : ℝ), y = x + 3 ↔ y ≠ (x^2 - 1) / (x - 1) ∧
              y ≠ (x^2 - 1) / (x - 1) ∧
              ∀ (y : ℝ), y = (x^2 - 1) / (x - 1) ↔ ∀ (z : ℝ), y ≠ x + 3 ∧ y ≠ x + 1) := sorry

end graphs_differ_l426_426432


namespace no_solution_for_factorial_equation_l426_426619

theorem no_solution_for_factorial_equation :
  ∀ m : ℕ, 7! * 4! ≠ m! := by
  sorry

end no_solution_for_factorial_equation_l426_426619


namespace rainfall_difference_l426_426786

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end rainfall_difference_l426_426786


namespace sampled_students_within_interval_l426_426646

/-- Define the conditions for the student's problem --/
def student_count : ℕ := 1221
def sampled_students : ℕ := 37
def sampling_interval : ℕ := student_count / sampled_students
def interval_lower_bound : ℕ := 496
def interval_upper_bound : ℕ := 825
def interval_range : ℕ := interval_upper_bound - interval_lower_bound + 1

/-- State the goal within the above conditions --/
theorem sampled_students_within_interval :
  interval_range / sampling_interval = 10 :=
sorry

end sampled_students_within_interval_l426_426646


namespace find_cuboid_length_l426_426938

theorem find_cuboid_length
  (b : ℝ) (h : ℝ) (S : ℝ)
  (hb : b = 10) (hh : h = 12) (hS : S = 960) :
  ∃ l : ℝ, 2 * (l * b + b * h + h * l) = S ∧ l = 16.36 :=
by
  sorry

end find_cuboid_length_l426_426938


namespace minimum_odd_sided_polygon_divided_into_parallelograms_l426_426807

theorem minimum_odd_sided_polygon_divided_into_parallelograms (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ P : set (set (ℝ × ℝ)), is_polygon P ∧ (forall p ∈ P, is_parallelogram p) ∧ P ⊆ (odd_sided_polygon n)) :
  n ≥ 7 :=
by
  sorry

end minimum_odd_sided_polygon_divided_into_parallelograms_l426_426807


namespace count_silver_coins_l426_426087

theorem count_silver_coins 
  (gold_value : ℕ)
  (silver_value : ℕ)
  (num_gold_coins : ℕ)
  (cash : ℕ)
  (total_money : ℕ) :
  gold_value = 50 →
  silver_value = 25 →
  num_gold_coins = 3 →
  cash = 30 →
  total_money = 305 →
  ∃ S : ℕ, num_gold_coins * gold_value + S * silver_value + cash = total_money ∧ S = 5 := 
by
  sorry

end count_silver_coins_l426_426087


namespace sphere_radius_in_cube_of_side_2_l426_426090

theorem sphere_radius_in_cube_of_side_2 :
  ∀ (r : ℝ), (∃ (cube : ℝ) (touches_center : Bool), 
              cube = 2 ∧ touches_center = true) → r = 1 := 
by
  intros r ⟨cube, touches_center, hc, ht⟩
  sorry

end sphere_radius_in_cube_of_side_2_l426_426090


namespace is_monotonically_increasing_then_positive_l426_426324

variable {α : Type*} [OrderedRing α]
variable {a : ℕ → α}
variable {q : α}
variable {T : ℕ → α}

-- Definitions of the conditions
def is_geometric_sequence (a : ℕ → α) (q : α) := ∀ n, a (n + 1) = a n * q
def is_sum_of_first_n_terms (T : ℕ → α) (a : ℕ → α) := ∀ n, T n = ∑ i in range (n+1), a i
def monotonically_increasing (T : ℕ → α) := ∀ n, T (n + 1) ≥ T n

-- The final problem statement: If the sequence {T_n} is monotonically increasing, then T_n > 0
theorem is_monotonically_increasing_then_positive 
  (hq : q > 1)
  (a_is_geo : is_geometric_sequence a q)
  (T_is_sum : is_sum_of_first_n_terms T a)
  (hmono : monotonically_increasing T)
  : ∀ n, T n > 0 := 
sorry

end is_monotonically_increasing_then_positive_l426_426324


namespace find_a_and_b_l426_426121

theorem find_a_and_b :
  (∀ (x : ℤ), (x = 2) → (x ^ 2 + 2 * x = 8)) ∧
  (∃ (n : ℤ), n ^ 2 + 2 * n = -1 ∧ 2 * n + 1 = -1) :=
begin
  split,
  { intros x hx,
    dsimp,
    rw hx,
    norm_num,
  },
  { existsi (-1),
    split,
    { norm_num, },
    { norm_num, } }
end

end find_a_and_b_l426_426121


namespace equal_circumradii_l426_426365

open EuclideanGeometry

variables {A B C H : Point}

-- Assume that H is the orthocenter of triangle ABC
axiom orthocenter_def : Orthocenter A B C H

theorem equal_circumradii (h_orthocenter : Orthocenter A B C H) :
  (circumradius A B C = circumradius A H B) ∧
  (circumradius A B C = circumradius B H C) ∧
  (circumradius A B C = circumradius A H C) :=
by
  sorry

end equal_circumradii_l426_426365


namespace tickets_distribution_l426_426078

theorem tickets_distribution (people tickets : ℕ) (h_people : people = 9) (h_tickets : tickets = 24)
  (h_each_gets_at_least_one : ∀ (i : ℕ), i < people → (1 : ℕ) ≤ 1) :
  ∃ (count : ℕ), count ≥ 4 ∧ ∃ (f : ℕ → ℕ), (∀ i, i < people → 1 ≤ f i ∧ f i ≤ tickets) ∧ (∀ i < people, ∃ j < people, f i = f j) :=
  sorry

end tickets_distribution_l426_426078


namespace factor_x4_minus_81_l426_426918

theorem factor_x4_minus_81 : ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intro x
  sorry

end factor_x4_minus_81_l426_426918


namespace maximum_value_of_N_l426_426227

def polynomial (n : ℕ) (c : ℕ → ℤ) : (ℤ[X]) :=
  X^n + ∑ i in Finset.range n, (c i) * X^i

theorem maximum_value_of_N :
  ∀ c : ℕ → ℤ,
  (∀ i < 2020, c i ∈ {-1, 0, 1}) →
  (∀ x < 0, f.eval x ≠ 0) →
  let f := polynomial 2020 c in
  ∃ N ≤ 10, ∀ x > 0, f.eval x = 0 → multiplicity x f = N :=
sorry

end maximum_value_of_N_l426_426227


namespace fraction_spent_on_museum_ticket_l426_426668

theorem fraction_spent_on_museum_ticket (initial_money : ℝ) (sandwich_fraction : ℝ) (book_fraction : ℝ) (remaining_money : ℝ) (h1 : initial_money = 90) (h2 : sandwich_fraction = 1/5) (h3 : book_fraction = 1/2) (h4 : remaining_money = 12) : (initial_money - remaining_money) / initial_money - (sandwich_fraction * initial_money + book_fraction * initial_money) / initial_money = 1/6 :=
by
  sorry

end fraction_spent_on_museum_ticket_l426_426668


namespace determine_c_l426_426266

noncomputable def ab5c_decimal (a b c : ℕ) : ℕ :=
  729 * a + 81 * b + 45 + c

theorem determine_c (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : ∃ k : ℕ, ab5c_decimal a b c = k^2) :
  c = 0 ∨ c = 7 :=
by
  sorry

end determine_c_l426_426266


namespace option_b_option_c_option_d_l426_426622

variable {a b : ℝ}

theorem option_b (h : a + b = 1) (ha : a > 0) (hb : b > 0) : a^2 + b^2 ≥ 1/2 :=
sorry

theorem option_c (h : a + b = 1) (ha : a > 0) (hb : b > 0) : sqrt a + sqrt b ≤ sqrt 2 :=
sorry

theorem option_d (h : a + b = 1) (ha : a > 0) (hb : b > 0) : (1 / (a + 2 * b)) + (1 / (2 * a + b)) ≥ 4 / 3 :=
sorry

end option_b_option_c_option_d_l426_426622


namespace square_field_area_l426_426801

def square_area (side_length : ℝ) : ℝ :=
  side_length * side_length

theorem square_field_area :
  square_area 20 = 400 := by
  sorry

end square_field_area_l426_426801


namespace sum_of_primes_divergence_l426_426445

theorem sum_of_primes_divergence (N : ℝ) : ∃ (p: ℕ), Prime p ∧ (∑ q in (Finset.filter Prime (Finset.range (p + 1))), (1 : ℝ) / q) > N :=
sorry

end sum_of_primes_divergence_l426_426445


namespace proof_log_diff_l426_426150

noncomputable def log_diff : ℝ :=
  let a := Real.log 32 / Real.log 2
  let b := Real.log (1 / 8) / Real.log 2
  a - b

theorem proof_log_diff : log_diff = 8 := by
  let a := Real.log 32 / Real.log 2
  let b := Real.log (1 / 8) / Real.log 2
  have ha : a = 5 := by sorry
  have hb : b = -3 := by sorry
  calc
    log_diff = a - b                  := by rfl
         ... = 5 - (-3)               := by rw [ha, hb]
         ... = 5 + 3                  := by ring
         ... = 8                      := by ring

end proof_log_diff_l426_426150


namespace number_of_keepers_l426_426643

theorem number_of_keepers
  (h₁ : 50 * 2 = 100)
  (h₂ : 45 * 4 = 180)
  (h₃ : 8 * 4 = 32)
  (h₄ : 12 * 8 = 96)
  (h₅ : 6 * 8 = 48)
  (h₆ : 100 + 180 + 32 + 96 + 48 = 456)
  (h₇ : 50 + 45 + 8 + 12 + 6 = 121)
  (h₈ : ∀ K : ℕ, (2 * (K - 5) + 6 + 2 = 2 * K - 2))
  (h₉ : ∀ K : ℕ, 121 + K + 372 = 456 + (2 * K - 2)) :
  ∃ K : ℕ, K = 39 :=
by
  sorry

end number_of_keepers_l426_426643


namespace largest_divisor_problem_l426_426002

theorem largest_divisor_problem (N : ℕ) :
  (∃ k : ℕ, let m := Nat.gcd N (N - 1) in
            N + m = 10^k) ↔ N = 75 :=
by 
  sorry

end largest_divisor_problem_l426_426002


namespace prove_f_neg1_eq_0_l426_426588

def f : ℝ → ℝ := sorry

theorem prove_f_neg1_eq_0
  (h1 : ∀ x : ℝ, f(x + 2) = f(2 - x))
  (h2 : ∀ x : ℝ, f(1 - 2 * x) = -f(2 * x + 1))
  : f(-1) = 0 := sorry

end prove_f_neg1_eq_0_l426_426588


namespace snail_path_impossible_l426_426490

-- Define the problem statement
def infinite_grid := ℕ
def edge_length := (1 : ℝ)
def colors := (3 : ℕ)

/-- Prove that it is not necessarily true that there exists
a vertex in an infinite grid of equilateral triangles with edges
colored in three colors such that a snail can crawl along monochromatic edges
for 100 cm without traversing the same edge twice. -/
theorem snail_path_impossible (G : infinite_grid) (e_len : ℝ) (color_count : ℕ) (edge_colored : infinite_grid → infinite_grid → ℕ)
  (h1 : e_len = edge_length)
  (h2 : color_count = colors) :
  ¬ (∃ vertex, ∃ path : list (infinite_grid × infinite_grid), 
    (∀ (e : infinite_grid × infinite_grid), e ∈ path → edge_colored e.fst e.snd = edge_colored (list.head path).fst (list.head path).snd)
    ∧ (path.length * e_len = 100)
    ∧ (∀ (v : infinite_grid), v ∈ path → v ≠ vertex)) :=
sorry

end snail_path_impossible_l426_426490


namespace digit_in_600th_place_of_4_div_7_is_8_l426_426430

theorem digit_in_600th_place_of_4_div_7_is_8 :
  let seq := "571428" in
  let n := String.length seq in
  n = 6 → -- the sequence repeats every 6 digits
  (600 % n) = 0 → -- 600 modulo 6 is 0
  (seq.get (n - 1) = '8') := -- the last digit in the repeating sequence is '8'
begin
  sorry
end

end digit_in_600th_place_of_4_div_7_is_8_l426_426430


namespace units_digit_R_10001_l426_426897

noncomputable def a := 3 + Real.sqrt 5
noncomputable def b := 3 - Real.sqrt 5
noncomputable def R : ℕ → ℝ
| 0       := 1
| (n + 1) := 6 * R n - 4 * R (n - 1)

theorem units_digit_R_10001 : (R 10001) % 10 = 3 := by
  sorry

end units_digit_R_10001_l426_426897


namespace lindsey_squat_weight_l426_426327

theorem lindsey_squat_weight :
  let bandA := 7
  let bandB := 5
  let bandC := 3
  let leg_weight := 10
  let dumbbell := 15
  let total_weight := (2 * bandA) + (2 * bandB) + (2 * bandC) + (2 * leg_weight) + dumbbell
  total_weight = 65 :=
by
  sorry

end lindsey_squat_weight_l426_426327


namespace california_avg_sq_ft_per_person_l426_426760

noncomputable def average_square_feet_per_person 
  (population : ℕ) (area_sq_miles : ℕ) (sq_ft_per_sq_mile : ℕ) : ℕ :=
  (area_sq_miles * sq_ft_per_sq_mile) / population

theorem california_avg_sq_ft_per_person :
  average_square_feet_per_person 39500000 163696 (5280 ^ 2) ≈ 115543 :=
by
  sorry

end california_avg_sq_ft_per_person_l426_426760


namespace regular_octagon_angle_l426_426422

theorem regular_octagon_angle (n : ℕ) (h₁ : n = 8) :
  let interior_angle := 135
  let exterior_angle := 45
  interior_angle = 135 ∧ exterior_angle = 45 :=
by
  let interior_sum := 180 * (n - 2)
  have h₂ : interior_sum = 1080 := by
    rw [h₁, Nat.sub_self, Nat.mul_one]
  let int_angle := interior_sum / n
  have h₃ : int_angle = 135 := by
    rw [h₂, h₁]
    norm_num
  let ext_angle := 180 - int_angle
  have h₄ : ext_angle = 45 := by
    rw [h₃]
    norm_num
  split
  · exact h₃
  · exact h₄
  sorry -- Finalizing the proof.

end regular_octagon_angle_l426_426422


namespace find_m_l426_426177

open Set

def U : Set ℕ := {0, 1, 2, 3}
def A (m : ℤ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}
def complement_A (m : ℤ) : Set ℕ := {1, 2}

theorem find_m (m : ℤ) (hA : complement_A m = U \ A m) : m = -3 :=
by
  sorry

end find_m_l426_426177


namespace find_x_l426_426366

theorem find_x : 
  let avg := (3 * x + 8 + (7 * x - 3) + (4 * x + 5)) / 3 in
  avg = 5 * x - 6 → 
  x = -28 :=
by
  intros avg h
  sorry

end find_x_l426_426366


namespace fraction_problem_l426_426242

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end fraction_problem_l426_426242


namespace lambda_range_l426_426229

theorem lambda_range (
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (λ : ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = a n / (a n + 2))
  (h3 : ∀ n : ℕ, 0 < n → b (n + 1) = (n - 2 * λ) * (1 / a n + 1))
  (h4 : b 1 = -λ)
  (h5 : ∀ n : ℕ, 0 < n → b n ≤ b (n + 1))
) : λ < 2 / 3 :=
sorry

end lambda_range_l426_426229


namespace min_value_square_distance_l426_426197

theorem min_value_square_distance (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) : 
  ∃ c, (∀ x y : ℝ, x^2 + y^2 - 4*x + 2 = 0 → x^2 + (y - 2)^2 ≥ c) ∧ c = 2 :=
sorry

end min_value_square_distance_l426_426197


namespace zeros_of_f_l426_426948

noncomputable def f (z : ℂ) : ℂ := 1 - complex.exp z

theorem zeros_of_f :
  ∃ (z : ℤ → ℂ), (∀ n : ℤ, f (z n) = 0) ∧
   (∀ n : ℤ, deriv f (z n) ≠ 0) :=
by
  let z_n := λ n : ℤ, (2 * n * real.pi * complex.I)
  use z_n
  split
  { intro n
    show f (z_n n) = 0
    calc
      f (z_n n)
          = 1 - complex.exp (z_n n) : by rfl
      ... = 1 - 1 : by rw [complex.exp_eq_one_of_2npiI _]
      ... = 0 : by norm_num },
  { intro n
    show deriv f (z_n n) ≠ 0
    calc
      deriv f (z_n n)
          = -complex.exp (z_n n) : by apply complex.deriv_exp
      ... = -1 : by rw [complex.exp_eq_one_of_2npiI _]
      ... ≠ 0 : by norm_num
  }

end zeros_of_f_l426_426948


namespace brandon_skittles_loss_l426_426127

theorem brandon_skittles_loss (original final : ℕ) (H1 : original = 96) (H2 : final = 87) : original - final = 9 :=
by sorry

end brandon_skittles_loss_l426_426127


namespace bus_passengers_remaining_l426_426831

theorem bus_passengers_remaining (initial_passengers : ℕ := 22) 
                                 (boarding_alighting1 : (ℤ × ℤ) := (4, -8)) 
                                 (boarding_alighting2 : (ℤ × ℤ) := (6, -5)) : 
                                 (initial_passengers : ℤ) + 
                                 (boarding_alighting1.fst + boarding_alighting1.snd) + 
                                 (boarding_alighting2.fst + boarding_alighting2.snd) = 19 :=
by
  sorry

end bus_passengers_remaining_l426_426831


namespace isosceles_triangle_of_bisectors_and_angle_ratios_l426_426690

theorem isosceles_triangle_of_bisectors_and_angle_ratios
  {A B C M N : Type*} [Point A B C M N]
  (h1 : M ∈ line_segment B C)
  (h2 : N ∈ line_segment A B)
  (h3 : angle_bisector A M (line B C))
  (h4 : angle_bisector C N (line A B))
  (h5 : ∠BNM / ∠MNC = ∠BMN / ∠NMA) 
  : AB = BC := sorry

end isosceles_triangle_of_bisectors_and_angle_ratios_l426_426690


namespace find_fifth_day_income_l426_426830

-- Define the incomes for the first four days
def income_day1 := 45
def income_day2 := 50
def income_day3 := 60
def income_day4 := 65

-- Define the average income over five days
def average_income := 58

-- Expressing the question in terms of a function to determine the fifth day's income
theorem find_fifth_day_income : 
  ∃ (income_day5 : ℕ), 
    (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = average_income 
    ∧ income_day5 = 70 :=
sorry

end find_fifth_day_income_l426_426830


namespace probability_sum_16_is_1_over_64_l426_426374

-- Define the problem setup
def octahedral_faces : Finset ℕ := Finset.range 9 \ {0} -- Faces labeled 1 through 8

-- Define the event for the sum of 16
def event_sum_16 : Finset (ℕ × ℕ) :=
  Finset.filter (λ p, p.1 + p.2 = 16) (octahedral_faces.product octahedral_faces)

-- Total outcomes with two octahedral dice
def total_outcomes : ℕ := octahedral_faces.card * octahedral_faces.card

-- Probability of rolling a sum of 16
def probability_sum_16 : ℚ := (event_sum_16.card : ℚ) / total_outcomes

theorem probability_sum_16_is_1_over_64 :
  probability_sum_16 = 1 / 64 := by
  sorry

end probability_sum_16_is_1_over_64_l426_426374


namespace pies_baked_l426_426913

theorem pies_baked (days : ℕ) (eddie_rate : ℕ) (sister_rate : ℕ) (mother_rate : ℕ)
  (H1 : eddie_rate = 3) (H2 : sister_rate = 6) (H3 : mother_rate = 8) (days_eq : days = 7) :
  eddie_rate * days + sister_rate * days + mother_rate * days = 119 :=
by
  sorry

end pies_baked_l426_426913


namespace exists_nat_x_y_x2_minus_y2_eq_1993_not_exists_nat_x_y_x3_minus_y3_eq_1993_not_exists_nat_x_y_x4_minus_y4_eq_1993_l426_426985

-- Definition of natural numbers
open Nat

-- Lean statement for Proof Problem a
theorem exists_nat_x_y_x2_minus_y2_eq_1993 (h : Prime 1993) :
  ∃ x y : ℕ, x^2 - y^2 = 1993 :=
sorry

-- Lean statement for Proof Problem b
theorem not_exists_nat_x_y_x3_minus_y3_eq_1993 (h : Prime 1993) :
  ¬ ∃ x y : ℕ, x^3 - y^3 = 1993 :=
sorry

-- Lean statement for Proof Problem c
theorem not_exists_nat_x_y_x4_minus_y4_eq_1993 (h : Prime 1993) :
  ¬ ∃ x y : ℕ, x^4 - y^4 = 1993 :=
sorry

end exists_nat_x_y_x2_minus_y2_eq_1993_not_exists_nat_x_y_x3_minus_y3_eq_1993_not_exists_nat_x_y_x4_minus_y4_eq_1993_l426_426985


namespace inequality_solution_l426_426556

noncomputable def inequality_proof (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 2) : Prop :=
  (1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a)) ≥ (27 / 13)

theorem inequality_solution (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 2) : 
  inequality_proof a b c h_positive h_sum :=
sorry

end inequality_solution_l426_426556


namespace find_ab_cde_l426_426680

def P (x : ℝ) : ℝ := x^2 - 5 * x - 4

theorem find_ab_cde (a b c d e : ℕ) :
  (∀ x, 3 ≤ x ∧ x ≤ 12 → ⌊real.sqrt (P x)⌋ = real.sqrt (P (⌊x⌋))) →
  a + b + c + d + e = 9 :=
sorry

end find_ab_cde_l426_426680


namespace interest_rate_calculation_l426_426472

theorem interest_rate_calculation (P : ℝ) (r : ℝ) (h1 : P * (1 + r / 100)^3 = 800) (h2 : P * (1 + r / 100)^4 = 820) :
  r = 2.5 := 
  sorry

end interest_rate_calculation_l426_426472


namespace sum_of_rearranged_digits_not_1999_nines_original_number_divisible_by_10_l426_426074

-- Part (a)
theorem sum_of_rearranged_digits_not_1999_nines 
  (n : ℕ) (a : Fin n → ℕ) (b : Fin n → ℕ)
  (h: ∀ i, a i + b i ≤ 9) :
  (Σ i, (a i + b i) * 10^i) ≠ 999 * (10 ^ 1998) + 999 :=
sorry

-- Part (b)
theorem original_number_divisible_by_10
  (n : ℕ) (a : Fin n → ℕ) (b : Fin n → ℕ)
  (h : Σ i, (a i + b i) * 10^i = 1010) :
  (Σ i, a i * 10^i) % 10 = 0 :=
sorry

end sum_of_rearranged_digits_not_1999_nines_original_number_divisible_by_10_l426_426074


namespace bob_questions_three_hours_l426_426123

theorem bob_questions_three_hours : 
  let first_hour := 13
  let second_hour := first_hour * 2
  let third_hour := second_hour * 2
  first_hour + second_hour + third_hour = 91 :=
by
  sorry

end bob_questions_three_hours_l426_426123


namespace max_team_members_l426_426872

/-- 
At the All-Union Olympiad, students from 8th, 9th, and 10th grades participate. 
The team from Leningrad includes k winners of the city Olympiad and the winners 
of last year's All-Union Olympiad. This theorem proves the maximum possible 
number of team members is 5k.
-/
theorem max_team_members (k : ℕ) : 
  let eighth_grade_winners := k in
  let ninth_grade_winners := k in
  let tenth_grade_winners := k in
  let previous_ninth_grade_winners := k in
  let previous_tenth_grade_winners := k in
  
  let max_possible_members := 
    eighth_grade_winners + 
    (ninth_grade_winners + previous_ninth_grade_winners) + 
    (tenth_grade_winners + previous_tenth_grade_winners)
  in 
  max_possible_members = 5 * k :=
by
  sorry

end max_team_members_l426_426872


namespace largest_divisor_power_of_ten_l426_426008

theorem largest_divisor_power_of_ten (N : ℕ) (m : ℕ) (k : ℕ) 
  (h1 : m ∣ N)
  (h2 : m < N)
  (h3 : N + m = 10^k) : N = 75 := sorry

end largest_divisor_power_of_ten_l426_426008


namespace largest_divisor_power_of_ten_l426_426010

theorem largest_divisor_power_of_ten (N : ℕ) (m : ℕ) (k : ℕ) 
  (h1 : m ∣ N)
  (h2 : m < N)
  (h3 : N + m = 10^k) : N = 75 := sorry

end largest_divisor_power_of_ten_l426_426010


namespace max_value_l426_426048

noncomputable def f (x y : ℝ) : ℝ := 8 * x ^ 2 + 9 * x * y + 18 * y ^ 2 + 2 * x + 3 * y
noncomputable def g (x y : ℝ) : Prop := 4 * x ^ 2 + 9 * y ^ 2 = 8

theorem max_value : ∃ x y : ℝ, g x y ∧ f x y = 26 :=
by
  sorry

end max_value_l426_426048


namespace prob_B_second_shot_prob_A_ith_shot_expected_A_shots_l426_426339

-- Define probabilities and parameters
def pA : ℝ := 0.6
def pB : ℝ := 0.8

-- Define the probability of selecting the first shooter as 0.5 for each player
def first_shot_prob : ℝ := 0.5

-- Proof that the probability that player B takes the second shot is 0.6
theorem prob_B_second_shot : (first_shot_prob * (1 - pA) + first_shot_prob * pB) = 0.6 := 
by sorry

-- Define the recursive probability for player A taking the nth shot
noncomputable def P (n : ℕ) : ℝ :=
if n = 0 then 0.5
else 0.4 * (P (n - 1)) + 0.2

-- Proof that the probability that player A takes the i-th shot is given by the formula
theorem prob_A_ith_shot (i : ℕ) : P i = (1 / 3) + (1 / 6) * ((2 / 5) ^ (i - 1)) :=
by sorry

-- Define the expected number of times player A shoots in the first n shots based on provided P formula
noncomputable def E_Y (n : ℕ) : ℝ :=
(sum i in finset.range n, P i)

-- Proof that the expected number of times player A shoots in the first n shots is given by the formula
theorem expected_A_shots (n : ℕ) : E_Y n = (5 / 18) * (1 - (2 / 5) ^ n) + (n / 3) :=
by sorry

end prob_B_second_shot_prob_A_ith_shot_expected_A_shots_l426_426339


namespace problem1_problem2_problem3_l426_426219

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem1 (a b c : ℝ) (h1 : f a b c 1 = -a / 2) (h2 : 3 * a > 2 * c)
  (h3 : 2 * c > 2 * b) (h4 : a > 0) : -3 < b / a ∧ b / a < -3 / 4 := sorry

theorem problem2 (a b c : ℝ) (h1 : f a b c 1 = -a / 2) (h2 : 3 * a > 2 * c)
  (h3 : 2 * c > 2 * b) (h4 : a > 0) : ∃ x ∈ (0, 2), f a b c x = 0 := sorry

theorem problem3 (a b c x1 x2 : ℝ) (h1 : f a b c 1 = -a / 2) (h2 : 3 * a > 2 * c)
  (h3 : 2 * c > 2 * b) (h4 : a > 0) (h5 : f a b c x1 = 0)
  (h6 : f a b c x2 = 0) (h7 : x1 ≠ x2) :
  sqrt 2 ≤ abs (x1 - x2) ∧ abs (x1 - x2) < sqrt 57 / 4 := sorry

end problem1_problem2_problem3_l426_426219


namespace evaluate_expression_l426_426915

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), (k^2 : ℚ)

theorem evaluate_expression :
  let sum1 := ∑ k in Finset.range 10, (k^2 : ℚ) / 2
  let sum2 := ∑ k in Finset.range 50, 1.5
  (sum1 * sum2 = 14437.5) :=
by
  sorry

end evaluate_expression_l426_426915


namespace octahedron_tetrahedron_volume_ratio_l426_426466

theorem octahedron_tetrahedron_volume_ratio (a : ℝ) :
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3
  V_o / V_t = 1 :=
by 
  -- Definitions from conditions
  let V_t := (a^3 * Real.sqrt 2) / 12
  let s := (a * Real.sqrt 2) / 2
  let V_o := (s^3 * Real.sqrt 2) / 3

  -- Proof omitted
  -- Proof goes here
  sorry

end octahedron_tetrahedron_volume_ratio_l426_426466


namespace find_n_l426_426168

theorem find_n (n : ℤ) (h : 0 ≤ n ∧ n ≤ 9) : n ≡ -5643 [MOD 10] ↔ n = 7 :=
by
  sorry

end find_n_l426_426168


namespace checkerboard_squares_count_l426_426450

-- Definitions of checkerboard and necessary conditions
def is_alternating_black_white (i j : ℕ) : bool :=
  (i + j) % 2 == 0

def contains_at_least_8_black_squares (n : ℕ) : bool :=
  ∃ i1 j1, 
    (∀ i2 j2, i1 ≤ i2 ∧ i2 < i1 + n ∧ j1 ≤ j2 ∧ j2 < j1 + n →
    is_alternating_black_white i2 j2) &&
    (∑ i2 in Finset.range n, ∑ j2 in Finset.range n, dite (is_alternating_black_white (i1 + i2) (j1 + j2)) (λ _, 1) (λ _, 0) ≥ 8)

def count_valid_squares (size board_size : ℕ) : ℕ :=
  if h : 1 ≤ size ∧ size ≤ board_size then
    ∑ i in Finset.range (board_size - size + 1), ∑ j in Finset.range (board_size - size + 1),
      if contains_at_least_8_black_squares size then 1 else 0
  else 0

theorem checkerboard_squares_count :
  let board_size := 10 in 
  (count_valid_squares 4 board_size + 
   count_valid_squares 5 board_size + 
   count_valid_squares 6 board_size + 
   count_valid_squares 7 board_size + 
   count_valid_squares 8 board_size + 
   count_valid_squares 9 board_size + 
   count_valid_squares 10 board_size) = 115 := sorry

end checkerboard_squares_count_l426_426450


namespace fraction_problem_l426_426251

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end fraction_problem_l426_426251


namespace rainfall_difference_l426_426781

theorem rainfall_difference :
  let day1 := 26
  let day2 := 34
  let day3 := day2 - 12
  let total_rainfall := day1 + day2 + day3
  let average_rainfall := 140
  (average_rainfall - total_rainfall = 58) :=
by
  sorry

end rainfall_difference_l426_426781


namespace range_of_a_l426_426605

noncomputable def f (a x : ℝ) : ℝ := x^3 - (3 / 2) * a * x^2

noncomputable def g (a x : ℝ) : ℝ := f a x + a

theorem range_of_a :
  {a : ℝ | ∃! x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ g a x₃ = 0 } = 
  {a : ℝ | a < - sqrt 2 ∨ a > sqrt 2} :=
sorry

end range_of_a_l426_426605


namespace tree_current_height_l426_426673

theorem tree_current_height 
  (growth_rate_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_height_after_4_months : ℕ) 
  (growth_rate_per_week_eq : growth_rate_per_week = 2)
  (weeks_per_month_eq : weeks_per_month = 4)
  (total_height_after_4_months_eq : total_height_after_4_months = 42) : 
  (∃ (current_height : ℕ), current_height = 10) :=
by
  sorry

end tree_current_height_l426_426673


namespace find_a_div_b_l426_426026

variable (a b : ℝ)
noncomputable def curve (x : ℝ) := x * Real.exp x
noncomputable def tangent_slope_at (x : ℝ) := Deriv (curve x)
noncomputable def point := (1, Real.exp 1)

-- Condition statements derived from the problem
def is_tangent_perpendicular (a b : ℝ) : Prop :=
  let k := tangent_slope_at 1
  k = 2 * Real.exp 1 ∧
  -a / b = -1 / k

theorem find_a_div_b (a b : ℝ) :
  is_tangent_perpendicular a b → a / b = 1 / (2 * Real.exp 1) :=
sorry

end find_a_div_b_l426_426026


namespace propositions_false_l426_426215

-- Definitions for lines and planes 
variable (Line Plane : Type)
variable intersects : Line → Line → Prop
variable in_plane : Line → Plane → Prop
variable intersection : Plane → Plane → Line
variable parallel : Line → Line → Prop

-- Conditions for skew lines
variable (a b : Line)
variable (α β : Plane)
variable (skew : (parallel a b → False) ∧ (¬ intersects a b))

-- Conditions places for propositions
variable (c : Line)
variable (prop1 : in_plane a α)
variable (prop2 : in_plane b β)
variable (prop3 : c = intersection α β)

-- Define propositions
def PropositionI : Prop := (intersects c a ∨ intersects c b) → (¬ intersects c a ∨ ¬ intersects c b)
def PropositionII : Prop := ∃ (L : Set Line), (∀ x y ∈ L, (¬ parallel x y ∧ ¬ intersects x y))

-- Proof stating both propositions are false
theorem propositions_false : ¬ PropositionI ∧ ¬ PropositionII :=
by 
  sorry

end propositions_false_l426_426215


namespace binomial_expansion_coeff_l426_426655

theorem binomial_expansion_coeff:
  let t := (x + (2/x))^4 in
  (binomial x (2/x) 4 2).coeff = 8 :=
by
  sorry

end binomial_expansion_coeff_l426_426655


namespace perpendicular_lines_l426_426410

theorem perpendicular_lines (a : ℝ) :
  let l1 := λ x y : ℝ, a * x + (1 + a) * y = 3,
      l2 := λ x y : ℝ, (a + 1) * x + (3 - 2a) * y = 2 in
  (exists (x₁ y₁) (x₂ y₂ : ℝ),
    l1 x₁ y₁ ∧ l2 x₂ y₂ ∧ 
    (a ≠ -1 ∧ a ≠ 3/2 → (-a / (1 + a)) * (-(a + 1) / (3 - 2a)) = -1)) →
  (a = -1 ∨ a = 3) :=
by sorry

end perpendicular_lines_l426_426410


namespace rectangle_ratios_l426_426102

-- Definitions based on given conditions
def Square (side : ℝ) := side > 0

structure Midpoint (coord1 coord2 : ℝ × ℝ) :=
  (x : ℝ)
  (y : ℝ)
  (is_midpoint : x = (coord1.1 + coord2.1) / 2 ∧ y = (coord1.2 + coord2.2) / 2)

-- Points coordinates based on given condition:
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (2, 0)
def D : ℝ × ℝ := (0, 0)

-- Midpoints E and F
def E : ℝ × ℝ := (B.1 + C.1) / 2, (B.2 + C.2) / 2
def F : ℝ × ℝ := (A.1 + D.1) / 2, (A.2 + D.2) / 2

-- Line AG is perpendicular to BF
def Perpendicular (p1 p2 p3 p4 : ℝ × ℝ) :=
  (p2.2 - p1.2) * (p4.2 - p3.2) + (p2.1 - p1.1) * (p4.1 - p3.1) = 0 

theorem rectangle_ratios (side : ℝ) (hside : Square side)
  (E := E (B, C)) 
  (F := F (A, D)) 
  (hE : E.is_midpoint)
  (hF : F.is_midpoint)
  (hPerpendicular : Perpendicular A G B F) 
  (hReassemble : true) : 
  let XY := sqrt((B.1 - F.1) ^ 2 + (B.2 - F.2) ^ 2)
  let YZ := sqrt((A.1 - G.1) ^ 2 + (A.2 - G.2) ^ 2)
  in XY / YZ = 5 := 
sorry

end rectangle_ratios_l426_426102


namespace factor_x4_minus_81_l426_426930

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
sorry

end factor_x4_minus_81_l426_426930


namespace find_y_l426_426172

theorem find_y (y : ℝ) (h_pos : y > 0) (h_eq : 2 * y * floor y = 162) : y = 9 :=
by
  sorry

end find_y_l426_426172


namespace marbles_solution_l426_426674

open Nat

def marbles_problem : Prop :=
  ∃ J_k J_j : Nat, (J_k = 3) ∧ (J_k = J_j - 4) ∧ (J_k + J_j = 10)

theorem marbles_solution : marbles_problem := by
  sorry

end marbles_solution_l426_426674


namespace solve_for_y_l426_426723

theorem solve_for_y (y : ℝ) : (5 ^ y + 15 = 4 * 5 ^ y - 45) ↔ (y = Real.log 20 / Real.log 5) :=
by
  sorry

end solve_for_y_l426_426723


namespace elijah_num_decks_l426_426525

theorem elijah_num_decks (total_cards deck_size : ℕ) (h1 : total_cards = 312) (h2 : deck_size = 52) :
  total_cards / deck_size = 6 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) (Nat.mul_eq_right_iff.mpr (or.inl rfl))

end elijah_num_decks_l426_426525


namespace find_XY_squared_l426_426682

variables {A B C T X Y : Type}

-- Conditions
variables (is_acute_scalene_triangle : ∀ A B C : Type, Prop) -- Assume scalene and acute properties
variable  (circumcircle : ∀ A B C : Type, Type) -- Circumcircle of the triangle
variable  (tangent_at : ∀ (ω : Type) B C, Type) -- Tangents at B and C
variables (BT CT : ℝ)
variables (BC : ℝ)
variables (projections : ∀ T (line : Type), Type)
variables (TX TY XY : ℝ)

-- Given conditions
axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom final_equation : TX^2 + TY^2 + XY^2 = 1552

-- Goal
theorem find_XY_squared : XY^2 = 884 := by
  sorry

end find_XY_squared_l426_426682


namespace horse_revolutions_l426_426456

theorem horse_revolutions (r1 r2  : ℝ) (rev1 rev2 : ℕ)
  (h1 : r1 = 30) (h2 : rev1 = 20) (h3 : r2 = 10) : rev2 = 60 :=
by
  sorry

end horse_revolutions_l426_426456


namespace triangle_area_OQK_l426_426657

noncomputable theory

-- Definitions for the equivalence proof problem
variables {A B C Q K M O : Type} 
variables [triangle_geom A B C]
variables (angle_bisector : bisector A B C 4)
variables (A60 : A.angle = 60)
variables (C90 : C.angle = 90)

-- Definition of points and lines relevant to the problem
variables (Q_center : center_circle_on_bisector B C Q)
variables (O_incircle : incircle_center A B C O)
variables (K_touch : touches_line Q_circle K A C)
variables (M_touch : touches_line Q_circle M A B)

-- Main theorem statement
theorem triangle_area_OQK 
  (h : triangle_geom A B C) 
  (Q_center : center_circle_on_bisector B C Q) 
  (O_incircle : incircle_center A B C O)
  (K_touch : touches_line Q_circle K A C)
  (M_touch : touches_line Q_circle M A B) : 
  area_triangle O Q K = 4.5 :=
sorry

end triangle_area_OQK_l426_426657


namespace binomial_coefficient_divisors_l426_426571

theorem binomial_coefficient_divisors {k n : ℕ} (hk : k ≥ 2) (hn : n ≥ k) :
  let S := finset.range(k).image(λ i, n - i) in
  ∃ (T : finset ℕ), T ⊆ S ∧ T.card = k - 1 ∧ ∀ t ∈ T, t ∣ nat.choose n k :=
sorry

end binomial_coefficient_divisors_l426_426571


namespace count_valid_sequences_eq_128_l426_426689

def is_valid_sequence (seq : List ℕ) : Prop :=
(seq.length = 8 ∧
 seq.nodup ∧
 ((∀ i, 2 ≤ i → i < 8 → (seq[i] + 2 ∈ seq.take i ∨ seq[i] - 2 ∈ seq.take i))
  ∧ (2 ∣ seq[0])))

theorem count_valid_sequences_eq_128 : 
  (finset.univ.filter is_valid_sequence).card = 128 :=
sorry

end count_valid_sequences_eq_128_l426_426689


namespace proof_problem_l426_426143

-- Definitions of conditions
def condition1 (x y : ℝ) : Prop := y = -2 * x + 3
def condition2 (regression_line : ℝ → ℝ) (mean : ℝ × ℝ) : Prop := regression_line mean.1 = mean.2
def condition3 (A B : Type) (k : ℝ) : Prop := true  -- Placeholder since the precise definition is informal
def condition4 (residual_sum_squares : ℝ) (correlation_coefficient : ℝ) : Prop := residual_sum_squares = 0 → correlation_coefficient = 1
-- count of correct statements
def number_of_correct_statements (cond1 cond2 cond3 cond4 : Prop) : ℕ :=
  (if cond1 then 1 else 0) + (if cond2 then 1 else 0) + (if cond3 then 1 else 0) + (if cond4 then 1 else 0)

-- The main theorem
theorem proof_problem : number_of_correct_statements (¬ condition1 0 0) (condition2 (λ x, x) (0, 0)) (condition3 ℝ ℝ 0) (condition4 0 1) = 3 := by
  sorry

end proof_problem_l426_426143


namespace increased_speed_l426_426357

theorem increased_speed (S : ℝ) : 
  (∀ (usual_speed : ℝ) (usual_time : ℝ) (distance : ℝ), 
    usual_speed = 20 ∧ distance = 100 ∧ usual_speed * usual_time = distance ∧ S * (usual_time - 1) = distance) → 
  S = 25 :=
by
  intros h1
  sorry

end increased_speed_l426_426357


namespace ordered_pair_a_c_l426_426762

theorem ordered_pair_a_c (a c : ℝ) (h_quad: ∀ x : ℝ, a * x^2 + 16 * x + c = 0)
    (h_sum: a + c = 25) (h_ineq: a < c) : (a = 3 ∧ c = 22) :=
by
  -- The proof is omitted
  sorry

end ordered_pair_a_c_l426_426762


namespace product_cubed_roots_floor_l426_426503

noncomputable def integer_floor (x : ℝ) : ℤ := int.floor x

theorem product_cubed_roots_floor : 
  (∏ k in (finset.range 2024).filter (λ k, odd (k+1) && k < 2023), integer_floor (real.cbrt (k + 1))) /
  (∏ k in (finset.range 2024).filter (λ k, even (k+1)), integer_floor (real.cbrt (k + 1))) = 1/4 := 
by
  sorry

end product_cubed_roots_floor_l426_426503


namespace furniture_purchase_price_l426_426092

variable (a : ℝ)

theorem furniture_purchase_price :
  let marked_price := 132
  let discount := 0.1
  let sale_price := marked_price * (1 - discount)
  let profit_relative_to_purchase := 0.1 * a
  (sale_price - a = profit_relative_to_purchase) -> a = 108 := by
  let marked_price := 132
  let discount := 0.1
  let sale_price := marked_price * (1 - discount)
  let profit_relative_to_purchase := 0.1 * a
  have h1 : sale_price = 118.8 := 
    by sorry -- substitution of values
  have h2 : sale_price - a = profit_relative_to_purchase := 
    by sorry -- given condition
  have h3 : 118.8 - a = 0.1 * a := 
    by rwa [h1] at h2
  have h4 : 118.8 = 1.1 * a := 
    by sorry -- re-arranging terms
  have h5 : a = 108 := 
    by sorry -- solving for a
  exact h5

end furniture_purchase_price_l426_426092


namespace sum_reciprocal_roots_l426_426550

theorem sum_reciprocal_roots : 
  ∀ (r1 r2 : ℝ), 
    r1 + r2 = 17 → 
    r1 * r2 = 8 → 
    (1 / r1 + 1 / r2 = 17 / 8) :=
by
  intros r1 r2 h_sum h_prod
  have h1 : 1 / r1 + 1 / r2 = (r1 + r2) / (r1 * r2), by sorry
  rw [h_sum, h_prod] at h1
  exact h1

end sum_reciprocal_roots_l426_426550


namespace angle_sum_eq_180_l426_426840

variables {α : Type*}
variables {A B C D X : α}
variables [convex_quad A B C D]
variables (AB CD BC DA : ℝ)
variables (angle_XAB angle_XCD angle_XBC angle_XDA : ℝ)
variable (angle_BXA : ℝ)
variable (angle_DXC : ℝ)

noncomputable def convex_quad (A B C D : α) : Prop := 
  (* Definition of convex quadrilateral changes depending on the structure of points and angles in Type α *)

axiom condition1 (A B C D : α) (h : convex_quad A B C D) : AB * CD = BC * DA
axiom condition2 (X_inside : ∃ p, p ∈ A ∧ p ∈ B ∧ p ∈ C ∧ p ∈ D ∧ (X ≠ p))
axiom condition3 : angle_XAB = angle_XCD
axiom condition4 : angle_XBC = angle_XDA

theorem angle_sum_eq_180 :
  angle_BXA + angle_DXC = 180 := 
sorry

end angle_sum_eq_180_l426_426840


namespace Felicity_used_23_gallons_l426_426533

variable (A Felicity : ℕ)
variable (h1 : Felicity = 4 * A - 5)
variable (h2 : A + Felicity = 30)

theorem Felicity_used_23_gallons : Felicity = 23 := by
  -- Proof steps would go here
  sorry

end Felicity_used_23_gallons_l426_426533


namespace percent_of_dollar_in_pocket_l426_426413

theorem percent_of_dollar_in_pocket :
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  (nickel + 2 * dime + quarter + half_dollar = 100) →
  (100 / 100 * 100 = 100) :=
by
  intros
  sorry

end percent_of_dollar_in_pocket_l426_426413


namespace roots_of_quadratic_equation_are_real_and_distinct_l426_426772

def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic_equation_are_real_and_distinct :
  quadratic_discriminant 1 (-2) (-6) > 0 :=
by
  norm_num
  sorry

end roots_of_quadratic_equation_are_real_and_distinct_l426_426772


namespace angle_XOY_is_120_l426_426481

-- Conditions and setup
variables (A B C D O X Y : Type) [triangle A B C]
variables [circumcircle_center A B C O]
variables [median A D C]
variables [line_intersects_circle AD O X Y]
variables (h : length AC = length AB + length AD)

-- Theorem statement
theorem angle_XOY_is_120 (h1 : (length AC = length AB + length AD)) : angle X O Y = 120 :=
sorry

end angle_XOY_is_120_l426_426481


namespace convex_ngon_triangle_area_l426_426839

theorem convex_ngon_triangle_area (n : ℕ) (h : 3 ≤ n) :
  ∀ (vertices : list (ℝ × ℝ)), 
  (∀ (v : ℝ × ℝ), v ∈ vertices → 0 ≤ v.1 ∧ v.1 ≤ 1 ∧ 0 ≤ v.2 ∧ v.2 ≤ 1) →
  (∀ (a b c : ℝ × ℝ), a ∈ vertices → b ∈ vertices → c ∈ vertices → 
  a ≠ b → b ≠ c → c ≠ a → 
  let S := abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)) / 2 in
  ∃ (A B C : ℝ × ℝ), A ∈ vertices ∧ B ∈ vertices ∧ C ∈ vertices ∧
  S ≤ 8 / (n * n)) :=
sorry

end convex_ngon_triangle_area_l426_426839


namespace polar_to_rectangular_l426_426900

theorem polar_to_rectangular :
  ∀ (r θ : ℝ), r = 6 → θ = π / 3 → 
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  (x, y) = (3, 3 * Real.sqrt 3) :=
by
  intros r θ hr hθ
  simp [hr, hθ]
  sorry

end polar_to_rectangular_l426_426900


namespace prove_b_minus_a_l426_426759

def rotated_then_reflected (a b : ℝ) : Prop :=
  let p_rot := (2 + (a - 2) * (Real.cos (Float.pi/4)) + (b - 3) * (Real.sin (Float.pi/4)),
                3 - (a - 2) * (Real.sin (Float.pi/4)) + (b - 3) * (Real.cos (Float.pi/4)))
  let p_reflect := (p_rot.2, p_rot.1)
  p_reflect = (5, -1)

theorem prove_b_minus_a (a b : ℝ) (h : rotated_then_reflected a b) : b - a = 3.1 :=
  sorry

end prove_b_minus_a_l426_426759


namespace find_complex_value_l426_426173

def z1 : ℂ := 1 + complex.I * real.sqrt 3
def z2 : ℂ := complex.I * real.sqrt 3 - 1
def solution : ℂ := -2 - 2 * complex.I

theorem find_complex_value : (z1 ^ 2) / z2 = solution := by
  sorry

end find_complex_value_l426_426173


namespace prob1_prob2_odd_prob2_monotonic_prob3_l426_426203

variable (a : ℝ) (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, f (log a x) = a / (a^2 - 1) * (x - 1 / x))
variable (ha : 0 < a ∧ a < 1)

-- Problem 1: Prove the expression for f(x)
theorem prob1 (x : ℝ) : f x = a / (a^2 - 1) * (a^x - a^(-x)) := sorry

-- Problem 2: Prove oddness and monotonicity of f(x)
theorem prob2_odd : ∀ x, f (-x) = -f x := sorry
theorem prob2_monotonic : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → (f x₁ < f x₂) := sorry

-- Problem 3: Determine the range of k
theorem prob3 (k : ℝ) : (∀ t : ℝ, 1 ≤ t ∧ t ≤ 3 → f (3 * t^2 - 1) + f (4 * t - k) > 0) → (k < 6) := sorry

end prob1_prob2_odd_prob2_monotonic_prob3_l426_426203


namespace number_of_distinct_pairs_l426_426903

theorem number_of_distinct_pairs :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^4 * y^4 - 16 * x^2 * y^2 + 15 = 0 :=
by
  have xy_nonneg : ∀ x y : ℕ, x > 0 → y > 0 → (x^4 * y^4 - 16 * x^2 * y^2 + 15) = 0 → x * y = 1, sorry
  -- use xy_nonneg to prove the assertion
  sorry

end number_of_distinct_pairs_l426_426903


namespace max_distance_circle_line_l426_426013

theorem max_distance_circle_line : 
    let circle_eq := ∀ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 10 = 0
    let line_eq := ∀ (x y : ℝ), x + y - 14 = 0
    ∃ (P : ℝ × ℝ), circle_eq P.1 P.2 ∧ 
    (∀ (Q : ℝ × ℝ), circle_eq Q.1 Q.2 → 
    ∃ (d : ℝ), d = abs (Q.1 + Q.2 - 14) / real.sqrt(2) ∧ 
    d ≤ 8 * real.sqrt(2)) :=
sorry

end max_distance_circle_line_l426_426013


namespace wizard_elixir_combinations_l426_426108

theorem wizard_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let invalid_combinations := 3
  herbs * crystals - invalid_combinations = 21 := 
by
  sorry

end wizard_elixir_combinations_l426_426108


namespace complex_conjugate_of_z_l426_426596

theorem complex_conjugate_of_z (a b : ℝ) (h_eq : ∃ x : ℝ, x^2 + (4 + complex.i) * x + (4 + a * complex.i) = 0 ∧ x = b) :
  complex.conj (a + b * complex.i) = 2 + 2 * complex.i :=
by { sorry }

end complex_conjugate_of_z_l426_426596


namespace largest_divisor_problem_l426_426004

theorem largest_divisor_problem (N : ℕ) :
  (∃ k : ℕ, let m := Nat.gcd N (N - 1) in
            N + m = 10^k) ↔ N = 75 :=
by 
  sorry

end largest_divisor_problem_l426_426004


namespace part_a_part_b_l426_426035

variables {A B C D P Q K L M K' L' M' : Type}
variables [Trapezoid ABCD AD] [Trapezoid APQD AD] -- Assuming Trapezoid is a defined class or structure

-- Definition of pairwise distinct lengths for the bases, can be expressed more precisely depending on implementation
def distinct_bases : Prop :=
  base_length AB ≠ base_length CD ∧
  base_length AB ≠ base_length AD ∧
  base_length AB ≠ base_length PQ ∧
  base_length CD ≠ base_length AD ∧
  base_length CD ≠ base_length PQ ∧
  base_length AD ≠ base_length PQ

-- Collinearity definitions
def collinear (X Y Z : Type) : Prop :=
  ∃ (l : Line), X ∈ l ∧ Y ∈ l ∧ Z ∈ l

theorem part_a (h1 : distinct_bases) :
  intersection_point (line_through A B) (line_through C D) = K →
  intersection_point (line_through A P) (line_through D Q) = L →
  intersection_point (line_through B P) (line_through C Q) = M →
  collinear K L M :=
sorry

theorem part_b (h1 : distinct_bases) :
  intersection_point (line_through A B) (line_through C D) = K' →
  intersection_point (line_through A Q) (line_through D P) = L' →
  intersection_point (line_through B Q) (line_through C P) = M' →
  collinear K' L' M' :=
sorry

end part_a_part_b_l426_426035


namespace cherries_per_quart_of_syrup_l426_426295

-- Definitions based on conditions
def time_to_pick_cherries : ℚ := 2
def cherries_picked_in_time : ℚ := 300
def time_to_make_syrup : ℚ := 3
def total_time_for_all_syrup : ℚ := 33
def total_quarts : ℚ := 9

-- Derivation of how many cherries are needed per quart
theorem cherries_per_quart_of_syrup : 
  (cherries_picked_in_time / time_to_pick_cherries) * (total_time_for_all_syrup - total_quarts * time_to_make_syrup) / total_quarts = 100 :=
by
  repeat { sorry }

end cherries_per_quart_of_syrup_l426_426295


namespace f_perfect_square_l426_426175

def f (N : ℕ) : ℕ :=
  ∑ g in Nat.divisors N, 2 ^ Nat.omega g * Nat.d (N / g)

theorem f_perfect_square (N : ℕ) (hN_pos : 0 < N) : ∃ k : ℕ, f N = k * k :=
by
  sorry

end f_perfect_square_l426_426175


namespace prove_target_value_l426_426581

-- We need variables for coefficients and polynomials
variables (a b x y : ℝ)

-- Define the polynomials
def poly1 := a * x^2 + 2 * x * y - x
def poly2 := 3 * x^2 - 2 * b * x * y + 3 * y

-- Define the condition that the difference between the polynomials has no quadratic term
def no_quadratic_term (p1 p2 : ℝ) := a = 3 ∧ 2 = -2 * b

-- Target value to prove
def target_value (a b : ℝ) := a^2 - 4 * b = 13

theorem prove_target_value : no_quadratic_term a b → target_value a b :=
by
  intros h
  cases h with ha hb
  rw [ha, hb]
  calc 3^2 - 4 * (-1) = 9 + 4 : by norm_num
  ... = 13 : by norm_num

end prove_target_value_l426_426581


namespace roots_diff_l426_426162

theorem roots_diff (m : ℝ) : 
  (∃ α β : ℝ, 2 * α * α - m * α - 8 = 0 ∧ 
              2 * β * β - m * β - 8 = 0 ∧ 
              α ≠ β ∧ 
              α - β = m - 1) ↔ (m = 6 ∨ m = -10 / 3) :=
by
  sorry

end roots_diff_l426_426162


namespace largest_angle_in_pentagon_l426_426653

theorem largest_angle_in_pentagon (P Q R S T : ℝ) 
          (h1 : P = 70) 
          (h2 : Q = 100)
          (h3 : R = S) 
          (h4 : T = 3 * R - 25)
          (h5 : P + Q + R + S + T = 540) : 
          T = 212 :=
by
  sorry

end largest_angle_in_pentagon_l426_426653


namespace susan_annual_percentage_increase_l426_426297

theorem susan_annual_percentage_increase :
  let initial_jerry := 14400
  let initial_susan := 6250
  let jerry_first_year := initial_jerry * (6 / 5 : ℝ)
  let jerry_second_year := jerry_first_year * (9 / 10 : ℝ)
  let jerry_third_year := jerry_second_year * (6 / 5 : ℝ)
  jerry_third_year = 18662.40 →
  (initial_susan : ℝ) * (1 + r)^3 = 18662.40 →
  r = 0.44 :=
by {
  sorry
}

end susan_annual_percentage_increase_l426_426297


namespace arithmetic_mean_of_powers_of_3_l426_426885

noncomputable def sum_powers_of_three : ℕ := 
  (3^1 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9)

def arithmetic_mean := (sum_powers_of_three / 9 : ℝ)

theorem arithmetic_mean_of_powers_of_3 : 
  arithmetic_mean = 2970 :=
by
  sorry

end arithmetic_mean_of_powers_of_3_l426_426885


namespace pentagon_diagonal_sum_l426_426311

theorem pentagon_diagonal_sum :
  let AB : ℝ := 5
  let BC : ℝ := 12
  let CD : ℝ := 5
  let DE : ℝ := 12
  let AE : ℝ := 16
  ∃ (p q : ℚ) (hpq_coprime : p.natAbs.gcd q.natAbs = 1), 
  (
    p/q = (3*20 + 256/5 + 125/4) ∧ 
    p + q = 1131
  ) :=
by
  let AB : ℝ := 5
  let BC : ℝ := 12
  let CD : ℝ := 5
  let DE : ℝ := 12
  let AE : ℝ := 16
  -- Definitions for diagonal lengths a, b, c are derived through equations
  -- Based on the given conditions in the problem.
  let a := 256 / 5
  let b := 125 / 4
  let c := 20
  let diagonal_sum := 3 * c + a + b
  have p : ℚ := 1111
  have q : ℚ := 20
  have hpq_coprime : Int.gcd p.natAbs q.natAbs = 1 := by sorry  -- Proof to show that 1111 and 20 are relatively prime
  use p, q, hpq_coprime
  -- Given condition should prove the sum of diagonals
  have h : p / q = diagonal_sum := by sorry  -- Verify that 1111/20 is indeed the diagonal sum
  have h_sum : p + q = 1131 := by sorry  -- Assert the final sum of p and q
  exact ⟨h, h_sum⟩

end pentagon_diagonal_sum_l426_426311


namespace percentage_of_teachers_with_neither_issue_l426_426859

theorem percentage_of_teachers_with_neither_issue 
  (total_teachers : ℕ)
  (teachers_with_bp : ℕ)
  (teachers_with_stress : ℕ)
  (teachers_with_both : ℕ)
  (h1 : total_teachers = 150)
  (h2 : teachers_with_bp = 90)
  (h3 : teachers_with_stress = 60)
  (h4 : teachers_with_both = 30) :
  let neither_issue_teachers := total_teachers - (teachers_with_bp + teachers_with_stress - teachers_with_both)
  let percentage := (neither_issue_teachers * 100) / total_teachers
  percentage = 20 :=
by
  -- skipping the proof
  sorry

end percentage_of_teachers_with_neither_issue_l426_426859


namespace sin_cos_identity_trig_identity_l426_426828

open Real

-- Problem I
theorem sin_cos_identity (α : ℝ) : 
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 → 
  sin α * cos α = 3 / 10 := 
sorry

-- Problem II
theorem trig_identity : 
  (sqrt (1 - 2 * sin (10 * π / 180) * cos (10 * π / 180))) / 
  (cos (10 * π / 180) - sqrt (1 - cos (170 * π / 180)^2)) = 1 := 
sorry

end sin_cos_identity_trig_identity_l426_426828


namespace max_balls_in_cube_l426_426806

-- Main definition
def radius : ℝ := 3
def side_length : ℝ := 10

-- Volumes
def volume_cube (side_length : ℝ) : ℝ := side_length ^ 3
def volume_ball (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

-- Proving the maximum number of balls
theorem max_balls_in_cube (r s : ℝ) (h_r : r = 3) (h_s : s = 10) :
  ⌊(volume_cube s) / (volume_ball r)⌋ = 8 := by
  -- Reiterate the conditions given
  rw [h_r, h_s]
  -- Continue with specific assumptions or steps
  sorry

end max_balls_in_cube_l426_426806


namespace sqrt_eq_2a_plus_2b_iff_zero_l426_426798

-- Define the conditions as hypotheses
variables (a b : ℝ)
-- Assuming a, b are non-negative real numbers
hypothesis h_nonneg_a : a ≥ 0
hypothesis h_nonneg_b : b ≥ 0

-- Statement of the problem
theorem sqrt_eq_2a_plus_2b_iff_zero :
  (sqrt (a^2 + b^2) = 2 * (a + b)) ↔ (a = 0 ∧ b = 0) :=
by
  sorry -- Proof goes here

end sqrt_eq_2a_plus_2b_iff_zero_l426_426798


namespace polar_to_rectangular_l426_426898

theorem polar_to_rectangular (r θ : ℝ) (h1 : r = 6) (h2 : θ = Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (3, 3 * Real.sqrt 3) :=
by
  rw [h1, h2]
  have h3 : Real.cos (Real.pi / 3) = 1 / 2 := Real.cos_pi_div_three
  have h4 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := Real.sin_pi_div_three
  simp [h3, h4]
  norm_num
  sorry

end polar_to_rectangular_l426_426898


namespace factorial_divides_expression_l426_426720

theorem factorial_divides_expression (n : ℕ) (a b : ℤ) (hn : 0 < n) :
  n.factorial ∣ (finset.range n).prod (λ k, (a + k * b)) * b ^ (n - 1) :=
sorry

end factorial_divides_expression_l426_426720


namespace area_covered_by_four_layers_l426_426559

theorem area_covered_by_four_layers (A S D T F: ℝ) 
  (h1 : A = 280) 
  (h2 : S = 180) 
  (h3 : D = 36) 
  (h4 : T = 16) :
  F = 12 :=
by
  have equation := (S - D - T - F) + 2 * D + 3 * T + 4 * F = A
  have simplified := (180 - 36 - 16 - F) + 2 * 36 + 3 * 16 + 4 * F = 280
  have rearranged := 244 + 3 * F = 280
  have final_calc := (280 - 244) / 3
  have F_val := 36 / 3
  exact F_val
-- assumption to skip proof details
  sorry

end area_covered_by_four_layers_l426_426559


namespace prove_a1_geq_2k_l426_426068

variable (n k : ℕ) (a : ℕ → ℕ)
variable (h1: ∀ i, 1 ≤ i → i ≤ n → 1 < a i)
variable (h2: ∀ i j, 1 ≤ i → i < j → j ≤ n → ¬ (a i ∣ a j))
variable (h3: 3^k < 2*n ∧ 2*n < 3^(k + 1))

theorem prove_a1_geq_2k : a 1 ≥ 2^k :=
by
  sorry

end prove_a1_geq_2k_l426_426068


namespace solve_equation_l426_426904

theorem solve_equation (x : ℝ) (h : (2 / (x - 3) = 3 / (x - 6))) : x = -3 :=
sorry

end solve_equation_l426_426904


namespace find_AB_l426_426283

variable (a h : ℝ)

-- Pyramid conditions
def pyramid_conditions : Prop :=
  PA = PB ∧ PA = PC ∧ PA = PD ∧ angle B M D = 90 ∧ 
  volume (pyramid P A B M D) = 288

-- Midpoint definition
def midpoint (M P C : ℝ) : Prop := M = (P + C) / 2

-- Volume of solid condition
def volume_condition : Prop := 
  volume (pyramid P A B C D) - volume (pyramid M B C D) = 288

-- Relationship given by the right angle
def angle_BMD_90 (B M D : ℝ) : Prop := B * B + M * M = D * D

theorem find_AB 
  (PA PB PC PD : ℝ) 
  (AB : ℝ) 
  (P A B C D M : ℝ)
  (V_PABMD : ℝ)
  (h : ℝ)
  (relationship : h = sqrt (3 / 2) * a)  
  (volume_condition : volume_condition)
  (angle_BMD_90 : angle_BMD_90 B M D)
  (pyramid_conditions : pyramid_conditions) :
  AB = 4 * sqrt 6 :=
begin
  sorry
end

end find_AB_l426_426283


namespace find_ratio_l426_426981

open Real

-- Definitions and conditions
variables (b1 b2 : ℝ) (F1 F2 : ℝ × ℝ)
noncomputable def ellipse_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 49) + (Q.2^2 / b1^2) = 1
noncomputable def hyperbola_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 16) - (Q.2^2 / b2^2) = 1
noncomputable def same_foci (Q : ℝ × ℝ) : Prop := true  -- Placeholder: Representing that both shapes have the same foci F1 and F2

-- The main theorem
theorem find_ratio (Q : ℝ × ℝ) (h1 : ellipse_eq b1 Q) (h2 : hyperbola_eq b2 Q) (h3 : same_foci Q) : 
  abs ((dist Q F1) - (dist Q F2)) / ((dist Q F1) + (dist Q F2)) = 4 / 7 := 
sorry

end find_ratio_l426_426981


namespace felicity_gas_usage_l426_426529

variable (A F : ℕ)

theorem felicity_gas_usage
  (h1 : F = 4 * A - 5)
  (h2 : A + F = 30) :
  F = 23 := by
  sorry

end felicity_gas_usage_l426_426529


namespace problem_solution_l426_426578

theorem problem_solution
  (α : ℝ)
  (h1 : 0 < α)
  (h2 : α < π / 2)
  (h3 : cos α - sin α = sqrt 5 / 5) :
  (1 - tan α) / (sin (2 * α) - cos (2 * α) + 1) = 5 / 12 := 
by
  sorry

end problem_solution_l426_426578


namespace perpendicular_vectors_m_eq_half_l426_426613

theorem perpendicular_vectors_m_eq_half (m : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (-1, m)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = 1 / 2 :=
sorry

end perpendicular_vectors_m_eq_half_l426_426613


namespace hyperbola_asymptote_l426_426742

theorem hyperbola_asymptote (x y : ℝ) : 
  (∀ x y : ℝ, (x^2 / 25 - y^2 / 16 = 1) → (y = (4 / 5) * x ∨ y = -(4 / 5) * x)) := 
by 
  sorry

end hyperbola_asymptote_l426_426742


namespace angle_relation_l426_426609

theorem angle_relation (R : ℝ) (hR : R > 0) (d : ℝ) (hd : d > R) 
  (α β : ℝ) : β = 3 * α :=
sorry

end angle_relation_l426_426609


namespace collinearity_necessary_but_not_sufficient_l426_426797

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u

def equal (u v : V) : Prop := u = v

theorem collinearity_necessary_but_not_sufficient (u v : V) :
  (collinear u v → equal u v) ∧ (equal u v → collinear u v) → collinear u v ∧ ¬(collinear u v ↔ equal u v) :=
sorry

end collinearity_necessary_but_not_sufficient_l426_426797


namespace sufficient_but_not_necessary_condition_l426_426263

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (floor x = floor y) → (abs (x - y) < 1) ∧ (¬ (abs (x - y) < 1) → (floor x ≠ floor y)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l426_426263


namespace sum_of_roots_of_quadratic_eq_l426_426427

theorem sum_of_roots_of_quadratic_eq (x : ℝ) (hx : x^2 = 8 * x + 15) :
  ∃ S : ℝ, S = 8 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l426_426427


namespace general_term_of_arithmetic_sequence_sum_first_n_b_n_l426_426594

-- Definitions based on conditions in a)
def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def a_n (n : ℕ) : ℕ := 2 * n

def b_n (n : ℕ) : ℕ := 3 ^ (a_n n / 2)

def sum_first_n_terms (f : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finset.range n, f i

-- Mathematical equivalent proof problems in Lean 4
theorem general_term_of_arithmetic_sequence :
  is_arithmetic_sequence a_n ∧ a_n 1 = 2 ∧ a_n 1 + a_n 2 + a_n 3 = 12 → ∀ n, a_n n = 2 * n :=
by sorry

theorem sum_first_n_b_n :
  (∀ n, b_n n = 3 ^ ((2 * n) / 2)) → ∀ n, sum_first_n_terms b_n n = (3 ^ (n + 1) - 3) / 2 :=
by sorry

end general_term_of_arithmetic_sequence_sum_first_n_b_n_l426_426594


namespace fraction_equal_l426_426246

variable {m n p q : ℚ}

-- Define the conditions
def condition1 := (m / n = 20)
def condition2 := (p / n = 5)
def condition3 := (p / q = 1 / 15)

-- State the theorem
theorem fraction_equal (h1 : condition1) (h2 : condition2) (h3 : condition3) : (m / q = 4 / 15) :=
  sorry

end fraction_equal_l426_426246


namespace total_carrots_l426_426715

theorem total_carrots (sally_carrots fred_carrots mary_carrots : ℕ)
  (h_sally : sally_carrots = 6)
  (h_fred : fred_carrots = 4)
  (h_mary : mary_carrots = 10) :
  sally_carrots + fred_carrots + mary_carrots = 20 := 
by sorry

end total_carrots_l426_426715


namespace P_not_77_for_all_integers_l426_426344

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_not_77_for_all_integers (x y : ℤ) : P x y ≠ 77 :=
sorry

end P_not_77_for_all_integers_l426_426344


namespace average_daily_rainfall_second_week_l426_426523

theorem average_daily_rainfall_second_week
    (total_rainfall : ℝ)
    (ratio_second_week : ℝ)
    (increase_rate : ℝ)
    (rainfall_second_week : ℝ) :
    total_rainfall = 50 →
    ratio_second_week = 1.5 →
    increase_rate = 0.10 →
    rainfall_second_week = (50 / 17.5) * 1.10 →
    rainfall_second_week ≈ 3.1427 :=
by
    intros h1 h2 h3 h4
    sorry

end average_daily_rainfall_second_week_l426_426523


namespace count_pairs_mn_lt_50_l426_426239

theorem count_pairs_mn_lt_50 :
  {p : ℕ × ℕ | let m := p.1 in let n := p.2 in 0 < m ∧ 0 < n ∧ m^2 + n^2 < 50}.to_finset.card = 32 :=
by sorry

end count_pairs_mn_lt_50_l426_426239


namespace floor_gt_a_minus_1_l426_426681

-- Define the floor function as the greatest integer less than or equal to a real number
def floor (a : ℝ) : ℤ := Int.floor a

-- The statement to be proved
theorem floor_gt_a_minus_1 (a : ℝ) : floor a > a - 1 :=
by
  sorry

end floor_gt_a_minus_1_l426_426681


namespace part1_part2_l426_426975

def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem part1 (m : ℝ) (h₁ : 0 < m) (h₂ : f' x = (-Real.log x) / x^2) :
  (∃ y ∈ Ioo m (m + 0.5), has_deriv_within_at f 0 y) → (0.5 < m ∧ m < 1) :=
sorry

theorem part2 (t : ℝ) :
  (∀ x ≥ 1, f x ≥ t / (x + 1)) ↔ t ≤ 2 :=
sorry

end part1_part2_l426_426975


namespace relationship_among_a_b_c_l426_426961

noncomputable def a : ℝ := (1 / 3) ^ 3
noncomputable def b (x : ℝ) : ℝ := x ^ 3
noncomputable def c (x : ℝ) : ℝ := Real.log x

theorem relationship_among_a_b_c (x : ℝ) (h : x > 2) : a < c x ∧ c x < b x :=
by {
  -- proof steps are skipped
  sorry
}

end relationship_among_a_b_c_l426_426961


namespace number_of_triangles_bound_l426_426133

theorem number_of_triangles_bound (n m T : ℕ) (h_graph : Graph n) (h_edges : h_graph.num_edges = m) (h_triangles : h_graph.num_triangles = T) :
  T ≥ m * (4 * m - n^2) / (3 * n) := 
sorry

end number_of_triangles_bound_l426_426133


namespace max_area_of_garden_l426_426567

theorem max_area_of_garden (total_fence : ℝ) (gate : ℝ) (remaining_fence := total_fence - gate) :
  total_fence = 60 → gate = 4 → (remaining_fence / 2) * (remaining_fence / 2) = 196 :=
by 
  sorry

end max_area_of_garden_l426_426567


namespace bob_questions_three_hours_l426_426124

theorem bob_questions_three_hours : 
  let first_hour := 13
  let second_hour := first_hour * 2
  let third_hour := second_hour * 2
  first_hour + second_hour + third_hour = 91 :=
by
  sorry

end bob_questions_three_hours_l426_426124


namespace weight_of_first_group_sugar_cube_l426_426076

theorem weight_of_first_group_sugar_cube (sugar_cubes1 sugar_cubes2 : ℕ) (hours1 hours2 : ℕ) (ants1 ants2 : ℕ) (weight_sugar_cube2 : ℕ)
  (rate1 : ants1 * hours1 = sugar_cubes1)
  (rate2 : ants2 * hours2 = sugar_cubes2)
  (weight_rate2 : weight_sugar_cube2 = 5) :
  weight_sugar_cube1 * 2 = 10 := 
by {
  let rate_per_ant1 := sugar_cubes1 / (ants1 * hours1),
  let rate_per_ant2 := sugar_cubes2 / (ants2 * hours2),
  have h1 : sugar_cubes1 = (hours1 * ants1 * rate_per_ant1), from sorry,
  have h2 : sugar_cubes2 = (hours2 * ants2 * rate_per_ant2), from sorry,
  have h3 : rate_per_ant2 = 2 * rate_per_ant1, from sorry,
  have h4 : weight_sugar_cube2 = 5, by apply weight_rate2,
  show weight_sugar_cube1 = 10 from sorry,
}

end weight_of_first_group_sugar_cube_l426_426076


namespace worker_savings_l426_426488

theorem worker_savings (P : ℝ) (f : ℝ) (h : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  have h1 : 12 * f * P = 4 * (1 - f) * P := h
  have h2 : P ≠ 0 := sorry  -- P should not be 0 for the worker to have a meaningful income.
  field_simp [h2] at h1
  linarith

end worker_savings_l426_426488


namespace Kaleb_savings_l426_426302

variable (k : ℝ) -- Kaleb's initial savings
variable (allowance : ℝ) -- Additional allowance Kaleb received
variable (toy_cost : ℝ) -- Cost per toy
variable (number_of_toys : ℝ) -- Number of toys Kaleb wants to buy

-- Given conditions
axiom allowance_eq : allowance = 15
axiom toy_cost_eq : toy_cost = 6
axiom number_of_toys_eq : number_of_toys = 6

-- Kaleb's savings after receiving the allowance
def total_savings := k + allowance

-- Total cost of toys
def total_cost := toy_cost * number_of_toys

-- Final condition based on the problem
axiom can_afford_toys : total_savings = total_cost

theorem Kaleb_savings : k = 21 := by
  rw [allowance_eq, toy_cost_eq, number_of_toys_eq, ←can_afford_toys]
  have hs : k + 15 = 6 * 6 := by sorry -- Using given axiom
  have h1 : k + 15 = 36 := by sorry -- Simplified
  have h2 : k = 21 := by sorry -- Solving for k
  exact h2

end Kaleb_savings_l426_426302


namespace roots_of_quadratic_l426_426765

theorem roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = -2 ∧ c = -6) :
  let Δ := b^2 - 4 * a * c in
  Δ > 0 :=
by
  obtain ⟨ha, hb, hc⟩ := h_eq
  have Δ_def : Δ = b^2 - 4 * a * c := rfl
  rw [ha, hb, hc] at Δ_def
  sorry

end roots_of_quadratic_l426_426765


namespace compute_expression_l426_426960
-- Start with importing math library utilities for linear algebra and dot product

-- Define vector 'a' and 'b' in Lean
def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (-1, 2)

-- Define dot product operation 
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- Define the expression and the theorem
theorem compute_expression : dot_product ((2 * a.1 + b.1, 2 * a.2 + b.2)) a = 1 :=
by
  -- Insert the proof steps here
  sorry

end compute_expression_l426_426960


namespace extreme_value_a_zero_monotonicity_a_pos_range_of_m_l426_426223

noncomputable def f (a x : ℝ) : ℝ := -a / 2 * x^2 + (a + 1) * x - Real.log x

theorem extreme_value_a_zero :
  (∃ x : ℝ, x = 1 ∧ 0 < x ∧ f 0 x = 1) :=
by
  have f_zero (x : ℝ) : f 0 x = x - Real.log x := by
    simp [f]
    calc
      - (0 : ℝ) / 2 * x ^ 2         = 0                 := by ring
      (0 + 1) * x - Real.log x      = x - Real.log x    := by ring
  use 1
  simp [f_zero, Real.log_one]
  exact one_gt_zero

theorem monotonicity_a_pos (a : ℝ) (ha : 0 < a) :
  (∀ x y : ℝ, x ∈ Ioo 0 1 → y ∈ Ioo 1 (1 / a) → f x < f y) ∧
  (∀ x y : ℝ, x ∈ Ioo (1 / a) 1 → y ∈ Ioo 1 ⊤ → f y < f x) :=
by
  sorry

theorem range_of_m (a : ℝ) (ha : 2 < a ∧ a < 3)
  (x1 x2 : ℝ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 2) (hx2 : 1 ≤ x2 ∧ x2 ≤ 2) :
  ∀ m : ℝ, m ≥ 1 / 8 ∧ (a^2 - 1) / 2 * m + Real.log 2 > |f a x1 - f a x2| :=
by
  sorry

end extreme_value_a_zero_monotonicity_a_pos_range_of_m_l426_426223


namespace imaginary_part_of_z_l426_426213

open Complex

theorem imaginary_part_of_z:
  ∀ (z : ℂ), (z = 1 - 4 * I) → (z.im = -4) :=
by
  assume z
  intro h
  rw [h]
  simp
  sorry

end imaginary_part_of_z_l426_426213


namespace angle_parallel_result_l426_426036

theorem angle_parallel_result (A B : ℝ) (h1 : A = 60) (h2 : (A = B ∨ A + B = 180)) : (B = 60 ∨ B = 120) :=
by
  sorry

end angle_parallel_result_l426_426036


namespace find_m_value_l426_426204

variables {R : Type*} [linear_ordered_field R]

def odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f (x)

def monotone_increasing_after (f : R → R) (m : R) : Prop :=
  ∀ x y : R, x ≥ m → y ≥ m → x ≤ y → f x ≤ f y

def has_unique_zero (f : R → R) : Prop :=
  ∃! z : R, f z = 0

theorem find_m_value (f : R → R) (m : R)
  (h1 : odd_function (λ x, f (x - 5)))
  (h2 : monotone_increasing_after f m) :
  m = -5 :=
sorry

end find_m_value_l426_426204


namespace find_exact_speed_l426_426333

variable (d t v : ℝ)

-- Conditions as Lean definitions
def distance_eq1 : d = 50 * (t - 1/12) := sorry
def distance_eq2 : d = 70 * (t + 1/12) := sorry
def travel_time : t = 1/2 := sorry -- deduced travel time from the equations and given conditions
def correct_speed : v = 42 := sorry -- Mr. Bird needs to drive at 42 mph to be exactly on time

-- Lean 4 statement proving the required speed is 42 mph
theorem find_exact_speed : v = d / t :=
  by
    sorry

end find_exact_speed_l426_426333


namespace combine_figures_to_symmetry_l426_426131

variables {figure : Type} [DecidableEq figure] [Inhabited figure]

-- Definition to describe a symmetric figure
def is_symmetric (f : figure) : Prop :=
∀ x, f x = f (-x)

-- Definition to say that the figure has an axis of symmetry
def has_axis_of_symmetry (f : figure) : Prop :=
∃ axis, ∀ x, f x = f (axis - x)

-- Given three figures that initially do not have symmetry
variables (f1 f2 f3 : figure)
(hf1 : ¬ has_axis_of_symmetry f1)
(hf2 : ¬ has_axis_of_symmetry f2)
(hf3 : ¬ has_axis_of_symmetry f3)

-- We need to combine f1, f2, and f3 to make a new figure f that does have a line of symmetry
def combined_figure (f1 f2 f3 : figure) : figure := sorry

-- The statement to be proven
theorem combine_figures_to_symmetry : 
  has_axis_of_symmetry (combined_figure f1 f2 f3) :=
sorry

end combine_figures_to_symmetry_l426_426131


namespace seating_arrangements_l426_426275

theorem seating_arrangements :
  let lakers : ℕ := 2
  let celtics : ℕ := 2
  let warriors : ℕ := 1
  (lakers * celtics * warriors = 5) →
  (∃ (ways : ℕ), ways = 3! * 2! * 2! * 1! ∧ ways = 24) :=
  by
    intros _ _ _ _ _ _ _ _
    cases _ 
    sorry

end seating_arrangements_l426_426275


namespace second_most_notebooks_l426_426299

def notebooks : Type := {name : String, count : Nat}

def Jian := {name := "Jian", count := 3}
def Doyun := {name := "Doyun", count := 5}
def Siu := {name := "Siu", count := 2}

theorem second_most_notebooks:
  ∃ x : notebooks, x.name = "Jian" ∧ x.count = 3 ∧ (
    Jian.count > Siu.count ∧ Jian.count < Doyun.count ∧
    ∀ y : notebooks, y ≠ Jian → (y.count > Jian.count → y = Doyun) ∧ (y.count < Jian.count → y = Siu)
  ) :=
sorry

end second_most_notebooks_l426_426299


namespace pencil_cost_l426_426791

theorem pencil_cost (p q : ℤ) (H1 : 3 * p + 4 * q = 287) (H2 : 5 * p + 2 * q = 236) : q = 52 :=
by
  -- Set up the system of linear equations
  let eq1 := H1
  let eq2 := H2

  -- Manipulate the equations (steps omitted for brevity)
  sorry

end pencil_cost_l426_426791


namespace percentage_neither_bp_nor_ht_l426_426474

noncomputable def percentage_teachers_neither_condition (total: ℕ) (high_bp: ℕ) (heart_trouble: ℕ) (both: ℕ) : ℚ :=
  let either_condition := high_bp + heart_trouble - both
  let neither_condition := total - either_condition
  (neither_condition * 100 : ℚ) / total

theorem percentage_neither_bp_nor_ht :
  percentage_teachers_neither_condition 150 90 50 30 = 26.67 :=
by
  sorry

end percentage_neither_bp_nor_ht_l426_426474


namespace problem1_problem2_l426_426602

noncomputable def f (x : ℝ): ℝ := x^3 - 12 * x + 8

theorem problem1 (a b : ℝ) (f : ℝ → ℝ) (h₁ : f x = x^3 - a * x + b) (h₂ : ∃ x₀, f'(x₀) = 0 ∧ f(x₀) = -8 ∧ x₀ = 2) :
  a = 12 ∧ b = 8 :=
sorry

theorem problem2 :
  ∀ x, x ∈ [-3, 3] → (f x ∈ [-8, 24]) :=
sorry

end problem1_problem2_l426_426602


namespace bob_calories_consumed_l426_426883

/-- Bob eats half of the pizza with 8 slices, each slice being 300 calories.
   Prove that Bob eats 1200 calories. -/
theorem bob_calories_consumed (total_slices : ℕ) (calories_per_slice : ℕ) (half_slices : ℕ) (calories_consumed : ℕ) 
  (h1 : total_slices = 8)
  (h2 : calories_per_slice = 300)
  (h3 : half_slices = total_slices / 2)
  (h4 : calories_consumed = half_slices * calories_per_slice) 
  : calories_consumed = 1200 := 
sorry

end bob_calories_consumed_l426_426883


namespace monotonically_increasing_interval_l426_426218

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + (Real.pi / 12))

noncomputable def f'' (x : ℝ) : ℝ := (deriv^[2]) f x

noncomputable def y (x : ℝ) : ℝ := 2 * f x + f'' x

def is_monotonically_increasing (y : ℝ → ℝ) (I : Set ℝ) := 
  ∀ x1 x2, x1 ∈ I → x2 ∈ I → x1 ≤ x2 → y x1 ≤ y x2

theorem monotonically_increasing_interval : 
  is_monotonically_increasing y (Set.Icc (7 * Real.pi / 12) (13 * Real.pi / 12)) :=
sorry

end monotonically_increasing_interval_l426_426218


namespace fiscal_expenditure_max_l426_426378

theorem fiscal_expenditure_max {x y b a e : ℝ} 
  (h1 : y = b * x + a + e)
  (h2 : b = 0.8)
  (h3 : a = 2)
  (h4 : |e| ≤ 0.5)
  (h5 : x = 10) :
  y ≤ 10.5 :=
begin
  sorry -- proof omitted
end

end fiscal_expenditure_max_l426_426378


namespace geometric_seq_min_3b2_7b3_l426_426692

theorem geometric_seq_min_3b2_7b3 (b_1 b_2 b_3 : ℝ) (r : ℝ) 
  (h_seq : b_1 = 2) (h_geom : b_2 = b_1 * r) (h_geom2 : b_3 = b_1 * r^2) :
  3 * b_2 + 7 * b_3 ≥ -16 / 7 :=
by
  -- Include the necessary definitions to support the setup
  have h_b1 : b_1 = 2 := h_seq
  have h_b2 : b_2 = 2 * r := by rw [h_geom, h_b1]
  have h_b3 : b_3 = 2 * r^2 := by rw [h_geom2, h_b1]
  sorry

end geometric_seq_min_3b2_7b3_l426_426692


namespace constant_force_l426_426023

variable (m λ : ℝ) -- mass and proportionality constant
variable (s : ℝ → ℝ) -- distance as a function of time
variable (t : ℝ) -- time

-- Condition 1: velocity v is proportional to the square root of distance s
def velocity (s : ℝ → ℝ) (λ : ℝ) (t : ℝ) : ℝ := λ * (s t) ^ (1/2)

-- Newton's Second Law: F = m * a, where a is the second derivative of s with respect to t
def force (m : ℝ) (s : ℝ → ℝ) (t : ℝ) : ℝ := m * ((s' t)' t)

-- Theorem: The force acting on the body is constant
theorem constant_force (h1 : ∀ t, (velocity s λ t) = (s' t))
  : force m s t = m * (λ^2 / 2) := 
  sorry

end constant_force_l426_426023


namespace isosceles_trapezoid_l426_426834

theorem isosceles_trapezoid
  (A B C D I : Type)
  [linear_ordered_field I]
  (ω : circle)
  (AB CD AI BI CI DI : I)
  (h : (AI + DI) ^ 2 + (BI + CI) ^ 2 = (AB + CD) ^ 2) :
  is_isosceles_trapezoid A B C D := sorry

end isosceles_trapezoid_l426_426834


namespace inradius_squared_eq_xyz_over_sum_l426_426359

variables {A B C D E F : Type} [Triangle A B C]

noncomputable def inradius (ABC : Triangle A B C) : ℝ := sorry
noncomputable def incircle_touches (ABC : Triangle A B C) (D E F : ℝ) : Prop := sorry

variables {x y z r : ℝ}

theorem inradius_squared_eq_xyz_over_sum (ABC_inradius : inradius (ABC) = r)
    (incircle_at : incircle_touches (ABC) x y z) :
    r^2 = (x * y * z) / (x + y + z) :=
by
  sorry

end inradius_squared_eq_xyz_over_sum_l426_426359


namespace coeff_x4_in_expansion_correct_l426_426544

noncomputable def coeff_x4_in_expansion (f g : ℕ → ℤ) := 
  ∀ (c : ℤ), c = 80 → f 4 + g 1 * g 3 = c

-- Definitions of the individual polynomials
def poly1 (x : ℤ) : ℤ := 4 * x^2 - 2 * x + 1
def poly2 (x : ℤ) : ℤ := 2 * x + 1

-- Expanded form coefficients
def coeff_poly1 : ℕ → ℤ
  | 0       => 1
  | 1       => -2
  | 2       => 4
  | _       => 0

def coeff_poly2_pow4 : ℕ → ℤ
  | 0       => 1
  | 1       => 8
  | 2       => 24
  | 3       => 32
  | 4       => 16
  | _       => 0

-- The theorem we want to prove
theorem coeff_x4_in_expansion_correct :
  coeff_x4_in_expansion coeff_poly1 coeff_poly2_pow4 := 
by
  sorry

end coeff_x4_in_expansion_correct_l426_426544


namespace parrots_false_statements_l426_426650

theorem parrots_false_statements (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 140 ∧ 
    (∀ statements : ℕ → Prop, 
      (statements 0 = false) ∧ 
      (∀ i : ℕ, 1 ≤ i → i < n → 
          (statements i = true → 
            (∃ fp : ℕ, fp < i ∧ 7 * (fp + 1) > 10 * i)))) := 
by
  sorry

end parrots_false_statements_l426_426650


namespace largest_diff_l426_426130

/-- Proof problem: Given the conditions of attendance estimates for three events in Chicago, Denver, and Miami, -/
-- Prove that the largest possible difference between the numbers attending any two events, to the nearest 1,000, is 45,000.
theorem largest_diff (C D M : ℝ) (hC : 38000 ≤ C ∧ C ≤ 42000) (hD : 47826 ≤ D ∧ D ≤ 64706) (hM : 67500 ≤ M ∧ M ≤ 82500) : 
  |((max (max C D) M) - (min (min C D) M)) ≈ 45000| := sorry

end largest_diff_l426_426130


namespace fraction_problem_l426_426244

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end fraction_problem_l426_426244


namespace probability_sum_16_is_1_over_64_l426_426375

-- Define the problem setup
def octahedral_faces : Finset ℕ := Finset.range 9 \ {0} -- Faces labeled 1 through 8

-- Define the event for the sum of 16
def event_sum_16 : Finset (ℕ × ℕ) :=
  Finset.filter (λ p, p.1 + p.2 = 16) (octahedral_faces.product octahedral_faces)

-- Total outcomes with two octahedral dice
def total_outcomes : ℕ := octahedral_faces.card * octahedral_faces.card

-- Probability of rolling a sum of 16
def probability_sum_16 : ℚ := (event_sum_16.card : ℚ) / total_outcomes

theorem probability_sum_16_is_1_over_64 :
  probability_sum_16 = 1 / 64 := by
  sorry

end probability_sum_16_is_1_over_64_l426_426375


namespace purely_periodic_fraction_l426_426379

theorem purely_periodic_fraction (A B : ℕ) (hA : A < 10) (hB : B < 10) :
  ∃ p : ℕ, ∀ n ≥ p, let sequence := λ n, nat.mod (nat.nat.iterate (λpair, (pair.2, (pair.1 + pair.2) % 10)) n (A, B)).1 10 in
  sequence (n + p) = sequence n := sorry

end purely_periodic_fraction_l426_426379


namespace twelve_position_in_circle_l426_426780

theorem twelve_position_in_circle (a : ℕ → ℕ) (h_cyclic : ∀ i, a (i + 20) = a i)
  (h_sum_six : ∀ i, a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) = 24)
  (h_first : a 1 = 1) :
  a 12 = 7 :=
sorry

end twelve_position_in_circle_l426_426780


namespace complex_polygon_area_l426_426033

-- Define the conditions
def side_length : ℝ := 6
def middle_rotation : ℝ := 45
def top_rotation : ℝ := 90

-- The question and correct answer combined in a theorem
theorem complex_polygon_area :
  let area := (3 * √2 * side_length * side_length) / 2 + side_length * side_length 
  in area = 36 + 18 * √2 := 
by
  sorry

end complex_polygon_area_l426_426033


namespace remaining_download_time_l426_426330

-- Define the relevant quantities
def total_size : ℝ := 1250
def downloaded : ℝ := 310
def download_speed : ℝ := 2.5

-- State the theorem
theorem remaining_download_time : (total_size - downloaded) / download_speed = 376 := by
  -- Proof will be filled in here
  sorry

end remaining_download_time_l426_426330


namespace isosceles_trapezoid_diagonal_length_l426_426135

theorem isosceles_trapezoid_diagonal_length :
  ∀ (AB CD AD: ℝ), (AB = 30) → (CD = 12) → (AD = 13) → 
  let BC : ℝ := AD
  let h : ℝ := sqrt (AD^2 - ((AB - CD) / 2)^2)
  let AC : ℝ := sqrt (AD^2 + h^2)
  AC = 2 * sqrt 26 :=
by
  intros AB CD AD hAC hCD hAD BC h hAC_proof
  let BC := AD
  let h := sqrt (AD^2 - ((AB - CD) / 2)^2)
  let AC := sqrt (AD^2 + h^2)
  sorry

end isosceles_trapezoid_diagonal_length_l426_426135


namespace combinations_with_repetition_l426_426869

theorem combinations_with_repetition : 
  (nat.choose (4 + 2 - 1) 2) = 10 :=
by
  sorry

end combinations_with_repetition_l426_426869


namespace point_in_quadrant_II_l426_426145

variables {A B C : ℝ}
variables (hA : 0 < A ∧ A < 90)
variables (hB : 0 < B ∧ B < 90)
variables (hABC : A + B + C = 180)

theorem point_in_quadrant_II (h1 : 0 < A ∧ A < 90) (h2 : 0 < B ∧ B < 90) (h3 : A + B + C = 180) :
  let x := cos B - sin A,
      y := sin B - cos A
  in x < 0 ∧ y > 0 :=
by
  sorry

end point_in_quadrant_II_l426_426145


namespace least_possible_value_of_b_plus_c_l426_426031

theorem least_possible_value_of_b_plus_c :
  ∃ (b c : ℕ), (b > 0) ∧ (c > 0) ∧ (∃ (r1 r2 : ℝ), r1 - r2 = 30 ∧ 2 * r1 ^ 2 + b * r1 + c = 0 ∧ 2 * r2 ^ 2 + b * r2 + c = 0) ∧ b + c = 126 := 
by
  sorry 

end least_possible_value_of_b_plus_c_l426_426031


namespace central_cell_value_l426_426910

def grid := List (List ℕ)

def is_valid_grid (g : grid) : Prop :=
  List.length g = 5 ∧ (∀ row, row ∈ g → List.length row = 5)

def sum_of_grid (g : grid) : ℕ :=
  g.foldl (λ acc row, acc + row.foldl (λ acc' n, acc' + n) 0) 0

def sum_of_subrectangle (g : grid) (r c : ℕ) : ℕ :=
  g[r]![c]! + g[r]![c+1]! + g[r]![c+2]!

def valid_subrectangles (g : grid) : Prop :=
  ∀ r, ∀ c, r < 5 → c < 3 → sum_of_subrectangle g r c = 23

theorem central_cell_value (g : grid) (hr : is_valid_grid g)
  (hs : sum_of_grid g = 200) (hv : valid_subrectangles g) : ∀ center, center = g[2]![2] → center = 16 :=
by
  sorry

end central_cell_value_l426_426910


namespace students_only_science_is_55_l426_426118

-- Definitions from conditions
variable (Total Students_in_Science Students_in_Art Students_in_Both Students_only_Science : ℕ)
variable h_total : Total = 120
variable h_sci : Students_in_Science = 80
variable h_art : Students_in_Art = 65
variable h_both : Students_in_Both = Students_in_Science + Students_in_Art - Total
variable h_only_sci : Students_only_Science = Students_in_Science - Students_in_Both

-- The theorem we need to prove
theorem students_only_science_is_55 : Students_only_Science = 55 :=
by
  rw [h_total, h_sci, h_art] at h_both
  rw [h_both] at h_only_sci
  have both_value : Students_in_Both = 25 := by
    linarith
  rw [both_value] at h_only_sci
  exact h_only_sci.trans rfl

end students_only_science_is_55_l426_426118


namespace juniors_score_l426_426479

/-- Mathematical proof problem stated in Lean 4 -/
theorem juniors_score 
  (total_students : ℕ) 
  (juniors seniors : ℕ)
  (junior_score senior_avg total_avg : ℝ)
  (h_total_students : total_students > 0)
  (h_juniors : juniors = total_students / 10)
  (h_seniors : seniors = (total_students * 9) / 10)
  (h_total_avg : total_avg = 84)
  (h_senior_avg : senior_avg = 83)
  (h_junior_score_same : ∀ j : ℕ, j < juniors → ∃ s : ℝ, s = junior_score)
  :
  junior_score = 93 :=
by
  sorry

end juniors_score_l426_426479


namespace problem_probability_of_40_cents_l426_426360

-- Definitions
def coin (penny nickel dime quarter half_dollar : bool) : ℤ :=
  (if penny then 1 else 0) + (if nickel then 5 else 0) +
  (if dime then 10 else 0) + (if quarter then 25 else 0) +
  (if half_dollar then 50 else 0)

def successful_outcomes (cfg : (bool × bool × bool × bool × bool)) : bool :=
  coin cfg.1 cfg.2.1 cfg.2.2.1 cfg.2.2.2.1 cfg.2.2.2.2 ≥ 40

noncomputable def probability_of_success : ℚ :=
  let outcomes := finset.univ.product (finset.univ.product (finset.univ.product (finset.univ.product finset.univ)))
  let success_count := (outcomes.filter successful_outcomes).card
  success_count / outcomes.card

-- Theorem statement (no proof provided)
theorem problem_probability_of_40_cents :
  probability_of_success = 9 / 16 :=
sorry

end problem_probability_of_40_cents_l426_426360


namespace train_crosses_signal_post_l426_426079

theorem train_crosses_signal_post :
  ∃ (time : ℕ), time = 40 ∧ 
  ∀ (length_train : ℕ) (time_bridge : ℕ) (distance_bridge : ℕ),
    length_train = 600 ∧ 
    time_bridge = 20 * 60 ∧ 
    distance_bridge = 18 * 1000 → 
    (distance_bridge / time_bridge) = (length_train / time) :=
begin
  sorry,
end

end train_crosses_signal_post_l426_426079


namespace percentage_decrease_area_l426_426815

theorem percentage_decrease_area (r : ℝ) (h : 0 < r) :
  let A := π * r^2,
      new_r := 0.5 * r,
      A_new := π * (new_r)^2 in
  ((A - A_new) / A) * 100 = 75 :=
by
  have A_def : A = π * r^2 := rfl
  have new_r_def : new_r = 0.5 * r := rfl
  have A_new_def : A_new = π * (new_r)^2 := rfl
  sorry

end percentage_decrease_area_l426_426815


namespace system_of_equations_solution_l426_426174

theorem system_of_equations_solution
  (x y z : ℤ)
  (h1 : x + y + z = 12)
  (h2 : 8 * x + 5 * y + 3 * z = 60) :
  (x = 0 ∧ y = 12 ∧ z = 0) ∨
  (x = 2 ∧ y = 7 ∧ z = 3) ∨
  (x = 4 ∧ y = 2 ∧ z = 6) :=
sorry

end system_of_equations_solution_l426_426174


namespace nathalie_total_coins_l426_426334

theorem nathalie_total_coins
  (quarters dimes nickels : ℕ)
  (ratio_condition : quarters = 9 * nickels ∧ dimes = 3 * nickels)
  (value_condition : 25 * quarters + 10 * dimes + 5 * nickels = 1820) :
  quarters + dimes + nickels = 91 :=
by
  sorry

end nathalie_total_coins_l426_426334


namespace xiaotian_sep_usage_plan_cost_effectiveness_l426_426826

noncomputable def problem₁ (units : List Int) : Real :=
  units.sum / 1024 + 5 * 6

theorem xiaotian_sep_usage (units : List Int) (h : units = [200, -100, 100, -100, 212, 200]) :
  problem₁ units = 30.5 :=
sorry

def plan_cost_a (x : Int) : Real := 5 * x + 4

def plan_cost_b (x : Int) : Real :=
  if h : 20 < x ∧ x <= 23 then 5 * x - 1
  else 3 * x + 45

theorem plan_cost_effectiveness (x : Int) (h : x > 23) :
  plan_cost_a x > plan_cost_b x :=
sorry

end xiaotian_sep_usage_plan_cost_effectiveness_l426_426826


namespace product_of_smallest_prime_and_composite_l426_426389

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n

def smallest_prime : ℕ := 2
def smallest_composite : ℕ := 4

theorem product_of_smallest_prime_and_composite : smallest_prime * smallest_composite = 8 := by
  have h_prime : is_prime smallest_prime := by
    dsimp [smallest_prime, is_prime]
    sorry -- Proof that 2 is prime

  have h_composite : is_composite smallest_composite := by
    dsimp [smallest_composite, is_composite, is_prime]
    sorry -- Proof that 4 is composite

  -- Calculate the product
  calc
    smallest_prime * smallest_composite = 2 * 4 : by rw [smallest_prime, smallest_composite]
    ... = 8 : by norm_num

end product_of_smallest_prime_and_composite_l426_426389


namespace proportion_solution_l426_426257

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 := 
by 
suffices h₀ : x = 6 / 5 by sorry
suffices h₁ : 6 / 5 = 1.2 by sorry
-- Proof steps go here
sorry

end proportion_solution_l426_426257


namespace tangent_line_at_1_l426_426984

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (1 + x) = 2 * f (1 - x) - x^2 + 3 * x + 1

theorem tangent_line_at_1 : ∀ x : ℝ, (f 1 = -1) ∧ (f' 1 = 1) → (x - (f 1) - 2 = 0) :=
by
  intro x
  intro h
  cases h with h1 h2
  sorry

end tangent_line_at_1_l426_426984


namespace ratio_of_work_done_by_women_to_men_l426_426077

theorem ratio_of_work_done_by_women_to_men 
  (total_work_men : ℕ := 15 * 21 * 8)
  (total_work_women : ℕ := 21 * 36 * 5) :
  (total_work_women : ℚ) / (total_work_men : ℚ) = 2 / 3 :=
by
  -- Proof goes here
  sorry

end ratio_of_work_done_by_women_to_men_l426_426077


namespace proof_problem_l426_426226

-- Definition of line and parabola equations
def line_eq (x y : ℝ) := y = 2*x - 3
def parabola_eq (x y : ℝ) := y^2 = 4*x

-- Definitions of points A and B where the line intersects the parabola
def point_A : ℝ × ℝ := (2 + real.sqrt 7 / 2, 1 + real.sqrt 7)
def point_B : ℝ × ℝ := (2 - real.sqrt 7 / 2, 1 - real.sqrt 7)

-- Definition of the slopes k1 and k2 of lines OA and OB
def k1 (x y : ℝ) : ℝ := y / x
def k2 (x y : ℝ) : ℝ := y / x

-- The theorem statement that needs to be proved
theorem proof_problem : 
  let k1 := (point_A.2 / point_A.1),
      k2 := (point_B.2 / point_B.1) in
  (1 / k1) + (1 / k2) = 1 / 2 :=
by
  sorry

end proof_problem_l426_426226


namespace intersection_correct_l426_426573

-- Definitions of the sets A and B
def A : set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ y = -x }
def B : set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ y = x^2 - 2 }

-- Intersection of A and B
def A_inter_B : set (ℝ × ℝ) := { p | p ∈ A ∧ p ∈ B }

-- Define the expected set
def expected : set (ℝ × ℝ) := { (-2, 2), (1, -1) }

-- Statement that A ∩ B is equal to the expected set
theorem intersection_correct : A_inter_B = expected :=
sorry

end intersection_correct_l426_426573


namespace product_of_a_and_c_l426_426406

theorem product_of_a_and_c (a b c : ℝ) (h1 : a + b + c = 100) (h2 : a - b = 20) (h3 : b - c = 30) : a * c = 378.07 :=
by
  sorry

end product_of_a_and_c_l426_426406


namespace find_a_l426_426395

-- Define the curve y = x^2 + x
def curve (x : ℝ) : ℝ := x^2 + x

-- Line equation ax - y + 1 = 0
def line (a : ℝ) (x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, line a x y → y = x^2 + x) ∧
  (deriv curve 1 = 2 * 1 + 1) →
  (2 * 1 + 1 = -1 / a) →
  a = -1 / 3 :=
by
  sorry

end find_a_l426_426395


namespace least_number_of_shoes_needed_l426_426816

def num_inhabitants : ℕ := 10000
def percent_one_legged : ℝ := 0.05
def num_one_legged : ℕ := (percent_one_legged * num_inhabitants).toNat
def num_two_legged : ℕ := num_inhabitants - num_one_legged
def barefooted_fraction : ℝ := 0.5
def barefooted_two_legged : ℕ := (barefooted_fraction * num_two_legged).toNat
def num_shoes_one_legged : ℕ := num_one_legged
def num_shoes_two_legged : ℕ := (num_two_legged - barefooted_two_legged) * 2
def total_num_shoes : ℕ := num_shoes_one_legged + num_shoes_two_legged

theorem least_number_of_shoes_needed : total_num_shoes = 10000 := sorry

end least_number_of_shoes_needed_l426_426816


namespace max_balls_in_cube_l426_426804

noncomputable def volume_cube (side_length : ℝ) : ℝ := side_length ^ 3
noncomputable def volume_ball (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

theorem max_balls_in_cube (side_length : ℝ) (radius : ℝ) (h_cube : side_length = 10) (h_ball : radius = 3) :
  Nat.floor (volume_cube side_length / volume_ball radius) = 8 :=
by
  rw [h_cube, h_ball]
  have V_cube : volume_cube 10 = 1000 := by norm_num [volume_cube]
  have V_ball : volume_ball 3 = 36 * Real.pi := by norm_num [volume_ball, Real.pi]
  sorry

end max_balls_in_cube_l426_426804


namespace find_m_value_l426_426579

variable {A B C O : Point}
variable (m : ℝ)
variable [Circle O] [AcuteTriangle A B C]
variable (h₁ : tan A = 1 / 2)
variable (h₂ : (cos B / sin C) • (vector AB) + (cos C / sin B) • (vector AC) = 2 * m • (vector AO))

theorem find_m_value :
  m = sqrt 5 / 5 :=
  sorry

end find_m_value_l426_426579


namespace union_eq_l426_426576

open Set

theorem union_eq (A B : Set ℝ) (hA : A = {x | -1 < x ∧ x < 1}) (hB : B = {x | 0 ≤ x ∧ x ≤ 2}) :
    A ∪ B = {x | -1 < x ∧ x ≤ 2} :=
by
  rw [hA, hB]
  ext x
  simp
  sorry

end union_eq_l426_426576


namespace equilateral_right_triangle_impossible_l426_426810
-- Import necessary library

-- Define the conditions and the problem statement
theorem equilateral_right_triangle_impossible :
  ¬(∃ (A B C : ℝ), A > 0 ∧ B > 0 ∧ C > 0 ∧ A = B ∧ B = C ∧ (A^2 + B^2 = C^2) ∧ (A + B + C = 180)) := sorry

end equilateral_right_triangle_impossible_l426_426810


namespace largest_divisor_power_of_ten_l426_426009

theorem largest_divisor_power_of_ten (N : ℕ) (m : ℕ) (k : ℕ) 
  (h1 : m ∣ N)
  (h2 : m < N)
  (h3 : N + m = 10^k) : N = 75 := sorry

end largest_divisor_power_of_ten_l426_426009


namespace find_a_l426_426761

noncomputable def P (a : ℚ) (k : ℕ) : ℚ := a * (1 / 2)^(k)

theorem find_a (a : ℚ) : (P a 1 + P a 2 + P a 3 = 1) → (a = 8 / 7) :=
by
  sorry

end find_a_l426_426761


namespace largest_divisor_problem_l426_426003

theorem largest_divisor_problem (N : ℕ) :
  (∃ k : ℕ, let m := Nat.gcd N (N - 1) in
            N + m = 10^k) ↔ N = 75 :=
by 
  sorry

end largest_divisor_problem_l426_426003


namespace proof_problem_l426_426323

noncomputable def ellipse_equation (a b x y : ℝ) : Prop :=
  y^2 / a^2 + x^2 / b^2 = 1

noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

def point_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  ellipse_equation a b x y

def line_passing_through (m : ℝ) (x y : ℝ) : Prop :=
  y = m * x + 2

theorem proof_problem :
  ∃ (a b : ℝ), ellipse_equation a b x y ∧
  eccentricity a (sqrt 2 / 2) = sqrt 2 / 2 ∧
  point_on_ellipse (sqrt 2) 1 (sqrt 2 / 2) (-1) ∧
  (∀ (A B : ℝ), line_passing_through m (-2) 0 → 
     -- proof of maximum area of triangle AOB 
     max_area = sqrt 2 / 2) ∧
  (∀ (m : ℝ), 2x - sqrt 14 y + 4 = 0 ∨ 2x + sqrt 14 y + 4 = 0 ∧ 
     equation_of ~line_when_max ~area maximum_places) :=
sorry

end proof_problem_l426_426323


namespace even_fn_increasing_on_interval_min_value_l426_426634

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
variable {β : Type*} [Preorder β] [OrderBot β] {f : α → β}

theorem even_fn_increasing_on_interval_min_value
    (h_even : ∀ x, f x = f (-x))
    (h_increasing : ∀ {x y}, 1 ≤ x → x ≤ 3 → 1 ≤ y → y ≤ 3 → x < y → f x ≤ f y)
    (h_min_value : ∃ x, 1 ≤ x ∧ x ≤ 3 ∧ f x = 5) :
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≥ 5) ∧ (∀ {x y}, -3 ≤ x → x < y → y ≤ -1 → f y ≤ f x) :=
sorry

end even_fn_increasing_on_interval_min_value_l426_426634


namespace tortoise_seeing_hare_fraction_is_q_l426_426847

def rational_lt_one (q : ℚ) : Prop := q < 1

def tortoise_runs_with_speed (q : ℚ) (time : ℚ) : Prop :=
  ∃ m n : ℕ, q = m / n ∧ m.gcd n = 1 ∧ time = 4 * m * n

def tortoise_sees_hare_fraction (q : ℚ) (S : ℕ → ℕ → ℕ) (m n : ℕ) : Prop :=
  ∃ h : rat, h = S m n / (4 * m * n) ∧ h = q

noncomputable def S (m n : ℕ) : ℕ :=
if 2 ∣ m * n then m * n
else if m % 4 = n % 4 ∧ (m % 4 = 1 ∨ m % 4 = 3) then m * n + 3
else if m % 4 = -(n % 4) ∧ (m % 4 = 1 ∨ m % 4 = 3) then m * n + 1
else 0

theorem tortoise_seeing_hare_fraction_is_q (q : ℚ) :
  rational_lt_one q →
  (∃ (time : ℚ), tortoise_runs_with_speed q time) →
  tortoise_sees_hare_fraction q (λ m n, S m n) 4 4 :=
sorry

end tortoise_seeing_hare_fraction_is_q_l426_426847


namespace minimize_m_n_l426_426233

noncomputable section

def vector_a (n : ℝ) : ℝ × ℝ := (1, 2 * n)
def vector_b (m n : ℝ) : ℝ × ℝ := (m + n, m)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem minimize_m_n (m n : ℝ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_dot : dot_product (vector_a n) (vector_b m n) = 1) :
  m + n ≥ real.sqrt 3 - 1 :=
sorry

end minimize_m_n_l426_426233


namespace monotonic_intervals_of_function_l426_426015

theorem monotonic_intervals_of_function :
  ∀ x : ℝ, (1 ≤ x → deriv (λ x, (sqrt x) / (x + 1)) x ≤ 0) ∧ (0 ≤ x ∧ x ≤ 1 → deriv (λ x, (sqrt x) / (x + 1)) x ≥ 0) :=
by
  sorry

end monotonic_intervals_of_function_l426_426015


namespace no_nontrivial_sum_periodic_functions_l426_426086

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

def is_nontrivial_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  periodic f p ∧ ∃ x y, x ≠ y ∧ f x ≠ f y

theorem no_nontrivial_sum_periodic_functions (g h : ℝ → ℝ) :
  is_nontrivial_periodic_function g 1 →
  is_nontrivial_periodic_function h π →
  ¬ ∃ T > 0, ∀ x, (g + h) (x + T) = (g + h) x :=
sorry

end no_nontrivial_sum_periodic_functions_l426_426086


namespace profit_difference_l426_426703

theorem profit_difference (num_records : ℕ) (sammy_price : ℕ) (bryan_price_interested : ℕ) (bryan_price_not_interested : ℕ)
  (h1 : num_records = 200) (h2 : sammy_price = 4) (h3 : bryan_price_interested = 6) (h4 : bryan_price_not_interested = 1) :
  let total_sammy := num_records * sammy_price,
      num_records_half := num_records / 2,
      total_bryan := (num_records_half * bryan_price_interested) + (num_records_half * bryan_price_not_interested),
      difference := total_sammy - total_bryan
  in difference = 100 :=
by
  sorry

end profit_difference_l426_426703


namespace cookies_taken_in_four_days_l426_426402

-- Define the initial conditions
def initial_cookies : ℕ := 70
def remaining_cookies : ℕ := 28
def days_in_week : ℕ := 7
def days_of_interest : ℕ := 4

-- Define the total cookies taken out in a week
def cookies_taken_week := initial_cookies - remaining_cookies

-- Define the cookies taken out each day
def cookies_taken_per_day := cookies_taken_week / days_in_week

-- Final statement to show the number of cookies taken out in four days
theorem cookies_taken_in_four_days : cookies_taken_per_day * days_of_interest = 24 := by
  sorry -- The proof steps will be here.

end cookies_taken_in_four_days_l426_426402


namespace min_perimeter_lateral_face_l426_426821

theorem min_perimeter_lateral_face (x h : ℝ) (V : ℝ) (P : ℝ): 
  (x > 0) → (h > 0) → (V = 4) → (V = x^2 * h) → 
  (∀ y : ℝ, y > 0 → 2*y + 2 * (4 / y^2) ≥ P) → P = 6 := 
by
  intro x_pos h_pos volume_eq volume_expr min_condition
  sorry

end min_perimeter_lateral_face_l426_426821


namespace billiard_path_equal_sum_diagonals_l426_426686

noncomputable def diagonal_length (A B C D : Point) : ℝ := dist A C + dist B D

theorem billiard_path_equal_sum_diagonals (A B l1 l2 l3 l4 : Point) :
  (are_vertices_of_rectangle A B l1 l2 l3 l4) →
  dist A A = diagonal_length A l1 l2 l3 :=
by
  sorry

end billiard_path_equal_sum_diagonals_l426_426686


namespace registration_methods_l426_426955

theorem registration_methods (students activities : ℕ) (one_activity : students = 4 ∧ activities = 3) :
  (3 ^ students) = 81 :=
by
  obtain ⟨hs, ha⟩ := one_activity
  rw [hs, ha]
  norm_num
  sorry

end registration_methods_l426_426955


namespace captain_max_coins_l426_426369

structure Sailor (b : ℕ) :=
  (b1 b2 b3 : ℕ)
  (b1_ge_b2_ge_b3 : b1 ≥ b2 ∧ b2 ≥ b3)
  (sum_b_eq_2009 : b1 + b2 + b3 = 2009)

structure Captain (a : ℕ) := 
  (a1 a2 a3 : ℕ)
  (a1_ge_a2_ge_a3 : a1 ≥ a2 ∧ a2 ≥ a3)
  (sum_a_eq_2009 : a1 + a2 + a3 = 2009)

noncomputable def maxCaptainCoins : ℕ :=
  669

theorem captain_max_coins (s : Sailor 2009) (c : Captain 2009) : 
  ∀ {b1 b2 b3 : ℕ}, 
    Sailor.b1_ge_b2_ge_b3 s → Sailor.sum_b_eq_2009 s → 
    Captain.a1_ge_a2_ge_a3 c → Captain.sum_a_eq_2009 c →
    ∃ n, n = maxCaptainCoins ∧ n ≥ 669 :=
  by
    sorry

end captain_max_coins_l426_426369


namespace price_per_pound_of_peanuts_is_2_40_l426_426850

-- Assume the conditions
def peanuts_price_per_pound (P : ℝ) : Prop :=
  let cashews_price := 6.00
  let mixture_weight := 60
  let mixture_price_per_pound := 3.00
  let cashews_weight := 10
  let total_mixture_price := mixture_weight * mixture_price_per_pound
  let total_cashews_price := cashews_weight * cashews_price
  let total_peanuts_price := total_mixture_price - total_cashews_price
  let peanuts_weight := mixture_weight - cashews_weight
  let P := total_peanuts_price / peanuts_weight
  P = 2.40

-- Prove the price per pound of peanuts
theorem price_per_pound_of_peanuts_is_2_40 (P : ℝ) : peanuts_price_per_pound P :=
by
  sorry

end price_per_pound_of_peanuts_is_2_40_l426_426850


namespace arithmetic_sequence_term_21_l426_426440

theorem arithmetic_sequence_term_21 : 
  ∀ (a d n : ℕ), a = 3 → d = 5 → n = 21 → a + (n - 1) * d = 103 :=
by
  intro a d n ha hd hn
  rw [ha, hd, hn]
  sorry

end arithmetic_sequence_term_21_l426_426440


namespace cube_cross_section_area_l426_426740

/-- Proof Problem: Prove that the area of the cross-section of the cube through points K, L, and M is 156. -/
theorem cube_cross_section_area {A B C D A₁ B₁ C₁ D₁ K L M : Point}
  (edge_length : ℝ)
  (h_edge_length : edge_length = 12)
  (K_on_BC_extension : ∃ (d : ℝ), d = 9 ∧ point_lies_on_extension K B C d)
  (L_on_AB : ∃ (d : ℝ), d = 5 ∧ point_lies_on_edge L A B d)
  (M_on_A₁C₁ : divides_segment M A₁ C₁ 1 3) :
  area_of_section K L M = 156 := sorry

end cube_cross_section_area_l426_426740


namespace distance_inequality_l426_426058

open Triangle

variables {A B C I L₁ : Point}
variable {a b c : ℝ}
variable [Incenter I A B C]

noncomputable def vertex_to_incenter : ℝ := distance A I
noncomputable def incenter_to_base : ℝ := distance I L₁

theorem distance_inequality {A B C : Point} (h : Incenter I A B C)
  (hL₁ : angle_bisector_intersection A B C L₁)
  (ha : side_length A B B C = a)
  (hb : side_length A I I L₁ = b + c) :
  vertex_to_incenter > incenter_to_base :=
by
  sorry

end distance_inequality_l426_426058


namespace volume_union_cone_sphere_l426_426098

variables (R S : ℝ)

theorem volume_union_cone_sphere (R_pos : 0 < R) (S_pos : 0 < S) :
  let V := (1 / 3) * S * R in
  V = (1 / 3) * S * R :=
by
  sorry

end volume_union_cone_sphere_l426_426098


namespace cut_triangles_into_similar_pieces_l426_426189

-- Definitions representing the given conditions
structure Triangle :=
  (A B C : Point)

-- Assume we have two triangles
variable (redTriangle blueTriangle : Triangle)

-- Main theorem statement translating the original problem
theorem cut_triangles_into_similar_pieces (redTriangle blueTriangle : Triangle) : 
  ∃ R S P Q : Point, 
    (is_similar_piece (cut_triangle redTriangle R S P Q) (cut_triangle blueTriangle R S P Q)) ∧
    (is_similar_piece (cut_triangle redTriangle R S Q P) (cut_triangle blueTriangle R S Q P)) ∧
    (is_similar_piece (cut_quadrilateral redTriangle R S P Q) (cut_quadrilateral blueTriangle R S P Q)) :=
sorry

end cut_triangles_into_similar_pieces_l426_426189


namespace min_rubles_for_50_points_l426_426045

theorem min_rubles_for_50_points : ∃ (n : ℕ), minimal_rubles n ∧ n = 11 := by
  sorry

def minimal_rubles (n : ℕ) : Prop :=
  ∀ m, (steps_to_reach_50 m) ∧ (total_cost m ≤ n)

def steps_to_reach_50 (steps : list ℕ) : Prop :=
  ∃ initial_score : ℕ, initial_score = 0 ∧ 
  count_steps_to_50 initial_score steps = 50

def count_steps_to_50 (score : ℕ) (steps : list ℕ) : ℕ :=
  match steps with
  | [] => score
  | h :: t =>
    if h = 1 then
      count_steps_to_50 (score + 1) t
    else if h = 2 then 
      count_steps_to_50 (2 * score) t
    else
      score  -- Invalid step

end min_rubles_for_50_points_l426_426045


namespace calories_consumed_l426_426879

theorem calories_consumed (slices : ℕ) (calories_per_slice : ℕ) (half_pizza : ℕ) :
  slices = 8 → calories_per_slice = 300 → half_pizza = slices / 2 → 
  half_pizza * calories_per_slice = 1200 :=
by
  intros h_slices h_calories_per_slice h_half_pizza
  rw [h_slices, h_calories_per_slice] at h_half_pizza
  rw [h_slices, h_calories_per_slice]
  sorry

end calories_consumed_l426_426879


namespace math_problem_l426_426957

variable (a a1 a2 a3 a4 : ℝ)

noncomputable def sum_alternating_coeffs (n : ℕ) (a : ℝ) (a_coeffs : Fin (n + 1) → ℝ) : ℝ :=
  a + ∑ i in Finset.range n, (-1)^(i+1) * a_coeffs i

theorem math_problem (n : ℕ) (hn : n = 4)
  (hcomb : (nat.choose 20 (2 * n + 6) = nat.choose 20 (n + 2)))
  (hcoeffs : ∀ x : ℝ, (2 - x)^n = a + a1 * x + a2 * x^2 + a3 * x^3 + (a_coeffs : Fin (n + 1) → ℝ) x) :
  sum_alternating_coeffs n a a_coeffs = 81 :=
by
  sorry

end math_problem_l426_426957


namespace tangent_line_at_pi_l426_426743

theorem tangent_line_at_pi :
  ∀ (x : ℝ), y = sin x - 2 * x → 
  tangent_eqn : ∀ (p : ℝ × ℝ), (p = (π, sin π - 2 * π)) → 
  (3 * x + y - π) = 0 :=
by
  sorry

end tangent_line_at_pi_l426_426743


namespace checkerboard_squares_count_l426_426451

theorem checkerboard_squares_count :
  let checkerboard : List (List Bool) :=
    List.replicate 10 (List.cycle [true, false]) ++
    List.replicate 10 (List.cycle [false, true])

  ∃ checkerboard : List (List Bool), 
    (∀ i j, checkerboard i j = 
      if (i + j) % 2 = 0 then true else false) → 
  ∑ n in [4, 5, 6, 7, 8, 9, 10], 
    (10 - n + 1)^2 
    = 140 :=
begin
  sorry
end

end checkerboard_squares_count_l426_426451


namespace amandas_garden_flowers_l426_426868

theorem amandas_garden_flowers (A : ℕ) (P : ℕ) :
  P = 3 * A ∧ P - 15 = 45 → A = 20 := by
  intros h
  cases h with h1 h2
  sorry

end amandas_garden_flowers_l426_426868


namespace fraction_equality_l426_426564

variables (x y : ℝ)

theorem fraction_equality (h : y / 2 = (2 * y - x) / 3) : y / x = 2 :=
sorry

end fraction_equality_l426_426564


namespace power_function_properties_range_of_a_l426_426228

noncomputable def power_function (m : ℕ) : ℝ → ℝ := λ x, x^(9 - 3 * m)

theorem power_function_properties (m : ℕ) (h_m : m ∈ {1, 2}) :
  (∀ x : ℝ, power_function m x = power_function m (-x)) ∧ 
  (∀ x y : ℝ, power_function m x < power_function m y → x < y) → m = 2 ∧ (∀ x : ℝ, power_function 2 x = x^3) :=
by {
  intros h₁ h₂,
  sorry
}

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, power_function 2 x = x^3) → (power_function 2 (a + 1) + power_function 2 (3 * a - 4) < 0) →
  a < 3 / 4 :=
by {
  intros h₁ h₂,
  sorry
}

end power_function_properties_range_of_a_l426_426228


namespace min_h21_l426_426114

-- Definitions based on given conditions
variable {f : ℕ → ℤ}
variable extensive : ∀ x y, 0 < x → 0 < y → (f x) + (f y) ≥ (x:ℤ)^2 + (y:ℤ)^2

-- Hypothesis that h(1) + h(2) + ... + h(30) is as small as possible.
noncomputable def T := ∑ n in finset.range 30, f (n + 1)
noncomputable def min_T : ℤ := 5110

-- Statement to be proved
theorem min_h21 (h : ℕ → ℤ) (h_extensive : extensive) (h_sum : T = min_T) : h 21 ≥ 301 :=
sorry

end min_h21_l426_426114


namespace find_N_l426_426007

-- Definition of the conditions
def is_largest_divisor_smaller_than (m N : ℕ) : Prop := m < N ∧ Nat.gcd m N = m

def produces_power_of_ten (N m : ℕ) : Prop := ∃ k : ℕ, k > 0 ∧ N + m = 10^k

-- Final statement to prove
theorem find_N (N : ℕ) : (∃ m : ℕ, is_largest_divisor_smaller_than m N ∧ produces_power_of_ten N m) → N = 75 :=
by
  sorry

end find_N_l426_426007


namespace geometric_sequence_term_l426_426262

noncomputable def a1 : ℝ := 3 / 2
noncomputable def q : ℝ := 1 / 2
def a (n : ℕ) : ℝ := a1 * q^n

theorem geometric_sequence_term :
  (∀ n : ℕ, a n > 0) →
  a 1 + 2 * a 2 = 3 →
  a 3^2 = 4 * a 2 * a 6 →
  a 4 = 3 / 16 :=
by
  intros
  unfold a
  sorry

end geometric_sequence_term_l426_426262


namespace externally_tangent_circles_radius_l426_426585

theorem externally_tangent_circles_radius :
  ∃ r : ℝ, r > 0 ∧ (∀ x y, (x^2 + y^2 = 1 ∧ ((x - 3)^2 + y^2 = r^2)) → r = 2) :=
sorry

end externally_tangent_circles_radius_l426_426585


namespace concentration_of_salt_solution_used_l426_426833

-- The conditions of the problem
def total_volume : ℝ := 1 + 0.25
def resulting_concentration : ℝ := 0.10
def volume_of_salt_solution : ℝ := 0.25
def volume_of_water : ℝ := 1

-- The question to be proved: the concentration of the salt solution used
theorem concentration_of_salt_solution_used : 
  let C := (total_volume * resulting_concentration) / volume_of_salt_solution in
  C = 0.50 :=
by
  sorry

end concentration_of_salt_solution_used_l426_426833


namespace cos_inverse_addition_l426_426534

def cos_addition_formula (a b c1 c2: Real): Real :=
  (Real.cos a * Real.cos b) - (Real.sin a * Real.sin b)

theorem cos_inverse_addition :
  let a := Real.acos (4/5)
  let b := Real.cot (-1) 3
  Real.cos (a + b) = (9 * Real.sqrt 10) / 50 :=
  by
  sorry

end cos_inverse_addition_l426_426534


namespace total_children_in_circle_l426_426351

theorem total_children_in_circle 
  (n : ℕ)  -- number of children
  (h_even : Even n)   -- condition: the circle is made up of an even number of children
  (h_pos : n > 0) -- condition: there are some children
  (h_opposite : (15 % n + 15 % n) % n = 0)  -- condition: the 15th child clockwise from Child A is facing Child A (implies opposite)
  : n = 30 := 
sorry

end total_children_in_circle_l426_426351


namespace sums_cannot_be_13_to_20_l426_426706

theorem sums_cannot_be_13_to_20 :
  let numbers := List.range 1 10 in
  let grid := List.permutations numbers in
  let row_sums := grid.map (fun row => row.sum) in
  let column_sums := List.transpose grid |>.map (fun col => col.sum) in
  let diagonal_sums := [ grid.zipWith (fun i row => row[i]) [0, 1, 2] |>.sum,
                         grid.zipWith (fun i row => row[2-i]) [0, 1, 2] |>.sum ] in
  let all_sums := row_sums ++ column_sums ++ diagonal_sums in
  let possible_sums := [13, 14, 15, 16, 17, 18, 19, 20] in
  ∃ (sums : List ℕ), sums ⊆ possible_sums ∧ sums.length = 8 ∧ sums.sum = 90 →
  False := 
by
  sorry

end sums_cannot_be_13_to_20_l426_426706


namespace total_pies_baked_in_7_days_l426_426912

-- Define the baking rates (pies per day)
def Eddie_rate : Nat := 3
def Sister_rate : Nat := 6
def Mother_rate : Nat := 8

-- Define the duration in days
def duration : Nat := 7

-- Define the total number of pies baked in 7 days
def total_pies : Nat := Eddie_rate * duration + Sister_rate * duration + Mother_rate * duration

-- Prove the total number of pies is 119
theorem total_pies_baked_in_7_days : total_pies = 119 := by
  -- The proof will be filled here, adding sorry to skip it for now
  sorry

end total_pies_baked_in_7_days_l426_426912


namespace exists_projectile_time_l426_426373

noncomputable def projectile_time := 
  ∃ t1 t2 : ℝ, (-4.9 * t1^2 + 31 * t1 - 40 = 0) ∧ ((abs (t1 - 1.8051) < 0.001) ∨ (abs (t2 - 4.5319) < 0.001))

theorem exists_projectile_time : projectile_time := 
sorry

end exists_projectile_time_l426_426373


namespace compute_sum_l426_426084

def f : ℝ → ℝ := sorry

axiom symmetric_about : ∀ x : ℝ, f(x) = -f(x + 1.5)

axiom f_at_minus_one : f(-1) = 1
axiom f_at_zero : f(0) = -2

axiom graph_symmetric : ∀ x : ℝ, f(x) = -f(-x - 1.5)

theorem compute_sum : (∑ i in Finset.range 2017, f(i + 1)) = 1 :=
by sorry

end compute_sum_l426_426084


namespace expand_product_l426_426526

theorem expand_product (x : ℝ) : (x + 2) * (x + 5) = x^2 + 7 * x + 10 := 
by 
  sorry

end expand_product_l426_426526


namespace calories_consumed_l426_426878

theorem calories_consumed (slices : ℕ) (calories_per_slice : ℕ) (half_pizza : ℕ) :
  slices = 8 → calories_per_slice = 300 → half_pizza = slices / 2 → 
  half_pizza * calories_per_slice = 1200 :=
by
  intros h_slices h_calories_per_slice h_half_pizza
  rw [h_slices, h_calories_per_slice] at h_half_pizza
  rw [h_slices, h_calories_per_slice]
  sorry

end calories_consumed_l426_426878


namespace find_f_l426_426995

noncomputable def f (x : ℝ) : ℝ := 
  e^x - (f' (0)) * x + 1

theorem find_f (x : ℝ) (h : f(x) = e^x - f'(0) * x + 1) : 
  f(x) = e^x - (1/2) * x + 1 :=
sorry

end find_f_l426_426995


namespace quadratic_has_two_distinct_real_roots_l426_426769

theorem quadratic_has_two_distinct_real_roots :
  ∀ x : ℝ, ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (x^2 - 2 * x - 6 = 0 ∧ x = r1 ∨ x = r2) :=
by sorry

end quadratic_has_two_distinct_real_roots_l426_426769


namespace ellipse_equation_l426_426741

theorem ellipse_equation (a c b : ℝ) (h1 : a = 4) (h2 : c = 2 * Real.sqrt 3) (h3 : b = Real.sqrt (a^2 - c^2)):
  (a > b) →
  let ellipse_eq := (y:ℝ) (x:ℝ) → (y^2 / 16) + (x^2 / 4) = 1
  in ellipse_eq y x :=
begin
  sorry
end

end ellipse_equation_l426_426741


namespace compute_y_l426_426618

def sum_prod_formula (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), k * (n + 1 - k)

theorem compute_y :
  sum_prod_formula 1995 = 1995 * 997 * 665 :=
by
  sorry

end compute_y_l426_426618


namespace part_a_part_b_l426_426866

-- Definitions of the basic tiles, colorings, and the proposition

inductive Color
| black : Color
| white : Color

structure Tile :=
(c00 c01 c10 c11 : Color)

-- Ali's forbidden tiles (6 types for part (a))
def forbiddenTiles_6 : List Tile := 
[ Tile.mk Color.black Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white,
  Tile.mk Color.black Color.white Color.white Color.black,
  Tile.mk Color.black Color.white Color.black Color.black,
  Tile.mk Color.black Color.black Color.black Color.black,
  Tile.mk Color.white Color.white Color.white Color.white
]

-- Ali's forbidden tiles (7 types for part (b))
def forbiddenTiles_7 : List Tile := 
[ Tile.mk Color.black Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white,
  Tile.mk Color.black Color.white Color.white Color.black,
  Tile.mk Color.black Color.white Color.black Color.black,
  Tile.mk Color.black Color.black Color.black Color.black,
  Tile.mk Color.white Color.white Color.white Color.white,
  Tile.mk Color.black Color.white Color.black Color.white
]

-- Propositions to be proved

-- Part (a): Mohammad can color the infinite table with no forbidden tiles present
theorem part_a :
  ∃f : ℕ × ℕ → Color, ∀ t ∈ forbiddenTiles_6, ∃ x y : ℕ, ¬(f (x, y) = t.c00 ∧ f (x, y+1) = t.c01 ∧ 
  f (x+1, y) = t.c10 ∧ f (x+1, y+1) = t.c11) := 
sorry

-- Part (b): Ali can present 7 forbidden tiles such that Mohammad cannot achieve his goal
theorem part_b :
  ∀ f : ℕ × ℕ → Color, ∃ t ∈ forbiddenTiles_7, ∃ x y : ℕ, (f (x, y) = t.c00 ∧ f (x, y+1) = t.c01 ∧ 
  f (x+1, y) = t.c10 ∧ f (x+1, y+1) = t.c11) := 
sorry

end part_a_part_b_l426_426866


namespace volume_of_cube_with_edge_7_l426_426428

-- Defining the edge length
def edge_length : ℕ := 7

-- Defining the volume of the cube with a given edge
def cube_volume (a : ℕ) : ℕ := a ^ 3

-- Proving the volume of the cube with an edge length of 7 cm
theorem volume_of_cube_with_edge_7 : cube_volume edge_length = 343 := by
  unfold cube_volume
  norm_num

end volume_of_cube_with_edge_7_l426_426428


namespace painted_cube_count_is_three_l426_426843

-- Define the colors of the faces
inductive Color
| Yellow
| Black
| White

-- Define a Cube with painted faces
structure Cube :=
(f1 f2 f3 f4 f5 f6 : Color)

-- Define rotational symmetry (two cubes are the same under rotation)
def equivalentUpToRotation (c1 c2 : Cube) : Prop := sorry -- Symmetry function

-- Define a property that counts the correct painting configuration
def paintedCubeCount : ℕ :=
  sorry -- Function to count correctly painted and uniquely identifiable cubes

theorem painted_cube_count_is_three :
  paintedCubeCount = 3 :=
sorry

end painted_cube_count_is_three_l426_426843


namespace jackson_tiles_per_square_foot_l426_426293

theorem jackson_tiles_per_square_foot :
  ∀ (length width : ℕ) (green_tile_cost red_tile_cost total_cost : ℚ),
    length = 10 →
    width = 25 →
    green_tile_cost = 3 →
    red_tile_cost = 1.5 →
    total_cost = 2100 →
    (0.4 * (total_cost / (0.4 * green_tile_cost + 0.6 * red_tile_cost))) / (length * width) = 4 :=
by 
  intros length width green_tile_cost red_tile_cost total_cost h_length h_width h_green_cost h_red_cost h_total_cost
  rw [h_length, h_width, h_green_cost, h_red_cost, h_total_cost]
  norm_num

end jackson_tiles_per_square_foot_l426_426293


namespace find_c_for_local_max_at_2_l426_426028

noncomputable def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

theorem find_c_for_local_max_at_2 :
  (∃ c : ℝ, ∀ x : ℝ, deriv (f x c) x = (3 * x^2 - 4 * c * x + c^2) ∧ deriv (f x c) 2 = 0 
  ∧ (∀ x, x < 2 → deriv (f x c) x > 0)
  ∧ (∀ x, x > 2 → deriv (f x c) x < 0)) ↔ (c = 6) :=
by
  sorry

end find_c_for_local_max_at_2_l426_426028


namespace find_charge_per_minute_per_dog_l426_426237

-- Define the variables
variables (charge_per_dog : ℕ) (charge_per_minute_per_dog : ℕ)
variables (one_dog_minutes two_dogs_minutes three_dogs_minutes : ℕ)
variables (total_earned : ℕ)

-- Define the amounts for each walking scenario based on conditions
def one_dog_earning := charge_per_dog + one_dog_minutes * charge_per_minute_per_dog
def two_dogs_earning := 2 * (charge_per_dog + two_dogs_minutes * charge_per_minute_per_dog)
def three_dogs_earning := 3 * (charge_per_dog + three_dogs_minutes * charge_per_minute_per_dog)

-- Define the total earning calculation
def total_calculated_earning := one_dog_earning + two_dogs_earning + three_dogs_earning

-- Given conditions
axiom h1 : charge_per_dog = 20
axiom h2 : one_dog_minutes = 10
axiom h3 : two_dogs_minutes = 7
axiom h4 : three_dogs_minutes = 9
axiom h5 : total_earned = 171

-- The theorem to prove
theorem find_charge_per_minute_per_dog : charge_per_minute_per_dog = 1 :=
by
  -- The first step is to express the total calculated earnings using the given conditions
  unfold total_calculated_earning
  -- sorry, as no proof is required
  sorry

end find_charge_per_minute_per_dog_l426_426237


namespace P_ne_77_for_integers_l426_426346

def P (x y : ℤ) : ℤ :=
  x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_ne_77_for_integers (x y : ℤ) : P x y ≠ 77 :=
by
  sorry

end P_ne_77_for_integers_l426_426346


namespace closest_four_place_decimals_to_three_eleven_l426_426624

-- Define the conditions
def is_four_place_decimal (r : ℝ) : Prop :=
  0 ≤ r ∧ r < 1 ∧ ∃ a b c d : ℕ, r = (a * 10^(-1) + b * 10^(-2) + c * 10^(-3) + d * 10^(-4))

theorem closest_four_place_decimals_to_three_eleven : 
  {r : ℝ // is_four_place_decimal r ∧ 0.2614 ≤ r ∧ r ≤ 0.2864}.to_finset.card = 250 :=
sorry

end closest_four_place_decimals_to_three_eleven_l426_426624


namespace total_path_length_l426_426468

/-- Definition of a square with given side length -/
structure Square (side_length : ℝ) :=
  (A X Y Z : ℝ × ℝ)
  (side_lengths : ∀ (i : ℕ), (A ↔ X ↔ Y ↔ Z ↔ A))
  (sides_length_condition : ∀ i, dist (side_lengths i) ≤ side_length)

/-- Definition of the right triangle -/
structure RightTriangle (hypotenuse_length : ℝ) :=
  (A B P : ℝ × ℝ)
  (hypotenuse : dist A B = hypotenuse_length)
  (right_angle_condition : ∃ C, dist A C * dist B C = 0)

/-- Main theorem stating total length traversal of vertex P -/
theorem total_path_length (s : Square 5) (t : RightTriangle 3) (hp : dist t.B s.X ≤ 0 ∧ dist t.B t.A = 3) : 
  let path_length := 4 * (3 * (2 * π) / 4) in
  path_length = 6 * π :=
sorry

end total_path_length_l426_426468


namespace polar_line_through_centers_l426_426286

-- Definitions based on conditions from the problem
def circle_C1 (rho theta : ℝ) : Prop := rho = 2 * real.cos theta
def circle_C2 (rho theta : ℝ) : Prop := rho = 2 * real.sin theta

-- Mathematically equivalent proof problem
theorem polar_line_through_centers (rho : ℝ) (theta : ℝ) :
  circle_C1 rho theta → circle_C2 rho theta → theta = π / 4 :=
by
  intros _ _
  -- Direct proof omitted
  sorry

end polar_line_through_centers_l426_426286


namespace find_n_value_l426_426557

theorem find_n_value (n : ℤ) : (5^3 - 7 = 6^2 + n) ↔ (n = 82) :=
by
  sorry

end find_n_value_l426_426557


namespace factor_x4_minus_81_l426_426922

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intros x
  sorry

end factor_x4_minus_81_l426_426922


namespace factorize_x4_minus_81_l426_426933

theorem factorize_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factorize_x4_minus_81_l426_426933


namespace problem_statement_l426_426140

def op (x y : ℝ) : ℝ := (x + 3) * (y - 1)

theorem problem_statement (a : ℝ) : (∀ x : ℝ, op (x - a) (x + a) > -16) ↔ -2 < a ∧ a < 6 :=
by
  sorry

end problem_statement_l426_426140


namespace rainfall_difference_l426_426782

theorem rainfall_difference :
  let day1 := 26
  let day2 := 34
  let day3 := day2 - 12
  let total_rainfall := day1 + day2 + day3
  let average_rainfall := 140
  (average_rainfall - total_rainfall = 58) :=
by
  sorry

end rainfall_difference_l426_426782


namespace problem_statements_correct_l426_426110

theorem problem_statements_correct :
  (∀ f : ℝ → ℝ, ∀ x0 : ℝ, differentiable_at ℝ f x0 → is_local_extremum f x0 → deriv f x0 = 0) ∧
  (inductive_reasoning := "specific to general") ∧
  (deductive_reasoning := "general to specific") ∧
  (synthetic_method := "cause to effect") ∧
  (analytic_method := "effect to cause") :=
sorry

end problem_statements_correct_l426_426110


namespace convex_quad_inequality_l426_426976

open EuclideanGeometry

variables {A B C D P : Point} {AC BD AP DP CD AB : ℝ}
variable [ConvexQuadrilateral A B C D]

-- Conditions
variable {H1 : Intersect AC BD = P}
variable {H2 : ∠BAC + ∠BDC = 180}
variable {H3 : Distance A (Line BC) < Distance D (Line BC)}

-- Theorem statement
theorem convex_quad_inequality 
  (h1 : Intersect AC BD = P)
  (h2 : ∠BAC + ∠BDC = 180)
  (h3 : Distance A (Line BC) < Distance D (Line BC)) :
  (AC / BD)^2 > (AP * CD) / (DP * AB) := 
by 
  sorry

end convex_quad_inequality_l426_426976


namespace number_between_24_and_28_l426_426171

def is_between (x a b : ℕ) : Prop :=
  a < x ∧ x < b

def valid_numbers : set ℕ := {20, 23, 26, 29}

theorem number_between_24_and_28 : ∃ x ∈ valid_numbers, is_between x 24 28 :=
by
  use 26
  split
  · show 26 ∈ valid_numbers
    sorry
  · show is_between 26 24 28
    sorry

end number_between_24_and_28_l426_426171


namespace number_of_parallelograms_l426_426338

theorem number_of_parallelograms : 
  (∀ b d k : ℕ, k > 1 → k * b * d = 500000 → (b * d > 0 ∧ y = x ∧ y = k * x)) → 
  (∃ N : ℕ, N = 720) :=
sorry

end number_of_parallelograms_l426_426338


namespace geometric_sequence_seventh_term_l426_426845

theorem geometric_sequence_seventh_term (a1 : ℕ) (a6 : ℕ) (r : ℚ)
  (ha1 : a1 = 3) (ha6 : a1 * r^5 = 972) : 
  a1 * r^6 = 2187 := 
by
  sorry

end geometric_sequence_seventh_term_l426_426845


namespace jimmy_yellow_marbles_correct_l426_426697

def lorin_black_marbles : ℕ := 4
def alex_black_marbles : ℕ := 2 * lorin_black_marbles
def alex_total_marbles : ℕ := 19
def alex_yellow_marbles : ℕ := alex_total_marbles - alex_black_marbles
def jimmy_yellow_marbles : ℕ := 2 * alex_yellow_marbles

theorem jimmy_yellow_marbles_correct : jimmy_yellow_marbles = 22 := by
  sorry

end jimmy_yellow_marbles_correct_l426_426697


namespace polar_to_rectangular_l426_426901

theorem polar_to_rectangular :
  ∀ (r θ : ℝ), r = 6 → θ = π / 3 → 
  let x := r * Real.cos θ in
  let y := r * Real.sin θ in
  (x, y) = (3, 3 * Real.sqrt 3) :=
by
  intros r θ hr hθ
  simp [hr, hθ]
  sorry

end polar_to_rectangular_l426_426901


namespace quadratic_roots_property_l426_426621

theorem quadratic_roots_property (a b : ℝ)
  (h1 : a^2 - 2 * a - 1 = 0)
  (h2 : b^2 - 2 * b - 1 = 0)
  (ha_b_sum : a + b = 2)
  (ha_b_product : a * b = -1) :
  a^2 + 2 * b - a * b = 6 :=
sorry

end quadratic_roots_property_l426_426621


namespace equilateral_triangle_union_area_l426_426353

theorem equilateral_triangle_union_area :
  ∀ (s : ℝ), s = 2 → (let A := 6 * (sqrt 3 * (s * s) / 4) - 5 * (sqrt 3 * (1 * 1) / 4) 
                      in A = 4.75 * sqrt 3) :=
by
  -- Let s be the side length of the equilateral triangles
  intro s
  -- Given condition s = 2
  assume h₁ : s = 2
  -- Define individual and total area calculations
  let A := 6 * (sqrt 3 * (s * s) / 4) - 5 * (sqrt 3 * (1 * 1) / 4)
  -- Ensure resulting area is 4.75 * sqrt 3
  show A = 4.75 * sqrt 3
  sorry

end equilateral_triangle_union_area_l426_426353


namespace car_faster_than_truck_l426_426482

theorem car_faster_than_truck :
  (difference_in_speed (distance_truck : ℕ) (time_truck : ℕ) (distance_car : ℕ) (time_car : ℕ) :=
    (distance_car / time_car) - (distance_truck / time_truck))
  (distance_truck = 240) (time_truck = 8) (distance_car = 240) (time_car = 5) =
  18 := 
by
  sorry

end car_faster_than_truck_l426_426482


namespace polar_circle_l426_426737

def is_circle (ρ θ : ℝ) : Prop :=
  ρ = Real.cos (Real.pi / 4 - θ)

theorem polar_circle : 
  ∀ ρ θ : ℝ, is_circle ρ θ ↔ ∃ (x y : ℝ), (x - 1/(2 * Real.sqrt 2))^2 + (y - 1/(2 * Real.sqrt 2))^2 = (1/(2 * Real.sqrt 2))^2 :=
by
  intro ρ θ
  sorry

end polar_circle_l426_426737


namespace construct_right_triangle_l426_426137

noncomputable section

structure RightTriangle where
  a b c : ℝ -- sides of the triangle
  hypotenuse : a^2 + b^2 = c^2 -- Pythagorean theorem
  sa : ℝ -- median to leg a
  φ : ℝ -- angle between leg b and median sc

theorem construct_right_triangle (s_a φ : ℝ) : ∃ (T : RightTriangle), T.sa = s_a ∧ T.φ = φ :=
sorry

end construct_right_triangle_l426_426137


namespace flowers_per_vase_l426_426486

-- Definitions of conditions in Lean 4
def number_of_carnations : ℕ := 7
def number_of_roses : ℕ := 47
def total_number_of_flowers : ℕ := number_of_carnations + number_of_roses
def number_of_vases : ℕ := 9

-- Statement in Lean 4
theorem flowers_per_vase : total_number_of_flowers / number_of_vases = 6 := by
  unfold total_number_of_flowers
  show (7 + 47) / 9 = 6
  sorry

end flowers_per_vase_l426_426486


namespace ryan_chinese_hours_l426_426155

theorem ryan_chinese_hours (spent_english_per_day : ℕ) (total_days : ℕ) (total_english_hours : ℕ) 
  (spent_english_per_day = 6) (total_days = 2) (total_english_hours = 12) : 
  ∃ (spent_chinese_per_day : ℕ), true :=
by sorry

end ryan_chinese_hours_l426_426155


namespace total_cost_formula_optimal_scrapping_time_l426_426832

def purchaseCost : ℝ := 169000
def annualCost : ℝ := 10000
def firstYearMaintenance : ℝ := 1000
def increasePerYear : ℝ := 2000

def totalCost (n : ℕ) : ℝ := 1000 * (n : ℝ)^2 + 10000 * (n : ℝ) + 169000

def avgAnnualCost (n : ℕ) : ℝ := totalCost n / (n : ℝ)

theorem total_cost_formula (n : ℕ) :
  totalCost n = 1000 * (n : ℝ)^2 + 10000 * (n : ℝ) + 169000 :=
by sorry

theorem optimal_scrapping_time (n : ℕ) :
  (n : ℝ)^2 = 169 ↔ n = 13 :=
by sorry

end total_cost_formula_optimal_scrapping_time_l426_426832


namespace region_R_l426_426141

theorem region_R (a b : ℝ) :
  (∀ z : ℂ, z^2 + a*z + b = 0 → abs z < 1) ↔
  b > a - 1 ∧ b > -a - 1 ∧ b ≤ a^2 / 4 ∧ |a| < 2 :=
sorry

end region_R_l426_426141


namespace no_real_solution_l426_426144

theorem no_real_solution :
  ∀ x : ℝ, ((x - 4 * x + 15)^2 + 3)^2 + 1 ≠ -|x|^2 :=
by
  intro x
  sorry

end no_real_solution_l426_426144


namespace fractions_correct_negative_fractions_correct_integers_correct_positive_integers_correct_positive_rational_numbers_correct_l426_426156

-- Definitions of the sets
def set_of_fractions := {x | x ∈ ℚ}
def set_of_negative_fractions := {x | x ∈ ℚ ∧ x < 0}
def set_of_integers := {x | x ∈ ℤ}
def set_of_positive_integers := {x | x ∈ ℤ ∧ x > 0}
def set_of_positive_rational_numbers := {x | x ∈ ℚ ∧ x > 0}

-- List of numbers to be categorized
def numbers := [-2/9, -9, -301, -3.14, 2004, 0, 22/7]

-- Theorems asserting correct categorization
theorem fractions_correct :
  { -2/9, 22/7 } = { x | x ∈ set_of_fractions ∧ x ∈ numbers } := by sorry

theorem negative_fractions_correct :
  { -2/9 } = { x | x ∈ set_of_negative_fractions ∧ x ∈ numbers } := by sorry

theorem integers_correct :
  { -9, -301, 2004, 0 } = { x | x ∈ set_of_integers ∧ x ∈ numbers } := by sorry

theorem positive_integers_correct :
  { 2004 } = { x | x ∈ set_of_positive_integers ∧ x ∈ numbers } := by sorry

theorem positive_rational_numbers_correct :
  { 2004, 22/7 } = { x | x ∈ set_of_positive_rational_numbers ∧ x ∈ numbers } := by sorry

end fractions_correct_negative_fractions_correct_integers_correct_positive_integers_correct_positive_rational_numbers_correct_l426_426156


namespace pq_values_are_correct_l426_426315

noncomputable def pq_proof : Prop :=
  ∃ p q : ℝ, 
    (p + 3 * complex.I) ≠ (q + 6 * complex.I) ∧
    (p + 3 * complex.I ≠ q + 6 * complex.I) ∧
    (p + 3 * complex.I) * (p + 3 * complex.I) = (complex.mk 9 63) ∧
    (p + 3 * complex.I) + (q + 6 * complex.I) = (complex.mk 12 11) ∧
    p = 9 ∧
    q = 3

theorem pq_values_are_correct: pq_proof :=
sorry

end pq_values_are_correct_l426_426315


namespace servings_per_bottle_l426_426491

-- Definitions based on conditions
def total_guests : ℕ := 120
def servings_per_guest : ℕ := 2
def total_bottles : ℕ := 40

-- Theorem stating that given the conditions, the servings per bottle is 6
theorem servings_per_bottle : (total_guests * servings_per_guest) / total_bottles = 6 := by
  sorry

end servings_per_bottle_l426_426491


namespace shaded_area_l426_426134

-- Defining the conditions
def small_square_side := 4
def large_square_side := 12
def half_large_square_side := large_square_side / 2

-- DG is calculated as (12 / 16) * small_square_side = 3
def DG := (large_square_side / (half_large_square_side + small_square_side)) * small_square_side

-- Calculating area of triangle DGF
def area_triangle_DGF := (DG * small_square_side) / 2

-- Area of the smaller square
def area_small_square := small_square_side * small_square_side

-- Area of the shaded region
def area_shaded_region := area_small_square - area_triangle_DGF

-- The theorem stating the question
theorem shaded_area : area_shaded_region = 10 := by
  sorry

end shaded_area_l426_426134


namespace ratio_of_cookies_l426_426329

-- Definitions based on the conditions
def initial_cookies : ℕ := 19
def cookies_to_friend : ℕ := 5
def cookies_left : ℕ := 5
def cookies_eaten : ℕ := 2

-- Calculating the number of cookies left after giving cookies to the friend
def cookies_after_giving_to_friend := initial_cookies - cookies_to_friend

-- Maria gave to her family the remaining cookies minus the cookies she has left and she has eaten.
def cookies_given_to_family := cookies_after_giving_to_friend - cookies_eaten - cookies_left

-- The ratio to be proven 1:2, which is mathematically 1/2
theorem ratio_of_cookies : (cookies_given_to_family : ℚ) / (cookies_after_giving_to_friend : ℚ) = 1 / 2 := by
  sorry

end ratio_of_cookies_l426_426329


namespace part_I_part_II_l426_426696

namespace ProofProblem

def f (x a : ℝ) : ℝ := abs (x / 2 + 1 / (2 * a)) + abs (x / 2 - a / 2)

-- First proof: Prove that f(x) ≥ 1 for all x and a > 0
theorem part_I (x a : ℝ) (ha : a > 0) : f x a ≥ 1 := sorry

-- Second proof: Given f(6) < 5, find the range of values for a
theorem part_II (a : ℝ) (h : f 6 a < 5) : 1 + Real.sqrt 2 < a ∧ a < 5 + 2 * Real.sqrt 6 := sorry

end ProofProblem

end part_I_part_II_l426_426696


namespace flowers_per_vase_l426_426485

-- Definitions of conditions in Lean 4
def number_of_carnations : ℕ := 7
def number_of_roses : ℕ := 47
def total_number_of_flowers : ℕ := number_of_carnations + number_of_roses
def number_of_vases : ℕ := 9

-- Statement in Lean 4
theorem flowers_per_vase : total_number_of_flowers / number_of_vases = 6 := by
  unfold total_number_of_flowers
  show (7 + 47) / 9 = 6
  sorry

end flowers_per_vase_l426_426485


namespace curve_is_two_lines_l426_426370

theorem curve_is_two_lines (x y : ℝ) (h : x^2 + x * y = x) : (x = 0) ∨ (x + y = 1) :=
by {
  -- Next line simplifies the given equation
  have h' : x * (x + y - 1) = 0, {
    rw [mul_add x x y, ← add_sub_assoc, sub_self 1],
    exact h,
  },
  -- From the simplified equation x(x + y - 1) = 0, either x = 0 or x + y - 1 = 0
  exact eq_zero_or_eq_zero_of_mul_eq_zero h',
} sorry

end curve_is_two_lines_l426_426370


namespace find_certain_number_l426_426808

-- Define the conditions as constants
def n1 : ℕ := 9
def n2 : ℕ := 70
def n3 : ℕ := 25
def n4 : ℕ := 21
def smallest_given_number : ℕ := 3153
def certain_number : ℕ := 3147

-- Lean theorem statement
theorem find_certain_number (n1 n2 n3 n4 smallest_given_number certain_number: ℕ) :
  (∀ x, (∀ y ∈ [n1, n2, n3, n4], y ∣ x) → x ≥ smallest_given_number → x = smallest_given_number + certain_number) :=
sorry -- Skips the proof

end find_certain_number_l426_426808


namespace percent_greater_than_average_l426_426610

variable (M N : ℝ)

theorem percent_greater_than_average (h : M > N) :
  (200 * (M - N)) / (M + N) = ((M - ((M + N) / 2)) / ((M + N) / 2)) * 100 :=
by 
  sorry

end percent_greater_than_average_l426_426610


namespace correct_conclusions_l426_426683

-- Define the function f and its derivative
def f (m x : ℝ) : ℝ := sin x + m * x

def f' (m x : ℝ) : ℝ := cos x + m

-- Statements of the conclusions
def conclusion_1 (m : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' m x1 = f' m x2

def conclusion_3 (m : ℝ) : Prop :=
  ∃ (a : ℕ → ℝ), (∀ n : ℕ, f' m (a n) = f' m (a 0))

def conclusion_4 (m : ℝ) : Prop :=
  ∃ (a : ℕ → ℝ), (∀ n : ℕ, f' m (a n) = f' m (a 0))

-- The main theorem to be proven
theorem correct_conclusions (m : ℝ) : conclusion_1 m ∧ conclusion_3 m ∧ conclusion_4 m :=
by sorry

end correct_conclusions_l426_426683


namespace max_interior_angle_new_triangle_l426_426651

def Triangle (A B C : ℝ × ℝ) : Prop :=
  let dist (x y : ℝ × ℝ) := real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)
  dist A B = dist B C ∧ dist B C = dist C A

def PointOnSegment (P B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P.1 = B.1 + t * (C.1 - B.1) ∧ P.2 = B.2 + t * (C.2 - B.2)

theorem max_interior_angle_new_triangle {A B C P : ℝ × ℝ} (hABC : Triangle A B C) 
    (hP : PointOnSegment P B C) : 
    ∃ θ : ℝ, θ = 120 ∧ θ = max (angle A P B) (max (angle A P C) (angle B P C)) := 
sorry

end max_interior_angle_new_triangle_l426_426651


namespace find_cost_price_l426_426439

theorem find_cost_price (SP PP : ℝ) (hSP : SP = 600) (hPP : PP = 25) : 
  ∃ CP : ℝ, CP = 480 := 
by
  sorry

end find_cost_price_l426_426439


namespace perfect_square_m_value_l426_426632

theorem perfect_square_m_value (m : ℤ) :
  (∃ a : ℤ, ∀ x : ℝ, (x^2 + (m : ℝ)*x + 1 : ℝ) = (x + (a : ℝ))^2) → m = 2 ∨ m = -2 :=
by
  sorry

end perfect_square_m_value_l426_426632


namespace proposition_1_proposition_2_proposition_3_proposition_4_correct_propositions_l426_426978

-- Definitions for perpendicular line and plane
def is_perpendicular (l : Line) (α : Plane) : Prop := ∀ p : Point, p ∈ l → p ∈ α → false 

-- Definitions for parallel line and plane
def is_parallel (l : Line) (α : Plane) : Prop := ∀ p : Point, p ∈ l → ∑ p' ∈ α, ¬∃ m : p' ∈ l ⊆ α

-- Definitions for lines’ and planes’ relationships
variables (l m : Line) (α β : Plane)

-- The conditions given
variable (h1 : is_perpendicular l α)
variable (h2 : m ∈ β)

-- Propositions to prove
theorem proposition_1 (h : α ∥ β) : is_perpendicular l m := sorry
theorem proposition_2 (h : α ⊥ β) : ∥ l m := sorry
theorem proposition_3 (h : ∥ l m) : α ⊥ β := sorry
theorem proposition_4 (h : is_perpendicular l m) : α ∥ β := sorry

-- Combine to state that only (1) and (3) are true
theorem correct_propositions : 
(proposition_1 h1 h2) ∧ 
¬ (proposition_2 h1 h2) ∧ 
(proposition_3 h1 h2) ∧ 
¬ (proposition_4 h1 h2) := sorry

end proposition_1_proposition_2_proposition_3_proposition_4_correct_propositions_l426_426978


namespace factor_expression_l426_426527

variable (a : ℤ)

theorem factor_expression : 58 * a^2 + 174 * a = 58 * a * (a + 3) := by
  sorry

end factor_expression_l426_426527


namespace find_length_MN_l426_426288

-- Definitions based on conditions
variables {A B C L K M N : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space L] [metric_space K] [metric_space M] [metric_space N]
noncomputable theory
open_locale classical

def triangle_A_B_C (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : Prop :=
dist A B = 150 ∧ dist A C = 130 ∧ dist B C = 140

def is_angle_bisector (vertex point_on_side : Type) : Prop :=
true -- actual definition elided

def is_foot_of_perpendicular (vertex point_on_line foot : Type) : Prop :=
true -- actual definition elided

def midpoint (P Q : Type) [metric_space P] [metric_space Q] : Type := sorry -- placeholder

variables (P Q : Type) [metric_space P] [metric_space Q]
variables [dist : has_dist A] [dist : has_dist B] [dist : has_dist C] [dist : has_dist L] [dist : has_dist K] [dist : has_dist M] [dist : has_dist N]

-- Problem statement based on the conclusions from the solution
theorem find_length_MN :
  triangle_A_B_C A B C →
  is_angle_bisector A L →
  is_angle_bisector B K →
  is_foot_of_perpendicular C K M →
  is_foot_of_perpendicular C L N →
  dist M N = 60 :=
by sorry

end find_length_MN_l426_426288


namespace prove_LIK_BUBLIK_prove_CVK_BARUK_l426_426284

-- Definition for the first problem
def there_exists_unique_digit_assignment_LIK_BUBLIK (n m : ℕ) :=
  (n = 376) ∧ (m = 141376) ∧ (n * n = m)

-- Definition for the second problem
def there_exists_unique_digit_assignment_CVK_BARUK :=
  625 -- given __625__ as the valid digit assignment.

theorem prove_LIK_BUBLIK : ∃ (n m : ℕ), there_exists_unique_digit_assignment_LIK_BUBLIK n m := 
  by
    use [376, 141376]
    split; sorry

theorem prove_CVK_BARUK : there_exists_unique_digit_assignment_CVK_BARUK= 625 := 
  by sorry

end prove_LIK_BUBLIK_prove_CVK_BARUK_l426_426284


namespace construct_triangle_l426_426511

theorem construct_triangle
  (B C A1 D A : Type)
  (BC : B → C → Prop)
  (foot_A1 : B → C → A1 → Prop)
  (angle_diff : ∀ (A B D C : Type), β A B D - γ A D C = δ)
  (is_collinear : ∀ (B A1 D : Type), B ∈ line A1 D)
  (is_circle_with_chord : ∀ (C D : Type), ∃ (circle : Type), is_chord circle C D ∧ ∀ (P : Type), inscribed_angle circle P = δ) :
  ∃ (A : Type), (triangle B C A ∧ height B C A1 A ∧ angle A B C - angle A C B = δ).
Proof := sorry

end construct_triangle_l426_426511


namespace smallest_m_divisible_by_7_l426_426950

theorem smallest_m_divisible_by_7 :
  ∃ m : ℕ, m = 6 ∧ (m^3 + 3^m) % 7 = 0 ∧ (m^2 + 3^m) % 7 = 0 ∧ 
  ∀ n : ℕ, 0 < n < 6 → ((n^3 + 3^n) % 7 ≠ 0 ∨ (n^2 + 3^n) % 7 ≠ 0) :=
by
  sorry

end smallest_m_divisible_by_7_l426_426950


namespace arrangement_count_l426_426277

theorem arrangement_count : 
  let math_books := 4 in
  let history_books := 6 in
  let arrangements := 
    math_books * (math_books - 1) *
    (Nat.choose history_books 2) *
    (Nat.factorial (history_books - 1)) *
    (Nat.choose (history_books - 1) 2) *
    (Nat.factorial 2)
  in
  arrangements = 518400 := 
by
  let math_books := 4
  let history_books := 6
  let arrangements := 
    math_books * (math_books - 1) * 
    (Nat.choose history_books 2) *
    (Nat.factorial (history_books - 1)) *
    (Nat.choose (history_books - 1) 2) *
    (Nat.factorial 2)
  show arrangements = 518400 from sorry

end arrangement_count_l426_426277


namespace ticket_cost_l426_426120

-- Let x be the price of an adult ticket
variable (x : ℝ)

-- Define child ticket price
def child_ticket_price : ℝ := x / 2

-- Pricing equation for 6 adult and 8 child tickets
def ticket_equation : Prop := (6 * x + 8 * child_ticket_price x = 46.50)

-- Total cost for 10 adult and 15 child tickets should be $81.375
theorem ticket_cost (h : ticket_equation x) : 10 * x + 15 * child_ticket_price x = 81.375 :=
sorry

end ticket_cost_l426_426120


namespace no_real_x_condition_l426_426542

theorem no_real_x_condition (x : ℝ) : 
(∃ a b : ℕ, 4 * x^5 - 7 = a^2 ∧ 4 * x^13 - 7 = b^2) → false := 
by {
  sorry
}

end no_real_x_condition_l426_426542


namespace arithmetic_sequence_geometric_subsequence_l426_426677

theorem arithmetic_sequence_geometric_subsequence (a b : ℝ) (h : b ≠ 0) :
  (∃ (f : ℕ → ℕ), strict_mono f ∧ (∃ k : ℝ, ∀ n : ℕ, a + (f n) * b = a * (k ^ n))) ↔ rational (a / b) := 
begin
  sorry
end

end arithmetic_sequence_geometric_subsequence_l426_426677


namespace proof_intersection_l426_426325

open Set

variable (U P Q : Set ℕ)
variable (complement_U_Q : Set ℕ)

-- Define the sets
def universal_set := {1, 2, 3, 4, 5, 6}
def P_set := {1, 2, 3, 4}
def Q_set := {3, 4, 5}

-- Complement of Q with respect to U
def complement_U_of_Q := universal_set \ Q_set

-- The intersection we want to prove
theorem proof_intersection :
  P_set ∩ complement_U_of_Q = {1, 2} :=
sorry

end proof_intersection_l426_426325


namespace find_h_l426_426638

def quadratic_expr : ℝ → ℝ := λ x, 3 * x^2 + 9 * x + 20

theorem find_h : ∃ (a k : ℝ) (h : ℝ), 
  (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 :=
sorry

end find_h_l426_426638


namespace carla_initial_marbles_l426_426129

theorem carla_initial_marbles
  (marbles_bought : ℕ)
  (total_marbles_now : ℕ)
  (h1 : marbles_bought = 134)
  (h2 : total_marbles_now = 187) :
  total_marbles_now - marbles_bought = 53 :=
by
  sorry

end carla_initial_marbles_l426_426129


namespace discount_rate_on_pony_jeans_is_15_l426_426813

noncomputable def discountProblem : Prop :=
  ∃ (F P : ℝ),
    (15 * 3 * F / 100 + 18 * 2 * P / 100 = 8.55) ∧ 
    (F + P = 22) ∧ 
    (P = 15)

theorem discount_rate_on_pony_jeans_is_15 : discountProblem :=
sorry

end discount_rate_on_pony_jeans_is_15_l426_426813


namespace bd_greater_than_ac_l426_426070

theorem bd_greater_than_ac {A B C D : Point}
    (h1 : lineSegment A C)
    (h2 : oppositeSides B D A C)
    (h3 : distance A B = distance C D)
    (h4 : angle A C B + angle A C D = 180) :
    distance B D > distance A C :=
sorry

end bd_greater_than_ac_l426_426070


namespace children_count_l426_426399

-- Define the total number of passengers on the airplane
def total_passengers : ℕ := 240

-- Define the ratio of men to women
def men_to_women_ratio : ℕ × ℕ := (3, 2)

-- Define the percentage of passengers who are either men or women
def percent_men_women : ℕ := 60

-- Define the number of children on the airplane
def number_of_children (total : ℕ) (percent : ℕ) : ℕ := 
  (total * (100 - percent)) / 100

theorem children_count :
  number_of_children total_passengers percent_men_women = 96 := by
  sorry

end children_count_l426_426399


namespace committee_selection_l426_426835

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_selection :
  let seniors := 10
  let members := 30
  let non_seniors := members - seniors
  let choices := binom seniors 2 * binom non_seniors 3 +
                 binom seniors 3 * binom non_seniors 2 +
                 binom seniors 4 * binom non_seniors 1 +
                 binom seniors 5
  choices = 78552 :=
by
  sorry

end committee_selection_l426_426835


namespace part1_a_gt_1_part1_0_lt_a_lt_1_part2_l426_426604

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  log a (1 - 2 / (x + 1))

theorem part1_a_gt_1 {a : ℝ} (ha : a > 1) (x : ℝ) :
  f x a > 0 → x < -1 :=
sorry

theorem part1_0_lt_a_lt_1 {a : ℝ} (ha : 0 < a) (ha_ne_1 : a < 1) (x : ℝ) :
  f x a > 0 → x > 1 :=
sorry

theorem part2 {m n : ℝ} (hmn : m < n) :
  ∃ a : ℝ, 0 < a ∧ a < (3 - 2 * real.sqrt 2) / 2 ∧ ∀ x ∈ set.Icc m n, f x a ∈ set.Icc (1 + log a (2 * n)) (1 + log a (2 * m)) :=
sorry

end part1_a_gt_1_part1_0_lt_a_lt_1_part2_l426_426604


namespace minimum_reflection_number_l426_426849

theorem minimum_reflection_number (a b : ℕ) :
  ((a + 2) * (b + 2) = 4042) ∧ (Nat.gcd (a + 1) (b + 1) = 1) → 
  (a + b = 129) :=
sorry

end minimum_reflection_number_l426_426849


namespace probability_sum_divisible_by_3_l426_426560

theorem probability_sum_divisible_by_3 (is_mod_3 : ℤ → ℤ)
    (count_33_numbers : card {n : ℤ | (1 ≤ n ∧ n ≤ 100) ∧ is_mod_3 n = 0} = 33)
    (count_34_numbers : card {n : ℤ | (1 ≤ n ∧ n ≤ 100) ∧ is_mod_3 n = 1} = 34)
    (count_33_numbers_2 : card {n : ℤ | (1 ≤ n ∧ n ≤ 100) ∧ is_mod_3 n = 2} = 33) :
    let favorable_outcomes := (choose 33 3) + (choose 34 3) + (choose 33 3) + (33 * 34 * 33)
    let total_outcomes := choose 100 3
    (favorable_outcomes : ℚ) / total_outcomes = 817 / 2450 :=
by
  sorry

end probability_sum_divisible_by_3_l426_426560


namespace ball_acceleration_before_hit_l426_426429

variable (V1 V2 a2 m g : ℝ)

theorem ball_acceleration_before_hit (h1 : V1 > 0) (h2 : V2 > 0) (h3 : m > 0) (h4 : g > 0) : 
  let a1 := (V1 / V2)^2 * (a2 - g) in 
  a1 = (V1 / V2)^2 * (a2 - g) := 
by 
  sorry

end ball_acceleration_before_hit_l426_426429


namespace logical_reasoning_sound_l426_426433

-- Conditions
def condition1 : Prop := ∀ (S : Type), S = "Sphere" → "S has properties".
def condition2 : Prop := ∀ (T : Type), T = "Triangle" → "Sum of interior angles of T = 180 degrees".
def condition3 : Prop := ∀ (C : Type), C = "Classroom of chairs" → "One chair is broken in C → All chairs are broken in C".
def condition4 : Prop := ∀ (P : Type), P = "Polygon" → "Sum of interior angles of P = (n-2) * 180 degrees".

-- Logical reasoning
def logical_reasoning (c1 c2 c3 c4 : Prop) : list Prop :=
  [if c1 then "Condition 1 is analogy reasoning" else "Condition 1 is not logical",
   if c2 then "Condition 2 is inductive reasoning" else "Condition 2 is not logical",
   if c3 then "Condition 3 is not logical" else "Condition 3 is logical",
   if c4 then "Condition 4 is inductive reasoning" else "Condition 4 is not logical"]

-- Prove that logical reasoning includes conditions 1, 2, and 4
theorem logical_reasoning_sound : 
  ∀ (c1 c2 c3 c4 : Prop), c1 → c2 → ¬c3 → c4 → logical_reasoning c1 c2 c3 c4 = ["Condition 1 is analogy reasoning", "Condition 2 is inductive reasoning", "Condition 3 is not logical", "Condition 4 is inductive reasoning"] :=
by
  intros;
  sorry  -- Only statement required, not proof.

end logical_reasoning_sound_l426_426433


namespace max_volume_of_tetrahedron_l426_426193

open Real

-- Definition of conditions
variables (O S A B C : Point) (r : ℝ)
-- O, A, B, and C are coplanar
def coplanar (O A B C : Point) : Prop := 
  ∃ (α β γ : ℝ), α * O.x + β * A.x + γ * B.x = 0
  ∧ α * O.y + β * A.y + γ * B.y = 0
  ∧ α * O.z + β * A.z + γ * B.z = 0

-- Triangle ABC is equilateral with side length 2
def equilateral_triangle (A B C : Point) (len : ℝ) : Prop :=
  dist A B = len ∧ dist B C = len ∧ dist C A = len

-- Plane SAB is perpendicular to plane ABC
def plane_perpendicular (S A B C : Point) : Prop :=
  (A - S) ⬝ (B - S) = 0 ∧ (A - S) ⬝ (C - S) = 0

-- Define volume of tetrahedron
noncomputable def volume_of_tetrahedron (S A B C : Point) : ℝ :=
  1 / 6 * abs ((B - A) ⬝ ((C - A) × (S - A)))

-- Problem Statement
theorem max_volume_of_tetrahedron
  (h1 : coplanar O A B C)
  (h2 : equilateral_triangle A B C 2)
  (h3 : plane_perpendicular S A B C)
  (h4 : dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O S = r) :
  volume_of_tetrahedron S A B C = sqrt 3 / 3 :=
sorry

end max_volume_of_tetrahedron_l426_426193


namespace equation_of_perpendicular_line_l426_426979

theorem equation_of_perpendicular_line (x y : ℝ) (l1 : 2*x - 3*y + 4 = 0) (pt : x = -2 ∧ y = -3) :
  3*(-2) + 2*(-3) + 12 = 0 := by
  sorry

end equation_of_perpendicular_line_l426_426979


namespace number_of_subsets_M_l426_426312

theorem number_of_subsets_M :
  let M := {x : ℤ | x^2 < 100 ∧ 100 < 2^x} in
  set.card (set.powerset M) = 8 := by
  sorry

end number_of_subsets_M_l426_426312


namespace reflection_example_l426_426093

noncomputable def reflection (u v w : ℝ × ℝ) : ℝ × ℝ :=
  let midpoint := ((u.1 + v.1) / 2, (u.2 + v.2) / 2)
  let dir := (midpoint.1 - u.1, midpoint.2 - u.2)
  let w_proj := ((w.1 * dir.1 + w.2 * dir.2) / (dir.1 * dir.1 + dir.2 * dir.2), 
                 (w.1 * dir.1 + w.2 * dir.2) / (dir.1 * dir.1 + dir.2 * dir.2))
  let projection := (w_proj.1 * dir.1, w_proj.2 * dir.2)
  (2 * projection.1 - w.1, 2 * projection.2 - w.2)

theorem reflection_example :
  reflection (2, 6) (4, -4) (1, 4) = (3.2, -2.6) :=
by
  sorry

end reflection_example_l426_426093


namespace C1_equation_l426_426648

noncomputable def equations_of_curves
    (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
    (phi theta : ℝ) (M : ℝ × ℝ) (D : ℝ × ℝ)
    (M_on_C1 : 2 = a * cos (π / 3) ∧ sqrt 3 = b * sin (π / 3))
    (D_on_C2 : sqrt 2 = 2 * (theta / 4)) :
  (a = 4 ∧ b = 2) ∧
  (∃ R : ℝ, R = 1 ∧ (√2 = 2 * R * √2 / 2) ∧ (x - R)^2 + y^2 = R^2) :=
sorry

theorem C1_equation (rho1 rho2 theta : ℝ)
    (rho1_pos : rho1 > 0) (rho2_pos : rho2 > 0)
    (A_on_C1 : (rho1^2 * cos^2 theta) / 16 + (rho1^2 * sin^2 theta) / 4 = 1)
    (B_on_C1 : (rho2^2 * sin^2 theta) / 16 + (rho2^2 * cos^2 theta) / 4 = 1) :
  (1 / rho1^2) + (1 / rho2^2) = 5 / 16 :=
sorry

end C1_equation_l426_426648


namespace max_sum_after_swap_l426_426714

section
variables (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ)
  (h1 : 100 * a1 + 10 * b1 + c1 + 100 * a2 + 10 * b2 + c2 + 100 * a3 + 10 * b3 + c3 = 2019)
  (h2 : 1 ≤ a1 ∧ a1 ≤ 9 ∧ 0 ≤ b1 ∧ b1 ≤ 9 ∧ 0 ≤ c1 ∧ c1 ≤ 9)
  (h3 : 1 ≤ a2 ∧ a2 ≤ 9 ∧ 0 ≤ b2 ∧ b2 ≤ 9 ∧ 0 ≤ c2 ∧ c2 ≤ 9)
  (h4 : 1 ≤ a3 ∧ a3 ≤ 9 ∧ 0 ≤ b3 ∧ b3 ≤ 9 ∧ 0 ≤ c3 ∧ c3 ≤ 9)

theorem max_sum_after_swap : 100 * c1 + 10 * b1 + a1 + 100 * c2 + 10 * b2 + a2 + 100 * c3 + 10 * b3 + a3 ≤ 2118 := 
  sorry

end

end max_sum_after_swap_l426_426714


namespace range_of_m_l426_426973

noncomputable def isEllipse (m : ℝ) : Prop := (m^2 > 2 * m + 8) ∧ (2 * m + 8 > 0)
noncomputable def intersectsXAxisAtTwoPoints (m : ℝ) : Prop := (2 * m - 3)^2 - 1 > 0

theorem range_of_m (m : ℝ) :
  ((m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∨ (2 * m - 3)^2 - 1 > 0) ∧
  ¬ (m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∧ (2 * m - 3)^2 - 1 > 0)) →
  (m ≤ -4 ∨ (-2 ≤ m ∧ m < 1) ∨ (2 < m ∧ m ≤ 4)) :=
by sorry

end range_of_m_l426_426973


namespace find_b_of_square_polynomial_l426_426754

theorem find_b_of_square_polynomial 
  (a b : ℚ)
  (h : ∃ p q : ℚ, (x^4 + x^3 - x^2 + a * x + b) = (x^2 + p * x + q)^2) :
  b = 25 / 64 :=
by 
  cases h with p hp
  cases hp with q hq
  sorry 

end find_b_of_square_polynomial_l426_426754


namespace value_of_x_l426_426256

theorem value_of_x :
  ∃ x : ℕ, x > 0 ∧ 1^(x+3) + 3^(x+1) + 4^(x-1) + 5^x = 3786 ∧ x = 5 :=
by {
  use 5,
  split, exact nat.succ_pos 4,
  split, sorry,
  refl,
}

end value_of_x_l426_426256


namespace fraction_problem_l426_426253

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end fraction_problem_l426_426253


namespace number_of_ordered_triples_l426_426631

theorem number_of_ordered_triples (a b c : ℕ) (h : (a > 0) ∧ (b > 0) ∧ (c > 0)) :
  ((a / c + a / b + 1) / (b / a + b / c + 1) = 11 ∧ a + 2 * b + c ≤ 40) →
  ((∃ s : ℕ, s = 42) :=
begin
  sorry
end

end number_of_ordered_triples_l426_426631


namespace Janna_sleep_weekend_each_day_l426_426294

variable (weekday_hours weekend_days total_sleep_hours : ℕ)

def total_weekday_sleep_hours : ℕ := weekday_hours * 5
def total_weekend_sleep_hours : ℕ := total_sleep_hours - total_weekday_sleep_hours

theorem Janna_sleep_weekend_each_day :
  weekday_hours = 7 → total_sleep_hours = 51 → weekend_days = 2 →
  (total_weekend_sleep_hours / weekend_days) = 8 := by
  intros h1 h2 h3
  unfold total_weekday_sleep_hours total_weekend_sleep_hours
  rw [h1,h2]
  simp
  sorry

end Janna_sleep_weekend_each_day_l426_426294


namespace problem_probability_of_40_cents_l426_426361

-- Definitions
def coin (penny nickel dime quarter half_dollar : bool) : ℤ :=
  (if penny then 1 else 0) + (if nickel then 5 else 0) +
  (if dime then 10 else 0) + (if quarter then 25 else 0) +
  (if half_dollar then 50 else 0)

def successful_outcomes (cfg : (bool × bool × bool × bool × bool)) : bool :=
  coin cfg.1 cfg.2.1 cfg.2.2.1 cfg.2.2.2.1 cfg.2.2.2.2 ≥ 40

noncomputable def probability_of_success : ℚ :=
  let outcomes := finset.univ.product (finset.univ.product (finset.univ.product (finset.univ.product finset.univ)))
  let success_count := (outcomes.filter successful_outcomes).card
  success_count / outcomes.card

-- Theorem statement (no proof provided)
theorem problem_probability_of_40_cents :
  probability_of_success = 9 / 16 :=
sorry

end problem_probability_of_40_cents_l426_426361


namespace limit_S_ratio_l426_426986

noncomputable def a_seq (a₁ p : ℝ) : ℕ → ℝ
| 0       => a₁
| (n + 1) => a_seq n * p

noncomputable def b_seq (b₁ q : ℝ) : ℕ → ℝ
| 0       => b₁
| (n + 1) => b_seq n * q

noncomputable def c_seq (a₁ p b₁ q : ℝ) (n : ℕ) : ℝ :=
a_seq a₁ p n + b_seq b₁ q n

noncomputable def S (a₁ p b₁ q : ℝ) : ℕ → ℝ
| 0       => 0
| (n + 1) => S n + c_seq a₁ p b₁ q n

theorem limit_S_ratio (a₁ b₁ p q : ℝ) (hpq : p > q) (hp1 : p ≠ 1) (hq1 : q ≠ 1) :
  (if p > 1 then (lim_n (λ n, (S a₁ p b₁ q n / S a₁ p b₁ q (n - 1))) = p)
  else (lim_n (λ n, (S a₁ p b₁ q n / S a₁ p b₁ q (n - 1))) = 1)) :=
sorry

end limit_S_ratio_l426_426986


namespace tournament_byes_and_games_l426_426276

/-- In a single-elimination tournament with 300 players initially registered,
- if the number of players in each subsequent round must be a power of 2,
- then 44 players must receive a bye in the first round, and 255 total games
- must be played to determine the champion. -/
theorem tournament_byes_and_games :
  let initial_players := 300
  let pow2_players := 256
  44 = initial_players - pow2_players ∧
  255 = pow2_players - 1 :=
by
  let initial_players := 300
  let pow2_players := 256
  have h_byes : 44 = initial_players - pow2_players := by sorry
  have h_games : 255 = pow2_players - 1 := by sorry
  exact ⟨h_byes, h_games⟩

end tournament_byes_and_games_l426_426276


namespace inv_7_mod_45_l426_426157

theorem inv_7_mod_45 : ∃ x, 0 ≤ x ∧ x < 45 ∧ (7 * x) % 45 = 1 :=
by
  use 32
  split
  exact zero_le 32
  split
  linarith
  norm_num
  sorry

end inv_7_mod_45_l426_426157


namespace eval_at_pi_over_12_l426_426998

def f (x : ℝ) : ℝ := sqrt 3 * sin x + cos x

theorem eval_at_pi_over_12 : f (π / 12) = sqrt 2 := by
  sorry

end eval_at_pi_over_12_l426_426998


namespace nonzero_rational_pow_zero_l426_426417

theorem nonzero_rational_pow_zero 
  (num : ℤ) (denom : ℤ) (hnum : num = -1241376497) (hdenom : denom = 294158749357) (h_nonzero: num ≠ 0 ∧ denom ≠ 0) :
  (num / denom : ℚ) ^ 0 = 1 := 
by 
  sorry

end nonzero_rational_pow_zero_l426_426417


namespace prove_f_neg1_eq_0_l426_426586

def f : ℝ → ℝ := sorry

theorem prove_f_neg1_eq_0
  (h1 : ∀ x : ℝ, f(x + 2) = f(2 - x))
  (h2 : ∀ x : ℝ, f(1 - 2 * x) = -f(2 * x + 1))
  : f(-1) = 0 := sorry

end prove_f_neg1_eq_0_l426_426586


namespace trig_identity_l426_426179

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : 
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 :=
by
  sorry

end trig_identity_l426_426179


namespace fourth_power_of_nested_sqrt_l426_426886

noncomputable def nested_sqrt : ℝ :=
  real.sqrt (1 + real.sqrt (1 + real.sqrt (1 + real.sqrt (1))))

theorem fourth_power_of_nested_sqrt :
  (nested_sqrt ^ 4) = 6 + 2 * real.sqrt 5 :=
by
  sorry

end fourth_power_of_nested_sqrt_l426_426886


namespace sector_area_l426_426469

theorem sector_area (R : ℝ) (hR_pos : R > 0) (h_circumference : 4 * R = 2 * R + arc_length) :
  (1 / 2) * arc_length * R = R^2 :=
by sorry

end sector_area_l426_426469


namespace Felicity_used_23_gallons_l426_426532

variable (A Felicity : ℕ)
variable (h1 : Felicity = 4 * A - 5)
variable (h2 : A + Felicity = 30)

theorem Felicity_used_23_gallons : Felicity = 23 := by
  -- Proof steps would go here
  sorry

end Felicity_used_23_gallons_l426_426532


namespace range_of_a_when_increasing_l426_426565

theorem range_of_a_when_increasing (a : ℝ) 
(h1: (∀ x y : ℝ, x < y ∧ x < 1 ∧ y < 1 → (5 - a) * x - 4 * a ≤ (5 - a) * y - 4 * a) ∧
 h2: (∀ x y : ℝ, x < y ∧ x ≥ 1 ∧ y ≥ 1 → a^x ≤ a^y) ∧
 h3: (∀ x : ℝ, x < 1 → (5 - a) * x - 4 * a ≤ a^1)) : 1 < a ∧ a < 5 :=
  sorry

end range_of_a_when_increasing_l426_426565


namespace sum_of_squares_of_medians_l426_426053

theorem sum_of_squares_of_medians (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  let ma := √((2 * b^2 + 2 * c^2 - a^2) / 4),
      mb := √((2 * a^2 + 2 * c^2 - b^2) / 4),
      mc := √((2 * a^2 + 2 * b^2 - c^2) / 4) in
  ma^2 + mb^2 + mc^2 = 337.5 :=
by
  sorry

end sum_of_squares_of_medians_l426_426053


namespace range_of_sqrt_meaningful_l426_426620

theorem range_of_sqrt_meaningful (x : ℝ) (h : sqrt(x + 1) ≥ 0) : x ≥ -1 :=
sorry

end range_of_sqrt_meaningful_l426_426620


namespace max_integer_value_of_x_l426_426777

theorem max_integer_value_of_x (x : ℤ) : 3 * x - (1 / 4 : ℚ) ≤ (1 / 3 : ℚ) * x - 2 → x ≤ -1 :=
by
  intro h
  sorry

end max_integer_value_of_x_l426_426777


namespace factor_x4_minus_81_l426_426925

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intros x
  sorry

end factor_x4_minus_81_l426_426925


namespace hyperbola_distances_l426_426958

theorem hyperbola_distances {a b x0 y0 : ℝ} (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
  (F1 F2 : ℝ) 
  (h4 : |F1 - F2| = 10) 
  (P : ℝ) 
  (h5 : |P - F1| - |P - F2| = 6) 
  (d1 d2 : ℝ) 
  (h6 : d1 = |(4 / 3) * x0 - y0| / sqrt ((4 / 3)^2 + 1))
  (h7 : d2 = |-(4 / 3) * x0 - y0| / sqrt ((4 / 3)^2 + 1)) : 
  sqrt (d1 * d2) = 12 / 5 := 
sorry

end hyperbola_distances_l426_426958


namespace hypergeometric_distribution_problems_l426_426109

theorem hypergeometric_distribution_problems :
  (∀ (P1 P2 : Prop), (P1 → ¬(P2)) → (P2 → ¬(P1)) → (P3 : Prop), P3)
  (P1 := ∀ (X : Type), (throwing_dice : X -> ℕ) -> independent_repeated_trials throwing_dice)
  (P2 := ∀ (X : Type), (germination_experiment : X -> ℕ) -> independent_repeated_trials germination_experiment)
  (P3 := ∀ (X : Type), (hypergeometric : X -> ℕ) -> problems_3_and_4 hypergeometric)
  (P4 := ∀ (X : Type), (hypergeometric : X -> ℕ) -> problems_3_and_4 hypergeometric),
  problems_3_and_4 P3 /\ problems_3_and_4 P4 := by
  sorry

end hypergeometric_distribution_problems_l426_426109


namespace monotonic_decreasing_interval_l426_426750

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem monotonic_decreasing_interval : 
  ∀ x, -1 < x ∧ x < 3 →  (deriv function_y x < 0) :=
by
  sorry

end monotonic_decreasing_interval_l426_426750


namespace jerry_mowing_income_l426_426670

theorem jerry_mowing_income (M : ℕ) (week_spending : ℕ) (money_weed_eating : ℕ) (weeks : ℕ)
  (H1 : week_spending = 5)
  (H2 : money_weed_eating = 31)
  (H3 : weeks = 9)
  (H4 : (M + money_weed_eating) = week_spending * weeks)
  : M = 14 :=
by {
  sorry
}

end jerry_mowing_income_l426_426670


namespace solve_seating_problem_l426_426398

-- Define the conditions of the problem
def valid_seating_arrangements (n : ℕ) : Prop :=
  (∃ (x y : ℕ), x < y ∧ x + 1 < y ∧ y < n ∧ 
    (n ≥ 5 ∧ y - x - 1 > 0)) ∧
  (∃! (x' y' : ℕ), x' < y' ∧ x' + 1 < y' ∧ y' < n ∧ 
    (n ≥ 5 ∧ y' - x' - 1 > 0))

-- State the theorem
theorem solve_seating_problem : ∃ n : ℕ, valid_seating_arrangements n ∧ n = 5 :=
by
  sorry

end solve_seating_problem_l426_426398


namespace probability_sum_at_least_four_l426_426722

-- Define the list of the amounts in the red packet
def amounts : List ℝ := [1.49, 1.31, 2.19, 3.40, 0.61]

-- Define the condition for A and B snatching amounts >= 4 yuan
def sum_at_least_four (a b : ℝ) : Prop :=
  a + b ≥ 4

-- Define the main theorem to prove the probability
theorem probability_sum_at_least_four : 
  let valid_pairs : List (ℝ × ℝ) := [(0.61, 3.40), (1.49, 3.40), (1.31, 3.40), (2.19, 3.40)] in
  list.length valid_pairs = 4 →
  (∀ a b, (a, b) ∈ valid_pairs → sum_at_least_four a b) →
  ∃ total_events : ℕ, total_events = 10 ∧ 
    ∃ favorable_events : ℕ, favorable_events = 4 ∧
      (favorable_events.toFloat / total_events.toFloat = (2 : ℝ) / 5) :=
by
  sorry

end probability_sum_at_least_four_l426_426722


namespace rainfall_difference_l426_426787

noncomputable def r₁ : ℝ := 26
noncomputable def r₂ : ℝ := 34
noncomputable def r₃ : ℝ := r₂ - 12
noncomputable def avg : ℝ := 140

theorem rainfall_difference : (avg - (r₁ + r₂ + r₃)) = 58 := 
by
  sorry

end rainfall_difference_l426_426787


namespace values_of_m_l426_426947

theorem values_of_m (m : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + (2 - m) * x + 12 = 0)) ↔ (m = -10 ∨ m = 14) := 
by
  sorry

end values_of_m_l426_426947


namespace find_b_of_square_polynomial_l426_426753

theorem find_b_of_square_polynomial 
  (a b : ℚ)
  (h : ∃ p q : ℚ, (x^4 + x^3 - x^2 + a * x + b) = (x^2 + p * x + q)^2) :
  b = 25 / 64 :=
by 
  cases h with p hp
  cases hp with q hq
  sorry 

end find_b_of_square_polynomial_l426_426753


namespace sequence_finitely_many_primes_l426_426870

theorem sequence_finitely_many_primes (n : ℤ) : 
  ∃ (a : ℕ → ℕ), (∀ i, a i > 0) ∧ (∀ i j, i < j → a i < a j) ∧ 
    (∃ N, ∀ k > N, ¬ prime (a k + n)) := sorry

end sequence_finitely_many_primes_l426_426870


namespace houses_per_block_correct_l426_426457

-- Define the conditions
def total_mail_per_block : ℕ := 32
def mail_per_house : ℕ := 8

-- Define the correct answer
def houses_per_block : ℕ := 4

-- Theorem statement
theorem houses_per_block_correct (total_mail_per_block mail_per_house : ℕ) : 
  total_mail_per_block = 32 →
  mail_per_house = 8 →
  total_mail_per_block / mail_per_house = houses_per_block :=
by
  intros h1 h2
  sorry

end houses_per_block_correct_l426_426457


namespace sum_of_special_sequence_l426_426194

-- Definition of an arithmetic sequence
def arithmetic_seq (a : ℕ → ℕ) (a1 : ℕ) (d : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a1 + n * d

-- Sum of the first n terms of the arithmetic sequence
def sum_seq (a S : ℕ → ℕ) (n : ℕ) : Prop :=
S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2

-- Conditions
variables {a : ℕ → ℕ} {S : ℕ → ℕ}
axiom a5_eq_5 : a 5 = 5
axiom S5_eq_15 : S 5 = 15

-- Problem to prove
theorem sum_of_special_sequence : 
  (∃ a d : ℕ, arithmetic_seq a a 1 d ∧ ∀ (S : ℕ → ℕ), sum_seq a S n → 
  (∑ k in range 100, 1 / (a k * a (k + 1)) = 100 / 101)) := 
sorry

end sum_of_special_sequence_l426_426194


namespace senior_ticket_cost_l426_426799

variable (tickets_total : ℕ)
variable (adult_ticket_price senior_ticket_price : ℕ)
variable (total_receipts : ℕ)
variable (senior_tickets_sold : ℕ)

theorem senior_ticket_cost (h1 : tickets_total = 529) 
                           (h2 : adult_ticket_price = 25)
                           (h3 : total_receipts = 9745)
                           (h4 : senior_tickets_sold = 348) 
                           (h5 : senior_ticket_price * 348 + 25 * (529 - 348) = 9745) : 
                           senior_ticket_price = 15 := by
  sorry

end senior_ticket_cost_l426_426799


namespace equal_vectors_implies_collinear_l426_426146

-- Definitions for vectors and their properties
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u 

def equal_vectors (u v : V) : Prop := u = v

theorem equal_vectors_implies_collinear (u v : V)
  (h : equal_vectors u v) : collinear u v :=
by sorry

end equal_vectors_implies_collinear_l426_426146


namespace distance_between_points_l426_426545

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_between_points (λ : ℝ) :
  distance (4, -2, 3) (2, 4, -7 + λ) = real.sqrt (40 + (λ - 10)^2) :=
by
  sorry

end distance_between_points_l426_426545


namespace total_loss_is_correct_l426_426117

variable (A P : ℝ)
variable (Ashok_loss Pyarelal_loss : ℝ)

-- Condition 1: Ashok's capital is 1/9 of Pyarelal's capital
def ashokCapital (A P : ℝ) : Prop :=
  A = (1 / 9) * P

-- Condition 2: Pyarelal's loss was Rs 1800
def pyarelalLoss (Pyarelal_loss : ℝ) : Prop :=
  Pyarelal_loss = 1800

-- Question: What was the total loss in the business?
def totalLoss (Ashok_loss Pyarelal_loss : ℝ) : ℝ :=
  Ashok_loss + Pyarelal_loss

-- The mathematically equivalent proof problem statement
theorem total_loss_is_correct (P A : ℝ) (Ashok_loss Pyarelal_loss : ℝ)
  (h1 : ashokCapital A P)
  (h2 : pyarelalLoss Pyarelal_loss)
  (h3 : Ashok_loss = (1 / 9) * Pyarelal_loss) :
  totalLoss Ashok_loss Pyarelal_loss = 2000 := by
  sorry

end total_loss_is_correct_l426_426117


namespace cookies_taken_in_four_days_l426_426403

def initial_cookies : ℕ := 70
def cookies_left : ℕ := 28
def days_in_week : ℕ := 7
def days_taken : ℕ := 4
def daily_cookies_taken (total_cookies_taken : ℕ) : ℕ := total_cookies_taken / days_in_week
def total_cookies_taken : ℕ := initial_cookies - cookies_left

theorem cookies_taken_in_four_days :
  daily_cookies_taken total_cookies_taken * days_taken = 24 := by
  sorry

end cookies_taken_in_four_days_l426_426403


namespace area_of_triangle_formed_by_intercepts_l426_426884

theorem area_of_triangle_formed_by_intercepts :
  let f : ℝ → ℝ := λ x, (x - 5)^2 * (x + 4)
  let x_intercepts := [5, -4]
  let y_intercept := f 0
  let base := (x_intercepts[0] - x_intercepts[1]).abs
  let height := y_intercept
  let area := 0.5 * base * height
  area = 450 :=
by
  sorry

end area_of_triangle_formed_by_intercepts_l426_426884


namespace sin_alpha_plus_theta_eq_neg_cos_alpha_l426_426971

variables {α θ x y : ℝ}

-- Given Conditions
def condition1 := sin α = y
def condition2 := cos α = x
def condition3 := sin (α + θ) = y
def condition4 := cos (α + θ) = -x

-- Theorem to prove
theorem sin_alpha_plus_theta_eq_neg_cos_alpha 
  (h1 : sin α = y)
  (h2 : cos α = x)
  (h3 : sin (α + θ) = y)
  (h4 : cos (α + θ) = -x) : 
  sin (α + θ) = -cos α :=
sorry

end sin_alpha_plus_theta_eq_neg_cos_alpha_l426_426971


namespace arithmetic_sequence_general_term_sum_first_n_terms_sequence_l426_426195

theorem arithmetic_sequence_general_term (d : ℕ) (dpos : d > 0) (h : (1 + 4 * d) ^ 2 = (1 + d) * (1 + 13 * d)) :
  d = 2 -> (∀ n : ℕ, a n = 2 * n - 1) :=
sorry

theorem sum_first_n_terms_sequence (d : ℕ) (dpos : d > 0) (h : (1 + 4 * d) ^ 2 = (1 + d) * (1 + 13 * d)) :
  d = 2 -> (∀ n : ℕ, 
  S n = (∑ i in range (n + 1), b i) = n / (2 * n + 1) + n ^ 2 ) :=
sorry

where 
  a (n : ℕ) := 2 * n - 1
  b (n : ℕ) := (1 / ((a n) * (a (n + 1)))) + (a n)
  S (n : ℕ) := ∑ i in range (n + 1), b i

end arithmetic_sequence_general_term_sum_first_n_terms_sequence_l426_426195


namespace economical_speed_l426_426371

variable (a k : ℝ)
variable (ha : 0 < a) (hk : 0 < k)

theorem economical_speed (v : ℝ) : 
  v = (a / (2 * k))^(1/3) :=
sorry

end economical_speed_l426_426371


namespace correct_option_l426_426059

theorem correct_option : 
  (∀ x, sqrt x = 2 ↔ x = 4) → 
  (cubeRoot (-27) = -3) → 
  (-2^2 = -4) → 
  (∀ x, sqrt (x^2) = x → -5 = 5) → 
  (∃ C, C = (-2^2 = -4)) :=
by
  intros h1 h2 h3 h4
  exact ⟨h3, h3⟩
  
-- Assumptions for cubeRoot definition
def cubeRoot (x : ℤ) : ℤ :=
  if x = -27 then -3 else 0 -- Dummy definition for illustration

-- Assumptions for sqrt definition
def sqrt (x : ℤ) : ℤ :=
  if x = 4 then 2
  else if x = 25 then 5
  else 0 -- Dummy definition for illustration

#check correct_option

end correct_option_l426_426059


namespace evaluate_expression_at_4_l426_426056

theorem evaluate_expression_at_4 :
  ∀ x : ℝ, x = 4 → (x^2 - 3 * x - 10) / (x - 5) = 6 :=
by
  intro x
  intro hx
  sorry

end evaluate_expression_at_4_l426_426056


namespace johns_pace_improvement_l426_426300

variable (initial_distance : ℕ) (initial_time : ℕ) (current_distance : ℕ) (current_time : ℕ)

noncomputable def initial_pace := initial_time / initial_distance
noncomputable def current_pace := current_time / current_distance
noncomputable def improvement := initial_pace - current_pace

theorem johns_pace_improvement
  (h₁ : initial_distance = 8)
  (h₂ : initial_time = 96)
  (h₃ : current_distance = 10)
  (h₄ : current_time = 100) :
  improvement initial_distance initial_time current_distance current_time = 2 := 
by
  sorry

end johns_pace_improvement_l426_426300


namespace compute_value_l426_426132

theorem compute_value (a b c : ℕ) (h : a = 262 ∧ b = 258 ∧ c = 150) : 
  (a^2 - b^2) + c = 2230 := 
by
  sorry

end compute_value_l426_426132


namespace roots_of_quadratic_equation_are_real_and_distinct_l426_426773

def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic_equation_are_real_and_distinct :
  quadratic_discriminant 1 (-2) (-6) > 0 :=
by
  norm_num
  sorry

end roots_of_quadratic_equation_are_real_and_distinct_l426_426773


namespace number_of_players_l426_426654

-- Definitions based on conditions in the problem
def cost_of_gloves : ℕ := 6
def cost_of_helmet : ℕ := cost_of_gloves + 7
def cost_of_cap : ℕ := 3
def total_expenditure : ℕ := 2968

-- Total cost for one player
def cost_per_player : ℕ := 2 * (cost_of_gloves + cost_of_helmet) + cost_of_cap

-- Statement to prove: number of players
theorem number_of_players : total_expenditure / cost_per_player = 72 := 
by
  sorry

end number_of_players_l426_426654


namespace speed_of_current_l426_426392

noncomputable def speed_in_still_water : ℝ := 16
noncomputable def distance_in_meters : ℝ := 100
noncomputable def time_in_seconds : ℝ := 17.998560115190784

noncomputable def distance_in_km : ℝ := distance_in_meters / 1000
noncomputable def time_in_hours : ℝ := time_in_seconds / 3600

noncomputable def speed_downstream : ℝ := distance_in_km / time_in_hours

theorem speed_of_current : 
  speed_downstream = 20.0008 → (speed_downstream - speed_in_still_water) = 4.0008 :=
by
  intros h
  have h1 : speed_downstream - speed_in_still_water = 20.0008 - 16 := by rw [h]
  have h2 : 20.0008 - 16 = 4.0008 := by norm_num
  rw [h1, h2]
  sorry

end speed_of_current_l426_426392


namespace students_playing_both_football_and_cricket_l426_426336

theorem students_playing_both_football_and_cricket
  (total_students : ℕ)
  (students_playing_football : ℕ)
  (students_playing_cricket : ℕ)
  (students_neither_football_nor_cricket : ℕ) :
  total_students = 250 →
  students_playing_football = 160 →
  students_playing_cricket = 90 →
  students_neither_football_nor_cricket = 50 →
  (students_playing_football + students_playing_cricket - (total_students - students_neither_football_nor_cricket)) = 50 :=
by
  intros h_total h_football h_cricket h_neither
  sorry

end students_playing_both_football_and_cricket_l426_426336


namespace dice_sum_to_24_prob_l426_426954

theorem dice_sum_to_24_prob : 
  let events := { (x, y, z, w) : ℕ × ℕ × ℕ × ℕ | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} ∧ z ∈ {1, 2, 3, 4, 5, 6} ∧ w ∈ {1, 2, 3, 4, 5, 6} }
  let event_sum_24 := { (x, y, z, w) ∈ events | x + y + z + w = 24 }
  (finset.card event_sum_24.to_finset) / (finset.card events.to_finset) = 1 / 1296 :=
begin
  sorry
end

end dice_sum_to_24_prob_l426_426954


namespace david_and_maria_ages_l426_426436

theorem david_and_maria_ages 
  (D Y M : ℕ)
  (h1 : Y = D + 7)
  (h2 : Y = 2 * D)
  (h3 : M = D + 4)
  (h4 : M = Y / 2)
  : D = 7 ∧ M = 11 := by
  sorry

end david_and_maria_ages_l426_426436


namespace meiotic_chromatid_separation_l426_426148

-- Definitions reflecting the conditions stated in the problem
inductive meiosis_stage : Type
| interphase_of_first_meiotic_division
| prophase_of_first_meiotic_division
| metaphase_of_first_meiotic_division
| anaphase_of_first_meiotic_division
| telophase_of_first_meiotic_division
| prophase_of_second_meiotic_division
| metaphase_of_second_meiotic_division
| anaphase_of_second_meiotic_division
| telophase_of_second_meiotic_division

-- The necessary condition about chromatids separation
def sister_chromatids_separation (stage : meiosis_stage) : Prop :=
match stage with
| meiosis_stage.anaphase_of_second_meiotic_division => true
| _ => false
end

-- The theorem representing the problem to prove
theorem meiotic_chromatid_separation : sister_chromatids_separation meiosis_stage.anaphase_of_second_meiotic_division = true :=
by
  sorry  -- This will be the proof placeholder.

end meiotic_chromatid_separation_l426_426148


namespace sum_floor_log2_l426_426536

theorem sum_floor_log2 : (∑ N in Finset.range (2048 + 1), nat.floor (Real.log (N : ℝ) / Real.log 2)) = 6157 := sorry

end sum_floor_log2_l426_426536


namespace train_speed_l426_426861

noncomputable def speed_of_train (distance time : ℕ) : ℝ :=
  (distance : ℝ) / (time : ℝ)

theorem train_speed (approx_len : ℝ) (cross_time : ℕ) (speed_approx : ℝ) : 
  approx_len = 500 → cross_time = 30 → speed_approx = 60 →
  speed_of_train 500 cross_time * 3.6 ≈ 60 :=
by
  intros h1 h2 h3
  rw [speed_of_train, h1, h2]
  have : (500 : ℝ) / (30 : ℝ) = 16.67 := by sorry
  rw this
  change 16.67 * 3.6 ≈ 60
  norm_num
  sorry

end train_speed_l426_426861


namespace number_of_permutations_fixed_multiple_of_3_l426_426905

theorem number_of_permutations_fixed_multiple_of_3 :
  ∃ (n : ℕ), n = 9 ∧ 
  ∀ (a : Fin 2016 → Fin 2016), (∀ i : Fin 2016, |a i - i| % 3 = 0) → 
  (a.permSupport = Finset.univ) → 
  n = (Finset.univ.filter (λ a, ∀ i : Fin 2016, | a i - i | % 3 = 0)).card :=
by sorry

end number_of_permutations_fixed_multiple_of_3_l426_426905


namespace number_of_correct_statements_l426_426268

-- Define the quadratic function f(x) = ax^2 + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g(x) = ax^2 - bx + c
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x + c

-- Define the condition that f(x) does not intersect y = x
def no_intersection_with_y_eq_x (a b c : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + b * x + c ≠ x

-- Define the statements to evaluate
def statement_1 (a b c : ℝ) : Prop :=
  ∀ x : ℝ, f a b c (f a b c x) ≠ x

def statement_2 (a b c : ℝ) : Prop :=
  a < 0 → ∃ x : ℝ, f a b c (f a b c x) > x

def statement_3 (a b c : ℝ) : Prop :=
  a + b + c = 0 → ∀ x : ℝ, f a b c (f a b c x) < x

def statement_4 (a b c : ℝ) : Prop :=
  ∀ x : ℝ, g a b c x ≠ -x

-- Prove that exactly 3 of the statements 1, 2, 3, and 4 are correct
theorem number_of_correct_statements (a b c : ℝ)
  (h₁ : no_intersection_with_y_eq_x a b c) :
  ({statement_1 a b c, statement_2 a b c,
    statement_3 a b c, statement_4 a b c}.to_finset.filter id).card = 3 := by
sorry

end number_of_correct_statements_l426_426268


namespace number_of_combinations_l426_426471

open Finset

theorem number_of_combinations : 
  ∃ (n m : ℕ), n = 4 ∧ m = 6 ∧ 
  (choose m n) * (choose m n) * nat.factorial n = 5400 := by
  sorry

end number_of_combinations_l426_426471


namespace sum_of_possible_ks_l426_426862

theorem sum_of_possible_ks : 
  let k := λ k : ℝ, 8 * k * k + 2 * k - 1 = 0 in
  (∀ k1 k2, k k1 ∧ k k2 → k1 + k2) = -1 / 4 :=
by
  sorry

end sum_of_possible_ks_l426_426862


namespace jerry_trips_l426_426072

theorem jerry_trips (carries : ℕ) (total : ℕ) (h1 : carries = 8) (h2 : total = 16) : (total + carries - 1) / carries = 2 :=
by
  rw [h1, h2]
  simp
  sorry

end jerry_trips_l426_426072


namespace problem_f_f10_l426_426601

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else log x / log 10

theorem problem_f_f10 : f(f(10)) = 2 :=
by
  have h1 : f 10 = 1 := by
    simp [f]
    exact log_self (by norm_num : 10 > 0)
  have h2 : f 1 = 2 := by
    simp [f]
    norm_num
  calc
    f(f(10)) = f 1 : by rw [h1]
    ... = 2 : by rw [h2]

end problem_f_f10_l426_426601


namespace cube_surface_area_l426_426844

-- Define length, width, and height of the cuboid
variable {l w h : ℝ}

-- The given conditions
def cuboid_conditions : Prop :=
  (l = w) ∧
  (l * w * h - l * w * (h - 2) = 50) ∧
  (l = h - 2)

-- The resulting shape is a cube
def is_cube (a : ℝ) : Prop :=
  a = l ∧ a = w ∧ a = h - 2

-- Surface area calculation for the cube
def surface_area_of_cube (a : ℝ) : ℝ :=
  6 * a^2

theorem cube_surface_area (a : ℝ) (h : ℝ) :
  cuboid_conditions → is_cube a → surface_area_of_cube a = 150 :=
by
  intros
  sorry

end cube_surface_area_l426_426844


namespace rotation_problem_l426_426793

-- Define the coordinates of the points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the triangles with given vertices
def P : Point := {x := 0, y := 0}
def Q : Point := {x := 0, y := 13}
def R : Point := {x := 17, y := 0}

def P' : Point := {x := 34, y := 26}
def Q' : Point := {x := 46, y := 26}
def R' : Point := {x := 34, y := 0}

-- Rotation parameters
variables (n : ℝ) (x y : ℝ) (h₀ : 0 < n) (h₁ : n < 180)

-- The mathematical proof problem
theorem rotation_problem :
  n + x + y = 180 := by
  sorry

end rotation_problem_l426_426793


namespace diameter_of_hemispherical_holes_l426_426736

theorem diameter_of_hemispherical_holes (s : ℝ) (r : ℝ) (d : ℝ) :
  s = 2 ∧ ∀ (a b : ℝ), a = r ∧ b = r → d = 2 * r → 
  let P := (1 : ℝ),
      Q := (1 : ℝ) in
      (P^2 + Q^2 = (2 * r)^2) ∧ d = sqrt 2 := 
by
  sorry

end diameter_of_hemispherical_holes_l426_426736


namespace sheep_count_l426_426030

theorem sheep_count {c s : ℕ} 
  (h1 : c + s = 20)
  (h2 : 2 * c + 4 * s = 60) : s = 10 :=
sorry

end sheep_count_l426_426030


namespace function_at_neg_one_zero_l426_426591

-- Define the function f with the given conditions
variable {f : ℝ → ℝ}

-- Declare the conditions as hypotheses
def domain_condition : ∀ x : ℝ, true := by sorry
def non_zero_condition : ∃ x : ℝ, f x ≠ 0 := by sorry
def even_function_condition : ∀ x : ℝ, f (x + 2) = f (2 - x) := by sorry
def odd_function_condition : ∀ x : ℝ, f (1 - 2 * x) = -f (2 * x + 1) := by sorry

-- The main theorem to be proved
theorem function_at_neg_one_zero :
  f (-1) = 0 :=
by
  -- Use the conditions to derive the result
  sorry

end function_at_neg_one_zero_l426_426591


namespace convexPolygonHasTwoEqualSides_l426_426083

-- Defining a convex polygon as a list of points
def convexPolygon (points : List (ℝ × ℝ)) : Prop :=
  -- Omitted definition of convexity for brevity; assume it exists
  sorry

-- Defining isosceles triangles formed by the given polygon points
def isoscelesTriangle (a b c : (ℝ × ℝ)) : Prop :=
  dist a b = dist a c ∨ dist a b = dist b c ∨ dist a c = dist b c

-- Ensuring non-intersecting diagonals
def nonIntersectingDiagonals (polygon : List (ℝ × ℝ)) (diagonal1 diagonal2 : ℝ × ℝ) : Prop :=
  -- Omitted definition assuming we have a function to check non-intersecting diagonals
  sorry

-- Main theorem statement
theorem convexPolygonHasTwoEqualSides (points : List (ℝ × ℝ)) 
    (h_convex : convexPolygon points)
    (h_iso_triangles : ∀ triangle ∈ (some_triangle_partition_function points), isoscelesTriangle triangle.1 triangle.2 triangle.3)
    (h_non_intersecting : ∀ diag1 diag2, nonIntersectingDiagonals points diag1 diag2) :
    ∃ p₁ p₂ ∈ points, p₁ ≠ p₂ ∧ dist p₁ p₂ = dist p₁ another_point :=
sorry

end convexPolygonHasTwoEqualSides_l426_426083


namespace solution_set_of_inequality_cauchy_schwarz_application_l426_426827

theorem solution_set_of_inequality (c : ℝ) (h1 : c > 0) (h2 : ∀ x : ℝ, x + |x - 2 * c| ≥ 2) : 
  c ≥ 1 :=
by
  sorry

theorem cauchy_schwarz_application (m p q r : ℝ) (h1 : m ≥ 1) (h2 : 0 < p ∧ 0 < q ∧ 0 < r) (h3 : p + q + r = 3 * m) : 
  p^2 + q^2 + r^2 ≥ 3 :=
by
  sorry

end solution_set_of_inequality_cauchy_schwarz_application_l426_426827


namespace factor_x4_minus_81_l426_426927

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
sorry

end factor_x4_minus_81_l426_426927


namespace max_intersections_of_two_quadrilaterals_l426_426049

-- Define quadrilaterals Q1 and Q2, each having 4 sides
def quadrilateral (Q : Type) := { sides : ℕ // sides = 4 }

-- Define the maximum number of intersection points function
def max_intersection_points (Q1 Q2 : Type) [quadrilateral Q1] [quadrilateral Q2] : ℕ :=
  4 * 4

-- The theorem to prove
theorem max_intersections_of_two_quadrilaterals (Q1 Q2 : Type) [quadrilateral Q1] [quadrilateral Q2] :
  max_intersection_points Q1 Q2 = 16 :=
by
  sorry

end max_intersections_of_two_quadrilaterals_l426_426049


namespace exists_good_matrix_l426_426858

def binary_matrix (n : ℕ) : Type := Matrix (Fin n) (Fin n) (Fin 2)

def is_good_matrix {n : ℕ} (A : binary_matrix n) : Prop :=
  ∃ t b : (Fin n → Fin n → Prop), 
    (∀ i j : Fin n, i < j → t i j) ∧
    (∀ i j : Fin n, j < i → b i j)

theorem exists_good_matrix (m : ℕ) (hm : m > 0) :
  ∃ M : ℕ, ∀ {n : ℕ} (hn : n > M) (A : binary_matrix n),
    ∃ (i : Finset (Fin n)), 
      i.card = m ∧ 
      is_good_matrix (A.minor (λ x, ⟨i.val x, sorry⟩) (λ x, ⟨i.val x, sorry⟩)) :=
by {
  use RamseyNumber (λ x y => exists_good_matrix m _),
  sorry
}

end exists_good_matrix_l426_426858


namespace number_of_people_third_day_l426_426464

variable (X : ℕ)
variable (total : ℕ := 246)
variable (first_day : ℕ := 79)
variable (second_day_third_day_diff : ℕ := 47)

theorem number_of_people_third_day :
  (first_day + (X + second_day_third_day_diff) + X = total) → 
  X = 60 := by
  sorry

end number_of_people_third_day_l426_426464


namespace find_acute_angle_of_geometric_seq_l426_426200

noncomputable def geometric_seq_val_acute_angle (a_n : ℕ → ℝ) (α : ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a_n n = a_n 0 * r^n) ∧
  (∃ a1 a8, a1 = a_n 1 ∧ a8 = a_n 8 ∧
    (a1 + a8 = 2 * real.sin α ∧ a1 * a8 = (a_n 3) * (a_n 6)  ∧
    (a1 + a8)^2 = 2*a_n 3 * a_n 6 + 6)) 

theorem find_acute_angle_of_geometric_seq 
  {a_n : ℕ → ℝ} {α : ℝ} 
  (h : geometric_seq_val_acute_angle a_n α):
  α = π / 3 :=
sorry

end find_acute_angle_of_geometric_seq_l426_426200


namespace exists_2018_integers_l426_426142

theorem exists_2018_integers (c : ℕ) (c_eq : c = 2018) :
  ∃ (xs : Fin c → ℕ), (∃ n, (∑ k in Finset.range c, xs k ^ 2) = n^3) ∧ (∃ m, (∑ k in Finset.range c, xs k ^ 3) = m^2) :=
by {
  have c_eq_2018 := sorry,
  use λ i, sorry, -- Here we should specify the sequence, but it is skipped for brevity.
  split,
  { use sorry, -- Here we should show that the sum of squares is a perfect cube
    sorry },
  { use sorry, -- Here we should show that the sum of cubes is a perfect square
    sorry },
}

end exists_2018_integers_l426_426142


namespace no_infinite_family_of_lines_l426_426282

theorem no_infinite_family_of_lines (l : ℕ → Line) (k : ℕ → ℝ)
  (h1 : ∀ n, (1, 1) ∈ l n)
  (h2 : ∀ n, let a := x_intercept (l n), let b := y_intercept (l n) in k (n+1) = a - b)
  (h3 : ∀ n, k n * k (n+1) ≥ 0) : 
  ¬ ∀ n, ∃ k₀ : ℝ, k₀ = k n :=
sorry

end no_infinite_family_of_lines_l426_426282


namespace max_distance_complex_l426_426321

open Complex

theorem max_distance_complex (z : ℂ) (hz : ‖z‖ = 3) :
  ‖(2 + 3 * I) * z^4 - z^6‖ ≤ 81 * sqrt 34 := sorry

end max_distance_complex_l426_426321


namespace factor_x4_minus_81_l426_426917

theorem factor_x4_minus_81 : ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intro x
  sorry

end factor_x4_minus_81_l426_426917


namespace regular_octagon_angle_l426_426423

theorem regular_octagon_angle (n : ℕ) (h₁ : n = 8) :
  let interior_angle := 135
  let exterior_angle := 45
  interior_angle = 135 ∧ exterior_angle = 45 :=
by
  let interior_sum := 180 * (n - 2)
  have h₂ : interior_sum = 1080 := by
    rw [h₁, Nat.sub_self, Nat.mul_one]
  let int_angle := interior_sum / n
  have h₃ : int_angle = 135 := by
    rw [h₂, h₁]
    norm_num
  let ext_angle := 180 - int_angle
  have h₄ : ext_angle = 45 := by
    rw [h₃]
    norm_num
  split
  · exact h₃
  · exact h₄
  sorry -- Finalizing the proof.

end regular_octagon_angle_l426_426423


namespace polynomial_non_real_root_product_sum_l426_426554

theorem polynomial_non_real_root_product_sum
  (a b c d : ℝ)
  (h_poly_roots : ∃ (z w : ℂ), (z * w = -7 + 4 * complex.I) ∧ (conj z + conj w = -2 + 3 * complex.I))
  (h_poly : polynomial (ℝ) := polynomial.C a + polynomial.C b * polynomial.X + polynomial.C c * polynomial.X^2 + polynomial.C d * polynomial.X^3 + polynomial.C 1 * polynomial.X^4) :
  b = -1 :=
begin
  sorry
end

end polynomial_non_real_root_product_sum_l426_426554


namespace period_of_f_max_value_f_l426_426603

noncomputable def f (x : ℝ) : ℝ := 4 * sin x * cos (x - π / 6) + 1

-- Part I: Prove that the period of f(x) is π
theorem period_of_f : ∀ x, f (x + π) = f x := sorry

-- Part II: Prove that the maximum value of f(x) in the interval [-π/6, π/4] is √3 + 2
theorem max_value_f : ∀ x ∈ Icc (-π / 6) (π / 4), f x ≤ √3 + 2 :=
sorry

end period_of_f_max_value_f_l426_426603


namespace number_15_unlocks_30_45_door_10_remains_unlocked_and_door_9_remains_locked_unlocked_doors_after_50_l426_426071

-- Define the condition functions
def door_toggles (n : Nat) : Nat → Bool
| 1 => if n % 1 = 0 then tt else ff
| 2 => if n % 2 = 0 then tt else ff
| 3 => if n % 3 = 0 then tt else ff
| 4 => if n % 4 = 0 then tt else ff
| 5 => if n % 5 = 0 then tt else ff
| 6 => if n % 6 = 0 then tt else ff
| 7 => if n % 7 = 0 then tt else ff
| 8 => if n % 8 = 0 then tt else ff
| 9 => if n % 9 = 0 then tt else ff
| 10 => if n % 10 = 0 then tt else ff
-- Repeat this pattern up to door 50 for thoroughness.

-- Main Lean statements for the proof
theorem number_15_unlocks_30_45 : ∀ (n : Nat), n = 15 → (door_toggles n 30 = tt) ∧ (door_toggles n 45 = tt) := by
  intros n h
  sorry

theorem door_10_remains_unlocked_and_door_9_remains_locked : 
  ∀ (n : Nat), n <= 50 → ((n = 10 → door_toggles n 50 = tt) ∧ (n = 9 → door_toggles n 50 = ff)) := by
  intros n hn
  sorry

theorem unlocked_doors_after_50 : ¬ ∀ (n : Nat), n <= 50 → ¬ (n = 1 ∨ n = 4 ∨ n = 9 ∨ n = 16 ∨ n = 25 ∨ n = 36 ∨ n = 49) := by
  intros n hn
  sorry

end number_15_unlocks_30_45_door_10_remains_unlocked_and_door_9_remains_locked_unlocked_doors_after_50_l426_426071


namespace part1_part2_l426_426871

-- Definitions of points and circles
variable (ω₁ ω₂ : Circle)
variable (A B K M Q R P: Point)
variable (hA : A ∈ ω₁ ∧ A ∈ ω₂)
variable (hB : B ∈ ω₁ ∧ B ∈ ω₂)
variable (hK : K ∈ ω₁ ∧ K ∈ LineBP B K ∧ K ≠ B)
variable (hM : M ∈ ω₂ ∧ M ∈ LineBP B M ∧ M ≠ B)
variable (hPQ_tangent_ω₁ : TangentLine PointPQ LineP Q ω₁ Q PQ)
variable (hPQ_parallel_AM : Parallel LinePQ LineAM)
variable (hPR_tangent_ω₂ : TangentLine PointPR LineP R ω₂ R PR)
variable (hPR_parallel_AK : Parallel LinePR LineAK)
variable (hQR_opposite_sides_KM : OppositeSides LineQR K M Q  R)

-- Proving statements

theorem part1 : A ∈ LineQR := by
  sorry

theorem part2 : P ∈ LineKM := by
  sorry

end part1_part2_l426_426871


namespace felicity_gas_usage_l426_426530

variable (A F : ℕ)

theorem felicity_gas_usage
  (h1 : F = 4 * A - 5)
  (h2 : A + F = 30) :
  F = 23 := by
  sorry

end felicity_gas_usage_l426_426530


namespace partI_solution_set_l426_426221

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) - abs (x - a^2 - a)

theorem partI_solution_set (x : ℝ) : 
  (f x 1 ≤ 1) ↔ (x ≤ -1) :=
sorry

end partI_solution_set_l426_426221


namespace max_elements_in_S_l426_426517

-- Definitions to establish the problem domain
def valid_element (a : ℕ) : Prop := a > 0 ∧ a ≤ 100

def condition_two (S : Finset ℕ) : Prop :=
  ∀ a b ∈ S, a ≠ b → ∃ c ∈ S, Nat.gcd a c = 1 ∧ Nat.gcd b c = 1

def condition_three (S : Finset ℕ) : Prop :=
  ∀ a b ∈ S, a ≠ b → ∃ d ∈ S, d ≠ a ∧ d ≠ b ∧ Nat.gcd a d > 1 ∧ Nat.gcd b d > 1

-- Statement of the theorem
theorem max_elements_in_S (S : Finset ℕ) : 
  (∀ s ∈ S, valid_element s) ∧ 
  condition_two S ∧ 
  condition_three S →
  Finset.card S ≤ 72 := 
sorry

end max_elements_in_S_l426_426517


namespace lines_intersection_l426_426796

/-- Two lines are defined by the equations y = 2x + c and y = 4x + d.
These lines intersect at the point (8, 12).
Prove that c + d = -24. -/
theorem lines_intersection (c d : ℝ) (h1 : 12 = 2 * 8 + c) (h2 : 12 = 4 * 8 + d) :
    c + d = -24 :=
by
  sorry

end lines_intersection_l426_426796


namespace sum_of_interior_angles_heptagon_l426_426115

theorem sum_of_interior_angles_heptagon :
  ∀ (n : ℕ), (n - 2) * 180 = 900 → n = 7 :=
begin
  intros n hn,
  -- proof goes here
  sorry
end

end sum_of_interior_angles_heptagon_l426_426115


namespace max_value_of_abc_sum_l426_426584

theorem max_value_of_abc_sum (a b c : ℕ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a < b ∧ b < c) (h3 : b^2 = a * c) (h4 : log 2016 (a * b * c) = 3) :
  a + b + c ≤ 4066273 :=
sorry

end max_value_of_abc_sum_l426_426584


namespace total_cages_required_l426_426852

theorem total_cages_required :
  ∀ (puppies_init adult_dogs_init kittens_init rabbits_init hamsters_init: ℕ)
    (puppies_sold_percent adult_dogs_sold_percent kittens_sold_percent rabbits_sold_percent hamsters_sold_percent: ℝ)
    (puppies_per_cage adult_dogs_per_cage kittens_per_cage rabbits_per_cage hamsters_per_cage: ℕ),
    puppies_init = 45 ∧ adult_dogs_init = 30 ∧ kittens_init = 25 ∧ rabbits_init = 15 ∧ hamsters_init = 10 ∧
    puppies_sold_percent = 0.75 ∧ adult_dogs_sold_percent = 0.5 ∧ kittens_sold_percent = 0.4 ∧ rabbits_sold_percent = 0.6 ∧ hamsters_sold_percent = 0.3 ∧
    puppies_per_cage = 3 ∧ adult_dogs_per_cage = 2 ∧ kittens_per_cage = 2 ∧ rabbits_per_cage = 4 ∧ hamsters_per_cage = 5 →
      let puppies_sold := (puppies_sold_percent * puppies_init).toNat,
          adult_dogs_sold := (adult_dogs_sold_percent * adult_dogs_init).toNat,
          kittens_sold := (kittens_sold_percent * kittens_init).toNat,
          rabbits_sold := (rabbits_sold_percent * rabbits_init).toNat,
          hamsters_sold := (hamsters_sold_percent * hamsters_init).toNat,
          puppies_left := puppies_init - puppies_sold,
          adult_dogs_left := adult_dogs_init - adult_dogs_sold,
          kittens_left := kittens_init - kittens_sold,
          rabbits_left := rabbits_init - rabbits_sold,
          hamsters_left := hamsters_init - hamsters_sold,
          puppy_cages := (puppies_left + puppies_per_cage - 1) / puppies_per_cage,
          adult_dog_cages := (adult_dogs_left + adult_dogs_per_cage - 1) / adult_dogs_per_cage,
          kitten_cages := (kittens_left + kittens_per_cage - 1) / kittens_per_cage,
          rabbit_cages := (rabbits_left + rabbits_per_cage - 1) / rabbits_per_cage,
          hamster_cages := (hamsters_left + hamsters_per_cage - 1) / hamsters_per_cage in
      puppy_cages + adult_dog_cages + kitten_cages + rabbit_cages + hamster_cages = 24 := by
  sorry

end total_cages_required_l426_426852


namespace negative_number_unique_l426_426111

theorem negative_number_unique (a b c d : ℚ) (h₁ : a = 1) (h₂ : b = 0) (h₃ : c = 1/2) (h₄ : d = -2) :
  ∃! x : ℚ, x < 0 ∧ (x = a ∨ x = b ∨ x = c ∨ x = d) :=
by 
  sorry

end negative_number_unique_l426_426111


namespace circle_with_diameter_eq_l426_426018

noncomputable def parabola_circle (a b c : ℝ) (ha : a ≠ 0) : Prop :=
  ∃ (A B : ℝ), (a * A^2 + b * A + c = 0) ∧ (a * B^2 + b * B + c = 0) ∧
  (circle_eq : ∀ (x y : ℝ), ((x + b / (2 * a))^2 + y^2 = (b^2 - 4 * a * c) / (4 * a^2)) ↔ 
    a * x^2 + b * x + c + a * y^2 = 0)

-- Declare the theorem to prove
theorem circle_with_diameter_eq (a b c : ℝ) (ha : a ≠ 0) : parabola_circle a b c ha :=
  sorry

end circle_with_diameter_eq_l426_426018


namespace polar_to_rectangular_l426_426899

theorem polar_to_rectangular (r θ : ℝ) (h1 : r = 6) (h2 : θ = Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (3, 3 * Real.sqrt 3) :=
by
  rw [h1, h2]
  have h3 : Real.cos (Real.pi / 3) = 1 / 2 := Real.cos_pi_div_three
  have h4 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := Real.sin_pi_div_three
  simp [h3, h4]
  norm_num
  sorry

end polar_to_rectangular_l426_426899


namespace factorize_x4_minus_81_l426_426932

theorem factorize_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factorize_x4_minus_81_l426_426932


namespace union_of_sets_l426_426575

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Prove that A ∪ B = {x | -1 < x ∧ x ≤ 2}
theorem union_of_sets (x : ℝ) : x ∈ (A ∪ B) ↔ x ∈ {x | -1 < x ∧ x ≤ 2} :=
by
  sorry

end union_of_sets_l426_426575


namespace octahedral_dice_sum_16_probability_l426_426377

theorem octahedral_dice_sum_16_probability : 
  let outcomes := (1:8) × (1:8)
  let successful_outcome := (8, 8) in
  (∑ outcome in outcomes, if outcome.1 + outcome.2 = 16 then 1 else 0) / (8 * 8) = 1 / 64 :=
sorry

end octahedral_dice_sum_16_probability_l426_426377


namespace fraction_problem_l426_426245

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end fraction_problem_l426_426245


namespace cone_volume_surface_area_sector_l426_426838

theorem cone_volume_surface_area_sector (V : ℝ):
  (∃ (r l h : ℝ), (π * r * (r + l) = 15 * π) ∧ (l = 6 * r) ∧ (h = Real.sqrt (l^2 - r^2)) ∧ (V = (1/3) * π * r^2 * h)) →
  V = (25 * Real.sqrt 3 / 7) * π :=
by 
  sorry

end cone_volume_surface_area_sector_l426_426838


namespace polynomial_square_b_value_l426_426755

theorem polynomial_square_b_value (a b p q : ℝ) (h : (∀ x : ℝ, x^4 + x^3 - x^2 + a * x + b = (x^2 + p * x + q)^2)) : b = 25/64 := by
  sorry

end polynomial_square_b_value_l426_426755


namespace find_x_l426_426260

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + 2y = 14) : x = 10 := by
  sorry

end find_x_l426_426260


namespace minimum_triangles_partition_l426_426841

theorem minimum_triangles_partition (A B C D E F : Type) (area_of_hexagon : ℕ) : 
  (area_of_hexagon = 63) → (∀ (triangle : Type), ¬(triangle has equal_area_with other_triangles) → triangle_minimally_partitioned) :=
begin
  sorry
end

end minimum_triangles_partition_l426_426841


namespace similar_triangles_and_area_ratio_l426_426508

open EuclideanGeometry

-- Given: A triangle ABC where BC = a, CA = b, and AB = c.
noncomputable def triangle (ABC : Triangle) (a b c : ℝ) (BC CA AB : LineSegment) := 
  BC.length = a ∧ CA.length = b ∧ AB.length = c

-- D is the midpoint of BC.
def midpoint (BC D : Point) (a : ℝ) := 
  2 * D.distance BC.A = a ∧ 2 * D.distance BC.B = a

-- E is the intersection of the angle bisector of ∠BAC with BC.
def angle_bisector (A B C E : Point) := 
  is_angle_bisector A E B C

-- The circle through A, D, E intersects AC and AB again at F and G respectively.
def intersect_circle (A D E F G : Point) (circle : Circle) (AC AB : Line) :=
  on_circle circle A ∧ on_circle circle D ∧ on_circle circle E ∧
  circle.intersects AC = F ∧ circle.intersects AB = G

-- H is a point on AB with BG = GH.
def points_on_AB (AB H G : Point) := 
  H ≠ B ∧ H ∈ AB ∧ G ∈ AB ∧ G.distance H = H.distance B

-- The similarity and the ratio of areas.
theorem similar_triangles_and_area_ratio
  (a b c : ℝ)
  (A B C D E F G H : Point)
  (BC : LineSegment)
  (tons_of_conditions : triangle (A, B, C) a b c BC ∧ midpoint BC D a ∧
    angle_bisector A B C E ∧ intersect_circle A D E F G (Circle.mk A D E) 
    (Line.mk A C) (Line.mk A B) ∧ points_on_AB (Line.mk A B) H G) : 
  similar (Triangle.mk E B H) (Triangle.mk A B C) ∧
  area_ratio (Triangle.mk E B H) (Triangle.mk A B C) = (a / (b + c))^2 :=
sorry

end similar_triangles_and_area_ratio_l426_426508


namespace students_in_class_l426_426752

theorem students_in_class (n m f r u : ℕ) (cond1 : 20 < n ∧ n < 30)
  (cond2 : f = 2 * m) (cond3 : n = m + f)
  (cond4 : r = 3 * u - 1) (cond5 : r + u = n) :
  n = 27 :=
sorry

end students_in_class_l426_426752


namespace find_N_l426_426006

-- Definition of the conditions
def is_largest_divisor_smaller_than (m N : ℕ) : Prop := m < N ∧ Nat.gcd m N = m

def produces_power_of_ten (N m : ℕ) : Prop := ∃ k : ℕ, k > 0 ∧ N + m = 10^k

-- Final statement to prove
theorem find_N (N : ℕ) : (∃ m : ℕ, is_largest_divisor_smaller_than m N ∧ produces_power_of_ten N m) → N = 75 :=
by
  sorry

end find_N_l426_426006


namespace problem_statement_l426_426313

variable {a b : ℝ}

theorem problem_statement (h₁ : a > 0) (h₂ : b ∈ ℝ) : a + 2b ≥ 2b - a := 
by
  sorry

end problem_statement_l426_426313


namespace cost_of_2500_pencils_l426_426080

theorem cost_of_2500_pencils (cost_per_100 : ℕ) (cost_100_pencils : cost_per_100 = 30) (num_pencils_to_buy : ℕ) (pencils : num_pencils_to_buy = 2500) : 
  ∃ total_cost : ℕ, total_cost = 750 :=
by 
  let cost_per_pencil := 30 / 100
  have total_cost := cost_per_pencil * 2500
  have h : total_cost = 750 := by sorry
  use total_cost
  exact h

end cost_of_2500_pencils_l426_426080


namespace sum_of_interior_angles_quadrilateral_sum_of_interior_angles_convex_pentagon_sum_of_interior_angles_convex_n_gon_l426_426549

-- Problem (1): Sum of interior angles of a quadrilateral
theorem sum_of_interior_angles_quadrilateral : 
  (sum_of_interior_angles (4 : ℕ) = 360) := 
by
  sorry

-- Problem (2): Sum of interior angles of a convex pentagon
theorem sum_of_interior_angles_convex_pentagon : 
  (sum_of_interior_angles (5 : ℕ) = 540) := 
by
  sorry

-- Problem (3): Sum of interior angles of a convex n-gon
theorem sum_of_interior_angles_convex_n_gon (n : ℕ) (h : n ≥ 3) : 
  (sum_of_interior_angles n = 180 * (n - 2)) := 
by
  sorry

end sum_of_interior_angles_quadrilateral_sum_of_interior_angles_convex_pentagon_sum_of_interior_angles_convex_n_gon_l426_426549


namespace simplify_expression_find_value_a_m_2n_l426_426499

-- Proof Problem 1
theorem simplify_expression : ( (-2 : ℤ) * x )^3 * x^2 + ( (3 : ℤ) * x^4 )^2 / x^3 = x^5 := by
  sorry

-- Proof Problem 2
theorem find_value_a_m_2n (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2*n) = 18 := by
  sorry

end simplify_expression_find_value_a_m_2n_l426_426499


namespace circle_equation_l426_426393

theorem circle_equation (h k r : ℝ) (hc : h = 1) (kc : k = -2) (rc : r = 3) :
  ∀ x y, (x - h)^2 + (y - k)^2 = r^2 ↔ (x - 1)^2 + (y + 2)^2 = 9 :=
by intros; simp [hc, kc, rc]; exact eq_true_intro rfl

end circle_equation_l426_426393


namespace coordinates_of_D_l426_426350

-- Definitions of the points and translation conditions
def A : (ℝ × ℝ) := (-1, 4)
def B : (ℝ × ℝ) := (-4, -1)
def C : (ℝ × ℝ) := (4, 7)

theorem coordinates_of_D :
  ∃ (D : ℝ × ℝ), D = (1, 2) ∧
  ∀ (translate : ℝ × ℝ), translate = (C.1 - A.1, C.2 - A.2) → 
  D = (B.1 + translate.1, B.2 + translate.2) :=
by
  sorry

end coordinates_of_D_l426_426350


namespace find_a_l426_426600

noncomputable def f (a : ℝ) (x : ℝ) :=
  if x >= 0 then real.sqrt (a * x - 1)
  else -x^2 - 4 * x

theorem find_a (a : ℝ) (x : ℝ) (hx : x = -2) (hf : f a (f a x) = 3) :
  a = 5 / 2 :=
by 
  have h1 : f a x = -(-2)^2 - 4*(-2) := by rw [hx]
  have h2 : f a (-(-2)^2 - 4*(-2)) = real.sqrt(4*a - 1) := by rw [h1]
  have h3 : real.sqrt(4*a - 1) = 3 := by rw [hf, h2]
  have h4 : (4*a - 1) = 9 := by rw [real.sqrt_squared 4*a - 1]
  have ha : a = 5/2 := by rw [h4]; sorry

end find_a_l426_426600


namespace length_of_median_l426_426662

theorem length_of_median (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (CB AC AB : ℕ)
  (hCB : CB = 7)
  (hAC : AC = 8)
  (hAB : AB = 9) :
  let D := midpoint AC
  in metric.dist B D = 7 := sorry

end length_of_median_l426_426662


namespace rhombus_angles_l426_426972

-- Define the rhombus ABCD with side lengths
def is_rhombus (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] (side: ℝ) : Prop :=
  dist A B = side ∧ dist B C = side ∧ dist C D = side ∧ dist D A = side 

-- Define the points M, N on sides BC and CD with a specific condition
def points_on_sides (B C D M N: Type) [metric_space B] [metric_space C] [metric_space D] [metric_space M] [metric_space N] : Prop :=
  dist M C + dist C N + dist M N = 2

-- Define the angle condition
def angle_condition (A M N B D: Type) [metric_space A] [metric_space M] [metric_space N] [metric_space B] [metric_space D] : Prop :=
  ∃ (α: ℝ), 2 * angle A M N = α ∧ α = angle B A D

-- Main theorem to prove the angles of the rhombus
theorem rhombus_angles (A B C D M N : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M] [metric_space N]:
  is_rhombus A B C D 1 →
  points_on_sides B C D M N →
  angle_condition A M N B D →
  ∃ (α: ℝ), (∀ (θ: ℝ), θ = α ∨ θ = 180 - α) :=
  by
  intros h1 h2 h3
  sorry

end rhombus_angles_l426_426972


namespace train_crossing_time_is_30_seconds_l426_426105

-- Define the parameters given in conditions
def train_length : ℝ := 170
def train_speed_kmh : ℝ := 45
def bridge_length : ℝ := 205
def speed_conversion_factor : ℝ := 1000 / 3600 -- km/hr to m/s

-- Define converted speed in meters per second
noncomputable def train_speed_ms : ℝ := train_speed_kmh * speed_conversion_factor

-- Define total distance to cover
noncomputable def total_distance : ℝ := train_length + bridge_length

-- Define the correct answer, the time it takes to cross the bridge
noncomputable def crossing_time : ℝ := total_distance / train_speed_ms

theorem train_crossing_time_is_30_seconds : crossing_time = 30 := by
  -- Time to cross the bridge should equal 30 seconds.
  sorry

end train_crossing_time_is_30_seconds_l426_426105


namespace jumps_ratio_l426_426238

noncomputable def ratio_of_jumps 
  (H1 : ℕ := 180)
  (L1 : ℕ := 3 * H1 / 4)
  (H2 : ℕ)
  (L2 : ℕ := H2 + 50)
  (total : ℕ := H1 + L1 + H2 + L2 := 605) 
  : ℚ := 
  sorry

theorem jumps_ratio (H1 : ℕ := 180) (L1 : ℕ := 3 * H1 / 4) (H2 : ℕ) (L2 : ℕ := H2 + 50) (total : ℕ := H1 + L1 + H2 + L2 := 605) : 
  L1 = 135 ∧ H2 = 120 ∧ (H2 : ℚ) / H1 = 2 / 3 :=
by 
  sorry

end jumps_ratio_l426_426238


namespace graph_is_two_lines_l426_426516

theorem graph_is_two_lines (x y : ℝ) : (x^2 - 25 * y^2 - 10 * x + 50 = 0) ↔
  (x = 5 + 5 * y) ∨ (x = 5 - 5 * y) :=
by
  sorry

end graph_is_two_lines_l426_426516


namespace problem_a_problem_b_problem_c_problem_d_l426_426890

-- Proof problem ①: √8 + √18 - 4√2 = √2
theorem problem_a : Real.sqrt 8 + Real.sqrt 18 - 4 * Real.sqrt 2 = Real.sqrt 2 :=
  sorry

-- Proof problem ②: (1-√2)^0 × √((-2)^2) + ∛8 = 4
theorem problem_b : (1 - Real.sqrt 2)^0 * Real.sqrt ((-2)^2) + Math.cbrt 8 = 4 :=
  sorry

-- Proof problem ③: (√18 - √(1/2)) × √8 = 10
theorem problem_c : (Real.sqrt 18 - Real.sqrt (1 / 2)) * Real.sqrt 8 = 10 :=
  sorry

-- Proof problem ④: (√5 + 3)(√5 - 3) - (√3 - 1)^2 = 2√3 - 8
theorem problem_d : (Real.sqrt 5 + 3) * (Real.sqrt 5 - 3) - (Real.sqrt 3 - 1)^2 = 2 * Real.sqrt 3 - 8 :=
  sorry

end problem_a_problem_b_problem_c_problem_d_l426_426890


namespace points_collinear_l426_426693

def setPoints (S : Set (Real × Real)) : Prop :=
  ∀ P ∈ S, ∃ (eq_points : Finset (Real × Real)), eq_points.card ≥ k ∧ ∀ x ∈ eq_points, dist P x = dist P (Finset.min' eq_points sorry)

theorem points_collinear (n k : ℕ) (h_n : n > 0) (h_k : k > 0) (S : Finset (Real × Real)) (h_S_card : S.card = n) 
  (h_noncollinear : ∀ (A B C : (Real × Real)), A ≠ B → B ≠ C → A ≠ C → 
    ∀ (a b c : (Real × Real) → Real), a A = 0 → a B ≠ 0 → 
    a C = b C → b C ≠ 0 → c A = c B → c B ≠ 0 → False) 
  (h_points : setPoints S) :
  k < (1 / 2) + Real.sqrt (2 * n) := 
sorry

end points_collinear_l426_426693


namespace hyperbola_eccentricity_sqrt_5_l426_426208

theorem hyperbola_eccentricity_sqrt_5
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1))
  (h2 : ∀ x : ℝ, (y = x^2 + 1))
  (h3 : ∃ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1) ∧ (y = x^2 + 1) ∧ 
        ∀ x' y' : ℝ, ((x'^2 / a^2) - (y'^2 / b^2) = 1) ∧ (y' = x^2 + 1) → (x, y) = (x', y')) :
  (∃ e : ℝ, e = sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_sqrt_5_l426_426208


namespace prime_square_sum_l426_426319

noncomputable def is_square (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem prime_square_sum (p q r : ℤ) 
  (hp: Nat.Prime p) (hq: Nat.Prime q) (hr: Nat.Prime r)
  (h1: is_square ((p * q) + 1))
  (h2: is_square ((p * r) + 1))
  (h3: is_square ((q * r) - p)) :
  is_square (p + 2 * q * r + 2) :=
by
  sorry

end prime_square_sum_l426_426319


namespace parabola_kite_area_l426_426758

theorem parabola_kite_area (a b : ℝ) :
  (∀ x, (y = a * x ^ 2 - 2) ∧ (y = 4 - b * x ^ 2)) ∧ 
  (kite_area : ℝ = 12) →
  a + b = 1.5 :=
by
  sorry

end parabola_kite_area_l426_426758


namespace factor_x4_minus_81_l426_426929

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
sorry

end factor_x4_minus_81_l426_426929


namespace maximize_probability_l426_426414

def numbers_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def pairs_summing_to_12 (l : List Int) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 + p.2 = 12) (List.product l l)

def distinct_pairs (pairs : List (Int × Int)) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 ≠ p.2) pairs

def valid_pairs (l : List Int) : List (Int × Int) :=
  distinct_pairs (pairs_summing_to_12 l)

def count_valid_pairs (l : List Int) : Nat :=
  List.length (valid_pairs l)

def remove_and_check (x : Int) : List Int :=
  List.erase numbers_list x

theorem maximize_probability :
  ∀ x : Int, count_valid_pairs (remove_and_check 6) ≥ count_valid_pairs (remove_and_check x) :=
sorry

end maximize_probability_l426_426414


namespace distance_between_P1_and_P2_l426_426495

-- Define the two points
def P1 : ℝ × ℝ := (2, 3)
def P2 : ℝ × ℝ := (5, 10)

-- Define the distance function
noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

-- Define the theorem we want to prove
theorem distance_between_P1_and_P2 :
  distance P1 P2 = Real.sqrt 58 :=
by sorry

end distance_between_P1_and_P2_l426_426495


namespace gcd_282_470_l426_426412

theorem gcd_282_470 : Nat.gcd 282 470 = 94 :=
by
  sorry

end gcd_282_470_l426_426412


namespace marble_probability_same_color_l426_426454

theorem marble_probability_same_color :
  ∀ (blues : ℕ) (yellows : ℕ) (blacks : ℕ),
  blues = 4 ∧ yellows = 5 ∧ blacks = 6 →
  let total_marbles := blues + yellows + blacks in
  let prob_two_blue := (blues / total_marbles) * ((blues - 1) / (total_marbles - 1)) in
  let prob_two_yellow := (yellows / total_marbles) * ((yellows - 1) / (total_marbles - 1)) in
  let prob_two_black := (blacks / total_marbles) * ((blacks - 1) / (total_marbles - 1)) in
  let total_probability_same_color := prob_two_blue + prob_two_yellow + prob_two_black in
  total_probability_same_color = 31/105 :=
begin
  sorry
end

end marble_probability_same_color_l426_426454


namespace probability_is_correct_l426_426362

-- Define the values of the coins
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def half_dollar := 50

-- Define the value function
def coin_value (counts : List Bool) : ℕ := 
  (if counts.head! then penny else 0) + 
  (if counts.get! 1 then nickel else 0) + 
  (if counts.get! 2 then dime else 0) + 
  (if counts.get! 3 then quarter else 0) + 
  (if counts.get! 4 then half_dollar else 0)

-- Define possible outcomes
def outcomes := List.replicate 32 (List.replicate 5 bool.default)

-- Probability computation
def probability_at_least_40_cents := outcomes.count (fun outcome => coin_value outcome ≥ 40) / outcomes.length

theorem probability_is_correct :
  probability_at_least_40_cents = 9 / 16 := by
  sorry

end probability_is_correct_l426_426362


namespace prime_div_p_sq_minus_one_l426_426630

theorem prime_div_p_sq_minus_one {p : ℕ} (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 
  (p % 10 = 1 ∨ p % 10 = 9) → 40 ∣ (p^2 - 1) :=
sorry

end prime_div_p_sq_minus_one_l426_426630


namespace E_not_integer_l426_426075

variable (a b c d : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

theorem E_not_integer : 1 < E a b c d ∧ E a b c d < 2 ∧ ¬(E a b c d ∈ ℕ) :=
  by
    -- Definition of E based on the provided formula
    let E := (a / (a + b + d)) + (b / (b + c + a)) + (c / (c + d + b)) + (d / (d + a + c))
    have : 1 < E := sorry
    have : E < 2 := sorry
    have : ¬(E ∈ ℕ) := sorry
    exact ⟨this, this_1, this_2⟩

end E_not_integer_l426_426075


namespace find_a_n_sum_first_n_terms_l426_426570

open Nat

-- Defining the sequence and sum
variable (a S : ℕ → ℕ)

-- Condition definitions
def condition1 : a 1 = 1 :=
  -- a_1 = 1
  sorry

def condition2 (n : ℕ) : 2 * S (n + 1) = (n + 1 + 1) * a (n + 1) :=
  -- 2S_(n+1) = ((n+1)+1)a_(n+1)
  sorry

-- To prove a_n = n
theorem find_a_n (n : ℕ) (h1 : condition1) (h2 : ∀ n, condition2 n) : a n = n := 
  sorry

-- To prove Σ (1 / S_k) = 2n / (n + 1) for k from 1 to n
theorem sum_first_n_terms (n : ℕ) (h1 : condition1) (h2 : ∀ n, condition2 n) :
  (finset.range n).sum (λ k, 1 / S (k + 1)) = 2 * n / (n + 1) :=
  sorry

end find_a_n_sum_first_n_terms_l426_426570


namespace order_of_abc_l426_426180

variable (a b c : ℝ)

-- Definitions
def def_a := (a = Real.sqrt 3)
def def_b := (b = Real.log 2 / Real.log 3)
def def_c := (c = Real.log 3 / Real.log 0.5)

-- Proof of correct order
theorem order_of_abc (ha : def_a) (hb : def_b) (hc : def_c) : c < b ∧ b < a := by
  sorry

end order_of_abc_l426_426180


namespace pump_rates_l426_426407

theorem pump_rates (x y z : ℝ)
(h1 : x + y + z = 14)
(h2 : z = x + 3)
(h3 : y = 11 - 2 * x)
(h4 : 9 / x = (28 - 2 * y) / z)
: x = 3 ∧ y = 5 ∧ z = 6 :=
by
  sorry

end pump_rates_l426_426407


namespace factor_x4_minus_81_l426_426919

theorem factor_x4_minus_81 : ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intro x
  sorry

end factor_x4_minus_81_l426_426919


namespace sum_of_squares_of_medians_l426_426051

/--
Given a triangle with side lengths 9, 12, and 15, the sum of the squares of the lengths of its medians is 443.25.
-/
theorem sum_of_squares_of_medians (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  let m_a := (2 * b^2 + 2 * c^2 - a^2) / 4,
      m_b := (2 * a^2 + 2 * c^2 - b^2) / 4,
      m_c := (2 * a^2 + 2 * b^2 - c^2) / 4 in
  m_a + m_b + m_c = 443.25 :=
by
  sorry

end sum_of_squares_of_medians_l426_426051


namespace coefficient_of_x_squared_expansion_l426_426163

theorem coefficient_of_x_squared_expansion :
  (∀ (x : ℂ), polynomial.coeff (polynomial.C (1 : ℂ) * polynomial.X ^ 7 - 3 * polynomial.X) 2 = 189 ) :=
sorry

end coefficient_of_x_squared_expansion_l426_426163


namespace neither_necessary_nor_sufficient_l426_426201

theorem neither_necessary_nor_sufficient (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  ¬(∀ a b, (a > b → (1 / a < 1 / b)) ∧ ((1 / a < 1 / b) → a > b)) := sorry

end neither_necessary_nor_sufficient_l426_426201


namespace find_a_l426_426566

theorem find_a (x : ℝ) (n : ℕ) (hx : x > 0) (hn : n > 0) :
  x + n^n * (1 / (x^n)) ≥ n + 1 :=
sorry

end find_a_l426_426566


namespace recurring_division_l426_426416

def recurring_to_fraction (recurring: ℝ) (part: ℝ): ℝ :=
  part * recurring

theorem recurring_division (recurring: ℝ) (part1 part2: ℝ):
  recurring_to_fraction recurring part1 = 0.63 →
  recurring_to_fraction recurring part2 = 0.18 →
  recurring ≠ 0 →
  (0.63:ℝ)/0.18 = (7:ℝ)/2 :=
by
  intros h1 h2 h3
  rw [recurring_to_fraction] at h1 h2
  sorry

end recurring_division_l426_426416


namespace ratio_PA_PM_eq_AA_AM_l426_426757

noncomputable def triangle (A B C : Point) : Triangle := sorry

noncomputable def orthocenter (T : Triangle) : Point := sorry

noncomputable def orthicTriangle (T : Triangle) : Triangle := sorry

noncomputable def altitudeIntersection (T : Triangle) (A' : Point) : Line := sorry

noncomputable def intersection (l1 l2 : Line) : Point := sorry

theorem ratio_PA_PM_eq_AA_AM
  (T : Triangle)
  (M : Point)
  (A' B' C' P : Point)
  (hM : M = orthocenter T)
  (hOrthic : orthicTriangle T = triangle A' B' C')
  (hAltInt : altitudeIntersection T A' = line containing B' and C')
  (hP : P = intersection (line containing A and A') (line containing B' and C')) :
  (PA/PM = A'A/A'M) := by    
  sorry

end ratio_PA_PM_eq_AA_AM_l426_426757


namespace geometric_sequence_when_k_is_neg_one_l426_426968

noncomputable def S (n : ℕ) (k : ℝ) : ℝ := 3^n + k

noncomputable def a (n : ℕ) (k : ℝ) : ℝ :=
  if n = 1 then S 1 k else S n k - S (n-1) k

theorem geometric_sequence_when_k_is_neg_one :
  ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, ∀ m : ℕ, m ≥ 1 → a m (-1) = a 1 (-1) * r^(m-1) :=
by
  sorry

end geometric_sequence_when_k_is_neg_one_l426_426968


namespace min_buses_needed_l426_426095

theorem min_buses_needed (students : ℕ) (cap1 cap2 : ℕ) (h_students : students = 530) (h_cap1 : cap1 = 40) (h_cap2 : cap2 = 45) :
  min (Nat.ceil (students / cap1)) (Nat.ceil (students / cap2)) = 12 :=
  sorry

end min_buses_needed_l426_426095


namespace largest_coefficient_expansion_l426_426212

theorem largest_coefficient_expansion (a : ℝ) (h_sum : (a - 1)^5 = 32) :
  ∃ r : ℕ, (choose 5 r) * (3^(5-r)) * (-1)^r = 270 ∧ (5 - 2 * r) = 1 := 
begin
  sorry
end

end largest_coefficient_expansion_l426_426212


namespace commission_amount_l426_426331

theorem commission_amount 
  (new_avg_commission : ℤ) (increase_in_avg : ℤ) (sales_count : ℤ) 
  (total_commission_before : ℤ) (total_commission_after : ℤ) : 
  new_avg_commission = 400 → increase_in_avg = 150 → sales_count = 6 → 
  total_commission_before = (sales_count - 1) * (new_avg_commission - increase_in_avg) → 
  total_commission_after = sales_count * new_avg_commission → 
  total_commission_after - total_commission_before = 1150 :=
by 
  sorry

end commission_amount_l426_426331


namespace f_prime_slope_l426_426224

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * log x - x

def a_cond (a : ℝ) : Prop := a ≠ 0

def increasing_interval_1 (a : ℝ) : set ℝ :=
if a ≤ -1/8 then Ioi 0 else ∅

def decreasing_interval_1 (a : ℝ) : set ℝ :=
if a ≤ -1/8 then ∅ else ∅

def increasing_interval_2 (a : ℝ) : set ℝ :=
if -1/8 < a ∧ a < 0 then Ioi (1 - sqrt (1 + 8 * a)) / 4 ∪ Ioi (1 + sqrt (1 + 8 * a)) / 4 else ∅

def decreasing_interval_2 (a : ℝ) : set ℝ :=
if -1/8 < a ∧ a < 0 then Ioi (1 - sqrt (1 + 8 * a)) / 4 ∧ Ioi (1 + sqrt (1 + 8 * a)) / 4 else ∅

def increasing_interval_3 (a : ℝ) : set ℝ :=
if a > 0 then Ioi (1 + sqrt (1 + 8 * a )) / 4 else ∅

def decreasing_interval_3 (a : ℝ) : set ℝ :=
if a > 0 then Ioi 0 ∧ Ioi (1 + sqrt (1 + 8 * a )) / 4 else ∅

theorem f_prime_slope (a x1 x2 : ℝ) (h : 0 < x1 ∧ x1 < x2):
    a > 0 → 
    let k := (f x1 a - f x2 a) / (x1 - x2) 
    in f'((x1 + 2 * x2) / 3) > k := sorry

end f_prime_slope_l426_426224


namespace sequence_sum_l426_426191

theorem sequence_sum (a : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, 0 < a n)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 2) = 1 + 1 / a n)
  (h₃ : a 2014 = a 2016) :
  a 13 + a 2016 = 21 / 13 + (1 + Real.sqrt 5) / 2 :=
sorry

end sequence_sum_l426_426191


namespace problem_solution_l426_426501

noncomputable def problem : ℤ :=
  let m := 100 in
  let x := (101^3 / (99 * 100)) - (99^3 / (100 * 101)) in
  Int.floor x

theorem problem_solution : problem = 8 :=
  sorry

end problem_solution_l426_426501


namespace line_through_center_and_parallel_to_l1_l426_426989

-- Define the given circle
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

-- Define the given line l_1
def line_l1 (x y : ℝ) : Prop := 2 * x - 3 * y + 6 = 0

-- Define the line l passing through the center of circle and parallel to l_1
def line_l (x y : ℝ) : Prop := 2 * x - 3 * y - 8 = 0

-- Formalize the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Prove that line l passes through the center of the circle
theorem line_through_center_and_parallel_to_l1 :
  ∀ x y : ℝ, circle_center = (x, y) → line_l x y :=
by
  intro x y h₁
  rw [circle_center] at h₁
  rw [Prod.mk.inj_iff] at h₁
  rcases h₁ with ⟨hx, hy⟩
  subst hx
  subst hy
  -- line_l 1 (-2) is true
  unfold line_l
  norm_num
  sorry

end line_through_center_and_parallel_to_l1_l426_426989


namespace Nadia_flower_shop_l426_426698

open Nat

/-- Nadia bought 20 roses. Each rose costs $5. Each lily costs twice the cost of a rose.
Nadia used $250 to buy both types of flowers. Prove that the ratio of the number of lilies to the number of roses is 3:4. -/
theorem Nadia_flower_shop :
  let roses := 20
  let cost_rose := 5
  let total_cost := 250
  let lily_cost := 2 * cost_rose
  ∃ (lilies : ℕ), (lilies : ℕ) = (total_cost - roses * cost_rose) / lily_cost ∧ Nat.gcd (lilies / gcd lilies roses) (roses / gcd lilies roses) = gcd 3 4 :=
by
  sorry

end Nadia_flower_shop_l426_426698


namespace find_r_l426_426626

variable {p q r x y : ℝ}

-- Definition of the cubic function
def cubic_function (x p q r: ℝ) : ℝ := x^3 + 3 * p * x^2 + 3 * q * x + r

-- The problem statement
theorem find_r (p q: ℝ) : 
  (cubic_function (-p) p q r = -27) →
  (2 * p^3 - 3 * q * p + r = -27) :=
begin
  sorry
end

end find_r_l426_426626


namespace total_bill_is_correct_l426_426492

def number_of_adults : ℕ := 2
def number_of_children : ℕ := 5
def meal_cost : ℕ := 8

-- Define total number of people
def total_people : ℕ := number_of_adults + number_of_children

-- Define the total bill
def total_bill : ℕ := total_people * meal_cost

-- Theorem stating the total bill amount
theorem total_bill_is_correct : total_bill = 56 := by
  sorry

end total_bill_is_correct_l426_426492


namespace probability_four_ones_in_five_rolls_l426_426259

-- Define the probability of rolling a 1 on a fair six-sided die
def prob_one_roll_one : ℚ := 1 / 6

-- Define the probability of not rolling a 1 on a fair six-sided die
def prob_one_roll_not_one : ℚ := 5 / 6

-- Define the number of successes needed, here 4 ones in 5 rolls
def num_successes : ℕ := 4

-- Define the total number of trials, here 5 rolls
def num_trials : ℕ := 5

-- Binomial probability calculation for 4 successes in 5 trials with probability of success prob_one_roll_one
def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_four_ones_in_five_rolls : binomial_prob num_trials num_successes prob_one_roll_one = 25 / 7776 := 
by
  sorry

end probability_four_ones_in_five_rolls_l426_426259


namespace cookies_taken_in_four_days_l426_426404

def initial_cookies : ℕ := 70
def cookies_left : ℕ := 28
def days_in_week : ℕ := 7
def days_taken : ℕ := 4
def daily_cookies_taken (total_cookies_taken : ℕ) : ℕ := total_cookies_taken / days_in_week
def total_cookies_taken : ℕ := initial_cookies - cookies_left

theorem cookies_taken_in_four_days :
  daily_cookies_taken total_cookies_taken * days_taken = 24 := by
  sorry

end cookies_taken_in_four_days_l426_426404


namespace S6_geometric_sum_l426_426024

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem S6_geometric_sum (a r : ℝ)
    (sum_n : ℕ → ℝ)
    (geo_seq : ∀ n, sum_n n = geometric_sequence_sum a r n)
    (S2 : sum_n 2 = 6)
    (S4 : sum_n 4 = 30) :
    sum_n 6 = 126 := 
by
  sorry

end S6_geometric_sum_l426_426024


namespace maximum_expression_value_l426_426210

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 - t, 1 + t)

def curve_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 - x = 0

def line_polar (ρ θ : ℝ) : Prop :=
  ρ * (Real.cos θ + Real.sin θ) = 3

def curve_C1_polar (ρ θ : ℝ) : Prop :=
  ρ = Real.cos θ

def OM_value (α : ℝ) : ℝ :=
  3 / (Real.cos α + Real.sin α)

def ON_value (α : ℝ) : ℝ :=
  Real.cos α

def expression_value (α : ℝ) : ℝ :=
  3 / OM_value α + ON_value α

theorem maximum_expression_value :
  ∃ α : ℝ, 0 < α ∧ α < Real.pi/2 ∧ expression_value α = Real.sqrt 5 :=
sorry

end maximum_expression_value_l426_426210


namespace ants_meet_again_l426_426795

/-- Two ants start crawling from point P, one along a larger circle with radius 6 inches at a speed 
of 4π inches per minute, and the other along a smaller circle with radius 3 inches at a speed of 3π inches per minute. 
This statement proves that they will meet again at point P after 6 minutes. -/
theorem ants_meet_again : 
  let radius_large := 6
  let radius_small := 3
  let speed_large := 4 * Real.pi
  let speed_small := 3 * Real.pi
  let circumference_large := 2 * Real.pi * radius_large
  let circumference_small := 2 * Real.pi * radius_small
  let time_large := circumference_large / speed_large
  let time_small := circumference_small / speed_small
  Nat.lcm (Int.toNat time_large) (Int.toNat time_small) = 6 :=
by
  sorry

end ants_meet_again_l426_426795


namespace find_pairs_l426_426540

noncomputable def gcd_condition_satisfy (x y : ℕ) : Prop :=
∀ n : ℕ, Nat.gcd (n * (Nat.factorial x - x * y - x - y + 2) + 2) (n * (Nat.factorial x - x * y - x - y + 3) + 3) > 1

theorem find_pairs (q : ℕ) (hq : q > 3) (hq_prime : Nat.prime q) :
  let x := q - 1 in
  let y := (Nat.factorial (q - 1) - (q - 1)) / q in
  gcd_condition_satisfy x y :=
sorry

end find_pairs_l426_426540


namespace sum_of_coefficients_l426_426744

theorem sum_of_coefficients (a b c d e x : ℝ) (h : 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) :
  a + b + c + d + e = 36 :=
by
  sorry

end sum_of_coefficients_l426_426744


namespace phase_shift_sin_l426_426887

theorem phase_shift_sin (x : ℝ) : 
  let B := 4
  let C := - (π / 2)
  let φ := - C / B
  φ = π / 8 := 
by 
  sorry

end phase_shift_sin_l426_426887


namespace original_fraction_is_two_thirds_l426_426258

theorem original_fraction_is_two_thirds
  (x y : ℕ)
  (h1 : x / (y + 1) = 1 / 2)
  (h2 : (x + 1) / y = 1) :
  x / y = 2 / 3 := by
  sorry

end original_fraction_is_two_thirds_l426_426258


namespace range_of_f_real_l426_426983

noncomputable def f (a x : ℝ) : ℝ := log a (x + a / x - 1)

theorem range_of_f_real (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : ∀ y : ℝ, ∃ x : ℝ, f a x = y) : 0 < a ∧ a ≤ 1 / 4 :=
by
  sorry

end range_of_f_real_l426_426983


namespace algebra_correct_option_B_l426_426431

theorem algebra_correct_option_B (a b c : ℝ) (h : b * (c^2 + 1) ≠ 0) : 
  (a * (c^2 + 1)) / (b * (c^2 + 1)) = a / b := 
by
  -- Skipping the proof to focus on the statement
  sorry

end algebra_correct_option_B_l426_426431


namespace train_length_proof_l426_426107

variables (speed_train : ℝ) (length_tunnel : ℝ) (time_to_pass : ℝ)

def length_of_train (speed_train length_tunnel : ℝ) (time_to_pass : ℝ) : ℝ :=
  (speed_train / 60) * time_to_pass - length_tunnel

theorem train_length_proof : 
  speed_train = 72 → 
  length_tunnel = 1.7 → 
  time_to_pass = 1.5 → 
  length_of_train speed_train length_tunnel time_to_pass = 0.1 :=
by 
  intros h_speed h_tunnel h_time
  rw [h_speed, h_tunnel, h_time]
  simp [length_of_train]
  norm_num
  sorry

end train_length_proof_l426_426107


namespace sequence_sum_product_l426_426154

theorem sequence_sum_product :
  let sum := 1342 + 2431 + 3124 + 4213 in
  sum * 3 = 33330 :=
by
  sorry

end sequence_sum_product_l426_426154


namespace min_value_of_f_in_strip_l426_426942

noncomputable def f (x y : ℝ) : ℝ :=
(x - y)^2 + (real.sqrt (2 - x^2) - 9 / y)^2

theorem min_value_of_f_in_strip :
  (∀ (x y : ℝ), 0 < x ∧ x < real.sqrt 2 ∧ 0 < y → f x y ≥ 8) ∧ 
  ∃ (x y : ℝ), 0 < x ∧ x < real.sqrt 2 ∧ 0 < y ∧ f x y = 8 :=
begin
  split,
  { intros x y hx,
    sorry },  -- Proof that f(x, y) ≥ 8 given the conditions
  { use [1, 3],
    split,
    { split, linarith, norm_num },
    split,
    { linarith },
    { simp [f, real.sqrt_one, pow_two], norm_num } }  -- Proof that the minimum value 8 is achieved at (1, 3)
end

end min_value_of_f_in_strip_l426_426942


namespace theater_rows_25_l426_426103

theorem theater_rows_25 (n : ℕ) (x : ℕ) (k : ℕ) (h : n = 1000) (h1 : k > 16) (h2 : (2 * x + k) * (k + 1) = 2000) : (k + 1) = 25 :=
by
  -- The proof goes here, which we omit for the problem statement.
  sorry

end theater_rows_25_l426_426103


namespace sum_of_series_l426_426894

theorem sum_of_series :
  (∑ (a : ℕ) in finset.Ico 1 (finset.univ), 
   (∑ (b : ℕ) in finset.Ico (a+1) (finset.univ),
    (∑ (c : ℕ) in finset.Ico (b+1) (finset.univ),
     (∑ (d : ℕ) in finset.Ico (c+1) (finset.univ),
      (1 : ℝ) / (2^a * 3^b * 5^c * 7^d))))) = 1 / 45435456 := by
  sorry

end sum_of_series_l426_426894


namespace length_of_faster_train_l426_426443

theorem length_of_faster_train (v_fast v_slow : ℕ) (t : ℕ) (h1 : v_fast = 90) (h2 : v_slow = 36) (h3 : t = 29) : 
  let relative_speed_kmph := v_fast - v_slow in
  let relative_speed_mps := relative_speed_kmph * 5 / 18 in
  let length := relative_speed_mps * t in
  length = 435 :=
by 
  sorry

end length_of_faster_train_l426_426443


namespace ticket_cost_l426_426857

-- Conditions
def seats : ℕ := 400
def capacity_percentage : ℝ := 0.8
def performances : ℕ := 3
def total_revenue : ℝ := 28800

-- Question: Prove that the cost of each ticket is $30
theorem ticket_cost : (total_revenue / (seats * capacity_percentage * performances)) = 30 := 
by
  sorry

end ticket_cost_l426_426857


namespace car_price_increase_l426_426735

-- Define the initial costs based on the ratio 4:3:2
def initial_raw_material_cost (x : ℝ) : ℝ := 4 * x
def initial_labor_cost (x : ℝ) : ℝ := 3 * x
def initial_overhead_cost (x : ℝ) : ℝ := 2 * x

-- Define the costs for the next year based on the given percentage changes
def next_year_raw_material_cost (x : ℝ) : ℝ := 4 * x * 1.10
def next_year_labor_cost (x : ℝ) : ℝ := 3 * x * 1.08
def next_year_overhead_cost (x : ℝ) : ℝ := 2 * x * 0.95

-- Define the total costs
def initial_total_cost (x : ℝ) : ℝ := initial_raw_material_cost x + initial_labor_cost x + initial_overhead_cost x
def next_year_total_cost (x : ℝ) : ℝ := next_year_raw_material_cost x + next_year_labor_cost x + next_year_overhead_cost x

-- Define the percentage increase calculation
def percentage_increase (initial_cost next_year_cost : ℝ) : ℝ := (next_year_cost - initial_cost) / initial_cost * 100

-- The theorem to prove the percentage increase is 6%
theorem car_price_increase (x : ℝ) : percentage_increase (initial_total_cost x) (next_year_total_cost x) = 6 := sorry

end car_price_increase_l426_426735


namespace rainfall_difference_l426_426783

theorem rainfall_difference :
  let day1 := 26
  let day2 := 34
  let day3 := day2 - 12
  let total_rainfall := day1 + day2 + day3
  let average_rainfall := 140
  (average_rainfall - total_rainfall = 58) :=
by
  sorry

end rainfall_difference_l426_426783


namespace calculate_v_sum_l426_426509

def v (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem calculate_v_sum :
  v (2) + v (-2) + v (1) + v (-1) = 4 :=
by
  sorry

end calculate_v_sum_l426_426509


namespace calories_consumed_l426_426880

theorem calories_consumed (slices : ℕ) (calories_per_slice : ℕ) (half_pizza : ℕ) :
  slices = 8 → calories_per_slice = 300 → half_pizza = slices / 2 → 
  half_pizza * calories_per_slice = 1200 :=
by
  intros h_slices h_calories_per_slice h_half_pizza
  rw [h_slices, h_calories_per_slice] at h_half_pizza
  rw [h_slices, h_calories_per_slice]
  sorry

end calories_consumed_l426_426880


namespace inequality_k_l426_426380

theorem inequality_k (k : ℝ) : (∀ x : ℝ, exp x ≥ k + x) → k ≤ 1 :=
sorry

end inequality_k_l426_426380


namespace inequality_nonnegative_reals_l426_426553

theorem inequality_nonnegative_reals (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  x^2 * y^2 + x^2 * y + x * y^2 ≤ x^4 * y + x + y^4 :=
sorry

end inequality_nonnegative_reals_l426_426553


namespace peculiar_poly_q1_value_l426_426896

-- Define a peculiar quadratic polynomial with parameters b and c
def peculiar_poly (b c : ℝ) := λ x : ℝ, x^2 - b * x + c

-- Define the condition for a polynomial to be peculiar
def is_peculiar (b c : ℝ) := 
  let q := peculiar_poly b c in
  ∃ r1 r2 r3 r4 : ℝ, r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧ 
  ∀ x : ℝ, (q (q x) = 0) → (x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4)

-- Define the maximum product of roots condition
noncomputable def max_product_of_roots (b c : ℝ) := ∃ (max_b max_c : ℝ), is_peculiar max_b max_c ∧ c = max_c -- Simplified

-- Final statement for the Lean theorem
theorem peculiar_poly_q1_value (b c : ℝ) (x : ℝ)
  (h1: is_peculiar b c)
  (h2: max_product_of_roots b c) :
    peculiar_poly b c 1 = PICK_AMONG_CHOICES :=
sorry

end peculiar_poly_q1_value_l426_426896


namespace direct_variation_y_value_l426_426627

theorem direct_variation_y_value (x y k : ℝ) (h1 : y = k * x) (h2 : ∀ x, x = 5 → y = 10) 
                                 (h3 : ∀ x, x < 0 → k = 4) (hx : x = -6) : y = -24 :=
sorry

end direct_variation_y_value_l426_426627


namespace sarah_cupcakes_l426_426149

theorem sarah_cupcakes (c k d : ℕ) (h1 : c + k = 6) (h2 : 90 * c + 40 * k = 100 * d) : c = 4 ∨ c = 6 :=
by {
  sorry -- Proof is omitted as requested.
}

end sarah_cupcakes_l426_426149


namespace part1_part2_l426_426496

-- Define y based on given condition
def y_def (x : ℝ) : ℝ := sqrt (x - 2) + sqrt (2 - x) + 3

-- Part 1: Prove that x - y = -1 for x = 2
theorem part1 (x : ℝ) (hx : x = 2) : x - y_def x = -1 :=
by 
  sorry

-- Part 2: Simplify and prove the value when x = √2
def expr (x : ℝ) : ℝ := x / (x - 2) / (2 + x - 4 / (2 - x))

theorem part2 (x : ℝ) (hx : x = sqrt 2) : expr x = sqrt 2 / 2 :=
by 
  sorry

end part1_part2_l426_426496


namespace total_pies_baked_in_7_days_l426_426911

-- Define the baking rates (pies per day)
def Eddie_rate : Nat := 3
def Sister_rate : Nat := 6
def Mother_rate : Nat := 8

-- Define the duration in days
def duration : Nat := 7

-- Define the total number of pies baked in 7 days
def total_pies : Nat := Eddie_rate * duration + Sister_rate * duration + Mother_rate * duration

-- Prove the total number of pies is 119
theorem total_pies_baked_in_7_days : total_pies = 119 := by
  -- The proof will be filled here, adding sorry to skip it for now
  sorry

end total_pies_baked_in_7_days_l426_426911


namespace quadrilateral_parallelogram_l426_426069

variables {A B C D : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
variables {points : Type} [Add points] [HasSmul ℝ points] [HasSub points] [HasVAdd points]

-- Define the conditions
def midpoint (M : points) (P Q : points) : Prop :=
  M = (P + Q) / 2

def bisects (A C M B D : points) : Prop :=
  A = (B + D) / 2 ∧ M = (A + C) / 2

def sides_condition (AB AD BC CD : ℝ) : Prop :=
  BC + CD = AB + AD

-- Prove that the quadrilateral is a parallelogram
theorem quadrilateral_parallelogram
  (A B C D M : points)
  (mid_AC : midpoint M A C)
  (mid_BD : midpoint M B D)
  (sides_eq : sides_condition (∥B - A∥) (∥D - A∥) (∥C - B∥) (∥C - D∥)) :
  ∀ (ABCD : Prop), is_parallelogram ABCD :=
begin
  sorry, -- Placeholder for the actual proof
end

end quadrilateral_parallelogram_l426_426069


namespace root_expression_value_l426_426688

theorem root_expression_value :
  let p q r s : ℂ := sorry,
    h : (∀ x, x^4 + 6*x^3 - 4*x^2 + 7*x + 3 = 0) → (p*x^(3) + q*x^(2) + r*x + s = 0)
  in (1 / (p * q) + 1 / (p * r) + 1 / (p * s) + 1 / (q * r) + 1 / (q * s) + 1 / (r * s)) = -4 / 3 :=
sorry

end root_expression_value_l426_426688


namespace quadrilateral_is_cyclic_l426_426676

noncomputable def cyclic_quadrilateral (A B C D E N : Type) (AB AD BC : ℝ) (CD : ℝ) (AB_GT_CD : AB > CD)
  (isosceles_ABCD : AB = AD ∧ AD = BC) (parallel_AB_CD : AB ∥ CD)
  (intersection_E : E = AC ∩ BD) (symmetric_N : symmetric E B AC) : Prop :=
  cyclic_quadrilateral (quadrilateral.mk A N D E)

-- conditions, definitions, and theorem
variable (A B C D E N : Type)

theorem quadrilateral_is_cyclic
  (isosceles_ABCD : AB = AD ∧ AD = BC)
  (parallel_AB_CD : AB ∥ CD)
  (AB_GT_CD : AB > CD)
  (intersection_E : E = AC ∩ BD)
  (symmetric_N : symmetric B N AC) :
  cyclic_quadrilateral A N D E :=
sorry

end quadrilateral_is_cyclic_l426_426676


namespace fred_current_money_l426_426306

-- Conditions
def initial_amount_fred : ℕ := 19
def earned_amount_fred : ℕ := 21

-- Question and Proof
theorem fred_current_money : initial_amount_fred + earned_amount_fred = 40 :=
by sorry

end fred_current_money_l426_426306


namespace polynomial_square_b_value_l426_426756

theorem polynomial_square_b_value (a b p q : ℝ) (h : (∀ x : ℝ, x^4 + x^3 - x^2 + a * x + b = (x^2 + p * x + q)^2)) : b = 25/64 := by
  sorry

end polynomial_square_b_value_l426_426756


namespace find_polar_coordinates_l426_426980

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ := 
  (real.sqrt (x^2 + y^2), real.arctan2 y x)

theorem find_polar_coordinates :
  (∃ x y : ℝ, 6 = 2 * x ∧ -3 = real.sqrt 3 * y ∧ polar_coordinates x y = (2 * real.sqrt 3, 11 * real.pi / 6)) := 
by
  sorry

end find_polar_coordinates_l426_426980


namespace relationship_between_a_b_c_l426_426181

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := 5^(1/3)
noncomputable def c : ℝ := (2 + Real.exp 1)^(1 / Real.exp 1)

theorem relationship_between_a_b_c : b < c ∧ c < a :=
by
  -- Explicitly define constants to avoid symbolic errors
  let e := Real.exp 1
  have h1 : 2 = a := rfl
  have h2 : 5^(1/3) = b := rfl
  have h3 : (2+e)^(1/e) = c := rfl

  -- Relevant inequalities are shown in the solution
  have h4 : 2 < e := Real.two_lt_exp_one
  have h5 : e < 3 := Real.exp_one_lt_3
  have h6 : f(2) > f(e) > f(3) where
    f := λ x : ℝ, (2 + x)^(1 / x)

  exact sorry -- Conclusion will be derived here

end relationship_between_a_b_c_l426_426181


namespace f_has_two_turning_points_l426_426314

def f (x : ℝ) (a : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ a then (1/a)*x else (1/(1-a))*(1-x)

def is_turning_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f (f x) = x ∧ f x ≠ x

theorem f_has_two_turning_points (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∃ x1 x2 : ℝ, is_turning_point (λ x => f x a) x1 ∧ 
               is_turning_point (λ x => f x a) x2 ∧ 
               x1 ≠ x2 ∧ 
               ∀ x : ℝ, is_turning_point (λ x => f x a) x → 
                         x = x1 ∨ x = x2 ∧ 
               x1 = a / (1 - a^2 + a) ∧ x2 = 1 / (1 - a^2 + a) :=
  by
  sorry

end f_has_two_turning_points_l426_426314


namespace number_of_solutions_is_zero_l426_426519

theorem number_of_solutions_is_zero : 
  ∀ x : ℝ, (x ≠ 0 ∧ x ≠ 5) → (3 * x^2 - 15 * x) / (x^2 - 5 * x) ≠ x - 2 :=
by
  sorry

end number_of_solutions_is_zero_l426_426519


namespace equation_of_AB_tangent_line_l426_426908

noncomputable def circle1 : (ℝ × ℝ) → ℝ := fun p => (p.1 - 1)^2 + (p.2)^2 - 1
noncomputable def circle2 : (ℝ × ℝ) → ℝ := fun p => (p.1 - 1)^2 + (p.2 + 1)^2 - 1

theorem equation_of_AB_tangent_line : ∀ (A B : ℝ × ℝ),
  circle1 A = 0 →
  circle1 B = 0 →
  circle2 A = 0 →
  circle2 B = 0 →
  (∃ k : ℝ, k = -1 / 2 ∧ ∀ p : ℝ × ℝ, p ∈ [A, B] → p.2 = k) :=
begin
  sorry
end

end equation_of_AB_tangent_line_l426_426908


namespace trees_per_day_l426_426348

def blocks_per_tree := 3
def total_blocks := 30
def days := 5

theorem trees_per_day : (total_blocks / days) / blocks_per_tree = 2 := by
  sorry

end trees_per_day_l426_426348


namespace characterizes_f_l426_426158

noncomputable theory

def satisfies_conditions (f : ℤ → ℤ) :=
  (f 1 ≠ f (-1)) ∧ ∀ m n : ℤ, (f (m + n))^2 ∣ (f m - f n)

theorem characterizes_f (f : ℤ → ℤ) : satisfies_conditions f →
  (∀ x : ℤ, f x ∈ {-1, 1} ∨ f x ∈ {-2, 2}) ∧ (f 1 = -f (-1)) :=
sorry

end characterizes_f_l426_426158


namespace max_balls_in_cube_l426_426803

noncomputable def volume_cube (side_length : ℝ) : ℝ := side_length ^ 3
noncomputable def volume_ball (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

theorem max_balls_in_cube (side_length : ℝ) (radius : ℝ) (h_cube : side_length = 10) (h_ball : radius = 3) :
  Nat.floor (volume_cube side_length / volume_ball radius) = 8 :=
by
  rw [h_cube, h_ball]
  have V_cube : volume_cube 10 = 1000 := by norm_num [volume_cube]
  have V_ball : volume_ball 3 = 36 * Real.pi := by norm_num [volume_ball, Real.pi]
  sorry

end max_balls_in_cube_l426_426803


namespace chord_length_l426_426082

theorem chord_length {r : ℝ} (h : r = 15) : 
  ∃ (CD : ℝ), CD = 26 * Real.sqrt 3 :=
by
  sorry

end chord_length_l426_426082


namespace sum_a_condition_l426_426184

def a (i j : Nat) : Int :=
  if j % i == 0 then 1 else -1

theorem sum_a_condition :
  (∑ j in Finrange 2 5, a 3 j) + (∑ i in Finrange 2 4, a i 4) = -1 :=
sorry

end sum_a_condition_l426_426184


namespace sum_of_ratios_geq_half_sum_l426_426317

variable {n : ℕ}
variable {a b : Fin n → ℝ}

theorem sum_of_ratios_geq_half_sum (h_pos_a : ∀ i, 0 < a i) (h_pos_b : ∀ i, 0 < b i)
  (h_sum_eq : (∑ i, a i) = (∑ i, b i)) :
  (∑ i, a i ^ 2 / (a i + b i)) ≥ 1 / 2 * (∑ i, a i) :=
by
  -- Proof will be inserted here
  sorry

end sum_of_ratios_geq_half_sum_l426_426317


namespace dice_probability_sum_24_l426_426951

-- Define the probability of each die showing a 6 as 1/6
def die_probability : ℝ := 1 / 6

-- Define the event of sum of four dice showing 24
def event_sum_24 := (die_probability ^ 4 = 1 / 1296)

theorem dice_probability_sum_24 : event_sum_24 :=
by
  sorry

end dice_probability_sum_24_l426_426951


namespace remainder_1425_1427_1429_mod_12_l426_426425

theorem remainder_1425_1427_1429_mod_12 :
  (1425 * 1427 * 1429) % 12 = 11 :=
by
  sorry

end remainder_1425_1427_1429_mod_12_l426_426425


namespace number_of_real_roots_l426_426751

theorem number_of_real_roots : 
  let number_of_roots := (finset.range 100).sum (λ k, if ∃ x, (k:ℝ) / 100 = real.sin x then 1 else 0)
  ∃ n, n = 63 ∧ number_of_roots = n := 
sorry

end number_of_real_roots_l426_426751


namespace problem1_problem2_l426_426637

theorem problem1 (b : ℤ) (h1 : 6 % 2 = b % 2) (h2 : 0 < b ∧ b < 6) : b = 2 ∨ b = 4 :=
sorry

theorem problem2 (m a : ℤ) (h1 : a % m = 10 % m) (h2 : a > 10) (h3 : m > 1)
                (h4 : ∑ i in finset.range (m - 1), (10 + (i + 1) * m) = 60 * (m - 1)) : m = 10 :=
sorry

end problem1_problem2_l426_426637


namespace wire_length_l426_426941

theorem wire_length (A : ℝ) (n : ℕ) (hA : A = 24336) (hn : n = 13) : 
  let s := Real.sqrt 24336 in 
  let P := 4 * s in 
  n * P = 8112 := by
  have s_def : s = Real.sqrt A := rfl
  have P_def : P = 4 * s := rfl
  sorry

end wire_length_l426_426941


namespace train_passing_time_l426_426860

def train_length : ℝ := 375
def train_speed_km_per_hr : ℝ := 120
def conversion_factor : ℝ := 5 / 18 -- km/hr to m/s conversion factor
def train_speed_m_per_s : ℝ := train_speed_km_per_hr * conversion_factor
def expected_time : ℝ := 11.25

theorem train_passing_time : train_length / train_speed_m_per_s = expected_time :=
by
  sorry

end train_passing_time_l426_426860


namespace angle_decrease_percentage_l426_426390

theorem angle_decrease_percentage (θ₁ θ₂ : ℝ)
  (h1 : θ₁ + θ₂ = 90)
  (h2 : θ₁ = 3 * 9)
  (h3 : θ₂ = 7 * 9) :
  let θ₁' := θ₁ * 1.20,
      θ₂' := 90 - θ₁' in
  (63 - θ₂') / 63 * 100 ≈ 8.57 := by
  sorry

end angle_decrease_percentage_l426_426390


namespace min_rubles_for_50_points_l426_426046

theorem min_rubles_for_50_points : ∃ (n : ℕ), minimal_rubles n ∧ n = 11 := by
  sorry

def minimal_rubles (n : ℕ) : Prop :=
  ∀ m, (steps_to_reach_50 m) ∧ (total_cost m ≤ n)

def steps_to_reach_50 (steps : list ℕ) : Prop :=
  ∃ initial_score : ℕ, initial_score = 0 ∧ 
  count_steps_to_50 initial_score steps = 50

def count_steps_to_50 (score : ℕ) (steps : list ℕ) : ℕ :=
  match steps with
  | [] => score
  | h :: t =>
    if h = 1 then
      count_steps_to_50 (score + 1) t
    else if h = 2 then 
      count_steps_to_50 (2 * score) t
    else
      score  -- Invalid step

end min_rubles_for_50_points_l426_426046


namespace real_part_of_conjugate_of_z_l426_426965

theorem real_part_of_conjugate_of_z :
  let z := (1 + 2 * complex.i) / (1 - complex.i) in
  real.re (complex.conj z) = - 1 / 2 :=
by
  sorry

end real_part_of_conjugate_of_z_l426_426965


namespace percent_increase_march_to_june_l426_426818

-- Define the conditions given in the problem
def profit_march := P : ℝ
def profit_april := profit_march * 1.3
def profit_may := profit_april * 0.8
def profit_june := profit_may * 1.5

-- Define the statement of the problem
theorem percent_increase_march_to_june (P : ℝ) (h: P > 0) :
  ((profit_june - profit_march) / profit_march) * 100 = 56 :=
-- The proof goes here
by
  sorry

end percent_increase_march_to_june_l426_426818


namespace millie_initial_bracelets_l426_426332

theorem millie_initial_bracelets (n : ℕ) (h1 : n - 2 = 7) : n = 9 :=
sorry

end millie_initial_bracelets_l426_426332


namespace loss_of_50_denoted_as_minus_50_l426_426281

def is_profit (x : Int) : Prop :=
  x > 0

def is_loss (x : Int) : Prop :=
  x < 0

theorem loss_of_50_denoted_as_minus_50 : is_loss (-50) :=
  by
    -- proof steps would go here
    sorry

end loss_of_50_denoted_as_minus_50_l426_426281


namespace geometric_series_arithmetic_sequence_l426_426274

noncomputable def geometric_seq_ratio (a : ℕ → ℝ) (q : ℝ) : Prop := 
∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_series_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_seq_ratio a q)
  (h_pos : ∀ n, a n > 0)
  (h_arith : a 1 = (a 0 + 2 * a 1) / 2) :
  a 5 / a 3 = 3 + 2 * Real.sqrt 2 :=
sorry

end geometric_series_arithmetic_sequence_l426_426274


namespace every_natural_number_appears_exactly_once_in_sequence_l426_426096

noncomputable def a : ℕ → ℕ
| 0 := 1
| 1 := 2
| 2 := 3
| n + 3 := Nat.some (Exists.intro (Nat.find_min (λ x, ∀ i < n + 3, x ≠ a i ∧ Nat.coprime x (a (n + 2)) ∧ ¬Nat.coprime x (a (n + 1)))) sorry)

theorem every_natural_number_appears_exactly_once_in_sequence :
  ∀ n : ℕ, ∃ k : ℕ, a k = n ∧ (∀ i : ℕ, a i = n ↔ i = k) :=
sorry

end every_natural_number_appears_exactly_once_in_sequence_l426_426096


namespace second_day_speed_faster_l426_426088

def first_day_distance := 18
def first_day_speed := 3
def first_day_time := first_day_distance / first_day_speed
def second_day_time := first_day_time - 1
def third_day_speed := 5
def third_day_time := 3
def third_day_distance := third_day_speed * third_day_time
def total_distance := 53

theorem second_day_speed_faster :
  ∃ r2, (first_day_distance + (second_day_time * r2) + third_day_distance = total_distance) → (r2 - first_day_speed = 1) :=
by
  sorry

end second_day_speed_faster_l426_426088


namespace students_sampled_C_is_40_l426_426837

-- Define the given conditions as constants.
constant total_students : ℕ := 1200
constant students_A : ℕ := 380
constant students_B : ℕ := 420
constant sample_size : ℕ := 120

-- Define the calculation for students in major C based on the conditions.
def students_C : ℕ := total_students - students_A - students_B

-- Define the calculation for the sampling rate.
def sampling_rate : ℚ := sample_size / total_students

-- Define the number of students to be sampled from major C.
def students_sampled_C : ℕ := students_C * sampling_rate.toInt

-- The theorem we need to prove.
theorem students_sampled_C_is_40 : students_sampled_C = 40 := by
  -- place proof here
  sorry

end students_sampled_C_is_40_l426_426837


namespace set_intersection_eq_l426_426231

noncomputable def A : Set ℕ := {x : ℕ | - (x:ℤ)^2 + x + 6 > 0}
def B : Set ℤ := {-1, 0, 1, 2}

theorem set_intersection_eq : (A ∩ B) = ({0, 1, 2} : Set ℤ) :=
by sorry

end set_intersection_eq_l426_426231


namespace intersecting_lines_configurations_l426_426232

theorem intersecting_lines_configurations :
  ∀ (A B : Plane) (a b : Line),
  -- Conditions
  (AngleBetweenLines a b = 30) →
  (AngleBetweenProjections a b = 30) →
  (AngleBetweenPlaneAndFirstPlane a b = 60) →
  -- Conclusion
  ∃ n : ℕ, n = 8 :=
begin
  sorry
end

end intersecting_lines_configurations_l426_426232


namespace number_of_solutions_l426_426547

-- Define the conditions
variables (m n : ℤ)

-- Statement using the conditions
theorem number_of_solutions :
  let count_pairs := {p : ℤ × ℤ | p.1 * p.2 ≥ 0 ∧ p.1 ^ 3 + p.2 ^ 3 + 104 * p.1 * p.2 = 36 ^ 3}.card in
  count_pairs = 38 :=
by {
  sorry
}

end number_of_solutions_l426_426547


namespace find_number_l426_426264

-- We define n, x, y as real numbers
variables (n x y : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := n * (x - y) = 4
def condition2 : Prop := 6 * x - 3 * y = 12

-- Define the theorem we need to prove: If the conditions hold, then n = 2
theorem find_number (h1 : condition1 n x y) (h2 : condition2 x y) : n = 2 := 
sorry

end find_number_l426_426264


namespace tiered_water_pricing_l426_426652

theorem tiered_water_pricing (x : ℝ) (y : ℝ) : 
  (∀ z, 0 ≤ z ∧ z ≤ 12 → y = 3 * z ∨
        12 < z ∧ z ≤ 18 → y = 36 + 6 * (z - 12) ∨
        18 < z → y = 72 + 9 * (z - 18)) → 
  y = 54 → 
  x = 15 :=
by
  sorry

end tiered_water_pricing_l426_426652


namespace possible_values_of_r_l426_426017

theorem possible_values_of_r :
  let lower_bound := (1 / 3 + 3 / 8) / 2,
      upper_bound := (3 / 10 + 3 / 8) / 2 in 
  (∃ a b c d : Fin 10, 
      lower_bound ≤ 0 + a / 10 + b / 100 + c / 1000 + d / 10000 ∧
      0 + a / 10 + b / 100 + c / 1000 + d / 10000 ≤ upper_bound) = 334 := 
by
  sorry

end possible_values_of_r_l426_426017


namespace olya_guarantees_win_l426_426855

def sasha_wins (n : ℕ) : Prop :=
  if n = 13 ∨ n = 14 ∨ n = 15 then True else False

theorem olya_guarantees_win (n : ℕ) :
  ¬sasha_wins n ↔ ¬(n = 13 ∨ n = 14 ∨ n = 15) :=
begin
  sorry
end

end olya_guarantees_win_l426_426855


namespace frank_allowance_l426_426956

theorem frank_allowance (savings : ℕ) (cost_per_toy : ℕ) (num_toys : ℕ) (total_money_needed : ℕ) (allowance : ℕ)
(h1 : savings = 3)
(h2 : cost_per_toy = 8)
(h3 : num_tys = 5)
(h4 : total_money_needed = num_toys * cost_per_toy)
(h5 : total_money_needed = savings + allowance) :
allowance = 37 := {
savings 3
cost_per_toy 8
num_toys 5
total_money_needed 40
allowance 37
}

end frank_allowance_l426_426956


namespace average_pieces_proof_l426_426892

-- Definitions based on conditions
def number_of_cookies : ℕ := 48
def chocolate_chips : ℕ := 108
def m_and_ms : ℕ := chocolate_chips / 3
def white_chocolate_chips : ℕ := m_and_ms / 2
def raisins : ℕ := 2 * white_chocolate_chips

-- Average number of chocolate pieces and raisins per cookie
def average_pieces_per_cookie : ℝ := 
  (chocolate_chips + m_and_ms + white_chocolate_chips + raisins).toReal / number_of_cookies

-- The assertion
theorem average_pieces_proof : average_pieces_per_cookie = 4.125 := by
  sorry

end average_pieces_proof_l426_426892


namespace gravel_cost_l426_426064

-- Variables and Conditions
variables (length width pathWidth : ℝ) (costPerSqMeterPaise : ℝ)
def totalArea : ℝ := length * width
def pathLength : ℝ := length - 2 * pathWidth
def pathWidthInside : ℝ := width - 2 * pathWidth
def insideArea : ℝ := pathLength * pathWidthInside
def pathArea : ℝ := totalArea - insideArea
def costPerSqMeter : ℝ := costPerSqMeterPaise / 100  -- converting paise to rupees
def totalCost : ℝ := pathArea * costPerSqMeter

theorem gravel_cost 
  (h_length : length = 110)
  (h_width : width = 65)
  (h_pathWidth : pathWidth = 2.5)
  (h_costPerSqMeterPaise : costPerSqMeterPaise = 60) :
  totalCost length width pathWidth costPerSqMeterPaise = 510 := by
  -- The proof would go here, for now we'll use sorry to skip the proof steps
  sorry

end gravel_cost_l426_426064


namespace real_solution_count_l426_426943

def a : ℝ := 2006 + 1 / 2006
def eq (x : ℝ) : Prop := x^2 + 1 / x^2 = a

theorem real_solution_count : (∃ (x : ℝ), eq x) → ∃ n : ℕ, n = 4 :=
by 
  sorry

end real_solution_count_l426_426943


namespace twenty_percent_correct_l426_426814

def certain_number := 400
def forty_percent (x : ℕ) : ℕ := 40 * x / 100
def twenty_percent_of_certain_number (x : ℕ) : ℕ := 20 * x / 100

theorem twenty_percent_correct : 
  (∃ x : ℕ, forty_percent x = 160) → twenty_percent_of_certain_number certain_number = 80 :=
by
  sorry

end twenty_percent_correct_l426_426814


namespace radius_of_inscribed_sphere_l426_426748

-- Definition of the given problem parameter
def height := 4
def side_length := 6

-- Midpoints M and K should be defined as points but since their exact nature isn't required for checking the specific radius, we reference geometrically
def M_midpoint := true  -- To specify that M is the midpoint of BC
def K_midpoint := true  -- To specify that K is the midpoint of CD

-- The main theorem statement
theorem radius_of_inscribed_sphere (PO : ℝ) (s : ℝ) (isMidpointM : Prop) (isMidpointK : Prop) :
  PO = height →
  s = side_length →
  isMidpointM → 
  isMidpointK →
  radius_of_inscribed_sphere_in_pyramid PMKC = (12 / (13 + Real.sqrt 41)) :=
by
  sorry

end radius_of_inscribed_sphere_l426_426748


namespace solution_set_of_fx_eq_zero_l426_426196

noncomputable def f (x : ℝ) : ℝ :=
if hx : x = 0 then 0 else if 0 < x then Real.log x / Real.log 2 else - (Real.log (-x) / Real.log 2)

lemma f_is_odd : ∀ x : ℝ, f (-x) = - f x :=
by sorry

lemma f_is_log_for_positive : ∀ x : ℝ, 0 < x → f x = Real.log x / Real.log 2 :=
by sorry

theorem solution_set_of_fx_eq_zero :
  {x : ℝ | f x = 0} = {-1, 0, 1} :=
by sorry

end solution_set_of_fx_eq_zero_l426_426196


namespace candy_bars_weeks_l426_426305

theorem candy_bars_weeks (buy_per_week : ℕ) (eat_per_4_weeks : ℕ) (saved_candies : ℕ) (weeks_passed : ℕ) :
  (buy_per_week = 2) →
  (eat_per_4_weeks = 1) →
  (saved_candies = 28) →
  (weeks_passed = 4 * (saved_candies / (4 * buy_per_week - eat_per_4_weeks))) →
  weeks_passed = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end candy_bars_weeks_l426_426305


namespace total_games_l426_426671

-- Define the conditions
def games_this_year : ℕ := 4
def games_last_year : ℕ := 9

-- Define the proposition that we want to prove
theorem total_games : games_this_year + games_last_year = 13 := by
  sorry

end total_games_l426_426671


namespace same_color_point_exists_l426_426063

theorem same_color_point_exists (painted : ℝ × ℝ → Prop)
  (h : ∀ x y : ℝ × ℝ, painted x ∨ ¬ painted x ∧ (0 ≤ x.1 ≤ 2) ∧ (0 ≤ y.2 ≤ 2)) :
  ∃ p1 p2 : ℝ × ℝ, (painted p1 = painted p2) ∧ (dist p1 p2 = 1) :=
by 
  sorry

end same_color_point_exists_l426_426063


namespace function_characterization_l426_426159

-- State the main theorem
theorem function_characterization 
  (f : ℝ+ → ℝ+) 
  (h1 : ∀ (x y : ℝ+), f (x * f y) = y * f x) 
  (h2 : filter.tendsto f filter.at_top (nhds 0)) : 
  ∀ x : ℝ+, f x = 1 / x := 
sorry

end function_characterization_l426_426159


namespace prob_B_second_shot_prob_A_ith_shot_expected_A_shots_l426_426340

-- Define probabilities and parameters
def pA : ℝ := 0.6
def pB : ℝ := 0.8

-- Define the probability of selecting the first shooter as 0.5 for each player
def first_shot_prob : ℝ := 0.5

-- Proof that the probability that player B takes the second shot is 0.6
theorem prob_B_second_shot : (first_shot_prob * (1 - pA) + first_shot_prob * pB) = 0.6 := 
by sorry

-- Define the recursive probability for player A taking the nth shot
noncomputable def P (n : ℕ) : ℝ :=
if n = 0 then 0.5
else 0.4 * (P (n - 1)) + 0.2

-- Proof that the probability that player A takes the i-th shot is given by the formula
theorem prob_A_ith_shot (i : ℕ) : P i = (1 / 3) + (1 / 6) * ((2 / 5) ^ (i - 1)) :=
by sorry

-- Define the expected number of times player A shoots in the first n shots based on provided P formula
noncomputable def E_Y (n : ℕ) : ℝ :=
(sum i in finset.range n, P i)

-- Proof that the expected number of times player A shoots in the first n shots is given by the formula
theorem expected_A_shots (n : ℕ) : E_Y n = (5 / 18) * (1 - (2 / 5) ^ n) + (n / 3) :=
by sorry

end prob_B_second_shot_prob_A_ith_shot_expected_A_shots_l426_426340


namespace cannot_return_to_start_l426_426047

-- Define the type for points in Cartesian plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of the initial point
def P : Point := { x := 1, y := Real.sqrt 2 }

-- Definition of the allowed moves given a point
namespace Point
  def move1 (p : Point) : Point := { p with y := p.y + 2 * p.x }
  def move2 (p : Point) : Point := { p with y := p.y - 2 * p.x }
  def move3 (p : Point) : Point := { p with x := p.x + 2 * p.y }
  def move4 (p : Point) : Point := { p with x := p.x - 2 * p.y }
end Point

-- Hypothesis defining the condition that no two steps should lead back to the same point
axiom no_reversal (p1 p2 : Point) : p1 ≠ p2 → 
  (p2 ≠ Point.move1 p1 ∧ p2 ≠ Point.move2 p1 ∧ p2 ≠ Point.move3 p1 ∧ p2 ≠ Point.move4 p1)

theorem cannot_return_to_start (p : Point) (n : ℕ) (steps : Fin n → Point)
  (h0 : steps 0 = P) (hn : steps (Fin.last n) = P) : False :=
sorry

end cannot_return_to_start_l426_426047


namespace concentric_circles_tangent_area_eq_square_area_l426_426794

theorem concentric_circles_tangent_area_eq_square_area
    (R r : ℝ) (h : (R / r) = sqrt 2):
    (ring_area_tangents_eq (R r) = inscribed_square_area r) :=
by sorry

def ring_area_tangents_eq (R r : ℝ) : ℝ := (R^2)
def inscribed_square_area (r : ℝ) : ℝ := 2 * r^2

end concentric_circles_tangent_area_eq_square_area_l426_426794


namespace bob_calories_consumed_l426_426882

/-- Bob eats half of the pizza with 8 slices, each slice being 300 calories.
   Prove that Bob eats 1200 calories. -/
theorem bob_calories_consumed (total_slices : ℕ) (calories_per_slice : ℕ) (half_slices : ℕ) (calories_consumed : ℕ) 
  (h1 : total_slices = 8)
  (h2 : calories_per_slice = 300)
  (h3 : half_slices = total_slices / 2)
  (h4 : calories_consumed = half_slices * calories_per_slice) 
  : calories_consumed = 1200 := 
sorry

end bob_calories_consumed_l426_426882


namespace omega_value_l426_426217

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem omega_value (ω x₁ x₂ : ℝ) (h_ω : ω > 0) (h_x1 : f ω x₁ = -2) (h_x2 : f ω x₂ = 0) (h_min : |x₁ - x₂| = Real.pi) :
  ω = 1 / 2 := 
by 
  sorry

end omega_value_l426_426217


namespace ratio_white_to_remaining_l426_426726

def total_beans : ℕ := 572

def red_beans (total : ℕ) : ℕ := total / 4

def remaining_beans_after_red (total : ℕ) (red : ℕ) : ℕ := total - red

def green_beans : ℕ := 143

def remaining_beans_after_green (remaining : ℕ) (green : ℕ) : ℕ := remaining - green

def white_beans (remaining : ℕ) : ℕ := remaining / 2

theorem ratio_white_to_remaining (total : ℕ) (red : ℕ) (remaining : ℕ) (green : ℕ) (white : ℕ) 
  (H_total : total = 572)
  (H_red : red = red_beans total)
  (H_remaining : remaining = remaining_beans_after_red total red)
  (H_green : green = 143)
  (H_remaining_after_green : remaining_beans_after_green remaining green = white)
  (H_white : white = white_beans remaining) :
  (white : ℚ) / (remaining : ℚ) = (1 : ℚ) / 2 := 
by sorry

end ratio_white_to_remaining_l426_426726


namespace triangle_cut_20_sided_polygon_l426_426514

-- Definitions based on the conditions
def is_triangle (T : Type) : Prop := ∃ (a b c : ℝ), a + b + c = 180 

def can_form_20_sided_polygon (pieces : List (ℝ × ℝ)) : Prop := pieces.length = 20

-- Theorem statement
theorem triangle_cut_20_sided_polygon (T : Type) (P1 P2 : (ℝ × ℝ)) :
  is_triangle T → 
  (P1 ≠ P2) → 
  can_form_20_sided_polygon [P1, P2] :=
sorry

end triangle_cut_20_sided_polygon_l426_426514


namespace inequality_proof_l426_426691

theorem inequality_proof
  {n : ℕ} (a : Fin (2 * n) → ℝ)
  (h_nonneg : ∀ i, 0 ≤ a i)
  (h_noninc : ∀ i j, i < j → a i ≥ a j)
  (h_sum : (∑ i, a i) = 1) :
  (∑ k in Finset.range n, (2 * k + 1) * a (2 * k) * a (2 * k + 1)) ≤ 1 / 4 :=
begin
  sorry
end

end inequality_proof_l426_426691


namespace men_seated_l426_426455

theorem men_seated (total_passengers : ℕ) (women_ratio : ℚ) (children_count : ℕ) (men_standing_ratio : ℚ) 
  (women_with_prams : ℕ) (disabled_passengers : ℕ) 
  (h_total_passengers : total_passengers = 48) 
  (h_women_ratio : women_ratio = 2 / 3) 
  (h_children_count : children_count = 5) 
  (h_men_standing_ratio : men_standing_ratio = 1 / 8) 
  (h_women_with_prams : women_with_prams = 3) 
  (h_disabled_passengers : disabled_passengers = 2) : 
  (total_passengers * (1 - women_ratio) - total_passengers * (1 - women_ratio) * men_standing_ratio = 14) :=
by sorry

end men_seated_l426_426455


namespace f_le_2x_not_f_le_1_9x_l426_426562

variable {f : ℝ → ℝ}

-- Given conditions
def isNonNegativeOnSegment (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f(x)

def isEqualAtOne (f : ℝ → ℝ) := f(1) = 1

def satisfiesFunctionalInequality (f : ℝ → ℝ) :=
  ∀ x1 x2, 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 + x2 ≤ 1 → f(x1 + x2) ≥ f(x1) + f(x2)

-- First theorem
theorem f_le_2x {f : ℝ → ℝ}
  (h1 : isNonNegativeOnSegment f)
  (h2 : isEqualAtOne f)
  (h3 : satisfiesFunctionalInequality f) :
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x) ≤ 2 * x := 
sorry

-- Second theorem
theorem not_f_le_1_9x {f : ℝ → ℝ}
  (h1 : isNonNegativeOnSegment f)
  (h2 : isEqualAtOne f)
  (h3 : satisfiesFunctionalInequality f) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f(x) > 1.9 * x :=
sorry

end f_le_2x_not_f_le_1_9x_l426_426562


namespace scientific_notation_570_million_l426_426730

theorem scientific_notation_570_million:
  (570 * 10^6 : ℝ) = (5.7 * 10^8 : ℝ) :=
sorry

end scientific_notation_570_million_l426_426730


namespace female_managers_count_l426_426119

-- Definitions based on conditions
def total_employees : Nat := 250
def female_employees : Nat := 90
def total_managers : Nat := 40
def male_associates : Nat := 160

-- Statement to prove
theorem female_managers_count : (total_managers = 40) :=
by
  sorry

end female_managers_count_l426_426119


namespace bob_calories_consumed_l426_426881

/-- Bob eats half of the pizza with 8 slices, each slice being 300 calories.
   Prove that Bob eats 1200 calories. -/
theorem bob_calories_consumed (total_slices : ℕ) (calories_per_slice : ℕ) (half_slices : ℕ) (calories_consumed : ℕ) 
  (h1 : total_slices = 8)
  (h2 : calories_per_slice = 300)
  (h3 : half_slices = total_slices / 2)
  (h4 : calories_consumed = half_slices * calories_per_slice) 
  : calories_consumed = 1200 := 
sorry

end bob_calories_consumed_l426_426881


namespace problem_statement_l426_426982

open Complex

theorem problem_statement :
  (3 - I) / (2 + I) = 1 - I :=
by
  sorry

end problem_statement_l426_426982


namespace range_of_tangent_points_l426_426606

noncomputable def f (x t : ℝ) : ℝ := x^2 - 2 * t * x - 4 * t - 4
noncomputable def g (x t : ℝ) : ℝ := 1 / x - (t + 2)^2
def h (x t : ℝ) : ℝ := 8 * x^3 - 4 * t * x^2 + 1

theorem range_of_tangent_points (t : ℝ) :
  (8 * x^3 - 4 * t * x^2 + 1 = 0).roots.count ℝ = 3 →
  t > (3 * (2^(1/3))) / 2 :=
sorry

end range_of_tangent_points_l426_426606


namespace base_conversion_subtraction_l426_426493

theorem base_conversion_subtraction :
  let n1_base9 := 3 * 9^2 + 2 * 9^1 + 4 * 9^0
  let n2_base7 := 1 * 7^2 + 6 * 7^1 + 5 * 7^0
  n1_base9 - n2_base7 = 169 :=
by
  sorry

end base_conversion_subtraction_l426_426493


namespace right_triangle_square_inscribed_l426_426116

theorem right_triangle_square_inscribed (A B C S P Q R : Point) (h₁ : right_triangle A B C)
  (h₂ : inscribed_square PQRS A B C) (h₃ : S ∈ segment B C) (h₄ : P ∈ segment C A)
  (h₅ : Q ∈ segment A B) (h₆ : R ∈ segment A B) : AB ≥ 3 * QR ∧ 
  (AB = 3 * QR → AQ = QR) :=
sorry

end right_triangle_square_inscribed_l426_426116


namespace coefficient_x2_y4_in_expansion_l426_426164

theorem coefficient_x2_y4_in_expansion :
  (∀ x y : ℤ, 
    ∃ (c : ℤ), 
      c = ∑ r in finset.range 7, 
        if r = 4 then binomial 6 r * (-2:ℤ)^r else 0 ∧ 
      x^(6-r) * y^r = x^2 * y^4 → 
      c = 3360 ) :=
by
  sorry

end coefficient_x2_y4_in_expansion_l426_426164


namespace Pascal_triangle_sum_l426_426352

def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def Pascal_sequence (n : ℕ) (i : ℕ) : ℕ :=
  binomial n i

def sum_b_over_c_minus_sum_a_over_b : ℚ :=
  (∑ i in Finset.range 11, (Pascal_sequence 11 i : ℚ) / (Pascal_sequence 12 i : ℚ)) -
  (∑ i in Finset.range 10, (Pascal_sequence 10 i : ℚ) / (Pascal_sequence 11 i : ℚ))

theorem Pascal_triangle_sum :
  sum_b_over_c_minus_sum_a_over_b = 67 / 132 :=
by
  sorry

end Pascal_triangle_sum_l426_426352


namespace union_eq_l426_426577

open Set

theorem union_eq (A B : Set ℝ) (hA : A = {x | -1 < x ∧ x < 1}) (hB : B = {x | 0 ≤ x ∧ x ≤ 2}) :
    A ∪ B = {x | -1 < x ∧ x ≤ 2} :=
by
  rw [hA, hB]
  ext x
  simp
  sorry

end union_eq_l426_426577


namespace no_such_polynomial_exists_l426_426147

theorem no_such_polynomial_exists : ¬(∃ (P : ℕ → ℤ), (∀ n : ℕ, P n = (∑ i in finset.range (n^2 + 1), ⌊(i : ℝ)^(1/3)⌋))) :=
sorry

end no_such_polynomial_exists_l426_426147


namespace smallest_phi_for_even_function_l426_426000

theorem smallest_phi_for_even_function (φ : ℝ) (hφ : φ > 0) :
  shifted_even_function φ → φ = π / 4 :=
by
-- Assuming the shifted_even_function is our condition that the function is even after the shift
sorry

/-
  Definitions and supporting functions
-/

def shifted_even_function (φ : ℝ) : Prop :=
  ∀ x, sin (2 * (x + π / 8) + φ) = sin (-(2 * (x + π / 8) + φ))

#eval by { rw sin_add, simp, ring }

end smallest_phi_for_even_function_l426_426000


namespace domain_j_l426_426418

noncomputable def j (x : ℝ) : ℝ := (1 / (x + 8)) + (1 / (x^2 + 8)) + (1 / (x^3 + 8))

theorem domain_j :
  {x : ℝ | x ≠ -8 ∧ x ≠ -2} = 
  {x : ℝ | x ∈ Ioo -∞ -8} ∪ {x : ℝ | x ∈ Ioo -8 -2} ∪ {x : ℝ | x ∈ Ioi -2} :=
begin
  sorry
end

end domain_j_l426_426418


namespace complete_sets_characterization_l426_426326

-- Definition of a complete set
def complete_set (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, (a + b ∈ A) → (a * b ∈ A)

-- Theorem stating that the complete sets of natural numbers are exactly
-- {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, ℕ.
theorem complete_sets_characterization :
  ∀ (A : Set ℕ), complete_set A ↔ (A = {1} ∨ A = {1, 2} ∨ A = {1, 2, 3} ∨ A = {1, 2, 3, 4} ∨ A = Set.univ) :=
sorry

end complete_sets_characterization_l426_426326


namespace quadratic_has_two_distinct_real_roots_l426_426768

theorem quadratic_has_two_distinct_real_roots :
  ∀ x : ℝ, ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (x^2 - 2 * x - 6 = 0 ∧ x = r1 ∨ x = r2) :=
by sorry

end quadratic_has_two_distinct_real_roots_l426_426768


namespace shirts_per_minute_l426_426113

theorem shirts_per_minute (S : ℕ) 
  (h1 : 12 * S + 14 = 156) : S = 11 := 
by
  sorry

end shirts_per_minute_l426_426113


namespace difference_in_profit_l426_426704

theorem difference_in_profit (n : ℕ)
  (h : n = 200)
  (sammy_offer_price_per_record : ℕ)
  (sammy_offer_price : sammy_offer_price_per_record = 4)
  (bryan_offer_price_first_half : ℕ)
  (bryan_offer_price_first_half = 6)
  (bryan_offer_price_second_half : ℕ)
  (bryan_offer_price_second_half = 1) :
  let sammy_total := n * sammy_offer_price_per_record,
      bryan_total := (n / 2) * bryan_offer_price_first_half + (n / 2) * bryan_offer_price_second_half in
  sammy_total - bryan_total = 100 :=
by sorry

end difference_in_profit_l426_426704


namespace polynomial_expansion_l426_426712

theorem polynomial_expansion (k : ℕ) (x : ℝ) :
  ((List.range (k + 1)).foldl (λ acc i, acc * (1 + x^(2^i))) 1) = 
  (List.range (2^(k+1))).foldr (λ i acc, x^i + acc) 0 :=
sorry

end polynomial_expansion_l426_426712


namespace even_numbers_count_greater_than_31000_count_not_adjacent_2_4_count_l426_426042

-- Part 1: Count of even five-digit numbers formed using {0, 1, 2, 3, 4} without repeating digits.
theorem even_numbers_count : ∃ n, n = 60 ∧ ∀ x : Finset (Fin 5), x.val = {0, 1, 2, 3, 4}.to_finset.val → x.card = 5 → digits.last x ∈ {0, 2, 4} → count (digits x) n := 
sorry

-- Part 2: Count of five-digit numbers greater than 31000 formed using {0, 1, 2, 3, 4} without repeating digits.
theorem greater_than_31000_count : ∃ n, n = 42 ∧ ∀ x : Finset (Fin 5), x.val = {0, 1, 2, 3, 4}.to_finset.val → x.card = 5 → digits.first x ≥ 3 ∧ (digits.first x = 3 → digits.second x ∈ {2, 4, 1}) → count (digits x) n := 
sorry

-- Part 3: Count of five-digit numbers where digits 2 and 4 are not adjacent formed using {0, 1, 2, 3, 4} without repeating digits.
theorem not_adjacent_2_4_count : ∃ n, n = 60 ∧ ∀ x : Finset (Fin 5), x.val = {0, 1, 2, 3, 4}.to_finset.val → x.card = 5 → ¬ adjacent x 2 4 → count (digits x) n := 
sorry


end even_numbers_count_greater_than_31000_count_not_adjacent_2_4_count_l426_426042


namespace felicity_gas_usage_l426_426528

variable (A F : ℕ)

theorem felicity_gas_usage
  (h1 : F = 4 * A - 5)
  (h2 : A + F = 30) :
  F = 23 := by
  sorry

end felicity_gas_usage_l426_426528


namespace angle_between_vectors_l426_426612

open Real

noncomputable def a : ℝ × ℝ := (1, sqrt 3)
noncomputable def b : ℝ × ℝ := (sqrt 3, 1)

theorem angle_between_vectors :
  let θ := real.arccos ((a.1 * b.1 + a.2 * b.2) / (sqrt (a.1 ^ 2 + a.2 ^ 2) * sqrt (b.1 ^ 2 + b.2 ^ 2))) in
  θ = π / 6 := 
sorry

end angle_between_vectors_l426_426612


namespace explicit_form_of_function_l426_426222

theorem explicit_form_of_function (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * f x + f x * f y + y - 1) = f (x * f x + x * y) + y - 1) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end explicit_form_of_function_l426_426222


namespace angle_equality_l426_426316

open EuclideanGeometry

-- Define the problem setup
variables {Γ : Circle} {O : Point} {P Q C A B : Point}
  (h_chord : IsChordOf A B Γ)
  (h_center : CenterOf O (Circle.normalizeTangentToLine O C))
  (h_tangent_AB : TangentAtLine (Circle.normalizeTangentToLine O C) AB C)
  (h_center_Γ : CenterOf O Γ)
  (h_tangent_Γ : InternalTangentCircles O Γ P)
  (h_tangent_point_C : Between C A B)
  (h_circ_POC : ∃ circ_POC : Circle, CircleOfTriangle circ_POC P O C)
  (h_intersect_Q : ∃ Q ≠ P, IntersectCirclesAtTwoPoints (CircleOfTriangle h_circ_POC) Γ Q P)

-- Define the proposition to be proved
theorem angle_equality (h_chord : IsChordOf A B Γ)
  (h_center : CenterOf O (Circle.normalizeTangentToLine O C))
  (h_tangent_AB : TangentAtLine (Circle.normalizeTangentToLine O C) AB C)
  (h_center_Γ : CenterOf O Γ)
  (h_tangent_Γ : InternalTangentCircles O Γ P)
  (h_tangent_point_C : Between C A B)
  (h_circ_POC : ∃ circ_POC : Circle, CircleOfTriangle circ_POC P O C)
  (h_intersect_Q : ∃ Q ≠ P, IntersectCirclesAtTwoPoints (CircleOfTriangle h_circ_POC) Γ Q P) :
  ∠ A Q P = ∠ C Q B :=
sorry

end angle_equality_l426_426316


namespace scientific_notation_of_570_million_l426_426727

theorem scientific_notation_of_570_million :
  570000000 = 5.7 * 10^8 := sorry

end scientific_notation_of_570_million_l426_426727


namespace tims_change_l426_426034

theorem tims_change (initial_amount candy_bar soda gum : ℝ) (h₁ : initial_amount = 3.75) (h₂ : candy_bar = 1.45) (h₃ : soda = 1.25) (h₄ : gum = 0.75) :
  initial_amount - (candy_bar + soda + gum) = 0.30 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num
  sorry

end tims_change_l426_426034


namespace total_carrots_l426_426718

def sally_carrots : ℕ := 6
def fred_carrots : ℕ := 4
def mary_carrots : ℕ := 10

theorem total_carrots : sally_carrots + fred_carrots + mary_carrots = 20 := by
  sorry

end total_carrots_l426_426718


namespace derivative_y_l426_426939

noncomputable def y (x : ℝ) : ℝ := 
  (Real.sqrt (49 * x^2 + 1) * Real.arctan (7 * x)) - 
  Real.log (7 * x + Real.sqrt (49 * x^2 + 1))

theorem derivative_y (x : ℝ) : 
  deriv y x = (7 * Real.arctan (7 * x)) / (2 * Real.sqrt (49 * x^2 + 1)) := by
  sorry

end derivative_y_l426_426939


namespace find_lambda_perpendicular_to_a_l426_426234

def vec (α : Type*) [Add α] := (α × α)

theorem find_lambda_perpendicular_to_a : 
  ∃ λ : ℝ, let a : vec ℝ := (3, -2)
           let b : vec ℝ := (1, 2)
           let c : vec ℝ := (a.1 + λ * b.1, a.2 + λ * b.2)
           a.1 * c.1 + a.2 * c.2 = 0 :=
begin
  use 13,
  let a : vec ℝ := (3, -2),
  let b : vec ℝ := (1, 2),
  let λ : ℝ := 13,
  let c : vec ℝ := (a.1 + λ * b.1, a.2 + λ * b.2),
  exact calc
    (a.1 * c.1 + a.2 * c.2) = (3 * (3 + 13 * 1) + (-2) * (13 * 2 - 2)) : by refl
                         ... = (3 * 16 + (-2) * 24) : by refl
                         ... = 48 - 48 : by ring
                         ... = 0 : by ring
end

end find_lambda_perpendicular_to_a_l426_426234


namespace sin_cos_inequality_l426_426310

theorem sin_cos_inequality (α : ℝ) 
  (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (h3 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  (Real.pi / 3 < α ∧ α < 4 * Real.pi / 3) :=
sorry

end sin_cos_inequality_l426_426310


namespace numbers_are_odd_l426_426775

theorem numbers_are_odd (n : ℕ) (sum : ℕ) (h1 : n = 49) (h2 : sum = 2401) : 
      (∀ i < n, ∃ j, sum = j * 2 * i + 1) :=
by
  sorry

end numbers_are_odd_l426_426775


namespace statement_A_statement_C_statement_D_l426_426636

variable (a : ℕ → ℝ) (A B : ℝ)

-- Condition: The sequence satisfies the recurrence relation
def recurrence_relation (n : ℕ) : Prop :=
  a (n + 2) = A * a (n + 1) + B * a n

-- Statement A: A=1 and B=-1 imply periodic with period 6
theorem statement_A (h : ∀ n, recurrence_relation a 1 (-1) n) :
  ∀ n, a (n + 6) = a n := 
sorry

-- Statement C: A=3 and B=-2 imply the derived sequence is geometric
theorem statement_C (h : ∀ n, recurrence_relation a 3 (-2) n) :
  ∃ r : ℝ, ∀ n, a (n + 1) - a n = r * (a n - a (n - 1)) :=
sorry

-- Statement D: A+1=B, a1=0, a2=B imply {a_{2n}} is increasing
theorem statement_D (hA : ∀ n, recurrence_relation a A (A + 1) n)
  (h1 : a 1 = 0) (h2 : a 2 = A + 1) :
  ∀ n, a (2 * (n + 1)) > a (2 * n) :=
sorry

end statement_A_statement_C_statement_D_l426_426636


namespace function_at_neg_one_zero_l426_426590

-- Define the function f with the given conditions
variable {f : ℝ → ℝ}

-- Declare the conditions as hypotheses
def domain_condition : ∀ x : ℝ, true := by sorry
def non_zero_condition : ∃ x : ℝ, f x ≠ 0 := by sorry
def even_function_condition : ∀ x : ℝ, f (x + 2) = f (2 - x) := by sorry
def odd_function_condition : ∀ x : ℝ, f (1 - 2 * x) = -f (2 * x + 1) := by sorry

-- The main theorem to be proved
theorem function_at_neg_one_zero :
  f (-1) = 0 :=
by
  -- Use the conditions to derive the result
  sorry

end function_at_neg_one_zero_l426_426590


namespace f_positive_sequence_decreasing_sequence_lower_bound_final_inequality_l426_426991

-- 1. Prove that f(x) > 0 for all x > 0
theorem f_positive (x : ℝ) (h : x > 0) : (x - 1) * Real.exp x + 1 > 0 := sorry

-- 2. Prove that x_n > x_{n+1} > \frac{1}{2^{n+1}} for all n in ℕ*, where x₁ = 1 and x_n is defined recursively
def sequence_x : ℕ+ → ℝ
| 1 => 1
| (k + 1) => classical.some (nonempty_of_exists (λ x => ((sequence_x k) * Real.exp x = Real.exp (sequence_x k) - 1)))

theorem sequence_decreasing (n : ℕ+) : sequence_x n > sequence_x (n + 1) := sorry
theorem sequence_lower_bound (n : ℕ+) : sequence_x n > 1 / 2^(n:ℕ) := sorry
theorem final_inequality (n : ℕ+) : sequence_x n > sequence_x (n + 1) ∧ sequence_x (n + 1) > 1 / 2^(n+1:ℕ) :=
  ⟨sequence_decreasing n, sequence_lower_bound (n+1)⟩

end f_positive_sequence_decreasing_sequence_lower_bound_final_inequality_l426_426991


namespace coeff_x5_in_q_cubed_l426_426255

/-- Define the polynomial q(x) -/
def q (x : ℝ) : ℝ := x^4 - 4 * x^2 + 5 * x + 1

/-- Prove that the coefficient of the x^5 term in (q(x))^3 is 95 -/
theorem coeff_x5_in_q_cubed (x : ℝ) : coeff (x^5) (q x)^3 = 95 :=
by
  sorry

end coeff_x5_in_q_cubed_l426_426255


namespace center_and_radius_sum_l426_426679

noncomputable def circle_center_and_radius : (ℝ × ℝ) :=
  let a := 3
  let b := 7
  (a, b)

noncomputable def radius : ℝ := Real.sqrt 15

noncomputable def value_of_a_b_r : ℝ :=
  let (a, b) := circle_center_and_radius in
  a + b + radius

theorem center_and_radius_sum :
  ∀ (a b : ℝ) (r : ℝ),
    (x^2 - 14 * y + 73 = -y^2 + 6 * x) →
    (a, b) = (3, 7) →
    r = Real.sqrt 15 →
    a + b + r = 10 + Real.sqrt 15 :=
by
  intros a b r _ _ _
  sorry

end center_and_radius_sum_l426_426679


namespace find_a_l426_426949

theorem find_a (a : ℝ) (h : (a - 1) ≠ 0) :
  (∃ x : ℝ, ((a - 1) * x^2 + x + a^2 - 1 = 0) ∧ x = 0) → a = -1 :=
by
  sorry

end find_a_l426_426949


namespace question_l426_426241

theorem question (A : ℕ → ℕ) (h : ∀ n : ℕ, A 2n ^ 3 = 9 * A n ^ 3) : 14 :=
by
  sorry

end question_l426_426241


namespace find_a_degree_l426_426633

-- Definitions from conditions
def monomial_degree (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

-- Statement of the proof problem
theorem find_a_degree (a : ℕ) (h : monomial_degree 2 a = 6) : a = 4 :=
by
  sorry

end find_a_degree_l426_426633


namespace factorize_x4_minus_81_l426_426934

theorem factorize_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factorize_x4_minus_81_l426_426934


namespace fifth_term_arithmetic_sequence_l426_426746

def arithmetic_sequence_fifth_term
  (x y : ℝ) 
  (h_sequence : [x + 2 * y, x - 2 * y, x ^ 2 - y ^ 2, x / y])
  (h_arithmetic : ∃ d : ℝ, ∀ n : ℕ, h_sequence[n+1] - h_sequence[n] = d) : Prop :=
  h_sequence[4] = sqrt 7 - 1 - 4 * y

theorem fifth_term_arithmetic_sequence
  (x y : ℝ)
  (y_ne_zero : y ≠ 0)
  (h_sequence : [x + 2 * y, x - 2 * y, x ^ 2 - y ^ 2, x / y])
  (h_arithmetic : ∃ d : ℝ, ∀ n : ℕ, h_sequence[n+1] - h_sequence[n] = d) : 
  arithmetic_sequence_fifth_term x y h_sequence h_arithmetic :=
sorry

end fifth_term_arithmetic_sequence_l426_426746


namespace range_of_f_l426_426994

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_f 
  (x : ℝ) : f (x - 1) + f (x + 1) > 0 ↔ x ∈ Set.Ioi 0 :=
by
  sorry

end range_of_f_l426_426994


namespace fraction_equal_l426_426249

variable {m n p q : ℚ}

-- Define the conditions
def condition1 := (m / n = 20)
def condition2 := (p / n = 5)
def condition3 := (p / q = 1 / 15)

-- State the theorem
theorem fraction_equal (h1 : condition1) (h2 : condition2) (h3 : condition3) : (m / q = 4 / 15) :=
  sorry

end fraction_equal_l426_426249


namespace area_on_map_correct_l426_426854

namespace FieldMap

-- Given conditions
def actual_length_m : ℕ := 200
def actual_width_m : ℕ := 100
def scale_factor : ℕ := 2000

-- Conversion from meters to centimeters
def length_cm := actual_length_m * 100
def width_cm := actual_width_m * 100

-- Dimensions on the map
def length_map_cm := length_cm / scale_factor
def width_map_cm := width_cm / scale_factor

-- Area on the map
def area_map_cm2 := length_map_cm * width_map_cm

-- Statement to prove
theorem area_on_map_correct : area_map_cm2 = 50 := by
  sorry

end FieldMap

end area_on_map_correct_l426_426854


namespace solution_of_system_l426_426893

noncomputable def system_of_equations := ∀ (x y : ℝ),
  (4 * x + 3 * y) / 3 + (6 * x - y) / 8 = 8 ∧
  (4 * x + 3 * y) / 6 + (6 * x - y) / 2 = 11

theorem solution_of_system : ∃ (x y : ℝ), system_of_equations x y ∧ x = 3 ∧ y = 2 := 
by
  exists 3, 2
  unfold system_of_equations
  simp
  split
  sorry
  sorry

end solution_of_system_l426_426893


namespace low_purine_food_days_l426_426040

noncomputable def K : ℝ := sorry -- solve for K from the given conditions

theorem low_purine_food_days 
  (U0 : ℝ) (U : ℝ → ℝ) (t t' : ℝ)
  (ln : ℝ → ℝ) (exp : ℝ → ℝ) 
  (h1 : U0 = 20)
  (h2 : t = 50)
  (h3 : U t = 15) :
  let K := exp ((15 / U0) - ln t) / t in
  (U t' ≤ 7) → 
  t' = 75 :=
by
  sorry

#check low_purine_food_days

end low_purine_food_days_l426_426040


namespace midpoints_collinear_l426_426383

variable (A B C A₁ B₁ C₁ A₂ B₂ C₂ : Point)
variable (m : Line)

-- Conditions
hypothesis h1 : m ∩ (line_through B C) = A₁
hypothesis h2 : m ∩ (line_through C A) = B₁
hypothesis h3 : m ∩ (line_through A B) = C₁

hypothesis h4 : harmonic_conjugates A₁ A₂ B C
hypothesis h5 : harmonic_conjugates B₁ B₂ C A
hypothesis h6 : harmonic_conjugates C₁ C₂ A B

-- Definition of midpoint
def midpoint (P Q : Point) : Point :=
  Point.mk ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

-- Question: Prove midpoints are collinear
theorem midpoints_collinear 
  (h1 : m ∩ (line_through B C) = A₁)
  (h2 : m ∩ (line_through C A) = B₁)
  (h3 : m ∩ (line_through A B) = C₁)
  (h4 : harmonic_conjugates A₁ A₂ B C)
  (h5 : harmonic_conjugates B₁ B₂ C A)
  (h6 : harmonic_conjugates C₁ C₂ A B) :
  collinear (midpoint A₁ A₂) (midpoint B₁ B₂) (midpoint C₁ C₂) :=
begin
  sorry
end

end midpoints_collinear_l426_426383


namespace John_eats_for_ten_days_l426_426672

def total_burritos (boxes : ℕ) (burritos_per_box : ℕ) : ℕ :=
  boxes * burritos_per_box

def burritos_given_away (total : ℕ) (fraction : ℚ) : ℕ :=
  (total : ℕ) * (fraction : ℚ).toDenominator / (fraction : ℚ).toNumerator

def burritos_left_to_eat (total : ℕ) (given_away : ℕ) : ℕ :=
  total - given_away

def burritos_eaten_total (total_left : ℕ) (left : ℕ) : ℕ :=
  total_left - left

def number_of_days (eaten : ℕ) (per_day : ℕ) : ℕ :=
  eaten / per_day

theorem John_eats_for_ten_days
  (boxes : ℕ)
  (burritos_per_box : ℕ)
  (fraction_given_away : ℚ)
  (eats_per_day : ℕ)
  (left : ℕ)
  (h1 : boxes = 3)
  (h2 : burritos_per_box = 20)
  (h3 : fraction_given_away = 1 / 3)
  (h4 : eats_per_day = 3)
  (h5 : left = 10) :
  number_of_days
    (burritos_eaten_total
      (burritos_left_to_eat
        (total_burritos boxes burritos_per_box)
        (burritos_given_away (total_burritos boxes burritos_per_box) fraction_given_away)
      )
      left
    )
    eats_per_day = 10 := sorry

end John_eats_for_ten_days_l426_426672


namespace expected_participants_correct_l426_426285

noncomputable def expectedParticipants : ℕ :=
  let p0 := 1000
  let rate := 1.3
  let p1 := p0 * rate
  let p2 := p1 * rate
  let p3 := p2 * rate
  let p4 := p3 * rate
  p4.toNat

theorem expected_participants_correct : expectedParticipants = 2856 := by
  sorry

end expected_participants_correct_l426_426285


namespace value_of_expression_l426_426055

theorem value_of_expression : 
  ∀ (x y : ℤ), x = -5 → y = -10 → (y - x) * (y + x) = 75 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end value_of_expression_l426_426055


namespace proof_problem_l426_426944

def sign (a : ℝ) : ℤ :=
if a > 0 then 1 else if a < 0 then -1 else 0

def system_solutions (x y z : ℝ) : Prop :=
x = 2 * (2020 - 2021 * sign (y + z)) ∧
y = 2 * (2020 - 2021 * sign (x + z)) ∧
z = 2 * (2020 - 2021 * sign (x + y))

theorem proof_problem :
  {t : ℝ × ℝ × ℝ // system_solutions t.1 t.2 t.2}.to_finset.card = 3 :=
by
  sorry

end proof_problem_l426_426944


namespace hyperbola_eccentricity_sqrt5_l426_426207

noncomputable def eccentricity_of_hyperbola (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b/a)^2)

theorem hyperbola_eccentricity_sqrt5
  (a b : ℝ)
  (h : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (y = x^2 + 1) → (x, y) = (1, 2)) :
  eccentricity_of_hyperbola a b = Real.sqrt 5 :=
by sorry

end hyperbola_eccentricity_sqrt5_l426_426207


namespace final_prob_eq_l426_426318

noncomputable def prob_eq_solutions (p q : ℕ) (h : Nat.gcd p q = 1) := 
  let c := [x ∈ Icc (-10 : ℝ) 10] 
  let equation_has_solutions := λ c : ℝ, 9 * c^4 - 48 * c^3 ≥ 0
  let interval_satisfying := (Icc (-10 : ℝ) 0) ∪ (Icc (5.33 : ℝ) 10)
  let probability := (interval_satisfying.to_Set.vol / (Icc (-10 : ℝ) 10).to_Set.vol)
  probability = 197 / 200

-- Placeholder to indicate where the proof would go.
theorem final_prob_eq : ∃ (p q : ℕ), (Nat.gcd p q = 1 ∧ prob_eq_solutions 197 200 (by norm_num)) :=
  sorry

end final_prob_eq_l426_426318


namespace range_of_f_l426_426779

theorem range_of_f :
  ∀ x : ℝ, -√3 ≤ (sin x - cos (x + π/6)) ∧ (sin x - cos (x + π/6)) ≤ √3 :=
by
  intro x
  sorry

end range_of_f_l426_426779


namespace compare_exponents_l426_426977

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem compare_exponents : b < a ∧ a < c :=
by
  have h1 : a = 2^(4/3) := rfl
  have h2 : b = 4^(2/5) := rfl
  have h3 : c = 25^(1/3) := rfl
  -- These are used to indicate the definitions, not the proof steps
  sorry

end compare_exponents_l426_426977


namespace average_speed_of_train_b_l426_426792

-- Given conditions
def distance_between_trains_initially := 13
def speed_of_train_a := 37
def time_to_overtake := 5
def distance_a_in_5_hours := speed_of_train_a * time_to_overtake
def distance_b_to_overtake := distance_between_trains_initially + distance_a_in_5_hours + 17

-- Prove: The average speed of Train B
theorem average_speed_of_train_b : 
  ∃ v_B, v_B = distance_b_to_overtake / time_to_overtake ∧ v_B = 43 :=
by
  -- The proof should go here, but we use sorry to skip it.
  sorry

end average_speed_of_train_b_l426_426792


namespace bob_calories_l426_426875

-- conditions
def slices : ℕ := 8
def half_slices (slices : ℕ) : ℕ := slices / 2
def calories_per_slice : ℕ := 300
def total_calories (half_slices : ℕ) (calories_per_slice : ℕ) : ℕ := half_slices * calories_per_slice

-- proof problem
theorem bob_calories : total_calories (half_slices slices) calories_per_slice = 1200 := by
  sorry

end bob_calories_l426_426875


namespace general_term_formula_a_sum_first_n_terms_b_l426_426592

noncomputable def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 3 else 4 * 3^(n-1) - 1

noncomputable def sequence_S : ℕ → ℕ 
| 0        := 3
| (n + 1)  := 3 * (sequence_S n) + 2 * n + 3

noncomputable def sequence_b (n : ℕ) : ℕ :=
  n * (sequence_a n + 1)

noncomputable def sequence_T : ℕ → ℕ 
| 0        := 0
| (n + 1)  := sequence_b (n + 1) + sequence_T n

theorem general_term_formula_a (n : ℕ) :
  sequence_a n = 4 * 3^(n-1) - 1 :=
sorry

theorem sum_first_n_terms_b (n : ℕ) :
  sequence_T n = 1 + (2 * n - 1) * 3^n :=
sorry

end general_term_formula_a_sum_first_n_terms_b_l426_426592


namespace exterior_angle_BAC_l426_426100

-- Conditions definitions
def regular_polygon_interior_angle (n : ℕ) : ℝ :=
  180 * (1 - 2 / n)

def square_interior_angle : ℝ :=
  90

-- Theorem statement
theorem exterior_angle_BAC (shared_side : ℝ) :
  let pentagon_angle := regular_polygon_interior_angle 5
  let square_angle := square_interior_angle
  ∃ (BAC : ℝ), BAC = 360 - pentagon_angle - square_angle ∧ BAC = 162 :=
begin
  sorry
end

end exterior_angle_BAC_l426_426100


namespace find_solution_set_l426_426211

noncomputable def solution_set_of_quadratic (a b c : ℝ) := {x : ℝ | ax^2 + bx + c > 0}

theorem find_solution_set (a b c : ℝ) :
  solution_set_of_quadratic a b c = { x : ℝ | 1 < x ∧ x < 2 } →
  {x : ℝ | c * x^2 - b * x + a > 0} = { x : ℝ | -1 < x ∧ x < -1/2 } :=
by
  sorry

end find_solution_set_l426_426211


namespace prob1_prob2_l426_426970

theorem prob1 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, a n + a (n + 1) = 2^(n+1)) :
  ∃ c r : ℝ, (∀ n : ℕ, (a n) / 2^n - 2/3 = c * r^n) ∧ (∀ n : ℕ, a n = (2^(n+1) + (-1)^n) / 3) :=
sorry

theorem prob2 (T : ℕ → ℝ) (a : ℕ → ℝ) (h₀ : ∀ n : ℕ, a n = (2^(n+1) + (-1)^n) / 3) :
  ∀ n : ℕ, T n = ∑ i in finset.range (n+1), (1 / a (i+1)) ∧ T n < 7/4 :=
sorry

end prob1_prob2_l426_426970


namespace max_AB_value_l426_426214

-- Definitions of conditions
variable (a b : ℝ)
variable (C : ℝ × ℝ → Prop)
variable (F : ℝ × ℝ)
variable (m : ℝ)

def ellipse_equation := ∀ x y : ℝ, C (x, y) ↔ (x^2 / a^2 + y^2 / b^2 = 1)
def focus_condition := F = (-real.sqrt 3, 0)
def point_on_ellipse := C (1, real.sqrt 3 / 2)

-- Proving the formula for |AB|
noncomputable def AB_dist := ∀ m : ℝ, ∃ A B : ℝ × ℝ, 
  (l A ∧ l B ∧ C A ∧ C B) → (|A - B| = (4 * real.sqrt 3 * |m|) / (m^2 + 1))

-- Proving the maximum value of |AB| is 2
theorem max_AB_value : ∀ m : ℝ, (|m| = real.sqrt 3) → ((4 * real.sqrt 3 * |m|) / (m^2 + 1)) ≤ 2
:= sorry

end max_AB_value_l426_426214


namespace sum_of_squares_of_medians_l426_426052

/--
Given a triangle with side lengths 9, 12, and 15, the sum of the squares of the lengths of its medians is 443.25.
-/
theorem sum_of_squares_of_medians (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  let m_a := (2 * b^2 + 2 * c^2 - a^2) / 4,
      m_b := (2 * a^2 + 2 * c^2 - b^2) / 4,
      m_c := (2 * a^2 + 2 * b^2 - c^2) / 4 in
  m_a + m_b + m_c = 443.25 :=
by
  sorry

end sum_of_squares_of_medians_l426_426052


namespace solve_equation_l426_426724

theorem solve_equation : ∃ x : ℚ, (2*x + 1) / 4 - 1 = x - (10*x + 1) / 12 ∧ x = 5 / 2 :=
by
  sorry

end solve_equation_l426_426724


namespace commute_days_l426_426461

-- Definitions of the variables
variables (a b c x : ℕ)

-- Given conditions
def condition1 : Prop := a + c = 12
def condition2 : Prop := b + c = 20
def condition3 : Prop := a + b = 14

-- The theorem to prove
theorem commute_days (h1 : condition1 a c) (h2 : condition2 b c) (h3 : condition3 a b) : a + b + c = 23 :=
sorry

end commute_days_l426_426461


namespace bob_calories_l426_426877

-- conditions
def slices : ℕ := 8
def half_slices (slices : ℕ) : ℕ := slices / 2
def calories_per_slice : ℕ := 300
def total_calories (half_slices : ℕ) (calories_per_slice : ℕ) : ℕ := half_slices * calories_per_slice

-- proof problem
theorem bob_calories : total_calories (half_slices slices) calories_per_slice = 1200 := by
  sorry

end bob_calories_l426_426877


namespace part1_part2_l426_426225

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * abs (x - 1)
def h (a x : ℝ) : ℝ := abs (f x) + g a x

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 :=
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → h a x ≤ 
    if a ≥ 0 then 3 * a + 3
    else if -3 ≤ a ∧ a < 0 then a + 3
    else if a < -3 then 0
    else false) :=
sorry

end part1_part2_l426_426225


namespace sequence_satisfies_conditions_l426_426543

theorem sequence_satisfies_conditions (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n ≤ n * nat.sqrt n)
  (h2 : ∀ m n : ℕ, m ≠ n → (m - n) ∣ (a m - a n)) :
  (∀ n, a n = 1) ∨ (∀ n, a n = n) :=
sorry

end sequence_satisfies_conditions_l426_426543


namespace juniors_score_l426_426480

/-- Mathematical proof problem stated in Lean 4 -/
theorem juniors_score 
  (total_students : ℕ) 
  (juniors seniors : ℕ)
  (junior_score senior_avg total_avg : ℝ)
  (h_total_students : total_students > 0)
  (h_juniors : juniors = total_students / 10)
  (h_seniors : seniors = (total_students * 9) / 10)
  (h_total_avg : total_avg = 84)
  (h_senior_avg : senior_avg = 83)
  (h_junior_score_same : ∀ j : ℕ, j < juniors → ∃ s : ℝ, s = junior_score)
  :
  junior_score = 93 :=
by
  sorry

end juniors_score_l426_426480


namespace max_perimeter_of_rectangle_l426_426337

theorem max_perimeter_of_rectangle :
  let leg1 := 2
  let leg2 := 3
  let n := 60
  let area_of_one_triangle := (1 / 2) * leg1 * leg2
  let total_area := n * area_of_one_triangle
  ∃ L W, L * W = total_area ∧ (2 * (L + W) = 184) :=
by
  let leg1 := 2
  let leg2 := 3
  let n := 60
  let area_of_one_triangle := (1 / 2) * leg1 * leg2
  let total_area := n * area_of_one_triangle
  use 2, 90
  split
  · calc
      2 * 90 = total_area :=
      by norm_num
  · calc
      2 * (2 + 90) = 184 :=
      by norm_num

end max_perimeter_of_rectangle_l426_426337


namespace chord_length_of_intersection_l426_426940

noncomputable def chord_length_parametric : ℝ :=
  let x := λ t : ℝ, 2 + t
  let y := λ t : ℝ, t * Real.sqrt 3
  let hyperbola := λ x y : ℝ, x^2 - y^2 = 1
  2 * Real.sqrt 10

theorem chord_length_of_intersection : 
  ∀ (t : ℝ), hyperbola (x t) (y t) → chord_length_parametric = 2 * Real.sqrt 10 :=
  sorry

end chord_length_of_intersection_l426_426940


namespace find_some_number_l426_426397

noncomputable def some_number (d : ℝ) : ℝ := 0.889 * 55 / d

theorem find_some_number 
  (h : Float.round (some_number 4.9) = 10.0) :
  true := 
sorry

end find_some_number_l426_426397


namespace prove_grocery_expense_l426_426790

def cost_of_expenses (rent utilities internet cleaning_supplies : Nat) : Nat :=
  rent + utilities + internet + cleaning_supplies

def cost_per_roommate (total_expenses roommates : Nat) : Nat :=
  total_expenses / roommates

def total_grocery_expense (one_roommate_payment rent utilities internet cleaning_supplies : Nat) (roommates : Nat) : Nat :=
  let total_expenses := cost_of_expenses rent utilities internet cleaning_supplies
  let per_person_expenses := cost_per_roommate total_expenses roommates
  (one_roommate_payment - per_person_expenses) * roommates

theorem prove_grocery_expense
  (rent utilities internet cleaning_supplies one_roommate_payment roommates : Nat)
  (h_rent : rent = 1100)
  (h_utilities : utilities = 114)
  (h_internet : internet = 60)
  (h_cleaning_supplies : cleaning_supplies = 40)
  (h_one_roommate_payment : one_roommate_payment = 924)
  (h_roommates : roommates = 3) : 
  total_grocery_expense one_roommate_payment rent utilities internet cleaning_supplies roommates = 1458 := 
by
  rw [h_rent, h_utilities, h_internet, h_cleaning_supplies, h_one_roommate_payment, h_roommates]
  simp [total_grocery_expense, cost_of_expenses, cost_per_roommate]
  -- Historical calculations can be put here
  sorry

end prove_grocery_expense_l426_426790


namespace investment_amount_l426_426240

noncomputable def PV (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  FV / (1 + r) ^ n

theorem investment_amount (FV : ℝ) (r : ℝ) (n : ℕ) (PV : ℝ) : FV = 1000000 ∧ r = 0.08 ∧ n = 20 → PV = 1000000 / (1 + 0.08)^20 :=
by
  intros
  sorry

end investment_amount_l426_426240


namespace fraction_equal_l426_426247

variable {m n p q : ℚ}

-- Define the conditions
def condition1 := (m / n = 20)
def condition2 := (p / n = 5)
def condition3 := (p / q = 1 / 15)

-- State the theorem
theorem fraction_equal (h1 : condition1) (h2 : condition2) (h3 : condition3) : (m / q = 4 / 15) :=
  sorry

end fraction_equal_l426_426247


namespace probability_point_not_above_x_axis_l426_426701

noncomputable def A := (1, 6) : ℝ × ℝ
noncomputable def B := (3, 0) : ℝ × ℝ
noncomputable def C := (5, 0) : ℝ × ℝ
noncomputable def D := (3, 6) : ℝ × ℝ

def is_point_not_above_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 ≤ 0

def area_parallelogram (A B C D : ℝ × ℝ) : ℝ :=
  let base := real.dist B C
  let height := real.dist B A
  base * height

theorem probability_point_not_above_x_axis :
  let area := area_parallelogram A B C D
  let area_below := 0
  area > 0 →
  (area_below / area) = 0 :=
by
  sorry

end probability_point_not_above_x_axis_l426_426701


namespace fraction_problem_l426_426243

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end fraction_problem_l426_426243


namespace find_average_weight_of_additional_friends_l426_426367

-- Define the conditions
variables {initial_friends additional_friends total_friends : ℕ}
variables {initial_avg_weight new_avg_weight weight_increase : ℝ}

-- Set the conditions for our problem
def problem_conditions : Prop :=
  initial_friends = 50 ∧ 
  additional_friends = 40 ∧ 
  total_friends = initial_friends + additional_friends ∧
  weight_increase = 12 ∧
  new_avg_weight = 46 ∧
  new_avg_weight - weight_increase = initial_avg_weight

-- Define what we need to prove
def average_weight_of_additional_friends : ℝ :=
  let total_initial_weight := initial_friends * initial_avg_weight in
  let total_new_weight := total_friends * new_avg_weight in
  let total_additional_weight := total_new_weight - total_initial_weight in
  total_additional_weight / additional_friends

-- Prove that the average weight of the additional friends is 61 kg
theorem find_average_weight_of_additional_friends (h : problem_conditions) : 
  average_weight_of_additional_friends = 61 :=
sorry


end find_average_weight_of_additional_friends_l426_426367


namespace maximize_S_minimize_S_under_constraint_l426_426916

theorem maximize_S (x : Fin 5 → ℕ) (h : ∑ i, x i = 2006) : 
  let S := ∑ i j in Finset.off_diag (Finset.univ : Finset (Fin 5)), x i * x j
  in
  S ≤ (∑ i j in Finset.off_diag (Finset.univ : Finset (Fin 5)), (if i = 0 then 402 else 401) * (if j = 0 then 402 else 401)) :=
sorry

theorem minimize_S_under_constraint (x : Fin 5 → ℕ) (h : ∑ i, x i = 2006) (h_dist : ∀ i j, |x i - x j| ≤ 2) : 
  let S := ∑ i j in Finset.off_diag (Finset.univ : Finset (Fin 5)), x i * x j
  in
  S ≥ (∑ i j in Finset.off_diag (Finset.univ : Finset (Fin 5)), (if i < 3 then 402 else 400) * (if j < 3 then 402 else 400)) :=
sorry

end maximize_S_minimize_S_under_constraint_l426_426916


namespace place_plus_min_sum_place_times_max_product_l426_426448

/-- 
  Given a number in the form 1 followed by 1991 digits of 9 and another 1,
  prove that:
  1. Placing a "+" sign between the 996th and 997th digits yields the minimum possible sum.
  2. Placing a "×" sign between the 995th and 996th digits yields the maximum possible product.
-/
theorem place_plus_min_sum (m : ℕ) (h1 : m = 996 ∨ m = 997) :
  let n : ℕ := 10^(1992)
  1 ∑ (2 * 10^m - 1) + (n / 10^m - 9) = 3 * 10 ^ 996 - 10 :=
sorry

theorem place_times_max_product (m : ℕ) (h1 : m = 995 ∨ m = 996) :
  let n : ℕ := 10^(1992)
  2 ∏ (2 * 10^m - 1) * (n / 10^m - 9) = 2 * 10 ^ 1992 - 1 :=
sorry

end place_plus_min_sum_place_times_max_product_l426_426448


namespace dice_probability_sum_24_l426_426952

-- Define the probability of each die showing a 6 as 1/6
def die_probability : ℝ := 1 / 6

-- Define the event of sum of four dice showing 24
def event_sum_24 := (die_probability ^ 4 = 1 / 1296)

theorem dice_probability_sum_24 : event_sum_24 :=
by
  sorry

end dice_probability_sum_24_l426_426952


namespace probability_detecting_drunk_driver_l426_426864

namespace DrunkDrivingProbability

def P_A : ℝ := 0.05
def P_B_given_A : ℝ := 0.99
def P_B_given_not_A : ℝ := 0.01

def P_not_A : ℝ := 1 - P_A

def P_B : ℝ := P_A * P_B_given_A + P_not_A * P_B_given_not_A

theorem probability_detecting_drunk_driver :
  P_B = 0.059 :=
by
  sorry

end DrunkDrivingProbability

end probability_detecting_drunk_driver_l426_426864


namespace simplify_expression_l426_426152

theorem simplify_expression : 
    1 - 1 / (1 + Real.sqrt (2 + Real.sqrt 3)) + 1 / (1 - Real.sqrt (2 - Real.sqrt 3)) 
    = 1 + (Real.sqrt (2 - Real.sqrt 3) + Real.sqrt (2 + Real.sqrt 3)) / (-1 - Real.sqrt 3) := 
by
  sorry

end simplify_expression_l426_426152


namespace difference_in_profit_l426_426705

theorem difference_in_profit (n : ℕ)
  (h : n = 200)
  (sammy_offer_price_per_record : ℕ)
  (sammy_offer_price : sammy_offer_price_per_record = 4)
  (bryan_offer_price_first_half : ℕ)
  (bryan_offer_price_first_half = 6)
  (bryan_offer_price_second_half : ℕ)
  (bryan_offer_price_second_half = 1) :
  let sammy_total := n * sammy_offer_price_per_record,
      bryan_total := (n / 2) * bryan_offer_price_first_half + (n / 2) * bryan_offer_price_second_half in
  sammy_total - bryan_total = 100 :=
by sorry

end difference_in_profit_l426_426705


namespace roots_of_quadratic_equation_are_real_and_distinct_l426_426771

def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_of_quadratic_equation_are_real_and_distinct :
  quadratic_discriminant 1 (-2) (-6) > 0 :=
by
  norm_num
  sorry

end roots_of_quadratic_equation_are_real_and_distinct_l426_426771


namespace find_a_l426_426997

-- Define the function
def f (a b x : ℝ) : ℝ := real.sqrt (a * x^2 + b * x)

-- Define the specific condition for b
variable {b : ℝ} (hb : b > 0)

-- State the theorem
theorem find_a (a : ℝ) (h : (a ≠ 0)) : 
  (∀ x : ℝ, 0 <= a * x^2 + b * x ↔ (0 <= x * (a + b * (1/x))) ) ->  
  (b := 0) -> 
  (∀ x : ℝ, f a b x ≠ 1) -> 
  (0 <= a * x ^ 2 + b * x) → 
  (D = A) :=
begin
  
end

-- State an example if a = -4
example : ∃ a : ℝ, a = -4 :=
begin
  existsi -4,
  refl,
end

end find_a_l426_426997


namespace bob_questions_created_l426_426125

theorem bob_questions_created :
  let q1 := 13
  let q2 := 2 * q1
  let q3 := 2 * q2
  q1 + q2 + q3 = 91 :=
by
  sorry

end bob_questions_created_l426_426125


namespace limit_expression_l426_426153

theorem limit_expression : 
  (filter.at_top.tendsto (λ n : ℕ, (2:ℝ)^(n+1) + (3:ℝ)^(n+1)) (λ n : ℕ, (2:ℝ)^n + (3:ℝ)^n) (3:ℝ)) :=
sorry

end limit_expression_l426_426153


namespace fraction_of_reciprocal_l426_426039

theorem fraction_of_reciprocal (x : ℝ) (hx : 0 < x) (h : (2/3) * x = y / x) (hx1 : x = 1) : y = 2/3 :=
by
  sorry

end fraction_of_reciprocal_l426_426039


namespace domain_of_f_l426_426902

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

theorem domain_of_f : { x : ℝ | x > 1 } = { x : ℝ | ∃ y, f y = f x } :=
by sorry

end domain_of_f_l426_426902


namespace num_consecutive_integers_in_list_D_l426_426328

-- Definitions based on conditions
def listConsecutive : Set Int := {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7}
def leastIntInD : Int := -4
def rangePositiveIntInD : Int := 6

-- Theorem statement
theorem num_consecutive_integers_in_list_D : (listConsecutive.card : Int) = 12 :=
by
  sorry

end num_consecutive_integers_in_list_D_l426_426328


namespace hyperbola_eccentricity_sqrt5_l426_426206

noncomputable def eccentricity_of_hyperbola (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b/a)^2)

theorem hyperbola_eccentricity_sqrt5
  (a b : ℝ)
  (h : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (y = x^2 + 1) → (x, y) = (1, 2)) :
  eccentricity_of_hyperbola a b = Real.sqrt 5 :=
by sorry

end hyperbola_eccentricity_sqrt5_l426_426206


namespace min_value_of_expression_l426_426546

theorem min_value_of_expression : ∀ x : ℝ, ∃ (M : ℝ), (∀ x, 16^x - 4^x - 4^(x+1) + 3 ≥ M) ∧ M = -4 :=
by
  sorry

end min_value_of_expression_l426_426546


namespace bob_questions_created_l426_426126

theorem bob_questions_created :
  let q1 := 13
  let q2 := 2 * q1
  let q3 := 2 * q2
  q1 + q2 + q3 = 91 :=
by
  sorry

end bob_questions_created_l426_426126


namespace linear_relationship_increase_in_y_l426_426647

theorem linear_relationship_increase_in_y (x y : ℝ) (hx : x = 12) (hy : y = 10 / 4 * x) : y = 30 := by
  sorry

end linear_relationship_increase_in_y_l426_426647


namespace flowers_per_vase_l426_426484

theorem flowers_per_vase (carnations roses vases total_flowers flowers_per_vase : ℕ)
  (h1 : carnations = 7)
  (h2 : roses = 47)
  (h3 : vases = 9)
  (h4 : total_flowers = carnations + roses)
  (h5 : flowers_per_vase = total_flowers / vases):
  flowers_per_vase = 6 := 
by {
  sorry
}

end flowers_per_vase_l426_426484


namespace dot_product_eq_two_l426_426611

variables (a b : ℝ^3)
-- Conditions
def cond1 : Prop := ∥a∥ = √3
def cond2 : Prop := ∥b∥ = 2
def cond3 : Prop := ∥a - 2 • b∥ = √11

theorem dot_product_eq_two
  (h1 : cond1 a)
  (h2 : cond2 b)
  (h3 : cond3 a b) : dot_product a b = 2 :=
sorry

end dot_product_eq_two_l426_426611


namespace geometric_seq_a5_a7_l426_426569

theorem geometric_seq_a5_a7 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : a 3 + a 5 = 6)
  (q : ℝ) :
  (a 5 + a 7 = 12) :=
sorry

end geometric_seq_a5_a7_l426_426569


namespace num_common_divisors_90_150_l426_426615

-- Definitions of the given conditions
def fact_90 : multiset ℕ := {2, 3, 3, 5}
def fact_150 : multiset ℕ := {2, 3, 5, 5}

-- Definition to calculate the GCD of two multisets of prime factors
def multiset_gcd (a b : multiset ℕ) : multiset ℕ :=
multiset.inter a b

-- GCD of 90 and 150
def gcd_90_150 : multiset ℕ := multiset_gcd fact_90 fact_150

-- Set of divisors of 30 (GCD of 90 and 150)
def divisors (n : ℕ) : list ℕ :=
list.filter (λ x, n % x = 0) (list.range (n + 1))

-- Divisors of 30
def divisors_30 : list ℕ := divisors 30

-- Main theorem
theorem num_common_divisors_90_150 : divisors_30.length = 8 := by
  sorry

end num_common_divisors_90_150_l426_426615


namespace tan_theta_eq_three_l426_426178

theorem tan_theta_eq_three
  (θ : ℝ)
  (h₀ : tan θ = 3) :
    (2 - 2 * cos θ) / sin θ - sin θ / (2 + 2 * cos θ) = 0 :=
  sorry

end tan_theta_eq_three_l426_426178


namespace total_surface_area_proof_l426_426733

-- Given Definitions
variables (a alpha beta : ℝ)
variables (Sin_alpha Cos_beta Cos_half_beta : ℝ)

-- Conditions
def rhombus_side := a
def rhombus_angle := alpha < 90
def inclination_angle := beta

-- Trigonometric Identities and Shape Area
def Sin_alpha_def := Sin_alpha = sin alpha
def Cos_beta_def := Cos_beta = cos beta
def Cos_half_beta_def := Cos_half_beta = cos (beta / 2)

-- Area of the slant surface
def area_slant := (a^2 * Sin_alpha) / Cos_beta

-- Total surface area
def total_surface_area := a^2 * Sin_alpha * (1 / Cos_beta + 1)

-- Proof statement
theorem total_surface_area_proof :
  (total_surface_area a Sin_alpha Cos_beta = (2 * a^2 * Sin_alpha * Cos_half_beta^2) / Cos_beta) :=
sorry

end total_surface_area_proof_l426_426733


namespace half_sum_of_squares_of_even_or_odd_l426_426347

theorem half_sum_of_squares_of_even_or_odd (n1 n2 : ℤ) (a b : ℤ) :
  (n1 % 2 = 0 ∧ n2 % 2 = 0 ∧ n1 = 2*a ∧ n2 = 2*b ∨
   n1 % 2 = 1 ∧ n2 % 2 = 1 ∧ n1 = 2*a + 1 ∧ n2 = 2*b + 1) →
  ∃ x y : ℤ, (n1^2 + n2^2) / 2 = x^2 + y^2 :=
by
  intro h
  sorry

end half_sum_of_squares_of_even_or_odd_l426_426347


namespace sum_of_coefficients_l426_426198

theorem sum_of_coefficients (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1):
  let T : ℕ → ℂ := λ n, α^n + β^n in
  (∑ coeff in (T 2005).coefficients, coeff) = 1 :=
sorry

end sum_of_coefficients_l426_426198


namespace bridge_length_is_255_l426_426817

-- Defining the constants and conditions
def train_length : ℝ := 120  -- in meters
def train_speed_kmh : ℝ := 45  -- in km/hr
def cross_time : ℝ := 30  -- in seconds

-- Constant for conversion factors
def km_to_meter : ℝ := 1000
def hr_to_second : ℝ := 3600

-- Conversion from km/hr to m/s
def train_speed_ms : ℝ := train_speed_kmh * (km_to_meter / hr_to_second)

-- Total distance covered in cross_time seconds
def total_distance : ℝ := train_speed_ms * cross_time

-- Length of the bridge
def bridge_length : ℝ := total_distance - train_length

-- The theorem to prove
theorem bridge_length_is_255 : bridge_length = 255 := by
  sorry

end bridge_length_is_255_l426_426817


namespace range_of_y_when_x_3_l426_426967

variable (a c : ℝ)

theorem range_of_y_when_x_3 (h1 : -4 ≤ a + c ∧ a + c ≤ -1) (h2 : -1 ≤ 4 * a + c ∧ 4 * a + c ≤ 5) :
  -1 ≤ 9 * a + c ∧ 9 * a + c ≤ 20 :=
sorry

end range_of_y_when_x_3_l426_426967


namespace geometric_sequence_sixth_term_l426_426945

theorem geometric_sequence_sixth_term (a : ℕ) (a2 : ℝ) (aₖ : ℕ → ℝ) (r : ℝ) (k : ℕ) (h1 : a = 3) (h2 : a2 = -1/6) (h3 : ∀ n, aₖ n = a * r^(n-1)) (h4 : r = a2 / a) (h5 : k = 6) :
  aₖ k = -1 / 629856 :=
by sorry

end geometric_sequence_sixth_term_l426_426945


namespace arrangement_count_l426_426094

theorem arrangement_count (n_classes : ℕ) (n_factories : ℕ) (classes_per_factory : Finset (Finset ℕ)) :
  n_classes = 5 → n_factories = 4 → 
  (∀ f ∈ classes_per_factory, Finset.card f ≥ 1) → 
  Finset.card classes_per_factory = n_factories →
  ∑ f in classes_per_factory, Finset.card f = n_classes →
  ∃ count : ℕ, count = 240 :=
begin
  intros h1 h2 h3 h4 h5,
  use 240,
  sorry
end

end arrangement_count_l426_426094


namespace math_proof_problem_l426_426187

-- Definitions for conditions:
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 2) = -f x
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x - 3 / 4) = -f (- (x - 3 / 4))

-- Statements to prove:
def statement1 (f : ℝ → ℝ) : Prop := ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x
def statement2 (f : ℝ → ℝ) : Prop := ∀ x, f (-(3 / 4) - x) = f (-(3 / 4) + x)
def statement3 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def statement4 (f : ℝ → ℝ) : Prop := ¬(∀ x y : ℝ, x < y → f x ≤ f y)

theorem math_proof_problem (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  statement1 f ∧ statement2 f ∧ statement3 f ∧ statement4 f :=
by
  sorry

end math_proof_problem_l426_426187


namespace possible_real_values_l426_426292

theorem possible_real_values (y: ℝ) (h_mean: (8 + 1 + 7 + 1 + y + 1 + 9) / 7 = (27 + y) / 7)
  (h_mode: 1 = 1) 
  (h_median: (y < 1 → 1 = 1) ∧ (1 < y ∧ y < 7 → y = y) ∧ (y ≥ 7 → 7 = 7)) 
  (h_ap: ∃a b c: ℝ, a = 1 ∧ b = 7 ∧ c = (27 + y) / 7 ∧ (b - a = 7 - 1) ∧ (c - b = (27 + y) / 7 - 7)): y = 64 := 
by {
  sorry
}

end possible_real_values_l426_426292


namespace tangent_line_at_e_monotonic_intervals_l426_426216

noncomputable def f (x a : ℝ) : ℝ := (2 * x^2 - 4 * a * x) * Real.log x + x^2

theorem tangent_line_at_e (x y a : ℝ) (h : a = 0) (hx : x = Real.exp 1) :
  8 * Real.exp 1 * x - y - 5 * (Real.exp 1)^2 = 0 :=
begin
  sorry
end

theorem monotonic_intervals (a : ℝ) :
  (if a ≤ 0 then 
    (∀ x, (0 < x ∧ x < Real.exp (-1)) → f x a < f (Real.exp (-1)) a) ∧ 
    (∀ x, (Real.exp (-1) < x) → f x a > f (Real.exp (-1)) a))
  else if 0 < a ∧ a < Real.exp (-1) then
    (∀ x, (0 < x ∧ x < a) → f x a < f a a) ∧ 
    (∀ x, (a < x ∧ x < Real.exp (-1)) → f x a > f a a) ∧ 
    (∀ x, (Real.exp (-1) < x) → f x a > f (Real.exp (-1)) a)
  else if a = Real.exp (-1) then
    (∀ x, (0 < x) → f x a > f (Real.exp 1) a)
  else 
    (∀ x, (0 < x ∧ x < Real.exp (-1)) → f x a < f (Real.exp (-1)) a) ∧ 
    (∀ x, (Real.exp (-1) < x ∧ x < a) → f x a > f a a) ∧ 
    (∀ x, (a < x) → f x a > f a a)) :=
begin
  sorry
end

end tangent_line_at_e_monotonic_intervals_l426_426216


namespace construct_square_within_sector_l426_426510
-- We start by importing the necessary libraries for geometric constructions.

-- We define the problem statement capturing the conditions and the question.
theorem construct_square_within_sector (O : Type*) [metric_space O] [normed_group O] 
  [normed_space ℝ O] (circle : O) (A B : O) 
  (h : dist A O = dist B O ∧ is_diameter A B circle) :
  ∃ C D : O, is_square_within_sector C D A B O :=
sorry -- Proof omitted

end construct_square_within_sector_l426_426510


namespace probability_of_x_add_y_lt_5_l426_426463

-- Define the square.
def square (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3

-- Define the condition for the problem.
def condition (x y : ℝ) : Prop := square x y ∧ x + y < 5

-- Define the probability function.
def probability (pred : ℝ → ℝ → Prop) : ℝ :=
  let area_condition := (3 - 0) * (3 - 0) - (1 / 2 * 1 * 1) in
  let area_total := (3 - 0) * (3 - 0) in
  area_condition / area_total

-- Prove that the probability of the point (x, y) satisfying x + y < 5 inside the square is 17/18.
theorem probability_of_x_add_y_lt_5 : probability condition = 17 / 18 := 
by
  sorry

end probability_of_x_add_y_lt_5_l426_426463


namespace men_per_table_l426_426863

theorem men_per_table 
  (num_tables : ℕ) 
  (women_per_table : ℕ) 
  (total_customers : ℕ) 
  (h1 : num_tables = 9) 
  (h2 : women_per_table = 7) 
  (h3 : total_customers = 90)
  : (total_customers - num_tables * women_per_table) / num_tables = 3 :=
by
  sorry

end men_per_table_l426_426863


namespace least_number_to_add_l426_426819

theorem least_number_to_add {n : ℕ} (h : n = 1202) : (∃ k : ℕ, (n + k) % 4 = 0 ∧ ∀ m : ℕ, (m < k → (n + m) % 4 ≠ 0)) ∧ k = 2 := by
  sorry

end least_number_to_add_l426_426819


namespace number_of_distinct_sums_l426_426873

def bag_A : Finset ℕ := {1, 3, 5}
def bag_B : Finset ℕ := {2, 4, 6}

theorem number_of_distinct_sums : (bag_A.product bag_B).image (λ (p : ℕ × ℕ), p.1 + p.2).card = 5 := by
  sorry

end number_of_distinct_sums_l426_426873


namespace area_ratio_of_smaller_square_l426_426101

-- Definitions required by conditions
def isInscribedSquare(circle : Type) (large_square : Type) : Prop :=
  -- Define what it means for a square to be inscribed in a circle
  sorry

def smallerSquareConfig(circle : Type) (large_square : Type) (small_square : Type) : Prop :=
  -- Define the configuration where the smaller square has one side
  -- coinciding with a side of the larger square and two vertices on the circle
  sorry

-- Mathematical equivalent proof problem
theorem area_ratio_of_smaller_square (circle large_square small_square : Type)
  (Hinscribed : isInscribedSquare circle large_square)
  (Hconfig : smallerSquareConfig circle large_square small_square) :
  (area small_square) = 0.04 * (area large_square) :=
sorry

end area_ratio_of_smaller_square_l426_426101


namespace valid_outfits_count_l426_426617

theorem valid_outfits_count :
  let shirts := 5
  let pants := 5
  let hats := 5
  let colors := 5
  (∑ (s in {1,2,3,4,5}) (p in {1,2,3,4,5}) (h in {1,2,3,4,5}), 
  (s ≠ p) ∧ (p ≠ h) ∧ (h ≠ s) : ℕ) = 60 := 
by
  let shirts := 5
  let pants := 5
  let hats := 5
  let colors := 5
  let total_outfits := shirts * pants * hats
  let restricted_outfits := 3 * colors * (colors - 1) + colors
  have total := total_outfits - restricted_outfits
  exact total

sorry

end valid_outfits_count_l426_426617


namespace find_flat_rate_l426_426236

variable (flat_rate total_amount_per_session amount_per_minute total_minutes : ℕ)

-- Given conditions
def given_conditions :=
  total_amount_per_session = 146 ∧
  amount_per_minute = 7 ∧
  total_minutes = 18

-- Calculate total amount charged for minutes
def total_amount_for_minutes :=
  total_minutes * amount_per_minute

-- Proof statement
theorem find_flat_rate (h : given_conditions) :
  total_amount_per_session - total_amount_for_minutes = 20 :=
by
  sorry

end find_flat_rate_l426_426236


namespace kiana_siblings_ages_l426_426304

/-- Kiana has two twin brothers, one is twice as old as the other, 
and their ages along with Kiana's age multiply to 72. Prove that 
the sum of their ages is 13. -/
theorem kiana_siblings_ages
  (y : ℕ) (K : ℕ) (h1 : 2 * y * K = 72) :
  y + 2 * y + K = 13 := 
sorry

end kiana_siblings_ages_l426_426304


namespace min_value_abc_squared_l426_426695

variables {a b c t : ℝ}

theorem min_value_abc_squared (h : a + b + c = t) : 
  ∃ u : ℝ, (u = a ∧ u = b ∧ u = c ∧ a^2 + b^2 + c^2 = u^2 + u^2 + u^2) ∧ a^2 + b^2 + c^2 ≥ t^2 / 3 :=
begin
  sorry
end

end min_value_abc_squared_l426_426695


namespace find_k_l426_426272

theorem find_k (k x : ℝ) (h1 : x + k - 4 = 0) (h2 : x = 2) : k = 2 :=
by
  sorry

end find_k_l426_426272


namespace gain_percent_l426_426067

-- Definitions for the problem
variables (MP CP SP : ℝ)
def cost_price := CP = 0.64 * MP
def selling_price := SP = 0.88 * MP

-- The statement to prove
theorem gain_percent (h1 : cost_price MP CP) (h2 : selling_price MP SP) :
  (SP - CP) / CP * 100 = 37.5 := 
sorry

end gain_percent_l426_426067


namespace range_of_a_l426_426202

def exists_unique_x2 (e : ℝ) (a : ℝ) :=
  ∀ x_1 : ℝ, (0 ≤ x_1 ∧ x_1 ≤ 1) → ∃! x_2 : ℝ, (−1 ≤ x_2 ∧ x_2 ≤ 1) ∧ (x_1 + x_2^2 * Real.exp(x_2) = a)

theorem range_of_a (e : ℝ) :
  (∃! x_1 ∈ Set.Icc 0 1, ∃ x_2 ∈ Set.Icc (-1 : ℝ) 1, x_1 + x_2^2 * Real.exp(x_2) = a) ↔ (a ∈ Set.Ioo (1 + 1/e) e) :=
by
  sorry

end range_of_a_l426_426202


namespace scientific_notation_of_570_million_l426_426728

theorem scientific_notation_of_570_million :
  570000000 = 5.7 * 10^8 := sorry

end scientific_notation_of_570_million_l426_426728


namespace intersection_M_N_l426_426271

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | x ≥ -1 / 3}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 / 3 ≤ x ∧ x < 4} :=
sorry

end intersection_M_N_l426_426271


namespace monthly_income_calculation_l426_426442

variable (deposit : ℝ)
variable (percentage : ℝ)
variable (monthly_income : ℝ)

theorem monthly_income_calculation 
    (h1 : deposit = 3800) 
    (h2 : percentage = 0.32) 
    (h3 : deposit = percentage * monthly_income) : 
    monthly_income = 11875 :=
by
  sorry

end monthly_income_calculation_l426_426442


namespace polynomial_roots_cases_l426_426504

-- Define variables and conditions
variables {a b c d : ℝ} 
def f (x : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + d

-- Define the theorem to be proven
theorem polynomial_roots_cases (h: ∃ α β γ : ℝ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ f α = 0 ∧ f β = 0 ∧ f γ = 0) :
  (∃ δ : ℝ, δ ≠ α ∧ δ ≠ β ∧ δ ≠ γ ∧ f δ = 0) ∨ (∃ ξ : ℝ, (ξ = α ∨ ξ = β ∨ ξ = γ) ∧ 4 * ξ^3 + 3 * a * ξ^2 + 2 * b * ξ + c = 0) :=
sorry

end polynomial_roots_cases_l426_426504


namespace range_of_a_l426_426273

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (1 - x - a) < 1) → -1/2 < a ∧ a < 3/2 :=
by
  sorry

end range_of_a_l426_426273


namespace not_divisible_by_5_l426_426558

theorem not_divisible_by_5 (b : ℕ) : b = 6 ↔ ¬ (5 ∣ (2 * b ^ 3 - 2 * b ^ 2 + 2 * b - 1)) :=
sorry

end not_divisible_by_5_l426_426558


namespace fifteenth_entry_l426_426551

def r_7 (n : ℕ) : ℕ := n % 7

theorem fifteenth_entry :
  ∃ (n_list : List ℕ), 
    (∀ n, n ∈ n_list → r_7 (3 * n) ≤ 3) ∧ 
    ∃ (n₁₅ : ℕ), n₁₅ = n_list.nthLe 14 (by linarith) ∧ 
    n₁₅ = 22 := 
sorry

end fifteenth_entry_l426_426551


namespace original_acid_percentage_l426_426851

-- Define the variables: a for acid and w for water in the original mixture.
variables (a w : ℝ)

-- Given conditions
def total_mixture := a + w = 10
def after_adding_1_ounce_water := a / (a + w + 1) = 1 / 4
def after_adding_1_ounce_acid := (a + 1) / (a + w + 2) = 0.4

-- Theorem to prove
theorem original_acid_percentage (h₁ : total_mixture) (h₂ : after_adding_1_ounce_water) (h₃ : after_adding_1_ounce_acid) :
  (a / 10) * 100 = 27.5 :=
sorry

end original_acid_percentage_l426_426851


namespace count_zeros_tan_function_l426_426747

noncomputable def tan_function (x : ℝ) : ℝ :=
  Real.tan (2015 * x) - Real.tan (2016 * x) + Real.tan (2017 * x)

theorem count_zeros_tan_function : 
  (∃ n : ℕ, n = 2017 ∧ ∀ x ∈ Icc (0 : ℝ) Real.pi, tan_function x = 0) :=
sorry

end count_zeros_tan_function_l426_426747


namespace correct_options_l426_426385

/-- Dedekind cut definitions and conditions --/

def is_dedekind_cut (M N : set ℚ) : Prop :=
  M ∪ N = set.univ ∧
  M ∩ N = ∅ ∧
  ∀ m ∈ M, ∀ n ∈ N, m < n

/-- Option B: M has no maximum element, N has a minimum element. --/

def option_B (M N : set ℚ) : Prop :=
  (¬ ∃ m, ∀ x ∈ M, x ≤ m) ∧
  (∃ n, ∀ y ∈ N, y ≥ n)

/-- Option D: M has no maximum element, N has no minimum element. --/

def option_D (M N : set ℚ) : Prop :=
  (¬ ∃ m, ∀ x ∈ M, x ≤ m) ∧
  (¬ ∃ n, ∀ y ∈ N, y ≥ n)

/-- The statement to prove --/

theorem correct_options (M N : set ℚ) (h : is_dedekind_cut M N) : option_B M N ∨ option_D M N :=
sorry

end correct_options_l426_426385


namespace shape_match_after_rotation_l426_426507

-- Definitions
constant Shape : Type
constant rotate90Clockwise : Shape → Shape
constant X : Shape
constant A : Shape
constant B : Shape
constant C : Shape
constant D : Shape
constant E : Shape

-- Theorem to prove
theorem shape_match_after_rotation :
  rotate90Clockwise X = C :=
sorry

end shape_match_after_rotation_l426_426507


namespace checkerboard_squares_count_l426_426449

-- Definitions of checkerboard and necessary conditions
def is_alternating_black_white (i j : ℕ) : bool :=
  (i + j) % 2 == 0

def contains_at_least_8_black_squares (n : ℕ) : bool :=
  ∃ i1 j1, 
    (∀ i2 j2, i1 ≤ i2 ∧ i2 < i1 + n ∧ j1 ≤ j2 ∧ j2 < j1 + n →
    is_alternating_black_white i2 j2) &&
    (∑ i2 in Finset.range n, ∑ j2 in Finset.range n, dite (is_alternating_black_white (i1 + i2) (j1 + j2)) (λ _, 1) (λ _, 0) ≥ 8)

def count_valid_squares (size board_size : ℕ) : ℕ :=
  if h : 1 ≤ size ∧ size ≤ board_size then
    ∑ i in Finset.range (board_size - size + 1), ∑ j in Finset.range (board_size - size + 1),
      if contains_at_least_8_black_squares size then 1 else 0
  else 0

theorem checkerboard_squares_count :
  let board_size := 10 in 
  (count_valid_squares 4 board_size + 
   count_valid_squares 5 board_size + 
   count_valid_squares 6 board_size + 
   count_valid_squares 7 board_size + 
   count_valid_squares 8 board_size + 
   count_valid_squares 9 board_size + 
   count_valid_squares 10 board_size) = 115 := sorry

end checkerboard_squares_count_l426_426449


namespace not_always_ac_gt_bc_l426_426235

variables {a b c : ℝ}

theorem not_always_ac_gt_bc (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c ≠ 0) :
  ¬ (∀ c : ℝ, ac > bc) := sorry

end not_always_ac_gt_bc_l426_426235


namespace problem_l426_426709

theorem problem (p q : Prop) (m : ℝ):
  (p = (m > 1)) →
  (q = (-2 ≤ m ∧ m ≤ 2)) →
  (¬q = (m < -2 ∨ m > 2)) →
  (¬(p ∧ q)) →
  (p ∨ q) →
  (¬q) →
  m > 2 :=
by
  sorry

end problem_l426_426709


namespace part_i_minimum_value_part_ii_range_of_a_l426_426220

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) := x^2 - a * real.log x

-- Part I: Minimum value when a = 4 on [1, e]
theorem part_i_minimum_value : ∃ x ∈ set.Icc 1 real.exp, f x 4 = -3 :=
sorry

-- Part II: Range of a such that f(x) ≥ (a-2)x for some x ∈ [2, e]
theorem part_ii_range_of_a : ∀ x ∈ set.Icc 2 real.exp, f x a ≥ (a - 2) * x ↔ a ∈ set.Iio (8 / (2 + real.log 2)) :=
sorry

end part_i_minimum_value_part_ii_range_of_a_l426_426220


namespace quadrant_of_complex_z_l426_426963

noncomputable def complex_z : ℂ := (2 - complex.I) * (1 - complex.I)

theorem quadrant_of_complex_z : complex_z.im < 0 ∧ complex_z.re > 0 :=
by
  unfold complex_z
  -- Commenting proof details as it's not to be included according to the instructions
  sorry

end quadrant_of_complex_z_l426_426963


namespace evaluate_product_l426_426151

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product : (∏ i in finset.range 12, (3 - w ^ (i+1))) = 797161 := 
by
  sorry

end evaluate_product_l426_426151


namespace original_price_of_computer_l426_426270

theorem original_price_of_computer :
  ∃ (P : ℝ), (1.30 * P = 377) ∧ (2 * P = 580) ∧ (P = 290) :=
by
  existsi (290 : ℝ)
  sorry

end original_price_of_computer_l426_426270


namespace minimize_fifth_item_l426_426192

noncomputable def vec_norm (v: ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def dot_product (v1 v2: ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def sequence (a1 d: ℝ × ℝ × ℝ) (n: ℕ) : ℝ × ℝ × ℝ := 
  (a1.1 + (n - 1) * d.1, a1.2 + (n - 1) * d.2, a1.3 + (n - 1) * d.3)

theorem minimize_fifth_item 
  (a1 d: ℝ × ℝ × ℝ)
  (h1: vec_norm a1 = 2)
  (h2: vec_norm d = real.sqrt 2 / 4)
  (h3: 2 * dot_product a1 d = -1)
  (h4: ∀ n: ℕ, n ≥ 2 → sequence a1 d n - sequence a1 d (n - 1) = d):
  ∀ n: ℕ, vec_norm (sequence a1 d 5) ≤ vec_norm (sequence a1 d n) := sorry

end minimize_fifth_item_l426_426192


namespace no_integer_solution_for_z_l426_426628

theorem no_integer_solution_for_z (z : ℤ) (h : 2 / z = 2 / (z + 1) + 2 / (z + 25)) : false :=
by
  sorry

end no_integer_solution_for_z_l426_426628


namespace intersection_is_3_l426_426230

open Set -- Open the Set namespace to use set notation

theorem intersection_is_3 {A B : Set ℤ} (hA : A = {1, 3}) (hB : B = {-1, 2, 3}) :
  A ∩ B = {3} :=
by {
-- Proof goes here
  sorry
}

end intersection_is_3_l426_426230


namespace kite_perimeter_l426_426089

-- Given the kite's diagonals, shorter sides, and longer sides
def diagonals : ℕ × ℕ := (12, 30)
def shorter_sides : ℕ := 10
def longer_sides : ℕ := 15

-- Problem statement: Prove that the perimeter is 50 inches
theorem kite_perimeter (diag1 diag2 short_len long_len : ℕ) 
                       (h_diag : diag1 = 12 ∧ diag2 = 30)
                       (h_short : short_len = 10)
                       (h_long : long_len = 15) : 
                       2 * short_len + 2 * long_len = 50 :=
by
  -- We provide no proof, only the statement
  sorry

end kite_perimeter_l426_426089


namespace unique_solution_to_equation_l426_426937

theorem unique_solution_to_equation (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x^y - y = 2005) : x = 1003 ∧ y = 1 :=
by
  sorry

end unique_solution_to_equation_l426_426937


namespace derivative_f1_derivative_f2_derivative_f3_derivative_f4_derivative_f5_l426_426165

section derivatives

-- Define variables
variable {x : ℝ}

-- Define the functions and their derivatives
def f1 (x : ℝ) : ℝ := (3 * x^2 - 4 * x) * (2 * x + 1)
def df1 (x : ℝ) : ℝ := 18 * x^2 - 10 * x - 4

def f2 (x : ℝ) : ℝ := sin (x / 2) * (1 - 2 * cos (x / 4)^2)
def df2 (x : ℝ) : ℝ := -1 / 2 * cos x

def f3 (x : ℝ) : ℝ := 3^x * exp x - 2^x + exp 1
def df3 (x : ℝ) : ℝ := (log 3 + 1) * (3 * exp 1)^x - 2^x * log 2

def f4 (x : ℝ) : ℝ := log x / (x^2 + 1)
def df4 (x : ℝ) : ℝ := (x^2 + 1 - 2 * x^2 * log x) / (x * (x^2 + 1)^2)

def f5 (x : ℝ) : ℝ := log (2 * x - 1) - log (2 * x + 1)
def df5 (x : ℝ) : ℝ := 4 / (4 * x^2 - 1)

-- Proof statements without proofs

theorem derivative_f1 : (fun x => (deriv (f1 x)) = df1 x) := sorry
theorem derivative_f2 : (fun x => (deriv (f2 x)) = df2 x) := sorry
theorem derivative_f3 : (fun x => (deriv (f3 x)) = df3 x) := sorry
theorem derivative_f4 : (fun x => (deriv (f4 x)) = df4 x) := sorry
theorem derivative_f5 : (fun x => (deriv (f5 x)) = df5 x) := sorry

end derivatives

end derivative_f1_derivative_f2_derivative_f3_derivative_f4_derivative_f5_l426_426165


namespace polynomial_real_or_constant_l426_426678

theorem polynomial_real_or_constant (p q : Polynomial ℝ) 
  (h : ∀ z : ℂ, (p.eval z) * (q.eval (conj z)) ∈ ℝ) : 
  ∃ k : ℝ, p = k • q ∨ q = 0 :=
sorry

end polynomial_real_or_constant_l426_426678


namespace triangle_incenter_equilateral_l426_426713

theorem triangle_incenter_equilateral (a b c : ℝ) (h : (b + c) / a = (a + c) / b ∧ (a + c) / b = (a + b) / c) : a = b ∧ b = c :=
by
  sorry

end triangle_incenter_equilateral_l426_426713


namespace count_valid_n_values_l426_426684

def proper_divisors_product (n : ℕ) : ℕ :=
  Nat.divisors n |>.filter (λ d => d > 1 ∧ d < n) |>.prod

def satisfies_condition (n : ℕ) : Prop :=
  n ∣ proper_divisors_product n

theorem count_valid_n_values : 
  Finset.card (Finset.range 101).filter (λ n => 2 ≤ n ∧ ¬ satisfies_condition n) = 32 :=
by
  sorry

end count_valid_n_values_l426_426684


namespace minimum_value_of_x_plus_y_l426_426625

-- Define the conditions as a hypothesis and the goal theorem statement.
theorem minimum_value_of_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / x + 9 / y = 1) :
  x + y = 16 :=
by
  sorry

end minimum_value_of_x_plus_y_l426_426625


namespace imaginary_part_of_z_l426_426182

-- Definition of the imaginary unit i
def i : ℂ := Complex.I

-- Definition of the complex number z = 5i / (3 + 4i)
def z : ℂ := 5 * Complex.I / (3 + 4 * Complex.I)

-- Statement verifying the imaginary part of z
theorem imaginary_part_of_z : Complex.im z = 3 / 5 := by
  sorry

end imaginary_part_of_z_l426_426182


namespace sum_parts_of_252525_as_fraction_is_349_l426_426016

def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  have h : x = 2.5252525 := sorry
  250 / 99

noncomputable def sum_of_numerator_and_denominator_in_lowest_terms (x : ℚ) : ℕ :=
  let frac := repeating_decimal_to_fraction x
  frac.num.natAbs + frac.denom
  
theorem sum_parts_of_252525_as_fraction_is_349 :
  sum_of_numerator_and_denominator_in_lowest_terms 2.5252525 = 349 :=
begin
  sorry
end

end sum_parts_of_252525_as_fraction_is_349_l426_426016


namespace total_coffee_blend_cost_l426_426867

-- Define the cost per pound of coffee types A and B
def cost_per_pound_A := 4.60
def cost_per_pound_B := 5.95

-- Given the pounds of coffee for Type A and the blend condition for Type B
def pounds_A := 67.52
def pounds_B := 2 * pounds_A

-- Total cost calculation
def total_cost := (pounds_A * cost_per_pound_A) + (pounds_B * cost_per_pound_B)

-- Theorem statement: The total cost of the coffee blend is $1114.08
theorem total_coffee_blend_cost : total_cost = 1114.08 := by
  -- Proof omitted
  sorry

end total_coffee_blend_cost_l426_426867


namespace lowest_temperature_at_noon_l426_426731

theorem lowest_temperature_at_noon (T : Fin 5 → ℝ) (h_avg : (∑ i, T i) / 5 = 50) (h_range : ∃ i j, T j - T i ≤ 25) : 
  ∃ L, L = 30 :=
by
  sorry

end lowest_temperature_at_noon_l426_426731


namespace impossible_to_flip_all_cups_down_l426_426405

def odd (m : ℕ) : Prop := ∃ k, m = 2 * k + 1
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem impossible_to_flip_all_cups_down 
  (m : ℕ) (n : ℕ)
  (h1 : m >= 3) (h2 : odd m) (h3 : n >= 2) (h4 : n < m) (h5 : even n) :
  ¬ (∃ k, all_cups_down_after_k_moves m n k) :=
sorry

def all_cups_down_after_k_moves (m n k : ℕ) : Prop :=
-- This represents the condition where all cups are facing 
-- downward after k moves. You should define it according 
-- to the constraints given in the problem.
sorry

end impossible_to_flip_all_cups_down_l426_426405


namespace find_a_and_b_prove_inequality_l426_426999

def f (x a b : ℝ) : ℝ := (x + b) * (Real.exp x - a)

-- Given conditions
variables (b : ℝ) (h_b : 0 < b)
variables (a : ℝ)
variable (tangent_condition : (e - 1) * (-1) + e * f (-1 a b) + e - 1 = 0)

noncomputable def f_deriv (x a b : ℝ) : ℝ := (x + b + 1) * Real.exp x - a

theorem find_a_and_b
    (h : tangent_condition):
    a = 1 ∧ b = 1 := sorry

theorem prove_inequality (m : ℝ) (h_m : m ≤ 0) : 
    ∃ a b : ℝ, f x a b ≥ m * x^2 + x := sorry

end find_a_and_b_prove_inequality_l426_426999


namespace factor_x4_minus_81_l426_426921

theorem factor_x4_minus_81 : ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intro x
  sorry

end factor_x4_minus_81_l426_426921


namespace equal_sum_subsets_of_1989_l426_426907

open Finset

def divides_into_equal_sum_subsets (n : ℕ) (k : ℕ) (m : ℕ) (s : ℕ) : Prop :=
  ∃ (A : Fin n → Finset (Fin m)),
  (∀ i, (A i).card = k) ∧
  (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
  (∀ i, (A i).sum id = s)

theorem equal_sum_subsets_of_1989 :
  divides_into_equal_sum_subsets 117 17 1989 (117 * 17 * (1989 + 1) / (2 * 117)) :=
sorry

end equal_sum_subsets_of_1989_l426_426907


namespace functional_inequality_solution_l426_426186

noncomputable def my_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ( (x + y) / (1 + x * y) ) = f x + f y

noncomputable def solution (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, f x = c * log ( abs ( (x + 1) / (x - 1) ) )

theorem functional_inequality_solution :
  (∀ f : ℝ → ℝ, continuous f → f 0 = 0 → my_function f → solution f) := sorry

end functional_inequality_solution_l426_426186


namespace tanQ_tanR_value_l426_426290

variables {P Q R S H : Type} [triangle P Q R]
variables (HS : ℝ) (HP : ℝ)
variables (tanQ tanR : ℝ)

-- Given conditions: PQR is a triangle, orthocenter divides altitude PS with given segment lengths.
axiom HS_condition : HS = 10
axiom HP_condition : HP = 25
axiom tanQR_relation : tanQ * tanR = 7 / 2

-- The theorem to prove
theorem tanQ_tanR_value : tanQ * tanR = 7 / 2 :=
by
  sorry

end tanQ_tanR_value_l426_426290


namespace extreme_value_points_range_maximum_value_of_M_l426_426993

noncomputable def f (a x : ℝ) : ℝ := (a + 1/a) * Real.log x - x + 1/x
noncomputable def f_prime (a x : ℝ) : ℝ := ((a + 1/a) / x - 1 - 1/x^2)

-- Problem statement 1: Range of a for which f(x) has extreme value points
-- Theorem: if f has an extreme value point in (0, +∞), then a ∈ (0, 1) ∪ (1, +∞)
theorem extreme_value_points_range (a : ℝ) (h : 0 < a) : 
  (∃ x ∈ Set.Ioi 0, f_prime a x = 0) ↔ a ≠ 1 := sorry

-- Problem statement 2: Maximum value of M(a)
-- Theorem: if a ∈ (1, e], then there is a maximum value for M(a), which is 4/e.
noncomputable def M (a : ℝ) : ℝ := 
  2 * (a + 1/a) * Real.log a - 2 * a + 2 * 1/a

theorem maximum_value_of_M (a : ℝ) (h1 : 1 < a) (h2 : a ≤ Real.exp 1) :
  ∃ max_val : ℝ, max_val = M Real.exp 1 ∧ M(a) ≤ max_val :=
begin
  use 4 / Real.exp 1,
  split,
  { rw [M, Real.exp],
    simp },
  { sorry }
end

end extreme_value_points_range_maximum_value_of_M_l426_426993


namespace function_value_at_2a_l426_426964

-- Define the function
def f (x : ℝ) : ℝ := 3^x + 3^(-x)

-- Given the condition and prove the required statement
theorem function_value_at_2a {a : ℝ} (h : f a = 4) : f (2 * a) = 14 :=
by sorry

end function_value_at_2a_l426_426964


namespace domain_h_l426_426419

noncomputable def h (x : ℝ) : ℝ := (2 * x - 3) / (x^2 - 4)

theorem domain_h : 
  ∀ x : ℝ, ¬ (x = 2 ∨ x = -2) ↔ x ∈ (-∞, -2) ∪ (-2, 2) ∪ (2, ∞) :=
by
  intro x
  sorry

end domain_h_l426_426419


namespace pony_speed_l426_426438

theorem pony_speed (v : ℝ) (head_start : ℝ) (catch_time : ℝ) (horse_speed : ℝ) 
    (horse_distance : ℝ) : (head_start = 3) → (catch_time = 4) → (horse_speed = 35) → 
    (horse_distance = 35 * 4) → (7 * v = horse_distance) → 
    v = 20 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  linarith

# Let's declare constants according to the conditions
constant head_start : ℝ := 3
constant catch_time : ℝ := 4
constant horse_speed : ℝ := 35
constant horse_distance : ℝ := 35 * 4

# Now we can prove the theorem
example : (7 * 20 = horse_distance) :=
by
  -- expand constants
  unfold horse_distance
  -- check calculations
  norm_num
  -- equality check
  linarith

end pony_speed_l426_426438


namespace cost_of_sneakers_l426_426459

theorem cost_of_sneakers (s c : ℝ) (h1 : s + c = 101) (h2 : s = 100 + c) : s = 100.5 :=
by
  sorry

end cost_of_sneakers_l426_426459


namespace triangle_equilateral_from_condition_l426_426823

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
a = b ∧ b = c

theorem triangle_equilateral_from_condition (a b c h_a h_b h_c : ℝ)
  (h : a + h_a = b + h_b ∧ b + h_b = c + h_c) :
  is_equilateral a b c :=
sorry

end triangle_equilateral_from_condition_l426_426823


namespace unique_pos_int_for_t_l426_426190

theorem unique_pos_int_for_t (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) 
  (h₀: t > 0) 
  (h₁: a 1 = 1)
  (h₂: ∀ n, 2 * S n = (n + 1) * a n)
  (h₃: ∀ n, a n ^ 2 - t * a n - 2 * t ^ 2 < 0) :
  t ∈ (1/2 : ℝ, 1 : ℝ] ↔ ∃! n : ℕ, 0 < n ∧ a n ^ 2 - t * a n - 2 * t ^ 2 < 0 := sorry

end unique_pos_int_for_t_l426_426190


namespace perimeter_of_quadrilateral_eq_fifty_l426_426409

theorem perimeter_of_quadrilateral_eq_fifty
  (a b : ℝ)
  (h1 : a = 10)
  (h2 : b = 15)
  (h3 : ∀ (p q r s : ℝ), p + q = r + s) : 
  2 * a + 2 * b = 50 := 
by
  sorry

end perimeter_of_quadrilateral_eq_fifty_l426_426409


namespace sum_max_min_ratio_l426_426136

theorem sum_max_min_ratio (x y : ℝ) 
  (h_ellipse : 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0) 
  : (∃ m_max m_min : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0 → y = m_max * x ∨ y = m_min * x) ∧ (m_max + m_min = 37 / 22)) :=
sorry

end sum_max_min_ratio_l426_426136


namespace neznaika_made_an_error_l426_426335

theorem neznaika_made_an_error : 
  ∀ (a : Fin 11 → ℕ), 
  (∃ (d : Fin 11 → ℤ), 
    (∀ i, d i = (a ((i + 1) % 11) + a (i % 11)) ∨ d i = (a (i % 11) + a ((i + 1) % 11)) ) ∧
    (∃ (count_1 count_2 count_3 : ℕ), 
      count_1 = 4 ∧ count_2 = 4 ∧ count_3 = 3 ∧ 
      (∑ i in (Finset.range 11), abs (d i)) = (count_1 * 1 + count_2 * 2 + count_3 * 3)) →
    false) :=
begin
  sorry
end

end neznaika_made_an_error_l426_426335


namespace sum_of_squares_of_medians_l426_426054

theorem sum_of_squares_of_medians (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  let ma := √((2 * b^2 + 2 * c^2 - a^2) / 4),
      mb := √((2 * a^2 + 2 * c^2 - b^2) / 4),
      mc := √((2 * a^2 + 2 * b^2 - c^2) / 4) in
  ma^2 + mb^2 + mc^2 = 337.5 :=
by
  sorry

end sum_of_squares_of_medians_l426_426054


namespace integral_correct_l426_426895

noncomputable def integral_result : ℝ :=
  ∫ x in -2 * Real.pi / 3 .. 0, (Real.cos x) / (1 + Real.cos x - Real.sin x)

theorem integral_correct :
  integral_result = Real.pi / 3 - Real.log 2 :=
by
  sorry

end integral_correct_l426_426895


namespace scientific_notation_of_wavelength_l426_426029

def wavelength_initial : ℝ := 0.00000094

theorem scientific_notation_of_wavelength : wavelength_initial = 9.4 * 10^(-7) := 
by
  sorry

end scientific_notation_of_wavelength_l426_426029


namespace fifth_term_arithmetic_sequence_l426_426745

def arithmetic_sequence_fifth_term
  (x y : ℝ) 
  (h_sequence : [x + 2 * y, x - 2 * y, x ^ 2 - y ^ 2, x / y])
  (h_arithmetic : ∃ d : ℝ, ∀ n : ℕ, h_sequence[n+1] - h_sequence[n] = d) : Prop :=
  h_sequence[4] = sqrt 7 - 1 - 4 * y

theorem fifth_term_arithmetic_sequence
  (x y : ℝ)
  (y_ne_zero : y ≠ 0)
  (h_sequence : [x + 2 * y, x - 2 * y, x ^ 2 - y ^ 2, x / y])
  (h_arithmetic : ∃ d : ℝ, ∀ n : ℕ, h_sequence[n+1] - h_sequence[n] = d) : 
  arithmetic_sequence_fifth_term x y h_sequence h_arithmetic :=
sorry

end fifth_term_arithmetic_sequence_l426_426745


namespace total_payment_correct_l426_426521

noncomputable def calculate_total_payment : ℝ :=
  let original_price_vase := 200
  let discount_vase := 0.35 * original_price_vase
  let sale_price_vase := original_price_vase - discount_vase
  let tax_vase := 0.10 * sale_price_vase

  let original_price_teacups := 300
  let discount_teacups := 0.20 * original_price_teacups
  let sale_price_teacups := original_price_teacups - discount_teacups
  let tax_teacups := 0.08 * sale_price_teacups

  let original_price_plate := 500
  let sale_price_plate := original_price_plate
  let tax_plate := 0.10 * sale_price_plate

  (sale_price_vase + tax_vase) + (sale_price_teacups + tax_teacups) + (sale_price_plate + tax_plate)

theorem total_payment_correct : calculate_total_payment = 952.20 :=
by sorry

end total_payment_correct_l426_426521


namespace original_length_wire_l426_426462

-- Define the conditions.
def length_cut_off_parts : ℕ := 10
def remaining_length_relation (L_remaining : ℕ) : Prop :=
  L_remaining = 4 * (2 * length_cut_off_parts) + 10

-- Define the theorem to prove the original length of the wire.
theorem original_length_wire (L_remaining : ℕ) (H : remaining_length_relation L_remaining) : 
  L_remaining + 2 * length_cut_off_parts = 110 :=
by 
  -- Use the given conditions
  unfold remaining_length_relation at H
  -- The proof would show that the equation holds true.
  sorry

end original_length_wire_l426_426462


namespace baron_can_visit_all_residential_rooms_l426_426874

-- Definitions based on problem conditions
def initial_halls : ℕ := 9
def central_hall_has_arsenal : Prop := True
def halls_divided : ℕ → ℕ := λ n, n * n
def rooms_divided : ℕ → ℕ := λ n, n * n
def residential_rooms : ℕ := (8 * 8) * 8 -- 8 halls, each divided into 9, minus the central hall
def neighboring_doors : Prop := True

-- Eulerian Path conditions for residential rooms
def eulerian_path_exists : Prop := 
  (∀ (r : ℕ), even r ∧ connected r) -- Adjust for the rooms and their doors within the structure

theorem baron_can_visit_all_residential_rooms :
  initial_halls = 9 → 
  central_hall_has_arsenal → 
  (halls_divided 8 = 64) → 
  (rooms_divided 8 = 64) → 
  neighboring_doors → 
  eulerian_path_exists →
  True := sorry

end baron_can_visit_all_residential_rooms_l426_426874


namespace students_sampled_C_is_40_l426_426836

-- Define the given conditions as constants.
constant total_students : ℕ := 1200
constant students_A : ℕ := 380
constant students_B : ℕ := 420
constant sample_size : ℕ := 120

-- Define the calculation for students in major C based on the conditions.
def students_C : ℕ := total_students - students_A - students_B

-- Define the calculation for the sampling rate.
def sampling_rate : ℚ := sample_size / total_students

-- Define the number of students to be sampled from major C.
def students_sampled_C : ℕ := students_C * sampling_rate.toInt

-- The theorem we need to prove.
theorem students_sampled_C_is_40 : students_sampled_C = 40 := by
  -- place proof here
  sorry

end students_sampled_C_is_40_l426_426836


namespace solve_equation1_solve_equation2_l426_426356

-- Define the first equation (x-3)^2 + 2x(x-3) = 0
def equation1 (x : ℝ) : Prop := (x - 3)^2 + 2 * x * (x - 3) = 0

-- Define the second equation x^2 - 4x + 1 = 0
def equation2 (x : ℝ) : Prop := x^2 - 4 * x + 1 = 0

-- Theorem stating the solutions for the first equation
theorem solve_equation1 : ∀ (x : ℝ), equation1 x ↔ x = 3 ∨ x = 1 :=
by
  intro x
  sorry  -- Proof is omitted

-- Theorem stating the solutions for the second equation
theorem solve_equation2 : ∀ (x : ℝ), equation2 x ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  intro x
  sorry  -- Proof is omitted

end solve_equation1_solve_equation2_l426_426356


namespace intersections_count_l426_426062

theorem intersections_count
  (c : ℕ)  -- crosswalks per intersection
  (l : ℕ)  -- lines per crosswalk
  (t : ℕ)  -- total lines
  (h_c : c = 4)
  (h_l : l = 20)
  (h_t : t = 400) :
  t / (c * l) = 5 :=
  by
    sorry

end intersections_count_l426_426062


namespace eccentricity_range_l426_426749

noncomputable def ellipse_eccentricity (a e : ℝ) : ℝ := e * a

theorem eccentricity_range (x y : ℝ) (θ : ℝ) (e : ℝ) :
  (∃ a, a = 3) →
  (∃ x y, (x - 3)^2 + (y - 2)^2 = 4) →
  (∃ e, ellipse_eccentricity 3 e = 3) →
  (∃ θ, -1 ≤ θ.cos ∧ θ.cos ≤ 1) →
  (3 / (6 + 2 * cos θ)) = e →
  (3 / 8 ≤ e ∧ e ≤ 3 / 4) :=
by
  intros _ _ _ _
  sorry

end eccentricity_range_l426_426749


namespace sqrt_nested_calculation_l426_426889

theorem sqrt_nested_calculation : sqrt (144 * sqrt (64 * sqrt 36)) = 48 * sqrt 3 :=
by
  sorry

end sqrt_nested_calculation_l426_426889


namespace find_m_l426_426520

open Real

theorem find_m (m : ℝ) : 
  (10:ℝ)^m = 10^10 * sqrt ((10^100) / (10^(-2))) → m = 61 := 
by
  sorry

end find_m_l426_426520


namespace Felicity_used_23_gallons_l426_426531

variable (A Felicity : ℕ)
variable (h1 : Felicity = 4 * A - 5)
variable (h2 : A + Felicity = 30)

theorem Felicity_used_23_gallons : Felicity = 23 := by
  -- Proof steps would go here
  sorry

end Felicity_used_23_gallons_l426_426531


namespace find_f_2014_l426_426593

variable {f : ℝ → ℝ}

theorem find_f_2014 (h1 : ∀ x y : ℝ, 4 * f(x) * f(y) = f(x + y) + f(x - y)) (h2 : f(1) = 1/4) :
  f(2014) = -1/4 := 
sorry

end find_f_2014_l426_426593


namespace problem1_calculation_l426_426825

theorem problem1_calculation :
  (2 * Real.tan (Real.pi / 4) + (-1 / 2) ^ 0 + |Real.sqrt 3 - 1|) = 2 + Real.sqrt 3 :=
by
  sorry

end problem1_calculation_l426_426825


namespace polynomial_divisibility_l426_426711

theorem polynomial_divisibility (P : Polynomial ℝ) (n : ℕ) (hn : n > 0) : 
  ∃ Q : Polynomial ℝ, P.eval₂ Polynomial.C x^[n] - Polynomial.C x = (P.eval₂ Polynomial.C x - Polynomial.C x) * Q := 
sorry

end polynomial_divisibility_l426_426711


namespace find_positive_solutions_l426_426541

noncomputable def satisfies_eq1 (x y : ℝ) : Prop :=
  2 * x - Real.sqrt (x * y) - 4 * Real.sqrt (x / y) + 2 = 0

noncomputable def satisfies_eq2 (x y : ℝ) : Prop :=
  2 * x^2 + x^2 * y^4 = 18 * y^2

theorem find_positive_solutions (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  satisfies_eq1 x y ∧ satisfies_eq2 x y ↔ 
  (x = 2 ∧ y = 2) ∨ 
  (x = Real.sqrt 286^(1/4) / 4 ∧ y = Real.sqrt 286^(1/4)) :=
sorry

end find_positive_solutions_l426_426541


namespace pairs_equality_l426_426386

theorem pairs_equality (K S : Finset ℕ) (hK : K.card = 500) (hS : S.card = 500) 
(hU : K ∪ S = Finset.range 1000) :
  (Finset.card (Finset.filter (λ p, (p.1 - p.2) % 100 = 7) (K.product S))) =
  (Finset.card (Finset.filter (λ p, (p.2 - p.1) % 100 = 7) (S.product K))) :=
sorry

end pairs_equality_l426_426386


namespace group_not_simple_and_contains_normal_subgroup_l426_426675

variables {G : Type*} [group G] [fintype G] [decidable_eq G]
variables (n m : ℕ) (hG : ¬comm_group G) (hodd : odd m) (helem : ∃ g : G, order_of g = 2^n)

theorem group_not_simple_and_contains_normal_subgroup (ht : fintype.card G = 2^n * m) :
  ¬ is_simple_group G ∧ ∃ (N : subgroup G), N.normal ∧ fintype.card N = m :=
sorry

end group_not_simple_and_contains_normal_subgroup_l426_426675


namespace find_num_workers_l426_426812

noncomputable def num_workers (W : ℕ) :=
  ∃ A : ℕ, W * 65 = (W + 10) * 55

theorem find_num_workers : num_workers 55 :=
by
  unfold num_workers
  use 55 * 55
  sorry

end find_num_workers_l426_426812


namespace recipe_total_l426_426763

theorem recipe_total (b f s : ℕ) (h_ratio : b = 1 * s / 5 ∧ f = 8 * s / 5) (h_sugar : s = 10) : b + f + s = 28 :=
by
  let part := s / 5
  have h1 : b = 2 := h_ratio.1.symm.subst (congrArg (fun x => 1 * x) (Nat.mul_div_cancel_left (Nat.succ_pos' 4))).symm
  have h2 : f = 16 := h_ratio.2.symm.subst (congrArg (fun x => 8 * x) (Nat.mul_div_cancel_left (Nat.succ_pos' 4))).symm
  rw [h_sugar, h1, h2]
  rfl

end recipe_total_l426_426763


namespace sum_geometric_seq_eq_l426_426946

-- Defining the parameters of the geometric sequence
def a : ℚ := 1 / 5
def r : ℚ := 2 / 5
def n : ℕ := 8

-- Required to prove the sum of the first eight terms equals the given fraction
theorem sum_geometric_seq_eq :
  (a * (1 - r^n) / (1 - r)) = (390369 / 1171875) :=
by
  -- Proof to be completed
  sorry

end sum_geometric_seq_eq_l426_426946


namespace sum_of_digits_in_T_shape_35_l426_426524

-- Define the set of digits
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the problem variables and conditions
theorem sum_of_digits_in_T_shape_35
  (a b c d e f g h : ℕ)
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
        d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
        e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
        f ≠ g ∧ f ≠ h ∧
        g ≠ h)
  (h2 : a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ 
        e ∈ digits ∧ f ∈ digits ∧ g ∈ digits ∧ h ∈ digits)
  (h3 : a + b + c + d = 26)
  (h4 : e + b + f + g + h = 20) :
  a + b + c + d + e + f + g + h = 35 := by
  sorry

end sum_of_digits_in_T_shape_35_l426_426524


namespace no_21_segments_of_length_2_in_10x10_grid_l426_426188

theorem no_21_segments_of_length_2_in_10x10_grid :
  let grid := (10 × 11 + 10 × 11)
  ∀ segments_of_length_2, 
  segments_of_length_2 = 21 → 
  False :=
by
  let segments_of_length_1 := 220 - 2 * segments_of_length_2
  have constraint : segments_of_length_1 % 2 = 0 :=
    by sorry

  sorry

end no_21_segments_of_length_2_in_10x10_grid_l426_426188


namespace incorrect_option_c_l426_426372

theorem incorrect_option_c (R : ℝ) : 
  let cylinder_lateral_area := 4 * π * R^2
  let sphere_surface_area := 4 * π * R^2
  cylinder_lateral_area = sphere_surface_area :=
  sorry

end incorrect_option_c_l426_426372


namespace fraction_equal_l426_426248

variable {m n p q : ℚ}

-- Define the conditions
def condition1 := (m / n = 20)
def condition2 := (p / n = 5)
def condition3 := (p / q = 1 / 15)

-- State the theorem
theorem fraction_equal (h1 : condition1) (h2 : condition2) (h3 : condition3) : (m / q = 4 / 15) :=
  sorry

end fraction_equal_l426_426248


namespace union_of_sets_l426_426574

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Prove that A ∪ B = {x | -1 < x ∧ x ≤ 2}
theorem union_of_sets (x : ℝ) : x ∈ (A ∪ B) ↔ x ∈ {x | -1 < x ∧ x ≤ 2} :=
by
  sorry

end union_of_sets_l426_426574


namespace alan_total_cost_l426_426865

theorem alan_total_cost :
  let price_AVN := 12
  let price_TheDark := 2 * price_AVN
  let total_cost_TheDark := 2 * price_TheDark
  let cost_90s_CDs := 0.40 * (price_AVN + total_cost_TheDark)
  let total_cost := price_AVN + total_cost_TheDark + cost_90s_CDs
  total_cost = 84 :=
by
  let price_AVN := 12
  let price_TheDark := 2 * price_AVN
  let total_cost_TheDark := 2 * price_TheDark
  let cost_90s_CDs := 0.40 * (price_AVN + total_cost_TheDark)
  let total_cost := price_AVN + total_cost_TheDark + cost_90s_CDs
  have h1 : total_cost = 12 + 48 + 24 := by calculate
  show total_cost = 84, from h1

end alan_total_cost_l426_426865


namespace part_I_part_II_part_III_l426_426599

def f (a : ℝ) (x : ℝ) := 2 * x^3 + 3 * a * x^2 + 1

-- Part (Ⅰ)
theorem part_I (a : ℝ) : (∀ x, x = 1 → (6 * x^2 + 6 * a * x) = 0) → a = -1 :=
by
  intros h
  specialize h 1
  simp at h
  sorry

-- Part (Ⅱ)
theorem part_II (a : ℝ) :
  (if a = 0 then (∀ x, f a x is increasing) else if a > 0 then
    (∀ x, (x < -a ∨ 0 < x → f a x is increasing) ∧ (-a < x < 0 → f a x is decreasing)) else
    (∀ x, (x < 0 ∨ -a < x → f a x is increasing) ∧ (0 < x < -a → f a x is decreasing))) :=
by
  -- translate conditions into monotonicity checks
  sorry

-- Part (Ⅲ)
theorem part_III (a : ℝ) :
  (if a ≥ 0 then ∀ x ∈ set.Icc 0 2, f a x ≥ 1
  else if -2 < a ∧ a < 0 then ∀ x ∈ set.Icc 0 2, f a x ≥ a^3 + 1
  else ∀ x ∈ set.Icc 0 2, f a x ≥ 17 + 12 * a) :=
by
  -- translate conditions into minimum value checks
  sorry

end part_I_part_II_part_III_l426_426599


namespace wenchuan_earthquake_amplitude_l426_426384

-- Define the context and necessary variables
variables {A A₀ : ℝ} {M₈ M₅ : ℝ}

-- Define the condition of the Richter scale formula
def richter_scale (M : ℝ) (A : ℝ) (A₀ : ℝ) : Prop :=
  M = log10 (A / A₀)

-- Define the statement to be proven
theorem wenchuan_earthquake_amplitude
  (M₈ : ℝ := 8) (M₅ : ℝ := 5) (A₁ A₂ : ℝ) (A₀ : ℝ)
  (h₁ : richter_scale M₈ A₁ A₀)
  (h₂ : richter_scale M₅ A₂ A₀) :
  A₁ = 1000 * A₂ :=
by sorry

end wenchuan_earthquake_amplitude_l426_426384


namespace pies_baked_l426_426914

theorem pies_baked (days : ℕ) (eddie_rate : ℕ) (sister_rate : ℕ) (mother_rate : ℕ)
  (H1 : eddie_rate = 3) (H2 : sister_rate = 6) (H3 : mother_rate = 8) (days_eq : days = 7) :
  eddie_rate * days + sister_rate * days + mother_rate * days = 119 :=
by
  sorry

end pies_baked_l426_426914


namespace find_prob_xi_less_78_l426_426391

noncomputable def math_test_score_distribution (σ : ℝ) : ProbabilityDist ℝ :=
  NormalDist.mk 85 σ

axiom given_probabilities : ∀ (σ : ℝ),
  P(83 < ↥(math_test_score_distribution σ)) (↥(math_test_score_distribution σ) < 87) = 0.3 ∧
  P(78 < ↥(math_test_score_distribution σ)) (↥(math_test_score_distribution σ) < 83) = 0.13

theorem find_prob_xi_less_78 {σ : ℝ} :
  P(↥(math_test_score_distribution σ) < 78) = 0.22 :=
sorry

end find_prob_xi_less_78_l426_426391


namespace elevator_initial_floors_down_l426_426489

theorem elevator_initial_floors_down (x : ℕ) (h1 : 9 - x + 3 + 8 = 13) : x = 7 := 
by
  -- Proof
  sorry

end elevator_initial_floors_down_l426_426489


namespace expression_bounds_l426_426320

theorem expression_bounds (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  (√(x^2 + (1 - y)^2) + √(y^2 + (1 - z)^2) + √(z^2 + (1 - w)^2) + √(w^2 + (1 - x)^2)) ∈ set.Icc (2 * √2) 4 :=
by
  sorry

end expression_bounds_l426_426320


namespace points_and_conditions_proof_l426_426707

noncomputable def points_and_conditions (x y : ℝ) : Prop := 
|x - 3| + |y + 5| = 0

noncomputable def min_AM_BM (m : ℝ) : Prop :=
|3 - m| + |-5 - m| = 7 / 4 * |8|

noncomputable def min_PA_PB (p : ℝ) : Prop :=
|p - 3| + |p + 5| = 8

noncomputable def min_PD_PO (p : ℝ) : Prop :=
|p + 1| - |p| = -1

noncomputable def range_of_a (a : ℝ) : Prop :=
a ∈ Set.Icc (-5) (-1)

theorem points_and_conditions_proof (x y : ℝ) (m p a : ℝ) :
  points_and_conditions x y → 
  x = 3 ∧ y = -5 ∧ 
  ((m = -8 ∨ m = 6) → min_AM_BM m) ∧ 
  (min_PA_PB p) ∧ 
  (min_PD_PO p) ∧ 
  (range_of_a a) :=
by 
  sorry

end points_and_conditions_proof_l426_426707


namespace fraction_problem_l426_426250

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end fraction_problem_l426_426250


namespace sum_x_y_eq_l426_426614

noncomputable def equation (x y : ℝ) : Prop :=
  2 * x^2 - 4 * x * y + 4 * y^2 + 6 * x + 9 = 0

theorem sum_x_y_eq (x y : ℝ) (h : equation x y) : x + y = -9 / 2 :=
by sorry

end sum_x_y_eq_l426_426614


namespace find_N_l426_426005

-- Definition of the conditions
def is_largest_divisor_smaller_than (m N : ℕ) : Prop := m < N ∧ Nat.gcd m N = m

def produces_power_of_ten (N m : ℕ) : Prop := ∃ k : ℕ, k > 0 ∧ N + m = 10^k

-- Final statement to prove
theorem find_N (N : ℕ) : (∃ m : ℕ, is_largest_divisor_smaller_than m N ∧ produces_power_of_ten N m) → N = 75 :=
by
  sorry

end find_N_l426_426005


namespace correct_operation_l426_426809

theorem correct_operation {a : ℝ} : (a ^ 6 / a ^ 2 = a ^ 4) :=
by sorry

end correct_operation_l426_426809


namespace height_of_tin_is_approximately_l426_426738

-- Define the cylinder attributes
def diameter : ℝ := 14
def volume : ℝ := 98

-- Define the radius as half of the diameter
def radius : ℝ := diameter / 2

-- Define the height of the cylinder
noncomputable def height : ℝ := volume / (Real.pi * radius^2)

-- The theorem stating the height is approximately 0.6366 cm
theorem height_of_tin_is_approximately :
  (diameter = 14) ∧ (volume = 98) → (radius = 7) → (height ≈ 0.6366) :=
by
  sorry

end height_of_tin_is_approximately_l426_426738


namespace find_f_sqrt_5753_l426_426021

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_sqrt_5753 (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) = x * f y + y * f x)
  (h2 : ∀ x y : ℝ, f (x + y) = f (x * 1993) + f (y * 1993)) :
  f (Real.sqrt 5753) = 0 :=
sorry

end find_f_sqrt_5753_l426_426021


namespace probability_of_winning_is_correct_l426_426019

theorem probability_of_winning_is_correct :
  ∀ (PWin PLoss PTie : ℚ),
    PLoss = 5/12 →
    PTie = 1/6 →
    PWin + PLoss + PTie = 1 →
    PWin = 5/12 := 
by
  intros PWin PLoss PTie hLoss hTie hSum
  sorry

end probability_of_winning_is_correct_l426_426019


namespace gcd_lcm_product_l426_426548

theorem gcd_lcm_product (a b : ℕ) (gcd_ab lcm_ab product : ℕ) : a = 24 → b = 54 → gcd_ab = Nat.gcd a b → lcm_ab = Nat.lcm a b → product = gcd_ab * lcm_ab → product = 1296 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, Nat.gcd_comm, Nat.lcm_comm] at *
  exact h5

end gcd_lcm_product_l426_426548


namespace P_not_77_for_all_integers_l426_426343

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_not_77_for_all_integers (x y : ℤ) : P x y ≠ 77 :=
sorry

end P_not_77_for_all_integers_l426_426343


namespace inscribed_circle_radius_l426_426822

theorem inscribed_circle_radius (A B C : Point) (r : ℝ) 
  (h_angle_BAC : ∠BAC = 120) (h_AB : distance A B = 4) (h_AC : distance A C = 2)
  (h_max_radius : is_inscribed_circle_radius A B C r) :
  r = (3 - Real.sqrt 7) / 2 := sorry

end inscribed_circle_radius_l426_426822


namespace sum_floor_log2_l426_426535

theorem sum_floor_log2 : (∑ N in Finset.range (2048 + 1), nat.floor (Real.log (N : ℝ) / Real.log 2)) = 6157 := sorry

end sum_floor_log2_l426_426535


namespace systematic_sampling_first_two_numbers_l426_426041

theorem systematic_sampling_first_two_numbers (n N : ℕ) (last_num sample_size population_size : ℕ)
  (h_pop_size : population_size = 8000)
  (h_sample_size : sample_size = 50)
  (h_last_num : last_num = 7900)
  (h_n : n = 2)
  (h_N : N = 160) : 
  -- The first two sample numbers are 0159 and 0319.
  let first_sample_num1 := 0159 in
  let first_sample_num2 := 0319 in
  (first_sample_num1, first_sample_num2) = (0159, 0319) :=
by
  sorry

end systematic_sampling_first_two_numbers_l426_426041


namespace min_pos_period_of_sine_func_l426_426014

theorem min_pos_period_of_sine_func : ∃ T > 0, ∀ x, 3 * sin (2 * (x + T) + (π / 4)) = 3 * sin (2 * x + (π / 4)) :=
by
  use π
  sorry

end min_pos_period_of_sine_func_l426_426014


namespace polynomial_roots_l426_426161

open Polynomial

theorem polynomial_roots :
  (roots (X^3 - 2 * X^2 - 5 * X + 6)).toFinset = {1, -2, 3} :=
sorry

end polynomial_roots_l426_426161


namespace proof_question_l426_426441

def main : IO Unit :=
  IO.println s!"Hello, Lean!"

theorem proof_question :
  ∀ (a b c : ℝ),
  a = 6 →
  b = 15 →
  c = 7 →
  a * b * c = (sqrt ((a + 2) * (b + 3))) / (c + 1) →
  a * b * c = 1.5 :=
by
  intros a b c ha hb hc h_eq
  sorry

end proof_question_l426_426441


namespace converse_proposition_proof_l426_426381

variable (a b : ℝ)

-- Defining the conditions and the proposition
def original_proposition : Prop :=
  a = 0 → a * b = 0

def converse_proposition : Prop :=
  a * b = 0 → a = 0

-- The problem statement we need to prove
theorem converse_proposition_proof : original_proposition → converse_proposition :=
by
  intro h
  sorry

end converse_proposition_proof_l426_426381


namespace prove_f_neg1_eq_0_l426_426587

def f : ℝ → ℝ := sorry

theorem prove_f_neg1_eq_0
  (h1 : ∀ x : ℝ, f(x + 2) = f(2 - x))
  (h2 : ∀ x : ℝ, f(1 - 2 * x) = -f(2 * x + 1))
  : f(-1) = 0 := sorry

end prove_f_neg1_eq_0_l426_426587


namespace final_last_digits_after_one_hour_l426_426291

-- Define the initial numbers
def initial_numbers : List ℕ := [1, 2, 4]

-- Define the operation of replacing the list with their pairwise sums
def pairwise_sums (lst : List ℕ) : List ℕ :=
  match lst with
  | [a, b, c] => [a + b, a + c, b + c]
  | _         => lst  -- This should never happen in our context

-- Define a function that performs the pairwise sums repeatedly for a given number of minutes
def iterate_pairwise_sums (minutes : ℕ) (initial : List ℕ) : List ℕ :=
  (List.range minutes).foldl (λ acc _, pairwise_sums acc) initial

-- After 60 minutes
def final_numbers := iterate_pairwise_sums 60 initial_numbers

-- Extract the last digits of each number
def last_digits (lst : List ℕ) : List ℕ :=
  lst.map (λ x => x % 10)

-- The goal is to prove that the last digits are 6, 7, and 9 in any order
theorem final_last_digits_after_one_hour :
  last_digits final_numbers ~ [6, 7, 9] := 
sorry

end final_last_digits_after_one_hour_l426_426291


namespace alloy_mixture_l426_426112

theorem alloy_mixture (x y : ℝ) 
  (h1 : x + y = 1000)
  (h2 : 0.25 * x + 0.50 * y = 450) : 
  x = 200 ∧ y = 800 :=
by
  -- Proof will follow here
  sorry

end alloy_mixture_l426_426112


namespace determinant_problem_l426_426959

theorem determinant_problem (a b c d : ℝ)
  (h : Matrix.det ![![a, b], ![c, d]] = 4) :
  Matrix.det ![![a, 5*a + 3*b], ![c, 5*c + 3*d]] = 12 := by
  sorry

end determinant_problem_l426_426959


namespace problem1_general_solution_l426_426355

variable {x y : ℝ}

theorem problem1_general_solution (C : ℝ) :
  ∃ u : ℝ × ℝ → ℝ, 
    (∀ x y, (2 * y - 3) * (∂u/∂x) (x, y) + (2 * x + 3 * y^2) * (∂u/∂y) (x, y) = 0) ∧
    (∀ x y, u (x, y) = 2 * x * y - 3 * x + y^3) :=
sorry

end problem1_general_solution_l426_426355


namespace find_smaller_number_l426_426829

theorem find_smaller_number (x : ℕ) (h1 : ∃ y, y = 3 * x) (h2 : x + 3 * x = 124) : x = 31 :=
by
  -- Proof will be here
  sorry

end find_smaller_number_l426_426829


namespace calc_result_l426_426128

noncomputable def sqrt57 := 7.550
noncomputable def sin60 := 0.866
noncomputable def calc_expr := 2 * sqrt57 - sin60

theorem calc_result : calc_expr = 14.2 :=
by
  have sqrt57_approx : sqrt57 = 7.550 := by sorry
  have sin60_approx : sin60 = 0.866 := by sorry
  have expr := calc_expr
  show 14.2 from sorry

end calc_result_l426_426128


namespace rainfall_difference_l426_426789

noncomputable def r₁ : ℝ := 26
noncomputable def r₂ : ℝ := 34
noncomputable def r₃ : ℝ := r₂ - 12
noncomputable def avg : ℝ := 140

theorem rainfall_difference : (avg - (r₁ + r₂ + r₃)) = 58 := 
by
  sorry

end rainfall_difference_l426_426789


namespace binary_sum_is_11_l426_426025

-- Define the binary numbers
def b1 : ℕ := 5  -- equivalent to 101 in binary
def b2 : ℕ := 6  -- equivalent to 110 in binary

-- Define the expected sum in decimal
def expected_sum : ℕ := 11

-- The theorem statement
theorem binary_sum_is_11 : b1 + b2 = expected_sum := by
  sorry

end binary_sum_is_11_l426_426025
