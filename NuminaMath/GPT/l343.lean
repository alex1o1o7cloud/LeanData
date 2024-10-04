import Mathlib
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Triangle.Basic
import Mathlib.GraphTheory.ChromaticNumber
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Logic.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution.Bernoulli
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlibimportant_email_count =important_email_count =

namespace solve_inequality_l343_343631

theorem solve_inequality :
  {x : ℝ | (3 * x + 1) * (2 * x - 1) < 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} :=
  sorry

end solve_inequality_l343_343631


namespace Probability_No_Rain_Next_Day_Given_No_Rain_Today_l343_343897
noncomputable def P_R : ℝ := 1/4
noncomputable def P_R_given_R_today : ℝ := 2/3
noncomputable def P_not_R_given_R_today : ℝ := 1/3
noncomputable def P_not_R_given_not_R_today : ℝ := 8/9

theorem Probability_No_Rain_Next_Day_Given_No_Rain_Today :
  ∀ (P_R P_R_given_R_today P_not_R_given_R_today : ℝ),
  P_R = 1/4 →
  P_R_given_R_today = 2/3 →
  P_not_R_given_R_today = 1/3 →
  let P_not_R := 1 - P_R,
      P_R_given_not_R_today := 1 - P_not_R_given_not_R_today in
  (P_R = P_R * P_R_given_R_today + P_not_R * P_R_given_not_R_today) →
  P_not_R_given_not_R_today = 8/9 :=
by
  intros P_R P_R_given_R_today P_not_R_given_R_today hP_R hP_R_given_R_today hP_not_R_given_R_today
  intro P_not_R P_R_given_not_R_today h_prob_eq
  sorry

end Probability_No_Rain_Next_Day_Given_No_Rain_Today_l343_343897


namespace evaluate_expression_l343_343773

theorem evaluate_expression : (125^(1/3 : ℝ)) * (81^(-1/4 : ℝ)) * (32^(1/5 : ℝ)) = (10 / 3 : ℝ) :=
by
  sorry

end evaluate_expression_l343_343773


namespace zero_function_l343_343419

variable (f : ℝ × ℝ × ℝ → ℝ)

theorem zero_function (h : ∀ x y z : ℝ, f (x, y, z) = 2 * f (z, x, y)) : ∀ x y z : ℝ, f (x, y, z) = 0 :=
by
  intros
  sorry

end zero_function_l343_343419


namespace find_radius_l343_343628

noncomputable def radius_wheel (v : ℕ) (omega : ℝ) : ℝ :=
  (v * 100000 / 60) / (omega * 2 * Real.pi)

theorem find_radius
  (v : ℕ) (omega : ℝ)
  (hv : v = 66)
  (homega : omega = 500.4549590536851) : 
  radius_wheel v omega ≈ 34.96 := by
  sorry

end find_radius_l343_343628


namespace gold_coins_l343_343256

theorem gold_coins (n c : Nat) : 
  n = 9 * (c - 2) → n = 6 * c + 3 → n = 45 :=
by 
  intros h1 h2 
  sorry

end gold_coins_l343_343256


namespace common_difference_is_three_halves_l343_343949

noncomputable def common_difference_arith_seq (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : p ≠ r) (h4 : p > 0) (h5 : q > 0) (h6 : r > 0) (h7 : q^2 = p * r) : ℝ :=
  (2 - (log r p)^2 - log r p) / (log r p + 1)

theorem common_difference_is_three_halves (p q r : ℝ) (h1 : p ≠ q) (h2 : q ≠ r) (h3 : p ≠ r) (h4 : p > 0) (h5 : q > 0) (h6 : r > 0) (h7 : q^2 = p * r) (h8 : (log r p) * (2 * (log p (sqrt (p * r))) + (log_q r)) = 6 - (log r p) * (log_q r)) : 
  common_difference_arith_seq p q r h1 h2 h3 h4 h5 h6 h7 = 3 / 2 := 
sorry

end common_difference_is_three_halves_l343_343949


namespace vasya_wins_strategy_l343_343056

/-!
  Given a grid of size 2020 × 2021. Petya and Vasya take turns placing chips 
  in free cells of the grid. Petya goes first. A player wins if after their move,
  every 4 × 4 square on the board contains at least one chip. Prove that Vasya
  can guarantee themselves a victory regardless of the opponent's moves.
-/

theorem vasya_wins_strategy :
  ∀ (grid : ℕ × ℕ) (initial_turn : bool) (win_condition : ℕ × ℕ → Prop),
  grid = (2020, 2021) →
  initial_turn = tt →  -- Petya goes first (true means Petya's turn, false means Vasya's turn)
  (∀ x y, win_condition (x, y) ↔ (x <= 2016 ∧ y <= 2017)) →
  ∃ strategy : (ℕ × ℕ) → ℕ × ℕ,
    ∀ current_position : ℕ × ℕ,
      (win_condition current_position → strategy current_position = (0, 0)) ∨
      strategy current_position ≠ (0, 0) →
    strategy = (λ p, p) :=
begin
  -- The actual proof omitted
  sorry
end

end vasya_wins_strategy_l343_343056


namespace evaluate_expression_l343_343777

theorem evaluate_expression : (125^(1/3 : ℝ)) * (81^(-1/4 : ℝ)) * (32^(1/5 : ℝ)) = (10 / 3 : ℝ) :=
by
  sorry

end evaluate_expression_l343_343777


namespace tan_double_sum_l343_343446

variable (α : ℝ)

noncomputable def tan_sum := Real.tan (α + Real.pi / 6) = 2

theorem tan_double_sum : Real.tan (2 * α + 7 * Real.pi / 12) = -1 / 7 :=
by
  have h1 : Real.tan (α + Real.pi / 6) = 2 := tan_sum α
  sorry

end tan_double_sum_l343_343446


namespace otimes_value_l343_343809

theorem otimes_value :
  ∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0
  ∧ y - z ≠ 0 ∧ z - x ≠ 0 ∧ x - y ≠ 0
  ∧ let otimes (x y z : ℝ) := x / (y - z)
    in let a := otimes 2 5 3
       ∧ let b := otimes 5 2 3
       ∧ let c := otimes 4 3 2
         in otimes (3 * a) b c = -1 / 3 := by
  sorry

end otimes_value_l343_343809


namespace sin_alpha_value_l343_343469

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : sin α + cos α = 1 / 5) : sin α = 4 / 5 :=
by
  sorry

end sin_alpha_value_l343_343469


namespace common_arithmetic_sequence_term_l343_343381

theorem common_arithmetic_sequence_term :
  let first_sequence (n : ℕ) := 3 + 8 * n
  let second_sequence (m : ℕ) := 5 + 9 * m
  ∃ (a : ℕ), (∃ n, a = first_sequence n) ∧ (∃ m, a = second_sequence m) ∧ 1 ≤ a ∧ a ≤ 150 ∧ ∀ b, (∃ n, b = first_sequence n) ∧ (∃ m, b = second_sequence m) ∧ 1 ≤ b ∧ b ≤ 150 → b ≤ a :=
  ∃ (a : ℕ), (∃ n, a = 3 + 8 * n) ∧ (∃ m, a = 5 + 9 * m) ∧ 1 ≤ a ∧ a ≤ 150 ∧ ∀ b, (∃ n, b = 3 + 8 * n) ∧ (∃ m, b = 5 + 9 * m) ∧ 1 ≤ b ∧ b ≤ 150 → b ≤ 131 :=
sorry

end common_arithmetic_sequence_term_l343_343381


namespace previous_income_l343_343967

-- Define the conditions as Lean definitions
variables (p : ℝ) -- Mrs. Snyder's previous monthly income

-- Condition 1: Mrs. Snyder used to spend 40% of her income on rent and utilities
def rent_and_utilities_initial (p : ℝ) : ℝ := (2 * p) / 5

-- Condition 2: Her salary was increased by $600
def new_income (p : ℝ) : ℝ := p + 600

-- Condition 3: After the increase, rent and utilities account for 25% of her new income
def rent_and_utilities_new (p : ℝ) : ℝ := (new_income p) / 4

-- Theorem: Proving that Mrs. Snyder's previous monthly income was $1000
theorem previous_income : (2 * p) / 5 = (new_income p) / 4 → p = 1000 :=
begin
  -- By mathlib, sorry as placeholder for proof
  sorry
end

end previous_income_l343_343967


namespace tan_arith_geom_l343_343854

noncomputable def a (n : ℕ) : ℝ := sorry  -- Define the arithmetic sequence
noncomputable def b (n : ℕ) : ℝ := sorry  -- Define the geometric sequence

axiom arith_seq (a_1001 a_1015 : ℝ) : a(1001) + a(1015) = π
axiom geom_seq_6_9 (b_6 b_9 : ℝ) : b(6) * b(9) = 2

theorem tan_arith_geom : 
  (a(1) + a(2015) = π) → 
  (b(7) = real.sqrt 2) → 
  (b(8) = real.sqrt 2) →
  (1 + b(7) * b(8) = 3) →
  tan (π / 3) = real.sqrt 3 :=
by sorry

end tan_arith_geom_l343_343854


namespace library_hospital_community_center_bells_ring_together_l343_343698

theorem library_hospital_community_center_bells_ring_together :
  ∀ (library hospital community : ℕ), 
    (library = 18) → (hospital = 24) → (community = 30) → 
    (∀ t, (t = 0) ∨ (∃ n₁ n₂ n₃ : ℕ, 
      t = n₁ * library ∧ t = n₂ * hospital ∧ t = n₃ * community)) → 
    true :=
by
  intros
  sorry

end library_hospital_community_center_bells_ring_together_l343_343698


namespace sqrt_288_eq_12_sqrt_2_l343_343254

theorem sqrt_288_eq_12_sqrt_2 : Real.sqrt 288 = 12 * Real.sqrt 2 :=
by
sorry

end sqrt_288_eq_12_sqrt_2_l343_343254


namespace max_min_value_function_l343_343796

noncomputable def given_function (x : ℝ) : ℝ :=
  (Real.sin x) ^ 2 + Real.cos x + 1

theorem max_min_value_function :
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≤ 9 / 4) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 9 / 4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 2) := by
  sorry

end max_min_value_function_l343_343796


namespace base_8_add_sub_l343_343033

-- Definitions of the numbers in base 8
def n1 : ℕ := 4 * 8^2 + 5 * 8^1 + 1 * 8^0
def n2 : ℕ := 1 * 8^2 + 6 * 8^1 + 2 * 8^0
def n3 : ℕ := 1 * 8^2 + 2 * 8^1 + 3 * 8^0

-- Convert the result to base 8
def to_base_8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let rem1 := n % 64
  let d1 := rem1 / 8
  let d0 := rem1 % 8
  d2 * 100 + d1 * 10 + d0

-- Proof statement
theorem base_8_add_sub :
  to_base_8 ((n1 + n2) - n3) = to_base_8 (5 * 8^2 + 1 * 8^1 + 0 * 8^0) :=
by
  sorry

end base_8_add_sub_l343_343033


namespace area_of_triangle_ABC_l343_343964

noncomputable def area_of_ABC : ℝ :=
  let BE := 10 in
  let AD := 2 * BE in
  let AG := (2 / 3) * AD in
  let BG := (2 / 3) * BE in
  let area_ABG := 1 / 2 * AG * BG in
  6 * area_ABG

theorem area_of_triangle_ABC :
  let BE := 10 in
  let AD := 2 * BE in
  (AD = 20) →
  (BE = 10) →
  (AD / 2 = BE) →
  ∃ (area : ℝ), area = area_of_ABC ∧ area = 2400 / 9 :=
by
  intro BE AD AD_eq_20 BE_eq_10 AD_BE_relation
  suffices area_of_ABC = 2400 / 9 by
  use area_of_ABC
  split
  exact rfl
  exact this
  -- Proceed with complete proof (omitted)
  sorry

end area_of_triangle_ABC_l343_343964


namespace equal_angles_CSM_CSN_l343_343235

-- Given definitions and conditions
variables {A B C M N S : Type} [Loc : LinearOrderedField ℝ]
variables (triangle_ABC : Triangle Loc A B C) (M N : Loc)
variables (AN_eq_AC : ∥A - N∥ = ∥A - C∥) (BM_eq_BC : ∥B - M∥ = ∥B - C∥)
variables (M_on_AB : PointOnLineSegment Loc M (LineSegment Loc A B))
variables (N_on_AB : PointOnLineSegment Loc N (LineSegment Loc A B))
variables (S : Loc)
variables (parallel_M_to_BC : Parallel Loc (LineThroughPoints Loc M S) (LineThroughPoints Loc B C))
variables (parallel_N_to_AC : Parallel Loc (LineThroughPoints Loc N S) (LineThroughPoints Loc A C))

-- Theorem to prove angle between S, C, M, and S, C, N are equal
theorem equal_angles_CSM_CSN : ∠ (LineThroughPoints Loc C S) (LineThroughPoints Loc S M) = ∠ (LineThroughPoints Loc C S) (LineThroughPoints Loc S N) := 
sorry

end equal_angles_CSM_CSN_l343_343235


namespace sum_of_fractions_is_correct_l343_343313

-- Definitions from the conditions
def half_of_third := (1 : ℚ) / 2 * (1 : ℚ) / 3
def third_of_quarter := (1 : ℚ) / 3 * (1 : ℚ) / 4
def quarter_of_fifth := (1 : ℚ) / 4 * (1 : ℚ) / 5
def sum_fractions := half_of_third + third_of_quarter + quarter_of_fifth

-- The theorem to prove
theorem sum_of_fractions_is_correct : sum_fractions = (3 : ℚ) / 10 := by
  sorry

end sum_of_fractions_is_correct_l343_343313


namespace original_price_of_article_l343_343330

theorem original_price_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 := 
by 
  sorry

end original_price_of_article_l343_343330


namespace total_water_capacity_of_coolers_l343_343189

theorem total_water_capacity_of_coolers :
  ∀ (first_cooler second_cooler third_cooler : ℕ), 
  first_cooler = 100 ∧ 
  second_cooler = first_cooler + first_cooler / 2 ∧ 
  third_cooler = second_cooler / 2 -> 
  first_cooler + second_cooler + third_cooler = 325 := 
by
  intros first_cooler second_cooler third_cooler H
  cases' H with H1 H2
  cases' H2 with H3 H4
  sorry

end total_water_capacity_of_coolers_l343_343189


namespace triangle_XYZ_XZ_l343_343924

theorem triangle_XYZ_XZ :
  ∀ (X Y Z : Type) [IsTriangle X Y Z],
      angle X = 60 ∧ angle Y = 75 ∧ side Y Z = 6 →
      side X Z = 3 * Real.sqrt 2 + Real.sqrt 6 :=
begin
  intros X Y Z h,
  simp at h,
  sorry
end

end triangle_XYZ_XZ_l343_343924


namespace equation_of_latus_rectum_l343_343430

theorem equation_of_latus_rectum (p : ℝ) (h : 4 * p = 2) : x = -p := 
by {
  have hp : p = 1/2,
  { linarith },
  rw hp,
  exact rfl,
}

end equation_of_latus_rectum_l343_343430


namespace time_to_cross_man_l343_343501

-- Conditions
def length_train : ℝ := 450  -- in meters
def speed_train : ℝ := 60 * 1000 / 3600  -- in meters per second
def speed_man : ℝ := 6 * 1000 / 3600  -- in meters per second

-- Proof statement
theorem time_to_cross_man : (length_train / (speed_train - speed_man)) = 30 :=
by
  sorry

end time_to_cross_man_l343_343501


namespace find_wall_length_l343_343716

def wall_length (side_length_mirror : ℕ) (width_wall : ℕ) : ℕ :=
  let area_mirror := side_length_mirror * side_length_mirror
  let area_wall := 2 * area_mirror
  let length_wall_float := (area_wall : ℝ) / (width_wall : ℝ)
  (Real.to_nat length_wall_float).toNat

theorem find_wall_length (side_length_mirror : ℕ) (width_wall : ℕ) (h1 : side_length_mirror = 54) (h2 : width_wall = 68) :
  wall_length side_length_mirror width_wall = 86 :=
sorry

end find_wall_length_l343_343716


namespace functional_equation_solution_l343_343420

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (hf : ∀ x y : ℝ, f x * f y = f (x^2 + y^2)) :
  (f = λ x, 0) ∨ (f = λ x, 1) ∨ (f = λ x, if x = 0 then 1 else 0) := sorry

end functional_equation_solution_l343_343420


namespace area_of_oblique_drawing_is_sqrt_2_l343_343522

-- Defining the given side length of the square
def side_length : ℝ := 2

-- Calculate the area of the original square
def area_original_square : ℝ := side_length * side_length

-- Define the area of the intuitive diagram using the given formula
def area_intuitive_diagram : ℝ := (Real.sqrt 2 / 4) * area_original_square

-- The theorem that we need to prove
theorem area_of_oblique_drawing_is_sqrt_2 :
  area_intuitive_diagram = Real.sqrt 2 :=
by sorry

end area_of_oblique_drawing_is_sqrt_2_l343_343522


namespace arithmetic_seq_geometric_seq_l343_343834

theorem arithmetic_seq_geometric_seq (a : ℕ → ℕ) (S : ℕ → ℕ) (d a1 ak Sk2 : ℕ) (k : ℕ) 
  (h1 : a_1 + a_3 = 8)
  (h2 : a_2 + a_4 = 12)
  (h3 : ∀ n, a n = a1 + (n - 1) * d)
  (h4 : ∀ n, S n = n * (2 * a1 + (n - 1) * d) / 2)
  (h5 : a1 = a_0)
  (h6 : ak = a k)
  (h7 : Sk2 = S (k + 2))
  (h8 : a1, ak, Sk2 form a geometric sequence) :
  k = 6 := sorry

end arithmetic_seq_geometric_seq_l343_343834


namespace min_perimeter_triangle_AED_l343_343458

theorem min_perimeter_triangle_AED :
  let V : Type := ℝ
  let A B C D E : V := sorry
  let VA : ℝ := 8
  let VB : ℝ := 8
  let VC : ℝ := 8
  let BC : ℝ := 4
  let perimeter_triangle (x y z : ℝ) := x + y + z
  -- Assuming we have the triangular prism and valid positions for A, B, C, D, E
  ∃ (D E : V), (D ≠ E) ∧ (perimeter_triangle (dist V.to_euclidean x D) (dist V.to_euclidean y E) (dist V.to_euclidean D E) = 11) :=
sorry

end min_perimeter_triangle_AED_l343_343458


namespace calculate_v2_using_horner_method_l343_343050

def f (x : ℕ) : ℕ := x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1

def horner_step (x b a : ℕ) := a * x + b

def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
coeffs.foldr (horner_step x) 0

theorem calculate_v2_using_horner_method :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  -- This is the theorem statement, the proof is not required as per instructions
  sorry

end calculate_v2_using_horner_method_l343_343050


namespace product_of_values_subtracted_from_65_l343_343801

theorem product_of_values_subtracted_from_65 : 
  (let t := [7, -7] in (65 - (7 * -7)) = 114) :=
by
  let t := [7, -7]
  have h1 : 7 * -7 = -49 := by sorry
  have h2 : 65 - (-49) = 114 := by sorry
  exact h2

end product_of_values_subtracted_from_65_l343_343801


namespace combined_payment_is_correct_l343_343329

-- Define the conditions for discounts
def discount_scheme (amount : ℕ) : ℕ :=
  if amount ≤ 100 then amount
  else if amount ≤ 300 then (amount * 90) / 100
  else (amount * 80) / 100

-- Given conditions for Wang Bo's purchases
def first_purchase := 80
def second_purchase_with_discount_applied := 252

-- Two possible original amounts for the second purchase
def possible_second_purchases : Set ℕ :=
  { x | discount_scheme x = second_purchase_with_discount_applied }

-- Total amount to be considered for combined buys with discounts
def total_amount_paid := {x + first_purchase | x ∈ possible_second_purchases}

-- discount applied on the combined amount
def discount_applied_amount (combined : ℕ) : ℕ :=
  discount_scheme combined

-- Prove the combined amount is either 288 or 316
theorem combined_payment_is_correct :
  ∃ combined ∈ total_amount_paid, discount_applied_amount combined = 288 ∨ discount_applied_amount combined = 316 :=
sorry

end combined_payment_is_correct_l343_343329


namespace area_of_circle_l343_343407

def circle_area (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 18 * y = -45

theorem area_of_circle :
  (∃ x y : ℝ, circle_area x y) → ∃ A : ℝ, A = 52 * Real.pi :=
by
  sorry

end area_of_circle_l343_343407


namespace geometric_sequence_problem_l343_343078

section 
variables (a : ℕ → ℝ) (r : ℝ) 

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n : ℕ, a (n + 1) = a n * r

-- Condition: a_4 + a_6 = 8
axiom a4_a6_sum : a 4 + a 6 = 8

-- Mathematical equivalent proof problem
theorem geometric_sequence_problem (h : is_geometric_sequence a r) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
sorry

end

end geometric_sequence_problem_l343_343078


namespace variance_is_stability_measure_l343_343645

def stability_measure (yields : Fin 10 → ℝ) : Prop :=
  let mean := (yields 0 + yields 1 + yields 2 + yields 3 + yields 4 + yields 5 + yields 6 + yields 7 + yields 8 + yields 9) / 10
  let variance := 
    ((yields 0 - mean)^2 + (yields 1 - mean)^2 + (yields 2 - mean)^2 + (yields 3 - mean)^2 + 
     (yields 4 - mean)^2 + (yields 5 - mean)^2 + (yields 6 - mean)^2 + (yields 7 - mean)^2 + 
     (yields 8 - mean)^2 + (yields 9 - mean)^2) / 10
  true -- just a placeholder, would normally state that this is the appropriate measure

theorem variance_is_stability_measure (yields : Fin 10 → ℝ) : stability_measure yields :=
by 
  sorry

end variance_is_stability_measure_l343_343645


namespace evaluate_f_f_neg2_l343_343483

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x else x^2

theorem evaluate_f_f_neg2 : f (f (-2)) = 4 := by
  sorry

end evaluate_f_f_neg2_l343_343483


namespace find_z_l343_343889

-- Define the imaginary unit 'i'.
def i : ℂ := complex.I

-- Define the complex number 'z'.
def z : ℂ := 3 - 2 * i

-- Given condition
def given_condition : Prop := i * z = 2 + 3 * i

-- Theorem statement
theorem find_z (h : given_condition) : z = 3 - 2 * i :=
by {
  -- Proof goes here
  sorry
}

end find_z_l343_343889


namespace find_m_l343_343048

-- Define variables and conditions
variables {θ m : ℝ}
def sin_θ := (m - 3) / (m + 5)
def cos_θ := (4 - 2m) / (m + 5)

-- Main problem statement to prove
theorem find_m : sin_θ ^ 2 + cos_θ ^ 2 = 1 → m = 0 ∨ m = 8 := by sorry

end find_m_l343_343048


namespace bisectors_intersect_on_circle_l343_343986

noncomputable def point (α β : Type) := sorry
noncomputable def line (α : Type) := sorry
noncomputable def circle (α : Type) := sorry

variables {α : Type} {A B C S : point α} {Γ : circle α}

-- Assume the existence of a triangle ABC
axiom triangle (A B C : point α) : Prop

-- Assume S is the intersection of the perpendicular bisector of [BC] and the circle Γ
axiom perpendicular_bisector_intersect_circle 
  (A B C S : point α) (Γ : circle α) :
  triangle A B C → Prop

-- Assume S lies on the angle bisector of ∠BAC
axiom angle_bisector 
  (A B C S : point α) :
  triangle A B C → Prop

theorem bisectors_intersect_on_circle 
  (A B C S : point α) (Γ : circle α) :
  triangle A B C →
  perpendicular_bisector_intersect_circle A B C S Γ →
  angle_bisector A B C S :=
sorry

end bisectors_intersect_on_circle_l343_343986


namespace failed_candidates_percentage_is_70_point_2_l343_343529

def percentage_failed (n_total n_girls : ℕ) (pct_boys_passed pct_girls_passed : ℝ) : ℝ :=
  let n_boys := n_total - n_girls in
  let boys_passed := pct_boys_passed * n_boys / 100 in
  let girls_passed := pct_girls_passed * n_girls / 100 in
  let total_passed := boys_passed + girls_passed in
  let total_failed := n_total - total_passed in
  (total_failed / n_total) * 100

theorem failed_candidates_percentage_is_70_point_2 :
  percentage_failed 2000 900 28 32 = 70.2 :=
by
  sorry

end failed_candidates_percentage_is_70_point_2_l343_343529


namespace transport_equivalence_l343_343147

theorem transport_equivalence (f : ℤ → ℤ) (x y : ℤ) (h : f x = -x) :
  f (-y) = y :=
by
  sorry

end transport_equivalence_l343_343147


namespace correct_statement_D_l343_343059

variables {l : Line} {alpha beta : Plane}

-- Conditions
axiom parallel_l_alpha : l ∥ alpha
axiom perpendicular_l_beta : l ⟂ beta

-- Theorem statement
theorem correct_statement_D : alpha ⟂ beta :=
sorry

end correct_statement_D_l343_343059


namespace blue_candy_count_l343_343296

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_candy_count :
  blue_pieces = 3264 := by
  sorry

end blue_candy_count_l343_343296


namespace eleventh_number_in_list_l343_343726

-- Define a predicate to check if the sum of the digits of a number is 12
def digit_sum_eq_12 (n : ℕ) : Prop := 
  (n.digits 10).sum = 12

-- Define a list of all positive integers whose digits sum to 12 and are in increasing order
def list_of_digit_sum_12 : List ℕ := 
  List.filter digit_sum_eq_12 (List.range 1000) -- considering numbers from 0 to 999 for instance

theorem eleventh_number_in_list : 
  list_of_digit_sum_12.nth 10 = some 147 :=
by
  sorry

end eleventh_number_in_list_l343_343726


namespace midpoints_collinear_l343_343605

open EuclideanGeometry

theorem midpoints_collinear
  {A B C H E1 F1 E2 F2 E3 F3 G1 G2 G3 : Point}
  (h1 : orthocenter A B C H)
  (h2 : perpendicular_from H E1 F1)
  (h3 : perpendicular_from H E2 F2)
  (h4 : perpendicular_from H E3 F3)
  (h5 : midpoint E1 F1 G1)
  (h6 : midpoint E2 F2 G2)
  (h7 : midpoint E3 F3 G3)
  : collinear G1 G2 G3 := 
sorry

end midpoints_collinear_l343_343605


namespace tan_alpha_eq_l343_343607

theorem tan_alpha_eq:
  (∀ (O A B C D E : Type) (n : ℝ),
     diameter AB
     ∧ equilateral_triangle O A B C
     ∧ (AD = 2 / n * AB : Prop)
     ∧ (CD_intersects_O_at_E : Prop)
     ∧ (alpha = ∠AOE : Prop))
  → (tan alpha = (sqrt (n^2 + 16 * n - 32) - n) / (n - 4) * sqrt(3) / 2) :=
  by
  assume O A B C D E n diameter_def equilateral_triangle_def AD_def CD_intersects_O_at_E_def alpha_def,
  sorry

end tan_alpha_eq_l343_343607


namespace domain_of_f_l343_343611

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (1 - x) + (1 / sqrt (1 + x))

theorem domain_of_f :
  ∀ x : ℝ, (1 - x ≥ 0) ∧ (1 + x > 0) ↔ (-1 < x ∧ x ≤ 1) :=
by
  intro x
  split
  · intro h
    cases h with h1 h2
    split
    · have : x > -1 := by linarith
      assumption
    · have : x ≤ 1 := by linarith
      assumption
  · intro h
    cases h with h1 h2
    split
    · linarith
    · linarith

end domain_of_f_l343_343611


namespace initial_amount_correct_l343_343386

noncomputable def initial_amount (A R T : ℝ) : ℝ :=
  A / (1 + (R * T) / 100)

theorem initial_amount_correct :
  initial_amount 2000 3.571428571428571 4 = 1750 :=
by
  sorry

end initial_amount_correct_l343_343386


namespace necessary_and_sufficient_condition_l343_343072

-- Define the conditions and question in Lean 4
variable (a : ℝ) 

-- State the theorem based on the conditions and the correct answer
theorem necessary_and_sufficient_condition :
  (a > 0) ↔ (
    let z := (⟨-a, -5⟩ : ℂ)
    ∃ (x y : ℝ), (z = x + y * I) ∧ x < 0 ∧ y < 0
  ) := by
  sorry

end necessary_and_sufficient_condition_l343_343072


namespace females_with_advanced_degrees_l343_343525

theorem females_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_advanced_degrees : ℕ)
  (males_college_degree_only : ℕ)
  (h1 : total_employees = 200)
  (h2 : total_females = 120)
  (h3 : total_advanced_degrees = 100)
  (h4 : males_college_degree_only = 40) :
  (total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60) :=
by
  -- proof will go here
  sorry

end females_with_advanced_degrees_l343_343525


namespace area_of_triangle_QCA_correct_l343_343757

-- Definitions of points and conditions
def Q := (0, 12)
def A := (3, 12)
def C (p : ℝ) := (0, p)
def right_angle_at_C := (0 : ℝ, 0) -- placeholder definition for right angle at C

-- Definition of the area of triangle QCA
def area_of_triangle_QCA (p : ℝ) : ℝ :=
  (36 - 3 * p) / 2

-- Theorem to prove
theorem area_of_triangle_QCA_correct (p : ℝ) : 
  ∃ (Q A C : ℝ × ℝ), 
    Q = (0, 12) ∧
    A = (3, 12) ∧
    C = (0, p) ∧
    right_angle_at_C →
    area_of_triangle_QCA p = (36 - 3 * p) / 2 :=
by
  sorry

end area_of_triangle_QCA_correct_l343_343757


namespace matrix_pow_three_l343_343395

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_pow_three :
  A^3 = !![-4, 2; -2, 1] := by
  sorry

end matrix_pow_three_l343_343395


namespace ants_of_species_X_on_day_6_l343_343895

/-- Given the initial populations of Species X and Species Y and their growth rates,
    prove the number of Species X ants on Day 6. -/
theorem ants_of_species_X_on_day_6 
  (x y : ℕ)  -- Number of Species X and Y ants on Day 0
  (h1 : x + y = 40)  -- Total number of ants on Day 0
  (h2 : 64 * x + 4096 * y = 21050)  -- Total number of ants on Day 6
  :
  64 * x = 2304 := 
sorry

end ants_of_species_X_on_day_6_l343_343895


namespace volume_computation_l343_343650

noncomputable def volume_of_remaining_region (r1 r2 r_hole : ℝ) : ℝ :=
  let V_larger := (4 / 3) * Real.pi * r2^3 in
  let V_smaller := (4 / 3) * Real.pi * r1^3 in
  let V_between := V_larger - V_smaller in
  let length_cylinder := 2 * r2 in
  let V_cylinder := Real.pi * r_hole^2 * length_cylinder in
  V_between - V_cylinder

theorem volume_computation :
  volume_of_remaining_region 5 10 2 = (3260 / 3) * Real.pi :=
by
  sorry

end volume_computation_l343_343650


namespace cistern_width_l343_343691

theorem cistern_width (length height wet_area: ℝ) (h₁: length = 9) 
                      (h₂: height = 2.25) (h₃: wet_area = 121.5):
                      ∃ (w: ℝ), w = 6 :=
by
  -- Expressions for the areas
  let bottom_area := length * w
  let longer_sides_area := 2 * (length * height)
  let shorter_sides_area := 2 * (w * height)
  
  -- Equation for the total wet surface area
  have eqn : wet_area = bottom_area + longer_sides_area + shorter_sides_area,
  {
    rw [h₁, h₂],
    -- Substitute known values
    refine h₃,
  },
  -- Combine and isolate w in the equation and prove w = 6
  sorry

end cistern_width_l343_343691


namespace company_percentage_increase_l343_343751

theorem company_percentage_increase (employees_jan employees_dec : ℝ) (P_increase : ℝ) 
  (h_jan : employees_jan = 391.304347826087)
  (h_dec : employees_dec = 450)
  (h_P : P_increase = 15) : 
  (employees_dec - employees_jan) / employees_jan * 100 = P_increase :=
by 
  sorry

end company_percentage_increase_l343_343751


namespace midpoints_distance_l343_343973

theorem midpoints_distance
  (A B C D M N : ℝ)
  (h1 : M = (A + C) / 2)
  (h2 : N = (B + D) / 2)
  (h3 : D - A = 68)
  (h4 : C - B = 26)
  : abs (M - N) = 21 := 
sorry

end midpoints_distance_l343_343973


namespace parabola_focus_l343_343025

theorem parabola_focus :
  ∃ f, (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (-f + 1/4))^2)) ∧ f = 1/8 :=
by
  sorry

end parabola_focus_l343_343025


namespace all_isosceles_and_similar_l343_343212

noncomputable section

open Real

def S := { (A B C P Q R r : ℝ) // 
  r > 0 ∧
  P > 0 ∧ Q > 0 ∧ R > 0 ∧
  5 * (1 / P + 1 / Q + 1 / R) - 3 / min P (min Q R) = 6 / r }

theorem all_isosceles_and_similar (A B C P Q R r : ℝ) (h : (A, B, C, P, Q, R, r) ∈ S) :
  (is_isosceles A B C ∧ is_similar A B C) :=
sorry

end all_isosceles_and_similar_l343_343212


namespace john_pool_cleanings_per_month_l343_343930

noncomputable def tip_percent : ℝ := 0.10
noncomputable def cost_per_cleaning : ℝ := 150
noncomputable def total_cost_per_cleaning : ℝ := cost_per_cleaning + (tip_percent * cost_per_cleaning)
noncomputable def chemical_cost_bi_monthly : ℝ := 200
noncomputable def monthly_chemical_cost : ℝ := 2 * chemical_cost_bi_monthly
noncomputable def total_monthly_pool_cost : ℝ := 2050
noncomputable def total_cleaning_cost : ℝ := total_monthly_pool_cost - monthly_chemical_cost

theorem john_pool_cleanings_per_month : total_cleaning_cost / total_cost_per_cleaning = 10 := by
  sorry

end john_pool_cleanings_per_month_l343_343930


namespace parabola_focus_l343_343028

theorem parabola_focus : ∃ f : ℝ, 
  (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (1/4 + f))^2)) ∧
  f = 1/8 := 
by
  sorry

end parabola_focus_l343_343028


namespace fourth_number_in_15th_row_of_pascals_triangle_l343_343539

-- Here we state and prove the theorem about the fourth entry in the 15th row of Pascal's Triangle.
theorem fourth_number_in_15th_row_of_pascals_triangle : 
    (nat.choose 15 3) = 455 := 
by 
    sorry -- Proof is omitted as per instructions

end fourth_number_in_15th_row_of_pascals_triangle_l343_343539


namespace fill_time_l343_343173

theorem fill_time (start_time : ℕ) (initial_rainfall : ℕ) (subsequent_rainfall_rate : ℕ) (subsequent_rainfall_duration : ℕ) (final_rainfall_rate : ℕ) (tank_height : ℕ) :
  start_time = 13 ∧
  initial_rainfall = 2 ∧
  subsequent_rainfall_rate = 1 ∧
  subsequent_rainfall_duration = 4 ∧
  final_rainfall_rate = 3 ∧
  tank_height = 18 →
  13 + 1 + subsequent_rainfall_duration = 18 - 6 / final_rainfall_rate + 18 - tank_height * 10 :=
begin
  sorry
end

end fill_time_l343_343173


namespace angle_DCE_invariant_l343_343956

noncomputable def semicircle_diameter (A B : Point) : Set Point := sorry
noncomputable def normal_line (P : Point, AB : Line) : Line := sorry
noncomputable def point_in_segment (A B P : Point) : Prop := sorry

noncomputable def inscribed_circle (segmented_area : Set Point) : Circle := sorry
noncomputable def tangent_to_line (c : Circle) (l : Line) : Prop := sorry
noncomputable def point_on_circle (P : Point) (c : Circle) : Prop := sorry

theorem angle_DCE_invariant
  (A B : Point)
  (h : semicircle_diameter A B)
  (P : Point)
  (C : Point)
  (PC : Line)
  (D E : Point)
  (circle_left circle_right : Circle) :
  point_in_segment A B P →
  point_on_circle C h →
  normal_line P (Line.mk A B) = Line.mk P C →
  tangent_to_line circle_left (Line.mk A B) →
  tangent_to_line circle_right (Line.mk A B) →
  tangent_to_line circle_left (Line.mk P C) →
  tangent_to_line circle_right (Line.mk P C) →
  ∀ P1, point_in_segment A B P1 → ∃ D1 E1, ∠(DCE) = ∠(D1 C E1) := 
sorry

end angle_DCE_invariant_l343_343956


namespace base_five_product_l343_343654

namespace BaseFiveMultiplication

def to_base_five (n : ℕ) : list ℕ := 
  -- Dummy implementation for illustration
  [n % 5, (n / 5) % 5, (n / 25) % 5, (n / 125) % 5].reverse

def from_base_five (digits : list ℕ) : ℕ := 
  digits.foldl (λ acc d => acc * 5 + d) 0

def base_five_mult (a b : list ℕ) : list ℕ :=
  to_base_five (from_base_five(a) * from_base_five(b))

theorem base_five_product : base_five_mult (to_base_five 132) (to_base_five 12) = to_base_five 2114 :=
  sorry

end BaseFiveMultiplication

end base_five_product_l343_343654


namespace wanda_blocks_total_l343_343305

theorem wanda_blocks_total : 
  let initial_blocks := 4 in
  let additional_blocks := 79 in
  initial_blocks + additional_blocks = 83 := 
by
  sorry

end wanda_blocks_total_l343_343305


namespace find_inverse_value_l343_343487

noncomputable theory

def f (x : ℤ) : ℤ :=
  if x = -1 then 2
  else if x = 4 then 15
  else if x = 7 then 10
  else if x = 10 then 4
  else if x = 14 then 0
  else if x = 20 then 9
  else 0 -- This case handles out of table values, could be -1 instead.

def f_inv (y : ℤ) : ℤ :=
  if y = 2 then -1
  else if y = 15 then 4
  else if y = 10 then 7
  else if y = 4 then 10
  else if y = 0 then 14
  else if y = 9 then 20
  else 0 -- Invalid case handling for reverse values.

theorem find_inverse_value :
  f_inv ((20 + 2 * 4) / 10) = -1 :=
by
  sorry

end find_inverse_value_l343_343487


namespace probability_three_le_f_x_le_fifteen_l343_343485

noncomputable def f (x : ℝ) := 2^x - 1

theorem probability_three_le_f_x_le_fifteen : 
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ 6) → 
  (Prob.{0} { t : ℝ | 3 ≤ f t ∧ f t ≤ 15 }) = 1/3 :=
sorry

end probability_three_le_f_x_le_fifteen_l343_343485


namespace complement_U_A_l343_343871

def U : Finset ℤ := {-2, -1, 0, 1, 2}
def A : Finset ℤ := {-2, -1, 1, 2}

theorem complement_U_A : (U \ A) = {0} := by
  sorry

end complement_U_A_l343_343871


namespace midpoint_arc_passes_through_line_MH_l343_343542

theorem midpoint_arc_passes_through_line_MH
  {A B C M H : Type*}
  [tri : triangle A B C]
  (angleC : ∠C = 60)
  (angleA : ∠A = 45)
  (midM : is_midpoint M B C)
  (orthocenterH : is_orthocenter H A B C) :
  passes_through_midpoint_arc_AB M H :=
sorry

end midpoint_arc_passes_through_line_MH_l343_343542


namespace minimum_value_expression_l343_343076

theorem minimum_value_expression {x1 x2 x3 x4 : ℝ} (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1)^2 + 1 / (Real.sin x1)^2) * (2 * (Real.sin x2)^2 + 1 / (Real.sin x2)^2) * (2 * (Real.sin x3)^2 + 1 / (Real.sin x3)^2) * (2 * (Real.sin x4)^2 + 1 / (Real.sin x4)^2) ≥ 81 :=
by {
  sorry
}

end minimum_value_expression_l343_343076


namespace problem_statement_l343_343953

theorem problem_statement
  (a b A B : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (def_f : ∀ θ : ℝ, f θ = 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)) :
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
by
  sorry

end problem_statement_l343_343953


namespace black_balls_count_l343_343685

theorem black_balls_count :
  ∀ (r k : ℕ), r = 10 -> (2 : ℚ) / 7 = r / (r + k : ℚ) -> k = 25 := by
  intros r k hr hprob
  sorry

end black_balls_count_l343_343685


namespace solution_set_l343_343075

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

noncomputable def is_increasing_on (f : ℝ → ℝ) (S : set ℝ) : Prop :=
∀ x y ∈ S, x ≤ y → f x ≤ f y

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
is_even_function f ∧ is_increasing_on f (set.Ici 0) ∧ f (1/3) = 0

theorem solution_set (f : ℝ → ℝ) (x : ℝ) :
  satisfies_conditions f →
  (f (Real.log_base (1/8) x) > 0 ↔ 0 < x ∧ x < 1/2 ∨ x > 2) :=
by {
  -- This is where the proof would go
  sorry
}

end solution_set_l343_343075


namespace towel_shrinkage_l343_343374

def shrinkage_rate_length (cotton polyester : ℚ) : ℚ := 
  (0.35 * cotton) + (0.25 * polyester)

def shrinkage_rate_breadth (cotton polyester : ℚ) : ℚ := 
  (0.45 * cotton) + (0.30 * polyester)

def percentage_decrease_area (length_shrink breadth_shrink : ℚ) : ℚ := 
  (1 - (1 - length_shrink) * (1 - breadth_shrink)) * 100

theorem towel_shrinkage : ∀ (cotton polyester : ℚ), 
  cotton = 0.60 → polyester = 0.40 →
  percentage_decrease_area 
    (shrinkage_rate_length cotton polyester) 
    (shrinkage_rate_breadth cotton polyester) 
  = 57.91 := 
by 
  intros cotton polyester hc hp
  have len_shrink := shrinkage_rate_length cotton polyester
  have brd_shrink := shrinkage_rate_breadth cotton polyester
  have ha : len_shrink = 0.31 := by 
    rw [hc, hp]
    norm_num
  have hb : brd_shrink = 0.39 := by 
    rw [hc, hp]
    norm_num
  rw [ha, hb]
  norm_num
  sorry -- Proof to be completed

end towel_shrinkage_l343_343374


namespace problem1_problem2_l343_343912

variables (m : ℝ × ℝ) (n : ℝ × ℝ)
variables (_1 : m = (1, 1))
variables (_2 : ∥n∥ = 1)
variables (_3 : real.angle m n = real.pi / 4)

-- Question 1
theorem problem1 : ∥2 • m + n∥ = real.sqrt 13 :=
sorry

-- Question 2
theorem problem2 (λ : ℝ) : (2 • m + λ • n) ⬝ (λ • m + 3 • n) < 0 ↔ 
  -6 < λ ∧ λ < -1 ∧ λ ≠ -real.sqrt 6 :=
sorry

end problem1_problem2_l343_343912


namespace max_correct_answers_l343_343355

theorem max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 25) 
  (h2 : 5 * c - 2 * w = 60) : 
  c ≤ 14 := 
sorry

end max_correct_answers_l343_343355


namespace intersection_A_B_l343_343104

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2 * x > 0}

-- Prove the intersection of A and B
theorem intersection_A_B :
  (A ∩ B) = {x | x < (3 / 2)} := sorry

end intersection_A_B_l343_343104


namespace sum_of_different_roots_eq_six_l343_343674

theorem sum_of_different_roots_eq_six (a b : ℝ) (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end sum_of_different_roots_eq_six_l343_343674


namespace max_volume_of_ABEF_l343_343915

noncomputable def volume_of_tetrahedron (AE BF EF : ℝ) (α β : Type*) 
  [plane α] [plane β]
  (h1 : AE ∈ α) (h2 : BF ∈ β)
  (h3 : AE ⊥ EF) (h4 : BF ⊥ EF)
  (hEF : EF = 1) (hAE : AE = 2) (hAB : AE.distance_to BF <|> {definition-of-AB}.distance_to) : 
  ℝ := 1 / 3

theorem max_volume_of_ABEF 
  {α β : Type*}
  [plane α] [plane β] 
  (AE BF EF : ℝ)
  (h1 : AE ∈ α) (h2 : BF ∈ β) 
  (h3 : AE ⊥ EF) (h4 : BF ⊥ EF)
  (hEF : EF = 1) (hAE : AE = 2) (hAB : AE.distance_to BF <|> {definition-of-AB}.distance_to) : 
  volume_of_tetrahedron AE BF EF α β h1 h2 h3 h4 hEF hAE hAB = 1 / 3 := 
sorry

end max_volume_of_ABEF_l343_343915


namespace garden_roller_diameter_l343_343696

theorem garden_roller_diameter 
  (length : ℝ) 
  (total_area : ℝ) 
  (num_revolutions : ℕ) 
  (pi : ℝ) 
  (A : length = 2)
  (B : total_area = 37.714285714285715)
  (C : num_revolutions = 5)
  (D : pi = 22 / 7) : 
  ∃ d : ℝ, d = 1.2 :=
by
  sorry

end garden_roller_diameter_l343_343696


namespace find_N_l343_343794

-- Define matrices A, B, and the target matrix N.
def A : Matrix (Fin 2) (Fin 2) ℚ := ![![2, -5], ![4, -3]]
def B : Matrix (Fin 2) (Fin 2) ℚ := ![[-20, 10], ![8, -4]]
def N : Matrix (Fin 2) (Fin 2) ℚ := ![![10 / 7, -40 / 7], ![-4 / 7, 16 / 7]]

-- Objective: prove that the given matrix N satisfies the condition.
theorem find_N : N * A = B :=
by
  -- Proof is omitted with a 'sorry' placeholder.
  sorry

end find_N_l343_343794


namespace sequence_sum_l343_343392

theorem sequence_sum : 
  let seq : ℕ → ℤ := λ n => if even n then 2 + 4 * (n / 2) else -(6 + 4 * (n / 2)) in
  (∑ n in finset.range 26, seq n) = -52 := 
by
  let seq : ℕ → ℤ := λ n => if even n then 2 + 4 * (n / 2) else -(6 + 4 * (n / 2))
  sorry

end sequence_sum_l343_343392


namespace base5_num_eq_179_l343_343404

-- Define a function to convert base-5 numbers to decimal
def base5_to_decimal (n : List ℕ) : ℕ :=
  n.reverse.foldl (λ acc d, acc * 5 + d) 0

-- Define the base-5 number 1204 as a list of digits
def base5_num : List ℕ := [1, 2, 0, 4]

-- State the theorem we want to prove
theorem base5_num_eq_179 : base5_to_decimal base5_num = 179 :=
by
  -- Proof goes here
  sorry

end base5_num_eq_179_l343_343404


namespace no_carry_addition_exists_l343_343891

theorem no_carry_addition_exists : ∃ k : ℕ, k > 0 ∧ ∀ d : ℕ, (d < 10 → ( (1996 * k + 1997 * k) % (10^d) ) < 10) :=
by
  let k := 222
  existsi k
  split
  -- k > 0
  exact Nat.succ_pos'
  -- ∀ d : ℕ, (d < 10 → ( (1996 * k + 1997 * k) % (10^d) ) < 10)
  intro d hd
  sorry

end no_carry_addition_exists_l343_343891


namespace Maria_students_l343_343566

variable (M J : ℕ)

def conditions : Prop :=
  (M = 4 * J) ∧ (M + J = 2500)

theorem Maria_students : conditions M J → M = 2000 :=
by
  intro h
  sorry

end Maria_students_l343_343566


namespace range_of_a_l343_343052

-- Define the piecewise function f(x)
noncomputable def f (a x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else log a x

-- The condition that f(x) is decreasing on (-∞, +∞)
def is_decreasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f a x ≥ f a y

-- The statement to prove
theorem range_of_a (a : ℝ) : 
  is_decreasing a ↔ (⅐ ≤ a ∧ a < ⅓) :=
sorry

end range_of_a_l343_343052


namespace unique_solution_l343_343422

-- Definitions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime (4 * q - 1) ∧ (p + q) * (r - p) = p + r

theorem unique_solution (p q r : ℕ) (h : satisfies_conditions p q r) : (p, q, r) = (2, 3, 3) :=
  sorry

end unique_solution_l343_343422


namespace simplify_fraction_mul_l343_343989

theorem simplify_fraction_mul (a b c d : ℕ) (h1 : a = 210) (h2 : b = 7350) (h3 : c = 1) (h4 : d = 35) (h5 : 210 / gcd 210 7350 = 1) (h6: 7350 / gcd 210 7350 = 35) :
  (a / b) * 14 = 2 / 5 :=
by
  sorry

end simplify_fraction_mul_l343_343989


namespace monotonically_decreasing_interval_l343_343626

noncomputable theory

def decreasing_interval (f : ℝ → ℝ) : Set ℝ := {x | ∀ ε > 0, f(x + ε) ≤ f x}

def f (x : ℝ) : ℝ := log 4 + 3 * x - x^2

def domain : Set ℝ := {x | -1 < x ∧ x < 4}

theorem monotonically_decreasing_interval :
  (∀ x ∈ domain, ∀ ε > 0, f(x + ε) ≤ f x) → {x | -1 < x ∧ x < 4} = {x | 3/2 ≤ x ∧ x < 4} :=
begin
  intros h,
  sorry
end

end monotonically_decreasing_interval_l343_343626


namespace sphere_tangent_BD_l343_343714

-- Given definitions
variables {A B C D E F G H: Point}
variable {S: Sphere}

-- Defining geometric relationships:
variable square_EFGH : isSquare E F G H
variable tangent_E_AB : isTangent (S) (A B) E
variable tangent_F_BC : isTangent (S) (B C) F
variable tangent_G_CD : isTangent (S) (C D) G
variable tangent_H_DA : isTangent (S) (D A) H
variable tangent_I_AC : isTangent (S) (A C) I

-- Goal: Prove that the sphere is tangent to edge BD
theorem sphere_tangent_BD (tangent_I_AC : isTangent (S) (A C) I) : isTangent (S) (B D) J :=
  sorry

end sphere_tangent_BD_l343_343714


namespace platyfish_count_l343_343294

-- Define the basic conditions
def num_goldfish : Nat := 3
def balls_per_goldfish : Nat := 10
def balls_per_platyfish : Nat := 5
def total_balls : Nat := 80

-- Prove the number of platyfish, P, is 10
theorem platyfish_count : ∃ (P : Nat), P = 10 :=
  exists.intro (total_balls - num_goldfish * balls_per_goldfish) / balls_per_platyfish sorry

end platyfish_count_l343_343294


namespace infinite_solutions_exists_l343_343983

theorem infinite_solutions_exists :
  ∃ᶠ (x y z : ℕ) in filter (λ n, n > 0) ⊤, (x + y + z)^2 + 2 * (x + y + z) = 5 * (x * y + y * z + z * x) :=
sorry

end infinite_solutions_exists_l343_343983


namespace brick_height_l343_343689

theorem brick_height
  (wall_length : ℝ)
  (wall_width : ℝ)
  (wall_thickness : ℝ)
  (brick_length : ℝ)
  (brick_width : ℝ)
  (num_bricks : ℝ)
  (wall_volume : ℝ)
  (brick_area : ℝ) :
  wall_length = 200 →
  wall_width = 300 →
  wall_thickness = 2 →
  brick_length = 25 →
  brick_width = 11 →
  num_bricks = 72.72727272727273 →
  wall_volume = wall_length * wall_width * wall_thickness →
  brick_area = brick_length * brick_width →
  num_bricks * brick_area * h = wall_volume →
  h = 6 :=
by
  intros wall_length_eq wall_width_eq wall_thickness_eq
    brick_length_eq brick_width_eq num_bricks_eq
    wall_volume_eq brick_area_eq
    eq _
  sorry

end brick_height_l343_343689


namespace smallest_union_cardinality_l343_343593

theorem smallest_union_cardinality (A B : Set α) (hA : A.card = 30) (hB : B.card = 20) : (A ∪ B).card = 35 :=
sorry

end smallest_union_cardinality_l343_343593


namespace parallel_SM_CL_l343_343205

noncomputable theory
open_locale classical

variables (A B C D O S L M : Type) [OrderedGeometry A B C D O S L M]

-- Definitions of points and conditions
variable [Rectangle ABCD]
variable [Center O ABCD]
variable [Angle DAC 60]

-- Intersection Points
variable [Intersection (angle_bisector DAC) DC S]
variable [Intersection (Line OS) AD L]
variable [Intersection (Line BL) AC M]

-- Proof statement
theorem parallel_SM_CL :
  Parallel (Line SM) (Line CL) :=
sorry

end parallel_SM_CL_l343_343205


namespace probability_monotonic_log_function_l343_343835

theorem probability_monotonic_log_function (a : ℝ) (h1 : 0 < a) (h2 : a < 5) (h3 : a ≠ 1) :
  (∀ x y : ℝ, x > 2 → y > 2 → x < y → (log a (a*x - 1) ≤ log a (a*y - 1) ∨ log a (a*x - 1) ≥ log a (a*y - 1))) →
  (∃ p : ℝ, p = (5 - (1/2)) / (5 - 0) ∧ p = 9/10) :=
by
  sorry

end probability_monotonic_log_function_l343_343835


namespace ramesh_paid_price_l343_343584

theorem ramesh_paid_price {P : ℝ} (h1 : P = 18880 / 1.18) : 
  (0.80 * P + 125 + 250) = 13175 :=
by sorry

end ramesh_paid_price_l343_343584


namespace axis_of_symmetry_l343_343258

/-- Stretch and shift the cosine function and prove axis of symmetry -/
theorem axis_of_symmetry {f : ℝ → ℝ} :
  (∀ x, f x = cos (x - π / 3)) →
  (∀ x, f (2 * x + π / 6) = cos (x / 2 - π / 4)) →
  (∀ k : ℤ, f (2 * k * π + π / 2) = cos ((2 * k * π + π / 2) / 2 - π / 4)) →
  (∃ k : ℤ, 2 * k * π + π / 2 = π / 2) :=
by
  intros h1 h2 h3
  use (0 : ℤ)
  sorry

end axis_of_symmetry_l343_343258


namespace decorative_window_ratio_l343_343693

theorem decorative_window_ratio
  (AB : ℝ) (AD : ℝ) (area_rectangle : ℝ) (area_semicircles_triangle : ℝ)
  (h1 : AB = 20)
  (h2 : AD = (3/2) * AB)
  (h3 : area_rectangle = AD * AB)
  (h4 : area_semicircles_triangle = (π * (AB / 2) ^ 2) + ((√3 / 4) * AB ^ 2)) :
  area_rectangle / area_semicircles_triangle = 6 / (π + √3) :=
by
  sorry

end decorative_window_ratio_l343_343693


namespace trailing_zeros_100_factorial_l343_343511

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def multiples (k n : ℕ) : ℕ := n / k

theorem trailing_zeros_100_factorial :
  let n := 100 in
  let num_trailing_zeros := multiples 5 n + multiples 25 n + multiples 125 n in
  num_trailing_zeros = 24 :=
by
  sorry

end trailing_zeros_100_factorial_l343_343511


namespace time_to_fill_bucket_l343_343134

theorem time_to_fill_bucket (t : ℝ) (h : 2/3 = 2 / t) : t = 3 :=
by
  sorry

end time_to_fill_bucket_l343_343134


namespace find_children_in_camp_l343_343165

-- Definitions based on the conditions
def children_in_camp (C : ℕ) : Prop :=
  -- 90% of the children are boys, meaning 10% are girls 
  let boys := 0.90 * C in
  let girls := 0.10 * C in
  -- After adding 100 boys, the girls should be 5% of the total number of children
  let total_children_after_adding_boys := C + 100 in
  let new_boys := boys + 100 in
  let new_girls_percentage := 0.05 * total_children_after_adding_boys in
  -- Current number of girls should equal 10% of C
  girls = new_girls_percentage

-- The proposition stating that given the conditions, we have C = 100
theorem find_children_in_camp : ∃ C : ℕ, children_in_camp C ∧ C = 100 := by
  sorry

end find_children_in_camp_l343_343165


namespace largest_prime_dividing_sum_of_sequence_l343_343001

theorem largest_prime_dividing_sum_of_sequence : ∀ (sequence : List ℕ), 
  (∀ (x : ℕ), x ∈ sequence → 1000 ≤ x ∧ x < 10000) →
  (∀ (i : ℕ), i < sequence.length - 1 →
    (sequence[i] % 1000 * 10 + sequence[i] / 1000 = sequence[i + 1] / 10 ∧ 
    sequence[0] % 1000 * 10 + sequence[0] / 1000 = sequence[sequence.length - 1] / 10)) →
  101 ∣ (sequence.foldl (λ acc x => acc + x) 0) :=
by
  sorry

end largest_prime_dividing_sum_of_sequence_l343_343001


namespace cindy_tenth_finger_l343_343004

noncomputable def g : ℕ → ℕ
| 0 := 0
| 1 := 2
| 2 := 3
| 3 := 4
| 4 := 1
| 5 := 8
| 6 := 5
| 7 := 6
| 8 := 7
| 9 := 0
| _ := 0

theorem cindy_tenth_finger :
  let start := 2
  let f := λ x n, nat.iterate g n x 
  f start 10 = 4 :=
by {
  -- omitted proof
  sorry
}

end cindy_tenth_finger_l343_343004


namespace problem1_extreme_values_problem2_monotonically_increasing_range_problem3_extreme_point_l343_343484

-- Problem 1
theorem problem1_extreme_values (x : ℝ) : 
  (∃ x, 0 < x ∧ x = Real.sqrt Real.exp 1) → 
  (f : ℝ → ℝ) → 
  (∀ x, f(x) = (Real.log x) / x^2) → 
  f (Real.sqrt Real.exp 1) = 1 / (2 * Real.exp 1) :=
sorry

-- Problem 2
theorem problem2_monotonically_increasing_range (a x : ℝ) : 
  (a < 0 ∧ 0 < x ∧ x < -a) → 
  (f : ℝ → ℝ) → 
  (∀ x, f(x) = (Real.log x) / (x + a)^2) → 
  a ≤ -2 / Real.sqrt (Real.exp 1) :=
sorry

-- Problem 3
theorem problem3_extreme_point (x₀ : ℝ) : 
  (0 < x₀ ∧ x₀ < 1) → 
  (a = -1) → 
  (f : ℝ → ℝ) → 
  (∀ x, f(x) = (Real.log x) / (x - 1)^2) → 
  f(x₀) < -2 :=
sorry

end problem1_extreme_values_problem2_monotonically_increasing_range_problem3_extreme_point_l343_343484


namespace true_propositions_count_l343_343489

theorem true_propositions_count (b : ℤ) :
  (b = 3 → b^2 = 9) → 
  (∃! p : Prop, p = (b^2 ≠ 9 → b ≠ 3) ∨ p = (b ≠ 3 → b^2 ≠ 9) ∨ p = (b^2 = 9 → b = 3) ∧ (p = (b^2 ≠ 9 → b ≠ 3))) :=
sorry

end true_propositions_count_l343_343489


namespace complement_union_l343_343494

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  (U \ M) ∪ N = {2, 3, 4} :=
sorry

end complement_union_l343_343494


namespace width_of_roads_l343_343365

-- Definitions for the conditions
def length_of_lawn := 80 
def breadth_of_lawn := 60 
def total_cost := 5200 
def cost_per_sq_m := 4 

-- Derived condition: total area based on cost
def total_area_by_cost := total_cost / cost_per_sq_m 

-- Statement to prove: width of each road w is 65/7
theorem width_of_roads (w : ℚ) : (80 * w) + (60 * w) = total_area_by_cost → w = 65 / 7 :=
by
  sorry

end width_of_roads_l343_343365


namespace ones_digit_of_31_pow_l343_343799

-- Definitions as per conditions
def onesDigit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_31_pow :
  onesDigit (31 ^ (15 * (7 ^ 7))) = 3 :=
by
  -- Applying the connection between 31 and 3
  have h1 : onesDigit (31 ^ (15 * (7 ^ 7))) = onesDigit (3 ^ (15 * (7 ^ 7))), from sorry,

  -- Analyzing the cycle of ones digits for powers of 3
  have cycle : list ℕ := [3, 9, 7, 1],
  have cycle_length : cycle.length = 4, from sorry,

  -- Calculating the relevant exponent modulo the cycle length
  have exponent_mod : (15 * (7 ^ 7)) % 4 = 1, from sorry,

  -- Applying the cycle to determine the ones digit
  have ones_digit_3 : onesDigit (3 ^ (15 * (7 ^ 7))) = cycle.nth_le (exponent_mod % cycle_length) (by simp [cycle_length, exponent_mod]), from sorry,

  -- Concluding the proof
  exact eq.trans h1 ones_digit_3

end ones_digit_of_31_pow_l343_343799


namespace circles_externally_tangent_l343_343408

noncomputable def circle1 : (ℝ → ℝ → Prop) :=
fun x y => x^2 + y^2 - 6y = 0

noncomputable def circle2 : (ℝ → ℝ → Prop) :=
fun x y => x^2 + y^2 - 8x + 12 = 0

theorem circles_externally_tangent : 
  let center1 := (0 : ℝ, 3 : ℝ)
  let center2 := (4 : ℝ, 0 : ℝ)
  let radius1 := (3 : ℝ)
  let radius2 := (2 : ℝ)
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius1 + radius2 :=
by
  sorry

end circles_externally_tangent_l343_343408


namespace volume_of_pyramid_TABC_l343_343977

open Real

-- Defining the points and distances
variables (A B C T : Type) 
variables (TA TB TC : ℝ)
variables (h1 : TA = 12) (h2 : TB = 12) (h3 : TC = 15)

-- Using the conditions to define the volume of the pyramid
theorem volume_of_pyramid_TABC :
  ∀ (A B C T : Type) (TA TB TC : ℝ),
  TA = 12 → TB = 12 → TC = 15 →
  (∃ V : ℝ, V = (1/3) * (1/2) * TA * TB * TC ∧ V = 360) :=
by
  intros A B C T TA TB TC h1 h2 h3,
  use (1/3) * (1/2) * TA * TB * TC,
  split,
  { rw [h1, h2, h3],
    norm_num },
  { rw [h1, h2, h3],
    norm_num }
  sorry

end volume_of_pyramid_TABC_l343_343977


namespace remainder_S_81_mod_81_l343_343761

def S_81 : ℕ := ∑ i in Finset.range 81, 10^i

theorem remainder_S_81_mod_81 : S_81 % 81 = 0 :=
by
  sorry

end remainder_S_81_mod_81_l343_343761


namespace find_term_1000_l343_343140

noncomputable def sequence : ℕ → ℕ
| 0     := 2040
| 1     := 2042
| (n+2) := (n : ℕ) + 1 - sequence (n) - sequence (n + 1)

lemma sequence_property (n : ℕ) (hn : 1 ≤ n) :
  sequence n + sequence (n + 1) + sequence (n + 2) = n + 1 := sorry

theorem find_term_1000 :
  sequence 999 + 333 = 2373 := 
begin
  have h₁ : sequence 0 = 2040 := by refl,
  have h₂ : sequence 1 = 2042 := by refl,
  sorry
end

end find_term_1000_l343_343140


namespace quadratic_function_range_l343_343441

-- Define the quadratic function and the domain
def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 + 1

-- State the proof problem
theorem quadratic_function_range : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → -8 ≤ quadratic_function x ∧ quadratic_function x ≤ 1 := 
by 
  intro x
  intro h
  sorry

end quadratic_function_range_l343_343441


namespace maximize_quadratic_function_l343_343320

theorem maximize_quadratic_function (x : ℝ) :
  (∀ x, -2 * x ^ 2 - 8 * x + 18 ≤ 26) ∧ (-2 * (-2) ^ 2 - 8 * (-2) + 18 = 26) :=
by (
  sorry
)

end maximize_quadratic_function_l343_343320


namespace max_sum_of_15x15_grid_l343_343908

def cell_value (n : ℕ) : Prop := n ≤ 4

def grid_sum_to_seven (grid : ℕ → ℕ → ℕ) : Prop := 
  ∀ i j, i < 14 ∧ j < 14 → grid i j + grid (i + 1) j + grid i (j + 1) + grid (i + 1) (j + 1) = 7

theorem max_sum_of_15x15_grid : ∃ (grid : ℕ → ℕ → ℕ), 
  (∀ i j, i < 15 ∧ j < 15 → cell_value (grid i j)) ∧ 
  grid_sum_to_seven grid ∧ 
  (Σ i j, if i < 15 ∧ j < 15 then grid i j else 0) = 417 := 
sorry

end max_sum_of_15x15_grid_l343_343908


namespace routes_from_A_to_B_l343_343874

-- Define the grid with given dimensions
def grid_width : ℕ := 3
def grid_height : ℕ := 2

-- Define the total number of moves needed and the distribution of right and down moves
def total_moves : ℕ := grid_width + grid_height
def right_moves : ℕ := grid_width
def down_moves : ℕ := grid_height

theorem routes_from_A_to_B : (nat.choose total_moves down_moves) = 10 := 
by
  have h : total_moves = 5 := dec_trivial
  have r : down_moves = 2 := dec_trivial
  rw [h, r]
  exact nat.choose_succ_succ_succ 3
  sorry

end routes_from_A_to_B_l343_343874


namespace min_max_of_f_l343_343008

def f (x : ℝ) : ℝ := -2 * x + 1

-- defining the minimum and maximum values
def min_val : ℝ := -3
def max_val : ℝ := 5

theorem min_max_of_f :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≥ min_val) ∧ 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f x ≤ max_val) :=
by 
  sorry

end min_max_of_f_l343_343008


namespace symmetry_about_origin_l343_343887

theorem symmetry_about_origin (m : ℝ) (A B : ℝ × ℝ) (hA : A = (2, -1)) (hB : B = (-2, m)) (h_sym : B = (-A.1, -A.2)) :
  m = 1 :=
by
  sorry

end symmetry_about_origin_l343_343887


namespace discounted_ticket_percentage_is_60_l343_343544

-- Definitions based on the conditions
def normal_price := 50
def website_tickets_cost := 2 * normal_price
def scalper_price_per_ticket := 2.4 * normal_price
def scalper_discount := 10
def scalper_cost := 2 * scalper_price_per_ticket - scalper_discount
def total_amount_paid := 360
def discounted_ticket_cost := total_amount_paid - (website_tickets_cost + scalper_cost)
def discounted_ticket_percentage_of_normal_price := (discounted_ticket_cost / normal_price) * 100

-- Statement to prove
theorem discounted_ticket_percentage_is_60 :
  discounted_ticket_percentage_of_normal_price = 60 := sorry

end discounted_ticket_percentage_is_60_l343_343544


namespace point_not_on_graph_l343_343328

theorem point_not_on_graph : (-1 : ℝ, -1 : ℝ) ∉ set_of (λ p : ℝ × ℝ, p.snd = (p.fst - 1) / (p.fst + 2)) :=
by
  sorry

end point_not_on_graph_l343_343328


namespace brody_battery_fraction_l343_343744

theorem brody_battery_fraction (full_battery : ℕ) (battery_left_after_exam : ℕ) (exam_duration : ℕ) 
  (battery_before_exam : ℕ) (battery_used : ℕ) (fraction_used : ℚ) 
  (h1 : full_battery = 60)
  (h2 : battery_left_after_exam = 13)
  (h3 : exam_duration = 2)
  (h4 : battery_before_exam = battery_left_after_exam + exam_duration)
  (h5 : battery_used = full_battery - battery_before_exam)
  (h6 : fraction_used = battery_used / full_battery) :
  fraction_used = 3 / 4 := 
sorry

end brody_battery_fraction_l343_343744


namespace area_of_octagon_is_100_sqrt_3_l343_343960

noncomputable def area_of_octagon_inscribed_in_circle : ℝ :=
let radius := 10 in
let perimeter := 40 in
let side_length := perimeter / 4 in
let diagonal := radius * 2 in
let midpoint_distance := radius in
let triangle_base := side_length in
let triangle_height := Real.sqrt (radius^2 - (side_length / 2)^2) in
let triangle_area := (1 / 2) * triangle_base * triangle_height in
let octagon_area := 4 * triangle_area in
octagon_area

theorem area_of_octagon_is_100_sqrt_3 :
  area_of_octagon_inscribed_in_circle = 100 * Real.sqrt 3 :=
by
  sorry

end area_of_octagon_is_100_sqrt_3_l343_343960


namespace solve_sqrt_eq_l343_343598

theorem solve_sqrt_eq (x : ℝ) (h : sqrt (2 + sqrt (3 + sqrt x)) = root 4 (2 + sqrt x)) : 
  x = 81 / 256 :=
by sorry

end solve_sqrt_eq_l343_343598


namespace find_speed_of_man_in_still_water_l343_343703

noncomputable def speed_of_man_in_still_water (v_m v_s : ℝ) 
  (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (downstream_time : ℝ) (upstream_time : ℝ) 
  (downstream_condition : downstream_distance / downstream_time = v_m + v_s) 
  (upstream_condition : upstream_distance / upstream_time = v_m - v_s) : Prop := 
  v_m = 9

theorem find_speed_of_man_in_still_water : 
  ∃ (v_m v_s : ℝ), 
    (∀ (downstream_distance upstream_distance downstream_time upstream_time : ℝ), 
      (downstream_distance = 36 ∧ upstream_distance = 18 ∧ 
       downstream_time = 3 ∧ upstream_time = 3) → 
      speed_of_man_in_still_water v_m v_s downstream_distance upstream_distance downstream_time upstream_time 
      (downstream_distance / downstream_time = v_m + v_s) 
      (upstream_distance / upstream_time = v_m - v_s)) := 
begin
  use 9,
  use 3, -- after solving, we get v_s = 3
  sorry
end

end find_speed_of_man_in_still_water_l343_343703


namespace probability_C_speaks_first_l343_343524

-- Definitions for students and positions
inductive Student
| A | B | C | D | E

-- Define probability
noncomputable def P (event : Set (List Student)) : ℚ :=
  event.card / 5.factorial

-- Event where student A is not the first and student B is not the last
def eventA : Set (List Student) :=
  { l | l.head ≠ Student.A ∧ l.last ≠ Student.B }

-- Event where student C speaks first
def eventB : Set (List Student) :=
  { l | l.head = Student.C }

-- Combined event: A not first, B not last, and C speaks first
def eventAB : Set (List Student) :=
  { l | l.head = Student.C ∧ l.last ≠ Student.B }

-- Number of permutations of the list
def all_permutations := 
  { l : List Student | l.permutations ∈ (List.permutations [Student.A, Student.B, Student.C, Student.D, Student.E])}

-- Calculate the probability P(AB) and P(A), then show the conditional probability P(B|A)
theorem probability_C_speaks_first :
  P eventAB / P eventA = (3 : ℚ) / 13 :=
by 
  sorry

end probability_C_speaks_first_l343_343524


namespace cot_difference_inequality_l343_343240

theorem cot_difference_inequality (x : ℝ) (n : ℕ) (h1 : 0 < x) (h2 : x < π) : 
  Real.cot (x / 2^n) - Real.cot x ≥ n := sorry

end cot_difference_inequality_l343_343240


namespace probability_cheryl_same_color_l343_343343

theorem probability_cheryl_same_color :
  let total_marble_count := 12
  let marbles_per_color := 3
  let carol_draw := 3
  let claudia_draw := 3
  let cheryl_draw := total_marble_count - carol_draw - claudia_draw
  let num_colors := 4

  0 < marbles_per_color ∧ marbles_per_color * num_colors = total_marble_count ∧
  0 < carol_draw ∧ carol_draw <= total_marble_count ∧
  0 < claudia_draw ∧ claudia_draw <= total_marble_count - carol_draw ∧
  0 < cheryl_draw ∧ cheryl_draw <= total_marble_count - carol_draw - claudia_draw ∧
  num_colors * (num_colors - 1) > 0
  →
  ∃ (p : ℚ), p = 2 / 55 := 
sorry

end probability_cheryl_same_color_l343_343343


namespace find_valid_pairs_l343_343406

def are_lines_non_horizontal_and_not_parallel_and_not_concurrent (s : ℕ) : Prop :=
  ∀ i j k : ℕ, (i < s + 1) ∧ (j < s + 1) ∧ (k < s + 1) →
    i ≠ j ∧ i ≠ k ∧ j ≠ k →
    (¬(lines(i).horizontal)) ∧ (¬(lines(j).horizontal)) ∧ (¬(lines(k).horizontal)) ∧
    (¬(lines(i).parallel lines(j))) ∧ (¬(lines(j).parallel lines(k))) ∧ (¬(lines(i).parallel lines(k))) ∧
    (¬(lines(i).concurrent_3lines lines(j) lines(k)))

def number_of_regions (h s : ℕ) : ℕ :=
  s * (s + 1) / 2 + 1 + h * (s + 1)

theorem find_valid_pairs :
  ∀ (h s : ℕ), are_lines_non_horizontal_and_not_parallel_and_not_concurrent s →
  number_of_regions h s = 1992 →
  (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by
  intros h s h_condition region_eq
  sorry

end find_valid_pairs_l343_343406


namespace unique_continuous_f_l343_343428

noncomputable def continuous_function := { f : ℝ+ → ℝ+ // continuous f }

theorem unique_continuous_f (f : continuous_function) (hf : ∀ x y > 0, f.1 (x + y) * (f.1 x + f.1 y) = f.1 x * f.1 y) :
  ∃ α > 0, ∀ x > 0, f.1 x = 1 / (α * x) := 
sorry

end unique_continuous_f_l343_343428


namespace melissa_games_played_l343_343231

-- Define the conditions mentioned:
def points_per_game := 12
def total_points := 36

-- State the proof problem:
theorem melissa_games_played : total_points / points_per_game = 3 :=
by sorry

end melissa_games_played_l343_343231


namespace common_arithmetic_sequence_term_l343_343380

theorem common_arithmetic_sequence_term :
  let first_sequence (n : ℕ) := 3 + 8 * n
  let second_sequence (m : ℕ) := 5 + 9 * m
  ∃ (a : ℕ), (∃ n, a = first_sequence n) ∧ (∃ m, a = second_sequence m) ∧ 1 ≤ a ∧ a ≤ 150 ∧ ∀ b, (∃ n, b = first_sequence n) ∧ (∃ m, b = second_sequence m) ∧ 1 ≤ b ∧ b ≤ 150 → b ≤ a :=
  ∃ (a : ℕ), (∃ n, a = 3 + 8 * n) ∧ (∃ m, a = 5 + 9 * m) ∧ 1 ≤ a ∧ a ≤ 150 ∧ ∀ b, (∃ n, b = 3 + 8 * n) ∧ (∃ m, b = 5 + 9 * m) ∧ 1 ≤ b ∧ b ≤ 150 → b ≤ 131 :=
sorry

end common_arithmetic_sequence_term_l343_343380


namespace ratio_DK_AB_l343_343985

-- Definitions and given conditions
variables {Point : Type*} [affine_space Point]
variables {A B C D C1: Point}

-- Rectangle condition
def is_rectangle (A B C D : Point) : Prop :=
affine_square.is_rectangle A B C D

-- Midpoint condition
def is_midpoint (C1 : Point) (A D : Point) : Prop :=
affine_square.is_midpoint C1 A D

-- Main theorem statement
theorem ratio_DK_AB 
  [h_rect : is_rectangle A B C D] 
  [h_mid : is_midpoint C1 A D] : 
  DK.to_line_segment / AB.to_line_segment = 1 / 3 := 
sorry

end ratio_DK_AB_l343_343985


namespace focus_of_parabola_y_2x2_l343_343021

theorem focus_of_parabola_y_2x2 :
  ∃ f, f = 1 / 8 ∧ (∀ x, sqrt (x^2 + (2*x^2 - f)^2) = abs (2*x^2 - (-f)))
:= sorry

end focus_of_parabola_y_2x2_l343_343021


namespace max_sum_of_elements_l343_343215

noncomputable def matrix_A (a b c : ℤ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (1/5 : ℝ) • Matrix.mk ![-3, a] ![b, c]

theorem max_sum_of_elements
  (a b c : ℤ)
  (hA : (matrix_A a b c ⬝ matrix_A a b c) = 1) :
  a + b + c = 20 :=
sorry

end max_sum_of_elements_l343_343215


namespace one_div_thirteen_150th_digit_is_three_l343_343652

noncomputable def decimal_repeating_digits : List ℕ := [0, 7, 6, 9, 2, 3]

theorem one_div_thirteen_150th_digit_is_three :
  decimal_repeating_digits.length = 6 →
  (∃ r : ℕ, r < decimal_repeating_digits.length ∧ 150 % decimal_repeating_digits.length = r ∧ decimal_repeating_digits[r] = 3) :=
by
  intros hlen
  use 5
  split
  · exact Nat.lt_of_succ_lt_succ (by decide)
  split
  · norm_num
  · rfl

end one_div_thirteen_150th_digit_is_three_l343_343652


namespace find_AX_length_l343_343580

noncomputable def AX_length (A B C D X : Point) 
  (circle : Circle Point) 
  (diameter : LineSegment A D)
  (on_circle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D → circle.has_point P)
  (X_on_AD : diameter.has_point X)
  (BX_eq_CX : distance B X = distance C X)
  (angle_conds : ∠ B A C = 10 ∧ ∠ B X C = 30) : Real :=
  cos 10 * sin 20 * csc 15

theorem find_AX_length (A B C D X : Point)
  (circle : Circle Point)
  (diameter : LineSegment A D)
  (on_circle : ∀ (P : Point), P = A ∨ P = B ∨ P = C ∨ P = D → circle.has_point P)
  (X_on_AD : diameter.has_point X)
  (BX_eq_CX : distance B X = distance C X)
  (angle_conds : ∠ B A C = 10 ∧ ∠ B X C = 30) : 
  AX_length A B C D X circle diameter on_circle X_on_AD BX_eq_CX angle_conds =
  cos 10 * sin 20 * csc 15 := by 
  sorry

end find_AX_length_l343_343580


namespace expected_value_lin_transform_l343_343813

noncomputable def E_5xi_plus_1 : ℝ := 3

namespace DefectiveItems

variables (ξ : ℕ) (σ : Type) [Fintype σ] [Uniform_Laws σ] (genuine defective : Finset σ)
variable [decidable_eq σ]

axiom batch_items : ∃ (genuine defective : Finset σ), 
  genuine.card = 13 ∧ defective.card = 2

axiom draw_items : 
  ∀ (s : Finset σ), s.card = 3 → 
  ∃ (ξ : ℕ), ξ = s.filter (λ x, x ∈ defective).card

theorem expected_value_lin_transform (ξ : ℕ) (E : ℕ → ℝ) :
  E (5 * ξ + 1) = 3 :=
sorry

end DefectiveItems

end expected_value_lin_transform_l343_343813


namespace correct_statements_proof_l343_343729

-- Define the statements as variables
variable (stmt1 : Prop) 
variable (stmt2 : Prop) 
variable (stmt3 : Prop)

-- Define the correct statements set
def correct_statements := {stmt1, stmt3}

-- The main theorem
theorem correct_statements_proof (h1 : stmt1) (h3 : stmt3) (h2 : ¬ stmt2) : 
  ∀ (s : Prop), s ∈ correct_statements ↔ (s = stmt1 ∨ s = stmt3) := 
by
  sorry

end correct_statements_proof_l343_343729


namespace cameron_sandra_remaining_task_days_l343_343182

-- Define the variables and conditions
def cameron_work_rate := 1 / 18
def cameron_work_9_days := 9 * cameron_work_rate
def remaining_task := 1 - cameron_work_9_days
def combined_work_rate := 1 / 7
def days_to_finish_remaining_task := remaining_task / combined_work_rate

theorem cameron_sandra_remaining_task_days:
  days_to_finish_remaining_task = 3.5 :=
by 
  unfold cameron_work_rate cameron_work_9_days remaining_task combined_work_rate days_to_finish_remaining_task
  sorry

end cameron_sandra_remaining_task_days_l343_343182


namespace increasing_or_decreasing_subseq_l343_343824

theorem increasing_or_decreasing_subseq {m n : ℕ} (a : Fin (m * n + 1) → ℝ) :
  ∃ (idx_incr : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_incr i) < a (idx_incr j)) ∨ 
  ∃ (idx_decr : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_decr i) > a (idx_decr j)) :=
by
  sorry

end increasing_or_decreasing_subseq_l343_343824


namespace parallel_planes_if_perpendicular_to_same_line_l343_343846

variables {m n : Type} [line m] [line n]
variables {α β : Type} [plane α] [plane β]

theorem parallel_planes_if_perpendicular_to_same_line (h1 : m ⟂ α) (h2 : m ⟂ β) : α ∥ β :=
sorry

end parallel_planes_if_perpendicular_to_same_line_l343_343846


namespace option_A_option_B_option_C_option_D_l343_343096

section
variables {a b r : ℝ}
def line := {p : ℝ × ℝ | a * p.1 + b * p.2 - r^2 = 0}
def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
def point (x y : ℝ) := (x, y)

theorem option_A (h : a^2 + b^2 = r^2) : ∀ p ∈ circle, p = point a b → ∀ l ∈ line, tangent l circle :=
sorry

theorem option_B (h : a^2 + b^2 < r^2) : ∀ l ∈ line, disjoint l circle :=
sorry

theorem option_C (h : a^2 + b^2 > r^2) : ∃ l ∈ line, intersects l circle :=
sorry

theorem option_D (h_on_l : a * a + b * b - r^2 = 0) (h : a^2 + b^2 = r^2) : ∀ p ∈ point a b, ∀ l ∈ line, tangent l circle :=
sorry
end

end option_A_option_B_option_C_option_D_l343_343096


namespace sequence_general_formula_l343_343167

noncomputable def a : ℕ+ → ℚ
| 1 := 2
| (n+1) := a n / (1 + a n)

theorem sequence_general_formula (n : ℕ+) : a n = 2 / (2 * n - 1) := 
sorry

end sequence_general_formula_l343_343167


namespace mean_geq_half_sum_l343_343555

theorem mean_geq_half_sum (m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) 
  (a : fin m → ℕ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) (h_range : ∀ i, 1 ≤ a i ∧ a i ≤ n)
  (h_condition : ∀ i j, i ≤ j → a i + a j ≤ n → ∃ k, k < m ∧ a i + a j = a k) :
  (∑ i, a i) / m ≥ (n + 1) / 2 :=
by
  sorry

end mean_geq_half_sum_l343_343555


namespace tetrahedron_cube_volume_ratio_l343_343666

theorem tetrahedron_cube_volume_ratio (a : ℝ) :
  let V_tetrahedron := (a * Real.sqrt 2)^3 * Real.sqrt 2 / 12
  let V_cube := a^3
  (V_tetrahedron / V_cube) = 1 / 3 :=
by
  sorry

end tetrahedron_cube_volume_ratio_l343_343666


namespace zero_in_A_l343_343103

-- Define the set A
def A : Set ℝ := { x | x * (x - 2) = 0 }

-- State the theorem
theorem zero_in_A : 0 ∈ A :=
by {
  -- Skipping the actual proof with "sorry"
  sorry
}

end zero_in_A_l343_343103


namespace operation_B_is_correct_l343_343325

theorem operation_B_is_correct (a b x : ℝ) : 
  2 * (a^2) * b * 4 * a * (b^3) = 8 * (a^3) * (b^4) :=
by
  sorry

-- Conditions for incorrect operations
lemma operation_A_is_incorrect (x : ℝ) : 
  x^8 / x^2 ≠ x^4 :=
by
  sorry

lemma operation_C_is_incorrect (x : ℝ) : 
  (-x^5)^4 ≠ -x^20 :=
by
  sorry

lemma operation_D_is_incorrect (a b : ℝ) : 
  (a + b)^2 ≠ a^2 + b^2 :=
by
  sorry

end operation_B_is_correct_l343_343325


namespace spherical_to_rectangular_conversion_l343_343006

theorem spherical_to_rectangular_conversion 
  (ρ θ φ : ℝ) 
  (hρ : ρ = 3)
  (hθ : θ = 5 * π / 12)
  (hφ : φ = π / 4) : 
  let x := ρ * sin φ * cos θ in
  let y := ρ * sin φ * sin θ in
  let z := ρ * cos φ in
  (x, y, z) = (3 * (sqrt 3 + 1) / 4, 3 * (sqrt 3 - 1) / 4, 3 * sqrt 2 / 2) := 
sorry

end spherical_to_rectangular_conversion_l343_343006


namespace partial_fraction_decomposition_l343_343016

theorem partial_fraction_decomposition (A B C : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-x^2 + 5*x - 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) →
  A = 6 ∧ B = -7 ∧ C = 5 :=
by
  intro h
  sorry

end partial_fraction_decomposition_l343_343016


namespace perpendicular_lines_l343_343620

theorem perpendicular_lines (a : ℝ) : 
  (∀ (x y : ℝ), (1 - 2 * a) * x - 2 * y + 3 = 0 → 3 * x + y + 2 * a = 0) → 
  a = 1 / 6 :=
by
  sorry

end perpendicular_lines_l343_343620


namespace conjugate_of_given_complex_l343_343265

open Complex

theorem conjugate_of_given_complex :
  conj (5 / (3 + 4 * I)) = (3 / 5) + (4 / 5) * I := by
sorry

end conjugate_of_given_complex_l343_343265


namespace tangent_sum_identity_l343_343635

theorem tangent_sum_identity :
  let t12 := Real.tan (Real.pi / 15) -- tan 12 degrees
  let t18 := Real.tan (Real.pi / 10) -- tan 18 degrees
  sqrt 3 * t12 + sqrt 3 * t18 + t12 * t18 = 1 :=
by
  let t12 := Real.tan (Real.pi / 15)
  let t18 := Real.tan (Real.pi / 10)
  sorry

end tangent_sum_identity_l343_343635


namespace function_passes_through_point_l343_343130

theorem function_passes_through_point (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, -1) ∧ ∀ x : ℝ, (y = a^(x-1) - 2) → y = -1 := by
  sorry

end function_passes_through_point_l343_343130


namespace lighthouses_distance_l343_343280

noncomputable def distance_between_lighthouses
  (AC BC : ℝ) (cos_ACB : ℝ) : ℝ :=
real.sqrt (AC^2 + BC^2 - 2 * AC * BC * cos_ACB)

def AC := 300 -- distance between C and A in meters
def BC := 500 -- distance between C and B in meters
def angle_ACB := 120 -- angle between AC and BC in degrees
def cos_120 := real.cos (120 * real.pi / 180) -- cosine of 120 degrees

theorem lighthouses_distance : 
  distance_between_lighthouses AC BC cos_120 = 700 := 
by
  simp [distance_between_lighthouses, AC, BC, cos_120]
  sorry

end lighthouses_distance_l343_343280


namespace arrange_B_to_base_A_l343_343690

theorem arrange_B_to_base_A :
  ∀ (A B C D base_A base_B base_C : Type),
  (∀ person : {A B C D}, person ∈ {base_A, base_B, base_C}) ∧
  (∀ base : {base_A, base_B, base_C}, ∃ person : {A B C D}, person ∈ base) →
  (card {arrangement | B ∈ base_A ∧
    ∀ person : {A B C D}, person ∈ {base_A, base_B, base_C} ∧
    ∀ base : {base_A, base_B, base_C},
      ∃ person : {A B C D}, person ∈ base}) = 12 := by sorry

end arrange_B_to_base_A_l343_343690


namespace no_quadrilateral_exists_l343_343312

theorem no_quadrilateral_exists (a b c d : ℝ) (h1 : a = 1) (h2 : b = 3) (h3 : c = 4) (h4 : d = 10) : 
  ¬(a + b > d ∧ a + d > b ∧ b + d > a ∧ a + c > d ∧ b + c > a ∧ c + d > b ∧ a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by {
  rw [h1, h2, h3, h4],
  intro h,
  have h' : ¬(1 + 3 > 10 ∧ 1 + 10 > 3 ∧ 3 + 10 > 1 ∧ 1 + 4 > 10 ∧ 3 + 4 > 1 ∧ 4 + 10 > 3 ∧ 1 + 3 + 4 > 10 ∧ 1 + 3 + 10 > 4 ∧ 1 + 4 + 10 > 3 ∧ 3 + 4 + 10 > 1),
  { simp,
    split; intro h;
    linarith },
  exact h' h,
}

end no_quadrilateral_exists_l343_343312


namespace pentagon_side_length_l343_343517

def pentagon_perimeter (perimeter : Float) (num_sides : Nat) (side_length : Float) : Prop :=
  side_length * num_sides = perimeter

theorem pentagon_side_length :
  pentagon_perimeter 23.4 5 4.68 :=
by
  sorry

end pentagon_side_length_l343_343517


namespace max_a_correct_answers_l343_343902

theorem max_a_correct_answers : 
  ∃ (a b c x y z w : ℕ), 
  a + b + c + x + y + z + w = 39 ∧
  a = b + c ∧
  (a + x + y + w) = a + 5 + (x + y + w) ∧
  b + z = 2 * (c + z) ∧
  23 ≤ a :=
sorry

end max_a_correct_answers_l343_343902


namespace graph_of_equation_is_shifted_hyperbola_l343_343858

-- Definitions
def given_equation (x y : ℝ) : Prop := x^2 - 4*y^2 - 2*x = 0

-- Theorem statement
theorem graph_of_equation_is_shifted_hyperbola :
  ∀ x y : ℝ, given_equation x y = ((x - 1)^2 = 1 + 4*y^2) :=
by
  sorry

end graph_of_equation_is_shifted_hyperbola_l343_343858


namespace fraction_of_profit_B_l343_343682

-- Define the conditions
def total_capital := ℝ
def A_contribution (C : total_capital) := (1/4) * C
def A_time := 15
def B_time := 10

-- Define the shares based on contributions and time
def A_share (C : total_capital) := (A_contribution C) * A_time
def B_contribution (C : total_capital) := (3/4) * C
def B_share (C : total_capital) := (B_contribution C) * B_time

-- Define the fractions of profit
def fraction_A_to_B (C : total_capital) := (A_share C) / (B_share C)
def fraction_B_profit (C : total_capital) := 1 / (1 + fraction_A_to_B C)

-- The theorem statement
theorem fraction_of_profit_B (C : total_capital) : fraction_B_profit C = 2 / 3 := by
  sorry

end fraction_of_profit_B_l343_343682


namespace inv_of_z_l343_343856

theorem inv_of_z (z : ℂ) (h : z = 1 - 2 * complex.i) : z⁻¹ = 1 / 5 + (2 / 5) * complex.i := by
  rw h
  sorry

end inv_of_z_l343_343856


namespace greatest_x_for_quadratic_inequality_l343_343431

theorem greatest_x_for_quadratic_inequality (x : ℝ) (h : x^2 - 12 * x + 35 ≤ 0) : x ≤ 7 :=
sorry

end greatest_x_for_quadratic_inequality_l343_343431


namespace problem_lean_l343_343011

theorem problem_lean : 12 * ((1/3) + (1/4) + (1/6))⁻¹ = 16 := by
  sorry

end problem_lean_l343_343011


namespace sqrt_quartic_equiv_l343_343385

-- Define x as a positive real number
variable (x : ℝ)
variable (hx : 0 < x)

-- Statement of the problem to prove
theorem sqrt_quartic_equiv (x : ℝ) (hx : 0 < x) : (x^2 * x^(1/2))^(1/4) = x^(5/8) :=
sorry

end sqrt_quartic_equiv_l343_343385


namespace map_scale_l343_343331

theorem map_scale (distance_on_map_in_inches : ℝ) (time_in_hours : ℝ) (speed_in_mph : ℝ) 
(h_distance_on_map : distance_on_map_in_inches = 5)
(h_time : time_in_hours = 6.5)
(h_speed : speed_in_mph = 60) :
  (distance_on_map_in_inches / (time_in_hours * speed_in_mph)) ≈ 0.01282 :=
by
  sorry

end map_scale_l343_343331


namespace find_multiple_l343_343548

-- Defining the conditions
def first_lock_time := 5
def second_lock_time (x : ℕ) := 5 * x - 3

-- Proving the multiple
theorem find_multiple : 
  ∃ x : ℕ, (5 * first_lock_time * x - 3) * 5 = 60 ∧ (x = 3) :=
by
  sorry

end find_multiple_l343_343548


namespace problem_statement_l343_343217

variable (f : ℝ → ℝ) (a x : ℝ)

-- Define f(x) as per the problem conditions
def f_def := λ x : ℝ, x^2 - x + 13

-- Hypotheses
hypotheses (ha : ∀ x : ℝ, |x - a| < 1)

-- The theorem statement
theorem problem_statement (f := f_def) (ha : |x - a| < 1) : |f(x) - f(a)| < 2(|a| + 1) := 
sorry

end problem_statement_l343_343217


namespace smallest_lambda_l343_343061

open Real

theorem smallest_lambda (n : ℕ) (h_pos : 0 < n) :
  ∃ λ : ℝ, (∀ (θ : Fin n → ℝ), (∀ i, 0 < θ i ∧ θ i < π / 2) ∧ 
    (∏ i, tan (θ i)) = 2 ^ (n / 2) → (∑ i, cos (θ i)) ≤ λ) ∧ λ = n - 1 :=
sorry

end smallest_lambda_l343_343061


namespace tenth_term_of_arithmetic_progression_l343_343017

variable (a d n T_n : ℕ)

def arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem tenth_term_of_arithmetic_progression :
  arithmetic_progression 8 2 10 = 26 :=
  by
  sorry

end tenth_term_of_arithmetic_progression_l343_343017


namespace evaluate_expression_l343_343770

theorem evaluate_expression :
  125^(1/3 : ℝ) * 81^(-1/4 : ℝ) * 32^(1/5 : ℝ) = 10/3 := by
  sorry

end evaluate_expression_l343_343770


namespace centroids_coincide_l343_343574

theorem centroids_coincide (A B C C₁ A₁ B₁ : Type)
  [metric_space A] [metric_space B] [metric_space C]
  (h1 : dist A C₁ = 2 * dist A B)
  (h2 : dist B A₁ = 2 * dist B C)
  (h3 : dist C B₁ = 2 * dist C A) :
  centroid ℝ [A, B, C] = centroid ℝ [A₁, B₁, C₁] :=
sorry

end centroids_coincide_l343_343574


namespace sum_of_possible_values_l343_343959

def f (x n : ℝ) : ℝ :=
  if x < n then x^3 + 3 else 3*x + 6

axiom continuity_condition (n : ℝ) :
  (λ x : ℝ, if x < n then x^3 + 3 else 3*x + 6) (n - 1) = 
  (λ x : ℝ, if x < n then x^3 + 3 else 3*x + 6) (n + 1)

theorem sum_of_possible_values : 
  let n := {n : ℝ | continuity_condition n} in
  ∑ n = 0 :=
sorry

end sum_of_possible_values_l343_343959


namespace geometric_relationships_l343_343099

variables {a b r : ℝ}
variables {A : ℝ × ℝ}
variables {l : ℝ × ℝ → Prop}

def circle (C : ℝ × ℝ → Prop) := ∀ p : ℝ × ℝ, C p ↔ p.1^2 + p.2^2 = r^2

def line (l : ℝ × ℝ → Prop) := ∀ p : ℝ × ℝ, l p ↔ a * p.1 + b * p.2 = r^2

def point_on_circle (C : ℝ × ℝ → Prop) (A : ℝ × ℝ) := C A
def point_inside_circle (C : ℝ × ℝ → Prop) (A : ℝ × ℝ) := A.1^2 + A.2^2 < r^2
def point_outside_circle (C : ℝ × ℝ → Prop) (A : ℝ × ℝ) := A.1^2 + A.2^2 > r^2
def point_on_line (l : ℝ × ℝ → Prop) (A : ℝ × ℝ) := l A

def tangent (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop) := 
  ∃ p : ℝ × ℝ, l p ∧ C p ∧ 
  (∃! q : ℝ × ℝ, C q ∧ l q)

def disjoint (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop) := 
  ∀ p : ℝ × ℝ, ¬ (l p ∧ C p)

theorem geometric_relationships (A : ℝ × ℝ) (h₀ : circle circle) (h₁ : line l) :
  (point_on_circle circle A → tangent l circle) ∧
  (point_inside_circle circle A → disjoint l circle) ∧
  (point_outside_circle circle A → ¬ disjoint l circle) ∧
  (point_on_line l A → tangent l circle) := 
sorry

end geometric_relationships_l343_343099


namespace max_right_angles_in_triangle_l343_343621

theorem max_right_angles_in_triangle (a b c : ℝ) (h : a + b + c = 180) (ha : a = 90 ∨ b = 90 ∨ c = 90) : a = 90 ∧ b ≠ 90 ∧ c ≠ 90 ∨ b = 90 ∧ a ≠ 90 ∧ c ≠ 90 ∨ c = 90 ∧ a ≠ 90 ∧ b ≠ 90 :=
sorry

end max_right_angles_in_triangle_l343_343621


namespace sum_a_b_range_l343_343763

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x^4)

theorem sum_a_b_range : let a := 0
                       let b := 3
                       a + b = 3 := by
  sorry

end sum_a_b_range_l343_343763


namespace smallest_value_fraction_l343_343995

theorem smallest_value_fraction (a b c : ℤ) (hc : c > 0) (ha : a = b + c) :
  ∃ k : ℝ, k > 0 ∧ (∀ a b : ℤ, a = b + c → (a - b ≠ 0) → ∃ (k = 2), ((a + b) / (a - b) + (a - b) / (a + b)) = k) :=
begin
  sorry
end

end smallest_value_fraction_l343_343995


namespace coefficient_of_x3_in_AB_l343_343423

def A (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + 3 * x + 4
def B (x : ℝ) : ℝ := 5 * x^2 + 7 * x + 6

theorem coefficient_of_x3_in_AB : 
  (A * B).coeff 3 = 47 :=
sorry

end coefficient_of_x3_in_AB_l343_343423


namespace num_even_pairs_in_set_l343_343937

theorem num_even_pairs_in_set {p q : ℕ} (h1 : p % 2 = 1) (h2 : q % 2 = 1) (hpq_coprime : Nat.coprime p q)
  (hp_gt1 : 1 < p) (hpq_order : p < q) (A : set (ℕ × ℕ))
  (hA : ∀ a b, 0 ≤ a ∧ a ≤ p - 2 → 0 ≤ b ∧ b ≤ q - 2 → 
    ( ( (a, b) ∈ A ∨ (a + 1, b + 1) ∈ A ) ∧ 
      ( (a, q - 1) ∈ A ∨ (a + 1, 0) ∈ A ) ∧ 
      ( (p - 1, b) ∈ A ∨ (0, b + 1) ∈ A ) )) :
  ∃ B : set (ℕ × ℕ), B ⊆ A ∧ (Nat.card B) ≥ (p - 1) * (q + 1) / 8
  ∧ ∀ (ab : ℕ × ℕ), ab ∈ B → ab.1 % 2 = 0 ∧ ab.2 % 2 = 0 :=
by
  sorry

end num_even_pairs_in_set_l343_343937


namespace speed_of_man_in_still_water_l343_343354

theorem speed_of_man_in_still_water 
  (v_m v_s : ℝ)
  (h1 : 32 = 4 * (v_m + v_s))
  (h2 : 24 = 4 * (v_m - v_s)) :
  v_m = 7 :=
by
  sorry

end speed_of_man_in_still_water_l343_343354


namespace students_in_class_C_l343_343722

theorem students_in_class_C 
    (total_students : ℕ := 80) 
    (percent_class_A : ℕ := 40) 
    (class_B_difference : ℕ := 21) 
    (h_percent : percent_class_A = 40) 
    (h_class_B_diff : class_B_difference = 21) 
    (h_total_students : total_students = 80) : 
    total_students - ((percent_class_A * total_students) / 100 - class_B_difference + (percent_class_A * total_students) / 100) = 37 := by
    sorry

end students_in_class_C_l343_343722


namespace number_of_excellent_tickets_l343_343565

def is_excellent_ticket (n : ℕ) : Prop :=
  (0 ≤ n ∧ n ≤ 999999) ∧
  ∃ (d1 d2 ... d6 : ℕ), (n = d1 * 100000 + d2 * 10000 + d3 * 1000 + d4 * 100 + d5 * 10 + d6) ∧
  (|d1 - d2| = 5 ∨ |d2 - d3| = 5 ∨ |d3 - d4| = 5 ∨ |d4 - d5| = 5 ∨ |d5 - d6| = 5)

theorem number_of_excellent_tickets : ∃ (k : ℕ), k = 409510 := sorry

end number_of_excellent_tickets_l343_343565


namespace actual_distance_travelled_l343_343124

theorem actual_distance_travelled :
  ∃ D : ℝ, (D / 12) = ((D + 30) / 18) ∧ D = 60 :=
begin
  use 60,
  split,
  { sorry },
  { refl },
end

end actual_distance_travelled_l343_343124


namespace find_angle_N_l343_343923

theorem find_angle_N (P Q R M N : Type) [HasAngle P Q R] [HasAngle Q N R]
  (PR RM RN NQ : LineSegment P R) (angleP angleQ angleN : ℝ) 
  (h1 : PR.length = RM.length) (h2 : RM.length = RN.length) 
  (h3 : RN.length = NQ.length) (h4 : angleP = 3 * angleQ) : 
  angleN = 75 := 
sorry

end find_angle_N_l343_343923


namespace negative_expression_l343_343378

theorem negative_expression : 
  (-3)^3 < 0 :=
by 
  have A : -(-2) - (-3) = 5 := by norm_num
  have B : (-2) * (-3) = 6 := by norm_num
  have C : (-2)^2 = 4 := by norm_num
  have D : (-3)^3 = -27 := by norm_num
  show (-3)^3 < 0
  rwa [D]

end negative_expression_l343_343378


namespace y_increasing_l343_343269

variable (f : ℝ → ℝ)

def domain := ∀ x : ℝ, x > 0 → True
def condition1 := ∀ x : ℝ, x > 0 → f(x) > 0
def condition2 := ∀ x : ℝ, x > 0 → deriv f x > 0

theorem y_increasing (h_dom : domain f) (h_cond1 : condition1 f) (h_cond2 : condition2 f) : 
  ∀ x : ℝ, x > 0 → deriv (λ x, x * f(x)) x > 0 := 
by 
  sorry

end y_increasing_l343_343269


namespace sin_alpha_minus_beta_l343_343815

theorem sin_alpha_minus_beta (α β : Real) 
  (h1 : Real.sin α = 12 / 13) 
  (h2 : Real.cos β = 4 / 5)
  (hα : π / 2 ≤ α ∧ α ≤ π)
  (hβ : -π / 2 ≤ β ∧ β ≤ 0) :
  Real.sin (α - β) = 33 / 65 := 
sorry

end sin_alpha_minus_beta_l343_343815


namespace exists_positive_x_eq_128_reciprocal_l343_343015

theorem exists_positive_x_eq_128_reciprocal : ∃ x : ℝ, 0 < x ∧ x + 8 = 128 * (1 / x) :=
by
  exists 8
  split
  rfl -- proof of 0 < 8
  sorry -- proof of the equation

end exists_positive_x_eq_128_reciprocal_l343_343015


namespace trig_identity_l343_343478

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ + Real.cos θ) / Real.sin θ + Real.sin θ * Real.sin θ = 23 / 10 :=
sorry

end trig_identity_l343_343478


namespace monotonicity_func_l343_343758

section
open Real

noncomputable def func (x : ℝ) (h : x ≠ 1) : ℝ := 2^(1/(x-1))

theorem monotonicity_func : 
  ∀ x y : ℝ, (x < y) ∧ (x < 1 ∨ 1 < x) ∧ (y < 1 ∨ 1 < y) → func x (by linarith) > func y (by linarith) :=
sorry

end

end monotonicity_func_l343_343758


namespace ratio_sum_odd_even_divisors_l343_343952

theorem ratio_sum_odd_even_divisors :
  let N := 48 * 48 * 55 * 125 * 81 in
  let sum_divisors (n : ℕ) := ∑ d in divisors n, d in
  let sum_odd_divisors := sum_divisors N - ∑ d in filter (λ d, d % 2 = 0) (divisors N), d in
  let sum_even_divisors := ∑ d in filter (λ d, d % 2 = 0) (divisors N), d in
  (sum_odd_divisors / sum_even_divisors) = (1 / 510) := sorry

end ratio_sum_odd_even_divisors_l343_343952


namespace problem1_problem2_l343_343597

-- Problem 1: Simplify and evaluate 
theorem problem1 (x : ℤ) (h : x = -1) :
  3*x^3 - (x^3 + (6*x^2 - 7*x)) - 2*(x^3 - 3*x^2 - 4*x) = -15 :=
by { 
  have h1 : x = -1 := h,
  sorry 
}

-- Problem 2: Simplify and evaluate
theorem problem2 (a b : ℤ) (ha : a = 2) (hb : b = 1) :
  2*(a*b^2 - 2*a^2*b) - 3*(a*b^2 - a^2*b) + (2*a*b^2 - 2*a^2*b) = -10 :=
by { 
  have h1 : a = 2 := ha,
  have h2 : b = 1 := hb,
  sorry 
}

end problem1_problem2_l343_343597


namespace lines_intersect_at_same_point_l343_343133

theorem lines_intersect_at_same_point (x y m : ℝ) :
  (y = 2 * x) ∧ (x + y = 3) ∧ (m * x - 2 * y - 5 = 0) → m = 9 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end lines_intersect_at_same_point_l343_343133


namespace part_I_part_II_l343_343861

noncomputable def f (x k : ℝ) : ℝ := (x - 1) * Real.exp(x) - k * x^2 + 2

theorem part_I (h : ∀ x, f x 0 ≥ f 0 0) : f 0 0 = 1 :=
by
  have H : f 0 0 = 1 := by sorry
  exact H

theorem part_II (h : ∀ x, x ∈ Set.Ici 0 → f x k ≥ 1) : k ≤ 1 / 2 :=
by
  have H : k ≤ 1 / 2 := by sorry
  exact H

end part_I_part_II_l343_343861


namespace hexagon_area_l343_343245

-- Define the regular hexagon and its properties
noncomputable def distance (A B : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def regular_hexagon_area (A C : ℝ × ℝ) : ℝ :=
  2 * (real.sqrt 3 / 4 * (distance A C)^2)

-- Define the vertices' coordinates
def A : (ℝ × ℝ) := (0, 0)
def C : (ℝ × ℝ) := (7, 1)

-- Proof goal
theorem hexagon_area :
  regular_hexagon_area A C = 25 * real.sqrt 3 := 
sorry

end hexagon_area_l343_343245


namespace average_speed_SanDiego_SanFrancisco_l343_343688

-- Define the various distances
def distance_SanDiego_LA := 120 -- miles
def distance_LA_SanJose := 340 -- miles
def distance_SanJose_SanFrancisco := 50 -- miles

-- Define speeds
def speed_SanDiego_LA := 51 -- miles per hour
def speed_LA_SanJose := 1.5 * speed_SanDiego_LA -- 50% faster
def speed_SanJose_SanFrancisco := 25 -- miles per hour

-- Define break times
def break_LA := 0.5 -- hours
def break_SanJose := 0.75 -- hours

-- Time calculations
def time_SanDiego_LA := distance_SanDiego_LA / speed_SanDiego_LA -- hours
def time_LA_SanJose := distance_LA_SanJose / speed_LA_SanJose -- hours
def time_SanJose_SanFrancisco := 2 * (distance_SanJose_SanFrancisco / speed_SanJose_SanFrancisco) -- hours

-- Total time taken including breaks
def total_time := time_SanDiego_LA + break_LA + time_LA_SanJose + break_SanJose + time_SanJose_SanFrancisco -- hours

-- Total distance traveled
def total_distance := distance_SanDiego_LA + distance_LA_SanJose + distance_SanJose_SanFrancisco -- miles

-- Calculate overall average speed
def overall_average_speed := total_distance / total_time -- miles per hour

-- State the theorem
theorem average_speed_SanDiego_SanFrancisco :
  overall_average_speed = 42.36 :=
by
  sorry

end average_speed_SanDiego_SanFrancisco_l343_343688


namespace combined_stickers_count_l343_343193

theorem combined_stickers_count :
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given
  june_total + bonnie_total = 189 :=
by
  -- Definitions
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  
  -- Calculations
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given

  -- Proof is omitted
  sorry

end combined_stickers_count_l343_343193


namespace equivalent_problem_statement_l343_343326

theorem equivalent_problem_statement :
  (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 1 → a^2 + b^2 ≥ 1 / 2) ∧
  (¬ (∀ (x : ℝ), x ≥ 0 → x^2 ≥ 0) ↔ ∃ (x : ℝ), x ≥ 0 ∧ x^2 < 0) ∧
  (∀ (f : ℝ → ℝ), (∀ (x : ℝ), -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 1 → -1 ≤ x ∧ x ≤ 3) → ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 3 → -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 1) ∧
  (∀ (x : ℝ), x ≥ -1 → ∃ y : ℝ, y = sqrt x - 1 ∧ f (sqrt x - 1) = x - 3 * sqrt x → f x = x^2 - x - 2)


end equivalent_problem_statement_l343_343326


namespace unit_square_congruent_divisions_number_of_valid_n_l343_343437

theorem unit_square_congruent_divisions : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 → ∃ (a b : ℕ), n = a * b :=
begin
  sorry
end

theorem number_of_valid_n : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 100 }.finite :=
begin
  sorry
end

end unit_square_congruent_divisions_number_of_valid_n_l343_343437


namespace parabola_focus_l343_343023

theorem parabola_focus :
  ∃ f, (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (-f + 1/4))^2)) ∧ f = 1/8 :=
by
  sorry

end parabola_focus_l343_343023


namespace geese_count_l343_343234

variables (k n : ℕ)

theorem geese_count (h1 : k * n = (k + 20) * (n - 75)) (h2 : k * n = (k - 15) * (n + 100)) : n = 300 :=
by
  sorry

end geese_count_l343_343234


namespace isabel_pop_albums_l343_343665

theorem isabel_pop_albums (total_songs : ℕ) (country_albums : ℕ) (songs_per_album : ℕ) (pop_albums : ℕ)
  (h1 : total_songs = 72)
  (h2 : country_albums = 4)
  (h3 : songs_per_album = 8)
  (h4 : total_songs - country_albums * songs_per_album = pop_albums * songs_per_album) :
  pop_albums = 5 :=
by
  sorry

end isabel_pop_albums_l343_343665


namespace solve_a_solve_b_solve_c_l343_343569

-- Definitions for part (a) 
def problem_a (n : ℕ) (m : ℕ → ℕ) : Prop :=
  n = 5 ∧ m 0 = 0 ∧ m 1 = 0 ∧ m 2 = 0 ∧ m 3 = 1 ∧ m 4 = 2

theorem solve_a : ∃ (n : ℕ) (m : ℕ → ℕ), problem_a n m ∧ ∃ i, m i < 0 :=
sorry

-- Definitions for part (b)
def k_n (n : ℕ) : ℕ := n^2 - 3 * n + 2

theorem solve_b (n : ℕ) (m : ℕ → ℕ) : 
  (m 0 + m 1 + ... + m (n-1)) ≥ k_n n → ∀ i, m i ≥ 0 :=
sorry

-- Definitions for part (c)
def problem_c (n : ℕ) (m : ℕ → ℕ) : Prop :=
  n = 5 ∧ ∃ avg, ∀ i, m i = avg

theorem solve_c : ∃ (n : ℕ) (m : ℕ → ℕ), problem_c n m :=
sorry

end solve_a_solve_b_solve_c_l343_343569


namespace ratio_F1F3_V1V3_l343_343944

-- Definitions for the given conditions and the required proof statement.
noncomputable def parabola_P (x: ℝ) := 4 * x^2

noncomputable def V1 : ℝ × ℝ := (0, 0)
noncomputable def F1 : ℝ × ℝ := (0, 1)

noncomputable def point_C (c: ℝ) : ℝ × ℝ := (c, 4 * c^2)
noncomputable def point_D (d: ℝ) : ℝ × ℝ := (d, 4 * d^2)

axiom angle_CV1D_eq_90 (c d : ℝ) (h : 4 * c * 4 * d = -1) : ∃ C D, C = point_C c ∧ D = point_D d ∧ ∠ C V1 D = 90

noncomputable def midpoint_locus (C D : ℝ × ℝ) :=
  let x := (C.1 + D.1) / 2
  let y := 2 * ((C.1 + D.1) / 2)^2 + 1/8
  (x, y)

noncomputable def V3 : ℝ × ℝ := (0, 1 / 8)
noncomputable def F3 : ℝ × ℝ := (0, (1 / 8 + 1 / 32))

theorem ratio_F1F3_V1V3 : (dist F1 F3) / (dist V1 V3) = -6.75 := 
  by sorry

end ratio_F1F3_V1V3_l343_343944


namespace evaluate_expression_l343_343783

theorem evaluate_expression:
  (125 = 5^3) ∧ (81 = 3^4) ∧ (32 = 2^5) → 
  125^(1/3) * 81^(-1/4) * 32^(1/5) = 10 / 3 := by
  sorry

end evaluate_expression_l343_343783


namespace find_minimal_sum_n_l343_343071

noncomputable def minimal_sum_n {a : ℕ → ℤ} {S : ℕ → ℤ} (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : ℕ := 
     5

theorem find_minimal_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : minimal_sum_n h1 h2 h3 = 5 :=
    sorry

end find_minimal_sum_n_l343_343071


namespace problem_equivalent_proof_l343_343400

noncomputable def sqrt (x : ℝ) := Real.sqrt x

theorem problem_equivalent_proof : ((sqrt 3 - 2) ^ 0 - Real.logb 2 (sqrt 2)) = 1 / 2 :=
by
  sorry

end problem_equivalent_proof_l343_343400


namespace number_of_ways_to_choose_a_pair_of_socks_same_color_l343_343512

theorem number_of_ways_to_choose_a_pair_of_socks_same_color
  (white black red green : ℕ) 
  (total_socks : ℕ)
  (h1 : white = 5)
  (h2 : black = 6)
  (h3 : red = 3)
  (h4 : green = 2)
  (h5 : total_socks = 16) :
  (nat.choose white 2) + (nat.choose black 2) + (nat.choose red 2) + (nat.choose green 2) = 29 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end number_of_ways_to_choose_a_pair_of_socks_same_color_l343_343512


namespace sum_of_repeating_decimals_correct_l343_343748

/-- Convert repeating decimals to fractions -/
def rep_dec_1 : ℚ := 1 / 9
def rep_dec_2 : ℚ := 2 / 9
def rep_dec_3 : ℚ := 1 / 3
def rep_dec_4 : ℚ := 4 / 9
def rep_dec_5 : ℚ := 5 / 9
def rep_dec_6 : ℚ := 2 / 3
def rep_dec_7 : ℚ := 7 / 9
def rep_dec_8 : ℚ := 8 / 9

/-- Define the terms in the sum -/
def term_1 : ℚ := 8 + rep_dec_1
def term_2 : ℚ := 7 + 1 + rep_dec_2
def term_3 : ℚ := 6 + 2 + rep_dec_3
def term_4 : ℚ := 5 + 3 + rep_dec_4
def term_5 : ℚ := 4 + 4 + rep_dec_5
def term_6 : ℚ := 3 + 5 + rep_dec_6
def term_7 : ℚ := 2 + 6 + rep_dec_7
def term_8 : ℚ := 1 + 7 + rep_dec_8

/-- Define the sum of the terms -/
def total_sum : ℚ := term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8

/-- Proof problem statement -/
theorem sum_of_repeating_decimals_correct : total_sum = 39.2 := 
sorry

end sum_of_repeating_decimals_correct_l343_343748


namespace find_k_l343_343848

noncomputable def f (x : ℝ) : ℝ := 3^(18 - x) - x

theorem find_k (x : ℝ) (k : ℕ) (h₀ : x > 0) (h₁ : x * 3^x = 3^18) (h₂ : k > 0) (h₃ : k < x ∧ x < k + 1) :
  k = 15 :=
begin
  sorry -- proof goes here
end

end find_k_l343_343848


namespace possible_values_p3_q3_r3_l343_343199

noncomputable def matrixN (p q r : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![[p, q, r], [q, r, p], [r, p, q]]

theorem possible_values_p3_q3_r3 (p q r : ℂ) (hN3 : (matrixN p q r) ^ 3 = 1) (hpqr : p * q * r = -1) : 
  p ^ 3 + q ^ 3 + r ^ 3 = -2 ∨ p ^ 3 + q ^ 3 + r ^ 3 = -4 := 
  sorry

end possible_values_p3_q3_r3_l343_343199


namespace six_digit_integers_count_l343_343498

theorem six_digit_integers_count :
  ∃ (n : ℕ), n = 60 ∧ n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) :=
by
  use 60
  split
  sorry
  sorry

end six_digit_integers_count_l343_343498


namespace maximum_a_condition_l343_343440

theorem maximum_a_condition :
  ∀ x ∈ set.Icc (1:ℝ) 12, x^2 + 25 + abs (x^3 - 5 * x^2) ≥ (10:ℝ) * x :=
by sorry

end maximum_a_condition_l343_343440


namespace remainder_sum_arithmetic_sequence_is_zero_l343_343317

theorem remainder_sum_arithmetic_sequence_is_zero :
  let a := 3
      d := 6
      n := 46
      sequence := List.range n
      sum_sequence := sequence.sum (λ k => a + k * d)
  in sum_sequence % 6 = 0 := by
  let a := 3
  let d := 6
  let n := 46
  let sequence := List.range n
  let sum_sequence := sequence.sum (λ k => a + k * d)
  sorry

end remainder_sum_arithmetic_sequence_is_zero_l343_343317


namespace sequence_formula_l343_343043

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 3^n) :
  ∀ n : ℕ, a n = (3^n - 1) / 2 :=
sorry

end sequence_formula_l343_343043


namespace minimum_area_of_triangle_MON_l343_343866

noncomputable def minimum_area_triangle_parabola : ℝ := 2

theorem minimum_area_of_triangle_MON
  (x y : ℝ) (h_parabola : x^2 = 4 * y)
  (h_focus : (0, 1))
  (line_through_focus : ∀ k, y = k * x + 1)
  (M N : ℝ × ℝ)
  (h_intersections : (M.1, M.2) = (N.1, N.2))
  (O : ℝ × ℝ := (0, 0)) :
  ∃ (S : ℝ), S = minimum_area_triangle_parabola ∧ S = 2 := 
sorry

end minimum_area_of_triangle_MON_l343_343866


namespace quadrilateral_not_exist_l343_343309

theorem quadrilateral_not_exist (a b c d : ℕ) (h₀ : set.insert a (set.insert b (set.insert c (set.singleton d))) = {1, 3, 4, 10}) :
  ¬∃ (p q r s : ℕ), p + q + r + s = 18 ∧ p + q > r ∧ p + r > q ∧ q + r > p :=
by
  sorry

end quadrilateral_not_exist_l343_343309


namespace general_term_formula_sum_of_first_n_terms_l343_343867

-- Definitions for the sequence
def a_seq : ℕ → ℝ 
| 0       := 2 -- Lean uses zero-based indexing for sequences
| (n + 1) := (3 - a_seq n) / 2

-- Theorem statements
theorem general_term_formula : ∀ n : ℕ, a_seq n = (- (1 / 2)) ^ n + 1 := sorry

theorem sum_of_first_n_terms : ∀ n : ℕ, (finset.range (n+1)).sum (λ i, a_seq i) = 
  (2 / 3) - (2 / 3) * (- (1 / 2)) ^ n + n := sorry

end general_term_formula_sum_of_first_n_terms_l343_343867


namespace percentage_increase_l343_343994

variables (J T P : ℝ)

def income_conditions (J T P : ℝ) : Prop :=
  (T = 0.5 * J) ∧ (P = 0.8 * J)

theorem percentage_increase (J T P : ℝ) (h : income_conditions J T P) :
  ((P / T) - 1) * 100 = 60 :=
by
  sorry

end percentage_increase_l343_343994


namespace mary_james_not_adjacent_l343_343963

open Finset

-- Define the set of chairs and relevant probabilities
def chairs := range 10

-- Define the event they sit next to each other
def adjacent_pairs : Finset (ℕ × ℕ) := 
  Finset.filter (λ (p : ℕ × ℕ), abs (p.1 - p.2) = 1) (chairs.product chairs)

def total_pairs : Finset (ℕ × ℕ) := 
  chairs.product chairs \ (chairs.product {x | x = 10 - 1})

-- Calculate the probability that they sit next to each other
def prob_adjacent : ℚ :=
  adjacent_pairs.card / total_pairs.card

noncomputable def prob_not_adjacent : ℚ :=
  1 - prob_adjacent

theorem mary_james_not_adjacent :
  prob_not_adjacent = 4 / 5 :=
by
  sorry

end mary_james_not_adjacent_l343_343963


namespace average_dandelions_picked_l343_343390

def billy_initial_pick : ℝ := 36
def george_initial_pick : ℝ := (2/5) * billy_initial_pick
def billy_additional_pick : ℝ := (5/3) * billy_initial_pick
def george_additional_pick : ℝ := (7/2) * george_initial_pick

-- Billy's total pick
def billy_total : ℝ := billy_initial_pick + billy_additional_pick
-- George's total pick
def george_total : ℝ := george_initial_pick + george_additional_pick
-- Combined total pick
def combined_total : ℝ := billy_total + george_total
-- Average pick
def average_pick : ℝ := combined_total / 2

theorem average_dandelions_picked :
  average_pick = 79.5 :=
by
  sorry

end average_dandelions_picked_l343_343390


namespace decreasing_interval_l343_343625

def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem decreasing_interval : {x : ℝ | x < 0} = {x : ℝ | ∀ y ∈ Iio x, f(y) > f(x)} :=
sorry

end decreasing_interval_l343_343625


namespace count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l343_343707

-- Setup the basic context
def Pocket := Finset (Fin 11)

-- The pocket contains 4 red balls and 7 white balls
def red_balls : Finset (Fin 11) := {0, 1, 2, 3}
def white_balls : Finset (Fin 11) := {4, 5, 6, 7, 8, 9, 10}

-- Question 1
theorem count_selection_4_balls :
  (red_balls.card.choose 4) + (red_balls.card.choose 3 * white_balls.card.choose 1) +
  (red_balls.card.choose 2 * white_balls.card.choose 2) = 115 := 
sorry

-- Question 2
theorem count_selection_5_balls_score_at_least_7_points :
  (red_balls.card.choose 2 * white_balls.card.choose 3) +
  (red_balls.card.choose 3 * white_balls.card.choose 2) +
  (red_balls.card.choose 4 * white_balls.card.choose 1) = 301 := 
sorry

end count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l343_343707


namespace find_point_D_l343_343066

def point (α : Type*) := (α × α)

variables {α : Type*} [field α] 

noncomputable def is_midpoint (A B M : point α) : Prop :=
  ∃ x0 y0 : α, M = (x0, y0) ∧ x0 = (A.1 + B.1) / 2 ∧ y0 = (A.2 + B.2) / 2

noncomputable def is_parallelogram (A B C D : point α) : Prop :=
  ∃ M : point α, is_midpoint B C M ∧ is_midpoint A D M

noncomputable def problem (A B C D M : point α) :=
  A = (-2, 1) ∧ B = (2, 5) ∧ C = (4, -1) ∧ M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧ 
  is_midpoint A D M ∧ is_parallelogram A B C D

theorem find_point_D (A B C D M : point ℚ) (h : problem A B C D M) : D = (8, 3) :=
sorry

end find_point_D_l343_343066


namespace circles_positional_relationship_l343_343488

noncomputable def circle1_eq : ℝ → ℝ → Prop := by 
  sorry

noncomputable def tangent_line := by
  sorry
  
noncomputable def circle2_eq : ℝ → ℝ → Prop := 
  λ x y, x^2 + y^2 - 2 * y - 3 = 0

theorem circles_positional_relationship : 
  ∃ r : ℝ, (r = sqrt 2) ∧ 
  ∀ x y, (circle1_eq x y) -> 
  ∃ h : ℝ → ℝ → Prop, h = circle2_eq →
  ∃ d, d = (1) → 
  (d < r + 2 ∧ d > r - 2) → (circle1_eq x y) ∧ (h x y) := 
by sorry

end circles_positional_relationship_l343_343488


namespace perpendicular_lines_to_parallel_planes_l343_343842

-- Define non-overlapping lines and planes in a 3D geometry space
variables {m n : line} {α β : plane}

-- Conditions:
-- m is a line
-- α and β are planes
-- m is perpendicular to α
-- m is perpendicular to β

-- To prove:
-- α is parallel to β

theorem perpendicular_lines_to_parallel_planes 
  (non_overlap_mn : m ≠ n) 
  (non_overlap_ab : α ≠ β) 
  (m_perp_α : m ⊥ α) 
  (m_perp_β : m ⊥ β) : parallel α β :=
sorry

end perpendicular_lines_to_parallel_planes_l343_343842


namespace intersection_point_M_perpendicular_line_l343_343853

def line1 (x y : ℝ) := 2 * x - 3 * y + 4 = 0
def line2 (x y : ℝ) := x + y - 3 = 0
def line3 (x y : ℝ) := x - 2 * y + 5 = 0
def pointM := (1, 2)

theorem intersection_point_M : ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ (x, y) = pointM :=
by {
  use [1, 2],
  split,
  use -3,
  split;
  sorry
}

theorem perpendicular_line : ∃ (x y : ℝ), l x y = 0 ∧ l (1 : ℝ) (2 : ℝ) :=
by {
  use [1, 2],
  split,
  use -3,
  split;
  sorry
}

end intersection_point_M_perpendicular_line_l343_343853


namespace transport_in_neg_20_repr_transport_out_20_l343_343146

theorem transport_in_neg_20_repr_transport_out_20
  (out_recording : ∀ x : ℝ, transporting_out x → recording (-x))
  (in_recording  : ∀ x : ℝ, transporting_in x → recording x) :
  recording (-(-20)) = recording (20) := by
  sorry

end transport_in_neg_20_repr_transport_out_20_l343_343146


namespace number_of_trees_l343_343339

theorem number_of_trees (initial_trees planted_trees : ℕ)
  (h1 : initial_trees = 13)
  (h2 : planted_trees = 12) :
  initial_trees + planted_trees = 25 := by
  sorry

end number_of_trees_l343_343339


namespace systematic_sampling_number_in_range_l343_343903

theorem systematic_sampling_number_in_range
  (N : ℕ) (k : ℕ) (a₁ : ℕ)
  (total_population : N = 960)
  (sample_size : k = 32)
  (first_draw : a₁ = 9)
  (sampling_interval : ℕ := N / k)
  (common_difference : ℕ := sampling_interval)
  (general_term : ℕ → ℕ := λ n, a₁ + (n - 1) * common_difference)
  (group_range : set ℕ := {x : ℕ | 401 ≤ x ∧ x ≤ 430})
  (n_value : ℕ := 15) :
  general_term n_value = 429 :=
begin
  have sampling_interval_correct : sampling_interval = 30, by sorry,
  have general_term_correct : general_term n_value = 30 * n_value - 21, by sorry,
  have n_in_group_range : 14 ≤ n_value ∧ n_value ≤ 15, by sorry,
  exact sorry,
end

end systematic_sampling_number_in_range_l343_343903


namespace coeff_x8_in_expansion_l343_343860

open BigOperators

theorem coeff_x8_in_expansion : 
  ∃ (c : ℕ), c = (6.choose 4) ∧ c = 15 := by
  have h1 : (1 + x^2)^6 = (∑ k in range (6 + 1), (6.choose k) * x^(2 * k)) := sorry,
  let coeff_x8 : ℕ := (6.choose 4),
  use coeff_x8,
  split,
  { exact rfl },
  { exact rfl }

end coeff_x8_in_expansion_l343_343860


namespace proof_expr1_proof_expr2_l343_343747

noncomputable def expr1 : ℝ :=
  32^4 + (27 / 64)^(1/3) - 2014^0

theorem proof_expr1 : expr1 = 2^32 - (1/4) :=
by
  sorry

noncomputable def expr2 : ℝ :=
  log10(0.1) + log (Real.sqrt (Real.exp 1)) + 3^(1 + log 2 / log 3)

theorem proof_expr2 : expr2 = 11 / 2 :=
by
  sorry

end proof_expr1_proof_expr2_l343_343747


namespace quadratic_sum_constants_l343_343285

-- Define the quadratic expression
def quadratic (x : ℝ) : ℝ := -3 * x^2 + 27 * x + 135

-- Define the representation of the quadratic in the form a(x + b)^2 + c
def quadratic_rewritten (a b c : ℝ) (x : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum_constants :
  ∃ a b c, (∀ x, quadratic x = quadratic_rewritten a b c x) ∧ a + b + c = 197.75 :=
by
  sorry

end quadratic_sum_constants_l343_343285


namespace scientific_notation_of_dna_diameter_l343_343614

def dna_diameter : ℝ := 2.01 * 10 ^ (-7)

theorem scientific_notation_of_dna_diameter :
  (0.000000201 : ℝ) = dna_diameter :=
by
  sorry

end scientific_notation_of_dna_diameter_l343_343614


namespace seq_initial_term_l343_343041

open Nat

noncomputable def sequence (A : ℕ → ℝ) : ℕ → ℝ := λ n, A (n+1) - A n

theorem seq_initial_term (A : ℕ → ℝ)
  (h1 : ∀ n, sequence (sequence A) n = 1)
  (h2 : A 19 = 0)
  (h3 : A 92 = 0) :
  A 1 = 819 :=
sorry

end seq_initial_term_l343_343041


namespace previous_monthly_income_l343_343969

variable (I : ℝ)

-- Conditions from the problem
def condition1 (I : ℝ) : Prop := 0.40 * I = 0.25 * (I + 600)

theorem previous_monthly_income (h : condition1 I) : I = 1000 := by
  sorry

end previous_monthly_income_l343_343969


namespace equation_one_solution_equation_two_solution_l343_343992

theorem equation_one_solution (x : ℝ) : ((x + 3) ^ 2 - 9 = 0) ↔ (x = 0 ∨ x = -6) := by
  sorry

theorem equation_two_solution (x : ℝ) : (x ^ 2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := by
  sorry

end equation_one_solution_equation_two_solution_l343_343992


namespace complement_intersection_l343_343106

-- Define the universal set U.
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set M.
def M : Set ℕ := {2, 3}

-- Define the set N.
def N : Set ℕ := {1, 3}

-- Define the complement of set M in U.
def complement_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- Define the complement of set N in U.
def complement_U_N : Set ℕ := {x ∈ U | x ∉ N}

-- The statement to be proven.
theorem complement_intersection :
  (complement_U_M ∩ complement_U_N) = {4, 5, 6} :=
sorry

end complement_intersection_l343_343106


namespace no_real_solutions_for_equation_l343_343991

theorem no_real_solutions_for_equation : ¬ (∃ x : ℝ, x + Real.sqrt (2 * x - 6) = 5) :=
sorry

end no_real_solutions_for_equation_l343_343991


namespace find_extra_page_number_l343_343627

theorem find_extra_page_number (n k: ℕ) (h1: n = 77) (h2: ∑ i in finset.range (n+1), i = 3003) (h3: 3003 + 2 * k = 3028) : k = 25 :=
by
  rw h1 at h2
  rw finset.sum_range_eq
  rw finset.sum_range_eq 77 at h2
  sorry

end find_extra_page_number_l343_343627


namespace exists_invisible_square_l343_343361

def invisible (p q : ℤ) : Prop := Int.gcd p q > 1

theorem exists_invisible_square (n : ℤ) (h : 0 < n) : 
  ∃ (a b : ℤ), ∀ i j : ℤ, (0 ≤ i) ∧ (i < n) ∧ (0 ≤ j) ∧ (j < n) → invisible (a + i) (b + j) :=
by {
  sorry
}

end exists_invisible_square_l343_343361


namespace area_ratio_half_l343_343940

open Classical

/-- Mathematics problem statement in Lean 4 without proof -/
noncomputable def area_ratio_BPD_to_ABC (A B C M P D : Point) (h_midpoint : Midpoint A B M) (h_P_on_AB : Collinear A B P) 
(h_P_between_A_M : Point_between A P M) (h_MD_parallel_PC : Parallel (Line M D) (Line P C)) 
(h_MD_intersect_BC_at_D : Intersect (Line M D) (Line B C) D) : Real :=
r

theorem area_ratio_half (A B C M P D : Point) (h_midpoint : Midpoint A B M) (h_P_on_AB : Collinear A B P) 
(h_P_between_A_M : Point_between A P M) (h_MD_parallel_PC : Parallel (Line M D) (Line P C)) 
(h_MD_intersect_BC_at_D : Intersect (Line M D) (Line B C) D) : area_ratio_BPD_to_ABC A B C M P D h_midpoint h_P_on_AB h_P_between_A_M h_MD_parallel_PC h_MD_intersect_BC_at_D = 1 / 2 := by 
  sorry

end area_ratio_half_l343_343940


namespace digit_difference_base4_base5_l343_343111

theorem digit_difference_base4_base5 (n : ℕ) (h : n = 1024) :
  (nat.log 4 n).toNat - (nat.log 5 n).toNat = 0 :=
by
  sorry

end digit_difference_base4_base5_l343_343111


namespace cos_trig_identity_l343_343833

theorem cos_trig_identity (α : Real) 
  (h : Real.cos (Real.pi / 6 - α) = 3 / 5) : 
  Real.cos (5 * Real.pi / 6 + α) = - (3 / 5) :=
by
  sorry

end cos_trig_identity_l343_343833


namespace P_2001_gt_1_over_64_l343_343281

variable (a b c : ℝ)
variable (P : ℝ → ℝ := λ x, x^3 + a * x^2 + b * x + c)
noncomputable variable (Q : ℝ → ℝ := λ x, x^2 + x + 2001)

def has_three_distinct_real_roots (P : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  P(x₁) = 0 ∧ P(x₂) = 0 ∧ P(x₃) = 0

def P_of_Q_has_no_real_roots (P : ℝ → ℝ) (Q : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P(Q(x)) ≠ 0

theorem P_2001_gt_1_over_64 (h1 : has_three_distinct_real_roots P)
    (h2 : P_of_Q_has_no_real_roots P Q) : 
    P(2001) > 1 / 64 :=
by
  sorry

end P_2001_gt_1_over_64_l343_343281


namespace fourth_number_in_pascals_triangle_row_15_l343_343540

theorem fourth_number_in_pascals_triangle_row_15 : (Nat.choose 15 3) = 455 :=
by sorry

end fourth_number_in_pascals_triangle_row_15_l343_343540


namespace problem_statement_l343_343561

theorem problem_statement (p q m n : ℕ) (x : ℚ)
  (h1 : p / q = 4 / 5) (h2 : m / n = 4 / 5) (h3 : x = 1 / 7) :
  x + (2 * q - p + 3 * m - 2 * n) / (2 * q + p - m + n) = 71 / 105 :=
by
  sorry

end problem_statement_l343_343561


namespace max_unmarried_women_l343_343971

theorem max_unmarried_women (total_people : ℕ) (frac_women : ℚ) (frac_married : ℚ)
  (h_total : total_people = 80) (h_frac_women : frac_women = 1 / 4) (h_frac_married : frac_married = 3 / 4) :
  ∃ (max_unmarried_women : ℕ), max_unmarried_women = 20 :=
by
  -- The proof will be filled here
  sorry

end max_unmarried_women_l343_343971


namespace prob_black_third_no_replacement_prob_black_third_with_replacement_xi_distribution_expectation_l343_343526

-- Define basic conditions
def total_balls := 10
def black_balls := 6
def white_balls := 4
def draws := 3

-- Question 1: Without Replacement
theorem prob_black_third_no_replacement
  (first_white : bool := true)
  (draws := {3, without_replacement}) :
  (first_white → (P(black_on_third_draw) = 2 / 3))
:=
sorry

-- Question 2: With Replacement
theorem prob_black_third_with_replacement
  (first_white : bool := true)
  (draws := {3, with_replacement}) :
  (first_white → (P(black_on_third_draw) = 3 / 5))
:=
sorry

-- Question 3: Distribution and Expectation of White Balls Drawn
def xi_distribution := pmf.binomial 3 (2 / 5)

theorem xi_distribution_expectation :
  (pmf.expectation xi_distribution = 6 / 5)
:=
sorry

end prob_black_third_no_replacement_prob_black_third_with_replacement_xi_distribution_expectation_l343_343526


namespace tom_watching_days_l343_343647

def show_a_season_1_time : Nat := 20 * 22
def show_a_season_2_time : Nat := 18 * 24
def show_a_season_3_time : Nat := 22 * 26
def show_a_season_4_time : Nat := 15 * 30

def show_b_season_1_time : Nat := 24 * 42
def show_b_season_2_time : Nat := 16 * 48
def show_b_season_3_time : Nat := 12 * 55

def show_c_season_1_time : Nat := 10 * 60
def show_c_season_2_time : Nat := 13 * 58
def show_c_season_3_time : Nat := 15 * 50
def show_c_season_4_time : Nat := 11 * 52
def show_c_season_5_time : Nat := 9 * 65

def show_a_total_time : Nat :=
  show_a_season_1_time + show_a_season_2_time +
  show_a_season_3_time + show_a_season_4_time

def show_b_total_time : Nat :=
  show_b_season_1_time + show_b_season_2_time + show_b_season_3_time

def show_c_total_time : Nat :=
  show_c_season_1_time + show_c_season_2_time +
  show_c_season_3_time + show_c_season_4_time +
  show_c_season_5_time

def total_time : Nat := show_a_total_time + show_b_total_time + show_c_total_time

def daily_watch_time : Nat := 120

theorem tom_watching_days : (total_time + daily_watch_time - 1) / daily_watch_time = 64 := sorry

end tom_watching_days_l343_343647


namespace percentage_discount_of_retail_price_l343_343368

theorem percentage_discount_of_retail_price {wp rp sp discount : ℝ} (h1 : wp = 99) (h2 : rp = 132) (h3 : sp = wp + 0.20 * wp) (h4 : discount = (rp - sp) / rp * 100) : discount = 10 := 
by 
  sorry

end percentage_discount_of_retail_price_l343_343368


namespace average_employees_per_week_l343_343692

theorem average_employees_per_week 
  (E2 : ℕ) -- Number of employees hired in the second week
  (h1 : ∀ E1 : ℕ, E1 = E2 + 200) -- Number of employees hired in the first week is 200 more than in the second week
  (h3 : ∀ E3 : ℕ, E2 = E3 - 150) -- Number of employees hired in the second week is 150 fewer than in the third week
  (h4 : ∀ E4 : ℕ, E4 = 2 * E3) -- Number of employees hired in the fourth week is twice the number hired in the third week
  (E4 : ℕ = 400) -- Number of employees hired in the fourth week is 400
: (250 + 50 + 200 + 400) / 4 = 225 :=
by sorry

end average_employees_per_week_l343_343692


namespace sum_infinite_geometric_series_l343_343415

theorem sum_infinite_geometric_series : 
  let a : ℝ := 2
  let r : ℝ := -5/8
  a / (1 - r) = 16/13 :=
by
  sorry

end sum_infinite_geometric_series_l343_343415


namespace num_three_digit_multiples_of_3_l343_343798

def digits : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

def is_valid_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (∀ d ∈ digits, d ∈ {n / 100, (n / 10) % 10, n % 10}) ∧ (n / 100 ∈ digits ∧ (n / 10) % 10 ∈ digits ∧ n % 10 ∈ digits)

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem num_three_digit_multiples_of_3 : 
  { n : ℕ | is_valid_three_digit n ∧ is_multiple_of_3 n }.card = 106 :=
sorry

end num_three_digit_multiples_of_3_l343_343798


namespace geometric_relationships_l343_343101

variables {a b r : ℝ}
variables {A : ℝ × ℝ}
variables {l : ℝ × ℝ → Prop}

def circle (C : ℝ × ℝ → Prop) := ∀ p : ℝ × ℝ, C p ↔ p.1^2 + p.2^2 = r^2

def line (l : ℝ × ℝ → Prop) := ∀ p : ℝ × ℝ, l p ↔ a * p.1 + b * p.2 = r^2

def point_on_circle (C : ℝ × ℝ → Prop) (A : ℝ × ℝ) := C A
def point_inside_circle (C : ℝ × ℝ → Prop) (A : ℝ × ℝ) := A.1^2 + A.2^2 < r^2
def point_outside_circle (C : ℝ × ℝ → Prop) (A : ℝ × ℝ) := A.1^2 + A.2^2 > r^2
def point_on_line (l : ℝ × ℝ → Prop) (A : ℝ × ℝ) := l A

def tangent (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop) := 
  ∃ p : ℝ × ℝ, l p ∧ C p ∧ 
  (∃! q : ℝ × ℝ, C q ∧ l q)

def disjoint (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop) := 
  ∀ p : ℝ × ℝ, ¬ (l p ∧ C p)

theorem geometric_relationships (A : ℝ × ℝ) (h₀ : circle circle) (h₁ : line l) :
  (point_on_circle circle A → tangent l circle) ∧
  (point_inside_circle circle A → disjoint l circle) ∧
  (point_outside_circle circle A → ¬ disjoint l circle) ∧
  (point_on_line l A → tangent l circle) := 
sorry

end geometric_relationships_l343_343101


namespace chromatic_number_bound_l343_343957

variables {G : SimpleGraph V} {n : ℕ} (V : Type)

noncomputable def chromatic_number_le_n_plus_one : Prop :=
  ∀ (G : SimpleGraph V), (∀ (e : G.edge_set), e.card ≤ n) → (G.chromatic_number ≤ n + 1)

theorem chromatic_number_bound (h₁ : 2 ≤ n) (h₂ : ∀ (e : G.edge_set), e.card ≤ n) : 
  (G.chromatic_number ≤ n + 1) :=
begin
  sorry
end

end chromatic_number_bound_l343_343957


namespace evaluate_expression_l343_343012

theorem evaluate_expression : (3 / (2 - (4 / (-5)))) = (15 / 14) :=
by
  sorry

end evaluate_expression_l343_343012


namespace find_line_l_l343_343476

-- Define the equations of the lines l1 and l2
def l1 (x y : ℝ) := 4 * x + y + 6 = 0
def l2 (x y : ℝ) := 3 * x - 5 * y - 6 = 0

-- Define the midpoint condition
def midpoint_condition (x1 y1 x2 y2 : ℝ) := (x1 + x2 = 0) ∧ (y1 + y2 = 0)

theorem find_line_l (x1 y1 x2 y2 : ℝ) (h_mid : midpoint_condition x1 y1 x2 y2)
                     (h1 : l1 x1 y1) (h2 : l2 x2 y2) :
  y1 = (6:ℝ) / 23 ∧ x1 = -(36:ℝ) / 23 → (∀ x y, y = - (1/6) * x) :=
begin
  intros h_coords,
  sorry -- Proof steps would go here
end

end find_line_l_l343_343476


namespace find_a_for_tangent_parallel_to_line_l343_343609

theorem find_a_for_tangent_parallel_to_line :
  ∃ a : ℝ, (∀ (x : ℝ), deriv (λ x, a * x^2 + real.log x) 1 = 2) → a = 1 / 2 := 
begin
  use 1 / 2,
  intro h,
  -- Provided condition that the derivative at x=1 must equal 2
  have h_slope := h 1,
  -- Simplifying the derivative at x = 1 shows the value of a must be 1/2
  sorry
end

end find_a_for_tangent_parallel_to_line_l343_343609


namespace inverse_proportion_l343_343663

theorem inverse_proportion {x y : ℝ} :
  (y = (3 / x)) -> ¬(y = x / 3) ∧ ¬(y = 3 / (x + 1)) ∧ ¬(y = 3 * x) :=
by
  sorry

end inverse_proportion_l343_343663


namespace coloring_problem_l343_343401

def hexagon_coloring : Nat :=
  32

theorem coloring_problem (c1 c2 c3 : ∀ (r c: Nat), c ∈ [0,1,2] → c ≠ r ∧ ¬ (r = 1 ∧ c = 0)) :
  (∃ (coloring_scheme : ∀ (row col: Nat), (row ≤ 2 ∧ col ≤ row) → (row = 0 ∧ col = 0) → coloring_scheme 0 0 = c1 ∨ coloring_scheme 1 0 ∈ [1, 2] ∨ coloring_scheme 1 1 ∈ [1, 2] ∨ coloring_scheme 2 0 ∈ [1, 2] ∨ coloring_scheme 2 2 ∈ [2, 1]),
    32) :=
sorry

end coloring_problem_l343_343401


namespace current_age_of_son_l343_343348

variables (S F : ℕ)

-- Define the conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F - 8 = 4 * (S - 8)

-- The theorem statement
theorem current_age_of_son (h1 : condition1 S F) (h2 : condition2 S F) : S = 24 :=
sorry

end current_age_of_son_l343_343348


namespace parallel_planes_if_perpendicular_to_same_line_l343_343844

variables {m n : Type} [line m] [line n]
variables {α β : Type} [plane α] [plane β]

theorem parallel_planes_if_perpendicular_to_same_line (h1 : m ⟂ α) (h2 : m ⟂ β) : α ∥ β :=
sorry

end parallel_planes_if_perpendicular_to_same_line_l343_343844


namespace charlie_snowballs_l343_343750

theorem charlie_snowballs (lucy_snowballs charlie_more_than_lucy : ℕ) 
  (h1 : lucy_snowballs = 19)
  (h2 : charlie_more_than_lucy = 31) :
  charlie_snowballs = lucy_snowballs + charlie_more_than_lucy :=
by 
  sorry

end charlie_snowballs_l343_343750


namespace probability_intervals_l343_343896

variable (ξ : ℝ → ℝ)
variable (σ : ℝ)
variable (hσ : σ > 0)

def normal_distribution (μ σ : ℝ) (ξ : ℝ → ℝ) : Prop := sorry -- definition of normal distribution

-- Conditions
axiom h1 : normal_distribution 1 σ ξ
axiom h2 : ∫ x in 0..1, ξ x = 0.4

-- Proof goal
theorem probability_intervals (hσ : σ > 0) (h1 : normal_distribution 1 σ ξ) (h2 : ∫ x in 0..1, ξ x = 0.4) : 
  ∫ x in 0..2, ξ x = 0.8 := by
  sorry

end probability_intervals_l343_343896


namespace rotated_ACB_angle_l343_343922

-- Definitions based on conditions
def original_angle_ACB : ℝ := 30
def rotation_angle : ℝ := 450

-- Lean 4 statement to prove the problem
theorem rotated_ACB_angle (original_angle_ACB = 30) (rotation_angle = 450) :
  rotated_angle (original_angle_ACB, rotation_angle) = 60 :=
sorry


end rotated_ACB_angle_l343_343922


namespace total_water_capacity_of_coolers_l343_343188

theorem total_water_capacity_of_coolers :
  ∀ (first_cooler second_cooler third_cooler : ℕ), 
  first_cooler = 100 ∧ 
  second_cooler = first_cooler + first_cooler / 2 ∧ 
  third_cooler = second_cooler / 2 -> 
  first_cooler + second_cooler + third_cooler = 325 := 
by
  intros first_cooler second_cooler third_cooler H
  cases' H with H1 H2
  cases' H2 with H3 H4
  sorry

end total_water_capacity_of_coolers_l343_343188


namespace number_of_solutions_eq_43_l343_343503

theorem number_of_solutions_eq_43 :
  let num_zeros := ((1 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (1 : ℝ)) ∧
                   ((2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (2 : ℝ)) ∧
                   ((3 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (3 : ℝ)) ∧
                   -- ... and so on up to 50
                   ((50 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (50 : ℝ))
  in let denom_zeros := ((1^2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (1^2 : ℝ)) ∧
                        ((2^2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (2^2: ℝ)) ∧
                        ((3^2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (3^2: ℝ)) ∧
                        -- ... and so on up to 7^2
                        ((7^2: ℕ) ∈ finset.range (50 + 1)).count (λ x, (7^2: ℝ))
  in num_zeros - denom_zeros = 43 :=
sorry

end number_of_solutions_eq_43_l343_343503


namespace max_value_F_l343_343220

def F (x y : ℝ) : ℝ :=
  min (2 ^ -x) (min (2 ^ (x - y)) (2 ^ (y - 1)))

theorem max_value_F :
  ∃ x y, (0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ F x y = (2 : ℝ) ^ (-1 / 3) :=
  sorry

end max_value_F_l343_343220


namespace regular_polygon_properties_l343_343366

theorem regular_polygon_properties (perimeter side_length : ℝ) (h_perimeter : perimeter = 150) (h_side_length : side_length = 10) :
  let n := (perimeter / side_length) in
  n = 15 ∧ ((n - 2) * 180 / n = 156) :=
by
  -- Let n be the number of sides of the polygon
  let n := (perimeter / side_length)
  -- Show that n = 15
  have h_n : n = 15, by {
    rw [h_perimeter, h_side_length],
    exact div_self (by norm_num),
  },
  -- Show that the internal angle is 156 degrees
  have h_angle : ((n - 2) * 180 / n = 156), by {
    rw h_n,
    norm_num,
  },
  -- Conclude
  exact ⟨h_n, h_angle⟩,

end regular_polygon_properties_l343_343366


namespace card_sum_to_8_probability_l343_343683

def sum_is_8_probability {α : Type} (A B : set ℕ) (cards : set ℕ) (prob : ℚ) : Prop :=
  (∀ a ∈ A, a ∈ cards) ∧ (∀ b ∈ B, b ∈ cards) ∧
  (cards = {1, 2, 3, 4, 5, 6, 7}) ∧
  (A = cards) ∧ (B = cards) ∧
  prob = 1 / 7

theorem card_sum_to_8_probability :
  sum_is_8_probability ({1, 2, 3, 4, 5, 6, 7}) ({1, 2, 3, 4, 5, 6, 7}) {1, 2, 3, 4, 5, 6, 7} (1 / 7) :=
by
  sorry

end card_sum_to_8_probability_l343_343683


namespace fg_minus_gf_eq_4_l343_343996

def f (x : ℝ) := 6 * x - 9
def g (x : ℝ) := x / 3 + 2

theorem fg_minus_gf_eq_4 (x : ℝ) : f(g(x)) - g(f(x)) = 4 := by
    sorry

end fg_minus_gf_eq_4_l343_343996


namespace parabola_midpoint_trajectory_l343_343831

theorem parabola_midpoint_trajectory :
  ∀ (P Q : ℝ × ℝ) (x y : ℝ), 
    (P ∈ (λ x, (x, 1/4 * x^2)) '' univ) ∧ Q = ((x y):ℝ × ℝ) ∧
    (Q.fst = P.fst / 2) ∧ (Q.snd = (P.snd + 1) / 2) →
    Q.fst^2 = 2 * Q.snd - 1 := by
  -- The exact proof would follow here
  sorry

end parabola_midpoint_trajectory_l343_343831


namespace centroid_of_S_is_M_l343_343951

variables {n : ℕ} (vertices : fin n → ℝ × ℝ) (M : ℝ × ℝ)
          (S : fin n → ℝ × ℝ)
          (r : ℝ)

def sum_of_distances_minimized (M : ℝ × ℝ) : Prop :=
  ∀ N : ℝ × ℝ, (finset.sum finset.univ (λ i, (dist M (vertices i)))) ≤ 
              (finset.sum finset.univ (λ i, (dist N (vertices i))))

def points_S (M : ℝ × ℝ) (vertices : fin n → ℝ × ℝ) (r : ℝ) : fin n → ℝ × ℝ :=
  λ i, let (xi, yi) := vertices i in let (x, y) := M in
       let θ := real.atan2 (yi - y) (xi - x) in
       (x + r * real.cos θ, y + r * real.sin θ)

theorem centroid_of_S_is_M (h_min : sum_of_distances_minimized vertices M)
                           (h_S : S = points_S M vertices r) :
  let centroid := (1 / n) • finset.univ.sum S in centroid = M :=
sorry

end centroid_of_S_is_M_l343_343951


namespace variance_is_stability_measure_l343_343644

def stability_measure (yields : Fin 10 → ℝ) : Prop :=
  let mean := (yields 0 + yields 1 + yields 2 + yields 3 + yields 4 + yields 5 + yields 6 + yields 7 + yields 8 + yields 9) / 10
  let variance := 
    ((yields 0 - mean)^2 + (yields 1 - mean)^2 + (yields 2 - mean)^2 + (yields 3 - mean)^2 + 
     (yields 4 - mean)^2 + (yields 5 - mean)^2 + (yields 6 - mean)^2 + (yields 7 - mean)^2 + 
     (yields 8 - mean)^2 + (yields 9 - mean)^2) / 10
  true -- just a placeholder, would normally state that this is the appropriate measure

theorem variance_is_stability_measure (yields : Fin 10 → ℝ) : stability_measure yields :=
by 
  sorry

end variance_is_stability_measure_l343_343644


namespace number_of_multiples_of_3003_l343_343500

theorem number_of_multiples_of_3003 (i j : ℕ) (h : 0 ≤ i ∧ i < j ∧ j ≤ 199): 
  (∃ n : ℕ, n = 3003 * k ∧ n = 10^j - 10^i) → 
  (number_of_solutions = 1568) :=
sorry

end number_of_multiples_of_3003_l343_343500


namespace parabola_focus_l343_343024

theorem parabola_focus :
  ∃ f, (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (-f + 1/4))^2)) ∧ f = 1/8 :=
by
  sorry

end parabola_focus_l343_343024


namespace contagious_positive_integers_l343_343308

def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_digits 10).sum

def sum_of_digits_seq (k : ℕ) : ℕ :=
  (list.range' k 1000).sum sum_of_digits

theorem contagious_positive_integers (N : ℕ) :
  (∃ k : ℕ, sum_of_digits_seq k = N) ↔ ∃ n : ℕ, N = 13500 + 1000 * n :=
by
  sorry

end contagious_positive_integers_l343_343308


namespace class_committee_selection_l343_343341

theorem class_committee_selection :
  let members := ["A", "B", "C", "D", "E"]
  let admissible_entertainment_candidates := ["C", "D", "E"]
  ∃ (entertainment : String) (study : String) (sports : String),
    entertainment ∈ admissible_entertainment_candidates ∧
    study ∈ members.erase entertainment ∧
    sports ∈ (members.erase entertainment).erase study ∧
    (3 * 4 * 3 = 36) :=
sorry

end class_committee_selection_l343_343341


namespace polynomial_four_distinct_real_roots_a_l343_343881

theorem polynomial_four_distinct_real_roots_a (a : ℝ) : 
  (∃ (f : ℝ → ℝ), f = (λ x, x^4 + 8*x^3 + 18*x^2 + 8*x + a) ∧
   (∀ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 →
    f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0)) ↔ a ∈ Ioo (-8 : ℝ) 1 := sorry

end polynomial_four_distinct_real_roots_a_l343_343881


namespace option_A_option_B_option_C_option_D_l343_343098

section
variables {a b r : ℝ}
def line := {p : ℝ × ℝ | a * p.1 + b * p.2 - r^2 = 0}
def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
def point (x y : ℝ) := (x, y)

theorem option_A (h : a^2 + b^2 = r^2) : ∀ p ∈ circle, p = point a b → ∀ l ∈ line, tangent l circle :=
sorry

theorem option_B (h : a^2 + b^2 < r^2) : ∀ l ∈ line, disjoint l circle :=
sorry

theorem option_C (h : a^2 + b^2 > r^2) : ∃ l ∈ line, intersects l circle :=
sorry

theorem option_D (h_on_l : a * a + b * b - r^2 = 0) (h : a^2 + b^2 = r^2) : ∀ p ∈ point a b, ∀ l ∈ line, tangent l circle :=
sorry
end

end option_A_option_B_option_C_option_D_l343_343098


namespace point_in_fourth_quadrant_l343_343156

theorem point_in_fourth_quadrant (x : ℝ) (y : ℝ) (hx : x = 8) (hy : y = -3) : x > 0 ∧ y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l343_343156


namespace coolers_total_capacity_l343_343187

theorem coolers_total_capacity :
  ∃ (C1 C2 C3 : ℕ), 
    C1 = 100 ∧ 
    C2 = C1 + (C1 / 2) ∧ 
    C3 = C2 / 2 ∧ 
    (C1 + C2 + C3 = 325) :=
sorry

end coolers_total_capacity_l343_343187


namespace no_prime_degree_measure_l343_343224

theorem no_prime_degree_measure :
  ∀ n, 10 ≤ n ∧ n < 20 → ¬ Nat.Prime (180 * (n - 2) / n) :=
by
  intros n h1 h2 
  sorry

end no_prime_degree_measure_l343_343224


namespace product_of_digits_of_specific_number_l343_343576

def is_not_divisible_by_5 (n : ℕ) : Prop :=
  n % 5 ≠ 0

def units_digit (n : ℕ) : ℕ :=
  n % 10

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem product_of_digits_of_specific_number :
  ∃ n : ℕ, (n = 3546 ∨ n = 3550 ∨ n = 3565 ∨ n = 3570 ∨ n = 3585) ∧ is_not_divisible_by_5 n ∧
  units_digit n * tens_digit n = 24 :=
by {
  use 3546,
  split,
  norm_num,
  split,
  norm_num,
  norm_num
}

end product_of_digits_of_specific_number_l343_343576


namespace range_BE_CD_proof_l343_343168

noncomputable def range_BE_CD (A B C : ℝ) (a b c : ℝ) (D E : ℝ) : Set ℝ :=
  { x : ℝ | x = (BE(B, C)/CD(B, C)) ∧ 
  (∃ (A B C : ℝ) (a b c : ℝ) (D E : ℝ), in_triangle a b c A B C D E ∧ 
  2 * sin C = 3 * sin B)}

theorem range_BE_CD_proof (a b c : ℝ) (A B C : ℝ) (D E : ℝ)
  (h_bc : side_correspondence a A b B c C)
  (h_midpoints : midpoints_of_sides D B E C)
  (h_sine : 2 * Real.sin C = 3 * Real.sin B) :
  range_BE_CD A B C a b c D E = Set.Ioo (8/7) 4 :=
sorry

end range_BE_CD_proof_l343_343168


namespace max_sum_15x15_grid_l343_343907

theorem max_sum_15x15_grid : ∀ (f : Fin 15 → Fin 15 → ℕ), 
  (∀ i j, f i j ≤ 4) ∧ 
  (∀ i j, i < 14 → j < 14 → f i j + f (i + 1) j + f i (j + 1) + f (i + 1) (j + 1) = 7) →
  (∑ i j, f i j) ≤ 417 :=
by
  intros f h
  sorry

end max_sum_15x15_grid_l343_343907


namespace solution_set_for_rational_inequality_l343_343034

theorem solution_set_for_rational_inequality (x : ℝ) :
  (x - 2) / (x - 1) > 0 ↔ x < 1 ∨ x > 2 := 
sorry

end solution_set_for_rational_inequality_l343_343034


namespace count_odd_numbers_between_150_and_350_l343_343112

theorem count_odd_numbers_between_150_and_350 : 
  ∃ (a l d : ℕ), a = 151 ∧ l = 349 ∧ d = 2 ∧ ((l - a) / d + 1 = 100) :=
by {
  use [151, 349, 2],
  split; exact rfl,
  split; exact rfl,
  split; exact rfl,
  calc (349 - 151) / 2 + 1 = (198) / 2 + 1 : by norm_num
                        ... = 99 + 1 : by norm_num
                        ... = 100 : by norm_num
}

end count_odd_numbers_between_150_and_350_l343_343112


namespace pluck_percentage_l343_343965

def feathers_per_flamingo := 20
def boas_needed := 12
def feathers_per_boa := 200
def flamingoes_harvested := 480
def percentage_to_pluck (f_need f_avail : ℕ) := (f_need : ℝ) / (f_avail : ℝ) * 100

theorem pluck_percentage : percentage_to_pluck (boas_needed * feathers_per_boa) (flamingoes_harvested * feathers_per_flamingo) = 25 := 
by
  sorry

end pluck_percentage_l343_343965


namespace find_c_l343_343890

theorem find_c (a b c d y1 y2 : ℝ) (h1 : y1 = a * 2^3 + b * 2^2 + c * 2 + d)
  (h2 : y2 = a * (-2)^3 + b * (-2)^2 + c * (-2) + d)
  (h3 : y1 - y2 = 12) : c = 3 - 4 * a := by
  sorry

end find_c_l343_343890


namespace part1_solution_set_part2_range_of_a_l343_343453

namespace Part1
def f1 (x : ℝ) : ℝ := 2 * |x + 2| - 2 * |x|

theorem part1_solution_set : {x : ℝ | f1 x > 2} = Ioi (-1/2) := 
sorry
end Part1

namespace Part2
def f2 (x : ℝ) (a : ℝ) : ℝ := 2 * |x + 2| - |a * x|

theorem part2_range_of_a : (∀ x : ℝ, (-1 < x ∧ x < 1) → f2 x a > x + 1) → (a > -2 ∧ a < 2) := 
sorry
end Part2

end part1_solution_set_part2_range_of_a_l343_343453


namespace sandbox_area_combined_approx_l343_343932

noncomputable def solve_sandbox_area : Real :=
  let w1 := 5 -- width of the first sandbox
  let l1 := 2 * w1 -- length of the first sandbox
  let w2 := Real.sqrt(22.5) -- width of the second sandbox
  let l2 := 3 * w2 -- length of the second sandbox
  let area1 := l1 * w1 -- area of the first sandbox
  let area2 := l2 * w2 -- area of the second sandbox
  area1 + area2 -- combined area

theorem sandbox_area_combined_approx :
  solve_sandbox_area ≈ 117.42 :=
by
  sorry

end sandbox_area_combined_approx_l343_343932


namespace sum_seven_smallest_nine_multiples_eq_l343_343319

def sum_of_seven_smallest_multiples_of_nine : ℕ := 252

theorem sum_seven_smallest_nine_multiples_eq : 
  (∑ i in Finset.range 7, 9 * (i + 1)) = 252 := 
by
  sorry

end sum_seven_smallest_nine_multiples_eq_l343_343319


namespace arithmetic_series_first_term_l343_343807

theorem arithmetic_series_first_term 
  (a d : ℚ)
  (h1 : 15 * (2 * a +  29 * d) = 450)
  (h2 : 15 * (2 * a + 89 * d) = 1650) :
  a = -13 / 3 :=
by
  sorry

end arithmetic_series_first_term_l343_343807


namespace mean_of_other_two_l343_343444

theorem mean_of_other_two (a b c d e f : ℕ) (h : a = 1867 ∧ b = 1993 ∧ c = 2019 ∧ d = 2025 ∧ e = 2109 ∧ f = 2121):
  ((a + b + c + d + e + f) - (4 * 2008)) / 2 = 2051 := by
  sorry

end mean_of_other_two_l343_343444


namespace area_triangle_RSH_eq_12_5_l343_343918

-- Define the necessary points and relationships
variables (P Q R S H K : Type) [IsScaleneTriangle P Q R]
  (SH_perp_QR : SH ⟂ QR) (RK_perp_PR : RK ⟂ PR)
  (S_on_PR : S ∈ PR) (H_on_QR : H ∈ QR) (K_on_PR : K ∈ PR)
  (PS_eq_SR : P S = R S) (angle_Q_gt_90 : angle PQR > 90)
  (area_PQR : area_triangle P Q R = 50) (length_PR : distance P R = 20)

-- Prove that the area of triangle RSH is 12.5
theorem area_triangle_RSH_eq_12_5 : 
  area_triangle R S H = 12.5 := 
sorry

end area_triangle_RSH_eq_12_5_l343_343918


namespace correct_percentage_is_69048_l343_343905

-- Assume total questions and correct answers
variables (total_questions correct_answers : ℕ)
axiom total_eq : total_questions = 84
axiom correct_eq : correct_answers = 58

-- Define the fraction of correct answers
def fraction_correct (total_questions correct_answers : ℕ) : ℚ := correct_answers / total_questions

-- Define the percentage of correct answers
def percentage_correct (total_questions correct_answers : ℕ) : ℚ :=
  fraction_correct total_questions correct_answers * 100

-- State the theorem
theorem correct_percentage_is_69048 :
  percentage_correct total_questions correct_answers = 69.048 :=
by
  -- Use the given conditions
  rw [total_eq, correct_eq]
  sorry

end correct_percentage_is_69048_l343_343905


namespace fitness_center_ratio_l343_343740

theorem fitness_center_ratio 
  (m f : ℕ) 
  (h_female_avg : 35 * f) 
  (h_male_avg : 30 * m) 
  (h_total_avg : (35 * f + 30 * m) / (f + m) = 32) :
  f / m = 2 / 3 :=
sorry

end fitness_center_ratio_l343_343740


namespace number_of_correct_statements_l343_343730

-- Define the statements
def statement_1 : Prop := ∀ (a : ℚ), |a| < |0| → a = 0
def statement_2 : Prop := ∃ (b : ℚ), ∀ (c : ℚ), b < 0 ∧ b ≥ c → c = b
def statement_3 : Prop := -4^6 = (-4) * (-4) * (-4) * (-4) * (-4) * (-4)
def statement_4 : Prop := ∀ (a b : ℚ), a + b = 0 → a ≠ 0 → b ≠ 0 → (a / b = -1)
def statement_5 : Prop := ∀ (c : ℚ), (0 / c = 0 ↔ c ≠ 0)

-- Define the overall proof problem
theorem number_of_correct_statements : (statement_1 ∧ statement_4) ∧ ¬(statement_2 ∨ statement_3 ∨ statement_5) :=
by
  sorry

end number_of_correct_statements_l343_343730


namespace matrix_cubed_l343_343397

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -2], ![2, -1]]

theorem matrix_cubed :
  (A * A * A) = ![![ -4, 2], ![-2, 1]] :=
by
  sorry

end matrix_cubed_l343_343397


namespace max_value_l343_343958

theorem max_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 2) : 
  2 * x * y + 2 * y * z * Real.sqrt 3 ≤ 4 :=
sorry

end max_value_l343_343958


namespace slope_line_CD_l343_343005

-- Define the two given circles
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 12 = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 18 * x + 16 * y + 30 = 0

-- Define the line equation resulting from the subtraction of the two circle equations
def line_eq (x y : ℝ) : Prop := x - y = -7 / 2

-- Proof statement: The line defined as a result of the system of circles has a slope of 1
theorem slope_line_CD : ∀ (x y : ℝ), circle1_eq x y → circle2_eq x y → line_eq x y ∧ line_eq x y = (1 : ℝ) :=
by
  intros x y h1 h2
  -- The statements to be filled accordingly
  sorry

end slope_line_CD_l343_343005


namespace factories_unchecked_l343_343417

theorem factories_unchecked (total_factories checked_by_first_group checked_by_second_group : ℕ) 
  (h1 : total_factories = 259) 
  (h2 : checked_by_first_group = 105) 
  (h3 : checked_by_second_group = 87) : 
  total_factories - (checked_by_first_group + checked_by_second_group) = 67 :=
by
  rw [h1, h2, h3]
  norm_num

end factories_unchecked_l343_343417


namespace round_to_nearest_hundredth_l343_343246

noncomputable def recurring_decimal (n : ℕ) : ℝ :=
  if n = 87 then 87 + 36 / 99 else 0 -- Defines 87.3636... for n = 87

theorem round_to_nearest_hundredth : recurring_decimal 87 = 87.36 :=
by sorry

end round_to_nearest_hundredth_l343_343246


namespace find_a_l343_343105

open Set

def universal_set (a : ℝ) : Set ℝ := {1, 2, a^2 + 2*a - 3}
def set_A (a : ℝ) : Set ℝ := {|a-2|, 2}
def complement_U_A (a : ℝ) : Set ℝ := {0}

theorem find_a : ∀ (a : ℝ), 
  0 ∈ universal_set a →
  1 ∈ set_A a →
  a = 1 :=
by {
  intros,
  -- 0 ∈ universal_set a → a^2 + 2*a - 3 = 0
  -- 1 ∈ set_A a → |a-2| = 1
  sorry
}

end find_a_l343_343105


namespace point_in_fourth_quadrant_l343_343153

def x : ℝ := 8
def y : ℝ := -3

theorem point_in_fourth_quadrant (h1 : x > 0) (h2 : y < 0) : (x > 0 ∧ y < 0) :=
by {
  sorry
}

end point_in_fourth_quadrant_l343_343153


namespace find_k_eq_3_l343_343210

-- Definitions based on the conditions
noncomputable def M (A B : ℝ^3) : ℝ^3 := (A + B) / 2
noncomputable def squared_distance (X Y : ℝ^3) : ℝ := (X - Y).dot (X - Y)

-- The given conditions and assumptions
variables (A B C Q : ℝ^3)

-- The theorem to prove
theorem find_k_eq_3 :
  squared_distance Q A + squared_distance Q B + squared_distance Q C = 
  3 * squared_distance Q (M A B) + squared_distance (M A B) A + squared_distance (M A B) B + squared_distance (M A B) C :=
by
  sorry

end find_k_eq_3_l343_343210


namespace squares_area_relation_l343_343739

/-- 
Given:
1. $\alpha$ such that $\angle 1 = \angle 2 = \angle 3 = \alpha$
2. The areas of the squares are given by:
   - $S_A = \cos^4 \alpha$
   - $S_D = \sin^4 \alpha$
   - $S_B = \cos^2 \alpha \sin^2 \alpha$
   - $S_C = \cos^2 \alpha \sin^2 \alpha$

Prove that:
$S_A \cdot S_D = S_B \cdot S_C$
--/

theorem squares_area_relation (α : ℝ) :
  (Real.cos α)^4 * (Real.sin α)^4 = (Real.cos α)^2 * (Real.sin α)^2 * (Real.cos α)^2 * (Real.sin α)^2 :=
by sorry

end squares_area_relation_l343_343739


namespace number_of_solutions_eq_43_l343_343504

theorem number_of_solutions_eq_43 :
  let num_zeros := ((1 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (1 : ℝ)) ∧
                   ((2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (2 : ℝ)) ∧
                   ((3 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (3 : ℝ)) ∧
                   -- ... and so on up to 50
                   ((50 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (50 : ℝ))
  in let denom_zeros := ((1^2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (1^2 : ℝ)) ∧
                        ((2^2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (2^2: ℝ)) ∧
                        ((3^2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (3^2: ℝ)) ∧
                        -- ... and so on up to 7^2
                        ((7^2: ℕ) ∈ finset.range (50 + 1)).count (λ x, (7^2: ℝ))
  in num_zeros - denom_zeros = 43 :=
sorry

end number_of_solutions_eq_43_l343_343504


namespace max_area_of_triangle_ABC_l343_343535

noncomputable def maxAreaTriangleABC : ℕ :=
  let QA := 5
  let QB := 12
  let QC := 13
  let BC := 10
  let QH := 12  -- calculated from the altitude from Q to BC
  let h := QH + QA
  let area := 1/2 * h * BC
  85

theorem max_area_of_triangle_ABC :
  ∀ (QA QB QC BC : ℕ), 
  QA = 5 → QB = 12 → QC = 13 → BC = 10 → 
  maxAreaTriangleABC = 85 := 
by
  intros
  simp [maxAreaTriangleABC]
  sorry

end max_area_of_triangle_ABC_l343_343535


namespace problem_1_problem_2_problem_3_l343_343521

-- Define the function f
def f (x : ℝ) := -x * |x| + 2 * x

-- Definition of a Gamma interval
def gamma_interval (f : ℝ → ℝ) (a b : ℝ) :=
  (∀ (x ∈ set.Icc a b), set.Icc (inv b) (inv a) = set.image f (set.Icc a b))

-- Problem 1: Proving the interval is not a Gamma interval
theorem problem_1 : ¬ gamma_interval f (1/2) (3/2) := sorry

-- Problem 2: Finding the Gamma interval within [1, +∞)
theorem problem_2 : gamma_interval f 1 ((1 + Real.sqrt 5) / 2) := sorry

-- Problem 3: Existence of a real number a
theorem problem_3 : ¬ ∃ a : ℝ, ∀ x : ℝ, f x = (1 / 4) * x^2 + a → (set.finset.count (↑((-∞, ∞)) ∩ {x | f x = (1 / 4) * x^2 + a}) set.Ico 2) := sorry

end problem_1_problem_2_problem_3_l343_343521


namespace find_I_l343_343920

def letters := fin 10

-- Define the variables
variables (F I V T E N : letters)

-- State the conditions
axiom distinct_digits : F ≠ I ∧ F ≠ V ∧ F ≠ T ∧ F ≠ E ∧ F ≠ N ∧ I ≠ V ∧ I ≠ T ∧ I ≠ E ∧ I ≠ N ∧ V ≠ T ∧ V ≠ E ∧ V ≠ N ∧ T ≠ E ∧ T ≠ N ∧ E ≠ N
axiom F_value : F = 8
axiom V_is_even : V = 0 ∨ V = 2 ∨ V = 4 ∨ V = 6 ∨ V = 8

theorem find_I : I = 3 :=
by {
  sorry
}

end find_I_l343_343920


namespace range_of_k_l343_343127

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

def is_not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ c d ∈ set.Ioo a b, f c < f d ∧ f d < f c

theorem range_of_k :
  {k : ℝ | ∃ x ∈ set.Ioo (k - 1) (k + 1), f' x = 0}. = set.Ioo (-3 : ℝ) (-1 : ℝ) ∪ set.Ioo (1 : ℝ) (3 : ℝ) :=
sorry

end range_of_k_l343_343127


namespace number_of_integers_satisfying_condition_l343_343030

-- Define the condition
def condition1 (n : ℤ) : Prop := 25 < n^2
def condition2 (n : ℤ) : Prop := n^2 < 144

-- The math proof problem: Prove that the number of integers n such that 25 < n^2 < 144 is 12.
theorem number_of_integers_satisfying_condition : 
  {n : ℤ | condition1 n ∧ condition2 n}.toFinset.card = 12 :=
by sorry

end number_of_integers_satisfying_condition_l343_343030


namespace geom_mean_of_sqrt2_minus_1_and_sqrt2_plus_1_eq_pm1_l343_343272

theorem geom_mean_of_sqrt2_minus_1_and_sqrt2_plus_1_eq_pm1 :
  ∃ x : ℝ, (x^2 = (real.sqrt 2 - 1) * (real.sqrt 2 + 1)) → (x = 1 ∨ x = -1) :=
by
  sorry

end geom_mean_of_sqrt2_minus_1_and_sqrt2_plus_1_eq_pm1_l343_343272


namespace num_possible_sums_is_9_l343_343637

open Classical

noncomputable def num_possible_sums : ℕ :=
  let balls := {1, 2, 3, 4, 5}
  let sums := {x + y | x y : ℕ, x ∈ balls, y ∈ balls}
  sums.to_finset.card

theorem num_possible_sums_is_9 : num_possible_sums = 9 :=
by
  sorry

end num_possible_sums_is_9_l343_343637


namespace lattice_points_on_hyperbola_l343_343708

theorem lattice_points_on_hyperbola :
  (finset.filter (λ p : ℤ × ℤ, p.1 ^ 2 - p.2 ^ 2 = 2000 ^ 2) (finset.Icc (-2000) 2000).product (finset.Icc (-2000) 2000)).card = 98 := 
sorry

end lattice_points_on_hyperbola_l343_343708


namespace algebraic_expression_value_l343_343451

-- Define the given condition as a predicate
def condition (a : ℝ) := a^2 + a - 4 = 0

-- Then the goal to prove with the given condition
theorem algebraic_expression_value (a : ℝ) (h : condition a) : (a^2 - 3) * (a + 2) = -2 :=
sorry

end algebraic_expression_value_l343_343451


namespace value_of_y_l343_343514

theorem value_of_y : (∃ y : ℝ, (1 / 3 - 1 / 4 = 4 / y) ∧ y = 48) :=
by
  sorry

end value_of_y_l343_343514


namespace gasoline_price_percentage_increase_l343_343273

theorem gasoline_price_percentage_increase 
  (price_month1_euros : ℝ) (price_month3_dollars : ℝ) (exchange_rate : ℝ) 
  (price_month1 : ℝ) (percent_increase : ℝ):
  price_month1_euros = 20 →
  price_month3_dollars = 15 →
  exchange_rate = 1.2 →
  price_month1 = price_month1_euros * exchange_rate →
  percent_increase = ((price_month1 - price_month3_dollars) / price_month3_dollars) * 100 →
  percent_increase = 60 :=
by intros; sorry

end gasoline_price_percentage_increase_l343_343273


namespace area_inside_quadrilateral_BCDE_outside_circle_l343_343219

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3) / 2 * side_length ^ 2

noncomputable def circle_area (radius : ℝ) : ℝ :=
  Real.pi * radius ^ 2

theorem area_inside_quadrilateral_BCDE_outside_circle :
  let side_length := 2
  let hex_area := hexagon_area side_length
  let hex_area_large := hexagon_area (2 * side_length)
  let circle_radius := 3
  let circle_area_A := circle_area circle_radius
  let total_area_of_interest := hex_area_large - circle_area_A
  let area_of_one_region := total_area_of_interest / 6
  area_of_one_region = 4 * Real.sqrt 3 - (3 / 2) * Real.pi :=
by
  sorry

end area_inside_quadrilateral_BCDE_outside_circle_l343_343219


namespace radius_of_new_circle_l343_343649

theorem radius_of_new_circle :
  let r1 := 17
  let r2 := 25
  let shaded_area := (Math.pi * (r2 ^ 2)) - (Math.pi * (r1 ^ 2))
  let new_circle_area := shaded_area
  let radius := Real.sqrt (shaded_area / Math.pi)
  radius = 4 * Real.sqrt 21 :=
by
  -- Let the necessary conditions
  let r1 := 17
  let r2 := 25
  let shaded_area := (Math.pi * (r2 ^ 2)) - (Math.pi * (r1 ^ 2))
  let new_circle_area := shaded_area
  let radius := Real.sqrt (shaded_area / Math.pi)

  -- Prove the statement
  sorry

end radius_of_new_circle_l343_343649


namespace integral_linearity_product_rule_log_product_rule_log_exp_identity_l343_343416

variable (f g : ℝ → ℝ)
variable (b x y : ℝ)

-- Conditions
-- Assuming f and g are differentiable functions.
-- Assuming b is a positive real number not equal to 1.
-- Assuming derivatives and logarithms are defined properly.
axiom h_diff_f : Differentiable ℝ f
axiom h_diff_g : Differentiable ℝ g
axiom h_b_pos : b > 0
axiom h_b_ne_one : b ≠ 1

-- Statements to prove:
-- Statement 1
theorem integral_linearity : (∫ x, (f x + g x)) = (∫ x, f x) + (∫ x, g x) :=
sorry

-- Statement 2
theorem product_rule : (deriv (λ x, f x * g x)) = (λ x, deriv f x * g x + f x * deriv g x) :=
sorry

-- Statement 3
theorem log_product_rule : log b (x * y) = log b x + log b y :=
sorry

-- Statement 4
theorem log_exp_identity : b^(log b x) = x :=
sorry

end integral_linearity_product_rule_log_product_rule_log_exp_identity_l343_343416


namespace price_of_72_cans_l343_343629

theorem price_of_72_cans (regular_price_per_can : ℝ) (discount_percentage : ℝ) :
  regular_price_per_can = 0.60 →
  discount_percentage = 0.20 →
  72 * (regular_price_per_can - (regular_price_per_can * discount_percentage)) = 34.56 :=
begin
  intros h1 h2,
  rw [h1, h2],
  norm_num,
end

end price_of_72_cans_l343_343629


namespace range_of_a_l343_343857

noncomputable def complex_first_quadrant (a : ℝ) : Prop := 
  let z := (1 - complex.I * a) * (a + 2 * complex.I) in
  z.re > 0 ∧ z.im > 0

theorem range_of_a (a : ℝ) : complex_first_quadrant a ↔ 0 < a ∧ a < real.sqrt 2 :=
by
  sorry

end range_of_a_l343_343857


namespace evaluate_expression_l343_343769

theorem evaluate_expression :
  125^(1/3 : ℝ) * 81^(-1/4 : ℝ) * 32^(1/5 : ℝ) = 10/3 := by
  sorry

end evaluate_expression_l343_343769


namespace perpendicular_lines_to_parallel_planes_l343_343841

-- Define non-overlapping lines and planes in a 3D geometry space
variables {m n : line} {α β : plane}

-- Conditions:
-- m is a line
-- α and β are planes
-- m is perpendicular to α
-- m is perpendicular to β

-- To prove:
-- α is parallel to β

theorem perpendicular_lines_to_parallel_planes 
  (non_overlap_mn : m ≠ n) 
  (non_overlap_ab : α ≠ β) 
  (m_perp_α : m ⊥ α) 
  (m_perp_β : m ⊥ β) : parallel α β :=
sorry

end perpendicular_lines_to_parallel_planes_l343_343841


namespace icosagon_diagonals_from_vertex_l343_343641

theorem icosagon_diagonals_from_vertex (n : ℕ) (h : n = 20) :
  let diagonals_per_vertex := n - 1 - 2 in 
  diagonals_per_vertex = 17 :=
begin
  dsimp only [diagonals_per_vertex],
  rw h,
  norm_num,
end

end icosagon_diagonals_from_vertex_l343_343641


namespace frog_paths_to_814_l343_343350

-- Defining the problem's conditions and statement
def is_valid_move (x y : ℕ) : Prop :=
  ¬ (x % 2 = 1 ∧ y % 2 = 1)

def number_of_paths (start goal : ℕ × ℕ) : ℕ :=
  if start ≠ (0, 0) ∨ goal ≠ (8, 14) then 0 else
  let steps := (goal.1 / 2, goal.2 / 2) in
  nat.choose (steps.1 + steps.2) steps.1

theorem frog_paths_to_814 : number_of_paths (0, 0) (8, 14) = 330 :=
by sorry

end frog_paths_to_814_l343_343350


namespace sqrt_288_simplified_l343_343252

-- Define the conditions
def factorization_288 : ℕ := 288
def perfect_square_144 : ℕ := 144
def sqrt_144 : ℕ := 12

-- The proof goal
theorem sqrt_288_simplified :
  sqrt factorization_288 = sqrt perfect_square_144 * sqrt 2 :=
by
  sorry

end sqrt_288_simplified_l343_343252


namespace sqrt_expression_evaluation_l343_343414

theorem sqrt_expression_evaluation (a b c d : ℝ) (h₁ : 5 + 4 * real.sqrt 3 = a)
  (h₂ : 5 - 4 * real.sqrt 3 = b) :
  (real.sqrt a - real.sqrt b)^2 = 10 + 2 * complex.I * real.sqrt 23 :=
by
  sorry

end sqrt_expression_evaluation_l343_343414


namespace projectile_reaches_30m_at_2_seconds_l343_343362

theorem projectile_reaches_30m_at_2_seconds:
  ∀ t : ℝ, -5 * t^2 + 25 * t = 30 → t = 2 ∨ t = 3 :=
by
  sorry

end projectile_reaches_30m_at_2_seconds_l343_343362


namespace root_of_f_in_interval_l343_343618

-- Define the function f
def f (x : ℝ) : ℝ := 3^x + 2 * x - 3

-- State the theorem to prove
theorem root_of_f_in_interval : ∃ x ∈ Ioo 0 1, f x = 0 :=
sorry

end root_of_f_in_interval_l343_343618


namespace angle_bisector_median_ineq_l343_343223

variables {A B C : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (l_a l_b l_c m_a m_b m_c : ℝ)

theorem angle_bisector_median_ineq
  (hl_a : l_a > 0) (hl_b : l_b > 0) (hl_c : l_c > 0)
  (hm_a : m_a > 0) (hm_b : m_b > 0) (hm_c : m_c > 0) :
  l_a / m_a + l_b / m_b + l_c / m_c > 1 :=
sorry

end angle_bisector_median_ineq_l343_343223


namespace driver_net_rate_of_pay_is_25_l343_343694

noncomputable def net_rate_of_pay_per_hour (hours_traveled : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) (fuel_cost_per_gallon : ℝ) : ℝ :=
  let total_distance := speed * hours_traveled
  let total_fuel_used := total_distance / fuel_efficiency
  let total_earnings := pay_per_mile * total_distance
  let total_fuel_cost := fuel_cost_per_gallon * total_fuel_used
  let net_earnings := total_earnings - total_fuel_cost
  net_earnings / hours_traveled

theorem driver_net_rate_of_pay_is_25 :
  net_rate_of_pay_per_hour 3 50 25 0.6 2.5 = 25 := sorry

end driver_net_rate_of_pay_is_25_l343_343694


namespace harmonic_inequality_l343_343936

theorem harmonic_inequality (n : ℕ) (h : n ≥ 2) : 
  (1 : ℝ) / (n + 1) * (∑ i in range (2 * n - 1), if i % 2 = 0 then 0 else 1 / (i + 1)) > 
  (1 : ℝ) / n * ∑ i in range (2 * n), if i % 2 = 1 then 0 else 1 / (i + 2) := 
sorry

end harmonic_inequality_l343_343936


namespace sin_one_lt_log3_sqrt7_l343_343667

open Real

theorem sin_one_lt_log3_sqrt7 : sin 1 < log 3 (sqrt 7) := 
sorry

end sin_one_lt_log3_sqrt7_l343_343667


namespace sum_of_ages_twins_l343_343013

-- Define that Evan has two older twin sisters and their ages are such that the product of all three ages is 162
def twin_sisters_ages (a : ℕ) (b : ℕ) (c : ℕ) : Prop :=
  a * b * c = 162

-- Given the above definition, we need to prove the sum of these ages is 20
theorem sum_of_ages_twins (a b c : ℕ) (h : twin_sisters_ages a b c) (ha : b = c) : a + b + c = 20 :=
by 
  sorry

end sum_of_ages_twins_l343_343013


namespace simplified_form_addition_l343_343391

theorem simplified_form_addition :
  ∃ c d : ℕ, (c > 0 ∧ d > 0 ∧ (∑ i in finset.range 4, c * d ^ (i + 1)) = 138) ∧ 
  ∀ x y : ℕ, (x = 3 ∧ y = 135) → (x + y = 138) :=
sorry

end simplified_form_addition_l343_343391


namespace union_sets_l343_343869

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_sets : A ∪ B = {x | -1 < x ∧ x < 2} := by
  sorry

end union_sets_l343_343869


namespace impossible_curve_l343_343821

-- Defining vertices as Cartesian coordinates
structure Vertex := (x : ℕ) (y : ℕ)

def A : Vertex := ⟨1, 1⟩
def B : Vertex := ⟨2, 1⟩
def C : Vertex := ⟨3, 1⟩
def D : Vertex := ⟨1, 2⟩
def E : Vertex := ⟨2, 2⟩
def F : Vertex := ⟨3, 2⟩
def G : Vertex := ⟨4, 2⟩
def H : Vertex := ⟨5, 2⟩
def I : Vertex := ⟨1, 3⟩
def J : Vertex := ⟨2, 3⟩
def K : Vertex := ⟨4, 3⟩
def L : Vertex := ⟨5, 3⟩

-- Defining edges as pairs of vertices
def Edge := (v1 : Vertex) (v2 : Vertex)

def e1 : Edge := (A, B)
def e2 : Edge := (B, C)
def e3 : Edge := (B, E)
def e4 : Edge := (E, F)
def e5 : Edge := (F, G)
def e6 : Edge := (G, H)
def e7 : Edge := (A, D)
def e8 : Edge := (C, F)
def e9 : Edge := (D, E)
def e10 : Edge := (E, J)
def e11 : Edge := (J, K)
def e12 : Edge := (K, L)
def e13 : Edge := (I, J)
def e14 : Edge := (I, D)
def e15 : Edge := (H, L)
def e16 : Edge := (J, K)

-- Defining regions based on the vertices and edges
structure Region := (vertices : list Vertex) (edges : list Edge)

def R1 : Region := { vertices := [A, B, F, E, D], edges := [e1, e3, e4, e9, e7] }
def R2 : Region := { vertices := [B, C, H, G, F], edges := [e2, e6, e5, e4, e3] }
def R3 : Region := { vertices := [E, F, G, K, J], edges := [e10, e11, e5, e4, e16] }
def OutsideRegion : Region := { vertices := [A, B, C, H, L, K, J, I, D], edges := [e1, e2, e6, e15, e12, e11, e16, e13, e14] }

-- The proof problem as a theorem
theorem impossible_curve :
  ¬ ∃ (curve : Vertex → Prop),
    -- No vertex is crossed
    (∀ v : Vertex, ¬ curve v) ∧
    -- Each edge is intersected exactly once
    (∀ e : Edge, curve e.1 ≠ curve e.2) :=
sorry -- Proof here

end impossible_curve_l343_343821


namespace eval_expression_l343_343782

theorem eval_expression : (125 ^ (1/3) * 81 ^ (-1/4) * 32 ^ (1/5) = (10/3)) :=
by
  have h1 : 125 = 5^3 := by norm_num
  have h2 : 81 = 3^4 := by norm_num
  have h3 : 32 = 2^5 := by norm_num
  sorry

end eval_expression_l343_343782


namespace firecrackers_defective_fraction_l343_343929

theorem firecrackers_defective_fraction (initial_total good_remaining confiscated : ℕ) 
(h_initial : initial_total = 48) 
(h_confiscated : confiscated = 12) 
(h_good_remaining : good_remaining = 15) : 
(initial_total - confiscated - 2 * good_remaining) / (initial_total - confiscated) = 1 / 6 := by
  sorry

end firecrackers_defective_fraction_l343_343929


namespace triangle_angle_sum_l343_343543

theorem triangle_angle_sum
  (E : ℝ) (hE : E = 18)
  (D : ℝ) (hD : D = 3 * E) :
  ∃ F : ℝ, F = 180 - (D + E) ∧ F = 108 :=
by
  use 180 - (54 + 18)  -- Directly substitute the known values
  split
  · apply rfl  -- This will always be true by definition
  · sorry  -- This step is correct by the steps in the mathematical proof, but we're skipping the proof part here

end triangle_angle_sum_l343_343543


namespace molly_takes_180_minutes_longer_l343_343668

noncomputable def time_for_Xanthia (pages_per_hour : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / pages_per_hour

noncomputable def time_for_Molly (pages_per_hour : ℕ) (total_pages : ℕ) : ℕ :=
  total_pages / pages_per_hour

theorem molly_takes_180_minutes_longer (pages : ℕ) (Xanthia_speed : ℕ) (Molly_speed : ℕ) :
  (time_for_Molly Molly_speed pages - time_for_Xanthia Xanthia_speed pages) * 60 = 180 :=
by
  -- Definitions specific to problem conditions
  let pages := 360
  let Xanthia_speed := 120
  let Molly_speed := 60

  -- Placeholder for actual proof
  sorry

end molly_takes_180_minutes_longer_l343_343668


namespace count_odd_multiples_of_3_l343_343438

theorem count_odd_multiples_of_3 :
  ∃ (count : ℕ), count = 333 ∧ 
    ∃ (n_set : finset ℕ), 
      (∀ n ∈ n_set, 1 ≤ n ∧ n ≤ 2000 ∧ n % 2 = 1 ∧ n % 3 = 0) ∧
      n_set.card = count :=
sorry

end count_odd_multiples_of_3_l343_343438


namespace curve_is_circle_with_radius_3_l343_343792

-- Define the equation in polar coordinates
def polar_eq (θ : ℝ) : ℝ := 3 * sin θ * csc θ

-- Define the Cartesian representation of a circle with radius 3, centered at the origin
def is_circle (r : ℝ) : Prop := r = 3 ∧ ∀ (x y : ℝ), x^2 + y^2 = r^2

-- The main theorem to prove
theorem curve_is_circle_with_radius_3 : 
  (∀ (θ : ℝ), polar_eq θ = 3) → (is_circle 3) :=
begin
  sorry
end

end curve_is_circle_with_radius_3_l343_343792


namespace inclination_angle_l343_343275

-- Define the line equation as a proposition in Lean.
def line_equation (x y : ℝ) : Prop := x + sqrt(3) * y - 5 = 0

theorem inclination_angle (x y : ℝ) :
  line_equation x y → ∃ θ : ℝ, 0 ≤ θ ∧ θ < 180 ∧ tan θ = -sqrt(3) / 3 ∧ θ = 150 :=
by {
  intros h,
  use 150,
  split; try {linarith},
  split; try {linarith},
  split; try {norm_num, rw [tan_def, sin_cos_iff_eq, cos_150, sin_150], 
      linarith, all_goals {rfield_tac; norm_num}},
  sorry
}

end inclination_angle_l343_343275


namespace area_of_triangle_ADE_l343_343169

theorem area_of_triangle_ADE (A B C D E : Type)
  [metric_space A]
  [metric_space B]
  [metric_space C]
  [metric_space D]
  [metric_space E]
  (AB AC BC AD AE : ℝ)
  (h1 : AB = 10)
  (h2 : AC = 13)
  (h3 : BC = 11)
  (h4 : AD = 5)
  (h5 : AE = 8) :
  ∃ [ADE_area: ℝ], ADE_area = (240 * real.sqrt 119) / 65 :=
  sorry

end area_of_triangle_ADE_l343_343169


namespace range_of_x_l343_343079

variable {f : ℝ → ℝ}

-- Define the function is_increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem range_of_x (h_inc : is_increasing f) (h_ineq : ∀ x : ℝ, f x < f (2 * x - 3)) :
  ∀ x : ℝ, 3 < x → f x < f (2 * x - 3) := 
sorry

end range_of_x_l343_343079


namespace probability_painted_cubes_identical_l343_343302

theorem probability_painted_cubes_identical :
  let face_colors := 3 in
  let faces_per_cube := 6 in
  let total_paintings := face_colors ^ faces_per_cube in
  let total_paintings_2_cubes := total_paintings ^ 2 in
  let identical_paintings := 3 + 36 + 90 in
  (identical_paintings : ℚ) / (total_paintings_2_cubes : ℚ) = 129 / 531441 :=
by
  sorry

end probability_painted_cubes_identical_l343_343302


namespace light_travel_distance_120_years_l343_343268

theorem light_travel_distance_120_years :
  let annual_distance : ℝ := 9.46e12
  let years : ℝ := 120
  (annual_distance * years) = 1.1352e15 := 
by
  sorry

end light_travel_distance_120_years_l343_343268


namespace negation_of_existential_proposition_l343_343068

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.exp x < 0) = (∀ x : ℝ, Real.exp x ≥ 0) :=
sorry

end negation_of_existential_proposition_l343_343068


namespace regression_line_decrease_l343_343062

theorem regression_line_decrease (x : ℝ) (y : ℝ) :
  let y1 := 2 - 1.5 * x,
      y2 := 2 - 1.5 * (x + 1)
  in y2 = y1 - 1.5 :=
by sorry

end regression_line_decrease_l343_343062


namespace find_f_2023_l343_343206

theorem find_f_2023 (f : ℕ → ℕ) 
  (h1 : f 1 = 1)
  (h2 : ∀ x y, (x + y) / 2 < f(x + y) ∧ f(x + y) ≤ f(x) + f(y))
  (h3 : ∀ n, f (4 * n + 1) < 2 * f (2 * n + 1))
  (h4 : ∀ n, f (4 * n + 3) ≤ 2 * f (2 * n + 1))
  : f 2023 = 1012 :=
by
  sorry

end find_f_2023_l343_343206


namespace range_OA_OB_l343_343467

noncomputable def circle_eq := ∀ (A B : ℝ × ℝ), 
  ((A.1 - 2) ^ 2 + A.2 ^ 2 = 1) ∧ ((B.1 - 2) ^ 2 + B.2 ^ 2 = 1)

noncomputable def AB_dist (A B : ℝ × ℝ) : Prop := 
  ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 2)

theorem range_OA_OB (A B : ℝ × ℝ) (O : ℝ × ℝ) (hA : circle_eq A) (hB : circle_eq B) 
  (hAB : AB_dist A B) (hO : O = (0, 0)) : 
  4 - real.sqrt 2 ≤ real.sqrt (((A.1 + B.1) ^ 2 + (A.2 + B.2) ^ 2)) ∧ 
  real.sqrt (((A.1 + B.1) ^ 2 + (A.2 + B.2) ^ 2)) ≤ 4 + real.sqrt 2 :=
sorry

end range_OA_OB_l343_343467


namespace quadratic_no_real_roots_l343_343364

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_no_real_roots
  (a b c: ℝ)
  (h1: ((b - 1)^2 - 4 * a * (c + 1) = 0))
  (h2: ((b + 2)^2 - 4 * a * (c - 2) = 0)) :
  ∀ x : ℝ, f a b c x ≠ 0 := 
sorry

end quadratic_no_real_roots_l343_343364


namespace find_a_l343_343563

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

theorem find_a (a : ℝ) : (∀ x : ℝ, f a x = f a (-x)) → a = -1 :=
by
  intro h_even
  have h_g : ∀ x, Real.exp x + a * Real.exp (-x) = -(Real.exp (-x) + a * Real.exp x) := sorry
  have h_g0 : Real.exp 0 + a * Real.exp 0 = 0 := by
    rw [Real.exp_zero, Real.exp_zero]
    sorry
  
  linarith

end find_a_l343_343563


namespace fourth_number_in_pascals_triangle_row_15_l343_343541

theorem fourth_number_in_pascals_triangle_row_15 : (Nat.choose 15 3) = 455 :=
by sorry

end fourth_number_in_pascals_triangle_row_15_l343_343541


namespace first_day_exceeds_target_l343_343899

-- Definitions based on the conditions
def initial_count : ℕ := 5
def daily_growth_factor : ℕ := 3
def target_count : ℕ := 200

-- The proof problem in Lean
theorem first_day_exceeds_target : ∃ n : ℕ, 5 * 3 ^ n > 200 ∧ ∀ m < n, ¬ (5 * 3 ^ m > 200) :=
by
  sorry

end first_day_exceeds_target_l343_343899


namespace regular_triangular_pyramid_volume_l343_343037

-- Definitions representing conditions
variables (h φ : ℝ)

-- Function representing the volume of the pyramid based on given conditions
def volume_of_regular_triangular_pyramid (h : ℝ) (φ : ℝ) : ℝ :=
  (h^3 * Real.sqrt 3 * (Real.cos (φ / 2))^2) / (3 - 4 * (Real.cos (φ / 2))^2)

-- Theorem statement representing the proof problem
theorem regular_triangular_pyramid_volume (h φ : ℝ) :
  volume_of_regular_triangular_pyramid h φ =
    (h^3 * Real.sqrt 3 * (Real.cos (φ / 2))^2) / (3 - 4 * (Real.cos (φ / 2))^2) :=
by
  sorry

end regular_triangular_pyramid_volume_l343_343037


namespace shortest_path_in_regular_hexagon_l343_343363

noncomputable def hexagon_side_length : ℝ := 4
noncomputable def travel_distance : ℝ := 10
noncomputable def shortest_distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem shortest_path_in_regular_hexagon :
  ∀ (s l : ℝ), 
  s = hexagon_side_length →
  l = travel_distance →
  shortest_distance 0 0 (1 : ℝ) (real.sqrt 3) = 2 := 
by
  sorry

end shortest_path_in_regular_hexagon_l343_343363


namespace circleAtBottomAfterRotation_l343_343914

noncomputable def calculateFinalCirclePosition (initialPosition : String) (sides : ℕ) : String :=
  if (sides = 8) then (if initialPosition = "bottom" then "bottom" else "unknown") else "unknown"

theorem circleAtBottomAfterRotation :
  calculateFinalCirclePosition "bottom" 8 = "bottom" :=
by
  sorry

end circleAtBottomAfterRotation_l343_343914


namespace number_of_integers_satisfying_condition_l343_343029

-- Define the condition
def condition1 (n : ℤ) : Prop := 25 < n^2
def condition2 (n : ℤ) : Prop := n^2 < 144

-- The math proof problem: Prove that the number of integers n such that 25 < n^2 < 144 is 12.
theorem number_of_integers_satisfying_condition : 
  {n : ℤ | condition1 n ∧ condition2 n}.toFinset.card = 12 :=
by sorry

end number_of_integers_satisfying_condition_l343_343029


namespace indeterminate_equation_solution_exists_l343_343465

theorem indeterminate_equation_solution_exists
  (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * c = b^2 + b + 1) :
  ∃ x y : ℤ, a * x^2 - (2 * b + 1) * x * y + c * y^2 = 1 := by
  sorry

end indeterminate_equation_solution_exists_l343_343465


namespace option_A_option_B_option_C_option_D_l343_343097

section
variables {a b r : ℝ}
def line := {p : ℝ × ℝ | a * p.1 + b * p.2 - r^2 = 0}
def circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
def point (x y : ℝ) := (x, y)

theorem option_A (h : a^2 + b^2 = r^2) : ∀ p ∈ circle, p = point a b → ∀ l ∈ line, tangent l circle :=
sorry

theorem option_B (h : a^2 + b^2 < r^2) : ∀ l ∈ line, disjoint l circle :=
sorry

theorem option_C (h : a^2 + b^2 > r^2) : ∃ l ∈ line, intersects l circle :=
sorry

theorem option_D (h_on_l : a * a + b * b - r^2 = 0) (h : a^2 + b^2 = r^2) : ∀ p ∈ point a b, ∀ l ∈ line, tangent l circle :=
sorry
end

end option_A_option_B_option_C_option_D_l343_343097


namespace perimeter_of_shape_l343_343610

-- Definition of conditions
def radii_equal (R : ℝ) : Prop :=
  ∃ (c1 c2 c3 : circle), 
  c1.radius = R ∧ c2.radius = R ∧ c3.radius = R ∧
  c1.center.1 = c2.center.1 ∧ c2.center.1 = c3.center.1 ∧
  (distance c2.center c1.center = R ∧ distance c2.center c3.center = R)

def centers_on_line (c1 c2 c3 : circle) : Prop :=
  ∃ (y : ℝ), 
  c1.center.2 = y ∧ c2.center.2 = y ∧ c3.center.2 = y

def circles_intersect_at_center (c1 c2 c3 : circle) : Prop :=
  distance c1.center c2.center = c3.radius ∧ distance c3.center c2.center = c1.radius

-- Theorem statement
theorem perimeter_of_shape (R : ℝ) (c1 c2 c3 : circle) (h_radii : radii_equal R) 
                           (h_centers : centers_on_line c1 c2 c3) 
                           (h_intersect : circles_intersect_at_center c1 c2 c3) :
  perimeter_of_shape c1 c2 c3 = (10 * π * R) / 3 :=
by sorry

end perimeter_of_shape_l343_343610


namespace exists_special_sequence_l343_343412

open List
open Finset
open BigOperators

theorem exists_special_sequence :
  ∃ s : ℕ → ℕ,
    (∀ n, s n > 0) ∧
    (∀ i j, i ≠ j → s i ≠ s j) ∧
    (∀ k, (∑ i in range (k + 1), s i) % (k + 1) = 0) :=
sorry  -- Proof from the provided solution steps.

end exists_special_sequence_l343_343412


namespace parabola_focus_l343_343026

theorem parabola_focus : ∃ f : ℝ, 
  (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (1/4 + f))^2)) ∧
  f = 1/8 := 
by
  sorry

end parabola_focus_l343_343026


namespace greatest_even_radius_l343_343888

theorem greatest_even_radius (r : ℕ) (h : π * r^2 < 100 * π) : r ≤ 9 ∧ r % 2 = 0 → r = 8 := 
begin
  sorry
end

end greatest_even_radius_l343_343888


namespace options_BD_correct_l343_343065

-- Define the given conditions
variables {a b c : ℝ}
axiom non_zero_a : a ≠ 0
axiom non_zero_b : b ≠ 0
axiom non_zero_c : c ≠ 0
axiom inequality : a > b ∧ b > c
axiom sum_zero : a + b + c = 0

-- Theorem to prove options B and D are correct
theorem options_BD_correct : (c / a + a / c ≤ -2) ∧ (c / a ∈ Ioc (-2 : ℝ) (-1 / 2 : ℝ)) :=
sorry

end options_BD_correct_l343_343065


namespace single_intersection_l343_343886

theorem single_intersection (k : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y^2 = x ∧ y + 1 = k * x) ↔ (k = 0 ∨ k = -1 / 4) :=
sorry

end single_intersection_l343_343886


namespace d_is_metric_l343_343209

-- Define the space D
def D (x : ℝ → ℝ) : Prop := 
  (∀ t < 1, ContinuousWithinAt x (Iio 1) t) ∧ 
  (∀ t > 0, ∃ l, Filter.Tendsto x (nhds_within t (Iio t)) (𝓝 l))

-- Define the set Λ of strictly increasing continuous functions from [0,1] to [0,1]
def Λ := {λ : ℝ → ℝ | StrictMono λ ∧ Continuous λ ∧ (λ 0 = 0 ∧ λ 1 = 1)}

-- Define the function d as the given infimum
noncomputable def d (x y : ℝ → ℝ) : ℝ :=
  inf ((λ λ, sup (λ t, abs (x t - y (λ t))) + sup (λ t, abs (t - λ t))) '' Λ)

-- Prove that d is a metric on D
theorem d_is_metric (x y z : ℝ → ℝ) (hx : D x) (hy : D y) (hz : D z) :
    d (x, y) ≥ 0 ∧
    (d (x, y) = 0 ↔ x = y) ∧
    d (x, y) = d (y, x) ∧
    d (x, z) ≤ d (x, y) + d (y, z) :=
by
  sorry

end d_is_metric_l343_343209


namespace evaluate_expression_l343_343786

theorem evaluate_expression:
  (125 = 5^3) ∧ (81 = 3^4) ∧ (32 = 2^5) → 
  125^(1/3) * 81^(-1/4) * 32^(1/5) = 10 / 3 := by
  sorry

end evaluate_expression_l343_343786


namespace distinct_prime_factors_of_a_le_19_l343_343261

variable {a b : ℕ}

theorem distinct_prime_factors_of_a_le_19 
  (h₁ : 0 < a ∧ 0 < b) 
  (h₂ : (nat.gcd a b).distinct_prime_factors.length = 8) 
  (h₃ : (nat.lcm a b).distinct_prime_factors.length = 30)
  (h₄ : a.distinct_prime_factors.length < b.distinct_prime_factors.length) : 
  a.distinct_prime_factors.length ≤ 19 :=
sorry

end distinct_prime_factors_of_a_le_19_l343_343261


namespace point_on_line_EF_P_l343_343938

noncomputable def point_P (r s : ℝ) (hr : 0 < r) (hs : 0 < s) : ℝ × ℝ :=
  (let P := (3 * r + 1) / 2, s / 2 in
  P)

theorem point_on_line_EF_P (r s : ℝ) (hr : 0 < r) (hs : 0 < s) :
  let P := point_P r s hr hs
  ∃ E : ℝ × ℝ, E.1 > 0 ∧ E.1 < 1 ∧ ∃ F : ℝ × ℝ,
    F ≠ E ∧ (F ∈ (circumcircle B C E)) ∧ (F ∈ (circumcircle A D E)) ∧
    (P ∈ line_through_points E F) := sorry

end point_on_line_EF_P_l343_343938


namespace geometric_relationships_l343_343100

variables {a b r : ℝ}
variables {A : ℝ × ℝ}
variables {l : ℝ × ℝ → Prop}

def circle (C : ℝ × ℝ → Prop) := ∀ p : ℝ × ℝ, C p ↔ p.1^2 + p.2^2 = r^2

def line (l : ℝ × ℝ → Prop) := ∀ p : ℝ × ℝ, l p ↔ a * p.1 + b * p.2 = r^2

def point_on_circle (C : ℝ × ℝ → Prop) (A : ℝ × ℝ) := C A
def point_inside_circle (C : ℝ × ℝ → Prop) (A : ℝ × ℝ) := A.1^2 + A.2^2 < r^2
def point_outside_circle (C : ℝ × ℝ → Prop) (A : ℝ × ℝ) := A.1^2 + A.2^2 > r^2
def point_on_line (l : ℝ × ℝ → Prop) (A : ℝ × ℝ) := l A

def tangent (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop) := 
  ∃ p : ℝ × ℝ, l p ∧ C p ∧ 
  (∃! q : ℝ × ℝ, C q ∧ l q)

def disjoint (l : ℝ × ℝ → Prop) (C : ℝ × ℝ → Prop) := 
  ∀ p : ℝ × ℝ, ¬ (l p ∧ C p)

theorem geometric_relationships (A : ℝ × ℝ) (h₀ : circle circle) (h₁ : line l) :
  (point_on_circle circle A → tangent l circle) ∧
  (point_inside_circle circle A → disjoint l circle) ∧
  (point_outside_circle circle A → ¬ disjoint l circle) ∧
  (point_on_line l A → tangent l circle) := 
sorry

end geometric_relationships_l343_343100


namespace exists_irrational_an_l343_343260

theorem exists_irrational_an (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n ≥ 1, a (n + 1)^2 = a n + 1) :
  ∃ n, ¬ ∃ q : ℚ, a n = q :=
sorry

end exists_irrational_an_l343_343260


namespace find_average_l343_343118

noncomputable def log_base (b x : ℝ) := real.log x / real.log b

theorem find_average (x y : ℝ) (h_cond1 : x > 0) (h_cond2 : y > 0) 
  (h_log : log_base y x + log_base x y = 9/2) (h_prod : x * y = 200) : 
  (x + y) / 2 = (real.sqrt (real.sqrt (real.sqrt (real.sqrt200))) + (real.sqrt (real.sqrt (real.sqrt (real.sqrt(200))))) ^ 3.5) / 2 := 
  sorry

end find_average_l343_343118


namespace line_through_points_l343_343166

theorem line_through_points (m n p : ℝ) 
  (h1 : m = 4 * n + 5) 
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 := 
by 
  sorry

end line_through_points_l343_343166


namespace move_factor_outside_sqrt_l343_343233

theorem move_factor_outside_sqrt (a b : ℝ) (h : a - b < 0) :
  (a - b) * real.sqrt (-1 / (a - b)) = - real.sqrt (b - a) :=
by
  sorry

end move_factor_outside_sqrt_l343_343233


namespace interest_rate_first_part_l343_343373

theorem interest_rate_first_part 
  (total_amount : ℤ) 
  (amount_at_first_rate : ℤ) 
  (amount_at_second_rate : ℤ) 
  (rate_second_part : ℤ) 
  (total_annual_interest : ℤ) 
  (r : ℤ) 
  (h_split : total_amount = amount_at_first_rate + amount_at_second_rate) 
  (h_second : rate_second_part = 5)
  (h_interest : (amount_at_first_rate * r) / 100 + (amount_at_second_rate * rate_second_part) / 100 = total_annual_interest) :
  r = 3 := 
by 
  sorry

end interest_rate_first_part_l343_343373


namespace sum_in_base_6_l343_343883

theorem sum_in_base_6 (S H E : ℕ) (h1 : S ≠ H) (h2 : H ≠ E) (h3 : S ≠ E)
  (h4 : 0 < S) (h5 : S < 6)
  (h6 : 0 < H) (h7 : H < 6)
  (h8 : 0 < E) (h9 : E < 6)
  (h_addition : (S + H) % 6 = S)
  (h_carry : (E + H) % 6 = E): 
  S + H + E = 13_6 := sorry

end sum_in_base_6_l343_343883


namespace planes_parallel_l343_343840

variables (m n : Line) (α β : Plane)

-- Non-overlapping lines and planes conditions
axiom non_overlapping_lines : m ≠ n
axiom non_overlapping_planes : α ≠ β

-- Parallel and perpendicular definitions
axiom parallel_lines (l k : Line) : Prop
axiom parallel_planes (π ρ : Plane) : Prop
axiom perpendicular (l : Line) (π : Plane) : Prop

-- Given conditions
axiom m_perpendicular_to_alpha : perpendicular m α
axiom m_perpendicular_to_beta : perpendicular m β

-- Proof statement
theorem planes_parallel (m_perpendicular_to_alpha : perpendicular m α)
  (m_perpendicular_to_beta : perpendicular m β) :
  parallel_planes α β := sorry

end planes_parallel_l343_343840


namespace first_day_is_painted_l343_343389

theorem first_day_is_painted
  (n : ℕ) -- The number of days in the month.
  (h1 : 1 ≤ n) -- There is at least one day in the month.
  (painted_days : set ℕ) -- The set of painted days.
  (h2 : painted_days.card = 3) -- Exactly three days are painted.
  (h3 : ∀ a b ∈ painted_days, a ≠ b → abs (a - b) ≠ 1) -- No two painted days are consecutive.
  (digits : set ℕ) -- The set of all digits of the month written consecutively.
  (h4 : ∀ (unpainted : set ℕ), unpainted ⊆ digits → unpainted.card > 0 → ∃ d, ∀ x y ∈ unpainted, ∃ s t : ℕ, x = d + s ∧ y = d + t) 
  -- All unpainted sections consist of the same number of digits.
  : 1 ∈ painted_days := 
sorry

end first_day_is_painted_l343_343389


namespace parabola_properties_l343_343457

theorem parabola_properties
  (p : ℝ) (h_p_pos : p > 0)
  (l : ℝ → ℝ) (focus : (ℝ × ℝ))
  (h_l_perpendicular : ∃ (k : ℝ), l = λ x, k)
  (area_tria_BOA : (1/2) * abs ((p ^ 2) / 2) = 8) :
  -- parts A:
  (∀ (y : ℝ), (∃ (x : ℝ), x^2 = 2*p*y) → (x^2 = 8 * y)) ∧
  -- parts B:
  (∀ (AB : ℝ), AB = 12 → (∃ (mid_AB : ℝ), mid_AB = 6 - 2 = 4)) ∧
  -- parts D:
  (∀ (A B : (ℝ × ℝ)), (∃ (F : (ℝ × ℝ)), |A.1 - F.1| + 4 * |B.1 - F.1| ≥ 18) :=
by
  sorry

end parabola_properties_l343_343457


namespace necessary_and_sufficient_condition_l343_343336

theorem necessary_and_sufficient_condition {a b : ℝ} :
  (a > b) ↔ (a^3 > b^3) := sorry

end necessary_and_sufficient_condition_l343_343336


namespace expected_adjacent_red_pairs_l343_343347

theorem expected_adjacent_red_pairs (total_cards : ℕ) (red_cards : ℕ) : 
  total_cards = 104 → 
  red_cards = 52 →
  let probability_adjacent_red := (red_cards - 1) / (total_cards - 1) in
  expected_value : ℚ :=
  52 * probability_adjacent_red = 52 * (51 / 103) → 
  expected_value = 2652 / 103 :=
by
  intros
  sorry

end expected_adjacent_red_pairs_l343_343347


namespace part1_solution_set_part2_comparison_l343_343087

noncomputable def f (x : ℝ) := -|x| - |x + 2|

theorem part1_solution_set (x : ℝ) : f x < -4 ↔ x < -3 ∨ x > 1 :=
by sorry

theorem part2_comparison (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = Real.sqrt 5) : 
  a^2 + b^2 / 4 ≥ f x + 3 :=
by sorry

end part1_solution_set_part2_comparison_l343_343087


namespace election_votes_l343_343142

theorem election_votes (total_votes : ℕ) (h1 : (4 / 15) * total_votes = 48) : total_votes = 180 :=
sorry

end election_votes_l343_343142


namespace value_of_fraction_imaginary_unit_l343_343837

theorem value_of_fraction_imaginary_unit : 
  (i : ℂ) (hi : i * i = -1) → (i / (1 - i) = (-1/2 + (1/2) * i)) :=
by
  intros i hi
  sorry

end value_of_fraction_imaginary_unit_l343_343837


namespace grandpa_max_movies_l343_343497

-- Definition of the conditions
def movie_duration : ℕ := 90

def tuesday_total_minutes : ℕ := 4 * 60 + 30

def tuesday_movies_watched : ℕ := tuesday_total_minutes / movie_duration

def wednesday_movies_watched : ℕ := 2 * tuesday_movies_watched

def total_movies_watched : ℕ := tuesday_movies_watched + wednesday_movies_watched

theorem grandpa_max_movies : total_movies_watched = 9 := by
  sorry

end grandpa_max_movies_l343_343497


namespace matrix_cubed_l343_343396

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![2, -2], ![2, -1]]

theorem matrix_cubed :
  (A * A * A) = ![![ -4, 2], ![-2, 1]] :=
by
  sorry

end matrix_cubed_l343_343396


namespace planes_parallel_l343_343839

variables (m n : Line) (α β : Plane)

-- Non-overlapping lines and planes conditions
axiom non_overlapping_lines : m ≠ n
axiom non_overlapping_planes : α ≠ β

-- Parallel and perpendicular definitions
axiom parallel_lines (l k : Line) : Prop
axiom parallel_planes (π ρ : Plane) : Prop
axiom perpendicular (l : Line) (π : Plane) : Prop

-- Given conditions
axiom m_perpendicular_to_alpha : perpendicular m α
axiom m_perpendicular_to_beta : perpendicular m β

-- Proof statement
theorem planes_parallel (m_perpendicular_to_alpha : perpendicular m α)
  (m_perpendicular_to_beta : perpendicular m β) :
  parallel_planes α β := sorry

end planes_parallel_l343_343839


namespace bound_deviation_correct_l343_343283

noncomputable def deviation_bound (p n desired_proba : ℝ) : ℝ :=
  let q := 1 - p in
  let z_value := 2.4 in
  z_value / (Math.sqrt (n / (p * q)))

theorem bound_deviation_correct :
  deviation_bound 0.3 90 0.9836 ≈ 0.1159 := by
  sorry

end bound_deviation_correct_l343_343283


namespace problem_statement_l343_343007

def binary_op (a b : ℚ) : ℚ := (a^2 + b^2) / (a^2 - b^2)

theorem problem_statement : binary_op (binary_op 8 6) 2 = 821 / 429 := 
by sorry

end problem_statement_l343_343007


namespace man_and_son_work_together_l343_343700

theorem man_and_son_work_together (man_days son_days : ℕ) (h_man : man_days = 15) (h_son : son_days = 10) :
  (1 / (1 / man_days + 1 / son_days) = 6) :=
by
  rw [h_man, h_son]
  sorry

end man_and_son_work_together_l343_343700


namespace sin_A_eq_a_div_c_l343_343152

-- Define the context and parameters of the problem.
variable (A B C : Type) [InnerProductSpace ℝ C]
variable (a b c : ℝ)

-- Condition: Right triangle with sides a, b, and hypotenuse c.
variable (hC : ∠ C = π / 2)
variable (ha : side_opposite A = a)
variable (hb : side_opposite B = b)
variable (hc : side_opposite C = c)

-- Theorem statement: The sine of angle A in the right triangle is a / c.
theorem sin_A_eq_a_div_c : sin A = a / c := 
begin
    sorry,
end

end sin_A_eq_a_div_c_l343_343152


namespace exists_zero_in_interval_l343_343617

noncomputable def f (x : ℝ) : ℝ := x^(1/2) - 2 + Real.log2 x

-- Stating the problem in terms of Lean
theorem exists_zero_in_interval :
  (∀ x, 0 < x → ContinuousAt f x) →
  f 1 < 0 →
  f 2 > 0 →
  ∃ c, c ∈ Set.Ioo 1 2 ∧ f c = 0 :=
by
  intro h1 h2 h3
  sorry

end exists_zero_in_interval_l343_343617


namespace even_function_x_lt_0_l343_343851

noncomputable def f (x : ℝ) : ℝ :=
if h : x >= 0 then 2^x + 1 else 2^(-x) + 1

theorem even_function_x_lt_0 (x : ℝ) (hx : x < 0) : f x = 2^(-x) + 1 :=
by {
  sorry
}

end even_function_x_lt_0_l343_343851


namespace mn_square_eq_t_eq_l343_343289

variables {A B C D M N : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] 
[MetricSpace M] [MetricSpace N]

theorem mn_square_eq
  (h1 : dist A M / dist M B = dist D N / dist N C = dist A D / dist B C) :
  dist M N * dist M N = 
    (dist A D * dist B C) / ((dist A D) + (dist B C))^2 * 
    (dist A C^2 + dist B D^2 + 2 * dist A D * dist B C - dist A B^2 - dist D C^2) :=
by sorry

theorem t_eq
  (h1 : dist A M / dist M B = dist D N / dist N C = dist A D / dist B C)
  (ψ : ℝ) :
  let t := (2 * dist B D * dist B C * cos (ψ / 2)) / (dist B D + dist B C) in
  t = (2 * dist B D * cos (ψ / 2)) / (dist B D + dist B C) :=
by sorry

end mn_square_eq_t_eq_l343_343289


namespace inverse_proportion_l343_343664

theorem inverse_proportion {x y : ℝ} :
  (y = (3 / x)) -> ¬(y = x / 3) ∧ ¬(y = 3 / (x + 1)) ∧ ¬(y = 3 * x) :=
by
  sorry

end inverse_proportion_l343_343664


namespace arithmetic_sequence_sum_l343_343825

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h1 : S 4 = 3) (h2 : S 8 = 7) : S 12 = 12 :=
by
  -- placeholder for the proof, details omitted
  sorry

end arithmetic_sequence_sum_l343_343825


namespace length_XZ_l343_343900

-- Definitions based on conditions
variables (O C B Z X : Point)
variables (r : ℝ) (angle_OCB angle_XCZ : ℝ)
variable (radius_OC : ℝ := 10)
hypothesis (h_sector : is_sector O C B)
hypothesis (h_angle_COB : angle O C B = real.pi / 2)
hypothesis (h_perpendicular : is_perpendicular C Z B)
hypothesis (h_intersection : intersects C Z B X)
hypothesis (h_radius : distance O C = radius_OC)

-- The statement to be proved
theorem length_XZ : distance X Z = 10 - 5 * real.sqrt 2 :=
sorry

end length_XZ_l343_343900


namespace least_three_digit_multiple_of_2_5_7_l343_343656

theorem least_three_digit_multiple_of_2_5_7
  (n : ℤ) (h1 : 100 ≤ n) (h2 : n < 1000) (h3 : 2 ∣ n) (h4 : 5 ∣ n) (h5 : 7 ∣ n) :
  n = 140 :=
begin
  sorry
end

end least_three_digit_multiple_of_2_5_7_l343_343656


namespace interval_length_l343_343035

theorem interval_length (x : ℝ) :
  (\{x : ℝ | (x^2 - 80 * x + 1500) / (x^2 - 55 * x + 700) < 0\}).measure = 25 :=
sorry

end interval_length_l343_343035


namespace inequality_AE_ED_AB_CD_BE_CE_l343_343571

/-- 
  Four distinct points A, B, C, and D are taken on a line in that order.
  E is a point not lying on the line AD.
  We want to show that:
  AE + ED + |AB - CD| > BE + CE
-/
theorem inequality_AE_ED_AB_CD_BE_CE 
  (A B C D E : Point)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ D)
  (h_order : A < B ∧ B < C ∧ C < D)
  (h_not_on_line_AD : ¬ collinear {A, D, E}) :
  distance A E + distance E D + abs (distance A B - distance C D) > distance B E + distance C E :=
by
  -- placeholder for proof
  sorry

end inequality_AE_ED_AB_CD_BE_CE_l343_343571


namespace least_value_x_minus_y_minus_z_l343_343332

/-- 
  If x, y, and z are positive integers such that x = p^2, y = q^3, and z = r^5 
  for distinct prime numbers p, q, and r, then the least possible value
  of x - y - z is -3148.
-/
theorem least_value_x_minus_y_minus_z :
  ∀ (p q r : ℕ), p.prime ∧ q.prime ∧ r.prime ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r →
  let x := p^2 in
  let y := q^3 in
  let z := r^5 in
  x - y - z = -3148 :=
by
  intros p q r h
  let x := p^2
  let y := q^3
  let z := r^5
  sorry

end least_value_x_minus_y_minus_z_l343_343332


namespace S_n_19_is_95_l343_343632

-- Define the arithmetic sequence and its terms.
variable (a_n : ℕ → ℕ) -- Assuming the sequence values are natural numbers.

-- Define the sum of the first n terms of the sequence.
variable (S_n : ℕ → ℕ)

-- Given condition: a₃ + a₁₇ = 10.
variable h1 : a_n 3 + a_n 17 = 10

-- Finding S₁₉.
theorem S_n_19_is_95 (h1 : a_n 3 + a_n 17 = 10) : S_n 19 = 95 := sorry

end S_n_19_is_95_l343_343632


namespace inequality_proof_l343_343449

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) : a < 2 * b - b^2 / a := 
by
  -- mathematical proof goes here
  sorry

end inequality_proof_l343_343449


namespace average_age_in_interval_l343_343367

-- Define the conditions of the problem
def employees := 20

def age_distribution : Type := fin 5 → ℕ

noncomputable def age_counts : age_distribution := ![5, 10, 0, 3, 2]

-- Calculate the total age in two extreme scenarios and then the average age
noncomputable def total_age_min := 5 * 38 + 10 * 39 + 3 * 41 + 2 * 42
noncomputable def total_age_max := 5 * 38 + 10 * 40 + 3 * 41 + 2 * 42

noncomputable def average_age_min : ℝ := total_age_min / employees
noncomputable def average_age_max : ℝ := total_age_max / employees

-- The statement we need to prove: avg_age is in (39, 40)
theorem average_age_in_interval : 39 < average_age_min  ∧ average_age_max < 40 := by
  sorry

end average_age_in_interval_l343_343367


namespace possible_values_f_prime_half_l343_343222

noncomputable def f : (ℝ → ℝ) := sorry

def condition (f : ℝ → ℝ) : Prop :=
∀ n : ℕ, ∀ a : ℕ, (a < 2^n ∧ (a % 2 = 1)) →
  ∃ b : ℕ, (b < 2^n ∧ (b % 2 = 1) ∧ f ((a : ℝ) / 2^n) = (b : ℝ) / 2^n)

theorem possible_values_f_prime_half :
  (differentiable ℝ f) 
  ∧ (∀ x ∈ (Ioo 0 1), continuous_at (deriv f) x) 
  ∧ condition f → 
  deriv f (1/2) ∈ ({-1, 1} : set ℝ) :=
begin
  sorry
end

end possible_values_f_prime_half_l343_343222


namespace count_six_digit_numbers_l343_343044

theorem count_six_digit_numbers : 
  (count_numbers (List.prod [1, 1, 2, 2, 3, 4]) ∉ adjacent_identical) = 84 :=
by 
  sorry

def count_numbers (lst : list ℕ) : ℕ :=
  -- Assume we have a function that counts the valid permutations
  sorry

def adjacent_identical (lst : list ℕ) : Prop := 
  -- Assume we have a predicate that checks for adjacent identical elements
  sorry

end count_six_digit_numbers_l343_343044


namespace matrix_pow_three_l343_343394

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_pow_three :
  A^3 = !![-4, 2; -2, 1] := by
  sorry

end matrix_pow_three_l343_343394


namespace circle_standard_equation_l343_343290

theorem circle_standard_equation (x y : ℝ) (h : (x + 1)^2 + (y - 2)^2 = 4) : 
  (x + 1)^2 + (y - 2)^2 = 4 :=
sorry

end circle_standard_equation_l343_343290


namespace sequence_next_terms_l343_343789

theorem sequence_next_terms (a : ℝ) : 
  ∃ b1 b2 : ℝ, (∀ n : ℕ, (b1 = 5*a^5) ∧ (b2 = -6*a^6)) :=
by trivial

end sequence_next_terms_l343_343789


namespace real_root_bound_l343_343200

noncomputable def P (n : ℕ → ℕ) (s : ℕ) (x : ℝ) : ℝ :=
  1 + x^2 + x^9 + ∑ i in Finset.range s, x^(n i) + x^1992

theorem real_root_bound
  (n : ℕ → ℕ)
  (s : ℕ)
  (x₀ : ℝ)
  (h_order : ∀ i j, i < j → n i < n j)
  (h_bound : ∀ i, 9 < n i ∧ n i < 1992)
  (h_root : P n s x₀ = 0) :
  x₀ ≤ (1 - Real.sqrt 5) / 2 :=
sorry

end real_root_bound_l343_343200


namespace octahedron_interior_diagonals_l343_343876

-- Define what constitutes an octahedron and its properties
def is_octahedron (V : Finset ℕ) (F : Finset (Finset ℕ)) : Prop :=
  V.card = 6 ∧ F.card = 8 ∧ 
  (∀ v ∈ V, ∃ S ⊆ V, S.card = 4 ∧ S ∈ F ∧ v ∈ S)

-- Define what constitutes an interior diagonal
def is_interior_diagonal (V : Finset ℕ) (F : Finset (Finset ℕ)) (d : Finset ℕ) : Prop :=
  d.card = 2 ∧ d ⊆ V ∧ (∀ f ∈ F, ¬ d ⊆ f)

-- Main statement to prove
theorem octahedron_interior_diagonals : 
  ∀ (V : Finset ℕ) (F : Finset (Finset ℕ)), 
  is_octahedron V F → 
  (Finset.filter (is_interior_diagonal V F) (Finset.powersetLen 2 V)).card = 3 :=
by 
  intros V F h
  sorry

end octahedron_interior_diagonals_l343_343876


namespace problem_statement_l343_343259

noncomputable def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ (n : ℕ), n ∣ p → n = 1 ∨ n = p

theorem problem_statement (n r : ℕ)
  (h : ∀ k : ℕ, n^2 + r - k * (k + 1) ≠ -1 ∧ ¬(∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n^2 + r - k * (k + 1))) :
  4 * n^2 + 4 * r + 1 = 1 ∨ 4 * n^2 + 4 * r + 1 = 9 ∨   ∃ p : ℕ, isPrime p ∧ 4 * n^2 + 4 * r + 1 = p :=
by
  sorry

end problem_statement_l343_343259


namespace cube_surface_area_l343_343270

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 6) : 6 * (edge_length ^ 2) = 216 :=
by
  -- Use the given condition to show that the edge length is 6.
  rw [h]
  -- Calculate the surface area when the edge length is 6.
  norm_num
  -- The expected result should simplify correctly.
  reflexivity

end cube_surface_area_l343_343270


namespace solve_for_x_l343_343810

theorem solve_for_x (x : ℝ) (h : (2 + x) / (4 + x) = (3 + x) / (7 + x)) : x = -1 :=
by {
  sorry
}

end solve_for_x_l343_343810


namespace symmetric_points_origin_l343_343534

theorem symmetric_points_origin (a b : ℝ) (h1 : a = -(-2)) (h2 : 1 = -b) : a + b = 1 :=
by
  sorry

end symmetric_points_origin_l343_343534


namespace total_conference_games_scheduled_l343_343263

-- Definitions of the conditions
def num_divisions : ℕ := 2
def teams_per_division : ℕ := 6
def intradivision_games_per_pair : ℕ := 3
def interdivision_games_per_pair : ℕ := 2

-- The statement to prove the total number of conference games
theorem total_conference_games_scheduled : 
  (num_divisions * (teams_per_division * (teams_per_division - 1) * intradivision_games_per_pair) / 2) 
  + (teams_per_division * teams_per_division * interdivision_games_per_pair) = 162 := 
by
  sorry

end total_conference_games_scheduled_l343_343263


namespace previous_monthly_income_l343_343968

variable (I : ℝ)

-- Conditions from the problem
def condition1 (I : ℝ) : Prop := 0.40 * I = 0.25 * (I + 600)

theorem previous_monthly_income (h : condition1 I) : I = 1000 := by
  sorry

end previous_monthly_income_l343_343968


namespace jackson_running_l343_343183

variable (x : ℕ)

theorem jackson_running (h : x + 4 = 7) : x = 3 := by
  sorry

end jackson_running_l343_343183


namespace max_sum_of_squares_of_sides_and_diagonals_l343_343053

noncomputable def sum_of_squares_of_sides_and_diagonals (n : ℕ) : ℝ :=
  let vertices := λ k : ℕ, complex.exp (2 * real.pi * complex.I * k / n) in
  let distances := λ s t, abs (vertices s - vertices t) ^ 2 in
  ∑ s in finset.range n, ∑ t in finset.range n, distances s t

theorem max_sum_of_squares_of_sides_and_diagonals (n : ℕ) (h_n : 3 ≤ n):
    sum_of_squares_of_sides_and_diagonals n = n^2 := by
  sorry

end max_sum_of_squares_of_sides_and_diagonals_l343_343053


namespace ratio_total_length_to_perimeter_l343_343711

noncomputable def length_initial : ℝ := 25
noncomputable def width_initial : ℝ := 15
noncomputable def extension : ℝ := 10
noncomputable def length_total : ℝ := length_initial + extension
noncomputable def perimeter_new : ℝ := 2 * (length_total + width_initial)
noncomputable def ratio : ℝ := length_total / perimeter_new

theorem ratio_total_length_to_perimeter : ratio = 35 / 100 := by
  sorry

end ratio_total_length_to_perimeter_l343_343711


namespace supplement_greater_than_complement_l343_343070

variable (angle1 : ℝ)

def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

theorem supplement_greater_than_complement (h : is_acute angle1) :
  180 - angle1 = 90 + (90 - angle1) :=
by {
  sorry
}

end supplement_greater_than_complement_l343_343070


namespace train_passing_time_proof_pass_train_l343_343719

theorem train_passing_time (
  train_length : ℝ,
  train_speed_kmph : ℝ,
  train_slope_deg : ℝ,
  trolley_speed_kmph : ℝ,
  trolley_slope_deg : ℝ,
  opposite_directions : bool
) : ℝ :=
  -- Conversion of km/hr to m/s
  let train_speed := train_speed_kmph * 1000 / 3600 in
  let trolley_speed := trolley_speed_kmph * 1000 / 3600 in
  -- The relative speed considering opposite directions
  if opposite_directions then
    let relative_speed := train_speed + trolley_speed in
    -- Calculation of passing time
    train_length / relative_speed
  else 
    0 -- This can be refined further 

-- Adding the conditions from the problem
def problem_instance := train_passing_time (250) (100) (5) (25) (8) (true)

theorem proof_pass_train : problem_instance = 7.20 := by sorry

end train_passing_time_proof_pass_train_l343_343719


namespace evaluate_expression_l343_343785

theorem evaluate_expression:
  (125 = 5^3) ∧ (81 = 3^4) ∧ (32 = 2^5) → 
  125^(1/3) * 81^(-1/4) * 32^(1/5) = 10 / 3 := by
  sorry

end evaluate_expression_l343_343785


namespace problem_1_problem_2_l343_343092

theorem problem_1 (k : ℝ) : 
  (∀ x : ℝ, k ≠ 0 ∧ (kx^2 - 2x + 6k < 0 ↔ x < -3 ∨ x > -2)) → k = -2/5 := 
sorry

theorem problem_2 (k : ℝ) : 
  (∀ x : ℝ, k ≠ 0 ∧ (kx^2 - 2x + 6k < 0)) ↔ k < -Real.sqrt (6) / 6 :=
sorry

end problem_1_problem_2_l343_343092


namespace age_ratio_proof_l343_343239

noncomputable def proof_problem : Prop :=
  ∃ R : ℕ, 
  let Roonie_one_year_ago := R - 1 in let Ronaldo_one_year_ago := 35 in 
  (Ronaldo_current := 36) ∧ 
  let Roonie_in_4_years := R + 4 in let Ronaldo_in_4_years := 40 in 
  (Roonie_in_4_years * 8 = Ronaldo_in_4_years * 7) ∧
  (Roonie_one_year_ago / Ronaldo_one_year_ago) = 6 / 7

theorem age_ratio_proof : proof_problem :=
  sorry

end age_ratio_proof_l343_343239


namespace find_original_number_l343_343704

theorem find_original_number (x : ℝ) : 1.5 * x = 525 → x = 350 := by
  sorry

end find_original_number_l343_343704


namespace smallest_sum_zero_l343_343727

theorem smallest_sum_zero : ∃ x ∈ ({-1, -2, 1, 2} : Set ℤ), ∀ y ∈ ({-1, -2, 1, 2} : Set ℤ), x + 0 ≤ y + 0 :=
sorry

end smallest_sum_zero_l343_343727


namespace intersection_setA_setB_l343_343490

noncomputable def setA : Set ℝ := { x : ℝ | abs (x - 1) < 2 }
noncomputable def setB : Set ℝ := { x : ℝ | (x - 2) / (x + 4) < 0 }

theorem intersection_setA_setB : 
  (setA ∩ setB) = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_setA_setB_l343_343490


namespace num_solutions_to_equation_l343_343510

noncomputable def numerator (x : ℕ) : ℕ := ∏ i in (finset.range 51).filter (λ i , i > 0), (x - i)

noncomputable def denominator (x : ℕ) : ℕ := ∏ i in (finset.range 51).filter (λ i, i > 0), (x - i^2)

theorem num_solutions_to_equation : 
  (∃ n, nat.card { x : ℕ | 1 ≤ x ∧ x ≤ 50 ∧ numerator x = 0 ∧ denominator x ≠ 0 } = n) ∧ n = 43 := 
by
  sorry

end num_solutions_to_equation_l343_343510


namespace shortest_distance_bug_crawl_l343_343369

theorem shortest_distance_bug_crawl (r h A_dist B_dist : ℝ) (h_eq : h = 250 * Real.sqrt 3) (r_eq : r = 500)
  (A_dist_eq : A_dist = 100) (B_dist_eq : B_dist = 300 * Real.sqrt 3) :
  let C := 2 * Real.pi * r,
      R := Real.sqrt (r^2 + h^2),
      theta := C / R,
      A := (A_dist, 0),
      B := (-B_dist * Real.sqrt 3, B_dist / (2 * Real.pi * Real.sqrt 7)) in
  Real.dist A B = 100 * Real.sqrt 23 := 
by
  sorry

end shortest_distance_bug_crawl_l343_343369


namespace part1_part2_l343_343466

open Set

-- Part 1: When m = 6, find (complement_R A) ∪ B
theorem part1 {ℝ : Type*} [ordered_ring ℝ] :
  let A := {x : ℝ | -3 ≤ x ∧ x ≤ 6},
      B := {x : ℝ | 0 < x ∧ x < 9} in
  (compl A) ∪ B = {x : ℝ | x < -3 ∨ x > 0} := 
  sorry

-- Part 2: If A ∪ B = A, find the range of real numbers for m
theorem part2 {ℝ : Type*} [ordered_ring ℝ] :
  let A := {x : ℝ | -3 ≤ x ∧ x ≤ 6},
      B := {x : ℝ | 6 - m < x ∧ x < m + 3},
      m_range := {m : ℝ | (m ≤ 3)} in 
  (A ∪ B = A) → (∃ m ∈ m_range) :=
  sorry

end part1_part2_l343_343466


namespace extra_men_needed_l343_343735

theorem extra_men_needed 
  (total_length : ℝ) (total_time : ℕ) (initial_men : ℕ) 
  (completed_work : ℝ) (days_spent : ℕ)
  (h_total_length : total_length = 15) (h_total_time : total_time = 300)
  (h_initial_men : initial_men = 30) (h_completed_work : completed_work = 2.5) (h_days_spent : days_spent = 100) :
  let remaining_work := total_length - completed_work in
  let remaining_time := total_time - days_spent in
  let current_rate := completed_work / days_spent in
  let required_rate := remaining_work / remaining_time in
  let required_men := required_rate / current_rate * initial_men in
  let extra_men := required_men - initial_men in
  extra_men = 45 :=
by
  sorry

end extra_men_needed_l343_343735


namespace prove_a_5_l343_343634

noncomputable def a_5_proof : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ),
    (∀ n, a n > 0) → 
    (a 1 + 2 * a 2 = 4) →
    ((a 1)^2 * q^6 = 4 * a 1 * q^2 * a 1 * q^6) →
    a 5 = 1 / 8

theorem prove_a_5 : a_5_proof := sorry

end prove_a_5_l343_343634


namespace Mark_sold_1_box_less_than_n_l343_343229

variable (M A n : ℕ)

theorem Mark_sold_1_box_less_than_n (h1 : n = 8)
 (h2 : A = n - 2)
 (h3 : M + A < n)
 (h4 : M ≥ 1) 
 (h5 : A ≥ 1)
 : M = 1 := 
sorry

end Mark_sold_1_box_less_than_n_l343_343229


namespace right_triangular_prism_volume_l343_343710

noncomputable def base_area (a : ℝ) : ℝ :=
  (math.sqrt 3 / 4) * a^2

noncomputable def triangular_prism_volume (base_edge : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area base_edge * height

theorem right_triangular_prism_volume :
  triangular_prism_volume 2 3 = math.sqrt 3 :=
by
  sorry

end right_triangular_prism_volume_l343_343710


namespace k_gt_4_l343_343870

theorem k_gt_4 {x y k : ℝ} (h1 : 2 * x + y = 2 * k - 1) (h2 : x + 2 * y = -4) (h3 : x + y > 1) : k > 4 :=
by
  -- This 'sorry' serves as a placeholder for the actual proof steps
  sorry

end k_gt_4_l343_343870


namespace min_max_x_l343_343357

-- Definitions for the initial conditions and surveys
def students : ℕ := 100
def like_math_initial : ℕ := 50
def dislike_math_initial : ℕ := 50
def like_math_final : ℕ := 60
def dislike_math_final : ℕ := 40

-- Variables for the students' responses
variables (a b c d : ℕ)

-- Conditions based on the problem statement
def initial_survey : Prop := a + d = like_math_initial ∧ b + c = dislike_math_initial
def final_survey : Prop := a + c = like_math_final ∧ b + d = dislike_math_final

-- Definition of x as the number of students who changed their answer
def x : ℕ := c + d

-- Prove the minimum and maximum value of x with given conditions
theorem min_max_x (a b c d : ℕ) 
  (initial_cond : initial_survey a b c d)
  (final_cond : final_survey a b c d)
  : 10 ≤ (x c d) ∧ (x c d) ≤ 90 :=
by
  -- This is where the proof would go, but we'll simply state sorry for now.
  sorry

end min_max_x_l343_343357


namespace unique_coprime_trio_l343_343434

theorem unique_coprime_trio :
  ∃! (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a < b ∧ b < c ∧
  Nat.coprime a b ∧ Nat.coprime a c ∧ Nat.coprime b c ∧
  (∃ k : ℕ, a + b = k * c) ∧
  (∃ m : ℕ, a + c = m * b) ∧
  (∃ n : ℕ, b + c = n * a) ∧
  (a = 1 ∧ b = 2 ∧ c = 3) :=
begin
  sorry
end

end unique_coprime_trio_l343_343434


namespace parabola_focus_l343_343027

theorem parabola_focus : ∃ f : ℝ, 
  (∀ x : ℝ, (x^2 + (2*x^2 - f)^2 = (2*x^2 - (1/4 + f))^2)) ∧
  f = 1/8 := 
by
  sorry

end parabola_focus_l343_343027


namespace product_ABC_l343_343208

def A : ℂ := 6 + 3 * Complex.i
def B : ℂ := 2 * Complex.i
def C : ℂ := 6 - 3 * Complex.i

theorem product_ABC : A * B * C = 90 * Complex.i := by
  sorry

end product_ABC_l343_343208


namespace planes_parallel_l343_343838

variables (m n : Line) (α β : Plane)

-- Non-overlapping lines and planes conditions
axiom non_overlapping_lines : m ≠ n
axiom non_overlapping_planes : α ≠ β

-- Parallel and perpendicular definitions
axiom parallel_lines (l k : Line) : Prop
axiom parallel_planes (π ρ : Plane) : Prop
axiom perpendicular (l : Line) (π : Plane) : Prop

-- Given conditions
axiom m_perpendicular_to_alpha : perpendicular m α
axiom m_perpendicular_to_beta : perpendicular m β

-- Proof statement
theorem planes_parallel (m_perpendicular_to_alpha : perpendicular m α)
  (m_perpendicular_to_beta : perpendicular m β) :
  parallel_planes α β := sorry

end planes_parallel_l343_343838


namespace sin_minus_cos_eq_sin_cos_cube_sum_eq_l343_343470

variable (α : ℝ)

-- Given condition
axiom sin_cos_sum : Real.sin α + Real.cos α = 1 / 5

-- Questions to verify
def sin_minus_cos := Real.sin α - Real.cos α
def sin_cos_cube_sum := Real.sin α ^ 3 + Real.cos α ^ 3

-- Statement for proving the first part
theorem sin_minus_cos_eq : sin_minus_cos α = 7 / 5 ∨ sin_minus_cos α = -7 / 5 :=
by sorry

-- Statement for the second part
theorem sin_cos_cube_sum_eq : sin_cos_cube_sum α = 37 / 125 :=
by sorry

end sin_minus_cos_eq_sin_cos_cube_sum_eq_l343_343470


namespace part2_l343_343816

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x - a * x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2 * a * x + 1
noncomputable def has_extreme_points (a x1 x2 : ℝ) [Fact (x1 < x2)] : Prop :=
  g a x1 = 0 ∧ g a x2 = 0

theorem part2 (a x1 x2 : ℝ) (h : has_extreme_points a x1 x2) [Fact (x1 < x2)] :
  -1 / 2 < f a x1 ∧ f a x1 < f a x2 :=
sorry

end part2_l343_343816


namespace batsman_average_after_17th_l343_343342

theorem batsman_average_after_17th (A : ℤ) (h1 : 86 + 16 * A = 17 * (A + 3)) : A + 3 = 38 :=
by
  sorry

end batsman_average_after_17th_l343_343342


namespace sum_of_values_l343_343402

def v (x : ℝ) : ℝ := -x + (3/2) * Real.sin (x * Real.pi / 2)

theorem sum_of_values : 
  v (-3.14) + v (-1) + v (1) + v (3.14) = 0 :=
by
  sorry

end sum_of_values_l343_343402


namespace correct_propositions_l343_343379

-- Definitions based on the given conditions
def prop_1 : Prop :=
  let f := λ x : ℝ, sqrt (x^2 - 1) + sqrt (1 - x^2)
  ∀ x : ℝ, ¬(f (-x) = f x ∧ f (-x) = -f x)

def prop_2 : Prop :=
  ∀ (a b c : ℝ), (a > 0 ∧ b^2 - 4 * a * c ≤ 0) ↔ ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

def prop_3 : Prop :=
  ∀ (f : ℝ → ℝ) (x : ℝ), f (1 - x) = f (x - 1) ↔ ∀ x : ℝ, f (x) = f (-x)

def prop_4 : Prop :=
  ∀ (A : ℝ) (ω : ℝ) (φ : ℝ),
    A ≠ 0 → (λ x, A * cos (ω * x + φ)) = (λ x, - A * cos (ω * x + φ)) ↔ ∃ k : ℤ, φ = (π / 2) + k * π

def prop_5 : Prop :=
  ∀ (x : ℝ), (0 < x ∧ x < π) → let y := sin x + 2 / sin x in y ≥ 2 * sqrt 2

-- The main theorem proving which propositions are correct
theorem correct_propositions (h1 : ¬ prop_1) (h2 : prop_2) (h3 : ¬ prop_3) (h4 : prop_4) (h5 : ¬ prop_5) :
  (prop_2 ∧ prop_4) :=
by {
  exact and.intro h2 h4,
  sorry
}

end correct_propositions_l343_343379


namespace sum_of_first_three_is_10_l343_343646

-- Conditions
def red_cards := {1, 2, 3, 4, 5}
def blue_cards := {2, 3, 4, 5, 6}

def alternates (seq : List (ℕ × ℕ)) : Prop :=
  ∀ i < seq.length - 1, (i % 2 = 0 → (seq[i].fst ∈ red_cards ∧ seq[i].snd ∈ blue_cards)) ∧
                         (i % 2 = 1 → (seq[i].fst ∈ blue_cards ∧ seq[i].snd ∈ red_cards))

def sum_pairs_multiple_of_3 (seq : List (ℕ × ℕ)) : Prop :=
  ∀ i < seq.length - 1, (seq[i].fst + seq[i].snd) % 3 = 0

-- Target Sum
def sum_first_three_correct (seq : List (ℕ × ℕ)) : Prop :=
  seq[0].fst + seq[0].snd + seq[1].fst = 10

-- Proof Problem
theorem sum_of_first_three_is_10 :
  ∃ seq : List (ℕ × ℕ), alternates seq ∧ sum_pairs_multiple_of_3 seq ∧ sum_first_three_correct seq :=
by
  sorry

end sum_of_first_three_is_10_l343_343646


namespace ratio_of_efficiencies_l343_343583

noncomputable theory

-- Define Ram's and Krish's efficiencies
variables (R K : ℝ)

-- Given conditions
def condition1 : Prop := R = (1/2) * K
def condition2 : Prop := R = 1/21
def condition3 : Prop := R + K = 1/7

-- The proof goal
theorem ratio_of_efficiencies (h1 : condition1) (h2 : condition2) (h3 : condition3) : R / K = 1 / 2 :=
by sorry

end ratio_of_efficiencies_l343_343583


namespace sum_of_first_n_terms_b_l343_343479

-- Define the sequences and conditions
def a (n : ℕ) : ℕ := 2^n -- general term formula for a_n

def b (n : ℕ) : ℕ := n / a (2 * n - 1) -- formula for b_n

-- Conditions
axiom a6 : a 6 = 64 -- condition a_6 = 64

axiom arithmetic_mean_eq : (a 4 + a 5) / 2 = 3 * a 3 -- arithmetic mean condition

-- Main theorem to prove
theorem sum_of_first_n_terms_b (n : ℕ) : 
    let T_n := ∑ i in range n, b i in
    T_n = (8 / 9 : ℚ) - (1 / 9 : ℚ) * (1 / 2^(2 * n - 3)) - (n / (3 * 2^(2 * n - 1))) :=
sorry

end sum_of_first_n_terms_b_l343_343479


namespace cube_face_opposites_l343_343738

noncomputable theory

def cube_labels : List String := ["小", "学", "希", "望", "杯", "赛"]

-- Given a function opposite_face which provides the opposite face character for a given character
def opposite_face : String → String
| "希" => "赛"
| "望" => "小"
| "杯" => "学"
| "赛" => "希"
| "小" => "望"
| "学" => "杯"
| _ => ""

theorem cube_face_opposites :
  opposite_face "希" = "赛" ∧
  opposite_face "望" = "小" ∧
  opposite_face "杯" = "学" :=
by
  -- Proof would go here, skipping with sorry.
  sorry

end cube_face_opposites_l343_343738


namespace pos_sum_inequality_l343_343077

theorem pos_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) ≥ (a + b + c) / 2 := 
sorry

end pos_sum_inequality_l343_343077


namespace paying_students_pay_7_l343_343741

/-- At a school, 40% of the students receive a free lunch. 
These lunches are paid for by making sure the price paid by the 
paying students is enough to cover everyone's meal. 
It costs $210 to feed 50 students. 
Prove that each paying student pays $7. -/
theorem paying_students_pay_7 (total_students : ℕ) 
  (free_lunch_percentage : ℤ)
  (cost_per_50_students : ℕ) : 
  free_lunch_percentage = 40 ∧ cost_per_50_students = 210 →
  ∃ (paying_students_pay : ℕ), paying_students_pay = 7 :=
by
  -- Let the proof steps and conditions be set up as follows
  -- (this part is not required, hence using sorry)
  sorry

end paying_students_pay_7_l343_343741


namespace vasya_wins_strategy_l343_343055

/-!
  Given a grid of size 2020 × 2021. Petya and Vasya take turns placing chips 
  in free cells of the grid. Petya goes first. A player wins if after their move,
  every 4 × 4 square on the board contains at least one chip. Prove that Vasya
  can guarantee themselves a victory regardless of the opponent's moves.
-/

theorem vasya_wins_strategy :
  ∀ (grid : ℕ × ℕ) (initial_turn : bool) (win_condition : ℕ × ℕ → Prop),
  grid = (2020, 2021) →
  initial_turn = tt →  -- Petya goes first (true means Petya's turn, false means Vasya's turn)
  (∀ x y, win_condition (x, y) ↔ (x <= 2016 ∧ y <= 2017)) →
  ∃ strategy : (ℕ × ℕ) → ℕ × ℕ,
    ∀ current_position : ℕ × ℕ,
      (win_condition current_position → strategy current_position = (0, 0)) ∨
      strategy current_position ≠ (0, 0) →
    strategy = (λ p, p) :=
begin
  -- The actual proof omitted
  sorry
end

end vasya_wins_strategy_l343_343055


namespace roots_real_condition_l343_343556

theorem roots_real_condition (p q : ℝ) :
  (∀ z : ℂ, z^2 - (14 + p * complex.I) * z + (48 + q * complex.I) = 0 → (∃ x y : ℝ, z = x + y * complex.I)) →
  p = 0 ∧ q = 0 := 
sorry

end roots_real_condition_l343_343556


namespace common_ratio_geometric_series_l343_343427

theorem common_ratio_geometric_series :
  let a := 2 / 3
  let b := 4 / 9
  let c := 8 / 27
  (b / a = 2 / 3) ∧ (c / b = 2 / 3) → 
  ∃ r : ℚ, r = 2 / 3 ∧ ∀ n : ℕ, (a * r^n) = (a * (2 / 3)^n) :=
by
  sorry

end common_ratio_geometric_series_l343_343427


namespace digit_count_product_l343_343398

theorem digit_count_product (a b : ℕ)
  (ha : a = 925743857234987123123) (hb : b = 10345678909876)
  (ha_len : a.digits 10 = 21) (hb_len : b.digits 10 = 15) :
  (a * b).digits 10 = 36 := 
sorry

end digit_count_product_l343_343398


namespace freshmen_more_than_sophomores_l343_343334

noncomputable def total_students := 800
noncomputable def juniors := 0.28 * total_students
noncomputable def not_sophomores := 0.75 * total_students
noncomputable def sophomores := total_students - not_sophomores
noncomputable def seniors := 160
noncomputable def freshmen := total_students - (sophomores + juniors + seniors)

theorem freshmen_more_than_sophomores : freshmen - sophomores = 16 :=
by
  sorry

end freshmen_more_than_sophomores_l343_343334


namespace cookies_baked_total_l343_343109

   -- Definitions based on the problem conditions
   def cookies_yesterday : ℕ := 435
   def cookies_this_morning : ℕ := 139

   -- The theorem we want to prove
   theorem cookies_baked_total : cookies_yesterday + cookies_this_morning = 574 :=
   by sorry
   
end cookies_baked_total_l343_343109


namespace focus_of_parabola_y_2x2_l343_343022

theorem focus_of_parabola_y_2x2 :
  ∃ f, f = 1 / 8 ∧ (∀ x, sqrt (x^2 + (2*x^2 - f)^2) = abs (2*x^2 - (-f)))
:= sorry

end focus_of_parabola_y_2x2_l343_343022


namespace find_f_of_f_21_over_4_l343_343553

noncomputable def f (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 0 then 4 * x^2 - 2
  else if 0 < x ∧ x < 1 then x
  else f (x - 3)

theorem find_f_of_f_21_over_4 : f (f (21 / 4)) = 1 / 4 := by
  sorry

end find_f_of_f_21_over_4_l343_343553


namespace solve_n_product_prime_eq_l343_343042

def product_of_primes_less_than (n : ℕ) : ℕ :=
  if n ≤ 2 then 1 else
  List.prod (List.filter Nat.prime (List.range n))

theorem solve_n_product_prime_eq (n : ℕ) (h_n_gt_3 : n > 3) :
  (n * product_of_primes_less_than n = 2 * n + 16) → n = 7 :=
  sorry

end solve_n_product_prime_eq_l343_343042


namespace MA_eq_BN_l343_343549

variables (circle : Type) (M N A B A' B' : point circle)

-- Conditions
def isChordOfCircle (M N : point circle) (circ : circle) : Prop := sorry
def isDiameter (A B : point circle) (circ : circle) : Prop := sorry
def orthogonalProjection (A : point circle) (MN : line) : point := sorry

-- Theorem statement
theorem MA_eq_BN :
  ∀ (circle : Type) (M N A B A' B' : point circle),
  isChordOfCircle M N circle → 
  isDiameter A B circle →
  A' = orthogonalProjection A (line_through M N) →
  B' = orthogonalProjection B (line_through M N) →
  dist M A' = dist B' N :=
sorry

end MA_eq_BN_l343_343549


namespace initial_books_donations_l343_343648

variable {X : ℕ} -- Initial number of book donations

def books_donated_during_week := 10 * 5
def books_borrowed := 140
def books_remaining := 210

theorem initial_books_donations :
  X + books_donated_during_week - books_borrowed = books_remaining → X = 300 :=
by
  intro h
  sorry

end initial_books_donations_l343_343648


namespace comparison_a_b_c_l343_343450

theorem comparison_a_b_c :
  let a := (1 / 2) ^ (1 / 3)
  let b := (1 / 3) ^ (1 / 2)
  let c := Real.log (3 / Real.pi)
  c < b ∧ b < a :=
by
  sorry

end comparison_a_b_c_l343_343450


namespace problem_solution_l343_343756

-- Define p as the product of the digits of a number x in the decimal system
def p (x : ℕ) : ℕ := (toDigits 10 x).prod id

-- Define the condition (equation)
def polynomial_condition (x : ℕ) : Prop := p x = x^2 - 10 * x - 22

-- The statement to be proved in Lean
theorem problem_solution :
  ∀ x, polynomial_condition x → x = 12 :=
by
  sorry

end problem_solution_l343_343756


namespace sally_cards_l343_343248

theorem sally_cards (x : ℕ) (h1 : 27 + x + 20 = 88) : x = 41 := by
  sorry

end sally_cards_l343_343248


namespace range_of_possible_slopes_l343_343455

theorem range_of_possible_slopes:
  let circle_eq := ∀ x y, x^2 + y^2 - 4*x - 4*y - 10 = 0 in
  let at_least_three_points_distance := ∀ l, ∃ a b: ℝ, (a ≠ 0 ∨ b ≠ 0) ∧ (∀ x y, circle_eq x y → (abs (a*x + b*y) / real.sqrt (a*a + b*b) = 2*real.sqrt 2)) in
  at_least_three_points_distance (a*x + b*y = 0) →
  ∃ k: ℝ, (2 - real.sqrt 3 ≤ k ∧ k ≤ 2 + real.sqrt 3) :=
sorry

end range_of_possible_slopes_l343_343455


namespace focus_of_parabola_y_2x2_l343_343020

theorem focus_of_parabola_y_2x2 :
  ∃ f, f = 1 / 8 ∧ (∀ x, sqrt (x^2 + (2*x^2 - f)^2) = abs (2*x^2 - (-f)))
:= sorry

end focus_of_parabola_y_2x2_l343_343020


namespace intersection_eq_l343_343481

-- Universal set and its sets M and N
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 > 9}
def N : Set ℝ := {x | -1 < x ∧ x < 4}
def complement_N : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}

-- Prove the intersection
theorem intersection_eq :
  M ∩ complement_N = {x | x < -3 ∨ x ≥ 4} :=
by
  sorry

end intersection_eq_l343_343481


namespace minimum_period_myFunction_l343_343622

noncomputable def myFunction : ℝ → ℝ := λ x, sin (2 * x) * cos (2 * x)

theorem minimum_period_myFunction : (∃ p > 0, ∀ x, myFunction (x + p) = myFunction x) ∧ (∀ q > 0, (∀ x, myFunction (x + q) = myFunction x) → q ≥ (π / 2)) :=
by
  sorry

end minimum_period_myFunction_l343_343622


namespace fill_time_l343_343174

theorem fill_time (start_time : ℕ) (initial_rainfall : ℕ) (subsequent_rainfall_rate : ℕ) (subsequent_rainfall_duration : ℕ) (final_rainfall_rate : ℕ) (tank_height : ℕ) :
  start_time = 13 ∧
  initial_rainfall = 2 ∧
  subsequent_rainfall_rate = 1 ∧
  subsequent_rainfall_duration = 4 ∧
  final_rainfall_rate = 3 ∧
  tank_height = 18 →
  13 + 1 + subsequent_rainfall_duration = 18 - 6 / final_rainfall_rate + 18 - tank_height * 10 :=
begin
  sorry
end

end fill_time_l343_343174


namespace largest_common_term_l343_343383

/-- 
An arithmetic sequence with first term 3 has a common difference of 8.
A second sequence begins with 5 and has a common difference of 9.
In the range between 1 and 150, the largest number common to both sequences is 131.
-/
theorem largest_common_term (a_n b_n : ℕ → ℕ) (n : ℕ) :
  ∀ (n1 n2 : ℕ), (a_n n1 = 3 + 8 * (n1 - 1)) → (b_n n2 = 5 + 9 * (n2 - 1)) → 
  1 ≤ 3 + 8 * (n1 - 1) ∧ 3 + 8 * (n1 - 1) ≤ 150 → 
  1 ≤ 5 + 9 * (n2 - 1) ∧ 5 + 9 * (n2 - 1) ≤ 150 → 
  (∃ m : ℕ, 3 + 8 * (n1 - 1) = 59 + 72 * m ∨ 5 + 9 * (n2 - 1) = 59 + 72 * m) → 
  ∃ k, 3 + 8 * (n1 - 1) = 5 + 9 * (n2 - 1) ∧ 
  ∃ y, 1 ≤ y ∧ y ≤ 150 ∧ 3 + 8 * (n1 - 1) = y ∧ 
  y = 131 := 
by
  sorry

end largest_common_term_l343_343383


namespace faster_speed_eq_15_l343_343360

theorem faster_speed_eq_15 :
  ∀ (v d t : ℝ), d = 30 ∧ t = 3 ∧ (d + 15) = v * t → v = 15 :=
by
  intros v d t h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw h4,
  rw h1,
  norm_num,
  norm_num at h3,
  linarith

end faster_speed_eq_15_l343_343360


namespace probability_of_two_queens_or_at_least_one_king_l343_343120

def probability_two_queens_or_at_least_one_king : ℚ := 2 / 13

theorem probability_of_two_queens_or_at_least_one_king :
  let probability_two_queens := (4/52) * (3/51)
  let probability_exactly_one_king := (2 * (4/52) * (48/51))
  let probability_two_kings := (4/52) * (3/51)
  let probability_at_least_one_king := probability_exactly_one_king + probability_two_kings
  let total_probability := probability_two_queens + probability_at_least_one_king
  total_probability = probability_two_queens_or_at_least_one_king := 
by
  sorry

end probability_of_two_queens_or_at_least_one_king_l343_343120


namespace mix_solutions_l343_343499

theorem mix_solutions {x : ℝ} (h : 0.60 * x + 0.75 * (20 - x) = 0.72 * 20) : x = 4 :=
by
-- skipping the proof with sorry
sorry

end mix_solutions_l343_343499


namespace part1_part2_l343_343090

noncomputable def f (a x : ℝ) : ℝ := x - a^x

theorem part1 (b : ℝ) (h : ∀ x : ℝ, 0 ≤ x → f exp x ≤ b - (1/2) * x^2) : b ≥ -1 := sorry

noncomputable def g (a : ℝ) : ℝ := 
  let t := (Real.log (1 / Real.log a)) / (Real.log a) in
  t - a^t

theorem part2 : ∀ a > 1, min_value (g a) = -1 := sorry

end part1_part2_l343_343090


namespace intersection_of_M_and_N_l343_343492

open Set

noncomputable def M := {x : ℝ | ∃ y:ℝ, y = Real.log (2 - x)}
noncomputable def N := {x : ℝ | x^2 - 3*x - 4 ≤ 0 }
noncomputable def I := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = I := 
  sorry

end intersection_of_M_and_N_l343_343492


namespace number_of_solutions_eq_43_l343_343507

theorem number_of_solutions_eq_43 :
  let S := {x | x ∈ (Finset.range 51)} -- S = {1, 2, 3, ..., 50}
  let T := {x | x ∈ {1, 4, 9, 16, 25, 36, 49}} -- T = {1, 4, 9, 16, 25, 36, 49}
  |S| - |T| = 43 :=
by
  sorry -- No proof required, placeholder

end number_of_solutions_eq_43_l343_343507


namespace ratio_combined_areas_semi_circles_l343_343592
 
theorem ratio_combined_areas_semi_circles {R : ℝ} (hR : R > 0) :
  let r := (2 / 3) * R in
  let area_semicircle := (2 / 9) * real.pi * R^2 in
  let combined_area := 2 * area_semicircle in
  let area_circle := real.pi * R^2 in
  (combined_area / area_circle) = (4 / 9) := by
  sorry

end ratio_combined_areas_semi_circles_l343_343592


namespace range_of_m_l343_343865

def f (x : ℝ) : ℝ := |x - 2|

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x + (x + 5) ≥ m) → m ∈ set.Iic 5 :=
begin
  -- This is the translated mathematical proof problem.
  sorry
end

end range_of_m_l343_343865


namespace correct_expression_l343_343322

theorem correct_expression :
  ¬ (|4| = -4) ∧
  ¬ (|4| = -4) ∧
  (-(4^2) ≠ 16)  ∧
  ((-4)^2 = 16) := by
  sorry

end correct_expression_l343_343322


namespace percentage_of_fish_gone_bad_l343_343185

-- Definitions based on conditions
def fish_per_roll : ℕ := 40
def total_fish_bought : ℕ := 400
def sushi_rolls_made : ℕ := 8

-- Definition of fish calculations
def total_fish_used (rolls: ℕ) (per_roll: ℕ) : ℕ := rolls * per_roll
def fish_gone_bad (total : ℕ) (used : ℕ) : ℕ := total - used
def percentage (part : ℕ) (whole : ℕ) : ℚ := (part : ℚ) / (whole : ℚ) * 100

-- Theorem to prove the percentage of bad fish
theorem percentage_of_fish_gone_bad :
  percentage (fish_gone_bad total_fish_bought (total_fish_used sushi_rolls_made fish_per_roll)) total_fish_bought = 20 := by
  sorry

end percentage_of_fish_gone_bad_l343_343185


namespace water_added_l343_343139

theorem water_added (initial_volume : ℕ) (ratio_milk_water_initial : ℚ) 
  (ratio_milk_water_final : ℚ) (w : ℕ)
  (initial_volume_eq : initial_volume = 45)
  (ratio_milk_water_initial_eq : ratio_milk_water_initial = 4 / 1)
  (ratio_milk_water_final_eq : ratio_milk_water_final = 6 / 5)
  (final_ratio_eq : ratio_milk_water_final = 36 / (9 + w)) :
  w = 21 := 
sorry

end water_added_l343_343139


namespace value_of_int_part_l343_343943

noncomputable def int_part (x : ℝ) : ℤ := ⌊x⌋

theorem value_of_int_part:
  int_part (- real.sqrt 17 + 1) = -4 :=
by
  sorry

end value_of_int_part_l343_343943


namespace evaluate_expression_l343_343784

theorem evaluate_expression:
  (125 = 5^3) ∧ (81 = 3^4) ∧ (32 = 2^5) → 
  125^(1/3) * 81^(-1/4) * 32^(1/5) = 10 / 3 := by
  sorry

end evaluate_expression_l343_343784


namespace trigonometric_identity_l343_343759

theorem trigonometric_identity : 
  (\sin (24 * real.pi / 180) * \cos (16 * real.pi / 180) + \cos (156 * real.pi / 180) * \cos (96 * real.pi / 180))
  /
  (\sin (26 * real.pi / 180) * \cos (14 * real.pi / 180) + \cos (154 * real.pi / 180) * \cos (94 * real.pi / 180))
  = 1 := by
  sorry

end trigonometric_identity_l343_343759


namespace solution_set_inequality_l343_343946

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃a b⦄, a ∈ s → b ∈ s → a < b → f a ≤ f b

axiom f_even : even_function f
axiom f_increasing : increasing_on f (Set.Ioi 0)
axiom f_at_2 : f 2 = 0

theorem solution_set_inequality :
  { x : ℝ | (f x + f (-x)) / x < 0 } = Set.Ioo (-∞) (-2) ∪ Set.Ioo 0 2 := sorry

end solution_set_inequality_l343_343946


namespace trigonometric_identity_l343_343990

theorem trigonometric_identity (x y : ℝ) :
  sin (x - y + π / 6) * cos (y + π / 6) + cos (x - y + π / 6) * sin (y + π / 6) = sin (x + π / 3) :=
by
  sorry

end trigonometric_identity_l343_343990


namespace f_23_plus_f_neg14_l343_343117

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x, f (x + 5) = f x
axiom odd_f : ∀ x, f (-x) = -f x
axiom f_one : f 1 = 1
axiom f_two : f 2 = 2

theorem f_23_plus_f_neg14 : f 23 + f (-14) = -1 := by
  sorry

end f_23_plus_f_neg14_l343_343117


namespace div_30_div_510_div_66_div_large_l343_343670

theorem div_30 (a : ℤ) : 30 ∣ (a^5 - a) := 
  sorry  

theorem div_510 (a : ℤ) : 510 ∣ (a^17 - a) := 
  sorry

theorem div_66 (a : ℤ) : 66 ∣ (a^11 - a) := 
  sorry

theorem div_large (a : ℤ) : (2 * 3 * 5 * 7 * 13 * 19 * 37 * 73) ∣ (a^73 - a) := 
  sorry  

end div_30_div_510_div_66_div_large_l343_343670


namespace sin_mul_tan_neg_of_second_quadrant_l343_343468

theorem sin_mul_tan_neg_of_second_quadrant (α : ℝ) (h1 : sin α > 0) (h2 : cos α < 0) (h3 : tan α < 0) :
  sin α * tan α < 0 :=
by
  sorry

end sin_mul_tan_neg_of_second_quadrant_l343_343468


namespace problem_l343_343817

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem (a b c : ℝ) (h0 : f a b c 0 = f a b c 4) (h1 : f a b c 0 > f a b c 1) :
  a > 0 ∧ 4 * a + b = 0 :=
by
  sorry

end problem_l343_343817


namespace monomial_properties_l343_343327

noncomputable def monomial_coeff : ℚ := -(3/5 : ℚ)

def monomial_degree (x y : ℤ) : ℕ :=
  1 + 2

theorem monomial_properties (x y : ℤ) :
  monomial_coeff = -(3/5) ∧ monomial_degree x y = 3 :=
by
  -- Proof is to be filled here
  sorry

end monomial_properties_l343_343327


namespace combined_stickers_count_l343_343192

theorem combined_stickers_count :
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given
  june_total + bonnie_total = 189 :=
by
  -- Definitions
  let initial_june_stickers := 76
  let initial_bonnie_stickers := 63
  let stickers_given := 25
  
  -- Calculations
  let june_total := initial_june_stickers + stickers_given
  let bonnie_total := initial_bonnie_stickers + stickers_given

  -- Proof is omitted
  sorry

end combined_stickers_count_l343_343192


namespace hyperbola_eccentricity_l343_343102

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
noncomputable def hyperbola (a b : ℝ) : set (ℝ × ℝ) := { (x, y) | x^2 / a^2 - y^2 / b^2 = 1 }

theorem hyperbola_eccentricity (p a b c : ℝ) 
  (hp : 0 < p) (ha : 0 < a) (hb : 0 < b)
  (focus_eq : p = 2 * c) 
  (intersection_point : ∃ x y, (x, y) ∈ parabola p ∧ (x, y) ∈ hyperbola a b ∧ y ≠ 0) : 
  real.sqrt(2) + 1 = (real.sqrt(1 + (b / a)^2)) :=
sorry

end hyperbola_eccentricity_l343_343102


namespace plane_overtake_time_is_80_minutes_l343_343301

noncomputable def plane_overtake_time 
  (speed_a speed_b : ℝ)
  (head_start : ℝ) 
  (t : ℝ) : Prop :=
  speed_a * (t + head_start) = speed_b * t

theorem plane_overtake_time_is_80_minutes :
  plane_overtake_time 200 300 (2/3) (80 / 60)
:=
  sorry

end plane_overtake_time_is_80_minutes_l343_343301


namespace parallel_planes_if_perpendicular_to_same_line_l343_343845

variables {m n : Type} [line m] [line n]
variables {α β : Type} [plane α] [plane β]

theorem parallel_planes_if_perpendicular_to_same_line (h1 : m ⟂ α) (h2 : m ⟂ β) : α ∥ β :=
sorry

end parallel_planes_if_perpendicular_to_same_line_l343_343845


namespace cos_neg245_l343_343814

-- Define the given condition and declare the theorem to prove the required equality
variable (a : ℝ)
def cos_25_eq_a : Prop := (Real.cos 25 * Real.pi / 180 = a)

theorem cos_neg245 :
  cos_25_eq_a a → Real.cos (-245 * Real.pi / 180) = -Real.sqrt (1 - a^2) :=
by
  intro h
  sorry

end cos_neg245_l343_343814


namespace tetrahedron_coloring_l343_343767

def color := ℕ

def is_valid_coloring (tetrahedron : ℕ → color) : Prop :=
  ∀ i j, i ≠ j → tetrahedron i ≠ tetrahedron j

def is_equivalent_colorings (c1 c2 : ℕ → color) : Prop :=
  ∃ (f : ℕ → ℕ), bijective f ∧ (∀ i, c1 i = c2 (f i))

def count_distinguishable_colorings : ℕ :=
  48

theorem tetrahedron_coloring : 
  ∃ m, m = count_distinguishable_colorings ∧ 
       (∃ (colors : ℕ → ℕ), 
          is_valid_coloring colors ∧ 
          is_equivalent_colorings colors colors → 
          m = 48) :=
  sorry

end tetrahedron_coloring_l343_343767


namespace distance_sum_equal_l343_343297

/-- Define points as vectors -/
structure Point :=
(x : ℝ) (y : ℝ)

variables {A B C A' B' C' G P : Point}

/-- Condition: Given triangle ABC and its centroid G -/
-- (We assume that the centroid is properly defined)
/-- Second triangle A'B'C' is obtained by a 180-degree rotation of triangle ABC around G -/
def rotated_triangle (A B C G: Point) : Point × Point × Point :=
  let rotate180 (p q : Point) : Point := ⟨2 * q.x - p.x, 2 * q.y - p.y⟩ in
  (rotate180 A G, rotate180 B G, rotate180 C G)

/-- Define the distance square between two points -/
def dist_square (p q : Point) : ℝ :=
  (p.x - q.x) ^ 2 + (p.y - q.y) ^ 2

/-- Theorem: Sum of the squares of distances from any point P to the vertices of triangle ABC
    is equal to the sum of the squares of distances from P to the vertices of the rotated triangle A'B'C' -/
theorem distance_sum_equal (H1 : rotated_triangle A B C G = (A', B', C')) :
  dist_square P A + dist_square P B + dist_square P C = dist_square P A' + dist_square P B' + dist_square P C' :=
sorry

end distance_sum_equal_l343_343297


namespace customers_not_wanting_change_l343_343732

-- Given Conditions
def cars_initial := 4
def cars_additional := 6
def cars_total := cars_initial + cars_additional
def tires_per_car := 4
def half_change_customers := 2
def tires_for_half_change_customers := 2 * 2 -- 2 cars, 2 tires each
def tires_left := 20

-- Theorem to Prove
theorem customers_not_wanting_change : 
  (cars_total * tires_per_car) - (tires_left + tires_for_half_change_customers) = 
  4 * tires_per_car -> 
  cars_total - ((tires_left + tires_for_half_change_customers) / tires_per_car) - half_change_customers = 4 :=
by
  sorry

end customers_not_wanting_change_l343_343732


namespace sequence_count_19_l343_343880

def f : ℕ → ℕ
| 3 := 1
| 4 := 1
| 5 := 1
| 6 := 2
| 7 := 2
| n := f (n - 4) + 2 * f (n - 5) + f (n - 6)

theorem sequence_count_19 : f 19 = 65 := sorry

end sequence_count_19_l343_343880


namespace evaluate_expression_l343_343775

theorem evaluate_expression : (125^(1/3 : ℝ)) * (81^(-1/4 : ℝ)) * (32^(1/5 : ℝ)) = (10 / 3 : ℝ) :=
by
  sorry

end evaluate_expression_l343_343775


namespace max_sum_15x15_grid_l343_343906

theorem max_sum_15x15_grid : ∀ (f : Fin 15 → Fin 15 → ℕ), 
  (∀ i j, f i j ≤ 4) ∧ 
  (∀ i j, i < 14 → j < 14 → f i j + f (i + 1) j + f i (j + 1) + f (i + 1) (j + 1) = 7) →
  (∑ i j, f i j) ≤ 417 :=
by
  intros f h
  sorry

end max_sum_15x15_grid_l343_343906


namespace number_of_ways_to_assign_tasks_l343_343295

theorem number_of_ways_to_assign_tasks : 
  ∀ (select : ℕ) (total : ℕ) (taskA : ℕ) (taskB : ℕ) (taskC : ℕ),
  total = 10 → select = 4 → taskA = 2 → 
  taskB = 1 → taskC = 1 → 
  ((nat.choose total select) * (nat.choose select taskA) * (nat.perm taskA taskA)) = 2520 :=
by
  intros select total taskA taskB taskC htots hsel htaskA htaskB htaskC
  have h: (nat.choose 10 4) * (nat.choose 4 2) * (nat.perm 2 2) = 2520 sorry

end number_of_ways_to_assign_tasks_l343_343295


namespace base_10_to_5_l343_343314

/-- Define the function to convert a base 10 (decimal) number to a base 5 equivalent list of digits. --/
def to_base_five (n : ℕ) : list ℕ :=
let rec := λ (n : ℕ) (acc : list ℕ), 
  if n = 0 then acc 
  else rec (n / 5) ((n % 5) :: acc)
in rec n []

/-- Prove that the base five equivalent of 158 in base 10 is 1133 in base 5 form --/
theorem base_10_to_5 (n : ℕ) (h : n = 158) : to_base_five n = [1, 1, 3, 3] :=
begin
  rw h,
  reflexivity
end

end base_10_to_5_l343_343314


namespace mode_of_scores_l343_343630

def mode (l : List ℕ) : List ℕ :=
  let grouped := l.groupBy id
  let maxFreq := grouped.map (λ g => g.length).max
  grouped.filter (λ g => g.length = maxFreq).map List.head

theorem mode_of_scores :
  mode [65, 65, 73, 82, 88, 91, 96, 96, 96, 96, 102, 104, 104, 104, 104, 110, 110, 110] = [96, 104] :=
by
  sorry

end mode_of_scores_l343_343630


namespace max_f_is_two_range_g_l343_343084

noncomputable def f (x : ℝ) : ℝ := (√3) * Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := f x * Real.cos x

theorem max_f_is_two : ∃ x ∈ Icc 0 (π/2), f x = 2 := 
sorry

theorem range_g :
  Set.image g (Set.Icc 0 (π/2)) = Set.Icc 1 (3/2) :=
sorry

end max_f_is_two_range_g_l343_343084


namespace median_temperature_l343_343975

open Real

def temps : List ℝ := [-36.5, 13.75, -15.25, -10.5]

def median (l : List ℝ) : ℝ := 
  let sorted := l.qsort (· < ·)
  if sorted.length % 2 = 1 then 
    sorted.get! (sorted.length / 2)
  else 
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2

theorem median_temperature : median temps = -12.875 :=
by
  sorry

end median_temperature_l343_343975


namespace convert_and_round_l343_343247

theorem convert_and_round (euro_amount : ℝ) (exchange_rate : ℝ) (rounded_value : ℝ)
  (h1 : euro_amount = 54.3627)
  (h2 : exchange_rate = 1.1)
  (h3 : rounded_value = 59.80) :
  Real.round (euro_amount * exchange_rate * 100) / 100 = rounded_value :=
by
  have step1 : euro_amount * exchange_rate = 59.79897 := by sorry
  have step2 : Real.round (59.79897 * 100) / 100 = 59.80 := by sorry
  exact step2

end convert_and_round_l343_343247


namespace expected_value_is_1_50_l343_343191

noncomputable def expected_value_of_winnings : ℚ :=
  let primes := {2, 3, 5, 7}
  let num_sides := 8
  let winnings (n : ℕ) : ℚ :=
    if n = 1 then -5
    else if n ∈ primes then n
    else 0
  (∑ i in (finset.range num_sides).map (int.cast), (if i = 0 then 0 else winnings i) / num_sides)

theorem expected_value_is_1_50 : expected_value_of_winnings = 1.5 :=
by sorry

end expected_value_is_1_50_l343_343191


namespace sum_of_underlined_is_positive_l343_343712

theorem sum_of_underlined_is_positive (n : ℕ) (a : ℕ → ℝ) :
  (∀ i < n, (a i > 0) ∨ (∃ k, i + k < n ∧ (∑ j in finset.range (k + 1), a (i + j)) > 0)) →
  (∑ i in finset.range n, if a i > 0 ∨ (∃ k, i + k < n ∧ (∑ j in finset.range (k + 1), a (i + j)) > 0) then a i else 0) > 0 := 
sorry

end sum_of_underlined_is_positive_l343_343712


namespace problem1_solution_problem2_solution_l343_343299

-- Conditions for Problem 1
def problem1_condition (x : ℝ) : Prop := 
  5 * (x - 20) + 2 * x = 600

-- Proof for Problem 1 Goal
theorem problem1_solution (x : ℝ) (h : problem1_condition x) : x = 100 := 
by sorry

-- Conditions for Problem 2
def problem2_condition (m : ℝ) : Prop :=
  (360 / m) + (540 / (1.2 * m)) = (900 / 100)

-- Proof for Problem 2 Goal
theorem problem2_solution (m : ℝ) (h : problem2_condition m) : m = 90 := 
by sorry

end problem1_solution_problem2_solution_l343_343299


namespace sqrt_288_simplified_l343_343251

-- Define the conditions
def factorization_288 : ℕ := 288
def perfect_square_144 : ℕ := 144
def sqrt_144 : ℕ := 12

-- The proof goal
theorem sqrt_288_simplified :
  sqrt factorization_288 = sqrt perfect_square_144 * sqrt 2 :=
by
  sorry

end sqrt_288_simplified_l343_343251


namespace percentage_of_sugar_in_second_solution_l343_343237

theorem percentage_of_sugar_in_second_solution 
  (W : ℝ) (hW : W > 0) :
  (let orig_sugar := 0.10 * W;
       removed_sugar := 0.025 * W;
       remaining_sugar := orig_sugar - removed_sugar;
       final_sugar := 0.16 * W in
  ∃ S : ℝ, let added_sugar := S * (W / 4) in S * 100 = 34 :=
  sorry

end percentage_of_sugar_in_second_solution_l343_343237


namespace arccos_cos_three_l343_343393

-- Defining the problem conditions
def three_radians : ℝ := 3

-- Main statement to prove
theorem arccos_cos_three : Real.arccos (Real.cos three_radians) = three_radians := 
sorry

end arccos_cos_three_l343_343393


namespace rhombus_oa_oc_constant_l343_343737

theorem rhombus_oa_oc_constant (A B C D O : Point) (h_rhombus : isRhombus A B C D)
    (h_AB : dist A B = 4) (h_OB_OD : dist O B = 6 ∧ dist O D = 6) : 
     ∃ k, dist O A * dist O C = k := 
sorry

end rhombus_oa_oc_constant_l343_343737


namespace product_area_perimeter_correct_l343_343572

open Real

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1)^2 * 4 + (q.2 - p.2)^2 * 4)

def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area (a b c : ℝ) (s : ℝ) : ℝ :=
  sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def product_area_perimeter :=
  let p := (1, 5)
  let q := (5, 2)
  let r := (2, 1)
  let pq := distance p q
  let pr := distance p r
  let qr := distance q r
  let s := semiperimeter pq pr qr
  let a := area pq pr qr s
  let perimeter := pq + pr + qr
  a * perimeter

theorem product_area_perimeter_correct :
  product_area_perimeter = 
    sqrt ((10 + sqrt 68 + sqrt 40) / 2 * ((10 + sqrt 68 + sqrt 40) / 2 - 10) 
    * ((10 + sqrt 68 + sqrt 40) / 2 - 2 * sqrt 17) 
    * ((10 + sqrt 68 + sqrt 40) / 2 - 2 * sqrt 10)) 
    * (10 + 2 * sqrt 17 + 2 * sqrt 10) := 
  sorry

end product_area_perimeter_correct_l343_343572


namespace coprime_with_sequence_l343_343460

theorem coprime_with_sequence :
  ∃ m ∈ ℕ, (∀ n ∈ ℕ, a_n = 2^n + 3^n + 6^n - 1 → Nat.coprime m (2^n + 3^n + 6^n - 1) = 1) ∧ m = 2 :=
begin
  sorry
end

end coprime_with_sequence_l343_343460


namespace initial_scooter_value_l343_343292

theorem initial_scooter_value (V : ℝ) (h : V * (3/4)^2 = 22500) : V = 40000 :=
by
  sorry

end initial_scooter_value_l343_343292


namespace area_of_triangle_l343_343135

theorem area_of_triangle (a b c : ℝ) (ha : a = 3) (hb : b = 5) (hc : c = 7) : 
  let A := real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) in
  let sinA := real.sin A in
  (1 / 2) * b * c * sinA = (15 * real.sqrt 3 / 4) :=
by
  rw [ha, hb, hc]
  let A := real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  have sinA_eq : real.sin A = 3 * real.sqrt 3 / 14 := sorry
  rw sinA_eq
  have area_eq : (1 / 2) * b * c * (3 * real.sqrt 3 / 14) = (15 * real.sqrt 3 / 4) := sorry
  exact area_eq

end area_of_triangle_l343_343135


namespace ratio_of_sums_l343_343557

theorem ratio_of_sums (p q r u v w : ℝ) 
  (h1 : p > 0) (h2 : q > 0) (h3 : r > 0) (h4 : u > 0) (h5 : v > 0) (h6 : w > 0)
  (h7 : p^2 + q^2 + r^2 = 49) (h8 : u^2 + v^2 + w^2 = 64)
  (h9 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 :=
by
  sorry

end ratio_of_sums_l343_343557


namespace transport_equivalence_l343_343148

theorem transport_equivalence (f : ℤ → ℤ) (x y : ℤ) (h : f x = -x) :
  f (-y) = y :=
by
  sorry

end transport_equivalence_l343_343148


namespace evaluate_expression_l343_343772

theorem evaluate_expression :
  125^(1/3 : ℝ) * 81^(-1/4 : ℝ) * 32^(1/5 : ℝ) = 10/3 := by
  sorry

end evaluate_expression_l343_343772


namespace prime_land_correct_l343_343137

/-- 
In Prime Land, there are seven major cities labelled C_0, C_1, ..., C_6.
The indices are taken modulo 7, i.e., C_{n+7} = C_n for all n.
Al starts at city C_0.

Each minute for ten minutes, Al flips a fair coin:
- If heads, and he is at city C_k, he moves to city C_{2k % 7};
- If tails, he moves to city C_{2k+1 % 7}.
The probability that Al is back at city C_0 after 10 moves is (m / 1024).
-/
def prime_land_probability : ℕ :=
  let m : ℕ := 147 in
  m

theorem prime_land_correct : prime_land_probability = 147 := 
by
  -- Proof steps will be inserted here. Currently an unproven statement.
  sorry

end prime_land_correct_l343_343137


namespace part1_part2_l343_343091

-- Part 1: Prove values of m and n.
theorem part1 (m n : ℝ) :
  (∀ x : ℝ, |x - m| ≤ n ↔ 0 ≤ x ∧ x ≤ 4) → m = 2 ∧ n = 2 :=
by
  intro h
  -- Proof omitted
  sorry

-- Part 2: Prove the minimum value of a + b.
theorem part2 (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 2) :
  a + b = (2 / a) + (2 / b) → a + b ≥ 2 * Real.sqrt 2 :=
by
  intro h
  -- Proof omitted
  sorry

end part1_part2_l343_343091


namespace parallel_lines_proof_l343_343045

variables {A B C P L M N A' B' C' : Point}
variables {O : Circle}
variables {BC AB CA : Line}

-- Hypotheses and conditions:
-- 1. Point P lies on the circumcircle of triangle ABC.
-- 2. Perpendiculars PL, PM, and PN are drawn from P to sides BC, AB, and CA respectively.
-- 3. PL intersects BC at L, PM intersects AB at M, and PN intersects CA at N.
-- 4. PL, PM, and PN intersect the circumcircle at points A', B', and C' respectively.

-- Given conditions as hypotheses
axiom circum_circle_triangle (tri : Triangle) (circ : Circle) : ∃! O, Circle.circumcircle tri = circ
axiom perpendicular_dropped (p : Point) (lineA lineB lineC : Line) : 
  perpendicular p lineA ∧ perpendicular p lineB ∧ perpendicular p lineC
axiom intersection_on_lines (pt1 pt2 line : Point) (lineA lineB lineC : Line) :
  intersects pt1 lineA ∧ intersects pt2 lineB ∧ intersects line.lineC
axiom intersection_on_circ (p q line : Circle) (lineA lineB lineC : Circle) :
  intersects_circ p lineA ∧ intersects_circ q lineB ∧ intersects_circ lineC.lineC

-- Goal: Proving the given parallel lines
theorem parallel_lines_proof : 
  circum_circle_triangle (ABC) O → 
  perpendicular_dropped P BC AB CA →
  intersection_on_lines L M N BC AB CA → 
  intersection_on_circ A' B' C' O →
  parallel (lineSegment A' A) (lineSegment B' B) ∧ 
  parallel (lineSegment B' B) (lineSegment C' C) := 
sorry

end parallel_lines_proof_l343_343045


namespace perfect_square_value_of_n_l343_343474

theorem perfect_square_value_of_n (n : ℕ) (h1 : 0 < n) (h2 : ∃ k : ℕ, 4^7 + 4^n + 4^{1998} = k^2) :
  n = 1003 ∨ n = 3988 :=
sorry

end perfect_square_value_of_n_l343_343474


namespace cantaloupe_total_l343_343197

noncomputable def total_cantaloupes (keith : ℕ) (fred : ℕ) (jason : ℕ) : ℕ := keith + fred + jason

theorem cantaloupe_total :
  (total_cantaloupes 29 16 20) = 65 :=
by
  unfold total_cantaloupes
  norm_num
  sorry

end cantaloupe_total_l343_343197


namespace combined_area_sandboxes_l343_343934

-- Definitions for the first sandbox
def width_sandbox1 : ℝ := 5
def length_sandbox1 : ℝ := 2 * width_sandbox1
def perimeter_sandbox1 : ℝ := 2 * length_sandbox1 + 2 * width_sandbox1

-- Definitions for the second sandbox
def diagonal_sandbox2 : ℝ := 15
def width_sandbox2 : ℝ := Real.sqrt (22.5)
def length_sandbox2 : ℝ := 3 * width_sandbox2

-- Calculating the area
def area_sandbox1 : ℝ := length_sandbox1 * width_sandbox1
def area_sandbox2 : ℝ := length_sandbox2 * width_sandbox2

-- The combined area
theorem combined_area_sandboxes : (area_sandbox1 + area_sandbox2) ≈ 117.42 := by
  sorry

end combined_area_sandboxes_l343_343934


namespace no_child_moves_after_n_plus_one_commands_l343_343901

theorem no_child_moves_after_n_plus_one_commands (n : ℕ) (initial_directions : Fin n → Bool) :
  (∀ k, (k < n) → initial_directions k ∈ { true, false }) →
  ∀ m, (m ≥ n) → (∀ i, (i < n) → (initial_directions i = initial_directions (i + 1))) →
  True := 
by
  sorry

end no_child_moves_after_n_plus_one_commands_l343_343901


namespace sum_of_first_100_triangular_numbers_l343_343606

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_first_100_triangular_numbers : 
  ∑ n in Finset.range 100, triangular_number (n + 1) = 171700 := by
  sorry

end sum_of_first_100_triangular_numbers_l343_343606


namespace solve_inequality_l343_343993

theorem solve_inequality (a : ℝ) (x : ℝ) :
  (a = 0 → x > 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (0 < a → a < 2 → 1 < x → x < 2 / a → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a = 2 → False → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a > 2 → 2 / a < x → x < 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a < 0 → ((x < 2 / a ∨ x > 1) → (ax^2 - (a + 2) * x + 2 < 0))) := sorry

end solve_inequality_l343_343993


namespace no_such_function_l343_343202

-- Define the properties of a convex quadrilateral
structure ConvexQuadrilateral (P : Type) :=
(A B C D : P)
(isConvex : ∀ {x : P}, x ∉ {A, B, C, D} → x lies outside the quadrilateral)

-- Define the properties of a concave quadrilateral
structure ConcaveQuadrilateral (P : Type) :=
(A B C D : P)
(isConcave : ∃ i j k l : ℕ, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧ i ∈ {1, 2, 3, 4} ∧ ...)

-- Define the function property f : P → P
def function_property {P : Type} (f : P → P) : Prop :=
∀ (Q : ConvexQuadrilateral P),
  let fQ := { f(Q.A), f(Q.B), f(Q.C), f(Q.D) } in
  ConcaveQuadrilateral.mk fQ

-- State the theorem we want to prove
theorem no_such_function (P : Type) (f : P → P) :
  ¬ function_property f :=
sorry

end no_such_function_l343_343202


namespace batsman_average_after_17th_inning_l343_343686

theorem batsman_average_after_17th_inning 
    (A : ℕ) 
    (hA : A = 15) 
    (runs_17th_inning : ℕ)
    (increase_in_average : ℕ) 
    (hscores : runs_17th_inning = 100)
    (hincrease : increase_in_average = 5) :
    (A + increase_in_average = 20) :=
by
  sorry

end batsman_average_after_17th_inning_l343_343686


namespace enclosed_area_of_equation_l343_343653

theorem enclosed_area_of_equation :
  (∀ x y : ℝ, x^2 + y^2 = 2 * (|x| + |y|) → true) →
  (1 / 2 * ∫ (x : ℝ) in 0..2, (2 - x) + (x - 2) → real.pi) :=
by
  sorry

end enclosed_area_of_equation_l343_343653


namespace toys_produced_per_day_l343_343669

theorem toys_produced_per_day :
  (3400 / 5 = 680) :=
by
  sorry

end toys_produced_per_day_l343_343669


namespace simplify_polynomial_l343_343988

variable (x : ℝ)

theorem simplify_polynomial : (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 6) = -4 * x + 3 :=
by
  sorry

end simplify_polynomial_l343_343988


namespace total_seeds_eaten_proof_l343_343387

-- Define the information about the number of seeds eaten by each player
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds

-- Sum the seeds eaten by all the players
def total_seeds_eaten : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds

-- Prove that the total number of seeds eaten is 380
theorem total_seeds_eaten_proof : total_seeds_eaten = 380 :=
by
  -- To be filled in by actual proof steps
  sorry

end total_seeds_eaten_proof_l343_343387


namespace maximal_distance_sum_l343_343537

-- Define the conditions given in a)
def polar_line (ρ θ : ℝ): Prop := ρ * Real.sin (θ + π/3) = 4
def polar_circle (ρ θ : ℝ): Prop := ρ = 4 * Real.sin θ

-- Translate the condition of rectangular coordinate system
def rectangular_line_eq (x y: ℝ): Prop := sqrt 3 * x + y - 8 = 0
def polar_to_rect (ρ θ: ℝ) := (ρ * Real.cos θ, ρ * Real.sin θ)
def parametric_circle (θ : ℝ) := (2 * Real.cos θ, 2 + 2 * Real.sin θ)

def distances (P : ℝ × ℝ) : ℝ × ℝ :=
let (Px, Py) := P in
(d := (abs (sqrt 3 * Px + Py - 8) / sqrt (1 + (sqrt 3)^2)),
 Py)

-- The theorem to prove
theorem maximal_distance_sum :
  ∀ θ : ℝ, 
  let P := parametric_circle θ in 
  let (d1, d2) := distances P in 
  P ∈ (polar_to_rect <$> univ.filter (polar_circle ρ θ)) ∧ d1 + d2 ≤ 7 :=
begin
  sorry -- proof placeholder
end

end maximal_distance_sum_l343_343537


namespace prime_root_quadratic_l343_343454

open Nat

theorem prime_root_quadratic (p : ℕ) (hp : Prime p)
  (h : ∃ x y : ℤ, x^2 - 2 * (p:ℤ) * x + (p^2 - 5 * (p:ℤ) - 1) = 0 ∧ y^2 - 2 * (p:ℤ) * y + (p^2 - 5 * (p:ℤ) - 1) = 0) :
  p = 3 ∨ p = 7 :=
by
  sorry

end prime_root_quadratic_l343_343454


namespace find_CE_l343_343144

-- Definitions and Conditions
variables (A B C D E : Type*) -- Points
variables (AB CD : ℝ) -- Lengths of the bases
variables (AC CE : ℝ) -- Lengths of the diagonals and segments

-- The conditions from the problem
def is_trapezoid (A B C D E : Type*)
  (AB CD AC : ℝ) (CE : ℝ) : Prop :=
  CD ≠ 0 ∧ AC = AB + CD

def trapezoid (A B C D E : Type*) :=
  A B C D E ∧ 
  (∃ AB CD CE AC, 
    is_trapezoid A B C D E AB CD AC CE ∧
    AC = 15 ∧
    AB = (3/2) * CD)

-- The theorem to be proved
theorem find_CE (A B C D E : Type*)
  (AB CD : ℝ)
  (h1 : AB = (3/2) * CD)
  (h2 : AC = 15) :
  CE = 6 :=
by
  sorry

end find_CE_l343_343144


namespace tank_filled_by_10pm_l343_343178

-- Defining the problem setup
def starts_raining (t : Real) : Prop := t ≥ 13  -- 13 represents 1 pm
def rainfall_rate (t : Real) : Real :=
  if t < 14 then 2       -- 2 inches from 1 pm to 2 pm
  else if t < 18 then 1  -- 1 inch per hour from 2 pm to 6 pm
  else 3                 -- 3 inches per hour from 6 pm onwards
def tank_height : Real := 18 -- Height of the fish tank is 18 inches

-- Calculating the total rainfall by integrating the rainfall rate
noncomputable def total_rainfall (t : Real) : Real :=
  ∫ τ in (13:ℝ)..t, rainfall_rate τ

-- The statement of the proof problem
theorem tank_filled_by_10pm : total_rainfall 22 = tank_height :=
by
  sorry

end tank_filled_by_10pm_l343_343178


namespace deepak_current_age_l343_343671

variable (A D : ℕ)

def ratio_condition : Prop := A * 5 = D * 2
def arun_future_age (A : ℕ) : Prop := A + 10 = 30

theorem deepak_current_age (h1 : ratio_condition A D) (h2 : arun_future_age A) : D = 50 := sorry

end deepak_current_age_l343_343671


namespace percentage_calculation_l343_343791

-- Define total and part amounts
def total_amount : ℕ := 800
def part_amount : ℕ := 200

-- Define the percentage calculation
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Theorem to show the percentage is 25%
theorem percentage_calculation :
  percentage part_amount total_amount = 25 :=
sorry

end percentage_calculation_l343_343791


namespace common_ratio_of_geometric_series_l343_343424

theorem common_ratio_of_geometric_series : ∃ r : ℝ, ∀ n : ℕ, 
  r = (if n = 0 then 2 / 3
       else if n = 1 then (2 / 3) * (2 / 3)
       else if n = 2 then (2 / 3) * (2 / 3) * (2 / 3)
       else sorry)
  ∧ r = 2 / 3 := sorry

end common_ratio_of_geometric_series_l343_343424


namespace samBill_l343_343356

def textMessageCostPerText := 8 -- cents
def extraMinuteCostPerMinute := 15 -- cents
def planBaseCost := 25 -- dollars
def includedPlanHours := 25
def centToDollar (cents: Nat) : Nat := cents / 100

def totalBill (texts: Nat) (hours: Nat) : Nat :=
  let textCost := centToDollar (texts * textMessageCostPerText)
  let extraHours := if hours > includedPlanHours then hours - includedPlanHours else 0
  let extraMinutes := extraHours * 60
  let extraMinuteCost := centToDollar (extraMinutes * extraMinuteCostPerMinute)
  planBaseCost + textCost + extraMinuteCost

theorem samBill :
  totalBill 150 26 = 46 := 
sorry

end samBill_l343_343356


namespace ticket_distribution_count_l343_343411

-- Defining the setup: six tickets, four people
def tickets : List ℕ := [1, 2, 3, 4, 5, 6]
def people : List Char := ['A', 'B', 'C', 'D']

-- Condition definitions: each person gets at least one and at most two consecutive tickets
def consecutive_splits (lst : List ℕ) (num_parts : ℕ) : List (List ℕ) → Prop :=
  λ parts, parts.length = num_parts ∧
           ∀ part ∈ parts, part.length ≥ 1 ∧ part.length ≤ 2 ∧
                           ((∀ i j, i < j → (part.get? i).get_or_else 0 + 1 = (part.get? j).get_or_else 0))

-- Proof problem statement
theorem ticket_distribution_count :
  ∃ dists : List (Char × (List ℕ)),
    (∀ (a : Char), a ∈ people → ∃ p, (a, p) ∈ dists) ∧
    ∃ ps, consecutive_splits tickets 4 ps ∧
          List.permutations (List.zip people ps) = dists ∧
          dists.length = 144 := sorry

end ticket_distribution_count_l343_343411


namespace find_x0_l343_343089

noncomputable def f (a c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + c

theorem find_x0 (a c : ℝ) (h : a ≠ 0) :
  ∃ x0 : ℝ, -1 < x0 ∧ x0 < 0 ∧ ∫ x in 0..1, f a c x = f a c x0 → x0 = - (real.sqrt 3) / 3 :=
by
  sorry

end find_x0_l343_343089


namespace function_property_l343_343955

theorem function_property (g : ℝ → ℝ) (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → g(x * y) = g(x) / y)
  (h₂ : g 800 = 4) : g 1000 = 16 / 5 := 
sorry

end function_property_l343_343955


namespace steven_has_15_more_peaches_than_jill_l343_343184

-- Definitions based on conditions
def peaches_jill : ℕ := 12
def peaches_jake : ℕ := peaches_jill - 1
def peaches_steven : ℕ := peaches_jake + 16

-- The proof problem
theorem steven_has_15_more_peaches_than_jill : peaches_steven - peaches_jill = 15 := by
  sorry

end steven_has_15_more_peaches_than_jill_l343_343184


namespace solve_equation_l343_343036

theorem solve_equation (x : ℝ) (h : 16 * x^2 = 81) : x = 9 / 4 ∨ x = - (9 / 4) :=
by
  sorry

end solve_equation_l343_343036


namespace inequality_proof_l343_343818

noncomputable def a : ℝ := 1.1 ^ 1.2
noncomputable def b : ℝ := 1.2 ^ 1.3
noncomputable def c : ℝ := 1.3 ^ 1.1

theorem inequality_proof : a < b ∧ b < c := 
by
  sorry

end inequality_proof_l343_343818


namespace circumscribed_circle_radius_isosceles_trapezoid_l343_343149

theorem circumscribed_circle_radius_isosceles_trapezoid :
  ∀ (a b h : ℕ), 
    a = 21 → 
    b = 9 → 
    h = 8 → 
    ∃ R : ℚ, R = 85 / 8 :=
by
  intros a b h ha hb hh
  use 85 / 8
  sorry

end circumscribed_circle_radius_isosceles_trapezoid_l343_343149


namespace horizontal_length_of_32inch_widescreen_l343_343568

noncomputable def diagonal_length : ℝ := 32
noncomputable def aspect_ratio_width : ℕ := 16
noncomputable def aspect_ratio_height : ℕ := 9

def aspect_ratio_diagonal_square : ℕ := aspect_ratio_width^2 + aspect_ratio_height^2

noncomputable def actual_ratio_diagonal : ℝ := real.sqrt aspect_ratio_diagonal_square

noncomputable def scaling_factor : ℝ := diagonal_length / actual_ratio_diagonal

noncomputable def horizontal_length : ℝ := aspect_ratio_width * scaling_factor

theorem horizontal_length_of_32inch_widescreen :
  horizontal_length ≈ 27.89 := 
begin
  sorry
end

end horizontal_length_of_32inch_widescreen_l343_343568


namespace factor_expression_l343_343752

-- Define the expressions E1 and E2
def E1 (y : ℝ) : ℝ := 12 * y^6 + 35 * y^4 - 5
def E2 (y : ℝ) : ℝ := 2 * y^6 - 4 * y^4 + 5

-- Define the target expression E
def E (y : ℝ) : ℝ := E1 y - E2 y

-- The main theorem to prove
theorem factor_expression (y : ℝ) : E y = 10 * (y^6 + 3.9 * y^4 - 1) := by
  sorry

end factor_expression_l343_343752


namespace combined_stickers_l343_343195

def initial_stickers_june : ℕ := 76
def initial_stickers_bonnie : ℕ := 63
def birthday_stickers : ℕ := 25

theorem combined_stickers : 
  (initial_stickers_june + birthday_stickers) + (initial_stickers_bonnie + birthday_stickers) = 189 := 
by
  sorry

end combined_stickers_l343_343195


namespace distance_from_origin_to_line_l343_343267

theorem distance_from_origin_to_line : 
  ∀ (x y : ℝ), (4 * x + 3 * y - 12 = 0) → (dist (0 : ℝ × ℝ) (x, y) = 12/5) :=
begin
  intros x y h,
  sorry
end

end distance_from_origin_to_line_l343_343267


namespace estimated_value_at_28_l343_343443

-- Definitions based on the conditions
def regression_equation (x : ℝ) : ℝ := 4.75 * x + 257

-- Problem statement
theorem estimated_value_at_28 : regression_equation 28 = 390 :=
by
  -- Sorry is used to skip the proof
  sorry

end estimated_value_at_28_l343_343443


namespace henry_total_l343_343873

def round_to_nearest_dollar (x : ℝ) : ℕ :=
  if x - x.floor < 0.5 then x.floor.to_nat else x.ceil.to_nat

theorem henry_total 
  (h₁ : 2.49)
  (h₂ : 3.75)
  (h₃ : 8.66) : 
  round_to_nearest_dollar (h₁) + round_to_nearest_dollar (h₂) + round_to_nearest_dollar (h₃) = 15 :=
by
  sorry

end henry_total_l343_343873


namespace find_multiple_of_A_l343_343588

def shares_division_problem (A B C : ℝ) (x : ℝ) : Prop :=
  C = 160 ∧
  x * A = 5 * B ∧
  x * A = 10 * C ∧
  A + B + C = 880

theorem find_multiple_of_A (A B C x : ℝ) (h : shares_division_problem A B C x) : x = 4 :=
by sorry

end find_multiple_of_A_l343_343588


namespace correct_operation_l343_343324

theorem correct_operation :
  (3 * x) ^ 2 = 9 * x ^ 2 ∧ 
  ¬(x ^ 2 * x ^ 3 = x ^ 6) ∧ 
  ¬(x ^ 6 / x ^ 3 = x ^ 2) ∧ 
  ¬((x * y ^ 2) ^ 3 = x * y ^ 6) :=
by
  split
  · sorry
  split
  · sorry
  split
  · sorry
  · sorry

end correct_operation_l343_343324


namespace find_a_l343_343804

theorem find_a (a : ℝ) (x : ℝ) (h₀ : a > 0) (h₁ : x > 0)
  (h₂ : a * Real.sqrt x = Real.log (Real.sqrt x))
  (h₃ : (a / (2 * Real.sqrt x)) = (1 / (2 * x))) : a = Real.exp (-1) :=
by
  sorry

end find_a_l343_343804


namespace distances_form_triangle_l343_343170

theorem distances_form_triangle 
  (A B C T : Type*)
  [metric_space A] [metric_space B] [metric_space C] [metric_space T]
  (h_eq_triangle : ∀ {x y z : T}, dist x y = dist y z ∧ dist y z = dist z x ∧ dist z x = dist x y)
  (hT_inside : ∀ {P : T}, dist T P < dist A B) :
  (dist T A < dist T B + dist T C) ∧ (dist T B < dist T A + dist T C) ∧ (dist T C < dist T A + dist T B) :=
  sorry

end distances_form_triangle_l343_343170


namespace number_of_solutions_eq_43_l343_343505

theorem number_of_solutions_eq_43 :
  let S := {x | x ∈ (Finset.range 51)} -- S = {1, 2, 3, ..., 50}
  let T := {x | x ∈ {1, 4, 9, 16, 25, 36, 49}} -- T = {1, 4, 9, 16, 25, 36, 49}
  |S| - |T| = 43 :=
by
  sorry -- No proof required, placeholder

end number_of_solutions_eq_43_l343_343505


namespace largest_5_power_dividing_product_l343_343211

theorem largest_5_power_dividing_product :
  let P := ∏ i in Finset.range 100, (2 * i + 1)
  ∃ k : ℕ, 5^k ∣ P ∧ ∀ m > k, ¬ 5^m ∣ P := 
by
  let P := ∏ i in Finset.range 100, (2 * i + 1)
  use 25
  sorry

end largest_5_power_dividing_product_l343_343211


namespace min_value_fraction_condition_l343_343448

noncomputable def minValue (a b : ℝ) := 1 / (2 * a) + a / (b + 1)

theorem min_value_fraction_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  minValue a b = 5 / 4 :=
by
  sorry

end min_value_fraction_condition_l343_343448


namespace f_f_neg_three_eq_zero_f_min_value_l343_343088

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x + (2 / x) - 3 else real.log (x^2 + 1)

theorem f_f_neg_three_eq_zero : f (f (-3)) = 0 :=
sorry

theorem f_min_value : ∀ x : ℝ, f x ≥ 2 * real.sqrt 2 - 3 :=
sorry

end f_f_neg_three_eq_zero_f_min_value_l343_343088


namespace cost_per_liter_fuel_is_correct_l343_343138

-- Definitions of conditions from part a)
def service_cost_per_vehicle : Float := 2.30
def num_minivans : Nat := 4
def num_trucks : Nat := 2
def total_cost : Float := 396
def minivan_tank_capacity : Float := 65.0
def truck_tank_increase_rate : Float := 1.2

-- Formulating the problem statement as a claim in Lean
theorem cost_per_liter_fuel_is_correct :
  let total_service_cost := ↑(num_minivans + num_trucks) * service_cost_per_vehicle,
      truck_tank_capacity := minivan_tank_capacity * (1 + truck_tank_increase_rate),
      total_fuel_capacity := num_minivans * minivan_tank_capacity + num_trucks * truck_tank_capacity,
      total_fuel_cost := total_cost - total_service_cost,
      cost_per_liter := total_fuel_cost / total_fuel_capacity
  in cost_per_liter = 0.70 :=
by
  sorry

end cost_per_liter_fuel_is_correct_l343_343138


namespace average_milk_per_girl_l343_343276

-- Define the conditions
def total_students : ℕ := 60
def percent_girls : ℕ := 40
def num_girls := total_students * percent_girls / 100
def num_boys := total_students - num_girls
def milk_per_boy : ℕ := 1
def total_milk : ℕ := 168

-- Define the question and the proof statement
theorem average_milk_per_girl :
  (total_milk - (num_boys * milk_per_boy)) / num_girls = 5.5 := by
  sorry

end average_milk_per_girl_l343_343276


namespace exists_lambda_l343_343439

def cubic_polynomial (x : ℝ) : Type := {P : ℝ → ℝ // ∃ (a b c d : ℝ), ∀ x, P x = a*x^3 + b*x^2 + c*x + d}

theorem exists_lambda (P Q R : ℝ → ℝ) (hP : cubic_polynomial P)
  (hQ : cubic_polynomial Q) (hR : cubic_polynomial R)
  (hPQ : ∀ x, P x ≤ Q x)
  (hQR : ∀ x, Q x ≤ R x)
  (hPuRu : ∃ u, P u = R u) :
  ∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ ∀ x, Q x = λ * P x + (1 - λ) * R x := by
sorry

end exists_lambda_l343_343439


namespace correct_operation_l343_343323

theorem correct_operation (a b : ℝ) : 
  (3 * Real.sqrt 7 + 7 * Real.sqrt 3 ≠ 10 * Real.sqrt 10) ∧ 
  (Real.sqrt (2 * a) * Real.sqrt (3) * a = Real.sqrt (6) * a) ∧ 
  (Real.sqrt a - Real.sqrt b ≠ Real.sqrt (a - b)) ∧ 
  (Real.sqrt (20 / 45) ≠ 4 / 9) :=
by
  sorry

end correct_operation_l343_343323


namespace min_abs_w_minus_i_of_complex_eq_l343_343950

noncomputable def w_min (w : ℂ) : ℝ :=
  abs (w - complex.I)

theorem min_abs_w_minus_i_of_complex_eq {w : ℂ}
  (h : abs (w^2 - 4) = abs (w * (w - 2 * complex.I))) :
  ∃ (w : ℂ), abs (w - complex.I) = 1 / real.sqrt 2 :=
by
  sorry

end min_abs_w_minus_i_of_complex_eq_l343_343950


namespace daughter_work_alone_12_days_l343_343353

/-- Given a man, his wife, and their daughter working together on a piece of work. The man can complete the work in 4 days, the wife in 6 days, and together with their daughter, they can complete it in 2 days. Prove that the daughter alone would take 12 days to complete the work. -/
theorem daughter_work_alone_12_days (h1 : (1/4 : ℝ) + (1/6) + D = 1/2) : D = 1/12 :=
by
  sorry

end daughter_work_alone_12_days_l343_343353


namespace count_n_satisfying_condition_l343_343032

theorem count_n_satisfying_condition : 
  (Finset.filter (λ n : ℤ, 25 < n^2 ∧ n^2 < 144) (Finset.range 145)).card = 12 :=
by
  sorry

end count_n_satisfying_condition_l343_343032


namespace find_a_increasing_intervals_geq_zero_set_l343_343083

noncomputable def f (x a : ℝ) : ℝ :=
  sin (x + π / 6) + sin (x - π / 6) + cos x + a

-- Given that the maximum value of f(x) is 1, determine the value of a
theorem find_a (h : ∀ x : ℝ, f x a ≤ 1) : a = -1 :=
sorry

-- Determine the intervals where f(x) is monotonically increasing
theorem increasing_intervals (a : ℝ) (h : a = -1) :
  ∃ k : ℤ, ∀ x ∈ set.Icc ((2 * k : ℝ) * π - 2 * π / 3) ((2 * k : ℝ) * π + π / 3),
    monotoneOn (λ x, f x a) (set.Icc ((2 * k : ℝ) * π - 2 * π / 3) ((2 * k : ℝ) * π + π / 3)) :=
sorry

-- Find the set of values of x for which f(x) ≥ 0
theorem geq_zero_set (a : ℝ) (h : a = -1) :
  ∀ k : ℤ, ∀ x ∈ set.Icc ((2 * k : ℝ) * π - π / 6) ((2 * k : ℝ) * π + π / 2), 0 ≤ f x a :=
sorry

end find_a_increasing_intervals_geq_zero_set_l343_343083


namespace base_11_arithmetic_l343_343602

-- Define the base and the numbers in base 11
def base := 11

def a := 6 * base^2 + 7 * base + 4  -- 674 in base 11
def b := 2 * base^2 + 7 * base + 9  -- 279 in base 11
def c := 1 * base^2 + 4 * base + 3  -- 143 in base 11
def result := 5 * base^2 + 5 * base + 9  -- 559 in base 11

theorem base_11_arithmetic :
  (a - b + c) = result :=
sorry

end base_11_arithmetic_l343_343602


namespace score_87_not_possible_l343_343972

def max_score := 15 * 6
def score (correct unanswered incorrect : ℕ) := 6 * correct + unanswered

theorem score_87_not_possible :
  ¬∃ (correct unanswered incorrect : ℕ), 
    correct + unanswered + incorrect = 15 ∧
    6 * correct + unanswered = 87 := 
sorry

end score_87_not_possible_l343_343972


namespace john_vowel_learning_days_l343_343039

theorem john_vowel_learning_days :
  let vowels := 5
  let days_per_vowel := 3
  vowels * days_per_vowel = 15 :=
by
  let vowels := 5
  let days_per_vowel := 3
  show vowels * days_per_vowel = 15 from
  sorry

end john_vowel_learning_days_l343_343039


namespace proof_problem_l343_343855

noncomputable def A : ℝ × ℝ := (-3, -4)
noncomputable def B : ℝ × ℝ := (5, -12)

-- Define vector AB
def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Calculate the magnitude of vector AB
def magnitude_vector_AB : ℝ := Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2)

-- Define vectors OA and OB
def vector_OA : ℝ × ℝ := A
def vector_OB : ℝ × ℝ := B

-- Calculate the dot product of vectors OA and OB
def dot_product : ℝ := vector_OA.1 * vector_OB.1 + vector_OA.2 * vector_OB.2

theorem proof_problem :
  vector_AB = (8, -8) ∧
  magnitude_vector_AB = 8 * Real.sqrt 2 ∧
  dot_product = 33 :=
by
  sorry

end proof_problem_l343_343855


namespace sequence_a_2016_l343_343822

theorem sequence_a_2016 : 
  ∀ (a : ℕ → ℝ) (T : ℕ → ℝ), 
  (∀ n : ℕ, 0 < n → T n = ∏ i in finset.range n, a (i + 1)) →
  (∀ n : ℕ, 0 < n → T n = 2 - 2 * a n) →
  a 2016 = 2017 / 2018 :=
by
  -- Proof details would go here
  sorry

end sequence_a_2016_l343_343822


namespace quadratic_inequality_ab_l343_343442

theorem quadratic_inequality_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + 1 > 0) ↔ -1 < x ∧ x < 1 / 3) :
  a * b = -6 :=
by
  -- Proof is omitted
  sorry

end quadratic_inequality_ab_l343_343442


namespace length_of_ribbon_per_gift_l343_343115

theorem length_of_ribbon_per_gift (total_length : ℝ) (number_of_presents : ℕ) : 
    total_length = 50.68 → number_of_presents = 7 → total_length / number_of_presents = 7.24 :=
begin
  intros h1 h2,
  rw h1,
  rw h2,
  norm_num,
end

end length_of_ribbon_per_gift_l343_343115


namespace sandbox_area_combined_approx_l343_343933

noncomputable def solve_sandbox_area : Real :=
  let w1 := 5 -- width of the first sandbox
  let l1 := 2 * w1 -- length of the first sandbox
  let w2 := Real.sqrt(22.5) -- width of the second sandbox
  let l2 := 3 * w2 -- length of the second sandbox
  let area1 := l1 * w1 -- area of the first sandbox
  let area2 := l2 * w2 -- area of the second sandbox
  area1 + area2 -- combined area

theorem sandbox_area_combined_approx :
  solve_sandbox_area ≈ 117.42 :=
by
  sorry

end sandbox_area_combined_approx_l343_343933


namespace chessboard_decomposition_max_rectangles_l343_343003

theorem chessboard_decomposition_max_rectangles :
  ∃ (p : ℕ) (a : ℕ → ℕ),
    (p = 7 ∧ (∀ i j, i < j → a i < a j) ∧ sum (finset.range p) a = 32 ∧
      (a = (fun i, [1, 2, 3, 4, 5, 8, 9].nth i.get_or_else 0) ∨
       a = (fun i, [1, 2, 3, 4, 5, 7, 10].nth i.get_or_else 0) ∨
       a = (fun i, [1, 2, 3, 4, 6, 7, 9].nth i.get_or_else 0) ∨
       a = (fun i, [1, 2, 3, 5, 6, 7, 8].nth i.get_or_else 0))) :=
sorry

end chessboard_decomposition_max_rectangles_l343_343003


namespace number_of_lucky_numbers_l343_343143

-- Defining the concept of sequence with even number of digit 8
def is_lucky (seq : List ℕ) : Prop :=
  seq.count 8 % 2 = 0

-- Define S(n) recursive formula
noncomputable def S : ℕ → ℝ
| 0 => 0
| n+1 => 4 * (1 - (1 / (2 ^ (n+1))))

theorem number_of_lucky_numbers (n : ℕ) :
  ∀ (seq : List ℕ), (seq.length ≤ n) → is_lucky seq → S n = 4 * (1 - 1 / (2 ^ n)) :=
sorry

end number_of_lucky_numbers_l343_343143


namespace molecular_weight_of_compound_l343_343658

def atomic_weights : ℕ → ℝ
| 0 := 12.01  -- Atomic weight of Carbon (C)
| 1 := 1.008  -- Atomic weight of Hydrogen (H)
| 2 := 16.00  -- Atomic weight of Oxygen (O)
| _ := 0

def molecular_weight (num_C num_H num_O : ℕ) : ℝ :=
  (num_C * atomic_weights 0) + (num_H * atomic_weights 1) + (num_O * atomic_weights 2)

theorem molecular_weight_of_compound :
    molecular_weight 4 8 2 = 88.104 :=
by
  sorry

end molecular_weight_of_compound_l343_343658


namespace combined_stickers_l343_343194

def initial_stickers_june : ℕ := 76
def initial_stickers_bonnie : ℕ := 63
def birthday_stickers : ℕ := 25

theorem combined_stickers : 
  (initial_stickers_june + birthday_stickers) + (initial_stickers_bonnie + birthday_stickers) = 189 := 
by
  sorry

end combined_stickers_l343_343194


namespace ball_14_probability_in_final_bags_l343_343812

/-- Define the conditions of the problem. -/
def initial_bags : List (List ℕ) := [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [], [], []]

/-- Define a function that simulates the next state of the bags after picking two balls. -/
def next_state (bags : List (List ℕ)) : List (List ℕ) :=
  -- Implementation for the state transition would go here.

noncomputable def final_bags (bags : List (List ℕ)) : List (List ℕ) := sorry

/-- The main theorem stating the probability that ball 14 is in one of the final bags. -/
theorem ball_14_probability_in_final_bags :
  let bag1 := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
      bag2 := [];
      bag3 := [];
      bag4 := [];
      -- List representation of bags
      bags := [bag1, bag2, bag3, bag4];
  final_bags bags ∋ 14 → 
    probability_of_ending_up_in_one_of_final_bags (14) = 2 / 3 := sorry

end ball_14_probability_in_final_bags_l343_343812


namespace simplify_expression_l343_343594

theorem simplify_expression :
  (√300 / √75 - √147 / √63) = (42 - 7 * √21) / 21 := by
  sorry

end simplify_expression_l343_343594


namespace transport_in_neg_20_repr_transport_out_20_l343_343145

theorem transport_in_neg_20_repr_transport_out_20
  (out_recording : ∀ x : ℝ, transporting_out x → recording (-x))
  (in_recording  : ∀ x : ℝ, transporting_in x → recording x) :
  recording (-(-20)) = recording (20) := by
  sorry

end transport_in_neg_20_repr_transport_out_20_l343_343145


namespace asymptotes_of_hyperbola_l343_343864

theorem asymptotes_of_hyperbola (a : ℝ) (h : a > 0) :
  ∀ x y : ℝ, (∃ e : ℝ, e = sqrt 3 ∧ ∃ b : ℝ, (x^2 / a^2) - (y^2 / 2) = 1 ∧ y = sqrt 2 * x ∨ y = - sqrt 2 * x) := 
by
  sorry

end asymptotes_of_hyperbola_l343_343864


namespace ratio_of_angles_l343_343160

-- Definitions of angles and relationships
noncomputable def Angle (A B C : Type*) := sorry

variables (A B C P Q N : Type*)

-- Given conditions
def angles_equal_parts (x : ℝ) :=
  Angle A B P = x ∧ Angle P B Q = x ∧ Angle Q B C = x

def bisects (x : ℝ) :=
  Angle Q B N = x/2 ∧ Angle N B P = x/2

theorem ratio_of_angles (x : ℝ) (A B C P Q N : Type*) :
  angles_equal_parts x →
  bisects x →
  (Angle N B Q) / (Angle A B Q) = 3 / 4 :=
by
  intros h1 h2
  sorry

end ratio_of_angles_l343_343160


namespace BC_eq_DE_l343_343570

-- Define the basic configuration and points
variables (O A B C D E : Point)
variables (circle : Circle)
variables (angle : Angle)

-- Hypotheses stating the conditions of the problem
hypotheses
  (h1 : circle.tangent O)
  (h2 : circle.diametrically_opposite A B)
  (h3 : tangent.circle_at B.intersects (angle.side1) C)
  (h4 : tangent.circle_at B.intersects (angle.side2) D)
  (h5 : tangent.circle_at B.intersects_line OA E)

theorem BC_eq_DE 
  (h1 : circle.tangent O)
  (h2 : circle.diametrically_opposite A B)
  (h3 : tangent.circle_at B.intersects (angle.side1) C)
  (h4 : tangent.circle_at B.intersects (angle.side2) D)
  (h5 : tangent.circle_at B.intersects_line OA E) :
  distance B C = distance D E :=
sorry

end BC_eq_DE_l343_343570


namespace count_isolated_subsets_l343_343552

def is_isolated (A : Set ℤ) (k : ℤ) : Prop :=
  k ∈ A ∧ k - 1 ∉ A ∧ k + 1 ∉ A

def isolated_subsets_card (S : Set ℤ) (n : ℕ) : ℕ :=
  let subsets := { A : Set ℤ | A ⊆ S ∧ Finset.card A = n ∧ ∀ k ∈ A, is_isolated A k }
  Finset.card subsets

theorem count_isolated_subsets :
  isolated_subsets_card (Set.ofFinite {1, 2, 3, 4, 5, 6, 7, 8}) 3 = 20 :=
by
  sorry

end count_isolated_subsets_l343_343552


namespace determine_base_y_l343_343925

noncomputable def is_equilateral (a b c : ℕ) : Prop :=
  a = b ∧ b = c

noncomputable def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def potential_base_y (x y : ℕ) : Prop :=
  2 * x + y = 30

noncomputable def shared_height (x y : ℕ) : Prop :=
  2 * sqrt (4 * x^2 - y^2) = 10 * sqrt 3

noncomputable def shared_area (x y : ℕ) : Prop :=
  y * sqrt (4 * x^2 - y^2) = 100 * sqrt 3

theorem determine_base_y (x y : ℕ) (hx : is_equilateral 10 10 10)
  (hp : perimeter 10 10 10 = 30)
  (hq : potential_base_y x y)
  (hr : shared_height x y)
  (hs : shared_area x y) :
  y = 20 := sorry

end determine_base_y_l343_343925


namespace tank_filled_by_10pm_l343_343177

-- Defining the problem setup
def starts_raining (t : Real) : Prop := t ≥ 13  -- 13 represents 1 pm
def rainfall_rate (t : Real) : Real :=
  if t < 14 then 2       -- 2 inches from 1 pm to 2 pm
  else if t < 18 then 1  -- 1 inch per hour from 2 pm to 6 pm
  else 3                 -- 3 inches per hour from 6 pm onwards
def tank_height : Real := 18 -- Height of the fish tank is 18 inches

-- Calculating the total rainfall by integrating the rainfall rate
noncomputable def total_rainfall (t : Real) : Real :=
  ∫ τ in (13:ℝ)..t, rainfall_rate τ

-- The statement of the proof problem
theorem tank_filled_by_10pm : total_rainfall 22 = tank_height :=
by
  sorry

end tank_filled_by_10pm_l343_343177


namespace p_neither_sufficient_nor_necessary_for_q_l343_343435

variables {d : ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ}

-- Define the arithmetic sequence terms and the sum of the first n terms
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_sequence (S a : ℕ → ℝ) : Prop :=
∀ n, S (n + 1) = S n + a (n + 1)

-- Define the conditions
def condition_p (d : ℝ) : Prop := d < 0
def condition_q (S : ℕ → ℝ) : Prop := ∀ n, S (n + 1) < S n

-- Define the theorem we intend to prove
theorem p_neither_sufficient_nor_necessary_for_q (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  sum_sequence S a →
  (¬((condition_p d → condition_q S) ∧ (condition_q S → condition_p d))) :=
begin
  sorry -- the proof is not required
end

end p_neither_sufficient_nor_necessary_for_q_l343_343435


namespace sum_of_two_integers_l343_343284

theorem sum_of_two_integers (a b : ℕ) (h1 : a * b + a + b = 113) (h2 : Nat.gcd a b = 1) (h3 : a < 25) (h4 : b < 25) : a + b = 23 := by
  sorry

end sum_of_two_integers_l343_343284


namespace closest_multiple_of_21_to_2023_l343_343659

theorem closest_multiple_of_21_to_2023 : ∃ k : ℤ, k * 21 = 2022 ∧ ∀ m : ℤ, m * 21 = 2023 → (abs (m - 2023)) > (abs (2022 - 2023)) :=
by
  sorry

end closest_multiple_of_21_to_2023_l343_343659


namespace base4_addition_correct_l343_343746

theorem base4_addition_correct : 
  let a := 213
  let b := 132
  let c := 321
  let sum_base10 := (2 * 4^2 + 1 * 4^1 + 3 * 4^0) + (1 * 4^2 + 3 * 4^1 + 2 * 4^0) + (3 * 4^2 + 2 * 4^1 + 1 * 4^0)
  let result_base4 := 1332
  in sum_base10 = 1 * 4^3 + 3 * 4^2 + 3 * 4^1 + 2 * 4^0 
:= by
  sorry

end base4_addition_correct_l343_343746


namespace largest_common_term_l343_343382

/-- 
An arithmetic sequence with first term 3 has a common difference of 8.
A second sequence begins with 5 and has a common difference of 9.
In the range between 1 and 150, the largest number common to both sequences is 131.
-/
theorem largest_common_term (a_n b_n : ℕ → ℕ) (n : ℕ) :
  ∀ (n1 n2 : ℕ), (a_n n1 = 3 + 8 * (n1 - 1)) → (b_n n2 = 5 + 9 * (n2 - 1)) → 
  1 ≤ 3 + 8 * (n1 - 1) ∧ 3 + 8 * (n1 - 1) ≤ 150 → 
  1 ≤ 5 + 9 * (n2 - 1) ∧ 5 + 9 * (n2 - 1) ≤ 150 → 
  (∃ m : ℕ, 3 + 8 * (n1 - 1) = 59 + 72 * m ∨ 5 + 9 * (n2 - 1) = 59 + 72 * m) → 
  ∃ k, 3 + 8 * (n1 - 1) = 5 + 9 * (n2 - 1) ∧ 
  ∃ y, 1 ≤ y ∧ y ≤ 150 ∧ 3 + 8 * (n1 - 1) = y ∧ 
  y = 131 := 
by
  sorry

end largest_common_term_l343_343382


namespace circumcircles_pass_through_fixed_point_locus_of_E_on_line_segment_KQ_l343_343201

open EuclideanGeometry

-- Let's define the given conditions
variables (C1 : Circle) (P : Point) (A B C D E : Point)

-- Conditions
def quadrilateral_on_circle : Prop :=
Quadrilateral.is_on_circle C1 A B C D

def rays_intersect_at_P : Prop :=
∃ X, Ray.through X A B ∧ Ray.through X C D ∧ X = P

def intersection_of_AC_BD : Prop :=
∃ X, Line.intersection (Line.through A C) (Line.through B D) = E ∧ X = E

-- Part (a) Proof that circumcircles pass through a fixed point
theorem circumcircles_pass_through_fixed_point :
  quadrilateral_on_circle C1 A B C D ∧
  rays_intersect_at_P P A B C D ∧
  intersection_of_AC_BD A C B D E →
  ∃ H : Point, 
    ∀ (circle_ADE circle_BEC : Circle),
      circle_ADE = Circle.of_triangles A D E H ∧
      circle_BEC = Circle.of_triangles B E C H :=
sorry

-- Part (b) Finding the locus of point E
theorem locus_of_E_on_line_segment_KQ :
  quadrilateral_on_circle C1 A B C D ∧
  rays_intersect_at_P P A B C D ∧
  intersection_of_AC_BD A C B D E →
  ∀ K Q : Point, LineSeg K Q ∋ E :=
sorry

end circumcircles_pass_through_fixed_point_locus_of_E_on_line_segment_KQ_l343_343201


namespace sequence_count_19_l343_343879

def f : ℕ → ℕ
| 3 := 1
| 4 := 1
| 5 := 1
| 6 := 2
| 7 := 2
| n := f (n - 4) + 2 * f (n - 5) + f (n - 6)

theorem sequence_count_19 : f 19 = 65 := sorry

end sequence_count_19_l343_343879


namespace repeating_decimal_sum_to_fraction_l343_343788

-- Define the repeating decimals as individual variables
def x := 0.𝕎3
def y := 0.𝕎005
def z := 0.𝕎0007

-- Define the Lean statement
theorem repeating_decimal_sum_to_fraction :
  x + y + z = 10170 / 29997 :=
sorry

end repeating_decimal_sum_to_fraction_l343_343788


namespace common_points_form_equilateral_triangle_l343_343000

noncomputable def is_equilateral_triangle (A B C : (ℝ × ℝ)) : Prop :=
  let dist := λ (P Q : (ℝ × ℝ)), (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 in
  dist A B = dist B C ∧ dist B C = dist C A

theorem common_points_form_equilateral_triangle :
  ∃ A B C : (ℝ × ℝ), 
    (A = (0, 2) ∨ A = (sqrt 3 / 2, 1 / 2) ∨ A = (- sqrt 3 / 2, 1 / 2)) ∧
    (B = (0, 2) ∨ B = (sqrt 3 / 2, 1 / 2) ∨ B = (- sqrt 3 / 2, 1 / 2)) ∧
    (C = (0, 2) ∨ C = (sqrt 3 / 2, 1 / 2) ∨ C = (- sqrt 3 / 2, 1 / 2)) ∧
    is_equilateral_triangle A B C :=
begin
  sorry
end

end common_points_form_equilateral_triangle_l343_343000


namespace john_memory_card_cost_l343_343931

-- Define conditions
def pictures_per_day : ℕ := 10
def days_per_year : ℕ := 365
def years : ℕ := 3
def pictures_per_card : ℕ := 50
def cost_per_card : ℕ := 60

-- Define total days
def total_days (years : ℕ) (days_per_year : ℕ) : ℕ := years * days_per_year

-- Define total pictures
def total_pictures (pictures_per_day : ℕ) (total_days : ℕ) : ℕ := pictures_per_day * total_days

-- Define required cards
def required_cards (total_pictures : ℕ) (pictures_per_card : ℕ) : ℕ :=
  (total_pictures + pictures_per_card - 1) / pictures_per_card  -- ceiling division

-- Define total cost
def total_cost (required_cards : ℕ) (cost_per_card : ℕ) : ℕ := required_cards * cost_per_card

-- Prove the total cost equals $13,140
theorem john_memory_card_cost : total_cost (required_cards (total_pictures pictures_per_day (total_days years days_per_year)) pictures_per_card) cost_per_card = 13140 :=
by
  sorry

end john_memory_card_cost_l343_343931


namespace arrangeable_circle_uniq_l343_343307

theorem arrangeable_circle_uniq (n : ℕ → ℤ) (h_length: ∀ᵢ < 2017, ∃ p q r s t : ℤ, 
  (p = n i) ∧ (q = n (i+1) % 2017) ∧ (r = n (i+2) % 2017) ∧ 
  (s = n (i+3) % 2017) ∧ (t = n (i+4) % 2017) ∧ (p - q + r - s + t = 29)) : 
  (∀ j < 2017, n j = 29) :=
sorry

end arrangeable_circle_uniq_l343_343307


namespace solution_exists_l343_343306

def number_of_books (x : ℕ) : Prop :=
  let math_books := 3 * x in
  let history_books := 2 * x in
  let science_books := 4 * x in
  let literature_books := x in
  (math_books + history_books + science_books + literature_books = 80) ∧
  (4 * math_books + 5 * history_books + 6 * science_books + 7 * literature_books = 520)

theorem solution_exists : ∃ x, number_of_books x :=
sorry

end solution_exists_l343_343306


namespace suraj_average_l343_343998

theorem suraj_average : 
  ∀ (A : ℝ), 
    (16 * A + 92 = 17 * (A + 4)) → 
      (A + 4) = 28 :=
by
  sorry

end suraj_average_l343_343998


namespace students_count_l343_343560

noncomputable def number_of_students (n : ℕ) (h : 2 ≤ n) : ℕ :=
  let num_students := n^2 + n + 1
  in num_students

theorem students_count (n : ℕ) (h : 2 ≤ n) (S : finset α) (P : finset β) 
  (num_students: S.card > n + 2)
  (solved_by: ∀ s1 s2 ∈ S, {p | p ∈ P ∧ s1 ∈ p ∧ s2 ∈ p}.card = 1)
  (solves: ∀ p1 p2 ∈ P, {s | s ∈ S ∧ p1 ∈ s ∧ p2 ∈ s}.card = 1) 
  (L: α ) (specific_problem: β ) (solved_L_P: {s | s ∈ S ∧ specific_problem ∈ s}.card = n + 1) :
  S.card = number_of_students n h := 
sorry

end students_count_l343_343560


namespace shortest_hypotenuse_max_inscribed_circle_radius_l343_343300

variable {a b c r : ℝ}

-- Condition 1: The perimeter of the right-angled triangle is 1 meter.
def perimeter_condition (a b : ℝ) : Prop :=
  a + b + Real.sqrt (a^2 + b^2) = 1

-- Problem 1: Prove the shortest length of the hypotenuse is √2 - 1.
theorem shortest_hypotenuse (a b : ℝ) (h : perimeter_condition a b) :
  Real.sqrt (a^2 + b^2) = Real.sqrt 2 - 1 :=
sorry

-- Problem 2: Prove the maximum value of the inscribed circle radius is 3/2 - √2.
theorem max_inscribed_circle_radius (a b r : ℝ) (h : perimeter_condition a b) :
  (a * b = r) → r = 3/2 - Real.sqrt 2 :=
sorry

end shortest_hypotenuse_max_inscribed_circle_radius_l343_343300


namespace fourth_number_in_15th_row_of_pascals_triangle_l343_343538

-- Here we state and prove the theorem about the fourth entry in the 15th row of Pascal's Triangle.
theorem fourth_number_in_15th_row_of_pascals_triangle : 
    (nat.choose 15 3) = 455 := 
by 
    sorry -- Proof is omitted as per instructions

end fourth_number_in_15th_row_of_pascals_triangle_l343_343538


namespace slope_divides_area_in_half_l343_343532

structure Point :=
(x : ℝ)
(y : ℝ)

def vertices : List Point := [
  {x := 0, y := 0},
  {x := 0, y := 4},
  {x := 4, y := 4},
  {x := 4, y := 2},
  {x := 7, y := 2},
  {x := 7, y := 0}
]

def area_triangle (a b c : Point) : ℝ :=
  (1 / 2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

def area_rectangle (a b c d : Point) :=
  abs ((a.x - c.x) * (a.y - c.y))

noncomputable def divide_area_slope : ℝ :=
  3 / 8

theorem slope_divides_area_in_half :
  (∃ m : ℝ, m = divide_area_slope
            ∧ let line := fun p : Point => p.y = m * p.x in
              let half_area := (area_rectangle {x:=0, y:=0} {x:=0, y:=4} {x:=4, y:=4} {x:=4, y:=0}
                              + area_rectangle {x:=4, y:=4} {x:=4, y:=2} {x:=7, y:=2} {x:=7, y:=0}) / 2 in
              ∀ p₁ p₂ : Point, 
                -- Check the area above and below the line are equal to half_area
               (area_triangle vertices[0] vertices[1] {x := 4, y := 1.5})
              = half_area / 2)
            :=
by 
  sorry

end slope_divides_area_in_half_l343_343532


namespace total_number_of_crayons_l343_343132

def number_of_blue_crayons := 3
def number_of_red_crayons := 4 * number_of_blue_crayons
def number_of_green_crayons := 2 * number_of_red_crayons
def number_of_yellow_crayons := number_of_green_crayons / 2

theorem total_number_of_crayons :
  number_of_blue_crayons + number_of_red_crayons + number_of_green_crayons + number_of_yellow_crayons = 51 :=
by 
  -- Proof is not required
  sorry

end total_number_of_crayons_l343_343132


namespace length_of_EF_l343_343151

theorem length_of_EF (WZ XY : ℝ) (A : ℝ) (hWZ : WZ = 8) (hXY : XY = 10)
(hArea : A = (WZ * XY) / 3) :
  ∃ EF : ℝ, EF = (8 * (√15)) / 3 :=
by sorry

end length_of_EF_l343_343151


namespace triangle_area_ABC_l343_343019

-- Given conditions
variables (A B C : Type) [euclidean_geometry A B C] 
variables (hypotenuse_AB : AB = 14)
variables (angle_B : ∠ B = 90 °)
variables (angle_C : ∠ C = 45 °)

-- Definition of the problem statement
theorem triangle_area_ABC :
  area (triangle ABC) = 49 :=
sorry

end triangle_area_ABC_l343_343019


namespace lines_skew_l343_343790

def line1 (b : ℝ) (t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 3 + 2 * t, b + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 4 + 3 * u, 1 + 2 * u)

theorem lines_skew (b : ℝ) : 
  ¬ ∃ t u : ℝ, line1 b t = line2 u ↔ b ≠ 4 := 
sorry

end lines_skew_l343_343790


namespace perpendicular_lines_to_parallel_planes_l343_343843

-- Define non-overlapping lines and planes in a 3D geometry space
variables {m n : line} {α β : plane}

-- Conditions:
-- m is a line
-- α and β are planes
-- m is perpendicular to α
-- m is perpendicular to β

-- To prove:
-- α is parallel to β

theorem perpendicular_lines_to_parallel_planes 
  (non_overlap_mn : m ≠ n) 
  (non_overlap_ab : α ≠ β) 
  (m_perp_α : m ⊥ α) 
  (m_perp_β : m ⊥ β) : parallel α β :=
sorry

end perpendicular_lines_to_parallel_planes_l343_343843


namespace inequality_1_over_1_plus_a_squared_equality_holds_for_l343_343559

theorem inequality_1_over_1_plus_a_squared 
(a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b) :=
begin
  sorry
end

theorem equality_holds_for (a b : ℝ) :
  0 < a = b ∧ a < 1 ↔ (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b)) :=
begin
  sorry
end

end inequality_1_over_1_plus_a_squared_equality_holds_for_l343_343559


namespace necessary_and_sufficient_problem_l343_343340

theorem necessary_and_sufficient_problem : 
  (¬ (∀ x : ℝ, (-2 < x ∧ x < 1) → (|x| > 1)) ∧ ¬ (∀ x : ℝ, (|x| > 1) → (-2 < x ∧ x < 1))) :=
by {
  sorry
}

end necessary_and_sufficient_problem_l343_343340


namespace find_f_one_l343_343218

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := if x ≤ 0 then 2 * x ^ 2 - x else 0  -- Dummy case for x > 0

theorem find_f_one (h_odd : is_odd_function f) (h_cond : ∀ x : ℝ, x ≤ 0 → f x = 2 * x ^ 2 - x) : f 1 = -3 :=
by
  sorry

end find_f_one_l343_343218


namespace vasya_wins_l343_343058

theorem vasya_wins : 
  ∀ (grid : Matrix (Fin 2020) (Fin 2021) ℕ), 
  (∀ (turns : ℕ) (move : Fin 2020 × Fin 2021), 
     ∃ square : Fin 2018 × Fin 2017, 
       (∀ (i j : ℕ), i < 4 → j < 4 → grid.get (square.fst + i) (square.snd + j) > 0) → False) → 
  False :=
sorry

end vasya_wins_l343_343058


namespace quadratic_eq_real_roots_roots_diff_l343_343082

theorem quadratic_eq_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + (m-2)*x - m = 0) ∧
  (y^2 + (m-2)*y - m = 0) := sorry

theorem roots_diff (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0)
  (h_roots : (m^2 + (m-2)*m - m = 0) ∧ (n^2 + (m-2)*n - m = 0)) :
  m - n = 5/2 := sorry

end quadratic_eq_real_roots_roots_diff_l343_343082


namespace sum_odd_terms_sum_S_4n_l343_343287

variable {a : ℕ → ℝ}

-- Condition: a_{n+1} + (-1)^n a_n = 2n - 1
axiom seq_rec (n : ℕ) : a (n + 1) + (-1)^n * a n = 2 * n - 1

-- Problem 1
theorem sum_odd_terms : 
  ∑ k in range 50, a (2 * k + 1) = 50 :=
by
  sorry

-- Problem 2
def S (n : ℕ) : ℝ := ∑ k in range (n + 1), a k

theorem sum_S_4n (n : ℕ) : 
  S (4 * n) = 8 * n^2 + 2 * n :=
by
  sorry

end sum_odd_terms_sum_S_4n_l343_343287


namespace geometric_sequence_product_l343_343164

variable {a1 a2 a3 a4 a5 a6 : ℝ}
variable (r : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions defining the terms of a geometric sequence
def is_geometric_sequence (seq : ℕ → ℝ) (a1 r : ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = seq n * r

-- Given condition: a_3 * a_4 = 5
def given_condition (seq : ℕ → ℝ) := (seq 2 * seq 3 = 5)

-- Proving the required question: a_1 * a_2 * a_5 * a_6 = 5
theorem geometric_sequence_product
  (h_geom : is_geometric_sequence seq a1 r)
  (h_given : given_condition seq) :
  seq 0 * seq 1 * seq 4 * seq 5 = 5 :=
sorry

end geometric_sequence_product_l343_343164


namespace find_radii_l343_343161

-- Definitions based on the problem conditions
def tangent_lengths (TP T'Q r r' PQ: ℝ) : Prop :=
  TP = 6 ∧ T'Q = 10 ∧ PQ = 16 ∧ r < r'

-- The main theorem to prove the radii are 15 and 5
theorem find_radii (TP T'Q r r' PQ: ℝ) 
  (h : tangent_lengths TP T'Q r r' PQ) :
  r = 15 ∧ r' = 5 :=
sorry

end find_radii_l343_343161


namespace area_of_polygon_ABLFKJ_l343_343162

theorem area_of_polygon_ABLFKJ 
  (side_length : ℝ) (area_square : ℝ) (midpoint_l : ℝ) (area_triangle : ℝ)
  (remaining_area_each_square : ℝ) (total_area : ℝ)
  (h1 : side_length = 6)
  (h2 : area_square = side_length * side_length)
  (h3 : midpoint_l = side_length / 2)
  (h4 : area_triangle = 0.5 * side_length * midpoint_l)
  (h5 : remaining_area_each_square = area_square - 2 * area_triangle)
  (h6 : total_area = 3 * remaining_area_each_square)
  : total_area = 54 :=
by
  sorry

end area_of_polygon_ABLFKJ_l343_343162


namespace fill_time_l343_343175

theorem fill_time (start_time : ℕ) (initial_rainfall : ℕ) (subsequent_rainfall_rate : ℕ) (subsequent_rainfall_duration : ℕ) (final_rainfall_rate : ℕ) (tank_height : ℕ) :
  start_time = 13 ∧
  initial_rainfall = 2 ∧
  subsequent_rainfall_rate = 1 ∧
  subsequent_rainfall_duration = 4 ∧
  final_rainfall_rate = 3 ∧
  tank_height = 18 →
  13 + 1 + subsequent_rainfall_duration = 18 - 6 / final_rainfall_rate + 18 - tank_height * 10 :=
begin
  sorry
end

end fill_time_l343_343175


namespace number_of_valid_sequences_is_65_l343_343878

/-- A sequence is valid if it starts with 0, ends with 0, has no two consecutive 0's, and no three consecutive 1's --/
def valid_sequence (s : List ℕ) : Prop :=
  ∃ l, s.length = l ∧ l = 19 ∧ s.head = 0 ∧ s.last = 0 ∧
  ∀ i, i < l - 1 → s.nth i ≠ 0 ∨ s.nth (i + 1) ≠ 0 ∧
  ∀ i, i < l - 2 → ¬ (s.nth i = 1 ∧ s.nth (i + 1) = 1 ∧ s.nth (i + 2) = 1)

/-- A function f(n) to count valid sequences of length n --/
def f : ℕ → ℕ
| 3 := 1
| 4 := 1
| 5 := 1
| 6 := 2
| n := f (n-4) + 2 * f (n-5) + f (n-6)

/-- Prove that the number of valid sequences of length 19 is equal to 65 --/
theorem number_of_valid_sequences_is_65 : f 19 = 65 := 
  sorry

end number_of_valid_sequences_is_65_l343_343878


namespace permutation_exists_l343_343753

theorem permutation_exists (p : ℕ) (N : ℕ) (a : Fin N → Fin p) 
  (hp : Nat.Prime p) (hN : N < 50 * p)
  (ha_bound : ∀ x, (Finset.univ.filter (λ i, a i = x)).card ≤ Nat.div (51 * N) 100)
  (ha_sum : ¬ (∑ i in Finset.univ, a i).val % p = 0) : 
  ∃ b : Fin N → Fin p, ∀ k : Fin N, (∑ i in Finset.range (k.1 + 1), b i).val % p ≠ 0 := sorry

end permutation_exists_l343_343753


namespace cost_comparisons_l343_343567

-- Define the constants for the problem
def FuelTankCapacity : ℕ := 40
def FuelPrice : ℕ := 9
def BatteryCapacity : ℕ := 60
def ElectricityPrice : ℕ := 0.6
def otherExpensesFuelCar : ℕ := 4800
def otherExpensesNewEnergyCar : ℕ := 7500

-- Define the cost per kilometer for both cars
def costPerKilometerFuelCar (a : ℕ) : ℕ := FuelTankCapacity * FuelPrice / a
def costPerKilometerNewEnergyCar (a : ℕ) : ℕ := 36 / a

-- State the main theorem to be proved
theorem cost_comparisons (a : ℕ) :
  costPerKilometerNewEnergyCar a = 36 / a ∧ 
  (costPerKilometerFuelCar a - costPerKilometerNewEnergyCar a = 0.54 → 
   a = 600 ∧ 
   costPerKilometerFuelCar 600 = 0.6 ∧ 
   costPerKilometerNewEnergyCar 600 = 0.06 ∧ 
   (∀ x : ℕ, x > 5000 → 
   (0.6 * x + otherExpensesFuelCar > 0.06 * x + otherExpensesNewEnergyCar))) :=
sorry

end cost_comparisons_l343_343567


namespace investment_percentage_l343_343701

theorem investment_percentage :
  ∃ (x : ℝ), 
    let y1 := 4000 * (x / 100),
        y2 := 3500 * 0.04,
        y3 := 2500 * 0.064 in
    y1 + y2 + y3 = 500 ∧ x = 5 :=
begin
  use 5,
  have y1 := 4000 * (5 / 100),
  have y2 := 3500 * 0.04,
  have y3 := 2500 * 0.064,
  split,
  { calc
      y1 + y2 + y3 = 4000 * (5 / 100) + 3500 * 0.04 + 2500 * 0.064 : by rw [y1, y2, y3]
                ... = 200 + 140 + 160 : by norm_num
                ... = 500 : by norm_num },
  { refl }
end

end investment_percentage_l343_343701


namespace number_of_elements_in_T_l343_343947

noncomputable def g (x : ℝ) : ℝ := (x + 8) / (x - 1)

def g_seq : ℕ → ℝ → ℝ
| 1, x => g x
| n + 1, x => g (g_seq n x)

set_option pp.explicit true

def T : set ℝ := {x | ∃ n : ℕ, n > 0 ∧ g_seq n x = x}

theorem number_of_elements_in_T : T.to_finset.card = 2 := sorry

end number_of_elements_in_T_l343_343947


namespace find_positive_real_solution_l343_343800

theorem find_positive_real_solution (x : ℝ) (h1 : x > 0) (h2 : (x - 5) / 8 = 5 / (x - 8)) : x = 13 := 
sorry

end find_positive_real_solution_l343_343800


namespace line_does_not_intersect_circle_l343_343760

-- Definitions based on the given conditions
def line_eq (x y : ℝ) := x - y - 4 = 0
def circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Definitions based on circle's center and radius
def center : ℝ × ℝ := (1, 1)
def radius : ℝ := 2

-- Formula for distance from a point to a line
def distance (P : ℝ × ℝ) (a b c : ℝ) := abs (a * P.1 + b * P.2 + c) / sqrt (a^2 + b^2)
def d := distance center 1 (-1) (-4)

-- Proof problem statement
theorem line_does_not_intersect_circle : d > radius := sorry

end line_does_not_intersect_circle_l343_343760


namespace Keenan_essay_length_l343_343196

-- Given conditions
def words_per_hour_first_two_hours : ℕ := 400
def first_two_hours : ℕ := 2
def words_per_hour_later : ℕ := 200
def later_hours : ℕ := 2

-- Total words written in 4 hours
def total_words : ℕ := words_per_hour_first_two_hours * first_two_hours + words_per_hour_later * later_hours

-- Theorem statement
theorem Keenan_essay_length : total_words = 1200 := by
  sorry

end Keenan_essay_length_l343_343196


namespace calculate_two_minus_neg_four_l343_343745

theorem calculate_two_minus_neg_four : 2 - (-4) = 6 :=
by 
  calc
    2 - (-4) = 2 + 4 : by sorry
    ... = 6 : by sorry

end calculate_two_minus_neg_four_l343_343745


namespace sqrt_288_eq_12_sqrt_2_l343_343253

theorem sqrt_288_eq_12_sqrt_2 : Real.sqrt 288 = 12 * Real.sqrt 2 :=
by
sorry

end sqrt_288_eq_12_sqrt_2_l343_343253


namespace harmonic_mean_124_l343_343274

-- Define the harmonic mean
def harmonic_mean (s : Finset ℝ) : ℝ :=
  let n := s.card
  let reciprocal_sum := ∑ x in s, x⁻¹
  n / reciprocal_sum

-- Given set of numbers
def nums : Finset ℝ := {1, 2, 4}

-- Theorem to prove harmonic mean of {1, 2, 4} is 12 / 7
theorem harmonic_mean_124 : harmonic_mean nums = 12 / 7 := by
  sorry

end harmonic_mean_124_l343_343274


namespace square_area_l343_343715

theorem square_area : ∃ (s: ℝ), (∀ x: ℝ, x^2 + 4*x + 1 = 7 → ∃ t: ℝ, t = x ∧ ∃ x2: ℝ, (x2 - x)^2 = s^2 ∧ ∀ y : ℝ, y = 7 ∧ y = x2^2 + 4*x2 + 1) ∧ s^2 = 40 :=
by
  sorry

end square_area_l343_343715


namespace tank_filled_at_10pm_l343_343181

def start_time := 13 -- 1 pm in 24-hour format
def first_hour_rain := 2 -- inches
def next_four_hours_rate := 1 -- inches per hour
def next_four_hours_duration := 4 -- hours
def remaining_day_rate := 3 -- inches per hour
def tank_height := 18 -- inches

theorem tank_filled_at_10pm :
  let accumulated_rain_by_6pm := first_hour_rain + next_four_hours_rate * next_four_hours_duration in
  let remaining_rain_needed := tank_height - accumulated_rain_by_6pm in
  let remaining_hours_to_fill := remaining_rain_needed / remaining_day_rate in
  (start_time + 1 + next_four_hours_duration + remaining_hours_to_fill) = 22 := -- 10 pm in 24-hour format
by
  sorry

end tank_filled_at_10pm_l343_343181


namespace minimum_positive_Sn_l343_343473

theorem minimum_positive_Sn (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ n, a (n+1) = a n + d) →
  a 11 / a 10 < -1 →
  (∃ N, ∀ n > N, S n < S (n + 1) ∧ S 1 ≤ S n ∧ ∀ n > N, S n < 0) →
  S 19 > 0 ∧ ∀ k < 19, S k > S 19 → S 19 < 0 →
  n = 19 :=
by
  sorry

end minimum_positive_Sn_l343_343473


namespace juans_income_theorem_l343_343675

def juans_income (J T M K X : ℝ) : Prop :=
  M = 1.60 * T ∧
  T = 0.50 * J ∧
  K = 1.20 * M ∧
  X = 0.70 * K ∧
  ∀x, 0.85 / 0.3792 * 100 ≈ 224.15
-- or if lean doesn't support approx. We can use =
theorem juans_income_theorem (J T M K X : ℝ) (h : juans_income J T M K X) : 
    ( 0.85 / 0.3792) * 100  ≈ 224.15:= sorry

end juans_income_theorem_l343_343675


namespace length_of_second_offset_l343_343018

theorem length_of_second_offset 
  (d : ℝ) (offset1 : ℝ) (area : ℝ) (offset2 : ℝ) 
  (h1 : d = 40)
  (h2 : offset1 = 9)
  (h3 : area = 300) :
  offset2 = 6 :=
by
  sorry

end length_of_second_offset_l343_343018


namespace min_M_l343_343085

def f (x : ℝ) : ℝ := x^2 - 4 * x + 1

theorem min_M (n : ℕ) (x : fin n → ℝ) (H : ∀ i : fin n, 1 ≤ x i ∧ x i ≤ 4) 
  (Hord : ∀ i j : fin n, i < j → x i < x j) :
  ∃ M, M = 5 ∧ 
    (Σ i in (fin (n-1)), |f (x i) - f (x (succ i))|) ≤ M := by
  sorry

end min_M_l343_343085


namespace number_of_distinct_13_length_words_l343_343002

def first_digit_of_power_of_2 (n : ℕ) := (2^n).digitFirst

-/ Prove that the number of distinct sequences of first digits of length 13 in the powers of 2 is 57. -/
theorem number_of_distinct_13_length_words : 
  (∃ (seqs : Finset (Fin 13 → Fin 10)), seqs.card = 57 ∧ 
  (∀ seq ∈ seqs, ∃ n : ℕ, ∀ i : Fin 13, seq i = first_digit_of_power_of_2 (n + i))) :=
sorry

end number_of_distinct_13_length_words_l343_343002


namespace eval_expression_l343_343778

theorem eval_expression : (125 ^ (1/3) * 81 ^ (-1/4) * 32 ^ (1/5) = (10/3)) :=
by
  have h1 : 125 = 5^3 := by norm_num
  have h2 : 81 = 3^4 := by norm_num
  have h3 : 32 = 2^5 := by norm_num
  sorry

end eval_expression_l343_343778


namespace bijective_mappings_l343_343827

theorem bijective_mappings :
  (∀ a1 a2 : Set.Ioo 0 3, a1 + 2 = a2 + 2 → a1 = a2) ∧ 
  (∀ a1 a2 : ℝ, 0 < a1 ∧ 0 < a2 → 1 / a1 = 1 / a2 → a1 = a2) :=
by {
  -- Proofs are omitted 
  sorry
}

end bijective_mappings_l343_343827


namespace counterexample_exists_l343_343221

theorem counterexample_exists :
  ∃ n : ℕ, let digit_sum := (fun n => (n.toString.data.map (λ c => c.toNat - '0'.toNat)).sum) n in
    digit_sum % 9 = 0 ∧
    n % 3 = 0 ∧
    n % 9 ≠ 0 :=
begin
  use 18,
  let digit_sum := (fun n => (n.toString.data.map (λ c => c.toNat - '0'.toNat)).sum) 18,
  have h1 : digit_sum = 9 := by norm_num,
  have h2 : 18 % 3 = 0 := by norm_num,
  have h3 : 18 % 9 ≠ 0 := by norm_num,
  sorry
end

end counterexample_exists_l343_343221


namespace inverse_composition_has_correct_value_l343_343604

noncomputable def f (x : ℝ) : ℝ := 5 * x + 7
noncomputable def f_inv (x : ℝ) : ℝ := (x - 7) / 5

theorem inverse_composition_has_correct_value : 
  f_inv (f_inv 9) = -33 / 25 := 
by 
  sorry

end inverse_composition_has_correct_value_l343_343604


namespace transformed_data_stats_l343_343461

noncomputable def data_set (n : ℕ) : Type := vector ℝ n

noncomputable def mean (s : data_set n) := (s.to_list.sum / n)

noncomputable def variance (s : data_set n) : ℝ := 
  let μ := mean s in 
  (s.to_list.map (λ x, (x - μ)^2)).sum / n

noncomputable def transformed_data (s : data_set n) : data_set n :=
  ⟨s.to_list.map (λ x, 3 * x + 2), sorry⟩

theorem transformed_data_stats (s : data_set n) 
  (h_avg : mean s = 2) 
  (h_var : variance s = 1) :
  mean (transformed_data s) = 8 ∧ variance (transformed_data s) = 9 := sorry

end transformed_data_stats_l343_343461


namespace log_x_125_l343_343515

theorem log_x_125 (x : ℝ) (h : log 8 (5 * x) = 3) : log x 125 = 4.5984 := by
  sorry

end log_x_125_l343_343515


namespace xy_in_m_of_comm_l343_343203

open Complex
open Matrix

def M2C := Matrix (Fin 2) (Fin 2) ℂ
def I2 : M2C := 1

def inM (A : M2C) : Prop := 
  ∀ z : ℂ, det (A - z • I2) = 0 → abs z < 1

theorem xy_in_m_of_comm {X Y : M2C} (hx : inM X) (hy : inM Y) (hxy : X ⬝ Y = Y ⬝ X) : inM (X ⬝ Y) := 
sorry

end xy_in_m_of_comm_l343_343203


namespace regular_polygon_sides_and_area_l343_343709

theorem regular_polygon_sides_and_area (P s a : ℝ) (hP : P = 180) (hs : s = 15) (ha : a = 12) :
  let n := P / s in
  n = 12 ∧ (1 / 2 * P * a = 1080) :=
by
  sorry

end regular_polygon_sides_and_area_l343_343709


namespace jovana_shells_l343_343547

theorem jovana_shells {
  (initialA initialB initialC additionalA additionalB additionalC fullA fullB fullC : ℕ)
  (hA : initialA = 5) 
  (hB : initialB = 8) 
  (hC : initialC = 3) 
  (hAddA : additionalA = 12) 
  (hAddB : additionalB = 15) 
  (hAddC : additionalC = 18) 
  (hFullA : fullA = initialA + additionalA) 
  (hFullB : fullB = initialB + additionalB) 
  (hFullC : fullC = initialC + additionalC) 
} :
  fullA + fullB + fullC = 61 
:= by
  sorry

end jovana_shells_l343_343547


namespace no_positive_integers_satisfy_l343_343241

theorem no_positive_integers_satisfy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 + 1 ≠ (x + 2)^5 + (y - 3)^5 :=
sorry

end no_positive_integers_satisfy_l343_343241


namespace dhoni_leftover_percentage_l343_343764

variable (E : ℝ) (spent_on_rent : ℝ) (spent_on_dishwasher : ℝ)

def percent_spent_on_rent : ℝ := 0.40
def percent_spent_on_dishwasher : ℝ := 0.32

theorem dhoni_leftover_percentage (E : ℝ) :
  (1 - (percent_spent_on_rent + percent_spent_on_dishwasher)) * E / E = 0.28 :=
by
  sorry

end dhoni_leftover_percentage_l343_343764


namespace area_of_union_of_triangles_l343_343805

theorem area_of_union_of_triangles :
  let s := 4
  let numTriangles := 5
  let area_one_triangle := (sqrt 3 / 4) * s^2
  let total_area_without_overlap := numTriangles * area_one_triangle
  let overlap_area := 4 * (sqrt 3 / 4) * (s / 2)^2
  let net_area := total_area_without_overlap - overlap_area
  net_area = 16 * sqrt 3 := by
  sorry

end area_of_union_of_triangles_l343_343805


namespace surface_area_to_lateral_surface_ratio_cone_l343_343520

noncomputable def cone_surface_lateral_area_ratio : Prop :=
  let radius : ℝ := 1
  let theta : ℝ := (2 * Real.pi) / 3
  let lateral_surface_area := Real.pi * radius^2 * (theta / (2 * Real.pi))
  let base_radius := (2 * Real.pi * radius * (theta / (2 * Real.pi))) / (2 * Real.pi)
  let base_area := Real.pi * base_radius^2
  let surface_area := lateral_surface_area + base_area
  (surface_area / lateral_surface_area) = (4 / 3)

theorem surface_area_to_lateral_surface_ratio_cone :
  cone_surface_lateral_area_ratio :=
  by
  sorry

end surface_area_to_lateral_surface_ratio_cone_l343_343520


namespace ellipse_equation_maximum_area_triangle_l343_343849

open Real

def c := 1
def a := 2
def b := sqrt 3
def F : ℝ × ℝ := (0, 1)
def parabola : Set (ℝ × ℝ) := {p | p.1^2 = 4 * p.2}
def ellipse : Set (ℝ × ℝ) := {p | p.2^2 / 4 + p.1^2 / 3 = 1}
def M (θ : ℝ) : ℝ × ℝ := (sqrt 3 * cos θ, 2 * sin θ)
def maximum_distance := 3
def maximum_area := 8 * sqrt 2

theorem ellipse_equation:
  Point F is the common focus of the parabola x^2 = 4y and the ellipse y^2 / a^2 + x^2 / b^2 = 1 (a > b > 0), and the maximum distance from point M on the ellipse to point F is $3$ ⟹
  The equation of the ellipse is y^2 / 4 + x^2 / 3 = 1 :=
by
  sorry

theorem maximum_area_triangle:
  Point F is the common focus of the parabola x^2 = 4y and the ellipse y^2 / a^2 + x^2 / b^2 = 1 (a > b > 0), and the maximum distance from point M on the ellipse to point F is $3$ ⟹
  The maximum area of triangle MAB is 8 * sqrt 2 :=
by
  sorry

end ellipse_equation_maximum_area_triangle_l343_343849


namespace parametric_graph_self_intersections_l343_343358

-- Define the parametric equations
def param_x (t : ℝ) : ℝ := cos t + t / 2
def param_y (t : ℝ) : ℝ := sin t

-- Define the statement of the problem in Lean
/--
The graph defined by the parametric equations intersects itself 12 times 
between x = 1 and x = 40.
-/
theorem parametric_graph_self_intersections : (finset.Icc 1 40).count (λ x, 
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ param_x t1 = x ∧ param_x t2 = x) = 12 := by
  sorry

end parametric_graph_self_intersections_l343_343358


namespace point_in_fourth_quadrant_l343_343154

def x : ℝ := 8
def y : ℝ := -3

theorem point_in_fourth_quadrant (h1 : x > 0) (h2 : y < 0) : (x > 0 ∧ y < 0) :=
by {
  sorry
}

end point_in_fourth_quadrant_l343_343154


namespace value_of_expression_l343_343121

theorem value_of_expression (a b : ℚ) (h1 : a = 2⁻¹) (h2 : b = 2 / 3) : (a⁻¹ + b⁻¹)⁻² = 4 / 49 :=
by
  sorry

end value_of_expression_l343_343121


namespace angle_FDA_30_degrees_l343_343921

variables (α : Type) [EuclideanGeometry α]

-- Definitions for geometric objects.
variables (A B C D E F : α)
variables [HParallelABDC : parallel A B D C]
variables (HDape90 : ∟ D A F = 90)

-- Points E and F with respective properties
variables (HPointE : ∃ E : α, on_line D C E ∧ line_segment_length E B = line_segment_length B C ∧ line_segment_length B C = line_segment_length C E)
variables (HPointF : ∃ F : α, on_line A B F ∧ parallel D F E B)

-- Problem: What is the measure of ∠FDA?
theorem angle_FDA_30_degrees : angle_measure F D A = 30 :=
sorry

end angle_FDA_30_degrees_l343_343921


namespace find_y_l343_343067

theorem find_y (y : ℚ) :
  let R := (-3, 8)
  let S := (5, y)
  let slope := λ (p₁ p₂ : ℚ × ℚ), (p₂.2 - p₁.2) / (p₂.1 - p₁.1)
  slope R S = -4 / 3 → y = -8 / 3 :=
by
  intros
  sorry

end find_y_l343_343067


namespace max_additional_circles_packed_l343_343238

theorem max_additional_circles_packed :
  let initial_number_of_circles := 100 * 100
  let new_total := 57 * (100 + 99) + 100
  new_total - initial_number_of_circles = 1443 :=
by
  have initial_number_of_circles := 10000
  have new_total := 11443
  simp [initial_number_of_circles, new_total]
  sorry

end max_additional_circles_packed_l343_343238


namespace vasya_wins_l343_343057

theorem vasya_wins : 
  ∀ (grid : Matrix (Fin 2020) (Fin 2021) ℕ), 
  (∀ (turns : ℕ) (move : Fin 2020 × Fin 2021), 
     ∃ square : Fin 2018 × Fin 2017, 
       (∀ (i j : ℕ), i < 4 → j < 4 → grid.get (square.fst + i) (square.snd + j) > 0) → False) → 
  False :=
sorry

end vasya_wins_l343_343057


namespace evaluate_expression_l343_343771

theorem evaluate_expression :
  125^(1/3 : ℝ) * 81^(-1/4 : ℝ) * 32^(1/5 : ℝ) = 10/3 := by
  sorry

end evaluate_expression_l343_343771


namespace probability_three_digit_divisible_by_3_l343_343046

def digits : List ℕ := [0, 1, 2, 3]

noncomputable def count_total_three_digit_numbers : ℕ :=
  (3 * 3 * 2) -- 3 choices for first digit (excluding 0), 3 choices for second, 2 for third

noncomputable def count_divisible_by_three : ℕ :=
  (4 + 6) -- combination of the different ways calculable

theorem probability_three_digit_divisible_by_3 :
  (count_divisible_by_three : ℚ) / (count_total_three_digit_numbers : ℚ) = 5 / 9 := 
  sorry

end probability_three_digit_divisible_by_3_l343_343046


namespace correct_answers_l343_343094

-- Definitions based on the given conditions
def line (a b r x y : ℝ) : Prop := a * x + b * y - r ^ 2 = 0
def circle (r x y : ℝ) : Prop := x ^ 2 + y ^ 2 = r ^ 2
def point_A (a b : ℝ) : (ℝ × ℝ) := (a, b)

-- Prove that the correct answers are A, B, and D.
theorem correct_answers (a b r : ℝ) :
  (circle r a b → ∀ x y, point_A a b = (a, b) → line a b r x y → d = r) ∧
  (a ^ 2 + b ^ 2 < r ^ 2 → ∀ x y, ¬ circle r x y → line a b r x y → d > r) ∧
  (line a b r a b → ∀ x y, point_A a b = (a, b) → line a b r x y → d = r) :=
sorry

end correct_answers_l343_343094


namespace log_base8_eq_3_l343_343418

theorem log_base8_eq_3 (x : ℝ) (h : log 8 (2 * x) = 3) : x = 256 :=
by
  sorry

end log_base8_eq_3_l343_343418


namespace solution_l343_343766

variable (x y z : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (hz : z > 0)

-- Condition 1: 20/x + 6/y = 1
axiom eq1 : 20 / x + 6 / y = 1

-- Condition 2: 4/x + 2/y = 2/9
axiom eq2 : 4 / x + 2 / y = 2 / 9

-- What we need to prove: 1/z = 1/x + 1/y
axiom eq3 : 1 / x + 1 / y = 1 / z

theorem solution : z = 14.4 := by
  -- Omitted proof, just the statement
  sorry

end solution_l343_343766


namespace length_of_train_l343_343375

theorem length_of_train 
  (speed_kmph : ℝ) (speed_kmph = 55)
  (platform_length : ℝ) (platform_length = 520)
  (crossing_time : ℝ) (crossing_time = 64.79481641468682)
  : (speed_kmph / 3.6 * crossing_time - platform_length = 470) :=
sorry

end length_of_train_l343_343375


namespace complex_identity_l343_343228

noncomputable def z : ℂ := (sqrt 3 + complex.I) / 2

theorem complex_identity :
  (∑ k in finset.range 11, z ^ (k ^ 2)) * (∑ k in finset.range 11, (1 / (z ^ (k ^ 2)))) = 25 := 
sorry

end complex_identity_l343_343228


namespace hyperbola_eccentricity_l343_343939

theorem hyperbola_eccentricity
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c^2 = a^2 + b^2)
  (h4 : angle (a, b) (-c, 0) (c, 0) = real.pi / 6) :
  (c / a) = 2 :=
by
  sorry

end hyperbola_eccentricity_l343_343939


namespace no_four_points_with_all_odd_distances_l343_343250

noncomputable def distances_are_odd (A B C D : ℝ × ℝ) : Prop :=
  ∀ p q ∈ {A, B, C, D}, p ≠ q → ∃ n : ℤ, dist p q = 2 * n + 1

theorem no_four_points_with_all_odd_distances :
  ¬ ∃ (A B C D : ℝ × ℝ), distances_are_odd A B C D :=
by
  sorry

end no_four_points_with_all_odd_distances_l343_343250


namespace area_gray_region_in_terms_of_pi_l343_343919

variable (r : ℝ)

theorem area_gray_region_in_terms_of_pi 
    (h1 : ∀ (r : ℝ), ∃ (outer_r : ℝ), outer_r = r + 3)
    (h2 : width_gray_region = 3)
    : ∃ (area_gray : ℝ), area_gray = π * (6 * r + 9) := 
sorry

end area_gray_region_in_terms_of_pi_l343_343919


namespace solve_y_l343_343599

theorem solve_y (y : ℚ) : 16^(2 * y - 3) = 4^(y + 4) → y = 10 / 3 :=
by
  sorry

end solve_y_l343_343599


namespace previous_income_l343_343966

-- Define the conditions as Lean definitions
variables (p : ℝ) -- Mrs. Snyder's previous monthly income

-- Condition 1: Mrs. Snyder used to spend 40% of her income on rent and utilities
def rent_and_utilities_initial (p : ℝ) : ℝ := (2 * p) / 5

-- Condition 2: Her salary was increased by $600
def new_income (p : ℝ) : ℝ := p + 600

-- Condition 3: After the increase, rent and utilities account for 25% of her new income
def rent_and_utilities_new (p : ℝ) : ℝ := (new_income p) / 4

-- Theorem: Proving that Mrs. Snyder's previous monthly income was $1000
theorem previous_income : (2 * p) / 5 = (new_income p) / 4 → p = 1000 :=
begin
  -- By mathlib, sorry as placeholder for proof
  sorry
end

end previous_income_l343_343966


namespace frames_sharing_point_with_line_e_l343_343639

def frame_shares_common_point_with_line (n : ℕ) : Prop := 
  n = 0 ∨ n = 1 ∨ n = 9 ∨ n = 17 ∨ n = 25 ∨ n = 33 ∨ n = 41 ∨ n = 49 ∨
  n = 6 ∨ n = 14 ∨ n = 22 ∨ n = 30 ∨ n = 38 ∨ n = 46

theorem frames_sharing_point_with_line_e :
  ∀ (i : ℕ), i < 50 → frame_shares_common_point_with_line i = 
  (i = 0 ∨ i = 1 ∨ i = 9 ∨ i = 17 ∨ i = 25 ∨ i = 33 ∨ i = 41 ∨ i = 49 ∨
   i = 6 ∨ i = 14 ∨ i = 22 ∨ i = 30 ∨ i = 38 ∨ i = 46) := 
by 
  sorry

end frames_sharing_point_with_line_e_l343_343639


namespace sandy_sunday_hours_l343_343590

theorem sandy_sunday_hours (hourly_rate : ℕ) (hours_friday : ℕ) (hours_saturday : ℕ) (total_earnings : ℕ) 
  (H_rate : hourly_rate = 15) (H_friday : hours_friday = 10) (H_saturday : hours_saturday = 6)
  (H_total : total_earnings = 450) : 
  ∃ hours_sunday : ℕ, hours_sunday = 14 :=
by
  apply Exists.intro 14
  rw [H_rate, H_friday, H_saturday, H_total]
  sorry

end sandy_sunday_hours_l343_343590


namespace solve_for_A_l343_343216

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^2
def g (B x : ℝ) : ℝ := B * x^2

-- A Lean theorem that formalizes the given math problem.
theorem solve_for_A (A B : ℝ) (h₁ : B ≠ 0) (h₂ : f A B (g B 1) = 0) : A = 3 :=
by {
  sorry
}

end solve_for_A_l343_343216


namespace max_outstanding_boys_l343_343518

structure Boy :=
  (height : ℕ)
  (weight : ℕ)

def not_inferior (A B : Boy) : Prop :=
  A.height > B.height ∨ A.weight > B.weight

def is_outstanding (A : Boy) (others : List Boy) : Prop :=
  ∀ B ∈ others, not_inferior A B

theorem max_outstanding_boys (boys : List Boy) (h : boys.length = 100) : 
  (∀ (A : Boy), A ∈ boys → is_outstanding A (List.erase boys A)) → boys.length = 100 := sorry

end max_outstanding_boys_l343_343518


namespace no_such_function_exists_l343_343171

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ n : ℕ, f(f(n)) = n + 2015 :=
sorry

end no_such_function_exists_l343_343171


namespace correct_answers_l343_343093

-- Definitions based on the given conditions
def line (a b r x y : ℝ) : Prop := a * x + b * y - r ^ 2 = 0
def circle (r x y : ℝ) : Prop := x ^ 2 + y ^ 2 = r ^ 2
def point_A (a b : ℝ) : (ℝ × ℝ) := (a, b)

-- Prove that the correct answers are A, B, and D.
theorem correct_answers (a b r : ℝ) :
  (circle r a b → ∀ x y, point_A a b = (a, b) → line a b r x y → d = r) ∧
  (a ^ 2 + b ^ 2 < r ^ 2 → ∀ x y, ¬ circle r x y → line a b r x y → d > r) ∧
  (line a b r a b → ∀ x y, point_A a b = (a, b) → line a b r x y → d = r) :=
sorry

end correct_answers_l343_343093


namespace total_weight_of_juvenile_female_muscovy_ducks_is_31_5_l343_343141

noncomputable def total_ducks := 120
noncomputable def muscovy_duck_percentage := 0.45
noncomputable def female_muscovy_percentage := 0.60
noncomputable def juvenile_female_muscovy_percentage := 0.30
noncomputable def juvenile_female_muscovy_weight := 3.5

noncomputable def total_muscovy_ducks := (muscovy_duck_percentage * total_ducks).toInt
noncomputable def total_female_muscovy_ducks := (female_muscovy_percentage * total_muscovy_ducks).toInt
noncomputable def total_juvenile_female_muscovy_ducks := (juvenile_female_muscovy_percentage * total_female_muscovy_ducks).toInt
noncomputable def total_weight_juvenile_female_muscovy_ducks := total_juvenile_female_muscovy_ducks * juvenile_female_muscovy_weight

theorem total_weight_of_juvenile_female_muscovy_ducks_is_31_5 :
  total_weight_juvenile_female_muscovy_ducks = 31.5 :=
by
  sorry

end total_weight_of_juvenile_female_muscovy_ducks_is_31_5_l343_343141


namespace point_in_fourth_quadrant_l343_343155

theorem point_in_fourth_quadrant (x : ℝ) (y : ℝ) (hx : x = 8) (hy : y = -3) : x > 0 ∧ y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l343_343155


namespace range_of_a_l343_343872

-- Definitions of the sets U and A
def U := {x : ℝ | 0 < x ∧ x < 9}
def A (a : ℝ) := {x : ℝ | 1 < x ∧ x < a}

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) (H_non_empty : A a ≠ ∅) (H_not_subset : ¬ ∀ x, x ∈ A a → x ∈ U) : 
  1 < a ∧ a ≤ 9 :=
sorry

end range_of_a_l343_343872


namespace max_projection_area_of_tetrahedron_l343_343724

theorem max_projection_area_of_tetrahedron (a : ℝ) (h1 : a > 0) :
  ∃ (A : ℝ), (A = a^2 / 2) :=
by
  sorry

end max_projection_area_of_tetrahedron_l343_343724


namespace find_k_l343_343754

noncomputable def digit_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem find_k :
  ∃ k : ℕ, digit_sum (5 * (5 * (10 ^ (k - 1) - 1) / 9)) = 600 ∧ k = 87 :=
by
  sorry

end find_k_l343_343754


namespace sin_double_angle_l343_343445

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end sin_double_angle_l343_343445


namespace square_side_length_in_right_triangle_l343_343585

theorem square_side_length_in_right_triangle (DE DF : ℝ) (hDE : DE = 9) (hDF : DF = 12) (hD : is_right_triangle D E F) :
  ∃ s : ℝ, s = 45 / 7 :=
by
  sorry

end square_side_length_in_right_triangle_l343_343585


namespace coolers_total_capacity_l343_343186

theorem coolers_total_capacity :
  ∃ (C1 C2 C3 : ℕ), 
    C1 = 100 ∧ 
    C2 = C1 + (C1 / 2) ∧ 
    C3 = C2 / 2 ∧ 
    (C1 + C2 + C3 = 325) :=
sorry

end coolers_total_capacity_l343_343186


namespace important_emails_l343_343577

theorem important_emails (total_emails : ℕ) (spam_frac : ℚ) (promotional_frac : ℚ) (spam_email_count : ℕ) (remaining_emails : ℕ) (promotional_email_count : ℕ) (important_email_count : ℕ) :
  total_emails = 800 ∧ spam_frac = 3 / 7 ∧ promotional_frac = 5 / 11 ∧ spam_email_count = 343 ∧ remaining_emails = 457 ∧ promotional_email_count = 208 →
sorry

end important_emails_l343_343577


namespace max_value_f_inequality_f_l343_343496

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x * (Real.sqrt 3 / 2)) + (Real.cos x * (1 / 2))

theorem max_value_f : ∃ x : ℝ, f(x) = 1 :=
sorry

theorem inequality_f (k : ℤ) : (2 * k * Real.pi) ≤ x ∧ x ≤ (2 * Real.pi / 3) + (2 * k * Real.pi) → f(x) ≥ 1 / 2 :=
sorry

end max_value_f_inequality_f_l343_343496


namespace smallest_sum_even_3digit_l343_343318

def isEven (n : ℕ) : Prop := n % 2 = 0

def is3Digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def usesDigitsOnce (n m : ℕ) (digits : List ℕ) : Prop :=
  (n.digits ++ m.digits).perm digits

theorem smallest_sum_even_3digit (a b c d e f : ℕ)
  (h : {a, b, c, d, e, f} = {1, 2, 3, 7, 8, 9})
  (hne : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (n m : ℕ) (hnm : is3Digit n ∧ is3Digit m ∧ (isEven n ∨ isEven m))
  (hdu : usesDigitsOnce n m [a, b, c, d, e, f])
  : n + m = 561 :=
by sorry

end smallest_sum_even_3digit_l343_343318


namespace height_range_l343_343619

def triangle_side_conditions (a b : ℝ) := a = 3 ∧ b = 9

theorem height_range (a b h : ℝ) (h1 : triangle_side_conditions a b) : 
    0 < h ∧ h ≤ 3 :=
begin
  obtain ⟨rfl, rfl⟩ := h1,
  -- Proof omitted, as per the instructions
  sorry,
end

end height_range_l343_343619


namespace determine_a_l343_343495

theorem determine_a :
  ∃ (a b c d : ℕ), 
  (18 ^ a) * (9 ^ (4 * a - 1)) * (27 ^ c) = (2 ^ 6) * (3 ^ b) * (7 ^ d) ∧ 
  a * c = 4 / (2 * b + d) ∧ 
  b^2 - 4 * a * c = d ∧ 
  a = 6 := 
by
  sorry

end determine_a_l343_343495


namespace fixed_point_of_f_l343_343852

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 - a^(x+1)

theorem fixed_point_of_f (a : ℝ) (h : a ^ 0 = 1) : f a (-1) = 2 :=
by
  unfold f
  rw [h]
  norm_num
  sorry

end fixed_point_of_f_l343_343852


namespace sum_of_distances_l343_343519

open Real

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 8 = 1

def circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 10

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem sum_of_distances {P : ℝ × ℝ} (hP_hyperbola : hyperbola P.1 P.2) (hP_circle : circle P.1 P.2) :
  (distance P.1 P.2 (-sqrt 10) 0) + (distance P.1 P.2 (sqrt 10) 0) = 6 * sqrt 2 :=
sorry

end sum_of_distances_l343_343519


namespace square_area_increase_l343_343462

variable (a : ℕ)

theorem square_area_increase (a : ℕ) :
  (a + 6) ^ 2 - a ^ 2 = 12 * a + 36 :=
by
  sorry

end square_area_increase_l343_343462


namespace least_subtracted_number_correct_l343_343678

noncomputable def least_subtracted_number (n : ℕ) : ℕ :=
  n - 13

theorem least_subtracted_number_correct (n : ℕ) : 
  least_subtracted_number 997 = 997 - 13 ∧
  (least_subtracted_number 997 % 5 = 3) ∧
  (least_subtracted_number 997 % 9 = 3) ∧
  (least_subtracted_number 997 % 11 = 3) :=
by
  let x := 997 - 13
  have : x = 984 := rfl
  have h5 : x % 5 = 3 := by sorry
  have h9 : x % 9 = 3 := by sorry
  have h11 : x % 11 = 3 := by sorry
  exact ⟨rfl, h5, h9, h11⟩

end least_subtracted_number_correct_l343_343678


namespace angle_BAL_eq_angle_CAA_l343_343823

-- Definitions and conditions
variables {A B C A' B' C' B* C* B# C# K L : Point}
variables [IsTriangle A B C]
variables (midpoints : Midpoints A B C A' B' C')
variables (altitudes : Altitudes B C B* C*)
variables (midpoints_altitudes : MidpointsAltitudes B* C* B# C#)
variables (intersection_K : Intersects (Line B' B#) (Line C' C#) K)
variables (intersection_L : Intersects (Line A K) (Line BC) L)

-- The proof statement
theorem angle_BAL_eq_angle_CAA' : ∠ BAL = ∠ CAA' :=
by
  -- Insert the proof here
  sorry

end angle_BAL_eq_angle_CAA_l343_343823


namespace jason_commute_distance_l343_343928

noncomputable def distance_to_first_and_last_store (dist1 : ℝ) (dist_ratio : ℝ) (total_commute : ℝ) : ℝ :=
  let dist2 := dist1 + dist_ratio * dist1
  in (total_commute - dist1 - dist2) / 2

theorem jason_commute_distance :
  let dist1 := 6
  let dist_ratio := 2 / 3
  let total_commute := 24
  distance_to_first_and_last_store dist1 dist_ratio total_commute = 4 :=
by
  let dist1 := 6
  let dist_ratio := 2 / 3
  let total_commute := 24
  unfold distance_to_first_and_last_store
  sorry

end jason_commute_distance_l343_343928


namespace infinite_primes_dividing_expression_l343_343243

theorem infinite_primes_dividing_expression (k : ℕ) (hk : k > 0) : 
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p ∣ (2017^n + k) :=
sorry

end infinite_primes_dividing_expression_l343_343243


namespace n_plus_one_sum_of_three_squares_l343_343550

theorem n_plus_one_sum_of_three_squares (n x : ℤ) (h1 : n > 1) (h2 : 3 * n + 1 = x^2) :
  ∃ a b c : ℤ, n + 1 = a^2 + b^2 + c^2 :=
by
  sorry

end n_plus_one_sum_of_three_squares_l343_343550


namespace distance_CD_smaller_larger_square_l343_343527

theorem distance_CD_smaller_larger_square :
  (∃ s1 s2 : ℝ, s1 = 12 / 4 ∧ s2 = sqrt 36 ∧ ∃ C D : ℝ × ℝ, 
    C = (s2, s2) ∧ D = (s1 + s2, 0) ∧ real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ≈ 9.5) :=
by
  sorry

end distance_CD_smaller_larger_square_l343_343527


namespace problem_conditions_l343_343820

theorem problem_conditions (x y : ℝ) (hx : x * (Real.exp x + Real.log x + x) = 1) (hy : y * (2 * Real.log y + Real.log (Real.log y)) = 1) :
  (0 < x ∧ x < 1) ∧ (y - x > 1) ∧ (y - x < 3 / 2) :=
by
  sorry

end problem_conditions_l343_343820


namespace top_number_multiple_of_four_l343_343528

-- Definition of the number pyramid structure, starting with three equal integers n at the second row.
def number_pyramid_top_multiple_of_four (n : ℤ) : Prop :=
  let third_row := [n + n, n + n] in
  let top := (third_row.head! + third_row.tail.head!) in
  top % 4 = 0

-- The theorem to prove the given number pyramid conditions.
theorem top_number_multiple_of_four (n : ℤ) : number_pyramid_top_multiple_of_four n :=
sorry

end top_number_multiple_of_four_l343_343528


namespace proof_problem_l343_343679

def is_sufficient_but_not_necessary_condition (x : ℝ) (k : ℤ) : Prop :=
  (tan x = 1) → (∃ k : ℤ, x = 2 * ↑k * real.pi) ∧ ¬((tan x = 1) ↔ (∃ k : ℤ, x = 2 * ↑k * real.pi))

theorem proof_problem : is_sufficient_but_not_necessary_condition x k := sorry

end proof_problem_l343_343679


namespace max_terms_and_positives_l343_343349

noncomputable def max_terms_sum (M m : ℕ) (sequence : ℕ → ℤ) : Prop :=
  (∀ n, (n ≤ sequence.length - 2017) → (sequence.slice n (n + 2017)).sum < 0) ∧
  (∀ n, (n ≤ sequence.length - 2018) → (sequence.slice n (n + 2018)).sum > 0) ∧
  M = sequence.length ∧
  m = (sequence.to_list.filter (λ x, x > 0)).length ∧
  (max : ℕ) = M + m

theorem max_terms_and_positives (M m : ℕ) (sequence : ℕ → ℤ) :
  max_terms_sum M m sequence :=
sorry

end max_terms_and_positives_l343_343349


namespace cubics_inequality_l343_343049

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

end cubics_inequality_l343_343049


namespace percentage_reduced_l343_343359

theorem percentage_reduced (P : ℝ) (h : (85 * P / 100) - 11 = 23) : P = 40 :=
by 
  sorry

end percentage_reduced_l343_343359


namespace transformed_expression_value_l343_343333

theorem transformed_expression_value :
  (240 / 80) * 60 / 40 + 10 = 14.5 :=
by
  sorry

end transformed_expression_value_l343_343333


namespace problem_sister_point_pairs_l343_343533

def is_sister_point_pair (f : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  (A.2 = f A.1) ∧ (B.2 = f B.1) ∧ (A.1 = -B.1) ∧ (A.2 = -B.2)

def num_sister_point_pairs (f : ℝ → ℝ) : ℕ :=
  {AB : ℝ × ℝ | is_sister_point_pair f AB.fst AB.snd}.to_finset.card / 2

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2 * x else 1 / real.exp x

theorem problem_sister_point_pairs :
  num_sister_point_pairs f = 2 :=
sorry


end problem_sister_point_pairs_l343_343533


namespace original_group_size_l343_343697

theorem original_group_size (x : ℕ) (h1 : ∀ (x : ℕ), (1 / 20:ℝ * x:ℝ) = 1) (h2 : ∀ (x : ℕ), (1 / 40:ℝ * (x:ℝ - 10)) = 1) : x = 20 :=
sorry

end original_group_size_l343_343697


namespace rings_on_fingers_arrangement_l343_343875

-- Definitions based on the conditions
def rings : ℕ := 5
def fingers : ℕ := 5

-- Theorem statement
theorem rings_on_fingers_arrangement : (fingers ^ rings) = 5 ^ 5 := by
  sorry  -- Proof skipped

end rings_on_fingers_arrangement_l343_343875


namespace max_books_borrowed_l343_343898

theorem max_books_borrowed (total_students books_per_student : ℕ) 
  (students_with_0_books students_with_1_book students_with_2_books : ℕ)
  (total_students_borrowed_0_1_2_books remaining_students total_books books_remaining : ℕ)
  (avg_books_per_student : Rat) :
  total_students = 40 →
  books_per_student = 2 →
  students_with_0_books = 2 →
  students_with_1_book = 12 →
  students_with_2_books = 12 →
  total_students_borrowed_0_1_2_books = (students_with_0_books + students_with_1_book + students_with_2_books) →
  remaining_students = total_students - total_students_borrowed_0_1_2_books →
  total_books = total_students * books_per_student →
  books_remaining = total_books - (students_with_1_book * 1 + students_with_2_books * 2) →
  avg_books_per_student = (total_books / total_students : Rat) →
  3 * remaining_students ≤ books_remaining →
  (books_remaining - 3 * remaining_students) ≤ remaining_students →
  (∃ max_books : ℕ, max_books = 5) := 
begin
  -- All conditions and relevant initial settings are declared above,
  -- the proof is still to be filled as 'sorry' for now.
  sorry
end

end max_books_borrowed_l343_343898


namespace son_is_four_times_younger_l343_343370

-- Given Conditions
def son_age : ℕ := 9
def dad_age : ℕ := 36
def age_difference : ℕ := dad_age - son_age -- Ensure the difference in ages

-- The proof problem
theorem son_is_four_times_younger : dad_age / son_age = 4 :=
by
  -- Ensure the conditions are correct and consistent.
  have h1 : dad_age = 36 := rfl
  have h2 : son_age = 9 := rfl
  have h3 : dad_age - son_age = 27 := rfl
  sorry

end son_is_four_times_younger_l343_343370


namespace round_negative_to_nearest_tens_l343_343586

theorem round_negative_to_nearest_tens (n : ℝ) (h : n = -3657.7421) : 
  (real.round (n / 10) * 10 = -3660) :=
by 
  rw h 
  sorry

end round_negative_to_nearest_tens_l343_343586


namespace midsegment_half_length_l343_343579

theorem midsegment_half_length (A B C M N : Point) 
  (hC : midpoint C A B) 
  (hM : M ∈ segment A C) 
  (hN : N ∈ segment B C) 
  (hRatio : ratio_equal (segment_ratio A M M C) (segment_ratio N C N B)) : 
  distance M N = (1 / 2) * distance A B :=
sorry

end midsegment_half_length_l343_343579


namespace parabola_equation_l343_343564

-- Conditions
variable (p : ℝ) (M F : Point) (C : Parabola)
variables (hyp1 : ∀ p, (p ≥ 0)) -- Given parabola has the form y^2 = 2px, p >= 0
variables (hyp2 : F = C.focus) -- The focus of the parabola is at point F.
variables (hyp3 : M ∈ C) -- Point M is on the parabola C
variables (hyp4 : dist M F = 5) -- The distance between M and F is 5.
variable (circle : Circle)
variables (hyp5 : circle.diameter = ⟨M, F⟩) -- The circle has diameter MF
variables (hyp6 : (0, 2) ∈ circle) -- The circle passes through the point (0,2)

-- Goal: The equation of C is y^2 = 4x or y^2 = 16x
theorem parabola_equation (p : ℝ) (M F : Point) (C : Parabola) (circle : Circle) : 
  (∀ p, (p ≥ 0)) → (F = C.focus) → (M ∈ C) → (dist M F = 5) → (circle.diameter = ⟨M, F⟩) → 
  ((0, 2) ∈ circle) → (C.equation = "y^2 = 4x" ∨ C.equation = "y^2 = 16x") :=
by  repeat { sorry }

end parabola_equation_l343_343564


namespace roots_sum_and_product_l343_343399

theorem roots_sum_and_product :
  let equation := (3 * x + 2) * (x - 5) + (3 * x + 2) * (x - 8) = 0 in
  let roots := { x | (3 * x + 2) * (2 * x - 13) = 0 } in
  (∃ r1 r2 ∈ roots, r1 + r2 = 35 / 6 ∧ r1 * r2 = -13 / 3) :=
by
  sorry

end roots_sum_and_product_l343_343399


namespace least_square_of_conditions_l343_343282

theorem least_square_of_conditions :
  ∃ (a x y : ℕ), 0 < a ∧ 0 < x ∧ 0 < y ∧ 
  (15 * a + 165 = x^2) ∧ 
  (16 * a - 155 = y^2) ∧ 
  (min (x^2) (y^2) = 481) := 
sorry

end least_square_of_conditions_l343_343282


namespace find_ab_sum_l343_343080

-- Definitions directly appearing from the conditions.
def Point (x y z : ℝ) := (x, y, z)
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, p2.1 = p1.1 + k * (p3.1 - p1.1) ∧ p2.2 = p1.2 + k * (p3.2 - p1.2) ∧ p2.3 = p1.3 + k * (p3.3 - p1.3)

-- Using the problem's conditions and given correct answer.
theorem find_ab_sum {a b : ℝ} 
  (h : collinear (2, a, b) (a, 3, b) (a, b, 4)) : 
  a + b = 6 :=
sorry

end find_ab_sum_l343_343080


namespace find_g_at_8_l343_343207

noncomputable def f : ℝ → ℝ := λ x, x^3 - 2*x^2 + x - 2

-- Define the cubic polynomial g with the given conditions
theorem find_g_at_8 (g : ℝ → ℝ) (h1 : ∀ x, f x = (x - 1)*(x - 1)*(-2)) (h2 : g 0 = 2) (h3 : ∀ x, ∃ a b c : ℝ, g x = a*(x - 1^3)*(x - 1^3)*(x - (-2)^3)) :
  g 8 = 0 :=
sorry

end find_g_at_8_l343_343207


namespace length_of_AB_l343_343581

theorem length_of_AB (x y u v : ℝ) 
  (h1 : P divides AB in the ratio of 3 to 2) 
  (h2 : Q divides AB in the ratio of 5 to 3) 
  (h3 : dist P Q = 3) 
  (h4 : u = x + 3) 
  (h5 : v = y - 3) : 
  x + y = 120 :=
sorry

end length_of_AB_l343_343581


namespace nat_number_as_sum_l343_343980

theorem nat_number_as_sum {n : ℕ} (hn : n ≥ 33) :
  ∃ (a : list ℕ), a.sum = n ∧ a.map (λ x, (x:ℝ)⁻¹).sum = 1 :=
sorry

end nat_number_as_sum_l343_343980


namespace number_of_customers_who_did_not_want_tires_change_l343_343734

noncomputable def total_cars_in_shop : Nat := 4 + 6
noncomputable def tires_per_car : Nat := 4
noncomputable def total_tires_bought : Nat := total_cars_in_shop * tires_per_car
noncomputable def half_tires_left : Nat := 2 * (tires_per_car / 2)
noncomputable def total_half_tires_left : Nat := 2 * half_tires_left
noncomputable def tires_left_after_half : Nat := 20
noncomputable def tires_left_after_half_customers : Nat := tires_left_after_half - total_half_tires_left
noncomputable def customers_who_did_not_change_tires : Nat := tires_left_after_half_customers / tires_per_car

theorem number_of_customers_who_did_not_want_tires_change : 
  customers_who_did_not_change_tires = 4 :=
by
  sorry 

end number_of_customers_who_did_not_want_tires_change_l343_343734


namespace sqrt_inequality_l343_343954

theorem sqrt_inequality (a b c : ℝ) (θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * (Real.cos θ)^2 + b * (Real.sin θ)^2 < c) :
  Real.sqrt a * (Real.cos θ)^2 + Real.sqrt b * (Real.sin θ)^2 < Real.sqrt c :=
sorry

end sqrt_inequality_l343_343954


namespace joint_cdf_mn_mn_joint_pdf_mn_mn_cdf_rn_pdf_rn_l343_343227

section
variables {X : Type*} [MeasurableSpace X] {n : ℕ} {F : X → ℝ} {f : X → ℝ}

def M_n (Xs : Fin n → X) : X := 
  finset.univ.sup Xs

def m_n (Xs : Fin n → X) : X := 
  finset.univ.inf Xs

def R_n (Xs : Fin n → X) : ℝ := 
  F (M_n Xs) - F (m_n Xs)

noncomputable def I (P : Prop) [decidable P] : ℝ := 
  if P then 1 else 0

theorem joint_cdf_mn_mn (x y : ℝ) (Xs : Fin n → X) : 
  ∀ (Xs_id : ∀ i, F (Xs i) = F (Xs 0)) (pos_n : 2 ≤ n),
  F (m_n Xs) ≤ x ∧ F (M_n Xs) ≤ y = 
    (if y > x then (F y)^n - (F y - F x)^n else (F y)^n) :=
sorry

theorem joint_pdf_mn_mn (x y : ℝ) (Xs : Fin n → X) : 
  ∀ (Xs_id : ∀ i, F (Xs i) = F (Xs 0)) (pos_n : 2 ≤ n),
  f (m_n Xs) ≤ x ∧ f (M_n Xs) ≤ y = 
    n * (n-1) * ((F y - F x)^(n-2)) * (f x) * (f y) * (I (y ≥ x)) :=
sorry

theorem cdf_rn (x : ℝ) (X : Fin n → X) :
  ∀ (pos_n : 2 ≤ n),
  F (R_n X) = 
  n * (I (x ≥ 0)) * (∫ F y, (F y - F (y - x))^(n-1) * (f y) ∂ measure_space.volume) :=
sorry

theorem pdf_rn (x : ℝ) (X : Fin n → X) :
  ∀ (pos_n : 2 ≤ n),
  f (R_n X) = 
  n * (n - 1) * (I (x ≥ 0)) * 
  (∫ F y, (F y - F (y - x))^(n-2) * f (y - x) * f (y) ∂ measure_space.volume) :=
sorry
end

end joint_cdf_mn_mn_joint_pdf_mn_mn_cdf_rn_pdf_rn_l343_343227


namespace modulus_of_z_l343_343264

theorem modulus_of_z :
  ∃ z : ℂ, z * (1 + complex.I) = (2 - complex.I) ∧ complex.abs z = real.sqrt 10 / 2 := sorry

end modulus_of_z_l343_343264


namespace combined_area_sandboxes_l343_343935

-- Definitions for the first sandbox
def width_sandbox1 : ℝ := 5
def length_sandbox1 : ℝ := 2 * width_sandbox1
def perimeter_sandbox1 : ℝ := 2 * length_sandbox1 + 2 * width_sandbox1

-- Definitions for the second sandbox
def diagonal_sandbox2 : ℝ := 15
def width_sandbox2 : ℝ := Real.sqrt (22.5)
def length_sandbox2 : ℝ := 3 * width_sandbox2

-- Calculating the area
def area_sandbox1 : ℝ := length_sandbox1 * width_sandbox1
def area_sandbox2 : ℝ := length_sandbox2 * width_sandbox2

-- The combined area
theorem combined_area_sandboxes : (area_sandbox1 + area_sandbox2) ≈ 117.42 := by
  sorry

end combined_area_sandboxes_l343_343935


namespace compute_b_l343_343131

theorem compute_b (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = 3) : b = 66 :=
sorry

end compute_b_l343_343131


namespace closest_point_on_line_l343_343433

theorem closest_point_on_line (P : ℝ × ℝ) (hP : P = (2, -1)) :
  ∃ Q : ℝ × ℝ, Q = (-6/5, 3/5) ∧ (∃ (x : ℝ), Q = (x, 2*x + 3)) ∧ 
  ∀ R : ℝ × ℝ, (∃ (x : ℝ), R = (x, 2*x + 3)) → dist P Q ≤ dist P R :=
by
  -- Initial conditions: line equation y = 2x + 3 and point (2, -1)
  let l_eq := λ x : ℝ, 2 * x + 3
  let pt := (2, -1)

  -- Correct answer (closest point)
  let closest_pt := (-6 / 5, 3 / 5)

  -- Theorems and conditions based on problem
  use closest_pt
  split
  . exact congrArg prod.mk (by norm_num) (by norm_num)
  split
  . use -6 / 5
    exact congrArg prod.mk rfl (by norm_num)
  . intro R hR
    obtain ⟨xR, hxR⟩ := hR
    let dPQ := dist pt closest_pt
    let dPR := dist pt R
    -- Skipping the exact distance calculations
    sorry

end closest_point_on_line_l343_343433


namespace evaluate_expression_l343_343776

theorem evaluate_expression : (125^(1/3 : ℝ)) * (81^(-1/4 : ℝ)) * (32^(1/5 : ℝ)) = (10 / 3 : ℝ) :=
by
  sorry

end evaluate_expression_l343_343776


namespace equilibrium_constant_l343_343304

theorem equilibrium_constant (C_NO2 C_O2 C_NO : ℝ) (h_NO2 : C_NO2 = 0.4) (h_O2 : C_O2 = 0.3) (h_NO : C_NO = 0.2) :
  (C_NO2^2 / (C_O2 * C_NO^2)) = 13.3 := by
  rw [h_NO2, h_O2, h_NO]
  sorry

end equilibrium_constant_l343_343304


namespace probability_interval_l343_343884

-- Define the intervals
def I : set ℝ := set.Icc 0 3
def J : set ℝ := set.Ioo 1 2

-- Length of the whole interval I
noncomputable def length_I : ℝ := 3

-- Length of the subinterval J
noncomputable def length_J : ℝ := 1

-- Probability that a randomly chosen number in I falls in J
theorem probability_interval :
  (length_J / length_I) = 1 / 3 :=
by
  sorry

end probability_interval_l343_343884


namespace tank_filled_by_10pm_l343_343176

-- Defining the problem setup
def starts_raining (t : Real) : Prop := t ≥ 13  -- 13 represents 1 pm
def rainfall_rate (t : Real) : Real :=
  if t < 14 then 2       -- 2 inches from 1 pm to 2 pm
  else if t < 18 then 1  -- 1 inch per hour from 2 pm to 6 pm
  else 3                 -- 3 inches per hour from 6 pm onwards
def tank_height : Real := 18 -- Height of the fish tank is 18 inches

-- Calculating the total rainfall by integrating the rainfall rate
noncomputable def total_rainfall (t : Real) : Real :=
  ∫ τ in (13:ℝ)..t, rainfall_rate τ

-- The statement of the proof problem
theorem tank_filled_by_10pm : total_rainfall 22 = tank_height :=
by
  sorry

end tank_filled_by_10pm_l343_343176


namespace correct_answers_l343_343095

-- Definitions based on the given conditions
def line (a b r x y : ℝ) : Prop := a * x + b * y - r ^ 2 = 0
def circle (r x y : ℝ) : Prop := x ^ 2 + y ^ 2 = r ^ 2
def point_A (a b : ℝ) : (ℝ × ℝ) := (a, b)

-- Prove that the correct answers are A, B, and D.
theorem correct_answers (a b r : ℝ) :
  (circle r a b → ∀ x y, point_A a b = (a, b) → line a b r x y → d = r) ∧
  (a ^ 2 + b ^ 2 < r ^ 2 → ∀ x y, ¬ circle r x y → line a b r x y → d > r) ∧
  (line a b r a b → ∀ x y, point_A a b = (a, b) → line a b r x y → d = r) :=
sorry

end correct_answers_l343_343095


namespace monotone_f_solve_inequality_range_of_a_l343_343074

noncomputable def e := Real.exp 1
noncomputable def f (x : ℝ) : ℝ := e^x + 1/(e^x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.log ((3 - a) * (f x - 1/e^x) + 1) - Real.log (3 * a) - 2 * x

-- Part 1: Monotonicity of f(x)
theorem monotone_f : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by sorry

-- Part 2: Solving the inequality f(2x) ≥ f(x + 1)
theorem solve_inequality : ∀ x : ℝ, f (2 * x) ≥ f (x + 1) ↔ x ≥ 1 ∨ x ≤ -1 / 3 :=
by sorry

-- Part 3: Finding the range of a
theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x → g x a ≤ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
by sorry

end monotone_f_solve_inequality_range_of_a_l343_343074


namespace solve_equation_l343_343601

def equation (x : ℝ) := (x / (x - 2)) + (2 / (x^2 - 4)) = 1

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) : 
  equation x ↔ x = -3 :=
by
  sorry

end solve_equation_l343_343601


namespace proposition_A_sufficient_but_not_necessary_l343_343828

-- Statement of the problem
variables {E F G H : Type*} [point_space E] [point_space F] [point_space G] [point_space H]

-- Definitions used in conditions
def not_coplanar (E F G H : point_space) : Prop :=
  ¬∃ (plane : set point_space), E ∈ plane ∧ F ∈ plane ∧ G ∈ plane ∧ H ∈ plane

def not_intersecting_lines (E F G H : point_space) : Prop :=
  let lineEF := line_through E F,
      lineGH := line_through G H in
  lineEF ∩ lineGH = ∅

-- Equivalent proof problem in Lean
theorem proposition_A_sufficient_but_not_necessary (E F G H : point_space)
  (h1 : not_coplanar E F G H)
  (h2 : not_intersecting_lines E F G H) :
  (h1 → h2) ∧ (¬(h2 → h1)) :=
by {
  sorry -- proof is omitted as per instructions
}

end proposition_A_sufficient_but_not_necessary_l343_343828


namespace tangent_parallel_to_line_at_point_l343_343388

theorem tangent_parallel_to_line_at_point :
  ∃ x y : ℝ, y = (1/3) * x^3 - 3 * x^2 + 8 * x + 4 ∧ 
  deriv (λ x : ℝ, (1/3) * x^3 - 3 * x^2 + 8 * x + 4) x = -1 ∧ 
  (x = 3 ∧ y = 10) :=
by
  use [3, 10]
  split
  {
    -- Show that y = (1/3) * 3^3 - 3 * 3^2 + 8 * 3 + 4
    sorry
  }
  split
  {
    -- Show that deriv (λ x : ℝ, (1/3) * x^3 - 3 * x^2 + 8 * x + 4) 3 = -1
    sorry
  }
  -- Show that (3, 10)
  {
    split
    {
      refl
    }
    {
      refl
    }
  }

end tangent_parallel_to_line_at_point_l343_343388


namespace linear_equation_a_val_l343_343916

theorem linear_equation_a_val (a x y : ℝ)
  (h1 : (a^2 - 4) * x^2 + (2 - 3a) * x + (a + 1) * y + 3a = 0)
  (h2 : (a^2 - 4) = 0)
  (h3 : (2 - 3a) ≠ 0)
  (h4 : (a + 1) ≠ 0) :
  a = 2 ∨ a = -2 :=
begin
  sorry
end

end linear_equation_a_val_l343_343916


namespace max_sum_of_15x15_grid_l343_343909

def cell_value (n : ℕ) : Prop := n ≤ 4

def grid_sum_to_seven (grid : ℕ → ℕ → ℕ) : Prop := 
  ∀ i j, i < 14 ∧ j < 14 → grid i j + grid (i + 1) j + grid i (j + 1) + grid (i + 1) (j + 1) = 7

theorem max_sum_of_15x15_grid : ∃ (grid : ℕ → ℕ → ℕ), 
  (∀ i j, i < 15 ∧ j < 15 → cell_value (grid i j)) ∧ 
  grid_sum_to_seven grid ∧ 
  (Σ i j, if i < 15 ∧ j < 15 then grid i j else 0) = 417 := 
sorry

end max_sum_of_15x15_grid_l343_343909


namespace sequence_is_arithmetic_not_geometric_l343_343288

-- Define the sequence as a function
def sequence (n : ℕ) : ℕ := 0

-- Definitions for arithmetic and geometric sequences
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℕ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = (a n : ℝ) * r

theorem sequence_is_arithmetic_not_geometric :
  is_arithmetic_sequence sequence 0 ∧ ¬ ∃ r : ℝ, is_geometric_sequence (λ n, (sequence n : ℝ)) r :=
by
  sorry

end sequence_is_arithmetic_not_geometric_l343_343288


namespace rate_of_interest_l343_343676

theorem rate_of_interest (P R : ℝ) :
  (2 * P * R) / 100 = 320 ∧
  P * ((1 + R / 100) ^ 2 - 1) = 340 →
  R = 12.5 :=
by
  intro h
  sorry

end rate_of_interest_l343_343676


namespace value_of_expression_l343_343278

theorem value_of_expression (y : ℝ) (h : 2*y^3 + 3*y^2 - 2*y - 8 = 0):
  y = 2 → (5 * y - 2)^2 = 64 :=
begin
  sorry
end

end value_of_expression_l343_343278


namespace odd_sum_of_conditions_l343_343516

theorem odd_sum_of_conditions 
  (a b c : ℕ) 
  (h_b_odd : b % 2 = 1) 
  (h_c_odd : c % 2 = 1) 
  : 
  let d := if c % 2 = 0 then 2 * c else c + 1 in 
  (3^a + (b-1)^2 * d) % 2 = 1 :=
by 
  sorry

end odd_sum_of_conditions_l343_343516


namespace pints_in_two_liters_nearest_tenth_l343_343847

def liters_to_pints (liters : ℝ) : ℝ :=
  2.1 * liters

theorem pints_in_two_liters_nearest_tenth :
  liters_to_pints 2 = 4.2 :=
by
  sorry

end pints_in_two_liters_nearest_tenth_l343_343847


namespace pos_int_solutions_3x_2y_841_l343_343279

theorem pos_int_solutions_3x_2y_841 :
  {n : ℕ // ∃ (x y : ℕ), 3 * x + 2 * y = 841 ∧ x > 0 ∧ y > 0} =
  {n : ℕ // n = 140} := 
sorry

end pos_int_solutions_3x_2y_841_l343_343279


namespace min_tickets_to_ensure_match_l343_343638

theorem min_tickets_to_ensure_match : 
  ∀ (host_ticket : Fin 50 → Fin 50),
  ∃ (tickets : Fin 26 → Fin 50 → Fin 50),
  ∀ (i : Fin 26), ∃ (k : Fin 50), host_ticket k = tickets i k :=
by sorry

end min_tickets_to_ensure_match_l343_343638


namespace rick_sean_total_money_l343_343249

theorem rick_sean_total_money :
  ∀ (fritz_money sean_money rick_money : ℕ),
  (sean_money = (fritz_money / 2) + 4) →
  (rick_money = 3 * sean_money) →
  (fritz_money = 40) →
  (rick_money + sean_money = 96) :=
by
  intros fritz_money sean_money rick_money
  assume h1 h2 h3
  sorry

end rick_sean_total_money_l343_343249


namespace rent_of_first_apartment_l343_343702

theorem rent_of_first_apartment (R : ℝ) :
  let cost1 := R + 260 + (31 * 20 * 0.58)
  let cost2 := 900 + 200 + (21 * 20 * 0.58)
  (cost1 - cost2 = 76) → R = 800 :=
by
  intro h
  sorry

end rent_of_first_apartment_l343_343702


namespace cards_with_1_count_l343_343054

theorem cards_with_1_count (m k : ℕ) 
  (h1 : k = m + 100) 
  (sum_of_products : (m * (m - 1) / 2) + (k * (k - 1) / 2) - m * k = 1000) : 
  m = 3950 :=
by
  sorry

end cards_with_1_count_l343_343054


namespace zarnin_staffing_l343_343749

theorem zarnin_staffing (n total unsuitable : ℕ) (unsuitable_factor : ℕ) (job_openings : ℕ)
  (h1 : total = 30) 
  (h2 : unsuitable_factor = 2 / 3) 
  (h3 : unsuitable = unsuitable_factor * total) 
  (h4 : n = total - unsuitable)
  (h5 : job_openings = 5) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 30240 := by
    sorry

end zarnin_staffing_l343_343749


namespace candidate_C_is_inverse_proportion_l343_343661

/--
Check whether the given function is an inverse proportion function.
-/
def is_inverse_proportion (f : ℝ → ℝ) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/--
The candidate functions are defined as follows:
A: y = x / 3
B: y = 3 / (x + 1)
C: xy = 3
D: y = 3x
-/
def candidate_A (x : ℝ) : ℝ := x / 3
def candidate_B (x : ℝ) : ℝ := 3 / (x + 1)
def candidate_C (x : ℝ) : ℝ := 3 / x
def candidate_D (x : ℝ) : ℝ := 3 * x

theorem candidate_C_is_inverse_proportion : is_inverse_proportion candidate_C :=
  sorry

end candidate_C_is_inverse_proportion_l343_343661


namespace inversion_to_concentric_circles_l343_343984

-- Given non-intersecting circles or a circle and a line
variables {S1 S2 : Type} [is_circle S1] [is_circle_or_line S2]
variables (O1 O2 : point) (C : point) -- Centers and point with equal tangents

open_locale classical

-- Definition of inversion resulting in concentric circles
theorem inversion_to_concentric_circles 
  (rad_axis_condition : is_radical_axis (S1, S2, C))
  (equal_tangents_condition : tangents_equal_length (S1, S2, C))
  (orthogonal_condition : orthogonal_to_both (S1, S2, circle C))
  : ∃ S1' S2', is_inverted S1' S2' ∧ concentric S1' S2' :=
by
  sorry

end inversion_to_concentric_circles_l343_343984


namespace arithmetic_sequence_sum_l343_343157

variable (a : ℕ → ℤ)

def arithmetic_sequence_condition_1 := a 5 = 3
def arithmetic_sequence_condition_2 := a 6 = -2

theorem arithmetic_sequence_sum :
  arithmetic_sequence_condition_1 a →
  arithmetic_sequence_condition_2 a →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 3 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_l343_343157


namespace rooks_placement_possible_l343_343172

/-- 
  It is possible to place 8 rooks on a chessboard such that they do not attack each other
  and each rook stands on cells of different colors, given that the chessboard is divided 
  into 32 colors with exactly two cells of each color.
-/
theorem rooks_placement_possible :
  ∃ (placement : Fin 8 → Fin 8 × Fin 8),
    (∀ i j, i ≠ j → (placement i).fst ≠ (placement j).fst ∧ (placement i).snd ≠ (placement j).snd) ∧
    (∀ i j, i ≠ j → (placement i ≠ placement j)) ∧
    (∀ c : Fin 32, ∃! p1 p2, placement p1 = placement p2 ∧ (placement p1).fst ≠ (placement p2).fst 
                        ∧ (placement p1).snd ≠ (placement p2).snd) :=
by
  sorry

end rooks_placement_possible_l343_343172


namespace correct_inequalities_l343_343962

variable {f g : ℝ → ℝ}
variable (a b : ℝ)
variable [Differentiable ℝ f] [Differentiable ℝ g]
variable (h_deriv : ∀ x, deriv f x > deriv g x)

theorem correct_inequalities (h_a_lt_b : a < b) :
  (∀ x, a < x ∧ x < b → f(x) + g(b) < g(x) + f(b)) ∧
  (∀ x, a < x ∧ x < b → f(x) + g(a) > g(x) + f(a)) :=
by
  sorry

end correct_inequalities_l343_343962


namespace convert_spherical_to_cartesian_l343_343060

theorem convert_spherical_to_cartesian :
  let ρ := 5
  let θ₁ := 3 * Real.pi / 4
  let φ₁ := 9 * Real.pi / 5
  let φ' := 2 * Real.pi - φ₁
  let θ' := θ₁ + Real.pi
  ∃ (θ : ℝ) (φ : ℝ),
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    0 ≤ φ ∧ φ ≤ Real.pi ∧
    (∃ (x y z : ℝ),
      x = ρ * Real.sin φ' * Real.cos θ' ∧
      y = ρ * Real.sin φ' * Real.sin θ' ∧
      z = ρ * Real.cos φ') ∧
    θ = θ' ∧ φ = φ' :=
by
  sorry

end convert_spherical_to_cartesian_l343_343060


namespace smallest_number_of_coins_l343_343377

theorem smallest_number_of_coins (d q : ℕ) (h₁ : 10 * d + 25 * q = 265) (h₂ : d > q) :
  d + q = 16 :=
sorry

end smallest_number_of_coins_l343_343377


namespace quadratic_root_probability_l343_343941

noncomputable def region_S : set (ℝ × ℝ) := 
  {p | -2 ≤ p.snd ∧ p.snd ≤ |p.fst| ∧ -2 ≤ p.fst ∧ p.fst ≤ 2}

noncomputable def quadratic_condition (p : ℝ × ℝ) : Prop :=
  let t1 := -(|p.fst| - 1)
  let t2 := |p.snd| - 2
  |p.fst| + |p.snd| < 2

theorem quadratic_root_probability :
  (measure_theory.measure_theory.measure (quadratic_condition '' region_S)) / 
  (measure_theory.measure_theory.measure region_S) = 3 / 4 :=
sorry

end quadratic_root_probability_l343_343941


namespace relationship_l343_343819

noncomputable def a : ℝ := 2^(1.3)
noncomputable def b : ℝ := 4^(0.7)
noncomputable def c : ℝ := Real.log 8 / Real.log 3

theorem relationship : c < a ∧ a < b := by
  sorry

end relationship_l343_343819


namespace four_S_eq_a2_minus_bc2_ctg_alpha2_l343_343979

theorem four_S_eq_a2_minus_bc2_ctg_alpha2
  (a b c α S : ℝ)
  (h1 : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α)
  (h2 : S = 1/2 * b * c * Real.sin α)
  (trig1 : Real.sin α = 2 * Real.sin(α / 2) * Real.cos(α / 2))
  (trig2 : Real.tan(α / 2) * Real.cot(α / 2) = 1)
  (trig3 : 1 - Real.cos α = 2 * Real.sin(α / 2) ^ 2) :
  4 * S = (a^2 - (b - c)^2) * Real.cot(α / 2) := 
sorry

end four_S_eq_a2_minus_bc2_ctg_alpha2_l343_343979


namespace ratio_of_building_heights_l343_343695

theorem ratio_of_building_heights (F_h F_s A_s B_s : ℝ) (hF_h : F_h = 18) (hF_s : F_s = 45)
  (hA_s : A_s = 60) (hB_s : B_s = 72) :
  let h_A := (F_h / F_s) * A_s
  let h_B := (F_h / F_s) * B_s
  (h_A / h_B) = 5 / 6 :=
by
  sorry

end ratio_of_building_heights_l343_343695


namespace divisibility_equivalence_l343_343214

theorem divisibility_equivalence (a b c d : ℤ) (h : a ≠ c) :
  (a - c) ∣ (a * b + c * d) ↔ (a - c) ∣ (a * d + b * c) :=
by
  sorry

end divisibility_equivalence_l343_343214


namespace even_number_of_three_colored_vertices_l343_343530

-- Define the conditions
structure ConvexPolyhedron where
  vertices : Type
  faces : vertices → Fin 3 → Sort
  coloring : faces → Fin 3
  condition1 : ∀ v : vertices, faces v 0 ≠ faces v 1 ∧ faces v 1 ≠ faces v 2 ∧ faces v 2 ≠ faces v 0
  condition2 : ∀ v : vertices, ∃ c : Fin 3 → ℕ, (c 0, c 1, c 2) = (red, yellow, blue) ∨ (c 0, c 1, c 2) = (red, blue, yellow) ∨ (c 0, c 1, c 2) = (yellow, red, blue) ∨ (c 0, c 1, c 2) = (yellow, blue, red) ∨ (c 0, c 1, c 2) = (blue, red, yellow) ∨ (c 0, c 1, c 2) = (blue, yellow, red)

-- The theorem stating the desired property
theorem even_number_of_three_colored_vertices (P : ConvexPolyhedron) : 
    ∃ n, (∃ (v : vertices P), faces P v 0 = red ∧ faces P v 1 = yellow ∧ faces P v 2 = blue) = 2 * n := 
by
    sorry

end even_number_of_three_colored_vertices_l343_343530


namespace larger_value_algebraic_expression_is_2_l343_343073

noncomputable def algebraic_expression (a b c d x : ℝ) : ℝ :=
  x^2 + a + b + c * d * x

theorem larger_value_algebraic_expression_is_2
  (a b c d : ℝ) (x : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : x = 1 ∨ x = -1) :
  max (algebraic_expression a b c d 1) (algebraic_expression a b c d (-1)) = 2 :=
by
  -- Proof is omitted.
  sorry

end larger_value_algebraic_expression_is_2_l343_343073


namespace nine_pointed_star_angles_l343_343970

theorem nine_pointed_star_angles (β : ℝ) (angles : Fin 9 → ℝ) :
  (∀ i, angles i = β) → ∑ i, angles i = 720 :=
by
  sorry

end nine_pointed_star_angles_l343_343970


namespace evaluate_expression_l343_343768

theorem evaluate_expression :
  125^(1/3 : ℝ) * 81^(-1/4 : ℝ) * 32^(1/5 : ℝ) = 10/3 := by
  sorry

end evaluate_expression_l343_343768


namespace jenny_stamps_l343_343546

theorem jenny_stamps :
  let num_books := 8
  let pages_per_book := 42
  let stamps_per_page := 6
  let new_stamps_per_page := 10
  let complete_books_in_new_system := 4
  let pages_in_fifth_book := 33
  (num_books * pages_per_book * stamps_per_page) % new_stamps_per_page = 6 :=
by
  sorry

end jenny_stamps_l343_343546


namespace customers_not_wanting_change_l343_343731

-- Given Conditions
def cars_initial := 4
def cars_additional := 6
def cars_total := cars_initial + cars_additional
def tires_per_car := 4
def half_change_customers := 2
def tires_for_half_change_customers := 2 * 2 -- 2 cars, 2 tires each
def tires_left := 20

-- Theorem to Prove
theorem customers_not_wanting_change : 
  (cars_total * tires_per_car) - (tires_left + tires_for_half_change_customers) = 
  4 * tires_per_car -> 
  cars_total - ((tires_left + tires_for_half_change_customers) / tires_per_car) - half_change_customers = 4 :=
by
  sorry

end customers_not_wanting_change_l343_343731


namespace mass_percentage_H_chromic_acid_l343_343432

-- Define the atomic masses
def mass_H : ℝ := 1.01
def mass_Cr : ℝ := 51.99
def mass_O : ℝ := 16.00

-- Define the molecular formula of chromic acid H2CrO4
def molar_mass_H2CrO4 : ℝ := (2 * mass_H) + mass_Cr + (4 * mass_O)

-- Calculate the total mass of hydrogen in H2CrO4
def total_mass_H_in_H2CrO4 : ℝ := 2 * mass_H

-- Calculate the mass percentage of hydrogen in chromic acid
def mass_percentage_H_in_H2CrO4 : ℝ := (total_mass_H_in_H2CrO4 / molar_mass_H2CrO4) * 100

-- Prove the mass percentage calculation
theorem mass_percentage_H_chromic_acid : mass_percentage_H_in_H2CrO4 = 1.712 := by
  -- Calculation steps are omitted in the theorem
  sorry

end mass_percentage_H_chromic_acid_l343_343432


namespace common_ratio_geometric_sequence_l343_343608

theorem common_ratio_geometric_sequence (n : ℕ) :
  ∃ q : ℕ, (∀ k : ℕ, q = 4^(2*k+3) / 4^(2*k+1)) ∧ q = 16 :=
by
  use 16
  sorry

end common_ratio_geometric_sequence_l343_343608


namespace num_pairs_eq_seven_l343_343110

theorem num_pairs_eq_seven :
  ∃ S : Finset (Nat × Nat), 
    (∀ (a b : Nat), (a, b) ∈ S ↔ (0 < a ∧ 0 < b ∧ a + b ≤ 100 ∧ (a + 1 / b) / (1 / a + b) = 13)) ∧
    S.card = 7 :=
sorry

end num_pairs_eq_seven_l343_343110


namespace evaluate_stability_of_yields_l343_343642

def variance (l : List ℝ) : ℝ :=
l.map (λ x, (x - l.sum / l.length)^2).sum / l.length

theorem evaluate_stability_of_yields (x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 x_9 x_{10} : ℝ) :
  let yields := [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_{10}] in
  let mean := yields.sum / yields.length in
  variance yields = (yields.map (λ x, (x - mean)^2)).sum / yields.length :=
  sorry

end evaluate_stability_of_yields_l343_343642


namespace seeds_per_bag_l343_343351

theorem seeds_per_bag 
  (ears_per_row : ℕ) (seeds_per_ear : ℕ) 
  (payment_per_row : ℕ) (dinner_cost : ℕ) 
  (bags_used : ℕ) (total_kids : ℕ) : 
  ears_per_row = 70 ∧ seeds_per_ear = 2 ∧ payment_per_row = 15 ∧ dinner_cost = 36 ∧ bags_used = 140 ∧ total_kids = 4 → 
  seeds_per_bag = 48 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  cases h6 with h7 h8
  cases h8 with h9 h10
  have ears_per_row_val : ears_per_row = 70 := h1
  have seeds_per_ear_val : seeds_per_ear = 2 := h2
  have payment_per_row_val : payment_per_row = 15 := h3
  have dinner_cost_val : dinner_cost = 36 := h4
  have bags_used_val : bags_used = 140 := h5
  have total_kids_val : total_kids = 4 := h6
  sorry

end seeds_per_bag_l343_343351


namespace probability_at_least_one_woman_selected_l343_343125

open Classical

noncomputable def probability_of_selecting_at_least_one_woman : ℚ :=
  1 - (10 / 15) * (9 / 14) * (8 / 13) * (7 / 12) * (6 / 11)

theorem probability_at_least_one_woman_selected :
  probability_of_selecting_at_least_one_woman = 917 / 1001 :=
sorry

end probability_at_least_one_woman_selected_l343_343125


namespace largest_valid_subset_l343_343603

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_set (A : Finset ℕ) : Prop :=
  A ⊆ { n ∈ Finset.range 2010 | True } ∧
  (∀ {a b : ℕ}, a ∈ A → b ∈ A → a ≠ b → ¬ is_prime (abs (a - b)))

theorem largest_valid_subset :
  ∃ A : Finset ℕ, valid_set A ∧ A.card = 503 ∧ 
  (∀ n, valid_set n → n.card <= 503) ∧ 
  (A = { n | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 502 ∧ n = 1 + 4 * k }) :=
begin
  sorry
end

end largest_valid_subset_l343_343603


namespace common_ratio_of_geometric_series_l343_343425

theorem common_ratio_of_geometric_series : ∃ r : ℝ, ∀ n : ℕ, 
  r = (if n = 0 then 2 / 3
       else if n = 1 then (2 / 3) * (2 / 3)
       else if n = 2 then (2 / 3) * (2 / 3) * (2 / 3)
       else sorry)
  ∧ r = 2 / 3 := sorry

end common_ratio_of_geometric_series_l343_343425


namespace triangle_angle_sum_l343_343536

theorem triangle_angle_sum {x : ℝ} (h : 60 + 5 * x + 3 * x = 180) : x = 15 :=
by
  sorry

end triangle_angle_sum_l343_343536


namespace max_value_x_plus_y_l343_343225

theorem max_value_x_plus_y :
  ∃ x y : ℝ, 5 * x + 3 * y ≤ 10 ∧ 3 * x + 5 * y = 15 ∧ x + y = 47 / 16 :=
by
  sorry

end max_value_x_plus_y_l343_343225


namespace NoMultipleWithSmallerDigitSum_l343_343765

-- Define the number N with m digits each being one
def N (m : ℕ) : ℕ := (List.replicate m 1).foldl (λ acc d, acc * 10 + d) 0

-- Define the digit sum of a number
def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (λ acc d, acc + d) 0

-- The theorem to prove
theorem NoMultipleWithSmallerDigitSum (m T : ℕ) (hT : T > 0) :
  (N m ∣ T) → (digitSum T < m) → False := 
by sorry

end NoMultipleWithSmallerDigitSum_l343_343765


namespace polynomial_root_l343_343116

theorem polynomial_root (h : ℚ) : Polynomial.eval 3 (Polynomial.Coeff (λ x : Polynomial ℤ, x^3 + h * x - 20)) = 0 -> h = -7 / 3 := by
  sorry

end polynomial_root_l343_343116


namespace number_of_solutions_eq_43_l343_343502

theorem number_of_solutions_eq_43 :
  let num_zeros := ((1 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (1 : ℝ)) ∧
                   ((2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (2 : ℝ)) ∧
                   ((3 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (3 : ℝ)) ∧
                   -- ... and so on up to 50
                   ((50 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (50 : ℝ))
  in let denom_zeros := ((1^2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (1^2 : ℝ)) ∧
                        ((2^2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (2^2: ℝ)) ∧
                        ((3^2 : ℕ) ∈ finset.range (50 + 1)).count (λ x, (3^2: ℝ)) ∧
                        -- ... and so on up to 7^2
                        ((7^2: ℕ) ∈ finset.range (50 + 1)).count (λ x, (7^2: ℝ))
  in num_zeros - denom_zeros = 43 :=
sorry

end number_of_solutions_eq_43_l343_343502


namespace problem_1_problem_2_problem_3_l343_343850

-- Defining the set M
def in_set_M (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f(a + x) * f(a - x) = b

-- Problem 1
theorem problem_1 (f1 f2 : ℝ → ℝ) :
  (f1 = λ x, x → ¬ in_set_M f1) ∧
  (f2 = λ x, 3^x → in_set_M f2) := by
  sorry

-- Problem 2
theorem problem_2 (t : ℝ) (f : ℝ → ℝ) :
  (f = λ x, (1 - t*x) / (1 + x) →
  ¬(∃ a b : ℝ, in_set_M f ∧ in_set_M (λ y, classical.some (classical.some_spec (classical.some_spec (exists_inv_function f)))))) := by
  sorry

-- Problem 3
theorem problem_3 (f : ℝ → ℝ) :
  in_set_M f →
  f 0 = 1 →
  f 1 = 4 →
  (∀ x [0,1], 1 ≤ f(x) ∧ f(x) ≤ 2) →
  ∀ x [-2016, 2016], 2^(-2016) ≤ f(x) ∧ f(x) ≤ 2^(2016) := 
  by sorry

end problem_1_problem_2_problem_3_l343_343850


namespace find_a9_l343_343826

variable (a : ℕ → ℤ)
variable (h1 : a 2 = -3)
variable (h2 : a 3 = -5)
variable (d : ℤ := a 3 - a 2)

theorem find_a9 : a 9 = -17 :=
by
  sorry

end find_a9_l343_343826


namespace convex_polygon_max_sides_no_adjacent_obtuse_l343_343657

theorem convex_polygon_max_sides_no_adjacent_obtuse (n : ℕ) (h : n ≥ 3) 
  (convex : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 1 ≤ ∠i ∧ ∠i ≤ 180)
  (no_adj_obtuse : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → ∠i < 180 → ∠(i + 1) < 180) 
  (exterior_sum : ∑ i in range(n), exterior_angle(i) = 360) :
  n ≤ 6 := 
sorry

end convex_polygon_max_sides_no_adjacent_obtuse_l343_343657


namespace bananas_first_day_l343_343352

theorem bananas_first_day (x : ℕ) (h : x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 100) : x = 8 := by
  sorry

end bananas_first_day_l343_343352


namespace height_of_remaining_cube_l343_343721

-- Definitions and conditions
def is_unit_cube (V : ℝ) := V = 1
def removed_pyramid_volume (Vₚ : ℝ) := Vₚ = 1 / 4
def height_of_pyramid (h : ℝ) := (1 / 3) * (sqrt 3 / 2) * h = 1 / 4

-- Main statement: calculate the height of the remaining cube
theorem height_of_remaining_cube (V Vₚ h : ℝ) (c1 : is_unit_cube V) (c2 : removed_pyramid_volume Vₚ) (c3 : height_of_pyramid h) :
  1 - h = 1 - sqrt 3 / 6 :=
by
  sorry

end height_of_remaining_cube_l343_343721


namespace intersection_is_singleton_l343_343491

namespace ProofProblem

def M : Set ℤ := {-3, -2, -1}

def N : Set ℤ := {x : ℤ | (x + 2) * (x - 3) < 0}

theorem intersection_is_singleton : M ∩ N = {-1} :=
by
  sorry

end ProofProblem

end intersection_is_singleton_l343_343491


namespace pencils_pens_total_l343_343150

theorem pencils_pens_total (x : ℕ) (h1 : 4 * x + 1 = 7 * (5 * x - 1)) : 4 * x + 5 * x = 45 :=
by
  sorry

end pencils_pens_total_l343_343150


namespace option_c_opposite_l343_343728

theorem option_c_opposite :
  let a := -|(-0.01)|
  let b := -(- (1 / 100))
  a = -0.01 ∧ b = 0.01 ∧ a = -b :=
by
  sorry

end option_c_opposite_l343_343728


namespace number_of_customers_who_did_not_want_tires_change_l343_343733

noncomputable def total_cars_in_shop : Nat := 4 + 6
noncomputable def tires_per_car : Nat := 4
noncomputable def total_tires_bought : Nat := total_cars_in_shop * tires_per_car
noncomputable def half_tires_left : Nat := 2 * (tires_per_car / 2)
noncomputable def total_half_tires_left : Nat := 2 * half_tires_left
noncomputable def tires_left_after_half : Nat := 20
noncomputable def tires_left_after_half_customers : Nat := tires_left_after_half - total_half_tires_left
noncomputable def customers_who_did_not_change_tires : Nat := tires_left_after_half_customers / tires_per_car

theorem number_of_customers_who_did_not_want_tires_change : 
  customers_who_did_not_change_tires = 4 :=
by
  sorry 

end number_of_customers_who_did_not_want_tires_change_l343_343733


namespace inequality_1_inequality_2_l343_343257

variable (x : ℝ)

theorem inequality_1 (h1 : x - 1 > 2) : x > 3 :=
sorry

theorem inequality_2 (h2 : -4x > 8) : x < -2 :=
sorry

end inequality_1_inequality_2_l343_343257


namespace number_of_factors_multiples_252_l343_343948

-- Definition of m
def m := 2^12 * 3^16 * 7^9

-- Definition of the problem statement
theorem number_of_factors_multiples_252 : 
  let m := 2^12 * 3^16 * 7^9 in
  let factors_of_m_multiples_of_252 := 
    (12 - 2 + 1) * (16 - 2 + 1) * (9 - 1 + 1) in
  factors_of_m_multiples_of_252 = 1485 := 
by
  -- Sorry, proof omitted
  sorry

end number_of_factors_multiples_252_l343_343948


namespace ratio_of_roots_l343_343802

theorem ratio_of_roots (a b c x₁ x₂ : ℝ) (h₁ : a ≠ 0) (h₂ : c ≠ 0) (h₃ : a * x₁^2 + b * x₁ + c = 0) (h₄ : a * x₂^2 + b * x₂ + c = 0) (h₅ : x₁ = 4 * x₂) : (b^2) / (a * c) = 25 / 4 :=
by
  sorry

end ratio_of_roots_l343_343802


namespace remainder_sum_arithmetic_sequence_is_zero_l343_343316

theorem remainder_sum_arithmetic_sequence_is_zero :
  let a := 3
      d := 6
      n := 46
      sequence := List.range n
      sum_sequence := sequence.sum (λ k => a + k * d)
  in sum_sequence % 6 = 0 := by
  let a := 3
  let d := 6
  let n := 46
  let sequence := List.range n
  let sum_sequence := sequence.sum (λ k => a + k * d)
  sorry

end remainder_sum_arithmetic_sequence_is_zero_l343_343316


namespace seventh_person_donation_l343_343009

/--
Given a group of 7 members where:
- 6 members each donate 10 forints,
- The seventh member donates 3 forints more than the average donation of all 7 members,
prove that the seventh member donates 13.5 forints.
-/
theorem seventh_person_donation :
  let total = 70 in
  let sixth_sum = 60 in
  let avg_donation = total / 7 in
  let seventh_donation := avg_donation + 3 in
  seventh_donation = 13.5 :=
by
  let total := 70
  let sixth_sum := 60
  let avg_donation := total / 7
  let seventh_donation := avg_donation + 3
  rw [total, sixth_sum]
  simp [avg_donation, seventh_donation]
  sorry

end seventh_person_donation_l343_343009


namespace num_solutions_to_equation_l343_343508

noncomputable def numerator (x : ℕ) : ℕ := ∏ i in (finset.range 51).filter (λ i , i > 0), (x - i)

noncomputable def denominator (x : ℕ) : ℕ := ∏ i in (finset.range 51).filter (λ i, i > 0), (x - i^2)

theorem num_solutions_to_equation : 
  (∃ n, nat.card { x : ℕ | 1 ≤ x ∧ x ≤ 50 ∧ numerator x = 0 ∧ denominator x ≠ 0 } = n) ∧ n = 43 := 
by
  sorry

end num_solutions_to_equation_l343_343508


namespace hexagon_arithmetic_sum_l343_343999

theorem hexagon_arithmetic_sum (a n : ℝ) (h : 6 * a + 15 * n = 720) : 2 * a + 5 * n = 240 :=
by
  sorry

end hexagon_arithmetic_sum_l343_343999


namespace convex_polygons_count_l343_343829

def binomial (n k : ℕ) : ℕ := if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def count_convex_polygons_with_two_acute_angles (m n : ℕ) : ℕ :=
  if 4 < m ∧ m < n then
    (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1))
  else 0

theorem convex_polygons_count (m n : ℕ) (h : 4 < m ∧ m < n) :
  count_convex_polygons_with_two_acute_angles m n = 
  (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1)) :=
by sorry

end convex_polygons_count_l343_343829


namespace total_games_won_l343_343894

-- Define the number of games won by the Chicago Bulls
def bulls_games : ℕ := 70

-- Define the number of games won by the Miami Heat
def heat_games : ℕ := bulls_games + 5

-- Define the total number of games won by both the Bulls and the Heat
def total_games : ℕ := bulls_games + heat_games

-- The theorem stating that the total number of games won by both teams is 145
theorem total_games_won : total_games = 145 := by
  -- Proof is omitted
  sorry

end total_games_won_l343_343894


namespace max_min_fraction_l343_343795

-- Given condition
def circle_condition (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 1 = 0

-- Problem statement
theorem max_min_fraction (x y : ℝ) (h : circle_condition x y) :
  -20 / 21 ≤ y / (x - 4) ∧ y / (x - 4) ≤ 0 :=
sorry

end max_min_fraction_l343_343795


namespace standard_eqn_parabola_l343_343803

-- Define the conditions as properties
def is_vertex_origin (x y : ℝ) : Prop := x = 0 ∧ y = 0

def is_axis_symmetry_coord (axis_symmetry : Boolean) : Prop := axis_symmetry = true

def distance_vertex_directrix (d : ℝ) : Prop := d = 4

def hyperbola_eq (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

def directrix_perpendicular (left_vertex : ℝ) : Prop := left_vertex = -3

-- State the main theorem
theorem standard_eqn_parabola (x y : ℝ) (d : ℝ) :
  is_vertex_origin x y →
  is_axis_symmetry_coord true →
  distance_vertex_directrix d →
  hyperbola_eq x y →
  directrix_perpendicular (-3) →
  (2 * 3 = 6 ∧ y^2 = 12 * x) :=
by
  intros
  sorry

end standard_eqn_parabola_l343_343803


namespace parallel_lines_a_l343_343107

theorem parallel_lines_a (a : ℝ) (l1 : ∀ x y : ℝ, x + a * y - 1 = 0) (l2 : ∀ x y : ℝ, (a - 1) * x + 2 * y - 3 = 0) : 
  (∃ k1 k2 : ℝ, (-1 / a = -((a - 1) / 2)) → ∃ a = -1) :=
by
  sorry

end parallel_lines_a_l343_343107


namespace most_likely_outcome_of_five_children_l343_343038

/-- 
Given that each child born is independently a boy or a girl with probability 1/2, 
prove that the most likely outcome among the following is having 4 children of one gender and 1 child of the other gender:
1. All 5 are boys
2. All 5 are girls
3. 3 are girls and 2 are boys
4. 4 are of one gender and 1 is of the other gender
5. All outcomes are equally likely
--/
theorem most_likely_outcome_of_five_children :
  (∀ (A B C D E : ℚ), (A = 1/32) → (B = 1/32) → (C = 5/16) → (D = 5/16) → (E = 1/5) → 
  (2 / 5) < (5 / 16) ∧ (4 / 32) = (2 / 16) ∧ D is_the_most_likely_outcome) :=
begin
  sorry
end

end most_likely_outcome_of_five_children_l343_343038


namespace line_fixed_point_l343_343699

theorem line_fixed_point (l : ℝ) (t : ℝ) : ∃ p : ℝ × ℝ, p = (-2, 3) :=
by
  use (-2, 3)
  sorry

end line_fixed_point_l343_343699


namespace milk_per_cow_per_day_l343_343344

-- Define the conditions
def num_cows := 52
def weekly_milk_production := 364000 -- ounces

-- State the theorem
theorem milk_per_cow_per_day :
  (weekly_milk_production / 7 / num_cows) = 1000 := 
by
  -- Here we would include the proof, so we use sorry as placeholder
  sorry

end milk_per_cow_per_day_l343_343344


namespace intersection_A_B_l343_343830

variable {α : Type*} [LinearOrder α]

def A (x : α) : Prop := 2 < x ∧ x < 4
def B (x : α) : Prop := (x - 1) * (x - 3) < 0

theorem intersection_A_B (x : α) : (A x ∧ B x) ↔ (2 < x ∧ x < 3) := by
  sorry

end intersection_A_B_l343_343830


namespace never_consecutive_again_l343_343573

theorem never_consecutive_again (n : ℕ) (seq : ℕ → ℕ) :
  (∀ k, seq k = seq 0 + k) → 
  ∀ seq' : ℕ → ℕ,
    (∀ i j, i < j → seq' (2*i) = seq i + seq (j) ∧ seq' (2*i+1) = seq i - seq (j)) →
    ¬ (∀ k, seq' k = seq' 0 + k) :=
by
  sorry

end never_consecutive_again_l343_343573


namespace correct_mean_of_dataset_l343_343346

theorem correct_mean_of_dataset
    (n : ℕ)
    (mean_initial : ℚ)
    (errors : list (ℚ × ℚ))
    (correct_mean : ℚ)
    (h_n_val : n = 70)
    (h_mean_initial_val : mean_initial = 350)
    (h_errors_val : errors = [(215.5, 195.5), (-30, 30), (720.8, 670.8), (-95.4, -45.4), (124.2, 114.2)])
    (h_correct_mean_val : correct_mean = 349.57) :
    let total_error := list.sum (errors.map (λ (e : ℚ × ℚ), e.1 - e.2)),
        original_sum := mean_initial * n,
        corrected_sum := original_sum + total_error,
        mean_computed := corrected_sum / n
    in mean_computed = correct_mean :=
by sorry

end correct_mean_of_dataset_l343_343346


namespace ratio_markus_age_son_age_l343_343230

variable (M S G : ℕ)

theorem ratio_markus_age_son_age (h1 : G = 20) (h2 : S = 2 * G) (h3 : M + S + G = 140) : M / S = 2 := by
  sorry

end ratio_markus_age_son_age_l343_343230


namespace four_digit_integer_solution_l343_343266

theorem four_digit_integer_solution : ∃ (a b c d : ℕ), 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
  a + b + c + d = 18 ∧
  b + c = 10 ∧
  a - d = 2 ∧
  (1000 * a + 100 * b + 10 * c + d) % 9 = 0 ∧
  1000 * a + 100 * b + 10 * c + d = 5643 :=
begin
  -- proof steps go here
  sorry,
end

end four_digit_integer_solution_l343_343266


namespace exists_indices_non_decreasing_l343_343981

theorem exists_indices_non_decreasing
    (a b c : ℕ → ℕ) :
    ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
  sorry

end exists_indices_non_decreasing_l343_343981


namespace Jane_indisposed_days_l343_343190

-- Definitions based on conditions
def John_completion_days := 18
def Jane_completion_days := 12
def total_task_days := 10.8
def work_per_day_by_john := 1 / John_completion_days
def work_per_day_by_jane := 1 / Jane_completion_days
def work_per_day_together := work_per_day_by_john + work_per_day_by_jane

-- Equivalent proof problem
theorem Jane_indisposed_days : 
  ∃ (x : ℝ), 
    (10.8 - x) * work_per_day_together + x * work_per_day_by_john = 1 ∧
    x = 6 := 
by 
  sorry

end Jane_indisposed_days_l343_343190


namespace find_a_geometric_sequence_l343_343213

theorem find_a_geometric_sequence (a : ℤ) (T : ℕ → ℤ) (b : ℕ → ℤ) :
  (∀ n, T n = 3 ^ n + a) →
  b 1 = T 1 →
  (∀ n, n ≥ 2 → b n = T n - T (n - 1)) →
  (∀ n, n ≥ 2 → (∃ r, r * b n = b (n - 1))) →
  a = -1 :=
by
  sorry

end find_a_geometric_sequence_l343_343213


namespace saree_sale_price_l343_343762

def originalPrice : ℝ := 1000
def discount1 : ℝ := 0.30
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10
def discount4 : ℝ := 0.05

def applyDiscount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

theorem saree_sale_price :
  let price1 := applyDiscount originalPrice discount1 in
  let price2 := applyDiscount price1 discount2 in
  let price3 := applyDiscount price2 discount3 in
  let price4 := applyDiscount price3 discount4 in
  Real.floor price4 = 509 :=
by
  sorry

end saree_sale_price_l343_343762


namespace partition_set_l343_343558

variable S : Set ℝ
variable [Finite S]
variable h : ∀ s ∈ S, 0 < s
variable n : ℕ
variable hs : S.card = n

def f (A : Set ℝ) : ℝ := ∑ s in A, s

theorem partition_set (hS : ∀ u ∈ S, u = u ∧ ∀ v ∈ S, u ≠ v → u < v) :
  ∃ (P : Fin n → Set (ℝ → ℝ)), (∀ i, (∃ T, P i = { f A | A ⊆ S ∧ A ≠ ∅ ∧ T.1 < f A ∧ f A ≤ T.2 }))
  ∧ (∀ i, ∃ min max, (min ∈ ∅ → min = 0) ∧
                     max ∈ ∅ → max = 0 ∧
                     (max > min ∧ max/min < 2)) :=
sorry

end partition_set_l343_343558


namespace infinitely_many_integers_not_prime_l343_343987

def is_not_prime_sum (a : ℕ) : Prop := ∀ n : ℕ, ¬ Prime (n^4 + a)

theorem infinitely_many_integers_not_prime (k : ℕ) (hk : k > 1) : 
  is_not_prime_sum (4 * k^4) :=
by 
  intros n
  have h : (n^4 + 4 * k^4) = (n^2 + 2 * n * k + 2 * k^2) * (n^2 - 2 * n * k + 2 * k^2),
  { sorry },
  sorry

end infinitely_many_integers_not_prime_l343_343987


namespace range_of_m_l343_343040

theorem range_of_m (m : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ (π / 2) → cos θ ^ 2 + 2 * m * sin θ - 2 * m - 2 < 0) →
  m > -1 / 2 :=
by
  -- proof goes here
  sorry

end range_of_m_l343_343040


namespace relational_bc_l343_343945

variables (a b c : Line)
-- This condition states that lines a and b are parallel
axiom parallel_ab : a ∥ b
-- This condition states that line a intersects line c
axiom intersects_ac : a ∩ c ≠ ∅

theorem relational_bc : (b ∩ c ≠ ∅ ∨ ∃ p₁ p₂, (p₁ ∈ b ∧ p₁ ∉ c) ∧ (p₂ ∈ c ∧ p₂ ∉ b)) :=
sorry

end relational_bc_l343_343945


namespace sum_of_lucky_numbers_eq_3720_l343_343808

theorem sum_of_lucky_numbers_eq_3720 :
  let lucky_number (n : ℕ) := (n + 6) ∣ (n ^ 3 + 1996)
  in ∑ n in Finset.filter lucky_number (Finset.range 2000), n = 3720 :=
by
  sorry

end sum_of_lucky_numbers_eq_3720_l343_343808


namespace systematic_sampling_example_l343_343293

theorem systematic_sampling_example :
  ∃ seq : List Nat, seq = [5, 10, 15, 20] ∧ 
  (∀ i, i < seq.length → seq.nth i - seq.nth (i - 1) = 5) :=
by
  sorry

end systematic_sampling_example_l343_343293


namespace sum_eq_twenty_x_l343_343123

variable {R : Type*} [CommRing R] (x y z : R)

theorem sum_eq_twenty_x (h1 : y = 3 * x) (h2 : z = 3 * y) : 2 * x + 3 * y + z = 20 * x := by
  sorry

end sum_eq_twenty_x_l343_343123


namespace find_theta_l343_343226

noncomputable def roots (θ : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℂ), (x₁^2 - 3 * (Real.sin θ) * x₁ + (Real.sin θ)^2 + 1 = 0)
              ∧ (x₂^2 - 3 * (Real.sin θ) * x₂ + (Real.sin θ)^2 + 1 = 0)
              ∧ (|x₁| + |x₂| = 2)
              ∧ θ ∈ Set.Ico 0 Real.pi

theorem find_theta : ∀ (θ : ℝ), roots θ → θ = 0 :=
by
  sorry

end find_theta_l343_343226


namespace total_jokes_after_eight_days_l343_343974

def jokes_counted (start_jokes : ℕ) (n : ℕ) : ℕ :=
  -- Sum of initial jokes until the nth day by doubling each day
  start_jokes * (2 ^ n - 1)

theorem total_jokes_after_eight_days (jessy_jokes : ℕ) (alan_jokes : ℕ) (tom_jokes : ℕ) (emily_jokes : ℕ)
  (total_days : ℕ) (days_per_week : ℕ) :
  total_days = 5 → days_per_week = 8 →
  jessy_jokes = 11 → alan_jokes = 7 → tom_jokes = 5 → emily_jokes = 3 →
  (jokes_counted jessy_jokes (days_per_week - total_days) +
   jokes_counted alan_jokes (days_per_week - total_days) +
   jokes_counted tom_jokes (days_per_week - total_days) +
   jokes_counted emily_jokes (days_per_week - total_days)) = 806 :=
by
  intros
  sorry

end total_jokes_after_eight_days_l343_343974


namespace crude_oil_temperature_change_l343_343244

-- Define the function f(x) = x^2 - 7x + 15
def f (x : ℝ) : ℝ := x^2 - 7 * x + 15

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 * x - 7

-- Define the statement to prove: (f'(4) = 1) and (f'(4) > 0 implies rising)
theorem crude_oil_temperature_change : (f' 4 = 1) ∧ (f' 4 > 0 → "rising" = "rising") :=
by {
  rw [(show f' 4 = 1, by sorry), (show f' 4 > 0, by sorry)],
  split,
  { exact sorry },
  { intro h,
    exact sorry },
}

end crude_oil_temperature_change_l343_343244


namespace first_digit_base4_389_is_one_l343_343403

-- Define a function to get the base-4 representation of a number as a list of digits
def base4_representation (n : ℕ) : List ℕ :=
  if n == 0 then [0] else
  let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n == 0 then acc else
    loop (n / 4) ((n % 4) :: acc)
  loop n []

-- Define the condition as a definition in Lean
def first_digit_of_base4_389 : ℕ :=
  match base4_representation 389 with
  | []      => 0 -- This case should not happen for 389
  | (d::_) => d

-- State the theorem
theorem first_digit_base4_389_is_one : first_digit_of_base4_389 = 1 :=
by
  sorry

end first_digit_base4_389_is_one_l343_343403


namespace percentage_70_to_79_correct_l343_343136

-- Define the frequencies as constants
def students_90_to_100 : ℕ := 5
def students_80_to_89 : ℕ := 7
def students_70_to_79 : ℕ := 8
def students_60_to_69 : ℕ := 4
def students_below_60 : ℕ := 3

-- Define the total number of students
def total_students : ℕ :=
  students_90_to_100 + students_80_to_89 + students_70_to_79 + students_60_to_69 + students_below_60

-- Define the percentage calculation
noncomputable def percentage_70_to_79 : ℚ :=
  (students_70_to_79 : ℚ) / total_students * 100

-- Theorem stating the percentage of students who scored between 70% and 79%
theorem percentage_70_to_79_correct :
  percentage_70_to_79 ≈ 29.63 :=
  sorry

end percentage_70_to_79_correct_l343_343136


namespace second_reduction_percentage_l343_343717

theorem second_reduction_percentage (P : ℝ) : 
  let first_reduction_price := 0.90 * P in
  let second_reduction_price := 0.774 * P in
  (first_reduction_price - second_reduction_price) / first_reduction_price * 100 = 14 :=
by
  sorry

end second_reduction_percentage_l343_343717


namespace distinct_values_g_l343_343554

def floor (r : ℝ) : ℤ := Int.floor r

def g (x : ℝ) : ℝ :=
  ∑ k in Finset.range 8, (floor (2 * (k+1) * x) - 2 * (k+1) * floor x)

theorem distinct_values_g (x : ℝ) (hx : x ≥ 0) : 
  (Finset.image g {y | y ≥ 0}).card = 32 := 
sorry

end distinct_values_g_l343_343554


namespace max_value_g_l343_343623

def f (x a : ℝ) := 2 * x^2 - 2 * a * x + 3

def g (a : ℝ) := 
if a < -2 then 2 * a + 5 else 
if -2 ≤ a ∧ a ≤ 2 then 3 - (a^2) / 2 else 
5 - 2 * a

theorem max_value_g {a : ℝ} : 
(∀ x, x ∈ Set.Icc (-1:ℝ) 1 → f x a ≥ g a) ∧ (∀ a, g_a := g a → g_a ≤ 3) :=
sorry

end max_value_g_l343_343623


namespace find_natural_pair_l343_343705

theorem find_natural_pair :
  ∃ (x y : ℕ), 1984 * x - 1983 * y = 1985 ∧ x = 27764 ∧ y = 27777 :=
begin
  use [27764, 27777],
  split,
  apply nat.well_founded_lt.fix,
  sorry
end

end find_natural_pair_l343_343705


namespace eval_expression_l343_343779

theorem eval_expression : (125 ^ (1/3) * 81 ^ (-1/4) * 32 ^ (1/5) = (10/3)) :=
by
  have h1 : 125 = 5^3 := by norm_num
  have h2 : 81 = 3^4 := by norm_num
  have h3 : 32 = 2^5 := by norm_num
  sorry

end eval_expression_l343_343779


namespace cylinder_height_to_diameter_ratio_l343_343372

theorem cylinder_height_to_diameter_ratio
  (r h : ℝ)
  (inscribed_sphere : h = 2 * r)
  (cylinder_volume : π * r^2 * h = 3 * (4/3) * π * r^3) :
  (h / (2 * r)) = 2 :=
by
  sorry

end cylinder_height_to_diameter_ratio_l343_343372


namespace find_a_l343_343562

open Set

variable (A : Set ℝ)
variable (B : Set ℝ)
variable (a : ℝ)

theorem find_a (h1 : A = {-1, 1, 3}) 
               (h2 : B = {a + 2, a^2 + 4}) 
               (h3 : A ∩ B = {1}) : 
               a = -1 := by
  sorry

end find_a_l343_343562


namespace base_five_product_l343_343655

namespace BaseFiveMultiplication

def to_base_five (n : ℕ) : list ℕ := 
  -- Dummy implementation for illustration
  [n % 5, (n / 5) % 5, (n / 25) % 5, (n / 125) % 5].reverse

def from_base_five (digits : list ℕ) : ℕ := 
  digits.foldl (λ acc d => acc * 5 + d) 0

def base_five_mult (a b : list ℕ) : list ℕ :=
  to_base_five (from_base_five(a) * from_base_five(b))

theorem base_five_product : base_five_mult (to_base_five 132) (to_base_five 12) = to_base_five 2114 :=
  sorry

end BaseFiveMultiplication

end base_five_product_l343_343655


namespace log_sum_eq_neg_one_l343_343961

noncomputable def tangent_intersection_x (n : ℕ) : ℝ := n / (n + 1)

theorem log_sum_eq_neg_one :
  (Finset.range 2015).sum (λ n, Real.logb 2016 (tangent_intersection_x (n + 1))) = -1 :=
begin
  sorry,
end

end log_sum_eq_neg_one_l343_343961


namespace sum_of_products_l343_343885

variable (a b c : ℝ)

theorem sum_of_products (h1 : a^2 + b^2 + c^2 = 250) (h2 : a + b + c = 16) : 
  ab + bc + ca = 3 :=
sorry

end sum_of_products_l343_343885


namespace even_function_value_l343_343480

theorem even_function_value (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_def : ∀ x : ℝ, 0 < x → f x = 2^x + 1) :
  f (-2) = 5 :=
  sorry

end even_function_value_l343_343480


namespace bamboo_probability_l343_343482

-- Define the lengths of the bamboo poles
def bamboo_poles : List ℝ := [2.5, 2.6, 2.7, 2.8, 2.9]

-- Function to calculate combinations
def choose (n k : ℕ) : ℕ := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Function to determine if a pair has the exact length difference
def has_length_difference (a b : ℝ) (d : ℝ) : Bool :=
  abs (a - b) = d

-- Define the problem statement as a theorem
theorem bamboo_probability : 
  let pairs := List.product bamboo_poles bamboo_poles
  let valid_pairs := pairs.filter (λ (x : ℝ × ℝ), has_length_difference x.1 x.2 0.3)
  let total_pairs := (choose 5 2)
  in (valid_pairs.length / total_pairs.to_float) = 0.2 :=
sorry

end bamboo_probability_l343_343482


namespace area_triangle_RYZ_l343_343255

/-- Given a square with area 225, WXYZ,
and points P, Q, R defined as follows:
 - P is on side WZ,
 - Q is the midpoint of WP,
 - R is the midpoint of YP,
 - Quadrilateral WQRP has area 45.

Prove that the area of triangle RYZ is 39.375--/
theorem area_triangle_RYZ (a : ℝ) (h₁ : a^2 = 225)
  (P Q R W X Y Z : ℝ × ℝ)
  (hP : P.1 = W.1 ∧ P.2 = Z.2)
  (hQ : 2 * Q = (W.1, W.2) + P.1)
  (hR : 2 * R = (Y.1, Y.2) + P.1)
  (hWQRP : area_of_quadrilateral (W.1, W.2) Q R P = 45) :
  area_of_triangle R Y Z = 39.375 := by sorry

end area_triangle_RYZ_l343_343255


namespace geometric_proof_l343_343531

variables {A B C D K M : Type*}
variables [IsSquare A B C D]
variables [IsPointOnSide K BC]
variables [IsPointOnSide M CD]
variables [IsAngleBisector AM KAD]

theorem geometric_proof : ∃ AK DM BK, AK = DM + BK :=
by
  sorry

end geometric_proof_l343_343531


namespace candidate_C_is_inverse_proportion_l343_343662

/--
Check whether the given function is an inverse proportion function.
-/
def is_inverse_proportion (f : ℝ → ℝ) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/--
The candidate functions are defined as follows:
A: y = x / 3
B: y = 3 / (x + 1)
C: xy = 3
D: y = 3x
-/
def candidate_A (x : ℝ) : ℝ := x / 3
def candidate_B (x : ℝ) : ℝ := 3 / (x + 1)
def candidate_C (x : ℝ) : ℝ := 3 / x
def candidate_D (x : ℝ) : ℝ := 3 * x

theorem candidate_C_is_inverse_proportion : is_inverse_proportion candidate_C :=
  sorry

end candidate_C_is_inverse_proportion_l343_343662


namespace num_integer_distance_pairs_5x5_grid_l343_343262

-- Define the problem conditions
def grid_size : ℕ := 5

-- Define a function to calculate the number of pairs of vertices with integer distances
noncomputable def count_integer_distance_pairs (n : ℕ) : ℕ := sorry

-- The theorem to prove
theorem num_integer_distance_pairs_5x5_grid : count_integer_distance_pairs grid_size = 108 :=
by
  sorry

end num_integer_distance_pairs_5x5_grid_l343_343262


namespace geometric_series_first_term_l343_343384

theorem geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (hr : r = 1 / 4)
  (hS : S = 50)
  (hSum : S = a / (1 - r)) :
  a = 75 / 2 :=
by
  have h1 : 1 - r = 3 / 4 := by
    rw [hr]
    norm_num
  have h2 : a / (3 / 4) = 50 := by
    rw [← h1]
    exact hSum.symm.trans hS
  have : a = 50 * (3 / 4) := by
    field_simp
    exact h2
  norm_num at this
  exact this

end geometric_series_first_term_l343_343384


namespace jog_to_gym_time_l343_343513

-- Definitions from the conditions.
def jog_at_constant_speed (d t : ℝ) := d / t
def time_to_jog_4_mi : ℝ := 30
def distance_to_park : ℝ := 4
def distance_to_gym : ℝ := 2

-- Problem statement.
theorem jog_to_gym_time : 
  let t := distance_to_gym * (time_to_jog_4_mi / distance_to_park) in
  t = 15 :=
by
  sorry

end jog_to_gym_time_l343_343513


namespace measure_minor_arc_l343_343159

theorem measure_minor_arc (MCB_angle : ℝ) (hMCB : MCB_angle = 60) (tangent_MC_at_C : tangent M C) 
  (B_on_circle : point_on_circle B R) : 
  minor_arc_MB_angle = 60 :=
by sorry

end measure_minor_arc_l343_343159


namespace derivative_at_α_l343_343128

variable (α : Real)

def f (x : Real) : Real := α ^ 2 - Real.cos x

theorem derivative_at_α : (Real.hasDerivAt (λ x, α ^ 2 - Real.cos x) (Real.sin α) α) :=
by sorry

end derivative_at_α_l343_343128


namespace problem_statement_l343_343069

theorem problem_statement (a b : ℝ) (h1 : 2^a > 2^b) (h2 : 2^b > 1) : (1 / 2)^a < (1 / 2)^b :=
  sorry

end problem_statement_l343_343069


namespace tan_mul_eq_rat_l343_343832

theorem tan_mul_eq_rat 
  (α β : ℝ)
  (h1 : cos(α - β) ^ 2 - cos(α + β) ^ 2 = 0.5)
  (h2 : (1 + cos (2 * α)) * (1 + cos (2 * β)) = 1 / 3) :
  tan α * tan β = 3 / 2 := 
sorry

end tan_mul_eq_rat_l343_343832


namespace divisibility_by_37_criterion_l343_343242

theorem divisibility_by_37_criterion (A : ℤ) (segments : List ℤ) :
  (A = List.foldl (λ acc x, acc + x) 0 segments) →
  (∀ segment ∈ segments, segment < 10^3) →
  (37 ∣ List.foldl (λ acc x, acc + x) 0 segments ↔ 37 ∣ A) :=
by sorry

end divisibility_by_37_criterion_l343_343242


namespace drawing_probability_consecutive_order_l343_343684

theorem drawing_probability_consecutive_order :
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  probability = 1 / 665280 :=
by
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  sorry

end drawing_probability_consecutive_order_l343_343684


namespace train_platform_crossing_time_l343_343720

-- Definitions
def train_length : ℕ := 1200  -- length of the train in meters
def tree_cross_time : ℕ := 120  -- time to cross a tree in seconds
def platform_length : ℕ := 700  -- length of the platform in meters

-- Prove the time required to pass the platform
theorem train_platform_crossing_time :
  let speed := train_length / tree_cross_time,
      total_distance := train_length + platform_length,
      crossing_time := total_distance / speed in
  crossing_time = 190 := by
sorry

end train_platform_crossing_time_l343_343720


namespace tank_filled_at_10pm_l343_343179

def start_time := 13 -- 1 pm in 24-hour format
def first_hour_rain := 2 -- inches
def next_four_hours_rate := 1 -- inches per hour
def next_four_hours_duration := 4 -- hours
def remaining_day_rate := 3 -- inches per hour
def tank_height := 18 -- inches

theorem tank_filled_at_10pm :
  let accumulated_rain_by_6pm := first_hour_rain + next_four_hours_rate * next_four_hours_duration in
  let remaining_rain_needed := tank_height - accumulated_rain_by_6pm in
  let remaining_hours_to_fill := remaining_rain_needed / remaining_day_rate in
  (start_time + 1 + next_four_hours_duration + remaining_hours_to_fill) = 22 := -- 10 pm in 24-hour format
by
  sorry

end tank_filled_at_10pm_l343_343179


namespace cannot_be_square_of_difference_formula_l343_343321

theorem cannot_be_square_of_difference_formula (x y c d a b m n : ℝ) :
  ¬ ((m - n) * (-m + n) = (x^2 - y^2) ∨ 
       (m - n) * (-m + n) = (c^2 - d^2) ∨ 
       (m - n) * (-m + n) = (a^2 - b^2)) :=
by sorry

end cannot_be_square_of_difference_formula_l343_343321


namespace find_angle_PCA_l343_343337

-- Definitions of our geometric entities and assumptions
variables (A B C P Q K L : Type)

-- Assumptions
axioms
  (h1 : ∃ (circle ω : Type), is_circumscribed_around_triangle ω A B C)
  (h2 : is_tangent_line_intersecting ω C P)
  (h3 : is_on_ray_beyond Q P C ∧ distance P C = distance Q C)
  (h4 : intersects_again B Q K ω)
  (h5 : is_on_smaller_arc L B K ω ∧ angle L A K = angle C Q B)
  (h6 : angle A L Q = 60)

-- Target
theorem find_angle_PCA : angle P C A = 30 :=
sorry

end find_angle_PCA_l343_343337


namespace find_deaf_students_l343_343371

-- Definitions based on conditions
variables (B D : ℕ)
axiom deaf_students_triple_blind_students : D = 3 * B
axiom total_students : D + B = 240

-- Proof statement
theorem find_deaf_students (h1 : D = 3 * B) (h2 : D + B = 240) : D = 180 :=
sorry

end find_deaf_students_l343_343371


namespace problem_l343_343410

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.exp (2 * x) - a * x

theorem problem (a x : ℝ) (h₁ : 0 < a) (h₂ : 0 < x) :
  f x a ≥ 2 * a + a * real.log (2 / a) :=
sorry

end problem_l343_343410


namespace number_of_valid_sequences_is_65_l343_343877

/-- A sequence is valid if it starts with 0, ends with 0, has no two consecutive 0's, and no three consecutive 1's --/
def valid_sequence (s : List ℕ) : Prop :=
  ∃ l, s.length = l ∧ l = 19 ∧ s.head = 0 ∧ s.last = 0 ∧
  ∀ i, i < l - 1 → s.nth i ≠ 0 ∨ s.nth (i + 1) ≠ 0 ∧
  ∀ i, i < l - 2 → ¬ (s.nth i = 1 ∧ s.nth (i + 1) = 1 ∧ s.nth (i + 2) = 1)

/-- A function f(n) to count valid sequences of length n --/
def f : ℕ → ℕ
| 3 := 1
| 4 := 1
| 5 := 1
| 6 := 2
| n := f (n-4) + 2 * f (n-5) + f (n-6)

/-- Prove that the number of valid sequences of length 19 is equal to 65 --/
theorem number_of_valid_sequences_is_65 : f 19 = 65 := 
  sorry

end number_of_valid_sequences_is_65_l343_343877


namespace composites_coprime_lcm_l343_343706

theorem composites_coprime_lcm (a b : ℕ) (ha : ¬ is_prime a ∧ (∃ p, p ∣ a ∧ p ≠ 1 ∧ p ≠ a)) (hb : ¬ is_prime b ∧ (∃ q, q ∣ b ∧ q ≠ 1 ∧ q ≠ b)) (h_gcd : Nat.gcd a b = 1) (h_lcm : Nat.lcm a b = 120) : 
a = 8 ∧ b = 15 := 
by
  sorry

end composites_coprime_lcm_l343_343706


namespace quadrilateral_not_exist_l343_343310

theorem quadrilateral_not_exist (a b c d : ℕ) (h₀ : set.insert a (set.insert b (set.insert c (set.singleton d))) = {1, 3, 4, 10}) :
  ¬∃ (p q r s : ℕ), p + q + r + s = 18 ∧ p + q > r ∧ p + r > q ∧ q + r > p :=
by
  sorry

end quadrilateral_not_exist_l343_343310


namespace sequence_inequality_l343_343997

theorem sequence_inequality {n : ℕ} (h : n > 0) :
  let a : ℕ → ℝ := λ k, 
    Nat.recOn k (1 / 2) (λ k ak, ak + (1 / n) * ak^2) in 
  1 - (1 / n) < a n ∧ a n < 1 :=
by
  sorry

end sequence_inequality_l343_343997


namespace intersection_cardinality_l343_343677

def setX : Set ℕ := { n | 1 ≤ n ∧ n ≤ 12 }
def setY : Set ℕ := { n | 0 ≤ n ∧ n ≤ 20 }

theorem intersection_cardinality :
  (setX ∩ setY).toFinset.card = 12 := by
  sorry

end intersection_cardinality_l343_343677


namespace no_quadrilateral_exists_l343_343311

theorem no_quadrilateral_exists (a b c d : ℝ) (h1 : a = 1) (h2 : b = 3) (h3 : c = 4) (h4 : d = 10) : 
  ¬(a + b > d ∧ a + d > b ∧ b + d > a ∧ a + c > d ∧ b + c > a ∧ c + d > b ∧ a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a) :=
by {
  rw [h1, h2, h3, h4],
  intro h,
  have h' : ¬(1 + 3 > 10 ∧ 1 + 10 > 3 ∧ 3 + 10 > 1 ∧ 1 + 4 > 10 ∧ 3 + 4 > 1 ∧ 4 + 10 > 3 ∧ 1 + 3 + 4 > 10 ∧ 1 + 3 + 10 > 4 ∧ 1 + 4 + 10 > 3 ∧ 3 + 4 + 10 > 1),
  { simp,
    split; intro h;
    linarith },
  exact h' h,
}

end no_quadrilateral_exists_l343_343311


namespace monthly_interest_payment_is_correct_l343_343736

-- Definitions given the conditions
def Principal : ℝ := 30800
def AnnualRate : ℝ := 0.09
def Time : ℝ := 1 -- in years
def AnnualInterest := Principal * AnnualRate * Time
def MonthlyInterest := AnnualInterest / 12

-- The theorem we need to prove: The monthly interest payment is $231
theorem monthly_interest_payment_is_correct : MonthlyInterest = 231 := by
  sorry

end monthly_interest_payment_is_correct_l343_343736


namespace bridget_block_collection_l343_343743

-- Defining the number of groups and blocks per group.
def num_groups : ℕ := 82
def blocks_per_group : ℕ := 10

-- Defining the total number of blocks calculation.
def total_blocks : ℕ := num_groups * blocks_per_group

-- Theorem stating the total number of blocks is 820.
theorem bridget_block_collection : total_blocks = 820 :=
  by
  sorry

end bridget_block_collection_l343_343743


namespace simplify_expression_l343_343595

theorem simplify_expression :
  (1 / ((3 / (Real.sqrt 5 + 2)) + (4 / (Real.sqrt 6 - 2)))) =
  ((3 * Real.sqrt 5 + 2 * Real.sqrt 6 + 2) / 29) :=
  sorry

end simplify_expression_l343_343595


namespace second_and_fourth_rows_identical_l343_343904

def count_occurrences (lst : List ℕ) (a : ℕ) (i : ℕ) : ℕ :=
  (lst.take (i + 1)).count a

def fill_next_row (current_row : List ℕ) : List ℕ :=
  current_row.enum.map (λ ⟨i, a⟩ => count_occurrences current_row a i)

theorem second_and_fourth_rows_identical (first_row : List ℕ) :
  let second_row := fill_next_row first_row 
  let third_row := fill_next_row second_row 
  let fourth_row := fill_next_row third_row 
  second_row = fourth_row :=
by
  sorry

end second_and_fourth_rows_identical_l343_343904


namespace complex_addition_l343_343882

theorem complex_addition (a b : ℝ) (i : ℂ) (h1 : i = complex.I) (h2 : 2 - 2 * (i^3) = a + b * i) : a + b = 4 := 
by 
  sorry

end complex_addition_l343_343882


namespace sausages_fried_l343_343198

def num_eggs : ℕ := 6
def time_per_sausage : ℕ := 5
def time_per_egg : ℕ := 4
def total_time : ℕ := 39
def time_per_sauteurs (S : ℕ) : ℕ := S * time_per_sausage

theorem sausages_fried (S : ℕ) (h : num_eggs * time_per_egg + S * time_per_sausage = total_time) : S = 3 :=
by
  sorry

end sausages_fried_l343_343198


namespace sum_of_coefficients_of_expanded_polynomial_l343_343122

theorem sum_of_coefficients_of_expanded_polynomial :
  let b := (2 : ℤ) * (1 : ℤ) + 3 in
  let p := b ^ 6 in
  p = 15625 :=
by
  let b := (2 : ℤ) * (1 : ℤ) + 3
  let p := b ^ 6
  exact congr rfl rfl sorry

end sum_of_coefficients_of_expanded_polynomial_l343_343122


namespace general_formula_sum_of_b_l343_343913

namespace ArithmeticSequence

-- Define the arithmetic sequence {a_n}, sum of the first n terms S_n,
-- given S_4 = 10, a_2 = 2
def a (n : ℕ) : ℕ := n

noncomputable def S (n : ℕ) : ℕ := n * (2 * 1 + (n - 1) * 1) / 2

-- Define the sequence {b_n} where b_n = 1 / (a_n * a_(n+1))
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Define the sum of the first n terms of sequence {b_n}, T_n
noncomputable def T (n : ℕ) : ℚ := (finset.range n).sum (λ k, b (k + 1))

-- Prove the general term a_n = n
theorem general_formula (n : ℕ) : a n = n := by
  sorry

-- Prove the sum formula for the first n terms of sequence {b_n}
theorem sum_of_b (n : ℕ) : T n = n / (n + 1) := by
  sorry

end ArithmeticSequence

end general_formula_sum_of_b_l343_343913


namespace prove_standard_equation_find_line_pq_l343_343047

noncomputable def focal_distance : ℝ := 6 / 2

def ellipse_equation (a b : ℝ) : Prop := 
  a > b ∧ b > 0 ∧ (a^2 = b^2 + focal_distance^2)

def is_point_on_ellipse (a b x y : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

variable (a b : ℝ)
variable (x1 y1 x2 y2 : ℝ)
variable (F1 F2 O : ℝ × ℝ)

def foci : Prop := 
  F1 = (-focal_distance, 0) ∧ F2 = (focal_distance, 0) ∧ O = (0, 0)

def chord_conditions (A B F1 F2: ℝ × ℝ) : Prop :=
  let P := 12 * real.sqrt 2 in
  A ≠ B ∧
  (fst A) ^ 2 / a^2 + (snd A) ^ 2 / b^2 = 1 ∧
  (fst B) ^ 2 / a^2 + (snd B) ^ 2 / b^2 = 1 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = P → 4 * a = P

def midpoint_condition (M: ℝ × ℝ) : Prop := 
  M = (2, 1) ∧
  x1 ≠ x2 ∧
  (x1 + x2) = 4 ∧
  (y1 + y2) = 2

theorem prove_standard_equation :
  foci F1 F2 O → chord_conditions (a := 3 * real.sqrt 2) F1 F2 → ellipse_equation a b →
  (a, b) = (3 * √2, √9) :=
sorry

theorem find_line_pq (M: ℝ × ℝ) :
  midpoint_condition M → is_point_on_ellipse a b x1 y1 → is_point_on_ellipse a b x2 y2 →
  x1 ≠ x2 →
  ∃ (m : ℝ), (m = -1) ∧ (m = (y1 - y2) / (x1 - x2)) ∧
  (x + y - 3 = 0) :=
sorry

end prove_standard_equation_find_line_pq_l343_343047


namespace arctan_sum_l343_343793

theorem arctan_sum : 
  Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/8) = Real.pi / 4 := 
by 
  sorry

end arctan_sum_l343_343793


namespace proof_problem_l343_343493

-- Definitions for the given conditions in the problem
def equations (a x y : ℝ) : Prop :=
(x + 5 * y = 4 - a) ∧ (x - y = 3 * a)

-- The conclusions from the problem
def conclusion1 (a x y : ℝ) : Prop :=
a = 1 → x + y = 4 - a

def conclusion2 (a x y : ℝ) : Prop :=
a = -2 → x = -y

def conclusion3 (a x y : ℝ) : Prop :=
2 * x + 7 * y = 6

def conclusion4 (a x y : ℝ) : Prop :=
x ≤ 1 → y > 4 / 7

-- The main theorem to be proven
theorem proof_problem (a x y : ℝ) :
  equations a x y →
  (¬ conclusion1 a x y ∨ ¬ conclusion2 a x y ∨ ¬ conclusion3 a x y ∨ ¬ conclusion4 a x y) →
  (∃ n : ℕ, n = 2 ∧ ((conclusion1 a x y ∨ conclusion2 a x y ∨ conclusion3 a x y ∨ conclusion4 a x y) → false)) :=
by {
  sorry
}

end proof_problem_l343_343493


namespace Q_zero_eq_one_l343_343640

-- Define the polynomial Q(x) with the given conditions
noncomputable def Q : ℚ[X] := Classical.some (exists_unique (λ (Q : ℚ[X]), 
  degree Q = 4 ∧ leading_coeff Q = 1 ∧ (Q.eval (sqrt 2 + sqrt 3) = 0)))

-- State the theorem
theorem Q_zero_eq_one :
  Q.eval 0 = 1 :=
sorry

end Q_zero_eq_one_l343_343640


namespace arithmetic_sequence_sum_formula_sum_of_reciprocals_formula_l343_343463

-- Define the arithmetic sequence and sum
def arithmetic_sequence_sum (a₁ aₙ n : ℕ) : ℕ := (n * (a₁ + aₙ)) / 2

theorem arithmetic_sequence_sum_formula (a₁ aₙ n : ℕ) :
  ∀ (d : ℕ), S_n = (n * (a₁ + aₙ)) / 2 := by
  sorry

-- Define the sum of reciprocals of sums
def sum_of_reciprocals_of_sequence (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1 / ((k+1) * (k+2) / 2))

theorem sum_of_reciprocals_formula (n : ℕ) :
  sum_of_reciprocals_of_sequence n = (2 * n) / (n + 1) := by
  sorry

end arithmetic_sequence_sum_formula_sum_of_reciprocals_formula_l343_343463


namespace mike_walked_approx_thirty_seven_hundred_miles_l343_343232

noncomputable def pedometer_flips : ℕ := 60
noncomputable def pedometer_end_of_year_reading : ℕ := 75000
noncomputable def steps_per_mile : ℕ := 1675
noncomputable def steps_per_flip : ℕ := 100000

noncomputable def total_steps (flips : ℕ) (end_reading : ℕ) (steps_per_flip : ℕ) : ℕ :=
  flips * steps_per_flip + end_reading

noncomputable def miles_walked (flips : ℕ) (end_reading : ℕ) (steps_per_flip : ℕ) (steps_per_mile : ℕ) : ℝ :=
  total_steps flips end_reading steps_per_flip / steps_per_mile

theorem mike_walked_approx_thirty_seven_hundred_miles :
  miles_walked pedometer_flips pedometer_end_of_year_reading steps_per_flip steps_per_mile ≈ 3700 := sorry

end mike_walked_approx_thirty_seven_hundred_miles_l343_343232


namespace infinitely_many_m_such_that_m_minus_f_eq_1989_l343_343204

/-- 
  Let m be a positive integer and define f(m) to be the number of factors of 2 in m! 
  (that is, the greatest positive integer k such that 2^k divides m!). 
  There exist infinitely many positive integers m such that m - f(m) = 1989.
-/
theorem infinitely_many_m_such_that_m_minus_f_eq_1989 :
  ∃ᶠ m in at_top, ∃ (f : ℕ → ℕ), f m = (∑ k in Finset.range (Nat.log2 m + 1), m / 2^k) ∧ m - f m = 1989 :=
sorry

end infinitely_many_m_such_that_m_minus_f_eq_1989_l343_343204


namespace solution_set_A_solution_set_B_subset_A_l343_343086

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem solution_set_A :
  {x : ℝ | f x > 6} = {x : ℝ | x < -1 ∨ x > 2} :=
sorry

theorem solution_set_B_subset_A {a : ℝ} :
  (∀ x, f x > |a-1| → x < -1 ∨ x > 2) → a ≤ -5 ∨ a ≥ 7 :=
sorry

end solution_set_A_solution_set_B_subset_A_l343_343086


namespace stock_value_order_l343_343578

-- Define the initial investment and yearly changes
def initialInvestment : Float := 100
def firstYearChangeA : Float := 1.30
def firstYearChangeB : Float := 0.70
def firstYearChangeG : Float := 1.10
def firstYearChangeD : Float := 1.00 -- unchanged

def secondYearChangeA : Float := 0.90
def secondYearChangeB : Float := 1.35
def secondYearChangeG : Float := 1.05
def secondYearChangeD : Float := 1.10

-- Calculate the final values after two years
def finalValueA : Float := initialInvestment * firstYearChangeA * secondYearChangeA
def finalValueB : Float := initialInvestment * firstYearChangeB * secondYearChangeB
def finalValueG : Float := initialInvestment * firstYearChangeG * secondYearChangeG
def finalValueD : Float := initialInvestment * firstYearChangeD * secondYearChangeD

-- Theorem statement - Prove that the final order of the values is B < D < G < A
theorem stock_value_order : finalValueB < finalValueD ∧ finalValueD < finalValueG ∧ finalValueG < finalValueA := by
  sorry

end stock_value_order_l343_343578


namespace standard_equation_of_ellipse_l343_343464

theorem standard_equation_of_ellipse :
  ∀ (m n : ℝ), 
    (m > 0 ∧ n > 0) →
    (∃ (c : ℝ), c^2 = m^2 - n^2 ∧ c = 2) →
    (∃ (e : ℝ), e = c / m ∧ e = 1 / 2) →
    (m = 4 ∧ n = 2 * Real.sqrt 3) →
    (∀ x y : ℝ, (x^2 / 16 + y^2 / 12 = 1)) :=
by
  intros m n hmn hc he hm_eq hn_eq
  sorry

end standard_equation_of_ellipse_l343_343464


namespace number_of_classes_l343_343345

theorem number_of_classes
  (s : ℕ)    -- s: number of students in each class
  (bpm : ℕ) -- bpm: books per month per student
  (months : ℕ) -- months: number of months in a year
  (total_books : ℕ) -- total_books: total books read by the entire student body in a year
  (H1 : bpm = 5)
  (H2 : months = 12)
  (H3 : total_books = 60)
  (H4 : total_books = s * bpm * months)
: s = 1 :=
by
  sorry

end number_of_classes_l343_343345


namespace minimum_value_of_f_range_of_a_for_monotonic_F_l343_343863

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.log x + 1 / x

-- Theorem: Prove the minimum value of f(x) on (0, +∞) is 1
theorem minimum_value_of_f : ∃ x, (0 < x) ∧ f x = 1 := 
sorry

-- Define the function F(x)
def F (x a : ℝ) : ℝ := Real.log x + 1 / x + a * x

-- Theorem: Prove the range of a for F(x) to be monotonic on [2, +∞)
theorem range_of_a_for_monotonic_F : 
  (∀ a : ℝ, (∀ x : ℝ, 2 ≤ x → (∀ y : ℝ, 2 ≤ y → (x ≤ y → F x a ≤ F y a))) ↔ (a ∈ Set.Iic (-1 / 4) ∪ Set.Ici 0)) :=
sorry

end minimum_value_of_f_range_of_a_for_monotonic_F_l343_343863


namespace red_marbles_given_l343_343405

theorem red_marbles_given (violet_marbles total_marbles : ℕ) (h1 : violet_marbles = 64) (h2 : total_marbles = 78) :
  total_marbles - violet_marbles = 14 :=
by
  rw [h1, h2]
  norm_num

end red_marbles_given_l343_343405


namespace rational_third_vertex_l343_343376

theorem rational_third_vertex (x1 y1 x2 y2 : ℚ) (x3 y3 : ℚ) :
  (∃ x3 y3 : ℚ, true) ↔ (∀ X, (X = 90 ∨ ∃ r : ℚ, tan X = r)) :=
sorry

end rational_third_vertex_l343_343376


namespace greatest_divisor_of_arithmetic_sum_l343_343315

theorem greatest_divisor_of_arithmetic_sum (a d : ℕ) (h : ∃ k : ℕ, d = k^2) :
  ∃ k : ℕ, ∀ a d, S = a + (a + d) + (a + 2d) + ... + (a + 14d) implies 15 ∣ (15a + 105d) :=
by sorry

end greatest_divisor_of_arithmetic_sum_l343_343315


namespace area_of_square_from_circles_l343_343811

theorem area_of_square_from_circles : 
  (radius : ℝ) (h_radius : radius = 7) 
  (h_touch : ∀ (c1 c2 : ℝ), c1 = radius ∧ c2 = radius) :  
  ∃ (area : ℝ), area = 28 ^ 2 := 
begin
  sorry
end

end area_of_square_from_circles_l343_343811


namespace saree_original_price_l343_343286

theorem saree_original_price :
  ∃ P : ℝ, (0.95 * 0.88 * P = 334.4) ∧ (P = 400) :=
by
  sorry

end saree_original_price_l343_343286


namespace inverse_negation_false_l343_343660

def P (x : ℝ) : Prop := x^2 = 1 → x = 1

theorem inverse_negation_false : ¬(¬P (x : ℝ)) := 
by sorry

end inverse_negation_false_l343_343660


namespace unique_solution_triple_l343_343797

theorem unique_solution_triple {a b c : ℝ} (h₀ : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h₁ : a^2 + b^2 + c^2 = 3) (h₂ : (a + b + c) * (a^2 * b + b^2 * c + c^2 * a) = 9) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ c = 1 ∧ b = 1) ∨ (b = 1 ∧ a = 1 ∧ c = 1) ∨ (b = 1 ∧ c = 1 ∧ a = 1) ∨ (c = 1 ∧ a = 1 ∧ b = 1) ∨ (c = 1 ∧ b = 1 ∧ a = 1) :=
sorry

end unique_solution_triple_l343_343797


namespace find_analytical_expression_and_a_l343_343862

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

theorem find_analytical_expression_and_a :
  (A > 0) → (ω > 0) → (0 < φ ∧ φ < π / 2) →
  (∀ x, ∃ k : ℤ, f (x + k * π / 2) = f (x)) →
  (∃ A, ∀ x, A * sin (ω * x + φ) ≤ 2) →
  ((∀ x, f (x - π / 6) = -f (-x + π / 6)) ∨ f 0 = sqrt 3 ∨ (∃ x, 2 * x + φ = k * π + π / 2)) →
  (∀ x, f x = 2 * sin (2 * x + π / 3)) ∧
  (∀ (A : ℝ), (0 < A ∧ A < π) → (f A = sqrt 3) →
  (c = 3 ∧ S = 3 * sqrt 3) →
  (a ^ 2 = ((4 * sqrt 3) ^ 2 + 3 ^ 2 - 2 * (4 * sqrt 3) * 3 * cos (π / 6))) → a = sqrt 21) :=
  sorry

end find_analytical_expression_and_a_l343_343862


namespace find_S6_l343_343456

   variable {a : ℕ → ℝ}  -- geometric sequence
   variable {S : ℕ → ℝ}  -- sum of the first n terms

   noncomputable def geometric_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
     a 1 * (q ^ n - 1) / (q - 1)

   axioms (h1 : a 1 + a 3 = 5)
          (h2 : S 4 = 15)
          (h3 : ∀ n, S n = ∑ k in finset.range n, a (k + 1))

   theorem find_S6 : S 6 = 63 := by
     sorry
   
end find_S6_l343_343456


namespace track_circumference_l343_343680

noncomputable def circumference_of_track (A_dist B_dist : ℝ) (first_meet second_meet_from_A : ℝ) : ℝ :=
  let x := (2 * B_dist * A_dist) / ((A_dist + B_dist) - (second_meet_from_A / first_meet) * (A_dist - B_dist)) in
  2 * x

theorem track_circumference (A B : ℝ) (first_meet := 150) (second_meet_from_A := 90) :
  B * 2 ≠ A + B ∧ B * 2 ≠ A - B ∧ first_meet > 0 ∧ second_meet_from_A > 0 → 
  circumference_of_track (A - 150) (B) first_meet second_meet_from_A = 720 :=
by
  intros
  sorry

end track_circumference_l343_343680


namespace james_run_time_l343_343926

theorem james_run_time
  (B : ℕ) -- number of bags
  (O : ℕ) -- ounces per bag
  (C : ℕ) -- calories per ounce
  (E : ℕ) -- excess calories
  (R : ℕ) -- calories burned per minute
  (total_calories : ℕ := B * O * C) -- total calories consumed
  (calories_to_burn : ℕ := total_calories - E) -- calories to be burned during run
  (T : ℕ := calories_to_burn / R) -- time of run in minutes
  : T = 40 := by
  -- Given problems' parameters
  have B_val : B = 3 := by rfl
  have O_val : O = 2 := by rfl
  have C_val : C = 150 := by rfl
  have E_val : E = 420 := by rfl
  have R_val : R = 12 := by rfl

  -- Plugging in the parameters
  rw [B_val, O_val, C_val, E_val, R_val] at total_calories calories_to_burn T

  -- Calculation
  dsimp [total_calories, calories_to_burn, T]

  -- Verifying the final result
  norm_num

  -- Final Assertion
  exact rfl

end james_run_time_l343_343926


namespace sin_cos_sum_l343_343291

theorem sin_cos_sum (α x y r : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : r = Real.sqrt 5)
    (h4 : ∀ θ, x = r * Real.cos θ) (h5 : ∀ θ, y = r * Real.sin θ) : 
    Real.sin α + Real.cos α = (- 1 / Real.sqrt 5) + (2 / Real.sqrt 5) :=
by
  sorry

end sin_cos_sum_l343_343291


namespace sequence_problem_l343_343868

/-- Given sequence a_n with specific values for a_2 and a_4 and the assumption that a_(n+1)
    is a geometric sequence, prove that a_6 equals 63. -/
theorem sequence_problem 
  {a : ℕ → ℝ} 
  (h1 : a 2 = 3) 
  (h2 : a 4 = 15) 
  (h3 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n ∧ q^2 = 4) : 
  a 6 = 63 := by
  sorry

end sequence_problem_l343_343868


namespace forest_enclosure_l343_343163

theorem forest_enclosure (n : ℕ) (h : Fin n → ℝ)
  (h_bound : ∀ i, 10 ≤ h i ∧ h i ≤ 50)
  (dist_bound : ∀ i j, abs (h i - h j) ≥ dist i j) :
  ∃ (fence_length : ℝ), fence_length = 80 := 
begin
  sorry
end

end forest_enclosure_l343_343163


namespace mall_b_more_cost_effective_at_300_cost_equal_at_400_l343_343010

-- Definitions based on problem's conditions
def cost_mall_a (x : ℝ) : ℝ :=
  if x <= 200 then x else 200 + 0.85 * (x - 200)

def cost_mall_b (x : ℝ) : ℝ :=
  if x <= 100 then x else 100 + 0.9 * (x - 100)

-- Problem 1: When the total shopping amount is 300, prove that Mall B is more cost-effective than Mall A
theorem mall_b_more_cost_effective_at_300 : cost_mall_b 300 < cost_mall_a 300 := 
by sorry

-- Problem 2: At what amount of shopping (greater than 100) will the cost be the same at both Mall A and Mall B?
theorem cost_equal_at_400 : cost_mall_a 400 = cost_mall_b 400 :=
by sorry

end mall_b_more_cost_effective_at_300_cost_equal_at_400_l343_343010


namespace gas_volume_at_25_degrees_l343_343436

theorem gas_volume_at_25_degrees :
  (∀ (T V : ℕ), (T = 40 → V = 30) →
  (∀ (k : ℕ), T = 40 - 5 * k → V = 30 - 6 * k) → 
  (25 = 40 - 5 * 3) → 
  (V = 30 - 6 * 3) → 
  V = 12) := 
by
  sorry

end gas_volume_at_25_degrees_l343_343436


namespace same_zero_l343_343129

noncomputable def f (a x : ℝ) := Real.logBase 2 (x + a)
noncomputable def g (a x : ℝ) := x^2 - (a + 1) * x - 4 * (a + 5)

theorem same_zero (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ g a x = 0) ↔ a = 5 ∨ a = -2 :=
by
  sorry

end same_zero_l343_343129


namespace digit_Phi_l343_343651

theorem digit_Phi (Phi : ℕ) (h1 : 220 / Phi = 40 + 3 * Phi) : Phi = 4 :=
by
  sorry

end digit_Phi_l343_343651


namespace sum_of_possible_values_sum_of_all_possible_values_l343_343892

theorem sum_of_possible_values (x : ℝ) 
  (h : |x - 12| = 100) : x = 112 ∨ x = -88 :=
begin
  have h_cases : x - 12 = 100 ∨ x - 12 = -100,
  { exact (abs_eq (x - 12) 100).mp h },
  cases h_cases,
  { left, linarith },
  { right, linarith },
end

theorem sum_of_all_possible_values : 
  (∑ x in {112, -88}.sum : ℝ) = 24 :=
calc
  (∑ x in {112, -88}, x) = 112 + -88 : by simp
...                       = 24       : by norm_num

end sum_of_possible_values_sum_of_all_possible_values_l343_343892


namespace eleventh_number_in_list_l343_343725

-- Define a predicate to check if the sum of the digits of a number is 12
def digit_sum_eq_12 (n : ℕ) : Prop := 
  (n.digits 10).sum = 12

-- Define a list of all positive integers whose digits sum to 12 and are in increasing order
def list_of_digit_sum_12 : List ℕ := 
  List.filter digit_sum_eq_12 (List.range 1000) -- considering numbers from 0 to 999 for instance

theorem eleventh_number_in_list : 
  list_of_digit_sum_12.nth 10 = some 147 :=
by
  sorry

end eleventh_number_in_list_l343_343725


namespace area_of_triangle_value_of_a_l343_343523

variable {A B C a b c : ℝ}
variable (cos_half_A tan_B : ℝ)
variable (dot_product_AB_AC : ℝ)

-- Given conditions
def cos_half_angle_condition : Prop :=
  cos_half_A = 2 * real.sqrt 5 / 5

def dot_product_condition : Prop :=
  dot_product_AB_AC = 15

def tan_B_condition : Prop :=
  tan_B = 2

-- Prove that the area of the triangle ABC is 10
theorem area_of_triangle (hc : cos_half_angle_condition) (hd : dot_product_condition) :
  let bc := (b * c) in (1 / 2) * bc * real.sqrt(1 - (3 / 5) ^ 2) = 10 :=
  sorry

-- Prove that the value of 'a' is 2√5 given additional condition of tan B
theorem value_of_a (hc : cos_half_angle_condition) (hd : dot_product_condition) (ht : tan_B_condition) :
  a = 2 * real.sqrt 5 :=
  sorry

end area_of_triangle_value_of_a_l343_343523


namespace common_ratio_geometric_series_l343_343426

theorem common_ratio_geometric_series :
  let a := 2 / 3
  let b := 4 / 9
  let c := 8 / 27
  (b / a = 2 / 3) ∧ (c / b = 2 / 3) → 
  ∃ r : ℚ, r = 2 / 3 ∧ ∀ n : ℕ, (a * r^n) = (a * (2 / 3)^n) :=
by
  sorry

end common_ratio_geometric_series_l343_343426


namespace ratio_of_triangle_ABC_area_of_triangle_ABC_l343_343064

-- Define the context of the problem
variables (A B C R : ℝ)
variables (a b c : ℝ)
variables (vector_m vector_n : ℝ × ℝ)

-- Define the conditions
def triangle_ABC_conditions : Prop :=
  vector_m = (a - b, 1) ∧
  vector_n = (a - c, 2) ∧
  (vector_m.2 * vector_n.1 = vector_n.2 * vector_m.1) ∧
  A = 120 * (real.pi / 180) -- convert degrees to radians

-- Define the ratio question and its answer
def ratio_a_b_c (a b c : ℝ) : Prop :=
  (a : b : c) = (7 : 5 : 3)

-- Define the area question and its answer
def area_ABC (a b c A R : ℝ) : Prop :=
  let S := 0.5 * b * c * real.sin A in
  R = 14 ∧ S = 45 * real.sqrt 3

-- Theorem to prove the ratio
theorem ratio_of_triangle_ABC (h : triangle_ABC_conditions A B C a b c vector_m vector_n) :
  ratio_a_b_c a b c :=
sorry

-- Theorem to prove the area
theorem area_of_triangle_ABC (h : triangle_ABC_conditions A B C a b c vector_m vector_n) (hR : R = 14) :
  area_ABC a b c A R :=
sorry

end ratio_of_triangle_ABC_area_of_triangle_ABC_l343_343064


namespace ellipse_equation_range_of_slopes_l343_343081

noncomputable def problem (x y a b c: ℝ) : Prop :=
  (a > b ∧ b > 0) ∧
  (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (c = a / (√2) ∧ 2 * c * (2 * a * b / (√(4 * a^2 + b^2))) = (4 * (√2)) / 3)

theorem ellipse_equation (a b c: ℝ) (h: problem 1 1 a b c) :
  (a = √2 ∧ b = 1) → (∀ x y, x^2 / 2 + y^2 = 1) :=
by
  sorry

noncomputable def line_parallel_to_l (m x y x₀ y₀ k₁ k₂: ℝ) : Prop :=
  (y = 2 * x + m) ∧
  (x₁ + x₂ = - (8 / 9) * m ∧ x₁ * x₂ = (2 * m^2 - 2) / 9) ∧
  (x₀ = - (4 / 9) * m ∧ y₀ = 2 * x₀ + m) ∧
  (k₁ + k₂ = (8 * m^2) / (81 - 16 * m^2))

theorem range_of_slopes (m x₀ y₀ k₁ k₂: ℝ) (h: line_parallel_to_l m 1 1 x₀ y₀ k₁ k₂):
  m^2 < 9 → (k₁ + k₂ ∈ (Set.Ioo 0 (1 / 0)) ∨ k₁ + k₂ ∈ (Set.Ioo (-1 / 0) (-8 / 7))) :=
by
  sorry

end ellipse_equation_range_of_slopes_l343_343081


namespace distinct_real_pairs_l343_343421

theorem distinct_real_pairs (x y : ℝ) (h1 : x ≠ y) (h2 : x^100 - y^100 = 2^99 * (x - y)) (h3 : x^200 - y^200 = 2^199 * (x - y)) :
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
sorry

end distinct_real_pairs_l343_343421


namespace planes_of_symmetry_cube_planes_of_symmetry_tetrahedron_l343_343113

theorem planes_of_symmetry_cube : 
    number_of_planes_of_symmetry geometric_shape.cube = 9 := 
sorry

theorem planes_of_symmetry_tetrahedron : 
    number_of_planes_of_symmetry geometric_shape.regular_tetrahedron = 6 := 
sorry

end planes_of_symmetry_cube_planes_of_symmetry_tetrahedron_l343_343113


namespace evaluate_expression_l343_343774

theorem evaluate_expression : (125^(1/3 : ℝ)) * (81^(-1/4 : ℝ)) * (32^(1/5 : ℝ)) = (10 / 3 : ℝ) :=
by
  sorry

end evaluate_expression_l343_343774


namespace distinct_arrangements_TOOL_l343_343114

/-- The word "TOOL" consists of four letters where "O" is repeated twice. 
Prove that the number of distinct arrangements of the letters in the word is 12. -/
theorem distinct_arrangements_TOOL : 
  let total_letters := 4
  let repeated_O := 2
  (Nat.factorial total_letters / Nat.factorial repeated_O) = 12 := 
by
  sorry

end distinct_arrangements_TOOL_l343_343114


namespace total_value_correct_l343_343545

def card := Nat
def points := Nat

def rare_value : points := 10
def nonrare_value : points := 3
def holo_value : points := 15
def first_edition_value : points := 8

-- Conditions
def jenny_initial_cards : card := 6
def jenny_rare_cards : card := jenny_initial_cards / 2
def jenny_nonrare_cards : card := jenny_initial_cards - jenny_rare_cards
def jenny_additional_holo_cards : card := 2
def jenny_additional_first_edition_cards : card := 2

def orlando_cards : card := jenny_initial_cards + 2
def orlando_rare_cards : card := (orlando_cards * 40) / 100
def orlando_nonrare_cards : card := orlando_cards - orlando_rare_cards

def richard_cards : card := orlando_cards * 3
def richard_rare_cards : card := (richard_cards * 25) / 100
def richard_nonrare_cards : card := richard_cards - richard_rare_cards

-- Values of the cards
def jenny_initial_value : points :=
  (jenny_rare_cards * rare_value) + (jenny_nonrare_cards * nonrare_value)
def jenny_additional_value : points :=
  (jenny_additional_holo_cards * holo_value) + (jenny_additional_first_edition_cards * first_edition_value)
def jenny_total_value : points := jenny_initial_value + jenny_additional_value

def orlando_value : points :=
  (orlando_rare_cards * rare_value) + (orlando_nonrare_cards * nonrare_value)

def richard_value : points :=
  (richard_rare_cards * rare_value) + (richard_nonrare_cards * nonrare_value)

def total_value : points := jenny_total_value + orlando_value + richard_value

-- Theorem statement
theorem total_value_correct : total_value = 244 := by
  sorry

end total_value_correct_l343_343545


namespace count_n_satisfying_condition_l343_343031

theorem count_n_satisfying_condition : 
  (Finset.filter (λ n : ℤ, 25 < n^2 ∧ n^2 < 144) (Finset.range 145)).card = 12 :=
by
  sorry

end count_n_satisfying_condition_l343_343031


namespace sequence_terms_interval_count_l343_343158

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + d * (n - 1)

theorem sequence_terms_interval_count :
  ∀ (a: ℕ → ℝ) (a1 d : ℝ),
    a 4 = 70 →
    a 21 = -100 →
    is_arithmetic_sequence a a1 d →
    ( ∃ (n1 n2 : ℕ), 9.2 ≤ n1 ∧ n1 ≤ 12.8 ∧ 9.2 ≤ n2 ∧ n2 ≤ 12.8 ∧ 
      (n1 = 10 ∨ n1 = 11 ∨ n1 = 12) ∧ 
      (n2 = 10 ∨ n2 = 11 ∨ n2 = 12) ∧ 
      n1 ≠ n2 ) →
    ( ∃ k : ℕ, k = 3 ) :=
by
  sorry

end sequence_terms_interval_count_l343_343158


namespace squares_parallel_and_equality_l343_343575

variables {A B C B_a C_a A_b C_b A_c B_c B_v B_u C_u : Type} [normed_field A] [normed_space ℝ A]

theorem squares_parallel_and_equality 
  (ABC_acute : ∀ {A B C : Type}, acute_angle_triangle A B C)
  (squares_on_sides : ∀ {A B C B_a C_a A_b C_b A_c B_c B_v B_u C_u : Type}
    (BC : A → A → A)
    (CA : A → A → A)
    (AB : A → A → A)
    (BB_aC_aC : A → A → A → A)
    (CC_bA_bA : A → A → A → A)
    (AA_cB_cB : A → A → A → A)
    (B_cB_vB_uB_a : A → A → A → A → A)
    (C_aC_uC_vC_b : A → A → A → A → A), true)
  : ∀ {B_u C_u BC : A → A → A},
  parallel B_u C_u BC ∧
  (distance B_u C_u = 4 * distance BC) :=
sorry

end squares_parallel_and_equality_l343_343575


namespace union_eq_self_l343_343942

variables {S T X : Set}

noncomputable theory

theorem union_eq_self (S T : Set) (h1 : S.nonempty) (h2 : T.nonempty) (h3 : S ∉ T) (h4 : T ∉ S) (h5 : X = S ∩ T) :
  S ∪ X = S :=
by
  sorry

end union_eq_self_l343_343942


namespace f_sum_inequality_l343_343486

def f (x : ℝ) : ℝ := 2^x / (2^x + 1)

theorem f_sum_inequality (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) : 
  f (a - b) + f (b - c) + f (c - a) ≤ 3 / 2 := 
by 
  sorry

end f_sum_inequality_l343_343486


namespace sequence_match_l343_343459

-- Define the sequence sum S_n
def S_n (n : ℕ) : ℕ := 2^(n + 1) - 1

-- Define the sequence a_n based on the problem statement
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 2^n

-- The theorem stating that sequence a_n satisfies the given sum condition S_n
theorem sequence_match (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end sequence_match_l343_343459


namespace find_A_range_sinB_sinC_l343_343893

-- Given conditions in a triangle
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h_cos_eq : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)

-- Angle A verification
theorem find_A (h_sum_angles : A + B + C = Real.pi) : A = Real.pi / 3 :=
  sorry

-- Range of sin B + sin C
theorem range_sinB_sinC (h_sum_angles : A + B + C = Real.pi) :
  (0 < B ∧ B < 2 * Real.pi / 3) →
  Real.sin B + Real.sin C ∈ Set.Ioo (Real.sqrt 3 / 2) (Real.sqrt 3) :=
  sorry

end find_A_range_sinB_sinC_l343_343893


namespace distance_between_points_l343_343687

def main : IO Unit :=
  IO.println "Start Lean 4 Proof Problem"

open Real

-- Definitions
def speed_car := 80
def speed_train := 50
def min_time := 7
def equidistant_distance := 861

-- The proof goal
theorem distance_between_points :
  ∀ {A B C : Type} [metric_space A] [metric_space B] [metric_space C],
  ∃ S : ℝ,
  (S = equidistant_distance) ∧
  (S = distance S (S / 2 * sin (pi / 3))) ∧
  (S = distance (S / 2 * sin (pi / 3)) S) :=
sorry  -- proof to be completed in Lean

end distance_between_points_l343_343687


namespace simplify_expression_l343_343596

variable {R : Type} [Field R] (x y : R)

theorem simplify_expression :
  (x + 2 * y)⁻² * (x⁻¹ + 2 * y⁻¹) = (y + 2 * x) * x⁻¹ * y⁻¹ * (x + 2 * y)⁻² :=
by { sorry }

end simplify_expression_l343_343596


namespace round_39_492_to_nearest_tenth_l343_343587

def nearestTenth (x : ℝ) : ℝ :=
  let tenths := (x * 10) % 10
  let hundredths := (x * 100) % 10
  if hundredths >= 5 then ((x * 10).trunc + 1) / 10 else (x * 10).trunc / 10

theorem round_39_492_to_nearest_tenth : nearestTenth 39.492 = 39.5 :=
by
  sorry

end round_39_492_to_nearest_tenth_l343_343587


namespace number_of_dimes_l343_343589

theorem number_of_dimes (x : ℕ) (h1 : 10 * x + 25 * x + 50 * x = 2040) : x = 24 :=
by {
  -- The proof will go here if you need to fill it out.
  sorry
}

end number_of_dimes_l343_343589


namespace evaluate_expression_l343_343787

theorem evaluate_expression:
  (125 = 5^3) ∧ (81 = 3^4) ∧ (32 = 2^5) → 
  125^(1/3) * 81^(-1/4) * 32^(1/5) = 10 / 3 := by
  sorry

end evaluate_expression_l343_343787


namespace exists_x_y_for_n_l343_343582

theorem exists_x_y_for_n (n : ℕ) (hn : n > 1) :
  ∃ x y : ℕ, (x ≤ y) ∧ (x > 0) ∧ (y > 0) ∧ (∑ k in Finset.range (y - x + 1) + x, 1 / (k * (k + 1)) = 1 / n) :=
by
  sorry

end exists_x_y_for_n_l343_343582


namespace wire_length_ratio_l343_343742

def bonnie_wire_length : ℕ := 12 * 8
def roark_prism_volume : ℕ := 2^3
def bonnie_prism_volume : ℕ := 8^3
def number_of_roark_prisms : ℕ := bonnie_prism_volume / roark_prism_volume
def roark_wire_per_prism : ℕ := 12 * 2
def total_roark_wire_length : ℕ := number_of_roark_prisms * roark_wire_per_prism

theorem wire_length_ratio : (96 : ℚ) / (1536 : ℚ) = 1 / 16 :=
by
  sorry

end wire_length_ratio_l343_343742


namespace simple_interest_rate_l343_343672

theorem simple_interest_rate (P : ℝ) (r : ℝ) (T : ℝ) (SI : ℝ)
  (h1 : SI = P / 5)
  (h2 : T = 10)
  (h3 : SI = (P * r * T) / 100) :
  r = 2 :=
by
  sorry

end simple_interest_rate_l343_343672


namespace largest_multiple_of_45_l343_343616

theorem largest_multiple_of_45 (m : ℕ) 
  (h₁ : m % 45 = 0) 
  (h₂ : ∀ d : ℕ, d ∈ m.digits 10 → d = 8 ∨ d = 0) : 
  m / 45 = 197530 := 
sorry

end largest_multiple_of_45_l343_343616


namespace sequence_problem_l343_343051

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def a (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 1 / 2
  else if n % 2 = 1 then
    let k := (n - 1) / 2 in
    if k % 2 = 0 then f (a k) else f (f (a k))
  else 
    let k := n / 2 in
    if k % 2 = 0 then a k else f (a k)

theorem sequence_problem (h : a 20 = a 18) : a 2016 + a 2017 = Real.sqrt 2 - 1 / 2 := by
  sorry

end sequence_problem_l343_343051


namespace tank_filled_at_10pm_l343_343180

def start_time := 13 -- 1 pm in 24-hour format
def first_hour_rain := 2 -- inches
def next_four_hours_rate := 1 -- inches per hour
def next_four_hours_duration := 4 -- hours
def remaining_day_rate := 3 -- inches per hour
def tank_height := 18 -- inches

theorem tank_filled_at_10pm :
  let accumulated_rain_by_6pm := first_hour_rain + next_four_hours_rate * next_four_hours_duration in
  let remaining_rain_needed := tank_height - accumulated_rain_by_6pm in
  let remaining_hours_to_fill := remaining_rain_needed / remaining_day_rate in
  (start_time + 1 + next_four_hours_duration + remaining_hours_to_fill) = 22 := -- 10 pm in 24-hour format
by
  sorry

end tank_filled_at_10pm_l343_343180


namespace cube_root_two_not_rational_comb_l343_343978

theorem cube_root_two_not_rational_comb (p q r : ℚ) :
  (∛(2 : ℚ) ≠ p + q * real.sqrt r) :=
by sorry

end cube_root_two_not_rational_comb_l343_343978


namespace min_containers_needed_l343_343338

theorem min_containers_needed 
  (total_boxes1 : ℕ) 
  (weight_box1 : ℕ) 
  (total_boxes2 : ℕ) 
  (weight_box2 : ℕ) 
  (weight_limit : ℕ) :
  total_boxes1 = 90000 →
  weight_box1 = 3300 →
  total_boxes2 = 5000 →
  weight_box2 = 200 →
  weight_limit = 100000 →
  (total_boxes1 * weight_box1 + total_boxes2 * weight_box2 + weight_limit - 1) / weight_limit = 3000 :=
by
  sorry

end min_containers_needed_l343_343338


namespace num_solutions_to_equation_l343_343509

noncomputable def numerator (x : ℕ) : ℕ := ∏ i in (finset.range 51).filter (λ i , i > 0), (x - i)

noncomputable def denominator (x : ℕ) : ℕ := ∏ i in (finset.range 51).filter (λ i, i > 0), (x - i^2)

theorem num_solutions_to_equation : 
  (∃ n, nat.card { x : ℕ | 1 ≤ x ∧ x ≤ 50 ∧ numerator x = 0 ∧ denominator x ≠ 0 } = n) ∧ n = 43 := 
by
  sorry

end num_solutions_to_equation_l343_343509


namespace count_logical_propositions_l343_343613

def proposition_1 : Prop := ∃ d : ℕ, d = 1
def proposition_2 : Prop := ∀ n : ℕ, n % 10 = 0 → n % 5 = 0
def proposition_3 : Prop := ∀ t : Prop, t → ¬t

theorem count_logical_propositions :
  (proposition_1 ∧ proposition_3) →
  (proposition_1 ∧ proposition_2 ∧ proposition_3) →
  (∃ (n : ℕ), n = 10 ∧ n % 5 = 0) ∧ n = 2 :=
sorry

end count_logical_propositions_l343_343613


namespace two_distinct_real_roots_l343_343859

theorem two_distinct_real_roots (a : ℝ) : a ≠ 0 ∧ a < 1 ↔ (ax^2 - 2x + 1 = 0) has_two_distinct_real_roots :=
by
  sorry

end two_distinct_real_roots_l343_343859


namespace selena_trip_length_l343_343591

variable (y : ℚ)

def selena_trip (y : ℚ) : Prop :=
  y / 4 + 16 + y / 6 = y

theorem selena_trip_length : selena_trip y → y = 192 / 7 :=
by
  sorry

end selena_trip_length_l343_343591


namespace distance_first_to_last_tree_l343_343413

theorem distance_first_to_last_tree 
    (n_trees : ℕ) 
    (distance_first_to_fifth : ℕ)
    (h1 : n_trees = 8)
    (h2 : distance_first_to_fifth = 80) 
    : ∃ distance_first_to_last, distance_first_to_last = 140 := by
  sorry

end distance_first_to_last_tree_l343_343413


namespace hexagon_area_sum_l343_343723

theorem hexagon_area_sum (a b c d e f : ℕ) (h : {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧ 
  (a - d = e - b) ∧ (e - b = c - f)) : 
  (a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 4 ∧ e = 2 ∧ f = 6 ∨
   a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 2 ∧ e = 3 ∧ f = 6) → 
  ∑ (perm : (exists (a b c d e f : ℕ), {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a ∧ (a - d = e - b) ∧ (e - b = c - f))), 
  (if (a = 1 ∧ b = 5 ∧ c = 3 ∧ d = 4 ∧ e = 2 ∧ f = 6) 
   then (3 * (sqrt 3) * (1^2 + 4^2) / 2) else 0) + 
  (if (a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 2 ∧ e = 3 ∧ f = 6) 
   then (3 * (sqrt 3) * (2^2 + 5^2) / 2) else 0) = 69 * (sqrt 3) :=
sorry

end hexagon_area_sum_l343_343723


namespace find_CF_length_l343_343917

noncomputable def rectangle_ABCD (A B C D : ℝ × ℝ) : Prop :=
  (A.1 = D.1) ∧ (B.1 = C.1) ∧ (A.2 = B.2) ∧ (C.2 = D.2) ∧
  (abs (B.1 - A.1) = 6) ∧ (abs (D.2 - A.2) = 7) ∧ (abs (C.1 - D.1) = 8)

noncomputable def CF_length (A B C D E F : ℝ × ℝ) : ℝ :=
  let D_to_F := sqrt ((D.1 - F.1)^2 + (D.2 - F.2)^2)
  let D_to_C := sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  D_to_F - D_to_C

theorem find_CF_length (A B C D E F : ℝ × ℝ)
  (h_rect : rectangle_ABCD A B C D)
  (h_centroid : centroid D E F = B)
  (h_C_on_DF : C.1 = D.1 + k * (F.1 - D.1) ∧ C.2 = D.2 + k * (F.2 - D.2)) :
  CF_length A B C D E F = 10.66 :=
sorry

end find_CF_length_l343_343917


namespace eval_expression_l343_343781

theorem eval_expression : (125 ^ (1/3) * 81 ^ (-1/4) * 32 ^ (1/5) = (10/3)) :=
by
  have h1 : 125 = 5^3 := by norm_num
  have h2 : 81 = 3^4 := by norm_num
  have h3 : 32 = 2^5 := by norm_num
  sorry

end eval_expression_l343_343781


namespace find_b_plus_c_l343_343119

variable {a b c d : ℝ}

theorem find_b_plus_c
  (h1 : a + b = 4)
  (h2 : c + d = 3)
  (h3 : a + d = 2) :
  b + c = 5 := 
  by
  sorry

end find_b_plus_c_l343_343119


namespace fixed_point_l343_343615

noncomputable def f (a : ℝ) (x : ℝ) := a^(x - 2) - 3

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 2 = -2 :=
by
  sorry

end fixed_point_l343_343615


namespace cos_sum_identity_l343_343472

theorem cos_sum_identity (θ : ℝ) (h1 : Real.tan θ = -5 / 12) (h2 : θ ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  Real.cos (θ + Real.pi / 4) = 17 * Real.sqrt 2 / 26 :=
sorry

end cos_sum_identity_l343_343472


namespace monic_poly_irrational_l343_343014

theorem monic_poly_irrational:
  ∀ (f : ℤ[X]), (monic f ∧ (∀ x ∈ ℤ, x = 0 → f.eval x = 2004) ∧ 
  (∀ x ∈ ℚ, irrational x → irrational (f.eval x))) → f = polynomial.monic (polynomial.X + 2004) :=
sorry

end monic_poly_irrational_l343_343014


namespace female_democrats_l343_343335

/-
There are 810 male and female participants in a meeting.
Half of the female participants and one-quarter of the male participants are Democrats.
One-third of all the participants are Democrats.
Prove that the number of female Democrats is 135.
-/

theorem female_democrats (F M : ℕ) (h : F + M = 810)
  (female_democrats : F / 2 = F / 2)
  (male_democrats : M / 4 = M / 4)
  (total_democrats : (F / 2 + M / 4) = 810 / 3) : 
  F / 2 = 135 := by
  sorry

end female_democrats_l343_343335


namespace magic_square_problem_l343_343277

-- Define initial conditions
def numbers : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def sum : Nat := 45
def line_sum : Nat := 15

-- Definition of a 3x3 matrix that represents the magic square
def magic_square (square : Matrix (Fin 3) (Fin 3) Nat) : Prop :=
  -- All numbers between 1 and 9 are used
  (∀ i j, square i j ∈ numbers) ∧
  (∀ n ∈ numbers, ∃ i j, square i j = n) ∧
  -- Sum of each row, column, and diagonal is 15
  (∀ i, (Finset.univ.sum (λ j, square i j) = line_sum)) ∧
  (∀ j, (Finset.univ.sum (λ i, square i j) = line_sum)) ∧
  (Finset.univ.sum (λ k, square k k) = line_sum) ∧
  (Finset.univ.sum (λ k, square k (Fin.cast k.val.modify (λ i, 2 - i))) = line_sum)

-- Definition of the conditions
def center_is_five (square : Matrix (Fin 3) (Fin 3) Nat) : Prop := square 1 1 = 5

def one_in_middle_edge (square : Matrix (Fin 3) (Fin 3) Nat) : Prop :=
  (square 1 0 = 1) ∨ (square 1 2 = 1) ∨ (square 0 1 = 1) ∨ (square 2 1 = 1)

def eight_in_corner (square : Matrix (Fin 3) (Fin 3) Nat) : Prop :=
  (square 0 0 = 8) ∨ (square 0 2 = 8) ∨ (square 2 0 = 8) ∨ (square 2 2 = 8)

-- Example valid magic square
def example_square : Matrix (Fin 3) (Fin 3) Nat :=
  matrix.of_fn (λ i j, 
    if i = 0 then 
      (if j = 0 then 8 else if j = 1 then 1 else 6)
    else if i = 1 then 
      (if j = 0 then 3 else if j = 1 then 5 else 7)
    else 
      (if j = 0 then 4 else if j = 1 then 9 else 2)
  )

-- The theorem statement
theorem magic_square_problem :
  ∃ (square : Matrix (Fin 3) (Fin 3) Nat), magic_square square ∧ center_is_five square ∧ one_in_middle_edge square ∧ eight_in_corner square :=
  ⟨example_square, sorry⟩

end magic_square_problem_l343_343277


namespace proof_problem_l343_343475

variable {A B C : ℝ} {a b c : ℝ} {S : ℝ}

def sin2_eq_formula (α β : ℝ) : Prop := sin (2 * α) + sin (2 * β) = 2 * sin (α + β) * cos (α - β)

def triangle_condition_1 (A B C : ℝ) : Prop :=
  sin (2 * A) + sin (A - B + C) = sin (C - A - B) + 1 / 2

def area_condition (S : ℝ) : Prop :=
  1 ≤ S ∧ S ≤ 3

def sides_condition (a b c : ℝ) : Prop := 
  ∀ (A B C : ℝ), true  -- Placeholder for side lengths' properties in Lean

def conclusion_1 (A B C : ℝ) : Prop := 
  sin A * sin B * sin C = 1 / 8

def conclusion_2 (a b c A B C : ℝ) : Prop :=
  let s := (a + b + c) / 2 in
  let S := sqrt (s * (s - a) * (s - b) * (s - c)) in
  let R := (a * b * c) / (4 * S) in
  4 ≤ (a + b + c) / (sin A + sin B + sin C) ∧ (a + b + c) / (sin A + sin B + sin C) ≤ 4 * sqrt 3

def conclusion_3 (a b c : ℝ) : Prop :=
  8 ≤ a * b * c ∧ a * b * c ≤ 16 * sqrt 2

def conclusion_4 (a b : ℝ) : Prop :=
  ab (a + b) > 8

theorem proof_problem :
  ∀ (A B C a b c : ℝ),
  sin2_eq_formula A B →
  triangle_condition_1 A B C → area_condition S →
  sides_condition a b c →
  conclusion_1 A B C ∧
  conclusion_2 a b c A B C ∧
  ¬conclusion_3 a b c ∧
  conclusion_4 a b :=
by
  sorry

end proof_problem_l343_343475


namespace price_change_series_l343_343911

-- Definitions of price changes
def initial_price : ℝ := 100
def january_price (P : ℝ) : ℝ := P * 1.10
def february_price (P : ℝ) : ℝ := P * 0.85
def march_price (P : ℝ) : ℝ := P * 1.20
def april_price (P : ℝ) : ℝ := P * 0.90

-- Final price after y% change in May
def may_price (P : ℝ) (y : ℝ) : ℝ := P * (1 + y / 100)

-- Main theorem statement
theorem price_change_series (P0 : ℝ) : 
  let P1 := january_price P0 in
  let P2 := february_price P1 in
  let P3 := march_price P2 in
  let P4 := april_price P3 in
  ∃ y : ℝ, may_price P4 y = P0 ∧ abs (y + 1) < 1 :=
by sorry

end price_change_series_l343_343911


namespace transformed_data_properties_l343_343063

variables {x : Type} [decidable_eq x] [fintype x]
variables {x1 x2 x3 x4 x5 : ℝ}

def mean (x1 x2 x3 x4 x5 : ℝ) : ℝ := (x1 + x2 + x3 + x4 + x5) / 5

def variance (x1 x2 x3 x4 x5 : ℝ) (mean : ℝ) : ℝ :=
  ((x1 - mean) ^ 2 + (x2 - mean) ^ 2 + (x3 - mean) ^ 2 + (x4 - mean) ^ 2 + (x5 - mean) ^ 2) / 5

theorem transformed_data_properties (h_mean : mean x1 x2 x3 x4 x5 = 2)
                                    (h_variance : variance x1 x2 x3 x4 x5 2 = 1 / 3) : 
  mean (2 * x1 - 1) (2 * x2 - 1) (2 * x3 - 1) (2 * x4 - 1) (2 * x5 - 1) = 3 ∧
  variance (2 * x1 - 1) (2 * x2 - 1) (2 * x3 - 1) (2 * x4 - 1) (2 * x5 - 1) 3 = 4 / 3 :=
by
  sorry

end transformed_data_properties_l343_343063


namespace eval_expression_l343_343780

theorem eval_expression : (125 ^ (1/3) * 81 ^ (-1/4) * 32 ^ (1/5) = (10/3)) :=
by
  have h1 : 125 = 5^3 := by norm_num
  have h2 : 81 = 3^4 := by norm_num
  have h3 : 32 = 2^5 := by norm_num
  sorry

end eval_expression_l343_343780


namespace original_survey_response_count_l343_343718

theorem original_survey_response_count (x : ℕ) (h1 : 63 * x ≥ 9 * 80)
  (h2 : 80 * 9 = 63 * x + 80 * 4) : x = 7 :=
by
  have h2' : 567 = 63 * x + 320 := congr_arg (· * 560) h2
  have h : 63 * x = 567 - 320 := eq_sub_of_add_eq h2'
  have h3 : 63 * x = 247 := by linarith
  have h4 : x = 7 := nat.mul_left_inj (by norm_num : 63 ≠ 0) h3
  exact h4


end original_survey_response_count_l343_343718


namespace meaningful_expression_l343_343126

-- Definition stating the meaningfulness of the expression (condition)
def is_meaningful (a : ℝ) : Prop := (a - 1) ≠ 0

-- Theorem stating that for the expression to be meaningful, a ≠ 1
theorem meaningful_expression (a : ℝ) : is_meaningful a ↔ a ≠ 1 :=
by sorry

end meaningful_expression_l343_343126


namespace math_problem_proof_l343_343633

variable (Zhang Li Wang Zhao Liu : Prop)
variable (n : ℕ)
variable (reviewed_truth : Zhang → n = 0 ∧ Li → n = 1 ∧ Wang → n = 2 ∧ Zhao → n = 3 ∧ Liu → n = 4)
variable (reviewed_lie : ¬Zhang → ¬(n = 0) ∧ ¬Li → ¬(n = 1) ∧ ¬Wang → ¬(n = 2) ∧ ¬Zhao → ¬(n = 3) ∧ ¬Liu → ¬(n = 4))
variable (some_reviewed : ∃ x, x ∧ ¬x)

theorem math_problem_proof: n = 1 :=
by
  -- Proof omitted, insert logic here
  sorry

end math_problem_proof_l343_343633


namespace area_of_white_portion_l343_343713

theorem area_of_white_portion (stroke_width height width : ℕ)
  (letter_areas : ℕ) (total_area : ℕ) (black_area : ℕ)
  (total_area_def : total_area = height * width)
  (stroke_width_def : stroke_width = 2)
  (height_def : height = 8)
  (width_def : width = 28)
  (letter_areas_def : letter_areas = 40 + 40 + 24 + 40)
  (black_area_def : black_area = letter_areas):
  total_area - black_area = 80 := by
  have total_area_val : total_area = 8 * 28 := by rw [height_def, width_def]; refl
  have black_area_val : black_area = 40 + 40 + 24 + 40 := by rw letter_areas_def
  have white_area_val : total_area - black_area = 80 := 
    by rw [total_area_val, black_area_val]; norm_num
  exact white_area_val


end area_of_white_portion_l343_343713


namespace stadium_capacity_l343_343303

theorem stadium_capacity 
  (C : ℕ)
  (entry_fee : ℕ := 20)
  (three_fourth_full_fees : ℕ := 3 / 4 * C * entry_fee)
  (full_fees : ℕ := C * entry_fee)
  (fee_difference : ℕ := full_fees - three_fourth_full_fees)
  (h : fee_difference = 10000) :
  C = 2000 :=
by
  sorry

end stadium_capacity_l343_343303


namespace tetrahedron_color_property_l343_343636

structure Tetrahedron where
  vertices : Fin 4 → Color
  edges : (Fin 4 × Fin 4) → Color
  color_invariant : ∀ {v1 v2 : Fin 4}, v1 ≠ v2 → edges (v1, v2) = vertices v1 ∨ edges (v1, v2) = vertices v2
  all_colors_used : ∀ (c : Color), ∃ (v : Fin 4), vertices v = c

inductive Color
  | blue
  | red
  | yellow
  | green

theorem tetrahedron_color_property (T : Tetrahedron) :
  (∃ (v : Fin 4), 
    ∃ (u1 u2 u3 : Fin 4), 
    u1 ≠ v ∧ u2 ≠ v ∧ u3 ≠ v ∧ u1 ≠ u2 ∧ u2 ≠ u3 ∧ u1 ≠ u3 ∧
    ((T.edges (v, u1) = Color.blue ∧ T.edges (v, u2) = Color.red ∧ T.edges (v, u3) = Color.green) ∨
     (T.edges (v, u1) = Color.red ∧ T.edges (v, u2) = Color.blue ∧ T.edges (v, u3) = Color.green) ∨
     (T.edges (v, u1) = Color.green ∧ T.edges (v, u2) = Color.blue ∧ T.edges (v, u3) = Color.red) ∨
     (T.edges (v, u1) = Color.green ∧ T.edges (v, u2) = Color.red ∧ T.edges (v, u3) = Color.blue) ∨
     (T.edges (v, u1) = Color.red ∧ T.edges (v, u2) = Color.green ∧ T.edges (v, u3) = Color.blue) ∨
     (T.edges (v, u1) = Color.blue ∧ T.edges (v, u2) = Color.green ∧ T.edges (v, u3) = Color.red))
  )
  ∨ 
  (∃ (u1 u2 u3 : Fin 4), 
    u1 ≠ u2 ∧ u2 ≠ u3 ∧ u1 ≠ u3 ∧
    ((T.edges (u1, u2) = Color.blue ∧ T.edges (u2, u3) = Color.red ∧ T.edges (u3, u1) = Color.green) ∨
     (T.edges (u1, u2) = Color.red ∧ T.edges (u2, u3) = Color.blue ∧ T.edges (u3, u1) = Color.green) ∨
     (T.edges (u1, u2) = Color.green ∧ T.edges (u2, u3) = Color.blue ∧ T.edges (u3, u1) = Color.red) ∨
     (T.edges (u1, u2) = Color.green ∧ T.edges (u2, u3) = Color.red ∧ T.edges (u3, u1) = Color.blue) ∨
     (T.edges (u1, u2) = Color.red ∧ T.edges (u2, u3) = Color.green ∧ T.edges (u3, u1) = Color.blue) ∨
     (T.edges (u1, u2) = Color.blue ∧ T.edges (u2, u3) = Color.green ∧ T.edges (u3, u1) = Color.red))
  ) :=
sorry

end tetrahedron_color_property_l343_343636


namespace coordinates_of_point_P_l343_343477

theorem coordinates_of_point_P {x y : ℝ} (hx : |x| = 2) (hy : y = 1 ∨ y = -1) (hxy : x < 0 ∧ y > 0) : 
  (x, y) = (-2, 1) := 
by 
  sorry

end coordinates_of_point_P_l343_343477


namespace odd_function_periodic_with_period_3_main_theorem_l343_343836

noncomputable def f : ℝ → ℝ :=
  λ x, if (x % 3) ∈ (-3 / 2 : ℝ) <..< (0 : ℝ) then Math.log 2 (1 - (x % 3)) else 0

theorem odd_function_periodic_with_period_3 :
  (∀ x, f (-x) = -f x) ∧ (∀ x k : ℤ, f (x + 3 * k) = f x) :=
  sorry

theorem main_theorem : f 2014 + f 2016 = -1 :=
by
  sorry

end odd_function_periodic_with_period_3_main_theorem_l343_343836


namespace mike_took_green_marbles_l343_343755

theorem mike_took_green_marbles (original_marbles remaining_marbles taken_marbles : ℝ)
  (h1 : original_marbles = 32.0) 
  (h2 : remaining_marbles = 9.0) 
  (h3 : taken_marbles = original_marbles - remaining_marbles) :
  taken_marbles = 23.0 := 
by
  rw [h1, h2] at h3
  exact h3

end mike_took_green_marbles_l343_343755


namespace tan_alpha_sol_expr_sol_l343_343447

noncomputable def tan_half_alpha (α : ℝ) : ℝ := 2

noncomputable def tan_alpha_from_half (α : ℝ) : ℝ := 
  let tan_half := tan_half_alpha α
  2 * tan_half / (1 - tan_half * tan_half)

theorem tan_alpha_sol (α : ℝ) (h : tan_half_alpha α = 2) : tan_alpha_from_half α = -4 / 3 := by
  sorry

noncomputable def expr_eval (α : ℝ) : ℝ :=
  let tan_α := tan_alpha_from_half α
  let sin_α := tan_α / Real.sqrt (1 + tan_α * tan_α)
  let cos_α := 1 / Real.sqrt (1 + tan_α * tan_α)
  (6 * sin_α + cos_α) / (3 * sin_α - 2 * cos_α)

theorem expr_sol (α : ℝ) (h : tan_half_alpha α = 2) : expr_eval α = 7 / 6 := by
  sorry

end tan_alpha_sol_expr_sol_l343_343447


namespace tangent_to_circumcircle_l343_343976

theorem tangent_to_circumcircle
  (A B C D : Point)
  (hD_on_AC : D ∈ segment A C)
  (h_ratio_AD_DC : 2 * (segment_length D C) = segment_length A D)
  (h_angle_C : angle A C B = 45)
  (h_angle_ADB : angle A D B = 60) :
  tangent_to (line A B) (circumcircle B C D) :=
sorry

end tangent_to_circumcircle_l343_343976


namespace term_2013_is_130_l343_343271

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

-- Define the sequence based on the given conditions
noncomputable def sequence : Nat → Nat
| 0       => 934
| (n + 1) => 13 * sum_of_digits (sequence n)

-- Prove that the 2013th term of the sequence is 130
theorem term_2013_is_130 : sequence 2013 = 130 := 
  sorry

end term_2013_is_130_l343_343271


namespace f_at_2_l343_343452

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x

theorem f_at_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -10 :=
by 
  sorry

end f_at_2_l343_343452


namespace find_c_l343_343612

theorem find_c (c x : ℝ) (h1 : 3 * x + 4 = 2) (h2 : c * x - 15 = 0) : 
  c = -45 / 2 :=
by
  have x_val : x = -2 / 3 := by linarith
  have c_eq : c * (-2 / 3) = 15 := by linarith [h2, x_val]
  have c_val : c = -45 / 2 := by linarith
  exact c_val

end find_c_l343_343612


namespace find_x_value_l343_343681

theorem find_x_value : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / 1 / 2 → x = -21 :=
by
  sorry

end find_x_value_l343_343681


namespace number_of_solutions_eq_43_l343_343506

theorem number_of_solutions_eq_43 :
  let S := {x | x ∈ (Finset.range 51)} -- S = {1, 2, 3, ..., 50}
  let T := {x | x ∈ {1, 4, 9, 16, 25, 36, 49}} -- T = {1, 4, 9, 16, 25, 36, 49}
  |S| - |T| = 43 :=
by
  sorry -- No proof required, placeholder

end number_of_solutions_eq_43_l343_343506


namespace cylinder_intersection_in_sphere_l343_343982

theorem cylinder_intersection_in_sphere
  (a b c d e f : ℝ)
  (x y z : ℝ)
  (h1 : (x - a)^2 + (y - b)^2 < 1)
  (h2 : (y - c)^2 + (z - d)^2 < 1)
  (h3 : (z - e)^2 + (x - f)^2 < 1) :
  (x - (a + f) / 2)^2 + (y - (b + c) / 2)^2 + (z - (d + e) / 2)^2 < 3 / 2 := 
sorry

end cylinder_intersection_in_sphere_l343_343982


namespace evaluate_stability_of_yields_l343_343643

def variance (l : List ℝ) : ℝ :=
l.map (λ x, (x - l.sum / l.length)^2).sum / l.length

theorem evaluate_stability_of_yields (x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 x_9 x_{10} : ℝ) :
  let yields := [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_{10}] in
  let mean := yields.sum / yields.length in
  variance yields = (yields.map (λ x, (x - mean)^2)).sum / yields.length :=
  sorry

end evaluate_stability_of_yields_l343_343643


namespace solve_differential_eq_l343_343600

noncomputable def differentialeq_solution (y : ℝ → ℝ) :=
  y'' = λ t, (∂ ∂ t (∂ ∂ t y t)) t 
  ∧ (y'' t - 2 * (∂ ∂ t y t) - 3 * y t  = e ^ (3 * t))
  ∧ (y 0 = 0)
  ∧ ((∂ ∂ t y) 0 = 0)

theorem solve_differential_eq : 
  ∀ t : ℝ, 
  let y := (λ t : ℝ, (1 / 4) * t * (Real.exp (3 * t)) - (1 / 16) * (Real.exp (3 * t)) + (1 / 16) * (Real.exp (-t))) in
  differentialeq_solution y :=
by 
  -- Proof goes here
  sorry

end solve_differential_eq_l343_343600


namespace sum_of_primes_58_l343_343910

/-
Problem: Prove that there are exactly 2 distinct pairs of prime numbers whose sum is 58, given the conditions:
1. All prime numbers greater than 2 are odd.
2. The only even prime number is 2.
3. 58 is even.
-/

theorem sum_of_primes_58 :
  ∃! pairs : Finset (ℕ × ℕ), 
    (∀ pair ∈ pairs, nat.prime pair.fst ∧ nat.prime pair.snd ∧ pair.fst + pair.snd = 58) ∧
    pairs.card = 2 :=
sorry

end sum_of_primes_58_l343_343910


namespace sum_of_alternating_sums_10_l343_343806

open Finset

-- Define the set 1 to 10
def S : Finset ℕ := range 10 + 1

-- Define the alternating sum function for a non-empty subset
def alt_sum (S : Finset ℕ) : ℕ :=
  (S.sort (· > ·)).foldl (λ sum x, sum + (-1) ^ S.card * x) 0

-- The main theorem to prove
theorem sum_of_alternating_sums_10 :
  ∑ S in (powerset S).filter (λ s, s ≠ ∅),
    alt_sum S = 5120 := sorry

end sum_of_alternating_sums_10_l343_343806


namespace janessa_gives_dexter_cards_l343_343927

def initial_cards : Nat := 4
def father_cards : Nat := 13
def ordered_cards : Nat := 36
def bad_cards : Nat := 4
def kept_cards : Nat := 20

theorem janessa_gives_dexter_cards :
  initial_cards + father_cards + ordered_cards - bad_cards - kept_cards = 29 := 
by
  sorry

end janessa_gives_dexter_cards_l343_343927


namespace bc_length_l343_343236

-- Definitions and properties of the given problem
structure Rectangle (A B C D E M : Point) :=
  (length_AD : ℝ) -- length of AD
  (length_AB : ℝ) -- length of AB
  (length_CD : ℝ) -- length of CD = length of AB
  (length_ED : ℝ) -- length of ED
  (on_AD_E : E ∈ segment A D) -- Point E is on AD
  (on_EC_M : M ∈ segment E C) -- Point M is on EC
  (AB_eq_BM : length (segment A B) = length (segment B M)) -- AB = BM
  (AE_eq_EM : length (segment A E) = length (segment E M)) -- AE = EM 

-- Given conditions in the problem
def rect_conditions := { 
  length_AD := 0, -- side AD length is not directly given
  length_AB := 0, -- side AB length is not directly given
  length_CD := 12,
  length_ED := 16,
  on_AD_E := some_condition, -- condition that E is on AD
  on_EC_M := some_condition, -- condition that M is on EC
  AB_eq_BM := some_condition, -- condition AB = BM
  AE_eq_EM := some_condition  -- condition AE = EM
}

-- Theorem statement to prove
theorem bc_length (A B C D E M : Point)
  (R : Rectangle A B C D E M)
  (h1 : R.length_CD = 12)
  (h2 : R.length_ED = 16)
  (h3 : R.length (segment A B) = R.length (segment B M))
  (h4 : R.length (segment A E) = R.length (segment E M)) :
  R.length (segment B C) = 20 := 
by 
  sorry

end bc_length_l343_343236


namespace years_before_marriage_l343_343298

theorem years_before_marriage {wedding_anniversary : ℕ} 
  (current_year : ℕ) (met_year : ℕ) (years_before_dating : ℕ) :
  wedding_anniversary = 20 →
  current_year = 2025 →
  met_year = 2000 →
  years_before_dating = 2 →
  met_year + years_before_dating + (current_year - met_year - wedding_anniversary) = current_year - wedding_anniversary - years_before_dating + wedding_anniversary - current_year :=
by
  sorry

end years_before_marriage_l343_343298


namespace find_diameter_of_circular_field_l343_343429

noncomputable def diameter_of_circle (cost_per_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let C := total_cost / cost_per_meter
  C / Real.pi

theorem find_diameter_of_circular_field :
  diameter_of_circle 2.50 172.79 ≈ 22.00 :=
by sorry

end find_diameter_of_circular_field_l343_343429


namespace ellipse_parabola_intersection_l343_343409

theorem ellipse_parabola_intersection (c : ℝ) : 
  (∀ x y : ℝ, (x^2 + (y^2 / 4) = c^2 ∧ y = x^2 - 2 * c) → false) ↔ c > 1 := by
  sorry

end ellipse_parabola_intersection_l343_343409


namespace probability_all_odd_slips_l343_343673

theorem probability_all_odd_slips :
  let slips := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      odd_slips := [1, 3, 5, 7, 9]
      total_slips := 10
      odd_count := 5
      draws := 4
  in 
  (5 / 10 : ℚ) * (4 / 9) * (3 / 8) * (2 / 7) = 1 / 42 :=
by
  sorry

end probability_all_odd_slips_l343_343673


namespace min_value_on_interval_l343_343624

noncomputable def f (x : ℝ) : ℝ := x^4 - 4 * x + 3

theorem min_value_on_interval : 
  ∃ (m : ℝ), (∀ x ∈ set.Icc (-2 : ℝ) (3 : ℝ), f x ≥ m) ∧ 
             (∃ x ∈ set.Icc (-2 : ℝ) (3 : ℝ), f x = m) ∧ m = 0 :=
by {
  sorry 
}

end min_value_on_interval_l343_343624


namespace largest_t_l343_343551

noncomputable def t := 3

theorem largest_t (p : ℕ) (hp : p ≥ 17 ∧ Nat.Prime p) :
  (∀ (a b c d : ℤ),
    (a * b * c) % p ≠ 0 →
    (a + b + c) % p = 0 →
    ∃ (x y z : ℤ), 
    0 ≤ x ∧ x ≤ ⌊p / t⌋ - 1 ∧
    0 ≤ y ∧ y ≤ ⌊p / t⌋ - 1 ∧
    0 ≤ z ∧ z ≤ ⌊p / t⌋ - 1 ∧
    (a * x + b * y + c * z + d) % p = 0) :=
sorry

end largest_t_l343_343551


namespace tan_sub_eq_one_third_l343_343471

theorem tan_sub_eq_one_third (α β : Real) (hα : Real.tan α = 3) (hβ : Real.tan β = 4/3) : 
  Real.tan (α - β) = 1/3 := by
  sorry

end tan_sub_eq_one_third_l343_343471


namespace part_I_part_II_l343_343108

-- Definitions of vectors
def vec_a : ℝ × ℝ := (1, 0)
def vec_b : ℝ × ℝ := (1, 1)
def vec_c : ℝ × ℝ := (-1, 1)

-- Helper function for dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Helper function to check perpendicularity
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Problem I: Find lambda such that vec_a + λ * vec_b is perpendicular to vec_a
theorem part_I (λ : ℝ) : is_perpendicular (vec_a.1 + λ * vec_b.1, vec_a.2 + λ * vec_b.2) vec_a ↔ λ = -1 := sorry

-- Helper function for vector addition
def add_vectors (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Helper function for checking parallelism
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Problem II: Given that m * vec_a + n * vec_b is parallel to vec_c, find m / n
theorem part_II (m n : ℝ) (h : is_parallel (m * vec_a.1 + n * vec_b.1, m * vec_a.2 + n * vec_b.2) vec_c) : m / n = -1 / 2 := sorry

end part_I_part_II_l343_343108
