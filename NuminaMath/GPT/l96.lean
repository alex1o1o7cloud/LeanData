import Mathlib
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Cubic
import Mathlib.Algebra.EuclideanMetric
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.LinearAlgebra
import Mathlib.Algebra.Log
import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Permutations
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Combinations
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.TetrahedralVolume
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Mathlib
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.MeasureTheory.Measure.Lebesgue
import Mathlib.MeasureTheory.ProbabilityMassFunction
import Mathlib.NumberTheory.GCD
import Mathlib.Probability
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Real
import tactic

namespace total_hours_watched_l96_96136

theorem total_hours_watched (Monday Tuesday Wednesday Thursday Friday : ℕ) (hMonday : Monday = 12) (hTuesday : Tuesday = 4) (hWednesday : Wednesday = 6) (hThursday : Thursday = (Monday + Tuesday + Wednesday) / 2) (hFriday : Friday = 19) :
  Monday + Tuesday + Wednesday + Thursday + Friday = 52 := by
  sorry

end total_hours_watched_l96_96136


namespace tv_selection_l96_96177

theorem tv_selection (A B : ℕ) (hA : A = 4) (hB : B = 5) : 
  ∃ n, n = 3 ∧ (∃ k, k = 70 ∧ 
    (n = 1 ∧ k = A * (B * (B - 1) / 2) + A * (A - 1) / 2 * B)) :=
sorry

end tv_selection_l96_96177


namespace minimum_value_tan_product_l96_96527

theorem minimum_value_tan_product
  (S A B C O : Point)
  (h₁ : is_tetrahedron_with_perpendicular_edges S A B C)
  (h₂ : is_point_inside_triangle O A B C) :
  ∃ W, W = tan (∠ O S A) * tan (∠ O S B) * tan (∠ O S C) ∧ W = 2 * sqrt 2 :=
sorry

end minimum_value_tan_product_l96_96527


namespace projection_of_a_in_direction_of_b_l96_96762

/--
  Given vectors a and b as defined below, 
  prove that the projection of vector a in the direction of vector b is -3.
-/
theorem projection_of_a_in_direction_of_b :
  let a := (-1, 3) : ℝ × ℝ;
      b := (3, -4) : ℝ × ℝ;
  (a.1 * b.1 + a.2 * b.2) / real.sqrt (b.1 ^ 2 + b.2 ^ 2) = -3 :=
by
  let a := (-1, 3) : ℝ × ℝ
  let b := (3, -4) : ℝ × ℝ
  let dot_product := a.1 * b.1 + a.2 * b.2
  have magnitude_b : real.sqrt (b.1 ^ 2 + b.2 ^ 2) = 5 := by sorry
  have dot_product_value : dot_product = -15 := by sorry
  show dot_product / magnitude_b = -3 
     -- This proof obligation should be satisfied based on conditions.
     sorry

end projection_of_a_in_direction_of_b_l96_96762


namespace maximize_daily_profit_achieve_84_percent_profit_l96_96508

-- Define the conditions
def sales_volume_linear (k b : ℝ) (x : ℝ) : ℝ := k * x + b

def cost_price : ℝ := 10
def price_upper_limit : ℝ := 110
def sales_volume_zero_at_upper_limit (k b : ℝ) : Prop := 
  sales_volume_linear k b price_upper_limit = 0

-- Define the profit function
def daily_profit (k : ℝ) (x : ℝ) : ℝ := k * (x - cost_price) * (x - price_upper_limit)

-- Theorem (1): To maximize daily profit, the price should be 60 yuan
theorem maximize_daily_profit (k : ℝ) (h : k < 0) (b : ℝ) 
  (H1 : sales_volume_zero_at_upper_limit k b) : 
  arg_max (daily_profit k) (fun x => cost_price < x ∧ x < price_upper_limit) = 60 :=
sorry

-- Theorem (2): To achieve 84% of the maximum daily profit, the price should be 40 yuan or 80 yuan
theorem achieve_84_percent_profit (k : ℝ) (h : k < 0) (b : ℝ) 
  (H1 : sales_volume_zero_at_upper_limit k b) : 
  ∃ x, (daily_profit k x) = 0.84 * (daily_profit k 60) ∧ (x = 40 ∨ x = 80) :=
sorry

end maximize_daily_profit_achieve_84_percent_profit_l96_96508


namespace simplify_neg_frac_exp_l96_96230

theorem simplify_neg_frac_exp : (- (1 / 343) : ℚ) ^ (-2 / 3) = 49 := 
  sorry

end simplify_neg_frac_exp_l96_96230


namespace problem_l96_96733

open Real

noncomputable def ellipseM (a b : ℝ) : (ℝ × ℝ) → Prop :=
λ p, (p.1^2 / a^2) + (p.2^2 / b^2) = 1

def ellipseN (p : ℝ × ℝ) : Prop :=
(p.1^2 / 9) + (p.2^2 / 5) = 1

theorem problem
  (a b : ℝ)
  (h1 : b = 2)
  (h2 : a^2 - b^2 = 4)
  (h3 : ellipseM a b (0, 2))
  (A B : ℝ × ℝ)
  (h4 : (A.2 = A.1 + 2) ∧ ellipseM a b A)
  (h5 : (B.2 = B.1 + 2) ∧ ellipseM a b B)
  (h6 : A.1 > B.1)
  (O : ℝ × ℝ) (hO : O = (0, 0)) :
  2 * a = 4 * sqrt 2 ∧ (O.1 * A.1 + O.2 * A.2) * (O.1 * B.1 + O.2 * B.2) = -4/3 :=
sorry

end problem_l96_96733


namespace first_visitor_arrived_at_0715_l96_96284

-- Definitions
def consistent_arrivals (N : ℕ → ℕ) : Prop :=
  ∀ t₁ t₂ : ℕ, t₁ ≥ t₂ → N t₁ - N t₂ = (t₁ - t₂) * (N 1 - N 0)

constant N : ℕ → ℕ

axiom consistent_visitors : consistent_arrivals N

axiom three_entry_points_ends_at_809 : N 9 = 27

axiom five_entry_points_ends_at_805 : N 5 = 25

-- The time the first visitor arrived
theorem first_visitor_arrived_at_0715 : N 0 = 0 ∧ (∃ t : ℕ, 7 * 60 + 15 = t) :=
by
  sorry

end first_visitor_arrived_at_0715_l96_96284


namespace find_angle_ratio_l96_96522

noncomputable def ratio_cone_spheres (k : ℝ) : Prop :=
  ∃ (α : ℝ), α = arccos ((1 + sqrt (1 - 2 * (k)^(1/3))) / 2) ∧ 0 < k ∧ k ≤ 1/8

theorem find_angle_ratio (k : ℝ) (h : 0 < k ∧ k ≤ 1/8) : 
  ratio_cone_spheres k := by
  sorry

end find_angle_ratio_l96_96522


namespace distance_between_parallel_lines_l96_96066

theorem distance_between_parallel_lines :
  let A := 6
  let B := 4
  let C1 := 1
  let C2 := -6
  let distance := (abs (C1 - C2) / real.sqrt (A^2 + B^2))
  distance = 7 * real.sqrt 13 / 26 :=
by {
  sorry
}

end distance_between_parallel_lines_l96_96066


namespace arithmetic_sequence_count_l96_96049

theorem arithmetic_sequence_count :
  ∃ n : ℕ, n = 675685 ∧ ∀ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d ∧ 
    a ≤ 2012 ∧ b ≤ 2012 ∧ c ≤ 2012 ∧ d ≤ 2012 ∧
    (∃ x : ℕ, b = a + x ∧ c = a + 2 * x ∧ d = a + 3 * x) →
    (∑ x in finset.range 671, 2012 - 3 * x) = n :=
begin
  use 675685,
  split,
  { refl, },
  { intros a b c d hab hbc hcd ha hb hc hd h_seq,
    simp only [finset.sum_range, nat.smul_eq_mul],
    calc (∑ x in finset.range 671, 2012 - 3 * x)
      = ∑ x in finset.range 671, 2012 - ∑ x in finset.range 671, 3 * x : sorry 
    ... = 2012 * 670 - 3 * (∑ x in finset.range 671, x) : sorry
    ... = 2012 * 670 - 3 * (670 * 671 / 2) : sorry
    ... = 675685 : sorry }
end

end arithmetic_sequence_count_l96_96049


namespace infinite_product_seq_a_l96_96885

def seq_a : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := 1 + (seq_a n - 1)^2

theorem infinite_product_seq_a :
  (∏ i in finset.range n, seq_a i)→ 2 / 3 ∈ at_top :=
sorry

end infinite_product_seq_a_l96_96885


namespace ball_attendance_l96_96855

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l96_96855


namespace count_last_digit_3_l96_96628

-- Define the units digit of 7^n
def units_digit(n : ℕ) : ℕ := (7 ^ n) % 10

-- Prove that the number of terms in the sequence 7, 7^2, ..., 7^2011 with last digit 3 is 503
theorem count_last_digit_3 : (finset.card (finset.filter (λ n => units_digit n = 3) (finset.range 2012))) = 503 :=
sorry

end count_last_digit_3_l96_96628


namespace dot_product_self_l96_96408

variables (v : ℝ^3)

theorem dot_product_self {v : ℝ^3} (h_v_norm : ∥v∥ = 5) : v ∙ v = 25 :=
by {
  sorry
}

end dot_product_self_l96_96408


namespace div_pow_sub_one_l96_96578

theorem div_pow_sub_one (n : ℕ) (h : n > 1) : (n - 1) ^ 2 ∣ n ^ (n - 1) - 1 :=
sorry

end div_pow_sub_one_l96_96578


namespace monotonic_increasing_interval_l96_96520

noncomputable def f (x : ℝ) := log (sin (π / 4 - 2 * x))

theorem monotonic_increasing_interval :
  ∀ x, (5 * π / 8 < x ∧ x < 7 * π / 8) ↔ ∃ δ > 0, ∀ ε > 0, 0 < y - x ∧ y - x < δ → f y > f x :=
sorry

end monotonic_increasing_interval_l96_96520


namespace permutations_modulo_l96_96873

theorem permutations_modulo (M : ℕ) :
  let str := "AAAABBBCCCCDDDD".toList in
  let countA := str.count (λ x => x == 'A') in
  let countB := str.count (λ x => x == 'B') in
  let countC := str.count (λ x => x == 'C') in
  let countD := str.count (λ x => x == 'D') in
  countA = 4 ∧ countB = 3 ∧ countC = 4 ∧ countD = 4 → 
  (M = nat.fact (countA + countB + countC + countD) /
   (nat.fact countA * nat.fact countB * nat.fact countC * nat.fact countD) ∧
  (str.take 5).all (λ x => x ≠ 'A') ∧
  (str.drop 5).take 4.all (λ x => x ≠ 'B') ∧
  (str.drop 9).take 4.all (λ x => x ≠ 'C') ∧
  (str.drop 13).all (λ x => x ≠ 'D'))
  → M % 1000 = 140 :=
begin
  intros str countA countB countC countD h,
  sorry
end

end permutations_modulo_l96_96873


namespace find_parallel_line_through_point_l96_96684

noncomputable def is_equation_of_parallel_line {α : Type*} [linear_ordered_field α] 
  (x y b : α) 
  (point : α × α) : Prop :=
(2 * point.1 + point.2 + b = 0) ∧ (2 ≠ 0)

theorem find_parallel_line_through_point {α : Type*} [linear_ordered_field α] 
  (P : α × α) 
  (hP : P = (-1, 2)) 
  (line1 : α → α → α) 
  (hline1 : line1 = λ x y, 2 * x + y - 5) : line1 (-1) 2 = 0 → 
  ∃ b : α, is_equation_of_parallel_line 2 1 b P ∧ b = 0 := 
by
  intro h
  use 0
  have h_eq : 2 * -1 + 2 + 0 = 0 := by linarith
  have h_parallel : 2 ≠ 0 := by norm_num
  exact ⟨⟨h_eq, h_parallel⟩, rfl⟩

end find_parallel_line_through_point_l96_96684


namespace remainder_sumF_div_1000_l96_96698

def f (n k : ℕ) : ℕ := n % k

def F (n : ℕ) : ℕ := if n > 1 then 
  finset.max' (finset.range ((n / 2) + 1)) (by linarith) (λ k, f n (k + 1))
else 0

def sumF : ℕ := (finset.range 81).sum (λ i, F (i + 20))

theorem remainder_sumF_div_1000 : (sumF % 1000) = 512 := 
by sorry

end remainder_sumF_div_1000_l96_96698


namespace ball_total_attendance_l96_96845

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l96_96845


namespace infinite_solutions_to_equation_l96_96171

theorem infinite_solutions_to_equation :
  ∃ᵐ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a^7 + b^8 = c^9 := 
sorry

end infinite_solutions_to_equation_l96_96171


namespace count_even_numbers_between_300_and_600_l96_96400

theorem count_even_numbers_between_300_and_600 : 
  card {n : ℕ | 300 < n ∧ n < 600 ∧ n % 2 = 0} = 149 := 
sorry

end count_even_numbers_between_300_and_600_l96_96400


namespace basketball_player_number_of_distinct_scores_l96_96988

theorem basketball_player_number_of_distinct_scores :
  ∀ (x y z : ℕ), x + y + z = 7 → 
  (∃ n : ℕ, n = 13 ∧ set = { P | P = 2*x + 3*y + 4*z }) :=
sorry

end basketball_player_number_of_distinct_scores_l96_96988


namespace simplify_tangent_sum_l96_96907

theorem simplify_tangent_sum :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 := by
  have h1 : Real.tan (45 * Real.pi / 180) = Real.tan ((10 + 35) * Real.pi / 180) := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h3 : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y) := by sorry
  sorry

end simplify_tangent_sum_l96_96907


namespace apple_in_box_C_l96_96533

def is_apple_in_box (boxes : ℕ → Prop) (truth_notes : ℕ → Prop) : Prop :=
  ∃ (n : ℕ), boxes n ∧ (∀ (m : ℕ), (m ≠ n → ¬ boxes m)) ∧
  (∃! (k : ℕ), truth_notes k)

def boxes (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

def truth_notes (n : ℕ) : Prop :=
  match n with
  | 1 => boxes 1
  | 2 => ¬ boxes 1
  | 3 => ¬ boxes 3
  | 4 => boxes 4
  | _ => false

theorem apple_in_box_C :
  is_apple_in_box (λ n, n = 3) truth_notes :=
by
  { 
    -- Place proof here
    sorry
  }

end apple_in_box_C_l96_96533


namespace negation_of_proposition_l96_96943

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 0 < x → (x^2 + x > 0)) ↔ ∃ x : ℝ, 0 < x ∧ (x^2 + x ≤ 0) :=
sorry

end negation_of_proposition_l96_96943


namespace solve_pier_problem_l96_96530

-- Define the piers
inductive Pier : Type
| p1 : Pier
| p2 : Pier
| p3 : Pier
| p4 : Pier
| p5 : Pier

-- Define the initial state with person and boat placement
structure State :=
(person_at : Pier → ℕ)  -- denotes the person at each pier
(boat_at : Pier)        -- denotes the pier where the boat is

-- Define the transition indicating a valid move
def valid_move (s : State) (from_pier to_pier : Pier) : Prop :=
  (s.boat_at = from_pier) ∧ (from_pier ≠ to_pier)

-- Define the movement function
def move (s : State) (from_pier to_pier : Pier) : State :=
{ person_at := fun p => if p = to_pier then s.person_at from_pier else s.person_at p,
  boat_at := to_pier }

-- Initial state configuration
def initial_state : State :=
{ person_at := fun
  | Pier.p1 => 1
  | Pier.p2 => 2
  | Pier.p3 => 3
  | Pier.p4 => 4
  | Pier.p5 => 5,
boat_at := Pier.p1 }

-- Define the goal state configuration
def goal_state : State :=
{ person_at := fun
  | Pier.p1 => 5
  | Pier.p2 => 1
  | Pier.p3 => 2
  | Pier.p4 => 3
  | Pier.p5 => 4,
boat_at := Pier.p1 }  -- The boat position doesn't matter for the final goal

-- Theorem statement: proving the existence of a sequence of valid moves to the goal state
theorem solve_pier_problem : ∃ (seq : list (Pier × Pier)),
  let final_state := seq.foldl (λ s move, move s move.1 move.2) initial_state in
  final_state = goal_state :=
sorry

end solve_pier_problem_l96_96530


namespace sum_of_sides_l96_96560

theorem sum_of_sides (triangle_sides square_sides hexagon_sides : ℕ)
  (h_triangle : triangle_sides = 3)
  (h_square : square_sides = 4)
  (h_hexagon : hexagon_sides = 6) :
  triangle_sides + square_sides + hexagon_sides = 13 :=
by
  rw [h_triangle, h_square, h_hexagon]
  sorry

end sum_of_sides_l96_96560


namespace prism_diagonal_length_l96_96637

noncomputable theory

variable (a b h : ℝ)

def base_area (a b : ℝ) := (1/2) * a * b = 63

def legs_relation (a b : ℝ) := b = a + 2

def height_relation (h b : ℝ) := h = b

theorem prism_diagonal_length (a b h d : ℝ)
    (h_base_area : base_area a b)
    (h_legs_relation : legs_relation a b)
    (h_height_relation : height_relation h b) :
    d = real.sqrt (h * h + b * b) :=
  sorry

end prism_diagonal_length_l96_96637


namespace complex_eq_solution_l96_96777

theorem complex_eq_solution (x y : ℝ) (h : x - 1 + y * complex.i = complex.i - 3 * x) :
  x = 1 / 4 ∧ y = 1 :=
by
  sorry

end complex_eq_solution_l96_96777


namespace total_attended_ball_lt_fifty_l96_96851

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l96_96851


namespace nested_radical_eq_5_l96_96693

noncomputable def nested_radical : ℝ :=
  Inf { x | x = Real.sqrt (20 + x)}

theorem nested_radical_eq_5 : ∃ x : ℝ, nested_radical = 5 :=
by
  sorry

end nested_radical_eq_5_l96_96693


namespace bridge_length_l96_96636

theorem bridge_length (train_length : ℝ)
                      (train_speed_km_hr : ℝ)
                      (cross_time : ℝ)
                      (train_conv_speed_m_s : train_speed_km_hr * (1000 / 3600) = 11.25)
                      (cross_distance : train_conv_speed_m_s * cross_time)
                      (bridge_length : cross_distance = train_length + bridge_length) : 
                      bridge_length = 217.5 :=
by
  -- We have train_length = 120
  -- We have train_speed_km_hr = 45
  -- We have cross_time = 30
  -- We have train_conv_speed_m_s = 11.25
  -- We have trav_dist = train_conv_speed_m_s * cross_time
  -- Hence, bridge_length = cross_distance - train_length = 217.5 meters
  sorry

end bridge_length_l96_96636


namespace DianeAgeInSixYears_l96_96672

def DeniseAgeInTwoYears : ℕ := 25
def AgeDifference : ℕ := 4

theorem DianeAgeInSixYears :
  let DeniseCurrentAge := DeniseAgeInTwoYears - 2 in
  let DianeCurrentAge := DeniseCurrentAge - AgeDifference in
  25 - DianeCurrentAge = 6 :=
by
  sorry

end DianeAgeInSixYears_l96_96672


namespace ternary_to_decimal_121_l96_96663

theorem ternary_to_decimal_121 : 
  let t : ℕ := 1 * 3^2 + 2 * 3^1 + 1 * 3^0 
  in t = 16 :=
by
  sorry

end ternary_to_decimal_121_l96_96663


namespace equivalency_of_conditions_l96_96442

variable {x y : ℤ}

def Cond1 : Prop := x = 5 * (y - 10)
def Cond2 : Prop := x - 10 = y + 10
def OptionA : Prop := (x + 10 - (y - 10) = 5 * (y - 10)) ∧ (x - 10 = y + 10)

theorem equivalency_of_conditions : Cond1 ∧ Cond2 ↔ OptionA :=
by
  sorry

end equivalency_of_conditions_l96_96442


namespace intersection_of_two_curves_l96_96985

theorem intersection_of_two_curves:
  let curve1 := { p : ℚ × ℚ | ∃ α : ℚ, p.1 = Real.cos α ∧ p.2 = 1 + Real.sin α }
  let curve2 := { p : ℚ × ℚ | (p.1 - 1)^2 + (p.2)^2 = 1 }
  (∀ p : ℚ × ℚ, p ∈ curve1 → p ∈ curve2 → p = (Real.cos (π/4), 1 + Real.sin (π/4)) ∨ p = (Real.cos (3*π/4), 1 + Real.sin (3*π/4))) ↔ 2 :=
sorry

end intersection_of_two_curves_l96_96985


namespace range_of_h_l96_96310

noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 9 * x^2)

theorem range_of_h {a b : ℝ} (h_range : (set.Ioo 0 3) = {y | ∃ x, h x = y}) : a = 0 ∧ b = 3 → a + b = 3 :=
by sorry

end range_of_h_l96_96310


namespace loraine_total_wax_l96_96154

-- Conditions
def large_animal_wax := 4
def small_animal_wax := 2
def small_animal_count := 12 / small_animal_wax
def large_animal_count := small_animal_count / 3
def total_wax := 12 + (large_animal_count * large_animal_wax)

-- The proof problem
theorem loraine_total_wax : total_wax = 20 := by
  sorry

end loraine_total_wax_l96_96154


namespace sin_ratio_triangle_l96_96819

def triangle_angles (A B C : Type) : Prop :=
  ∠ABC = 45 ∧ ∠ACB = 60

def divides_ratio (B C D : Type) : Prop :=
  BD / CD = 3 / 1

theorem sin_ratio_triangle (A B C D : Type) (h1 : triangle_angles A B C) (h2 : divides_ratio B C D) :
  sin (angle A B D) / sin (angle A C D) = sqrt 6 := 
sorry 

end sin_ratio_triangle_l96_96819


namespace copper_wire_diameter_l96_96982

noncomputable def required_wire_diameter
  (distance : ℝ)
  (internal_resistance_bell : ℝ)
  (cell_emf : ℝ)
  (cell_internal_resistance : ℝ)
  (num_cells : ℕ)
  (min_current : ℝ)
  (resistance_per_meter_per_mm2 : ℝ) : ℝ :=
let total_emf := num_cells * cell_emf in
let required_total_resistance := total_emf / min_current in
let combined_internal_resistance := 2 * cell_internal_resistance in
let total_internal_resistance := combined_internal_resistance + internal_resistance_bell in
let wire_resistance := required_total_resistance - total_internal_resistance in
let wire_length := 2 * distance in
let wire_cross_sectional_area := (λ d : ℝ, (d / 2)^2 * Real.pi) in
solve_for_diameter wire_resistance wire_length resistance_per_meter_per_mm2 wire_cross_sectional_area

theorem copper_wire_diameter :
  required_wire_diameter 30 2 1.5 1 2 0.4 (1 / 55) = 0.63 :=
sorry

end copper_wire_diameter_l96_96982


namespace worker_b_alone_time_l96_96972

theorem worker_b_alone_time (A B C : ℝ) (h1 : A + B = 1 / 8)
  (h2 : A = 1 / 12) (h3 : C = 1 / 18) :
  1 / B = 24 :=
sorry

end worker_b_alone_time_l96_96972


namespace abs_d_is_thirteen_over_two_l96_96304

theorem abs_d_is_thirteen_over_two (d : ℂ) 
    (h : (λ Q : ℂ → ℂ, Q = (λ x, (x^2 - 3 * x + 3) * (x^2 - d * x + 5) * (x^2 - 5 * x + 15))) 
    ∧ (∃ z1 z2 z3 z4 : ℂ, z1 ≠ z2 ∧ z2 ≠ z3 ∧ z3 ≠ z4 ∧ z1 ≠ z3 ∧ z1 ≠ z4 ∧ z2 ≠ z4 
    ∧ (Q z1 = 0) 
    ∧ (Q z2 = 0) 
    ∧ (Q z3 = 0) 
    ∧ (Q z4 = 0))) :
    |d| = 13 / 2 :=
sorry

end abs_d_is_thirteen_over_two_l96_96304


namespace value_f_5_7_range_of_m_l96_96051

noncomputable def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 3 then x^2 + 4 else
  if (x-3) < 1 then ((x-3) ^ 2 + 4) else (x-6) ^ 2 + 4

theorem value_f_5_7 : f 5 + f 7 = 13 :=
sorry

theorem range_of_m (m : ℝ) : (∀ x ∈ set.Icc 4 6, f x = m * x^2) → 4/13 ≤ m ∧ m ≤ 13/36 :=
sorry

end value_f_5_7_range_of_m_l96_96051


namespace limit_proof_l96_96705

variable {α : Type*} [Real α]
variables (f : α → α) (x0 : α)

noncomputable def limit_expression (f''_x0 : α) : α :=
  limit (fun (k : α) => (f(x0 - 1/2 * k) - f(x0)) / k) zero

theorem limit_proof (h : deriv (deriv f) x0 = 2) :
  limit_expression f x0 2 = -1 :=
by
  sorry

end limit_proof_l96_96705


namespace largest_number_of_blocks_l96_96225

-- Definitions for the dimensions of blocks and the box
def small_block_length := 1.5
def small_block_width := 2
def small_block_height := 2

def large_box_length := 4
def large_box_width := 6
def large_box_height := 3

-- Definition for volumes
def volume (l w h : ℝ) : ℝ := l * w * h

-- Theorem statement for the largest number of smaller blocks fitting in the larger box
theorem largest_number_of_blocks : 
  (4 * 6 * 3) / (1.5 * 2 * 2) = 12 :=
by
  calc (4 * 6 * 3) / (1.5 * 2 * 2) = 72 / 6 : by congr; norm_num
                                ... = 12 : by norm_num

end largest_number_of_blocks_l96_96225


namespace car_distance_covered_l96_96608

def distance_covered_by_car (time : ℝ) (speed : ℝ) : ℝ :=
  speed * time

theorem car_distance_covered :
  distance_covered_by_car (3 + 1/5 : ℝ) 195 = 624 :=
by
  sorry

end car_distance_covered_l96_96608


namespace order_of_logs_l96_96047

noncomputable def a := Real.logBase 2 (Real.sqrt 2)
noncomputable def b := Real.logBase (Real.sqrt 3) 2
noncomputable def c := Real.logBase 3 5

theorem order_of_logs : a < b ∧ b < c := by
  sorry

end order_of_logs_l96_96047


namespace hexagon_arith_prog_angle_l96_96926

theorem hexagon_arith_prog_angle (a d : ℝ) (h1 : 6 * a + 15 * d = 720) :
  ∃ k : ℕ, k ∈ {0, 1, 2, 3, 4, 5} ∧ a + k * d = 120 :=
by
  use 0
  simp [h1]
  exact  ⟨rfl⟩

end hexagon_arith_prog_angle_l96_96926


namespace scientific_notation_for_35000_l96_96280

-- Defining the given conditions using Lean constructs
def scientific_notation (a : ℝ) (n : ℤ) := (1 ≤ a) ∧ (a < 10) ∧ (35_000 = a * 10^n)

-- The equivalent math proof problem in Lean 4
theorem scientific_notation_for_35000 : scientific_notation 3.5 4 :=
by
  -- Assume to show the correct notation
  unfold scientific_notation
  -- Conditions to be proved
  split
  sorry  -- Proof for 1 ≤ 3.5
  split
  sorry  -- Proof for 3.5 < 10
  sorry  -- Proof for 35_000 = 3.5 * 10^4

end scientific_notation_for_35000_l96_96280


namespace decreasing_interval_log_composite_l96_96187

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem decreasing_interval_log_composite :
  ∀ x : ℝ, (∃ t : ℝ, t = 2 * x^2 - 3 * x + 1 ∧ t > 0 ∧ x ∈ (1, +∞)) →
  ∀ x1 x2 : ℝ, x1 ∈ (1, +∞) → x2 ∈ (1, +∞) → x1 ≤ x2 → log_base (1/3 : ℝ) (2*x1^2 - 3*x1 + 1) ≥ log_base (1/3 : ℝ) (2*x2^2 - 3*x2 + 1) :=
by
  intro x h1 x1 x2 hx1 hx2 hx12
  sorry

end decreasing_interval_log_composite_l96_96187


namespace probability_of_two_approvals_in_four_l96_96110

-- Conditions
def prob_approval : ℝ := 0.6
def prob_disapproval : ℝ := 1 - prob_approval

-- Proof statement
theorem probability_of_two_approvals_in_four :
  (4.choose 2) * (prob_approval^2 * prob_disapproval^2) = 0.3456 :=
by
  sorry

end probability_of_two_approvals_in_four_l96_96110


namespace milk_powder_sampling_l96_96064

theorem milk_powder_sampling (total_bags sampled_bags : ℕ) (first_sampled_bag group_number interval : ℕ) (h_total : total_bags = 3000) (h_sampled : sampled_bags = 150) (h_first : first_sampled_bag = 11) (h_group : group_number = 61) (h_interval : interval = total_bags / sampled_bags) :
  first_sampled_bag + (group_number - 1) * interval = 1211 :=
by
  rw [h_total, h_sampled] at h_interval
  simp at h_interval
  rw [h_first, h_group, h_interval]
  norm_num
  sorry

end milk_powder_sampling_l96_96064


namespace number_of_club_members_l96_96252

theorem number_of_club_members
  (num_committee : ℕ)
  (pair_of_committees_has_unique_member : ∀ (c1 c2 : Fin num_committee), c1 ≠ c2 → ∃! m : ℕ, c1 ≠ c2 ∧ c2 ≠ c1 ∧ m = m)
  (members_belong_to_two_committees : ∀ m : ℕ, ∃ (c1 c2 : Fin num_committee), c1 ≠ c2 ∧ m = m)
  : num_committee = 5 → ∃ (num_members : ℕ), num_members = 10 :=
by
  sorry

end number_of_club_members_l96_96252


namespace find_n_l96_96682

theorem find_n (n : ℕ) (h : 2^7 * 3^4 * n = nat.factorial 9) : n = 35 :=
  by sorry

end find_n_l96_96682


namespace circle_center_reflection_final_coordinates_l96_96186

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, p.fst)

def reflect_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.snd, -p.fst)

theorem circle_center_reflection_final_coordinates :
  let initial_center := (3 : ℝ, -8 : ℝ)
  let first_reflection := reflect_y_eq_x initial_center
  let final_center := reflect_y_eq_neg_x first_reflection
  final_center = (-3, 8) :=
by
  sorry

end circle_center_reflection_final_coordinates_l96_96186


namespace triangle_area_cubed_l96_96894

variables {A B C M N P : Type} [OrderedField ℝ]

def ratio_AM_MC (k : ℝ) (AM MC : ℝ) : Prop := 
  AM / MC = k

def ratio_CN_NB (k : ℝ) (CN NB : ℝ) : Prop := 
  CN / NB = k

def ratio_MP_PN (k : ℝ) (MP PN : ℝ) : Prop := 
  MP / PN = k

def area_AMP (AMP : ℝ) : ℝ := AMP
def area_BNP (BNP : ℝ) : ℝ := BNP
def area_ABC (ABC : ℝ) : ℝ := ABC

theorem triangle_area_cubed (k : ℝ) (AM MC CN NB MP PN AMP BNP ABC : ℝ) :
  ratio_AM_MC k AM MC →
  ratio_CN_NB k CN NB →
  ratio_MP_PN k MP PN →
  real.cbrt (area_AMP AMP) + real.cbrt (area_BNP BNP) = real.cbrt (area_ABC ABC) :=
sorry

end triangle_area_cubed_l96_96894


namespace tetrahedron_volume_proof_l96_96007

-- Defining the conditions
variables (A B C D : Type) [metric_space A]
variables (angle_ABC_BCD : real)
variables (area_ABC area_BCD BC : real)

-- Given conditions
def tetrahedron_conditions : Prop :=
  angle_ABC_BCD = 45 ∧ area_ABC = 150 ∧ area_BCD = 90 ∧ BC = 12

-- Prove the volume of tetrahedron ABCD is 375√2 under given conditions
theorem tetrahedron_volume_proof (A B C D : Type) [metric_space A]
  (angle_ABC_BCD : real) (area_ABC area_BCD BC : real) :
  tetrahedron_conditions A B C D angle_ABC_BCD area_ABC area_BCD BC →
  volume (A, B, C, D) = 375 * real.sqrt 2 :=
begin
  sorry
end

end tetrahedron_volume_proof_l96_96007


namespace proof_l96_96915

def main_theorem : Prop :=
  let x := (1 : ℚ) / 2 in
  5 * x^2 - (x^2 - 2 * (2 * x - 3)) = -3

theorem proof : main_theorem :=
by
  sorry

end proof_l96_96915


namespace solve_quadratic_equation_l96_96183

theorem solve_quadratic_equation : 
  ∃ (a b c : ℤ), (0 < a) ∧ (64 * x^2 + 48 * x - 36 = 0) ∧ ((a * x + b)^2 = c) ∧ (a + b + c = 56) := 
by
  sorry

end solve_quadratic_equation_l96_96183


namespace ratio_correct_l96_96947

theorem ratio_correct : 
    (2^17 * 3^19) / (6^18) = 3 / 2 :=
by sorry

end ratio_correct_l96_96947


namespace nested_radical_eq_5_l96_96692

noncomputable def nested_radical : ℝ :=
  Inf { x | x = Real.sqrt (20 + x)}

theorem nested_radical_eq_5 : ∃ x : ℝ, nested_radical = 5 :=
by
  sorry

end nested_radical_eq_5_l96_96692


namespace fertilizer_proportion_l96_96258

theorem fertilizer_proportion :
  ∀ (F T A : ℕ), A = 9600 → T = 5600 → F = 700 → (F * A = T * 1200) :=
by
  intros F T A hA hT hF
  rw [hA, hT, hF]
  sorry

end fertilizer_proportion_l96_96258


namespace locus_of_point_P_l96_96621

-- Definitions and conditions
def circle_M (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 4
def A_point : ℝ × ℝ := (2, 1)
def chord_BC (x y x₀ y₀ : ℝ) : Prop := (x₀ - 1) * x + y₀ * y - x₀ - 3 = 0
def point_P_locus (x₀ y₀ : ℝ) : Prop := ∃ x y, (chord_BC x y x₀ y₀) ∧ x = 2 ∧ y = 1

-- Lean 4 statement to be proved
theorem locus_of_point_P (x₀ y₀ : ℝ) (h : point_P_locus x₀ y₀) : x₀ + y₀ - 5 = 0 :=
  by
  sorry

end locus_of_point_P_l96_96621


namespace lcm_of_numbers_l96_96327

/-- Define the numbers involved -/
def a := 456
def b := 783
def c := 935
def d := 1024
def e := 1297

/-- Prove the LCM of these numbers is 2308474368000 -/
theorem lcm_of_numbers :
  Int.lcm (Int.lcm (Int.lcm (Int.lcm a b) c) d) e = 2308474368000 :=
by
  sorry

end lcm_of_numbers_l96_96327


namespace points_below_line_l96_96358

theorem points_below_line (d q x1 x2 y1 y2 : ℝ) 
  (h1 : 2 = 1 + 3 * d)
  (h2 : x1 = 1 + d)
  (h3 : x2 = x1 + d)
  (h4 : 2 = q ^ 3)
  (h5 : y1 = q)
  (h6 : y2 = q ^ 2) :
  x1 > y1 ∧ x2 > y2 :=
by {
  sorry
}

end points_below_line_l96_96358


namespace evaluate_expression_l96_96002

theorem evaluate_expression : 
  (3^2015 + 3^2013 + 3^2012) / (3^2015 - 3^2013 + 3^2012) = 31 / 25 :=
by
  sorry

end evaluate_expression_l96_96002


namespace performance_comparison_l96_96803

def scores_A : List ℕ := [89, 93, 88, 91, 94, 90, 88, 87]
def scores_B : List ℕ := [92, 90, 85, 93, 95, 86, 87, 92]

noncomputable def range (scores : List ℕ) : ℕ :=
  List.maximum scores - List.minimum scores

noncomputable def average (scores : List ℕ) : ℚ :=
  (List.sum scores : ℚ) / scores.length

noncomputable def mode (scores : List ℕ) : ℕ :=
  scores.groupBy id |> List.maximumBy fun g => g.length |> List.head!

noncomputable def median (scores : List ℕ) : ℚ :=
  let sorted := scores.quicksort
  if scores.length % 2 = 0 then
    (sorted[scores.length/2 - 1] + sorted[scores.length/2]) / 2
  else
    sorted[scores.length / 2]

noncomputable def variance (scores : List ℕ) : ℚ :=
  let avg := average scores
  (List.sum (scores.map fun x => (x - avg).pow 2) : ℚ) / scores.length

theorem performance_comparison :
  range scores_B > range scores_A ∧
  average scores_A = average scores_B ∧
  mode scores_B > mode scores_A ∧
  median scores_B > median scores_A ∧
  variance scores_A < variance scores_B :=
by sorry

end performance_comparison_l96_96803


namespace profit_correct_A_B_l96_96251

noncomputable def profit_per_tire_A (batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A : ℕ) : ℚ :=
  let cost_first_5000 := batch_cost_A1 + (cost_per_tire_A1 * 5000)
  let revenue_first_5000 := sell_price_tire_A1 * 5000
  let profit_first_5000 := revenue_first_5000 - cost_first_5000
  let cost_remaining := batch_cost_A2 + (cost_per_tire_A2 * (produced_A - 5000))
  let revenue_remaining := sell_price_tire_A2 * (produced_A - 5000)
  let profit_remaining := revenue_remaining - cost_remaining
  let total_profit := profit_first_5000 + profit_remaining
  total_profit / produced_A

noncomputable def profit_per_tire_B (batch_cost_B cost_per_tire_B sell_price_tire_B produced_B : ℕ) : ℚ :=
  let cost := batch_cost_B + (cost_per_tire_B * produced_B)
  let revenue := sell_price_tire_B * produced_B
  let profit := revenue - cost
  profit / produced_B

theorem profit_correct_A_B
  (batch_cost_A1 : ℕ := 22500) 
  (batch_cost_A2 : ℕ := 20000) 
  (cost_per_tire_A1 : ℕ := 8) 
  (cost_per_tire_A2 : ℕ := 6) 
  (sell_price_tire_A1 : ℕ := 20) 
  (sell_price_tire_A2 : ℕ := 18) 
  (produced_A : ℕ := 15000)
  (batch_cost_B : ℕ := 24000) 
  (cost_per_tire_B : ℕ := 7) 
  (sell_price_tire_B : ℕ := 19) 
  (produced_B : ℕ := 10000) :
  profit_per_tire_A batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A = 9.17 ∧
  profit_per_tire_B batch_cost_B cost_per_tire_B sell_price_tire_B produced_B = 9.60 :=
by
  sorry

end profit_correct_A_B_l96_96251


namespace prism_properties_l96_96270

noncomputable def side_length_of_base := 1.5
noncomputable def height_of_prism := 1.5
noncomputable def base_area := side_length_of_base ^ 2
noncomputable def lateral_surface_area := 6 ^ 2
noncomputable def total_surface_area := 2 * base_area + lateral_surface_area
noncomputable def volume := base_area * height_of_prism

theorem prism_properties :
  total_surface_area = 40.5 ∧ volume = 3.375 :=
by
  sorry

end prism_properties_l96_96270


namespace cyclic_quadrilateral_circumradius_cyclic_quadrilateral_inequality_l96_96111

variable {a b c d Q R : ℝ}

/-- Given a cyclic quadrilateral ABCD with side lengths a, b, c, d, area Q, and circumradius R, 
    prove that R^2 = ( (ab + cd) (ac + bd) (ad + bc) ) / (16 Q^2). -/
theorem cyclic_quadrilateral_circumradius (h1 : isCyclicQuadrilateral a b c d Q R) :
    R^2 = ((a * b + c * d) * (a * c + b * d) * (a * d + b * c)) / (16 * Q^2) :=
sorry

/-- Given the same conditions as the above, deduce that R ≥ ( (abcd)^(3/4) ) / ( Q √2 ),
    with equality if and only if ABCD is a square. -/
theorem cyclic_quadrilateral_inequality (h1 : isCyclicQuadrilateral a b c d Q R) :
    R ≥ ((a * b * c * d)^(3 / 4)) / (Q * Real.sqrt 2) :=
sorry

end cyclic_quadrilateral_circumradius_cyclic_quadrilateral_inequality_l96_96111


namespace projection_of_a_onto_b_l96_96763

namespace VectorProjection

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def scalar_projection (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

theorem projection_of_a_onto_b :
  scalar_projection (1, -2) (3, 4) = -1 := by
    sorry

end VectorProjection

end projection_of_a_onto_b_l96_96763


namespace range_of_a_l96_96390

variable (a : ℝ)

def p := ∀ x : ℝ, x^2 + a ≥ 0
def q := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (h : p a ∧ q a) : 0 ≤ a :=
by
  sorry

end range_of_a_l96_96390


namespace total_distance_travelled_l96_96236

-- Definitions and propositions
def distance_first_hour : ℝ := 15
def distance_second_hour : ℝ := 18
def distance_third_hour : ℝ := 1.25 * distance_second_hour

-- Conditions based on the problem
axiom second_hour_distance : distance_second_hour = 18
axiom second_hour_20_percent_more : distance_second_hour = 1.2 * distance_first_hour
axiom third_hour_25_percent_more : distance_third_hour = 1.25 * distance_second_hour

-- Proof of the total distance James traveled
theorem total_distance_travelled : 
  distance_first_hour + distance_second_hour + distance_third_hour = 55.5 :=
by
  sorry

end total_distance_travelled_l96_96236


namespace race_dead_heat_l96_96574

theorem race_dead_heat 
  (L Vb : ℝ) 
  (speed_a : ℝ := (16/15) * Vb)
  (speed_c : ℝ := (20/15) * Vb) 
  (time_a : ℝ := L / speed_a)
  (time_b : ℝ := L / Vb)
  (time_c : ℝ := L / speed_c) :
  (1 / (16 / 15) = 3 / 4) → 
  (1 - 3 / 4) = 1 / 4 :=
by 
  sorry

end race_dead_heat_l96_96574


namespace ternary_to_decimal_l96_96667

def to_decimal (ternary : Nat) : Nat :=
  match ternary with
  | 121 => 1 * 3^2 + 2 * 3^1 + 1 * 3^0
  | _ => 0

theorem ternary_to_decimal : to_decimal 121 = 16 := by
  sorry

end ternary_to_decimal_l96_96667


namespace value_of_f_f_2_l96_96745

def f (x : ℤ) : ℤ := 4 * x^2 + 2 * x - 1

theorem value_of_f_f_2 : f (f 2) = 1481 := by
  sorry

end value_of_f_f_2_l96_96745


namespace solve_equation_l96_96916

-- Definitions for the problem conditions
def right_hand_side_non_negative (x : ℝ) : Prop := 4 * sin x - real.sqrt 3 ≥ 0

-- Definition of the equation
def equation (x : ℝ) : Prop :=
  real.sqrt (3 + 4 * real.sqrt 6 - (16 * real.sqrt 3 - 8 * real.sqrt 2) * sin x) = 
  4 * sin x - real.sqrt 3

-- Final Lean statement combining the above definitions and the solution
theorem solve_equation (x : ℝ) (k : ℤ) (h : right_hand_side_non_negative x) : 
  equation x ↔ x = (-1)^k * (π/4) + 2*k*π :=
sorry

end solve_equation_l96_96916


namespace base10_to_base5_addition_l96_96229

-- Let's define the problem statement and conditions

def value_of_32_plus_45_in_base5 : ℕ := 302

theorem base10_to_base5_addition :
  let n1 : ℕ := 32
  let n2 : ℕ := 45
  n1 + n2 = 77 ∧ 77 = 3*25 + 0*5 + 2 →
  nat.digits 5 (n1 + n2) = [2, 0, 3] →
  nat.of_digits 5 [2, 0, 3] = value_of_32_plus_45_in_base5 :=
by
  sorry

end base10_to_base5_addition_l96_96229


namespace sufficient_but_not_necessary_condition_for_q_l96_96059

def proposition_p (a : ℝ) := (1 / a) > (1 / 4)
def proposition_q (a : ℝ) := ∀ x : ℝ, (a * x^2 + a * x + 1) > 0

theorem sufficient_but_not_necessary_condition_for_q (a : ℝ) :
  proposition_p a → proposition_q a → (∃ a : ℝ, 0 < a ∧ a < 4) ∧ (∃ a : ℝ, 0 < a ∧ a < 4 ∧ ¬ proposition_p a) 
  := sorry

end sufficient_but_not_necessary_condition_for_q_l96_96059


namespace angle_C_120_l96_96368

theorem angle_C_120 (a b c : ℝ) (h : (a + b - c) * (a + b + c) = a * b) :
  ∠C = 120 :=
sorry

end angle_C_120_l96_96368


namespace remainder_of_sum_of_powers_div_2_l96_96558

theorem remainder_of_sum_of_powers_div_2 : 
  (1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7 + 8^8 + 9^9) % 2 = 1 :=
by 
  sorry

end remainder_of_sum_of_powers_div_2_l96_96558


namespace compute_z_power_12_l96_96867

def z : ℂ := (-1 + complex.I * real.sqrt 3) / 2

theorem compute_z_power_12 : z^12 = 1 := by
  sorry

end compute_z_power_12_l96_96867


namespace problem_statement_l96_96094

variable (m : ℝ) -- We declare m as a real number

theorem problem_statement (h : m + 1/m = 10) : m^2 + 1/m^2 + 4 = 102 := 
by 
  sorry -- The proof is omitted

end problem_statement_l96_96094


namespace simultaneous_equations_solution_l96_96222

theorem simultaneous_equations_solution (x y : ℚ) :
  3 * x^2 + x * y - 2 * y^2 = -5 ∧ x^2 + 2 * x * y + y^2 = 1 ↔ 
  (x = 3/5 ∧ y = -8/5) ∨ (x = -3/5 ∧ y = 8/5) :=
by
  sorry

end simultaneous_equations_solution_l96_96222


namespace Misha_probability_l96_96596

open Probability

-- Definitions
def classesMonday := 5
def classesTuesday := 6
def totalClasses := 11
def totalCorrect := 7
def mondayCorrect := 3

-- Calculating binomial probabilities
noncomputable def P_A1 := (binomial classesMonday mondayCorrect) * (1 / 2) ^ classesMonday
noncomputable def P_A2 := (binomial classesTuesday (totalCorrect - mondayCorrect)) * (1 / 2) ^ classesTuesday
noncomputable def P_B := (binomial totalClasses totalCorrect) * (1 / 2) ^ totalClasses

-- Theorem stating the required probability
theorem Misha_probability :
  P_A1 * P_A2 / P_B = 5 / 11 :=
by
  sorry

end Misha_probability_l96_96596


namespace evaluate_fraction_l96_96462

theorem evaluate_fraction (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - b * (1 / a) ≠ 0) :
  (a^2 - 1 / b^2) / (b^2 - 1 / a^2) = a^2 / b^2 :=
by
  sorry

end evaluate_fraction_l96_96462


namespace mod_equiv_1_l96_96505

theorem mod_equiv_1 : (179 * 933 / 7) % 50 = 1 := by
  sorry

end mod_equiv_1_l96_96505


namespace total_earning_correct_l96_96235

noncomputable def Wa := 3 * (100 / 5)
noncomputable def Wb := 4 * (100 / 5)
noncomputable def Wc := 100

def da := 6
def db := 9
def dc := 4

def Ta := da * Wa
def Tb := db * Wb
def Tc := dc * Wc

def TotalEarning := Ta + Tb + Tc

theorem total_earning_correct : TotalEarning = 1480 :=
by {
    sorry
}

end total_earning_correct_l96_96235


namespace solve_system_l96_96918

theorem solve_system :
  ∃ (x y : ℝ), (x^2 + y^2 ≤ 1) ∧ (x^4 - 18 * x^2 * y^2 + 81 * y^4 - 20 * x^2 - 180 * y^2 + 100 = 0) ∧
    ((x = -1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = -1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10)) :=
  by
  sorry

end solve_system_l96_96918


namespace neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l96_96649

-- (1) Prove -(-2) = 2
theorem neg_neg_two : -(-2) = 2 := 
sorry

-- (2) Prove -6 + 6 = 0
theorem neg_six_plus_six : -6 + 6 = 0 := 
sorry

-- (3) Prove (-3) * 5 = -15
theorem neg_three_times_five : (-3) * 5 = -15 := 
sorry

-- (4) Prove 2x - 3x = -x
theorem two_x_minus_three_x (x : ℝ) : 2 * x - 3 * x = - x := 
sorry

end neg_neg_two_neg_six_plus_six_neg_three_times_five_two_x_minus_three_x_l96_96649


namespace angle_ABD_is_130_degrees_l96_96983

theorem angle_ABD_is_130_degrees
  (A B C D : Type)
  [InnerProductSpace ℝ A]
  [InnerProductSpace ℝ B]
  [InnerProductSpace ℝ C]
  [InnerProductSpace ℝ D]
  (h1 : AC = BC)
  (h2 : ∠C = 80)
  (h3 : D ∈ Line BA ∧ BA ∉ Ray A)

  : m∠ ABD = 130 :=
sorry

end angle_ABD_is_130_degrees_l96_96983


namespace uncle_ben_eggs_l96_96548

def number_of_eggs (C R N E : ℕ) :=
  let hens := C - R
  let laying_hens := hens - N
  laying_hens * E

-- Given:
-- C = 440 (Total number of chickens)
-- R = 39 (Number of roosters)
-- N = 15 (Non-laying hens)
-- E = 3 (Eggs per laying hen)
theorem uncle_ben_eggs : number_of_eggs 440 39 15 3 = 1158 := by
  unfold number_of_eggs
  calc
    403 - 15 * 3 : sorry

end uncle_ben_eggs_l96_96548


namespace reflection_correct_l96_96021

def reflect_over_vector (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (x y : ℝ × ℝ) := x.1 * y.1 + x.2 * y.2
  let scale_vector (c : ℝ) (x : ℝ × ℝ) : ℝ × ℝ := (c * x.1, c * x.2)
  let add_vectors (x y : ℝ × ℝ) : ℝ × ℝ := (x.1 + y.1, x.2 + y.2)
  let subtract_vectors (x y : ℝ × ℝ) : ℝ × ℝ := (x.1 - y.1, x.2 - y.2)
  let proj_onto (u v : ℝ × ℝ) : ℝ × ℝ :=
    scale_vector (dot_product u v / dot_product v v) v
  let p := proj_onto u v
  subtract_vectors (scale_vector 2 p) u

theorem reflection_correct :
  reflect_over_vector (3, -2) (2, -1) = (17 / 5, -6 / 5) :=
by
  -- Skipping the actual proof steps required
  sorry

end reflection_correct_l96_96021


namespace hooked_red_triangles_even_l96_96952

theorem hooked_red_triangles_even (yellow_points : Fin 3 → ℝ × ℝ × ℝ)
  (red_points : Fin 40 → ℝ × ℝ × ℝ)
  (h_not_coplanar : ∀ (p1 p2 p3 p4 : Fin 43),
      let points := (List.ofFn yellow_points).append (List.ofFn red_points)
      in affine_independent ℝ (points.get p1 :: points.get p2 :: points.get p3 :: points.get p4 :: [])) :
  ∀ (number_of_hooked_triangles : ℕ),
    (number_of_hooked_triangles % 2 = 0) → ¬(number_of_hooked_triangles = 2023) :=
by
  sorry

end hooked_red_triangles_even_l96_96952


namespace simplify_tan_product_l96_96911

noncomputable def tan_deg (d : ℝ) : ℝ := Real.tan (d * Real.pi / 180)

theorem simplify_tan_product :
  (1 + tan_deg 10) * (1 + tan_deg 35) = 2 := 
by
  -- Given conditions
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have h2 : tan_deg 10 + tan_deg 35 = 1 - tan_deg 10 * tan_deg 35 :=
    by sorry -- Use tan addition formula here
  -- Proof of the theorem follows from here
  sorry

end simplify_tan_product_l96_96911


namespace simplify_tangent_expression_l96_96912

theorem simplify_tangent_expression :
  (1 + Real.tan (Real.pi / 18)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  sorry

end simplify_tangent_expression_l96_96912


namespace sin_squared_theta_half_l96_96095

theorem sin_squared_theta_half (θ : ℝ) 
  (cond : cos θ ^ 4 + sin θ ^ 4 + (cos θ * sin θ) ^ 4 + 1 / (cos θ ^ 4 + sin θ ^ 4) = 41 / 16) : 
  sin θ ^ 2 = 1 / 2 :=
sorry

end sin_squared_theta_half_l96_96095


namespace largest_central_symmetric_area_smallest_central_symmetric_area_l96_96056

-- Define the conditions
variable (T : Triangle) (hT : T.area = 1)

-- Part (a)
theorem largest_central_symmetric_area (m : Polygon) 
  (h1 : m.is_central_symmetric)
  (h2 : m.inscribed_in T) :
  m.area = (2 : ℝ) / 3 :=
sorry

-- Part (b)
theorem smallest_central_symmetric_area (M : Polygon)
  (h1 : M.is_central_symmetric)
  (h2 : M.circumscribes T) :
  M.area = 2 :=
sorry

end largest_central_symmetric_area_smallest_central_symmetric_area_l96_96056


namespace circumscribed_circle_radius_isosceles_trapezoid_l96_96690

theorem circumscribed_circle_radius_isosceles_trapezoid:
  ∀ (a b l : ℝ), a = 2 → b = 14 → l = 10 →
  ∃ (R : ℝ), R = 5 * Real.sqrt 2 :=
by
  intros a b l ha hb hl
  use 5 * Real.sqrt 2
  sorry

end circumscribed_circle_radius_isosceles_trapezoid_l96_96690


namespace quadratic_solution_transform_l96_96190

theorem quadratic_solution_transform (a b c : ℝ) (hA : 0 = a * (-3)^2 + b * (-3) + c) (hB : 0 = a * 4^2 + b * 4 + c) :
  (∃ x1 x2 : ℝ, a * (x1 - 1)^2 + b * (x1 - 1) + c = 0 ∧ a * (x2 - 1)^2 + b * (x2 - 1) + c = 0 ∧ x1 = -2 ∧ x2 = 5) :=
  sorry

end quadratic_solution_transform_l96_96190


namespace relationship_among_a_b_c_l96_96347

noncomputable def a := ∫ x in (0 : ℝ)..(1 : ℝ), x
noncomputable def b := ∫ x in (0 : ℝ)..(1 : ℝ), x^2
noncomputable def c := ∫ x in (0 : ℝ)..(1 : ℝ), real.sqrt x

theorem relationship_among_a_b_c : b < a ∧ a < c := by 
  sorry

end relationship_among_a_b_c_l96_96347


namespace root_exists_in_between_l96_96456

theorem root_exists_in_between
  {a b c r s : ℝ}
  (h_r : a * r^2 + b * r + c = 0)
  (h_s : -a * s^2 + b * s + c = 0) :
  ∃ t : ℝ, t > r ∧ t < s ∧ (a / 2) * t^2 + b * t + c = 0 :=
begin
  sorry
end

end root_exists_in_between_l96_96456


namespace parity_of_standard_pairs_l96_96921

open Nat

theorem parity_of_standard_pairs 
  (m n : ℕ) (h_m : m ≥ 3) (h_n : n ≥ 3) 
  (colors : Fin m → Fin n → Bool) :
  let S := ∑ i in range m, ∑ j in range n, (if (i + j) % 2 = 1 then 1 else 0) in
  let edge_squares := {i | i.1 = 0 ∨ i.1 = m-1 ∨ i.2 = 0 ∨ i.2 = n-1} in
  let blue_edge_squares := ∑ i in edge_squares, if colors i.1 i.2 = false then 1 else 0 in
  (S % 2 = 0) ↔ (blue_edge_squares % 2 = 0) :=
  sorry

end parity_of_standard_pairs_l96_96921


namespace proof_problem_max_area_of_CEFD_l96_96715

-- Define the condition of the moving point Q and the fixed points
def moving_point_Q (P Q : (ℝ × ℝ)) :=
 ∃ P1 P2 : (ℝ × ℝ), P1 = (-√2, 0) ∧ P2 = (√2, 0) ∧
   let (x, y) := Q in  -- Coordinates of Q
   let (x1, y1) := P1 in -- Coordinates of fixed point 1
   let (x2, y2) := P2 in -- Coordinates of fixed point 2
   (y / (x + x1)) * (y / (x - x2)) = -1 / 2

-- Define the condition of line l passing through (-2, 0) and its intersections with M
def line_condition (l : ℝ → ℝ) (P : (ℝ × ℝ)) :=
  ∃ m n : ℝ, let (x, y) := P in x = -2 ∧ y = 0 ∧
  ∀ Q : (ℝ × ℝ), moving_point_Q P Q → l (Q.2) = m * Q.2 + n

-- Define the trajectory M and its equation
def trajectory_M (Q : (ℝ × ℝ)) :=
 moving_point_Q (-√2, 0) Q ∧ moving_point_Q (√2, 0) Q

-- Define the maximum area of quadrilateral CEFD
noncomputable def max_area_CEFD :=
  let m := 2 in
  let n := √3 in
  (2 * √2 / 9) * √(n ^ 2 * (6 - n ^ 2))

-- The main theorem
theorem proof_problem :
  ∀ Q : (ℝ × ℝ), ∃ P1 P2 : (ℝ × ℝ), P1 = (-√2, 0) ∧ P2 = (√2, 0) ∧
  let (x, y) := Q in
  (y / (x + P1.1)) * (y / (x - P2.1)) = -1 / 2 → trajectory_M Q :=
by
   sorry

theorem max_area_of_CEFD :
  (∃ l : ℝ → ℝ, line_condition l (-2,0)) →
  max_area_CEFD = 2 * √2 / 3 :=
by
   sorry

end proof_problem_max_area_of_CEFD_l96_96715


namespace perimeter_of_rectangle_l96_96625

-- Define the given conditions
variables {b l : ℝ}

-- Condition 1: The length is thrice the breadth
def length_is_thrice_breadth : Prop := l = 3 * b

-- Condition 2: The area of the rectangle is 108 square meters
def area_is_108 : Prop := l * b = 108

-- The problem: Prove the perimeter is 48 meters
theorem perimeter_of_rectangle (h1 : length_is_thrice_breadth) (h2 : area_is_108) : 2 * l + 2 * b = 48 :=
sorry

end perimeter_of_rectangle_l96_96625


namespace prove_k_range_l96_96048

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x - b * Real.log x

theorem prove_k_range (a b k : ℝ) (h1 : a - b = 1) (h2 : f 1 a b = 2) :
  (∀ x ≥ 1, f x a b ≥ k * x) → k ≤ 2 - 1 / Real.exp 1 :=
by
  sorry

end prove_k_range_l96_96048


namespace seth_spent_more_on_ice_cream_l96_96490

-- Definitions based on the conditions
def cartons_ice_cream := 20
def cartons_yogurt := 2
def cost_per_carton_ice_cream := 6
def cost_per_carton_yogurt := 1

-- Theorem statement
theorem seth_spent_more_on_ice_cream :
  (cartons_ice_cream * cost_per_carton_ice_cream) - (cartons_yogurt * cost_per_carton_yogurt) = 118 :=
by
  sorry

end seth_spent_more_on_ice_cream_l96_96490


namespace sqrt2_times_sqrt5_eq_sqrt10_l96_96563

theorem sqrt2_times_sqrt5_eq_sqrt10 : (Real.sqrt 2) * (Real.sqrt 5) = Real.sqrt 10 := 
by
  sorry

end sqrt2_times_sqrt5_eq_sqrt10_l96_96563


namespace triangle_tangent_sum_zero_l96_96818

theorem triangle_tangent_sum_zero
    (A B C D : Type)
    [Inhabited A] [Plane A]
    (BC : LineSegment B C)
    (D_midpoint : midpoint D BC)
    (AD AC : Vector A)
    (orthogonal_AD_AC : dot_product AD AC = 0) :
    (tan A + 2 * tan C = 0) :=
sorry

end triangle_tangent_sum_zero_l96_96818


namespace distance_P_to_T_l96_96808

theorem distance_P_to_T :
  ∀ (P Q R S T U: Type) 
  [metric_space P] [metric_space Q] [metric_space R] [metric_space S] [metric_space T],
  PQ ⟂ QR →
  QR ⟂ RS →
  RS ⟂ ST →
  PQ = 4 →
  QR = 8 →
  RS = 8 →
  ST = 3 →
  dist P T = 13 := sorry

end distance_P_to_T_l96_96808


namespace segment_lengths_unique_l96_96726

def cutting_ratio (p q : ℕ) (x : ℝ) : ℝ → ℝ := 
  if p = q then 1
  else if p > q then Real.log x / Real.log (p / (p + q)) + 1
  else Real.log x / Real.log (q / (p + q)) + 1

noncomputable def a (p q : ℕ) (x : ℝ) : ℝ :=
  if p = q then 1 
  else if p > q then Nat.ceil (Real.log x / Real.log (p / (p + q))) + 1
  else Nat.ceil (Real.log x / Real.log (q / (p + q))) + 1

theorem segment_lengths_unique (p q : ℕ) (x : ℝ) : a(p, q, x) = 
  if p = q then 1
  else if p > q then Nat.ceil (Real.log x / Real.log (p / (p + q))) + 1
  else Nat.ceil (Real.log x / Real.log (q / (p + q))) + 1 := 
by 
  -- skipping the proof
  sorry

end segment_lengths_unique_l96_96726


namespace average_temperature_Addington_l96_96290

def average_temperature (temps : List Int) : Real :=
  (temps.sum : Real) / temps.length

theorem average_temperature_Addington :
  let temps := [40, 47, 45, 41, 39]
  average_temperature temps ≈ 42.4 := 
by
  let temps := [40, 47, 45, 41, 39]
  show average_temperature temps ≈ 42.4
  sorry

end average_temperature_Addington_l96_96290


namespace base_7_divisible_by_19_l96_96184

theorem base_7_divisible_by_19 (x : ℕ) (h : x ∈ {0, 1, 2, 3, 4, 5, 6}) :
  let n := 5 * 7^3 + 2 * 7^2 + x * 7 + 4
  n % 19 = 0 ↔ x = 4 :=
by
  have n_val : n = 7 * x + 1817 := rfl
  sorry

end base_7_divisible_by_19_l96_96184


namespace arithmetic_sequence_100th_term_diff_l96_96283

theorem arithmetic_sequence_100th_term_diff :
  ∀ (a d : ℝ), (∀ n : ℕ, n < 300 → (20 ≤ a + n * d ∧ a + n * d ≤ 200)) →
  (∑ i in finset.range 300, (a + i * d) = 15000) →
  let L := (a + 99 * d) - 201 * (30 / 299),
  let G := (a + 99 * d) + 201 * (30 / 299) in
  G - L = 2 * (6030 / 299) :=
begin
  -- Proof goes here
  sorry
end

end arithmetic_sequence_100th_term_diff_l96_96283


namespace extra_interest_l96_96994

def principal : ℝ := 7000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def interest (P R T : ℝ) : ℝ := P * R * T

theorem extra_interest :
  interest principal rate1 time - interest principal rate2 time = 840 := by
  sorry

end extra_interest_l96_96994


namespace cosine_of_angle_between_vectors_l96_96761

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

theorem cosine_of_angle_between_vectors 
  (a b : ℝ)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (h : ⟪a, a + 2 • b⟫ = 0) : 
  real.cos θ = -1/2 := 
sorry

end cosine_of_angle_between_vectors_l96_96761


namespace chess_amateurs_count_l96_96216

theorem chess_amateurs_count (n : ℕ) (h1 : ∀ i : ℕ, i < n → (∀ j : ℕ, j < n → i ≠ j → n = 15)) (h2 : ∑ i in finset.range n, finset.card ({j | j ∈ finset.range n ∧ j ≠ i}) = 45) : n = 10 :=
sorry

end chess_amateurs_count_l96_96216


namespace silver_excess_in_third_chest_l96_96281

theorem silver_excess_in_third_chest :
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ),
    x1 + x2 + x3 = 40 →
    y1 + y2 + y3 = 40 →
    x1 = y1 + 7 →
    y2 = x2 - 15 →
    y3 = x3 + 22 :=
by
  intros x1 y1 x2 y2 x3 y3 h1 h2 h3 h4
  sorry

end silver_excess_in_third_chest_l96_96281


namespace find_y_l96_96680

-- Define the condition that log_y 8 = log_64 4
def log_condition (y : ℝ) : Prop := log y 8 = log 64 4

-- The theorem to prove
theorem find_y (y : ℝ) (h : log_condition y) : y = 512 :=
by
  -- proof to be provided
  sorry

end find_y_l96_96680


namespace inequality_proof_l96_96457

def u (a : ℝ) (n : ℕ) : ℝ := (3 / 2^(n + 1)) * (-1)^(Int.floor (2^(n + 1) * a))
def v (a : ℝ) (n : ℕ) : ℝ := (3 / 2^(n + 1)) * (-1)^(n + Int.floor (2^(n + 1) * a))

theorem inequality_proof (a : ℝ) (h : 1 / 2 ≤ a ∧ a ≤ 3 / 2) :
  ((∑ n in Finset.range 2019, u a n) ^ 2 + (∑ n in Finset.range 2019, v a n) ^ 2) ≤
    72 * a ^ 2 - 48 * a + 10 + 2 / (4 ^ 2019) :=
sorry

end inequality_proof_l96_96457


namespace race_course_length_proof_l96_96973

def race_course_length (L : ℝ) (v_A v_B : ℝ) : Prop :=
  v_A = 4 * v_B ∧ (L / v_A = (L - 66) / v_B) → L = 88

theorem race_course_length_proof (v_A v_B : ℝ) : race_course_length 88 v_A v_B :=
by 
  intros
  sorry

end race_course_length_proof_l96_96973


namespace trihedral_plane_angles_acute_l96_96170

theorem trihedral_plane_angles_acute {α β γ : ℝ} (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) :
  ∀ α' β' γ' : ℝ, plane_angles_of_trihedral α β γ α' β' γ' → 
    (0 < α' ∧ α' < π / 2) ∧ (0 < β' ∧ β' < π / 2) ∧ (0 < γ' ∧ γ' < π / 2) :=
sorry

end trihedral_plane_angles_acute_l96_96170


namespace part_a_odd_points_always_even_odd_points_l96_96832

-- Definition of a point on a circumference
def Point := ℕ

-- Definition that Juquinha places points on a circumference and connects some of them
def connects (n : ℕ) : Point → Point → Bool := sorry

-- Definition of degree of a point (number of connecting points)
def degree (c : ℕ → ℕ → Bool) (p : ℕ) : ℕ := sorry

-- Definition of odd-point based on degree
def is_odd_point (c : ℕ → ℕ → Bool) (p : ℕ) : Bool :=
  odd (degree c p)

-- Part (a): Prove that exactly two points are odd
theorem part_a_odd_points (c : ℕ → ℕ → Bool) (h1: ∀ i j, c i j → i ≠ j)
  (h2 : ∀ i j, (i ≤ 4 ∧ j ≤ 4) → c i j = true)
  (h3 : ∃ i j, c i j = false) :
  2 = cardinality {p | is_odd_point c p} := sorry

-- Part (c): Prove that the number of odd-points is always even
theorem always_even_odd_points (c : ℕ → ℕ → Bool) :
  even (cardinality {p | is_odd_point c p}) := sorry

end part_a_odd_points_always_even_odd_points_l96_96832


namespace find_point_B_l96_96374

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-2, 5⟩

def is_parallel_to_x_axis (p1 p2 : Point) : Prop := p1.y = p2.y

def length (p1 p2 : Point) : ℝ := real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem find_point_B (B : Point) (h_parallel : is_parallel_to_x_axis A B) (h_length : length A B = 3) :
  B = ⟨-5, 5⟩ ∨ B = ⟨1, 5⟩ := by
  sorry

end find_point_B_l96_96374


namespace total_attended_ball_lt_fifty_l96_96852

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l96_96852


namespace arithmetic_sequence_a15_l96_96364

theorem arithmetic_sequence_a15 (a_n S_n : ℕ → ℝ) (a_9 : a_n 9 = 4) (S_15 : S_n 15 = 30) :
  let a_1 := (-12 : ℝ)
  let d := (2 : ℝ)
  a_n 15 = 16 :=
by
  sorry

end arithmetic_sequence_a15_l96_96364


namespace intersecting_triangle_perimeter_l96_96961

-- Define the triangle with given side lengths
structure Triangle :=
  (PR QR PQ : ℝ)
  (h1 : PR = 180)
  (h2 : QR = 240)
  (h3 : PQ = 200)

-- Define the lengths of the intersections within the triangle
structure Intersections :=
  (m_P m_Q m_R : ℝ)
  (h4 : m_P = 60)
  (h5 : m_Q = 40)
  (h6 : m_R = 20)

-- Noncomputable definition to avoid focusing on the specific lengths in computation.
noncomputable def perimeter_of_intersecting_triangle (T : Triangle) (I : Intersections) : ℝ :=
  45 + 44.44 + 24  -- Directly applying the calculated perimeter components

-- The main theorem we need to prove
theorem intersecting_triangle_perimeter : 
  ∀ (T : Triangle) (I : Intersections), 
    T.h1 → T.h2 → T.h3 → I.h4 → I.h5 → I.h6 → 
    perimeter_of_intersecting_triangle T I = 113.44 :=
sorry

end intersecting_triangle_perimeter_l96_96961


namespace ball_attendance_l96_96866

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l96_96866


namespace trigonometric_identity_l96_96345

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.sin (2 * x + Real.pi / 5) = Real.sqrt 3 / 3) : 
  Real.sin (4 * Real.pi / 5 - 2 * x) + Real.sin (3 * Real.pi / 10 - 2 * x)^2 = (2 + Real.sqrt 3) / 3 :=
by
  sorry

end trigonometric_identity_l96_96345


namespace sum_eq_m_square_l96_96788

theorem sum_eq_m_square 
  (K M : ℕ)
  (h1 : 1 + 2 + … + K = M^2)
  (h2 : K * (K + 1) = 4 * M^2)
  (h3 : M < 30) :
  K = 7 :=
by
  -- Proof to be constructed here
  sorry

end sum_eq_m_square_l96_96788


namespace angle_between_a_and_d_is_ninety_degrees_l96_96874

def vector_a : ℝ × ℝ × ℝ := (2, -3, 4)
def vector_b : ℝ × ℝ × ℝ := (real.sqrt 3, 5, -2)
def vector_c : ℝ × ℝ × ℝ := (11, -6, 23)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def scalar_multiply (s : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (s * v.1, s * v.2, s * v.3)

noncomputable def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

noncomputable def vector_d : ℝ × ℝ × ℝ :=
  vector_sub (scalar_multiply (dot_product vector_a vector_c) vector_b)
             (scalar_multiply (dot_product vector_a vector_b) vector_c)

noncomputable def angle_is_ninety_degrees (u v : ℝ × ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem angle_between_a_and_d_is_ninety_degrees :
  angle_is_ninety_degrees vector_a vector_d :=
by sorry

end angle_between_a_and_d_is_ninety_degrees_l96_96874


namespace die_probabilities_sum_to_one_l96_96997

theorem die_probabilities_sum_to_one :
  let die_outcomes : List ℕ := [1, 2, 3, 4, 5, 6] in
  let is_odd (x : ℕ) := x % 2 = 1 in
  let A := [(1, 1, 1), (3, 3, 3), (5, 5, 5)] in
  let B := [(1, 1, 2), (1, 3, 2), (1, 5, 2), (3, 1, 2), (3, 3, 2), (3, 5, 2), (5, 1, 2), (5, 3, 2), (5, 5, 2),
            (1, 2, 1), (3, 2, 1), (5, 2, 1), (1, 2, 3), (3, 2, 3), (5, 2, 3), (1, 2, 5), (3, 2, 5), (5, 2, 5),
            (2, 1, 1), (2, 3, 1), (2, 5, 1), (2, 1, 3), (2, 3, 3), (2, 5, 3), (2, 1, 5), (2, 3, 5), (2, 5, 5)] in
  let C := [(2, 2, 2), (4, 4, 4), (6, 6, 6), (2, 2, 4), (2, 4, 2), (4, 2, 2), (2, 2, 6), (2, 6, 2), (6, 2, 2),
            (4, 4, 2), (4, 2, 4), (2, 4, 4), (4, 4, 6), (4, 6, 4), (6, 4, 4), (6, 6, 2), (6, 2, 6),
            (2, 6, 6), (4, 4, 6), (4, 6, 6), (6, 4, 6), (6, 6, 4)] in
  (A.card / 216 + B.card / 216 + C.card / 216 = 1) := by
    sorry

end die_probabilities_sum_to_one_l96_96997


namespace quadrilateral_OPQR_is_parallelogram_l96_96395

variables {x1 y1 x2 y2 : ℝ}

def vector_a := (x1, y1)
def vector_b := (x2, y2)
def vector_c := (x1 - x2, y1 - y2)

def point_P := vector_a
def point_Q := vector_b
def point_R := vector_c

theorem quadrilateral_OPQR_is_parallelogram :
  -- Define a condition to check if a quadrilateral is a parallelogram
  (∀ (O P Q R : ℝ × ℝ),
    P = vector_a ∧ Q = vector_b ∧ R = vector_c ∧ 
    P.1 - Q.1 = R.1 ∧ P.2 - Q.2 = R.2 → (O, P, Q, R) forms a parallelogram) :=
sorry

end quadrilateral_OPQR_is_parallelogram_l96_96395


namespace solve_system_l96_96917

theorem solve_system :
  ∃ (x y : ℝ), (x^2 + y^2 ≤ 1) ∧ (x^4 - 18 * x^2 * y^2 + 81 * y^4 - 20 * x^2 - 180 * y^2 + 100 = 0) ∧
    ((x = -1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = -1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = 3 / Real.sqrt 10) ∨
    (x = 1 / Real.sqrt 10 ∧ y = -3 / Real.sqrt 10)) :=
  by
  sorry

end solve_system_l96_96917


namespace locus_of_centers_l96_96434

-- Definitions related to the problem conditions
variables (A B : ℝ × ℝ)
axiom dist_AB : real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4

def circle_center (O : ℝ × ℝ) (r : ℝ) (P : ℝ × ℝ) :=
  real.sqrt ((O.1 - P.1)^2 + (O.2 - P.2)^2) = r

theorem locus_of_centers (O : ℝ × ℝ) :
  (circle_center O 3 A) ∧ (circle_center O 3 B) ↔
    (O = (2, sqrt(5))) ∨ (O = (2, -sqrt(5))) :=
sorry

end locus_of_centers_l96_96434


namespace simplify_fraction_complex_l96_96179

open Complex

theorem simplify_fraction_complex :
  (3 - I) / (2 + 5 * I) = (1 / 29) - (17 / 29) * I := by
  sorry

end simplify_fraction_complex_l96_96179


namespace sqrt2_cos30_eq_l96_96600

theorem sqrt2_cos30_eq :
  let cos30 : ℝ := real.cos (30 * real.pi / 180) in
  cos30 = (real.sqrt 3 / 2) →
  real.sqrt 2 * cos30 = real.sqrt 6 / 2 :=
by
  intro cos30 h
  rw [h]
  dsimp [cos30]
  have : real.sqrt 2 * (real.sqrt 3 / 2) = real.sqrt 6 / 2 := by
    rw [← real.sqrt_mul, mul_div_assoc, mul_comm (real.sqrt 2), mul_one]
    exact congr_arg (λ x, x / 2) (real.sqrt_mul (by norm_num : (0:ℝ) ≤ _) (by norm_num : 0 ≤ 3))
  exact this

end sqrt2_cos30_eq_l96_96600


namespace volume_of_right_square_prism_is_32_l96_96524

-- Defining the conditions
def side_unfolding_is_square : Prop := true
def side_length_unfolding_square : ℝ := 8

-- Defining the square prism's base side length and height based on the given conditions.
def base_side_length : ℝ := side_length_unfolding_square / 4
def height : ℝ := side_length_unfolding_square

-- The volume of the right square prism given the conditions
def volume (b h : ℝ) : ℝ := b^2 * h

theorem volume_of_right_square_prism_is_32 (h_sq : side_unfolding_is_square) (h_len : side_length_unfolding_square = 8) :
  volume base_side_length height = 32 :=
by
  sorry

end volume_of_right_square_prism_is_32_l96_96524


namespace estimate_larger_than_difference_l96_96117

theorem estimate_larger_than_difference
  (u v δ γ : ℝ)
  (huv : u > v)
  (hδ : δ > 0)
  (hγ : γ > 0)
  (hδγ : δ > γ) : (u + δ) - (v - γ) > u - v := by
  sorry

end estimate_larger_than_difference_l96_96117


namespace proof_a_value_range_l96_96346

noncomputable def log_helper (a : ℝ) : ℝ → ℝ := λ x, log a x

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := let Δ := (2*a - 3)^2 - 4 in Δ > 0

def final_statement (a : ℝ) : Prop := 
  (¬ p a ∧ q a) → (a ∈ Ico (1/2 : ℝ) 1 ∨ a ∈ Ioi (5/2 : ℝ))

theorem proof_a_value_range (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  final_statement a :=
sorry

end proof_a_value_range_l96_96346


namespace winning_candidate_votes_l96_96539

theorem winning_candidate_votes (T V₁ V₂ : ℕ) (P : ℚ)
  (hV₁ : V₁ = 4136) (hV₂ : V₂ = 7636) 
  (hP : P = 0.4969230769230769)
  (hT : T = V₁ + V₂ + (P * T).toNat) :
  (P * T).toNat = 11628 := 
by 
  sorry

end winning_candidate_votes_l96_96539


namespace fraction_inequality_solution_l96_96202

open Set

theorem fraction_inequality_solution :
  {x : ℝ | 7 * x - 3 ≥ x^2 - x - 12 ∧ x ≠ 3 ∧ x ≠ -4} = Icc (-1 : ℝ) 3 ∪ Ioo (3 : ℝ) 4 ∪ Icc 4 9 :=
by
  sorry

end fraction_inequality_solution_l96_96202


namespace shaded_area_of_semicircles_l96_96126

variables {U V W X Y Z A B : Point}
variables (d : ℝ) (UV VW WX XY YZ ZB AB UA : ℝ) (π : ℝ)

-- Define the conditions
def on_straight_line (U V W X Y Z A B : Point) : Prop := 
  -- Presumed condition that all points lie on a line
  true 

def equal_segments (UV VW WX XY YZ ZB AB : ℝ) (d : ℝ) : Prop := 
  UV = d ∧ VW = d ∧ WX = d ∧ XY = d ∧ YZ = d ∧ ZB = d ∧ AB = d

-- Define the theorem
theorem shaded_area_of_semicircles 
  (h1 : on_straight_line U V W X Y Z A B)
  (h2 : equal_segments UV VW WX XY YZ ZB AB 5)
  (h3 : UA = 35)
  : (area_of_shaded_region U V W X Y Z A B = (625 / 4) * π) :=
sorry

end shaded_area_of_semicircles_l96_96126


namespace train_speed_is_18_kmh_l96_96196

noncomputable def speed_of_train (length_of_bridge length_of_train time : ℝ) : ℝ :=
  (length_of_bridge + length_of_train) / time * 3.6

theorem train_speed_is_18_kmh
  (length_of_bridge : ℝ)
  (length_of_train : ℝ)
  (time : ℝ)
  (h1 : length_of_bridge = 200)
  (h2 : length_of_train = 100)
  (h3 : time = 60) :
  speed_of_train length_of_bridge length_of_train time = 18 :=
by
  sorry

end train_speed_is_18_kmh_l96_96196


namespace sum_a6_to_a9_l96_96054

-- Given definitions and conditions
def sequence_sum (n : ℕ) : ℕ := n^3
def a (n : ℕ) : ℕ := sequence_sum (n + 1) - sequence_sum n

-- Theorem to be proved
theorem sum_a6_to_a9 : a 6 + a 7 + a 8 + a 9 = 604 :=
by sorry

end sum_a6_to_a9_l96_96054


namespace area_ratio_of_extended_equilateral_triangle_l96_96870

noncomputable theory

-- Definitions of the problem
def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def extension_by_factor (A B B' : Point) (factor : ℝ) : Prop :=
  dist B B' = factor * dist A B

-- Main statement with the necessary conditions
theorem area_ratio_of_extended_equilateral_triangle (A B C A' B' C' : Point)
  (h_eq : equilateral_triangle A B C)
  (h_AB : extension_by_factor A B B' 3)
  (h_BC : extension_by_factor B C C' 3)
  (h_CA : extension_by_factor C A A' 3)
  : area (triangle A' B' C') = 16 * area (triangle A B C) :=
sorry

end area_ratio_of_extended_equilateral_triangle_l96_96870


namespace min_value_of_f_l96_96337

def f (x : ℝ) : ℝ := max (abs (x + 1)) (x^2)

theorem min_value_of_f :
  ∃ x ∈ set.Icc ((1 - real.sqrt 5) / 2) ((1 + real.sqrt 5) / 2), f x = (3 - real.sqrt 5) / 2 :=
sorry

end min_value_of_f_l96_96337


namespace hannah_payment_l96_96767

def costWashingMachine : ℝ := 100
def costDryer : ℝ := costWashingMachine - 30
def totalCostBeforeDiscount : ℝ := costWashingMachine + costDryer
def discount : ℝ := totalCostBeforeDiscount * 0.1
def finalCost : ℝ := totalCostBeforeDiscount - discount

theorem hannah_payment : finalCost = 153 := by
  simp [costWashingMachine, costDryer, totalCostBeforeDiscount, discount, finalCost]
  sorry

end hannah_payment_l96_96767


namespace uncle_ben_eggs_l96_96549

noncomputable def total_eggs (total_chickens : ℕ) (roosters : ℕ) (non_egg_laying_hens : ℕ) (eggs_per_hen : ℕ) : ℕ :=
  let total_hens := total_chickens - roosters
  let egg_laying_hens := total_hens - non_egg_laying_hens
  egg_laying_hens * eggs_per_hen

theorem uncle_ben_eggs :
  total_eggs 440 39 15 3 = 1158 :=
by
  unfold total_eggs
  -- Correct steps to prove the theorem can be skipped with sorry
  sorry

end uncle_ben_eggs_l96_96549


namespace convex_polygon_enclosed_in_parallelogram_l96_96168

noncomputable theory

open Set

-- Definition of a convex polygon
structure ConvexPolygon (V : Type*) :=
(vertices : Finset V)
(convex : ConvexHull vertices)

-- Definition of area function (placeholder, assuming the existence of an area function)
def area {V : Type*} [MetricSpace V] (p : ConvexPolygon V) : ℝ := sorry

-- Main theorem statement
theorem convex_polygon_enclosed_in_parallelogram {V : Type*} [MetricSpace V] :
    ∀ (M : ConvexPolygon V), area M = 1 → ∃ (P : ConvexPolygon V), area P = 2 :=
by
    intros M hM
    sorry

end convex_polygon_enclosed_in_parallelogram_l96_96168


namespace converse_statement_2_true_implies_option_A_l96_96776

theorem converse_statement_2_true_implies_option_A :
  (∀ x : ℕ, x = 1 ∨ x = 2 → (x^2 - 3 * x + 2 = 0)) →
  (x = 1 ∨ x = 2) :=
by
  intro h
  sorry

end converse_statement_2_true_implies_option_A_l96_96776


namespace tire_spacing_correct_l96_96231

def bicycle_tire_spacing (d: ℝ) (w: ℝ) : ℝ := (d * Real.pi) - w

theorem tire_spacing_correct (d w : ℝ) (h1 : d = 60) (h2 : w = 20) : 
    bicycle_tire_spacing d w = (60 * Real.pi) - 20 := sorry

end tire_spacing_correct_l96_96231


namespace dan_bought_18_stickers_l96_96658

variable (S D : ℕ)

-- Given conditions
def stickers_initially_same : Prop := S = S -- Cindy and Dan have the same number of stickers initially
def cindy_used_15_stickers : Prop := true -- Cindy used 15 of her stickers
def dan_bought_D_stickers : Prop := true -- Dan bought D stickers
def dan_has_33_more_stickers_than_cindy : Prop := (S + D) = (S - 15 + 33)

-- Question: Prove that the number of stickers Dan bought is 18
theorem dan_bought_18_stickers (h1 : stickers_initially_same S)
                               (h2 : cindy_used_15_stickers)
                               (h3 : dan_bought_D_stickers)
                               (h4 : dan_has_33_more_stickers_than_cindy S D) : D = 18 :=
sorry

end dan_bought_18_stickers_l96_96658


namespace total_marbles_l96_96792

theorem total_marbles (r b g : ℕ) (total : ℕ) 
  (h_ratio : 2 * g = 4 * b) 
  (h_blue_marbles : b = 36) 
  (h_total_formula : total = r + b + g) 
  : total = 108 :=
by
  sorry

end total_marbles_l96_96792


namespace functions_not_necessarily_equal_l96_96105

-- Define the domain and range
variables {α β : Type*}

-- Define two functions f and g with the same domain and range
variables (f g : α → β)

-- Lean statement for the given mathematical problem
theorem functions_not_necessarily_equal (h_domain : ∀ x : α, (∃ x : α, true))
  (h_range : ∀ y : β, (∃ y : β, true)) : ¬(f = g) :=
sorry

end functions_not_necessarily_equal_l96_96105


namespace largest_integer_crates_same_oranges_l96_96638

theorem largest_integer_crates_same_oranges :
  ∃ n : ℕ, (∀ (crates : Finset ℕ), 
  (∀ c ∈ crates, 135 ≤ c ∧ c ≤ 165) → crates.card = 200 → 
  (∃ count, (count ∈ crates) ∧ (crates.filter (λ x, x = count)).card ≥ n) ∧ n = 7) := sorry

end largest_integer_crates_same_oranges_l96_96638


namespace distance_and_ratio_correct_l96_96324

noncomputable def distance_and_ratio (a : ℝ) : ℝ × ℝ :=
  let dist : ℝ := a / Real.sqrt 3
  let ratio : ℝ := 1 / 2
  ⟨dist, ratio⟩

theorem distance_and_ratio_correct (a : ℝ) :
  distance_and_ratio a = (a / Real.sqrt 3, 1 / 2) := by
  -- Proof omitted
  sorry

end distance_and_ratio_correct_l96_96324


namespace factor_quartic_l96_96319

theorem factor_quartic (w : ℂ) : (w^4 - 81) = (w - 3) * (w + 3) * (w - 3i) * (w + 3i) := 
sorry

end factor_quartic_l96_96319


namespace variance_cows_l96_96620

-- Define the number of cows and incidence rate.
def n : ℕ := 10
def p : ℝ := 0.02

-- The variance of the binomial distribution, given n and p.
def variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Statement to prove
theorem variance_cows : variance n p = 0.196 :=
by
  sorry

end variance_cows_l96_96620


namespace find_f_l96_96706

variable (f : ℤ → ℤ)

-- Condition given in the problem
def condition1 : Prop := ∀ x : ℤ, f(x + 1) = 2 * x - 1

-- Problem Statement
theorem find_f (h : condition1 f) : ∀ x : ℤ, f x = 2 * x - 3 := 
sorry

end find_f_l96_96706


namespace ternary_to_decimal_l96_96669

def to_decimal (ternary : Nat) : Nat :=
  match ternary with
  | 121 => 1 * 3^2 + 2 * 3^1 + 1 * 3^0
  | _ => 0

theorem ternary_to_decimal : to_decimal 121 = 16 := by
  sorry

end ternary_to_decimal_l96_96669


namespace range_of_a_l96_96737

noncomputable def monotone_decreasing {α β : Type*} [Preorder α] [Preorder β] (f : α → β) : Prop :=
  ∀ ⦃x y⦄, x ≤ y → f y ≤ f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f(1 + (Real.sin x)^2) ≤ f(a - 2 * Real.cos x)) ∧
  monotone_decreasing f → a ≤ -1 :=
by
  sorry

end range_of_a_l96_96737


namespace smallest_value_of_3a_plus_2_l96_96089

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 3 * a + 2 = -1 :=
by
  sorry

end smallest_value_of_3a_plus_2_l96_96089


namespace solve_for_y_l96_96497

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l96_96497


namespace radical_axis_ratio_divides_midline_l96_96134

variable (A B C M N : Type)
variable [Triangle A B C]
variable [Angle (A B C) 60]
variable [Angle (B C A) 45]
variable (midline_parallel_to_AB : Type)
variable (radical_axis : Type)

-- assumption: median points M and N of sides BC, and AC respectively.
variable (AM_median : Median A B C M)
variable (BN_median : Median B A C N)

-- circles on medians AM and BN as diameters
variable (circle_AM : Circle (diameter AM_median))
variable (circle_BN : Circle (diameter BN_median))

-- radical axis condition
variable (radical_axis_divides_midline : radical_axis divides midline_parallel_to_AB)

theorem radical_axis_ratio_divides_midline :
  radical_axis_divides_midline = sqrt(3) := sorry

end radical_axis_ratio_divides_midline_l96_96134


namespace total_dollar_amount_l96_96990

/-- Definitions of base 5 numbers given in the problem -/
def pearls := 1 * 5^0 + 2 * 5^1 + 3 * 5^2 + 4 * 5^3
def silk := 1 * 5^0 + 1 * 5^1 + 1 * 5^2 + 1 * 5^3
def spices := 1 * 5^0 + 2 * 5^1 + 2 * 5^2
def maps := 0 * 5^0 + 1 * 5^1

/-- The theorem to prove the total dollar amount in base 10 -/
theorem total_dollar_amount : pearls + silk + spices + maps = 808 :=
by
  sorry

end total_dollar_amount_l96_96990


namespace trig_identity1_trig_identity2_l96_96602

-- Problem 1: 
theorem trig_identity1 (α : ℝ) (h1 : sin α - cos α = 1 / 5) (h2 : π < α ∧ α < 3 * π / 2) :
  (sin α * cos α = 12 / 25) ∧ (sin α + cos α = -7 / 5) :=
sorry

-- Problem 2:
theorem trig_identity2 (x : ℝ) (h1 : cos (40 * (π / 180) + x) = 1 / 4) (h2 : -π < x ∧ x < -π / 2) :
  cos (140 * (π / 180) - x) + cos^2 (50 * (π / 180) - x) = 11 / 16 :=
sorry

end trig_identity1_trig_identity2_l96_96602


namespace find_b_l96_96937

theorem find_b (b : ℝ) : 
  (∀ x y : ℝ, x + y = b ↔ x + y = 11.5) ∧ 
  (∀ p1 p2 : (ℝ × ℝ), p1 = (0, 5) ∧ p2 = (8, 10) →
   let mid_x := (p1.1 + p2.1) / 2 in
   let mid_y := (p1.2 + p2.2) / 2 in
   p1.1 = 0 ∧ p1.2 = 5 ∧ p2.1 = 8 ∧ p2.2 = 10 ∧
   (mid_x, mid_y) = (4, 7.5) ∧ 
   4 + 7.5 = 11.5) →
  b = 11.5 :=
by {
  sorry
}

end find_b_l96_96937


namespace distinct_ways_to_satisfy_equation_l96_96343

def valid_selections : Set (ℕ × ℕ × ℕ) := {
  (a, b, c) | a ∈ {4, 5, 6, 7, 8, 9} ∧ b ∈ {4, 5, 6, 7, 8, 9} ∧ 
              c ∈ {4, 5, 6, 7, 8, 9} ∧ a ≠ b ∧ a - b = c
}

def count_valid_selections : ℕ := valid_selections.toFinset.card

theorem distinct_ways_to_satisfy_equation : count_valid_selections = 13 := by
  sorry

end distinct_ways_to_satisfy_equation_l96_96343


namespace quilt_shaded_fraction_l96_96950

theorem quilt_shaded_fraction (total_squares : ℕ) (fully_shaded : ℕ) (half_shaded_squares : ℕ) (half_shades_per_square: ℕ) : 
  (((fully_shaded) + (half_shaded_squares * half_shades_per_square / 2)) / total_squares) = (1 / 4) :=
by 
  let fully_shaded := 2
  let half_shaded_squares := 4
  let half_shades_per_square := 1
  let total_squares := 16
  sorry

end quilt_shaded_fraction_l96_96950


namespace part1_part2_l96_96755

noncomputable def quadratic_eq (x m : ℝ) : ℝ := x^2 - m * x + m - 2

theorem part1 (m : ℝ) (h : quadratic_eq (-1) m = 0) : m = 1 / 2 :=
by sorry

theorem part2 (m : ℝ) : ∃ a b c : ℝ, a = 1 ∧ b = -m ∧ c = m - 2 ∧ (b^2 - 4 * a * c) > 0 :=
by
  use [1, -m, m - 2],
  split, rfl,
  split, rfl,
  split, rfl,
  calc
    (-m)^2 - 4 * 1 * (m - 2)
      = m^2 - 4 * m + 8 : by ring
    ... > 0 : by
      have : (m - 2)^2 + 4 > 0 := by
        apply add_pos_of_nonneg_of_pos; norm_num; apply pow_two_nonneg,
      rwa [← add_assoc, ← sub_eq_add_neg, add_sub_assoc] at this
  sorry

end part1_part2_l96_96755


namespace rectangle_width_decreased_l96_96195

theorem rectangle_width_decreased (l w : ℝ) (h_l_positive: l > 0) (h_w_positive: w > 0) :
  let new_length := 1.30 * l,
      new_width := (l * w) / new_length in
  (1 - new_width / w) * 100 = 23.08 := by
  sorry

end rectangle_width_decreased_l96_96195


namespace range_of_f_l96_96731

def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def function_defined_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x, (x ∈ set.Icc a b) → (f x).is_something -- appropriate specifier, placeholder here

noncomputable def f (a b : ℝ) : ℝ → ℝ := 
λ x, a * x^2 + b * x + 2

theorem range_of_f (a : ℝ) (h1 : is_even_function (f a 0)) (h2 : function_defined_on_interval (f a 0) (1 + a) 2) :
    set.range (f a 0) = set.Icc (-10 : ℝ) 2 :=
begin
  sorry
end

end range_of_f_l96_96731


namespace sum_mod_17_eq_0_l96_96025

theorem sum_mod_17_eq_0 :
  (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 0 :=
by
  sorry

end sum_mod_17_eq_0_l96_96025


namespace value_of_nested_radical_l96_96694

def nested_radical : ℝ := 
  sorry -- Definition of the recurring expression is needed here, let's call it x
  
theorem value_of_nested_radical :
  (nested_radical = 5) :=
sorry -- The actual proof steps will be written here.

end value_of_nested_radical_l96_96694


namespace integral_substitution_l96_96981

open Set Function Filter

variables {a b : ℝ} {φ : ℝ → ℝ} (f : ℝ → ℝ)

-- Assumptions
-- 1. φ = φ(y) is a smooth function on [a, b] such that φ(a) < φ(b)
-- 2. f = f(x) is a Borel measurable function integrable on [φ(a), φ(b)]
-- Define "φ smooth" as having continuous derivative, which is continuous differentiability
def φ_smooth (a b : ℝ) (φ : ℝ → ℝ) : Prop := Continuous φ ∧ Continuous (λ y, deriv φ y)

-- Prove: ∫ (x in φ(a)..φ(b)), f(x) dx = ∫ (y in a..b), f(φ(y)) * deriv φ y dy
theorem integral_substitution (ha : φ_smooth a b φ) (hφa : φ a < φ b) 
  {μ : MeasureTheory.Measure ℝ}
  (μab : μ (Ioc a b) < ∞)
  (h_int_f : MeasureTheory.IntegrableOn f (Ioc (φ a) (φ b)) μ) : 
  ∫ x in (Set.interval (φ a) (φ b)), f x ∂μ = 
  ∫ y in (Set.interval a b), f (φ y) * deriv φ y ∂μ :=
by
  sorry

end integral_substitution_l96_96981


namespace ball_attendance_l96_96856

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l96_96856


namespace tetrahedron_volume_proof_l96_96008

-- Defining the conditions
variables (A B C D : Type) [metric_space A]
variables (angle_ABC_BCD : real)
variables (area_ABC area_BCD BC : real)

-- Given conditions
def tetrahedron_conditions : Prop :=
  angle_ABC_BCD = 45 ∧ area_ABC = 150 ∧ area_BCD = 90 ∧ BC = 12

-- Prove the volume of tetrahedron ABCD is 375√2 under given conditions
theorem tetrahedron_volume_proof (A B C D : Type) [metric_space A]
  (angle_ABC_BCD : real) (area_ABC area_BCD BC : real) :
  tetrahedron_conditions A B C D angle_ABC_BCD area_ABC area_BCD BC →
  volume (A, B, C, D) = 375 * real.sqrt 2 :=
begin
  sorry
end

end tetrahedron_volume_proof_l96_96008


namespace inequality_neg_3_l96_96060

theorem inequality_neg_3 (a b : ℝ) : a < b → -3 * a > -3 * b :=
by
  sorry

end inequality_neg_3_l96_96060


namespace sum_of_three_integers_l96_96205

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 125) : a + b + c = 31 :=
sorry

end sum_of_three_integers_l96_96205


namespace minimum_omega_l96_96378

noncomputable def omega_min : ℝ :=
  if h : 0 < 4 then 4 else 0

theorem minimum_omega (omega : ℝ) (h_omega_pos : omega > 0)
  (f : ℝ → ℝ := λ x, Real.sin (omega * x + Real.pi / 6))
  (h_symmetry : ∀ x : ℝ, Real.sin (omega * x + Real.pi / 6) = Real.sin (omega * (x + Real.pi / 12) + Real.pi / 6)) :
  omega_min = 4 := by
  sorry

end minimum_omega_l96_96378


namespace solve_for_y_l96_96499

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l96_96499


namespace misha_class_predictions_probability_l96_96587

-- Definitions representing the conditions
def monday_classes : ℕ := 5
def tuesday_classes : ℕ := 6
def total_classes : ℕ := monday_classes + tuesday_classes
def total_flips : ℕ := total_classes
def correct_predictions : ℕ := 7
def correct_monday_predictions : ℕ := 3

-- Lean theorem representing the proof problem
theorem misha_class_predictions_probability :
  (probability_of_correct_predictions total_flips correct_predictions monday_classes correct_monday_predictions) =
    (5 / 11) :=
  sorry

end misha_class_predictions_probability_l96_96587


namespace sequence_bounds_l96_96816

open Nat Real

noncomputable def x : ℕ → ℝ
| 0     := 3
| (n+1) := sorry -- left as an exercise; this would be computed based on some recurrence relation

theorem sequence_bounds (n : ℕ) :
  (∀ n, (4 * x (n+1) - 3 * x n) < 2) →
  (∀ n, (2 * x (n+1) - x n) < 2) →
  (2 + (1/2)^n < x (n+1) ∧ x (n+1) < 2 + (3/4)^n) :=
by
  intros h1 h2
  sorry

end sequence_bounds_l96_96816


namespace ternary_to_decimal_121_l96_96662

theorem ternary_to_decimal_121 : 
  let t : ℕ := 1 * 3^2 + 2 * 3^1 + 1 * 3^0 
  in t = 16 :=
by
  sorry

end ternary_to_decimal_121_l96_96662


namespace triangles_similar_l96_96807

variable {A B C D E F : Type}

-- Assume we have points A, B, and C that form an acute-angled triangle ABC.
-- And AD, BE, and CF are the altitudes of the triangle.

-- Define the necessary assumptions.
variables (h_acute : ∀ (A B C : Type), triangle A B C → acuteTriangle A B C)
          (h_altitudes : ∀ (A B C D E F : Type), altitude A D B C × altitude B E A C × altitude C F A B)

-- Define the similarity statements to be proved.
theorem triangles_similar
        (A B C D E F : Type)
        [triangle A B C]
        [acuteTriangle A B C]
        [altitude A D B C]
        [altitude B E A C]
        [altitude C F A B] :
  similar (triangle AFC) (triangle ABE) ∧
  similar (triangle BFC) (triangle ABD) ∧
  similar (triangle BEC) (triangle ADC) :=
by
  sorry

end triangles_similar_l96_96807


namespace united_telephone_additional_charge_l96_96220

theorem united_telephone_additional_charge :
  ∃ x : ℝ, (7 + 100 * x) = (12 + 100 * 0.20) ∧ x = 0.25 :=
by 
  use 0.25
  split
  · linarith
  · refl

end united_telephone_additional_charge_l96_96220


namespace product_of_numerator_and_denominator_of_fraction_l96_96557

def repeating_decimal_to_fraction (x : Real) (hx : x = 0.009009009...) : Fraction :=
  sorry -- placeholder for a function that converts repeating decimals to fractional form

theorem product_of_numerator_and_denominator_of_fraction
  (x : Real)
  (hx : x = 0.009009009...)
  (f : Fraction := repeating_decimal_to_fraction x hx) :
  (f.numerator * f.denominator = 111) :=
sorry -- the proof itself, not required as per instructions

end product_of_numerator_and_denominator_of_fraction_l96_96557


namespace five_in_range_of_f_l96_96699

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b*x + 3

theorem five_in_range_of_f (b : ℝ) : 
  ∀ y, (y = 5) -> ∃ x : ℝ, f x b = y :=
by {
  -- define the function
  let f := λ (x : ℝ) (b : ℝ), x^2 + b*x + 3,

  -- non-computable proof goes here
  sorry
}

end five_in_range_of_f_l96_96699


namespace angle_between_c_and_d_is_zero_l96_96461

variables {ℝ : Type} [inner_product_space ℝ ℝ] (c d : ℝ)

-- c and d are unit vectors
def unit_vector (v : ℝ) : Prop := inner_product_space.norm_sq v = 1

-- c + 3d and 3c - 2d are orthogonal
def orthogonal (u v : ℝ) : Prop := inner_product_space.inner u v = 0

-- The angle between c and d is 0 degrees
theorem angle_between_c_and_d_is_zero
  (hc : unit_vector c)
  (hd : unit_vector d)
  (h_orth : orthogonal (c + 3 • d) (3 • c - 2 • d)) :
  inner_product_space.angle c d = 0 :=
sorry

end angle_between_c_and_d_is_zero_l96_96461


namespace fraction_evaluation_l96_96599

theorem fraction_evaluation :
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2) / (0 - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) = 5 / 4 :=
by
  sorry

end fraction_evaluation_l96_96599


namespace polynomial_solution_bounded_sequence_l96_96986

noncomputable def p (m : ℤ) : ℕ :=
if m = 0 then ∘∞
else
  let abs_m := abs m
  let factors := List.filter Nat.prime (Nat.factors abs_m)
  factors.maximum'.getOrElse 1

def is_bounded_above (S : ℕ → ℤ) (C : ℤ) : Prop :=
∀ n, S n < C

def sequence_bounded (f : ℤ → ℤ) : Prop :=
∃ C, is_bounded_above (λ n, p(f(n^2)) - 2 * n) C

def form_of_f (f : ℤ → ℤ) : Prop :=
∃ (k : ℕ) (a : Fin k → ℤ) (c : ℤ), 
  (∀ i, i < k → a i % 2 = 1) ∧ 
  c ≠ 0 ∧ 
  f = λ x, c * List.product (List.map (λ i, 4 * x - (a i)^2) (List.finRange k))

theorem polynomial_solution_bounded_sequence :
  ∀ f : ℤ → ℤ, sequence_bounded f → form_of_f f :=
sorry

end polynomial_solution_bounded_sequence_l96_96986


namespace volume_of_region_l96_96696

theorem volume_of_region :
    ∀ (x y z : ℝ), 
    |x - y + z| + |x - y - z| ≤ 12 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 
    → true := by
    sorry

end volume_of_region_l96_96696


namespace hannah_total_payment_l96_96766

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def total_cost_before_discount : ℝ := washing_machine_cost + dryer_cost
def discount : ℝ := 0.10
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount)

theorem hannah_total_payment : total_cost_after_discount = 153 := by
  sorry

end hannah_total_payment_l96_96766


namespace count_even_numbers_between_300_and_600_l96_96399

theorem count_even_numbers_between_300_and_600 : 
  card {n : ℕ | 300 < n ∧ n < 600 ∧ n % 2 = 0} = 149 := 
sorry

end count_even_numbers_between_300_and_600_l96_96399


namespace ball_attendance_l96_96863

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l96_96863


namespace grid_sum_21_proof_l96_96200

-- Define the condition that the sum of the horizontal and vertical lines are 21
def valid_grid (nums : List ℕ) (x : ℕ) : Prop :=
  nums ≠ [] ∧ (((nums.sum + x) = 42) ∧ (21 + 21 = 42))

-- Define the main theorem to prove x = 7
theorem grid_sum_21_proof (nums : List ℕ) (h : valid_grid nums 7) : 7 ∈ nums :=
  sorry

end grid_sum_21_proof_l96_96200


namespace distance_from_P_to_AB_l96_96489

noncomputable def rectangle_intersection_distance : ℝ :=
  let D := (0, 0)
  let C := (6, 0)
  let B := (6, 8)
  let A := (0, 8)
  let M := (3, 0)
  let circle_M := λ x y, (x - 3) ^ 2 + y ^ 2 = 9
  let circle_B := λ x y, (x - 6) ^ 2 + (y - 8) ^ 2 = 25
  let intersection_points := {p : ℝ × ℝ // circle_M p.1 p.2 ∧ circle_B p.1 p.2}
  let P := (18/5 : ℝ, 24/5 : ℝ)
  if (⟨P.1, P.2⟩ : intersection_points) then P.1 else 0

theorem distance_from_P_to_AB : rectangle_intersection_distance = 18/5 := by
  sorry

end distance_from_P_to_AB_l96_96489


namespace sequence_general_term_l96_96320

theorem sequence_general_term (n : ℕ) : 
  (∀ (a : ℕ → ℚ), (a 1 = 1) ∧ (a 2 = 2 / 3) ∧ (a 3 = 3 / 7) ∧ (a 4 = 4 / 15) ∧ (a 5 = 5 / 31) → a n = n / (2^n - 1)) :=
by
  sorry

end sequence_general_term_l96_96320


namespace forget_percentage_is_correct_l96_96580

-- Define the number of students in groups A and B
def groupA_students : ℕ := 30
def groupB_students : ℕ := 50

-- Define the percentage of students from each group that forgot their homework
def groupA_forget_percentage : ℝ := 0.20
def groupB_forget_percentage : ℝ := 0.12

-- Define the number of students who forgot their homework in each group
def forget_groupA : ℕ := (groupA_forget_percentage * groupA_students).toNat
def forget_groupB : ℕ := (groupB_forget_percentage * groupB_students).toNat

-- Define the total number of students and the total number of students who forgot their homework
def total_students : ℕ := groupA_students + groupB_students
def total_forget : ℕ := forget_groupA + forget_groupB

-- Define the percentage of sixth graders who forgot their homework
def forget_percentage : ℝ := (total_forget.toRat / total_students.toRat) * 100

-- Theorem: The percentage of sixth graders who forgot their homework is 15%
theorem forget_percentage_is_correct : forget_percentage = 15 := by
  sorry

end forget_percentage_is_correct_l96_96580


namespace find_matrix_M_l96_96329

open Matrix

def M : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![(-1), (-5)], ![(0.5), (3.5)]]

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2, -5], ![4, -3]]

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![-21, 19], ![15, -13]]

theorem find_matrix_M : M ⬝ A = B := by
  sorry

end find_matrix_M_l96_96329


namespace curve_area_eq_square_l96_96743

theorem curve_area_eq_square (a : ℝ) (h : a > 0) : 
  ((∫ x in 0..a, real.sqrt x) = a^2) → a = (2/3)^(2/3) :=
by
  sorry

end curve_area_eq_square_l96_96743


namespace book_count_after_borrowing_l96_96796

theorem book_count_after_borrowing:
  (initial_books net_change1 net_change2 : ℤ) 
  (on_first_day : net_change1 = -3 + 1) 
  (on_second_day : net_change2 = -1 + 2) 
  (initial_books = 20) :
  initial_books + net_change1 + net_change2 = 19 :=
by
  sorry

end book_count_after_borrowing_l96_96796


namespace total_area_of_shaded_regions_l96_96445

-- Definitions and conditions provided
def larger_circle_area : ℝ := 64 * Real.pi
def larger_circle_radius : ℝ := Real.sqrt 64
def larger_circle_shaded_area : ℝ := (1 / 2) * larger_circle_area

def smaller_circle_diameter : ℝ := larger_circle_radius
def smaller_circle_radius : ℝ := smaller_circle_diameter / 2
def smaller_circle_area : ℝ := Real.pi * smaller_circle_radius ^ 2
def smaller_circle_shaded_area : ℝ := (1 / 2) * smaller_circle_area

def total_shaded_area : ℝ := larger_circle_shaded_area + smaller_circle_shaded_area

-- Mathematical goal to prove
theorem total_area_of_shaded_regions :
  total_shaded_area = 40 * Real.pi := by
  sorry

end total_area_of_shaded_regions_l96_96445


namespace distances_AB_BC_l96_96627

variable (c α β V : ℝ)

-- Conditions
def schooner_speed := c  -- Speed of the schooner relative to still water
def total_time_AC := α  -- Time from A to C
def total_time_return := β  -- Return time
def time_ratio_CB_BA := 1 / 3  -- Time from C to B is one-third the time from B to A

-- Distances to solve for AB and BC
noncomputable def AB : ℝ := 3 * β * c / 4
noncomputable def BC : ℝ := β * c * (4 * α - 3 * β) / (4 * (2 * α - β))

-- Theorem to prove distances AB and BC
theorem distances_AB_BC (h1 : schooner_speed = c)
                        (h2 : total_time_AC = α)
                        (h3 : total_time_return = β)
                        (h4 : time_ratio_CB_BA = 1 / 3) :
  AB = 3 * β * c / 4 ∧ BC = β * c * (4 * α - 3 * β) / (4 * (2 * α - β)) := by
  sorry

end distances_AB_BC_l96_96627


namespace peanuts_difference_is_correct_l96_96830

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- Define the difference in the number of peanuts between Kenya and Jose
def peanuts_difference : ℕ := Kenya_peanuts - Jose_peanuts

-- Prove that the number of peanuts Kenya has minus the number of peanuts Jose has is equal to 48
theorem peanuts_difference_is_correct : peanuts_difference = 48 := by
  sorry

end peanuts_difference_is_correct_l96_96830


namespace simplify_tan_product_l96_96910

noncomputable def tan_deg (d : ℝ) : ℝ := Real.tan (d * Real.pi / 180)

theorem simplify_tan_product :
  (1 + tan_deg 10) * (1 + tan_deg 35) = 2 := 
by
  -- Given conditions
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have h2 : tan_deg 10 + tan_deg 35 = 1 - tan_deg 10 * tan_deg 35 :=
    by sorry -- Use tan addition formula here
  -- Proof of the theorem follows from here
  sorry

end simplify_tan_product_l96_96910


namespace part_I_part_II_l96_96703

-- Define the variables and assumptions
variables (a x : ℝ) (n : ℕ)
variable ha : a > 0

-- Part I: Given y = (x - a) ^ n, prove y' = n * (x - a) ^ (n - 1)
theorem part_I (y : ℝ) (h1 : y = (x - a) ^ n) (hn_pos : 0 < n) : 
  has_deriv_at (λ x, (x - a) ^ n) (n * (x - a) ^ (n - 1)) x :=
sorry

-- Part II: Given f_n(x) = x^n - (x - a)^n, prove f'_{(n+1)}(n+1) > (n+1)f'_n(n)
theorem part_II (hn_ge_a : n ≥ a) :
  has_deriv_at (λ x, x^(n+1) - (x - a)^(n+1)) (λ x, (n+1) * x^n - (n+1) * (x - a)^n) (n + 1) ∧
  (n + 1) * ((n + 1)^n - (n + 1 - a)^n) > (n + 1) * ((n^n) - (n * (n - a)^(n - 1))) :=
sorry

end part_I_part_II_l96_96703


namespace number_of_happy_polynomials_l96_96265

-- Define the concept of a happy polynomial of degree n or less.
def happy_polynomial (n : ℕ) (f : ℕ → ℚ) : Prop :=
  (∀ i < n + 1, 0 ≤ f i ∧ f i < 1) ∧
  (∀ x : ℕ, (∑ i in finset.range (n + 1), f i * x^i : ℚ) ∈ (set.range (coe : ℕ → ℚ)))

-- Define the number of happy polynomials of degree n or less as a function.
noncomputable def count_happy_polynomials (n : ℕ) : ℕ :=
nat.factorial n

-- The statement to be proved: the number of happy polynomials of degree n or less is 1! * 2! * ... * n!.
theorem number_of_happy_polynomials (n : ℕ) :
  count_happy_polynomials n = ∏ i in finset.range (n + 1), nat.factorial i :=
sorry

end number_of_happy_polynomials_l96_96265


namespace polygons_exist_polygon_sides_count_polygon_area_conservation_l96_96534

variables {A B C D E F O X Y Z T : Type}

-- Assuming 2D points and triangles
variable [has_coe (fin 2 → ℝ) O]

variables [is_triangleABC : fin 3 → affine ℝ]
variables [is_triangleDEF : fin 3 → affine ℝ]

-- Definitions:
def in_triangle (X : fin 2 → ℝ) (T : fin 3 → affine ℝ) : Prop :=
∃ u v w : ℝ, u ≥ 0 ∧ v ≥ 0 ∧ w ≥ 0 ∧ u + v + w = 1 ∧ u • T 0 + v • T 1 + w • T 2 = X

def parallelogram (O X Y Z : fin 2 → ℝ) : Prop :=
  ∃ P, P = (X - O) + (Y - O) / 2 ∧ P + O = Z

-- Problem:
theorem polygons_exist (hOX : in_triangle X is_triangleABC) (hOY : in_triangle Y is_triangleDEF) (h1 : parallelogram O X Y Z) (h2 : parallelogram O T X Y) :
  ∃ P, is_polygon Z ∧ is_polygon T := 
sorry

theorem polygon_sides_count (hOX : in_triangle X is_triangleABC) (hOY : in_triangle Y is_triangleDEF) (h1 : parallelogram O X Y Z) (h2 : parallelogram O T X Y) :
  ∃ n : ℕ, n = 6 := 
sorry

theorem polygon_area_conservation (hOX : in_triangle X is_triangleABC) (hOY : in_triangle Y is_triangleDEF) (h1 : parallelogram O X Y Z) (h2 : parallelogram O T X Y) :
  ∃ A ABC DEF, A = area_polygon Z + area_polygon T := 
sorry

end polygons_exist_polygon_sides_count_polygon_area_conservation_l96_96534


namespace tangent_circle_pairs_bound_l96_96435

noncomputable theory
open set real

-- Definition of the problem conditions
def num_pairs_of_tangent_circles (n : ℕ) (d : ℝ) (l_n : ℕ) : Prop :=
  (n = 3 → l_n ≤ 3) ∧
  (n = 4 → l_n ≤ 5) ∧
  (n = 5 → l_n ≤ 7) ∧
  (n = 7 → l_n ≤ 12) ∧
  (n = 8 → l_n ≤ 14) ∧
  (n = 9 → l_n ≤ 16) ∧
  (n = 10 → l_n ≤ 19) ∧
  (n ≥ 9 → l_n ≤ 3 * n - 11)

-- Proof problem statement
theorem tangent_circle_pairs_bound (n : ℕ) (d : ℝ) (h : n ≥ 9) :
  ∃ l_n : ℕ, num_pairs_of_tangent_circles n d l_n :=
by {
  sorry
}

end tangent_circle_pairs_bound_l96_96435


namespace solution_of_inequality_l96_96526

noncomputable def solution_set : set ℝ := {x : ℝ | x > 2}

theorem solution_of_inequality (x : ℝ) : 
  abs (x^2 - 5 * x + 6) < x^2 - 4 ↔ x ∈ solution_set := 
sorry

end solution_of_inequality_l96_96526


namespace tangent_points_l96_96744

noncomputable def curve : ℝ → ℝ := λ x, x^3 + x - 2
noncomputable def line_slope : ℝ := 4
noncomputable def curve_slope (x : ℝ) : ℝ := 3 * x^2 + 1
noncomputable def y (x : ℝ) := curve x

theorem tangent_points :
  ∃ P_0 : (ℝ × ℝ), (P_0 = (1,0) ∨ P_0 = (-1, -4)) ∧
    (curve_slope P_0.1 = line_slope ∧ y P_0.1 = P_0.2) :=
begin
  sorry
end

end tangent_points_l96_96744


namespace minimum_attempts_to_make_radio_work_l96_96221

theorem minimum_attempts_to_make_radio_work : 
  ∃ n, n = 12 ∧ (∀ (batteries : Finset ℕ), batteries.card = 8 → 
  (∃ charged_batteries uncharged_batteries, 
    charged_batteries.card = 4 ∧ 
    uncharged_batteries.card = 4 ∧ 
    charged_batteries ∪ uncharged_batteries = batteries) → 
  (∀ attempts : Finset (Finset ℕ), 
    attempts.card = n ∧ 
    (∀ attempt ∈ attempts, attempt.card = 2) → 
    ∃ attempt ∈ attempts, 
      (attempt ⊆ charged_batteries) := 
  sorry

end minimum_attempts_to_make_radio_work_l96_96221


namespace volume_of_tetrahedron_l96_96006

-- Define the given conditions of the problem.
def area_ABC := 150
def area_BCD := 90
def BC := 12
def angle_ABC_BCD := Real.pi / 4  -- 45 degrees in radians

-- Based on these conditions, the goal is to prove the volume of the tetrahedron.
theorem volume_of_tetrahedron :
  let h := 2 * area_BCD / BC in
  let h' := h * Real.sin angle_ABC_BCD in
  (1 / 3) * area_ABC * h' = 375 * Real.sqrt 2 :=
by
  let h := 2 * area_BCD / BC
  let h' := h * Real.sin angle_ABC_BCD
  have h_def : h = 15 := sorry  -- height from D to BC calculated in step 1 of solution
  have h'_def : h' = 15 * Real.sqrt 2 / 2 := sorry -- height from D to plane of ABC
  have volume_def : (1 / 3 * area_ABC * h') = 375 * Real.sqrt 2 := sorry 
  -- volume calculation as in step 3 of solution
  exact volume_def

end volume_of_tetrahedron_l96_96006


namespace select_independent_teams_l96_96987

theorem select_independent_teams :
  ∀ (teams : Finset ℕ), 
  teams.card = 20 → 
  (∀ u ∈ teams, ∃ v ∈ teams, u ≠ v ∧ u_has_played_with v) → 
  (∃ selected_teams : Finset ℕ, selected_teams.card = 10 ∧ ∀ (u v : ℕ), u ∈ selected_teams → v ∈ selected_teams → u ≠ v → ¬u_has_played_with v) :=
by
  intros teams teams_card plays_condition
  sorry

end select_independent_teams_l96_96987


namespace value_of_x_l96_96153

theorem value_of_x :
  let x := (8 * 5.4 - 0.6 * 10 / 1.2) ^ 2 in
  x = 1459.24 :=
by
  let x := (8 * 5.4 - 0.6 * 10 / 1.2) ^ 2
  sorry

end value_of_x_l96_96153


namespace trapezoid_problem_l96_96130

variables {Point : Type} [MetricSpace Point]

def is_trapezoid (A B C D : Point) : Prop :=
(line_parallel A D B C ∧ ¬line_parallel A B C D)

def is_parallel (L1 L2 : Line Point) : Prop :=
line_parallel L1 L2

def segment_eq (P1 P2 Q1 Q2 : Point) : Prop :=
(dist P1 P2 = dist Q1 Q2)

variables (A B C D E F : Point)

theorem trapezoid_problem
  (h_trap : is_trapezoid A B C D)
  (h_eq_bd_ad : dist B D = dist A D)
  (h_diag_intersect : line_intersect A C B D E)
  (h_parallel_ef_cd : is_parallel (line_through E F) (line_through C D))
  : dist B E = dist D F := 
sorry

end trapezoid_problem_l96_96130


namespace deduce_T1_cannot_deduce_T2_cannot_deduce_T3_l96_96354

def quib : Type := sorry -- Placeholder for the definition of 'quib'.
def daa : Type := sorry -- Placeholder for the definition of 'daa'.

-- Postulates
axiom P1 (q : quib) : set daa
axiom P2 (q1 q2 q3 : quib) (h1 : q1 ≠ q2) (h2 : q2 ≠ q3) (h3 : q1 ≠ q3) : (P1 q1) ∩ (P1 q2) ∩ (P1 q3) = {d} := sorry
axiom P3 (d : daa) : ∃ (q1 q2 q3 : quib), (d ∈ P1 q1) ∧ (d ∈ P1 q2) ∧ (d ∈ P1 q3) ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3
axiom P4 : ∃ (quibs : set quib), quibs.finite ∧ quibs.card = 6
axiom P5 : ∀ (q1 q2 q3 q4 : quib), (P1 q1) ∩ (P1 q2) ∩ (P1 q3) ∩ (P1 q4) = {d1, d2} := sorry

-- Theorems
def T1 : Prop := ∃S : set daa, S.finite ∧ S.card = 10
def T2 : Prop := ∀q : quib, ∃S : set daa, S.finite ∧ S.card = 4
def T3 : Prop := ∀d : daa, ∃d1 d2 : daa, d1 ≠ d2 ∧ ∀q : quib, d ∈ P1 q → d1 ∉ P1 q ∧ d2 ∉ P1 q

-- Goal
theorem deduce_T1 : T1 := sorry
theorem cannot_deduce_T2 : ¬ T2 := sorry
theorem cannot_deduce_T3 : ¬ T3 := sorry

end deduce_T1_cannot_deduce_T2_cannot_deduce_T3_l96_96354


namespace product_of_pairs_l96_96217

theorem product_of_pairs (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2015)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2014)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2015)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2014)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2015)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2014):
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -4 / 1007 :=
sorry

end product_of_pairs_l96_96217


namespace min_AB_value_l96_96352

theorem min_AB_value (t φ : ℝ) (hφ : 0 < φ ∧ φ < π) :
  let x := t * sin φ,
      y := 1 + t * cos φ,
      l : set (ℝ × ℝ) := {p | ∃ t, p = (t * sin φ, 1 + t * cos φ)},
      C : set (ℝ × ℝ) := {p | (p.fst) ^ 2 = 4 * (p.snd)}
  in
  ∃ t1 t2, l ∩ C = {(t1 * sin φ, 1 + t1 * cos φ), (t2 * sin φ, 1 + t2 * cos φ)} ∧
  |t1 - t2| = 4 :=
by {
  sorry
}

end min_AB_value_l96_96352


namespace hannah_payment_l96_96768

def costWashingMachine : ℝ := 100
def costDryer : ℝ := costWashingMachine - 30
def totalCostBeforeDiscount : ℝ := costWashingMachine + costDryer
def discount : ℝ := totalCostBeforeDiscount * 0.1
def finalCost : ℝ := totalCostBeforeDiscount - discount

theorem hannah_payment : finalCost = 153 := by
  simp [costWashingMachine, costDryer, totalCostBeforeDiscount, discount, finalCost]
  sorry

end hannah_payment_l96_96768


namespace systematic_sampling_ID_l96_96429

/-
Problem: In a total of 48 students with sample size 4 using systematic sampling, given that 
students with IDs 06, 30, 42 are in the sample, prove that the ID of the fourth student in the sample is 18.
-/

theorem systematic_sampling_ID :
  ∀ (total_students sample_size : ℕ) (ids_in_sample : List ℕ),
  total_students = 48 →
  sample_size = 4 →
  ids_in_sample = [6, 30, 42] →
  ∃ ID4, ID4 = 18 := 
by
  intros total_students sample_size ids_in_sample
  intros h_total h_sample h_ids
  use 18
  sorry -- Proof goes here

end systematic_sampling_ID_l96_96429


namespace angle_BAC_is_75_degrees_l96_96610

-- Define the structures and conditions of the problem
variables {A B C K L M N : Point}
variables {ω : Circle}
variables [InCircle B ω] [InCircle K ω] [InCircle L ω]
variables [TouchMidpoint M (Segment A C)]
variables [ArcCondition BL (not K) N]
variables [AngleEquals (L K N) (A C B)]
variables [EquilateralTriangle C K N]

-- Define the goal
theorem angle_BAC_is_75_degrees :
  ∠BAC = 75 :=
sorry

end angle_BAC_is_75_degrees_l96_96610


namespace minimal_segments_proof_l96_96357

open Nat

noncomputable def minimal_connected_segments (k : ℕ) : ℕ :=
  let n := 3 * k + 1
  let t_n_3 := 3 * k^2 + 2 * k
  let k_n := (9 * k^2 + 3 * k) / 2
  t_n_3 + n^2 + k_n + 1

theorem minimal_segments_proof (k : ℕ) (h : k ≥ 1) : 
  let n := 3 * k + 1 
  ∃ m, ∀ (points : Fin (n^2) → ℝ × ℝ), 
    (∀ (a b c : Fin (n^2)), a ≠ b → b ≠ c → a ≠ c → 
       ¬ collinear (points a) (points b) (points c)) → 
    minimal_connected_segments k = m := 
by
  sorry

end minimal_segments_proof_l96_96357


namespace polygon_sides_l96_96622

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 - 180 = 2190) : n = 15 :=
sorry

end polygon_sides_l96_96622


namespace min_value_x1_squared_plus_x2_squared_plus_x3_squared_l96_96878

theorem min_value_x1_squared_plus_x2_squared_plus_x3_squared
    (x1 x2 x3 : ℝ) 
    (h1 : 3 * x1 + 2 * x2 + x3 = 30) 
    (h2 : x1 > 0) 
    (h3 : x2 > 0) 
    (h4 : x3 > 0) : 
    x1^2 + x2^2 + x3^2 ≥ 125 := 
  by sorry

end min_value_x1_squared_plus_x2_squared_plus_x3_squared_l96_96878


namespace part1_part2_l96_96753

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - 5 * a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + 3

-- (1)
theorem part1 (x : ℝ) : abs (g x) < 8 → -4 < x ∧ x < 6 :=
by
  sorry

-- (2)
theorem part2 (a : ℝ) : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) → (a ≥ 0.4 ∨ a ≤ -0.8) :=
by
  sorry

end part1_part2_l96_96753


namespace smallest_N_l96_96026

/-- Find the smallest natural number N > 9 that is not divisible by 7,
  but becomes divisible by 7 when any one of its digits is replaced by the digit 7. --/
theorem smallest_N (N : ℕ) (hN_gt9: N > 9) (hN_not_div7: ¬ (∃ k : ℕ, N = 7 * k))
  (h_digits: ∀ d in digits N, d ≠ 0 ∧ d ≠ 7)
  (h_replace7: ∀ n : ℕ, (N = (replaceDigit N n 7) ∧ ∃ k : ℕ, (replaceDigit N n 7) = 7 * k)) :
  N = 13264513 := sorry

end smallest_N_l96_96026


namespace units_digit_p_plus_one_l96_96734

theorem units_digit_p_plus_one (p : ℕ) (h1 : p % 2 = 0) (h2 : p % 10 ≠ 0)
  (h3 : (p ^ 3) % 10 = (p ^ 2) % 10) : (p + 1) % 10 = 7 :=
  sorry

end units_digit_p_plus_one_l96_96734


namespace probability_different_color_l96_96701

theorem probability_different_color :
  let total_ways := Nat.choose 5 2,
      same_color_ways := 2,
      probability_same_color := same_color_ways / total_ways,
      probability_different_color := 1 - probability_same_color in
  probability_different_color = 4 / 5 :=
by
  -- Define the individual components
  let total_ways := Nat.choose 5 2
  let same_color_ways := 2
  let probability_same_color := same_color_ways / total_ways
  let probability_different_color := 1 - probability_same_color
  -- Provide the expected result
  sorry

end probability_different_color_l96_96701


namespace slope_angle_line_l96_96381
open Real

theorem slope_angle_line (x y : ℝ) :
  x + sqrt 3 * y - 1 = 0 → ∃ θ : ℝ, θ = 150 ∧
  ∃ (m : ℝ), m = -sqrt 3 / 3 ∧ θ = arctan m :=
by
  sorry

end slope_angle_line_l96_96381


namespace hannah_total_payment_l96_96765

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def total_cost_before_discount : ℝ := washing_machine_cost + dryer_cost
def discount : ℝ := 0.10
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount)

theorem hannah_total_payment : total_cost_after_discount = 153 := by
  sorry

end hannah_total_payment_l96_96765


namespace minimum_value_of_a_l96_96787

theorem minimum_value_of_a :
  (∀ x : ℝ, x > 0 → (a : ℝ) * x * Real.exp x - x - Real.log x ≥ 0) → a ≥ 1 / Real.exp 1 :=
by
  sorry

end minimum_value_of_a_l96_96787


namespace function_has_two_common_points_l96_96073

theorem function_has_two_common_points (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (1/3 * x1^3 - 3 * x1 + m = 0) ∧ (1/3 * x2^3 - 3 * x2 + m = 0)) →
  m = -2 * real.sqrt 3 ∨ m = 2 * real.sqrt 3 :=
by
  sorry -- Proof placeholder

end function_has_two_common_points_l96_96073


namespace expression_values_l96_96409

variable {a b c : ℚ}

theorem expression_values (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = 2) ∨ 
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = -2) := 
sorry

end expression_values_l96_96409


namespace main_theorem_l96_96769

variable (x : ℤ)

def H : ℤ := 12 - (3 + 7) + x
def T : ℤ := 12 - 3 + 7 + x

theorem main_theorem : H - T + x = -14 + x :=
by
  sorry

end main_theorem_l96_96769


namespace sequential_inequality_probability_l96_96033

open MeasureTheory

noncomputable def uniform_distribution (i : ℕ) := 
MeasureTheory.ProbabilityMassFunction.uniform (Set.Icc 0 (i ^ 2))

def sequential_probability (n : ℕ) : ℝ := 
∏ i in Finset.range (n - 1), 
((∫ x in 0..(i ^ 2), ∫ y in x..((i + 1) ^ 2), 
  (1 / (i ^ 2)) * (1 / ((i + 1) ^ 2)) d(y) d(x)) * (1 / (1 - x)))

theorem sequential_inequality_probability : 
(∫ x in 0..1, ∫ y in x..4, (1 / 1 ^ 2) * (1 / 2 ^ 2) d(y) d(x)) *
(∫ x in 0..4, ∫ y in x..9, (1 / 2 ^ 2) * (1 / 3 ^ 2) d(y) d(x)) *
(∫ x in 0..9, ∫ y in x..16, (1 / 3 ^ 2) * (1 / 4 ^ 2) d(y) d(x)) *
(∫ x in 0..16, ∫ y in x..25, (1 / 4 ^ 2) * (1 / 5 ^ 2) d(y) d(x)) *
(∫ x in 0..25, ∫ y in x..36, (1 / 5 ^ 2) * (1 / 6 ^ 2) d(y) d(x)) *
(∫ x in 0..36, ∫ y in x..49, (1 / 6 ^ 2) * (1 / 7 ^ 2) d(y) d(x)) *
(∫ x in 0..49, ∫ y in x..64, (1 / 7 ^ 2) * (1 / 8 ^ 2) d(y) d(x)) *
(∫ x in 0..64, ∫ y in x..81, (1 / 8 ^ 2) * (1 / 9 ^ 2) d(y) d(x)) *
(∫ x in 0..81, ∫ y in x..100, (1 / 9 ^ 2) * (1 / 10 ^ 2) d(y) d(x)) ≈ 0.003679 :=
sorry

end sequential_inequality_probability_l96_96033


namespace residue_mod_2021_l96_96027

def series_sum : ℤ :=
  (List.range 2020).sum (λ n, if n % 2 = 0 then (n + 1 : ℤ) else -(n + 1 : ℤ))

theorem residue_mod_2021 : series_sum % 2021 = 1011 := by
  sorry

end residue_mod_2021_l96_96027


namespace remainder_when_divided_by_3x_minus_6_l96_96559

def polynomial (x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 9 * x^4 + 3 * x^3 - 7

def evaluate_at (f : ℝ → ℝ) (a : ℝ) : ℝ := f a

theorem remainder_when_divided_by_3x_minus_6 :
  evaluate_at polynomial 2 = 897 :=
by
  -- Compute this value manually or use automated tools
  sorry

end remainder_when_divided_by_3x_minus_6_l96_96559


namespace non_similar_quadrilaterals_count_l96_96403

theorem non_similar_quadrilaterals_count :
  ∃ (n d: ℕ), n = 90 ∧ 1 ≤ d ∧ d ≤ 29 ∧ 
  ∀ a b c d : ℕ, 
  a = n - 3 * d ∧ b = n - d ∧ c = n + d ∧ d = n + 3 * d ∧ 
  a + b + c + d = 360 ∧ 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d
  → 29 := 
begin
  sorry
end

end non_similar_quadrilaterals_count_l96_96403


namespace part_I_part_II_l96_96905

def f (x a : ℝ) := |x - a| + |x - 1|

theorem part_I {x : ℝ} : Set.Icc 0 4 = {y | f y 3 ≤ 4} := 
sorry

theorem part_II {a : ℝ} : (∀ x, ¬ (f x a < 2)) ↔ a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
sorry

end part_I_part_II_l96_96905


namespace arithmetic_sequence_sum_l96_96102

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_roots : (a 3) * (a 10) - 3 * (a 3 + a 10) - 5 = 0) : a 5 + a 8 = 3 :=
sorry

end arithmetic_sequence_sum_l96_96102


namespace f_of_5_l96_96936

/- The function f(x) is defined by f(x) = x^2 - x. Prove that f(5) = 20. -/
def f (x : ℤ) : ℤ := x^2 - x

theorem f_of_5 : f 5 = 20 := by
  sorry

end f_of_5_l96_96936


namespace find_z_m_n_l96_96713

noncomputable def z := (5 : ℂ) - (3 : ℂ) * complex.I -- Representing 5 - 3i
noncomputable def m := (-9 : ℝ)
noncomputable def n := (30 : ℝ)

theorem find_z_m_n (h1 : |1 - z| + z = 10 - 3 * complex.I) (h2 : z^2 + m * z + n = 1 - 3 * complex.I) :
  z = 5 - 3 * complex.I ∧ m = -9 ∧ n = 30 :=
by {
  sorry
}

end find_z_m_n_l96_96713


namespace highest_average_speed_from_15_to_16_l96_96644

def average_speed (Δd : ℝ) (Δt : ℝ) : ℝ := Δd / Δt

variable Δd_0_1 : ℝ -- change in distance from 0-1 hours
variable Δd_1_2 : ℝ -- change in distance from 1-2 hours
variable Δd_14_15 : ℝ -- change in distance from 14-15 hours
variable Δd_15_16 : ℝ -- change in distance from 15-16 hours
variable Δd_23_24 : ℝ -- change in distance from 23-24 hours

theorem highest_average_speed_from_15_to_16 :
  average_speed Δd_15_16 1 > average_speed Δd_0_1 1 ∧
  average_speed Δd_15_16 1 > average_speed Δd_1_2 1 ∧
  average_speed Δd_15_16 1 > average_speed Δd_14_15 1 ∧
  average_speed Δd_15_16 1 > average_speed Δd_23_24 1 :=
sorry

end highest_average_speed_from_15_to_16_l96_96644


namespace coins_in_second_stack_l96_96670

theorem coins_in_second_stack (total_coins : ℕ) (stack1_coins : ℕ) (stack2_coins : ℕ) 
  (H1 : total_coins = 12) (H2 : stack1_coins = 4) : stack2_coins = 8 :=
by
  -- The proof is omitted.
  sorry

end coins_in_second_stack_l96_96670


namespace merill_has_30_marbles_l96_96477

variable (M E : ℕ)

-- Conditions
def merill_twice_as_many_as_elliot : Prop := M = 2 * E
def together_five_fewer_than_selma : Prop := M + E = 45

theorem merill_has_30_marbles (h1 : merill_twice_as_many_as_elliot M E) (h2 : together_five_fewer_than_selma M E) : M = 30 := 
by
  sorry

end merill_has_30_marbles_l96_96477


namespace cab_speed_fraction_l96_96248

theorem cab_speed_fraction :
  ∀ (S R : ℝ),
    (75 * S = 90 * R) →
    (R / S = 5 / 6) :=
by
  intros S R h
  sorry

end cab_speed_fraction_l96_96248


namespace sum_of_digits_of_smallest_N_l96_96138

-- Defining the conditions
def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k
def P (N : ℕ) : ℚ := ((2/3 : ℚ) * N * (1/3 : ℚ) * N) / ((N + 2) * (N + 3))
def S (n : ℕ) : ℕ := (n % 10) + ((n / 10) % 10) + (n / 100)

-- The statement of the problem
theorem sum_of_digits_of_smallest_N :
  ∃ N : ℕ, is_multiple_of_6 N ∧ P N < (4/5 : ℚ) ∧ S N = 6 :=
sorry

end sum_of_digits_of_smallest_N_l96_96138


namespace positive_diff_arithmetic_sequence_l96_96226

theorem positive_diff_arithmetic_sequence :
  let a : ℕ := 2
  let d : ℕ := 7
  let a_n (n : ℕ) := a + (n - 1) * d
  let a_2050 := a_n 2050
  let a_2060 := a_n 2060
  |a_2060 - a_2050| = 70 := by
  sorry

end positive_diff_arithmetic_sequence_l96_96226


namespace problem1_problem2_problem3_problem4_problem5_l96_96341

-- Problem (Ⅰ)
theorem problem1 :
  let students := ["A", "B", "C", "D"],
      projects := ["P1", "P2", "P3"] in
  fintype.card (students → projects) = 81 := 
by sorry

-- Problem (Ⅱ)
theorem problem2 :
  let students := ["A", "B", "C", "D"],
      projects := ["P1", "P2", "P3"] in
  fintype.card {f : students → projects // 
    f "A" = "P1" ∧ f "B" = "P2"} = 9 := 
by sorry

-- Problem (Ⅲ)
theorem problem3 :
  let students := ["A", "B", "C", "D"],
      projects := ["P1", "P2", "P3"] in
  fintype.card {f : students → projects // 
    f "A" = f "B" ∧ f "C" ≠ "P1"} = 18 := 
by sorry

-- Problem (Ⅳ)
theorem problem4 :
  let students := ["A", "B", "C", "D"],
      projects := ["P1", "P2", "P3"] in
  fintype.card {f : students → projects // 
    ∀ p ∈ projects, ∃ s ∈ students, f s = p} = 36 := 
by sorry

-- Problem (Ⅴ)
theorem problem5 :
  let students := ["A", "B", "C", "D"],
      projects := ["P1", "P2", "P3"] in
  fintype.card {f : students → projects // 
    f "A" ≠ "P1" ∧ (multiset.card (f⁻¹ '' {"P2"}) = 2 ∧ multiset.card (f⁻¹ '' {"P3"}) = 2) ∨ 
    (multiset.card (f⁻¹ '' {"P2"}) = 1 ∧ multiset.card (f⁻¹ '' {"P3"}) = 1)} = 12 :=
by sorry

end problem1_problem2_problem3_problem4_problem5_l96_96341


namespace odd_handshakes_even_l96_96764

theorem odd_handshakes_even (V : Type) (E : set (V × V)) :
  let G := (V, E)
  ∑ v in V, deg(G, v) = 2 * |E| :=
  let V_odd := { v : V | deg(G, v) % 2 = 1 } in
  even (card V_odd) :=
by
  sorry

end odd_handshakes_even_l96_96764


namespace tangent_circle_radius_l96_96923

theorem tangent_circle_radius (M A B O : Type) 
  (tangent_to_circle_at : M → A → Prop)
  (tangent_to_circle_at' : M → B → Prop)
  (center_of_circle : O)
  (angle_AMB : ℝ)
  (AB_length : ℝ) :
  (tan (angle_AMB / 2) = AB_length / (2 * (radius : ℝ))) →
  (radius = AB_length / (2 * (cos(angle_AMB / 2)))) :=
sorry

end tangent_circle_radius_l96_96923


namespace probability_adjacent_faces_is_five_over_eighteen_l96_96272

noncomputable def probability_adjacent_faces : ℚ :=
  let total_outcomes := (6 : ℕ) * (6 : ℕ)
  let adjacent_outcomes := 10
  adjacent_outcomes / total_outcomes

theorem probability_adjacent_faces_is_five_over_eighteen :
  probability_adjacent_faces = 5 / 18 :=
  by
    have h1 : total_outcomes = 36 := rfl
    have h2 : adjacent_outcomes = 10 := rfl
    sorry

end probability_adjacent_faces_is_five_over_eighteen_l96_96272


namespace angle_between_vectors_l96_96741

variables {a b : ℝ}
variables (u v : ℝ → ℝ → Prop) (dot : ℝ → ℝ → ℝ → ℝ)

-- Define the vectors a and b as non-zero vectors.
axiom ha : a ≠ 0
axiom hb : b ≠ 0

-- Define the orthogonality conditions.
axiom orthogonality1 : dot (a + 3*b) (7*a - 5*b) = 0
axiom orthogonality2 : dot (a - 4*b) (7*a - 2*b) = 0

-- Define the cosine of the angle theta between vectors a and b.
noncomputable def cos_theta (a b : ℝ) : ℝ := (dot a b) / (||a|| * ||b||)

-- The goal is to prove that the angle between vectors a and b is 60 degrees.
theorem angle_between_vectors : cos_theta a b = 1/2 := 
sorry

end angle_between_vectors_l96_96741


namespace robbers_divide_and_choose_l96_96771

/-- A model of dividing loot between two robbers who do not trust each other -/
def divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) : Prop :=
  ∀ (B : ℕ → ℕ), B (max P1 P2) ≥ B P1 ∧ B (max P1 P2) ≥ B P2

theorem robbers_divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) :
  divide_and_choose P1 P2 A :=
sorry

end robbers_divide_and_choose_l96_96771


namespace determine_x_l96_96773

theorem determine_x (x y : Real) (h1 : 12 * 3^x = 4^(y + 5)) (h2 : y = -3) : x = Real.log 4 / Real.log 3 - 1 := 
by 
  sorry

end determine_x_l96_96773


namespace chair_capacity_l96_96431

theorem chair_capacity
  (total_chairs : ℕ)
  (total_board_members : ℕ)
  (not_occupied_fraction : ℚ)
  (occupied_people_per_chair : ℕ)
  (attending_board_members : ℕ)
  (total_chairs_eq : total_chairs = 40)
  (not_occupied_fraction_eq : not_occupied_fraction = 2/5)
  (occupied_people_per_chair_eq : occupied_people_per_chair = 2)
  (attending_board_members_eq : attending_board_members = 48)
  : total_board_members = 48 := 
by
  sorry

end chair_capacity_l96_96431


namespace repeating_decimal_difference_l96_96459

theorem repeating_decimal_difference :
  let G := 0.862862862 -- infinite repeating decimal 0.862862862...
  in (∃ (n d : ℕ), G = n / d ∧ Nat.gcd n d = 1 ∧ d - n = 137) :=
sorry

end repeating_decimal_difference_l96_96459


namespace additional_money_needed_l96_96657

-- Define the initial conditions as assumptions
def initial_bales : ℕ := 15
def previous_cost_per_bale : ℕ := 20
def multiplier : ℕ := 3
def new_cost_per_bale : ℕ := 27

-- Define the problem statement
theorem additional_money_needed :
  let initial_cost := initial_bales * previous_cost_per_bale 
  let new_bales := initial_bales * multiplier
  let new_cost := new_bales * new_cost_per_bale
  new_cost - initial_cost = 915 :=
by
  sorry

end additional_money_needed_l96_96657


namespace intersection_A_B_l96_96984

noncomputable theory

open Set

def A : Set ℝ := {x | |x| ≤ 1}
def B : Set ℝ := {x | ∃ y:ℝ, y = x^2 ∧ x ∈ univ}

theorem intersection_A_B :
  A ∩ B = {x: ℝ | 0 ≤ x ∧ x ≤ 1} :=
by
  ext
  sorry

end intersection_A_B_l96_96984


namespace area_of_transformed_parallelogram_l96_96510

variables {R : Type*} [NormedField R] [NormedSpace R (R^3)]
variables (a b c : R^3)

-- Condition: |a × b| = 12
axiom area_a_cross_b : ∥a × b∥ = 12

-- Condition: c is orthogonal to both a and b
axiom c_orthogonal_to_a_b : 
  (c ⬝ a = 0) ∧ (c ⬝ b = 0) -- dot product orthogonality

-- Theorem Statement: The area of the parallelogram generated by 3a + 4b + c and 2a - 6b + 2c is 120.
theorem area_of_transformed_parallelogram : 
  ∥(3 • a + 4 • b + c) × (2 • a - 6 • b + 2 • c)∥ = 120 :=
sorry

end area_of_transformed_parallelogram_l96_96510


namespace combined_volume_of_all_cubes_l96_96475

/-- Lily has 4 cubes each with side length 3, Mark has 3 cubes each with side length 4,
    and Zoe has 2 cubes each with side length 5. Prove that the combined volume of all
    the cubes is 550. -/
theorem combined_volume_of_all_cubes 
  (lily_cubes : ℕ := 4) (lily_side_length : ℕ := 3)
  (mark_cubes : ℕ := 3) (mark_side_length : ℕ := 4)
  (zoe_cubes : ℕ := 2) (zoe_side_length : ℕ := 5) :
  (lily_cubes * lily_side_length ^ 3) + 
  (mark_cubes * mark_side_length ^ 3) + 
  (zoe_cubes * zoe_side_length ^ 3) = 550 :=
by
  have lily_volume : ℕ := lily_cubes * lily_side_length ^ 3
  have mark_volume : ℕ := mark_cubes * mark_side_length ^ 3
  have zoe_volume : ℕ := zoe_cubes * zoe_side_length ^ 3
  have total_volume : ℕ := lily_volume + mark_volume + zoe_volume
  sorry

end combined_volume_of_all_cubes_l96_96475


namespace largest_of_five_l96_96406

theorem largest_of_five {a b : ℝ} (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  max (max (max (max (a^b) (b^a)) (Real.log a b)) b) (Real.log b a) = Real.log b a :=
by
  sorry

end largest_of_five_l96_96406


namespace cubic_roots_sum_of_cubes_l96_96411

theorem cubic_roots_sum_of_cubes (r s t a b c : ℚ) 
  (h1 : r + s + t = a) 
  (h2 : r * s + r * t + s * t = b)
  (h3 : r * s * t = c) 
  (h_poly : ∀ x : ℚ, x^3 - a*x^2 + b*x - c = 0 ↔ (x = r ∨ x = s ∨ x = t)) :
  r^3 + s^3 + t^3 = a^3 - 3 * a * b + 3 * c :=
sorry

end cubic_roots_sum_of_cubes_l96_96411


namespace total_books_is_595_l96_96043

-- Definitions of the conditions
def satisfies_conditions (a : ℕ) : Prop :=
  ∃ R L : ℕ, a = 12 * R + 7 ∧ a = 25 * L - 5 ∧ 500 < a ∧ a < 650

-- The theorem statement
theorem total_books_is_595 : ∃ a : ℕ, satisfies_conditions a ∧ a = 595 :=
by
  use 595
  split
  · apply exists.intro 49, exists.intro 24, split
    -- a = 12R + 7
    · exact rfl
    -- a = 25L - 5
    · exact rfl
  -- Next check 500 < a and a < 650
  split
  · exact nat.lt_of_le_of_lt (by norm_num) (by norm_num)
  · exact nat.lt_of_le_of_lt (by norm_num) (by norm_num)

end total_books_is_595_l96_96043


namespace Mike_earnings_l96_96156

def total_games : ℕ := 15
def non_working_games : ℕ := 9
def price_per_game : ℕ := 5

theorem Mike_earnings : 
  let working_games := total_games - non_working_games in
  let earned_amount := working_games * price_per_game in
  earned_amount = 30 :=
by 
  let working_games := total_games - non_working_games 
  let earned_amount := working_games * price_per_game 
  have : working_games = 6 := by sorry
  have : earned_amount = 30 := by sorry
  exact this

end Mike_earnings_l96_96156


namespace jellybean_problem_l96_96606

theorem jellybean_problem (G O : ℕ) (h1 : O = G - 1) (h2 : 8 + G + O = 27) : G - 8 = 2 :=
by
  have h3 : 8 + G + (G - 1) = 27 := by rw [h1]
  sorry

end jellybean_problem_l96_96606


namespace find_a_b_find_c_range_l96_96884

noncomputable def f (a b c x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x + 8 * c

theorem find_a_b (a b c : ℝ) (extreme_x1 extreme_x2 : ℝ) (h1 : extreme_x1 = 1) (h2 : extreme_x2 = 2) 
  (h3 : (deriv (f a b c) 1) = 0) (h4 : (deriv (f a b c) 2) = 0) : 
  a = -3 ∧ b = 4 :=
by sorry

theorem find_c_range (c : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f (-3) 4 c x < c^2) : 
  c ∈ Set.Iio (-1) ∪ Set.Ioi 9 :=
by sorry

end find_a_b_find_c_range_l96_96884


namespace geometric_solid_is_tetrahedron_l96_96192

-- Definitions based on the conditions provided
def top_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def front_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def side_view_is_triangle : Prop := sorry -- Placeholder for the actual definition

-- Theorem statement to prove the geometric solid is a triangular pyramid
theorem geometric_solid_is_tetrahedron 
  (h_top : top_view_is_triangle)
  (h_front : front_view_is_triangle)
  (h_side : side_view_is_triangle) :
  -- Conclusion that the solid is a triangular pyramid (tetrahedron)
  is_tetrahedron :=
sorry

end geometric_solid_is_tetrahedron_l96_96192


namespace units_digit_of_7_pow_1000_l96_96004

theorem units_digit_of_7_pow_1000 : 
  (7 ^ 1000) % 10 = 1 :=
by
  have cycle : ∀ n, (7 ^ (4 * n + 1)) % 10 = 7 ∧ 
                  (7 ^ (4 * n + 2)) % 10 = 9 ∧ 
                  (7 ^ (4 * n + 3)) % 10 = 3 ∧ 
                  (7 ^ (4 * n + 4)) % 10 = 1, from sorry,
  have key_calc : (10 ^ 3) % 4 = 0 := by norm_num,
  have exp_mod_4 : (7 ^ 1000) % 10 = (7 ^ (4 * 250)) % 10, from sorry,
  exact (cycle 250).right.right.right

end units_digit_of_7_pow_1000_l96_96004


namespace number_of_internal_cubes_l96_96535

theorem number_of_internal_cubes :
  ∀ l w h : ℕ,
    w = 2 * l →
    w = 2 * h →
    w = 6 →
    l * w * h = 54 :=
by
  intros l w h h_w2l h_w2h h_w
  rw [h_w2l, h_w2h] at h_w
  have l_value := (h_w.symm ▸ rfl : l = 3)
  have h_value := (h_w.symm ▸ rfl : h = 3)
  rw [l_value, h_w, h_value]
  norm_num

end number_of_internal_cubes_l96_96535


namespace g_g_g_g_2_eq_16_l96_96306

def g (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 else 5 * x + 1

theorem g_g_g_g_2_eq_16 : g (g (g (g 2))) = 16 := by
  sorry

end g_g_g_g_2_eq_16_l96_96306


namespace original_students_on_second_bus_l96_96504

theorem original_students_on_second_bus (x : ℕ) 
  (h1 : 38) 
  (h2 : x - 4 = 40) : 
  x = 44 := 
begin
  sorry
end

end original_students_on_second_bus_l96_96504


namespace unique_pair_exists_for_each_n_l96_96899

theorem unique_pair_exists_for_each_n (n : ℕ) (h : n > 0) : 
  ∃! (a b : ℕ), a > 0 ∧ b > 0 ∧ n = (a + b - 1) * (a + b - 2) / 2 + a :=
sorry

end unique_pair_exists_for_each_n_l96_96899


namespace range_S_l96_96101

theorem range_S {a b c S : ℝ} 
  (h1 : a ≠ 0)
  (h2 : ∀ (x : ℝ), y = a * x ^ 2 + b * x + c)
  (h3 : ∃ c, y = a * 0 ^ 2 + b * 0 + c = 1)
  (h4 : ∃ c, y = a * (-1) ^ 2 + b * (-1) + c = 0)
  (h5 : a = b - 1)
  (h6 : c = 1)
  (h7 : S = a + b + c) :
  0 < S ∧ S < 2 := 
sorry

end range_S_l96_96101


namespace find_q_10_l96_96149

noncomputable def q (x : ℝ) : ℝ := sorry

theorem find_q_10 (hq : ∀ x, (q(x))^2 - x = 0 → x = 2 ∨ x = -2 ∨ x = 5 ∨ x = 7) : 
  q (2) = real.sqrt(2) ∧
  q (-2) = -real.sqrt(2) ∧
  q (5) = real.sqrt(5) ∧
  q (7) = real.sqrt(7) ∧
  ∃ (a b c d : ℝ), 
    q (x) = a * x^3 + b * x^2 + c * x + d ∧ 
    q (10) = 1000 * a + 100 * b + 10 * c + d :=
begin
  sorry
end

end find_q_10_l96_96149


namespace total_attended_ball_lt_fifty_l96_96850

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l96_96850


namespace eat_cells_to_fraction_l96_96360

noncomputable def large_chocolate_bar (a b n : ℕ) : ℕ := a * b * n ^ 10

theorem eat_cells_to_fraction (m n a b : ℕ) (h1 : m < n) (h2 : n ^ 5 ≤ a * n ^ 5) :
  ∃ c d : ℕ, c * d = large_chocolate_bar a b m n ∧ c * d = a * b * (m ^ 10) :=
  sorry

end eat_cells_to_fraction_l96_96360


namespace value_of_nested_radical_l96_96695

def nested_radical : ℝ := 
  sorry -- Definition of the recurring expression is needed here, let's call it x
  
theorem value_of_nested_radical :
  (nested_radical = 5) :=
sorry -- The actual proof steps will be written here.

end value_of_nested_radical_l96_96695


namespace hyperbola_standard_equation_l96_96371

/-- Given a hyperbola passes through the point (4, sqrt 3) and its asymptotes are represented by
the equations y = ± 1/2*x, prove that the standard equation of the hyperbola is 
1/4*x^2 - y^2 = 1. -/
theorem hyperbola_standard_equation :
  (∃ λ : ℝ, ∀ x y : ℝ, (x = 4 ∧ y = sqrt 3 ∧ y^2 - (1/4) * x^2 = λ) →
  λ = -1) → (∀ x y : ℝ, (y^2 - (1/4) * x^2 = -1) ↔ (1/4 * x^2 - y^2 = 1)) :=
sorry

end hyperbola_standard_equation_l96_96371


namespace total_metal_wasted_l96_96268

noncomputable def wasted_metal (a b : ℝ) (h : b ≤ 2 * a) : ℝ := 
  2 * a * b - (b ^ 2 / 2)

theorem total_metal_wasted (a b : ℝ) (h : b ≤ 2 * a) : 
  wasted_metal a b h = 2 * a * b - b ^ 2 / 2 :=
sorry

end total_metal_wasted_l96_96268


namespace sum_reciprocal_primes_l96_96031

def is_prime (p: ℕ) : Prop := Nat.Prime p

def largest_prime_leq (n: ℕ) : ℕ := 
  Nat.findGreatest (λ p, p ≤ n ∧ is_prime p)

def smallest_prime_gt (n: ℕ) : ℕ := 
  Nat.find (λ p, n < p ∧ is_prime p)

def u (n: ℕ) : ℕ := largest_prime_leq n
def v (n: ℕ) : ℕ := smallest_prime_gt n

theorem sum_reciprocal_primes :
  \[ \sum_{n=2}^{2010} \frac{1}{u(n) * v(n)} = \frac{1}{2} - \frac{1}{2011} \]
  :=
  sorry

end sum_reciprocal_primes_l96_96031


namespace exists_another_wrapper_infinitely_many_wrappers_l96_96279

-- Wrapper definition and conditions.
structure Wrapper (width height : ℝ) :=
(area_eq_2 : width * height = 2)

def is_standard_wrapper (w : Wrapper) : Prop :=
  w.width = 2 ∧ w.height = 1 ∨ w.width = sqrt 2 ∧ w.height = sqrt 2

-- Part a: Prove that there are other wrappers.
theorem exists_another_wrapper : ∃ (w : Wrapper), ¬ is_standard_wrapper w :=
  sorry

-- Part b: Prove that there are infinitely many wrappers.
theorem infinitely_many_wrappers : ∃ (f : ℕ → Wrapper), function.injective f :=
  sorry

end exists_another_wrapper_infinitely_many_wrappers_l96_96279


namespace distance_between_points_l96_96224

def point1 : ℝ × ℝ := (-3/2, -7/2)
def point2 : ℝ × ℝ := (5/2, -11/2)

theorem distance_between_points : 
  (Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2)) = 2 * Real.sqrt 5 :=
by
  sorry

end distance_between_points_l96_96224


namespace cans_per_bag_l96_96895

theorem cans_per_bag (bags_saturday bags_sunday total_cans : ℕ) (h1 : bags_saturday = 6) (h2 : bags_sunday = 3) (h3 : total_cans = 72) :
  (total_cans) / (bags_saturday + bags_sunday) = 8 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end cans_per_bag_l96_96895


namespace distance_to_other_focus_l96_96415

-- Definitions of the conditions
def is_on_ellipse (P : ℝ × ℝ) : Prop := (P.1^2 / 100) + (P.2^2 / 36) = 1

def distance_to_focus (P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2)

-- The proof statement
theorem distance_to_other_focus (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) 
  (h1 : is_on_ellipse P)
  (h2 : distance_to_focus P F₁ = 6) 
  (h3 : F₁ = (c, 0)) -- assuming first focus is (c, 0)
  (h4 : F₂ = (-c, 0)) -- assuming second focus is (-c, 0)
  (h5 : c^2 = a^2 - b^2) -- where a = 10 and b = 6 in this case 
  : distance_to_focus P F₂ = 14 :=
by sorry

end distance_to_other_focus_l96_96415


namespace find_n_l96_96199

-- Define the arithmetic sequence terms
def A := log a
def B := log b

def term1 := 3 * A + 7 * B
def term2 := 5 * A + 12 * B
def term3 := 8 * A + 15 * B

-- Arithmetic sequence condition
def arithmetic_sequence := term2 - term1 = term3 - term2

-- Define the 12-th term
def T12 := log (b ^ n)

theorem find_n :
  arithmetic_sequence →
  T12 = log (b ^ 112) →
  n = 112 :=
by
  intros h1 h2
  rw [arithmetic_sequence, T12] at *
  sorry

end find_n_l96_96199


namespace increasing_iff_0_lt_a_le_1_max_value_f_x_on_Ioc_0_1_l96_96922

variable {a x : ℝ}
variable {f : ℝ → ℝ}
variable (h_cond1 : x ∈ Set.Ioc 0 1)
variable (h_cond2 : a ≠ 0)

def f_def (x : ℝ) : ℝ := -a * x + x + a

theorem increasing_iff_0_lt_a_le_1 : (∀ x ∈ Set.Ioc 0 1, 0 < f_def a x) ↔ (0 < a ∧ a ≤ 1) :=
by sorry

theorem max_value_f_x_on_Ioc_0_1 (h_a : 0 < a ∧ a ≤ 1) : ∀ x ∈ Set.Ioc 0 1, finset.sup' (Set.Ioc 0 1) (Set.nonempty_Ioc.2 h_cond1) f_def = 1 :=
by sorry

end increasing_iff_0_lt_a_le_1_max_value_f_x_on_Ioc_0_1_l96_96922


namespace find_13_numbers_l96_96013

theorem find_13_numbers :
  ∃ (a : Fin 13 → ℕ),
    (∀ i, a i % 21 = 0) ∧
    (∀ i j, i ≠ j → ¬(a i ∣ a j) ∧ ¬(a j ∣ a i)) ∧
    (∀ i j, i ≠ j → (a i ^ 5) % (a j ^ 4) = 0) :=
sorry

end find_13_numbers_l96_96013


namespace least_base_ten_number_with_seven_digits_in_base_three_l96_96561

-- Define the statement of the problem
theorem least_base_ten_number_with_seven_digits_in_base_three :
  ∃ (n : ℕ), (∀ k, k < n → digits 3 k < 7) ∧ digits 3 n = 7 := 
begin
  use 729,
  split,
  { -- Prove that for any k < 729, its ternary representation has less than 7 digits
    intros k hk,
    have : k < 3^6, by linarith,
    rw [← nat.lt_pow_self 3 six_pos],
    exact lt_of_lt_of_le this (nat.digits_fself_le 3 k),
  },
  { -- Prove that 729 has exactly 7 digits in its ternary representation
    rw [nat.digits_eq_len, list.length],
    exact digits_eq_len_base 10 3,
    sorry
  }
end

end least_base_ten_number_with_seven_digits_in_base_three_l96_96561


namespace length_AD_l96_96119

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [NormedGroup ℝ]

noncomputable def distance_between_points (a b : ℝ) : ℝ := (a - b).abs

-- Given conditions
variables (AB BC CD : ℝ)
axiom h1 : AB = 5
axiom h2 : BC = 8
axiom h3 : CD = 20
axiom angle_B_right : ∀ B' C', (∠ B' - C') = 90
axiom angle_C_right : ∀ C' D', (∠ C' - D') = 90

-- Question to prove
theorem length_AD : distance_between_points A D = 17 :=
by
  sorry

end length_AD_l96_96119


namespace count_valid_subsets_l96_96331

open Finset
open Fintype

def is_valid_subset (S : Finset ℕ) : Prop :=
  S.nonempty ∧ ∀ T ⊆ S, T.sum id ≠ 10

theorem count_valid_subsets : ∃ (n : ℕ), n = 34 ∧ (univ.filter is_valid_subset).card = n :=
by
  sorry

end count_valid_subsets_l96_96331


namespace envelope_fee_count_l96_96654

def envelope_fee_required (l h : ℝ) : Prop :=
  l / h < 1.5 ∨ l / h > 3.0

def envelope_X : ℝ × ℝ := (3, 2)
def envelope_Y : ℝ × ℝ := (10, 3)
def envelope_Z : ℝ × ℝ := (5, 5)
def envelope_W : ℝ × ℝ := (15, 4)

def count_envelopes_requiring_fee (envelopes : List (ℝ × ℝ)) : ℕ :=
  envelopes.count (λ e => envelope_fee_required e.1 e.2)

theorem envelope_fee_count : count_envelopes_requiring_fee [envelope_X, envelope_Y, envelope_Z, envelope_W] = 3 := by
  sorry -- Proof to be provided

end envelope_fee_count_l96_96654


namespace domain_of_h_h_is_even_l96_96386

def f (a x : ℝ) : ℝ := log a (x + 1)
def g (a x : ℝ) : ℝ := log a (1 - x)

def h (a x : ℝ) : ℝ := f a x + g a x

theorem domain_of_h (a : ℝ) (ha : 0 < a ∧ a ≠ 1) :
  ∀ x : ℝ, -1 < x ∧ x < 1 ↔ (x + 1 > 0 ∧ 1 - x > 0) := by
{
  sorry
}

theorem h_is_even (a : ℝ) (ha : 0 < a ∧ a ≠ 1) :
  ∀ x : ℝ, h a (-x) = h a x := by
{
  sorry
}

end domain_of_h_h_is_even_l96_96386


namespace total_attended_ball_lt_fifty_l96_96853

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l96_96853


namespace parking_fee_l96_96203

/-- The parking fee for this parking lot is 500 won for 30 minutes, and 200 won is added 
for every 10 minutes after 30 minutes. If Ha-Young's father parked in this parking lot for 
1 hour and 20 minutes, the total parking fee is 1500 won. -/
theorem parking_fee
  (parking_time : ℕ) -- parking time in minutes
  (initial_fee : ℕ) (additional_fee : ℕ) 
  (initial_time : ℕ) (additional_time : ℕ) :
  parking_time = 80 → 
  initial_fee = 500 →
  additional_fee = 200 →
  initial_time = 30 →
  additional_time = 10 →
  let additional_time_fee := ((parking_time - initial_time) / additional_time) * additional_fee in
  let total_fee := initial_fee + additional_time_fee in
  total_fee = 1500 :=
begin
  intros ht hf1 hf2 ht1 ht2,
  have h_additional_time_fee : additional_time_fee = 1000,
  { rw [ht, ht1, ht2, ←nat.sub_sub, nat.sub_self, nat.zero_sub, nat.abs_zero, nat.mul_comm (200 : ℕ) 5, 
        mul_comm, nat.mul_div_cancel_left 10 (by norm_num : 10 > 0)],
    exact nat.mul_div_right 50 10 (by norm_num : 10 > 0) },
  have h_total_fee : total_fee = initial_fee + additional_time_fee := rfl,
  rw [h_total_fee, hf1, h_additional_time_fee],
  exact rfl,
end

end parking_fee_l96_96203


namespace min_moves_to_alternate_l96_96259

theorem min_moves_to_alternate (n : ℕ) (h : n = 7) : 
  ∃ m : ℕ, m = 4 ∧ (∀ coins : list bool, coins.length = n → 
  coins.all (λ x, x = tt) → 
  (∃ moves : list (ℕ × ℕ), moves.length = m ∧ 
  all_adjacent_flips coins moves (alternate coins))) := sorry

def all_adjacent_flips (coins : list bool) (moves : list (ℕ × ℕ)) (target : list bool) : Prop := 
  moves.foldl 
  (λ coins mov, coins.modify_nth mov.fst not) 
  coins = target

def alternate (coins : list bool) : list bool :=
  coins.enumerate.map (λ ⟨i, _⟩, if i % 2 = 0 then tt else ff)

end min_moves_to_alternate_l96_96259


namespace total_cost_of_topsoil_l96_96485

def cost_per_cubic_foot : ℝ := 8
def cubic_yards_to_cubic_feet : ℝ := 27
def volume_in_yards : ℝ := 7

theorem total_cost_of_topsoil :
  (cubic_yards_to_cubic_feet * volume_in_yards) * cost_per_cubic_foot = 1512 :=
by
  sorry

end total_cost_of_topsoil_l96_96485


namespace rook_even_moves_l96_96641

structure Graph :=
  (V : Type)
  (adj : V → V → Prop)
  (even_degree : ∀ v : V, ∃ n : ℕ, 2 * n = (Finset.filter (λ w, adj v w) (Finset.univ V)).card)

structure Path (G : Graph) :=
  (vertices : list G.V)
  (adjacent : ∀ (i : ℕ), i < (vertices.length - 1) → G.adj (vertices.nth_le i sorry) (vertices.nth_le (i + 1) sorry))
  (start_finish : vertices.head = vertices.last sorry)

def even_moves_on_closed_path (G : Graph) (P : Path G) : Prop :=
  (P.vertices.length - 1) % 2 = 0

theorem rook_even_moves
  (G : Graph)
  (P : Path G)
  : even_moves_on_closed_path G P := 
  sorry

end rook_even_moves_l96_96641


namespace inequality_example_l96_96882

variable (a b : ℝ)

theorem inequality_example (h1 : a > 1/2) (h2 : b > 1/2) : a + 2 * b - 5 * a * b < 1/4 :=
by
  sorry

end inequality_example_l96_96882


namespace right_triangle_side_81_exists_arithmetic_progression_l96_96213

theorem right_triangle_side_81_exists_arithmetic_progression :
  ∃ (a d : ℕ), a > 0 ∧ d > 0 ∧ (a - d)^2 + a^2 = (a + d)^2 ∧ (3*d = 81 ∨ 4*d = 81 ∨ 5*d = 81) :=
sorry

end right_triangle_side_81_exists_arithmetic_progression_l96_96213


namespace simplify_tangent_expression_l96_96914

theorem simplify_tangent_expression :
  (1 + Real.tan (Real.pi / 18)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  sorry

end simplify_tangent_expression_l96_96914


namespace div_by_all_primes_under_1966_l96_96486

theorem div_by_all_primes_under_1966 (n : ℕ) : 
  ∀ p : ℕ, prime p ∧ p < 1966 → p ∣ n * ∏ i in finset.range 1966, (i+1) * n + 1 := 
by 
  sorry

end div_by_all_primes_under_1966_l96_96486


namespace simplify_expression_l96_96493

theorem simplify_expression (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y^2) - 5 * (2 + 3 * y) = -4 * y^2 - 17 * y - 8 :=
by
  sorry

end simplify_expression_l96_96493


namespace depth_of_grass_sheet_l96_96927

-- Given conditions
def playground_area : ℝ := 5900
def grass_cost_per_cubic_meter : ℝ := 2.80
def total_cost : ℝ := 165.2

-- Variable to solve for
variable (d : ℝ)

-- Theorem statement
theorem depth_of_grass_sheet
  (h : total_cost = (playground_area * d) * grass_cost_per_cubic_meter) :
  d = 0.01 :=
by
  sorry

end depth_of_grass_sheet_l96_96927


namespace find_real_k_l96_96016

theorem find_real_k (k : ℝ) : 
  (‖k • (⟨3, -2⟩ : ℝ × ℝ) - ⟨6, -1⟩‖ = √34) ↔ (k = 3 ∨ k = 1 / 13) :=
by sorry

end find_real_k_l96_96016


namespace not_divisible_l96_96239

theorem not_divisible (x y : ℕ) (hx : x % 61 ≠ 0) (hy : y % 61 ≠ 0) (h : (7 * x + 34 * y) % 61 = 0) : (5 * x + 16 * y) % 61 ≠ 0 := 
sorry

end not_divisible_l96_96239


namespace sum_modulo_remainder_l96_96023

theorem sum_modulo_remainder :
  ((82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17) = 12 :=
by
  sorry

end sum_modulo_remainder_l96_96023


namespace kelly_initial_games_l96_96834

theorem kelly_initial_games :
  ∃ g : ℕ, (g - 15 = 35) ↔ (g = 50) :=
begin
  sorry,
end

end kelly_initial_games_l96_96834


namespace c_profit_is_21000_l96_96575

/-
Given:
  - The total profit for one year.
  - The profit sharing ratio among a, b, and c.

To Prove:
  - The profit of c is $21000.
-/

-- Define the total profit and the profit-sharing ratio
def total_profit : ℝ := 56700
def ratio_a : ℝ := 8
def ratio_b : ℝ := 9
def ratio_c : ℝ := 10

-- Define the sum of the ratio parts
def total_parts := ratio_a + ratio_b + ratio_c

-- Define c's share
def c_share := (ratio_c / total_parts) * total_profit

-- Theorem: c's profit
theorem c_profit_is_21000 : c_share = 21000 := by
  sorry

end c_profit_is_21000_l96_96575


namespace johns_earnings_without_bonus_l96_96452
-- Import the Mathlib library to access all necessary functions and definitions

-- Define the conditions of the problem
def hours_without_bonus : ℕ := 8
def bonus_amount : ℕ := 20
def extra_hours_for_bonus : ℕ := 2
def hours_with_bonus : ℕ := hours_without_bonus + extra_hours_for_bonus
def hourly_wage_with_bonus : ℕ := 10

-- Define the total earnings with the performance bonus
def total_earnings_with_bonus : ℕ := hours_with_bonus * hourly_wage_with_bonus

-- Statement to prove the earnings without the bonus
theorem johns_earnings_without_bonus :
  total_earnings_with_bonus - bonus_amount = 80 :=
by
  -- Placeholder for the proof
  sorry

end johns_earnings_without_bonus_l96_96452


namespace fewest_orders_to_minimize_total_cost_l96_96630

def original_price_per_item : ℝ := 48
def discount_rate : ℝ := 0.6
def discounted_price_per_item := original_price_per_item * discount_rate
def extra_discount_threshold := 300
def extra_discount := 100
def total_items := 42

theorem fewest_orders_to_minimize_total_cost : 
  let price_with_discount (n : ℝ) := n * discounted_price_per_item
  let full_price (n : ℝ) := if price_with_discount n > extra_discount_threshold then price_with_discount n - extra_discount else price_with_discount n
  let items_per_order := ⌊extra_discount_threshold / discounted_price_per_item⌋.to_nat + 1
  let orders := (total_items / items_per_order).ceil.to_nat
  orders = 4 := 
by
  sorry

end fewest_orders_to_minimize_total_cost_l96_96630


namespace find_x_in_subsets_l96_96756

theorem find_x_in_subsets {x : ℝ} :
  let A := {0, 1, 2} in
  let B := {1, 1/x} in
  B ⊆ A → x = 1/2 :=
by
  assume A B hsubset
  sorry

end find_x_in_subsets_l96_96756


namespace triangle_RQR_area_is_17_5_l96_96811

-- Definitions based on the conditions
variables {P Q R S T U L M G H : Point} (hexagon : Hexagon P Q R S T U)
          (square_PQLM : Square P Q L M) (square_TUHG : Square T U H G)
          (isosceles_triangle_LMR : IsoscelesTriangle L M R)
          (length_PQ_UQ : PQ = 5) (length_TU_QR : TU = QR = 7)

noncomputable def area_triangle_RQR (RQR: Triangle R Q R) : ℝ :=
  1 / 2 * QR * 5

-- The theorem to be proved
theorem triangle_RQR_area_is_17_5 :
  area_triangle_RQR (triangle R Q R) = 17.5 :=
sorry

end triangle_RQR_area_is_17_5_l96_96811


namespace prob_both_A_B_at_A_lean_prob_A_B_different_positions_lean_prob_exactly_one_at_A_lean_l96_96112

variable (volunteers : Fin 5)
variable (positions : Fin 4 := [A, B, C, D])

-- Condition: Each position is served by at least one volunteer.
axiom all_positions_served (pos : Fin 4) : ∃ v : Fin 5, serves_at v pos

-- Define the probability calculations
def prob_both_A_B_at_A : ℝ := 1 / 40
def prob_A_B_different_positions : ℝ := 9 / 10
def prob_exactly_one_at_A : ℝ := 3 / 4

-- Proof statements
theorem prob_both_A_B_at_A_lean : get_prob (both_at A B position.A) = prob_both_A_B_at_A := sorry

theorem prob_A_B_different_positions_lean : get_prob (different_positions A B) = prob_A_B_different_positions := sorry

theorem prob_exactly_one_at_A_lean : get_prob (exactly_one_at position.A) = prob_exactly_one_at_A := sorry

end prob_both_A_B_at_A_lean_prob_A_B_different_positions_lean_prob_exactly_one_at_A_lean_l96_96112


namespace ball_attendance_l96_96837

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l96_96837


namespace minimum_time_for_information_sharing_l96_96215

theorem minimum_time_for_information_sharing : 
  ∀ (people : List ℕ), people.length = 8 ∧ (∀ p, p ∈ people → ∃ info, true) → 
  ∃ t : ℕ, (t = 9) :=
by
  intro people
  intro people_len
  intro info_per_person
  use 9
  sorry -- The proof can be developed here

end minimum_time_for_information_sharing_l96_96215


namespace M_inter_N_eq_l96_96081

-- Definitions
def M : Set ℤ := { x | ∃ a : ℤ, x = a^2 + 1 }
def N : Set ℤ := { y | y ∈ Finset.range 7 ∧ 1 ≤ y }

-- Proposition to prove
theorem M_inter_N_eq : M ∩ N = { 1, 2, 5 } := by
  sorry

end M_inter_N_eq_l96_96081


namespace totalNumberOfPupils_l96_96626

-- Definitions of the conditions
def numberOfGirls : Nat := 232
def numberOfBoys : Nat := 253

-- Statement of the problem
theorem totalNumberOfPupils : numberOfGirls + numberOfBoys = 485 := by
  sorry

end totalNumberOfPupils_l96_96626


namespace characterization_of_magic_numbers_l96_96781

def is_magic_number (N : Nat) : Prop :=
  ∀ M : Nat, (10^(Nat.log10 N + 1) * M + N) % N = 0

theorem characterization_of_magic_numbers (N : Nat) :
  is_magic_number N ↔
  ∃ k : Nat, N = 2 * 10^k ∨ N = 10^k ∨ N = 5 * 10^k ∨ N = 25 * 10^k ∨ N = 125 * 10^k :=
sorry

end characterization_of_magic_numbers_l96_96781


namespace num_4_digit_positive_integers_l96_96085

theorem num_4_digit_positive_integers : 
    let digits := {0, 2, 3, 6, 8}
    let is_valid (a b c d : ℕ) :=
      a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
      a ≠ 0 ∧ d = 0 ∧ (a + b + c + d) % 3 = 0
    (finset.univ.filter (λ (abcd : ℕ × ℕ × ℕ × ℕ), is_valid abcd.1 abcd.2.1 abcd.2.2.1 abcd.2.2.2)).card = 36 := sorry

end num_4_digit_positive_integers_l96_96085


namespace perpendicular_BC_l96_96287

open EuclideanGeometry

/-- Let ABC be an isosceles triangle, I be the incenter of ABC. Circles Γ₁ centered at A with radius AB, 
  Γ₂ centered at I with radius IB, Γ₃ passes through points B and I and intersects Γ₁ at P and Γ₂ at Q different from B.
  Let IP and BQ intersect at R. Prove that BR is perpendicular to CR -/
theorem perpendicular_BC 
  (A B C I P Q R : Point)
  (h_isosceles : AB = AC)
  (h_incenter : is_incenter I A B C)
  (Γ₁ : Circle)
  (Γ₂ : Circle)
  (Γ₃ : Circle)
  (h_circ_1 : Γ₁.center = A ∧ Γ₁.radius = AB)
  (h_circ_2 : Γ₂.center = I ∧ Γ₂.radius = IB)
  (h_circ_3 : passes_through_points Γ₃ B I)
  (h_intersect_1 : intersects_at_two_points Γ₃ Γ₁ B P)
  (h_intersect_2 : intersects_at_two_points Γ₃ Γ₂ B Q)
  (h_distinct : P ≠ B ∧ Q ≠ B)
  (h_intersect_lines : ∃ R, line_through I P ∧ line_through B Q) :
  perpendicular (line_through B R) (line_through C R) := 
sorry

end perpendicular_BC_l96_96287


namespace solve_for_y_l96_96501

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l96_96501


namespace trig_problems_l96_96427

variable {A B C : ℝ}
variable {a b c : ℝ}

-- The main theorem statement to prove the magnitude of angle B and find b under given conditions.
theorem trig_problems
  (h₁ : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h₂ : a = Real.sqrt 3)
  (h₃ : c = Real.sqrt 3) :
  Real.cos B = 1 / 2 ∧ b = Real.sqrt 3 := by
sorry

end trig_problems_l96_96427


namespace number_of_books_in_library_l96_96039

theorem number_of_books_in_library 
  (a : ℕ) 
  (R L : ℕ) 
  (h1 : a = 12 * R + 7) 
  (h2 : a = 25 * L - 5) 
  (h3 : 500 < a ∧ a < 650) : 
  a = 595 :=
begin
  sorry
end

end number_of_books_in_library_l96_96039


namespace positive_results_count_l96_96545

open Nat

theorem positive_results_count : 
  (Set.univ.filter (λ (AB : ℕ × ℕ), 
      let A := AB.1;
      let B := AB.2 in
      1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ A ≠ B ∧ A - 10 * B + 45 > 0)).toFinset.card = 36 :=
by
  sorry

end positive_results_count_l96_96545


namespace P5_to_P_infty_length_l96_96902

-- Define the type for points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Function to represent centroid of four points
def centroid (p1 p2 p3 p4 : Point) : Point :=
  { x := (p1.x + p2.x + p3.x + p4.x) / 4,
    y := (p1.y + p2.y + p3.y + p4.y) / 4,
    z := (p1.z + p2.z + p3.z + p4.z) / 4 }

-- Regular tetrahedron with side length 1
constant P1 P2 P3 P4 : Point
axiom regular_tetrahedron : (dist P1 P2 = 1) ∧ (dist P1 P3 = 1) ∧ (dist P1 P4 = 1) ∧ (dist P2 P3 = 1) ∧ (dist P2 P4 = 1) ∧ (dist P3 P4 = 1)

-- Define sequence P_n where P_i is the centroid of previous four points for i > 4
noncomputable def P : ℕ → Point
| 1 := P1
| 2 := P2
| 3 := P3
| 4 := P4
| (n + 5) := centroid (P n) (P (n + 1)) (P (n + 2)) (P (n + 3))

-- Define the limit point P_∞ as n → ∞
noncomputable def P_infty : Point := sorry -- The actual definition would involve proving the convergence, skipped here

-- Prove that the distance between P_5 and P_∞ is 1 / (2 * sqrt 2)
theorem P5_to_P_infty_length : dist (P 5) P_infty = 1 / (2 * sqrt 2) :=
sorry

end P5_to_P_infty_length_l96_96902


namespace probability_correct_predictions_monday_l96_96592

def number_of_classes_monday : ℕ := 5
def number_of_classes_tuesday : ℕ := 6
def total_classes : ℕ := number_of_classes_monday + number_of_classes_tuesday

def total_correct_predictions : ℕ := 7
def correct_predictions_monday : ℕ := 3
def correct_predictions_tuesday : ℕ := total_correct_predictions - correct_predictions_monday

noncomputable def binomial_coefficient (n k : ℕ) : ℝ := (nat.choose n k : ℝ)

noncomputable def probability_exact_correct_monday (n k : ℕ) : ℝ :=
  binomial_coefficient n k * (real.exp 11 (ln 2⁻¹ * total_classes))

theorem probability_correct_predictions_monday :
  probability_exact_correct_monday number_of_classes_monday correct_predictions_monday *
  probability_exact_correct_monday number_of_classes_tuesday correct_predictions_tuesday /
  probability_exact_correct_monday total_classes total_correct_predictions = 5 / 11 :=
sorry

end probability_correct_predictions_monday_l96_96592


namespace min_m_plus_n_l96_96506

open Nat

theorem min_m_plus_n (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 45 * m = n^3) (h_mult_of_five : 5 ∣ n) :
  m + n = 90 :=
sorry

end min_m_plus_n_l96_96506


namespace person_A_days_l96_96546

-- Let x be the number of days it takes for person A to complete the work alone
variable (x : ℝ)

-- The amount of work person B can complete in one day
def work_B_per_day : ℝ := 1 / 45

-- The combined work of A and B in one day
def combined_work_per_day (x : ℝ) : ℝ := (1 / x) + work_B_per_day

-- The condition that 0.38888888888888884 part of the work is completed in 7 days
axiom combined_work_in_7_days (x : ℝ) : 7 * combined_work_per_day x = 0.38888888888888884

-- Prove that person A alone can complete the work in 90 days
theorem person_A_days (x : ℝ) : combined_work_in_7_days x → x = 90 :=
by
  intro h,
  sorry

end person_A_days_l96_96546


namespace general_term_sum_of_sequence_l96_96053

-- Definitions based on conditions
variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}

-- Conditions
axiom pos_terms (n : ℕ) : a_n n > 0
axiom sum_terms (n : ℕ) : S_n n = ∑ k in finset.range (n+1), a_n k
axiom first_term : a_n 1 = 0.5
axiom arith_seq (n : ℕ) : 2 * (a_n n) = S_n n + 0.5
axiom square_rel (n : ℕ) : a_n n ^ 2 = 2 ^ (4 - 2 * n)
axiom seq_c_n (n : ℕ) : ∀ b_n : ℕ → ℝ, b_n n = 4 - 2 * n → c_n : ℕ → ℝ, c_n n = b_n n / a_n n

-- General term formula a_n
theorem general_term (n : ℕ) : a_n (n+1) = 2 ^ (n - 1) :=
sorry

-- Sum of the first n terms for the sequence {c_n}
theorem sum_of_sequence (n : ℕ) : ∑ k in finset.range n, c_n k = (8 * n) / (2 ^ n) :=
sorry

end general_term_sum_of_sequence_l96_96053


namespace ellipse_eccentricity_l96_96380

-- Define the ellipse parameters: a > b > 0
variables (a b c : ℝ) (h_ab : a > b) (h_b0 : b > 0)

-- Define the coordinates of the points
def A : ℝ × ℝ := (-a, 0)
def F1 : ℝ × ℝ := (-c, 0)
def F2 : ℝ × ℝ := (c, 0)
def D : ℝ × ℝ := (0, b)

-- Define the vectors DFi
def vec_DF1 : ℝ × ℝ := ((-c) - 0, 0 - b)
def vec_DA : ℝ × ℝ := ((-a) - 0, 0 - b)
def vec_DF2 : ℝ × ℝ := (c - 0, 0 - b)

-- Define the condition 3 vec_DF1 = vec_DA + 2 vec_DF2
def condition : Prop := 3 * vec_DF1 = (vec_DA + 2 * vec_DF2)

-- Define the eccentricity e
def e : ℝ := c / a

-- Prove that the eccentricity e equals 1/5 given the condition
theorem ellipse_eccentricity (h : 3 * vec_DF1 = vec_DA + 2 * vec_DF2) : e a c = 1 / 5 :=
by
  sorry

end ellipse_eccentricity_l96_96380


namespace problem_1_problem_2_l96_96747

noncomputable theory
open Real

def f (x a : ℝ) : ℝ := exp x - (1/2) * a * x^2 + (a - exp 1) * x

theorem problem_1 (x : ℝ) (hx : 0 ≤ x) : 
  ∀ f : ℝ → ℝ, (f = λ x, exp x - exp 1 * x) → f 1 = 0 :=
begin
  sorry
end

theorem problem_2 (x : ℝ) (a : ℝ) (hx : 0 ≤ x) (h1 : 1 < a) (h2 : a < exp 1) : 
  ∃ n : ℕ, (n = 2 ∧ a ≤ exp 1 - 1) ∨ (n = 3 ∧ exp 1 - 1 < a ∧ a < exp 1) :=
begin
  sorry
end

end problem_1_problem_2_l96_96747


namespace distance_between_parallel_lines_l96_96325

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 4 = 0
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y + 1 = 0

theorem distance_between_parallel_lines :
  let A := (3 : ℝ)
  let B := (4 : ℝ)
  let C1 := (-4 : ℝ)
  let C2 := (-1 : ℝ)
  real.abs (C1 - C2) / real.sqrt (A * A + B * B) = 1 :=
by
  sorry

end distance_between_parallel_lines_l96_96325


namespace ball_attendance_l96_96861

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l96_96861


namespace ball_attendance_l96_96840

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l96_96840


namespace part1_part2_l96_96736

noncomputable def a_n (n : ℕ) : ℕ :=
  2^(n - 1)

noncomputable def b_n (n : ℕ) : ℕ :=
  2 * n

noncomputable def S_n (n : ℕ) : ℕ :=
  n^2 + n

theorem part1 (n : ℕ) : 
  S_n n = n^2 + n := 
sorry

noncomputable def C_n (n : ℕ) : ℚ :=
  (n^2 + n) / 2^(n - 1)

theorem part2 (n : ℕ) (k : ℕ) (k_gt_0 : 0 < k) : 
  (∀ n, C_n n ≤ C_n k) ↔ (k = 2 ∨ k = 3) :=
sorry

end part1_part2_l96_96736


namespace bubbles_from_half_ounce_of_mixture_l96_96778

-- Given conditions
def dawn_bubbles_per_ounce : ℕ := 200000
def dr_bronner_bubbles_per_ounce : ℕ := 2 * dawn_bubbles_per_ounce
def mixture_dawn_fraction : ℚ := 1 / 2
def mixture_dr_bronner_fraction : ℚ := 1 / 2

-- Desired result
def bubbles_per_half_ounce_mixture : ℕ :=
  let dawn_half_ounce_bubbles := (mixture_dawn_fraction * dawn_bubbles_per_ounce).toNat
  let dr_bronner_half_ounce_bubbles := (mixture_dr_bronner_fraction * dr_bronner_bubbles_per_ounce).toNat
  (dawn_half_ounce_bubbles + dr_bronner_half_ounce_bubbles) / 2

-- Goal
theorem bubbles_from_half_ounce_of_mixture : bubbles_per_half_ounce_mixture = 150000 := by
  sorry

end bubbles_from_half_ounce_of_mixture_l96_96778


namespace ball_attendance_l96_96838

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l96_96838


namespace sum_of_three_integers_l96_96208

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 5^3) : a + b + c = 31 := by
  sorry

end sum_of_three_integers_l96_96208


namespace least_multiple_of_25_gt_450_correct_l96_96556

def least_multiple_of_25_gt_450 : ℕ :=
  475

theorem least_multiple_of_25_gt_450_correct (n : ℕ) (h1 : 25 ∣ n) (h2 : n > 450) : n ≥ least_multiple_of_25_gt_450 :=
by
  sorry

end least_multiple_of_25_gt_450_correct_l96_96556


namespace even_numbers_count_l96_96402

theorem even_numbers_count :
  (∃ S : Finset ℕ, (∀ n ∈ S, 300 < n ∧ n < 600 ∧ n % 2 = 0) ∧ S.card = 149) :=
by
  sorry

end even_numbers_count_l96_96402


namespace area_rectangle_EFGH_l96_96121

def point : Type := ℝ × ℝ

def E : point := (-2, -7)
def F : point := (998, 93)
def H (y : ℤ) : point := (0, y)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def area_of_rectangle (p1 p2 p3 : point) : ℝ :=
  let l1 := distance p1 p2
  let l2 := distance p1 p3
  l1 * l2

theorem area_rectangle_EFGH (y : ℤ) (hy : y = -27) :
  area_of_rectangle E F (H y) = 202020 :=
by
  sorry

end area_rectangle_EFGH_l96_96121


namespace solve_for_a_l96_96750

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x^2 - 1 else x - 2

theorem solve_for_a (a : ℝ) : f (f a) = 3 → a = Real.sqrt 3 :=
by
  sorry

end solve_for_a_l96_96750


namespace find_n_coins_l96_96601

theorem find_n_coins (n : ℕ) : 
  (n * (n - 1) / 2 ∧ 2^(n + 1) = 5 / 32) → n = 6 :=
by sorry

end find_n_coins_l96_96601


namespace largest_3_digit_sum_l96_96407

-- Defining the condition that ensures X, Y, Z are different digits ranging from 0 to 9
def valid_digits (X Y Z : ℕ) : Prop :=
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- Problem statement: Proving the largest possible 3-digit sum is 994
theorem largest_3_digit_sum : ∃ (X Y Z : ℕ), valid_digits X Y Z ∧ 111 * X + 11 * Y + Z = 994 :=
by
  sorry

end largest_3_digit_sum_l96_96407


namespace candied_apple_price_l96_96298

theorem candied_apple_price
  (x : ℝ) -- price of each candied apple in dollars
  (h1 : 15 * x + 12 * 1.5 = 48) -- total earnings equation
  : x = 2 := 
sorry

end candied_apple_price_l96_96298


namespace magnitude_of_vector_sum_l96_96465

noncomputable def vec := (ℝ × ℝ)

def dot_product (a b : vec) : ℝ := a.1 * b.1 + a.2 * b.2
def magnitude (a : vec) : ℝ := Real.sqrt (a.1^2 + a.2^2)

theorem magnitude_of_vector_sum :
  ∀ (x y : ℝ),
  let a : vec := (x, 2)
  let b : vec := (1, y)
  let c : vec := (2, -6)
  (dot_product a c = 0) →
  (∃ k : ℝ, b = (k * c.1, k * c.2)) →
  magnitude (a.1 + b.1, a.2 + b.2) = 5 * Real.sqrt 2 :=
by
  intros x y a b c h1 h2
  sorry

end magnitude_of_vector_sum_l96_96465


namespace sequence_sum_l96_96652

def sequence : ℕ → ℤ 
| 1 => -2
| n => if n % 2 = 0 then sequence (n-2) + 5 else sequence (n-2) - 11

theorem sequence_sum : (∑ i in Finset.range 19, sequence (i + 1)) = 9 := by
  sorry

end sequence_sum_l96_96652


namespace sin_cos_sum_third_quadrant_l96_96373

noncomputable def thirdQuadrant (α : ℝ) : Prop :=
  π < α ∧ α < 3 * π / 2

theorem sin_cos_sum_third_quadrant (α : ℝ) (h1 : thirdQuadrant α) (h2 : Real.tan α = 3 / 4) :
  Real.sin α + Real.cos α = -7 / 5 :=
by
  sorry

end sin_cos_sum_third_quadrant_l96_96373


namespace cubes_not_touching_foil_l96_96538

-- Define the variables for length, width, height, and total cubes
variables (l w h : ℕ)

-- Conditions extracted from the problem
def width_is_twice_length : Prop := w = 2 * l
def width_is_twice_height : Prop := w = 2 * h
def foil_covered_prism_width : Prop := w + 2 = 10

-- The proof statement
theorem cubes_not_touching_foil (l w h : ℕ) 
  (h1 : width_is_twice_length l w) 
  (h2 : width_is_twice_height w h) 
  (h3 : foil_covered_prism_width w) : 
  l * w * h = 128 := 
by sorry

end cubes_not_touching_foil_l96_96538


namespace range_of_x0_l96_96742

noncomputable theory

open Real

def circle_C (x y : ℝ) := x^2 + y^2 = 1
def line_L (x y : ℝ) := x - y - 2 = 0

theorem range_of_x0 :
  (∀ P Q : ℝ × ℝ, (exists Q : ℝ × ℝ, circle_C Q.1 Q.2 ∧ ∃ θ, θ = π / 6 ∧ line_L P.1 P.2 ∧ ∃ O : ℝ × ℝ, O = (0, 0)) → 0 ≤ P.1 ∧ P.1 ≤ 2) := sorry

end range_of_x0_l96_96742


namespace petya_wins_iff_even_l96_96954

theorem petya_wins_iff_even (n : ℕ) (h : n ≥ 5) : (∃ moves : ℕ → (ℕ × ℕ),  ∀ t : ℕ, t ≤ n → valid_move n (moves t) → winning_strategy moves t) ↔ even n := 
sorry

end petya_wins_iff_even_l96_96954


namespace original_profit_percentage_l96_96618

noncomputable def originalCost : ℝ := 80
noncomputable def P := 30
noncomputable def profitPercentage : ℝ := ((100 - originalCost) / originalCost) * 100

theorem original_profit_percentage:
  ∀ (S C : ℝ),
  C = originalCost →
  ( ∀ (newCost : ℝ),
    newCost = 0.8 * C →
    ∀ (newSell : ℝ),
    newSell = S - 16.8 →
    newSell = 1.3 * newCost → P = 30 ) →
  profitPercentage = 25 := sorry

end original_profit_percentage_l96_96618


namespace ratio_BD_CD_l96_96181

noncomputable def triangle_sides (AB AC BC : ℝ) : Prop :=
AB = 13 ∧ AC = 15 ∧ BC = 14

noncomputable def unique_point_D (BD CD : ℝ) : Prop :=
∃ D : Point, D ∈ lineSegment BC ∧
(let Euler_line_parallel (T₁ T₂ : Triangle) : Prop := -- Define parallel property here
sorry in
Euler_line_parallel (Triangle.mk A B D) (Triangle.mk A C D))

theorem ratio_BD_CD :
triangle_sides AB AC BC →
unique_point_D BD CD →
BD / CD = (93 + 56 * real.sqrt(3)) / 33 :=
by intros h_triangle_sides h_unique_point_D; sorry

end ratio_BD_CD_l96_96181


namespace integral_cosine_eighth_l96_96980

theorem integral_cosine_eighth :
  ∫ x in 0..π, 2^4 * cos(x / 2)^8 = 35 * π / 8 :=
by sorry

end integral_cosine_eighth_l96_96980


namespace new_average_after_changes_l96_96793

theorem new_average_after_changes :
  ∀ (n1 n2 n3 : ℕ) (avg1 avg2 avg3 : ℕ) 
    (new_avg : ℚ),
    n1 = 12 ∧ 
    n2 = 6 ∧ 
    n3 = 2 ∧ 
    avg1 = 36 ∧ 
    avg2 = 42 ∧ 
    avg3 = 48 ∧ 
    new_avg = (864 + 302.4 + 86.4) / 20 → 
  new_avg = 62.64 :=
by {
  intros,
  sorry
}

end new_average_after_changes_l96_96793


namespace reduced_price_tickets_first_week_l96_96316

theorem reduced_price_tickets_first_week (total_tickets sold_at_full_price : ℕ) 
  (condition1 : total_tickets = 25200) 
  (condition2 : sold_at_full_price = 16500)
  (condition3 : ∃ R, total_tickets = R + 5 * R) : 
  ∃ R : ℕ, R = 3300 := 
by sorry

end reduced_price_tickets_first_week_l96_96316


namespace ball_attendance_l96_96862

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l96_96862


namespace parabola_vertex_coordinates_l96_96514

theorem parabola_vertex_coordinates :
  ∃ h k, ∀ x, 5 * (x - h)^2 + k = 5 * (x - 2)^2 + 6 ∧ h = 2 ∧ k = 6 := 
by
  use [2, 6]
  intro x
  split
  · simp [sub_eq_add_neg, add_sq, mul_self_add, ring]
  · simp

end parabola_vertex_coordinates_l96_96514


namespace probability_correct_predictions_monday_l96_96589

def number_of_classes_monday : ℕ := 5
def number_of_classes_tuesday : ℕ := 6
def total_classes : ℕ := number_of_classes_monday + number_of_classes_tuesday

def total_correct_predictions : ℕ := 7
def correct_predictions_monday : ℕ := 3
def correct_predictions_tuesday : ℕ := total_correct_predictions - correct_predictions_monday

noncomputable def binomial_coefficient (n k : ℕ) : ℝ := (nat.choose n k : ℝ)

noncomputable def probability_exact_correct_monday (n k : ℕ) : ℝ :=
  binomial_coefficient n k * (real.exp 11 (ln 2⁻¹ * total_classes))

theorem probability_correct_predictions_monday :
  probability_exact_correct_monday number_of_classes_monday correct_predictions_monday *
  probability_exact_correct_monday number_of_classes_tuesday correct_predictions_tuesday /
  probability_exact_correct_monday total_classes total_correct_predictions = 5 / 11 :=
sorry

end probability_correct_predictions_monday_l96_96589


namespace simplify_tangent_sum_l96_96906

theorem simplify_tangent_sum :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 := by
  have h1 : Real.tan (45 * Real.pi / 180) = Real.tan ((10 + 35) * Real.pi / 180) := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h3 : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y) := by sorry
  sorry

end simplify_tangent_sum_l96_96906


namespace zero_is_natural_number_l96_96562

theorem zero_is_natural_number : 0 ∈ ℕ :=
sorry

end zero_is_natural_number_l96_96562


namespace log_base2_probability_l96_96227

theorem log_base2_probability (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : ∃ k : ℕ, n = 2^k) : 
  ∃ p : ℚ, p = 1/300 :=
  sorry

end log_base2_probability_l96_96227


namespace find_missing_dimension_l96_96617

def carton_volume (l w h : ℕ) : ℕ := l * w * h

def soapbox_base_area (l w : ℕ) : ℕ := l * w

def total_base_area (n l w : ℕ) : ℕ := n * soapbox_base_area l w

def missing_dimension (carton_volume total_base_area : ℕ) : ℕ := carton_volume / total_base_area

theorem find_missing_dimension 
  (carton_l carton_w carton_h : ℕ) 
  (soapbox_l soapbox_w : ℕ) 
  (n : ℕ) 
  (h_carton_l : carton_l = 25)
  (h_carton_w : carton_w = 48)
  (h_carton_h : carton_h = 60)
  (h_soapbox_l : soapbox_l = 8)
  (h_soapbox_w : soapbox_w = 6)
  (h_n : n = 300) :
  missing_dimension (carton_volume carton_l carton_w carton_h) (total_base_area n soapbox_l soapbox_w) = 5 := 
by 
  sorry

end find_missing_dimension_l96_96617


namespace symmetrical_line_range_l96_96714

theorem symmetrical_line_range {k : ℝ} :
  (∀ x y : ℝ, (y = k * x - 1) ∧ (x + y - 1 = 0) → y ≠ -x + 1) → k > 1 ↔ k > 1 :=
by
  sorry

end symmetrical_line_range_l96_96714


namespace Misha_probability_l96_96597

open Probability

-- Definitions
def classesMonday := 5
def classesTuesday := 6
def totalClasses := 11
def totalCorrect := 7
def mondayCorrect := 3

-- Calculating binomial probabilities
noncomputable def P_A1 := (binomial classesMonday mondayCorrect) * (1 / 2) ^ classesMonday
noncomputable def P_A2 := (binomial classesTuesday (totalCorrect - mondayCorrect)) * (1 / 2) ^ classesTuesday
noncomputable def P_B := (binomial totalClasses totalCorrect) * (1 / 2) ^ totalClasses

-- Theorem stating the required probability
theorem Misha_probability :
  P_A1 * P_A2 / P_B = 5 / 11 :=
by
  sorry

end Misha_probability_l96_96597


namespace vivi_total_yards_l96_96964

theorem vivi_total_yards (spent_checkered spent_plain cost_per_yard : ℝ)
  (h1 : spent_checkered = 75)
  (h2 : spent_plain = 45)
  (h3 : cost_per_yard = 7.50) :
  (spent_checkered / cost_per_yard + spent_plain / cost_per_yard) = 16 :=
by 
  sorry

end vivi_total_yards_l96_96964


namespace subset_M_N_l96_96151

-- Definition of the sets
def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | 1/x < 3 }

theorem subset_M_N : M ⊆ N :=
by
  -- sorry to skip the proof
  sorry

end subset_M_N_l96_96151


namespace sum_a2_to_a5_eq_zero_l96_96413

theorem sum_a2_to_a5_eq_zero 
  (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : ∀ x : ℝ, x * (1 - 2 * x)^4 = a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_2 + a_3 + a_4 + a_5 = 0 :=
sorry

end sum_a2_to_a5_eq_zero_l96_96413


namespace salt_solution_percentage_l96_96246

theorem salt_solution_percentage
  (x : ℝ)
  (y : ℝ)
  (h1 : 600 + y = 1000)
  (h2 : 600 * x + y * 0.12 = 1000 * 0.084) :
  x = 0.06 :=
by
  -- The proof goes here.
  sorry

end salt_solution_percentage_l96_96246


namespace parabola_problem_l96_96377

theorem parabola_problem 
  (p : ℝ) (hp : p > 0) 
  (A B : ℝ × ℝ) (D M F : ℝ × ℝ) 
  (hC : ∀ x y, (y^2 = 2 * p * x → (x, y) ∈ {A, B}))
  (hlM : M = (-1, 0))
  (hDirectrix : ∀ x, (x = -p / 2 → x = -1))
  (hLineThroughM : ∀ m y, (y = m * x - 1 → ∀ x y, (y^2 = 2 * p * x → (x, y) ∈ {A, B})))
  (hPerpendicular : B.2 = B.2 → ∀ y, (y = B.2)) :
  (D.1 - 1) / (-1) = (A.2 - B.2) →
  |B.1 - F.1| = 3 * |A.1 - F.1| ∧ |A.2 - B.2| + |B.2 - F.2| > 2 * |M.1 - F.1| :=
sorry

end parabola_problem_l96_96377


namespace ways_to_assign_numbers_l96_96934

theorem ways_to_assign_numbers :
  ∃ s : Finset (Fin 8)ᵉ8, (s.toList.perm (range 8)) ∧
  (∀ C ∈ ({({0, 1, 2, 3} : Finset (Fin 8)), 
             ({0, 1, 4, 5} : Finset (Fin 8)), 
             ({2, 3, 6, 7} : Finset (Fin 8)), 
             ({4, 5, 6, 7} : Finset (Fin 8)), 
             ({0, 3, 5, 7} : Finset (Fin 8))} : Finset (Finset (Fin 8)))), 
             (s.filter (λ x : Fin 8, x ∈ C)).val.sum = 12) → 
  (∃! f : Finset (Fin 8)ᵉ8, (∃! f', true → (s.toList.perm (range 8))), 8) :=
begin
  sorry
end

end ways_to_assign_numbers_l96_96934


namespace equilateral_triangle_coloring_l96_96643

/-- Given an equilateral triangle divided into 4 smaller triangles by its medians, and midpoints 
taken on each of the sides of these triangles, color each of these 15 specific points either 
red or blue. Prove that there must be at least three points of the same color that form the vertices 
of an equilateral triangle. --/
theorem equilateral_triangle_coloring (points : Fin 15 → Fin 2) :
  ∃ (a b c : Fin 15), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (equilateral a b c) ∧ (points a = points b ∧ points b = points c) := by
  sorry

end equilateral_triangle_coloring_l96_96643


namespace max_voters_interviewed_l96_96801

theorem max_voters_interviewed (x : ℕ) (h1 : 98 + 9 * x ≥ 95 + 9.5 * x) : 100 + 10 * x = 160 :=
by
  have : x ≤ 6 := sorry
  exact sorry

end max_voters_interviewed_l96_96801


namespace empty_set_implies_a_range_l96_96103

theorem empty_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(a * x^2 - 2 * a * x + 1 < 0)) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end empty_set_implies_a_range_l96_96103


namespace area_OPQ_triangle_l96_96146

-- Define the necessary variables and assumptions
variables (O F P Q : Type) (a b : ℝ) [T1: F ≠ O] [T2: P ≠ Q]

-- Define the lengths as given conditions
def OF_dist := dist O F = a
def PQ_length := dist P Q = b
def F_in_PQ := True  -- Assuming F is on the chord PQ

-- The final theorem to prove the area of triangle OPQ is a * sqrt(a * b)
theorem area_OPQ_triangle
  (h1 : OF_dist O F a)
  (h2 : PQ_length P Q b)
  (h3 : F_in_PQ) :
  area O P Q = a * real.sqrt (a * b) :=
sorry

end area_OPQ_triangle_l96_96146


namespace shopkeeper_loss_percent_l96_96256

theorem shopkeeper_loss_percent 
  (C : ℝ) (P : ℝ) (L : ℝ) 
  (hC : C = 100) 
  (hP : P = 10) 
  (hL : L = 50) : 
  ((C - (((C * (1 - L / 100)) * (1 + P / 100))) / C) * 100) = 45 :=
by
  sorry

end shopkeeper_loss_percent_l96_96256


namespace l1_parallel_l2_l1_perpendicular_l2_l96_96393

-- Definitions and conditions
def l1 (a : ℝ) : affine_equation := {x | (a - 1) * x.1 + 2 * x.2 + 1 = 0}
def l2 (a : ℝ) : affine_equation := {x | x.1 + a * x.2 + 1 = 0}

def are_parallel (a : ℝ) : Prop := ∃ k : ℝ, k ≠ 1 ∧ (a - 1) = k ∧ 2 = k * a
def are_perpendicular (a : ℝ) : Prop := (a - 1) + 2 * a = 0

-- Statements to prove
theorem l1_parallel_l2 (a : ℝ) : are_parallel a → a = -1 :=
by
  sorry

theorem l1_perpendicular_l2 (a : ℝ) : are_perpendicular a → a = 1 / 3 :=
by
  sorry

end l1_parallel_l2_l1_perpendicular_l2_l96_96393


namespace fourth_intersection_point_of_curve_and_circle_l96_96125

theorem fourth_intersection_point_of_curve_and_circle (h k R : ℝ)
  (h1 : (3 - h)^2 + (2 / 3 - k)^2 = R^2)
  (h2 : (-4 - h)^2 + (-1 / 2 - k)^2 = R^2)
  (h3 : (1 / 2 - h)^2 + (4 - k)^2 = R^2) :
  ∃ (x y : ℝ), xy = 2 ∧ (x, y) ≠ (3, 2 / 3) ∧ (x, y) ≠ (-4, -1 / 2) ∧ (x, y) ≠ (1 / 2, 4) ∧ 
    (x - h)^2 + (y - k)^2 = R^2 ∧ (x, y) = (2 / 3, 3) := 
sorry

end fourth_intersection_point_of_curve_and_circle_l96_96125


namespace range_of_a_l96_96069

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem range_of_a :
  (∀ a x, x ∈ Icc (-∞ : ℝ) 2 → f a x ≤ f a 2) ∧
  (∀ a x₁ x₂, x₁ ∈ Icc 1 (a + 1) ∧ x₂ ∈ Icc 1 (a + 1) → |f a x₁ - f a x₂| ≤ 4) 
  → a ∈ Icc 2 3 :=
begin
  sorry
end

end range_of_a_l96_96069


namespace square_area_double_sides_l96_96104

theorem square_area_double_sides (s : ℝ) (h : s > 0) :
  let A_original := s^2 in
  let A_resultant := (2 * s)^2 in
  A_resultant = 4 * A_original :=
by
  sorry

end square_area_double_sides_l96_96104


namespace a_eq_zero_l96_96392

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, ax + 2 ≠ 0) : a = 0 := by
  sorry

end a_eq_zero_l96_96392


namespace volume_ratio_l96_96930

-- Define the parameters and objects
variables {α R H : ℝ}

-- Define the condition that the base of the pyramid is a right triangle with an acute angle α
def base_right_triangle (α : ℝ) : Prop :=
  ∃ A B C : ℝ × ℝ, ∠ACB = 90 ∧ ∠BAC = α

-- Define the condition that this triangle is inscribed in the base of the cone
def inscribed_in_cone (α : ℝ) (R : ℝ) : Prop :=
  base_right_triangle α ∧
  ∃ O, midpoint (hypotenuse (A B C)) = O ∧ distance O A = R

-- Define the condition that the vertex of the pyramid coincides with the midpoint of one of the generatrices of the cone
def vertex_coincides_midpoint (H : ℝ) : Prop :=
  ∃ F D E, generatrix D E = DE ∧ F = midpoint (DE) ∧ height DO = H

-- Define the volumes of the cone and the pyramid
noncomputable def volume_of_cone (R H : ℝ) : ℝ := (1/3) * π * R^2 * H
noncomputable def volume_of_pyramid (R H α : ℝ) : ℝ := (1/6) * R^2 * H * sin (2 * α)

-- The main theorem that proves the ratio of the volume of the cone to the volume of the pyramid
theorem volume_ratio (α R H : ℝ) 
  (h1 : base_right_triangle α) 
  (h2 : inscribed_in_cone α R) 
  (h3 : vertex_coincides_midpoint H) :
  volume_of_cone R H / volume_of_pyramid R H α = 2 * π / sin (2 * α) :=
by
  sorry

end volume_ratio_l96_96930


namespace intersection_three_points_l96_96564

def circle_eq (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2
def parabola_eq (a : ℝ) (x y : ℝ) : Prop := y = x^2 - 3 * a

theorem intersection_three_points (a : ℝ) :
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    circle_eq a x1 y1 ∧ parabola_eq a x1 y1 ∧
    circle_eq a x2 y2 ∧ parabola_eq a x2 y2 ∧
    circle_eq a x3 y3 ∧ parabola_eq a x3 y3 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3)) ↔ 
  a = 1/3 := by
  sorry

end intersection_three_points_l96_96564


namespace solve_for_y_l96_96498

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l96_96498


namespace abs_ineq_solution_l96_96503

theorem abs_ineq_solution (x : ℝ) : 
  (|x - 3| + |x + 4| < 10) ↔ (x ∈ Set.Ioo (-11 / 2) 9 / 2) :=
by
  sorry

end abs_ineq_solution_l96_96503


namespace hoseok_add_8_l96_96096

theorem hoseok_add_8 (x : ℕ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end hoseok_add_8_l96_96096


namespace volleyball_team_starters_l96_96163

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem volleyball_team_starters :
  let players := 16
  let quadruplets := 4
  let other_players := players - quadruplets
  let starters := 6
  ∑ with_no_quad := choose other_players starters
  ∑ with_one_quad := quadruplets * choose other_players (starters - 1)
  (with_no_quad + with_one_quad) = 4092 :=
by
  sorry

end volleyball_team_starters_l96_96163


namespace no_such_seq_exists_l96_96822

def seq_exists (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, a (n + 1) = a n + m) ∧ -- Differences {a(n+1) - a(n)} take every natural value exactly once
  (∀ n : ℕ, ∃ m : ℕ, m > 2015 ∧ a (n + 2) = a n + m) -- Differences {a(n+2) - a(n)} take every natural value greater than 2015 exactly once

theorem no_such_seq_exists : ¬ ∃ (a : ℕ → ℕ), seq_exists a :=
begin
  sorry
end

end no_such_seq_exists_l96_96822


namespace sum_of_ratios_at_least_2n_l96_96712

theorem sum_of_ratios_at_least_2n (n : ℕ) (h : n ≥ 3) (a : ℕ → ℕ)
  (hcirc : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a (i - 1) + a (i + 1) = k * a i ∧ k ∈ ℕ) :
  ∑ i in (finset.range n).erase 0, (a (i - 1 % n) + a ((i + 1) % n)) / a i ≥ 2 * n :=
begin
  sorry,
end

end sum_of_ratios_at_least_2n_l96_96712


namespace turtle_reaches_watering_hole_l96_96570

theorem turtle_reaches_watering_hole:
  ∀ (x y : ℝ)
   (dist_first_cub : ℝ := 6 * y)
   (dist_turtle : ℝ := 32 * x)
   (time_first_encounter := (dist_first_cub - dist_turtle) / (y - x))
   (time_second_encounter := dist_turtle / (x + 1.5 * y))
   (time_between_encounters = 2.4),
   time_second_encounter - time_first_encounter = time_between_encounters →
   (time_turtle_reaches := time_second_encounter + 32 - 3.2) -- since total_time - the time between second 
   →
   time_turtle_reaches = 28.8 :=
begin
  intros x y,
  sorry
end

end turtle_reaches_watering_hole_l96_96570


namespace find_number_l96_96425

noncomputable def x : ℝ :=
  36 / (Real.sqrt 0.85 * (2 / 3 * Real.pi))

theorem find_number : x ≈ 18.656854 :=
by
  have h1 : Real.sqrt 0.85 ≈ 0.921954 := sorry
  have h2 : 2 / 3 * Real.pi ≈ 2.094395 := sorry
  have h3 : Real.sqrt 0.85 * (2 / 3 * Real.pi) ≈ 1.930297 := sorry
  show x ≈ 18.656854 from
    calc
      x = 36 / (Real.sqrt 0.85 * (2 / 3 * Real.pi)) : rfl
      ... ≈ 36 / 1.930297 : by { rw [h3] }
      ... ≈ 18.656854    : sorry

end find_number_l96_96425


namespace find_f_2_l96_96709

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- The statement to prove: if f is monotonically increasing and satisfies the functional equation
-- for all x, then f(2) = e^2 + 1.
theorem find_f_2
  (h_mono : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2)
  (h_eq : ∀ x : ℝ, f (f x - exp x) = exp 1 + 1) :
  f 2 = exp 2 + 1 := sorry

end find_f_2_l96_96709


namespace incorrect_statement_curve_l96_96784

theorem incorrect_statement_curve 
  (t : ℝ) 
  (C : ℝ → ℝ → Prop := λ x y, (x^2 / (2 - t)) - (y^2 / (1 - t)) = 1) :
  ¬ (∀ t, 1 < t ∧ t < 2 → ∃ t, t = 2 ∧ t = 1) :=
sorry

end incorrect_statement_curve_l96_96784


namespace correlation_l96_96068

-- Defining the relationships in the problem
def area_of_square_and_side_length : Prop := ∃ (a : ℝ), ∃ (s : ℝ), a = s^2
def height_and_weight : Prop := ∃ (h w : ℝ), ∀ (h1 w1 : ℝ), h1 < h → w1 > w
def distance_and_time : Prop := ∃ (d : ℝ), ∃ (t : ℝ), ∃ (v : ℝ), d = v * t
def radius_and_volume : Prop := ∃ (r : ℝ), ∃ (v : ℝ), v = (4 / 3) * π * r^3

-- The proof that only height and weight have a correlation
theorem correlation : height_and_weight := sorry

end correlation_l96_96068


namespace andrew_total_payment_l96_96645

-- Given conditions
def quantity_of_grapes := 14
def rate_per_kg_grapes := 54
def quantity_of_mangoes := 10
def rate_per_kg_mangoes := 62

-- Calculations
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Theorem to prove
theorem andrew_total_payment : total_amount_paid = 1376 := by
  sorry

end andrew_total_payment_l96_96645


namespace find_number_l96_96992

theorem find_number (x : ℝ) :
  (7 * (x + 10) / 5) - 5 = 44 → x = 25 :=
by
  sorry

end find_number_l96_96992


namespace john_unanswered_questions_l96_96827

theorem john_unanswered_questions :
  ∃ (c w u : ℕ), (30 + 4 * c - w = 84) ∧ (5 * c + 2 * u = 93) ∧ (c + w + u = 30) ∧ (u = 9) :=
by
  sorry

end john_unanswered_questions_l96_96827


namespace complete_square_form_l96_96176

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 10 * x + 15

theorem complete_square_form (b c : ℤ) (h : ∀ x : ℝ, quadratic_expr x = 0 ↔ (x + b)^2 = c) :
  b + c = 5 :=
sorry

end complete_square_form_l96_96176


namespace move_line_left_and_up_l96_96188

/--
The equation of the line obtained by moving the line y = 2x - 3
2 units to the left and then 3 units up is y = 2x + 4.
-/
theorem move_line_left_and_up :
  ∀ (x y : ℝ), y = 2*x - 3 → ∃ x' y', x' = x + 2 ∧ y' = y + 3 ∧ y' = 2*x' + 4 :=
by
  sorry

end move_line_left_and_up_l96_96188


namespace merchant_product_quantities_l96_96991

theorem merchant_product_quantities
  (x p1 : ℝ)
  (h1 : 4000 = x * p1)
  (h2 : 8800 = 2 * x * (p1 + 4))
  (h3 : (8800 / (2 * x)) - (4000 / x) = 4):
  x = 100 ∧ 2 * x = 200 :=
by sorry

end merchant_product_quantities_l96_96991


namespace find_x_l96_96616

-- Define the conditions as variables and the target equation
variable (x : ℝ)

theorem find_x : 67 * x - 59 * x = 4828 → x = 603.5 := by
  intro h
  sorry

end find_x_l96_96616


namespace dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l96_96650

-- Define a regular dodecagon
def dodecagon_sides : ℕ := 12

-- Prove that the number of diagonals in a regular dodecagon is 54
theorem dodecagon_diagonals_eq_54 : (dodecagon_sides * (dodecagon_sides - 3)) / 2 = 54 :=
by sorry

-- Prove that the number of possible triangles formed from a regular dodecagon vertices is 220
theorem dodecagon_triangles_eq_220 : Nat.choose dodecagon_sides 3 = 220 :=
by sorry

end dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l96_96650


namespace triangle_properties_l96_96821

theorem triangle_properties (a b c AD : ℝ) (h1 : a = 2) (h2 : AD = 2) :
  (b^2 + c^2 = 10) ∧ (3/5 ≤ Real.cos A ∧ Real.cos A < 1) ∧ (angle BAD ≤ π / 6) :=
sorry

end triangle_properties_l96_96821


namespace remainder_division_l96_96685

theorem remainder_division (G Q1 R1 Q2 : ℕ) (hG : G = 88)
  (h1 : 3815 = G * Q1 + R1) (h2 : 4521 = G * Q2 + 33) : R1 = 31 :=
sorry

end remainder_division_l96_96685


namespace four_circles_sum_of_distances_l96_96722

theorem four_circles_sum_of_distances
  {A B C D P Q R : Type*}
  (hP : ∀ (x : Type*), x ∈ [{A, B, C, D}] → P ∈ x)
  (hQ : ∀ (x : Type*), x ∈ [{A, B, C, D}] → Q ∈ x)
  (hR : R = (P + Q) / 2)
  (radius_A radius_B radius_C radius_D : ℝ)
  (h_rad_AB : radius_A = (4/7) * radius_B)
  (h_rad_CD : radius_C = (4/7) * radius_D)
  (AB CD PQ : ℝ)
  (h_AB : AB = 50)
  (h_CD : CD = 50)
  (h_PQ : PQ = 60) :
  (dist A R) + (dist B R) + (dist C R) + (dist D R) = 220 :=
by sorry

end four_circles_sum_of_distances_l96_96722


namespace factor_polynomial_l96_96299

theorem factor_polynomial (x : ℝ) : 
  (20 * x^3 + 100 * x - 10) - (-3 * x^3 + 5 * x - 15) = 5 * (23 * x^3 + 19 * x + 1) := 
by 
  -- Proof can be filled in here
  sorry

end factor_polynomial_l96_96299


namespace ball_attendance_l96_96857

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l96_96857


namespace quadratic_term_elimination_l96_96959

theorem quadratic_term_elimination (m : ℝ) :
  (3 * (x : ℝ) ^ 2 - 10 - 2 * x - 4 * x ^ 2 + m * x ^ 2) = -(x : ℝ) * (2 * x + 10) ↔ m = 1 := 
by sorry

end quadratic_term_elimination_l96_96959


namespace malcom_cards_left_l96_96295

theorem malcom_cards_left (brandon_cards : ℕ) (h1 : brandon_cards = 20) (h2 : ∀ malcom_cards : ℕ, malcom_cards = brandon_cards + 8) (h3 : ∀ mark_cards : ℕ, mark_cards = (malcom_cards / 2)) : 
  malcom_cards - mark_cards = 14 := 
by 
  let malcom_cards := 28
  let mark_cards := (malcom_cards / 2)
  have h4 : mark_cards = 14, from rfl
  show malcom_cards - mark_cards = 14, from sorry

end malcom_cards_left_l96_96295


namespace minimize_expression_l96_96673

open Real

noncomputable def expression (x : ℝ) : ℝ :=
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + x + 2023

theorem minimize_expression :
  ∃ x : ℝ, expression x = 2022 - (5 + sqrt 5) / 2 :=
sorry

end minimize_expression_l96_96673


namespace log_product_one_l96_96090

theorem log_product_one {M N P : ℝ}
  (h1 : log M N = log N M)
  (h2 : log P M = log M P)
  (h3 : M ≠ 1 ∧ N ≠ 1 ∧ P ≠ 1)
  (h4 : M > 0 ∧ N > 0 ∧ P > 0)
  (h5 : M ≠ N ∧ N ≠ P ∧ M ≠ P):
  M * N * P = 1 :=
by 
  sorry

end log_product_one_l96_96090


namespace simplify_f_find_value_f_l96_96707

variables (θ : ℝ)

def f (θ : ℝ) : ℝ := (sin (θ + (5/2) * real.pi) * cos ((3/2) * real.pi - θ) * cos (θ + 3 * real.pi)) / (cos (-real.pi / 2 - θ) * sin (- (3/2) * real.pi - θ))

-- Prove the simplified form of f(θ) is -cos(θ)
theorem simplify_f : ∀ θ : ℝ, f θ = -cos θ := 
by sorry

-- Given condition: sin(θ - π/6) = 3/5,
variable (h : sin (θ - real.pi / 6) = 3 / 5)

-- Prove that f(θ + π/3) = 3/5
theorem find_value_f : f (θ + real.pi / 3) = 3 / 5 :=
by sorry

end simplify_f_find_value_f_l96_96707


namespace misha_class_predictions_probability_l96_96588

-- Definitions representing the conditions
def monday_classes : ℕ := 5
def tuesday_classes : ℕ := 6
def total_classes : ℕ := monday_classes + tuesday_classes
def total_flips : ℕ := total_classes
def correct_predictions : ℕ := 7
def correct_monday_predictions : ℕ := 3

-- Lean theorem representing the proof problem
theorem misha_class_predictions_probability :
  (probability_of_correct_predictions total_flips correct_predictions monday_classes correct_monday_predictions) =
    (5 / 11) :=
  sorry

end misha_class_predictions_probability_l96_96588


namespace complex_set_equality_l96_96210

theorem complex_set_equality : 
  (∃ (Z : set ℂ), Z = {Z | ∃ (n : ℤ), Z = (complex.i ^ n + complex.i ^ -n)}) :=
sorry

end complex_set_equality_l96_96210


namespace ellipse_eccentricity_range_l96_96883

theorem ellipse_eccentricity_range (a : ℝ) (h1 : 1 < a) :
  let e := sqrt (1 - 1 / a^2) in
  ∀ circle_centered_at_A (A : ℝ × ℝ), A = (0, 1) →
  (∀ c : set ℝ × set ℝ, is_circle c → (intersect_points c (Γ )) ≤ 3) →
  e ∈ set.Ioo 0 (1 / 2) :=
by
  sorry

end ellipse_eccentricity_range_l96_96883


namespace area_of_triangle_AEB_l96_96120

theorem area_of_triangle_AEB
  (ABCD : Rectangle)
  (A B C D F G E : Point)
  (h_rect : ABCD.is_rectangle)
  (h_AB : distance A B = 8)
  (h_BC : distance B C = 4)
  (h_DF : distance D F = 3)
  (h_GC : distance G C = 1)
  (h_AF_BG_intersect_E : IntersectingLines (Line.through A F) (Line.through B G) E)
  : area (Triangle.mk A E B) = 8 :=
sorry

end area_of_triangle_AEB_l96_96120


namespace binary_1111_to_decimal_l96_96965

-- Define the binary number "1111"
def binary_num : list ℕ := [1, 1, 1, 1]

-- Function to convert a binary number to its decimal representation
def binary_to_decimal (bn : list ℕ) : ℕ :=
  bn.reverse.zip_with (λ b i => b * (2^i)) (list.range bn.length) |>.sum

-- Prove that the decimal representation of the binary number "1111" is 15
theorem binary_1111_to_decimal : binary_to_decimal binary_num = 15 :=
by
  -- binary_to_decimal [1, 1, 1, 1] = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0
  -- = 8 + 4 + 2 + 1
  -- = 15
  sorry

end binary_1111_to_decimal_l96_96965


namespace sum_and_product_of_roots_l96_96948

theorem sum_and_product_of_roots (m n : ℝ) (h1 : (m / 3) = 9) (h2 : (n / 3) = 20) : m + n = 87 :=
by
  sorry

end sum_and_product_of_roots_l96_96948


namespace sum_seq_l96_96447

noncomputable def a_n (n : ℕ) : ℝ := sorry
noncomputable def S_n (n : ℕ) : ℝ := sorry

theorem sum_seq (q : ℝ) (hq : q ≠ 0) 
  (h_a1 : a_n 1 = 1) (h_sn3_sn6 : 9 * S_n 3 = S_n 6)
  (h_geometric : ∀ n, a_n (n + 1) = q * a_n n) :
  (∑ i in finset.range 5, 1 / a_n (i + 1)) = 31 / 16 :=
sorry

end sum_seq_l96_96447


namespace smallest_consecutive_composite_sum_102_l96_96097

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ m * k = n

noncomputable def find_smallest_consecutive_composite_sum (n m k l : ℕ) : ℕ :=
  if is_composite n ∧ is_composite m ∧ is_composite k ∧ is_composite l ∧ n + 1 = m ∧ m + 1 = k ∧ k + 1 = l then
    n + m + k + l
  else
    0

theorem smallest_consecutive_composite_sum_102 : find_smallest_consecutive_composite_sum 24 25 26 27 = 102 := 
by
  sorry

end smallest_consecutive_composite_sum_102_l96_96097


namespace find_first_number_l96_96940

theorem find_first_number {A : ℕ} :
  nat.lcm A 913 = 2310 ∧ nat.gcd A 913 = 83 → A = 210 :=
by
  sorry  -- skipping the proof

end find_first_number_l96_96940


namespace first_pipe_fill_time_l96_96266

-- Define the conditions
def rate_first_pipe (T : ℝ) : ℝ := 1 / T
def rate_second_pipe : ℝ := 1 / 6
def combined_rate : ℝ := 1 / 3.75

-- Define the problem to prove
theorem first_pipe_fill_time :
  ∀ T : ℝ,
    rate_first_pipe T + rate_second_pipe = combined_rate →
    T ≈ 8.18 := 
by sorry

end first_pipe_fill_time_l96_96266


namespace value_of_b_2_pow_4_l96_96141

noncomputable def b : ℕ → ℕ
| 1 := 3
| (2 * n) := 2 * n * b n + 1
| _ := 0 -- edge case that should not be reached.

theorem value_of_b_2_pow_4 : b (2^4) = 3729 := by
  sorry

end value_of_b_2_pow_4_l96_96141


namespace count_three_digit_integers_l96_96404

theorem count_three_digit_integers (h : ℕ → Prop) (t : ℕ → Prop) (o : ℕ → Prop) :
  (∀ n, 100 ≤ n ∧ n ≤ 999 →
    (h (n / 100) ∧ t ((n / 10) % 10) ∧ o (n % 10)) ↔ h (n / 100) ∧ t ((n / 10) % 10) ∧ o (n % 10)) ∧
  (∀ x, 1 < x → x ∈ {5, 6, 7, 8, 9} → h x) ∧
  (∀ x, 1 < x → x ∈ {2, 3, 4, 5, 6, 7, 8, 9} → t x) ∧
  (∀ x, 1 < x → x = 5 → o x) →
  ∃ L : ℕ, L = 40 := 
begin
  sorry
end

end count_three_digit_integers_l96_96404


namespace solve_roots_of_equation_l96_96691

noncomputable def roots_of_equation (x : ℝ) : Prop :=
  3 * real.sqrt x + 3 * x^(-1/2) = 9

theorem solve_roots_of_equation :
  ∀ x : ℝ, x > 0 → roots_of_equation x →
  (x = (9 + 3 * real.sqrt 5) / 6 ^ 2) ∨ ( x = (9 - 3 * real.sqrt 5) / 6 ^ 2) :=
by
  sorry

end solve_roots_of_equation_l96_96691


namespace false_propositions_l96_96052

variable {a b c x1 x2 : ℝ}
variable (h_eq : a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)  -- condition
variable (h_reals : a ≠ 0)  -- additional condition to prevent division by zero

theorem false_propositions :
  ∃ (p1 p2 : Prop), 
    (p1 = (complex.conjugate x1 = x1 ∧ x1 ≠ x2)) ∧ 
    (p2 = (a * x^2 + b * x + c ≠ (x - x1) * (x - x2))) ∧ 
    p1 ∧ p2 := 
by
  sorry

end false_propositions_l96_96052


namespace jackson_total_calories_l96_96823

def lettuce_calories : ℕ := 50
def carrots_calories : ℕ := 2 * lettuce_calories
def dressing_calories : ℕ := 210
def salad_calories : ℕ := lettuce_calories + carrots_calories + dressing_calories

def crust_calories : ℕ := 600
def pepperoni_calories : ℕ := crust_calories / 3
def cheese_calories : ℕ := 400
def pizza_calories : ℕ := crust_calories + pepperoni_calories + cheese_calories

def jackson_salad_fraction : ℚ := 1 / 4
def jackson_pizza_fraction : ℚ := 1 / 5

noncomputable def total_calories : ℚ := 
  jackson_salad_fraction * salad_calories + jackson_pizza_fraction * pizza_calories

theorem jackson_total_calories : total_calories = 330 := by
  sorry

end jackson_total_calories_l96_96823


namespace largest_prime_divisor_for_primality_check_l96_96107

theorem largest_prime_divisor_for_primality_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) : 
  ∃ p, Prime p ∧ p ≤ Int.sqrt 1050 ∧ ∀ q, Prime q → q ≤ Int.sqrt n → q ≤ p := sorry

end largest_prime_divisor_for_primality_check_l96_96107


namespace sum_of_lengths_of_intervals_eq_one_l96_96035

noncomputable def f (x : ℝ) := (⌊x⌋ : ℝ) * (2013 ^ (x - ⌊x⌋) - 1)

theorem sum_of_lengths_of_intervals_eq_one :
  (∑ k in Finset.range 2012 + 1, Real.log (1 + 1 / k)) = 1 := by
  sorry

end sum_of_lengths_of_intervals_eq_one_l96_96035


namespace number_of_digits_in_sequence_9000_pow_1000_l96_96481

noncomputable def smallest_nonzero_digit (n : ℕ) : ℕ :=
  n.digits.base 10 |>
  List.filter (λ d => d ≠ 0) |>
  List.minimum' (0)

noncomputable def sequence (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ k a_k, a_k + smallest_nonzero_digit a_k)

theorem number_of_digits_in_sequence_9000_pow_1000 :
  ∃ k, Nat.digits 10 (sequence (9 * 1000 ^ 1000)) = k ∧ k = 3001 :=
by
  sorry

end number_of_digits_in_sequence_9000_pow_1000_l96_96481


namespace profit_margin_increase_to_10_l96_96629

-- Problem definitions
variables (x : ℕ) (c s : ℚ)
-- Conditions
def current_profit_margin := s = c * (1 + x / 100)
def discounted_cost := ∃ c', c' = 0.88 * c
def new_profit_margin :=  ∃ c', c * (1 + x / 100) = c' * (1 + (x + 15) / 100)

-- Theorem statement
theorem profit_margin_increase_to_10 :
  (current_profit_margin x c s) →
  (discounted_cost c) →
  (new_profit_margin x c s) →
  x = 10 :=
begin
  intros h1 h2 h3,
  sorry, -- Proof goes here
end

end profit_margin_increase_to_10_l96_96629


namespace triangle_equality_l96_96760

-- Define the problem conditions using Lean definitions
variables {A B C P Q R : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace R]

-- Assume midpoint conditions for points A and P
axiom A_bisects_QR : midpoint A Q R
axiom P_bisects_BC : midpoint P B C

-- Angles bisected by segments
axiom QR_bisects_angle_BAC : bisects (angle B A C) Q R
axiom BC_bisects_angle_QPR : bisects (angle Q P R) B C

-- Required proof
theorem triangle_equality :
  AB + AC = PQ + PR :=
sorry

end triangle_equality_l96_96760


namespace train_clicks_to_time_l96_96209

theorem train_clicks_to_time 
  (x : ℕ) -- Assume x is the speed in mph, where mph is a positive integer
  (rails_length : ℝ) (click_heard : ℝ) : 
  rails_length = 40 → click_heard = 1 →
  let clicks_per_minute := (5280 * x) / (60 * 40) 
  let time_minutes := x / clicks_per_minute 
  let time_seconds := time_minutes * 60 
  time_seconds ≈ 27.27 :=
by
  intro h1 h2
  -- conversions and calculations
  sorry

end train_clicks_to_time_l96_96209


namespace shortest_path_on_cube_l96_96286

theorem shortest_path_on_cube 
  (cube : Type) 
  (edge_length : cube → ℝ) 
  (midpoint : cube → cube → cube) 
  (A B C D : cube)
  (hA : edge_length A = 1)
  (hB : edge_length B = 1)
  (hC : edge_length C = 1)
  (hD : edge_length D = 1)
  (mid_AB : midpoint A B = midpoint B A)
  (mid_CD : midpoint C D = midpoint D C)
  : shortest_path_length cube edge_length midpoint A C = 2 * real.sqrt 2 + 1 := 
sorry

end shortest_path_on_cube_l96_96286


namespace shaded_area_calculation_l96_96214

-- Definitions based on the given conditions
def side_length_of_square : ℝ := 8
def radius_of_circle : ℝ := 3
def area_square : ℝ := side_length_of_square ^ 2

-- Calculated based on the structure of the problem where the area of sectors and triangles are subtracted
def area_of_circular_sector : ℝ := (radius_of_circle ^ 2) * (Real.pi / 4)
def total_area_of_sectors : ℝ := 2 * area_of_circular_sector
def area_of_triangle : ℝ := (1 / 2) * ((side_length_of_square - radius_of_circle) ^ 2) 
def total_area_of_triangles : ℝ := 4 * area_of_triangle

-- Proving the shaded area
theorem shaded_area_calculation :
    (area_square - total_area_of_sectors - total_area_of_triangles) = (14 - (9 * Real.pi) / 2) :=
by
    sorry  -- Proof omitted

end shaded_area_calculation_l96_96214


namespace isos_trap_perimeter_l96_96118

theorem isos_trap_perimeter :
  ∀ (x : ℝ), x > 0 →
    let EF := 2 * x
    let GH := x
    let area := 72
    let h := 48 / x
    let leg_length := sqrt ((48 / x)^2 + (x / 2)^2)
  in area = (1 / 2) * (EF + GH) * h → -- ensure the area condition is included
    18 + 2 * (leg_length) = 18 + 2 * sqrt 73 
:= by 
  intros x hx pos_x h_eq area_eq 
  let EF := 2 * x
  let GH := x
  let h := 48 / x
  let leg_length := sqrt ((48 / x)^2 + (x / 2)^2)
  sorry

end isos_trap_perimeter_l96_96118


namespace find_n_l96_96391

open Real

def a_n (n : ℕ) (h : n > 0) : ℝ := log 2 (n / (n + 1))

def S_n (n : ℕ) (h : n > 0) : ℝ := ∑ i in Finset.range n + 1, a_n i (nat.succ_pos _)

theorem find_n : ∃ n : ℕ, n > 0 ∧ S_n n sorry < -4 ∧ ∀ m, m < n → S_n m sorry ≥ -4 :=
by
  sorry

end find_n_l96_96391


namespace first_marvelous_monday_after_school_start_l96_96802

noncomputable def school_start_date := ⟨2023, 10, 2⟩ -- October 2, 2023 is a Monday

def is_monday (d : Nat) (m : Nat) (year : Nat) : Prop := 
  -- Dummy predicate to check if a given date is a Monday
  sorry

def is_marvelous_monday (d : Nat) (m : Nat) (year : Nat) : Prop := 
  -- Dummy predicate to check if a given date is a Marvelous Monday
  sorry

theorem first_marvelous_monday_after_school_start : 
  ∃ day : Nat, is_marvelous_monday day 10 2023 ∧ day = 30 :=
by
  sorry

end first_marvelous_monday_after_school_start_l96_96802


namespace distinct_points_count_l96_96308

theorem distinct_points_count :
  ∃ (P : Finset (ℝ × ℝ)), 
    (∀ p ∈ P, p.1^2 + p.2^2 = 1 ∧ p.1^2 + 9 * p.2^2 = 9) ∧ P.card = 2 :=
by
  sorry

end distinct_points_count_l96_96308


namespace range_of_f_prime_tangent_line_equation_l96_96071

-- Definition of the function
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x

-- Problem 1: Prove the range of f' on [-2, 1] is [-3, 9]
theorem range_of_f_prime : Set.range (λ x : {a : ℝ // -2 ≤ a ∧ a ≤ 1}, f' x) = Set.Icc (-3) 9 := sorry

-- Parametric Line definition
def l (t : ℝ) : (ℝ × ℝ) := 
  (1 / 2 + (3 * Real.sqrt 10) / 10 * t, 
   1 / 3 + (Real.sqrt 10) / 10 * t)

-- Problem 2: Prove the equation of the tangent line that is perpendicular to the given line
theorem tangent_line_equation :
  ∃ (a b : ℝ), (2*a - 6*b + 1 = 0) ∧ (b = f a) ∧ (3*a^2 + 6*a = -3) ∧ (∀ (x y), y + 3*x + 6 = 0) := sorry

end range_of_f_prime_tangent_line_equation_l96_96071


namespace geometric_sequence_sum_cn_l96_96718

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i

theorem geometric_sequence (a : ℕ → ℝ) (h : ∀ n : ℕ, 2 * Sn a n = 3 * (a n - 1)) :
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

theorem sum_cn (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ n : ℕ, 2 * Sn a n = 3 * (a n - 1))
  (h2 : ∀ n : ℕ, b n = Real.log 3 (a n))
  (h3 : ∀ n : ℕ, c n = a n * b n)
  (h4 : ∀ n : ℕ, a n = 3^n) :
  ∑ i in Finset.range (n + 1), c i = (2 * (n : ℝ) - 1) * 3 ^ (n + 1) / 4 + 3 / 4 :=
sorry

end geometric_sequence_sum_cn_l96_96718


namespace sequence_general_terms_sum_of_first_n_terms_l96_96271

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 -- ℕ starting from 1 so handling 0 case
  else n

noncomputable def b_n (n : ℕ) : ℕ :=
  2 ^ (a_n n + 1)

noncomputable def log2 (n : ℕ) : ℕ :=
  if n = 0 then 0 -- handling log2 of 0 case
  else Nat.log2 n

noncomputable def T_n (n : ℕ) : ℚ :=
  let term (k : ℕ) := (1 : ℚ) / (a_n k * log2 (b_n k))
  (List.range n).sum term

theorem sequence_general_terms (n : ℕ) (hn : n ≥ 1) : 
  (a_n n = n) ∧ (b_n n = 2^(n+1)) :=
by
  -- The proof will go here
  sorry

theorem sum_of_first_n_terms (n : ℕ) (hn : n ≥ 1) :
  T_n n = (n : ℚ) / (n + 1) :=
by
  -- The proof will go here
  sorry

end sequence_general_terms_sum_of_first_n_terms_l96_96271


namespace Malcom_cards_after_giving_away_half_l96_96293

def Brandon_cards : ℕ := 20
def Malcom_initial_cards : ℕ := Brandon_cards + 8
def Malcom_remaining_cards : ℕ := Malcom_initial_cards - (Malcom_initial_cards / 2)

theorem Malcom_cards_after_giving_away_half :
  Malcom_remaining_cards = 14 :=
by
  sorry

end Malcom_cards_after_giving_away_half_l96_96293


namespace ternary_to_decimal_121_l96_96661

theorem ternary_to_decimal_121 : 
  let t : ℕ := 1 * 3^2 + 2 * 3^1 + 1 * 3^0 
  in t = 16 :=
by
  sorry

end ternary_to_decimal_121_l96_96661


namespace number_of_books_in_library_l96_96041

theorem number_of_books_in_library 
  (a : ℕ) 
  (R L : ℕ) 
  (h1 : a = 12 * R + 7) 
  (h2 : a = 25 * L - 5) 
  (h3 : 500 < a ∧ a < 650) : 
  a = 595 :=
begin
  sorry
end

end number_of_books_in_library_l96_96041


namespace determine_c_div_d_l96_96305

theorem determine_c_div_d (x y c d : ℝ) (h1 : 4 * x + 8 * y = c) (h2 : 5 * x - 10 * y = d) (h3 : d ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) : c / d = -4 / 5 :=
by
sorry

end determine_c_div_d_l96_96305


namespace domain_of_log_function_l96_96245

theorem domain_of_log_function (x : ℝ) : 
  (0 < x ∧ x < 1) ↔ ∃ y, y = ln (1 / x - 1) := 
by
  sorry

end domain_of_log_function_l96_96245


namespace count_both_symmetries_is_three_l96_96642

-- Definitions for the given shapes and properties
def isAxisymmetric (shape : String) : Prop :=
  shape = "equilateral triangle" ∨ shape = "rectangle" ∨ shape = "rhombus" ∨
  shape = "square" ∨ shape = "regular pentagon"

def isCentrallySymmetric (shape : String) : Prop :=
  shape = "rectangle" ∨ shape = "rhombus" ∨ shape = "square"

-- List of shapes
def shapes := ["equilateral triangle", "parallelogram", "rectangle", "rhombus", "square", "regular pentagon"]

-- Definition for counting shapes that are both axisymmetric and centrally symmetric
def countBothSymmetries (shapes : List String) : Nat :=
  shapes.count (λ shape, isAxisymmetric shape ∧ isCentrallySymmetric shape)

-- Theorem to prove
theorem count_both_symmetries_is_three : countBothSymmetries shapes = 3 :=
by
  sorry

end count_both_symmetries_is_three_l96_96642


namespace find_integers_solution_l96_96683

theorem find_integers_solution (a b : ℕ) (h₁ : a ≠ b)
    (h₂ : ∃ k : ℕ, a + b^2 = 3^k)
    (h₃ : a + b^2 ∣ a^2 + b) : (a, b) = (5, 2) :=
begin
  sorry
end

end find_integers_solution_l96_96683


namespace ball_attendance_l96_96860

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l96_96860


namespace area_of_rectangle_l96_96572

theorem area_of_rectangle :
  ∀ (A B C D E F : Point) (L L' : Line),
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
  parallel L L' ∧ passes_through L A ∧ passes_through L' C ∧
  perp_to L DB ∧ perp_to L' DB ∧
  divides DB E F ∧ length DB = 3 ∧
  DB = segment A 1 D ∧ E = segment D 1 B ∧ F = segment E 1 B
  → area ABCD = 4.2 :=
sorry

end area_of_rectangle_l96_96572


namespace find_value_of_expression_l96_96028

theorem find_value_of_expression :
  3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 :=
by
  sorry

end find_value_of_expression_l96_96028


namespace sector_angle_measure_l96_96421

theorem sector_angle_measure (r α : ℝ) 
  (h1 : 2 * r + α * r = 6)
  (h2 : (1 / 2) * α * r^2 = 2) :
  α = 1 ∨ α = 4 := 
sorry

end sector_angle_measure_l96_96421


namespace ternary_to_decimal_l96_96664

theorem ternary_to_decimal (n : ℕ) (h : n = 121) : 
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 16 :=
by sorry

end ternary_to_decimal_l96_96664


namespace Mary_has_36_nickels_l96_96155

def initial_nickels : ℕ := 7
def dads_nickels : ℕ := 5
def moms_multipler : ℕ := 2

theorem Mary_has_36_nickels (nickels_initial : ℕ) (nickels_dad : ℕ) (mult_mom : ℕ) : 
  let total_initial : ℕ := nickels_initial + nickels_dad in
  let total_mom : ℕ := mult_mom * total_initial in
  let total_nickels := total_initial + total_mom in
  total_nickels = 36 :=
by
  sorry

#eval Mary_has_36_nickels 7 5 2 -- Should evaluate to true as long as the proof obligation is correct.

end Mary_has_36_nickels_l96_96155


namespace max_value_case1_max_value_case2_max_value_case3_two_zero_points_l96_96070

-- Definitions for the function and conditions given
def f (x : ℝ) (a : ℝ) : ℝ := (x / a) - real.exp x

theorem max_value_case1 (a : ℝ) (h : a > 0) (h₁ : real.log (1 / a) ≥ 2) :
  ∀ x ∈ set.Icc 1 2, f x a ≤ f 2 a := 
sorry

theorem max_value_case2 (a : ℝ) (h : a > 0) (h₁ : 1 < real.log (1 / a)) (h₂ : real.log (1 / a) < 2) :
  ∀ x ∈ set.Icc 1 2, f x a ≤ f (real.log (1 / a)) a := 
sorry

theorem max_value_case3 (a : ℝ) (h : a > 0) (h₁ : real.log (1 / a) ≤ 1) :
  ∀ x ∈ set.Icc 1 2, f x a ≤ f 1 a := 
sorry

-- Definitions for the zero points and the range of a
def g (x : ℝ) : ℝ := x / real.exp x

theorem two_zero_points (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ↔ 0 < a ∧ a < 1 / real.exp 1 := 
sorry

end max_value_case1_max_value_case2_max_value_case3_two_zero_points_l96_96070


namespace coupon_value_l96_96455

theorem coupon_value (orig_price : ℕ) (discount_rate : ℚ) (num_bottles : ℕ) (total_cost : ℕ) (value_of_coupon : ℚ) :
  orig_price = 1500 ∧ discount_rate = 0.20 ∧ num_bottles = 3 ∧ total_cost = 3000 → value_of_coupon = 200 := by
  intros h
  let ⟨h1, h2, h3, h4⟩ := h
  let discounted_price_per_bottle := orig_price * (1 - discount_rate)
  let total_without_coupons := discounted_price_per_bottle * num_bottles
  let total_savings := total_without_coupons - total_cost
  let coupon_value := total_savings / num_bottles
  have : coupon_value = 200 := by sorry
  exact this

end coupon_value_l96_96455


namespace binomial_constant_term_l96_96323

theorem binomial_constant_term :
  let T := λ r : ℕ, (Nat.choose 4 r) * (2 ^ (4 - r)) * (x ^ (2 * r - 4)) 
  in (T 2 : ℤ) = 24 :=
by
  sorry

end binomial_constant_term_l96_96323


namespace linear_transform_determined_by_points_l96_96492

theorem linear_transform_determined_by_points
  (z1 z2 w1 w2 : ℂ)
  (h1 : z1 ≠ z2)
  (h2 : w1 ≠ w2)
  : ∃ (a b : ℂ), ∀ (z : ℂ), a = (w2 - w1) / (z2 - z1) ∧ b = (w1 * z2 - w2 * z1) / (z2 - z1) ∧ (a * z1 + b = w1) ∧ (a * z2 + b = w2) := 
sorry

end linear_transform_determined_by_points_l96_96492


namespace find_second_number_l96_96929

theorem find_second_number
  (average : ℕ)
  (a b : ℕ)
  (n : ℕ)
  (h_average : average = 20)
  (h_a : a = 3)
  (h_b : b = 33)
  (h_n : n = 27) :
  let sum := 4 * average in
  let c := n + 1 in
  let known_sum := a + b + c in
  let second_number := sum - known_sum in
  second_number = 16 :=
by
  sorry

end find_second_number_l96_96929


namespace mustard_amount_l96_96888

def ginger := 9 -- teaspoons
def cardamom := 1 -- teaspoon
def garlic := 6 -- teaspoons
def mustard (m : ℕ) := m -- teaspoons
def chile (m : ℕ) := 4 * m -- teaspoons

theorem mustard_amount (m : ℕ) (h : Float.floor (100 * ginger / (ginger + cardamom + mustard m + garlic + chile m)) = 43) : m = 1 :=
by
  -- Lean Proof steps will go here
  sorry

end mustard_amount_l96_96888


namespace problem_statement_l96_96458

noncomputable def verify_sum (n k : ℕ) (xs : Fin n → ℝ)
  (h_n : 3 ≤ n)
  (h_ordered : ∀ (i j : Fin n), i < j → xs i > xs j)
  (h_k : xs k > 0 ∧ xs (k + 1) ≤ 0) :
  Real :=
∑ i in Finset.range k, (xs i) ^ (n - 2) * ∏ j in (Finset.univ \ {i}), 1 / ((xs i) - (xs j))

theorem problem_statement (n k : ℕ) (xs : Fin n → ℝ)
  (h_n : 3 ≤ n)
  (h_ordered : ∀ (i j : Fin n), i < j → xs i > xs j)
  (h_k : xs k > 0 ∧ xs (k + 1) ≤ 0) :
  0 ≤ verify_sum n k xs h_n h_ordered h_k :=
sorry

end problem_statement_l96_96458


namespace angles_of_triangle_ABC_l96_96528

-- Definitions based on conditions
def vertex_angle (A : Type) [InnerProductSpace ℝ A] : ℝ := 70 * (π / 180)
def angle_AOB (A O B : Type) [InnerProductSpace ℝ A] : ℝ := 30 * (π / 180)
def angle_COB (C O B : Type) [InnerProductSpace ℝ C] : ℝ := 40 * (π / 180)
def total_angle (A O C : Type) [InnerProductSpace ℝ A] : ℝ := angle_AOB A O B + angle_COB C O B

-- Triangle ABC implies from these setups
def triangle_angles (A B C : Type) [InnerProductSpace ℝ A] :=
  let α := angle_AOB A O B * (180 / π) in
  let β := angle_COB C O B * (180 / π) in
  let γ := total_angle A O C * (180 / π) - α - β in
  (α, β, γ)

-- Lean statement for the proof problem
theorem angles_of_triangle_ABC :
  ∃ (A B C : Type) [InnerProductSpace ℝ A], 
  let (α, β, γ) := triangle_angles A B C in
  α = 30 ∧ β = 40 ∧ γ = 110 :=
by 
  -- We are only stating the theorem, the proof is not required.
  sorry

end angles_of_triangle_ABC_l96_96528


namespace decreasing_order_l96_96065

namespace ProofProblem

-- Define that y = f(x-1) is an even function
def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f(x - 1) = f(-x - 1)

-- Define the monotonically decreasing property
def monotone_decreasing (f : ℝ → ℝ) : Prop :=
∀ {x₁ x₂ : ℝ}, x₁ ∈ [-1,∞) → x₂ ∈ [-1,∞) → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0

-- Define the specific points a, b, c
noncomputable def a (f : ℝ → ℝ) : ℝ :=
f ((Real.log 7 - Real.log 2) / -Real.log 2)

noncomputable def b (f : ℝ → ℝ) : ℝ :=
f ((Real.log 7 - Real.log 2) / -Real.log 3)

noncomputable def c (f : ℝ → ℝ) : ℝ :=
f ((Real.log 3) / (2 * Real.log 2))

-- Now we prove the decreasing order b > a > c
theorem decreasing_order (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_mono : monotone_decreasing f): 
  b f > a f ∧ a f > c f :=
sorry

end ProofProblem

end decreasing_order_l96_96065


namespace library_books_l96_96037

theorem library_books (a : ℕ) (R L : ℕ) :
  (∃ R, a = 12 * R + 7) ∧ (∃ L, a = 25 * L - 5) ∧ 500 < a ∧ a < 650 → a = 595 :=
by
  sorry

end library_books_l96_96037


namespace mutually_exclusive_not_opposed_l96_96218

-- Define the types for cards and people
inductive Card
| red : Card
| white : Card
| black : Card

inductive Person
| A : Person
| B : Person
| C : Person

-- Define the event that a person receives a specific card
def receives (p : Person) (c : Card) : Prop := sorry

-- Conditions
axiom A_receives_red : receives Person.A Card.red → ¬ receives Person.B Card.red
axiom B_receives_red : receives Person.B Card.red → ¬ receives Person.A Card.red

-- The proof problem statement
theorem mutually_exclusive_not_opposed :
  (receives Person.A Card.red → ¬ receives Person.B Card.red) ∧
  (¬(receives Person.A Card.red ∧ receives Person.B Card.red)) ∧
  (¬∀ p : Person, receives p Card.red) :=
sorry

end mutually_exclusive_not_opposed_l96_96218


namespace area_of_region_l96_96223

theorem area_of_region (x y : ℝ) (h : x^2 + y^2 + 12 * x + 16 * y = 0) : real.pi * 100 = 100 * real.pi :=
by
  sorry

end area_of_region_l96_96223


namespace product_of_odd_integers_l96_96967

theorem product_of_odd_integers :
  (∏ i in Finset.filter (λ x => x % 2 = 1) (Finset.range 1000), i) =
  (1000.factorial / (2 ^ 500 * 500.factorial)) :=
by
  sorry

end product_of_odd_integers_l96_96967


namespace area_of_triangle_DEF_l96_96543

/-
  Given:
  - Vertex D is at (0, 2)
  - Vertex E is at (6, 0)
  - Vertex F is at (3, 4)
  - Rectangle has dimensions 6 units by 4 units
  
  Prove:
  - The area of triangle DEF is 9 square units.
-/

theorem area_of_triangle_DEF :
  let D := (0, 2)
  let E := (6, 0)
  let F := (3, 4)
  let rect_length := 6
  let rect_width := 4
  let rect_area := rect_length * rect_width
  let triangle_areas := (1/2 * rect_length * 2) + (1/2 * 3 * 4) + (1/2 * 3 * 2)
in rect_area - triangle_areas = 9 := 
sorry

end area_of_triangle_DEF_l96_96543


namespace perpendicular_passes_through_incircle_center_l96_96800

-- Definitions of points and setup condition
variables {A B C A1 B1: Type} [plane_points A B C A1 B1]
variables (triangle_ABC : acute_angled_triangle A B C)
variables (alt_A: altitude A1)
variables (alt_B: altitude B1)
variables (incircle_tangency: incircle_tangent_point A B C)
variables (perpendicular_M_AC: perpendicular_drop_on_line M AC)

-- Theorem statement
theorem perpendicular_passes_through_incircle_center :
  let I := incircle_center triangle_ABC in
  let O := intersection (perpendicular_M_AC) (line IC) in
  O = incircle_center (triangle A1 C B1) :=
sorry

end perpendicular_passes_through_incircle_center_l96_96800


namespace interval_of_decrease_range_of_m_l96_96135

variables (x m : Real)
def OA := (2 * Real.cos x, Real.sqrt 3)
def OB := (Real.sin x + Real.sqrt 3 * Real.cos x, -1)
def f (x : Real) := OA x.1 * OB x.1 + OA x.2 * OB x.2 + 2

theorem interval_of_decrease (k : ℤ) :
  ∀ (x : Real), k * π + π / 12 ≤ x ∧ x ≤ k * π + 7 * π / 12 → 
  (deriv f x < 0) :=
begin
  sorry
end

theorem range_of_m (x : Real) (hx : 0 < x ∧ x < π / 2) (hf : f x + m = 0) :
  -4 ≤ m ∧ m < Real.sqrt 3 - 2 :=
begin
  sorry
end

end interval_of_decrease_range_of_m_l96_96135


namespace length_of_platform_l96_96277

noncomputable def calculate_length_of_platform (train_speed_kmph : ℕ) (time_crossing_platform_s : ℕ) (time_crossing_man_s : ℕ) : ℕ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let length_train := train_speed_mps * time_crossing_man_s
  let total_distance := train_speed_mps * time_crossing_platform_s
  total_distance - length_train

theorem length_of_platform (train_speed_kmph : ℕ) (time_crossing_platform_s : ℕ) (time_crossing_man_s : ℕ)
  (h_train_speed : train_speed_kmph = 72) (h_time_platform : time_crossing_platform_s = 33) (h_time_man : time_crossing_man_s = 18) :
  calculate_length_of_platform train_speed_kmph time_crossing_platform_s time_crossing_man_s = 300 :=
by
  simp [calculate_length_of_platform, h_train_speed, h_time_platform, h_time_man]
  sorry

end length_of_platform_l96_96277


namespace smallest_number_drawn_l96_96253

theorem smallest_number_drawn (total_units : ℕ) (units_selected : ℕ) (sum_drawn : ℕ) 
  (h_units : total_units = 28) (h_selected : units_selected = 4) (h_sum : sum_drawn = 54) : 
  ∃ x : ℕ, (x + (x + 7) + (x + 14) + (x + 21)) = sum_drawn ∧ x = 3 :=
by {
  use 3,
  rw [← h_sum, add_assoc, add_assoc, add_comm 3 7, ← add_assoc, ← add_assoc],
  norm_num,
}

end smallest_number_drawn_l96_96253


namespace trajectory_of_moving_point_P_slope_of_line_BT_l96_96123

section trajectory

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem trajectory_of_moving_point_P :
  (∀ (P : ℝ × ℝ), distance P (1, 0) / abs (P.1 - 2) = real.sqrt 2 / 2 →
  (∃ x y : ℝ, P = (x, y) ∧ (x^2) / 2 + y^2 = 1)) :=
sorry

end trajectory

section slope_BT

variables {x1 y1 x2 y2 : ℝ}

def distance_from_fixed_point (P : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - 1) ^ 2 + P.2 ^ 2)

axiom dist_formed_arithmetic_seq (P1 P2 P3 : ℝ × ℝ) :
  (λ A B C : ℝ × ℝ, distance_from_fixed_point B = (distance_from_fixed_point A + distance_from_fixed_point C) / 2) (P1) (P2) (P3)

theorem slope_of_line_BT
  (A C : ℝ × ℝ)
  (B : ℝ × ℝ := (1, real.sqrt 2 / 2))
  (T : ℝ × ℝ := (0.5, 0))
  (h_AC : (A.1 + C.1 = 2)) :
  dist_formed_arithmetic_seq A B C →
  (A.1^2 / 2 + A.2^2 = 1 ∧ C.1^2 / 2 + C.2^2 = 1) →
  let k := (B.2 - T.2) / (B.1 - T.1)
  in k = real.sqrt 2 :=
sorry

end slope_BT

end trajectory_of_moving_point_P_slope_of_line_BT_l96_96123


namespace total_distance_traveled_l96_96178

noncomputable def minimal_total_distance : Real :=
  7 / 2 * (3 * (2 * 30 * Real.sin (4 * Real.pi / 7)) + (2 * 30 * Real.sin (6 * Real.pi / 7)))

theorem total_distance_traveled :
  (7 / 2 * (3 * (2 * 30 * Real.sin (4 * Real.pi / 7)) + (2 * 30 * Real.sin (6 * Real.pi / 7)))) ≈ 583.67 :=
by
  sorry

end total_distance_traveled_l96_96178


namespace uncle_ben_eggs_l96_96550

noncomputable def total_eggs (total_chickens : ℕ) (roosters : ℕ) (non_egg_laying_hens : ℕ) (eggs_per_hen : ℕ) : ℕ :=
  let total_hens := total_chickens - roosters
  let egg_laying_hens := total_hens - non_egg_laying_hens
  egg_laying_hens * eggs_per_hen

theorem uncle_ben_eggs :
  total_eggs 440 39 15 3 = 1158 :=
by
  unfold total_eggs
  -- Correct steps to prove the theorem can be skipped with sorry
  sorry

end uncle_ben_eggs_l96_96550


namespace sum_of_three_integers_l96_96206

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 125) : a + b + c = 31 :=
sorry

end sum_of_three_integers_l96_96206


namespace line_equation_l96_96813

theorem line_equation (k b : ℚ) :
  (∀ x y : ℚ, (y = k * (x - 3) + 5 + b) ↔ (y = (k * (x - 4)) - 2 + 5 + b)) ∧
  (∀ m n : ℚ, (l : ℚ → ℚ → Prop := λ x y, y = k * x + b) ∧
    (∀ x y : ℚ, l x y → (y = k * (x - 3 - 1) + b + 5 - 2) ∧
      (x, y) = (4 - m, 6 - (k * m + b))) →
  ∃ x y : ℚ, 6 * x - 8 * y + 1 = 0 :=
sorry

end line_equation_l96_96813


namespace no_information_loss_chart_is_stem_and_leaf_l96_96632

theorem no_information_loss_chart_is_stem_and_leaf :
  "The correct chart with no information loss" = "Stem-and-leaf plot" :=
sorry

end no_information_loss_chart_is_stem_and_leaf_l96_96632


namespace correct_exponentiation_operation_l96_96567

theorem correct_exponentiation_operation (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end correct_exponentiation_operation_l96_96567


namespace least_multiple_of_25_gt_450_correct_l96_96555

def least_multiple_of_25_gt_450 : ℕ :=
  475

theorem least_multiple_of_25_gt_450_correct (n : ℕ) (h1 : 25 ∣ n) (h2 : n > 450) : n ≥ least_multiple_of_25_gt_450 :=
by
  sorry

end least_multiple_of_25_gt_450_correct_l96_96555


namespace monkey_ascent_l96_96263

theorem monkey_ascent (x : ℝ) :
  (∀ n : ℕ, (1 ≤ n ∧ n % 2 = 1) → monkey_position x n < 10) →
  (monkey_position x 17 = 10) →
  x = 1.8 :=
sorry

-- Define the position of the monkey ascending and slipping mechanism
def monkey_position (x : ℝ) (t : ℕ) : ℝ :=
  if t = 17
  then 10
  else if t % 2 = 1
       then x * ((t + 1) / 2).toℝ  -- Odd minute, ascending
       else x * (t / 2).toℝ - (t / 2).toℝ  -- Even minute, slipping

-- Setting up the condition for the monkey reaching at minute 17
axiom reaching_condition (x : ℝ):
  monkey_position x 17 = 10 → x = 1.8

end monkey_ascent_l96_96263


namespace trigonometric_identity_l96_96739

noncomputable def cos_75_plus_alpha (alpha : ℝ) : ℝ := 1 / 3

def sin_and_cos_negative (alpha : ℝ) : Prop :=
  sin alpha < 0 ∧ cos alpha < 0

theorem trigonometric_identity (alpha : ℝ) (h_sin_cos_neg : sin_and_cos_negative alpha)
  (h_cos_75_alpha : cos_75_plus_alpha alpha = 1 / 3) :
  cos (105 * (π / 180) - alpha) + sin (alpha - 105 * (π / 180)) = (2 * real.sqrt 2 - 1) / 3 :=
  sorry

end trigonometric_identity_l96_96739


namespace cubes_not_touching_foil_l96_96537

-- Define the variables for length, width, height, and total cubes
variables (l w h : ℕ)

-- Conditions extracted from the problem
def width_is_twice_length : Prop := w = 2 * l
def width_is_twice_height : Prop := w = 2 * h
def foil_covered_prism_width : Prop := w + 2 = 10

-- The proof statement
theorem cubes_not_touching_foil (l w h : ℕ) 
  (h1 : width_is_twice_length l w) 
  (h2 : width_is_twice_height w h) 
  (h3 : foil_covered_prism_width w) : 
  l * w * h = 128 := 
by sorry

end cubes_not_touching_foil_l96_96537


namespace sum_modulo_remainder_l96_96022

theorem sum_modulo_remainder :
  ((82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17) = 12 :=
by
  sorry

end sum_modulo_remainder_l96_96022


namespace max_consecutive_indivisible_l96_96257

def is_indivisible (n : ℕ) : Prop :=
  n >= 10000 ∧ n < 100000 ∧ ∀ a b : ℕ, (100 ≤ a ∧ a < 1000) → (100 ≤ b ∧ b < 1000) → a * b ≠ n

theorem max_consecutive_indivisible :
  ∃ max_n : ℕ, (max_n = 99 ∧ ∀ k (h₁ : k > max_n), ∃ m (h₂ : m ≥ 10000 ∧ m < 100000), 
  is_indivisible (m + k))
  sorry

end max_consecutive_indivisible_l96_96257


namespace b_2016_result_l96_96817

theorem b_2016_result (b : ℕ → ℤ) (h₁ : b 1 = 1) (h₂ : b 2 = 5)
  (h₃ : ∀ n : ℕ, b (n + 2) = b (n + 1) - b n) : b 2016 = -4 := sorry

end b_2016_result_l96_96817


namespace quadratic_with_conditions_l96_96623

noncomputable def quadratic_root (a b c x : ℤ) : Prop := a*x^2 + b*x + c = -55

theorem quadratic_with_conditions (a b c x m n : ℤ)
  (hacond : a ≠ 0)
  (hroots : m ≠ n ∧ m > 0 ∧ n > 0)
  (quadratic_eq : (a * (x^2) + b * x + c) = -55)
  (sum_is_prime : prime (a + b + c))
  (eq1 : a * (1 - (m + n) + mn) = -55)
  : (m = 2 ∧ n = 17) ∨ (m = 17 ∧ n = 2) :=
sorry

end quadratic_with_conditions_l96_96623


namespace finite_cut_possible_l96_96480

theorem finite_cut_possible (N : ℕ) (grid : ℕ × ℕ → bool) (black_cells : finset (ℕ × ℕ)) 
  (hN : black_cells.card = N) :
  ∃ (squares : finset (finset (ℕ × ℕ))),
  (∀ (square ∈ squares), ∃ k, square.card = k ∧ 0.2 * k ≤ (black_cells ∩ square).card ∧ (black_cells ∩ square).card ≤ 0.8 * k) ∧
  (black_cells ⊆ ⋃₀ squares) ∧
  squares.finite := 
sorry

end finite_cut_possible_l96_96480


namespace value_of_8pow_3y_minus_2_l96_96363

theorem value_of_8pow_3y_minus_2 (y : ℝ) (h : 8 ^ (3 * y) = 512) : 8 ^ (3 * y - 2) = 8 :=
by
  sorry

end value_of_8pow_3y_minus_2_l96_96363


namespace find_min_value_omega_l96_96383

noncomputable def min_value_ω (ω : ℝ) : Prop :=
  ∀ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 2 * Real.sin (ω * x)) → ω > 0 →
  (∀ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -2) →
  ω = 3 / 2

-- The statement to be proved:
theorem find_min_value_omega : ∃ ω : ℝ, min_value_ω ω :=
by
  use 3 / 2
  sorry

end find_min_value_omega_l96_96383


namespace triangle_angles_determinant_zero_l96_96467

open Matrix

theorem triangle_angles_determinant_zero (P Q R : ℝ) (h : P + Q + R = 180) :
  det !![ [Real.cos P, Real.sin P, 1], [Real.cos Q, Real.sin Q, 1], [Real.cos R, Real.sin R, 1] ] = 0 :=
by 
  sorry

end triangle_angles_determinant_zero_l96_96467


namespace cycling_hours_with_tailwind_l96_96288

theorem cycling_hours_with_tailwind 
  (x : ℝ)
  (h1 : 15 * x + 10 * (12 - x) = 150)
  : x = 6 :=
begin
  sorry
end

end cycling_hours_with_tailwind_l96_96288


namespace max_attempts_to_open_rooms_l96_96197

theorem max_attempts_to_open_rooms (n : ℕ) (h : n = 10) : 
  let max_attempts := (n * (n + 1)) / 2 in max_attempts = 55 :=
begin
  sorry
end

end max_attempts_to_open_rooms_l96_96197


namespace max_size_subset_no_div_coprime_l96_96355

theorem max_size_subset_no_div_coprime (n : ℕ) : 
  ∃ S : Finset ℕ, 
    S ⊆ Finset.range (n + 1) ∧
    (∀ a b ∈ S, a ≠ b → ¬ (a ∣ b ∨ b ∣ a)) ∧
    (∀ x y ∈ S, x ≠ y → Int.gcd x y ≠ 1) ∧
    S.card = Nat.floor((n + 2) / 4) := 
sorry

end max_size_subset_no_div_coprime_l96_96355


namespace line_intersects_parabola_once_l96_96675

theorem line_intersects_parabola_once (k : ℝ) :
  (x = k)
  ∧ (x = -3 * y^2 - 4 * y + 7)
  ∧ (3 * y^2 + 4 * y + (k - 7)) = 0
  ∧ ((4)^2 - 4 * 3 * (k - 7) = 0)
  → k = 25 / 3 := 
by
  sorry

end line_intersects_parabola_once_l96_96675


namespace find_matrix_N_l96_96019

open Matrix

def N : Matrix (Fin 3) (Fin 3) ℚ :=
  !![
    [8, 5, 0],
    [6, 4, 0],
    [0, 0, 1]
  ]

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  !![
    [-4, 5, 0],
    [6, -8, 0],
    [0, 0, 1]
  ]

def I : Matrix (Fin 3) (Fin 3) ℚ :=
  1

theorem find_matrix_N : N ⬝ A = I := by
  sorry

end find_matrix_N_l96_96019


namespace Misha_probability_l96_96595

open Probability

-- Definitions
def classesMonday := 5
def classesTuesday := 6
def totalClasses := 11
def totalCorrect := 7
def mondayCorrect := 3

-- Calculating binomial probabilities
noncomputable def P_A1 := (binomial classesMonday mondayCorrect) * (1 / 2) ^ classesMonday
noncomputable def P_A2 := (binomial classesTuesday (totalCorrect - mondayCorrect)) * (1 / 2) ^ classesTuesday
noncomputable def P_B := (binomial totalClasses totalCorrect) * (1 / 2) ^ totalClasses

-- Theorem stating the required probability
theorem Misha_probability :
  P_A1 * P_A2 / P_B = 5 / 11 :=
by
  sorry

end Misha_probability_l96_96595


namespace integral_x_squared_l96_96003

theorem integral_x_squared : ∫ x in 0..1, x^2 = 1 / 3 :=
by
  sorry

end integral_x_squared_l96_96003


namespace sheila_hourly_wage_l96_96491

def weekly_working_hours : Nat :=
  (8 * 3) + (6 * 2)

def weekly_earnings : Nat :=
  468

def hourly_wage : Nat :=
  weekly_earnings / weekly_working_hours

theorem sheila_hourly_wage : hourly_wage = 13 :=
by
  sorry

end sheila_hourly_wage_l96_96491


namespace distance_downstream_l96_96247

variables (speed_boat : ℕ) (speed_stream : ℕ) (time_downstream : ℕ)

theorem distance_downstream (h1 : speed_boat = 16) (h2 : speed_stream = 5) (h3 : time_downstream = 4) :
  let effective_speed := speed_boat + speed_stream in
  let distance := effective_speed * time_downstream in
  distance = 84 :=
by
  sorry

end distance_downstream_l96_96247


namespace monogram_count_l96_96157

theorem monogram_count : (Finset.combo 12 2).card = 66 :=
by sorry

end monogram_count_l96_96157


namespace student_mistake_l96_96725

theorem student_mistake (m n : ℕ) (h1 : n ≤ 100) (h2 : decimal_contains_sequence (m / n) "167") : false :=
sorry

end student_mistake_l96_96725


namespace range_of_m_in_second_quadrant_l96_96349

theorem range_of_m_in_second_quadrant (m : ℝ) : 
  let z := (m - 1 : ℂ) + (m + 2) * complex.I in
  z.re < 0 ∧ z.im > 0 ↔ -2 < m ∧ m < 1 :=
by
  sorry

end range_of_m_in_second_quadrant_l96_96349


namespace distance_between_vertices_l96_96872

-- Define the equations of the parabolas
def C_eq (x : ℝ) : ℝ := x^2 + 6 * x + 13
def D_eq (x : ℝ) : ℝ := -x^2 + 2 * x + 8

-- Define the vertices of the parabolas
def vertex_C : (ℝ × ℝ) := (-3, 4)
def vertex_D : (ℝ × ℝ) := (1, 9)

-- Prove that the distance between the vertices is sqrt 41
theorem distance_between_vertices : 
  dist (vertex_C) (vertex_D) = Real.sqrt 41 := 
by
  sorry

end distance_between_vertices_l96_96872


namespace polynomial_roots_cubed_l96_96464

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 3
noncomputable def g (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 3

theorem polynomial_roots_cubed {r : ℝ} (h : f r = 0) :
  g (r^3) = 0 := by
  sorry

end polynomial_roots_cubed_l96_96464


namespace ratio_of_hypotenuse_of_45_45_90_triangle_l96_96046

theorem ratio_of_hypotenuse_of_45_45_90_triangle (a c : ℝ)
    (h_square : ∀ (P1 P2 P3 P4 : ℝ × ℝ), 
        dist P1 P2 = a ∧ dist P2 P3 = a ∧ dist P3 P4 = a ∧ dist P4 P1 = a 
        ∧ dist P1 P3 = 2 * a ∧ dist P2 P4 = 2 * a)
    (h_triangle : ∀ (P1 P2 P3 : ℝ × ℝ),
        dist P1 P3 = 2 * a ∧ dist P2 P3 = c
        ∧  dist P1 P2 = a * sqrt 2 
        ∧ dist P1 P2 = dist P2 P3 ≠ 0 
        ∧ dist P1 P3 = c)  : 
  c / a = 2 :=
sorry

end ratio_of_hypotenuse_of_45_45_90_triangle_l96_96046


namespace shooting_test_mode_mean_l96_96273

theorem shooting_test_mode_mean (x : ℕ) (h1 : multiset.mode {9, 9, x, 8} = 9)
  (h2 : (9 + 9 + x + 8) / 4 = 9) : x = 10 :=
by
  sorry

end shooting_test_mode_mean_l96_96273


namespace eight_sided_dice_theorem_l96_96904
open Nat

noncomputable def eight_sided_dice_probability : ℚ :=
  let total_outcomes := 8^8
  let favorable_outcomes := 8!
  let probability_all_different := favorable_outcomes / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same

theorem eight_sided_dice_theorem :
  eight_sided_dice_probability = 16736996 / 16777216 := by
    sorry

end eight_sided_dice_theorem_l96_96904


namespace not_p_sufficient_not_q_l96_96708

variable (x : ℝ)

def p : Prop := |x - 2| > 3
def q : Prop := x > 5

theorem not_p_sufficient_not_q : (¬p → ¬q) ∧ ¬(¬q → ¬p) :=
by {
  sorry
}

end not_p_sufficient_not_q_l96_96708


namespace sum_mod_17_eq_0_l96_96024

theorem sum_mod_17_eq_0 :
  (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 0 :=
by
  sorry

end sum_mod_17_eq_0_l96_96024


namespace nth_15_nth_2014_l96_96128

def subseq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 
    let m := 2 * n - 1 in
    let k := n + 1 in
    if k % 2 = 0 then (k div 2 + (m - 1)) * (m + 1)
    else ((k + 1) div 2) * (m - 1) + 1

theorem nth_15 : subseq 14 = 25 :=
by sorry

theorem nth_2014 : subseq 2013 = 3965 :=
by sorry

end nth_15_nth_2014_l96_96128


namespace union_of_S_and_T_l96_96727

-- Declare sets S and T
def S : Set ℕ := {3, 4, 5}
def T : Set ℕ := {4, 7, 8}

-- Statement about their union
theorem union_of_S_and_T : S ∪ T = {3, 4, 5, 7, 8} :=
sorry

end union_of_S_and_T_l96_96727


namespace probability_two_females_l96_96432

theorem probability_two_females (total_contestants females males choose_two : ℕ) 
  (h_total : total_contestants = 7) (h_females : females = 4) (h_males : males = 3) 
  (h_choose_two : choose_two = 2) :
  (females.choose choose_two : ℚ) / (total_contestants.choose choose_two : ℚ) = 2 / 7 :=
by
  rw [h_total, h_females, h_males, h_choose_two]
  norm_num
  sorry

end probability_two_females_l96_96432


namespace food_price_before_tax_and_tip_l96_96974

-- Definitions
def totalAmountPaid : ℝ := 198
def tipRate : ℝ := 0.20
def taxRate : ℝ := 0.10
def totalRate : ℝ := 1 + taxRate + tipRate + taxRate * tipRate

-- Proof statement
theorem food_price_before_tax_and_tip :
  ∃ (P : ℝ), totalAmountPaid = totalRate * P ∧ P = 150 :=
by
  use 150
  rw [← mul_assoc]
  sorry

end food_price_before_tax_and_tip_l96_96974


namespace dried_grapes_water_percentage_l96_96700

theorem dried_grapes_water_percentage :
  ∃ (water_percentage : ℕ), 
    (∃ (initial_weight fresh_weight dried_weight non_water_weight water_weight : ℕ),
      90% * initial_weight = (initial_weight - non_water_weight) ∧
      initial_weight = 30 ∧
      dried_weight = 3.75 ∧
      water_weight = dried_weight - non_water_weight ∧
      ((water_weight / dried_weight) * 100) = water_percentage
    ) → water_percentage = 20 := 
sorry

end dried_grapes_water_percentage_l96_96700


namespace side_length_of_hexagon_l96_96162

theorem side_length_of_hexagon (h : ℝ) (s : ℝ) (hex_regular : regular_hexagon) (opp_sides_distance : opposite_sides_distance hex_regular = 18)
  : s = 12 * Real.sqrt 3 :=
by
  -- Given: The distance between opposite sides of a regular hexagon is 18 inches
  have distance_eq : 18 = (Real.sqrt 3 / 2) * s := sorry
  -- Solve for the side length s of the hexagon
  have s_calc : s = 36 / Real.sqrt 3 := by
    rw [distance_eq]
    linarith
  -- Simplify the expression
  have s_simpl : s = 12 * Real.sqrt 3 := by
    rw [s_calc]
    linarith
  exact s_simpl

end side_length_of_hexagon_l96_96162


namespace like_terms_set_l96_96568

theorem like_terms_set (a b : ℕ) (x y : ℝ) : 
  (¬ (a = b)) ∧
  ((-2 * x^3 * y^3 = y^3 * x^3)) ∧ 
  (¬ (1 * x * y = 2 * x * y^3)) ∧ 
  (¬ (-6 = x)) :=
by
  sorry

end like_terms_set_l96_96568


namespace peanuts_difference_is_correct_l96_96829

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- Define the difference in the number of peanuts between Kenya and Jose
def peanuts_difference : ℕ := Kenya_peanuts - Jose_peanuts

-- Prove that the number of peanuts Kenya has minus the number of peanuts Jose has is equal to 48
theorem peanuts_difference_is_correct : peanuts_difference = 48 := by
  sorry

end peanuts_difference_is_correct_l96_96829


namespace mrs_lee_earnings_percentage_l96_96791

noncomputable def percentage_earnings_june (T : ℝ) : ℝ :=
  let L := 0.5 * T
  let L_June := 1.2 * L
  let total_income_june := T
  (L_June / total_income_june) * 100

theorem mrs_lee_earnings_percentage (T : ℝ) (hT : T ≠ 0) : percentage_earnings_june T = 60 :=
by
  sorry

end mrs_lee_earnings_percentage_l96_96791


namespace revenue_change_l96_96212

theorem revenue_change (T C : ℝ) (T_new C_new : ℝ)
  (h1 : T_new = 0.81 * T)
  (h2 : C_new = 1.15 * C)
  (R : ℝ := T * C) : 
  ((T_new * C_new - R) / R) * 100 = -6.85 :=
by
  sorry

end revenue_change_l96_96212


namespace evaluate_expression_l96_96000

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l96_96000


namespace problem_goal_l96_96814

-- Define the problem stating that there is a graph of points (x, y) satisfying the condition
def area_of_graph_satisfying_condition : Real :=
  let A := 2013
  -- Define the pairs (a, b) which are multiples of 2013
  let pairs := [(1, 2013), (3, 671), (11, 183), (33, 61)]
  -- Calculate the area of each region formed by pairs
  let area := pairs.length * 4
  area

-- Problem goal statement proving the area is equal to 16
theorem problem_goal : area_of_graph_satisfying_condition = 16 := by
  sorry

end problem_goal_l96_96814


namespace fraction_paint_available_l96_96612

theorem fraction_paint_available 
  (initial_paint : ℝ)
  (first_day_usage : initial_paint / 2) 
  (second_day_remaining : initial_paint / 2)
  (second_day_usage : second_day_remaining / 2) 
  (spill_effect : (second_day_remaining - second_day_usage) / 4) 
  (final_remaining_paint : second_day_remaining - second_day_usage - spill_effect) :
  final_remaining_paint / initial_paint = 3 / 16 := 
  by sorry

end fraction_paint_available_l96_96612


namespace turtle_reaches_watering_hole_l96_96569

theorem turtle_reaches_watering_hole:
  ∀ (x y : ℝ)
   (dist_first_cub : ℝ := 6 * y)
   (dist_turtle : ℝ := 32 * x)
   (time_first_encounter := (dist_first_cub - dist_turtle) / (y - x))
   (time_second_encounter := dist_turtle / (x + 1.5 * y))
   (time_between_encounters = 2.4),
   time_second_encounter - time_first_encounter = time_between_encounters →
   (time_turtle_reaches := time_second_encounter + 32 - 3.2) -- since total_time - the time between second 
   →
   time_turtle_reaches = 28.8 :=
begin
  intros x y,
  sorry
end

end turtle_reaches_watering_hole_l96_96569


namespace integer_solutions_of_xyz_equation_l96_96321

/--
  Find all integer solutions of the equation \( x + y + z = xyz \).
  The integer solutions are expected to be:
  \[
  (1, 2, 3), (2, 1, 3), (3, 1, 2), (3, 2, 1), (1, 3, 2), (2, 3, 1), (-a, 0, a) \text{ for } (a : ℤ).
  \]
-/
theorem integer_solutions_of_xyz_equation (x y z : ℤ) :
    x + y + z = x * y * z ↔ 
    (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
    (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) ∨ 
    (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ 
    ∃ a : ℤ, (x = -a ∧ y = 0 ∧ z = a) := by
  sorry


end integer_solutions_of_xyz_equation_l96_96321


namespace integral_equation_solution_l96_96180

theorem integral_equation_solution (f : ℝ → ℝ) (λ : ℝ) (hλ : λ < 1 / 2) :
  (∀ x, φ(x) = f(x) + λ * ∫ t in set.univ, real.exp (- abs (x - t)) * φ(t)) →
  (φ(x) = real.exp (- abs x) / real.sqrt (1 - 2 * λ)) :=
by
  assume hφ : ∀ x, φ(x) = f(x) + λ * ∫ t in set.univ, real.exp (- abs (x - t)) * φ(t)
  have hf : f = λ x, real.exp (- abs x), sorry
  sorry

end integral_equation_solution_l96_96180


namespace problem_l96_96142

noncomputable def f (x a b : ℝ) := x^2 + a*x + b
noncomputable def g (x c d : ℝ) := x^2 + c*x + d

theorem problem (a b c d : ℝ) (h_min_f : f (-a/2) a b = -25) (h_min_g : g (-c/2) c d = -25)
  (h_intersection_f : f 50 a b = -50) (h_intersection_g : g 50 c d = -50)
  (h_root_f_of_g : g (-a/2) c d = 0) (h_root_g_of_f : f (-c/2) a b = 0) :
  a + c = -200 := by
  sorry

end problem_l96_96142


namespace ternary_to_decimal_l96_96666

theorem ternary_to_decimal (n : ℕ) (h : n = 121) : 
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 16 :=
by sorry

end ternary_to_decimal_l96_96666


namespace grandmother_dolls_l96_96174

-- Define the conditions
variable (S G : ℕ)

-- Rene has three times as many dolls as her sister
def rene_dolls : ℕ := 3 * S

-- The sister has two more dolls than their grandmother
def sister_dolls_eq : Prop := S = G + 2

-- Together they have a total of 258 dolls
def total_dolls : Prop := (rene_dolls S) + S + G = 258

-- Prove that the grandmother has 50 dolls given the conditions
theorem grandmother_dolls : sister_dolls_eq S G → total_dolls S G → G = 50 :=
by
  intros h1 h2
  sorry

end grandmother_dolls_l96_96174


namespace problem_l96_96953

variables {n : ℕ} (a : fin n.succ → ℝ) (hpos : ∀ i, 0 < a i)

theorem problem (b : fin n.succ → ℝ) :
  (∀ i, b i ≥ a i) →
  (∀ i j, b i ≥ b j → ∃ k : ℕ, b i = b j * (2 ^ k)) →
  ∃ (b : fin n.succ → ℝ), 
    (∀ i, b i ≥ a i) ∧
    (∀ i j, b i ≥ b j → ∃ k : ℕ, b i = b j * (2 ^ k)) ∧
    (finset.univ.prod b ≤ 2 ^ ((n.succ - 1) / 2) * finset.univ.prod a) :=
sorry

end problem_l96_96953


namespace sum_of_first_five_terms_l96_96061

def a (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem sum_of_first_five_terms :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 5 / 6 := 
by 
  unfold a
  -- sorry is used as a placeholder for the actual proof
  sorry

end sum_of_first_five_terms_l96_96061


namespace min_sum_fraction_sqrt_l96_96330

open Real

theorem min_sum_fraction_sqrt (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ min, min = sqrt 2 ∧ ∀ z, (z = (x / sqrt (1 - x) + y / sqrt (1 - y))) → z ≥ sqrt 2 :=
sorry

end min_sum_fraction_sqrt_l96_96330


namespace reduced_price_tickets_first_week_l96_96315

theorem reduced_price_tickets_first_week (total_tickets sold_at_full_price : ℕ) 
  (condition1 : total_tickets = 25200) 
  (condition2 : sold_at_full_price = 16500)
  (condition3 : ∃ R, total_tickets = R + 5 * R) : 
  ∃ R : ℕ, R = 3300 := 
by sorry

end reduced_price_tickets_first_week_l96_96315


namespace hyperbola_asymptote_angle_l96_96075

theorem hyperbola_asymptote_angle 
  {x y : ℝ} (h : x - y^2 = -1) : 
  angle_of_asymptotes x y h = π / 3 :=
sorry

end hyperbola_asymptote_angle_l96_96075


namespace find_length_AX_l96_96437

-- Conditions
variables (A B C X : Type)
variables (triangle_ABC : Triangle A B C)
variable (angle_bisector_of_C_intersects_AB_at_X : Intersects (AngleBisector C A B) A B X)
variables (AB_eq_80 : SegmentLength A B = 80)
variables (AC_eq_42 : SegmentLength A C = 42)
variables (BC_eq_28 : SegmentLength B C = 28)

-- Goal
theorem find_length_AX : SegmentLength A X = 48 :=
sorry

end find_length_AX_l96_96437


namespace range_of_a_l96_96348

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x
noncomputable def g (x a : ℝ) : ℝ := -(x - 1)^2 + a^2

theorem range_of_a (a : ℝ) (x x1 x2 : ℝ) (hx : x > 0) (hx1x2 : f x2 ≤ g x1 a) : 
  a ≤ -Real.sqrt (Real.exp 1) ∨ a ≥ Real.sqrt (Real.exp 1) :=
begin
  sorry
end

end range_of_a_l96_96348


namespace ball_total_attendance_l96_96844

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l96_96844


namespace find_expression_l96_96093

theorem find_expression (k : ℕ) (h : 2^k = 4) : k = 2 ∧ (∃ e : ℕ, 2^e = 64 ∧ e = 6) :=
by {
  have hk : k = 2 := sorry,
  use 6,
  split,
  {
    exact hk,
  },
  { 
    exact by {
      use 6,
      split,
      { exact pow_succ' 2 5, },
      { rfl, }
    }
  }
}

end find_expression_l96_96093


namespace Kelly_initial_games_l96_96836

-- Condition definitions
variable (give_away : ℕ) (left_over : ℕ)
variable (initial_games : ℕ)

-- Given conditions
axiom h1 : give_away = 15
axiom h2 : left_over = 35

-- Proof statement
theorem Kelly_initial_games : initial_games = give_away + left_over :=
sorry

end Kelly_initial_games_l96_96836


namespace EulerLine_concurrency_or_parallel_l96_96799

variables {A B C A₁ B₁ C₁ P Q : Point}
variables (ABC : Triangle A B C)
variables (hAcute : IsAcuteTriangle ABC)
variables (hFeet : 
  (AltitudeFeet A₁ B₁ C₁ ABC ∧ 
   CircumcircleIntersects ABC (Triangle AB₁C₁) P A ∧ 
   CircumcircleIntersects ABC (Triangle BC₁A₁) Q B))

theorem EulerLine_concurrency_or_parallel (hConcurrent: 
  ConcurrentLines (Line A Q) (Line B P) (EulerLine ABC)) 
  (hParallel: 
  ParallelLines (Line A Q) (Line B P) (EulerLine ABC)) : 
  hConcurrent ∨ hParallel :=
sorry

end EulerLine_concurrency_or_parallel_l96_96799


namespace find_x2_plus_y2_l96_96414

theorem find_x2_plus_y2
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := 
by
  sorry

end find_x2_plus_y2_l96_96414


namespace quadratic_root_property_l96_96076

theorem quadratic_root_property (m p : ℝ) 
  (h1 : (p^2 - 2 * p + m - 1 = 0)) 
  (h2 : (p^2 - 2 * p + 3) * (m + 4) = 7)
  (h3 : ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 - 2 * r1 + m - 1 = 0 ∧ r2^2 - 2 * r2 + m - 1 = 0) : 
  m = -3 :=
by 
  sorry

end quadratic_root_property_l96_96076


namespace volume_of_pyramid_l96_96512

-- Define the conditions
def isosceles_right_triangle_area (leg : ℝ) : ℝ := 
  (1/2) * leg * leg

def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  (1/3) * base_area * height

-- Define the theorem we want to prove
theorem volume_of_pyramid (pyramid_leg_length : ℝ) (pyramid_height : ℝ) :
  pyramid_leg_length = 3 ∧ pyramid_height = 4 →
  pyramid_volume (isosceles_right_triangle_area pyramid_leg_length) pyramid_height = 6 := 
by 
  intros h
  cases h with h_leg h_height
  -- Placeholders (sorry) for the omitted proof steps
  sorry

end volume_of_pyramid_l96_96512


namespace point_on_circle_l96_96350

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def circle_radius := 5

def A : Point := {x := 2, y := -3}
def M : Point := {x := 5, y := -7}

theorem point_on_circle :
  distance A.x A.y M.x M.y = circle_radius :=
by
  sorry

end point_on_circle_l96_96350


namespace Misha_probability_l96_96594

open Probability

-- Definitions
def classesMonday := 5
def classesTuesday := 6
def totalClasses := 11
def totalCorrect := 7
def mondayCorrect := 3

-- Calculating binomial probabilities
noncomputable def P_A1 := (binomial classesMonday mondayCorrect) * (1 / 2) ^ classesMonday
noncomputable def P_A2 := (binomial classesTuesday (totalCorrect - mondayCorrect)) * (1 / 2) ^ classesTuesday
noncomputable def P_B := (binomial totalClasses totalCorrect) * (1 / 2) ^ totalClasses

-- Theorem stating the required probability
theorem Misha_probability :
  P_A1 * P_A2 / P_B = 5 / 11 :=
by
  sorry

end Misha_probability_l96_96594


namespace distinct_ordered_pairs_count_l96_96398

theorem distinct_ordered_pairs_count :
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ pairs → m > 0 ∧ n > 0 ∧ (1 / m.toRatl + 1 / n.toRatl) = 1 / 5) ∧
    (pairs.length = 3) := 
begin
  let pairs := [(6,30), (10,10), (30,6)],
  use pairs,
  split,
  { intros m n h,
    cases h; 
    simp [h],
    repeat { split; norm_num, linarith }
  },
  all_goals { norm_num }
end

end distinct_ordered_pairs_count_l96_96398


namespace four_digit_numbers_equiv_980_l96_96086

theorem four_digit_numbers_equiv_980 :
  (∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ a % 2 = 1 ∧
                     0 ≤ b ∧ b ≤ 9 ∧ b % 2 = 0 ∧
                     0 ≤ c ∧ c ≤ 9 ∧
                     0 ≤ d ∧ d ≤ 9 ∧
                     a ≠ b ∧ a ≠ c ∧ a ≠ d ∧
                     b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
                     (a + b + c + d) % 5 = 0 ∧
                     1000 * a + 100 * b + 10 * c + d) = 980 :=
sorry

end four_digit_numbers_equiv_980_l96_96086


namespace circumcenter_distance_to_line_l96_96344

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 1⟩
def B : Point := ⟨-real.sqrt 3, 0⟩
def C : Point := ⟨-real.sqrt 3, 2⟩

def line (m b : ℝ) (p : Point) : Prop := p.y = m * p.x + b

def is_circumcenter (O : Point) (A B C : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

theorem circumcenter_distance_to_line :
  ∃ O : Point, is_circumcenter O A B C ∧
  dist_to_line O (-real.sqrt 3) 0 = 1 / 2 :=
by
  sorry

end circumcenter_distance_to_line_l96_96344


namespace hannah_age_l96_96083

theorem hannah_age :
  let brothers_ages := [8, 10, 12, 14, 16]
  let sum_of_ages := brothers_ages.sum
  2.5 * sum_of_ages = 150 :=
by
  let brothers_ages := [8, 10, 12, 14, 16]
  let sum_of_ages := brothers_ages.sum
  have h_sum : sum_of_ages = 60 := by
    simp [sum_of_ages, brothers_ages]
  have h_hannah : 2.5 * sum_of_ages = 150 := by
    rw [h_sum]
    norm_num
  exact h_hannah

end hannah_age_l96_96083


namespace count_valid_x_eq_234_l96_96397

theorem count_valid_x_eq_234 :
  {x : ℕ | 100 ≤ x ∧ x < 334}.card = 234 := sorry

end count_valid_x_eq_234_l96_96397


namespace solution_set_of_inequality_eq_l96_96525

noncomputable def inequality_solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem solution_set_of_inequality_eq :
  {x : ℝ | (2 * x) / (x - 1) < 1} = inequality_solution_set := by
  sorry

end solution_set_of_inequality_eq_l96_96525


namespace investment_value_in_june_l96_96428

theorem investment_value_in_june (
    i: ℚ,
    h_jan: i * (1 + 3 / 4) = 7 / 4,
    h_feb: (7 / 4) * (1 - 1 / 4) = 21 / 16,
    h_mar: (21 / 16) * (1 + 1 / 3) = 7 / 4,
    h_apr: (7 / 4) * (1 - 1 / 5) = 7 / 5,
    h_may: (7 / 5) * (1 + 1 / 7) = 8 / 5,
    h_june: 1 * (1 - (3 / 8)) = 1
) : 3 + 8 = 11 :=
by {
    sorry
}

end investment_value_in_june_l96_96428


namespace sum_of_areas_l96_96088

theorem sum_of_areas (l₁ l₂ : ℕ) (h₁ : l₁ = 11) (h₂ : l₂ = 5) : l₁ * l₁ + l₂ * l₂ = 146 := by
  rw [h₁, h₂]
  norm_num
  sorry

end sum_of_areas_l96_96088


namespace moles_O2_combined_l96_96689

-- Define the quantities of substances involved
def moles_C2H6 : ℝ := 1
def moles_O2 : ℝ := 1 / 2
def moles_C2H4O : ℝ := 1

-- Define the balanced chemical reaction for verification
def balanced_reaction (c2h6 o2 c2h4o : ℝ) :=
  c2h6 = 1 ∧ o2 = 1 / 2 → c2h4o = 1

-- Theorem statement
theorem moles_O2_combined :
  balanced_reaction moles_C2H6 moles_O2 moles_C2H4O :=
by
  sorry

end moles_O2_combined_l96_96689


namespace find_n_satisfying_conditions_l96_96015

noncomputable def exists_set_satisfying_conditions (n : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧
  (∀ x ∈ S, x < 2^(n-1)) ∧
  ∀ A B : Finset ℕ, A ⊆ S → B ⊆ S → A ≠ B → A ≠ ∅ → B ≠ ∅ → A.sum id ≠ B.sum id

theorem find_n_satisfying_conditions : ∀ n : ℕ, (n ≥ 4) ↔ exists_set_satisfying_conditions n :=
sorry

end find_n_satisfying_conditions_l96_96015


namespace triangle_ratio_l96_96448

/-- Problem statement -/
theorem triangle_ratio
  {A B C D E : Type}
  {angle_A : ℝ} {angle_B : ℝ} {angle_ADE : ℝ}
  {area_ADE_area_ABC_ratio : ℝ}
  (hA : angle_A = 60)
  (hB : angle_B = 45)
  (hADE : angle_ADE = 75)
  (h_area_ratio : area_ADE_area_ABC_ratio = 1 / 3) :
  AD / AB = 1 / 3 :=
sorry

end triangle_ratio_l96_96448


namespace ball_attendance_l96_96865

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l96_96865


namespace john_extra_hours_l96_96828

theorem john_extra_hours (daily_earnings : ℕ) (hours_worked : ℕ) (bonus : ℕ) (hourly_wage : ℕ) (total_earnings_with_bonus : ℕ) (total_hours_with_bonus : ℕ) : 
  daily_earnings = 80 ∧ 
  hours_worked = 8 ∧ 
  bonus = 20 ∧ 
  hourly_wage = 10 ∧ 
  total_earnings_with_bonus = daily_earnings + bonus ∧
  total_hours_with_bonus = total_earnings_with_bonus / hourly_wage → 
  total_hours_with_bonus - hours_worked = 2 := 
by 
  sorry

end john_extra_hours_l96_96828


namespace milk_quality_check_l96_96182

/-
Suppose there is a collection of 850 bags of milk numbered from 001 to 850. 
From this collection, 50 bags are randomly selected for testing by reading numbers 
from a random number table. Starting from the 3rd line and the 1st group of numbers, 
continuing to the right, we need to find the next 4 bag numbers after the sequence 
614, 593, 379, 242.
-/

def random_numbers : List Nat := [
  78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279,
  43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820,
  61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636,
  63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421,
  42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983
]

noncomputable def next_valid_numbers (nums : List Nat) (start_idx : Nat) : List Nat :=
  nums.drop start_idx |>.filter (λ n => n ≤ 850) |>.take 4

theorem milk_quality_check :
  next_valid_numbers random_numbers 18 = [203, 722, 104, 88] :=
sorry

end milk_quality_check_l96_96182


namespace complex_pure_imaginary_solution_l96_96098

theorem complex_pure_imaginary_solution (m : ℝ) 
  (h_real_part : m^2 + 2*m - 3 = 0) 
  (h_imaginary_part : m - 1 ≠ 0) : 
  m = -3 :=
sorry

end complex_pure_imaginary_solution_l96_96098


namespace min_squares_for_symmetry_l96_96809

theorem min_squares_for_symmetry :
  let initial_shaded := [(0, 5), (2, 3), (3, 2), (5, 0)],
      grid_size := 6
  in  ∃ additional_shaded : Nat,
        (additional_shaded = 1) ∧
        (∀ x y, 
          (x < grid_size) ∧ (y < grid_size) →
          ((x, y) ∈ initial_shaded ∨
           (x, y) ∈ additional_shaded ∨
           (grid_size - 1 - x, y) ∈ initial_shaded ∨
           (x, grid_size - 1 - y) ∈ initial_shaded ∨
           (grid_size - 1 - x, grid_size - 1 - y) ∈ initial_shaded)) :=
sorry

end min_squares_for_symmetry_l96_96809


namespace problem_622_l96_96300

noncomputable def sequence (x : ℕ → ℝ) (h : ∀ n, x n > 0) : Prop :=
  x 1 = 1 ∧ (∀ n < 11, x (n+2) = (x (n+1) + 1) * (x (n+1) - 1) / x n) ∧ x 12 = 0

theorem problem_622 : ∃ a b c : ℕ, a > b ∧ ¬(∃ k : ℕ, k^2 ∣ a) ∧ ¬(∃ k : ℕ, k^2 ∣ b) ∧ 
  let x₂ := (√(a:ℝ) + √(b:ℝ)) / (c:ℝ) in
  x₂ = 1.8027756 ∧ 100 * a + 10 * b + c = 622 :=
by sorry

end problem_622_l96_96300


namespace probability_correct_predictions_monday_l96_96591

def number_of_classes_monday : ℕ := 5
def number_of_classes_tuesday : ℕ := 6
def total_classes : ℕ := number_of_classes_monday + number_of_classes_tuesday

def total_correct_predictions : ℕ := 7
def correct_predictions_monday : ℕ := 3
def correct_predictions_tuesday : ℕ := total_correct_predictions - correct_predictions_monday

noncomputable def binomial_coefficient (n k : ℕ) : ℝ := (nat.choose n k : ℝ)

noncomputable def probability_exact_correct_monday (n k : ℕ) : ℝ :=
  binomial_coefficient n k * (real.exp 11 (ln 2⁻¹ * total_classes))

theorem probability_correct_predictions_monday :
  probability_exact_correct_monday number_of_classes_monday correct_predictions_monday *
  probability_exact_correct_monday number_of_classes_tuesday correct_predictions_tuesday /
  probability_exact_correct_monday total_classes total_correct_predictions = 5 / 11 :=
sorry

end probability_correct_predictions_monday_l96_96591


namespace max_ab_is_5_l96_96376

noncomputable def max_ab : ℝ :=
  sorry

theorem max_ab_is_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h : a / 4 + b / 5 = 1) : max_ab = 5 :=
  sorry

end max_ab_is_5_l96_96376


namespace peg_board_unique_arrangement_l96_96302

-- Let's assume a type of colors for clarity
inductive Color
| yellow
| red
| green
| blue
| orange

-- The number of pegs for each color
def yellow_pegs := 6
def red_pegs := 5
def green_pegs := 4
def blue_pegs := 3
def orange_pegs := 2

noncomputable def unique_peg_arrangement : Prop :=
  ∃ arrangement : Fin yellow_pegs → Fin yellow_pegs → Option Color,
    (∀ i, (∃ c, ∀ j, arrangement i j = some c) ∧ (∀ j₁ j₂, j₁ ≠ j₂ → arrangement i j₁ ≠ arrangement i j₂)) ∧
    (∀ j, (∃ c, ∀ i, arrangement i j = some c) ∧ (∀ i₁ i₂, i₁ ≠ i₂ → arrangement i j ≠ arrangement i j₂)) ∧
    (∃! σ, ∀ (i : Fin yellow_pegs), arrangement i i = some σ)

theorem peg_board_unique_arrangement : unique_peg_arrangement := 
by sorry

end peg_board_unique_arrangement_l96_96302


namespace m_range_l96_96422

theorem m_range (m : ℤ) (h : ∀ x : ℕ, 1 ≤ x ∧ x ≤ 3 → 3 * x - m ≤ 0) : 9 ≤ m ∧ m < 12 :=
by
  have h1 : 3 * 1 - m ≤ 0 := h 1 ⟨le_refl _, le_of_eq rfl⟩
  have h2 : 3 * 2 - m ≤ 0 := h 2 ⟨le_of_eq rfl, le_of_eq rfl⟩
  have h3 : 3 * 3 - m ≤ 0 := h 3 ⟨le_of_eq rfl, le_refl _⟩
  sorry

end m_range_l96_96422


namespace trigonometric_expression_l96_96730

theorem trigonometric_expression (α : ℝ) (hα : α > π/3 ∧ α < π) (hcos : cos (2 * α) = 1 / 2) :
  (1 - tan α) / (1 + tan α) = 2 + sqrt 3 :=
sorry

end trigonometric_expression_l96_96730


namespace g_inv_sum_l96_96876

noncomputable def g : ℝ → ℝ :=
λ x, if x ≤ 2 then 1 - x else x^2 - 3 * x + 2

noncomputable def g_inv (y : ℝ) : ℝ :=
if y = 1 then 0
else if y = -1 then 2
else if y = 4 then 3
else 0  -- Placeholder to satisfy Lean's type requirements

theorem g_inv_sum :
  g_inv 1 + g_inv (-1) + g_inv 4 = 5 :=
by { dsimp [g_inv], norm_num, }

end g_inv_sum_l96_96876


namespace total_attended_ball_lt_fifty_l96_96854

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l96_96854


namespace find_k_inv_h_neg10_l96_96646

variable {X Y Z : Type}
variable (h : X → Y) (k : Z → X)
variable (h_inv : Y → X) (k_inv : X → Z)

noncomputable def k_inv_h_neg10_condition := 
  h_inv (k (-10)) = 3 * (-10) - 1

theorem find_k_inv_h_neg10 
  (h_inv_property : ∀ y, h_inv (h y) = y) 
  (k_inv_property : ∀ x, k_inv (k x) = x) 
  (condition : ∀ z, h_inv (k z) = 3 * z - 1)
  : k_inv (h (-10)) = -3 := 
  by {
    sorry
  }

end find_k_inv_h_neg10_l96_96646


namespace students_decrement_l96_96798

theorem students_decrement:
  ∃ d : ℕ, ∃ A : ℕ, 
  (∃ n1 n2 n3 n4 n5 : ℕ, n1 = A ∧ n2 = A - d ∧ n3 = A - 2 * d ∧ n4 = A - 3 * d ∧ n5 = A - 4 * d) ∧
  (5 = 5) ∧
  (n1 + n2 + n3 + n4 + n5 = 115) ∧
  (A = 27) → d = 2 :=
by {
  sorry
}

end students_decrement_l96_96798


namespace triangles_similar_iff_sqrt_eq_l96_96901

theorem triangles_similar_iff_sqrt_eq (a b c a1 b1 c1 : ℝ) :
  (a / a1 = b / b1 ∧ b / b1 = c / c1) ↔
  (sqrt (a * a1) + sqrt (b * b1) + sqrt (c * c1) = sqrt ((a + b + c) * (a1 + b1 + c1))) :=
by sorry

end triangles_similar_iff_sqrt_eq_l96_96901


namespace right_triangle_hypotenuse_l96_96219

theorem right_triangle_hypotenuse 
  (A B C X Y : Type) 
  [Real : Type]: 
  (h1: right_triangle A B C)
  (h2: on_leg X A B)
  (h3: on_leg Y A C)
  (h4: (dist A X) / (dist X B) = 1 / 2)
  (h5: (dist A Y) / (dist Y C) = 1 / 2)
  (h6: (dist B Y) = 16)
  (h7: (dist C X) = 28) 
  : dist B C = 6 * real.sqrt 26 := 
sorry

end right_triangle_hypotenuse_l96_96219


namespace part_one_prove_a_plus_b_eq_2c_part_two_find_min_cosC_l96_96108

noncomputable def proof_parts (A B C a b c : ℝ) :=
  ∆ABC (opposite a A) (opposite b B) (opposite c C) ∧ 
  2 * (tan A + tan B) = (tan A) / (cos B) + (tan B) / (cos A)

theorem part_one_prove_a_plus_b_eq_2c {A B C a b c : ℝ} 
(h : proof_parts A B C a b c) : 
  a + b = 2 * c := 
sorry

theorem part_two_find_min_cosC {A B C a b c : ℝ} 
(h : proof_parts A B C a b c) :
  ∃ min_ : ℝ, min_ = 1 / 2 ∧ 
  ∀ cosC ∈ {x : ℝ | x = cos C ∧ x ≤ 1}, cosC >= min_ := 
sorry

end part_one_prove_a_plus_b_eq_2c_part_two_find_min_cosC_l96_96108


namespace volume_of_cube_l96_96029

theorem volume_of_cube (d : ℝ) : volume (2d^3 * real.sqrt 2) :=
begin
  sorry
end

end volume_of_cube_l96_96029


namespace six_unique_squares_cannot_form_rectangle_l96_96900

theorem six_unique_squares_cannot_form_rectangle
  (a b c d e f : ℕ)
  (ha : a ≠ b) (hb : a ≠ c) (hc : a ≠ d) (hd : a ≠ e) (he : a ≠ f) (hf : b ≠ c)
  (hg : b ≠ d) (hh : b ≠ e) (hi : b ≠ f) (hj : c ≠ d) (hk : c ≠ e) (hl : c ≠ f)
  (hm : d ≠ e) (hn : d ≠ f) (ho : e ≠ f) :
  ¬ (∃ (x y : ℕ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e ∨ x = f) ∧
                  (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e ∨ y = f) ∧
                  (∃ (h1 h2 h3 h4 h5 h6 : (Σ' abcde, ℕ) ),
                   h1.2 = a ∧ h2.2 = b ∧ h3.2 = c ∧ h4.2 = d ∧ h5.2 = e ∧ h6.2 = f ∧
                   (h1.2 + h2.2 + h3.2 + h4.2 + h5.2 + h6.2 = x ∧
                    h1.2 + h2.2 + h3.2 + h4.2 + h5.2 + h6.2 = y ∧
                    ∀ (i j : ℕ) (hi : h1.fst = hi.1) (hj : h2.fst = hj.1), hi ≠ hj ) ∧
                   true ∧ true ∧ true  )) :=
sorry

end six_unique_squares_cannot_form_rectangle_l96_96900


namespace probability_of_same_parity_l96_96387

def parity (f : ℝ → ℝ) : Option String :=
  if ∀ x, f (-x) = f x then some "even"
  else if ∀ x, f (-x) = -f x then some "odd"
  else none

def f1 (x : ℝ) : ℝ := x^3 + 3 * x^2
def f2 (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2
def f3 (x : ℝ) : ℝ := Real.log2 (3 - x) - Real.log2 (3 + x)
def f4 (x : ℝ) : ℝ := x * Real.sin x

theorem probability_of_same_parity :
  let functions := [f1, f2, f3, f4]
  let parities := functions.map parity
  let even_functions := parities.countp (λ p => p = some "even")
  let odd_functions := parities.countp (λ p => p = some "odd")
  let number_of_pairs := Nat.choose 4 2
  let favorable_pairs := Nat.choose even_functions 2 + Nat.choose odd_functions 2
  (favorable_pairs : ℚ) / number_of_pairs = (1 / 6 : ℚ) := 
by
  sorry

end probability_of_same_parity_l96_96387


namespace projection_on_base_l96_96812

theorem projection_on_base (A B C A₁ B₁ C₁ H : Type) [InnerProductSpace ℝ A]
  (h₁ : angle A B C = 90)
  (h₂ : B ⟂ᵥ A C) :
  ∃ (H : A), H ∈ line_segment A B → H ≠ C :=
sorry

end projection_on_base_l96_96812


namespace parabola_equation_is_correct_point_C_coordinates_l96_96372
open real

-- Definitions based on conditions
def A := (1:ℝ, 0:ℝ)
def B := (0:ℝ, -3:ℝ)
def axis_of_symmetry : ℝ := 2
def parabola (a k : ℝ) (x : ℝ) : ℝ := a * (x - axis_of_symmetry)^2 + k

-- Theorem stating the equation of the parabola is as derived
theorem parabola_equation_is_correct :
  ∃ a k, 
    (parabola a k A.1 = A.2) ∧ 
    (parabola a k B.1 = B.2) ∧ 
    ∀ x, parabola (-1) 1 x = (-1:ℝ) * (x - axis_of_symmetry)^2 + 1 :=
begin
  -- Proof goes here
  sorry
end

-- Theorem stating the coordinates of point C
theorem point_C_coordinates :
  ∃ m1 m2,
    (m1 = 0 ∨ m1 = 4) ∧
    (parabola (-1) 1 m1 = -3) ∧
    (∃ C, C = (m1, -3) ∨ C = (m2, -3)) :=
begin
  -- Proof goes here
  sorry
end

end parabola_equation_is_correct_point_C_coordinates_l96_96372


namespace solve_for_y_l96_96495

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l96_96495


namespace height_of_cone_l96_96633

theorem height_of_cone (e : ℝ) (bA : ℝ) (v : ℝ) :
  e = 6 ∧ bA = 54 ∧ v = e^3 → ∃ h : ℝ, (1/3) * bA * h = v ∧ h = 12 := by
  sorry

end height_of_cone_l96_96633


namespace evaluate_expression_l96_96679

theorem evaluate_expression : 60 + (105 / 15) + (25 * 16) - 250 + (324 / 9) ^ 2 = 1513 := by
  sorry

end evaluate_expression_l96_96679


namespace curve_contains_four_collinear_points_l96_96045

theorem curve_contains_four_collinear_points (α : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y = x^4 + 9 * x^3 + α * x^2 + 9 * x + 4) →
  (∃ x1 x2 x3 x4 y : ℝ, 
    y = x1^4 + 9 * x1^3 + α * x1^2 + 9 * x1 + 4 ∧
    y = x2^4 + 9 * x2^3 + α * x2^2 + 9 * x2 + 4 ∧
    y = x3^4 + 9 * x3^3 + α * x3^2 + 9 * x3 + 4 ∧
    y = x4^4 + 9 * x4^3 + α * x4^2 + 9 * x4 + 4 ∧
    (x1, y) ≠ (x2, y) ∧ (x1, y) ≠ (x3, y) ∧ (x1, y) ≠ (x4, y) ∧
    (x2, y) ≠ (x3, y) ∧ (x2, y) ≠ (x4, y) ∧ (x3, y) ≠ (x4, y)) 
  → (α < 30.375) :=
begin
  sorry
end

end curve_contains_four_collinear_points_l96_96045


namespace find_k_value_l96_96738

theorem find_k_value (k : ℝ) :
  let α := (1, 2, -2)
  let β := (-2, -4, k)
  (∃ λ : ℝ, β = λ • α) → k = 4 :=
by
  intro h
  -- Emulate the conditions and argument steps for the proof.
  -- This proof would involve vector theory and scalar multiplication
  sorry

end find_k_value_l96_96738


namespace average_percentage_of_kernels_popped_l96_96164

theorem average_percentage_of_kernels_popped :
  let bag1_popped := 60
  let bag1_total := 75
  let bag2_popped := 42
  let bag2_total := 50
  let bag3_popped := 82
  let bag3_total := 100
  let percentage (popped total : ℕ) := (popped : ℚ) / total * 100
  let p1 := percentage bag1_popped bag1_total
  let p2 := percentage bag2_popped bag2_total
  let p3 := percentage bag3_popped bag3_total
  let avg := (p1 + p2 + p3) / 3
  avg = 82 :=
by
  sorry

end average_percentage_of_kernels_popped_l96_96164


namespace unique_real_solution_l96_96020

theorem unique_real_solution :
  ∀ x : ℝ, (∃ x : ℝ, (∛ (4 - x^3 / 2) = -2)) ↔ (x = ∛24) :=
by
  sorry

end unique_real_solution_l96_96020


namespace difference_in_dimes_l96_96175

variables (q : ℝ)

def samantha_quarters : ℝ := 3 * q + 2
def bob_quarters : ℝ := 2 * q + 8
def quarter_to_dimes : ℝ := 2.5

theorem difference_in_dimes :
  quarter_to_dimes * (samantha_quarters q - bob_quarters q) = 2.5 * q - 15 :=
by sorry

end difference_in_dimes_l96_96175


namespace number_of_valid_quadruples_l96_96309

-- Define the conditions where a, b, c, d are positive integers such that their product is 216
section
noncomputable def valid_quadruples : ℕ :=
  (finset.univ.filter (λ (abcd : ℕ × ℕ × ℕ × ℕ),
    let (a, b, c, d) := abcd in a * b * c * d = 216)).card

-- The statement to prove
theorem number_of_valid_quadruples : valid_quadruples = 400 :=
sorry
end

end number_of_valid_quadruples_l96_96309


namespace uncle_ben_eggs_l96_96547

def number_of_eggs (C R N E : ℕ) :=
  let hens := C - R
  let laying_hens := hens - N
  laying_hens * E

-- Given:
-- C = 440 (Total number of chickens)
-- R = 39 (Number of roosters)
-- N = 15 (Non-laying hens)
-- E = 3 (Eggs per laying hen)
theorem uncle_ben_eggs : number_of_eggs 440 39 15 3 = 1158 := by
  unfold number_of_eggs
  calc
    403 - 15 * 3 : sorry

end uncle_ben_eggs_l96_96547


namespace flour_in_first_combination_l96_96515

-- Define the variables involved in the problem
def cost_per_pound : ℝ := 0.45
def total_cost (sugar lbs_flour : ℝ) : ℝ := 26
def total_cost_eq (lbs_sugar lbs_flour cost_per_pound total_cost_per : ℝ) : Prop :=
  lbs_sugar * cost_per_pound + lbs_flour * cost_per_pound = total_cost_per

-- Stating the conditions in the problem
def condition1 : Prop := total_cost_eq 40 x cost_per_pound 26
def condition2 : Prop := total_cost_eq 30 25 cost_per_pound 26

-- Prove the amount of flour in the first combination
theorem flour_in_first_combination (condition1 condition2 : Prop) : x = 17.78 :=
  sorry

end flour_in_first_combination_l96_96515


namespace freshman_percent_l96_96647

noncomputable theory

variables {T : ℝ} -- Total number of students
variables {F : ℝ} -- Percent of freshmen (in decimal form)

-- Conditions
axiom condition1 : 0.60 * F * T = 0.60 * (F * T)
axiom condition2 : 0.50 * (0.60 * F) * T = 0.30 * F * T
axiom condition3 : 0.30 * F * T = 0.24 * T

-- Proof statement
theorem freshman_percent : F = 0.8 :=
by sorry

end freshman_percent_l96_96647


namespace misha_class_predictions_probability_l96_96586

-- Definitions representing the conditions
def monday_classes : ℕ := 5
def tuesday_classes : ℕ := 6
def total_classes : ℕ := monday_classes + tuesday_classes
def total_flips : ℕ := total_classes
def correct_predictions : ℕ := 7
def correct_monday_predictions : ℕ := 3

-- Lean theorem representing the proof problem
theorem misha_class_predictions_probability :
  (probability_of_correct_predictions total_flips correct_predictions monday_classes correct_monday_predictions) =
    (5 / 11) :=
  sorry

end misha_class_predictions_probability_l96_96586


namespace Malcom_cards_after_giving_away_half_l96_96294

def Brandon_cards : ℕ := 20
def Malcom_initial_cards : ℕ := Brandon_cards + 8
def Malcom_remaining_cards : ℕ := Malcom_initial_cards - (Malcom_initial_cards / 2)

theorem Malcom_cards_after_giving_away_half :
  Malcom_remaining_cards = 14 :=
by
  sorry

end Malcom_cards_after_giving_away_half_l96_96294


namespace simplify_tangent_expression_l96_96913

theorem simplify_tangent_expression :
  (1 + Real.tan (Real.pi / 18)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 :=
by
  sorry

end simplify_tangent_expression_l96_96913


namespace percentage_increase_twice_l96_96944

theorem percentage_increase_twice {P : ℝ} (x : ℝ) :
  (P * (1 + x)^2) = (P * (1 + 0.6900000000000001)) →
  x = 0.30 :=
by
  sorry

end percentage_increase_twice_l96_96944


namespace beth_novel_percentage_l96_96292

def total_books : ℕ := 120
def graphic_novels : ℕ := 18
def proportion_comic_books : ℚ := 0.20
def comic_books : ℕ := (proportion_comic_books * total_books).toNat
def novels : ℕ := total_books - (graphic_novels + comic_books)
def percentage_of_novels : ℚ := (novels / total_books) * 100

theorem beth_novel_percentage : percentage_of_novels = 65 := by
  sorry

end beth_novel_percentage_l96_96292


namespace max_and_min_of_f_l96_96604

def f (x : ℝ) := x^2 - 6 * x + 8

theorem max_and_min_of_f :
  ∀ x ∈ set.Icc (-1 : ℝ) 5, f x ≤ 15 ∧ -1 ≤ f x :=
by {
  -- define the relevant properties
  sorry -- proof is omitted
}

end max_and_min_of_f_l96_96604


namespace rectangle_area_is_588_l96_96609

-- Definitions based on the conditions of the problem
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- The statement to prove that the area of the rectangle is 588
theorem rectangle_area_is_588 : length * width = 588 :=
by
  -- Omitted proof
  sorry

end rectangle_area_is_588_l96_96609


namespace problem_solution_l96_96470

noncomputable def complex_expression (y : ℂ) :=
  (3 * y + y^2) * (3 * y^2 + y^4) * (3 * y^3 + y^6) * (3 * y^4 + y^8) * 
  (3 * y^5 + y^10) * (3 * y^6 + y^12) * (3 * y^7 + y^14) * (3 * y^8 + y^16)

noncomputable def simplified_expression (a b c d : ℂ) :=
  (10 + 3 * a) * (10 + 3 * b) * (10 + 3 * c) * (10 + 3 * d)

theorem problem_solution : 
  let y := complex.exp (2 * real.pi * complex.I / 9) in
  let a := y + y^8 in
  let b := y^2 + y^7 in
  let c := y^3 + y^6 in
  let d := y^4 + y^5 in
  y^9 = 1 → a + b + c + d = -1 →
  complex_expression y = simplified_expression a b c d := 
by
  intros y a b c d hyp1 hyp2
  sorry

end problem_solution_l96_96470


namespace quadratic_eq_exactly_one_real_solution_l96_96340

theorem quadratic_eq_exactly_one_real_solution :
  ∃ k : ℝ, (k = -10 + 2 * √21 ∨ k = -10 - 2 * √21) ∧
   ∀ x : ℝ, (3 * x + 8) * (x - 6) = -55 + k * x → (3 * x ^ 2 - (10 + k) * x + 7 = 0) ∧
   ∀ a b c : ℝ, a = 3 ∧ b = -(10 + k) ∧ c = 7 → (b^2 - 4 * a * c = 0) :=
sorry

end quadratic_eq_exactly_one_real_solution_l96_96340


namespace trumpington_marching_band_max_members_l96_96509

noncomputable def max_band_members : ℕ :=
  let m := 39 in 24 * m

theorem trumpington_marching_band_max_members
  (m : ℕ)
  (h1 : 24 * m % 30 = 6)
  (h2 : 24 * m < 1000) :
  24 * m = max_band_members :=
by
  have m_eq : m = 39 := sorry
  rw [m_eq]
  rfl

end trumpington_marching_band_max_members_l96_96509


namespace sum_binom_odd_coeff_alternating_signs_l96_96969

theorem sum_binom_odd_coeff_alternating_signs {S : ℤ} :
  S = ∑ k in Finset.range 51, (-1)^k * Nat.choose 100 (2*k + 1) → S = 0 :=
by
  sorry

end sum_binom_odd_coeff_alternating_signs_l96_96969


namespace max_unmarried_women_l96_96977

def num_people : ℕ := 100
def ratio_women : ℚ := 2 / 5
def ratio_married : ℚ := 1 / 4

theorem max_unmarried_women : 
  let num_women := (ratio_women * num_people) in
  let num_married := (ratio_married * num_people) in
  num_women = 40 → num_married = 25 → 
  ∃ max_unmarried_women, max_unmarried_women = 40 :=
by 
  intros h_women h_married
  use 40
  sorry

end max_unmarried_women_l96_96977


namespace club_suit_ratio_l96_96410

def club_suit (n m : ℕ) : ℕ := n^2 * m^3

theorem club_suit_ratio :
  (3 \clubsuit 5) / (5 \clubsuit 3) = 5 / 3 :=
by
  sorry

end club_suit_ratio_l96_96410


namespace units_digit_of_product_l96_96968

theorem units_digit_of_product (h1 : ∀ n : ℕ, (5^n % 10) = 5) : (5^11 * 2^3) % 10 = 0 :=
by
  -- Use the hypothesis to get the units digit of 5^11
  have u1: (5^11 % 10) = 5, from h1 11
  -- Calculate the units digit of 2^3
  have u2 : (2^3 % 10) = 8, by norm_num
  -- Units digit of the product
  show (5^11 * 2^3) % 10 = 0, from
    calc
      (5^11 * 2^3) % 10
      = (5 * 8) % 10       : by rw [u1, u2, pow_succ, pow_succ]
      = 40 % 10            : by norm_num
      = 0                  : by norm_num

end units_digit_of_product_l96_96968


namespace total_journey_distance_l96_96976

/-- 
A woman completes a journey in 5 hours. She travels the first half of the journey 
at 21 km/hr and the second half at 24 km/hr. Find the total journey in km.
-/
theorem total_journey_distance :
  ∃ D : ℝ, (D / 2) / 21 + (D / 2) / 24 = 5 ∧ D = 112 :=
by
  use 112
  -- Please prove the following statements
  sorry

end total_journey_distance_l96_96976


namespace total_training_hours_l96_96958

-- Define Thomas's training conditions
def hours_per_day := 5
def days_initial := 30
def days_additional := 12
def total_days := days_initial + days_additional

-- State the theorem to be proved
theorem total_training_hours : total_days * hours_per_day = 210 :=
by
  sorry

end total_training_hours_l96_96958


namespace range_of_m_l96_96785

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + 1 > 0) ↔ (-2 < m ∧ m < 2) :=
  sorry

end range_of_m_l96_96785


namespace num_black_balls_l96_96109

theorem num_black_balls 
  (R W B : ℕ) 
  (R_eq : R = 30) 
  (prob_white : (W : ℝ) / 100 = 0.47) 
  (total_balls : R + W + B = 100) : B = 23 := 
by 
  sorry

end num_black_balls_l96_96109


namespace problem1_l96_96720

noncomputable def ellipse_equation : String :=
  "The equation of the ellipse is \\frac{x^2}{4} + \\frac{y^2}{1} = 1"

noncomputable def value_of_m : Real :=
  17 / 8

theorem problem1 :
  (∃ (F1 F2 : Point), 
       F1.x = -sqrt(3) ∧ F1.y = 0 ∧ 
       F2.x = sqrt(3)  ∧ F2.y = 0 ∧
       (∃ (A B : Point), 
           A.x = 0 ∧ A.y = 1 ∧ 
           B.x = 0 ∧ B.y = -1 ∧ 
           dist A F2 = dist B F2)
   → ellipse_equation = "The equation of the ellipse is \\frac{x^2}{4} + \\frac{y^2}{1} = 1") ∧
  (∀ (l: Line) 
       (E : Point) 
       (m: Real), 
       E.x = m ∧ E.y = 0 ∧ 
       l.contains Point.mk 1 0 ∧
       (l.intersects (Ellipse.mk Point.mk (-sqrt(3)) (sqrt(3)))) → 
       (collision_points : List Point),
       collision_points.length = 2 →
       (p1 p2 : Point),
       p1 ∈ collision_points ∧
       p2 ∈ collision_points ∧
       is_constant (dot (vector_on_x E p1) (vector_on_x E p2)) ∧
       value_of_m = m) 
  sorry

end problem1_l96_96720


namespace alberto_bjorn_distance_diff_l96_96639

-- Definitions of conditions based on the given problem

def distance_alberto := 80
def distance_bjorn := 20 * 2 + 25 * 2
def distance_difference := distance_alberto - distance_bjorn

-- Statement to prove the required difference
theorem alberto_bjorn_distance_diff : distance_difference = -10 := by
  sorry

end alberto_bjorn_distance_diff_l96_96639


namespace max_digit_d_condition_l96_96014

theorem max_digit_d_condition :
  ∃ d e : ℕ, (d ≤ 9 ∧ 0 ≤ d) ∧ (e ≤ 9 ∧ 0 ≤ e) ∧ (e % 2 = 0) ∧ (707_340 + 10 * d + e) % 34 = 0 ∧ d = 13 :=
by {
  -- Proof will be provided here
  sorry
}

end max_digit_d_condition_l96_96014


namespace x_squared_plus_y_squared_geq_five_l96_96710

theorem x_squared_plus_y_squared_geq_five (x y : ℝ) (h : abs (x - 2 * y) = 5) : x^2 + y^2 ≥ 5 := 
sorry

end x_squared_plus_y_squared_geq_five_l96_96710


namespace max_tries_needed_to_open_lock_l96_96996

-- Definitions and conditions
def num_buttons : ℕ := 9
def sequence_length : ℕ := 4
def opposite_trigrams : ℕ := 2  -- assumption based on the problem's example
def total_combinations : ℕ := 3024

theorem max_tries_needed_to_open_lock :
  (total_combinations - (8 * 1 * 7 * 6 + 8 * 6 * 1 * 6 + 8 * 6 * 4 * 1)) = 2208 :=
by
  sorry

end max_tries_needed_to_open_lock_l96_96996


namespace volume_of_intersection_of_two_cylinders_l96_96336

theorem volume_of_intersection_of_two_cylinders (a : ℝ) (h_pos : 0 < a) :
  volume (intersection (cylinder a) (right_angle (cylinder a))) = (16 / 3) * a^3 :=
sorry

end volume_of_intersection_of_two_cylinders_l96_96336


namespace polynomial_coeff_diff_l96_96780

theorem polynomial_coeff_diff (a b c d e f : ℝ) :
  ((3*x + 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (a - b + c - d + e - f = 32) :=
by
  sorry

end polynomial_coeff_diff_l96_96780


namespace period_of_sine_function_l96_96966

theorem period_of_sine_function :
  ∀ x : ℝ, 3 * sin (5 * x - (π / 4)) = 3 * sin (5 * (x + (2 * π / 5)) - (π / 4)) :=
by
  sorry

end period_of_sine_function_l96_96966


namespace triangle_lengths_l96_96804

open Real

variables (A B C P Q O : Point)
variables (hBC : dist B C = 5)
variables (hO : midpoint B C = O)
variables (hO_radius : dist O P = 2 ∧ dist O Q = 2 ∧ dist OA = 2.5 ∧ dist OB = 2.5)

theorem triangle_lengths
  (hABC : right_triangle A B C)
  (hCircle : ∀ΠO, circle O 2)
  (hIntersect : intersects_circle O 2 (BC := [B, C]) P Q) : 
  dist A P ^ 2 + dist A Q ^ 2 + dist P Q ^ 2 = 73 / 2 :=
by
  sorry

end triangle_lengths_l96_96804


namespace number_of_values_l96_96339

def d (n : ℕ) : ℕ := (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).card

def g_1 (n : ℕ) : ℕ := 4 * d n

def g : ℕ → ℕ → ℕ
| 1     n := g_1 n
| (j+1) n := g_1 (g j n)

def count_valid_n (N : ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, g 50 n = 16) (Finset.range (N + 1)))

theorem number_of_values (N : ℕ) (hN : 100 ≤ N) : count_valid_n N = 27 :=
sorry

end number_of_values_l96_96339


namespace range_of_m_value_of_m_l96_96079

variables (m p x : ℝ)

-- Conditions: The quadratic equation x^2 - 2x + m - 1 = 0 must have two real roots.
def discriminant (m : ℝ) := (-2)^2 - 4 * 1 * (m - 1)

-- Part 1: Finding the range of values for m
theorem range_of_m (h : discriminant m ≥ 0) : m ≤ 2 := 
by sorry

-- Additional Condition: p is a real root of the equation x^2 - 2x + m - 1 = 0
def is_root (p m : ℝ) := p^2 - 2 * p + m - 1 = 0

-- Another condition: (p^2 - 2p + 3)(m + 4) = 7
def satisfies_condition (p m : ℝ) := (p^2 - 2 * p + 3) * (m + 4) = 7

-- Part 2: Finding the value of m given p is a real root and satisfies (p^2 - 2p + 3)(m + 4) = 7
theorem value_of_m (h1 : is_root p m) (h2 : satisfies_condition p m) : m = -3 := 
by sorry

end range_of_m_value_of_m_l96_96079


namespace ball_total_attendance_l96_96843

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l96_96843


namespace vertical_line_divides_triangle_l96_96721

noncomputable def triangle := {(0, 4), (0, 0), (4, 4)}

theorem vertical_line_divides_triangle :
  ∃ (a : ℝ), (a = 2) ∧ (∃ (triangle' : set (ℝ × ℝ)), triangle' ⊆ triangle ∧ measure_theory.measure_space.volume triangle' = 4) :=
sorry

end vertical_line_divides_triangle_l96_96721


namespace roots_cubic_identity_l96_96091

theorem roots_cubic_identity (r s : ℚ) (h1 : 3 * r^2 + 5 * r + 2 = 0) (h2 : 3 * s^2 + 5 * s + 2 = 0) :
  (1 / r^3) + (1 / s^3) = -27 / 35 :=
sorry

end roots_cubic_identity_l96_96091


namespace sum_of_three_integers_l96_96207

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 5^3) : a + b + c = 31 := by
  sorry

end sum_of_three_integers_l96_96207


namespace ball_total_attendance_l96_96848

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l96_96848


namespace find_m_find_other_root_l96_96351

variable (m : ℝ)
noncomputable def z : ℂ := complex.mk (m + 2) (m - 2)
noncomputable def z_conj : ℂ := complex.conj z

theorem find_m (h: z + z_conj = 6) : m = 1 :=
by
  sorry

variable (a b : ℝ)
noncomputable def quadratic_equation_root1 : ℂ := z - complex.mk 0 3
noncomputable def quadratic_equation_root2 : ℂ := 3 + complex.mk 0 4

theorem find_other_root (h1: z = 3 - complex.mk 0 1) (h2: polynomial.aeval z quadratic_equation = 0) : 
  polynomial.aeval (3 + complex.mk 0 4) quadratic_equation = 0 :=
by
  sorry

end find_m_find_other_root_l96_96351


namespace joan_already_put_in_cups_l96_96451

def recipe_cups : ℕ := 7
def cups_needed : ℕ := 4

theorem joan_already_put_in_cups : (recipe_cups - cups_needed = 3) :=
by
  sorry

end joan_already_put_in_cups_l96_96451


namespace find_pairs_l96_96322

noncomputable def solution_pairs : Set (ℝ × ℝ) :=
  {p | let x := p.1; let y := p.2 in
  (x ∈ (0, Real.pi / 2) ∧ y ∈ (0, Real.pi / 2)) ∧
  (cos x / cos y = 2 * cos y ^ 2) ∧
  (sin x / sin y = 2 * sin y ^ 2)}

theorem find_pairs : (Real.pi / 4, Real.pi / 4) ∈ solution_pairs :=
  sorry

end find_pairs_l96_96322


namespace geometric_seq_inequality_l96_96740

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := 
  ∀ n, a (n + 1) = q * a n 

theorem geometric_seq_inequality {a : ℕ → ℝ} {q : ℝ} 
  (h_geometric_seq : geometric_sequence a q) 
  (h_pos : ∀ n, 0 < a n) (h_q : 0 < q ∧ q < 1) :
  let P := (a 3 + a 9) / 2 in
  let Q := sqrt (a 5 * a 7) in
  a 9 < Q ∧ Q < P ∧ P < a 3 :=
  sorry

end geometric_seq_inequality_l96_96740


namespace max_value_frac_sixth_roots_eq_two_l96_96122

noncomputable def max_value_frac_sixth_roots (α β : ℝ) (t : ℝ) (q : ℝ) : ℝ :=
  if α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t then
    max (1 / α^6 + 1 / β^6) 2
  else
    0

theorem max_value_frac_sixth_roots_eq_two (α β : ℝ) (t : ℝ) (q : ℝ) :
  (α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t) →
  ∃ m, max_value_frac_sixth_roots α β t q = m ∧ m = 2 :=
sorry

end max_value_frac_sixth_roots_eq_two_l96_96122


namespace polynomial_max_bound_l96_96379

noncomputable def polynomial_bound (n : ℕ) (P : ℝ[X]) : Prop :=
  degree P ≤ 2 * n ∧ ∀ (k : ℤ), -n ≤ k ∧ k ≤ n → abs (P.eval (k : ℝ)) ≤ 1

theorem polynomial_max_bound (n : ℕ) (P : ℝ[X])
  (hdeg : degree P ≤ 2 * n)
  (hbound : ∀ (k : ℤ), -n ≤ k ∧ k ≤ n → abs (P.eval (k : ℝ)) ≤ 1) :
  ∀ (x : ℝ), -n ≤ x ∧ x ≤ n → abs (P.eval x) ≤ 2^(2 * n) :=
by {
  sorry,
}

end polynomial_max_bound_l96_96379


namespace total_spent_on_toys_l96_96282

-- Definition of the costs
def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

-- The theorem to prove the total amount spent on toys
theorem total_spent_on_toys : football_cost + marbles_cost = 12.30 :=
by sorry

end total_spent_on_toys_l96_96282


namespace concyclic_points_concyclic_points_bi_implication_cyclic_l96_96880

-- Define the circles and points
variable (Γ₁ Γ₂ Γ₃ Γ₄ : Type) -- Assuming these are some types representing the circles
variable (P₁ P₂ P₃ P₄ Q₁ Q₂ Q₃ Q₄ : Type) -- Assuming these are some types representing the points

-- Conditions given in the problem
variable [Intersects Γ₁ Γ₂ P₁ Q₁] -- Γ₁ intersects Γ₂ at points P₁, Q₁
variable [Intersects Γ₂ Γ₃ P₂ Q₂] -- Γ₂ intersects Γ₃ at points P₂, Q₂
variable [Intersects Γ₃ Γ₄ P₃ Q₃] -- Γ₃ intersects Γ₄ at points P₃, Q₃
variable [Intersects Γ₄ Γ₁ P₄ Q₄] -- Γ₄ intersects Γ₁ at points P₄, Q₄

-- Theorem statement
theorem concyclic_points (h₁ : Cyclic P₁ P₂ P₃ P₄) : Cyclic Q₁ Q₂ Q₃ Q₄ :=
sory

theorem concyclic_points' (h₂ : Cyclic Q₁ Q₂ Q₃ Q₄) : Cyclic P₁ P₂ P₃ P₄ :=
sory

theorem bi_implication_cyclic:
  (Cyclic P₁ P₂ P₃ P₄ ↔ Cyclic Q₁ Q₂ Q₃ Q₄) :=
  ⟨concyclic_points, concyclic_points'⟩


end concyclic_points_concyclic_points_bi_implication_cyclic_l96_96880


namespace students_wearing_specific_shirt_and_accessory_count_l96_96797

theorem students_wearing_specific_shirt_and_accessory_count :
  let total_students := 1000
  let blue_shirt_percent := 0.40
  let red_shirt_percent := 0.25
  let green_shirt_percent := 0.20
  let blue_shirt_students := blue_shirt_percent * total_students
  let red_shirt_students := red_shirt_percent * total_students
  let green_shirt_students := green_shirt_percent * total_students
  let blue_shirt_stripes_percent := 0.30
  let blue_shirt_polka_dots_percent := 0.35
  let red_shirt_stripes_percent := 0.20
  let red_shirt_polka_dots_percent := 0.40
  let green_shirt_stripes_percent := 0.25
  let green_shirt_polka_dots_percent := 0.25
  let accessory_hat_percent := 0.15
  let accessory_scarf_percent := 0.10
  let red_polka_dot_students := red_shirt_polka_dots_percent * red_shirt_students
  let red_polka_dot_hat_students := accessory_hat_percent * red_polka_dot_students
  let green_no_pattern_students := green_shirt_students - (green_shirt_stripes_percent * green_shirt_students + green_shirt_polka_dots_percent * green_shirt_students)
  let green_no_pattern_scarf_students := accessory_scarf_percent * green_no_pattern_students
  red_polka_dot_hat_students + green_no_pattern_scarf_students = 25 := by
    sorry

end students_wearing_specific_shirt_and_accessory_count_l96_96797


namespace remainder_5161_div_101_l96_96686

theorem remainder_5161_div_101:
  let G := 101
  let R2 := 10
  (4351 % G = 8) → (5161 % G = R2) :=
by
  intros h1,
  let k := 4351 / G,
  have h2 : 4351 = G * k + 8, 
    from nat.mod_add_div 4351 G ▸ (eq_add_of_sub_eq h1.symm),
  let m := 5161 / G,
  have h3 : 5161 = G * m + 5161 % G, 
    from nat.mod_add_div 5161 G,
  have h4 : 101 * 51 = 5151,
    from by ring,
  have h5 : 5161 - 101 * 51 = 10,
    from by norm_num,
  have : 5161 % 101 = 10,
    by {rw ← h4, rw h5},
  exact this

end remainder_5161_div_101_l96_96686


namespace solve_inequality_when_a_is_one_range_of_values_for_a_l96_96385

open Real

-- Part (1) Statement
theorem solve_inequality_when_a_is_one (a x : ℝ) (h : a = 1) : 
  |x - a| + |x + 2| ≤ 5 → -3 ≤ x ∧ x ≤ 2 := 
by sorry

-- Part (2) Statement
theorem range_of_values_for_a (a : ℝ) : 
  (∃ x_0 : ℝ, |x_0 - a| + |x_0 + 2| ≤ |2 * a + 1|) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by sorry

end solve_inequality_when_a_is_one_range_of_values_for_a_l96_96385


namespace difference_in_elevation_difference_in_running_time_l96_96084

structure Day :=
  (distance_km : ℝ) -- kilometers
  (pace_min_per_km : ℝ) -- minutes per kilometer
  (elevation_gain_m : ℝ) -- meters

def monday : Day := { distance_km := 9, pace_min_per_km := 6, elevation_gain_m := 300 }
def wednesday : Day := { distance_km := 4.816, pace_min_per_km := 5.5, elevation_gain_m := 150 }
def friday : Day := { distance_km := 2.095, pace_min_per_km := 7, elevation_gain_m := 50 }

noncomputable def calculate_running_time(day : Day) : ℝ :=
  day.distance_km * day.pace_min_per_km

noncomputable def total_elevation_gain(wednesday friday : Day) : ℝ :=
  wednesday.elevation_gain_m + friday.elevation_gain_m

noncomputable def total_running_time(wednesday friday : Day) : ℝ :=
  calculate_running_time wednesday + calculate_running_time friday

theorem difference_in_elevation :
  monday.elevation_gain_m - total_elevation_gain wednesday friday = 100 := by 
  sorry

theorem difference_in_running_time :
  calculate_running_time monday - total_running_time wednesday friday = 12.847 := by 
  sorry

end difference_in_elevation_difference_in_running_time_l96_96084


namespace find_sum_of_m1_m2_l96_96140

-- Define the quadratic equation and the conditions
def quadratic (m : ℂ) (x : ℂ) : ℂ := m * x^2 - (3 * m - 2) * x + 7

-- Define the roots a and b
def are_roots (m a b : ℂ) : Prop := quadratic m a = 0 ∧ quadratic m b = 0

-- The condition given in the problem
def root_condition (a b : ℂ) : Prop := a / b + b / a = 3 / 2

-- Main theorem to be proved
theorem find_sum_of_m1_m2 (m1 m2 a1 a2 b1 b2 : ℂ) 
  (h1 : are_roots m1 a1 b1) 
  (h2 : are_roots m2 a2 b2) 
  (hc1 : root_condition a1 b1) 
  (hc2 : root_condition a2 b2) : 
  m1 + m2 = 73 / 18 :=
by sorry

end find_sum_of_m1_m2_l96_96140


namespace pedro_more_squares_l96_96896

theorem pedro_more_squares 
  (squares_jesus : ℕ)
  (multiplier_jesus : ℕ)
  (squares_linden : ℕ)
  (multiplier_linden : ℕ)
  (squares_pedro : ℕ)
  (multiplier_pedro : ℕ)
  (h_jesus : squares_jesus = 60)
  (h_mult_jesus : multiplier_jesus = 2)
  (h_linden : squares_linden = 75)
  (h_mult_linden : multiplier_linden = 3)
  (h_pedro : squares_pedro = 200)
  (h_mult_pedro : multiplier_pedro = 4) :
  (squares_pedro * multiplier_pedro) - ((squares_jesus * multiplier_jesus) + (squares_linden * multiplier_linden)) = 455 :=
by
  -- Carry forward proof obligations
  rw [h_jesus, h_mult_jesus, h_linden, h_mult_linden, h_pedro, h_mult_pedro]
  calc
    200 * 4 - (60 * 2 + 75 * 3) = 800 - (120 + 225) : by refl
    ... = 800 - 345 : by refl
    ... = 455 : by refl

end pedro_more_squares_l96_96896


namespace machine_A_sprockets_per_hour_l96_96476

theorem machine_A_sprockets_per_hour :
  ∀ (A T : ℝ),
    (T > 0 ∧
    (∀ P Q, P = 1.1 * A ∧ Q = 330 / P ∧ Q = 330 / A + 10) →
      A = 3) := 
by
  intro A T
  intro h
  sorry

end machine_A_sprockets_per_hour_l96_96476


namespace avg_of_remaining_two_l96_96238

variables {x1 x2 x3 x4 x5 x6 : ℝ}

-- Definitions based on the conditions
def avg_of_six : Prop := (x1 + x2 + x3 + x4 + x5 + x6) / 6 = 5.40
def avg_of_first_two : Prop := (x1 + x2) / 2 = 5.2
def avg_of_second_two : Prop := (x3 + x4) / 2 = 5.8

-- The statement to prove
theorem avg_of_remaining_two (h1 : avg_of_six) (h2 : avg_of_first_two) (h3 : avg_of_second_two) :
  (x5 + x6) / 2 = 5.20 :=
sorry

end avg_of_remaining_two_l96_96238


namespace arithmetic_geometric_mean_identity_l96_96488

theorem arithmetic_geometric_mean_identity (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 96) : x^2 + y^2 = 1408 :=
by
  sorry

end arithmetic_geometric_mean_identity_l96_96488


namespace problem_704_l96_96312

noncomputable def rounding_probability_sum_to_100 (a b c : ℝ) [fact (a > 0)] [fact (b > 0)] [fact (c > 0)] : ℚ :=
if h : a + b + c = 100 then
  let fractions (x : ℝ) : ℝ := if abs(x - round x) = 0.5 then round (x * 2) / 2 else round x
  let rounded_sum := fractions a + fractions b + fractions c
  if rounded_sum = 100 then 7 / 4 else 0
else 0

theorem problem_704 (m n : ℕ) (hmrel : nat.coprime m n) (hprob : rounding_probability_sum_to_100 100 = ↑m / ↑n) :
  100 * m + n = 704 :=
by sorry

end problem_704_l96_96312


namespace distance_between_intersections_l96_96999

-- Definitions of the hyperbola and the parabola
def hyperbola (x y : ℝ) := x^2 / 16 - y^2 / 9 = 1
def parabola (x y : ℝ) := x = y^2 / 10 + 5 / 2

-- Define a function that calculates the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Distance between the two intersection points of the hyperbola and parabola
theorem distance_between_intersections :
  ∃ p1 p2 : ℝ × ℝ, hyperbola p1.1 p1.2 ∧ parabola p1.1 p1.2 ∧
                  hyperbola p2.1 p2.2 ∧ parabola p2.1 p2.2 ∧
                  p1 ≠ p2 ∧
                  distance p1 p2 = 10 := sorry

end distance_between_intersections_l96_96999


namespace minimum_sum_at_nine_l96_96057

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem minimum_sum_at_nine {a1 d : ℤ} (h_a1_neg : a1 < 0) 
    (h_sum_equal : sum_of_arithmetic_sequence a1 d 12 = sum_of_arithmetic_sequence a1 d 6) :
  ∀ n : ℕ, (n = 9) → sum_of_arithmetic_sequence a1 d n ≤ sum_of_arithmetic_sequence a1 d m :=
sorry

end minimum_sum_at_nine_l96_96057


namespace avg_weight_difference_l96_96579

-- Define the weights of the boxes following the given conditions.
def box1_weight : ℕ := 200
def box3_weight : ℕ := box1_weight + (25 * box1_weight / 100)
def box2_weight : ℕ := box3_weight + (20 * box3_weight / 100)
def box4_weight : ℕ := 350
def box5_weight : ℕ := box4_weight * 100 / 70

-- Define the average weight of the four heaviest boxes.
def avg_heaviest : ℕ := (box2_weight + box3_weight + box4_weight + box5_weight) / 4

-- Define the average weight of the four lightest boxes.
def avg_lightest : ℕ := (box1_weight + box2_weight + box3_weight + box4_weight) / 4

-- Define the difference between the average weights of the heaviest and lightest boxes.
def avg_difference : ℕ := avg_heaviest - avg_lightest

-- State the theorem with the expected result.
theorem avg_weight_difference : avg_difference = 75 :=
by
  -- Proof is not provided.
  sorry

end avg_weight_difference_l96_96579


namespace problem_I_problem_II_problem_III_l96_96752

def g (x a : ℝ) := x^2 - a * x + 6
def f (x : ℝ) := Real.logBase (1/2) (x^2 + 1)

theorem problem_I (a : ℝ) : (∀ x : ℝ, g x a = g (-x) a) → a = 0 :=
by
  intro h
  sorry

theorem problem_II (a : ℝ) (h : ∀ x, 2 < x ∧ x < 3 → g x a < 0) : 
  a = 5 → (∀ x, x > 1 → (g x a / (x - 1)) ≥ 2 * Real.sqrt 2 - 3) :=
by
  intro h
  sorry

theorem problem_III (a : ℝ) : (∀ x1 : ℝ, (1 ≤ x1) → (∀ x2 : ℝ, (-2 ≤ x2 ∧ x2 ≤ 4) → f x1 ≤ g x2 a)) → 
  -11 / 2 ≤ a ∧ a ≤ 2 * Real.sqrt 7 :=
by
  intro h
  sorry

end problem_I_problem_II_problem_III_l96_96752


namespace power_product_rule_l96_96651

theorem power_product_rule (a : ℤ) : (-a^2)^3 = -a^6 := 
by 
  sorry

end power_product_rule_l96_96651


namespace classroom_ratio_l96_96611

theorem classroom_ratio (length width : ℕ) (h₁ : length = 25) (h₂ : width = 15) : 3 * (2 * (length + width)) = 16 * width :=
by
  rw [h₁, h₂]
  simp
  -- Here the calculation steps would be assumed to be solved.
  sorry

end classroom_ratio_l96_96611


namespace math_equivalence_proof_l96_96242

noncomputable def problem_conditions : Prop :=
∀ (A B C D E : ℝ) (circle : ℝ),
  AB = 5 ∧ BC = 5 ∧ CD = 5 ∧ DE = 5 ∧ AE = 2

theorem math_equivalence_proof :
  problem_conditions → 
  (1 - real.cos angle_B) * (1 - real.cos angle_ACE) = (1/25) :=
by
  intro h
  sorry

end math_equivalence_proof_l96_96242


namespace speed_in_still_water_l96_96234

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 25) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 45 :=
by
  sorry

end speed_in_still_water_l96_96234


namespace total_books_is_595_l96_96042

-- Definitions of the conditions
def satisfies_conditions (a : ℕ) : Prop :=
  ∃ R L : ℕ, a = 12 * R + 7 ∧ a = 25 * L - 5 ∧ 500 < a ∧ a < 650

-- The theorem statement
theorem total_books_is_595 : ∃ a : ℕ, satisfies_conditions a ∧ a = 595 :=
by
  use 595
  split
  · apply exists.intro 49, exists.intro 24, split
    -- a = 12R + 7
    · exact rfl
    -- a = 25L - 5
    · exact rfl
  -- Next check 500 < a and a < 650
  split
  · exact nat.lt_of_le_of_lt (by norm_num) (by norm_num)
  · exact nat.lt_of_le_of_lt (by norm_num) (by norm_num)

end total_books_is_595_l96_96042


namespace olivers_score_l96_96479

theorem olivers_score (n : ℕ) (grade_avg24 : ℕ) (new_avg25 : ℕ) (grade24_sum grade25_sum : ℕ)
  (h1 : n = 25) 
  (h2 : grade_avg24 = 76)
  (h3 : new_avg25 = 78)
  (h4 : grade24_sum = 24 * grade_avg24)
  (h5 : grade25_sum = n * new_avg25)
  (h6 : grade25_sum = grade24_sum + 126) : 
  ∃ x, x = 126 :=
by
  use 126
  sorry

end olivers_score_l96_96479


namespace problem_l96_96317

-- Define the dimensions of the matrix
def n : ℕ := 1980
def m : ℕ := 1981

-- Define the matrix and its entries
variable (A : Fin n → Fin m → ℤ)

-- Define the conditions for the problem
def valid_entries : Prop :=
∀ i j, A i j ∈ { -1, 0, 1 }

def sum_zero : Prop :=
∑ i in Finset.finRange n, ∑ j in Finset.finRange m, A i j = 0

-- Define the theorem to be proven
theorem problem (h1 : valid_entries A) (h2 : sum_zero A) :
  ∃ i1 i2 : Fin n, ∃ j1 j2 : Fin m,
    i1 ≠ i2 ∧ j1 ≠ j2 ∧ 
    A i1 j1 + A i1 j2 + A i2 j1 + A i2 j2 = 0 := sorry

end problem_l96_96317


namespace length_of_chord_l96_96440

noncomputable def circle_center : ℝ × ℝ := (-1, 2)
noncomputable def circle_radius : ℝ := 5

def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 25

def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y = 10

theorem length_of_chord : ∀ (x y : ℝ), circle_equation x y ∧ line_equation x y → 
    2 * real.sqrt (circle_radius^2 - (real.dist circle_center (x, y))^2) = 4 * real.sqrt 6 := 
sorry

end length_of_chord_l96_96440


namespace profit_percentage_l96_96607

theorem profit_percentage (purchase_price sell_price : ℝ) (h1 : purchase_price = 600) (h2 : sell_price = 624) :
  ((sell_price - purchase_price) / purchase_price) * 100 = 4 := by
  sorry

end profit_percentage_l96_96607


namespace AC_length_l96_96790

noncomputable theory

-- Given conditions
def triangle_ABC (A B C : ℝ) (angle_A angle_B : ℝ) (BC : ℝ) : Prop :=
  angle_A = 60 ∧ angle_B = 45 ∧ BC = real.sqrt 3

-- Proving AC = sqrt(2)
theorem AC_length {A B C AC : ℝ} (h : triangle_ABC A B C 60 45 (real.sqrt 3)) : 
  AC = real.sqrt 2 := 
sorry

end AC_length_l96_96790


namespace max_product_dist_to_lines_l96_96449

variables {A B C O : Type} [metric_space A]

-- Definition that O is inside the triangle ABC
def in_triangle {A B C : point} (O : point) : Prop := sorry

-- Definition of perpendicular distances from a point to the sides of the triangle
def dist_to_line {A B C O : point} (ab ac : line) (defO : in_triangle O) :
  (d_a d_b d_c : ℝ) := sorry

-- Definitions of centroid of a triangle
def is_centroid {A B C : point} (O : point) : Prop := sorry

-- Final statement of the proof problem
theorem max_product_dist_to_lines {A B C O : point} 
  (defO : in_triangle O)
  (d_a d_b d_c : ℝ)
  (distances : dist_to_line defO d_a d_b d_c) :
  (d_a * d_b * d_c maximized) ↔ (is_centroid defO) :=
sorry

end max_product_dist_to_lines_l96_96449


namespace exposed_surface_area_is_4pi_l96_96254

structure CylindricalContainer where
  height : ℝ
  radius : ℝ
  deriving Repr

structure CutInfo where
  first_cut : ℝ
  second_cut : ℝ
  third_cut : ℝ
  deriving Repr

def calculate_exposed_surface_area (container : CylindricalContainer) (cuts : CutInfo) : ℝ :=
  let hA := cuts.first_cut
  let hB := cuts.second_cut
  let hC := cuts.third_cut
  let hD := container.height - (hA + hB + hC)
  let lateral_surface_area := 2 * Real.pi * container.radius * (hA + hB + hC + hD)
  let top_bottom_surface_area := 2 * Real.pi * container.radius ^ 2
  lateral_surface_area + top_bottom_surface_area

theorem exposed_surface_area_is_4pi :
  calculate_exposed_surface_area ⟨1, 1⟩ ⟨1/3, 1/4, 1/6⟩ = 4 * Real.pi := by
  sorry

end exposed_surface_area_is_4pi_l96_96254


namespace symmetry_center_of_g_l96_96072

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

def g (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6) - 1

theorem symmetry_center_of_g :
  ∃ (x : ℝ), x = Real.pi / 6 ∧ g x = -1 :=
sorry

end symmetry_center_of_g_l96_96072


namespace distance_between_cars_l96_96979

def initial_distance : ℝ := 105
def first_car_run1 : ℝ := 25
def first_car_turn_right : ℝ := 15
def first_car_turn_left : ℝ := 25
def second_car_run : ℝ := 35

theorem distance_between_cars : 
  let first_car_back := first_car_run1 in
  let second_car_remaining := initial_distance - first_car_back in
  let final_distance := second_car_remaining - second_car_run in 
  final_distance = 45 := by 
  sorry

end distance_between_cars_l96_96979


namespace ternary_to_decimal_l96_96665

theorem ternary_to_decimal (n : ℕ) (h : n = 121) : 
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 16 :=
by sorry

end ternary_to_decimal_l96_96665


namespace avg_speed_B_C_l96_96573

-- Definitions and conditions
def distance_A_B : ℝ := 120
def avg_speed_total : ℝ := 30
def distance_B_C : ℝ := distance_A_B / 2
def t_from_A_to_B (t : ℝ) := 3 * t -- the ride from A to B lasts 3 times as many hours as from B to C

-- Proof problem statement
theorem avg_speed_B_C (t : ℝ) : 
  avg_speed_total = (distance_A_B + distance_B_C) / (t_from_A_to_B t + t) → 
  (distance_B_C / t) = 40 :=
by
  intro h
  sorry

end avg_speed_B_C_l96_96573


namespace number_of_nonzero_terms_l96_96307

def poly1 : Polynomial ℤ := 2 * X + 5
def poly2 : Polynomial ℤ := 3 * X^2 + X + 6
def poly3 : Polynomial ℤ := X^3 + 3 * X^2 - 4 * X + 1

theorem number_of_nonzero_terms :
  (Polynomial.Coeff (poly1 * poly2 - 4 * poly3) (X^3) ≠ 0) ∧
  (Polynomial.Coeff (poly1 * poly2 - 4 * poly3) (X^2) ≠ 0) ∧
  (Polynomial.Coeff (poly1 * poly2 - 4 * poly3) (X^1) ≠ 0) ∧
  (Polynomial.Coeff (poly1 * poly2 - 4 * poly3) (X^0) ≠ 0) ∧
  (Polynomial.Coeff (poly1 * poly2 - 4 * poly3) (X^n) = 0 for n > 3 ∧ n < 0) :=
sorry

end number_of_nonzero_terms_l96_96307


namespace fraction_shaded_of_square_l96_96897

theorem fraction_shaded_of_square (x : ℝ) : 
  let P : ℝ × ℝ := (0, 0),
      Q : ℝ × ℝ := (x, x / 2),
      area_square := x^2,
      area_triangle := 1 / 2 * x * (x / 2) in
  (area_square - area_triangle) / area_square = 3 / 4 :=
by
  intro
  have A1 : area_square = x^2 := rfl
  have A2 : area_triangle = 1 / 2 * x * (x / 2) := rfl
  have A3 : area_square - area_triangle = x^2 - (1 / 4) * x^2 := by
    rw [A2]
    ring
  have A4 : (area_square - area_triangle) / area_square = (x^2 - (1 / 4) * x^2) / x^2 := by
    rw [A3]
    ring
  have A5 : (x^2 - (1 / 4) * x^2) / x^2 = 3 / 4 := by
    ring
  exact A5

#eval fraction_shaded_of_square 4 -- This will execute the theorem for a given x value, e.g., x = 4

end fraction_shaded_of_square_l96_96897


namespace find_multiplier_l96_96334

theorem find_multiplier :
  ∃ x : ℕ, 72514 * x = 724777430 ∧ x = 10001 :=
by
  use 10001
  split
  · norm_num
  · exact rfl

end find_multiplier_l96_96334


namespace paige_science_problems_l96_96483

variable (S : ℤ)

theorem paige_science_problems (h1 : 43 + S - 44 = 11) : S = 12 :=
by
  sorry

end paige_science_problems_l96_96483


namespace range_of_m_value_of_m_l96_96078

variables (m p x : ℝ)

-- Conditions: The quadratic equation x^2 - 2x + m - 1 = 0 must have two real roots.
def discriminant (m : ℝ) := (-2)^2 - 4 * 1 * (m - 1)

-- Part 1: Finding the range of values for m
theorem range_of_m (h : discriminant m ≥ 0) : m ≤ 2 := 
by sorry

-- Additional Condition: p is a real root of the equation x^2 - 2x + m - 1 = 0
def is_root (p m : ℝ) := p^2 - 2 * p + m - 1 = 0

-- Another condition: (p^2 - 2p + 3)(m + 4) = 7
def satisfies_condition (p m : ℝ) := (p^2 - 2 * p + 3) * (m + 4) = 7

-- Part 2: Finding the value of m given p is a real root and satisfies (p^2 - 2p + 3)(m + 4) = 7
theorem value_of_m (h1 : is_root p m) (h2 : satisfies_condition p m) : m = -3 := 
by sorry

end range_of_m_value_of_m_l96_96078


namespace solution_system_of_inequalities_l96_96919

theorem solution_system_of_inequalities (x : ℝ) : 
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * (x^2) - x - 1 > 0 ↔ (-2 ≤ x ∧ x < -1/2) ∨ (1 < x ∧ x < 6) :=
by {
  sorry
}

end solution_system_of_inequalities_l96_96919


namespace total_books_is_595_l96_96044

-- Definitions of the conditions
def satisfies_conditions (a : ℕ) : Prop :=
  ∃ R L : ℕ, a = 12 * R + 7 ∧ a = 25 * L - 5 ∧ 500 < a ∧ a < 650

-- The theorem statement
theorem total_books_is_595 : ∃ a : ℕ, satisfies_conditions a ∧ a = 595 :=
by
  use 595
  split
  · apply exists.intro 49, exists.intro 24, split
    -- a = 12R + 7
    · exact rfl
    -- a = 25L - 5
    · exact rfl
  -- Next check 500 < a and a < 650
  split
  · exact nat.lt_of_le_of_lt (by norm_num) (by norm_num)
  · exact nat.lt_of_le_of_lt (by norm_num) (by norm_num)

end total_books_is_595_l96_96044


namespace water_intersection_points_approx_l96_96995

noncomputable def cube_edge_water_intersection : ℝ :=
  let edge_length: ℝ := 1 in
  let water_height: ℝ := edge_length * (2 / 3) in
  let approx_pos: ℝ := 0.27 in
  approx_pos

theorem water_intersection_points_approx :
  ∃ (x: ℝ), x ≈ 0.27 :=
by {use cube_edge_water_intersection, sorry}

end water_intersection_points_approx_l96_96995


namespace solve_for_y_l96_96500

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l96_96500


namespace triangle_division_equal_areas_l96_96868

def triangle {T : Type} (A B C X Y Z : T) : Prop :=
  ∀ (AX AY AZ BX BY BZ CX CY CZ : T → T), 
    (non_intersecting_segments A B C AX AY AZ BX BY BZ CX CY CZ) → 
    (equal_areas A B C AX AY AZ BX BY BZ CX CY CZ)

theorem triangle_division_equal_areas {A B C X Y Z : ℝ} : 
  triangle A B C X Y Z →
  (segment_ratio B X = 3 * segment_ratio A X) ∧
  (segment_ratio C Y = 3 * segment_ratio B Y) ∧
  (segment_ratio A Z = 3 * segment_ratio C Z) :=
by
  sorry

end triangle_division_equal_areas_l96_96868


namespace log_tan_ratio_l96_96366

noncomputable def sin_add (α β : ℝ) : ℝ := Real.sin (α + β)
noncomputable def sin_sub (α β : ℝ) : ℝ := Real.sin (α - β)
noncomputable def tan_ratio (α β : ℝ) : ℝ := Real.tan α / Real.tan β

theorem log_tan_ratio (α β : ℝ)
  (h1 : sin_add α β = 1 / 2)
  (h2 : sin_sub α β = 1 / 3) :
  Real.logb 5 (tan_ratio α β) = 1 := by
sorry

end log_tan_ratio_l96_96366


namespace coin_toss_probability_coin_toss_probability_solution_l96_96544

theorem coin_toss_probability
  (A B : Prop)
  (one_head : A)
  (heads_heads : B) :
  (prob : ℝ) :=
by
  have possible_outcomes := [("H", "H"), ("H", "T"), ("T", "H")]
  have nA := possible_outcomes.count (λ x, x.1 = "H" ∨ x.2 = "H")
  have nAB := possible_outcomes.count (λ x, x = ("H", "H"))
  exact nAB.to_real / nA.to_real

theorem coin_toss_probability_solution :
  coin_toss_probability (λ _, true) (λ _, true) true true = (1 / 3 : ℝ) :=
by sorry

end coin_toss_probability_coin_toss_probability_solution_l96_96544


namespace find_original_number_l96_96582

def original_four_digit_number (N : ℕ) : Prop :=
  N >= 1000 ∧ N < 10000 ∧ (70000 + N) - (10 * N + 7) = 53208

theorem find_original_number (N : ℕ) (h : original_four_digit_number N) : N = 1865 :=
by
  sorry

end find_original_number_l96_96582


namespace hygiene_disease_study_risk_ratio_l96_96619

theorem hygiene_disease_study (n a b c d: ℕ) (ha : a = 40) (hb : b = 60) (hc : c = 10) (hd : d = 90) (hn : n = 200) :
  let k_squared := (n * (a*d - b*c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  k_squared = 24 := sorry

theorem risk_ratio (P_A_given_B P_A_given_B_compl P_A_compl_given_B P_A_compl_given_B_compl: ℚ) 
  (h1 : P_A_given_B = 2/5) (h2 : P_A_given_B_compl = 1/10) (h3 : P_A_compl_given_B = 3/5) (h4 : P_A_compl_given_B_compl = 9/10) : 
  let R := (P_A_given_B / P_A_compl_given_B) * (P_A_compl_given_B_compl / P_A_given_B_compl) in
  R = 6 := sorry

end hygiene_disease_study_risk_ratio_l96_96619


namespace servings_in_box_l96_96249

def number_of_servings (total_cereal : ℝ) (serving_size : ℝ) : ℝ :=
  total_cereal / serving_size

theorem servings_in_box : number_of_servings 24.5 1.75 = 14 := 
  by
    sorry

end servings_in_box_l96_96249


namespace ratio_R_eq_39_div_32_l96_96794

-- Defining the coordinates of cube vertices
def A := (0, 0, 0)
def B (s : ℝ) := (s, 0, 0)
def C (s : ℝ) := (s, 0, s)
def D (s : ℝ) := (0, 0, s)
def E (s : ℝ) := (0, s, 0)
def F (s : ℝ) := (s, s, 0)
def G (s : ℝ) := (s, s, s)
def H (s : ℝ) := (0, s, s)

-- Defining points J and I based on the conditions they are exposed to
def J (s : ℝ) := (s, s / 4, 0)
def I (s : ℝ) := (0, s, s / 4)

-- Compute lengths EC and IJ
def length_EC (s : ℝ) := real.sqrt (3 * s ^ 2)
def length_IJ (s : ℝ) := real.sqrt (s ^ 2 + (-3 * s / 4) ^ 2 + (-s / 4) ^ 2)

-- Area of the rhombus EJCI and the ratio R. Computing R^2
def area_EJCI (s : ℝ) := (real.sqrt 78 * s ^ 2) / 8
def ratio_R (s : ℝ) := ((real.sqrt 78) / 8)
def square_ratio_R (s : ℝ) := (ratio_R s) ^ 2

-- The theorem statement
theorem ratio_R_eq_39_div_32 : ∀ s : ℝ, square_ratio_R s = 39 / 32 :=
by
  intros,
  sorry

end ratio_R_eq_39_div_32_l96_96794


namespace num_boys_and_girls_l96_96274

def num_ways_to_select (x : ℕ) := (x * (x - 1) / 2) * (8 - x) * 6

theorem num_boys_and_girls (x : ℕ) (h1 : num_ways_to_select x = 180) :
    x = 5 ∨ x = 6 :=
by
  sorry

end num_boys_and_girls_l96_96274


namespace negation_equiv_l96_96417

-- Definitions for constant sequence and arithmetic sequence
def is_constant_sequence {α : Type} (s : ℕ → α) : Prop := ∀ n m : ℕ, s n = s m
def is_arithmetic_sequence {α : Type} [Add α] [HasSmul ℕ α] (s : ℕ → α) : Prop := ∃ d : α, ∀ n : ℕ, s (n + 1) = s n + d

-- Proposition p and its negation
def prop_p {α : Type} [Add α] [HasSmul ℕ α] (s : ℕ → α) : Prop := is_constant_sequence s → is_arithmetic_sequence s
def neg_prop_p {α : Type} [Add α] [HasSmul ℕ α] (s : ℕ → α) : Prop := ∃ s : ℕ → α, is_constant_sequence s ∧ ¬is_arithmetic_sequence s

-- The main theorem statement
theorem negation_equiv {α : Type} [Add α] [HasSmul ℕ α] (s : ℕ → α) :
  ¬prop_p s ↔ neg_prop_p s :=
sorry -- proof comes here

end negation_equiv_l96_96417


namespace handshake_count_l96_96092

-- Define the number of boys
def num_boys : ℕ := 12

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := n.choose k

-- The theorem to prove the number of handshakes
theorem handshake_count : combination num_boys 2 = 66 :=
by
  have h : combination num_boys 2 = (num_boys * (num_boys - 1)) / 2, from
    by unfold combination; apply nat.choose_eq_falling_mul_div falling_fac! sorry
  rw h
  simp [num_boys]
  norm_num
  done

end handshake_count_l96_96092


namespace three_pow_1000_mod_seven_l96_96970

theorem three_pow_1000_mod_seven : (3 ^ 1000) % 7 = 4 := 
by 
  -- proof omitted
  sorry

end three_pow_1000_mod_seven_l96_96970


namespace valid_paths_count_l96_96148

theorem valid_paths_count (n : ℕ) (h : n > 6) : 
  let paths_count : ℚ := (1/6 : ℚ) * (n-6) * (n-1) * (n+1) in
  paths_count = (1/6 : ℚ) * (n-6) * (n-1) * (n+1) :=
sorry

end valid_paths_count_l96_96148


namespace tshirt_cost_is_correct_l96_96656

-- Definitions from conditions
def total_amount_spent : ℝ := 115
def number_of_tshirts : ℝ := 12

-- Derived definition for cost per t-shirt
def cost_per_tshirt : ℝ := total_amount_spent / number_of_tshirts

-- Proof statement
theorem tshirt_cost_is_correct : cost_per_tshirt ≈ 9.58 := by
  sorry

end tshirt_cost_is_correct_l96_96656


namespace triangle_area_length_AD_l96_96150

-- Define the given conditions
variables (A B C : ℝ) -- Internal angles of the triangle
variables (a b c : ℝ) -- Sides opposite to the angles
variables (sin cos : ℝ → ℝ) -- Sine and cosine functions

-- Define assumptions
axiom triangle_abc : (c * cos B = sqrt 3 * b * sin C)
axiom triangle_condition_one : (a^2 * sin C = 4 * sqrt 3 * sin A)
axiom triangle_sides : (a = 2 * sqrt 3) (b = sqrt 7) (c > b)

-- Define the midpoint condition and point D
variable (D : ℝ × ℝ) -- Midpoint of side BC

-- Problem 1: Prove the area of the triangle
theorem triangle_area : (1 / 2 * a * c * sin B = sqrt 3) :=
sorry

-- Problem 2: Prove the length of AD
theorem length_AD : (AD = sqrt 13) :=
sorry

-- Placeholders for actual implementations of functions
variables (AD : ℝ) -- Length from A to D

end triangle_area_length_AD_l96_96150


namespace negation_universal_to_existential_l96_96783

-- Setup the necessary conditions and types
variable (a : ℝ) (ha : 0 < a ∧ a < 1)

-- Negate the universal quantifier
theorem negation_universal_to_existential :
  (¬ ∀ x < 0, a^x > 1) ↔ ∃ x_0 < 0, a^(x_0) ≤ 1 :=
by sorry

end negation_universal_to_existential_l96_96783


namespace volume_tetrahedron_l96_96805

variables (P Q R S : Type) 
variables [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S]

def length_PQ (pq : ℝ) := pq = 4
def area_PQR (area : ℝ) := area = 18
def area_PQS (area : ℝ) := area = 16
def angle_PQR_PQS (angle : ℝ) := angle = 45

theorem volume_tetrahedron (pq : ℝ) (area1 : ℝ) (area2 : ℝ) (angle : ℝ) 
  (h1 : length_PQ pq) (h2 : area_PQR area1) (h3 : area_PQS area2) (h4 : angle_PQR_PQS angle) : 
  (1 / 6) * area1 * area2 * real.sin (angle * real.pi / 180) = 17 :=
by 
  sorry

end volume_tetrahedron_l96_96805


namespace ball_attendance_l96_96841

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l96_96841


namespace range_of_a_for_unique_zero_l96_96786

noncomputable def has_exactly_one_zero (a : ℝ) : Prop := 
  ∃ x : ℝ, f a x = 0 ∧ ∀ y : ℝ, f a y = 0 → y = x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - x - 1

theorem range_of_a_for_unique_zero (a : ℝ) : has_exactly_one_zero a ↔ a = 0 ∨ a = -1/4 := 
by
  sorry

end range_of_a_for_unique_zero_l96_96786


namespace sector_area_is_one_l96_96067

def sector_area (θ : ℝ) (l : ℝ) (r : ℝ) (P : ℝ) : ℝ := 
  1/2 * l * r

theorem sector_area_is_one (θ l r P : ℝ) 
  (h1 : θ = 2)
  (h2 : P = 4) 
  (h3 : l + 2 * r = 4) 
  (h4 : l / r = 2) : 
  sector_area θ l r P = 1 := 
by
  sorry

end sector_area_is_one_l96_96067


namespace sum_of_cubes_l96_96424

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^3 + b^3 = 9 :=
by
  sorry

end sum_of_cubes_l96_96424


namespace geometric_sequence_common_ratio_q_is_one_l96_96433

-- Definitions and conditions from part (a)
variable (a : ℕ → ℝ) (q : ℝ)
variable [geometric_sequence : ∃ r, ∀ n, a n = a 1 * r ^ (n - 1)]

-- Existing conditions
variable (neg_a : ∀ n : ℕ, 0 < n → a n < 0)
variable (ineq : a 3 + a 7 ≥ 2 * a 5)

-- Expected proof statement
theorem geometric_sequence_common_ratio_q_is_one :
  ∃ r : ℝ, (∀ n : ℕ, 0 < n → a n < 0) → 
  (a 3 + a 7 ≥ 2 * a 5) → 
  r = 1 :=
by
  -- the proof is omitted
  sorry

end geometric_sequence_common_ratio_q_is_one_l96_96433


namespace solve_for_y_l96_96502

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l96_96502


namespace jenna_interest_l96_96826

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

def interest_earned (P r : ℝ) (n : ℕ) : ℝ :=
  compound_interest P r n - P

theorem jenna_interest :
  interest_earned 1500 0.05 5 = 414.42 :=
by
  sorry

end jenna_interest_l96_96826


namespace x_seq_bounds_a_seq_formula_l96_96353

noncomputable def x_seq : ℕ → ℝ
| 1       := 1 / 2
| (n + 1) := x_seq n / (2 - x_seq n)

def a_seq (n : ℕ) : ℝ := 1 / (x_seq n)

theorem x_seq_bounds (n : ℕ) (hn : n > 0) : 0 < x_seq n ∧ x_seq n < 1 := sorry

theorem a_seq_formula (n : ℕ) (hn : n > 0) : a_seq n = 2^(n - 1) + 1 := sorry

end x_seq_bounds_a_seq_formula_l96_96353


namespace set_M_is_real_l96_96338

theorem set_M_is_real (Z : ℂ) : {Z | (Z - 1) ^ 2 = (abs (Z - 1)) ^ 2} = {Z : ℂ | Z.im = 0} :=
sorry

end set_M_is_real_l96_96338


namespace sum_of_arithmetic_sequence_min_l96_96367

theorem sum_of_arithmetic_sequence_min (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) 
  (h1 : ∀ n, a (n + 1) = a 1 + n * 3)
  (h2 : a 1 = -26)
  (h3 : a 8 + a 13 = 5) 
  (hS : ∀ n, S n = n * (a 1 + a n) / 2)
  (hn : n = 9) : 
  ∀ k, k ∈ ℕ → S n ≤ S k := 
sorry

end sum_of_arithmetic_sequence_min_l96_96367


namespace pentagon_concyclic_l96_96204

-- Define concyclic points
def concyclic (A B C D E : Point) : Prop :=
∃ (circle : Circle), A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧ E ∈ circle

-- Define the geometric configuration
variables (A B C D E A' B' C' D' E' A'' B'' C'' D'' E'' : Point)

-- Pentagon and its extended triangles forming given geometric condition
variables (circumcircle_ABD' circumcircle_BCE' circumcircle_CDA' circumcircle_DEB' circumcircle_EAC' : Circle)
variables 
  (h1 : A ∈ circumcircle_ABD') (h2 : B ∈ circumcircle_ABD') (h3 : D' ∈ circumcircle_ABD')
  (h4 : B ∈ circumcircle_BCE') (h5 : C ∈ circumcircle_BCE') (h6 : E' ∈ circumcircle_BCE')
  (h7 : C ∈ circumcircle_CDA') (h8 : D ∈ circumcircle_CDA') (h9 : A' ∈ circumcircle_CDA')
  (h10 : D ∈ circumcircle_DEB') (h11 : E ∈ circumcircle_DEB') (h12 : B' ∈ circumcircle_DEB')
  (h13 : E ∈ circumcircle_EAC') (h14 : A ∈ circumcircle_EAC') (h15 : C' ∈ circumcircle_EAC')
  (hA'' : A'' ∈ circumcircle_ABD') (hB'' : B'' ∈ circumcircle_BCE')
  (hC'' : C'' ∈ circumcircle_CDA') (hD'' : D'' ∈ circumcircle_DEB') (hE'' : E'' ∈ circumcircle_EAC')

-- Lean statement
theorem pentagon_concyclic :
  concyclic A'' B'' C'' D'' E'' :=
sorry

end pentagon_concyclic_l96_96204


namespace product_of_points_is_correct_l96_96640

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 4
  else 0

def totalPoints (rolls : List ℕ) : ℕ :=
  rolls.map f |> List.sum

def AlexRolls := [6, 4, 3, 2, 1]
def BobRolls := [5, 6, 2, 3, 3]

def AlexPoints := totalPoints AlexRolls
def BobPoints := totalPoints BobRolls

theorem product_of_points_is_correct : AlexPoints * BobPoints = 672 := by
  sorry

end product_of_points_is_correct_l96_96640


namespace tangent_lines_minimize_pm_l96_96603

-- Definition of the given circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Problem statement 1: tangent lines with equal intercepts
theorem tangent_lines (x y : ℝ) : 
  circle_eq x y → (y = 0 ∨ y = x ∨ x + y = 1 + real.sqrt 2 ∨ x + y = 1 - real.sqrt 2) :=
by
  sorry

-- Problem statement 2: minimizing |PM|
theorem minimize_pm (x₀ y₀ : ℝ) : 
  2*x₀ - 4*y₀ + 1 = 0 ∧ (x₀ + 1)^2 + (y₀ - 2)^2 > 4 → real.sqrt ((x₀ + 1)^2 + (y₀ - 2)^2 - 4) = real.sqrt 2 :=
by
  sorry

end tangent_lines_minimize_pm_l96_96603


namespace proof_problem_l96_96704

noncomputable theory

def problem_statement : Prop :=
  let a := (10 / 11) * real.exp (1 / 11) in
  let b := 11 * real.log 1.1 in
  (1 < a * b) ∧ (a * b < b)

theorem proof_problem : problem_statement :=
sorry

end proof_problem_l96_96704


namespace number_of_books_in_library_l96_96040

theorem number_of_books_in_library 
  (a : ℕ) 
  (R L : ℕ) 
  (h1 : a = 12 * R + 7) 
  (h2 : a = 25 * L - 5) 
  (h3 : 500 < a ∧ a < 650) : 
  a = 595 :=
begin
  sorry
end

end number_of_books_in_library_l96_96040


namespace rectangle_area_l96_96624

theorem rectangle_area (a b c: ℝ) (h₁ : a = 7.1) (h₂ : b = 8.9) (h₃ : c = 10.0) (L W: ℝ)
  (h₄ : L = 2 * W) (h₅ : 2 * (L + W) = a + b + c) : L * W = 37.54 :=
by
  sorry

end rectangle_area_l96_96624


namespace coefficient_x_in_binomial_expansion_l96_96063

noncomputable def a : ℝ := ∫ x in - (π / 2)..(π / 2), Real.cos x 

theorem coefficient_x_in_binomial_expansion : 
  let binom := (a * x^2 - (1 / x))^5
  ∃ c : ℝ, ∃ t, t ∈ (finset.range 6).map (nat.cast_automorphism int.coe_nat), 
    (c • (x ^ (1 : ℕ)) = t) → c = -40 :=
sorry

end coefficient_x_in_binomial_expansion_l96_96063


namespace derivative_of_f_l96_96419

noncomputable def f (x : ℝ) : ℝ := -3 * x - 1

theorem derivative_of_f : deriv f = λ x, -3 :=
by
  sorry

end derivative_of_f_l96_96419


namespace mode_of_shoe_sizes_is_25_5_l96_96993

def sales_data := [(24, 2), (24.5, 5), (25, 3), (25.5, 6), (26, 4)]

theorem mode_of_shoe_sizes_is_25_5 
  (h : ∀ x ∈ sales_data, 2 ≤ x.1 ∧ 
        (∀ y ∈ sales_data, x.2 ≤ y.2 → x.1 = 25.5 ∨ x.2 < 6)) : 
  (∃ s, s ∈ sales_data ∧ s.1 = 25.5 ∧ s.2 = 6) :=
sorry

end mode_of_shoe_sizes_is_25_5_l96_96993


namespace infinite_decimals_less_than_one_l96_96521

theorem infinite_decimals_less_than_one : Infinite (set_of (λ x : ℝ, x < 1)) :=
sorry

end infinite_decimals_less_than_one_l96_96521


namespace B_finishes_job_in_48_days_l96_96975

variable (A B : ℝ) (h1 : A = 1/2 * B) (h2 : (A + B) * 32 = 1)

theorem B_finishes_job_in_48_days : B ≠ 0 → 1 / B = 48 := by
  intro hB
  have h3 : A + B = 3/2 * B := by
    rw [h1]
    linarith
  have h4 : (3/2 * B) * 32 = 1 := by
    rw [← h3, ← h2]
    linarith
  have h5 : B = 1 / 48 := by
    field_simp [mul_comm, div_eq_mul_inv, h4]
    linarith
  field_simp [h5]
  linarith

end B_finishes_job_in_48_days_l96_96975


namespace rationalize_denominator_and_product_l96_96173

theorem rationalize_denominator_and_product :
  let A := -11
  let B := -5
  let C := 5
  let expr := (3 + Real.sqrt 5) / (2 - Real.sqrt 5)
  (expr * (2 + Real.sqrt 5) / (2 + Real.sqrt 5) = A + B * Real.sqrt C) ∧ (A * B * C = 275) :=
by
  sorry

end rationalize_denominator_and_product_l96_96173


namespace distinct_roots_iff_l96_96172

theorem distinct_roots_iff (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m * x1 + m + 3 = 0 ∧ x2^2 + m * x2 + m + 3 = 0) ↔ (m < -2 ∨ m > 6) := 
sorry

end distinct_roots_iff_l96_96172


namespace trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l96_96871

variable {a b : ℝ}
variable {M N : ℝ}

/-- Trapezoid problem statements -/
theorem trapezoid_problem_case1 (h : a < 2 * b) : M - N = a - 2 * b := 
sorry

theorem trapezoid_problem_case2 (h : a = 2 * b) : M - N = 0 := 
sorry

theorem trapezoid_problem_case3 (h : a > 2 * b) : M - N = 2 * b - a := 
sorry

end trapezoid_problem_case1_trapezoid_problem_case2_trapezoid_problem_case3_l96_96871


namespace point_in_second_quadrant_l96_96124

-- Define the point in question
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Given conditions based on the problem statement
def P (x : ℝ) : Point :=
  Point.mk (-2) (x^2 + 1)

-- The theorem we aim to prove
theorem point_in_second_quadrant (x : ℝ) : (P x).x < 0 ∧ (P x).y > 0 → 
  -- This condition means that the point is in the second quadrant
  (P x).x < 0 ∧ (P x).y > 0 :=
by
  sorry

end point_in_second_quadrant_l96_96124


namespace problem_1_problem_2_problem_3_l96_96717

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ (k : ℕ), k > 0 → ∃ i, i ≤ k ∧ a (k + 1) = a k + a i

def sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in finset.range n, a (i + 1)

theorem problem_1 (a : ℕ → ℝ) (h : sequence a) :
  ∀ (i : ℕ), i > 0 → a i > 1 := sorry

theorem problem_2 (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : sequence a) (h2 : sum a S) :
  (∀ n, ∃ r, a n = a 1 * r ^ (n - 1)) → S 6 / a 3 = 63 / 4 := sorry

theorem problem_3 (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : sequence a) (h2 : sum a S) :
  ∀ (n : ℕ), 0 < n → (1 / 2 * n * (n + 1) ≤ S n ∧ S n ≤ 2 ^ n - 1) := sorry

end problem_1_problem_2_problem_3_l96_96717


namespace max_a_for_one_solution_l96_96166

-- Conditions
variables {a b : ℕ}
variables (ha : 1 < a) (hb : a < b)

-- The statement to be proved
theorem max_a_for_one_solution :
  ∃ x y : ℝ, y = -2 * x + 4033 ∧ y = |x - 1| + |x + a| + |x - b| → a ≤ 4031 :=
begin
  sorry
end

end max_a_for_one_solution_l96_96166


namespace correct_average_weight_l96_96511

theorem correct_average_weight
  (avg_weight : ℝ)
  (num_boys : ℕ)
  (misread_weight : ℝ)
  (actual_weight : ℝ)
  (incorrect_avg_weight : avg_weight = 58.4)
  (number_of_boys : num_boys = 20)
  (misread : misread_weight = 56)
  (actual : actual_weight = 62) :
  let incorrect_total_weight := avg_weight * num_boys,
      weight_difference := actual_weight - misread_weight,
      correct_total_weight := incorrect_total_weight + weight_difference,
      correct_avg_weight := correct_total_weight / num_boys in
  correct_avg_weight = 58.7 :=
begin
  sorry,
end

end correct_average_weight_l96_96511


namespace library_books_l96_96036

theorem library_books (a : ℕ) (R L : ℕ) :
  (∃ R, a = 12 * R + 7) ∧ (∃ L, a = 25 * L - 5) ∧ 500 < a ∧ a < 650 → a = 595 :=
by
  sorry

end library_books_l96_96036


namespace divisors_alternating_sum_prime_or_four_l96_96869

theorem divisors_alternating_sum_prime_or_four (n : ℕ) (d : ℕ → ℕ) (k : ℕ) 
  (h1 : d 1 = n) (h2 : d k = 1) (h3 : ∀ i : ℕ, 1 ≤ i ∧ i < k → d (i + 1) < d i) :
  (∑ i in finset.range (k - 1), (-1)^i * d (i + 1) = n - 1) ↔ (nat.prime n ∨ n = 4) :=
begin
  sorry,
end

end divisors_alternating_sum_prime_or_four_l96_96869


namespace find_BE_length_l96_96267

theorem find_BE_length (AB BC BE : ℝ) (hAB : AB = 5) (hBC : BC = 8) 
(h_area_eq : AB * BC = (1/2) * AB * BE) : BE = 16 := 
by
  have h1 : AB * BC = 40 := by
    rw [hAB, hBC]
    norm_num
  have h2 : (1/2) * AB * BE = 40 := by
    rw [h1, h_area_eq]
  have h3 : (1/2) * 5 * BE = 40 := by
    rw [hAB] at h2
    exact h2
  have h4 : 2.5 * BE = 40 := by
    norm_num at h3
    exact h3
  have h5 : BE = 40 / 2.5 := by
    norm_num at h4
    exact eq_div_iff_mul_eq _ _ _ rfl h4
  norm_num at h5
  exact h5

end find_BE_length_l96_96267


namespace library_books_l96_96038

theorem library_books (a : ℕ) (R L : ℕ) :
  (∃ R, a = 12 * R + 7) ∧ (∃ L, a = 25 * L - 5) ∧ 500 < a ∧ a < 650 → a = 595 :=
by
  sorry

end library_books_l96_96038


namespace first_fun_friday_is_march_31_l96_96438

-- Definitions for the conditions.
def isWednesday (date : ℕ) : Prop := true -- Placeholder for the condition "January 4 is a Wednesday"
def monthHasFiveFridays (month : ℕ) : Prop := 
  (month = 1 ∧ 5 = 5) ∨ (month = 3 ∧ 5 = 5) -- Placeholder, would need proper calculation of Fridays

-- Theorem statement: "What is the date of the first Fun Friday?"
theorem first_fun_friday_is_march_31 : ∃ date : ℕ, date = 31 ∧ monthHasFiveFridays 3 := 
by 
  existsi 31
  split
  · refl
  · sorry

end first_fun_friday_is_march_31_l96_96438


namespace g_value_at_6_l96_96614

noncomputable def g (v : ℝ) : ℝ :=
  let x := (v + 2) / 4
  x^2 - x + 2

theorem g_value_at_6 :
  g 6 = 4 := by
  sorry

end g_value_at_6_l96_96614


namespace correct_product_proof_l96_96116

variable (c d c' : ℕ)
variable (h_cd : c' * d = 143)
variable (h_c_reverse : reverse_digits c = c')

noncomputable def correct_product : Prop := c * d = 341

theorem correct_product_proof (c d c' : ℕ)
    (h_cd : c' * d = 143)
    (h_c_reverse : reverse_digits c = c') : correct_product c d :=
by
    sorry

end correct_product_proof_l96_96116


namespace average_difference_correct_l96_96938

def daily_diff : List ℤ := [15, 0, -15, 25, 5, -5, 10]
def number_of_days : ℤ := 7

theorem average_difference_correct :
  (daily_diff.sum : ℤ) / number_of_days = 5 := by
  sorry

end average_difference_correct_l96_96938


namespace find_list_price_l96_96519

noncomputable def list_price (P : ℝ) : ℝ := 
  0.90 * P * 0.9500000000000001

theorem find_list_price (P : ℝ) (h : list_price P = 59.85) : P ≈ 70 :=
by
  sorry -- proof omitted

end find_list_price_l96_96519


namespace binomial_coefficient_div_lcm_l96_96144

theorem binomial_coefficient_div_lcm (n : ℕ) (hn : 1 ≤ n) : 
  nat.choose (2 * n) n ∣ nat.lcm_list (list.range (2 * n + 1)) :=
sorry

end binomial_coefficient_div_lcm_l96_96144


namespace binom_1293_1_eq_1293_l96_96659

theorem binom_1293_1_eq_1293 : (Nat.choose 1293 1) = 1293 := 
  sorry

end binom_1293_1_eq_1293_l96_96659


namespace Misha_probability_l96_96598

open Probability

-- Definitions
def classesMonday := 5
def classesTuesday := 6
def totalClasses := 11
def totalCorrect := 7
def mondayCorrect := 3

-- Calculating binomial probabilities
noncomputable def P_A1 := (binomial classesMonday mondayCorrect) * (1 / 2) ^ classesMonday
noncomputable def P_A2 := (binomial classesTuesday (totalCorrect - mondayCorrect)) * (1 / 2) ^ classesTuesday
noncomputable def P_B := (binomial totalClasses totalCorrect) * (1 / 2) ^ totalClasses

-- Theorem stating the required probability
theorem Misha_probability :
  P_A1 * P_A2 / P_B = 5 / 11 :=
by
  sorry

end Misha_probability_l96_96598


namespace smallest_positive_integer_g_l96_96576

theorem smallest_positive_integer_g (g : ℕ) (h_pos : g > 0) (h_square : ∃ k : ℕ, 3150 * g = k^2) : g = 14 := 
  sorry

end smallest_positive_integer_g_l96_96576


namespace ellipse_problem_l96_96356

def ellipse_equation (a b : ℝ) (h₀ : 0 < b) (h₁ : b < a) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (a = sqrt 2 ∧ b = 1)

def isosceles_right_triangle (a b c : ℝ) : Prop :=
  (b = c) ∧ (a^2 - b^2 = c^2) ∧ (1/2 * a^2 = 1)

def line_intersects_ellipse (m x₁ x₂ : ℝ) : Prop :=
  ∀ (x₁ x₂ : ℝ), (3 * x₁ ^ 2 - 4 * m * x₁ + 2 * m ^ 2 - 2 = 0) ∧
  ( ∃ Δ, (16 * m ^ 2 - 12 * (2 * m ^ 2 - 2) = Δ) ∧ Δ > 0) ∧
  (1 / 2 * abs(4 / 3 * sqrt(3 - m^2)) = sqrt(6) / 2)

theorem ellipse_problem (a b c m x₁ x₂ : ℝ)
  (h₀ : 0 < b)
  (h₁ : b < a)
  (h₂ : isosceles_right_triangle a b c)
  (h₃ : line_intersects_ellipse m x₁ x₂) :
  ellipse_equation a b h₀ h₁ ∧ (m = sqrt 6 / 2 ∨ m = -sqrt 6 / 2) :=
sorry

end ellipse_problem_l96_96356


namespace surface_area_increase_l96_96255

theorem surface_area_increase (r h : ℝ) (cs : Bool) : -- cs is a condition switch, True for circular cut, False for rectangular cut
  0 < r ∧ 0 < h →
  let inc_area := if cs then 2 * π * r^2 else 2 * h * r 
  inc_area > 0 :=
by 
  sorry

end surface_area_increase_l96_96255


namespace baker_cakes_sold_l96_96289

-- Define the variables from the conditions
def first_batch := 54
def second_batch := (2 / 3) * first_batch
def left_first_batch := 13
def left_second_batch := Nat.floor (0.1 * second_batch) -- rounding down

-- Definition to calculate the total cakes sold
def total_cakes_sold := (first_batch - left_first_batch) + (second_batch - left_second_batch)

-- Lean statement to prove total cakes sold is 74
theorem baker_cakes_sold : total_cakes_sold = 74 := by
  sorry

end baker_cakes_sold_l96_96289


namespace ball_attendance_l96_96839

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l96_96839


namespace problem1_l96_96243

theorem problem1 (f : ℝ → ℝ) (x : ℝ) : 
  (f (x + 1/x) = x^2 + 1/x^2) -> f x = x^2 - 2 := 
sorry

end problem1_l96_96243


namespace arithmetic_sequence_general_term_b_n_bound_l96_96716

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)

def sqrt_condition (n : ℕ) := (sqrt (2 * S n) = (a n + 2) / 2)
def sum_condition (n : ℕ) := (S n = ∑ i in finset.range n, a i)

theorem arithmetic_sequence_general_term :
  (∀ n : ℕ, sqrt_condition a S n) →
  (∀ n : ℕ, sum_condition a S n) →
  ∃ (d : ℝ), ∀ n : ℕ, (a n = 4 * n - 2) :=
by
  intros h1 h2
  sorry

def b (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), 1 / (a n + a i)

theorem b_n_bound (n : ℕ) :
  (∀ n : ℕ, sqrt_condition a S n) →
  (∀ n : ℕ, sum_condition a S n) →
  (a = λ n, 4 * n - 2) →
  b n ≤ 3 / 8 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_sequence_general_term_b_n_bound_l96_96716


namespace find_x_l96_96728

-- Define vectors a and b
def a : ℝ × ℝ × ℝ := (2, -1, 2)
def b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

-- Define the dot product for 3-dimensional vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the condition of perpendicular vectors (dot product is zero)
def is_perpendicular (u v : ℝ × ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- The theorem to prove that under the given conditions x = 5
theorem find_x (x : ℝ) (h : is_perpendicular a (b x)) : x = 5 :=
by sorry

end find_x_l96_96728


namespace rabbit_roaming_area_l96_96825

noncomputable def rabbit_area_midpoint_long_side (r: ℝ) : ℝ :=
  (1/2) * Real.pi * r^2

noncomputable def rabbit_area_3_ft_from_corner (R r: ℝ) : ℝ :=
  (3/4) * Real.pi * R^2 - (1/4) * Real.pi * r^2

theorem rabbit_roaming_area (r R : ℝ) (h_r_pos: 0 < r) (h_R_pos: r < R) :
  rabbit_area_3_ft_from_corner R r - rabbit_area_midpoint_long_side R = 22.75 * Real.pi :=
by
  sorry

end rabbit_roaming_area_l96_96825


namespace gcd_442872_312750_l96_96552

theorem gcd_442872_312750 : Nat.gcd 442872 312750 = 18 :=
by
  sorry

end gcd_442872_312750_l96_96552


namespace midpoint_complex_number_l96_96443

open Complex

/-- The complex number corresponding to point P is 1,
given that points M and N correspond to complex numbers 2 / (1 + I) and 2 / (1 - I) respectively,
and P is the midpoint of the line segment MN. -/
theorem midpoint_complex_number (M N P : ℂ) (hM : M = 2 / (1 + I)) (hN : N = 2 / (1 - I)) : 
  P = (M + N) / 2 → P = 1 := 
by 
  intro hP
  rw [hM, hN] at hP
  calc 
    P = (2 / (1 + I) + 2 / (1 - I)) / 2 : by exact hP
    ... = (1 - I + 1 + I) / ((1 + I) * (1 - I)) : by field_simp
    ... = 1 : by norm_num

end midpoint_complex_number_l96_96443


namespace walnut_trees_planted_l96_96955

theorem walnut_trees_planted (initial_trees : ℕ) (final_trees : ℕ) (num_trees_planted : ℕ) : initial_trees = 107 → final_trees = 211 → num_trees_planted = final_trees - initial_trees → num_trees_planted = 104 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end walnut_trees_planted_l96_96955


namespace trigonometric_identity_l96_96702

theorem trigonometric_identity (α : ℝ) (h : (sin α - cos α) / (sin α + cos α) = 1 / 2) :
  cos (2 * α) = -4 / 5 := 
sorry

end trigonometric_identity_l96_96702


namespace ball_total_attendance_l96_96847

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l96_96847


namespace polar_to_cartesian_and_segment_length_l96_96806

theorem polar_to_cartesian_and_segment_length :
  (∀ (θ : ℝ), 4 * cos θ = sqrt ((2 + 2 * cos θ)^2 + (2 * sin θ)^2)) ∧
  ∃ (x y : ℝ), (x - y = 4 ∧ x^2 + y^2 = 4 * x) ∧
    let M := (4, 0), N := (2, -2) in
    sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 * sqrt 2 :=
by
  sorry

end polar_to_cartesian_and_segment_length_l96_96806


namespace max_slope_without_lattice_points_l96_96262

theorem max_slope_without_lattice_points :
  ∃ b : ℚ,
    (∀ m : ℚ, (1 : ℚ) / 3 < m → m < b → 
      ∀ x : ℤ, 1 ≤ x → x ≤ 150 → ∀ y : ℤ, y ≠ m * x + 3) ∧
    b = 50 / 151 := 
begin
  -- proof here
  sorry
end

end max_slope_without_lattice_points_l96_96262


namespace second_movie_time_difference_l96_96453

def first_movie_length := 90 -- 1 hour and 30 minutes in minutes
def popcorn_time := 10 -- Time spent making popcorn in minutes
def fries_time := 2 * popcorn_time -- Time spent making fries in minutes
def total_time := 4 * 60 -- Total time for cooking and watching movies in minutes

theorem second_movie_time_difference :
  (total_time - (popcorn_time + fries_time + first_movie_length)) - first_movie_length = 30 :=
by
  sorry

end second_movie_time_difference_l96_96453


namespace lottery_win_is_random_event_l96_96566

inductive Event
| no_moisture_seed_germination
| at_least_2_people_same_birthday
| melting_ice_at_minus_1C
| lottery_win

def is_random_event : Event → Prop
| Event.no_moisture_seed_germination := False
| Event.at_least_2_people_same_birthday := False
| Event.melting_ice_at_minus_1C := False
| Event.lottery_win := True

theorem lottery_win_is_random_event : is_random_event Event.lottery_win := 
by
  exact True.intro

end lottery_win_is_random_event_l96_96566


namespace prism_side_length_l96_96815

theorem prism_side_length (α : ℝ) :
  let a := 2
  ∃ (AB : ℝ),
  AB = 4 * sin (α / 2) / sqrt (3 - 4 * sin (α / 2)^2) := sorry

end prism_side_length_l96_96815


namespace max_score_75_l96_96532

/-- There are 9 boxes in a row, Player A distributes 25 balls per turn, Player B can remove balls from any two consecutive boxes each turn.
Prove that the maximum score Player A can achieve is 75 balls in one of the boxes. --/
theorem max_score_75 (boxes : ℕ := 9) (balls_each_turn : ℕ := 25) : ℕ :=
  let max_box_count := 75 in
  have ample_supply : Prop := (∀ n, n > 0 → exists m, m > n),
  have player_a_strategy : Prop := (∀ t, t > 0 → ∀ b : ℕ → b mod 2 = 1),
  have player_b_strategy : Prop := (∀ t, t > 0 → ∀ i, i < boxes - 1 → (b i, b (i + 1)) → b i + b (i + 1)),
  have end_game_condition : Prop := (∃ b, b ∈ (0 .. boxes) → b = max_box_count),
  max_box_count = 75

end max_score_75_l96_96532


namespace total_profit_at_100_max_profit_price_l96_96989

noncomputable def sales_volume (x : ℝ) : ℝ := 15 - 0.1 * x
noncomputable def floating_price (S : ℝ) : ℝ := 10 / S
noncomputable def supply_price (x : ℝ) : ℝ := 30 + floating_price (sales_volume x)
noncomputable def profit_per_set (x : ℝ) : ℝ := x - supply_price x
noncomputable def total_profit (x : ℝ) : ℝ := profit_per_set x * sales_volume x

-- Theorem 1: Total profit when each set is priced at 100 yuan is 340 ten thousand yuan
theorem total_profit_at_100 : total_profit 100 = 340 := by
  sorry

-- Theorem 2: The price per set that maximizes profit per set is 140 yuan
theorem max_profit_price : ∃ x, profit_per_set x = 100 ∧ x = 140 := by
  sorry

end total_profit_at_100_max_profit_price_l96_96989


namespace mass_percentage_O_in_BaO_l96_96328

theorem mass_percentage_O_in_BaO 
  (M_Ba : ℝ) (M_O : ℝ) 
  (h_Ba : M_Ba = 137.33) 
  (h_O : M_O = 16.00) : 
  ((M_O / (M_Ba + M_O)) * 100) ≈ 10.43 :=
sorry

end mass_percentage_O_in_BaO_l96_96328


namespace range_of_a_l96_96671

open Real

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, otimes x (x + a) < 1) ↔ -1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l96_96671


namespace objects_meeting_conditions_l96_96772

-- Definitions
variable (a τ g : ℝ)
-- Conditions
axiom initial_velocity_positive : a > 0
axiom gravity_positive : g > 0
-- General condition for meeting at any time
noncomputable def meets_at_all (b : ℝ) : Prop := b > a - g * τ
-- Specific condition for meeting at peak ascent
noncomputable def meets_at_peak_ascent (b : ℝ) : Prop := τ = a / g → b = a / Real.sqrt 2

-- Theorem for the conditions
theorem objects_meeting_conditions : ∀ (b : ℝ),
  (meets_at_all a τ g b) ∧ (meets_at_peak_ascent a τ g b) :=
by
  intro b
  split
  case eq_1 => sorry -- Proof that objects meet at all if meets_at_all is satisfied
  case eq_2 => sorry -- Proof that objects meet at peak ascent if meets_at_peak_ascent is satisfied

end objects_meeting_conditions_l96_96772


namespace factor_expression_l96_96010

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l96_96010


namespace lcm_fraction_value_l96_96460

/-- Let P be the least common multiple (LCM) of all the integers from 15 through 35, inclusive. 
Let Q be the LCM of P and the integers 36, 37, 38, 39, 40, 41, 42, 43, and 45.
Prove that the value of Q / P is 65231. -/
theorem lcm_fraction_value : 
  let P := Nat.lcm_list ([15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]) in
  let Q := Nat.lcm_list ([P, 36, 37, 38, 39, 40, 41, 42, 43, 45]) in
    Q / P = 65231 :=
by
  sorry

end lcm_fraction_value_l96_96460


namespace simplify_tangent_sum_l96_96908

theorem simplify_tangent_sum :
  (1 + Real.tan (10 * Real.pi / 180)) * (1 + Real.tan (35 * Real.pi / 180)) = 2 := by
  have h1 : Real.tan (45 * Real.pi / 180) = Real.tan ((10 + 35) * Real.pi / 180) := by sorry
  have h2 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h3 : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y) := by sorry
  sorry

end simplify_tangent_sum_l96_96908


namespace g_36_equals_72_l96_96875

noncomputable def g : ℕ → ℕ := sorry

-- g is an increasing function
axiom g_increasing : ∀ m n : ℕ, m < n → g(m) < g(n)

-- g(mn) = g(m) + g(n) + mn for all positive integers m and n
axiom g_multiplicative : ∀ m n : ℕ, 0 < m → 0 < n → g(m * n) = g(m) + g(n) + m * n

-- If m ≠ n and m^n = n^m, then g(m) = n or g(n) = m
axiom g_power_property : ∀ m n : ℕ, m ≠ n → m^n = n^m → (g(m) = n ∨ g(n) = m)

-- The function to find g(36) and check if it equals 72
theorem g_36_equals_72 : g(36) = 72 := 
sorry

end g_36_equals_72_l96_96875


namespace sum_of_reciprocals_of_squares_l96_96237

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 3) :
  (1 / (a : ℚ)^2) + (1 / (b : ℚ)^2) = 10 / 9 :=
sorry

end sum_of_reciprocals_of_squares_l96_96237


namespace find_xy_l96_96474

theorem find_xy (x y : ℝ) (h1 : (x / 6) * 12 = 11) (h2 : 4 * (x - y) + 5 = 11) : 
  x = 5.5 ∧ y = 4 :=
sorry

end find_xy_l96_96474


namespace clock_angle_at_5_50_l96_96925

/-- The angle formed by the hands of a clock at 5:50 -/
theorem clock_angle_at_5_50 : 
  let hour_deg_per_hour := 30 -- degrees per hour for the hour hand
  let minute_deg_per_minute := 6 -- degrees per minute for the minute hand
  let hour_at_5 := 5 * hour_deg_per_hour -- degrees at 5:00
  let hour_additional_moves := (hour_deg_per_hour / 60) * 50 -- additional degrees for the hour hand in 50 minutes
  let hour_at_5_50 := hour_at_5 + hour_additional_moves -- total degrees for the hour hand at 5:50
  let minute_at_50 := 50 * minute_deg_per_minute -- degrees for the minute hand at 50 minutes
  abs (minute_at_50 - hour_at_5_50) = 125 :=
by
  sorry

end clock_angle_at_5_50_l96_96925


namespace distance_to_axes_l96_96933

def point (P : ℝ × ℝ) : Prop :=
  P = (3, 5)

theorem distance_to_axes (P : ℝ × ℝ) (hx : P = (3, 5)) : 
  abs P.2 = 5 ∧ abs P.1 = 3 :=
by 
  sorry

end distance_to_axes_l96_96933


namespace unique_prime_roots_quadratic_l96_96648

theorem unique_prime_roots_quadratic (k : ℕ) (h : ∃ p q : ℕ, prime p ∧ prime q ∧ p + q = 58 ∧ p * q = k) : k = 265 :=
sorry

end unique_prime_roots_quadratic_l96_96648


namespace intervals_of_monotonicity_and_extreme_values_l96_96687

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x + 4

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x, (x < -2 ∨ x > 2) → f' x > 0) ∧
  (∀ x, -2 < x ∧ x < 2 → f' x < 0) ∧
  (f (-2) = 28/3) ∧
  (f (2) = -4/3) :=
by
  sorry

end intervals_of_monotonicity_and_extreme_values_l96_96687


namespace probability_correct_predictions_monday_l96_96590

def number_of_classes_monday : ℕ := 5
def number_of_classes_tuesday : ℕ := 6
def total_classes : ℕ := number_of_classes_monday + number_of_classes_tuesday

def total_correct_predictions : ℕ := 7
def correct_predictions_monday : ℕ := 3
def correct_predictions_tuesday : ℕ := total_correct_predictions - correct_predictions_monday

noncomputable def binomial_coefficient (n k : ℕ) : ℝ := (nat.choose n k : ℝ)

noncomputable def probability_exact_correct_monday (n k : ℕ) : ℝ :=
  binomial_coefficient n k * (real.exp 11 (ln 2⁻¹ * total_classes))

theorem probability_correct_predictions_monday :
  probability_exact_correct_monday number_of_classes_monday correct_predictions_monday *
  probability_exact_correct_monday number_of_classes_tuesday correct_predictions_tuesday /
  probability_exact_correct_monday total_classes total_correct_predictions = 5 / 11 :=
sorry

end probability_correct_predictions_monday_l96_96590


namespace locus_is_ellipse_l96_96660

-- Given three points A, B, and P in a metric space
variables {α : Type*} [metric_space α]
variables (A B P : α)

-- Define the distance between points A and B
def dist_AB : ℝ := dist A B

-- Define the condition that the sum of distances from P to A and P to B is equal to 2 times distance AB
def condition (P A B : α) : Prop :=
  dist P A + dist P B = 2 * dist_AB A B

-- Theorem stating the locus of P given the condition
theorem locus_is_ellipse (A B : α) (d : ℝ) :
  (∀ P, condition P A B) →
  ∃ e : set α, (∀ P ∈ e, dist P A + dist P B = 2 * d) ∧ is_ellipse_with_foci e A B :=
sorry

end locus_is_ellipse_l96_96660


namespace soccer_team_games_l96_96478

theorem soccer_team_games (pizzas : ℕ) (slices_per_pizza : ℕ) (average_goals_per_game : ℕ) (total_games : ℕ) 
  (h1 : pizzas = 6) 
  (h2 : slices_per_pizza = 12) 
  (h3 : average_goals_per_game = 9) 
  (h4 : total_games = (pizzas * slices_per_pizza) / average_goals_per_game) :
  total_games = 8 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end soccer_team_games_l96_96478


namespace area_of_enclosed_shape_l96_96018

theorem area_of_enclosed_shape :
  let curve := λ x : ℝ, x^2 - 2 * x
  let line := λ x : ℝ, -x
  let integral := ∫ x in 0..1, line x - curve x
  integral = 1 / 6 :=
by
  sorry

end area_of_enclosed_shape_l96_96018


namespace parabola_translation_l96_96960

theorem parabola_translation :
  ∀ x y, (y = -2 * x^2) →
    ∃ x' y', y' = -2 * (x' - 2)^2 + 1 ∧ x' = x ∧ y' = y + 1 :=
sorry

end parabola_translation_l96_96960


namespace Louise_needs_23_boxes_l96_96887

-- Given conditions
def number_red_pencils := 45
def red_box_capacity := 15
def number_blue_pencils := 3 * number_red_pencils
def blue_box_capacity := 25
def number_yellow_pencils := 80
def yellow_box_capacity := 10
def number_green_pencils := number_red_pencils + number_blue_pencils
def green_box_capacity := 30

-- Prove that the total number of boxes needed is 23
theorem Louise_needs_23_boxes :
  (number_red_pencils / red_box_capacity).ceil +
  (number_blue_pencils / blue_box_capacity).ceil +
  (number_yellow_pencils / yellow_box_capacity).ceil +
  (number_green_pencils / green_box_capacity).ceil = 23 := by
  sorry

end Louise_needs_23_boxes_l96_96887


namespace angle_quadrant_proof_l96_96472

def quadrant_of_complex_number (θ : ℝ) : Prop :=
  ∃ k : ℤ, (π / 2 + 2 * k * π) < 2 * θ ∧ 2 * θ < π + 2 * k * π

def angle_quadrant (θ : ℝ) : Prop :=
  ∃ k : ℤ, (π / 4 + k * π) < θ ∧ θ < (π / 2 + k * π)

theorem angle_quadrant_proof (θ : ℝ) (h : quadrant_of_complex_number θ) : 
  angle_quadrant θ :=
sorry

end angle_quadrant_proof_l96_96472


namespace line_through_point_tangent_to_parabola_l96_96326

theorem line_through_point_tangent_to_parabola:
  ∀ (l : ℝ → ℝ) (p : ℝ × ℝ), 
    (p = (0,1)) ∧ (∀ x, (l x)^2 = 2 * x) → (l 1 = 1) ∨ 
                                          (l 0 = 0) ∨ 
                                          ((l 2 - 2 * 1 + 2) = 0) :=
begin
  sorry -- Proof not provided
end

end line_through_point_tangent_to_parabola_l96_96326


namespace ratio_left_handed_to_non_throwers_l96_96158

-- Total number of players on the team
def total_players : ℕ := 70

-- Number of players who are throwers
def throwers : ℕ := 52

-- Total number of right-handed players
def right_handed_players : ℕ := 64

-- All throwers are right-handed
theorem ratio_left_handed_to_non_throwers :
  ∀ (total_players throwers right_handed_players non_throwers left_handed_non_throwers : ℕ), 
  non_throwers = total_players - throwers → 
  left_handed_non_throwers = non_throwers - (right_handed_players - throwers) →
  (total_players = 70) → (throwers = 52) → (right_handed_players = 64) →
  left_handed_non_throwers : non_throwers = 1 : 3 :=
by
  sorry

end ratio_left_handed_to_non_throwers_l96_96158


namespace permutation_sum_integer_l96_96034

theorem permutation_sum_integer (n : ℕ) (h : n > 0) : 
  ∃ s_n, (∀ (a : Finset (Fin n)), 
    (\sum i in a, ((a.val i).val + 1)/((i.val + 1) : ℕ)) ∈ ℤ) ∧ s_n >= n :=
sorry

end permutation_sum_integer_l96_96034


namespace first_stack_height_l96_96482

theorem first_stack_height (x : ℕ) (h1 : x + (x + 2) + (x - 3) + (x + 2) = 21) : x = 5 :=
by
  sorry

end first_stack_height_l96_96482


namespace tangent_line_k_value_l96_96099

theorem tangent_line_k_value :
  ∃ k : ℝ, (∀ (m : ℝ), m = real.exp (-1/2) → k = 2 * real.sqrt real.exp 1) :=
begin
  sorry
end

end tangent_line_k_value_l96_96099


namespace find_angle_A_l96_96820

theorem find_angle_A
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) : A = 50 :=
by 
  sorry

end find_angle_A_l96_96820


namespace total_money_shared_l96_96831

-- Define the variables and conditions
def joshua_share : ℕ := 30
def justin_share : ℕ := joshua_share / 3
def total_shared_money : ℕ := joshua_share + justin_share

-- State the theorem to prove
theorem total_money_shared : total_shared_money = 40 :=
by
  -- proof will go here
  sorry

end total_money_shared_l96_96831


namespace product_of_g_xi_l96_96466

open Polynomial

noncomputable def g (x : ℝ) := x^2 - 2

noncomputable def h : Polynomial ℝ := X^5 - 3*X^3 + X + 6

theorem product_of_g_xi :
  let xi := h.roots
  ∏ i in finset.range 5, g (xi i) = 10 :=
by
  sorry

end product_of_g_xi_l96_96466


namespace centroids_prove_l96_96583

variable {Ω : Type*} [AffineSpace ℝ Ω]

def isConvexQuadrilateral (A B C D : Ω) : Prop := 
  ∃ E F G H : Ω, 
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
  affineConvexCombination A B E ∧
  affineConvexCombination B C F ∧
  affineConvexCombination C D G ∧
  affineConvexCombination D A H

def midpoint (A B : Ω) : Ω := (A + B) / 2

def centerOfGravity (A B C D : Ω) : Ω :=
  (A + B + C + D) / 4

theorem centroids_prove
  (A B C D : Ω)
  (hconvex : isConvexQuadrilateral A B C D)
  (diagonal1 : Line AC)
  (diagonal2 : Line BD)
  (K : Ω)
  (K_def : parallel_to (submodule.span (AC : Set Ω)) (CD : Set Ω) ∧
          parallel_to (submodule.span (BD : Set Ω)) (CD : Set Ω) ∧
          intersection_lines (submodule.span diagonal1) (submodule.span diagonal2) = K)
  (M : Ω)
  (M_def : M = midpoint A B) :
  let O := centerOfGravity A B C D in
  ∃ K M : Ω, K ≠ M ∧ ∃ c : ℝ, c > 0 ∧ affine_ratio K O M 2 1 := sorry

end centroids_prove_l96_96583


namespace remainder_of_largest_divided_by_next_largest_l96_96956

/-
  Conditions:
  Let a = 10, b = 11, c = 12, d = 13.
  The largest number is d (13) and the next largest number is c (12).

  Question:
  What is the remainder when the largest number is divided by the next largest number?

  Answer:
  The remainder is 1.
-/

theorem remainder_of_largest_divided_by_next_largest :
  let a := 10 
  let b := 11
  let c := 12
  let d := 13
  d % c = 1 :=
by
  sorry

end remainder_of_largest_divided_by_next_largest_l96_96956


namespace standard_eq_of_ellipse_value_of_k_l96_96719

-- Definitions and conditions
def is_ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0

def eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = (Real.sqrt 2) / 2 ∧ a^2 = b^2 + (a * e)^2

def minor_axis_length (b : ℝ) : Prop :=
  2 * b = 2

def is_tangency (k m : ℝ) : Prop := 
  m^2 = 1 + k^2

def line_intersect_ellipse (k m : ℝ) : Prop :=
  (4 * k * m)^2 - 4 * (1 + 2 * k^2) * (2 * m^2 - 2) > 0

def dot_product_condition (k m : ℝ) : Prop :=
  let x1 := -(4 * k * m) / (1 + 2 * k^2)
  let x2 := (2 * m^2 - 2) / (1 + 2 * k^2)
  let y1 := k * x1 + m
  let y2 := k * x2 + m
  x1 * x2 + y1 * y2 = 2 / 3

-- To prove the standard equation of the ellipse
theorem standard_eq_of_ellipse {a b : ℝ} (h_ellipse : is_ellipse a b)
  (h_eccentricity : eccentricity a b ((Real.sqrt 2) / 2)) 
  (h_minor_axis : minor_axis_length b) : 
  ∃ a, a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y, (x^2 / 2 + y^2 = 1)) := 
sorry

-- To prove the value of k
theorem value_of_k {k m : ℝ} (h_tangency : is_tangency k m) 
  (h_intersect : line_intersect_ellipse k m)
  (h_dot_product : dot_product_condition k m) :
  k = 1 ∨ k = -1 :=
sorry

end standard_eq_of_ellipse_value_of_k_l96_96719


namespace length_after_5th_cut_l96_96634

theorem length_after_5th_cut (initial_length : ℝ) (n : ℕ) (h1 : initial_length = 1) (h2 : n = 5) :
  initial_length / 2^n = 1 / 2^5 := by
  sorry

end length_after_5th_cut_l96_96634


namespace water_fountain_length_l96_96244

noncomputable def fountain_length_built_by_20_men_in_7_days 
  (man_days_35_men_3_days : ℕ) (length_35_men_3_days : ℕ) (man_days_20_men_7_days : ℕ) : ℕ := 
  let man_days_per_meter := man_days_35_men_3_days / length_35_men_3_days in
  man_days_20_men_7_days / man_days_per_meter

theorem water_fountain_length 
  (length_35_men_3_days : ℕ)
  (man_days_35_men_3_days : ℕ)
  (man_days_20_men_7_days : ℕ) 
  (h1 : man_days_35_men_3_days = 35 * 3)
  (h2 : length_35_men_3_days = 42)
  (h3 : man_days_20_men_7_days = 20 * 7) :
  fountain_length_built_by_20_men_in_7_days man_days_35_men_3_days length_35_men_3_days man_days_20_men_7_days = 56 :=
by
  -- Proof goes here
  sorry

end water_fountain_length_l96_96244


namespace students_in_line_l96_96920

theorem students_in_line (between : ℕ) (Yoojung Eunji : ℕ) (h1 : Yoojung = 1) (h2 : Eunji = 1) : 
  between + Yoojung + Eunji = 16 :=
  sorry

end students_in_line_l96_96920


namespace largest_number_with_digits_4_or_2_sum_20_l96_96553

theorem largest_number_with_digits_4_or_2_sum_20 :
  ∃ n : ℕ, n = 44444 ∧ (∀ d ∈ [4, 4, 4, 4, 4], d = 4 ∨ d = 2) ∧ (4 + 4 + 4 + 4 + 4 = 20) := 
begin
  use 44444,
  split,
  { refl, },
  split,
  { intro d,
    simp,
    intros h,
    exact or.inl h, },
  { norm_num, },
end

end largest_number_with_digits_4_or_2_sum_20_l96_96553


namespace new_persons_joined_l96_96260

theorem new_persons_joined :
  ∀ (A : ℝ) (N : ℕ) (avg_new : ℝ) (avg_combined : ℝ), 
  N = 15 → avg_new = 15 → avg_combined = 15.5 → 1 = (N * avg_combined + N * avg_new - 232.5) / (avg_combined - avg_new) := by
  intros A N avg_new avg_combined
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end new_persons_joined_l96_96260


namespace find_y_l96_96681

-- Define the condition that log_y 8 = log_64 4
def log_condition (y : ℝ) : Prop := log y 8 = log 64 4

-- The theorem to prove
theorem find_y (y : ℝ) (h : log_condition y) : y = 512 :=
by
  -- proof to be provided
  sorry

end find_y_l96_96681


namespace geometric_sequence_terms_l96_96735

theorem geometric_sequence_terms
  (a : ℚ) (l : ℚ) (r : ℚ) (n : ℕ)
  (h_a : a = 9 / 8)
  (h_l : l = 1 / 3)
  (h_r : r = 2 / 3)
  (h_geo : l = a * r^(n - 1)) :
  n = 4 :=
by
  sorry

end geometric_sequence_terms_l96_96735


namespace theon_speed_l96_96529

theorem theon_speed (VTheon VYara D : ℕ) (h1 : VYara = 30) (h2 : D = 90) (h3 : D / VTheon = D / VYara + 3) : VTheon = 15 := by
  sorry

end theon_speed_l96_96529


namespace find_salary_l96_96978

theorem find_salary (x y : ℝ) (h1 : x + y = 2000) (h2 : 0.05 * x = 0.15 * y) : x = 1500 :=
sorry

end find_salary_l96_96978


namespace increasing_interval_f_l96_96193

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 3)

theorem increasing_interval_f :
  (∀ x, x ∈ Set.Ioi 3 → f x ∈ Set.Ioi 3) := sorry

end increasing_interval_f_l96_96193


namespace volume_ratio_inequality_l96_96185

theorem volume_ratio_inequality 
  (S A B C D M N K: Type)
  (base_parallelogram : parallelogram ABCD)
  (midpoint_K : midpoint K S C)
  (plane_intersect : plane_through A K)
  (M_intersect : intersection M plane_intersect SB)
  (N_intersect : intersection N plane_intersect SD)
  (V : ℝ)
  (V1 : ℝ)
  (v1_volume_tetrahedron : V1 = volume_tetrahedron S A M K N) 
  (v_volume_tetrahedron : V = volume_tetrahedron S A B C D) : 
  (1 / 3 : ℝ) ≤ V1 / V ∧ V1 / V ≤ 3 / 8 :=
by
  sorry

end volume_ratio_inequality_l96_96185


namespace evaluate_composite_function_l96_96758

def f (x : ℝ) : ℝ := x^2 - 2 * x + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_composite_function :
  f (g (-2)) = 26 := by
  sorry

end evaluate_composite_function_l96_96758


namespace pq_truth_values_l96_96782

theorem pq_truth_values (p q : Prop) (h1 : p ∨ q) (h2 : ¬p) : p = false ∧ q = true :=
by
  have h3 : ¬q → p ∨ q := λ hq, or.inr (by contradiction)
  have h4 : ¬q := λ hq, h2 (by contradiction)
  have h5 : q := by_contradiction h4
  exact ⟨h2, h5⟩
sorry

end pq_truth_values_l96_96782


namespace math_problem_l96_96729

theorem math_problem (x : ℕ) (h : (2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x = 512)) : (x + 2) * (x - 2) = 32 :=
sorry

end math_problem_l96_96729


namespace same_terminal_side_angles_l96_96017

theorem same_terminal_side_angles (θ : ℝ) (k : ℤ):
  θ = -2010 :=
  
  -- Condition (1): The smallest positive angle
  let smallest_positive_angle := 150 in
  θ % 360 = smallest_positive_angle

  -- Condition (2): The largest negative angle
  let largest_negative_angle := -210 in
  (-360 * ↑((-2010 / 360).to_int + 1) - largest_negative_angle)  % 360 = largest_negative_angle

  -- Condition (3): Angles within the range -720° to 720°
  let angles_in_range := [-570, -210, 150, 510] in
  let angles_same_terminal_side :=  list.map (λ k, k * 360 + 150) [-2, -1, 0, 1]

  let angles_in_range_subset :=
    angles_same_terminal_side.all (λ angle, angle >= -720 ∧ angle < 720)

  ∧ angles_in_range_subset = true)

  sorry

end same_terminal_side_angles_l96_96017


namespace correct_option_is_C_l96_96565

-- Definitions for given conditions
def optionA (x y : ℝ) : Prop := 3 * x + 3 * y = 6 * x * y
def optionB (x y : ℝ) : Prop := 4 * x * y^2 - 5 * x * y^2 = -1
def optionC (x : ℝ) : Prop := -2 * (x - 3) = -2 * x + 6
def optionD (a : ℝ) : Prop := 2 * a + a = 3 * a^2

-- The proof statement to show that Option C is the correct calculation
theorem correct_option_is_C (x y a : ℝ) : 
  ¬ optionA x y ∧ ¬ optionB x y ∧ optionC x ∧ ¬ optionD a :=
by
  -- Proof not required, using sorry to compile successfully
  sorry

end correct_option_is_C_l96_96565


namespace vector_ratio_l96_96394

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem vector_ratio (h₀ : a ≠ 0 ∧ b ≠ 0)
  (h₁ : ∥a - b∥ = ∥a + 2 • b∥)
  (h₂ : ⟪a, b⟫ = - (1 / 4) * ∥a∥ * ∥b∥) :
  ∥a∥ / ∥b∥ = 2 :=
by
  sorry

end vector_ratio_l96_96394


namespace inequality_proof_l96_96169

theorem inequality_proof (a b : ℝ) (h : a + b > 0) : 
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := 
sorry

end inequality_proof_l96_96169


namespace total_attended_ball_lt_fifty_l96_96849

-- Conditions from part a)
variable (n : ℕ) -- number of ladies
variable (m : ℕ) -- number of gentlemen

def total_people (n m : ℕ) : ℕ := n + m

-- Quarter of the ladies not invited means three-quarters danced
def ladies_danced (n : ℕ) : ℕ := 3 * n / 4

-- Two-sevenths of the gentlemen did not invite anyone
def gents_invited (m : ℕ) : ℕ := 5 * m / 7

-- From the condition we have these relations
axiom ratio_relation (h : n * 3 / 4 = m * 5 / 7) : True

-- Prove the total number of people attended the ball
theorem total_attended_ball_lt_fifty 
  (h : n * 3 / 4 = m * 5 / 7) 
  (hnm_lt_50 : total_people n m < 50) : total_people n m = 41 := 
by
  sorry

end total_attended_ball_lt_fifty_l96_96849


namespace largest_perfect_square_factor_9240_l96_96554

theorem largest_perfect_square_factor_9240 :
  ∃ n : ℕ, n * n = 36 ∧ ∃ m : ℕ, m ∣ 9240 ∧ m = n * n :=
by
  -- We will construct the proof here using the prime factorization
  sorry

end largest_perfect_square_factor_9240_l96_96554


namespace lcm_18_20_25_l96_96333

-- Lean 4 statement to prove the smallest positive integer divisible by 18, 20, and 25 is 900
theorem lcm_18_20_25 : Nat.lcm (Nat.lcm 18 20) 25 = 900 :=
by
  sorry

end lcm_18_20_25_l96_96333


namespace parallel_condition_orthogonal_condition_l96_96396

noncomputable theory

-- Define the vectors a and b based on lambda
def vector_a (λ : ℚ) : ℚ × ℚ := (-1, 3 * λ)
def vector_b (λ : ℚ) : ℚ × ℚ := (5, λ - 1)

-- Define when two vectors are parallel
def parallel (a b : ℚ × ℚ) : Prop :=
  a.1 * b.2 = a.2 * b.1

-- Define the condition for orthogonality
def orthogonal (a b : ℚ × ℚ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Part 1: Prove that if a and b are parallel, then λ = 1/16
theorem parallel_condition (λ : ℚ) (h : parallel (vector_a λ) (vector_b λ)) : λ = 1 / 16 :=
sorry

-- Define the combined vectors for Part 2
def vector_c (λ : ℚ) : ℚ × ℚ := (2 * (-1) + 5, 2 * 3 * λ + (λ - 1))
def vector_d (λ : ℚ) : ℚ × ℚ := ((-1) - 5, 3 * λ - (λ - 1))

-- Part 2: Prove that if vector_c and vector_d are orthogonal, then λ = 1 or λ = -19/14
theorem orthogonal_condition (λ : ℚ) (h : orthogonal (vector_c λ) (vector_d λ)) : λ = 1 ∨ λ = -19 / 14 :=
sorry

end parallel_condition_orthogonal_condition_l96_96396


namespace find_angles_find_range_l96_96133

-- Part 1: Prove the angles A and B in the given triangle
theorem find_angles (a b : ℝ) (A B C : ℝ) 
  (h1 : C = π / 6) 
  (h2 : (a - b) / b = cos A / cos B + 1) 
  (h3 : 0 < A ∧ A < π) 
  (h4 : 0 < B ∧ B < π) 
  (h5 : 0 < C ∧ C < π): 
  A = 5 * π / 8 ∧ B = 5 * π / 24 := 
  sorry

-- Part 2: Prove the range of the given expression
theorem find_range (a b : ℝ) (A B : ℝ) 
  (h1 : A = 5 * π / 8) 
  (h2 : B = 5 * π / 24) 
  (h3 : 0 < B ∧ B < π / 4):
  ∃ t, (t = (sqrt 2) ∧ 4 * cos B - 1 / cos B = t) ∨ (t = 3 ∧ 4 * cos B - 1 / cos B = t) :=
  sorry

end find_angles_find_range_l96_96133


namespace symmetric_point_in_third_quadrant_l96_96441

-- Define a structure for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to find the symmetric point about the y-axis
def symmetric_about_y (P : Point) : Point :=
  Point.mk (-P.x) P.y

-- Define the original point P
def P : Point := { x := 3, y := -2 }

-- Define the symmetric point P' about the y-axis
def P' : Point := symmetric_about_y P

-- Define a condition to determine if a point is in the third quadrant
def is_in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- The theorem stating that the symmetric point of P about the y-axis is in the third quadrant
theorem symmetric_point_in_third_quadrant : is_in_third_quadrant P' :=
  by
  sorry

end symmetric_point_in_third_quadrant_l96_96441


namespace product_of_last_two_digits_l96_96416

theorem product_of_last_two_digits (A B : ℕ) (hn1 : 10 * A + B ≡ 0 [MOD 5]) (hn2 : A + B = 16) : A * B = 30 :=
sorry

end product_of_last_two_digits_l96_96416


namespace min_moves_to_balance_is_2_l96_96160

-- Define the initial conditions for the board configuration and movement rules
structure Board :=
  (rows : ℕ)
  (cols : ℕ)
  (tokens : ℕ → ℕ → ℕ) -- a function representing the number of tokens at each position (row, col)
  (move : (ℕ × ℕ) → (ℕ × ℕ) → Prop) -- a relation representing allowed moves

def initialBoard : Board :=
{ rows := 4,
  cols := 4,
  tokens := λ (r c : ℕ), if (r = 2 ∧ c = 3) ∨ (r = 2 ∧ c = 4) ∨ (r = 2 ∧ c = 5) ∨ (r = 3 ∧ c = 3) ∨ (r = 4 ∧ c = 3) then 1 else 0,
  move := λ (pos1 pos2 : ℕ × ℕ),
    let (r1, c1) := pos1 in
    let (r2, c2) := pos2 in
    (|r1 - r2| ≤ 1 ∧ |c1 - c2| ≤ 1) }

-- Define the goal to check that 2 moves are enough to balance the board with exactly 2 tokens per row and column
def minMovesToBalance (b : Board) : ℕ :=
if (∀ i : ℕ, i < b.rows → ∀ j : ℕ, j < b.cols → ∑ n in (0 : ℕ) → b.tokens i n = 2 ∧ ∑ m in (0 : ℕ) → b.tokens m j = 2) then 2 else 0

-- Main theorem statement
theorem min_moves_to_balance_is_2 : minMovesToBalance initialBoard = 2 :=
sorry -- The proof is omitted

end min_moves_to_balance_is_2_l96_96160


namespace greatest_integer_difference_l96_96779

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 7) : y - x = 3 :=
sorry

end greatest_integer_difference_l96_96779


namespace speed_of_man_l96_96605

-- Definition of the given conditions
def length_of_train : ℝ := 900
def time_to_cross : ℝ := 53.99568034557235
def speed_of_train : ℝ := 63 * (1000 / 3600)  -- Conversion from km/hr to m/s

-- Theorem statement: Proving the speed of the man
theorem speed_of_man (L : ℝ) (T : ℝ) (V_t : ℝ) (V_m : ℝ) 
  (hL : L = length_of_train) 
  (hT : T = time_to_cross) 
  (hVt : V_t = speed_of_train)
  (hVr : V_t - V_m = L / T) : 
  V_m ≈ 0.832 :=
by
  sorry

end speed_of_man_l96_96605


namespace largest_number_l96_96540

theorem largest_number (a b c : ℕ) (h1: a ≤ b) (h2: b ≤ c) 
  (h3: (a + b + c) = 90) (h4: b = 32) (h5: b = a + 4) : c = 30 :=
sorry

end largest_number_l96_96540


namespace ticket_at_door_cost_l96_96924

-- Definitions from the conditions
def total_tickets : ℕ := 800
def advanced_ticket_cost : ℝ := 14.50
def total_revenue : ℝ := 16640
def tickets_sold_at_door : ℕ := 672
def tickets_sold_advanced : ℕ := total_tickets - tickets_sold_at_door

-- Theorem statement based on the proof problem
theorem ticket_at_door_cost :
  let x := (total_revenue - (tickets_sold_advanced * advanced_ticket_cost)) / tickets_sold_at_door in
  x = 22 := by
  sorry

end ticket_at_door_cost_l96_96924


namespace factor_expression_l96_96012

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) := 
by 
sorry

end factor_expression_l96_96012


namespace num_solutions_sin_cubic_eq_l96_96674

noncomputable def sin_cubic_eq : ℝ → ℝ := λ x, 3 * (Real.sin x) ^ 3 - 7 * (Real.sin x) ^ 2 + 3 * (Real.sin x)

theorem num_solutions_sin_cubic_eq : (finset.filter (λ x, sin_cubic_eq x = 0) (finset.Icc 0 (2 * Real.pi))).card = 5 :=
sorry

end num_solutions_sin_cubic_eq_l96_96674


namespace number_of_votes_for_winner_l96_96581

-- Define the conditions
def total_votes : ℝ := 1000
def winner_percentage : ℝ := 0.55
def margin_of_victory : ℝ := 100

-- The statement to prove
theorem number_of_votes_for_winner :
  0.55 * total_votes = 550 :=
by
  -- We are supposed to provide the proof but it's skipped here
  sorry

end number_of_votes_for_winner_l96_96581


namespace math_proof_problem_l96_96362

noncomputable def problem_statement (α : ℝ) :=

-- Conditions
0 < α ∧ α < π / 2 ∧
cos (2 * π - α) - sin (π - α) = - sqrt 5 / 5

theorem math_proof_problem (α : ℝ)
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : cos (2 * π - α) - sin (π - α) = - sqrt 5 / 5) :
  (sin α + cos α = 3 * sqrt 5 / 5) ∧
  (2 * sin α * cos α - sin (π / 2 + α) + 1) / (1 - cot (3 * π / 2 - α)) = (sqrt 5 - 9) / 5 :=
sorry

end math_proof_problem_l96_96362


namespace log_base_0_3_not_increasing_l96_96518

noncomputable def f (x : ℝ) : ℝ := Real.logBase 0.3 (abs (x^2 - 6 * x + 5))

theorem log_base_0_3_not_increasing : 
  ∀ x : ℝ, x < 1 → f 0.3 x' > f 0.3 x :=
by
  sorry

end log_base_0_3_not_increasing_l96_96518


namespace malcom_cards_left_l96_96296

theorem malcom_cards_left (brandon_cards : ℕ) (h1 : brandon_cards = 20) (h2 : ∀ malcom_cards : ℕ, malcom_cards = brandon_cards + 8) (h3 : ∀ mark_cards : ℕ, mark_cards = (malcom_cards / 2)) : 
  malcom_cards - mark_cards = 14 := 
by 
  let malcom_cards := 28
  let mark_cards := (malcom_cards / 2)
  have h4 : mark_cards = 14, from rfl
  show malcom_cards - mark_cards = 14, from sorry

end malcom_cards_left_l96_96296


namespace find_a_l96_96062

theorem find_a 
  (x : ℤ) 
  (a : ℤ) 
  (h1 : x = 2) 
  (h2 : y = a) 
  (h3 : 2 * x - 3 * y = 5) : a = -1 / 3 := 
by 
  sorry

end find_a_l96_96062


namespace range_of_a_l96_96697

theorem range_of_a (a : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 3 → x^2 - a * x - 3 ≤ 0) ↔ (2 ≤ a) := by
  sorry

end range_of_a_l96_96697


namespace monotonic_if_and_only_if_l96_96384

theorem monotonic_if_and_only_if (a : ℝ) : 
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 ≤ x2 → 
    (sqrt (x1^2 + 4) + a * x1) ≤ (sqrt (x2^2 + 4) + a * x2)) ↔
  (a ∈ set.Iic (-1) ∨ a ∈ set.Ici 0) := 
by
  sorry

end monotonic_if_and_only_if_l96_96384


namespace sum_of_all_possible_n_values_l96_96278

theorem sum_of_all_possible_n_values :
  let Δ := Triangle.mk (Point.mk 0 0) (Point.mk 2 2) (Point.mk (8 * n) 0) in
  ∀ n : ℝ, (Δ.area_dividing_line (Line.mk 0 n 1)).equal_area →
  ∑_{n : ℝ, 2n^2 + n - 1 = 0} n = -1 / 2 :=
by
  -- Proof goes here
  sorry

end sum_of_all_possible_n_values_l96_96278


namespace expression_divisibility_l96_96167

theorem expression_divisibility (x y : ℝ) : 
  ∃ P : ℝ, (x^2 - x * y + y^2)^3 + (x^2 + x * y + y^2)^3 = (2 * x^2 + 2 * y^2) * P := 
by 
  sorry

end expression_divisibility_l96_96167


namespace solve_for_y_l96_96496

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l96_96496


namespace number_of_sheep_with_only_fleas_l96_96577

-- Definitions based on the conditions
variables {sheep : Type} (has_fleas has_lice : sheep → Prop)

-- Number of sheep
variable (S : Nat)

-- Number of sheep with both pests
def both_pests : Nat := 84

-- Derived from the problem conditions
axiom half_have_lice : ∃ n, n = S / 2
axiom have_lice : ∃ n, n = 94

-- The proof problem statement
theorem number_of_sheep_with_only_fleas :
  ∀ (only_fleas only_lice : Nat),
  (both_pests + only_lice = 94) ∧
  (both_pests + only_lice = S / 2) ∧
  S = 2 * 94 →
  only_fleas = 94 :=
by
  intros only_fleas only_lice h,
  sorry

end number_of_sheep_with_only_fleas_l96_96577


namespace quadratic_root_property_l96_96077

theorem quadratic_root_property (m p : ℝ) 
  (h1 : (p^2 - 2 * p + m - 1 = 0)) 
  (h2 : (p^2 - 2 * p + 3) * (m + 4) = 7)
  (h3 : ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ r1^2 - 2 * r1 + m - 1 = 0 ∧ r2^2 - 2 * r2 + m - 1 = 0) : 
  m = -3 :=
by 
  sorry

end quadratic_root_property_l96_96077


namespace find_ab_l96_96074

theorem find_ab 
  (a : ℝ)
  (b : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h_max : ∀ x : ℝ, x ∈ Icc (-3/2) 0 → x = 0 → b + a^(x^2 + 2*x) = 3)
  (h_min : ∀ x : ℝ, x ∈ Icc (-3/2) 0 → x = -1 → b + a^(x^2 + 2*x) = 5/2) :
  (a = 2 ∧ b = 2) ∨ (a = 2/3 ∧ b = 3/2) :=
sorry

end find_ab_l96_96074


namespace remaining_water_in_cooler_l96_96571

def convert_gallons_to_ounces (gallons: ℕ): ℕ := gallons * 128

def total_chairs (rows: ℕ) (chairs_per_row: ℕ): ℕ := rows * chairs_per_row

def water_needed (number_of_chairs: ℕ) (water_per_cup: ℕ): ℕ := number_of_chairs * water_per_cup

def water_left_in_cooler (initial_water: ℕ) (used_water: ℕ): ℕ := initial_water - used_water

theorem remaining_water_in_cooler:
  let gallons_in_cooler := 3 in
  let ounces_per_gallon := 128 in
  let ounces_in_cooler := convert_gallons_to_ounces gallons_in_cooler in
  let rows := 5 in
  let chairs_per_row := 10 in
  let total_chairs := total_chairs rows chairs_per_row in
  let ounces_per_cup := 6 in
  let needed_water := water_needed total_chairs ounces_per_cup in
  water_left_in_cooler ounces_in_cooler needed_water = 84 :=
by
  sorry

end remaining_water_in_cooler_l96_96571


namespace number_of_internal_cubes_l96_96536

theorem number_of_internal_cubes :
  ∀ l w h : ℕ,
    w = 2 * l →
    w = 2 * h →
    w = 6 →
    l * w * h = 54 :=
by
  intros l w h h_w2l h_w2h h_w
  rw [h_w2l, h_w2h] at h_w
  have l_value := (h_w.symm ▸ rfl : l = 3)
  have h_value := (h_w.symm ▸ rfl : h = 3)
  rw [l_value, h_w, h_value]
  norm_num

end number_of_internal_cubes_l96_96536


namespace purchase_combinations_l96_96484

theorem purchase_combinations :
  ∃ (ways : Nat), 
  (∀ (x y : Nat), 
      (x ≥ 8) → 
      (y ≥ 2) → 
      (120 * x + 140 * y ≤ 1500) → 
         (ways = (if ((120 * 8 + 140 * 2 ≤ 1500) + 
                      (120 * 8 + 140 * 3 ≤ 1500) + 
                      (120 * 9 + 140 * 2 ≤ 1500) + 
                      (120 * 9 + 140 * 3 ≤ 1500) + 
                      (120 * 10 + 140 * 2 ≤ 1500) 
                    ) then 5 else 0)
  ) :=
begin
  existsi 5,
  intros x y hx hy hbudget,
  repeat {cases hx, cases hy, cases hbudget},
  sorry
end

end purchase_combinations_l96_96484


namespace ball_attendance_l96_96858

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l96_96858


namespace parallel_vectors_m_eq_neg3_l96_96082

theorem parallel_vectors_m_eq_neg3 : 
  ∀ m : ℝ, (∀ (a b : ℝ × ℝ), a = (1, -2) → b = (1 + m, 1 - m) → a.1 * b.2 - a.2 * b.1 = 0) → m = -3 :=
by
  intros m h_par
  specialize h_par (1, -2) (1 + m, 1 - m) rfl rfl
  -- We need to show m = -3
  sorry

end parallel_vectors_m_eq_neg3_l96_96082


namespace most_reasonable_sampling_method_l96_96261

-- Conditions
def student_grades := {first_grade, second_grade, third_grade}
def significant_differences_in_understanding (grade1 grade2 : student_grades) : Prop := 
true -- It states there are significant differences in understanding among grades.

-- Stratified sampling method
def stratified_sampling : Prop := true

-- The question is to identify the most reasonable sampling method, given the conditions
theorem most_reasonable_sampling_method : stratified_sampling :=
sorry

end most_reasonable_sampling_method_l96_96261


namespace pies_not_eaten_with_forks_l96_96113

variables (apple_pe_forked peach_pe_forked cherry_pe_forked chocolate_pe_forked lemon_pe_forked : ℤ)
variables (total_pies types_of_pies : ℤ)

def pies_per_type (total_pies types_of_pies : ℤ) : ℤ :=
  total_pies / types_of_pies

def not_eaten_with_forks (percentage_forked : ℤ) (pies : ℤ) : ℤ :=
  pies - (pies * percentage_forked) / 100

noncomputable def apple_not_forked  := not_eaten_with_forks apple_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def peach_not_forked  := not_eaten_with_forks peach_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def cherry_not_forked := not_eaten_with_forks cherry_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def chocolate_not_forked := not_eaten_with_forks chocolate_pe_forked (pies_per_type total_pies types_of_pies)
noncomputable def lemon_not_forked := not_eaten_with_forks lemon_pe_forked (pies_per_type total_pies types_of_pies)

theorem pies_not_eaten_with_forks :
  (apple_not_forked = 128) ∧
  (peach_not_forked = 112) ∧
  (cherry_not_forked = 84) ∧
  (chocolate_not_forked = 76) ∧
  (lemon_not_forked = 140) :=
by sorry

end pies_not_eaten_with_forks_l96_96113


namespace base_comparison_l96_96471

theorem base_comparison 
  (a b n : ℕ) 
  (h₁ : a > 1) 
  (h₂ : b > 1) 
  (h₃ : n > 1) 
  (h₄ : a > b)
  (x : ℕ → ℕ)
  (hxn : x n ≠ 0)
  (hxnm1 : x (n - 1) ≠ 0) :
  let Aₙ₋₁ := ∑ i in Finset.range n, x i * a^i,
      Aₙ := ∑ i in Finset.range (n + 1), x i * a^i,
      Bₙ₋₁ := ∑ i in Finset.range n, x i * b^i,
      Bₙ := ∑ i in Finset.range (n + 1), x i * b^i
  in (Aₙ₋₁ / Aₙ : ℚ) < (Bₙ₋₁ / Bₙ : ℚ) :=
by
  sorry

end base_comparison_l96_96471


namespace phase_shift_of_sine_function_l96_96332

theorem phase_shift_of_sine_function :
  ∀ x : ℝ, y = 3 * Real.sin (3 * x + π / 4) → (∃ φ : ℝ, φ = -π / 12) :=
by sorry

end phase_shift_of_sine_function_l96_96332


namespace percentage_problem_l96_96250

noncomputable def percentage_of_value (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
  (y / x) * 100

theorem percentage_problem :
  percentage_of_value 2348 (528.0642570281125 * 4.98) = 112 := 
by
  sorry

end percentage_problem_l96_96250


namespace total_students_l96_96430

-- Define the conditions
variables (S : ℕ) -- total number of students
variable (h1 : (3/5 : ℚ) * S + (1/5 : ℚ) * S + 10 = S)

-- State the theorem
theorem total_students (HS : S = 50) : 3 / 5 * S + 1 / 5 * S + 10 = S := by
  -- Here we declare the proof is to be filled in later.
  sorry

end total_students_l96_96430


namespace candies_shared_l96_96165

theorem candies_shared (y b d x : ℕ) (h1 : x = 2 * y + 10) (h2 : x = 3 * b + 18) (h3 : x = 5 * d - 55) (h4 : x + y + b + d = 2013) : x = 990 :=
by
  sorry

end candies_shared_l96_96165


namespace susie_total_earnings_l96_96507

def pizza_prices (type : String) (is_whole : Bool) : ℝ :=
  match type, is_whole with
  | "Margherita", false => 3
  | "Margherita", true  => 15
  | "Pepperoni", false  => 4
  | "Pepperoni", true   => 18
  | "Veggie Supreme", false => 5
  | "Veggie Supreme", true  => 22
  | "Meat Lovers", false => 6
  | "Meat Lovers", true  => 25
  | "Hawaiian", false   => 4.5
  | "Hawaiian", true    => 20
  | _, _                => 0

def topping_price (is_weekend : Bool) : ℝ :=
  if is_weekend then 1 else 2

def happy_hour_price : ℝ := 3

noncomputable def susie_earnings : ℝ :=
  let margherita_slices := 12 * happy_hour_price + 12 * pizza_prices "Margherita" false
  let pepperoni_slices := 8 * happy_hour_price + 8 * pizza_prices "Pepperoni" false + 6 * topping_price true
  let veggie_supreme_pizzas := 4 * pizza_prices "Veggie Supreme" true + 8 * topping_price true
  let margherita_whole_discounted := 3 * pizza_prices "Margherita" true - (3 * pizza_prices "Margherita" true) * 0.1
  let meat_lovers_slices := 10 * happy_hour_price + 10 * pizza_prices "Meat Lovers" false
  let hawaiian_slices := 12 * pizza_prices "Hawaiian" false + 4 * topping_price true
  let pepperoni_whole := pizza_prices "Pepperoni" true + 3 * topping_price true
  margherita_slices + pepperoni_slices + veggie_supreme_pizzas + margherita_whole_discounted + meat_lovers_slices + hawaiian_slices + pepperoni_whole

theorem susie_total_earnings : susie_earnings = 439.5 := by
  sorry

end susie_total_earnings_l96_96507


namespace solve_for_N_l96_96211

theorem solve_for_N (a b c N : ℝ) 
  (h1 : a + b + c = 72) 
  (h2 : a - 7 = N) 
  (h3 : b + 7 = N) 
  (h4 : 2 * c = N) : 
  N = 28.8 := 
sorry

end solve_for_N_l96_96211


namespace perimeter_of_ABCD_is_as_expected_l96_96444

def is_right_triangle (A B C : Triangle) : Prop := 
  A.angle B C = 90 

def is_30_60_90_triangle (A B C : Triangle) : Prop := 
  A.angle B C = 90 ∧ A.angle C = 60 ∧ C.angle A = 30

structure Triangle : Type :=
  (vertices : Fin 3 → Point)
  (angle : Fin 3 → Fin 3 → ℝ)

structure Point : Type :=
  (x : ℝ)
  (y : ℝ)

noncomputable def AE : ℝ := 36

noncomputable def BE : ℝ := 18 -- half of AE
noncomputable def CE : ℝ := 9  -- half of BE
noncomputable def DE : ℝ := 4.5-- half of CE
noncomputable def AB : ℝ := 18 * Real.sqrt 3 -- sqrt(3)/2 of AE
noncomputable def BC : ℝ := 9 * Real.sqrt 3  -- sqrt(3)/2 of BE
noncomputable def CD : ℝ := 4.5 * Real.sqrt 3  -- sqrt(3)/2 of CE
noncomputable def DA : ℝ := DE + AE -- 4.5 + 36
noncomputable def perimeter : ℝ := AB + BC + CD + DA

theorem perimeter_of_ABCD_is_as_expected : perimeter = 40.5 + 31.5 * Real.sqrt 3 :=
by sorry

end perimeter_of_ABCD_is_as_expected_l96_96444


namespace shortest_distance_and_midpoint_coordinates_l96_96080

variable {p l : ℝ}

theorem shortest_distance_and_midpoint_coordinates
  (hp : p > 0)
  (parabola : ∀ (y x : ℝ), y^2 = 2 * p * x)
  (chord_length : ∀ (y1 y2 : ℝ), (y1 - y2)^2 * (p^2 + (y1 + y2)^2) / (4 * p^2) = l^2) :
  let M : ℝ × ℝ := (l^2 / (8 * p), 0)
  in M.1 = l^2 / (8 * p) ∧ l^2 / (8 * p) = l^2 / (8 * p) := by
  sorry

end shortest_distance_and_midpoint_coordinates_l96_96080


namespace geometric_sequence_term_l96_96810

noncomputable def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_term {a : ℕ → ℤ} {q : ℤ}
  (h1 : geometric_sequence a q)
  (h2 : a 7 = 10)
  (h3 : q = -2) :
  a 10 = -80 :=
by
  sorry

end geometric_sequence_term_l96_96810


namespace smallest_three_digit_number_divisible_by_6_5_8_9_l96_96228

open Nat

theorem smallest_three_digit_number_divisible_by_6_5_8_9 :
  ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % lcm (lcm (lcm 6 5) 8) 9 = 0 ∧ n = 360 :=
by
  sorry

end smallest_three_digit_number_divisible_by_6_5_8_9_l96_96228


namespace feet_to_inches_conversion_l96_96191

-- Define the constant equivalence between feet and inches
def foot_to_inches := 12

-- Prove the conversion factor between feet and inches
theorem feet_to_inches_conversion:
  foot_to_inches = 12 :=
by
  sorry

end feet_to_inches_conversion_l96_96191


namespace min_triangle_area_l96_96732

theorem min_triangle_area :
  ∀ (A1 A2 A3 A4 : ℝ × ℝ),
    (A1.2 ^ 2 = 4 * A1.1) →
    (A2.2 ^ 2 = 4 * A2.1) →
    (A3.2 ^ 2 = 4 * A3.1) →
    (A4.2 ^ 2 = 4 * A4.1) →
    (A1.2 * (A3.1 - 1) = A1.1 * A3.2 - 4 * A1.1 * A3.1) →
    (A2.2 * (A4.1 - 1) = A2.1 * A4.2 - 4 * A2.1 * A4.1) →
    let M := ((A1.1 * A2.2 - A2.1 * A1.2) / (A1.2 - A2.2), (A3.1 * A4.2 - A4.1 * A3.2) / (A3.2 - A4.2)) in
    let N := ((A1.1 * A4.2 - A4.1 * A1.2) / (A1.2 - A4.2), (A2.1 * A3.2 - A3.1 * A2.2) / (A2.2 - A3.2)) in
    let area := (1 / 2) * abs ((M.1 - N.1) * 2) in
    area ≥ 4 :=
sorry

end min_triangle_area_l96_96732


namespace factor_expression_l96_96009

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l96_96009


namespace intersection_y_condition_l96_96789

theorem intersection_y_condition (a : ℝ) :
  (∃ x y : ℝ, 2 * x - a * y + 2 = 0 ∧ x + y = 0 ∧ y < 0) → a < -2 :=
by
  sorry

end intersection_y_condition_l96_96789


namespace ball_total_attendance_l96_96846

theorem ball_total_attendance :
    ∃ n m : ℕ, n + m = 41 ∧
    (n + m < 50) ∧
    (3 * n / 4 = (5 * m / 7)) :=
begin
  sorry
end

end ball_total_attendance_l96_96846


namespace members_taken_course_not_passed_l96_96957

def swim_club_total_members : ℕ := 100
def swim_club_percentage_passed : ℚ := 0.30
def swim_club_not_taken_course : ℕ := 30

theorem members_taken_course_not_passed :
  let T := swim_club_total_members in
  let P := T * swim_club_percentage_passed in
  let N := T - P in
  let C := N - swim_club_not_taken_course in
  C = 40 :=
by
  sorry

end members_taken_course_not_passed_l96_96957


namespace chord_length_intercepted_by_line_l96_96931

-- Definitions from conditions
def center : ℝ × ℝ := (3, 1)
def radius : ℝ := 5
def circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 25
def line (x y : ℝ) : Prop := x + 2*y = 0

-- The theorem statement
theorem chord_length_intercepted_by_line :
  ∃ l : ℝ, l = 4 * Real.sqrt 5 ∧ ∃ x1 y1 x2 y2 : ℝ, circle x1 y1 ∧ circle x2 y2 ∧ line x1 y1 ∧ line x2 y2 ∧ Real.dist (x1, y1) (x2, y2) = l :=
sorry

end chord_length_intercepted_by_line_l96_96931


namespace projection_of_a_in_a_plus_b_is_sqrt_two_over_two_l96_96365

variable {V : Type} [InnerProductSpace ℝ V]

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

theorem projection_of_a_in_a_plus_b_is_sqrt_two_over_two
  (a b : V) (h1 : is_unit_vector a) (h2 : is_unit_vector b)
  (h3 : ∥a + b∥ = ∥a - b∥) :
  (inner a (a + b)) / ∥a + b∥ = (real.sqrt 2) / 2 :=
by sorry

end projection_of_a_in_a_plus_b_is_sqrt_two_over_two_l96_96365


namespace sum_of_series_l96_96030

section GeometricSeries

-- Conditions
variables {r b : ℝ} (T : ℝ → ℝ) 
hypothesis h_r : -1 < r ∧ r < 1
hypothesis h_b : -1 < b ∧ b < 1
hypothesis h_T_series : ∀ (r : ℝ), T(r) = 18 / (1 - r)
hypothesis h_TbTb_neg_b : T(b) * T(-b) = 2916

-- The proof we want to show
theorem sum_of_series (h_T_series : ∀ (r : ℝ), T(r) = 18 / (1 - r))
 (h_b : -1 < b ∧ b < 1) (h_TbTb_neg_b : T(b) * T(-b) = 2916) : 
 T(b) + T(-b) = 324 := 
sorry

end GeometricSeries

end sum_of_series_l96_96030


namespace tickets_sold_at_reduced_price_first_week_l96_96314

variable (T : ℕ) (F : ℕ) (R : ℕ)
variable hT : T = 25200
variable hF : F = 16500
variable hR : T - F = R
variable hFullPriceMultiple : F = 5 * R

theorem tickets_sold_at_reduced_price_first_week : 
  R = 8700 :=
by
  rw [hT, hF, hR, hFullPriceMultiple] at *
  sorry

end tickets_sold_at_reduced_price_first_week_l96_96314


namespace similar_triangles_side_proportionality_l96_96962

theorem similar_triangles_side_proportionality (X Y Z P Q R : Type) 
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [Triangle X Y Z] [Triangle P Q R]
  (h1 : similar X Y Z P Q R)
  (h2 : dist Y Z = 12)
  (h3 : dist P Q = 6)
  (h4 : dist X R = 9) :
  dist Q R = 4.5 :=
by
  sorry

end similar_triangles_side_proportionality_l96_96962


namespace product_complex_numbers_l96_96775

noncomputable def Q : ℂ := 3 + 4 * Complex.I
noncomputable def E : ℂ := 2 * Complex.I
noncomputable def D : ℂ := 3 - 4 * Complex.I
noncomputable def R : ℝ := 2

theorem product_complex_numbers : Q * E * D * (R : ℂ) = 100 * Complex.I := by
  sorry

end product_complex_numbers_l96_96775


namespace solution_set_ineq_l96_96189

variable {f : ℝ → ℝ}
variable (f_diff : ∀ x > 0, differentiable_at ℝ f x)
variable (f'_ineq : ∀ x > 0, has_deriv_at f (deriv f x) x)
variable (cond : ∀ x > 0, deriv f x + (2 / x) * f x > 0)

theorem solution_set_ineq (x : ℝ) (hx : -1 < x ∧ x < 1) :
  ((x + 1) * f(x + 1)) / 4 < f 2 / (x + 1) :=
  sorry

end solution_set_ineq_l96_96189


namespace prime_divisors_of_30_factorial_l96_96087

-- Define the list of all prime numbers less than or equal to 30.
def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- The main theorem stating the number of prime divisors of 30!.
theorem prime_divisors_of_30_factorial :
  (primes_up_to_30.count (λ p, Nat.Prime p ∧ p ∣ Nat.factorial 30)) = 10 :=
by
  sorry

end prime_divisors_of_30_factorial_l96_96087


namespace find_b_squared_l96_96615

theorem find_b_squared (a b : ℝ)
  (h1 : ∀ z : ℂ, abs ((a + b * complex.I) * z - z) = abs ((a + b * complex.I) * z))
  (h2 : abs (a + b * complex.I) = 5) : b^2 = 99 / 4 :=
sorry

end find_b_squared_l96_96615


namespace filling_time_with_ab_l96_96318

theorem filling_time_with_ab (a b c l : ℝ) (h1 : a + b + c - l = 5 / 6) (h2 : a + c - l = 1 / 2) (h3 : b + c - l = 1 / 3) : 
  1 / (a + b) = 1.2 :=
by
  sorry

end filling_time_with_ab_l96_96318


namespace _l96_96161

noncomputable def polynomial_distance_theorem (P : Polynomial ℤ) (x1 x2 : ℤ) :
  x1 ≠ x2 ∧ P.eval x1 ∈ ℤ ∧ P.eval x2 ∈ ℤ ∧ 
  ∃ d : ℤ, d = Int.sqrt ((x1 - x2)^2 + (P.eval x1 - P.eval x2)^2) →
  P.eval x1 = P.eval x2 := 
by 
  sorry

end _l96_96161


namespace min_value_of_a_l96_96754

theorem min_value_of_a (a b c : ℝ) (h₁ : a > 0) (h₂ : ∃ p q : ℝ, 0 < p ∧ p < 2 ∧ 0 < q ∧ q < 2 ∧ 
  ∀ x, ax^2 + bx + c = a * (x - p) * (x - q)) (h₃ : 25 * a + 10 * b + 4 * c ≥ 4) (h₄ : c ≥ 1) : 
  a ≥ 16 / 25 :=
sorry

end min_value_of_a_l96_96754


namespace volume_of_tetrahedron_l96_96005

-- Define the given conditions of the problem.
def area_ABC := 150
def area_BCD := 90
def BC := 12
def angle_ABC_BCD := Real.pi / 4  -- 45 degrees in radians

-- Based on these conditions, the goal is to prove the volume of the tetrahedron.
theorem volume_of_tetrahedron :
  let h := 2 * area_BCD / BC in
  let h' := h * Real.sin angle_ABC_BCD in
  (1 / 3) * area_ABC * h' = 375 * Real.sqrt 2 :=
by
  let h := 2 * area_BCD / BC
  let h' := h * Real.sin angle_ABC_BCD
  have h_def : h = 15 := sorry  -- height from D to BC calculated in step 1 of solution
  have h'_def : h' = 15 * Real.sqrt 2 / 2 := sorry -- height from D to plane of ABC
  have volume_def : (1 / 3 * area_ABC * h') = 375 * Real.sqrt 2 := sorry 
  -- volume calculation as in step 3 of solution
  exact volume_def

end volume_of_tetrahedron_l96_96005


namespace find_B_l96_96879

def is_divisible_by (n k : ℕ) : Prop := n % k = 0

def digit_sum (n : ℕ) : ℕ := if n < 10 then n else digit_sum (n / 10) + (n % 10)

def satisfies_conditions (B : ℕ) : Prop :=
  let n := 35380840 + B in
  is_divisible_by n 2 ∧
  is_divisible_by n 4 ∧
  is_divisible_by n 5 ∧
  is_divisible_by n 6 ∧
  is_divisible_by n 8 ∧
  is_divisible_by (digit_sum n) 9

theorem find_B : satisfies_conditions 0 :=
  by
    sorry

end find_B_l96_96879


namespace typing_speed_equation_l96_96232

theorem typing_speed_equation (x : ℕ) (h_pos : x > 0) :
  120 / x = 180 / (x + 6) :=
sorry

end typing_speed_equation_l96_96232


namespace inequality_does_not_hold_l96_96613

noncomputable def f : ℝ → ℝ := sorry -- define f satisfying the conditions from a)

theorem inequality_does_not_hold :
  (∀ x, f (-x) = f x) ∧ -- f is even
  (∀ x, f x = f (x + 2)) ∧ -- f is periodic with period 2
  (∀ x, 3 ≤ x ∧ x ≤ 4 → f x = 2^x) → -- f(x) = 2^x when x is in [3, 4]
  ¬ (f (Real.sin 3) < f (Real.cos 3)) := by
  -- skipped proof
  sorry

end inequality_does_not_hold_l96_96613


namespace find_width_of_jordan_rectangle_l96_96655

theorem find_width_of_jordan_rectangle (width : ℕ) (h1 : 12 * 15 = 9 * width) : width = 20 :=
by
  sorry

end find_width_of_jordan_rectangle_l96_96655


namespace largest_numbers_among_options_l96_96688

-- Definitions of the options as real numbers
def optionA := 8.12334
def optionB := real.mk (8 + 123/1000 + 3/10000 / (1 - 1/10)) -- 8.1233333...
def optionC := real.mk (8 + 123/1000 + 3/10000 / (1 - 1/100)) -- 8.1233333...
def optionD := real.mk (8 + 1/10 + 233/10000 / (1 - 1/1000)) -- 8.12332332...
def optionE := real.mk (8 + 1233/10000 / (1 - 1/10000)) -- 8.12331233...

-- The proof statement
theorem largest_numbers_among_options : (optionB = real.mk (8 + 123/1000 + 3/10000 / (1 - 1/10))) ∧ (optionC = real.mk (8 + 123/1000 + 3/10000 / (1 - 1/100))) := 
sorry

end largest_numbers_among_options_l96_96688


namespace top_and_bottom_area_each_l96_96890

def long_side_area : ℕ := 2 * 8 * 6
def short_side_area : ℕ := 2 * 5 * 6
def total_sides_area : ℕ := long_side_area + short_side_area
def total_needed_area : ℕ := 236
def top_and_bottom_area : ℕ := total_needed_area - total_sides_area

theorem top_and_bottom_area_each :
  top_and_bottom_area / 2 = 40 := by
  sorry

end top_and_bottom_area_each_l96_96890


namespace dot_product_condition_l96_96106

variables (a b : ℝ^3)

theorem dot_product_condition 
  (h1 : ∥a + b∥ = real.sqrt 10)
  (h2 : ∥a - b∥ = real.sqrt 6) : 
  (a ⬝ b) = 1 :=
sorry

end dot_product_condition_l96_96106


namespace total_truck_loads_needed_l96_96269

noncomputable def truck_loads_of_material : ℝ :=
  let sand := 0.16666666666666666 * Real.pi
  let dirt := 0.3333333333333333 * Real.exp 1
  let cement := 0.16666666666666666 * Real.sqrt 2
  let gravel := 0.25 * Real.log 5 -- log is the natural logarithm in Lean
  sand + dirt + cement + gravel

theorem total_truck_loads_needed : truck_loads_of_material = 1.8401374808985008 := by
  sorry

end total_truck_loads_needed_l96_96269


namespace solution1_solution2_l96_96361

noncomputable def problem1 (a : ℝ) : Prop :=
  (∃ x : ℝ, -2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + 4 < 0)

theorem solution1 (a : ℝ) : problem1 a ↔ a < -3 ∨ a ≥ -1 := 
  sorry

noncomputable def problem2 (a : ℝ) (x : ℝ) : Prop :=
  (-2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0)

noncomputable def condition2 (a x : ℝ) : Prop :=
  (2*a < x ∧ x < a+1)

theorem solution2 (a : ℝ) : (∀ x, condition2 a x → problem2 a x) → a ≥ -1/2 :=
  sorry

end solution1_solution2_l96_96361


namespace ball_attendance_l96_96842

theorem ball_attendance
  (n m : ℕ)
  (h_total : n + m < 50)
  (h_ladies_dance : 3 * n = 4 * (Ladies who danced invited)
  (h_gentlemen_invited : 5 * m = 7 * (Gentlemen who invited)
  (h_dance_pairs : 3 * n = 20 * m) :
  n + m = 41 :=
by sorry

end ball_attendance_l96_96842


namespace intersection_A_B_l96_96757

def A : set ℝ := {x | x^2 - 11 * x - 12 < 0}
def B : set ℝ := {x | ∃ n : ℤ, x = 3 * n + 1}

theorem intersection_A_B : A ∩ B = {1, 4, 7, 10} := by
  sorry

end intersection_A_B_l96_96757


namespace sufficient_condition_l96_96241

/-- Define the condition 'log_1/2(x + 2) < 0' for the Lean environment --/
def condition_log (x : ℝ) : Prop := log (1 / 2) (x + 2) < 0

/-- Define the predicate 'x > 1' for the Lean environment --/
def greater_than_one (x : ℝ) : Prop := x > 1

/-- Define the statement 'sufficient but not necessary' condition --/
def sufficient_but_not_necessary (P Q : Prop) : Prop := (P → Q) ∧ ¬(Q → P)

/-- The proof problem: Prove that 'greater_than_one' is a sufficient but not necessary condition for 'condition_log' --/
theorem sufficient_condition (x : ℝ) : 
  sufficient_but_not_necessary (greater_than_one x) (condition_log x) :=
  sorry

end sufficient_condition_l96_96241


namespace counterexample_function_l96_96233

noncomputable def f : ℝ → ℝ := λ x, 3^x - 1

theorem counterexample_function :
  (∀ x : ℝ, f x > -1) ∧ ¬(∃ m : ℝ, ∀ x : ℝ, f x ≥ m) := by
  dsimp only [f]
  sorry

end counterexample_function_l96_96233


namespace potato_slab_solution_l96_96635

theorem potato_slab_solution (x y : ℝ) (hx_cond : x^2 - y^2 = 2000) (hy_cond : x + y = 600) :
  x ≈ 301.67 ∧ y ≈ 298.33 :=
by sorry

end potato_slab_solution_l96_96635


namespace constant_a_value_l96_96751

theorem constant_a_value 
  (a : ℝ) 
  (h : ∀ x : ℝ, abs ((sin x) ^ 2 - 4 * sin x - a) ≤ 4) :
  a = 1 :=
by
  sorry

end constant_a_value_l96_96751


namespace probability_correct_predictions_monday_l96_96593

def number_of_classes_monday : ℕ := 5
def number_of_classes_tuesday : ℕ := 6
def total_classes : ℕ := number_of_classes_monday + number_of_classes_tuesday

def total_correct_predictions : ℕ := 7
def correct_predictions_monday : ℕ := 3
def correct_predictions_tuesday : ℕ := total_correct_predictions - correct_predictions_monday

noncomputable def binomial_coefficient (n k : ℕ) : ℝ := (nat.choose n k : ℝ)

noncomputable def probability_exact_correct_monday (n k : ℕ) : ℝ :=
  binomial_coefficient n k * (real.exp 11 (ln 2⁻¹ * total_classes))

theorem probability_correct_predictions_monday :
  probability_exact_correct_monday number_of_classes_monday correct_predictions_monday *
  probability_exact_correct_monday number_of_classes_tuesday correct_predictions_tuesday /
  probability_exact_correct_monday total_classes total_correct_predictions = 5 / 11 :=
sorry

end probability_correct_predictions_monday_l96_96593


namespace evaluate_expression_l96_96001

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l96_96001


namespace ways_to_go_from_first_to_fifth_l96_96951

theorem ways_to_go_from_first_to_fifth (floors : ℕ) (staircases_per_floor : ℕ) (total_ways : ℕ) 
    (h1 : floors = 5) (h2 : staircases_per_floor = 2) (h3 : total_ways = 2^4) : total_ways = 16 :=
by
  sorry

end ways_to_go_from_first_to_fifth_l96_96951


namespace count_arrangements_with_conditions_l96_96541

theorem count_arrangements_with_conditions : 
  (∃ (students teachers : set ℕ), students.card = 3 ∧ teachers.card = 2 ∧
    ∀ a b ∈ teachers, a ≠ b →
    ((bundle teachers)) ∧ 
    (∃ (A B : ℕ), A ≠ B ∧ 
      ∀ perm : list ℕ, perm.card = 5 →
      (A stands to the left of B in perm) →
      (both stand next to each other in perm)) →
    (∃ perm : list ℕ, perm.card = 5 ∧ count_permutations perm = 24).

end count_arrangements_with_conditions_l96_96541


namespace negation_proposition_l96_96945

theorem negation_proposition :
  (¬ (∀ x : ℝ, ∃ n : ℕ, n > 0 ∧ n ≥ x)) ↔ (∃ x : ℝ, ∀ n : ℕ, n > 0 → n < x^2) := 
by
  sorry

end negation_proposition_l96_96945


namespace infinite_solution_triples_l96_96450

theorem infinite_solution_triples : 
  ∃ f : ℤ → ℤ × ℤ × ℤ, (∀ k, let (a, b, c) := f k in a^2 + b^2 = 2 * (c^2 + 1) ∧ ∀ m n : ℤ, f m ≠ f n → m ≠ n) :=
by
  sorry

end infinite_solution_triples_l96_96450


namespace problem_part1_problem_part2_l96_96388

open Real

noncomputable section

def hyperbola : set (ℝ × ℝ) := { p | p.1^2 - p.2^2 / 4 = 1 }

def focus1 : ℝ × ℝ := (sqrt 5, 0)
def focus2 : ℝ × ℝ := (-sqrt 5, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem_part1 (A B : ℝ × ℝ)
  (tangent_right : ∀ p ∈ hyperbola, p.1 > 0 → distance 𝑂 p ≤ distance A p + distance B p)
  (intersects_asymptotes : A.2 = 2 * A.1 ∧ B.2 = -2 * B.1) :
  distance (0,0) A * distance (0,0) B = distance (0,0) focus1 ^ 2 :=
sorry

theorem problem_part2 (A B : ℝ × ℝ)
  (tangent_right : ∀ p ∈ hyperbola, p.1 > 0 → distance 𝑂 p ≤ distance A p + distance B p)
  (intersects_asymptotes : A.2 = 2 * A.1 ∧ B.2 = -2 * B.1) :
  ∃ (C : ℝ × ℝ), circle_through C focus1 focus2 A B :=
sorry

end problem_part1_problem_part2_l96_96388


namespace edward_mowed_lawns_l96_96678

theorem edward_mowed_lawns (L : ℕ) (h1 : 8 * L + 7 = 47) : L = 5 :=
by
  sorry

end edward_mowed_lawns_l96_96678


namespace square_divided_equal_areas_l96_96631

variables {S S1 S2 S3 S4 : ℝ}

/-- A square is divided into four parts by two perpendicular lines whose intersection point lies 
inside the square. Prove that if the areas of three of these parts are equal, then the areas of all four parts 
are equal. -/
theorem square_divided_equal_areas (hS : S = S1 + S2 + S3 + S4)
    (h_perp_lines : ⟨S_1, S2, S3, S4⟩ --TODO: more geometric definition leads to perpendicular squares
    (h_areas_equal : S1 = S2 ∧ S2 = S3) : S1 = S2 ∧ S2 = S3 ∧ S3 = S4 := 
by
  sorry

end square_divided_equal_areas_l96_96631


namespace simplify_tan_product_l96_96909

noncomputable def tan_deg (d : ℝ) : ℝ := Real.tan (d * Real.pi / 180)

theorem simplify_tan_product :
  (1 + tan_deg 10) * (1 + tan_deg 35) = 2 := 
by
  -- Given conditions
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have h2 : tan_deg 10 + tan_deg 35 = 1 - tan_deg 10 * tan_deg 35 :=
    by sorry -- Use tan addition formula here
  -- Proof of the theorem follows from here
  sorry

end simplify_tan_product_l96_96909


namespace angle_measure_parallel_lines_l96_96129

theorem angle_measure_parallel_lines
  (m n : Line)
  (transversal : Line)
  (angle_x angle_adjacent : ℝ)
  (h_parallel : IsParallel m n)
  (h_transversal : IsTransversal transversal m n)
  (h_adjacent : angle_adjacent = 45)
  (h_ninety_degrees : TransversalFormsAngleWithLine transversal n 90) :
  angle_x = 45 := 
sorry

end angle_measure_parallel_lines_l96_96129


namespace Dan_age_is_28_l96_96291

-- Definitions based on conditions
variables (Ben_age Dan_age : ℕ)
variable (S : ℕ)
hypothesis h1 : Ben_age = 25
hypothesis h2 : S = 53
hypothesis h3 : S = Ben_age + Dan_age

-- Proof problem statement
theorem Dan_age_is_28 : Dan_age = 28 :=
by
  -- ⊢ Dan_age = 28
  sorry

end Dan_age_is_28_l96_96291


namespace average_age_of_girls_l96_96114

theorem average_age_of_girls (total_students : ℕ) (boys_avg_age : ℝ) (school_avg_age : ℚ)
    (girls_count : ℕ) (total_age_school : ℝ) (boys_count : ℕ) 
    (total_age_boys : ℝ) (total_age_girls : ℝ): (total_students = 640) →
    (boys_avg_age = 12) →
    (school_avg_age = 47 / 4) →
    (girls_count = 160) →
    (total_students - girls_count = boys_count) →
    (boys_avg_age * boys_count = total_age_boys) →
    (school_avg_age * total_students = total_age_school) →
    (total_age_school - total_age_boys = total_age_girls) →
    total_age_girls / girls_count = 11 :=
by
  intros h_total_students h_boys_avg_age h_school_avg_age h_girls_count 
         h_boys_count h_total_age_boys h_total_age_school h_total_age_girls
  sorry

end average_age_of_girls_l96_96114


namespace avg_of_abcd_is_4_l96_96551

-- Definitions based on conditions
def fills_squares (squares : ℕ → ℕ) : Prop :=
  ∀ n, squares n ∈ {1, 3, 5, 7}

def no_repetition_in_row (squares : ℕ → ℕ) : Prop :=
  ∀ r : ℕ, r < 4 → (finset.univ.image (λ c, squares (4 * r + c))).card = 4

def no_repetition_in_column (squares : ℕ → ℕ) : Prop :=
  ∀ c : ℕ, c < 4 → (finset.univ.image (λ r, squares (4 * r + c))).card = 4

def no_repetition_in_block (squares : ℕ → ℕ) (block : ℕ) : Prop :=
  ∀ b : ℕ, b < 4 → (finset.univ.image (λ i, squares (4 * (block / 2) + 2 * (b / 2) + i % 2 + 2 * (b % 2)))).card = 4

def avg_abcd (squares : ℕ → ℕ) (A B C D : ℕ) : ℕ :=
  (squares A + squares B + squares C + squares D) / 4

-- The Lean statement proving the average is 4
theorem avg_of_abcd_is_4 (squares : ℕ → ℕ) (A B C D : ℕ) :
  fills_squares squares →
  no_repetition_in_row squares →
  no_repetition_in_column squares →
  no_repetition_in_block squares 0 →
  no_repetition_in_block squares 1 →
  no_repetition_in_block squares 2 →
  no_repetition_in_block squares 3 →
  avg_abcd squares A B C D = 4 := by sorry

end avg_of_abcd_is_4_l96_96551


namespace quadratic_distinct_real_roots_l96_96420

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + m - 1 = 0) ∧ (x2^2 - 4*x2 + m - 1 = 0)) → m < 5 := sorry

end quadratic_distinct_real_roots_l96_96420


namespace find_angle_A_minimum_side_a_l96_96426

variable {a b c A B C : ℝ}
variable {AB AC : EuclideanSpace ℝ (Fin 2)}
variable {m n : EuclideanSpace ℝ (Fin 2)}

-- Condition 1: Vectors m and n
def vector_m (b c : ℝ) : EuclideanSpace ℝ (Fin 2) := ![(2 * b - c), a]
def vector_n (A C : ℝ) : EuclideanSpace ℝ (Fin 2) := ![cos C, cos A]

-- Condition 2: Parallel vectors
axiom parallel_vectors (b c A C : ℝ) : 
  (vector_m b c = λ i, (2 * b - c) * cos A - a * cos C = 0)

-- Condition 3: Given dot product
axiom dot_product_AB_AC (bc : ℝ) : 
  (bc * cos 60 = 4)

-- Theorem 1: Measure of angle A
theorem find_angle_A (A : ℝ) (b c a : ℝ) 
  (h1 : parallel_vectors b c A (60) )
  (h2 : cos (A) = 1 / 2) : A = 60 :=
sorry

-- Theorem 2: Minimum value of side a
theorem minimum_side_a (a b c : ℝ) 
  (h1 : bc = 8)
  (h2 : a^2 ≥ 8)
  ((b = 2 * sqrt 2) ∧ (c = 2 * sqrt 2))
  : a = 2 * sqrt 2 :=
sorry

end find_angle_A_minimum_side_a_l96_96426


namespace diagonals_of_seven_sided_polygon_l96_96264

-- Define the number of sides of the polygon
def n : ℕ := 7

-- Calculate the number of diagonals in a polygon with n sides
def numberOfDiagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- The statement to prove
theorem diagonals_of_seven_sided_polygon : numberOfDiagonals n = 14 := by
  -- Here we will write the proof steps, but they're not needed now.
  sorry

end diagonals_of_seven_sided_polygon_l96_96264


namespace sum_of_digits_l96_96892

theorem sum_of_digits (d : ℕ) (h1 : d % 5 = 0) (h2 : 3 * d - 75 = d) : 
  (d / 10 + d % 10) = 11 :=
by {
  -- Placeholder for the proof
  sorry
}

end sum_of_digits_l96_96892


namespace circles_externally_tangent_l96_96523

theorem circles_externally_tangent:
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x - 2 * y + 1 = 0 ∧ x^2 + y^2 - 8 * x - 10 * y + 25 = 0)
  → (let cx1 := 1 in
     let cy1 := 1 in
     let r1 := 1 in
     let cx2 := 4 in
     let cy2 := 5 in
     let r2 := 4 in
     ((cx2 - cx1)^2 + (cy2 - cy1)^2 = (r1 + r2)^2)) := sorry

end circles_externally_tangent_l96_96523


namespace negation_of_proposition_l96_96942

theorem negation_of_proposition:
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 > 0) :=
by
  sorry

end negation_of_proposition_l96_96942


namespace radius_of_2007_l96_96517

-- Define the conditions
def given_condition (n : ℕ) (r : ℕ → ℝ) : Prop :=
  r 1 = 1 ∧ (∀ i, 1 ≤ i ∧ i < n → r (i + 1) = 3 * r i)

-- State the theorem we want to prove
theorem radius_of_2007 (r : ℕ → ℝ) : given_condition 2007 r → r 2007 = 3^2006 :=
by
  sorry -- Proof placeholder

end radius_of_2007_l96_96517


namespace triangle_area_is_12_l96_96127

theorem triangle_area_is_12 
  (ABCDEF_equiangular: ∀ (v: char), true)
  (ABJI_square: true)
  (ABJI_area: nat = 18)
  (FEHG_square: true)
  (FEHG_area: nat = 32)
  (JBK_equilateral: true)
  (FE_eq_BC: nat = 0) : 
  ∃ (K B C : Type) (BC BK: ℕ) (area: ℕ),
  (BC = 4 * √2) → 
  (BK = 3 * √2) → 
  (area = 1 / 2 * BC * BK) → 
  (area = 12) :=
by sorry

end triangle_area_is_12_l96_96127


namespace hyperbola_sequence_start_positions_l96_96139

theorem hyperbola_sequence_start_positions :
  let C := { p : ℝ × ℝ | p.2^2 - 4 * p.1^2 = 4 }
  in let P (n : ℕ) := (fun x_n : ℝ => (x_n, 0)) n
  in let l (n : ℕ) := (fun x_n : ℝ => { p : ℝ × ℝ | p.2 = 2 * (p.1 - x_n) }) n
  in let next_p (x_n : ℝ) := (x:ℝ -> x^2-1)/(2*x_n)
  in let sequence_terminates (x_n : ℝ) : ℝ -> Prop := x_n = 0
  in (∃ (start_x0 : ℝ), (∀ (n : ℕ), ∃ x_n, P n = (x_n, 0) ∧ 
                                            ∃ p ∈ (C ∩ (l n x_n)), 
                                            (next_p p.1 = x_{n+1}) ∧
                                            sequence_terminates x_10 → 
                                            (start_x0 = x_10)) → 
                                        start_x0 ∈ 1..2^{10}-2)
:= sorry

end hyperbola_sequence_start_positions_l96_96139


namespace complex_number_value_l96_96370

theorem complex_number_value (i : ℂ) (h : i^2 = -1) : i^13 * (1 + i) = -1 + i :=
by
  sorry

end complex_number_value_l96_96370


namespace kelly_initial_games_l96_96833

theorem kelly_initial_games :
  ∃ g : ℕ, (g - 15 = 35) ↔ (g = 50) :=
begin
  sorry,
end

end kelly_initial_games_l96_96833


namespace bounded_function_inequality_l96_96468

theorem bounded_function_inequality (X : set ℝ) (hX : X = { x | 0 ≤ x })
  (f : ℝ → ℝ) (hf1 : ∀ x ∈ X, f x ≥ 0) 
  (hf2 : ∀ (x y : ℝ), x ∈ X ∧ y ∈ X → f(x) * f(y) ≤ x^2 * f(y / 2) + y^2 * f(x / 2))
  (hf3 : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x ≤ 1) :
  ∀ (x : ℝ), x ∈ X → f x ≤ x^2 :=
by
  sorry

end bounded_function_inequality_l96_96468


namespace earliest_year_exceeds_target_l96_96159

/-- Define the initial deposit and annual interest rate -/
def initial_deposit : ℝ := 100000
def annual_interest_rate : ℝ := 0.10

/-- Define the amount in the account after n years -/
def amount_after_years (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

/-- Define the target amount to exceed -/
def target_amount : ℝ := 150100

/-- Define the year the initial deposit is made -/
def initial_year : ℕ := 2021

/-- Prove that the earliest year the amount exceeds the target is 2026 -/
theorem earliest_year_exceeds_target :
  ∃ n : ℕ, n > 0 ∧ amount_after_years initial_deposit annual_interest_rate n > target_amount ∧ (initial_year + n) = 2026 :=
by
  sorry

end earliest_year_exceeds_target_l96_96159


namespace units_digit_of_expression_l96_96311

theorem units_digit_of_expression :
  (3 * 19 * 1981 - 3^4) % 10 = 6 :=
sorry

end units_digit_of_expression_l96_96311


namespace selected_people_l96_96342

variables (A B C D : Prop)

theorem selected_people :
  (A → B) ∧ (¬C → ¬B) ∧ (C → ¬D) → (B ∧ C ∧ ¬A ∧ ¬D) :=
begin
  intro h,
  sorry
end

end selected_people_l96_96342


namespace even_numbers_count_l96_96401

theorem even_numbers_count :
  (∃ S : Finset ℕ, (∀ n ∈ S, 300 < n ∧ n < 600 ∧ n % 2 = 0) ∧ S.card = 149) :=
by
  sorry

end even_numbers_count_l96_96401


namespace area_of_triangle_DEF_l96_96963

noncomputable def area_of_equilateral_triangle (R : ℝ) : ℝ :=
  let l := 2 * R * Math.sqrt(3) / 3 in -- length of each side of the triangle
  let h := l * Math.sqrt(3) / 2 in -- height of the equilateral triangle
  (1 / 2) * l * h -- area

theorem area_of_triangle_DEF :
  ∀ (R : ℝ), (π * R ^ 2 = 9 * π) → area_of_equilateral_triangle R = 9 * Math.sqrt(3) :=
by
  intros R hR
  rw [<-hR]
  sorry

end area_of_triangle_DEF_l96_96963


namespace angle_BDC_invariant_l96_96513

noncomputable theory
open_locale classical

variables {Point : Type} [MetricSpace Point]

-- Definitions of circles intersecting, lines through points and intersection of tangents
def intersect (S₁ S₂ : set Point) (A : Point) : Prop := A ∈ S₁ ∧ A ∈ S₂ 

def line_through (A B C : Point) : Prop := collinear A B C

def tangents_intersect_at (S₁ S₂ : set Point) (B C : Point) (D : Point) : Prop := 
  ∃ T₁ T₂ : set Point, tangent T₁ S₁ B ∧ tangent T₂ S₂ C ∧ D ∈ T₁ ∧ D ∈ T₂

-- The theorem statement
theorem angle_BDC_invariant 
  {S₁ S₂ : set Point} {A B C D O₁ O₂ : Point} 
  (h1 : intersect S₁ S₂ A) 
  (h2 : line_through A B C) 
  (h3 : tangents_intersect_at S₁ S₂ B C D) :
  ∃ θ : ℝ, angle BDC = θ ∧ θ = 180 - angle O₁AO₂ :=
sorry

end angle_BDC_invariant_l96_96513


namespace Kelly_initial_games_l96_96835

-- Condition definitions
variable (give_away : ℕ) (left_over : ℕ)
variable (initial_games : ℕ)

-- Given conditions
axiom h1 : give_away = 15
axiom h2 : left_over = 35

-- Proof statement
theorem Kelly_initial_games : initial_games = give_away + left_over :=
sorry

end Kelly_initial_games_l96_96835


namespace find_k_viruses_after_5h_l96_96050

-- Define the conditions for the growth law
def growth_law (k : ℝ) (t : ℝ) : ℝ := real.exp (k * t)

-- Given conditions
axiom double_every_30_minutes : growth_law k 0.5 = 2
axiom initial_value : growth_law k 0 = 1

-- Part 1: Proving the constant k
theorem find_k : k = real.log 4 :=
by
  -- Skip the detailed proof
  sorry

-- Part 2: Proving the number of viruses after 5 hours
theorem viruses_after_5h : growth_law (real.log 4) 5 = 1024 :=
by
  -- Skip the detailed proof
  sorry

end find_k_viruses_after_5h_l96_96050


namespace sum_grey_angles_l96_96446

-- Defining the conditions given in the problem statement.
def five_lines_intersecting (p: Point) (l1 l2 l3 l4 l5: Line): Prop :=
  all_line_intersect_at p [l1, l2, l3, l4, l5]

def given_angle (θ: ℝ): Prop :=
  θ = 34

-- Define the problem to prove sum of grey angles
theorem sum_grey_angles (p: Point) (l1 l2 l3 l4 l5: Line):
  five_lines_intersecting p l1 l2 l3 l4 l5 →
  given_angle 34 →
  ∑ θ in grey_angles, θ = 146 :=
by
  sorry

end sum_grey_angles_l96_96446


namespace triangle_inequality_l96_96898

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (A / 2)) + Real.sqrt 3 * Real.tan (A / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (B / 2)) + Real.sqrt 3 * Real.tan (B / 2)) +
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (B / 2)) + Real.sqrt 3 * Real.tan (B / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (C / 2)) + Real.sqrt 3 * Real.tan (C / 2)) +
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (C / 2)) + Real.sqrt 3 * Real.tan (C / 2)) * 
  (1 - Real.sqrt (Real.sqrt 3 * Real.tan (A / 2)) + Real.sqrt 3 * Real.tan (A / 2)) ≥ 3 :=
by
  sorry

end triangle_inequality_l96_96898


namespace min_dist_AB_l96_96439

-- Definitions of the conditions
structure Point3D where
  x : Float
  y : Float
  z : Float

def O := Point3D.mk 0 0 0
def B := Point3D.mk (Float.sqrt 3) (Float.sqrt 2) 2

def dist (P Q : Point3D) : Float :=
  Float.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Given points
variables (A : Point3D)
axiom AO_eq_1 : dist A O = 1

-- Minimum value of |AB|
theorem min_dist_AB : dist A B ≥ 2 := 
sorry

end min_dist_AB_l96_96439


namespace find_biology_marks_l96_96454

variable eng math phys chem bio : ℕ 
variable avg : ℕ

def total_marks_four := eng + math + phys + chem
def total_marks_all := avg * 5

theorem find_biology_marks (h_eng : eng = 96) (h_math : math = 65)
  (h_phys : phys = 82) (h_chem : chem = 67) (h_avg : avg = 79) :
  bio = total_marks_all - total_marks_four :=
by
  rw [h_eng, h_math, h_phys, h_chem, h_avg]
  -- total_marks_four = 96 + 65 + 82 + 67 = 310
  -- total_marks_all = 79 * 5 = 395
  -- bio = 395 - 310 = 85
  simp; sorry

end find_biology_marks_l96_96454


namespace infinitely_many_H_points_l96_96774

-- Define the curve C
def curve_C (P : ℝ × ℝ) : Prop := P.1 / 4 + P.2^2 = 1

-- Define what an H point is
def H_point (P : ℝ × ℝ) : Prop :=
 ∃ (A : ℝ × ℝ) (B : ℝ × ℝ),
  A ≠ P ∧
  curve_C A ∧
  B.1 = 4 ∧
  (dist P A = dist P B ∨ dist P A = dist A B)

-- Prove the existence of infinitely many H points
theorem infinitely_many_H_points :
  ∃ (S : set (ℝ × ℝ)), 
  (∀ P, P ∈ S → H_point P) ∧ 
  (S ⊆ {P : ℝ × ℝ | curve_C P}) ∧ 
  (S.infinite) ∧ 
  (S ≠ {P : ℝ × ℝ | curve_C P}) :=
sorry

end infinitely_many_H_points_l96_96774


namespace sqrt_calc1_sqrt_calc2_l96_96653

-- Problem 1 proof statement
theorem sqrt_calc1 : ( (Real.sqrt 2 + Real.sqrt 3) ^ 2 - (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) = 4 + 2 * Real.sqrt 6 ) :=
  sorry

-- Problem 2 proof statement
theorem sqrt_calc2 : ( (2 - Real.sqrt 3) ^ 2023 * (2 + Real.sqrt 3) ^ 2023 - 2 * abs (-Real.sqrt 3 / 2) - (-Real.sqrt 2) ^ 0 = -Real.sqrt 3 ) :=
  sorry

end sqrt_calc1_sqrt_calc2_l96_96653


namespace f_g_minus_g_f_l96_96469

def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := x / 2 - 1

theorem f_g_minus_g_f {x : ℝ} : f(g(x)) - g(f(x)) = -1 / 2 := by
  sorry

end f_g_minus_g_f_l96_96469


namespace vector_constructions_l96_96152

variable (c : Vector ℝ)

theorem vector_constructions :
  (∃ u : Vector ℝ, ∥u∥ = 4 ∧ ∀ s, s * u = s * c) ∧
  (-4 • c = (-4 : ℝ) • c) ∧
  (1.5 • c = (1.5 : ℝ) • c) :=
by {
  sorry
}

end vector_constructions_l96_96152


namespace equilateral_triangle_fill_l96_96285

theorem equilateral_triangle_fill (a b : ℝ) (h1 : a = 12) (h2 : b = 2) : 
  let A_large := (sqrt 3 / 4) * a^2,
      A_small := (sqrt 3 / 4) * b^2 in 
  A_large / A_small = 36 :=
by
  sorry

end equilateral_triangle_fill_l96_96285


namespace stream_bottom_width_l96_96932

-- Definitions of the conditions
def top_width : ℝ := 10
def area : ℝ := 640
def depth : ℝ := 80

-- Width at the bottom
def bottom_width (b : ℝ) :=
  2 * area / depth - top_width

-- Theorem stating the width at the bottom is 6 meters
theorem stream_bottom_width : bottom_width 6 = 6 :=
by
  unfold bottom_width
  -- Prove the calculation step by step
  have h : 2 * area / depth - top_width = 6
  · sorry
  exact h

end stream_bottom_width_l96_96932


namespace last_two_nonzero_digits_of_70_fact_l96_96198

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  let factorial := (Nat.factorial n) in
  let trim_zeros := factorial / (10 ^ (Nat.log10 (factorial / 10 ) + 1)) * 10 in
  trim_zeros % 100

theorem last_two_nonzero_digits_of_70_fact : last_two_nonzero_digits 70 = 8 :=
  sorry

end last_two_nonzero_digits_of_70_fact_l96_96198


namespace determine_g_l96_96143

theorem determine_g (f g h: Polynomial ℝ) 
  (Hf : f ≠ 0) 
  (Hg : g ≠ 0) 
  (Hh : h ≠ 0) 
  (Hqf : f.degree = 2) 
  (Hqg : g.degree = 2) 
  (Hqh : h.degree = 2)
  (H : ∀ x, f.eval (g.eval x) = f.eval x + g.eval x * h.eval x)
  (H2 : g.eval 2 = 12) :
  g = Polynomial.Coeff 1 0 * X^2 + Polynomial.Coeff 0 3 * X + Polynomial.Coeff 0 2 :=
sorry

end determine_g_l96_96143


namespace factor_expression_l96_96011

theorem factor_expression (x : ℝ) : 5 * x * (x - 2) + 9 * (x - 2) = (x - 2) * (5 * x + 9) := 
by 
sorry

end factor_expression_l96_96011


namespace list_price_of_article_l96_96941

theorem list_price_of_article
  (P : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (final_price : ℝ)
  (h1 : discount1 = 0.10)
  (h2 : discount2 = 0.01999999999999997)
  (h3 : final_price = 61.74) :
  P = 70 :=
by
  sorry

end list_price_of_article_l96_96941


namespace misha_class_predictions_probability_l96_96585

-- Definitions representing the conditions
def monday_classes : ℕ := 5
def tuesday_classes : ℕ := 6
def total_classes : ℕ := monday_classes + tuesday_classes
def total_flips : ℕ := total_classes
def correct_predictions : ℕ := 7
def correct_monday_predictions : ℕ := 3

-- Lean theorem representing the proof problem
theorem misha_class_predictions_probability :
  (probability_of_correct_predictions total_flips correct_predictions monday_classes correct_monday_predictions) =
    (5 / 11) :=
  sorry

end misha_class_predictions_probability_l96_96585


namespace Randy_trees_on_farm_l96_96487

theorem Randy_trees_on_farm :
  let initial_mango_trees := 60
  let additional_avocado_trees := 10
  let additional_lemon_trees := 8
  let loss_percentage := 0.20
  let mango_trees_after_disease := initial_mango_trees - initial_mango_trees * loss_percentage
  let coconut_trees := initial_mango_trees / 2 - 5
  let total_trees := mango_trees_after_disease + coconut_trees + additional_avocado_trees + additional_lemon_trees
  total_trees = 91 :=
by
  let initial_mango_trees := 60
  let additional_avocado_trees := 10
  let additional_lemon_trees := 8
  let loss_percentage := 0.20
  let mango_trees_after_disease := initial_mango_trees - initial_mango_trees * loss_percentage
  let coconut_trees := initial_mango_trees / 2 - 5
  let total_trees := mango_trees_after_disease + coconut_trees + additional_avocado_trees + additional_lemon_trees
  show total_trees = 91
  sorry

end Randy_trees_on_farm_l96_96487


namespace range_of_f_l96_96946

noncomputable def f (x : ℝ) : ℝ := (|Real.sin x| * Real.cos x) + (|Real.cos x| * Real.sin x)

theorem range_of_f : Set.range f = [-1, 1] := sorry

end range_of_f_l96_96946


namespace range_of_m_l96_96423

theorem range_of_m (m : ℝ) (x : ℝ) : (∀ x, (1 - m) * x = 2 - 3 * x → x > 0) ↔ m < 4 :=
by
  sorry

end range_of_m_l96_96423


namespace find_f2a_eq_zero_l96_96369

variable {α : Type} [LinearOrderedField α]

-- Definitions for the function f and its inverse
variable (f : α → α)
variable (finv : α → α)

-- Given conditions
variable (a : α)
variable (h_nonzero : a ≠ 0)
variable (h_inverse1 : ∀ x : α, finv (x + a) = f (x + a)⁻¹)
variable (h_inverse2 : ∀ x : α, f (x) = finv⁻¹ x)
variable (h_fa : f a = a)

-- Statement to be proved in Lean
theorem find_f2a_eq_zero : f (2 * a) = 0 :=
sorry

end find_f2a_eq_zero_l96_96369


namespace probability_of_four_odd_slips_l96_96677

-- Define the conditions
def number_of_slips : ℕ := 10
def odd_slips : ℕ := 5
def even_slips : ℕ := 5
def slips_drawn : ℕ := 4

-- Define the required probability calculation
def probability_four_odd_slips : ℚ := (5 / 10) * (4 / 9) * (3 / 8) * (2 / 7)

-- State the theorem we want to prove
theorem probability_of_four_odd_slips :
  probability_four_odd_slips = 1 / 42 :=
by
  sorry

end probability_of_four_odd_slips_l96_96677


namespace directrix_of_parabola_l96_96935

theorem directrix_of_parabola :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ y₀ : ℝ, y₀ = -1 ∧ ∀ y' : ℝ, y' = y₀) :=
by
  sorry

end directrix_of_parabola_l96_96935


namespace geometric_sequence_sum_l96_96881
   noncomputable def geo_seq_sum : ℕ → ℕ → (ℕ → ℝ) → ℝ := sorry
   
   theorem geometric_sequence_sum 
     (a : ℕ → ℝ) 
     (common_ratio gt_one : ∀ n, a n = a 1 * (common_ratio ^ (n - 1)) ∧ a n > 1) 
     : (Real.log (a 1) * Real.log (a 2012) * (∑ i in Finset.range 2011, 1 / (Real.log (a i) * Real.log (a (i+1))))) = 2011 := 
   sorry
   
end geometric_sequence_sum_l96_96881


namespace triangle_angles_30_60_90_l96_96131

-- Definition of the angles based on the given ratio
def angles_ratio (A B C : ℝ) : Prop :=
  A / B = 1 / 2 ∧ B / C = 2 / 3

-- The main statement to be proved
theorem triangle_angles_30_60_90
  (A B C : ℝ)
  (h1 : angles_ratio A B C)
  (h2 : A + B + C = 180) :
  A = 30 ∧ B = 60 ∧ C = 90 := 
sorry

end triangle_angles_30_60_90_l96_96131


namespace carla_glasses_lemonade_l96_96893

theorem carla_glasses_lemonade (time_total : ℕ) (rate : ℕ) (glasses : ℕ) 
  (h1 : time_total = 3 * 60 + 40) 
  (h2 : rate = 20) 
  (h3 : glasses = time_total / rate) : 
  glasses = 11 := 
by 
  -- We'll fill in the proof here in a real scenario
  sorry

end carla_glasses_lemonade_l96_96893


namespace value_two_sd_below_mean_l96_96928

theorem value_two_sd_below_mean :
  let mean := 14.5
  let stdev := 1.7
  mean - 2 * stdev = 11.1 :=
by
  sorry

end value_two_sd_below_mean_l96_96928


namespace max_tetrahedron_volume_on_unit_sphere_l96_96723

open EuclideanGeometry

theorem max_tetrahedron_volume_on_unit_sphere
  (A B C D : EuclideanGeometry.Ch.Point 3)
  (unit_sphere : Metric.sphere (0 : EuclideanGeometry.Ch.Point 3) 1)
  (hA : A ∈ unit_sphere)
  (hB : B ∈ unit_sphere)
  (hC : C ∈ unit_sphere)
  (hD : D ∈ unit_sphere)
  (hAB : EuclideanGeometry.dist A B = EuclideanGeometry.dist A C)
  (hAC : EuclideanGeometry.dist A C = EuclideanGeometry.dist A D)
  (hBC : EuclideanGeometry.dist B C = EuclideanGeometry.dist C D)
  (hBD : EuclideanGeometry.dist B D = EuclideanGeometry.dist C D) :
  tetrahedron_volume A B C D ≤ (8 * Real.sqrt 3) / 27 := sorry

end max_tetrahedron_volume_on_unit_sphere_l96_96723


namespace range_of_a_l96_96100

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a ∈ [-1, 3]) := 
by
  sorry

end range_of_a_l96_96100


namespace exists_irrational_in_interval_times_polynomial_is_integer_l96_96676

theorem exists_irrational_in_interval_times_polynomial_is_integer :
  ∃ (x : ℝ), (0.3 ≤ x ∧ x ≤ 0.4) ∧ irrational x ∧ (∃ k : ℤ, x * (x + 1) * (x + 2) = k) :=
sorry

end exists_irrational_in_interval_times_polynomial_is_integer_l96_96676


namespace distance_center_to_plane_of_equilateral_triangle_on_sphere_l96_96055

noncomputable def sphere_radius (O : Point) : ℝ :=
  1

noncomputable def points_on_sphere (O A B C: Point) (r: ℝ) : Prop :=
  dist O A = r ∧ dist O B = r ∧ dist O C = r

noncomputable def equal_spherical_distances (A B C: Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def distance_from_center_to_plane (O A B C : Point) (d : ℝ) : Prop :=
  -- Function for distance from point to a plane
  let plane := plane_ABC A B C in
  dist_point_to_plane O plane = d

-- Definitions of Point and Plane would be required in a full implementation
-- Here they are treated as placeholders and the noncomputable is added assuming 
-- necessary definitions and imports are available in Mathlib.
constants (Point : Type) (plane_ABC : Point → Point → Point → Plane)
           (dist : Point → Point → ℝ) (dist_point_to_plane : Point → Plane → ℝ)

theorem distance_center_to_plane_of_equilateral_triangle_on_sphere
  (O A B C : Point)
  (h1 : sphere_radius O = 1)
  (h2 : points_on_sphere O A B C 1)
  (h3 : equal_spherical_distances A B C) :
  distance_from_center_to_plane O A B C B :=
sorry

end distance_center_to_plane_of_equilateral_triangle_on_sphere_l96_96055


namespace plantation_length_l96_96889

theorem plantation_length 
  (L : ℕ) 
  (plantation_area : L * 500 = 250000)
  (sqft_to_grams : 1 * 50 = 50)
  (peanuts_to_butter : 20 * 5 = 100)
  (butter_to_money : ∀ kg : ℕ, 10 * kg = 10 * kg)
  (total_money : 31250) :
  L = 500 :=
by
  sorry

end plantation_length_l96_96889


namespace weight_loss_l96_96824

variable (J S x : ℝ)

def weight_condition_1 := J = 113
def weight_condition_2 := J + S = 153
def weight_condition_3 := J - x = 2 * S

theorem weight_loss : weight_condition_1 → weight_condition_2 → weight_condition_3 → x = 33 :=
by
  intro h1 h2 h3
  sorry

end weight_loss_l96_96824


namespace sin_shift_graph_l96_96542

theorem sin_shift_graph (x : ℝ) : 
  (∀ x, sin (3 * x) = sin (3 * (x + π / 12))) → 
  y = sin (π / 4 - 3 * x) := 
by
  sorry

end sin_shift_graph_l96_96542


namespace jason_combinations_l96_96137

-- Definitions of conditions
def is_even (n : ℕ) := n % 2 = 0
def is_odd (n : ℕ) := n % 2 = 1

-- Set of digits Jason uses
def digits : set ℕ := {1, 2, 3, 4, 5, 6}

-- Condition that even numbers are followed by odd numbers and vice versa
def valid_combination (xs : list ℕ) : Prop :=
  (∀ i < xs.length - 1, (is_even (xs.nth_le i (by linarith)) → is_odd (xs.nth_le (i + 1) (by linarith))) ∧
                         (is_odd (xs.nth_le i (by linarith)) → is_even (xs.nth_le (i + 1) (by linarith))))

-- Theorem statement
theorem jason_combinations : 
  (finset.univ.filter (λ xs : vector ℕ 6, (∀ i, (xs.nth i ∈ digits)) ∧ valid_combination xs.to_list)).card = 1458 :=
by sorry

end jason_combinations_l96_96137


namespace train_length_is_900_meters_l96_96276

noncomputable def length_of_train (time_to_cross : ℝ) (speed_of_man : ℝ) (speed_of_train : ℝ) : ℝ :=
  let relative_speed := (speed_of_train - speed_of_man) * (1000 / 3600) -- convert km/hr to m/s
  in relative_speed * time_to_cross

theorem train_length_is_900_meters :
  length_of_train 53.99568034557235 3 63 = 900 := by
  sorry

end train_length_is_900_meters_l96_96276


namespace num_isosceles_equilateral_triangles_l96_96473

theorem num_isosceles_equilateral_triangles :
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧
  let a := n / 100 in
  let b := (n / 10) % 10 in
  let c := n % 10 in
  (a = b ∨ a = c ∨ b = c) ∧ (a ≠ 0) ∧ (a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b) } = 165 := sorry

end num_isosceles_equilateral_triangles_l96_96473


namespace tangent_line_correct_inequality_holds_l96_96382

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x^2

-- Statement verifying the tangent line condition
theorem tangent_line_correct (a := 1) : 
  let x := 1 in
  let tangent := λ x y : ℝ, y = -x in
  tangent 1 (f a 1) :=
begin
  unfold f,
  simp,
  sorry
end

-- Statement verifying the inequality condition
theorem inequality_holds (a := 0.5) (x : ℝ) (h : x > 0) : 
  f a x < 0 :=
begin
  unfold f,
  sorry
end

end tangent_line_correct_inequality_holds_l96_96382


namespace solve_for_x_l96_96494

theorem solve_for_x (x : ℂ) (h : 5 - 2 * I * x = 4 - 5 * I * x) : x = I / 3 :=
by
  sorry

end solve_for_x_l96_96494


namespace expected_score_particular_player_l96_96301

-- Define types of dice
inductive DiceType : Type
| A | B | C

-- Define the faces of each dice type
def DiceFaces : DiceType → List ℕ
| DiceType.A => [2, 2, 4, 4, 9, 9]
| DiceType.B => [1, 1, 6, 6, 8, 8]
| DiceType.C => [3, 3, 5, 5, 7, 7]

-- Define a function to calculate the score of a player given their roll and opponents' rolls
def player_score (p_roll : ℕ) (opp_rolls : List ℕ) : ℕ :=
  opp_rolls.foldl (λ acc roll => if roll < p_roll then acc + 1 else acc) 0

-- Define a function to calculate the expected score of a player
noncomputable def expected_score (dice_choice : DiceType) : ℚ :=
  let rolls := DiceFaces dice_choice
  let total_possibilities := (rolls.length : ℚ) ^ 3
  let score_sum := rolls.foldl (λ acc p_roll =>
    acc + rolls.foldl (λ acc1 opp1_roll =>
        acc1 + rolls.foldl (λ acc2 opp2_roll =>
            acc2 + player_score p_roll [opp1_roll, opp2_roll]
          ) 0
      ) 0
    ) 0
  score_sum / total_possibilities

-- The main theorem statement
theorem expected_score_particular_player : (expected_score DiceType.A + expected_score DiceType.B + expected_score DiceType.C) / 3 = 
(8 : ℚ) / 9 := sorry

end expected_score_particular_player_l96_96301


namespace inverse_function_eq_l96_96194

noncomputable def f_inv (x : ℝ) : ℝ := log x + 1

noncomputable def f (x : ℝ) : ℝ := 2^(x - 1) - 1

theorem inverse_function_eq :
  (∀ x, x > 1 → f (f_inv x) = x) ∧ (∀ y, y > 1 → f_inv (f y) = y) :=
by {
  sorry
}

end inverse_function_eq_l96_96194


namespace problem_statement_l96_96145

noncomputable def α : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def β : ℝ := 3 - 2 * Real.sqrt 2
noncomputable def x : ℝ := α ^ 50
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end problem_statement_l96_96145


namespace range_of_a_l96_96746

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 3) * x^3 + (1 - a) / 2 * x^2 - a * x - a

theorem range_of_a (a : ℝ) (h₀ : 0 < a)
  (h₁ : ∃ x y : ℝ, x ≠ y ∧ x ∈ Ioo (-2 : ℝ) 0 ∧ y ∈ Ioo (-2 : ℝ) 0 ∧ f a x = 0 ∧ f a y = 0) :
  0 < a ∧ a < 1 / 3 :=
by
  sorry

end range_of_a_l96_96746


namespace range_of_m_l96_96748

noncomputable def f (x m : ℝ) : ℝ := real.exp x * (real.log x + (x - m)^2)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → deriv (λ x, f x m) x - f x m > 0) →
  m < 2 * real.sqrt 2 :=
by 
  sorry

end range_of_m_l96_96748


namespace complement_A_U_l96_96759

open Set

variable (U : Set ℕ) (A : Set ℕ)

def U := {1, 2, 3, 4, 5, 6}
def A := {2, 4, 6}

theorem complement_A_U :
  U \ A = {1, 3, 5} := by
  sorry

end complement_A_U_l96_96759


namespace max_gcd_is_121_l96_96201

-- Definitions from the given problem
def a (n : ℕ) : ℕ := 120 + n^2
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- The statement we want to prove
theorem max_gcd_is_121 : ∃ n : ℕ, d n = 121 := sorry

end max_gcd_is_121_l96_96201


namespace count_madam_paths_l96_96770

def grid : list (list char) :=
[['M', 'A', 'D', 'A'],
 ['A', 'M', 'A', 'M'],
 ['D', 'A', 'M', 'A'],
 ['A', 'M', 'A', 'D']]

def is_next (p1 p2 : ℕ × ℕ) : Prop :=
(p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 = p2.2 - 1)) ∨
(p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 = p2.1 - 1))

def is_valid_path_madam (path : list (ℕ × ℕ)) : Prop :=
path.length = 5 ∧
list.pairwise is_next path ∧
path.head = some (0, 0) ∧
(path.nth 1 = some ('M', 0) ∧
 path.nth 2 = some ('A', 1) ∧
 path.nth 3 = some ('D', 2) ∧
 path.nth 4 = some ('A', 3) ∧
 path.nth 5 = some ('M', 4))

theorem count_madam_paths : 
  (list.countp is_valid_path_madam (list.nested_product (list.descend (list.words "MADAM")))) = 80 := 
sorry

end count_madam_paths_l96_96770


namespace minimize_PR_RQ_at_R_l96_96058

/-!
# Problem Statement

Given points \( P(-1,-2) \) and \( Q(4,2) \) in the \( xy \)-plane, point \( R(1,m) \) is taken so that \( PR + RQ \) is a minimum. Then \( m \) equals \( -\frac{2}{5} \).
-/

def point := ℝ × ℝ

def P : point := (-1, -2)
def Q : point := (4, 2)
def R (m : ℝ) : point := (1, m)

def distance (A B : point) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)
  
-- Define minimization condition equation
def minimize_condition (m : ℝ) : Prop :=
  let R_m := (1, m)
  ∃ m, (distance P R_m + distance R_m Q) = distance P Q

theorem minimize_PR_RQ_at_R :
  minimize_condition (-2 / 5) :=
sorry

end minimize_PR_RQ_at_R_l96_96058


namespace tickets_sold_at_reduced_price_first_week_l96_96313

variable (T : ℕ) (F : ℕ) (R : ℕ)
variable hT : T = 25200
variable hF : F = 16500
variable hR : T - F = R
variable hFullPriceMultiple : F = 5 * R

theorem tickets_sold_at_reduced_price_first_week : 
  R = 8700 :=
by
  rw [hT, hF, hR, hFullPriceMultiple] at *
  sorry

end tickets_sold_at_reduced_price_first_week_l96_96313


namespace arithmetic_sequence_ratio_l96_96032

-- Given sequences a_n and b_n are arithmetic sequences
variable (a : ℕ → ℚ)
variable (b : ℕ → ℚ)

-- The sums of the first n terms of a and b are denoted as S_n and T_n
def S_n (n : ℕ) := (∑ i in finset.range (n + 1), a i)  -- Sum of first n terms of a
def T_n (n : ℕ) := (∑ i in finset.range (n + 1), b i)  -- Sum of first n terms of b

-- Given condition
axiom sum_ratio_condition (n : ℕ) : S_n a n / T_n b n = (2 * n : ℚ) / (3 * n + 1)

-- Prove the desired ratio
theorem arithmetic_sequence_ratio :
  (a 4 + a 6) / (b 3 + b 7) = 9 / 14 :=
sorry

end arithmetic_sequence_ratio_l96_96032


namespace geometric_mean_solution_l96_96724

theorem geometric_mean_solution :
  ∃ x : ℕ, let a := 4, b := 9 in x * x = a * b ∧ x = 6 :=
begin
  use 6,
  simp,
  norm_num,
end

end geometric_mean_solution_l96_96724


namespace find_f_log_log2_l96_96711

def f (x : ℝ) (a : ℝ) : ℝ := a * Real.sin x + x^(1/(2019:ℝ)) + 1

theorem find_f_log_log2 (a : ℝ) (h : f (Real.log (Real.log 10 / Real.log 2)) a = 3) : 
  f (Real.log (Real.log 2)) a = -1 :=
sorry

end find_f_log_log2_l96_96711


namespace find_a_and_b_nth_equation_conjecture_l96_96891

theorem find_a_and_b {a b : ℤ} (h1 : 1^2 + 2^2 - 3^2 = 1 * a - b)
                                        (h2 : 2^2 + 3^2 - 4^2 = 2 * 0 - b)
                                        (h3 : 3^2 + 4^2 - 5^2 = 3 * 1 - b)
                                        (h4 : 4^2 + 5^2 - 6^2 = 4 * 2 - b):
    a = -1 ∧ b = 3 :=
    sorry

theorem nth_equation_conjecture (n : ℤ) :
  n^2 + (n+1)^2 - (n+2)^2 = n * (n-2) - 3 :=
  sorry

end find_a_and_b_nth_equation_conjecture_l96_96891


namespace system_of_equations_solution_l96_96939

theorem system_of_equations_solution : 
  ∃ x y : ℝ, (2 * x - y = -1) ∧ (x + y = 4) ∧ (x = 1) ∧ (y = 3) :=
by
  use 1, 3
  split
  case left =>
    show 2 * 1 - 3 = -1
    calc
      2 * 1 - 3 = 2 - 3 : by ring
      ... = -1 : by norm_num
  case right =>
    split
    case left =>
      show 1 + 3 = 4
      calc
        1 + 3 = 4 : by norm_num
    case right =>
      split
      case left =>
        show 1 = 1
        rfl
      case right =>
        show 3 = 3
        rfl

end system_of_equations_solution_l96_96939


namespace Linda_journey_length_l96_96886

theorem Linda_journey_length : 
  (∃ x : ℝ, x = 30 + x * 1/4 + x * 1/7) → x = 840 / 17 :=
by
  sorry

end Linda_journey_length_l96_96886


namespace integer_units_digit_is_6_l96_96375

def units_digit (n : ℤ) : ℤ :=
n % 10

def type_of_integer_p (p : ℤ) : Prop :=
have h1 : units_digit p > 0, from sorry, -- Positive units digit
have h2 : units_digit (p^3) - units_digit (p^2) = 0, from sorry, -- Units digit condition on p^2 and p^3
have h3 : units_digit (p + 4) = 0, from sorry, -- Units digit of p+4 being 10
units_digit p = 6

theorem integer_units_digit_is_6 (p : ℤ) : type_of_integer_p p :=
sorry

end integer_units_digit_is_6_l96_96375


namespace question_l96_96971

-- Define the necessary geometrical terms and properties
def is_circumcenter (P : Point) (Δ : Triangle) : Prop :=
  ∀ (A B C : Point), A ∈ Δ ∧ B ∈ Δ ∧ C ∈ Δ → dist P A = dist P B ∧ dist P B = dist P C

def congruent_angles (A B : Quadrilateral) : Prop :=
  ∃ (θ₁ θ₂ : Angle), A.has_angle θ₁ ∧ A.has_angle θ₂ ∧ B.has_angle θ₁ ∧ B.has_angle θ₂

def equal_diagonals_and_bisect (Ω : Quadrilateral) : Prop :=
  (dist Ω.diag₁ Ω.diag₂) = (dist Ω.diag₃ Ω.diag₄) ∧ Ω.diag₁_mid = Ω.diag₃_mid ∧ Ω.diag₂_mid = Ω.diag₄_mid

def formed_by_midpoints (A : Quadrilateral) (B : Quadrilateral) : Prop :=
  ∀ P ∈ A.vertices, ∃ Q ∈ B.vertices, Q.is_midpoint P

-- Definitions for the Lean 4 equivalent statements
theorem question (P Δ : Point) (A B C Ω Π : Quadrilateral) :
  (is_circumcenter P Δ) →
  (congruent_angles A B) →
  (equal_diagonals_and_bisect Ω) →
  (formed_by_midpoints Ω Π) →
  false_statement D :=
begin
  sorry
end

end question_l96_96971


namespace ball_attendance_l96_96864

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end ball_attendance_l96_96864


namespace alternating_sum_series_l96_96297

theorem alternating_sum_series : 
  (-1 + 3 - 5 + 7 - 9 + 11 - ⋯ - 1989 + 1991 - 1993) = -997 :=
sorry

end alternating_sum_series_l96_96297


namespace more_green_peaches_than_red_l96_96531

theorem more_green_peaches_than_red : 
  let red_peaches := 7
  let green_peaches := 8
  green_peaches - red_peaches = 1 := 
by
  let red_peaches := 7
  let green_peaches := 8
  show green_peaches - red_peaches = 1 
  sorry

end more_green_peaches_than_red_l96_96531


namespace find_m_for_eccentric_ellipse_l96_96516

theorem find_m_for_eccentric_ellipse (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/5 + (y^2)/m = 1) ∧
  (∀ e : ℝ, e = (Real.sqrt 10)/5) → 
  (m = 25/3 ∨ m = 3) := sorry

end find_m_for_eccentric_ellipse_l96_96516


namespace six_digit_count_div_by_217_six_digit_count_div_by_218_l96_96405

-- Definitions for the problem
def six_digit_format (n : ℕ) : Prop :=
  ∃ a b : ℕ, (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10) ∧ n = 100001 * a + 10010 * b + 100 * a + 10 * b + a

def divisible_by (n : ℕ) (divisor : ℕ) : Prop :=
  n % divisor = 0

-- Problem Part a: How many six-digit numbers of the form are divisible by 217
theorem six_digit_count_div_by_217 :
  ∃ count : ℕ, count = 3 ∧ ∀ n : ℕ, six_digit_format n → divisible_by n 217  → (n = 313131 ∨ n = 626262 ∨ n = 939393) :=
sorry

-- Problem Part b: How many six-digit numbers of the form are divisible by 218
theorem six_digit_count_div_by_218 :
  ∀ n : ℕ, six_digit_format n → divisible_by n 218 → false :=
sorry

end six_digit_count_div_by_217_six_digit_count_div_by_218_l96_96405


namespace ball_attendance_l96_96859

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end ball_attendance_l96_96859


namespace cyclic_quadrilateral_ratio_l96_96436

-- Definitions for vertices and triangle
variable {A B C A1 B1 C1 : Type}
variable [triangle_ABC : Triangle A B C]
variable [angle_bisectors_meet : AngleBisectorsMeet A B C A1 B1 C1]
variable [cyclic : CyclicQuadrilateral B A1 B1 C1]

-- The theorem statement
theorem cyclic_quadrilateral_ratio :
  (AC : ℝ) / ((AB : ℝ) + (BC : ℝ)) = 
  (AB : ℝ) / ((AC : ℝ) + (CB : ℝ)) + 
  (BC : ℝ) / ((BA : ℝ) + (AC : ℝ)) :=
by
  /- Proof omitted -/
  sorry

-- Assumptions for Triangle, AngleBisectorsMeet and CyclicQuadrilateral
structure Triangle (A B C : Type) := 
  (exists_points : ∃ (a b c : Type), True)

structure AngleBisectorsMeet (A B C A1 B1 C1 : Type) :=
  (exists_bisectors : ∃ (a1 b1 c1 : Type), True)

structure CyclicQuadrilateral (A B C D : Type) :=
  (exists_cycle : ∃ (a b c d : Type), True)

-- Definitions of distances (to match algebraic operations)
noncomputable def AB : ℝ := sorry
noncomputable def BC : ℝ := sorry
noncomputable def AC : ℝ := sorry
noncomputable def BA : ℝ := sorry
noncomputable def CB : ℝ := sorry

end cyclic_quadrilateral_ratio_l96_96436


namespace quadratic_eq_has_two_distinct_real_roots_l96_96949

theorem quadratic_eq_has_two_distinct_real_roots (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - 2*m*x - m - 1 = 0 ↔ x = x1 ∨ x = x2) :=
by
  sorry

end quadratic_eq_has_two_distinct_real_roots_l96_96949


namespace final_number_is_odd_l96_96240

theorem final_number_is_odd :
  (∃ n : ℕ, n > 0 ∧
   ∀ (numbers : Finset ℕ),
     (∀ i, i ∈ numbers → (1 ≤ i ∧ i ≤ 2014)) →
     (∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → a ≠ b →
       numbers \ {a, b} ∪ {|a - b|} ≠ ∅ ∧
       numbers ≠ ∅) →
     (∃ k, numbers = {k} ∧ k % 2 = 1)) :=
by
  sorry

end final_number_is_odd_l96_96240


namespace range_of_f_lt_zero_l96_96418

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (S : set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ S → y ∈ S → x < y → f x > f y

theorem range_of_f_lt_zero {f : ℝ → ℝ} 
  (h_even : is_even_function f)
  (h_decreasing : is_decreasing_on f (set.Iic 0))
  (h_f_2_eq_0 : f 2 = 0) : 
  {x : ℝ | f x < 0} = set.Ioo (-2) 2 := 
sorry

end range_of_f_lt_zero_l96_96418


namespace abc_inequality_l96_96147

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
  sorry

end abc_inequality_l96_96147


namespace expression1_value_expression2_value_l96_96335

open Real

-- Definitions directly from the conditions
noncomputable def expr1 : ℝ := (2 + 1/4)^(1/2) - (π - 1)^0 - (3 + 3/8)^(-2/3)
noncomputable def log_base (a b : ℝ) := log b / log a
noncomputable def lg (x : ℝ) := log x / log 10

-- Proof statement for the first expression
theorem expression1_value : expr1 = 1 / 18 := sorry

-- Line by line definitions for the second expression
noncomputable def term1 : ℝ := log_base 3 (sqrt (1 / 3))
noncomputable def term2 : ℝ := log_base 10 100000
noncomputable def term3 : ℝ := lg √5
noncomputable def term4 : ℝ := 2

noncomputable def expr2 : ℝ := term1 + term2 + term3 + term4

-- Proof statement for the second expression
theorem expression2_value : expr2 = 3 / 2 := sorry

end expression1_value_expression2_value_l96_96335


namespace minimum_production_avoiding_loss_maximum_profit_l96_96998

def cost (x : ℝ) : ℝ := 2 + x
def revenue (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then -0.4 * x^2 + 4.2 * x - 0.8
  else 14.7 - 9 / (x - 3)

def profit (x : ℝ) : ℝ := revenue x - cost x

theorem minimum_production_avoiding_loss :
  ∀ x, x ≥ 1 → 0 < x → x ≤ 5 → profit x ≥ 0 := by
  sorry

theorem maximum_profit :
  ∃ x, x = 6 ∧ profit x = 3.7 := by
  sorry

end minimum_production_avoiding_loss_maximum_profit_l96_96998


namespace minimum_value_nine_l96_96463

noncomputable def min_value (a b c k : ℝ) : ℝ :=
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a

theorem minimum_value_nine (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  min_value a b c k ≥ 9 :=
sorry

end minimum_value_nine_l96_96463


namespace sum_possible_values_frac_l96_96303

theorem sum_possible_values_frac :
  (∑ x in ({x : ℝ | 4 - 9/x + 4/x^2 = 0}).to_finset, 3/x) = 51/4 := 
sorry

end sum_possible_values_frac_l96_96303


namespace misha_class_predictions_probability_l96_96584

-- Definitions representing the conditions
def monday_classes : ℕ := 5
def tuesday_classes : ℕ := 6
def total_classes : ℕ := monday_classes + tuesday_classes
def total_flips : ℕ := total_classes
def correct_predictions : ℕ := 7
def correct_monday_predictions : ℕ := 3

-- Lean theorem representing the proof problem
theorem misha_class_predictions_probability :
  (probability_of_correct_predictions total_flips correct_predictions monday_classes correct_monday_predictions) =
    (5 / 11) :=
  sorry

end misha_class_predictions_probability_l96_96584


namespace quadratic_inequality_solution_set_l96_96389

theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x, 1 < x ∧ x < 3 → x^2 < ax + b) : b^a = 81 :=
sorry

end quadratic_inequality_solution_set_l96_96389


namespace ternary_to_decimal_l96_96668

def to_decimal (ternary : Nat) : Nat :=
  match ternary with
  | 121 => 1 * 3^2 + 2 * 3^1 + 1 * 3^0
  | _ => 0

theorem ternary_to_decimal : to_decimal 121 = 16 := by
  sorry

end ternary_to_decimal_l96_96668


namespace minimum_green_points_l96_96795

theorem minimum_green_points : 
  ∃ (n : ℕ), 
    (∀ (configuration : Finset (Point)), 
      configuration.card = 2020 → 
      (∀ (p : Point), p ∈ configuration → 
        (black p → (∃ (gp1 gp2 : Point), gp1 ∈ configuration ∧ gp2 ∈ configuration ∧ 
          ¬black gp1 ∧ ¬black gp2 ∧ distance p gp1 = 2020 ∧ distance p gp2 = 2020))) → 
        n = 45)
:=
sorry

end minimum_green_points_l96_96795


namespace x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l96_96412

variable {x : ℝ}

theorem x_cubed_lt_one_of_x_lt_one (hx : x < 1) : x^3 < 1 :=
sorry

theorem abs_x_lt_one_of_x_lt_one (hx : x < 1) : |x| < 1 :=
sorry

end x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l96_96412


namespace find_largest_and_second_largest_comparisons_avoid_fewer_comparisons_impossible_l96_96115

noncomputable def log2 (n : ℕ) : ℝ := Real.log (n : ℝ) / Real.log 2

-- Proof problem 1: It is possible to find the largest and second largest now ms in n distinct numbers
theorem find_largest_and_second_largest_comparisons (n : ℕ) (m : ℕ) :
  (m - 1 < log2 n ∧ log2 n ≤ m) →
  ∃ comparisons : ℕ, comparisons = n + m - 2 :=
sorry

-- Proof problem 2: It is generally impossible to avoid fewer comparisons than n + m - 2 pairs
theorem avoid_fewer_comparisons_impossible (n : ℕ) (m : ℕ) :
  (m - 1 < log2 n ∧ log2 n ≤ m) →
  ¬ ∃ comparisons : ℕ, comparisons < n + m - 2 :=
sorry

end find_largest_and_second_largest_comparisons_avoid_fewer_comparisons_impossible_l96_96115


namespace correct_propositions_l96_96359

variables (l m : Line) (α β : Plane)
variables (h1 : Perp l α) (h2 : Subset m β)

theorem correct_propositions :
  ( (α ∥ β → Perp l m) ∧
    ¬(Perp l m → α ∥ β) ∧
    ¬(Perp α β → ∥ l m) ∧
    (∥ l m → Perp α β) ) := by
  sorry

end correct_propositions_l96_96359


namespace range_cos_A_l96_96132

theorem range_cos_A {A B C : ℚ} (h : 1 / (Real.tan B) + 1 / (Real.tan C) = 1 / (Real.tan A))
  (h_non_neg_A: 0 ≤ A) (h_less_pi_A: A ≤ π): 
  (Real.cos A ∈ Set.Ico (2 / 3) 1) :=
sorry

end range_cos_A_l96_96132


namespace problem1_problem2_l96_96749

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Prove the solution set for the inequality
theorem problem1 : {x : ℝ | f(x) + f(2*x + 1) ≥ 6} = {x | x ≤ -1} ∪ {x | x ≥ 3} :=
by
  sorry

-- Prove the range of m given conditions
theorem problem2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ x : ℝ, f(x - m) - f(-x) ≤ (4 / a) + (1 / b)) → (-13 ≤ m ∧ m ≤ 5) :=
by
  sorry

end problem1_problem2_l96_96749


namespace solid_of_rotating_rectangle_is_cylinder_l96_96903

theorem solid_of_rotating_rectangle_is_cylinder :
  ∀ (a b : ℝ) (h : a > 0 ∧ b > 0), (rotate_rectangle a b) = cylinder :=
by
  sorry

def rotate_rectangle (a b : ℝ) : Type := 
-- Define the structure of the solid obtained by rotation.
sorry

def cylinder : Type := 
-- Define the structure of a cylinder.
sorry

end solid_of_rotating_rectangle_is_cylinder_l96_96903


namespace number_of_valid_rearrangements_l96_96877

def H := 6
def M := 12
def T := 6
def total_pairs := 12

noncomputable def valid_arrangements : ℕ :=
  (Nat.factorial total_pairs) / ((Nat.factorial H / 2) * (Nat.factorial (M / 2)) * (Nat.factorial (T / 2)))

theorem number_of_valid_rearrangements : valid_arrangements ≈ 78556 :=
sorry

end number_of_valid_rearrangements_l96_96877


namespace taxi_fare_12_2km_l96_96275

def taxi_fare (x : ℝ) : ℝ :=
  if x < 3 then 7
  else if x ≤ 7 then 7 + 1.6 * (x - 3)
  else 13.4 + 2.2 * (x - 7) + 1

theorem taxi_fare_12_2km : taxi_fare 12.2 = 26 :=
by
  sorry

end taxi_fare_12_2km_l96_96275
