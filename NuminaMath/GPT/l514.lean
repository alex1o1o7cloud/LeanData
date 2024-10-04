import Data.Real.Basic
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Analysis.SpecialFunctions.Factorial
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Sign
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Probability.ProbabilitySpace
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rational.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.GCD
import Mathlib.Probability.Distribution
import Mathlib.Probability.ProbabilityBorel
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.MetricSpace.Basic
import Probability.Probability

namespace circle_chord_square_length_l514_514175

-- Define the problem in terms of Lean constructs
theorem circle_chord_square_length :
  ∃ (O₄ O₈ O₁₂ : ℝ) (A₄ A₈ A₁₂ : ℝ) (PQ : ℝ),
    O₄ = 4 ∧
    O₈ = 8 ∧
    O₁₂ = 12 ∧
    A₈ = 8 ∧
    A₄ = 4 ∧
    A₁₂ = (2 * A₈ + A₄) / 3 ∧
    PQ^2 = 4 * (O₁₂^2 - A₁₂^2) → 
    PQ^2 = 3584 / 9 :=
by
  -- Define variables based on conditions
  let O₄ := 4
  let O₈ := 8
  let O₁₂ := 12
  let A₈ := 8
  let A₄ := 4
  let A₁₂ := (2 * A₈ + A₄) / 3
  let s : ℝ := O₁₂^2 - A₁₂^2
  let PQ := real.sqrt(4 * s)

  -- Using given values to verify
  have h1 : O₄ = 4 := rfl
  have h2 : O₈ = 8 := rfl
  have h3 : O₁₂ = 12 := rfl
  have h4 : A₈ = 8 := rfl
  have h5 : A₄ = 4 := rfl
  have h6 : A₁₂ = (2 * A₈ + A₄) / 3 := rfl

  -- Calculate parts
  have h7: A₁₂ = 20 / 3 := rfl
  have h8: PQ^2 = 4 * (12^2 - (20 / 3)^2) := sorry
  have h9: PQ^2 = 4 * (144 - 400 / 9) := sorry
  have h10: PQ^2 = 4 * (1296 / 9 - 400 / 9) := sorry
  have h11: PQ^2 = 4 * (896 / 9) := sorry
  have h12: PQ^2 = 3584 / 9 := sorry

  -- Conclude the theorem
  exact ⟨O₄, O₈, O₁₂, A₄, A₈, A₁₂, PQ, h1, h2, h3, h4, h5, h6, (λ _, h12)⟩

end circle_chord_square_length_l514_514175


namespace diana_owes_l514_514087

-- Define the conditions
def initial_charge : ℝ := 60
def annual_interest_rate : ℝ := 0.06
def time_in_years : ℝ := 1

-- Define the simple interest calculation
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

-- Define the total amount owed calculation
def total_amount_owed (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

-- State the theorem: Diana will owe $63.60 after one year
theorem diana_owes : total_amount_owed initial_charge (simple_interest initial_charge annual_interest_rate time_in_years) = 63.60 :=
by sorry

end diana_owes_l514_514087


namespace necessary_but_not_sufficient_condition_l514_514619

-- Define the lines
def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x + 2 * y - 1 = 0
def line2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, (a + 1) * x - 2 * a * y + 1 = 0

-- Slopes of the lines
noncomputable def slope1 (a : ℝ) : ℝ := -a / 2
noncomputable def slope2 (a : ℝ) : ℝ := (a + 1) / (-2 * a)

-- Perpendicularity condition
def perpendicular (a : ℝ) : Prop := slope1 a * slope2 a = -1

-- The proof statement
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  perpendicular a ↔ (a = 0 ∨ a = 3) :=
sorry

end necessary_but_not_sufficient_condition_l514_514619


namespace angle_between_2a_minus_b_and_2b_minus_a_l514_514273

variable (a b : ℝ^3)
variable ha : ‖a‖ = 1
variable hb : ‖b‖ = 1
variable hab : a • b = 1 / 2

theorem angle_between_2a_minus_b_and_2b_minus_a :
  let u := (2 : ℝ) • a - b
  let v := (2 : ℝ) • b - a
  real.angle u v = 2 * real.pi / 3 :=
by
  sorry

end angle_between_2a_minus_b_and_2b_minus_a_l514_514273


namespace ellipse_standard_equation_l514_514648

theorem ellipse_standard_equation
    (a b : ℝ)
    (e e1 : ℝ)
    (h_hyperbola : ∀ x y : ℝ, (y^2 / 4 - x^2 / 12 = 1))
    (h_sum_eccentricities : e + e1 = 13/5)
    (h_ellipse_vertices : b = 4)
    (h_foci_e_1_eq_2 : e1 = 2)
    (h_eccentricity_relation : e = 3/5)
    (h_foci_relation : a^2 = b^2 + (a * 3 / 5)^2) :
    (∀ x y : ℝ, (x^2 / 25 + y^2 / 16 = 1)) :=
begin
  sorry,
end

end ellipse_standard_equation_l514_514648


namespace monotonic_interval_l514_514317

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - log x

theorem monotonic_interval (k : ℝ) :
  (∀ x ∈ set.Ioo (k - 1) (k + 1), 0 < x) →
  ¬ (∀ x1 x2 ∈ set.Ioo (k - 1) (k + 1), x1 < x2 → f(x1) ≤ f(x2) ∨ f(x1) ≥ f(x2)) →
  1 ≤ k ∧ k < 3 / 2 :=
by
  sorry

end monotonic_interval_l514_514317


namespace percentage_of_150_l514_514073

theorem percentage_of_150 : (1 / 5 * (1 / 100) * 150 : ℝ) = 0.3 := by
  sorry

end percentage_of_150_l514_514073


namespace math_problem_l514_514159

theorem math_problem : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end math_problem_l514_514159


namespace triangle_area_perimeter_ratio_l514_514924

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514924


namespace solve_digits_l514_514479

theorem solve_digits : ∃ A B C : ℕ, (A = 1 ∧ B = 0 ∧ (C = 9 ∨ C = 1)) ∧ 
  (∃ (X : ℕ), X ≥ 2 ∧ (C = X - 1 ∨ C = 1)) ∧ 
  (A * 1000 + B * 100 + B * 10 + C) * (C * 100 + C * 10 + A) = C * 100000 + C * 10000 + C * 1000 + C * 100 + A * 10 + C :=
by sorry

end solve_digits_l514_514479


namespace golden_triangle_ratio_l514_514462

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_ratio :
  let t := golden_ratio in
  (1 - 2 * Real.sin (27 * Real.pi / 180) ^ 2) / (2 * t * Real.sqrt (4 - t ^ 2)) = 1 / 4 := 
by
  let t := golden_ratio
  sorry

end golden_triangle_ratio_l514_514462


namespace father_age_l514_514204

theorem father_age (F D : ℕ) (h1 : F = 4 * D) (h2 : (F + 5) + (D + 5) = 50) : F = 32 :=
by
  sorry

end father_age_l514_514204


namespace total_spent_l514_514242

variable (B D : ℝ)

-- Conditions
def condition1 : Prop := D = 0.90 * B
def condition2 : Prop := B = D + 15

-- Question
theorem total_spent : condition1 B D ∧ condition2 B D → B + D = 285 := 
by
  intros h
  sorry

end total_spent_l514_514242


namespace find_line_through_point_with_area_l514_514556

noncomputable def line_equation (m b : ℝ) : Prop := ∀ (x y : ℝ), y = m * x + b
noncomputable def point_on_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b
noncomputable def triangle_area (m b : ℝ) : Prop :=
  let x_intercept := - (b / m) in
  let y_intercept := b in
  (1/2) * |x_intercept| * |y_intercept| = 5

theorem find_line_through_point_with_area :
  ∃ (m b : ℝ), 
    point_on_line m b (-5) (-4) ∧ 
    triangle_area m b ∧ 
    (∀ x y : ℝ, y = (8/5) * x + 4 → line_equation m b x y) :=
begin
  sorry
end

end find_line_through_point_with_area_l514_514556


namespace smallest_total_number_of_books_l514_514451

theorem smallest_total_number_of_books (n p c b : ℕ) 
  (h1 : 3 * p = 2 * c) 
  (h2 : 4 * c = 3 * b) 
  (h3 : n = p + c + b) : 
  n ≥ 13 :=
begin
  -- Provide the proof here, setting appropriate values to satisfy the constraints
  sorry
end

end smallest_total_number_of_books_l514_514451


namespace triangle_area_perimeter_ratio_l514_514925

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514925


namespace sqrt_n_pattern_l514_514769

theorem sqrt_n_pattern (n : ℕ) (hn : n ≥ 2) : 
  √(n + n / (n^2 - 1 : ℚ)) = n * √(n / (n^2 - 1)) := 
by 
  -- Proof to be completed here
  sorry

end sqrt_n_pattern_l514_514769


namespace count_valid_pairs_l514_514692

open Nat
open Finset

def satisfies_condition (x y : ℕ) : Prop :=
  max 60 (min x y) = min (max 60 x) y

def valid_pairs_count : ℕ :=
  (univ.filter (λ x => 1 ≤ x ∧ x ≤ 100)).sum (λ x => 
    (univ.filter (λ y => 1 ≤ y ∧ y ≤ 100 ∧ satisfies_condition x y)).card)

theorem count_valid_pairs : valid_pairs_count = 4100 :=
  sorry

end count_valid_pairs_l514_514692


namespace count_good_sequences_l514_514362

def is_good_sequence (seq : List ℕ) : Prop :=
  let last_row := seq.zipWith (λ a b, (a + b) % 2) seq.tail
  seq = last_row

def triangle_first_elements_count (n : ℕ) : ℕ :=
  2 ^ Nat.ceil (n / 2)

theorem count_good_sequences (n : ℕ) : 
  ∃ S : List (List ℕ), 
  (∀ seq ∈ S, is_good_sequence seq ∧ seq.length = n) → 
  S.length = 2 ^ Nat.ceil (n / 2) := 
sorry

end count_good_sequences_l514_514362


namespace repeated_digit_percentage_l514_514690

theorem repeated_digit_percentage (total : ℕ := 90000) (non_repeated_count : ℕ := 9 * 9 * 8 * 7 * 6) : 
  let repeated_count := total - non_repeated_count in
  let y := (repeated_count : ℚ) / total * 100 in
  y ≈ 69.8 :=
by
  sorry

end repeated_digit_percentage_l514_514690


namespace range_of_k_l514_514514

theorem range_of_k (k : Real) : 
  (∀ (x y : Real), x^2 + y^2 - 12 * x - 4 * y + 37 = 0)
  → ((k < -Real.sqrt 2) ∨ (k > Real.sqrt 2)) :=
by
  sorry

end range_of_k_l514_514514


namespace triangle_probability_correct_l514_514247

noncomputable def lengths : List ℕ := [2, 3, 5, 6]

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_combinations : List (ℕ × ℕ × ℕ) :=
  [(2, 3, 5), (2, 3, 6), (2, 5, 6), (3, 5, 6)]

def count_valid_triangles (combs : List (ℕ × ℕ × ℕ)) : ℕ :=
  combs.countp (λ (abc : ℕ × ℕ × ℕ), 
    triangle_inequality abc.1 abc.2 abc.3)

theorem triangle_probability_correct :
  count_valid_triangles valid_combinations = 2 →
  valid_combinations.length = 4 →
  (count_valid_triangles valid_combinations : ℚ) / valid_combinations.length = 1 / 2 :=
by
  intros
  sorry

end triangle_probability_correct_l514_514247


namespace variance_binom_4_half_l514_514644

-- Define the binomial variance function
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Define the conditions
def n := 4
def p := 1 / 2

-- The target statement
theorem variance_binom_4_half : binomial_variance n p = 1 :=
by
  -- The proof goes here
  sorry

end variance_binom_4_half_l514_514644


namespace marbles_ratio_l514_514535

theorem marbles_ratio :
  ∀ (initial_marbles lost_marbles remaining_marbles : ℕ),
  initial_marbles = 24 →
  lost_marbles = 4 →
  remaining_marbles = 10 →
  let dog_ate := lost_marbles / 2 in
  let marbles_before_giving_away := initial_marbles - lost_marbles - dog_ate in
  let gave_away := marbles_before_giving_away - remaining_marbles in
  gave_away / lost_marbles = 2 :=
by
  intros initial_marbles lost_marbles remaining_marbles h1 h2 h3
  let dog_ate := lost_marbles / 2
  let marbles_before_giving_away := initial_marbles - lost_marbles - dog_ate
  let gave_away := marbles_before_giving_away - remaining_marbles
  sorry

end marbles_ratio_l514_514535


namespace regular_polygon_sides_l514_514117

-- define the condition of the exterior angle of a regular polygon
def exterior_angle_of_regular_polygon (n : ℕ) : ℝ :=
  360 / n

-- state the theorem that for a regular polygon with an exterior angle of 20 degrees, the number of sides is 18
theorem regular_polygon_sides (h : exterior_angle_of_regular_polygon n = 20) : n = 18 :=
sorry

end regular_polygon_sides_l514_514117


namespace five_coins_not_155_l514_514237

def coin_values : List ℕ := [5, 25, 50]

def can_sum_to (n : ℕ) (count : ℕ) : Prop :=
  ∃ (a b c : ℕ), a + b + c = count ∧ a * 5 + b * 25 + c * 50 = n

theorem five_coins_not_155 : ¬ can_sum_to 155 5 :=
  sorry

end five_coins_not_155_l514_514237


namespace log_add_squared_l514_514658

def f (x : ℝ) : ℝ := log x / log 10

variable (a b : ℝ)
variable (ha : f (a * b) = 1)

theorem log_add_squared (a b : ℝ) (ha : f (a * b) = 1) : f (a ^ 2) + f (b ^ 2) = 2 := by
  sorry

end log_add_squared_l514_514658


namespace prime_sides_triangle_area_not_integer_l514_514446

noncomputable def heron_formula (a b c : ℕ) : ℝ :=
  let p := (a + b + c) / 2
  in real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem prime_sides_triangle_area_not_integer (a b c : ℕ) (ha : nat.prime a) (hb : nat.prime b) (hc : nat.prime c) :
  ¬ ∃ (S : ℤ), S = real.to_int (heron_formula a b c) :=
by sorry

end prime_sides_triangle_area_not_integer_l514_514446


namespace not_all_elements_distinct_l514_514120

open Rational

-- Define the sequence as a function from ℕ to non-negative rational numbers
def sequence (a : ℕ → ℚ) := ∀ m n : ℕ, a m + a n = a (m * n)

-- Define the proof goal: not all elements of the sequence are distinct
theorem not_all_elements_distinct (a : ℕ → ℚ) (h_seq : sequence a) : ∃ m n : ℕ, m ≠ n ∧ a m = a n :=
sorry

end not_all_elements_distinct_l514_514120


namespace Sum_b_n_eq_40_div_11_l514_514723

-- Definitions of the sequences
def a_n (n : ℕ+) : ℚ :=
  (Finset.range (n : ℕ)).sum (λ i, (i + 1) / ((n : ℕ) + 1))

def b_n (n : ℕ+) : ℚ :=
  let a_n_1 := a_n n
      a_n_2 := a_n ⟨(n + 1), by sorry⟩ in
  1 / (a_n_1 * a_n_2)

-- Prove that the sum of the first 10 terms of b{n} is 40 / 11
theorem Sum_b_n_eq_40_div_11 : 
  (Finset.range 10).sum (λ i, b_n ⟨i + 1, by sorry⟩) = 40 / 11 := by
  sorry

end Sum_b_n_eq_40_div_11_l514_514723


namespace abs_sum_l514_514367

noncomputable def x : ℝ := 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + ...))

theorem abs_sum (A B C : ℤ) (hA : A = 5) (hB : B = 3) (hC : C = -22)
  (hx : x = 2 + Real.sqrt 3 / x)
  (h_formula : 1 / ((x + 2) * (x - 3)) = (A + Real.sqrt B) / C) :
  Int.natAbs A + Int.natAbs B + Int.natAbs C = 30 :=
by 
  sorry

end abs_sum_l514_514367


namespace q_divisible_by_5k_l514_514737

noncomputable def largest_power_of_five_dividing_Q : ℕ := 35

theorem q_divisible_by_5k : 
  let Q := (finset.range (299 + 1)).filter (λ n, n % 2 = 1).prod (λ n, (n : ℕ))
  let k := 5
  let l := largest_power_of_five_dividing_Q
  ∃ l : ℕ, l = 35 ∧ Q % (5 ^ l) = 0 :=
sorry

end q_divisible_by_5k_l514_514737


namespace inequality_additive_l514_514741

variable {a b c d : ℝ}

theorem inequality_additive (h1 : a > b) (h2 : c > d) : a + c > b + d :=
by
  sorry

end inequality_additive_l514_514741


namespace non_intersecting_chords_20points_10chords_l514_514450

theorem non_intersecting_chords_20points_10chords : 
  let a : ℕ → ℕ := λ n, if n = 0 then 1 else 
                           if n = 1 then 1 else 
                           ∑ i in finset.range n, a i * a (n - i - 1) in
  a 10 = 16796 :=
by
  let a : ℕ → ℕ := λ n, if n = 0 then 1 else 
                              if n = 1 then 1 else 
                              ∑ i in finset.range n, a i * a (n - i - 1)
  exact sorry

end non_intersecting_chords_20points_10chords_l514_514450


namespace equilateral_triangle_ratio_l514_514912

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514912


namespace find_n_collinear_vectors_l514_514670

theorem find_n_collinear_vectors 
    (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ) (h_collinear : (b - a).fst / c.fst = (b - a).snd / c.snd) :
    b = (-3, 3) :=
by
  have ha : a = (1, 2) := sorry
  have hc : c = (4, -1) := sorry
  let x : ℝ := b.fst
  have h_vec_diff : (b - a) = (x - 1, 1) := sorry
  have h_eq : (x - 1) / 4 = 1 / -1 := h_collinear
  sorry

end find_n_collinear_vectors_l514_514670


namespace train_speed_is_60_0131_l514_514522

noncomputable def train_speed (speed_of_man_kmh : ℝ) (length_of_train_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_of_man_ms := speed_of_man_kmh * 1000 / 3600
  let relative_speed := length_of_train_m / time_s
  let train_speed_ms := relative_speed - speed_of_man_ms
  train_speed_ms * 3600 / 1000

theorem train_speed_is_60_0131 :
  train_speed 6 330 17.998560115190788 = 60.0131 := by
  sorry

end train_speed_is_60_0131_l514_514522


namespace number_of_students_like_basketball_but_not_table_tennis_l514_514704

-- Given definitions
def total_students : Nat := 40
def students_like_basketball : Nat := 24
def students_like_table_tennis : Nat := 16
def students_dislike_both : Nat := 6

-- Proposition to prove
theorem number_of_students_like_basketball_but_not_table_tennis : 
  students_like_basketball - (students_like_basketball + students_like_table_tennis - (total_students - students_dislike_both)) = 18 := 
by
  sorry

end number_of_students_like_basketball_but_not_table_tennis_l514_514704


namespace pq_parallel_bc_l514_514346

variables {A B C P Q : Type} [triangle ABC] [projection A P (angle_bisector ABC)] [projection A Q (angle_bisector ACB)]

-- Define the relationship for projections in triangle
def projections (A P Q : Type) [triangle ABC] [projection A P (angle_bisector ABC)] [projection A Q (angle_bisector ACB)] : Prop :=
P lies_on_angle_bisector ABC ∧ Q lies_on_angle_bisector ACB ∧ is_projection A P ∧ is_projection A Q

-- The theorem to prove PQ parallel to BC
theorem pq_parallel_bc (ABC : Type) (A B C P Q : Type) [triangle ABC] [projection A P (angle_bisector ABC)] [projection A Q (angle_bisector ACB)] :
  projections A P Q → PQ ∥ BC :=
by
  sorry

end pq_parallel_bc_l514_514346


namespace min_living_allowance_inequality_l514_514825

variable (x : ℝ)

-- The regulation stipulates that the minimum living allowance should not be less than 300 yuan.
def min_living_allowance_regulation (x : ℝ) : Prop := x >= 300

theorem min_living_allowance_inequality (x : ℝ) :
  min_living_allowance_regulation x ↔ x ≥ 300 := by
  sorry

end min_living_allowance_inequality_l514_514825


namespace maximize_log_power_l514_514360

theorem maximize_log_power (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (hab : a * b = 100) :
  ∃ x : ℝ, (a ^ (Real.logb 10 b)^2 = 10^x) ∧ x = 32 / 27 :=
by
  sorry

end maximize_log_power_l514_514360


namespace square_area_sum_exceeds_l514_514258

noncomputable def sum_of_areas (n : ℕ) : ℝ := 50 * (1 - (1/2)^n)

theorem square_area_sum_exceeds :
  ∃ n : ℕ, sum_of_areas n > 49
:= by
  use 6
  simp [sum_of_areas]
  linarith

end square_area_sum_exceeds_l514_514258


namespace slices_with_both_pepperoni_and_mushrooms_l514_514096

theorem slices_with_both_pepperoni_and_mushrooms (n : ℕ)
  (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ)
  (all_have_topping : ∀ (s : ℕ), s < total_slices → s < pepperoni_slices ∨ s < mushroom_slices ∨ s < (total_slices - pepperoni_slices - mushroom_slices) )
  (total_condition : total_slices = 16)
  (pepperoni_condition : pepperoni_slices = 8)
  (mushroom_condition : mushroom_slices = 12) :
  (8 - n) + (12 - n) + n = 16 → n = 4 :=
sorry

end slices_with_both_pepperoni_and_mushrooms_l514_514096


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514892

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514892


namespace number_of_moves_even_l514_514510

theorem number_of_moves_even (s : ℕ) (h₀: s = 2021 ^ 2021) :
  (∀ (a b : ℕ), (b > 0) → (∃ k : ℕ, 2^k = b) → (∃ m : ℕ, (s = a + b) ∧ (a = 2^m ∨ b = 2^m))) →
  (∀ (c : ℕ), (∃ n : ℕ, 2^n = c)) →
  ∃ k : ℕ, even k :=
by {
  sorry
}

end number_of_moves_even_l514_514510


namespace correct_choice_l514_514762

variable p : Prop
variable q : Prop

-- Proposition p: The function y=cos 2x has a minimum positive period of π/2
axiom hyp_p : p = false

-- Proposition q: The graph of the function f(x)=sin (x + π/3) has a symmetry axis at x=π/6
axiom hyp_q : q = true

theorem correct_choice : ¬q = false := by
  rw hyp_q
  exact rfl

end correct_choice_l514_514762


namespace solve_for_x_l514_514408

theorem solve_for_x 
    (x : ℝ) 
    (h : (4 * x - 2) / (5 * x - 5) = 3 / 4) 
    : x = -7 :=
sorry

end solve_for_x_l514_514408


namespace solve_equation_l514_514008

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) : x = 5 / 3 := 
by sorry

end solve_equation_l514_514008


namespace distance_between_parallel_lines_l514_514427

open Real

def lineDistance (A B C1 C2 : ℝ) : ℝ :=
  abs (C2 - C1) / sqrt (A ^ 2 + B ^ 2)

theorem distance_between_parallel_lines :
  let l1 := (1 : ℝ, -1, 1)
  let l2 := (3 : ℝ, -3, 1)
  normalize (A B C : ℝ) := (1, -1, C / A)
  lineDistance 1 (-1) 1 (1 / 3) = sqrt 2 / 3 :=
by
  sorry

end distance_between_parallel_lines_l514_514427


namespace pups_more_than_adults_l514_514352

-- Define the counts of dogs
def H := 5  -- number of huskies
def P := 2  -- number of pitbulls
def G := 4  -- number of golden retrievers

-- Define the number of pups each type of dog had
def pups_per_husky_and_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky_and_pitbull + additional_pups_per_golden_retriever

-- Calculate the total number of pups
def total_pups := H * pups_per_husky_and_pitbull + P * pups_per_husky_and_pitbull + G * pups_per_golden_retriever

-- Calculate the total number of adult dogs
def total_adult_dogs := H + P + G

-- Prove that the number of pups is 30 more than the number of adult dogs
theorem pups_more_than_adults : total_pups - total_adult_dogs = 30 :=
by
  -- fill in the proof later
  sorry

end pups_more_than_adults_l514_514352


namespace probability_expression_neg_half_probability_both_expression_neg_sixth_l514_514842

-- Define the operations
def operations := { (+), (-), ( * ), ( / ) }

-- Helper definitions to express the results of the operations
def apply_op (op : (ℝ → ℝ → ℝ)) (a b : ℝ) : ℝ := op a b

-- Guarantee the operations and define the conditions
variable (a : ℝ) (b : ℝ)
def is_negative (a : ℝ) : Prop := a < 0

def expression_prob_eq_half : Prop :=
  let results := Set.image (λ op, apply_op op 2 (-1/2)) operations
  let negatives := results.filter is_negative
  (negatives.card : ℤ)/(operations.card : ℤ) = 1/2

def both_expression_prob_eq_sixth : Prop :=
  let pairs := operations.prod operations
  let negatives := pairs.filter (λ (op1 op2), 
    is_negative (apply_op op1 2 (-1/2)) ∧ 
    is_negative (apply_op op2 2 (-1/2)))
  (negatives.card : ℤ)/(pairs.card : ℤ) = 1/6

-- Statements for Lean
theorem probability_expression_neg_half : expression_prob_eq_half :=
by { sorry }

theorem probability_both_expression_neg_sixth : both_expression_prob_eq_sixth :=
by { sorry }

end probability_expression_neg_half_probability_both_expression_neg_sixth_l514_514842


namespace base_rate_is_12_l514_514470

noncomputable def base_rate_of_second_telephone_company : ℕ :=
  let rate_united := 9 + 60 * 25 / 100
  let rate_second := 60 * 20 / 100
  rate_united - rate_second

theorem base_rate_is_12 :
  base_rate_of_second_telephone_company = 12 :=
by {
  let rate_united := 9 + 60 * 25 / 100
  let rate_second := 60 * 20 / 100
  have h : 24 = rate_united := by norm_num,
  have h' : 12 = rate_second := by norm_num,
  rw [h, h'],
  simp,
  norm_num,
  sorry
}

end base_rate_is_12_l514_514470


namespace intersection_A_B_l514_514301

def A : Set ℝ := { y : ℝ | ∃ x, x > 1 ∧ y = log 3 x }
def B : Set ℝ := { y : ℝ | ∃ x, x > 1 ∧ y = (1 / 3) ^ x }

theorem intersection_A_B : A ∩ B = { y : ℝ | 0 < y ∧ y < 1 / 3 } :=
by
  sorry

end intersection_A_B_l514_514301


namespace mode_and_median_are_8_l514_514982

open Finset

/-- A problem to verify that in a given dataset, the mode and median are both equal to 8. -/
theorem mode_and_median_are_8 :
  let data := [11, 9, 7, 8, 6, 8, 12, 8] in
  let mode := (data.filter(λ x, x = 8)).length in
  let sorted_data := sort data in
  let median_index := data.length / 2 in
  mode = 3 ∧ sorted_data.nth median_index = 8 :=
by
  sorry

end mode_and_median_are_8_l514_514982


namespace point_on_transformed_plane_l514_514090

def point_in_transformed_plane (A : ℝ × ℝ × ℝ) (plane : ℝ → ℝ → ℝ → ℝ) (k : ℝ) : Prop :=
  plane (A.1) (A.2) (A.3) = 0

theorem point_on_transformed_plane :
  let A := (2, 3, -2)
  let k := -(4/3 : ℝ)
  let plane := λ x y z : ℝ, 3 * x - 2 * y + 4 * z - 6
  let transformed_plane := λ x y z : ℝ, 3 * x - 2 * y + 4 * z + 8
  point_in_transformed_plane A transformed_plane k :=
by
  sorry

end point_on_transformed_plane_l514_514090


namespace part1_part2_l514_514296

-- Define the first part of the problem
theorem part1 (a b : ℝ) :
  (∀ x : ℝ, |x^2 + a * x + b| ≤ 2 * |x - 4| * |x + 2|) → (a = -2 ∧ b = -8) :=
sorry

-- Define the second part of the problem
theorem part2 (a b m : ℝ) :
  (∀ x : ℝ, x > 1 → x^2 + a * x + b ≥ (m + 2) * x - m - 15) → m ≤ 2 :=
sorry

end part1_part2_l514_514296


namespace einstein_total_amount_l514_514560

noncomputable def total_amount_raised (pizza_price fries_price soda_price : ℝ) (n_pizzas n_fries n_sodas : ℕ) : ℝ :=
  (n_pizzas * pizza_price) + (n_fries * fries_price) + (n_sodas * soda_price)

def amount_needed : ℝ := 258

theorem einstein_total_amount (pizza_price fries_price soda_price : ℝ)
  (n_pizzas n_fries n_sodas : ℕ) :
  pizza_price = 12 ∧ fries_price = 0.30 ∧ soda_price = 2 →
  n_pizzas = 15 ∧ n_fries = 40 ∧ n_sodas = 25 →
  total_amount_raised pizza_price fries_price soda_price n_pizzas n_fries n_sodas + amount_needed = 500 :=
by
  intros h1 h2
  cases h1 with h_pizza_price h_fries_soda_price
  cases h_fries_soda_price with h_fries_price h_soda_price
  cases h2 with h_n_pizzas h_n_fries_sodas
  cases h_n_fries_sodas with h_n_fries h_n_sodas
  simp [total_amount_raised, h_pizza_price, h_fries_price, h_soda_price, h_n_pizzas, h_n_fries, h_n_sodas, amount_needed]
  norm_num
  sorry

end einstein_total_amount_l514_514560


namespace monic_cubic_polynomial_has_root_l514_514211

noncomputable def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem monic_cubic_polynomial_has_root :
  Q (Real.cbrt 3 + 2) = 0 :=
sorry

end monic_cubic_polynomial_has_root_l514_514211


namespace cube_root_of_neg_eight_l514_514426

theorem cube_root_of_neg_eight : ∃ x : ℝ, x ^ 3 = -8 ∧ x = -2 := by 
  sorry

end cube_root_of_neg_eight_l514_514426


namespace triangle_area_perimeter_ratio_l514_514922

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514922


namespace find_k_l514_514743

noncomputable def arithmetic_sequence (b_1 d : ℝ) (n : ℕ) : ℝ :=
  b_1 + (n - 1) * d

theorem find_k (b_1 d : ℝ) (h1 : arithmetic_sequence b_1 d 5 + arithmetic_sequence b_1 d 8 + arithmetic_sequence b_1 d 11 = 25)
               (h2 : ∑ n in finset.range (16 - 5 + 1), arithmetic_sequence b_1 d (n + 5) = 120)
               (h3 : ∃ k, arithmetic_sequence b_1 d k = 20) :
  ∃ k, arithmetic_sequence b_1 d k = 20 ∧ k = 14 :=
sorry

end find_k_l514_514743


namespace min_value_expression_l514_514742

open Real

theorem min_value_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 27) :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 60 := 
  sorry

end min_value_expression_l514_514742


namespace number_of_distinct_fixed_points_l514_514366

-- Define the sequence of functions
noncomputable def g (x : ℝ) : ℝ := (x + 8) / (x - 1)

noncomputable def g_seq : ℕ → (ℝ → ℝ)
| 0     := g
| (n+1) := g ∘ g_seq n

theorem number_of_distinct_fixed_points :
  ∃ n : ℕ, ∀ x : ℝ, g_seq n x = x → x = 4 ∨ x = -2 → finset.card { x : ℝ | ∃ n : ℕ, g_seq n x = x } = 2 :=
  sorry

end number_of_distinct_fixed_points_l514_514366


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514899

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514899


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514891

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514891


namespace magnitude_of_a_plus_2b_l514_514381

variables (e1 e2 : EuclideanSpace ℝ (Fin 2))

def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ‖v‖ = 1

def is_perpendicular (v1 v2 : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ⟪v1, v2⟫ = 0

theorem magnitude_of_a_plus_2b 
  (h1 : is_unit_vector e1)
  (h2 : is_unit_vector e2)
  (h3 : is_perpendicular e1 e2) :
  let a := 2 • e1 - e2,
      b := e2 in
  ‖a + 2 • b‖ = Real.sqrt 5 :=
by
  sorry

end magnitude_of_a_plus_2b_l514_514381


namespace smallest_m_l514_514543

noncomputable def b_sequence (n : ℕ) : ℕ → ℕ
| 0 => 2
| (n+1) => b_sequence n + (sorry : ℕ) -- To be defined

theorem smallest_m (m : ℕ) (h : m > 1) (h_int : ∃ b : ℕ, b = b_sequence m) : m = 13 :=
sorry

end smallest_m_l514_514543


namespace ratio_P_S_l514_514434

theorem ratio_P_S (S N P : ℝ) 
  (hN : N = S / 4) 
  (hP : P = N / 4) : 
  P / S = 1 / 16 := 
by 
  sorry

end ratio_P_S_l514_514434


namespace tiles_reduction_operations_l514_514121
-- Importing all mathematical capabilities in Lean

-- Defining the initial condition and operations
noncomputable theory
open Set

def initial_tiles : Set ℕ := {n | 1 ≤ n ∧ n ≤ 144}

def remove_perfect_squares (tiles : Set ℕ) : Set ℕ :=
  tiles \ {n | ∃ m, m ^ 2 = n}

def renumber (tiles : Set ℕ) : Set ℕ :=
  (Finset.image tiles.to_finset (λ n, n - ↑(tiles.to_finset.findIdx (λ x, x = n)) + 1)).to_set

-- Function to simulate multiple operations until one tile remains and count the operations
def operations_until_one_tile : ℕ → Set ℕ → ℕ
| count, tiles =>
  if tiles.size = 1 then count
  else
    let new_tiles := renumber (remove_perfect_squares tiles)
    operations_until_one_tile (count + 1) new_tiles

-- Initial tiles set size and answers
def initial_set_size := initial_tiles.size

-- The main theorem statement
theorem tiles_reduction_operations :
  operations_until_one_tile 0 initial_tiles = 12 :=
sorry

end tiles_reduction_operations_l514_514121


namespace distinct_lines_count_l514_514287

theorem distinct_lines_count: 
  let S := {1, 2, 3, 6, 7, 8}
  ∃ (A B: ℤ), A ∈ S ∧ B ∈ S ∧ A ≠ B ∧ card (finset.filter (λ p: ℤ × ℤ, p.fst ∈ S ∧ p.snd ∈ S ∧ p.fst ≠ p.snd) (S.product S)) = 26 :=
by
  sorry

end distinct_lines_count_l514_514287


namespace integral_1_integral_2_integral_3_integral_4_integral_5_integral_6_l514_514223

-- Integral problem 1
theorem integral_1 (C : ℝ) : 
  ∫ (x : ℝ) in 0..1, (2 * x) / (x^4 + 3) = (1/√3) * arctan (x^2 / √3) + C :=
sorry

-- Integral problem 2
theorem integral_2 (C : ℝ) : 
  ∫ (x : ℝ) in 0..1, sin x / (√(1 + 2 * cos x)) = -√(1 + 2 * cos x) + C :=
sorry

-- Integral problem 3
theorem integral_3 (C a : ℝ) :
  ∫ (x : ℝ) in 0..1, x / (^(x^2 + a)^(1/3)) = (3/4) * (x^2 + a)^(2/3) + C :=
sorry

-- Integral problem 4
theorem integral_4 (C : ℝ) :
  ∫ (x : ℝ) in 0..1, (√(1 + ln x)) / x = (2/3) * (1 + ln x)^(3/2) + C :=
sorry

-- Integral problem 5
theorem integral_5 (C : ℝ) :
  ∫ (y : ℝ) in 0..1, 1 / (√(e^y + 1)) = ln |(√(e^y + 1) - 1) / (√(e^y + 1) + 1)| + C :=
sorry

-- Integral problem 6
theorem integral_6 (C : ℝ) : 
  ∫ (t : ℝ) in 0..1, 1 / (√((1 - t^2)^3)) = (t / (√(1 - t^2))) + C :=
sorry

end integral_1_integral_2_integral_3_integral_4_integral_5_integral_6_l514_514223


namespace Q_divisible_by_5_pow_38_l514_514735

theorem Q_divisible_by_5_pow_38 :
  let Q := ∏ i in finset.range 150, (2 * i + 1)
  ∃ k : ℕ, Q % 5^k = 0 ∧ k = 38 :=
by
  sorry

end Q_divisible_by_5_pow_38_l514_514735


namespace maximum_elements_in_S_l514_514576

-- Definition of the set S with the necessary conditions
def satisfies_conditions (S : Finset ℕ) : Prop :=
  (∀ a ∈ S, a ≤ 100) ∧
  (∀ a b ∈ S, a ≠ b → ∃ c ∈ S, Nat.gcd a c = 1 ∧ Nat.gcd b c = 1) ∧
  (∀ a b ∈ S, a ≠ b → ∃ d ∈ S, Nat.gcd a d > 1 ∧ Nat.gcd b d > 1)

-- Main theorem
theorem maximum_elements_in_S : ∃ (S : Finset ℕ), satisfies_conditions S ∧ S.card = 72 := 
sorry

end maximum_elements_in_S_l514_514576


namespace equilateral_triangle_ratio_l514_514969

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514969


namespace cost_to_make_each_pop_l514_514516

-- Define the conditions as given in step a)
def selling_price : ℝ := 1.50
def pops_sold : ℝ := 300
def pencil_cost : ℝ := 1.80
def pencils_to_buy : ℝ := 100

-- Define the total revenue from selling the ice-pops
def total_revenue : ℝ := pops_sold * selling_price

-- Define the total cost to buy the pencils
def total_pencil_cost : ℝ := pencils_to_buy * pencil_cost

-- Define the total profit
def total_profit : ℝ := total_revenue - total_pencil_cost

-- Define the cost to make each ice-pop
theorem cost_to_make_each_pop : total_profit / pops_sold = 0.90 :=
by
  sorry

end cost_to_make_each_pop_l514_514516


namespace right_handed_players_total_l514_514388

theorem right_handed_players_total (total_players : ℕ) (throwers : ℕ) 
  (all_throwers_right_handed : ∀ t, t < throwers → t % 2 = 0) 
  (one_third_non_throwers_left_handed : ∀ non_throwers, non_throwers = total_players - throwers → non_throwers / 3) : 
  let left_handed_non_throwers := (total_players - throwers) / 3 in
  let right_handed_non_throwers := (total_players - throwers) - left_handed_non_throwers in
  total_players = 67 ∧ throwers = 37 ∧ all_throwers_right_handed ∧ one_third_non_throwers_left_handed (total_players - throwers) 
  → right_handed_non_throwers + throwers = 57 :=
by sorry

end right_handed_players_total_l514_514388


namespace equilateral_triangle_ratio_l514_514936

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514936


namespace parallelogram_area_l514_514489

theorem parallelogram_area (base height : ℝ) (h_base : base = 14) (h_height : height = 24) :
  base * height = 336 :=
by 
  rw [h_base, h_height]
  sorry

end parallelogram_area_l514_514489


namespace x5_minv_x5_eq_zero_l514_514643

theorem x5_minv_x5_eq_zero (x : ℂ) (hx : x + 1/x = real.sqrt 2) : x^5 - 1/x^5 = 0 := 
sorry

end x5_minv_x5_eq_zero_l514_514643


namespace light_match_first_l514_514976

-- Define the conditions
def dark_room : Prop := true
def has_candle : Prop := true
def has_kerosene_lamp : Prop := true
def has_ready_to_use_stove : Prop := true
def has_single_match : Prop := true

-- Define the main question as a theorem
theorem light_match_first (h1 : dark_room) (h2 : has_candle) (h3 : has_kerosene_lamp) (h4 : has_ready_to_use_stove) (h5 : has_single_match) : true :=
by
  sorry

end light_match_first_l514_514976


namespace equilateral_triangle_ratio_l514_514878

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514878


namespace sum_of_all_integers_a_l514_514319

theorem sum_of_all_integers_a (a x y : ℤ) :
  (x + 1 ≤ (2 * x - 5) / 3) → (a - x > 1) → (4 + y / (y - 3) = (a - 1) / (3 - y)) → 
  (∃ k : ℕ, y = k) → ∑ a in {a : ℤ | a > -7 ∧ ∃ y : ℤ, y = (13 - a) / 5 ∧ (y = 0 ∨ y ∈ ℕ) ∧ 4 + y / (y - 3) = (a - 1) / (3 - y)}, a = 24 :=
by
  sorry

end sum_of_all_integers_a_l514_514319


namespace three_digit_multiples_of_6_and_9_l514_514309

theorem three_digit_multiples_of_6_and_9 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ k, n = k * 18)}.finite.card = 50 :=
by
  sorry

end three_digit_multiples_of_6_and_9_l514_514309


namespace minimum_distance_l514_514694

-- Define conditions and problem

def lies_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 4 = 0

theorem minimum_distance (P : ℝ × ℝ) (h : lies_on_line P) : P.1^2 + P.2^2 ≥ 8 :=
sorry

end minimum_distance_l514_514694


namespace prove_coordinate_C_l514_514038

noncomputable def findCoordinateC : Prop :=
  ∃ (C : ℝ × ℝ),
    C.2^2 = 4 * C.1 ∧
    (let A := (9 + 4 * Real.sqrt 5, 4 + 2 * Real.sqrt 5);
         B := (9 - 4 * Real.sqrt 5, 4 - 2 * Real.sqrt 5) in
     let mAC := (C.2 - A.2) / (C.1 - A.1);
         mBC := (C.2 - B.2) / (C.1 - B.1) in
     mAC * mBC = -1 ∧ (C = (1, -2) ∨ C = (9, -6)))

theorem prove_coordinate_C : findCoordinateC :=
  sorry

end prove_coordinate_C_l514_514038


namespace hidden_numbers_avg_l514_514137

noncomputable def hidden_numbers := {x : ℤ // x % 2 = 1}

theorem hidden_numbers_avg :
  ∀ (x y z : hidden_numbers), 
    x.val + y.val + z.val = 54 →
    x.val < y.val ∧ y.val < z.val →
    (x.val + y.val + z.val) / 3 = 18 :=
by
  intros x y z h_sum h_order
  have x_is_odd := x.property
  have y_is_odd := y.property
  have z_is_odd := z.property
  rw [int.add_div_left, int.add_div_right, ← int.add_div_right, ← int.add_div_right, ← h_sum]
  exact (54 : ℤ) / 3
  all_goals {
    assumption,
    norm_num }
  sorry

end hidden_numbers_avg_l514_514137


namespace part_a_part_b_l514_514390

-- Define the problem as described
noncomputable def can_transform_to_square (figure : Type) (parts : ℕ) (all_triangles : Bool) : Bool :=
sorry  -- This is a placeholder for the actual implementation

-- The figure satisfies the condition to cut into four parts and rearrange into a square
theorem part_a (figure : Type) : can_transform_to_square figure 4 false = true :=
sorry

-- The figure satisfies the condition to cut into five triangular parts and rearrange into a square
theorem part_b (figure : Type) : can_transform_to_square figure 5 true = true :=
sorry

end part_a_part_b_l514_514390


namespace perfect_square_of_quadratic_l514_514315

theorem perfect_square_of_quadratic {c : ℝ} (h : ∃ b : ℝ, (λ (x : ℝ), (x + b)^2) = (λ (x : ℝ), x^2 + 14*x + c)) : c = 49 :=
by
  sorry

end perfect_square_of_quadratic_l514_514315


namespace smallest_a_b_sum_l514_514635

theorem smallest_a_b_sum :
  ∃ (a b : ℕ), 3^6 * 5^3 * 7^2 = a^b ∧ a + b = 317 := 
sorry

end smallest_a_b_sum_l514_514635


namespace solve_equation_l514_514414

noncomputable def solution (x y z : ℝ) : Prop :=
  (sin x ≠ 0)
  ∧ (cos y ≠ 0)
  ∧ ( (sin x)^2 + 1 / (sin x)^2 ) ^ 3
    + ( (cos y)^2 + 1 / (cos y)^2 ) ^ 3
    = 16 * (sin z)^2

theorem solve_equation (x y z : ℝ) (n m k : ℤ):
  solution x y z ↔
  (x = (π / 2) + π * n) ∧ (y = π * m) ∧ (z = (π / 2) + π * k) :=
by
  sorry

end solve_equation_l514_514414


namespace square_of_chord_l514_514167

-- Definitions of the circles and tangency conditions
def radius1 := 4
def radius2 := 8
def radius3 := 12

-- The internal and external tangents condition
def externally_tangent (r1 r2 : ℕ) : Prop := ∃ O1 O2, dist O1 O2 = r1 + r2
def internally_tangent (r_in r_out : ℕ) : Prop := ∃ O_in O_out, dist O_in O_out = r_out - r_in

theorem square_of_chord :
  externally_tangent radius1 radius2 ∧ 
  internally_tangent radius1 radius3 ∧
  internally_tangent radius2 radius3 →
  (∃ PQ : ℚ, PQ^2 = 3584 / 9) :=
by
  intros h
  sorry

end square_of_chord_l514_514167


namespace perpendicular_lines_l514_514348

-- Definitions based on the conditions
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a, 4, -2)
def line2 (b : ℝ) : ℝ × ℝ × ℝ := (2, -5, b)

-- Slope calculation involved in the problem
def slope (a b c : ℝ) : ℝ := -a / b

-- The statement we need to prove
theorem perpendicular_lines (a b : ℝ) :
  (slope a 4 (-2)) * (slope 2 (-5) b) = -1 ↔ a = 10 :=
by
  sorry

end perpendicular_lines_l514_514348


namespace minimum_of_a_plus_b_l514_514253

theorem minimum_of_a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 4/b = 1) : a + b ≥ 9 :=
by sorry

end minimum_of_a_plus_b_l514_514253


namespace ships_converge_at_shanghai_in_30_days_l514_514844

theorem ships_converge_at_shanghai_in_30_days :
  ∃ (d : ℕ), (d ≠ 0) ∧ (∀ n : ℕ, d = n * Nat.lcm 2 3 5) ∧ d = 30 :=
by
  use 30
  split
  · exact dec_trivial
  split
  · intro n
    exact (Nat.mul_div_cancel _ (Nat.lcm_pos_of_pos_left 2 (Nat.pos_of_ne_zero (by decide)))).symm
  · exact rfl

end ships_converge_at_shanghai_in_30_days_l514_514844


namespace solve_trig_eq_l514_514412

theorem solve_trig_eq (x y z : ℝ) (n m k : ℤ) :
  x ≠ 0 ∧ y ≠ 0 →
  (sin x)^2 ≠ 0 ∧ (cos y)^2 ≠ 0 →
  ((sin x)^2 + 1 / (sin x)^2)^3 + ((cos y)^2 + 1 / (cos y)^2)^3 = 16 * (sin z)^2 →
  ∃ n m k : ℤ, x = π / 2 + π * n ∧ y = π * m ∧ z = π / 2 + π * k :=
begin
  intros h1 h2 h3,
  sorry
end

end solve_trig_eq_l514_514412


namespace complexExpr_eq_l514_514235

def complexExpr : ℂ := (1 / (1 - complex.i)) + (complex.i / (1 + complex.i))

theorem complexExpr_eq : complexExpr = 1 + complex.i := by
  sorry

end complexExpr_eq_l514_514235


namespace angle_C_of_quadrilateral_ABCD_l514_514717

theorem angle_C_of_quadrilateral_ABCD
  (AB CD BC AD : ℝ) (D : ℝ) (h_AB_CD : AB = CD) (h_BC_AD : BC = AD) (h_ang_D : D = 120) :
  ∃ C : ℝ, C = 60 :=
by
  sorry

end angle_C_of_quadrilateral_ABCD_l514_514717


namespace cake_initial_mass_l514_514588

variable (x : ℝ)

-- Conditions:
def initial_mass_of_cake (x : ℝ) : Prop :=
  let karlson_part := 0.4 * x
  let after_karlson := 0.6 * x
  let after_malish := after_karlson - 150
  let freken_bok_part := 0.3 * after_malish + 120
  let remaining_after_freken := after_malish - freken_bok_part
  remaining_after_freken = 90

-- Goal is to prove that x = 750 grams, given the conditions
theorem cake_initial_mass :
  ∃ x, initial_mass_of_cake x ∧ x = 750 :=
by
  apply Exists.intro 750
  unfold initial_mass_of_cake
  split
  sorry

end cake_initial_mass_l514_514588


namespace quadratic_function_inequalities_l514_514256

theorem quadratic_function_inequalities
  (a c : ℝ)
  (h_a_lt_zero : a < 0) :
  let y := λ x, a * (x - 1)^2 + c
  in y 0 < y (Real.sqrt 2) ∧ y (Real.sqrt 2) < y 1 :=
  sorry

end quadratic_function_inequalities_l514_514256


namespace repeating_decimals_count_l514_514604

theorem repeating_decimals_count :
  { n : ℕ | 1 ≤ n ∧ n ≤ 20 ∧ (¬∃ k, n = 9 * k) }.to_finset.card = 18 :=
by
  sorry

end repeating_decimals_count_l514_514604


namespace cricket_bat_profit_percentage_l514_514986

noncomputable def selling_price : ℝ := 850
noncomputable def profit : ℝ := 225

def cost_price (SP : ℝ) (P : ℝ) : ℝ := SP - P
def profit_percentage (P : ℝ) (CP : ℝ) : ℝ := (P / CP) * 100

theorem cricket_bat_profit_percentage (SP : ℝ) (P : ℝ) :
  SP = 850 → P = 225 → profit_percentage P (cost_price SP P) = 36 := by
sry

end cricket_bat_profit_percentage_l514_514986


namespace largest_n_divides_factorial_l514_514224

theorem largest_n_divides_factorial :
  ∃ (n : ℕ), (∀ m : ℕ, 18^m ∣ nat.factorial 25 ↔ m ≤ 5) :=
begin
  sorry
end

end largest_n_divides_factorial_l514_514224


namespace modified_counting_game_53rd_term_l514_514354

theorem modified_counting_game_53rd_term :
  let a : ℕ := 1
  let d : ℕ := 2
  a + (53 - 1) * d = 105 :=
by 
  sorry

end modified_counting_game_53rd_term_l514_514354


namespace percentage_donated_to_orphan_house_l514_514509

def donated_percentage (income distributed_percent_per_child deposited_percent_to_wife end_amount: ℝ) : ℝ :=
  let total_distributed = (distributed_percent_per_child * 3) + deposited_percent_to_wife
  let remaining = income * (1 - total_distributed / 100)
  let donated = remaining - end_amount
  (donated / remaining) * 100

theorem percentage_donated_to_orphan_house
  (income : ℝ)
  (distributed_percent_per_child : ℝ)
  (deposited_percent_to_wife : ℝ)
  (end_amount : ℝ)
  (h_income: income = 800000)
  (h_distributed_percent_per_child: distributed_percent_per_child = 20)
  (h_deposited_percent_to_wife: deposited_percent_to_wife = 30)
  (h_end_amount: end_amount = 40000)
  : donated_percentage income distributed_percent_per_child deposited_percent_to_wife end_amount = 50 :=
by 
  rw [h_income, h_distributed_percent_per_child, h_deposited_percent_to_wife, h_end_amount]
  simp
  sorry

end percentage_donated_to_orphan_house_l514_514509


namespace count_repeating_decimals_l514_514609

theorem count_repeating_decimals (s : Set ℕ) :
  (∀ n, n ∈ s ↔ 1 ≤ n ∧ n ≤ 20 ∧ ¬∃ k, k * 3 = n) →
  (s.card = 14) :=
by 
  sorry

end count_repeating_decimals_l514_514609


namespace smallest_n_for_g4_l514_514744

def g (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc a => acc + (List.range (n + 1)).countP (λ b => a * a + b * b = n)) 0

theorem smallest_n_for_g4 : ∃ n : ℕ, g n = 4 ∧ 
  (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 65
  -- Proof would go here
  sorry

end smallest_n_for_g4_l514_514744


namespace child_stops_incur_yearly_cost_at_age_18_l514_514730

def john_contribution (years: ℕ) (cost_per_year: ℕ) : ℕ :=
  years * cost_per_year / 2

def university_contribution (university_cost: ℕ) : ℕ :=
  university_cost / 2

def total_contribution (years_after_8: ℕ) : ℕ :=
  john_contribution 8 10000 +
  john_contribution years_after_8 20000 +
  university_contribution 250000

theorem child_stops_incur_yearly_cost_at_age_18 :
  (total_contribution n = 265000) → (n + 8 = 18) :=
by
  sorry

end child_stops_incur_yearly_cost_at_age_18_l514_514730


namespace mouse_jumps_28_inches_further_than_grasshopper_l514_514812

theorem mouse_jumps_28_inches_further_than_grasshopper :
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  g_to_m_difference = 28 :=
by
  let g_initial := 19
  let g_obstacle := 3
  let g_actual := g_initial - g_obstacle
  let f_difference := 10
  let f_actual := g_initial + f_difference
  let m_difference := 20
  let m_obstacle := 5
  let m_actual := f_actual + m_difference - m_obstacle
  let g_to_m_difference := m_actual - g_actual
  show g_to_m_difference = 28
  sorry

end mouse_jumps_28_inches_further_than_grasshopper_l514_514812


namespace probability_no_one_receives_own_letter_four_l514_514453

def num_derangements : ℕ → ℕ
| 0     := 1
| 1     := 0
| (n+2) := (n+1) * (num_derangements (n+1) + num_derangements n)

def factorial (n : ℕ) : ℕ := nat.factorial n

def probability_no_one_receives_own_letter (n : ℕ) : ℝ :=
  num_derangements n / factorial n

theorem probability_no_one_receives_own_letter_four :
  probability_no_one_receives_own_letter 4 = 3 / 8 :=
by
  -- We already know that num_derangements 4 = 9 and factorial 4 = 24
  suffices h : num_derangements 4 = 9 ∧ factorial 4 = 24, by
    simp [probability_no_one_receives_own_letter, h.left, h.right],
  split,
  -- these values can be computed, but are written here for clarity:
  exact rfl,  -- From computation: num_derangements 4 = 9
  exact rfl   -- From computation: factorial 4 = 24

end probability_no_one_receives_own_letter_four_l514_514453


namespace factorization_sum_l514_514809

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x ^ 2 + 9 * x + 18 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x ^ 2 + 19 * x + 90 = (x + b) * (x + c)) :
  a + b + c = 22 := by
sorry

end factorization_sum_l514_514809


namespace segment_problem_l514_514545

theorem segment_problem 
  (A C : ℝ) (B D : ℝ) (P Q : ℝ) (x y k : ℝ)
  (hA : A = 0) (hC : C = 0) 
  (hB : B = 6) (hD : D = 9)
  (hx : x = P - A) (hy : y = Q - C) 
  (hxk : x = 3 * k)
  (hxyk : x + y = 12 * k) :
  k = 2 :=
  sorry

end segment_problem_l514_514545


namespace sequence_formula_l514_514486

-- Define the properties of the sequence
axiom seq_prop_1 (a : ℕ → ℝ) (m n : ℕ) (h : m > n) : a (m - n) = a m - a n

axiom seq_increasing (a : ℕ → ℝ) : ∀ n m : ℕ, n < m → a n < a m

-- Formulate the theorem to prove the general sequence formula
theorem sequence_formula (a : ℕ → ℝ) (h1 : ∀ m n : ℕ, m > n → a (m - n) = a m - a n)
    (h2 : ∀ n m : ℕ, n < m → a n < a m) :
    ∃ k > 0, ∀ n, a n = k * n :=
sorry

end sequence_formula_l514_514486


namespace sum_of_roots_l514_514428

-- Define the quadratic equation whose roots are the excluded domain values C and D
def quadratic_eq (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

-- Define C and D as the roots of the quadratic equation
def is_root (x : ℝ) : Prop := quadratic_eq x

-- Define C and D as the specific roots of the given quadratic equation
axiom C : ℝ
axiom D : ℝ

-- Assert that C and D are the roots of the quadratic equation
axiom hC : is_root C
axiom hD : is_root D

-- Statement to prove
theorem sum_of_roots : C + D = 3 :=
by sorry

end sum_of_roots_l514_514428


namespace equilateral_triangle_ratio_l514_514875

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514875


namespace exterior_angle_DEF_correct_l514_514784

-- Define the interior angle of a regular polygon.
def interior_angle (n : ℕ) (h : 3 ≤ n) : ℝ :=
  180 * (n - 2) / n

-- Define the exterior angle given the interior angles of two polygons.
def exterior_angle (interior_angle1 interior_angle2 : ℝ) : ℝ :=
  360 - (interior_angle1 + interior_angle2)

-- Define the conditions:
-- For a regular hexagon (6 sides)
def angle_hexagon : ℝ := interior_angle 6 (by norm_num)

-- For a regular heptagon (7 sides)
def angle_heptagon : ℝ := interior_angle 7 (by norm_num)

-- The measure of the exterior angle DEF
def measure_exterior_angle_DEF : ℝ :=
  exterior_angle angle_hexagon angle_heptagon

-- The theorem to prove the measure of DEF
theorem exterior_angle_DEF_correct : measure_exterior_angle_DEF = 111.43 :=
by sorry

end exterior_angle_DEF_correct_l514_514784


namespace prove_sets_equal_l514_514370

theorem prove_sets_equal
  (a b c d : ℕ)
  (h₁ : 0 ≤ a ∧ a ≤ 99)
  (h₂ : 0 ≤ b ∧ b ≤ 99)
  (h₃ : 0 ≤ c ∧ c ≤ 99)
  (h₄ : 0 ≤ d ∧ d ≤ 99)
  (congr_cond : (101 * a - 100 * 2^a + 101 * b - 100 * 2^b) % 10100 = (101 * c - 100 * 2^c + 101 * d - 100 * 2^d) % 10100) :
  ({a, b} = {c, d}) :=
sorry

end prove_sets_equal_l514_514370


namespace fourth_term_largest_l514_514615

theorem fourth_term_largest (x : ℝ) :
  (5 / 8) < x ∧ x < (20 / 21) ↔ ∀ k ∈ {1, 2, 3, 5, 6, 7, 8, 9, 10}, 
    ∃ T_k T_4, (binomial 10 (k-1)) * (5 ^ (10 - (k-1))) * ((3 * x) ^ (k-1)) < 
                (binomial 10 3) * (5 ^ 7) * ((3 * x) ^ 3) :=
sorry

end fourth_term_largest_l514_514615


namespace polynomial_remainder_l514_514579

-- Define the polynomial
def f (x : ℝ) : ℝ := x^2 - 5 * x + 8

-- Define the value we need to test (a = 3)
def a : ℝ := 3

-- State the theorem that we need to prove
theorem polynomial_remainder : f(a) = 2 :=
by
  -- (Proof would go here, omitted as per instructions)
  sorry

end polynomial_remainder_l514_514579


namespace find_quadratic_function_l514_514017

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant condition
def discriminant_zero (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

-- Given derivative
def given_derivative (x : ℝ) : ℝ := 2 * x + 2

-- Prove that if these conditions hold, then f(x) = x^2 + 2x + 1
theorem find_quadratic_function :
  ∃ (a b c : ℝ), (∀ x, quadratic_function a b c x = 0 → discriminant_zero a b c) ∧
                (∀ x, (2 * a * x + b) = given_derivative x) ∧
                (quadratic_function a b c x = x^2 + 2 * x + 1) := 
by
  sorry

end find_quadratic_function_l514_514017


namespace largest_common_value_less_than_1000_l514_514425

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a = 999 ∧ (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 8 * m) ∧ a < 1000 :=
by
  sorry

end largest_common_value_less_than_1000_l514_514425


namespace average_height_of_trees_l514_514517

theorem average_height_of_trees :
  ∃ (h : ℕ → ℕ), (h 2 = 12) ∧ (∀ i, h i = 2 * h (i+1) ∨ h i = h (i+1) / 2) ∧ (h 1 * h 2 * h 3 * h 4 * h 5 * h 6 = 4608) →
  (h 1 + h 2 + h 3 + h 4 + h 5 + h 6) / 6 = 21 :=
sorry

end average_height_of_trees_l514_514517


namespace crayons_and_erasers_difference_l514_514774

theorem crayons_and_erasers_difference 
  (initial_crayons : ℕ) (initial_erasers : ℕ) (remaining_crayons : ℕ) 
  (h1 : initial_crayons = 601) (h2 : initial_erasers = 406) (h3 : remaining_crayons = 336) : 
  initial_erasers - remaining_crayons = 70 :=
by
  sorry

end crayons_and_erasers_difference_l514_514774


namespace wheel_radius_increase_l514_514196

theorem wheel_radius_increase :
  ∀ (r_old r_new : ℝ) (d_old d_new : ℕ) (inches_per_mile : ℕ),
  r_old = 12 →
  d_old = 300 →
  d_new = 290 →
  inches_per_mile = 63360 →
  let circumference_old := 2 * Real.pi * r_old,
      rotations_old := d_old / (circumference_old / inches_per_mile : ℝ) in
  let r_new := ((d_old : ℝ) * (circumference_old / inches_per_mile) / (d_new / (circumference_old / inches_per_mile)) : ℝ) / (2 * Real.pi) in
  (r_new - r_old) = 0.25 :=
by {
  intros r_old r_new d_old d_new inches_per_mile h_r_old h_d_old h_d_new h_inches_per_mile,
  sorry
}

end wheel_radius_increase_l514_514196


namespace sin_cos_neg_seven_fifths_l514_514637

theorem sin_cos_neg_seven_fifths (α : ℝ) (h1 : sin (2 * α) = 24 / 25) 
(h2 : α ∈ set.Ioo π (3 * π / 2)) : sin α + cos α = -7 / 5 :=
by
  sorry

end sin_cos_neg_seven_fifths_l514_514637


namespace soccer_camp_afternoon_kids_l514_514051

def num_kids_in_camp : ℕ := 2000
def fraction_going_to_soccer_camp : ℚ := 1 / 2
def fraction_going_to_soccer_camp_in_morning : ℚ := 1 / 4

noncomputable def num_kids_going_to_soccer_camp := num_kids_in_camp * fraction_going_to_soccer_camp
noncomputable def num_kids_going_to_soccer_camp_in_morning := num_kids_going_to_soccer_camp * fraction_going_to_soccer_camp_in_morning
noncomputable def num_kids_going_to_soccer_camp_in_afternoon := num_kids_going_to_soccer_camp - num_kids_going_to_soccer_camp_in_morning

theorem soccer_camp_afternoon_kids : num_kids_going_to_soccer_camp_in_afternoon = 750 :=
by
  sorry

end soccer_camp_afternoon_kids_l514_514051


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514952

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514952


namespace monotonic_decreasing_interval_l514_514436

noncomputable def f (x : ℝ) := Real.sin (2 * x + π / 6)

// Prove that the interval (π/6, 2π/3) is the monotonic decreasing interval for the function f
theorem monotonic_decreasing_interval : 
  ∀ x, (π / 6 < x ∧ x < 2 * π / 3) → 
  f ' x ≤ 0 :=
sorry  -- the proof part to be filled by the user

end monotonic_decreasing_interval_l514_514436


namespace height_of_A_correct_l514_514632

noncomputable def height_of_A (a : ℝ) (α β : ℝ) (hαβ : α > β) : ℝ :=
a * (Real.sin α) * (Real.sin β) / (Real.sin (α - β))

theorem height_of_A_correct (a : ℝ) (α β : ℝ) (hαβ : α > β) :
  let AB := height_of_A a α β hαβ in
  AB = a * (Real.sin α) * (Real.sin β) / (Real.sin (α - β)) :=
by
  sorry

end height_of_A_correct_l514_514632


namespace percentage_repeated_digits_five_digit_numbers_l514_514683

theorem percentage_repeated_digits_five_digit_numbers : 
  let total_five_digit_numbers := 90000
  let non_repeated_digits_number := 9 * 9 * 8 * 7 * 6
  let repeated_digits_number := total_five_digit_numbers - non_repeated_digits_number
  let y := (repeated_digits_number.toFloat / total_five_digit_numbers.toFloat) * 100 
  y = 69.8 :=
by
  sorry

end percentage_repeated_digits_five_digit_numbers_l514_514683


namespace simplify_expression_l514_514083

theorem simplify_expression (a b : ℝ) : a + (5 * a - 3 * b) = 6 * a - 3 * b :=
by sorry

end simplify_expression_l514_514083


namespace base_9_units_digit_of_sum_l514_514232

def base_n_units_digit (n : ℕ) (a : ℕ) : ℕ :=
a % n

theorem base_9_units_digit_of_sum : base_n_units_digit 9 (45 + 76) = 2 :=
by
  sorry

end base_9_units_digit_of_sum_l514_514232


namespace find_PF2_value_l514_514663

-- Definition of the hyperbola and variables
def hyperbola (x y : ℝ) (b : ℝ) : Prop := (x^2 / 36) - (y^2 / (b^2)) = 1
def eccentricity (a c : ℝ) : ℝ := c / a

theorem find_PF2_value (b : ℝ) (h_b_pos : b > 0)
  (h_ecc : eccentricity 6 (Real.sqrt (36 + b^2)) = 5 / 3)
  (P : ℝ × ℝ) (h_P_on_hyperbola : hyperbola P.1 P.2 b)
  (h_dist_PF1 : dist P (F1 (Real.sqrt (36 + b^2))) = 15) :
  dist P (F2 (Real.sqrt (36 + b^2))) = 27 :=
sorry

end find_PF2_value_l514_514663


namespace initial_girls_count_l514_514010

theorem initial_girls_count (p : ℕ) (h1 : (60 * p) / 100 = (3 * 4 * 5)) (h2 : ((60 * p / 100) - 3) = ((p + 2) / 2)) : 
  (60 * p / 100) = 24 :=
by {
  have h3 : (60 * p / 100 - 3) / (p + 2) = 1 / 2, from sorry,
  have h4 : 60 * p / 100 - 3 = 0.5 * (p + 2), from sorry,
  have h5 : 0.6 * p - 3 = 0.5 * (p + 2), from sorry,
  have h6 : 0.6 * p - 0.5 * p = 4, from sorry,
  have h7 : 0.1 * p = 4, from sorry,
  have h8 : p = 40, from sorry,
  have h9 : 0.6 * 40 = 24, from sorry,
  exact h9
}

end initial_girls_count_l514_514010


namespace grass_consumption_l514_514056

-- Define the problem conditions
def cows := ℕ
def days := ℕ
def grass := ℕ

structure ProblemConditions :=
  (initial_grass : grass)
  (growth_rate : grass)  -- grass units per day
  (initial_cows : cows)
  (additional_cows : cows)
  (initial_days : days)
  (additional_days : days)
  (total_days_20_cows : cows → days → grass) -- Total consumption by 20 cows
  (total_days_30_cows : cows → days → grass) -- Total consumption by 30 cows

-- Define the conditions based on given problem
def grassland_condition : ProblemConditions :=
{ initial_grass := 840,
  growth_rate := 6,
  initial_cows := 6,
  additional_cows := 10,
  initial_days := 30,
  additional_days := 84,
  total_days_20_cows := λ c d, if c = 20 ∧ d = 60 then 1200 else 0,
  total_days_30_cows := λ c d, if c = 30 ∧ d = 35 then 1050 else 0
}

-- Define the proposition to be proved
theorem grass_consumption (pc : ProblemConditions) : 
  (∃ y : days, pc.initial_grass + pc.growth_rate * y = (pc.initial_cows + pc.additional_cows) * y) :=
sorry

end grass_consumption_l514_514056


namespace tiles_covering_the_floor_l514_514512

-- Defining the conditions
def rectangular_floor_tiles (w : ℕ) (tiles_along_diagonal : ℕ) : Prop :=
  let length := 2 * w in
  let area := w * length in
  let num_tiles := area in
  let diagonal := (w^2 + length^2).sqrt in
  tiles_along_diagonal = 25

-- Theorem statement
theorem tiles_covering_the_floor (w : ℕ) (tiles_along_diagonal : ℕ) (h : rectangular_floor_tiles w tiles_along_diagonal) :
  2 * w^2 = 242 :=
  sorry

end tiles_covering_the_floor_l514_514512


namespace probability_jake_dice_l514_514350

theorem probability_jake_dice :
  (∃ (d1 d2 : ℕ), 1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 
  ((3 ≤ (10 * d1 + d2) / 10 ∧ (10 * d1 + d2) / 10 ≤ 4) ∨
   (3 ≤ (10 * d2 + d1) / 10 ∧ (10 * d2 + d1) / 10 ≤ 4)) ∧ 
  (10 * d1 + d2).toReal ≤ 40 ∧ 
  30 ≤ (10 * d1 + d2).toReal)
  → (11 / 36) := 
sorry

end probability_jake_dice_l514_514350


namespace sum_q_evals_l514_514028

noncomputable def q : ℕ → ℤ := sorry -- definition of q will be derived from conditions

theorem sum_q_evals :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) + (q 7) + (q 8) + (q 9) +
  (q 10) + (q 11) + (q 12) + (q 13) + (q 14) + (q 15) + (q 16) + (q 17) + (q 18) = 456 :=
by
  -- Given conditions
  have h1 : q 1 = 3 := sorry
  have h6 : q 6 = 23 := sorry
  have h12 : q 12 = 17 := sorry
  have h17 : q 17 = 31 := sorry
  -- Proof outline (solved steps omitted for clarity)
  sorry

end sum_q_evals_l514_514028


namespace largest_common_value_under_800_l514_514799

-- Let's define the problem conditions as arithmetic sequences
def sequence1 (a : ℤ) : Prop := ∃ n : ℤ, a = 4 + 5 * n
def sequence2 (a : ℤ) : Prop := ∃ m : ℤ, a = 7 + 8 * m

-- Now we state the theorem that the largest common value less than 800 is 799
theorem largest_common_value_under_800 : 
  ∃ a : ℤ, sequence1 a ∧ sequence2 a ∧ a < 800 ∧ ∀ b : ℤ, sequence1 b ∧ sequence2 b ∧ b < 800 → b ≤ a :=
sorry

end largest_common_value_under_800_l514_514799


namespace doubling_time_l514_514439

theorem doubling_time (N₀ N : ℕ) (t : ℝ) (T : ℝ) 
  (H₀ : N₀ = 1000) 
  (H₁ : N = 500000) 
  (H₂ : t = 53.794705707972525)
  (H3: T = t / Real.logb 2 (N / N₀)) : 
  T ≈ 6 := by
  sorry

end doubling_time_l514_514439


namespace solve_for_a_l514_514618

theorem solve_for_a (x y a : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : x - a * y = 3) : a = -1 :=
sorry

end solve_for_a_l514_514618


namespace crossing_time_l514_514505

noncomputable def speed_km_per_hr := 8
noncomputable def bridge_length_m := 2000
noncomputable def speed_m_per_min := 133.33
noncomputable def time_in_min := 15

theorem crossing_time (s_km_hr : ℕ) (l_m : ℕ) (s_m_min : ℚ) :
  s_km_hr = speed_km_per_hr → l_m = bridge_length_m → 
  s_m_min = speed_m_per_min → time_in_min ≈ l_m / s_m_min :=
by sorry

end crossing_time_l514_514505


namespace probability_X_abs_l514_514379

def X_prob : ℤ → ℚ 
| -1 := 1/3
| 0 := m
| 1 := 1/4
| 2 := 1/6
| _ := 0

theorem probability_X_abs : X_prob(-1) = 1 / 3 → (X_prob(0) + X_prob(1) + X_prob(2)) = 2 / 3 :=
by
  intro h
  sorry

end probability_X_abs_l514_514379


namespace count_repeating_decimals_l514_514606

theorem count_repeating_decimals (s : Set ℕ) :
  (∀ n, n ∈ s ↔ 1 ≤ n ∧ n ≤ 20 ∧ ¬∃ k, k * 3 = n) →
  (s.card = 14) :=
by 
  sorry

end count_repeating_decimals_l514_514606


namespace golden_triangle_ratio_l514_514461

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_ratio :
  let t := golden_ratio in
  (1 - 2 * Real.sin (27 * Real.pi / 180) ^ 2) / (2 * t * Real.sqrt (4 - t ^ 2)) = 1 / 4 := 
by
  let t := golden_ratio
  sorry

end golden_triangle_ratio_l514_514461


namespace math_problem_l514_514157

theorem math_problem : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end math_problem_l514_514157


namespace least_area_in_rectangle_l514_514511

theorem least_area_in_rectangle
  (x y : ℤ)
  (h1 : 2 * (x + y) = 150)
  (h2 : x > 0)
  (h3 : y > 0) :
  ∃ x y : ℤ, (2 * (x + y) = 150) ∧ (x * y = 74) := by
  sorry

end least_area_in_rectangle_l514_514511


namespace min_total_rope_cut_l514_514133

theorem min_total_rope_cut (len1 len2 len3 p1 p2 p3 p4: ℕ) (hl1 : len1 = 52) (hl2 : len2 = 37)
  (hl3 : len3 = 25) (hp1 : p1 = 7) (hp2 : p2 = 3) (hp3 : p3 = 1) 
  (hp4 : ∃ x y z : ℕ, x * p1 + y * p2 + z * p3 = len1 + len2 - len3 ∧ x + y + z ≤ 25) :
  p4 = 82 := 
sorry

end min_total_rope_cut_l514_514133


namespace border_area_l514_514513

theorem border_area (photo_height photo_width border_width : ℕ) (h1 : photo_height = 12) (h2 : photo_width = 16) (h3 : border_width = 3) : 
  let framed_height := photo_height + 2 * border_width 
  let framed_width := photo_width + 2 * border_width 
  let area_of_photo := photo_height * photo_width
  let area_of_framed := framed_height * framed_width 
  let area_of_border := area_of_framed - area_of_photo 
  area_of_border = 204 := 
by
  sorry

end border_area_l514_514513


namespace _l514_514129

noncomputable def geometrical_theorem (A B C : Point) (incircle excircle : Circle) (M N : Point)
  (hBC_touches_incircle : touches_segment BC incircle M)
  (hBC_touches_excircle : touches_segment BC excircle N)
  (hangle_BAC : angle A B C = 2 * angle M A N) : BC = 2 * MN := 
sorry

end _l514_514129


namespace regular_polygon_sides_l514_514118

-- define the condition of the exterior angle of a regular polygon
def exterior_angle_of_regular_polygon (n : ℕ) : ℝ :=
  360 / n

-- state the theorem that for a regular polygon with an exterior angle of 20 degrees, the number of sides is 18
theorem regular_polygon_sides (h : exterior_angle_of_regular_polygon n = 20) : n = 18 :=
sorry

end regular_polygon_sides_l514_514118


namespace find_f_of_4_l514_514655

def f : ℝ → ℝ
| x := if x ≤ 1 then 2^x - 1 else f (x-2)

theorem find_f_of_4 : f 4 = 0 :=
by
  sorry

end find_f_of_4_l514_514655


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514882

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514882


namespace equilateral_triangle_ratio_l514_514965

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514965


namespace determine_x_l514_514030

theorem determine_x (x : ℝ) 
  (h1 : (9*x^2) + (16*x^2) + (6*x^2) = 1000) :
  x = 10 * Real.sqrt(31) / 31 := 
sorry

end determine_x_l514_514030


namespace line_tangent_to_ellipse_l514_514552

theorem line_tangent_to_ellipse (m : ℝ) (a : ℝ) (b : ℝ) (h_a : a = 3) (h_b : b = 1) :
  m^2 = 1 / 3 := by
  sorry

end line_tangent_to_ellipse_l514_514552


namespace find_AD_l514_514720

noncomputable def AB := 30
noncomputable def cos_A := 4 / 5
noncomputable def sin_C := 2 / 5

theorem find_AD (AB_eq : AB = 30) (angle_ADB_right : ∠ADB = 90) (cos_A_eq : cos_A = 4 / 5) : ∃ AD, AD = 24 :=
by {
  use 24,
  have h : 4 / 5 = 24 / 30 := by norm_num,
  rw ←h at cos_A_eq,
  exact cos_A_eq,
  sorry
}

end find_AD_l514_514720


namespace vandal_evan_cost_l514_514070

theorem vandal_evan_cost :
  ∀ (n : ℕ), n = 100 →
  let init_sides := 4 in
  let max_sides_per_cut := 4 in
  let total_sides := init_sides + n * max_sides_per_cut in
  let pieces := n + 1 in
  let total_cost := ∑ i in (finset.range pieces), (5 - ((total_sides / pieces) : ℕ)) in
  total_cost > 100 :=
by
  sorry

end vandal_evan_cost_l514_514070


namespace smallest_n_for_g4_l514_514746

def g (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc a => acc + (List.range (n + 1)).countP (λ b => a * a + b * b = n)) 0

theorem smallest_n_for_g4 : ∃ n : ℕ, g n = 4 ∧ 
  (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 65
  -- Proof would go here
  sorry

end smallest_n_for_g4_l514_514746


namespace camel_height_in_feet_l514_514327

theorem camel_height_in_feet (h_ht_14 : ℕ) (ratio : ℕ) (inch_to_ft : ℕ) : ℕ :=
  let hare_height := 14
  let camel_height_in_inches := hare_height * 24
  let camel_height_in_feet := camel_height_in_inches / 12
  camel_height_in_feet
#print camel_height_in_feet

example : camel_height_in_feet 14 24 12 = 28 := by sorry

end camel_height_in_feet_l514_514327


namespace train_length_proof_l514_514988

-- Definitions used in the condition
def speed_km_per_hr : ℝ := 120
def time_seconds : ℝ := 12
def length_of_train_in_meters : ℝ := 400

-- Given the speed in km/hr and time in seconds, prove the length of the train is 400 meters
theorem train_length_proof : 
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600 in
  (speed_m_per_s * time_seconds) = length_of_train_in_meters :=
by
  sorry

end train_length_proof_l514_514988


namespace calculate_expression_l514_514162

theorem calculate_expression : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  calc
    10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2
    _ = 10 - 9 + 56 + 6 - 20 + 3 - 2 : by rw [mul_comm 8 7, mul_comm 5 4] -- Perform multiplications
    _ = 1 + 56 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 10 - 9
    _ = 57 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 1 + 56
    _ = 63 - 20 + 3 - 2 : by norm_num  -- Simplify 57 + 6
    _ = 43 + 3 - 2 : by norm_num -- Simplify 63 - 20
    _ = 46 - 2 : by norm_num -- Simplify 43 + 3
    _ = 44 : by norm_num -- Simplify 46 - 2

end calculate_expression_l514_514162


namespace smallest_n_for_g4_l514_514752

def g (n : ℕ) : ℕ :=
  ((sum (λ a, if a > 0 ∧ (∃ b > 0, a^2 + b^2 = n) then 1 else 0)) -
  ((sum (λ ⟨a, b⟩, if a > 0 ∧ b > 0 ∧ a^2 + b^2 = n then 1 else 0)) / 2)) + 1

theorem smallest_n_for_g4 :
  (∃ n : ℕ, n > 0 ∧ g(n) = 4 ∧ ∀ m : ℕ, m > 0 ∧ m < n → g(m) ≠ 4) ∧
  (∀ n : ℕ, n = 65 → g(n) = 4) :=
begin
  sorry
end

end smallest_n_for_g4_l514_514752


namespace ratio_eq_sqrt3_div_2_l514_514941

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514941


namespace coloring_possible_l514_514620

theorem coloring_possible :
  ∃ (color : (Fin 6 → Fin 6 → Fin 5)), 
    (∀i: Fin 6, ∀j k: Fin 6, i ≠ j → i ≠ k → j ≠ k → color i j ≠ color i k) :=
begin
  sorry
end

end coloring_possible_l514_514620


namespace two_connected_graph_contains_cycle_or_bipartite_l514_514589

theorem two_connected_graph_contains_cycle_or_bipartite (r : ℕ) :
  ∃ n : ℕ, ∀ (G : Type*) [graph G], 
    (G.is_minimally_two_connected ∧ G.order ≥ n) → 
    (G.contains_cycle_of_length r ∨ G.contains_complete_bipartite 2 r) :=
sorry

end two_connected_graph_contains_cycle_or_bipartite_l514_514589


namespace find_ab_l514_514810

noncomputable def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

theorem find_ab (a b : ℝ) :
  (f 1 a b = 10) ∧ ((3 * 1^2 - 2 * a * 1 - b = 0)) → (a, b) = (-4, 11) ∨ (a, b) = (3, -3) :=
by
  sorry

end find_ab_l514_514810


namespace exists_k_l514_514324

theorem exists_k (n : ℕ) (x : ℕ → ℝ)
  (h1 : ∑ i in range n, x i = 0)
  (h3 : ∑ i in range n, x i ^ 3 = 0)
  (h5 : ∑ i in range n, x i ^ 5 = 0)
  (h7 : ∑ i in range n, x i ^ 7 = 0)
  (h9 : ∑ i in range n, x i ^ 9 = 0)
  (h11 : ∑ i in range n, x i ^ 11 = 0)
  (h13 : ∑ i in range n, x i ^ 13 = 0)
  (h15 : ∑ i in range n, x i ^ 15 = 0)
  (h17 : ∑ i in range n, x i ^ 17 = 0)
  (h19 : ∑ i in range n, x i ^ 19 = 0)
  (h21 : ∑ i in range n, x i ^ 21 = 1) :
  n = 21 :=
sorry

end exists_k_l514_514324


namespace floor_plus_r_eq_10_3_implies_r_eq_5_3_l514_514217

noncomputable def floor (x : ℝ) : ℤ := sorry -- Assuming the function exists

theorem floor_plus_r_eq_10_3_implies_r_eq_5_3 (r : ℝ) 
  (h : floor r + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_r_eq_10_3_implies_r_eq_5_3_l514_514217


namespace simplify_expression_l514_514151

variable (x : ℝ)

theorem simplify_expression : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3 - 3 * x^3) 
  = (-x^3 - x^2 + 23 * x - 3) :=
by
  sorry

end simplify_expression_l514_514151


namespace min_value_of_function_l514_514820

theorem min_value_of_function : ∃ (x : ℝ), ∀ x' : ℝ, 2 * cos x - 1 ≤ 2 * cos x' - 1 ∧ 2 * cos x - 1 = -3 := by
  sorry

end min_value_of_function_l514_514820


namespace min_additional_links_l514_514050

theorem min_additional_links (n : ℕ) (h_n : 2 ≤ n) :
  ∃ (L : list (ℕ × ℕ)), (L.length ≤ 3 * (n - 1) * (Real.log (Real.log n) / Real.log 2).ceil.toNat) ∧
  ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ n → ∃ path : path i j, path.length ≤ 3 ∧
  ∀ (l : ℕ × ℕ), l ∈ path.edges → (l.1 < l.2) :=
by
  sorry

end min_additional_links_l514_514050


namespace perimeter_of_monster_is_correct_l514_514332

/-
  The problem is to prove that the perimeter of a shaded sector of a circle
  with radius 2 cm and a central angle of 120 degrees (where the mouth is a chord)
  is equal to (8 * π / 3 + 2 * sqrt 3) cm.
-/

noncomputable def perimeter_of_monster (r : ℝ) (theta_deg : ℝ) : ℝ :=
  let theta_rad := theta_deg * Real.pi / 180
  let chord_length := 2 * r * Real.sin (theta_rad / 2)
  let arc_length := (2 * (2 * Real.pi) * (240 / 360))
  arc_length + chord_length

theorem perimeter_of_monster_is_correct : perimeter_of_monster 2 120 = (8 * Real.pi / 3 + 2 * Real.sqrt 3) :=
by
  sorry

end perimeter_of_monster_is_correct_l514_514332


namespace find_a_l514_514338

noncomputable def A : ℝ × ℝ := (0, 3)
noncomputable def l (x : ℝ) : ℝ := 2 * x - 4

def radius_C : ℝ := 1
noncomputable def center_of_C_on_line (a : ℝ) : Prop := ∃ x y, y = l x ∧ (x, y) = (a, 2 * a - 4)

def moving_point (M : ℝ × ℝ) : Prop :=
  let (x, y) := M in
  sqrt (x^2 + (y - 3)^2) = 2 * sqrt (x^2 + y^2)

theorem find_a (a : ℝ):
  center_of_C_on_line a →
  (0 ≤ a ∧ a ≤ 12 / 5) :=
by
  sorry

end find_a_l514_514338


namespace initial_velocity_l514_514527

noncomputable def displacement (t : ℝ) : ℝ := 3 * t - t^2

theorem initial_velocity :
  (deriv displacement 0) = 3 :=
by
  sorry

end initial_velocity_l514_514527


namespace num_repeating_decimals_1_to_20_l514_514611

theorem num_repeating_decimals_1_to_20 : 
  (∃ count_repeating : ℕ, count_repeating = 18 ∧ 
    ∀ n, 1 ≤ n ∧ n ≤ 20 → ((∃ k, n = 9 * k ∨ n = 18 * k) → false) → 
        (∃ d, (∃ normalized, n / 18 = normalized ∧ normalized.has_repeating_decimal))) :=
sorry

end num_repeating_decimals_1_to_20_l514_514611


namespace min_squares_exceeding_area_l514_514261

theorem min_squares_exceeding_area (n : ℕ) : 
  let a₁ := 25
  let r := 1 / 2
  let S_n := a₁ * 2 * (1 - (r ^ n))
  in S_n > 49 ↔ n ≥ 6 :=
by
  let a₁ := 25
  let r := 1 / 2
  let S_n := a₁ * 2 * (1 - (r ^ n))
  sorry

end min_squares_exceeding_area_l514_514261


namespace cos_sum_to_product_l514_514203

theorem cos_sum_to_product (x y : ℝ) : cos (x + y) + cos (x - y) = 2 * cos x * cos y :=
by sorry

end cos_sum_to_product_l514_514203


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514883

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514883


namespace desired_result_l514_514032

variable {R : Type*} [Field R] (g : R → R)
variable (b c : R)

axiom condition (h : R) : c^2 * g(b) = b^2 * g(c)
axiom g3_ne_zero : g(3) ≠ 0

theorem desired_result : (g(6) - g(4)) / g(3) = 20 / 9 := 
by 
  sorry

end desired_result_l514_514032


namespace length_of_platform_is_300_l514_514105

-- Given conditions
def speed_km_per_hr : ℝ := 72
def time_sec : ℝ := 26
def length_train_m : ℝ := 220

-- Hypothesis of the converted speed in m/s
def speed_m_per_s (v : ℝ) : ℝ := v * (5 / 18)

-- Hypothesis of the total distance covered
def total_distance (v : ℝ) (t : ℝ) : ℝ := (speed_m_per_s v) * t

-- Hypothesis for length of platform
def length_platform (v : ℝ) (t : ℝ) (L_t : ℝ) : ℝ := total_distance v t - L_t

-- Main theorem to be proven
theorem length_of_platform_is_300 :
  length_platform speed_km_per_hr time_sec length_train_m = 300 := sorry

end length_of_platform_is_300_l514_514105


namespace radius_of_tangent_circle_l514_514463

theorem radius_of_tangent_circle (r1 r2 r3 : ℝ) (r : ℝ) 
  (h1 : r1 = 1) (h2 : r2 = 2) (h3 : r3 = 3) 
  (h_tangent : tangent_externally r1 r2 r3):
  r = 6 :=
sorry

end radius_of_tangent_circle_l514_514463


namespace problem1_problem2_l514_514540

theorem problem1 : (-(3 / 4) - (5 / 8) + (9 / 12)) * (-24) = 15 := by
  sorry

theorem problem2 : (-1 ^ 6 + |(-2) ^ 3 - 10| - (-3) / (-1) ^ 2023) = 14 := by
  sorry

end problem1_problem2_l514_514540


namespace triangle_formation_l514_514700

theorem triangle_formation (k : ℝ) :
  (k ≠ 5 ∧ k ≠ -5) ↔ 
  ∃ (l1 l2 l3 : set (ℝ × ℝ)), 
    l1 = {p | p.1 - p.2 = 0} ∧
    l2 = {p | p.1 + p.2 - 2 = 0} ∧
    l3 = {p | 5 * p.1 - k * p.2 - 15 = 0} ∧
    (∀ (a1 b1 a2 b2 : ℝ), 
      a1 ≠ a2 ∧ b1 ≠ b2 → 
      ∃ (p1 p2 : ℝ × ℝ),
        {p1.1 = p2.1, p1.2 = p2.2}) :=
begin
  sorry
end

end triangle_formation_l514_514700


namespace range_of_b_l514_514732

-- Define the region D described by the inequality (x-1)^2 + y^2 <= 1
def region_D (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 ≤ 1

-- Define the line equation x + sqrt(3) * y + b = 0
def line (x y b : ℝ) : Prop :=
  x + (sqrt 3) * y + b = 0

-- The main theorem to prove
theorem range_of_b (b : ℝ) :
  (∃ x y : ℝ, region_D x y ∧ line x y b) → -3 ≤ b ∧ b ≤ 1 :=
sorry

end range_of_b_l514_514732


namespace missing_condition_l514_514998

theorem missing_condition (x y : ℕ) 
  (h1 : y = 2 * x + 9) 
  (h2 : y = 3 * (x - 2)) : 
  "Three people ride in one car, and there are two empty cars" :=
by sorry

end missing_condition_l514_514998


namespace min_area_x_coord_l514_514343

noncomputable def point (x y : ℝ) := (x, y)

def line1 : ℝ → ℝ := λ x, 3 * x - 3

def M : ℝ × ℝ := (1, 2)

def intersects_at_B : ℝ × ℝ := (-1, -6)

def line_eq_through_M (k : ℝ) : ℝ → ℝ := λ x, k * x + 2 - k

def A (k : ℝ) : ℝ × ℝ := (5 - k) / (3 - k), line1 ((5 - k) / (3 - k))

def C (k : ℝ) : ℝ × ℝ := (-1, -2 * k + 2)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * (abs ((fst B) - (fst A)) * abs ((snd C) - (snd A)))

theorem min_area_x_coord
  (k : ℝ) :
  let a := fst (A 2)
  in a = 3 :=
by
  sorry

end min_area_x_coord_l514_514343


namespace volume_difference_and_ratio_l514_514345

-- Define the given conditions
def volume_of_water (rainfall_cm : ℕ) (area_hectares : ℕ) : ℕ := 
  rainfall_cm * area_hectares * 10

-- Constants for the given conditions
def rainfall_A : ℕ := 7
def area_A : ℕ := 2
def rainfall_B : ℕ := 11
def area_B : ℕ := 3.5
def rainfall_C : ℕ := 15
def area_C : ℕ := 5

-- Volumes calculated based on the given conditions
def volume_A : ℕ := volume_of_water rainfall_A area_A
def volume_B : ℕ := volume_of_water rainfall_B area_B
def volume_C : ℕ := volume_of_water rainfall_C area_C

-- Lean theorem statement
theorem volume_difference_and_ratio :
  volume_C - volume_A = 610 ∧
  (volume_A, volume_B, volume_C) = (140, 385, 750) :=
by
  -- Placeholder for now: Replace with the actual proof
  sorry

end volume_difference_and_ratio_l514_514345


namespace chip_sum_values_count_l514_514531

theorem chip_sum_values_count :
  let bagX := {1, 4, 7}
  let bagY := {3, 5, 8}
  let sums := {x + y | x ∈ bagX, y ∈ bagY}
  sums.size = 7 :=
by
  sorry

end chip_sum_values_count_l514_514531


namespace angle_between_a_b_eq_120_degrees_l514_514305

variables {V : Type*} [InnerProductSpace ℝ V]
variables (a b : V)

-- Conditions
def a_norm_eq_one : ‖a‖ = 1 := sorry
def b_norm_eq_two : ‖b‖ = 2 := sorry
def a_b_orthogonal : ⟪a + b, a⟫ = 0 := sorry

-- Statement
theorem angle_between_a_b_eq_120_degrees :
  ∀ (a b : V), ‖a‖ = 1 → ‖b‖ = 2 → ⟪a + b, a⟫ = 0 → 
  real.angle_between a b = real.pi * (2/3 : ℝ) := sorry

end angle_between_a_b_eq_120_degrees_l514_514305


namespace ratio_of_areas_l514_514458

noncomputable def circle_area (r : ℝ) : ℝ :=
  π * r^2

theorem ratio_of_areas (R : ℝ) (h : R > 0) :
  let OQ := R
  let OY := R / 3 in
  circle_area OY / circle_area OQ = 1 / 9 :=
by
  -- This is a placeholder for the proof steps
  sorry

end ratio_of_areas_l514_514458


namespace cos_law_l514_514715

theorem cos_law (a b c : ℝ) (C : ℝ)
  (h₁ : 0 < a ∧ 0 < b ∧ 0 < c)
  (h₂ : SABC : ℝ := 1/2 * sqrt (a^2 * b^2 - (1/2 * (a^2 + b^2 - c^2))^2))
  (h₃ : SABC = 1/2 * a * b * sin C) :
  cos C = (a^2 + b^2 - c^2) / (2 * a * b) :=
sorry

end cos_law_l514_514715


namespace fixed_point_for_all_k_l514_514239

theorem fixed_point_for_all_k (k : ℝ) : (5, 225) ∈ { p : ℝ × ℝ | ∃ k : ℝ, p.snd = 9 * p.fst^2 + k * p.fst - 5 * k } :=
by
  sorry

end fixed_point_for_all_k_l514_514239


namespace eccentricity_of_hyperbola_is_sqrt_10_div_2_l514_514733

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (AF1 AF2 : ℝ)
  (h1 : ∃ A : ℝ × ℝ, (A.1^2 / a^2 - A.2^2 / b^2 = 1) ∧ (∠ (A, (F1, A, F2)) = π / 2))
  (h2 : AF1 = 3 * AF2) : ℝ :=
let t : ℝ := AF2 in
let c : ℝ := (Real.sqrt 10) * t / 2 in
c / a

theorem eccentricity_of_hyperbola_is_sqrt_10_div_2 (a b AF1 AF2 : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ A : ℝ × ℝ, (A.1^2 / a^2 - A.2^2 / b^2 = 1) ∧ (∠ (A, (F1, A, F2)) = π / 2))
  (h2 : AF1 = 3 * AF2) :
  hyperbola_eccentricity a b ha hb AF1 AF2 h1 h2 = Real.sqrt 10 / 2 := 
sorry

end eccentricity_of_hyperbola_is_sqrt_10_div_2_l514_514733


namespace proposition_1_proposition_2_proposition_3_proposition_4_l514_514304

variables {α β : Type*} {l m : α} [planes : β]

-- Definitions of perpendicularity, containment, and parallelism
def perpendicular (a b : β) : Prop := sorry
def contained (a : α) (b : β) : Prop := sorry
def parallel (a b : β) : Prop := sorry
def line_perpendicular (a : α) (b : β) : Prop := sorry
def line_parallel (a b : α) : Prop := sorry

-- Conditions
variable (h1 : line_perpendicular l α)
variable (h2 : contained m β)

theorem proposition_1 (h_parallel : parallel α β) : line_perpendicular l m :=
sorry

theorem proposition_2 (h_perpendicular_plane : perpendicular α β) : line_parallel l m :=
sorry

theorem proposition_3 (h_parallel_line : line_parallel l m) : perpendicular α β :=
sorry

theorem proposition_4 (h_perpendicular_line : line_perpendicular l m) : parallel β α :=
sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l514_514304


namespace count_first_digit_8_l514_514739

-- Define the set S
def S : set ℕ := {k | 0 ≤ k ∧ k ≤ 3000}

-- Number of digits in 8^k
def num_digits (k : ℕ) : ℕ := (Real.log 8 / Real.log 10 * k).floor + 1

-- Condition: 8^3000 has 2714 digits
def condition1 : Prop := num_digits 3000 = 2714

-- Condition: First digit of 8^3000 is 8
def first_digit (n : ℕ) : ℕ := (n : ℝ / 10 ^ (Real.log 10 / Real.log 8 * (n : ℝ)).floor).floor

def condition2 : Prop := first_digit (8^3000) = 8

-- Main theorem stating the number of elements of S where 8^k has 8 as the first digit is roughly 333
theorem count_first_digit_8 : condition1 ∧ condition2 → (∑ k in S, if first_digit (8^k) = 8 then 1 else 0) ≈ 333 := 
sorry

end count_first_digit_8_l514_514739


namespace larger_parts_three_times_smaller_parts_l514_514546

theorem larger_parts_three_times_smaller_parts
    (r : ℝ) -- radius of the circumscribed sphere
    (ABCD : ∀ (v : ℝ^3), (∃ (a b c d : ℝ^3), 
      (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ d) ∧ (d ≠ a) ∧ (a ≠ c) ∧ (b ≠ d) ∧ 
      (dist a b = dist b c) ∧ (dist b c = dist c d) ∧ (dist c d = dist d a))) 
    -- regular tetrahedron
    (planes_midperp : ∀ (h : ℝ^3), (∃ (p : set (ℝ^3)), p = {m ∈ ℝ^3 | m•h = 0}))
    -- planes perpendicular to altitudes at midpoints
    (sphere_division : ∀ (sections : set (set ℝ^3)), 
      sections = { S | ∃ (s : set ℝ^3), is_spherical_segment s r })
    -- sections dividing the surface of the circumscribed sphere
    : (∀ (section: set ℝ^3), section ∈ sphere_division → 
      (is_larger_section section → area section = 3 * area (smaller_corresponding_section section))) :=
sorry

end larger_parts_three_times_smaller_parts_l514_514546


namespace maximum_contribution_l514_514088

theorem maximum_contribution (n : ℕ) (t : ℝ) (x : ℕ → ℝ) (h1 : n = 12) (h2 : t = 20) (h3 : ∀ i, 1 ≤ x i) (h4 : ∑ i in finset.range n, x i = t) : 
  ∃ i, x i = 9 :=
by
  have h5 : ∑ i in finset.range (n - 1), x i ≥ 11 := sorry
  have h6 : (∃ i, x i = t - 11) := sorry
  exact h6

end maximum_contribution_l514_514088


namespace max_trees_cut_l514_514092

theorem max_trees_cut (n : ℕ) (h : n = 2001) :
  (∃ m : ℕ, m = n * n ∧ ∀ (x y : ℕ), x < n ∧ y < n → (x % 2 = 0 ∧ y % 2 = 0 → m = 1001001)) := sorry

end max_trees_cut_l514_514092


namespace least_number_of_stamps_l514_514000

theorem least_number_of_stamps : ∃ n, n ≡ 2 [MOD 7] ∧ n ≡ 3 [MOD 4] ∧ n = 23 := 
by {
  use 23,
  split,
  { norm_num, },
  split,
  { norm_num, },
  -- Conclusion that n = 23
  refl,
}

end least_number_of_stamps_l514_514000


namespace sum_of_ages_l514_514384

-- Given conditions and definitions
variables (M J : ℝ)

def condition1 : Prop := M = J + 8
def condition2 : Prop := M + 6 = 3 * (J - 3)

-- Proof goal
theorem sum_of_ages (h1 : condition1 M J) (h2 : condition2 M J) : M + J = 31 := 
by sorry

end sum_of_ages_l514_514384


namespace central_angle_unchanged_l514_514443

theorem central_angle_unchanged (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0) :
  (s / r) = (2 * s / (2 * r)) :=
by
  sorry

end central_angle_unchanged_l514_514443


namespace inscribed_triangle_area_l514_514166

theorem inscribed_triangle_area (r : ℝ) (h_r : r = 8) :
  (∃ (base height area : ℝ),
    base = 2 * r ∧ height = r ∧ area = (1/2) * base * height ∧ area = 64) :=
by
  use [2 * r, r, (1/2) * (2 * r) * r]
  split
  sorry

end inscribed_triangle_area_l514_514166


namespace triangle_to_20_sided_polygon_l514_514721

noncomputable def triangle := Type -- Definition of a triangle

-- Definition of a 20-sided polygon
noncomputable def is_20_sided_polygon (P : Type) : Prop :=
  exists (sides : fin 20 → ℝ), P = polygon sides

-- The theorem to prove based on the given problem
theorem triangle_to_20_sided_polygon (Δ : triangle) :
  ∃ parts : list triangle, parts.length = 2 ∧ 
                            is_20_sided_polygon (assemble_parts parts) :=
sorry

end triangle_to_20_sided_polygon_l514_514721


namespace prove_equal_l514_514240

def τ (k : ℕ) : ℕ := 
  if k = 0 then 0 else ((finset.range (k + 1)).filter (λ d, k % d = 0)).card

def condition (any_n : ℕ) (a b: ℕ) : Prop := 
  τ(τ(a * any_n)) = τ(τ(b * any_n))

theorem prove_equal (a b : ℕ) (h : ∀ n : ℕ, condition n a b) : a = b :=
by 
  -- Proof is omitted
  exact sorry

end prove_equal_l514_514240


namespace winning_lottery_ticket_is_random_l514_514081

-- Definitions of the events
inductive Event
| certain : Event
| impossible : Event
| random : Event

open Event

-- Conditions
def boiling_water_event : Event := certain
def lottery_ticket_event : Event := random
def athlete_running_30mps_event : Event := impossible
def draw_red_ball_event : Event := impossible

-- Problem Statement
theorem winning_lottery_ticket_is_random : 
    lottery_ticket_event = random :=
sorry

end winning_lottery_ticket_is_random_l514_514081


namespace trapezoid_angle_bisector_sum_eq_base_l514_514798

theorem trapezoid_angle_bisector_sum_eq_base
  (A B C D K : Point)
  (h_trapezoid : is_trapezoid A B C D)
  (h_base : is_base AD BC)
  (h_angle_bisectors : angle_bisector (angle_at B) (angle_at C) K)
  (h_K_on_AD : K ∈ AD) :
  AD = AB + CD :=
by
  sorry

end trapezoid_angle_bisector_sum_eq_base_l514_514798


namespace friends_meeting_probability_l514_514469

noncomputable def n_value (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2) : ℝ :=
  d - e * Real.sqrt f

theorem friends_meeting_probability (n : ℝ) (d e f : ℝ) (h1 : d = 60) (h2 : e = 30) (h3 : f = 2)
  (H : n = n_value d e f h1 h2 h3) : d + e + f = 92 :=
  by
  sorry

end friends_meeting_probability_l514_514469


namespace repeated_digit_percentage_l514_514689

theorem repeated_digit_percentage (total : ℕ := 90000) (non_repeated_count : ℕ := 9 * 9 * 8 * 7 * 6) : 
  let repeated_count := total - non_repeated_count in
  let y := (repeated_count : ℚ) / total * 100 in
  y ≈ 69.8 :=
by
  sorry

end repeated_digit_percentage_l514_514689


namespace distinct_solutions_equation_number_of_solutions_a2019_l514_514989

theorem distinct_solutions_equation (a : ℕ) (ha : a > 1) : 
  ∃ (x y : ℕ), (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / (a : ℚ)) ∧ x > 0 ∧ y > 0 ∧ (x ≠ y) ∧ 
  ∃ (x₁ y₁ x₂ y₂ : ℕ), (1 / (x₁ : ℚ) + 1 / (y₁ : ℚ) = 1 / (a : ℚ)) ∧
  (1 / (x₂ : ℚ) + 1 / (y₂ : ℚ) = 1 / (a : ℚ)) ∧
  x₁ ≠ y₁ ∧ x₂ ≠ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) := 
sorry

theorem number_of_solutions_a2019 :
  ∃ n, n = (3 * 3) := 
by {
  -- use 2019 = 3 * 673 and divisor count
  sorry 
}

end distinct_solutions_equation_number_of_solutions_a2019_l514_514989


namespace tan_alpha_eq_neg_one_l514_514282

theorem tan_alpha_eq_neg_one (α : ℝ) (h1 : |Real.sin α| = |Real.cos α|)
    (h2 : π / 2 < α ∧ α < π) : Real.tan α = -1 :=
sorry

end tan_alpha_eq_neg_one_l514_514282


namespace advanced_class_students_l514_514672

theorem advanced_class_students (total_students : ℕ) 
                                (percentage_moving : ℕ) 
                                (num_grade_levels : ℕ) 
                                (normal_classes_per_grade : ℕ) 
                                (students_per_normal_class : ℕ) :
  total_students = 1590 →
  percentage_moving = 40 →
  num_grade_levels = 3 →
  normal_classes_per_grade = 6 →
  students_per_normal_class = 32 →
  let moving_students := (percentage_moving * total_students) / 100 in
  let students_per_grade := moving_students / num_grade_levels in
  let normal_class_students := normal_classes_per_grade * students_per_normal_class in
  students_per_grade - normal_class_students = 20 :=
begin
  sorry
end

end advanced_class_students_l514_514672


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514958

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514958


namespace lines_parallel_l514_514761

theorem lines_parallel (m : ℝ) : 
  (m = 2 ↔ ∀ x y : ℝ, (2 * x - m * y - 1 = 0) ∧ ((m - 1) * x - y + 1 = 0) → 
  (∃ k : ℝ, (2 * x - m * y - 1 = k * ((m - 1) * x - y + 1)))) :=
by sorry

end lines_parallel_l514_514761


namespace symmetric_line_equation_l514_514221

theorem symmetric_line_equation (x y : ℝ) :
  let L₁ := 3 * x - 4 * y + 5 = 0,
      axis := x + y = 0,
      L₂ := 4 * x - 3 * y + 5 = 0
  in (L₁ ∧ axis) → L₂ := by
  sorry

end symmetric_line_equation_l514_514221


namespace equilateral_triangle_ratio_l514_514861

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514861


namespace solve_sin_cos_eq_l514_514416

theorem solve_sin_cos_eq (x y z : ℝ) (n m k : ℤ) :
  (sin x ≠ 0 ∧ cos y ≠ 0 ∧ 
  (sin x)^2 + ((sin x)^2)⁻¹ = 2 ∧ 
  (cos y)^2 + ((cos y)^2)⁻¹ = 2 ∧ 
  (sin x)^2 + (cos y)^2 = 1 ) → 
  ( ∃ n m k : ℤ, x = π / 2 + π * n ∧ y = π * m ∧ z = π / 2 + π * k ) := 
sorry

end solve_sin_cos_eq_l514_514416


namespace find_point_P_l514_514617

def point_A : ℝ × ℝ × ℝ := (1, -2, 1)
def point_B : ℝ × ℝ × ℝ := (2, 2, 2)
def point_P (x : ℝ) : ℝ × ℝ × ℝ := (x, 0, 0)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

theorem find_point_P (x : ℝ) :
  (∃ x : ℝ, point_P x = (3, 0, 0)) ↔
  distance (point_P x) point_A = distance (point_P x) point_B :=
sorry

end find_point_P_l514_514617


namespace mean_inequalities_l514_514251

variables {n : ℕ} (x : ℕ → ℝ) [fact (0 < n)]

-- Definitions of means
def harmonic_mean (x : ℕ → ℝ) [fintype (fin n)] : ℝ := n / finset.univ.sum (λ i, 1 / x i)
noncomputable def geometric_mean (x : ℕ → ℝ) [fintype (fin n)] : ℝ := (finset.univ.prod (x)) ^ (1 / n : ℝ)
def arithmetic_mean (x : ℕ → ℝ) [fintype (fin n)] : ℝ := finset.univ.sum (x) / n
noncomputable def quadratic_mean (x : ℕ → ℝ) [fintype (fin n)] : ℝ := real.sqrt (finset.univ.sum (λ i, (x i) ^ 2) / n)

-- The goal statement
theorem mean_inequalities (x : ℕ → ℝ) [fintype (fin n)] : 
  harmonic_mean x ≤ geometric_mean x ∧ 
  geometric_mean x ≤ arithmetic_mean x ∧ 
  arithmetic_mean x ≤ quadratic_mean x :=
sorry

end mean_inequalities_l514_514251


namespace POTOP_correct_l514_514080

def POTOP : Nat := 51715

theorem POTOP_correct :
  (99999 * POTOP) % 1000 = 285 := by
  sorry

end POTOP_correct_l514_514080


namespace eccentricity_of_hyperbola_l514_514814

theorem eccentricity_of_hyperbola (a b : ℝ) (P O F1 F2 : ℝ → ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∃ (x y : ℝ), P = (x, y) ∧ (x^2 / a^2 - y^2 / b^2 = 1))
  (h4 : | |P - O| = 1/2 * |F1 - F2| )
  (h5 : slope (O, P) = sqrt 3) :
  eccentricity a b = sqrt 3 + 1 :=
sorry

end eccentricity_of_hyperbola_l514_514814


namespace ball_passing_l514_514834

theorem ball_passing (m n : ℕ) (h_m : m ≥ 3) (h_n : n ≥ 2) :
  let c_n := (1 : ℚ) / m * ((m - 1) ^ n + (m - 1) * (-1) ^ n) in
  ∃ c_n : ℚ, c_n = (1 : ℚ) / m * ((m - 1) ^ n + (m - 1) * (-1) ^ n) :=
by
  sorry

end ball_passing_l514_514834


namespace golden_triangle_expression_l514_514460

noncomputable def t : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_expression :
  t = (Real.sqrt 5 - 1) / 2 →
  (1 - 2 * (Real.sin (27 * Real.pi / 180))^2) / (2 * t * Real.sqrt (4 - t^2)) = 1 / 4 :=
by
  intro h_t
  have h1 : t = (Real.sqrt 5 - 1) / 2 := h_t
  sorry

end golden_triangle_expression_l514_514460


namespace common_tangent_at_origin_l514_514696

theorem common_tangent_at_origin (a b : ℝ) (f g : ℝ → ℝ) (m : ℝ) 
  (hf : f = λ x, a * Real.cos x) 
  (hg : g = λ x, x^2 + b * x + 1) 
  (common_tangent : f 0 = g 0 ∧ (deriv f 0 = deriv g 0)) : a + b = 1 :=
by
  sorry

end common_tangent_at_origin_l514_514696


namespace product_of_consecutive_integers_divisible_by_factorial_l514_514786

-- Define the conditions of the problem
variables {m n : ℕ}
hypothesis (h : m > n > 0)

-- Define the binomial coefficient
def binomial_coefficient (m n : ℕ) : ℕ := Nat.factorial m / (Nat.factorial n * Nat.factorial (m - n))

-- State the theorem to be proved
theorem product_of_consecutive_integers_divisible_by_factorial 
  (h : m > n > 0) : 
  ∃ k : ℕ, binomial_coefficient m n = k :=
sorry

end product_of_consecutive_integers_divisible_by_factorial_l514_514786


namespace coefficient_x_neg_4_expansion_l514_514340

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the function to calculate the coefficient of the term containing x^(-4)
def coeff_term_x_neg_4 : ℕ :=
  let k := 10
  binom 12 k

theorem coefficient_x_neg_4_expansion :
  coeff_term_x_neg_4 = 66 := by
  -- Calculation here would show that binom 12 10 is indeed 66
  sorry

end coefficient_x_neg_4_expansion_l514_514340


namespace phi_value_l514_514697

theorem phi_value (φ : ℝ) (h : sin (π / 3 + φ) = 0) : φ = 2 * π / 3 :=
sorry

end phi_value_l514_514697


namespace ratio_eq_sqrt3_div_2_l514_514938

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514938


namespace stratified_sampling_l514_514119

theorem stratified_sampling (total_students first_grade_students sample_first_grade : ℕ)
  (third_grade second_grade sample_second_grade : ℕ) :
  total_students = 1290 →
  first_grade_students = 480 →
  second_grade = third_grade + 30 →
  2 * third_grade + 510 = total_students →
  sample_first_grade = 96 →
  sample_second_grade = sample_first_grade * second_grade / first_grade_students :=
begin
  sorry
end

end stratified_sampling_l514_514119


namespace georg_can_identify_fake_coins_l514_514562

theorem georg_can_identify_fake_coins :
  ∀ (coins : ℕ) (baron : ℕ → ℕ → ℕ) (queries : ℕ),
    coins = 100 →
    ∃ (fake_count : ℕ → ℕ) (exaggeration : ℕ),
      (∀ group_size : ℕ, 10 ≤ group_size ∧ group_size ≤ 20) →
      (∀ (show_coins : ℕ), show_coins ≤ group_size → fake_count show_coins = baron show_coins exaggeration) →
      queries < 120 :=
by
  sorry

end georg_can_identify_fake_coins_l514_514562


namespace range_of_a_l514_514020

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x ∈ set.Icc 0 1 ∧ 2^x * (3*x + a) < 1) → a < 1 := 
by 
  intro h
  sorry

end range_of_a_l514_514020


namespace count_divisors_num_divisors_of_221_8_l514_514673

-- Define the conditions
def is_perfect_square (n : ℕ) : Prop := ∃ (m : ℕ), n = m * m
def is_perfect_cube (n : ℕ) : Prop := ∃ (m : ℕ), n = m * m * m
def is_perfect_sixth_power (n : ℕ) : Prop := ∃ (m : ℕ), n = m ^ 6

-- Define the main theorem
theorem count_divisors (P Q : ℕ → Prop)
  (hP : ∀ n, P n → is_perfect_square n)
  (hQ : ∀ n, Q n → is_perfect_cube n)
  (n : ℕ) :
  (P n ∨ Q n) ↔ P n ∪ Q n  :=
sorry

-- Specific application to the problem
theorem num_divisors_of_221_8 :
  let divisors := { d : ℕ | (∃ a b : ℕ, 0 ≤ a ∧ a ≤ 8 ∧ 0 ≤ b ∧ b ≤ 8 ∧ d = (13 ^ a) * (17 ^ b)) } in
  let perfect_squares := { d ∈ divisors | is_perfect_square d } in
  let perfect_cubes := { d ∈ divisors | is_perfect_cube d } in
  30 = (card (perfect_squares ∪ perfect_cubes)) :=
sorry

end count_divisors_num_divisors_of_221_8_l514_514673


namespace find_unit_prices_find_ordering_schemes_l514_514061

def unit_prices (unit_price_JW : ℝ) (unit_price_MBEB : ℝ) : Prop :=
  unit_price_MBEB = 1.4 * unit_price_JW ∧
  (14000 / unit_price_MBEB) - (7000 / unit_price_JW) = 300

theorem find_unit_prices (unit_price_JW unit_price_MBEB : ℝ) :
  unit_prices unit_price_JW unit_price_MBEB → 
  unit_price_JW = 10 ∧ unit_price_MBEB = 14 :=
  sorry

def ordering_schemes (m : ℕ) (total_cost : ℕ) : Prop :=
  3 ≤ m ∧ m ≤ 6 ∧ total_cost ≤ 124 ∧ (
    ∃ n, n = 10 - m ∧ 
    (total_cost = 14 * m + 10 * n) ∧ total_cost = min_total_cost (m + n)
  )

theorem find_ordering_schemes :
  (∃ m total_cost, ordering_schemes m total_cost) →
  (∀ (m : ℕ), m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6) ∧ min_total_cost = 112 :=
  sorry

end find_unit_prices_find_ordering_schemes_l514_514061


namespace smaller_number_is_five_l514_514046

theorem smaller_number_is_five (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end smaller_number_is_five_l514_514046


namespace temperature_difference_l514_514036

theorem temperature_difference (T_high T_low : ℝ) (h1 : T_high = 8) (h2 : T_low = -2) : T_high - T_low = 10 :=
by
  sorry

end temperature_difference_l514_514036


namespace length_DO_eq_8_l514_514805

theorem length_DO_eq_8 
  (A B C D O : Type) 
  [Quadrilateral A B C D] 
  (AB BC CD : ℝ) 
  (h1 : AB = BC) 
  (h2 : BC = CD) 
  (h3 : AO = (8 : ℝ)) 
  (angle_BOC : ℝ) 
  (h4 : angle_BOC = (120 : ℝ)) :
  (length DO = 8) :=
sorry

end length_DO_eq_8_l514_514805


namespace triangle_area_perimeter_ratio_l514_514921

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514921


namespace correct_operation_l514_514981

theorem correct_operation (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 :=
sorry

end correct_operation_l514_514981


namespace volume_of_pyramid_l514_514025

-- The base quadrilateral ABCD.
structure Base_Quadrilateral (A B C D : Type) :=
  (AB : ℝ)
  (AD : ℝ)
  (CB : ℝ)
  (CD : ℝ)
  (perpendicular_AB_BC : Prop)

-- The pyramid SABCD with specified properties.
structure Pyramid (A B C D S : Type) :=
  (base : Base_Quadrilateral A B C D)
  (inclined_lateral_faces_angle : ℝ)
  (volume : ℝ)

theorem volume_of_pyramid (A B C D S : Type) 
  (base : Base_Quadrilateral A B C D) 
  (H : Icospherical H := 60) 
  (V_gt_12 : volume > 12) :
  Pyramid_SABCD A B C D S base H.volume = 12 * sqrt(3) :=
sorry

end volume_of_pyramid_l514_514025


namespace weight_problem_l514_514109

theorem weight_problem (w1 w2 w3 : ℝ) (h1 : w1 + w2 + w3 = 100)
  (h2 : w1 + 2 * w2 + w3 = 101) (h3 : w1 + w2 + 2 * w3 = 102) : 
  w1 ≥ 90 ∨ w2 ≥ 90 ∨ w3 ≥ 90 :=
by
  sorry

end weight_problem_l514_514109


namespace find_a_value_l514_514448

noncomputable def sum_first_n_terms (a : ℝ) (n : ℕ) : ℝ :=
  a * 2^n + a - 2

def a1 (a : ℝ) : ℝ := sum_first_n_terms a 1
def a2 (a : ℝ) : ℝ := sum_first_n_terms a 2 - sum_first_n_terms a 1
def a3 (a : ℝ) : ℝ := sum_first_n_terms a 3 - sum_first_n_terms a 2

theorem find_a_value (a : ℝ) : 
  a2 a ^ 2 = a1 a * a3 a → a = 1 :=
by 
  -- Proof would go here.
  sorry

end find_a_value_l514_514448


namespace num_diagonals_120_length_typical_diagonal_120_5_l514_514577

-- Define the number of sides of the polygon
def n : ℕ := 120

-- Define the side length of the polygon
def s : ℝ := 5.0

-- Define the formula for the number of diagonals in a polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Prove that the number of diagonals in a 120-sided polygon is 7020
theorem num_diagonals_120 : number_of_diagonals n = 7020 := by
  sorry

-- Define the formula for the length of a typical diagonal in a regular polygon
def length_of_typical_diagonal (n : ℕ) (s : ℝ) (k : ℕ) : ℝ := 2 * s * Real.sin (k * Real.pi / n)

-- Define the value for k for a typical diagonal
def k : ℕ := n / 2

-- Prove that the length of a typical diagonal in a regular 120-sided polygon 
-- with each side length 5 cm is 10 cm
theorem length_typical_diagonal_120_5 : length_of_typical_diagonal n s k = 10 := by
  sorry

end num_diagonals_120_length_typical_diagonal_120_5_l514_514577


namespace seating_arrangements_l514_514098

/--
Proof that given 6 Democrats and 5 Republicans, they can be seated around a circular table such that no two Republicans sit next to each other in 86400 distinct ways.
-/
theorem seating_arrangements : 
  ∀ (Democrats Republicans : ℕ) (Democrats_count Republicans_count : Democrats = 6 ∧ Republicans = 5 ∧ no_adjacent_Republicans : Democrats × Republicans → Prop),
  (Democrats = 6) ∧ (Republicans = 5) ∧ no_adjacent_Republicans (Democrats, Republicans) → 
  86400 = num_ways_arrange(Democrats, Republicans) := 
  sorry

end seating_arrangements_l514_514098


namespace highest_consumption_is_may_january_bill_is_85_july_bill_as_function_l514_514336

def charging_rate (consumption: ℕ) : ℝ :=
  if consumption ≤ 50 then 0.5
  else if consumption ≤ 200 then 0.6
  else 0.8

def electricity_bill (consumption: ℕ) : ℝ :=
  if consumption ≤ 50 then 0.5 * consumption
  else if consumption ≤ 200 then 0.5 * 50 + 0.6 * (consumption - 50)
  else 0.5 * 50 + 0.6 * 150 + 0.8 * (consumption - 200)

-- Example conditions for Xiaogang's family
def january_consumption : ℕ := 150
def may_consumption : ℕ := 236

-- Proof Problem 1: Prove May has the highest consumption and the value is 236 kWh
theorem highest_consumption_is_may :
  may_consumption = 236 := by
  sorry

-- Proof Problem 2: Prove the electricity bill for January is 85 yuan
theorem january_bill_is_85 :
  electricity_bill january_consumption = 85 := by
  sorry

-- Proof Problem 3: Prove the electricity bill for July as a function of x kWh
theorem july_bill_as_function (x : ℕ) :
  electricity_bill x = 
    if x ≤ 50 then 0.5 * x
    else if x ≤ 200 then 0.5 * 50 + 0.6 * (x - 50)
    else 0.5 * 50 + 0.6 * 150 + 0.8 * (x - 200) := by
  sorry

end highest_consumption_is_may_january_bill_is_85_july_bill_as_function_l514_514336


namespace ian_investment_percentage_change_l514_514701

theorem ian_investment_percentage_change :
  let initial_investment := 200
  let first_year_loss := 0.10
  let second_year_gain := 0.25
  let amount_after_loss := initial_investment * (1 - first_year_loss)
  let amount_after_gain := amount_after_loss * (1 + second_year_gain)
  let percentage_change := (amount_after_gain - initial_investment) / initial_investment * 100
  percentage_change = 12.5 := 
by
  sorry

end ian_investment_percentage_change_l514_514701


namespace equilateral_triangle_ratio_l514_514871

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514871


namespace number_of_planes_l514_514369

theorem number_of_planes {a b : ℕ} (S : Finset (EuclideanSpace ℝ (fin 3))) (h₁ : S.card = a + b + 3) 
(h₂ : ∀ s ⊆ S, s.card = 4 → ¬ (coplanar (s : set (EuclideanSpace ℝ (fin 3))))) : 
  ∃ n, n = 2 * (a + 1) * (b + 1) :=
by {
  have h₃ : ∀ {s : Finset (EuclideanSpace ℝ (fin 3))}, s ⊆ S → s.card = 3 → ∃ p : EuclideanSpace ℝ (fin 3), (s ∪ {p}) ⊆ S ∧ ∀ q ∈ S \ (s ∪ {p}), q ≠ p ∧ coplanar {s.1, s.2, s.3, q} := sorry,
  sorry
}

end number_of_planes_l514_514369


namespace solve_sin_cos_eq_l514_514417

theorem solve_sin_cos_eq (x y z : ℝ) (n m k : ℤ) :
  (sin x ≠ 0 ∧ cos y ≠ 0 ∧ 
  (sin x)^2 + ((sin x)^2)⁻¹ = 2 ∧ 
  (cos y)^2 + ((cos y)^2)⁻¹ = 2 ∧ 
  (sin x)^2 + (cos y)^2 = 1 ) → 
  ( ∃ n m k : ℤ, x = π / 2 + π * n ∧ y = π * m ∧ z = π / 2 + π * k ) := 
sorry

end solve_sin_cos_eq_l514_514417


namespace equilateral_triangle_ratio_l514_514870

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514870


namespace parabola_focus_directrix_l514_514220

theorem parabola_focus_directrix {x y : ℝ} :
  (∃ a : ℝ, x = - (1 / 8) * y^2) →
  (∃ f : ℝ × ℝ, f = (-(1 / 32), 0)) ∧
  (∃ d : ℝ, x = 1 / 32) :=
begin
  sorry,
end

end parabola_focus_directrix_l514_514220


namespace marbles_left_l514_514186

def initial_marbles : ℕ := 143
def marbles_given_away : ℕ := 73

theorem marbles_left (initial_marbles : ℕ) (marbles_given_away : ℕ) : initial_marbles - marbles_given_away = 70 :=
by
  have h1 := initial_marbles
  have h2 := marbles_given_away
  rw [h1, h2]
  show 143 - 73 = 70
  sorry

end marbles_left_l514_514186


namespace find_value_of_A_l514_514478

-- Define the conditions
variable (A : ℕ)
variable (divisor : ℕ := 9)
variable (quotient : ℕ := 2)
variable (remainder : ℕ := 6)

-- The main statement of the proof problem
theorem find_value_of_A (h : A = quotient * divisor + remainder) : A = 24 :=
by
  -- Proof would go here
  sorry

end find_value_of_A_l514_514478


namespace simplify_expression1_simplify_expression2_l514_514407

variable {a b x y : ℝ}

theorem simplify_expression1 : 3 * a - 5 * b - 2 * a + b = a - 4 * b :=
by sorry

theorem simplify_expression2 : 4 * x^2 + 5 * x * y - 2 * (2 * x^2 - x * y) = 7 * x * y :=
by sorry

end simplify_expression1_simplify_expression2_l514_514407


namespace repeated_digit_percentage_l514_514685

noncomputable def percentage_repeated_digits : ℝ :=
  let total_numbers := 90000
  let non_repeated_digits := 9 * 9 * 8 * 7 * 6
  let repeated_digits := total_numbers - non_repeated_digits
  ((repeated_digits.toReal / total_numbers.toReal) * 100).round / 10

theorem repeated_digit_percentage (y : ℝ) : 
  percentage_repeated_digits = y → y = 69.8 :=
by
  intro h
  have : percentage_repeated_digits = 69.8 := sorry
  rw this at h
  exact h

end repeated_digit_percentage_l514_514685


namespace ratio_of_medium_to_total_l514_514521

-- Define the given conditions as Lean definitions
def total_posters : ℕ := 50
def small_poster_fraction : ℚ := 2 / 5
def large_posters : ℕ := 5

-- Calculate small and medium posters based on given conditions
def small_posters : ℕ := (small_poster_fraction * total_posters).toInt
def medium_posters : ℕ := total_posters - small_posters - large_posters

-- Define the target ratio
def target_ratio : ℚ := 1 / 2

-- State the proof problem
theorem ratio_of_medium_to_total :
  (medium_posters : ℚ) / total_posters = target_ratio :=
by
  sorry

end ratio_of_medium_to_total_l514_514521


namespace determine_r_l514_514554

theorem determine_r (r : ℝ) : 8 = 2^(5 * r + 2) → r = 1 / 5 :=
by
  sorry

end determine_r_l514_514554


namespace avg_age_l514_514024

-- Given conditions
variables (A B C : ℕ)
variable (h1 : (A + C) / 2 = 29)
variable (h2 : B = 20)

-- to prove
theorem avg_age (A B C : ℕ) (h1 : (A + C) / 2 = 29) (h2 : B = 20) : (A + B + C) / 3 = 26 :=
sorry

end avg_age_l514_514024


namespace geometric_sequence_arithmetic_Sn_l514_514365

theorem geometric_sequence_arithmetic_Sn (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (n : ℕ) :
  (∀ n, a n = a1 * q ^ (n - 1)) →
  (∀ n, S n = a1 * (1 - q ^ n) / (1 - q)) →
  (∀ n, S (n + 1) - S n = S n - S (n - 1)) →
  q = 1 :=
by
  sorry

end geometric_sequence_arithmetic_Sn_l514_514365


namespace coeff_x4_expansion_l514_514550

theorem coeff_x4_expansion :
  let exp := (2 + sqrt x - 1 / x^2016)^10
  coeff_of_x4 (exp) = 180 :=
by
  sorry

end coeff_x4_expansion_l514_514550


namespace positive_root_gt_1008_l514_514138

noncomputable def P (x : ℝ) : ℝ := sorry
-- where P is a non-constant polynomial with integer coefficients bounded by 2015 in absolute value
-- Assume it has been properly defined according to the conditions in the problem statement

theorem positive_root_gt_1008 (x : ℝ) (hx : 0 < x) (hroot : P x = 0) : x > 1008 := 
sorry

end positive_root_gt_1008_l514_514138


namespace correct_answer_l514_514847

def coin_events : Finset (Finset Char) := 
  {{'H', 'H', 'H'}, {'H', 'H', 'T'}, {'H', 'T', 'H'}, {'H', 'T', 'T'}, {'T', 'H', 'H'}, {'T', 'H', 'T'}, {'T', 'T', 'H'}, {'T', 'T', 'T'}}

def at_least_one_heads (s : Finset Char) : Prop := ∃ h ∈ s, h = 'H'

def at_most_one_heads (s : Finset Char) : Prop := s.count('H') ≤ 1

def exactly_two_heads (s : Finset Char) : Prop := s.count('H') = 2

def at_least_two_heads (s : Finset Char) : Prop := s.count('H') ≥ 2

def at_most_two_heads (s : Finset Char) : Prop := s.count('H') ≤ 2

theorem correct_answer :
  ∃ s1 s2 : Finset Char,
  s1 ∈ coin_events ∧
  (at_most_one_heads s1 ∧ exactly_two_heads s2) ∧
  (s1 ∩ s2 = ∅) ∧ ¬(s1 ∪ s2 = coin_events) :=
sorry

end correct_answer_l514_514847


namespace probability_one_of_each_color_l514_514465

theorem probability_one_of_each_color :
  let total_marbles := 9 in
  let choose_three_total := Nat.choose total_marbles 3 in
  let red := 3 in
  let blue := 3 in
  let green := 3 in
  let favorable_outcomes := red * blue * green in
  (favorable_outcomes: ℚ) / choose_three_total = 9 / 28 :=
by
  sorry

end probability_one_of_each_color_l514_514465


namespace arithmetic_sequence_general_term_sum_of_sequence_b_l514_514285

-- Definitions of the sequences and conditions
def a (n : ℕ) : ℝ := n + 1
def b (n : ℕ) : ℝ := 1 / (a n * a (n + 1))
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, b i

-- Proposition stating the first part
theorem arithmetic_sequence_general_term :
  ∀ n : ℕ, a n = n + 1 := 
by {
  intros,
  sorry
}

-- Proposition stating the second part
theorem sum_of_sequence_b :
  ∀ n : ℕ, S n = (n : ℝ) / (2 * (n + 2)) := 
by {
  intros,
  sorry
}

end arithmetic_sequence_general_term_sum_of_sequence_b_l514_514285


namespace isosceles_right_triangle_C_coordinates_l514_514709

theorem isosceles_right_triangle_C_coordinates :
  ∃ C : ℝ × ℝ, (let A : ℝ × ℝ := (1, 0)
                let B : ℝ × ℝ := (3, 1) 
                ∃ (x y: ℝ), C = (x, y) ∧ 
                ((x-1)^2 + y^2 = 10) ∧ 
                (((x-3)^2 + (y-1)^2 = 10))) ∨
                ((x = 2 ∧ y = 3) ∨ (x = 4 ∧ y = -1)) :=
by
  sorry

end isosceles_right_triangle_C_coordinates_l514_514709


namespace smallest_n_for_g4_l514_514750

def g (n : ℕ) : ℕ :=
  ((sum (λ a, if a > 0 ∧ (∃ b > 0, a^2 + b^2 = n) then 1 else 0)) -
  ((sum (λ ⟨a, b⟩, if a > 0 ∧ b > 0 ∧ a^2 + b^2 = n then 1 else 0)) / 2)) + 1

theorem smallest_n_for_g4 :
  (∃ n : ℕ, n > 0 ∧ g(n) = 4 ∧ ∀ m : ℕ, m > 0 ∧ m < n → g(m) ≠ 4) ∧
  (∀ n : ℕ, n = 65 → g(n) = 4) :=
begin
  sorry
end

end smallest_n_for_g4_l514_514750


namespace sum_a_b_rational_l514_514633

theorem sum_a_b_rational (a b : ℚ) (h : 5 - real.sqrt 3 * a = 2 * b + real.sqrt 3 - a) : a + b = 1 :=
sorry

end sum_a_b_rational_l514_514633


namespace monotonicity_and_range_l514_514662

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) * (Real.exp x - a) - a^2 * x

theorem monotonicity_and_range (a : ℝ) :
  (∀ x : ℝ, a = 0 → 0 < f(x, a)) ∧
  (∀ x : ℝ, a > 0 → (x < Real.log a → f(x, a) < 0) ∧ (x > Real.log a → f(x, a) > 0)) ∧
  (∀ x : ℝ, a < 0 → (x < Real.log (-a/2) → f(x, a) < 0) ∧ (x > Real.log (-a/2) → f(x, a) > 0)) ∧
  (∀ x : ℝ, 0 < a → a ≤ 1 ∧ f(x, a) ≥ 0) ∧
  (∀ x : ℝ, -2 * Real.exp (3 / 4) ≤ a ∧ a < 0 → f(x, a) ≥ 0)
:= sorry

end monotonicity_and_range_l514_514662


namespace leader_boy_probability_l514_514504

theorem leader_boy_probability
  (group : Finset ℕ)
  (h_group_size : group.card = 5)
  (girls : Finset ℕ)
  (boys : Finset ℕ)
  (h_girls_count : girls.card = 3)
  (h_boys_count : boys.card = 2)
  (h_partition : girls ∪ boys = group):
  (1 / (group.card.choose 2).to_rat) * (boys.card * (group.card - 1) / ((group.card - 1).choose 1).to_rat) = 2 / 5 :=
by sorry

end leader_boy_probability_l514_514504


namespace perpendicular_segments_l514_514496

variables {O A B C D E F : Type*}
variables [metric_space O] [circle AB O] [point D] [tangent CB B O]

-- Definitions of the geometric entities in the problem
def circle_diameter (AB : Type*) (O : Type*) : Prop := ∃ (O : Type*), ∃ (AB : Type*), O ∉ AB ∧ arc O AB
def tangent_line (CB : Type*) (B : Type*) (O : Type*) : Prop := ∃ (B : Type*) (tangent : Type*), B ∉ tangent ∧ tangent O B

-- Main theorem statement for the proof problem
theorem perpendicular_segments
  (ABd : circle_diameter AB O)
  (CBt : tangent_line CB B O)
  (CdF : intersects_point CD F O)
  (AD : intersects_point AD D O)
  (OC_E : intersects_point AD E O ∧ intersects_point OC E O):
  ∃ (EB FB : Type*), perpendicular EB FB :=
by
  sorry

end perpendicular_segments_l514_514496


namespace solve_equation_l514_514009

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) : x = 5 / 3 := 
by sorry

end solve_equation_l514_514009


namespace proof_value_of_expression_l514_514676

theorem proof_value_of_expression (a b c d m : ℝ) 
  (h1: a + b = 0)
  (h2: c * d = 1)
  (h3: |m| = 4) : 
  m + c * d + (a + b) / m = 5 ∨ m + c * d + (a + b) / m = -3 := by
  sorry

end proof_value_of_expression_l514_514676


namespace sum_of_digits_of_N_l514_514523

theorem sum_of_digits_of_N :
  (N : ℕ) (h : N * (N + 1) / 2 = 5050) → (N.digits.sum = 1) :=
by
  sorry

end sum_of_digits_of_N_l514_514523


namespace polynomial_remainder_l514_514977

noncomputable def f (x : ℝ) : ℝ := x^4 + 2 * x^2 - 3
noncomputable def g (x : ℝ) : ℝ := x^2 + x - 2
noncomputable def r (x : ℝ) : ℝ := 5 * x^2 - 2 * x - 3

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = g x * q x + r x :=
sorry

end polynomial_remainder_l514_514977


namespace expression_evaluation_l514_514154

theorem expression_evaluation :
  10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 :=
by
  sorry

end expression_evaluation_l514_514154


namespace solve_ln_eq_l514_514409

noncomputable def solve_exp_eq : Set ℝ :=
  {x | Real.ln (x^2 - 5 * x + 9) = 2}

theorem solve_ln_eq :
  solve_exp_eq = {4.65, 0.35} :=
by
  sorry

end solve_ln_eq_l514_514409


namespace solve_trig_eq_l514_514410

theorem solve_trig_eq (x y z : ℝ) (n m k : ℤ) :
  x ≠ 0 ∧ y ≠ 0 →
  (sin x)^2 ≠ 0 ∧ (cos y)^2 ≠ 0 →
  ((sin x)^2 + 1 / (sin x)^2)^3 + ((cos y)^2 + 1 / (cos y)^2)^3 = 16 * (sin z)^2 →
  ∃ n m k : ℤ, x = π / 2 + π * n ∧ y = π * m ∧ z = π / 2 + π * k :=
begin
  intros h1 h2 h3,
  sorry
end

end solve_trig_eq_l514_514410


namespace total_bike_price_l514_514385

theorem total_bike_price 
  (marion_bike_cost : ℝ := 356)
  (stephanie_bike_base_cost : ℝ := 2 * marion_bike_cost)
  (stephanie_discount_rate : ℝ := 0.10)
  (patrick_bike_base_cost : ℝ := 3 * marion_bike_cost)
  (patrick_discount_rate : ℝ := 0.75)
  (stephanie_bike_cost : ℝ := stephanie_bike_base_cost * (1 - stephanie_discount_rate))
  (patrick_bike_cost : ℝ := patrick_bike_base_cost * patrick_discount_rate):
  marion_bike_cost + stephanie_bike_cost + patrick_bike_cost = 1797.80 := 
by 
  sorry

end total_bike_price_l514_514385


namespace ratio_of_PQ_QR_l514_514781

theorem ratio_of_PQ_QR (PQ QR RS PS RT QT PT TS : ℝ) 
  (h1 : ∠Q = 90°) (h2 : ∠R = 90°) (h3 : Δ PQR ∼ Δ QRS)
  (h4 : PQ > QR) (h5 : ∃ T, T ∈ interior PQRS ∧ Δ PQR ∼ Δ RTQ) 
  (h6 : area (triangle PTS) = 14 * area (triangle RTQ)) :
  PQ / QR = 2 + sqrt 3 :=
sorry

end ratio_of_PQ_QR_l514_514781


namespace count_true_propositions_l514_514189

def proposition1 : Prop := ∀ x : ℝ, even (λ x, cos x)
def proposition2 : Prop := ∀ x y : ℝ, (x = y) → (x^2 = y^2) → (x ≠ y) → (x^2 ≠ y^2)
def proposition3 : Prop := ∀ x : ℝ, (x ≥ 2) ↔ (x^2 - x - 2 ≥ 0)
def proposition4 : Prop := (∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0)

theorem count_true_propositions : (proposition1 ∨ proposition2 ∨ proposition4) ∧ ¬proposition3 :=
  by sorry

end count_true_propositions_l514_514189


namespace range_of_a_l514_514646

noncomputable def parabola_above_line (a : ℝ) : Prop :=
  ∀ x ∈ set.Icc a (a + 1), x^2 - a * x + 3 > 9 / 4

theorem range_of_a (a : ℝ) : parabola_above_line a → a ∈ set.Ioo (-real.sqrt 3) (real.top) := by
  sorry

end range_of_a_l514_514646


namespace Q_root_l514_514209

def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem Q_root : Q (3^(1 / 3 : ℝ) + 2) = 0 := sorry

end Q_root_l514_514209


namespace equilateral_triangle_ratio_l514_514867

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514867


namespace find_f_zero_l514_514642

-- Define the function f and the conditions from the problem
def f (x : ℝ) := Real.cos (ω * x + φ)
def is_adj_extreme_points (ω : ℝ) (φ : ℝ) : Prop :=
  x = 1 ∧ x = 5 ∧ ∀ ω > 0, f (5) - f (1) = 4

-- Define the derivative condition
def derivative_condition (f' : ℝ → ℝ) : Prop :=
  f' 2 < 0

-- The final problem to be proven.
theorem find_f_zero (ω φ : ℝ) (f' : ℝ → ℝ) (h_extreme : is_adj_extreme_points ω φ) (h_deriv : derivative_condition f') : f 0 = Real.sqrt 2 / 2 :=
by
  sorry

end find_f_zero_l514_514642


namespace calculate_expression_l514_514160

theorem calculate_expression : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  calc
    10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2
    _ = 10 - 9 + 56 + 6 - 20 + 3 - 2 : by rw [mul_comm 8 7, mul_comm 5 4] -- Perform multiplications
    _ = 1 + 56 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 10 - 9
    _ = 57 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 1 + 56
    _ = 63 - 20 + 3 - 2 : by norm_num  -- Simplify 57 + 6
    _ = 43 + 3 - 2 : by norm_num -- Simplify 63 - 20
    _ = 46 - 2 : by norm_num -- Simplify 43 + 3
    _ = 44 : by norm_num -- Simplify 46 - 2

end calculate_expression_l514_514160


namespace difference_in_length_l514_514566

-- Defining the lengths of the white line and the blue line as constants
constant white_line : ℝ := 7.666666666666667
constant blue_line : ℝ := 3.3333333333333335

-- Defining the proof statement
theorem difference_in_length : (white_line - blue_line = 4.333333333333333) :=
by
  sorry

end difference_in_length_l514_514566


namespace find_smaller_number_l514_514044

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end find_smaller_number_l514_514044


namespace equilateral_triangle_ratio_l514_514960

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514960


namespace real_rate_of_return_is_10_percent_l514_514393

-- Given definitions based on conditions
def nominal_rate := 0.21
def inflation_rate := 0.10

-- Statement to prove
theorem real_rate_of_return_is_10_percent (r : ℝ) :
  1 + r = (1 + nominal_rate) / (1 + inflation_rate) → r = 0.10 := 
by
  sorry

end real_rate_of_return_is_10_percent_l514_514393


namespace solve_for_x_l514_514973

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 ↔ x = 2 / 9 := by
  sorry

end solve_for_x_l514_514973


namespace cannot_construct_110_l514_514770

def valid_angle (angle : ℕ) : Bool :=
  angle == 90 ∨ angle == 45 ∨ angle == 30 ∨ angle == 60

def can_construct (angle : ℕ) : Prop :=
  ∃ (a b : ℕ), (valid_angle a) ∧ (valid_angle b) ∧ (angle = a + b ∨ angle = abs (a - b))

theorem cannot_construct_110 :
  ¬ can_construct 110 := by
  sorry

end cannot_construct_110_l514_514770


namespace equilateral_triangle_ratio_l514_514877

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514877


namespace clock_hands_right_angles_l514_514490

theorem clock_hands_right_angles (h12 : ℕ) (h5d : ℕ) :
  (h12 = 22) → (h5d = h12 * 2 * 5) → h5d = 220 := 
by
  intros h12_eq h5d_eq
  rw [h12_eq, h5d_eq]
  norm_num
  sorry

end clock_hands_right_angles_l514_514490


namespace integer_solutions_of_inequalities_l514_514580

theorem integer_solutions_of_inequalities (x : ℤ) :
  ( (x + 8) / (x + 2) > 2 ) ∧ (log (x - 1) < 1) ↔ (x = 2 ∨ x = 3) := 
sorry

end integer_solutions_of_inequalities_l514_514580


namespace infinite_same_prime_factors_in_arith_prog_l514_514757

theorem infinite_same_prime_factors_in_arith_prog (a d : ℕ) :
  ∃ S ⊆ { n : ℕ | ∃ k : ℕ, a + k * d = n }, Infinite S ∧ ∀ n ∈ S, (∃ m ∈ primes, m ∣ n) :=
sorry

end infinite_same_prime_factors_in_arith_prog_l514_514757


namespace find_x_l514_514691

variables (a b c d x : ℤ)

theorem find_x (h1 : a - b = c + d + 9) (h2 : a - c = 3) (h3 : a + b = c - d - x) : x = 3 :=
sorry

end find_x_l514_514691


namespace number_of_squares_with_center_60_45_l514_514391

-- Definitions for the conditions of the problem:
def is_axis_aligned_square (x y : ℕ) (center : ℕ × ℕ) : Prop :=
  let (cx, cy) := center in
  (x = cx ∨ y = cy)

def is_tilted_square (x y : ℕ) (center : ℕ × ℕ) : Prop :=
  let (cx, cy) := center in
  (15 ≤ x ∧ x ≤ 59) ∧ (0 ≤ y ∧ y ≤ 44)

def is_square (x y : ℕ) (center : ℕ × ℕ) : Prop :=
  is_axis_aligned_square x y center ∨ is_tilted_square x y center

-- The main statement:
theorem number_of_squares_with_center_60_45 :
  let center := (60, 45) in
  let n := 2070 in
  ∃ (squares : ℕ), (squares = n) ∧ 
  (∀x y, is_square x y center → ∃ (count : ℕ), count = n) :=
sorry

end number_of_squares_with_center_60_45_l514_514391


namespace number_of_arrangements_l514_514499

open Finset

theorem number_of_arrangements : 
  let students := {1, 2, 3, 4, 5, 6} in
  let A := 1 in
  let B := 2 in
  ∑ p in (students.permutes.filter (λ p, p.head ≠ A ∧ p.last ≠ B)), 1 = 5 :=
sorry

end number_of_arrangements_l514_514499


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514886

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514886


namespace time_planeA_took_off_l514_514065

noncomputable def plane_flight := ℕ

-- Define plane A speed
def planeA_speed : ℕ := 200

-- Define plane B speed
def planeB_speed : ℕ := 300

-- Define time taken by plane B to overtake plane A in minutes
def planeB_overtake_time : ℕ := 80

-- Define the time in minutes plane A took off before plane B
def takeoff_time_before (T : ℕ) : Prop := 200 * (T / 60 + 4 / 3) = 400

theorem time_planeA_took_off : ∃ T : plane_flight, takeoff_time_before T ∧ T = 40 :=
by
  sorry

end time_planeA_took_off_l514_514065


namespace min_value_in_interval_l514_514821

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x - 2

theorem min_value_in_interval : 
  ∃ x ∈ set.Icc 1 4, ∀ y ∈ set.Icc 1 4, f x ≤ f y ∧ f x = -2 := 
sorry

end min_value_in_interval_l514_514821


namespace cost_of_one_deck_of_basketball_cards_l514_514729

theorem cost_of_one_deck_of_basketball_cards :
  let mary_euro_total_after_discount := (2 * 50 + 100) * 0.90;
  let jack_pound_total_after_tax := (80 + 40 + 65) * 1.08;
  let mary_spent_in_usd := mary_euro_total_after_discount * 1.2;
  let jack_spent_in_usd := jack_pound_total_after_tax * 1.3;
  let rose_spent_total_in_usd := jack_spent_in_usd;
  let rose_shoes_total_after_discount := 150 * 0.95;
  let rose_shoes_total_after_tax := rose_shoes_total_after_discount * 1.07;
  let total_spent_on_cards := rose_spent_total_in_usd - rose_shoes_total_after_tax;
  let cost_per_deck := total_spent_on_cards / 3;
  cost_per_deck ≈ 35.76 := 
begin
  sorry
end

end cost_of_one_deck_of_basketball_cards_l514_514729


namespace camel_height_in_feet_correct_l514_514329

def hare_height_in_inches : ℕ := 14
def multiplication_factor : ℕ := 24
def inches_to_feet_ratio : ℕ := 12

theorem camel_height_in_feet_correct :
  (hare_height_in_inches * multiplication_factor) / inches_to_feet_ratio = 28 := by
  sorry

end camel_height_in_feet_correct_l514_514329


namespace floor_plus_r_eq_10_3_implies_r_eq_5_3_l514_514218

noncomputable def floor (x : ℝ) : ℤ := sorry -- Assuming the function exists

theorem floor_plus_r_eq_10_3_implies_r_eq_5_3 (r : ℝ) 
  (h : floor r + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_r_eq_10_3_implies_r_eq_5_3_l514_514218


namespace games_did_not_work_l514_514199

theorem games_did_not_work 
  (games_from_friend : ℕ) 
  (games_from_garage_sale : ℕ) 
  (good_games : ℕ) 
  (total_games : ℕ := games_from_friend + games_from_garage_sale) 
  (did_not_work : ℕ := total_games - good_games) :
  games_from_friend = 41 ∧ 
  games_from_garage_sale = 14 ∧ 
  good_games = 24 → 
  did_not_work = 31 := 
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end games_did_not_work_l514_514199


namespace part_a_parallelogram_part_b_no_parallelogram_l514_514585

-- Part (a)
theorem part_a_parallelogram (n : ℕ) (h : n > 1) : 
  ∀ (pieces : Finset (Fin n × Fin n)), pieces.card = 2 * n → 
  ∃ (a b c d : Fin n × Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ is_parallelogram a b c d :=
sorry

-- Definition: What it means for four points to form a parallelogram
def is_parallelogram {n : ℕ} (a b c d : Fin n × Fin n) : Prop :=
  (a.1 = b.1 ∧ c.1 = d.1 ∧ a.2 = c.2 ∧ b.2 = d.2 ∧ (b.2 - a.2) = (d.2 - c.2)) ∨
  (a.1 = d.1 ∧ b.1 = c.1 ∧ a.2 = b.2 ∧ c.2 = d.2 ∧ (c.2 - b.2) = (d.2 - a.2))

-- Part (b)
theorem part_b_no_parallelogram (n : ℕ) (h : n > 1) : 
  ∃ (pieces : Finset (Fin n × Fin n)), pieces.card = 2 * n - 1 ∧ 
  (∀ (a b c d : Fin n × Fin n), a ≠ b → b ≠ c → c ≠ d → d ≠ a → ¬ is_parallelogram a b c d) :=
sorry

end part_a_parallelogram_part_b_no_parallelogram_l514_514585


namespace equilateral_triangle_ratio_l514_514959

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514959


namespace blake_spending_on_oranges_l514_514148

theorem blake_spending_on_oranges (spending_on_oranges spending_on_apples spending_on_mangoes initial_amount change_amount: ℝ)
  (h1 : spending_on_apples = 50)
  (h2 : spending_on_mangoes = 60)
  (h3 : initial_amount = 300)
  (h4 : change_amount = 150)
  (h5 : initial_amount - change_amount = spending_on_oranges + spending_on_apples + spending_on_mangoes) :
  spending_on_oranges = 40 := by
  sorry

end blake_spending_on_oranges_l514_514148


namespace square_of_chord_length_is_39804_l514_514180

noncomputable def square_of_chord_length (r4 r8 r12 : ℝ) (externally_tangent : (r4 + r8) < r12) : ℝ := 
  let r4 := 4
  let r8 := 8
  let r12 := 12
  let PQ_sq := 4 * ((r12^2) - ((2 * r8 + 1 * r4) / 3)^2) in
  PQ_sq

theorem square_of_chord_length_is_39804 : 
  square_of_chord_length 4 8 12 ((4 + 8) < 12) = 398.04 := 
by
  sorry

end square_of_chord_length_is_39804_l514_514180


namespace number_of_valid_polygons_l514_514244

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

def is_integral_angle (n : ℕ) : Prop := (360 / n).den = 1

def is_valid_polygon (n : ℕ) : Prop := n > 2 ∧ is_integral_angle n

theorem number_of_valid_polygons : 
  (finset.filter is_valid_polygon (finset.range 361)).card = 22 :=
by 
  sorry

end number_of_valid_polygons_l514_514244


namespace minimum_chosen_squares_sum_l514_514771

-- We have an 8x8 chessboard
def label (i j : ℕ) : ℚ := 1 / (i + j - 1)

-- The function gives the sum of chosen squares' labels
def chosen_squares_sum (f : Fin 8 → Fin 8) : ℚ :=
  ∑ i, label (f i).val i.val

-- We need to prove the minimum sum of the labels
theorem minimum_chosen_squares_sum : ∃ (f : Fin 8 → Fin 8), (∀ i j, i ≠ j → f i ≠ f j) ∧ 
  chosen_squares_sum f = 1 :=
by
  use (λ i, ⟨8 - i.val, by linarith⟩)
  split
  -- proof of injectiveness of the function which indicates one square in each row and column
  {
    intros i j h
    simp only [Function.Injective]
    intro h₁
    exact h (Fin.val_injective h₁)
  }
  -- proof that the given sum is equal to 1
  {
    simp [chosen_squares_sum, label]
    sorry
  }

end minimum_chosen_squares_sum_l514_514771


namespace geom_seq_product_equals_16_l514_514708

theorem geom_seq_product_equals_16
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : ∀ m n, a (m + 1) - a m = a (n + 1) - a n)
  (non_zero_diff : ∃ d, d ≠ 0 ∧ ∀ n, a (n + 1) - a n = d)
  (h_cond : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0)
  (h_geom : ∀ m n, b (m + 1) / b m = b (n + 1) / b n)
  (h_b7 : b 7 = a 7):
  b 6 * b 8 = 16 := 
sorry

end geom_seq_product_equals_16_l514_514708


namespace total_bars_is_7_l514_514559

variable (x : ℕ)

-- Each chocolate bar costs $3
def cost_per_bar := 3

-- Olivia sold all but 4 bars
def bars_sold (total_bars : ℕ) := total_bars - 4

-- Olivia made $9
def amount_made (total_bars : ℕ) := cost_per_bar * bars_sold total_bars

-- Given conditions
def condition1 (total_bars : ℕ) := amount_made total_bars = 9

-- Proof that the total number of bars is 7
theorem total_bars_is_7 : condition1 x -> x = 7 := by
  sorry

end total_bars_is_7_l514_514559


namespace repeated_digit_percentage_l514_514686

noncomputable def percentage_repeated_digits : ℝ :=
  let total_numbers := 90000
  let non_repeated_digits := 9 * 9 * 8 * 7 * 6
  let repeated_digits := total_numbers - non_repeated_digits
  ((repeated_digits.toReal / total_numbers.toReal) * 100).round / 10

theorem repeated_digit_percentage (y : ℝ) : 
  percentage_repeated_digits = y → y = 69.8 :=
by
  intro h
  have : percentage_repeated_digits = 69.8 := sorry
  rw this at h
  exact h

end repeated_digit_percentage_l514_514686


namespace inradius_of_triangle_l514_514130

theorem inradius_of_triangle (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h_triangle : (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2)) :
  let s := (a + b + c) / 2 in
  let A := real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  let r := A / s in
  r = 4 :=
by
  sorry

end inradius_of_triangle_l514_514130


namespace carson_can_ride_giant_slide_exactly_twice_l514_514542

noncomputable def Carson_Carnival : Prop := 
  let total_time_available := 240
  let roller_coaster_time := 30
  let tilt_a_whirl_time := 60
  let giant_slide_time := 15
  let vortex_time := 45
  let bumper_cars_time := 25
  let roller_coaster_rides := 4
  let tilt_a_whirl_rides := 2
  let vortex_rides := 1
  let bumper_cars_rides := 3

  let total_time_spent := 
    roller_coaster_time * roller_coaster_rides +
    tilt_a_whirl_time * tilt_a_whirl_rides +
    vortex_time * vortex_rides +
    bumper_cars_time * bumper_cars_rides

  total_time_available - (total_time_spent + giant_slide_time * 2) = 0

theorem carson_can_ride_giant_slide_exactly_twice : Carson_Carnival :=
by
  unfold Carson_Carnival
  sorry -- proof will be provided here

end carson_can_ride_giant_slide_exactly_twice_l514_514542


namespace alex_silver_tokens_l514_514135

theorem alex_silver_tokens :
  ∃ x y : ℕ, 
    (100 - 3 * x + y ≤ 2) ∧ 
    (50 + 2 * x - 4 * y ≤ 3) ∧
    (x + y = 74) :=
by
  sorry

end alex_silver_tokens_l514_514135


namespace hire_charges_paid_by_b_l514_514985

theorem hire_charges_paid_by_b (total_cost : ℕ) (hours_a : ℕ) (hours_b : ℕ) (hours_c : ℕ) 
  (total_hours : ℕ) (cost_per_hour : ℕ) : 
  total_cost = 520 → hours_a = 7 → hours_b = 8 → hours_c = 11 → total_hours = hours_a + hours_b + hours_c 
  → cost_per_hour = total_cost / total_hours → 
  (hours_b * cost_per_hour) = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end hire_charges_paid_by_b_l514_514985


namespace money_remaining_l514_514383

variable (initial_amount : ℝ) (spent_book : ℝ) (spent_sweets : ℝ)
variable (given_andy : ℝ) (given_sam : ℝ) (given_kate : ℝ) (given_nick : ℝ)

def remaining_money (initial_amount : ℝ) (spent_book : ℝ) (spent_sweets : ℝ)
    (given_andy : ℝ) (given_sam : ℝ) (given_kate : ℝ) (given_nick : ℝ) : ℝ :=
  let total_spent = spent_book + spent_sweets
  let remaining_after_spend = initial_amount - total_spent
  let total_given = given_andy + given_sam + given_kate + given_nick
  let remaining_after_give = remaining_after_spend - total_given
  let spent_on_game = (remaining_after_give / 2).round
  let final_remaining = remaining_after_give - spent_on_game
  final_remaining

theorem money_remaining (h_initial : initial_amount = 105.65)
                        (h_spent_book : spent_book = 15.24)
                        (h_spent_sweets : spent_sweets = 7.65)
                        (h_given_andy : given_andy = 6.15)
                        (h_given_sam : given_sam = 10.75)
                        (h_given_kate : given_kate = 9.40)
                        (h_given_nick : given_nick = 20.85) :
  remaining_money initial_amount spent_book spent_sweets given_andy given_sam given_kate given_nick = 17.80 := 
by
  sorry

end money_remaining_l514_514383


namespace ratio_eq_sqrt3_div_2_l514_514947

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514947


namespace system_exactly_three_solutions_l514_514570

theorem system_exactly_three_solutions (a : ℝ) :
  (∃ x y : ℝ, ((|y - 4| + |x + 12| - 3) * (x^2 + y^2 - 12) = 0) ∧ ((x + 5)^2 + (y - 4)^2 = a)) ↔ 
  (a = 16 ∨ a = 53 + 4 * sqrt 123) := 
sorry

end system_exactly_three_solutions_l514_514570


namespace a3_eq_5_l514_514629

variable {a_n : ℕ → Real} (S : ℕ → Real)
variable (a1 d : Real)

-- Define arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → Real) (a1 d : Real) : Prop :=
  ∀ n : ℕ, n > 0 → a_n n = a1 + (n - 1) * d

-- Define sum of first n terms
def sum_of_arithmetic (S : ℕ → Real) (a_n : ℕ → Real) : Prop :=
  ∀ n : ℕ, S n = n / 2 * (a_n 1 + a_n n)

-- Given conditions: S_5 = 25
def S_5_eq_25 (S : ℕ → Real) : Prop :=
  S 5 = 25

-- Goal: prove a_3 = 5
theorem a3_eq_5 (h_arith : is_arithmetic_sequence a_n a1 d)
                (h_sum : sum_of_arithmetic S a_n)
                (h_S5 : S_5_eq_25 S) : a_n 3 = 5 :=
  sorry

end a3_eq_5_l514_514629


namespace percentage_repeated_digits_five_digit_numbers_l514_514684

theorem percentage_repeated_digits_five_digit_numbers : 
  let total_five_digit_numbers := 90000
  let non_repeated_digits_number := 9 * 9 * 8 * 7 * 6
  let repeated_digits_number := total_five_digit_numbers - non_repeated_digits_number
  let y := (repeated_digits_number.toFloat / total_five_digit_numbers.toFloat) * 100 
  y = 69.8 :=
by
  sorry

end percentage_repeated_digits_five_digit_numbers_l514_514684


namespace min_value_x_plus_4_over_x_l514_514238

noncomputable def minValueOfExpression (x : ℝ) (hx : 0 < x) : ℝ := x + 4/x

theorem min_value_x_plus_4_over_x : ∀ (x : ℝ), 0 < x → (minValueOfExpression x) ≥ 4 :=
begin
  intros x hx,
  -- Here the actual proof would follow, the inequality (x + 4/x) ≥ 4 and attainability
  sorry
end

end min_value_x_plus_4_over_x_l514_514238


namespace find_parallelepiped_dimensions_l514_514573

theorem find_parallelepiped_dimensions :
  ∃ (x y z : ℕ),
    (x * y * z = 2 * (x * y + y * z + z * x)) ∧
    (x = 6 ∧ y = 6 ∧ z = 6 ∨
     x = 5 ∧ y = 5 ∧ z = 10 ∨
     x = 4 ∧ y = 8 ∧ z = 8 ∨
     x = 3 ∧ y = 12 ∧ z = 12 ∨
     x = 3 ∧ y = 7 ∧ z = 42 ∨
     x = 3 ∧ y = 8 ∧ z = 24 ∨
     x = 3 ∧ y = 9 ∧ z = 18 ∨
     x = 3 ∧ y = 10 ∧ z = 15 ∨
     x = 4 ∧ y = 5 ∧ z = 20 ∨
     x = 4 ∧ y = 6 ∧ z = 12) :=
by
  sorry

end find_parallelepiped_dimensions_l514_514573


namespace solve_trig_eq_l514_514411

theorem solve_trig_eq (x y z : ℝ) (n m k : ℤ) :
  x ≠ 0 ∧ y ≠ 0 →
  (sin x)^2 ≠ 0 ∧ (cos y)^2 ≠ 0 →
  ((sin x)^2 + 1 / (sin x)^2)^3 + ((cos y)^2 + 1 / (cos y)^2)^3 = 16 * (sin z)^2 →
  ∃ n m k : ℤ, x = π / 2 + π * n ∧ y = π * m ∧ z = π / 2 + π * k :=
begin
  intros h1 h2 h3,
  sorry
end

end solve_trig_eq_l514_514411


namespace no_six_when_mean_median_two_l514_514245

theorem no_six_when_mean_median_two (l : List ℕ) (h_len : l.length = 5) (h_mean : l.sum / l.length = 2) (h_median : l.nth_le 2 (by simp [h_len]) = 2) :
  6 ∉ l := sorry

end no_six_when_mean_median_two_l514_514245


namespace solve_for_A_l514_514790

noncomputable def polynomial_denominator (x : ℝ) : ℝ :=
  x^4 - 2 * x^3 - 29 * x^2 + 70 * x + 120

noncomputable def polynomial: (x: ℝ) -> ℝ :=
  (x + 4) * (x - 3) * (x - 2)^2

theorem solve_for_A (A B C D : ℝ) :
  (∀ x, 1 / polynomial_denominator x = A / (x + 4) + B / (x - 2) + C / (x - 2)^2 + D / (x - 3))
  → A = -1 / 252 := by
  sorry

end solve_for_A_l514_514790


namespace coin_flip_probability_l514_514491

-- Define a discrete probability space for a fair coin
def coin_flip := {0, 1}  -- 0 for tails, 1 for heads

-- Probability that a fair coin lands heads
def prob_heads : ℚ := 1 / 2

-- Probability that a fair coin lands tails
def prob_tails : ℚ := 1 / 2

-- Define the event of interest: heads on first flip and tails on last four flips
def event_of_interest (flips : List ℕ) : Prop :=
  flips.length = 5 ∧
  flips.head = 1 ∧
  flips.tail = [0, 0, 0, 0]

theorem coin_flip_probability :
  ∀ flips : List ℕ,
  (∀ flip ∈ flips, flip ∈ coin_flip) →
  event_of_interest flips →
  (prob_heads * prob_tails ^ 4) = (1 / 32) :=
by
  sorry

end coin_flip_probability_l514_514491


namespace product_of_positive_integer_values_l514_514190

-- Conditions
def quadratic_eq (a b c : ℝ) := λ x : ℝ, a * x^2 + b * x + c

def discriminant_pos (a b c : ℝ) := b^2 - 4 * a * c > 0

-- Main theorem
theorem product_of_positive_integer_values (c : ℕ) :
  (∀ c < 16, discriminant_pos 10 25 (c : ℝ)) →
  (∏ i in Finset.range 16, if i > 0 then i else 1 = 1307674368000) :=
by
  intros h
  sorry

end product_of_positive_integer_values_l514_514190


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514903

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514903


namespace mass_percentage_O_in_N2O_is_approximately_36_35_l514_514575

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def number_of_N : ℕ := 2
noncomputable def number_of_O : ℕ := 1

noncomputable def molar_mass_N2O : ℝ := (number_of_N * atomic_mass_N) + (number_of_O * atomic_mass_O)

noncomputable def mass_percentage_O : ℝ := (atomic_mass_O / molar_mass_N2O) * 100

theorem mass_percentage_O_in_N2O_is_approximately_36_35 :
  abs (mass_percentage_O - 36.35) < 0.01 := sorry

end mass_percentage_O_in_N2O_is_approximately_36_35_l514_514575


namespace not_all_odd_square_sum_eq_square_l514_514143

theorem not_all_odd_square_sum_eq_square (a1 a2 a3 a4 a5 b : ℤ) 
  (h : a1^2 + a2^2 + a3^2 + a4^2 + a5^2 = b^2) :
  ¬(odd a1 ∧ odd a2 ∧ odd a3 ∧ odd a4 ∧ odd a5 ∧ odd b) :=
sorry

end not_all_odd_square_sum_eq_square_l514_514143


namespace range_of_function_l514_514049

noncomputable def range_y : Set ℝ :=
  {y : ℝ | ∃ x : ℝ, ∃ t : ℝ, t = sin x ∧ t ∈ Icc (-1 : ℝ) 1 ∧ y = t^2 + t - 1}

theorem range_of_function :
  range_y = Icc (-5 / 4 : ℝ) 1 :=
by
  sorry

end range_of_function_l514_514049


namespace cement_mixture_weight_l514_514488

variable {W : ℕ}

-- Conditions
def is_sand (W : ℕ) : Prop := W / 5
def is_water (W : ℕ) : Prop := 3 * W / 4
def is_gravel (W : ℕ) : Prop := W - W / 5 - 3 * W / 4 = 6

-- Theorem to prove
theorem cement_mixture_weight (h1 : is_sand W) (h2 : is_water W) (h3 : is_gravel W) : W = 120 :=
by sorry

end cement_mixture_weight_l514_514488


namespace probability_two_months_sum_le_7_may_prediction_is_reliable_l514_514048

-- Definitions for water consumption
def water_consumption : List ℝ := [2.5, 3, 4, 4.5, 5.2]

-- Condition for reliable prediction
def reliable_prediction (pred actual : ℝ) : Prop := abs (pred - actual) ≤ 0.05

-- Function to calculate regression line parameters
def regression_line (xs ys : List ℝ) : ℝ × ℝ :=
  let n := xs.length
  let x_bar := (xs.sum / n)
  let y_bar := (ys.sum / n)
  let b := ((List.zipWith (*) xs ys).sum - n * x_bar * y_bar) / ((xs.map (λ x => x^2)).sum - n * x_bar^2)
  let a := y_bar - b * x_bar
  (a, b)

-- Function to predict value using regression line
def predict (a b x : ℝ) : ℝ := a + b * x

theorem probability_two_months_sum_le_7 :
  let favorable_pairs := List.filter (λ (xy : ℝ × ℝ) => xy.1 + xy.2 ≤ 7) [(2.5, 3), (2.5, 4), (2.5, 4.5), (3, 4), (3, 4.5), (4, 4.5), (4, 5.2), (4.5, 5.2)]
  let probability := (favorable_pairs.length : ℝ) / 10
  probability = 2 / 5 := by sorry

theorem may_prediction_is_reliable :
  let (4_months_water_consumption : List ℝ) := water_consumption.take 4
  let months : List ℝ := [1, 2, 3, 4]
  let (a, b) := regression_line months 4_months_water_consumption
  let y_may := predict a b 5
  reliable_prediction y_may 5.2 := by sorry

end probability_two_months_sum_le_7_may_prediction_is_reliable_l514_514048


namespace percentage_assigned_exam_l514_514703

-- Define the conditions of the problem
def total_students : ℕ := 100
def average_assigned : ℝ := 0.55
def average_makeup : ℝ := 0.95
def average_total : ℝ := 0.67

-- Define the proof problem statement
theorem percentage_assigned_exam :
  ∃ (x : ℝ), (x / total_students) * average_assigned + ((total_students - x) / total_students) * average_makeup = average_total ∧ x = 70 :=
by
  sorry

end percentage_assigned_exam_l514_514703


namespace square_of_chord_l514_514168

-- Definitions of the circles and tangency conditions
def radius1 := 4
def radius2 := 8
def radius3 := 12

-- The internal and external tangents condition
def externally_tangent (r1 r2 : ℕ) : Prop := ∃ O1 O2, dist O1 O2 = r1 + r2
def internally_tangent (r_in r_out : ℕ) : Prop := ∃ O_in O_out, dist O_in O_out = r_out - r_in

theorem square_of_chord :
  externally_tangent radius1 radius2 ∧ 
  internally_tangent radius1 radius3 ∧
  internally_tangent radius2 radius3 →
  (∃ PQ : ℚ, PQ^2 = 3584 / 9) :=
by
  intros h
  sorry

end square_of_chord_l514_514168


namespace binomial_sum_l514_514207

theorem binomial_sum (n : ℕ) (m : ℕ) (h : 3 * m + 1 ≤ n) :
  (Finset.range (m + 1)).sum (λ k, Nat.choose n (3 * k + 1)) = 
  1 / 3 * (2 ^ n + 2 * Real.cos ((n - 2) * Real.pi / 3)) :=
sorry

end binomial_sum_l514_514207


namespace triangle_area_perimeter_ratio_l514_514918

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514918


namespace sum_of_valid_a_values_l514_514320

theorem sum_of_valid_a_values :
  let valid_a_values := {a | (∀ x : ℝ, ((x + 1 ≤ (2 * x - 5) / 3) ∧ (a - x > 1) → x ≤ -8)) ∧
                              (∃ y : ℤ, y ≥ 0 ∧ (4 + (y / (y - 3:ℝ)) = (a - 1) / (3 - y:ℝ)))}
  in (∑ a in valid_a_values, a) = 24 :=
by
  sorry

end sum_of_valid_a_values_l514_514320


namespace solution_set_sgn_inequality_l514_514303

noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1 else if x < 0 then -1 else 0

theorem solution_set_sgn_inequality :
  {x : ℝ | (x + 1) * sgn x > 2} = {x : ℝ | x < -3} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_sgn_inequality_l514_514303


namespace correct_answer_l514_514669

noncomputable def prop1 (f : ℝ → ℝ) : Prop := ∀ x, f(x + 2) = f(-(x + 2))
noncomputable def prop2 (f : ℝ → ℝ) : Prop := ∀ x < 2, f' x < 0 ∧ ∀ x > 2, f' x > 0

def f1 (x : ℝ) : ℝ := abs (x + 2)
def f2 (x : ℝ) : ℝ := (x - 2) ^ 2
def f3 (x : ℝ) : ℝ := cos (x - 2)

theorem correct_answer : (prop1 f1 ∧ prop2 f1) = false ∧
                          (prop1 f2 ∧ prop2 f2) = true ∧
                          (prop1 f3 ∧ prop2 f3) = false := by
  sorry

end correct_answer_l514_514669


namespace find_triangle_sides_l514_514040

open Real

noncomputable def sqrt_ir (x : ℝ) : ℝ := Real.sqrt x  -- For providing the necessary sqrt function for real numbers

theorem find_triangle_sides
  (BD CE : ℝ)
  (angle_BMC : ℝ)
  (BM : BD = 6) 
  (CM : CE = 9) 
  (angle_BMC_eq : angle_BMC = 120) :
  let BC := sqrt_ir ((4)^2 + (6)^2 + 2 * 4 * 6 * (1 / 2)),
      AB := 2 * sqrt_ir 13,
      AC := 4 * sqrt_ir 7
  in
  BC = 2 * sqrt_ir 19 ∧ AB = 2 * sqrt_ir 13 ∧ AC = 4 * sqrt_ir 7 :=
by 
  sorry

end find_triangle_sides_l514_514040


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514950

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514950


namespace circumscribed_sphere_surface_area_l514_514631

theorem circumscribed_sphere_surface_area 
  (SA ABC: ℝ3)
  (A B C S: ℝ3)
  (h1: SA ∠ ABC = 90)
  (h2: AB ∠ AC = 90)
  (h3: SA = 3)
  (h4: AB = 2)
  (h5: AC = 2)
  : 4 * (2^2 + 2^2 + 3^2) * real.pi = 17 * real.pi :=
by
  sorry

end circumscribed_sphere_surface_area_l514_514631


namespace smallest_d_l514_514440

noncomputable def abc_identity_conditions (a b c d e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  ∀ x : ℝ, (x + a) * (x + b) * (x + c) = x^3 + 3 * d * x^2 + 3 * x + e^3

theorem smallest_d (a b c d e : ℝ) (h : abc_identity_conditions a b c d e) : d = 1 := 
sorry

end smallest_d_l514_514440


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514888

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514888


namespace sum_of_intercepts_eq_neg_half_l514_514831

theorem sum_of_intercepts_eq_neg_half :
  (let line_eq := λ x y : ℝ, x - 2 * y + 1 in
  let x_intercept := -1 in
  let y_intercept := 1 / 2 in
  x_intercept + y_intercept = -1 / 2) := by
  sorry

end sum_of_intercepts_eq_neg_half_l514_514831


namespace minimum_distance_PA_l514_514653

theorem minimum_distance_PA :
  let C := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 }
  ∃ P ∈ C, ∃ A, x y : ℝ, (x = (P.1)) → (y = 2 - (P.1)) → (min (abs (y - P.2)) = 2 - real.sqrt 2) 
  := sorry

end minimum_distance_PA_l514_514653


namespace pure_imaginary_value_b_l514_514650

noncomputable def Z1 : ℂ := 3 - 4 * complex.I
noncomputable def Z2 (b : ℝ) : ℂ := 4 + b * complex.I

theorem pure_imaginary_value_b (b : ℝ) (h : ∃ c : ℝ, Z1 * Z2 b = c * complex.I) :
  b = -3 :=
by
  sorry

end pure_imaginary_value_b_l514_514650


namespace sum_of_all_odd_digit_three_digit_numbers_l514_514581

/-- Sum of all three-digit numbers with all digits being odd is 69375. -/
theorem sum_of_all_odd_digit_three_digit_numbers :
  let digits := {1, 3, 5, 7, 9}
  ∑ x in {(a, b, c) | a ∈ digits ∧ b ∈ digits ∧ c ∈ digits}, (100 * a + 10 * b + c) = 69375 := by
  -- To be proven
  sorry

end sum_of_all_odd_digit_three_digit_numbers_l514_514581


namespace acute_triangle_tangent_sum_geq_3_sqrt_3_l514_514091

theorem acute_triangle_tangent_sum_geq_3_sqrt_3 {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) (h_sum : α + β + γ = π)
  (acute_α : α < π / 2) (acute_β : β < π / 2) (acute_γ : γ < π / 2) :
  Real.tan α + Real.tan β + Real.tan γ >= 3 * Real.sqrt 3 :=
sorry

end acute_triangle_tangent_sum_geq_3_sqrt_3_l514_514091


namespace polynomial_fraction_at_2_l514_514811

open Polynomial

noncomputable def p (k : ℝ) : Polynomial ℝ := k * (X - 4) * (X - 2)
def q : Polynomial ℝ := (X - 4) * (X + 3)

theorem polynomial_fraction_at_2 {k : ℝ} (hk : k = -4) (h4 : k * (2 - 4) * (2 - 2) = 0) :
  (p k).eval 2 / (q).eval 2 = 0 := 
by
  unfold p q
  rw [h4]
  rw [Polynomial.eval_mul, Polynomial.eval_sub]
  sorry

end polynomial_fraction_at_2_l514_514811


namespace scale_drawing_l514_514515

theorem scale_drawing (length_cm : ℝ) (representation : ℝ) : length_cm * representation = 3750 :=
by
  let length_cm := 7.5
  let representation := 500
  sorry

end scale_drawing_l514_514515


namespace three_digit_multiples_of_6_and_9_l514_514308

theorem three_digit_multiples_of_6_and_9 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ k, n = k * 18)}.finite.card = 50 :=
by
  sorry

end three_digit_multiples_of_6_and_9_l514_514308


namespace circle_chord_square_length_l514_514177

-- Define the problem in terms of Lean constructs
theorem circle_chord_square_length :
  ∃ (O₄ O₈ O₁₂ : ℝ) (A₄ A₈ A₁₂ : ℝ) (PQ : ℝ),
    O₄ = 4 ∧
    O₈ = 8 ∧
    O₁₂ = 12 ∧
    A₈ = 8 ∧
    A₄ = 4 ∧
    A₁₂ = (2 * A₈ + A₄) / 3 ∧
    PQ^2 = 4 * (O₁₂^2 - A₁₂^2) → 
    PQ^2 = 3584 / 9 :=
by
  -- Define variables based on conditions
  let O₄ := 4
  let O₈ := 8
  let O₁₂ := 12
  let A₈ := 8
  let A₄ := 4
  let A₁₂ := (2 * A₈ + A₄) / 3
  let s : ℝ := O₁₂^2 - A₁₂^2
  let PQ := real.sqrt(4 * s)

  -- Using given values to verify
  have h1 : O₄ = 4 := rfl
  have h2 : O₈ = 8 := rfl
  have h3 : O₁₂ = 12 := rfl
  have h4 : A₈ = 8 := rfl
  have h5 : A₄ = 4 := rfl
  have h6 : A₁₂ = (2 * A₈ + A₄) / 3 := rfl

  -- Calculate parts
  have h7: A₁₂ = 20 / 3 := rfl
  have h8: PQ^2 = 4 * (12^2 - (20 / 3)^2) := sorry
  have h9: PQ^2 = 4 * (144 - 400 / 9) := sorry
  have h10: PQ^2 = 4 * (1296 / 9 - 400 / 9) := sorry
  have h11: PQ^2 = 4 * (896 / 9) := sorry
  have h12: PQ^2 = 3584 / 9 := sorry

  -- Conclude the theorem
  exact ⟨O₄, O₈, O₁₂, A₄, A₈, A₁₂, PQ, h1, h2, h3, h4, h5, h6, (λ _, h12)⟩

end circle_chord_square_length_l514_514177


namespace average_weight_is_2992_l514_514780

open Real

theorem average_weight_is_2992 :
  let deviations := [+0.4, -0.2, -0.8, -0.4, 1, +0.3, +0.5, -2, +0.5, -0.1]
  let standard_weight := 30
  let n := 10
  let total_deviation := List.foldl (fun acc x => acc + x) 0 deviations
  let avg_deviation := total_deviation / n
  let avg_weight := standard_weight + avg_deviation
  avg_weight = 29.92 := by
  sorry

end average_weight_is_2992_l514_514780


namespace scientific_notation_of_number_l514_514839

theorem scientific_notation_of_number :
  (0.000000014 : ℝ) = 1.4 * 10 ^ (-8) :=
sorry

end scientific_notation_of_number_l514_514839


namespace repeated_digit_percentage_l514_514688

theorem repeated_digit_percentage (total : ℕ := 90000) (non_repeated_count : ℕ := 9 * 9 * 8 * 7 * 6) : 
  let repeated_count := total - non_repeated_count in
  let y := (repeated_count : ℚ) / total * 100 in
  y ≈ 69.8 :=
by
  sorry

end repeated_digit_percentage_l514_514688


namespace general_term_an_b_n_b_nplus2_lt_b_nplus1_sq_l514_514334

-- Define the arithmetic sequence a_n
noncomputable def a : ℕ → ℕ
| 0     := 0
| (n+1) := n+1

-- Condition: Sum of the first n terms of a equals half the product of a_n and a_(n+1)
def sum_arithmetic_sequence (n : ℕ) : ℕ :=
(n + 1) * n / 2

theorem general_term_an (n : ℕ) :
  (∀ (n : ℕ), n ≠ 0 → sum_arithmetic_sequence n = (a n * a (n+1)) / 2) → a n = n :=
sorry

-- Define the sequence b_n
noncomputable def b : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+1) := b n + 2^n

-- Prove that b_n * b_(n+2) < b_(n+1)^2 for all n
theorem b_n_b_nplus2_lt_b_nplus1_sq (n : ℕ) :
  b(n) * b(n+2) < (b(n+1))^2 :=
sorry

end general_term_an_b_n_b_nplus2_lt_b_nplus1_sq_l514_514334


namespace minimal_rotations_triangle_l514_514071

/-- Given a triangle with angles α, β, γ at vertices 1, 2, 3 respectively.
    The triangle returns to its original position after 15 rotations around vertex 1 by α,
    and after 6 rotations around vertex 2 by β.
    We need to show that the minimal positive integer n such that the triangle returns
    to its original position after n rotations around vertex 3 by γ is 5. -/
theorem minimal_rotations_triangle :
  ∃ (α β γ : ℝ) (k m l n : ℤ), 
    (15 * α = 360 * k) ∧ 
    (6 * β = 360 * m) ∧ 
    (α + β + γ = 180) ∧ 
    (n * γ = 360 * l) ∧ 
    (∀ n' : ℤ, n' > 0 → (∃ k' m' l' : ℤ, 
      (15 * α = 360 * k') ∧ 
      (6 * β = 360 * m') ∧ 
      (α + β + γ = 180) ∧ 
      (n' * γ = 360 * l') → n <= n')) ∧ 
    n = 5 := by
  sorry

end minimal_rotations_triangle_l514_514071


namespace lewis_earnings_during_harvest_l514_514382

-- Define the conditions
def regular_earnings_per_week : ℕ := 28
def overtime_earnings_per_week : ℕ := 939
def number_of_weeks : ℕ := 1091

-- Define the total earnings per week
def total_earnings_per_week := regular_earnings_per_week + overtime_earnings_per_week

-- Define the total earnings during the harvest season
def total_earnings_during_harvest := total_earnings_per_week * number_of_weeks

-- Theorem statement
theorem lewis_earnings_during_harvest : total_earnings_during_harvest = 1055497 := by
  sorry

end lewis_earnings_during_harvest_l514_514382


namespace tangent_line_circle_l514_514106

theorem tangent_line_circle :
  ∀ (b : ℝ),
  let line := (λ x y, 6 * x + 8 * y - b = 0)
  let circle := (λ x y, (x - 1) ^ 2 + (y - 1) ^ 2 = 1) in
  (∀ x y, circle x y → ∃ t, line x y ∧ ((6 * 1 + 8 * 1 - b).abs = 10 → (b = 4 ∨ b = 24))) 
:=
begin
  sorry
end

end tangent_line_circle_l514_514106


namespace probability_bd_greater_than_6_l514_514848

theorem probability_bd_greater_than_6 :
  ∀ {A B C P D : Type}
    (triangle_ABC : (ABC : Triangle) (right_triangle : ∠ A C B = π/2))
    (angle_ABC_eq_45 : ∠ A B C = π/4)
    (AB_eq_12 : AB = 12)
    (BP_meets_AC_at_D : extends (BP) (meets AC D))
    (P_random : random_in_triangle ABC),
    (probability (BD > 6)) = (√2 / 2) := 
by
  sorry

end probability_bd_greater_than_6_l514_514848


namespace solve_for_y_l514_514652

theorem solve_for_y (x y : ℝ) (h : 5 * x + 3 * y = 1) : y = (1 - 5 * x) / 3 :=
by
  sorry

end solve_for_y_l514_514652


namespace minimum_value_f_range_of_a_ln_inequality_l514_514295

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3

theorem minimum_value_f : ∃ x : ℝ, x > 0 ∧ f(x) = -1 / Real.exp 1 :=
sorry

theorem range_of_a (a : ℝ) : ∀ x : ℝ, x > 0 → 2 * f(x) ≥ g(x, a) ↔ a ≤ 4 :=
sorry

theorem ln_inequality : ∀ x : ℝ, x > 0 → Real.log x > 1 / (Real.exp 2) - 2 / (Real.exp 1 * x) :=
sorry

end minimum_value_f_range_of_a_ln_inequality_l514_514295


namespace sqrt_sq_eq_abs_l514_514979

theorem sqrt_sq_eq_abs (a : ℝ) : Real.sqrt (a^2) = |a| :=
sorry

end sqrt_sq_eq_abs_l514_514979


namespace complex_quadrant_l514_514194

theorem complex_quadrant (z : ℂ) (h : z = (↑(1/2) : ℂ) + (↑(1/2) : ℂ) * I ) : 
  0 < z.re ∧ 0 < z.im :=
by {
sorry -- Proof goes here
}

end complex_quadrant_l514_514194


namespace ratio_PQ_QR_l514_514725

-- Definitions of points and vectors in triangle XYZ
variables {X Y Z P Q R : Type} 
variables [add_comm_group X] [vector_space ℝ X]
variables [add_comm_group Y] [vector_space ℝ Y]
variables [add_comm_group Z] [vector_space ℝ Z]
variables [affine_space X Y] [affine_space Y Z] [affine_space X Z]

-- Conditions
variable (XP_PY_ratio : 4 * (P - X) = 1 * (Y - P))
variable (YQ_QZ_ratio : 4 * (Q - Y) = 1 * (Z - Q))
variable (P_line_XY : ∃ p, P = p • X + (1 - p) • Y)
variable (Q_line_YZ : ∃ q, Q = q • Y + (1 - q) • Z)
variable (R_on_line_XZ : ∃ r, R = r • X + (1 - r) • Z)

-- Conclusion to prove
theorem ratio_PQ_QR (XP_PY_ratio YQ_QZ_ratio P_line_XY Q_line_YZ R_on_line_XZ) :
  ∃ k : ℝ, PQ = k * QR ∧ k = 1 / 4 :=
sorry

end ratio_PQ_QR_l514_514725


namespace problem_statement_l514_514299

variables (P q : Prop)

-- Definitions of propositions P and q
def Prop_P := ∀ x y : ℝ, x > y → -x > -y
def Prop_q := ∀ x y : ℝ, x > y → x^2 > y^2

-- Statements to be proved
theorem problem_statement (hP : Prop_P = false) (hq : Prop_q = false) :
  (¬Prop_P ∨ ¬Prop_q) ∧ (¬Prop_P ∨ Prop_q) :=
by
  split;
  { sorry }

end problem_statement_l514_514299


namespace area_of_fenced_region_l514_514029

theorem area_of_fenced_region :
  let total_area := 20 * 18 in
  let square_cutout := 4 * 4 in
  let rect_cutout := 2 * 5 in
  let fenced_area := total_area - square_cutout - rect_cutout in
  fenced_area = 334 :=
by
  let total_area := 20 * 18
  let square_cutout := 4 * 4
  let rect_cutout := 2 * 5
  let fenced_area := total_area - square_cutout - rect_cutout
  show fenced_area = 334 from sorry

end area_of_fenced_region_l514_514029


namespace ratio_eq_sqrt3_div_2_l514_514939

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514939


namespace count_repeating_decimals_l514_514605

theorem count_repeating_decimals (s : Set ℕ) :
  (∀ n, n ∈ s ↔ 1 ≤ n ∧ n ≤ 20 ∧ ¬∃ k, k * 3 = n) →
  (s.card = 14) :=
by 
  sorry

end count_repeating_decimals_l514_514605


namespace equilateral_triangle_ratio_l514_514862

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514862


namespace solve_common_foci_problem_l514_514793

variables {F1 F2 P : Type} [EuclideanGeometry P]

-- Assume point P, foci F1, F2, eccentricities e1 and e2, and the given condition
def ellipse_hyperbola_common_foci (P F1 F2 : P) (e1 e2 : ℝ) : Prop :=
  -- Conditions
  let c := dist F1 F2 / 2 in
  let a := c / e1 in
  let m := c / e2 in
  dist (vector P F1) * dist (vector P F2) = 0 ∧
  -- Proof Goal
  1 / (e1 * e1) + 1 / (e2 * e2) = 2

theorem solve_common_foci_problem (P F1 F2 : P) (e1 e2 : ℝ) :
  ellipse_hyperbola_common_foci P F1 F2 e1 e2 → 1 / (e1 * e1) + 1 / (e2 * e2) = 2 :=
by
  sorry

end solve_common_foci_problem_l514_514793


namespace num_repeating_decimals_between_1_and_20_l514_514593

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end num_repeating_decimals_between_1_and_20_l514_514593


namespace length_of_bridge_is_correct_l514_514128

-- Define the constants provided in the conditions
def length_of_train : ℝ := 200
def time_to_pass_bridge : ℝ := 21.04615384615385
def speed_of_train_kmh : ℝ := 65
def speed_of_train_ms : ℝ := 65 * 1000 / 3600
def total_distance_covered : ℝ := speed_of_train_ms * time_to_pass_bridge
def length_of_bridge : ℝ := total_distance_covered - length_of_train

-- The theorem that we need to prove
theorem length_of_bridge_is_correct : length_of_bridge ≈ 180 := sorry

end length_of_bridge_is_correct_l514_514128


namespace probability_divisible_by_3_l514_514464

noncomputable def count_valid_triples : Nat :=
  (Finset.filter (λ (xyz : ℕ × ℕ × ℕ), 
    let (x, y, z) := xyz;
    (xyz - xy - yz - zx + x + y + z) % 3 = 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (Finset.product (Finset.product (Finset.range 16) (Finset.range 16)) (Finset.range 16))).card

theorem probability_divisible_by_3 :
  count_valid_triples / (Nat.choose 15 3) = 12 / 91 := 
sorry

end probability_divisible_by_3_l514_514464


namespace equilateral_triangle_ratio_l514_514866

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514866


namespace preservation_time_at_33_degrees_l514_514824

noncomputable def preservation_time (x : ℝ) (k : ℝ) (b : ℝ) : ℝ :=
  Real.exp (k * x + b)

theorem preservation_time_at_33_degrees (k b : ℝ) 
  (h1 : Real.exp b = 192)
  (h2 : Real.exp (22 * k + b) = 48) :
  preservation_time 33 k b = 24 := by
  sorry

end preservation_time_at_33_degrees_l514_514824


namespace sum_s_sum_t_l514_514361

noncomputable def quadratic_residues (p : ℕ) : List ℕ :=
  (List.range (p-1) |>.map (λ k, (k^2) % p))

-- Part (a)
theorem sum_s (p : ℕ) (hp : p ∈ SetOf Prime) (h1 : p > 3) (h2 : p % 4 = 1) :
  (∑ k in Finset.range ((p - 1) / 2 + 1), (2 * k^2 / p).floor − 2 * (k^2 / p).floor) = (p - 1) / 4 := by
  sorry

-- Part (b)
theorem sum_t (p : ℕ) (hp : p ∈ SetOf Prime) (h1 : p > 3) (h2 : p % 8 = 1) :
  (∑ k in Finset.range ((p - 1) / 2 + 1), (k^2 / p).floor) = 0 := by
  sorry

end sum_s_sum_t_l514_514361


namespace secant_circle_ratio_l514_514518

theorem secant_circle_ratio (IJ r : ℝ) (s₁ s₂ : ℕ) (h₁ : s₁ = 5) (h₂ : s₂ = 9) (h₃ : IJ = 15)
  (h₄ : ∀ s₁ s₂, s₁ / s₂ = 5 / 9) : (IJ / r) = 3 * (√10 / 5) := 
sorry

end secant_circle_ratio_l514_514518


namespace card_game_total_l514_514055

theorem card_game_total (C E O : ℝ) (h1 : E = (11 / 20) * C) (h2 : O = (9 / 20) * C) (h3 : E = O + 50) : C = 500 :=
sorry

end card_game_total_l514_514055


namespace continuous_implies_defined_defined_does_not_imply_continuous_l514_514555

-- Define function continuity at a point x = a
def continuous_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - a) < δ → abs (f x - f a) < ε

-- Prove that if f is continuous at x = a, then f is defined at x = a
theorem continuous_implies_defined (f : ℝ → ℝ) (a : ℝ) : 
  continuous_at f a → ∃ y, f a = y :=
by
  sorry  -- Proof omitted

-- Prove that the definition of f at x = a does not guarantee continuity at x = a
theorem defined_does_not_imply_continuous (f : ℝ → ℝ) (a : ℝ) :
  (∃ y, f a = y) → ¬ continuous_at f a :=
by
  sorry  -- Proof omitted

end continuous_implies_defined_defined_does_not_imply_continuous_l514_514555


namespace series_simplification_l514_514406

theorem series_simplification :
  ∑ k in Finset.range 2016, (1 / (k * real.sqrt (k + 1) + (k + 1) * real.sqrt k)) = 1 :=
by
  sorry

end series_simplification_l514_514406


namespace percentage_change_in_area_is_50_l514_514991

variable (L B : ℝ)

-- Define the original area
def Area_original := L * B

-- Define the new length and breadth
def L_new := L / 2
def B_new := 3 * B

-- Define the new area
def Area_new := L_new * B_new

-- Define the percentage change formula
def percentage_change := ((Area_new L B - Area_original L B) / Area_original L B) * 100

-- The theorem stating the percentage change in area
theorem percentage_change_in_area_is_50 :
  percentage_change L B = 50 := by
  sorry

end percentage_change_in_area_is_50_l514_514991


namespace number_of_ordered_triplets_gcd_distinct_l514_514241

/-- For each positive integer \( n \), let \( g(n) \) be the greatest common divisor (GCD) of \( n \) and 2015.
     Find the number of ordered triplets \((a, b, c)\) that satisfy the following conditions:
 1. \( a, b, c \in \{1, 2, \cdots, 2015\} \);
 2. The seven numbers \( g(a), g(b), g(c), g(a+b), g(b+c), g(c+a), g(a+b+c) \) are all distinct. -/
theorem number_of_ordered_triplets_gcd_distinct : 
  (Finset.card
    { (a, b, c) : ℕ × ℕ × ℕ | 
      a ∈ Finset.range 2016 ∧ 
      b ∈ Finset.range 2016 ∧ 
      c ∈ Finset.range 2016 ∧ 
      let g : ℕ → ℕ := λ n, gcd n 2015 in 
      (g a ≠ g b) ∧ (g a ≠ g c) ∧ (g b ≠ g c) ∧
      (g a ≠ g (a + b)) ∧ (g a ≠ g (a + c)) ∧ (g a ≠ g (b + c)) ∧
      (g b ≠ g (a + b)) ∧ (g b ≠ g (a + c)) ∧ (g b ≠ g (b + c)) ∧
      (g c ≠ g (a + b)) ∧ (g c ≠ g (a + c)) ∧ (g c ≠ g (b + c)) ∧
      (g (a + b) ≠ g (a + c)) ∧ (g (a + b) ≠ g (b + c)) ∧ (g (a + c) ≠ g (b + c)) ∧
      (g a ≠ g (a + b + c)) ∧ (g b ≠ g (a + b + c)) ∧ (g c ≠ g (a + b + c)) ∧
      (g (a + b) ≠ g (a + b + c)) ∧ (g (a + c) ≠ g (a + b + c)) ∧ (g (b + c) ≠ g (a + b + c))
    }) = 138240 :=
by
  sorry

end number_of_ordered_triplets_gcd_distinct_l514_514241


namespace minimize_weighted_sum_of_distances_l514_514236

theorem minimize_weighted_sum_of_distances 
  (ABC : Prop) (O XY : Point) (m n p AO BO CO XY YZ ZX : Real)
  (h_pos_m : m > 0) (h_pos_n : n > 0) (h_pos_p : p > 0) 
  (h_AO : AO = m * λ) 
  (h_BO : BO = n * λ) 
  (h_CO : CO = p * λ) 
  (triangle_ABC : ABC) :
  ∃ O, O ∈ triangle_ABC ∧ 
         ∀ XYZ, (m * XY + n * YZ + p * ZX) is minimized :=
sorry

end minimize_weighted_sum_of_distances_l514_514236


namespace exists_not_perfect_square_l514_514405

theorem exists_not_perfect_square (a b c : ℤ) : ∃ (n : ℕ), n > 0 ∧ ¬ ∃ k : ℕ, n^3 + a * n^2 + b * n + c = k^2 :=
by
  sorry

end exists_not_perfect_square_l514_514405


namespace sum_of_numerator_denominator_of_b3_equals_33_l514_514548

theorem sum_of_numerator_denominator_of_b3_equals_33 :
  let b : ℕ → ℚ :=
    λ n, if n = 1 then 2 else
         if n = 2 then 5 / 11 else
         (b (n-2) * b (n-1)) / (3 * b (n-2) - 2 * b (n-1))
  in  (let ⟨num, denom⟩ := Rat.num_denom (b 3) in num + denom = 33) :=
by
  sorry

end sum_of_numerator_denominator_of_b3_equals_33_l514_514548


namespace johnny_money_left_l514_514356

def total_saved (september october november : ℕ) : ℕ := september + october + november

def money_left (total amount_spent : ℕ) : ℕ := total - amount_spent

theorem johnny_money_left 
    (saved_september : ℕ)
    (saved_october : ℕ)
    (saved_november : ℕ)
    (spent_video_game : ℕ)
    (h1 : saved_september = 30)
    (h2 : saved_october = 49)
    (h3 : saved_november = 46)
    (h4 : spent_video_game = 58) :
    money_left (total_saved saved_september saved_october saved_november) spent_video_game = 67 := 
by sorry

end johnny_money_left_l514_514356


namespace equilateral_triangle_ratio_l514_514907

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514907


namespace original_commercial_length_l514_514471

theorem original_commercial_length (x : ℝ) (h : 0.70 * x = 21) : x = 30 := sorry

end original_commercial_length_l514_514471


namespace square_area_sum_exceeds_l514_514259

noncomputable def sum_of_areas (n : ℕ) : ℝ := 50 * (1 - (1/2)^n)

theorem square_area_sum_exceeds :
  ∃ n : ℕ, sum_of_areas n > 49
:= by
  use 6
  simp [sum_of_areas]
  linarith

end square_area_sum_exceeds_l514_514259


namespace rate_per_square_meter_is_3_l514_514819

def floor_painting_rate 
  (length : ℝ) 
  (total_cost : ℝ)
  (length_more_than_breadth_by_percentage : ℝ)
  (expected_rate : ℝ) : Prop :=
  ∃ (breadth : ℝ) (rate : ℝ),
    length = (1 + length_more_than_breadth_by_percentage / 100) * breadth ∧
    total_cost = length * breadth * rate ∧
    rate = expected_rate

-- Given conditions
theorem rate_per_square_meter_is_3 :
  floor_painting_rate 15.491933384829668 240 200 3 :=
by
  sorry

end rate_per_square_meter_is_3_l514_514819


namespace remove_one_digit_to_fair_number_l514_514254

def is_odd_digit_number (n : ℕ) : Prop := (n % 2 = 1)

def num_sevens_in_positions (n : ℕ) (positions : finset ℕ) : ℕ :=
positions.filter (λ i, (n / 10^i % 10 = 7)).card

def fair_number (n : ℕ) : Prop :=
  (num_sevens_in_positions n {i | i % 2 = 0}) = (num_sevens_in_positions n {i | i % 2 = 1})

theorem remove_one_digit_to_fair_number (n : ℕ) (h : is_odd_digit_number n) : 
  ∃ m, (m < n) ∧ fair_number m :=
sorry

end remove_one_digit_to_fair_number_l514_514254


namespace circle_area_and_diameter_l514_514801

-- Define the conditions
def C : ℝ := 36
def r : ℝ := C / (2 * Real.pi)
def A : ℝ := Real.pi * r^2
def d : ℝ := 2 * r

-- Define the theorem to be proved
theorem circle_area_and_diameter :
  A = 324 / Real.pi ∧ d = 36 / Real.pi :=
by
  sorry -- Proof to be provided

end circle_area_and_diameter_l514_514801


namespace triangle_perimeter_inscribed_circle_l514_514818

theorem triangle_perimeter_inscribed_circle
  (radius : ℝ)
  (DP PE : ℝ) :
  radius = 15 → DP = 19 → PE = 31 → 
  2 * (50 + (DP + PE + (2 * radius - DP - PE))) = (10075 : ℝ) / 91 :=
by
  intro h_radius h_DP h_PE
  sorry -- proof omitted

-- Definitions of provided values
def radius : ℝ := 15
def DP : ℝ := 19
def PE : ℝ := 31

-- Proves the theorem with the given conditions
#reduce triangle_perimeter_inscribed_circle 15 19 31 rfl rfl rfl

end triangle_perimeter_inscribed_circle_l514_514818


namespace ellipse_standard_and_trajectory_l514_514651

-- Problem Statement
theorem ellipse_standard_and_trajectory 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : a > 0)
  (h4 : (a^2 - b^2) = 1)
  (M : ℝ × ℝ) (hM : M = (3/2, sqrt 6))
  (h_ellipse : (M.1^2) / (a^2) + (M.2^2) / (b^2) = 1)
  (focal_length : ℝ) (h_focal_length : focal_length = 2)
  (c : ℝ) (hc : 2*c = focal_length)
  (P1 : ℝ × ℝ) (P2 : ℝ × ℝ) (hP1 : P1.2 ≠ 0 ∧ abs P1.1 < a)
  (hP2 : P2 = (P1.1, -P1.2))
  (A1 A2 : ℝ × ℝ) (hA1 : A1 = (-a, 0)) (hA2 : A2 = (a, 0))
  (x y : ℝ) (h_non_vertex : x ≠ ±a)
: (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a > b) ∧ ∃ (F_1 F_2 : ℝ × ℝ), ∀ M : ℝ × ℝ,
   ((M.1^2) / a^2 + (M.2^2) / b^2 = 1) → ((M.1 + 1) + (M.1 - 1)) = 2*a ∧ b^2 = 8 ∧ (M = P1)) ∧
   (∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∧ x ≠ ±a)
:= sorry

end ellipse_standard_and_trajectory_l514_514651


namespace trains_cross_each_other_in_time_l514_514850

-- Conditions
def length := 120 -- in meters
def time1 := 10 -- seconds for the first train to cross a telegraph post
def time2 := 50 -- seconds for the second train to cross a telegraph post

-- Speed calculations
def V1 := length / time1 -- Speed of the first train in meters/second
def V2 := length / time2 -- Speed of the second train in meters/second
def V_relative := V1 + V2 -- Relative speed when traveling in opposite directions

-- Total distance to be covered
def D_total := length + length -- Sum of the lengths of both trains in meters

-- Time to cross each other
def T := D_total / V_relative -- Time in seconds to cross each other

-- Theorem to prove
theorem trains_cross_each_other_in_time : T = 240 / 14.4 := by
  sorry

end trains_cross_each_other_in_time_l514_514850


namespace truck_travel_distance_l514_514131

theorem truck_travel_distance (distance_per_10_gallons : ℕ) (gallons_used : ℕ) (new_gallons : ℕ) :
  distance_per_10_gallons = 240 → gallons_used = 10 → new_gallons = 15 →
  (distance_per_10_gallons / gallons_used) * new_gallons = 360 :=
by
  intros h_dist h_gallons h_new
  rw [h_dist, h_gallons, h_new]
  sorry

end truck_travel_distance_l514_514131


namespace ratio_eq_sqrt3_div_2_l514_514946

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514946


namespace math_problem_l514_514158

theorem math_problem : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end math_problem_l514_514158


namespace degrees_of_remainder_division_l514_514079

theorem degrees_of_remainder_division (f g : Polynomial ℝ) (h : g = Polynomial.C 3 * Polynomial.X ^ 3 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 1 * Polynomial.X + Polynomial.C (-8)) :
  ∃ r q : Polynomial ℝ, f = g * q + r ∧ (r.degree < 3) := 
sorry

end degrees_of_remainder_division_l514_514079


namespace teachers_with_neither_percentage_l514_514125

def total_teachers : Nat := 150
def teachers_with_high_bp : Nat := 90
def teachers_with_heart_trouble : Nat := 50
def teachers_with_both : Nat := 30

theorem teachers_with_neither_percentage :
  (total_teachers - ((teachers_with_high_bp - teachers_with_both) + (teachers_with_heart_trouble - teachers_with_both) + teachers_with_both)) * 100 / total_teachers = 26.67 :=
by
  sorry

end teachers_with_neither_percentage_l514_514125


namespace quadratic_k_value_l514_514665

theorem quadratic_k_value (a b c k : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : 4 * b * b - k * a * c = 0): 
  k = 16 / 3 :=
by
  sorry

end quadratic_k_value_l514_514665


namespace sum_of_coordinates_l514_514537

open Real

-- Definition of the problem's conditions
def point_distance (p q : ℝ × ℝ) (d : ℝ) : Prop :=
  dist p q = d

def point_line_distance (p : ℝ × ℝ) (y_line : ℝ) (d : ℝ) : Prop :=
  abs (p.2 - y_line) = d

-- Define the conditions using the given parameters
def condition (p : ℝ × ℝ) : Prop :=
  point_distance p (6, 15) 11 ∧ point_line_distance p 15 4

-- Define the statement of the problem's conclusion
theorem sum_of_coordinates :
  let points := {p : ℝ × ℝ | condition p} in
  ∑ p in points.to_finset, (p.1 + p.2) = 84 :=
sorry

end sum_of_coordinates_l514_514537


namespace equal_segments_condition_l514_514832

theorem equal_segments_condition (n : ℕ) :
  (∃ (f : Fin n → Bool), let line_segment : Fin n → Fin n → Bool := 
    λ i j => (f i = f j) in 
    (∑ i j in range n, ite (line_segment i j) 1 0) = 
    (∑ i j in range n, ite (¬ line_segment i j) 1 0)) ↔ ∃ k : ℕ, n = k ^ 2 :=
sorry

end equal_segments_condition_l514_514832


namespace equilateral_triangle_ratio_l514_514864

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514864


namespace quadratic_roots_shift_is_correct_l514_514755

theorem quadratic_roots_shift_is_correct :
  let p q : ℝ
  (hroots : ∀ (p q : ℝ), (3 * p^2 - 5 * p - 7 = 0) ∧ (3 * q^2 - 5 * q - 7 = 0)) in
  let c : ℝ := (p + 2) * (q + 2) in
  c = 5 :=
by
  -- Define roots
  let p := (-(-5) ± real.sqrt ((-5)^2 - 4 * 3 * (-7))) / (2 * 3)
  let q := (-(-5) ± real.sqrt ((-5)^2 - 4 * 3 * (-7))) / (2 * 3)
  
  -- Compute c
  let c := (p + 2) * (q + 2)

  -- Check the answer
  show c = 5
  sorry

end quadratic_roots_shift_is_correct_l514_514755


namespace As_annual_income_l514_514437

theorem As_annual_income :
  let Cm := 14000
  let Bm := Cm + 0.12 * Cm
  let Am := (5 / 2) * Bm
  Am * 12 = 470400 := by
  sorry

end As_annual_income_l514_514437


namespace num_repeating_decimals_1_to_20_l514_514614

theorem num_repeating_decimals_1_to_20 : 
  (∃ count_repeating : ℕ, count_repeating = 18 ∧ 
    ∀ n, 1 ≤ n ∧ n ≤ 20 → ((∃ k, n = 9 * k ∨ n = 18 * k) → false) → 
        (∃ d, (∃ normalized, n / 18 = normalized ∧ normalized.has_repeating_decimal))) :=
sorry

end num_repeating_decimals_1_to_20_l514_514614


namespace isosceles_right_triangle_rel_l514_514364

theorem isosceles_right_triangle_rel {a x : ℝ} (A B C P : ℝ × ℝ)
  (hA : A = (-a, 0)) (hB : B = (a, 0)) (hC : C = (0, a))
  (hP : P = (x, 0)) :
  let AP := (P.1 + a)^2 + P.2^2,
      BP := (P.1 - a)^2 + P.2^2,
      CP := P.1^2 + (a - P.2)^2,
      s := AP + BP in
  s = 2 * CP := by
  sorry

end isosceles_right_triangle_rel_l514_514364


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514956

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514956


namespace zeroes_y_minus_a_l514_514657

open Real

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then |2 ^ x - 1| else 3 / (x - 1)

theorem zeroes_y_minus_a (a : ℝ) : (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) → (0 < a ∧ a < 1) :=
sorry

end zeroes_y_minus_a_l514_514657


namespace teachers_with_neither_percentage_l514_514124

def total_teachers : Nat := 150
def teachers_with_high_bp : Nat := 90
def teachers_with_heart_trouble : Nat := 50
def teachers_with_both : Nat := 30

theorem teachers_with_neither_percentage :
  (total_teachers - ((teachers_with_high_bp - teachers_with_both) + (teachers_with_heart_trouble - teachers_with_both) + teachers_with_both)) * 100 / total_teachers = 26.67 :=
by
  sorry

end teachers_with_neither_percentage_l514_514124


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514957

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514957


namespace equation_of_line_l514_514445

-- Definitions for the conditions
variables {u v m x y : ℝ}

-- Hypotheses based on the given conditions
hypotheses
  (h1 : x = (1 - u^2 - v^2) / ((1 - u)^2 + v^2))
  (h2 : y = 2*v / ((1 - u)^2 + v^2))
  (h3 : v = m * u)
  (h4 : m ≠ 0)

-- The proof goal stating the relationship in terms of x, y, and m
theorem equation_of_line:
  x^2 + (y - 1/m)^2 = 1 + 1/m^2 := 
sorry

end equation_of_line_l514_514445


namespace equilateral_triangle_ratio_l514_514961

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514961


namespace triangle_area_perimeter_ratio_l514_514916

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514916


namespace triangle_area_perimeter_ratio_l514_514919

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514919


namespace percent_neither_condition_l514_514122

namespace TeachersSurvey

variables (Total HighBloodPressure HeartTrouble Both: ℕ)

theorem percent_neither_condition :
  Total = 150 → HighBloodPressure = 90 → HeartTrouble = 50 → Both = 30 →
  (HighBloodPressure + HeartTrouble - Both) = 110 →
  ((Total - (HighBloodPressure + HeartTrouble - Both)) * 100 / Total) = 2667 / 100 :=
by
  intros hTotal hBP hHT hBoth hUnion
  sorry

end TeachersSurvey

end percent_neither_condition_l514_514122


namespace min_value_norm_sub_proof_l514_514649

variable {V : Type*} [inner_product_space ℝ V]

noncomputable def min_value_norm_sub (a b : V) (t : ℝ) : ℝ :=
  if h : t > 0 ∧ ∥b∥ = 1 ∧ real.arccos ((⟪a, b⟫ / ∥a∥).to_real) = π / 3
  then real.sqrt ((t * ∥a∥ - 1 / 2) ^ 2 + 3 / 4)
  else 0

theorem min_value_norm_sub_proof (a b : V) (t : ℝ) (ht : t > 0) (hb : ∥b∥ = 1)
  (h_angle : real.arccos ((⟪a, b⟫ / ∥a∥).to_real) = π / 3) :
  min_value_norm_sub a b t = real.sqrt 3 / 2 :=
by sorry

end min_value_norm_sub_proof_l514_514649


namespace discounted_price_is_correct_l514_514099

def marked_price : ℕ := 125
def discount_rate : ℚ := 4 / 100

def calculate_discounted_price (marked_price : ℕ) (discount_rate : ℚ) : ℚ :=
  marked_price - (discount_rate * marked_price)

theorem discounted_price_is_correct :
  calculate_discounted_price marked_price discount_rate = 120 := by
  sorry

end discounted_price_is_correct_l514_514099


namespace pentagon_height_ordering_l514_514150

variable {P : Type} [Polygon P]
variable (angles_108 : ∀ i, angle P i = 108)
variable (heights_different : ∀ i j, i ≠ j → height P i ≠ height P j)

theorem pentagon_height_ordering :
  ∃ (labeling : Fin 5 → Fin 5), 
  let m := λ i, height P (labeling i) in
  m 0 > m 2 ∧
  m 2 > m 3 ∧
  m 3 > m 4 ∧
  m 4 > m 1 := by
  sorry

end pentagon_height_ordering_l514_514150


namespace smallest_n_l514_514826

def sequence (a S : ℕ → ℕ) : Prop :=
  (S 3 = 13) ∧ (∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1)

noncomputable def a5 (a S : ℕ → ℕ) [sequence a S] : ℕ :=
  a 5

noncomputable def Sn (n : ℕ) (a S : ℕ → ℕ) [sequence a S] : ℕ :=
  S n

theorem smallest_n (a S : ℕ → ℕ) [sequence a S] :
  ∃ n : ℕ, Sn n a S > a5 a S ∧ ∀ m : ℕ, m < n → Sn m a S ≤ a5 a S :=
begin
  sorry
end

end smallest_n_l514_514826


namespace cars_cleaned_per_day_l514_514011

theorem cars_cleaned_per_day
  (money_per_car : ℕ)
  (total_money : ℕ)
  (days : ℕ)
  (h1 : money_per_car = 5)
  (h2 : total_money = 2000)
  (h3 : days = 5) :
  (total_money / (money_per_car * days)) = 80 := by
  sorry

end cars_cleaned_per_day_l514_514011


namespace probability_is_2_over_7_l514_514782

def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop := ¬ is_rational x

def given_numbers := [36 / 7, 3.1415926, (real.of_int 10 / 3), real.sqrt 4, real.sqrt 5, -real.cbrt 8, real.cbrt 9]

def irrational_number_count : ℕ := given_numbers.countp (λ x, is_irrational x)

def total_number_count : ℕ := given_numbers.length

def probability_of_irrational : ℚ := irrational_number_count / total_number_count

theorem probability_is_2_over_7 : probability_of_irrational = 2 / 7 :=
sorry

end probability_is_2_over_7_l514_514782


namespace ab_cd_not_prime_l514_514371

theorem ab_cd_not_prime {a b c d : ℤ} (h1: a > b) (h2: b > c) (h3: c > d) (h4: d > 0)
    (h5: ac + bd = (b + d + a - c) * (b + d - a + c)) : ¬ prime (ab + cd) :=
begin
  -- proof goes here
  sorry
end

end ab_cd_not_prime_l514_514371


namespace square_of_chord_l514_514170

-- Definitions of the circles and tangency conditions
def radius1 := 4
def radius2 := 8
def radius3 := 12

-- The internal and external tangents condition
def externally_tangent (r1 r2 : ℕ) : Prop := ∃ O1 O2, dist O1 O2 = r1 + r2
def internally_tangent (r_in r_out : ℕ) : Prop := ∃ O_in O_out, dist O_in O_out = r_out - r_in

theorem square_of_chord :
  externally_tangent radius1 radius2 ∧ 
  internally_tangent radius1 radius3 ∧
  internally_tangent radius2 radius3 →
  (∃ PQ : ℚ, PQ^2 = 3584 / 9) :=
by
  intros h
  sorry

end square_of_chord_l514_514170


namespace least_positive_integer_divisible_by_7_11_13_l514_514857

theorem least_positive_integer_divisible_by_7_11_13 : ∃ n : ℕ, n > 0 ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) ∧ n = 1001 :=
by {
  use 1001,
  split,
  by norm_num,
  split,
  by norm_num [Int.ediv, Int.dvd_rem_div_eq],
  split,
  by norm_num [Int.ediv, Int.dvd_rem_div_eq],
  split,
  by norm_num [Int.ediv, Int.dvd_rem_div_eq],
  refl,
}

end least_positive_integer_divisible_by_7_11_13_l514_514857


namespace two_circles_common_tangents_l514_514066

theorem two_circles_common_tangents (r1 r2 : ℝ) (h : r1 ≠ r2) :
  ∃ n : ℕ, n ∈ {0, 1, 2, 3} ∧ (∀ t, t ≠ n → ¬ (t is the number of common tangents of the circles)) := sorry

end two_circles_common_tangents_l514_514066


namespace range_of_a_log_l514_514191

noncomputable def positive_log_condition (a : ℝ) : Prop :=
  ∀ x ∈ Icc (1 : ℝ) (2 : ℝ), 0 < x^2 - a * x + 3 ∧ x^2 - a * x + 3 < 1

theorem range_of_a_log (a : ℝ) : positive_log_condition a ↔ 3 < a ∧ a < 2 * Real.sqrt 3 :=
sorry

end range_of_a_log_l514_514191


namespace repeating_decimals_for_n_div_18_l514_514596

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end repeating_decimals_for_n_div_18_l514_514596


namespace uki_profit_l514_514067

theorem uki_profit :
  let cupcake_price := 1.5
  let cookie_price := 2
  let biscuit_price := 1
  let cupcake_cost := 0.75
  let cookie_cost := 1
  let biscuit_cost := 0.5
  let daily_cupcakes := 20
  let daily_cookies := 10
  let daily_biscuits := 20 in
  let daily_earnings := (daily_cupcakes * cupcake_price) + (daily_cookies * cookie_price) + (daily_biscuits * biscuit_price) in
  let daily_expenses := (daily_cupcakes * cupcake_cost) + (daily_cookies * cookie_cost) + (daily_biscuits * biscuit_cost) in
  let daily_profit := daily_earnings - daily_expenses in
  daily_profit * 5 = 175 :=
by
  sorry

end uki_profit_l514_514067


namespace ellipse_graph_equivalence_l514_514187

theorem ellipse_graph_equivalence :
  ∀ x y : ℝ, x^2 + 4 * y^2 - 6 * x + 8 * y + 9 = 0 ↔ (x - 3)^2 / 4 + (y + 1)^2 / 1 = 1 := by
  sorry

end ellipse_graph_equivalence_l514_514187


namespace problem_l514_514313

theorem problem (h : ℤ) : (∃ x : ℤ, x = -2 ∧ x^3 + h * x - 12 = 0) → h = -10 := by
  sorry

end problem_l514_514313


namespace no_such_function_exists_l514_514226

def satisfies_condition (f : ℤ → ℤ) : Prop :=
  ∀ x y z : ℤ, f (x * y) + f (x * z) - f x * f (y * z) ≤ -1

theorem no_such_function_exists : (∃ f : ℤ → ℤ, satisfies_condition f) = false :=
by
  sorry

end no_such_function_exists_l514_514226


namespace central_angle_remains_unchanged_l514_514441

theorem central_angle_remains_unchanged
  (r l : ℝ)
  (h_r : r > 0)
  (h_l : l > 0) :
  (l / r) = (2 * l) / (2 * r) :=
by
  sorry

end central_angle_remains_unchanged_l514_514441


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514887

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514887


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514953

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514953


namespace equilateral_triangle_ratio_l514_514868

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514868


namespace ratio_eq_sqrt3_div_2_l514_514940

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514940


namespace inequality_does_not_hold_l514_514312

theorem inequality_does_not_hold {x y : ℝ} (h : x > y) : ¬ (-2 * x > -2 * y) ∧ (2023 * x > 2023 * y) ∧ (x - 1 > y - 1) ∧ (-x / 3 < -y / 3) :=
by {
  sorry
}

end inequality_does_not_hold_l514_514312


namespace range_of_function_l514_514228

theorem range_of_function :
  ∀ x, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -3 ≤ 2 * Real.sin x - 1 ∧ 2 * Real.sin x - 1 ≤ 1 :=
by
  intros x h
  sorry

end range_of_function_l514_514228


namespace find_x_l514_514508

theorem find_x (x n q r : ℕ) (h_n : n = 220080) (h_sum : n = (x + 445) * (2 * (x - 445)) + r) (h_r : r = 80) : 
  x = 555 :=
by
  have eq1 : n = 220080 := h_n
  have eq2 : n =  (x + 445) * (2 * (x - 445)) + r := h_sum
  have eq3 : r = 80 := h_r
  sorry

end find_x_l514_514508


namespace find_x_eq_3_l514_514754

theorem find_x_eq_3 (x : ℕ) (h_pos : 0 < x) (h_quotient : ∃ q r, n = q * d + r ∧ r < d) :
  x = 3 :=
by
  let n := x^2 + 4 * x + 23
  let d := 3 * x + 7
  sorry

end find_x_eq_3_l514_514754


namespace triangle_area_isosceles_l514_514467

theorem triangle_area_isosceles (A B C P : ℝ) (h1 : is_isosceles_triangle A B C) (h2 : A + B = 65) (h3 : C = 65) (h4 : perpendicular_distance P A B = 24) (h5 : perpendicular_distance P A C = 36) :
  area_of_triangle A B C = 2535 :=
by
  sorry

end triangle_area_isosceles_l514_514467


namespace income_before_taxes_l514_514990

/-- Define given conditions -/
def net_income (x : ℝ) : ℝ := x - 0.10 * (x - 3000)

/-- Prove that the income before taxes must have been 13000 given the conditions. -/
theorem income_before_taxes (x : ℝ) (hx : net_income x = 12000) : x = 13000 :=
by sorry

end income_before_taxes_l514_514990


namespace axis_of_symmetry_l514_514678

theorem axis_of_symmetry (g : ℝ → ℝ) (h: ∀ x, g x = g (3 - x)) : 
  is_axis_of_symmetry g 1.5 := 
sorry

end axis_of_symmetry_l514_514678


namespace radius_of_wheel_l514_514132

theorem radius_of_wheel (π := Real.pi) :
  (∃ radius : ℝ, 
    ∀ (distance_covered total_revolutions : ℝ), 
    distance_covered = 2816 ∧ total_revolutions = 2000 → 
    radius ≈ 0.224
  ) :=
by
  sorry

end radius_of_wheel_l514_514132


namespace cylinder_ratio_max_volume_l514_514564

theorem cylinder_ratio_max_volume 
    (l w : ℝ) 
    (r : ℝ) 
    (h : ℝ)
    (H_perimeter : 2 * l + 2 * w = 12)
    (H_length_circumference : l = 2 * π * r)
    (H_width_height : w = h) :
    (∀ V : ℝ, V = π * r^2 * h) →
    (∀ r : ℝ, r = 2 / π) →
    ((2 * π * r) / h = 2) :=
sorry

end cylinder_ratio_max_volume_l514_514564


namespace equilateral_triangle_ratio_l514_514913

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514913


namespace percentage_divisible_by_5_l514_514974

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def count_divisible_by_5 (N : ℕ) : ℕ := (finset.filter is_divisible_by_5 (finset.range (N + 1))).card

def percentage (part whole : ℕ) : ℝ := (part.to_real / whole.to_real) * 100

theorem percentage_divisible_by_5 : percentage (count_divisible_by_5 100) 100 = 20 :=
by
  sorry

end percentage_divisible_by_5_l514_514974


namespace units_digit_of_fourth_power_l514_514481

theorem units_digit_of_fourth_power (n : ℕ) : ¬ (n ^ 4 % 10 = 7) :=
by
  intro h
  have H : List.mem (n ^ 4 % 10) [0, 1, 6, 5] := by
    cases n % 10 <;> simp
    all_goals sorry
  rw h at H
  exact List.not_mem_of_mem H (by simp)

end units_digit_of_fourth_power_l514_514481


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514890

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514890


namespace count_possible_values_l514_514804

-- Define the decimal representation conditions
def is_four_place_decimal (r : ℝ) : Prop :=
  ∃ a b c d : ℕ, r = a * 0.1 + b * 0.01 + c * 0.001 + d * 0.0001

-- Define the specific range for r
def in_specific_range (r : ℝ) : Prop :=
  0.2679 ≤ r ∧ r ≤ 0.2929

-- Main theorem statement
theorem count_possible_values : (∃ r, is_four_place_decimal r ∧ in_specific_range r) → 251 :=
by sorry

end count_possible_values_l514_514804


namespace tommy_balloons_l514_514064

/-- Tommy had some balloons. He received 34 more balloons from his mom,
gave away 15 balloons, and exchanged the remaining balloons for teddy bears
at a rate of 3 balloons per teddy bear. After these transactions, he had 30 teddy bears.
Prove that Tommy started with 71 balloons -/
theorem tommy_balloons : 
  ∃ B : ℕ, (B + 34 - 15) = 3 * 30 ∧ B = 71 := 
by
  have h : (71 + 34 - 15) = 3 * 30 := by norm_num
  exact ⟨71, h, rfl⟩

end tommy_balloons_l514_514064


namespace equilateral_triangle_ratio_l514_514962

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514962


namespace options_simplify_to_AD_l514_514082

def optionA := sorry -- \(\overrightarrow{MB} + \overrightarrow{AD} - \overrightarrow{BM}\)
def optionB := sorry -- \((\overrightarrow{AD} + \overrightarrow{MB}) + (\overrightarrow{BC} + \overrightarrow{CM})\)
def optionC := sorry -- \((\overrightarrow{AB} + \overrightarrow{CD}) + \overrightarrow{BC}\)
def optionD := sorry -- \(\overrightarrow{OC} - \overrightarrow{OA} + \overrightarrow{CD}\)
def AD := sorry -- \(\overrightarrow{AD}\)

theorem options_simplify_to_AD : (optionB = AD) ∧ (optionC = AD) ∧ (optionD = AD) :=
by {
  sorry
}

end options_simplify_to_AD_l514_514082


namespace usual_walk_time_l514_514473

theorem usual_walk_time (S T : ℝ)
  (h : S / (2/3 * S) = (T + 15) / T) : T = 30 :=
by
  sorry

end usual_walk_time_l514_514473


namespace chord_length_square_l514_514174

theorem chord_length_square {r₁ r₂ R : ℝ} (r1_eq_4 : r₁ = 4) (r2_eq_8 : r₂ = 8) 
  (R_eq_12 : R = 12)
  (externally_tangent : tangent r₁ r₂)
  (internally_tangent_1 : tangent_internally r₁ R)
  (internally_tangent_2 : tangent_internally r₂ R)
  : exists (PQ : ℝ), PQ^2 = 3584 / 9 :=
by sorry

end chord_length_square_l514_514174


namespace orthogonal_projection_center_excircle_l514_514072

variables (A B C D E F: Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables [add_comm_group A] [module ℝ A] [add_comm_group B] [module ℝ B] [add_comm_group C] [module ℝ C]
variables [add_comm_group D] [module ℝ D] [add_comm_group E] [module ℝ E] [add_comm_group F] [module ℝ F]

/-- Orthogonal projection of vertex C of the triangle onto the plane of triangle DEF coincides with 
the center of the excircle of triangle DEF opposite to side DE -/
theorem orthogonal_projection_center_excircle
  (h_ABC : isosceles_triangle A B C)
  (h_D : is_closer D A B)
  (h_E : is_closer E B A)
  (h_rotationACD : rotated_around A C D F)
  (h_rotationBCE : rotated_around B C E F) :
  orthogonal_projection C (plane DEF) = center_excircle DEF DE :=
sorry

end orthogonal_projection_center_excircle_l514_514072


namespace total_amount_l514_514085

theorem total_amount (x y z : ℝ) 
  (hy : y = 0.45 * x) 
  (hz : z = 0.30 * x) 
  (hy_value : y = 54) : 
  x + y + z = 210 := 
by
  sorry

end total_amount_l514_514085


namespace discount_percentage_l514_514528

theorem discount_percentage (C M A : ℝ) (h1 : M = 1.40 * C) (h2 : A = 1.05 * C) :
    (M - A) / M * 100 = 25 :=
by
  sorry

end discount_percentage_l514_514528


namespace constant_term_in_binomial_expansion_l514_514802

theorem constant_term_in_binomial_expansion (x : ℝ) (hx : x ≠ 0) :
  constant_term (∑ k in Finset.range 11, (Nat.choose 10 k) * 
    (((1 : ℝ) / (Real.sqrt x)) ^ k) * ((-x^2) ^ (10 - k))) = 45 :=
by 
  sorry

end constant_term_in_binomial_expansion_l514_514802


namespace repeating_decimals_count_l514_514602

theorem repeating_decimals_count :
  { n : ℕ | 1 ≤ n ∧ n ≤ 20 ∧ (¬∃ k, n = 9 * k) }.to_finset.card = 18 :=
by
  sorry

end repeating_decimals_count_l514_514602


namespace decimal_to_octal_2009_l514_514068

theorem decimal_to_octal_2009 : Nat.toDigits 8 2009 = [3, 7, 3, 1] :=
by
  sorry

end decimal_to_octal_2009_l514_514068


namespace repeating_decimals_for_n_div_18_l514_514597

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end repeating_decimals_for_n_div_18_l514_514597


namespace repeating_decimals_count_l514_514603

theorem repeating_decimals_count :
  { n : ℕ | 1 ≤ n ∧ n ≤ 20 ∧ (¬∃ k, n = 9 * k) }.to_finset.card = 18 :=
by
  sorry

end repeating_decimals_count_l514_514603


namespace find_m_for_tangent_line_to_circle_l514_514039

def line_tangent_to_circle (m : ℝ) : Prop :=
  let circle_eq := (x + 1)^2 + (y + 1)^2 = 2
  let line_eq := (x = m * y + 2)
  tangent_to_circle line_eq circle_eq

theorem find_m_for_tangent_line_to_circle :
  ∃ (m : ℝ), line_tangent_to_circle m ∧ (m = 1 ∨ m = -7) :=
by
  sorry

end find_m_for_tangent_line_to_circle_l514_514039


namespace triangle_area_perimeter_ratio_l514_514917

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514917


namespace min_sum_of_inscribed_circle_areas_l514_514623

theorem min_sum_of_inscribed_circle_areas 
  (r : Real) 
  (A B P : Point) 
  (hAB : A ≠ B) 
  (hPonz : ∃ (P : Point), P ≠ A ∧ P ≠ B ∧ P ≠ midpoint A B)
  (x y : Real) : 
  x^2 + y^2 = r^2 → 
  min_sum_of_areas = (π * r^2 / 4) * (3 - 2 * Real.sqrt 2) := sorry

end min_sum_of_inscribed_circle_areas_l514_514623


namespace seventy_second_number_in_s_l514_514763

def set_s : Set ℤ := { x | ∃ n : ℤ, x = 8 * n + 5 ∧ x > 0 }

theorem seventy_second_number_in_s : ∃ x ∈ set_s, (∃ n : ℕ, n = 71 ∧ x = 8 * n + 5) ∧ x = 573 := 
by
  let x := 8 * 71 + 5
  have h : 71 ∈ ℕ := by {apply Nat.zero_le}
  use x
  split
  . use 71
    split
    . refl
    . split
      . rw [← Int.coe_nat_eq_coe_nat_iff]
        exact h
      . ring
  . refl

end seventy_second_number_in_s_l514_514763


namespace value_of_expression_l514_514681

theorem value_of_expression (x : ℤ) (h : x = 5) : 2 * x + 3 - 2 = 11 :=
by
  rw h
  norm_num

end value_of_expression_l514_514681


namespace sum_abcd_e_equals_28_l514_514341

-- Define unique digit conditions for A, B, C, D, E
def is_unique_digits (A B C D E : ℕ) : Prop :=
  ([A, B, C, D, E].nodup ∧ ∀ n ∈ [A, B, C, D, E], n < 10)

-- Convert AB and CD to appropriate two-digit numbers
def to_two_digit (x y : ℕ) : ℕ := (10 * x) + y

-- Main statement: Prove that the sum of A + B + C + D + E is 28 given the conditions
theorem sum_abcd_e_equals_28 {A B C D E : ℕ} 
  (h_unique: is_unique_digits A B C D E)
  (h_eqn: (to_two_digit A B) * (to_two_digit C D) = 111 * E) :
  A + B + C + D + E = 28 :=
sorry

end sum_abcd_e_equals_28_l514_514341


namespace value_of_x_l514_514113

theorem value_of_x (x : ℝ) (h_pos : x > 0) (h_eq : real.sqrt ((10 * x) / 3) = x) : x = 10 / 3 :=
sorry

end value_of_x_l514_514113


namespace ratio_eq_sqrt3_div_2_l514_514945

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514945


namespace first_group_men_l514_514695

noncomputable def men_in_first_group (m: ℕ) (d1 d2 t2: ℕ) : ℕ :=
  (20 * t2) / d1

theorem first_group_men : 
  let d1 : ℕ := 80,
      d2 : ℕ := 20,
      t2 : ℕ := 52 in
  men_in_first_group d2 d1 t2 = 13 :=
by {
  unfold men_in_first_group,
  simp,
  sorry
}

end first_group_men_l514_514695


namespace smallest_set_with_mean_property_l514_514519

def arithmetic_mean (s : Finset ℝ) : ℝ :=
  s.sum id / s.card

theorem smallest_set_with_mean_property :
  ∃ (s : Finset ℝ), s.card = 5 ∧
    ∃ (s2 : Finset ℝ), s2.card = 2 ∧ s2 ⊆ s ∧
    ∃ (s3 : Finset ℝ), s3.card = 3 ∧ s3 ⊆ s ∧
    ∃ (s4 : Finset ℝ), s4.card = 4 ∧ s4 ⊆ s ∧
    arithmetic_mean s2 = arithmetic_mean s3 ∧
    arithmetic_mean s3 = arithmetic_mean s4 :=
by 
  sorry

end smallest_set_with_mean_property_l514_514519


namespace equilateral_triangle_ratio_l514_514964

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514964


namespace calculate_expression_l514_514152

theorem calculate_expression : (3.15 * 2.5) - 1.75 = 6.125 := 
by
  -- The proof is omitted, indicated by sorry
  sorry

end calculate_expression_l514_514152


namespace equilateral_triangle_ratio_l514_514880

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514880


namespace factorial_floor_l514_514184

theorem factorial_floor :
  \(\left\lfloor \frac{2011! + 2008!}{2010! + 2009!}\right\rfloor = 2010\) :=
begin
  sorry
end

end factorial_floor_l514_514184


namespace problem_statement_l514_514250

noncomputable def exists_point_A (n : ℕ) (P : Fin (n - 1) → ℂ) : Prop :=
  ∃ A : ℂ, abs A = 1 ∧ (∏ k : Fin (n - 1), abs (A - P k)) ≥ 1

theorem problem_statement (n : ℕ) (h : n > 1) (P : Fin (n - 1) → ℂ) :
    exists_point_A n P := 
sorry

end problem_statement_l514_514250


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514954

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514954


namespace lacy_percentage_correct_l514_514389

variable (x : ℕ)

-- Definitions from the conditions
def total_problems := 8 * x
def missed_problems := 2 * x
def answered_problems := total_problems - missed_problems
def bonus_problems := x
def bonus_points := 2 * bonus_problems
def regular_points := answered_problems - bonus_problems
def total_points_scored := bonus_points + regular_points
def total_available_points := 8 * x + 2 * x

theorem lacy_percentage_correct :
  total_points_scored / total_available_points * 100 = 90 := by
  -- Proof steps would go here, but are not required per instructions.
  sorry

end lacy_percentage_correct_l514_514389


namespace polynomial_product_l514_514983

theorem polynomial_product (x : ℝ) : (x - 1) * (x + 3) * (x + 5) = x^3 + 7*x^2 + 7*x - 15 :=
by
  sorry

end polynomial_product_l514_514983


namespace camel_height_in_feet_l514_514328

theorem camel_height_in_feet (h_ht_14 : ℕ) (ratio : ℕ) (inch_to_ft : ℕ) : ℕ :=
  let hare_height := 14
  let camel_height_in_inches := hare_height * 24
  let camel_height_in_feet := camel_height_in_inches / 12
  camel_height_in_feet
#print camel_height_in_feet

example : camel_height_in_feet 14 24 12 = 28 := by sorry

end camel_height_in_feet_l514_514328


namespace find_m_value_l514_514268

/-- 
If point P(-1, 4) lies on line l₁, and l₁ is parallel to line l₂: 2x - y + 5 = 0, 
then, given that the distance between line l₃: 4x - 2y + m = 0 and line l₁ 
is 2√5, the value of m can be -8 or 32. 
-/
theorem find_m_value (m : ℝ) :
  let l₁ := (2, -1, 6) in  -- line l₁: 2x - y + 6 = 0
  let l₂ := (2, -1, 5) in  -- line l₂: 2x - y + 5 = 0 (for the condition of parallel lines)
  let l₃ := (4, -2, m) in  -- line l₃: 4x - 2y + m = 0
  let P := (-1, 4) in      -- point P(-1, 4)
  dist_line_point l₁ l₃ = 2 * sqrt 5 → -- distance condition
  (m = -8 ∨ m = 32) :=
sorry

end find_m_value_l514_514268


namespace repeated_digit_percentage_l514_514687

noncomputable def percentage_repeated_digits : ℝ :=
  let total_numbers := 90000
  let non_repeated_digits := 9 * 9 * 8 * 7 * 6
  let repeated_digits := total_numbers - non_repeated_digits
  ((repeated_digits.toReal / total_numbers.toReal) * 100).round / 10

theorem repeated_digit_percentage (y : ℝ) : 
  percentage_repeated_digits = y → y = 69.8 :=
by
  intro h
  have : percentage_repeated_digits = 69.8 := sorry
  rw this at h
  exact h

end repeated_digit_percentage_l514_514687


namespace integer_lengths_impossible_l514_514374

open Triangle

theorem integer_lengths_impossible 
  (A B C D E I : Point)
  (hA90 : ∠ BAC = 90)
  (hD_on_AC : D ∈ Line AC)
  (hE_on_AB : E ∈ Line AB)
  (h_angle_ABD_DBC : ∠ ABD = ∠ DBC)
  (h_angle_ACE_ECB : ∠ ACE = ∠ ECB)
  (h_meet_I : Meets (Line BD) (Line CE) I) :
  ¬ (is_integer_length AB ∧ is_integer_length AC ∧
     is_integer_length BI ∧ is_integer_length ID ∧ 
     is_integer_length CI ∧ is_integer_length IE) :=
begin
  sorry
end

end integer_lengths_impossible_l514_514374


namespace graph_admits_acyclic_orientation_l514_514404

-- Definition of a graph structure
structure Graph (V : Type) :=
  (E : set (V × V))

-- Predicate to check if a graph is acyclic when edges are oriented
def acyclic_orientation {V : Type} (G : Graph V) (oriented_edges : V → V → Prop) : Prop :=
  ∀ cycle, ¬(cycle ∈ G.E)

-- The main theorem stating that every graph admits an acyclic orientation
theorem graph_admits_acyclic_orientation {V : Type} (G : Graph V) :
  ∃ (oriented_edges : V → V → Prop), acyclic_orientation G oriented_edges :=
by
  sorry

end graph_admits_acyclic_orientation_l514_514404


namespace equilateral_triangle_ratio_l514_514869

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514869


namespace triangle_area_conversion_l514_514262

/-- Define the given conditions and the conversion factor -/
def base := 12
def height_meters := 5
def height_feet := height_meters * 3.28084 -- 1 meter = 3.28084 feet
def conversion_factor := 10.7639

/-- Define the area calculation in square meters -/
def area_meters := (base * height_meters) / 2

/-- Convert the area from square meters to square feet -/
def area_feet := area_meters * conversion_factor

/-- The target proof statement -/
theorem triangle_area_conversion : 
  area_feet = 323.117 :=
by 
  sorry

end triangle_area_conversion_l514_514262


namespace equilateral_triangle_ratio_l514_514874

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514874


namespace smaller_number_is_five_l514_514047

theorem smaller_number_is_five (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end smaller_number_is_five_l514_514047


namespace last_three_digits_of_product_odds_l514_514492

def product_of_odds_up_to (n : ℕ) : ℕ :=
  ∏ i in finset.range (n // 2 + 1), 2 * i + 1

theorem last_three_digits_of_product_odds :
  product_of_odds_up_to 2019 % 1000 = 875 :=
by
  sorry

end last_three_digits_of_product_odds_l514_514492


namespace scientific_notation_of_number_l514_514840

theorem scientific_notation_of_number :
  (0.000000014 : ℝ) = 1.4 * 10 ^ (-8) :=
sorry

end scientific_notation_of_number_l514_514840


namespace sum_of_valid_a_values_l514_514321

theorem sum_of_valid_a_values :
  let valid_a_values := {a | (∀ x : ℝ, ((x + 1 ≤ (2 * x - 5) / 3) ∧ (a - x > 1) → x ≤ -8)) ∧
                              (∃ y : ℤ, y ≥ 0 ∧ (4 + (y / (y - 3:ℝ)) = (a - 1) / (3 - y:ℝ)))}
  in (∑ a in valid_a_values, a) = 24 :=
by
  sorry

end sum_of_valid_a_values_l514_514321


namespace square_of_chord_length_is_39804_l514_514182

noncomputable def square_of_chord_length (r4 r8 r12 : ℝ) (externally_tangent : (r4 + r8) < r12) : ℝ := 
  let r4 := 4
  let r8 := 8
  let r12 := 12
  let PQ_sq := 4 * ((r12^2) - ((2 * r8 + 1 * r4) / 3)^2) in
  PQ_sq

theorem square_of_chord_length_is_39804 : 
  square_of_chord_length 4 8 12 ((4 + 8) < 12) = 398.04 := 
by
  sorry

end square_of_chord_length_is_39804_l514_514182


namespace sum_turnover_first_20_quarters_turnover_18_percent_profit_l514_514142

-- Definitions for conditions
def annual_turnover := 1.1  -- billion yuan
def turnover_increase := 0.05  -- billion yuan
def initial_profit := 0.16  -- billion yuan
def profit_growth := 0.04  -- 4%

-- Part 1: Sum of the turnover for the first 20 quarters
theorem sum_turnover_first_20_quarters : 
  ∑ k in (finset.range 20), (annual_turnover + k * turnover_increase) = 31.5 := 
begin
  sorry
end

-- Part 2: Turnover is 18% of the profit in the second quarter of the year 2027
theorem turnover_18_percent_profit (n : ℕ) (hn : n = 25) :
  (annual_turnover + n / 4 * turnover_increase) * 0.18 = 
    (initial_profit * (1 + profit_growth) ^ n):= 
begin
  sorry
end

end sum_turnover_first_20_quarters_turnover_18_percent_profit_l514_514142


namespace prove_cardinality_l514_514758

-- Definitions used in Lean 4 Statement adapted from conditions
variable (a b : ℕ)
variable (A B : Finset ℕ)

-- Hypotheses
variable (ha : a > 0)
variable (hb : b > 0)
variable (h_disjoint : Disjoint A B)
variable (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B)

-- The statement to prove
theorem prove_cardinality (a b : ℕ) (A B : Finset ℕ)
  (ha : a > 0) (hb : b > 0) (h_disjoint : Disjoint A B)
  (h_condition : ∀ i ∈ (A ∪ B), i + a ∈ A ∨ i - b ∈ B) :
  a * A.card = b * B.card :=
by 
  sorry

end prove_cardinality_l514_514758


namespace solutions_count_l514_514674

theorem solutions_count (a : ℝ) (h : a < 0) :
  ∃ (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, (a - 1) * (sin (2 * x) + cos x) + (a + 1) * (sin x - cos (2 * x)) = 0 
  ∧ -π < x ∧ x < π :=
sorry

end solutions_count_l514_514674


namespace positive_integer_as_sum_of_distinct_factors_l514_514398

-- Defining that all elements of a list are factors of a given number
def AllFactorsOf (factors : List ℕ) (n : ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Defining that the sum of elements in the list equals a given number
def SumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

-- Theorem statement
theorem positive_integer_as_sum_of_distinct_factors (n m : ℕ) (hn : 0 < n) (hm : 1 ≤ m ∧ m ≤ n!) :
  ∃ factors : List ℕ, factors.length ≤ n ∧ AllFactorsOf factors n! ∧ SumList factors = m := 
sorry

end positive_integer_as_sum_of_distinct_factors_l514_514398


namespace equivalent_proof_problem_l514_514264

noncomputable def conditions : Prop :=
  ∃ C : set (ℝ × ℝ), ∃ f1 f2 : ℝ × ℝ, f1 = (0, sqrt 2) ∧ f2 = (0, -sqrt 2) ∧ 
    (C \ (eq (y^2 - x^2 - 1 = 0)) ∧
    (ecc C = sqrt 6 / 3) ∧
    (lower_vertex C = A) ∧
    (M ≠ A ∧ N ≠ A ∧ M ∈ C ∧ N ∈ C) ∧
    (slopes_prod (AM) (AN) = -3))

theorem equivalent_proof_problem : 
  conditions →
  (standard_eq C = y^2 / 3 + x^2 = 1) ∧ 
  (fixed_point_for_lines (M) (N) = (0,0)) ∧ 
  (∀ P : ℝ × ℝ, P ∈ C ∧ P ≠ M ∧ P ≠ N ∧ dist P M = dist P N →
    min_area_triangle M N P = 3/2) :=
by sorry

end equivalent_proof_problem_l514_514264


namespace real_rate_of_return_is_10_percent_l514_514392

-- Given definitions based on conditions
def nominal_rate := 0.21
def inflation_rate := 0.10

-- Statement to prove
theorem real_rate_of_return_is_10_percent (r : ℝ) :
  1 + r = (1 + nominal_rate) / (1 + inflation_rate) → r = 0.10 := 
by
  sorry

end real_rate_of_return_is_10_percent_l514_514392


namespace num_repeating_decimals_1_to_20_l514_514610

theorem num_repeating_decimals_1_to_20 : 
  (∃ count_repeating : ℕ, count_repeating = 18 ∧ 
    ∀ n, 1 ≤ n ∧ n ≤ 20 → ((∃ k, n = 9 * k ∨ n = 18 * k) → false) → 
        (∃ d, (∃ normalized, n / 18 = normalized ∧ normalized.has_repeating_decimal))) :=
sorry

end num_repeating_decimals_1_to_20_l514_514610


namespace equilateral_triangle_ratio_l514_514909

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514909


namespace Janet_sold_six_action_figures_l514_514353

variable {x : ℕ}

theorem Janet_sold_six_action_figures
  (h₁ : 10 - x + 4 + 2 * (10 - x + 4) = 24) :
  x = 6 :=
by
  sorry

end Janet_sold_six_action_figures_l514_514353


namespace definite_integral_sin_cos_l514_514192

theorem definite_integral_sin_cos :
  ∫ x in -Real.pi / 2 .. Real.pi / 2, (Real.sin x + Real.cos x) = 2 :=
by
  sorry

end definite_integral_sin_cos_l514_514192


namespace quarterly_business_tax_cost_l514_514022

theorem quarterly_business_tax_cost
    (price_federal : ℕ := 50)
    (price_state : ℕ := 30)
    (Q : ℕ)
    (num_federal : ℕ := 60)
    (num_state : ℕ := 20)
    (num_quart_business : ℕ := 10)
    (total_revenue : ℕ := 4400)
    (revenue_equation : num_federal * price_federal + num_state * price_state + num_quart_business * Q = total_revenue) :
    Q = 80 :=
by 
  sorry

end quarterly_business_tax_cost_l514_514022


namespace evaluate_expression_l514_514200

/-- Given conditions: -/
def a : ℕ := 3998
def b : ℕ := 3999

theorem evaluate_expression :
  b^3 - 2 * a * b^2 - 2 * a^2 * b + (b - 2)^3 = 95806315 :=
  sorry

end evaluate_expression_l514_514200


namespace smallest_x_value_l514_514971

theorem smallest_x_value : 
  ∃ x : ℝ, 3 * x^2 + 24 * x - 92 = x * (x + 15) ∧ ∀ y : ℝ, 3 * y^2 + 24 * y - 92 = y * (y + 15) → x ≤ y :=
begin
  sorry
end

end smallest_x_value_l514_514971


namespace greatest_real_part_z_to_5_l514_514193

theorem greatest_real_part_z_to_5 (
  z_A : ℂ := -3,
  z_B : ℂ := -2 * (sqrt 2) + 2 * (sqrt 2) * I,
  z_C : ℂ := - (sqrt 3) + 2 * I,
  z_D : ℂ := -2 + (sqrt 6) * I,
  z_E : ℂ := -I
) : 
  (∀ z : ℂ, z ∈ {z_A, z_B, z_C, z_D, z_E} → ∃ z_D, ∀ z' ∈ {z_A, z_B, z_C, z_E}, z'.re ^ 5 < z_D.re ^ 5) := 
sorry

end greatest_real_part_z_to_5_l514_514193


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514895

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514895


namespace percentage_repeated_digits_five_digit_numbers_l514_514682

theorem percentage_repeated_digits_five_digit_numbers : 
  let total_five_digit_numbers := 90000
  let non_repeated_digits_number := 9 * 9 * 8 * 7 * 6
  let repeated_digits_number := total_five_digit_numbers - non_repeated_digits_number
  let y := (repeated_digits_number.toFloat / total_five_digit_numbers.toFloat) * 100 
  y = 69.8 :=
by
  sorry

end percentage_repeated_digits_five_digit_numbers_l514_514682


namespace equilateral_triangle_ratio_l514_514914

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514914


namespace min_squares_exceeding_area_l514_514260

theorem min_squares_exceeding_area (n : ℕ) : 
  let a₁ := 25
  let r := 1 / 2
  let S_n := a₁ * 2 * (1 - (r ^ n))
  in S_n > 49 ↔ n ≥ 6 :=
by
  let a₁ := 25
  let r := 1 / 2
  let S_n := a₁ * 2 * (1 - (r ^ n))
  sorry

end min_squares_exceeding_area_l514_514260


namespace pascals_triangle_sum_l514_514853

theorem pascals_triangle_sum : 
  let row := 48 in
  let nth_number := Nat.choose row 46 in
  let third_number := Nat.choose row 2 in
  nth_number = 1128 ∧ (nth_number + third_number) = 2256 := by
  sorry

end pascals_triangle_sum_l514_514853


namespace find_cost_of_apple_l514_514787

theorem find_cost_of_apple (A O : ℝ) 
  (h1 : 6 * A + 3 * O = 1.77) 
  (h2 : 2 * A + 5 * O = 1.27) : 
  A = 0.21 :=
by 
  sorry

end find_cost_of_apple_l514_514787


namespace negation_of_universal_proposition_l514_514822

theorem negation_of_universal_proposition 
  (h : ∀ x : ℝ, 2^x > 0) : (∃ x_0 : ℝ, 2^x_0 ≤ 0) ↔ ¬ (∀ x : ℝ, 2^x > 0):=
by sorry

end negation_of_universal_proposition_l514_514822


namespace classroom_students_l514_514454

theorem classroom_students (n : ℕ) (h1 : 20 < n ∧ n < 30) 
  (h2 : ∃ n_y : ℕ, n = 3 * n_y + 1) 
  (h3 : ∃ n_y' : ℕ, n = (4 * (n - 1)) / 3 + 1) :
  n = 25 := 
by sorry

end classroom_students_l514_514454


namespace sum_of_digits_9x_l514_514002

theorem sum_of_digits_9x (a b c d e : ℕ) (x : ℕ) :
  (1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9) →
  x = 10000 * a + 1000 * b + 100 * c + 10 * d + e →
  (b - a) + (c - b) + (d - c) + (e - d) + (10 - e) = 9 :=
by
  sorry

end sum_of_digits_9x_l514_514002


namespace smallest_n_for_g4_l514_514751

def g (n : ℕ) : ℕ :=
  ((sum (λ a, if a > 0 ∧ (∃ b > 0, a^2 + b^2 = n) then 1 else 0)) -
  ((sum (λ ⟨a, b⟩, if a > 0 ∧ b > 0 ∧ a^2 + b^2 = n then 1 else 0)) / 2)) + 1

theorem smallest_n_for_g4 :
  (∃ n : ℕ, n > 0 ∧ g(n) = 4 ∧ ∀ m : ℕ, m > 0 ∧ m < n → g(m) ≠ 4) ∧
  (∀ n : ℕ, n = 65 → g(n) = 4) :=
begin
  sorry
end

end smallest_n_for_g4_l514_514751


namespace concyclic_MPNQ_l514_514707

open EuclideanGeometry

variables {A B C E F M N P Q H : Point}

-- Given conditions
variables (hABC : Triangle A B C) (hacute: acute_angle_triangle A B C)
          (BE_altitude : Altitude B E A C) (CF_altitude : Altitude C F A B)
          (circle_AB : Circle (midpoint A B) ((dist A B) / 2))
          (circle_AC : Circle (midpoint A C) ((dist A C) / 2))
          (M_on_CF : On_circle M circle_AB) (M_on_CF_line : On_line M C F)
          (N_on_CF : On_circle N circle_AB) (N_on_CF_line : On_line N C F)
          (P_on_BE : On_circle P circle_AC) (P_on_BE_line : On_line P B E)
          (Q_on_BE : On_circle Q circle_AC) (Q_on_BE_line : On_line Q B E)

theorem concyclic_MPNQ :
  Concyclic M P N Q :=
  sorry

end concyclic_MPNQ_l514_514707


namespace equilateral_triangle_ratio_l514_514910

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514910


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514949

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514949


namespace axis_of_symmetry_l514_514679

open Real

theorem axis_of_symmetry (g : ℝ → ℝ) (h : ∀ x, g(x) = g(3 - x)) : ∀ y, y = g 1.5 :=
by
  sorry

end axis_of_symmetry_l514_514679


namespace possible_digits_A_l514_514031

theorem possible_digits_A :
  ∀ (A : ℕ), (18 * 1000 + A * 10 + 4 < 1853) ↔ (A ∈ {0, 1, 2, 3, 4}) := by
  sorry

end possible_digits_A_l514_514031


namespace solution_to_inequality_l514_514043

-- Define the combination function C(n, k)
def combination (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the permutation function A(n, k)
def permutation (n k : ℕ) : ℕ :=
  n.factorial / (n - k).factorial

-- State the final theorem
theorem solution_to_inequality : 
  ∀ x : ℕ, (combination 5 x + permutation x 3 < 30) ↔ (x = 3 ∨ x = 4) :=
by
  -- The actual proof is not required as per the instructions
  sorry

end solution_to_inequality_l514_514043


namespace largest_whole_number_value_l514_514475

theorem largest_whole_number_value (n : ℕ) : 
  (1 : ℚ) / 5 + (n : ℚ) / 8 < 9 / 5 → n ≤ 12 := 
sorry

end largest_whole_number_value_l514_514475


namespace A_min_votes_for_victory_l514_514420

theorem A_min_votes_for_victory:
  ∀ (initial_votes_A initial_votes_B initial_votes_C total_votes remaining_votes min_votes_A: ℕ),
  initial_votes_A = 350 →
  initial_votes_B = 370 →
  initial_votes_C = 280 →
  total_votes = 1500 →
  remaining_votes = 500 →
  min_votes_A = 261 →
  initial_votes_A + min_votes_A > initial_votes_B + (remaining_votes - min_votes_A) :=
by
  intros _ _ _ _ _ _
  sorry

end A_min_votes_for_victory_l514_514420


namespace exists_non_zero_super_integers_with_zero_product_l514_514163

/-
  Definitions based on conditions:
  1. Super-integer is defined formally.
  2. Product of two super-integers gives last n digits of product.
  3. Zero super-integers have all their digits zero.
-/

/-- A super-integer is an infinite sequence of natural numbers (digits) -/
def SuperInteger := ℕ → ℕ

/-- The last n digits of a super-integer, formalized -/
def last_digits (n : ℕ) (si : SuperInteger) : ℕ :=
Nat.mod (si n) (10 ^ n)

/-- The product of two super-integers -/
def super_integer_product (x y : ℕ → ℕ) (n : ℕ) : ℕ :=
last_digits n (fun k => (x k) * (y k))

/-- A zero super-integer has all its digits equal to zero -/
def is_zero_super_integer (si : SuperInteger) : Prop :=
∀ n, si n = 0

/-- The main theorem: existence of two non-zero super-integers with zero product -/
theorem exists_non_zero_super_integers_with_zero_product :
  ∃ (x y : SuperInteger),
    ¬ is_zero_super_integer x ∧
    ¬ is_zero_super_integer y ∧
    is_zero_super_integer (fun n => super_integer_product x y n) :=
begin
  sorry
end

end exists_non_zero_super_integers_with_zero_product_l514_514163


namespace symmetric_points_sum_l514_514339

variable {p q : ℤ}

theorem symmetric_points_sum (h1 : p = -6) (h2 : q = 2) : p + q = -4 := by
  sorry

end symmetric_points_sum_l514_514339


namespace hermione_max_profit_l514_514712

def TC (Q : ℝ) : ℝ := 5 * Q^2

def demand_ws (P : ℝ) : ℝ := 26 - 2 * P
def demand_s (P : ℝ) : ℝ := 10 - P

noncomputable def max_profit : ℝ := 7.69

theorem hermione_max_profit :
  ∃ P Q, (P > 0 ∧ Q > 0) ∧ (Q = demand_ws P + demand_s P) ∧
  (P * Q - TC Q = max_profit) := sorry

end hermione_max_profit_l514_514712


namespace greatest_possible_difference_l514_514316

theorem greatest_possible_difference (x y : ℝ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) :
  ∃ n : ℤ, n = 9 ∧ ∀ x' y' : ℤ, (6 < x' ∧ x' < 10 ∧ 10 < y' ∧ y' < 17) → (y' - x' ≤ n) :=
by {
  -- here goes the actual proof
  sorry
}

end greatest_possible_difference_l514_514316


namespace measure_angle_LOR_l514_514854

noncomputable def angle_LOR (n : ℕ) (L M N O P Q R S : Set Point) (regular_octagon : RegularPolygon) : ℝ := 
  if regular_octagon = RegularPolygon.mk 8 L M N O P Q R S then 135 else 0

theorem measure_angle_LOR (n : ℕ) (L M N O P Q R S : Set Point) (regular_octagon : RegularPolygon) :
  n = 8 → regular_octagon = RegularPolygon.mk 8 L M N O P Q R S → angle_LOR n L M N O P Q R S regular_octagon = 135 :=
by
  sorry

end measure_angle_LOR_l514_514854


namespace find_rate_of_investment_l514_514533

-- Definitions
def total_investment := 22000
def investment_a := 7000
def investment_b := total_investment - investment_a
def interest_total := 3360
def rate_b := 0.14

-- To prove: The rate at which $7000 was invested is 0.18
theorem find_rate_of_investment (r : ℚ) :
  investment_a * r + investment_b * rate_b = interest_total → r = 0.18 :=
by
  unfold total_investment investment_a investment_b interest_total rate_b
  sorry

end find_rate_of_investment_l514_514533


namespace express_y_in_terms_of_x_l514_514808

theorem express_y_in_terms_of_x (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 := 
by { sorry }

end express_y_in_terms_of_x_l514_514808


namespace pups_more_than_adults_l514_514351

-- Define the counts of dogs
def H := 5  -- number of huskies
def P := 2  -- number of pitbulls
def G := 4  -- number of golden retrievers

-- Define the number of pups each type of dog had
def pups_per_husky_and_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky_and_pitbull + additional_pups_per_golden_retriever

-- Calculate the total number of pups
def total_pups := H * pups_per_husky_and_pitbull + P * pups_per_husky_and_pitbull + G * pups_per_golden_retriever

-- Calculate the total number of adult dogs
def total_adult_dogs := H + P + G

-- Prove that the number of pups is 30 more than the number of adult dogs
theorem pups_more_than_adults : total_pups - total_adult_dogs = 30 :=
by
  -- fill in the proof later
  sorry

end pups_more_than_adults_l514_514351


namespace find_m_and_other_root_l514_514277

theorem find_m_and_other_root (m : ℝ) (r : ℝ) :
    (∃ x : ℝ, x^2 + m*x - 2 = 0) ∧ (x = -1) → (m = -1 ∧ r = 2) :=
by
  sorry

end find_m_and_other_root_l514_514277


namespace fn_simplified_l514_514827

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
| 0       := 0
| 1       := 1
| (n + 1) :=
  if h : n + 1 ≥ 3 then
    let m := n in
    0.5 * (m + 1) * a m + 0.5 * (m + 1) * m * a (m - 1) + (-1)^(m + 1) * (1 - 0.5 * (m + 1))
  else
    if n = 0 then 0 else 1

-- Define the combinatorial function binom
def binom (n k : ℕ) : ℝ := (nat.factorial n / (nat.factorial k * nat.factorial (n - k)))

-- Define the function f_n
noncomputable def f (n : ℕ) : ℝ :=
a n + (list.range n).sum (λ k, (k + 1) * binom n k * a (n - (k + 1)))

-- Define the statement to prove that f_n equals 2 * n! - (n + 1)
theorem fn_simplified (n : ℕ) : f n = 2 * (nat.factorial n) - (n + 1) := sorry

end fn_simplified_l514_514827


namespace lisa_speed_l514_514767

-- Define conditions
def distance : ℕ := 256
def time : ℕ := 8

-- Define the speed calculation theorem
theorem lisa_speed : (distance / time) = 32 := 
by {
  sorry
}

end lisa_speed_l514_514767


namespace golden_triangle_expression_l514_514459

noncomputable def t : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_expression :
  t = (Real.sqrt 5 - 1) / 2 →
  (1 - 2 * (Real.sin (27 * Real.pi / 180))^2) / (2 * t * Real.sqrt (4 - t^2)) = 1 / 4 :=
by
  intro h_t
  have h1 : t = (Real.sqrt 5 - 1) / 2 := h_t
  sorry

end golden_triangle_expression_l514_514459


namespace monotonic_implies_existence_of_increasing_pair_l514_514306

variables {R : Type*} [linear_ordered_field R]

def monotonically_increasing (f : R → R) : Prop :=
∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

noncomputable def statementB (f : R → R) : Prop :=
∃ x₁ x₂, x₁ < x₂ ∧ f x₁ < f x₂

theorem monotonic_implies_existence_of_increasing_pair (f : R → R) :
  monotonically_increasing f → statementB f :=
by
  intros h
  use [0, 1]
  specialize h 0 1
  sorry  -- The remaining proof steps are skipped since they are not required.

end monotonic_implies_existence_of_increasing_pair_l514_514306


namespace fifth_student_guess_l514_514027

theorem fifth_student_guess (s1 s2 s3 s4 s5 : ℕ) 
(h1 : s1 = 100)
(h2 : s2 = 8 * s1)
(h3 : s3 = s2 - 200)
(h4 : s4 = (s1 + s2 + s3) / 3 + 25)
(h5 : s5 = s4 + s4 / 5) : 
s5 = 630 :=
sorry

end fifth_student_guess_l514_514027


namespace symmetric_point_origin_l514_514803

-- Define the point P
structure Point3D where
  x : Int
  y : Int
  z : Int

def P : Point3D := { x := 1, y := 3, z := -5 }

-- Define the symmetric function w.r.t. the origin
def symmetric_with_origin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Define the expected result
def Q : Point3D := { x := -1, y := -3, z := 5 }

-- The theorem to prove
theorem symmetric_point_origin : symmetric_with_origin P = Q := by
  sorry

end symmetric_point_origin_l514_514803


namespace card_draw_prob_l514_514843

theorem card_draw_prob (h_deck : fin 52 → Card) (h_face : ∀ c : Card, c ∈ (Jack, Queen, King)) (h_hearts : ∀ c : Card, c.suit = Heart) (h_tens : ∀ c : Card, c.rank = 10):
  probability
    (λ s : fin 3 → Card, s 0 ∈ h_face ∧ s 1 ∈ h_hearts ∧ s 2 ∈ h_tens)
    (uniform_fun (fin 3) h_deck) = 1 / 217 :=
by
  sorry

end card_draw_prob_l514_514843


namespace find_x_for_sum_condition_l514_514452

theorem find_x_for_sum_condition :
  ∃ x : ℕ, x > 0 ∧ (∀ n < 15, ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = 2 * a + x * b) ∧ (∀ x', x = 5) :=
by
  use 5
  split
  · norm_num
  · intro n hn
    use_nat_cases hn; sorry
  · intro x2
    exact dec_trivial
)

end find_x_for_sum_condition_l514_514452


namespace symmetric_about_origin_unique_l514_514526

def f1 (x : ℝ) : ℝ := -abs (sin x)
def f2 (x : ℝ) : ℝ := -x * sin (abs x)
def f3 (x : ℝ) : ℝ := sin (-abs x)
def f4 (x : ℝ) : ℝ := sin (abs x)

theorem symmetric_about_origin_unique :
  (∀ x : ℝ, f1 (-x) = f1 x) ∧
  (∀ x : ℝ, f2 (-x) = -f2 x) ∧
  (∀ x : ℝ, f3 (-x) = f3 x) ∧
  (∀ x : ℝ, f4 (-x) = f4 x) → 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f (-x) = -f x) ∧ f = f2) :=
sorry

end symmetric_about_origin_unique_l514_514526


namespace carlton_bananas_l514_514165

theorem carlton_bananas : 
  ∃ b : ℕ, (∑ i in finset.range 7, b + 5 * i) = 189 ∧ (b + 30) = 42 :=
by 
  -- Skipping the proof for now
  sorry

end carlton_bananas_l514_514165


namespace sum_ends_in_zero_squares_end_same_digit_l514_514779

theorem sum_ends_in_zero_squares_end_same_digit (a b : ℕ) (h : (a + b) % 10 = 0) : (a^2 % 10) = (b^2 % 10) := 
sorry

end sum_ends_in_zero_squares_end_same_digit_l514_514779


namespace expression_evaluation_l514_514156

theorem expression_evaluation :
  10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 :=
by
  sorry

end expression_evaluation_l514_514156


namespace polynomial_roots_product_l514_514012

noncomputable def roots := {z | ∃ (θ φ : ℝ), z = 2 * complex.exp (complex.I * θ) ∨ z = 2 * complex.exp (-complex.I * θ) ∨ z = 2 * complex.exp (complex.I * φ) ∨ z = 2 * complex.exp (-complex.I * φ)}

theorem polynomial_roots_product (p q r s : ℝ) (h : ∀ z ∈ roots, z^4 + p * z^3 + q * z^2 + r * z + s = 0) :
  s = 16 :=
by
  sorry

end polynomial_roots_product_l514_514012


namespace chord_length_square_l514_514173

theorem chord_length_square {r₁ r₂ R : ℝ} (r1_eq_4 : r₁ = 4) (r2_eq_8 : r₂ = 8) 
  (R_eq_12 : R = 12)
  (externally_tangent : tangent r₁ r₂)
  (internally_tangent_1 : tangent_internally r₁ R)
  (internally_tangent_2 : tangent_internally r₂ R)
  : exists (PQ : ℝ), PQ^2 = 3584 / 9 :=
by sorry

end chord_length_square_l514_514173


namespace range_of_a_l514_514402

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → x + (4 / x) - 1 - a^2 + 2 * a > 0) : -1 < a ∧ a < 3 :=
sorry

end range_of_a_l514_514402


namespace smallest_integer_value_of_n_l514_514706

-- Definition of the conditions
def players (n : ℕ) := 4 * n
def matches (n : ℕ) := (4 * n) * (4 * n - 1) / 2
def matches_won_by_women (x : ℕ) := 4 * x
def matches_won_by_men (x : ℕ) := 11 * x

theorem smallest_integer_value_of_n :
  ∃ n : ℕ, ∃ x : ℕ, 2 * (matches n) = 30 * x ∧ 15 * x = matches (n+1) ∧ n = 4 :=
by
  sorry

end smallest_integer_value_of_n_l514_514706


namespace find_smaller_number_l514_514045

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end find_smaller_number_l514_514045


namespace kitchen_percentage_longer_l514_514846

open nat

theorem kitchen_percentage_longer
  (bedrooms_time : ℕ)
  (kitchen_time : ℕ)
  (livingroom_time : ℕ)
  (total_time : ℕ)
  (percentage_longer : ℕ) : 
  bedrooms_time = 3 * 4 ∧ 
  livingroom_time = 2 * (bedrooms_time + kitchen_time) ∧ 
  total_time = bedrooms_time + kitchen_time + livingroom_time ∧ 
  total_time = 54 →
  percentage_longer = 50 :=
begin
  intros h,
  sorry
end

end kitchen_percentage_longer_l514_514846


namespace triangle_perimeter_inscribed_circle_l514_514817

theorem triangle_perimeter_inscribed_circle
  (radius : ℝ)
  (DP PE : ℝ) :
  radius = 15 → DP = 19 → PE = 31 → 
  2 * (50 + (DP + PE + (2 * radius - DP - PE))) = (10075 : ℝ) / 91 :=
by
  intro h_radius h_DP h_PE
  sorry -- proof omitted

-- Definitions of provided values
def radius : ℝ := 15
def DP : ℝ := 19
def PE : ℝ := 31

-- Proves the theorem with the given conditions
#reduce triangle_perimeter_inscribed_circle 15 19 31 rfl rfl rfl

end triangle_perimeter_inscribed_circle_l514_514817


namespace number_of_ways_to_choose_one_top_and_one_bottom_l514_514144

theorem number_of_ways_to_choose_one_top_and_one_bottom :
  let number_of_hoodies := 5
  let number_of_sweatshirts := 4
  let number_of_jeans := 3
  let number_of_slacks := 5
  let total_tops := number_of_hoodies + number_of_sweatshirts
  let total_bottoms := number_of_jeans + number_of_slacks
  total_tops * total_bottoms = 72 := 
by
  sorry

end number_of_ways_to_choose_one_top_and_one_bottom_l514_514144


namespace scienceStudyTime_l514_514563

def totalStudyTime : ℕ := 60
def mathStudyTime : ℕ := 35

theorem scienceStudyTime : totalStudyTime - mathStudyTime = 25 :=
by sorry

end scienceStudyTime_l514_514563


namespace missing_condition_l514_514997

theorem missing_condition (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : y = 3 * (x - 2)) :
  true := -- The equivalent mathematical statement asserts the correct missing condition.
sorry

end missing_condition_l514_514997


namespace equilateral_triangle_ratio_l514_514863

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514863


namespace smallest_n_for_g_n_eq_4_l514_514749

/-- 
  Let g(n) be the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n.
  Prove that the smallest positive integer n for which g(n) = 4 is 25.
-/
def g (n : ℕ) : ℕ :=
  (finset.univ.product finset.univ).filter (λ (ab : ℕ × ℕ), ab.1 ^ 2 + ab.2 ^ 2 = n ∧ ab.1 ≠ ab.2).card

theorem smallest_n_for_g_n_eq_4 :
  ∃ n : ℕ, g n = 4 ∧ (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 25
  sorry

end smallest_n_for_g_n_eq_4_l514_514749


namespace determine_redox_potential_l514_514994

-- Definitions based on conditions
def E0 : ℝ := 1.51
def n : ℕ := 5
def Cox : ℝ := 2 * 10^(-6)
def Cred : ℝ := 10^(-2)

-- Statement of the theorem
theorem determine_redox_potential : 
  let E := E0 + (0.059 / n) * Real.log10 (Cox / Cred)
  in E ≈ 1.46 :=
sorry

end determine_redox_potential_l514_514994


namespace coloring_count_is_2_l514_514813

noncomputable def count_colorings (initial_color : String) : Nat := 
  if initial_color = "R" then 2 else 0 -- Assumes only the case of initial red color is valid for simplicity

theorem coloring_count_is_2 (h1 : True) (h2 : True) (h3 : True) (h4 : True):
  count_colorings "R" = 2 := by
  sorry

end coloring_count_is_2_l514_514813


namespace remaining_movies_to_watch_l514_514054

theorem remaining_movies_to_watch (total_movies watched_movies remaining_movies : ℕ) 
  (h1 : total_movies = 8) 
  (h2 : watched_movies = 4) 
  (h3 : remaining_movies = total_movies - watched_movies) : 
  remaining_movies = 4 := 
by
  sorry

end remaining_movies_to_watch_l514_514054


namespace find_a_for_symmetric_circle_l514_514283

theorem find_a_for_symmetric_circle :
  ∃ a : ℝ, (∀ x y : ℝ, (x + 3)^2 + (y + a)^2 = 25 → x - y + 1 = 0 → a = 2) :=
begin
  sorry
end

end find_a_for_symmetric_circle_l514_514283


namespace max_five_digit_divisible_by_eleven_l514_514112

def isDivisibleByEleven (n : ℕ) : Prop := n % 11 = 0

def max_value : ℕ := 96569

theorem max_five_digit_divisible_by_eleven : ∃ (A B D : ℕ), 
  A ≠ B ∧ A ≠ D ∧ B ≠ D ∧ 
  0 ≤ A ∧ A ≤ 9 ∧ 
  0 ≤ B ∧ B ≤ 9 ∧ 
  0 ≤ D ∧ D ≤ 9 ∧ 
  let n := 10000 * A + 1000 * B + 100 * D + 10 * B + A in 
  isDivisibleByEleven n ∧ 
  n = max_value :=
by
  sorry

end max_five_digit_divisible_by_eleven_l514_514112


namespace num_repeating_decimals_between_1_and_20_l514_514591

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end num_repeating_decimals_between_1_and_20_l514_514591


namespace smallest_possible_b_l514_514783

noncomputable def smallest_b (a b : ℝ) : Prop :=
  (2 < a) ∧ (a < b) ∧
  ∀ x y z : ℝ, (x = 2 ∧ y = a ∧ z = b → ¬(x + y > z ∧ x + z > y ∧ y + z > x)) ∧
               (x = 1/b ∧ y = 1/a ∧ z = 1/2 → ¬(x + y > z ∧ x + z > y ∧ y + z > x))

theorem smallest_possible_b (a b : ℝ) (h : smallest_b a b) : b = 3 + sqrt 5 :=
sorry

end smallest_possible_b_l514_514783


namespace bc_fraction_of_ad_l514_514397

theorem bc_fraction_of_ad
  {A B D C : Type}
  (length_AB length_BD length_AC length_CD length_AD length_BC : ℝ)
  (h1 : length_AB = 3 * length_BD)
  (h2 : length_AC = 4 * length_CD)
  (h3 : length_AD = length_AB + length_BD + length_CD)
  (h4 : length_BC = length_AC - length_AB) :
  length_BC / length_AD = 5 / 6 :=
by sorry

end bc_fraction_of_ad_l514_514397


namespace equilateral_triangle_ratio_l514_514904

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514904


namespace shaded_percentage_of_grid_l514_514975

def percent_shaded (total_squares shaded_squares : ℕ) : ℚ :=
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100

theorem shaded_percentage_of_grid :
  percent_shaded 36 16 = 44.44 :=
by 
  sorry

end shaded_percentage_of_grid_l514_514975


namespace value_of_f_at_2019_l514_514500

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_positive : ∀ x : ℝ, f x > 0)
variable (h_functional : ∀ x : ℝ, f (x + 2) = 1 / (f x))

theorem value_of_f_at_2019 : f 2019 = 1 :=
by
  sorry

end value_of_f_at_2019_l514_514500


namespace probability_at_least_two_red_balls_l514_514794

noncomputable def prob_red_balls (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (drawn_balls : ℕ) : ℚ :=
if total_balls = 6 ∧ red_balls = 3 ∧ white_balls = 2 ∧ black_balls = 1 ∧ drawn_balls = 3 then
  1 / 2
else
  0

theorem probability_at_least_two_red_balls :
  prob_red_balls 6 3 2 1 3 = 1 / 2 :=
by 
  sorry

end probability_at_least_two_red_balls_l514_514794


namespace equilateral_triangle_ratio_l514_514928

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514928


namespace find_divisor_l514_514772

theorem find_divisor :
  ∃ x : ℤ, 95 = 6 * x + 5 ∧ x = 15 :=
by
  use 15
  split
  · sorry
  · sorry

end find_divisor_l514_514772


namespace simplified_expr_correct_value_at_specific_x_l514_514984

-- Define the function to simplify the expression
def simplify_expr (x : ℝ) (hx : x ≠ -1 ∧ x ≠ 1) : ℝ :=
  (3 * x + 2) / (x + 1) - 2

-- Define the original expression as a function
def original_expr (x : ℝ) (hx : x ≠ -1 ∧ x ≠ 1) : ℝ :=
  (simplify_expr x hx) / (x / (x^2 - 1))

-- Prove that the simplified expression is equivalent to x - 1
theorem simplified_expr_correct (x : ℝ) (hx : x ≠ -1 ∧ x ≠ 1) :
  original_expr x hx = x - 1 :=
by sorry

-- Given a specific value of x
def specific_x : ℝ := (real.sqrt 16 - (1 / 4)⁻¹ - (real.pi - 3)^0)

-- Calculate the value of the original expression when x = specific_x
theorem value_at_specific_x : original_expr specific_x sorry = -2 :=
by sorry

end simplified_expr_correct_value_at_specific_x_l514_514984


namespace find_angle_difference_l514_514276

   variables {A B : ℝ}

   -- Define the vectors a and b based on the angles A and B
   def vector_a : ℝ × ℝ := (2 * Real.cos A, 2 * Real.sin A)
   def vector_b : ℝ × ℝ := (3 * Real.cos B, 3 * Real.sin B)

   -- Define the dot product of vectors a and b
   def dot_product (u v : ℝ × ℝ) : ℝ :=
     u.fst * v.fst + u.snd * v.snd

   -- Define the magnitudes of vectors a and b
   def magnitude (v : ℝ × ℝ) : ℝ :=
     Real.sqrt (v.fst * v.fst + v.snd * v.snd)

   -- Define the cosine of the angle between vectors a and b
   def cosine_angle (u v : ℝ × ℝ) : ℝ :=
     dot_product u v / (magnitude u * magnitude v)

   -- Given condition that the angle between vector_a and vector_b is π/3
   def given_condition : Prop :=
     cosine_angle vector_a vector_b = Real.cos (π / 3)

   -- Theorem to prove
   theorem find_angle_difference (h : given_condition) : A - B = π / 3 ∨ A - B = -π / 3 :=
   by
     sorry
   
end find_angle_difference_l514_514276


namespace alberto_travel_more_miles_l514_514134

def v_A1 : ℝ := 18 -- Initial speed of Alberto
def v_B : ℝ := 15 -- Speed of Bjorn
def t1 : ℝ := 2 -- Hours before Alberto changes speed
def v_A2 : ℝ := 22 -- Speed of Alberto after 2 hours
def t_total : ℝ := 5 -- Total time of the race
def t2 : ℝ := t_total - t1 -- Remaining time for Alberto's changed speed

theorem alberto_travel_more_miles :
  let distance_B := v_B * t_total in
  let distance_A := (v_A1 * t1) + (v_A2 * t2) in
  (distance_A - distance_B = 27) :=
by
  let distance_B := v_B * t_total
  let distance_A := (v_A1 * t1) + (v_A2 * t2)
  have h1 : distance_B = 75 := by sorry
  have h2 : distance_A = 102 := by sorry
  calc
    distance_A - distance_B = 102 - 75 := by rw [h1, h2]
                       ... = 27 := by norm_num

end alberto_travel_more_miles_l514_514134


namespace dot_product_triangle_l514_514278

theorem dot_product_triangle (A B C : ℝ^3) (h_area : 1/2 * (|A - B| * |A - C| * real.sin (real.pi / 3)) = 2 * real.sqrt 3) :
  (A - B) ∙ (A - C) = 4 :=
sorry

end dot_product_triangle_l514_514278


namespace num_non_self_intersecting_chains_l514_514457

/-- The number of non-closed, non-self-intersecting 9-sided polygonal chains 
that can be formed with 10 points marked on a circle as vertices is 1280. -/
theorem num_non_self_intersecting_chains (n : ℕ) (h : n = 10) : 
  let chains := 10 * 2^8 / 2 in chains = 1280 := 
by
  sorry

end num_non_self_intersecting_chains_l514_514457


namespace minimum_value_of_f_value_of_f_when_n_is_special_exists_n_such_that_f_n_is_2012_l514_514586

/-- Sum of the digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The function f(n) as defined in the problem -/
def f (n : ℕ) : ℕ := sum_of_digits (3 * n^2 + n + 1)

theorem minimum_value_of_f :
  ∃ n : ℕ, n > 0 ∧ f(n) = 3 :=
sorry

theorem value_of_f_when_n_is_special (k : ℕ) (hk : k > 0) :
  f (2 * 10^k - 1) = 9 * k - 4 :=
sorry

theorem exists_n_such_that_f_n_is_2012 :
  ∃ n : ℕ, n > 0 ∧ f(n) = 2012 :=
sorry

end minimum_value_of_f_value_of_f_when_n_is_special_exists_n_such_that_f_n_is_2012_l514_514586


namespace Q_root_l514_514208

def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem Q_root : Q (3^(1 / 3 : ℝ) + 2) = 0 := sorry

end Q_root_l514_514208


namespace find_sum_f_neg1_f_3_l514_514639

noncomputable def f : ℝ → ℝ := sorry

-- condition: odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f x

-- condition: symmetry around x=1
def symmetric_around_one (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (1 - x) = f (1 + x)

-- condition: specific value at x=1
def value_at_one (f : ℝ → ℝ) : Prop := f 1 = 2

-- Theorem to prove
theorem find_sum_f_neg1_f_3 (h1 : odd_function f) (h2 : symmetric_around_one f) (h3 : value_at_one f) : f (-1) + f 3 = -4 := by
  sorry

end find_sum_f_neg1_f_3_l514_514639


namespace scientific_notation_chip_gate_width_l514_514838

theorem scientific_notation_chip_gate_width :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end scientific_notation_chip_gate_width_l514_514838


namespace visitors_current_day_l514_514524

-- Define the number of visitors on the previous day and the additional visitors
def v_prev : ℕ := 600
def v_add : ℕ := 61

-- Prove that the number of visitors on the current day is 661
theorem visitors_current_day : v_prev + v_add = 661 :=
by
  sorry

end visitors_current_day_l514_514524


namespace scientific_notation_chip_gate_width_l514_514837

theorem scientific_notation_chip_gate_width :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end scientific_notation_chip_gate_width_l514_514837


namespace find_a_intervals_monotonicity_range_b_l514_514624

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * real.log (1 + x) + x^2 - 10 * x

-- 1. Prove the value of the real number a
theorem find_a (h : ∀ x, deriv (f a) = (a / (1 + x)) + 2 * x - 10) : a = 16 :=
sorry

-- 2. Determine the intervals of monotonicity for the function f(x)
theorem intervals_monotonicity (a : ℝ) (x : ℝ) (h : a = 16) :
  (∀ x, differentiable at (f a) x) ∧
  (f a).deriv x > 0 ∨ (∀ x, f a).deriv x < 0 :=
sorry

-- 3. If the line y = b intersects the graph of y = f(x) at three points, find the range of values for b
theorem range_b (a : ℝ) (b : ℝ) (h : a = 16)
  (local_max : ∃ x, f a 1 = 16 * real.log 2 - 9)
  (local_min : ∃ x, f a 3 = 32 * real.log 2 - 21) :
  32 * real.log 2 - 21 < b ∧ b < 16 * real.log 2 - 9 :=
sorry

end find_a_intervals_monotonicity_range_b_l514_514624


namespace normal_distribution_problem_l514_514284

noncomputable def normal_probability_condition (X : ℝ → ℝ) (σ : ℝ) : Prop :=
  (∀ x, X(x) ∼ Normal 0 (σ^2)) ∧ ℙ(X ∈ Set.Icc (-2 : ℝ) 0) = 0.4

theorem normal_distribution_problem (X : ℝ → ℝ) (σ : ℝ) (h : normal_probability_condition X σ) :
  ℙ(X > 2) = 0.1 :=
sorry

end normal_distribution_problem_l514_514284


namespace distinct_meals_count_l514_514146

theorem distinct_meals_count :
  let appetizers := ["Fries", "Salad", "Onion Rings"]
  let main_courses := ["Burger", "Chicken Wrap", "Pizza"]
  let drinks := ["Soda", "Lemonade", "Tea"]
  let desserts := ["Ice Cream", "Cake"]
  let fries_available_drinks := ["Lemonade", "Tea"]
  let main_course_count := 3
  let dessert_count := 2

  -- Meals with Fries (without Soda)
  let meals_with_fries := 1 * main_course_count * 2 * dessert_count
  -- Meals without Fries (Salad or Onion Rings with all drinks)
  let meals_without_fries := 2 * main_course_count * 3 * dessert_count

  -- Total meals
  meals_with_fries + meals_without_fries = 48 :=
by
  let appetizers := ["Fries", "Salad", "Onion Rings"]
  let main_courses := ["Burger", "Chicken Wrap", "Pizza"]
  let drinks := ["Soda", "Lemonade", "Tea"]
  let desserts := ["Ice Cream", "Cake"]
  let fries_available_drinks := ["Lemonade", "Tea"]
  let main_course_count := 3
  let dessert_count := 2

  let meals_with_fries := 1 * main_course_count * 2 * dessert_count
  let meals_without_fries := 2 * main_course_count * 3 * dessert_count
  
  have h1 : meals_with_fries = 12 := by rfl
  have h2 : meals_without_fries = 36 := by rfl
  
  have h3 : meals_with_fries + meals_without_fries = 48 := by
    calc
    12 + 36 = 48 : by norm_num
  sorry

end distinct_meals_count_l514_514146


namespace square_of_chord_l514_514169

-- Definitions of the circles and tangency conditions
def radius1 := 4
def radius2 := 8
def radius3 := 12

-- The internal and external tangents condition
def externally_tangent (r1 r2 : ℕ) : Prop := ∃ O1 O2, dist O1 O2 = r1 + r2
def internally_tangent (r_in r_out : ℕ) : Prop := ∃ O_in O_out, dist O_in O_out = r_out - r_in

theorem square_of_chord :
  externally_tangent radius1 radius2 ∧ 
  internally_tangent radius1 radius3 ∧
  internally_tangent radius2 radius3 →
  (∃ PQ : ℚ, PQ^2 = 3584 / 9) :=
by
  intros h
  sorry

end square_of_chord_l514_514169


namespace min_value_l514_514640

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) : 
  ∃ m, m = (1 / x + 1 / y) ∧ m = 9 :=
by
  sorry

end min_value_l514_514640


namespace find_d_l514_514233

theorem find_d (x y d : ℕ) (h_midpoint : (1 + 5)/2 = 3 ∧ (3 + 11)/2 = 7) 
  : x + y = d ↔ d = 10 := 
sorry

end find_d_l514_514233


namespace part_I_part_II_l514_514828

open Real

noncomputable def a_seq (n : ℕ) : ℝ := sorry

def b_seq (n : ℕ) : ℝ :=
  (∑ k in Finset.range (n + 1), (1 - a_seq k / a_seq (k + 1)) / sqrt (a_seq (k + 1)))

theorem part_I (n : ℕ) : 0 ≤ b_seq n ∧ b_seq n < 2 := sorry

theorem part_II (c : ℝ) (h : 0 ≤ c ∧ c < 2) : ∃ (a_seq : ℕ → ℝ), (1 = a_seq 0 ∧ ∀ n, a_seq n ≤ a_seq (n + 1)) ∧ ∃ᶠ n in at_top, b_seq n > c := sorry

end part_I_part_II_l514_514828


namespace olympic_photo_arrangement_l514_514558

theorem olympic_photo_arrangement : 
  ∃ (A B C D E : Type), (5 = 1 + 1 + 3) →
  (∀ (l r : Type), (l = A ∨ l = B) ∧ r ≠ A) →
  (number_of_arrangements l r = 42) :=
by
  sorry

end olympic_photo_arrangement_l514_514558


namespace find_perimeter_eq_135_point_57_l514_514849

variables (P Q R : Type) [euclidean_space P] [euclidean_space Q] [euclidean_space R]
variables (PQ QR PR m_P m_Q m_R : ℝ)
variables (triangle_PQR : triangle P Q R)
variables (length_m_P length_m_Q length_m_R : ℝ)
variable (perimeter_triangle : ℝ)

def triangle_similar_segments (PQ QR PR m_P m_Q m_R : ℝ) : Prop :=
  PQ = 150 ∧ QR = 270 ∧ PR = 210 ∧ m_P ∥ QR ∧ m_Q ∥ PR ∧ m_R ∥ PQ ∧
  m_P = 60 ∧ m_Q = 35 ∧ m_R = 25

def expected_perimeter (perimeter_triangle : ℝ) : Prop :=
  perimeter_triangle = 135.57

theorem find_perimeter_eq_135_point_57 
  (h : triangle_similar_segments 150 270 210 60 35 25) : expected_perimeter 135.57 :=
sorry

end find_perimeter_eq_135_point_57_l514_514849


namespace midpoint_XY_equidistant_l514_514331

variables {A B C H M X Y F : Type}
variables (triangle_ABC : acute_triangle A B C)
variables (altitude_AH : altitude A H)
variables (median_AM : median A M)
variables (on_AB_X : on_segment A B X)
variables (on_AC_Y : on_segment A C Y)
variables (equidistant_AX_XC : AX = XC)
variables (equidistant_AY_YB : AY = YB)

theorem midpoint_XY_equidistant (midpoint_XY : midpoint X Y F) :
  dist F H = dist F M := 
sorry

end midpoint_XY_equidistant_l514_514331


namespace solve_equation_l514_514230

noncomputable def smallest_solution : ℝ :=
(15 - Real.sqrt 549) / 6

theorem solve_equation :
  ∃ x : ℝ, 
    (3 * x / (x - 3) + (3 * x^2 - 27) / x = 18) ∧
    x = smallest_solution :=
by
  sorry

end solve_equation_l514_514230


namespace equilateral_triangle_ratio_l514_514927

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514927


namespace problem_statement_l514_514033

-- Define the function f with the given properties
variable {f : ℝ → ℝ}

-- Conditions
axiom condition1 : ∀ x : ℝ, f(x + 2) = f(-x)
axiom condition2 : ∀ x : ℝ, f(x - 1) = f(-x)
axiom condition3 : f(1) = 4

-- The statement to prove
theorem problem_statement : f(2016) + f(2017) + f(2018) = 4 :=
by
  sorry

end problem_statement_l514_514033


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514889

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514889


namespace diagonal_of_square_plot_l514_514992

/-- Let a rectangular field have a length of 90 meters and a breadth of 80 meters. 
The area of a square plot is equal to the area of this rectangular field. 
We need to prove that the length of the diagonal of the square plot is 120 meters. -/
theorem diagonal_of_square_plot (length_rectangle : ℝ) (breadth_rectangle : ℝ) 
  (side_square : ℝ) (diagonal_square : ℝ) 
  (h1 : length_rectangle = 90) 
  (h2 : breadth_rectangle = 80)
  (h3 : length_rectangle * breadth_rectangle = side_square * side_square) 
  (h4 : diagonal_square = real.sqrt (2 * (side_square * side_square))) : 
  diagonal_square = 120 := 
sorry

end diagonal_of_square_plot_l514_514992


namespace three_digit_non_multiples_4_9_l514_514311

theorem three_digit_non_multiples_4_9 :
  let total_three_digit := 999 - 100 + 1,
      multiples_4 := 249 - 25 + 1,
      multiples_9 := 111 - 12 + 1,
      multiples_36 := 27 - 3 + 1,
      multiples_union := multiples_4 + multiples_9 - multiples_36 in
  total_three_digit - multiples_union = 600 :=
by
  sorry

end three_digit_non_multiples_4_9_l514_514311


namespace tv_program_reform_analysis_l514_514062

-- Define the conditions based on the given problem
def surveyed_before_after_reform := ∀ n, n = 100
def given_K_squared_approx_zero_nine_nine := K^2 ≈ 0.99

-- State the theorem to be proved
theorem tv_program_reform_analysis (surveyed_before_after_reform : ∀ n, n = 100)
                                   (given_K_squared_approx_zero_nine_nine : K^2 ≈ 0.99) :
  ¬ (∃ evidence : Prop, evidence) :=
sorry

end tv_program_reform_analysis_l514_514062


namespace limit_of_integral_l514_514622

open Real topological_space

noncomputable def integrand (α β : ℝ) (λ x : ℝ) : ℝ := (β * x + α * (1 - x)) ^ λ

theorem limit_of_integral (α β : ℝ) (h_cond : 0 < α ∧ α < β) :
  (λ → 0) (λ (λ → real) ((λ λ : ℝ) → ∫ (x : ℝ) in 0..1, integrand α β λ x)^(1/λ))
      = e^((β * log β - α * log α)/(β - α)) :=
by
  sorry

end limit_of_integral_l514_514622


namespace percent_neither_condition_l514_514123

namespace TeachersSurvey

variables (Total HighBloodPressure HeartTrouble Both: ℕ)

theorem percent_neither_condition :
  Total = 150 → HighBloodPressure = 90 → HeartTrouble = 50 → Both = 30 →
  (HighBloodPressure + HeartTrouble - Both) = 110 →
  ((Total - (HighBloodPressure + HeartTrouble - Both)) * 100 / Total) = 2667 / 100 :=
by
  intros hTotal hBP hHT hBoth hUnion
  sorry

end TeachersSurvey

end percent_neither_condition_l514_514123


namespace conditional_probability_second_sci_given_first_sci_l514_514705

-- Definitions based on the conditions
def total_questions : ℕ := 6
def science_questions : ℕ := 4
def humanities_questions : ℕ := 2
def first_draw_is_science : Prop := true

-- The statement we want to prove
theorem conditional_probability_second_sci_given_first_sci : 
    first_draw_is_science → (science_questions - 1) / (total_questions - 1) = 3 / 5 := 
by
  intro h
  have num_sci_after_first : ℕ := science_questions - 1
  have total_after_first : ℕ := total_questions - 1
  have prob_second_sci := num_sci_after_first / total_after_first
  sorry

end conditional_probability_second_sci_given_first_sci_l514_514705


namespace geometric_sequence_value_l514_514634

theorem geometric_sequence_value (χ : ℝ) : 
  (-1, χ, -4) form a geometric sequence → (χ = 2 ∨ χ = -2) := 
by
  sorry

end geometric_sequence_value_l514_514634


namespace pyramid_total_triangular_face_area_l514_514078

-- Define the problem parameters
def base_edge_length : ℝ := 10
def lateral_edge_length : ℝ := 13

-- Problem statement
theorem pyramid_total_triangular_face_area : 
  let half_base := base_edge_length / 2 in
  let height_of_triangle :=
    Real.sqrt (lateral_edge_length^2 - half_base^2) in
  let area_of_one_triangle := (1 / 2) * base_edge_length * height_of_triangle in
  4 * area_of_one_triangle = 240 :=
by
  sorry

end pyramid_total_triangular_face_area_l514_514078


namespace equilateral_triangle_ratio_l514_514932

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514932


namespace lower_right_square_is_2_l514_514060

def valid_grid (grid : Matrix (Fin 5) (Fin 5) ℕ) : Prop :=
  (∀ i, (List.nodup (Fin.toList (fun j => grid i j))) ∧
        List.sum (Fin.toList (fun j => grid 0 j)) = 15) ∧
  (∀ j, (List.nodup (Fin.toList (fun i => grid i j))))

def partial_grid : Matrix (Fin 5) (Fin 5) ℕ :=
  ![[1, 0, 3, 4, 0],
    [5, 0, 1, 0, 3],
    [0, 4, 0, 5, 0],
    [4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]]

theorem lower_right_square_is_2 : ∃ (grid : Matrix (Fin 5) (Fin 5) ℕ),
  valid_grid grid ∧ grid = partial_grid ⟹ grid (Fin.val 4) (Fin.val 4) = 2 :=
begin
  sorry
end

end lower_right_square_is_2_l514_514060


namespace find_y_given_conditions_l514_514021

theorem find_y_given_conditions :
  (length width perimeter: ℕ) (h1: length = 10) (h2: width = 24) 
  (new_length new_width: ℕ) (h3: 2 * new_length + 2 * new_width = 52)
  (h4: new_length * new_width = length * width) (y: ℕ) :
  y = new_length / 4 → y = 5 :=
by
  sorry

end find_y_given_conditions_l514_514021


namespace polynomial_count_l514_514551

theorem polynomial_count :
  let n := 2 in
  ∃ (a_0 a_1 a_2 : ℤ), abs a_0 + abs a_1 + abs a_2 + 2 * n = 8 ∧
  (has_count {x : ℤ × ℤ × ℤ | (abs x.1 + abs x.2 + abs (x.2) + 2 * n = 8)} 12) :=
begin
  -- dummy proof to make sure lean code builds successfully
  sorry
end

end polynomial_count_l514_514551


namespace stationary_salmon_oxygen_consumption_l514_514423

theorem stationary_salmon_oxygen_consumption (x : ℝ) (v : ℝ) : 
  (v = 0) ∧ (v = (1/2) * log 3 ((x / 100) * real.pi)) → 
  x = 100 / real.pi :=
by
  intros h
  sorry

end stationary_salmon_oxygen_consumption_l514_514423


namespace father_age_38_l514_514107

variable (F S : ℕ)
variable (h1 : S = 14)
variable (h2 : F - 10 = 7 * (S - 10))

theorem father_age_38 : F = 38 :=
by
  sorry

end father_age_38_l514_514107


namespace solve_fractional_equation_l514_514007

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 1) :
  (x^2 - x + 2) / (x - 1) = x + 3 ↔ x = 5 / 3 :=
by
  sorry

end solve_fractional_equation_l514_514007


namespace count_repeating_decimals_l514_514607

theorem count_repeating_decimals (s : Set ℕ) :
  (∀ n, n ∈ s ↔ 1 ≤ n ∧ n ≤ 20 ∧ ¬∃ k, k * 3 = n) →
  (s.card = 14) :=
by 
  sorry

end count_repeating_decimals_l514_514607


namespace fraction_bounds_l514_514852

theorem fraction_bounds (n : ℕ) (h : 0 < n) : (1 : ℚ) / 2 ≤ n / (n + 1 : ℚ) ∧ n / (n + 1 : ℚ) < 1 :=
by
  sorry

end fraction_bounds_l514_514852


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514896

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514896


namespace coefficient_of_x2_in_expansion_l514_514572

def binomial_coefficient (n k : Nat) : Nat := Nat.choose k n

def binomial_term (a x : ℕ) (n r : ℕ) : ℕ :=
  a^(n-r) * binomial_coefficient n r * x^r

theorem coefficient_of_x2_in_expansion : 
  binomial_term 2 1 5 2 = 80 := by sorry

end coefficient_of_x2_in_expansion_l514_514572


namespace axis_of_symmetry_l514_514677

theorem axis_of_symmetry (g : ℝ → ℝ) (h: ∀ x, g x = g (3 - x)) : 
  is_axis_of_symmetry g 1.5 := 
sorry

end axis_of_symmetry_l514_514677


namespace repeating_decimals_for_n_div_18_l514_514599

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end repeating_decimals_for_n_div_18_l514_514599


namespace no_primes_satisfy_square_l514_514547

theorem no_primes_satisfy_square (p q r : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime) :
  ¬∃ n : ℕ, (p^2 + p) * (q^2 + q) * (r^2 + r) = n^2 :=
by
  sorry

end no_primes_satisfy_square_l514_514547


namespace solution_correctness_l514_514058

def is_prime (n : ℕ) : Prop := Nat.Prime n

def problem_statement (a b c : ℕ) : Prop :=
  (a * b * c = 56) ∧
  (a * b + b * c + a * c = 311) ∧
  is_prime a ∧ is_prime b ∧ is_prime c

theorem solution_correctness (a b c : ℕ) (h : problem_statement a b c) :
  a = 2 ∨ a = 13 ∨ a = 19 ∧
  b = 2 ∨ b = 13 ∨ b = 19 ∧
  c = 2 ∨ c = 13 ∨ c = 19 :=
by
  sorry

end solution_correctness_l514_514058


namespace quadratic_has_real_root_l514_514004

theorem quadratic_has_real_root (p : ℝ) : 
  ∃ x : ℝ, 3 * (p + 2) * x^2 - p * x - (4 * p + 7) = 0 :=
sorry

end quadratic_has_real_root_l514_514004


namespace slope_of_tangent_at_A_l514_514291

-- Definitions
def f (x : ℝ) : ℝ := 2^x

-- Theorem statement
theorem slope_of_tangent_at_A : Deriv f 1 = 2 * Real.log 2 :=
by
  sorry

end slope_of_tangent_at_A_l514_514291


namespace equilateral_triangle_ratio_l514_514935

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514935


namespace angle_addition_l514_514636

open Real

theorem angle_addition (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : tan α = 1 / 3) (h₄ : cos β = 3 / 5) : α + 3 * β = 3 * π / 4 :=
by
  sorry

end angle_addition_l514_514636


namespace sum_binomial_coeffs_l514_514830

theorem sum_binomial_coeffs (n : ℕ) : 
  (∑ k in finset.range (n + 1), nat.choose 5 k) = 32 :=
by {
  sorry
}

end sum_binomial_coeffs_l514_514830


namespace vikki_take_home_pay_l514_514472

-- Define the conditions
def hours_worked : ℕ := 42
def pay_rate : ℝ := 10
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5

-- Define the gross earnings function
def gross_earnings (hours_worked : ℕ) (pay_rate : ℝ) : ℝ := hours_worked * pay_rate

-- Define the deductions functions
def tax_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def insurance_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def total_deductions (tax : ℝ) (insurance : ℝ) (dues : ℝ) : ℝ := tax + insurance + dues

-- Define the take-home pay function
def take_home_pay (gross : ℝ) (deductions : ℝ) : ℝ := gross - deductions

theorem vikki_take_home_pay :
  take_home_pay (gross_earnings hours_worked pay_rate)
    (total_deductions (tax_deduction (gross_earnings hours_worked pay_rate) tax_rate)
                      (insurance_deduction (gross_earnings hours_worked pay_rate) insurance_rate)
                      union_dues) = 310 :=
by
  sorry

end vikki_take_home_pay_l514_514472


namespace equilateral_triangle_ratio_l514_514860

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514860


namespace inequality_l514_514659

noncomputable theory

-- Define the function f
def f (x : ℝ) : ℝ := 1 - |x - 1| - |x - 2|

-- The conditions as assumptions
variables (x a b c y z : ℝ)
axiom solution_set : {x : ℝ | f(x + 1) ≥ 0} = set.interval 0 1

-- Ensure the constraints on the given variables
axiom cond_x_sq_y_sq_z_sq : x^2 + y^2 + z^2 = 1
axiom cond_a_sq_b_sq_c_sq : a^2 + b^2 + c^2 = 1

-- The conjecture we seek to prove
theorem inequality : a * x + b * y + c * z ≤ 1 := 
sorry

end inequality_l514_514659


namespace solution_set_of_inequality_l514_514042

theorem solution_set_of_inequality (x : ℝ) : (x * |x - 1| > 0) ↔ (0 < x ∧ x < 1 ∨ 1 < x) := 
by
  sorry

end solution_set_of_inequality_l514_514042


namespace ratio_of_girls_with_long_hair_l514_514053

theorem ratio_of_girls_with_long_hair (total_people boys girls short_hair long_hair : ℕ)
  (h1 : total_people = 55)
  (h2 : boys = 30)
  (h3 : girls = total_people - boys)
  (h4 : short_hair = 10)
  (h5 : long_hair = girls - short_hhair) :
  long_hair / gcd long_hair girls = 3 ∧ girls / gcd long_hair girls = 5 := 
by {
  -- This placeholder indicates where the proof should be.
  sorry
}

end ratio_of_girls_with_long_hair_l514_514053


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514948

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514948


namespace hyperbola_equation_l514_514699

theorem hyperbola_equation
  (a b c : ℝ)
  (ellipse_eq : ∀ x y : ℝ, x^2 + y^2/2 = 1)
  (hyperbola_eq : ∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1)
  (vertices_hyperbola : a = sqrt 2)
  (eccentricity_product : (sqrt 2) * (1 / (sqrt 2)) = 1) :
  ∀ x y : ℝ, y^2 / 2 - x^2 / 2 = 1 :=
by
  intros x y
  sorry

end hyperbola_equation_l514_514699


namespace range_of_a_l514_514037

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| - |x + 1| ≤ a) → a ∈ set.Ici 2 :=
by
  -- Proof is omitted.
  sorry

end range_of_a_l514_514037


namespace trisection_points_on_circle_implies_equilateral_l514_514978

theorem trisection_points_on_circle_implies_equilateral (A B C : Point) :
  let C₁ := trisection_point A B 1
  let C₂ := trisection_point A B 2
  let A₁ := trisection_point B C 1
  let A₂ := trisection_point B C 2
  let B₁ := trisection_point C A 1
  let B₂ := trisection_point C A 2
  (circle C₁ C₂ A₁ A₂ B₁ B₂) → (equilateral A B C) :=
by
  sorry

end trisection_points_on_circle_implies_equilateral_l514_514978


namespace sum_of_all_integers_a_l514_514318

theorem sum_of_all_integers_a (a x y : ℤ) :
  (x + 1 ≤ (2 * x - 5) / 3) → (a - x > 1) → (4 + y / (y - 3) = (a - 1) / (3 - y)) → 
  (∃ k : ℕ, y = k) → ∑ a in {a : ℤ | a > -7 ∧ ∃ y : ℤ, y = (13 - a) / 5 ∧ (y = 0 ∨ y ∈ ℕ) ∧ 4 + y / (y - 3) = (a - 1) / (3 - y)}, a = 24 :=
by
  sorry

end sum_of_all_integers_a_l514_514318


namespace kelcie_books_multiple_l514_514386

theorem kelcie_books_multiple (x : ℕ) :
  let megan_books := 32
  let kelcie_books := megan_books / 4
  let greg_books := x * kelcie_books + 9
  let total_books := megan_books + kelcie_books + greg_books
  total_books = 65 → x = 2 :=
by
  intros megan_books kelcie_books greg_books total_books h
  sorry

end kelcie_books_multiple_l514_514386


namespace linear_congruence_solution_l514_514347

theorem linear_congruence_solution (a b m : ℕ) :
  ∃ x₀, (a * x₀ ≡ b [MOD m]) ↔ (gcd a m) ∣ b := sorry

noncomputable def general_solution (a b m : ℕ) (d : ℕ := gcd a m)
  (m₁ : ℕ := m / d) (a₁ : ℕ := a / d) (b₁ : ℕ := b / d)
  (x₀ : ℕ := (nat.coe_mod_eq rfl).some) :
  set ℕ :=
{ x | ∃ k : ℤ, x = x₀ + k * m₁ } 

end linear_congruence_solution_l514_514347


namespace cannotFormFortyCentsWithFiveCoins_l514_514584

-- Define the value of each coin type
inductive Coin
| penny | nickel | dime | quarter | half_dollar
deriving DecidableEq

def coinValue : Coin → ℕ
| Coin.penny := 1
| Coin.nickel := 5
| Coin.dime := 10
| Coin.quarter := 25
| Coin.half_dollar := 50

-- Define the main theorem
theorem cannotFormFortyCentsWithFiveCoins :
  ¬ ∃ (coins : Finset Coin), coins.card = 5 ∧ coins.sum coinValue = 40 :=
by
  sorry

end cannotFormFortyCentsWithFiveCoins_l514_514584


namespace area_of_triangle_is_six_l514_514274

variables (a b c : ℝ)

def valid_triangle : Prop :=
  (a + b + c = 12) ∧
  ((a + 4) / 3 = (b + 3) / 2) ∧
  ((a + 4) / 3 = (c + 8) / 4)

def calculate_area (b c : ℝ) : ℝ := 1/2 * b * c

theorem area_of_triangle_is_six (h : valid_triangle a b c) : 
  calculate_area b c = 6 :=
sorry

end area_of_triangle_is_six_l514_514274


namespace chef_egg_usage_l514_514800

theorem chef_egg_usage :
  ∀ (total_eggs eggs_in_fridge cakes: ℕ),
  total_eggs = 60 →
  eggs_in_fridge = 10 →
  cakes = 10 →
  (total_eggs - eggs_in_fridge) / cakes = 5 := 
by
  intros total_eggs eggs_in_fridge cakes h_total h_fridge h_cakes
  rw [h_total, h_fridge, h_cakes]
  simp
  sorry

end chef_egg_usage_l514_514800


namespace shift_graph_function_l514_514063

theorem shift_graph_function :
  ∀ (x : ℝ), ∃ (g : ℝ → ℝ), (g x = 2^(1 - x)) ∧ (∀ (x : ℝ), g x = (λ x, 2^(-x)) (x - 1)) :=
by
  sorry

end shift_graph_function_l514_514063


namespace range_of_t_l514_514661

noncomputable def f (x t : ℝ) := (Real.log x + ((x - t)^2)) / x

theorem range_of_t (t : ℝ) :
  (∀ x ∈ Set.Icc 1 2, (deriv (deriv (λ x, f x t)) x) * x + f x t > 0) →
  t < 3/2 :=
sorry

end range_of_t_l514_514661


namespace area_of_quadrilateral_extension_l514_514773

variable (ABCD A_1 B_1 C_1 D_1 : Type)
variable (S : ℝ) -- Let S be the area of quadrilateral ABCD

-- Given the conditions on sides extension
variable (BB1_equal_AB : BB_1 = 2 * AB)
variable (CC1_equal_BC : CC_1 = 2 * BC)
variable (DD1_equal_CD : DD_1 = 2 * CD)
variable (AA1_equal_DA : AA_1 = 2 * DA)

theorem area_of_quadrilateral_extension (h1 : BB1_equal_AB) (h2 : CC1_equal_BC)
  (h3 : DD1_equal_CD) (h4 : AA1_equal_DA) : 
  children: ABCD, B, B_1, C, C_1, D, D_1, A, A_1
  have area_A1B1C1D1 : area A_1B_1C_1D_1 = 5 * area ABCD := sorry
 
end area_of_quadrilateral_extension_l514_514773


namespace not_divisible_by_3_l514_514836

def total_mass (n : ℕ) : ℕ := n * (n + 1) / 2

theorem not_divisible_by_3 : ¬ (∃ (a b c : List ℕ) (h₁ : a.all (λ x, x ∈ List.range 68)) (h₂ : b.all (λ x, x ∈ List.range 68)) (h₃ : c.all (λ x, x ∈ List.range 68)),
                                a.sum = b.sum ∧ b.sum = c.sum ∧ a.sum + b.sum + c.sum = total_mass 67) :=
by
  let n := 67
  have h1 : total_mass n = 2278 := by sorry
  have h2 : 2278 % 3 ≠ 0 := by sorry
  intro h
  cases' h with a h
  cases' h with b h
  cases' h with c h
  cases' h with ha h
  cases' h with hb h
  cases' h with hc h
  have sum_correct : a.sum + b.sum + c.sum = total_mass 67 := h.right
  show 2278 % 3 = 0 from h.right ▸ sum_correct.divisible.PQ
  contradiction

end not_divisible_by_3_l514_514836


namespace integer_length_impossible_l514_514376

theorem integer_length_impossible
  (A B C D E I : Type)
  [is_triangle (angle A B C = 90)]
  (D_on_AC : D ∈ line AC)
  (E_on_AB : E ∈ line AB)
  (angle_REQ1 : angle ABD = angle DBC)
  (angle_REQ2 : angle ACE = angle ECB)
  (AB AC BI ID CI IE : ℕ) :
  ¬(integral_lengths AB AC BI ID CI IE) :=
begin
  sorry,
end

end integer_length_impossible_l514_514376


namespace gcd_2_pow_2010_minus_3_2_pow_2001_minus_3_l514_514855

-- Definitions based on conditions
def a := (2:ℤ) ^ 2010 - 3
def b := (2:ℤ) ^ 2001 - 3

-- The proof statement
theorem gcd_2_pow_2010_minus_3_2_pow_2001_minus_3 :
  Int.gcd a b = 1533 := by
  sorry

end gcd_2_pow_2010_minus_3_2_pow_2001_minus_3_l514_514855


namespace parallelogram_angle_ratios_l514_514716

-- Definitions related to the given conditions
variables (A B C D O : Type) [parallelogram A B C D]
variable (θ : ℝ) -- angle DBA
variables [real.cos_angles A B C D O θ]

-- Assertion of the specific problem with conditions and goal
theorem parallelogram_angle_ratios (h1 : ∠CAB = 3 * θ)
                                   (h2 : ∠DBC = 3 * θ)
                                   (h3 : ∠DBA = θ) :
                                   ∠ACB / ∠AOB = 5 / 8 :=
by
  sorry

end parallelogram_angle_ratios_l514_514716


namespace problem1_problem2_l514_514660

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := abs (m * x - 2) - abs (m * x + 1)

theorem problem1 (x : ℝ) : f 1 x ≤ 1 ↔ 0 ≤ x :=
by sorry

theorem problem2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 3) :
  sqrt a + sqrt b + sqrt c ≤ 3 :=
by sorry

end problem1_problem2_l514_514660


namespace work_completion_days_l514_514326

-- Define the initial conditions
def initial_men : ℝ := 12
def initial_hours_per_day : ℝ := 8
def initial_days : ℝ := 10
def total_man_hours : ℝ := initial_men * initial_hours_per_day * initial_days -- 960 man-hours

-- Define the new setup conditions
def new_men : ℝ := 9.00225056264066
def new_hours_per_day : ℝ := 13.33

-- Define the variable for the new number of days
def new_days : ℝ := 960 / (new_men * new_hours_per_day) -- should compute to approximately 8

theorem work_completion_days :
  new_days ≈ 8 := 
begin
  -- Using the values defined above, we prove new_days is approximately 8 days.
  sorry
end

end work_completion_days_l514_514326


namespace number_of_birds_is_20_l514_514788

-- Define the given conditions
def distance_jim_disney : ℕ := 50
def distance_disney_london : ℕ := 60
def total_travel_distance : ℕ := 2200

-- Define the number of birds
def num_birds (B : ℕ) : Prop :=
  (distance_jim_disney + distance_disney_london) * B = total_travel_distance

-- The theorem stating the number of birds
theorem number_of_birds_is_20 : num_birds 20 :=
by
  unfold num_birds
  sorry

end number_of_birds_is_20_l514_514788


namespace complement_solution_set_solve_inequality_with_respect_to_x_l514_514498

-- Problem 1
theorem complement_solution_set (x : ℝ) : (∃ x, (2 * x + 1) / (3 - x) < 2) ↔ x ∈ (set.Icc (5 / 4 : ℝ) 3) :=
sorry

-- Problem 2
theorem solve_inequality_with_respect_to_x (a x : ℝ) : 
  (a = 0 ∧ x ≥ 1) ∨
  (a ≠ 0 ∧ ((a < 0 ∧ x ∈ (set.Iic (4 / a) ∪ set.Ici 1)) ∨
            (0 < a ∧ a < 4 ∧ x ∈ set.Icc 1 (4 / a)) ∨
            (a = 4 ∧ x = 1) ∨
            (a > 4 ∧ x ∈ set.Icc (4 / a) 1))) :=
sorry

end complement_solution_set_solve_inequality_with_respect_to_x_l514_514498


namespace no_solution_eq_b_no_solution_eq_c_l514_514482

def eq_b (x : ℝ) : Prop := abs (2 * x) + 7 = 0
def eq_c (x : ℝ) : Prop := sqrt (3 * x) + 2 = 0

theorem no_solution_eq_b : ¬ ∃ x, eq_b x := 
by {
  sorry
}

theorem no_solution_eq_c : ¬ ∃ x, eq_c x := 
by {
  sorry
}

end no_solution_eq_b_no_solution_eq_c_l514_514482


namespace Q_divisible_by_5_pow_38_l514_514736

theorem Q_divisible_by_5_pow_38 :
  let Q := ∏ i in finset.range 150, (2 * i + 1)
  ∃ k : ℕ, Q % 5^k = 0 ∧ k = 38 :=
by
  sorry

end Q_divisible_by_5_pow_38_l514_514736


namespace min_value_2a_3b_equality_case_l514_514638

theorem min_value_2a_3b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) : 
  2 * a + 3 * b ≥ 25 :=
sorry

theorem equality_case (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) :
  (a = 5) ∧ (b = 5) → 2 * a + 3 * b = 25 :=
sorry

end min_value_2a_3b_equality_case_l514_514638


namespace hexagon_area_from_apothem_l514_514675

noncomputable def regular_octagon_apothem : ℝ := 3

def is_midpoint (P₁ P₂ : ℝ × ℝ) (Q : ℝ × ℝ) : Prop :=
  Q.1 = (P₁.1 + P₂.1) / 2 ∧ Q.2 = (P₁.2 + P₂.2) / 2

def hexagon_area (s : ℝ) : ℝ :=
  (3 * real.sqrt 3 / 2) * s^2

theorem hexagon_area_from_apothem :
  let P₁ := (1 : ℝ) -- Coordinates are not important for the proof without the actual computation
      P₂ := (2 : ℝ)--) as calculations are based on geometric properties.
      P₃ := (3 : ℝ)--)
      P₄ := (4 : ℝ)--)
      P₅ := (5 : ℝ)--)
      P₆ := (6 : ℝ)--)
      P₇ := (7 : ℝ)--)
      P₈ := (8 : ℝ)--)
      Q₁ := (is_midpoint P₁ P₂)
      Q₂ := (is_midpoint P₂ P₃)
      Q₃ := (is_midpoint P₃ P₄)
      Q₄ := (is_midpoint P₄ P₅)
      Q₅ := (is_midpoint P₅ P₆)
      Q₆ := (is_midpoint P₆ P₇)
  hex_midpoints := [Q₁, Q₂, Q₃, Q₄, Q₅, Q₆] -- These represent the midpoints Q₁ to Q₆ as both conditions and definitions
  (area : ℝ) = 162 * real.sqrt 3 - 108 * real.sqrt 6 :=
sorry

end hexagon_area_from_apothem_l514_514675


namespace contest_path_count_l514_514520

def Grid : Type := -- model the grid structure
sorry

constant valid_paths : Grid → String → Nat
constant CONTESTA : String := "CONTESTA"

-- Define the specific grid for this problem (its exact representation is skipped here)
constant specific_grid : Grid

theorem contest_path_count : valid_paths specific_grid CONTESTA = 4375 := 
sorry -- proof is not required

end contest_path_count_l514_514520


namespace intersection_and_complement_l514_514378

variable (x : ℝ)

def A : set ℝ := { x | x >= -1 }
def B : set ℝ := { x | 1 < x ∧ x ≤ 3 }
def CR_B : set ℝ := { x | x ≤ 1 ∨ x > 3 }

theorem intersection_and_complement :
  (A ∩ CR_B) = { x | -1 ≤ x ∧ x ≤ 1 ∨ x > 3 } :=
by
  sorry

end intersection_and_complement_l514_514378


namespace triangle_ABC_l514_514322

variable {A B C : ℝ}
variable {a b c : ℝ}

-- Given conditions
def cond1 : Prop := (sqrt 3) * b * cos A = a * sin B
def cond2 : Prop := a = 6
def cond3 : Prop := (1 / 2) * b * c * (√ 3 / 2) = 9 * √ 3

-- Mathematical equivalent proof problem
theorem triangle_ABC :
  cond1 → cond2 → cond3 → 
  A = π / 3 ∧ b = 6 ∧ c = 6 :=
by
  intros
  sorry

end triangle_ABC_l514_514322


namespace correctly_identify_digit_l514_514995

theorem correctly_identify_digit (X Y Z : ℕ) (h1 : X ≠ Y) (h2 : X ≠ Z) (h3 : Y = 2 * X + 2 * X \ /* and some more conditions specifying digit relationships */) : 
  X = 6 :=
by
  sorry

end correctly_identify_digit_l514_514995


namespace sum_of_digits_T_l514_514833

theorem sum_of_digits_T (T S : ℕ) (k : ℕ → ℕ) (h₀ : ∀ i, 1 ≤ i ∧ i ≤ 10 → k i = i)
  (h₁ : S = 2520) (h₂: T = 60) (h_least : ∃ n, T = n ∧ n > 0 ∧ (card {i | 1 ≤ i ∧ i ≤ 10 ∧ T % k i = 0} ≥ 5)) :
  (∑ d in (Nat.digits 10 T), d) = 6 :=
by
  sorry

end sum_of_digits_T_l514_514833


namespace camel_height_in_feet_correct_l514_514330

def hare_height_in_inches : ℕ := 14
def multiplication_factor : ℕ := 24
def inches_to_feet_ratio : ℕ := 12

theorem camel_height_in_feet_correct :
  (hare_height_in_inches * multiplication_factor) / inches_to_feet_ratio = 28 := by
  sorry

end camel_height_in_feet_correct_l514_514330


namespace total_husk_is_30_bags_l514_514325

-- Define the total number of cows and the number of days.
def numCows : ℕ := 30
def numDays : ℕ := 30

-- Define the rate of consumption: one cow eats one bag in 30 days.
def consumptionRate (cows : ℕ) (days : ℕ) : ℕ := cows / days

-- Define the total amount of husk consumed in 30 days by 30 cows.
def totalHusk (cows : ℕ) (days : ℕ) (rate : ℕ) : ℕ := cows * rate

-- State the problem in a theorem.
theorem total_husk_is_30_bags : totalHusk numCows numDays 1 = 30 := by
  sorry

end total_husk_is_30_bags_l514_514325


namespace repeating_decimals_for_n_div_18_l514_514595

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end repeating_decimals_for_n_div_18_l514_514595


namespace symmetric_graphs_y_axis_l514_514645

open Function

theorem symmetric_graphs_y_axis (f g : ℝ → ℝ) (h : ∀ x, f x = f (-x)) :
  (f x = x^2 - 2*x) → (λ x, g x) = (λ x, x^2 + 2*x) :=
begin
  intros h1,
  sorry  -- Proof to be filled in
end

end symmetric_graphs_y_axis_l514_514645


namespace equilateral_triangle_ratio_l514_514929

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514929


namespace smallest_possible_product_l514_514776

theorem smallest_possible_product : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    {a, b, c, d} = {6, 7, 8, 9} ∧ 
    (∃ (x y : ℕ), 
        (x = 10 * a + b ∧ y = 10 * c + d ∨ 
         x = 10 * a + c ∧ y = 10 * b + d ∨ 
         x = 10 * a + d ∧ y = 10 * b + c ∨ 
         x = 10 * b + a ∧ y = 10 * c + d ∨ 
         x = 10 * b + c ∧ y = 10 * a + d ∨ 
         x = 10 * b + d ∧ y = 10 * a + c ∨ 
         x = 10 * c + a ∧ y = 10 * b + d ∨ 
         x = 10 * c + b ∧ y = 10 * a + d ∨ 
         x = 10 * c + d ∧ y = 10 * a + b ∨ 
         x = 10 * d + a ∧ y = 10 * b + c ∨ 
         x = 10 * d + b ∧ y = 10 * a + c ∨ 
         x = 10 * d + c ∧ y = 10 * a + b) ∧ 
        x * y = 5372) := 
sorry

end smallest_possible_product_l514_514776


namespace solve_fractional_equation_l514_514006

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 1) :
  (x^2 - x + 2) / (x - 1) = x + 3 ↔ x = 5 / 3 :=
by
  sorry

end solve_fractional_equation_l514_514006


namespace num_repeating_decimals_between_1_and_20_l514_514592

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end num_repeating_decimals_between_1_and_20_l514_514592


namespace area_of_triangle_EFG_correct_l514_514792

noncomputable def area_of_triangle_EFG (A B C D E F G : Type) [triangle A B C] (AB BC CA : ℝ) (H1 : AB = 13) (H2 : BC = 15) (H3 : CA = 14) 
(H4 : midpoint B C D) (H5 : midpoint A D E) (H6 : midpoint B E F) (H7 : midpoint D F G) : ℝ := by
  -- sorry is placeholder for the proof
  sorry

-- The proof statement
theorem area_of_triangle_EFG_correct :
  area_of_triangle_EFG A B C D E F G AB BC CA H1 H2 H3 H4 H5 H6 H7 = 21 / 4 := 
by
  -- sorry is placeholder for the proof
  sorry

end area_of_triangle_EFG_correct_l514_514792


namespace evaluate_expression_l514_514539

theorem evaluate_expression :
  (π - 2023) ^ 0 + |(-9)| - 3 ^ 2 = 1 :=
by
  sorry

end evaluate_expression_l514_514539


namespace find_QS_l514_514791

theorem find_QS (R S Q : ℝ) (cosR : ℝ) (RS QR QS : ℝ) (h1 : cosR = 4 / 9) (h2 : RS = 9) (h3 : cosR = QR / RS) (h4 : QR = 4) :
  QS = Real.sqrt (RS * RS - QR * QR) :=
by
  rw [h2, h4]
  norm_num
  exact Real.sqrt_sq_eq_of_nonneg (by norm_num)

#eval find_QS (0) (0) (0) (4/9) (9) (4) (Real.sqrt 65) -- Evaluates to "true"

end find_QS_l514_514791


namespace find_t_l514_514206

theorem find_t : ∃ t, ∀ (x y : ℝ), (x, y) = (0, 1) ∨ (x, y) = (-6, -3) → (t, 7) ∈ {p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ ((0, 1) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) ∧ ((-6, -3) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) } → t = 9 :=
by
  sorry

end find_t_l514_514206


namespace chord_length_square_l514_514172

theorem chord_length_square {r₁ r₂ R : ℝ} (r1_eq_4 : r₁ = 4) (r2_eq_8 : r₂ = 8) 
  (R_eq_12 : R = 12)
  (externally_tangent : tangent r₁ r₂)
  (internally_tangent_1 : tangent_internally r₁ R)
  (internally_tangent_2 : tangent_internally r₂ R)
  : exists (PQ : ℝ), PQ^2 = 3584 / 9 :=
by sorry

end chord_length_square_l514_514172


namespace slices_per_pizza_l514_514534

noncomputable def slices_in_pizza : Nat := 
s : Nat 

constants (S : ℕ) (b t s j : ℕ)
axiom h_b : b = S / 2
axiom h_t : t = S / 3
axiom h_s : s = S / 6
axiom h_j : j = S / 4
axiom h_total : b + t + s + j = 2 * S - 9

theorem slices_per_pizza : S = 12 := by
  -- Proof goes here
  sorry

end slices_per_pizza_l514_514534


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514893

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514893


namespace new_class_average_l514_514323

theorem new_class_average (total_students : ℕ) (students_group1 : ℕ) (avg1 : ℝ) (students_group2 : ℕ) (avg2 : ℝ) : 
  total_students = 40 → students_group1 = 28 → avg1 = 68 → students_group2 = 12 → avg2 = 77 → 
  ((students_group1 * avg1 + students_group2 * avg2) / total_students) = 70.7 :=
by
  sorry

end new_class_average_l514_514323


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514902

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514902


namespace closest_point_on_line_to_point_l514_514578

theorem closest_point_on_line_to_point :
  let d := (1, 4) in
  let v := (2 - 0, -1 - 3) in
  let proj_v_onto_d := ((2 * 1 + (-4) * 4) / (1^2 + 4^2)) * (1, 4) in
  let closest_point := (0, 3) + proj_v_onto_d in
  closest_point = (-14 / 17, 5 / 17) :=
by
  sorry

end closest_point_on_line_to_point_l514_514578


namespace tangent_line_at_zero_a_eq_1_monotonicity_of_f_maximum_value_h_l514_514294

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x - a * x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  f x a - x

noncomputable def h (a : ℝ) : ℝ :=
  g (Real.log (a + 1)) a

theorem tangent_line_at_zero_a_eq_1 :
  ∀ x : ℝ, a = 1 → (y = 1) := 
  sorry

theorem monotonicity_of_f :
  ∀ a x : ℝ, 
    -1 < a ∧ a ≤ 0 → 
    (∀ x, Deriv f x a > 0) ∧ 
    (a > 0 → ((∀ x < Real.log a, Deriv f x a < 0) 
      ∧ (∀ x > Real.log a, Deriv f x a > 0))) :=
  sorry

theorem maximum_value_h :
  ∀ a : ℝ, h a ≤ 1 :=
  sorry

end tangent_line_at_zero_a_eq_1_monotonicity_of_f_maximum_value_h_l514_514294


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514901

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514901


namespace triangular_number_reciprocal_sum_l514_514005

-- Define the k-th triangular number
def triangular_number (k : ℕ) : ℕ :=
  (k * (k + 1)) / 2

-- Problem statement: 
-- For any n ≠ 2, there exist n triangular numbers a₁, a₂, ..., aₙ such that ∑_{i=1}^n 1/aᵢ = 1

theorem triangular_number_reciprocal_sum (n : ℕ) (h1 : n > 0) (h2 : n ≠ 2) :
  ∃ (a : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → a i = triangular_number i) ∧ ∑ i in finset.range n, ((1 : ℚ) / (a i)) = 1 :=
by
  sorry

end triangular_number_reciprocal_sum_l514_514005


namespace stock_price_never_same_l514_514447

open BigOperators

/-- The stock price changes by 17% either increases or decreases every day at 12:00 PM. -/
def stock_price (initial_price : ℝ) (n : ℕ) (seq : Fin n → Bool) : ℝ :=
  initial_price * ∏ i, if seq i then 1.17 else 0.83

/-- The statement that stock price cannot take the same value twice -/
theorem stock_price_never_same (initial_price : ℝ) (n m : ℕ) (seq1 : Fin n → Bool) (seq2 : Fin m → Bool) :
  (stock_price initial_price n seq1 = stock_price initial_price m seq2) → 
  (seq1 = seq2) :=
sorry

end stock_price_never_same_l514_514447


namespace num_functions_equal_y1_l514_514429

-- Define the specific functions
def f1 (x : ℝ) : ℝ := if x = 0 then 0 else x / x
def f2 (t : ℝ) : ℝ := if t = -1 then 0 else (t + 1) / (t + 1)
def f3 (x : ℝ) : ℝ := if -1 ≤ x ∧ x < 1 then 1 else 0
def f4 (x : ℝ) : ℝ := if x = 0 then 0 else x ^ 0

-- Define the constant function y = 1
def y1 (x : ℝ) : ℝ := 1

-- Theorem to prove the number of functions that are the same as y=1 is 0
theorem num_functions_equal_y1 : 
  (if ∀ x, f1 x = y1 x then 1 else 0) + 
  (if ∀ t, f2 t = y1 t then 1 else 0) + 
  (if ∀ x, f3 x = y1 x then 1 else 0) + 
  (if ∀ x, f4 x = y1 x then 1 else 0) = 0 := 
by sorry

end num_functions_equal_y1_l514_514429


namespace garage_sale_items_total_l514_514086

-- Given conditions
def radio_18th_highest (n : Nat) : Prop := 
  n = 18

def radio_25th_lowest (m : Nat) : Prop := 
  m = 25

-- Proof problem statement
theorem garage_sale_items_total (hi : Nat) (lo : Nat) (n : Nat) (radio_18th_highest hi) (radio_25th_lowest lo) : n = hi + lo - 1 :=
by 
  -- Since hi = 18 and lo = 25, the total number of items n should be 18 + 25 - 1 = 42
  sorry

end garage_sale_items_total_l514_514086


namespace calculate_expression_l514_514161

theorem calculate_expression : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  calc
    10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2
    _ = 10 - 9 + 56 + 6 - 20 + 3 - 2 : by rw [mul_comm 8 7, mul_comm 5 4] -- Perform multiplications
    _ = 1 + 56 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 10 - 9
    _ = 57 + 6 - 20 + 3 - 2 : by norm_num  -- Simplify 1 + 56
    _ = 63 - 20 + 3 - 2 : by norm_num  -- Simplify 57 + 6
    _ = 43 + 3 - 2 : by norm_num -- Simplify 63 - 20
    _ = 46 - 2 : by norm_num -- Simplify 43 + 3
    _ = 44 : by norm_num -- Simplify 46 - 2

end calculate_expression_l514_514161


namespace least_four_digit_divisible_1_2_4_8_l514_514856

theorem least_four_digit_divisible_1_2_4_8 : ∃ n : ℕ, ∀ d1 d2 d3 d4 : ℕ, 
  n = d1*1000 + d2*100 + d3*10 + d4 ∧
  1000 ≤ n ∧ n < 10000 ∧
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d4 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d4 ∧
  n % 1 = 0 ∧
  n % 2 = 0 ∧
  n % 4 = 0 ∧
  n % 8 = 0 ∧
  n = 1248 :=
by
  sorry

end least_four_digit_divisible_1_2_4_8_l514_514856


namespace students_in_class_l514_514456

variable (G B : ℕ)

def total_plants (G B : ℕ) : ℕ := 3 * G + B / 3

theorem students_in_class (h1 : total_plants G B = 24) (h2 : B / 3 = 6) : G + B = 24 :=
by
  sorry

end students_in_class_l514_514456


namespace indefinite_integral_solution_l514_514222

theorem indefinite_integral_solution (x : ℝ) :
  ∫ (1 : ℝ) in 0..x, ( (1 + (x ^ (3 / 4))) ^ (2 / 3)) / (x ^ (9 / 4)) =
  -((4/5) * ((x ^ (-3 / 4) + 1) ^ (5 / 3))) + C :=
sorry

end indefinite_integral_solution_l514_514222


namespace shara_shells_l514_514785

def initial_shells : ℕ := 20
def first_vacation_day1_3 : ℕ := 5 * 3
def first_vacation_day4 : ℕ := 6
def second_vacation_day1_2 : ℕ := 4 * 2
def second_vacation_day3 : ℕ := 7
def third_vacation_day1 : ℕ := 8
def third_vacation_day2 : ℕ := 4
def third_vacation_day3_4 : ℕ := 3 * 2

def total_shells : ℕ :=
  initial_shells + 
  (first_vacation_day1_3 + first_vacation_day4) +
  (second_vacation_day1_2 + second_vacation_day3) + 
  (third_vacation_day1 + third_vacation_day2 + third_vacation_day3_4)

theorem shara_shells : total_shells = 74 :=
by
  sorry

end shara_shells_l514_514785


namespace solve_equation_l514_514413

noncomputable def solution (x y z : ℝ) : Prop :=
  (sin x ≠ 0)
  ∧ (cos y ≠ 0)
  ∧ ( (sin x)^2 + 1 / (sin x)^2 ) ^ 3
    + ( (cos y)^2 + 1 / (cos y)^2 ) ^ 3
    = 16 * (sin z)^2

theorem solve_equation (x y z : ℝ) (n m k : ℤ):
  solution x y z ↔
  (x = (π / 2) + π * n) ∧ (y = π * m) ∧ (z = (π / 2) + π * k) :=
by
  sorry

end solve_equation_l514_514413


namespace smallest_positive_period_of_f_sum_of_max_min_values_of_f_on_interval_l514_514289

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x) - 1

theorem smallest_positive_period_of_f :
  let T := π
  ∃ T > 0, ∀ x, f (x + T) = f x :=
by
  sorry

theorem sum_of_max_min_values_of_f_on_interval :
  let interval := Set.Icc (- π / 6) (- π / 12)
  let max_val := Real.sqrt 2 * Real.sin (π / 12)
  let min_val := Real.sqrt 2 * Real.sin (-π / 12)
  Set.finSup interval f + Set.finInf interval f = 0 :=
by
  sorry

end smallest_positive_period_of_f_sum_of_max_min_values_of_f_on_interval_l514_514289


namespace pq_sum_l514_514760

open Real

section Problem
variables (p q : ℝ)
  (hp : p^3 - 21 * p^2 + 35 * p - 105 = 0)
  (hq : 5 * q^3 - 35 * q^2 - 175 * q + 1225 = 0)

theorem pq_sum : p + q = 21 / 2 :=
sorry
end Problem

end pq_sum_l514_514760


namespace floor_plus_self_eq_l514_514216

theorem floor_plus_self_eq (r : ℝ) (h : ⌊r⌋ + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_self_eq_l514_514216


namespace largest_integer_x_cubed_lt_three_x_squared_l514_514474

theorem largest_integer_x_cubed_lt_three_x_squared : 
  ∃ x : ℤ, x^3 < 3 * x^2 ∧ (∀ y : ℤ, y^3 < 3 * y^2 → y ≤ x) :=
  sorry

end largest_integer_x_cubed_lt_three_x_squared_l514_514474


namespace chessboard_squares_equal_l514_514114

theorem chessboard_squares_equal :
  ∀ (m n : ℕ), m = 8 → n = 7 →
  let dark_squares_in_odd_rows := (n + 1) / 2 * (m + 1) / 2,
      dark_squares_in_even_rows := n / 2 * m / 2,
      dark_squares := dark_squares_in_odd_rows + dark_squares_in_even_rows,
      light_squares := m * n - dark_squares
  in dark_squares = light_squares :=
by
  intros m n hm hn
  have : m * n = 56 := by rw [hm, hn]; norm_num
  have : (m + 1) / 2 = 4 := by rw [hm]; norm_num
  have : m / 2 = 4 := by rw [hm]; norm_num
  have : (n + 1) / 2 = 4 := by rw [hn]; norm_num
  have : n / 2 = 3 := by rw [hn]; norm_num
  have: dark_squares = 4 * 4 + 3 * 4 := rfl
  have: light_squares = 56 - (16 + 12) := by rw [this]
  norm_num,
  refl,

end chessboard_squares_equal_l514_514114


namespace mass_percentage_H_correct_l514_514858

noncomputable def mass_percentage_H_in_CaH2 : ℝ :=
  let molar_mass_Ca : ℝ := 40.08
  let molar_mass_H : ℝ := 1.01
  let molar_mass_CaH2 : ℝ := molar_mass_Ca + 2 * molar_mass_H
  (2 * molar_mass_H / molar_mass_CaH2) * 100

theorem mass_percentage_H_correct :
  |mass_percentage_H_in_CaH2 - 4.80| < 0.01 :=
by
  sorry

end mass_percentage_H_correct_l514_514858


namespace largest_c_median_inequality_l514_514225

theorem largest_c_median_inequality {x : Fin 201 → ℝ} (h_sum : (∑ i, x i) = 0) :
  ∃ (c : ℝ), (∀ M : ℝ, M = x ⟨100⟩ → (∑ i, x i ^ 2) ≥ c * M ^ 2) ∧ c = 203.01 :=
by {
  -- This is where the proof would go
  sorry
}

end largest_c_median_inequality_l514_514225


namespace total_people_can_ride_l514_514145

theorem total_people_can_ride (num_people_per_teacup : Nat) (num_teacups : Nat) (h1 : num_people_per_teacup = 9) (h2 : num_teacups = 7) : num_people_per_teacup * num_teacups = 63 := by
  sorry

end total_people_can_ride_l514_514145


namespace functional_equation_solutions_l514_514213

noncomputable def F : ℝ → ℝ
variable x y : ℝ

theorem functional_equation_solutions (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x) * f(y) + f(x + y) = x * y) ↔ (f = λ x, x - 1 ∨ f = λ x, -x - 1) := 
sorry

end functional_equation_solutions_l514_514213


namespace inradius_one_third_height_l514_514399

theorem inradius_one_third_height
  (a d h r : ℝ)
  (ha_pos : a > 0)
  (hd_pos : d > 0)
  (hh_pos : h > 0)
  (hr_pos : r > 0) :
  let s := (a + (a + d) + (a + 2d)) / 2 in
  let A := (a + d) * h / 2 in
  let A' := s * r in
  (a + d) * h / 2 = s * r → r = h / 3 :=
by
  intros s A A' h_area_eq_s_ins motiv
  sorry

end inradius_one_third_height_l514_514399


namespace center_of_circle_l514_514571

theorem center_of_circle :
  ∀ (x y : ℝ), (x^2 - 8 * x + y^2 - 4 * y = 16) → (x, y) = (4, 2) :=
by
  sorry

end center_of_circle_l514_514571


namespace solve_sin_cos_eq_l514_514418

theorem solve_sin_cos_eq (x y z : ℝ) (n m k : ℤ) :
  (sin x ≠ 0 ∧ cos y ≠ 0 ∧ 
  (sin x)^2 + ((sin x)^2)⁻¹ = 2 ∧ 
  (cos y)^2 + ((cos y)^2)⁻¹ = 2 ∧ 
  (sin x)^2 + (cos y)^2 = 1 ) → 
  ( ∃ n m k : ℤ, x = π / 2 + π * n ∧ y = π * m ∧ z = π / 2 + π * k ) := 
sorry

end solve_sin_cos_eq_l514_514418


namespace valid_values_of_X_Y_l514_514851

-- Stating the conditions
def odd_combinations := 125
def even_combinations := 64
def revenue_diff (X Y : ℕ) := odd_combinations * X - even_combinations * Y = 5
def valid_limit (n : ℕ) := 0 < n ∧ n < 250

-- The theorem we want to prove
theorem valid_values_of_X_Y (X Y : ℕ) :
  revenue_diff X Y ∧ valid_limit X ∧ valid_limit Y ↔ (X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205) :=
  sorry

end valid_values_of_X_Y_l514_514851


namespace marathon_y_distance_l514_514506

theorem marathon_y_distance (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (total_yards : ℕ) (y : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : yards_per_marathon = 312) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 8) 
  (H5 : total_yards = num_marathons * yards_per_marathon) 
  (H6 : total_yards % yards_per_mile = y) 
  (H7 : 0 ≤ y) 
  (H8 : y < yards_per_mile) : 
  y = 736 :=
by 
  sorry

end marathon_y_distance_l514_514506


namespace count_repeating_decimals_l514_514608

theorem count_repeating_decimals (s : Set ℕ) :
  (∀ n, n ∈ s ↔ 1 ≤ n ∧ n ≤ 20 ∧ ¬∃ k, k * 3 = n) →
  (s.card = 14) :=
by 
  sorry

end count_repeating_decimals_l514_514608


namespace odd_function_and_monotonic_decreasing_l514_514625

variable (f : ℝ → ℝ)

-- Given conditions:
axiom condition_1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom condition_2 : ∀ x : ℝ, x > 0 → f x < 0

-- Statement to prove:
theorem odd_function_and_monotonic_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2) := by
  sorry

end odd_function_and_monotonic_decreasing_l514_514625


namespace range_of_a_l514_514267

theorem range_of_a (a : ℝ) (x : ℝ) : (x^2 + 2*x > 3) → (x > a) → (¬ (x^2 + 2*x > 3) → ¬ (x > a)) → a ≥ 1 :=
by
  intros hp hq hr
  sorry

end range_of_a_l514_514267


namespace repeating_decimals_count_l514_514600

theorem repeating_decimals_count :
  { n : ℕ | 1 ≤ n ∧ n ≤ 20 ∧ (¬∃ k, n = 9 * k) }.to_finset.card = 18 :=
by
  sorry

end repeating_decimals_count_l514_514600


namespace union_P_Q_l514_514666

noncomputable def P : Set ℝ := {x : ℝ | abs x ≥ 3}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x - 1}

theorem union_P_Q :
  (P ∪ Q) = Set.Iic (-3) ∪ Set.Ici (-1) :=
by {
  sorry
}

end union_P_Q_l514_514666


namespace number_of_correct_propositions_is_three_l514_514438

-- Definitions of the propositions
def prop1 : Prop := ∀ ΔABC ΔDEF : Triangle,
  (ΔABC.right_angle_triangle ∧ ΔDEF.right_angle_triangle) →
  (ΔABC.corresponding_right_angle_sides = ΔDEF.corresponding_right_angle_sides) →
  ΔABC ≅ ΔDEF

def prop2 : Prop := ∀ ΔABC ΔDEF : Triangle,
  (ΔABC.right_angle_triangle ∧ ΔDEF.right_angle_triangle) →
  (ΔABC.corresponding_acute_angles = ΔDEF.corresponding_acute_angles) →
  ΔABC ≅ ΔDEF

def prop3 : Prop := ∀ ΔABC ΔDEF : Triangle,
  (ΔABC.right_angle_triangle ∧ ΔDEF.right_angle_triangle) →
  (ΔABC.hypotenuse = ΔDEF.hypotenuse ∧
   ΔABC.corresponding_right_angle_sides = ΔDEF.corresponding_right_angle_sides) →
  ΔABC ≅ ΔDEF

def prop4 : Prop := ∀ ΔABC ΔDEF : Triangle,
  (ΔABC.right_angle_triangle ∧ ΔDEF.right_angle_triangle) →
  (ΔABC.corresponding_acute_angles = ΔDEF.corresponding_acute_angles ∧
   ΔABC.hypotenuse = ΔDEF.hypotenuse) →
  ΔABC ≅ ΔDEF

-- Thematical proof problem statement
theorem number_of_correct_propositions_is_three :
  (prop1 → True) ∧
  (prop2 → False) ∧
  (prop3 → True) ∧
  (prop4 → True) :=
by sorry

end number_of_correct_propositions_is_three_l514_514438


namespace q_divisible_by_5k_l514_514738

noncomputable def largest_power_of_five_dividing_Q : ℕ := 35

theorem q_divisible_by_5k : 
  let Q := (finset.range (299 + 1)).filter (λ n, n % 2 = 1).prod (λ n, (n : ℕ))
  let k := 5
  let l := largest_power_of_five_dividing_Q
  ∃ l : ℕ, l = 35 ∧ Q % (5 ^ l) = 0 :=
sorry

end q_divisible_by_5k_l514_514738


namespace graduation_photo_number_of_ways_l514_514455

def number_of_arrangements (A B C D E F G : Type) : Nat :=
  let entities := [(B, C), D, E, F, G]
  let arrangements_without_A := 5.factorial * 2    -- 5! * 2!
  let valid_insertion_positions_for_A := 5
  arrangements_without_A * valid_insertion_positions_for_A

theorem graduation_photo_number_of_ways :
  number_of_arrangements A B C D E F G = 1200 :=
by
  sorry

end graduation_photo_number_of_ways_l514_514455


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514894

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514894


namespace correct_inverse_proportion_function_l514_514084

def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x = k / x

def function_A := λ (x : ℝ), k / x
def function_B := λ (x : ℝ), 2 * x / 3
def function_C := λ (x : ℝ), -x + 5
def function_D := λ (x : ℝ), 2 * x⁻¹

theorem correct_inverse_proportion_function : is_inverse_proportion function_D :=
by
  sorry

end correct_inverse_proportion_function_l514_514084


namespace exists_m_integer_l514_514013

def ceil (x : ℝ) : ℤ := Int.ceil x

def f (x : ℝ) : ℝ := x * (ceil x)

def f_iter : ℕ → ℝ → ℝ
| 0, x => x
| (n + 1), x => f (f_iter n x)

theorem exists_m_integer (k : ℕ) (hk : 0 < k) :
  ∃ m : ℕ, ∃ r : ℝ, r = k + 1 / 2 ∧ f_iter m r ∈ ℤ := 
begin
  let r : ℝ := k + 1 / 2,
  sorry
end

end exists_m_integer_l514_514013


namespace B_time_to_complete_work_l514_514987

variable {W : ℝ} {R_b : ℝ} {T_b : ℝ}

theorem B_time_to_complete_work (h1 : 3 * R_b * (T_b - 10) = R_b * T_b) : T_b = 15 :=
by
  sorry

end B_time_to_complete_work_l514_514987


namespace kit_time_to_ticket_window_l514_514357

theorem kit_time_to_ticket_window 
  (rate : ℝ)
  (remaining_distance : ℝ)
  (yard_to_feet_conv : ℝ)
  (new_rate : rate = 90 / 30)
  (remaining_distance_in_feet : remaining_distance = 100 * yard_to_feet_conv)
  (yard_to_feet_conv_val : yard_to_feet_conv = 3) :
  (remaining_distance / rate = 100) := 
by 
  simp [new_rate, remaining_distance_in_feet, yard_to_feet_conv_val]
  sorry

end kit_time_to_ticket_window_l514_514357


namespace boat_speed_proof_l514_514101

noncomputable def speed_in_still_water : ℝ := sorry -- Defined but proof skipped

def stream_speed : ℝ := 4
def distance_downstream : ℝ := 32
def distance_upstream : ℝ := 16

theorem boat_speed_proof (v : ℝ) :
  (distance_downstream / (v + stream_speed) = distance_upstream / (v - stream_speed)) →
  v = 12 :=
by
  sorry

end boat_speed_proof_l514_514101


namespace sufficient_number_of_cuts_for_100_similar_polygons_l514_514775

theorem sufficient_number_of_cuts_for_100_similar_polygons :
  ∃ n : ℕ, ∀ (P : set (set ℕ)), (P₀ := {4}) ∧ 
  (∀ k > 0, ∀ Q ∈ P, ∃ Q₁ Q₂ ∈ P, (Q₁ ∪ Q₂ = Q) ∧ 
   ((Q₁ ∩ Q₂ = ∅) ∧ (∀ x ∈ Q₁, x ∈ Q → convex x) ∧ (∀ x ∈ Q₂, x ∈ Q → convex x))) ∧
  ∀ Q₃ ∈ P, (Q₃ ⊆ Q ∧ vertex_count Q₃ + vertex_count Q ≥ vertex_count Q + 1) →
  ∃ Q₄ : set ℕ, Q₄ ⊆ P ∧ vertex_count Q₄ = 100 := sorry

end sufficient_number_of_cuts_for_100_similar_polygons_l514_514775


namespace number_of_elements_in_A_inter_B_eq_one_l514_514272

def A : set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, exp x)}
def B : set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, log (1/2) x)}

theorem number_of_elements_in_A_inter_B_eq_one : ∃! p : ℝ × ℝ, p ∈ A ∧ p ∈ B :=
sorry

end number_of_elements_in_A_inter_B_eq_one_l514_514272


namespace integer_length_impossible_l514_514375

theorem integer_length_impossible
  (A B C D E I : Type)
  [is_triangle (angle A B C = 90)]
  (D_on_AC : D ∈ line AC)
  (E_on_AB : E ∈ line AB)
  (angle_REQ1 : angle ABD = angle DBC)
  (angle_REQ2 : angle ACE = angle ECB)
  (AB AC BI ID CI IE : ℕ) :
  ¬(integral_lengths AB AC BI ID CI IE) :=
begin
  sorry,
end

end integer_length_impossible_l514_514375


namespace number_of_valid_configurations_l514_514421

-- Definitions based on the conditions
def is_valid_combination (square1 square2 : char) : Prop := sorry -- Define the valid combination based on problem's condition

-- The main theorem to prove
theorem number_of_valid_configurations : 
  let squares := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] in
  ∑ s1 in squares, ∑ s2 in squares, if s1 < s2 ∧ is_valid_combination s1 s2 then 1 else 0 = 7 :=
sorry

end number_of_valid_configurations_l514_514421


namespace length_of_bridge_correct_l514_514127

open Real

noncomputable def length_of_bridge (length_of_train : ℝ) (time_to_cross : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed := speed_kmph * (1000 / 3600)
  let total_distance := speed * time_to_cross
  total_distance - length_of_train

theorem length_of_bridge_correct :
  length_of_bridge 200 34.997200223982084 36 = 149.97200223982084 := by
  sorry

end length_of_bridge_correct_l514_514127


namespace equilateral_triangle_ratio_l514_514911

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514911


namespace ratio_eq_sqrt3_div_2_l514_514943

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514943


namespace complex_power_identity_l514_514153

section
variable (i : ℂ)
hypothesis hi : i^2 = -1

theorem complex_power_identity : (((1 + i) / (1 - i)) ^ 100) = 1 :=
by
  sorry
end

end complex_power_identity_l514_514153


namespace debt_doubles_in_correct_time_l514_514671

noncomputable def debt_doubling_time_Hannah (initial_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  2 * initial_amount / (initial_amount * interest_rate)

noncomputable def debt_doubling_time_Julia (initial_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  2 * initial_amount / (initial_amount * interest_rate)

theorem debt_doubles_in_correct_time :
  debt_doubling_time_Hannah 200 0.08 = 12.5 ∧ debt_doubling_time_Julia 300 0.06 = 50 / 3 :=
by
  unfold debt_doubling_time_Hannah debt_doubling_time_Julia
  split
  calc
    2 * 200 / (200 * 0.08) = 400 / 16 := by norm_num
    ... = 25 := by norm_num
  calc
    2 * 300 / (300 * 0.06) = 600 / 18 := by norm_num
    ... = 50 / 3 := by norm_num
  sorry

end debt_doubles_in_correct_time_l514_514671


namespace sum_of_solutions_l514_514582

noncomputable def sumOfRealSolutions : ℝ :=
  (83 : ℝ) / 196

theorem sum_of_solutions (x : ℝ) (hx : x > 0) :
  (sqrt x + sqrt (9 / x) + sqrt (x + 9 / x + 1) = 7) →
  x = sumOfRealSolutions := 
sorry

end sum_of_solutions_l514_514582


namespace num_repeating_decimals_between_1_and_20_l514_514590

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end num_repeating_decimals_between_1_and_20_l514_514590


namespace area_parallelogram_is_26_sqrt_3_l514_514495

variables (a b p q : EuclideanSpace ℝ (Fin 2)) -- Euclidean space of dimension 2

-- Definitions from conditions
def a := 6 * p - q
def b := p + 2 * q
def norm_p : ℝ := 8
def norm_q : ℝ := 1 / 2
def angle_pq : ℝ := Real.pi / 3

-- Positively verify norms and angle
axiom h_norm_p : ‖p‖ = norm_p
axiom h_norm_q : ‖q‖ = norm_q
axiom h_angle_pq : ∠(p, q) = angle_pq

-- Theorem statement
theorem area_parallelogram_is_26_sqrt_3 : ‖a × b‖ = 26 * Real.sqrt 3 :=
by
  -- Ensure to include necessary logic to complete the proof, currently skipped
  sorry

end area_parallelogram_is_26_sqrt_3_l514_514495


namespace allan_plums_l514_514003

theorem allan_plums (A : ℕ) (h1 : 7 - A = 3) : A = 4 :=
sorry

end allan_plums_l514_514003


namespace spurs_total_basketballs_l514_514797

theorem spurs_total_basketballs (players : ℕ) (basketballs_per_player : ℕ) (h1 : players = 22) (h2 : basketballs_per_player = 11) : players * basketballs_per_player = 242 := by
  sorry

end spurs_total_basketballs_l514_514797


namespace solve_problem_l514_514718

noncomputable def diameter_perpendicular_to_chord 
(O : Type*) [MetricSpace O] [Circle O] 
(A B C D E F : O) 
(hAB : is_diameter A B) 
(hCD_perp_AB : is_perpendicular C D A B E) 
(hEF_perp_DB : is_perpendicular E F D B F) 
(hAB_len : distance A B = 6) 
(hAE_len : distance A E = 1) 
: Prop :=
  let DF := distance D F
  let DB := distance D B
  DF * DB = 5

-- Statement of the proof problem
theorem solve_problem : ∀ (O : Type*) [MetricSpace O] [Circle O] 
  (A B C D E F : O) 
  (hAB : is_diameter A B) 
  (hCD_perp_AB : is_perpendicular C D A B E) 
  (hEF_perp_DB : is_perpendicular E F D B F) 
  (hAB_len : distance A B = 6) 
  (hAE_len : distance A E = 1), 
  diameter_perpendicular_to_chord O A B C D E F hAB hCD_perp_AB hEF_perp_DB hAB_len hAE_len :=
by 
  sorry

end solve_problem_l514_514718


namespace sum_function_values_l514_514764

-- Define the function f
def f (x : ℝ) : ℝ := x / (2 * x - 1)

-- State the main theorem
theorem sum_function_values :
  (∑ k in finset.range 4010, f (k / 4011)) = 2005 := by sorry

end sum_function_values_l514_514764


namespace modulus_of_z_l514_514275

open Complex

noncomputable def z : ℂ := (sqrt 3 + I) / ((1 + I)^2)

theorem modulus_of_z : abs z = 1 := by
  sorry

end modulus_of_z_l514_514275


namespace tayzia_tip_l514_514796

theorem tayzia_tip (haircut_women : ℕ) (haircut_children : ℕ) (num_women : ℕ) (num_children : ℕ) (tip_percentage : ℕ) :
  ((num_women * haircut_women + num_children * haircut_children) * tip_percentage / 100) = 24 :=
by
  -- Given conditions
  let haircut_women := 48
  let haircut_children := 36
  let num_women := 1
  let num_children := 2
  let tip_percentage := 20
  -- Perform the calculations as shown in the solution steps
  sorry

end tayzia_tip_l514_514796


namespace range_of_a_l514_514664

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc 1 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0) →
  (a ≤ -2 ∨ a = 1) :=
by
  sorry

end range_of_a_l514_514664


namespace parabola_focal_sum_l514_514019

open Real

/-- Given conditions: 
1. The equation of the parabola x^2 = 12y.
2. The focal point F of the parabola.
3. The line l passing through point P(2, 1) intersects the parabola at points A and B.
4. Point P is the midpoint of the line segment AB.

Prove that the sum of the lengths |AF| + |BF| equals 8.
-/

theorem parabola_focal_sum :
  ∃ (F : ℝ × ℝ) (A B : ℝ × ℝ),
    let P : ℝ × ℝ := (2, 1)
    let parabola := λ (x y : ℝ), x ^ 2 = 12 * y
    P = ((prod.fst A + prod.fst B) / 2, (prod.snd A + prod.snd B) / 2) ∧
    parabola (prod.fst A) (prod.snd A) ∧
    parabola (prod.fst B) (prod.snd B) ∧
    let AF := real.sqrt ((prod.fst A - 0) ^ 2 + (prod.snd A - 3) ^ 2)
    let BF := real.sqrt ((prod.fst B - 0) ^ 2 + (prod.snd B - 3) ^ 2)
    AF + BF = 8 :=
sorry

end parabola_focal_sum_l514_514019


namespace root_probability_l514_514380

-- Definition of the binomial distribution and related probability functions
def binomial_prob (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

-- Condition
def X_binomial_distribution : Prop := (X : ℕ) → binomial_prob 5 (1/2) X ≠ 0

-- Theorem proving the root probability
theorem root_probability {X : ℕ} (hX : X_binomial_distribution) : 
  (∑ k in finset.range 5, binomial_prob 5 (1/2) k) = 31/32 :=
begin
  -- We leave the proof to a more detailed development {specific steps required}
  sorry
end

end root_probability_l514_514380


namespace geometric_series_sum_l514_514476

theorem geometric_series_sum :
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  S = 117775277204 / 30517578125 := by
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  have : S = 117775277204 / 30517578125 := sorry
  exact this

end geometric_series_sum_l514_514476


namespace parallel_lines_iff_equal_diagonals_l514_514358

/-- Let ABCD be a quadrilateral with non-perpendicular diagonals AC and BD.
The sides AB and CD are not parallel. Let O be the intersection of the diagonals AC and BD.
Let H_1 and H_2 be the orthocenters of triangles AOB and COD respectively.
Let M and N be midpoints of segments AB and CD respectively. Then, the lines H_1H_2 and MN
are parallel if and only if AC = BD. -/
theorem parallel_lines_iff_equal_diagonals
  {A B C D O H1 H2 M N : Type*} 
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  [metric_space O] [metric_space H1] [metric_space H2]
  [metric_space M] [metric_space N]
  (H_non_perpendicular_diagonals : ∀ (AC BD : line), AC ⊥ BD → false)
  (H_non_parallel_sides : ∀ (AB CD : line), AB ∥ CD → false)
  (H_O_intersection : ∀ (AC BD : line), O ∈ AC ∧ O ∈ BD)
  (H_H1_orthocenter : ∀ (AOB : triangle), orthocenter AOB = H1)
  (H_H2_orthocenter : ∀ (COD : triangle), orthocenter COD = H2)
  (H_M_midpoint : midpoint A B = M)
  (H_N_midpoint : midpoint C D = N) :
  parallel (line_through H1 H2) (line_through M N) ↔ dist A C = dist B D :=
sorry

end parallel_lines_iff_equal_diagonals_l514_514358


namespace probability_arithmetic_sequence_l514_514298

open Finset

def combinations {α : Type} [DecidableEq α] (s : Finset α) (k : ℕ) : Finset (Finset α) :=
  (Fintype.piFinset (λ _ : Finₓ k, s)).filter (λ t, t.card = k)

theorem probability_arithmetic_sequence :
  let s := {1, 2, 3, 4, 5, 6}
  let n := (combinations s 3).card
  let favorable := { {1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}, {1, 3, 5}, {2, 4, 6} }
  (n = 20) → (favorable.card = 6) → ((favorable.card : ℚ) / n = 3 / 10) := 
by
  intros s n favorable h1 h2
  sorry

end probability_arithmetic_sequence_l514_514298


namespace ratio_of_games_played_to_losses_l514_514100

-- Conditions
def games_played : ℕ := 10
def games_won : ℕ := 5
def games_lost : ℕ := games_played - games_won

-- Prove the ratio of games played to games lost is 2:1
theorem ratio_of_games_played_to_losses
  (h_played : games_played = 10)
  (h_won : games_won = 5) :
  (games_played / Nat.gcd games_played games_lost : ℕ) /
  (games_lost / Nat.gcd games_played games_lost : ℕ) = 2 / 1 :=
by
  sorry

end ratio_of_games_played_to_losses_l514_514100


namespace cost_of_fencing_l514_514829

/-- The sides of a rectangular field are in the ratio 3:4.
If the area of the field is 10092 sq. m and the cost of fencing the field is 25 paise per meter,
then the cost of fencing the field is 101.5 rupees. --/
theorem cost_of_fencing (area : ℕ) (fencing_cost : ℝ) (ratio1 ratio2 perimeter : ℝ)
  (h_area : area = 10092)
  (h_ratio : ratio1 = 3 ∧ ratio2 = 4)
  (h_fencing_cost : fencing_cost = 0.25)
  (h_perimeter : perimeter = 406) :
  perimeter * fencing_cost = 101.5 := by
  sorry

end cost_of_fencing_l514_514829


namespace inequality_solution_set_l514_514279

theorem inequality_solution_set (f : ℝ → ℝ) (h_deriv : ∀ x : ℝ, (f' x) / 2 - f x > 2) (h_initial : f 0 = -1) :
  {x : ℝ | (f x + 2) / (Real.exp (2 * x)) > 1} = set.Ioi 0 :=
by
  -- Sorry: Proof is omitted as instructed
  sorry

end inequality_solution_set_l514_514279


namespace ernaldo_friends_count_l514_514494

-- Define the members of the group
inductive Member
| Arnaldo
| Bernaldo
| Cernaldo
| Dernaldo
| Ernaldo

open Member

-- Define the number of friends for each member
def number_of_friends : Member → ℕ
| Arnaldo  => 1
| Bernaldo => 2
| Cernaldo => 3
| Dernaldo => 4
| Ernaldo  => 0  -- This will be our unknown to solve

-- The main theorem we need to prove
theorem ernaldo_friends_count : number_of_friends Ernaldo = 2 :=
sorry

end ernaldo_friends_count_l514_514494


namespace hermione_max_profit_l514_514711

def TC (Q : ℝ) : ℝ := 5 * Q^2

def demand_ws (P : ℝ) : ℝ := 26 - 2 * P
def demand_s (P : ℝ) : ℝ := 10 - P

noncomputable def max_profit : ℝ := 7.69

theorem hermione_max_profit :
  ∃ P Q, (P > 0 ∧ Q > 0) ∧ (Q = demand_ws P + demand_s P) ∧
  (P * Q - TC Q = max_profit) := sorry

end hermione_max_profit_l514_514711


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514898

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514898


namespace geometric_sequence_properties_l514_514255

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (a_geom : ∃ q : ℝ, ∀ n, a (n+1) = a n * q)
  (a_pos : ∃ q > 0, ∀ n, a (n+1) = a n * q)
  (a4_eq_a2a3 : a 4 = a 2 * a 3)
  (sum_S3 : a 1 + a 2 + a 3 = 13)
  (b_def : ∀ n, b n = log 3 (a n) + n)
  :
  (∀ n, a n = 3^(n-1)) ∧
  (∀ n, ∑ i in range n, 1 / (b i * b (i+1)) = n / (2 * n + 1)) :=
by
  sorry

end geometric_sequence_properties_l514_514255


namespace equal_pair_l514_514140

theorem equal_pair (a b c d : ℤ) : (a = 9, b = 9, c ≠ 4, d ≠ -4) ↔ ((-3)^2 = sqrt 81) := by 
  sorry

end equal_pair_l514_514140


namespace max_grid_crossings_impossible_l514_514727

theorem max_grid_crossings_impossible (m n : ℕ) (hm : m = 20) (hn : n = 30) :
  ¬ ∃ k : ℕ, k = 50 ∧ (m + n - Nat.gcd m n) ≥ k :=
by
  -- Declaration of the necessary assumptions and results.
  rw [hm, hn]
  have gcd_20_30 : Nat.gcd 20 30 = 10 := by sorry
  simulate_algebra
  exact λ h, by sorry

end max_grid_crossings_impossible_l514_514727


namespace square_of_chord_length_is_39804_l514_514181

noncomputable def square_of_chord_length (r4 r8 r12 : ℝ) (externally_tangent : (r4 + r8) < r12) : ℝ := 
  let r4 := 4
  let r8 := 8
  let r12 := 12
  let PQ_sq := 4 * ((r12^2) - ((2 * r8 + 1 * r4) / 3)^2) in
  PQ_sq

theorem square_of_chord_length_is_39804 : 
  square_of_chord_length 4 8 12 ((4 + 8) < 12) = 398.04 := 
by
  sorry

end square_of_chord_length_is_39804_l514_514181


namespace compare_a_b_c_l514_514249

noncomputable def a : ℝ := (1 / 3)^(1 / 3)
noncomputable def b : ℝ := Real.log (1 / 2)
noncomputable def c : ℝ := Real.logb (1 / 3) (1 / 4)

theorem compare_a_b_c : b < a ∧ a < c := by
  sorry

end compare_a_b_c_l514_514249


namespace wedge_volume_is_216pi_l514_514502

/-- Define the diameter of the cylindrical log -/
def log_diameter : ℝ := 12

/-- Define the radius of the cylindrical log (half of the diameter) -/
def log_radius : ℝ := log_diameter / 2

/-- Define the height of the cylindrical log -/
def log_height : ℝ := log_diameter

/-- The volume of the whole cylinder -/
def cylinder_volume : ℝ := π * (log_radius ^ 2) * log_height

/-- The volume of the wedge, which is half of the cylinder's volume -/
def wedge_volume : ℝ := cylinder_volume / 2

/-- n is the integer part of the wedge volume divided by π -/
def n : ℤ := (wedge_volume / π).toInt

theorem wedge_volume_is_216pi : n = 216 :=
by
  sorry

end wedge_volume_is_216pi_l514_514502


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514900

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514900


namespace equilateral_triangle_ratio_l514_514966

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514966


namespace equilateral_triangle_ratio_l514_514881

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514881


namespace equilateral_triangle_ratio_l514_514933

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514933


namespace adult_ticket_price_is_3_l514_514097

-- Define the conditions
def total_seats : ℕ := 200
def child_ticket_price : ℕ := 1.5
def total_income : ℕ := 510
def num_children : ℕ := 60

-- Define the number of adults and total income produced by tickets
def num_adults (total_seats num_children : ℕ) : ℕ := total_seats - num_children
def total_child_income (num_children : ℕ) (child_ticket_price : ℕ) : ℕ := num_children * child_ticket_price
def total_adult_income (num_adults : ℕ) (adult_ticket_price : ℕ) : ℕ := num_adults * adult_ticket_price

-- Statement to prove
theorem adult_ticket_price_is_3
  (num_adults total_seats num_children : ℕ)
  (total_child_income total_income : ℕ)
  (h1 : num_adults = total_seats - num_children)
  (h2 : total_child_income = num_children * child_ticket_price)
  (h3 : total_income = total_child_income + total_adult_income)
  : ∃ (A : ℕ), total_adult_income = num_adults * A ∧ A = 3 :=
sorry

end adult_ticket_price_is_3_l514_514097


namespace find_quadratic_function_l514_514016

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant condition
def discriminant_zero (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

-- Given derivative
def given_derivative (x : ℝ) : ℝ := 2 * x + 2

-- Prove that if these conditions hold, then f(x) = x^2 + 2x + 1
theorem find_quadratic_function :
  ∃ (a b c : ℝ), (∀ x, quadratic_function a b c x = 0 → discriminant_zero a b c) ∧
                (∀ x, (2 * a * x + b) = given_derivative x) ∧
                (quadratic_function a b c x = x^2 + 2 * x + 1) := 
by
  sorry

end find_quadratic_function_l514_514016


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514884

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514884


namespace intersection_of_altitudes_is_midpoint_l514_514141

variables {Ω : Type*} [metric_space Ω]
variables {A B C P D E M : Ω}
variables (circ : circle Ω)
variables (triangle : set Ω)

# Conditions
assume (hABC_acute : acute_triangle A B C)
assume (hABC_inscribed : triangle_in_circle triangle A B C circ)
assume (hTangents : tangent_to_circle circ B P ∧ tangent_to_circle circ C P)
assume (hD_perpendicular : perpendicular_from_point_to_line P A B D)
assume (hE_perpendicular : perpendicular_from_point_to_line P A C E)

-- Question to prove that M is the midpoint of segment BC
theorem intersection_of_altitudes_is_midpoint:
  ∃ (M : Ω), M = midpoint B C ∧ orthocenter_of_triangle A D E M := sorry

end intersection_of_altitudes_is_midpoint_l514_514141


namespace correct_average_of_10_numbers_l514_514993

theorem correct_average_of_10_numbers
  (incorrect_avg : ℕ)
  (n : ℕ)
  (incorrect_read : ℕ)
  (correct_read : ℕ)
  (incorrect_total_sum : ℕ) :
  incorrect_avg = 19 →
  n = 10 →
  incorrect_read = 26 →
  correct_read = 76 →
  incorrect_total_sum = incorrect_avg * n →
  (correct_total_sum : ℕ) = incorrect_total_sum - incorrect_read + correct_read →
  (correct_avg : ℕ) = correct_total_sum / n →
  correct_avg = 24 :=
by
  intros
  sorry

end correct_average_of_10_numbers_l514_514993


namespace parameterization_function_l514_514432

theorem parameterization_function (f : ℝ → ℝ) :
  (∀ t : ℝ, ∃ x y : ℝ, (x, y) = (f(t), 20t - 10) ∧ y = 2x - 30) ↔ (f(t) = 10t + 10) :=
by
  sorry

end parameterization_function_l514_514432


namespace purely_imaginary_condition_l514_514368

theorem purely_imaginary_condition (x y : ℝ) :
  ((x = 0) → (z = x + y * I) → (z.im ≠ 0)) ∧
  ((z = x + y * I) → (z.im ≠ 0 → x = 0)) :=
by
  sorry

end purely_imaginary_condition_l514_514368


namespace correct_option_is_A_l514_514483

-- Define the various options as propositions
def OptionA : Prop := ∃ x : ℕ, PRINT 4 * x
def OptionB : Prop := ∃ var : ℕ, INPUT
def OptionC : Prop := INPUTB = 3
def OptionD : Prop := PRINT (λ y : ℕ, y = 2 * x + 1)

-- Define the correctness of each option
def OptionA_is_correct : Prop := OptionA
def OptionB_is_correct : Prop := ¬ OptionB
def OptionC_is_correct : Prop := ¬ OptionC
def OptionD_is_correct : Prop := ¬ OptionD

-- Define the main statement to prove
theorem correct_option_is_A : OptionA_is_correct ∧ OptionB_is_correct ∧ OptionC_is_correct ∧ OptionD_is_correct := by
  -- Proof here
  sorry

end correct_option_is_A_l514_514483


namespace train_passes_man_in_6_seconds_l514_514126

theorem train_passes_man_in_6_seconds:
  ∀ (L : ℝ) (v_train : ℝ) (v_man : ℝ),
    L = 110 ∧ v_train = 60 ∧ v_man = 6 ∧ (opposite_direction : true) →
    (L / ((v_train + v_man) * (5 / 18)) ≈ 6) :=
by
  intros L v_train v_man h
  cases' h with H_length H_speeds
  cases' H_speeds with H_vt H_vm
  cases' H_vm with H_vman today
  rw [H_length, H_vt, H_vman]
  have : 60 + 6 = 66 := rfl
  rw [add_comm 60 6]
  have : 66 * (5 / 18) ≈ 18.333 := by norm_num
  rw this
  have : 110 / 18.333 ≈ 6 := by norm_num
  exact this
  sorry

end train_passes_man_in_6_seconds_l514_514126


namespace cafe_table_count_l514_514026

theorem cafe_table_count (cafe_seats_base7 : ℕ) (seats_per_table : ℕ) (cafe_seats_base10 : ℕ)
    (h1 : cafe_seats_base7 = 3 * 7^2 + 1 * 7^1 + 2 * 7^0) 
    (h2 : seats_per_table = 3) : cafe_seats_base10 = 156 ∧ (cafe_seats_base10 / seats_per_table) = 52 := 
by {
  sorry
}

end cafe_table_count_l514_514026


namespace equilateral_triangle_ratio_l514_514873

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514873


namespace range_of_a_l514_514281

-- Define the decreasing condition of the function f on the interval (-1,1)
def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) := ∀ x y ∈ I, x < y → f y ≤ f x

-- Define the properties given in the problem
def conditions (f : ℝ → ℝ) : Prop :=
  is_decreasing_on f (set.Ioo (-1:ℝ) (1:ℝ)) ∧ ∀ a : ℝ, f (2 * a - 1) < f (1 - a)

-- The main theorem stating the range of a
theorem range_of_a (f : ℝ → ℝ) (h : conditions f) : ∀ a : ℝ, (f (2 * a - 1) < f (1 - a)) → (2/3 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l514_514281


namespace cos_neg_pi_six_val_l514_514553

noncomputable theory -- Add this to handle any non-computable aspects

-- Define the constants for the angles and their cosine values
def neg_pi_six : Real := -Real.pi / 6
def pi_six : Real := Real.pi / 6
def sqrt_three_over_two : Real := Real.sqrt 3 / 2

-- State the conditions
axiom cos_neg_pi_six_eq_cos_pi_six : Real.cos neg_pi_six = Real.cos pi_six
axiom cos_pi_six_val : Real.cos pi_six = sqrt_three_over_two

-- Prove the statement
theorem cos_neg_pi_six_val : Real.cos neg_pi_six = sqrt_three_over_two :=
by sorry

end cos_neg_pi_six_val_l514_514553


namespace initial_girls_count_l514_514246

theorem initial_girls_count (b g : ℕ) 
    (h1 : b = 3 * (g - 20)) 
    (h2 : 4 * (b - 60) = g - 20) :
    ¬ ∃ n : ℕ, g = n ∧ g ≈ 42 :=
by
  sorry

end initial_girls_count_l514_514246


namespace sum_of_slopes_eq_zero_l514_514627

theorem sum_of_slopes_eq_zero
  (p : ℝ) (a : ℝ) (hp : p > 0) (ha : a > 0)
  (P Q : ℝ × ℝ)
  (hP : P.2 ^ 2 = 2 * p * P.1)
  (hQ : Q.2 ^ 2 = 2 * p * Q.1)
  (hcollinear : ∃ m : ℝ, ∀ (x y : (ℝ × ℝ)), y = P ∨ y = Q ∨ y = (-a, 0) → y.2 = m * (y.1 + a)) :
  let k_AP := (P.2) / (P.1 - a)
  let k_AQ := (Q.2) / (Q.1 - a)
  k_AP + k_AQ = 0 := by
    sorry

end sum_of_slopes_eq_zero_l514_514627


namespace symmetry_xOz_A_l514_514724

-- Define the symmetry transformation
def symmetry_xOz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, p.3)

-- Define point A
def A : ℝ × ℝ × ℝ := (9, 8, 5)

-- State the theorem to prove the symmetry property
theorem symmetry_xOz_A : symmetry_xOz A = (9, -8, 5) := 
  by
    -- Skip the proof steps using sorry
    sorry

end symmetry_xOz_A_l514_514724


namespace repeating_decimals_for_n_div_18_l514_514598

theorem repeating_decimals_for_n_div_18 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → (¬ (∃ m : ℕ, m * 18 = n * (2^k * 5^l) ∧ 0 < k ∧ 0 < l)) :=
by
  sorry

end repeating_decimals_for_n_div_18_l514_514598


namespace trigonometric_sum_property_l514_514778

theorem trigonometric_sum_property (m n p : ℤ) :
  (∑ k in finset.range (p-1), real.sin (k * m * real.pi / p) * real.sin (k * n * real.pi / p)) =
  if 2 * p ∣ (m + n) ∧ ¬ (2 * p ∣ (m - n)) then 
    -p / 2
  else if 2 * p ∣ (m - n) ∧ ¬ (2 * p ∣ (m + n)) then 
    p / 2
  else 
    0 := 
sorry

end trigonometric_sum_property_l514_514778


namespace compare_log_values_l514_514621

-- Define the logarithmic values
noncomputable def a : ℝ := Real.log e / Real.log 4
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log 5 / Real.log 4

-- Prove the inequality
theorem compare_log_values : a < c ∧ c < b := by
  sorry

end compare_log_values_l514_514621


namespace tan_cot_value_l514_514018

-- Define the main constraint
axiom sec_csc_condition (θ : Real) : (1 + Real.sec θ) * (1 + Real.csc θ) = 6

-- Define the target expression
noncomputable def target_expression (θ : Real) : Real := 
  (1 + Real.tan θ) * (1 + Real.cot θ)

-- Formulate the theorem that we need to prove
theorem tan_cot_value (θ : Real) (h : (1 + Real.sec θ) * (1 + Real.csc θ) = 6) : 
  target_expression θ = 49 / 12 :=
sorry

end tan_cot_value_l514_514018


namespace sum_place_values_63130_l514_514972

theorem sum_place_values_63130 : 
  let num := 63130
  let hundreds_place_value := 3 * 100
  let tens_place_value := 3 * 10
  in hundreds_place_value + tens_place_value = 330 := 
by
  sorry

end sum_place_values_63130_l514_514972


namespace remainder_of_98_pow_50_mod_50_l514_514076

theorem remainder_of_98_pow_50_mod_50
  (h1 : 98 ≡ -2 [MOD 50])
  (h2 : (-2) ^ 50 = 2 ^ 50)
  (h3 : 32 ≡ -18 [MOD 50])
  (h4 : 18 ^ 2 ≡ 24 [MOD 50])
  (h5 : 18 ^ 4 ≡ 26 [MOD 50])
  (h6 : 18 ^ 8 ≡ 26 [MOD 50])
  (h7 : 18 ^ 10 ≡ 24 [MOD 50]) :
  98 ^ 50 ≡ 24 [MOD 50] := 
sorry

end remainder_of_98_pow_50_mod_50_l514_514076


namespace mul_decimal_l514_514205

theorem mul_decimal (h1 : 0.5 = 5 * 10⁻¹) (h2 : 0.3 = 3 * 10⁻¹) : 
  0.5 * 0.3 = 0.15 :=
by
  -- Lean proof will go here
  sorry

end mul_decimal_l514_514205


namespace find_g100_zero_values_l514_514753

def g_0 (x : ℝ) : ℝ := 
  if x >= 50 then 3 * x 
  else if x >= -50 then 100 
  else -x

def g : ℕ → ℝ → ℝ
| 0, x := g_0 x
| (n + 1), x := abs (g n x) - 2

theorem find_g100_zero_values :
  let zero_count := (Finset.filter (λ x : ℤ, g 100 (x : ℝ) = 0) (Finset.range 1000)).card in
  zero_count = 429 := sorry

end find_g100_zero_values_l514_514753


namespace Amelia_wins_probability_l514_514139

-- Define Amelia's and Blaine's coin probabilities
def p_A : ℚ := 2/7
def p_B : ℚ := 1/3

-- Probability Amelia gets heads twice in one turn
def P_A_double_heads : ℚ := p_A * p_A

-- Probability Blaine gets heads twice in one turn
def P_B_double_heads : ℚ := p_B * p_B

-- Probability Amelia doesn't get heads twice in one turn
def P_A_not_double_heads : ℚ := 1 - P_A_double_heads

-- Probability Blaine doesn't get heads twice in one turn
def P_B_not_double_heads : ℚ := 1 - P_B_double_heads

-- Combined probability that neither Amelia nor Blaine get double heads in one complete round
def P_round_neither_double_heads : ℚ := P_A_not_double_heads * P_B_not_double_heads

-- Use of geometric series to find the total probability Amelia eventually wins
-- Recall that the sum of an infinite series a + ar + ar^2 + ... is a / (1 - r)
def P_Amelia_wins : ℚ := P_A_double_heads / (1 - P_round_neither_double_heads)

theorem Amelia_wins_probability : P_Amelia_wins = 4/9 :=
by sorry

end Amelia_wins_probability_l514_514139


namespace cube_surface_area_eq_prism_volume_l514_514116

theorem cube_surface_area_eq_prism_volume 
  (l w h : ℝ)
  (h1 : l = 10) 
  (h2 : w = 5) 
  (h3 : h = 20)
  : let V := l * w * h
    let s := real.cbrt V
    let surface_area := 6 * s^2
  in surface_area = 600 :=
by 
  sorry

end cube_surface_area_eq_prism_volume_l514_514116


namespace prove_ellipse_prove_line_AC_l514_514630

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
    (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) ∧ 
    (∀ (x y : ℝ), (y^2 = 4 * x)) ∧ 
    ∃(M : ℝ × ℝ), M.1 > 0 ∧ M.2 > 0 ∧ M.1 + 1 = 5 / 3 ∧ (|M.1 - 1|^2 + M.2^2)^0.5 = 5 / 3

noncomputable def line_AC_equation : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧
    (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) ∧ 
    (∀ (x y : ℝ), (y^2 = 4 * x)) ∧ 
    ∃ (A C: ℝ × ℝ), (7 * A.1 - 7 * A.2 + 1 = 0) ∧ (7 * C.1 - 7 * C.2 + 1 = 0) ∧
    (A.1 + A.2 = 4/3) ∧ (C.1 + C.2 = 4/3) ∧ (A.1 + C.1 + A.2 + C.2 = 2 * (4/3)) ∧ 
    (A.1 + C.1 = 6/7) ∧ ((A.2 + C.2) = 3/7) ∧
    ∀ (a : ℝ),  7 * A.1 - 7 * A.2 + 1 = 0 ↔ (A.1 + A.2 + 1 = 0)

theorem prove_ellipse : ellipse_equation := sorry

theorem prove_line_AC : line_AC_equation := sorry

end prove_ellipse_prove_line_AC_l514_514630


namespace isosceles_triangle_AUV_l514_514726

variable {A B C X Y U V : Type}
variables [IncidenceGeo A B C X Y U V] [Triangle A B C] [Line BC]

-- Assuming the given conditions
variables (BX CY AB AC : ℝ)
variable (H : BX * AC = CY * AB)

-- Circumcenters of ΔACX and ΔABY
variable {O₁ : Type} [Circumcenter A C X O₁]
variable {O₂ : Type} [Circumcenter A B Y O₂]

-- Points of intersection
variable (U : BC.Intersect O₁O₂)
variable (V : BC.Intersect O₁O₂)

-- Given that O₁ and O₂ are the circumcenters and line O₁O₂ intersects AB and AC at points U and V respectively
theorem isosceles_triangle_AUV : IsIsosceles Δ A U V :=
  sorry

end isosceles_triangle_AUV_l514_514726


namespace ratio_area_perimeter_eq_triangle_side_length_six_l514_514897

theorem ratio_area_perimeter_eq_triangle_side_length_six :
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  (area / perimeter) = Real.sqrt 3 / 2 :=
by 
  let s := 6
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  have h : (area / perimeter) = Real.sqrt 3 / 2 := sorry
  exact h

end ratio_area_perimeter_eq_triangle_side_length_six_l514_514897


namespace neg_3_14_gt_neg_pi_l514_514567

theorem neg_3_14_gt_neg_pi (π : ℝ) (h : 0 < π) : -3.14 > -π := 
sorry

end neg_3_14_gt_neg_pi_l514_514567


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514951

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514951


namespace equilateral_triangle_ratio_l514_514865

def side_length : ℝ := 6
def area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4
def perimeter (s : ℝ) : ℝ := 3 * s
def ratio (s : ℝ) : ℝ := (area s) / (perimeter s)

theorem equilateral_triangle_ratio :
  ratio side_length = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514865


namespace find_largest_n_l514_514544

noncomputable def largest_n_with_triangle_property (n : ℕ) : Prop :=
  ∀ (S : Finset ℕ), S ⊆ Finset.range (n + 3) ∧ S.card = 8 →
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a < b → b < c → c < a + b → false

theorem find_largest_n : largest_n_with_triangle_property 75 :=
by
  unfold largest_n_with_triangle_property
  sorry -- Proof part is omitted

end find_largest_n_l514_514544


namespace math_problem_l514_514538

noncomputable def power_zero (a : ℝ) (h : a ≠ 0) : ℝ := a ^ 0

noncomputable def reciprocal (a : ℝ) (h : a ≠ 0) : ℝ := a⁻¹

noncomputable def absolute_value (a : ℝ) : ℝ := abs a

theorem math_problem : 
  let term1 := power_zero (sqrt 5 - 1) (by norm_num)
  let term2 := reciprocal 3 (by norm_num)
  let term3 := absolute_value (-1/3)
  term1 + term2 - term3 = 1 :=
by
  let term1 := power_zero (sqrt 5 - 1) (by norm_num)
  let term2 := reciprocal 3 (by norm_num)
  let term3 := absolute_value (-1/3)
  have h1 : term1 = 1 := by { unfold power_zero, norm_num }
  have h2 : term2 = 1/3 := by { unfold reciprocal, norm_num }
  have h3 : term3 = 1/3 := by { unfold absolute_value, norm_num }
  calc
    term1 + term2 - term3 = 1 + 1/3 - 1/3 : by rw [h1, h2, h3]
                      ... = 1 : by norm_num

end math_problem_l514_514538


namespace eunice_pots_l514_514565

theorem eunice_pots (total_seeds pots_with_3_seeds last_pot_seeds : ℕ)
  (h1 : total_seeds = 10)
  (h2 : pots_with_3_seeds * 3 + last_pot_seeds = total_seeds)
  (h3 : last_pot_seeds = 1) : pots_with_3_seeds + 1 = 4 :=
by
  -- Proof omitted
  sorry

end eunice_pots_l514_514565


namespace set_intersection_complement_l514_514094

def P := {0, 1, 2}
def N := {x : ℝ | x ^ 2 - 3 * x + 2 = 0}

theorem set_intersection_complement :
  P ∩ {x : ℝ | x ∉ N} = {0} :=
sorry

end set_intersection_complement_l514_514094


namespace regular_polygon_sides_l514_514197

theorem regular_polygon_sides (h : ∀ n : ℕ, 140 * n = 180 * (n - 2)) : n = 9 :=
sorry

end regular_polygon_sides_l514_514197


namespace sum_of_coordinates_l514_514777

theorem sum_of_coordinates (x : ℚ) : (0, 0) = (0, 0) ∧ (x, -3) = (x, -3) ∧ ((-3 - 0) / (x - 0) = 4 / 5) → x - 3 = -27 / 4 := 
sorry

end sum_of_coordinates_l514_514777


namespace shadow_building_length_l514_514104

-- Definitions based on conditions
def height_flagpole : ℚ := 18
def shadow_flagpole : ℚ := 45
def height_building : ℚ := 24

-- Question to be proved
theorem shadow_building_length : 
  ∃ (shadow_building : ℚ), 
    (height_flagpole / shadow_flagpole = height_building / shadow_building) ∧ 
    shadow_building = 60 :=
by
  sorry

end shadow_building_length_l514_514104


namespace equilateral_triangle_ratio_l514_514879

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514879


namespace largest_divisor_of_expression_l514_514314

theorem largest_divisor_of_expression (x : ℤ) (h_even : x % 2 = 0) :
  ∃ k, (∀ x, x % 2 = 0 → k ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) ∧ 
       (∀ m, (∀ x, x % 2 = 0 → m ∣ (10 * x + 4) * (10 * x + 8) * (5 * x + 2)) → m ≤ k) ∧ 
       k = 32 :=
sorry

end largest_divisor_of_expression_l514_514314


namespace not_all_squares_congruent_l514_514485

-- Define what it means to be a square
structure Square :=
  (side : ℝ)
  (angle : ℝ)
  (is_square : side > 0 ∧ angle = 90)

-- Define congruency of squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side = s2.side ∧ s1.angle = s2.angle

-- The main statement to prove 
theorem not_all_squares_congruent : ∃ s1 s2 : Square, ¬ congruent s1 s2 :=
by
  sorry

end not_all_squares_congruent_l514_514485


namespace maximize_play_time_l514_514387

theorem maximize_play_time :
  let weekly_pay := 100
  let half_weekly_pay := weekly_pay / 2
  let arcade_snacks_cost := 5
  let remaining_budget := half_weekly_pay - arcade_snacks_cost
  let bundleA_cost := 25
  let bundleB_cost := 45
  let bundleC_cost := 60
  let bundleB_playtime := 3 -- hours

  remaining_budget = 45 →
  bundleB_cost ≤ remaining_budget →
  bundleB_playtime * 60 = 180 :=
by {
  let weekly_pay := 100
  let half_weekly_pay := weekly_pay / 2
  let arcade_snacks_cost := 5
  let remaining_budget := half_weekly_pay - arcade_snacks_cost
  let bundleA_cost := 25
  let bundleB_cost := 45
  let bundleC_cost := 60
  let bundleB_playtime := 3 -- hours
  
  intros h1 h2
  have h3 : remaining_budget = 45 := h1
  have h4 : bundleB_cost ≤ 45 := h2
  exact calc
    bundleB_playtime * 60 = 3 * 60 : by rfl
                       ... = 180   : by rfl
}

end maximize_play_time_l514_514387


namespace number_of_elements_in_A_inter_Z_l514_514300

noncomputable def A : Set ℝ := { x | x^2 < 3 * x + 4 }

theorem number_of_elements_in_A_inter_Z : (A ∩ Set.univ).toFinset.card = 4 := 
  sorry

end number_of_elements_in_A_inter_Z_l514_514300


namespace probability_f_ge_0_in_interval_is_1_4_l514_514765

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

theorem probability_f_ge_0_in_interval_is_1_4 : 
  ∀ x0 ∈ set.Icc (2 : ℝ) 6, (f x0) >= 0 → ∃ P : ℝ, P = 1/4 :=
by
  sorry

end probability_f_ge_0_in_interval_is_1_4_l514_514765


namespace area_is_12_l514_514710

-- Definitions based on conditions
def isosceles_triangle (a b m : ℝ) : Prop :=
  a = b ∧ m > 0 ∧ a > 0

def median (height base_length : ℝ) : Prop :=
  height > 0 ∧ base_length > 0

noncomputable def area_of_isosceles_triangle_with_given_median (a m : ℝ) : ℝ :=
  let base_half := Real.sqrt (a^2 - m^2)
  let base := 2 * base_half
  (1 / 2) * base * m

-- Prove that the area of the isosceles triangle is correct given conditions
theorem area_is_12 :
  ∀ (a m : ℝ), isosceles_triangle a a m → median m (2 * Real.sqrt (a^2 - m^2)) → area_of_isosceles_triangle_with_given_median a m = 12 := 
by
  intros a m hiso hmed
  sorry  -- Proof steps are omitted

end area_is_12_l514_514710


namespace real_return_l514_514394

theorem real_return (n i r: ℝ) (h₁ : n = 0.21) (h₂ : i = 0.10) : 
  (1 + r) = (1 + n) / (1 + i) → r = 0.10 :=
by
  intro h₃
  sorry

end real_return_l514_514394


namespace floor_plus_self_eq_l514_514215

theorem floor_plus_self_eq (r : ℝ) (h : ⌊r⌋ + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_self_eq_l514_514215


namespace collinear_K_L_E_l514_514359

open Set

variables {α : Type*}
variables [TopologicalSpace α]
variables {ω : Set α} -- circumcircle of Δ ABC
variable {A B C D E F K L : α} -- points mentioned in the problem
variable [Is_connected ω] -- assuming ω is connected (implied by circumcircle)
variable (hAC : |A - C| = |AB|) -- |AB| = |AC|
variable (hD : D ∈ ω ∧ D ≠ A ∧ D ≠ C) -- D is on minor arc AC
variable (hE : ∃ R : Line α, R ∧ R.contains A ∧ R.contains D ∧ B = reflection E R) -- E is reflection of B in AD
variable (hF : F ∈ ω ∧ F ≠ B ∧ F ≠ E ∧ lies_on Line B E) -- F is on ω and line BE
variable (hK : K ∈ Line A C ∧ tangent_at_circumcircle ω ∧ F ∈ tangent ω) -- K is intersection of AC and tangent at F
variable (hL : intersection (Line A B) Line F D) -- L is intersection of AB and FD

theorem collinear_K_L_E : Collinear ({K, L, E}) :=
sorry

end collinear_K_L_E_l514_514359


namespace determinant_of_matrix_l514_514377

noncomputable def determinant_matrix (a b c : ℂ) : ℂ :=
  a * (b * c - a^2) - c * (c^2 - a * b) + b * (c * b - a * c)

theorem determinant_of_matrix
  (a b c p q r : ℂ)
  (h_roots : ∀ x, (x - a) * (x - b) * (x - c) = x^3 + p * x^2 + q * x + r) :
  determinant_matrix a b c = -c^3 + b^2c :=
by
  -- Proof goes here.
  sorry

end determinant_of_matrix_l514_514377


namespace sum_first_100_terms_l514_514431

-- Define the sequence {a_n}
def a (n : ℕ) : ℤ := (-1)^(n-1) * (4 * n - 3)

-- Define S_100
def S_100 : ℤ := ∑ n in Finset.range 100, a (n + 1)

-- State the theorem
theorem sum_first_100_terms : S_100 = -200 :=
by
  -- Proof goes here
  sorry

end sum_first_100_terms_l514_514431


namespace largest_base_for_digit_sum_not_equal_l514_514574

def sum_of_digits (n : ℕ) (b : ℕ) : ℕ :=
  nat.digits b n |> list.sum

theorem largest_base_for_digit_sum_not_equal (b : ℕ)
  (h1 : b ≥ 7) (h2 : ∀ k, k > b → sum_of_digits ((b + 2)^4) k ≠ 32) :
  sum_of_digits ((b + 2)^4) b ≠ 32 :=
begin
  sorry
end

end largest_base_for_digit_sum_not_equal_l514_514574


namespace num_repeating_decimals_between_1_and_20_l514_514594

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end num_repeating_decimals_between_1_and_20_l514_514594


namespace find_genuine_coin_in_three_weighings_l514_514835

theorem find_genuine_coin_in_three_weighings (coins : Fin 15 → ℝ)
  (even_number_of_counterfeit : ∃ n : ℕ, 2 * n < 15 ∧ (∀ i, coins i = 1) ∨ (∃ j, coins j = 0.5)) : 
  ∃ i, coins i = 1 :=
by sorry

end find_genuine_coin_in_three_weighings_l514_514835


namespace at_least_one_not_less_than_l514_514271

variables {A B C D a b c : ℝ}

theorem at_least_one_not_less_than :
  (a = A * C) →
  (b = A * D + B * C) →
  (c = B * D) →
  (a + b + c = (A + B) * (C + D)) →
  a ≥ (4 * (A + B) * (C + D) / 9) ∨ b ≥ (4 * (A + B) * (C + D) / 9) ∨ c ≥ (4 * (A + B) * (C + D) / 9) :=
by
  intro h1 h2 h3 h4
  sorry

end at_least_one_not_less_than_l514_514271


namespace cover_ways_of_2x6_grid_l514_514069

theorem cover_ways_of_2x6_grid :
  let ways_to_cover_grid : ℕ :=
    large_tiles * small_tiles :=
     1
  in
  ways_to_cover_grid = 45 :=
begin
  sorry
end

end cover_ways_of_2x6_grid_l514_514069


namespace smallest_n_for_g_n_eq_4_l514_514748

/-- 
  Let g(n) be the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n.
  Prove that the smallest positive integer n for which g(n) = 4 is 25.
-/
def g (n : ℕ) : ℕ :=
  (finset.univ.product finset.univ).filter (λ (ab : ℕ × ℕ), ab.1 ^ 2 + ab.2 ^ 2 = n ∧ ab.1 ≠ ab.2).card

theorem smallest_n_for_g_n_eq_4 :
  ∃ n : ℕ, g n = 4 ∧ (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 25
  sorry

end smallest_n_for_g_n_eq_4_l514_514748


namespace money_problem_l514_514466

variable {c d : ℝ}

theorem money_problem (h1 : 3 * c - 2 * d < 30) (h2 : 4 * c + d = 60) : 
  c < 150 / 11 ∧ d > 60 / 11 := 
by 
  sorry

end money_problem_l514_514466


namespace circle_chord_square_length_l514_514176

-- Define the problem in terms of Lean constructs
theorem circle_chord_square_length :
  ∃ (O₄ O₈ O₁₂ : ℝ) (A₄ A₈ A₁₂ : ℝ) (PQ : ℝ),
    O₄ = 4 ∧
    O₈ = 8 ∧
    O₁₂ = 12 ∧
    A₈ = 8 ∧
    A₄ = 4 ∧
    A₁₂ = (2 * A₈ + A₄) / 3 ∧
    PQ^2 = 4 * (O₁₂^2 - A₁₂^2) → 
    PQ^2 = 3584 / 9 :=
by
  -- Define variables based on conditions
  let O₄ := 4
  let O₈ := 8
  let O₁₂ := 12
  let A₈ := 8
  let A₄ := 4
  let A₁₂ := (2 * A₈ + A₄) / 3
  let s : ℝ := O₁₂^2 - A₁₂^2
  let PQ := real.sqrt(4 * s)

  -- Using given values to verify
  have h1 : O₄ = 4 := rfl
  have h2 : O₈ = 8 := rfl
  have h3 : O₁₂ = 12 := rfl
  have h4 : A₈ = 8 := rfl
  have h5 : A₄ = 4 := rfl
  have h6 : A₁₂ = (2 * A₈ + A₄) / 3 := rfl

  -- Calculate parts
  have h7: A₁₂ = 20 / 3 := rfl
  have h8: PQ^2 = 4 * (12^2 - (20 / 3)^2) := sorry
  have h9: PQ^2 = 4 * (144 - 400 / 9) := sorry
  have h10: PQ^2 = 4 * (1296 / 9 - 400 / 9) := sorry
  have h11: PQ^2 = 4 * (896 / 9) := sorry
  have h12: PQ^2 = 3584 / 9 := sorry

  -- Conclude the theorem
  exact ⟨O₄, O₈, O₁₂, A₄, A₈, A₁₂, PQ, h1, h2, h3, h4, h5, h6, (λ _, h12)⟩

end circle_chord_square_length_l514_514176


namespace equilateral_triangle_ratio_l514_514876

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514876


namespace num_repeating_decimals_1_to_20_l514_514612

theorem num_repeating_decimals_1_to_20 : 
  (∃ count_repeating : ℕ, count_repeating = 18 ∧ 
    ∀ n, 1 ≤ n ∧ n ≤ 20 → ((∃ k, n = 9 * k ∨ n = 18 * k) → false) → 
        (∃ d, (∃ normalized, n / 18 = normalized ∧ normalized.has_repeating_decimal))) :=
sorry

end num_repeating_decimals_1_to_20_l514_514612


namespace complement_intersection_l514_514766

open Set

variable (I : Set ℕ) (A B : Set ℕ)

-- Given the universal set and specific sets A and B
def universal_set : Set ℕ := {1,2,3,4,5}
def set_A : Set ℕ := {2,3,5}
def set_B : Set ℕ := {1,2}

-- To prove that the complement of B in I intersects A to be {3,5}
theorem complement_intersection :
  (universal_set \ set_B) ∩ set_A = {3,5} :=
sorry

end complement_intersection_l514_514766


namespace parabola_tangent_parameter_l514_514859

theorem parabola_tangent_parameter (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hp : p ≠ 0) :
  ∃ p : ℝ, (∀ y, y^2 + (2 * p * b / a) * y + (2 * p * c^2 / a) = 0) ↔ (p = 2 * a * c^2 / b^2) := 
by
  sorry

end parabola_tangent_parameter_l514_514859


namespace find_quadratic_l514_514015

theorem find_quadratic (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c)
  (h2 : ∃ x₁ : ℝ, ∀ x : ℝ, f x = 0 → x = x₁)
  (h3 : ∀ x : ℝ, deriv f x = 2 * x + 2) : 
  f = λ x, x^2 + 2 * x + 1 := 
sorry

end find_quadratic_l514_514015


namespace real_return_l514_514395

theorem real_return (n i r: ℝ) (h₁ : n = 0.21) (h₂ : i = 0.10) : 
  (1 + r) = (1 + n) / (1 + i) → r = 0.10 :=
by
  intro h₃
  sorry

end real_return_l514_514395


namespace remaining_battery_life_l514_514536

theorem remaining_battery_life :
  let capacity1 := 60
  let capacity2 := 80
  let capacity3 := 120
  let used1 := capacity1 * (3 / 4 : ℚ)
  let used2 := capacity2 * (1 / 2 : ℚ)
  let used3 := capacity3 * (2 / 3 : ℚ)
  let remaining1 := capacity1 - used1 - 2
  let remaining2 := capacity2 - used2 - 2
  let remaining3 := capacity3 - used3 - 2
  remaining1 + remaining2 + remaining3 = 89 := 
by
  sorry

end remaining_battery_life_l514_514536


namespace solution_l514_514734

open Real

def M : Set ℝ := {x | x^2 - x - 2 > 0}
def N : Set ℝ := {x | 1 ≤ 2^(x - 1) ∧ 2^(x - 1) ≤ 8}
def intersection := {x | 2 < x ∧ x ≤ 4}

theorem solution : M ∩ N = intersection := 
by sorry

end solution_l514_514734


namespace find_angle_D_l514_514342

theorem find_angle_D (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : B = C + 40) : D = 70 := by
  sorry

end find_angle_D_l514_514342


namespace smallest_n_for_g_n_eq_4_l514_514747

/-- 
  Let g(n) be the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 = n.
  Prove that the smallest positive integer n for which g(n) = 4 is 25.
-/
def g (n : ℕ) : ℕ :=
  (finset.univ.product finset.univ).filter (λ (ab : ℕ × ℕ), ab.1 ^ 2 + ab.2 ^ 2 = n ∧ ab.1 ≠ ab.2).card

theorem smallest_n_for_g_n_eq_4 :
  ∃ n : ℕ, g n = 4 ∧ (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 25
  sorry

end smallest_n_for_g_n_eq_4_l514_514747


namespace arithmetic_sequence_properties_l514_514647

variable {a_n : ℕ → ℤ}

noncomputable def general_formula (a : ℕ → ℤ) : ℤ → ℤ := 
  if ∀ n, a n = -3 * n + 5 then -3 * n + 5 else 3 * n - 7

axiom sum_condition : a_n 0 + a_n 1 + a_n 2 = -3
axiom product_condition : a_n 0 * a_n 1 * a_n 2 = 8

noncomputable def sum_of_abs_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ := 
  if a 0 = -4 ∧ a 1 = -1 ∧ a 2 = 2 then n^2 - n + 10 else 0

theorem arithmetic_sequence_properties :
  (general_formula a_n = -3 * n + 5 ∨ general_formula a_n = 3 * n - 7)
  ∧ (∀ n, a_n = 3 * n - 7 → sum_of_abs_sequence a_n n = n^2 - n + 10) :=
  by
    sorry

end arithmetic_sequence_properties_l514_514647


namespace investment_difference_l514_514768

theorem investment_difference :
  let P := 10000
  let r := 0.05
  let t := 3
  let maria_amount := P * (1 + r)^t
  let liam_amount := P * (1 + r / 2)^(2 * t)
  let difference := liam_amount - maria_amount
  difference ≈ 16 := by
{
  let P := 10000
  let r := 0.05
  let t := 3
  let maria_amount := P * (1 + r)^t
  let liam_amount := P * (1 + r / 2)^(2 * t)
  let difference := liam_amount - maria_amount
  sorry
}

end investment_difference_l514_514768


namespace solve_system_l514_514419

theorem solve_system :
  ∃ (x y : ℝ), x + 2 * sqrt y = 6 ∧ sqrt x + y = 4 ∧ 
  0 ≤ x ∧ x ≤ 6 ∧ 0 ≤ y ∧ y ≤ 4 ∧ x = 2.985 ∧ y = 2.272 :=
sorry

end solve_system_l514_514419


namespace central_angle_unchanged_l514_514444

theorem central_angle_unchanged (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0) :
  (s / r) = (2 * s / (2 * r)) :=
by
  sorry

end central_angle_unchanged_l514_514444


namespace k_any_real_l514_514722

-- Define the conditions
theorem k_any_real (k r : ℝ) (a : ℝ) :
  (a = (k * r)^3) → (a = 0.125 * (k * (r / 2))^3) → (k ∈ set.univ) :=
by
  sorry

end k_any_real_l514_514722


namespace part1_part2_l514_514756

-- Definitions based on the conditions.
variable {f : ℝ → ℝ}
variable {n : ℕ}
axiom cond1 : ∀ x, 0 ≤ x ∧ x ≤ 2 → f(2 - x) = f(x) ∧ f(x) ≥ 1 ∧ (x = 1 → f(x) = 3)
axiom cond2 : ∀ x y, 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 2 ∧ x + y ≥ 3 → f(x) + f(y) ≤ f(x + y - 2) + 1

-- Proof statements.
theorem part1 : ∀ (n : ℕ), n > 0 → f (1 / 3^n) ≤ (2 / 3^n) + 1 :=
by
  sorry

theorem part2 : ∀ x, 1 ≤ x ∧ x ≤ 2 → 1 ≤ f(x) ∧ f(x) ≤ 13 - 6 * x :=
by
  sorry

end part1_part2_l514_514756


namespace product_evaluation_l514_514202

theorem product_evaluation : 
  (7 - 5) * (7 - 4) * (7 - 3) * (7 - 2) * (7- 1) * 7 = 5040 := 
by 
  sorry

end product_evaluation_l514_514202


namespace ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514885

-- Define the side length of the equilateral triangle
def s : ℝ := 6

-- Define the equilateral triangle (not necessary in Lean as definitions follow mathematically)
-- The area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- The perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Statement of the theorem to prove the ratio of area to perimeter
theorem ratio_of_area_to_perimeter_of_equilateral_triangle :
  (area s) / (perimeter s) = Real.sqrt 3 / 2 :=
by
  sorry

end ratio_of_area_to_perimeter_of_equilateral_triangle_l514_514885


namespace part_one_part_two_l514_514626

noncomputable def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

noncomputable def S_n (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem part_one
  (a₁ d : ℝ)
  (h1 : a₁ ≥ 0)
  (h2 : d > 0)
  (m n p : ℕ)
  (h3 : m > 0 ∧ n > 0 ∧ p > 0)
  (h4 : m + n = 2 * p) :
  (1 / S_n a₁ d m) + (1 / S_n a₁ d n) ≥ 2 / S_n a₁ d p :=
sorry

theorem part_two
  (a₁ d : ℝ)
  (h1 : a₁ ≥ 0)
  (h2 : d > 0)
  (h3 : arithmetic_seq a₁ d 503 ≤ 1 / 1005) :
  (∑ n in Finset.range 2007, 1 / S_n a₁ d (n + 1)) > 2008 :=
sorry

end part_one_part_two_l514_514626


namespace circle_chord_square_length_l514_514178

-- Define the problem in terms of Lean constructs
theorem circle_chord_square_length :
  ∃ (O₄ O₈ O₁₂ : ℝ) (A₄ A₈ A₁₂ : ℝ) (PQ : ℝ),
    O₄ = 4 ∧
    O₈ = 8 ∧
    O₁₂ = 12 ∧
    A₈ = 8 ∧
    A₄ = 4 ∧
    A₁₂ = (2 * A₈ + A₄) / 3 ∧
    PQ^2 = 4 * (O₁₂^2 - A₁₂^2) → 
    PQ^2 = 3584 / 9 :=
by
  -- Define variables based on conditions
  let O₄ := 4
  let O₈ := 8
  let O₁₂ := 12
  let A₈ := 8
  let A₄ := 4
  let A₁₂ := (2 * A₈ + A₄) / 3
  let s : ℝ := O₁₂^2 - A₁₂^2
  let PQ := real.sqrt(4 * s)

  -- Using given values to verify
  have h1 : O₄ = 4 := rfl
  have h2 : O₈ = 8 := rfl
  have h3 : O₁₂ = 12 := rfl
  have h4 : A₈ = 8 := rfl
  have h5 : A₄ = 4 := rfl
  have h6 : A₁₂ = (2 * A₈ + A₄) / 3 := rfl

  -- Calculate parts
  have h7: A₁₂ = 20 / 3 := rfl
  have h8: PQ^2 = 4 * (12^2 - (20 / 3)^2) := sorry
  have h9: PQ^2 = 4 * (144 - 400 / 9) := sorry
  have h10: PQ^2 = 4 * (1296 / 9 - 400 / 9) := sorry
  have h11: PQ^2 = 4 * (896 / 9) := sorry
  have h12: PQ^2 = 3584 / 9 := sorry

  -- Conclude the theorem
  exact ⟨O₄, O₈, O₁₂, A₄, A₈, A₁₂, PQ, h1, h2, h3, h4, h5, h6, (λ _, h12)⟩

end circle_chord_square_length_l514_514178


namespace find_quadratic_l514_514014

theorem find_quadratic (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c)
  (h2 : ∃ x₁ : ℝ, ∀ x : ℝ, f x = 0 → x = x₁)
  (h3 : ∀ x : ℝ, deriv f x = 2 * x + 2) : 
  f = λ x, x^2 + 2 * x + 1 := 
sorry

end find_quadratic_l514_514014


namespace joan_has_6_balloons_l514_514355

theorem joan_has_6_balloons (initial_balloons : ℕ) (lost_balloons : ℕ) (h1 : initial_balloons = 8) (h2 : lost_balloons = 2) : initial_balloons - lost_balloons = 6 :=
sorry

end joan_has_6_balloons_l514_514355


namespace min_omega_l514_514034

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + 1)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * (x - 1) + 1)

def condition1 (ω : ℝ) : Prop := ω > 0
def condition2 (ω : ℝ) (x : ℝ) : Prop := g ω x = Real.sin (ω * x - ω + 1)
def condition3 (ω : ℝ) (k : ℤ) : Prop := ∃ k : ℤ, ω = 1 - k * Real.pi

theorem min_omega (ω : ℝ) (k : ℤ) (x : ℝ) : condition1 ω → condition2 ω x → condition3 ω k → ω = 1 :=
by
  intros h1 h2 h3
  sorry

end min_omega_l514_514034


namespace term_250_of_sequence_omitting_squares_and_cubes_l514_514074

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

def is_sixth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m * m * m * m = n

def sequence_omitting_squares_and_cubes : ℕ → ℕ
| 0 := 1
| (n+1) := let next_candidate := sequence_omitting_squares_and_cubes n + 1 in
            if is_perfect_square next_candidate ∨ is_perfect_cube next_candidate then
              sequence_omitting_squares_and_cubes (n + 1)
            else
              next_candidate

theorem term_250_of_sequence_omitting_squares_and_cubes :
  sequence_omitting_squares_and_cubes 249 = 269 :=
sorry

end term_250_of_sequence_omitting_squares_and_cubes_l514_514074


namespace factorial_sum_perfect_square_iff_l514_514569

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, m * m = n

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map Nat.factorial |>.sum

theorem factorial_sum_perfect_square_iff (n : Nat) :
  n = 1 ∨ n = 3 ↔ is_perfect_square (sum_of_factorials n) := by {
  sorry
}

end factorial_sum_perfect_square_iff_l514_514569


namespace skill_testing_question_l514_514845

theorem skill_testing_question : (5 * (10 - 6) / 2) = 10 := by
  sorry

end skill_testing_question_l514_514845


namespace car_speeds_l514_514541

theorem car_speeds (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v) / 2) :=
sorry

end car_speeds_l514_514541


namespace tangency_of_AT_l514_514731

open EuclideanGeometry

-- Let's define the given conditions first
variables {A B C D E T : Point}

-- Given: Pentagon ABCDE is convex
-- Given: Triangles ΔABE, ΔBEC, ΔEDB are similar
-- Given: Lines BE and CD intersect at point T

def similar_triangles (P Q R S T U : Point) : Prop :=
  ∀ (ABC_def : geometric primitive of similar triangle), 
    -- Here "geometric primitive of similar triangle" is to represent similarity 
    -- condition that can include angles or sides ratio
    ∃ φ ψ χ : ℝ, ∠PQR = φ ∧ ∠QRS = ψ ∧ ∠RSP = χ

 -- The statement we want to prove
theorem tangency_of_AT 
    (h1 : ∃ P : Point, convex_pentagon A B C D E)
    (h2 :  similar_triangles A B E ∧ similar_triangles B E C ∧ similar_triangles E D B)
    (h3 : BE ∩ CD = T) :
    tangent (line AT) (circumcircle of (ΔACD)) :=
sorry

end tangency_of_AT_l514_514731


namespace larger_rectangle_area_l514_514806

/-- Given a smaller rectangle made out of three squares each of area 25 cm²,
    where two vertices of the smaller rectangle lie on the midpoints of the
    shorter sides of the larger rectangle and the other two vertices lie on
    the longer sides, prove the area of the larger rectangle is 150 cm². -/
theorem larger_rectangle_area (s : ℝ) (l W S_Larger W_Larger : ℝ)
  (h_s : s^2 = 25) 
  (h_small_dim : l = 3 * s ∧ W = s ∧ l * W = 3 * s^2) 
  (h_vertices : 2 * W = W_Larger ∧ l = S_Larger) :
  (S_Larger * W_Larger = 150) := 
by
  sorry

end larger_rectangle_area_l514_514806


namespace added_number_after_doubling_l514_514108

theorem added_number_after_doubling (x y : ℤ) (h1 : x = 4) (h2 : 3 * (2 * x + y) = 51) : y = 9 :=
by
  -- proof goes here
  sorry

end added_number_after_doubling_l514_514108


namespace teddy_bears_per_shelf_l514_514795

theorem teddy_bears_per_shelf :
  (98 / 14 = 7) := 
by
  sorry

end teddy_bears_per_shelf_l514_514795


namespace square_of_chord_length_is_39804_l514_514179

noncomputable def square_of_chord_length (r4 r8 r12 : ℝ) (externally_tangent : (r4 + r8) < r12) : ℝ := 
  let r4 := 4
  let r8 := 8
  let r12 := 12
  let PQ_sq := 4 * ((r12^2) - ((2 * r8 + 1 * r4) / 3)^2) in
  PQ_sq

theorem square_of_chord_length_is_39804 : 
  square_of_chord_length 4 8 12 ((4 + 8) < 12) = 398.04 := 
by
  sorry

end square_of_chord_length_is_39804_l514_514179


namespace number_of_handshakes_l514_514529

def child (n: ℕ) := n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}

def younger (i j: ℕ) : Prop := i < j

def participates (n: ℕ) : Prop := n > 1 

def handshakes : ℕ := 
  ∑ i in {2, 3, 4, 5, 6, 7, 8, 9}, 
    ∑ j in {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
    if participates i ∧ child j ∧ younger j i then 1 else 0

theorem number_of_handshakes : handshakes = 36 := 
by
  sorry

end number_of_handshakes_l514_514529


namespace only_const_polynomials_l514_514214

noncomputable def is_const_poly (P : ℤ[X]) : Prop :=
  ∃ c : ℤ, P = polynomial.C c

theorem only_const_polynomials (P : ℤ[X]) :
  (∀ n : ℕ, n > 0 → ∀ p : ℕ, nat.prime p → (p ∣ n * P.eval (n : ℤ) → false) → nat.ord_p p n ≥ nat.ord_p p (P.eval (n : ℤ))) →
  is_const_poly P :=
sorry

end only_const_polynomials_l514_514214


namespace sisters_work_together_days_l514_514349

-- Definitions based on conditions
def task_completion_rate_older_sister : ℚ := 1/10
def task_completion_rate_younger_sister : ℚ := 1/20
def work_done_by_older_sister_alone : ℚ := 4 * task_completion_rate_older_sister
def remaining_task_after_older_sister : ℚ := 1 - work_done_by_older_sister_alone
def combined_work_rate : ℚ := task_completion_rate_older_sister + task_completion_rate_younger_sister

-- Statement of the proof problem
theorem sisters_work_together_days : 
  (combined_work_rate * x = remaining_task_after_older_sister) → 
  (x = 4) :=
by
  sorry

end sisters_work_together_days_l514_514349


namespace seq_a_n_minus_2_geometric_T_n_formula_l514_514740

open Classical

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Given conditions
def a1 := a 1 = 1
def Sn_an_eq_2n (n : ℕ) (_: n > 0) := S n + a n = 2 * n

-- Prove that {a_n - 2} forms a geometric sequence
theorem seq_a_n_minus_2_geometric :
  (∀ n : ℕ, n > 0 → (a n - 2) = -(1/2)^(n-1)) :=
sorry

-- Find the sum of the first n terms of S_n sequence, denoted as T_n
def T (n : ℕ) := ∑ i in Finset.range n, S (i + 1)

-- Prove T_n formula
theorem T_n_formula (n : ℕ) :
  T n = n^2 - n + 2 - (1/2)^(n-1) :=
sorry

end seq_a_n_minus_2_geometric_T_n_formula_l514_514740


namespace angle_between_vectors_l514_514693

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

def angle_between (a b : V) : ℝ := real.acos ((inner a b) / (∥a∥ * ∥b∥))

theorem angle_between_vectors (ha : a ≠ 0) (hb : b ≠ 0) (h : ∥a∥ = ∥b∥)
  (h_dot : inner ((sqrt 3) • a - 2 • b) a = 0) : angle_between a b = π / 6 :=
sorry

end angle_between_vectors_l514_514693


namespace num_repeating_decimals_1_to_20_l514_514613

theorem num_repeating_decimals_1_to_20 : 
  (∃ count_repeating : ℕ, count_repeating = 18 ∧ 
    ∀ n, 1 ≤ n ∧ n ≤ 20 → ((∃ k, n = 9 * k ∨ n = 18 * k) → false) → 
        (∃ d, (∃ normalized, n / 18 = normalized ∧ normalized.has_repeating_decimal))) :=
sorry

end num_repeating_decimals_1_to_20_l514_514613


namespace integer_lengths_impossible_l514_514373

open Triangle

theorem integer_lengths_impossible 
  (A B C D E I : Point)
  (hA90 : ∠ BAC = 90)
  (hD_on_AC : D ∈ Line AC)
  (hE_on_AB : E ∈ Line AB)
  (h_angle_ABD_DBC : ∠ ABD = ∠ DBC)
  (h_angle_ACE_ECB : ∠ ACE = ∠ ECB)
  (h_meet_I : Meets (Line BD) (Line CE) I) :
  ¬ (is_integer_length AB ∧ is_integer_length AC ∧
     is_integer_length BI ∧ is_integer_length ID ∧ 
     is_integer_length CI ∧ is_integer_length IE) :=
begin
  sorry
end

end integer_lengths_impossible_l514_514373


namespace train_ride_length_l514_514149

theorem train_ride_length :
  let reading_time := 2
  let eating_time := 1
  let watching_time := 3
  let napping_time := 3
  reading_time + eating_time + watching_time + napping_time = 9 := 
by
  sorry

end train_ride_length_l514_514149


namespace dot_product_equilateral_l514_514335

variables (A B C : Type) [inner_product_space ℝ A]

-- Hypotheses for the equilateral triangle with side lengths sqrt(2)
variables [equilateral A B C] (side_len : ℝ)
  (a b c : A) 
  (h_AB : ∥A - B∥ = side_len) 
  (h_BC : ∥B - C∥ = side_len) 
  (h_CA : ∥C - A∥ = side_len)
  (h_side_len : side_len = real.sqrt 2)

-- Definitions of vectors
def vec_ab : A := A - B
def vec_bc : A := B - C
def vec_ca : A := C - A

-- Hypotheses for vectors corresponding to the triangle
variables (h_vec_ab : vec_ab = c) (h_vec_bc : vec_bc = a) (h_vec_ca : vec_ca = b)

theorem dot_product_equilateral :
  vec_bc ⬝ vec_ca + vec_ca ⬝ vec_ab + vec_ab ⬝ vec_bc = -3 :=
by sorry

end dot_product_equilateral_l514_514335


namespace missing_condition_l514_514996

theorem missing_condition (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : y = 3 * (x - 2)) :
  true := -- The equivalent mathematical statement asserts the correct missing condition.
sorry

end missing_condition_l514_514996


namespace probability_A_selected_l514_514001

-- Define the three individuals A, B, and C as a type
inductive Individuals
| A : Individuals
| B : Individuals
| C : Individuals

open Individuals

-- Define the function that calculates the probability
def probability_of (selected: Individuals → Prop) : ℚ :=
  let total_combinations := {S | ∃ (x y : Individuals), x ≠ y ∧ (selected x) ∧ (selected y)}
  let favorable_combinations := total_combinations.filter (λ s, selected A)
  (favorable_combinations.card : ℚ) / (total_combinations.card : ℚ)

-- Statement to prove
theorem probability_A_selected : probability_of (λ x, x = A ∨ x = B ∨ x = C) = 2 / 3 :=
sorry

end probability_A_selected_l514_514001


namespace number_of_space_diagonals_l514_514103

-- A convex polyhedron Q has the following properties:
def Q_vertices : ℕ := 30
def Q_edges : ℕ := 70
def Q_faces : ℕ := 42
def Q_triangular_faces : ℕ := 30
def Q_pentagonal_faces : ℕ := 12

-- The question to be proved is that the number of space diagonals is 305.
theorem number_of_space_diagonals : 
  ∃ d : ℕ, d = (Q_vertices.choose 2) - Q_edges - (Q_pentagonal_faces * ((5.choose 2) - 5)) ∧ d = 305 :=
by
  sorry

end number_of_space_diagonals_l514_514103


namespace equivalence_statement_l514_514059

open Complex

noncomputable def distinct_complex (a b c d : ℂ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem equivalence_statement (a b c d : ℂ) (h : distinct_complex a b c d) :
  (∀ (z : ℂ), (abs (z - a) + abs (z - b) ≥ abs (z - c) + abs (z - d)))
  ↔ (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ c = t * a + (1 - t) * b ∧ d = (1 - t) * a + t * b) :=
sorry

end equivalence_statement_l514_514059


namespace natural_numbers_satisfy_l514_514549

theorem natural_numbers_satisfy (n : ℕ) (h : n ≥ 2) :
  ∃ a b : ℕ, (a divides n) ∧ (b divides n) ∧ (a ≠ 1 ∧ a ≤ b) ∧ n = a^2 + b^2 ∧
  (n = 4 ∨ ∃ (k j : ℕ), k ≥ 2 ∧ j ∈ {1, 2, ..., k} ∧ n = 2^k * (2^(k*(j - 1)) + 1)) :=
sorry

end natural_numbers_satisfy_l514_514549


namespace equilateral_triangle_ratio_l514_514931

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514931


namespace ratio_of_projection_l514_514433

theorem ratio_of_projection (a b : ℝ) 
  (h : (⟨⟨1/18, -2/9⟩, ⟨-2/9, 8/9⟩⟩ : Matrix (Fin 2) (Fin 2) ℝ) ⬝ (λ i, [a, b].nth i.1) = λ i, [a, b].nth i.1)
  (h_nonzero : a ≠ 0 ∨ b ≠ 0) :
  b / a = 17 / 4 :=
sorry

end ratio_of_projection_l514_514433


namespace r_earns_per_day_l514_514089

theorem r_earns_per_day (P Q R : ℝ) 
(h1 : P + Q + R = 200) 
(h2 : P + R = 120) 
(h3 : Q + R = 130) : 
R = 50 := 
begin
  -- The proof would go here.
  -- We are focusing on the statement only, as directed.
  sorry
end

end r_earns_per_day_l514_514089


namespace triangle_area_perimeter_ratio_l514_514920

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514920


namespace monic_cubic_polynomial_has_root_l514_514210

noncomputable def Q (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 11

theorem monic_cubic_polynomial_has_root :
  Q (Real.cbrt 3 + 2) = 0 :=
sorry

end monic_cubic_polynomial_has_root_l514_514210


namespace solution_exists_l514_514568

noncomputable def equation (x : ℝ) := 
  (x^2 - 5 * x + 4) / (x - 1) + (2 * x^2 + 7 * x - 4) / (2 * x - 1)

theorem solution_exists : equation 2 = 4 := by
  sorry

end solution_exists_l514_514568


namespace pyramid_coloring_l514_514183

noncomputable def pyramidColoringWays (colors : ℕ) : ℕ := 
  let case1 := colors * (colors - 1) * (colors - 2) * 1 * (colors - 2)
  let case2 := colors * (colors - 1) * (colors - 2) * (colors - 3) * (colors - 3)
  case1 + case2

theorem pyramid_coloring (colors : ℕ) (h : colors = 5) : pyramidColoringWays colors = 420 :=
by 
  rw [h]
  simp [pyramidColoringWays]
  sorry

end pyramid_coloring_l514_514183


namespace length_BD_of_parallelogram_l514_514719

def A : ℂ := complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * complex.I

noncomputable def D : ℂ := 3 + 3 * complex.I

theorem length_BD_of_parallelogram :
  complex.abs (D - B) = sqrt 13 := 
sorry

end length_BD_of_parallelogram_l514_514719


namespace equilateral_triangle_ratio_l514_514967

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514967


namespace part_1_part_2_part_3_l514_514628

noncomputable def sequence_an (n : ℕ) : ℝ := n
noncomputable def Sn (n : ℕ) : ℝ := (n * (n + 1)) / 2 -- Sum of first n natural numbers

theorem part_1 (n : ℕ) (hn : n > 0) : 
  2 * Sn n = sequence_an n * (sequence_an n + 1) := by
sorry

noncomputable def sequence_bn (n : ℕ) : ℝ := (1 / 2) ^ n * sequence_an n
noncomputable def Tn (n : ℕ) : ℝ := ∑ i in finset.range n, sequence_bn i 

theorem part_2 (n : ℕ) (hn : n > 0) :
  ∑ i in finset.range n, 1 / (sequence_an i + 2)^2 < 1 / 2 := by
sorry

theorem part_3 (n : ℕ) (hn : n > 0) (λ : ℝ) :
  (-2)^(n - 1) * λ < Tn n + n / 2^n - 2^(n - 1) -> (λ < 0 ∨ λ > 1 / 4) := by
sorry

end part_1_part_2_part_3_l514_514628


namespace find_number_of_terms_in_AP_l514_514333

theorem find_number_of_terms_in_AP 
  (a d : ℤ) (n : ℕ) 
  (h1 : n % 2 = 0) 
  (h2 : ∑ i in finset.range (n/2), (a + (2 * i) * d) = 36) 
  (h3 : ∑ i in finset.range (n/2), (a + (2 * i + 1) * d) = 44) 
  (h4 : (n - 1) * d = 12) 
  : n = 8 :=
sorry

end find_number_of_terms_in_AP_l514_514333


namespace retailer_items_sold_without_discount_l514_514198

theorem retailer_items_sold_without_discount (P N : ℕ) (D : ℚ) 
  (hP : 0.10 * P = 60)
  (hDiscount : D = 222.22222222222223) :
  N = 111 :=
by
  -- Define profit per item and check it
  let profit_per_item := 60
  let item_price := 600 
  let discount := 0.05 * item_price
  let new_price := item_price - discount
  let new_profit_per_item := profit_per_item - discount
  have hnew_price : new_price = 570 := by norm_num [new_price, discount, item_price]
  have hnew_profit_per_item : new_profit_per_item = 30 := by norm_num [new_profit_per_item, profit_per_item, discount]
  -- Calculate number of items sold without the discount
  let total_profit_with_discount := D * new_profit_per_item
  let N := total_profit_with_discount / profit_per_item
  sorry

end retailer_items_sold_without_discount_l514_514198


namespace TetrahedronPropertiesAreEquivalent_l514_514400

noncomputable def TetrahedronPropertiesEquivalence (P Q R S T U V W X Y Z A B C D : Prop) :=
  (P ↔ Q) ∧ (Q ↔ R) ∧ (R ↔ S) ∧ (S ↔ T) ∧ (T ↔ U) ∧ (U ↔ V) ∧ (V ↔ W) ∧ (W ↔ X) ∧ (X ↔ Y) ∧ (Y ↔ Z) ∧ (Z ↔ A) ∧ (A ↔ B) ∧ (B ↔ C) ∧ (C ↔ D) ∧ (D ↔ P)

def Property1 := sorry -- All faces have the same area
def Property2 := sorry -- Each edge is equal to its opposite edge
def Property3 := sorry -- All faces are congruent
def Property4 := sorry -- The centers of the circumscribed and inscribed spheres coincide
def Property5 := sorry -- The sums of the angles at each vertex are equal
def Property6 := sorry -- The sum of the plane angles at each vertex is 180°
def Property7 := sorry -- The development of the tetrahedron is an acute triangle with median lines
def Property8 := sorry -- All faces are acute triangles with the same circumradius
def Property9 := sorry -- The orthogonal projection of the tetrahedron onto each of the three planes parallel to two opposite edges is a rectangle
def Property10 := sorry -- The parallelepiped obtained by passing three pairs of parallel planes through opposite edges is rectangular
def Property11 := sorry -- The heights of the tetrahedron are equal
def Property12 := sorry -- The point of intersection of the medians coincides with the center of the circumscribed sphere
def Property13 := sorry -- The point of intersection of the medians coincides with the center of the inscribed sphere
def Property14 := sorry -- The sum of the plane angles at three vertices is 180°
def Property15 := sorry -- The sum of the plane angles at two vertices is 180° and two opposite edges are equal

theorem TetrahedronPropertiesAreEquivalent :
  TetrahedronPropertiesEquivalence Property1 Property2 Property3 Property4 Property5 Property6 Property7 Property8 Property9 Property10 Property11 Property12 Property13 Property14 Property15 :=
  sorry

end TetrahedronPropertiesAreEquivalent_l514_514400


namespace parker_distance_l514_514530

variable (P : ℝ) -- Distance Parker threw the ball
variable (G : ℝ) -- Distance Grant threw the ball
variable (K : ℝ) -- Distance Kyle threw the ball

-- Conditions provided in the problem
def condition1 : Prop := G = 1.25 * P
def condition2 : Prop := K = 2 * G
def condition3 : Prop := K = P + 24

-- Prove that Parker's distance P is 16
theorem parker_distance : condition1 → condition2 → condition3 → P = 16 := by
  intros h1 h2 h3
  sorry

end parker_distance_l514_514530


namespace integral_of_f_l514_514435

-- Define the function f(x) with the given condition
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + 2*x + m

theorem integral_of_f (m : ℝ) (h : ∀ x, f x m = x^2 + 2*x + m) (h_min : ∀ x, f x m ≥ -1) :
  ∫ x in 1..2, f x m = 16 / 3 :=
by
  -- Additional necessary steps to define min, etc.
  sorry

end integral_of_f_l514_514435


namespace equilateral_triangle_ratio_l514_514905

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514905


namespace billy_weight_l514_514147

variable (B Bd C D : ℝ)

theorem billy_weight
  (h1 : B = Bd + 9)
  (h2 : Bd = C + 5)
  (h3 : C = D - 8)
  (h4 : C = 145)
  (h5 : D = 2 * Bd) :
  B = 85.5 :=
by
  sorry

end billy_weight_l514_514147


namespace number_of_consecutive_integer_sets_sum_100_l514_514310

/-- There are exactly two sets of at least two consecutive positive integers that sum to 100. -/
theorem number_of_consecutive_integer_sets_sum_100 : 
  (∃ s : Set ℕ, (∀ n ∈ s, n > 0) ∧ (s.card ≥ 2) ∧ (s.sum id = 100)) = 2 :=
sorry

end number_of_consecutive_integer_sets_sum_100_l514_514310


namespace flight_distance_each_way_l514_514111

variables (D : ℝ) (T_out T_return total_time : ℝ)

-- Defining conditions
def condition1 : Prop := T_out = D / 300
def condition2 : Prop := T_return = D / 500
def condition3 : Prop := total_time = 8

-- Given conditions
axiom h1 : condition1 D T_out
axiom h2 : condition2 D T_return
axiom h3 : condition3 total_time

-- The proof problem statement
theorem flight_distance_each_way : T_out + T_return = total_time → D = 1500 :=
by
  sorry

end flight_distance_each_way_l514_514111


namespace intersection_tetra_volume_l514_514115

-- Define the dimensions of the rectangular prism
variables (a b c : ℝ)

-- The theorem stating the volume of the intersection polyhedron V
theorem intersection_tetra_volume (a b c : ℝ) : volume_of_intersection_tetra (a b c) = (1 / 6) * a * b * c := sorry

end intersection_tetra_volume_l514_514115


namespace ellipse_equation_dot_product_range_line_PQ_fixed_point_l514_514266

-- Define the parameters and conditions
variable (x y : ℝ)
variable (a b c : ℝ)
variable (F₁ F₂ P : ℝ) -- For coordinates of F₁, F₂ and any point P on the ellipse
variable (t : ℝ) -- Parameter for intersection point M (4, t)

-- Given conditions
axiom h1 : a > b ∧ b > 0
axiom h2 : 2 * a = 4
axiom h3 : 2 * c = 2 * Real.sqrt 3
axiom h4 : a^2 = b^2 + c^2

-- Proving the equation of the ellipse
theorem ellipse_equation :
  a = 2 ∧ b = 1 → ∀ x y, (x^2 / 4 + y^2 = 1) :=
by {
  intros ha hb hxy,
  sorry
}

-- Define vectors and dot product
noncomputable def PF₁ (x₀ y₀ : ℝ) := (-Real.sqrt 3 - x₀, -y₀)
noncomputable def PF₂ (x₀ y₀ : ℝ) := (Real.sqrt 3 - x₀, -y₀)

noncomputable def dot_product (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2

-- Proving the range of the dot product
theorem dot_product_range :
  (∀ x₀ y₀, x₀^2 / 4 + y₀^2 = 1) ∧ (0 ≤ x₀^2 ∧ x₀^2 ≤ 4) → ∀ x₀, -2 ≤ dot_product (PF₁ x₀ 0) (PF₂ x₀ 0) ≤ 1 :=
by {
  intros hx hy,
  sorry
}

-- Proving line PQ passes through the fixed point
theorem line_PQ_fixed_point :
  (∀ x M A B : ℝ, M = (4, t) ∧ A = (-2, 0) ∧ B = (2, 0)) → 
  (∃ (fixed_point : ℝ × ℝ), fixed_point = (1, 0)) :=
by {
  intros x M A B hM hA hB,
  use (1, 0),
  sorry
}

end ellipse_equation_dot_product_range_line_PQ_fixed_point_l514_514266


namespace mouse_grasshopper_diff_l514_514035

def grasshopper_jump: ℕ := 19
def frog_jump: ℕ := grasshopper_jump + 10
def mouse_jump: ℕ := frog_jump + 20

theorem mouse_grasshopper_diff:
  (mouse_jump - grasshopper_jump) = 30 :=
by
  sorry

end mouse_grasshopper_diff_l514_514035


namespace num_elements_in_set_is_one_l514_514823

noncomputable def log_eq_condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ Real.log (x^3 + (1 / 3) * y^3 + 1 / 9) = Real.log x + Real.log y

theorem num_elements_in_set_is_one :
  (Set.to_finset {p : ℝ × ℝ | log_eq_condition p.1 p.2}).card = 1 :=
by
  sorry

end num_elements_in_set_is_one_l514_514823


namespace sequence_sum_S9_l514_514257

noncomputable def a_n : ℕ → ℝ
| 0     := 0     -- This is just a placeholder, not used in nat*.
| (n+1) := if n = 0 then 1 else a_n n + 1/2

def sum_n (n : ℕ) := (n + 1) / 2 * (a_n 0 + a_n n)

theorem sequence_sum_S9 : 
  (a_n 9 = 5) → 
  (∀ n ∈ (Nat∗ : Set Nat), 2 * a_n (n+1) = 2 * a_n n + 1) → 
  sum_n 8 = 27 := -- sum_n 8 represents S_9 since Lean indices from 0
by {
  sorry
}

end sequence_sum_S9_l514_514257


namespace z_in_fourth_quadrant_l514_514286

def z (z : ℂ) := (1 - I) / (z - 2) = (1 + I)

theorem z_in_fourth_quadrant (z : ℂ) (h : z z) : 
  ∃ x y : ℝ, z = x + y * I ∧ x > 0 ∧ y < 0 :=
by
  sorry

end z_in_fourth_quadrant_l514_514286


namespace sum_a_m_eq_2_pow_n_b_n_l514_514372

noncomputable def a_n (x : ℝ) (n : ℕ) : ℝ := (Finset.range (n + 1)).sum (λ k => x ^ k)

noncomputable def b_n (x : ℝ) (n : ℕ) : ℝ := 
  (Finset.range (n + 1)).sum (λ k => ((x + 1) / 2) ^ k)

theorem sum_a_m_eq_2_pow_n_b_n 
  (x : ℝ) (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ m => a_n x m * Nat.choose (n + 1) (m + 1)) = 2 ^ n * b_n x n :=
by
  sorry

end sum_a_m_eq_2_pow_n_b_n_l514_514372


namespace find_value_of_composite_function_l514_514656

def f (x : ℝ) : ℝ := 
  if x < 2 then x + 1 else x^2 - 3

theorem find_value_of_composite_function :
  f (f 2) = 2 := by
  sorry

end find_value_of_composite_function_l514_514656


namespace intersection_implies_range_of_k_l514_514297

noncomputable def intersection_condition (k : ℝ) : Prop :=
  let a := (1 - k^2)
  let b := -2 * k
  let c := -2
  (4 * k^2 + 8 * (1 - k^2) > 0) ∧ ((2 * k) / (1 - k^2) > 0) ∧ (-2 / (1 - k^2) > 0)

def range_of_k (k : ℝ) : Prop :=
  (- real.sqrt 2 < k) ∧ (k < -1)

theorem intersection_implies_range_of_k (k : ℝ) :
  intersection_condition k → range_of_k k :=
sorry

end intersection_implies_range_of_k_l514_514297


namespace arithmetic_seq_a2_a6_l514_514023

variable (a : ℕ → ℤ) (d : ℤ)
hypothesis h1 : a 3 = 4
hypothesis h2 : d = -2

theorem arithmetic_seq_a2_a6 : a 2 + a 6 = 4 := by
  sorry

end arithmetic_seq_a2_a6_l514_514023


namespace smallest_z_minus_x_l514_514057

noncomputable def nine_factorial : ℕ := Nat.factorial 9

theorem smallest_z_minus_x (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = nine_factorial) (hxy : x < y) (hyz : y < z) : 
  ∃ x y z, x * y * z = nine_factorial ∧ x < y ∧ y < z ∧ z - x = 186 :=
by
  apply Exists.intro 24
  apply Exists.intro 72
  apply Exists.intro 210
  simp [nine_factorial, Nat.factorial]
  sorry

end smallest_z_minus_x_l514_514057


namespace minimum_value_of_f_on_interval_l514_514280

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x^2 + 4*x - 5
  else (x - 2)^2 - 9

theorem minimum_value_of_f_on_interval :
  ∃ y ∈ set.Icc 3 5, ∀ x ∈ set.Icc 3 5, f x ≥ f y ∧ f y = -8 :=
begin
  sorry
end

end minimum_value_of_f_on_interval_l514_514280


namespace standard_ellipse_eq_min_max_dot_product_exists_point_B_l514_514265

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Condition 1: Prove the standard equation of the ellipse
theorem standard_ellipse_eq :
  ∃ (a b : ℝ), a = 2 ∧ b = sqrt (a^2 - (1:ℝ)^2) ∧ ellipse_eq x y := sorry

-- Define the points F1 and F2
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the dot product function
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

-- Condition 2: Prove the maximum and minimum of dot product
theorem min_max_dot_product (M : ℝ × ℝ) (hM : ellipse_eq M.1 M.2) :
  2 ≤ dot_product (-1 - M.1) (-M.2) (1 - M.1) (-M.2) ∧ 
  dot_product (-1 - M.1) (-M.2) (1 - M.1) (-M.2) ≤ 3 := sorry

-- Condition 3: Existence of point B with described property
theorem exists_point_B :
  ∃ (B : ℝ × ℝ), B = (1, 0) ∧ ∀ (P : ℝ × ℝ), ellipse_eq P.1 P.2 →
    (∃ λ : ℝ, λ = 1/2 ∧ (sqrt ((P.1 - B.1)^2 + P.2^2) / abs (P.1 - 4) = λ)) := sorry

end standard_ellipse_eq_min_max_dot_product_exists_point_B_l514_514265


namespace correct_statements_sequence_l514_514525

/-- The negation of the proposition "There exists an x in ℝ such that x^2 + 1 > 3x"
is "For all x in ℝ, it holds that x^2 + 1 ≤ 3x". -/
def negation_proof : Prop :=
  (∃ x : ℝ, x^2 + 1 > 3x) ↔ ¬(∀ x : ℝ, x^2 + 1 ≤ 3x)

/-- If "p or q" is a false proposition, then "not p and not q" is a true proposition. -/
def disjunction_truth (p q : Prop) : Prop :=
  (¬(p ∨ q)) ↔ (¬p ∧ ¬q)

/-- If p is a sufficient but not necessary condition for q, then not p is a necessary
but not sufficient condition for not q. -/
def suff_nec_conditions (p q : Prop) : Prop :=
  ((p → q) ∧ ((¬q) → (¬p))) ↔ ((¬p → ¬q) ∧ (¬(¬p → ¬q)))

/-- The graph of the function y = sin(-2x) can be obtained by shifting all points of the
graph to the right by π/8 units to get the function y = sin(-2x + π/4). -/
def trig_shift : Prop :=
  ∀ x : ℝ, sin (-2 * x) = sin (-2 * (x - π/8) + π/4)

-- The correct sequence numbers of the statements are therefore: 1234.
theorem correct_statements_sequence :
  negation_proof ∧ disjunction_truth ∧ suff_nec_conditions ∧ trig_shift :=
sorry

end correct_statements_sequence_l514_514525


namespace max_tan_angle_PAS_l514_514668

-- Definitions from the given conditions
def triangle (P Q R : Type) := 
  ∃ (angle_R : ℝ) (QR : ℝ), angle_R = 45 ∧ QR = 6 

def midpoint (S Q R : Type) := 
  ∃ QR, QR / 2 = S 

-- The given problem's statement rephrased:
theorem max_tan_angle_PAS (P A S : Type) (triangle_PQR : triangle P Q R) (mid_S_QR : midpoint S Q R) 
  : ∃ max_tan_value : ℝ, max_tan_value = (2 * real.sqrt 2) / (6 * (real.sqrt 2) - 7.5) := 
  sorry

end max_tan_angle_PAS_l514_514668


namespace inequality_abc_l514_514270

theorem inequality_abc (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a * b * c = 8) :
  (a^2 / Real.sqrt ((1 + a^3) * (1 + b^3))) + (b^2 / Real.sqrt ((1 + b^3) * (1 + c^3))) +
  (c^2 / Real.sqrt ((1 + c^3) * (1 + a^3))) ≥ 4 / 3 :=
sorry

end inequality_abc_l514_514270


namespace ratio_eq_sqrt3_div_2_l514_514944

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514944


namespace rectangle_sides_l514_514229

theorem rectangle_sides (x y : ℕ) :
  (2 * x + 2 * y = x * y) →
  x > 0 →
  y > 0 →
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) ∨ (x = 4 ∧ y = 4) :=
by
  sorry

end rectangle_sides_l514_514229


namespace xiao_ming_equation_l514_514487

-- Defining the parameters of the problem
def distance : ℝ := 2000
def regular_time (x : ℝ) := x
def increased_speed := 5
def time_saved := 2

-- Problem statement to be proven in Lean 4:
theorem xiao_ming_equation (x : ℝ) (h₁ : x > 2) : 
  (distance / (x - time_saved)) - (distance / regular_time x) = increased_speed :=
by
  sorry

end xiao_ming_equation_l514_514487


namespace A_wins_one_prob_A_wins_at_least_2_of_3_prob_l514_514396

-- Define the probability of A and B guessing correctly
def prob_A_correct : ℚ := 5/6
def prob_B_correct : ℚ := 3/5

-- Definition of the independent events for A and B
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- The probability of A winning in one activity
def prob_A_wins_one : ℚ := prob_A_correct * prob_B_incorrect

-- Proof (statement) that A's probability of winning one activity is 1/3
theorem A_wins_one_prob :
  prob_A_wins_one = 1/3 :=
sorry

-- Binomial coefficient for choosing 2 wins out of 3 activities
def binom_coeff_n_2 (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Probability of A winning exactly 2 out of 3 activities
def prob_A_wins_exactly_2_of_3 : ℚ :=
  binom_coeff_n_2 3 2 * prob_A_wins_one^2 * (1 - prob_A_wins_one)

-- Probability of A winning all 3 activities
def prob_A_wins_all_3 : ℚ :=
  prob_A_wins_one^3

-- The probability of A winning at least 2 out of 3 activities
def prob_A_wins_at_least_2_of_3 : ℚ :=
  prob_A_wins_exactly_2_of_3 + prob_A_wins_all_3

-- Proof (statement) that A's probability of winning at least 2 out of 3 activities is 7/27
theorem A_wins_at_least_2_of_3_prob :
  prob_A_wins_at_least_2_of_3 = 7/27 :=
sorry

end A_wins_one_prob_A_wins_at_least_2_of_3_prob_l514_514396


namespace equilateral_triangle_ratio_l514_514934

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514934


namespace laundry_time_l514_514164

/-- 
Proof that the time it takes to do one load of laundry is 9 minutes given:
- Sweeping takes 3 minutes per room.
- Washing the dishes takes 2 minutes per dish.
- Anna sweeps 10 rooms.
- Billy does two loads of laundry.
- Billy washes 6 dishes.
- Each child should spend the same amount of time on chores.
-/
theorem laundry_time : 
  let sweeping_time_per_room := 3 in
  let dish_washing_time_per_dish := 2 in
  let anna_rooms := 10 in
  let billy_loads := 2 in
  let billy_dishes := 6 in
  (anna_rooms * sweeping_time_per_room = billy_loads * L + billy_dishes * dish_washing_time_per_dish) → 
  L = 9 :=
by
  intro sweeping_time_per_room dish_washing_time_per_dish anna_rooms billy_loads billy_dishes anna_eq_billy
  let anna_time := anna_rooms * sweeping_time_per_room
  let billy_dishes_time := billy_dishes * dish_washing_time_per_dish
  let billy_total_time := billy_loads * L + billy_dishes_time
  have eqn : 30 = 2 * L + 12 := by sorry
  sorry

end laundry_time_l514_514164


namespace problem_solution_l514_514252

theorem problem_solution (x: ℝ) (S : Fin 51 → ℝ)
  (hS : ∀ i, S i = x^i)
  (h_A_50 : (A^[50] S) 0 = 1 / 2^25)
  (h_pos : 0 < x) : x = Real.sqrt 2 - 1 :=
sorry

end problem_solution_l514_514252


namespace equilateral_triangle_ratio_l514_514906

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514906


namespace determinant_matrix_l514_514185

variable (a b θ : ℝ)

theorem determinant_matrix :
  det ![
    ![1, Real.sin (a - b + θ), Real.sin a],
    ![Real.sin (a - b + θ), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ] = 1 - Real.sin a ^ 2 - Real.sin b ^ 2 - Real.sin (a - b + θ) ^ 2 + 2 * Real.sin a * Real.sin b * Real.sin (a - b + θ) :=
by 
  sorry

end determinant_matrix_l514_514185


namespace problem1_problem2a_problem2b_l514_514288

def f (x : ℝ) : ℝ := (Real.cos x)^4 - 2 * (Real.sin x) * (Real.cos x) - (Real.sin x)^4

theorem problem1 (x : ℝ) (hx1 : 0 < x) (hx2 : x < π) (hf : f x = -sqrt(2) / 2) :
  x = 5 * π / 24 ∨ x = 13 * π / 24 :=
sorry

theorem problem2a (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) :
  f x ≥ -sqrt(2) :=
sorry

theorem problem2b (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) :
  f x = -sqrt(2) ↔ x = 3 * π / 8 :=
sorry

end problem1_problem2a_problem2b_l514_514288


namespace break_even_number_of_books_l514_514728

-- Definitions from conditions.
def fixed_cost : ℝ := 50000
def variable_cost_per_book : ℝ := 4
def selling_price_per_book : ℝ := 9

-- Main statement proving the break-even point.
theorem break_even_number_of_books 
  (x : ℕ) : (selling_price_per_book * x = fixed_cost + variable_cost_per_book * x) → (x = 10000) :=
by
  sorry

end break_even_number_of_books_l514_514728


namespace triangle_median_concurrency_l514_514363

variables {A B C E F : Type*}
variables [plane_geometry A B C E F]

/- 
  Let \(ABC\) be a triangle and \(d\) be a line parallel to the side \( [BC] \) 
  intersecting the segment \( [AB] \) at point \( E \) and the segment \( [AC] \) at point \( F \).
-/

def is_parallel (d : A → A → Prop) (BC : A → A → Prop) : Prop := sorry
def intersects (d : A → A → Prop) (AB : A → A → Prop) (E : A) : Prop := sorry 
def intersects (d : A → A → Prop) (AC : A → A → Prop) (F : A) : Prop := sorry 
def median (A : A) (BC_midpoint : A) : Prop := sorry
def lies_on (intersection : A) (median : A → Prop) : Prop := sorry

theorem triangle_median_concurrency
  (ABC_triangle : A)
  (BC_parallel_d : is_parallel d (λ a b, b = C ∧ is_on_segment a B C))
  (E_on_AB : intersects d (λ a b, b = B ∧ is_on_segment a A B) E)
  (F_on_AC : intersects d (λ a b, b = C ∧ is_on_segment a A C) F) :
  lies_on (λ p, p = point_of_intersection E C F B) (median A (midpoint B C)) := sorry

end triangle_median_concurrency_l514_514363


namespace value_of_m_l514_514041

theorem value_of_m (m : ℤ) (h₁ : |m| = 2) (h₂ : m ≠ 2) : m = -2 :=
by
  sorry

end value_of_m_l514_514041


namespace inclination_angle_range_l514_514227

theorem inclination_angle_range (α θ : ℝ) (h : 0 ≤ θ ∧ θ < π) :
  (θ = atan (-(cos α) / (sqrt 3)) → θ ∈ [0, π/6] ∨ θ ∈ [5*π/6, π]) := 
sorry

end inclination_angle_range_l514_514227


namespace cards_added_l514_514052

theorem cards_added (initial_cards total_cards added_cards : ℕ) 
  (h1 : initial_cards = 4) 
  (h2 : total_cards = 7) 
  (h3 : total_cards = initial_cards + added_cards) : 
  added_cards = 3 := 
by 
  have h : added_cards = total_cards - initial_cards := by linarith
  rw [h1, h2] at h
  simp only [Nat.add_sub_cancel_left] at h
  exact h

end cards_added_l514_514052


namespace cylinder_original_radius_l514_514561

theorem cylinder_original_radius 
  (r h : ℝ) 
  (hr_eq : h = 3)
  (volume_increase_radius : Real.pi * (r + 8)^2 * 3 = Real.pi * r^2 * 11) :
  r = 8 :=
by
  -- the proof steps will be here
  sorry

end cylinder_original_radius_l514_514561


namespace line_parameterization_l514_514095

theorem line_parameterization (r m : ℝ) (h1 : ∀ t : ℝ, (let p := (5 + t * m, r + t * 6) in p.2 = 3 * p.1 + 2)) :
  r = 17 ∧ m = 2 :=
by
  sorry

end line_parameterization_l514_514095


namespace equilateral_triangle_ratio_l514_514963

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514963


namespace part1_decreasing_function_l514_514290

theorem part1_decreasing_function
  (f : ℝ → ℝ) (ω : ℝ) (hω : ω > 0)
  (H : ∀ x ∈ Ioc (π / 2) π, antitone_on (λ t, f (ω * t - π / 12)) (Ioc (π / 2) π)) :
  ω ∈ set.Icc (1 / 2) (5 / 4) :=
sorry

end part1_decreasing_function_l514_514290


namespace necessary_but_not_sufficient_condition_for_ellipse_l514_514789

-- Definitions based on the conditions
variable (P A B : Point)
variable (a : ℝ) (ha : a > 0)
variable (hPA : dist P A + dist P B = 2 * a)
variable (ellipse_trajectory : ∀ P : Point, dist P A + dist P B = 2 * a → is_ellipse P)

-- The problem statement in Lean 4
theorem necessary_but_not_sufficient_condition_for_ellipse 
  (h : ∀ P, dist P A + dist P B = 2 * a ↔ is_ellipse P) : necessary_but_not_sufficient h :=
sorry

end necessary_but_not_sufficient_condition_for_ellipse_l514_514789


namespace distance_PM_eq_four_thirds_l514_514269

noncomputable def secant_point_distance (P k : Type) (A B C D M : P) :=
  -- Given conditions
  (AB : P) -- secant line passing through \( P \) such that \( PA = AB = 1 \)
  (PA : P -> ℝ) -- \( PA = 1 \)
  (AB_length : ℝ) -- \( AB = 1 \)
  (tangents : PA -> k -> (C, D)) -- Tangents from \( P \) touching circle \( k \)

  -- Prove the distance from \( P \) to \( M \)
  (PM : P -> ℝ)
  
-- Prove
theorem distance_PM_eq_four_thirds :
  ∀ (P k : Type) (A B C D M : P),
    secant_point_distance P k A B C D M ->
    PA = 1 ->
    AB_length = 1 ->
    PM = (4 / 3) :=
by
  sorry

end distance_PM_eq_four_thirds_l514_514269


namespace solve_equation_l514_514415

noncomputable def solution (x y z : ℝ) : Prop :=
  (sin x ≠ 0)
  ∧ (cos y ≠ 0)
  ∧ ( (sin x)^2 + 1 / (sin x)^2 ) ^ 3
    + ( (cos y)^2 + 1 / (cos y)^2 ) ^ 3
    = 16 * (sin z)^2

theorem solve_equation (x y z : ℝ) (n m k : ℤ):
  solution x y z ↔
  (x = (π / 2) + π * n) ∧ (y = π * m) ∧ (z = (π / 2) + π * k) :=
by
  sorry

end solve_equation_l514_514415


namespace minimal_sensors_125_l514_514110

noncomputable def smallest_sensors (n : ℕ) (x : ℕ → ℝ) : Prop :=
  (∑ i in Finset.range n, x i) = 500 ∧
  (∑ i in Finset.range n, (x i) ^ 4) = 80000

theorem minimal_sensors_125 :
  ∃ (n : ℕ) (x : ℕ → ℝ), smallest_sensors n x ∧ n = 125 :=
by
  sorry

end minimal_sensors_125_l514_514110


namespace equilateral_triangle_ratio_l514_514908

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) :
  let area := (s^2 * Real.sqrt 3) / 4
  let perimeter := 3 * s
  area / perimeter = Real.sqrt 3 / 2 :=
by
  rw [h]
  let area := (6^2 * Real.sqrt 3) / 4
  let perimeter := 3 * 6
  have area_result : area = 9 * Real.sqrt 3 := by sorry
  have perimeter_result : perimeter = 18 := by sorry
  have ratio := area_result / perimeter_result
  have final_result : ratio = Real.sqrt 3 / 2 := by sorry
  exact final_result

end equilateral_triangle_ratio_l514_514908


namespace circle_center_distance_l514_514263

theorem circle_center_distance (R : ℝ) : 
  ∃ (d : ℝ), 
  (∀ (θ : ℝ), θ = 30 → 
  ∀ (r : ℝ), r = 2.5 →
  ∀ (center_on_other_side : ℝ), center_on_other_side = R + R →
  d = 5) :=
by 
  use 5
  intros θ θ_eq r r_eq center_on_other_side center_eq
  sorry

end circle_center_distance_l514_514263


namespace find_g_at_4_l514_514430

def g (x : ℝ) : ℝ := sorry

theorem find_g_at_4 (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x ^ 2) : g 4 = 5.5 :=
by
  sorry

end find_g_at_4_l514_514430


namespace find_a_b_l514_514759

theorem find_a_b (a b : ℝ) (h_poly : ∀ x : ℂ, (x^3 + a * x^2 + 2 * x + b = 0) ↔ 
               (x = 2 - 3 * complex.I ∨ x = 2 + 3 * complex.I ∨ x = s)) :

  (a, b) = (-5 / 4, 143 / 4) :=
by {
  have h_s : 2 = 4 * s + 13, from sorry, -- Derived from the problem conditions
  have s := (-11 / 4), -- Derived from h_s
  have ha : a = -(-(11 / 4) + 4), from sorry,
  have hb : b = -13 * (-11 / 4), from sorry,
  split,
  { exact ha },
  { exact hb }
}

end find_a_b_l514_514759


namespace n_is_square_of_integer_l514_514493

variable {n : ℕ}
variable (x : Fin n -> ℤ)

theorem n_is_square_of_integer 
  (H : ∀ i : Fin n, x i = 1 ∨ x i = -1)
  (H1 : (Finset.univ.sum (λ i, x i * x (i + 1 % n))) = 0)
  (Hk : ∀ k : ℕ, 1 ≤ k ∧ k < n → (Finset.univ.sum (λ i, x i * x ((i + k) % n))) = 0) :
  ∃ m : ℕ, n = m ^ 2 := sorry

end n_is_square_of_integer_l514_514493


namespace least_value_q_minus_p_l514_514344

def p : ℝ := 2
def q : ℝ := 5

theorem least_value_q_minus_p (y : ℝ) (h : p < y ∧ y < q) : q - p = 3 :=
by
  sorry

end least_value_q_minus_p_l514_514344


namespace employee_salary_l514_514468

theorem employee_salary (X Y : ℝ) 
  (h1 : X + Y = 560) 
  (h2 : X = 1.2 * Y) : 
  Y ≈ 255 :=
by
  sorry

end employee_salary_l514_514468


namespace problem1_problem2_l514_514093

theorem problem1 (m : ℝ) 
  (M : set ℂ := {2, (m^2 - 2 * m) + (m^2 + m - 2) * complex.i}) 
  (P : set ℂ := {-1, 2, 4 * complex.i}) : 
  M ∪ P = P → m = 1 ∨ m = 2 := 
sorry

theorem problem2 (a : ℝ) 
  (hx1 : (-2 + complex.i) ∈ {x : ℂ | x^2 + 4 * complex.x + a = 0}) : 
  a = 5 ∧ ∃ x2 : ℂ, x2 = -2 - complex.i ∧ x2^2 + 4 * x2 + a = 0 :=
sorry

end problem1_problem2_l514_514093


namespace expression_evaluation_l514_514155

theorem expression_evaluation :
  10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 :=
by
  sorry

end expression_evaluation_l514_514155


namespace fixed_constant_t_l514_514841

-- Define the parabola y = x^2 and the constant c
def parabola (x : ℝ) : ℝ := x^2

-- Assume there exists a corde AB passing through the point C = (0, c)
def passes_through (A B : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  C.snd = C.fst * (A.snd - B.snd) / (A.fst - B.fst)

-- Define the distance formula
def distance (P Q : ℝ × ℝ) : ℝ := 
real.sqrt ((P.fst - Q.fst)^2 + (P.snd - Q.snd)^2)

-- Establish the constant t in terms of distances AC and BC
def constant_t (A B C : ℝ × ℝ) : ℝ :=
(1 / (distance A C)^2) + (1 / (distance B C)^2)

-- Prove that there exists a constant c such that the value of t is fixed
theorem fixed_constant_t :
  (∀ c : ℝ, ∃ (A B : ℝ × ℝ), A.snd = parabola A.fst ∧ B.snd = parabola B.fst ∧ passes_through A B (0, c) ∧ constant_t A B (0, c) = 4) :=
sorry

end fixed_constant_t_l514_514841


namespace equilateral_triangle_ratio_l514_514930

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514930


namespace ratio_eq_sqrt3_div_2_l514_514937

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514937


namespace r_exceeds_s_by_six_l514_514667

theorem r_exceeds_s_by_six (x y : ℚ) (h1 : 3 * x + 2 * y = 16) (h2 : x + 3 * y = 26 / 5) :
  x - y = 6 := by
  sorry

end r_exceeds_s_by_six_l514_514667


namespace option_b_correct_l514_514480

theorem option_b_correct (a : ℝ) : (-a)^3 / (-a)^2 = -a :=
by sorry

end option_b_correct_l514_514480


namespace difference_between_good_and_bad_numbers_l514_514587

def is_good_number (n : ℕ) : Prop :=
  n.digits.count 2 > n.digits.count 3

def is_bad_number (n : ℕ) : Prop :=
  n.digits.count 3 > n.digits.count 2

def count_good_numbers_up_to (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter is_good_number |>.card

def count_bad_numbers_up_to (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter is_bad_number |>.card

theorem difference_between_good_and_bad_numbers : count_good_numbers_up_to 2023 - count_bad_numbers_up_to 2023 = 22 := 
  sorry

end difference_between_good_and_bad_numbers_l514_514587


namespace contrapositive_equivalence_l514_514980

-- Definitions based on the conditions
variables (R S : Prop)

-- Statement of the proof
theorem contrapositive_equivalence (h : ¬R → S) : ¬S → R := 
sorry

end contrapositive_equivalence_l514_514980


namespace p_necessary_not_sufficient_for_q_l514_514641

open Real

noncomputable def p (x : ℝ) : Prop := |x| < 3
noncomputable def q (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem p_necessary_not_sufficient_for_q : 
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬ q x) :=
by
  sorry

end p_necessary_not_sufficient_for_q_l514_514641


namespace birds_joined_l514_514501

-- Definitions based on the conditions of the problem
def initial_birds := 4
def current_birds := 10

-- Proof statement
theorem birds_joined (initial_birds current_birds : ℕ) (h₁ : initial_birds = 4) (h₂ : current_birds = 10) : ∃ b, b = current_birds - initial_birds ∧ b = 6 :=
by
  use current_birds - initial_birds
  split
  · sorry -- proof of b = current_birds - initial_birds
  · sorry -- proof of b = 6

end birds_joined_l514_514501


namespace perimeter_of_triangle_l514_514816

theorem perimeter_of_triangle
  (r : ℝ) (DP : ℝ) (PE : ℝ) (P: Point) (D E F: Point) (tangent_to: Triangle → Point → Prop)
  (inscribed_circle_radius : Triangle → ℝ) 
  (tangent_point_distance_DE : ∀ (T : Triangle) (P : Point), tangent_to T P → ℝ)
  (tangent_point_distance_DF : ∀ (T : Triangle) (P : Point), tangent_to T P → ℝ)
  (tangent_point_distance_EF : ∀ (T : Triangle) (P : Point), tangent_to T P → ℝ) 
  (DEF : Triangle)
  (H1 : inscribed_circle_radius DEF = 15)
  (H2 : tangent_point_distance_DE DEF P = 19)
  (H3 : tangent_point_distance_DE DEF P + tangent_point_distance_EF DEF P = 50)
  (H4 : tangent_point_distance_DE DEF P + tangent_point_distance_DF DEF P = 50):
  Geometry.perimeter DEF =  1475 / 182 := sorry

end perimeter_of_triangle_l514_514816


namespace triangle_area_perimeter_ratio_l514_514915

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514915


namespace union_M_N_l514_514302

def M : Set ℕ := {1, 2}
def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end union_M_N_l514_514302


namespace divide_equivalence_l514_514195

noncomputable def divide_approx : ℝ :=
  27903.672 / 1946.73

theorem divide_equivalence : divide_approx ≈ 14.340 :=
by
  sorry

end divide_equivalence_l514_514195


namespace max_profit_l514_514714

noncomputable def total_cost (Q : ℝ) : ℝ := 5 * Q^2

noncomputable def demand_non_slytherin (P : ℝ) : ℝ := 26 - 2 * P

noncomputable def demand_slytherin (P : ℝ) : ℝ := 10 - P

noncomputable def combined_demand (P : ℝ) : ℝ :=
  if P >= 13 then demand_non_slytherin P else demand_non_slytherin P + demand_slytherin P

noncomputable def inverse_demand (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q / 2 else 12 - Q / 3

noncomputable def revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then Q * (13 - Q / 2) else Q * (12 - Q / 3)

noncomputable def marginal_revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q else 12 - 2 * Q / 3

noncomputable def marginal_cost (Q : ℝ) : ℝ := 10 * Q

theorem max_profit :
  ∃ Q P TR TC π,
    P = inverse_demand Q ∧
    TR = P * Q ∧
    TC = total_cost Q ∧
    π = TR - TC ∧
    π = 7.69 :=
sorry

end max_profit_l514_514714


namespace area_perimeter_ratio_is_sqrt3_div_2_l514_514955

def side_length : ℝ := 6

def area_of_equilateral_triangle (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

def perimeter_of_equilateral_triangle (s : ℝ) : ℝ := 3 * s

theorem area_perimeter_ratio_is_sqrt3_div_2 :
  area_of_equilateral_triangle side_length / perimeter_of_equilateral_triangle side_length = Real.sqrt 3 / 2 :=
by
  sorry

end area_perimeter_ratio_is_sqrt3_div_2_l514_514955


namespace find_a_value_l514_514583

theorem find_a_value : (15^2 * 8^3 / 256 = 450) :=
by
  sorry

end find_a_value_l514_514583


namespace repeating_decimals_count_l514_514601

theorem repeating_decimals_count :
  { n : ℕ | 1 ≤ n ∧ n ≤ 20 ∧ (¬∃ k, n = 9 * k) }.to_finset.card = 18 :=
by
  sorry

end repeating_decimals_count_l514_514601


namespace smallest_positive_period_determine_a_b_l514_514497

noncomputable def f (x a b : ℝ) : ℝ :=
  sin (x + π / 6) + sin (x - π / 6) + a * cos x + b

-- Prove smallest positive period of f(x) is 2π
theorem smallest_positive_period (a b : ℝ) :
  ∀ x, f x a b = f (x + 2 * π) a b :=
sorry

-- Prove values of a and b given monotonicity and min value conditions
theorem determine_a_b (a b : ℝ) (h : ∀ (x : ℝ), -π / 3 ≤ x ∧ x ≤ 0 → f x a b ≥ 2) :
  f 0 a b = 2 → 
  f (-π / 3) a b = 2 → 
  (a = -1 ∧ b = 4) :=
sorry

end smallest_positive_period_determine_a_b_l514_514497


namespace p_plus_q_eq_zero_l514_514698

-- Lean 4 Statement
theorem p_plus_q_eq_zero (p q : ℝ) (h : (X^2 + p * X + q = 0) → (1 : ℂ) + (1 : ℑ) ∈ polynomial_roots_complex (X^2 + p * X + q)) : p + q = 0 :=
by {
  have h_conjugate : (1 : ℂ) + (1 : ℑ).conj ∈ polynomial_roots_complex (X^2 + p * X + q),
  { sorry }, -- Since the coefficients are real, the conjugate root must also be a root

  have h_sum_of_roots : (1 : ℝ) + (1 : ℤ) + (1 : ℝ) - (1 : ℤ) = -p,
  { sorry }, -- Using Vieta's formulas to relate the sum of the roots to the coefficient -p

  have h_product_of_roots : (1 : ℝ) + (1 : ℤ) * (1 : ℝ) - (1 : ℤ) = q,
  { sorry }, -- Using Vieta's formulas to relate the product of the roots to the constant term q

  linarith, -- Hence, p + q = 0
}

end p_plus_q_eq_zero_l514_514698


namespace simple_random_sampling_independent_l514_514337

-- Define the conditions
def simple_random_sampling (selection: Type*) : Prop :=
  ∀ (individual: selection) (order: list selection),
    (prob_individual_selected individual order = prob_individual_selected individual (permute order))

-- Define the probability function
noncomputable def prob_individual_selected {selection: Type*} (individual: selection) (order: list selection) : ℝ :=
  arbitrary ℝ  -- Placeholder for the actual probability calculation

-- Define permutation function (assuming an existing function permute)
noncomputable def permute {α : Type*} (l : list α) -> list α := sorry

-- Define the main theorem
theorem simple_random_sampling_independent {selection: Type*} 
  (individual: selection) (order: list selection) :
  simple_random_sampling selection → 
  (prob_individual_selected individual order) = (prob_individual_selected individual (permute order)) :=
sorry

end simple_random_sampling_independent_l514_514337


namespace max_profit_l514_514713

noncomputable def total_cost (Q : ℝ) : ℝ := 5 * Q^2

noncomputable def demand_non_slytherin (P : ℝ) : ℝ := 26 - 2 * P

noncomputable def demand_slytherin (P : ℝ) : ℝ := 10 - P

noncomputable def combined_demand (P : ℝ) : ℝ :=
  if P >= 13 then demand_non_slytherin P else demand_non_slytherin P + demand_slytherin P

noncomputable def inverse_demand (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q / 2 else 12 - Q / 3

noncomputable def revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then Q * (13 - Q / 2) else Q * (12 - Q / 3)

noncomputable def marginal_revenue (Q : ℝ) : ℝ :=
  if Q <= 6 then 13 - Q else 12 - 2 * Q / 3

noncomputable def marginal_cost (Q : ℝ) : ℝ := 10 * Q

theorem max_profit :
  ∃ Q P TR TC π,
    P = inverse_demand Q ∧
    TR = P * Q ∧
    TC = total_cost Q ∧
    π = TR - TC ∧
    π = 7.69 :=
sorry

end max_profit_l514_514713


namespace smallest_five_digit_congruent_11_mod_14_l514_514970

theorem smallest_five_digit_congruent_11_mod_14 : ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 14 = 11 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 14 = 11 → n ≤ m := 
begin
  use 10007,
  repeat {sorry}
end

end smallest_five_digit_congruent_11_mod_14_l514_514970


namespace chord_length_square_l514_514171

theorem chord_length_square {r₁ r₂ R : ℝ} (r1_eq_4 : r₁ = 4) (r2_eq_8 : r₂ = 8) 
  (R_eq_12 : R = 12)
  (externally_tangent : tangent r₁ r₂)
  (internally_tangent_1 : tangent_internally r₁ R)
  (internally_tangent_2 : tangent_internally r₂ R)
  : exists (PQ : ℝ), PQ^2 = 3584 / 9 :=
by sorry

end chord_length_square_l514_514171


namespace mike_spent_on_new_tires_l514_514243

-- Define the given amounts
def amount_spent_on_speakers : ℝ := 118.54
def total_amount_spent_on_car_parts : ℝ := 224.87

-- Define the amount spent on new tires
def amount_spent_on_new_tires : ℝ := total_amount_spent_on_car_parts - amount_spent_on_speakers

-- The theorem we want to prove
theorem mike_spent_on_new_tires : amount_spent_on_new_tires = 106.33 :=
by
  -- the proof would go here
  sorry

end mike_spent_on_new_tires_l514_514243


namespace problem_conditions_l514_514484

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

theorem problem_conditions :
  (√2 : ℝ) * ∥a - b∥ ∥a∥ + ∥b∥ ∧
  ∥(2 : ℝ) - 3 * ∥(- 1 / 2 : ℝ) ∥(3 / 4 : ℝ) ∧
  ∥a∥ = ∥b∥  ∥a - b∥ ∧
  (∥b - (5 / 5 : ℝ) * a∥ = sorry) → 
  ¬(∥Projection b a∥ = sqrt(10/2)*a)  ∧
  (∠a (a + b) ≠ 60°) → 
  ∃ h : ℝ, h ≠ 0  h ≠ ∞ :=
by sorry

end problem_conditions_l514_514484


namespace find_b_l514_514212

theorem find_b (b : ℕ) (h1 : 40 < b) (h2 : b < 120) 
    (h3 : b % 4 = 3) (h4 : b % 5 = 3) (h5 : b % 6 = 3) : 
    b = 63 := by
  sorry

end find_b_l514_514212


namespace determinant_evaluation_l514_514201

-- Define the variables involved
variables {x y z : ℝ}

-- Define the 3x3 matrix
def matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2 * x, y + z],
    ![1, x + y, z],
    ![1, x, y + 2 * z]]

-- Define the expected determinant
def expected_determinant : ℝ := x * y + 3 * x * z - z - y

-- State the theorem that needs to be proved
theorem determinant_evaluation : matrix.det = expected_determinant :=
  sorry

end determinant_evaluation_l514_514201


namespace ratio_eq_sqrt3_div_2_l514_514942

-- Define the side length of the equilateral triangle
def s := 6

-- Define the formula for the area of an equilateral triangle
def area (s : ℝ) : ℝ := (s^2 * Real.sqrt 3) / 4

-- Define the formula for the perimeter of an equilateral triangle
def perimeter (s : ℝ) : ℝ := 3 * s

-- Calculate the ratio of the area to the perimeter
def ratio (s : ℝ) : ℝ := area s / perimeter s

-- Prove the ratio for the given side length
theorem ratio_eq_sqrt3_div_2 : ratio s = Real.sqrt 3 / 2 := by
  sorry

end ratio_eq_sqrt3_div_2_l514_514942


namespace divisors_sum_l514_514231

theorem divisors_sum (a b c d e : ℤ) (ha : a = 48) (hb : b = 96) (hc : c = -24) (hd : d = 144) (he : e = 120) :
  ∑ x in (Finset.filter (λ x, ∀ y ∈ {a, b, c, d, e}, x ∣ y) (Finset.range (Nat.gcd_list [|48, 96, 24, 144, 120|] + 1))), x = 60 :=
by 
  -- Details of the proof
  sorry


end divisors_sum_l514_514231


namespace min_value_of_f_eq_3_set_of_x_satisfying_f_le_5_l514_514293

noncomputable def f (x a : ℝ) : ℝ := abs (x - 4) + abs (x - a)

theorem min_value_of_f_eq_3 {a : ℝ} (h : a > 1) (h_min : ∃ x, f x a = 3) : a = 7 :=
sorry

theorem set_of_x_satisfying_f_le_5 : {x : ℝ | f x 7 ≤ 5} = set.Icc 3 8 :=
sorry

end min_value_of_f_eq_3_set_of_x_satisfying_f_le_5_l514_514293


namespace find_multiple_l514_514507

theorem find_multiple :
  let n := 220080
  let a := 555
  let b := 445
  let s := a + b
  let d := a - b
  let r := 80 in
  (n - r) / s / d = 2 := 
by
  sorry

end find_multiple_l514_514507


namespace angle_between_clock_hands_at_3_25_l514_514075

theorem angle_between_clock_hands_at_3_25 : 
  let minute_angle := 25 * 6 
  let hour_angle := 3 * 30 + 25 * 0.5 
  let angle := abs (hour_angle - minute_angle)
  angle = 47.5 := 
by {
  sorry 
}

end angle_between_clock_hands_at_3_25_l514_514075


namespace yancheng_marathon_half_marathon_estimated_probability_l514_514422

noncomputable def estimated_probability
  (surveyed_participants_frequencies : List (ℕ × Real)) : Real :=
by
  -- Define the surveyed participants and their corresponding frequencies
  -- In this example, [(20, 0.35), (50, 0.40), (100, 0.39), (200, 0.415), (500, 0.418), (2000, 0.411)]
  sorry

theorem yancheng_marathon_half_marathon_estimated_probability :
  let surveyed_participants_frequencies := [
    (20, 0.350),
    (50, 0.400),
    (100, 0.390),
    (200, 0.415),
    (500, 0.418),
    (2000, 0.411)
  ]
  estimated_probability surveyed_participants_frequencies = 0.40 :=
by
  sorry

end yancheng_marathon_half_marathon_estimated_probability_l514_514422


namespace emily_avg_speed_l514_514557

def total_distance : ℕ := 450 + 540
def total_time: ℝ := 7 + 0.5 + 1 + 8
def avg_speed : ℝ := total_distance / total_time

theorem emily_avg_speed :
  avg_speed = 60 := by
  sorry

end emily_avg_speed_l514_514557


namespace range_f_l514_514477

noncomputable def f (x : ℝ) := 8 * x + 1 / (4 * x - 5)

theorem range_f (x : ℝ) (h : x < 5 / 4) : 
  set_of (λ y, y ≤ 10 - 2 * real.sqrt 2) = set.range (λ x, f x) :=
sorry

end range_f_l514_514477


namespace axis_of_symmetry_l514_514680

open Real

theorem axis_of_symmetry (g : ℝ → ℝ) (h : ∀ x, g(x) = g(3 - x)) : ∀ y, y = g 1.5 :=
by
  sorry

end axis_of_symmetry_l514_514680


namespace distinct_prime_factors_of_n_l514_514449

def letters : List (Char × Nat) :=
  [('c', 1), ('e', 2), ('s', 2), ('o', 2), ('n', 1), 
   ('t', 1), ('i', 1), ('a', 1), ('x', 1), ('u', 1)]

def accentOptions : Char → Nat
| 'c' => 2
| 'e' => 5
| 's' => 1
| 'o' => 3
| 'n' => 1
| 't' => 1
| 'i' => 2
| 'a' => 3
| 'x' => 1
| 'u' => 4
| _   => 1

noncomputable def compute_n : Nat :=
  let wordSplits := Nat.choose 12 2
  let accentCounts := letters.foldr (λ (letter_freq : Char × Nat) acc => 
   acc * (accentOptions letter_freq.fst) ^ letter_freq.snd) 1
  wordSplits * accentCounts

theorem distinct_prime_factors_of_n : (nat.factors (compute_n)).toFinset.card = 4 := by
  sorry

end distinct_prime_factors_of_n_l514_514449


namespace non_similar_triangles_count_l514_514307

theorem non_similar_triangles_count : 
  ∃ (n : ℕ) (d : ℕ), (n - d) + n + (n + d) = 180 ∧ d % 5 = 0 ∧ 0 < d ∧ d < 60 ∧ (finset.range 12).card = 11 := 
begin
  sorry
end

end non_similar_triangles_count_l514_514307


namespace sum_of_fourth_powers_under_hundred_l514_514077

theorem sum_of_fourth_powers_under_hundred : 
    ∑ n in {1, 16, 81}, id n = 98 := sorry

end sum_of_fourth_powers_under_hundred_l514_514077


namespace perimeter_of_triangle_l514_514815

theorem perimeter_of_triangle
  (r : ℝ) (DP : ℝ) (PE : ℝ) (P: Point) (D E F: Point) (tangent_to: Triangle → Point → Prop)
  (inscribed_circle_radius : Triangle → ℝ) 
  (tangent_point_distance_DE : ∀ (T : Triangle) (P : Point), tangent_to T P → ℝ)
  (tangent_point_distance_DF : ∀ (T : Triangle) (P : Point), tangent_to T P → ℝ)
  (tangent_point_distance_EF : ∀ (T : Triangle) (P : Point), tangent_to T P → ℝ) 
  (DEF : Triangle)
  (H1 : inscribed_circle_radius DEF = 15)
  (H2 : tangent_point_distance_DE DEF P = 19)
  (H3 : tangent_point_distance_DE DEF P + tangent_point_distance_EF DEF P = 50)
  (H4 : tangent_point_distance_DE DEF P + tangent_point_distance_DF DEF P = 50):
  Geometry.perimeter DEF =  1475 / 182 := sorry

end perimeter_of_triangle_l514_514815


namespace triangle_area_perimeter_ratio_l514_514923

theorem triangle_area_perimeter_ratio (s : ℕ) (h : s = 6) : 
    (let P := 3 * s in 
    let A := (s^2 * Real.sqrt 3) / 4 in 
    A / P = Real.sqrt 3 / 2) :=
by
  rw h
  sorry

end triangle_area_perimeter_ratio_l514_514923


namespace missing_condition_l514_514999

theorem missing_condition (x y : ℕ) 
  (h1 : y = 2 * x + 9) 
  (h2 : y = 3 * (x - 2)) : 
  "Three people ride in one car, and there are two empty cars" :=
by sorry

end missing_condition_l514_514999


namespace area_enclosed_by_lines_and_curve_l514_514424

-- Define the functions and conditions given in the problem
def x1 : ℝ := Real.pi / 2
def x2 : ℝ := 3 * Real.pi / 2
def y (x : ℝ) := Real.cos x

-- Define the integral representing the area under the curve y = cos x between x = pi/2 and x = 3*pi/2
def area : ℝ := ∫ x in x1..x2, - (y x)

-- State the theorem that the area is equal to 2
theorem area_enclosed_by_lines_and_curve : area = 2 := sorry

end area_enclosed_by_lines_and_curve_l514_514424


namespace find_k_value_l514_514234

theorem find_k_value
  (k : ℤ)
  (h : 3 * 2^2001 - 3 * 2^2000 - 2^1999 + 2^1998 = k * 2^1998) : k = 11 :=
by
  sorry

end find_k_value_l514_514234


namespace find_lambda_l514_514248

/-- Given vectors a and b, and the condition that (a + λb) is orthogonal to a,
    find the value of λ --/
theorem find_lambda : 
  let a := (1 : ℝ, 0 : ℝ)
  let b := (1 : ℝ, 1 : ℝ)
  ∃ λ : ℝ, (a.1 + λ * b.1) * a.1 + (a.2 + λ * b.2) * a.2 = 0 ∧ λ = -1 := 
by
  sorry

end find_lambda_l514_514248


namespace root_max_imaginary_part_is_77_14_l514_514219

noncomputable def polynomial := Polynomial.Coeff (Polynomial.X ^ 6 - Polynomial.X ^ 4 + Polynomial.X ^ 3 - Polynomial.X + 1)

def root_max_imaginary_part_within_range := 
  ∃ φ : ℝ, -90 ≤ φ ∧ φ ≤ 90 ∧ 
  (∃ z : ℂ, polynomial.eval z polynomial = 0 ∧ complex.imag_part z = complex.sin (φ * real.pi / 180)) ∧ 
  (∀ z' : ℂ, polynomial.eval z' polynomial = 0 → complex.imag_part z' ≤ complex.imag_part z)

theorem root_max_imaginary_part_is_77_14 :
    ∃ φ : ℝ, φ = 77.14 ∧ root_max_imaginary_part_within_range :=
  sorry

end root_max_imaginary_part_is_77_14_l514_514219


namespace equilateral_triangle_ratio_l514_514926

theorem equilateral_triangle_ratio (s : ℝ) (h : s = 6) : 
  let area := (s^2 * Real.sqrt 3) / 4 in
  let perimeter := 3 * s in
  area / perimeter = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514926


namespace central_angle_remains_unchanged_l514_514442

theorem central_angle_remains_unchanged
  (r l : ℝ)
  (h_r : r > 0)
  (h_l : l > 0) :
  (l / r) = (2 * l) / (2 * r) :=
by
  sorry

end central_angle_remains_unchanged_l514_514442


namespace mass_percent_oxygen_h3bo3_is_77_58_l514_514188

structure Element (name : String) :=
  (molar_mass : ℚ)

def H : Element := { name := "Hydrogen", molar_mass := 1.01 }
def B : Element := { name := "Boron", molar_mass := 10.81 }
def O : Element := { name := "Oxygen", molar_mass := 16.00 }

noncomputable def molar_mass_h3bo3 : ℚ :=
  3 * H.molar_mass + B.molar_mass + 3 * O.molar_mass

noncomputable def mass_percent_oxygen_h3bo3 : ℚ :=
  (3 * O.molar_mass / molar_mass_h3bo3) * 100

theorem mass_percent_oxygen_h3bo3_is_77_58 :
  mass_percent_oxygen_h3bo3 = 77.58 := 
by 
  sorry

end mass_percent_oxygen_h3bo3_is_77_58_l514_514188


namespace average_percentage_reduction_l514_514102

variable (P : ℝ) (x : ℝ)
variable (h1 : 0 < P)
variable (h2 : 0 < x)
variable (h3 : x < 1)

theorem average_percentage_reduction :
  let new_price := P * (0.64)^2 in
  (1 - x) ^ 2 = 0.4096 → x = 0.20 :=
by
  intro new_price
  sorry

end average_percentage_reduction_l514_514102


namespace range_of_f_l514_514292

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem range_of_f : set.image f (set.Icc 1 5) = set.Icc 4 (29 / 5) :=
by
  sorry

end range_of_f_l514_514292


namespace prob_two_consecutive_heads_l514_514503

theorem prob_two_consecutive_heads (n : ℕ) (p : ℝ) (h : n = 4) (hp : p = 1 / 2) : 
  let prob := ∑ x in (finset.filter (λ (s : list bool), ∃ (i : ℕ), i < s.length - 1 ∧ s[i] = tt ∧ s[i+1] = tt) 
                      (finset.univ : finset (list bool))), 
                      p ^ s.length : ℝ
  in prob = 9 / 16 := sorry

end prob_two_consecutive_heads_l514_514503


namespace domain_of_sqrt_function_l514_514807

theorem domain_of_sqrt_function (x : ℝ) :
  (x + 4 ≥ 0) ∧ (1 - x ≥ 0) ∧ (x ≠ 0) ↔ (-4 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) := 
sorry

end domain_of_sqrt_function_l514_514807


namespace certain_number_is_65_l514_514532

-- Define the conditions
variables (N : ℕ)
axiom condition1 : N < 81
axiom condition2 : ∀ k : ℕ, k ≤ 15 → N + k < 81
axiom last_consecutive : N + 15 = 80

-- Prove the theorem
theorem certain_number_is_65 (h1 : N < 81) (h2 : ∀ k : ℕ, k ≤ 15 → N + k < 81) (h3 : N + 15 = 80) : N = 65 :=
sorry

end certain_number_is_65_l514_514532


namespace probability_different_colors_l514_514702

def total_chips : ℕ := 12

def blue_chips : ℕ := 5
def yellow_chips : ℕ := 3
def red_chips : ℕ := 4

def prob_diff_color (x y : ℕ) : ℚ :=
(x / total_chips) * (y / total_chips) + (y / total_chips) * (x / total_chips)

theorem probability_different_colors :
  prob_diff_color blue_chips yellow_chips +
  prob_diff_color blue_chips red_chips +
  prob_diff_color yellow_chips red_chips = 47 / 72 := by
sorry

end probability_different_colors_l514_514702


namespace equilateral_triangle_ratio_l514_514968

theorem equilateral_triangle_ratio (s : ℕ) (h : s = 6) :
  (let area := s^2 * Real.sqrt 3 / 4 in
   let perimeter := 3 * s in
   area / perimeter = Real.sqrt 3 / 2) :=
by
  intros
  sorry

end equilateral_triangle_ratio_l514_514968


namespace num_two_digit_markers_l514_514616

def distance_between (A B : Nat) : Nat := 999

def km_marker (n : Nat) : String :=
  let str := Nat.digits 10 n
  String.intercalate "|" (str.map (λ l => l))

def has_two_distinct_digits (s : String) : Prop :=
  ∃ (a b : Char), a ≠ b ∧ ∀ (c : Char), str.count c = 0 ∨ str.count c = 1

theorem num_two_digit_markers :
  let markers := List.range (distance_between 0 999 + 1)
  let km_markers := markers.map km_marker
  (km_markers.filter (λ m => has_two_distinct_digits m)).length = 40 :=
by
  sorry

end num_two_digit_markers_l514_514616


namespace difference_in_investments_l514_514136

noncomputable def aliceInterest (P r : ℝ) (n t : ℕ) :=
  P * (1 + r / n)^(n * t)

noncomputable def bobInterest (P r : ℝ) (n t : ℕ) :=
  P * (1 + r / n)^(n * t)

theorem difference_in_investments :
  let P := 30000
  let r := 0.05
  let t := 3
  let A_alice := aliceInterest P r 1 t
  let A_bob := bobInterest P r 12 t
  A_bob - A_alice = 121.56 :=
by
  let P := 30000
  let r := 0.05
  let t := 3
  let A_alice := aliceInterest P r 1 t
  let A_bob := bobInterest P r 12 t
  sorry

end difference_in_investments_l514_514136


namespace proposition1_proposition2_proposition3_proposition4_l514_514401

def f (x : ℝ) : ℝ := Real.sin (2 * x) - Real.cos (2 * x)

theorem proposition1 : Function.periodic f Real.pi := sorry

theorem proposition2 : ¬ Function.symmetric f (Real.pi / 4) := sorry

theorem proposition3 : Function.center_of_symmetry f (Real.pi / 8, 0) := sorry

theorem proposition4 : ¬ ( ∀ x, f (x + Real.pi / 4) = sqrt 2 * Real.sin (2 * x) ) := sorry

end proposition1_proposition2_proposition3_proposition4_l514_514401


namespace equilateral_triangle_ratio_l514_514872

-- Define the conditions and statement we need to prove:
theorem equilateral_triangle_ratio :
  let s : ℕ := 6 in
  let A := (s ^ 2 * Real.sqrt 3) / 4 in
  let P := 3 * s in
  A / P = Real.sqrt 3 / 2 :=
by
  sorry

end equilateral_triangle_ratio_l514_514872


namespace smallest_n_for_g4_l514_514745

def g (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc a => acc + (List.range (n + 1)).countP (λ b => a * a + b * b = n)) 0

theorem smallest_n_for_g4 : ∃ n : ℕ, g n = 4 ∧ 
  (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 65
  -- Proof would go here
  sorry

end smallest_n_for_g4_l514_514745


namespace three_digit_number_count_correct_l514_514654

noncomputable
def count_three_digit_numbers (digits : List ℕ) : ℕ :=
  if h : digits.length = 5 then
    (5 * 4 * 3 : ℕ)
  else
    0

theorem three_digit_number_count_correct :
  count_three_digit_numbers [1, 3, 5, 7, 9] = 60 :=
by
  unfold count_three_digit_numbers
  simp only [List.length, if_pos]
  rfl

end three_digit_number_count_correct_l514_514654


namespace sarah_rye_flour_l514_514403

-- Definitions
variables (b c p t r : ℕ)

-- Conditions
def condition1 : Prop := b = 10
def condition2 : Prop := c = 3
def condition3 : Prop := p = 2
def condition4 : Prop := t = 20

-- Proposition to prove
theorem sarah_rye_flour : condition1 b → condition2 c → condition3 p → condition4 t → r = t - (b + c + p) → r = 5 :=
by
  intros h1 h2 h3 h4 hr
  rw [h1, h2, h3, h4] at hr
  exact hr

end sarah_rye_flour_l514_514403
