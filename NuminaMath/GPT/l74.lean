import Mathlib
import Mathlib.Algebra.ArithmeticFunction
import Mathlib.Algebra.BaseEquiv
import Mathlib.Algebra.Divisibility
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.Int.ModEq
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Init.Data.Real.Basic
import Mathlib.Probability
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Trigonometric
import Real
import mathlib

namespace speed_of_current_11_00448_l74_74453

/-- 
  The speed at which a man can row a boat in still water is 25 kmph.
  He takes 7.999360051195905 seconds to cover 80 meters downstream.
  Prove that the speed of the current is 11.00448 km/h.
-/
theorem speed_of_current_11_00448 :
  let speed_in_still_water_kmph := 25
  let distance_m := 80
  let time_s := 7.999360051195905
  (distance_m / time_s) * 3600 / 1000 - speed_in_still_water_kmph = 11.00448 :=
by
  sorry

end speed_of_current_11_00448_l74_74453


namespace standard_deviation_of_sample_l74_74454

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  data.map (λ x => (x - m) ^ 2).sum / data.length

noncomputable def stddev (data : List ℝ) : ℝ :=
  (variance data).sqrt

theorem standard_deviation_of_sample :
  stddev [4, 2, 1, 0, -2] = 2 := 
  sorry

end standard_deviation_of_sample_l74_74454


namespace largest_binomial_coefficient_in_expansion_l74_74356

theorem largest_binomial_coefficient_in_expansion :
  let a := (x : ℝ)
  let b := (1 / real.sqrt x)
  let n := (10 : ℕ)
  let t := (n+1) / 2
  ∀ r : ℕ, 0 ≤ r ∧ r ≤ n →
  (r = t) → (binom n r ≥ binom n r) :=
by
  sorry

end largest_binomial_coefficient_in_expansion_l74_74356


namespace determine_day_from_statements_l74_74874

/-- Define the days of the week as an inductive type. -/
inductive Day where
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
  deriving DecidableEq, Repr

open Day

/-- Define the properties of the lion lying on specific days. -/
def lion_lies (d : Day) : Prop :=
  d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Define the properties of the lion telling the truth on specific days. -/
def lion_truth (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday ∨ d = Sunday

/-- Define the properties of the unicorn lying on specific days. -/
def unicorn_lies (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday

/-- Define the properties of the unicorn telling the truth on specific days. -/
def unicorn_truth (d : Day) : Prop :=
  d = Sunday ∨ d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Function to determine the day before a given day. -/
def yesterday (d : Day) : Day :=
  match d with
  | Monday    => Sunday
  | Tuesday   => Monday
  | Wednesday => Tuesday
  | Thursday  => Wednesday
  | Friday    => Thursday
  | Saturday  => Friday
  | Sunday    => Saturday

/-- Define the lion's statement: "Yesterday was a day when I lied." -/
def lion_statement (d : Day) : Prop :=
  lion_lies (yesterday d)

/-- Define the unicorn's statement: "Yesterday was a day when I lied." -/
def unicorn_statement (d : Day) : Prop :=
  unicorn_lies (yesterday d)

/-- Prove that today must be Thursday given the conditions and statements. -/
theorem determine_day_from_statements (d : Day) :
    lion_statement d ∧ unicorn_statement d → d = Thursday := by
  sorry

end determine_day_from_statements_l74_74874


namespace total_mangoes_l74_74183

-- Definitions of the entities involved
variables (Alexis Dilan Ashley Ben : ℚ)

-- Conditions given in the problem
def condition1 : Prop := Alexis = 4 * (Dilan + Ashley) ∧ Alexis = 60
def condition2 : Prop := Ashley = 2 * Dilan
def condition3 : Prop := Ben = (1/2) * (Dilan + Ashley)

-- The theorem we want to prove: total mangoes is 82.5
theorem total_mangoes (Alexis Dilan Ashley Ben : ℚ)
  (h1 : condition1 Alexis Dilan Ashley)
  (h2 : condition2 Dilan Ashley)
  (h3 : condition3 Dilan Ashley Ben) :
  Alexis + Dilan + Ashley + Ben = 82.5 :=
sorry

end total_mangoes_l74_74183


namespace sushi_father_lollipops_l74_74432

variable (x : ℕ)

theorem sushi_father_lollipops (h : x - 5 = 7) : x = 12 := by
  sorry

end sushi_father_lollipops_l74_74432


namespace domino_tiling_even_stars_l74_74895

theorem domino_tiling_even_stars :
  ∀ (r c a b : ℕ) (Rc : matrix (fin r) (fin c) bool),
  r = 5 →
  c = 200 →
  a = 1 →
  b = 2 →
  (∀ i : fin r, ∃ k : ℕ, Rc.row i = repeat tt k ++ repeat ff (c - k) ∧ k % 2 = 0) →
  (∀ j : fin c, ∃ k : ℕ, Rc.col j = repeat tt k ++ repeat ff (r - k) ∧ k % 2 = 0) →
  ∃ f : (fin 500) → (fin 5 × fin 200) × (fin 5 × fin 200),
    ∀ n, f n = (cv, rs) ∧ ((matrix.1 cv).1 * 1 + 2 + (matrix.1 rs).1 * 1 + 2 = 0) := 
sorry

end domino_tiling_even_stars_l74_74895


namespace num_hard_shell_tacos_family_bought_l74_74864

-- Define the conditions
def soft_taco_price : ℝ := 2
def hard_shell_taco_price : ℝ := 5
def num_soft_tacos_family : ℕ := 3
def num_customers : ℕ := 10
def soft_tacos_per_customer : ℕ := 2
def total_revenue : ℝ := 66

-- Define the proof problem
theorem num_hard_shell_tacos_family_bought : 
  ∃ H : ℝ, (num_soft_tacos_family * soft_taco_price
          + num_customers.succ * (soft_tacos_per_customer * soft_taco_price)
          + H * hard_shell_taco_price = total_revenue 
          ∧ H = 4) :=
begin
  have num_soft_tacos_customers := num_customers * soft_tacos_per_customer,
  have revenue_soft_tacos := (num_soft_tacos_family + num_soft_tacos_customers) * soft_taco_price,
  use ((total_revenue - revenue_soft_tacos) / hard_shell_taco_price),
  split,
  {
    rw [mul_add, mul_comm num_soft_tacos_family soft_taco_price, mul_assoc],
    exact add_assoc _ _ _,
  },
  {
    -- This will need to be proven
    -- sorry is here to skip detailed solution steps
    sorry,
  },
end

end num_hard_shell_tacos_family_bought_l74_74864


namespace intersecting_translated_polyhedra_l74_74245

theorem intersecting_translated_polyhedra
  (M : Polyhedron)
  (A : Vertex M)
  (vertices : Fin 9 → Vertex M)
  (h_convex : Convex M)
  (h_distinct : ∀ i j : Fin 9, i ≠ j → vertices i ≠ vertices j)
  (h_translation: ∀ i : Fin 9, i ≠ 0 → TranslatedPolyhedron M (vertices i) = M) :
  ∃ i j : Fin 8, i ≠ j ∧ ∃ p, p ∈ (TranslatedPolyhedron M (vertices i)) ∧ p ∈ (TranslatedPolyhedron M (vertices j)) :=
sorry

end intersecting_translated_polyhedra_l74_74245


namespace count_arrangements_l74_74465

noncomputable def num_arrangements : Nat :=
  96

-- Let M be the set of males
def M : Set (String) := { "M1", "M2", "M3" }

-- Let F be the set of females
def F : Set (String) := { "F1", "F2", "F3" }

-- Define a couple as (Mi, Fi)
def couple (i : Fin 3) : String × String :=
  match i with
  | ⟨0, _⟩ => ("M1", "F1")
  | ⟨1, _⟩ => ("M2", "F2")
  | ⟨2, _⟩ => ("M3", "F3")

-- Define arrangement conditions
def valid_arrangement (arr : Vector (String × String) 6) : Prop :=
  ∀ (i j : Fin 6),
    -- No couple can sit next to each other in the same row.
    (i.1 / 3 = j.1 / 3) → ¬ (arr.val i = couple j / 3 .fst .snd ∨ arr.val j = couple j / 3 .snd .fst) ∧
    -- No couple can sit one behind the other in the same column.
    (i.1 % 3 = j.1 % 3) → ¬ (arr.val i = couple (i.val / 3).fst .snd ∨ arr.val j = couple (i.val / 3).snd .fst)

-- Main theorem stating the number of valid arrangements
theorem count_arrangements : ∃ arr : Finset (Vector (String × String) 6), 
  (∀ a ∈ arr, valid_arrangement a) ∧ arr.card = num_arrangements := by
  sorry

end count_arrangements_l74_74465


namespace sin_alpha_value_l74_74320

theorem sin_alpha_value {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 2)
  (h : tan (2 * α) = cos α / (2 - sin α)) : sin α = 1 / 4 :=
sorry

end sin_alpha_value_l74_74320


namespace cryptarithm_base_solution_l74_74945

theorem cryptarithm_base_solution :
  ∃ (K I T : ℕ) (d : ℕ), 
    O = 0 ∧
    2 * T = I ∧
    T + 1 = K ∧
    K + I = d ∧ 
    d = 7 ∧ 
    K ≠ I ∧ K ≠ T ∧ K ≠ O ∧
    I ≠ T ∧ I ≠ O ∧
    T ≠ O :=
sorry

end cryptarithm_base_solution_l74_74945


namespace employee_y_payment_l74_74499

variable (x y : ℝ)

def total_payment (x y : ℝ) : ℝ := x + y
def x_payment (y : ℝ) : ℝ := 1.20 * y

theorem employee_y_payment : (total_payment x y = 638) ∧ (x = x_payment y) → y = 290 :=
by
  sorry

end employee_y_payment_l74_74499


namespace correct_base_notation_l74_74117

theorem correct_base_notation:
  (∀ (n : ℕ) (b : ℕ), b < 9 → n < b → n ∈ {7, 5, 1}) → 
  (¬∀ (n : ℕ) (b : ℕ), b < 7 → n < b → n ∈ {7, 5, 1}) → 
  (¬∀ (n : ℕ) (b : ℕ), b < 2 → n < b → n ∈ {9, 0, 1}) →
  (¬∀ (n : ℕ) (b : ℕ), b < 12 → n < b → n ∈ {0, 9, 5}) → 
  (7 < 9 ∧ 5 < 9 ∧ 1 < 9) :=
begin
  sorry
end

end correct_base_notation_l74_74117


namespace remainder_1234567_div_by_137_l74_74200

theorem remainder_1234567_div_by_137 :
  (1234567 % 137) = 102 :=
by {
  sorry
}

end remainder_1234567_div_by_137_l74_74200


namespace four_distinct_elements_with_integer_geometric_mean_l74_74964

theorem four_distinct_elements_with_integer_geometric_mean :
  ∀ (M : Finset ℕ), 
    M.card = 1985 →
    (∀ m ∈ M, ∀ p, prime p → p ∣ m → p ≤ 26) →
    ∃ a b c d ∈ M, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ int.sqrt (a * b * c * d) ^ 4 = a * b * c * d :=
by
  sorry

end four_distinct_elements_with_integer_geometric_mean_l74_74964


namespace handshake_problem_l74_74342

theorem handshake_problem (N : ℕ) (hN : N > 3) :
  (∃ A : finset ℕ, A.card = N ∧ 
  ∃ (a₁ a₂ : ℕ) (ha₁ : a₁ ∈ A) (ha₂ : a₂ ∈ A), a₁ ≠ a₂ ∧ 
    (∀ a₃ ∈ A, a₃ ≠ a₁ → a₃ ≠ a₂ → a₃ ∈ A)) →
  ∀ (a : ℕ) (ha : a ∈ A), ∃ (B : finset ℕ), B.card = N - 2 ∧ (B ⊆ A) :=
begin
  sorry
end

end handshake_problem_l74_74342


namespace problem1_problem2_problem3_l74_74028

namespace ProofProblem

-- Define the function f and the conditions a > 0 and a ≠ 1
variable {a : ℝ} (ha0 : a > 0) (ha1 : a ≠ 1)

-- Define f(x) given parameters a and k
def f (x : ℝ) (k : ℝ) : ℝ := a^x - (k-1) * a^(-x)

-- Problem (1): Prove that k = 2 given f is an odd function
theorem problem1 (hf_odd : ∀ x : ℝ, f x 2 = -f (-x) 2) : 2 = 2 :=
by
  sorry

-- Problem (2): Given f(1) < 0, prove monotonicity and range of t for inequality
theorem problem2 (hf1_neg : f 1 2 < 0) (a_range : 1 > a > 0) :
  (∀ x, f x 2 is_monotonically_decreasing) ∧ (∀ x, x^2 + t * x + 4 > 0 for -3 < t < 5) :=
by
  sorry

-- Problem (3): Given f(1) = 3/2 and g(x) has minimum value -2 on [1, +∞), find m
def g (x : ℝ) (m : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m * f x 2

theorem problem3 (hf1_eq : f 1 2 = 3/2) (hg_min : ∀ x ∈ Icc 1 Real.Inf 1, g x 2 has_min_value (-2)) : 
  m = 2 :=
by
  sorry

end ProofProblem

end problem1_problem2_problem3_l74_74028


namespace vending_machine_beverages_total_l74_74096

theorem vending_machine_beverages_total (n_machines : ℕ) (n_front : ℕ) (n_back : ℕ) (n_top : ℕ) (n_bottom : ℕ)
  (h1 : n_machines = 28)
  (h2 : n_front = 14)
  (h3 : n_back = 20)
  (h4 : n_top = 3)
  (h5 : n_bottom = 2) :
  let total_beverages_in_column := n_front + n_back - 1,
      total_rows := n_top + n_bottom - 1,
      total_beverages_in_vending_machine := total_beverages_in_column * total_rows in
  n_machines * total_beverages_in_vending_machine = 3696 :=
by
  let total_beverages_in_column := n_front + n_back - 1
  let total_rows := n_top + n_bottom - 1
  let total_beverages_in_vending_machine := total_beverages_in_column * total_rows
  have : n_machines * total_beverages_in_vending_machine = 28 * (33 * 4),
    by rw [h1, h2, h3, h4, h5]; rfl
  show n_machines * total_beverages_in_vending_machine = 3696, from this

end vending_machine_beverages_total_l74_74096


namespace triangle_inequality_l74_74719

theorem triangle_inequality (ABC: Triangle) (M : Point) (a b c : ℝ)
  (h1 : a = BC) (h2 : b = CA) (h3 : c = AB) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 3 / (MA^2 + MB^2 + MC^2) := 
sorry

end triangle_inequality_l74_74719


namespace move_and_tangent_l74_74360

section
variables {ρ θ α : ℝ}
noncomputable def line_polar := ρ = - 6 / (3 * cos θ + 4 * sin θ)
noncomputable def curve_rectangular (x y : ℝ) := (x - 3)^2 + (y - 5)^2 = 25
noncomputable def curve_polar (ρ θ : ℝ) := ρ^2 - 6 * ρ * cos θ - 10 * ρ * sin θ + 9 = 0

theorem move_and_tangent (m : ℝ) :
  (∃ m, (m = 5 / 2 ∨ m = 15)) :=
sorry

end

end move_and_tangent_l74_74360


namespace probabilityAtLeastOneThreeIsOneHalf_l74_74523

-- Define the problem
def isValidOutcome (x1 x2 x3 x4 : ℕ) : Prop :=
  (1 ≤ x1 ∧ x1 ≤ 6) ∧
  (1 ≤ x2 ∧ x2 ≤ 6) ∧
  (1 ≤ x3 ∧ x3 ≤ 6) ∧
  (1 ≤ x4 ∧ x4 ≤ 6) ∧
  (x1 + x2 + x3 = x4)

-- Count the total number of valid outcomes
def totalValidOutcomes : ℕ :=
  finset.card $ { (x1, x2, x3, x4) | isValidOutcome x1 x2 x3 x4 }.to_finset

-- Count the total number of valid outcomes where at least one toss is 3
def validOutcomesWithThree : ℕ :=
  finset.card $ { (x1, x2, x3, x4) | isValidOutcome x1 x2 x3 x4 ∧ (x1 = 3 ∨ x2 = 3 ∨ x3 = 3 ∨ x4 = 3) }.to_finset

-- Define the probability of at least one 3 being tossed
noncomputable def probabilityAtLeastOneThree : ℚ :=
  validOutcomesWithThree / totalValidOutcomes

-- Statement to prove
theorem probabilityAtLeastOneThreeIsOneHalf : 
  probabilityAtLeastOneThree = 1 / 2 :=
by sorry

end probabilityAtLeastOneThreeIsOneHalf_l74_74523


namespace alice_wins_5_alice_wins_6_bob_wins_8_l74_74184

-- Define the game parameters: number of seats, initial player (true for Alice, false for Bob)
def game (n : ℕ) (initial_player : Bool) : Prop :=
  ∃ strategy : List ℕ, -- strategies are positions in the list of seats
    (∀ s ∈ strategy, 1 ≤ s ∧ s ≤ n)
    ∧ (∀ i, i < strategy.length - 1 →
          abs (strategy[i] - strategy[i+1]) > 1)
    → ((initial_player ∧ strategy.length % 2 = 1) ∨ (¬initial_player ∧ strategy.length % 2 = 0))

-- Alice's winning strategy for n = 5
theorem alice_wins_5 : game 5 true := sorry

-- Alice's winning strategy for n = 6
theorem alice_wins_6 : game 6 true := sorry

-- Bob's winning strategy for n = 8
theorem bob_wins_8 : game 8 false := sorry

end alice_wins_5_alice_wins_6_bob_wins_8_l74_74184


namespace proof_polynomials_possibility_l74_74010

noncomputable def possible_to_make_identical_polynomials (n : ℕ) (hn : n ≥ 2) : Prop :=
  ∀ (f : ℕ → Polynomial ℤ),
    (∀ k, 1 ≤ k ∧ k ≤ n → (f k) = Polynomial.monomial n 1 + Polynomial.C 1) →
    ∃ (moves : ℕ → List ℕ), ∀ k, 
      1 ≤ k ∧ k ≤ n → 
      (∀ i, f k i = x → k mod 2 ≠ 0) → -- Ensure that the process stops with identical polynomials
      (∃ p : Polynomial ℤ, ∀ i, 1 ≤ i ∧ i ≤ n → f i = p)

theorem proof_polynomials_possibility {n : ℕ} (hn : n ≥ 2) : 
  possible_to_make_identical_polynomials n hn :=
begin
  sorry
end

end proof_polynomials_possibility_l74_74010


namespace bridge_length_is_correct_l74_74826

def speed_km_hr : ℝ := 45
def train_length_m : ℝ := 120
def crossing_time_s : ℝ := 30

noncomputable def speed_m_s : ℝ := speed_km_hr * 1000 / 3600
noncomputable def total_distance_m : ℝ := speed_m_s * crossing_time_s
noncomputable def bridge_length_m : ℝ := total_distance_m - train_length_m

theorem bridge_length_is_correct : bridge_length_m = 255 := by
  sorry

end bridge_length_is_correct_l74_74826


namespace loss_percentage_l74_74324

theorem loss_percentage (original_price sale_price : ℝ) 
  (h1 : original_price = 490) (h2 : sale_price = 465.50) : 
  ((original_price - sale_price) / original_price) * 100 = 5 := 
by
  sorry

end loss_percentage_l74_74324


namespace volume_of_sphere_l74_74166

noncomputable def cuboid_volume (a b c : ℝ) := a * b * c

noncomputable def sphere_volume (r : ℝ) := (4/3) * Real.pi * r^3

theorem volume_of_sphere
  (a b c : ℝ) 
  (sphere_radius : ℝ)
  (h1 : a = 1)
  (h2 : b = Real.sqrt 3)
  (h3 : c = 2)
  (h4 : sphere_radius = Real.sqrt (a^2 + b^2 + c^2) / 2)
  : sphere_volume sphere_radius = (8 * Real.sqrt 2 / 3) * Real.pi := 
by
  sorry

end volume_of_sphere_l74_74166


namespace median_number_of_moons_l74_74809

theorem median_number_of_moons :
  let moons := [0, 0, 0, 1, 2, 3, 5, 17, 20, 25],
      sorted_moons := List.sort moons,
      n := List.length sorted_moons,
      fifth_value := sorted_moons.get ⟨4, by simp [sorted_moons]⟩,
      sixth_value := sorted_moons.get ⟨5, by simp [sorted_moons]⟩
  in n % 2 = 0 → (fifth_value + sixth_value) / 2 = 2.5 :=
by
  intro moons sorted_moons n fifth_value sixth_value evn
  show (fifth_value + sixth_value) / 2 = 2.5
  sorry

end median_number_of_moons_l74_74809


namespace intersection_point_exists_l74_74830

-- Definitions for the conditions
def line_at_param (t : ℝ) : ℝ × ℝ × ℝ := (2 + 4 * t, 1 - 3 * t, -3 - 2 * t)

def on_plane (P : ℝ × ℝ × ℝ) : Prop :=
  3 * P.1 - P.2 + 4 * P.3 = 0

def is_intersection_point (P : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P = line_at_param t ∧ on_plane P

-- The mathematically equivalent proof problem
theorem intersection_point_exists : is_intersection_point (6, -2, -5) := 
  sorry

end intersection_point_exists_l74_74830


namespace proof_medians_perpendicular_l74_74445

-- Given a triangle ABC with vertices A, B, and C
variables {A B C : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
(h : medians_are_perpendicular A B C)

-- We are required to prove that cot A + cot B >= 2 / 3
def cot_inequality (A B : ℝ) : Prop :=
  let cot := λ (angle : ℝ), Math.atan (1 / angle) in
  cot A + cot B >= 2 / 3

theorem proof_medians_perpendicular
  (h : medians_are_perpendicular A B C) :
  cot_inequality (angle_at A B C) (angle_at B C A) :=
   sorry

end proof_medians_perpendicular_l74_74445


namespace sampling_is_stratified_l74_74834

-- Given Conditions
def number_of_male_students := 500
def number_of_female_students := 400
def sampled_male_students := 25
def sampled_female_students := 20

-- Definition of stratified sampling according to the problem context
def is_stratified_sampling (N_M F_M R_M R_F : ℕ) : Prop :=
  (R_M > 0 ∧ R_F > 0 ∧ R_M < N_M ∧ R_F < N_M ∧ N_M > 0 ∧ N_M > 0)

-- Proving that the sampling method is stratified sampling
theorem sampling_is_stratified : 
  is_stratified_sampling number_of_male_students number_of_female_students sampled_male_students sampled_female_students = true :=
by
  sorry

end sampling_is_stratified_l74_74834


namespace solve_quadratic_equation_l74_74063

theorem solve_quadratic_equation (x : ℝ) : 2 * (x + 1) ^ 2 - 49 = 1 ↔ (x = 4 ∨ x = -6) := 
sorry

end solve_quadratic_equation_l74_74063


namespace count_true_propositions_l74_74773

theorem count_true_propositions :
  let prop1 := false ∨ (∃ (T1 T2 : Triangle), (area T1 = area T2 → T1 ≠ T2))
  let prop2 := ∀x y : ℝ, (|x| + |y| = 0 → xy = 0)
  let prop3 := ∀a b c : ℝ, (a ≤ b → a + c ≤ b + c)
  let prop4 := false
  prop1 + prop2 + prop3 + prop4 = 2 :=
by
  -- This is the extracted problem statement
  sorry

end count_true_propositions_l74_74773


namespace third_number_in_decomposition_of_5_power_4_l74_74607

theorem third_number_in_decomposition_of_5_power_4 :
  let decomposition_5_4 : List ℕ := [121, 123, 125, 127, 129] in
  decomposition_5_4.nth 2 = some 125 :=
begin
  sorry
end

end third_number_in_decomposition_of_5_power_4_l74_74607


namespace cost_of_pie_crust_l74_74663

-- Given conditions
def servings := 8
def cost_per_serving := 1
def pounds_of_apples := 2
def cost_per_pound_of_apples := 2
def cost_of_lemon := 0.5
def cost_of_butter := 1.5

-- Theorem to prove
theorem cost_of_pie_crust : 
  let total_cost_of_pie := servings * cost_per_serving in
  let total_cost_of_apples := pounds_of_apples * cost_per_pound_of_apples in
  let total_cost_of_ingredients_excluding_crust := total_cost_of_apples + cost_of_lemon + cost_of_butter in
  let pie_crust_cost := total_cost_of_pie - total_cost_of_ingredients_excluding_crust in
  pie_crust_cost = 2 :=
by
  sorry

end cost_of_pie_crust_l74_74663


namespace infinite_primes_exist_l74_74605

theorem infinite_primes_exist (k : ℕ) (h : k > 0) :
  ∃ᶠ p in filter (λ p : ℕ, nat.prime p) at_top, 
  ∃ w : ℤ, (¬ (p ∣ (w^2 - 1))) ∧ 
           (nat.order_of (w % p) (coe (nat.gcd (w % p, p))) = nat.order_of (w % (p^k)) (coe (nat.gcd (w % (p^k), p^k)))) :=
sorry

end infinite_primes_exist_l74_74605


namespace higher_probability_of_white_piece_l74_74515

theorem higher_probability_of_white_piece (white_pieces black_pieces : ℕ) (h1 : white_pieces = 10) (h2 : black_pieces = 2) :
  (white_pieces.to_rat / (white_pieces + black_pieces).to_rat) > (black_pieces.to_rat / (white_pieces + black_pieces).to_rat) :=
by
  rw [h1, h2]
  -- Here, the proof is skipped. We just state the theorem.
  sorry

end higher_probability_of_white_piece_l74_74515


namespace greatest_natural_a_l74_74930

def f (n : ℕ) : ℝ :=
  (∑ k in finset.range (2 * n + 1), 1 / (n + k + 1 : ℝ))

theorem greatest_natural_a :
  ∃ a : ℕ, (∀ n : ℕ, f n > 2 * a - 5) ∧ a = 3 :=
by
  use 3
  intros n
  sorry

end greatest_natural_a_l74_74930


namespace corey_lowest_score_l74_74903

theorem corey_lowest_score
  (e1 e2 e3 e4 : ℕ)
  (h1 : e1 = 84)
  (h2 : e2 = 67)
  (max_score : ∀ (e : ℕ), e ≤ 100)
  (avg_at_least_75 : (e1 + e2 + e3 + e4) / 4 ≥ 75) :
  e3 ≥ 49 ∨ e4 ≥ 49 :=
by
  sorry

end corey_lowest_score_l74_74903


namespace six_points_three_dist_one_l74_74393

theorem six_points_three_dist_one :
  ∃ (P : ℕ → ℝ × ℝ), (∀ i, i < 6 → P i = (x, y) → P j = (x', y') → 
    if (i ≠ j) then 
       (dist (x, y) (x', y') = 1) ∧ 
       ((∃ k₁ k₂ k₃, 
         k₁ ≠ i ∧ k₂ ≠ i ∧ k₃ ≠ i ∧ 
         dist (x, y) (P k₁) = 1 ∧ 
         dist (x, y) (P k₂) = 1 ∧ 
         dist (x, y) (P k₃) = 1 ∧
         (∀ l ≠ k₁ ∧ l ≠ k₂ ∧ l ≠ k₃, dist (x, y) (P l) ≠ 1))) else true :=
by sorry

end six_points_three_dist_one_l74_74393


namespace range_of_f_l74_74285

def g (x : ℝ) : ℝ := x^2 - 2

def f (x : ℝ) : ℝ :=
if x < g x then g x + x + 4 else g x - x

theorem range_of_f : Set.Ioo (-2.25 : ℝ) 0 ∪ Set.Ioi 2 = 
{y : ℝ | ∃ x : ℝ, f x = y} :=
sorry

end range_of_f_l74_74285


namespace find_perpendicular_vector_l74_74661

def vector_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def vector_magnitude_equal (v1 v2 : ℝ × ℝ) : Prop :=
  (v1.1 ^ 2 + v1.2 ^ 2) = (v2.1 ^ 2 + v2.2 ^ 2)

theorem find_perpendicular_vector (a b : ℝ) :
  ∃ n : ℝ × ℝ, vector_perpendicular (a, b) n ∧ vector_magnitude_equal (a, b) n ∧ n = (b, -a) :=
by
  sorry

end find_perpendicular_vector_l74_74661


namespace point_inside_circle_l74_74979

theorem point_inside_circle (a b : ℝ) (h : ∀ x y : ℝ, ax + by + 1 = 0 → x^2 + y^2 ≠ 1) :
  a^2 + b^2 < 1 := 
  sorry

end point_inside_circle_l74_74979


namespace min_value_of_expression_l74_74027

noncomputable def min_val_expr (x y : ℝ) : ℝ :=
  (8 / (x + 1)) + (1 / y)

theorem min_value_of_expression
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hcond : 2 * x + y = 1) :
  min_val_expr x y = (25 / 3) :=
sorry

end min_value_of_expression_l74_74027


namespace inequality_solution_l74_74478

theorem inequality_solution (x : ℝ) : x ^ 2 < |x| + 2 ↔ -2 < x ∧ x < 2 :=
by
  sorry

end inequality_solution_l74_74478


namespace sum_of_all_roots_l74_74897

def roots_sum (f : ℚ → ℚ) : ℚ :=
  (some (classical.some_spec (polynomial.roots_X_add_C f (by compute))))

noncomputable def sum_of_roots : ℚ :=
  (-4 / 3) + 6

theorem sum_of_all_roots :
  roots_sum (λ x, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)) = sum_of_roots := by
  sorry

end sum_of_all_roots_l74_74897


namespace log_inequalities_l74_74212

noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log_inequalities : 
  let a := log4 1 
  let b := log2 3 
  let c := log2 π 
  in c > b ∧ b > a :=
by
  sorry

end log_inequalities_l74_74212


namespace true_propositions_proof_l74_74277

theorem true_propositions_proof :
  (1 : ℕ) + (if ∀ k : ℝ, k > 0 → (∃ x : ℝ, x^2 + 2 * x - k = 0) then 1 else 0)
  + (if ∀ a b c : ℝ, a > b → (a + c > b + c) then 1 else 0)
  + (if ¬∀ quadrilateral : Type, (∀ r1 r2, quadrilateral r1 r2 → (r1 = r2)) → (∀ (r1 r2 : quadrilateral), (r1 ≠ r2) → quadrilateral r1 r2)
         then 1 else 0)
  + (if ∀ x y : ℝ, x * y = 0 → (x = 0) ∨ (y = 0) then 1 else 0) = 3 := sorry

end true_propositions_proof_l74_74277


namespace find_m_of_polynomial_has_two_distinct_positive_integer_roots_l74_74509

theorem find_m_of_polynomial_has_two_distinct_positive_integer_roots (m : ℤ) :
  (∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ (x^2 + m * x - m + 1 = 0) →
   m = -5 :=
begin
  sorry
end

end find_m_of_polynomial_has_two_distinct_positive_integer_roots_l74_74509


namespace distance_PQ_parallel_x_max_distance_PQ_l74_74884

open Real

def parabola (x : ℝ) : ℝ := x^2

/--
1. When PQ is parallel to the x-axis, find the distance from point O to PQ.
-/
theorem distance_PQ_parallel_x (m : ℝ) (h₁ : m ≠ 0) (h₂ : parabola m = 1) : 
  ∃ d : ℝ, d = 1 := by
  sorry

/--
2. Find the maximum value of the distance from point O to PQ.
-/
theorem max_distance_PQ (a b : ℝ) (h₁ : a * b = -1) (h₂ : ∀ x, ∃ y, y = a * x + b) :
  ∃ d : ℝ, d = 1 := by
  sorry

end distance_PQ_parallel_x_max_distance_PQ_l74_74884


namespace johns_mistake_l74_74369

theorem johns_mistake (a b : ℕ) (h1 : 10000 * a + b = 11 * a * b)
  (h2 : 100 ≤ a ∧ a ≤ 999) (h3 : 1000 ≤ b ∧ b ≤ 9999) : a + b = 1093 :=
sorry

end johns_mistake_l74_74369


namespace students_in_section_B_l74_74095

theorem students_in_section_B :
  let number_of_students_A := 50
  let avg_weight_A := 50
  let avg_weight_B := 70
  let avg_weight_class := 61.67
  let total_weight_A := number_of_students_A * avg_weight_A
  ∃ (b : ℕ), total_weight_A + b * avg_weight_B = 
               (number_of_students_A + b) * avg_weight_class ∧ b ≈ 70 := 
by
  sorry

end students_in_section_B_l74_74095


namespace hot_drinks_prediction_at_2_deg_l74_74539

-- Definition of the regression equation as a function
def regression_equation (x : ℝ) : ℝ :=
  -2.35 * x + 147.77

-- The statement to be proved
theorem hot_drinks_prediction_at_2_deg :
  abs (regression_equation 2 - 143) < 1 :=
sorry

end hot_drinks_prediction_at_2_deg_l74_74539


namespace find_functions_l74_74222

noncomputable def epsilon (r : ℕ) : ℤ := if (r % 2 = 1) then 1 else -1

theorem find_functions (f : ℕ → ℤ)
  (h1 : ∀ k > 0, |f k| ≤ k)
  (p : ℕ) (hp : Prime p) (hp_big : p > 2024)
  (h2 : ∀ a : ℕ, a * f (a + p) = a * f a + p * f a)
  (h3 : ∀ a : ℕ, p ∣ (a ^ ((p + 1) / 2) - f a)) :
  ∀ k r : ℕ, f (k * p + r) = epsilon r * (k * p + r) :=
by
  sorry

end find_functions_l74_74222


namespace angle_BGD_l74_74349

noncomputable def is_isosceles_triangle (C D E : Type) [PlaneTriangle C D E] : Prop :=
  PlaneTriangle.angles C D E = [70, 70, 40]

noncomputable def is_square (A B C D : Type) [PlaneSquare A B C D] : Prop :=
  PlaneSquare.angles A B C D = [90, 90, 90, 90]

theorem angle_BGD (A B C D E F G : Type) [PlaneSquare A B C D] [PlaneSquare D E F G] : 
  is_isosceles_triangle C D E → is_square A B C D → is_square D E F G → angle B G D = 180 :=
by
  intro h1 h2 h3
  sorry

end angle_BGD_l74_74349


namespace problem_eq_solution_l74_74008

variables (a b x y : ℝ)

theorem problem_eq_solution
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : a + b + x + y < 2)
  (h6 : a + b^2 = x + y^2)
  (h7 : a^2 + b = x^2 + y) :
  a = x ∧ b = y :=
by
  sorry

end problem_eq_solution_l74_74008


namespace sum_of_sines_greater_than_two_l74_74416

theorem sum_of_sines_greater_than_two (α β γ : ℝ) (h1 : α + β + γ = π) 
  (h2 : α < π / 2) (h3 : β < π / 2) (h4 : γ < π / 2) : 
  sin α + sin β + sin γ > 2 := 
sorry

end sum_of_sines_greater_than_two_l74_74416


namespace condition_general_formula_sum_bn_l74_74257

-- Definition of the sequence and the condition given in the problem.
def seq (n : ℕ) : ℝ := n - 1 / 2

def Sn (n : ℕ) : ℝ := (Finset.range n).sum seq

-- Condition 2Sn = (an + 1/2)^2
theorem condition (n : ℕ) : 2 * Sn n = (seq n + 1 / 2) ^ 2 := sorry

-- Prove the general formula for the terms of the sequence {a_n}
theorem general_formula (n : ℕ) : seq n = n - 1 / 2 := sorry

-- Given b_n and its sum T_n
def bn (n : ℕ) : ℝ := let an := seq n; let an1 := seq (n + 1) in (an + an1) / (an ^ 2 * an1 ^ 2)

def Tn (n : ℕ) : ℝ := (Finset.range n).sum bn

-- Prove the sum of the first n terms for the sequence {b_n} is Tn = (16n^2 + 16n) / (4n^2 + 4n + 1)
theorem sum_bn (n : ℕ) : Tn n = (16 * n^2 + 16 * n) / (4 * n^2 + 4 * n + 1) := sorry

end condition_general_formula_sum_bn_l74_74257


namespace subset_y_exists_l74_74024

theorem subset_y_exists (X : Type) (n : ℕ) (A : Fin 100 → Set X) 
  (hX : Fintype.card X = n) (hn : n ≥ 4) 
  (hA : ∀ i, Fintype.card (A i) > 3 * n / 4) : 
  ∃ Y : Set X, Fintype.card Y ≤ 4 ∧ ∀ i, (A i ∩ Y).Nonempty :=
begin
  sorry
end

end subset_y_exists_l74_74024


namespace toys_gained_l74_74529

theorem toys_gained
  (sp : ℕ) -- selling price of 18 toys
  (cp_per_toy : ℕ) -- cost price per toy
  (sp_val : sp = 27300) -- given selling price value
  (cp_per_val : cp_per_toy = 1300) -- given cost price per toy value
  : (sp - 18 * cp_per_toy) / cp_per_toy = 3 := by
  -- Conditions of the problem are stated
  -- Proof is omitted with 'sorry'
  sorry

end toys_gained_l74_74529


namespace round_subsets_parity_l74_74863

def is_round (n : ℕ) (S : set ℕ) : Prop :=
  S.nonempty ∧ (S.sum id % S.card = 0)

theorem round_subsets_parity (n : ℕ) :
  (finset.filter (is_round n) (finset.powerset (finset.range (n + 1)))).card % 2 = n % 2 := 
sorry

end round_subsets_parity_l74_74863


namespace tan_theta_eq_neg_sqrt_3_l74_74298

theorem tan_theta_eq_neg_sqrt_3 (theta : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h_a : a = (Real.cos theta, Real.sin theta))
  (h_b : b = (Real.sqrt 3, 1))
  (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.tan theta = -Real.sqrt 3 :=
sorry

end tan_theta_eq_neg_sqrt_3_l74_74298


namespace total_distance_hiked_l74_74740

theorem total_distance_hiked
  (a b c d e : ℕ)
  (h1 : a + b + c = 34)
  (h2 : b + c = 24)
  (h3 : c + d + e = 40)
  (h4 : a + c + e = 38)
  (h5 : d = 14) :
  a + b + c + d + e = 48 :=
by
  sorry

end total_distance_hiked_l74_74740


namespace cricketer_average_after_19_innings_l74_74493

theorem cricketer_average_after_19_innings
  (A : ℝ) 
  (total_runs_after_18 : ℝ := 18 * A) 
  (runs_in_19th : ℝ := 99) 
  (new_avg : ℝ := A + 4) 
  (total_runs_after_19 : ℝ := total_runs_after_18 + runs_in_19th) 
  (equation : 19 * new_avg = total_runs_after_19) : 
  new_avg = 27 :=
by
  sorry

end cricketer_average_after_19_innings_l74_74493


namespace find_n_l74_74858

def is_power_of_two (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 ^ k

def is_balearic_set (S : set ℕ) : Prop :=
  ∃ s s' ∈ S, is_power_of_two (s + s')

def non_balearic_set (S : set ℕ) : Prop :=
  ¬ is_balearic_set S

theorem find_n : ∃ n, (∀ S ⊆ finset.range n, 
  finset.card S = 100 → is_balearic_set S)
  ∧ (∃ T ⊆ finset.range n, finset.card T = 99 ∧ non_balearic_set T) 
  ∧ n = 198 := 
sorry

end find_n_l74_74858


namespace total_revenue_l74_74170

-- Definitions based on the conditions
def ticket_price : ℕ := 25
def first_show_tickets : ℕ := 200
def second_show_tickets : ℕ := 3 * first_show_tickets

-- Statement to prove the problem
theorem total_revenue : (first_show_tickets * ticket_price + second_show_tickets * ticket_price) = 20000 :=
by
  sorry

end total_revenue_l74_74170


namespace standard_equation_of_Γ_find_lambda_l74_74258

-- Definitions for the ellipse problem
def ellipse (a b x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def eccentricity (a c : ℝ) := c / a
def foc_dist_to_point (a : ℝ) := a - 1
def point_on_ellipse (x y a b : ℝ) := ellipse a b x y

-- Given conditions as Lean 4 definitions
def Γ := λ (x y : ℝ), ellipse 2 1 x y (x^2 / 2) + y^2 = 1
def point_P := λ (x y : ℝ), Γ x y → point_on_ellipse x y 2 1
def vec_eq (p f1 fa : (ℝ × ℝ)) := (p.1 - f1.1, p.2 - f1.2) = 2 * (f1.1 - fa.1, f1.2 - fa.2)
def vec_eq_lambda (p f2 fb : (ℝ × ℝ)) (λ : ℝ) := (p.1 - f2.1, p.2 - f2.2) = λ * (f2.1 - fb.1, f2.2 - fb.2)

-- Final theorem statements
theorem standard_equation_of_Γ : ∀ (x y : ℝ), ellipse 2 1 x y ↔ (x^2 / 2) + y^2 = 1 := by
  sorry

theorem find_lambda : ∀ (P F1 F2 A B : (ℝ × ℝ)) (λ > 0), 
  (Γ P.1 P.2 ∧ point_P P.1 P.2 ∧ vec_eq P F1 A ∧ vec_eq_lambda P F2 B λ) → λ = 4 := by
  sorry

end standard_equation_of_Γ_find_lambda_l74_74258


namespace ratio_of_areas_l74_74064

-- Define the geometrical condition
def squares_with_condition (s : ℝ) : Prop :=
  ∃ (W X Y Z I J K L : ℝ × ℝ),
    W = (0, 0) ∧
    X = (4 * s, 0) ∧
    Z = (0, 4 * s) ∧
    Y = (4 * s, 4 * s) ∧
    I = (3 * s, 0) ∧
    ∃ (I : ℝ × ℝ),
      I.1 = 3 * s ∧
      ∃ (J K L : ℝ × ℝ),
        -- Assuming J, K, L are defined symmetrically with I
        let side_length_ijkl := s * real.sqrt(2) in
        (I.1 - J.1)^2 + (I.2 - J.2)^2 = side_length_ijkl^2 ∧
        (I.1 - L.1)^2 + (I.2 - L.2)^2 = side_length_ijkl^2 ∧
        -- Using symmetry: distance from I to J (similar vertex placement as described in solution)
        -- Complete the square definition similarly
        
        -- conditions to maintain square properties and orientation

-- Define the Lean theorem
theorem ratio_of_areas
  (s : ℝ)
  (h : squares_with_condition s) :
  let area_WXYZ := (4 * s)^2 in
  let area_IJKL := (s * real.sqrt(2))^2 in
  (area_IJKL / area_WXYZ) = 1 / 8 :=
by
  sorry

end ratio_of_areas_l74_74064


namespace trig_identity_cos_sum_l74_74894

theorem trig_identity_cos_sum :
  cos (π / 3) * cos (π / 6) - sin (π / 3) * sin (π / 6) = 0 := 
by 
  sorry

end trig_identity_cos_sum_l74_74894


namespace vertex_of_quadratic_function_l74_74764

theorem vertex_of_quadratic_function :
  ∀ (x : ℝ), (2 * (x - 3)^2 + 1) = y → exists h k : ℝ, (h, k) = (3, 1) := 
by
  intros x h k
  sorry

end vertex_of_quadratic_function_l74_74764


namespace smallest_students_l74_74734

def is_divisor_count (n d : ℕ) : Prop := (nat.divisors n).length = d

theorem smallest_students : 
  ∃ (n : ℕ), (n % 10 = 0) ∧ is_divisor_count n 9 ∧ n = 36 :=
by
  sorry

end smallest_students_l74_74734


namespace min_queries_bob_needs_l74_74890

-- Define the problem conditions
def pairwise_relatively_prime (xs : List ℕ) : Prop :=
  ∀ (i j : ℕ), i < xs.length → j < xs.length → i ≠ j → Nat.gcd (xs.nthLe i sorry) (xs.nthLe j sorry) = 1

def min_queries_to_identify (n : ℕ) : ℕ :=
  int.to_nat (Nat.log2 n) + 1

-- Problem statement
theorem min_queries_bob_needs :
  ∀ (xs : List ℕ), xs.length = 2021 → pairwise_relatively_prime xs → min_queries_to_identify 2021 = 11 :=
  sorry

end min_queries_bob_needs_l74_74890


namespace find_number_l74_74139

theorem find_number (x : ℝ) (h : (1/3) * x = 12) : x = 36 :=
sorry

end find_number_l74_74139


namespace bottle_caps_total_l74_74551

theorem bottle_caps_total (groups : ℕ) (bottle_caps_per_group : ℕ) (h1 : groups = 7) (h2 : bottle_caps_per_group = 5) : (groups * bottle_caps_per_group = 35) :=
by
  sorry

end bottle_caps_total_l74_74551


namespace intersection_of_KP_with_sides_l74_74131

open Classical

-- Define the points as pairs of real numbers
structure Point where
  x : ℝ
  y : ℝ

-- Define the reflection of a point H across a line (represented by two points A and B)
def reflection (H A B : Point) : Point :=
  let dx := B.x - A.x
  let dy := B.y - A.y
  let a := (dx * dx - dy * dy) / (dx * dx + dy * dy)
  let b := 2 * dx * dy / (dx * dx + dy * dy)
  { x := a * (H.x - A.x) + b * (H.y - A.y) + A.x
  , y := b * (H.x - A.x) - a * (H.y - A.y) + A.y }

-- Given points A, B, and C of triangle, and H as the foot of the altitude from C to AB
variables (A B C H : Point)
-- Conditions: H is the foot of altitude from C to AB
-- K and P are reflections across sides AC and BC, respectively
variables (K : Point := reflection H A C) 
variables (P : Point := reflection H B C)

-- Define the line through two points
def line (P Q : Point) : ℝ × ℝ × ℝ :=
  let a := Q.y - P.y
  let b := P.x - Q.x
  let c := a * P.x + b * P.y
  (a, b, -c)

-- Define the intersection between two lines
def intersection (l1 l2 : ℝ × ℝ × ℝ) : Option Point :=
  let ⟨a1, b1, c1⟩ := l1
  let ⟨a2, b2, c2⟩ := l2
  let det := a1 * b2 - a2 * b1
  if det = 0 then
    none
  else
    some { x := (b2 * -c1 - b1 * -c2) / det
         , y := (a1 * -c2 - a2 * -c1) / det }

-- The orthocenter is the intersection of altitudes, we compute it
def orthocenter (A B C : Point) : Point :=
  let altitude1 := line B (reflection B A C)
  let altitude2 := line C (reflection C B A)
  (intersection altitude1 altitude2).getD {x := 0, y := 0} -- assume some default point if none

-- We need to prove that the intersection of KP with sides AC and BC is the orthocenter.
theorem intersection_of_KP_with_sides : 
  let KP := line K P
  let AC := line A C
  let BC := line B C
  intersection KP AC = some (orthocenter A B C) ∧ 
  intersection KP BC = some (orthocenter A B C) := 
by 
  sorry

end intersection_of_KP_with_sides_l74_74131


namespace smallest_percent_increase_l74_74073

-- Define the values of each question
def question_values : List ℕ :=
  [150, 250, 400, 600, 1100, 2300, 4700, 9500, 19000, 38000, 76000, 150000, 300000, 600000, 1200000]

-- Define a function to calculate the percent increase between two questions
def percent_increase (v1 v2 : ℕ) : Float :=
  ((v2 - v1).toFloat / v1.toFloat) * 100

-- Define the specific question transitions and their percent increases
def percent_increase_1_to_4 : Float := percent_increase question_values[0] question_values[3]  -- Question 1 to 4
def percent_increase_2_to_6 : Float := percent_increase question_values[1] question_values[5]  -- Question 2 to 6
def percent_increase_5_to_10 : Float := percent_increase question_values[4] question_values[9]  -- Question 5 to 10
def percent_increase_9_to_15 : Float := percent_increase question_values[8] question_values[14] -- Question 9 to 15

-- Prove that the smallest percent increase is from Question 1 to 4
theorem smallest_percent_increase :
  percent_increase_1_to_4 < percent_increase_2_to_6 ∧
  percent_increase_1_to_4 < percent_increase_5_to_10 ∧
  percent_increase_1_to_4 < percent_increase_9_to_15 :=
by
  sorry

end smallest_percent_increase_l74_74073


namespace pool_perimeter_l74_74549

theorem pool_perimeter (garden_length : ℝ) (plot_area : ℝ) (plot_count : ℕ) : 
  garden_length = 9 ∧ plot_area = 20 ∧ plot_count = 4 →
  ∃ (pool_perimeter : ℝ), pool_perimeter = 18 :=
by
  intros h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end pool_perimeter_l74_74549


namespace count_coprime_fractions_eq_112_l74_74382

noncomputable def count_coprime_fractions := 
  let possible_pairs := (λ (r : ℚ), ∃ (p q : ℕ), r = p / q ∧ 0 < r ∧ r < 1 ∧ (Nat.coprime p q) ∧ (p * q).1.dvd 3600)
  let valid_fractions := {r : ℚ | possible_pairs r}
  (finset.card valid_fractions)

theorem count_coprime_fractions_eq_112 : count_coprime_fractions = 112 :=
  sorry

end count_coprime_fractions_eq_112_l74_74382


namespace f_decreasing_l74_74441

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

def decreasing_interval : set ℝ := (0, 2)

theorem f_decreasing : ∀ x : ℝ, x ∈ decreasing_interval → (deriv f x) < 0 :=
by
  intros x hx
  have h_deriv : deriv f x = 3 * x ^ 2 - 6 * x :=
    by sorry

  -- Prove that this derivative is less than 0 on the interval (0, 2)
  have h_decreasing : 3 * x ^ 2 - 6 * x < 0 :=
    by sorry

  exact h_decreasing

end f_decreasing_l74_74441


namespace fgh_deriv_at_0_l74_74805

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def h : ℝ → ℝ := sorry

-- Function Values at x = 0
axiom f_zero : f 0 = 1
axiom g_zero : g 0 = 2
axiom h_zero : h 0 = 3

-- Derivatives of the pairwise products at x = 0
axiom d_gh_zero : (deriv (λ x => g x * h x)) 0 = 4
axiom d_hf_zero : (deriv (λ x => h x * f x)) 0 = 5
axiom d_fg_zero : (deriv (λ x => f x * g x)) 0 = 6

-- We need to prove that the derivative of the product of f, g, h at x = 0 is 16
theorem fgh_deriv_at_0 : (deriv (λ x => f x * g x * h x)) 0 = 16 := by
  sorry

end fgh_deriv_at_0_l74_74805


namespace cyclist_speed_25_l74_74384

def speeds_system_eqns (x : ℝ) (y : ℝ) : Prop :=
  (20 / x - 20 / 50 = y) ∧ (70 - (8 / 3) * x = 50 * (7 / 15 - y))

theorem cyclist_speed_25 :
  ∃ y : ℝ, speeds_system_eqns 25 y :=
by
  sorry

end cyclist_speed_25_l74_74384


namespace problem1_problem2_l74_74134

-- First problem
theorem problem1 (n : ℕ) (h : binomial n 2 = binomial n 5) : n = 7 :=
sorry

-- Second problem
theorem problem2 (C : ℕ → ℕ → ℕ) (sum_odd_coeffs_eq_128 : Σ (k : ℕ), ∑ i in finset.range (k + 1), binomial (2 * i + 1) i = 128) :
  let n := 8 in (C n 4 * (x * sqrt x) ^ 4 * (-(2 / sqrt x)) ^ 4 = 1120 * x ^ 4) :=
sorry

end problem1_problem2_l74_74134


namespace triangle_inequality_squares_l74_74386

theorem triangle_inequality_squares (a b c : ℝ) (h₁ : a < b + c) (h₂ : b < a + c) (h₃ : c < a + b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + a * c) :=
sorry

end triangle_inequality_squares_l74_74386


namespace sale_overlap_count_in_august_l74_74841

/-- 
A bookstore has a sale on days of the month that are multiples of 4.
A shoe store has a sale every 7 days starting from August 1.
Prove that the number of overlapping sales days in August is 2.
--/
theorem sale_overlap_count_in_august :
  let bookstore_days := {d : ℕ | 1 ≤ d ∧ d ≤ 31 ∧ d % 4 = 0}
  let shoe_store_days := {d : ℕ | 1 ≤ d ∧ d ≤ 31 ∧ (d - 1) % 7 = 0}
  (bookstore_days ∩ shoe_store_days).card = 2 := 
by
  sorry

end sale_overlap_count_in_august_l74_74841


namespace first_player_winning_strategy_l74_74784

theorem first_player_winning_strategy :
  ∃ strategy : (fin 2002 → set ℕ) → (fin 2002 → set ℕ),
    (∀ state : (fin 2002 → set ℕ), 
      ∃ move : ℕ, move ∈ state ∧ ∀ n, n ∣ move → (n = move ∨ n = 1) →
      (set.remove state move).nonempty) :=
begin
  -- Sorry to skip the actual proof
  sorry
end

end first_player_winning_strategy_l74_74784


namespace number_of_buildings_l74_74882

theorem number_of_buildings (studio_apartments : ℕ) (two_person_apartments : ℕ) (four_person_apartments : ℕ)
    (occupancy_percentage : ℝ) (current_occupancy : ℕ)
    (max_occupancy_building : ℕ) (max_occupancy_complex : ℕ) (num_buildings : ℕ)
    (h_studio : studio_apartments = 10)
    (h_two_person : two_person_apartments = 20)
    (h_four_person : four_person_apartments = 5)
    (h_occupancy_percentage : occupancy_percentage = 0.75)
    (h_current_occupancy : current_occupancy = 210)
    (h_max_occupancy_building : max_occupancy_building = 10 * 1 + 20 * 2 + 5 * 4)
    (h_max_occupancy_complex : max_occupancy_complex = current_occupancy / occupancy_percentage)
    (h_num_buildings : num_buildings = max_occupancy_complex / max_occupancy_building) :
    num_buildings = 4 :=
by
  sorry

end number_of_buildings_l74_74882


namespace max_div_result_is_10_l74_74807

noncomputable def max_div_result (a b : ℕ) : ℚ :=
  (10 * a + b) / (a + b)

theorem max_div_result_is_10 :
  ∃ a b : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ max_div_result a b = 10 :=
by
  use [1, 0]
  split
  repeat {
    split
    norm_num
  }
  sorry -- The detailed proof would go here, verifying that the maximum value is indeed 10 for appropriate a and b.

end max_div_result_is_10_l74_74807


namespace calculate_fraction_l74_74053

theorem calculate_fraction :
  (-1 / 42) / (1 / 6 - 3 / 14 + 2 / 3 - 2 / 7) = -1 / 14 :=
by
  sorry

end calculate_fraction_l74_74053


namespace find_N_l74_74906

theorem find_N : {N : ℕ // N > 0 ∧ ∃ k : ℕ, 2^N - 2 * N = k^2} = {1, 2} := 
    sorry

end find_N_l74_74906


namespace segment_EC_length_l74_74634

noncomputable def length_of_segment_EC (a b c : ℕ) (angle_A_deg BC : ℝ) (BD_perp_AC CE_perp_AB : Prop) (angle_DBC_eq_3_angle_ECB : Prop) : ℝ :=
  a * (Real.sqrt b + Real.sqrt c)

theorem segment_EC_length
  (a b c : ℕ)
  (angle_A_deg BC : ℝ)
  (BD_perp_AC CE_perp_AB : Prop)
  (angle_DBC_eq_3_angle_ECB : Prop)
  (h1 : angle_A_deg = 45)
  (h2 : BC = 10)
  (h3 : BD_perp_AC)
  (h4 : CE_perp_AB)
  (h5 : angle_DBC_eq_3_angle_ECB)
  (h6 : length_of_segment_EC a b c angle_A_deg BC BD_perp_AC CE_perp_AB angle_DBC_eq_3_angle_ECB = 5 * (Real.sqrt 3 + Real.sqrt 1)) :
  a + b + c = 9 :=
  by
    sorry

end segment_EC_length_l74_74634


namespace water_outflow_time_l74_74536

theorem water_outflow_time (H R : ℝ) (flow_rate : ℝ → ℝ)
  (h_initial : ℝ) (t_initial : ℝ) (empty_height : ℝ) :
  H = 12 →
  R = 3 →
  (∀ h, flow_rate h = -h) →
  h_initial = 12 →
  t_initial = 0 →
  empty_height = 0 →
  ∃ t, t = (72 : ℝ) * π / 16 :=
by
  intros hL R_eq flow_rate_eq h_initial_eq t_initial_eq empty_height_eq
  sorry

end water_outflow_time_l74_74536


namespace correct_proposition_is_D_l74_74545

-- Definitions of the propositions based on given conditions
def PropositionA : Prop :=
  ∀ (k : ℝ) (x y : ℕ), confidence (x, y, k^2) = someFunction k

def PropositionB : Prop :=
  ∀ (r : ℝ), strongerCorrelation (r) → abs r = 0

def PropositionC : Prop :=
  ∀ (x : ℕ → ℝ), variance (λ n, 2 * x n) = 2 * variance x

def PropositionD : Prop :=
  ∀ (R2 : ℝ), fittingEffect (R2) → betterFittingEffect R2

-- Main statement to be proven
theorem correct_proposition_is_D : PropositionD := sorry

end correct_proposition_is_D_l74_74545


namespace T_sum_geometric_l74_74603

noncomputable def T (r : ℝ) : ℝ := 15 / (1 - r)

theorem T_sum_geometric (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T(b) * T(-b) = 2250) : T(b) + T(-b) = 300 := 
by
  sorry

end T_sum_geometric_l74_74603


namespace sum_of_sequence_l74_74643

theorem sum_of_sequence (n : ℕ) (h : n ≥ 2) 
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, n ≥ 2 → S n * S (n-1) + a n = 0) :
  S n = 2 / (2 * n - 1) := by
  sorry

end sum_of_sequence_l74_74643


namespace f_at_3_l74_74616

theorem f_at_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 2) = x + 3) : f 3 = 4 := 
sorry

end f_at_3_l74_74616


namespace zoo_animal_difference_l74_74871

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  monkeys - zebras = 35 := by
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  show monkeys - zebras = 35
  sorry

end zoo_animal_difference_l74_74871


namespace mass_percentage_Ca_in_CaCO3_l74_74597

def molar_mass_Ca : ℝ := 40.08
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def molar_mass_CaCO3 : ℝ := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O

theorem mass_percentage_Ca_in_CaCO3 :
  (molar_mass_Ca / molar_mass_CaCO3) * 100 ≈ 40.04 :=
by
  sorry

end mass_percentage_Ca_in_CaCO3_l74_74597


namespace trigonometric_identity_l74_74613

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  4 * Real.sin α * Real.cos α + 3 * (Real.cos α) ^ 2 = 11 / 5 :=
sorry

end trigonometric_identity_l74_74613


namespace correct_system_of_equations_l74_74091

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : 3 * x = 5 * y - 6)
  (h2 : y = 2 * x - 10) : 
  (3 * x = 5 * y - 6) ∧ (y = 2 * x - 10) :=
by
  sorry

end correct_system_of_equations_l74_74091


namespace initial_students_count_l74_74071

theorem initial_students_count (n W : ℝ)
    (h1 : W = n * 28)
    (h2 : W + 10 = (n + 1) * 27.4) :
    n = 29 :=
by
  sorry

end initial_students_count_l74_74071


namespace find_ratio_of_sides_l74_74703

variables {A B C : ℝ} {a b c : ℝ}

-- Define triangle sides conditions
def triangle_sides_opposite :=
  ∃ a b c, (a * sin A) - (b * sin B) = 4 * c * sin C

-- Define cosine condition
def cos_condition :=
  cos A = - (1 / 4)

-- The theorem proving that b / c = 6
theorem find_ratio_of_sides
  (h1 : triangle_sides_opposite)
  (h2 : cos_condition) :
  b / c = 6 :=
sorry

end find_ratio_of_sides_l74_74703


namespace finite_swaps_books_width_sorted_l74_74718

variable (n : ℕ) (h : n ≥ 2)
variable (books : Fin n → ℕ) -- the height function
variable (widths : Fin n → ℕ) -- the width function

-- Conditions
variable (h_incr : ∀ i j : Fin n, i < j → books i < books j)
variable (unique_height : Function.Injective books)
variable (unique_width : Function.Injective widths)

-- Moves definition
def can_swap (i : Fin (n-1)) : Prop :=
  (widths i > widths (i + 1) ∧ books i < books (i + 1))

-- Finite number of moves
theorem finite_swaps : ∃ k : ℕ, ∀ i : ℕ, i ≥ k → ¬∃ i : Fin (n-1), can_swap n books widths i :=
sorry

-- Books sorted by width
theorem books_width_sorted : ∀ i j : Fin n, i < j → widths i < widths j :=
sorry

end finite_swaps_books_width_sorted_l74_74718


namespace domain_of_f_l74_74770

-- Define the function
def f (x : ℝ) := real.sqrt (x^2 - 1) + real.log x / real.log 2

-- State the conditions and objective as a theorem
theorem domain_of_f : {x : ℝ | x > 1} = {x : ℝ | x^2 - 1 ≥ 0} ∩ {x : ℝ | x - 1 > 0} :=
by {
  sorry
}

end domain_of_f_l74_74770


namespace least_cans_required_l74_74154

def maaza : ℕ := 20
def pepsi : ℕ := 144
def sprite : ℕ := 368

def GCD (a b : ℕ) : ℕ := Nat.gcd a b

def total_cans (maaza pepsi sprite : ℕ) : ℕ :=
  let gcd_maaza_pepsi := GCD maaza pepsi
  let gcd_all := GCD gcd_maaza_pepsi sprite
  (maaza / gcd_all) + (pepsi / gcd_all) + (sprite / gcd_all)

theorem least_cans_required : total_cans maaza pepsi sprite = 133 := by
  sorry

end least_cans_required_l74_74154


namespace intersection_points_for_7_lines_l74_74699

-- Definitions used in the conditions problem
def lines : ℕ := 7

-- Mathematical property that combination of n taken r at a time
def combinations (n r : ℕ) : ℕ :=
  Nat.choose n r

-- Property of maximum possible intersections of 7 lines
def max_intersections : ℕ :=
  combinations 7 2

-- Valid points of intersection for 7 lines
def valid_points_of_intersection : Finset ℕ :=
  ((Finset.range (max_intersections + 1)).erase 2).erase 3. erase 4. erase 5

theorem intersection_points_for_7_lines :
  (8 ∈ valid_points_of_intersection) ∧ 
  (0 ∈ valid_points_of_intersection) ∧ 
  (1 ∈ valid_points_of_intersection) ∧ 
  (∀ n ∈ valid_points_of_intersection, n ∈ {0, 1} ∨ (6 ≤ n ∧ n ≤ 21)) :=
by
  sorry

end intersection_points_for_7_lines_l74_74699


namespace trip_time_l74_74851

theorem trip_time :
  let distance_Carville_Nikpath := 315
  let distance_Nikpath_Finstown := 70
  let speed := 60
  let break_time := 0.5
  let time_Carville_Nikpath := distance_Carville_Nikpath / speed
  let time_Nikpath_Finstown := distance_Nikpath_Finstown / speed
  let total_time := time_Carville_Nikpath + break_time + time_Nikpath_Finstown
  float.round (total_time * 100) / 100 = 6.92 :=
by
  sorry

end trip_time_l74_74851


namespace distance_from_vertex_to_plane_l74_74167

theorem distance_from_vertex_to_plane
  (a b c d : ℝ³) (dim_x dim_y dim_z : ℝ)
  (h_prism : a = (0, 0, 0) ∧ b = (dim_x, 0, 0) ∧ c = (dim_x, dim_y, 0) ∧ d = (dim_x, dim_y, dim_z))
  (h_dim : dim_x = 4 ∧ dim_y = 4 ∧ dim_z = 3) :
  distance_point_plane d (plane_span a b c) = 2.1 :=
begin
  sorry
end

end distance_from_vertex_to_plane_l74_74167


namespace problem_interval_tan_cot_sum_l74_74816

theorem problem_interval_tan_cot_sum (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) : 
  let y := tan (3 * x) * (cot x)^3 in
  (∀ y, y ∉ set.Ioo (2:ℝ) (65:ℝ)) → 2 + 65 = 34 :=
by
  sorry

end problem_interval_tan_cot_sum_l74_74816


namespace quadratic_trinomial_unique_l74_74229

theorem quadratic_trinomial_unique
  (a b c : ℝ)
  (h1 : b^2 - 4 * (a + 1) * c = 0)
  (h2 : (b + 1)^2 - 4 * a * c = 0)
  (h3 : b^2 - 4 * a * (c + 1) = 0) :
  a = 1 / 8 ∧ b = -3 / 4 ∧ c = 1 / 8 :=
begin
  -- statement for the theorem, proof not required
  sorry
end

end quadratic_trinomial_unique_l74_74229


namespace quarterly_to_annual_interest_rate_l74_74399

theorem quarterly_to_annual_interest_rate :
  ∃ s : ℝ, (1 + 0.02)^4 = 1 + s / 100 ∧ abs (s - 8.24) < 0.01 :=
by
  sorry

end quarterly_to_annual_interest_rate_l74_74399


namespace B_subset_A_l74_74972

/-- Define sets A and B -/
def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5}

/-- Prove that B is a subset of A -/
theorem B_subset_A : B ⊆ A :=
sorry

end B_subset_A_l74_74972


namespace find_integer_x_l74_74232

theorem find_integer_x (x y : ℕ) (h_gt : x > y) (h_gt_zero : y > 0) (h_eq : x + y + x * y = 99) : x = 49 :=
sorry

end find_integer_x_l74_74232


namespace two_pow_n_minus_one_divisible_by_seven_l74_74223

theorem two_pow_n_minus_one_divisible_by_seven (n : ℕ) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, n = 3 * k := 
sorry

end two_pow_n_minus_one_divisible_by_seven_l74_74223


namespace minimize_weighted_sum_distances_l74_74220

variables {A B C X : Type} [MetricSpace X]
variables {a b c : X}
variables {m n p : ℝ} (h_m : m > 0) (h_n : n > 0) (h_p : p > 0)

noncomputable def distance_minimizing_point (m n p : ℝ) (m_pos : m > 0) (n_pos : n > 0) (p_pos : p > 0) 
  (a b c : X) : X :=
if h_m_ge_np : m ≥ n + p then a
else if h_n_ge_mp : n ≥ m + p then b
else if h_p_ge_mn : p ≥ m + n then c
else sorry  -- Placeholder for the geometric minimization case

theorem minimize_weighted_sum_distances (m n p : ℝ) (a b c : X)
  (h_m : m > 0) (h_n : n > 0) (h_p : p > 0) :
  ∃ x : X, x = distance_minimizing_point m n p h_m h_n h_p a b c :=
begin
  use distance_minimizing_point m n p h_m h_n h_p a b c,
  split_ifs,
  { refl },
  { refl },
  { refl },
  { sorry }  -- Placeholder for the proof of the geometric minimization case
end

end minimize_weighted_sum_distances_l74_74220


namespace largest_three_digit_solution_l74_74833

theorem largest_three_digit_solution :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 45 * n % 315 = 270 % 315 ∧ n = 993 :=
begin
  sorry
end

end largest_three_digit_solution_l74_74833


namespace naomi_total_wheels_l74_74888

theorem naomi_total_wheels 
  (regular_bikes : ℕ) (children_bikes : ℕ) (tandem_bikes_4_wheels : ℕ) (tandem_bikes_6_wheels : ℕ)
  (wheels_per_regular_bike : ℕ) (wheels_per_children_bike : ℕ) (wheels_per_tandem_4wheel : ℕ) (wheels_per_tandem_6wheel : ℕ) :
  regular_bikes = 7 →
  children_bikes = 11 →
  tandem_bikes_4_wheels = 5 →
  tandem_bikes_6_wheels = 3 →
  wheels_per_regular_bike = 2 →
  wheels_per_children_bike = 4 →
  wheels_per_tandem_4wheel = 4 →
  wheels_per_tandem_6wheel = 6 →
  (regular_bikes * wheels_per_regular_bike) + 
  (children_bikes * wheels_per_children_bike) + 
  (tandem_bikes_4_wheels * wheels_per_tandem_4wheel) + 
  (tandem_bikes_6_wheels * wheels_per_tandem_6wheel) = 96 := 
by
  intros; sorry

end naomi_total_wheels_l74_74888


namespace find_def_79_exists_post_int_l74_74464

theorem find_def_79_exists_post_int
  (d e f : ℕ)
  (h1 : 4 * real.sqrt (real.cbrt 7 - real.cbrt 3) = real.cbrt d + real.cbrt e - real.cbrt f)
  (h2 : d > 0)
  (h3 : e > 0)
  (h4 : f > 0) :
  d + e + f = 79 :=
sorry

end find_def_79_exists_post_int_l74_74464


namespace quadratic_roots_b_c_quadratic_root_b_l74_74256

theorem quadratic_roots_b_c (b c: ℝ)
  (h1: ∀ x, (x + 1) * (x - 1) = 0 → x^2 + 2 * b * x + c = 0)
  : b = 0 ∧ c = -1 := by
  sorry

theorem quadratic_root_b (b: ℝ)
  (h2: c = b^2 + 2 * b + 3)
  (h3: ∀ x1 x2, (x1 + 1) * (x2 + 1) = 8 → f x = 0 ∧ (x1 + x2 = -2 * b ∧ x1 * x2 = c))
  : b = -2 := by
  sorry

end quadratic_roots_b_c_quadratic_root_b_l74_74256


namespace compute_f1_l74_74104

theorem compute_f1 :
  (∀ x, f(x) + g(x) = 2) →
  (∀ x, f(f(x)) = g(g(x))) →
  (f(0) = 2022) →
  f(1) = 2022 - 505 / 1011 :=
  by
  intros h1 h2 h3
  -- problem solving steps go here
  sorry

end compute_f1_l74_74104


namespace units_digit_4_pow_10_l74_74113

theorem units_digit_4_pow_10 : Nat.unitsDigit (4 ^ 10) = 6 := 
by
  /-
    Conditions: 
    1. \forall n, Nat.unitsDigit (4 ^ (2 * n + 1)) = 4
    2. \forall n, Nat.unitsDigit (4 ^ (2 * n)) = 6
  -/
  sorry

end units_digit_4_pow_10_l74_74113


namespace min_max_f_l74_74712

noncomputable def f (n : ℕ) (r : Fin m.succ → ℚ) : ℕ :=
  n - (Finset.univ.sum (λ k => ⌊r k * n⌋))

variables {m : ℕ}
variables (r : Fin m.succ → ℚ)
variables (h : ∑ i, r i = 1)

theorem min_max_f (n : ℕ) (hn : 0 < n) :
  0 ≤ f n r ∧ f n r < m :=
by
  unfold f
  sorry

end min_max_f_l74_74712


namespace complement_intersection_l74_74724

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end complement_intersection_l74_74724


namespace remaining_cookies_l74_74727

variable (total_initial_cookies : ℕ)
variable (cookies_taken_day1 : ℕ := 3)
variable (cookies_taken_day2 : ℕ := 3)
variable (cookies_eaten_day2 : ℕ := 1)
variable (cookies_put_back_day2 : ℕ := 2)
variable (cookies_taken_by_junior : ℕ := 7)

theorem remaining_cookies (total_initial_cookies cookies_taken_day1 cookies_taken_day2
                          cookies_eaten_day2 cookies_put_back_day2 cookies_taken_by_junior : ℕ) :
  (total_initial_cookies = 2 * (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior))
  → (total_initial_cookies - (cookies_taken_day1 + cookies_eaten_day2 + cookies_taken_by_junior) = 11) :=
by
  sorry

end remaining_cookies_l74_74727


namespace pappus_theorem_l74_74473

open ProjectiveGeometry

-- Assume the existence of points A, B, C, A1, B1, C1
variable (A B C A1 B1 C1 : Point)

-- Conditions: 
-- 1. Assume four collinear points: A, B, B1, C1
axiom collinear_ABB1C1 : Collinear A B B1 C1

-- 2. Another set of four collinear points: C, A, A1, B1
axiom collinear_CAA1B1 : Collinear C A A1 B1

-- 3. Define the intersections of lines created from these points
-- Intersection points of the given lines
noncomputable def P : Point := intersection (line_through A B1) (line_through B A1)
noncomputable def Q : Point := intersection (line_through B C1) (line_through C B1)
noncomputable def R : Point := intersection (line_through C A1) (line_through A C1)
noncomputable def R1 : Point := intersection (line_through P Q) (line_through C A1)

-- Theorem: Prove that R = R1 (Pappus's theorem)
theorem pappus_theorem :
  R = R1 := 
sorry

end pappus_theorem_l74_74473


namespace number_of_orders_to_break_targets_l74_74687

theorem number_of_orders_to_break_targets : 
  let targets := multiset.of_list ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
  in multiset.countp ((=) 'A') targets = 3 ∧
     multiset.countp ((=) 'B') targets = 3 ∧
     multiset.countp ((=) 'C') targets = 3 →
     multiset.permutations targets.length = 1680 :=
by
  sorry

end number_of_orders_to_break_targets_l74_74687


namespace add_to_get_l74_74114

theorem add_to_get (z : ℂ) (h1 : 5 - 3 * complex.I + z = -4 + 9 * complex.I) : z = -9 + 12 * complex.I :=
by {
  sorry
}

end add_to_get_l74_74114


namespace find_side_length_and_radius_l74_74268

/-
Given:
1. Circles x and y have the same area.
2. Circle x has a circumference of 20π.
3. Circle y is inscribed in a square and intersects with the two diagonal lines from the vertices of the square.

Prove:
1. The side length of the square is 20 units.
2. Half of the radius of circle y is 5 units.
-/

section

variables {x y : Type} [metric_space x] [metric_space y]
variable (r : ℝ) -- radius
variable (s : ℝ) -- side length of the square
variable (hxr : x.radius = r)
variable (hyr : y.radius = r)
variable (hxC : 2 * real.pi * r = 20 * real.pi) -- circumference condition
variable (inscribe : ∃ s, 2 * r = s) -- circle y is inscribed in a square

theorem find_side_length_and_radius (h_area : ∀ (a b : Type) [metric_space a] [metric_space b], a.area = b.area) :
  s = 20 ∧ r / 2 = 5 :=
by
  -- the actual proof would go here
  sorry

end

end find_side_length_and_radius_l74_74268


namespace fractional_cake_eaten_l74_74490

def total_cake_eaten : ℚ :=
  1 / 3 + 1 / 3 + 1 / 6 + 1 / 12 + 1 / 24 + 1 / 48

theorem fractional_cake_eaten :
  total_cake_eaten = 47 / 48 := by
  sorry

end fractional_cake_eaten_l74_74490


namespace fn_digital_sum_l74_74711

theorem fn_digital_sum (n : ℕ) (h1 : 0 < n) :
  let f_n := λ (x : ℕ), x * (10^n - 1) / 9 in
  (f_n 3)^2 + f_n 2 = f_n 1 * (10^n + 1) :=
by
  -- Definitions and variables
  let f_n := λ (x : ℕ), x * (10^n - 1) / 9;
  -- Solution omitted
  sorry

end fn_digital_sum_l74_74711


namespace exponential_order_l74_74387

theorem exponential_order (x y : ℝ) (a : ℝ) (hx : x > y) (hy : y > 1) (ha1 : 0 < a) (ha2 : a < 1) : a^x < a^y :=
sorry

end exponential_order_l74_74387


namespace difference_proof_l74_74364

-- Define the dimensions of the large rectangle
def A : ℕ := 20
def B : ℕ := 30

-- Define the dimensions of the smaller rectangle
def a : ℕ := 4
def b : ℕ := 7

-- Calculate the area of the large rectangle
def area_large : ℕ := A * B

-- Calculate the area of the smaller rectangle
def area_small : ℕ := a * b

-- Calculate the difference between the total area of the yellow and green quadrilaterals
def difference : ℕ := abs (0 - (area_large - area_small))

-- Prove that the difference is 572
theorem difference_proof : difference = 572 := by
  sorry

end difference_proof_l74_74364


namespace spent_on_books_l74_74052

theorem spent_on_books {X : ℝ} :
  let books_first_shop := 55 in
  let books_second_shop := 60 in
  let cost_second_shop := 340 in
  let avg_price := 16 in
  books_first_shop + books_second_shop = 115 →
  X + cost_second_shop = avg_price * (books_first_shop + books_second_shop) →
  X = 1500 :=
begin
  intros h1 h2,
  sorry
end

end spent_on_books_l74_74052


namespace find_real_a_l74_74649

open Complex

theorem find_real_a (a : ℝ) (h1 : 1 + a * I = (1 + a * I) ^ 3) (h2 : (2: ℂ) ^ (3/4) * (cos (3 * π / 8) + I * sin (3 * π / 8)) ^ 2) :
  a = 1 := by
    sorry

end find_real_a_l74_74649


namespace no_real_m_for_equal_roots_l74_74213

theorem no_real_m_for_equal_roots :
  ∀ m : ℝ, ¬ (∀ x : ℝ, (3 * x^2 * (x - 2) - (2 * m + 3)) / ((x - 2) * (m - 2)) = (2 * x^2 / m) → x * x = 0) :=
begin
  sorry
end

end no_real_m_for_equal_roots_l74_74213


namespace hexagon_inequality_l74_74182

variable (hexagon : Finset (Fin 6)) (edges : hexagon → hexagon → Bool)

-- Conditions
def is_red (a b : hexagon) : Bool := edges a b
def red_count (a : hexagon) : ℕ := (hexagon.filter (is_red a)).card

-- The proof problem
theorem hexagon_inequality (H : ∀ j k m : hexagon, j ≠ k → k ≠ m → j ≠ m → 
                             is_red j k ∨ is_red k m ∨ is_red j m) :
  ∑ k in hexagon, (2 * red_count k - 7) ^ 2 ≤ 54 :=
sorry

end hexagon_inequality_l74_74182


namespace money_r_gets_l74_74422

def total_amount : ℕ := 1210
def p_to_q := 5 / 4
def q_to_r := 9 / 10

theorem money_r_gets :
  let P := (total_amount * 45) / 121
  let Q := (total_amount * 36) / 121
  let R := (total_amount * 40) / 121
  R = 400 := by
  sorry

end money_r_gets_l74_74422


namespace Brenda_new_lead_l74_74569

noncomputable def Brenda_initial_lead : ℤ := 22
noncomputable def Brenda_play_points : ℤ := 15
noncomputable def David_play_points : ℤ := 32

theorem Brenda_new_lead : 
  Brenda_initial_lead + Brenda_play_points - David_play_points = 5 := 
by
  sorry

end Brenda_new_lead_l74_74569


namespace problem_inequality_l74_74263

theorem problem_inequality (a b x y : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by
  sorry

end problem_inequality_l74_74263


namespace correct_observation_value_l74_74444

theorem correct_observation_value (n : ℕ) (mean_initial mean_corrected wrong_value correct_value : ℝ)
    (h1 : n = 40) 
    (h2 : mean_initial = 36)
    (h3 : mean_corrected = 36.45)
    (h4 : wrong_value = 20) 
    (h_sum_initial : n * mean_initial = 1440)
    (h_sum_corrected : n * mean_corrected = 1458)
    (h_correct_value : correct_value = (wrong_value + (h_sum_corrected - (h_sum_initial - wrong_value)))) :
    correct_value = 58 := 
sorry

end correct_observation_value_l74_74444


namespace derivative_of_reciprocal_l74_74435

noncomputable def y (x : ℝ) : ℝ := 1 / x

theorem derivative_of_reciprocal (x : ℝ) (hx : x ≠ 0) : deriv y x = -x⁻² :=
by sorry

end derivative_of_reciprocal_l74_74435


namespace identify_strongest_fighter_l74_74512

def fighter_strengths := Fin 52 → ℕ
def stronger_defeats_weaker (strengths : fighter_strengths) (i j : Fin 52) : Prop :=
  (strengths i > strengths j) ∨ (strengths j = 0 ∧ strengths i = 51)

theorem identify_strongest_fighter (strengths : fighter_strengths) :
  ∃ (S : Fin 52), 
  ∀ (matches : List (Fin 52 × Fin 52)), 
  (∀ (i j : Fin 52), (i, j) ∈ matches → stronger_defeats_weaker strengths i j) ∧
  length matches ≤ 64 →
  (∃ s, ∀ t, (s, t) ∈ matches ∨ (t, s) ∈ matches → strengths s >= strengths t) :=
begin
  sorry
end

end identify_strongest_fighter_l74_74512


namespace clerts_for_45_degrees_l74_74044

theorem clerts_for_45_degrees (fullCircleMars : ℝ) (fullCircleEarth : ℝ) (angleEarth : ℝ) :
  fullCircleMars = 400 ∧ fullCircleEarth = 360 ∧ angleEarth = 45 → 
  (angleEarth / fullCircleEarth) * fullCircleMars = 50 := 
by 
  intros h,
  cases h with h_mars h_earth,
  cases h_earth with h_earth_full h_angle,
  sorry

end clerts_for_45_degrees_l74_74044


namespace sum_of_squares_divisible_by_24_l74_74744

theorem sum_of_squares_divisible_by_24 (p : Fin 24 → ℕ) (prime_p : ∀ i, Nat.Prime (p i)) (ge_5 : ∀ i, p i ≥ 5) :
  (Finset.univ.sum (λ i, (p i)^2)) % 24 = 0 := sorry

end sum_of_squares_divisible_by_24_l74_74744


namespace highest_power_of_3_l74_74080

def concatenated_integer (start: ℕ) (stop: ℕ) : ℕ :=
    let nums := List.range' start (stop - start + 1)
    let digits := nums.bind (λ x => toDigits 10 x)
    fromDigits 10 digits

def sum_of_digits (n : ℕ) : ℕ :=
    (toDigits 10 n).sum

theorem highest_power_of_3 (k : ℕ) :
    let N := concatenated_integer 19 92
    k = 1 ↔ ∃ k, 3^k ∣ N ∧ ∀ m, 3^(m + 1) ∤ N :=
by 
  intro k
  let N := concatenated_integer 19 92
  sorry

end highest_power_of_3_l74_74080


namespace neg_exists_gt_one_eq_forall_le_one_l74_74447

theorem neg_exists_gt_one_eq_forall_le_one (P : Prop) :
  (¬ (∃ x : ℝ, x > 1)) ↔ ∀ x : ℝ, x ≤ 1 :=
begin
  sorry
end

end neg_exists_gt_one_eq_forall_le_one_l74_74447


namespace speed_in_still_water_is_32_l74_74118

-- Define the upstream and downstream speeds as given conditions
def upstream_speed : ℝ := 22
def downstream_speed : ℝ := 42

-- Define the speed in still water
def speed_in_still_water : ℝ := (upstream_speed + downstream_speed) / 2

-- State that the speed in still water is 32 kmph
theorem speed_in_still_water_is_32 : speed_in_still_water = 32 := by
  sorry

end speed_in_still_water_is_32_l74_74118


namespace inequalities_satisfied_l74_74481

variables (a b c x y z : ℝ)

open Real

theorem inequalities_satisfied (h1 : |x| < |a|) (h2 : |y| < |b|) (h3 : |z| < |c|) : 
  (|x * y| + |y * z| + |z * x| < |a * b| + |b * c| + |c * a|) ∧
  (x^2 + z^2 < a^2 + c^2) :=
begin
  sorry
end

end inequalities_satisfied_l74_74481


namespace cos_alpha_mul_sin_beta_l74_74677

variables (α β : ℝ)
variables (x y m : ℝ)

def sin (θ : ℝ) := sin θ
def cos (θ : ℝ) := cos θ

-- Conditions
def condition_1 := y = sqrt 3 * x
def condition_2 := (1 / 2, m) ∈ { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 1 }
def condition_3 := sin α * cos β < 0

theorem cos_alpha_mul_sin_beta :
  x = -1 / 2 → y = -sqrt 3 / 2 →
  condition_1 →
  condition_2 →
  condition_3 →
  cos α = -1 / 2 →
  (m = sqrt 3 / 2 ∨ m = -sqrt 3 / 2) →
  (cos α * sin β = sqrt 3 / 4 ∨ cos α * sin β = -sqrt 3 / 4) :=
sorry

end cos_alpha_mul_sin_beta_l74_74677


namespace sqrt_of_x_plus_six_l74_74973

theorem sqrt_of_x_plus_six (x : ℝ) (h : 2 * x = real.cbrt 216) : real.sqrt (x + 6) = 3 ∨ real.sqrt (x + 6) = -3 :=
by
  -- Placeholder for the proof
  sorry

end sqrt_of_x_plus_six_l74_74973


namespace rebate_percentage_l74_74425

theorem rebate_percentage (initial_worth final_amount : ℕ) (sales_tax_rate : ℝ) (rebate_percentage : ℝ) : 
    initial_worth = 6650 → 
    final_amount = 6876.1 → 
    sales_tax_rate = 0.10 → 
    rebate_percentage = 6 :=
by
  intros h1 h2 h3
  sorry

end rebate_percentage_l74_74425


namespace perfect_square_form_l74_74913

theorem perfect_square_form (N : ℕ) (hN : 0 < N) : 
  ∃ x : ℤ, 2^N - 2 * (N : ℤ) = x^2 ↔ N = 1 ∨ N = 2 :=
by
  sorry

end perfect_square_form_l74_74913


namespace math_problem_l74_74202

def f(x : ℝ) : ℝ := (x + 2) * (x - 4)
def g(x : ℝ) : ℝ := -f(x)
def h(x : ℝ) : ℝ := f(2 * x)
def a : ℕ := 2
def b : ℕ := 2

theorem math_problem : 10 * a + b = 22 := 
by 
  rw [a, b]
  norm_num
-- sorry

end math_problem_l74_74202


namespace smallest_six_digit_number_divisible_by_3_7_13_l74_74937

theorem smallest_six_digit_number_divisible_by_3_7_13 : 
  ∃ (n : ℕ), n % 3 = 0 ∧ n % 7 = 0 ∧ n % 13 = 0 ∧ 100000 ≤ n ∧ n ≤ 999999 ∧ 
  ∀ m, (m % 3 = 0 ∧ m % 7 = 0 ∧ m % 13 = 0 ∧ 100000 ≤ m ∧ m ≤ 999999) → n ≤ m :=
begin
  use 100191,
  split, norm_num, -- divisibility by 3
  split, norm_num, -- divisibility by 7
  split, norm_num, -- divisibility by 13
  split, norm_num, -- lower bound
  split, norm_num, -- upper bound
  intro m,
  simp [nat.le_def, nat.lt],
  intros h1 h2 h3 h4 h5,
  sorry
end

end smallest_six_digit_number_divisible_by_3_7_13_l74_74937


namespace sum_of_functions_positive_l74_74655

open Real

noncomputable def f (x : ℝ) : ℝ := (exp x - exp (-x)) / 2

theorem sum_of_functions_positive (x1 x2 x3 : ℝ) (h1 : x1 + x2 > 0) (h2 : x2 + x3 > 0) (h3 : x3 + x1 > 0) : f x1 + f x2 + f x3 > 0 := by
  sorry

end sum_of_functions_positive_l74_74655


namespace triangle_division_odd_l74_74109

theorem triangle_division_odd (n : ℕ) :
  ∃ k : ℕ, k = 2 * n + 1 :=
begin
  -- proof goes here
  sorry
end

end triangle_division_odd_l74_74109


namespace Jeremy_payment_total_l74_74705

theorem Jeremy_payment_total :
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  total_payment = (553 : ℚ) / 40 :=
by {
  -- Definitions
  let room_rate := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let window_rate := (5 : ℚ) / 2
  let windows_cleaned := (11 : ℚ) / 4
  let payment_rooms := room_rate * rooms_cleaned
  let payment_windows := window_rate * windows_cleaned
  let total_payment := payment_rooms + payment_windows
  
  -- Main goal
  sorry
}

end Jeremy_payment_total_l74_74705


namespace sum_of_y_coordinates_of_parallelogram_l74_74271

theorem sum_of_y_coordinates_of_parallelogram (x1 x2 y1 y2 : ℝ)
  (hx : (x1, y1) = (2, 15 ∨ (x1, y1) = (8, -6))
  (hy : (x2, y2) = (8, -6) ∨ (x2, y2) = (2, 15) : 
  (y1 + y2 / 2 = 4.5)→(y1∧ y2 = 9)):(x2-x1)>0 ∨(x2-x1) = 9 :=
by
  sorry

end sum_of_y_coordinates_of_parallelogram_l74_74271


namespace certain_event_l74_74116

theorem certain_event :
  (∀ (a b : ℝ), a * b = b * a) :=
by 
  assume a b : ℝ,
  exact mul_comm a b

end certain_event_l74_74116


namespace magician_balls_possible_2005_l74_74336

theorem magician_balls_possible_2005 :
  ∃ n : ℕ, 5 + 4 * n = 2005 :=
begin
  sorry
end

end magician_balls_possible_2005_l74_74336


namespace knights_divisible_by_4_l74_74920

-- Define the conditions: Assume n is the total number of knights (n > 0).
-- Condition 1: Knights from two opposing clans A and B
-- Condition 2: Number of knights with an enemy to the right equals number of knights with a friend to the right.

open Nat

theorem knights_divisible_by_4 (n : ℕ) (h1 : 0 < n)
  (h2 : ∃k : ℕ, 2 * k = n ∧ ∀ (i : ℕ), (i < n → ((i % 2 = 0 → (i+1) % 2 = 1) ∧ (i % 2 = 1 → (i+1) % 2 = 0)))) :
  n % 4 = 0 :=
sorry

end knights_divisible_by_4_l74_74920


namespace sum_of_solutions_x_squared_eq_36_sum_of_solutions_x_squared_eq_36_sum_l74_74318

theorem sum_of_solutions_x_squared_eq_36 (x : ℝ) (hx : x^2 = 36) : x = 6 ∨ x = -6 :=
by
  have soln1 : 6^2 = 36 := by norm_num
  have soln2 : (-6)^2 = 36 := by norm_num
  cases eq_or_eq_neg_eq.mpr ⟨by exact hx, by exact soln1, by exact soln2⟩

theorem sum_of_solutions_x_squared_eq_36_sum (x : ℝ) : (x = 6 ∨ x = -6) → (∑ x in {6, -6}, x = 0) :=
by 
  sorry

end sum_of_solutions_x_squared_eq_36_sum_of_solutions_x_squared_eq_36_sum_l74_74318


namespace problem_part_1_problem_part_2_problem_part_3_l74_74252

noncomputable def circle (x y : ℝ) (b : ℝ) : Prop := (x + 2)^2 + (y - b)^2 = 3
noncomputable def line (x y m : ℝ) : Prop := y = x + m
noncomputable def is_tangent_line (x y b m : ℝ) : Prop :=
  let center_dist := |0 - 1|
  center_dist = center_dist

theorem problem_part_1 (b : ℝ) (h_b : b > 0)
  (h_1 : (circle (-2 + real.sqrt 2) 0 b)) : 
  b = 1 :=
sorry

theorem problem_part_2 (m b : ℝ) (h_t : is_tangent_line b (3 + real.sqrt 6) b m) :
  m = 3 + real.sqrt 6 ∨ m = 3 - real.sqrt 6 :=
sorry

theorem problem_part_3 (m : ℝ) :
  ∃ x1 x2 y1 y2 : ℝ, (line x1 y1 m) ∧ (line x2 y2 m) ∧ x1*x2 + y1*y2 = 0 →
  m = 1 ∨ m = 2 :=
sorry

end problem_part_1_problem_part_2_problem_part_3_l74_74252


namespace lioness_hyena_age_ratio_l74_74068

variables {k H : ℕ}

-- Conditions
def lioness_age (lioness_age hyena_age : ℕ) : Prop := ∃ k, lioness_age = k * hyena_age
def lioness_is_12 (lioness_age : ℕ) : Prop := lioness_age = 12
def baby_age (mother_age baby_age : ℕ) : Prop := baby_age = mother_age / 2
def baby_ages_sum_in_5_years (baby_l_age baby_h_age sum : ℕ) : Prop := 
  (baby_l_age + 5) + (baby_h_age + 5) = sum

-- The statement to be proved
theorem lioness_hyena_age_ratio (H : ℕ)
  (h1 : lioness_age 12 H) 
  (h2 : baby_age 12 6) 
  (h3 : baby_age H (H / 2)) 
  (h4 : baby_ages_sum_in_5_years 6 (H / 2) 19) : 12 / H = 2 := 
sorry

end lioness_hyena_age_ratio_l74_74068


namespace log_x_64_eq_3_imp_x_eq_4_l74_74927

theorem log_x_64_eq_3_imp_x_eq_4 (x : ℝ) (h : Real.log x 64 = 3) : x = 4 :=
by
  sorry

end log_x_64_eq_3_imp_x_eq_4_l74_74927


namespace farmer_land_area_l74_74402

-- Variables representing the total land, and the percentages and areas.
variable {T : ℝ} (h_cleared : 0.85 * T =  V) (V_10_percent : 0.10 * V + 0.70 * V + 0.05 * V + 500 = V)
variable {total_acres : ℝ} (correct_total_acres : total_acres = 3921.57)

theorem farmer_land_area (h_cleared : 0.85 * T = V) (h_planted : 0.85 * V = 500) : T = 3921.57 :=
by
  sorry

end farmer_land_area_l74_74402


namespace total_length_of_ribbon_l74_74749

-- Define the conditions
def length_per_piece : ℕ := 73
def number_of_pieces : ℕ := 51

-- The theorem to prove
theorem total_length_of_ribbon : length_per_piece * number_of_pieces = 3723 :=
by
  sorry

end total_length_of_ribbon_l74_74749


namespace function_symmetry_l74_74186

theorem function_symmetry : 
  ∀ x : ℝ, ∃ y : ℝ, y = tan (x + π / 6) →
  ∃ y' : ℝ, y' = tan ((π / 3 - x) + π / 6) ∧ -y = y' :=
by
  sorry

end function_symmetry_l74_74186


namespace quadratic_solution_result_l74_74717

theorem quadratic_solution_result :
  (∃ d e : ℝ, 4*d^2 + 8*d - 48 = 0 ∧ 4*e^2 + 8*e - 48 = 0 ∧ (d - e)^2 + 4 = 68) :=
begin
  sorry
end

end quadratic_solution_result_l74_74717


namespace a_le_5_of_monotone_on_l74_74280

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 3 * x ^ 3 - a * x ^ 2 + x - 5

theorem a_le_5_of_monotone_on (a : ℝ) :
  (∀ x ∈ set.Icc (1 : ℝ) (2 : ℝ), deriv (f x a) x ≥ 0) → a ≤ 5 :=
sorry

end a_le_5_of_monotone_on_l74_74280


namespace floor_abs_sum_l74_74586

def abs (x : ℝ) : ℝ := if x < 0 then -x else x
def floor (x : ℝ) : ℤ := ⌊x⌋

theorem floor_abs_sum : floor (abs (-5.7)) + abs (floor (-5.7)) = 11 :=
by
  sorry

end floor_abs_sum_l74_74586


namespace AM_HM_Inequality_l74_74420

theorem AM_HM_Inequality {n : ℕ} (hn : 1 < n) (x : Fin n → ℝ) (h_pos : ∀ i : Fin n, 0 < x i) :
    (∑ i : Fin n, (x i) / (∑ j : Fin n, if i = j then 0 else x j)) ≥ (n : ℝ) / (n - 1) :=
sorry

end AM_HM_Inequality_l74_74420


namespace problem1_problem2_l74_74554

theorem problem1 :
  log 25 + log 4 - (27 / 8)^(1/3) + 3^(log 3 2) + (sqrt 2)^0 = 5 / 2 :=
by sorry

theorem problem2 (α : ℝ):
  (cos (π / 2 - α) * cos (α + π) * tan (α - 5 * π)) /
  (cos (α - π) * sin (3 * π - α) * sin (-α - π)) = tan α :=
by sorry

end problem1_problem2_l74_74554


namespace cristina_pace_5_4_mps_l74_74040

-- Conditions
variable (race_length : ℕ) (head_start_time : ℕ) (nicky_pace : ℕ) (time_before_catch : ℕ)
variable (nicky_faster_than_cristina : Prop)
variable (nickys_pace_eq_3mps : nicky_pace = 3)
variable (headstart_is_12_sec : head_start_time = 12)
variable (running_time_is_30_sec : time_before_catch = 30)
variable (race_length_eq_500_m: race_length = 500)

-- Correct answer
theorem cristina_pace_5_4_mps :
  ∀ (pace : ℕ) (distance_covered_by_nicky := (head_start_time * nicky_pace) + (time_before_catch * nicky_pace)) 
    (distance_caught_up_by_cristina := distance_covered_by_nicky + (head_start_time * nicky_pace)),
    distance_caught_up_by_cristina / time_before_catch = 54 / 10  :=
begin
  intros,
  unfold distance_covered_by_nicky distance_caught_up_by_cristina,
  rw [nickys_pace_eq_3mps, headstart_is_12_sec, running_time_is_30_sec],
  exact (162, 30, 54, 10)
end

end cristina_pace_5_4_mps_l74_74040


namespace area_of_rectangle_l74_74127

-- Definitions for the conditions
def breadth (b : ℝ) : Prop := b > 0
def length (l : ℝ) (b : ℝ) : Prop := l = 3 * b
def perimeter (P : ℝ) (l : ℝ) (b : ℝ) : Prop := P = 2 * (l + b)

-- Main theorem to prove the area is 108 square meters given the conditions
theorem area_of_rectangle (b l P A : ℝ) 
  (hb : breadth b) 
  (hl : length l b) 
  (hP : perimeter P l b) 
  (hP_val : P = 48) 
  : A = l * b :=
by 
  rw [hl, hP] at hP_val
  have hb_val : b = 6, by linarith
  have hl_val : l = 3 * b, by exact hl
  have l_val : l = 18, by linarith
  have A_val : A = l * b, by linarith
  sorry

end area_of_rectangle_l74_74127


namespace find_general_term_and_sum_l74_74981

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def sum_first_n_terms (a₁ d : ℝ) (n : ℕ) : ℝ := 
  n * a₁ + (n * (n - 1) / 2) * d

theorem find_general_term_and_sum (S3 S5 : ℝ) :
  S3 = 0 → S5 = -5 → 
  (∃ (a₁ d : ℝ), 
    let a_n := λ n, arithmetic_sequence a₁ d n in 
    a_n 3 - a_n 2 = 0 ∧ a_n 5 - a_n 3 = -1) ∧
  (∃ (a₁ d : ℝ) (n : ℕ), 
    let a_n := λ n, arithmetic_sequence a₁ d n in 
    let b_n := λ n : ℕ, 1 / (a_n (2 * n - 1) * a_n (2 * n + 1)) in 
    ∑ i in finset.range n, b_n i = n / (1 - 2 * (n : ℝ))) :=
begin
  sorry,
end

end find_general_term_and_sum_l74_74981


namespace sequence_preservation_l74_74218

-- Define constants and sequences.
def K (b : ℕ) : ℕ := 1 * b ^ 3 + 2 * b ^ 2 + 2 * b + 1

theorem sequence_preservation (b : ℕ) :
  (b = 13 ∨ b = 12 ∨ b = 11 ∨ b = 10) →
  (∀ b, b = 10 → ¬ (K b = K (b - 1))) :=
by {
  intros h,
  specialize h 10,
  sorry
}

end sequence_preservation_l74_74218


namespace count_primes_in_range_l74_74308

open Nat

-- Define the conditions
def is_prime : ℕ → Prop
| n := n > 1 ∧ (∀d, d ∣ n → d = 1 ∨ d = n)

def prime_remainder_cond (p : ℕ) : Prop :=
  is_prime (p % 10) ∧ is_prime (p % 3)

-- Set the main theorem
theorem count_primes_in_range : 
  Nat.card {p : ℕ | 50 ≤ p ∧ p ≤ 90 ∧ is_prime p ∧ prime_remainder_cond p} = 6 :=
by
  sorry

end count_primes_in_range_l74_74308


namespace distance_between_cities_l74_74801

-- Define necessary variables
variables (v : ℝ) -- Speed of car A and initial speed of car B in km/min
variables (x : ℝ) -- Total time car A takes to travel between the two cities in minutes

-- Conditions
def car_A_speed : Prop := (v > 0)
def initial_distance_5_min : Prop := (∀ (d : ℝ), d = 5 * v)
def car_B_speed_reduced : Prop := (v_reduced : ℝ) (v_reduced = (2/5) * v)
def carB_arrival_late : Prop := (x_B : ℝ) (x_A : x_B + 15 = x)
def carB_arrival_late_after_4km : Prop := (x_B_star : ℝ) 
                                       (x - ((15 * v) - (5 * v + 4)) / v
                                            = x_B_star + 10)

-- Theorem: Prove the distance between the two cities is 18 km
theorem distance_between_cities 
  (h1 : car_A_speed v)
  (h2 : initial_distance_5_min v)
  (h3 : car_B_speed_reduced v)
  (h4 : carB_arrival_late v x)
  (h5 : carB_arrival_late_after_4km v x):
  (d : ℝ) (d = 18) :=
sorry

end distance_between_cities_l74_74801


namespace find_original_savings_l74_74031

variable (S : ℝ)

-- Conditions
def spent_on_furniture : ℝ := (3 / 4) * S
def spent_on_appliances : ℝ := (1 / 8) * S
def remaining_savings : ℝ := S - spent_on_furniture - spent_on_appliances
def tv_cost_after_discount : ℝ := 250
def discount_rate : ℝ := 0.1
def original_tv_price : ℝ := tv_cost_after_discount / (1 - discount_rate)

-- Statement
theorem find_original_savings
  (H1 : spent_on_furniture = (3 / 4) * S)
  (H2 : spent_on_appliances = (1 / 8) * S)
  (H3 : remaining_savings = S - spent_on_furniture - spent_on_appliances)
  (H4 : remaining_savings = original_tv_price)
  (H5 : tv_cost_after_discount = 250)
  (H6 : original_tv_price = tv_cost_after_discount / (1 - discount_rate)) :
  S = 2222.24 :=
by
  sorry

end find_original_savings_l74_74031


namespace solve_fractional_equation_l74_74431

theorem solve_fractional_equation (x : ℝ) (h : (x + 1) / (4 * x^2 - 1) = (3 / (2 * x + 1)) - (4 / (4 * x - 2))) : x = 6 := 
by
  sorry

end solve_fractional_equation_l74_74431


namespace N_square_solutions_l74_74909

theorem N_square_solutions :
  ∀ N : ℕ, (N > 0 → ∃ k : ℕ, 2^N - 2 * N = k^2) → (N = 1 ∨ N = 2) :=
by
  sorry

end N_square_solutions_l74_74909


namespace travel_time_to_15_feet_above_bottom_l74_74143

-- Definitions of the given conditions
def radius : ℝ := 30
def revolution_rate : ℝ := 1  -- revolutions per minute

-- Constant rate of revolution implies angular speed in radians per second
-- Since there are 2π radians in one revolution and 60 seconds in a minute
def angular_speed : ℝ := (2 * Real.pi) / 60

-- Time calculation. We need to prove this:
theorem travel_time_to_15_feet_above_bottom 
  (r : ℝ) (revolution_rate : ℝ) (angular_speed : ℝ) (target_height : ℝ) 
    (t : ℝ) :
  r = radius → revolution_rate = 1 → angular_speed = (2 * Real.pi) / 60 →
  target_height = 15 →
  target_height = r * (1 + Real.cos(angular_speed * t)) / 2 → t = 20 := by sorry

end travel_time_to_15_feet_above_bottom_l74_74143


namespace calc_value_l74_74901

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem calc_value :
  ((diamond (diamond 3 4) 2) - (diamond 3 (diamond 4 2))) = -13 / 28 :=
by sorry

end calc_value_l74_74901


namespace max_num_planks_l74_74610

noncomputable def numPlanks (diameter : ℝ) (thickness : ℝ) (width : ℝ) : ℕ :=
  let radius := diameter / 2
  let strips := ⌊diameter / thickness⌋
  let planksInStrip := ⌊2 * (√(radius ^ 2 - (thickness / 2) ^ 2)) / width⌋
  let plankCount := (strips - 5) * planksInStrip + 16  -- Based on the calculated fit
  plankCount

theorem max_num_planks : numPlanks 46 4 12 = 29 := by
  sorry

end max_num_planks_l74_74610


namespace sqrt_div_simplify_l74_74585

theorem sqrt_div_simplify (x y : ℝ) 
  (h: ( ( 1 / 3 ) ^ 2 + ( 1 / 4 ) ^ 2 ) / ( ( 1 / 5 ) ^ 2 + ( 1 / 6 ) ^ 2 ) = 25 * x / (61 * y) ) : 
  (sqrt x) / (sqrt y) = 5 / 2 :=
by
  sorry

end sqrt_div_simplify_l74_74585


namespace sides_ratio_of_triangle_A1_B1_C4_l74_74332

theorem sides_ratio_of_triangle_A1_B1_C4 
  (A B C a b c : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : A / B = 1)
  (h3 : B / C = 1/4) 
  (h4 : SineLaw a b c A B C) :
  a : b : c = 1 : 1 : Real.sqrt(3) := 
sorry

end sides_ratio_of_triangle_A1_B1_C4_l74_74332


namespace calc_limit_l74_74009

noncomputable def a_n (n : ℕ) : ℝ :=
  real.arctan (4 * n^2 / (4 * n^4 - 1))

theorem calc_limit :
  (filter.tendsto (λ p : ℕ, finset.sum (finset.range p) (λ n, a_n (n+1))) filter.at_top (nhds (real.pi / 2))) :=
begin
  sorry
end

end calc_limit_l74_74009


namespace remaining_area_l74_74819

-- Definitions based on conditions:
-- Znayka cut out a semicircle from paper with diameter AB.
-- Neznayka marked a point D on the diameter AB.
-- Neznayka cut out two semicircles with diameters AD and DB.
-- The length of the chord passing through point D perpendicular to AB inside the remaining shape is 6.

-- We need to prove:
theorem remaining_area (AB AD DB : ℝ) (h1 : AB = AD + DB) (h2 : 6^2 = AD * DB) : (π * (9 : ℝ)) = 28.27 :=
by
  have h3 : AD * DB = 36 := by exact h2
  have h4 : AD * DB = 36 := h2
  have key : 9 * π = 28.273159 := by
    sorry
  exact key

end remaining_area_l74_74819


namespace least_n_divisibility_l74_74110

theorem least_n_divisibility : ∃ n : ℕ, 
  (n > 0) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n * (n - 1) % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n * (n - 1) % k ≠ 0) ∧ 
  n = 5 :=
by
  sorry

end least_n_divisibility_l74_74110


namespace blue_notebook_cost_l74_74037

theorem blue_notebook_cost 
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (cost_per_red : ℕ)
  (green_notebooks : ℕ)
  (cost_per_green : ℕ)
  (blue_notebooks : ℕ)
  (total_cost_blue : ℕ)
  (cost_per_blue : ℕ)
  (h1 : total_spent = 37)
  (h2 : total_notebooks = 12)
  (h3 : red_notebooks = 3)
  (h4 : cost_per_red = 4)
  (h5 : green_notebooks = 2)
  (h6 : cost_per_green = 2)
  (h7 : total_cost_blue = total_spent - (red_notebooks * cost_per_red + green_notebooks * cost_per_green))
  (h8 : blue_notebooks = total_notebooks - (red_notebooks + green_notebooks))
  (h9 : cost_per_blue = total_cost_blue / blue_notebooks)
  : cost_per_blue = 3 :=
sorry

end blue_notebook_cost_l74_74037


namespace inequality_am_gm_boundary_l74_74012

theorem inequality_am_gm_boundary (a b : ℝ) (x : ℕ → ℝ) (n : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : ∀ i, x i ∈ set.Icc a b) :
  (finset.sum finset.univ (λ i, x i)) * (finset.sum finset.univ (λ i, x i⁻¹)) ≤ (a + b)^2 / (4 * a * b) * n^2 := 
sorry

end inequality_am_gm_boundary_l74_74012


namespace coeff_sum_l74_74668

noncomputable def poly : Polynomial ℝ := 
  Polynomial.X^2 + 1 * (Polynomial.X - 2)^9

theorem coeff_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} : ℝ),
  poly = a_0 + a_1 * Polynomial.X + a_2 * Polynomial.X^2 + a_3 * Polynomial.X^3 + a_4 * Polynomial.X^4 + a_5 * Polynomial.X^5 + a_6 * Polynomial.X^6 + a_7 * Polynomial.X^7 + a_8 * Polynomial.X^8 + a_9 * Polynomial.X^9 + a_{10} * Polynomial.X^{10} + a_{11} * Polynomial.X^{11} →
  let a_0 := (1:ℝ) * (-2)^9 in
  (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11}) = 510 :=
begin
  sorry
end

end coeff_sum_l74_74668


namespace possible_B_values_l74_74078

theorem possible_B_values :
  ∃ (count : Nat), count = 10 ∧
  ∀ A B : Nat, (0 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ (9 + A + B) % 3 = 0 ∧ 
  (9 - (A + B) = 0 ∨ 9 - (A + B) = 11 ∨ 9 - (A + B) = -11) →
  count = 10 :=
by
  sorry

end possible_B_values_l74_74078


namespace solve_N1N2_identity_l74_74016

theorem solve_N1N2_identity :
  (∃ N1 N2 : ℚ,
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 3 →
      (42 * x - 37) / (x^2 - 4 * x + 3) =
      N1 / (x - 1) + N2 / (x - 3)) ∧ 
      N1 * N2 = -445 / 4) :=
by
  sorry

end solve_N1N2_identity_l74_74016


namespace concert_ticket_cost_l74_74887

noncomputable def price_of_adult_ticket (total_cost: ℕ) (adult_tickets: ℕ) (child_tickets: ℕ) (child_ticket_ratio: ℚ) : ℚ :=
  total_cost / (adult_tickets + child_tickets * child_ticket_ratio)

theorem concert_ticket_cost :
  ∀ (a c a': ℚ), (c = 2/3 * a) →
  (6 * a + 5 * c = 35) →
  (9 * a + 7 * c = 51.25) :=
by
  intros a c a'_ h₁ h₂
  rw h₁ at *
  sorry

end concert_ticket_cost_l74_74887


namespace discriminant_of_quadratic_l74_74577

-- Define the quadratic equation coefficients
def a : ℝ := 5
def b : ℝ := -11
def c : ℝ := 4

-- Prove the discriminant of the quadratic equation
theorem discriminant_of_quadratic :
    b^2 - 4 * a * c = 41 :=
by
  sorry

end discriminant_of_quadratic_l74_74577


namespace employee_wage_is_correct_l74_74401

-- Define the initial conditions
def revenue_per_month : ℝ := 400000
def tax_rate : ℝ := 0.10
def marketing_rate : ℝ := 0.05
def operational_cost_rate : ℝ := 0.20
def wage_rate : ℝ := 0.15
def number_of_employees : ℕ := 10

-- Compute the intermediate values
def taxes : ℝ := tax_rate * revenue_per_month
def after_taxes : ℝ := revenue_per_month - taxes
def marketing_ads : ℝ := marketing_rate * after_taxes
def after_marketing : ℝ := after_taxes - marketing_ads
def operational_costs : ℝ := operational_cost_rate * after_marketing
def after_operational : ℝ := after_marketing - operational_costs
def total_wages : ℝ := wage_rate * after_operational

-- Compute the wage per employee
def wage_per_employee : ℝ := total_wages / number_of_employees

-- The proof problem statement ensuring the calculated wage per employee is 4104
theorem employee_wage_is_correct :
  wage_per_employee = 4104 := by 
  sorry

end employee_wage_is_correct_l74_74401


namespace becky_necklaces_count_l74_74195

-- Define the initial conditions
def initial_necklaces := 50
def broken_necklaces := 3
def new_necklaces := 5
def given_away_necklaces := 15

-- Define the final number of necklaces
def final_necklaces (initial : Nat) (broken : Nat) (bought : Nat) (given_away : Nat) : Nat :=
  initial - broken + bought - given_away

-- The theorem stating that after performing the series of operations,
-- Becky should have 37 necklaces.
theorem becky_necklaces_count :
  final_necklaces initial_necklaces broken_necklaces new_necklaces given_away_necklaces = 37 :=
  by
    -- This proof is just a placeholder to ensure the code can be built successfully.
    -- Actual proof logic needs to be filled in to complete the theorem.
    sorry

end becky_necklaces_count_l74_74195


namespace alpha_odd_iff_domain_all_real_and_odd_function_l74_74954

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem alpha_odd_iff_domain_all_real_and_odd_function (α : ℤ) :
  (∀ x : ℝ, x ^ α = x ^ α) ∧ is_odd_function (λ x : ℝ, x ^ (α : ℝ))
  ↔ ∃ k : ℤ, α = 2 * k + 1 :=
begin
  sorry
end

end alpha_odd_iff_domain_all_real_and_odd_function_l74_74954


namespace range_g_l74_74579

noncomputable def g (A : Real) : Real :=
  (Real.cos A * (2 * Real.sin A ^ 2 + Real.sin A ^ 4 + 2 * Real.cos A ^ 2 + Real.cos A ^ 2 * Real.sin A ^ 2)) /
  (Real.cot A * (Real.csc A - Real.cos A * Real.cot A))

theorem range_g (A : Real) (h : ¬ ∃ n : Int, A = n * Real.pi) : Set.Icc 2 3 = { y : Real | ∃ x : Real, g x = y } :=
by
  sorry

end range_g_l74_74579


namespace tape_recorder_cost_l74_74424

-- Define the conditions
def conditions (x p : ℚ) : Prop :=
  170 < p ∧ p < 195 ∧
  2 * p = x * (x - 2) ∧
  1 * x = x - 2 + 2

-- Define the statement to be proved
theorem tape_recorder_cost (x : ℚ) (p : ℚ) : conditions x p → p = 180 := by
  sorry

end tape_recorder_cost_l74_74424


namespace find_coordinates_D_l74_74690

/-- In quadrilateral ABCD, the coordinates of points A, B, and C are respectively (1,0), 
    (3,0), and (2,2). Given that AB is parallel to CD and AB = CD, prove that 
    the coordinates of point D are (0,2). -/
theorem find_coordinates_D :
  ∃ (D : ℝ × ℝ), D = (0, 2) ∧
  let A : ℝ × ℝ := (1, 0),
      B : ℝ × ℝ := (3, 0),
      C : ℝ × ℝ := (2, 2),
      AB := dist A B in
  AB = 2 ∧
  parallel (line_through A B) (line_through C D) ∧
  dist A B = dist C D :=
begin
  sorry
end

end find_coordinates_D_l74_74690


namespace olivia_average_speed_l74_74733

theorem olivia_average_speed (total_distance : ℕ) (total_cycling_time : ℕ) (total_distance = 50) (total_cycling_time = 8) : (total_distance / total_cycling_time = 6.25) :=
by sorry

end olivia_average_speed_l74_74733


namespace volunteer_assigned_probability_l74_74244

theorem volunteer_assigned_probability :
  let volunteers := ["A", "B", "C", "D"]
  let areas := ["Beijing", "Zhangjiakou"]
  let total_ways := 14
  let favorable_ways := 6
  ∃ (p : ℚ), p = 6/14 → (1 / total_ways) * favorable_ways = 3/7
:= sorry

end volunteer_assigned_probability_l74_74244


namespace chemical_reaction_rate_of_B_l74_74700

-- Defining the variables and parameters given in the problem
variables (vA : ℝ) (vB : ℝ)

-- Given conditions as definitions in Lean 4
def stoichiometric_relation (vA vB : ℝ) : Prop := vB = 3 * vA
def given_reaction_rate_of_A : Prop := vA = 0.2

-- The theorem statement that poses the problem
theorem chemical_reaction_rate_of_B :
  (vA = 0.2) → (stoichiometric_relation vA vB) → vB = 0.6 :=
by
  intros hva hrelation
  rw hrelation
  rw hva
  sorry

end chemical_reaction_rate_of_B_l74_74700


namespace infinite_l_and_min_l_l74_74254

theorem infinite_l_and_min_l (m : ℕ) (h1 : 0 < m) : 
  ∃ l : ℕ, 5 ^ (5^m * l) ∣ (∏ i in finset.range (4 * (5^m + 1) * l + 1), (4 * l + i)) 
  ∧ ¬ (5 ^ (5^m * l + 1) ∣ (∏ i in finset.range (4 * (5^m + 1) * l + 1), (4 * l + i)))
  ∧ l = (5 ^ (m+1) - 1) / 4 := 
sorry

end infinite_l_and_min_l_l74_74254


namespace height_of_box_l74_74838

theorem height_of_box (h : ℝ) (r_big : ℝ) (r_small : ℝ) 
  (box_dim : ℝ) 
  (cond1 : (box_dim = 4)) 
  (cond2 : (r_big = 2)) 
  (cond3 : (r_small = 1)) 
  (cond4 : (∀ s : ℝ, s ∈ ({sphere | sphere.radius = r_small} : set ℝ) → sphere_tangent_to_sides s box_dim)) 
  (cond5 : ∀ s : ℝ, s ∈ ({sphere | sphere.radius = r_small} : set ℝ) → tangent_spheres s r_big) :
  h = 2 + 2 * Real.sqrt 7 :=
by
  sorry

end height_of_box_l74_74838


namespace percentage_of_whole_equals_part_l74_74477

theorem percentage_of_whole_equals_part (whole part : ℝ) (h_whole : whole = 360) (h_part : part = 162) : 
  (part / whole) * 100 = 45 := 
by
  -- Given whole = 360 and part = 162
  rw [h_whole, h_part]
  -- Perform the division
  have h_div : 162 / 360 = 0.45 := by norm_num
  rw h_div
  -- Perform the multiplication
  norm_num

end percentage_of_whole_equals_part_l74_74477


namespace negation_exists_real_negation_of_quadratic_l74_74083

theorem negation_exists_real (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

def quadratic (x : ℝ) : Prop := x^2 - 2*x + 3 ≤ 0

theorem negation_of_quadratic :
  (¬ ∀ x : ℝ, quadratic x) ↔ ∃ x : ℝ, ¬ quadratic x :=
by exact negation_exists_real quadratic

end negation_exists_real_negation_of_quadratic_l74_74083


namespace simplest_sqrt_l74_74483

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y ∈ {sqrt 15, sqrt 18, sqrt (1 / 2), sqrt 9}, x = sqrt 15 → (sqrt 15 ≤ y)

theorem simplest_sqrt :
  is_simplest_sqrt (sqrt 15) :=
sorry

end simplest_sqrt_l74_74483


namespace correct_oblique_method_statement_l74_74054

-- Definitions of the conditions using the oblique drawing method
def is_triangle_intuitive_diagram (t : Type) [triangle t] : Prop :=
  intuitive_diagram t = t

def is_parallelogram_intuitive_diagram (p: Type) [parallelogram p] : Prop :=
  intuitive_diagram p = p

def square_to_parallelogram (s: Type) [square s] : Prop :=
  intuitive_diagram s = parallelogram

def rhombus_to_parallelogram (r: Type) [rhombus r] : Prop :=
  intuitive_diagram r = parallelogram

-- The mathematical problem rewritten as a Lean 4 statement
theorem correct_oblique_method_statement
  (t : Type) [triangle t]
  (p : Type) [parallelogram p]
  (s : Type) [square s]
  (r : Type) [rhombus r] :
  is_triangle_intuitive_diagram t ->
  is_parallelogram_intuitive_diagram p ->
  square_to_parallelogram s ->
  rhombus_to_parallelogram r ->
  option = B :=
by
  sorry

end correct_oblique_method_statement_l74_74054


namespace catch_up_distance_l74_74822

def v_a : ℝ := 10 -- A's speed in kmph
def v_b : ℝ := 20 -- B's speed in kmph
def t : ℝ := 10 -- Time in hours when B starts after A

theorem catch_up_distance : v_b * t + v_a * t = 200 :=
by sorry

end catch_up_distance_l74_74822


namespace functional_equation_solution_l74_74905

theorem functional_equation_solution (f : ℤ → ℤ) 
    (h : ∀ m n : ℤ, f(f(m) + n) + f(m) - f(n) + f(3*m) + 2014 = 0) : 
    ∀ n : ℤ, f(n) = 2*n + 1007 := 
by 
  sorry

end functional_equation_solution_l74_74905


namespace P_n_eq_x_solutions_l74_74249

noncomputable def P_1 (x : ℝ) : ℝ := x^2 - 2
noncomputable def P : ℕ → (ℝ → ℝ)
| 1       => P_1
| (n + 1) => λ x => P_1 (P n x)

lemma P_n_cos (n : ℕ) (t : ℝ) : P n (2 * Real.cos t) = 2 * Real.cos (2^n * t) := sorry

theorem P_n_eq_x_solutions (n : ℕ) (x : ℝ) :
  P n x = x ↔
  ∃ m k : ℕ, 
    (x = 2 * Real.cos (2 * m * Real.pi / (2^n - 1)) ∧ m < 2^(n - 1)) ∨ 
    (x = 2 * Real.cos (2 * k * Real.pi / (2^n + 1)) ∧ k < 2^(n - 1)) := sorry

end P_n_eq_x_solutions_l74_74249


namespace exists_large_subset_no_receives_l74_74683

variable (Club Member : Type)
variable [Fintype Club] [Nonempty Club]

structure HatExchange where
  sender receiver : Member

variable [DecidableEq Member]

def sends_hats (exchanges : Finset (HatExchange Member)) :=
  ∀ {a b : Member}, a ≠ b → (∃ e ∈ exchanges, e.sender = a ∧ e.receiver = b)

noncomputable def maximal_no_receive_subsets 
  (exchanges : Finset (HatExchange Member)) (members : Finset Member) : Finset (Finset Member) :=
  -- This body is left for the implementation of finding the maximal subsets
  sorry

theorem exists_large_subset_no_receives (exchanges : Finset (HatExchange Member)) (S : Finset Member) :
  ∃ (T : Finset Member), T ⊆ S ∧ T.card ≥ 10 ∧ ¬ ∃ x y ∈ T, x ≠ y ∧ (∃ e ∈ exchanges, e.sender = x ∧ e.receiver = y) :=
begin
  -- The proof of the theorem, which we skip (as this is the statement only)
  sorry
end

end exists_large_subset_no_receives_l74_74683


namespace cheaper_module_cost_l74_74345

theorem cheaper_module_cost (x : ℝ) :
  (21 * x + 10 = 62.50) → (x = 2.50) :=
by
  intro h
  sorry

end cheaper_module_cost_l74_74345


namespace area_of_square_l74_74449

theorem area_of_square (P : ℝ) (A_circle : ℝ) (side : ℝ) (area_square : ℝ) : 
  A_circle = 39424 → P = sqrt (A_circle / real.pi) → P = 4 * side → area_square = side ^ 2 → area_square = 784 :=
by {
  intros hA_circle hP_side h_perim h_area,
  sorry
}

end area_of_square_l74_74449


namespace ratio_of_inscribed_squares_l74_74860

theorem ratio_of_inscribed_squares (x y : ℝ) 
  (h1 : ∃ (triangle : Triangle), inscribed_square_at_right_angle_vertex triangle 5 12 13 x)
  (h2 : ∃ (triangle : Triangle), inscribed_square_on_hypotenuse triangle 5 12 13 y) :
  x / y = 89 / 110 := 
sorry

end ratio_of_inscribed_squares_l74_74860


namespace complement_P_inter_Q_l74_74659

def P : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}
def complement_P : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_P_inter_Q : (complement_P ∩ Q) = {x | 1 < x ∧ x < 2} := by
  sorry

end complement_P_inter_Q_l74_74659


namespace sin_sum_of_acute_triangle_l74_74418

theorem sin_sum_of_acute_triangle (α β γ : ℝ) (h_acute : α < π / 2 ∧ β < π / 2 ∧ γ < π / 2) (h_sum : α + β + γ = π) : 
  sin α + sin β + sin γ > 2 :=
sorry

end sin_sum_of_acute_triangle_l74_74418


namespace smallest_Y_l74_74600

theorem smallest_Y (U : ℕ) (Y : ℕ) (hU : U = 15 * Y) 
  (digits_U : ∀ d ∈ Nat.digits 10 U, d = 0 ∨ d = 1) 
  (div_15 : U % 15 = 0) : Y = 74 :=
sorry

end smallest_Y_l74_74600


namespace geometric_sequence_ratio_l74_74622

/-
Given a geometric sequence {a_n} with common ratio q ≠ -1 and q ≠ 1,
and S_n is the sum of the first n terms of the geometric sequence.
Given S_{12} = 7 S_{4}, prove:
S_{8}/S_{4} = 3
-/

theorem geometric_sequence_ratio {a_n : ℕ → ℝ} (q : ℝ) (h₁ : q ≠ -1) (h₂ : q ≠ 1)
  (S : ℕ → ℝ) (hSn : ∀ n, S n = a_n 0 * (1 - q ^ n) / (1 - q)) (h : S 12 = 7 * S 4) :
  S 8 / S 4 = 3 :=
by
  sorry

end geometric_sequence_ratio_l74_74622


namespace intersection_cardinality_l74_74629

open Set Int

noncomputable def A : Set ℝ := {x | x^2 - 5 * x + 4 ≥ 0}
noncomputable def A_complement : Set ℝ := {x | 1 < x ∧ x < 4}
noncomputable def B : Set ℤ := {x | abs (x - 1) ≤ 2}

theorem intersection_cardinality :
  (A_complement ∩ B.to_set).card = 2 :=
sorry

end intersection_cardinality_l74_74629


namespace circles_intersect_l74_74259

noncomputable def circle1 := {c : ℝ × ℝ // c = (-1, -4)}
noncomputable def circle2 := {c : ℝ × ℝ // c = (2, 2)}

noncomputable def radius1 : ℝ := 5
noncomputable def radius2 : ℝ := real.sqrt 10

noncomputable def distance_centers : ℝ := real.sqrt ((2 + 1)^2 + (2 + 4)^2)

theorem circles_intersect 
  (h1 : radius1 = 5)
  (h2 : radius2 = real.sqrt 10)
  (h3 : distance_centers = real.sqrt 25 * 3)
  : radius1 - radius2 < distance_centers ∧ distance_centers < radius1 + radius2 := 
sorry

end circles_intersect_l74_74259


namespace jackson_weeks_of_school_l74_74367

def jackson_sandwich_per_week : ℕ := 2

def missed_wednesdays : ℕ := 1
def missed_fridays : ℕ := 2
def total_missed_sandwiches : ℕ := missed_wednesdays + missed_fridays

def total_sandwiches_eaten : ℕ := 69

def total_sandwiches_without_missing : ℕ := total_sandwiches_eaten + total_missed_sandwiches

def calculate_weeks_of_school (total_sandwiches : ℕ) (sandwiches_per_week : ℕ) : ℕ :=
total_sandwiches / sandwiches_per_week

theorem jackson_weeks_of_school : calculate_weeks_of_school total_sandwiches_without_missing jackson_sandwich_per_week = 36 :=
by
  sorry

end jackson_weeks_of_school_l74_74367


namespace Sn_solution_l74_74646

-- Definitions for the problem
def sequence_sum (a : ℕ → ℝ) (n : ℕ) := ∑ i in range n, a (i + 1)
def a_1 : ℝ := 2
def S_n (a : ℕ → ℝ) (n : ℕ) := sequence_sum a n
def S_eq (a : ℕ → ℝ) (n : ℕ) : ℝ := S_n a n

-- Condition for n >= 2
def cond (a : ℕ → ℝ) (n : ℕ) : Prop :=
  n ≥ 2 ∧ S_eq a n ≠ S_eq a (n - 1)

-- Theorem statement
theorem Sn_solution (a : ℕ → ℝ) (n : ℕ) (hn : n ≥ 2) (h_sum_cond : ∀ n, S_eq a n * S_eq a (n - 1) + a n = 0) :
  S_eq a n = 2 / (2 * n - 1) :=
by {
  sorry
}

end Sn_solution_l74_74646


namespace jason_lemonade_calories_l74_74368

def lemon_juice_calories_per_100g : ℝ := 30
def sugar_calories_per_100g : ℝ := 400
def water_calories_per_100g : ℝ := 0

def lemon_juice_grams : ℝ := 150
def sugar_grams : ℝ := 200
def water_grams : ℝ := 500
def lemonade_grams : ℝ := 300

theorem jason_lemonade_calories :
  let total_calories := lemon_juice_grams * (lemon_juice_calories_per_100g / 100) +
                        sugar_grams * (sugar_calories_per_100g / 100) +
                        water_grams * (water_calories_per_100g / 100),
      total_grams := lemon_juice_grams + sugar_grams + water_grams,
      calories_per_gram := total_calories / total_grams
  in lemonade_grams * calories_per_gram = 298 :=
by {
  let total_calories := lemon_juice_grams * (lemon_juice_calories_per_100g / 100) + 
                        sugar_grams * (sugar_calories_per_100g / 100) + 
                        water_grams * (water_calories_per_100g / 100),
      total_grams := lemon_juice_grams + sugar_grams + water_grams,
      calories_per_gram := total_calories / total_grams

  calc lemonade_grams * calories_per_gram = 300 * (845 / 850) : by sorry
    ... = 298 : by sorry
}

end jason_lemonade_calories_l74_74368


namespace gibbs_free_energy_change_and_spontaneity_l74_74828

-- Defining the constants and conditions
def T_C := 95 -- Temperature in Celsius
def T := 273 + T_C -- Temperature in Kelvin
def ΔH₀ := -147.2 -- Enthalpy change in kJ
def ΔS₀_J_per_K := -17.18 -- Entropy change in J/K
def ΔS₀ := ΔS₀_J_per_K / 1000 -- Entropy change in kJ/K

-- Statement to prove the Gibbs free energy change and spontaneity
theorem gibbs_free_energy_change_and_spontaneity :
  ΔG₀ = ΔH₀ - T * ΔS₀ ∧ ΔG₀ < 0 := 
  by
    let ΔG₀ := ΔH₀ - T * ΔS₀,
    have h_ΔG₀ : ΔG₀ = -140.9 := sorry,
    have h_spontaneous : ΔG₀ < 0 := sorry,
    exact ⟨h_ΔG₀, h_spontaneous⟩

end gibbs_free_energy_change_and_spontaneity_l74_74828


namespace initial_percentage_of_chemical_x_l74_74835

theorem initial_percentage_of_chemical_x (P : ℝ) (h1 : 20 + 80 * P = 44) : P = 0.3 :=
by sorry

end initial_percentage_of_chemical_x_l74_74835


namespace tray_height_is_correct_l74_74179

noncomputable def height_of_tray (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : ℝ :=
  if side_length = 120 ∧ cut_distance = 5 ∧ cut_angle = 45 then
    (cut_distance * sqrt 3) / 2
  else
    0

theorem tray_height_is_correct :
  height_of_tray 120 5 45 = (5 * sqrt 3) / 2 :=
by
  sorry

end tray_height_is_correct_l74_74179


namespace find_x_l74_74926

theorem find_x (x : ℝ) (h : log x 64 = 3) : x = 4 :=
sorry

end find_x_l74_74926


namespace cube_properties_l74_74151

-- Define basic variables and constants
def edge_length : ℝ := 8
def conversion_factor_cm_to_mm : ℝ := 100

-- Define the expressions for total edge length, surface area, and volume
def total_length_of_edges (a : ℝ) : ℝ := 12 * a
def surface_area_in_cm2 (a : ℝ) : ℝ := 6 * a ^ 2
def surface_area_in_mm2 (a : ℝ) : ℝ := surface_area_in_cm2 a * conversion_factor_cm_to_mm ^ 2
def volume_of_cube (a : ℝ) : ℝ := a ^ 3

-- Translate the mathematical problem to a Lean 4 statement
theorem cube_properties : 
  total_length_of_edges edge_length = 96 ∧
  surface_area_in_mm2 edge_length = 38400 ∧
  volume_of_cube edge_length = 512 := 
by
  unfold total_length_of_edges surface_area_in_cm2 surface_area_in_mm2 volume_of_cube edge_length conversion_factor_cm_to_mm
  split
  sorry
  split
  sorry
  sorry

end cube_properties_l74_74151


namespace sum_of_angles_less_than_450_l74_74440

-- Define a heptagon inscribed in a circle
def heptagon (A₁ A₂ A₃ A₄ A₅ A₆ A₇ : Point) : Prop :=
  circular A₁ A₂ A₃ A₄ A₅ A₆ A₇

-- Define the condition that the center of the circle is inside the heptagon
def center_inside_heptagon {A₁ A₂ A₃ A₄ A₅ A₆ A₇ C : Point} 
  (h_heptagon: heptagon A₁ A₂ A₃ A₄ A₅ A₆ A₇) : Prop :=
  inside_heptagon C A₁ A₂ A₃ A₄ A₅ A₆ A₇

-- Define the angles at vertices A₁, A₃, A₅
def angle_A₁ {A₁ A₂ A₃ A₄ A₅ A₆ A₇ : Point} := angle A₁

def angle_A₃ {A₁ A₂ A₃ A₄ A₅ A₆ A₇ : Point} := angle A₃ 

def angle_A₅ {A₁ A₂ A₃ A₄ A₅ A₆ A₇ : Point} := angle A₅ 

-- Main theorem statement
theorem sum_of_angles_less_than_450 {A₁ A₂ A₃ A₄ A₅ A₆ A₇ C : Point}
  (h_heptagon: heptagon A₁ A₂ A₃ A₄ A₅ A₆ A₇) 
  (h_center_inside: center_inside_heptagon h_heptagon):
  angle_A₁ + angle_A₃ + angle_A₅ < 450 := 
sorry

end sum_of_angles_less_than_450_l74_74440


namespace number_of_vegetarian_only_l74_74684

theorem number_of_vegetarian_only (total_veg : ℕ) (both_veg_nonveg : ℕ) (total_veg = 21) (both_veg_nonveg = 8) : 
  total_veg - both_veg_nonveg = 13 :=
by
  -- Proof goes here
  sorry

end number_of_vegetarian_only_l74_74684


namespace process_cannot_continue_indefinitely_l74_74783

theorem process_cannot_continue_indefinitely (n : ℕ) (hn : 2018 ∣ n) :
  ¬(∀ m, ∃ k, (10*m + k) % 11 = 0 ∧ (10*m + k) / 11 ∣ n) :=
sorry

end process_cannot_continue_indefinitely_l74_74783


namespace seeds_per_flowerbed_l74_74738

theorem seeds_per_flowerbed :
  ∀ (total_seeds flowerbeds seeds_per_bed : ℕ), 
  total_seeds = 32 → 
  flowerbeds = 8 → 
  seeds_per_bed = total_seeds / flowerbeds → 
  seeds_per_bed = 4 :=
  by 
    intros total_seeds flowerbeds seeds_per_bed h_total h_flowerbeds h_calc
    rw [h_total, h_flowerbeds] at h_calc
    exact h_calc

end seeds_per_flowerbed_l74_74738


namespace flower_bed_length_l74_74762

theorem flower_bed_length (a b : ℝ) :
  ∀ width : ℝ, (6 * a^2 - 4 * a * b + 2 * a = 2 * a * width) → width = 3 * a - 2 * b + 1 :=
by
  intros width h
  sorry

end flower_bed_length_l74_74762


namespace find_u_v_w_x_l74_74713

theorem find_u_v_w_x (A B C : ℝ) (hC_obtuse : C > π / 2)
  (h1 : cos A ^ 2 + cos C ^ 2 + 2 * sin A * sin C * cos B = 16 / 9)
  (h2 : cos C ^ 2 + cos B ^ 2 + 2 * sin C * sin B * cos A = 17 / 10) :
  ∃ u v w x : ℤ, u > 0 ∧ v > 0 ∧ w > 0 ∧ x > 0 ∧ gcd (u + v) x = 1 ∧
    ¬ ∃ p : ℤ, p.prime ∧ p^2 ∣ w ∧ 
    cos B ^ 2 + cos A ^ 2 + 2 * sin B * sin A * cos C = (u - v * real.sqrt w) / x ∧ 
    u + v + w + x = 368 :=
by
  sorry

end find_u_v_w_x_l74_74713


namespace vector_dot_product_zero_implies_orthogonal_l74_74875

theorem vector_dot_product_zero_implies_orthogonal
  (a b : ℝ → ℝ)
  (h0 : ∀ (x y : ℝ), a x * b y = 0) :
  ¬(a = 0 ∨ b = 0) := 
sorry

end vector_dot_product_zero_implies_orthogonal_l74_74875


namespace additional_savings_l74_74180

def window_price : ℕ := 100

def special_offer (windows_purchased : ℕ) : ℕ :=
  windows_purchased + windows_purchased / 6 * 2

def dave_windows : ℕ := 10

def doug_windows : ℕ := 12

def total_windows := dave_windows + doug_windows

def calculate_windows_cost (windows_needed : ℕ) : ℕ :=
  if windows_needed % 8 = 0 then (windows_needed / 8) * 6 * window_price
  else ((windows_needed / 8) * 6 + (windows_needed % 8)) * window_price

def separate_savings : ℕ :=
  window_price * (dave_windows + doug_windows) - (calculate_windows_cost dave_windows + calculate_windows_cost doug_windows)

def combined_savings : ℕ :=
  window_price * total_windows - calculate_windows_cost total_windows

theorem additional_savings :
  separate_savings + 200 = combined_savings :=
sorry

end additional_savings_l74_74180


namespace X_lies_on_incircle_of_ABC_l74_74737

theorem X_lies_on_incircle_of_ABC
    {A B C X : Point} (hX_AC : OnSegment X A C)
    (h_tangent : TangentAtIncircle (triangle A B X) (triangle B C X)) :
    OnIncircle X (triangle A B C) := 
sorry

end X_lies_on_incircle_of_ABC_l74_74737


namespace cyclic_quadrilateral_after_3_operations_permissible_quadrilateral_after_6_operations_l74_74484

-- Define a type for points in the plane
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define a function for the symmetry operation. Given a point P and midpoint M, it returns the symmetric point.
def symmetric_point (P M : Point) : Point :=
{ x := 2 * M.x - P.x,
  y := 2 * M.y - P.y }

-- Extra definitions for diagonal midpoints and specific operations may be needed
noncomputable def midpoint (P1 P2 : Point) : Point :=
{ x := (P1.x + P2.x) / 2,
  y := (P1.y + P2.y) / 2 }

-- Define convex quadrilateral structure
structure ConvexQuadrilateral :=
(A B C D : Point)
(is_convex : Prop) -- Add necessary properties to ensure convexity and distinct sides

-- Define the operations
def operation (quad: ConvexQuadrilateral) (i: ℕ) : ConvexQuadrilateral :=
-- Implementation of operation based on the cyclic process
sorry

-- Main theorems
theorem cyclic_quadrilateral_after_3_operations {quad : ConvexQuadrilateral} 
  (cyclic : Prop) : (operation (operation (operation quad 0) 1) 2) = quad :=
sorry

theorem permissible_quadrilateral_after_6_operations {quad : ConvexQuadrilateral} :
  (operation (operation (operation (operation (operation (operation quad 0) 1) 2) 3) 4) 5) = quad :=
sorry

end cyclic_quadrilateral_after_3_operations_permissible_quadrilateral_after_6_operations_l74_74484


namespace hyperbola_passing_point_l74_74287

theorem hyperbola_passing_point (x y k : ℝ) (h : y = k / x) (hx1 : x = 1) (hy_2 : y = -2) : k = -2 :=
by
  -- Use the given conditions and the given hyperbola equation
  rw [hx1, hy_2] at h
  have h : -2 = k / 1 := by exact h
  -- Simplify the equation to solve for k
  rw div_one at h
  exact h

end hyperbola_passing_point_l74_74287


namespace find_g_of_2_l74_74066

-- Define the assumptions
variables (g : ℝ → ℝ)
axiom condition : ∀ x : ℝ, x ≠ 0 → 5 * g (1 / x) + (3 * g x) / x = Real.sqrt x

-- State the theorem to prove
theorem find_g_of_2 : g 2 = -(Real.sqrt 2) / 16 :=
by
  sorry

end find_g_of_2_l74_74066


namespace magnitude_complex_sum_l74_74923

-- Define the complex numbers
def z1 : ℂ := 3 + 4 * Complex.I
def z2 : ℂ := 2 - 5 * Complex.I

-- Define the sum of the two complex numbers
def z_sum : ℂ := z1 + z2

-- State the theorem about the magnitude of the sum
theorem magnitude_complex_sum : |z_sum| = Real.sqrt 26 := by
  sorry

end magnitude_complex_sum_l74_74923


namespace part_1_values_of_a_b_part_2_f_decreasing_on_interval_part_3_find_range_of_m_l74_74991

def f (x : ℝ) : ℝ := 1 - (2 * 3 ^ x) / (3 ^ x + 1)

theorem part_1_values_of_a_b :
  ∃ a b : ℝ, a = 2 ∧ b = 2 := 
by
  sorry

theorem part_2_f_decreasing_on_interval :
  ∀ x1 x2 : ℝ, -4 < x1 ∧ x1 < 2 ∧ -4 < x2 ∧ x2 < 2 ∧ x1 < x2 → f x1 > f x2 :=
by
  sorry

theorem part_3_find_range_of_m :
  ∀ m : ℝ, f (m - 2) + f (2 * m + 1) > 0 → 0 < m ∧ m < 1 / 3 :=
by
  sorry

end part_1_values_of_a_b_part_2_f_decreasing_on_interval_part_3_find_range_of_m_l74_74991


namespace sum_of_first_n_terms_l74_74362

noncomputable def sum_series (n : ℕ) : ℝ :=
  8 * (1 - 1 / (n + 1))

def a_seq (n : ℕ) : ℝ :=
  (∑ i in Finset.range (n + 1), i / (n + 1))

def b_seq (n : ℕ) : ℝ :=
  2 / (a_seq n * a_seq (n + 1))

def sum_b_seq (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b_seq i

theorem sum_of_first_n_terms (n : ℕ) :
  sum_b_seq n = sum_series n := by 
  sorry

end sum_of_first_n_terms_l74_74362


namespace min_n_for_inequality_l74_74026

def chain_power (base: ℕ) : ℕ → ℕ
| 0     := 1
| 1     := base
| (n+2) := base ^ chain_power base (n+1)

theorem min_n_for_inequality : ∃ n : ℕ, (n = 10 ∧ chain_power 9 9 < chain_power 3 n ∧ ∀ m < 10, chain_power 9 9 ≥ chain_power 3 m) := sorry

end min_n_for_inequality_l74_74026


namespace barycentric_vector_identity_l74_74055

variables {A B C X : Type} [AddCommGroup X] [Module ℝ X]
variables (α β γ : ℝ) (A B C X : X)

-- Defining the barycentric coordinates condition
axiom barycentric_coords : α • A + β • B + γ • C = X

-- Additional condition that sum of coordinates is 1
axiom sum_coords : α + β + γ = 1

-- The theorem to prove
theorem barycentric_vector_identity :
  (X - A) = β • (B - A) + γ • (C - A) :=
sorry

end barycentric_vector_identity_l74_74055


namespace total_worth_of_stock_l74_74538

theorem total_worth_of_stock (W : ℝ) 
    (h1 : 0.2 * W * 0.1 = 0.02 * W)
    (h2 : 0.6 * (0.8 * W) * 0.05 = 0.024 * W)
    (h3 : 0.2 * (0.8 * W) = 0.16 * W)
    (h4 : (0.024 * W) - (0.02 * W) = 400) 
    : W = 100000 := 
sorry

end total_worth_of_stock_l74_74538


namespace probability_defective_unit_l74_74496

theorem probability_defective_unit 
  (T : ℝ)
  (machine_a_output : ℝ := 0.4 * T)
  (machine_b_output : ℝ := 0.6 * T)
  (machine_a_defective_rate : ℝ := 9 / 1000)
  (machine_b_defective_rate : ℝ := 1 / 50)
  (total_defective_units : ℝ := (machine_a_output * machine_a_defective_rate) + (machine_b_output * machine_b_defective_rate))
  (probability_defective : ℝ := total_defective_units / T) :
  probability_defective = 0.0156 :=
by
  sorry

end probability_defective_unit_l74_74496


namespace points_of_intersection_l74_74568

open Real

-- Define the graphs as functions
def f1 (x : ℝ) := log x / log 5
def f2 (x : ℝ) := 3 / (log x / log 5)
def f3 (x : ℝ) := log x / log (1/5)
def f4 (x : ℝ) := 5 * (log (1/5) / log x)

-- Prove that there are exactly 5 points of intersection among these graphs for positive x
theorem points_of_intersection : 
  ∃ S : set ℝ, (∀ x ∈ S, 0 < x) ∧ (∀ x1 x2 ∈ S, x1 ≠ x2 → (f1 x1 = f2 x1 ∨ f1 x1 = f3 x1 ∨ f1 x1 = f4 x1 ∨ 
    f2 x1 = f3 x1 ∨ f2 x1 = f4 x1 ∨ f3 x1 = f4 x1) ∧ 
     (f1 x2 = f2 x2 ∨ f1 x2 = f3 x2 ∨ f1 x2 = f4 x2 ∨ f2 x2 = f3 x2 ∨ f2 x2 = f4 x2 ∨ f3 x2 = f4 x2)) ∧ 
   S.finset.card = 5 := sorry

end points_of_intersection_l74_74568


namespace woman_l74_74855

def man's_rate : ℝ := 5 -- miles per hour
def woman's_waiting_time : ℝ := 2 / 60 -- hours
def man's_walking_time : ℝ := 6 / 60 -- hours

theorem woman's_travel_rate (W : ℝ) :
  (man's_rate * man's_walking_time = W * woman's_waiting_time) → W = 15 := by
sorry

end woman_l74_74855


namespace sqrt_set_l74_74787

theorem sqrt_set (a : ℝ) (h₁ : a^2 = 256) : 
  (16 : ℝ) = sqrt 256 → {x : ℝ | x^2 = 16} = {-4, 4} := 
by
  sorry

end sqrt_set_l74_74787


namespace find_angle_equiv_terminal_side_find_angle_equiv_terminal_side_l74_74506

theorem find_angle_equiv_terminal_side :
  ∃ (α : ℝ), 0 < α ∧ α < 360 ∧ ∃ (k : ℤ), α = -60 + k * 360 := 
begin
  use 300,
  split,
  { norm_num, },
  split,
  { norm_num, },
  use 1,
  norm_num,
end

# Allow the theorem above to be left without an explicit proof since the question's focus is on the correct statement setup.
-- sorry is a placeholder for the actual proof
theorem find_angle_equiv_terminal_side' :
  ∃ (α : ℝ), 0 < α ∧ α < 360 ∧ ∃ (k : ℤ), α = -60 + k * 360 := 
begin
  sorry
end

end find_angle_equiv_terminal_side_find_angle_equiv_terminal_side_l74_74506


namespace maximum_height_l74_74839

-- Define the quadratic function h(t)
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 60

-- Define our proof problem
theorem maximum_height : ∃ t : ℝ, h t = 140 :=
by
  let t := -80 / (2 * -20)
  use t
  sorry

end maximum_height_l74_74839


namespace four_intersections_implies_abs_k_gt_sqrt3_over_3_max_area_ABCD_is_4_sqrt3_rhombus_implies_k_sqrt15_over_3_l74_74780

variable {k : ℝ}

def line1 (x : ℝ) : ℝ := k * x + 2
def line2 (x : ℝ) : ℝ := k * x - 2
def ellipse (x y : ℝ) : Prop := (x ^ 2) / 6 + (y ^ 2) / 2 = 1

theorem four_intersections_implies_abs_k_gt_sqrt3_over_3 :
  (∃ x y₁, ellipse x y₁ ∧ y₁ = line1 x) ∧
  (∃ x y₂, ellipse x y₂ ∧ y₂ = line2 x) →
  |k| > sqrt 3 / 3 :=
sorry

theorem max_area_ABCD_is_4_sqrt3 :
  (∃ A B C D : ℝ × ℝ, 
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse C.1 C.2 ∧ ellipse D.1 D.2 ∧
    ∃ k, line1 A.1 = A.2 ∧ line1 B.1 = B.2 ∧ line2 C.1 = C.2 ∧ line2 D.1 = D.2) →
  (∃ S : ℝ, S = 4 * sqrt 3) :=
sorry

theorem rhombus_implies_k_sqrt15_over_3 :
  (∃ A B C D : ℝ × ℝ, 
    ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ellipse C.1 C.2 ∧ ellipse D.1 D.2 ∧
    ∃ k, line1 A.1 = A.2 ∧ line1 B.1 = B.2 ∧ line2 C.1 = C.2 ∧ line2 D.1 = D.2 ∧
    -- The vectors form a rhombus
    (∃ m : ℝ, ∀ X Y : ℝ × ℝ, X ∈ {A, B, C, D} → Y ∈ {A, B, C, D} → X ≠ Y → 
      m = sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2))) →
  k = sqrt 15 / 3 ∨ k = -sqrt 15 / 3 :=
sorry

end four_intersections_implies_abs_k_gt_sqrt3_over_3_max_area_ABCD_is_4_sqrt3_rhombus_implies_k_sqrt15_over_3_l74_74780


namespace exists_triangle_perimeter_lt_1cm_circumradius_gt_1km_l74_74214

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (a * b * c) / (4 * Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem exists_triangle_perimeter_lt_1cm_circumradius_gt_1km :
  ∃ (A B C : ℝ) (a b c : ℝ), a + b + c < 0.01 ∧ circumradius a b c > 1000 :=
by
  sorry

end exists_triangle_perimeter_lt_1cm_circumradius_gt_1km_l74_74214


namespace area_of_square_field_l74_74827

-- Define side length
def side_length : ℕ := 20

-- Theorem statement about the area of the square field
theorem area_of_square_field : (side_length * side_length) = 400 := by
  sorry

end area_of_square_field_l74_74827


namespace tan_sin_relation_l74_74701

theorem tan_sin_relation (A B : ℝ) (hA : 0 < A) (hB : 0 < B) (hπA : A < π) (hπB : B < π) :
  ¬((tan A > tan B) ↔ (sin A > sin B)) :=
sorry

end tan_sin_relation_l74_74701


namespace triangle_side_condition_angle_condition_l74_74343

variable (a b c A B C : ℝ)

theorem triangle_side_condition (a_eq : a = 2) (b_eq : b = Real.sqrt 7) (h : a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B) :
  c = 3 :=
  sorry

theorem angle_condition (angle_eq : Real.sqrt 3 * Real.sin (2 * A - π / 6) - 2 * Real.sin (C - π / 12)^2 = 0) :
  A = π / 4 :=
  sorry

end triangle_side_condition_angle_condition_l74_74343


namespace carl_wins_in_4950_configurations_l74_74726

noncomputable def num_distinct_configurations_at_Carl_win : ℕ :=
  sorry
  
theorem carl_wins_in_4950_configurations :
  num_distinct_configurations_at_Carl_win = 4950 :=
sorry

end carl_wins_in_4950_configurations_l74_74726


namespace no_such_b_c_exist_l74_74918

theorem no_such_b_c_exist :
  ¬ ∃ (b c : ℝ), (∃ (k l : ℤ), (k ≠ l ∧ (k ^ 2 + b * ↑k + c = 0) ∧ (l ^ 2 + b * ↑l + c = 0))) ∧
                  (∃ (m n : ℤ), (m ≠ n ∧ (2 * (m ^ 2) + (b + 1) * ↑m + (c + 1) = 0) ∧ 
                                        (2 * (n ^ 2) + (b + 1) * ↑n + (c + 1) = 0))) :=
sorry

end no_such_b_c_exist_l74_74918


namespace number_of_sets_P_satisfying_conditions_l74_74030

def U : Set Int := {x | abs x < 4}
def S : Set Int := {-2, 1, 3}
def complement_U_P_is_subset_S (P : Set Int) : Prop := (U \ P) ⊆ S

theorem number_of_sets_P_satisfying_conditions : 
  (∃ (Ps : Finset (Set Int)), 
    Ps.card = 8 ∧ 
    ∀ (P ∈ Ps), complement_U_P_is_subset_S P) :=
sorry

end number_of_sets_P_satisfying_conditions_l74_74030


namespace camping_trip_total_percentage_l74_74671

variable (T : ℝ) (P_more_than_100 : ℝ) (P_less_than_100 : ℝ)
variable (h1 : P_more_than_100 = 18 / 100)
variable (h2 : P_less_than_100 = 54 / 100)
variable (h3 : T = P_more_than_100 + P_less_than_100)

theorem camping_trip_total_percentage :
  T = 72 / 100 :=
by
  rw [h1, h2, h3]
  calc
    T = 18 / 100 + 54 / 100 : by rw h3
    _ = 72 / 100 : by ring

end camping_trip_total_percentage_l74_74671


namespace cost_of_items_l74_74156

namespace GardenCost

variables (B T C : ℝ)

/-- Given conditions defining the cost relationships and combined cost,
prove the specific costs of bench, table, and chair. -/
theorem cost_of_items
  (h1 : T + B + C = 650)
  (h2 : T = 2 * B - 50)
  (h3 : C = 1.5 * B - 25) :
  B = 161.11 ∧ T = 272.22 ∧ C = 216.67 :=
sorry

end GardenCost

end cost_of_items_l74_74156


namespace starting_number_of_range_l74_74789

-- Given that the sum of the first n consecutive odd integers is n^2
-- and the sum of all odd integers between a and 41 inclusive is 416,
-- prove that a is 11
theorem starting_number_of_range (n : ℕ) (a : ℕ) (h1 : ∀ n : ℕ, (finset.range (2 * n)).filter odd).sum id = n^2
  (h2 : (finset.range (42 - a)).filter (λ x, odd (a + 2 * x)).sum id = 416) : a = 11 :=
sorry

end starting_number_of_range_l74_74789


namespace value_of_m_l74_74647

theorem value_of_m (m : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 = 4 → (abs (x + y - m) / real.sqrt (2) = 1)) → m = real.sqrt 2 ∨ m = -real.sqrt 2) := sorry

end value_of_m_l74_74647


namespace non_empty_proper_subsets_of_three_element_set_l74_74309

theorem non_empty_proper_subsets_of_three_element_set :
  ∃ M : Finset (Fin 3), ¬(M = ∅) ∧ M ⊂ Finset.univ → (Finset.univ.filter (λ M, ¬(M = ∅ ∧ M ⊂ Finset.univ))).card = 6 :=
by
  sorry

end non_empty_proper_subsets_of_three_element_set_l74_74309


namespace first_term_in_pq_expansion_l74_74043

theorem first_term_in_pq_expansion (p q : ℕ) (hp : 2 ≤ p) (hq : 2 ≤ q) : 
  ∃ first_term : ℕ, first_term = p^(q-1) - p + 1 :=
by
  use p^(q-1) - p + 1
  sorry

end first_term_in_pq_expansion_l74_74043


namespace line_through_point_intersects_yaxis_triangular_area_l74_74527

theorem line_through_point_intersects_yaxis_triangular_area 
  (a T : ℝ) 
  (h : 0 < a) 
  (line_eqn : ∀ x y : ℝ, x = -a * y + a → 2 * T * x + a^2 * y - 2 * a * T = 0) 
  : ∃ (m b : ℝ), (forall x y : ℝ, y = m * x + b) := 
by
  sorry

end line_through_point_intersects_yaxis_triangular_area_l74_74527


namespace money_left_after_expenditures_l74_74904

variable (initial_amount : ℝ) (P : initial_amount = 15000)
variable (gas_percentage food_fraction clothing_fraction entertainment_percentage : ℝ) 
variable (H1 : gas_percentage = 0.35) (H2 : food_fraction = 0.2) (H3 : clothing_fraction = 0.25) (H4 : entertainment_percentage = 0.15)

theorem money_left_after_expenditures
  (money_left : ℝ):
  money_left = initial_amount * (1 - gas_percentage) *
                (1 - food_fraction) * 
                (1 - clothing_fraction) * 
                (1 - entertainment_percentage) → 
  money_left = 4972.50 :=
by
  sorry

end money_left_after_expenditures_l74_74904


namespace min_balls_to_draw_l74_74842

theorem min_balls_to_draw (red green yellow blue white black purple : ℕ)
  (h_red : red = 35)
  (h_green : green = 18)
  (h_yellow : yellow = 15)
  (h_blue : blue = 17)
  (h_white : white = 12)
  (h_black : black = 12)
  (h_purple : purple = 8) : 
  ∃ n, n = 89 ∧ ∀ (draws : list ℕ), draws.length = n →
    ∃ (color : ℕ), (draws.count color ≥ 15) :=
by
  sorry

end min_balls_to_draw_l74_74842


namespace oblique_asymptote_l74_74806

def f (x : ℝ) : ℝ := (3 * x^2 - 4 * x - 8) / (2 * x + 3)

theorem oblique_asymptote : 
  ∀ (x : ℝ), tendsto (λ x, f x - ((3/2) * x - 5)) (at_top) (nhds 0) := 
sorry

end oblique_asymptote_l74_74806


namespace trigonometric_identity_l74_74771

theorem trigonometric_identity
  (a b c d : ℕ)
  (h : ∀ x : ℝ, cos (2 * x) + cos (4 * x) + cos (8 * x) + cos (10 * x) = a * cos (b * x) * cos (c * x) * cos (d * x)) :
  a + b + c + d = 14 :=
by { sorry }

end trigonometric_identity_l74_74771


namespace abs_ab_eq_2_sqrt_111_l74_74274

theorem abs_ab_eq_2_sqrt_111 (a b : ℝ) (h1 : b^2 - a^2 = 25) (h2 : a^2 + b^2 = 49) : |a * b| = 2 * Real.sqrt 111 := sorry

end abs_ab_eq_2_sqrt_111_l74_74274


namespace digit_B_for_divisibility_l74_74079

theorem digit_B_for_divisibility (B : ℕ) (h : (40000 + 1000 * B + 100 * B + 20 + 6) % 7 = 0) : B = 1 :=
sorry

end digit_B_for_divisibility_l74_74079


namespace hcf_of_specific_fractions_l74_74501

def hcf_of_fractions (a b c : ℚ) : ℚ :=
  let numerators := [a.num, b.num, c.num]
  let denominators := [a.denom, b.denom, c.denom]
  let hcf_numerators := numerators.gcd
  let lcm_denominators := denominators.lcm
  (hcf_numerators : ℚ) / (lcm_denominators : ℚ)

theorem hcf_of_specific_fractions : hcf_of_fractions (2/3) (4/9) (1/3) = 1/9 :=
by
  sorry

end hcf_of_specific_fractions_l74_74501


namespace floor_abs_sum_l74_74587

def abs (x : ℝ) : ℝ := if x < 0 then -x else x
def floor (x : ℝ) : ℤ := ⌊x⌋

theorem floor_abs_sum : floor (abs (-5.7)) + abs (floor (-5.7)) = 11 :=
by
  sorry

end floor_abs_sum_l74_74587


namespace ball_distribution_l74_74576

theorem ball_distribution (n : ℕ) (P_white P_red P_yellow : ℚ) (num_white num_red num_yellow : ℕ) 
  (total_balls : n = 6)
  (prob_white : P_white = 1/2)
  (prob_red : P_red = 1/3)
  (prob_yellow : P_yellow = 1/6) :
  num_white = 3 ∧ num_red = 2 ∧ num_yellow = 1 := 
sorry

end ball_distribution_l74_74576


namespace cost_price_per_meter_l74_74491

theorem cost_price_per_meter
    (meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) (total_price : ℕ)
    (h1 : meters = 85) (h2 : selling_price = 8925) (h3 : profit_per_meter = 25) :
  total_price = selling_price - profit_per_meter * meters → total_price / meters = 80 :=
by {
  -- Given conditions
  intros h_total_price,
  sorry
}

end cost_price_per_meter_l74_74491


namespace number_of_integer_palindromes_with_even_tens_digit_l74_74208

/-- 
Prove that the number of integer palindromes between 100 and 1000 
with an even tens digit is 45.
-/
theorem number_of_integer_palindromes_with_even_tens_digit :
  let palindromes := {n | 100 ≤ n ∧ n < 1000 ∧ is_palindrome n ∧ even (n / 10 % 10)}, -- Defining the set of palindromes between 100 and 1000 with an even tens digit
  {n | 100 ≤ n ∧ n < 1000 ∧ is_palindrome n ∧ even (n / 10 % 10)}.card = 45 :=
by
  sorry

/-- A helper function to check if an integer is a palindrome. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

end number_of_integer_palindromes_with_even_tens_digit_l74_74208


namespace find_breadth_l74_74929

-- Define variables and constants
variables (SA l h w : ℝ)

-- Given conditions
axiom h1 : SA = 2400
axiom h2 : l = 15
axiom h3 : h = 16

-- Define the surface area equation for a cuboid 
def surface_area := 2 * (l * w + l * h + w * h)

-- Statement to prove
theorem find_breadth : surface_area l w h = SA → w = 30.97 := sorry

end find_breadth_l74_74929


namespace average_fifth_and_eighth_l74_74069

theorem average_fifth_and_eighth (q : Fin 12 → ℕ) 
  (h1 : (∑ i, q i) = 216)
  (h2 : (∑ i in Finset.range 4, q i) = 56)
  (h3 : (∑ i in (Finset.range 9).filter (λ i, 4 ≤ i ∧ i ≠ 7), q i) = 80) 
  (h4 : (∑ i in Finset.Ico 8 12, q i) = 50) :
  (q 4 + q 7) / 2 = 15 := 
by 
  sorry

end average_fifth_and_eighth_l74_74069


namespace ratio_problem_l74_74670

theorem ratio_problem 
  (x y z w : ℚ) 
  (h1 : x / y = 12) 
  (h2 : z / y = 4) 
  (h3 : z / w = 3 / 4) : 
  w / x = 4 / 9 := 
  sorry

end ratio_problem_l74_74670


namespace jovana_added_23_pounds_l74_74706

def initial_weight : ℕ := 5
def final_weight : ℕ := 28

def added_weight : ℕ := final_weight - initial_weight

theorem jovana_added_23_pounds : added_weight = 23 := 
by sorry

end jovana_added_23_pounds_l74_74706


namespace positive_a_makes_log_inequality_true_l74_74814

theorem positive_a_makes_log_inequality_true (c : ℝ) (h1 : c > 1) (a : ℝ) (h2 : a ≠ 1) :
  (log a c > log 3 c) ↔ a = 2 :=
by sorry

end positive_a_makes_log_inequality_true_l74_74814


namespace cylinder_water_depth_l74_74849

theorem cylinder_water_depth 
  (height radius : ℝ)
  (h_ge_zero : height ≥ 0)
  (r_ge_zero : radius ≥ 0)
  (total_height : height = 1200)
  (total_radius : radius = 100)
  (above_water_vol : 1 / 3 * π * radius^2 * height = 1 / 3 * π * radius^2 * 1200) :
  height - 800 = 400 :=
by
  -- Use provided constraints and logical reasoning on structures
  sorry

end cylinder_water_depth_l74_74849


namespace no_positive_a_b_for_all_primes_l74_74747

theorem no_positive_a_b_for_all_primes :
  ∀ (a b : ℕ), 0 < a → 0 < b → ∃ (p q : ℕ), p > 1000 ∧ q > 1000 ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ ¬Prime (a * p + b * q) :=
by
  sorry

end no_positive_a_b_for_all_primes_l74_74747


namespace parallel_vectors_lambda_collinear_vectors_k_l74_74133

/-- 
Problem 1: Given vectors a = (1, 2), b = (-3, 2), c = (3, 4) and a real number λ. 
Prove that if (a + λb) is parallel to c, then λ = -1/9. 
--/
theorem parallel_vectors_lambda (λ : ℝ) : 
  let a := (1 : ℝ, 2 : ℝ)
  let b := (-3 : ℝ, 2 : ℝ)
  let c := (3 : ℝ, 4 : ℝ)
  ((a.1 + λ * b.1) / c.1 = (a.2 + λ * b.2) / c.2) → λ = -1/9 :=
sorry

/-- 
Problem 2: Given non-zero, not collinear vectors e₁ and e₂.
Prove that if vectors k * e₁ + e₂ and e₁ + k * e₂ are collinear, then k = 1 or k = -1. 
--/
theorem collinear_vectors_k (k : ℝ) (e₁ e₂ : ℝ × ℝ) (hne₁ : e₁ ≠ (0, 0)) (hne₂ : e₂ ≠ (0, 0)) :
  let v1 := (k * e₁.1 + e₂.1, k * e₁.2 + e₂.2)
  let v2 := (e₁.1 + k * e₂.1, e₁.2 + k * e₂.2)
  (∃ m : ℝ, v1 = (m * v2.1, m * v2.2)) → k = 1 ∨ k = -1 :=
sorry

end parallel_vectors_lambda_collinear_vectors_k_l74_74133


namespace radius_scaling_l74_74325

theorem radius_scaling (r R : ℝ) (hr : 0 < r) :
  (π * R^2 = 6 * π * r^2) → (R = real.sqrt 6 * r) :=
by {
  -- Here's where the proof would go
  sorry
}

end radius_scaling_l74_74325


namespace parabola_distance_l74_74962

theorem parabola_distance {x y : ℝ} (h₁ : y^2 = 4 * x) (h₂ : abs (x + 2) = 5) : 
  (sqrt ((x + 1)^2 + y^2) = 4) :=
sorry

end parabola_distance_l74_74962


namespace investment_percentage_l74_74159

theorem investment_percentage (x : ℝ) :
  (4000 * (x / 100) + 3500 * 0.04 + 2500 * 0.064 = 500) ↔ (x = 5) :=
by
  sorry

end investment_percentage_l74_74159


namespace principal_sum_l74_74120

theorem principal_sum (A1 A2 : ℝ) (I P : ℝ) 
  (hA1 : A1 = 1717) 
  (hA2 : A2 = 1734) 
  (hI : I = A2 - A1)
  (h_simple_interest : A1 = P + I) : P = 1700 :=
by
  sorry

end principal_sum_l74_74120


namespace concyclic_points_l74_74013

theorem concyclic_points
  {A B C P Q R S : Point}
  (hABC : ∀ (∠A : ∠ABC), ∠A < 90)
  (h_perp_AC_B : perpendicular Line.AC Line.BP)
  (h_perp_AB_C : perpendicular Line.AB Line.CR)
  (h_circle_AC : Circle.diameter A C)
  (h_circle_AB : Circle.diameter A B)
  (h_intersect_AC : intersects Line.BP Circle_AC P Q)
  (h_intersect_AB : intersects Line.CR Circle_AB R S) :
  concyclic {P, Q, R, S} :=
sorry

end concyclic_points_l74_74013


namespace total_floor_area_l74_74138

noncomputable theory

def slab_length_cm : ℕ := 200
def slabs_count : ℕ := 30

theorem total_floor_area : 
    let slab_area_m2 := (slab_length_cm * slab_length_cm) / 10000 in
    let total_area := slab_area_m2 * slabs_count in
    total_area = 120 :=
by
  sorry

end total_floor_area_l74_74138


namespace range_of_a_l74_74636

-- Let us define the problem conditions and statement in Lean
theorem range_of_a
  (a : ℝ)
  (h : ∀ x y : ℝ, x < y → (3 - a)^x > (3 - a)^y) :
  2 < a ∧ a < 3 :=
sorry

end range_of_a_l74_74636


namespace brenda_leads_by_5_l74_74572

theorem brenda_leads_by_5
  (initial_lead : ℕ)
  (brenda_play : ℕ)
  (david_play : ℕ)
  (h_initial : initial_lead = 22)
  (h_brenda_play : brenda_play = 15)
  (h_david_play : david_play = 32) :
  initial_lead + brenda_play - david_play = 5 :=
by {
  rw [h_initial, h_brenda_play, h_david_play],
  norm_num, -- simplify to get the answer
  sorry
}

end brenda_leads_by_5_l74_74572


namespace perfect_square_form_l74_74914

theorem perfect_square_form (N : ℕ) (hN : 0 < N) : 
  ∃ x : ℤ, 2^N - 2 * (N : ℤ) = x^2 ↔ N = 1 ∨ N = 2 :=
by
  sorry

end perfect_square_form_l74_74914


namespace vertex_of_quadratic_l74_74767

theorem vertex_of_quadratic (a h k : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a * (x - h)^2 + k) → (h, k) = (3, 1) :=
by
  intro hf
  have hf_def : ∀ x, f x = 2 * (x - 3)^2 + 1 := hf
  sorry

end vertex_of_quadratic_l74_74767


namespace product_of_N_l74_74674

variable (N K : ℝ)

theorem product_of_N (h : N + 3 / N = K) : 
  let roots := { N | N * N - K * N + 3 = 0 } in
  ∏ x in roots, x = 3 :=
by
  sorry

end product_of_N_l74_74674


namespace IO_intersects_AB_BC_l74_74373

variables {A B C I O : Type}
variables [EuclideanGeometry I A B C O]

noncomputable def is_acute_triangle (ABC : Triangle) : Prop :=
  acute (angle A B C) ∧ acute (angle B C A) ∧ acute (angle C A B)

noncomputable def AB_AC_BC (ABC : Triangle) : Prop := 
  side A B < side A C ∧ side A C < side B C

noncomputable def is_incenter (I : Point) (ABC : Triangle) : Prop :=
  ∃ (incenter : Point), ∀ (P : Point), distance I A = distance I B ∧ distance I B = distance I C

noncomputable def is_circumcenter (O : Point) (ABC : Triangle) : Prop :=
  ∃ (circumcenter : Point), ∀ (P : Point), distance O P = distance O A ∧ distance O P = distance O B ∧ distance O P = distance O C

theorem IO_intersects_AB_BC (ABC : Triangle) (I O : Point) : 
  is_acute_triangle ABC -> 
  AB_AC_BC ABC -> 
  is_incenter I ABC -> 
  is_circumcenter O ABC -> 
  (line IO intersects segment A B) ∧ (line IO intersects segment B C) := 
by
  sorry


end IO_intersects_AB_BC_l74_74373


namespace max_possible_acute_angled_triangles_l74_74736
-- Define the sets of points on lines a and b
def maxAcuteAngledTriangles (n : Nat) : Nat :=
  let sum1 := (n * (n - 1) / 2)  -- Sum of first (n-1) natural numbers
  let sum2 := (sum1 * 50) - (n * (n - 1) * (2 * n - 1) / 6) -- Applying the given formula
  (2 * sum2)  -- Multiply by 2 for both colors of alternating points

-- Define the main theorem
theorem max_possible_acute_angled_triangles : maxAcuteAngledTriangles 50 = 41650 := by
  sorry

end max_possible_acute_angled_triangles_l74_74736


namespace nested_series_approx_l74_74580

theorem nested_series_approx:
  let S := 1502 + (1501 / 3) + (1500 / 3^2) + ... + (13 / 3^1489) + (12 / 3^1490)
  in S ≈ 3004.75 :=
by
  sorry

end nested_series_approx_l74_74580


namespace items_purchased_l74_74861

/-- Define the price of the nth item. -/
def price (n : ℕ) : ℝ :=
  0.99 + n

/-- Define the total cost of n items. -/
def total_cost (n : ℕ) : ℝ :=
  ∑ i in finset.range n, price i

/-- The main theorem stating the number of items Uncle Mané could have bought. -/
theorem items_purchased : ∃ (n : ℕ), total_cost n = 125.74 :=
by sorry

end items_purchased_l74_74861


namespace parallel_planes_of_perpendicular_to_line_l74_74627

variables {l : ℝ^3} {α β : set ℝ^3}

-- We define the perpendicularity of a line to a plane
def is_perpendicular (l : ℝ^3) (α : set ℝ^3) : Prop :=
  ∀ p ∈ α, l ⋅ p = 0

-- We define the parallelism of two planes
def are_parallel (α β : set ℝ^3) : Prop :=
  ∀ p1 p2 ∈ α, ∀ p3 p4 ∈ β, (p2 - p1) ⋅ (p4 - p3) = 0

theorem parallel_planes_of_perpendicular_to_line 
  (h1 : is_perpendicular l α) 
  (h2 : is_perpendicular l β) :
  are_parallel α β :=
sorry

end parallel_planes_of_perpendicular_to_line_l74_74627


namespace number_of_ways_place_dominos_l74_74011

-- Define the problem conditions
variables {n t : ℕ}
def A : ℕ → ℕ → ℕ

-- Base cases
def A : ℕ → ℕ → ℕ 
| 0, 0 := 1
| n, 1 := 2 * (n - 1)

-- Recurrence relation
def A : ℕ → ℕ → ℕ
| n, t := A (n - 1) t + 2 * A (n - 2) (t - 1) + A (n - 3) (t - 2)

-- Theorem to be proven
theorem number_of_ways_place_dominos (n t : ℕ) (h_n : 0 < n) (h_t : 0 < t) :
  A n t = Nat.choose (2 * (n - t)) t := sorry

end number_of_ways_place_dominos_l74_74011


namespace find_ab_g_interval_l74_74630

noncomputable def f (a b x : ℝ) := -2 * a * sin (2 * x + (Real.pi / 6)) + 2 * a + b
noncomputable def g (x : ℝ) := 4 * sin (2 * x + (Real.pi / 6)) - 1

theorem find_ab_g_interval:
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → -5 ≤ f 2 (-5) x ∧ f 2 (-5) x ≤ 1) →
  (∀ x, log (g x) > 0) →
  (∀ k : ℤ, ∀ x,
    k * Real.pi < x ∧ x < k * Real.pi + Real.pi / 6 ∨
    k * Real.pi + Real.pi / 6 ≤ x ∧ x < k * Real.pi + Real.pi / 3 →
    (∀ a b, a = 2 ∧ b = -5 ∧ g x > 1) →
    (0 < deriv (f a b) x ∧ k * Real.pi < x ∧ x < k * Real.pi + Real.pi / 6) ∨
    (0 > deriv (f a b) x ∧ k * Real.pi + Real.pi / 6 ≤ x ∧ x < k * Real.pi + Real.pi / 3)) :=
begin
  sorry
end

end find_ab_g_interval_l74_74630


namespace michael_regular_hours_l74_74730

-- Define the constants and conditions
def regular_rate : ℝ := 7
def overtime_rate : ℝ := 14
def total_earnings : ℝ := 320
def total_hours : ℝ := 42.857142857142854

-- Declare the proof problem
theorem michael_regular_hours :
  ∃ R O : ℝ, (regular_rate * R + overtime_rate * O = total_earnings) ∧ (R + O = total_hours) ∧ (R = 40) :=
by
  sorry

end michael_regular_hours_l74_74730


namespace part1_prob_seventh_grade_part2_prob_one_seventh_one_eighth_l74_74403

-- Definitions corresponding to conditions
def total_students : ℕ := 4
def seventh_grade_students : ℕ := 2
def eighth_grade_students : ℕ := 2

-- Probability problem statements to be proven
theorem part1_prob_seventh_grade : 
  (seventh_grade_students.to_real / total_students.to_real) = 1 / 2 := 
by 
  sorry

theorem part2_prob_one_seventh_one_eighth : 
  ((seventh_grade_students.to_real * eighth_grade_students.to_real * 2) / (total_students * (total_students - 1)).to_real) = 2 / 3 := 
by 
  sorry

end part1_prob_seventh_grade_part2_prob_one_seventh_one_eighth_l74_74403


namespace distinct_right_triangles_l74_74305

theorem distinct_right_triangles (a b c : ℕ) (h : a^2 = 2016) :
  (∃ b c, b ∈ ℕ ∧ c ∈ ℕ ∧ c^2 = a^2 + b^2) → ∃ n, n = 12 :=
by {
  sorry
}

end distinct_right_triangles_l74_74305


namespace chord_proof_point_proof_l74_74917

def circle (a : ℝ) := λ (x y : ℝ), (x - a)^2 + (y - a - 1)^2 = 9
def line (x y : ℝ) := x + y - 3 = 0
def point_A : ℝ × ℝ := (3, 0)
def point_O : ℝ × ℝ := (0, 0)
def M_on_circle (x y a : ℝ) := circle a x y

noncomputable def chord_condition (a : ℝ) : Prop :=
  let center := (a, a + 1)
  let d := 2 * Real.sqrt 2
  Real.abs (a - 1) = 2

noncomputable def point_condition (a : ℝ) : Prop :=
  let dist_sqr := (λ (x y : ℝ), (x - 3)^2 + y^2)
  let radius_sqr := (λ (x y : ℝ), 4 * (x^2 + y^2))
  ∃ (x y : ℝ), circle a x y ∧ 
    dist_sqr x y = radius_sqr x y

theorem chord_proof (a : ℝ) : chord_condition a → (a = -1 ∨ a = 3) := 
sorry

theorem point_proof (a : ℝ) : point_condition a → 
  a ∈ Set.Ioc (-1 - (5 * Real.sqrt 2) / 2) (-1 - Real.sqrt 2 / 2) ∨ 
  a ∈ Set.Ioc (-1 + Real.sqrt 2 / 2) (-1 + (5 * Real.sqrt 2) / 2) := 
sorry

end chord_proof_point_proof_l74_74917


namespace complement_intersection_l74_74388

-- Definitions for the sets
def U : Set ℕ := {1, 2, 3, 4, 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Definition of the complement of a set with respect to a universal set
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Theorem to prove
theorem complement_intersection :
  complement U (A ∩ B) = {1, 4, 6} :=
by
  sorry

end complement_intersection_l74_74388


namespace find_special_integers_l74_74225

theorem find_special_integers (n : ℕ) (h : n > 1) :
  (∀ d, d ∣ n ∧ d > 1 → ∃ a r, a > 0 ∧ r > 1 ∧ d = a^r + 1) ↔ (n = 10 ∨ ∃ a, a > 0 ∧ n = a^2 + 1) :=
by
  sorry

end find_special_integers_l74_74225


namespace rays_intersect_l74_74709

noncomputable def midpoint (A B : Point) (M : Point) : Prop :=
  dist A M = dist B M ∧ M.is_on_line_segment A B

def same_side (A B C X Y : Point) : Prop :=
  ∃ ℓ : Line, C ∉ ℓ ∧ A ∈ ℓ ∧ B ∈ ℓ ∧ X ∈ ℓᶜ ∧ Y ∈ ℓᶜ

def angle_eq (A B C X Y M : Point) : Prop :=
  ∃ ℓ1 ℓ2 ℓ3 ℓ4 : Line,
  is_angle_eq (angle (line_through B A) (line_through A X)) (angle (line_through A C) (line_through C M)) ∧
  is_angle_eq (angle (line_through B Y) (line_through Y A)) (angle (line_through M C) (line_through C B))

theorem rays_intersect (A B C M X Y : Point) 
  (h1 : midpoint A B M)
  (h2 : angle_eq A B C X Y M)
  (h3 : same_side A B C X Y) : 
  ∃ P : Point, P.is_on_ray A X ∧ P.is_on_ray B Y ∧ P.is_on_line_through C M :=
sorry

end rays_intersect_l74_74709


namespace find_N_l74_74908

theorem find_N : {N : ℕ // N > 0 ∧ ∃ k : ℕ, 2^N - 2 * N = k^2} = {1, 2} := 
    sorry

end find_N_l74_74908


namespace cylinder_radius_l74_74791

variables (a : ℝ) (A B D1 A1 B1 C1 D C : Type) [cube : is_cube ABCD A1B1C1D1]

/-- The radius of the base of the cylinder given the conditions of the problem -/
theorem cylinder_radius (hA : A ∈ cylinder_lateral_surface)
                       (hB : B ∈ cylinder_lateral_surface)
                       (hD1 : D1 ∈ cylinder_lateral_surface)
                       (haxis : cylinder.axis ∥ line(DC1))
                       (h_edge : edge_length ABCD A1B1C1D1 = a) :
    cylinder.radius = (3 * a * real.sqrt 2) / 4 :=
by sorry

end cylinder_radius_l74_74791


namespace animal_products_sampled_l74_74518

theorem animal_products_sampled
  (grains : ℕ)
  (oils : ℕ)
  (animal_products : ℕ)
  (fruits_vegetables : ℕ)
  (total_sample : ℕ)
  (total_food_types : grains + oils + animal_products + fruits_vegetables = 100)
  (sample_size : total_sample = 20)
  : (animal_products * total_sample / 100) = 6 := by
  sorry

end animal_products_sampled_l74_74518


namespace farmer_apples_l74_74772

theorem farmer_apples (original_apples given_apples remaining_apples : ℕ) (h1 : original_apples = 127) (h2 : remaining_apples = 39) : given_apples = 88 :=
by
  have h : given_apples = original_apples - remaining_apples
  rw [h1, h2]
  exact Nat.sub_self
  sorry

end farmer_apples_l74_74772


namespace alice_bush_count_l74_74185

theorem alice_bush_count :
  let side_length := 24
  let num_sides := 3
  let bush_space := 3
  (num_sides * side_length) / bush_space = 24 :=
by
  sorry

end alice_bush_count_l74_74185


namespace determine_z2_l74_74353

noncomputable def i : ℂ := complex.I

def circle_property (z : ℂ) : Prop :=
  complex.abs (z - i) = 1

theorem determine_z2 (z1 z2 : ℂ) (h1: circle_property z1) (h2: circle_property z2) (h3: complex.re (z1 * z2) = 0) (h4: complex.arg z1 = π / 6) :
  z2 = -√3 / 2 + (3 / 2) * i :=
sorry

end determine_z2_l74_74353


namespace any_two_vectors_in_plane_cannot_always_be_basis_l74_74191

-- Definitions: Let V be a 2-dimensional vector space (essentially ℝ^2)
noncomputable def V : Type := ℝ × ℝ

-- Definition: Two vectors are linearly independent if and only if they are not collinear
def linearly_independent (v1 v2 : V) : Prop :=
  ¬ ∃ (c : ℝ), (v2 = (c • v1) ∨ v1 = (c • v2))

-- Statement: Any two vectors in a plane can be used as a basis is false (given they may be collinear)
theorem any_two_vectors_in_plane_cannot_always_be_basis :
  ¬ ∀ (v1 v2 : V), ¬linearly_independent v1 v2 → False :=
sorry

end any_two_vectors_in_plane_cannot_always_be_basis_l74_74191


namespace base_4_vs_base_7_digit_difference_l74_74306

theorem base_4_vs_base_7_digit_difference :
  let n := 1234
  let num_digits_base (b : Nat) (n : Nat) : Nat :=
    Nat.log n b + 1
  num_digits_base 4 n - num_digits_base 7 n = 2 :=
by
  let n := 1234
  let num_digits_base := λ b n, Nat.log n b + 1
  sorry

end base_4_vs_base_7_digit_difference_l74_74306


namespace divisible_by_12_factorial_divisible_by_13_factorial_l74_74390

-- Definition of the conditions
def is_correct_stack (deck : List (ℕ × ℕ)) : Prop :=
  deck.head = (0, 1) ∧ -- The top card is Ace of Spades
  ∀ (i : ℕ), i < 51 → (deck.get i).fst = (deck.get (i+1)).fst ∨ (deck.get i).snd = (deck.get (i+1)).snd ∧ -- Adjacent cards must match by suit or rank
  (deck.get 51).fst = (deck.get 0).fst ∨ (deck.get 51).snd = (deck.get 0).snd -- Top and bottom cards must also match

-- Question a: Number of ways is divisible by 12!
theorem divisible_by_12_factorial (deck : List (ℕ × ℕ)) (h : is_correct_stack deck) : 
  (number_of_ways_to_stack_deck deck) % (12.factorial) = 0 := sorry

-- Question b: Number of ways is divisible by 13!
theorem divisible_by_13_factorial (deck : List (ℕ × ℕ)) (h : is_correct_stack deck) : 
  (number_of_ways_to_stack_deck deck) % (13.factorial) = 0 := sorry

end divisible_by_12_factorial_divisible_by_13_factorial_l74_74390


namespace initial_water_percentage_l74_74397

theorem initial_water_percentage (W : ℕ) (V1 V2 V3 W3 : ℕ) (h1 : V1 = 10) (h2 : V2 = 15) (h3 : V3 = V1 + V2) (h4 : V3 = 25) (h5 : W3 = 2) (h6 : (W * V1) / 100 = (W3 * V3) / 100) : W = 5 :=
by
  sorry

end initial_water_percentage_l74_74397


namespace find_m_on_line_through_c_and_d_l74_74526

theorem find_m_on_line_through_c_and_d (c d : ℝ^n) (hc : c ≠ d) (m : ℝ) :
  (∃ t : ℝ, c + t • (d - c) = m • c + (5/6 : ℝ) • d) ↔ m = -1/6 := by
  sorry

end find_m_on_line_through_c_and_d_l74_74526


namespace mean_temperature_is_minus_one_l74_74760

-- Define the list of temperatures
def temperatures : List ℝ := [-6.5, -3, -2, 4, 2.5]

-- Define the mean function
def mean (temps : List ℝ) : ℝ :=
  temps.sum / (temps.length : ℝ)

-- The theorem we need to prove
theorem mean_temperature_is_minus_one : mean temperatures = -1 := by
  have : temperatures.sum = -5 := by simp [temperatures]
  have : (temperatures.length : ℝ) = 5 := by simp [temperatures]
  show mean temperatures = -1
  simp [mean, this]
  sorry

end mean_temperature_is_minus_one_l74_74760


namespace find_yz_over_x_squared_l74_74939

theorem find_yz_over_x_squared :
  ∃ (x y z k : ℝ),
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    (det ![
      ![1, 2 * k, 4],
      ![4, 2 * k, -3],
      ![3, 5, -4]
    ] = 0) ∧
    (x + 2 * k * y + 4 * z = 0) ∧
    (4 * x + 2 * k * y - 3 * z = 0) ∧
    (3 * x + 5 * y - 4 * z = 0) ∧
    ∃ ratio : ℝ, ratio = y * z / x^2 :=
sorry

end find_yz_over_x_squared_l74_74939


namespace frequency_of_2_in_20231222_l74_74487

def count_occurrences (s : String) (c : Char) : Nat :=
  s.foldl (fun count ch => if ch = c then count + 1 else count) 0

theorem frequency_of_2_in_20231222 :
  let s := "20231222"
  let total_digits := String.length s
  let count_2 := count_occurrences s '2'
  total_digits = 8 →
  count_2 = 5 →
  count_2 / total_digits = 5 / 8 :=
by
     intro s total_digits count_2 h1 h2
     rw [h1, h2]
     exact rfl
     sorry

end frequency_of_2_in_20231222_l74_74487


namespace largest_n_multiple_of_7_l74_74475

theorem largest_n_multiple_of_7 (n : ℕ) (h1 : n < 50000) (h2 : (5*(n-3)^5 - 3*n^2 + 20*n - 35) % 7 = 0) : n = 49999 :=
sorry

end largest_n_multiple_of_7_l74_74475


namespace max_sector_area_proof_l74_74832

noncomputable def max_sector_area (c : ℝ) : ℝ :=
  c^2 / 16

theorem max_sector_area_proof (R c α : ℝ) (h_perimeter : 2 * R + R * α = c) :
  let α := (c / R) - 2
  let S := (1/2) * R * α * R in
  S ≤ max_sector_area c := 
sorry

end max_sector_area_proof_l74_74832


namespace area_of_triangle_FGC_l74_74695

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def area_of_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_FGC :
  let A : ℝ × ℝ := (0, 5)
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (7, 0)
  let F : ℝ × ℝ := midpoint A B
  let G : ℝ × ℝ := midpoint A C
  area_of_triangle F G C = 4.375 := by
  sorry

end area_of_triangle_FGC_l74_74695


namespace sum_of_ages_l74_74036

variable (M E : ℝ)
variable (h1 : M = E + 9)
variable (h2 : M + 5 = 3 * (E - 3))

theorem sum_of_ages : M + E = 32 :=
by
  sorry

end sum_of_ages_l74_74036


namespace find_a_plus_b_l74_74019

noncomputable def det_3x3 (m₁ m₂ m₃ : ℝ × ℝ × ℝ) : ℝ :=
  m₁.1 * (m₂.2 * m₃.3 - m₂.3 * m₃.2) -
  m₁.2 * (m₂.1 * m₃.3 - m₂.3 * m₃.1) +
  m₁.3 * (m₂.1 * m₃.2 - m₂.2 * m₃.1)

theorem find_a_plus_b (a b : ℝ) (h_distinct: a ≠ b) 
  (h_det: det_3x3 (1, 6, 16) (4, a, b) (4, b, a) = 0) : 
  a + b = 88 := 
sorry

end find_a_plus_b_l74_74019


namespace book_length_ratio_is_4_l74_74034

-- Define the initial conditions
def pages_when_6 : ℕ := 8
def age_when_start := 6
def multiple_at_twice_age := 5
def multiple_eight_years_after := 3
def current_pages : ℕ := 480

def pages_when_12 := pages_when_6 * multiple_at_twice_age
def pages_when_20 := pages_when_12 * multiple_eight_years_after

theorem book_length_ratio_is_4 :
  (current_pages : ℚ) / pages_when_20 = 4 := by
  -- We need to show the proof for the equality
  sorry

end book_length_ratio_is_4_l74_74034


namespace log_inequality_conditions_l74_74782

theorem log_inequality_conditions (a x : ℝ) : 
  (a > 2 ∧ x > 1) ↔ (log (a - 1) (2 * x - 1) > log (a - 1) (x - 1)) := 
by 
  sorry

end log_inequality_conditions_l74_74782


namespace alice_coins_percentage_l74_74544

theorem alice_coins_percentage :
  let penny := 1
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let total_cents := penny + dime + quarter + half_dollar
  (total_cents / 100) * 100 = 86 :=
by
  sorry

end alice_coins_percentage_l74_74544


namespace B_wins_for_n_eq_8_C_wins_for_n_eq_7_l74_74507

-- Define the chessboard and the moving rules
structure Chessboard (n : ℕ) :=
  (row : ℕ)
  (col : ℕ)
  (valid : row > 0 ∧ row ≤ n ∧ col > 0 ∧ col ≤ n)

-- Define what it means for a square to be closer to the bottom-left corner
def closer_to_bottom_left {n : ℕ} (p q : Chessboard n) : Prop :=
  (p.row < q.row ∧ p.col <= q.col) ∨ (p.row <= q.row ∧ p.col < q.col)

-- Define the game rules and who the winner is
def game_winner (n : ℕ) : Chessboard n → string
| (⟨1, 1, _⟩) := "loser" -- square (1, 1) means the player to move loses
| pos := 
    if n = 8 then "B wins"
    else if n = 7 then "C wins"
    else "undetermined"

-- Define the main theorem statements
theorem B_wins_for_n_eq_8 : game_winner 8 (⟨0, 8, sorry⟩) = "B wins" :=
  sorry

theorem C_wins_for_n_eq_7 : game_winner 7 (⟨0, 7, sorry⟩) = "C wins" :=
  sorry

end B_wins_for_n_eq_8_C_wins_for_n_eq_7_l74_74507


namespace sum_of_angular_defects_eq_4pi_l74_74574

namespace Polyhedron

def angular_defect (α : ℝ) (angles : List ℝ) : ℝ := 
  2 * Real.pi - angles.sum

theorem sum_of_angular_defects_eq_4pi (V E F : ℕ) (angles_at_vertices : Fin V → List ℝ) 
  (hc : V - E + F = 2) :
  ∑ i, angular_defect (2 * Real.pi) (angles_at_vertices i) = 4 * Real.pi :=
by
  sorry

end Polyhedron

end sum_of_angular_defects_eq_4pi_l74_74574


namespace cos_theta_value_l74_74314

theorem cos_theta_value (θ : ℝ) (h1 : 6 * tan θ = 5 * sin θ) (h2 : 0 < θ ∧ θ < π) : cos θ = 5 / 6 :=
by sorry

end cos_theta_value_l74_74314


namespace part1_part2_l74_74989

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (h : m > 1) : ∃ x : ℝ, f x = (4 / (m - 1)) + m :=
by
  sorry

end part1_part2_l74_74989


namespace ratio_of_shaded_to_white_l74_74111

-- Definitions of conditions
def vertices_in_middle_of_sides (squares : List Square) : Prop :=
  ∀ s ∈ squares.tail, s.vertices ⊆ midpoints_of_corresponding_sides s.pred

def quarter_diagram : Diagram := upper_right_quarter_of full_figure

def shaded_region_triangles (diagram : Diagram) : ℕ :=
  count_triangles diagram shaded_region

def white_region_triangles (diagram : Diagram) : ℕ :=
  count_triangles diagram white_region

-- Theorem statement
theorem ratio_of_shaded_to_white (squares : List Square) (diagram : Diagram) :
  vertices_in_middle_of_sides squares →
  diagram = quarter_diagram →
  shaded_region_triangles diagram = 5 →
  white_region_triangles diagram = 3 →
  (shaded_region_triangles diagram) / (white_region_triangles diagram) = 5 / 3 :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_shaded_to_white_l74_74111


namespace mike_practices_hours_on_saturday_l74_74396

-- Definitions based on conditions
def weekday_hours : ℕ := 3
def weekdays_per_week : ℕ := 5
def total_hours : ℕ := 60
def weeks : ℕ := 3

def calculate_total_weekday_hours (weekday_hours weekdays_per_week weeks : ℕ) : ℕ :=
  weekday_hours * weekdays_per_week * weeks

def calculate_saturday_hours (total_hours total_weekday_hours weeks : ℕ) : ℕ :=
  (total_hours - total_weekday_hours) / weeks

-- Statement to prove
theorem mike_practices_hours_on_saturday :
  calculate_saturday_hours total_hours (calculate_total_weekday_hours weekday_hours weekdays_per_week weeks) weeks = 5 :=
by 
  sorry

end mike_practices_hours_on_saturday_l74_74396


namespace points_after_operations_l74_74045

theorem points_after_operations (initial_points : ℕ) (num_operations : ℕ) :
  (initial_points = 2010) → (num_operations = 3) → (∑ i in list.range num_operations, initial_points - 1 + i * (initial_points - 1)) = 16073 :=
by
  intros h_initial h_operations
  rw [h_initial, h_operations]
  sorry

end points_after_operations_l74_74045


namespace max_barons_l74_74680

def Knight : Type := {knight : Type // Prop}
def vassal_relation (k1 k2 : Knight) : Prop := sorry  -- placeholder for the vassal relationship
def wealthier (k1 k2 : Knight) : Prop := sorry  -- placeholder for the wealthier relationship

-- Conditions
axiom num_knights : ∀ (S : Set Knight), S.card = 32
axiom vassal_has_liege : ∀ (k1 k2 : Knight), vassal_relation k1 k2 → wealthier k2 k1
axiom knight_has_one_liege : ∀ (k1 : Knight), ∃! k2 : Knight, vassal_relation k1 k2
axiom knight_with_vassals_is_baron : ∀ (k : Knight), (∃ (S : Set Knight), S.card ≥ 4 ∧ ∀ k' ∈ S, vassal_relation k' k) → is_baron k
axiom vassal_of_vassal_not_my_vassal : ∀ (k1 k2 k3: Knight), vassal_relation k1 k2 → vassal_relation k2 k3 → ¬ vassal_relation k1 k3

-- Conclusion
theorem max_barons : ∃ S : Set Knight, (∀ k ∈ S, is_baron k) ∧ S.card = 7 :=
sorry

end max_barons_l74_74680


namespace ducks_in_larger_pond_l74_74685

theorem ducks_in_larger_pond (D : ℕ) (h1 : ∃ n : ℕ, n = 20) (h2 : 20 * 4 / 100 = 4) (h3 : 15 * D / 100) :
  ((4 + 15 * D / 100) = 16 * (20 + D) / 100) -> D = 80 :=
by
  intro h
  sorry

end ducks_in_larger_pond_l74_74685


namespace fraction_of_volume_occupied_l74_74948

theorem fraction_of_volume_occupied :
  ∀ (r : ℝ) (n : ℕ), 
  r = 1 ∧ 
  n = 14 ∧ 
  let tet_vol := (1/6) * (2 * sqrt 2 * sqrt 3)^3 in
  let sphere_vol := n * (4/3) * π * r^3 in
  abs ((sphere_vol / tet_vol) - 0.5116) < 0.0001 := by
  intros r n
  sorry

end fraction_of_volume_occupied_l74_74948


namespace vector_perpendicularity_l74_74297

theorem vector_perpendicularity (a b : ℝ × ℝ)
  (h1 : a = (1, 0))
  (h2 : b = (1/2, 1/2)) :
  let c := (a.1 - b.1, a.2 - b.2) in
  c.1 * b.1 + c.2 * b.2 = 0 :=
by sorry

end vector_perpendicularity_l74_74297


namespace minimize_abs_diff_factorial_l74_74531

theorem minimize_abs_diff_factorial (a1 a2 ... am b1 b2 ... bn : ℕ)
  (h_eq : 1729 = (a1.factorial * a2.factorial * ... * am.factorial) / (b1.factorial * b2.factorial * ... * bn.factorial))
  (h_a : a1 ≥ a2 ∧ a2 ≥ ... ∧ am > 0)
  (h_b : b1 ≥ b2 ∧ b2 ≥ ... ∧ bn > 0)
  (h_min : ∀ x y, x ≥ a1 ∧ y ≥ b1 → x + y ≥ a1 + b1) :
  (a1 - b1).abs = 2 := sorry

end minimize_abs_diff_factorial_l74_74531


namespace no_arithmetic_mean_l74_74651

def fraction1 := 5 / 8
def fraction2 := 3 / 4
def fraction3 := 9 / 12

def mean (x y : ℚ) := (x + y) / 2

theorem no_arithmetic_mean :
  (fraction1 ≠ mean fraction2 fraction3) ∧
  (fraction2 ≠ mean fraction1 fraction3) ∧
  (fraction3 ≠ mean fraction1 fraction2) :=
by
  sorry

end no_arithmetic_mean_l74_74651


namespace Harold_doughnuts_l74_74408

theorem Harold_doughnuts (cost_doughnut cost_coffee h_pay m_pay : ℝ) (D M : ℕ) (cost_d : cost_doughnut = 0.45) 
  (harold_payment : cost_doughnut * D + 4 * cost_coffee = 4.91)
  (melinda_payment : cost_doughnut * 5 + 6 * cost_coffee = 7.59) : D = 3 := by
  have cost_coffee_eq : cost_coffee = 0.89 :=
    calc
      cost_coffee = (7.59 - (cost_doughnut * 5)) / 6 : by { rw [melinda_payment], ring }
                ... = (7.59 - 2.25) / 6           : by { rw cost_d, norm_num }
                ... = 5.34 / 6                    : by norm_num
                ... = 0.89                        : by norm_num
  have h_eq : D = 3 :=
    calc
      D = (4.91 - 4 * cost_coffee) / cost_doughnut   : by { rw [harold_payment], ring }
        ... = (4.91 - 4 * 0.89) / 0.45             : by { rw cost_coffee_eq }
        ... = (4.91 - 3.56) / 0.45                 : by norm_num
        ... = 1.35 / 0.45                          : by norm_num
        ... = 3                                    : by norm_num
  exact h_eq

end Harold_doughnuts_l74_74408


namespace show_revenue_l74_74177

theorem show_revenue (tickets_first_showing : ℕ) 
                     (tickets_second_showing : ℕ) 
                     (ticket_price : ℕ) :
                      tickets_first_showing = 200 →
                      tickets_second_showing = 3 * tickets_first_showing →
                      ticket_price = 25 →
                      (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 :=
by
  intros h1 h2 h3
  have h4 : tickets_first_showing + tickets_second_showing = 800 := sorry -- Calculation step
  have h5 : (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 := sorry -- Calculation step
  exact h5

end show_revenue_l74_74177


namespace find_angle_ABC_l74_74669

-- Conditions definitions
def angleCBD : ℝ := 90
def angleSumAtB : ℝ := 180
def angleABD : ℝ := 30

-- Main theorem statement
theorem find_angle_ABC (h1 : angleCBD = 90) 
                       (h2 : angleSumAtB = 180) 
                       (h3 : angleABD = 30) : 
  let angleABC : ℝ := angleSumAtB - (angleABD + angleCBD)
  in angleABC = 60 := 
by
  sorry

end find_angle_ABC_l74_74669


namespace all_n_conditions_satisfy_l74_74224

open Nat

noncomputable def valid_n_values : Set ℕ :=
  { n | n > 1 ∧ (∀ k (a : Fin k → ℕ), 
    (∀ i, a i < n) ∧ 
    (∀ i, a i.coprime n) ∧ 
    (∀ i j, i < j → a i < a j) → 
    (∀ i, i < k - 1 → (a i + a (i + 1)) % 3 ≠ 0)) }

theorem all_n_conditions_satisfy : valid_n_values = {2, 4, 10} :=
by 
  sorry

end all_n_conditions_satisfy_l74_74224


namespace complement_A_l74_74376

-- Definitions for the conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}

-- Proof statement
theorem complement_A : (U \ A) = {x | x ≥ 1} := by
  sorry

end complement_A_l74_74376


namespace show_revenue_l74_74176

theorem show_revenue (tickets_first_showing : ℕ) 
                     (tickets_second_showing : ℕ) 
                     (ticket_price : ℕ) :
                      tickets_first_showing = 200 →
                      tickets_second_showing = 3 * tickets_first_showing →
                      ticket_price = 25 →
                      (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 :=
by
  intros h1 h2 h3
  have h4 : tickets_first_showing + tickets_second_showing = 800 := sorry -- Calculation step
  have h5 : (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 := sorry -- Calculation step
  exact h5

end show_revenue_l74_74176


namespace probability_both_hit_target_l74_74106

def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.7

theorem probability_both_hit_target :
  prob_A * prob_B = 0.56 :=
by sorry

end probability_both_hit_target_l74_74106


namespace largest_angle_in_hexagon_l74_74157

-- Defining the conditions
variables (A B x y : ℝ)
variables (C D E F : ℝ)
variable (sum_of_angles_in_hexagon : ℝ) 

-- Given conditions
def condition1 : A = 100 := by sorry
def condition2 : B = 120 := by sorry
def condition3 : C = x := by sorry
def condition4 : D = x := by sorry
def condition5 : E = (2 * x + y) / 3 + 30 := by sorry
def condition6 : 100 + 120 + C + D + E + F = 720 := by sorry

-- Statement to prove
theorem largest_angle_in_hexagon :
  ∃ (largest_angle : ℝ), largest_angle = max A (max B (max C (max D (max E F)))) ∧ largest_angle = 147.5 := sorry

end largest_angle_in_hexagon_l74_74157


namespace find_m_l74_74994

def A (m : ℝ) := {1, 3, real.sqrt m}
def B (m : ℝ) := {1, m}

theorem find_m (m : ℝ):
  (A m ∩ B m = B m) → (m = 0 ∨ m = 3) :=
by 
  sorry

end find_m_l74_74994


namespace N_square_solutions_l74_74910

theorem N_square_solutions :
  ∀ N : ℕ, (N > 0 → ∃ k : ℕ, 2^N - 2 * N = k^2) → (N = 1 ∨ N = 2) :=
by
  sorry

end N_square_solutions_l74_74910


namespace smaller_triangle_perimeter_l74_74398

theorem smaller_triangle_perimeter (p : ℕ) (p1 : ℕ) (p2 : ℕ) (p3 : ℕ) 
  (h₀ : p = 11)
  (h₁ : p1 = 5)
  (h₂ : p2 = 7)
  (h₃ : p3 = 9) : 
  p1 + p2 + p3 - p = 10 := by
  sorry

end smaller_triangle_perimeter_l74_74398


namespace geometric_sum_first_10_terms_l74_74621

open Function

-- Definitions and problem setup
variable {a : ℕ → ℝ} -- Geometric sequence
def a_1 := 1 -- First term
def a_4 := 1 / 8 -- Fourth term

-- Common ratio calculation
def q := ((1 / 8) / 1) ^ (1 / 3)

-- Proof goal
theorem geometric_sum_first_10_terms :
  let S_10 := (1 * (1 - (q^10)) / (1 - q)) in
  (S_10 = 2 - (1 / (2^9))) :=
by 
  have h1 : a 1 = 1 := by sorry
  have h4 : a 4 = 1 / 8 := by sorry
  have hq : q = (1 / 8) ^ (1 / 3) := by sorry
  sorry

end geometric_sum_first_10_terms_l74_74621


namespace cost_to_color_pattern_l74_74243

-- Define the basic properties of the squares
def square_side_length : ℕ := 4
def number_of_squares : ℕ := 4
def unit_cost (num_overlapping_squares : ℕ) : ℕ := num_overlapping_squares

-- Define the number of unit squares overlapping by different amounts
def unit_squares_overlapping_by_4 : ℕ := 1
def unit_squares_overlapping_by_3 : ℕ := 6
def unit_squares_overlapping_by_2 : ℕ := 12
def unit_squares_overlapping_by_1 : ℕ := 18

-- Calculate the total cost
def total_cost : ℕ :=
  unit_cost 4 * unit_squares_overlapping_by_4 +
  unit_cost 3 * unit_squares_overlapping_by_3 +
  unit_cost 2 * unit_squares_overlapping_by_2 +
  unit_cost 1 * unit_squares_overlapping_by_1

-- Statement to prove
theorem cost_to_color_pattern : total_cost = 64 := 
  sorry

end cost_to_color_pattern_l74_74243


namespace real_common_solution_l74_74581

theorem real_common_solution (y : ℝ) :
  (∃ x : ℝ, x^2 + y^2 = 3 ∧ x^2 - 4y + 6 = 0) ↔
  (y = -2 + sqrt 13 ∨ y = -2 - sqrt 13) :=
by {
  sorry,
}

end real_common_solution_l74_74581


namespace log_expression_simplifies_to_zero_l74_74559

theorem log_expression_simplifies_to_zero : 
  (1/2 : ℝ) * (Real.log 4) + Real.log 5 - Real.exp (0 * Real.log (Real.pi + 1)) = 0 := 
by
  sorry

end log_expression_simplifies_to_zero_l74_74559


namespace intersection_points_l74_74413

-- Definitions for the conditions
def point (name : String) := Type

def circle (name : String) := Type

def radius (C : circle name) : ℝ := 5 -- radius of circle D

def distance (P Q : point name) : ℝ := 8 -- distance from Q to the center of D

def circle_at (C : circle name) (r : ℝ) := Type

-- Define the circle centered at Q with radius 4cm
def circle_Q := circle_at (point "Q") 4

-- Define the circle D with radius 5cm
def circle_D := circle_at (circle "D") (radius (circle "D"))

-- Check the intersection points
theorem intersection_points : ∀ (Q : point "Q") (C : circle "D"), 
  (distance Q C < radius C + 4) ∧ (abs (radius C - 4) < distance Q C) → #(circle_D ∩ circle_Q) = 2 := 
by sorry

end intersection_points_l74_74413


namespace maria_threw_out_carrots_l74_74728

theorem maria_threw_out_carrots (initially_picked: ℕ) (picked_next_day: ℕ) (total_now: ℕ) (carrots_thrown_out: ℕ) :
  initially_picked = 48 → 
  picked_next_day = 15 → 
  total_now = 52 → 
  (initially_picked + picked_next_day - total_now = carrots_thrown_out) → 
  carrots_thrown_out = 11 :=
by
  intros
  sorry

end maria_threw_out_carrots_l74_74728


namespace even_sum_probability_l74_74462

theorem even_sum_probability :
  (let wheel1_even := 3 / 6;
       wheel1_odd := 3 / 6;
       wheel2_even := 3 / 4;
       wheel2_odd := 1 / 4;
       even_even := wheel1_even * wheel2_even;
       odd_odd := wheel1_odd * wheel2_odd;
       even_sum := even_even + odd_odd in
       even_sum = 1 / 2) :=
by
  let wheel1_even := 3 / 6
  let wheel1_odd := 3 / 6
  let wheel2_even := 3 / 4
  let wheel2_odd := 1 / 4
  let even_even := wheel1_even * wheel2_even
  let odd_odd := wheel1_odd * wheel2_odd
  let even_sum := even_even + odd_odd
  show even_sum = 1 / 2
  sorry

end even_sum_probability_l74_74462


namespace find_r_divisible_polynomial_l74_74606

theorem find_r_divisible_polynomial :
  ∃ r : ℝ, (9 * (r : ℝ) ^ 2 - (5 / 9) * r - (48 / 9) = 0 ∧ 9 * r ^ 3 - 54 = 0 ∧ r = 8 / 3) ∧
  ∃ s : ℝ, (9 * r ^ 2 - (5 / 9) - 48 + 54 = 9 * ((x - r)^2 * (x - s))) :=
begin
  sorry
end

end find_r_divisible_polynomial_l74_74606


namespace find_quadratic_polynomial_exists_l74_74237

theorem find_quadratic_polynomial_exists :
  ∃ (q : ℚ[X]), q.eval (-5) = 0 ∧ q.eval 3 = 0 ∧ q.eval 2 = -24 ∧ 
  q = Polynomial.C (24 / 7) * Polynomial.X ^ 2 + Polynomial.C (48 / 7) * Polynomial.X - Polynomial.C (360 / 7) :=
begin
  sorry
end

end find_quadratic_polynomial_exists_l74_74237


namespace tan_alpha_minus_pi_over_4_l74_74951

theorem tan_alpha_minus_pi_over_4 (α : Real) (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
  (Real.tan (α - π / 4) = -1/7) ∨ (Real.tan (α - π / 4) = -7) :=
by
  sorry

end tan_alpha_minus_pi_over_4_l74_74951


namespace frequency_of_2_l74_74488

def num_set := "20231222"
def total_digits := 8
def count_of_2 := 5

theorem frequency_of_2 : (count_of_2 : ℚ) / total_digits = 5 / 8 := by
  sorry

end frequency_of_2_l74_74488


namespace find_real_solutions_l74_74589

variable (x : ℝ)

theorem find_real_solutions :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔ 
  (x = -2 * Real.sqrt 14 ∨ x = 2 * Real.sqrt 14) := 
sorry

end find_real_solutions_l74_74589


namespace probability_two_red_two_blue_l74_74514

theorem probability_two_red_two_blue (r b total selected : ℕ) 
  (hr : r = 15) (hb : b = 12) (htotal : total = r + b) 
  (hselected : selected = 4)
  (prob : ℝ) (hprob : prob = (6 * (15 * 14 * 12 * 11) / (27 * 26 * 25 * 24))) :
  prob = (154 / 225) :=
by
  have : (15 * 14 * 12 * 11 : ℝ) = 27720 := by norm_num
  have : (27 * 26 * 25 * 24 : ℝ) = 405300 := by norm_num
  rw [hprob, ← mul_assoc],
  field_simp,
  norm_num,
  apply sorry

end probability_two_red_two_blue_l74_74514


namespace max_distance_dog_from_origin_l74_74407

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem max_distance_dog_from_origin {r : ℝ} {x_center y_center : ℝ}
    (dog_tied : distance 0 0 x_center y_center = r)
    (rope_length : r = 15)
    (center : x_center = 6 ∧ y_center = 8)
    (wall : ∀ x, x ≤ 10 → y = 0) :
  ∃ d, d = 10 :=
by
  sorry

end max_distance_dog_from_origin_l74_74407


namespace value_of_v_l74_74087

theorem value_of_v (n : ℝ) (v : ℝ) (h1 : 10 * n = v - 2 * n) (h2 : n = -4.5) : v = -9 := by
  sorry

end value_of_v_l74_74087


namespace num_subsets_of_60_elements_l74_74921

open Set

theorem num_subsets_of_60_elements :
  ∀ (S: Set ℕ), (Set.finite S ∧ S.toFinset.card = 60) → S.powerset.toFinset.card = 2^60 :=
by
  sorry

end num_subsets_of_60_elements_l74_74921


namespace find_m_l74_74662

-- Given vectors and conditions.
def vec_oa := (0 : ℝ, 1 : ℝ)
def vec_ob := (1 : ℝ, 3 : ℝ)
def vec_oc (m : ℝ) := (m, m)
def vec_ab := (vec_ob.1 - vec_oa.1, vec_ob.2 - vec_oa.2)
def vec_ac (m : ℝ) := (vec_oc m).1 - vec_oa.1, (vec_oc m).2 - vec_oa.2

-- The proposition to prove.
theorem find_m (m : ℝ) (h : vec_ab = (1, 2)) (h_parallel : ∃ λ : ℝ, λ • vec_ac m = vec_ab) :
  m = -1 :=
by
  sorry

end find_m_l74_74662


namespace smallest_odd_number_with_five_prime_factors_excluding_11_l74_74476

theorem smallest_odd_number_with_five_prime_factors_excluding_11 :
  ∃ n, n % 2 = 1 ∧
  (∃ p1 p2 p3 p4 p5 : ℕ,
    nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧
    nat.prime p4 ∧ nat.prime p5 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧
    p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5 ∧
    11 ≠ p1 ∧ 11 ≠ p2 ∧ 11 ≠ p3 ∧ 11 ≠ p4 ∧ 11 ≠ p5 ∧
    n = p1 * p2 * p3 * p4 * p5) ∧
  ∀ m, m % 2 = 1 → 
  (∃ q1 q2 q3 q4 q5 : ℕ,
    nat.prime q1 ∧ nat.prime q2 ∧ nat.prime q3 ∧
    nat.prime q4 ∧ nat.prime q5 ∧
    q1 ≠ q2 ∧ q1 ≠ q3 ∧ q1 ≠ q4 ∧ q1 ≠ q5 ∧ q2 ≠ q3 ∧ q2 ≠ q4 ∧ 
    q2 ≠ q5 ∧ q3 ≠ q4 ∧ q3 ≠ q5 ∧ q4 ≠ q5 ∧
    11 ≠ q1 ∧ 11 ≠ q2 ∧ 11 ≠ q3 ∧ 11 ≠ q4 ∧ 11 ≠ q5 ∧
    m = q1 * q2 * q3 * q4 * q5) → m ≥ 23205 :=
sorry

end smallest_odd_number_with_five_prime_factors_excluding_11_l74_74476


namespace train_speed_in_kmph_l74_74866

def train_length : ℕ := 125
def time_to_cross_pole : ℕ := 9
def conversion_factor : ℚ := 18 / 5

theorem train_speed_in_kmph
  (d : ℕ := train_length)
  (t : ℕ := time_to_cross_pole)
  (cf : ℚ := conversion_factor) :
  d / t * cf = 50 := 
sorry

end train_speed_in_kmph_l74_74866


namespace papaya_tree_height_after_5_years_l74_74160

def first_year_growth := 2
def second_year_growth := first_year_growth + (first_year_growth / 2)
def third_year_growth := second_year_growth + (second_year_growth / 2)
def fourth_year_growth := third_year_growth * 2
def fifth_year_growth := fourth_year_growth / 2

theorem papaya_tree_height_after_5_years : 
  first_year_growth + second_year_growth + third_year_growth + fourth_year_growth + fifth_year_growth = 23 :=
by
  sorry

end papaya_tree_height_after_5_years_l74_74160


namespace team_A_more_points_than_team_B_l74_74423

def binomial_probability := (n k : ℕ) → (p : ℝ) → ℝ
def no_ties : Prop := true
def independent_outcomes : Prop := true

theorem team_A_more_points_than_team_B : 
  let team_A_wins_first_game := true in
  let outcomes_independent := independent_outcomes in
  let each_game_has_no_ties := no_ties in
  let probability_team_A_more_points := (319 : ℝ) / 512 in
  probability_team_A_more_points = 319 / 512 :=
sorry

end team_A_more_points_than_team_B_l74_74423


namespace intersection_points_locus_of_centers_l74_74136

noncomputable def cubic_discriminant (a : ℝ) : ℝ :=
  4 * a^3 - 27

theorem intersection_points (a : ℝ) (h : a > 3 * Real.sqrt (3) / 2) :
  ∃ x y z : ℝ, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ (x^3 - a * x - 1 = 0) ∧ (y^3 - a * y - 1 = 0) ∧ (z^3 - a * z - 1 = 0) :=
sorry

theorem locus_of_centers (a : ℝ) (h : a > 3 * Real.sqrt (3) / 2) :
  let k := (1 - a) / 2 in
  (∀ p : ℝ × ℝ, (∃ x y z : ℝ, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ (x^3 - a * x - 1 = 0) ∧ (y^3 - a * y - 1 = 0) ∧ (z^3 - a * z - 1 = 0) ∧ p = (1/2, k)) → p.1 = 1/2 ∧ p.2 < (2 - 3 * Real.sqrt (3)) / 4) :=
sorry

end intersection_points_locus_of_centers_l74_74136


namespace proof_for_f_g_4_l74_74721

noncomputable def f (x : ℚ) : ℚ := 3 * real.sqrt x + 15 / real.sqrt x + x
noncomputable def g (x : ℚ) : ℚ := 2 * x ^ 2 - 2 * x - 3

theorem proof_for_f_g_4 :
  f (g 4) = (78 + 441) / real.sqrt 21 :=
by
  sorry

end proof_for_f_g_4_l74_74721


namespace cos_beta_value_l74_74639

theorem cos_beta_value (α β : ℝ) (hα1 : 0 < α ∧ α < π/2) (hβ1 : 0 < β ∧ β < π/2) 
  (h1 : Real.sin α = 4/5) (h2 : Real.cos (α + β) = -12/13) : 
  Real.cos β = -16/65 := 
by 
  sorry

end cos_beta_value_l74_74639


namespace total_revenue_l74_74171

-- Definitions based on the conditions
def ticket_price : ℕ := 25
def first_show_tickets : ℕ := 200
def second_show_tickets : ℕ := 3 * first_show_tickets

-- Statement to prove the problem
theorem total_revenue : (first_show_tickets * ticket_price + second_show_tickets * ticket_price) = 20000 :=
by
  sorry

end total_revenue_l74_74171


namespace remainder_of_a_coprime_implies_multiple_l74_74626

variable (a b c a' b' c' : ℤ)
variable (λ : ℤ)

-- Given conditions
def proportion := (a = λ * a') ∧ (b = λ * b') ∧ (c = λ * c')

-- First Part
theorem remainder_of_a'_by_b' (h1 : proportion) (h2 : a % b = c) : a' % b' = c' :=
by
  sorry

-- Second Part
theorem coprime_implies_multiple (h1 : proportion) (h2 : Nat.gcd a b = 1) : c' ∣ c :=
by
  sorry

end remainder_of_a_coprime_implies_multiple_l74_74626


namespace min_value_expression_is_4_l74_74815

noncomputable def min_value_expression (x : ℝ) : ℝ :=
(3 * x^2 + 6 * x + 5) / (0.5 * x^2 + x + 1)

theorem min_value_expression_is_4 : ∃ x : ℝ, min_value_expression x = 4 :=
sorry

end min_value_expression_is_4_l74_74815


namespace trajectory_equation_l74_74296

def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

noncomputable def P (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def MN_len : ℝ := real.sqrt ((2 - (-2))^2 + (0 - 0)^2)
noncomputable def MP_len (x y : ℝ) : ℝ := real.sqrt ((x + 2)^2 + y^2)
noncomputable def NP (x y : ℝ) : ℝ × ℝ := (x - 2, y)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem trajectory_equation (x y : ℝ) (h : MN_len * MP_len x y + dot_product (2 - (-2), 0 - 0) (NP x y) = 0) :
  y^2 = -8 * x :=
sorry

end trajectory_equation_l74_74296


namespace range_of_ab_c2_l74_74950

theorem range_of_ab_c2 (a b c : ℝ) (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) (h4 : -2 < c) (h5 : c < -1) :
    0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
sorry

end range_of_ab_c2_l74_74950


namespace sum_of_sequence_l74_74644

theorem sum_of_sequence (n : ℕ) (h : n ≥ 2) 
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, n ≥ 2 → S n * S (n-1) + a n = 0) :
  S n = 2 / (2 * n - 1) := by
  sorry

end sum_of_sequence_l74_74644


namespace radius_squared_l74_74617

-- Definitions of the conditions
def point_A := (2, -1)
def line_l1 (x y : ℝ) := x + y = 1
def line_l2 (x y : ℝ) := 2 * x + y = 0

-- Circle with center (h, k) and radius r
def circle_equation (h k r x y : ℝ) := (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

-- Prove statement: r^2 = 2 given the conditions
theorem radius_squared (h k r : ℝ) 
  (H1 : circle_equation h k r 2 (-1))
  (H2 : line_l1 h k)
  (H3 : line_l2 h k):
  r ^ 2 = 2 := sorry

end radius_squared_l74_74617


namespace find_sets_l74_74231

theorem find_sets (a b c d : ℕ) (h₁ : 1 < a) (h₂ : a < b) (h₃ : b < c) (h₄ : c < d)
  (h₅ : (abcd - 1) % ((a-1) * (b-1) * (c-1) * (d-1)) = 0) :
  (a = 3 ∧ b = 5 ∧ c = 17 ∧ d = 255) ∨ (a = 2 ∧ b = 4 ∧ c = 10 ∧ d = 80) :=
by
  sorry

end find_sets_l74_74231


namespace prism_edge_length_l74_74542

theorem prism_edge_length :
  ∃ (a : ℝ), (a = 7 / Real.sqrt 97 ∨ a = Real.sqrt 6 / 2) ∧
  ∀ K M L : Vector3 ℝ, 
  ∀ (ABC A₁B₁C₁ : RegularTriangularPrism ℝ),
  ∀ (hK : K ∈ segment ABC.A ABC.B),
  ∀ (hM : M ∈ line B₁C₁),
  ∀ (hL : L ∈ plane ACC₁A₁),
  ∀ (hKL : ∠(KL, plane ABC) = ∠(KL, plane ABB₁A₁)),
  ∀ (hLM : ∠(LM, plane BCC₁B₁) = ∠(LM, plane ACC₁A₁)),
  ∀ (hKM : ∠(KM, plane BCC₁B₁) = ∠(KM, plane ACC₁A₁)),
  (|KL| = 1 ∧ |KM| = 1) :=
by {
    sorry
}

end prism_edge_length_l74_74542


namespace part1_part2_l74_74990

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (h : m > 1) : ∃ x : ℝ, f x = (4 / (m - 1)) + m :=
by
  sorry

end part1_part2_l74_74990


namespace num_ways_to_place_balls_in_boxes_l74_74558

theorem num_ways_to_place_balls_in_boxes (num_balls num_boxes : ℕ) (hB : num_balls = 4) (hX : num_boxes = 3) : 
  (num_boxes ^ num_balls) = 81 := by
  rw [hB, hX]
  sorry

end num_ways_to_place_balls_in_boxes_l74_74558


namespace nails_per_large_plank_l74_74943

theorem nails_per_large_plank (x : ℕ) (total_large_planks : ℕ := 13) (additional_nails : ℕ := 8) (total_nails : ℕ := 229) :
  13 * x + 8 = 229 → x = 17 :=
by {
  intro h,
  -- Given h : 13 * x + 8 = 229,
  -- using the provided conditions and algebraic manipulation:
  -- subtract 8 from both sides
  have h1 : 13 * x = 221 := by linarith,
  -- divide both sides by 13
  have h2 : x = 17 := by linarith,
  exact h2,
}

end nails_per_large_plank_l74_74943


namespace complex_number_corresponding_to_vector_AB_is_minus_sqrt3i_l74_74694

def ω : ℂ := -1/2 + (Real.sqrt 3 / 2) * Complex.I
def ω_squared : ℂ := -1/2 - (Real.sqrt 3 / 2) * Complex.I

def OA : ℂ := ω
def OB : ℂ := ω_squared
def AB : ℂ := OB - OA

theorem complex_number_corresponding_to_vector_AB_is_minus_sqrt3i : AB = -Real.sqrt 3 * Complex.I := 
by
  sorry

end complex_number_corresponding_to_vector_AB_is_minus_sqrt3i_l74_74694


namespace days_before_reinforcement_l74_74852

/-- A garrison of 2000 men originally has provisions for 62 days.
    After some days, a reinforcement of 2700 men arrives.
    The provisions are found to last for only 20 days more after the reinforcement arrives.
    Prove that the number of days passed before the reinforcement arrived is 15. -/
theorem days_before_reinforcement 
  (x : ℕ) 
  (num_men_orig : ℕ := 2000) 
  (num_men_reinf : ℕ := 2700) 
  (days_orig : ℕ := 62) 
  (days_after_reinf : ℕ := 20) 
  (total_provisions : ℕ := num_men_orig * days_orig)
  (remaining_provisions : ℕ := num_men_orig * (days_orig - x))
  (consumption_after_reinf : ℕ := (num_men_orig + num_men_reinf) * days_after_reinf) 
  (provisions_eq : remaining_provisions = consumption_after_reinf) : 
  x = 15 := 
by 
  sorry

end days_before_reinforcement_l74_74852


namespace distance_rowed_downstream_l74_74840

noncomputable def speed_of_boat := 14
noncomputable def time_downstream := 10
noncomputable def distance_upstream := 96
noncomputable def time_upstream := 12

def speed_of_river (x : ℝ) := x
def effective_speed_downstream (x : ℝ) := speed_of_boat + speed_of_river x
def effective_speed_upstream (x : ℝ) := speed_of_boat - speed_of_river x

def distance_downstream (x : ℝ) := effective_speed_downstream x * time_downstream

theorem distance_rowed_downstream (x : ℝ) : x = 6 → distance_downstream x = 200 :=
by 
  intro h
  have hx : x = 6 := h
  have dist := (effective_speed_downstream x) * time_downstream
  rw [hx]
  rw [←hx]
  have calc_eff_speed : effective_speed_downstream 6 = 14 + 6 := rfl
  rw [calc_eff_speed]
  norm_num
  exact rfl

end distance_rowed_downstream_l74_74840


namespace Peter_work_rate_l74_74035

theorem Peter_work_rate:
  ∀ (m p j : ℝ),
    (m + p + j) * 20 = 1 →
    (m + p + j) * 10 = 0.5 →
    (p + j) * 10 = 0.5 →
    j * 15 = 0.5 →
    p * 60 = 1 :=
by
  intros m p j h1 h2 h3 h4
  sorry

end Peter_work_rate_l74_74035


namespace tetrahedrons_equal_volume_l74_74093

structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

structure Tetrahedron :=
  (A B C D : Point)

def parallel_lines (p1 p2 p3 p4 : Point) : Prop := sorry
def lies_in_plane (p1 p2 p3 p : Point) : Prop := sorry
def volume (t : Tetrahedron) : ℝ := sorry

theorem tetrahedrons_equal_volume
  (A B C D A' B' C' D' : Point)
  (T1 : Tetrahedron := ⟨A, B, C, D⟩)
  (T2 : Tetrahedron := ⟨A', B', C', D'⟩)
  (h_parallel : parallel_lines A A' B B' ∧ parallel_lines B B' C C' ∧ parallel_lines C C' D D' ∧ parallel_lines D D')
  (h_faces_nonintersect : ¬ lies_in_plane A' B' C' A ∧ ¬ lies_in_plane A' B' C' B ∧ ¬ lies_in_plane A' B' C' C)
  (h_D_in_plane : lies_in_plane A' B' C' D)
  (h_D'_in_plane : lies_in_plane A B C D') :
  volume T1 = volume T2 :=
sorry

end tetrahedrons_equal_volume_l74_74093


namespace constant_term_of_expansion_l74_74594

noncomputable def constant_term_in_expansion (f g : Polynomial ℤ) : ℤ :=
  sorry -- Placeholder for the function to compute the constant term

theorem constant_term_of_expansion :
  constant_term_in_expansion (Polynomial.of_list [(x^2, 1), (3, 0)]) (Polynomial.of_list [(1/x^2, 1), (-1, 0)] ^ 5) = 2 :=
sorry

end constant_term_of_expansion_l74_74594


namespace graph_of_f_1_minus_x_l74_74286

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x else log x / log (1 / 2)

def g (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^(1 - x) else log (1 - x) / log (1 / 2)

theorem graph_of_f_1_minus_x : ∀ x, g x = (f (1 - x)) := 
by
  sorry

end graph_of_f_1_minus_x_l74_74286


namespace product_of_two_numbers_l74_74460

theorem product_of_two_numbers (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h_sum : a + b = 210) (h_lcm : Nat.lcm a b = 1547) : a * b = 10829 :=
by
  sorry

end product_of_two_numbers_l74_74460


namespace tim_stacked_bales_today_l74_74100

theorem tim_stacked_bales_today (initial_bales : ℕ) (current_bales : ℕ) (initial_eq : initial_bales = 54) (current_eq : current_bales = 82) : 
  current_bales - initial_bales = 28 :=
by
  -- conditions
  have h1 : initial_bales = 54 := initial_eq
  have h2 : current_bales = 82 := current_eq
  sorry

end tim_stacked_bales_today_l74_74100


namespace angle_TUV_degrees_l74_74354

-- Given parameters
variables (l m : Line)
variables (U V T : Point)
variables (angle_UTV : ℝ)
variables (parallel_l_m : l ∥ m)
variables (perpendicular_UV_l : ∃ P, P ∈ UV ∧ UV ⊥ l)
variables (triangle_angles_sum_UTV : ∀ (A B C : Point), A ≠ B ∧ B ≠ C ∧ C ≠ A → A = U ∧ B = T ∧ C = V → ∠A + ∠B + ∠C = 180)
variables (angle_UTV_val : angle_UTV = 110)

-- Conclusion statement for Lean proof
theorem angle_TUV_degrees :
  ∠TUV = 20 := 
sorry

end angle_TUV_degrees_l74_74354


namespace length_of_portion_of_XY_in_cube_l74_74609

def point := (ℝ × ℝ × ℝ)

def cube (origin : point) (edge : ℝ) : Set point :=
  { p : point | ∃ x y z, origin = (x, y, z) ∧ 0 ≤ p.1 - x ∧ p.1 - x ≤ edge ∧ 0 ≤ p.2 - y ∧ p.2 - y ≤ edge ∧ 0 ≤ p.3 - z ∧ p.3 - z ≤ edge }

def line_segment_length (X Y : point) : ℝ :=
  real.sqrt ((Y.1 - X.1) ^ 2 + (Y.2 - X.2) ^ 2 + (Y.3 - X.3) ^ 2)

def intersection_length (line_start line_end cube_origin : point) (edge : ℝ) : ℝ :=
  let cube := cube cube_origin edge in
  let L := line_segment_length (0,0,0) (min line_end.1 edge - cube_origin.1, min line_end.2 edge - cube_origin.2, min line_end.3 edge - cube_origin.3) in
  line_segment_length cube_origin (cube_origin.1 + L, cube_origin.2 + L, cube_origin.3 + L)

theorem length_of_portion_of_XY_in_cube :
  let X : point := (0, 0, 0)
  let Y : point := (5, 5, 12)
  let cube_origin : point := (0, 0, 3)
  let edge : ℝ := 4 in
  intersection_length X Y cube_origin edge = 4 * real.sqrt 3 :=
by
  sorry

end length_of_portion_of_XY_in_cube_l74_74609


namespace smallest_number_among_given_values_l74_74189

theorem smallest_number_among_given_values : 
  ∃ (x : ℝ), x ∈ ({0, (- (1/3))^2, -π, -2}: set ℝ) ∧ ∀ y ∈ ({0, (- (1/3))^2, -π, -2}: set ℝ), x <= y ∧ x = -π :=
by
  sorry

end smallest_number_among_given_values_l74_74189


namespace percent_germination_second_plot_l74_74942

-- Conditions
def plot1_seeds : ℕ := 500
def plot2_seeds : ℕ := 200
def plot1_germination_rate : ℝ := 0.30
def total_germination_rate : ℝ := 0.35714285714285715

-- Number of seeds that germinated in each plot
def plot1_germinated_seeds : ℕ := plot1_seeds * plot1_germination_rate := 150
def total_seeds : ℕ := plot1_seeds + plot2_seeds := 700
def total_germinated_seeds : ℕ := total_seeds * total_germination_rate := 250
def plot2_germinated_seeds : ℕ := total_germinated_seeds - plot1_germinated_seeds := 100
def plot2_germination_rate := (plot2_germinated_seeds : ℝ) / (plot2_seeds : ℝ) := 0.50

-- Theorem to prove
theorem percent_germination_second_plot : plot2_germination_rate * 100 = 50 := 
by
  sorry

end percent_germination_second_plot_l74_74942


namespace earnings_calculation_l74_74032

def tommy_earnings : ℕ := 15
def lisa_earnings : ℕ := tommy_earnings + 15
def total_earnings : ℕ := 2 * lisa_earnings

theorem earnings_calculation
  (T : ℕ) (L : ℕ)
  (h1 : L = T + 15)
  (h2 : T = 15) :
  total_earnings = 60 := by 
  rw [h2, h1]
  sorry

end earnings_calculation_l74_74032


namespace february_saving_l74_74048

-- Definitions for the conditions
variable {F D : ℝ}

-- Condition 1: Saving in January
def january_saving : ℝ := 2

-- Condition 2: Saving in March
def march_saving : ℝ := 8

-- Condition 3: Total savings after 6 months
def total_savings : ℝ := 126

-- Condition 4: Savings increase by a fixed amount D each month
def fixed_increase : ℝ := D

-- Condition 5: Difference between savings in March and January
def difference_jan_mar : ℝ := 8 - 2

-- The main theorem to prove: Robi saved 50 in February
theorem february_saving : F = 50 :=
by
  -- The required proof is omitted
  sorry

end february_saving_l74_74048


namespace three_digit_number_possibilities_l74_74383

theorem three_digit_number_possibilities (A B C : ℕ) (hA : A ≠ 0) (hC : C ≠ 0) (h_diff : A - C = 5) :
  ∃ (x : ℕ), x = 100 * A + 10 * B + C ∧ (x - (100 * C + 10 * B + A) = 495) ∧ ∃ n, n = 40 :=
by
  sorry

end three_digit_number_possibilities_l74_74383


namespace surface_area_of_sphere_l74_74434

theorem surface_area_of_sphere (a : ℝ) (h1 : (π * (√3 * a)^2) = 4 * π) (h2 : true) : 4 * π * (2 * a)^2 = 64 * π / 3 :=
by
  -- placeholder for the proof
  sorry

end surface_area_of_sphere_l74_74434


namespace factor_of_increase_l74_74540

-- Define the conditions
def interest_rate : ℝ := 0.25
def time_period : ℕ := 4

-- Define the principal amount as a variable
variable (P : ℝ)

-- Define the simple interest formula
def simple_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ := P * R * (T : ℝ)

-- Define the total amount function
def total_amount (P : ℝ) (SI : ℝ) : ℝ := P + SI

-- The theorem that we need to prove: The factor by which the sum of money increases is 2
theorem factor_of_increase :
  total_amount P (simple_interest P interest_rate time_period) = 2 * P := by
  sorry

end factor_of_increase_l74_74540


namespace PascalTriangle_Estimate_l74_74411

noncomputable def a : ℕ → ℕ → ℤ
| 0, 0 => 1
| n, 0 => 1
| n, k => if k = n then 1 else a (n-1) k - a (n-1) (k-1)

def sum_abs_a_div_choose (n : ℕ) : ℝ :=
∑ k in Finset.range (n+1), (|a n k| : ℝ) / Nat.choose n k

theorem PascalTriangle_Estimate :
  abs (sum_abs_a_div_choose 2018 - 780.9280674537) < 1 := by
  sorry

end PascalTriangle_Estimate_l74_74411


namespace X_eq_Y_almost_surely_l74_74745

open MeasureTheory

variables {Ω : Type*} {F : measurable_space Ω} {P : measure Ω}
variables {X Y : Ω → ℝ}

-- Assuming X, Y ∈ L²
def is_L2 (f : Ω → ℝ) : Prop := mem_ℒp f 2 P

-- Given conditions
axiom X_Y_in_L2 : is_L2 X ∧ is_L2 Y
axiom E_X_given_Y : condexp_L2 F X = Y
axiom E_Y_given_X : condexp_L2 F Y = X

-- Proof statement
theorem X_eq_Y_almost_surely (hX : is_L2 X) (hY : is_L2 Y)
  (hE_X_given_Y : condexp_L2 F X = Y) (hE_Y_given_X : condexp_L2 F Y = X) :
  X =ᵐ[P] Y :=
sorry

end X_eq_Y_almost_surely_l74_74745


namespace fraction_problem_l74_74924

noncomputable def zero_point_one_five : ℚ := 5 / 33
noncomputable def two_point_four_zero_three : ℚ := 2401 / 999

theorem fraction_problem :
  (zero_point_one_five / two_point_four_zero_three) = (4995 / 79233) :=
by
  sorry

end fraction_problem_l74_74924


namespace car_cost_per_month_l74_74611

theorem car_cost_per_month
  (rental_cost : ℕ)         -- Renting cost per month
  (new_car_period : ℕ)      -- Period paying for new car
  (cost_difference : ℤ)     -- Difference in cost over the period
  (new_car_cost : ℕ) :      -- Cost of new car per month
  rental_cost = 20 →
  new_car_period = 12 →
  cost_difference = 120 →
  new_car_cost = 30 :=
begin
  sorry
end

end car_cost_per_month_l74_74611


namespace projection_of_a_on_b_l74_74982

-- Definitions for the vectors a and b
def a : ℝ × ℝ := (4, 3)
def b : ℝ × ℝ := (5, -12)

-- Function to compute the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to compute the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Statement of the theorem
theorem projection_of_a_on_b :
  dot_product a b / magnitude b = -16 / 13 :=
by
  -- Proof content will go here
  sorry

end projection_of_a_on_b_l74_74982


namespace eccentricity_of_ellipse_tangent_line_slope_l74_74723

variables (a b c : ℝ) (h1 : a > b > 0) (h2 : |sqrt(3) / 2 * 2 * c| = |sqrt(3) / 2 * 2 * c|)
include h1 h2

-- Statement for the first question
theorem eccentricity_of_ellipse (ha2b2 : a^2 + b^2 = 3 * c^2) (hb : b^2 = a^2 - c^2) :
  (c^2 / a^2 = 1 / 2) ∧ (sqrt(2) / 2 ≠ 0) :=
by
  sorry

-- Statement for the second question
theorem tangent_line_slope (hx0y0 : ∃ (x0 y0 : ℝ), ((x0 + c) * c + y0 * c = 0) ∧ (x0 ≠ 0) ∧ (x0 = - 4 / 3 * c) ∧ (y0 = c / 3))
  (hx1y1 : ∃ (x1 y1 : ℝ), (x1 = (-2 / 3) * c) ∧ (y1 = (2 / 3) * c)) :
  ∃ (k : ℝ), k = 4 + sqrt 15 ∨ k = 4 - sqrt 15 :=
by
  sorry

end eccentricity_of_ellipse_tangent_line_slope_l74_74723


namespace min_students_needed_l74_74881

-- Define the four interest groups
inductive InterestGroup
| PE | MA | P | C

open InterestGroup

-- Define maximum participation condition
def maxParticipation (s : Finset InterestGroup) : Prop :=
  s.card ≤ 2

-- Prove that the minimum number of students required is 51
theorem min_students_needed (students : Finset (Finset InterestGroup)) :
  (∀ s ∈ students, maxParticipation s) →
  ∃ s ∈ students, 5 < (students.filter (λ t, t = s)).card :=
  sorry

end min_students_needed_l74_74881


namespace decreasing_condition_l74_74953

variable (m : ℝ)

def quadratic_fn (x : ℝ) : ℝ := x^2 + m * x + 1

theorem decreasing_condition (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → (deriv (quadratic_fn m) x ≤ 0)) :
    m ≤ -10 := 
by
  -- Proof omitted
  sorry

end decreasing_condition_l74_74953


namespace dot_product_value_l74_74391

variables (a b : ℝ^3)

-- Definitions based on conditions
def condition1 : Prop := (a + b).norm = 2
def condition2 : Prop := a.norm_sq + b.norm_sq = 5

-- Theorem statement
theorem dot_product_value (h1 : condition1 a b) (h2 : condition2 a b) : a ⬝ b = -1/2 :=
sorry

end dot_product_value_l74_74391


namespace product_of_two_numbers_l74_74803

-- Definitions and conditions
def HCF (a b : ℕ) : ℕ := 9
def LCM (a b : ℕ) : ℕ := 200

-- Theorem statement
theorem product_of_two_numbers (a b : ℕ) (H₁ : HCF a b = 9) (H₂ : LCM a b = 200) : a * b = 1800 :=
by
  -- Injecting HCF and LCM conditions into the problem
  sorry

end product_of_two_numbers_l74_74803


namespace probability_three_out_of_five_dice_show_greater_than_ten_l74_74550

noncomputable def binom : ℕ → ℕ → ℚ
| n, k := (nat.choose n k : ℚ)

theorem probability_three_out_of_five_dice_show_greater_than_ten :
  let dice := 5
  let probability_single_die_greater_than_ten := 1 / 2
  let combinations := binom dice 3
  in combinations * probability_single_die_greater_than_ten^3 * probability_single_die_greater_than_ten^2 = (5 / 16) := 
by {
  let dice := 5,
  let probability_single_die_greater_than_ten := 1 / 2,
  let combinations := binom dice 3,
  calc
    combinations * probability_single_die_greater_than_ten^3 * probability_single_die_greater_than_ten^2
      = 10 * (1/2)^3 * (1/2)^2 : by sorry
  ... = 10 * (1/2)^5 : by sorry
  ... = 10 * 1/32 : by sorry
  ... = 10 / 32 : by sorry
  ... = 5 / 16 : by sorry
}

end probability_three_out_of_five_dice_show_greater_than_ten_l74_74550


namespace perpendicular_tangents_add_l74_74029

open Real

noncomputable def f1 (x : ℝ): ℝ := x^2 - 2 * x + 2
noncomputable def f2 (x : ℝ) (a : ℝ) (b : ℝ): ℝ := -x^2 + a * x + b

-- Definitions of derivatives for the given functions
noncomputable def f1' (x : ℝ): ℝ := 2 * x - 2
noncomputable def f2' (x : ℝ) (a : ℝ): ℝ := -2 * x + a

theorem perpendicular_tangents_add (x0 y0 a b : ℝ)
  (h1 : y0 = f1 x0)
  (h2 : y0 = f2 x0 a b)
  (h3 : f1' x0 * f2' x0 a = -1) :
  a + b = 5 / 2 := sorry

end perpendicular_tangents_add_l74_74029


namespace BA_eq_AB_l74_74017

variable {α : Type*} [Field α]

noncomputable def matrixA : Matrix (Fin 2) (Fin 2) α := ![![a, b], ![b, c]]
noncomputable def matrixB : Matrix (Fin 2) (Fin 2) α := ![![d, e], ![e, f]]
noncomputable def matrixAB : Matrix (Fin 2) (Fin 2) α := ![![5, -3], ![-3, 2]]

theorem BA_eq_AB : 
  Symmetric matrixA ∧ Symmetric matrixB ∧
  (matrixA + matrixB = matrixA ⬝ matrixB) ∧
  (matrixA ⬝ matrixB = matrixAB) →
  (matrixB ⬝ matrixA = matrixAB) :=
by
  intros _ _ _ _
  sorry

end BA_eq_AB_l74_74017


namespace distinct_right_triangles_l74_74302

theorem distinct_right_triangles (a b c : ℕ) (h : a = Nat.sqrt 2016)
    (hyp : c^2 = a^2 + b^2) :
  (∃ a b c : ℕ, a = Nat.sqrt 2016 ∧ c^2 = a^2 + b^2 ∧ (a^2 = 2016)) → (∃ n k : ℕ, (n * k = 2016) ∧ (∀ c b : ℕ, n = c - b ∧ k = c + b) ∧ n % 2 = 0 ∧ k % 2 = 0 →
    (c ∈ ℕ) ∧ (b ∈ ℕ) ∧ count distinct (n, k) pairs = 12) := 
  by
    sorry

end distinct_right_triangles_l74_74302


namespace brass_total_l74_74088

theorem brass_total (p_cu : ℕ) (p_zn : ℕ) (m_zn : ℕ) (B : ℕ) 
  (h_ratio : p_cu = 13) 
  (h_zn_ratio : p_zn = 7) 
  (h_zn_mass : m_zn = 35) : 
  (h_brass_total :  p_zn / (p_cu + p_zn) * B = m_zn) → B = 100 :=
sorry

end brass_total_l74_74088


namespace directrix_of_parabola_l74_74289

theorem directrix_of_parabola (x y : ℝ) : 
  (x^2 = - (1/8) * y) → (y = 1/32) :=
sorry

end directrix_of_parabola_l74_74289


namespace digits_difference_l74_74115

theorem digits_difference (n m : ℕ) (h1 : n = 500) (h2 : m = 2500) :
  (nat.log2 m - nat.log2 n) = 3 :=
by
  rw [h1, h2]
  sorry

end digits_difference_l74_74115


namespace solve_fraction_eq_l74_74062

theorem solve_fraction_eq : 
  ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 := by
  intros x h_ne_zero h_eq
  sorry

end solve_fraction_eq_l74_74062


namespace vertex_of_quadratic_l74_74766

theorem vertex_of_quadratic (a h k : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = a * (x - h)^2 + k) → (h, k) = (3, 1) :=
by
  intro hf
  have hf_def : ∀ x, f x = 2 * (x - 3)^2 + 1 := hf
  sorry

end vertex_of_quadratic_l74_74766


namespace line_intersects_parabola_at_one_point_l74_74082
   
   theorem line_intersects_parabola_at_one_point (k : ℝ) :
     (∃ y : ℝ, (x = 3 * y^2 - 7 * y + 2 ∧ x = k) → x = k) ↔ k = (-25 / 12) :=
   by
     -- your proof goes here
     sorry
   
end line_intersects_parabola_at_one_point_l74_74082


namespace jinho_received_ribbon_l74_74794

theorem jinho_received_ribbon :
  ∀ (Minsu_class Jinho_class : ℕ) (total_length_in_m : ℝ) (ribbon_per_student_in_minsu_class : ℝ),
  Minsu_class = 8 →
  Jinho_class = Minsu_class + 1 →
  total_length_in_m = 3.944 →
  ribbon_per_student_in_minsu_class = 29.05 →
  let total_length_in_cm := total_length_in_m * 100 in
  let ribbon_given_to_minsu_class := ribbon_per_student_in_minsu_class * Minsu_class in
  let remaining_ribbon := total_length_in_cm - ribbon_given_to_minsu_class in
  let ribbon_per_student_in_jinho_class := remaining_ribbon / Jinho_class in
  ribbon_per_student_in_jinho_class = 18 :=
by
  intros Minsu_class Jinho_class total_length_in_m ribbon_per_student_in_minsu_class
         h1 h2 h3 h4 total_length_in_cm ribbon_given_to_minsu_class remaining_ribbon ribbon_per_student_in_jinho_class
  sorry

end jinho_received_ribbon_l74_74794


namespace ben_and_sue_answer_l74_74181

theorem ben_and_sue_answer :
  let x := 8
  let y := 3 * (x + 2)
  let z := 3 * (y - 2)
  z = 84
:= by
  let x := 8
  let y := 3 * (x + 2)
  let z := 3 * (y - 2)
  show z = 84
  sorry

end ben_and_sue_answer_l74_74181


namespace final_distribution_l74_74752

def total_sheep : ℕ := 128
def min_sheep_for_expropriation : ℕ := 64
def expropriations : ℕ := 7

theorem final_distribution (sheep : ℕ) (peasants : ℕ → ℕ) :
  (total number of sheep = 128) →
  (∀ (i : ℕ), peasants i < 128) →
  (∀ (i j : ℕ), i ≠ j → peasants i ≥ 64 ∨ peasants j ≥ 64 → peasants i + peasants j = 128) →
  (number of expropriations = 7) →
  ∃ p, ∀ i, i ≠ p → peasants i = 0 :=
begin
  sorry
end

end final_distribution_l74_74752


namespace positive_solution_exists_l74_74236

theorem positive_solution_exists (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq_x : y^3 = x) (h_eq_z : z = 4.26) : 
  sqrt[4] (4 * x + sqrt[4] (4 * x + sqrt[4] (4 * x + ... )) = sqrt[4] (x * sqrt[4] (x * sqrt[4] (x * ... )) := 
by 
  sorry

end positive_solution_exists_l74_74236


namespace circle_equation_l74_74641

-- Definitions and conditions of the problem
def tangent_to_lines (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  let (cx, cy) := center in
  let d₁ := abs (cx - cy) / Real.sqrt 2 in
  let d₂ := abs (cx - cy - 4) / Real.sqrt 2 in
  d₁ = radius ∧ d₂ = radius 

def center_on_line (center : ℝ × ℝ) : Prop :=
  let (cx, cy) := center in
  cx + cy = 0

-- Main theorem to prove
theorem circle_equation : ∃ c : ℝ × ℝ, c = (1, -1) ∧ tangent_to_lines c (Real.sqrt 2) ∧ center_on_line c :=
by
  exists (1, -1)
  split
  { rfl }
  split
  { sorry } -- proof for tangent_to_lines (1, -1) (Real.sqrt 2)
  { sorry } -- proof for center_on_line (1, -1)

end circle_equation_l74_74641


namespace count_correct_inequalities_l74_74250

variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b < 0)
variable (h3 : a + b > 0)

def inequality1 : Prop := a^2 * b < b^3
def inequality2 : Prop := 1 / a > 0 ∧ 0 > 1 / b
def inequality3 : Prop := a^3 < a * b^2
def inequality4 : Prop := a^3 > b^3

theorem count_correct_inequalities :
  (inequality1 a b h1 h2 h3 ∨ ¬inequality1 a b h1 h2 h3) ∧
  (inequality2 a b h1 h2 h3 ∨ ¬inequality2 a b h1 h2 h3) ∧
  (inequality3 a b h1 h2 h3 ∨ ¬inequality3 a b h1 h2 h3) ∧
  (inequality4 a b h1 h2 h3 ∨ ¬inequality4 a b h1 h2 h3) →
  (inequality1 a b h1 h2 h3 ∨ inequality2 a b h1 h2 h3 ∨ inequality3 a b h1 h2 h3 ∨ inequality4 a b h1 h2 h3) ∧
  (inequality1 a b h1 h2 h3 ∨ inequality2 a b h1 h2 h3 ∨ inequality3 a b h1 h2 h3) ->
  (∃ inequ1 inequ2 inequ3 inequ4, (inequ1 ∧ ¬inequ2 ∧ inequ3 ∧ ¬inequ4) = 3) :=
by
  sorry

end count_correct_inequalities_l74_74250


namespace minimum_value_x_squared_plus_y_squared_l74_74675

-- We define our main proposition in Lean
theorem minimum_value_x_squared_plus_y_squared (x y : ℝ) 
  (h : (x + 5)^2 + (y - 12)^2 = 196) : x^2 + y^2 ≥ 169 :=
sorry

end minimum_value_x_squared_plus_y_squared_l74_74675


namespace staff_duty_arrangements_l74_74301

theorem staff_duty_arrangements :
  let n := 3 in
  let days := [1, 2, 3, 4, 5] in
  let staff := ['A', 'B', 'C'] in
  let arrangements := 
    ∑ a in Finset.univ.filter (λ f : Finset (Fin 5), f.card = 2), 
       ∑ b in Finset.univ.filter (λ g : Finset (Fin 5), g.card = 2 ∧ g ≠ a), 
          1 in
  arrangements = 180 :=
by
  sorry

end staff_duty_arrangements_l74_74301


namespace show_revenue_l74_74175

variable (tickets_first_show : Nat) (tickets_cost : Nat) (multiplicator : Nat)
variable (tickets_second_show : Nat := multiplicator * tickets_first_show)
variable (total_tickets : Nat := tickets_first_show + tickets_second_show)
variable (total_revenue : Nat := total_tickets * tickets_cost)

theorem show_revenue :
    tickets_first_show = 200 ∧ tickets_cost = 25 ∧ multiplicator = 3 →
    total_revenue = 20000 := 
by
    intros h
    sorry

end show_revenue_l74_74175


namespace double_mean_value_range_l74_74575

-- Define the function f and its derivative f'
def f (x : ℝ) : ℝ := x^3 - (6/5) * x^2
def f' (x : ℝ) : ℝ := 3 * x^2 - (12/5) * x

-- Formalize the condition for double mean value function
def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧ f' x₁ = (f b - f a) / (b - a) ∧ f' x₂ = (f b - f a) / (b - a)

-- Lean statement encapsulating the problem
theorem double_mean_value_range : 
  is_double_mean_value_function f 0 t → (t ∈ set.Ioo (3/5) (6/5)) :=
by
  sorry

end double_mean_value_range_l74_74575


namespace constant_AN_BM_l74_74969

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def eccentricity (a c : ℝ) : Prop := c / a = sqrt 3 / 2

noncomputable def triangle_area (a b : ℝ) : Prop := (1 / 2) * a * b = 1

theorem constant_AN_BM (a b c x₀ y₀ : ℝ) (h₁ : 0 < b) (hb : b < a) (he : eccentricity a c) (ha : triangle_area a b)
    (hx₀y₀ : (x₀^2 / 4) + y₀^2 = 1) :
    |(2 + x₀ / (y₀ - 1))| * |(1 + 2 * y₀ / (x₀ - 2))| = 4 := 
  sorry

end constant_AN_BM_l74_74969


namespace concurrent_tangents_proof_l74_74618

noncomputable def tangents_concurrent (circle : Type) (A B X Z Y : circle) (t_A t_B t_X : line) : Prop :=
  let AB : line := diameter (circle, A, B)
  let AX : line := line (A, X)
  let BX : line := line (B, X)
  let t_A : line := tangent (circle, A)
  let t_B : line := tangent (circle, B)
  let t_X : line := tangent (circle, X)
  let Z : point := intersection (AX, t_B)
  let Y : point := intersection (BX, t_A)
  are_concurrent (YZ, t_X, AB)

theorem concurrent_tangents_proof 
  (circle : Type) (A B X Z Y : circle) (t_A t_B t_X : line)
  (diameter : Π (circle : Type) (A B : circle), line)
  (tangent : Π (circle : Type) (P : circle), line)
  (line : Π (P Q : circle), line)
  (intersection : Π (l m : line), circle)
  (are_concurrent : Π (l m n : line), Prop) :
  tangents_concurrent circle A B X Z Y t_A t_B t_X :=
begin
  sorry,
end

end concurrent_tangents_proof_l74_74618


namespace chord_length_invalid_l74_74149

-- Define the circle radius
def radius : ℝ := 5

-- Define the maximum possible chord length in terms of the diameter
def max_chord_length (r : ℝ) : ℝ := 2 * r

-- The problem statement proving that 11 cannot be a chord length given the radius is 5
theorem chord_length_invalid : ¬ (11 ≤ max_chord_length radius) :=
by {
  sorry
}

end chord_length_invalid_l74_74149


namespace sum_binary_integers_l74_74638

def binary_sum (n : ℕ) : ℕ :=
  2^(2*n-1) * Nat.choose (2*n-1) n + (2^(2*n-1) - 1) * Nat.choose (2*n-2) n

theorem sum_binary_integers (n : ℕ) (hn : 0 < n) (h15n : 15*n = m) : 
  (Σ (k : ℕ) in {k : ℕ | (Nat.bits '' k).length = 2*n ∧ (Nat.bits '' k).count 1 = n ∧ (Nat.bits '' k).head = 1}, k) = binary_sum n :=
sorry

end sum_binary_integers_l74_74638


namespace real_solution_count_l74_74934

noncomputable def num_real_solutions : ℝ → ℝ → ℝ → ℝ
| a, b, c := a * b + b * c - c * a

theorem real_solution_count :
  let eq := (λ x : ℝ, (6 * x) / (x^2 + x + 1) + (7 * x) / (x^2 - 7 * x + 1) = -1) in
  (∃ x1 x2 x3 x4 : ℝ, eq x1 ∧ eq x2 ∧ eq x3 ∧ eq x4) ∧ ¬∃ x5 : ℝ, eq x5 ∧ x5 ≠ x1 ∧ x5 ≠ x2 ∧ x5 ≠ x3 ∧ x5 ≠ x4 :=
sorry

end real_solution_count_l74_74934


namespace range_x_l74_74970

-- Function definitions for conditions stated.
variables {f : ℝ → ℝ}
variable  even_f : ∀ x : ℝ, f x = f (-x)
variable  mono_dec_f : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x
variable  f_2_zero : f 2 = 0

-- Statement of the theorem
theorem range_x {x : ℝ} (h : f (x - 1) > 0) : -1 < x ∧ x < 3 :=
by
  sorry

end range_x_l74_74970


namespace gain_percentage_is_30_l74_74553

def sellingPrice : ℕ := 195
def gain : ℕ := 45
def costPrice : ℕ := sellingPrice - gain

def gainPercentage : ℚ := (gain : ℚ) / (costPrice : ℚ) * 100

theorem gain_percentage_is_30 :
  gainPercentage = 30 := 
sorry

end gain_percentage_is_30_l74_74553


namespace original_population_correct_l74_74541

def original_population_problem :=
  let original_population := 6731
  let final_population := 4725
  let initial_disappeared := 0.10 * original_population
  let remaining_after_disappearance := original_population - initial_disappeared
  let left_after_remaining := 0.25 * remaining_after_disappearance
  let remaining_after_leaving := remaining_after_disappearance - left_after_remaining
  let disease_affected := 0.05 * original_population
  let disease_died := 0.02 * disease_affected
  let disease_migrated := 0.03 * disease_affected
  let remaining_after_disease := remaining_after_leaving - (disease_died + disease_migrated)
  let moved_to_village := 0.04 * remaining_after_disappearance
  let total_after_moving := remaining_after_disease + moved_to_village
  let births := 0.008 * original_population
  let deaths := 0.01 * original_population
  let final_population_calculated := total_after_moving + (births - deaths)
  final_population_calculated = final_population

theorem original_population_correct :
  original_population_problem ↔ True :=
by
  sorry

end original_population_correct_l74_74541


namespace max_area_of_triangle_ABP_l74_74628

noncomputable def maxAreaTriangle (A B P : ℝ × ℝ) : ℝ :=
  let base := abs (B.1 - A.1)
  let y := (A.2 - P.2)
  (1 / 2) * base * abs y

theorem max_area_of_triangle_ABP :
  let A := (-2, 0) : ℝ × ℝ
  let B := (2, 0) : ℝ × ℝ
  ∃ P : ℝ × ℝ,
    (dist P A = 2 * dist P B) →
    maxAreaTriangle A B P = 16 / 3 :=
by
  let A := (-2: ℝ, 0)
  let B := (2: ℝ, 0)
  use (10/3, 8/3)
  intro h
  sorry

end max_area_of_triangle_ABP_l74_74628


namespace slope_of_tangent_at_A_l74_74239

def f (x : ℝ) : ℝ := x^2 + 3 * x

def f' (x : ℝ) : ℝ := 2 * x + 3

theorem slope_of_tangent_at_A : f' 2 = 7 := by
  sorry

end slope_of_tangent_at_A_l74_74239


namespace no_such_functions_l74_74746

open Real

theorem no_such_functions : ¬ ∃ (f g : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + g y) - f (x^2) + g (y) - g (x) ≤ 2 * y) ∧ (∀ x : ℝ, f (x) ≥ x^2) := by
  sorry

end no_such_functions_l74_74746


namespace find_third_number_l74_74458

-- Definitions based on given conditions
def A : ℕ := 200
def C : ℕ := 100
def B : ℕ := 2 * C

-- The condition that the sum of A, B, and C is 500
def sum_condition : Prop := A + B + C = 500

-- The proof statement
theorem find_third_number : sum_condition → C = 100 := 
by
  have h1 : A = 200 := rfl
  have h2 : B = 2 * C := rfl
  have h3 : A + B + C = 500 := sorry
  sorry

end find_third_number_l74_74458


namespace AM_divides_trapezoid_ABCD_l74_74363

-- Definitions based on conditions
variable (AD BC CM : ℝ)
variable (M_on_extension_of_BC : Prop)
variable (trapezoid_ABCD : Prop)

-- Given conditions
axiom AD_eq : AD = 12
axiom BC_eq : BC = 8
axiom CM_eq : CM = 2.4
axiom M_is_on_extension_of_BC : M_on_extension_of_BC

-- Mathematical statement to be proven
theorem AM_divides_trapezoid_ABCD (trapezoid_ABCD AD BC CM) (M_on_extension_of_BC) : 
  ∃ AM, ∃ AB CD, 
  trapezoid_ABCD ↔ 
  (AD = 12 ∧ BC = 8 ∧ CM = 2.4 ∧ 
  (1/2) * (area_of_part1 = area_of_part2)) :=
sorry

end AM_divides_trapezoid_ABCD_l74_74363


namespace probability_zeros_in_ℝ_l74_74857

/-- Define the function f(x) = x^2 + mx + 1 --/
def f (m x : ℝ) : ℝ := x^2 + m * x + 1

/-- Define the condition for the discriminant to be non-negative --/
def discriminant_nonneg (m : ℝ) : Prop := m^2 - 4 ≥ 0

/-- Define the interval from which m is chosen --/
def interval : set ℝ := set.Icc (-3) 4

/-- The measure of the interval --/
def interval_length : ℝ := 4 - (-3)

/-- The measure of m values that satisfy the discriminant_nonneg condition --/
def valid_length : ℝ := (4 - 2) + (-2 - (-3))

/-- The probability that the function f(x) has zeros in ℝ --/
def probability : ℝ := valid_length / interval_length

/-- The theorem stating the desired probability --/
theorem probability_zeros_in_ℝ : probability = 3 / 7 :=
  sorry

end probability_zeros_in_ℝ_l74_74857


namespace optimal_group_sizes_l74_74300

theorem optimal_group_sizes (a : Fin 30 → ℕ)
  (h_uniq : ∀ i j, i ≠ j → a i ≠ a j)
  (h_sum : ∑ i, a i = 1989)
  (h_ans : multiset.map a (Finset.univ : Finset (Fin 30)) = {51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81}) :
  ∀ b, (∀ i j, i ≠ j → b i ≠ b j) → ∑ i, b i = 1989 → multiset.map b (Finset.univ : Finset (Fin 30)) = {51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81} → false :=
by
  sorry

end optimal_group_sizes_l74_74300


namespace john_bought_10_flags_l74_74046

-- Define the conditions given in the problem
def first_stripe_red (flag : ℕ → bool) : Prop :=
  flag 0 = true  -- Assume true means red and false means white

def half_of_remaining_red (flag : ℕ → bool) : Prop :=
  (∑ i in Finset.range 12, if flag (i + 1) then 1 else 0) = 6

def thirteen_stripes (flag : ℕ → bool) : Prop :=
  ∀ i, i >= 13 → flag i = false

-- Given the conditions, prove the number of flags John bought
theorem john_bought_10_flags (flag : ℕ → bool) (total_stripes : ℕ) :
  first_stripe_red flag ∧ half_of_remaining_red flag ∧ thirteen_stripes flag ∧ total_stripes = 70 →
  total_stripes / 7 = 10 :=
by
  intros h
  sorry

end john_bought_10_flags_l74_74046


namespace smallest_integral_value_of_y_l74_74936

theorem smallest_integral_value_of_y :
  ∃ y : ℤ, (1 / 4 : ℝ) < y / 7 ∧ y / 7 < 2 / 3 ∧ ∀ z : ℤ, (1 / 4 : ℝ) < z / 7 ∧ z / 7 < 2 / 3 → y ≤ z :=
by
  -- The statement is defined and the proof is left as "sorry" to illustrate that no solution steps are used directly.
  sorry

end smallest_integral_value_of_y_l74_74936


namespace problem_l74_74534

def g (x : ℝ) (d e f : ℝ) := d * x^2 + e * x + f

theorem problem (d e f : ℝ) (h_vertex : ∀ x : ℝ, g d e f (x + 2) = -1 * (x + 2)^2 + 5) :
  d + e + 3 * f = 14 := 
sorry

end problem_l74_74534


namespace sum_of_solutions_x_squared_eq_36_sum_of_solutions_x_squared_eq_36_sum_l74_74319

theorem sum_of_solutions_x_squared_eq_36 (x : ℝ) (hx : x^2 = 36) : x = 6 ∨ x = -6 :=
by
  have soln1 : 6^2 = 36 := by norm_num
  have soln2 : (-6)^2 = 36 := by norm_num
  cases eq_or_eq_neg_eq.mpr ⟨by exact hx, by exact soln1, by exact soln2⟩

theorem sum_of_solutions_x_squared_eq_36_sum (x : ℝ) : (x = 6 ∨ x = -6) → (∑ x in {6, -6}, x = 0) :=
by 
  sorry

end sum_of_solutions_x_squared_eq_36_sum_of_solutions_x_squared_eq_36_sum_l74_74319


namespace hyperbola_eccentricity_l74_74288

theorem hyperbola_eccentricity 
  (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  (h_focus : (c, 0))
  (h_asymptote_intersect : (a, b))
  (h_O : (0, 0))
  (h_area_OAF : (1/2) * b * c = 3 * a^2 / 16) :
  let e := c / a in e = 3 * sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_l74_74288


namespace expression_value_l74_74246

theorem expression_value (a b : ℝ) (h₁ : a - 2 * b = 0) (h₂ : b ≠ 0) : 
  ( (b / (a - b) + 1) * (a^2 - b^2) / a^2 ) = 3 / 2 := 
by 
  sorry

end expression_value_l74_74246


namespace frequency_of_2_l74_74489

def num_set := "20231222"
def total_digits := 8
def count_of_2 := 5

theorem frequency_of_2 : (count_of_2 : ℚ) / total_digits = 5 / 8 := by
  sorry

end frequency_of_2_l74_74489


namespace num_correct_statements_l74_74777

noncomputable def f (x : ℝ) : ℝ := sorry

def F (x : ℝ) : ℝ := (f x)^2 + (f (-x))^2

theorem num_correct_statements (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) (h3 : b < -a) (h4 : ∀ x, (a ≤ x ∧ x ≤ b) → f x ≠ 0) (h5 : ∀ x1 x2, (a ≤ x1 ∧ x1 ≤ b ∧ a ≤ x2 ∧ x2 ≤ b ∧ x1 < x2) → f x1 < f x2) : 
      let statements := [
        -- ① The domain is [-b, b]
        ((-b <= x ∧ x <= b) ↔ (a <= x ∧ x <= b ∨ a <= -x ∧ -x <= b)),
        -- ② F(x) is even
        (∀ x, F x = F (-x)),
        -- ③ The minimum value is 0
        (∃ x, (-b <= x ∧ x <= b) ∧ F x = 0),
        -- ④ F(x) is monotonically increasing within its domain
        (∀ x1 x2, (-b <= x1 ∧ x1 <= b ∧ -b <= x2 ∧ x2 <= b ∧ x1 < x2) → F x1 < F x2)
        ] in (num_statements statements) = 2 := sorry

end num_correct_statements_l74_74777


namespace diagonal_difference_l74_74513

def original_matrix : matrix (fin 5) (fin 5) ℕ :=
  ![
    ![1, 2, 3, 4, 5],
    ![6, 7, 8, 9, 10],
    ![11, 12, 13, 14, 15],
    ![16, 17, 18, 19, 20],
    ![21, 22, 23, 24, 25]
  ]

def modified_matrix : matrix (fin 5) (fin 5) ℕ :=
  ![
    ![1, 2, 3, 4, 5],
    ![10, 9, 8, 7, 6],
    ![11, 12, 13, 14, 15],
    ![16, 17, 18, 19, 20],
    ![25, 24, 23, 22, 21]
  ]

def main_diagonal_sum (m : matrix (fin 5) (fin 5) ℕ) : ℕ :=
  finset.sum (finset.fin_range 5) (λ i, m i i)

def anti_diagonal_sum (m : matrix (fin 5) (fin 5) ℕ) : ℕ :=
  finset.sum (finset.fin_range 5) (λ i, m i (4 - i))

theorem diagonal_difference : abs (main_diagonal_sum modified_matrix - anti_diagonal_sum modified_matrix) = 4 :=
by
  sorry

end diagonal_difference_l74_74513


namespace angle_ACB_correct_l74_74468

theorem angle_ACB_correct {A B C D E F : Type*} (h1 : AB = 3 * AC) 
  (h2 : ∃ D E, D ∈ segment AB ∧ E ∈ segment BC) 
  (h3 : ∃ x, ∠BAE = 2 * ∠ACD) 
  (h4 : isosceles △CFE (∠CFE = ∠ECF)) 
  (h5 : F = intersection AE CD) :
  ∠ACB = 108 :=
sorry

end angle_ACB_correct_l74_74468


namespace closest_point_on_line_l74_74598

theorem closest_point_on_line 
  (P : ℝ × ℝ) (x Qy : ℝ) 
  (line_eq : ∀ x, y = 2 * x + 3) 
  (Qx = -1/5)
  (Qy = 13/5) :
  ∃ Q : ℝ × ℝ, Q = (Qx, Qy) ∧ 
  ∀ R : ℝ × ℝ, 
  (R = (x, line_eq x)) → 
    dist Q P ≤ dist R P :=
sorry

end closest_point_on_line_l74_74598


namespace find_number_l74_74323

theorem find_number (x : ℝ) (h : x / 14.5 = 171) : x = 2479.5 :=
by
  sorry

end find_number_l74_74323


namespace prove_prices_and_min_A_l74_74517

-- Define the variables and conditions
variables {x y m : ℕ}

-- Define the equations and inequality as given in the problem
def eq1 : Prop := x + 2 * y = 220
def eq2 : Prop := 2 * x + 3 * y = 360
def ineq : Prop := 60 * m + 80 * (30 - m) ≤ 2300

-- Define the price of type A and type B prizes
def priceA := 60
def priceB := 80

-- Define the minimum number of type A prizes
def minA := 5

-- State the theorem
theorem prove_prices_and_min_A (h1 : eq1) (h2 : eq2) (h3 : ineq) :
  x = priceA ∧ y = priceB ∧ 5 ≤ m :=
sorry

end prove_prices_and_min_A_l74_74517


namespace problem_solution_l74_74632

variable (f : ℝ → ℝ)

theorem problem_solution (h : ∀ x : ℝ, (deriv^[2] f x) < f x) :
  (exp 2) * f (-2) < f 0 ∧ f 2017 < (exp 2017) * f 0 :=
by
  sorry

end problem_solution_l74_74632


namespace area_common_to_all_four_circles_l74_74802

noncomputable def common_area (R : ℝ) : ℝ :=
  (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6

theorem area_common_to_all_four_circles (R : ℝ) :
  ∃ (O1 O2 A B : ℝ × ℝ),
    dist O1 O2 = R ∧
    dist O1 A = R ∧
    dist O2 A = R ∧
    dist O1 B = R ∧
    dist O2 B = R ∧
    dist A B = R ∧
    common_area R = (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6 :=
by
  sorry

end area_common_to_all_four_circles_l74_74802


namespace expression_evaluation_valid_l74_74650

theorem expression_evaluation_valid (a : ℝ) (h1 : a = 4) :
  (1 + (4 / (a ^ 2 - 4))) * ((a + 2) / a) = 2 := by
  sorry

end expression_evaluation_valid_l74_74650


namespace largest_pos_integer_n_l74_74234

theorem largest_pos_integer_n (n : ℕ) :
  (∀ (a : ℤ), gcd (a, n) = 1 → a ^ 2 ≡ 1 [ZMOD n]) ↔ n = 24 := sorry

end largest_pos_integer_n_l74_74234


namespace collinear_point_min_distance_l74_74129

variable {P : Type}
variables {dist : P → P → ℝ} (A B C D E P : P)
variables (h : ∀ (x y z : P), dist x y ≥ 0 ∧ dist x y = dist y x ∧ (dist x z ≤ dist x y + dist y z))
variables (collinear : ∀ (P₁ P₂ P₃ : P), P₁ ∈ ({A, B, C, D, E} : set P) → P₂ ∈ ({A, B, C, D, E} : set P) → P₃ ∈ ({A, B, C, D, E} : set P) → ∃ (r : ℝ), dist A P₃  = r ∧ dist B P₃ = r - 1 ∧ dist C P₃ = r - 2 ∧ dist D P₃ = r - 4 ∧ dist E P₃ = r - 13)
variables (AB BC CD DE: ℝ) (H_AB : AB = 1) (H_BC : BC = 1) (H_CD : CD = 2) (H_DE : DE = 9)

theorem collinear_point_min_distance :
  (∃ (r : ℝ), r = 4) →
  (AB = 1) →
  (BC = 1) →
  (CD = 2) →
  (DE = 9) →
  (∀ (r : ℝ), r = 4 → let AP := r in let BP := r - 1 in let CP := r - 2 in let DP := r - 4 in let EP := r - 13 in AP^2 + BP^2 + CP^2 + DP^2 + EP^2) :=
begin
  existsi (4 : ℝ),
  intros hyp_AB hyp_BC hyp_CD hyp_DE h_4,
  simp [h_4, AP, BP, CP, DP, EP],
  sorry
end

end collinear_point_min_distance_l74_74129


namespace arithmetic_mean_l74_74593

variable {x b c : ℝ}

theorem arithmetic_mean (hx : x ≠ 0) (hb : b ≠ c) : 
  (1 / 2) * ((x + b) / x + (x - c) / x) = 1 + (b - c) / (2 * x) :=
by
  sorry

end arithmetic_mean_l74_74593


namespace arithmetic_sequence_max_sum_proof_l74_74269

noncomputable def arithmetic_sequence_max_sum (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_max_sum_proof (a_1 d : ℝ) 
  (h1 : 3 * a_1 + 6 * d = 9)
  (h2 : a_1 + 5 * d = -9) :
  ∃ n : ℕ, n = 3 ∧ arithmetic_sequence_max_sum a_1 d n = 21 :=
by
  sorry

end arithmetic_sequence_max_sum_proof_l74_74269


namespace subset_contains_power_or_sum_is_power_l74_74371

open Set

noncomputable def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem subset_contains_power_or_sum_is_power {A : Set ℕ} (hA₁ : A ⊆ { n | n ≤ 1997 }) (hA₂ : 1000 < A.card) :
  (∃ n ∈ A, isPowerOfTwo n) ∨ (∃ a b ∈ A, a ≠ b ∧ isPowerOfTwo (a + b)) := 
sorry

end subset_contains_power_or_sum_is_power_l74_74371


namespace angle_between_unit_vectors_l74_74975

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- We assume a and b are unit vectors and the given dot product condition
def a_unit : Prop := ∥a∥ = 1
def b_unit : Prop := ∥b∥ = 1
def dot_product_condition : Prop := (2 • a + b) ⬝ (a - 2 • b) = - (3 * Real.sqrt 3) / 2

-- We aim to prove the angle between a and b is π/6
theorem angle_between_unit_vectors (ha : a_unit) (hb : b_unit) (h : dot_product_condition) :
  Real.angle a b = π / 6 :=
by
  sorry

end angle_between_unit_vectors_l74_74975


namespace defective_rate_is_twenty_percent_l74_74267

-- Definitions of the conditions
def total_products := 10
def defective_probability (ξ : ℕ) : ℚ := if ξ = 1 then 16/45 else 0
def max_defective_rate := 0.4

-- Questions translated to Lean
def defective_items (n : ℕ) : Prop :=
  ∃ (ξ : ℕ), ξ = n ∧
  (defective_probability ξ = 16/45) ∧
  (n ≤ total_products * max_defective_rate)

-- Conclude the defective rate
theorem defective_rate_is_twenty_percent (n : ℕ) : defective_items n → n = 2 := by
  sorry

end defective_rate_is_twenty_percent_l74_74267


namespace problem1_problem2_l74_74983

noncomputable def ellipse := { p : ℝ × ℝ // (p.fst^2 / 4) + p.snd^2 = 1 }

def on_ellipse (p : ℝ × ℝ) : Prop := (p.fst^2 / 4) + p.snd^2 = 1

def perpendicular (a b c d : ℝ × ℝ) : Prop := 
  let slope_ab := (b.snd - a.snd) / (b.fst - a.fst),
      slope_ad := (d.snd - a.snd) / (d.fst - a.fst)
  in slope_ad = -1 / slope_ab

def exists_lambda (k1 k2 : ℝ) : Prop := ∃ λ : ℝ, k1 = λ * k2

theorem problem1 (A B D M N : ℝ × ℝ) (hA : A.fst ≠ 0 ∧ A.snd ≠ 0) (hB : B = (-A.fst, -A.snd))
  (hD : on_ellipse D) (hAD_perp_AB : perpendicular A B A D) 
  (k1 k2 : ℝ) (hBD : ∃ k1 : ℝ, ∀ (x : ℝ), (N = (0, -((3 / 4) * A.snd)) ∧ M = (3 * A.fst, 0)))
  (hAM : ∃ k2 : ℝ, ∀ (x : ℝ), (M = (x, 0) ∧ N = (0, -((3 / 4) * x)))) :
  exists_lambda k1 k2 := sorry

theorem problem2 (A M N : ℝ × ℝ) (hA : A.fst ≠ 0 ∧ A.snd ≠ 0) 
  (hM : M = (3 * A.fst, 0)) (hN : N = (0, -(3 / 4) * A.snd)) :
  ∃ S : ℝ, S = 9 / 8 := 
begin
  use (9 / 8),
  sorry
end

end problem1_problem2_l74_74983


namespace at_least_one_genuine_l74_74679

theorem at_least_one_genuine (batch : Finset ℕ) 
  (h_batch_size : batch.card = 12) 
  (genuine_items : Finset ℕ)
  (h_genuine_size : genuine_items.card = 10)
  (defective_items : Finset ℕ)
  (h_defective_size : defective_items.card = 2)
  (h_disjoint : genuine_items ∩ defective_items = ∅)
  (drawn_items : Finset ℕ)
  (h_draw_size : drawn_items.card = 3)
  (h_subset : drawn_items ⊆ batch)
  (h_union : genuine_items ∪ defective_items = batch) :
  (∃ (x : ℕ), x ∈ drawn_items ∧ x ∈ genuine_items) :=
sorry

end at_least_one_genuine_l74_74679


namespace max_height_of_projectile_l74_74164

noncomputable def height_function : ℝ → ℝ := λ t, -4 * t^2 + 40 * t + 25

theorem max_height_of_projectile : 
  ∃ t_max, t_max = 5 ∧ height_function t_max = 125 := 
by
  use 5
  have h_t_max : height_function 5 = 125 := by
    sorry
  exact ⟨rfl, h_t_max⟩

end max_height_of_projectile_l74_74164


namespace problem1_problem2_l74_74505

-- For Problem (1)
theorem problem1 (x : ℝ) : 2 * x - 3 > x + 1 → x > 4 := 
by sorry

-- For Problem (2)
theorem problem2 (a b : ℝ) (h : a^2 + 3 * a * b = 5) : (a + b) * (a + 2 * b) - 2 * b^2 = 5 := 
by sorry

end problem1_problem2_l74_74505


namespace ratio_of_new_to_original_area_l74_74624

-- Definitions for the conditions
variables {a b : ℝ} (h : a > 0) (h' : b > 0)
def original_area : ℝ := a * b
def new_area : ℝ := (a / 2) * (b / 2)

-- Main theorem statement
theorem ratio_of_new_to_original_area 
  (ha : a > 0) (hb : b > 0) : 
  (new_area ha hb) / (original_area ha hb) = 1 / 4 :=
by
  sorry

end ratio_of_new_to_original_area_l74_74624


namespace prop_4_prop_5_l74_74315

variables (f g : ℝ → ℝ)

-- Definitions representing monotonic functions
def is_increasing (h : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → h x ≤ h y

def is_decreasing (h : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → h x ≥ h y

-- Propositions 4 and 5 in Lean
theorem prop_4 (hf_inc : is_increasing f) (hg_dec : is_decreasing g) : is_increasing (λ x, f x - g x) := sorry

theorem prop_5 (hf_dec : is_decreasing f) (hg_inc : is_increasing g) : is_decreasing (λ x, f x - g x) := sorry

end prop_4_prop_5_l74_74315


namespace non_shaded_area_l74_74697

theorem non_shaded_area (r : ℝ) (A : ℝ) (shaded : ℝ) (non_shaded : ℝ) :
  (r = 5) ∧ (A = 4 * (π * r^2)) ∧ (shaded = 8 * (1 / 4 * π * r^2 - (1 / 2 * r * r))) ∧
  (non_shaded = A - shaded) → 
  non_shaded = 50 * π + 100 :=
by
  intro h
  obtain ⟨r_eq_5, A_eq, shaded_eq, non_shaded_eq⟩ := h
  rw [r_eq_5] at *
  sorry

end non_shaded_area_l74_74697


namespace clock_ticks_clock_ticks_at_other_time_l74_74886

theorem clock_ticks (intervals_6_ticks : 5 > 0) (time_6_ticks : ℝ) (time_other_ticks : ℝ) : ℝ :=
  let interval := time_6_ticks / intervals_6_ticks
  let intervals_other := time_other_ticks / interval
  let ticks_other := intervals_other + 1
  ticks_other

theorem clock_ticks_at_other_time (intervals_6_ticks : 5 > 0) (time_6_ticks : 30) (time_other_ticks : 42) : ticks_other = 8 :=
  by sorry

end clock_ticks_clock_ticks_at_other_time_l74_74886


namespace circle_equation_and_line_intersection_l74_74997

theorem circle_equation_and_line_intersection : 
  ∃ (D E F : ℝ), 
    triangle_ABC_has_circumcircle_eqn (0, 1) (-real.sqrt 3, 0) (real.sqrt 3, 0) (x^2 + y^2 + D*x + E*y + F = 0)
    ∧ standard_eqn_circumcircle_is (x - 0)^2 + (y + 1)^2 = 4 
    ∧ (line_through_point_intersects_circumcircle (1, 3) (⟨x=1⟩ ∨ ⟨15*x - 8*y + 9=0⟩) 
       ∧ PQ_length_is 2*real.sqrt 3) :=
sorry

end circle_equation_and_line_intersection_l74_74997


namespace work_hours_calc_l74_74510

-- Definitions according to conditions
def man_hours (n : ℕ) (h : ℕ) (d : ℕ) : ℕ := n * h * d

-- Theorem statement according to the proof problem
theorem work_hours_calc (x : ℕ) (hx : x > 10) (w1 : man_hours 10 7 18 = 1260) :
  ∃ y, y = 1260 / (12 * x) :=
by {
  use 1260 / (12 * x),
  exact w1,
  sorry
}

end work_hours_calc_l74_74510


namespace quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l74_74642

-- Part 1: Expression of the quadratic function
theorem quadratic_function_expression (a : ℝ) (h : a = 0) : 
  ∀ x, (x^2 + (a-2)*x + 3) = x^2 - 2*x + 3 :=
by sorry

-- Part 2: Range of y for 0 < x < 3
theorem quadratic_function_range (x y : ℝ) (h : ∀ x, y = x^2 - 2*x + 3) (hx : 0 < x ∧ x < 3) :
  2 ≤ y ∧ y < 6 :=
by sorry

-- Part 3: Range of m for y1 > y2
theorem quadratic_function_m_range (m y1 y2 : ℝ) (P Q : ℝ × ℝ)
  (h1 : P = (m - 1, y1)) (h2 : Q = (m, y2)) (h3 : y1 > y2) :
  m < 3 / 2 :=
by sorry

end quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l74_74642


namespace rectangle_width_eq_4_l74_74885

theorem rectangle_width_eq_4 (a b : ℝ) (hA : a = 6) (hB : b = 9) : 
  ∃ w : ℝ, w = 4 := 
by
  have h1 := square_area_eq_36 a hA
  have h2 := rectangle_area_eq_width_times_9 b hB
  have h3 := eq.symm (square_area_eq_36_eq_rectangle_area_9w h1 h2)
  sorry

-- Definitions and supplementary lemmas
def square_area_eq_36 (a : ℝ) : a = 6 → a * a = 36 := 
λ h, by rw [h, (6:ℝ) * (6:ℝ)]

def rectangle_area_eq_width_times_9 (b : ℝ) : b = 9 → ∃ w : ℝ, b * w = 36 := 
λ h, by rw [h, 9 * (36/9:ℝ)]

def square_area_eq_36_eq_rectangle_area_9w (h1 : 36 = 36) (h2 : ∃ w : ℝ, 9 * w = 36) :
  ∃ w : ℝ, w = 4 :=
Exists.intro 4 (by norm_num)

end rectangle_width_eq_4_l74_74885


namespace ratio_of_volumes_l74_74847

theorem ratio_of_volumes (s : ℝ) (π : ℝ) (hπ : Real.pi = π) :
  let r := s / 2,
  let V_cylinder := π * r^2 * s,
  let V_cube := s^3,
  V_cylinder / V_cube = π / 4 := 
by
  sorry

end ratio_of_volumes_l74_74847


namespace find_a_l74_74722

theorem find_a (a : ℝ) : let A := {-1, 1, 3}
                          let B := {a + 1, a^2 + 4}
                          A ∩ B = {3} → a = 2 :=
by
  let A := {-1 : ℝ, 1, 3}
  let B := {a + 1, a^2 + 4}
  assume h : A ∩ B = {3}
  sorry

end find_a_l74_74722


namespace sequence_general_term_l74_74993

def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = 2 * a (n - 1) + n - 2)

theorem sequence_general_term {a : ℕ → ℕ} (h : sequence a) :
  ∀ n, a n = 2^n - n :=
sorry

end sequence_general_term_l74_74993


namespace karen_savings_over_30_years_l74_74002

theorem karen_savings_over_30_years 
  (P_exp : ℕ) (L_exp : ℕ) 
  (P_cheap : ℕ) (L_cheap : ℕ) 
  (T : ℕ)
  (hP_exp : P_exp = 300)
  (hL_exp : L_exp = 15)
  (hP_cheap : P_cheap = 120)
  (hL_cheap : L_cheap = 5)
  (hT : T = 30) : 
  (P_cheap * (T / L_cheap) - P_exp * (T / L_exp)) = 120 := 
by 
  sorry

end karen_savings_over_30_years_l74_74002


namespace michael_remaining_books_l74_74335

theorem michael_remaining_books (total_books : ℕ) (read_percentage : ℚ) 
  (H1 : total_books = 210) (H2 : read_percentage = 0.60) : 
  (total_books - (read_percentage * total_books) : ℚ) = 84 :=
by
  sorry

end michael_remaining_books_l74_74335


namespace parabola_point_focus_distance_l74_74960

/-- 
  Given a point P on the parabola y^2 = 4x, and the distance from P to the line x = -2
  is 5 units, prove that the distance from P to the focus of the parabola is 4 units.
-/
theorem parabola_point_focus_distance {P : ℝ × ℝ} 
  (hP : P.2^2 = 4 * P.1) 
  (h_dist : (P.1 + 2)^2 + P.2^2 = 25) : 
  dist P (1, 0) = 4 :=
sorry

end parabola_point_focus_distance_l74_74960


namespace zero_of_f_in_interval_l74_74463

-- Define the function.
def f (x : ℝ) := Real.log x + 2 * x - 6

-- State the theorem that the zero of the function is in the interval (2, 3).
theorem zero_of_f_in_interval : ∃ x ∈ Ioo (2 : ℝ) 3, f x = 0 := sorry

end zero_of_f_in_interval_l74_74463


namespace percentage_of_class_that_are_men_l74_74337

-- Define the conditions
def percentage_women_science := 0.30
def percentage_non_science := 0.60
def percentage_men_science := 0.55

-- Definitions for the variables
def total_class := 100
def science_majors := total_class * (1 - percentage_non_science)
def percentage_women (percentage_men : ℝ) := total_class - percentage_men
def women_science_majors (percentage_men : ℝ) := percentage_women_science * percentage_women percentage_men
def men_science_majors (percentage_men : ℝ) := percentage_men_science * percentage_men

-- The proof problem statement
theorem percentage_of_class_that_are_men (percentage_men : ℝ) :
  (women_science_majors percentage_men + men_science_majors percentage_men = science_majors) → percentage_men = 40 := 
by
  intro h
  sorry

end percentage_of_class_that_are_men_l74_74337


namespace anoop_joined_after_6_months_l74_74192

/- Conditions -/
def arjun_investment : ℕ := 20000
def arjun_months : ℕ := 12
def anoop_investment : ℕ := 40000

/- Main theorem -/
theorem anoop_joined_after_6_months (x : ℕ) (h : arjun_investment * arjun_months = anoop_investment * (arjun_months - x)) : 
  x = 6 :=
sorry

end anoop_joined_after_6_months_l74_74192


namespace condition_sufficient_not_necessary_l74_74265

theorem condition_sufficient_not_necessary (x : ℝ) : (1 < x ∧ x < 2) → ((x - 2) ^ 2 < 1) ∧ ¬ ((x - 2) ^ 2 < 1 → (1 < x ∧ x < 2)) :=
by
  sorry

end condition_sufficient_not_necessary_l74_74265


namespace inequality_solution_l74_74230

variable {x : ℝ}

theorem inequality_solution (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) : 
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| 
  ∧ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔ 
  (Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4) :=
sorry

end inequality_solution_l74_74230


namespace first_group_count_l74_74070

theorem first_group_count (n : ℕ)
  (h1 : ∑ i in finset.range n, i / n = 20)
  (h2 : ∑ i in finset.range 20, i / 20 = 30)
  (h3 : ∑ i in finset.range (n + 20), i / (n + 20) = 24) : n = 30 :=
by
  sorry

end first_group_count_l74_74070


namespace fifth_term_arithmetic_sequence_is_19_l74_74693

def arithmetic_sequence_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem fifth_term_arithmetic_sequence_is_19 :
  arithmetic_sequence_nth_term 3 4 5 = 19 := 
  by
  sorry

end fifth_term_arithmetic_sequence_is_19_l74_74693


namespace angle_MXK_135_l74_74566

-- Definitions for the square, circle and points
variables {ABCD : Type*} [square ABCD]
variables {circle : Type*} [circle circle (inscribed_in ABCD)]
variables {M K L X : Type*}
variables {AB_CD_tangent M K : tangent_points circle (sides AB CD : shapes)}
variables {BK_intersects_circle : intersects BK (circle_points K L)}
variables {midpoint_X : midpoint X K L}

-- Theorem statement for the angle ∠ MXK
theorem angle_MXK_135 (h1 : tangent_points circle (side AB M))
                      (h2 : tangent_points circle (side CD K))
                      (h3 : intersects BK (circle_points K L))
                      (h4 : midpoint X K L)
                      : angle MXK = 135 :=
by
  sorry

end angle_MXK_135_l74_74566


namespace trihedral_angle_sum_is_2pi_l74_74380

-- Given definitions for the problem
structure Cylinder :=
  (center : Point)
  (radius : ℝ)
  (height : ℝ)

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def dihedral_angle (O A B C : Point) : ℝ := sorry

def angle_sum (O A B C : Point) : ℝ :=
  dihedral_angle O A B C 

theorem trihedral_angle_sum_is_2pi
  (O A B C : Point)
  (hO : O = Cylinder.center)
  (hA₁ : A.x ^ 2 + A.y ^ 2 = Cylinder.radius ^ 2)
  (hB₁ : B.x ^ 2 + B.y ^ 2 = Cylinder.radius ^ 2)
  (hC₁ : C.x ^ 2 + C.y ^ 2 = Cylinder.radius ^ 2)
  (hA₂ : A.z = Cylinder.center.z)
  (hB₂ : B.z = Cylinder.center.z)
  (hC₂ : C.z = Cylinder.center.z + Cylinder.height)
  (hAB : dist A B = 2 * Cylinder.radius) :
  angle_sum O A B C = 2 * π :=
sorry

end trihedral_angle_sum_is_2pi_l74_74380


namespace proof_problem_l74_74022

open EuclideanGeometry

noncomputable def square_side_length : ℝ := 1 -- for simplicity, let the side length be 1

variables (A B C D P M N Q : Point)
variable (K : Circle)

-- Hypotheses for the problem
hypothesis h_square : square A B C D
hypothesis h_circle : diameter_circle K A B
hypothesis h_point_P : on_side P C D
hypothesis h_intersection_MN : intersects_again (line_through A P) K M
hypothesis h_intersection_BP : intersects_again (line_through B P) K N
hypothesis h_intersection_DM : intersects (line_through D M) (line_through C N) Q

-- Statements to prove
theorem proof_problem :
  lies_on_circle Q K ∧ (ratio_eq (segment_length A Q) (segment_length Q B) (segment_length D P) (segment_length P C)) :=
by
  sorry

end proof_problem_l74_74022


namespace minimum_value_of_f_l74_74446

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem minimum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -1 / Real.exp 1 :=
by
  -- Proof to be provided
  sorry

end minimum_value_of_f_l74_74446


namespace digits_right_of_decimal_l74_74664

theorem digits_right_of_decimal (a b c : ℕ) (h : (a, b, c) = (5, 10, 8)) :
  let expr := (a^7) / ((b^5) * (c^2))
  in digits_right_of_point expr = 6 := 
by sorry

end digits_right_of_decimal_l74_74664


namespace percentage_of_english_books_in_country_is_correct_l74_74361

-- Define the total number of books
def total_books : ℕ := 2300

-- Define the percentage of English books
def percentage_english_books : ℝ := 0.80

-- Define the number of English books published outside the country
def english_books_outside : ℕ := 736

-- Calculate the total number of English books
def english_books_total : ℕ := (percentage_english_books * total_books).toInt

-- Calculate the number of English books published in the country
def english_books_in_country : ℕ := english_books_total - english_books_outside

-- Calculate the percentage of English books published in the country
def percentage_english_books_in_country : ℝ := (english_books_in_country.toFloat / english_books_total.toFloat) * 100

theorem percentage_of_english_books_in_country_is_correct :
  percentage_english_books_in_country ≈ 59.78 :=
by
  -- Proof omitted
  sorry

end percentage_of_english_books_in_country_is_correct_l74_74361


namespace shaded_area_l74_74696

variable {r OQ OP PQ : ℝ}

-- Conditions in the problem
def larger_circle_radius (OQ : ℝ) := OQ = 9
def radii_ratio (OP PQ : ℝ) := OP / PQ = 1 / 2
def radius_relation (OQ OP PQ : ℝ) := OQ = OP + PQ

-- Definition for the areas of the circles
def area (r : ℝ) : ℝ := Real.pi * r^2

-- Goal: Prove the area of the shaded region
theorem shaded_area : 
  larger_circle_radius OQ → 
  radii_ratio OP PQ → 
  radius_relation OQ OP PQ → 
  (area OQ - area OP) = 72 * Real.pi :=
by
  -- omitting the proof
  sorry

end shaded_area_l74_74696


namespace total_length_of_T_is_128_l74_74381

open Real

-- define T as the set of points satisfying the given condition
def T : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | abs (abs (abs p.1 - 3) - 1) + abs (abs (abs p.2 - 3) - 1) = 2}

-- state the theorem that the total length of all the lines that make up T is 128
theorem total_length_of_T_is_128 : 
  (∑ p in T, 8) = 128 :=
sorry

end total_length_of_T_is_128_l74_74381


namespace octahedron_cut_area_l74_74535

theorem octahedron_cut_area:
  let a := 9
  let b := 3
  let c := 8
  a + b + c = 20 :=
by
  sorry

end octahedron_cut_area_l74_74535


namespace percentage_comedies_l74_74215

theorem percentage_comedies (a : ℕ) (d c T : ℕ) 
  (h1 : d = 5 * a) 
  (h2 : c = 10 * a) 
  (h3 : T = c + d + a) : 
  (c : ℝ) / T * 100 = 62.5 := 
by 
  sorry

end percentage_comedies_l74_74215


namespace solution_1_solution_2_l74_74653

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (a + 1) * x + Real.log x

def critical_point_condition (a x : ℝ) : Prop :=
  (x = 1 / 4) → deriv (f a) x = 0

def pseudo_symmetry_point_condition (a : ℝ) (x0 : ℝ) : Prop :=
  let f' := fun x => 2 * x^2 - 5 * x + Real.log x
  let g := fun x => (4 * x0^2 - 5 * x0 + 1) / x0 * (x - x0) + 2 * x0^2 - 5 * x0 + Real.log x0
  ∀ x : ℝ, 
    (0 < x ∧ x < x0) → (f' x - g x < 0) ∧ 
    (x > x0) → (f' x - g x > 0)

theorem solution_1 (a : ℝ) (h1 : a > 0) (h2 : critical_point_condition a (1/4)) :
  a = 4 := 
sorry

theorem solution_2 (x0 : ℝ) (h1 : x0 = 1/2) :
  pseudo_symmetry_point_condition 4 x0 :=
sorry


end solution_1_solution_2_l74_74653


namespace Sunzi_problem_correctness_l74_74350

theorem Sunzi_problem_correctness (x y : ℕ) :
  3 * (x - 2) = 2 * x + 9 ∧ (y / 3) + 2 = (y - 9) / 2 :=
by
  sorry

end Sunzi_problem_correctness_l74_74350


namespace monotonic_implies_m_l74_74326

noncomputable def cubic_function (x m : ℝ) : ℝ := x^3 + x^2 + m * x + 1

theorem monotonic_implies_m (m : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + m) ≥ 0) → m ≥ 1 / 3 :=
  sorry

end monotonic_implies_m_l74_74326


namespace black_less_equal_three_white_l74_74162

-- Definitions based on the conditions
variables (Polygon : Type) [fintype Polygon] (triangles : set (set Polygon))
variables (black_triangles white_triangles : set (set Polygon))
variables (B W : ℕ)

-- Conditions: Polygon divided into triangles , triangles are colored black and white such that any two triangles sharing a side are colored differently
def polygon_divided : Prop := 
  ∀ T ∈ triangles, 
    (T ⊆ Polygon) ∧ (∃ (s : set (set Polygon)), s = black_triangles ∪ white_triangles ∧ s = triangles)

def coloring_condition : Prop :=
   ∀ t₁ t₂ ∈ triangles, (t₁ ≠ t₂ ∧ t₁ ∩ t₂ ≠ ∅) → 
     ∃ (b ∈ black_triangles), ∃ (w ∈ white_triangles), ((t₁ = b ∧ t₂ = w) ∨ (t₁ = w ∧ t₂ = b))

-- The proof statement given conditions
theorem black_less_equal_three_white :
  polygon_divided Polygon triangles →
  coloring_condition triangles black_triangles white_triangles →
  B = black_triangles.card →
  W = white_triangles.card →
  B ≤ 3 * W :=
begin
  intros h_poly h_coloring hB hW,
  sorry
end

end black_less_equal_three_white_l74_74162


namespace petya_vasya_cubic_roots_diff_2014_l74_74410

theorem petya_vasya_cubic_roots_diff_2014 :
  ∀ (p q r : ℚ), ∃ (x1 x2 x3 : ℚ), x1 ≠ 0 ∧ (x1 - x2 = 2014 ∨ x1 - x3 = 2014 ∨ x2 - x3 = 2014) :=
sorry

end petya_vasya_cubic_roots_diff_2014_l74_74410


namespace prop1_prop4_prop5_l74_74276

-- Proposition 1: If a^2 + b^2 = 0, then a = 0 and b = 0
theorem prop1 {a b : Vector} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

-- Proposition 4: Given a, b, c are three non-zero vectors, if a + b = 0, then |a • c| = |b • c|
theorem prop4 {a b c : Vector} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b = 0) : |a • c| = |b • c| := by
  sorry

-- Proposition 5: Given λ₁ > 0, λ₂ > 0, e₁, e₂ is a basis, a = λ₁ e₁ + λ₂ e₂, then a is not collinear with e₁ and a is not collinear with e₂
theorem prop5 {λ₁ λ₂ : Real} {e₁ e₂ : Vector} (hλ₁ : λ₁ > 0) (hλ₂ : λ₂ > 0) (hbasis : is_basis [e₁, e₂]) (h : a = λ₁ • e₁ + λ₂ • e₂) : ¬ collinear a e₁ ∧ ¬ collinear a e₂ := by
  sorry

end prop1_prop4_prop5_l74_74276


namespace power_of_power_l74_74818

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end power_of_power_l74_74818


namespace gcd_max_of_sum_1980_l74_74461

theorem gcd_max_of_sum_1980 (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1980) : 
  ∃ (d : ℕ), d = Nat.gcd a b ∧ d ∣ 1980 ∧ ∀ e, (e ∣ 1980 → e = Nat.gcd a b → e ≤ 990) :=
begin
  have h_gcd_div_sum : Nat.gcd a b ∣ 1980,
  { rw h_sum,
    apply Nat.gcd_dvd_add_self },
  sorry
end

end gcd_max_of_sum_1980_l74_74461


namespace part1_part2_l74_74987
open Real

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (hm : m > 1) : ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by
  sorry

end part1_part2_l74_74987


namespace find_x_l74_74341

variables (x : ℝ)

def A := (0 : ℝ, -3 : ℝ)
def B := (3 : ℝ,  3 : ℝ)
def C := (x, -1 : ℝ)

def vector_AB := (B.1 - A.1, B.2 - A.2)
def vector_BC := (C.1 - B.1, C.2 - B.2)

theorem find_x (h : vector_AB = -3/2 • vector_BC) : x = 1 :=
by 
  have h1 : vector_AB = (3, 6) := rfl
  have h2 : vector_BC = (x - 3, -4) := rfl
  calc
    x - 3    = (2 / -3) * 6               : sorry -- k calculated and rewritten here
    x        = 1                          : sorry -- solving the above for x

end find_x_l74_74341


namespace train_cross_time_l74_74122

theorem train_cross_time (train_length bridge_length : ℕ) (train_speed_km_hr : ℕ) :
  train_length = 100 →
  bridge_length = 135 →
  train_speed_km_hr = 75 →
  (235 / (75 * (1000 / 3600) : ℕ) : ℝ) ≈ 11.28 :=
by
  intros h1 h2 h3
  sorry

end train_cross_time_l74_74122


namespace arc_length_of_circle_l74_74150

section circle_arc_length

def diameter (d : ℝ) : Prop := d = 4
def central_angle_deg (θ_d : ℝ) : Prop := θ_d = 36

theorem arc_length_of_circle
  (d : ℝ) (θ_d : ℝ) (r : ℝ := d / 2) (θ : ℝ := θ_d * (π / 180)) (l : ℝ := θ * r) :
  diameter d → central_angle_deg θ_d → l = 2 * π / 5 :=
by
  intros h1 h2
  sorry

end circle_arc_length

end arc_length_of_circle_l74_74150


namespace train_passing_time_l74_74825

theorem train_passing_time (length_of_train : ℝ) (speed_of_train_kmhr : ℝ) :
  length_of_train = 180 → speed_of_train_kmhr = 36 → (length_of_train / (speed_of_train_kmhr * (1000 / 3600))) = 18 :=
by
  intro h1 h2
  sorry

end train_passing_time_l74_74825


namespace sequence_bounded_proof_l74_74025

variable {α : Type*}
noncomputable def sequence_bounded (a : ℕ → ℝ) : Prop :=
  ∃ M ≥ 0, ∀ n, a n ≤ M

theorem sequence_bounded_proof (c : ℝ) (a : ℕ → ℝ)
  (h_c : c > 2)
  (cond1 : ∀ m n ≥ 1, a (m + n) ≤ 2 * a m + 2 * a n)
  (cond2 : ∀ k, a (2^k) ≤ 1 / (k + 1)^c) :
  sequence_bounded a :=
sorry

end sequence_bounded_proof_l74_74025


namespace find_a_of_odd_function_l74_74266

theorem find_a_of_odd_function (a : ℝ) (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg : ∀ x, x < 0 → f x = x^2 + a * x)
  (h_pos_value : f 2 = 6) : a = 5 := by
  sorry

end find_a_of_odd_function_l74_74266


namespace find_third_number_l74_74457

theorem find_third_number (first_number second_number third_number : ℕ) 
  (h1 : first_number = 200)
  (h2 : first_number + second_number + third_number = 500)
  (h3 : second_number = 2 * third_number) :
  third_number = 100 := sorry

end find_third_number_l74_74457


namespace part_a_part_b_part_c_l74_74831

-- Given conditions and questions
variable (x y : ℝ)
variable (h : (x - y)^2 - 2 * (x + y) + 1 = 0)

-- Part (a): Prove neither x nor y can be negative
theorem part_a (h : (x - y)^2 - 2 * (x + y) + 1 = 0) : x ≥ 0 ∧ y ≥ 0 := 
sorry

-- Part (b): Prove if x > 1 and y < x, then sqrt{x} - sqrt{y} = 1
theorem part_b (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x > 1) (hy : y < x) : 
  Real.sqrt x - Real.sqrt y = 1 := 
sorry

-- Part (c): Prove if x < 1 and y < 1, then sqrt{x} + sqrt{y} = 1
theorem part_c (h : (x - y)^2 - 2 * (x + y) + 1 = 0) (hx : x < 1) (hy : y < 1) : 
  Real.sqrt x + Real.sqrt y = 1 := 
sorry

end part_a_part_b_part_c_l74_74831


namespace ratio_D_to_C_l74_74562

-- Defining the terms and conditions
def speed_ratio (C Ch D : ℝ) : Prop :=
  (C = 2 * Ch) ∧
  (D / Ch = 6)

-- The theorem statement
theorem ratio_D_to_C (C Ch D : ℝ) (h : speed_ratio C Ch D) : (D / C = 3) :=
by
  sorry

end ratio_D_to_C_l74_74562


namespace solution_inequality_l74_74092

theorem solution_inequality (x : ℝ) : (x - 1) / x ≥ 2 ↔ x ∈ Iic (-1) := by
  sorry

end solution_inequality_l74_74092


namespace divide_polygon_equal_area_l74_74275

theorem divide_polygon_equal_area (fig : Π (angles_right : ∀ α, α = 90°), 
  (∃ (rects : list rectangle), 
    ∀ r ∈ rects, 
    ∃ (center_sym : point), 
    line_through_center center_sym 
    ∧ equal_area (divide_by_line r center_sym))) : 
  ∃ (line : line), divides_into_equal_area_parts line fig :=
sorry

end divide_polygon_equal_area_l74_74275


namespace seating_arrangement_count_l74_74094

def students := ["K1", "K2", "K3", "C1", "C2", "C3", "J1", "J2", "J3"]
def is_valid_seating (seating : List String) : Prop :=
  ∀ i : Fin 9, (seating.get i) != (seating.get ((i + 1) % 9))

def num_valid_seatings : Nat :=
  40320

theorem seating_arrangement_count :
  (Card (Quotient (List.perm.setoid students))) =
  num_valid_seatings :=
by
  sorry

end seating_arrangement_count_l74_74094


namespace relation_of_M_and_N_l74_74673

-- Define the functions for M and N
def M (x : ℝ) : ℝ := (x - 3) * (x - 4)
def N (x : ℝ) : ℝ := (x - 1) * (x - 6)

-- Formulate the theorem to prove M < N for all x
theorem relation_of_M_and_N (x : ℝ) : M x < N x := sorry

end relation_of_M_and_N_l74_74673


namespace smallest_number_neg3_l74_74878

def smallest_number_in_set (s : Set ℤ) (n : ℤ) : Prop :=
  ∀ m ∈ s, n ≤ m

theorem smallest_number_neg3 : smallest_number_in_set ({-3, 2, -2, 0} : Set ℤ) (-3) :=
by
  intro m hm
  cases hm
  case inl hm_eq { rw [hm_eq] }
  case inr hm_in {
    cases hm_in
    case inl hm_eq { rw [hm_eq] }
    case inr hm_in {
      cases hm_in
      case inl hm_eq { rw [hm_eq] }
      case inr hm_eq { rw [hm_eq] }
    }
  }
  show -3 ≤ m
  -- each comparison will be trivial and skipped 
  sorry

end smallest_number_neg3_l74_74878


namespace Leila_spent_on_sweater_l74_74004

noncomputable def amount_spent_on_sweater (T : ℝ) : ℝ :=
  (1 / 4) * T

theorem Leila_spent_on_sweater :
  ∃ (T : ℝ), (∀ (T : ℝ), (3 / 4) * T - 20 = (1 / 4) * T + 60) ∧ 
  (amount_spent_on_sweater T = 40) :=
begin
  use 160,
  sorry,
end

end Leila_spent_on_sweater_l74_74004


namespace solve_equation_and_find_c_d_l74_74755

theorem solve_equation_and_find_c_d : 
  ∃ (c d : ℕ), (∃ x : ℝ, x^2 + 14 * x = 84 ∧ x = Real.sqrt c - d) ∧ c + d = 140 := 
sorry

end solve_equation_and_find_c_d_l74_74755


namespace smaller_group_men_l74_74836

-- Define the main conditions of the problem
def men_work_days : ℕ := 36 * 18  -- 36 men for 18 days

-- Define the theorem we need to prove
theorem smaller_group_men (M : ℕ) (h: M * 72 = men_work_days) : M = 9 :=
by
  -- proof is not required
  sorry

end smaller_group_men_l74_74836


namespace inverse_contrapositive_proof_l74_74442

theorem inverse_contrapositive_proof (x y : ℝ) : (x = 0 ∧ y = 2) → (x + y = 2) :=
by
  assume h : x = 0 ∧ y = 2
  show x + y = 2 from sorry

end inverse_contrapositive_proof_l74_74442


namespace relationship_among_a_b_c_l74_74090

def a := Real.sqrt 2
def b := (1/2)^2
def c := Real.log 0.5 / Real.log 2

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  sorry

end relationship_among_a_b_c_l74_74090


namespace max_n_satisfying_conditions_l74_74021

theorem max_n_satisfying_conditions : 
  ∃ n (a : Fin n → ℕ), 
    1 = a 0 ∧ 
    a (Fin.last n) = 2009 ∧ 
    (∀ i j, i < j → a i < a j) ∧ 
    (∀ i, (∑ k in Finset.univ.erase i, a k) % (n-1) = 0) :=
sorry

end max_n_satisfying_conditions_l74_74021


namespace possible_value_of_phi_l74_74800

theorem possible_value_of_phi : 
  ∃ (φ : ℝ), 
    (∀ x : ℝ, sin (2 * (x + π / 8) + φ) = sin (2 * x + (π / 4) + φ)) ∧ 
    (∀ x : ℝ, sin (2 * x + (π / 4) + φ) = sin (2 * (-x) + (π / 4) + φ)) ∧ 
    φ = π / 4 :=
begin
  sorry
end

end possible_value_of_phi_l74_74800


namespace volume_ratio_l74_74810

def cube_volume (side_length : ℝ) : ℝ :=
  side_length ^ 3

theorem volume_ratio : 
  let a := (4 : ℝ) / 12   -- 4 inches converted to feet
  let b := (2 : ℝ)       -- 2 feet
  cube_volume a / cube_volume b = 1 / 216 :=
by
  sorry

end volume_ratio_l74_74810


namespace smallest_n_for_arrangement_exists_l74_74751

theorem smallest_n_for_arrangement_exists :
  ∃ n : ℕ, n = 65 ∧
  (∀ (a b : ℕ), (¬ is_connected a b → Nat.gcd (a^2 + b^2) n = 1) ∧
                (is_connected a b → Nat.gcd (a^2 + b^2) n > 1)) := 
sorry

end smallest_n_for_arrangement_exists_l74_74751


namespace mass_percentage_hydrogen_in_lysozyme_l74_74915

-- Define the atomic weights
def atomic_weight (element : String) := 
  match element with
  | "C" => 12.01
  | "H" => 1.008
  | "N" => 14.01
  | "O" => 16.00
  | "S" => 32.07
  | _   => 0.0

-- Define the molecular formula of lysozyme
def lysozyme_formula := [("C", 612), ("H", 964), ("N", 166), ("O", 188), ("S", 3)]

-- Calculate the total molar mass of lysozyme
def molar_mass_lysozyme := 
  (612 : Float) * atomic_weight "C" +
  (964 : Float) * atomic_weight "H" +
  (166 : Float) * atomic_weight "N" +
  (188 : Float) * atomic_weight "O" +
  (3   : Float) * atomic_weight "S"

-- Calculate the molar mass of hydrogen component in lysozyme
def molar_mass_hydrogen_lysozyme := (964 : Float) * atomic_weight "H"

-- Calculate the mass percentage of hydrogen in lysozyme
def mass_percentage_hydrogen : Float :=
  (molar_mass_hydrogen_lysozyme / molar_mass_lysozyme) * 100

-- The final theorem to prove
theorem mass_percentage_hydrogen_in_lysozyme :
  abs (mass_percentage_hydrogen - 7.07) < 0.01 :=
by
  -- The proof part is omitted
  sorry

end mass_percentage_hydrogen_in_lysozyme_l74_74915


namespace system_of_equations_solution_l74_74430

theorem system_of_equations_solution (x y z : ℝ) :
  x^2 - y * z = -23 ∧ y^2 - z * x = -4 ∧ z^2 - x * y = 34 →
  (x = 5 ∧ y = 6 ∧ z = 8) ∨ (x = -5 ∧ y = -6 ∧ z = -8) :=
by
  sorry

end system_of_equations_solution_l74_74430


namespace compare_abc_l74_74615

noncomputable def a : ℝ := Real.log 4 / Real.log 3
def b : ℝ := (1 / 5) ^ 0
noncomputable def c : ℝ := Real.log 10 / Real.log (1 / 3)

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l74_74615


namespace Sn_solution_l74_74645

-- Definitions for the problem
def sequence_sum (a : ℕ → ℝ) (n : ℕ) := ∑ i in range n, a (i + 1)
def a_1 : ℝ := 2
def S_n (a : ℕ → ℝ) (n : ℕ) := sequence_sum a n
def S_eq (a : ℕ → ℝ) (n : ℕ) : ℝ := S_n a n

-- Condition for n >= 2
def cond (a : ℕ → ℝ) (n : ℕ) : Prop :=
  n ≥ 2 ∧ S_eq a n ≠ S_eq a (n - 1)

-- Theorem statement
theorem Sn_solution (a : ℕ → ℝ) (n : ℕ) (hn : n ≥ 2) (h_sum_cond : ∀ n, S_eq a n * S_eq a (n - 1) + a n = 0) :
  S_eq a n = 2 / (2 * n - 1) :=
by {
  sorry
}

end Sn_solution_l74_74645


namespace train_speed_l74_74867

def speed_of_train (distance_m : ℕ) (time_s : ℕ) : ℕ :=
  ((distance_m : ℚ) / 1000) / ((time_s : ℚ) / 3600)

theorem train_speed (h_distance : 125 = 125) (h_time : 9 = 9) :
  speed_of_train 125 9 = 50 :=
by
  -- Proof is required here
  sorry

end train_speed_l74_74867


namespace solve_eq1_solve_eq2_solve_eq3_l74_74429

def equation1 (x : ℝ) : Prop := x^2 - 6 * x + 5 = 0
def solution1 (x : ℝ) : Prop := x = 5 ∨ x = 1

theorem solve_eq1 : ∀ x : ℝ, equation1 x ↔ solution1 x := sorry

def equation2 (x : ℝ) : Prop := 3 * x * (2 * x - 1) = 4 * x - 2
def solution2 (x : ℝ) : Prop := x = 1/2 ∨ x = 2/3

theorem solve_eq2 : ∀ x : ℝ, equation2 x ↔ solution2 x := sorry

def equation3 (x : ℝ) : Prop := x^2 - 2 * Real.sqrt 2 * x - 2 = 0
def solution3 (x : ℝ) : Prop := x = Real.sqrt 2 + 2 ∨ x = Real.sqrt 2 - 2

theorem solve_eq3 : ∀ x : ℝ, equation3 x ↔ solution3 x := sorry

end solve_eq1_solve_eq2_solve_eq3_l74_74429


namespace painting_problem_l74_74672

theorem painting_problem
    (H_rate : ℝ := 1 / 60)
    (T_rate : ℝ := 1 / 90)
    (combined_rate : ℝ := H_rate + T_rate)
    (time_worked : ℝ := 15)
    (wall_painted : ℝ := time_worked * combined_rate):
  wall_painted = 5 / 12 := 
by
  sorry

end painting_problem_l74_74672


namespace A_share_of_profit_l74_74123

noncomputable def total_profit : ℝ := 2300

theorem A_share_of_profit (C T : ℝ) :
  let A_investment := (1/6) * C * (1/6) * T in
  let B_investment := (1/3) * C * (1/3) * T in
  let C_investment := (1/2) * C * T in
  let total_investment := A_investment + B_investment + C_investment in
  let A_ratio := A_investment / total_investment in
  let A_share := A_ratio * total_profit in
  A_share = 100 :=
by
  let A_investment := (1/6) * C * (1/6) * T
  let B_investment := (1/3) * C * (1/3) * T
  let C_investment := (1/2) * C * T
  let total_investment := A_investment + B_investment + C_investment
  let A_ratio := A_investment / total_investment
  let A_share := A_ratio * total_profit
  have hA_investment : A_investment = (C * T) / 36 := by sorry
  have hB_investment : B_investment = (C * T) / 9 := by sorry
  have hC_investment : C_investment = (C * T) / 2 := by sorry
  have htotal_investment : total_investment = (C * T) / 36 + (C * T) / 9 + (C * T) / 2 := by sorry
  have htotal_investment_simplified : total_investment = 23 * (C * T) / 36 := by sorry
  have hA_ratio : A_ratio = 1 / 23 := by sorry
  have hA_share : A_share = (1 / 23) * total_profit := by sorry
  have hA_share_value : A_share = 100 := by sorry
  exact hA_share_value

end A_share_of_profit_l74_74123


namespace equivalent_proof_statement_l74_74262

open Real

def proposition_p (x : ℝ) : Prop := sin x + 4 / sin x ≥ 4

def proposition_q (a : ℕ) : Prop := (a = -1) ↔ (∀ (k : ℝ), (a-1) * k + (a+3) * k = 0)

theorem equivalent_proof_statement : (∀ x : ℝ, ¬(proposition_p x)) ∧ proposition_q (-1) :=
by {
  -- show ¬(sin x + 4 / sin x ≥ 4)
  sorry,
  -- show ∀ (k : ℝ), (-1-1) * k + (-1+3) * k = 0
  sorry 
}

end equivalent_proof_statement_l74_74262


namespace oil_tank_depth_l74_74848

theorem oil_tank_depth (h : ℝ) : 
  (∀ (r : ℝ), r = 4) → 
  (∀ (c : ℝ), c = 4) → 
  (∀ (l : ℝ), l = 12) → 
  (∀ (A : ℝ), A = 48) → 
  h^2 - 8*h + 4 = 0 → 
  h = 4 + 2*sqrt 3 :=
by
  intro r r_eq
  intro c c_eq
  intro l l_eq
  intro A A_eq
  intro h_eq
  subst r_eq
  subst c_eq
  subst l_eq
  subst A_eq
  sorry

end oil_tank_depth_l74_74848


namespace fraction_unclaimed_l74_74947

-- Define initial conditions
variable {x : ℝ} (hx : x > 0)

-- Define the fraction calculations based on participants mistaken beliefs
def dave_share := (4/10) * x
def emma_share := (3/10) * (x - dave_share)
def frank_share := (2/10) * (x - dave_share - emma_share)
def george_share := (1/10) * (x - dave_share - emma_share - frank_share)

-- Sum of all shares taken
def total_claimed := dave_share + emma_share + frank_share + george_share

-- Calculate remaining chocolates
def remaining := x - total_claimed

-- Prove the fraction of chocolates that remain unclaimed
theorem fraction_unclaimed (hx : x > 0) : remaining / x = (37.8 / 125) :=
by
  simp [remaining, total_claimed, dave_share, emma_share, frank_share, george_share]
  -- Calculation details will be handled here
  sorry

end fraction_unclaimed_l74_74947


namespace limit_proof_l74_74197

noncomputable def limit_problem_statement : Prop :=
  (∀ (f : ℝ → ℝ) (g : ℝ → ℝ), (∀ x, f x = (sin (4 * x) / x) ^ (2 / (x + 2))) →
    filter.tendsto f (nhds 0) (nhds 4))

theorem limit_proof : limit_problem_statement :=
sorry

end limit_proof_l74_74197


namespace M_inter_N_eq_l74_74375

noncomputable def M := { x : ℝ | log x > 0 }
noncomputable def N := { x : ℝ | -3 ≤ x - 1 ∧ x - 1 ≤ 1 }

theorem M_inter_N_eq : M ∩ N = { x : ℝ | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end M_inter_N_eq_l74_74375


namespace isosceles_triangle_vertex_angle_l74_74689

theorem isosceles_triangle_vertex_angle (T : Type) [triangle T]
  {A B C : T} (h_iso : is_isosceles_triangle A B C)
  (h_angle : ∃ θ : ℝ, (θ = 70 ∧ (angle A B C = θ ∨ angle B C A = θ ∨ angle C A B = θ)))
  : vertex_angle A B C = 70 ∨ vertex_angle A B C = 40 := 
by 
  sorry

end isosceles_triangle_vertex_angle_l74_74689


namespace hexagon_regular_l74_74922

theorem hexagon_regular
  (A1 A2 A3 A4 A5 A6 : Type)
  (hexagon : ConvexHexagon A1 A2 A3 A4 A5 A6)
  (P : A1)
  (r1 r2 r3 r4 r5 r6 : ℝ)
  (circle1 : Circle A1 r1)
  (circle2 : Circle A2 r2)
  (circle3 : Circle A3 r3)
  (circle4 : Circle A4 r4)
  (circle5 : Circle A5 r5)
  (circle6 : Circle A6 r6)
  (radius_eq_side : 
    circle1.radius = hexagon.shorter_side A1 ∧
    circle2.radius = hexagon.shorter_side A2 ∧
    circle3.radius = hexagon.shorter_side A3 ∧
    circle4.radius = hexagon.shorter_side A4 ∧
    circle5.radius = hexagon.shorter_side A5 ∧
    circle6.radius = hexagon.shorter_side A6)
  (P_in_circles : 
    P ∈ circle1 ∧
    P ∈ circle2 ∧
    P ∈ circle3 ∧
    P ∈ circle4 ∧
    P ∈ circle5 ∧
    P ∈ circle6) :
  regular_hexagon A1 A2 A3 A4 A5 A6 :=
by
  sorry

end hexagon_regular_l74_74922


namespace no_tangent_to_x_axis_max_a_monotonically_increasing_l74_74654

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 2) * Real.exp x - (a / 2) * x^2

theorem no_tangent_to_x_axis (a : ℝ) : ¬∃ t : ℝ, f t a = 0 ∧ (t - 1) * Real.exp t = at := 
sorry

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (x - 2) * Real.exp x - (a / 2) * x^2 + 2 * x

theorem max_a_monotonically_increasing : ∃ t : ℝ, g t 1 ≥ 0 ∧ g' 1 > 0 :=
sorry

end no_tangent_to_x_axis_max_a_monotonically_increasing_l74_74654


namespace kelly_chickens_l74_74003

theorem kelly_chickens
  (chicken_egg_rate : ℕ)
  (chickens : ℕ)
  (egg_price_per_dozen : ℕ)
  (total_money : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (dozen : ℕ)
  (total_eggs_sold : ℕ)
  (total_days : ℕ)
  (total_eggs_laid : ℕ) : 
  chicken_egg_rate = 3 →
  egg_price_per_dozen = 5 →
  total_money = 280 →
  weeks = 4 →
  days_per_week = 7 →
  dozen = 12 →
  total_eggs_sold = total_money / egg_price_per_dozen * dozen →
  total_days = weeks * days_per_week →
  total_eggs_laid = chickens * chicken_egg_rate * total_days →
  total_eggs_sold = total_eggs_laid →
  chickens = 8 :=
by
  intros
  sorry

end kelly_chickens_l74_74003


namespace problem_1_problem_2_l74_74018

noncomputable def perimeter_of_triangle (a b c : ℝ) : ℝ :=
  a + b + c

noncomputable def cos_A_minus_C (a b c cos_C : ℝ) : ℝ :=
  let sin_C := Real.sqrt (1 - cos_C^2)
  let sin_A := (a * sin_C) / c
  let cos_A := Real.sqrt (1 - sin_A^2)
  in cos_A * cos_C + sin_A * sin_C

theorem problem_1 (a b c : ℝ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 2) (hyp : c^2 = a^2 + b^2 - 2 * a * b * cos (Real.arccos (1/4))) :
  perimeter_of_triangle a b c = 5 := 
by
  rw [h_a, h_b, h_c]
  unfold perimeter_of_triangle
  norm_num

theorem problem_2 (a b c : ℝ) (cos_C : ℝ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 2) (h_cosC : cos_C = 1/4) :
  cos_A_minus_C a b c cos_C = 11 / 16 := 
by
  rw [h_a, h_b, h_c, h_cosC]
  unfold cos_A_minus_C
  norm_num
  sorry

end problem_1_problem_2_l74_74018


namespace product_of_two_distinct_divisors_l74_74715

theorem product_of_two_distinct_divisors (T : Finset ℕ) : 
  (∃ (n : ℕ), n = 216000 ∧ T = n.divisors ∧ ∃ (m : ℕ), m = 531 ∧ ∀ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b → (a * b ∈ T ∧ T.card = m)) :=
by
  let n := 216000
  let T := n.divisors
  let m := 531
  use n
  split
  { refl }
  split
  { rw ←Nat.mem_divisors at *,
    convert Nat.divisors_eq_of_eq_primeFactorization
    simp only [n] }
  use m
  split
  { refl }
  intros a b h_a h_b h_ab
  split
  { rw ←Nat.mem_divisors
    refine ⟨_, _⟩
    { apply Nat.pos_of_mem_divisors h_a }
    { apply Nat.pos_of_mem_divisors h_b } }
  { sorry }

end product_of_two_distinct_divisors_l74_74715


namespace min_value_function_l74_74932

theorem min_value_function (x y: ℝ) (hx: x > 2) (hy: y > 2) : 
  (∃c: ℝ, c = (x^3/(y - 2) + y^3/(x - 2)) ∧ ∀x y: ℝ, x > 2 → y > 2 → (x^3/(y - 2) + y^3/(x - 2)) ≥ c) ∧ c = 96 :=
sorry

end min_value_function_l74_74932


namespace john_spent_on_sweets_l74_74033

theorem john_spent_on_sweets (initial_amount : ℝ) (amount_given_per_friend : ℝ) (friends : ℕ) (amount_left : ℝ) (total_spent_on_sweets : ℝ) :
  initial_amount = 20.10 →
  amount_given_per_friend = 1.00 →
  friends = 2 →
  amount_left = 17.05 →
  total_spent_on_sweets = initial_amount - (amount_given_per_friend * friends) - amount_left →
  total_spent_on_sweets = 1.05 :=
by
  intros h_initial h_given h_friends h_left h_spent
  sorry

end john_spent_on_sweets_l74_74033


namespace max_min_f_for_m_eq_2_monotonic_increasing_range_m_l74_74952

-- First part of the proof problem
theorem max_min_f_for_m_eq_2 :
  let f := λ x : ℝ, (1 / 2) * x ^ 2 - 2 * Real.log x in
  (f 1 = 1 / 2) ∧ 
  (f Real.exp = (Real.exp ^ 2 - 4) / 2) ∧ 
  (f (Real.sqrt 2) = 1 - Real.log 2) :=
by
  sorry

-- Second part of the proof problem
theorem monotonic_increasing_range_m :
  (∀ x : ℝ, 1 / 2 < x → (x - m / x) ≥ 0) → m ≤ 1 / 4 :=
by
  sorry

end max_min_f_for_m_eq_2_monotonic_increasing_range_m_l74_74952


namespace arithmetic_sequence_properties_l74_74968

noncomputable def a_n (n : ℕ) : ℝ := 5 - 2 * n
noncomputable def S_n (n : ℕ) : ℝ := - n^2 + 4 * n

theorem arithmetic_sequence_properties :
  ∃ (a : ℕ → ℝ),
  (a 3 = -1) ∧
  (a 6 = -7) ∧
  (∀ n : ℕ, a n = 5 - 2 * n) ∧
  (∀ n : ℕ, (S_n : ℕ → ℝ) n = -n^2 + 4 * n) :=
begin
  use a_n,
  repeat {split},
  { sorry },
  { sorry },
  { intros n,
    sorry },
  { intros n,
    sorry }
end

end arithmetic_sequence_properties_l74_74968


namespace C_investment_l74_74339

def A_investment_eq : Prop :=
  ∀ (C T : ℝ), (C * T) / 36 = (1 / 6 : ℝ) * C * (1 / 6 : ℝ) * T

def B_investment_eq : Prop :=
  ∀ (C T : ℝ), (C * T) / 9 = (1 / 3 : ℝ) * C * (1 / 3 : ℝ) * T

def C_investment_eq (x : ℝ) : Prop :=
  ∀ (C T : ℝ), x * C * T = (x : ℝ) * C * T

theorem C_investment (x : ℝ) :
  (∀ (C T : ℝ), A_investment_eq) ∧
  (∀ (C T : ℝ), B_investment_eq) ∧
  (∀ (C T : ℝ), C_investment_eq x) ∧
  (∀ (C T : ℝ), 100 / 2300 = (C * T / 36) / ((C * T / 36) + (C * T / 9) + (x * C * T))) →
  x = 1 / 2 :=
by
  intros
  sorry

end C_investment_l74_74339


namespace sum_S30_l74_74351

-- Define the arithmetic sequence and necessary sums
structure arithmetic_seq (a : ℕ → ℝ) := 
  (is_arithmetic : ∃ d, ∀ n, a (n + 1) - a n = d)

-- Definition of partial sum function S
def partial_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in range (n + 1), a i

theorem sum_S30 (a : ℕ → ℝ) (h_arith : arithmetic_seq a) (h_S10 : partial_sum a 10 = 10) (h_S20 : partial_sum a 20 = 30) :
  partial_sum a 30 = 60 :=
sorry

end sum_S30_l74_74351


namespace distinct_right_triangles_l74_74303

theorem distinct_right_triangles (a b c : ℕ) (h : a = Nat.sqrt 2016)
    (hyp : c^2 = a^2 + b^2) :
  (∃ a b c : ℕ, a = Nat.sqrt 2016 ∧ c^2 = a^2 + b^2 ∧ (a^2 = 2016)) → (∃ n k : ℕ, (n * k = 2016) ∧ (∀ c b : ℕ, n = c - b ∧ k = c + b) ∧ n % 2 = 0 ∧ k % 2 = 0 →
    (c ∈ ℕ) ∧ (b ∈ ℕ) ∧ count distinct (n, k) pairs = 12) := 
  by
    sorry

end distinct_right_triangles_l74_74303


namespace pipe_A_filling_time_l74_74525

noncomputable def time_to_fill_tanker (t_A t_B total_fill_time half_fill_time : ℝ) : Prop :=
  half_fill_time = total_fill_time / 2 ∧
  t_B = 40 ∧
  total_fill_time = 29.999999999999993 ∧
  (half_fill_time / t_B + half_fill_time * (1 / t_A + 1 / t_B) = 1)

theorem pipe_A_filling_time :
  ∃ t_A : ℝ, time_to_fill_tanker t_A 40 29.999999999999993 (29.999999999999993 / 2) ∧ t_A = 60 :=
begin
  use 60,
  unfold time_to_fill_tanker,
  split; try {ring},
  split; try {ring},
  split; try {ring},
  field_simp, ring,
  sorry,
end

end pipe_A_filling_time_l74_74525


namespace cos_value_l74_74612

theorem cos_value (α : Real) (h : sin (π / 6 - α) = 1 / 3) : cos (2 * π / 3 + 2 * α) = -7 / 9 := 
by
  sorry

end cos_value_l74_74612


namespace die_opposite_faces_sum_seven_l74_74790

noncomputable def opposite_faces_sum_seven (die : list (ℕ × ℕ)) : Prop :=
  die = [(1, 6), (2, 5), (3, 4)] ∧
  ∀ (x y : ℕ), (x, y) ∈ die → x + y = 7

theorem die_opposite_faces_sum_seven :
  opposite_faces_sum_seven [(1, 6), (2, 5), (3, 4)] :=
by {
  sorry
}

end die_opposite_faces_sum_seven_l74_74790


namespace coordinates_of_P_on_x_axis_l74_74348

open Real

noncomputable def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

theorem coordinates_of_P_on_x_axis (P : ℝ × ℝ × ℝ) (h : P.2 = 0 ∧ P.3 = 0) :
    distance P (0, sqrt 2, 3) = 2 * sqrt 3 → 
    P = (1, 0, 0) ∨ P = (-1, 0, 0) :=
by
  intro hdist
  sorry

end coordinates_of_P_on_x_axis_l74_74348


namespace can_capacity_is_14_l74_74495

noncomputable def capacity_of_can (M W : ℕ) : ℕ :=
  M + W + 2

theorem can_capacity_is_14 (M W : ℕ) 
  (h1 : M / W = 1 / 5) 
  (h2 : (M + 2) / W = 2.00001 / 5.00001) :
  capacity_of_can M W = 14 := 
sorry

end can_capacity_is_14_l74_74495


namespace find_base_l74_74240

theorem find_base (a : ℕ) (h : a > 11) :
  let B_a := 11
  ∃ a, 396_a + 574_a = 96B_a ∧ a = 12 :=
by {
  -- proof will go here
  sorry
}

end find_base_l74_74240


namespace combined_work_rate_l74_74144

theorem combined_work_rate (A B C D : ℝ) (hA : A = 1/10) (hB : B = 1/15) (hC : C = 1/20) (hD : D = 1/30) :
  1 / (A + B + C + D) = 4 :=
by
  have combined_rate := (1/10 : ℝ) + (1/15) + (1/20) + (1/30)
  have correct_rate := 1 / combined_rate
  calc
    A + B + C + D = combined_rate : by rw [hA, hB, hC, hD]
    1 / (A + B + C + D) = correct_rate : by rw [combined_rate]
    1 / (15 / 60) = 4 : by norm_num
    4 = 4 : by rfl

end combined_work_rate_l74_74144


namespace nonconvex_quadrilateral_partition_l74_74365

noncomputable def is_partitioned_into_six_parts (quadrilateral : Type) [non_convex_quadrilateral quadrilateral] : Prop :=
  ∃ (L1 L2 : line) (P : point), 
    L1 ∩ quadrilateral ≠ ∅ ∧ 
    L2 ∩ quadrilateral ≠ ∅ ∧ 
    (L1 ∩ L2).nonempty ∧
    partitioned_into_six_parts quadrilateral L1 L2

theorem nonconvex_quadrilateral_partition : 
  ∀ (quadrilateral : Type) [non_convex_quadrilateral quadrilateral], 
    is_partitioned_into_six_parts quadrilateral :=
by
  -- The proof would go here
  sorry

end nonconvex_quadrilateral_partition_l74_74365


namespace polynomial_coeff_sum_l74_74667

variables {a_0 a_1 a_2 a_3 a_4 a_5 : ℝ}

theorem polynomial_coeff_sum :
  (∑ i in {0, 1, 2, 3, 4, 5}, a_i) = -1 ∧
  (∑ i in {0, 1, 2, 3, 4, 5}, (-1)^i * a_i) = 243 →
  (a_0 + a_2 + a_4)^2 - (a_1 + a_3 + a_5)^2 = -243 :=
by
  intros h
  sorry

end polynomial_coeff_sum_l74_74667


namespace days_production_l74_74492

theorem days_production (n : ℕ) (avg_past_n_days : ℕ) (today_production : ℕ) (new_avg : ℕ) :
  avg_past_n_days = 50 → today_production = 115 → new_avg = 55 → 
  (50 * n + 115 = 55 * (n + 1)) → n = 12 :=
by
  intros h1 h2 h3 h4
  calc
    50 * n + 115      = 55 * (n + 1) : by assumption
                ...   = 55 * n + 55  : by ring
    50 * n + 115 - 55 = 55 * n + 55 - 55 : by rw h3
                ...   = 55 * n        : by ring
    50 * n + 60       = 55 * n        : by nat_sub
                ...   = 50 * n + 5 * n : by ring_exp (1 + 1)
    5 * n            = 60             : by linarith
    n = 60 / 5        = 12            : by nat_div
sorry

# Check that the statement can be built successfully:
# theorem days_production : ∀ n : ℕ, ∀ avg_past_n_days : ℕ, 
# ∀ today_production : ℕ, ∀ new_avg : ℕ,
# avg_past_n_days = 50 → today_production = 115 → 
# new_avg = 55 →  (50 * n + 115 = 55 * (n + 1)) → n = 12 :=

end days_production_l74_74492


namespace sum_of_vectors_nonzero_l74_74203

open Complex

noncomputable def zeta (n : ℕ) : ℂ := exp (2 * π * I / n)

theorem sum_of_vectors_nonzero {n : ℕ} (hn : Prime n) (h : n = 1987) :
    ∀ I : finset ℕ, I ⊆ finset.range n → I ≠ ∅ → I ≠ finset.range n → 
    ∑ i in I, zeta n ^ i ≠ 0 :=
begin
  sorry
end

end sum_of_vectors_nonzero_l74_74203


namespace problem_statement_l74_74241

noncomputable def a (n : ℕ) [h : Fact (1 < n)] : ℝ :=
  1 / Real.logb (1000 : ℝ) n

noncomputable def b : ℝ :=
  a 3 + a 4 + a 5 + a 6

noncomputable def c : ℝ :=
  a 10 + a 15 + a 20

theorem problem_statement : b - c = -0.92082 := by
  sorry

end problem_statement_l74_74241


namespace meal_total_l74_74892

noncomputable def meal_price (appetizer entree dessert drink sales_tax tip : ℝ) : ℝ :=
  let total_before_tax := appetizer + (2 * entree) + dessert + (2 * drink)
  let tax_amount := (sales_tax / 100) * total_before_tax
  let subtotal := total_before_tax + tax_amount
  let tip_amount := (tip / 100) * subtotal
  subtotal + tip_amount

theorem meal_total : 
  meal_price 9 20 11 6.5 7.5 22 = 95.75 :=
by
  sorry

end meal_total_l74_74892


namespace inequality_proof_l74_74614

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := 2 ^ 0.3
noncomputable def c : ℝ := 0.3 ^ 0.2

theorem inequality_proof : b > c ∧ c > a := by
  sorry

end inequality_proof_l74_74614


namespace mrs_franklin_gave_38_packs_l74_74731

-- Define the initial number of Valentines
def initial_valentines : Int := 450

-- Define the remaining Valentines after giving some away
def remaining_valentines : Int := 70

-- Define the size of each pack
def pack_size : Int := 10

-- Define the number of packs given away
def packs_given (initial remaining pack_size : Int) : Int :=
  (initial - remaining) / pack_size

theorem mrs_franklin_gave_38_packs :
  packs_given 450 70 10 = 38 := sorry

end mrs_franklin_gave_38_packs_l74_74731


namespace prob_seventh_grade_prob_mixed_grade_l74_74406

-- Define students as types within their respective grades
inductive Student
| A | B | C | D

def seventh_grade : set Student := { Student.A, Student.B }
def eighth_grade : set Student := { Student.C, Student.D }

-- Total number of students with outstanding awards
def total_students := 4

-- Number of seventh-grade students with outstanding awards
def seventh_grade_students := 2

-- Probability that a randomly selected outstanding student is from the seventh grade
def seventh_grade_prob := (seventh_grade_students : ℚ) / total_students

theorem prob_seventh_grade : seventh_grade_prob = 1 / 2 :=
by
  -- calculation here
  sorry

-- Number of outcomes where one student is from seventh grade and the other from eighth grade
def favorable_outcomes := 8

-- Total possible outcomes
def possible_outcomes := total_students * (total_students - 1) / 2

-- Probability of selecting one from seventh grade and one from eighth grade
def prob_mixed_grade := (favorable_outcomes : ℚ) / possible_outcomes

theorem prob_mixed_grade : prob_mixed_grade = 2 / 3 :=
by
  -- calculation here
  sorry

end prob_seventh_grade_prob_mixed_grade_l74_74406


namespace exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l74_74503

theorem exists_integers_for_x_squared_minus_y_squared_eq_a_fifth (a : ℤ) : 
  ∃ x y : ℤ, x^2 - y^2 = a^5 :=
sorry

end exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l74_74503


namespace area_of_rectangle_abcd_l74_74099

-- Given conditions
def shorter_side (r : ℝ) : Prop := r = 7
def longer_side (r : ℝ) : Prop := r = 14

-- Given proof problem
theorem area_of_rectangle_abcd (a b : ℝ) (h1 : shorter_side a) (h2 : longer_side b) :
  a + a = b -> a * b = 196 :=
by
  -- variables
  let shorter_side_length := 7
  let longer_side_length := 14

  -- Given that the shorter side of each smaller rectangle is 7 feet
  have hs : shorter_side_length = 7 := by exact (by rfl : shorter_side shorter_side_length)

  -- Given that the longer side of each smaller rectangle is 14 feet
  have hl : longer_side_length = 14 := by exact (by rfl : longer_side longer_side_length)

  -- Calculate the area of the larger rectangle ABCD
  calc
    shorter_side_length * longer_side_length = 7 * 14 : by rw [hs, hl]
    ... = 196 : by norm_num

-- sorry placeholder to skip the proof.
sorry

end area_of_rectangle_abcd_l74_74099


namespace problem1_problem2_problem3_problem4_l74_74283

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x - 1
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - a

-- Prove that if f is increasing on ℝ, then a ∈ (-∞, 0]
theorem problem1 (a : ℝ) : (∀ x y : ℝ, x ≤ y → f x a ≤ f y a) → a ≤ 0 :=
sorry

-- Prove that if f is decreasing on (-1, 1), then a ∈ [3, ∞)
theorem problem2 (a : ℝ) : (∀ x y : ℝ, -1 < x → x < 1 → -1 < y → y < 1 → x ≤ y → f x a ≥ f y a) → 3 ≤ a :=
sorry

-- Prove that if the decreasing interval of f is (-1, 1), then a = 3
theorem problem3 (a : ℝ) : (∀ x : ℝ, (abs x < 1) ↔ f' x a < 0) → a = 3 :=
sorry

-- Prove that if f is not monotonic on (-1, 1), then a ∈ (0, 3)
theorem problem4 (a : ℝ) : (¬(∀ x : ℝ, -1 < x → x < 1 → (f' x a = 0) ∨ (f' x a ≠ 0))) → (0 < a ∧ a < 3) :=
sorry

end problem1_problem2_problem3_problem4_l74_74283


namespace rhombus_area_l74_74436

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) : (d1 * d2) / 2 = 60 := by
  -- substitution of given values
  rw [h1, h2]
  -- calculate the area
  norm_num
  sorry

end rhombus_area_l74_74436


namespace fit_rectangles_within_region_l74_74209

theorem fit_rectangles_within_region :
  ∃ (rectangles : ℕ), rectangles = 34 ∧
  ∀ (region : set (ℤ × ℤ)),
    (∀ (x y : ℤ), region (x, y) ↔ 
      (y ≤ 2 * x) ∧ (y ≥ -2) ∧ (x ≤ 10)) →
    (rectangles = ∑ x in finset.Icc 0 8,
      (finset.Icc (-2) (2 * x)).card) := 
by
  sorry

end fit_rectangles_within_region_l74_74209


namespace exists_natural_number_n_l74_74415

theorem exists_natural_number_n (t : ℕ) (ht : t > 0) :
  ∃ n : ℕ, n > 1 ∧ Nat.gcd n t = 1 ∧ ∀ k : ℕ, k > 0 → ∃ m : ℕ, m > 1 → n^k + t ≠ m^m :=
by
  sorry

end exists_natural_number_n_l74_74415


namespace phone_call_cost_5_5_equals_4_24_l74_74768

def ceil (x : ℝ) : ℝ := ⌈x⌉

def phone_call_cost (m : ℝ) : ℝ :=
  1.06 * (0.5 * ceil m + 1)

theorem phone_call_cost_5_5_equals_4_24 (m : ℝ) (h : m = 5.5) : phone_call_cost m = 4.24 :=
by
  sorry

end phone_call_cost_5_5_equals_4_24_l74_74768


namespace circle_radius_doubling_l74_74451

theorem circle_radius_doubling (r : ℝ) : 
  let new_radius := 2 * r
  let original_circumference := 2 * Real.pi * r
  let new_circumference := 2 * Real.pi * new_radius
  let original_area := Real.pi * r^2
  let new_area := Real.pi * (new_radius)^2
  (new_circumference = 2 * original_circumference) ∧ (new_area = 4 * original_area) :=
by
  let new_radius := 2 * r
  let original_circumference := 2 * Real.pi * r
  let new_circumference := 2 * Real.pi * new_radius
  let original_area := Real.pi * r^2
  let new_area := Real.pi * (new_radius)^2
  have hc : new_circumference = 2 * original_circumference := by
    sorry
  have ha : new_area = 4 * original_area := by
    sorry
  exact ⟨hc, ha⟩

end circle_radius_doubling_l74_74451


namespace combined_age_l74_74137

-- Define the conditions as Lean assumptions
def avg_age_three_years_ago := 19
def number_of_original_members := 6
def number_of_years_passed := 3
def current_avg_age := 19

-- Calculate the total age three years ago
def total_age_three_years_ago := number_of_original_members * avg_age_three_years_ago 

-- Calculate the increase in total age over three years
def total_increase_in_age := number_of_original_members * number_of_years_passed 

-- Calculate the current total age of the original members
def current_total_age_of_original_members := total_age_three_years_ago + total_increase_in_age

-- Define the number of current total members and the current total age
def number_of_current_members := 8
def current_total_age := number_of_current_members * current_avg_age

-- Formally state the problem and proof
theorem combined_age : 
  (current_total_age - current_total_age_of_original_members = 20) := 
by
  sorry

end combined_age_l74_74137


namespace garden_area_garden_perimeter_l74_74472

noncomputable def length : ℝ := 30
noncomputable def width : ℝ := length / 2
noncomputable def area : ℝ := length * width
noncomputable def perimeter : ℝ := 2 * (length + width)

theorem garden_area :
  area = 450 :=
sorry

theorem garden_perimeter :
  perimeter = 90 :=
sorry

end garden_area_garden_perimeter_l74_74472


namespace range_m_satisfying_p_and_q_l74_74978

theorem range_m_satisfying_p_and_q : {m : ℝ // 2 < m ∧ m < 3} :=
begin
  sorry
end

end range_m_satisfying_p_and_q_l74_74978


namespace positive_difference_x_coordinates_lines_l74_74949

theorem positive_difference_x_coordinates_lines :
  let l := fun x : ℝ => -2 * x + 4
  let m := fun x : ℝ => - (1 / 5) * x + 1
  let x_l := (- (10 - 4) / 2)
  let x_m := (- (10 - 1) * 5)
  abs (x_l - x_m) = 42 := by
  sorry

end positive_difference_x_coordinates_lines_l74_74949


namespace superMonotonous_count_l74_74205

def isSuperMonotonous (n : ℕ) : Prop :=
  (∃ d, n = d ∧ 1 ≤ d ∧ d ≤ 9 ∧ d % 3 = 0) ∨
  (∀ i j, i < j → digits n i < digits n j ∧ n % 3 = 0)

def digits (n : ℕ) : list ℕ := 
  nat.digits 10 n

theorem superMonotonous_count : 
  (∃ S : finset ℕ, ∀ s ∈ S, isSuperMonotonous s ∧ s < 10^9 ∧ S.card = 43) :=
sorry

end superMonotonous_count_l74_74205


namespace shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l74_74850

def false_weight_kgs (false_weight_g : ℕ) : ℚ := false_weight_g / 1000

def shopkeeper_gain_percentage (false_weight_g price_per_kg : ℕ) : ℚ :=
  let actual_price := false_weight_kgs false_weight_g * price_per_kg
  let gain := price_per_kg - actual_price
  (gain / actual_price) * 100

theorem shopkeeper_gain_first_pulse :
  shopkeeper_gain_percentage 950 10 = 5.26 := 
sorry

theorem shopkeeper_gain_second_pulse :
  shopkeeper_gain_percentage 960 15 = 4.17 := 
sorry

theorem shopkeeper_gain_third_pulse :
  shopkeeper_gain_percentage 970 20 = 3.09 := 
sorry

end shopkeeper_gain_first_pulse_shopkeeper_gain_second_pulse_shopkeeper_gain_third_pulse_l74_74850


namespace sum_of_lengths_eq_n_MN_volume_between_planes_eq_s_MN_l74_74107

variables {Point : Type*} [MetricSpace Point]
variables (n : ℕ) 
variables (A B : Fin n → Point)
variables (M N : Point) (s : ℝ) (MN : dist M N)

-- Given conditions
def is_centroid (C : Point) (P : Fin n → Point) : Prop :=
  (n : ℝ) • C = ∑ i, P i

def planes_intersect_lateral_edges (A : Fin n → Point) (B : Fin n → Point) : Prop :=
  ∀ i, ∃ p q : ℝ, (p ≠ q)

theorem sum_of_lengths_eq_n_MN (hA_centroid: is_centroid M A)
  (hB_centroid: is_centroid N B)
  (h_planes_no_common_points: planes_intersect_lateral_edges A B) :
  ∑ i, dist (A i) (B i) = n * dist M N := sorry

theorem volume_between_planes_eq_s_MN (hA_centroid: is_centroid M A)
  (hB_centroid: is_centroid N B)
  (h_planes_no_common_points: planes_intersect_lateral_edges A B)
  (base_area: ℝ) :
  volume_between_planes base_area (dist M N) = base_area * dist M N := sorry

end sum_of_lengths_eq_n_MN_volume_between_planes_eq_s_MN_l74_74107


namespace burgers_prepared_in_10_minutes_l74_74704

def jackson_preparation_time := 40 -- Jackson can prepare a burger every 40 seconds
def alex_preparation_time := 50 -- Alex can prepare a burger every 50 seconds
def break_time := 10 -- Each takes a 10-second break after preparing 4 burgers
def total_time := 600 -- Total working time is 10 minutes (600 seconds)

theorem burgers_prepared_in_10_minutes : 
  let combined_rate := (1 / jackson_preparation_time) + (1 / alex_preparation_time)
  let burgers_per_cycle := 4
  let cycle_time_without_break := burgers_per_cycle * (1 / combined_rate)
  let total_cycle_time := cycle_time_without_break + break_time
  let total_cycles := total_time / total_cycle_time
  (floor total_cycles) * burgers_per_cycle = 24 := 
  sorry

end burgers_prepared_in_10_minutes_l74_74704


namespace ac_square_sufficient_not_necessary_l74_74132

theorem ac_square_sufficient_not_necessary (a b c: ℝ) (hc: c ≠ 0) : (ac^2 > bc^2) → (a > b) ∧ ¬(a > b → ac^2 > bc^2) :=
by
  sorry

end ac_square_sufficient_not_necessary_l74_74132


namespace roots_value_of_quadratic_l74_74716

-- Define the quadratic equation and roots
def quadratic_equation := ∀ (x : ℝ), x^2 + x - 2023 = 0

-- Prove that the value of a^2 + 2a + b is 2022
theorem roots_value_of_quadratic (a b : ℝ) (ha : quadratic_equation a) (hb : quadratic_equation b) : 
  a^2 + 2 * a + b = 2022 :=
by 
  sorry

end roots_value_of_quadratic_l74_74716


namespace exists_infinite_composite_l74_74919

theorem exists_infinite_composite : 
  ∃ (k : ℕ), (∀ n : ℕ, n > 0 → ∃ p : ℕ, prime p ∧ p ∣ (k * 2^n + 1) ∧ p ≠ (k * 2^n + 1)) := sorry

end exists_infinite_composite_l74_74919


namespace total_items_8_l74_74331

def sandwiches_cost : ℝ := 5.0
def soft_drinks_cost : ℝ := 1.5
def total_money : ℝ := 40.0

noncomputable def total_items (s : ℕ) (d : ℕ) : ℕ := s + d

theorem total_items_8 :
  ∃ (s d : ℕ), 5 * (s : ℝ) + 1.5 * (d : ℝ) = 40 ∧ s + d = 8 := 
by
  sorry

end total_items_8_l74_74331


namespace speed_with_stream_l74_74158

variable (V_as V_m V_ws : ℝ)

theorem speed_with_stream (h1 : V_as = 6) (h2 : V_m = 2) : V_ws = V_m + (V_as - V_m) :=
by
  sorry

end speed_with_stream_l74_74158


namespace N_square_solutions_l74_74911

theorem N_square_solutions :
  ∀ N : ℕ, (N > 0 → ∃ k : ℕ, 2^N - 2 * N = k^2) → (N = 1 ∨ N = 2) :=
by
  sorry

end N_square_solutions_l74_74911


namespace sum_of_divisors_is_12_l74_74591

-- Define the list of numbers.
def list_of_numbers := [48, 144, -24, 30, 192]

-- Define a function that calculates the sum of the four common positive divisors.
noncomputable def sum_of_common_divisors : ℕ :=
  let common_divisors := [1, 2, 3, 6] in
  common_divisors.foldl (λ acc x => acc + x) 0

-- The statement to prove.
theorem sum_of_divisors_is_12 : sum_of_common_divisors = 12 :=
  by
    sorry

end sum_of_divisors_is_12_l74_74591


namespace probability_of_multiple_of_2_4_5_l74_74750

theorem probability_of_multiple_of_2_4_5 (n : ℕ) (cards : set ℕ) (h1 : n = 120)
  (h2 : cards = {i | 1 ≤ i ∧ i ≤ 120}) : 
  (∃ card ∈ cards, card % 2 = 0 ∨ card % 4 = 0 ∨ card % 5 = 0) ∧ (∃ p : ℚ, p = 11 / 20) :=
by
  sorry

end probability_of_multiple_of_2_4_5_l74_74750


namespace area_triangle_60_cm_squared_l74_74089

def area_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem area_triangle_60_cm_squared 
  (length_rect : ℕ) (width_rect : ℕ) (ratio_rect_triangle : ℕ × ℕ) 
  (length_rect_eq : length_rect = 6) (width_rect_eq : width_rect = 4) 
  (ratio_rect_triangle_eq : ratio_rect_triangle = (2, 5)) :
  ∃ area_triangle : ℕ, area_triangle = 60 :=
  
  let area_rect := area_rectangle length_rect width_rect in
  have area_rect_eq : area_rect = 24, by {
    rw [length_rect_eq, width_rect_eq],
    exact rfl,
  },
  have ratio := ratio_rect_triangle_eq,
  have A_triangle := (area_rect * ratio_rect_triangle.2) / ratio_rect_triangle.1,
  have : A_triangle = 60, by sorry,
  ⟨A_triangle, this⟩

end area_triangle_60_cm_squared_l74_74089


namespace sequence_sum_l74_74625

noncomputable def a_sequence (n : ℕ) : ℝ := 2 * (1 / 3) ^ n

def b_sequence (n : ℕ) : ℝ := 2 * n - 1

def c_sequence (n : ℕ) : ℝ := a_sequence n * b_sequence n

def sum_first_n_terms (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum f

theorem sequence_sum (n : ℕ) :
  sum_first_n_terms c_sequence n = 2 - (2 * n + 2) / 3 ^ n :=
sorry

end sequence_sum_l74_74625


namespace gcd_of_three_numbers_l74_74439

theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd 279 372) 465 = 93 := 
by 
  sorry

end gcd_of_three_numbers_l74_74439


namespace monotonicity_of_f_range_of_a_l74_74278

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log x - a / x

theorem monotonicity_of_f (a : ℝ) (h : 0 < a) :
  ∀ x y : ℝ, (0 < x) → (0 < y) → (x < y) → (f x a < f y a) :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a < x ^ 2) ↔ (-1 ≤ a) :=
by
  sorry

end monotonicity_of_f_range_of_a_l74_74278


namespace trajectory_equation_l74_74648

theorem trajectory_equation (z : ℂ) (h : ∥z∥^2 - 2 * ∥z∥ - 3 = 0) :
  ∃ (x y : ℝ), z = x + y * complex.I ∧ x^2 + y^2 = 9 :=
by
  sorry

end trajectory_equation_l74_74648


namespace range_of_x_l74_74329

theorem range_of_x (x : ℝ) (h : 2 * x + 1 ≤ 0) : x ≤ -1 / 2 := 
  sorry

end range_of_x_l74_74329


namespace pentagon_coloring_count_l74_74097

/-- A pentagon ABCDE where each vertex can be colored with one of the three colors: red, yellow, green.
    Adjacent vertices must have different colors. Prove that the number of different valid coloring methods is 30. -/
theorem pentagon_coloring_count :
  let colors := {1, 2, 3} -- Representing red, yellow, green as 1, 2, 3
  in let valid_coloring (v : Fin 5 → ℕ) := 
       (v 0 ≠ v 1) ∧ (v 1 ≠ v 2) ∧ (v 2 ≠ v 3) ∧ (v 3 ≠ v 4) ∧ (v 4 ≠ v 0)
  in finset.card 
       {f : Fin 5 → ℕ | (∀ i, f i ∈ colors) ∧ valid_coloring f} = 30 := sorry

end pentagon_coloring_count_l74_74097


namespace young_or_old_woman_lawyer_probability_l74_74837

/-- 
40 percent of the members of a study group are women.
Among these women, 30 percent are young lawyers.
10 percent are old lawyers.
Prove the probability that a member randomly selected is a young or old woman lawyer is 0.16.
-/
theorem young_or_old_woman_lawyer_probability :
  let total_members := 100
  let women_percentage := 40
  let young_lawyers_percentage := 30
  let old_lawyers_percentage := 10
  let total_women := (women_percentage * total_members) / 100
  let young_women_lawyers := (young_lawyers_percentage * total_women) / 100
  let old_women_lawyers := (old_lawyers_percentage * total_women) / 100
  let women_lawyers := young_women_lawyers + old_women_lawyers
  let probability := women_lawyers / total_members
  probability = 0.16 := 
by {
  sorry
}

end young_or_old_woman_lawyer_probability_l74_74837


namespace num_ordered_quadruples_eq_58_l74_74933

theorem num_ordered_quadruples_eq_58 :
  (∃! (S : Finset (ℕ × ℕ × ℕ × ℕ)), ∀ (x : ℕ × ℕ × ℕ × ℕ), x ∈ S ↔ (0 < x.1.1 ∧ 0 < x.1.2 ∧ 0 < x.2.1 ∧ 0 < x.2.2 ∧ (x.1.1 * x.1.2 + x.2.1 * x.2.2 = 10)) ∧ S.card = 58) :=
by
  sorry

end num_ordered_quadruples_eq_58_l74_74933


namespace find_m_l74_74377

def triangleABC (A B C : Point) :=
  rightTriangleAt A B C ∧
  distance A B = 80 ∧
  distance A C = 150 ∧
  distance B C = 170

def inscribedCircle (ABC : Triangle) (circleC : Circle) :=
  inscribedCircleOf ABC circleC

def inscribedCirclesDistanceSquared (C1 C2 C3 : Circle) (d : ℝ) :=
  distance (centerOf C2) (centerOf C3) = d ∧
  d = sqrt(10 * 1765.88125)

theorem find_m (A B C D E F G : Point) (C1 C2 C3 : Circle) (m : ℝ) :
  triangleABC A B C →
  tangentSegmentOf C1 D E A C →
  tangentSegmentOf C1 F G A B →
  inscribedCircle (triangleOf A B C) C1 →
  inscribedCircle (triangleOf B D E) C2 →
  inscribedCircle (triangleOf A F G) C3 →
  inscribedCirclesDistanceSquared C1 C2 C3 (sqrt(17658.8125)) →
  m = 1765.88125 :=
by
  sorry

end find_m_l74_74377


namespace customer_paid_l74_74086

theorem customer_paid (cost_price : ℝ) (markup_rate : ℝ) (final_price : ℝ) : 
  cost_price = 6947.5 → 
  markup_rate = 0.20 → 
  final_price = cost_price * (1 + markup_rate) → 
  final_price = 8337 := by
  intros hcp hmr hfp
  rw [hcp, hmr] at hfp
  simp at hfp
  exact hfp

end customer_paid_l74_74086


namespace max_real_roots_l74_74578

theorem max_real_roots (n : ℕ) (hn : 0 < n) :
  ∀ x : ℝ, (P x = 0 → ((even n → ∃! x : ℝ, P x = 0) ∧ (odd n → ¬ ∃ x : ℝ, P x = 0))) :=
by
  sorry

def P (x : ℝ) (n : ℕ) : ℝ :=
  ∑ i in range (n + 1), if even i then x^i else -x^i

namespace hidden

variables {x : ℝ} {n : ℕ}

end hidden

end max_real_roots_l74_74578


namespace positive_integral_solution_l74_74221

theorem positive_integral_solution :
  ∃ (n : ℕ), 0 < n ∧ (∑ k in Finset.range n, (2 * k + 1)) / (∑ k in Finset.range n, (2 * (k + 1))) = 121 / 122 :=
  sorry

end positive_integral_solution_l74_74221


namespace plywood_cut_difference_l74_74141

theorem plywood_cut_difference :
  ∀ (length width : ℕ) (n : ℕ) (perimeter_greatest perimeter_least : ℕ),
    length = 8 ∧ width = 4 ∧ n = 4 ∧
    (∀ l w, (l = (length / 2) ∧ w = width) ∨ (l = length ∧ w = (width / 2)) → (perimeter_greatest = 2 * (l + w))) ∧
    (∀ l w, (l = (length / n) ∧ w = width) ∨ (l = length ∧ w = (width / n)) → (perimeter_least = 2 * (l + w))) →
    length = 8 ∧ width = 4 ∧ n = 4 ∧ perimeter_greatest = 18 ∧ perimeter_least = 12 →
    (perimeter_greatest - perimeter_least) = 6 :=
by
  intros length width n perimeter_greatest perimeter_least h1 h2
  sorry

end plywood_cut_difference_l74_74141


namespace ryegrass_percentage_l74_74060

theorem ryegrass_percentage (R : ℝ) :
  (R / 100) * (1 / 3) + (25 / 100) * (2 / 3) = 30 / 100 →
  R = 40 :=
by
  intro h1
  have h : R / 3 + 25 * 2 / 3 = 30 * 1 by norm_num at h1
  -- Convert R into a percentage and solve
  have : R + 50 = 90 :=
  by calc
    R + 50 = 90 := sorry -- Detailed algebraic steps left out
  -- Subtract 50 from both sides
  show R = 40, from sorry

end ryegrass_percentage_l74_74060


namespace equal_angles_seen_from_P_l74_74469

theorem equal_angles_seen_from_P
  {circle₁ circle₂ : Circle}
  {P Q A B C : Point}
  (h₀ : circle₁.radius = circle₂.radius)
  (h₁ : P ∈ circle₁)
  (h₂ : P ∈ circle₂)
  (h₃ : Q ∈ circle₁)
  (h₄ : Q ∈ circle₂)
  (h₅ : A ∈ circle₁)
  (h₆ : B ∈ circle₂)
  (h₇ : Q ∈ Line(A, B))
  (h₈ : TangentAt(circle₁, A) ∩ TangentAt(circle₂, B) = {C}) :
  ∠BPC = ∠QPA := 
sorry

end equal_angles_seen_from_P_l74_74469


namespace percent_dried_fruit_correct_l74_74757

noncomputable def total_weight (sue_mix jane_mix tom_mix : ℕ) (sue_nuts_p jane_nuts_p tom_nuts_p : ℚ) : ℚ :=
sue_mix * sue_nuts_p + jane_mix * jane_nuts_p + tom_mix * tom_nuts_p

noncomputable def combined_weight : ℚ := 9.3 / 0.45

noncomputable def total_dried_fruit (sue_mix tom_mix : ℕ) (sue_dried_f_p tom_dried_f_p : ℚ) : ℚ :=
sue_mix * sue_dried_f_p + tom_mix * tom_dried_f_p

noncomputable def percent_dried_fruit (total_dried_fruit combined_weight : ℚ) : ℚ :=
(total_dried_fruit / combined_weight) * 100

theorem percent_dried_fruit_correct :
  let sue_mix := 5
  let jane_mix := 7
  let tom_mix := 9
  let sue_nuts_p := 0.30
  let jane_nuts_p := 0.60
  let tom_nuts_p := 0.40
  let sue_dried_f_p := 0.70
  let tom_dried_f_p := 0.50 in
  percent_dried_fruit (total_dried_fruit sue_mix tom_mix sue_dried_f_p tom_dried_f_p) combined_weight ≈ 38.71 := 
sorry

end percent_dried_fruit_correct_l74_74757


namespace complex_power_identity_l74_74977

-- Given condition
variable (z : Complex)
variable (h : z + 1/z = 2 * Real.cos (Real.pi / 36))

-- Proof Problem
theorem complex_power_identity :
  z ^ 1000 + 1 / (z ^ 1000) = -2 * Real.cos (Real.pi * 40 / 180) :=
sorry

end complex_power_identity_l74_74977


namespace find_height_of_box_l74_74072

/-- The dimensions of a certain rectangular box are 5 inches by 2 inches by some height h. 
The face of greatest area has an area of 15 square inches. 
Prove that the height of the missing dimension is 3 inches. -/
theorem find_height_of_box 
  (h : ℝ) 
  (area_greatest_face: 15)
  (dim1: 5) 
  (dim2: 2)
  (greatest_area_face: 15 = 5 * h) :
  h = 3 := 
by
  sorry

end find_height_of_box_l74_74072


namespace hyperbola_angle_asymptotes_l74_74327

noncomputable def angle_between_asymptotes (m : ℝ) : ℝ :=
  arccos (7 / 25)

theorem hyperbola_angle_asymptotes :
  ∃ m : ℝ, (∀ x y : ℝ, (x, y) = (4 * real.sqrt 2, 3) → (x ^ 2) / 16 - (y ^ 2) / m = 1) →
  angle_between_asymptotes m = arccos (7 / 25) :=
begin
  use 9,
  intros x y h,
  simp at h,
  sorry
end

end hyperbola_angle_asymptotes_l74_74327


namespace pq_sum_equals_4_l74_74378

theorem pq_sum_equals_4 (p q : ℝ) (h : (Polynomial.C 1 + Polynomial.C q * Polynomial.X + Polynomial.C p * Polynomial.X^2 + Polynomial.X^4).eval (2 + I) = 0) :
  p + q = 4 :=
sorry

end pq_sum_equals_4_l74_74378


namespace triangle_angle_inequality_l74_74804

theorem triangle_angle_inequality (α β γ : ℝ) (h_triangle : α + β + γ = 180)
    (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) : false :=
by {
    have h_total : α + β + γ > 60 + 60 + 60 := add_lt_add (add_lt_add h1 h2) h3,
    linarith
}

end triangle_angle_inequality_l74_74804


namespace train_passing_bridge_time_l74_74821

theorem train_passing_bridge_time
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ)
  (h_train : length_train = 320)
  (h_bridge : length_bridge = 140)
  (h_speed : speed_kmh = 45) :
  let total_distance := length_train + length_bridge in
  let speed_mps := speed_kmh * (1000 / 3600) in
  let time := total_distance / speed_mps in
  time = 36.8 :=
by
  sorry

end train_passing_bridge_time_l74_74821


namespace train_speed_l74_74868

def speed_of_train (distance_m : ℕ) (time_s : ℕ) : ℕ :=
  ((distance_m : ℚ) / 1000) / ((time_s : ℚ) / 3600)

theorem train_speed (h_distance : 125 = 125) (h_time : 9 = 9) :
  speed_of_train 125 9 = 50 :=
by
  -- Proof is required here
  sorry

end train_speed_l74_74868


namespace part1_prob_seventh_grade_part2_prob_one_seventh_one_eighth_l74_74404

-- Definitions corresponding to conditions
def total_students : ℕ := 4
def seventh_grade_students : ℕ := 2
def eighth_grade_students : ℕ := 2

-- Probability problem statements to be proven
theorem part1_prob_seventh_grade : 
  (seventh_grade_students.to_real / total_students.to_real) = 1 / 2 := 
by 
  sorry

theorem part2_prob_one_seventh_one_eighth : 
  ((seventh_grade_students.to_real * eighth_grade_students.to_real * 2) / (total_students * (total_students - 1)).to_real) = 2 / 3 := 
by 
  sorry

end part1_prob_seventh_grade_part2_prob_one_seventh_one_eighth_l74_74404


namespace justify_buying_skates_l74_74883

noncomputable def admission_fee : ℝ := 5
noncomputable def rental_fee_per_visit : ℝ := 2.5
noncomputable def cost_of_new_skates : ℝ := 65
noncomputable def sales_tax_rate : ℝ := 0.09
noncomputable def lifespan_of_skates_in_years : ℕ := 2

noncomputable def total_cost_of_skates : ℝ := 
  cost_of_new_skates + cost_of_new_skates * sales_tax_rate

noncomputable def visits_to_justify_purchase : ℝ := 
  total_cost_of_skates / rental_fee_per_visit

theorem justify_buying_skates : 
  ceil visits_to_justify_purchase = 29 :=
by
  sorry

end justify_buying_skates_l74_74883


namespace brooke_earns_144_dollars_l74_74552

noncomputable def total_revenue : ℝ := 144

theorem brooke_earns_144_dollars 
  (price_milk_per_gallon : ℝ)
  (gallons_to_sticks : ℝ)
  (price_butter_per_stick : ℝ)
  (num_cows : ℕ)
  (milk_per_cow : ℝ)
  (num_customers : ℕ)
  (milk_per_customer : ℝ)
  (total_gallons_of_milk : ℝ := num_cows * milk_per_cow)  
  (total_revenue_from_milk : ℝ := total_gallons_of_milk * price_milk_per_gallon)
  (total_sticks_of_butter : ℝ := total_gallons_of_milk * gallons_to_sticks)
  (total_revenue_from_butter : ℝ := total_sticks_of_butter * price_butter_per_stick) : 
  total_revenue = total_revenue_from_milk :=
by
  sorry

@[simp] def assumptions := 
  (price_milk_per_gallon = 3) ∧ 
  (gallons_to_sticks = 2) ∧ 
  (price_butter_per_stick = 1.5) ∧ 
  (num_cows = 12) ∧ 
  (milk_per_cow = 4) ∧ 
  (num_customers = 6) ∧ 
  (milk_per_customer = 6)

example : brooke_earns_144_dollars 3 2 1.5 12 4 6 6 := 
by
  simp [assumptions]
  sorry

end brooke_earns_144_dollars_l74_74552


namespace sequence_has_infinitely_many_primes_l74_74786
-- Importing necessary libraries

-- Defining the sequence as described
noncomputable def x : ℕ → ℤ
| 0     := 0  -- x_0 is not defined in base problem, so we set it to 0 to avoid issues with n = 0.
| 1     := if 0 ≤ 1 ∧ 1 < 204 then 1 else 0  -- x1 is some integer less than 204
| (m+2) := let n := m + 1 in
           (↑n / 2004 + 1 / ↑n) * (x (m + 1))^2 - ↑n^3 / 2004 + 1

-- The main theorem to prove the sequence contains infinitely many primes
theorem sequence_has_infinitely_many_primes :
  ∃ (f : ℕ → ℕ) (hf : function.injective f), ∀ n, nat.prime (x (f n) ) := by
  sorry

end sequence_has_infinitely_many_primes_l74_74786


namespace triangle_side_length_c_l74_74264

theorem triangle_side_length_c (a b c : ℕ) (h1 : |a - 7| + (b - 2) ^ 2 = 0) (h2 : c % 2 = 1) :
  c = 7 :=
sorry

end triangle_side_length_c_l74_74264


namespace jen_age_proof_l74_74124

-- Definitions
def son_age := 16
def son_present_age := son_age
def jen_present_age := 41

-- Conditions
axiom jen_older_25 (x : ℕ) : ∀ y : ℕ, x = y + 25 → y = son_present_age
axiom jen_age_formula (j s : ℕ) : j = 3 * s - 7 → j = son_present_age + 25

-- Proof problem statement
theorem jen_age_proof : jen_present_age = 41 :=
by
  -- Declare variables
  let j := jen_present_age
  let s := son_present_age
  -- Apply conditions (in Lean, sorry will skip the proof)
  sorry

end jen_age_proof_l74_74124


namespace quadratic_trinomial_unique_l74_74226

theorem quadratic_trinomial_unique
  (a b c : ℝ)
  (h1 : b^2 - 4*(a+1)*c = 0)
  (h2 : (b+1)^2 - 4*a*c = 0)
  (h3 : b^2 - 4*a*(c+1) = 0) :
  a = 1/8 ∧ b = -3/4 ∧ c = 1/8 :=
by
  sorry

end quadratic_trinomial_unique_l74_74226


namespace total_payment_l74_74543

theorem total_payment (n_choco n_straw : ℕ) (p_choco p_straw : ℕ) (h1 : n_choco = 3) (h2 : p_choco = 12) (h3 : n_straw = 6) (h4 : p_straw = 22) :
  (n_choco * p_choco + n_straw * p_straw) = 168 :=
by {
  -- variables used in the proof
  -- n_choco: number of chocolate cakes
  -- p_choco: price per chocolate cake
  -- n_straw: number of strawberry cakes
  -- p_straw: price per strawberry cake
  rw [h1, h2, h3, h4],
  norm_num,
  sorry
}

end total_payment_l74_74543


namespace convex_quadrilateral_division_l74_74051

-- Definitions for convex quadrilateral and some basic geometric objects.
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)
  (convex : ∀ (X Y Z : Point), (X ≠ Y) ∧ (Y ≠ Z) ∧ (Z ≠ X))

-- Definitions for lines and midpoints.
def is_midpoint (M X Y : Point) : Prop :=
  M.x = (X.x + Y.x) / 2 ∧ M.y = (X.y + Y.y) / 2

-- Preliminary to determining equal area division.
def equal_area_division (Q : Quadrilateral) (L : Point → Point → Prop) : Prop :=
  ∃ F,
    is_midpoint F Q.A Q.B ∧
    -- Assuming some way to relate area with F and L
    L Q.D F ∧
    -- Placeholder for equality of areas (details depend on how we calculate area)
    sorry

-- Problem statement in Lean 4
theorem convex_quadrilateral_division (Q : Quadrilateral) :
  ∃ L, equal_area_division Q L :=
by
  -- Proof will be constructed here based on steps in the solution
  sorry

end convex_quadrilateral_division_l74_74051


namespace shifted_parabola_is_correct_l74_74753

theorem shifted_parabola_is_correct (x : ℝ) :
  let y := 3 * x ^ 2 in
  let y_up := y + 2 in
  let y_right := (x - 3) ^ 2 in
  3 * y_right + 2 = 3 * (x - 3) ^ 2 + 2 :=
by 
  sorry

end shifted_parabola_is_correct_l74_74753


namespace range_of_independent_variable_l74_74357

theorem range_of_independent_variable
  (x : ℝ) 
  (h1 : 2 - 3*x ≥ 0) 
  (h2 : x ≠ 0) 
  : x ≤ 2/3 ∧ x ≠ 0 :=
by 
  sorry

end range_of_independent_variable_l74_74357


namespace katie_bead_necklaces_l74_74707

theorem katie_bead_necklaces (B : ℕ) (gemstone_necklaces : ℕ := 3) (cost_each_necklace : ℕ := 3) (total_earnings : ℕ := 21) :
  gemstone_necklaces * cost_each_necklace + B * cost_each_necklace = total_earnings → B = 4 :=
by
  intro h
  sorry

end katie_bead_necklaces_l74_74707


namespace quarters_count_l74_74038

theorem quarters_count (total_money : ℝ) (value_of_quarter : ℝ) (h1 : total_money = 3) (h2 : value_of_quarter = 0.25) : total_money / value_of_quarter = 12 :=
by sorry

end quarters_count_l74_74038


namespace find_n_l74_74233

theorem find_n (n : ℕ) (h1 : 0 ≤ n ∧ n ≤ 360) (h2 : Real.cos (n * Real.pi / 180) = Real.cos (340 * Real.pi / 180)) : 
  n = 20 ∨ n = 340 := 
by
  sorry

end find_n_l74_74233


namespace sum_of_sines_greater_than_two_l74_74417

theorem sum_of_sines_greater_than_two (α β γ : ℝ) (h1 : α + β + γ = π) 
  (h2 : α < π / 2) (h3 : β < π / 2) (h4 : γ < π / 2) : 
  sin α + sin β + sin γ > 2 := 
sorry

end sum_of_sines_greater_than_two_l74_74417


namespace log_x_64_eq_3_imp_x_eq_4_l74_74928

theorem log_x_64_eq_3_imp_x_eq_4 (x : ℝ) (h : Real.log x 64 = 3) : x = 4 :=
by
  sorry

end log_x_64_eq_3_imp_x_eq_4_l74_74928


namespace exists_face_with_perpendicular_foot_interior_l74_74958

open Set

-- Definitions
variable (P : Set Point) [Convex P] (p : Point)

-- Assumption
axiom point_inside_polyhedron : p ∈ interior P

-- Theorem (goal)
theorem exists_face_with_perpendicular_foot_interior : ∃ (F : Face), foot_of_perpendicular p F ∈ interior F := sorry

end exists_face_with_perpendicular_foot_interior_l74_74958


namespace find_S₃₀_l74_74967

-- Define arithmetic sequence and sum of the first n terms
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, a i)

-- Given conditions
def S₁₀ := ∫ x in 1..Real.exp 1, Real.log x
def S₂₀ := 17

lemma integral_ln_x : ∫ x in 1..Real.exp 1, Real.log x = 1 := by
  calc
    ∫ x in 1..Real.exp 1, Real.log x = (Real.exp 1 * Real.log (Real.exp 1) - Real.exp 1) - (1 * Real.log 1 - 1) : by apply intervalIntegral.integral_primitive

-- State the problem in Lean 4
theorem find_S₃₀ (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) (hS₁₀ : S₁₀ = 1) (hS₂₀ : sequence_sum a 20 = S₂₀) :
  sequence_sum a 30 = 48 := by
  sorry

end find_S₃₀_l74_74967


namespace person_speed_in_mph_l74_74532

def meters_to_miles (meters : ℝ) : ℝ := meters / 1609.34
def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60
def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem person_speed_in_mph :
  let distance_meters := 2500
  let time_minutes := 8
  let distance_miles := meters_to_miles distance_meters
  let time_hours := minutes_to_hours time_minutes
  let computed_speed := speed distance_miles time_hours
  computed_speed ≈ 11.65 := sorry

end person_speed_in_mph_l74_74532


namespace two_digit_powers_of_three_l74_74313

theorem two_digit_powers_of_three : 
  (Finset.filter (λ n : ℕ, 10 ≤ 3^n ∧ 3^n ≤ 99) (Finset.range 6)).card = 3 := by 
sorry

end two_digit_powers_of_three_l74_74313


namespace worker_overtime_hours_l74_74869

theorem worker_overtime_hours :
  ∃ (x y : ℕ), 60 * x + 90 * y = 3240 ∧ x + y = 50 ∧ y = 8 :=
by
  sorry

end worker_overtime_hours_l74_74869


namespace draw_sequence_count_l74_74795

noncomputable def total_sequences : ℕ :=
  (Nat.choose 4 3) * (Nat.factorial 4) * 5

theorem draw_sequence_count : total_sequences = 480 := by
  sorry

end draw_sequence_count_l74_74795


namespace angle_B_leq_60_l74_74688

/-- In an acute-angled triangle, if the greatest altitude is equal to the median, then the angle opposite to the median is less than or equal to 60 degrees. -/
theorem angle_B_leq_60 (A B C H M : Type) [acute_triangle ABC] 
  (H_AH_eq_BM : greatest_altitude A B C H = median B M) : 
  ∠ B ≤ 60 :=
by
  sorry

end angle_B_leq_60_l74_74688


namespace exists_root_in_interval_l74_74710

theorem exists_root_in_interval (n : ℕ) (f : polynomial ℝ)
  (hn : n ≥ 3)
  (hrealroots : ∀ r : ℝ, f.root r → r ∈ set.range (λ i : fin n.succ, (r : ℝ)))
  (ha_ratio : (polynomial.coeff f (n - 1) / polynomial.coeff f n) > (n : ℝ) + 1)
  (ha_n_minus_2 : polynomial.coeff f (n-2) = 0) :
  ∃ r : ℝ, f.root r ∧ r ∈ Ioo (-1/2 : ℝ) (1 / ((n : ℝ) + 1)) :=
sorry

end exists_root_in_interval_l74_74710


namespace max_tickets_l74_74216

theorem max_tickets (n : ℕ) (H : 15 * n ≤ 120) : n ≤ 8 :=
by sorry

end max_tickets_l74_74216


namespace solve_exponential_problem_l74_74781

noncomputable def satisfies_condition (a : ℝ) : Prop :=
  let max_value := if a > 1 then a^2 else a
  let min_value := if a > 1 then a else a^2
  max_value - min_value = a / 2

theorem solve_exponential_problem (a : ℝ) (hpos : a > 0) (hne1 : a ≠ 1) :
  satisfies_condition a ↔ (a = 1 / 2 ∨ a = 3 / 2) :=
sorry

end solve_exponential_problem_l74_74781


namespace man_receives_dividend_l74_74528

-- Define the given conditions
def investment : ℝ := 14400
def face_value_per_share : ℝ := 100
def premium_rate : ℝ := 0.20
def dividend_rate : ℝ := 0.06

-- Calculate the cost per share including the premium
def cost_per_share := face_value_per_share * (1 + premium_rate)

-- Calculate the number of shares bought
def number_of_shares := investment / cost_per_share

-- Calculate the dividend per share
def dividend_per_share := face_value_per_share * dividend_rate

-- Calculate the total dividend received
def total_dividend := dividend_per_share * number_of_shares

-- State the theorem
theorem man_receives_dividend : total_dividend = 720 := by
  sorry

end man_receives_dividend_l74_74528


namespace max_value_of_ab_over_3a_plus_b_l74_74631

open Real

theorem max_value_of_ab_over_3a_plus_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h : 9 * a^2 + b^2 = 1) :
  ∃ m, (∀ x y, 0 < x → 0 < y → 9 * x^2 + y^2 = 1 → (x * y) / (3 * x + y) ≤ m) ∧ 
  ((a * b) / (3 * a + b) = m) ∧ (m = sqrt 2 / 12) :=
sorry

end max_value_of_ab_over_3a_plus_b_l74_74631


namespace construct_parallelogram_l74_74998

noncomputable def midpoint (A C : ℝ × ℝ) : (ℝ × ℝ) :=
((A.1 + C.1) / 2, (A.2 + C.2) / 2)

noncomputable def reflection (M P : ℝ × ℝ) : (ℝ × ℝ) :=
(2 * M.1 - P.1, 2 * M.2 - P.2)

theorem construct_parallelogram (A C : ℝ × ℝ) (Γ : set (ℝ × ℝ)) :
  ∃ B D : ℝ × ℝ, B ∈ Γ ∧ D ∈ Γ ∧
  midpoint A C = midpoint B D ∧
  let M := midpoint A C in
  B = reflection M D ∧
  D = reflection M B :=
by
  sorry

end construct_parallelogram_l74_74998


namespace domain_sqrt_tan_sub_one_l74_74075

theorem domain_sqrt_tan_sub_one (k : ℤ) : 
  (∃ x : ℝ, k * real.pi + real.pi / 4 ≤ x ∧ x < k * real.pi + real.pi / 2) ↔ 
  (∃ f : ℝ → ℝ, f = λ x, real.sqrt (real.tan x - 1) ∧ real.tan x ≥ 1) :=
begin
  sorry
end

end domain_sqrt_tan_sub_one_l74_74075


namespace sum_of_digits_of_reciprocal_five_pow_sum_of_digits_power_2_eq_5_l74_74946

theorem sum_of_digits_of_reciprocal_five_pow {n : ℕ} : 
  (let d := 1 / (5 ^ n); sum_of_digits d = 5) ↔ n = 5 := 
by sorry

-- Define a helper function to compute the sum of digits
def sum_of_digits (n : ℕ) : ℕ := 
  (n.toString.foldl (λ acc c, acc + c.toNat - '0'.toNat) 0)

-- State the theorem in terms of sum of digits of 2^n since it is equivalent
theorem sum_of_digits_power_2_eq_5 (n : ℕ) : 
    sum_of_digits (2 ^ n) = 5 ↔ n = 5 := 
by sorry

end sum_of_digits_of_reciprocal_five_pow_sum_of_digits_power_2_eq_5_l74_74946


namespace arithmetic_geometric_sequence_problem_l74_74966

section arithmetic_geometric

variable {a_n b_n c_n : ℕ → ℝ}
variable {λ : ℝ}
variable {q : ℝ}

/-- The sum of the first n terms of an arithmetic sequence --/
def S_n (n : ℕ) : ℝ := ∑ i in finset.range n, a_n i

theorem arithmetic_geometric_sequence_problem 
  (a1 : a_n 1 = 2) 
  (b1 : b_n 1 = 1)
  (S2 : S_n 2 = 3 * b_n 2) 
  (a2_eq_b3 : a_n 2 = b_n 3)
  (h_arith : ∀ n, a_n n = 2 * n)
  (h_geo : ∀ n, b_n n = 2^(n-1))
  : 
  (∀ n, c_n n = 2 * b_n n - λ * 3^(a_n n / 2) → c_n (n+1) < c_n n → λ > 1/3) :=
begin
  sorry,
end

end arithmetic_geometric

end arithmetic_geometric_sequence_problem_l74_74966


namespace totalCroissants_is_18_l74_74334

def jorgeCroissants : ℕ := 7
def giulianaCroissants : ℕ := 5
def matteoCroissants : ℕ := 6

def totalCroissants : ℕ := jorgeCroissants + giulianaCroissants + matteoCroissants

theorem totalCroissants_is_18 : totalCroissants = 18 := by
  -- Proof will be provided here
  sorry

end totalCroissants_is_18_l74_74334


namespace tangent_line_condition_min_value_c_l74_74282

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.cos x + b * Real.exp x

theorem tangent_line_condition (a b : ℝ) (H : ∀ x, f a b x = a * Real.cos x + b * Real.exp x) : 
  (f a b 0 = a + b) ∧ (f a b 0 = -0) ∧ (f' a b 0 = b) ∧ (f' a b 0 = -1) → 
  a = 1 ∧ b = -1 :=
by 
  intros; sorry

theorem min_value_c (c : ℤ) (H1 : ∀ x ∈ set.Icc (-Real.pi / 2) (∈) ∞, f 1 (-1) x ≤ c) : 
  c = 1 :=
by 
  intros; sorry

end tangent_line_condition_min_value_c_l74_74282


namespace semicircle_perimeter_is_35_12_l74_74128

-- Define the radius of the semicircle
def radius : ℝ := 6.83

-- Function to calculate the perimeter of a semicircle
def semicirclePerimeter (r : ℝ) : ℝ :=
  π * r + 2 * r

-- The property we want to prove
theorem semicircle_perimeter_is_35_12 :
  semicirclePerimeter radius ≈ 35.12 := sorry

end semicircle_perimeter_is_35_12_l74_74128


namespace calculate_exponent_l74_74893

theorem calculate_exponent (m : ℝ) : (243 : ℝ)^(1 / 3) = 3^m → m = 5 / 3 :=
by
  sorry

end calculate_exponent_l74_74893


namespace boxes_containing_pans_l74_74059

def num_boxes : Nat := 26
def num_teacups_per_box : Nat := 20
def num_cups_broken_per_box : Nat := 2
def teacups_left : Nat := 180

def num_teacup_boxes (num_boxes : Nat) (num_teacups_per_box : Nat) (num_cups_broken_per_box : Nat) (teacups_left : Nat) : Nat :=
  teacups_left / (num_teacups_per_box - num_cups_broken_per_box)

def num_remaining_boxes (num_boxes : Nat) (num_teacup_boxes : Nat) : Nat :=
  num_boxes - num_teacup_boxes

def num_pans_boxes (num_remaining_boxes : Nat) : Nat :=
  num_remaining_boxes / 2

theorem boxes_containing_pans : ∀ (num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left : Nat),
  num_boxes = 26 →
  num_teacups_per_box = 20 →
  num_cups_broken_per_box = 2 →
  teacups_left = 180 →
  num_pans_boxes (num_remaining_boxes num_boxes (num_teacup_boxes num_boxes num_teacups_per_box num_cups_broken_per_box teacups_left)) = 8 :=
by
  intros
  sorry

end boxes_containing_pans_l74_74059


namespace circle_chords_area_sum_l74_74682

theorem circle_chords_area_sum (r m n d AE BE CE DE E : ℝ) (radius_30 : r = 30)
  (chord_50_A : m = 50) (chord_50_B : n = 50) (distance_center_14 : d = 14)
  (AE_eq_CE : AE = CE) (BE_eq_DE : BE = DE) (AE_lt_BE : AE < BE)
  (condition : ∀ (m n d : ℕ), (∃ m n d : ℕ, m * π - n * sqrt d)) : (m + n + d = 162) :=
begin
  sorry
end

end circle_chords_area_sum_l74_74682


namespace problem_sum_value_l74_74633

open_locale big_operators

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f (x)

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

variable (f : ℝ → ℝ)
variable (h₁ : is_even_function f)
variable (h₂ : f 2 = -1)
variable (h₃ : is_odd_function (λ x, f (x - 1)))

theorem problem_sum_value :
  (∑ k in finset.range 2010, f (k + 1)) = -1 :=
sorry

end problem_sum_value_l74_74633


namespace find_geckos_sold_last_year_l74_74891

-- Define the conditions
def geckos_last_year (G : ℕ) := 
  let geckos_year_before := 2 * G in
  G + geckos_year_before = 258

-- Define the main statement
theorem find_geckos_sold_last_year (G : ℕ) (h : geckos_last_year G) : G = 86 :=
  by
    sorry

end find_geckos_sold_last_year_l74_74891


namespace ωB_ωC_intersect_on_BC_l74_74023

variable {A B C M N X : Type}

-- Declaring the given conditions as hypotheses
variables [Triangle A B C]
variables (M_midpoint_AB : Midpoint A B M)
variables (N_midpoint_AC : Midpoint A C N)
variables (AX_tangent_circumcircle_ABC : TangentCircle A X (Circumcircle A B C))
variables (ω_B_circle_through_M_B_tangent_MX : CircleThroughAndTangent M B (LineSegment M X))
variables (ω_C_circle_through_N_C_tangent_NX : CircleThroughAndTangent N C (LineSegment N X))

-- The theorem statement to be proven
theorem ωB_ωC_intersect_on_BC :
  IntersectLine (Circle ω_B) (Circle ω_C) (Line B C) :=
sorry

end ωB_ωC_intersect_on_BC_l74_74023


namespace arithmetic_series_sum_l74_74602

theorem arithmetic_series_sum :
  let a1 := 20
  let d := (1 : ℝ) / 5
  let an := 40
  let n := ((an - a1) / d).toNat + 1
  (n * (a1 + an)) / 2 = 3030 :=
by
  let a1 := 20
  let d := (1 : ℝ) / 5
  let an := 40
  let n := ((an - a1) / d).toNat + 1
  have : n = 101 := by sorry
  calc
    (n * (a1 + an)) / 2 = (101 * (20 + 40)) / 2 : by rw [this]
                    ... = 3030                     : by norm_num

end arithmetic_series_sum_l74_74602


namespace cycles_reappear_at_line_40_l74_74081

theorem cycles_reappear_at_line_40:
  let letters := "BKLRSTUQ"
  let digits := "20203"
  (∀ n : ℕ, (cycle (letters.toList) n).asString = "BKLRSTUQ" → n % 8 = 0)
  →
  (∀ n : ℕ, (cycle (digits.toList) n).asString = "20203" → n % 5 = 0)
  →
  ∃ k: ℕ, k = 40 ∧ (cycle (letters.toList) 40).asString = "BKLRSTUQ" ∧ (cycle (digits.toList) 40).asString = "20203".
Proof.
  sorry.

end cycles_reappear_at_line_40_l74_74081


namespace min_value_l74_74235

theorem min_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (∃ x y, x > 2 ∧ y > 2 ∧ (x = 3 ∧ y = 3) ∧
    ( ( (x^2 + 2*x) / (y - 2) ) + ( (y^2 + 2*y) / (x - 2) ) = 22 )) :=
by {
  use [3, 3],
  split,
  exact lt_add_of_pos_right 2 zero_lt_one,
  split,
  exact lt_add_of_pos_right 2 zero_lt_one,
  split,
  refl,
  split,
  refl,
  have hx_eq : 3 = 3 := rfl,
  have hy_eq : 3 = 3 := rfl,
  simp [hx_eq, hy_eq],
  sorry
}

end min_value_l74_74235


namespace petrol_consumption_reduction_l74_74328

theorem petrol_consumption_reduction :
  ∀ (P : ℝ) (maintenance_fraction : ℝ) (maintenance_increase : ℝ) (price_increase_A : ℝ) (price_increase_B : ℝ),
    maintenance_fraction = 0.30 →
    maintenance_increase = 0.10 →
    price_increase_A = 0.20 →
    price_increase_B = 0.15 →
    let petrol_increase := (price_increase_A + price_increase_B) / 2 in
    let total_increase := petrol_increase + maintenance_fraction * maintenance_increase in
    total_increase = 0.205 :=
begin
  intros P maintenance_fraction maintenance_increase price_increase_A price_increase_B,
  assume h_maf : maintenance_fraction = 0.30,
  assume h_mai : maintenance_increase = 0.10,
  assume h_pia : price_increase_A = 0.20,
  assume h_pib : price_increase_B = 0.15,
  let petrol_increase := (price_increase_A + price_increase_B) / 2,
  let total_increase := petrol_increase + maintenance_fraction * maintenance_increase,
  show total_increase = 0.205, from sorry
end

end petrol_consumption_reduction_l74_74328


namespace distance_town_a_to_town_b_approx_l74_74196

noncomputable def distance_town_a_to_town_b : ℝ :=
  let net_west_dist := (30 - 15 + 20 - 23 : ℝ)
  let net_north_dist := (12 - 25 + 22 + 35 : ℝ)
  real.sqrt (net_west_dist^2 + net_north_dist^2)

theorem distance_town_a_to_town_b_approx :
  distance_town_a_to_town_b ≈ 45.6 :=
by
  sorry

end distance_town_a_to_town_b_approx_l74_74196


namespace boys_skip_count_l74_74142

theorem boys_skip_count 
  (x y : ℕ)
  (avg_jumps_boys : ℕ := 85)
  (avg_jumps_girls : ℕ := 92)
  (avg_jumps_all : ℕ := 88)
  (h1 : x = y + 10)
  (h2 : (85 * x + 92 * y) / (x + y) = 88) : x = 40 :=
  sorry

end boys_skip_count_l74_74142


namespace standard_deviations_below_l74_74938

variable (σ : ℝ)
variable (mean : ℝ)
variable (score98 : ℝ)
variable (score58 : ℝ)

-- Conditions translated to Lean definitions
def condition_1 : Prop := score98 = mean + 3 * σ
def condition_2 : Prop := mean = 74
def condition_3 : Prop := σ = 8

-- Target statement: Prove that the score of 58 is 2 standard deviations below the mean
theorem standard_deviations_below : condition_1 σ mean score98 → condition_2 mean → condition_3 σ → score58 = 74 - 2 * σ :=
by
  intro h1 h2 h3
  sorry

end standard_deviations_below_l74_74938


namespace dot_product_l74_74637

open Real

variables (a b : ℝ^3)

-- Given conditions
axiom a_norm : ‖a‖ = 4
axiom b_norm : ‖b‖ = 5
axiom ab_norm : ‖a + b‖ = sqrt 21

-- Proof problem statement
theorem dot_product (a b : ℝ^3) (a_norm : ‖a‖ = 4) (b_norm : ‖b‖ = 5) (ab_norm : ‖a + b‖ = sqrt 21) : a ∙ b = -10 :=
sorry

end dot_product_l74_74637


namespace unique_intersection_a_l74_74658

theorem unique_intersection_a {a : ℝ} 
  (h : ({1, a, 5} ∩ {2, a^2 + 1}).card = 1) : a = 0 ∨ a = -2 :=
sorry

end unique_intersection_a_l74_74658


namespace max_value_of_f_prime_over_f_l74_74155

noncomputable def f (x : ℝ) : ℝ := sorry -- Define f following the given condition (unspecified due to proof omission)

theorem max_value_of_f_prime_over_f :
  (∀ x : ℝ, f' x - f x = x * exp x) ∧ (f 0 = 1/2) →
  (∀ x : ℝ, x * exp x = x^2 + 2 * x * f x / (x^2 + 1)) →
  (sup (λ x, f' x / f(x)) = 2) :=
sorry

end max_value_of_f_prime_over_f_l74_74155


namespace absolute_value_difference_base_six_l74_74592

theorem absolute_value_difference_base_six (C D : ℕ) (h1 : C < 6) (h2 : D < 6)
(h3 : C + D = 5) : |C - D| = 0 :=
by
  sorry

end absolute_value_difference_base_six_l74_74592


namespace parabola_directrix_l74_74437

theorem parabola_directrix (x y : ℝ) (h : y = 16 * x^2) : y = -1/64 :=
sorry

end parabola_directrix_l74_74437


namespace minimum_value_f_l74_74931

open Real

noncomputable def f (x : ℝ) : ℝ :=
  x + (3 * x) / (x^2 + 3) + (x * (x + 3)) / (x^2 + 1) + (3 * (x + 1)) / (x * (x^2 + 1))

theorem minimum_value_f (x : ℝ) (hx : x > 0) : f x ≥ 7 :=
by
  -- Proof omitted
  sorry

end minimum_value_f_l74_74931


namespace max_value_of_f_l74_74776

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 - 3 * x else -2 * x + 1

theorem max_value_of_f : ∃ x : ℝ, f x = 2 :=
by
  sorry

end max_value_of_f_l74_74776


namespace lucy_distance_nearest_quarter_mile_l74_74735

-- Define the conditions as assumptions in Lean
def sound_speed := 1100 -- feet per second
def time_interval := 15 -- seconds
def feet_per_mile := 5280 -- feet

-- Lean function to compute the approximate distance from the flash
def calculate_distance (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time

def distance_in_miles (distance_feet : ℕ) (conversion_factor : ℕ) : ℚ :=
  distance_feet / conversion_factor

theorem lucy_distance_nearest_quarter_mile :
  let distance_feet := calculate_distance sound_speed time_interval,
      distance_miles := distance_in_miles distance_feet feet_per_mile in
  distance_miles = 3.125 :=
by
  sorry

end lucy_distance_nearest_quarter_mile_l74_74735


namespace train_speed_in_kmph_l74_74865

def train_length : ℕ := 125
def time_to_cross_pole : ℕ := 9
def conversion_factor : ℚ := 18 / 5

theorem train_speed_in_kmph
  (d : ℕ := train_length)
  (t : ℕ := time_to_cross_pole)
  (cf : ℚ := conversion_factor) :
  d / t * cf = 50 := 
sorry

end train_speed_in_kmph_l74_74865


namespace cans_used_for_37_rooms_l74_74058

noncomputable def rooms_initial : ℕ := 50
noncomputable def rooms_after_misplacing : ℕ := 37
noncomputable def cans_misplaced : ℕ := 5
noncomputable def rooms_lost := rooms_initial - rooms_after_misplacing
noncomputable def rooms_per_can := rooms_lost / cans_misplaced       -- in this case 13/5
noncomputable def cans_for_37_rooms := rooms_after_misplacing / rooms_per_can -- in this case 37 / (13/5)

theorem cans_used_for_37_rooms : cans_for_37_rooms ≈ 15 := sorry

end cans_used_for_37_rooms_l74_74058


namespace intersect_P_M_l74_74014

def P : Set ℝ := {x | 0 ≤ x ∧ x < 3}
def M : Set ℝ := {x | |x| ≤ 3}

theorem intersect_P_M : (P ∩ M) = {x | 0 ≤ x ∧ x < 3} := by
  sorry

end intersect_P_M_l74_74014


namespace find_f_neg4_l74_74985

noncomputable def f : ℝ → ℝ
| x => if x < 3 then f (x + 2) else (1/2) ^ x

theorem find_f_neg4 : f (-4) = 1 / 16 := by
  sorry

end find_f_neg4_l74_74985


namespace angle_value_in_quadrilateral_l74_74691

theorem angle_value_in_quadrilateral
  (A B C D P : Type)
  [quadrilateral A B C D]
  (h1 : AB = AD)
  (h2 : BC = CD)
  (h3 : ∠APB = 90)
  (h4 : ∠BAP = ∠PAD)
  (h5 : ∠DCP = ∠PCB)
  (x : real)
  (hx1 : ∠BAP = x)
  (hx2 : ∠DCP = x) :
  x = 90 :=
sorry

end angle_value_in_quadrilateral_l74_74691


namespace part1_part2_l74_74988
open Real

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (hm : m > 1) : ∃ x : ℝ, f x = 4 / (m - 1) + m :=
by
  sorry

end part1_part2_l74_74988


namespace mens_wages_l74_74140

theorem mens_wages (total_earnings : ℝ) (total_boys_units : ℕ) (earnings_per_boy_unit : ℝ) (mens_wages : ℝ) :
  total_earnings = 150 ∧ total_boys_units = 16 ∧ earnings_per_boy_unit = total_earnings / total_boys_units ∧ mens_wages = earnings_per_boy_unit * 8 →
  mens_wages = 75 :=
begin
  sorry
end

end mens_wages_l74_74140


namespace distance_to_second_friend_is_correct_l74_74467

-- Definitions for the conditions
def distance_to_first_friend_d := 120 -- in miles
def time_to_first_friend_t := 3 -- in hours
def time_to_second_friend_t := 5 -- in hours

-- Assuming noncomputable speed as real calculation 
noncomputable def speed := distance_to_first_friend_d / time_to_first_friend_t

-- The main theorem to prove
theorem distance_to_second_friend_is_correct :
  let distance_to_first_friend_d := 120 in
  let time_to_first_friend_t := 3 in
  let time_to_second_friend_t := 5 in
  let speed := distance_to_first_friend_d / time_to_first_friend_t in
  let distance_to_second_friend_d := speed * time_to_second_friend_t in
  distance_to_second_friend_d = 200 :=
by
  sorry

end distance_to_second_friend_is_correct_l74_74467


namespace sufficient_not_necessary_condition_l74_74504

theorem sufficient_not_necessary_condition (a : ℝ) : (a = 2 → (a^2 - a) * 1 + 1 = 0) ∧ (¬ ((a^2 - a) * 1 + 1 = 0 → a = 2)) :=
by sorry

end sufficient_not_necessary_condition_l74_74504


namespace chord_length_of_circle_PA_dot_PB_l74_74980

noncomputable def polarLineEquation (ρ θ : ℝ) : ℝ := 
  3 * ρ * cos θ + ρ * sin θ - 6

noncomputable def circleParametricX (α : ℝ) : ℝ := 
  sqrt 5 * cos α

noncomputable def circleParametricY (α : ℝ) : ℝ := 
  1 + sqrt 5 * sin α

noncomputable def circleCartesian (x y : ℝ) : Prop :=
  x ^ 2 + (y - 1) ^ 2 = 5

noncomputable def lineCartesian (x y : ℝ) : Prop :=
  3 * x + y - 6 = 0

theorem chord_length_of_circle (x y : ℝ) (h1 : circleCartesian x y) (h2 : lineCartesian x y) : 
  2 * (sqrt(5 - 10 / 4)) = sqrt 10 / 2 := 
sorry

noncomputable def ParametricLineY (t θ : ℝ) : ℝ :=
  -2 + sin θ * t

theorem PA_dot_PB (t1 t2 θ : ℝ) (h : t1 * t2 - 6 * sin θ * t1 + 4 = 0) :
  t1 * t2 = 4 := 
sorry

end chord_length_of_circle_PA_dot_PB_l74_74980


namespace multiple_of_Jills_age_l74_74056

theorem multiple_of_Jills_age (m : ℤ) : 
  ∀ (J R F : ℤ),
  J = 20 →
  F = 40 →
  R = m * J + 5 →
  (R + 15) - (J + 15) = (F + 15) - 30 →
  m = 2 :=
by
  intros J R F hJ hF hR hDiff
  sorry

end multiple_of_Jills_age_l74_74056


namespace sum_cos_i_l74_74557

theorem sum_cos_i (i_pow_two : i * i = -1) :
  (finset.sum (finset.range 31) (λ n, (complex.i^n) * real.cos (30 + 120 * n))) = (23 - 15 * complex.I) * (real.sqrt 3 / 2) :=
sorry

end sum_cos_i_l74_74557


namespace unlock_probability_l74_74485

/--
Xiao Ming set a six-digit passcode for his phone using the numbers 0-9, but he forgot the last digit.
The probability that Xiao Ming can unlock his phone with just one try is 1/10.
-/
theorem unlock_probability (n : ℕ) (h : n ≥ 0 ∧ n ≤ 9) : 
  1 / 10 = 1 / (10 : ℝ) :=
by
  -- Skipping proof
  sorry

end unlock_probability_l74_74485


namespace correct_statement_3_l74_74876

/-- Statement ③ -/
def statement_3 (sample_size: ℕ) : Prop :=
  sample_size > 0 ∧ ∀ N P, estimateDistribution sample_size N P → accuracyImprovement sample_size N P

/-- The Proof problem -/
theorem correct_statement_3 (not_classical_prototype: ¬is_classical_prototype, 
    unequal_probabilities: ¬equal_probabilities_two_coins) : 
    ∃ sample_size > 0, statement_3 sample_size := 
  sorry

end correct_statement_3_l74_74876


namespace find_third_number_l74_74456

theorem find_third_number (first_number second_number third_number : ℕ) 
  (h1 : first_number = 200)
  (h2 : first_number + second_number + third_number = 500)
  (h3 : second_number = 2 * third_number) :
  third_number = 100 := sorry

end find_third_number_l74_74456


namespace intersection_of_A_and_B_l74_74294

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l74_74294


namespace correct_statements_among_given_l74_74547

-- Define each condition based on the given problem
def mode_data_is_4_and_6 : Prop := mode [4, 6, 6, 7, 9, 4] = {4, 6}
def mean_mode_median_central_tendency : Prop := True
def mean_center_of_gravity_histogram : Prop := True
def area_of_each_small_rectangle_frequency_rate_not_frequency : Prop := True

-- Problem statement to prove that (2) and (3) are the correct statements
theorem correct_statements_among_given :
    mean_mode_median_central_tendency ∧ mean_center_of_gravity_histogram ∧ ¬area_of_each_small_rectangle_frequency_rate_not_frequency := by
  sorry

end correct_statements_among_given_l74_74547


namespace distinct_paths_from_C_to_D_l74_74666

-- Definitions based on conditions
def grid_rows : ℕ := 7
def grid_columns : ℕ := 8
def total_steps : ℕ := grid_rows + grid_columns -- 15 in this case
def steps_right : ℕ := grid_columns -- 8 in this case

-- Theorem statement
theorem distinct_paths_from_C_to_D :
  Nat.choose total_steps steps_right = 6435 :=
by
  -- The proof itself
  sorry

end distinct_paths_from_C_to_D_l74_74666


namespace probability_of_conditions_probability_rectangle_l74_74533

noncomputable def probability (x y : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 3 ∧ x < y ∧ x + y < 5
then 1
else 0

theorem probability_of_conditions : 
  (λ (x y : ℝ), if 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 3 ∧ x < y ∧ x + y < 5 then 1 else 0) = λ x y, probability x y := 
begin
  sorry
end

theorem probability_rectangle : 
  let rect_area := 12 in
  let tri_area := (3 * 3) / 2 in
  let prob := tri_area / rect_area in
  prob = 3 / 8 :=
by
  let rect_area := 12
  let tri_area := (3 * 3) / 2
  let prob := tri_area / rect_area
  exact eq.refl 3 / 8

end probability_of_conditions_probability_rectangle_l74_74533


namespace closest_whole_number_of_ratio_l74_74555

theorem closest_whole_number_of_ratio :
  (Int.toNat (Real.round ((10^3000 + 10^3004) / (2 * 10^3002)))) = 50 := 
sorry

end closest_whole_number_of_ratio_l74_74555


namespace parabola_equation_l74_74352

noncomputable def point : (ℝ × ℝ) := (1, 1)

def bisector_eq (x y : ℝ) : Prop := x + y - 1 = 0

def parabola (p : ℝ) : Prop := y^2 = 2 * p * x

theorem parabola_equation (h : bisector_eq (p/2) 0) (h_pos : p > 0) : parabola 2 :=
by 
  sorry

end parabola_equation_l74_74352


namespace greatest_common_divisor_of_180_and_n_l74_74105

theorem greatest_common_divisor_of_180_and_n {n : ℕ} (h :  ∃ (d : set ℕ), d = {1, 3, 9} ∧ ∀ a ∈ d, a ∣ 180 ∧ a ∣ n ) : 
  (∀ m ∈ {1, 3, 9}, m ≤ 9) := by sorry

end greatest_common_divisor_of_180_and_n_l74_74105


namespace angle_B_sin_C_l74_74333

variables {A B C a b c : ℝ}
hypothesis (h1 : a * sin (2 * B) = sqrt 3 * b * sin A)
hypothesis (h2 : cos A = 1 / 3)

theorem angle_B (h1 : a * sin (2 * B) = sqrt 3 * b * sin A) : B = π / 6 :=
sorry

theorem sin_C (h2 : cos A = 1 / 3) (B : ℝ) (hB : B = π / 6) : sin C = (2 * sqrt 6 + 1) / 6 :=
sorry

end angle_B_sin_C_l74_74333


namespace probability_X_k_l74_74161

def number_of_keys : ℕ := n

def key_opens_door (key : ℕ) : Prop := 
key = selected_key -- Only one key can open the door

def event_A (k : ℕ) : Prop := 
X = k -- Event that the door is successfully opened on the k-th attempt

def event_B (k : ℕ) : Prop := 
X > k -- Event that the door fails to open on the k-th attempt

theorem probability_X_k (X : ℕ → ℕ) (n : ℕ) (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) :
  (∃! k, key_opens_door k) → -- Only one key can open the door
  (∀ i, event_A(i) ∨ event_B(i)) → -- Trying each key one by one
  X = k → 
  P(X = k) = 1 / n :=
sorry

end probability_X_k_l74_74161


namespace remainder_when_divided_by_296_l74_74856

theorem remainder_when_divided_by_296 (N : ℤ) (Q : ℤ) (R : ℤ)
  (h1 : N % 37 = 1)
  (h2 : N = 296 * Q + R)
  (h3 : 0 ≤ R) 
  (h4 : R < 296) :
  R = 260 := 
sorry

end remainder_when_divided_by_296_l74_74856


namespace tan_theta_monotone_l74_74299

theorem tan_theta_monotone {θ : ℝ} (h1 : (sin θ + cos θ, 1) ∙ (5, 1) = 0) (h2 : 0 < θ) (h3 : θ < real.pi) :
  tan θ = -3/4 :=
by
  sorry

end tan_theta_monotone_l74_74299


namespace find_a_b_l74_74935

theorem find_a_b : 
  ∃ (a b : ℕ), a < b ∧ a > 0 ∧ b > 0 ∧ (sqrt (1 + sqrt (25 + 20 * (sqrt 3))) = sqrt a + sqrt b) :=
by
  use 1
  use 3
  sorry

end find_a_b_l74_74935


namespace wickets_before_last_match_l74_74119

theorem wickets_before_last_match (R W : ℕ) 
  (initial_average : ℝ) (runs_last_match wickets_last_match : ℕ) (average_decrease : ℝ)
  (h_initial_avg : initial_average = 12.4)
  (h_last_match_runs : runs_last_match = 26)
  (h_last_match_wickets : wickets_last_match = 5)
  (h_avg_decrease : average_decrease = 0.4)
  (h_initial_runs_eq : R = initial_average * W)
  (h_new_average : (R + runs_last_match) / (W + wickets_last_match) = initial_average - average_decrease) :
  W = 85 :=
by
  sorry

end wickets_before_last_match_l74_74119


namespace find_x_l74_74925

theorem find_x (x : ℝ) (h : log x 64 = 3) : x = 4 :=
sorry

end find_x_l74_74925


namespace derivative_of_f_tangent_line_at_P_l74_74986

noncomputable def f (x : ℝ) := x^2 + x * Real.log x

theorem derivative_of_f :
  (fun x => Derivative (fun x => x^2 + x * Real.log x) x = 2*x + Real.log x + 1) :=
by
  intro x
  calc
    Derivative (fun x => x^2 + x * Real.log x) x =
    2*x + Real.log x + 1 := sorry

theorem tangent_line_at_P :
  (fun x y : ℝ => x = 1 → y = 1 → Derivative (f 1) = 3 ∧ y - 1 = 3 * (x - 1) → y = 3*x - 2) :=
by
  intros x y hx hy
  have deriv_at_1 : (f 1) = 3 
  from
    calc
      (Derivative (f 1) = 2*1 + Real.log 1 + 1)
      = 2 + 0 + 1
      = 3  := sorry

  have tangent_line_eq : y - 1 = 3 * (x - 1)
  from
    calc
      y - 1 = 3 * (x - 1) := sorry

  show y = 3 * x - 2
    from
      calc
        y - 1 = 3 * (x - 1)
          ... = 3 * x - 3 * 1
          ... = 3 * x - 3
          ... + 1 = 3 * x - 2 := sorry

end derivative_of_f_tangent_line_at_P_l74_74986


namespace gears_cannot_rotate_l74_74340

/-- In a plane, 11 gears are arranged in such a way that
    the first is meshed with the second, the second with the third, ...,
    and the eleventh with the first. -/
def eleven_gears := list ℤ → Prop :=
λ gears, gears.length = 11 ∧
  ∀ i, 0 ≤ i ∧ i < 11 → gears.nth i ≠ none → gears.nth i = some 1 ∨ gears.nth i = some 2

/-- Gears rotate in opposite directions if they are meshed together. -/
def opposite_directions (g1 g2 : ℤ) : Prop := g1 * g2 = -1

theorem gears_cannot_rotate (gears : list ℤ) (h_loop : eleven_gears gears) :
  ∃ i, 0 ≤ i ∧ i < 11 ∧ gears.nth i = some 1 ∧ gears.nth (i % 11) = some (-1) :=
sorry

end gears_cannot_rotate_l74_74340


namespace hyperbola_other_asymptote_l74_74049

-- Define the problem conditions
def one_asymptote (x y : ℝ) : Prop := y = 2 * x
def foci_x_coordinate : ℝ := -4

-- Define the equation of the other asymptote
def other_asymptote (x y : ℝ) : Prop := y = -2 * x - 16

-- The statement to be proved
theorem hyperbola_other_asymptote : 
  (∀ x y, one_asymptote x y) → (∀ x, x = -4 → ∃ y, ∃ C, other_asymptote x y ∧ y = C + -2 * x - 8) :=
by
  sorry

end hyperbola_other_asymptote_l74_74049


namespace parametric_curve_C2_minimum_distance_M_C_l74_74272

-- Given conditions
def curve_C (ρ θ : ℝ) : Prop := 2 * ρ * sin (θ + ρ) * cos θ = 10
def curve_C1 (α : ℝ) : (ℝ × ℝ) := (cos α, sin α)
def scaling_transformation (x y : ℝ) : (ℝ × ℝ) := (3 * x, 2 * y)
def curve_C2 (θ : ℝ) : (ℝ × ℝ) := (3 * cos θ, 2 * sin θ)

-- 1. We prove the parametric equations of curve_C2
theorem parametric_curve_C2 (θ : ℝ) : 
curve_C2 θ = (3 * cos θ, 2 * sin θ) := 
sorry

-- 2. We prove the minimum distance between point M and curve C
def point_M (θ : ℝ) : (ℝ × ℝ) := (3 * cos θ, 2 * sin θ)
def curve_C_equation (M : ℝ × ℝ) : ℝ := (2 * M.snd + M.fst)

theorem minimum_distance_M_C : 
(∀ θ : ℝ, (curve_C_equation (point_M θ) - 10).abs / sqrt 5 >= sqrt 5) :=
sorry

end parametric_curve_C2_minimum_distance_M_C_l74_74272


namespace least_possible_b_prime_l74_74344

theorem least_possible_b_prime :
  ∃ b a : ℕ, Nat.Prime a ∧ Nat.Prime b ∧ 2 * a + b = 180 ∧ a > b ∧ b = 2 :=
by
  sorry

end least_possible_b_prime_l74_74344


namespace sum_of_values_sum_of_all_possible_values_l74_74316

theorem sum_of_values (x : ℝ) (hx : x^2 = 36) : x = 6 ∨ x = -6 :=
begin
  simp [hx],
  have h₁ : x = 6 ∨ x = -6, from or.intro_right (x = 6) rfl,
  exact h₁,
end

noncomputable def sum_of_possible_values : ℝ :=
if hx : ∃ x : ℝ, x^2 = 36 then 6 + (-6) else 0

theorem sum_of_all_possible_values : sum_of_possible_values = 0 :=
begin
  simp [sum_of_possible_values],
  sorry
end

end sum_of_values_sum_of_all_possible_values_l74_74316


namespace find_k_values_l74_74608

theorem find_k_values (k : ℕ) (h : k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 8) :
  ∃ (a b : ℕ), 2^a * 3^b = k * (k + 1) :=
by {
  cases h;
  { use (0, 0), use (1, 1), use (2, 1), use (3, 2) },
  { exact dec_trivial },
}

end find_k_values_l74_74608


namespace cricketer_hits_two_sixes_l74_74846

-- Definitions of the given conditions
def total_runs : ℕ := 132
def boundaries_count : ℕ := 12
def running_percent : ℚ := 54.54545454545454 / 100

-- Function to calculate runs made by running
def runs_by_running (total: ℕ) (percent: ℚ) : ℚ :=
  percent * total

-- Function to calculate runs made from boundaries
def runs_from_boundaries (count: ℕ) : ℕ :=
  count * 4

-- Function to calculate runs made from sixes
def runs_from_sixes (total: ℕ) (boundaries_runs: ℕ) (running_runs: ℚ) : ℚ :=
  total - boundaries_runs - running_runs

-- Function to calculate number of sixes hit
def number_of_sixes (sixes_runs: ℚ) : ℚ :=
  sixes_runs / 6

-- The proof statement for the cricketer hitting 2 sixes
theorem cricketer_hits_two_sixes:
  number_of_sixes (runs_from_sixes total_runs (runs_from_boundaries boundaries_count) (runs_by_running total_runs running_percent)) = 2 := by
  sorry

end cricketer_hits_two_sixes_l74_74846


namespace bottles_produced_l74_74057

theorem bottles_produced (rate_per_6_machines_per_minute : ℕ) (num_machines : ℕ) (time_minutes : ℕ) (h : rate_per_6_machines_per_minute = 420) :
  (10 * (rate_per_6_machines_per_minute / 6) * time_minutes) = 2800 :=
by
  have rate_per_machine := rate_per_6_machines_per_minute / 6
  have rate_per_10_machines := 10 * rate_per_machine
  have bottles_in_4_minutes := rate_per_10_machines * time_minutes
  rw [h, mul_comm, mul_assoc]
  simp only [rate_per_machine, rate_per_10_machines, bottles_in_4_minutes]
  norm_num
  sorry

-- Explanation for the definitions:
-- rate_per_6_machines_per_minute represents the production rate of 6 machines in bottles per minute.
-- num_machines is given as 10 in the context of the problem.
-- time_minutes represents the time span, given as 4 minutes in the context of the problem.
-- h represents the condition that the 6 machines produce 420 bottles per minute.

end bottles_produced_l74_74057


namespace problem_statement_l74_74204

def Delta (a b : ℝ) : ℝ := a^2 - b

theorem problem_statement : Delta (2 ^ (Delta 5 8)) (4 ^ (Delta 2 7)) = 17179869183.984375 := by
  sorry

end problem_statement_l74_74204


namespace maximize_probability_with_C_in_second_match_l74_74520

noncomputable def probability_of_winning_two_consecutive_matches (p1 p2 p3 : ℝ) (h₀ : 0 < p1) (h₁ : p1 < p2) (h₂ : p2 < p3) (h₃ : p3 ≤ 1) : ℝ :=
  -- Define the probabilities when different players are in the second match
  2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3)

theorem maximize_probability_with_C_in_second_match (p1 p2 p3 : ℝ) (h₀ : 0 < p1) (h₁ : p1 < p2) (h₂ : p2 < p3) (h₃ : p3 ≤ 1) :
  let PA := 2 * (p1 * (p2 + p3) - 2 * p1 * p2 * p3),
      PB := 2 * (p2 * (p1 + p3) - 2 * p1 * p2 * p3),
      PC := 2 * (p1 * p3 + p2 * p3 - 2 * p1 * p2 * p3) in
  PC > PA ∧ PC > PB :=
by
  sorry

end maximize_probability_with_C_in_second_match_l74_74520


namespace min_distance_C1_to_C2_l74_74692

-- Definitions of points and trajectory
def line1 (t k : ℝ) : ℝ × ℝ := (t - real.sqrt 3, k * t)
def line2 (m k : ℝ) : ℝ × ℝ := (real.sqrt 3 - m, m / (3 * k))
def C1 (x y : ℝ) : Prop := (x^2 / 3 + y^2 = 1)

-- Definition of curve C2 in Cartesian coordinates
def C2 (x y : ℝ) : Prop := (x + y = 8)

-- Proving the given minimum distance
theorem min_distance_C1_to_C2 :
  ∃ Q : ℝ × ℝ, C1 Q.1 Q.2 ∧
  ∀ R : ℝ × ℝ, C1 R.1 R.2 → dist (Q.1, Q.2) (R.1, R.2) ≥ 3 * real.sqrt 2 :=
sorry

end min_distance_C1_to_C2_l74_74692


namespace similar_quadratic_surd_exists_l74_74741

theorem similar_quadratic_surd_exists (a : ℚ) (ha : a ≠ 0) : ∃ x : ℚ, x * Real.sqrt 2 = a * Real.sqrt 2 :=
by
  use a
  sorry

end similar_quadratic_surd_exists_l74_74741


namespace roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l74_74748

-- Lean 4 statements to capture the proofs without computation.
theorem roman_created_171 (a b : ℕ) (h_sum : a + b = 17) (h_diff : a - b = 1) : 
  a = 9 ∧ b = 8 ∨ a = 8 ∧ b = 9 := 
  sorry

theorem roman_created_1513_m1 (a b : ℕ) (h_sum : a + b = 15) (h_diff : a - b = 13) : 
  a = 14 ∧ b = 1 ∨ a = 1 ∧ b = 14 := 
  sorry

theorem roman_created_1513_m2 (a b : ℕ) (h_sum : a + b = 151) (h_diff : a - b = 3) : 
  a = 77 ∧ b = 74 ∨ a = 74 ∧ b = 77 := 
  sorry

theorem roman_created_largest (a b : ℕ) (h_sum : a + b = 188) (h_diff : a - b = 10) : 
  a = 99 ∧ b = 89 ∨ a = 89 ∧ b = 99 := 
  sorry

end roman_created_171_roman_created_1513_m1_roman_created_1513_m2_roman_created_largest_l74_74748


namespace range_of_k_l74_74656

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^2 + k * real.sqrt (1 - x^2)

/-- 
  Given the function f(x) = x^2 + k * sqrt(1 - x^2), 
  prove that for any real numbers a, b, c ∈ [-1, 1] where f(a), f(b), and f(c) form the sides of a triangle, 
  the range of k is (4 - 2 * sqrt(3), 2). 
-/
theorem range_of_k (k : ℝ) 
  (h : ∀ a b c ∈ Icc (-1:ℝ) 1, let fa := f k a, fb := f k b, fc := f k c in fa + fb > fc ∧ fb + fc > fa ∧ fc + fa > fb) : 
  4 - 2 * real.sqrt 3 < k ∧ k < 2 := 
sorry

end range_of_k_l74_74656


namespace graph_reflection_eq_E_l74_74438

def f (x : ℝ) :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2) ^ 2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0

def f_neg (x : ℝ) := f (-x)

-- Let's hypothesize a function that checks if the graph of f_neg (y = f(-x)) 
-- corresponds to the graphical representation option E
def graph_option_E (x : ℝ) : Prop := sorry

theorem graph_reflection_eq_E : ∀ x : ℝ, f_neg x = graph_option_E x := sorry

end graph_reflection_eq_E_l74_74438


namespace speed_of_sound_l74_74516

theorem speed_of_sound (d₁ d₂ t : ℝ) (speed_car : ℝ) (speed_km_hr_to_m_s : ℝ) :
  d₁ = 1200 ∧ speed_car = 108 ∧ speed_km_hr_to_m_s = (speed_car * 1000 / 3600) ∧ t = 3.9669421487603307 →
  (d₁ + speed_km_hr_to_m_s * t) / t = 332.59 :=
by sorry

end speed_of_sound_l74_74516


namespace determinant_A_l74_74567

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![  2,  4, -2],
    ![  3, -1,  5],
    ![-1,  3,  2]
  ]

theorem determinant_A : det A = -94 := by
  sorry

end determinant_A_l74_74567


namespace quadrilateral_perimeter_l74_74508

theorem quadrilateral_perimeter (ABCD : ConvexQuadrilateral)
    (O₁ O₂ O₃ O₄ : Point)
    (R₁ R₂ R₃ R₄ : ℝ)
    (s : ℝ)
    (h1 : dist O₁ O₄ ^ 2 - dist O₂ O₃ ^ 2 = (R₁ + R₂ + R₃ + R₄) * ((R₁ - R₃) - (R₂ - R₄)))
    (h2 : dist O₂ O₁ ^ 2 - dist O₃ O₄ ^ 2 = (R₁ + R₂ + R₃ + R₄) * ((R₁ - R₃) + (R₂ - R₄)))
    (h3 : dist O₁ O₃ ^ 2 = s ^ 2 + (R₁ - R₃) ^ 2)
    (h4 : dist O₂ O₄ ^ 2 = s ^ 2 + (R₂ - R₄) ^ 2) :
  2 * s = perimeter ABCD :=
sorry

end quadrilateral_perimeter_l74_74508


namespace intersection_A_B_l74_74389

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | (x - 1) / (4 - x) ≥ 0 }
def B : Set ℝ := { x | Real.log2 x ≤ 2 }

theorem intersection_A_B : A ∩ B = { x | 1 ≤ x ∧ x < 4 } :=
by sorry

end intersection_A_B_l74_74389


namespace passing_game_prob_l74_74412

theorem passing_game_prob (n : ℕ): 
  (PlayerA_starts : Bool) →
  (pass_probability : ℚ → ℚ → ℚ) →
  (pass_count : ℕ) →
  (Pn : ℕ → ℚ) →
  PlayerA_starts = true →
  pass_probability = (λ (p1 p2 : ℚ), 1 / 2 * p2) →
  pass_count = n →
  Pn 2 = 1 / 2 ∧ Pn n = 1 / 2 * (1 - Pn (n - 1)) ∧ Pn n = 1 / 3 - 1 / 3 * (-1 / 2)^(n - 1) := 
by
  intros
  sorry

end passing_game_prob_l74_74412


namespace find_length_of_AC_l74_74702

variables (A B C : Point)
variables (AB AC BC : ℝ)
variables (angleB : ℝ)
variables (tanC : ℝ)

-- Conditions
def conditions : Prop :=
  angleB = 90 ∧
  tanC = 4 / 3 ∧
  AB = 3

-- Proof problem
theorem find_length_of_AC
  (h : conditions A B C AB angleB tanC) :
  AC = 5 :=
sorry

end find_length_of_AC_l74_74702


namespace correct_exponent_operation_l74_74482

theorem correct_exponent_operation (a : ℝ) : a^4 / a^3 = a := 
by
  sorry

end correct_exponent_operation_l74_74482


namespace max_imaginary_part_root_theta_l74_74210

theorem max_imaginary_part_root_theta :
  ∃ θ : ℝ, z^6 - z^4 + z^3 - z + 1 = 0 ∧ (-90 ≤ θ ∧ θ ≤ 90) ∧ 
  (∀ z ∈ complex.roots (⇑polynomial.eval z) (z^6 - z^4 + z^3 - z + 1), (z.im = sin θ) → θ = 90) :=
begin
  sorry
end

end max_imaginary_part_root_theta_l74_74210


namespace count_ordered_triples_l74_74165

theorem count_ordered_triples (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : 2 * a * b * c = 2 * (a * b + b * c + a * c)) : 
  ∃ n, n = 10 :=
by
  sorry

end count_ordered_triples_l74_74165


namespace solve_system_l74_74321

theorem solve_system (X Y Z : ℝ)
  (h1 : 0.15 * 40 = 0.25 * X + 2)
  (h2 : 0.30 * 60 = 0.20 * Y + 3)
  (h3 : 0.10 * Z = X - Y) :
  X = 16 ∧ Y = 75 ∧ Z = -590 :=
by
  sorry

end solve_system_l74_74321


namespace five_cds_cost_with_discount_l74_74103

theorem five_cds_cost_with_discount
  (price_2_cds : ℝ)
  (discount_rate : ℝ)
  (num_cds : ℕ)
  (total_cost : ℝ) 
  (h1 : price_2_cds = 40)
  (h2 : discount_rate = 0.10)
  (h3 : num_cds = 5)
  : total_cost = 90 :=
by
  sorry

end five_cds_cost_with_discount_l74_74103


namespace arithmetic_sequence_sum_l74_74725

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith_seq: ∀ n: ℕ, S n = S 0 + n * (S 1 - S 0)) 
  (h5 : S 5 = 10) (h10 : S 10 = 30) : S 15 = 60 :=
by
  sorry

end arithmetic_sequence_sum_l74_74725


namespace area_increase_proof_l74_74448

def percentage_increase_area (L W : ℝ) : ℝ :=
  let A_original := L * W
  let L_new := 1.20 * L
  let W_new := 1.20 * W
  let A_new := L_new * W_new
  ((A_new - A_original) / A_original) * 100

theorem area_increase_proof (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  percentage_increase_area L W = 44 :=
by {
  let A_original := L * W,
  let L_new := 1.20 * L,
  let W_new := 1.20 * W,
  let A_new := L_new * W_new,
  let percentage_increase := ((A_new - A_original) / A_original) * 100,
  have : L_new = 1.20 * L := rfl,
  have : W_new = 1.20 * W := rfl,
  have : A_new = 1.44 * L * W := by rw [L_new, W_new]; ring,
  have : percentage_increase = 44 := by {
    rw [A_new, A_original],
    field_simp [A_original],
    norm_num,
  },
  exact this,
}

end area_increase_proof_l74_74448


namespace sum_unique_remainders_10_l74_74522

theorem sum_unique_remainders_10 : 
    let remainders (n d : ℕ) := n % d in
    let all_remainders := {r | ∃ d : ℕ, 1 ≤ d ∧ d < 10 ∧ r = remainders 10 d} in
    let unique_remainders := all_remainders.to_finset in
    (unique_remainders.sum id) = 10 := 
by
  -- Definitions for clarity
  -- remainder is defined as the result of modulus operation
  let remainders (n d : ℕ) := n % d in
  -- all_remainders is the set of remainders for divisors from 1 to 9
  let all_remainders := {r | ∃ d : ℕ, 1 ≤ d ∧ d < 10 ∧ r = remainders 10 d} in
  -- Convert the set of remainders to a finset for summation
  let unique_remainders := all_remainders.to_finset in
  -- Assertion that the sum of unique remainders is 10
  (unique_remainders.sum id) = 10
  sorry

end sum_unique_remainders_10_l74_74522


namespace passengers_off_in_texas_l74_74190

variable (x : ℕ) -- number of passengers who got off in Texas
variable (initial_passengers : ℕ := 124)
variable (texas_boarding : ℕ := 24)
variable (nc_off : ℕ := 47)
variable (nc_boarding : ℕ := 14)
variable (virginia_passengers : ℕ := 67)

theorem passengers_off_in_texas {x : ℕ} :
  (initial_passengers - x + texas_boarding - nc_off + nc_boarding) = virginia_passengers → 
  x = 48 :=
by
  sorry

end passengers_off_in_texas_l74_74190


namespace eq_f_a_l74_74652

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then log 3 (x + 2) + a else exp x - 1

-- State the main theorem
theorem eq_f_a (a : ℝ) (h : f a (f a (log 2)) = 2 * a) : f a a = 2 := 
sorry

end eq_f_a_l74_74652


namespace min_time_to_one_ball_l74_74041

-- Define the problem in Lean
theorem min_time_to_one_ball (n : ℕ) (h : n = 99) : 
  ∃ T : ℕ, T = 98 ∧ ∀ t < T, ∃ ball_count : ℕ, ball_count > 1 :=
by
  -- Since we are not providing the proof, we use "sorry"
  sorry

end min_time_to_one_ball_l74_74041


namespace cyclic_quadrilateral_opposite_angles_sum_l74_74421

theorem cyclic_quadrilateral_opposite_angles_sum {A B C D : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (O : Type*) [Inhabited O]
  (circle : O → A → B → C → D → Prop)
  (inscribed : circle O A B C D) :
  ∀ (α β γ δ : ℝ), (α + γ = 180 ∧ β + δ = 180) :=
by {
  sorry
}

end cyclic_quadrilateral_opposite_angles_sum_l74_74421


namespace find_strawberry_jelly_l74_74426

variable (total_jelly blueberry_jelly strawberry_jelly : ℕ)

theorem find_strawberry_jelly (h_total : total_jelly = 6310) (h_blueberry : blueberry_jelly = 4518) :
  strawberry_jelly = total_jelly - blueberry_jelly := by
  have h : strawberry_jelly = 1792 := sorry,
  rw [h],
  sorry

end find_strawberry_jelly_l74_74426


namespace remainder_when_divided_by_51_l74_74374

def number_formed_by_even_integers := String.toNat (String.concat ["2", "4", "6", "8", "10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "30", "32", "34", "36", "38", "40", "42", "44", "46", "48", "50"])

theorem remainder_when_divided_by_51 : number_formed_by_even_integers % 51 = 34 := by
  sorry

end remainder_when_divided_by_51_l74_74374


namespace symmetric_point_l74_74829

structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def plane_eq (M : Point3D) : Prop :=
  2 * M.x - 4 * M.y - 4 * M.z - 13 = 0

-- Given Point M
def M : Point3D := { x := 3, y := -3, z := -1 }

-- Symmetric Point M'
def M' : Point3D := { x := 2, y := -1, z := 1 }

theorem symmetric_point (H : plane_eq M) : plane_eq M' ∧ 
  (M'.x = 2 * (3 + 2 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.x) ∧ 
  (M'.y = 2 * (-3 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.y) ∧ 
  (M'.z = 2 * (-1 - 4 * ((-13 + 2*3 - 4*(-3) - 4*(-1)) / 36)) - M.z) :=
sorry

end symmetric_point_l74_74829


namespace max_neg_ints_l74_74823

theorem max_neg_ints (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) : 
  ((a < 0).toNat + (b < 0).toNat + (c < 0).toNat + (d < 0).toNat + (e < 0).toNat + (f < 0).toNat) ≤ 4 :=
sorry

end max_neg_ints_l74_74823


namespace max_log_value_l74_74976

open Real

theorem max_log_value (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_ab : a * b = 8) :
  ∃ c : ℝ, c = 4 ∧ ∀ x : ℝ, (x > 0) → (a * x = 8) → log 2 a * log 2 (2 * x) ≤ c :=
by
  sorry

end max_log_value_l74_74976


namespace Barney_no_clean_towels_l74_74194

noncomputable def BarneyWillNotHaveCleanTowelsFor7Days : Prop :=
  ∀ (total_towels : ℕ) (daily_use : ℕ) (missed_laundry : ℕ) (extra_towels : ℕ) (expected_guests : ℕ),
    total_towels = 18 →
    daily_use = 2 →
    missed_laundry = 7 →
    extra_towels = 5 →
    expected_guests = 3 →
    (total_towels - (daily_use * missed_laundry + extra_towels) <= 0) →
    (total_towels - (daily_use * missed_laundry + extra_towels) + total_towels - (daily_use * 7 + expected_guests) <= -7)
  
theorem Barney_no_clean_towels (total_towels daily_use missed_laundry extra_towels expected_guests : ℕ) :
  BarneyWillNotHaveCleanTowelsFor7Days :=
by
  unfold BarneyWillNotHaveCleanTowelsFor7Days sorry

end Barney_no_clean_towels_l74_74194


namespace total_material_weight_l74_74845

def gravel_weight : ℝ := 5.91
def sand_weight : ℝ := 8.11

theorem total_material_weight : gravel_weight + sand_weight = 14.02 := by
  sorry

end total_material_weight_l74_74845


namespace bracelet_display_contains_8_bracelets_l74_74854

-- Definitions of the problem
def total_number_of_necklaces := 12
def current_number_of_necklaces := 5
def necklace_cost_per_unit := 4

def total_number_of_rings := 30
def current_number_of_rings := 18
def ring_cost_per_unit := 10

def total_number_of_bracelets := 15
def bracelet_cost_per_unit := 5

def total_cost_to_fill_displays := 183

-- Calculate remaining necklaces and their cost.
def remaining_necklaces := total_number_of_necklaces - current_number_of_necklaces
def cost_necklaces := remaining_necklaces * necklace_cost_per_unit

-- Calculate remaining rings and their cost.
def remaining_rings := total_number_of_rings - current_number_of_rings
def cost_rings := remaining_rings * ring_cost_per_unit

-- Let current number of bracelets be B, we need to prove B = 8.
variable (B : ℕ)

-- Calculate remaining bracelets and their cost.
def remaining_bracelets := total_number_of_bracelets - B
def cost_bracelets := remaining_bracelets * bracelet_cost_per_unit

-- Formulate total cost equation
theorem bracelet_display_contains_8_bracelets :
  B = 8 :=
by
  have cost_cases : cost_necklaces + cost_rings + cost_bracelets = total_cost_to_fill_displays
  sorry

end bracelet_display_contains_8_bracelets_l74_74854


namespace calculate_product_l74_74556

theorem calculate_product : (12 * 0.2 * 3 * 0.1 / 0.6 : ℚ) = 6 / 5 := 
by
  have h1 : (0.2 : ℚ) = 1 / 5 := by norm_num
  have h2 : (0.1 : ℚ) = 1 / 10 := by norm_num
  have h3 : (0.6 : ℚ) = 3 / 5 := by norm_num
  rw [h1, h2, h3]
  calc 
    12 * (1 / 5) * 3 * (1 / 10) / (3 / 5) 
      = (12 * 1 / 5 * 3 * 1 / 10) / (3 / 5) : by norm_num
      = (12 * 3 / 50) / (3 / 5) : by norm_num
      = 36 / 50 / (3 / 5) : by norm_num
      = 36 / 50 * 5 / 3 : by field_simp
      = 36 * 5 / (50 * 3) : by norm_num
      = 180 / 150 : by norm_num
      = 6 / 5 : by norm_num

end calculate_product_l74_74556


namespace prob_more_heads_than_tails_l74_74322

theorem prob_more_heads_than_tails :
  let total_outcomes := 1024
  let prob_exactly_5_heads := 252 / 1024
  let prob_getting_more_heads := (1 - prob_exactly_5_heads) / 2
  prob_getting_more_heads = 193 / 512 := 
begin
  sorry
end

end prob_more_heads_than_tails_l74_74322


namespace abs_neg_three_plus_two_sin_thirty_minus_sqrt_nine_l74_74561

theorem abs_neg_three_plus_two_sin_thirty_minus_sqrt_nine : |-3| + 2 * real.sin (real.pi / 6) - real.sqrt 9 = 1 :=
by
  sorry

end abs_neg_three_plus_two_sin_thirty_minus_sqrt_nine_l74_74561


namespace men_became_absent_l74_74524

theorem men_became_absent (num_men absent : ℤ) 
  (num_men_eq : num_men = 180) 
  (days_planned : ℤ) (days_planned_eq : days_planned = 55)
  (days_taken : ℤ) (days_taken_eq : days_taken = 60)
  (work_planned : ℤ) (work_planned_eq : work_planned = num_men * days_planned)
  (work_taken : ℤ) (work_taken_eq : work_taken = (num_men - absent) * days_taken)
  (work_eq : work_planned = work_taken) :
  absent = 15 :=
  by sorry

end men_became_absent_l74_74524


namespace log_sum_of_geometric_l74_74640

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, 0 < a n

theorem log_sum_of_geometric {a : ℕ → ℝ}
  (h_pos : geometric_sequence a)
  (h_eq : a 5 * a 6 + a 4 * a 7 = 18) :
  ∑ i in finset.range 10, real.log (a i.succ) / real.log 3 = 10 :=
by
  sorry

end log_sum_of_geometric_l74_74640


namespace not_perfect_square_l74_74896

theorem not_perfect_square (x y : ℤ) : ¬ ∃ k : ℤ, k^2 = (x^2 + x + 1)^2 + (y^2 + y + 1)^2 :=
by
  sorry

end not_perfect_square_l74_74896


namespace total_paint_remaining_l74_74582

-- Definitions based on the conditions
def paint_per_statue : ℚ := 1 / 16
def statues_to_paint : ℕ := 14

-- Theorem statement to prove the answer
theorem total_paint_remaining : (statues_to_paint : ℚ) * paint_per_statue = 7 / 8 := 
by sorry

end total_paint_remaining_l74_74582


namespace find_multiple_of_games_l74_74799

-- declaring the number of video games each person has
def Tory_videos := 6
def Theresa_videos := 11
def Julia_videos := Tory_videos / 3

-- declaring the multiple we need to find
def multiple_of_games := Theresa_videos - Julia_videos * 5

-- Theorem stating the problem
theorem find_multiple_of_games : ∃ m : ℕ, Julia_videos * m + 5 = Theresa_videos :=
by
  sorry

end find_multiple_of_games_l74_74799


namespace show_revenue_l74_74174

variable (tickets_first_show : Nat) (tickets_cost : Nat) (multiplicator : Nat)
variable (tickets_second_show : Nat := multiplicator * tickets_first_show)
variable (total_tickets : Nat := tickets_first_show + tickets_second_show)
variable (total_revenue : Nat := total_tickets * tickets_cost)

theorem show_revenue :
    tickets_first_show = 200 ∧ tickets_cost = 25 ∧ multiplicator = 3 →
    total_revenue = 20000 := 
by
    intros h
    sorry

end show_revenue_l74_74174


namespace steven_owes_jeremy_l74_74065

theorem steven_owes_jeremy (payment_per_room : ℚ) (rooms_cleaned : ℚ) : 
  (payment_per_room = 13/3) → (rooms_cleaned = 8/5) → 
  (payment_per_room * rooms_cleaned = 104/15) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end steven_owes_jeremy_l74_74065


namespace quadratic_trinomial_unique_l74_74227

theorem quadratic_trinomial_unique
  (a b c : ℝ)
  (h1 : b^2 - 4*(a+1)*c = 0)
  (h2 : (b+1)^2 - 4*a*c = 0)
  (h3 : b^2 - 4*a*(c+1) = 0) :
  a = 1/8 ∧ b = -3/4 ∧ c = 1/8 :=
by
  sorry

end quadratic_trinomial_unique_l74_74227


namespace debby_brought_10_bottles_l74_74941

theorem debby_brought_10_bottles (d : ℕ) (t : ℕ) (h : d = 8) (h2 : t = 2) :
  d + t = 10 := 
by
  rw [h, h2]
  exact Nat.add_comm 8 2
  sorry

end debby_brought_10_bottles_l74_74941


namespace lower_rate_is_12_l74_74148

-- Given definitions
def principal : ℝ := 14000
def higher_rate : ℝ := 15 / 100
def lower_rate (P : ℝ) : ℝ := P / 100
def time : ℝ := 2
def interest_difference : ℝ := 840

-- Interest calculations
def interest (P : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time)

-- Interest received at higher rate
def I1 : ℝ := interest principal higher_rate time

-- Interest received at lower rate
def I2 (P : ℝ) : ℝ := interest principal (lower_rate P) time

-- Proof problem statement
theorem lower_rate_is_12 :
  ∃ P : ℝ, (P = 12) ∧ (I1 = (I2 P) + interest_difference) :=
by
  have I1_calc : I1 = 4200 := by calc
    I1 = 14000 * (15 / 100) * 2 : sorry -- intermediate steps
      ... = 4200 : sorry
  have I2_calc : ∀ P, I2 P = 14000 * (P / 100) * 2 := sorry -- intermediate steps
  use 12
  split
  . exact rfl
  . calc
    I1 = 4200 : I1_calc
    ... = 280 * 12 + 840 : sorry -- final calculated
    ... = (I2 12) + 840 : sorry

end lower_rate_is_12_l74_74148


namespace unit_prices_possible_combinations_l74_74147

-- Part 1: Unit Prices
theorem unit_prices (x y : ℕ) (h1 : x = y - 20) (h2 : 3 * x + 2 * y = 340) : x = 60 ∧ y = 80 := 
by 
  sorry

-- Part 2: Possible Combinations
theorem possible_combinations (a : ℕ) (h3 : 60 * a + 80 * (150 - a) ≤ 10840) (h4 : 150 - a ≥ 3 * a / 2) : 
  a = 58 ∨ a = 59 ∨ a = 60 := 
by 
  sorry

end unit_prices_possible_combinations_l74_74147


namespace sampling_interval_l74_74163

theorem sampling_interval (N n : ℕ) (hN : N = 1000) (hn : n = 20) : N / n = 50 :=
by {
  rw [hN, hn],
  norm_num,
  sorry
}

end sampling_interval_l74_74163


namespace range_of_m_l74_74955

-- Definitions of the functions and intervals
def f (x : ℝ) : ℝ := x^2 - 2 * x
def g (m x : ℝ) : ℝ := m * x + 2

-- Main theorem statement
theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ set.Icc (-1 : ℝ) 2, ∃ x0 ∈ set.Icc (-1 : ℝ) 2, g m x1 = f x0) ↔ m ∈ set.Icc (-1 : ℝ) (1 / 2) :=
sorry

end range_of_m_l74_74955


namespace greatest_length_proof_l74_74358

noncomputable def length_AE : ℝ := real.sqrt (5^2 + 9^2)
def length_CD : ℕ := 5
noncomputable def length_CF : ℝ := real.sqrt (2^2 + 4^2)
noncomputable def length_AC : ℝ := real.sqrt (3^2 + 4^2)
noncomputable def length_CE : ℝ := real.sqrt (2^2 + 5^2)
noncomputable def length_FD : ℝ := real.sqrt (2^2 + 9^2)

theorem greatest_length_proof : 
  max length_AE (max (length_CD + length_CF) (max (length_AC + length_CF) (max length_FD (length_AC + length_CE)))) = length_AC + length_CE :=
sorry

end greatest_length_proof_l74_74358


namespace fraction_of_marbles_lost_l74_74564

variable (initial_marbles_in_jar : ℕ)
variable (marbles_first_taken_out : ℕ)
variable (marbles_second_taken_out : ℕ)
variable (new_marbles_added : ℕ)
variable (total_marbles_after_game : ℕ)
variable (marbles_lost : ℕ)

-- Definitions based on the conditions of the problem
def marbles_first_taken_out := 12
def marbles_second_taken_out := 10
def new_marbles_added := 25
def total_marbles_after_game := 41
def marbles_lost := marbles_first_taken_out - (total_marbles_after_game - marbles_second_taken_out - new_marbles_added)

theorem fraction_of_marbles_lost :
  marbles_lost / marbles_first_taken_out = 1 / 2 :=
by
  -- Proof goes here
  sorry

end fraction_of_marbles_lost_l74_74564


namespace quadratic_trinomial_unique_l74_74228

theorem quadratic_trinomial_unique
  (a b c : ℝ)
  (h1 : b^2 - 4 * (a + 1) * c = 0)
  (h2 : (b + 1)^2 - 4 * a * c = 0)
  (h3 : b^2 - 4 * a * (c + 1) = 0) :
  a = 1 / 8 ∧ b = -3 / 4 ∧ c = 1 / 8 :=
begin
  -- statement for the theorem, proof not required
  sorry
end

end quadratic_trinomial_unique_l74_74228


namespace hexagon_perimeter_correct_l74_74853

-- Define the points of the hexagon
def point1 : (ℝ × ℝ) := (0, 1)
def point2 : (ℝ × ℝ) := (1, 2)
def point3 : (ℝ × ℝ) := (2, 2)
def point4 : (ℝ × ℝ) := (2, 1)
def point5 : (ℝ × ℝ) := (3, 1)
def point6 : (ℝ × ℝ) := (2, 0)

-- Define the distance formula
def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Calculate the distances
def dist1 : ℝ := distance point1 point2
def dist2 : ℝ := distance point2 point3
def dist3 : ℝ := distance point3 point4
def dist4 : ℝ := distance point4 point5
def dist5 : ℝ := distance point5 point6
def dist6 : ℝ := distance point6 point1

-- Define the total perimeter
def perimeter : ℝ := dist1 + dist2 + dist3 + dist4 + dist5 + dist6

-- Define the expected form of the perimeter and the result a + b + c
def expected_a : ℕ := 3
def expected_b : ℕ := 2
def expected_c : ℕ := 1

def expected_perimeter : ℝ := (expected_a:ℝ) + expected_b * real.sqrt 2 + expected_c * real.sqrt 5

def final_answer : ℕ := expected_a + expected_b + expected_c

-- The proof statement: The perimeter of the hexagon matches the expected form, and hence, a + b + c = 6
theorem hexagon_perimeter_correct :
  perimeter = expected_perimeter ∧ final_answer = 6 :=
by
  -- skipping the proof here
  sorry

end hexagon_perimeter_correct_l74_74853


namespace no_function_f_satisfies_condition_l74_74427

theorem no_function_f_satisfies_condition :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = f x + y^2 :=
by
  sorry

end no_function_f_satisfies_condition_l74_74427


namespace small_glass_cost_l74_74739

theorem small_glass_cost 
  (S : ℝ)
  (small_glass_cost : ℝ)
  (large_glass_cost : ℝ := 5)
  (initial_money : ℝ := 50)
  (num_small : ℝ := 8)
  (change : ℝ := 1)
  (num_large : ℝ := 5)
  (spent_money : ℝ := initial_money - change)
  (total_large_cost : ℝ := num_large * large_glass_cost)
  (total_cost : ℝ := num_small * S + total_large_cost)
  (total_cost_eq : total_cost = spent_money) :
  S = 3 :=
by
  sorry

end small_glass_cost_l74_74739


namespace perpendicular_chords_cosine_bound_l74_74273

theorem perpendicular_chords_cosine_bound 
  (a b : ℝ) 
  (h_ab : a > b) 
  (h_b0 : b > 0) 
  (θ1 θ2 : ℝ) 
  (x y : ℝ → ℝ) 
  (h_ellipse : ∀ t, x t = a * Real.cos t ∧ y t = b * Real.sin t) 
  (h_theta1 : ∃ t1, (x t1 = a * Real.cos θ1 ∧ y t1 = b * Real.sin θ1)) 
  (h_theta2 : ∃ t2, (x t2 = a * Real.cos θ2 ∧ y t2 = b * Real.sin θ2)) 
  (h_perpendicular: θ1 = θ2 + π / 2 ∨ θ1 = θ2 - π / 2) :
  0 ≤ |Real.cos (θ1 - θ2)| ∧ |Real.cos (θ1 - θ2)| ≤ (a ^ 2 - b ^ 2) / (a ^ 2 + b ^ 2) :=
sorry

end perpendicular_chords_cosine_bound_l74_74273


namespace terminal_side_equiv_l74_74479

theorem terminal_side_equiv (θ : ℝ) (hθ : θ = 23 * π / 3) : 
  ∃ k : ℤ, θ = 2 * π * k + 5 * π / 3 := by
  sorry

end terminal_side_equiv_l74_74479


namespace domain_of_f_l74_74074

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 - x) / (2 + x))

theorem domain_of_f : ∀ x : ℝ, (2 - x) / (2 + x) > 0 ∧ 2 + x ≠ 0 ↔ -2 < x ∧ x < 2 :=
by
  intro x
  sorry

end domain_of_f_l74_74074


namespace area_of_square_field_l74_74500

def side_length : ℕ := 7
def expected_area : ℕ := 49

theorem area_of_square_field : (side_length * side_length) = expected_area := 
by
  -- The proof steps will be filled here
  sorry

end area_of_square_field_l74_74500


namespace solve_Mary_height_l74_74729

theorem solve_Mary_height :
  ∃ (m s : ℝ), 
  s = 150 ∧ 
  s * 1.2 = 180 ∧ 
  m = s + (180 - s) / 2 ∧ 
  m = 165 :=
by
  sorry

end solve_Mary_height_l74_74729


namespace ninetieth_percentile_data_l74_74686

def data : List ℕ := [7, 8, 8, 9, 11, 13, 15, 17, 20, 22]

def percentile_position (p : ℝ) (n : ℕ) : ℝ :=
  p * n

def nth_percentile (sorted_list : List ℕ) (p : ℝ) : ℕ :=
  let n := sorted_list.length
  let pos := percentile_position p n
  if (pos.fract = 0) then
    sorted_list.getD (pos.to_nat - 1) 0
  else
    let lower_idx := pos.floor.to_nat - 1
    let upper_idx := lower_idx + 1
    (sorted_list.getD lower_idx 0 + sorted_list.getD upper_idx 0) / 2

theorem ninetieth_percentile_data :
  nth_percentile data 0.90 = 21 :=
by
  sorry

end ninetieth_percentile_data_l74_74686


namespace oranges_bought_l74_74001

theorem oranges_bought (total_cost : ℝ) 
  (selling_price_per_orange : ℝ) 
  (profit_per_orange : ℝ) 
  (cost_price_per_orange : ℝ) 
  (h1 : total_cost = 12.50)
  (h2 : selling_price_per_orange = 0.60)
  (h3 : profit_per_orange = 0.10)
  (h4 : cost_price_per_orange = selling_price_per_orange - profit_per_orange) :
  (total_cost / cost_price_per_orange) = 25 := 
by
  sorry

end oranges_bought_l74_74001


namespace divisible_by_17_l74_74785

theorem divisible_by_17 (a b c d : ℕ) (h1 : a + b + c + d = 2023)
    (h2 : 2023 ∣ (a * b - c * d))
    (h3 : 2023 ∣ (a^2 + b^2 + c^2 + d^2))
    (h4 : ∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 7 ∣ x) :
    (∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 17 ∣ x) := 
sorry

end divisible_by_17_l74_74785


namespace pyr_sphere_ineq_l74_74168

open Real

theorem pyr_sphere_ineq (h a : ℝ) (R r : ℝ) 
  (h_pos : h > 0) (a_pos : a > 0) 
  (pyr_in_sphere : ∀ h a : ℝ, R = (2*a^2 + h^2) / (2*h))
  (pyr_circ_sphere : ∀ h a : ℝ, r = (a * h) / (sqrt (h^2 + a^2) + a)) :
  R ≥ (sqrt 2 + 1) * r := 
sorry

end pyr_sphere_ineq_l74_74168


namespace percent_non_unionized_women_is_80_l74_74824

noncomputable def employeeStatistics :=
  let total_employees := 100
  let percent_men := 50
  let percent_unionized := 60
  let percent_unionized_men := 70
  let men := (percent_men / 100) * total_employees
  let unionized := (percent_unionized / 100) * total_employees
  let unionized_men := (percent_unionized_men / 100) * unionized
  let non_unionized_men := men - unionized_men
  let non_unionized := total_employees - unionized
  let non_unionized_women := non_unionized - non_unionized_men
  let percent_non_unionized_women := (non_unionized_women / non_unionized) * 100
  percent_non_unionized_women

theorem percent_non_unionized_women_is_80 :
  employeeStatistics = 80 :=
by
  sorry

end percent_non_unionized_women_is_80_l74_74824


namespace power_mod_eight_l74_74811

theorem power_mod_eight : 3 ^ 2007 % 8 = 3 % 8 := by
  sorry

end power_mod_eight_l74_74811


namespace cube_construction_ways_l74_74521

theorem cube_construction_ways :
  let cube_size := 3
  let total_cubes := cube_size * cube_size * cube_size
  let white_cubes := 13
  let black_cubes := 14
  let distinct_ways := 385244
  (total_cubes = 27 ∧ (white_cubes + black_cubes = total_cubes) ∧ (white_cubes == 13) ∧ (black_cubes == 14)) →
  (num_distinct_constructions := 
    -- A function (hypothetically) that computes distinct ways considering rotations;
    sorry:
    num_distinct_constructions 
  = distinct_ways) :=
by
  -- This part should include the full proof, but we are skipping it with sorry.
  sorry

end cube_construction_ways_l74_74521


namespace overall_percentage_favoring_new_tool_l74_74862

theorem overall_percentage_favoring_new_tool (teachers students : ℕ) 
  (favor_teachers favor_students : ℚ) 
  (surveyed_teachers surveyed_students : ℕ) : 
  surveyed_teachers = 200 → 
  surveyed_students = 800 → 
  favor_teachers = 0.4 → 
  favor_students = 0.75 → 
  ( ( (favor_teachers * surveyed_teachers) + (favor_students * surveyed_students) ) / (surveyed_teachers + surveyed_students) ) * 100 = 68 := 
by 
  sorry

end overall_percentage_favoring_new_tool_l74_74862


namespace photographers_possible_l74_74366

-- Definition of the problem conditions
def canPhotograph (A B : Fin 6) (position : Fin 6 → Fin 2 → ℝ) : Prop :=
  ∀ C : Fin 6, C ≠ A ∧ C ≠ B → ¬ (position C ∈ segment (position A) (position B))

-- Prove that positioning six photographers is possible
theorem photographers_possible : 
  ∃ (position : Fin 6 → Fin 2 → ℝ),
  ∀ (A : Fin 6),
  ∃ (B C D E : Fin 6), 
  (B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ B
   ∧ B ≠ A ∧ C ≠ A ∧ D ≠ A ∧ E ≠ A
   ∧ canPhotograph A B position
   ∧ canPhotograph A C position
   ∧ canPhotograph A D position
   ∧ canPhotograph A E position) :=
sorry

end photographers_possible_l74_74366


namespace order_f_values_l74_74020

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem order_f_values (hf_even : even_function f) (hf_inc : increasing_on_nonneg f) :
  f (-2) < f 3 ∧ f 3 < f (-π) :=
by
  have : 2 < π := Real.pi_pos
  have h1 : f (-2) = f 2 := hf_even 2
  have h2 : f (-π) = f π := hf_even π
  have h3 : f 2 < f 3 := hf_inc 2 3 (by linarith) (by linarith)
  have h4 : f 3 < f π := hf_inc 3 π (by linarith) (by linarith)
  linarith

end order_f_values_l74_74020


namespace perfect_square_form_l74_74912

theorem perfect_square_form (N : ℕ) (hN : 0 < N) : 
  ∃ x : ℤ, 2^N - 2 * (N : ℤ) = x^2 ↔ N = 1 ∨ N = 2 :=
by
  sorry

end perfect_square_form_l74_74912


namespace initial_children_on_bus_l74_74756

-- Define the conditions
variables (x : ℕ)

-- Define the problem statement
theorem initial_children_on_bus (h : x + 7 = 25) : x = 18 :=
sorry

end initial_children_on_bus_l74_74756


namespace sum_of_possible_amounts_of_change_is_108_cents_l74_74370

noncomputable def possibleAmountsUnderQuarters : List ℕ := 
  [0, 25, 50, 75].map (λ x => x + 4)

noncomputable def possibleAmountsUnderNickels : List ℕ := 
  (List.range 20).map (λ x => 5 * x + 3).filter (λ x => x < 100)

def commonAmounts : List ℕ :=
  possibleAmountsUnderQuarters.filter (λ x => x ∈ possibleAmountsUnderNickels)

def sumCommonAmounts : ℕ := 
  commonAmounts.foldl (λ acc x => acc + x) 0

theorem sum_of_possible_amounts_of_change_is_108_cents :
  sumCommonAmounts = 108 := 
sorry

end sum_of_possible_amounts_of_change_is_108_cents_l74_74370


namespace range_of_a_l74_74261

def p (a : ℝ) : Prop := ∀ (x : ℝ), ax^2 - x + a / 16 > 0
def q (a : ℝ) : Prop := ∀ (x : ℝ), 3^x - 9^x < a

theorem range_of_a (a : ℝ) : ¬ (p a ∧ q a) → a ≤ 2 :=
sorry

end range_of_a_l74_74261


namespace find_a_l74_74455

theorem find_a (x y a : ℕ) (h1 : ((10 : ℕ) ^ ((32 : ℕ) / y)) ^ a - (64 : ℕ) = (279 : ℕ))
                 (h2 : a > 0)
                 (h3 : x * y = 32) :
  a = 1 :=
sorry

end find_a_l74_74455


namespace maximum_sum_of_differences_l74_74792

theorem maximum_sum_of_differences :
  ∃ s : list ℕ, s.perm (list.range 1 22) ∧
  ∑ i in finset.range 21, abs (s.nth_le i (by linarith) - s.nth_le ((i + 1) % 21) (by linarith)) = 220 :=
sorry

end maximum_sum_of_differences_l74_74792


namespace count_two_digit_powers_of_three_l74_74310

theorem count_two_digit_powers_of_three : 
  ∃ (n1 n2 : ℕ), 10 ≤ 3^n1 ∧ 3^n1 < 100 ∧ 10 ≤ 3^n2 ∧ 3^n2 < 100 ∧ n1 ≠ n2 ∧ ∀ n : ℕ, (10 ≤ 3^n ∧ 3^n < 100) → (n = n1 ∨ n = n2) ∧ n1 = 3 ∧ n2 = 4 := by
  sorry

end count_two_digit_powers_of_three_l74_74310


namespace log_base_32_eq_five_half_l74_74219

theorem log_base_32_eq_five_half (x : ℝ) (hx : log x 32 = 5 / 2) : x = 4 := by
  sorry

end log_base_32_eq_five_half_l74_74219


namespace square_circumcircle_intersection_l74_74409

theorem square_circumcircle_intersection
  (A B M N C D : Point)
  (sq_AM : is_square AM C)
  (sq_MB : is_square MB D)
  (circ_AM : is_circumscribed circle_AM)
  (circ_MB : is_circumscribed circle_MB)
  (intersect_AM_MB : circles_intersect_at circle_AM circle_MB N)
  : lies_on_line A N D ∧ right_angle_triangle A N B :=
by
  sorry

end square_circumcircle_intersection_l74_74409


namespace g_at_five_l74_74992

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_five :
  (∀ x : ℝ, g (3 * x - 7) = 4 * x + 6) →
  g (5) = 22 :=
by
  intros h
  sorry

end g_at_five_l74_74992


namespace solve_for_a_l74_74775

def f (x : ℝ) : ℝ := if x < 2 then 3^(x-2) else real.log_base 3 (x^2 - 1)

theorem solve_for_a :
  (∀ a : ℝ, (f a = 1) → (a = 2)) := 
sorry

end solve_for_a_l74_74775


namespace only_n_eq_1_divides_2_pow_n_minus_1_l74_74207

theorem only_n_eq_1_divides_2_pow_n_minus_1 (n : ℕ) (h1 : 1 ≤ n) (h2 : n ∣ 2^n - 1) : n = 1 :=
sorry

end only_n_eq_1_divides_2_pow_n_minus_1_l74_74207


namespace sum_of_coefficients_is_minus_five_l74_74778

def polynomial (A B C : ℤ) : (x : ℝ) → ℝ := λ x, x^3 + A * x^2 + B * x + C

theorem sum_of_coefficients_is_minus_five :
  ∃ A B C : ℤ, (∀ x : ℝ, polynomial A B C x = (x + 3) * (x - 2) * x) ∧ (A + B + C = -5) :=
by
  use 1, -6, 0
  simp [polynomial]
  sorry

end sum_of_coefficients_is_minus_five_l74_74778


namespace units_digit_of_7_pow_6_cubed_l74_74812

-- Define the repeating cycle of unit digits for powers of 7
def unit_digit_of_power_of_7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0 -- This case is actually unreachable given the modulus operation

-- Define the main problem statement
theorem units_digit_of_7_pow_6_cubed : unit_digit_of_power_of_7 (6 ^ 3) = 1 :=
by
  sorry

end units_digit_of_7_pow_6_cubed_l74_74812


namespace container_emptying_l74_74098

theorem container_emptying (a b c : ℕ) : ∃ m n k : ℕ,
  (m = 0 ∨ n = 0 ∨ k = 0) ∧
  (∀ a' b' c', 
    (a' = a ∧ b' = b ∧ c' = c) ∨ 
    (a' + 2 * b' = a' ∧ b' = b ∧ c' + 2 * b' = c') ∨ 
    (a' + 2 * c' = a' ∧ b' + 2 * c' = b' ∧ c' = c') ∨ 
    (a + 2 * b' + c' = a' + 2 * m * (a + b') ∧ b' = n * (a + b') ∧ c' = k * (a + b')) 
  -> (a' = 0 ∨ b' = 0 ∨ c' = 0)) :=
sorry

end container_emptying_l74_74098


namespace tile_problem_l74_74859

noncomputable def total_tiles (total_black_tiles : ℕ) : ℕ :=
  let side_length := (total_black_tiles + 1) / 2
  side_length * side_length

theorem tile_problem (h : total_black_tiles = 101) : total_tiles total_black_tiles = 2601 :=
by {
  rw h,
  unfold total_tiles,
  simp,
  norm_num,
  sorry -- This is where the detailed solution steps would be filled in.
}

end tile_problem_l74_74859


namespace isosceles_triangle_condition_l74_74965

variable {A B C a b c : ℝ}

theorem isosceles_triangle_condition :
  (acos A = b * cos B) ↔ (isosceles_triangle A B C a b c) :=
sorry

/-- Helper function to define an isosceles triangle. -/
def isosceles_triangle (A B C a b c : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

end isosceles_triangle_condition_l74_74965


namespace geometric_sum_5_l74_74957

variables {α : Type*} [linear_ordered_field α] 

noncomputable def geometric_sequence (a : α) (q : α) (n : ℕ) := a * q^n

theorem geometric_sum_5
  (a : α) (q : α)
  (h1 : 0 < a) (h2 : 0 < q)
  (h3 : (geometric_sequence a q 0) * (geometric_sequence a q 5) = 2 * (geometric_sequence a q 2))
  (h4 : (geometric_sequence a q 3 + (geometric_sequence a q 5)) / 2 = 5) :
  (a * (1 - q^5) / (1 - q) = 31 / 4) :=
by sorry

end geometric_sum_5_l74_74957


namespace radian_measure_of_central_angle_l74_74963

theorem radian_measure_of_central_angle
  (r : ℝ) (S : ℝ) (α : ℝ)
  (h_r : r = 8)
  (h_S : S = 32)
  (h_area : S = 1/2 * α * r^2) :
  α = 1 :=
by {
  rewrite [h_r, h_S] at h_area,
  -- We would simplify h_area here to conclude α = 1.
  sorry
}

end radian_measure_of_central_angle_l74_74963


namespace parallel_lines_distance_l74_74660

noncomputable def line1 (x y : ℝ) := 2 * x + y + 1 = 0
noncomputable def line2 (x y : ℝ) := 2 * x + y - 1 = 0

theorem parallel_lines_distance :
  (∀ x y : ℝ, line1 x y) →
  (∀ x y : ℝ, line2 x y) →
  ∀ l1 l2 : ℝ, l1 = 2 ∧ l2 = 2
  → (1 - (-1)) / sqrt (4 + 1) = (2 * sqrt 5) / 5 :=
by
  intros _ _ _ hl hr
  sorry

end parallel_lines_distance_l74_74660


namespace regular_dodecahedron_has_12_faces_l74_74665

-- Define a structure to represent a regular dodecahedron
structure RegularDodecahedron where

-- The main theorem to state that a regular dodecahedron has 12 faces
theorem regular_dodecahedron_has_12_faces (D : RegularDodecahedron) : ∃ faces : ℕ, faces = 12 := by
  sorry

end regular_dodecahedron_has_12_faces_l74_74665


namespace find_N_l74_74907

theorem find_N : {N : ℕ // N > 0 ∧ ∃ k : ℕ, 2^N - 2 * N = k^2} = {1, 2} := 
    sorry

end find_N_l74_74907


namespace magnitude_diff_eq_one_l74_74996

-- Define the vectors a and b
def a : ℝ × ℝ := (real.cos (15 * real.pi / 180), real.sin (15 * real.pi / 180))
def b : ℝ × ℝ := (real.sin (15 * real.pi / 180), real.cos (15 * real.pi / 180))

-- The statement to prove
theorem magnitude_diff_eq_one : 
  ‖(a.1 - b.1, a.2 - b.2)‖ = 1 := 
sorry

end magnitude_diff_eq_one_l74_74996


namespace tangent_line_at_one_eq_sixx_plus_one_l74_74984

noncomputable def f (x : ℝ) : ℝ := -x^2 + 8 * x

theorem tangent_line_at_one_eq_sixx_plus_one :
  let x := (1 : ℝ)
  let y := f x
  let m := ( -2 * x + 8 : ℝ)
  ∃ φ : ℝ → ℝ, (φ(x) = y) ∧
               (∀ x', φ(x') = m * (x' - x) + y) ∧
               φ = fun x => 6*x + 1 :=
begin
  sorry
end

end tangent_line_at_one_eq_sixx_plus_one_l74_74984


namespace tangent_lines_to_circle_through_M_l74_74253

noncomputable def circle_eq : ℝ × ℝ → ℝ := λ p, p.1^2 + p.2^2 - 4 * p.1 - 6 * p.2 - 3

def point_M : ℝ × ℝ := (-2, 0)

def tangent_lines (p : ℝ × ℝ) : Prop :=
  (p.1 = -2 ∧ p.2 = 0) ∨ (7 * p.1 + 24 * p.2 + 14 = 0)

theorem tangent_lines_to_circle_through_M :
  ∀ (p : ℝ × ℝ), circle_eq p = 0 → ∃ t : ℝ × ℝ, tangent_lines t ∧ t ≠ point_M :=
begin
  sorry,
end

end tangent_lines_to_circle_through_M_l74_74253


namespace carnival_tickets_l74_74758

theorem carnival_tickets (x : ℕ) (won_tickets : ℕ) (found_tickets : ℕ) (ticket_value : ℕ) (total_value : ℕ)
  (h1 : won_tickets = 5 * x)
  (h2 : found_tickets = 5)
  (h3 : ticket_value = 3)
  (h4 : total_value = 30)
  (h5 : total_value = (won_tickets + found_tickets) * ticket_value) :
  x = 1 :=
by
  -- Proof omitted
  sorry

end carnival_tickets_l74_74758


namespace smallest_number_among_neg3_2_neg2_0_l74_74880

theorem smallest_number_among_neg3_2_neg2_0 : 
  ∀ (x : ℤ), x ∈ ({-3, 2, -2, 0} : set ℤ) → -3 ≤ x := 
begin
  intros x hx,
  fin_cases hx,
  repeat {simp},
  sorry
end

end smallest_number_among_neg3_2_neg2_0_l74_74880


namespace perpendicular_lines_foot_parallel_lines_distance_l74_74971

-- Condition definitions for perpendicular case
def line1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) := x - y + a = 0

theorem perpendicular_lines_foot {a : ℝ} :
  (∀ x y : ℝ, line1 a x y → line2 a x y → true) →
  a = 2 ∧ ∃ P : ℝ × ℝ, P = (-5/4, 3/4) :=
sorry
  
-- Condition definitions for parallel case
theorem parallel_lines_distance {a : ℝ} :
  (∀ x y : ℝ, line1 a x y → line2 a x y → false) →
  a = -2 ∧ ∀ c1 c2 m : ℝ, m = -1 → c1 = -(1/2) → c2 = -2 → real.abs (c1 - c2) / real.sqrt (m^2 + 1) = (3 * real.sqrt 2 / 4) :=
sorry

end perpendicular_lines_foot_parallel_lines_distance_l74_74971


namespace remaining_length_l74_74145

variable (L₁ L₂: ℝ)
variable (H₁: L₁ = 0.41)
variable (H₂: L₂ = 0.33)

theorem remaining_length (L₁ L₂: ℝ) (H₁: L₁ = 0.41) (H₂: L₂ = 0.33) : L₁ - L₂ = 0.08 :=
by
  sorry

end remaining_length_l74_74145


namespace gcd_1260_924_l74_74779

theorem gcd_1260_924 : Nat.gcd 1260 924 = 84 :=
by
  sorry

end gcd_1260_924_l74_74779


namespace two_digit_powers_of_three_l74_74312

theorem two_digit_powers_of_three : 
  (Finset.filter (λ n : ℕ, 10 ≤ 3^n ∧ 3^n ≤ 99) (Finset.range 6)).card = 3 := by 
sorry

end two_digit_powers_of_three_l74_74312


namespace factorial_contains_9450_l74_74112

/-- Prove that 9450 is a factor of n! for the smallest positive integer n -/
theorem factorial_contains_9450 (n : ℕ) : 
  (∀ k, 9450 = 2 * 3^3 * 5^2 * 7 ∧ k ≤ n → k! % 9450 = 0) ↔ (n = 10) :=
begin
  sorry
end

end factorial_contains_9450_l74_74112


namespace abs_gt_two_nec_but_not_suff_l74_74125

theorem abs_gt_two_nec_but_not_suff (x : ℝ) : (|x| > 2 → x < -2) ∧ (¬ (|x| > 2 ↔ x < -2)) := 
sorry

end abs_gt_two_nec_but_not_suff_l74_74125


namespace diamond_sum_l74_74573

def diamond (x : ℚ) : ℚ := (x^3 + 2 * x^2 + 3 * x) / 6

theorem diamond_sum : diamond 2 + diamond 3 + diamond 4 = 92 / 3 := by
  sorry

end diamond_sum_l74_74573


namespace team_A_games_42_l74_74759

noncomputable def team_games (a b : ℕ) : Prop :=
  (a * 2 / 3 + 7) = b * 5 / 8

theorem team_A_games_42 (a b : ℕ) (h1 : a * 2 / 3 = b * 5 / 8 - 7)
                                 (h2 : b = a + 14) :
  a = 42 :=
by
  sorry

end team_A_games_42_l74_74759


namespace valid_3_word_sentences_count_l74_74067

-- Definitions of the 4 words in Gnollish language
inductive Word
| splargh | glumph | amr | kreet

-- Conditions for validity of sentences
def valid_sentence (w1 w2 w3 : Word) : Prop :=
  ¬(w1 = Word.splargh ∧ w2 = Word.glumph) ∧
  ¬(w1 = Word.kreet ∧ w2 = Word.amr) ∧
  ¬(w2 = Word.splargh ∧ w3 = Word.glumph) ∧
  ¬(w2 = Word.kreet ∧ w3 = Word.amr)

-- Definition of the proof problem
theorem valid_3_word_sentences_count :
  {s : List Word // s.length = 3 ∧ valid_sentence s.head! (s.get! 1) (s.get! 2)}.card = 48 := 
sorry

end valid_3_word_sentences_count_l74_74067


namespace prob_seventh_grade_prob_mixed_grade_l74_74405

-- Define students as types within their respective grades
inductive Student
| A | B | C | D

def seventh_grade : set Student := { Student.A, Student.B }
def eighth_grade : set Student := { Student.C, Student.D }

-- Total number of students with outstanding awards
def total_students := 4

-- Number of seventh-grade students with outstanding awards
def seventh_grade_students := 2

-- Probability that a randomly selected outstanding student is from the seventh grade
def seventh_grade_prob := (seventh_grade_students : ℚ) / total_students

theorem prob_seventh_grade : seventh_grade_prob = 1 / 2 :=
by
  -- calculation here
  sorry

-- Number of outcomes where one student is from seventh grade and the other from eighth grade
def favorable_outcomes := 8

-- Total possible outcomes
def possible_outcomes := total_students * (total_students - 1) / 2

-- Probability of selecting one from seventh grade and one from eighth grade
def prob_mixed_grade := (favorable_outcomes : ℚ) / possible_outcomes

theorem prob_mixed_grade : prob_mixed_grade = 2 / 3 :=
by
  -- calculation here
  sorry

end prob_seventh_grade_prob_mixed_grade_l74_74405


namespace sum_of_values_sum_of_all_possible_values_l74_74317

theorem sum_of_values (x : ℝ) (hx : x^2 = 36) : x = 6 ∨ x = -6 :=
begin
  simp [hx],
  have h₁ : x = 6 ∨ x = -6, from or.intro_right (x = 6) rfl,
  exact h₁,
end

noncomputable def sum_of_possible_values : ℝ :=
if hx : ∃ x : ℝ, x^2 = 36 then 6 + (-6) else 0

theorem sum_of_all_possible_values : sum_of_possible_values = 0 :=
begin
  simp [sum_of_possible_values],
  sorry
end

end sum_of_values_sum_of_all_possible_values_l74_74317


namespace sum_of_prime_values_of_h_l74_74242

-- Define the function h(n) as per the conditions.
def h (n : ℕ) : ℤ := n^3 - 150*n + 300

-- Define a predicate to check if a number is prime.
def is_prime (p : ℤ) : Prop := p > 1 ∧ (∀ d : ℤ, d ∣ p → d = 1 ∨ d = p)

-- Define the problem statement.
theorem sum_of_prime_values_of_h :
  (∑ n in Finset.filter (λ n, is_prime (h n)) (Finset.range 10), h n) = 151 := 
sorry

end sum_of_prime_values_of_h_l74_74242


namespace max_abs_z_minus_z1_l74_74270

noncomputable def z1 : ℂ := 2 - 2 * complex.I

theorem max_abs_z_minus_z1 :
  (∀ z : ℂ, |z| = 1 → ∃ z' : ℂ, |z' - z1| ≤ |z - z1|) ∧
  (∀ z : ℂ, |z| = 1 → |z - z1| ≤ 2 * real.sqrt 2 + 1) :=
by {
  sorry
}

end max_abs_z_minus_z1_l74_74270


namespace problem_solution_l74_74385

/-- Main theorem to prove: given \((1 + sqrt 2)^5 = a + b * sqrt 2\), with \(a\) and \(b\) positive integers,
prove that \(a + b = 70\). --/
theorem problem_solution : 
  ∃ (a b : ℤ), 0 < a ∧ 0 < b ∧ (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 ∧ a + b = 70 :=
begin
  sorry
end

end problem_solution_l74_74385


namespace sqrt_490000_is_700_l74_74754

theorem sqrt_490000_is_700 : Real.sqrt 490000 = 700 := by
  have h1 : 490000 = (7^2) * (100^2) := by norm_num
  rw [h1, Real.sqrt_mul (Real.pow_pos (by norm_num) 2) (Real.pow_pos (by norm_num) 2)]
  rw [Real.sqrt_sq, Real.sqrt_sq]
  norm_num
  sorry

end sqrt_490000_is_700_l74_74754


namespace collinear_A_X_Y_l74_74005

open Classical

-- Defining points and geometrical constructs
variables {A B C D E F P Q X Y : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited P] [Inhabited Q] [Inhabited X] [Inhabited Y]

-- Conditions
noncomputable def is_parallelogram (A B C D : Type*) : Prop := sorry
noncomputable def is_reflection (P X DE : Type*) : Prop := sorry

-- Geometrical assumptions
axiom parallelogram_ABCD : is_parallelogram A B C D
axiom angle_DAB_lt_90 : sorry
axiom E_on_BC : sorry ∈ line B C
axiom AE_eq_AB : distance A E = distance A B
axiom F_on_CD : sorry ∈ line C D
axiom AF_eq_AD : distance A F = distance A D
axiom circumcircle_CEF_intersects_AE_at_P : on_circle P (circumcircle C E F)
axiom circumcircle_CEF_intersects_AF_at_Q : on_circle Q (circumcircle C E F)
axiom X_reflection_of_P_over_DE : is_reflection P X (line D E)
axiom Y_reflection_of_Q_over_BF : is_reflection Q Y (line B F)

-- Proving the collinearity of A, X, Y
theorem collinear_A_X_Y : collinear ({A, X, Y} : set Type*) :=
sorry

end collinear_A_X_Y_l74_74005


namespace fraction_cube_l74_74474

theorem fraction_cube (a b : ℚ) (h : (a / b) ^ 3 = 15625 / 1000000) : a / b = 1 / 4 :=
by
  sorry

end fraction_cube_l74_74474


namespace princesses_count_l74_74193

theorem princesses_count (x y : ℕ) (h1 : x + y = 22) (h2 : y = 6 + x) : x = 8 :=
by
  -- Definitions to set up the conditions
  have h3 : x + (6 + x) = 22, from Eq.trans (congrArg (· + x) h2) h1
  have h4 : 2 * x + 6 = 22, from Nat.add_sub_cancel' h3
  have h5 : 2 * x = 16, from Eq.trans' (eq_sub_of_add_eq h4) (by Nat.add_sub_cancel' 6 22)
  have h6 : x = 8, from Nat.mul_right_inj (Ne.symm zero_ne_two) h5
  exact h6
   

end princesses_count_l74_74193


namespace clock_angle_minutes_l74_74761

theorem clock_angle_minutes (n : ℕ) (h1 : 0 < n) (h2 : n < 720) (h3 : abs ((11 * n) / 2) = 1) :
  n = 262 ∨ n = 458 :=
sorry

end clock_angle_minutes_l74_74761


namespace best_graph_representation_l74_74395

theorem best_graph_representation :
  let city_traffic_slow_increase := (d : ℝ) (t : ℝ) => t ∈ (0, t₁) → f t = m₁ * t
  let highway_fast_increase := (d : ℝ) (t : ℝ) => t ∈ (t₁, t₂) → f t = f t₁ + m₂ * (t - t₁)
  let mall_constant := (d : ℝ) (t : ℝ) => t ∈ (t₂, t₃) → f t = c₁
  let friend_constant := (d : ℝ) (t : ℝ) => t ∈ (t₃, t₄) → f t = c₂
  let highway_slow_decrease := (d : ℝ) (t : ℝ) => t ∈ (t₄, t₅) → f t = c₃ - m₃ * (t - t₄)
  let traffic_jam := (d : ℝ) (t : ℝ) => t ∈ (t₅, t₆) → f t = c₄
  let highway_fast_decrease := (d : ℝ) (t : ℝ) => t ∈ (t₆, t₇) → f t = f t₆ - m₄ * (t - t₆)
  let city_traffic_slow_decrease := (d : ℝ) (t : ℝ) => t ∈ (t₇, t₈) → f t = m₅ * (t₈ - t)
  let graph_C := (t : ℝ) → (d : ℝ) in
  
        graph_C = city_traffic_slow_increase ⟦0, t₁⟧
               ∪ highway_fast_increase ⟦t₁, t₂⟧
               ∪ mall_constant ⟦t₂, t₃⟧
               ∪ friend_constant ⟦t₃, t₄⟧
               ∪ highway_slow_decrease ⟦t₄, t₅⟧
               ∪ traffic_jam ⟦t₅, t₆⟧
               ∪ highway_fast_decrease ⟦t₆, t₇⟧  
               ∪ city_traffic_slow_decrease ⟦t₇, t₈⟧
               → 
              
              result = "C" := 
sorry

end best_graph_representation_l74_74395


namespace relationship_among_log_sin_exp_l74_74635

theorem relationship_among_log_sin_exp (x : ℝ) (h₁ : 0 < x) (h₂ : x < 1) (a b c : ℝ) 
(h₃ : a = Real.log 3 / Real.log x) (h₄ : b = Real.sin x)
(h₅ : c = 2 ^ x) : a < b ∧ b < c := 
sorry

end relationship_among_log_sin_exp_l74_74635


namespace verify_cathy_stats_verify_chris_stats_verify_clara_stats_l74_74201

def hours_worked_per_week (staff: String) : ℕ :=
  if staff = "Cathy" then 20
  else if staff = "Chris" then 20
  else if staff = "Clara" then 20
  else 0

def hourly_rate (staff: String) : ℕ :=
  if staff = "Cathy" then 12
  else if staff = "Chris" then 14
  else if staff = "Clara" then 13
  else 0

def bonus (staff: String) : ℕ :=
  if staff = "Cathy" then 50
  else if staff = "Chris" then 70
  else if staff = "Clara" then 60
  else 0

def total_hours (staff : String) : ℕ :=
  if staff = "Cathy" then 180
  else if staff = "Chris" then 170
  else if staff = "Clara" then 130
  else 0

def regular_earnings (staff : String) : ℕ :=
  if staff = "Cathy" then 2160
  else if staff = "Chris" then 2380
  else if staff = "Clara" then 1690
  else 0

def total_earnings (staff : String) : ℕ :=
  if staff = "Cathy" then 2210
  else if staff = "Chris" then 2450
  else if staff = "Clara" then 1750
  else 0

theorem verify_cathy_stats :
  let hours = hours_worked_per_week "Cathy" * 8 + 20
  let earnings = hours * hourly_rate "Cathy"
  let total_earnings = earnings + bonus "Cathy"
  (total_hours "Cathy" = hours) ∧
  (regular_earnings "Cathy" = earnings) ∧
  (total_earnings "Cathy" = total_earnings) :=
by {
  sorry
}

theorem verify_chris_stats :
  let hours = (hours_worked_per_week "Chris" * 7) + 30
  let earnings = hours * hourly_rate "Chris"
  let total_earnings = earnings + bonus "Chris"
  (total_hours "Chris" = hours) ∧
  (regular_earnings "Chris" = earnings) ∧
  (total_earnings "Chris" = total_earnings) :=
by {
  sorry
}

theorem verify_clara_stats :
  let hours = hours_worked_per_week "Clara" * 6.5
  let earnings = hours * hourly_rate "Clara"
  let total_earnings = earnings + bonus "Clara"
  (total_hours "Clara" = hours) ∧
  (regular_earnings "Clara" = earnings) ∧
  (total_earnings "Clara" = total_earnings) :=
by {
  sorry
}

end verify_cathy_stats_verify_chris_stats_verify_clara_stats_l74_74201


namespace sequence_product_2014_l74_74293

noncomputable def sequence : ℕ → ℝ
| 0     := 2
| (n+1) := (1 + sequence n) / (1 - sequence n)

theorem sequence_product_2014 :
  ∏ i in (Finset.range 2014), sequence i = -6 := sorry

end sequence_product_2014_l74_74293


namespace point_in_third_quadrant_l74_74248

def imaginary_unit : ℂ := complex.I

theorem point_in_third_quadrant : 
  let z := imaginary_unit * (imaginary_unit - 1) in 
  z.re < 0 ∧ z.im < 0 :=
by {
  let z := imaginary_unit * (imaginary_unit - 1),
  have h : z = -1 - complex.I, from by {
    simp [imaginary_unit, complex.I_mul, complex.I_sub_nat_mul_I],
    ring,
  },
  rw h,
  simp
  sorry
}

end point_in_third_quadrant_l74_74248


namespace estimate_survival_probability_l74_74774

noncomputable def survival_probability
    (a : List ℕ) 
    (b : List ℕ) 
    (rates : List ℝ) : Option ℝ :=
if h : a.length = b.length ∧ b.length = rates.length ∧ 
    List.all (List.zipWith (· = ·) rates (List.zipWith (λ x y => (y:ℝ) / x) a b)) then 
    some 0.9 
else none

theorem estimate_survival_probability (a b : List ℕ) (rates : List ℝ) 
    (h : a.length = b.length ∧ b.length = rates.length ∧ 
        List.all (List.zipWith (· = ·) rates (List.zipWith (λ x y => (y:ℝ) / x) a b))) :
    survival_probability a b rates = some 0.9 :=
sorry

end estimate_survival_probability_l74_74774


namespace smallest_number_neg3_l74_74877

def smallest_number_in_set (s : Set ℤ) (n : ℤ) : Prop :=
  ∀ m ∈ s, n ≤ m

theorem smallest_number_neg3 : smallest_number_in_set ({-3, 2, -2, 0} : Set ℤ) (-3) :=
by
  intro m hm
  cases hm
  case inl hm_eq { rw [hm_eq] }
  case inr hm_in {
    cases hm_in
    case inl hm_eq { rw [hm_eq] }
    case inr hm_in {
      cases hm_in
      case inl hm_eq { rw [hm_eq] }
      case inr hm_eq { rw [hm_eq] }
    }
  }
  show -3 ≤ m
  -- each comparison will be trivial and skipped 
  sorry

end smallest_number_neg3_l74_74877


namespace black_stone_at_center_l74_74338

theorem black_stone_at_center (stones : ℕ) (position : ℕ) (initial_colors : Finₓ (stones + 1) → Bool)
  (h_stones : stones = 2020)
  (h_initial : initial_colors position = true)
  (h_condition : ∀ {i : Finₓ (stones + 1)}, (i ≠ 0 ∧ i ≠ Finₓ.last _) → ∃ j, (initial_colors j = false ∧ (j ≠ i - 1 ∧ j ≠ i + 1) → initial_colors (i - 1) = initial_colors (i + 1))) :
  ∃ (num_operations : ℕ), (∀ i : Finₓ (stones + 1), initial_colors i = true) ↔ position = stones / 2 + 1 :=
by
  sorry

end black_stone_at_center_l74_74338


namespace solution_set_l74_74620

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn (a b : ℝ) : f (a + b) = f a + f b - 1
axiom monotonic (x y : ℝ) : x ≤ y → f x ≤ f y
axiom initial_condition : f 4 = 5

theorem solution_set : {m : ℝ | f (3 * m^2 - m - 2) < 3} = {m : ℝ | -4/3 < m ∧ m < 1} :=
by
  sorry

end solution_set_l74_74620


namespace proof_a2016_add_b2016_l74_74900

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def b (n : ℕ) : ℝ := sorry

theorem proof_a2016_add_b2016 (θ : ℝ) (hθ : θ = real.pi / 2) :
  a 2016 + b 2016 = 1 :=
by
  sorry

end proof_a2016_add_b2016_l74_74900


namespace count_valid_permutations_l74_74346

def is_valid_permutation (l : List ℕ) : Prop :=
  l.length = 5 ∧ l.head ≠ 0 ∧ l.perm (List.ofFn (λ i, [7, 5, 5, 1, 0].nthLe i (by simp)) i)

theorem count_valid_permutations : Nat.card { l // is_valid_permutation l } = 48 :=
by
  sorry

end count_valid_permutations_l74_74346


namespace cosA_value_area_of_triangle_l74_74678

noncomputable def cosA (a b c : ℝ) (cos_C : ℝ) : ℝ :=
  if (a ≠ 0 ∧ cos_C ≠ 0) then (2 * b - c) * cos_C / a else 1 / 2

noncomputable def area_triangle (a b c : ℝ) (cosA_val : ℝ) : ℝ :=
  let S := a * b * (Real.sqrt (1 - cosA_val ^ 2)) / 2
  S

theorem cosA_value (a b c : ℝ) (cos_C : ℝ) : a * cos_C = (2 * b - c) * (cosA a b c cos_C) → cosA a b c cos_C = 1 / 2 :=
by
  sorry

theorem area_of_triangle (a b c : ℝ) (cos_A : ℝ) (cos_A_proof : a * cos_C = (2 * b - c) * cos_A) (h₀ : a = 6) (h₁ : b + c = 8) : area_triangle a b c cos_A = 7 * Real.sqrt 3 / 3 :=
by
  sorry

end cosA_value_area_of_triangle_l74_74678


namespace orthocenter_exists_l74_74797

noncomputable def orthocenter_construction (A B C O : Point) (circumcircle : set Point) 
  (h1 : circumcenter O circumcircle ABC) : Point :=
sorry

theorem orthocenter_exists (A B C O : Point) (circumcircle : set Point)
  (h1 : circumcenter O circumcircle ABC) :
  ∃ H, orthocenter H triangle ABC :=
sorry

end orthocenter_exists_l74_74797


namespace percentage_less_than_l74_74494

theorem percentage_less_than (x y : ℕ) (h : y = x * 17 / 10) : (y - x) * 100 / y = 41.18 := 
by 
  sorry

end percentage_less_than_l74_74494


namespace winning_votes_l74_74798

-- Definitions based on conditions
def two_candidates : Prop := -- There were two candidates in the election
def winner_percentage : ℚ := 0.62 -- The percentage of votes the winner got
def votes_difference : ℕ := 300 -- The difference in votes between winner and loser

-- Main Lean 4 statement of the problem
theorem winning_votes (V : ℕ) (hV : V = 1250) : 
  0.62 * V = 775 := by
  sorry

end winning_votes_l74_74798


namespace total_revenue_l74_74172

-- Definitions based on the conditions
def ticket_price : ℕ := 25
def first_show_tickets : ℕ := 200
def second_show_tickets : ℕ := 3 * first_show_tickets

-- Statement to prove the problem
theorem total_revenue : (first_show_tickets * ticket_price + second_show_tickets * ticket_price) = 20000 :=
by
  sorry

end total_revenue_l74_74172


namespace vertex_of_quadratic_function_l74_74765

theorem vertex_of_quadratic_function :
  ∀ (x : ℝ), (2 * (x - 3)^2 + 1) = y → exists h k : ℝ, (h, k) = (3, 1) := 
by
  intros x h k
  sorry

end vertex_of_quadratic_function_l74_74765


namespace sum_of_integer_solutions_l74_74601

-- Definitions based on conditions
def quartic_eqn (x : ℝ) : Prop := x^4 - 29 * x^2 + 180 = 0

-- Theorem stating that the sum of all integer solutions to the quartic equation is 0
theorem sum_of_integer_solutions : 
  (∑ x in {x : ℤ | quartic_eqn x}, x) = 0 :=
by
  sorry

end sum_of_integer_solutions_l74_74601


namespace parallelepiped_properities_l74_74076

noncomputable def length_A1A := 32

noncomputable def volume_parallelepiped := 8192

noncomputable def radius_sphere := 4 * Real.sqrt 17

-- Given conditions for the Lean theorem statement
theorem parallelepiped_properities 
    (A1 A AD BC B1 C C1 K D1 : Point)
    (omega : Sphere)
    (H1 : Perpendicular (Edge A1 A) (Face A B C D)) 
    (H2 : Touches omega (Edge B B1))
    (H3 : Touches omega (Edge B1 C1))
    (H4 : Touches omega (Edge C1 C))
    (H5 : Touches omega (Edge C B))
    (H6 : Touches omega (Edge C1 D1))
    (H7 : Touches omega (Edge AD))
    (H8 : Length (Segment C1 K) = 16)
    (H9 : Length (Segment K D1) = 1)
  : Length (Edge A1 A) = length_A1A ∧ Volume (Parallelepiped A B C D A1 B1 C1 D1) = volume_parallelepiped ∧ Radius omega = radius_sphere :=
  sorry

end parallelepiped_properities_l74_74076


namespace additional_land_cost_l74_74565

noncomputable def initial_land := 300
noncomputable def final_land := 900
noncomputable def cost_per_square_meter := 20

theorem additional_land_cost : (final_land - initial_land) * cost_per_square_meter = 12000 :=
by
  -- Define the amount of additional land purchased
  let additional_land := final_land - initial_land
  -- Calculate the cost of the additional land            
  show additional_land * cost_per_square_meter = 12000
  sorry

end additional_land_cost_l74_74565


namespace distance_AD_bounds_l74_74742

noncomputable def point (α : Type) := α × α

variables {α : Type} [real α] (O A B C D : point α)

-- Coordinates and relevant conditions
def point_B_due_east_of_A (A B : point α) : Prop := B.1 = A.1 + 1 ∧ B.2 = A.2
def point_C_due_north_of_B (B C : point α) : Prop := C.1 = B.1 ∧ C.2 = B.2 + 1
def distance_between_A_and_C (A C : point α) : α := dist A C = 15 * sqrt 2
def angle_BAC_30_degrees (A B C : point α) : Prop := angle A B C = 30
def point_D_due_north_of_C (C D : point α) : Prop := D.1 = C.1 ∧ D.2 = C.2 + 25

-- Main theorem
theorem distance_AD_bounds (A B C D : point α)
    (hAB : point_B_due_east_of_A A B) 
    (hBC : point_C_due_north_of_B B C) 
    (hAC : distance_between_A_and_C A C) 
    (hBAC : angle_BAC_30_degrees A B C) 
    (hCD : point_D_due_north_of_C C D) : 
  44 < dist A D ∧ dist A D < 45 := 
sorry

end distance_AD_bounds_l74_74742


namespace max_fraction_value_l74_74015

noncomputable def S (n : ℕ) : ℕ := 2^n - 1

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 1 else S n - S (n - 1)

theorem max_fraction_value : 
  (∀ n : ℕ, S n = 2^n - 1) →
  (∃ n : ℕ, 
    (∀ k : ℕ, k > 0 → k < n → a k ≤ a n) ∧ 
    (a 1 = 1) ∧ 
    (∀ m : ℕ, m ≥ 2 → a m = 2^(m-1)) ∧ 
    (∃ n : ℕ, max_value = (1 : ℝ) / 15)) :=
begin
  intros,
  sorry
end

end max_fraction_value_l74_74015


namespace four_digit_number_exists_l74_74206

noncomputable def is_valid_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000
  
noncomputable def satisfies_first_division_scheme (n d : ℕ) : Prop :=
  d < 10 ∧ d > 1 ∧ ((n / d) >= 10 ∧ (n / d) < 100) ∧ ((n % d) >= 10 ∧ (n % d) < 100)

noncomputable def satisfies_second_division_scheme (n d : ℕ) : Prop :=
  d < 10 ∧ d > 1 ∧ ((n / d) >= 10 ∧ (n / d) < 100) ∧ ((n / d) >= 10 ∧ (n / d) < 100) ∧ (n / d) / 10 = 1

theorem four_digit_number_exists :
  ∃ (n : ℕ), is_valid_four_digit(n) ∧ 
             (∃ d1 : ℕ, satisfies_first_division_scheme(n, d1)) ∧ 
             (∃ d2 : ℕ, satisfies_second_division_scheme(n, d2)) := by
  sorry
  
end four_digit_number_exists_l74_74206


namespace probability_neither_selects_D_l74_74101

noncomputable def drawD_is_random : Prop :=
  let projects := ['A', 'B', 'C', 'D']
  let draw := projects.random_element
  draw = 'D' ↔ "random"

theorem probability_neither_selects_D : (fin 1 / 2 : ℚ) :=
by
  let projects := ['A', 'B', 'C', 'D']
  let choices := list.product projects projects
  let favorable_outcomes := choices.filter (λ x => fst x ≠ 'D' ∧ snd x ≠ 'D')
  let total_outcomes := list.length choices
  let favorable_count := list.length favorable_outcomes
  (favorable_count : ℚ) / (total_outcomes : ℚ) = 1 / 2 := by 
  simp [total_outcomes == 12, favorable_count == 6, favorable_count / total_outcomes == 1 / 2]
  sorry

end probability_neither_selects_D_l74_74101


namespace count_correct_inequalities_l74_74251

-- Define the conditions
variables (a b : ℝ)
axiom pos_a : a > 0
axiom neg_b : b < 0
axiom sum_pos : a + b > 0

-- Define the inequalities
def ineq1 := 1/a < 1/b
def ineq2 := 1/a > 1/b
def ineq3 := a^3 * b < a * b^3
def ineq4 := a^3 < a * b^2
def ineq5 := a^2 * b < b^3

-- Define the theorem to prove the number of correct inequalities
theorem count_correct_inequalities : 
  (¬ ineq1 ∧ ineq2 ∧ ineq3 ∧ ¬ ineq4 ∧ ineq5) → 3 := by
  sorry

end count_correct_inequalities_l74_74251


namespace sin_sum_of_acute_triangle_l74_74419

theorem sin_sum_of_acute_triangle (α β γ : ℝ) (h_acute : α < π / 2 ∧ β < π / 2 ∧ γ < π / 2) (h_sum : α + β + γ = π) : 
  sin α + sin β + sin γ > 2 :=
sorry

end sin_sum_of_acute_triangle_l74_74419


namespace distinct_factors_count_l74_74916

theorem distinct_factors_count : 
  let n := 4^5 * 5^2 * 6^3 * (nat.factorial 7) in 
  let prime_factorization_n := 
    (2^17 * 3^5 * 5^3 * 7^1 : ℕ) in 
  (factors_count prime_factorization_n = 864) :=
by sorry

def factors_count (n : ℕ) : ℕ :=
if n = 0 then 0 else
  let p := nat.factorization n in 
  (p.find 2 + 1) * (p.find 3 + 1) * (p.find 5 + 1) * (p.find 7 + 1)

end distinct_factors_count_l74_74916


namespace percentage_increase_l74_74708

theorem percentage_increase (initial_amount final_amount : ℝ) (h1 : initial_amount = 100) (h2 : final_amount = 110) :
  (((final_amount - initial_amount) / initial_amount) * 100) = 10 :=
by
  rw [h1, h2]
  -- sorry

end percentage_increase_l74_74708


namespace josh_siblings_count_l74_74000

theorem josh_siblings_count (initial_candies : ℕ) (candies_per_sibling : ℕ) (josh_eats : ℕ) (remaining_candies : ℕ) :
  let S := (initial_candies - 2 * (remaining_candies + josh_eats)) / (2 * candies_per_sibling)
  in initial_candies = 100 ∧ candies_per_sibling = 10 ∧ josh_eats = 16 ∧ remaining_candies = 19 → S = 3 :=
by
  intros
  -- Proof would go here
  sorry

end josh_siblings_count_l74_74000


namespace brenda_leads_by_5_l74_74571

theorem brenda_leads_by_5
  (initial_lead : ℕ)
  (brenda_play : ℕ)
  (david_play : ℕ)
  (h_initial : initial_lead = 22)
  (h_brenda_play : brenda_play = 15)
  (h_david_play : david_play = 32) :
  initial_lead + brenda_play - david_play = 5 :=
by {
  rw [h_initial, h_brenda_play, h_david_play],
  norm_num, -- simplify to get the answer
  sorry
}

end brenda_leads_by_5_l74_74571


namespace sum_of_powers_l74_74956

/-
Given:
1. A sequence \( \{b_n\} \) defined by \( b_n = 2^{2n+1} \).
2. \( S_n \) is the sum of the first \( n \) terms of \( \{b_n\} \).

Prove:
The sum \( S_n \) is equal to \( \frac{n(n+3)}{2} \).
-/

theorem sum_of_powers (n : ℕ) : 
  let b : ℕ → ℕ := λ n, 2^(2*n+1) in
  let S : ℕ → ℕ := λ n, (range n).sum (λ i, b (i + 1)) in
  S n = n * (n + 3) / 2 :=
by 
  sorry

end sum_of_powers_l74_74956


namespace max_in_C_l74_74006

def int_part (x : ℕ) : ℕ := nat.floor (real.sqrt x)

def S (x : ℕ) : ℕ := x + 1

def E (x : ℕ) : ℕ := x - int_part x ^ 2

def C : set (ℕ → ℕ) :=
  {f | ∃ g ∈ {S, E}, f = g ∨ 
       ∃ h k ∈ C, (f = h + k ∨ f = h * k ∨ f = h ∘ k)}

theorem max_in_C (f g : ℕ → ℕ)
  (H1 : f ∈ C)
  (H2 : g ∈ C) :
  (λ x, max (f x - g x) 0) ∈ C := 
sorry

end max_in_C_l74_74006


namespace ratio_proportion_l74_74546

theorem ratio_proportion (a b : ℚ) (h : a / b = 4 / 3) : (1 / 3) / (1 / 4) = a / b := 
begin
  sorry
end

end ratio_proportion_l74_74546


namespace base12_addition_l74_74872

theorem base12_addition : ∀ a b : ℕ, a = 956 ∧ b = 273 → (a + b) = 1009 := by
  sorry

end base12_addition_l74_74872


namespace sequence_unbounded_l74_74169

noncomputable def sequence : ℕ → ℤ
| 0     := 1
| 1     := 1
| 2     := 1
| (n+3) := -sequence n - sequence (n+1)

theorem sequence_unbounded : ∀ M : ℝ, ∃ n : ℕ, |(sequence n : ℤ)| > M := by
  sorry

end sequence_unbounded_l74_74169


namespace infinite_solutions_iff_a_eq_neg12_l74_74944

theorem infinite_solutions_iff_a_eq_neg12 {a : ℝ} : 
  (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + 16)) ↔ a = -12 :=
by 
  sorry

end infinite_solutions_iff_a_eq_neg12_l74_74944


namespace find_third_number_l74_74459

-- Definitions based on given conditions
def A : ℕ := 200
def C : ℕ := 100
def B : ℕ := 2 * C

-- The condition that the sum of A, B, and C is 500
def sum_condition : Prop := A + B + C = 500

-- The proof statement
theorem find_third_number : sum_condition → C = 100 := 
by
  have h1 : A = 200 := rfl
  have h2 : B = 2 * C := rfl
  have h3 : A + B + C = 500 := sorry
  sorry

end find_third_number_l74_74459


namespace max_true_statements_l74_74379

theorem max_true_statements
  (x y : ℝ) 
  (hx : x < 0) 
  (hy : y > 0) :
  let s1 := (1 / x < 1 / y)
  let s2 := (x^2 > y^2)
  let s3 := (x < y)
  let s4 := (x < 0)
  let s5 := (y > 0)
  in (s1 ∧ s2 ∧ s3 ∧ s4 ∧ s5 = false ∧ 
      s1 ∧ s2 ∧ s3 ∧ s4 ∨ 
      s1 ∧ s2 ∧ s3 ∧ s5 ∨ 
      s1 ∧ s2 ∧ s4 ∧ s5 ∨
      s1 ∧ s3 ∧ s4 ∧ s5 ∨
      s2 ∧ s3 ∧ s4 ∧ s5) :=
begin
  sorry
end

end max_true_statements_l74_74379


namespace problem_l74_74480

noncomputable theory

open Real

def option_A_correct : Prop := (1 + tan (28 * π / 180)) * (1 + tan (17 * π / 180)) = 2
def option_B_correct : Prop := cos (π / 9) * cos (2 * π / 9) * cos (4 * π / 9) = 1 / 8
def option_C_incorrect (θ : Real) : Prop := (3 * sin θ - 4 * cos θ) = max_univ (λ x, 3 * sin x - 4 * cos x) → sin θ ≠ 4 / 5
def option_D_correct (θ : Real) : Prop :=
  let a := (cos θ, sin θ) in
  let b := (sqrt 2, sqrt 2) in
  a.1 * b.1 + a.2 * b.2 = (3 * sqrt 2) / 5 → sin (2 * θ) = -16 / 25

theorem problem (θ : Real) :
  option_A_correct ∧ option_B_correct ∧ option_C_incorrect θ ∧ option_D_correct θ :=
by
  sorry

end problem_l74_74480


namespace number_of_local_champions_l74_74720

-- Define the weight function w(c)
def w (a b c : ℤ) : ℤ := 
  let pairs := {p : ℤ × ℤ | a * p.1 + b * p.2 = c}
  pairs.toFinset.map (λ p, (|p.1| + |p.2|)).min'

-- Define the condition of local champion
def is_local_champion (a b c : ℤ) : Prop :=
  w a b c ≥ w a b (c + a) ∧
  w a b c ≥ w a b (c - a) ∧
  w a b c ≥ w a b (c + b) ∧
  w a b c ≥ w a b (c - b)

-- Given conditions
variables {a b : ℤ}
variables (a b > 1)

-- Prove the number of local champions
theorem number_of_local_champions : 
  (RelativelyPrime a b ∧ a > b > 1 ∧ ¬Even a ∧ ¬Even b) → 
    (∃ (count : ℕ), count = b - 1) ∧
  ((RelativelyPrime a b ∧ a > b > 1 ∧ (Even a ∨ Even b)) → 
    (∃ (count : ℕ), count = 2 * (b - 1)) := 
begin
  sorry
end

end number_of_local_champions_l74_74720


namespace total_games_played_l74_74681

theorem total_games_played (n : ℕ) (k : ℕ) (h_n : n = 50) (h_k : k = 4) : (n * (n - 1) / 2) * k = 4900 := by
  rw [h_n, h_k]
  have h1 : 50 * 49 / 2 = 1225 := by norm_num
  rw [h1]
  norm_num
  sorry

end total_games_played_l74_74681


namespace time_per_toy_is_3_l74_74870

-- Define the conditions
variable (total_toys : ℕ) (total_hours : ℕ)

-- Define the given condition
def given_condition := (total_toys = 50 ∧ total_hours = 150)

-- Define the statement to be proved
theorem time_per_toy_is_3 (h : given_condition total_toys total_hours) :
  total_hours / total_toys = 3 := by
sorry

end time_per_toy_is_3_l74_74870


namespace smallest_number_among_neg3_2_neg2_0_l74_74879

theorem smallest_number_among_neg3_2_neg2_0 : 
  ∀ (x : ℤ), x ∈ ({-3, 2, -2, 0} : set ℤ) → -3 ≤ x := 
begin
  intros x hx,
  fin_cases hx,
  repeat {simp},
  sorry
end

end smallest_number_among_neg3_2_neg2_0_l74_74879


namespace count_two_digit_powers_of_three_l74_74311

theorem count_two_digit_powers_of_three : 
  ∃ (n1 n2 : ℕ), 10 ≤ 3^n1 ∧ 3^n1 < 100 ∧ 10 ≤ 3^n2 ∧ 3^n2 < 100 ∧ n1 ≠ n2 ∧ ∀ n : ℕ, (10 ≤ 3^n ∧ 3^n < 100) → (n = n1 ∨ n = n2) ∧ n1 = 3 ∧ n2 = 4 := by
  sorry

end count_two_digit_powers_of_three_l74_74311


namespace postcard_cost_l74_74732

theorem postcard_cost (x : ℕ) (h₁ : 9 * x < 1000) (h₂ : 10 * x > 1100) : x = 111 :=
by
  sorry

end postcard_cost_l74_74732


namespace number_of_arrangements_l74_74061

theorem number_of_arrangements (students : List char) (A : char) (B : char) (C : char) :
  students = ['A', 'B', 'C', 'D', 'E', 'F'] →
  ∀ (orderings : List (List char)),
  (A ∉ [List.head orderings, List.head (List.reverse orderings)]) →
  (B :: C :: List.nil ∈ orderings ∨ C :: B :: List.nil ∈ orderings) →
  List.length orderings = List.permutations students →
  List.length orderings = 144 :=
begin
  intros,
  sorry
end

end number_of_arrangements_l74_74061


namespace values_of_m_l74_74999

theorem values_of_m (m : ℝ) :
  let A := {x : ℝ | x^2 - 9 * x - 10 = 0},
      B := {x : ℝ | m * x + 1 = 0} in
  (A ∪ B = A) ↔ (m = 0 ∨ m = 1 ∨ m = -1/10) :=
by
  let A := {x : ℝ | x^2 - 9 * x - 10 = 0}
  let B := {x : ℝ | m * x + 1 = 0}
  sorry

end values_of_m_l74_74999


namespace employees_hourly_wage_l74_74530

def num_employees := 20
def shirts_per_person_per_day := 20
def working_hours_per_day := 8
def pay_per_shirt := 5
def selling_price_per_shirt := 35
def nonemployee_expenses_per_day := 1000
def daily_profit := 9080

theorem employees_hourly_wage:
  let total_shirts_per_day := num_employees * shirts_per_person_per_day,
      total_revenue_per_day := total_shirts_per_day * selling_price_per_shirt,
      pay_for_shirts := total_shirts_per_day * pay_per_shirt,
      total_expenses := pay_for_shirts + nonemployee_expenses_per_day,
      total_wages := total_revenue_per_day - total_expenses - daily_profit,
      wages_per_employee_per_day := total_wages / num_employees,
      hourly_wage := wages_per_employee_per_day / working_hours_per_day
  in hourly_wage = 12 := by
  sorry

end employees_hourly_wage_l74_74530


namespace difference_between_sums_and_products_l74_74247

noncomputable def a : ℚ := (1 / 8)⁻¹
noncomputable def b : ℚ := (2024 - Real.pi)^0
noncomputable def c : ℝ := |1 - Real.sqrt 3|
noncomputable def d : ℝ := Real.sqrt 3 * 3

theorem difference_between_sums_and_products :
  let sum_rationals := (a + b : ℝ)
  let product_irrationals := (c * d : ℝ)
  (sum_rationals - product_irrationals) = 3 * Real.sqrt 3 :=
by
  -- Show that the definitions of a, b, c and d hold using sorry
  have ha : a = 8 := sorry
  have hb : b = 1 := sorry
  have hc : c = Real.sqrt 3 - 1 := sorry
  have hd : d = 3 * Real.sqrt 3 := sorry

  -- Simplify the expression using the definitions
  let sum_rationals := 8 + 1
  let product_irrationals := (Real.sqrt 3 - 1) * 3 * Real.sqrt 3
  have result : (sum_rationals - product_irrationals) = 3 * Real.sqrt 3 := sorry

  exact result

end difference_between_sums_and_products_l74_74247


namespace length_CD_l74_74130

variable (AB AM AC : ℝ) (x : ℝ)

theorem length_CD (h1 : AB ≠ 0)
                   (h2 : AM ≠ 0)
                   (h3 : AC ≠ 0)
                   (h4 : 0 ≤ x ∧ x ≤ 180) :
  let C : ℝ := AC * Math.cos x in C = AC * Math.cos x :=
sorry

end length_CD_l74_74130


namespace sister_height_is_correct_l74_74108

/-- Vlad's height is given as 6 feet, 3 inches. --/
def Vlad_height_feet : ℤ := 6
def Vlad_height_inches : ℕ := 3

/-- Height difference between Vlad and his sister is 41 inches. --/
def height_difference_inches : ℕ := 41

/-- Vlad's sister's height is some feet, 10 inches. --/
def Sister_height_inches : ℕ := 10

/-- Convert Vlad's height to inches. --/
def Vlad_height_total_inches : ℕ := (Vlad_height_feet * 12 + Vlad_height_inches : ℕ)

/-- Compute Vlad's sister's height in inches. --/
def Sister_height_total_inches : ℕ := Vlad_height_total_inches - height_difference_inches

/-- Compute Vlad's sister's height in feet. --/
def Sister_height_feet : ℤ := (Sister_height_total_inches / 12 : ℤ)

theorem sister_height_is_correct :
  Sister_height_feet = 2 :=
by
  sorry

end sister_height_is_correct_l74_74108


namespace find_phi_l74_74281

noncomputable def f (x ω φ : ℝ) := Real.cos (ω * x + φ)

theorem find_phi (
  ω φ : ℝ)
  (h_omega_pos : ω > 0)
  (h_phi_bound : |φ| ≤ π / 2)
  (h_min_value : f (- π / 4) ω φ = min (f (- π / 4) ω φ))
  (h_max_value : f (π / 4) ω φ = max (f (π / 4) ω φ))
  (h_monotonic_interval : ∀ x1 x2 ∈ Ioo (π / 18) (5 * π / 36), x1 < x2 → f x1 ω φ < f x2 ω φ )
  : φ = -π / 2 := 
sorry

end find_phi_l74_74281


namespace range_of_a_l74_74788

theorem range_of_a (a : ℝ) (A : Set ℝ) (hA : ∀ x, x ∈ A ↔ a / (x - 1) < 1) (h_not_in : 2 ∉ A) : a ≥ 1 := 
sorry

end range_of_a_l74_74788


namespace quadratic_roots_m_eq_2_quadratic_discriminant_pos_l74_74255

theorem quadratic_roots_m_eq_2 (x : ℝ) (m : ℝ) (h1 : m = 2) : x^2 + 2 * x - 3 = 0 ↔ (x = -3 ∨ x = 1) :=
by sorry

theorem quadratic_discriminant_pos (m : ℝ) : m^2 + 12 > 0 :=
by sorry

end quadratic_roots_m_eq_2_quadratic_discriminant_pos_l74_74255


namespace max_distance_from_circle_to_line_l74_74359

theorem max_distance_from_circle_to_line :
  let circle := λ (ρ θ : ℝ), ρ = 8 * real.sin θ in
  let line := λ (θ : ℝ), θ = π / 3 in
  ∀ (ρ θ : ℝ), circle ρ θ → ∀ (θ' : ℝ), line θ' → 
  (let C := (0, 4) in
  let r := 4 in
  let d := 2 in
  d + r = 6) := 
sorry

end max_distance_from_circle_to_line_l74_74359


namespace abs_neg_one_fourth_l74_74433

theorem abs_neg_one_fourth : |(- (1 / 4))| = (1 / 4) :=
by
  sorry

end abs_neg_one_fourth_l74_74433


namespace frequency_of_2_in_20231222_l74_74486

def count_occurrences (s : String) (c : Char) : Nat :=
  s.foldl (fun count ch => if ch = c then count + 1 else count) 0

theorem frequency_of_2_in_20231222 :
  let s := "20231222"
  let total_digits := String.length s
  let count_2 := count_occurrences s '2'
  total_digits = 8 →
  count_2 = 5 →
  count_2 / total_digits = 5 / 8 :=
by
     intro s total_digits count_2 h1 h2
     rw [h1, h2]
     exact rfl
     sorry

end frequency_of_2_in_20231222_l74_74486


namespace tracy_candies_l74_74102

variable (x : ℕ) -- number of candies Tracy started with

theorem tracy_candies (h1: x % 4 = 0)
                      (h2 : 46 ≤ x / 2 - 40 ∧ x / 2 - 40 ≤ 50) 
                      (h3 : ∃ k, 2 ≤ k ∧ k ≤ 6 ∧ x / 2 - 40 - k = 4) 
                      (h4 : ∃ n, x = 4 * n) : x = 96 :=
by
  sorry

end tracy_candies_l74_74102


namespace water_level_rate_l74_74152

-- Definitions based on the problem conditions
theorem water_level_rate
  (cube_side_length : ℝ) (cube_rate_fall : ℝ) (cylinder_radius : ℝ)
  (h_cube : cube_side_length = 1)
  (h_rate : cube_rate_fall = 0.01) -- in meters per second (1 cm/s)
  (h_radius : cylinder_radius = 1) :
  let dV_dt_cube := cube_side_length^2 * cube_rate_fall,
      dV_dt_cylinder := π * cylinder_radius^2 * (↑1 / π) 
  in 
  dV_dt_cylinder = dV_dt_cube :=
by {
  -- Definition of the volume change rates in cube and cylinder as per the problem
  sorry
}

end water_level_rate_l74_74152


namespace ratio_evaluation_l74_74217

theorem ratio_evaluation : (5^3003 * 2^3005) / (10^3004) = 2 / 5 := by
  sorry

end ratio_evaluation_l74_74217


namespace GPS_practical_reason_l74_74077

theorem GPS_practical_reason 
  (H1 : ∀ (mobilePhones : Type) (GPS_function : mobilePhones → Prop), ∃ (part_of_daily_life : Prop), GPS_function mobilePhones → part_of_daily_life)
  (H2 : ∀ (GPS_technology : Prop) (aid_for_travel : Prop), GPS_technology → aid_for_travel)
  : (∀ (choose_routes : Prop) (reduce_costs : Prop), choose_routes → reduce_costs) :=
sorry

end GPS_practical_reason_l74_74077


namespace largest_good_number_smallest_bad_number_l74_74940

def is_good_number (M : ℕ) : Prop :=
  ∃ a b c d : ℤ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

theorem largest_good_number :
  ∀ M : ℕ, is_good_number M ↔ M ≤ 576 :=
by sorry

theorem smallest_bad_number :
  ∀ M : ℕ, ¬ is_good_number M ↔ M ≥ 443 :=
by sorry

end largest_good_number_smallest_bad_number_l74_74940


namespace red_balls_in_box_l74_74793

theorem red_balls_in_box (initial_red_balls added_red_balls : ℕ) (initial_blue_balls : ℕ) 
  (h_initial : initial_red_balls = 5) (h_added : added_red_balls = 2) : 
  initial_red_balls + added_red_balls = 7 :=
by {
  sorry
}

end red_balls_in_box_l74_74793


namespace trapezoid_base_solutions_l74_74198

theorem trapezoid_base_solutions (A h : ℕ) (d : ℕ) (bd : ℕ → Prop)
  (hA : A = 1800) (hH : h = 60) (hD : d = 10) (hBd : ∀ (x : ℕ), bd x ↔ ∃ (k : ℕ), x = d * k) :
  ∃ m n : ℕ, bd (10 * m) ∧ bd (10 * n) ∧ 10 * (m + n) = 60 ∧ m + n = 6 :=
by
  simp [hA, hH, hD, hBd]
  sorry

end trapezoid_base_solutions_l74_74198


namespace solve_system_of_equations_l74_74817

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x - 3 * y = 5 ∧ 4 * x - 3 * y = 2 ∧ x = -1 ∧ y = -2 :=
by
  use [-1, -2]
  simp
  sorry -- proof of equivalence

end solve_system_of_equations_l74_74817


namespace probability_sum_divisible_by_3_l74_74330

open Finset

def first_eight_primes := {2, 3, 5, 7, 11, 13, 17, 19}

def residue_mod_3 (n : ℕ) : ℕ := n % 3

def count_valid_pairs (s : Finset ℕ) : ℕ :=
  (s.filter (λ x, residue_mod_3 x = 1)).card.choose 2 +
  (s.filter (λ x, residue_mod_3 x = 2)).card.choose 2

theorem probability_sum_divisible_by_3 :
  (count_valid_pairs first_eight_primes : ℚ) / (first_eight_primes.card.choose 2 : ℚ) = 9 / 28 :=
by 
  sorry

end probability_sum_divisible_by_3_l74_74330


namespace propositions_correct_l74_74187

/-- Define the propositions P1 to P4 based on the given mathematical statements --/
def P1 : Prop := ∀ x : ℝ, (x ≠ 1 ∧ x ≠ 2) → x^2 - 3*x + 2 ≠ 0
def P2 : Prop := ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0)
def P3 : Prop := ∀ m : ℝ, (m = 1 / 2) ↔ 
  (let line1 := (m+2)*x + 3*m*y + 1 = 0 
   ∧ let line2 := (m-2)*x + (m+2)*y - 3 = 0 
   in ∃ k : ℝ, line1.slope * line2.slope = -1)
def P4 : Prop := ∀ (m n : ℝ), 
  ((∃ x : ℝ, x^2 - m*x + n = 0) ↔ (∃ (y : ℝ), y^2 - n*y + m = 0)) → 
  (let curve := (x^2 / m) + (y^2 / n) = 1
   in ∃ k : ℝ, k ≠ 19)
  
/-- Lean statement asserting that P2 and P3 are true, while P1 and P4 are false --/
theorem propositions_correct : ({P1, P2, P3, P4} = {P2, P3}) :=
  sorry

end propositions_correct_l74_74187


namespace correct_conclusions_l74_74292

def quadratic_values (x : ℝ) : ℝ :=
  match x with
  | -2 => 5
  | -1 => 0
  | 0  => -3
  | 1  => -4
  | 3  => 0
  | _  => sorry

theorem correct_conclusions :
  let y (x : ℝ) := x^2 + x - 3 in
  let parabola_opens_upwards := 1 > 0 in
  let axis_of_symmetry := (1 - (1 - 3)) / 2 = 1 in
  let another_solution := y 4 = 5 in
  let interval_condition := ∀ x, -1 < x ∧ x < 3 → y x < 0 in
  let distance_between_roots := 3 - (-1) = 4 in
  ∀ conclusions_correct : ℕ,
    conclusions_correct = 4 ↔
    parabola_opens_upwards ∧
    axis_of_symmetry ∧
    another_solution ∧
    ¬interval_condition ∧
    distance_between_roots := sorry

end correct_conclusions_l74_74292


namespace linear_function_properties_l74_74188

def linear_function (x : ℝ) : ℝ := -2 * x + 1

theorem linear_function_properties :
  (∀ x, linear_function x = -2 * x + 1) ∧
  (∀ x₁ x₂, x₁ < x₂ → linear_function x₁ > linear_function x₂) ∧
  (linear_function 0 = 1) ∧
  ((∃ x, x > 0 ∧ linear_function x > 0) ∧ (∃ x, x < 0 ∧ linear_function x > 0) ∧ (∃ x, x > 0 ∧ linear_function x < 0))
  :=
by
  sorry

end linear_function_properties_l74_74188


namespace cross_product_correct_l74_74595

/-- Define the two vectors -/
def v1 : ℝ^3 := ![4, -2, 1]
def v2 : ℝ^3 := ![1, 3, -5]

/-- Define the expected result vector -/
def expected_result : ℝ^3 := ![7, 21, 14]

/-- Prove that the cross product of the two vectors equals the expected result -/
theorem cross_product_correct : v1 × v2 = expected_result := by
  sorry

end cross_product_correct_l74_74595


namespace nori_gave_more_crayons_to_Lea_than_Mae_l74_74042

-- Definitions for conditions
def initial_crayons : ℕ := 7 * 15
def crayons_given_to_Mae : ℕ := 12
def crayons_given_to_Rey : ℕ := 20
def crayons_left : ℕ := 25

-- Proving goal
theorem nori_gave_more_crayons_to_Lea_than_Mae :
  let crayons_used := crayons_given_to_Mae + crayons_given_to_Rey in
  let crayons_given_to_Lea := initial_crayons - crayons_used - crayons_left in
  crayons_given_to_Lea - crayons_given_to_Mae = 36 := by
  sorry

end nori_gave_more_crayons_to_Lea_than_Mae_l74_74042


namespace knight_responses_l74_74511

namespace KnightLiarProblem

variables {n k l : ℕ}
-- Variables defined as natural numbers (ℕ)

-- Condition: Total number of individuals is 2n where n > 5
def total_individuals : Prop := 2 * n > 10

-- Condition: Number of liars is fewer than the number of knights
def fewer_liars_than_knights : Prop := l < k

-- Condition: The number of responses knights and liars give
def number_of_responses := 4 * n

-- Condition: Liars (\(L\)) always lie and knights (\(R\)) always tell the truth
-- Given that all these conditions hold true, we need to prove the number of "Knight" responses is 4

theorem knight_responses : total_individuals → fewer_liars_than_knights → k > l → number_of_responses = 4 * n → (2 * k) = 4 :=
by
  intros
  -- Assuming the necessary conditions
  have h1 : total_individuals := sorry,
  have h2 : fewer_liars_than_knights := sorry,
  have h3 : k > l := sorry,
  have h4 : number_of_responses = 4 * n := sorry,
  -- Proof goes here (omitted)
  sorry

end KnightLiarProblem

end knight_responses_l74_74511


namespace angle_DEC_eq_90_l74_74537

noncomputable theory
open_locale classical

variables (A B C D E : Type*) [euclidean_space A B C] [circle_around A B C] [triangle_ABC A B C]
variables [right_angle (angle A C B = 90)] [midpoint_arc E A B C] [length_eq AC BD]

theorem angle_DEC_eq_90 :
  ∠ DEC = 90 :=
  sorry

end angle_DEC_eq_90_l74_74537


namespace velocity_function_displacement_function_l74_74763

noncomputable def acceleration (t : ℝ) : ℝ := 6 * t - 4

axiom initial_velocity : ℝ := 4
axiom initial_position : ℝ := 0

theorem velocity_function : (v : ℝ → ℝ) 
  (h_v : ∀ t, deriv (v t) = 6 * t - 4) 
  (h_v_initial : v 0 = initial_velocity) :
  v = λ t, 3 * t^2 - 4 * t + 4 := 
sorry

theorem displacement_function : (s : ℝ → ℝ) 
  (v : ℝ → ℝ) 
  (h_v : ∀ t, deriv (v t) = 6 * t - 4) 
  (h_s : ∀ t, deriv (s t) = v t) 
  (h_s_initial : s 0 = initial_position)
  (h_v_initial : v 0 = initial_velocity)
  (h_v_function : v = λ t, 3 * t^2 - 4 * t + 4) :
  s = λ t, t^3 - 2 * t^2 + 4 * t :=
sorry

end velocity_function_displacement_function_l74_74763


namespace intersectionPointDivision_l74_74085

variables (A B C D M N E F Q : Point)
variables (m n p q : ℕ)
variable [Involution Geometry] 

-- Definitions of point divisions
def isDividedInRatio (X Y S : Point) (r s : ℕ) :=  -- Definition for division in ratio
  (r / (r + s)) • (Y - X) + (s / (r + s)) • (S - X) = 0

-- Conditions from (a)
def quadrilateralDivisions := 
  isDividedInRatio A D M m n ∧
  isDividedInRatio B C N m n ∧
  isDividedInRatio A B E p q ∧
  isDividedInRatio C D F p q

-- Goal from (b)
theorem intersectionPointDivision :
  quadrilateralDivisions A B C D M N E F m n p q →
  let Q := intersectionPointOfLines M N E F in   -- Intersection of line MN and line EF
  isDividedInRatio E F Q m n ∧
  isDividedInRatio M N Q p q :=
begin
  sorry
end

end intersectionPointDivision_l74_74085


namespace limit_a_n_zero_l74_74604

noncomputable def a_n (c : ℝ) (n : ℕ) : ℝ :=
∫ x in c..1, n * x^(n-1) * (Real.log (1/x))^n

theorem limit_a_n_zero (c : ℝ) (hc : 0 < c ∧ c < 1) :
  tendsto (λ n, a_n c n) at_top (𝓝 0) :=
sorry

end limit_a_n_zero_l74_74604


namespace show_revenue_l74_74173

variable (tickets_first_show : Nat) (tickets_cost : Nat) (multiplicator : Nat)
variable (tickets_second_show : Nat := multiplicator * tickets_first_show)
variable (total_tickets : Nat := tickets_first_show + tickets_second_show)
variable (total_revenue : Nat := total_tickets * tickets_cost)

theorem show_revenue :
    tickets_first_show = 200 ∧ tickets_cost = 25 ∧ multiplicator = 3 →
    total_revenue = 20000 := 
by
    intros h
    sorry

end show_revenue_l74_74173


namespace parabola_point_focus_distance_l74_74959

/-- 
  Given a point P on the parabola y^2 = 4x, and the distance from P to the line x = -2
  is 5 units, prove that the distance from P to the focus of the parabola is 4 units.
-/
theorem parabola_point_focus_distance {P : ℝ × ℝ} 
  (hP : P.2^2 = 4 * P.1) 
  (h_dist : (P.1 + 2)^2 + P.2^2 = 25) : 
  dist P (1, 0) = 4 :=
sorry

end parabola_point_focus_distance_l74_74959


namespace train_speed_l74_74121

def train_length : ℝ := 400  -- Length of the train in meters
def crossing_time : ℝ := 40  -- Time to cross the electric pole in seconds

theorem train_speed : train_length / crossing_time = 10 := by
  sorry  -- Proof to be completed

end train_speed_l74_74121


namespace first_player_win_strategy_l74_74471

theorem first_player_win_strategy (a b : ℕ) (H1 : a > 0) (H2 : b > 0) :
  ∃ strategy : (ℕ → ℕ × ℕ) → (ℕ → ℕ × ℕ),
  (∀ (opponent_strat : ℕ → ℕ × ℕ) (n : ℕ), 
    let opp_move := opponent_strat n,
        my_move := strategy opponent_strat n
    in my_move = (if n = 0 then (a / 2, b / 2) else ((a + a / 2 - opp_move.1), (b + b / 2 - opp_move.2)))) ∧ 
  (∀ (opponent_strat : ℕ → ℕ × ℕ), 
    ∃ final_turn : ℕ, ∀ n ≥ final_turn, opponent_strat n = (0, 0) → 
      (strategy opponent_strat n) ≠ (0, 0)) :=
by
  sorry

end first_player_win_strategy_l74_74471


namespace average_of_first_20_even_numbers_l74_74126

theorem average_of_first_20_even_numbers : 
  let n := 20 
  let a1 := 2 
  let a20 := 40 
in (List.sum (List.map (λ k, 2 * k) (List.range n)) / n = 21) := by
  sorry

end average_of_first_20_even_numbers_l74_74126


namespace MN_parallel_AD_l74_74623

open Parallelogram Point

variable (A B C D M N : Point)

-- Definitions to indicate a parallelogram ABCD and points M and N
def is_parallelogram (A B C D : Point) : Prop :=
  (A,C).parallel (B,D) ∧ (A,B).parallel (D,C) ∧ (A,B).length = (C,D).length ∧ (B,C).length = (D,A).length

def is_angle_bisector (P Q R M : Point) : Prop :=
  ∠PQM = ∠MQ_R

def is_midpoint (M A B : Point) : Prop :=
  (dist A M = dist M B)

-- Main theorem statement
theorem MN_parallel_AD : 
  is_parallelogram A B C D → 
  is_angle_bisector A B M D → 
  is_angle_bisector B A M D → 
  is_angle_bisector C D N A → 
  is_angle_bisector D C N A → 
  (M ∈ Line CD) → 
  (N ∈ Line AB) → 
  Line MN ∥ Line AD :=
by sorry

end MN_parallel_AD_l74_74623


namespace curve_cartesian_eq_range_PA_PB_l74_74291

noncomputable def curve_parametric_eq := 
  ∀ (t: ℝ), 
  (x = (t^2 - 4) / (t^2 + 4)) ∧ (y = (8 * t) / (t^2 + 4))

theorem curve_cartesian_eq :
  (x, y : ℝ), (∃ t : ℝ, x = (t^2 - 4) / (t^2 + 4) ∧ y = (8 * t) / (t^2 + 4)) ↔ (x^2 + (y^2 / 4) = 1) := 
  sorry

noncomputable def point_p := (0, 1 : ℝ)
noncomputable def line_through_p (alpha t : ℝ) :=
  (x = t * cos alpha ∧ y = 1 + t * sin alpha)

theorem range_PA_PB :
  ∀ (α: ℝ), ∃ (A B : ℝ × ℝ), A ≠ B ∧
  (A.1 = t1 * cos α ∧ A.2 = 1 + t1 * sin α) ∧
  (B.1 = t2 * cos α ∧ B.2 = 1 + t2 * sin α) ∧
  (x^2 + (y^2 / 4) = 1) ∧
  let PA := sqrt ((A.1)^2 + (A.2 - 1)^2)
      PB := sqrt ((B.1)^2 + (B.2 - 1)^2)
  in (2 * sqrt(3) ≤ PA * PB) ∧ (PA * PB ≤ 4) :=
  sorry

end curve_cartesian_eq_range_PA_PB_l74_74291


namespace nurses_count_l74_74498

theorem nurses_count (D N : ℕ) (h1 : D + N = 456) (h2 : D * 11 = 8 * N) : N = 264 :=
by
  sorry

end nurses_count_l74_74498


namespace cyclic_quadrilateral_diagonals_tangent_circles_l74_74769

variables {A B C D P X Y Z T S: Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
  [AddGroup P] [AddGroup X] [AddGroup Y] [AddGroup Z] [AddGroup T] [AddGroup S]

/-- Prove that SP ⊥ ST for given cyclic quadrilateral ABCD with diagonals intersecting at P
and circles Γ and Ω with specified tangency and passing points properties. -/
theorem cyclic_quadrilateral_diagonals_tangent_circles 
  (cyclic_quad : CyclicQuadrilateral A B C D P)
  (circle_Γ : TangentCircleExtensions Γ A B C D P X Y Z T)
  (circle_Ω : ExternallyTangentCircle Ω Γ A B S) :
  Perpendicular (SP) (ST) :=
sorry

end cyclic_quadrilateral_diagonals_tangent_circles_l74_74769


namespace average_is_73_l74_74497

-- Define the scores obtained by Reeya
def scores : List ℕ := [55, 67, 76, 82, 85]

-- Define the number of subjects
def number_of_subjects := 5

-- Define the total score
def total_score := scores.sum

-- Define the average score
def average_score := total_score / number_of_subjects

-- Prove that the average score is 73
theorem average_is_73 : average_score = 73 := by
  -- Calculation of total score
  have h1 : total_score = 55 + 67 + 76 + 82 + 85 := by
    unfold total_score scores
    simp
  rw [h1]

  -- Calculation of average
  have h2 : average_score = 365 / 5 := by
    unfold average_score total_score number_of_subjects
    simp
  rw [h2]

  -- Conclude with the final calculation
  norm_num
  sorry

end average_is_73_l74_74497


namespace find_unique_number_l74_74502

theorem find_unique_number :
  ∃ (a b c d e : ℕ), 
    {a, b, c, d, e} = {1, 2, 3, 4, 5} ∧
    (a * 100 + b * 10 + c) % 4 = 0 ∧
    (b * 100 + c * 10 + d) % 5 = 0 ∧
    (c * 100 + d * 10 + e) % 3 = 0 ∧
    (a = 1) ∧ (b = 2) ∧ (c = 4) ∧ (d = 5) ∧ (e = 3) :=
sorry

end find_unique_number_l74_74502


namespace ellipse_equation_AB_perpendicular_to_MF_exists_point_M_l74_74290

-- Define the parabola C: x^2 = 4y
def parabola_C (x y : ℝ) : Prop := x^2 = 4 * y

-- Define focus F
def focus_F : ℝ × ℝ := (0, 1)

-- Define the line l passing through F (0, 1) intersecting the parabola at points A and B
def line_l (k x : ℝ) : ℝ := k * x + 1

-- Ellipse E with center at origin, foci on x-axis, vertex at F, and eccentricity sqrt(3)/2
def ellipse_E (a b : ℝ) : Prop := 
    a = 2 ∧ b = 1 ∧ (∀ x y, (x^2 / (a^2) + y^2 / (b^2) = 1))

-- Question 1: Prove the equation of ellipse E
theorem ellipse_equation :
    ∃ (a b : ℝ), a = 2 ∧ b = 1 ∧ (∀ x y, ellipse_E a b → (x^2 / 4 + y^2 = 1)) := sorry

-- Question 2: Prove that AB is perpendicular to MF
theorem AB_perpendicular_to_MF :
    ∀ (A B : ℝ × ℝ), A ≠ B → 
    parabola_C A.fst A.snd ∧ parabola_C B.fst B.snd ∧ 
    (∃ M, true) → -- Further conditions to express M
    sorry := sorry

-- Question 3: Prove the existence of M' and calculate the specific area
theorem exists_point_M'_area :
    ∃ (M' : ℝ × ℝ), M' ∈ ellipse_E 2 1 ∧ 
    (tangent_exists_to_parabola M'.fst M'.snd) ∧ (area_bounded 2) = 4/3 := sorry

end ellipse_equation_AB_perpendicular_to_MF_exists_point_M_l74_74290


namespace range_of_negative_a_l74_74238

theorem range_of_negative_a (a : ℝ) (h : a ≤ -2) : 
  ∀ x : ℝ, sin x * sin x + a * cos x + a * a ≥ 1 + cos x := 
by 
  sorry

end range_of_negative_a_l74_74238


namespace find_real_solutions_l74_74590

theorem find_real_solutions : 
  ∀ x : ℝ, 1 / ((x - 2) * (x - 3)) 
         + 1 / ((x - 3) * (x - 4)) 
         + 1 / ((x - 4) * (x - 5)) 
         = 1 / 8 ↔ x = 7 ∨ x = -2 :=
by
  intro x
  sorry

end find_real_solutions_l74_74590


namespace odd_numbers_in_first_15_rows_l74_74698

-- Defining a function to check if a binomial coefficient is odd.
def is_odd_binom (n k : ℕ) : Prop :=
  (bitwise_band k n) = k

-- Counting the number of odd binomial coefficients in Pascal's Triangle up to a given row.
def odd_count (r : ℕ) : ℕ :=
  (list.range (r + 1)).sum (λ n, (list.range (n + 1)).countp (λ k, is_odd_binom n k))

-- Proving the specific case of the first 15 rows.
theorem odd_numbers_in_first_15_rows : odd_count 14 = 511 :=
by sorry

end odd_numbers_in_first_15_rows_l74_74698


namespace trajectory_equation_range_for_a_l74_74584

-- Definition of conditions
def polar_equation_l1 (rho theta a : ℝ) : Prop :=
  rho = -1 / (Real.sin theta + a * Real.cos theta)

def polar_equation_l2 (rho theta a : ℝ) : Prop :=
  rho = 1 / (Real.cos theta - a * Real.sin theta)

-- Question 1: Prove the Cartesian equation of the trajectory C
theorem trajectory_equation (a x y : ℝ) :
  (polar_equation_l1 (sqrt (x*x + y*y)) (Real.atan2 y x) a) ∧
  (polar_equation_l2 (sqrt (x*x + y*y)) (Real.atan2 y x) a) →
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
sorry

-- Question 2: Prove the range for a
theorem range_for_a (a : ℝ) :
  (∀ x y, polar_equation_l1 (sqrt (x*x + y*y)) (Real.atan2 y x) a →
  polar_equation_l2 (sqrt (x*x + y*y)) (Real.atan2 y x) a →
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2) →
  (∃ d < sqrt 2 / 2, d = abs ((1/2 * a + 1 / 2) / sqrt (a * a + 1))) →
  a ≠ 1 →
  (a ≤ 1 ∧ a ≠ 1) ∨ a ≥ 1 :=
sorry

end trajectory_equation_range_for_a_l74_74584


namespace show_revenue_l74_74178

theorem show_revenue (tickets_first_showing : ℕ) 
                     (tickets_second_showing : ℕ) 
                     (ticket_price : ℕ) :
                      tickets_first_showing = 200 →
                      tickets_second_showing = 3 * tickets_first_showing →
                      ticket_price = 25 →
                      (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 :=
by
  intros h1 h2 h3
  have h4 : tickets_first_showing + tickets_second_showing = 800 := sorry -- Calculation step
  have h5 : (tickets_first_showing + tickets_second_showing) * ticket_price = 20000 := sorry -- Calculation step
  exact h5

end show_revenue_l74_74178


namespace coefficient_x2_in_expansion_is_240_l74_74355

theorem coefficient_x2_in_expansion_is_240 :
  let f := (λ x : ℚ, (2 * x - 1 / x) ^ 6) in
  ∀ (x : ℚ), x ≠ 0 → coefficient (f x) 2 = 240 := sorry

end coefficient_x2_in_expansion_is_240_l74_74355


namespace urn_probability_l74_74548

/-- An urn initially contains two red balls and two blue balls. John performs four rounds
    of a modified operation: at each round, he first removes one ball from the
    urn at random, then he draws one of the remaining balls at random and adds
    one more of the same color from a supplementary supply. At the end of four
    rounds, if the urn at each start has an equal number of balls of each color,
    calculate the probability that the urn contains three balls of each color 
    after the four operations. -/
theorem urn_probability : 
  ∀ urn : list (list ℕ), 
  (urn = [[2, 2]]) →
  (∀ i, (1 ≤ i ∧ i ≤ 4) → 
    ∃ j : list ℕ, 
      (urn = list.cons j [[2, 2]]) →
      (urn.nth (i - 1) = j) ∧ 
      list.count [[3, 3]] urn = 1) →
  probability_of (list.count [[3, 3]] urn = 1) = 1 / 216 :=
sorry

end urn_probability_l74_74548


namespace calculate_expression_l74_74560

noncomputable def sqrt_eight : ℝ := Real.sqrt 8
noncomputable def sin_forty_five : ℝ := Real.sin (Real.pi / 4)
def one_third_inverse : ℝ := (1 / 3)⁻¹

theorem calculate_expression : 
  sqrt_eight - 4 * sin_forty_five - one_third_inverse = -3 := 
by
  have h1 : sqrt_eight = 2 * Real.sqrt 2 := by 
    sorry
  have h2 : 4 * sin_forty_five = 2 * Real.sqrt 2 := by 
    sorry
  have h3 : one_third_inverse = 3 := by 
    sorry
  rw [h1, h2, h3]
  calc
    2 * Real.sqrt 2 - 2 * Real.sqrt 2 - 3 = 0 - 3 := by ring
    ... = -3 := by ring

end calculate_expression_l74_74560


namespace polynomial_correct_l74_74599

noncomputable def q (x : ℝ) : ℝ :=
  (1 / 6) * x^4 - (8 / 3) * x^3 - (14 / 3) * x^2 - (8 / 3) * x - (16 / 3)

theorem polynomial_correct :
  q 1 = -8 ∧ q 2 = -18 ∧ q 3 = -40 ∧ q 4 = -80 ∧ q 5 = -140 :=
by
  split; -- Splitting the conjunction into individual statements
  { show q 1 = -8, sorry },
  { split;
    { show q 2 = -18, sorry },
    { split;
      { show q 3 = -40, sorry },
      { split;
        { show q 4 = -80, sorry },
        { show q 5 = -140, sorry }
      }
    }
  }

end polynomial_correct_l74_74599


namespace employee_wage_is_correct_l74_74400

-- Define the initial conditions
def revenue_per_month : ℝ := 400000
def tax_rate : ℝ := 0.10
def marketing_rate : ℝ := 0.05
def operational_cost_rate : ℝ := 0.20
def wage_rate : ℝ := 0.15
def number_of_employees : ℕ := 10

-- Compute the intermediate values
def taxes : ℝ := tax_rate * revenue_per_month
def after_taxes : ℝ := revenue_per_month - taxes
def marketing_ads : ℝ := marketing_rate * after_taxes
def after_marketing : ℝ := after_taxes - marketing_ads
def operational_costs : ℝ := operational_cost_rate * after_marketing
def after_operational : ℝ := after_marketing - operational_costs
def total_wages : ℝ := wage_rate * after_operational

-- Compute the wage per employee
def wage_per_employee : ℝ := total_wages / number_of_employees

-- The proof problem statement ensuring the calculated wage per employee is 4104
theorem employee_wage_is_correct :
  wage_per_employee = 4104 := by 
  sorry

end employee_wage_is_correct_l74_74400


namespace max_checkers_under_attack_l74_74808

theorem max_checkers_under_attack : 
  ∃ (n : ℕ), n = 32 ∧ 
  ∀ (i j : ℕ), i < 8 ∧ j < 8 → 
  (i, j) ∉ boundary_positions → 
  checker_board_condition i j n := 
sorry

def boundary_positions (i j : ℕ) : Prop :=
(i = 0 ∨ i = 7) ∨ (j = 0 ∨ j = 7)

def checker_board_condition (i j n : ℕ) : Prop :=
∀ (k l : ℕ), (k, l) ∈ adjacent_positions (i, j) → checker_on_position k l

def adjacent_positions (i j : ℕ) : list (ℕ × ℕ) :=
[(i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]

def checker_on_position (i j : ℕ) : Prop :=
-- Omitted actual placement logic for brevity
sorry

end max_checkers_under_attack_l74_74808


namespace number_of_tangent_lines_through_origin_l74_74279

def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

def f_prime (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := f x₀ + f_prime x₀ * (x - x₀)

theorem number_of_tangent_lines_through_origin : 
  ∃! (x₀ : ℝ), x₀^3 - 3*x₀^2 + 4 = 0 := 
sorry

end number_of_tangent_lines_through_origin_l74_74279


namespace integer_solutions_equation_2255_l74_74084

theorem integer_solutions_equation_2255 : 
  (∃ x y : ℤ, 2^(2*x) - 3^(2*y) = 55) ∧
  (∀ x y x' y' : ℤ, 2^(2*x) - 3^(2*y) = 55 ∧ 2^(2*x') - 3^(2*y') = 55 → x = x' ∧ y = y') :=
by
  sorry

end integer_solutions_equation_2255_l74_74084


namespace problem1_problem2_l74_74898

theorem problem1 : (1 * (-2016 : ℝ)^0 + 32 * 2^(2/3) + (1/4)^(-1/2)) = 5 := 
by 
  sorry

-- Note: The second problem involves approximate equality. We typically do this by using an error term such as 0.001 for demonstration purposes.
theorem problem2 : (log 3 81 + log 10 20 + log 10 5 + 4^(log 4 2) + log 5 1) ≈ 8.301 := 
by 
  -- Here, we would use tactics to show that the difference is less than a small epsilon.
  -- For now, we provide a placeholder.
  sorry

end problem1_problem2_l74_74898


namespace least_number_subtracted_l74_74813

/-- The least number that must be subtracted from 50248 so that the 
remaining number is divisible by both 20 and 37 is 668. -/
theorem least_number_subtracted (n : ℕ) (x : ℕ ) (y : ℕ ) (a : ℕ) (b : ℕ) :
  n = 50248 → x = 20 → y = 37 → (a = 20 * 37) →
  (50248 - b) % a = 0 → 50248 - b < a → b = 668 :=
by
  sorry

end least_number_subtracted_l74_74813


namespace pat_moved_chairs_l74_74563

theorem pat_moved_chairs (total_chairs : ℕ) (carey_moved : ℕ) (left_to_move : ℕ) (pat_moved : ℕ) :
  total_chairs = 74 →
  carey_moved = 28 →
  left_to_move = 17 →
  pat_moved = total_chairs - left_to_move - carey_moved →
  pat_moved = 29 :=
by
  intros h_total h_carey h_left h_equation
  rw [h_total, h_carey, h_left] at h_equation
  exact h_equation

end pat_moved_chairs_l74_74563


namespace aladdin_can_find_heavy_coins_l74_74873

theorem aladdin_can_find_heavy_coins :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ x ≠ y ∧ (x + y ≥ 28) :=
by
  sorry

end aladdin_can_find_heavy_coins_l74_74873


namespace open_safe_in_6_attempts_min_attempts_to_open_safe_l74_74452

-- Define a function to represent the unique digits property
def unique_digits (l : List Nat) : Prop :=
  l.length = 7 ∧ l.nodup

-- Define a function to verify if the ith digit is in the correct position
def correct_position (l code : List Nat) : Prop :=
  ∃ i, i < 7 ∧ l.get? i = code.get? i

-- Part a: Proving that it is possible to open the safe in at most 6 attempts
theorem open_safe_in_6_attempts (code : List Nat) (h1 : code.length = 7) (h2 : code.nodup) :
  ∃ (attempts : List (List Nat)), attempts.length ≤ 6 ∧
  (∀ attempt ∈ attempts, unique_digits attempt) ∧
  (∃ attempt ∈ attempts, correct_position attempt code) := sorry

-- Part b: Proving that the minimum number of attempts required to open the safe is 6
theorem min_attempts_to_open_safe (code : List Nat) (h1 : code.length = 7) (h2 : code.nodup) :
  ∀ n, (∀ (attempts : List (List Nat)), attempts.length = n →
  (∀ attempt ∈ attempts, unique_digits attempt) →
  (∃ attempt ∈ attempts, correct_position attempt code)) → n ≥ 6 := sorry

end open_safe_in_6_attempts_min_attempts_to_open_safe_l74_74452


namespace nickys_running_pace_l74_74039

theorem nickys_running_pace (head_start : ℕ) (pace_cristina : ℕ) (time_nicky : ℕ) (distance_meet : ℕ) :
  head_start = 12 →
  pace_cristina = 5 →
  time_nicky = 30 →
  distance_meet = (pace_cristina * (time_nicky - head_start)) →
  (distance_meet / time_nicky = 3) :=
by
  intros h_start h_pace_c h_time_n d_meet
  sorry

end nickys_running_pace_l74_74039


namespace number_of_correct_propositions_l74_74295

section

variables (Line Plane : Type) [Geometry Line Plane]
variables (m n : Line) (α β : Plane)

-- Properties and conditions
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (intersection : Line → Plane → Line)

-- Propositions
def proposition_1 : Prop := ∀ (m n : Line) (α : Plane),
  perpendicular m n ∧ perpendicular m α → ∃ p : Plane, subset n p ∧ ¬ parallel n α

def proposition_2 : Prop := ∀ (m n : Line) (α β : Plane),
  perpendicular m α ∧ perpendicular n β ∧ parallel m n → parallel α β

def proposition_3 : Prop := ∀ (m n : Line) (α β : Plane),
  skew m n ∧ subset m α ∧ subset n β ∧ parallel m β ∧ parallel n α → parallel α β

def proposition_4 : Prop := ∀ (a : Line) (α β : Plane) (m : Line),
  perpendicular a β ∧ intersection a β = m ∧ subset n β ∧ perpendicular n m → perpendicular n α

-- Theorem statement: The number of correct propositions is 3
theorem number_of_correct_propositions : 
  (proposition_1 Line Plane perpendicular parallel intersection subset → false) ∧ 
  (proposition_2 Line Plane perpendicular parallel → true) ∧ 
  (proposition_3 Line Plane skew subset parallel → true) ∧ 
  (proposition_4 Line Plane perpendicular intersection subset → true) ∧ 
  (3 = 3) := sorry

end

end number_of_correct_propositions_l74_74295


namespace correct_equation_l74_74844

-- Conditions:
def number_of_branches (x : ℕ) := x
def number_of_small_branches (x : ℕ) := x * x
def total_number (x : ℕ) := 1 + number_of_branches x + number_of_small_branches x

-- Proof Problem:
theorem correct_equation (x : ℕ) : total_number x = 43 → x^2 + x + 1 = 43 :=
by 
  sorry

end correct_equation_l74_74844


namespace parabola_distance_l74_74961

theorem parabola_distance {x y : ℝ} (h₁ : y^2 = 4 * x) (h₂ : abs (x + 2) = 5) : 
  (sqrt ((x + 1)^2 + y^2) = 4) :=
sorry

end parabola_distance_l74_74961


namespace multiples_2_3_not_5_count_l74_74307

theorem multiples_2_3_not_5_count : 
  (finset.filter (λ n : ℕ, n ≤ 200 ∧ ((n % 2 = 0 ∨ n % 3 = 0) ∧ n % 5 ≠ 0)) (finset.range 201)).card = 107 :=
by
  sorry

end multiples_2_3_not_5_count_l74_74307


namespace cyclic_quadrilateral_angle_EBC_l74_74619

noncomputable def cyclic_quadrilateral (a b c d e : Type) [MetricSpace a]
  (ABCD : cylic_quadrilateral a b c d) extends (A : Point) (B : Point) (C : Point) (D : Point) (E : Point) :=
(bad : ∠A B D = 85)
(adc : ∠A D B = 70)
  
theorem cyclic_quadrilateral_angle_EBC
  (a b c d e : Type)
  [MetricSpace a]
  (ABCD : cyclic_quadrilateral a b c d)
  [AB_extends_to : extends A B E]
  (bad : ∠BAD = 85)
  (adc : ∠ADC = 70) :
  ∠EBC = 70 := 
sorry

end cyclic_quadrilateral_angle_EBC_l74_74619


namespace isosceles_triangle_external_tangent_angle_90_l74_74007

theorem isosceles_triangle_external_tangent_angle_90
  (A B C I F E P : Point)
  (h_iso : is_isosceles_triangle A B C)
  (h_ABAC : AB = AC)
  (h_incircle : incircle_center A B C I F E)
  (h_circumcircle : circumcircle_contains A F E)
  (h_ext_tangents : external_tangents P I A C)
  (h_parallel : tangent_parallel_to_side P AC) :
  angle P B I = π / 2 :=
sorry

end isosceles_triangle_external_tangent_angle_90_l74_74007


namespace parallelogram_and_area_l74_74450

noncomputable def A : ℝ × ℝ × ℝ := (2, -5, 3)
noncomputable def B : ℝ × ℝ × ℝ := (4, -9, 6)
noncomputable def C : ℝ × ℝ × ℝ := (3, -4, 1)
noncomputable def D : ℝ × ℝ × ℝ := (5, -8, 4)

def vec_sub (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)

noncomputable def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem parallelogram_and_area :
  vec_sub B A = vec_sub D C ∧
  norm (cross_product (vec_sub B A) (vec_sub C A)) = real.sqrt 110 :=
by
  sorry

end parallelogram_and_area_l74_74450


namespace margo_total_distance_l74_74392

-- Definitions related to the conditions
def time_jog_in_hours : ℝ := 12 / 60
def time_walk_in_hours : ℝ := 25 / 60
def total_time_in_hours : ℝ := time_jog_in_hours + time_walk_in_hours
def average_speed_in_mph : ℝ := 5

-- The proof that the total distance traveled is 3.085 miles
theorem margo_total_distance : 
  total_time_in_hours * average_speed_in_mph = 3.085 := 
  by 
    sorry

end margo_total_distance_l74_74392


namespace common_tangent_line_at_P_l74_74657

theorem common_tangent_line_at_P (a b c : ℝ) (f g : ℝ → ℝ)
  (h₁ : f = λ x, 2 * x^3 + a * x)
  (h₂ : g = λ x, b * x^2 + c)
  (h₃ : f 2 = 0)
  (h₄ : g 2 = 0)
  (h₅ : ∃ k, deriv f 2 = k ∧ deriv g 2 = k) :
  (f = λ x, 2 * x^3 - 8 * x) ∧ (g = λ x, 4 * x^2 - 16) ∧ (∀ x y, 16 * x - y - 32 = 0) :=
by
  sorry

end common_tangent_line_at_P_l74_74657


namespace cost_price_per_meter_l74_74820

theorem cost_price_per_meter (number_of_meters : ℕ) (selling_price : ℝ) (profit_per_meter : ℝ) (total_cost_price : ℝ) (cost_per_meter : ℝ) :
  number_of_meters = 85 →
  selling_price = 8925 →
  profit_per_meter = 15 →
  total_cost_price = selling_price - (profit_per_meter * number_of_meters) →
  cost_per_meter = total_cost_price / number_of_meters →
  cost_per_meter = 90 :=
by
  intros h1 h2 h3 h4 h5 
  sorry

end cost_price_per_meter_l74_74820


namespace gcd_same_remainder_mod_three_l74_74743

theorem gcd_same_remainder_mod_three (a b c d e f g : ℕ) (h_distinct : list.nodup [a, b, c, d, e, f, g]) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g) :
  ∃ x y z, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ (gcd x y % 3 = gcd y z % 3 ∧ gcd y z % 3 = gcd z x % 3) :=
by sorry

end gcd_same_remainder_mod_three_l74_74743


namespace white_dandelions_on_saturday_l74_74153

-- Represent the dandelion lifecycle and counts
def lifecycle := { bloom: ℕ, yellow: ℕ, white: ℕ, disperse: ℕ }

def monday_daytime := { yellow: ℕ := 20, white: ℕ := 14 }
def wednesday_daytime := { yellow: ℕ := 15, white: ℕ := 11 }

-- The number of white dandelions on saturday
theorem white_dandelions_on_saturday : ∀ l: lifecycle, l.bloom = 0 → l.yellow = 3 → l.white = 4 → l.disperse = 5 ∧
  ∀ m: lifespan, m.yellow = 20 ∧ m.white = 14 ∧ 
  ∀ w: lifespan, w.yellow = 15 ∧ w.white = 11 → 
  ∃ n: ℕ, n = 6 := 
begin
  sorry
end

end white_dandelions_on_saturday_l74_74153


namespace no_distinct_integers_cycle_l74_74414

theorem no_distinct_integers_cycle (p : Polynomial ℤ) (n : ℕ) (h_n : n ≥ 3) 
  (x : Fin n → ℤ) (h_distinct : Function.Injective x) 
  (h_cycle : ∀ i : Fin n, p.eval (x i) = x ((i+1) % n)) : False :=
by
  sorry

end no_distinct_integers_cycle_l74_74414


namespace domain_f_range_f_correct_g_l74_74284

-- Define the function f
def f (x : ℝ) : ℝ :=
  Real.sqrt (1 + x) + Real.sqrt (1 - x)

-- Prove the domain of f is [-1, 1]
theorem domain_f : ∀ x, (-1 : ℝ) ≤ x ∧ x ≤ 1 ↔ (1 + x ≥ 0) ∧ (1 - x ≥ 0) :=
by intro x; split; intro h;
   { split; linarith } <|> { linarith };

-- Prove the range of f is [sqrt(2), 2]
theorem range_f : ∀ y, (Real.sqrt 2 ≤ y ∧ y ≤ 2) ↔ ∃ x, (-1 : ℝ) ≤ x ∧ x ≤ 1 ∧ y = f x :=
sorry

-- Define the function F
def F (x m : ℝ) : ℝ :=
  m * Real.sqrt (1 - x^2) + f x

-- Prove the function g(m) is defined correctly
def g (m : ℝ) : ℝ :=
  if m ≤ -Real.sqrt 2 / 2 then Real.sqrt 2
  else if -Real.sqrt 2 / 2 < m ∧ m ≤ -1/2 then -1 / (2 * m) - m
  else m + 2

theorem correct_g :
  ∀ m : ℝ, g m = if m ≤ -Real.sqrt 2 / 2 then Real.sqrt 2
    else if -Real.sqrt 2 / 2 < m ∧ m ≤ -1/2 then -1 / (2 * m) - m
    else m + 2 :=
sorry

end domain_f_range_f_correct_g_l74_74284


namespace greatest_possible_value_l74_74899

theorem greatest_possible_value {t q α β : ℂ} 
  (h_eq : α + β = (α ^ 2 + β ^ 2) ∧ (α ^ 2 + β ^ 2) = (α ^ 3 + β ^ 3) ∧ ∀ n : ℕ, (1 ≤ n) → (α ^ n + β ^ n) = (α ^ (n + 1) + β ^ (n + 1))) 
  (h_roots : α * β = q ∧ α + β = t) :
  ∃ (α β : ℂ), α * β = q ∧ α + β = t ∧ (α = β) ∧ (α = 1 ∧ β = 1) → (1 / (α ^ 2011) + 1 / (β ^ 2011) = 2) := 
begin
  have h_αβ : α * β = q,
  sorry,
  have h_sums : ∀ n : ℕ, (1 ≤ n) → (α ^ n + β ^ n) = t,
  sorry,
  have h_values : (t = 2 ∧ q = 1) ∧ α = β ∧ (α = 1 ∧ β = 1),
  sorry,
  use [1, 1],
  sorry
end

end greatest_possible_value_l74_74899


namespace value_of_a_plus_b_l74_74974

theorem value_of_a_plus_b :
  ∀ (a b x y : ℝ), x = 3 → y = -2 → 
  a * x + b * y = 2 → b * x + a * y = -3 → 
  a + b = -1 := 
by
  intros a b x y hx hy h1 h2
  subst hx
  subst hy
  sorry

end value_of_a_plus_b_l74_74974


namespace minimize_square_distances_centroid_minimize_square_distances_ratios_l74_74588

variable {A B C : Type} [MetricSpace A] (triangle : Triangle A B C)

def centroid (triangle : Triangle A B C) : A := sorry -- Assume a definition for the centroid

def sum_square_distances_to_vertices (p : A) (triangle : Triangle A B C) : ℝ :=
  (dist p triangle.A)^2 + (dist p triangle.B)^2 + (dist p triangle.C)^2

def sum_square_perpendicular_distances_to_sides (p : A) (triangle : Triangle A B C) : ℝ :=
  let d₁ := sorry -- distance from p to side BC
  let d₂ := sorry -- distance from p to side CA
  let d₃ := sorry -- distance from p to side AB
  d₁^2 + d₂^2 + d₃^2

theorem minimize_square_distances_centroid :
  ∀ p : A, sum_square_distances_to_vertices p triangle ≥ sum_square_distances_to_vertices (centroid triangle) triangle := 
  sorry

theorem minimize_square_distances_ratios :
  ∃ p : A, (∀ q : A, sum_square_perpendicular_distances_to_sides q triangle ≥ sum_square_perpendicular_distances_to_sides p triangle) ∧
           (dist_ratio_maintained p triangle) := 
  sorry

-- Assuming a definition for dist_ratio_maintained
def dist_ratio_maintained (p : A) (triangle : Triangle A B C) : Prop := sorry

end minimize_square_distances_centroid_minimize_square_distances_ratios_l74_74588


namespace AB_distance_extremes_l74_74347

namespace CartesianCurves

def C1_parametric (t α : ℝ) : ℝ × ℝ :=
  (2 + t * cos α, sqrt 3 + t * sin α)

def C2_polar (ρ θ : ℝ) : Prop :=
  ρ = 8 * cos (θ - π / 3)

lemma C2_cartesian_eqn (x y : ℝ) : (x^2 + y^2 = 4 * x + 4 * sqrt 3 * y) :=
  sorry

theorem AB_distance_extremes (α : ℝ) (t1 t2 : ℝ) (h1: t1 + t2 = 2 * sqrt 3 * sin α) (h2: t1 * t2 = -13) :
  (sqrt(12 * sin α ^ 2 + 52) = max 2 (sqrt 13) 8) := by
  sorry

end CartesianCurves

end AB_distance_extremes_l74_74347


namespace largest_prime_factor_of_101101101101_l74_74443

theorem largest_prime_factor_of_101101101101 :
  ∃ p : ℕ, prime p ∧ p = 9901 ∧
    (101101101101 = 101 * 101 * 7 * 11 * 13 * p) :=
by
  sorry

end largest_prime_factor_of_101101101101_l74_74443


namespace problem_inequality_a3_a2_problem_inequality_relaxed_general_inequality_l74_74135

theorem problem_inequality_a3_a2 (a : ℝ) (ha : a > 1) : 
  a^3 + (1 / a^3) > a^2 + (1 / a^2) := 
sorry

theorem problem_inequality_relaxed (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : 
  a^3 + (1 / a^3) > a^2 + (1 / a^2) := 
sorry

theorem general_inequality (a : ℝ) (m n : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hmn1 : m > n) (hmn2 : n > 0) : 
  a^m + (1 / a^m) > a^n + (1 / a^n) := 
sorry

end problem_inequality_a3_a2_problem_inequality_relaxed_general_inequality_l74_74135


namespace general_solution_of_differential_eq_l74_74596

theorem general_solution_of_differential_eq :
  ∀ (y : ℝ → ℝ) (y' y'' : ℝ → ℝ) (C1 C2 : ℝ),
  (∀ x, y'' x - 12 * y' x + 36 * y x = Real.sin (3 * x)) →
  (∀ x, y' x = (y x)') →
  (∀ x, y'' x = (y' x)') →
  y = λ x, (C1 + C2 * x) * Real.exp (6 * x) + (4 / 225) * Real.cos (3 * x) + (1 / 75) * Real.sin (3 * x) :=
begin
  sorry
end

end general_solution_of_differential_eq_l74_74596


namespace exists_valid_triangle_l74_74902

-- Definition of the problem in Lean 4
structure GeometricData (Point : Type) :=
  (M A' A'' : Point)

-- Definition of a valid triangle construction given the data
def validTriangleConstruction {Point : Type} [MetricSpace Point] 
  (data : GeometricData Point) : Prop :=
  ∃ (A B C : Point),
    -- Orthocenter condition
    isOrthocenter A B C data.M ∧
    -- Point A' is the internal angle bisector's intersection with BC
    isAngleBisectorIntersection A B C data.A' ∧
    -- Point A'' is the external angle bisector's intersection with BC
    isExternalAngleBisectorIntersection A B C data.A''

-- Lean 4 statement for the problem
theorem exists_valid_triangle {Point : Type} [MetricSpace Point]
  (data : GeometricData Point) : validTriangleConstruction data :=
by
  sorry

end exists_valid_triangle_l74_74902


namespace baker_cakes_remaining_l74_74889

def InitialCakes : ℕ := 48
def SoldCakes : ℕ := 44
def RemainingCakes (initial sold : ℕ) : ℕ := initial - sold

theorem baker_cakes_remaining : RemainingCakes InitialCakes SoldCakes = 4 := 
by {
  -- placeholder for the proof
  sorry
}

end baker_cakes_remaining_l74_74889


namespace intersection_A_B_l74_74995

noncomputable def A : set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }
noncomputable def B : set ℝ := { x | |x| ≤ 2 }

theorem intersection_A_B : A ∩ B = { x | -2 ≤ x ∧ x ≤ -1 } :=
by
  sorry

end intersection_A_B_l74_74995


namespace value_of_a_l74_74047

theorem value_of_a (a : ℝ) (H1 : A = a) (H2 : B = 1) (H3 : C = a - 3) (H4 : C + B = 0) : a = 2 := by
  sorry

end value_of_a_l74_74047


namespace car_travel_mpg_l74_74843

def car_miles_per_gallon (H : ℝ) : Prop :=
  let highway_miles := 4
  let city_miles := 4
  let city_mpg := 20
  (highway_miles / H + city_miles / city_mpg) = (8 / H) * 1.35

theorem car_travel_mpg : ∃ H : ℝ, H = 34 ∧ car_miles_per_gallon H :=
  begin
    use 34,
    unfold car_miles_per_gallon,
    have hw_eq : (4 / 34 + 4 / 20) = (8 / 34) * 1.35,
    { -- this is just for illustrative purposes, the actual proof should go here
      sorry },
    exact ⟨hw_eq⟩,
  end

end car_travel_mpg_l74_74843


namespace find_missed_percentage_l74_74519

-- Define the given conditions
def principal : ℝ := 4200
def interest_rate_high : ℝ := 0.18 -- 18%
def time_period : ℝ := 2
def interest_difference : ℝ := 504

-- Define the statement to be proven
theorem find_missed_percentage (P : ℝ) :
  principal * interest_rate_high * time_period - principal * (P / 100) * time_period = interest_difference →
  P = 12 :=
by
  intro h
  rw [mul_sub, mul_assoc, mul_assoc, ←mul_sub, mul_div] at h
  have h' : 1512 - 84 * P / 100 = 504, from h,
  sorry

end find_missed_percentage_l74_74519


namespace weight_of_each_bag_of_flour_l74_74394

-- Definitions based on the given conditions
def cookies_eaten_by_Jim : ℕ := 15
def cookies_left : ℕ := 105
def total_cookies : ℕ := cookies_eaten_by_Jim + cookies_left

def cookies_per_dozen : ℕ := 12
def pounds_per_dozen : ℕ := 2

def dozens_of_cookies := total_cookies / cookies_per_dozen
def total_pounds_of_flour := dozens_of_cookies * pounds_per_dozen

def bags_of_flour : ℕ := 4

-- Question to be proved
theorem weight_of_each_bag_of_flour : total_pounds_of_flour / bags_of_flour = 5 := by
  sorry

end weight_of_each_bag_of_flour_l74_74394


namespace segment_AB_length_l74_74470

noncomputable def segment_length (R1 R2 α : ℝ) : ℝ :=
  (R1 + R2) * Real.cot (α / 2)

theorem segment_AB_length (R1 R2 α : ℝ) (hR1 : 0 < R1) (hR2 : 0 < R2) (hα : 0 < α ∧ α < 2 * Real.pi) :
  segment_length R1 R2 α = (R1 + R2) * Real.cot (α / 2) :=
by
  sorry

end segment_AB_length_l74_74470


namespace infinite_divisors_of_expression_l74_74050

theorem infinite_divisors_of_expression : ∀ (n : ℕ+), ∃ (p : ℕ) (prime p), p ∣ (2^(n^3 + 1) - 3^(n^2 + 1) + 5^(n + 1)) :=
by sorry

end infinite_divisors_of_expression_l74_74050


namespace polynomial_coefficients_correct_l74_74676

-- Define the polynomial equation
def polynomial_equation (x a b c d : ℝ) : Prop :=
  x^4 + 4*x^3 + 3*x^2 + 2*x + 1 = (x+1)^4 + a*(x+1)^3 + b*(x+1)^2 + c*(x+1) + d

-- The problem to prove
theorem polynomial_coefficients_correct :
  ∀ x : ℝ, polynomial_equation x 0 (-3) 4 (-1) :=
by
  intro x
  unfold polynomial_equation
  sorry

end polynomial_coefficients_correct_l74_74676


namespace distinct_right_triangles_l74_74304

theorem distinct_right_triangles (a b c : ℕ) (h : a^2 = 2016) :
  (∃ b c, b ∈ ℕ ∧ c ∈ ℕ ∧ c^2 = a^2 + b^2) → ∃ n, n = 12 :=
by {
  sorry
}

end distinct_right_triangles_l74_74304


namespace scrap_rate_independence_l74_74146

theorem scrap_rate_independence (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  (1 - (1 - a) * (1 - b)) = 1 - (1 - a) * (1 - b) :=
by
  sorry

end scrap_rate_independence_l74_74146


namespace median_of_sequence_l74_74211

theorem median_of_sequence (l : List ℕ) (h : l = [1, 1, 3, 3, 5]) : List.median l = some 3 :=
by
  rw h
  -- Add a placeholder to complete the proof structure
  sorry

end median_of_sequence_l74_74211


namespace ordering_of_PQR_l74_74260

variable (a b : ℝ) (ha : a > 0) (hb : b > 0)

def P : ℝ := sqrt ((a ^ 2 + b ^ 2) / 2) - (a + b) / 2
def Q : ℝ := (a + b) / 2 - sqrt (a * b)
def R : ℝ := sqrt (a * b) - (2 * a * b) / (a + b)

theorem ordering_of_PQR : Q a b ≥ P a b ∧ P a b ≥ R a b :=
by
  sorry

end ordering_of_PQR_l74_74260


namespace arccos_equation_solution_l74_74428

noncomputable def solve_arccos_equation : Prop :=
  ∃ x : ℝ, (arccos (3 * x) - arccos x = π / 6) ∧
           (x = sqrt (1 / (40 - 12 * sqrt 3)) ∨
            x = -sqrt (1 / (40 - 12 * sqrt 3)))

theorem arccos_equation_solution : solve_arccos_equation :=
by
  sorry

end arccos_equation_solution_l74_74428


namespace prime_factors_product_176543_l74_74199

theorem prime_factors_product_176543 : 
  let n := 176543
  let p1 := 2
  let p2 := 17
  prime p1 ∧ prime p2 ∧ p1 * p2 * 5192 = n ∧ 5192 % p1 ≠ 0 → p1 * p2 = 34 :=
by
  sorry

end prime_factors_product_176543_l74_74199


namespace dig_time_comparison_l74_74466

open Nat

theorem dig_time_comparison :
  (3 * 420 / 9) - (5 * 40 / 2) = 40 :=
by
  sorry

end dig_time_comparison_l74_74466


namespace pedal_triangle_inequality_l74_74372

variables {A B C D E F P Q R : Type*}

-- Assumed definitions for points and lines
variable {triABC : Triangle}
variable [IsAcuteAngleTriangle triABC]

-- Assume given conditions
def feet_perpendiculars_from_vertices 
  (triABC : Triangle) 
  (D E F P Q R : Point) 
  (hD : PerpendicularFromVertexToOppSide triABC A D BC)
  (hE : PerpendicularFromVertexToOppSide triABC B E CA)
  (hF : PerpendicularFromVertexToOppSide triABC C F AB)
  (hP : PerpendicularFromVertexToSide triABC A P EF)
  (hQ : PerpendicularFromVertexToSide triABC B Q FD)
  (hR : PerpendicularFromVertexToSide triABC C R DE) : Prop := sorry

theorem pedal_triangle_inequality 
  (triABC : Triangle) 
  (D E F P Q R : Point)
  [hD : PerpendicularFromVertexToOppSide triABC A D BC]
  [hE : PerpendicularFromVertexToOppSide triABC B E CA]
  [hF : PerpendicularFromVertexToOppSide triABC C F AB]
  [hP : PerpendicularFromVertexToSide triABC A P EF]
  [hQ : PerpendicularFromVertexToSide triABC B Q FD]
  [hR : PerpendicularFromVertexToSide triABC C R DE] :
  2 * (PQ + QR + RP) ≥ DE + EF + FD :=
sorry

end pedal_triangle_inequality_l74_74372


namespace distribution_schemes_count_l74_74583

theorem distribution_schemes_count :
  ∃ (S : Finset (α → β)) (M F : Finset (α → β)) (A B : Finset (α → β)), 
    (|S| = 5) ∧
    (|M| = 3) ∧
    (|F| = 2) ∧
    ((A ∪ B) = S) ∧ 
    (|A| ≥ 2) ∧
    (|B| ≥ 2) ∧
    (∃ (f ∈ F), f ∈ A) ∧
    (16 = (number_of_possible_distributions S M F A B))

noncomputable def number_of_possible_distributions (S M F A B : Finset (α → β)) : ℕ :=
  16 -- Example placeholder, typically should be the calculated value
  -- Note: Actual function implementation needed for counting

-- Example data setup to satisfy the theorem needed
sorry

end distribution_schemes_count_l74_74583


namespace Brenda_new_lead_l74_74570

noncomputable def Brenda_initial_lead : ℤ := 22
noncomputable def Brenda_play_points : ℤ := 15
noncomputable def David_play_points : ℤ := 32

theorem Brenda_new_lead : 
  Brenda_initial_lead + Brenda_play_points - David_play_points = 5 := 
by
  sorry

end Brenda_new_lead_l74_74570


namespace angle_CAB_plus_angle_COP_lt_90_l74_74714

-- Definition of the crucial elements in the problem
variables {A B C O P : Point}
variables {angle_BCA angle_ABC : ℝ}

-- Conditions given in the problem
def is_circumcenter (O : Point) (A B C : Point) : Prop := sorry
def is_foot_of_altitude (P A : Point) (BC : Line) : Prop := sorry
def is_acute_angled_triangle (A B C : Point) : Prop := sorry

-- Main theorem we need to prove in Lean
theorem angle_CAB_plus_angle_COP_lt_90
  (h1 : is_circumcenter O A B C) 
  (h2 : is_foot_of_altitude P A (Line B C)) 
  (h3 : is_acute_angled_triangle A B C) 
  (h4 : angle_BCA ≥ angle_ABC + 30) :
  ∠CAB + ∠COP < 90 := 
sorry

end angle_CAB_plus_angle_COP_lt_90_l74_74714


namespace total_students_l74_74796

theorem total_students (T : ℕ) (h1 : 0.10 * T = T - 45) : T = 50 :=
by 
sorry

end total_students_l74_74796
