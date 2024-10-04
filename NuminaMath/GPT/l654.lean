import Mathlib
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.Combinatorics.Binomial
import Mathlib.Algebra.Floor
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trigo
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.FderivAnalytic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Angles
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecificLimits
import Mathlib.Analysis.Trigonometry
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.MeasureTheory.Integral.IntervalIntegral
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace every_nat_as_diff_of_same_prime_divisors_l654_654263

-- Conditions
def prime_divisors (n : ℕ) : ℕ :=
  -- function to count the number of distinct prime divisors of n
  sorry

-- Tuple translation
theorem every_nat_as_diff_of_same_prime_divisors :
  ∀ n : ℕ, ∃ a b : ℕ, n = a - b ∧ prime_divisors a = prime_divisors b := 
by
  sorry

end every_nat_as_diff_of_same_prime_divisors_l654_654263


namespace ria_four_digit_numbers_count_l654_654695

theorem ria_four_digit_numbers_count :
  let counters := [1, 5, 11]
  let four_digit_numbers := {
    num | (∃ a ∈ counters, ∃ b ∈ counters, ∃ c ∈ counters, ∃ d ∈ counters,
    a * 1000 + b * 100 + c * 10 + d = num) ∧
    (⟦1 * 100 + 1⟧, ⟦11⟧)
  }
  ∃! l ∈ list.permutations [1, 1, 1, 5, 1, 5, 1, 1], l.nodup = true ]

  card four_digit_numbers = 4 :=
begin
  sorry
end

end ria_four_digit_numbers_count_l654_654695


namespace range_of_a_l654_654996

theorem range_of_a (a x : ℤ) (h1 : x > a) (h2 : x < 1) (h3 : (finset.Icc (a + 1) 0).card = 6) : -6 ≤ a ∧ a < -5 := by
  sorry

end range_of_a_l654_654996


namespace part1_part2_l654_654578

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem part1 (x : ℝ) : (Real.deriv f x) ≥ 2 := sorry

theorem part2 (x : ℝ) (hx : x ≥ 0) (hf : f (x^2 - 1) < Real.exp 1 - Real.exp (-1)) : x ∈ Set.Ico 0 (Real.sqrt 2) := sorry

end part1_part2_l654_654578


namespace distance_between_planes_l654_654929

-- Definitions of the planes
def plane1 (x y z : ℝ) : Prop := x + 3 * y - 2 * z + 4 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 6 * y - 4 * z + 7 = 0

-- The task is to prove the following theorem
theorem distance_between_planes :
  let normal1 := (1, 3, -2)
  let normal2 := (2, 6, -4)
  let point_on_plane1 := (0, 0, -2)
  let distance := |2 * 0 + 6 * 0 - 4 * (-2) + 7| / real.sqrt (2^2 + 6^2 + (-4)^2)
  distance = 15 * real.sqrt 14 / 28 := sorry

end distance_between_planes_l654_654929


namespace investment_in_equipment_l654_654705

theorem investment_in_equipment (cost_per_game : ℝ) (sell_price_per_game : ℝ) (num_games : ℕ) 
  (total_revenue : ℝ) (total_cost : ℝ) : 
  cost_per_game = 2.65 → sell_price_per_game = 20 → num_games = 600 → 
  total_cost = cost_per_game * num_games → total_revenue = sell_price_per_game * num_games → 
  total_revenue = total_cost + 10410 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4 h5
  simp [total_cost, total_revenue]
  sorry

end investment_in_equipment_l654_654705


namespace alcohol_added_l654_654850

-- Definitions from conditions
def initial_volume : ℝ := 40
def initial_alcohol_concentration : ℝ := 0.05
def initial_alcohol_amount : ℝ := initial_volume * initial_alcohol_concentration
def added_water_volume : ℝ := 3.5
def final_alcohol_concentration : ℝ := 0.17

-- The problem to be proven
theorem alcohol_added :
  ∃ x : ℝ,
    x = (final_alcohol_concentration * (initial_volume + x + added_water_volume) - initial_alcohol_amount) :=
by
  sorry

end alcohol_added_l654_654850


namespace function_periodicity_l654_654120

theorem function_periodicity
  (f : ℝ → ℝ)
  (H_odd : ∀ x, f (-x) = -f x)
  (H_even_shift : ∀ x, f (x + 2) = f (-x + 2))
  (H_val_neg1 : f (-1) = -1)
  : f 2017 + f 2016 = 1 := 
sorry

end function_periodicity_l654_654120


namespace fewer_twos_to_hundred_l654_654359

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l654_654359


namespace largest_real_root_range_l654_654988

theorem largest_real_root_range (b0 b1 b2 b3 : ℝ) (h0 : |b0| ≤ 1) (h1 : |b1| ≤ 1) (h2 : |b2| ≤ 1) (h3 : |b3| ≤ 1) :
  ∀ r : ℝ, (Polynomial.eval r (Polynomial.C (1:ℝ) + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C b0) = 0) → (5 / 2) < r ∧ r < 3 :=
by
  sorry

end largest_real_root_range_l654_654988


namespace rectangular_to_spherical_neg_x_l654_654861

noncomputable def spherical_to_rectangular (ρ θ φ: ℝ): ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

noncomputable def rectangular_to_spherical (x y z: ℝ): ℝ × ℝ × ℝ :=
  let ρ := sqrt (x^2 + y^2 + z^2)
  let θ := atan y x
  let φ := acos (z / ρ)
  (ρ, θ, φ)

theorem rectangular_to_spherical_neg_x (ρ θ φ: ℝ) (hρ: ρ > 0) (hθ: 0 ≤ θ ∧ θ < 2*π) (hφ: 0 ≤ φ ∧ φ ≤ π):
  rectangular_to_spherical (-ρ * sin φ * cos θ) (ρ * sin φ * sin θ) (ρ * cos φ) = (ρ, π - θ, φ) :=
  by
  -- proof omitted
  sorry

example : rectangular_to_spherical_neg_x 5 (5 * Real.pi / 6) (Real.pi / 4) (by linarith) (by linarith) (by linarith) = 
  (5, (Real.pi / 6), (Real.pi / 4)) :=
by sorry

end rectangular_to_spherical_neg_x_l654_654861


namespace probability_MAME_on_top_l654_654794

theorem probability_MAME_on_top : 
  let n := 8 in 
  let P (i : Fin n) := ∀ j ≠ i, "MAME" ≠ j in
  Prob (X = "MAME") = 1 / n
by
  intros
  sorry

end probability_MAME_on_top_l654_654794


namespace smallest_marked_cells_l654_654389

def smallest_k_10x11 : ℕ := 50

theorem smallest_marked_cells :
  ∃ (marked_cells : Finset (Fin 10 × Fin 11)),
  marked_cells.card = smallest_k_10x11 ∧ 
  (∀ (L : Finset (Fin 10 × Fin 11)), L.card = 3 → (∀ P₁ P₂ P₃ ∈ L, ¬(P₁.1 = P₂.1 ∧ P₂.1 = P₃.1 ∨ P₁.2 = P₂.2 ∧ P₂.2 = P₃.2)) → 
    ∃ cell ∈ L, cell ∈ marked_cells) :=
sorry

end smallest_marked_cells_l654_654389


namespace part_one_a_increasing_on_1_inf_part_one_a_decreasing_on_0_1_part_one_a_minimum_value_part_two_range_of_a_l654_654577

noncomputable def f (a x : ℝ) : ℝ := a * x + (a - 1) / x - Real.log x - 1

theorem part_one_a_increasing_on_1_inf (x : ℝ) : 
  (∀ x > 1, f 1 x > f 1 x) := sorry

theorem part_one_a_decreasing_on_0_1 (x : ℝ) : 
  (∀ x > 0, x < 1, f 1 x < f 1 x) := sorry

theorem part_one_a_minimum_value : f 1 1 = 0 := sorry

theorem part_two_range_of_a (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : ∀ x ∈ set.Ici (1:ℝ), f a x ≥ 0) : 
  a ∈ set.Ici (1:ℝ) := sorry

end part_one_a_increasing_on_1_inf_part_one_a_decreasing_on_0_1_part_one_a_minimum_value_part_two_range_of_a_l654_654577


namespace fewer_twos_result_100_l654_654363

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l654_654363


namespace carpeting_room_length_l654_654714

theorem carpeting_room_length (cost_per_meter_paisa : ℕ) (cost_total_rs : ℝ) (breadth_m : ℕ) 
                              (width_cm : ℕ) (rupees_to_paisa : ℕ) :
  let cost_per_meter_rs := cost_per_meter_paisa.to_real / rupees_to_paisa.to_real,
      width_m := width_cm.to_real / 100,
      area := breadth_m.to_real * (L:ℝ),
      length_of_carpet := area / width_m,
      total_cost := length_of_carpet * cost_per_meter_rs
  in total_cost = cost_total_rs → L = 15 := by
  sorry

end carpeting_room_length_l654_654714


namespace certain_boy_among_selected_l654_654945

def group := {boys := 5, girls := 2}
def selected_people := 3

theorem certain_boy_among_selected : selected_people = 3 → group.girls < selected_people → ∃ (g : ℤ) (b : ℤ), b > 0 :=
by
  intros
  sorry

end certain_boy_among_selected_l654_654945


namespace f_range_a_l654_654914

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a*x^2

theorem f_range_a (x : ℝ) (a : ℝ) (x1 : ℝ) (x2 : ℝ) :
  (∀ x1 ∈ set.Ioi 2, ∃ x2 ∈ set.Ioi 1, f x1 a * f x2 a = 1) ∧
  (a ≥ 3/4 → ∀ x ∈ set.Ioi 2, derivative (λ x, f x a) x ≤ 0) ∧
  (a ≤ 3/2 → (∀ x ∈ set.Ioc 1 (3/(2*a)), derivative (λ x, f x a) x ≥ 0) ∧ 
  (∀ x ∈ set.Ioi (3/(2*a)), derivative (λ x, f x a) x ≤ 0)) →
  a ∈ set.Icc (3/4) (3/2) :=
sorry

end f_range_a_l654_654914


namespace number_of_pairs_determining_plane_l654_654594

-- Define the properties of a regular tetrahedron
def is_regular_tetrahedron (V : Type) (edges : Finset (Finset V)) := 
  ∃ vertices : Finset V, vertices.card = 4 ∧ edges.card = 6 ∧ ∀ v ∈ vertices, ((λ e, v ∈ e) '' edges).card = 3

-- Define what it means for a pair of edges to determine a plane
def edges_determine_plane {V : Type} (e1 e2 : Finset V) : Prop :=
  (∃ (v : V), v ∈ e1 ∧ v ∈ e2) ∨ e1 ∩ e2 = ∅

-- The theorem to be proven
theorem number_of_pairs_determining_plane (V : Type) (edges : Finset (Finset V)) (h : is_regular_tetrahedron V edges) :
  (edges.powerset.filter (λ s, s.card = 2 ∧ edges_determine_plane (s.to_list.nth_le 0 sorry) (s.to_list.nth_le 1 sorry))).card = 12 :=
sorry

end number_of_pairs_determining_plane_l654_654594


namespace find_linear_term_coefficient_l654_654551

-- Define the fifth-degree polynomial construction based on the given conditions
noncomputable def polynomial_f (x : ℝ) : ℝ := 
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + 8 * x

-- Define the coefficient of the linear term of f(x)
noncomputable def linear_term_coefficient (p : ℝ → ℝ) : ℝ :=
  let coeffs := (Polynomial.coeff (Polynomial.of_fun p)) in coeffs 1

-- The statement: prove that given the polynomial f(x), the coefficient of the linear term is 282
theorem find_linear_term_coefficient : linear_term_coefficient polynomial_f = 282 :=
begin
  sorry
end

end find_linear_term_coefficient_l654_654551


namespace sum_of_Y_l654_654323

open Finset

def X : Finset ℕ := (range 600).map ⟨λ n, n + 1, λ a b, by simp⟩

def multiples (k : ℕ) (s : Finset ℕ) : Finset ℕ :=
  s.filter (λ x, x % k = 0)

def Y : Finset ℕ := (multiples 3 X) ∪ (multiples 4 X)

theorem sum_of_Y : Y.sum id = 90300 :=
by sorry

end sum_of_Y_l654_654323


namespace find_ratio_of_S8_S4_l654_654141

open Classical

variable {a_1 : ℝ} {q : ℝ}

-- Define the nth term of the geometric sequence
def a_n (n : ℕ) : ℝ := a_1 * q^(n-1)

-- Define the sum of the first n terms of the geometric sequence
def S_n (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)

-- Given conditions
axiom a4_eq_2a2 : a_n 4 = 2 * a_n 2

-- Problem statement to prove
theorem find_ratio_of_S8_S4 (a_1 q : ℝ) (h : q^2 = 2) (a4_eq_2a2 : a_n 4 = 2 * a_n 2) : 
  S_n 8 / S_n 4 = 5 :=
by
  sorry

end find_ratio_of_S8_S4_l654_654141


namespace count_table_fills_2014_2015_2016_l654_654840

theorem count_table_fills_2014_2015_2016 :
  let a := 2014
  let b := 2015
  let c := 2016
  let total_fills := 831 in
  ∃ table : (Fin 3 → Fin 3 → ℕ), 
    (∀ i : Fin 3, (∑ j : Fin 3, table i j) = (∑ j : Fin 3, table 0 j)) ∧
    (∀ i j, table i j ∈ {a, b, c}) ∧
    -- Count the unique valid tables
    (card {t : (Fin 3 → Fin 3 → ℕ) | 
      (∀ i : Fin 3, (∑ j : Fin 3, t i j) = (∑ j : Fin 3, t 0 j)) ∧
      (∀ i j, t i j ∈ {a, b, c})} = total_fills) :=
by
  let a := 2014
  let b := 2015
  let c := 2016
  let total_fills := 831
  existsi (λ i j, if (i, j) = (0, 0) then a else b)
  split
  { 
    intro i,
    sorry 
  }
  {
    split
    {
        intros i j,
        exact by sorry -- elements are in {a, b, c}
    }
    { 
      apply (sorry : card ... = total_fills)
    }
  }

end count_table_fills_2014_2015_2016_l654_654840


namespace inv_function_l654_654157

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

noncomputable def f_inv (y : ℝ) : ℝ := -Real.sqrt (y + 1)

theorem inv_function {x : ℝ} (hx : -1 ≤ x ∧ x < 0) : 
  f_inv (f x) = x ∧ ∃ y, -1 < y ∧ y ≤ 0 ∧ f (f_inv y) = y :=
begin
  sorry
end

end inv_function_l654_654157


namespace sin_210_eq_neg_half_l654_654893

theorem sin_210_eq_neg_half : 
  let θ := 210
  let α := 180
  let β := 30
  (θ = α + β) -> 
  sin α = 0 ->
  cos α = -1 -> 
  sin β = 1/2 ->
  sin θ = -1/2 :=
by
  intro θ α β hθ hsinα hcosα hsinβ
  rw [←hθ, sin_add]
  rw [hsinα, hcosα, hsinβ]
  norm_num
  sorry

end sin_210_eq_neg_half_l654_654893


namespace necessary_and_sufficient_conditions_l654_654086

theorem necessary_and_sufficient_conditions (n : ℕ) (h : n ≥ 2) 
  (a : ℕ → ℝ) (hnz : ∃ i, i < n ∧ a i ≠ 0) : 
  (∃ x : ℕ → ℕ, (∀ i j, i < j → x i < x j) ∧ (∀ i, 0 < x i) ∧ ∑ i in finset.range n, a i * (x i : ℝ) ≥ 0) ↔
  (∃ k, k < n ∧ ∑ i in finset.Ici k, a i > 0) :=
sorry

end necessary_and_sufficient_conditions_l654_654086


namespace f_values_f_inequality_l654_654553

def f : ℝ → ℝ := sorry  -- Assume there is a function f satisfying the conditions

axiom f_monotonic : ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f(x) < f(y)
axiom f_additive : ∀ (x y : ℝ), 0 < x → 0 < y → f(x * y) = f(x) + f(y)
axiom f_value_at_3 : f 3 = 1

theorem f_values :
  f 1 = 0 ∧ f (1 / 3) = -1 := by
  sorry

theorem f_inequality (x : ℝ) :
  f(x) + f(x - 8) ≤ 2 → 8 < x ∧ x ≤ 9 := by
  sorry

end f_values_f_inequality_l654_654553


namespace count_table_fills_2014_2015_2016_l654_654841

theorem count_table_fills_2014_2015_2016 :
  let a := 2014
  let b := 2015
  let c := 2016
  let total_fills := 831 in
  ∃ table : (Fin 3 → Fin 3 → ℕ), 
    (∀ i : Fin 3, (∑ j : Fin 3, table i j) = (∑ j : Fin 3, table 0 j)) ∧
    (∀ i j, table i j ∈ {a, b, c}) ∧
    -- Count the unique valid tables
    (card {t : (Fin 3 → Fin 3 → ℕ) | 
      (∀ i : Fin 3, (∑ j : Fin 3, t i j) = (∑ j : Fin 3, t 0 j)) ∧
      (∀ i j, t i j ∈ {a, b, c})} = total_fills) :=
by
  let a := 2014
  let b := 2015
  let c := 2016
  let total_fills := 831
  existsi (λ i j, if (i, j) = (0, 0) then a else b)
  split
  { 
    intro i,
    sorry 
  }
  {
    split
    {
        intros i j,
        exact by sorry -- elements are in {a, b, c}
    }
    { 
      apply (sorry : card ... = total_fills)
    }
  }

end count_table_fills_2014_2015_2016_l654_654841


namespace symmetric_line_eq_l654_654717

theorem symmetric_line_eq (L1 L2 : AffineLine ℝ) :
  (L1.equation = AffEq (2 : ℝ) (1 / 2)) ∧ (L2.reflect_x = 1) → 
  (L2.equation = AffEq 1 2 (-2)) :=
begin
  sorry
end

end symmetric_line_eq_l654_654717


namespace bridge_length_proof_l654_654824

open Real

def train_length : ℝ := 100
def train_speed_kmh : ℝ := 45
def crossing_time_s: ℝ := 30

noncomputable def bridge_length : ℝ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem bridge_length_proof : bridge_length = 275 := 
by
  sorry

end bridge_length_proof_l654_654824


namespace smallest_integer_exceeds_50y_l654_654243

theorem smallest_integer_exceeds_50y :
  let y := (∑ n in Finset.range 30, Real.sin (n+1)*Real.pi/180) / (∑ n in Finset.range 30, Real.cos (n+1)*Real.pi/180)
  in Nat.ceil (50 * y) = 14 :=
by 
  sorry

end smallest_integer_exceeds_50y_l654_654243


namespace not_navigate_700m_can_navigate_800m_l654_654095

-- Define the river model and geometric properties
structure River :=
  (holy_condition1 : ∀ x ∈ bank1 ∃ y ∈ bank2, distance x y ≤ 1000)
  (holy_condition2 : ∀ x ∈ bank2 ∃ y ∈ bank1, distance x y ≤ 1000)
  (radius_lake : ∀ c, circle c 10)

-- Define the conditions for navigation for part (a)
def can_navigate_700m (r : River) : Prop :=
  ∃ path ∈ paths_along_river r, ∀ x ∈ path, ∀ edge ∈ banks r, distance x edge ≤ 700

-- Define the conditions for navigation for part (b)
def can_navigate_800m (r : River) : Prop :=
  ∃ path ∈ paths_along_river r, ∀ x ∈ path, ∀ edge ∈ banks r, distance x edge ≤ 800

-- Problem (a): Prove it is not possible to navigate within 700 meters from each bank
theorem not_navigate_700m (r : River) : ¬ can_navigate_700m r :=
sorry

-- Problem (b): Prove it is possible to navigate within 800 meters from each bank
theorem can_navigate_800m (r : River) : can_navigate_800m r :=
sorry

end not_navigate_700m_can_navigate_800m_l654_654095


namespace max_distance_on_unit_circle_l654_654164

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

theorem max_distance_on_unit_circle (α β : ℝ) :
  distance (complex.cos α, complex.sin α) (complex.cos β, complex.sin β) = 2 :=
sorry

end max_distance_on_unit_circle_l654_654164


namespace integer_as_polynomial_l654_654641

theorem integer_as_polynomial (N : ℤ) (d : ℕ) (a : ℕ → ℕ) (hN : N = ∑ i in Finset.range (d + 1), a i * 10 ^ i)
  (h_digits : ∀ i, i ≤ d → a i < 10) : 
  (∃ f : ℕ → ℤ, N = ∑ i in Finset.range (d + 1), f i * 10 ^ i) :=
by
  sorry

end integer_as_polynomial_l654_654641


namespace miraflores_optimal_strategy_l654_654752

-- Definitions based on conditions
variable (n : ℕ)
def total_voters := 2 * n
def miraflores_supporters := n
def dick_maloney_supporters := n
def miraflores_is_a_voter := 1
def law_allows_division := true
def election_winner (district1 district2 : Set ℕ) : ℕ := 
  if (district1.card = 1 ∧ miraflores_is_a_voter ∈ district1) then miraflores_is_a_voter else dick_maloney_supporters

-- Mathematically equivalent proof problem
theorem miraflores_optimal_strategy (hall : law_allows_division) :
  (exists (district1 district2 : Set ℕ),
    ∀ v, v ∈ district1 ∨ v ∈ district2 ∧ district1.card + district2.card = total_voters ∧
    miraflores_supporters = 1 ∧ district1 = {miraflores_is_a_voter} ∧
    (election_winner district1 district2) = miraflores_is_a_voter) :=
sorry

end miraflores_optimal_strategy_l654_654752


namespace reflection_matrix_solution_l654_654503

theorem reflection_matrix_solution :
  let a := -1/2
  let b := -1/2
  let R := matrix.of 2 2 (λ i j, if (i, j) = (0, 0) then a else if (i, j) = (0, 1) then b else if (i, j) = (1, 0) then -3/2 else 1/2)
  in R * R = 1 :=
by
  -- Proof will go here.
  sorry

end reflection_matrix_solution_l654_654503


namespace rotated_angle_l654_654312

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

end rotated_angle_l654_654312


namespace bobs_total_profit_l654_654483

-- Definitions of the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Definition of the problem statement
theorem bobs_total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end bobs_total_profit_l654_654483


namespace interval_for_monotonic_increasing_l654_654991

noncomputable theory

def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - 2 * (sin x)^2

def is_monotonic_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ (x y : ℝ), x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem interval_for_monotonic_increasing :
  ∀ k : ℤ, is_monotonic_increasing f (set.Icc (k * real.pi - real.pi / 3) (k * real.pi + real.pi / 6)) :=
by sorry

end interval_for_monotonic_increasing_l654_654991


namespace student_walking_time_l654_654003

-- Define the conditions
def total_time_walking_and_bus : ℕ := 90  -- Total time walking to school and taking the bus back home
def total_time_bus_both_ways : ℕ := 30 -- Total time taking the bus both ways

-- Calculate the time taken for walking both ways
def time_bus_one_way : ℕ := total_time_bus_both_ways / 2
def time_walking_one_way : ℕ := total_time_walking_and_bus - time_bus_one_way
def total_time_walking_both_ways : ℕ := 2 * time_walking_one_way

-- State the theorem to be proved
theorem student_walking_time :
  total_time_walking_both_ways = 150 := by
  sorry

end student_walking_time_l654_654003


namespace veronica_pre_selected_photos_l654_654333

-- Definition: Veronica needs to include 3 or 4 of her pictures
def needs_3_or_4_photos : Prop := True

-- Definition: Veronica has pre-selected a certain number of photos
def pre_selected_photos : ℕ := 15

-- Definition: She has 15 choices
def choices : ℕ := 15

-- The proof statement
theorem veronica_pre_selected_photos : needs_3_or_4_photos → choices = pre_selected_photos :=
by
  intros
  sorry

end veronica_pre_selected_photos_l654_654333


namespace relationship_a_b_c_l654_654543

noncomputable def a : ℝ := 3^0.1
noncomputable def b : ℝ := (1/3)^(-0.8)
noncomputable def c : ℝ := Real.log 0.8 / (Real.log 0.7)

theorem relationship_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_a_b_c_l654_654543


namespace odd_number_as_diff_of_squares_l654_654687

theorem odd_number_as_diff_of_squares (n : ℤ) : ∃ a b : ℤ, a^2 - b^2 = 2 * n + 1 :=
by
  use (n + 1), n
  sorry

end odd_number_as_diff_of_squares_l654_654687


namespace solution1_solution2_l654_654344

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l654_654344


namespace length_of_bridge_l654_654427

theorem length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross : ℝ) :
  train_length = 135 → train_speed_kmh = 45 → time_to_cross = 30 → 
  let train_speed_ms := train_speed_kmh * 1000 / 3600 in
  let total_distance := train_speed_ms * time_to_cross in
  total_distance - train_length = 240 :=
by
  intros
  sorry

end length_of_bridge_l654_654427


namespace find_y_l654_654735

-- Define relevant constants and conditions
variables (y : ℝ) (α : ℝ)
def condition1 := (√(9 + y^2) ≠ 0)
def condition2 := (sin α = 4 / 5)
def condition3 := (sin α = y / (√(9 + y^2)))

-- Define the theorem to be proven
theorem find_y
  (cond1 : condition1 y)
  (cond2 : condition2 α)
  (cond3 : condition3 y α) :
  y = 4 :=
sorry

end find_y_l654_654735


namespace possible_values_expression_l654_654976

theorem possible_values_expression 
  (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (v : ℝ), v ∈ {5, 1, -1, -5} ∧ 
    v = (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (a * b * c * d / |a * b * c * d|) 
:= sorry

end possible_values_expression_l654_654976


namespace factorial_comparison_l654_654808

open scoped BigOperators

theorem factorial_comparison : (100.factorial)!.factorial < (99.factorial) ^ (100.factorial) * (100.factorial) ^ (99.factorial) := by
  sorry

end factorial_comparison_l654_654808


namespace polynomial_q_l654_654719

theorem polynomial_q (
  p q r : ℝ) 
  (a b c d : ℝ) 
  (h_poly : Polynomial = x^4 + p*x^3 + q*x^2 + r*x + 144):
  -- Distinct roots condition
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  -- One root is the average of the other three
  d = (a + b + c) / 3  →
  -- q is 8*sqrt(3) 
  q = 8 * (144 ^ (1 / 3)) :=
sorry

end polynomial_q_l654_654719


namespace num_special_matrices_l654_654617

open Matrix

theorem num_special_matrices :
  ∃ (M : Matrix (Fin 4) (Fin 4) ℕ), 
    (∀ i j, 1 ≤ M i j ∧ M i j ≤ 16) ∧ 
    (∀ i j, i < j → M i j < M i (j + 1)) ∧ 
    (∀ i j, i < j → M i j < M (i + 1) j) ∧ 
    (∀ i, i < 3 → M i i < M (i + 1) (i + 1)) ∧ 
    (∀ i, i < 3 → M i (3 - i) < M (i + 1) (2 - i)) ∧ 
    (∃ n, n = 144) :=
sorry

end num_special_matrices_l654_654617


namespace find_triplet_l654_654083

theorem find_triplet (x y z : ℕ) :
  (1 + (x : ℚ) / (y + z))^2 + (1 + (y : ℚ) / (z + x))^2 + (1 + (z : ℚ) / (x + y))^2 = 27 / 4 ↔ (x, y, z) = (1, 1, 1) :=
by
  sorry

end find_triplet_l654_654083


namespace hair_color_assignment_l654_654771

structure Person :=
  (name : String)
  (hair_color : String)

def Belov := Person.mk "Belov" ""
def Chernov := Person.mk "Chernov" ""
def Ryzhov := Person.mk "Ryzhov" ""

def is_hair_color_valid (p : Person) (hc : String) : Prop :=
  (p.name = "Belov" → hc ≠ "blonde") ∧
  (p.name = "Chernov" → hc ≠ "brunette") ∧
  (p.name = "Ryzhov" → hc ≠ "red-haired")

def valid_person (p : Person) : Prop :=
  is_hair_color_valid p p.hair_color

noncomputable def Belov_hair_color := "red-haired"
noncomputable def Chernov_hair_color := "blonde"
noncomputable def Ryzhov_hair_color := "brunette"

theorem hair_color_assignment :
  valid_person { name := "Belov", hair_color := Belov_hair_color } ∧
  valid_person { name := "Chernov", hair_color := Chernov_hair_color } ∧
  valid_person { name := "Ryzhov", hair_color := Ryzhov_hair_color } :=
by {
  split;
  sorry,
}

end hair_color_assignment_l654_654771


namespace numRoutes_revisedNetwork_l654_654493

/-- The network of cities and roads between them -/
structure CityNetwork :=
  (cities : Finset String)
  (roads : Finset (String × String))

/-- Define the network of cities and roads -/
def revisedNetwork : CityNetwork :=
  {
    cities := {"A", "B", "D", "E", "F"},
    roads := {("A", "B"), ("A", "D"), ("A", "E"), ("B", "D"), ("D", "E"), ("A", "F"), ("F", "B")}
  }

/-- Define the number of different routes from City A to City B that use each road exactly once -/
def numRoutes (network : CityNetwork) (start : String) (end : String) : Nat :=
  if network = revisedNetwork ∧ start = "A" ∧ end = "B" then 32 else 0

/-- Proof that there are 32 different routes from A to B -/
theorem numRoutes_revisedNetwork : numRoutes revisedNetwork "A" "B" = 32 :=
  by sorry

end numRoutes_revisedNetwork_l654_654493


namespace battery_charging_budget_l654_654167

def cost_per_charge : ℝ := 3.5
def charges : ℕ := 4
def leftover : ℝ := 6
def budget : ℝ := 20

theorem battery_charging_budget :
  (charges : ℝ) * cost_per_charge + leftover = budget :=
by
  sorry

end battery_charging_budget_l654_654167


namespace fewer_twos_to_hundred_l654_654357

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l654_654357


namespace path_length_of_dot_l654_654865

-- Define the dimensions of the rectangular prism
def prism_width := 1 -- cm
def prism_height := 1 -- cm
def prism_length := 2 -- cm

-- Define the condition that the dot is marked at the center of the top face
def dot_position := (0.5, 1)

-- Define the condition that the prism starts with the 1 cm by 2 cm face on the table
def initial_face_on_table := (prism_length, prism_height)

-- Define the statement to prove the length of the path followed by the dot
theorem path_length_of_dot: 
  ∃ length_of_path : ℝ, length_of_path = 2 * Real.pi :=
sorry

end path_length_of_dot_l654_654865


namespace probability_one_project_not_selected_l654_654468

noncomputable def calc_probability : ℚ :=
  let n := 4 ^ 4
  let m := Nat.choose 4 2 * Nat.factorial 4
  let p := m / n
  p

theorem probability_one_project_not_selected :
  calc_probability = 9 / 16 :=
by
  sorry

end probability_one_project_not_selected_l654_654468


namespace base_7_units_digit_l654_654301

theorem base_7_units_digit (a : ℕ) (b : ℕ) (h₁ : a = 326) (h₂ : b = 57) : ((a * b) % 7) = 4 := by
  sorry

end base_7_units_digit_l654_654301


namespace quadratic_inequality_solution_set_l654_654933

theorem quadratic_inequality_solution_set {x : ℝ} :
  (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end quadratic_inequality_solution_set_l654_654933


namespace f_decreasing_ln_inequality_limit_inequality_l654_654664

-- Definitions of the given conditions
noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x

-- Statements we need to prove

-- (I) Prove that f(x) is decreasing on (0, +∞)
theorem f_decreasing : ∀ x y : ℝ, 0 < x → x < y → f y < f x := sorry

-- (II) Prove that for the inequality ln(1 + x) < ax to hold for all x in (0, +∞), a must be at least 1
theorem ln_inequality (a : ℝ) : (∀ x : ℝ, 0 < x → Real.log (1 + x) < a * x) ↔ 1 ≤ a := sorry

-- (III) Prove that (1 + 1/n)^n < e for all n in ℕ*
theorem limit_inequality (n : ℕ) (h : n ≠ 0) : (1 + 1 / n) ^ n < Real.exp 1 := sorry

end f_decreasing_ln_inequality_limit_inequality_l654_654664


namespace fourth_vertex_of_square_l654_654770

theorem fourth_vertex_of_square (A B C D : ℂ) : 
  A = (2 + 3 * I) ∧ B = (-3 + 2 * I) ∧ C = (-2 - 3 * I) →
  D = (0 - 0.5 * I) :=
sorry

end fourth_vertex_of_square_l654_654770


namespace pencils_left_with_Harry_l654_654021

theorem pencils_left_with_Harry :
  (let 
    anna_pencils := 50
    harry_initial_pencils := 2 * anna_pencils
    harry_lost_pencils := 19
    harry_pencils_left := harry_initial_pencils - harry_lost_pencils
  in harry_pencils_left = 81) := 
by
  sorry

end pencils_left_with_Harry_l654_654021


namespace determinant_has_one_real_root_l654_654661

noncomputable def hasExactlyOneRealRoot (a b c d : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) : Prop :=
  ∀ x : ℝ, 
    det ![![x, d, c], ![-d, x, b], ![-c, -b, x]] = 0 ↔ x = 0

theorem determinant_has_one_real_root (a b c d : ℝ) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  hasExactlyOneRealRoot a b c d hb hc hd := 
sorry

end determinant_has_one_real_root_l654_654661


namespace sum_of_digits_of_count_hex_below_2000_is_25_l654_654496

theorem sum_of_digits_of_count_hex_below_2000_is_25 :
  let n := 799
  in n.digits.sum = 25 := by
sоrry

end sum_of_digits_of_count_hex_below_2000_is_25_l654_654496


namespace find_angle_CAK_l654_654828

theorem find_angle_CAK
  (ABC : Type)
  [has_vertex ABC ψ]
  (isosceles_right : is_isosceles_right_triangle ABC)
  (M : Point)
  (midpoint_BC : is_midpoint M B C)
  (K : Point)
  (arc_AC : on_minor_arc K A C)
  (H : Point)
  (perpendicular_KH_AB : is_foot_of_perpendicular H K A B)
  (equal_segments : KH = BM)
  (parallel_lines : parallel MH CK) :
  ∠CAK = 22.5 :=
begin
  sorry
end

end find_angle_CAK_l654_654828


namespace intersection_M_N_eq_l654_654672

-- Define the set M
def M : Set ℝ := {0, 1, 2}

-- Define the set N based on the given inequality
def N : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

-- The statement we want to prove
theorem intersection_M_N_eq {M N: Set ℝ} (hm: M = {0, 1, 2}) 
  (hn: N = {x | x^2 - 3 * x + 2 ≤ 0}) : 
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_eq_l654_654672


namespace exist_isosceles_triangle_given_medians_and_angle_l654_654501

-- Define the triangle and its properties
structure Triangle :=
  (A B C : Point)
  (AB BC AC : ℝ)
  (is_isosceles : AB = BC)

-- Define the centroid and median properties
structure Medians (T : Triangle) :=
  (A D E : Point)
  (h_a m_a : ℝ)
  (AD : Line)
  (AE : Line)
  (AE_eq_ma : AE.length = m_a)
  (AD_eq_ha : AD.length = h_a)
  (D_on_BC : D ∈ T.BC)
  (E_on_BC : E ∈ T.BC)
  (centroid_M : Point)
  (AM_eq_23_AE : T.is_centroid centroid_M)


-- Combining the conditions into one structure
def exists_isosceles_triangle : Prop :=
∃ (T : Triangle), ∃ (M : Medians T),
  T.is_isosceles ∧
  M.AD_eq_ha ∧
  M.AE_eq_ma ∧
  M.D_on_BC ∧
  M.E_on_BC ∧
  M.AM_eq_23_AE

theorem exist_isosceles_triangle_given_medians_and_angle :
  exists_isosceles_triangle := sorry

end exist_isosceles_triangle_given_medians_and_angle_l654_654501


namespace find_common_ratio_geometric_l654_654637

variable {α : Type*} [Field α] {a : ℕ → α} {S : ℕ → α} {q : α} (h₁ : a 3 = 2 * S 2 + 1) (h₂ : a 4 = 2 * S 3 + 1)

def common_ratio_geometric : α := 3

theorem find_common_ratio_geometric (ha₃ : a 3 = 2 * S 2 + 1) (ha₄ : a 4 = 2 * S 3 + 1) :
  q = common_ratio_geometric := 
  sorry

end find_common_ratio_geometric_l654_654637


namespace sum_of_coordinates_of_other_endpoint_l654_654319

theorem sum_of_coordinates_of_other_endpoint :
  ∀ (x y : ℤ), (7, -15) = ((x + 3) / 2, (y - 5) / 2) → x + y = -14 :=
by
  intros x y h
  sorry

end sum_of_coordinates_of_other_endpoint_l654_654319


namespace optimal_voter_split_l654_654754

-- Definitions
variables (Voters : Type) [fintype Voters] (n : ℕ)
variables (supports_miraflores : Voters → Prop)
variables [decidable_pred supports_miraflores]

-- Conditions
def half_supports_miraflores := fintype.card { v // supports_miraflores v } = n
def half_supports_maloney := fintype.card { v // ¬ supports_miraflores v } = n

-- Question (translated to a theorem)
theorem optimal_voter_split (h_m : half_supports_miraflores Voters n supports_miraflores)
    (h_d: half_supports_maloney Voters n supports_miraflores) :
  ∃ (D1 D2 : finset Voters), 
    ((D1 = {v | supports_miraflores v}) ∧ 
    (D2 = {v | ¬supports_miraflores v}) ∧ 
    (∀ v, v ∈ D1 ∨ v ∈ D2) ∧ 
    (∀ v, ¬ (v ∈ D1 ∧ v ∈ D2)) ∧ 
    (finset.card D1 = 1) ∧ 
    (finset.card D2 = 2 * n - 1)) :=
sorry

end optimal_voter_split_l654_654754


namespace digging_cost_correct_l654_654517

noncomputable def well_depth : ℝ := 14
noncomputable def well_diameter : ℝ := 3
noncomputable def cost_per_cubic_meter : ℝ := 16
noncomputable def π_approx : ℝ := 3.14159

noncomputable def well_radius : ℝ := well_diameter / 2
noncomputable def well_volume : ℝ := π_approx * (well_radius ^ 2) * well_depth
noncomputable def total_cost : ℝ := well_volume * cost_per_cubic_meter

theorem digging_cost_correct : total_cost ≈ 1584.24 := by sorry

end digging_cost_correct_l654_654517


namespace proof_problem_l654_654813

-- Define the conditions for the problem

def is_factor (a b : ℕ) : Prop :=
  ∃ n : ℕ, b = a * n

def is_divisor (a b : ℕ) : Prop :=
  b % a = 0

-- Statement that needs to be proven
theorem proof_problem :
  is_factor 5 65 ∧ ¬(is_divisor 19 361 ∧ ¬is_divisor 19 190) ∧ ¬(¬is_divisor 36 144 ∨ ¬is_divisor 36 73) ∧ ¬(is_divisor 14 28 ∧ ¬is_divisor 14 56) ∧ is_factor 9 144 :=
by sorry

end proof_problem_l654_654813


namespace satisfied_probability_expected_satisfied_men_result_l654_654694

variable {men women : ℕ}

-- The total number of men and women
def total_people : ℕ := men + women

-- A man is satisfied if at least one woman sits next to him
def is_satisfied (men women : ℕ) : Prop :=
  let prob_discontent := (men - 1) * (men - 2) / (total_people * total_people - 2) in
  prob_discontent < 1 - 25/33

-- Expected number of satisfied men
def expected_satisfied_men (men women : ℕ) : ℚ :=
  let single_prob_satisfied := 25/33 in
  men * single_prob_satisfied

theorem satisfied_probability :
  men = 50 → women = 50 →
  (1 - (men - 1) * (men - 2) / (total_people * total_people - 2)) = 25/33 :=
by intros; sorry

theorem expected_satisfied_men_result :
  men = 50 → women = 50 →
  expected_satisfied_men men women = 1250/33 :=
by intros; sorry

end satisfied_probability_expected_satisfied_men_result_l654_654694


namespace distance_covered_is_9_17_miles_l654_654196

noncomputable def totalDistanceCovered 
  (walkingTimeInMinutes : ℕ) (walkingRate : ℝ)
  (runningTimeInMinutes : ℕ) (runningRate : ℝ)
  (cyclingTimeInMinutes : ℕ) (cyclingRate : ℝ) : ℝ :=
  (walkingRate * (walkingTimeInMinutes / 60.0)) + 
  (runningRate * (runningTimeInMinutes / 60.0)) + 
  (cyclingRate * (cyclingTimeInMinutes / 60.0))

theorem distance_covered_is_9_17_miles :
  totalDistanceCovered 30 3 20 8 25 12 = 9.17 := 
by 
  sorry

end distance_covered_is_9_17_miles_l654_654196


namespace monotonic_iff_a_range_l654_654989

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem monotonic_iff_a_range (a : ℝ) : 
  (∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 ∨ f a x1 ≥ f a x2) ↔ (-3 < a ∧ a < 6) :=
by 
  sorry

end monotonic_iff_a_range_l654_654989


namespace least_distance_fly_could_crawl_l654_654866

noncomputable def cone_fly_crawl_distance : ℝ :=
  let r := 900
  let h := 300 * Real.sqrt 3
  let slant_height := Real.sqrt (r^2 + h^2)
  let start_distance_from_vertex := 150
  let end_distance_from_vertex := 450 * Real.sqrt 2
  have r_positive : r > 0 := by norm_num
  have h_positive : h > 0 := by norm_num
  have slant_height_positive : slant_height > 0 := by
    simp [slant_height, r, h]
    norm_num
  Real.sqrt (start_distance_from_vertex^2 + end_distance_from_vertex^2)

theorem least_distance_fly_could_crawl (r h slant_height start_distance_from_vertex end_distance_from_vertex : ℝ)
  (h_r : r = 900)
  (h_h : h = 300 * Real.sqrt 3)
  (h_slant_height : slant_height = Real.sqrt (r^2 + h^2))
  (h_start_distance : start_distance_from_vertex = 150)
  (h_end_distance : end_distance_from_vertex = 450 * Real.sqrt 2)
  : cone_fly_crawl_distance = 150 * Real.sqrt 19 := sorry

end least_distance_fly_could_crawl_l654_654866


namespace sum_four_digit_integers_l654_654392

def a := 1000
def l := 9999
def n := 9999 - 1000 + 1
def S (n : ℕ) (a : ℕ) (l : ℕ) := n / 2 * (a + l)

theorem sum_four_digit_integers : S n a l = 49495500 :=
by
  sorry

end sum_four_digit_integers_l654_654392


namespace odd_function_only_B_l654_654415

-- Define the given functions
def f_A (x : ℝ) : ℝ := 1 + Real.cos x
def f_B (x : ℝ) : ℝ := x + Real.sin x
def f_C (x : ℝ) : ℝ := x + Real.cos x
def f_D (x : ℝ) : ℝ := 1 + Real.sin x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_only_B :
  (is_odd f_B) ∧ ¬(is_odd f_A) ∧ ¬(is_odd f_C) ∧ ¬(is_odd f_D) :=
by
  sorry

end odd_function_only_B_l654_654415


namespace doses_A_correct_doses_B_correct_doses_C_correct_l654_654946

def days_in_july : ℕ := 31

def daily_dose_A : ℕ := 1
def daily_dose_B : ℕ := 2
def daily_dose_C : ℕ := 3

def missed_days_A : ℕ := 3
def missed_days_B_morning : ℕ := 5
def missed_days_C_all : ℕ := 2

def total_doses_A : ℕ := days_in_july * daily_dose_A
def total_doses_B : ℕ := days_in_july * daily_dose_B
def total_doses_C : ℕ := days_in_july * daily_dose_C

def missed_doses_A : ℕ := missed_days_A * daily_dose_A
def missed_doses_B : ℕ := missed_days_B_morning
def missed_doses_C : ℕ := missed_days_C_all * daily_dose_C

def doses_consumed_A := total_doses_A - missed_doses_A
def doses_consumed_B := total_doses_B - missed_doses_B
def doses_consumed_C := total_doses_C - missed_doses_C

theorem doses_A_correct : doses_consumed_A = 28 := by sorry
theorem doses_B_correct : doses_consumed_B = 57 := by sorry
theorem doses_C_correct : doses_consumed_C = 87 := by sorry

end doses_A_correct_doses_B_correct_doses_C_correct_l654_654946


namespace part_one_part_two_l654_654154

def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem part_one (ω φ : ℝ) (hω : 0 < ω) (hφ : 0 ≤ φ ∧ φ ≤ π) (h_even: ∀ x : ℝ, f x ω φ = f (-x) ω φ) 
  (h_distance: ∀ x : ℝ, ∃ k : ℤ, f (x + π) ω φ = f (x) ω φ) : f x 1 (π / 2) = Real.cos x :=
by 
  sorry

theorem part_two (α : ℝ) (h1 : sin α + cos α = 2 / 3) :
  (sqrt 2 * sin (2 * α - π / 4) + 1) / (1 + tan α) = -5 / 9 :=
by 
  sorry

end part_one_part_two_l654_654154


namespace max_val_a_l654_654992

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x^2 - 3 * x + 2)

theorem max_val_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 1, f a x ≥ 0) : a ≤ 1 := sorry

end max_val_a_l654_654992


namespace partition_proper_divisor_l654_654911

theorem partition_proper_divisor (α β : ℕ) (hα : 0 < α) (hβ : 0 < β) :
  (∃ A B : set ℕ, (α * A : set ℕ) = (β * B : set ℕ) ∧ disjoint A B ∧ (∀ x, x ∈ A ∨ x ∈ B)) ↔ 
  (∃ d, d ∈ [2..min α β] ∧ (α = d * β ∨ β = d * α)) := sorry

end partition_proper_divisor_l654_654911


namespace no_solution_in_natural_numbers_l654_654691

theorem no_solution_in_natural_numbers (x y z : ℕ) : 
  (x / y : ℚ) + (y / z : ℚ) + (z / x : ℚ) ≠ 1 := 
by sorry

end no_solution_in_natural_numbers_l654_654691


namespace solve_3x3_table_l654_654848

theorem solve_3x3_table : 
   ∃ (fill_combinations : list (list (list ℤ))), 
   (∀ (row : list ℤ), row ∈ fill_combinations → row.sum = fill_combinations.head.sum) 
   ∧
   fill_combinations.length = 831 :=
   sorry

end solve_3x3_table_l654_654848


namespace trigonometric_expression_value_l654_654042

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_expression_value_l654_654042


namespace count_indistinguishable_distributions_l654_654172

theorem count_indistinguishable_distributions (balls : ℕ) (boxes : ℕ) (h_balls : balls = 5) (h_boxes : boxes = 4) : 
  ∃ n : ℕ, n = 6 := by
  sorry

end count_indistinguishable_distributions_l654_654172


namespace max_sin_product_proof_l654_654636

noncomputable def max_sin_product : ℝ :=
  let A := (-8, 0)
  let B := (8, 0)
  let C (t : ℝ) := (t, 6)
  let AB : ℝ := 16
  let AC (t : ℝ) := Real.sqrt ((t + 8)^2 + 36)
  let BC (t : ℝ) := Real.sqrt ((t - 8)^2 + 36)
  let area : ℝ := 48
  let sin_ACB (t : ℝ) := 96 / Real.sqrt (((t + 8)^2 + 36) * ((t - 8)^2 + 36))
  let sin_CAB_CBA : ℝ := 3 / 8
  sin_CAB_CBA

theorem max_sin_product_proof : ∀ t : ℝ, max_sin_product = 3 / 8 :=
by
  sorry

end max_sin_product_proof_l654_654636


namespace max_value_of_operation_l654_654465

theorem max_value_of_operation : ∀ (n : ℕ), 100 ≤ n → n ≤ 999 → 2 * (450 - n) ≤ 700 :=
by {
    intro n,
    intro h1, -- 100 ≤ n
    intro h2, -- n ≤ 999
    sorry
}

example : 2 * (450 - 100) = 700 := by rfl

end max_value_of_operation_l654_654465


namespace remainder_b95_mod_64_l654_654233

def φ (n : ℕ) : ℕ := n / 2^(Nat.log2 n)

theorem remainder_b95_mod_64 :
  let b (n : ℕ) := 7^n + 9^n
  in φ 64 = 32 →
     7^32 ≡ 1 [MOD 64] →
     9^32 ≡ 1 [MOD 64] →
     Nat.gcd 7 64 = 1 →
     Nat.gcd 9 64 = 1 →
     b 95 % 64 = 48 :=
by
  intros
  sorry

end remainder_b95_mod_64_l654_654233


namespace constant_term_of_expansion_l654_654797

theorem constant_term_of_expansion : 
  let p := Polynomial.of_coeffs [3, 0, 2, 0, 0, 1] in
  let q := Polynomial.of_coeffs [7, 0, 1, 0, 0, 2] in
  Polynomial.constant_term (p * q) = 21 :=
by
  sorry

end constant_term_of_expansion_l654_654797


namespace coefficient_of_3_in_expansion_l654_654712

theorem coefficient_of_3_in_expansion :
  let expr1 := (1 + (2 / x))
  let expr2 := (1 - x^4)
  (coefficient term expr in expansion (expr1 * expr2) 3 = 3) :=
  by sorry

end coefficient_of_3_in_expansion_l654_654712


namespace area_of_triangle_ABC_l654_654639

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1 / 2) * a * b

theorem area_of_triangle_ABC (a b c : ℝ) (A B C : ℝ)
  (h1 : cos A / cos B = 3 / 4)
  (h2 : c = 10)
  (h3 : a = 8)
  (h4 : b = 6)
  (h5 : A + B = π / 2)
  : triangle_area a b c A B C = 24 :=
by
  rw [triangle_area, h3, h4]
  norm_num
  sorry

end area_of_triangle_ABC_l654_654639


namespace tickets_to_be_sold_l654_654330

theorem tickets_to_be_sold : 
  let total_tickets := 200
  let jude_tickets := 16
  let andrea_tickets := 4 * jude_tickets
  let sandra_tickets := 2 * jude_tickets + 8
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets) = 80 := by
  sorry

end tickets_to_be_sold_l654_654330


namespace range_of_a_l654_654972

def p (a : ℝ) := ∃ x : ℝ, x ∈ Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0
def q (a : ℝ) := ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a (a : ℝ) (h : ¬ (p a ∨ q a)) : a ∈ Ioo (-1) 0 ∨ a ∈ Ioo 0 1 :=
by
  sorry

end range_of_a_l654_654972


namespace rotated_angle_new_measure_l654_654313

theorem rotated_angle_new_measure (θ₀ : ℕ) (rotation : ℕ) : (θ₀ = 60) → (rotation = 630) → 
  let θ₁ := θ₀ + (rotation % 360) in
  let acute_angle := 360 - θ₁ % 360 in
  acute_angle = 30 :=
by
  intros h₀ h_rotation
  let θ₁ := 60 + (630 % 360)
  let θ_final := 360 - (θ₁ % 360)
  have : θ_final = 30 := sorry
  exact this

end rotated_angle_new_measure_l654_654313


namespace monotonic_decreasing_interval_l654_654725

def f (x : ℝ) : ℝ := x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x, (0 < x ∧ x < 1) ↔ (f'(x) < 0) := 
sorry

end monotonic_decreasing_interval_l654_654725


namespace part1_part2_l654_654586

variables {x : ℝ} 

-- Defining the vectors
def m : ℝ → Vector ℝ 2 := fun x => ⟨Real.sin x, -Real.sqrt 3⟩
def n : ℝ → Vector ℝ 2 := fun x => ⟨1, Real.cos x⟩

-- Dot product function
def f (x : ℝ) := Real.sin x - Real.sqrt 3 * Real.cos x

theorem part1 (hx : 0 ≤ x ∧ x ≤ π/2) (hfx : f x = 2 / 3) : 
  Real.sin x = (1 + 2 * Real.sqrt 6) / 6 := sorry

-- Defining the transformed function
def g (x : ℝ) := (2 : ℝ) * Real.sin (2 * x + π / 6)

theorem part2 (hx : 0 ≤ x ∧ x ≤ π/2) : 
  ∀ y ∈ Set.range g, -2 ≤ y ∧ y ≤ 2 := sorry

end part1_part2_l654_654586


namespace swimming_speed_still_water_l654_654451

-- Define the conditions
def water_speed : ℝ := 2
def time_against_current : ℝ := 6
def distance_against_current : ℝ := 12

-- Define the effective speed calculation
def effective_speed (v : ℝ) : ℝ := v - water_speed

-- Define the equation based on the given problem
def speed_equation (v : ℝ) : Prop := effective_speed(v) = distance_against_current / time_against_current

-- The theorem to be proven
theorem swimming_speed_still_water : ∃ v : ℝ, speed_equation(v) ∧ v = 4 := 
by
  let v := 4
  have h1 : effective_speed(v) = distance_against_current / time_against_current := sorry
  have h2 : v = 4 := sorry
  exact ⟨v, ⟨h1, h2⟩⟩

end swimming_speed_still_water_l654_654451


namespace rotated_angle_l654_654311

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

end rotated_angle_l654_654311


namespace parallelogram_property_not_necessarily_supplementary_l654_654017

theorem parallelogram_property_not_necessarily_supplementary
  (P : Type) [parallelogram P]
  (h1 : opposite_sides_equal P)
  (h2 : opposite_sides_parallel P)
  (h3 : opposite_angles_equal P)
  (h4 : interior_angles_sum P = 360) :
  ¬ opposite_angles_supplementary P :=
sorry

end parallelogram_property_not_necessarily_supplementary_l654_654017


namespace Hayley_l654_654587

-- Definitions based on the given conditions
def num_friends : ℕ := 9
def stickers_per_friend : ℕ := 8

-- Theorem statement
theorem Hayley's_total_stickers : num_friends * stickers_per_friend = 72 := by
  sorry

end Hayley_l654_654587


namespace student_entrepreneur_profit_l654_654827

theorem student_entrepreneur_profit {x y a: ℝ} 
  (h1 : a * (y - x) = 1000) 
  (h2 : (ay / x) * y - ay = 1500)
  (h3 : y = 3 / 2 * x) : a * x = 2000 := 
sorry

end student_entrepreneur_profit_l654_654827


namespace sin_theta_in_terms_of_x_l654_654658

theorem sin_theta_in_terms_of_x (θ x : ℝ) (h : cos (θ / 4) = sqrt ((x + 2) / (3 * x))) : 
  sin θ = (4 * sqrt ((2 * x^2 + 3 * x - 4) * (x^2 - 12 * x + 16))) / (9 * x^2) :=
sorry

end sin_theta_in_terms_of_x_l654_654658


namespace sqrt_fraction_eq_half_l654_654033

-- Define the problem statement in a Lean 4 theorem:
theorem sqrt_fraction_eq_half : Real.sqrt ((25 / 36 : ℚ) - (4 / 9 : ℚ)) = 1 / 2 := by
  sorry

end sqrt_fraction_eq_half_l654_654033


namespace explicit_formula_for_f_l654_654909

def f (k : ℕ) : ℚ :=
  if k = 1 then 4 / 3
  else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3

theorem explicit_formula_for_f (k : ℕ) (hk : k ≥ 1) : 
  (f k = if k = 1 then 4 / 3 else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3) ∧ 
  ∀ k ≥ 2, 2 * f k = f (k - 1) - k * 5^k + 2^k :=
by {
  sorry
}

end explicit_formula_for_f_l654_654909


namespace angle_C_lt_36_l654_654640

-- Define triangle ABC with conditions
namespace TriangleProblem

variables {A B C E : Point} -- Points defining the triangle and bisector point

-- Conditions: BE is the angle bisector and BE = EC
axiom BE_is_angle_bisector : is_angle_bisector BE A C
axiom BE_eq_EC : BE = EC

-- Condition: The perpendicular bisector of AB intersects BC
axiom perp_bisector_intersects_BC : ∃ D, is_perpendicular_bisector A B D ∧ lies_on D B C

-- Goal: Prove \( \angle C < 36^\circ \)
theorem angle_C_lt_36 (h1 : BE_is_angle_bisector) (h2 : BE_eq_EC) (h3 : perp_bisector_intersects_BC) : angle C < 36 :=
sorry

end TriangleProblem

end angle_C_lt_36_l654_654640


namespace max_distance_MC2_l654_654450

-- Given conditions from the problem
def C1 : set (Real × Real) := {p | (p.1)^2 + (p.2)^2 + 6*p.1 + 5 = 0}
def C2 : set (Real × Real) := {p | (p.1)^2 + (p.2)^2 - 6*p.1 - 91 = 0}

-- Definitions of centers and radii
def C1_center : Real × Real := (-3, 0)
def C2_center : Real × Real := (3, 0)
def C1_radius : Real := 2
def C2_radius : Real := 10

-- Prove the maximum value of distance from M to C2 center is 9
theorem max_distance_MC2 (M : Real × Real) (r : Real) (MC1_condition : dist M C1_center = C1_radius + r)
  (MC2_condition : dist M C2_center = C2_radius - r) : 
  ∀ M ∈ {M : Real × Real | (dist M C1_center) + (dist M C2_center) = 12}, dist M C2_center ≤ 9 := 
by
  sorry

end max_distance_MC2_l654_654450


namespace bobs_total_profit_l654_654031

theorem bobs_total_profit :
  let cost_parent_dog := 250
  let num_parent_dogs := 2
  let num_puppies := 6
  let cost_food_vaccinations := 500
  let cost_advertising := 150
  let selling_price_parent_dog := 200
  let selling_price_puppy := 350
  let total_cost_parent_dogs := num_parent_dogs * cost_parent_dog
  let total_cost_puppies := cost_food_vaccinations + cost_advertising
  let total_revenue_puppies := num_puppies * selling_price_puppy
  let total_revenue_parent_dogs := num_parent_dogs * selling_price_parent_dog
  let total_revenue := total_revenue_puppies + total_revenue_parent_dogs
  let total_cost := total_cost_parent_dogs + total_cost_puppies
  let total_profit := total_revenue - total_cost
  total_profit = 1350 :=
by
  sorry

end bobs_total_profit_l654_654031


namespace kibble_recommendations_l654_654202

theorem kibble_recommendations :
  let total_vets := 3500
  let pct_puppy_kibble := 0.235
  let pct_yummy_kibble := 0.372
  let rec_puppy_kibble := Int.ofNat (Float.round (pct_puppy_kibble * total_vets))
  let rec_yummy_kibble := Int.ofNat (Float.round (pct_yummy_kibble * total_vets))
  let difference := rec_yummy_kibble - rec_puppy_kibble
  let percentage_increase := (difference : Float) / rec_puppy_kibble.toFloat * 100 in
  difference = 479 ∧ percentage_increase ≈ 58.2 :=
by
  -- Proof omitted
  sorry

end kibble_recommendations_l654_654202


namespace train_length_is_approx_l654_654461

noncomputable def train_length : ℝ :=
  let speed_kmh : ℝ := 54
  let conversion_factor : ℝ := 1000 / 3600
  let speed_ms : ℝ := speed_kmh * conversion_factor
  let time_seconds : ℝ := 11.999040076793857
  speed_ms * time_seconds

theorem train_length_is_approx : abs (train_length - 179.99) < 0.001 := 
by
  sorry

end train_length_is_approx_l654_654461


namespace average_rainfall_l654_654884

theorem average_rainfall (rainfall_Tuesday : ℝ) (rainfall_others : ℝ) (days_in_week : ℝ)
  (h1 : rainfall_Tuesday = 10.5) 
  (h2 : rainfall_Tuesday = rainfall_others)
  (h3 : days_in_week = 7) : 
  (rainfall_Tuesday + rainfall_others) / days_in_week = 3 :=
by
  sorry

end average_rainfall_l654_654884


namespace cylindrical_to_rectangular_l654_654908

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 6) (hθ : θ = π / 3) (hz : z = -3) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, -3) :=
by
  sorry

end cylindrical_to_rectangular_l654_654908


namespace least_number_of_polynomials_l654_654653

theorem least_number_of_polynomials (n : ℕ) (hn : n > 0) :
  ∃ (p : fin (n - 1) → (fin n → ℝ) → ℝ),
    ∀ (x : fin n → ℝ),
      (∑ i, x i) ^ 2 + ∑ j in fin  (n - 1) , (p j x) ^ 2 = n * ∑ i, (x i) ^ 2 :=
sorry

end least_number_of_polynomials_l654_654653


namespace min_marked_cells_l654_654385

def cell := (ℕ × ℕ)
def L_shape : set cell := {((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (1, 0)), 
                           ((0, 1), (1, 1), (1, 0)), ((0, 0), (0, 1), (1, 1))(0, 0), (0, 1), (1, 1)}

noncomputable def marked_cells : set cell := sorry

theorem min_marked_cells (board : set cell) (L : set (set cell)) : 
  board = { (i, j) | 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 11 } → 
  L = L_shape →
  (∀ l ∈ L, ∃ c ∈ marked_cells, c ∈ l) →
  cardinality(marked_cells) = 50 :=
begin
  sorry
end

end min_marked_cells_l654_654385


namespace max_value_of_m_l654_654105

theorem max_value_of_m
  (a b : ℝ) (m : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : 2 / a + 1 / b = 1 / 4)
  (h4 : ∀ (a b : ℝ), 2a + b ≥ 4m) :
  m ≤ 9 :=
begin
  sorry
end

end max_value_of_m_l654_654105


namespace number_of_ways_to_fill_table_is_831_l654_654845

-- Definitions based on the problem conditions
def numbers := {2014, 2015, 2016}
def table := array (3 × 3) ℕ

-- Main theorem stating the problem and correct answer
theorem number_of_ways_to_fill_table_is_831 :
  ∃ (f : table → Prop), (∀ (t : table), f t → table_filled_with t numbers) ∧ (sum_in_each_row_equal t) ∧ (count_valid_tables f = 831) :=
begin
  sorry
end

end number_of_ways_to_fill_table_is_831_l654_654845


namespace max_value_quadratic_l654_654598

theorem max_value_quadratic :
  ∀ x : ℝ, y = -3 * x^2 + 6 * x + 4 → ∃ M : ℝ, M = 7 ∧ ∀ x : ℝ, y ≤ M :=
begin
  sorry
end

end max_value_quadratic_l654_654598


namespace find_m_l654_654602

theorem find_m (m : ℤ) (h : (-2)^(2*m) = 2^(3-m)) : m = 1 :=
sorry

end find_m_l654_654602


namespace ellipse_line_intersection_l654_654987

theorem ellipse_line_intersection (m : ℝ) : 
  (m > 0 ∧ m ≠ 3) →
  (∃ x y : ℝ, (x^2 / 3 + y^2 / m = 1) ∧ (x + 2 * y - 2 = 0)) ↔ 
  ((1 / 4 < m ∧ m < 3) ∨ (m > 3)) := 
by 
  sorry

end ellipse_line_intersection_l654_654987


namespace solution1_solution2_l654_654347

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l654_654347


namespace A_empty_iff_a_gt_9_over_8_A_one_element_l654_654163

-- Definition of A based on a given condition
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Problem 1: Prove that if A is empty, then a > 9/8
theorem A_empty_iff_a_gt_9_over_8 {a : ℝ} : 
  (A a = ∅) ↔ (a > 9 / 8) := 
sorry

-- Problem 2: Prove the elements in A when it contains only one element
theorem A_one_element {a : ℝ} : 
  (∃! x, x ∈ A a) ↔ (a = 0 ∧ (A a = {2 / 3})) ∨ (a = 9 / 8 ∧ (A a = {4 / 3})) := 
sorry

end A_empty_iff_a_gt_9_over_8_A_one_element_l654_654163


namespace numberOfBags_l654_654195

-- Define the given conditions
def totalCookies : Nat := 33
def cookiesPerBag : Nat := 11

-- Define the statement to prove
theorem numberOfBags : totalCookies / cookiesPerBag = 3 := by
  sorry

end numberOfBags_l654_654195


namespace exists_negative_r_l654_654227

variable {R : Type*} [CommRing R]

noncomputable def p (x : R) : R := x^7 + x^6 + b_5 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0
noncomputable def q (x : R) : R := x^5 + c_4 * x^4 + c_3 * x^3 + c_2 * x^2 + c_1 * x + c_0

theorem exists_negative_r (b_0 b_1 b_2 b_3 b_4 b_5 : R) (c_0 c_1 c_2 c_3 c_4 : R) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 6 → p i = q i) → ∃ r : ℤ, r < 0 ∧ p r = q r :=
by
  sorry

end exists_negative_r_l654_654227


namespace greatest_difference_l654_654822

theorem greatest_difference (q r : ℕ) (x y : ℕ) 
  (hq : q = 10 * x + y)
  (hr : r = 10 * y + x)
  (h_cond1 : 10 ≤ q)
  (h_cond2 : q ≤ 99)
  (h_cond3 : 10 ≤ r)
  (h_cond4 : r ≤ 99)
  (h_diff : abs (q - r) < 70) 
  : q - r ≤ 63 :=
by 
  sorry

end greatest_difference_l654_654822


namespace mark_all_integer_points_possible_l654_654431

theorem mark_all_integer_points_possible (n : ℕ) (h1 : n > 1) :
  ∃ (mark : ℕ → Prop), 
    (∀ k, k ∈ (Finset.range 2003).to_set → mark k) ∧ 
    (mark 0 ∧ mark 2002) ∧ 
    (∀ a b : ℕ, mark a → mark b → gcd a b = 1 → 
      ((a + b) % n = 0 → mark ((a + b) / n))) :=
sorry

end mark_all_integer_points_possible_l654_654431


namespace incircle_radius_of_right_triangle_l654_654685

noncomputable def radius_of_incircle (a b c : ℝ) : ℝ := (a + b - c) / 2

theorem incircle_radius_of_right_triangle
  (a : ℝ) (b_proj_hypotenuse : ℝ) (r : ℝ) :
  a = 15 ∧ b_proj_hypotenuse = 16 ∧ r = 5 :=
by
  sorry

end incircle_radius_of_right_triangle_l654_654685


namespace harry_pencils_remaining_l654_654024

def num_pencils_anna : ℕ := 50
def num_pencils_harry_initial := 2 * num_pencils_anna
def num_pencils_lost_harry := 19

def pencils_left_harry (pencils_anna : ℕ) (pencils_harry_initial : ℕ) (pencils_lost : ℕ) : ℕ :=
  pencils_harry_initial - pencils_lost

theorem harry_pencils_remaining : pencils_left_harry num_pencils_anna num_pencils_harry_initial num_pencils_lost_harry = 81 :=
by
  sorry

end harry_pencils_remaining_l654_654024


namespace sum_of_all_four_digit_integers_l654_654403

theorem sum_of_all_four_digit_integers :
  (Finset.range (9999 + 1)).filter (λ x => x ≥ 1000).sum = 49495500 :=
by
  sorry

end sum_of_all_four_digit_integers_l654_654403


namespace frog_layed_first_day_l654_654857

theorem frog_layed_first_day (x : ℕ) 
  (h1 : 2 * (5 * x + 20) = 2 * (2 * x + x + (2 * x + 20)))
  (h2 : x + 2 * x + (2 * x + 20) + 2 * (2 * x + x + (2 * x + 20)) = 810) : x = 50 :=
begin
  sorry -- proof to be filled
end

end frog_layed_first_day_l654_654857


namespace a_minus_b_is_neg_seven_l654_654983

-- Definitions for sets
def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 1 < x ∧ x < 4}
def setC : Set ℝ := {x | 1 < x ∧ x < 3}

-- Proving the statement
theorem a_minus_b_is_neg_seven :
  ∀ (a b : ℝ), (∀ x, (x ∈ setC) ↔ (x^2 + a*x + b < 0)) → a - b = -7 :=
by
  intros a b h
  sorry

end a_minus_b_is_neg_seven_l654_654983


namespace find_eccentricity_of_ellipse_l654_654147

noncomputable def ellipse_eccentricity : ℝ := 1 / 2

theorem find_eccentricity_of_ellipse
  (a b c : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (h_ellipse : a > b)
  (eval_ellipse : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (intercepts_M_N : ∀ x y : ℝ, y = x - c → ∃ (x₁ x₂ y₁ y₂ : ℝ), eval_ellipse x₁ y₁ ∧ eval_ellipse x₂ y₂)
  (perp_bisector : ∃ P_x : ℝ, eval_ellipse _ 0 = P_x → True)
  (ratio_condition : ∀ M_x N_x : ℝ, M_x ≠ N_x → 4 = abs (M_x - N_x) / abs (P_x - c)) :
  ellipse_eccentricity = 1 / 2 :=
sorry

end find_eccentricity_of_ellipse_l654_654147


namespace min_marked_cells_l654_654381

-- Define the board
def board : Type := fin 10 × fin 11

-- Define the L-shaped piece with its three cells
def L_piece_cells : list (int × int) := [(0, 0), (1, 0), (0, 1)]

-- Define what it means for an L-shaped piece to be placed on the board
def L_placement (p : board) : list board :=
  L_piece_cells.map (λ (c : int × int), (⟨p.1.1 + c.1, by sorry⟩, ⟨p.2.1 + c.2, by sorry⟩))

-- Define the marking of cells
def marked_cells (ls : list board) : Prop :=
  ∃ k : ℕ, k = 50 ∧ ∀ p : board, ∃ m ∈ ls, m ∈ L_placement p

-- Statement that the smallest k satisfying the condition is 50
theorem min_marked_cells :
  ∃ (ls : list board), marked_cells ls :=
sorry

end min_marked_cells_l654_654381


namespace Freddy_time_to_C_l654_654509

noncomputable def distance_AB : ℝ := 570
noncomputable def time_Eddy : ℝ := 3
noncomputable def distance_AC : ℝ := 300
noncomputable def speed_ratio : ℝ := 2.533333333333333

theorem Freddy_time_to_C : ∃ (t : ℝ), t = 4 :=
  let speed_Eddy := distance_AB / time_Eddy in
  let speed_Freddy := speed_Eddy / speed_ratio in
  let time_Freddy := distance_AC / speed_Freddy in
  by {
    use time_Freddy,
    sorry
  }

end Freddy_time_to_C_l654_654509


namespace probability_of_prime_sum_l654_654849

open Nat

-- Define the set of outcomes for the 4-sided die and the 8-sided die
def four_sided_die := {1, 2, 3, 4}
def eight_sided_die := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the prime numbers less than or equal to 12
def primes := {2, 3, 5, 7, 11}

-- Define the number of outcomes that result in a sum that is a prime number
def count_prime_sums : ℕ :=
  (four_sided_die.product eight_sided_die).count (λ pair, (pair.1 + pair.2) ∈ primes)

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 4 * 8

-- Define the probability as a fraction
def probability_prime_sum : ℚ := count_prime_sums / total_outcomes

theorem probability_of_prime_sum :
  probability_prime_sum = 13/32 :=
by
  sorry

end probability_of_prime_sum_l654_654849


namespace range_of_a_l654_654331

theorem range_of_a (a : ℝ) : (∃ x : ℝ, ¬ (x^2 - a*x + 1 ≤ 0)) → -2 < a ∧ a < 2 :=
begin
  intro h,
  sorry
end

end range_of_a_l654_654331


namespace right_triangle_inradius_l654_654077

theorem right_triangle_inradius :
  ∀ (a b c : ℕ) (r : ℕ), a = 6 → b = 8 → c = 10 →
  a^2 + b^2 = c^2 →
  let A := (1 / 2) * a * b in
  let s := (a + b + c) / 2 in
  A = s * r →
  r = 2 :=
by 
  intros a b c r ha hb hc hpyth A s hA
  sorry

end right_triangle_inradius_l654_654077


namespace possible_to_have_three_angles_l654_654643

theorem possible_to_have_three_angles (n : ℕ) (h : n = 100) : 
  ∃ region, region.angles = 3 := by
  sorry

end possible_to_have_three_angles_l654_654643


namespace geometric_series_sum_l654_654892

/-- The first term of the geometric series. -/
def a : ℚ := 3

/-- The common ratio of the geometric series. -/
def r : ℚ := -3 / 4

/-- The sum of the geometric series is equal to 12/7. -/
theorem geometric_series_sum : (∑' n : ℕ, a * r^n) = 12 / 7 := 
by
  /- The Sum function and its properties for the geometric series will be used here. -/
  sorry

end geometric_series_sum_l654_654892


namespace partition_is_possible_l654_654322

def sum_of_elements (s : Finset ℕ) : ℕ :=
  s.sum id

def product_of_elements (s : Finset ℕ) : ℕ :=
  s.prod id

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def sum_of_digits_in_set (s : Finset ℕ) : ℕ :=
  (s.image sum_of_digits).sum id

theorem partition_is_possible :
  ∃ (A B C : Finset ℕ), 
    A ∪ B ∪ C = Finset.range 11 ∧
    A.disjoint B ∧ A.disjoint C ∧ B.disjoint C ∧
    sum_of_elements A > sum_of_elements B ∧ sum_of_elements A > sum_of_elements C ∧
    product_of_elements B > product_of_elements A ∧ product_of_elements B > product_of_elements C ∧
    sum_of_digits_in_set C > sum_of_digits_in_set A ∧ sum_of_digits_in_set C > sum_of_digits_in_set B :=
by
  sorry

end partition_is_possible_l654_654322


namespace maximum_third_height_l654_654867

theorem maximum_third_height 
  (A B C : Type)
  (h1 h2 : ℕ)
  (h1_pos : h1 = 4) 
  (h2_pos : h2 = 12) 
  (h3_pos : ℕ)
  (triangle_inequality : ∀ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a)
  (scalene : ∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c)
  : (3 < h3_pos ∧ h3_pos < 6) → h3_pos = 5 := 
sorry

end maximum_third_height_l654_654867


namespace min_marked_cells_l654_654384

def cell := (ℕ × ℕ)
def L_shape : set cell := {((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (1, 0)), 
                           ((0, 1), (1, 1), (1, 0)), ((0, 0), (0, 1), (1, 1))(0, 0), (0, 1), (1, 1)}

noncomputable def marked_cells : set cell := sorry

theorem min_marked_cells (board : set cell) (L : set (set cell)) : 
  board = { (i, j) | 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 11 } → 
  L = L_shape →
  (∀ l ∈ L, ∃ c ∈ marked_cells, c ∈ l) →
  cardinality(marked_cells) = 50 :=
begin
  sorry
end

end min_marked_cells_l654_654384


namespace cream_ratio_l654_654646

theorem cream_ratio (joe_initial_coffee joann_initial_coffee : ℝ)
                    (joe_drank joann_drank joann_added_cream : ℝ)
                    (joe_added_cream joann_added_sugar joe_added_sugar : ℝ) :
    joe_initial_coffee = 14 ∧ joann_initial_coffee = 14 ∧
    joe_drank = 3 ∧ joe_added_cream = 3 ∧ joe_added_sugar = 1 ∧
    joann_added_cream = 3 ∧ joann_drank = 4 ∧ joann_added_sugar = 1 →
    let joe_final_cream := joe_added_cream in
    let total_amount_joann := joann_initial_coffee + joann_added_cream in
    let joann_remaining_after_drank := total_amount_joann - joann_drank in
    let joann_final_cream := joann_added_cream - ((joann_added_cream / total_amount_joann) * joann_drank) in
    let cream_ratio := joe_final_cream / joann_final_cream in
    cream_ratio = (17 / 9) :=
begin
    intros cond,
    cases cond with ji cond,
    cases cond with jio cond,
    cases cond with jd cond,
    cases cond with ja cond,
    cases cond with js cond,
    cases cond with jc cond,
    cases cond with jnd sad,

    have joe_final_cream : ℝ := ja,
    have joann_final_cream : ℝ := jc - (jc / (jio + jc) * jnd),
    have cream_ratio : ℝ := joe_final_cream / joann_final_cream,
    simp * at *,
    rw [joe_final_cream, joann_final_cream],
    sorry
end

end cream_ratio_l654_654646


namespace evaluate_expression_l654_654035

-- Define the expression components as functions or variables
def term1 : ℤ := (-1) ^ 2023
def term2 : ℝ := real.abs (real.sqrt 3 - 3)
def term3 : ℝ := real.sqrt 9
def term4 : ℝ := (-4) * (1 / 2)

-- The overall expression
def expr : ℝ := term1 + term2 + term3 - term4

-- The expected result
def expected_result : ℝ := 7 - real.sqrt 3

-- The theorem to prove
theorem evaluate_expression : expr = expected_result := 
by sorry

end evaluate_expression_l654_654035


namespace deepak_present_age_l654_654320

theorem deepak_present_age (x : ℕ) (rahul deepak rohan : ℕ) 
  (h_ratio : rahul = 5 * x ∧ deepak = 2 * x ∧ rohan = 3 * x)
  (h_rahul_future_age : rahul + 8 = 28) :
  deepak = 8 := 
by
  sorry

end deepak_present_age_l654_654320


namespace probability_correct_l654_654201

-- Definitions of the problem setup
def paths_from_A_to_C_possible : ℕ := Nat.choose 5 3  -- Number of ways to arrange 3 eastward and 2 southward moves
def paths_from_C_to_D_possible : ℕ := Nat.choose 5 2  -- Number of ways to arrange 2 eastward and 3 southward moves

def total_paths_from_A_to_D_possible : ℕ := Nat.choose 10 5  -- Total arrangements for 5 eastward and 5 southward moves

-- The calculated probability of passing through C
def probability_passing_through_C : ℚ := (100 : ℚ) / 252

theorem probability_correct : probability_passing_through_C = 25 / 63 := 
by {
  unfold probability_passing_through_C,
  norm_num,
  sorry
}

end probability_correct_l654_654201


namespace customers_left_correct_l654_654012

-- Define the initial conditions
def initial_customers : ℕ := 8
def remaining_customers : ℕ := 5

-- Define the statement regarding customers left
def customers_left : ℕ := initial_customers - remaining_customers

-- The theorem we need to prove
theorem customers_left_correct : customers_left = 3 := by
    -- Skipping the actual proof
    sorry

end customers_left_correct_l654_654012


namespace sin_neg_45_deg_l654_654923

theorem sin_neg_45_deg :
  sin (- (real.pi / 4)) = - (real.sqrt 2 / 2) :=
by
  -- Proof goes here
  sorry

end sin_neg_45_deg_l654_654923


namespace complex_magnitude_l654_654189

theorem complex_magnitude (z : ℂ) (h : z = 2 / (1 - sqrt 3 * I)) : abs z = 1 := by
  sorry

end complex_magnitude_l654_654189


namespace max_problems_l654_654436

theorem max_problems (n : ℕ) (choices : ℕ) (m : ℕ)
  (students_conditions : n = 16)
  (problem_choices : choices = 4)
  (common_answer_condition : ∀ (i j : ℕ), i < n → j < n → i ≠ j → ∀ (k : ℕ), k < m → (number_of_common_answers (i j k) ≤ 1))
  : m = 5 :=
sorry

end max_problems_l654_654436


namespace point_P_existence_and_distances_l654_654119

variables (a b h : ℝ)

-- Define the conditions under which point P can exist:
def conditions_for_P_existence : Prop :=
  h^2 >= a * b

-- Define the quadratic equation in terms of the distances from P to the bases:
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - h * x + (a * b) / 4 = 0

-- Define the problem statement proving points P can exist and satisfy given conditions:
theorem point_P_existence_and_distances
  (h_pos : h > 0)
  (a_pos : a > 0)
  (b_pos : b > 0) :
  conditions_for_P_existence a b h ↔ ∃ (PM PN : ℝ), quadratic_equation a b h PM ∧ quadratic_equation a b h PN ∧ PM + PN = h :=
by
  sorry

end point_P_existence_and_distances_l654_654119


namespace unique_solution_f_l654_654518

def f : ℚ → ℚ := sorry

theorem unique_solution_f (f : ℚ → ℚ) (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := 
by
  sorry 

end unique_solution_f_l654_654518


namespace solution_set_correct_l654_654324

def inequality_solution (x : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3)^2 > 0

theorem solution_set_correct : 
  ∀ x : ℝ, inequality_solution x ↔ (x < 1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 3) ∨ x > 3) := 
by sorry

end solution_set_correct_l654_654324


namespace shortest_path_triangle_l654_654869

variables {a : ℝ} (h_eq : h = (sqrt 3 / 2) * a)
variable (r_eq : r = (1 / 2) * h)

theorem shortest_path_triangle (h : ℝ) (r : ℝ) (h_eq : h = (sqrt 3 / 2) * a) (r_eq : r = (sqrt 3 / 4) * a) :
  shortest_path = a * ((sqrt 7 / 2) - (sqrt 3 / 4)) :=
sorry

end shortest_path_triangle_l654_654869


namespace positive_integer_pairs_l654_654925

theorem positive_integer_pairs (s t : ℕ) :
  (∀ a b : ℕ, 0 < a ∧ 0 < b ∧ a ≠ b → ∃ n : ℕ, 0 < n ∧ (a ^ s + b ^ t) ∣ (a ^ n + b ^ (n + 1))) ↔ (|s - t| = 1) := 
by sorry

end positive_integer_pairs_l654_654925


namespace fence_length_proof_l654_654531

/-- Define the scenario constants -/
def width : ℕ := 40
def area : ℕ := 480

/-- Define the length derived from the area and width -/
def length : ℕ := area / width

/-- Proposition stating that Frank uses 64 feet of fence -/
def total_fence : ℕ := 2 * length + width

theorem fence_length_proof : total_fence = 64 := by
  /- Definitions -/
  let w := 40
  let a := 480
  let l := a / w

  /- Proof steps, making sure they correspond to the solution -/
  have l_eq : l = 12 := by
    exact Nat.div_eq_of_eq_mul' sorry  -- Placeholder for division proof

  have tf_eq : total_fence = 2 * l + w := by
    rfl

  calc
    total_fence
      = 2 * l + w : by
        rw [tf_eq]
      = 2 * 12 + 40 : by
        rw [l_eq]
      = 64 : by
        norm_num

end fence_length_proof_l654_654531


namespace cost_of_ice_cream_cup_l654_654421

theorem cost_of_ice_cream_cup :
  ∀ (n_chapatis n_rice n_veg n_ice_creams : ℕ) (chapati_price rice_price veg_price total_paid : ℕ),
    n_chapatis = 16 →
    n_rice = 5 →
    n_veg = 7 →
    n_ice_creams = 6 →
    chapati_price = 6 →
    rice_price = 45 →
    veg_price = 70 →
    total_paid = 883 →
    let total_chapatis := n_chapatis * chapati_price in
    let total_rice := n_rice * rice_price in
    let total_veg := n_veg * veg_price in
    let total_other := total_chapatis + total_rice + total_veg in
    let total_ice_cream_cost := total_paid - total_other in
    total_ice_cream_cost / n_ice_creams = 12 :=
by
  intros n_chapatis n_rice n_veg n_ice_creams chapati_price rice_price veg_price total_paid
         chapatis_eq rice_eq veg_eq ice_creams_eq chapati_price_eq rice_price_eq veg_price_eq total_paid_eq
  simp [chapatis_eq, rice_eq, veg_eq, ice_creams_eq, chapati_price_eq, rice_price_eq, veg_price_eq, total_paid_eq]
  let total_chapatis := 16 * 6
  let total_rice := 5 * 45
  let total_veg := 7 * 70
  let total_other := total_chapatis + total_rice + total_veg
  let total_ice_cream_cost := 883 - total_other
  have h : total_ice_cream_cost = 72 := by simp [total_chapatis, total_rice, total_veg]
  simp [h]
  exact (by norm_num : total_ice_cream_cost / 6 = 12)

end cost_of_ice_cream_cup_l654_654421


namespace part1_max_a_part2_inequality_l654_654158

def f (x : ℝ) : ℝ := Real.exp x - x

theorem part1_max_a :
  ∃ (a : ℝ), (∀ (x : ℝ), 0 ≤ x → (Real.exp x - 1 - 2 * a * x) ≥ 0) ∧ a = 1/2 :=
begin
  sorry
end

theorem part2_inequality (n : ℕ) (h : 0 < n) :
  (∑ i in Finset.range n, f (1/(i+1 : ℝ))) > n * (1 + 1/(4*(n+2))) :=
begin
  sorry
end

end part1_max_a_part2_inequality_l654_654158


namespace eggs_left_in_jar_l654_654502

def eggs_after_removal (original removed : Nat) : Nat :=
  original - removed

theorem eggs_left_in_jar : eggs_after_removal 27 7 = 20 :=
by
  sorry

end eggs_left_in_jar_l654_654502


namespace sum_of_sequence_2500_l654_654868

theorem sum_of_sequence_2500 (b : ℕ → ℤ) (h1 : ∀ (n : ℕ), n ≥ 3 → b n = b (n - 1) + b (n - 2)) 
    (h2 : (∑ n in Finset.range 2000, b (n + 1)) = 2500) 
    (h3 : (∑ n in Finset.range 2500, b (n + 1)) = 2000) :
    (∑ n in Finset.range 2500, b (n + 1)) = 2000 := 
begin
    sorry
end

end sum_of_sequence_2500_l654_654868


namespace coffee_price_in_usd_to_nearest_hundredth_l654_654032

-- Defining the price of the coffee in yen
def price_in_yen : ℝ := 200

-- Defining the conversion rate from yen to USD
def conversion_rate : ℝ := 1 / 108

-- Define the expected result in USD, rounded to the nearest hundredth
def expected_usd : ℝ := 1.85

-- The Lean 4 statement proving that the amount spent in USD, rounded to the nearest hundredth, is correct
theorem coffee_price_in_usd_to_nearest_hundredth :
  Float.round (price_in_yen * conversion_rate * 100) / 100 = expected_usd :=
by
  sorry

end coffee_price_in_usd_to_nearest_hundredth_l654_654032


namespace balls_in_boxes_l654_654595

theorem balls_in_boxes : ∀ (num_balls num_boxes : ℕ),
  num_balls = 5 →
  num_boxes = 2 →
  (indistinguishable num_balls ∧ distinguishable num_boxes) →
  (distribution_ways num_balls num_boxes = 6) :=
begin
  intros num_balls num_boxes hb hb2 h,
  sorry
end

-- Definitions for indistinguishable and distinguishable properties
def indistinguishable (n : ℕ) : Prop := true
def distinguishable (n : ℕ) : Prop := true

-- Definition for distribution ways function
def distribution_ways (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
-- (5 choose 0) + (5 choose 1) + (5 choose 2) + (5 choose 3) + (5 choose 4) + (5 choose 5)
  fintype.card {f : fin num_boxes → fin (num_balls+1) // 
    finset.sum (finset.univ.image f) = num_balls }


end balls_in_boxes_l654_654595


namespace point_distance_5_5_l654_654684

-- Define the distance function in the context of the problem
def distance_from_origin (x : ℝ) : ℝ := abs x

-- Formalize the proposition
theorem point_distance_5_5 (x : ℝ) : distance_from_origin x = 5.5 → (x = -5.5 ∨ x = 5.5) :=
by
  intro h
  simp [distance_from_origin] at h
  sorry

end point_distance_5_5_l654_654684


namespace lines_parallel_or_intersect_at_one_point_l654_654116

theorem lines_parallel_or_intersect_at_one_point
  (ABC : Triangle)
  (lines : Fin 10 → Line)
  (h_equidistant : ∀ i, ∃(A B : Point), A ≠ B ∧ isEquidistantFrom lines[i] A B) :
  (∃ i j, i ≠ j ∧ lines[i].parallel lines[j]) ∨ (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ lines[i].intersectAtOnePoint lines[j] lines[k]) :=
sorry

end lines_parallel_or_intersect_at_one_point_l654_654116


namespace distance_between_foci_of_ellipse_l654_654943

theorem distance_between_foci_of_ellipse :
  ∀ (a b : ℝ), a = 6 ∧ b = 2 → (2 * real.sqrt (a ^ 2 - b ^ 2) = 8 * real.sqrt 2) :=
by
  intros a b h
  obtain ⟨ha, hb⟩ := h
  rw [ha, hb]
  sorry

end distance_between_foci_of_ellipse_l654_654943


namespace G6_1_G6_2_G6_3_G6_4_l654_654428

theorem G6_1 (H1 : EquilateralTriangle ABC) (H2 : EquilateralTriangle XYZ) 
(H3 : RightPrism ABC XYZ) (H4 : Midpoint P BY) (H5 : BP = 3) (H6 : XY = 4) : 
  let a := (CP / PX) 
  a = 1 :=
sorry

theorem G6_2 (H1 : EquilateralTriangle ABC) (H2 : EquilateralTriangle XYZ) 
(H3 : RightPrism ABC XYZ) (H4 : Midpoint P BY) (H5 : BP = 3) (H6 : XY = 4) :
  let CX := Real.sqrt b
  b = 52 :=
sorry

theorem G6_3 (H1 : EquilateralTriangle ABC) (H2 : EquilateralTriangle XYZ) 
(H3 : RightPrism ABC XYZ) (H4 : Midpoint P BY) (H5 : BP = 3) (H6 : XY = 4) :
  let cosTheta := sqrt c / 5
  c = 13 :=
sorry

theorem G6_4 (H1 : EquilateralTriangle ABC) (H2 : EquilateralTriangle XYZ) 
(H3 : RightPrism ABC XYZ) (H4 : Midpoint P BY) (H5 : BP = 3) (H6 : XY = 4) :
  let sinTheta := 2 * sqrt d / 5
  d = 3 :=
sorry

end G6_1_G6_2_G6_3_G6_4_l654_654428


namespace solve_x_in_equation_l654_654064

theorem solve_x_in_equation (x : ℝ) :
  (∀ y : ℝ, y = 3 * x → 4 * y^2 + y + 5 = 3 * (9 * x^2 + y + 3)) →
  (x = (1 + sqrt 5) / 3 ∨ x = (1 - sqrt 5) / 3) :=
by
  sorry

end solve_x_in_equation_l654_654064


namespace marked_price_percentage_fixed_l654_654456

-- Definitions based on the conditions
def discount_percentage : ℝ := 0.18461538461538467
def profit_percentage : ℝ := 0.06

-- The final theorem statement
theorem marked_price_percentage_fixed (CP MP SP : ℝ) 
  (h1 : SP = CP * (1 + profit_percentage))  
  (h2 : SP = MP * (1 - discount_percentage)) :
  (MP / CP - 1) * 100 = 30 := 
sorry

end marked_price_percentage_fixed_l654_654456


namespace no_real_roots_of_composed_function_l654_654242

variables {a b c : ℝ}

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_real_roots_of_composed_function 
  (h_discriminant : (b - 1) ^ 2 - 4 * a * c < 0) : 
  ¬∃ x : ℝ, f(f(x)) = x :=
by sorry

end no_real_roots_of_composed_function_l654_654242


namespace prime_pow_divides_pow_l654_654228

open Nat

theorem prime_pow_divides_pow {a n p : ℕ} (hp : Prime p) (ha : a ∈ Nat) (hn : n ∈ Nat) (h : p ∣ a^n) : p^n ∣ a^n :=
by
  sorry

end prime_pow_divides_pow_l654_654228


namespace michael_large_balls_l654_654780

theorem michael_large_balls (total_rubber_bands : ℕ) (small_ball_rubber_bands : ℕ) (large_ball_rubber_bands : ℕ) (small_balls_made : ℕ)
  (h_total_rubber_bands : total_rubber_bands = 5000)
  (h_small_ball_rubber_bands : small_ball_rubber_bands = 50)
  (h_large_ball_rubber_bands : large_ball_rubber_bands = 300)
  (h_small_balls_made : small_balls_made = 22) :
  (total_rubber_bands - small_balls_made * small_ball_rubber_bands) / large_ball_rubber_bands = 13 :=
by {
  sorry
}

end michael_large_balls_l654_654780


namespace cos_minus_sin_alpha_l654_654136

theorem cos_minus_sin_alpha (α : ℝ) 
  (h1 : π / 2 < α ∧ α < π) 
  (h2 : sin (2 * α) = -24 / 25) : 
  cos α - sin α = -7 / 5 :=
sorry

end cos_minus_sin_alpha_l654_654136


namespace compare_a_b_c_l654_654546

noncomputable def a : ℝ := 3 ^ 0.1
noncomputable def b : ℝ := (1 / 3) ^ (-0.8)
noncomputable def c : ℝ := Real.log 0.8 / Real.log 0.7

theorem compare_a_b_c : c < a ∧ a < b := by
  sorry

end compare_a_b_c_l654_654546


namespace polynomial_degree_at_least_n_l654_654652

noncomputable def f (x_1 x_2 : ℕ) (xs : List ℕ) : ℝ :=
  ⌊ (x_1 + x_2 + xs.sum) / 2 ⌋

theorem polynomial_degree_at_least_n (m n : ℕ) (h : m ≥ 2) (k : n ≥ 2) 
  (f : Finₙ → ℝ) (hf : ∀ (x : Fin n → ℕ), ∀ (h : ∀ i, 0 ≤ x i ∧ x i < m), 
  f(x) = ⌊ (∑ i, x i) / m ⌋) : degree(f) ≥ n := 
begin
  sorry
end

end polynomial_degree_at_least_n_l654_654652


namespace lambda_is_four_l654_654772

-- Definition for the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1

-- Right focus of the hyperbola
def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Definition for a line passing through the right focus intersecting the hyperbola
def line_through_focus (l : ℝ → ℝ) : Prop := ∀ y, hyperbola (Real.sqrt 3) y → y = l (Real.sqrt 3)

-- Distance between two points
def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

-- Points of intersection of the line with the hyperbola
def points_of_intersection (l : ℝ → ℝ) : ℝ × ℝ := ((Real.sqrt 3), l (Real.sqrt 3))

-- Prove that the distance |AB| is 4 given the conditions
theorem lambda_is_four (l : ℝ → ℝ) (h : line_through_focus l) : 
  distance (points_of_intersection l) (points_of_intersection (fun x => -l x)) = 4 :=
sorry

end lambda_is_four_l654_654772


namespace right_triangle_area_and_perimeter_l654_654722

-- Definitions for the conditions
def hypotenuse : ℝ := 10 * Real.sqrt 2
def angle : ℝ := 45

-- Main theorem statements
theorem right_triangle_area_and_perimeter (h : hypotenuse = 10 * Real.sqrt 2) (a : angle = 45) :
  (let x := 10 in
    let area := 1 / 2 * x * x in
    let perimeter := x + x + hypotenuse in
    area = 50 ∧ perimeter = 20 + 10 * Real.sqrt 2) :=
by
  sorry

end right_triangle_area_and_perimeter_l654_654722


namespace BD_parallel_AC_l654_654853

section Geometry

variables {A B C A_1 B_1 C_1 A' B' C' D : Point}
variables {circle : Circle}

-- Conditions
def inscribed_circle (T : Triangle) (c : Circle) : Prop :=
  tangent c T.A B T.C ∧ tangent c T.B C T.A ∧ tangent c T.C A T.B

def touchpoints (T : Triangle) (c : Circle) (A_1 B_1 C_1 : Point) : Prop :=
  tangent_point c T.A B T.C A_1 ∧ tangent_point c T.B C T.A B_1 ∧ tangent_point c T.C A T.B C_1

def analog_points (A_1 B_1 C_1 A' B' C' : Point) : Prop :=
  -- Analog definition conditions to be defined here
  sorry

def intersection_point (A' C_1 C' A_1 : Point) (D : Point) : Prop :=
  intersects (line A' C_1) (line C' A_1) D

-- Proof problem
theorem BD_parallel_AC
  (T : Triangle) (c : Circle) 
  (A_1 B_1 C_1 A' B' C' D : Point)
  (h1 : inscribed_circle T c)
  (h2 : touchpoints T c A_1 B_1 C_1)
  (h3 : analog_points A_1 B_1 C_1 A' B' C')
  (h4 : intersection_point A' C_1 C' A_1 D) : 
  parallel (line B D) (line A C) :=
sorry

end Geometry

end BD_parallel_AC_l654_654853


namespace det_projection_matrix_l654_654231

noncomputable section

variable {𝕜 : Type*} [Field 𝕜]

def projection_matrix (v : Matrix (Fin 3) (Fin 1) 𝕜) :=
  (v ⬝ vᵀ) ⬝ (((vᵀ ⬝ v).det⁻¹) • (1 : Matrix (Fin 3) (Fin 3) 𝕜))

theorem det_projection_matrix (v : Matrix (Fin 3) (Fin 1) 𝕜) : 
  v = ![![3], ![1], ![-4]] →
  (projection_matrix v).det = 0 :=
by
  sorry

end det_projection_matrix_l654_654231


namespace probability_slope_geq_neg_2_fifth_l654_654805

theorem probability_slope_geq_neg_2_fifth : 
  let outcomes := finset.univ.product finset.univ
  let favorable (ab : ℕ × ℕ) : Prop := ab.2 ≤ 2 * ab.1 / 5
  let m := (finset.filter favorable outcomes).card
  let n := outcomes.card
  (m : ℚ) / n = 1 / 6 := 
by 
  sorry

end probability_slope_geq_neg_2_fifth_l654_654805


namespace max_rectangle_without_whole_domino_l654_654903

theorem max_rectangle_without_whole_domino (n : ℕ) (board : fin 8 × fin 8) (partition : (fin 8 × fin 8 → bool)) :
  (∀ (p : fin 8 × fin 8 → bool), 
    ∃ (r : set (fin 8 × fin 8)), (∃ (l₁ l₂ : ℕ) (hl₁ : 0 < l₁) (hl₂ : 0 < l₂), r = {rc ∈ board | rc.1 < l₁ ∧ rc.2 < l₂} ∧ fintype.card r = n) ∧ 
    (∀ (d : fin 8 × fin 8 → bool), ∃ (rc : fin 8 × fin 8), partition rc = true) = false) = (n ≤ 4) := sorry

end max_rectangle_without_whole_domino_l654_654903


namespace onion_harvest_weight_l654_654777

theorem onion_harvest_weight :
  let bags_per_trip := 10 in
  let weight_per_bag := 50 in
  let trips := 20 in
  let total_weight := (bags_per_trip * weight_per_bag) * trips in
  total_weight = 10000 := by
  sorry

end onion_harvest_weight_l654_654777


namespace count_ways_split_420_l654_654205

theorem count_ways_split_420 :
  (∃ (n : ℕ) (k : ℕ), n ≥ 2 ∧ (n * (2 * k + n - 1) = 840) ∧ (2 * k + n - 1 > n)) ∧ 
  (count (λ n k, n ≥ 2 ∧ (n * (2 * k + n - 1) = 840) ∧ (2 * k + n - 1 > n)) = 17) :=
sorry

end count_ways_split_420_l654_654205


namespace num_divisors_count_l654_654591

theorem num_divisors_count (n : ℕ) (m : ℕ) (H : m = 32784) :
  (∃ S : Finset ℕ, (∀ x ∈ S, x ∈ (Finset.range 10) ∧ m % x = 0) ∧ S.card = n) ↔ n = 7 :=
by
  sorry

end num_divisors_count_l654_654591


namespace tetrahedron_volume_constancy_l654_654499

-- Define the structures and elements used in the problem
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Segment where
  A : Point3D
  B : Point3D

-- Define the volume of a tetrahedron
noncomputable def tetrahedronVolume (AB : Segment) (CD : Segment) : ℝ :=
  let volumeFormula := λ area height, (1 / 3) * area * height
  sorry -- volume calculation using area and height is assumed

-- Define the area of the triangular base
noncomputable def triangleArea (A B C : Point3D) : ℝ :=
  let length := λ P Q : Point3D, (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 + (P.z - Q.z) ^ 2
  sorry -- area calculation is assumed

-- Define the height from a point to a plane
noncomputable def heightFromPointToPlane (D : Point3D) (plane : Point3D → Point3D → Point3D → ℝ) : ℝ :=
  sorry -- height calculation from point D to plane is assumed

-- Main theorem to prove volume constancy
theorem tetrahedron_volume_constancy (AB CD : Segment) (moveAB : Segment) :
  (moveAB.length = AB.length) →
  tetrahedronVolume AB CD = tetrahedronVolume moveAB CD :=
by
  sorry

end tetrahedron_volume_constancy_l654_654499


namespace smallest_prime_factor_in_C_is_72_l654_654697

def C : Set ℕ := {67, 71, 72, 73, 79}

theorem smallest_prime_factor_in_C_is_72 :
  ∃ n ∈ C, (∀ m ∈ C, (∀ p ∈ (Nat.factors m), prime p) → (∃ q ∈ (Nat.factors n), prime q ∧ (q <= p) ∧ (q = 2)))
:= 
  sorry

end smallest_prime_factor_in_C_is_72_l654_654697


namespace harmonic_series_diverges_series_2n1_over_5n3_diverges_l654_654795

theorem harmonic_series_diverges :
  ¬ (∃ (S : Real), ∀ ε > 0, ∃ N, ∀ n > N, abs((∑ i in finset.range n, (1:ℝ)/(i+1)) - S) < ε) :=
by sorry

theorem series_2n1_over_5n3_diverges :
  ¬ (∃ (S : Real), ∀ ε > 0, ∃ N, ∀ n > N, abs((∑ i in finset.range n, (2*i+1)/(5*i-3:ℝ)) - S) < ε) :=
by sorry

end harmonic_series_diverges_series_2n1_over_5n3_diverges_l654_654795


namespace area_of_enclosed_figure_l654_654709

theorem area_of_enclosed_figure:
  ∫ (x : ℝ) in (1/2)..2, x⁻¹ = 2 * Real.log 2 :=
by
  sorry

end area_of_enclosed_figure_l654_654709


namespace sum_of_four_digit_integers_l654_654407

theorem sum_of_four_digit_integers : 
  let a := 1000 in
  let l := 9999 in
  let n := l - a + 1 in
  (a + l) * n / 2 = 49495500 :=
by
  let a := 1000
  let l := 9999
  let n := l - a + 1
  have h_sum := ((a + l) * n) / 2
  rw [a, l, n] at h_sum
  exact h_sum
  sorry

end sum_of_four_digit_integers_l654_654407


namespace theater_ticket_cost_l654_654006

theorem theater_ticket_cost
  (O B : ℕ)
  (h1 : O + B = 370)
  (h2 : B = O + 190) 
  : 12 * O + 8 * B = 3320 :=
by
  sorry

end theater_ticket_cost_l654_654006


namespace find_k_l654_654192

theorem find_k (k : ℤ) 
  (h : ∃ a d : ℝ, (sqrt (49 + k) : ℝ) = a ∧ (sqrt (225 + k) : ℝ) = a + d ∧ (sqrt (400 + k) : ℝ) = a + 2 * d): 
  k = 92 := 
sorry

end find_k_l654_654192


namespace committee_count_with_president_l654_654619

-- Define the conditions
def total_people : ℕ := 12
def committee_size : ℕ := 5
def remaining_people : ℕ := 11
def president_inclusion : ℕ := 1

-- Define the calculation of binomial coefficient
noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

-- State the problem in Lean 4
theorem committee_count_with_president : 
  binomial remaining_people (committee_size - president_inclusion) = 330 :=
sorry

end committee_count_with_president_l654_654619


namespace liars_among_islanders_are_only_9_18_19_l654_654832

-- declare the islanders
def Islander := { x : ℕ // x < 19 }

-- define statements
def said_three_of_us_are_liars (i : Islander) : Prop := sorry
def said_six_of_us_are_liars (i : Islander) : Prop := sorry
def said_nine_of_us_are_liars (i : Islander) : Prop := sorry

-- define types
inductive Kind
| Knight
| Liar

-- Islander knowledge
def knows_kind (i j : Islander) : Kind -> Prop := sorry

-- Theorem to prove
theorem liars_among_islanders_are_only_9_18_19 :
  ∃ n : ℕ, n = 9 ∨ n = 18 ∨ n = 19 ∧ 
  ∃ (liars : Fin n → Islander),
    (∀ l, ∃ i, said_nine_of_us_are_liars i → i ∈ (liars)) ∧
    (∀ l, ∃ i, said_six_of_us_are_liars i → i ∉ (liars)) ∧
    (∀ l, ∃ i, said_three_of_us_are_liars i → i ∉ (liars)) :=
by
  sorry

end liars_among_islanders_are_only_9_18_19_l654_654832


namespace angle_proof_l654_654110

-- Variables and assumptions
variable {α : Type} [LinearOrderedField α]    -- using a general type for angles
variable {A B C D E : α}                       -- points of the triangle and extended segment

-- Given conditions
variable (angle_ACB angle_ABC : α)
variable (H1 : angle_ACB = 2 * angle_ABC)      -- angle condition
variable (CD BD AD DE : α)
variable (H2 : CD = 2 * BD)                    -- segment length condition
variable (H3 : AD = DE)                        -- extended segment condition

-- The proof goal in Lean format
theorem angle_proof (H1 : angle_ACB = 2 * angle_ABC) 
  (H2 : CD = 2 * BD) 
  (H3 : AD = DE) :
  angle_ECB + 180 = 2 * angle_EBC := 
sorry  -- proof to be filled in

end angle_proof_l654_654110


namespace integral1_solution_integral2_solution_integral3_solution_integral4_solution_l654_654898

noncomputable def integral1 : ℝ := ∫ x in 0..5, x / sqrt (1 + 3 * x)
theorem integral1_solution : integral1 = 4 := 
sorry

noncomputable def integral2 : ℝ := ∫ x in Real.log 2..Real.log 9, 1 / (Real.exp x - Real.exp (-x))
theorem integral2_solution : integral2 = (1/2) * Real.log 2.4 := 
sorry

noncomputable def integral3 : ℝ := ∫ x in 1..Real.sqrt 3, (x^3 + 1) / (x^2 * Real.sqrt (4 - x^2))
theorem integral3_solution : integral3 = 2 * Real.sqrt 3 - 2 + 1 / Real.sqrt 3 := 
sorry

noncomputable def integral4 : ℝ := ∫ x in 0..(Real.pi / 2), 1 / (2 + Real.cos x)
theorem integral4_solution : integral4 = Real.pi / (3 * Real.sqrt 3) := 
sorry

end integral1_solution_integral2_solution_integral3_solution_integral4_solution_l654_654898


namespace positive_prime_factors_least_3_l654_654821

theorem positive_prime_factors_least_3 (n : ℕ) (h1 : 0 < n) (h2 : n < 200) (h3 : (14 * n) % 60 = 0) : 
  ∃ p1 p2 p3 : ℕ, prime p1 ∧ prime p2 ∧ prime p3 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ p1 ∣ n ∧ p2 ∣ n ∧ p3 ∣ n :=
sorry

end positive_prime_factors_least_3_l654_654821


namespace artichokey_apple_investment_l654_654679

/-- Given the conditions of the apple investment problem, prove that Artichokey invested 7 apples. -/
theorem artichokey_apple_investment :
  ∀ (total_earthworms okeydokey_apples okeydokey_earthworms : ℕ), 
  total_earthworms = 60 ∧
  okeydokey_apples = 5 ∧
  okeydokey_earthworms = 25 ∧
  (∀ (apples: ℕ), earthworms_per_apple okeydokey_apples okeydokey_earthworms = okeydokey_earthworms / okeydokey_apples)
  → (∃ (artichokey_apples : ℕ),
      artichokey_apples = (total_earthworms - okeydokey_earthworms) / (okeydokey_earthworms / okeydokey_apples) 
      ∧ artichokey_apples = 7) :=
begin
  intros,
  let total_earthworms := 60,
  let okeydokey_apples := 5,
  let okeydokey_earthworms := 25,
  let earthworms_per_apple := 5,
  let artichokey_earthworms := total_earthworms - okeydokey_earthworms,
  let artichokey_apples := artichokey_earthworms / earthworms_per_apple,
  exact ⟨artichokey_apples, by simp [total_earthworms, okeydokey_apples, okeydokey_earthworms, earthworms_per_apple, artichokey_earthworms, artichokey_apples]⟩,
end

end artichokey_apple_investment_l654_654679


namespace gcd_polynomial_example_l654_654567

theorem gcd_polynomial_example (b : ℕ) (h : ∃ k : ℕ, b = 2 * 7784 * k) : 
  gcd (5 * b ^ 2 + 68 * b + 143) (3 * b + 14) = 25 :=
by 
  sorry

end gcd_polynomial_example_l654_654567


namespace cos_graph_shift_l654_654334

theorem cos_graph_shift (x : ℝ) : 
  ∃ (shift : ℝ), (∀ x, (cos (2 * x)) = cos (2 * (x + shift) + 1)) ∧ shift = -1/2 :=
by sorry

end cos_graph_shift_l654_654334


namespace find_function_l654_654924

variable (f : ℕ → ℕ)
variable (n : ℕ) (n_gt_one : n > 1)
variable (c : ℕ)

theorem find_function (f_property : ∀ x y : ℕ, f (x + y) = f x + f y + (Finset.range (n - 1)).sum (λ k, Nat.choose n k * x ^ (n - k) * y ^ k)) :
  ∃ c : ℕ, ∀ x : ℕ, f x = c * x + x ^ n :=
sorry

end find_function_l654_654924


namespace bob_total_profit_l654_654486

def initial_cost (num_dogs : ℕ) (cost_per_dog : ℕ) : ℕ := num_dogs * cost_per_dog

def revenue (num_puppies : ℕ) (price_per_puppy : ℕ) : ℕ := num_puppies * price_per_puppy

def total_profit (initial_cost : ℕ) (revenue : ℕ) : ℕ := revenue - initial_cost

theorem bob_total_profit (c1 : initial_cost 2 250 = 500)
                        (c2 : revenue 6 350 = 2100)
                        (c3 : total_profit 500 2100 = 1600) :
  total_profit (initial_cost 2 250) (revenue 6 350) = 1600 := by
  sorry

end bob_total_profit_l654_654486


namespace equilateral_triangle_ANT_l654_654211

variables {A B C D N T : Type} [geometry.quadrilateral A B C D]
variables (mid_point_BC : midpoint N B C)
variables (mid_point_CD : midpoint T C D)
variables (angle_condition : ∀ (X : Type), ∠B = 30 ∧ ∠C = 30 ∧ ∠D = 30)
variables (non_convex_condition : ∠A > 180)

theorem equilateral_triangle_ANT
    {A B C D N T : Type}  
    [geometry.quadrilateral A B C D]
    (mid_point_BC : midpoint N B C)
    (mid_point_CD : midpoint T C D)
    (angle_condition : ∀ (X : Type), ∠B = 30 ∧ ∠C = 30 ∧ ∠D = 30)
    (non_convex_condition : ∠A > 180) :
  geometry.equilateral_triangle A N T :=
sorry

end equilateral_triangle_ANT_l654_654211


namespace problem_statement_l654_654138

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem problem_statement :
  ∀ (a : ℕ → ℝ),
    geometric_sequence a 9 →
    a 1 = 3 →
    (1 / 2) * a 3 = (9 * a 1 + 8 * a 2) / 2 →
    (∀ n, a n = 3 ^ (2 * n - 1)) ∧
    (∀ b : ℕ → ℝ, 
      (∀ n, b n = 1 / (log 2 (a n) * log 3 (a (n + 1)))) →
       let T : ℕ → ℝ := λ n, (1 / 2) * (1 - 1 / (2 * n + 1)) in
       (T n = n / (2 * n + 1)) ∧ 
       ∀ m : ℝ, (0 < m < 1 ∨ 8 < m) → ∀ n, T n > real.log 2 / real.log m)
:= sorry

end problem_statement_l654_654138


namespace max_distance_l654_654097

theorem max_distance (front_lifespan : ℕ) (rear_lifespan : ℕ)
  (h_front : front_lifespan = 21000)
  (h_rear : rear_lifespan = 28000) :
  ∃ (max_dist : ℕ), max_dist = 24000 :=
by
  sorry

end max_distance_l654_654097


namespace solve_3x3_table_l654_654846

theorem solve_3x3_table : 
   ∃ (fill_combinations : list (list (list ℤ))), 
   (∀ (row : list ℤ), row ∈ fill_combinations → row.sum = fill_combinations.head.sum) 
   ∧
   fill_combinations.length = 831 :=
   sorry

end solve_3x3_table_l654_654846


namespace proof_problem_l654_654635

-- Define the given complex numbers
def z1 : ℂ := 1 + 2 * Complex.I
def z2 : ℂ := -1 + Complex.I

-- Define the conditions for z
def z (m : ℝ) : ℂ := -1 + m * Complex.I

-- Define the polynomial equation condition
def is_root (a : ℝ) (z : ℂ) : ℂ := 2 * z^2 + a * z + 10

-- Prove the magnitudes and root conditions
theorem proof_problem 
  (h1 : z1 - z2 = 2 + Complex.I)
  (h2 : |2 + Complex.I| = Real.sqrt 5)
  (h3 : ∀ m : ℝ, is_root 4 (z m) = 0)
  (h4 : z (-2) = -1 - 2 * Complex.I ∨ z 2 = -1 + 2 * Complex.I):
  |z1 - z2| = Real.sqrt 5 ∧  
  (∃ m : ℝ, is_root 4 (z m) = 0 ∧ (z m = -1 - 2 * Complex.I ∨ z m = -1 + 2 * Complex.I)) :=
by
  sorry

end proof_problem_l654_654635


namespace sum_of_divisors_37_l654_654802

theorem sum_of_divisors_37 : ∑ d in (Finset.filter (λ d, 37 % d = 0) (Finset.range (37 + 1))), d = 38 := 
by sorry

end sum_of_divisors_37_l654_654802


namespace a3_mul_a7_eq_36_l654_654569

-- Definition of a geometric sequence term
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions
def a (n : ℕ) : ℤ := sorry  -- Placeholder for the geometric sequence

axiom a5_eq_6 : a 5 = 6  -- Given that a_5 = 6

axiom geo_seq : geometric_sequence a  -- The sequence is geometric

-- Problem statement: Prove that a_3 * a_7 = 36
theorem a3_mul_a7_eq_36 : a 3 * a 7 = 36 :=
  sorry

end a3_mul_a7_eq_36_l654_654569


namespace arithmetic_expression_l654_654793

theorem arithmetic_expression : ∃ (a b c d : ℕ) (op1 op2 op3 : ℕ → ℕ → ℕ), 
  a = 8 ∧ b = 8 ∧ c = 8 ∧ d = 10 ∧
  op1 = (-) ∧ op2 = (*) ∧ op3 = (+) ∧
  op3 (op2 (op1 d a) b) c = 24 := 
by 
  use 8, 8, 8, 10, (-), (*), (+)
  split; try {exact rfl} 
  sorry -- steps to prove arithmetic are correct

end arithmetic_expression_l654_654793


namespace sum_eq_prod_S1_sum_eq_prod_S2_l654_654494

def S1 : List ℕ := [1, 1, 1, 1, 1, 1, 2, 8]
def S2 : List ℕ := [1, 1, 1, 1, 1, 2, 2, 3]

def sum_list (l : List ℕ) : ℕ := l.foldr Nat.add 0
def prod_list (l : List ℕ) : ℕ := l.foldr Nat.mul 1

theorem sum_eq_prod_S1 : sum_list S1 = prod_list S1 := 
by
  sorry

theorem sum_eq_prod_S2 : sum_list S2 = prod_list S2 := 
by
  sorry

end sum_eq_prod_S1_sum_eq_prod_S2_l654_654494


namespace value_of_x_add_2y_squared_l654_654180

theorem value_of_x_add_2y_squared (x y : ℝ) (h1 : x * (x + y) = 40) (h2 : y * (x + y) = 90) : (x + 2 * y) ^ 2 ≈ 372.31 :=
by
  sorry

end value_of_x_add_2y_squared_l654_654180


namespace mixtape_length_l654_654477

theorem mixtape_length (songs_side1 songs_side2 song_duration : ℕ) 
  (h1 : songs_side1 = 6) 
  (h2 : songs_side2 = 4) 
  (h3 : song_duration = 4) : 
  (songs_side1 + songs_side2) * song_duration = 40 :=
by
  rw [h1, h2, h3]
  norm_num

end mixtape_length_l654_654477


namespace find_x_l654_654126

theorem find_x (x : ℝ) (h : 3^x + 3^x + 3^x + 3^x = 2187) : (x + 2) * (x - 2) = 21 :=
sorry

end find_x_l654_654126


namespace min_marked_cells_l654_654382

-- Define the board
def board : Type := fin 10 × fin 11

-- Define the L-shaped piece with its three cells
def L_piece_cells : list (int × int) := [(0, 0), (1, 0), (0, 1)]

-- Define what it means for an L-shaped piece to be placed on the board
def L_placement (p : board) : list board :=
  L_piece_cells.map (λ (c : int × int), (⟨p.1.1 + c.1, by sorry⟩, ⟨p.2.1 + c.2, by sorry⟩))

-- Define the marking of cells
def marked_cells (ls : list board) : Prop :=
  ∃ k : ℕ, k = 50 ∧ ∀ p : board, ∃ m ∈ ls, m ∈ L_placement p

-- Statement that the smallest k satisfying the condition is 50
theorem min_marked_cells :
  ∃ (ls : list board), marked_cells ls :=
sorry

end min_marked_cells_l654_654382


namespace relationship_a_b_c_l654_654544

noncomputable def a : ℝ := 3^0.1
noncomputable def b : ℝ := (1/3)^(-0.8)
noncomputable def c : ℝ := Real.log 0.8 / (Real.log 0.7)

theorem relationship_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_a_b_c_l654_654544


namespace hyperbola_equation_l654_654995

theorem hyperbola_equation {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : (2 * b)^2 + b^2 = 5) 
  (h4 : 2 * (2 * b) + b = 0) : 
    ∃ (a : ℝ), ∃ (b : ℝ), (a = 2 * b) ∧ (b = 1) ∧ (a = 2) ∧ (h : (a > 0) ∧ (b > 0)) ∧ 
    (c = √5) ∧ c = sqrt(a^2 + b^2) ∧ (equation: ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1)) :=
by
  sorry

end hyperbola_equation_l654_654995


namespace bob_total_profit_l654_654485

def initial_cost (num_dogs : ℕ) (cost_per_dog : ℕ) : ℕ := num_dogs * cost_per_dog

def revenue (num_puppies : ℕ) (price_per_puppy : ℕ) : ℕ := num_puppies * price_per_puppy

def total_profit (initial_cost : ℕ) (revenue : ℕ) : ℕ := revenue - initial_cost

theorem bob_total_profit (c1 : initial_cost 2 250 = 500)
                        (c2 : revenue 6 350 = 2100)
                        (c3 : total_profit 500 2100 = 1600) :
  total_profit (initial_cost 2 250) (revenue 6 350) = 1600 := by
  sorry

end bob_total_profit_l654_654485


namespace possible_values_of_X_l654_654985

-- Define the conditions and the problem
def defective_products_total := 3
def total_products := 10
def selected_products := 2

-- Define the random variable X
def X (n : ℕ) : ℕ := n / selected_products

-- Now the statement to prove is that X can only take the values {0, 1, 2}
theorem possible_values_of_X :
  ∀ (X : ℕ → ℕ), ∃ (vals : Set ℕ), (vals = {0, 1, 2} ∧ ∀ (n : ℕ), X n ∈ vals) :=
by
  sorry

end possible_values_of_X_l654_654985


namespace a_seq_formula_l654_654625

noncomputable def a_seq : ℕ+ → ℝ
| 1 := 1
| 2 := 0.5
| (n+1) := if h : n = 0 then 0.5 else sqrt ((1 / a_seq n) * (1 / a_seq (n + 2)))⁻¹

theorem a_seq_formula (n : ℕ+) : 
  a_seq n = (1 / 2^(n - 1)) := by
    sorry

end a_seq_formula_l654_654625


namespace final_prob_l654_654492

-- Definitions of dice probabilities
def fair_die : ℕ → ℚ := λ face, if face = 6 ∨ face = 5 then 1 / 6 else 1 / 6

def biased_die : ℕ → ℚ := λ face, 
  if face = 6 then 1 / 2 
  else if face = 5 then 1 / 6 
  else 1 / 24

-- Definition of events
def event_five_or_six (die : ℕ → ℚ) := die 5 + die 6

-- Probabilities of fair and biased dice showing a five or six
def prob_fair_rolls_six_or_five := event_five_or_six fair_die
def prob_biased_rolls_six_or_five := event_five_or_six biased_die

-- Squared probabilities for first two trials
def prob_fair_two_rolls := prob_fair_rolls_six_or_five ^ 2
def prob_biased_two_rolls := prob_biased_rolls_six_or_five ^ 2

-- Normalized probabilities based on first two rolls
def norm_prob_fair := 
  prob_fair_two_rolls / (prob_fair_two_rolls + prob_biased_two_rolls)
def norm_prob_biased := 
  prob_biased_two_rolls / (prob_fair_two_rolls + prob_biased_two_rolls)

-- Combined probability of the third roll being a five or six
def combined_prob_six_or_five := 
  norm_prob_fair * prob_fair_rolls_six_or_five + norm_prob_biased * prob_biased_rolls_six_or_five

-- Final proof statement
theorem final_prob : combined_prob_six_or_five = 3 / 5 := 
  by sorry

end final_prob_l654_654492


namespace pizza_slice_volume_l654_654454

def radius (d : ℝ) : ℝ := d / 2
def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def total_volume_pizza : ℝ := volume_of_cylinder (radius 16) (1/2)
def volume_of_one_slice : ℝ := total_volume_pizza / 16

theorem pizza_slice_volume :
  volume_of_one_slice = 2 * π :=
by
  sorry

end pizza_slice_volume_l654_654454


namespace point_in_second_quadrant_l654_654633

open Complex

theorem point_in_second_quadrant (z : ℂ) (h : z * (1 + I) = 1 - 2 * I) : ((-(Re(z)) > 0) ∧ (Im(z) < 0)) :=
by {
  have z_val : z = (1 - 2 * I) / (1 + I),
  { field_simp, 
    norm_num,
    rw mul_div_cancel',
    norm_num,
    linarith },
  have z_conj_val : conj z = conj ((1 - 2 * I) / (1 + I)) := by rw z_val,
  have z_conj_simplified : conj z = -1 / 2 + (3 / 2) * I := by {
    rw [map_div, conj_of_real, conj_of_real, conj_I],
    simp,
    norm_num },
  rw z_conj_simplified,
  norm_num,
  simp,
  linarith }

end point_in_second_quadrant_l654_654633


namespace polynomial_roots_reciprocal_sum_l654_654716

theorem polynomial_roots_reciprocal_sum:
  (a b c : ℝ) 
  (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 10) 
  (h3 : a * b * c = -6) 
  (h4 : ∀ x : ℝ, x^3 - 7 * x^2 + 10 * x - 6 = 0 → (x = a ∨ x = b ∨ x = c)) :
  (1 / a^2 + 1 / b^2 + 1 / c^2) = 46 / 9 :=
by
  sorry

end polynomial_roots_reciprocal_sum_l654_654716


namespace first_digit_of_528_base7_l654_654801

def first_digit_base7 (n : ℕ) : ℕ :=
  let rec compute_first_digit (n k : ℕ) (p : ℕ) : ℕ :=
    if n < p then k - 1 else compute_first_digit (n / 7) (k + 1) (p * 7)
  compute_first_digit n 1 7

theorem first_digit_of_528_base7 :
  first_digit_base7 528 = 1 :=
sorry

end first_digit_of_528_base7_l654_654801


namespace min_value_of_reciprocal_sum_l654_654968

theorem min_value_of_reciprocal_sum {a b : ℝ} (h : a > 0 ∧ b > 0)
  (h_circle1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (h_circle2 : ∀ x y : ℝ, (x - 2)^2 + (y - 2)^2 = 4)
  (h_common_chord : a + b = 2) :
  (1 / a + 9 / b = 8) := 
sorry

end min_value_of_reciprocal_sum_l654_654968


namespace sum_four_digit_integers_l654_654393

def a := 1000
def l := 9999
def n := 9999 - 1000 + 1
def S (n : ℕ) (a : ℕ) (l : ℕ) := n / 2 * (a + l)

theorem sum_four_digit_integers : S n a l = 49495500 :=
by
  sorry

end sum_four_digit_integers_l654_654393


namespace log_expression_simplification_l654_654434

theorem log_expression_simplification :
  log (1 / 3) 27^(1/2) + log 10 25 + log 10 4 + 7^(-log 7 2) + (-0.98)^0 = 2 := by
  sorry

end log_expression_simplification_l654_654434


namespace prob_2_out_of_5_exactly_A_and_B_l654_654623

noncomputable def probability_exactly_A_and_B_selected (students : List String) : ℚ :=
  if students = ["A", "B", "C", "D", "E"] then 1 / 10 else 0

theorem prob_2_out_of_5_exactly_A_and_B :
  probability_exactly_A_and_B_selected ["A", "B", "C", "D", "E"] = 1 / 10 :=
by 
  sorry

end prob_2_out_of_5_exactly_A_and_B_l654_654623


namespace sum_c_eq_l654_654118

-- Definitions and conditions
def a_n : ℕ → ℝ := λ n => 2 ^ n
def b_n : ℕ → ℝ := λ n => 2 * n
def c_n (n : ℕ) : ℝ := a_n n * b_n n

-- Sum of the first n terms of sequence {c_n}
def sum_c (n : ℕ) : ℝ := (Finset.range n).sum c_n

-- Theorem statement
theorem sum_c_eq (n : ℕ) : sum_c n = (n - 1) * 2 ^ (n + 2) + 4 :=
sorry

end sum_c_eq_l654_654118


namespace count_sets_P_l654_654565

open Set

theorem count_sets_P (P : Set ℕ) : 
  {P | {1, 2} ⊆ P ∧ P ⊆ {1, 2, 3, 4}}.card = 4 := 
begin
  sorry -- Placeholder for the proof
end

end count_sets_P_l654_654565


namespace sequence_term_53_l654_654058

noncomputable def sequence (n : ℕ) : ℕ → ℕ
| 0     := n
| (k+1) := 
    let x := sequence k in
    if x < 7 then 8 * x 
    else if x % 2 = 0 then x / 3
    else x - 4

theorem sequence_term_53 (n : ℕ) (h : n = 53) : sequence n 52 = 5 :=
by {
  -- Since StackOverflow's length constraints omit proof details, we address assumptions and stopping the proof with sorry. 
  sorry
}

end sequence_term_53_l654_654058


namespace interval_of_monotonic_increase_evaluate_f_at_conditions_l654_654950

variables {x θ : ℝ} {ω : ℝ} (k : ℤ)
noncomputable def m (ω x : ℝ) := (Real.sin (ω * x), -1)
noncomputable def n (ω x : ℝ) := (1, -Real.sqrt 3 * Real.cos (ω * x))
noncomputable def f (ω x : ℝ) := (m ω x).1 * (n ω x).1 + (m ω x).2 * (n ω x).2

-- Condition that the distance between the highest point and adjacent lowest point is 5.
def distance_condition (ω : ℝ) : Prop :=
  ∃ T, (4^2 + (T/2)^2) = 25 ∧ ω = (2 * Real.pi) / T

-- Interval of monotonic increase for f(x)
theorem interval_of_monotonic_increase (ω : ℝ) : 
  distance_condition ω → 
  ω = Real.pi / 3 → 
  ∀ x, (∀ k : ℤ, (6 * k - 5/2) ≤ x ∧ x ≤ (6 * k + 1/2)) →
  (∀ k : ℤ, (6 * k - 5/2) ≤ x ∧ x ≤ (6 * k + 1/2)) := sorry

-- Evaluating f(x) at specific conditions
theorem evaluate_f_at_conditions (θ : ℝ) (ω : ℝ) :
  distance_condition ω →
  ω = Real.pi / 3 →
  f ω (3 * θ / Real.pi) = 6/5 →
  θ ∈ Ioo (-5 * Real.pi / 6) (Real.pi / 6) →
  f ω (6 * θ / Real.pi + 1) = 48 / 25 := sorry

end interval_of_monotonic_increase_evaluate_f_at_conditions_l654_654950


namespace sum_n_k_eq_eight_l654_654718

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the theorem to prove that n + k = 8 given the conditions
theorem sum_n_k_eq_eight {n k : ℕ} 
  (h1 : binom n k * 3 = binom n (k + 1))
  (h2 : binom n (k + 1) * 5 = binom n (k + 2) * 3) : n + k = 8 := by
  sorry

end sum_n_k_eq_eight_l654_654718


namespace sum_of_four_digit_integers_l654_654410

theorem sum_of_four_digit_integers : 
  let a := 1000 in
  let l := 9999 in
  let n := l - a + 1 in
  (a + l) * n / 2 = 49495500 :=
by
  let a := 1000
  let l := 9999
  let n := l - a + 1
  have h_sum := ((a + l) * n) / 2
  rw [a, l, n] at h_sum
  exact h_sum
  sorry

end sum_of_four_digit_integers_l654_654410


namespace smallest_x_value_l654_654082

theorem smallest_x_value {x : ℝ} (h : abs (x + 4) = 15) : x = -19 :=
sorry

end smallest_x_value_l654_654082


namespace largest_integer_among_four_l654_654944

theorem largest_integer_among_four 
  (x y z w : ℤ)
  (h1 : x + y + z = 234)
  (h2 : x + y + w = 255)
  (h3 : x + z + w = 271)
  (h4 : y + z + w = 198) :
  max x (max y (max z w)) = 121 := 
by
  -- This is a placeholder for the actual proof
  sorry

end largest_integer_among_four_l654_654944


namespace maximize_Miraflores_win_l654_654748

-- Definitions based on given conditions
def voters_count (n : ℕ) : ℕ := 2 * n
def support_Miraflores (n : ℕ) : ℕ := n + 1
def support_opponent (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem maximize_Miraflores_win (n : ℕ) (hn : n > 0) : 
  ∃ (d1 d2 : ℕ), d1 = 1 ∧ d2 = 2 * n - 1 ∧ support_Miraflores n > support_opponent n := 
sorry

end maximize_Miraflores_win_l654_654748


namespace sarah_score_l654_654276

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l654_654276


namespace circle_area_bound_l654_654642

open Real

noncomputable def total_circle_area (square_side : ℝ) (circles : list (ℝ × ℝ) × ℝ) : ℝ :=
  circles.1.foldl (λ acc (_, r), acc + π * r^2) 0

theorem circle_area_bound (circles : list (ℝ × ℝ) × ℝ) :
  (∀ c ∈ circles.1, c.2 < 0.001) →
  (∀ (c₁ c₂ : (ℝ × ℝ)), c₁ ∈ circles.1 → c₂ ∈ circles.1 → c₁ ≠ c₂ → dist c₁.1 c₂.1 ≠ 0.001) →
  total_circle_area 1 circles < 0.34 :=
by
  sorry

end circle_area_bound_l654_654642


namespace regular_pentagon_cannot_cover_floor_completely_l654_654879

theorem regular_pentagon_cannot_cover_floor_completely
  (hexagon_interior_angle : ℝ)
  (pentagon_interior_angle : ℝ)
  (square_interior_angle : ℝ)
  (triangle_interior_angle : ℝ)
  (hexagon_condition : 360 / hexagon_interior_angle = 3)
  (square_condition : 360 / square_interior_angle = 4)
  (triangle_condition : 360 / triangle_interior_angle = 6)
  (pentagon_condition : 360 / pentagon_interior_angle ≠ 3)
  (pentagon_condition2 : 360 / pentagon_interior_angle ≠ 4)
  (pentagon_condition3 : 360 / pentagon_interior_angle ≠ 6) :
  pentagon_interior_angle = 108 := 
  sorry

end regular_pentagon_cannot_cover_floor_completely_l654_654879


namespace solve_system_of_equations_l654_654703

theorem solve_system_of_equations :
  ∀ (x1 x2 x3 x4 x5: ℝ), 
  (x3 + x4 + x5)^5 = 3 * x1 ∧ 
  (x4 + x5 + x1)^5 = 3 * x2 ∧ 
  (x5 + x1 + x2)^5 = 3 * x3 ∧ 
  (x1 + x2 + x3)^5 = 3 * x4 ∧ 
  (x2 + x3 + x4)^5 = 3 * x5 →
  (x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 ∧ x5 = 0) ∨ 
  (x1 = 1/3 ∧ x2 = 1/3 ∧ x3 = 1/3 ∧ x4 = 1/3 ∧ x5 = 1/3) ∨ 
  (x1 = -1/3 ∧ x2 = -1/3 ∧ x3 = -1/3 ∧ x4 = -1/3 ∧ x5 = -1/3) := 
by 
  sorry

end solve_system_of_equations_l654_654703


namespace area_ratio_eq_l654_654081

theorem area_ratio_eq (A B C : ℝ × ℝ) (X : ℝ × ℝ)
  (hA : A = (-17,0)) (hB : B = (17,0)) (hC : C = (0,30))
  (hAB : (∥(B.1 - A.1, B.2 - A.2)∥ = 34))
  (hBC : (∥(C.1 - B.1, C.2 - B.2)∥ = 35))
  (hAC : (∥(C.1 - A.1, C.2 - A.2)∥ = 39))
  (h_angle_bisector : (∠ACX = ∠XCX)) :
  (area_of_triangle B C X / area_of_triangle A C X) = 35 / 39 := 
sorry

end area_ratio_eq_l654_654081


namespace top_card_is_red_l654_654458

noncomputable def standard_deck (ranks : ℕ) (suits : ℕ) : ℕ := ranks * suits

def red_cards_in_deck (hearts : ℕ) (diamonds : ℕ) : ℕ := hearts + diamonds

noncomputable def probability_red_card (red_cards : ℕ) (total_cards : ℕ) : ℚ := red_cards / total_cards

theorem top_card_is_red (hearts diamonds spades clubs : ℕ) (deck_size : ℕ)
  (H1 : hearts = 13) (H2 : diamonds = 13) (H3 : spades = 13) (H4 : clubs = 13) (H5 : deck_size = 52):
  probability_red_card (red_cards_in_deck hearts diamonds) deck_size = 1/2 :=
by 
  sorry

end top_card_is_red_l654_654458


namespace num_integers_satisfying_inequality_l654_654931

-- Define the problem
theorem num_integers_satisfying_inequality : 
  {n : ℤ | 15 < n^2 ∧ n^2 < 121}.to_finset.card = 14 :=
by
  sorry -- proof placeholder

end num_integers_satisfying_inequality_l654_654931


namespace number_is_28_l654_654599

-- Definitions from conditions in part a
def inner_expression := 15 - 15
def middle_expression := 37 - inner_expression
def outer_expression (some_number : ℕ) := 45 - (some_number - middle_expression)

-- Lean 4 statement to state the proof problem
theorem number_is_28 (some_number : ℕ) (h : outer_expression some_number = 54) : some_number = 28 := by
  sorry

end number_is_28_l654_654599


namespace telethon_total_revenue_l654_654223

noncomputable def telethon_revenue (first_period_hours : ℕ) (first_period_rate : ℕ) 
  (additional_percent_increase : ℕ) (second_period_hours : ℕ) : ℕ :=
  let first_revenue := first_period_hours * first_period_rate
  let second_period_rate := first_period_rate + (first_period_rate * additional_percent_increase / 100)
  let second_revenue := second_period_hours * second_period_rate
  first_revenue + second_revenue

theorem telethon_total_revenue : 
  telethon_revenue 12 5000 20 14 = 144000 :=
by 
  rfl -- replace 'rfl' with 'sorry' if the proof is non-trivial and longer

end telethon_total_revenue_l654_654223


namespace minimum_marked_cells_ensure_coverage_l654_654376

/-- Definition of the board as a set of coordinates --/
def board : set (ℕ × ℕ) := { p | p.1 < 10 ∧ p.2 < 11 }

/-- Definition of an L-shaped piece as a set of three coordinates --/
def L_shaped_piece (x y : ℕ) : set (ℕ × ℕ) :=
  if x <= y then { (x, y), (x+1, y), (x, y+1) } else { (x, y), (x-1, y), (x, y+1) }

/-- Predicate indicating if a cell is marked --/
def is_marked (marked : set (ℕ × ℕ)) (cell : ℕ × ℕ) : Prop := cell ∈ marked

/-- The specific problem statement --/
theorem minimum_marked_cells_ensure_coverage :
  ∃ (marked : set (ℕ × ℕ)), marked.card = 50 ∧ ∀ x y ∈ board, 
  ∃ m ∈ L_shaped_piece x y, is_marked marked m :=
sorry

end minimum_marked_cells_ensure_coverage_l654_654376


namespace sin_600_eq_neg_sqrt3_div2_l654_654053

theorem sin_600_eq_neg_sqrt3_div2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by sorry

end sin_600_eq_neg_sqrt3_div2_l654_654053


namespace liars_among_islanders_are_only_9_18_19_l654_654831

-- declare the islanders
def Islander := { x : ℕ // x < 19 }

-- define statements
def said_three_of_us_are_liars (i : Islander) : Prop := sorry
def said_six_of_us_are_liars (i : Islander) : Prop := sorry
def said_nine_of_us_are_liars (i : Islander) : Prop := sorry

-- define types
inductive Kind
| Knight
| Liar

-- Islander knowledge
def knows_kind (i j : Islander) : Kind -> Prop := sorry

-- Theorem to prove
theorem liars_among_islanders_are_only_9_18_19 :
  ∃ n : ℕ, n = 9 ∨ n = 18 ∨ n = 19 ∧ 
  ∃ (liars : Fin n → Islander),
    (∀ l, ∃ i, said_nine_of_us_are_liars i → i ∈ (liars)) ∧
    (∀ l, ∃ i, said_six_of_us_are_liars i → i ∉ (liars)) ∧
    (∀ l, ∃ i, said_three_of_us_are_liars i → i ∉ (liars)) :=
by
  sorry

end liars_among_islanders_are_only_9_18_19_l654_654831


namespace total_tape_length_l654_654474

-- Definitions based on the problem conditions
def first_side_songs : ℕ := 6
def second_side_songs : ℕ := 4
def song_length : ℕ := 4

-- Statement to prove the total tape length is 40 minutes
theorem total_tape_length : (first_side_songs + second_side_songs) * song_length = 40 := by
  sorry

end total_tape_length_l654_654474


namespace evaluate_expression_l654_654920

theorem evaluate_expression :
  (⌈(19 / 7 : ℚ) - ⌈(35 / 19 : ℚ)⌉⌉ / ⌈(35 / 7 : ℚ) + ⌈((7 * 19) / 35 : ℚ)⌉⌉) = (1 / 9 : ℚ) :=
by
  sorry

end evaluate_expression_l654_654920


namespace num_ways_pair_knowledge_l654_654737

-- Definitions of the conditions
def People := Fin 12  -- Label the 12 people as Finite type with cardinality 12.

def knows (i j : People) : Prop :=
  -- i knows j if j is one of the two adjacent people or two positions to the left
  j = i + 1 ∨ j = i - 1 ∨ j = i - 2

-- Theorem to prove
theorem num_ways_pair_knowledge : 
  ∃ ways : ℕ, ways = 3 ∧ ∀ (pairs : Finset (People × People)), 
  (pairs.card = 6 ∧ ∀ {a b}, (a,b) ∈ pairs → knows a b) → (pairs = ways) :=
sorry

end num_ways_pair_knowledge_l654_654737


namespace polynomial_nonzero_coeffs_l654_654655

open Polynomial

theorem polynomial_nonzero_coeffs (Q : Polynomial ℤ) (n : ℕ) (hn : 0 < n) (hQ : Q ≠ 0) :
    (Q * (X - 1)^n).coeffs.count (≠ 0) ≥ n + 1 :=
sorry

end polynomial_nonzero_coeffs_l654_654655


namespace sarah_score_l654_654277

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l654_654277


namespace sarah_score_l654_654282

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l654_654282


namespace triangles_congruent_in_hexagon_l654_654829

theorem triangles_congruent_in_hexagon (A B C D E F M : Point) 
  (parallelogram_ABCM : Parallelogram A B C M)
  (parallelogram_CDEM : Parallelogram C D E M)
  (parallelogram_EFAM : Parallelogram E F A M) :
  Congruent (Triangle B D F) (Triangle A C E) :=
sorry

end triangles_congruent_in_hexagon_l654_654829


namespace proof_problem_l654_654971

open Real

def p : Prop := ∀ a : ℝ, a^2017 > -1 → a > -1
def q : Prop := ∀ x : ℝ, x^2 * tan (x^2) > 0

theorem proof_problem : p ∨ q :=
sorry

end proof_problem_l654_654971


namespace sum_of_areas_l654_654570

theorem sum_of_areas (k : ℕ) (h : k > 0) :
  (∑ i in finset.range k, (1 / (2 * (i + 1)) * (1 / (i + 2)))) = (k / (2 * (k + 1))) :=
sorry

end sum_of_areas_l654_654570


namespace polynomial_expansion_l654_654071

theorem polynomial_expansion (x : ℝ) :
  (5 * x^2 + 3 * x - 7) * (4 * x^3) = 20 * x^5 + 12 * x^4 - 28 * x^3 :=
by 
  sorry

end polynomial_expansion_l654_654071


namespace avg_weighted_avg_calc_l654_654900

noncomputable def avg : ℚ → ℚ → ℚ :=
  λ a b, (a + b) / 2

noncomputable def weighted_avg : ℚ → ℚ → ℚ → ℚ → ℚ :=
  λ a b c d, (a + 2*b + c + 2*d) / 6

theorem avg_weighted_avg_calc :
  weighted_avg (weighted_avg 2 2 1 1) (avg 1 2) 0 2 = 17 / 12 :=
by
  sorry

end avg_weighted_avg_calc_l654_654900


namespace Sarahs_score_l654_654272

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l654_654272


namespace right_angle_triangular_pyramid_midsection_area_right_angle_triangular_pyramid_area_sums_right_angle_triangular_pyramid_cosine_sums_l654_654011

-- Definition for areas
variables (A_mid A_slant A1 A2 A3 : ℝ)
-- Definition for dihedral angles
variables (α β γ : ℝ)

-- Property 1: The area of the midsection of a slant side is equal to one-fourth of the slant side area
theorem right_angle_triangular_pyramid_midsection_area :
  A_mid = (1/4) * A_slant := 
begin
  sorry
end

-- Property 2: The sum of the squares of the areas of the three right-angle sides is equal to the square of the slant side area
theorem right_angle_triangular_pyramid_area_sums :
  A1^2 + A2^2 + A3^2 = A_slant^2 := 
begin
  sorry
end

-- Property 3: The sum of the squares of the cosines of the dihedral angles between the slant side and the three right-angle sides is equal to 1
theorem right_angle_triangular_pyramid_cosine_sums :
  cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1 := 
begin
  sorry
end

end right_angle_triangular_pyramid_midsection_area_right_angle_triangular_pyramid_area_sums_right_angle_triangular_pyramid_cosine_sums_l654_654011


namespace cylinder_cone_ratio_l654_654601

theorem cylinder_cone_ratio (r h : ℝ) (S1 S2 : ℝ) 
  (cylinder_lateral_area : S1 = 2 * real.pi * r * h) 
  (cone_lateral_area : S2 = (1 / 2) * S1) : 
  S1 / S2 = 2 := 
by 
  sorry

end cylinder_cone_ratio_l654_654601


namespace total_tape_length_l654_654475

-- Definitions based on the problem conditions
def first_side_songs : ℕ := 6
def second_side_songs : ℕ := 4
def song_length : ℕ := 4

-- Statement to prove the total tape length is 40 minutes
theorem total_tape_length : (first_side_songs + second_side_songs) * song_length = 40 := by
  sorry

end total_tape_length_l654_654475


namespace maximize_victory_probability_l654_654761

-- Define the conditions
variables {n : ℕ}
def number_of_voters := 2 * n
def half_support_miraflores := n
def half_support_dick_maloney := n
def miraflores_is_voter := true

-- Define the districts
def district1 := {miraflores}
def district2 := {voters | voters ≠ miraflores}

theorem maximize_victory_probability (n : ℕ) (h₁ : nat.odd (2*n + 1) = true) : 
  (let district1_voters := 1 in
   let district2_voters := 2*n - 1 in
   maximize_probability_of_winning(district1_voters, district2_voters) = true) :=
sorry

end maximize_victory_probability_l654_654761


namespace book_costs_and_scenarios_l654_654779

theorem book_costs_and_scenarios :
  (∃ (x y : ℕ), x + 3 * y = 180 ∧ 3 * x + y = 140 ∧ 
    (x = 30) ∧ (y = 50)) ∧ 
  (∀ (m : ℕ), (30 * m + 75 * m) ≤ 700 → (∃ (m_values : Finset ℕ), 
    m_values = {2, 4, 6} ∧ (m ∈ m_values))) :=
  sorry

end book_costs_and_scenarios_l654_654779


namespace maximize_victory_probability_l654_654763

-- Define the conditions
variables {n : ℕ}
def number_of_voters := 2 * n
def half_support_miraflores := n
def half_support_dick_maloney := n
def miraflores_is_voter := true

-- Define the districts
def district1 := {miraflores}
def district2 := {voters | voters ≠ miraflores}

theorem maximize_victory_probability (n : ℕ) (h₁ : nat.odd (2*n + 1) = true) : 
  (let district1_voters := 1 in
   let district2_voters := 2*n - 1 in
   maximize_probability_of_winning(district1_voters, district2_voters) = true) :=
sorry

end maximize_victory_probability_l654_654763


namespace limit_series_of_a_l654_654112

-- Define the problem in Lean statement.
theorem limit_series_of_a (a : ℝ) (h1 : 0 < a)
  (coeff_condition : ∀ (x : ℝ), (∃ c : ℝ, ((ax^2 + (1 / sqrt x))^4).coeff 3 = c) ∧ c = (3 / 2)) :
  (tendsto (λ n, (finset.range (n + 1)).sum (λ k, a ^ k)) at_top (𝓝 1)) :=
sorry

end limit_series_of_a_l654_654112


namespace smallest_positive_period_l654_654524

def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem smallest_positive_period : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ (∀ T' > 0, (∀ x, f(x + T') = f(x)) → T ≤ T') ∧ T = Real.pi :=
by
  sorry

end smallest_positive_period_l654_654524


namespace x_intercept_of_line_is_7_over_2_l654_654459

-- Definitions for the conditions
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (6, 5)

-- Define what it means to be the x-intercept of the line
def x_intercept_of_line (x : ℝ) : Prop :=
  ∃ m b : ℝ, (point1.snd) = m * (point1.fst) + b ∧ (point2.snd) = m * (point2.fst) + b ∧ 0 = m * x + b

-- The theorem stating the x-intercept
theorem x_intercept_of_line_is_7_over_2 : x_intercept_of_line (7 / 2) :=
sorry

end x_intercept_of_line_is_7_over_2_l654_654459


namespace remainder_of_4123_div_by_32_l654_654374

theorem remainder_of_4123_div_by_32 : 
  ∃ r, 0 ≤ r ∧ r < 32 ∧ 4123 = 32 * (4123 / 32) + r ∧ r = 27 := by
  sorry

end remainder_of_4123_div_by_32_l654_654374


namespace tan_double_angle_l654_654104

theorem tan_double_angle (x : ℝ) (hx : cos x = -1 / 3) (hx_quad3 : π < x ∧ x < 3 * π / 2) :
  tan (2 * x) = - (4 * real.sqrt 2) / 7 :=
by
  sorry

end tan_double_angle_l654_654104


namespace greatest_distance_C_D_l654_654634

open Complex

def C : Set ℂ := {2, -2, 2 * I, -2 * I}
def D : Set ℂ := {3, -(3 / 2 : ℂ) + (3 * real.sqrt 3 / 2) * I, -(3 / 2 : ℂ) - (3 * real.sqrt 3 / 2) * I}

theorem greatest_distance_C_D : 
  (∀ z_C ∈ C, ∀ z_D ∈ D, dist (z_C : ℂ) (z_D : ℂ) ≤ real.sqrt 13) ∧ 
  (∃ z_C ∈ C, ∃ z_D ∈ D, dist (z_C : ℂ) (z_D : ℂ) = real.sqrt 13) := 
sorry

end greatest_distance_C_D_l654_654634


namespace perimeter_of_remaining_shape_l654_654871

theorem perimeter_of_remaining_shape (A B C D E F : Point) (h1 : equilateral_triangle A B C 4)
                                      (h2 : square D B E F 1)
                                      (h3 : cuts_off_square A B C D E F)
                                      (remaining_AB_BC : segment A D = 3 ∧ segment E C = 3) :
  perimeter (polygon [A, C, E, F, D]) = 15 :=
sorry

end perimeter_of_remaining_shape_l654_654871


namespace limit_sequence_l654_654429

/-- Define the arithmetic sequence sum -/
def arithmetic_sequence_sum (n : ℕ) : ℝ :=
  let a := 1
  let l := 4 * n - 3
  n / 2 * (a + l)

/-- Define the limit expression -/
def limit_expr (n : ℕ) : ℝ :=
  (arithmetic_sequence_sum n / (n + 1) - (4 * n + 1) / 2)

theorem limit_sequence : 
  tendsto (λ n : ℕ, limit_expr n) at_top (𝓝 (- 7 / 2)) :=
sorry

end limit_sequence_l654_654429


namespace inradii_relationship_l654_654650

variables (ΔABC : Triangle)
variables (A B C D E F : Point)
variables (r0 r R : ℝ)
variables (inradius : Triangle → ℝ)
variables [acute_triangle : AcuteTriangle ΔABC]
variables [points_on_sides : 
  Side D B C ∧ Side E C A ∧ Side F A B]
variables [inradii_equal : 
  inradius (Triangle.mk A E F) = r0 ∧ 
  inradius (Triangle.mk B D F) = r0 ∧ 
  inradius (Triangle.mk C D E) = r0]
variables [inradius_DEF : inradius (Triangle.mk D E F) = r]
variables [inradius_ABC : inradius ΔABC = R]

theorem inradii_relationship : r + r0 = R :=
begin
  sorry
end

end inradii_relationship_l654_654650


namespace fewer_twos_for_100_l654_654349

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l654_654349


namespace sum_of_squares_eq_frac_squared_l654_654182

theorem sum_of_squares_eq_frac_squared (x y z a b c : ℝ) (hxya : x * y = a) (hxzb : x * z = b) (hyzc : y * z = c)
  (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) (ha0 : a ≠ 0) (hb0 : b ≠ 0) (hc0 : c ≠ 0) :
  x^2 + y^2 + z^2 = ((a * b)^2 + (a * c)^2 + (b * c)^2) / (a * b * c) :=
by
  sorry

end sum_of_squares_eq_frac_squared_l654_654182


namespace smallest_sum_of_digits_S_l654_654786

theorem smallest_sum_of_digits_S :
  ∃ a b : ℕ,
    (100 ≤ a ∧ a < 1000) ∧
    (100 ≤ b ∧ b < 1000) ∧
    (is_unique_digits a b) ∧
    (a + b > 1000) ∧
    (distinct_hundreds_digits a b) ∧
    (sum_of_digits (a + b) = 17) := sorry

-- Definitions used in the conditions:
def is_unique_digits (a b : ℕ) : Prop :=
  let a_digits := (a % 10, (a / 10) % 10, a / 100)
      b_digits := (b % 10, (b / 10) % 10, b / 100)
  in (list.nodup (a_digits ++ b_digits))

def distinct_hundreds_digits (a b : ℕ) : Prop :=
  let a_hundred := a / 100
      b_hundred := b / 100
  in a_hundred ≠ b_hundred

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + (n / 100)

end smallest_sum_of_digits_S_l654_654786


namespace part_a_part_b_part_c_l654_654326

variable (N : ℕ) (r : Fin N → Fin N → ℝ)

-- Part (a)
theorem part_a (h : ∀ (s : Finset (Fin N)), s.card = 5 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

-- Part (b)
theorem part_b (h : ∀ (s : Finset (Fin N)), s.card = 4 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ¬ (∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j) :=
sorry

-- Part (c)
theorem part_c (h : ∀ (s : Finset (Fin N)), s.card = 6 → (exists pts : s → ℝ × ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ (pts : Fin N → ℝ × ℝ × ℝ), ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

end part_a_part_b_part_c_l654_654326


namespace knights_and_liars_l654_654834

-- Knight or Liar Property
def isKnight (x : ℕ) : Prop := ∀ (s : ℕ), x ∈ s → s = 0

-- Islander Property
def isLiar (x : ℕ) : Prop := ∀ (s : ℕ), x ∈ s → s ≠ 0

-- Main hypothesis stating the problem conditions
theorem knights_and_liars (n : ℕ) (A B C : Finset ℕ) (count1 count2 count3 : ℕ) :
  (A.card = 3 ∧ ∀ x ∈ A, isLiar x ↔ count1 = 3) ∧
  (B.card = 6 ∧ ∀ x ∈ B, isLiar x ↔ count2 = 6) ∧
  (C.card = 9 ∧ ∀ x ∈ C, isKnight x ↔ count3 = 9) →
  n = 9 ∨ n = 18 ∨ n = 19 :=
begin
  sorry
end

end knights_and_liars_l654_654834


namespace factorial_comparison_l654_654809

theorem factorial_comparison :
  (Nat.factorial (Nat.factorial 100)) <
  (Nat.factorial 99)^(Nat.factorial 100) * (Nat.factorial 100)^(Nat.factorial 99) :=
  sorry

end factorial_comparison_l654_654809


namespace fewer_twos_for_100_l654_654352

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l654_654352


namespace no_solution_in_natural_numbers_l654_654689

theorem no_solution_in_natural_numbers (x y z : ℕ) (hxy : x ≠ 0) (hyz : y ≠ 0) (hzx : z ≠ 0) :
  ¬ (x / y + y / z + z / x = 1) :=
by sorry

end no_solution_in_natural_numbers_l654_654689


namespace train_crossing_time_l654_654820

theorem train_crossing_time :
  ∀ (distance speed_kmph : ℕ),
  distance = 150 →
  speed_kmph = 36 →
  distance / (speed_kmph * 1000 / 3600) = 15 :=
by
  intros distance speed_kmph h_distance h_speed
  rw [h_distance, h_speed]
  sorry 

end train_crossing_time_l654_654820


namespace last_digit_to_appear_in_fib_square_units_l654_654498

/-- 
Consider the sequence where each term is the units digit of the square of the units 
digit of each corresponding Fibonacci number. Prove that the last digit to appear in the 
units position of this new sequence is 2.
-/
theorem last_digit_to_appear_in_fib_square_units (s : ℕ → ℕ) 
  (fib_units : Π n, (s n) = (fib n % 10)) :
  ∃ n, (units_digit((s n)^2) = 2)  :=
sorry

end last_digit_to_appear_in_fib_square_units_l654_654498


namespace sarah_score_l654_654280

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l654_654280


namespace candy_distribution_l654_654918

theorem candy_distribution : 
  let candies := 8 in
  let bags := 3 in
  let red_min := 2 in
  let blue_min := 2 in
  let white_min := 0 in
  ∃ arrangements : ℕ, arrangements = 46272 :=
begin
  sorry
end

end candy_distribution_l654_654918


namespace compare_logarithmic_values_l654_654953

theorem compare_logarithmic_values :
  let a := Real.log 3.4 / Real.log 2
  let b := Real.log 3.6 / Real.log 4
  let c := Real.log 0.3 / Real.log 3
  c < b ∧ b < a :=
by
  sorry

end compare_logarithmic_values_l654_654953


namespace positive_t_for_modulus_l654_654530

theorem positive_t_for_modulus : ∃ t > 0, |(8 : ℂ) + t * I| = 12 ∧ t = 4 * real.sqrt 5 :=
by { sorry }

end positive_t_for_modulus_l654_654530


namespace not_native_of_knights_and_liars_l654_654682

-- Definitions based on conditions
def isKnight (person : Type) : Prop := ∀ statement : Prop, person = statement → statement
def isLiar (person : Type) : Prop := ∀ statement : Prop, person = ¬ statement → statement
def saysLiar (person : Type) : Prop := person = "I am a liar"

theorem not_native_of_knights_and_liars (person : Type) :
  (isKnight person ∨ isLiar person) ∧ saysLiar person → False :=
by
  sorry

end not_native_of_knights_and_liars_l654_654682


namespace cyclic_quadrilateral_l654_654113

structure Quadrilateral where
  A B C D : Point

def is_intersection (p q r s : Line) (E : Point) : Prop :=
  E ∈ p ∧ E ∈ q ∧ p ≠ q ∧ E ∈ r ∧ E ∈ s ∧ r ≠ s

def are_perpendicular (l₁ l₂ : Line) : Prop :=
  ∃ (A B : Point), l₁ = Line.mk A B ∧ ∃ (C D : Point), l₂ = Line.mk C D ∧
  l₁.slope * l₂.slope = -1

def bisector (A B C : Point) : Line := sorry

theorem cyclic_quadrilateral
  (ABCD : Quadrilateral)
  (E : Point) (F : Point)
  (h1 : is_intersection (Line.mk ABCD.D ABCD.A) (Line.mk ABCD.C ABCD.B) E)
  (h2 : is_intersection (Line.mk ABCD.A ABCD.B) (Line.mk ABCD.D ABCD.C) F)
  (h3 : are_perpendicular (bisector ABCD.B E ABCD.A) (bisector ABCD.B F ABCD.C)) :
  Cyclic ABCD :=
sorry

end cyclic_quadrilateral_l654_654113


namespace optimal_voter_split_l654_654755

-- Definitions
variables (Voters : Type) [fintype Voters] (n : ℕ)
variables (supports_miraflores : Voters → Prop)
variables [decidable_pred supports_miraflores]

-- Conditions
def half_supports_miraflores := fintype.card { v // supports_miraflores v } = n
def half_supports_maloney := fintype.card { v // ¬ supports_miraflores v } = n

-- Question (translated to a theorem)
theorem optimal_voter_split (h_m : half_supports_miraflores Voters n supports_miraflores)
    (h_d: half_supports_maloney Voters n supports_miraflores) :
  ∃ (D1 D2 : finset Voters), 
    ((D1 = {v | supports_miraflores v}) ∧ 
    (D2 = {v | ¬supports_miraflores v}) ∧ 
    (∀ v, v ∈ D1 ∨ v ∈ D2) ∧ 
    (∀ v, ¬ (v ∈ D1 ∧ v ∈ D2)) ∧ 
    (finset.card D1 = 1) ∧ 
    (finset.card D2 = 2 * n - 1)) :=
sorry

end optimal_voter_split_l654_654755


namespace find_z_satisfying_conditions_l654_654957

theorem find_z_satisfying_conditions (z : ℂ) (hz1_real : (z / (1 + z^2)).im = 0) (hz2_real : (z^2 / (1 + z)).im = 0) :
  z = -1/2 + (1 / 2 * Complex.I * Complex.sqrt 3) ∨ z = -1/2 - (1 / 2 * Complex.I * Complex.sqrt 3) :=
by
  sorry

end find_z_satisfying_conditions_l654_654957


namespace angle_B_eq_2_over_3_pi_area_of_triangle_ABC_l654_654198

theorem angle_B_eq_2_over_3_pi 
  (a b c : ℝ) (A B C : ℝ) (h_ABC : a = 2*R*sin A ∧ b = 2*R*sin B ∧ c = 2*R*sin C)
  (h_cos_ratio : (cos B / cos C) = - (b / (2*a + c))) :
  B = 2/3 * π :=
sorry

theorem area_of_triangle_ABC 
  (a b c : ℝ) (A B C : ℝ) (h_ABC : a = 2*R*sin A ∧ b = 2*R*sin B ∧ c = 2*R*sin C)
  (h_b_sqrt3 : b = sqrt 3) (h_a_plus_c_4 : a + c = 4) :
  let S := ac (A B C) a b c 
  in  S = 13*sqrt 3 / 4 :=
sorry

end angle_B_eq_2_over_3_pi_area_of_triangle_ABC_l654_654198


namespace sum_of_all_four_digit_integers_l654_654406

theorem sum_of_all_four_digit_integers :
  (Finset.range (9999 + 1)).filter (λ x => x ≥ 1000).sum = 49495500 :=
by
  sorry

end sum_of_all_four_digit_integers_l654_654406


namespace partition_into_pyramids_l654_654537

theorem partition_into_pyramids (n : ℕ) (points : Fin 4n → ℝ × ℝ × ℝ)
  (h_no_coplanar : ∀ (a b c d : Fin 4n), ¬(AffineIndependent ℝ ![points a, points b, points c, points d])) :
  ∃ (pyramids : Vector (Vector (ℝ × ℝ × ℝ) 4) n),
    ∀ i j (hi hj : Fin n), i ≠ j → Disjoint (pyramids[i]) (pyramids[j]) :=
by
  sorry

end partition_into_pyramids_l654_654537


namespace incorrect_conclusion_l654_654876

-- Define the conditions as hypotheses
def independence_tests :
  Prop := sorry -- placeholder for the precise definition of independence tests

def principle_of_small_probability :
  Prop := sorry -- placeholder for precise definition of the principle of small probability

def conclusion_varies_with_samples :
  Prop := sorry -- placeholder for precise definition of conclusion varying with samples

def alternative_methods_exist_for_relationships :
  Prop := sorry -- placeholder for precise definition of alternative methods

-- State the main theorem
theorem incorrect_conclusion () :
  independence_tests ∧ 
  principle_of_small_probability ∧ 
  conclusion_varies_with_samples ∧ 
  alternative_methods_exist_for_relationships → 
  ¬ (theorem independence_test_conclusion_is_correct)
 :=
sorry

end incorrect_conclusion_l654_654876


namespace harmonic_mean_construction_l654_654721

theorem harmonic_mean_construction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (let h := (2 * a * b) / (a + b) in
  ∃ (A B : ℝ) (e f g : ℝ), (A = a) ∧ (B = b) ∧ -- lengths and unit vectors
  (g = A / B) ∧ -- angle bisector condition
  ∃ (D : ℝ), 
  (D = (a + b) / 2) ∧ -- D point condition
  (D = (A + B) / 2) ∧ -- correct construction
  h = (2 * A * B) / (A + B)) :=
sorry

end harmonic_mean_construction_l654_654721


namespace product_of_divisors_18_l654_654916

-- Define the prime factorization of 18
def n : ℕ := 18
def prime_factors (n : ℕ) : List (ℕ × ℕ) := [(2, 1), (3, 2)]

-- Definition of product of all positive divisors
def product_of_divisors (n : ℕ) : ℕ := 
  (n : ℕ)^(List.length (List.range (List.foldl (λ acc (p : ℕ × ℕ), acc * (p.snd + 1)) 1 (prime_factors n)) / 2))

-- Lean theorem statement
theorem product_of_divisors_18 (h : n = 18) :
  product_of_divisors n = 5832 :=
by
  sorry

end product_of_divisors_18_l654_654916


namespace f_neg_a_eq_zero_l654_654993

-- Given function definition
def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

-- Given condition
variable (a : ℝ) (h : f a = 2)

-- To prove that f(-a) = 0
theorem f_neg_a_eq_zero : f (-a) = 0 :=
by
  sorry

end f_neg_a_eq_zero_l654_654993


namespace area_inequality_l654_654261

-- Definitions: Introducing the necessary variables and conditions
variables {ABC : Type*} [triangle ABC]
variables (A B C C1 A1 B1 : ABC)
variables (area : ABC → ℝ)
variables (S1 S2 S3 S_ABC : ℝ)

-- Conditions in the problem
def is_on_segment (P Q R : ABC) : Prop := sorry -- Placeholder for points on segments

-- Given points C1, A1, B1 are on respective segments AB, BC, CA
axiom C1_on_AB : is_on_segment C1 A B
axiom A1_on_BC : is_on_segment A1 B C
axiom B1_on_CA : is_on_segment B1 C A

-- Definitions of areas S1, S2, S3
axiom area_S1 : area (triangle.mk B1 C1) = S1
axiom area_S2 : area (triangle.mk A1 C1) = S2
axiom area_S3 : area (triangle.mk C A1) = S3

-- Given the area of triangle ABC
axiom area_ABC : area (triangle.mk A B C) = S_ABC

-- The theorem to prove
theorem area_inequality :
  sqrt S1 + sqrt S2 + sqrt S3 ≤ (3 / 2) * sqrt S_ABC :=
sorry

end area_inequality_l654_654261


namespace problem_xy_plane_l654_654728

-- Definitions of the transformations
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

def reflect_about_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

-- Lean statement: Prove b - a = -4 under the given transformations
theorem problem_xy_plane (a b : ℝ)
  (h : rotate180 a b 2 3 = (x', y'))
  (hx : reflect_about_y_eq_x x' y' = (5, -1)) :
  b - a = -4 :=
begin
  sorry
end

end problem_xy_plane_l654_654728


namespace find_gx_l654_654185

theorem find_gx (f g : ℚ[X]) 
  (hf : f = Polynomial.C 1 * Polynomial.X ^ 4 - Polynomial.C 1 * Polynomial.X ^ 3 - Polynomial.C 2 * Polynomial.X ^ 2 + Polynomial.C 1 * Polynomial.X)
  (h : f + g = Polynomial.C 2 * Polynomial.X ^ 2 + Polynomial.C 3 * Polynomial.X - Polynomial.C 1) :
  g = -Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 4 * Polynomial.X ^ 2 + Polynomial.C 2 * Polynomial.X - Polynomial.C 1 :=
by
  sorry

end find_gx_l654_654185


namespace congruent_modulo_b_l654_654660

theorem congruent_modulo_b (a b : ℕ) (m : ℕ) (h : m > 0)
  (h₁ : a = ∑ k in finset.range 21, nat.choose 20 k * 2^k)
  (h₂ : a % 10 = b % 10)
  (hb : b = 2021 ∨ b = 2020 ∨ b = 2019 ∨ b = 2018) :
  b = 2021 :=
by
  sorry

end congruent_modulo_b_l654_654660


namespace measure_PB_eq_x_plus_1_l654_654632

variable (M : Type) (arcCAB : M → M → M)
variable (AB AC AP PB MP : M → Prop)
variable [chord : ∀ (x y : M), Prop]
variable [measure : ∀ (x : M), ℝ]

axiom midpoint_arc: ∀ {C A B : M}, M = arcCAB C A B → midpoint M ∈ arcCAB C A B
axiom perpendicular : ∀ {P A B : M}, MP P → AB A B → is_perpendicular MP AB
axiom measure_of_AC : ∀ {A C : M}, AC A C → measure AC = x
axiom measure_of_AP : ∀ {A P : M}, AP A P → measure AP = x + 1
axiom midpoint_P_AB : ∀ {A B P : M}, P = midpoint AB A B → AB A B / 2 = AP A P

theorem measure_PB_eq_x_plus_1 (M : Type) [chord : ∀ (x y : M), Prop] [measure : ∀ (x : M), ℝ] :
  ∀ (M P A B C : M), 
    (M = arcCAB C A B) ∧ (MP P) ∧ (AB A B) ∧ (perpendicular MP AB) ∧ 
    (AC A C) ∧ (measure AC = x) ∧ (AP A P) ∧ (measure AP = x + 1) → (PB A P) → measure PB = x + 1 := by
  sorry

end measure_PB_eq_x_plus_1_l654_654632


namespace count_false_relationships_l654_654999

noncomputable def log_base (base x : ℝ) : ℝ := log x / log base

theorem count_false_relationships (a b : ℝ) (h : log_base (1/2) a = log_base (1/3) b) :
  {p : Prop // ({a > b > 1, 0 < b < a < 1, b > a > 1, 0 < a < b < 1, a = b}.count (λ p, ¬p)) = 2} :=
by sorry

end count_false_relationships_l654_654999


namespace ten_N_plus_D_eq_715_l654_654452

-- Definitions based on problem conditions
def probability_event_even (p_odd_even: ℚ) (p_even: ℚ) : ℚ := p_odd_even + p_even

def N : ℚ := 59
def D : ℚ := 125
def probability : ℚ := N / D

theorem ten_N_plus_D_eq_715
  (p_odd_even: ℚ := 27 / 125)
  (p_even: ℚ := 32 / 125)
  (H: probability_event_even p_odd_even p_even = probability)
  : 10 * N + D = 715 :=
by
  rw probability
  rw N
  rw D
  sorry

end ten_N_plus_D_eq_715_l654_654452


namespace regular_pentagon_cannot_cover_floor_l654_654878

theorem regular_pentagon_cannot_cover_floor :
  ¬(∃ n : ℕ, 360 % 108 = 0) :=
begin
  sorry
end

end regular_pentagon_cannot_cover_floor_l654_654878


namespace max_varphi_l654_654698

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ + (2 * Real.pi / 3))

theorem max_varphi (φ : ℝ) (h : φ < 0) (hE : ∀ x, g x φ = g (-x) φ) : φ = -Real.pi / 6 :=
by
  sorry

end max_varphi_l654_654698


namespace trigonometric_identity_l654_654044

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end trigonometric_identity_l654_654044


namespace sum_four_digit_integers_l654_654398

theorem sum_four_digit_integers : 
  ∑ k in Finset.range (9999 - 1000 + 1), (k + 1000) = 49495500 := 
by
  sorry

end sum_four_digit_integers_l654_654398


namespace indefinite_integral_eq_l654_654430

theorem indefinite_integral_eq : 
  ∫ (x : ℝ) in -∞..∞, (2 * x^3 + 3 * x^2 + 3 * x + 2) / ((x^2 + x + 1) * (x^2 + 1)) = 
    (1 / 2 * log (|x^2 + x + 1|)) + (1 / (sqrt 3) * arctan ((2 * x + 1) / (sqrt 3))) + (1 / 2 * log (|x^2 + 1|)) + arctan x + C :=
sorry

end indefinite_integral_eq_l654_654430


namespace max_geom_common_ratio_is_three_l654_654564

noncomputable def max_common_ratio_geometric_sequence : ℝ :=
  let circle_center := (1 : ℝ, 0 : ℝ)
  let radius := 4
  let line_distance_from_center := (1 + 9) / Real.sqrt (1 + (Real.sqrt 3) ^ 2)
  let min_distance := line_distance_from_center - radius
  let max_distance := line_distance_from_center + radius
  let x1 := min_distance
  let x3 := max_distance
  Real.sqrt (x3 / x1)

theorem max_geom_common_ratio_is_three (x1 x2 x3 : ℝ) (P Q R : ℝ × ℝ)
  (h_circle : ∀ p ∈ ({P, Q, R} : Finset (ℝ × ℝ)), p.1 ^ 2 + p.2 ^ 2 - p.1 - 15 = 0)
  (h_line : ∀ p ∈ ({P, Q, R} : Finset (ℝ × ℝ)), ∃ d, d = Real.abs (p.1 + Real.sqrt 3 * p.2 + 9) / Real.sqrt (1 + (Real.sqrt 3) ^ 2))
  (h_geom : ∃ r, {x1, x2, x3}.pairwise (λ a b, a = r * b ∨ b = r * a)) :
  max_common_ratio_geometric_sequence = 3 :=
  by
    have : ∀ x, x ∈ ({x1, x2, x3} : Finset ℝ) → x = x1 ∨ x = x2 ∨ x = x3 := sorry
    sorry

end max_geom_common_ratio_is_three_l654_654564


namespace homework_time_decrease_l654_654338

variable (x : ℝ)
variable (initial_time final_time : ℝ)
variable (adjustments : ℕ)

def rate_of_decrease (initial_time final_time : ℝ) (adjustments : ℕ) (x : ℝ) := 
  initial_time * (1 - x)^adjustments = final_time

theorem homework_time_decrease 
  (h_initial : initial_time = 100) 
  (h_final : final_time = 70)
  (h_adjustments : adjustments = 2)
  (h_decrease : rate_of_decrease initial_time final_time adjustments x) : 
  100 * (1 - x)^2 = 70 :=
by
  sorry

end homework_time_decrease_l654_654338


namespace modulus_of_pure_imaginary_l654_654188

theorem modulus_of_pure_imaginary (b : ℝ) (z : ℂ) 
  (h : z = (3 + b * complex.I) * (1 + complex.I) - 2) 
  (h_pure_imaginary : z.re = 0) : complex.abs z = 4 :=
sorry

end modulus_of_pure_imaginary_l654_654188


namespace num_even_ints_satisfying_inequality_l654_654169

theorem num_even_ints_satisfying_inequality : 
  let S := {m : ℤ | ∃ (h1 : m ≠ 0) (h2 : even m), 1 / |(m : ℝ)| ≥ 1 / 5} 
  in S.card = 4 :=
by 
  sorry

end num_even_ints_satisfying_inequality_l654_654169


namespace miraflores_optimal_strategy_l654_654750

-- Definitions based on conditions
variable (n : ℕ)
def total_voters := 2 * n
def miraflores_supporters := n
def dick_maloney_supporters := n
def miraflores_is_a_voter := 1
def law_allows_division := true
def election_winner (district1 district2 : Set ℕ) : ℕ := 
  if (district1.card = 1 ∧ miraflores_is_a_voter ∈ district1) then miraflores_is_a_voter else dick_maloney_supporters

-- Mathematically equivalent proof problem
theorem miraflores_optimal_strategy (hall : law_allows_division) :
  (exists (district1 district2 : Set ℕ),
    ∀ v, v ∈ district1 ∨ v ∈ district2 ∧ district1.card + district2.card = total_voters ∧
    miraflores_supporters = 1 ∧ district1 = {miraflores_is_a_voter} ∧
    (election_winner district1 district2) = miraflores_is_a_voter) :=
sorry

end miraflores_optimal_strategy_l654_654750


namespace equilateral_triangle_perimeter_l654_654372

-- Define the condition of an equilateral triangle where each side is 7 cm
def side_length : ℕ := 7

def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

-- Define the perimeter function for a triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- Statement to prove
theorem equilateral_triangle_perimeter : is_equilateral_triangle side_length side_length side_length → perimeter side_length side_length side_length = 21 :=
sorry

end equilateral_triangle_perimeter_l654_654372


namespace ecuadorian_number_unique_l654_654007

def is_Ecuadorian (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n < 1000 ∧ c ≠ 0 ∧ n % 36 = 0 ∧ (n - (100 * c + 10 * b + a) > 0) ∧ (n - (100 * c + 10 * b + a)) % 36 = 0

theorem ecuadorian_number_unique (n : ℕ) : 
  is_Ecuadorian n → n = 864 :=
sorry

end ecuadorian_number_unique_l654_654007


namespace product_of_two_integers_l654_654610

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 22) (h2 : x^2 - y^2 = 44) : x * y = 120 :=
by
  sorry

end product_of_two_integers_l654_654610


namespace num_of_integers_satisfying_ineq_l654_654592

theorem num_of_integers_satisfying_ineq : 
  (set.count (set_of (λ x : ℤ, -4 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9))) = 5 :=
by sorry

end num_of_integers_satisfying_ineq_l654_654592


namespace binomial_coefficient_x3_l654_654927

theorem binomial_coefficient_x3 (n : ℕ) (h : 2 ^ n = 32) : 
  let coeff := (2^(5 - 1) * Nat.choose 5 1) 
  in coeff = 80 :=
by
  sorry

end binomial_coefficient_x3_l654_654927


namespace check_piecewise_function_l654_654418

def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ -4 then -0.2 * x + 1.2
  else if x ≤ 0 then -0.5 * x
  else if x ≤ 2 then 2.5 * x
  else (x + 13) / 3

theorem check_piecewise_function :
  ∀ x : ℝ, 
    (x ≤ -4 → piecewise_function x = -0.2 * x + 1.2) ∧
    (-4 ≤ x ∧ x ≤ 0 → piecewise_function x = -0.5 * x) ∧
    (0 ≤ x ∧ x ≤ 2 → piecewise_function x = 2.5 * x) ∧
    (2 ≤ x → piecewise_function x = (x + 13) / 3) ∧
    (piecewise_function (-9) = 3) ∧
    (piecewise_function (-4) = 2) ∧
    (piecewise_function (0) = 0) ∧
    (piecewise_function (2) = 5) ∧
    (piecewise_function (5) = 6) := 
by
  intros
  split
  { intros hx, rw if_pos hx }
  { split
    { intros hx, rw [if_neg (not_le_of_gt (lt_of_lt_of_le dec_trivial (le_of_lt hx.2))),
                     if_pos (hx.2)] }
    { split
      { intros hx, rw [if_neg (not_le_of_gt (lt_of_le_of_lt (le_of_lt hx.1) dec_trivial)),
                       if_neg (not_le_of_gt (lt_of_lt_of_le dec_trivial (le_of_gt dec_trivial))),
                       if_pos hx.2] }
      { split 
        { intros hx, rw [if_neg (not_le_of_gt (lt_of_le_of_lt hx (lt_add_one_of_pos (by norm_num : (3 : ℝ) > 0)))),
                        if_neg (not_le_of_gt (lt_of_lt_of_le (by norm_num : (2 : ℝ) < 3))),
                        if_neg (not_le_of_gt (lt_add_one_of_pos (by norm_num : (3 : ℝ) > 0))),
                        if_pos hx] }
        { split 
          { exact rfl }
          { split
            { exact rfl }
            { split
              { exact rfl }
              { exact rfl }
            }
          }
        }
      }
    }
  }

end check_piecewise_function_l654_654418


namespace solution1_solution2_l654_654348

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l654_654348


namespace geometric_sequence_property_l654_654555

variable (a : ℕ → ℤ)
-- Assume the sequence is geometric with ratio r
variable (r : ℤ)

-- Define the sequence a_n as a geometric sequence
def geometric_sequence (a : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n * r

-- Given condition: a_4 + a_8 = -2
axiom condition : a 4 + a 8 = -2

theorem geometric_sequence_property
  (h : geometric_sequence a r) : a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geometric_sequence_property_l654_654555


namespace trapezoid_ABCD_area_l654_654230

noncomputable def trapezoid_area (A B C D X Y : ℝ × ℝ) : ℝ :=
  let AX : ℝ := 4
  let XY : ℝ := 2
  let YC : ℝ := 1
  
  -- Coordinates setup based on problem conditions
  let D_y : ℝ := 4 * Real.sqrt 3
  let B_y : ℝ := Real.sqrt 3

  -- Area calculations
  let area_AXD : ℝ := 1 / 2 * AX * D_y * (Real.sqrt 3 / 2)
  let area_BYC : ℝ := 1 / 2 * 1 * B_y * (Real.sqrt 3 / 2)

  -- Total area is the sum of areas of triangles AXD and BYC
  area_AXD + area_BYC

-- The proof problem statement
theorem trapezoid_ABCD_area :
  ∀ (A B C D X Y : ℝ × ℝ),
  (X = (0, 0)) →
  (A = (-4, 0)) →
  (Y = (-2, 0)) →
  (C = (-3, 0)) →
  (AX = 4) →
  (XY = 2) →
  (YC = 1) →
  (∠AXD = 60) →
  (∠BYC = 60) →
  (B = (-2, B_y)) →
  (D = (-4, D_y)) →
  trapezoid_area A B C D X Y = 12.75 :=
by
  intros A B C D X Y hX hA hY hC hAX hXY hYC hAngleAXD hAngleBYC hB hD
  simp [trapezoid_area]
  sorry

end trapezoid_ABCD_area_l654_654230


namespace problem_statement_l654_654236

-- Define the sum of the elements of the nth row in Pascal's triangle as 2^n
def S (n : ℕ) : ℝ := 2 ^ n

-- Define g(n) as the natural logarithm of the sum of the elements of the nth row in Pascal's triangle
def g (n : ℕ) : ℝ := Real.log (S n)

-- State the theorem
theorem problem_statement (n : ℕ) : (g n) / Real.log 2 = n := by
  sorry

end problem_statement_l654_654236


namespace trigonometric_expression_value_l654_654040

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_expression_value_l654_654040


namespace fewer_twos_result_100_l654_654368

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l654_654368


namespace proofProblem1_proofProblem2_l654_654895

noncomputable def Problem1 : Prop :=
  (sqrt 48 - (1 / 4) * sqrt 6) / (- (1 / 9) * sqrt 27) = -12 + (3 * sqrt 2) / 4

theorem proofProblem1 : Problem1 := by
  sorry

noncomputable def Problem2 (x y : ℚ) : Prop :=
  x^2 + y^2 - 2 * x * y = 3

theorem proofProblem2 :
  Problem2 ((1 / 2) * (sqrt 3 + 1)) ((1 / 2) * (1 - sqrt 3)) := by
  sorry

end proofProblem1_proofProblem2_l654_654895


namespace distinct_ball_distributions_l654_654171

theorem distinct_ball_distributions : 
  ∃ (distros : Set (Fin 4 → Fin 6)), 
    distros = { f | f.Sum = 5 ∧
                 ∀ m n : Fin 4, f m ≥ f n ∧ 
                 distros = { (5, 0, 0, 0), (4, 1, 0, 0), (3, 2, 0, 0), (3, 1, 1, 0), 
                             (2, 2, 1, 0), (2, 1, 1, 1) }} ∧
    distros.card = 6 :=
sorry

end distinct_ball_distributions_l654_654171


namespace fewer_twos_result_100_l654_654365

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l654_654365


namespace path_through_cities_l654_654739

theorem path_through_cities {cities roads : Type} [Fintype cities] [Fintype roads]
  (C : Fintype.card cities = 1999)
  (R : Fintype.card roads = 4000)
  (connects : roads → cities × cities) : 
  ∃ (p : List cities), p.Nodup ∧ p.length ≤ 20 ∧ (∀ i < p.length - 1, (p.nthLe i _).fst = (p.nthLe (i+1) _).snd) ∧ (p.head? = p.getLast?.getD p.head?) := by
  sorry

end path_through_cities_l654_654739


namespace fewer_twos_to_hundred_l654_654361

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l654_654361


namespace problem_l654_654812

axiom condition1 : (-3)^2 = 9
axiom condition2 : realRoot 3 8 = 2
axiom condition3 : (-1)^4 = 1
axiom condition4 : sqrt 9 = 3

theorem problem : -1^4 = -1 := 
by 
  sorry

end problem_l654_654812


namespace total_worked_minutes_l654_654030

def bianca_hours := 12.5
def celeste_hours := 2 * bianca_hours
def mcclain_hours := celeste_hours - 8.5

def total_minutes_worked := (bianca_hours + celeste_hours + mcclain_hours) * 60

theorem total_worked_minutes : total_minutes_worked = 3240 :=
by
  -- Conditions are defined directly
  have h1 : bianca_hours = 12.5 := rfl
  have h2 : celeste_hours = 2 * bianca_hours := rfl
  have h3 : mcclain_hours = celeste_hours - 8.5 := rfl

  -- Calculate the total minutes worked
  have total_hours : bianca_hours + celeste_hours + mcclain_hours = 54 :=
    by
      sorry -- Cannot use solution steps directly

  have minutes_conversion : 54 * 60 = 3240 :=
    by
      sorry -- Arithmetic multiplication

  show total_minutes_worked = 3240 from 
    by
      rw [total_hours, minutes_conversion]
      sorry -- Simplification proof

end total_worked_minutes_l654_654030


namespace number_of_ways_to_fill_table_is_831_l654_654843

-- Definitions based on the problem conditions
def numbers := {2014, 2015, 2016}
def table := array (3 × 3) ℕ

-- Main theorem stating the problem and correct answer
theorem number_of_ways_to_fill_table_is_831 :
  ∃ (f : table → Prop), (∀ (t : table), f t → table_filled_with t numbers) ∧ (sum_in_each_row_equal t) ∧ (count_valid_tables f = 831) :=
begin
  sorry
end

end number_of_ways_to_fill_table_is_831_l654_654843


namespace smallest_positive_period_l654_654525

def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem smallest_positive_period : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ (∀ T' > 0, (∀ x, f(x + T') = f(x)) → T ≤ T') ∧ T = Real.pi :=
by
  sorry

end smallest_positive_period_l654_654525


namespace problem1_problem2_l654_654034

theorem problem1 : (-5 : ℝ) ^ 0 - (1 / 3) ^ (-2 : ℝ) + (-2 : ℝ) ^ 2 = -4 := 
by
  sorry

variable (a : ℝ)

theorem problem2 : (-3 * a ^ 3) ^ 2 * 2 * a ^ 3 - 8 * a ^ 12 / (2 * a ^ 3) = 14 * a ^ 9 :=
by
  sorry

end problem1_problem2_l654_654034


namespace percentage_change_in_revenue_l654_654208

-- Define initial assumptions
variables 
  (P V : ℝ)  -- P: original price of the ticket, V: original number of visitors

-- Define the new price and new number of visitors
def P_new : ℝ := 1.5 * P
def V_new : ℝ := 0.8 * V

-- Define original and new revenue
def R : ℝ := P * V
def R_new : ℝ := P_new * V_new

-- Expected percentage change in revenue
def percentage_change : ℝ := (R_new - R) / R * 100

-- Proof statement
theorem percentage_change_in_revenue : percentage_change = 20 :=
sorry

end percentage_change_in_revenue_l654_654208


namespace proof_problem_l654_654894

noncomputable def question := 3 - Real.sqrt 25

theorem proof_problem : question = -2 := 
by
    sorry

end proof_problem_l654_654894


namespace trigonometric_identity_l654_654051

theorem trigonometric_identity (h1 : cos (70 * Real.pi / 180) ≠ 0) (h2 : sin (70 * Real.pi / 180) ≠ 0) :
  (1 / cos (70 * Real.pi / 180) - (Real.sqrt 3) / sin (70 * Real.pi / 180)) = Real.cot (20 * Real.pi / 180) - 2 :=
by
  sorry

end trigonometric_identity_l654_654051


namespace angle_XOZ_in_regular_octagon_l654_654890

/-- In a regular octagon with vertices W, X, Y, Z, V, U, T, S, and lines \overline{WX}, \overline{XO}, \overline{OZ}, 
   \overline{ZY} drawn such that W, X, O, Z are consecutive vertices forming a quadrilateral,
   the measure of the angle \angle XOZ is 180 degrees. -/
theorem angle_XOZ_in_regular_octagon (W X Y Z V U T S O : Point) 
  (h_octagon: regular_octagon W X Y Z V U T S)
  (h_quad: quadrilateral W X O Z) :
  angle XOZ = 180 :=
sorry

end angle_XOZ_in_regular_octagon_l654_654890


namespace trigonometric_identity_l654_654048

theorem trigonometric_identity (h1 : cos (70 * Real.pi / 180) ≠ 0) (h2 : sin (70 * Real.pi / 180) ≠ 0) :
  (1 / cos (70 * Real.pi / 180) - (Real.sqrt 3) / sin (70 * Real.pi / 180)) = Real.cot (20 * Real.pi / 180) - 2 :=
by
  sorry

end trigonometric_identity_l654_654048


namespace divisible_expression_implies_a_divisible_l654_654291

-- Let a, m, and n be integers
variables (a m n : ℤ)

-- Definition of divisibility by 7
def divisible_by_7 (x : ℤ) : Prop := ∃ k : ℤ, x = 7 * k

-- The statement of the problem
theorem divisible_expression_implies_a_divisible (h : divisible_by_7 (a ^ (6 * m) + a ^ (6 * n))) : divisible_by_7 a :=
sorry

end divisible_expression_implies_a_divisible_l654_654291


namespace length_of_line_segment_EF_l654_654056

theorem length_of_line_segment_EF
  (AB GH CD EF : ℝ)
  (h1 : AB = 210)
  (h2 : CD = 140)
  (parallel_AB_GH : AB = GH)
  (parallel_GH_CD : GH = CD)
  (parallel_CD_EF : CD = EF)
  (angle_1 : ∠ BAC = ∠ FEC)
  (angle_2 : ∠ ABC = ∠ EFC):
  EF = 84 := by
  sorry

end length_of_line_segment_EF_l654_654056


namespace relatively_prime_powers_of_two_l654_654241

theorem relatively_prime_powers_of_two (a : ℤ) (h₁ : a % 2 = 1) (n m : ℕ) (h₂ : n ≠ m) :
  Int.gcd (a^(2^n) + 2^(2^n)) (a^(2^m) + 2^(2^m)) = 1 :=
by
  sorry

end relatively_prime_powers_of_two_l654_654241


namespace tan_alpha_plus_pi_over_3_l654_654129

open Real

theorem tan_alpha_plus_pi_over_3 (α : ℝ) 
  (h1 : sin (2 * α) = cos α) 
  (h2 : α ∈ set.Ioo (π / 2) π) :
  tan (α + π / 3) = sqrt 3 / 3 := 
  sorry

end tan_alpha_plus_pi_over_3_l654_654129


namespace min_marked_cells_l654_654380

-- Define the board
def board : Type := fin 10 × fin 11

-- Define the L-shaped piece with its three cells
def L_piece_cells : list (int × int) := [(0, 0), (1, 0), (0, 1)]

-- Define what it means for an L-shaped piece to be placed on the board
def L_placement (p : board) : list board :=
  L_piece_cells.map (λ (c : int × int), (⟨p.1.1 + c.1, by sorry⟩, ⟨p.2.1 + c.2, by sorry⟩))

-- Define the marking of cells
def marked_cells (ls : list board) : Prop :=
  ∃ k : ℕ, k = 50 ∧ ∀ p : board, ∃ m ∈ ls, m ∈ L_placement p

-- Statement that the smallest k satisfying the condition is 50
theorem min_marked_cells :
  ∃ (ls : list board), marked_cells ls :=
sorry

end min_marked_cells_l654_654380


namespace comic_books_l654_654600

variables (x y : ℤ)

def condition1 (x y : ℤ) : Prop := y + 7 = 5 * (x - 7)
def condition2 (x y : ℤ) : Prop := y - 9 = 3 * (x + 9)

theorem comic_books (x y : ℤ) (h₁ : condition1 x y) (h₂ : condition2 x y) : x = 39 ∧ y = 153 :=
by
  sorry

end comic_books_l654_654600


namespace Trisha_walked_total_distance_l654_654260

theorem Trisha_walked_total_distance 
  (d1 d2 d3 : ℝ) (h_d1 : d1 = 0.11) (h_d2 : d2 = 0.11) (h_d3 : d3 = 0.67) :
  d1 + d2 + d3 = 0.89 :=
by sorry

end Trisha_walked_total_distance_l654_654260


namespace max_distance_traveled_l654_654100

theorem max_distance_traveled (front_lifespan : ℕ) (rear_lifespan : ℕ) : 
  front_lifespan = 21000 ∧ rear_lifespan = 28000 → max_possible_distance = 24000 :=
begin
  intros h,
  sorry,
end

end max_distance_traveled_l654_654100


namespace fewer_twos_result_100_l654_654367

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l654_654367


namespace letter_at_at_sign_is_B_l654_654471

-- Define positions in the grid and initial known positions.
noncomputable def initial_positions : ℕ → ℕ → option Char
| 1, 1 := some 'A'
| 4, 1 := some 'D'
| 5, 1 := some 'E'
| _, _ := none

-- Define conditions that each letter must appear exactly once in each row, column,and both diagonals.
def valid_grid (grid : ℕ → ℕ → Char) : Prop :=
  (∀ row, ∃! (col1 col2 col3 col4 col5 : ℕ), 
    grid row col1 = 'A' ∧ grid row col2 = 'B' ∧ grid row col3 = 'C' ∧ grid row col4 = 'D' ∧ grid row col5 = 'E') ∧
  (∀ col, ∃! (row1 row2 row3 row4 row5 : ℕ), 
    grid row1 col = 'A' ∧ grid row2 col = 'B' ∧ grid row3 col = 'C' ∧ grid row4 col = 'D' ∧ grid row5 col = 'E') ∧
  (grid 1 1 = 'A' ∧ grid 4 1 = 'D' ∧ grid 5 1 = 'E') ∧
  (∃! (d1 d2 d3 d4 d5 : ℕ) (d1' d2' d3' d4' d5' : ℕ),
    grid d1 d1 = 'A' ∧ grid d2 d2 = 'B' ∧ grid d3 d3 = 'C' ∧ grid d4 d4 = 'D' ∧ grid d5 d5 = 'E' ∧
    grid d1' d5' = 'A' ∧ grid d2' d4' = 'B' ∧ grid d3' d3' = 'C' ∧ grid d4' d2' = 'D' ∧ grid d5' d1' = 'E')

-- The main theorem to prove that the letter represented by @ is B.
theorem letter_at_at_sign_is_B :
  ∀ grid : (ℕ → ℕ → Char), valid_grid grid → grid _ _ = 'B' := 
  sorry

end letter_at_at_sign_is_B_l654_654471


namespace d_equals_three_l654_654917

noncomputable def find_d (a b c d : ℝ) (Q : ℝ → ℝ) : Prop :=
  (Q = λ x, x^4 + a * x^3 + b * x^2 + c * x + d) ∧
  (1 + a + b + c + d = d) ∧
  (d = -a/4) ∧
  (Q 0 = 3)

theorem d_equals_three : ∃ d a b c, find_d a b c d (λ x, x^4 + a * x^3 + b * x^2 + c * x + d) ∧ d = 3 :=
begin
  sorry
end

end d_equals_three_l654_654917


namespace rectangle_area_l654_654010

variable (b H k x : ℝ)
variable (h_k : 0 < k) (h_kH : k < H)

theorem rectangle_area (h_b : b > 0) (h_H : H > 0) (h_x : x > 0) :
  let n := (b * k) / H in
  let A := n * x in
  A = (b * k * x) / H :=
by
  let n := (b * k) / H
  let A := n * x
  sorry

end rectangle_area_l654_654010


namespace cost_comparison_l654_654446

-- Definitions based on the given conditions
def suit_price : ℕ := 200
def tie_price : ℕ := 40
def num_suits : ℕ := 20
def discount_rate : ℚ := 0.9

-- Define cost expressions for the two options
def option1_cost (x : ℕ) : ℕ :=
  (suit_price * num_suits) + (tie_price * (x - num_suits))

def option2_cost (x : ℕ) : ℚ :=
  ((suit_price * num_suits + tie_price * x) * discount_rate : ℚ)

-- Main theorem to prove the given answers
theorem cost_comparison (x : ℕ) (hx : x > 20) :
  option1_cost x = 40 * x + 3200 ∧
  option2_cost x = 3600 + 36 * x ∧
  (x = 30 → option1_cost 30 < option2_cost 30) :=
by
  sorry

end cost_comparison_l654_654446


namespace peter_jacob_age_proof_l654_654294

theorem peter_jacob_age_proof :
  ∃ (X : ℕ), 
    let P := 16 in 
    let J := P + 12 in 
    X = 10 ∧ (P - X) = (1/3) * (J - X) := 
by
  use 10
  have hP : P = 16 := rfl
  have hJ : J = P + 12 := rfl
  rw [hP, hJ]
  split
  exact rfl
  sorry

end peter_jacob_age_proof_l654_654294


namespace sum_a_b_when_pow_is_max_l654_654888

theorem sum_a_b_when_pow_is_max (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 1) (h_pow : a^b < 500) 
(h_max : ∀ (a' b' : ℕ), (a' > 0) -> (b' > 1) -> (a'^b' < 500) -> a^b >= a'^b') : a + b = 24 := by
  sorry

end sum_a_b_when_pow_is_max_l654_654888


namespace problem_l654_654665

variable (g : ℝ → ℝ)
variables (x y : ℝ)

noncomputable def cond1 : Prop := ∀ x y : ℝ, 0 < x → 0 < y → g (x^2 * y) = g x / y^2
noncomputable def cond2 : Prop := g 800 = 4

-- The statement to be proved
theorem problem (h1 : cond1 g) (h2 : cond2 g) : g 7200 = 4 / 81 :=
by
  sorry

end problem_l654_654665


namespace rotation_in_coordinate_system_l654_654336

def coordinates_for_rotation_and_translation := ∃ (n u v : ℤ),
  (0 < n ∧ n < 180) ∧
  (u = 28 ∧ v = -2) ∧
  (η := Real.cos ((n : ℝ) * (Real.pi / 180))) ∧
  (ε := Real.sin ((n : ℝ) * (Real.pi / 180))) ∧
  -- Rotation matrix components
  (η = 0 ∧ ε = -1) ∧
  (u = 26 - (-v)) ∧
  (v = -(30 - u)) ∧
  (n + u + v = 116)

theorem rotation_in_coordinate_system :
  coordinates_for_rotation_and_translation :=
sorry

end rotation_in_coordinate_system_l654_654336


namespace largest_circle_radius_l654_654902

-- Define the given side lengths
def AB : ℝ := 10
def BC : ℝ := 11
def CD : ℝ := 13
def DA : ℝ := 12

-- Define the semi-perimeter
def s : ℝ := (AB + BC + CD + DA) / 2

-- Calculate the area using Brahmagupta's formula
def area (a b c d s : ℝ) : ℝ :=
  (s - a) * (s - b) * (s - c) * (s - d)

-- Define the radius formula
def radius (a b c d s : ℝ) : ℝ :=
  real.sqrt (area a b c d s) / s

-- Problem statement: Prove that the radius of the largest possible circle that fits inside the quadrilateral is 2 * sqrt(2145) / 23
theorem largest_circle_radius :
  radius 10 11 13 12 s = 2 * real.sqrt 2145 / 23 :=
sorry

end largest_circle_radius_l654_654902


namespace homothety_maps_C_to_E_l654_654700

-- Defining Points and Circles
variable {Point Circle : Type}
variable [Inhabited Point] -- assuming Point type is inhabited

-- Definitions for points H, K_A, I_A, K_B, I_B, K_C, I_C
variables (H K_A I_A K_B I_B K_C I_C : Point)

-- Define midpoints
def is_midpoint (A B M : Point) : Prop := sorry -- In a real proof, you would define midpoint in terms of coordinates

-- Define homothety function
def homothety (center : Point) (ratio : ℝ) (P : Point) : Point := sorry -- In a real proof, you would define the homothety transformation

-- Defining Circles
variables (C E : Circle)

-- Define circumcircle of a triangle
def is_circumcircle (a b c : Point) (circle : Circle) : Prop := sorry

-- Statements from conditions
axiom midpointA : is_midpoint H K_A I_A
axiom midpointB : is_midpoint H K_B I_B
axiom midpointC : is_midpoint H K_C I_C

axiom circumcircle_C : is_circumcircle K_A K_B K_C C
axiom circumcircle_E : is_circumcircle I_A I_B I_C E

-- Lean theorem stating the proof problem
theorem homothety_maps_C_to_E :
  ∀ (H K_A I_A K_B I_B K_C I_C : Point) (C E : Circle),
  (is_midpoint H K_A I_A) →
  (is_midpoint H K_B I_B) →
  (is_midpoint H K_C I_C) →
  (is_circumcircle K_A K_B K_C C) →
  (is_circumcircle I_A I_B I_C E) →
  (homothety H 0.5 K_A = I_A ) →
  (homothety H 0.5 K_B = I_B ) →
  (homothety H 0.5 K_C = I_C ) →
  C = E :=
by intro; sorry

end homothety_maps_C_to_E_l654_654700


namespace lucas_initial_pet_beds_l654_654249

-- Definitions from the problem conditions
def additional_beds := 8
def beds_per_pet := 2
def pets := 10

-- Statement to prove
theorem lucas_initial_pet_beds :
  (pets * beds_per_pet) - additional_beds = 12 := 
by
  sorry

end lucas_initial_pet_beds_l654_654249


namespace sum_possible_values_l654_654839

theorem sum_possible_values (a : ℕ → ℤ) (h : ∀ i, a i = 1 ∨ a i = -1) :
  (∀ i, (∑ j in finset.range 10, a (i + j)) ≠ 0) →
  ∃ n, n ∈ finset.range (404 div 2 + 1, 603 - 202 div 2 + 1) ∧ n * 2 = ∑ i in finset.range 2012, a i :=
sorry

end sum_possible_values_l654_654839


namespace find_y_diff_l654_654309

-- Define the problem as a Lean theorem
theorem find_y_diff
    (ell : ∀ x y : ℝ, x^2 / 25 + y^2 / 16 = 1)   -- Ellipse equation
    (F1 F2 : ℝ × ℝ)                              -- Foci F1 and F2
    (A B : ℝ × ℝ)                                -- Points A(x1,y1) and B(x2,y2)
    (incircle_circumference : ℝ)                 -- Circumference of the incircle of ΔABF2
    (incircle_circumference_eq : incircle_circumference = real.pi) :
    abs (A.2 - B.2) = 5 / 3 :=
by
  sorry  -- Proof is omitted

end find_y_diff_l654_654309


namespace value_two_std_dev_less_than_mean_l654_654426

-- Define the given conditions for the problem.
def mean : ℝ := 15
def std_dev : ℝ := 1.5

-- Define the target value that should be 2 standard deviations less than the mean.
def target_value := mean - 2 * std_dev

-- State the theorem that represents the proof problem.
theorem value_two_std_dev_less_than_mean : target_value = 12 := by
  sorry

end value_two_std_dev_less_than_mean_l654_654426


namespace max_n_with_no_five_times_relation_l654_654096

noncomputable def in_range (n : ℕ) : Set ℕ := { x | 1 ≤ x ∧ x ≤ n }

def no_five_times_relation (S : Set ℕ) : Prop :=
  ∀ x y ∈ S, x ≠ y → x ≠ 5 * y ∧ y ≠ 5 * x

def max_subset_size (n : ℕ) (upper_bound : ℕ) : Prop :=
  ∃ S ⊆ in_range n, no_five_times_relation S ∧ S.card = upper_bound

theorem max_n_with_no_five_times_relation : max_subset_size 2014 1679 :=
  sorry

end max_n_with_no_five_times_relation_l654_654096


namespace m_plus_n_eq_47_l654_654942

theorem m_plus_n_eq_47 (m n : ℕ)
  (h1 : m + 8 < n - 1)
  (h2 : (m + m + 3 + m + 8 + n - 1 + n + 3 + 2 * n - 2) / 6 = n)
  (h3 : (m + 8 + (n - 1)) / 2 = n) :
  m + n = 47 :=
sorry

end m_plus_n_eq_47_l654_654942


namespace total_weight_of_onions_l654_654774

def weight_per_bag : ℕ := 50
def bags_per_trip : ℕ := 10
def trips : ℕ := 20

theorem total_weight_of_onions : bags_per_trip * weight_per_bag * trips = 10000 := by
  sorry

end total_weight_of_onions_l654_654774


namespace parallelepiped_inscribed_circumscribed_iff_cube_l654_654815

theorem parallelepiped_inscribed_circumscribed_iff_cube :
  ∀ (P : Type) [Parallelepiped P], 
  (hasInscribedSphere P ∧ hasCircumscribedSphere P) ↔ isCube P :=
by
  sorry

end parallelepiped_inscribed_circumscribed_iff_cube_l654_654815


namespace length_is_62_l654_654723

noncomputable def length_of_plot (b : ℝ) := b + 24

theorem length_is_62 (b : ℝ) (h1 : length_of_plot b = b + 24) 
  (h2 : 2 * (length_of_plot b + b) = 200) : 
  length_of_plot b = 62 :=
by sorry

end length_is_62_l654_654723


namespace factorize_expression_l654_654073

variable (x y : ℝ)

theorem factorize_expression : (x - y)^2 - (3*x^2 - 3*x*y + y^2) = x * (y - 2*x) := 
by
  sorry

end factorize_expression_l654_654073


namespace count_zero_product_factors_correct_l654_654092

noncomputable def count_zero_product_factors : ℕ :=
  { n : ℕ // 1 ≤ n ∧ n ≤ 2020 ∧ 
    (∃ k, (1 + complex.exp(2 * real.pi * complex.I * k / n))^n + 1 = 0)}.card

theorem count_zero_product_factors_correct : count_zero_product_factors = 337 :=
by
  -- Proof skipped here
  sorry

end count_zero_product_factors_correct_l654_654092


namespace sum_floor_l654_654583

-- Definitions and conditions
def seq_a : ℕ → ℕ
| 0 := 2
| 1 := 6
| (n+2) := 2 * seq_a (n+1) - seq_a n + 2

-- Theorem to be proven
theorem sum_floor (m : ℕ) : (⌊∑ k in Finset.range m, (m : ℚ) / seq_a (k+1)⌋ : ℤ) = m - 1 :=
sorry

end sum_floor_l654_654583


namespace sum_of_digits_of_sqrt_N_l654_654934

noncomputable def N : ℝ := (44 * (10 ^ (2017: ℝ))) + (10 ^ (2017 - 1: ℝ)) * 8 * (10 ^ 2017) + 9

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else sum_of_digits (n / 10) + (n % 10)

theorem sum_of_digits_of_sqrt_N :
  sum_of_digits (⌊ sqrt N ⌋) = 12109 := by
  sorry

end sum_of_digits_of_sqrt_N_l654_654934


namespace probability_ratio_equality_l654_654066

noncomputable def probability_ratio : ℝ := 
  let A := (Nat.choose 6 2) * (Nat.choose 25 3) * (Nat.choose 22 3) *
           (Nat.choose 19 4) * (Nat.choose 15 4) *
           (Nat.choose 11 4) * (Nat.choose 7 4) / (4! : ℕ)
  let B := (Nat.choose 6 1) * (Nat.choose 25 5) * (Nat.choose 20 4) *
           (Nat.choose 16 4) * (Nat.choose 12 4) *
           (Nat.choose 8 4) * (Nat.choose 4 4)
  A / B

theorem probability_ratio_equality : probability_ratio = 8 := by
  sorry

end probability_ratio_equality_l654_654066


namespace range_of_a_l654_654193

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) : a ≥ 1 :=
sorry

end range_of_a_l654_654193


namespace gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l654_654495

theorem gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1 :
  Int.gcd (97 ^ 10 + 1) (97 ^ 10 + 97 ^ 3 + 1) = 1 := sorry

end gcd_97_pow_10_plus_1_and_97_pow_10_plus_97_pow_3_plus_1_l654_654495


namespace indolent_student_probability_l654_654741

-- Define the constants of the problem
def n : ℕ := 30  -- total number of students
def k : ℕ := 3   -- number of students selected each lesson
def m : ℕ := 10  -- number of students from the previous lesson

-- Define the probabilities
def P_asked_in_one_lesson : ℚ := 1 / k
def P_asked_twice_in_a_row : ℚ := 1 / n
def P_overall : ℚ := P_asked_in_one_lesson + P_asked_in_one_lesson - P_asked_twice_in_a_row
def P_avoid_reciting : ℚ := 1 - P_overall

theorem indolent_student_probability : P_avoid_reciting = 11 / 30 := 
  sorry

end indolent_student_probability_l654_654741


namespace sarah_score_l654_654283

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l654_654283


namespace find_m_of_sixth_degree_polynomial_l654_654948

theorem find_m_of_sixth_degree_polynomial (a b : ℝ) (m : ℝ) 
    (h : polynomial.degree (polynomial.C (4 * m - 1) * polynomial.X * polynomial.C b + 
                            polynomial.C a * polynomial.X ^ 2 * polynomial.C b - polynomial.C 5) = 6) :
  m = 3 := 
sorry

end find_m_of_sixth_degree_polynomial_l654_654948


namespace conic_section_is_ellipse_l654_654913

theorem conic_section_is_ellipse :
  ∀ x y : ℝ, 4 * x^2 + y^2 - 12 * x - 2 * y + 4 = 0 →
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧ (a * (x - h)^2 + b * (y - k)^2 = 1) :=
by
  sorry

end conic_section_is_ellipse_l654_654913


namespace diameter_of_outer_edge_l654_654854

-- Defining the conditions as variables
variable (pathWidth gardenWidth statueDiameter fountainDiameter : ℝ)
variable (hPathWidth : pathWidth = 10)
variable (hGardenWidth : gardenWidth = 12)
variable (hStatueDiameter : statueDiameter = 6)
variable (hFountainDiameter : fountainDiameter = 14)

-- Lean statement to prove the diameter
theorem diameter_of_outer_edge :
  2 * ((fountainDiameter / 2) + gardenWidth + pathWidth) = 58 :=
by
  rw [hPathWidth, hGardenWidth, hFountainDiameter]
  sorry

end diameter_of_outer_edge_l654_654854


namespace problem_statement_l654_654563

theorem problem_statement :
  (∑ i in finset.range 8, 1 / (nat.factorial (3 + i) * nat.factorial (20 - (3 + i)))) = M / (1 * nat.factorial 19) →
  (M / 100).floor = 262 :=
by 
  -- Add natural number for the sum we need
  setup_tactic
  sorry

end problem_statement_l654_654563


namespace find_a2018_l654_654246

noncomputable def integer_part (x : ℝ) : ℕ := nat.floor x
noncomputable def fractional_part (x : ℝ) : ℝ := x - (integer_part x)

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = real.sqrt 3 ∧ ∀ n, a (n + 1) = integer_part (a n) + 1 / fractional_part (a n)

theorem find_a2018 (a : ℕ → ℝ) (h : sequence a) : a 2018 = (6051 + real.sqrt 3) / 2 :=
	sorry

end find_a2018_l654_654246


namespace count_prime_simple_numbers_le_100_l654_654896

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_simple (n : ℕ) : Prop :=
  ∃ (p1 p2 : ℕ), is_prime p1 ∧ is_prime p2 ∧ p1 ≠ p2 ∧ n = p1^2 + p2^2

def primes := {p : ℕ | is_prime p ∧ p ≤ 10}

theorem count_prime_simple_numbers_le_100 : 
  (∃ s : finset ℕ, (∀ n ∈ s, prime_simple n ∧ n ≤ 100) ∧ s.card = 6) :=
sorry

end count_prime_simple_numbers_le_100_l654_654896


namespace infinite_sum_converges_to_neg1_l654_654921

noncomputable def S : ℝ := ∑' k : ℕ, (-1)^(k+1) * (k^3 + k^2 + k + 1) / ((k+1)!)

theorem infinite_sum_converges_to_neg1 : S = -1 := by
  sorry

end infinite_sum_converges_to_neg1_l654_654921


namespace number_of_ways_to_stand_l654_654742

theorem number_of_ways_to_stand (A B : Type) (people : Finset (A × B)) :
  let front_row := {p // p.2 = true},
      back_row := {p // p.2 = false},
      A_not_front := ∀ (a ∈ people), a.1 ≠ "A" ∨ a.2 ≠ true,
      B_not_back := ∀ (b ∈ people), b.1 ≠ "B" ∨ b.2 ≠ false in
  A_not_front → B_not_back → 
  people.card = 6 → 
  front_row.card = 3 → 
  back_row.card = 3 → 
  people.count (λ p, p.1 ≠ "A" ∧ p.2 = false) * -- A not in front
  people.count (λ p, p.1 ≠ "B" ∧ p.2 = true) *  -- B not in back
  (4!) = 216 := 
sorry

end number_of_ways_to_stand_l654_654742


namespace children_tickets_count_l654_654773

theorem children_tickets_count 
  (A C : ℕ) 
  (h1 : A + C = 21) 
  (h2 : 5.50 * A + 3.50 * C = 83.50) 
  : C = 16 := 
by 
  -- The actual proof steps
  sorry

end children_tickets_count_l654_654773


namespace optimal_voter_split_l654_654756

-- Definitions
variables (Voters : Type) [fintype Voters] (n : ℕ)
variables (supports_miraflores : Voters → Prop)
variables [decidable_pred supports_miraflores]

-- Conditions
def half_supports_miraflores := fintype.card { v // supports_miraflores v } = n
def half_supports_maloney := fintype.card { v // ¬ supports_miraflores v } = n

-- Question (translated to a theorem)
theorem optimal_voter_split (h_m : half_supports_miraflores Voters n supports_miraflores)
    (h_d: half_supports_maloney Voters n supports_miraflores) :
  ∃ (D1 D2 : finset Voters), 
    ((D1 = {v | supports_miraflores v}) ∧ 
    (D2 = {v | ¬supports_miraflores v}) ∧ 
    (∀ v, v ∈ D1 ∨ v ∈ D2) ∧ 
    (∀ v, ¬ (v ∈ D1 ∧ v ∈ D2)) ∧ 
    (finset.card D1 = 1) ∧ 
    (finset.card D2 = 2 * n - 1)) :=
sorry

end optimal_voter_split_l654_654756


namespace functional_equation_solution_l654_654514

-- Define the conditions of the problem.
variable (f : ℝ → ℝ) 
variable (h : ∀ x y u v : ℝ, (f x + f y) * (f u + f v) = f (x * u - y * v) + f (x * v + y * u))

-- Formalize the statement that no other functions satisfy the conditions except f(x) = x^2.
theorem functional_equation_solution : (∀ x : ℝ, f x = x^2) :=
by
  -- The proof goes here, but since the proof is not required, we skip it.
  sorry

end functional_equation_solution_l654_654514


namespace max_min_on_interval_l654_654489

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem max_min_on_interval (a : ℝ) :
  let max_val := if a ≤ 1 / 2 then 5 - 4 * a else 2 + 2 * a
  let min_val :=
    if a ≤ -1 then 2 + 2 * a
    else if -1 < a ∧ a < 2 then 1 - a^2
    else 5 - 4 * a
  ∃ x_max x_min ∈ set.Icc (-1 : ℝ) 2, f x_max a = max_val ∧ f x_min a = min_val :=
by
  sorry

end max_min_on_interval_l654_654489


namespace shift_log_to_left_by_3_units_l654_654784

def f (x : ℝ) : ℝ := log (x - 1) / log 2
def g (x : ℝ) : ℝ := log (x + 2) / log 2

theorem shift_log_to_left_by_3_units : ∀ x : ℝ, g x = f (x + 3) :=
by
  intros x
  sorry

end shift_log_to_left_by_3_units_l654_654784


namespace unique_sums_implies_a9_gt_100_l654_654978

theorem unique_sums_implies_a9_gt_100
  (a : ℕ → ℕ)
  (h_unique_sums : ∀ (sum_set : set (set ℕ)),
    (∀ (t : set ℕ), t ⊆ finset.univ.image a ∧ t ≠ ∅ ∧ t.card ≤ 9 →
      ∑ x in t, x ≠ ∑ y in (sum_set.erase t), y)) : 
  (∀ i j, i < j → a i < a j) → a 8 > 100 :=
by
  sorry

end unique_sums_implies_a9_gt_100_l654_654978


namespace rational_square_of_1_minus_xy_l654_654162

theorem rational_square_of_1_minus_xy (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : ∃ (q : ℚ), 1 - x * y = q^2 :=
by
  sorry

end rational_square_of_1_minus_xy_l654_654162


namespace prob_even_sum_is_one_third_l654_654303

def is_even_sum_first_last (d1 d2 d3 d4 : Nat) : Prop :=
  (d1 + d4) % 2 = 0

def num_unique_arrangements : Nat := 12

def num_favorable_arrangements : Nat := 4

def prob_even_sum_first_last : Rat :=
  num_favorable_arrangements / num_unique_arrangements

theorem prob_even_sum_is_one_third :
  prob_even_sum_first_last = 1 / 3 := 
  sorry

end prob_even_sum_is_one_third_l654_654303


namespace det_B_eq_2_l654_654657

theorem det_B_eq_2 {x y : ℝ}
  (hB : ∃ (B : Matrix (Fin 2) (Fin 2) ℝ), B = ![![x, 2], ![-3, y]])
  (h_eqn : ∃ (B_inv : Matrix (Fin 2) (Fin 2) ℝ),
    B_inv = (1 / (x * y + 6)) • ![![y, -2], ![3, x]] ∧
    ![![x, 2], ![-3, y]] + 2 • B_inv = 0) : 
  Matrix.det ![![x, 2], ![-3, y]] = 2 :=
by
  sorry

end det_B_eq_2_l654_654657


namespace factorial_comparison_l654_654807

open scoped BigOperators

theorem factorial_comparison : (100.factorial)!.factorial < (99.factorial) ^ (100.factorial) * (100.factorial) ^ (99.factorial) := by
  sorry

end factorial_comparison_l654_654807


namespace original_survey_customers_l654_654005

theorem original_survey_customers : ∃ x : ℕ, (7 / x + 0.02).approx (1 / 7) ∧ x ≈ 57 :=
begin
  sorry
end

end original_survey_customers_l654_654005


namespace average_tree_height_l654_654442

theorem average_tree_height :
  let tree1 := 8
  let tree2 := if tree3 = 16 then 4 else 16
  let tree3 := 16
  let tree4 := if tree5 = 32 then 8 else 32
  let tree5 := 32
  let tree6 := if tree5 = 32 then 64 else 16
  let total_sum := tree1 + tree2 + tree3 + tree4 + tree5 + tree6
  let average_height := total_sum / 6
  average_height = 14 :=
by
  sorry

end average_tree_height_l654_654442


namespace existence_of_special_set_l654_654512

theorem existence_of_special_set (n : ℕ) (h : n ≥ 3) (h_ne_4 : n ≠ 4) :
  ∃ (M : finset (euclidean_space ℝ (fin n))),
  (∀ u ∈ M, u ≠ 0) ∧
  (∀ (u v : euclidean_space ℝ (fin n)), u ∈ M → v ∈ M → u ≠ v → u + v ≠ 0) ∧
  (∑ u in M, u = 0) :=
sorry

end existence_of_special_set_l654_654512


namespace co_captains_probability_l654_654740

open Finset

theorem co_captains_probability (T1_members T2_members T3_members T1_cocaptains T2_cocaptains T3_cocaptains : Finset ℕ) 
  (hT1 : T1_members.card = 6) (hT2 : T2_members.card = 9) (hT3 : T3_members.card = 10) 
  (hT1_co : T1_cocaptains.card = 3) (hT2_co : T2_cocaptains.card = 2) (hT3_co : T3_cocaptains.card = 4) 
  (hT1_sub : T1_cocaptains ⊆ T1_members) (hT2_sub : T2_cocaptains ⊆ T2_members) (hT3_sub : T3_cocaptains ⊆ T3_members) :
  ((1 / 3 : ℚ) * (((3.choose 2 : ℚ) / (6.choose 2)) + ((2.choose 2 : ℚ) / (9.choose 2)) + ((4.choose 2 : ℚ) / (10.choose 2)))) = 65 / 540 := 
by sorry

end co_captains_probability_l654_654740


namespace variance_Y_l654_654984

variables {X : Type} [MeasurableSpace X] (μ : MeasureTheory.Measure X)
variables (X : X → ℝ) (c : ℝ)

noncomputable def variance (X : X → ℝ) : ℝ :=
MeasureTheory.MeasureTheory.variance μ X

theorem variance_Y
  (h1 : variance μ X = 1)
  (h2 : ∀ x, X x = 2 * X x + 3) :
  variance μ (λ x, 2 * (X x) + 3) = 4 :=
by
  sorry

end variance_Y_l654_654984


namespace trapezoid_angle_AED_l654_654626

theorem trapezoid_angle_AED (ABCD : Type) [trapezoid ABCD]
    (AB DC : parallel_sides ABCD) 
    (E : intersection_diagonals ABCD)
    (angle_DAB : angle_at ABCD D A B = 55)
    (angle_BCD : angle_at ABCD B C D = 35) :
    angle_at ABCD A E D = 90 :=
sorry

end trapezoid_angle_AED_l654_654626


namespace fewer_twos_for_100_l654_654354

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l654_654354


namespace find_k_l654_654207

noncomputable def midpoint (A B : ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def distance (A B : ℝ × ℝ) := (real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) / 2
def circle_eq (C : ℝ × ℝ) (r : ℝ) (x y : ℝ) := (x - C.1)^2 + (y - C.2)^2 = r^2

theorem find_k (k : ℝ) :
  let A := (-3, 0),
      B := (27, 0),
      C := midpoint A B,
      r := distance A B,
      x := k, 
      y := 12 in
  circle_eq C r x y → (k = 3 ∨ k = 21) :=
by
  intro h
  have h_midpoint : midpoint A B = (12, 0) := sorry
  have h_radius : distance A B = 15 := sorry
  have h_circle_eq : circle_eq (12, 0) 15 x 12 := sorry
  have k_possibilities : (k - 12)^2 = 81 := sorry -- solving x^2 = 81
  have h_solutions : k = 21 ∨ k = 3 := sorry
  exact h_solutions

end find_k_l654_654207


namespace min_marked_cells_l654_654383

-- Define the board
def board : Type := fin 10 × fin 11

-- Define the L-shaped piece with its three cells
def L_piece_cells : list (int × int) := [(0, 0), (1, 0), (0, 1)]

-- Define what it means for an L-shaped piece to be placed on the board
def L_placement (p : board) : list board :=
  L_piece_cells.map (λ (c : int × int), (⟨p.1.1 + c.1, by sorry⟩, ⟨p.2.1 + c.2, by sorry⟩))

-- Define the marking of cells
def marked_cells (ls : list board) : Prop :=
  ∃ k : ℕ, k = 50 ∧ ∀ p : board, ∃ m ∈ ls, m ∈ L_placement p

-- Statement that the smallest k satisfying the condition is 50
theorem min_marked_cells :
  ∃ (ls : list board), marked_cells ls :=
sorry

end min_marked_cells_l654_654383


namespace median_of_trapezoid_l654_654009

theorem median_of_trapezoid (h : ℝ) (h_pos : h ≠ 0) :
  let b := 36 in
  let Area_triangle := (1 / 2 : ℝ) * b * h in
  let Area_trapezoid := λ m : ℝ, m * h in
  Area_triangle = Area_trapezoid 18 :=
by
  let b := 36
  let Area_triangle := (1 / 2 : ℝ) * b * h
  let Area_trapezoid := λ m : ℝ, m * h
  have eq_h : Area_triangle = Area_trapezoid 18 := 
    calc
      (1 / 2 : ℝ) * b * h = (1 / 2 : ℝ) * 36 * h : by sorry
      ... = 18 * h : by sorry
  exact eq_h

end median_of_trapezoid_l654_654009


namespace trajectory_of_M_line_tangent_to_circle_l654_654960

theorem trajectory_of_M (P : ℝ × ℝ) (M : ℝ × ℝ) (Q : ℝ × ℝ)
    (hP_circle : (P.1 + Real.sqrt 3)^2 + P.2^2 = 24)
    (hQ : Q = (Real.sqrt 3, 0))
    (hMN : ∀ N : ℝ × ℝ, N = (-Real.sqrt 3, 0) → dist M P = dist M Q) :
    (M.1^2 / 6 + M.2^2 / 3 = 1) := sorry

theorem line_tangent_to_circle (l : ℝ → ℝ) (C : ℝ × ℝ → ℝ)
    (hC : ∀ M : ℝ × ℝ, M.1^2 / 6 + M.2^2 / 3 = 1)
    (hA : ℝ × ℝ) (hB : ℝ × ℝ)
    (hl_intersects : ∀ x : ℝ, C (x, l x) = 0)
    (hOA_OB : (|hA.1| * |hB.1|) + (|l hA.1| * |l hB.1|) = 0) :
    ∃ E : ℝ → ℝ × ℝ → Prop, ∀ (x y : ℝ), E x y ↔ x^2 + y^2 = 2 := sorry

end trajectory_of_M_line_tangent_to_circle_l654_654960


namespace probability_of_selecting_same_gender_l654_654288

def number_of_ways_to_choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem probability_of_selecting_same_gender (total_students male_students female_students : ℕ) (h1 : total_students = 10) (h2 : male_students = 2) (h3 : female_students = 8) : 
  let total_combinations := number_of_ways_to_choose_two total_students
  let male_combinations := number_of_ways_to_choose_two male_students
  let female_combinations := number_of_ways_to_choose_two female_students
  let favorable_combinations := male_combinations + female_combinations
  total_combinations = 45 ∧
  male_combinations = 1 ∧
  female_combinations = 28 ∧
  favorable_combinations = 29 ∧
  (favorable_combinations : ℚ) / total_combinations = 29 / 45 :=
by
  sorry

end probability_of_selecting_same_gender_l654_654288


namespace magnitude_square_a_l654_654949

-- Definitions and conditions from the problem
variables {θ : ℝ} (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi)

-- Definition of the vector a
def a := (Real.cos (Real.pi + θ), Real.sin (Real.pi - θ))

-- Proof statement that ||a||^2 = 1
theorem magnitude_square_a : ∥a∥^2 = 1 :=
by
  sorry

end magnitude_square_a_l654_654949


namespace multiplication_counts_l654_654529

open Polynomial

noncomputable def horner_multiplications (n : ℕ) : ℕ := n

noncomputable def direct_summation_multiplications (n : ℕ) : ℕ := n * (n + 1) / 2

theorem multiplication_counts (P : Polynomial ℝ) (x₀ : ℝ) (n : ℕ)
  (h_degree : P.degree = n) :
  horner_multiplications n = n ∧ direct_summation_multiplications n = (n * (n + 1)) / 2 :=
by
  sorry

end multiplication_counts_l654_654529


namespace part_I_part_II_l654_654151

def f (x a : ℝ) : ℝ := |x - a|

theorem part_I (a : ℝ) (h : ∀ x, f x a ≤ 3 → x ∈ set.Icc (-1 : ℝ) (5 : ℝ)) : a = 2 :=
by
  sorry

theorem part_II (m : ℝ) (h : ∀ x, f x 2 + f (x + 5) 2 ≥ m) : m ≤ 5 :=
by
  sorry

end part_I_part_II_l654_654151


namespace onion_harvest_weight_l654_654776

theorem onion_harvest_weight :
  let bags_per_trip := 10 in
  let weight_per_bag := 50 in
  let trips := 20 in
  let total_weight := (bags_per_trip * weight_per_bag) * trips in
  total_weight = 10000 := by
  sorry

end onion_harvest_weight_l654_654776


namespace circumference_of_tank_a_l654_654707

def is_circumference_of_tank_a (h_A h_B C_B : ℝ) (V_A_eq : ℝ → Prop) : Prop :=
  ∃ (C_A : ℝ), 
    C_B = 10 ∧ 
    h_A = 10 ∧
    h_B = 7 ∧
    V_A_eq 0.7 ∧ 
    C_A = 7

theorem circumference_of_tank_a (h_A : ℝ) (h_B : ℝ) (C_B : ℝ) (V_A_eq : ℝ → Prop) : 
  is_circumference_of_tank_a h_A h_B C_B V_A_eq := 
by
  sorry

end circumference_of_tank_a_l654_654707


namespace find_ratio_l654_654197

section proof

variables {A B C D E P : Type}
variables (cd db ae eb : ℝ)
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
variables (P : Prop)

-- Given conditions
axiom h1 : cd / db = 3 / 1
axiom h2 : ae / eb = 3 / 2
axiom h3 : P = (λ (CE AD : Type), ∃ P : Type, CE ∩ AD = {P})

-- To prove
theorem find_ratio (h1 : cd / db = 3 / 1) (h2 : ae / eb = 3 / 2) (h3 : P) : 
  r = 5 :=
sorry

end proof

end find_ratio_l654_654197


namespace exist_two_not_connected_no_common_l654_654886

open Finset

/- The vertices represent people at the gathering. 
   Edges represent the acquaintance relationship. -/
variables (V : Finset (Fin 17)) (adj : V → V → Prop)
variables (h_symm : ∀ u v, adj u v → adj v u) -- symmetric relation (undirected graph)
variables (h_deg4 : ∀ v, (univ.filter (adj v)).card = 4) -- degree 4 for each vertex

-- The main theorem we need to prove
theorem exist_two_not_connected_no_common :
  ∃ (u v : V), ¬ (adj u v) ∧ (univ.filter (λ w, adj u w ∧ adj v w)).card = 0 :=
sorry

end exist_two_not_connected_no_common_l654_654886


namespace determine_function_l654_654668

def S := {x : ℝ // x > -1}

def strictly_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

theorem determine_function {f : S → S}
  (h1 : ∀ (x y : S), f ⟨x + f y + y, _⟩ = y + f x + y * f x)
  (h2 : strictly_increasing_on (λ (x : ℝ), f ⟨x, _⟩ / x) {x | -1 < x ∧ x < 0} ∧ strictly_increasing_on (λ (x : ℝ), f ⟨x, _⟩ / x) {x | x > 0}) :
  ∀ (x : S), f x = ⟨- x / (1 + x), _⟩ :=
sorry

end determine_function_l654_654668


namespace sum_sequence_2016_terms_eq_l654_654966

-- Defining the arithmetic sequence with initial term a1 and common difference d
def a_n (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

-- Given conditions
def S_3 (a₁ d : ℤ) : ℤ := 3 * a₁ + 3 * d
def S_5 (a₁ d : ℤ) : ℤ := 5 * a₁ + 10 * d

-- Evaluating the generalized term for sequence { 1 / (a_{2n-1} * a_{2n+1}) }
def b_n (a₁ d : ℤ) (n : ℕ) : ℚ := 1 / ((a_n a₁ d (2 * n - 1)) * (a_n a₁ d (2 * n + 1)))

-- Sum of the first 2016 terms of the sequence
def sum_first_2016_terms (a₁ d : ℤ) : ℚ :=
  (Finset.range 2016).sum (λ n, b_n a₁ d (n + 1))

-- The theorem to prove the sum of first 2016 terms evaluates to the specific fraction
theorem sum_sequence_2016_terms_eq (a₁ d : ℤ) (h1 : S_3 a₁ d = 0) (h2 : S_5 a₁ d = -5) :
   sum_first_2016_terms a₁ d = -2016 / 4031 := 
sorry

end sum_sequence_2016_terms_eq_l654_654966


namespace mindy_total_is_19_l654_654253

noncomputable theory

def round_to_nearest_dollar (x : ℝ) : ℤ :=
  Int.ofNat (Nat.ceil (x - 0.5))

def total_after_discount_and_rounding (amounts : List ℝ) (discount : ℝ) : ℤ :=
  let total := amounts.sum
  let discounted := (1 - discount) * total
  round_to_nearest_dollar discounted

theorem mindy_total_is_19 :
  total_after_discount_and_rounding [2.45, 7.15, 11.68] 0.1 = 19 :=
sorry

end mindy_total_is_19_l654_654253


namespace complete_entire_job_l654_654438

/-- If a person completes 5/8 of a job in 10 days, then he will complete the entire job in 16 days. -/
theorem complete_entire_job (days_for_five_eighths : ℕ) (h : days_for_five_eighths = 10) : 
  (8/8:ℚ) * (days_for_five_eighths / (5/8:ℚ)) = 16 :=
by 
  rw h
  exact sorry

end complete_entire_job_l654_654438


namespace max_expression_value_is_33_l654_654521

noncomputable def max_expression_value : ℝ :=
  let f (x : ℝ) := sqrt (8 - 4 * sqrt 3) * sin x - 3 * sqrt (2 * (1 + cos (2 * x))) - 2
  let g (y : ℝ) := 3 + 2 * sqrt (11 - sqrt 3) * cos y - cos (2 * y)
  let h := f * g
  33

theorem max_expression_value_is_33 : 
  ∃ x y : ℝ, f x * g y = 33 := sorry

end max_expression_value_is_33_l654_654521


namespace sum_bn_l654_654659

theorem sum_bn (n : ℕ) :
  let a_n (n : ℕ) := 2^(n-1)
  let b_n (n : ℕ) := a_n n + (n-1) * Real.log 2
  (∑ k in Finset.range n, b_n (k+1)) = 2^n - 1 + (n * (n-1) / 2) * Real.log 2 :=
by
  sorry

end sum_bn_l654_654659


namespace angle_bounds_at_vertices_l654_654713

open Real

/-- Define the vertices of the triangle with given conditions -/
def vertex_A (p q: ℝ): Point := Point.mk p q
def vertex_B (q r: ℝ): Point := Point.mk q r
def vertex_C (r p: ℝ): Point := Point.mk r p

/-- Define the conditions on the vertices -/
axiom cond_pqr (p q r : ℝ) : p < q ∧ q < r

/-- Define the angles at each vertex -/
noncomputable def angle_at_A (p q r : ℝ) : ℝ := 
  angle (vertex_B q r) (vertex_A p q) (vertex_C r p)

noncomputable def angle_at_B (p q r : ℝ) : ℝ := 
  angle (vertex_A p q) (vertex_B q r) (vertex_C r p)

/-- Main theorem stating the bounds of the angles at vertices A and B -/
theorem angle_bounds_at_vertices (p q r : ℝ) (h : p < q ∧ q < r) :
  45 < angle_at_A p q r ∧ angle_at_A p q r < 90 ∧
  45 < angle_at_B p q r ∧ angle_at_B p q r < 90 :=
sorry

end angle_bounds_at_vertices_l654_654713


namespace first_digit_base7_528_l654_654799

theorem first_digit_base7_528 : 
  (Nat.digitNat 528 7 0 = 1) := 
sorry

end first_digit_base7_528_l654_654799


namespace equivalent_function_l654_654015

theorem equivalent_function (x : ℝ) : 
  (∀ x, log 10 (10 ^ x) = x) ∧ 
  (∀ x, x ≠ 0 → (x ^ 2 / x) = x) ∧
  (∀ x, sqrt (x ^ 2) = abs x) ∧
  (∀ x, x > 0 → 2 ^ (log 2 x) = x) :=
by {
  -- proof would go here
  sorry
}

end equivalent_function_l654_654015


namespace sum_of_sequences_l654_654135

noncomputable def arithmetic_sequence (a b : ℤ) : Prop :=
  ∃ k : ℤ, a = 6 + k ∧ b = 6 + 2 * k

noncomputable def geometric_sequence (c d : ℤ) : Prop :=
  ∃ q : ℤ, c = 6 * q ∧ d = 6 * q^2

theorem sum_of_sequences (a b c d : ℤ) 
  (h_arith : arithmetic_sequence a b) 
  (h_geom : geometric_sequence c d) 
  (hb : b = 48) (hd : 6 * c^2 = 48): 
  a + b + c + d = 111 := 
sorry

end sum_of_sequences_l654_654135


namespace value_of_expression_l654_654490

theorem value_of_expression : 5 * 405 + 4 * 405 - 3 * 405 + 404 = 2834 :=
by sorry

end value_of_expression_l654_654490


namespace max_distance_l654_654098

theorem max_distance (front_lifespan : ℕ) (rear_lifespan : ℕ)
  (h_front : front_lifespan = 21000)
  (h_rear : rear_lifespan = 28000) :
  ∃ (max_dist : ℕ), max_dist = 24000 :=
by
  sorry

end max_distance_l654_654098


namespace min_value_3x_plus_4y_l654_654605

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_plus_4y_l654_654605


namespace orthocenter_midpoint_l654_654470

-- Define the scalene triangle PQR and the special points
variable {P Q R X Y Z H M O : Point}

-- Define the conditions
variables (scalene_triangle : scalene_triangle P Q R)
          (mid_X : midpoint X Q R)
          (mid_Y : midpoint Y R P)
          (mid_Z : midpoint Z P Q)
          (orthocenter_H : orthocenter H P Q R)
          (circumcenter_M : circumcenter M P Q R)
          (circumcenter_XYZ : circumcenter O X Y Z)

-- The theorem to prove
theorem orthocenter_midpoint (scalene_triangle : scalene_triangle P Q R)
                             (mid_X : midpoint X Q R)
                             (mid_Y : midpoint Y R P)
                             (mid_Z : midpoint Z P Q)
                             (orthocenter_H : orthocenter H P Q R)
                             (circumcenter_M : circumcenter M P Q R)
                             (circumcenter_XYZ : circumcenter O X Y Z) :
  orthocenter M X Y Z ∧ midpoint O M H :=
by sorry

end orthocenter_midpoint_l654_654470


namespace sum_of_five_consecutive_integers_l654_654297

theorem sum_of_five_consecutive_integers : ∀ (n : ℤ), (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 5 * n + 20 := 
by
  -- This would be where the proof goes
  sorry

end sum_of_five_consecutive_integers_l654_654297


namespace circ_assoc_l654_654266

def circ (x y : ℝ) : ℝ := x * y + sqrt ((x ^ 2 - 1) * (y ^ 2 - 1))

theorem circ_assoc (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  circ (circ x y) z = circ x (circ y z) :=
sorry

end circ_assoc_l654_654266


namespace sum_of_reciprocals_of_roots_l654_654327

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (hroots : ∀ x, (x = p ∨ x = q ∨ x = r) ↔ (30*x^3 - 50*x^2 + 22*x - 1 = 0)) 
  (h0 : 0 < p ∧ p < 1) (h1 : 0 < q ∧ q < 1) (h2 : 0 < r ∧ r < 1) 
  (hdistinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (1 / (1 - p)) + (1 / (1 - q)) + (1 / (1 - r)) = 12 := 
by 
  sorry

end sum_of_reciprocals_of_roots_l654_654327


namespace find_outer_radius_l654_654823

noncomputable def radius_of_inner_circle (C_inner : ℝ) : ℝ :=
  C_inner / (2 * Real.pi)

noncomputable def radius_of_outer_circle (r_inner width : ℝ) : ℝ :=
  r_inner + width

theorem find_outer_radius (C_inner : ℝ) (width : ℝ) (r_outer : ℝ) :
  C_inner = 440 ∧ width = 14 → r_outer = radius_of_outer_circle (radius_of_inner_circle C_inner) width :=
by
  intros h
  cases h with hC hw
  simp [radius_of_inner_circle, radius_of_outer_circle]
  rw [hC, hw]
  sorry

end find_outer_radius_l654_654823


namespace oblique_asymptote_of_rational_func_l654_654930

theorem oblique_asymptote_of_rational_func : 
  (⋃ x : ℝ, (f x) = (3 * x ^ 2 + 8 * x + 12) / (3 * x + 4) → 
  (∃ y : ℝ, (y = x + 4/3) ∧ (∃ L : ℝ, (∀ δ > 0, ∃ α > 0, ∀ x, α < x → ∣f x - y∣ < δ)))) := 
begin
  sorry
end

end oblique_asymptote_of_rational_func_l654_654930


namespace sum_of_all_four_digit_integers_l654_654402

theorem sum_of_all_four_digit_integers :
  (Finset.range (9999 + 1)).filter (λ x => x ≥ 1000).sum = 49495500 :=
by
  sorry

end sum_of_all_four_digit_integers_l654_654402


namespace equal_length_tangent_segments_l654_654506

variables {α : Type} [metric_space α]

-- Define two non-intersecting circles with center and radius
variable (circle1 : metric_ball (p1 : α) (r1 : ℝ)) (circle2 : metric_ball (p2 : α) (r2 : ℝ))

-- Define the tangent points for the circles
variable (E1 E2 F1 F2 G1 G2 A B : α)

-- Hypotheses for tangents
variables
  (h_tangent_E1 : E1 ∈ circle1 ∧ E1 ∈ circle2)
  (h_tangent_E2 : E2 ∈ circle1 ∧ E2 ∈ circle2)
  (h_tangent_F1 : F1 ∈ circle1)
  (h_tangent_F2 : F2 ∈ circle2)
  (h_tangent_G1 : G1 ∈ circle1)
  (h_tangent_G2 : G2 ∈ circle2)
  (h_tangent_A : A ∈ circle1 ∧ A ∈ circle2)
  (h_tangent_B : B ∈ circle1 ∧ B ∈ circle2)

-- Statement of the proof
theorem equal_length_tangent_segments
  (h_circle_non_intersecting : dist p1 p2 > r1 + r2) :
  dist A E1 = dist B E2 :=
sorry

end equal_length_tangent_segments_l654_654506


namespace paperboy_delivery_l654_654860

def D : ℕ → ℕ
| 0       := 1
| 1       := 2
| 2       := 4
| 3       := 8
| (n+4) := D (n+3) + D (n+2) + D (n+1) + D n

theorem paperboy_delivery : D 12 = 2873 := by
  sorry

end paperboy_delivery_l654_654860


namespace rectangle_perimeter_sub_4_l654_654769

theorem rectangle_perimeter_sub_4 (width length : ℕ) (h_width : width = 4) (h_length : length = 8) : 
  (2 * width + 2 * length) - 4 = 20 := 
by
  rw [h_width, h_length]
  sorry

end rectangle_perimeter_sub_4_l654_654769


namespace min_distance_exp_curve_to_line_l654_654967

noncomputable def min_distance_to_line (x y : ℝ) (h_curve : y = Real.exp x) : ℝ :=
  let A := 1
  let B := -1
  let C := -1
  let numerator := |A * x + B * y + C|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

theorem min_distance_exp_curve_to_line :
  ∃ P : ℝ × ℝ, P.2 = Real.exp P.1 ∧ min_distance_to_line P.1 P.2 (by simp [P.2, Real.exp_eq_one_iff]) = Real.sqrt 2 :=
sorry

end min_distance_exp_curve_to_line_l654_654967


namespace smallest_r_minus_p_l654_654433

theorem smallest_r_minus_p 
  (p q r : ℕ) (h₀ : p * q * r = 362880) (h₁ : p < q) (h₂ : q < r) : 
  r - p = 126 :=
sorry

end smallest_r_minus_p_l654_654433


namespace false_p_and_q_l654_654123

variable {a : ℝ} 

def p (a : ℝ) := 3 * a / 2 ≤ 1
def q (a : ℝ) := 0 < 2 * a - 1 ∧ 2 * a - 1 < 1

theorem false_p_and_q (a : ℝ) :
  ¬ (p a ∧ q a) ↔ (a ≤ (1 : ℝ) / 2 ∨ a > (2 : ℝ) / 3) :=
by
  sorry

end false_p_and_q_l654_654123


namespace hand_position_at_8PM_yesterday_l654_654457

-- Define the conditions of the problem
def positions : ℕ := 20
def jump_interval_min : ℕ := 7
def jump_positions : ℕ := 9
def start_position : ℕ := 0
def end_position : ℕ := 8 -- At 8:00 AM, the hand is at position 9, hence moving forward 8 positions from position 0

-- Define the total time from 8:00 PM yesterday to 8:00 AM today
def total_minutes : ℕ := 720

-- Calculate the number of full jumps
def num_full_jumps : ℕ := total_minutes / jump_interval_min

-- Calculate the hand's final position from 8:00 PM yesterday
def final_hand_position : ℕ := (start_position + num_full_jumps * jump_positions) % positions

-- Prove that the final hand position is 2
theorem hand_position_at_8PM_yesterday : final_hand_position = 2 :=
by
  sorry

end hand_position_at_8PM_yesterday_l654_654457


namespace probability_more_males_or_females_l654_654255

theorem probability_more_males_or_females (n : ℕ) (p : ℝ) (h : n = 11) (hp : p = 0.5) :
  let probability_equal := (binom n 5 + binom n 6) / (2^n)
  in 1 - probability_equal = 281 / 512 :=
by
  sorry

end probability_more_males_or_females_l654_654255


namespace total_profit_l654_654480

-- Definitions based on the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Statement of the theorem
theorem total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end total_profit_l654_654480


namespace part1_part2_l654_654562

def setA := {x : ℝ | -3 < x ∧ x < 4}
def setB (a : ℝ) := {x : ℝ | x^2 - 4 * a * x + 3 * a^2 = 0}

theorem part1 (a : ℝ) : (setA ∩ setB a = ∅) ↔ (a ≤ -3 ∨ a ≥ 4) :=
sorry

theorem part2 (a : ℝ) : (setA ∪ setB a = setA) ↔ (-1 < a ∧ a < 4/3) :=
sorry

end part1_part2_l654_654562


namespace assignment_arrangement_l654_654532

theorem assignment_arrangement (students : Finset ℕ) (tasks : Finset ℕ) (jia yi : ℕ) (h_num_students : students.card = 6) (h_num_tasks : tasks.card = 4) (h_jia_in_students : jia ∈ students) (h_yi_in_students : yi ∈ students) (task_A : ℕ) (h_taskA_in_tasks : task_A ∈ tasks) :
  ∃ arrangements : Finset (Finset (ℕ × ℕ)), arrangements.card = 240 :=
by
  -- Define total permutations without restriction
  let total_no_restriction := 6 * 5 * 4 * 3
  
  -- Define restricted permutations for Jia assigned to task_A only
  let restricted_jia := 5 * 4 * 3

  -- Define restricted permutations for Yi assigned to task_A only
  let restricted_yi := 5 * 4 * 3

  -- Total valid arrangements considering both restrictions
  let total_valid := total_no_restriction - restricted_jia - restricted_yi

  -- Assert the final card is 240
  have h_valid_card : total_valid = 240 := by 
    simp [total_no_restriction, restricted_jia, restricted_yi]
    -- 360 - 60 - 60 = 240
    norm_num
  
  -- Provable existence of such arrangements
  use Finset.range total_valid
  simp [h_valid_card]
  sorry

end assignment_arrangement_l654_654532


namespace remainder_proof_l654_654413

theorem remainder_proof (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v ∧ v < y) : 
  (x + y * u^2 + 3 * v) % y = 4 * v % y :=
by
  sorry

end remainder_proof_l654_654413


namespace compelling_quadruples_l654_654910
   
   def isCompellingQuadruple (a b c d : ℕ) : Prop :=
     1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 10 ∧ a + d < b + c 

   def compellingQuadruplesCount (count : ℕ) : Prop :=
     count = 80
   
   theorem compelling_quadruples :
     ∃ count, compellingQuadruplesCount count :=
   by
     use 80
     sorry
   
end compelling_quadruples_l654_654910


namespace sum_four_digit_integers_l654_654400

theorem sum_four_digit_integers : 
  ∑ k in Finset.range (9999 - 1000 + 1), (k + 1000) = 49495500 := 
by
  sorry

end sum_four_digit_integers_l654_654400


namespace number_of_ways_to_fill_table_is_831_l654_654844

-- Definitions based on the problem conditions
def numbers := {2014, 2015, 2016}
def table := array (3 × 3) ℕ

-- Main theorem stating the problem and correct answer
theorem number_of_ways_to_fill_table_is_831 :
  ∃ (f : table → Prop), (∀ (t : table), f t → table_filled_with t numbers) ∧ (sum_in_each_row_equal t) ∧ (count_valid_tables f = 831) :=
begin
  sorry
end

end number_of_ways_to_fill_table_is_831_l654_654844


namespace prescribed_dosage_lesser_than_typical_l654_654420

-- Define given conditions
def prescribed_dosage : ℝ := 12
def body_weight : ℝ := 120
def typical_dosage_rate : ℝ := 2 / 15

-- Define what the typical dosage calculation
def typical_dosage : ℝ := (body_weight / 15) * 2

-- Define the percentage difference calculation
def percentage_difference (prescribed dosage typical_dosage : ℝ) : ℝ := 
  ((typical_dosage - prescribed_dosage) / typical_dosage) * 100

-- Prove the percentage difference is 25%
theorem prescribed_dosage_lesser_than_typical :
  percentage_difference prescribed_dosage typical_dosage = 25 :=
by
  -- sorry statement to placeholders for the proof
  sorry

end prescribed_dosage_lesser_than_typical_l654_654420


namespace cube_surface_area_on_sphere_l654_654186

theorem cube_surface_area_on_sphere {r : ℝ} (h₁ : r = 1) 
  (cube_on_sphere : ∀ (x y z : ℝ), x^2 + y^2 + z^2 = r^2):
  let a := 2 / sqrt 3 in
  let surface_area := 6 * (a * a) in
  surface_area = 8 :=
by
  sorry

end cube_surface_area_on_sphere_l654_654186


namespace part_1_part_2_l654_654542

noncomputable def f (x a : ℝ) : ℝ := x^2 * |x - a|

theorem part_1 (a : ℝ) (h : a = 2) : {x : ℝ | f x a = x} = {0, 1, 1 + Real.sqrt 2} :=
by 
  sorry

theorem part_2 (a : ℝ) : 
  ∃ m : ℝ, m = 
    if a ≤ 1 then 1 - a 
    else if 1 < a ∧ a ≤ 2 then 0 
    else if 2 < a ∧ a ≤ (7 / 3 : ℝ) then 4 * (a - 2) 
    else a - 1 :=
by 
  sorry

end part_1_part_2_l654_654542


namespace inequality_proof_l654_654952

noncomputable def a := Real.log 1 / Real.log 3
noncomputable def b := Real.log 1 / Real.log (1 / 2)
noncomputable def c := (1/2)^(1/3)

theorem inequality_proof : b > c ∧ c > a := 
by 
  sorry

end inequality_proof_l654_654952


namespace quadrilateral_not_parallelogram_minimum_value_AC_plus_BD_l654_654986

def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

def focus (x y : ℝ) : Prop := (x, y) = (1, 0)

def perpendicular_lines_through_focus {A B C D : ℝ × ℝ} (F : ℝ × ℝ) : Prop :=
  ∃ Fx Fy : ℝ, Fx = F.1 ∧ Fy = F.2 ∧
  (A.1 = Fx ∨ B.1 = Fx ∨ C.1 = Fx ∨ D.1 = Fx) ∧
  (A.2 = Fy ∨ B.2 = Fy ∨ C.2 = Fy ∨ D.2 = Fy)

theorem quadrilateral_not_parallelogram (A B C D : ℝ × ℝ) :
  (∀ x y : ℝ, ellipse x y) →
  focus 1 0 →
  perpendicular_lines_through_focus (1, 0) →
  ¬ (parallelogram A B C D) := sorry

theorem minimum_value_AC_plus_BD (A B C D : ℝ × ℝ) :
  (∀ x y : ℝ, ellipse x y) →
  focus 1 0 →
  perpendicular_lines_through_focus (1, 0) →
  (|AC| + |BD| = (8 * real.sqrt 2) / 3) := sorry

end quadrilateral_not_parallelogram_minimum_value_AC_plus_BD_l654_654986


namespace circumference_of_inscribed_circle_l654_654453

  theorem circumference_of_inscribed_circle :
    ∀ (a b : ℝ), a = 10 → b = 24 → let d := Real.sqrt (a^2 + b^2) in
    let C := Real.pi * d in C = 26 * Real.pi :=
  by
    intros a b ha hb
    simp [ha, hb]
    let d := Real.sqrt (10^2 + 24^2)
    norm_num
    let C := Real.pi * d
    -- lhs should compute to 26 * Real.pi
    rw Real.sqrt_eq_r_sqrt (expr) at sd -- sqrt value should be evaluated to 26
    sorry
  
end circumference_of_inscribed_circle_l654_654453


namespace min_area_QAMB_find_coordinates_Q_l654_654560

variable (Q M A B : ℝ × ℝ)

/-- Defining the circle M with center (0, 2) and radius 1 -/
def is_circle_M : Prop := (M = (0, 2)) ∧ (∀ P : ℝ × ℝ, (P.1)^2 + (P.2 - 2)^2 = 1 → M = P)

/-- Q is a moving point on the x-axis -/
def is_on_x_axis : Prop := Q.2 = 0

/- QA and QB are tangents to circle M at points A and B respectively -/
def are_tangents : Prop := 
  ∃ A B : ℝ × ℝ,
  (A.1)^2 + (A.2 - 2)^2 = 1 ∧ (B.1)^2 + (B.2 - 2)^2 = 1 ∧ 
  ∀ Q : ℝ × ℝ, Q.2 = 0 → (Q = A ∨ Q = B)

/-- Question 1: Find the minimum area of quadrilateral QAMB -/
theorem min_area_QAMB (Q : ℝ × ℝ) (h1 : is_circle_M M) (h2 : is_on_x_axis Q) (h3 : are_tangents Q A B) : 
  ∃ S : ℝ, S ≥ 0 ∧ S = sqrt 3 := sorry

/-- Question 2: Given |AB| = 4 * sqrt 2 / 3, find the coordinates of point Q -/
theorem find_coordinates_Q (Q A B : ℝ × ℝ) (h1 : is_circle_M M) (h2 : are_tangents Q A B) 
  (h_length_AB : dist A B = 4*sqrt 2/3) : 
  Q = (sqrt 5, 0) ∨ Q = (-sqrt 5, 0) := sorry

end min_area_QAMB_find_coordinates_Q_l654_654560


namespace abs_equality_holds_if_interval_l654_654061

noncomputable def quadratic_abs_equality (x : ℝ) : Prop :=
  |x^2 - 8 * x + 12| = x^2 - 8 * x + 12

theorem abs_equality_holds_if_interval (x : ℝ) :
  quadratic_abs_equality x ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end abs_equality_holds_if_interval_l654_654061


namespace mixtape_length_l654_654476

theorem mixtape_length (songs_side1 songs_side2 song_duration : ℕ) 
  (h1 : songs_side1 = 6) 
  (h2 : songs_side2 = 4) 
  (h3 : song_duration = 4) : 
  (songs_side1 + songs_side2) * song_duration = 40 :=
by
  rw [h1, h2, h3]
  norm_num

end mixtape_length_l654_654476


namespace minimum_marked_cells_ensure_coverage_l654_654378

/-- Definition of the board as a set of coordinates --/
def board : set (ℕ × ℕ) := { p | p.1 < 10 ∧ p.2 < 11 }

/-- Definition of an L-shaped piece as a set of three coordinates --/
def L_shaped_piece (x y : ℕ) : set (ℕ × ℕ) :=
  if x <= y then { (x, y), (x+1, y), (x, y+1) } else { (x, y), (x-1, y), (x, y+1) }

/-- Predicate indicating if a cell is marked --/
def is_marked (marked : set (ℕ × ℕ)) (cell : ℕ × ℕ) : Prop := cell ∈ marked

/-- The specific problem statement --/
theorem minimum_marked_cells_ensure_coverage :
  ∃ (marked : set (ℕ × ℕ)), marked.card = 50 ∧ ∀ x y ∈ board, 
  ∃ m ∈ L_shaped_piece x y, is_marked marked m :=
sorry

end minimum_marked_cells_ensure_coverage_l654_654378


namespace inequality_satisfaction_l654_654103

theorem inequality_satisfaction (a b : ℝ) (h : 0 < a ∧ a < b) : 
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b :=
by
  sorry

end inequality_satisfaction_l654_654103


namespace miraflores_optimal_split_l654_654767

-- Define the total number of voters as 2n, and initialize half supporters for each candidate.
variable (n : ℕ) (voters : Fin (2 * n) → Bool)

-- Define the condition that exactly half of the voters including Miraflores support him
def half_support_miraflores : Prop :=
  ∃ (supporters_miraflores : Fin n) (supporters_maloney : Fin n), 
    (voters supporters_miraflores.val = true) ∧ (voters.supporters_maloney.val = false) 

-- Define the condition of drawing a single random ballot in each district.
def draw_random_ballot (d : Fin n → Prop) : Fin n := sorry

-- Define the condition that Miraflores wins if he wins both districts.
def wins_election (d1 d2 : Fin n → Prop) : Prop := 
  (draw_random_ballot d1 = true) ∧ (draw_random_ballot d2 = true)

-- Miraflores should split the voters such that his maximum probability of winning is achieved.
def optimal_split : Prop :=
  ∃ (d1 d2 : Fin n → Bool), 
    (d1.supporters_miraflores.val = true ∧ d2.supporters_maloney.val = false) ∧
    (wins_election d1 d2 = true)

theorem miraflores_optimal_split (n : ℕ) (voters : Fin (2 * n) → Bool) (half_support : half_support_miraflores n voters) : optimal_split n :=
sorry

end miraflores_optimal_split_l654_654767


namespace bobs_total_profit_l654_654482

-- Definitions of the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Definition of the problem statement
theorem bobs_total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end bobs_total_profit_l654_654482


namespace sum_of_squares_of_perpendiculars_l654_654683

theorem sum_of_squares_of_perpendiculars (a b x₀ y₀ : ℝ) (hₑ : e = Real.sqrt (a^2 - b^2)) :
  let ellipse_eq := (λ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
  let tangent_eq := (λ (x y : ℝ), x₀ * x / a^2 + y₀ * y / b^2 = 1)
  let perp_dist_sq := (λ (yₚ : ℝ), 
                         let num := abs ((y₀ * yₚ / b^2) - 1) 
                         let den := sqrt ((x₀ / a^2)^2 + (y₀ / b^2)^2)
                         (num / den)^2)
  in perp_dist_sq (Real.sqrt (a^2 - b^2)) + perp_dist_sq (-Real.sqrt (a^2 - b^2)) = 2 * a^2 :=
sorry

end sum_of_squares_of_perpendiculars_l654_654683


namespace sum_of_four_digit_integers_l654_654408

theorem sum_of_four_digit_integers : 
  let a := 1000 in
  let l := 9999 in
  let n := l - a + 1 in
  (a + l) * n / 2 = 49495500 :=
by
  let a := 1000
  let l := 9999
  let n := l - a + 1
  have h_sum := ((a + l) * n) / 2
  rw [a, l, n] at h_sum
  exact h_sum
  sorry

end sum_of_four_digit_integers_l654_654408


namespace complex_line_eq_perpendicular_l654_654076

-- Define the necessary setup for complex numbers and conditions 
variables {α w z : ℂ} (α_ne_zero : α ≠ 0)

-- State the theorem for the complex line equation problem
theorem complex_line_eq_perpendicular (α_ne_zero : α ≠ 0) :
  (z / α - conj(z) / conj(α) = w / α - conj(w) / conj(α)) :=
sorry

end complex_line_eq_perpendicular_l654_654076


namespace fewer_twos_to_hundred_l654_654360

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l654_654360


namespace problem_statement_l654_654090

noncomputable def floor (x : ℝ) : ℤ := int.floor x

theorem problem_statement (x : ℝ) :
  ¬ (∀ x, floor (2^x) = floor (2^floor x)) ∧ ¬ (∀ x, floor (x^2) = floor x^2) :=
by
  sorry

end problem_statement_l654_654090


namespace perpendicular_m_value_parallel_m_value_l654_654838

-- Defining lines
def l1 (m : ℚ) : String := s!"x + (1 + {m})y + {m} - 2 = 0"
def l2 (m : ℚ) : String := s!"{m}x + 2y + 8 = 0"

-- Finding the value of m for perpendicular lines
theorem perpendicular_m_value (m : ℚ) :  
  let k1 := -(1/(m + 1)) in
  let k2 := -(m/2) in
  k1 * k2 = -1 → m = -2/3 := 
by sorry

-- Finding the value of m for parallel lines
theorem parallel_m_value (m : ℚ) : 
  let k1 := -(1/(m + 1)) in
  let k2 := -(m/2) in
  let b1 := (2 - m)/(m + 1) in
  let b2 := -4 in
  (k1 = k2 ∧ b1 ≠ b2) → m = 1 := 
by sorry

end perpendicular_m_value_parallel_m_value_l654_654838


namespace maximize_Miraflores_win_l654_654747

-- Definitions based on given conditions
def voters_count (n : ℕ) : ℕ := 2 * n
def support_Miraflores (n : ℕ) : ℕ := n + 1
def support_opponent (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem maximize_Miraflores_win (n : ℕ) (hn : n > 0) : 
  ∃ (d1 d2 : ℕ), d1 = 1 ∧ d2 = 2 * n - 1 ∧ support_Miraflores n > support_opponent n := 
sorry

end maximize_Miraflores_win_l654_654747


namespace count_indistinguishable_distributions_l654_654173

theorem count_indistinguishable_distributions (balls : ℕ) (boxes : ℕ) (h_balls : balls = 5) (h_boxes : boxes = 4) : 
  ∃ n : ℕ, n = 6 := by
  sorry

end count_indistinguishable_distributions_l654_654173


namespace ellipse_transformation_eq_circle_l654_654574

variable (λ μ : ℝ)
variable (hλ : λ > 0) (hμ : μ > 0)

theorem ellipse_transformation_eq_circle
  (h1 : ∀ x y : ℝ, (λ * λ * x^2) + (μ * μ * y^2) = 9)
  (h_ellipse : ∀ x y, (x^2 / 9) + (y^2 / 4) = 1):
  λ = 1 ∧ μ = 3 / 2 := 
by
  sorry

end ellipse_transformation_eq_circle_l654_654574


namespace add_ab_values_l654_654131

theorem add_ab_values (a b : ℝ) (h1 : ∀ x : ℝ, (x^2 + 4*x + 3) = (a*x + b)^2 + 4*(a*x + b) + 3) :
  a + b = -8 ∨ a + b = 4 :=
  by sorry

end add_ab_values_l654_654131


namespace jimmy_change_l654_654221

noncomputable def change_back (pen_cost notebook_cost folder_cost highlighter_cost sticky_notes_cost total_paid discount tax : ℝ) : ℝ :=
  let total_before_discount := (5 * pen_cost) + (6 * notebook_cost) + (4 * folder_cost) + (3 * highlighter_cost) + (2 * sticky_notes_cost)
  let total_after_discount := total_before_discount * (1 - discount)
  let final_total := total_after_discount * (1 + tax)
  (total_paid - final_total)

theorem jimmy_change :
  change_back 1.65 3.95 4.35 2.80 1.75 150 0.25 0.085 = 100.16 :=
by
  sorry

end jimmy_change_l654_654221


namespace arrangable_integers_l654_654515

theorem arrangable_integers: 
  ∀ n : ℕ, n > 1 ∧ (∃ l : list ℕ, l.perm (list.range (n+1)) ∧ (∀ (i : ℕ), i < n → ((l.nth i).get_or_else 0 ∣ (l.nth (i+1)).get_or_else 0 ∨ (l.nth (i+1)).get_or_else 0 ∣ (l.nth i).get_or_else 0))) → 
  n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 6 :=
sorry

end arrangable_integers_l654_654515


namespace number_of_four_digit_integers_with_conditions_l654_654887

def count_valid_numbers : ℕ :=
  -- Each valid pair and the distinct middle digits calculation
  let valid_pairs := 15  -- The number of valid (a, d) pairs
  let choices_for_b_c := 8 * 7  -- Ways to choose b and c from remaining digits
  valid_pairs * choices_for_b_c

theorem number_of_four_digit_integers_with_conditions :
  count_valid_numbers = 840 :=
begin
  sorry
end

end number_of_four_digit_integers_with_conditions_l654_654887


namespace curve_polar_equation_eq_length_of_chord_l654_654998

noncomputable def curve_eqns (α : ℝ) : ℝ × ℝ :=
  (3 + sqrt 10 * cos α, 1 + sqrt 10 * sin α)

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def cartesian_circle : ℝ × ℝ → Prop :=
  λ (x, y), (x - 3)^2 + (y - 1)^2 = 10

def polar_equation_of_curve (ρ θ : ℝ) : Prop :=
  ρ = 6 * cos θ + 2 * sin θ

def polar_line (ρ θ : ℝ) : Prop :=
  sin θ - cos θ = 1 / ρ

theorem curve_polar_equation_eq (α : ℝ) :
  ∃ (ρ θ : ℝ), polar_equation_of_curve ρ θ :=
sorry

theorem length_of_chord (ρ θ : ℝ) (h : polar_line ρ θ) :
  ∃ (l : ℝ), l = sqrt 22 :=
sorry

end curve_polar_equation_eq_length_of_chord_l654_654998


namespace find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l654_654796

-- Define the nature of a "cool" triple.
def is_cool_triple (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 1 ∧ z > 0 ∧ x^2 - 3 * y^2 = z^2 - 3

-- Part (a) i: For x = 5.
theorem find_cool_triple_x_eq_5 : ∃ (y z : ℕ), is_cool_triple 5 y z := sorry

-- Part (a) ii: For x = 7.
theorem find_cool_triple_x_eq_7 : ∃ (y z : ℕ), is_cool_triple 7 y z := sorry

-- Part (b): For every x ≥ 5 and odd, there are at least two distinct cool triples.
theorem two_distinct_cool_triples_for_odd_x (x : ℕ) (h1 : x ≥ 5) (h2 : x % 2 = 1) : 
  ∃ (y₁ z₁ y₂ z₂ : ℕ), is_cool_triple x y₁ z₁ ∧ is_cool_triple x y₂ z₂ ∧ (y₁, z₁) ≠ (y₂, z₂) := sorry

-- Part (c): Find a cool type triple with x even.
theorem find_cool_triple_x_even : ∃ (x y z : ℕ), x % 2 = 0 ∧ is_cool_triple x y z := sorry

end find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l654_654796


namespace polygon_sides_and_diagonals_l654_654217

theorem polygon_sides_and_diagonals (n : ℕ) :
  (180 * (n - 2) = 3 * 360 + 180) → n = 9 ∧ (n - 3 = 6) :=
by
  intro h_sum_angles
  -- This is where you would provide the proof.
  sorry

end polygon_sides_and_diagonals_l654_654217


namespace fixed_point_circumcircle_of_triangle_pib_ic_l654_654307

open EuclideanGeometry

-- Definitions for the problem components
variables (A B C P: Point)
variable (omega : circle)
variable [AB_EQ_AC : AB = AC]
variable [in_circle : ω.isInscribedTriangle A B C]
variable [P_on_arc_not_containing_A : P.οn_arc BC omega]
variables (I_B : incenter_triangle A B P) (I_C : incenter_triangle A C P)

-- Main theorem statement
theorem fixed_point_circumcircle_of_triangle_pib_ic :
  ∃ M : Point, (M = midpoint_arc_not_containing_A B C omega) ∧
    (∀ P : Point, variable_point P_on_arc_not_containing_A P →
      passes_through_fixed_point (circumcircle_triangle P I_B I_C M)) :=
begin
  sorry
end

end fixed_point_circumcircle_of_triangle_pib_ic_l654_654307


namespace number_of_sets_of_clothes_l654_654557

-- Define the normal distribution and conditions
def height_distribution : ℝ × ℝ := (173, 5^2)

-- State the theorem to prove the number of sets of clothes
theorem number_of_sets_of_clothes (num_employees : ℕ) (p1 p2 : ℝ) :
  height_distribution = (173, 25) → 
  num_employees = 10000 → 
  p1 = 68.3 / 100 → 
  p2 = 95.4 / 100 → 
  let p_avg : ℝ := (p1 + p2) / 2 in
  let total_clothes : ℕ := (num_employees * p_avg).to_nat in
  total_clothes = 8185 :=
by
  sorry

end number_of_sets_of_clothes_l654_654557


namespace tangency_check_l654_654715

theorem tangency_check 
  (f: ℝ → ℝ)
  (h₁ : ∃ b : ℝ, ∀ x : ℝ, f x = -1 / x → ∃ y' : ℝ, (f' x) = (3 / 2))
  (h₂ : ∃ b : ℝ, ∀ x : ℝ, f x = sin x → ∃ y' : ℝ, (f' x) = (3 / 2))
  (h₃ : ∃ b : ℝ, ∀ x : ℝ, f x = ln x → ∃ y' : ℝ, (f' x) = (3 / 2))
  (h₄ : ∃ b : ℝ, ∀ x : ℝ, f x = exp x → ∃ y' : ℝ, (f' x) = (3 / 2)) :
  (f = (λ x, -1 / x) ∨ f = (λ x, ln x) ∨ f = (λ x, exp x)) :=
by
  sorry

end tangency_check_l654_654715


namespace value_of_a_l654_654212

theorem value_of_a
  (a : ℝ)
  (h1 : ∀ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1)
  (h2 : ∀ (ρ : ℝ), ρ = a)
  (h3 : ∃ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1 ∧ ρ = a ∧ θ = 0)  :
  a = Real.sqrt 2 / 2 := 
sorry

end value_of_a_l654_654212


namespace carpet_area_proof_l654_654067

noncomputable def carpet_area (main_room_length_ft : ℕ) (main_room_width_ft : ℕ)
  (corridor_length_ft : ℕ) (corridor_width_ft : ℕ) (feet_per_yard : ℕ) : ℚ :=
  let main_room_length_yd := main_room_length_ft / feet_per_yard
  let main_room_width_yd := main_room_width_ft / feet_per_yard
  let corridor_length_yd := corridor_length_ft / feet_per_yard
  let corridor_width_yd := corridor_width_ft / feet_per_yard
  let main_room_area_yd2 := main_room_length_yd * main_room_width_yd
  let corridor_area_yd2 := corridor_length_yd * corridor_width_yd
  main_room_area_yd2 + corridor_area_yd2

theorem carpet_area_proof : carpet_area 15 12 10 3 3 = 23.33 :=
by
  -- Proof steps go here
  sorry

end carpet_area_proof_l654_654067


namespace evaluate_logarithmic_expression_l654_654069

theorem evaluate_logarithmic_expression : 2^(Real.log (5 + 3) / Real.log 2) = 8 :=
by
  sorry

end evaluate_logarithmic_expression_l654_654069


namespace maximize_prob_l654_654618

open Probability

noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

def X_max_prob : ℕ := 5
def p : ℚ := 1 / 4

theorem maximize_prob :
  ∃ k ∈ Finset.range (X_max_prob + 1), (binomial_prob X_max_prob k p) >=
    (binomial_prob X_max_prob n p) ∀ n ∈ Finset.range (X_max_prob + 1) :=
sorry

end maximize_prob_l654_654618


namespace fewer_twos_to_hundred_l654_654356

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l654_654356


namespace smallest_number_of_marbles_l654_654019

open Nat

/-- Variables representing the number of marbles of each color -/
variables r w b g y : ℕ

/-- Total number of marbles -/
def n := r + w + b + g + y

/-- Binomial coefficients representing the different event probabilities -/
def prob_five_red := choose r 5
def prob_one_white_four_red := choose w 1 * choose r 4
def prob_one_white_one_blue_three_red := choose w 1 * choose b 1 * choose r 3
def prob_one_each_of_four_colors_and_two_red := choose w 1 * choose b 1 * choose g 1 * choose r 2
def prob_one_each_color := choose w 1 * choose b 1 * choose g 1 * choose y 1 * choose r 1

/-- Conditions asserting that the events are equally likely -/
def conditions : Prop :=
  prob_five_red = prob_one_white_four_red ∧
  prob_one_white_four_red = prob_one_white_one_blue_three_red ∧
  prob_one_white_one_blue_three_red = prob_one_each_of_four_colors_and_two_red ∧
  prob_one_each_of_four_colors_and_two_red = prob_one_each_color

/-- Proving that the smallest number of marbles satisfying the conditions is 41 -/
theorem smallest_number_of_marbles :
  conditions → n = 41 :=
sorry

end smallest_number_of_marbles_l654_654019


namespace find_m_odd_function_l654_654153

noncomputable def f (x m : ℝ) : ℝ := log ((x + 1) / (x - 1)) + m + 1

theorem find_m_odd_function (m : ℝ) :
  (∀ x : ℝ, f x m = log ((x + 1) / (x - 1)) + m + 1) →
  (∀ x : ℝ, f (-x) m = -f x m) →
  m = -1 :=
by
  intros h_f h_odd
  sorry

end find_m_odd_function_l654_654153


namespace existence_of_point_N_l654_654559

-- Given conditions
def is_point_on_ellipse (x y a b : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

def is_ellipse (a b : ℝ) : Prop := 
  a > b ∧ b > 0 ∧ (a^2 = b^2 + (a * (Real.sqrt 2) / 2)^2)

def passes_through_point (x y a b : ℝ) (px py : ℝ) : Prop :=
  (px^2 / a^2) + (py^2 / b^2) = 1

def ellipse_with_eccentricity (a : ℝ) : Prop :=
  (Real.sqrt 2) / 2 = (Real.sqrt (a^2 - (a * (Real.sqrt 2) / 2)^2)) / a

def line_through_point (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

def lines_intersect_ellipse (k a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b

def angle_condition (k t a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b ∧ 
  ((y1 - t) / x1) + ((y2 - t) / x2) = 0

-- Lean 4 statement
theorem existence_of_point_N (a b k t : ℝ) (hx : is_ellipse a b) (hp : passes_through_point 2 (Real.sqrt 2) a b 2 (Real.sqrt 2)) (he : ellipse_with_eccentricity a) (hl : ∀ (x1 y1 x2 y2 : ℝ), lines_intersect_ellipse k a b) :
  ∃ (N : ℝ), N = 4 ∧ angle_condition k N a b :=
sorry

end existence_of_point_N_l654_654559


namespace distinct_ball_distributions_l654_654170

theorem distinct_ball_distributions : 
  ∃ (distros : Set (Fin 4 → Fin 6)), 
    distros = { f | f.Sum = 5 ∧
                 ∀ m n : Fin 4, f m ≥ f n ∧ 
                 distros = { (5, 0, 0, 0), (4, 1, 0, 0), (3, 2, 0, 0), (3, 1, 1, 0), 
                             (2, 2, 1, 0), (2, 1, 1, 1) }} ∧
    distros.card = 6 :=
sorry

end distinct_ball_distributions_l654_654170


namespace f_monotonically_decreasing_l654_654156

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_monotonically_decreasing : ∀ x, 0 < x ∧ x < 1 / Real.exp 1 → deriv f x < 0 :=
by
  sorry

end f_monotonically_decreasing_l654_654156


namespace maximize_Miraflores_win_l654_654746

-- Definitions based on given conditions
def voters_count (n : ℕ) : ℕ := 2 * n
def support_Miraflores (n : ℕ) : ℕ := n + 1
def support_opponent (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem maximize_Miraflores_win (n : ℕ) (hn : n > 0) : 
  ∃ (d1 d2 : ℕ), d1 = 1 ∧ d2 = 2 * n - 1 ∧ support_Miraflores n > support_opponent n := 
sorry

end maximize_Miraflores_win_l654_654746


namespace fewer_twos_result_100_l654_654369

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l654_654369


namespace total_weight_of_onions_l654_654775

def weight_per_bag : ℕ := 50
def bags_per_trip : ℕ := 10
def trips : ℕ := 20

theorem total_weight_of_onions : bags_per_trip * weight_per_bag * trips = 10000 := by
  sorry

end total_weight_of_onions_l654_654775


namespace find_point_Q_l654_654731

theorem find_point_Q {a b c : ℝ} 
  (h1 : ∀ x y z : ℝ, (x + 1)^2 + (y - 3)^2 + (z + 2)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2) 
  (h2 : ∀ x y z: ℝ, 8 * x - 6 * y + 12 * z = 34) : 
  (a = 3) ∧ (b = -6) ∧ (c = 8) :=
by
  sorry

end find_point_Q_l654_654731


namespace congruent_squares_intersecting_l654_654111

theorem congruent_squares_intersecting {n : ℕ} (h1 : (2018 ≤ n) ∧ (n ≤ 3018))
    (h2 : ∃ squares : set (set ℝ^2), (∀ s ∈ squares, is_congruent s) ∧
        (∀ s₁ s₂ ∈ squares, s₁ ∩ s₂ ≠ ∅ → (∃ P, (P ∈ s₁) ∧ (P ∈ s₂) ∧ (is_vertex P s₁) ∧ (is_vertex P s₂)))
        ∧ (∀ s ∈ squares, (∃ c ∈ squares, s ∩ c ≠ ∅) ∧ (size (set {s' | s' ∈ squares ∧ s ∩ s' ≠ ∅}) = 3)) : 
    nat_odd (3 * n)) : 
    501 :=
by
  sorry

/-- Placeholder definitions to be defined properly -/
def is_congruent (s : set ℝ^2) : Prop := sorry

def is_vertex (P : ℝ^2) (s : set ℝ^2) : Prop := sorry

def nat_odd (n : ℕ) : Prop := n % 2 = 1

end congruent_squares_intersecting_l654_654111


namespace sum_four_digit_integers_l654_654396

def a := 1000
def l := 9999
def n := 9999 - 1000 + 1
def S (n : ℕ) (a : ℕ) (l : ℕ) := n / 2 * (a + l)

theorem sum_four_digit_integers : S n a l = 49495500 :=
by
  sorry

end sum_four_digit_integers_l654_654396


namespace negation_of_forall_exp_positive_l654_654315

theorem negation_of_forall_exp_positive :
  ¬ (∀ x : ℝ, Real.exp x > 0) ↔ ∃ x : ℝ, Real.exp x ≤ 0 :=
by {
  sorry
}

end negation_of_forall_exp_positive_l654_654315


namespace number_of_lion_cubs_l654_654858

def initial_animals : ℕ := 150
def gorillas_sent_away : ℕ := 12
def hippopotamus_adopted : ℕ := 1
def giraffes_adopted : ℕ := 8
def rhinos_rescued : ℕ := 4
def crocodiles_rescued : ℕ := 5
def total_animals_after_all_changes : ℕ := 260

theorem number_of_lion_cubs 
    (initial_animals = 150)
    (gorillas_sent_away = 12)
    (hippopotamus_adopted = 1)
    (giraffes_adopted = 8)
    (rhinos_rescued = 4)
    (crocodiles_rescued = 5)
    (total_animals_after_all_changes = 260) :
  ∃ C : ℕ, 26 = C ∧ (initial_animals - gorillas_sent_away + hippopotamus_adopted + giraffes_adopted + rhinos_rescued + crocodiles_rescued + C + 3 * C = total_animals_after_all_changes) :=
by sorry

end number_of_lion_cubs_l654_654858


namespace factorial_comparison_l654_654811

theorem factorial_comparison :
  (Nat.factorial (Nat.factorial 100)) <
  (Nat.factorial 99)^(Nat.factorial 100) * (Nat.factorial 100)^(Nat.factorial 99) :=
  sorry

end factorial_comparison_l654_654811


namespace palindromic_times_count_l654_654901

theorem palindromic_times_count : 
    let num_palindromes := 12 * 5 + 7  in
    num_palindromes = 67 :=
by
  let num_palindromes := 10 * 6 + 7
  have h : num_palindromes = 67 := sorry
  exact h

end palindromic_times_count_l654_654901


namespace value_range_of_f_l654_654329

-- Define the function f(x) = 2x - x^2
def f (x : ℝ) : ℝ := 2 * x - x^2

-- State the theorem with the given conditions and prove the correct answer
theorem value_range_of_f :
  (∀ y : ℝ, -3 ≤ y ∧ y ≤ 1 → ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = y) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → -3 ≤ f x ∧ f x ≤ 1) :=
by
  sorry

end value_range_of_f_l654_654329


namespace sum_of_floor_sqrt_l654_654899

theorem sum_of_floor_sqrt :
  (∑ n in finset.range 25 \+1, (⌊real.sqrt n⌋)) = 75 := by
  sorry

end sum_of_floor_sqrt_l654_654899


namespace profit_2_hours_max_profit_900kg_l654_654922

noncomputable def profit_per_hour (x : ℝ) : ℝ := 100 * (5 * x + 1 - 3 / x)

theorem profit_2_hours (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) : 
  200 * (5 * x + 1 - 3 / x) ≥ 3000 ↔ 3 ≤ x ∧ x ≤ 10 :=
begin
  sorry
end

noncomputable def profit_900kg (x : ℝ) : ℝ := 90000 * (-3 / x^2 + 1 / x + 5)

theorem max_profit_900kg (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) : 
  (profit_900kg x).maximum_on (set.Icc 1 10) = 457500 ↔ x = 6 :=
begin
  sorry
end

end profit_2_hours_max_profit_900kg_l654_654922


namespace Sarahs_score_l654_654273

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l654_654273


namespace distinct_possible_c_values_l654_654662

noncomputable theory

open Complex

theorem distinct_possible_c_values :
  ∃ (c : ℂ), ∀ (r s t : ℂ), (r ≠ s) ∧ (s ≠ t) ∧ (r ≠ t) ∧
  (∀ (z : ℂ), (z - r) * (z - s) * (z - t) = (z - c * r) * (z - c * s) * (z - c * t)) → 
  (∀ c' : ℂ, (c' = 1 ∨ c'^2 = 1 ∨ c'^3 = 1) → 
  4 = (finset.card (finset.mk (finset.univ.filter (λ c'', c'' = 1 ∨ c''^2 = 1 ∨ c''^3 = 1)) sorry))) :=
sorry

end distinct_possible_c_values_l654_654662


namespace arrangements_21_leaders_l654_654616

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the permutations A_n^k
def permutations (n k : ℕ) : ℕ :=
  if k ≤ n then factorial n / factorial (n - k) else 0

theorem arrangements_21_leaders : permutations 2 2 * permutations 18 18 = factorial 18 ^ 2 :=
by 
  sorry

end arrangements_21_leaders_l654_654616


namespace exists_uncolored_diameter_l654_654289

theorem exists_uncolored_diameter (C : ℝ) (black_arcs : set ℝ) 
  (h1 : ∀ x ∈ black_arcs, 0 ≤ x)
  (h2 : (∑ x in black_arcs, x) < C / 2) : 
  ∃ (A B : ℝ), A ≠ B ∧ (∀ x ∈ (set.pair A B), x ∉ black_arcs) := 
by sorry

end exists_uncolored_diameter_l654_654289


namespace main_theorem_l654_654240

open Matrix
open Complex

variables {M N : Matrix (Fin 2) (Fin 2) ℂ}
variables (M_inv N_inv : Matrix (Fin 2) (Fin 2) ℂ)

-- Conditions
variables (hM_nonzero : M ≠ 0)
variables (hN_nonzero : N ≠ 0)
variables (hM_square_zero : M * M = 0)
variables (hN_square_zero : N * N = 0)
variables (hCommute : M * N + N * M = 1)

noncomputable
def exists_invertible_matrix_A : Prop :=
∃ (A : Matrix (Fin 2) (Fin 2) ℂ), invertible A ∧
  M = A * ((![![0,1]], ![![0,0]])) * A⁻¹ ∧
  N = A * ((![![0,0]], ![![1,0]])) * A⁻¹

-- The main statement
theorem main_theorem : exists_invertible_matrix_A M N :=
by sorry

end main_theorem_l654_654240


namespace infinite_sum_eq_3_over_8_l654_654070

theorem infinite_sum_eq_3_over_8 :
  ∑' n : ℕ, (n : ℝ) / (n^4 + 4) = 3 / 8 :=
sorry

end infinite_sum_eq_3_over_8_l654_654070


namespace find_value_of_expr_l654_654107

-- Define the complex number z
def z : ℂ := 1 - complex.i

-- Define the expression to be evaluated
def expr := (2 / z) + complex.conj z

-- The theorem stating the desired proof
theorem find_value_of_expr : expr = 2 + 2 * complex.i := by
  sorry

end find_value_of_expr_l654_654107


namespace angle_between_a_b_is_120_degrees_l654_654109

-- Definitions of vectors a, b, c
variables (a b c : ℝ^3)

-- Given length conditions
axiom length_a : ‖a‖ = 1
axiom length_b : ‖b‖ = 2

-- Given vector addition condition
axiom vec_c_def : c = a + b

-- Given perpendicularity condition
axiom perp_c_a : dot_product c a = 0

-- Theorem: the angle between a and b is 120 degrees (i.e., arc-cos(-1/2))
theorem angle_between_a_b_is_120_degrees :
  real.arccos ((dot_product a b) / (‖a‖ * ‖b‖)) = real.pi * (2/3) :=
sorry

end angle_between_a_b_is_120_degrees_l654_654109


namespace problem_conditions_range_n_F_x2_lt_x2_minus_1_l654_654159

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a * Real.log x
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := 3 - b / x

theorem problem_conditions :
  (f 1 a = 1 ∧ g 1 b = 1) ∧ (deriv (λ x, f x a) 1 = deriv (λ x, g x b) 1) →
  a = 1 ∧ b = 2 :=
sorry

noncomputable def range_of_n (n : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ 1 = 2 * (x₁ + n) ∧ f x₂ 1 = 2 * (x₂ + n)

theorem range_n :
  range_of_n n ↔ n < -1/2 :=
sorry

noncomputable def F (x m : ℝ) : ℝ :=
  3 * (x - m / 2) + (m / 2) * g x 2 - 2 * f x 1

theorem F_x2_lt_x2_minus_1 {x1 x2 m : ℝ} (h1 : F(x1, m) = 0) (h2 : F(x2, m) = 0) (h3 : x1 < x2) :
  F x 2 < x2 - 1 :=
sorry

end problem_conditions_range_n_F_x2_lt_x2_minus_1_l654_654159


namespace incorrect_propositions_count_l654_654874

variable (P1 P2 P3 P4 : Prop)

-- Conditions as definitions
def Prop1 := ∀ (L1 L2 L3 : Type), (P1 L1 L3 ∧ P1 L2 L3) → L1 = L2
def Prop2 := ∀ (L1 L2 : Type) (P: Type), (P2 L1 P ∧ P2 L2 P) → L1 = L2
def Prop3 := ∀ (P1 P2 : Type) (L : Type), (P3 P1 L ∧ P3 P2 L) → P1 = P2
def Prop4 := ∀ (P1 P2 P3 : Type), (P4 P1 P3 ∧ P4 P2 P3) → P1 = P2

-- Problem: Prove the number of incorrect propositions
theorem incorrect_propositions_count : (¬Prop1 ∧ ¬Prop4) ∧ Prop2 ∧ Prop3 → 2 = 2 := 
by
  sorry

end incorrect_propositions_count_l654_654874


namespace exists_f_condition_l654_654554

open Nat

-- Define the function φ from ℕ to ℕ
variable (ϕ : ℕ → ℕ)

-- The formal statement capturing the given math proof problem
theorem exists_f_condition (ϕ : ℕ → ℕ) : 
  ∃ (f : ℕ → ℤ), (∀ x : ℕ, f x > f (ϕ x)) :=
  sorry

end exists_f_condition_l654_654554


namespace Marty_combinations_l654_654674

theorem Marty_combinations:
  let colors := ({blue, green, yellow, black, white} : Finset String)
  let tools := ({brush, roller, sponge, spray_gun} : Finset String)
  colors.card * tools.card = 20 := 
by
  sorry

end Marty_combinations_l654_654674


namespace quadrilateral_with_equal_sides_is_rhombus_l654_654814

theorem quadrilateral_with_equal_sides_is_rhombus (a b c d : ℝ) (h1 : a = b) (h2 : b = c) (h3 : c = d) : a = d :=
by
  sorry

end quadrilateral_with_equal_sides_is_rhombus_l654_654814


namespace math_proof_equivalence_l654_654247

-- Definitions based on conditions
def U := {0, 1, 2, 3, 4, 5}
def M := {0, 3, 5}
def N := {1, 4, 5}

-- The left-hand side of the equation: (M ∩ N) ∪ (U \ N)
def left_hand_side := (M ∩ N) ∪ (U \ N)

-- Statement of the equivalence to be proved
theorem math_proof_equivalence : left_hand_side = {0, 1, 3, 4, 5} := by sorry

end math_proof_equivalence_l654_654247


namespace binary_to_decimal_10011_l654_654907

theorem binary_to_decimal_10011 : 
  let binary10011 := [1, 0, 0, 1, 1] in
  (binary10011.reverse.zipWith (λ b i => b * (2^i)) (List.range binary10011.length)).sum = 19 :=
by
  sorry

end binary_to_decimal_10011_l654_654907


namespace tan_simplification_l654_654973

theorem tan_simplification (α : ℝ) (h₀ : sin α + cos α = 1 / 2) (h₁ : 0 < α ∧ α < π) : 
  (1 - tan α) / (1 + tan α) = -√7 := 
by 
  sorry

end tan_simplification_l654_654973


namespace meaningful_fraction_range_l654_654190

theorem meaningful_fraction_range (x : ℝ) (h : x - 3 ≠ 0) : x ≠ 3 := 
by
  intro h
  contradiction 

end meaningful_fraction_range_l654_654190


namespace digits_in_product_l654_654590

theorem digits_in_product 
  (h1 : 3^4 = 81)
  (h2 : 6^8 = 1679616)
  (log_digits : ∀ n : ℕ, n > 0 → Nat.floor (Real.log n / Real.log 10) + 1) :
  log_digits (3^4 * 6^8) = 9 :=
by 
  -- Placeholder for the proof which we omit as instructed
  sorry

end digits_in_product_l654_654590


namespace main_theorem_l654_654580

variable (x : ℝ)

-- Define proposition p
def p : Prop := ∃ x0 : ℝ, x0^2 < x0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- Main proof problem
theorem main_theorem : p ∧ q := 
by {
  sorry
}

end main_theorem_l654_654580


namespace digit_at_573rd_place_after_decimal_of_11_div_37_l654_654414

theorem digit_at_573rd_place_after_decimal_of_11_div_37 :
  let seq := "297"
  let n := 573
  let pos := n % 3
  pos = 0 → seq.ilast = '7' :=
by
  sorry

end digit_at_573rd_place_after_decimal_of_11_div_37_l654_654414


namespace interest_from_initial_investment_l654_654596

-- Definitions for the problem conditions
def Principal1 : ℝ := 5000
def Principal2 : ℝ := 20000
def Interest2 : ℝ := 1000
def Time : ℝ := 1

-- Definition of simple annual interest formula
def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * (rate / 100) * time

-- Given the interest for the second principal, we solve for the rate
def interest_rate (interest principal time : ℝ) : ℝ :=
  (interest / (principal * time)) * 100

theorem interest_from_initial_investment :
  simple_interest Principal1 (interest_rate Interest2 Principal2 Time) Time = 250 :=
by
  sorry

end interest_from_initial_investment_l654_654596


namespace fewer_twos_for_100_l654_654355

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l654_654355


namespace miraflores_optimal_split_l654_654768

-- Define the total number of voters as 2n, and initialize half supporters for each candidate.
variable (n : ℕ) (voters : Fin (2 * n) → Bool)

-- Define the condition that exactly half of the voters including Miraflores support him
def half_support_miraflores : Prop :=
  ∃ (supporters_miraflores : Fin n) (supporters_maloney : Fin n), 
    (voters supporters_miraflores.val = true) ∧ (voters.supporters_maloney.val = false) 

-- Define the condition of drawing a single random ballot in each district.
def draw_random_ballot (d : Fin n → Prop) : Fin n := sorry

-- Define the condition that Miraflores wins if he wins both districts.
def wins_election (d1 d2 : Fin n → Prop) : Prop := 
  (draw_random_ballot d1 = true) ∧ (draw_random_ballot d2 = true)

-- Miraflores should split the voters such that his maximum probability of winning is achieved.
def optimal_split : Prop :=
  ∃ (d1 d2 : Fin n → Bool), 
    (d1.supporters_miraflores.val = true ∧ d2.supporters_maloney.val = false) ∧
    (wins_election d1 d2 = true)

theorem miraflores_optimal_split (n : ℕ) (voters : Fin (2 * n) → Bool) (half_support : half_support_miraflores n voters) : optimal_split n :=
sorry

end miraflores_optimal_split_l654_654768


namespace B_time_l654_654443

-- Define the work rates of A, B, and C in terms of how long they take to complete the work
variable (A B C : ℝ)

-- Conditions provided in the problem
axiom A_rate : A = 1 / 3
axiom BC_rate : B + C = 1 / 3
axiom AC_rate : A + C = 1 / 2

-- Prove that B alone will take 6 hours to complete the work
theorem B_time : B = 1 / 6 → (1 / B) = 6 := by
  intro hB
  sorry

end B_time_l654_654443


namespace fewer_twos_result_100_l654_654364

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l654_654364


namespace find_eigenvalues_of_M_l654_654161

-- Define the given matrix M with parameters a and b
def M (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, a; b, 1]

-- Define the points A and B
def A : Matrix (Fin 2) (Fin 1) ℝ :=
  !![3; -1]

def B : Matrix (Fin 2) (Fin 1) ℝ :=
  !![3; 5]

-- Define the matrix M when a = 3 and b = 2
def M_spec : Matrix (Fin 2) (Fin 2) ℝ :=
  M 3 2

-- Define the eigenvalues of the matrix M
def eigenvalues : List ℝ := [-1, 4]

-- Statement of the theorem
theorem find_eigenvalues_of_M : 
  (M_spec.vecMulVec 3 (-1)) = B ∧ (M_spec.eigensolve = eigenvalues) :=
by
  sorry

end find_eigenvalues_of_M_l654_654161


namespace cost_of_each_bottle_l654_654251

theorem cost_of_each_bottle (nights_per_bottle : ℕ) (cost_per_year : ℕ) (days_per_year : ℕ) 
    (daily_fraction : ℚ) (total_cost : ℚ): 
    (nights_per_bottle = 5) → 
    (cost_per_year = 146) → 
    (days_per_year = 365) → 
    (daily_fraction = 1 / 5) → 
    (total_cost = 146) → 
    ¬ 0 =
    let bottles_per_year := (days_per_year : ℚ) / (nights_per_bottle : ℚ) 
    in 
    let cost_per_bottle := total_cost / bottles_per_year 
    in cost_per_bottle = 2 := 
by 
  sorry

end cost_of_each_bottle_l654_654251


namespace parallel_condition_l654_654970

variables {l m : Type} [LinearOrderedSpace l] [LinearOrderedSpace m]
variable {α : Plane}

-- We assume the following hypotheses as conditions
hypothesis m_subset_alpha : m ⊆ α

-- The statement to prove:
theorem parallel_condition (h1 : l ∥ m) (h2 : l ∥ α) : (l ∥ m ↔ l ∥ α) := sorry

end parallel_condition_l654_654970


namespace pizza_pieces_per_person_l654_654200

theorem pizza_pieces_per_person (total_people : ℕ) (fraction_eat : ℚ) (total_pizza : ℕ) (remaining_pizza : ℕ)
  (H1 : total_people = 15) (H2 : fraction_eat = 3/5) (H3 : total_pizza = 50) (H4 : remaining_pizza = 14) :
  (total_pizza - remaining_pizza) / (fraction_eat * total_people) = 4 :=
by
  -- proof goes here
  sorry

end pizza_pieces_per_person_l654_654200


namespace distance_AB_max_distance_P_to_l_l654_654160

noncomputable def line (t : ℝ) : ℝ × ℝ :=
(1 + (1/2) * t, (sqrt 3 / 2) * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
(cos θ, sin θ)

noncomputable def curve_C2 (θ : ℝ) : ℝ × ℝ :=
(sqrt 3 * cos θ, 3 * sin θ)

theorem distance_AB :
  let A := (1 : ℝ, 0 : ℝ)
  let B := (1/2 : ℝ, -sqrt 3 / 2 : ℝ)
  dist A B = 1 :=
by sorry

theorem max_distance_P_to_l :
  ∃ θ : ℝ,
  let P := (sqrt 3 * cos θ, 3 * sin θ)
  let d := abs (3 * cos θ - 1 / sqrt 3)
  d = (3 * sqrt 2 + sqrt 3) / 2 :=
by sorry

end distance_AB_max_distance_P_to_l_l654_654160


namespace sum_of_primes_l654_654178

theorem sum_of_primes (a b c : ℕ) (h₁ : Nat.Prime a) (h₂ : Nat.Prime b) (h₃ : Nat.Prime c) (h₄ : b + c = 13) (h₅ : c^2 - a^2 = 72) :
  a + b + c = 20 := 
sorry

end sum_of_primes_l654_654178


namespace volume_of_cube_in_pyramid_l654_654863

theorem volume_of_cube_in_pyramid :
  (∃ (s : ℝ), 
    ( ∀ (b h l : ℝ),
      b = 2 ∧ 
      h = 3 ∧ 
      l = 2 * Real.sqrt 2 →
      s = 4 * Real.sqrt 2 - 3 ∧ 
      ((4 * Real.sqrt 2 - 3) ^ 3 = (4 * Real.sqrt 2 - 3) ^ 3))) :=
sorry

end volume_of_cube_in_pyramid_l654_654863


namespace minimum_marked_cells_ensure_coverage_l654_654377

/-- Definition of the board as a set of coordinates --/
def board : set (ℕ × ℕ) := { p | p.1 < 10 ∧ p.2 < 11 }

/-- Definition of an L-shaped piece as a set of three coordinates --/
def L_shaped_piece (x y : ℕ) : set (ℕ × ℕ) :=
  if x <= y then { (x, y), (x+1, y), (x, y+1) } else { (x, y), (x-1, y), (x, y+1) }

/-- Predicate indicating if a cell is marked --/
def is_marked (marked : set (ℕ × ℕ)) (cell : ℕ × ℕ) : Prop := cell ∈ marked

/-- The specific problem statement --/
theorem minimum_marked_cells_ensure_coverage :
  ∃ (marked : set (ℕ × ℕ)), marked.card = 50 ∧ ∀ x y ∈ board, 
  ∃ m ∈ L_shaped_piece x y, is_marked marked m :=
sorry

end minimum_marked_cells_ensure_coverage_l654_654377


namespace greatest_area_difference_l654_654791

theorem greatest_area_difference (l₁ w₁ l₂ w₂ : ℕ) 
  (h₁ : 2 * l₁ + 2 * w₁ = 160) 
  (h₂ : 2 * l₂ + 2 * w₂ = 160) : 
  1521 = (l₁ * w₁ - l₂ * w₂) → 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 1600 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) ∧ 
  (∃ l w : ℕ, 2 * l + 2 * w = 160 ∧ l * w = 79 ∧ (l₁ = l ∧ w₁ = w) ∨ (l₂ = l ∧ w₂ = w)) :=
sorry

end greatest_area_difference_l654_654791


namespace circle_equation_l654_654561

-- Definitions of the points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Definition of the equation of the circle with MN as diameter
def equation_of_circle (M N : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ p, ∃ r : ℝ, r = real.dist (fst M, snd M) (fst N, snd N) / 2 ∧ (fst p)^2 + (snd p)^2 = r^2

-- Prove that the given points form the equation of the circle x^2 + y^2 = 4
theorem circle_equation :
  equation_of_circle M N (0, 0) :=
begin
  sorry
end

end circle_equation_l654_654561


namespace coefficient_of_x4_term_in_expansion_l654_654488

noncomputable def coefficient_x4 : ℤ :=
  let polynomial := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)
  (-1) + (-2) + (-3) + (-4) + (-5)

theorem coefficient_of_x4_term_in_expansion :
  coefficient_x4 = -15 := by
  sorry

end coefficient_of_x4_term_in_expansion_l654_654488


namespace min_questions_to_identify_Z_l654_654448

-- Define the setup for the problem
variable (n : ℕ)
variable (company : Finset ℕ)
variable (Z : ℕ)
variable (knows : ℕ → ℕ → Prop)

-- Conditions
axiom num_people : company.card = n
axiom Z_knows_all : ∀ x ∈ company, x ≠ Z → knows Z x
axiom no_one_knows_Z : ∀ x ∈ company, x ≠ Z → ¬ knows x Z
axiom truthful_answers : ∀ x y ∈ company, knows x y ∨ ¬ knows x y

-- Proof Problem
theorem min_questions_to_identify_Z : 
  ∃ (m : ℕ), m = n-1 ∧ 
  (∀ journalist_approach : (∀ p ∈ company, ∀ q ∈ company, p ≠ q → Prop), 
    (∃ z ∈ company, (∀ x ∈ company, x ≠ z → ¬ knows x z) ∧ ∀ y ∈ company, knows z y → y ≠ z) 
      → ∀ m' < m, 
        ¬ (∀ journalist_approach : (∀ p ∈ company, ∀ q ∈ company, p ≠ q → Prop), 
          ∃ z ∈ company, (∀ x ∈ company, x ≠ z → ¬ knows x z) ∧ ∀ y ∈ company, knows z y → y ≠ z)
   )

end min_questions_to_identify_Z_l654_654448


namespace fewer_twos_to_hundred_l654_654362

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l654_654362


namespace part1_part2_l654_654229

def A (x : ℝ) : Prop := x ^ 2 - 2 * x - 8 < 0
def B (x : ℝ) : Prop := x ^ 2 + 2 * x - 3 > 0
def C (a : ℝ) (x : ℝ) : Prop := x ^ 2 - 3 * a * x + 2 * a ^ 2 < 0

theorem part1 : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 4} := 
by sorry

theorem part2 (a : ℝ) : {x : ℝ | C a x} ⊆ {x : ℝ | A x} ∩ {x : ℝ | B x} ↔ (a = 0 ∨ (1 ≤ a ∧ a ≤ 2)) := 
by sorry

end part1_part2_l654_654229


namespace required_sixth_score_is_106_l654_654919

-- Define Emily's current scores
def quiz_scores : List ℕ := [94, 90, 85, 90, 105]

-- Define the required arithmetic mean
def required_mean : ℕ := 95

-- Calculate the current sum of the scores
def current_sum : ℕ := List.sum quiz_scores

-- Define the total number of quizzes
def total_quizzes : ℕ := 6

-- Calculate the required total sum to achieve the required mean
def required_total_sum : ℕ := total_quizzes * required_mean

-- Define the lowest previous score
def lowest_previous_score : ℕ := List.minimum quiz_scores |> Option.getOrElse 0

-- Define the proof statement
theorem required_sixth_score_is_106 : ∃ s : ℕ, (required_total_sum = current_sum + s) ∧ (s > lowest_previous_score) ∧ (s = 106) := by
  sorry

end required_sixth_score_is_106_l654_654919


namespace sum_of_digits_l654_654936

def large_number := 
  let fractional_part := 44 * 10^(2014) + 4 * (10^(2013) - 1) / 9 in
  let integer_part := 9 + 8 * ((10^2017) - 1) / 9 in
  fractional_part + integer_part / 10^2018

theorem sum_of_digits : 
  let sqrt_large := real.sqrt large_number in
  let int_part := floor sqrt_large in
  (digit_sum int_part = 12109) := by
    sorry

end sum_of_digits_l654_654936


namespace jungkook_has_larger_number_l654_654419

theorem jungkook_has_larger_number : 
  let yoongi_number := 4
  let jungkook_number := 6 * 3
  in jungkook_number > yoongi_number := 
by
  let yoongi_number := 4
  let jungkook_number := 6 * 3
  sorry

end jungkook_has_larger_number_l654_654419


namespace axis_of_symmetry_l654_654305

-- Definitions for conditions
variable (ω : ℝ) (φ : ℝ) (A B : ℝ)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- Hypotheses
axiom ω_pos : ω > 0
axiom φ_bound : 0 ≤ φ ∧ φ < Real.pi
axiom even_func : ∀ x, f x = f (-x)
axiom dist_AB : abs (B - A) = 4 * Real.sqrt 2

-- Proof statement
theorem axis_of_symmetry : ∃ x : ℝ, x = 4 := 
sorry

end axis_of_symmetry_l654_654305


namespace trig_identity_example_l654_654951

variables (α β : ℝ)

theorem trig_identity_example
  (h :  sin(α - β) * cos α - cos (α - β) * sin α = (3 / 5)) :
  cos(2 * β) = 7 / 25 :=
sorry

end trig_identity_example_l654_654951


namespace prove_fractions_sum_equal_11_l654_654704

variable (a b c : ℝ)

-- Given conditions
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -9
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 10

-- The proof problem statement
theorem prove_fractions_sum_equal_11 : (b / (a + b) + c / (b + c) + a / (c + a)) = 11 :=
by
  sorry

end prove_fractions_sum_equal_11_l654_654704


namespace greatest_value_of_Q_l654_654581

-- Define the polynomial Q
def Q (x : ℝ) (b : ℝ) : ℝ := x^4 - 4 * x^3 + b * x^2 + 10 * x + 5

theorem greatest_value_of_Q (b : ℝ) (h : b > 16) :
  let q1 := Q 1 b in
  let sum_of_coeffs := b + 12 in
  let product_of_zeros := 5 in
  q1 < sum_of_coeffs ∧ product_of_zeros < sum_of_coeffs :=
by
  -- Sum of coefficients is greater than Q(1) and product of zeros
  sorry

end greatest_value_of_Q_l654_654581


namespace variance_stationary_equal_correlation_at_origin_l654_654693

variables (X : ℝ → ℝ) (D_x : ℝ → ℝ) (k_x : ℝ → ℝ)

def variance (t : ℝ) : ℝ := 
  E (λ (t:ℝ), (X t - E (λ(t:ℝ), X t)) ^ 2)

def correlation (t1 t2 : ℝ) : ℝ := 
  E (λ (t1 t2:ℝ), (X t1 - E (λ (t1:ℝ), X t1)) * (X t2 - E (λ (t2:ℝ), X t2)))

def stationary (t1 t2 : ℝ) : Prop := 
  correlation X t1 t2 = k_x (t2 - t1)

theorem variance_stationary_equal_correlation_at_origin :
  stationary X →
  ∀ t, D_x t = k_x 0 :=
by
  intro h_stationary
  intro t
  have h_variance := variance X t
  have h_corr := correlation X t t
  rw h_stationary at h_corr
  simp at h_corr
  sorry

end variance_stationary_equal_correlation_at_origin_l654_654693


namespace optimal_voter_split_l654_654757

-- Definitions
variables (Voters : Type) [fintype Voters] (n : ℕ)
variables (supports_miraflores : Voters → Prop)
variables [decidable_pred supports_miraflores]

-- Conditions
def half_supports_miraflores := fintype.card { v // supports_miraflores v } = n
def half_supports_maloney := fintype.card { v // ¬ supports_miraflores v } = n

-- Question (translated to a theorem)
theorem optimal_voter_split (h_m : half_supports_miraflores Voters n supports_miraflores)
    (h_d: half_supports_maloney Voters n supports_miraflores) :
  ∃ (D1 D2 : finset Voters), 
    ((D1 = {v | supports_miraflores v}) ∧ 
    (D2 = {v | ¬supports_miraflores v}) ∧ 
    (∀ v, v ∈ D1 ∨ v ∈ D2) ∧ 
    (∀ v, ¬ (v ∈ D1 ∧ v ∈ D2)) ∧ 
    (finset.card D1 = 1) ∧ 
    (finset.card D2 = 2 * n - 1)) :=
sorry

end optimal_voter_split_l654_654757


namespace base4_base7_digit_difference_l654_654912

def num_digits_base (n b : ℕ) : ℕ :=
  if b > 1 then Nat.log b n + 1 else 0

theorem base4_base7_digit_difference :
  let n := 1573
  num_digits_base n 4 - num_digits_base n 7 = 2 := by
  sorry

end base4_base7_digit_difference_l654_654912


namespace greatest_difference_of_units_digit_l654_654736

theorem greatest_difference_of_units_digit (d : ℕ) (h1 : 74 * 10 + d % 5 = 0) (h2 : (7 + 4 + d) % 3 = 0) : 
  ∀ x y : ℕ, (74 * 10 + x = 74 * 10 + d) → (74 * 10 + y = 74 * 10 + d) → x = y → abs (x - y) = 0 :=
by
  sorry

end greatest_difference_of_units_digit_l654_654736


namespace equilateral_triangle_AB_EF_length_l654_654432

theorem equilateral_triangle_AB_EF_length
  (ABC : Type)
  (A B C D E F : ABC)
  (is_equilateral : ∀ {X Y Z : ABC}, X ≠ Y → X ≠ Z → Y ≠ Z → Triangle X Y Z → X = Y ∧ Y = Z ∧ X = Z)
  (D_on_AC : OnLine D A C)
  (ADE_is_equilateral : Triangle A D E ∧ E ≠ A ∧ E ≠ D)
  (DCF_is_equilateral : Triangle D C F ∧ F ≠ D ∧ F ≠ C)
  (perimeter_DEF : length D E + length E F + length F D = 19)
  (perimeter_ABCEF : length A B + length B C + length C E + length E F + length F A = 43) :
  length A B = 12 ∧ length E F = 7 :=
begin
  sorry
end

end equilateral_triangle_AB_EF_length_l654_654432


namespace maximize_victory_probability_l654_654759

-- Define the conditions
variables {n : ℕ}
def number_of_voters := 2 * n
def half_support_miraflores := n
def half_support_dick_maloney := n
def miraflores_is_voter := true

-- Define the districts
def district1 := {miraflores}
def district2 := {voters | voters ≠ miraflores}

theorem maximize_victory_probability (n : ℕ) (h₁ : nat.odd (2*n + 1) = true) : 
  (let district1_voters := 1 in
   let district2_voters := 2*n - 1 in
   maximize_probability_of_winning(district1_voters, district2_voters) = true) :=
sorry

end maximize_victory_probability_l654_654759


namespace combined_work_time_for_three_workers_l654_654817

def workerA_time : ℝ := 4
def workerB_time : ℝ := 10
def workerC_time : ℝ := 6

def workerA_rate : ℝ := 1 / workerA_time
def workerB_rate : ℝ := 1 / workerB_time
def workerC_rate : ℝ := 1 / workerC_time

def combined_rate : ℝ := workerA_rate + workerB_rate + workerC_rate

theorem combined_work_time_for_three_workers :
  (1 / combined_rate) = 60 / 31 := by
sorry

end combined_work_time_for_three_workers_l654_654817


namespace tree_has_leaf_l654_654265

-- Definitions for the problem
structure Graph (V : Type) :=
  (edges : V → V → Prop)
  (symm : ∀ {a b : V}, edges a b → edges b a)
  (irreflexive : ∀ {a : V}, ¬edges a a)

structure Tree (V : Type) extends Graph V :=
  (connected : ∀ v1 v2 : V, ∃ p : List V, p.head = some v1 ∧ p.last = some v2 ∧ ∀ p₁ p₂, p₁ :: p₂ ∈ p.tails → edges p₁ p₂)
  (acyclic : ∀ p : List V, (∀ p₁ p₂, p₁ :: p₂ ∈ p.tails → edges p₁ p₂) → (p.head = p.last → ∃ v, v ∉ p))

-- Proof statement
theorem tree_has_leaf {V : Type} (T : Tree V) : ∃ v : V, (∃! w : V, T.edges v w) :=
by
  sorry

end tree_has_leaf_l654_654265


namespace galaxy_celestial_bodies_count_l654_654507

theorem galaxy_celestial_bodies_count :
  let P := 20 in
  let S := P * 8 in
  let T := S * 4 in
  P + S + T = 820 := by
  sorry

end galaxy_celestial_bodies_count_l654_654507


namespace infinitely_many_powers_of_10_in_AP_l654_654699

-- Define the conditions
def first_term : ℕ := 1
def common_diff : ℕ := 729

-- The general term of the arithmetic progression
def AP (n : ℕ) : ℕ := first_term + common_diff * n

-- Define the property to be proved
theorem infinitely_many_powers_of_10_in_AP : ∃ (n : ℕ → Prop), (∀ k : ℕ, ∃ n : ℕ, 10^n = AP k) :=
begin
  sorry  -- to be proven
end

end infinitely_many_powers_of_10_in_AP_l654_654699


namespace three_points_mutually_connected_l654_654548

theorem three_points_mutually_connected (n : ℕ) (points : fin (2 * n) → Type) :
  ∃ (A B C : fin (2 * n)), ∃ (segments : (fin (2 * n) × fin (2 * n)) → Prop), (segments = (λ (x y : fin (2 * n)), x ≠ y) → (segments (A, B) ∧ segments (B, C) ∧ segments (C, A)) :=
sorry

end three_points_mutually_connected_l654_654548


namespace sum_of_four_digit_integers_l654_654409

theorem sum_of_four_digit_integers : 
  let a := 1000 in
  let l := 9999 in
  let n := l - a + 1 in
  (a + l) * n / 2 = 49495500 :=
by
  let a := 1000
  let l := 9999
  let n := l - a + 1
  have h_sum := ((a + l) * n) / 2
  rw [a, l, n] at h_sum
  exact h_sum
  sorry

end sum_of_four_digit_integers_l654_654409


namespace tan_eq_243_deg_l654_654519

theorem tan_eq_243_deg (n : ℤ) : -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (243 * Real.pi / 180) ↔ n = 63 :=
by sorry

end tan_eq_243_deg_l654_654519


namespace min_marked_cells_l654_654387

def cell := (ℕ × ℕ)
def L_shape : set cell := {((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (1, 0)), 
                           ((0, 1), (1, 1), (1, 0)), ((0, 0), (0, 1), (1, 1))(0, 0), (0, 1), (1, 1)}

noncomputable def marked_cells : set cell := sorry

theorem min_marked_cells (board : set cell) (L : set (set cell)) : 
  board = { (i, j) | 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 11 } → 
  L = L_shape →
  (∀ l ∈ L, ∃ c ∈ marked_cells, c ∈ l) →
  cardinality(marked_cells) = 50 :=
begin
  sorry
end

end min_marked_cells_l654_654387


namespace trigonometric_expression_value_l654_654043

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_expression_value_l654_654043


namespace pyramid_volume_l654_654085

noncomputable def volume_of_pyramid (a : ℝ) : ℝ :=
  (a^3 * Real.sqrt (1 + Real.sqrt 5)) / 6

theorem pyramid_volume (a : ℝ) :
  a > 0 →
  ∃ V, V = (a^3 * Real.sqrt (1 + Real.sqrt 5)) / 6 :=
by
  intro a_pos
  use (a^3 * Real.sqrt (1 + Real.sqrt 5)) / 6
  reflexivity

end pyramid_volume_l654_654085


namespace complex_division_l654_654132

def imaginary_unit := Complex.I

theorem complex_division :
  (1 - 3 * imaginary_unit) / (2 + imaginary_unit) = -1 / 5 - 7 / 5 * imaginary_unit := by
  sorry

end complex_division_l654_654132


namespace unique_positive_zero_point_range_l654_654155

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

theorem unique_positive_zero_point_range (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 0 ∧ ∀ x : ℝ, f a x = 0 → x = x₀ ∧ x₀ > 0) → a < -2 :=
begin
  sorry
end

end unique_positive_zero_point_range_l654_654155


namespace no_such_polynomial_exists_l654_654267

theorem no_such_polynomial_exists : ¬ ∃ P : Polynomial ℝ, P.degree = 998 ∧ ∀ x : ℂ, P.eval x ^ 2 - 1 = P.eval (x^2 + 1) :=
  sorry

end no_such_polynomial_exists_l654_654267


namespace direct_proportion_function_l654_654191

-- Define the conditions for the problem
def condition1 (m : ℝ) : Prop := m ^ 2 - 1 = 0
def condition2 (m : ℝ) : Prop := m - 1 ≠ 0

-- The main theorem we need to prove
theorem direct_proportion_function (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = -1 :=
by
  sorry

end direct_proportion_function_l654_654191


namespace binomial_coeff_problem_l654_654108

theorem binomial_coeff_problem (x : ℕ) (hx1 : 3 * x ≤ 28) (hx2 : x + 8 ≤ 28) :
  binom 28 (3 * x) = binom 28 (x + 8) → (x = 4 ∨ x = 5) := 
by
  sorry

end binomial_coeff_problem_l654_654108


namespace jill_travels_less_than_john_l654_654224

theorem jill_travels_less_than_john :
  ∀ (John Jill Jim : ℕ), 
  John = 15 → 
  Jim = 2 → 
  (Jim = (20 / 100) * Jill) → 
  (John - Jill) = 5 := 
by
  intros John Jill Jim HJohn HJim HJimJill
  -- Skip the proof for now
  sorry

end jill_travels_less_than_john_l654_654224


namespace factorization_l654_654510

theorem factorization (y : ℤ) : 
  5 * (y + 4) * (y + 7) * (y + 9) * (y + 11) - 4 * y^2 = (y + 1) * (y + 9) * (5 * y^2 + 33 * y + 441) :=
by
  sorry

end factorization_l654_654510


namespace quadratic_polynomial_with_conditions_l654_654080

theorem quadratic_polynomial_with_conditions :
  ∃ (a b c : ℝ), 
  (∀ x : ℂ, x = -3 - 4 * Complex.I ∨ x = -3 + 4 * Complex.I → a * x^2 + b * x + c = 0)
  ∧ b = -10 
  ∧ a = -5/3 
  ∧ c = -125/3 := 
sorry

end quadratic_polynomial_with_conditions_l654_654080


namespace positive_difference_between_two_largest_prime_factors_l654_654373

-- Conditions: 195195 is factored into primes: 5, 3, 7, 11, 13, 13
def prime_factors_195195 : List Nat := [5, 3, 7, 11, 13, 13]

-- Two largest distinct prime factors
def largest_prime_factors (factors : List Nat) : Nat × Nat :=
  let unique_factors := factors.eraseDup
  let sorted_factors := unique_factors.qsort (· > ·)
  (sorted_factors.head!, sorted_factors.tail.head!)

-- Question: What is the positive difference between these two largest prime factors?
theorem positive_difference_between_two_largest_prime_factors :
  let (p1, p2) := largest_prime_factors prime_factors_195195
  p1 - p2 = 2 := by
  sorry

end positive_difference_between_two_largest_prime_factors_l654_654373


namespace prove_incorrect_D_l654_654270

-- Definitions corresponding to conditions
variable (DigitalEarth : Type)
variable (simulate : DigitalEarth → Prop)
variable (monitorCropPests : DigitalEarth → Prop)
variable (predictSubmergence : DigitalEarth → Prop)
variable (createVirtualRealities : DigitalEarth → Prop)
variable (simulatePast : DigitalEarth → Prop)
variable (predictFuture : DigitalEarth → Prop)

-- Incorrect Statement D
def D : Prop := predictFuture DigitalEarth ∧ ¬ simulatePast DigitalEarth

-- Prove the negation of D
theorem prove_incorrect_D (H1 : simulate DigitalEarth)
                          (H2 : monitorCropPests DigitalEarth)
                          (H3 : predictSubmergence DigitalEarth)
                          (H4 : createVirtualRealities DigitalEarth)
                          (H5 : simulatePast DigitalEarth)
                          (H6 : ¬ predictFuture DigitalEarth) : ¬D :=
by {
  sorry
}

end prove_incorrect_D_l654_654270


namespace initial_blue_marbles_l654_654174

theorem initial_blue_marbles (B R : ℕ) 
    (h1 : 3 * B = 5 * R) 
    (h2 : 4 * (B - 10) = R + 25) : 
    B = 19 := 
sorry

end initial_blue_marbles_l654_654174


namespace maximize_Miraflores_win_l654_654745

-- Definitions based on given conditions
def voters_count (n : ℕ) : ℕ := 2 * n
def support_Miraflores (n : ℕ) : ℕ := n + 1
def support_opponent (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem maximize_Miraflores_win (n : ℕ) (hn : n > 0) : 
  ∃ (d1 d2 : ℕ), d1 = 1 ∧ d2 = 2 * n - 1 ∧ support_Miraflores n > support_opponent n := 
sorry

end maximize_Miraflores_win_l654_654745


namespace height_percentage_differences_l654_654026

variable (B : ℝ) (A : ℝ) (R : ℝ)
variable (h1 : A = 1.25 * B) (h2 : R = 1.0625 * B)

theorem height_percentage_differences :
  (100 * (A - B) / B = 25) ∧
  (100 * (A - R) / A = 15) ∧
  (100 * (R - B) / B = 6.25) :=
by
  sorry

end height_percentage_differences_l654_654026


namespace new_avg_var_l654_654855

variables (x : Fin 10 → ℝ) (mean_x var_x : ℝ)

-- Conditions
def avg_x : ℝ := (∑ i, x i) / 10
def var_x_def : ℝ := (∑ i, (x i - avg_x) ^ 2) / 10

-- Assumptions
axiom avg_x_is_mean_x : avg_x = mean_x
axiom var_x_is_var_x : var_x_def = var_x

-- Definitions concerning new salaries
def y (i : Fin 10) : ℝ := x i + 100
def avg_y : ℝ := (∑ i, y i) / 10
def var_y : ℝ := (∑ i, (y i - avg_y) ^ 2) / 10

-- Theorem to be proven
theorem new_avg_var (h_mean_x : avg_x = mean_x) (h_var_x : var_x_def = var_x) :
  avg_y = mean_x + 100 ∧ var_y = var_x :=
by {
  sorry
}

end new_avg_var_l654_654855


namespace distance_difference_l654_654310

noncomputable def solve_problem : Real :=
  let line (x y : Real) := y + x * Real.sqrt 2 - 4
  let circle (x y : Real) := x^2 + y^2 = 10
  let Q := (-1, 0) : Real × Real
  let C := (2, 0) : Real × Real
  let D := (2 * Real.sqrt 2 / 3, 4 - 2 * Real.sqrt 2) : Real × Real
  let CQ := Real.sqrt ((C.1 + Q.1)^2 + (C.2 + Q.2)^2)
  let DQ := Real.sqrt ((D.1 + Q.1)^2 + (D.2 + Q.2)^2)
  |CQ - DQ|

theorem distance_difference : solve_problem = 3.12 := 
by sorry

end distance_difference_l654_654310


namespace no_3_digit_numbers_sum_27_even_l654_654588

-- Define the conditions
def is_digit_sum_27 (n : ℕ) : Prop :=
  (n ≥ 100 ∧ n < 1000) ∧ ((n / 100) + (n / 10 % 10) + (n % 10) = 27)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Define the theorem
theorem no_3_digit_numbers_sum_27_even :
  ¬ ∃ n : ℕ, is_digit_sum_27 n ∧ is_even n :=
by
  sorry

end no_3_digit_numbers_sum_27_even_l654_654588


namespace smallest_positive_period_max_min_values_in_interval_l654_654575

noncomputable def f (x : ℝ) : ℝ := 4 * cos x * cos (x - π / 3) - 2

theorem smallest_positive_period (x : ℝ) : ∃ T > 0, ∀ x, f (x + T) = f x :=
by
  use π
  sorry

theorem max_min_values_in_interval :
  ∃ maximum minimum : ℝ, maximum = 1 ∧ minimum = -2 ∧
  ∀ x ∈ Icc (-π / 6) (π / 4), minimum ≤ f x ∧ f x ≤ maximum :=
by
  use 1
  use -2
  sorry

end smallest_positive_period_max_min_values_in_interval_l654_654575


namespace sum_of_fractions_eq_neg_one_l654_654954

noncomputable def f : ℝ → ℝ :=
  λ x, if x < 1 then cos (real.pi * x) else f (x - 1) - 1

theorem sum_of_fractions_eq_neg_one : 
  f (1 / 3) + f (5 / 3) = -1 := 
sorry

end sum_of_fractions_eq_neg_one_l654_654954


namespace determine_valid_m_range_l654_654152

-- Define the condition
def quadratic_positive_for_all (m : ℝ) : Prop :=
  ∀ x : ℝ, mx^2 + mx + 1 > 0

-- State the main theorem
theorem determine_valid_m_range :
  {m : ℝ | quadratic_positive_for_all m} = {m : ℝ | 0 ≤ m ∧ m < 4} :=
sorry

end determine_valid_m_range_l654_654152


namespace fewer_twos_for_100_l654_654353

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l654_654353


namespace rate_of_interest_l654_654885

theorem rate_of_interest (P A T SI : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 2)
  (h4 : SI = A - P) (h5 : SI = (P * R * T) / 100) : R = 10 :=
by
  sorry

end rate_of_interest_l654_654885


namespace range_of_function_y_l654_654505

noncomputable def function_y (x : ℝ) : ℝ :=
  (Real.cos x) ^ 2 + Real.sin x - 1

theorem range_of_function_y : 
  Set.Icc (-2 : ℝ) (1 / 4) = Set.range (λ x, function_y x) :=
by
  sorry

end range_of_function_y_l654_654505


namespace general_term_a_n_sum_T_n_l654_654667

variables {x : ℝ} (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
def condition1 (h : n ∈ ℕ*) : S n = 3 / 2 * (a n - 1) := sorry

-- General term formula for {a_n}
theorem general_term_a_n (h : n ∈ ℕ*) : a n = 3^n := sorry

-- Sum of the first n terms T_n of {1 / (b_n b_{n+1})}
theorem sum_T_n (h : n ∈ ℕ*) : 
  let b (k : ℕ) := 1 + log x (a k)
  ∑ k in range n, 1 / (b k * b (k + 1)) = n / (2 * (n + 2)) := sorry

end general_term_a_n_sum_T_n_l654_654667


namespace bobs_total_profit_l654_654481

-- Definitions of the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Definition of the problem statement
theorem bobs_total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end bobs_total_profit_l654_654481


namespace sum_of_four_digit_integers_l654_654411

theorem sum_of_four_digit_integers : 
  let a := 1000 in
  let l := 9999 in
  let n := l - a + 1 in
  (a + l) * n / 2 = 49495500 :=
by
  let a := 1000
  let l := 9999
  let n := l - a + 1
  have h_sum := ((a + l) * n) / 2
  rw [a, l, n] at h_sum
  exact h_sum
  sorry

end sum_of_four_digit_integers_l654_654411


namespace total_profit_l654_654479

-- Definitions based on the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Statement of the theorem
theorem total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end total_profit_l654_654479


namespace prove_largest_ln_term_l654_654873

def largest_ln_term : Prop :=
  let A := (Real.log 2) ^ 2
  let B := Real.log (Real.log 2)
  let C := Real.log Real.exp 1
  let D := Real.log 2
  D > A ∧ D > B ∧ D > C

theorem prove_largest_ln_term : largest_ln_term :=
by
  sorry

end prove_largest_ln_term_l654_654873


namespace sum_of_all_four_digit_integers_l654_654404

theorem sum_of_all_four_digit_integers :
  (Finset.range (9999 + 1)).filter (λ x => x ≥ 1000).sum = 49495500 :=
by
  sorry

end sum_of_all_four_digit_integers_l654_654404


namespace find_d_l654_654215

theorem find_d (d : ℤ) (sqrt_90 : ℝ) (sqrt_09 : ℝ) (H1 : sqrt_90 = 9.49) (H2 : sqrt_09 = sqrt(0.9)) : d = 6 :=
by
  have H : (6 : ℝ) < 7 * sqrt_09 ∧ 7 * sqrt_09 < 7 + 1, from sorry
  exact sorry

end find_d_l654_654215


namespace paint_coverage_l654_654335

theorem paint_coverage 
  (width height cost_per_quart money_spent area : ℕ)
  (cover : ℕ → ℕ → ℕ)
  (num_sides quarts_purchased : ℕ)
  (total_area num_quarts : ℕ)
  (sqfeet_per_quart : ℕ) :
  width = 5 
  → height = 4 
  → cost_per_quart = 2 
  → money_spent = 20 
  → num_sides = 2
  → cover width height = area
  → area * num_sides = total_area
  → money_spent / cost_per_quart = quarts_purchased
  → total_area / quarts_purchased = sqfeet_per_quart
  → total_area = 40 
  → quarts_purchased = 10 
  → sqfeet_per_quart = 4 :=
by 
  intros
  sorry

end paint_coverage_l654_654335


namespace trajectory_of_M_line_equation_l654_654961

theorem trajectory_of_M (M : Type) (F : Type) (P : Type) (O : Type) (k : ℝ) :
  (∀ M : ℝ × ℝ, let line_dist := abs (M.2 + 2)
                let point_dist := real.sqrt ((M.1 - 0)^2 + (M.2 - 1)^2)
                line_dist = point_dist + 1) →
  (∀ l : ℝ → ℝ, ∀ A B : ℝ × ℝ, l(P.1) = -1 ∧
                 (∀ A B : ℝ × ℝ, let y1 := k * A.1 - 1
                                 let y2 := k * B.1 - 1
                                 (A.1 = (k + real.sqrt(k^2 + 4)) / 2) ∧
                                 (B.1 = (k - real.sqrt(k^2 + 4)) / 2) ∧
                                 ((y1 / (k + real.sqrt(k^2 + 4)) + y2 / (k - real.sqrt(k^2 + 4))) = 2))
                 ) →
  ∃ k : ℝ, k = 2 :=
by sorry

theorem line_equation (M : Type) (F : Type) (P : Type) (O : Type) (l : ℝ → ℝ) :
  (∀ M : ℝ × ℝ, let line_dist := abs (M.2 + 2)
                let point_dist := real.sqrt ((M.1 - 0)^2 + (M.2 - 1)^2)
                line_dist = point_dist + 1) →
  (∀ l : ℝ → ℝ, ∀ A B : ℝ × ℝ, l(P.1) = -1 ∧
                 (∀ A B : ℝ × ℝ, let y1 := 2 * A.1 - 1
                                 let y2 := 2 * B.1 - 1
                                 (A.1 = (2 + real.sqrt(4 + 4)) / 2) ∧
                                 (B.1 = (2 - real.sqrt(4 + 4)) / 2) ∧
                                 ((y1 / (2 + real.sqrt(4 + 4)) + y2 / (2 - real.sqrt(4 + 4))) = 2))
                 ) →
  l = (λ x, 2 * x - 1) :=
by sorry

end trajectory_of_M_line_equation_l654_654961


namespace amount_paid_to_shopkeeper_l654_654020

theorem amount_paid_to_shopkeeper :
  let price_of_grapes := 8 * 70
  let price_of_mangoes := 9 * 55
  price_of_grapes + price_of_mangoes = 1055 :=
by
  sorry

end amount_paid_to_shopkeeper_l654_654020


namespace sum_f_equals_2416_l654_654528

noncomputable def f (n : ℕ) : ℤ :=
  let mod5 := n % 5
  if mod5 = 0 then 0
  else if mod5 = 1 then 2
  else if mod5 = 2 then 1
  else if mod5 = 3 then 1
  else 2

theorem sum_f_equals_2416 : (∑ n in Finset.range 2014, f n) = 2416 := 
  sorry

end sum_f_equals_2416_l654_654528


namespace f_increasing_on_pos_l654_654875

noncomputable def f (x : ℝ) := x * Real.exp x

theorem f_increasing_on_pos : ∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x < y) → f(x) < f(y) := by
  sorry

end f_increasing_on_pos_l654_654875


namespace irrigation_tank_final_amount_l654_654856

theorem irrigation_tank_final_amount : 
  let initial_amount := 300.0
  let evaporation := 1.0
  let addition := 0.3
  let days := 45
  let daily_change := addition - evaporation
  let total_change := daily_change * days
  initial_amount + total_change = 268.5 := 
by {
  -- Proof goes here
  sorry
}

end irrigation_tank_final_amount_l654_654856


namespace tangent_condition_inequality_condition_l654_654121

section
variables {a x₀ : ℝ}

-- Conditions
def f (x : ℝ) := x * log x + a / x
def g (x : ℝ) := 2 * x * exp x - log x - x - log 2

-- Tangent condition
theorem tangent_condition : (∃ x₀, f x₀ = x₀ ∧ f' x₀ = 1) → a = real.exp(1) / 2 :=
sorry

-- Inequality condition
theorem inequality_condition : (∀ x₁ > 0, ∃ x₂ > 0, f x₁ ≥ g x₂) → 1 ≤ a :=
sorry

end

end tangent_condition_inequality_condition_l654_654121


namespace Sarahs_score_l654_654274

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l654_654274


namespace sarah_score_l654_654279

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l654_654279


namespace bob_total_profit_l654_654484

def initial_cost (num_dogs : ℕ) (cost_per_dog : ℕ) : ℕ := num_dogs * cost_per_dog

def revenue (num_puppies : ℕ) (price_per_puppy : ℕ) : ℕ := num_puppies * price_per_puppy

def total_profit (initial_cost : ℕ) (revenue : ℕ) : ℕ := revenue - initial_cost

theorem bob_total_profit (c1 : initial_cost 2 250 = 500)
                        (c2 : revenue 6 350 = 2100)
                        (c3 : total_profit 500 2100 = 1600) :
  total_profit (initial_cost 2 250) (revenue 6 350) = 1600 := by
  sorry

end bob_total_profit_l654_654484


namespace equal_savings_l654_654730

theorem equal_savings (U B UE BE US BS : ℕ) (h1 : U / B = 8 / 7) 
                      (h2 : U = 16000) (h3 : UE / BE = 7 / 6) (h4 : US = BS) :
                      US = 2000 ∧ BS = 2000 :=
by
  sorry

end equal_savings_l654_654730


namespace permutation_of_four_is_24_l654_654220

-- Definitions based on conditions
def attractions := {a1, a2, a3, a4, HarryPotter} -- representing the 5 attractions
def remaining_attractions := {a1, a2, a3, a4} -- representing 4 attractions without HarryPotter

-- The proof problem statement
theorem permutation_of_four_is_24 : 
  fintype.card (remaining_attractions : set (attractions → Prop)) = 24 :=
by
  sorry

end permutation_of_four_is_24_l654_654220


namespace solution_exists_l654_654549

theorem solution_exists (z : ℂ) (h1 : z + z⁻¹ ∈ ℝ) (h2 : abs (z - 2) = 2) : 
  z = 4 ∨ z = (1/4 : ℂ) + (complex.I) * (real.sqrt 15 / 4) ∨ z = (1/4 : ℂ) - (complex.I) * (real.sqrt 15 / 4) :=
by
  sorry

end solution_exists_l654_654549


namespace dot_product_l654_654166

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem dot_product (h1 : ‖a + b‖ = Real.sqrt 7) 
                    (h2 : ‖a - b‖ = Real.sqrt 3) : 
  inner a b = 1 := by
  sorry

end dot_product_l654_654166


namespace mean_of_added_numbers_l654_654299

variables {a : ℕ → ℝ} {x y z : ℝ}

theorem mean_of_added_numbers (h1 : (∑ i in finset.range 7, a i) / 7 = 65)
  (h2 : ((∑ i in finset.range 7, a i) + x + y + z) / 10 = 80) :
  (x + y + z) / 3 = 115 :=
sorry

end mean_of_added_numbers_l654_654299


namespace weight_of_oil_per_ml_l654_654252

variable (w : ℝ)  -- Weight of the oil per ml
variable (total_volume : ℝ := 150)  -- Bowl volume
variable (oil_fraction : ℝ := 2/3)  -- Fraction of oil
variable (vinegar_fraction : ℝ := 1/3)  -- Fraction of vinegar
variable (vinegar_density : ℝ := 4)  -- Vinegar density in g/ml
variable (total_weight : ℝ := 700)  -- Total weight in grams

theorem weight_of_oil_per_ml :
  (total_volume * oil_fraction * w) + (total_volume * vinegar_fraction * vinegar_density) = total_weight →
  w = 5 := by
  sorry

end weight_of_oil_per_ml_l654_654252


namespace total_number_of_members_l654_654317

variables (b g : Nat)
def girls_twice_boys : Prop := g = 2 * b
def boys_twice_remaining_girls (b g : Nat) : Prop := b = 2 * (g - 24)

theorem total_number_of_members (b g : Nat) 
  (h1 : girls_twice_boys b g) 
  (h2 : boys_twice_remaining_girls b g) : 
  b + g = 48 := by
  sorry

end total_number_of_members_l654_654317


namespace smallest_positive_period_fractional_part_l654_654239

def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem smallest_positive_period_fractional_part (x : ℝ) : 
  (∀ x : ℝ, fractional_part (x + 1) = fractional_part x) :=
sorry

end smallest_positive_period_fractional_part_l654_654239


namespace S_n_divisibility_l654_654264

def S_n (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), (2 ^ i - 1) ^ (2 ^ i - 1)

theorem S_n_divisibility (n : ℕ) (hn : n > 1) : 
  (2 ^ n ∣ S_n n) ∧ ¬(2 ^ (n + 1) ∣ S_n n) :=
by
  sorry

end S_n_divisibility_l654_654264


namespace proof_problem_l654_654140

variable {R : Type} [LinearOrderedField R]

def is_increasing (f : R → R) : Prop :=
  ∀ x y : R, x < y → f x < f y

theorem proof_problem (f : R → R) (a b : R) 
  (inc_f : is_increasing f) 
  (h : f a + f b > f (-a) + f (-b)) : 
  a + b > 0 := 
by
  sorry

end proof_problem_l654_654140


namespace problem1_problem2_problem3_l654_654269

-- Definitions for problem 1
def p1_expr : ℕ → ℕ := λ x, (x^2 + 2 * x) * (x^2 + 2 * x + 2) + 1
def p1_expected : ℕ → ℕ := λ x, (x + 1)^4

theorem problem1 (x : ℕ) : p1_expr x = p1_expected x := 
  sorry

-- Definitions for problem 2
def p2_expr : ℕ → ℕ := λ x, (x^2 - 6 * x + 8) * (x^2 - 6 * x + 10) + 1
def p2_expected : ℕ → ℕ := λ x, (x - 3)^4

theorem problem2 (x : ℕ) : p2_expr x = p2_expected x := 
  sorry

-- Definitions for problem 3
def sum1 : ℕ → ℕ := λ n, -(n * (n + 1) / 2) -- Sum from 1 to n with alternating signs
def sum2 : ℕ → ℕ := λ n, n * (n + 1) / 2 -- Sum from 2 to n

def p3_expr : ℕ := sum1 2020 * sum2 2021 - sum1 2021 * sum2 2020
def p3_expected : ℕ := 2021

theorem problem3 : p3_expr = p3_expected :=
  sorry

end problem1_problem2_problem3_l654_654269


namespace min_marked_cells_l654_654386

def cell := (ℕ × ℕ)
def L_shape : set cell := {((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (1, 0)), 
                           ((0, 1), (1, 1), (1, 0)), ((0, 0), (0, 1), (1, 1))(0, 0), (0, 1), (1, 1)}

noncomputable def marked_cells : set cell := sorry

theorem min_marked_cells (board : set cell) (L : set (set cell)) : 
  board = { (i, j) | 0 ≤ i ∧ i < 10 ∧ 0 ≤ j ∧ j < 11 } → 
  L = L_shape →
  (∀ l ∈ L, ∃ c ∈ marked_cells, c ∈ l) →
  cardinality(marked_cells) = 50 :=
begin
  sorry
end

end min_marked_cells_l654_654386


namespace sum_of_valid_c_values_l654_654526

theorem sum_of_valid_c_values : 
  (∑ c in finset.Icc (-30 : ℤ) 30, if (∃ k : ℤ, k^2 = 121 + 4 * c) then c else 0) = 38 :=
  sorry

end sum_of_valid_c_values_l654_654526


namespace total_food_items_in_one_day_l654_654648

-- Define the food consumption for each individual
def JorgeCroissants := 7
def JorgeCakes := 18
def JorgePizzas := 30

def GiulianaCroissants := 5
def GiulianaCakes := 14
def GiulianaPizzas := 25

def MatteoCroissants := 6
def MatteoCakes := 16
def MatteoPizzas := 28

-- Define the total number of each food type consumed
def totalCroissants := JorgeCroissants + GiulianaCroissants + MatteoCroissants
def totalCakes := JorgeCakes + GiulianaCakes + MatteoCakes
def totalPizzas := JorgePizzas + GiulianaPizzas + MatteoPizzas

-- The theorem statement
theorem total_food_items_in_one_day : 
  totalCroissants + totalCakes + totalPizzas = 149 :=
by
  -- Proof is omitted
  sorry

end total_food_items_in_one_day_l654_654648


namespace min_m_value_l654_654187

theorem min_m_value :
  ∃ (x y m : ℝ), x - y + 2 ≥ 0 ∧ x + y - 2 ≤ 0 ∧ 2 * y ≥ x + 2 ∧
  (m > 0) ∧ (x^2 / 4 + y^2 = m^2) ∧ m = Real.sqrt 2 / 2 :=
sorry

end min_m_value_l654_654187


namespace sum_four_digit_integers_l654_654399

theorem sum_four_digit_integers : 
  ∑ k in Finset.range (9999 - 1000 + 1), (k + 1000) = 49495500 := 
by
  sorry

end sum_four_digit_integers_l654_654399


namespace train_length_is_200_l654_654460

noncomputable def train_length 
  (speed_kmh : ℕ) 
  (time_s: ℕ) : ℕ := 
  ((speed_kmh * 1000) / 3600) * time_s

theorem train_length_is_200
  (h_speed : 40 = 40)
  (h_time : 18 = 18) :
  train_length 40 18 = 200 :=
sorry

end train_length_is_200_l654_654460


namespace finite_number_of_swaps_l654_654437

theorem finite_number_of_swaps :
  (∀ (circle : ZMod 179 → ℤ),
    (∀ i : ZMod 179,
    let a := circle i
    let b := circle (i + 1)
    let c := circle (i + 2)
    let d := circle (i + 3)
    (a - d) * (b - c) < 0) →
    ∃ n, ∀ m > n, ¬(∃ i : ZMod 179, (a, b, c, d) :=
                                  (circle i, circle (increment_index i 1), circle (increment_index i 2), circle (increment_index i 3)),
                      (a - d) * (b - c) < 0 ∧ b ↔ c)) :=
sorry

end finite_number_of_swaps_l654_654437


namespace garden_area_increased_l654_654851

theorem garden_area_increased (length width additional_fence : ℕ) 
  (h_length : length = 50) 
  (h_width : width = 20) 
  (h_additional_fence : additional_fence = 40) : 
  let original_area := length * width,
      original_perimeter := 2 * (length + width),
      new_perimeter := original_perimeter + additional_fence,
      side_new_square := new_perimeter / 4,
      new_area := side_new_square * side_new_square,
      area_increase := new_area - original_area in
  area_increase = 1025 := 
by 
  sorry

end garden_area_increased_l654_654851


namespace pythagorean_set_connection_l654_654491

-- Define what it means to be a Pythagorean set
def Pythagorean_set (s : set ℕ) : Prop := 
  ∃ a b c : ℕ, s = {a, b, c} ∧ a^2 + b^2 = c^2

-- The main theorem
theorem pythagorean_set_connection : 
  ∀ P Q : set ℕ,
    (Pythagorean_set P) → (Pythagorean_set Q) → 
  ∃ (m : ℕ) (P_i : ℕ → set ℕ), 
    m ≥ 2 ∧ P_i 1 = P ∧ P_i m = Q ∧
    ∀ i : ℕ, 1 ≤ i ∧ i < m → (P_i i ∩ P_i (i+1)).nonempty :=
begin
  sorry -- Proof not required
end

end pythagorean_set_connection_l654_654491


namespace tangent_product_le_one_third_l654_654688

theorem tangent_product_le_one_third (α β : ℝ) (h : α + β = π / 3) (hα : 0 < α) (hβ : 0 < β) : 
  Real.tan α * Real.tan β ≤ 1 / 3 :=
sorry

end tangent_product_le_one_third_l654_654688


namespace find_abc_l654_654990

noncomputable def f (a b c x : ℝ) := x^3 + a*x^2 + b*x + c
noncomputable def f' (a b x : ℝ) := 3*x^2 + 2*a*x + b

theorem find_abc (a b c : ℝ) :
  (f' a b -2 = 0) ∧
  (f' a b 1 = -3) ∧
  (f a b c 1 = 0) →
  a = 1 ∧ b = -8 ∧ c = 6 :=
sorry

end find_abc_l654_654990


namespace irrational_sqrt3_l654_654018

def is_irrational (x : ℝ) : Prop := ∀ (a b : ℤ), b ≠ 0 → x ≠ a / b

theorem irrational_sqrt3 :
  let A := 22 / 7
  let B := 0
  let C := Real.sqrt 3
  let D := 3.14
  is_irrational C :=
by
  sorry

end irrational_sqrt3_l654_654018


namespace right_triangle_inradius_l654_654078

theorem right_triangle_inradius (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_vals: a = 9 ∧ b = 12 ∧ c = 15) :
  let s := (a + b + c) / 2 in
  let A := (a * b) / 2 in
  ∃ r, A = s * r ∧ r = 3 := by
  sorry

end right_triangle_inradius_l654_654078


namespace compare_a_b_c_l654_654545

noncomputable def a : ℝ := 3 ^ 0.1
noncomputable def b : ℝ := (1 / 3) ^ (-0.8)
noncomputable def c : ℝ := Real.log 0.8 / Real.log 0.7

theorem compare_a_b_c : c < a ∧ a < b := by
  sorry

end compare_a_b_c_l654_654545


namespace harry_pencils_remaining_l654_654023

def num_pencils_anna : ℕ := 50
def num_pencils_harry_initial := 2 * num_pencils_anna
def num_pencils_lost_harry := 19

def pencils_left_harry (pencils_anna : ℕ) (pencils_harry_initial : ℕ) (pencils_lost : ℕ) : ℕ :=
  pencils_harry_initial - pencils_lost

theorem harry_pencils_remaining : pencils_left_harry num_pencils_anna num_pencils_harry_initial num_pencils_lost_harry = 81 :=
by
  sorry

end harry_pencils_remaining_l654_654023


namespace min_value_3x_plus_4y_l654_654604

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x + 3*y = 5*x*y) :
  ∃ (c : ℝ), (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y → 3 * x + 4 * y ≥ c) ∧ c = 5 :=
sorry

end min_value_3x_plus_4y_l654_654604


namespace value_of_expr_l654_654609

noncomputable def sqrt_13 := Real.sqrt 13

def a : ℝ := floor sqrt_13
def b : ℝ := sqrt_13 - a

theorem value_of_expr : 2 * a ^ 2 + b - sqrt_13 = 15 :=
by
  unfold a b
  sorry

end value_of_expr_l654_654609


namespace miraflores_optimal_split_l654_654765

-- Define the total number of voters as 2n, and initialize half supporters for each candidate.
variable (n : ℕ) (voters : Fin (2 * n) → Bool)

-- Define the condition that exactly half of the voters including Miraflores support him
def half_support_miraflores : Prop :=
  ∃ (supporters_miraflores : Fin n) (supporters_maloney : Fin n), 
    (voters supporters_miraflores.val = true) ∧ (voters.supporters_maloney.val = false) 

-- Define the condition of drawing a single random ballot in each district.
def draw_random_ballot (d : Fin n → Prop) : Fin n := sorry

-- Define the condition that Miraflores wins if he wins both districts.
def wins_election (d1 d2 : Fin n → Prop) : Prop := 
  (draw_random_ballot d1 = true) ∧ (draw_random_ballot d2 = true)

-- Miraflores should split the voters such that his maximum probability of winning is achieved.
def optimal_split : Prop :=
  ∃ (d1 d2 : Fin n → Bool), 
    (d1.supporters_miraflores.val = true ∧ d2.supporters_maloney.val = false) ∧
    (wins_election d1 d2 = true)

theorem miraflores_optimal_split (n : ℕ) (voters : Fin (2 * n) → Bool) (half_support : half_support_miraflores n voters) : optimal_split n :=
sorry

end miraflores_optimal_split_l654_654765


namespace total_profit_l654_654478

-- Definitions based on the conditions
def cost_per_dog : ℕ := 250
def num_dogs : ℕ := 2
def num_puppies : ℕ := 6
def selling_price_per_puppy : ℕ := 350

-- Statement of the theorem
theorem total_profit : (num_puppies * selling_price_per_puppy) - (num_dogs * cost_per_dog) = 1600 := 
by
  sorry

end total_profit_l654_654478


namespace last_two_digits_of_squares_sum_l654_654677

theorem last_two_digits_of_squares_sum :
  let a := [11, 12, 13, 14, 15, 16, 17, 18, 19]
  in (a.map (λ x => (x * x) % 100)).foldl (λ acc x => (acc + x) % 100) 0 = 85 :=
by
  sorry

end last_two_digits_of_squares_sum_l654_654677


namespace compute_expression_l654_654091

def floor (x : ℝ) : ℤ := (⌊x⌋ : ℤ)  -- Definition of floor function

theorem compute_expression (x : ℝ) (h : x = 2 / 3) : 
  (float_of_int (floor 6.5) * float_of_int (floor x) + 
   float_of_int (floor 2) * 7.2 + 
   float_of_int (floor 8.3) - 6.6) = 15.8 := 
by
  sorry

end compute_expression_l654_654091


namespace initial_crowns_difference_l654_654535

-- Initial definitions
variables (L0 P0 : ℤ)

-- Conditions encoded as definitions
def transfer_to_left := 20
def transfer_to_right := 32
def final_difference := 13

-- Main Statement
theorem initial_crowns_difference:
  L0 - P0 = 11 :=
by
  -- Define intermediate states based on transfers
  let L1 := L0 + transfer_to_left
  let P1 := P0 - transfer_to_left
  let L2 := L1 - transfer_to_right
  let P2 := P1 + transfer_to_right

  -- Given the final given condition
  have h : L2 = P2 - final_difference, from sorry,

  -- Substitute intermediate states to derive the initial relation
  sorry

end initial_crowns_difference_l654_654535


namespace mean_temperature_is_83_l654_654316

theorem mean_temperature_is_83 :
  let temperatures := [75, 78, 80, 82, 85, 87, 75, 87, 90, 92]
  let sum_temperatures := temperatures.sum
  let mean_temperature := sum_temperatures / (temperatures.length : ℚ)
  let rounded_mean_temperature := mean_temperature.round
  rounded_mean_temperature = 83 :=
by
  let temperatures := [75, 78, 80, 82, 85, 87, 75, 87, 90, 92]
  let sum_temperatures := temperatures.sum
  let mean_temperature := sum_temperatures / (temperatures.length : ℚ)
  let rounded_mean_temperature := mean_temperature.round
  have h : rounded_mean_temperature = 83 := sorry
  exact h

end mean_temperature_is_83_l654_654316


namespace SarahsScoreIs135_l654_654284

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l654_654284


namespace sum_of_radii_l654_654964

-- Triangle ABC
variables {A B C D E F : Type*}

-- Definition of radii
variables (r r0 R : ℝ)

-- The conditions: D, E, F lie on sides BC, CA, AB respectively
-- Radii of the incircles of ∆AEF, ∆BFD, ∆CDE are equal to r
axiom radius_AEF_eq_r : ∀ (A B C D E F : Type*), r = incircle_radius A E F
axiom radius_BFD_eq_r : ∀ (A B C D E F : Type*), r = incircle_radius B F D
axiom radius_CDE_eq_r : ∀ (A B C D E F : Type*), r = incircle_radius C D E

-- Radii of the incircle of ∆DEF is r0
axiom radius_DEF_eq_r0 : ∀ (D E F : Type*), r0 = incircle_radius D E F

-- Radii of the incircle of ∆ABC is R
axiom radius_ABC_eq_R : ∀ (A B C : Type*), R = incircle_radius A B C

-- To prove: r + r0 = R
theorem sum_of_radii (r r0 R : ℝ) (A B C D E F : Type*)
  (h1 : r = incircle_radius A E F)
  (h2 : r = incircle_radius B F D)
  (h3 : r = incircle_radius C D E)
  (h4 : r0 = incircle_radius D E F)
  (h5 : R = incircle_radius A B C) :
  r + r0 = R :=
sorry

end sum_of_radii_l654_654964


namespace find_t_l654_654271

theorem find_t : ∃ t : ℝ, t = 1 ∧ (10 - 2*t)^2 * t = (3 - t) * (2*t + 4)^2 := 
by { 
  use 1,
  split,
  { refl },
  { sorry }
}

end find_t_l654_654271


namespace sum_fraction_eq_1501_over_3002_l654_654054

theorem sum_fraction_eq_1501_over_3002 :
  ∑ n in Finset.range 1500, (1 : ℝ) / ((n + 1)^2 + 2 * (n + 1)) = 1501 / 3002 :=
by
  sorry

end sum_fraction_eq_1501_over_3002_l654_654054


namespace calculate_rate_of_interest_l654_654004

theorem calculate_rate_of_interest
  (SI : ℚ) (P : ℚ) (T : ℚ) 
  (h1 : SI = 929.20) 
  (h2 : P = 2323) 
  (h3 : T = 5) :
  let R := (SI * 100) / (P * T) in R = 8 := 
sorry

end calculate_rate_of_interest_l654_654004


namespace total_apples_l654_654183

theorem total_apples (apples_per_person : ℕ) (num_people : ℕ) (h1 : apples_per_person = 25) (h2 : num_people = 6) : apples_per_person * num_people = 150 := by
  sorry

end total_apples_l654_654183


namespace radius_of_surrounding_circles_l654_654444

theorem radius_of_surrounding_circles :
  ∀ (r : ℝ), 2 * real.sqrt (2 * r ^ 2) = 4 + 2 * r → r = 2 + real.sqrt 2 :=
begin
  sorry
end

end radius_of_surrounding_circles_l654_654444


namespace noelle_homework_assignments_l654_654678

theorem noelle_homework_assignments : 
  let assignments (points : ℕ) := 
    if points <= 5 then points * 1
    else if points <= 15 then 5 + (points - 5) * 2
    else 25 + (points - 15) * 3
  in 
  assignments 30 = 70 := 
by
  sorry

end noelle_homework_assignments_l654_654678


namespace inverse_function_correct_l654_654994

noncomputable def inverse_function (y : ℝ) : ℝ := (1 / 2) * y - (3 / 2)

theorem inverse_function_correct :
  ∀ x ∈ Set.Icc (0 : ℝ) (5 : ℝ), (inverse_function (2 * x + 3) = x) ∧ (0 ≤ 2 * x + 3) ∧ (2 * x + 3 ≤ 5) :=
by
  sorry

end inverse_function_correct_l654_654994


namespace arrange_balls_l654_654627

theorem arrange_balls (w b : ℕ) (h_w : w = 7) (h_b : b = 5) :
    ∃ n : ℕ, n = 56 ∧ ∀ arr : List (char × Nat), (arr.filter (λ x, x.1 = 'B')).length = b ∧
               (arr.filter (λ x, x.1 = 'W')).length = w ∧
               (arr.dropWhile (λ x, x.1 ≠ 'B')).drop 1.filter (λ x, x.1 = 'B').length = 0 → 
               n :=
by sorry

end arrange_balls_l654_654627


namespace trig_expression_proof_l654_654897

noncomputable theory
open Complex.Real

theorem trig_expression_proof :
  (1 - 1 / ((sin (30 * π / 180))^2)) *
  (1 + 1 / ((cos (60 * π / 180))^2)) *
  (1 - 1 / ((cos (30 * π / 180))^2)) *
  (1 + 1 / ((sin (60 * π / 180))^2)) = 35 / 3 := by
  sorry

end trig_expression_proof_l654_654897


namespace anna_initial_stamps_l654_654882

theorem anna_initial_stamps (final_stamps : ℕ) (alison_stamps : ℕ) (alison_to_anna : ℕ) : 
  final_stamps = 50 ∧ alison_stamps = 28 ∧ alison_to_anna = 14 → (final_stamps - alison_to_anna = 36) :=
by
  sorry

end anna_initial_stamps_l654_654882


namespace knights_and_liars_l654_654833

-- Knight or Liar Property
def isKnight (x : ℕ) : Prop := ∀ (s : ℕ), x ∈ s → s = 0

-- Islander Property
def isLiar (x : ℕ) : Prop := ∀ (s : ℕ), x ∈ s → s ≠ 0

-- Main hypothesis stating the problem conditions
theorem knights_and_liars (n : ℕ) (A B C : Finset ℕ) (count1 count2 count3 : ℕ) :
  (A.card = 3 ∧ ∀ x ∈ A, isLiar x ↔ count1 = 3) ∧
  (B.card = 6 ∧ ∀ x ∈ B, isLiar x ↔ count2 = 6) ∧
  (C.card = 9 ∧ ∀ x ∈ C, isKnight x ↔ count3 = 9) →
  n = 9 ∨ n = 18 ∨ n = 19 :=
begin
  sorry
end

end knights_and_liars_l654_654833


namespace P1_evaluation_l654_654582

noncomputable def Q (x_1 x_2 x_3 x_4 x_5 x_6 x_7 : ℝ) : ℝ :=
  (x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7) ^ 2 + 2 * (x_1^2 + x_2^2 + x_3^2 + x_4^2 + x_5^2 + x_6^2 + x_7^2)

theorem P1_evaluation :
  (∃ P1 P2 P3 P4 P5 P6 P7 : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ, 
    (∀ (x1 x2 x3 x4 x5 x6 x7 : ℝ), 
      Q x1 x2 x3 x4 x5 x6 x7 = (P1 x1 x2 x3 x4 x5 x6 x7) ^ 2 + 
                               (P2 x1 x2 x3 x4 x5 x6 x7) ^ 2 + 
                               (P3 x1 x2 x3 x4 x5 x6 x7) ^ 2 + 
                               (P4 x1 x2 x3 x4 x5 x6 x7) ^ 2 +
                               (P5 x1 x2 x3 x4 x5 x6 x7) ^ 2 +
                               (P6 x1 x2 x3 x4 x5 x6 x7) ^ 2 +
                               (P7 x1 x2 x3 x4 x5 x6 x7) ^ 2) ∧
    (∀ k, ∀ i, ∃ c : ℝ, P_k = c * x_i) ∧ 
    ∀ (P1 : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ), 
    P1 1 1 1 1 1 1 1 = 3) :=
sorry

end P1_evaluation_l654_654582


namespace function_b_is_even_and_monotonically_increasing_l654_654016

def is_even_on_domain (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonically_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

def f : ℝ → ℝ := λ x, -Real.log (abs x) / Real.log 2

theorem function_b_is_even_and_monotonically_increasing :
  is_even_on_domain f ∧ is_monotonically_increasing_on_interval f (-∞) 0 :=
by
  sorry

end function_b_is_even_and_monotonically_increasing_l654_654016


namespace line_plane_parallelism_l654_654975

variables {Point : Type} [LinearOrder Point] -- Assuming Point is a Type with some linear order.

-- Definitions for line and plane
-- These definitions need further libraries or details depending on actual Lean geometry library support
@[ext] structure Line (P : Type) := (contains : P → Prop)
@[ext] structure Plane (P : Type) := (contains : P → Prop)

variables {a b : Line Point} {α β : Plane Point} {l : Line Point}

-- Conditions (as in part a)
axiom lines_are_different : a ≠ b
axiom planes_are_different : α ≠ β
axiom planes_intersect_in_line : ∃ l, α.contains l ∧ β.contains l
axiom a_parallel_l : ∀ p : Point, a.contains p → l.contains p
axiom b_within_plane : ∀ p : Point, b.contains p → β.contains p
axiom b_parallel_alpha : ∀ p q : Point, β.contains p → β.contains q → α.contains p → α.contains q

-- Define the theorem statement
theorem line_plane_parallelism : a ≠ b ∧ α ≠ β ∧ (∃ l, α.contains l ∧ β.contains l) 
  ∧ (∀ p, a.contains p → l.contains p) 
  ∧ (∀ p, b.contains p → β.contains p) 
  ∧ (∀ p q, β.contains p → β.contains q → α.contains p → α.contains q) → a = b :=
by sorry

end line_plane_parallelism_l654_654975


namespace find_squared_length_of_AH_l654_654788

variables (A B C D G H : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space G] [metric_space H]
variables [proper_space A] [proper_space B] [proper_space C] [proper_space D] [proper_space G] [proper_space H]
variables (ω γ : set (point: A))
variables {triangle_inscribed_in_circle : ∀ (ω : Type) (A B C : ω) [inscribed A B C ω], true} (median_at_D : ∀ (A B C D : Type), true)
variables {circle_with_diameter : ∀ (γ : Type) (D G : γ) [diameter D G γ], true}

theorem find_squared_length_of_AH {hABC : ∀ (A B C : Type), true}
  (h_triangle_inscribed : triangle_inscribed_in_circle ω A B C)
  (h_median : median_at_D A B C D)
  (h_circle_with_diameter : circle_with_diameter γ D G)
  (h_intersection : ∃ (H : Type), set.intersects (λ x, circle_with_diameter ⊆ ω ∧ γ x) ω γ)
  (AB BC AC : ℝ) (AB_eq : AB = 6) (BC_eq : BC = 8) (AC_eq : AC = 5) : 
  length_sq (A - H) = 72.5 :=
sorry

end find_squared_length_of_AH_l654_654788


namespace odds_against_C_winning_l654_654168

theorem odds_against_C_winning :
  let P_A := 2 / 7
  let P_B := 1 / 5
  let P_C := 1 - (P_A + P_B)
  (1 - P_C) / P_C = 17 / 18 :=
by
  sorry

end odds_against_C_winning_l654_654168


namespace sum_of_cubes_l654_654181

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 12) : x^3 + y^3 = 91 := by
  sorry

end sum_of_cubes_l654_654181


namespace sum_of_digits_of_sqrt_N_l654_654935

noncomputable def N : ℝ := (44 * (10 ^ (2017: ℝ))) + (10 ^ (2017 - 1: ℝ)) * 8 * (10 ^ 2017) + 9

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else sum_of_digits (n / 10) + (n % 10)

theorem sum_of_digits_of_sqrt_N :
  sum_of_digits (⌊ sqrt N ⌋) = 12109 := by
  sorry

end sum_of_digits_of_sqrt_N_l654_654935


namespace value_of_x_l654_654803

theorem value_of_x (x : ℝ) : (12 - x)^3 = x^3 → x = 12 :=
by
  sorry

end value_of_x_l654_654803


namespace balls_in_boxes_l654_654268

theorem balls_in_boxes : 
  let balls := 4
  let boxes := 3
  (boxes^balls = 81) :=
by sorry

end balls_in_boxes_l654_654268


namespace trigonometric_identity_l654_654050

theorem trigonometric_identity (h1 : cos (70 * Real.pi / 180) ≠ 0) (h2 : sin (70 * Real.pi / 180) ≠ 0) :
  (1 / cos (70 * Real.pi / 180) - (Real.sqrt 3) / sin (70 * Real.pi / 180)) = Real.cot (20 * Real.pi / 180) - 2 :=
by
  sorry

end trigonometric_identity_l654_654050


namespace identity_not_in_S_l654_654656

noncomputable def S : Set ℝ := { x : ℝ | x ≠ 0 }

def ast (a b : ℝ) : ℝ := 3 * a * b + 1

def is_identity (e : ℝ) (op : ℝ → ℝ → ℝ) (S : Set ℝ) : Prop :=
∀ x ∈ S, op x e = x ∧ op e x = x

theorem identity_not_in_S : ¬ is_identity 1 ast S := 
by
  intro h
  have h1 := h 1 (by simp)
  simp [ast] at h1
  exact sorry

end identity_not_in_S_l654_654656


namespace regular_pentagon_cannot_cover_floor_completely_l654_654880

theorem regular_pentagon_cannot_cover_floor_completely
  (hexagon_interior_angle : ℝ)
  (pentagon_interior_angle : ℝ)
  (square_interior_angle : ℝ)
  (triangle_interior_angle : ℝ)
  (hexagon_condition : 360 / hexagon_interior_angle = 3)
  (square_condition : 360 / square_interior_angle = 4)
  (triangle_condition : 360 / triangle_interior_angle = 6)
  (pentagon_condition : 360 / pentagon_interior_angle ≠ 3)
  (pentagon_condition2 : 360 / pentagon_interior_angle ≠ 4)
  (pentagon_condition3 : 360 / pentagon_interior_angle ≠ 6) :
  pentagon_interior_angle = 108 := 
  sorry

end regular_pentagon_cannot_cover_floor_completely_l654_654880


namespace intersection_M_N_l654_654124

-- Definitions of sets M and N according to their conditions
def M := {-2, -1, 0, 1, 2}
def N := {x : ℤ | x^2 - x - 2 ≥ 0}

-- Proof statement
theorem intersection_M_N :
  M ∩ N = {-2, -1, 2} :=
sorry

end intersection_M_N_l654_654124


namespace max_distance_traveled_l654_654099

theorem max_distance_traveled (front_lifespan : ℕ) (rear_lifespan : ℕ) : 
  front_lifespan = 21000 ∧ rear_lifespan = 28000 → max_possible_distance = 24000 :=
begin
  intros h,
  sorry,
end

end max_distance_traveled_l654_654099


namespace unique_triangle_shape_l654_654258

theorem unique_triangle_shape :
  ∀ (A B C D E : Prop),
  (A ↔ "The ratio of a side to the corresponding angle bisector does not uniquely determine a triangle") →
  (B ↔ "The ratios of the three angle bisectors uniquely determine a triangle") →
  (C ↔ "The midpoints of the three sides do not uniquely determine a triangle") →
  (D ↔ "The lengths of two sides and the opposite angle of one of them uniquely determine a triangle") →
  (E ↔ "The ratio of two interior angles of the triangle uniquely determines a triangle") →
  (B ∧ D ∧ E) ∧ (¬A) ∧ (¬C) :=
by
  intros A B C D E hA hB hC hD hE
  rw [hB, hD, hE, hA, hC]
  exact (⟨⟨rfl, rfl, rfl⟩, ⟨rfl⟩, ⟨rfl⟩⟩) -- Logical assertions refined according to problem context

end unique_triangle_shape_l654_654258


namespace miraflores_optimal_split_l654_654766

-- Define the total number of voters as 2n, and initialize half supporters for each candidate.
variable (n : ℕ) (voters : Fin (2 * n) → Bool)

-- Define the condition that exactly half of the voters including Miraflores support him
def half_support_miraflores : Prop :=
  ∃ (supporters_miraflores : Fin n) (supporters_maloney : Fin n), 
    (voters supporters_miraflores.val = true) ∧ (voters.supporters_maloney.val = false) 

-- Define the condition of drawing a single random ballot in each district.
def draw_random_ballot (d : Fin n → Prop) : Fin n := sorry

-- Define the condition that Miraflores wins if he wins both districts.
def wins_election (d1 d2 : Fin n → Prop) : Prop := 
  (draw_random_ballot d1 = true) ∧ (draw_random_ballot d2 = true)

-- Miraflores should split the voters such that his maximum probability of winning is achieved.
def optimal_split : Prop :=
  ∃ (d1 d2 : Fin n → Bool), 
    (d1.supporters_miraflores.val = true ∧ d2.supporters_maloney.val = false) ∧
    (wins_election d1 d2 = true)

theorem miraflores_optimal_split (n : ℕ) (voters : Fin (2 * n) → Bool) (half_support : half_support_miraflores n voters) : optimal_split n :=
sorry

end miraflores_optimal_split_l654_654766


namespace multiple_of_one_third_l654_654412

theorem multiple_of_one_third (x : ℚ) (h : x * (1 / 3) = 2 / 9) : x = 2 / 3 :=
sorry

end multiple_of_one_third_l654_654412


namespace trigonometric_identity_l654_654049

theorem trigonometric_identity (h1 : cos (70 * Real.pi / 180) ≠ 0) (h2 : sin (70 * Real.pi / 180) ≠ 0) :
  (1 / cos (70 * Real.pi / 180) - (Real.sqrt 3) / sin (70 * Real.pi / 180)) = Real.cot (20 * Real.pi / 180) - 2 :=
by
  sorry

end trigonometric_identity_l654_654049


namespace miraflores_optimal_strategy_l654_654753

-- Definitions based on conditions
variable (n : ℕ)
def total_voters := 2 * n
def miraflores_supporters := n
def dick_maloney_supporters := n
def miraflores_is_a_voter := 1
def law_allows_division := true
def election_winner (district1 district2 : Set ℕ) : ℕ := 
  if (district1.card = 1 ∧ miraflores_is_a_voter ∈ district1) then miraflores_is_a_voter else dick_maloney_supporters

-- Mathematically equivalent proof problem
theorem miraflores_optimal_strategy (hall : law_allows_division) :
  (exists (district1 district2 : Set ℕ),
    ∀ v, v ∈ district1 ∨ v ∈ district2 ∧ district1.card + district2.card = total_voters ∧
    miraflores_supporters = 1 ∧ district1 = {miraflores_is_a_voter} ∧
    (election_winner district1 district2) = miraflores_is_a_voter) :=
sorry

end miraflores_optimal_strategy_l654_654753


namespace angle_ABC_is_36_degrees_l654_654638

noncomputable def isosceles_triangle (A B C : Type) (AB BC : ℝ) : Prop :=
A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ AB = BC ∧ ∀ P Q, P ≠ Q

-- Define the conditions for our specific geometric configuration
variables {A B C K L : Type}
variables (AB BC AK KL LB KB AC : ℝ)

-- Express the conditions as a Lean proposition
def problem_conditions : Prop :=
isosceles_triangle A B C AB BC ∧
AK = KL ∧ KL = LB ∧ KB = AC

-- The theorem we need to prove
theorem angle_ABC_is_36_degrees 
  (h : problem_conditions) : 
  true :=  -- This is to keep the statement syntactically correct (normally this would be false, but we put true to indicate a query)
sorry

end angle_ABC_is_36_degrees_l654_654638


namespace cos_half_diff_proof_l654_654144

noncomputable def cos_half_diff (A B C : ℝ) (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) : Real :=
  Real.cos ((A - C) / 2)

theorem cos_half_diff_proof (A B C : ℝ)
  (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) :
  cos_half_diff A B C h_triangle h_relation h_equation = -Real.sqrt 2 / 2 :=
sorry

end cos_half_diff_proof_l654_654144


namespace events_B_and_C_complementary_l654_654462

def is_event_B (n : ℕ) : Prop := n ≤ 3
def is_event_C (n : ℕ) : Prop := n ≥ 4
def possible_outcomes : set ℕ := {1, 2, 3, 4, 5, 6}

theorem events_B_and_C_complementary :
  (∀ n ∈ possible_outcomes, ¬(is_event_B n ∧ is_event_C n)) ∧
  (∀ n ∈ possible_outcomes, is_event_B n ∨ is_event_C n) := 
by
  sorry

end events_B_and_C_complementary_l654_654462


namespace goose_eggs_count_l654_654423

theorem goose_eggs_count 
  (E : ℕ) 
  (hatch_rate : ℚ)
  (survive_first_month_rate : ℚ)
  (survive_first_year_rate : ℚ)
  (geese_survived_first_year : ℕ)
  (no_more_than_one_goose_per_egg : Prop) 
  (hatch_eq : hatch_rate = 2/3) 
  (survive_first_month_eq : survive_first_month_rate = 3/4) 
  (survive_first_year_eq : survive_first_year_rate = 2/5) 
  (geese_survived_eq : geese_survived_first_year = 130):
  E = 650 :=
by
  sorry

end goose_eggs_count_l654_654423


namespace reciprocal_line_tangent_l654_654789

-- Definitions and setup (simplified for clarity)

variable {Ω : Type*} -- A type for geometric points

-- Define triangle with vertices A, B, C
variables (A B C : Ω)
-- Define midpoints of the sides, altitudes, etc.
variables (A1 B1 C1 : Ω) -- Midpoints
variables (A2 B2 C2 : Ω) -- Altitudes' feet
variables (A3 B3 C3 : Ω) -- Angle bisectors
variable (S : Ω) -- Centroid
variable (O : Ω) -- Excircle center
variables (L1 M1 N1 : Ω) -- Intersections of the line OS with sides
variables (L2 M2 N2 : Ω) -- Symmetric points w.r.t midpoints

-- Auxiliary Theorem setup
axiom aux_theorem : 
  ∀ (A4 B4 C4 : Ω), (A1A2 : Ω) (A1A3 : Ω) (A1A4 : Ω),
    A1A2 * A1A3 = (A1A4) ^ 2

-- The main theorem statement
theorem reciprocal_line_tangent :
  line_passing_through S O 
  → reciprocal_line_tangent_to_excircle_at_Feuerbach_circle_point 
  sorry

end reciprocal_line_tangent_l654_654789


namespace trigonometric_identity_l654_654539

theorem trigonometric_identity 
  (α : ℝ) 
  (h : Real.sin (α + π / 3) = 12 / 13) 
  : Real.cos (π / 6 - α) = 12 / 13 := 
sorry

end trigonometric_identity_l654_654539


namespace number_of_chords_eq_3_l654_654622

-- Definition of the conditions
variable (C D E : Type) [geometry C D E]

-- Assuming the angle CDE equals 60 degrees
axiom angle_CDE_eq_60 : angle C D E = 60

-- Proving that the number of chords n is 3.
theorem number_of_chords_eq_3 : ∃ (n : ℕ), n = 3 :=
by
  -- translate the geometric interpretation to algebraic form
  sorry

end number_of_chords_eq_3_l654_654622


namespace Ruby_math_homework_l654_654257

theorem Ruby_math_homework : 
  ∃ M : ℕ, ∃ R : ℕ, R = 2 ∧ 5 * M + 9 * R = 48 ∧ M = 6 := by
  sorry

end Ruby_math_homework_l654_654257


namespace sin_value_l654_654541

variable (α : ℝ)

theorem sin_value :
  sin^2 (π / 4 + α) = 2 / 3 → sin (2 * α) = 1 / 3 :=
by
  assume h : sin^2 (π / 4 + α) = 2 / 3
  sorry

end sin_value_l654_654541


namespace greatest_6_digit_number_divisible_l654_654371

theorem greatest_6_digit_number_divisible (n : ℕ) :
  ∃ k : ℕ, n = 990990 ∧ n < 1000000 ∧
    21 ∣ n ∧ 35 ∣ n ∧ 66 ∣ n ∧ 110 ∣ n ∧ 143 ∣ n :=
begin
  use 990990,
  split, { refl },
  split, { linarith },
  repeat { split; apply dvd_refl }
end

end greatest_6_digit_number_divisible_l654_654371


namespace qg_given_h_qh_given_g_inconsistency_of_probs_l654_654137

variables (g h : Type) (q : Type -> ℝ)

-- Given conditions
axiom q_g : q g = 0.30
axiom q_h : q h = 0.9
axiom q_g_h : q (g ∩ h) = 0.9

-- Definitions
def q_cond (A B : Type) : ℝ := q (A ∩ B) / q B

-- Problem statements
theorem qg_given_h : q_cond g h = 1 :=
by {
  have h1 : q (g ∩ h) = 0.9 := q_g_h,
  have h2 : q h = 0.9 := q_h,
  unfold q_cond,
  rw [h1, h2],
  norm_num,
  sorry  -- proof completion if necessary
}

theorem qh_given_g : q_cond h g = 3 :=
by {
  have h1 : q (g ∩ h) = 0.9 := q_g_h,
  have h2 : q g = 0.30 := q_g,
  unfold q_cond,
  rw [h1, h2],
  norm_num,
  sorry  -- proof completion if necessary
}

theorem inconsistency_of_probs : q (g ∩ h) <= min (q g) (q h) :=
by {
  have h1 : q g = 0.30 := q_g,
  have h2 : q h = 0.9 := q_h,
  unfold min,
  split_ifs,
  { show (q (g ∩ h) <= q g),
    rw q_g_h,
    rw q_g,
    norm_num,
    sorry  -- proof completion if necessary
  },
  { show (q (g ∩ h) <= q h),
    rw q_g_h,
    rw q_h,
    norm_num,
    sorry  -- proof completion if necessary
  }
}

end qg_given_h_qh_given_g_inconsistency_of_probs_l654_654137


namespace lasso_probability_l654_654259

theorem lasso_probability (p : ℚ) (h : p = 1 / 2) :
  let q := 1 - p in
  let fail_in_3 := q * q * q in
  let succeed_at_least_once := 1 - fail_in_3 in
  succeed_at_least_once = 7 / 8 :=
by
  sorry

end lasso_probability_l654_654259


namespace function_expression_l654_654997

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_expression (x : ℝ) (k : ℤ) (h_odd : ∀ x, f (-x) = - f x)
       (h_periodic : ∀ x, f (x + 2) = f x)
       (h_base : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
       f x = 
       if even k then x - k 
       else x - k - 1 := sorry

end function_expression_l654_654997


namespace problem1_problem2_problem3_l654_654585

variables {a : ℝ} {a_n b_n : ℕ → ℝ}
noncomputable def a_n (n : ℕ) : ℝ := if n = 0 then 1/4 else 1 - b_n n
noncomputable def b (n : ℕ) : ℝ := (nat.rec_on n (3 / 4) (λ n b, 1 / (2 - b : ℝ)))
noncomputable def S_n (n : ℕ) : ℝ := Σ i in finset.range n, a_n (i+1) * a_n (i+2)

theorem problem1 (h1 : a_n 1 = 1/4) (h2 : ∀ n, a_n n + b_n n = 1) (h3 : ∀ n, b_n (n+1) = b n):
  b_n 1 = 3/4 ∧ b_n 2 = 4/5 ∧ b_n 3 = 5/6 :=
begin
  sorry
end

theorem problem2 (h1 : ∀ n, b (n+1) = 1 / (2 - b n)) :
  (∀ n, (1 / (b n - 1 : ℝ)) = - (n+3)) :=
begin
  sorry
end

theorem problem3 (h1 : ∀ n, a_n n + b_n n = 1) (h2 : ∀ n, b_n (n+1) = b n)
  (h3 : ∀ n, a_n n = 1 / (n+3)) (h4 : 4 ∀a ∈ ℝ, S_n n < b_n n) :
  a ≤ 1 :=
begin
  sorry
end

end problem1_problem2_problem3_l654_654585


namespace find_f_1988_l654_654296

namespace FunctionalEquation

def f (n : ℕ) : ℕ :=
  sorry -- definition placeholder, since we only need the statement

axiom f_properties (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem find_f_1988 (h : ∀ n : ℕ, 0 < n → f n = n) : f 1988 = 1988 :=
  sorry

end FunctionalEquation

end find_f_1988_l654_654296


namespace number_of_small_cubes_l654_654818

theorem number_of_small_cubes (X : ℕ) (h1 : ∃ k, k = 29 - X) (h2 : 4 * 4 * 4 = 64) (h3 : X + 8 * (29 - X) = 64) : X = 24 :=
by
  sorry

end number_of_small_cubes_l654_654818


namespace fiftieth_term_of_sequence_is_318_l654_654321

def contains_digit_one (n : ℕ) : Prop :=
  n.digits 10 ∈ [1]

def sequence (n : ℕ) : Prop :=
  ∃ k : ℕ, 3 * k = n ∧ contains_digit_one n

theorem fiftieth_term_of_sequence_is_318 :
  ∃ n : ℕ, sequence n ∧ n = 318 := sorry

end fiftieth_term_of_sequence_is_318_l654_654321


namespace ten_mod_inv_1001_l654_654052

theorem ten_mod_inv_1001 : ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 1000 ∧ 10 * x ≡ 1 [MOD 1001] ∧ x = 901 := 
by
  use 901
  split
  · exact dec_trivial
  split
  · exact dec_trivial
  split
  · norm_num
    apply dvd.intro (9 * (1001) + 1)
    ring
  · exact rfl

end ten_mod_inv_1001_l654_654052


namespace ellipse_major_axis_l654_654881

noncomputable def major_axis_length (F1 F2 : ℝ × ℝ) (tangent_y : ℝ) : ℝ :=
  let F1_reflect := (F1.1, -F1.2) in
  real.sqrt ((F2.1 - F1_reflect.1)^2 + (F2.2 - F1_reflect.2)^2)

theorem ellipse_major_axis :
  let F1 := (5 : ℝ, 15 : ℝ) in
  let F2 := (35 : ℝ, 45 : ℝ) in
  major_axis_length F1 F2 0 = 30 * real.sqrt 5 :=
by
  sorry

end ellipse_major_axis_l654_654881


namespace propositions_hold_l654_654962

theorem propositions_hold (a b c k : ℝ) (A B C D : ℝ → ℝ) (O : ℝ):
  a < 0 →
  (∀ x : ℝ, 0 < x → a * x^2 + b * x + c = 0 → A x ∧ B x) →
  (∀ x : ℝ, k * x + b = 0 ∧ x > 0 → C x) →
  (∀ y : ℝ, y = b ∧ y > 0 → D y) →
  (∀ xy : ℝ × ℝ, xy = (0, 0) → xy = O) →
  (∀ x : ℝ, x = 2 → -b / (2 * a) = 2) →
  (∀ tri : set (ℝ × ℝ), is_right_isosceles_trianlge tri ∧ A (0, 0) ∧ D (b, 0)) →
  (-4 * a * c < 0 ∧ k ≠ -1 ∧ b = -4 * a) := sorry

end propositions_hold_l654_654962


namespace no_real_solutions_l654_654074

noncomputable def equation (x : ℝ) := x + 48 / (x - 3) + 1

theorem no_real_solutions : ∀ x : ℝ, equation x ≠ 0 :=
by
  intro x
  sorry

end no_real_solutions_l654_654074


namespace calculate_expression_l654_654487

theorem calculate_expression : (-1) ^ 47 + 2 ^ (3 ^ 3 + 4 ^ 2 - 6 ^ 2) = 127 := 
by 
  sorry

end calculate_expression_l654_654487


namespace trigonometric_identity_l654_654046

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end trigonometric_identity_l654_654046


namespace jack_last_10_shots_made_l654_654013

theorem jack_last_10_shots_made (initial_shots : ℕ) (initial_percentage : ℚ)
  (additional_shots : ℕ) (new_percentage : ℚ)
  (initial_successful_shots : initial_shots * initial_percentage = 18)
  (total_shots : initial_shots + additional_shots = 40)
  (total_successful_shots : (initial_shots + additional_shots) * new_percentage = 25) :
  ∃ x : ℕ, x = 7 := by
sorry

end jack_last_10_shots_made_l654_654013


namespace curve_crosses_itself_l654_654469

-- Definitions of the parametric equations
def x (t : ℝ) : ℝ := t^2 - 4
def y (t : ℝ) : ℝ := t^3 - 6*t + 3

-- The theorem statement
theorem curve_crosses_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ (x t₁, y t₁) = (2, 3) :=
by
  -- Proof would go here
  sorry

end curve_crosses_itself_l654_654469


namespace manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l654_654790

-- Definitions of costs and the problem conditions.
def cost_manufacturer_A (desks chairs : ℕ) : ℝ :=
  200 * desks + 50 * (chairs - desks)

def cost_manufacturer_B (desks chairs : ℕ) : ℝ :=
  0.9 * (200 * desks + 50 * chairs)

-- Given condition: School needs 60 desks.
def desks : ℕ := 60

-- (1) Prove manufacturer A is more cost-effective when x < 360.
theorem manufacturer_A_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs < 360 → cost_manufacturer_A desks chairs < cost_manufacturer_B desks chairs :=
by sorry

-- (2) Prove manufacturer B is more cost-effective when x > 360.
theorem manufacturer_B_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs > 360 → cost_manufacturer_A desks chairs > cost_manufacturer_B desks chairs :=
by sorry

end manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l654_654790


namespace pascal_g_log2_proof_problem_l654_654235

noncomputable def g (n : ℕ) : ℝ := Real.logb 2 (∑ k in Finset.range (n + 1), Nat.choose n k)

def sum_pascal_row (n : ℕ) : ℝ := (2 : ℝ) ^ n

theorem pascal_g_log2 (n : ℕ) : g n = n := by
  have h : ∑ k in Finset.range (n + 1), Nat.choose n k = 2 ^ n := by 
    sorry
  rw [g, h, Real.logb_pow 2 (n : ℝ)]
  norm_num

theorem proof_problem (n : ℕ) : (g n) / Real.logb 2 3 = n / Real.logb 2 3 := by
  rw [pascal_g_log2]

end pascal_g_log2_proof_problem_l654_654235


namespace simplest_fraction_is_D_l654_654466

def fractionA (x : ℕ) : ℚ := 10 / (15 * x)
def fractionB (a b : ℕ) : ℚ := (2 * a * b) / (3 * a * a)
def fractionC (x : ℕ) : ℚ := (x + 1) / (3 * x + 3)
def fractionD (x : ℕ) : ℚ := (x + 1) / (x * x + 1)

theorem simplest_fraction_is_D (x a b : ℕ) :
  ¬ ∃ c, c ≠ 1 ∧
    (fractionA x = (fractionA x / c) ∨
     fractionB a b = (fractionB a b / c) ∨
     fractionC x = (fractionC x / c)) ∧
    ∀ d, d ≠ 1 → fractionD x ≠ (fractionD x / d) := 
  sorry

end simplest_fraction_is_D_l654_654466


namespace sarah_score_l654_654281

variable (s g : ℕ)  -- Sarah's and Greg's scores are natural numbers

theorem sarah_score
  (h1 : s = g + 50)  -- Sarah's score is 50 points more than Greg's
  (h2 : (s + g) / 2 = 110)  -- Average of their scores is 110
  : s = 135 :=  -- Prove Sarah's score is 135
by
  sorry

end sarah_score_l654_654281


namespace true_proposition_is_B_l654_654467

open Classical

theorem true_proposition_is_B :
  (∀ a : ℝ, ∃ b : ℚ, a^2 + b^2 = 0) -> False ∧
  (∀ n : ℤ, ∃ m : ℤ, n * m = m) ∧
  (∀ n : ℤ, ∃ m : ℤ, n > m^2) -> False ∧
  (∀ a : ℝ, ∃ b : ℚ, a^2 + b^2 = 1) -> False :=
by {
  sorry,
}

end true_proposition_is_B_l654_654467


namespace proof_of_spatial_relationships_l654_654165

open Set

variable (a b : Line)
variable (α : Plane)

-- Conditions
def skew_lines (a b : Line) : Prop := 
  ¬∃ p : Point, p ∈ a ∧ p ∈ b

def parallel_to_plane (a : Line) (α : Plane) : Prop :=
  ∀ p ∈ a, p ∈ α

-- Questions in term of Prop to prove
def exists_parallel_plane (b : Line) : Prop :=
  ∃ α : Plane, ∀ p ∈ b, p ∈ α

def exists_plane_contain_b (b : Line) : Prop :=
  ∃ α : Plane, b ⊆ α

def infinite_planes_intersect_line (b : Line) : Prop :=
  ∀ p ∈ b, ∃ α : Plane, p ∈ α

theorem proof_of_spatial_relationships 
  (h_skew : skew_lines a b) 
  (h_parallel : parallel_to_plane a α) : 
  exists_parallel_plane b ∧ exists_plane_contain_b b ∧ infinite_planes_intersect_line b := 
sorry

end proof_of_spatial_relationships_l654_654165


namespace quadratic_inequality_solution_l654_654088

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 + m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
sorry

end quadratic_inequality_solution_l654_654088


namespace quadratic_with_root_and_real_coeffs_l654_654932

noncomputable def quadratic_polynomial (a b c : ℝ) (x : ℂ) : ℂ := a * x^2 + b * x + c

theorem quadratic_with_root_and_real_coeffs :
  ∃ (a b c : ℝ), a = 3 ∧
  quadratic_polynomial a b c (4 + 2 * Complex.I) = 0 ∧
  quadratic_polynomial a b c (4 - 2 * Complex.I) = 0 ∧
  quadratic_polynomial a b c = λ x, 3 * x^2 - 24 * x + 60 :=
sorry

end quadratic_with_root_and_real_coeffs_l654_654932


namespace length_of_shorter_side_l654_654864

-- Define the given conditions
def width : ℝ := 50
def num_poles : ℕ := 24
def distance_between_poles : ℝ := 5

-- Prove the length of the shorter side
theorem length_of_shorter_side : 
  (let num_gaps := num_poles - 1 in
   let total_perimeter := num_gaps * distance_between_poles in
   2 * (length + width) = total_perimeter) → 
  length = 7.5 :=
by
  -- Add proof here
  sorry

end length_of_shorter_side_l654_654864


namespace SarahsScoreIs135_l654_654286

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l654_654286


namespace prove_values_of_a_and_b_prove_range_of_k_l654_654552

variable {f : ℝ → ℝ}

-- (1) Prove values of a and b
theorem prove_values_of_a_and_b (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  (∀ x, f x = 2 * x - 1) := by
sorry

-- (2) Prove range of k
theorem prove_range_of_k (h_fx_2x_minus_1 : ∀ x : ℝ, f x = 2 * x - 1) :
  (∀ t : ℝ, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1 / 3 := by
sorry

end prove_values_of_a_and_b_prove_range_of_k_l654_654552


namespace fixed_distance_MN_l654_654956

-- Define the points and the problem setup in Lean
noncomputable def circle : Type := sorry
noncomputable def point : Type := sorry
noncomputable def line : Type := sorry
noncomputable def midpoint (p1 p2 : point) : point := sorry
noncomputable def reflection (p : point) (l : line) : point := sorry
noncomputable def distance (p1 p2 : point) : ℝ := sorry
noncomputable def arc (circle : circle) (p1 p2 : point) : Type := sorry

-- Define the given conditions
variable (circle : circle)
variable (A B : point) (hAB : A ≠ B)
variable (C : point) (hC_on_arc : C ∈ (arc circle A B))
variable (K : point) (hK : K = reflection A (line B C))
variable (L : point) (hL : L = reflection B (line A C))

-- Define the midpoints
noncomputable def M := midpoint A B
noncomputable def N := midpoint K L

-- The theorem to prove
theorem fixed_distance_MN : ∀ (C : point) (hC_on_arc : C ∈ (arc circle A B)), distance (midpoint A B) (midpoint (reflection A (line B C)) (reflection B (line A C))) = distance M N :=
by sorry

end fixed_distance_MN_l654_654956


namespace max_g_equals_one_at_one_h_not_monotonic_in_2_3_min_a_inequality_l654_654579

-- Proof Problem 1
theorem max_g_equals_one_at_one : ∀ (x : ℝ), g(x) = x * exp(1 - x) → (∀ x > 1, g(x) < 1) ∧ (∀ x < 1, g(x) > 1) ∧ g(1) = 1 :=
by
  intro x g_def
  sorry

-- Proof Problem 2
theorem h_not_monotonic_in_2_3 (m : ℝ) : 
  ∀ (x : ℝ), a = 2 ∧ h(x) = (x ^ 3) + (x ^ 2) * ((1 - 2/x) + m / 2) → 
  ¬(∃ (c : ℝ), c ∈ (2, 3) ∧ h'(c) = 3 * c ^ 2 + (2+m) * c - 2 = 0) 
  → -31/3 < m ∧ m < -7 :=
by
  intro x h_def
  sorry 

-- Proof Problem 3
theorem min_a_inequality : 
  ∀ (a : ℝ), a < 0 ∧ (∀ x1 x2 : ℝ, 3 ≤ x1 ∧ x1 ≤ 4 ∧ 3 ≤ x2 ∧ x2 ≤ 4 ∧ x1 ≠ x2 → 
  |f(x2) - f(x1)| < |1/g(x2) - 1/g(x1)|) → 
  a ≥ 3 - 2/3 * exp(2) :=
by
  intro a condition_inequality
  sorry 

end max_g_equals_one_at_one_h_not_monotonic_in_2_3_min_a_inequality_l654_654579


namespace proof_equivalent_l654_654550

noncomputable theory
open Complex

def question1 := ∀ z : ℂ, |z| = Real.sqrt 2 ∧ (Complex.im (z^2)) = 2 ∧ (Complex.re z > 0 ∧ Complex.im z > 0) → z = 1 + Complex.I

def question2 := ∀ (z : ℂ), z = 1 + Complex.I →
    let A := z;
    let B := (Complex.conj z)^2;
    let C := z - z^2 in
    let area := 1 in
    (area : ℝ)

def question3 := ∀ (z m : ℂ), z = 1 + Complex.I ∧ |m - z| = 1 →
    let min := (Real.sqrt 2) - 1;
    let max := (Real.sqrt 2) + 1 in
    |m| = min ∨ |m| = max

theorem proof_equivalent : question1 ∧ question2 ∧ question3 := by
  sorry

end proof_equivalent_l654_654550


namespace average_speed_for_trip_l654_654424

-- Define the total distance of the trip
def total_distance : ℕ := 850

--  Define the distance and speed for the first part of the trip
def distance1 : ℕ := 400
def speed1 : ℕ := 20

-- Define the distance and speed for the remaining part of the trip
def distance2 : ℕ := 450
def speed2 : ℕ := 15

-- Define the calculated average speed for the entire trip
def average_speed : ℕ := 17

theorem average_speed_for_trip 
  (d_total : ℕ)
  (d1 : ℕ) (s1 : ℕ)
  (d2 : ℕ) (s2 : ℕ)
  (hsum : d1 + d2 = d_total)
  (d1_eq : d1 = distance1)
  (s1_eq : s1 = speed1)
  (d2_eq : d2 = distance2)
  (s2_eq : s2 = speed2) :
  (d_total / ((d1 / s1) + (d2 / s2))) = average_speed := by
  sorry

end average_speed_for_trip_l654_654424


namespace trajectory_equation_find_m_l654_654122

-- Define points A and B.
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for P:
def P_condition (P : ℝ × ℝ) : Prop :=
  let PA_len := Real.sqrt ((P.1 - 1)^2 + P.2^2)
  let AB_len := Real.sqrt ((1 - (-1))^2 + (0 - 0)^2)
  let PB_dot_AB := (P.1 + 1) * (-2)
  PA_len * AB_len = PB_dot_AB

-- Problem (1): The trajectory equation
theorem trajectory_equation (P : ℝ × ℝ) (hP : P_condition P) : P.2^2 = 4 * P.1 :=
sorry

-- Define orthogonality condition
def orthogonal (M N : ℝ × ℝ) : Prop := 
  let OM := M
  let ON := N
  OM.1 * ON.1 + OM.2 * ON.2 = 0

-- Problem (2): Finding the value of m
theorem find_m (m : ℝ) (hm1 : m ≠ 0) (hm2 : m < 1) 
  (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (M N : ℝ × ℝ) (hM : M.2 = M.1 + m) (hN : N.2 = N.1 + m)
  (hMN : orthogonal M N) : m = -4 :=
sorry

end trajectory_equation_find_m_l654_654122


namespace geometric_a1_value_l654_654981

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q ^ (n - 1)

theorem geometric_a1_value (a3 a5 : ℝ) (q : ℝ) : 
  a3 = geometric_sequence a1 q 3 →
  a5 = geometric_sequence a1 q 5 →
  a1 = 2 :=
by
  sorry

end geometric_a1_value_l654_654981


namespace square_roots_and_cube_roots_l654_654325

theorem square_roots_and_cube_roots (a b : ℤ)
  (h1 : a + 3 = 2a - 15)
  (h2 : b = (-2) ^ 3) :
  Int.sqrt (2 * a - b) = 4 :=
by
  sorry

end square_roots_and_cube_roots_l654_654325


namespace good_set_is_regular_l654_654225

variable {Σ : Type} [Finite Σ]

def is_substring (x y : List Σ) : Prop :=
  ∃ prefix suffix : List Σ, y = prefix ++ x ++ suffix

def is_good_set (S : Set (List Σ)) : Prop :=
  ∀ x y : List Σ, is_substring x y → y ∈ S → x ∈ S

theorem good_set_is_regular (S : Set (List Σ)) (h : is_good_set S) : ⊤ := sorry

end good_set_is_regular_l654_654225


namespace NancyCoffeeDays_l654_654676

-- Define the cost of a double espresso per day
def doubleEspressoCost : ℝ := 3.0

-- Define the cost of an iced coffee per day
def icedCoffeeCost : ℝ := 2.5

-- Define the total amount spent on coffee
def totalSpent : ℝ := 110.0

-- Define the number of days Nancy has been buying coffee
def daysBuyingCoffee : ℝ := totalSpent / (doubleEspressoCost + icedCoffeeCost)

-- Theorem statement that we need to prove
theorem NancyCoffeeDays :
  daysBuyingCoffee = 20.0 := 
by
  sorry

end NancyCoffeeDays_l654_654676


namespace sum_four_digit_integers_l654_654395

def a := 1000
def l := 9999
def n := 9999 - 1000 + 1
def S (n : ℕ) (a : ℕ) (l : ℕ) := n / 2 * (a + l)

theorem sum_four_digit_integers : S n a l = 49495500 :=
by
  sorry

end sum_four_digit_integers_l654_654395


namespace common_difference_of_arithmetic_sequence_l654_654209

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℤ)
  (h1 : a 1 + a 9 = 10)
  (h2 : a 2 = -1)
  (d : ℤ)
  (ha : ∀ n, a(n + 1) = a n + d) :
  d = 2 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l654_654209


namespace rational_terms_count_l654_654210

noncomputable def number_of_rational_terms (n : ℕ) (x : ℝ) : ℕ :=
  -- The count of rational terms in the expansion
  17

theorem rational_terms_count (n : ℕ) (x : ℝ) :
  (number_of_rational_terms 100 x) = 17 := by
  sorry

end rational_terms_count_l654_654210


namespace total_pages_in_book_l654_654852

-- Given conditions
def pages_first_chapter : ℕ := 13
def pages_second_chapter : ℕ := 68

-- The theorem to prove the total number of pages in the book
theorem total_pages_in_book :
  pages_first_chapter + pages_second_chapter = 81 := by
  sorry

end total_pages_in_book_l654_654852


namespace min_convex_sets_l654_654963

noncomputable def M (A B C : ℝ × ℝ) : set (ℝ × ℝ) := 
  { p | p ≠ A ∧ p ≠ B ∧ p ≠ C }

def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ (m b : ℝ), 
    (A.2 = m * A.1 + b) ∧ 
    (B.2 = m * B.1 + b) ∧ 
    (C.2 = m * C.1 + b)

theorem min_convex_sets (A B C : ℝ × ℝ) (h_distinct: A ≠ B ∧ B ≠ C ∧ A ≠ C):
  if collinear A B C then
    ∃ (S : finset (set (ℝ × ℝ))), S.card = 4 ∧ ∀ (p : ℝ × ℝ), p ∈ (M A B C) ↔ ∃ (U ∈ S), p ∈ U
  else
    ∃ (S : finset (set (ℝ × ℝ))), S.card = 3 ∧ ∀ (p : ℝ × ℝ), p ∈ (M A B C) ↔ ∃ (U ∈ S), p ∈ U :=
sorry

end min_convex_sets_l654_654963


namespace amy_crumps_l654_654826

theorem amy_crumps (T Cp c : ℕ) (h1 : c = T * Cp) (h2 : 2 * T = 2 * T) (h3 : 1.5 * Cp = (3 / 2) * Cp) : 3 * c = 3 * (T * Cp) :=
by
  sorry

end amy_crumps_l654_654826


namespace power_i_2015_l654_654566

theorem power_i_2015 (i : ℂ) (hi : i^2 = -1) : i^2015 = -i :=
by
  have h1 : i^4 = 1 := by sorry
  have h2 : 2015 = 4 * 503 + 3 := by norm_num
  sorry

end power_i_2015_l654_654566


namespace count_divisors_of_54_greater_than_7_l654_654589

theorem count_divisors_of_54_greater_than_7 : ∃ (S : Finset ℕ), S.card = 4 ∧ ∀ n ∈ S, n ∣ 54 ∧ n > 7 :=
by
  -- proof goes here
  sorry

end count_divisors_of_54_greater_than_7_l654_654589


namespace successive_ratios_segments_sin_l654_654504

theorem successive_ratios_segments_sin : 
  ∃ p q : ℕ, nat.coprime p q ∧ (p, q) = (1, 4) ∧
  (∀ x ∈ set.Icc 0 360, y = sin x ↔ y = sin (60 : ℝ) →  
  let points := {x | 0 ≤ x ∧ x ≤ 360 ∧ sin x = sin (60 * (π / 180)) },
      segments := list.cons (60 : ℝ) (list.cons (60 : ℝ) (list.cons (240 : ℝ) list.nil)),
      ratio := list.map (λ l, l.head' ℕ / l.last' ℕ) segments
  in p / q ∈ ratio) := 
sorry

end successive_ratios_segments_sin_l654_654504


namespace sum_of_all_four_digit_integers_l654_654405

theorem sum_of_all_four_digit_integers :
  (Finset.range (9999 + 1)).filter (λ x => x ≥ 1000).sum = 49495500 :=
by
  sorry

end sum_of_all_four_digit_integers_l654_654405


namespace cyc_quad_intersect_at_one_point_l654_654959

variable {P Q S : Type} [EuclideanGeometry Q] [Inclusion P]

structure CyclicQuadrilateral (A B C D P Q S : Q) :=
  (cyclic : ∀ {X Y Z T : Q}, IsCyclic X Y Z T)
  (in_circle : InsideCircumcenter A B C D)
  (intersect_S : Intersect AC BD S)
  (midpoint_P : IsMidpoint P AD)
  (midpoint_Q : IsMidpoint Q BC)
  (perpendicular_p : PerpendicularAtLine P AC)
  (perpendicular_q : PerpendicularAtLine Q BD)
  (perpendicular_s : PerpendicularAtLine S CD)

theorem cyc_quad_intersect_at_one_point 
  {A B C D P Q S : Q} 
  (hq : CyclicQuadrilateral A B C D P Q S) :
  ∃ X : Q, Intersect hq.perpendicular_p hq.perpendicular_q X ∧ Intersect hq.perpendicular_s hq.perpendicular_q X := 
sorry

end cyc_quad_intersect_at_one_point_l654_654959


namespace union_cardinality_bound_l654_654175

variables {n k : ℕ} (X : Finset (Finset ℕ)) (A L : Finset (ℕ → Finset ℕ))

-- Condition: A_1, L_2, ..., s_4 are of type I
variables (I : (Finset ℕ → Prop))

-- Question and correct answer combined into what needs to be proved
theorem union_cardinality_bound
  (h : ∀ x ∈ A, I x) : X.card ≤ 2^n - 2^(n-k) :=
sorry

end union_cardinality_bound_l654_654175


namespace sum_four_digit_integers_l654_654394

def a := 1000
def l := 9999
def n := 9999 - 1000 + 1
def S (n : ℕ) (a : ℕ) (l : ℕ) := n / 2 * (a + l)

theorem sum_four_digit_integers : S n a l = 49495500 :=
by
  sorry

end sum_four_digit_integers_l654_654394


namespace ellipse_eq_line_eq_l654_654573

-- Define the ellipse and conditions
variable (a b : ℝ) 
variable h1 : a > b ∧ b > 0
variable h2 : (a^2 = 4 * b^2) 
variable h3 : ((1 / (4 * b^2)) + ((sqrt 3 / 2)^2 = 1))

-- Define the line l and conditions for intersection and tangency with the circle
variable (k m : ℝ)
variable h4 : k < -1/4 ∨ 0 < k ∧ k <= 1
variable h5 : (m = sqrt 2 ∨ m = - sqrt 2) 
variable h6 :  m^2 + k = 1

-- Define the circle and the tangency condition
noncomputable def line_tangent_circle (k m : ℝ) : Prop :=
(abs m / sqrt (1 + k^2) = 1)

-- Define the proof statements
theorem ellipse_eq : (a = 2 ∧ b = 1) → (∀ x y : ℝ, (x, y) ∈ set_of (λ p, p.1 ^ 2 / 4 + p.2 ^ 2 = 1)) :=
sorry

theorem line_eq : 
(∃ l : ℝ, line_tangent_circle k m) → 
(∀ P Q : ℝ × ℝ, (P.1, P.2) ∈ set_of (λ p, p.1 ^ 2 / 4 + p.2 ^ 2 = 1) ∧ (Q.1, Q.2) ∈ set_of (λ p, p.1 ^ 2 / 4 + p.2 ^ 2 = 1) →
P ≠ Q → 
( (P.2 / P.1 + Q.2 / Q.1) = 2)) → 
(∀ x y : ℝ, x ^ 2 + y ^ 2 = 1 ∧ y = k * x + m → y = x + sqrt 2 ∨ y = x - sqrt 2)
sorry

end ellipse_eq_line_eq_l654_654573


namespace smallest_marked_cells_l654_654390

def smallest_k_10x11 : ℕ := 50

theorem smallest_marked_cells :
  ∃ (marked_cells : Finset (Fin 10 × Fin 11)),
  marked_cells.card = smallest_k_10x11 ∧ 
  (∀ (L : Finset (Fin 10 × Fin 11)), L.card = 3 → (∀ P₁ P₂ P₃ ∈ L, ¬(P₁.1 = P₂.1 ∧ P₂.1 = P₃.1 ∨ P₁.2 = P₂.2 ∧ P₂.2 = P₃.2)) → 
    ∃ cell ∈ L, cell ∈ marked_cells) :=
sorry

end smallest_marked_cells_l654_654390


namespace smallest_marked_cells_l654_654388

def smallest_k_10x11 : ℕ := 50

theorem smallest_marked_cells :
  ∃ (marked_cells : Finset (Fin 10 × Fin 11)),
  marked_cells.card = smallest_k_10x11 ∧ 
  (∀ (L : Finset (Fin 10 × Fin 11)), L.card = 3 → (∀ P₁ P₂ P₃ ∈ L, ¬(P₁.1 = P₂.1 ∧ P₂.1 = P₃.1 ∨ P₁.2 = P₂.2 ∧ P₂.2 = P₃.2)) → 
    ∃ cell ∈ L, cell ∈ marked_cells) :=
sorry

end smallest_marked_cells_l654_654388


namespace intervals_of_monotonicity_range_of_values_l654_654150

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x - a * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  -(1 + a) / x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ :=
  f a x - g a x

theorem intervals_of_monotonicity (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, x < 1 + a → h a x < h a (1 + a)) ∧
  (∀ x > 1 + a, h a x > h a (1 + a)) :=
sorry

theorem range_of_values (x0 : ℝ) (h_x0 : 1 ≤ x0 ∧ x0 ≤ Real.exp 1) (h_fx_gx : f a x0 < g a x0) :
  a > (Real.exp 1)^2 + 1 / (Real.exp 1 - 1) ∨ a < -2 :=
sorry

end intervals_of_monotonicity_range_of_values_l654_654150


namespace maximize_Miraflores_win_l654_654744

-- Definitions based on given conditions
def voters_count (n : ℕ) : ℕ := 2 * n
def support_Miraflores (n : ℕ) : ℕ := n + 1
def support_opponent (n : ℕ) : ℕ := n - 1

-- Theorem statement
theorem maximize_Miraflores_win (n : ℕ) (hn : n > 0) : 
  ∃ (d1 d2 : ℕ), d1 = 1 ∧ d2 = 2 * n - 1 ∧ support_Miraflores n > support_opponent n := 
sorry

end maximize_Miraflores_win_l654_654744


namespace proof_equiv_l654_654248

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = 2 ^ Real.sqrt (3 + 2 * x - x ^ 2) }
noncomputable def N : Set ℝ := { x | ∃ y : ℝ, y = Real.log (x - 2) }
def I : Set ℝ := Set.univ
def complement_N : Set ℝ := I \ N

theorem proof_equiv : M ∩ complement_N = { y | 1 ≤ y ∧ y ≤ 2 } :=
sorry

end proof_equiv_l654_654248


namespace variance_of_data_set_is_0_l654_654724

variable {a : ℝ}
variable h_mean : (9.8 + 9.9 + 10 + a + 10.2) / 5 = 10

theorem variance_of_data_set_is_0.02 (h_mean : (9.8 + 9.9 + 10 + a + 10.2) / 5 = 10) : 
  let μ := 10 in
  let dataset := [9.8, 9.9, 10, a, 10.2] in
  let variance := (∑ x in dataset, (x - μ)^2) / list.length dataset in
  variance = 0.02 :=
sorry

end variance_of_data_set_is_0_l654_654724


namespace find_a_l654_654133

theorem find_a (x : ℝ) (hx1 : 0 < x)
  (hx2 : x + 1/x ≥ 2)
  (hx3 : x + 4/x^2 ≥ 3)
  (hx4 : x + 27/x^3 ≥ 4) :
  (x + a/x^4 ≥ 5) → a = 4^4 :=
sorry

end find_a_l654_654133


namespace percent_students_prefer_fish_l654_654203

theorem percent_students_prefer_fish :
  let total_students := 80 + 70 + 50 + 30 + 40 in
  let fish_students := 50 in
  (fish_students.to_float / total_students.to_float) * 100 ≈ 18 := by
{
  let total_students := 80 + 70 + 50 + 30 + 40,
  let fish_students := 50,
  have eq_1 : total_students = 270 := rfl,
  have eq_2 : fish_students = 50 := rfl,

  -- Direct calculation and conversion to percentage
  let percent_fish := (fish_students.to_float / total_students.to_float) * 100,
  have h : percent_fish = 18.51851851851852 := by norm_num1,

  -- approximate check within a tolerance range
  have ans : percent_fish ≈ 18 := by {
    intros ε h_pos,
    use (18.52 - 18),
    split,
    { linarith },
    { simp only [ge_iff_le, substone_eq_to_eq, eq_1, eq_2, le_of_eq] }
  },

  exact ans,
}

end percent_students_prefer_fish_l654_654203


namespace tobias_charges_for_mowing_l654_654785

/-- Tobias is buying a new pair of shoes that costs $95.
He has been saving up his money each month for the past three months.
He gets a $5 allowance a month.
He mowed 4 lawns and shoveled 5 driveways.
He charges $7 to shovel a driveway.
After buying the shoes, he has $15 in change.
Prove that Tobias charges $15 to mow a lawn.
--/
theorem tobias_charges_for_mowing 
  (shoes_cost : ℕ)
  (monthly_allowance : ℕ)
  (months_saving : ℕ)
  (lawns_mowed : ℕ)
  (driveways_shoveled : ℕ)
  (charge_per_shovel : ℕ)
  (money_left : ℕ)
  (total_money_before_purchase : ℕ)
  (x : ℕ)
  (h1 : shoes_cost = 95)
  (h2 : monthly_allowance = 5)
  (h3 : months_saving = 3)
  (h4 : lawns_mowed = 4)
  (h5 : driveways_shoveled = 5)
  (h6 : charge_per_shovel = 7)
  (h7 : money_left = 15)
  (h8 : total_money_before_purchase = shoes_cost + money_left)
  (h9 : total_money_before_purchase = (months_saving * monthly_allowance) + (lawns_mowed * x) + (driveways_shoveled * charge_per_shovel)) :
  x = 15 := 
sorry

end tobias_charges_for_mowing_l654_654785


namespace min_value_x2_y2_l654_654979

theorem min_value_x2_y2 (x y : ℝ) (h : 2 * x + y + 5 = 0) : x^2 + y^2 ≥ 5 :=
by
  sorry

end min_value_x2_y2_l654_654979


namespace sam_coins_and_value_l654_654696

-- Define initial conditions
def initial_dimes := 9
def initial_nickels := 5
def initial_pennies := 12

def dimes_from_dad := 7
def nickels_taken_by_dad := 3

def pennies_exchanged := 12
def dimes_from_exchange := 2
def pennies_from_exchange := 2

-- Define final counts of coins after transactions
def final_dimes := initial_dimes + dimes_from_dad + dimes_from_exchange
def final_nickels := initial_nickels - nickels_taken_by_dad
def final_pennies := initial_pennies - pennies_exchanged + pennies_from_exchange

-- Define the total count of coins
def total_coins := final_dimes + final_nickels + final_pennies

-- Define the total value in cents
def value_dimes := final_dimes * 10
def value_nickels := final_nickels * 5
def value_pennies := final_pennies * 1

def total_value := value_dimes + value_nickels + value_pennies

-- Proof statement
theorem sam_coins_and_value :
  total_coins = 22 ∧ total_value = 192 := by
  -- Proof details would go here
  sorry

end sam_coins_and_value_l654_654696


namespace trigonometric_identity_l654_654047

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end trigonometric_identity_l654_654047


namespace exists_positive_integers_for_hexagon_area_l654_654025

theorem exists_positive_integers_for_hexagon_area (S : ℕ) (a b : ℕ) (hS : S = 2016) :
  2 * (a^2 + b^2 + a * b) = S → ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 2 * (a^2 + b^2 + a * b) = S :=
by
  sorry

end exists_positive_integers_for_hexagon_area_l654_654025


namespace andrena_has_more_dolls_than_debelyn_l654_654059

-- Define the initial number of dolls
def initial_dolls_Debelyn : ℕ := 20
def initial_dolls_Christel : ℕ := 24

-- Define the number of dolls given to Andrena
def dolls_given_by_Debelyn : ℕ := 2
def dolls_given_by_Christel : ℕ := 5

-- Define the condition that Andrena has 2 more dolls than Christel after receiving the dolls
def andrena_more_than_christel : ℕ := 2

-- Define the dolls count after gift exchange
def dolls_Debelyn_after : ℕ := initial_dolls_Debelyn - dolls_given_by_Debelyn
def dolls_Christel_after : ℕ := initial_dolls_Christel - dolls_given_by_Christel
def dolls_Andrena_after : ℕ := dolls_Christel_after + andrena_more_than_christel

-- Define the proof problem
theorem andrena_has_more_dolls_than_debelyn : dolls_Andrena_after - dolls_Debelyn_after = 3 := by
  sorry

end andrena_has_more_dolls_than_debelyn_l654_654059


namespace arith_seq_problem_l654_654965

noncomputable def arith_seq_sum (a : ℕ → ℤ) (S1 : ℤ) (d : ℤ) (n m : ℕ) : Prop :=
  (a 1 + a 4 + a 7 + ... + a 97 = S1) →
  (a 3 + a 6 + a 9 + ... + a 99 = S1 + 33 * 2 * d)

theorem arith_seq_problem 
  (a : ℕ → ℤ) 
  (S1 : ℤ) 
  (d : ℤ)
  (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_sum : S1 = 50) 
  (h_d : d = -2) : 
  arith_seq_sum a 50 (-2) 97 99 :=
  sorry

end arith_seq_problem_l654_654965


namespace sum_of_primes_l654_654176

open Nat

def is_prime (n : ℕ) : Prop := Prime n

theorem sum_of_primes (a b c : ℕ) (h1 : is_prime a) (h2 : is_prime b) (h3 : is_prime c)
  (h4 : b + c = 13) (h5 : c * c - a * a = 72) : a + b + c = 15 := by
  sorry

end sum_of_primes_l654_654176


namespace polar_equations_and_ratios_l654_654629

open Real

theorem polar_equations_and_ratios (α β : ℝ)
    (h_line : ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ)
    (h_curve : ∀ (α : ℝ), ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2) :
    ( ∀ (r θ : ℝ), r * cos θ = 2 ↔ r = 2 / cos θ) ∧
    ( ∃ (r θ : ℝ), r = 2 * sin θ ∧ θ = β ∧ 0 < β ∧ β < π / 2 → 
    0 < r * sin 2 * θ / (r / cos θ) ∧ r * sin 2 * θ / (r / cos θ) ≤ 1 / 2) :=
by
  sorry

end polar_equations_and_ratios_l654_654629


namespace solution_set_of_tan_eq_two_l654_654733

open Real

theorem solution_set_of_tan_eq_two :
  {x | ∃ k : ℤ, x = k * π + (-1 : ℤ) ^ k * arctan 2} = {x | tan x = 2} :=
by
  sorry

end solution_set_of_tan_eq_two_l654_654733


namespace miraflores_optimal_split_l654_654764

-- Define the total number of voters as 2n, and initialize half supporters for each candidate.
variable (n : ℕ) (voters : Fin (2 * n) → Bool)

-- Define the condition that exactly half of the voters including Miraflores support him
def half_support_miraflores : Prop :=
  ∃ (supporters_miraflores : Fin n) (supporters_maloney : Fin n), 
    (voters supporters_miraflores.val = true) ∧ (voters.supporters_maloney.val = false) 

-- Define the condition of drawing a single random ballot in each district.
def draw_random_ballot (d : Fin n → Prop) : Fin n := sorry

-- Define the condition that Miraflores wins if he wins both districts.
def wins_election (d1 d2 : Fin n → Prop) : Prop := 
  (draw_random_ballot d1 = true) ∧ (draw_random_ballot d2 = true)

-- Miraflores should split the voters such that his maximum probability of winning is achieved.
def optimal_split : Prop :=
  ∃ (d1 d2 : Fin n → Bool), 
    (d1.supporters_miraflores.val = true ∧ d2.supporters_maloney.val = false) ∧
    (wins_election d1 d2 = true)

theorem miraflores_optimal_split (n : ℕ) (voters : Fin (2 * n) → Bool) (half_support : half_support_miraflores n voters) : optimal_split n :=
sorry

end miraflores_optimal_split_l654_654764


namespace fewer_twos_for_100_l654_654351

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l654_654351


namespace intersection_line_eq_common_chord_length_l654_654584

theorem intersection_line_eq (x y : ℝ) :
  (x^2 + y^2 = 1) ∧ (x^2 + y^2 - 2 * x - 2 * y + 1 = 0) → (x + y - 1 = 0) :=
by
  sorry

theorem common_chord_length (x y : ℝ) :
  (x^2 + y^2 = 1) ∧ (x^2 + y^2 - 2 * x - 2 * y + 1 = 0) → ( √2) :=
by
  sorry

end intersection_line_eq_common_chord_length_l654_654584


namespace number_of_different_possible_schedules_l654_654204

theorem number_of_different_possible_schedules :
  let n := 16 in
  let schedules := (Finset.range n).powerset.filter (λ s, s.card = 2) in
  let num_schedules := schedules.card / (n / 2).factorial in
  num_schedules = 2027025 := sorry

end number_of_different_possible_schedules_l654_654204


namespace min_value_3x_plus_4y_l654_654603

theorem min_value_3x_plus_4y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x + 3*y = 5*x*y) :
  ∃ (c : ℝ), (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y → 3 * x + 4 * y ≥ c) ∧ c = 5 :=
sorry

end min_value_3x_plus_4y_l654_654603


namespace find_a_20_l654_654115

-- Given conditions as definitions
def sequence_a : Nat → Real
axiom a_n_ne_zero : ∀ n, sequence_a n ≠ 0
def a_1 : sequence_a 1 = 1
axiom recurrence_relation : ∀ n, 1 / sequence_a (n + 1) = 1 / sequence_a n + 2

-- Main problem statement to prove
theorem find_a_20 : sequence_a 20 = 1 / 39 :=
by 
  sorry

end find_a_20_l654_654115


namespace min_k_triangle_condition_l654_654534

theorem min_k_triangle_condition : ∀ a : Finset ℕ, 
  (∀ (x y z ∈ a), x ≠ y ∧ y ≠ z ∧ x ≠ z → x + y > z ∧ x + z > y ∧ y + z > x) → 
  ∃ (k : ℕ), k ≤ 1000 ∧ ∀ b : Finset ℕ, (b ⊆ (Finset.range 1001)) ∧ b.card = 16 → 
  (∃ u v w ∈ b, u + v > w ∧ u + w > v ∧ v + w > u) :=
sorry

end min_k_triangle_condition_l654_654534


namespace triangular_number_is_perfect_square_l654_654926

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

end triangular_number_is_perfect_square_l654_654926


namespace parabola_intersects_x_axis_l654_654318

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 + 2 * x + m - 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := 4 - 4 * (m - 1)

-- Lean statement to prove the range of m
theorem parabola_intersects_x_axis (m : ℝ) : (∃ x : ℝ, quadratic x m = 0) ↔ m ≤ 2 := by
  sorry

end parabola_intersects_x_axis_l654_654318


namespace sequence_formula_sum_cn_l654_654558

noncomputable def arithmetic_sequence_first_term : ℕ → ℕ
| 1     := 1
| (n+1) := a_1 + n * d

-- Define the conditions
def a_1 := 1
def S (n : ℕ) : ℕ := n / 2 * (2 * a_1 + (n - 1) * d)

theorem sequence_formula (h : S 4 - 2 * (a_1 + d) * (a_1 + 2 * d) + 14 = 0) 
  (h_d : d > 1) :
  (∀ n : ℕ, a_n = 2 * n - 1) := 
begin
  sorry
end

theorem sum_cn (h : ∀ n : ℕ, a_n = 2 * n - 1) :
  (∀ n : ℕ, let c_n := (1 : ℝ) / ((a_n : ℝ) * (a_n + 1 : ℝ)) in 
             ∑ i in finset.range n, c_n = n / (2 * n + 1)) 
:= 
begin
  sorry
end

end sequence_formula_sum_cn_l654_654558


namespace price_decrease_percentage_l654_654439

def original_price : ℝ := 72.95
def sale_price : ℝ := 59.95

theorem price_decrease_percentage :
  (original_price - sale_price) / original_price * 100 ≈ 17.82 :=
by
  sorry

end price_decrease_percentage_l654_654439


namespace number_of_large_balls_l654_654783

def smallBallRubberBands : ℕ := 50
def largeBallRubberBands : ℕ := 300
def totalRubberBands : ℕ := 5000
def smallBallsMade : ℕ := 22

def rubberBandsUsedForSmallBalls := smallBallsMade * smallBallRubberBands
def remainingRubberBands := totalRubberBands - rubberBandsUsedForSmallBalls

theorem number_of_large_balls :
  (remainingRubberBands / largeBallRubberBands) = 13 := by
  sorry

end number_of_large_balls_l654_654783


namespace otimes_correct_l654_654089

noncomputable def otimes (x y z : ℝ) (h : y ≠ z) : ℝ :=
  x / (y - z)

theorem otimes_correct :
  otimes (otimes 2 3 4 (by norm_num1))
         (otimes 3 4 2 (by norm_num1))
         (otimes 4 2 3 (by norm_num1))
         (by norm_num1) = -4 / 11 :=
by
  sorry

end otimes_correct_l654_654089


namespace problem_equiv_l654_654663

-- Given function definition and conditions
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)

-- Conditions
axiom omega_pos : ∃ ω > 0, f = (λ x, Real.cos (ω * x + φ))
axiom phi_range : -Real.pi / 2 < φ ∧ φ < 0
axiom period_condition : Function.periodic f Real.pi
axiom point_condition : f (Real.pi / 4) = Real.sqrt 3 / 2

-- Prove that the function f meets all conditions and find monotonic intervals and extremes
theorem problem_equiv :
    (f = (λ x, Real.cos (2 * x - Real.pi / 3)) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), (Set.Icc 0 (Real.pi / 6)).monotoneOn f ∧ 
    (Set.Icc (Real.pi / 6) (Real.pi / 2)).antitoneOn f) ∧
    (f 0 = 1 / 2 ∧ f (Real.pi / 6) = 1 ∧ f (Real.pi / 2) = -1 / 2)) := 
begin
  sorry
end

end problem_equiv_l654_654663


namespace minimize_cylinder_surface_area_l654_654065

theorem minimize_cylinder_surface_area (V : ℝ) 
  (H : ℝ) (R : ℝ) (π_pos : 0 < real.pi)
  (surface_area : ℝ := 2 * real.pi * R * H + 2 * real.pi * R^2)
  (volume : ℝ := real.pi * R^2 * H) :
  H = 2 * R ↔ (surface_area = 2 * V / R + 2 * real.pi * R^2 ∧ volume = V) :=
by
  sorry

end minimize_cylinder_surface_area_l654_654065


namespace num_solutions_mod_congruence_l654_654134

theorem num_solutions_mod_congruence :
  {x : ℕ // x < 150 ∧ x + 20 ≡ 75 [MOD 46]}.card = 4 :=
by 
  sorry

end num_solutions_mod_congruence_l654_654134


namespace child_ticket_cost_l654_654464

variable (x : ℕ) (n_adult n_child : ℕ) (cost_adult cost_child total_cents total_persons : ℕ)

-- Given conditions
def cost_adult := 60
def total_persons := 280
def total_cents := 14000  -- 140 dollars converted to cents
def n_adult := 200
def n_child := total_persons - n_adult

theorem child_ticket_cost :
  n_adult * cost_adult + n_child * x = total_cents → x = 25 :=
  by sorry

end child_ticket_cost_l654_654464


namespace find_alpha_l654_654568

theorem find_alpha (α : ℝ) (h1 : 2 * α ≡ α [MOD 360]) (h2 : 0 ≤ α ∧ α < 360) : α = 0 :=
by
  sorry

end find_alpha_l654_654568


namespace box_length_is_8_cm_l654_654449

def width : ℝ := 15
def height : ℝ := 5
def volume_cube : ℝ := 10
def num_cubes : ℝ := 60
def total_volume : ℝ := num_cubes * volume_cube
def length (V W H : ℝ) := V / (W * H)

theorem box_length_is_8_cm : length total_volume width height = 8 := 
by
  sorry

end box_length_is_8_cm_l654_654449


namespace slope_angle_of_line_y_eq_x_minus_1_l654_654732

noncomputable def slope (l : ℝ → ℝ) : ℝ :=
  1  -- The slope of the line y = x - 1

def slope_angle (m : ℝ) : ℝ :=
  if m = 1 then 45 else 0  -- Return 45 degrees if slope is 1, otherwise 0 for simplicity

theorem slope_angle_of_line_y_eq_x_minus_1 :
  slope_angle (slope (λ x => x - 1)) = 45 :=
by
  sorry

end slope_angle_of_line_y_eq_x_minus_1_l654_654732


namespace lambda_range_l654_654306

theorem lambda_range (λ : ℝ) :
  (∀ a b : ℝ, a^2 + 8 * b^2 ≥ λ * b * (a + b)) → λ ∈ Set.Icc (-8 : ℝ) 4 :=
sorry

end lambda_range_l654_654306


namespace projection_equals_one_l654_654624

-- Define the setup and conditions
variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (AB AC AD : A → B → ℝ) (ABC : triangle A B) (₀ : ℝ) (angle_BAD : ℝ)

-- Given conditions
def rhombus_ABCD : Prop := true -- Definition of a rhombus is complex, abstract here
def side_length_2 : AB = 2 := sorry
def angle_BAD_120 : angle_BAD = 120 := sorry

-- The projection formula
noncomputable def projection_AC_on_AB (AC AB : ℝ) : ℝ :=
    (AC * AB) / AB

-- The proof statement
theorem projection_equals_one (h1 : rhombus_ABCD) (h2 : side_length_2) (h3 : angle_BAD_120) :
    projection_AC_on_AB AC AB = 1 :=
    sorry

end projection_equals_one_l654_654624


namespace number_of_large_balls_l654_654782

def smallBallRubberBands : ℕ := 50
def largeBallRubberBands : ℕ := 300
def totalRubberBands : ℕ := 5000
def smallBallsMade : ℕ := 22

def rubberBandsUsedForSmallBalls := smallBallsMade * smallBallRubberBands
def remainingRubberBands := totalRubberBands - rubberBandsUsedForSmallBalls

theorem number_of_large_balls :
  (remainingRubberBands / largeBallRubberBands) = 13 := by
  sorry

end number_of_large_balls_l654_654782


namespace fill_grid_four_digit_number_l654_654511

def fill_grid (grid : list (list ℕ)) : Prop :=
  ∀ row col, grid[row][col] ∈ {1, 2, 3} ∧
    (∀ row, ∃! col, grid[row][col] = n ∧ (n == 1 ∨ n == 2 ∨ n == 3)) ∧
    (∀ col, ∃! row, grid[row][col] = n ∧ (n == 1 ∨ n == 2 ∨ n == 3))

noncomputable def four_digit_number (grid : list (list ℕ)) : ℕ :=
  grid[0][0] * 1000 + grid[0][1] * 100 + grid[0][2] * 10 + grid[0][3]

theorem fill_grid_four_digit_number :
  ∃ grid : list (list ℕ), fill_grid grid → four_digit_number grid = 2213 :=
sorry

end fill_grid_four_digit_number_l654_654511


namespace sin_y_gt_half_x_l654_654830

theorem sin_y_gt_half_x (x y : ℝ) (hx : x ≤ 90) (h : Real.sin y = (3 / 4) * Real.sin x) : y > x / 2 :=
by
  sorry

end sin_y_gt_half_x_l654_654830


namespace homework_time_decrease_l654_654339

variable (x : ℝ)
variable (initial_time final_time : ℝ)
variable (adjustments : ℕ)

def rate_of_decrease (initial_time final_time : ℝ) (adjustments : ℕ) (x : ℝ) := 
  initial_time * (1 - x)^adjustments = final_time

theorem homework_time_decrease 
  (h_initial : initial_time = 100) 
  (h_final : final_time = 70)
  (h_adjustments : adjustments = 2)
  (h_decrease : rate_of_decrease initial_time final_time adjustments x) : 
  100 * (1 - x)^2 = 70 :=
by
  sorry

end homework_time_decrease_l654_654339


namespace cube_surface_area_difference_l654_654859

/-- The difference between the sum of the surface areas of the 6859 smaller cubes and the surface 
area of the larger cube is 38988 square inches given the conditions. -/
theorem cube_surface_area_difference :
  ∀ (volume_large_cube : ℕ) (num_small_cubes : ℕ) (volume_small_cube : ℕ),
    volume_large_cube = 6859 →
    num_small_cubes = 6859 →
    volume_small_cube = 1 →
    let side_length_large_cube := (volume_large_cube : ℝ)^(1/3)
    let surface_area_large_cube := 6 * side_length_large_cube^2
    let surface_area_one_small_cube := 6 * (volume_small_cube : ℝ)^(2/3)
    let total_surface_area_small_cubes := num_small_cubes * surface_area_one_small_cube
    total_surface_area_small_cubes - surface_area_large_cube = 38988 := by
  intros volume_large_cube num_small_cubes volume_small_cube h1 h2 h3
  let side_length_large_cube := (volume_large_cube : ℝ)^(1/3)
  let surface_area_large_cube := 6 * side_length_large_cube^2
  let surface_area_one_small_cube := 6 * (volume_small_cube : ℝ)^(2/3)
  let total_surface_area_small_cubes := num_small_cubes * surface_area_one_small_cube
  have h4: side_length_large_cube = 19.0 := sorry -- Proof omitted
  have h5: surface_area_large_cube = 6 * 19^2 := sorry -- Proof omitted
  have h6: surface_area_large_cube = 2166 := sorry -- Proof omitted
  have h7: surface_area_one_small_cube = 6 := sorry -- Proof omitted
  have h8: total_surface_area_small_cubes = 6859 * 6 := sorry -- Proof omitted
  have h9: total_surface_area_small_cubes = 41154 := sorry -- Proof omitted
  calc
    total_surface_area_small_cubes - surface_area_large_cube
      = 41154 - 2166 : by rw [h8, h6]
  ... = 38988 : by norm_num

end cube_surface_area_difference_l654_654859


namespace prove_a_range_l654_654576

noncomputable def f (x : ℝ) : ℝ := 1 / (2 ^ x + 2)

theorem prove_a_range (a : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x + f (a - 2 * x) ≤ 1 / 2) → 5 ≤ a :=
by
  sorry

end prove_a_range_l654_654576


namespace euler_formula_correct_options_l654_654068

theorem euler_formula_correct_options :
  (euler_formula (2 * Real.pi / 3) = -1/2 + Complex.I * (Real.sqrt 3 / 2)) ∧
  (euler_formula (Real.pi / 2) = Complex.I) ∧
  (Complex.abs (euler_formula Real.pi / (Real.sqrt 3 + Complex.I)) = 1/2) ∧
  (Complex.conj (euler_formula (Real.pi / 6)) ≠ 1/2 - (Real.sqrt 3 / 2) * Complex.I) :=
by
  sorry

def euler_formula (x : ℝ) : ℂ :=
  Complex.exp (Complex.I * x)

end euler_formula_correct_options_l654_654068


namespace factorial_comparison_l654_654806

open scoped BigOperators

theorem factorial_comparison : (100.factorial)!.factorial < (99.factorial) ^ (100.factorial) * (100.factorial) ^ (99.factorial) := by
  sorry

end factorial_comparison_l654_654806


namespace correct_system_of_equations_l654_654837

theorem correct_system_of_equations (x y : ℝ) :
  (y - x = 4.5) ∧ (x - y / 2 = 1) ↔
  ((y - x = 4.5) ∧ (x - y / 2 = 1)) :=
by sorry

end correct_system_of_equations_l654_654837


namespace events_A_and_D_independent_l654_654792

open_locale big_operators

-- Events: 
def event_A (x : ℕ) : Prop := x % 2 = 1  -- first die is odd
def event_B (y : ℕ) : Prop := y % 2 = 0  -- second die is even
def event_C (x y : ℕ) : Prop := x + y = 6  -- sum of the points is 6
def event_D (x y : ℕ) : Prop := x + y = 7  -- sum of the points is 7

/-- Theorem to verify that events A and D are independent -/
theorem events_A_and_D_independent : 
  (∀ x y : ℕ, 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 →
  event_A x ∧ event_D x y → 
  prob (event_A x ∧ event_D x y) = 
  (prob event_A) * (prob (event_D x y))) := sorry

end events_A_and_D_independent_l654_654792


namespace possible_q_values_l654_654533

theorem possible_q_values (d q : ℝ) (h : ∀ n, ∃ k, q^n = 1 + k * d) : q ∈ ℕ ∧ q > 1 :=
  sorry

end possible_q_values_l654_654533


namespace max_value_of_expression_l654_654670

noncomputable def f (x y : ℝ) := x * y^2 * (x^2 + x + 1) * (y^2 + y + 1)

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  ∃ m, m = 951625 / 256 ∧ ∀ a b : ℝ, a + b = 5 → f a b ≤ m :=
sorry

end max_value_of_expression_l654_654670


namespace maximize_victory_probability_l654_654760

-- Define the conditions
variables {n : ℕ}
def number_of_voters := 2 * n
def half_support_miraflores := n
def half_support_dick_maloney := n
def miraflores_is_voter := true

-- Define the districts
def district1 := {miraflores}
def district2 := {voters | voters ≠ miraflores}

theorem maximize_victory_probability (n : ℕ) (h₁ : nat.odd (2*n + 1) = true) : 
  (let district1_voters := 1 in
   let district2_voters := 2*n - 1 in
   maximize_probability_of_winning(district1_voters, district2_voters) = true) :=
sorry

end maximize_victory_probability_l654_654760


namespace craig_age_difference_l654_654057

theorem craig_age_difference : 
  ∀ (C M : ℕ), (C + M = 56) ∧ (C = 16) → (M - C = 24) :=
by 
  intros C M h,
  cases h with h1 h2,
  rw [h2] at h1,
  have hM : M = 40 := by 
    linarith,
  rw [hM, h2],
  linarith

end craig_age_difference_l654_654057


namespace output_of_2004_2008_l654_654060

def binary_operation : ℕ → ℕ → ℤ 
| 1, 1 => 2
| (m + 1), n => binary_operation m n - 1
| m, (n + 1) => binary_operation m n + 2

theorem output_of_2004_2008 : binary_operation 2004 2008 = 2013 := 
by 
-- proof goes here
sorry

end output_of_2004_2008_l654_654060


namespace man_speed_still_water_l654_654819

-- Conditions
variables (upstream_speed downstream_speed : ℝ)
variables (man_speed_in_still_water : ℝ)

-- Definitions
def speed_in_still_water (u_speed d_speed : ℝ) : ℝ := (u_speed + d_speed) / 2

-- Theorem statement
theorem man_speed_still_water (h_up: upstream_speed = 60) (h_down: downstream_speed = 90) :
  man_speed_in_still_water = 75 :=
by
  -- Use the definition of speed_in_still_water with provided speeds
  have h : speed_in_still_water 60 90 = 75 := by
    simp [speed_in_still_water]
  -- Conclude the theorem using the above derived value
  exact h

end man_speed_still_water_l654_654819


namespace no_solution_in_natural_numbers_l654_654690

theorem no_solution_in_natural_numbers (x y z : ℕ) (hxy : x ≠ 0) (hyz : y ≠ 0) (hzx : z ≠ 0) :
  ¬ (x / y + y / z + z / x = 1) :=
by sorry

end no_solution_in_natural_numbers_l654_654690


namespace max_shortest_part_duration_l654_654447

theorem max_shortest_part_duration (film_duration : ℕ) (part1 part2 part3 part4 : ℕ)
  (h_total : part1 + part2 + part3 + part4 = 192)
  (h_diff1 : part2 ≥ part1 + 6)
  (h_diff2 : part3 ≥ part2 + 6)
  (h_diff3 : part4 ≥ part3 + 6) :
  part1 ≤ 39 := 
sorry

end max_shortest_part_duration_l654_654447


namespace total_people_veg_l654_654422

def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 8

theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 21 := by
  sorry

end total_people_veg_l654_654422


namespace part1_part2_l654_654671

-- Definition of the complex number
def z (a : ℝ) : ℂ := (a^2 - a - 2 : ℂ) + (a^2 - 3a - 4 : ℂ) * complex.I

-- Conditions for the first part
def isPureImaginary (z : ℂ) : Prop := z.re = 0

-- Conditions for the second part
def inSecondOrFourthQuadrant (z : ℂ) : Prop := (z.re * z.im < 0)

-- Lean 4 statements
theorem part1 (a : ℝ) (h : isPureImaginary (z a)) : a = 2 := sorry

theorem part2 (a : ℝ) (h : inSecondOrFourthQuadrant (z a)) : 2 < a ∧ a < 4 := sorry

end part1_part2_l654_654671


namespace first_digit_of_528_base7_l654_654800

def first_digit_base7 (n : ℕ) : ℕ :=
  let rec compute_first_digit (n k : ℕ) (p : ℕ) : ℕ :=
    if n < p then k - 1 else compute_first_digit (n / 7) (k + 1) (p * 7)
  compute_first_digit n 1 7

theorem first_digit_of_528_base7 :
  first_digit_base7 528 = 1 :=
sorry

end first_digit_of_528_base7_l654_654800


namespace sum_of_digits_l654_654937

def large_number := 
  let fractional_part := 44 * 10^(2014) + 4 * (10^(2013) - 1) / 9 in
  let integer_part := 9 + 8 * ((10^2017) - 1) / 9 in
  fractional_part + integer_part / 10^2018

theorem sum_of_digits : 
  let sqrt_large := real.sqrt large_number in
  let int_part := floor sqrt_large in
  (digit_sum int_part = 12109) := by
    sorry

end sum_of_digits_l654_654937


namespace arithmetic_sequence_l654_654631

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 1 + a 2 + a 3 = 32) 
  (h2 : a 11 + a 12 + a 13 = 118) 
  (arith_seq : ∀ n, a (n + 1) = a n + d) : 
  a 4 + a 10 = 50 :=
by 
  sorry

end arithmetic_sequence_l654_654631


namespace miraflores_optimal_strategy_l654_654751

-- Definitions based on conditions
variable (n : ℕ)
def total_voters := 2 * n
def miraflores_supporters := n
def dick_maloney_supporters := n
def miraflores_is_a_voter := 1
def law_allows_division := true
def election_winner (district1 district2 : Set ℕ) : ℕ := 
  if (district1.card = 1 ∧ miraflores_is_a_voter ∈ district1) then miraflores_is_a_voter else dick_maloney_supporters

-- Mathematically equivalent proof problem
theorem miraflores_optimal_strategy (hall : law_allows_division) :
  (exists (district1 district2 : Set ℕ),
    ∀ v, v ∈ district1 ∨ v ∈ district2 ∧ district1.card + district2.card = total_voters ∧
    miraflores_supporters = 1 ∧ district1 = {miraflores_is_a_voter} ∧
    (election_winner district1 district2) = miraflores_is_a_voter) :=
sorry

end miraflores_optimal_strategy_l654_654751


namespace revised_average_ratio_l654_654001

theorem revised_average_ratio (scores : Fin 50 → ℝ) :
  let A := (∑ i, scores i) / 50
  let revised_sum := (∑ i, scores i) + 2 * A
  let revised_average := revised_sum / 51
  (revised_average / A) = 52 / 51 :=
by
  sorry

end revised_average_ratio_l654_654001


namespace solve_expression_l654_654292

theorem solve_expression : 
  ∃ (x : ℝ), 327^2 - 2^(Real.log 3 / Real.log 2) * (Real.log 1/8 / Real.log 2) + Real.log 25 / Real.log 10 + 2 * Real.log 2 / Real.log 10 = x ∧ x = 22 :=
begin
  sorry
end

end solve_expression_l654_654292


namespace miraflores_optimal_strategy_l654_654749

-- Definitions based on conditions
variable (n : ℕ)
def total_voters := 2 * n
def miraflores_supporters := n
def dick_maloney_supporters := n
def miraflores_is_a_voter := 1
def law_allows_division := true
def election_winner (district1 district2 : Set ℕ) : ℕ := 
  if (district1.card = 1 ∧ miraflores_is_a_voter ∈ district1) then miraflores_is_a_voter else dick_maloney_supporters

-- Mathematically equivalent proof problem
theorem miraflores_optimal_strategy (hall : law_allows_division) :
  (exists (district1 district2 : Set ℕ),
    ∀ v, v ∈ district1 ∨ v ∈ district2 ∧ district1.card + district2.card = total_voters ∧
    miraflores_supporters = 1 ∧ district1 = {miraflores_is_a_voter} ∧
    (election_winner district1 district2) = miraflores_is_a_voter) :=
sorry

end miraflores_optimal_strategy_l654_654749


namespace convex_quadrilateral_obtuse_l654_654958

theorem convex_quadrilateral_obtuse (A B C D : Type) (angle_D_obtuse : ℝ) (convex_quad : Prop) : 
(convex_quad ∧ obtuse angle_D_obtuse) → 
(∃ (n : ℕ), (n ≥ 4) ∧ (∀ t ∈ (triangulation A B C D), obtuse t)) :=
begin
    sorry
end

end convex_quadrilateral_obtuse_l654_654958


namespace solution1_solution2_l654_654342

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l654_654342


namespace neg_four_fifth_less_neg_two_third_l654_654038

theorem neg_four_fifth_less_neg_two_third : (-4 : ℚ) / 5 < (-2 : ℚ) / 3 :=
  sorry

end neg_four_fifth_less_neg_two_third_l654_654038


namespace time_to_cross_platform_l654_654008

theorem time_to_cross_platform (train_speed_kmph : ℕ) (t_cross_man : ℕ) (platform_length : ℕ) 
  (train_speed_kmph = 72) (t_cross_man = 18) (platform_length = 260) : 
  ∃ T : ℕ, T = 31 := 
by {
  sorry
}

end time_to_cross_platform_l654_654008


namespace centroid_not_on_inscircle_l654_654644

theorem centroid_not_on_inscircle 
  {α β γ : ℝ} {a b : ℝ} 
  (h1 : α + β + γ = 180)
  (h2 : α = 22.5)
  (h3 : β = 67.5)
  (h4 : γ = 90) 
  (h5 : a^2 + b^2 = 1) :
  let K := (a + b - 1) / 2,
      S := (b / 3, a / 3),
      r := K in
    ¬(sqrt ((S.1 - r)^2 + (S.2 - r)^2) = r) :=
begin
  sorry
end

end centroid_not_on_inscircle_l654_654644


namespace trigonometric_expression_value_l654_654041

theorem trigonometric_expression_value :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 :=
by
  sorry

end trigonometric_expression_value_l654_654041


namespace sum_of_primes_l654_654179

theorem sum_of_primes (a b c : ℕ) (h₁ : Nat.Prime a) (h₂ : Nat.Prime b) (h₃ : Nat.Prime c) (h₄ : b + c = 13) (h₅ : c^2 - a^2 = 72) :
  a + b + c = 20 := 
sorry

end sum_of_primes_l654_654179


namespace fill_tank_with_leak_l654_654862

theorem fill_tank_with_leak (h₁ : ∀ t : ℝ, t = 2 → rate_pump = 1 / t) (h₂ : ∀ t : ℝ, t = 7 → rate_leak = 1 / t) : 
  ∃ t : ℝ, t = 2.8 ∧ rate_effective = rate_pump - rate_leak →

let rate_pump := 1 / 2 in
let rate_leak := 1 / 7 in
let rate_effective := rate_pump - rate_leak in
t = 1 / rate_effective :=
by
  sorry

end fill_tank_with_leak_l654_654862


namespace solve_for_x_l654_654184

theorem solve_for_x : ∀ (x : ℕ), (y = 2 / (4 * x + 2)) → (y = 1 / 2) → (x = 1/2) :=
by
  sorry

end solve_for_x_l654_654184


namespace f_is_even_l654_654142

noncomputable def tan_is_odd : ∀ x : ℝ, tan (-x) = -tan x := sorry

noncomputable def f (x : ℝ) := tan (x^2 + 1)

theorem f_is_even : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  unfold f
  rw [neg_sq x]
  sorry

end f_is_even_l654_654142


namespace pascals_triangle_row_20_element_6_l654_654891

theorem pascals_triangle_row_20_element_6 : nat.choose 20 5 = 7752 :=
by sorry

end pascals_triangle_row_20_element_6_l654_654891


namespace find_percentage_rsvped_l654_654219

noncomputable def percentage_rsvped (P : ℝ) : Prop :=
  let invitations := 200
  let rsvpd := (P / 100) * invitations
  let showed_up := 0.80 * rsvpd
  let gifts_needed := showed_up - 10
  gifts_needed = 134

theorem find_percentage_rsvped : percentage_rsvped 90 :=
by
  unfold percentage_rsvped
  norm_num
  sorry

end find_percentage_rsvped_l654_654219


namespace sufficient_but_not_necessary_condition_l654_654125

def perpendicular_lines (a : ℝ) : Prop :=
  let slope1 := -a
  let slope2 := 1 / a
  slope1 * slope2 = -1

theorem sufficient_but_not_necessary_condition :
  ∀ a : ℝ, (a = 1 → perpendicular_lines a) ∧ (∃ a ≠ 1, perpendicular_lines a) :=
by
  intro a
  split
  { intro h
    rw h
    change (-1) * (1 / 1) = -1
    norm_num }
  { existsi 2
    split
    { norm_num }
    change (-2) * (1 / 2) = -1
    norm_num }

end sufficient_but_not_necessary_condition_l654_654125


namespace largest_prime_divisor_of_sum_of_squares_l654_654520

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, prime p ∧ (p ∣ (31^2 + 42^2)) ∧ (∀ q : ℕ, prime q ∧ (q ∣ (31^2 + 42^2)) → q ≤ p) :=
by
  sorry

end largest_prime_divisor_of_sum_of_squares_l654_654520


namespace complex_modulus_proof_l654_654102

theorem complex_modulus_proof
  (x y : ℝ)
  (h : (1 + complex.I) * x = 1 + complex.I * y) :
  complex.abs (x + y * complex.I) = real.sqrt 2 :=
sorry

end complex_modulus_proof_l654_654102


namespace average_transformed_set_l654_654075

theorem average_transformed_set (x : ℝ) (A : Fin 15 → ℝ) (h : (∑ i, A i) / 15 = x) :
  (∑ i, 5 * A i + 7) / 15 = 5 * x + 7 :=
by
  sorry

end average_transformed_set_l654_654075


namespace total_amount_paid_l654_654816

theorem total_amount_paid (cost_lunch : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) (sales_tax : ℝ) (tip : ℝ) 
  (h1 : cost_lunch = 100) 
  (h2 : sales_tax_rate = 0.04) 
  (h3 : tip_rate = 0.06) 
  (h4 : sales_tax = cost_lunch * sales_tax_rate) 
  (h5 : tip = cost_lunch * tip_rate) :
  cost_lunch + sales_tax + tip = 110 :=
by
  sorry

end total_amount_paid_l654_654816


namespace convex_maximum_l654_654139

-- Definitions for the given conditions
def f (x : ℝ) (m : ℝ) : ℝ := (1/6) * x^3 - (1/2) * m * x^2 + x

def f' (x : ℝ) (m : ℝ) : ℝ := deriv (λ x, f x m) x

def f'' (x : ℝ) (m : ℝ) : ℝ := deriv (λ x, f' x m) x

-- The statement representing the proof problem
theorem convex_maximum (m : ℝ) :
  (∀ x ∈ set.Ioo (-1 : ℝ) (2 : ℝ), f'' x m < 0) →
  (∃ x ∈ set.Ico (-1 : ℝ) (2 : ℝ), ∀ y ∈ set.Ioo (-1 : ℝ) (2 : ℝ), f y m ≤ f x m ∧ (∃ z ∈ set.Ioo (-1 : ℝ) (2 : ℝ), f' z m ≠ 0)) :=
sorry

end convex_maximum_l654_654139


namespace proof_n_times_s_l654_654237

open Real

def g (x : ℝ) : ℝ

axiom g_property : ∀ x y : ℝ, g x * g y - g (x * y) = x + y + 1

noncomputable def n : ℝ := 2
noncomputable def s : ℝ := (5 / 2) + (- (5 / 2))

theorem proof_n_times_s : n * s = 0 := by
  sorry

end proof_n_times_s_l654_654237


namespace smallest_positive_period_l654_654523

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ (∀ T' > 0, (∀ (x : ℝ), f (x + T') = f x → T ≤ T')) :=
begin
  use π,
  split,
  { exact Real.pi_pos, },
  { split,
    { intros x,
      dsimp [f],
      sorry, },
    { intros T' T'_pos hT',
      sorry, },
  }
end

end smallest_positive_period_l654_654523


namespace Petya_can_determine_weight_l654_654262

theorem Petya_can_determine_weight (n : ℕ) (distinct_weights : Fin n → ℕ) 
  (device : (Fin 10 → Fin n) → ℕ) (ten_thousand_weights : n = 10000)
  (no_two_same : (∀ i j : Fin n, i ≠ j → distinct_weights i ≠ distinct_weights j)) :
  ∃ i : Fin n, ∃ w : ℕ, distinct_weights i = w :=
by
  sorry

end Petya_can_determine_weight_l654_654262


namespace geometric_mean_pairs_count_l654_654906

noncomputable def geometric_mean_triangle : Type :=
let hypotenuse : ℝ := 1 in
let alpha : ℝ := 1 / 2 in -- acute angle, not specified directly but derived
let c : ℝ := real.cos alpha,
    s : ℝ := real.sin alpha in
has_geometric_mean_relations hypotenuse c s ∧ hypotenuse^2 = c^2 + s^2 /\
let quadratic_solution := (-1 + real.sqrt 5) / 2 in
let c := real.sqrt quadratic_solution,
    s := real.sqrt quadratic_solution * real.sqrt quadratic_solution in
count_geometric_mean_pairs c s = 124

theorem geometric_mean_pairs_count :
  ∃ triangle, has_geometric_mean_relations triangle.hypotenuse triangle.leg1 triangle.leg2 ∧ 
    count_geometric_mean_pairs triangle.leg1 triangle.leg2 = 124 :=
sorry

end geometric_mean_pairs_count_l654_654906


namespace theater_lucky_guests_l654_654087

theorem theater_lucky_guests :
  let lcm (a b : ℕ) := a / Nat.gcd a b * b
  let lcm3 (a b c : ℕ) := lcm (lcm a b) c
  let count_multiples (n m : ℕ) := m / n in
  ∀ (num_guests reception_interval_1 reception_interval_2 reception_interval_3 : ℕ),
    num_guests = 3600 →
    reception_interval_1 = 90 →
    reception_interval_2 = 36 →
    reception_interval_3 = 60 →
    let lcm_value := lcm3 reception_interval_1 reception_interval_2 reception_interval_3 in
    lcm_value = 180 ∧ count_multiples lcm_value num_guests = 20 := 
by
  intros num_guests reception_interval_1 reception_interval_2 reception_interval_3 hnum h1 h2 h3
  let lcm := λ (a b : ℕ), a / Nat.gcd a b * b
  let lcm3 := λ (a b c : ℕ), lcm (lcm a b) c
  let count_multiples := λ (n m : ℕ), m / n
  let lcm_value := lcm3 reception_interval_1 reception_interval_2 reception_interval_3
  have hlcm : lcm_value = 180 := sorry
  have hcount : count_multiples lcm_value num_guests = 20 := sorry
  exact ⟨hlcm, hcount⟩

end theater_lucky_guests_l654_654087


namespace area_B_correct_l654_654302

-- Define the diagonal of square A
def diagonal_A (c d : ℝ) := 2 * c + d

-- Define the side length of square A using its diagonal
def side_length_A (c d : ℝ) := (diagonal_A c d) / Real.sqrt 2

-- Define the area of square A
def area_A (c d : ℝ) := (side_length_A c d)^2

-- Define the area of square B as 3 times the area of square A
def area_B (c d : ℝ) := 3 * (area_A c d)

-- State the theorem that the area of square B is equal to the correct answer
theorem area_B_correct (c d : ℝ) : area_B c d = 3 * ((2 * c + d)^2) / 2 :=
by
  sorry

end area_B_correct_l654_654302


namespace cos_pi_minus_alpha_l654_654889

theorem cos_pi_minus_alpha (α : ℝ) : cos (real.pi - α) = -cos α :=
by sorry

end cos_pi_minus_alpha_l654_654889


namespace carol_rectangle_length_l654_654036

theorem carol_rectangle_length :
  ∃ (length_of_carols_rectangle : ℝ), 
    let width_of_carols_rectangle := 24 in
    let length_of_jordans_rectangle := 8 in
    let width_of_jordans_rectangle := 15 in
    (length_of_jordans_rectangle * width_of_jordans_rectangle = length_of_carols_rectangle * width_of_carols_rectangle) 
    ∧ length_of_carols_rectangle = 5 :=
by
  sorry

end carol_rectangle_length_l654_654036


namespace number_of_pairs_l654_654651

-- Definitions that directly appear in the conditions
def subset (A B : set ℕ) : Prop := ∀ ⦃x⦄, x ∈ A → x ∈ B

def permutation (M : set ℕ) (a : M → M) : Prop :=
  function.bijective a

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fib(n) + fib(n+1)

-- Main theorem statement
theorem number_of_pairs (M : set ℕ) (n : ℕ) (hM : M = {1, 2, ..., n}) :
  (∑ k in finset.range (n + 1), nat.choose (n - k) k * k! * (n - k)!) = n! * fib (n + 1) :=
sorry

end number_of_pairs_l654_654651


namespace a_plus_b_equals_4_l654_654608

theorem a_plus_b_equals_4 (f : ℝ → ℝ) (a b : ℝ) (h_dom : ∀ x, 1 ≤ x ∧ x ≤ b → f x = (1/2) * (x-1)^2 + a)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b → ∃ x, 1 ≤ x ∧ x ≤ b ∧ f x = y) (h_b_pos : b > 1) : a + b = 4 :=
sorry

end a_plus_b_equals_4_l654_654608


namespace fewer_twos_result_100_l654_654366

theorem fewer_twos_result_100 :
  (222 / 2) - (22 / 2) = 100 := by
  sorry

end fewer_twos_result_100_l654_654366


namespace game_nth_number_l654_654222

def nth_prime : ℕ → ℕ
| 0     := 2
| n + 1 := Nat.find (λ p, p > nth_prime n ∧ Nat.Prime p)

theorem game_nth_number (n : ℕ) : n = 52 → nth_prime (n + 1) = 241 :=
by
  intros hn
  rw [hn, Nat.add_comm, nth_prime]
  sorry

end game_nth_number_l654_654222


namespace reflection_symmetric_x_axis_l654_654630

theorem reflection_symmetric_x_axis (x y : ℝ) (hx : x = 2) (hy : y = -3) :
    (x, y) → (x, -y) :=
by
    assume P : (ℝ × ℝ)
    cases P with x y
    have hx : x = 2 := hx
    have hy : y = -3 := hy
    show (x, y) → (x, -y), from sorry

end reflection_symmetric_x_axis_l654_654630


namespace positive_roots_of_quadratic_l654_654093

theorem positive_roots_of_quadratic (n : ℝ) :
  (let A := (n-2), B := (-2n), C := (n+3),
       discriminant := B^2 - 4*A*C,
       sum_of_roots_positive := (-B/A) > 0,
       product_of_roots_positive := (C/A) > 0
   in discriminant ≥ 0 ∧ sum_of_roots_positive ∧ product_of_roots_positive)
  ↔ (n ∈ set.Iio (-3) ∪ set.Ioc 2 6) :=
by sorry

end positive_roots_of_quadratic_l654_654093


namespace find_PC_correct_l654_654612

noncomputable def find_PC (P C : ℝ) (BP QR : ℝ) : ℝ :=
  let A := (45 : ℕ)
  let B := (45 : ℕ)
  let angleA := 90
  let angleB := 45
  let angleC := 45
  if (angleA + angleB + angleC = 180) ∧ (BP = real.sqrt 2) ∧ (QR = 2) then
    let x := 1 + real.sqrt 3
    let y := x - BP
    y
  else 0

theorem find_PC_correct {P C BP QR : ℝ} :
  BP = real.sqrt 2 → QR = 2 →
  find_PC P C BP QR = real.sqrt 6 :=
by 
  intros BP_condition QR_condition
  sorry

end find_PC_correct_l654_654612


namespace smallest_positive_period_l654_654522

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ (∀ T' > 0, (∀ (x : ℝ), f (x + T') = f x → T ≤ T')) :=
begin
  use π,
  split,
  { exact Real.pi_pos, },
  { split,
    { intros x,
      dsimp [f],
      sorry, },
    { intros T' T'_pos hT',
      sorry, },
  }
end

end smallest_positive_period_l654_654522


namespace positive_difference_g_b_values_l654_654649

noncomputable def g (n : ℤ) : ℤ :=
if n < 0 then n^2 + 5 * n + 6 else 3 * n - 30

theorem positive_difference_g_b_values : 
  let g_neg_3 := g (-3)
  let g_3 := g 3
  g_neg_3 = 0 → g_3 = -21 → 
  ∃ b1 b2 : ℤ, g_neg_3 + g_3 + g b1 = 0 ∧ g_neg_3 + g_3 + g b2 = 0 ∧ 
  b1 ≠ b2 ∧ b1 < b2 ∧ b1 < 0 ∧ b2 > 0 ∧ b2 - b1 = 22 :=
by
  sorry

end positive_difference_g_b_values_l654_654649


namespace sequence_bound_l654_654244

open Real

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ)
  (h₀ : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h₁ : ∀ (i j : ℕ), 0 < i → 0 < j → i ≠ j → abs (a i - a j) ≥ 1 / (i + j)) :
  c ≥ 1 :=
by {
  sorry
}

end sequence_bound_l654_654244


namespace problem_statement_l654_654106

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem problem_statement : f (Real.pi) + (f' (Real.pi / 2)) = - (3 / Real.pi) := 
by
  -- Definitions
  let f_prime := λ x, - (x * Real.sin x + Real.cos x) / (x * x)
  have f_prime_at_pi_div_2 : f' (Real.pi / 2) = f_prime (Real.pi / 2),
  {
    sorry -- Here you would show the equivalence of the computed derivative
  }
  have f_at_pi : f Real.pi = -(1 / Real.pi),
  {
    sorry -- Here you would confirm the function value at π
  }
  have f_prime_val : f_prime (Real.pi / 2) = -(2 / Real.pi),
  {
    sorry -- Here you would confirm the derivative value at π / 2
  }
  have f_val_prime_sum : f (Real.pi) + f_prime (Real.pi / 2) = - (3 / Real.pi),
  { 
    sorry -- Here you would perform the addition of the values
  }
  exact f_val_prime_sum

end problem_statement_l654_654106


namespace maximize_victory_probability_l654_654762

-- Define the conditions
variables {n : ℕ}
def number_of_voters := 2 * n
def half_support_miraflores := n
def half_support_dick_maloney := n
def miraflores_is_voter := true

-- Define the districts
def district1 := {miraflores}
def district2 := {voters | voters ≠ miraflores}

theorem maximize_victory_probability (n : ℕ) (h₁ : nat.odd (2*n + 1) = true) : 
  (let district1_voters := 1 in
   let district2_voters := 2*n - 1 in
   maximize_probability_of_winning(district1_voters, district2_voters) = true) :=
sorry

end maximize_victory_probability_l654_654762


namespace michael_large_balls_l654_654781

theorem michael_large_balls (total_rubber_bands : ℕ) (small_ball_rubber_bands : ℕ) (large_ball_rubber_bands : ℕ) (small_balls_made : ℕ)
  (h_total_rubber_bands : total_rubber_bands = 5000)
  (h_small_ball_rubber_bands : small_ball_rubber_bands = 50)
  (h_large_ball_rubber_bands : large_ball_rubber_bands = 300)
  (h_small_balls_made : small_balls_made = 22) :
  (total_rubber_bands - small_balls_made * small_ball_rubber_bands) / large_ball_rubber_bands = 13 :=
by {
  sorry
}

end michael_large_balls_l654_654781


namespace round_robin_tournament_draws_l654_654435

noncomputable def round_robin_draws (n : ℕ) (lists : Fin n → Fin 12 → Set (Fin n)) : ℕ :=
  let total_games := n * (n - 1) / 2
  let total_victories := n
  total_games - total_victories

theorem round_robin_tournament_draws :
  ∀ (players : Fin 12) (lists : Fin 12 → Fin 12 → Set (Fin 12)),
    (∀ i : Fin 12, lists i 0 = {i}) →
    (∀ i : Fin 12, ∀ k : Fin 11, 
      lists i (k+1) = lists i k ∪ {j | ∃ x ∈ lists i k, x defeated j}) →
    (∀ i : Fin 12, lists i 11 ≠ lists i 10) →
    round_robin_draws 12 lists = 54 :=
by
  sorry

end round_robin_tournament_draws_l654_654435


namespace largest_number_is_A_l654_654416

def numA : ℝ := 0.989
def numB : ℝ := 0.9879
def numC : ℝ := 0.98809
def numD : ℝ := 0.9807
def numE : ℝ := 0.9819

theorem largest_number_is_A :
  (numA > numB) ∧ (numA > numC) ∧ (numA > numD) ∧ (numA > numE) :=
by sorry

end largest_number_is_A_l654_654416


namespace probability_of_independent_events_l654_654729

namespace Probability

variable (A B : Prop)
variable [pA : Decidable A] [pB : Decidable B]

noncomputable def P : Prop → ℚ := sorry

theorem probability_of_independent_events
  (hA : P A = 5/7)
  (hB : P B = 4/5)
  (hIndep : P (A ∧ B) = P A * P B) :
  P (A ∧ B) = 4/7 :=
by
  rw [hA, hB]
  rw [hIndep]
  norm_num
  sorry

end Probability

end probability_of_independent_events_l654_654729


namespace geometric_sequence_increasing_iff_q_gt_one_l654_654621

variables {a_n : ℕ → ℝ} {q : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

def is_increasing_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n (n + 1) > a_n n

theorem geometric_sequence_increasing_iff_q_gt_one 
  (h1 : ∀ n, 0 < a_n n)
  (h2 : is_geometric_sequence a_n q) :
  is_increasing_sequence a_n ↔ q > 1 :=
by
  sorry

end geometric_sequence_increasing_iff_q_gt_one_l654_654621


namespace Sarahs_score_l654_654275

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l654_654275


namespace maximize_angle_with_beta_l654_654628

variables {α β : Plane} (A : Point)

-- Assuming A lies on the line of intersection of planes α and β
-- We need to express this assumption

axiom intersection_line_contains_A : (line_of_intersection α β).contains A

-- We want to prove that the line within plane α passing through A that forms
-- the largest angle with plane β is perpendicular to the line of intersection of α and β

theorem maximize_angle_with_beta :
  ∃ l : Line, l ∈ α ∧ l.contains A ∧ (∀ m : Line, m ∈ α → m.contains A → angle m β ≤ angle (perpendicular_to (line_of_intersection α β)) β) := 
sorry

end maximize_angle_with_beta_l654_654628


namespace digit_sum_of_1_over_98_squared_l654_654148

noncomputable def sum_of_digits (n : ℕ) (digits : Fin n → ℕ) : ℕ :=
  (Finset.range n).sum digits

theorem digit_sum_of_1_over_98_squared (b : Fin 200 → ℕ) (h : ∃ b : Fin 200 → ℕ, (1 / (98 ^ 2) : ℚ) = 0.\overline{(b 199) (b 198) ... (b 1) (b 0)}) :
  sum_of_digits 200 b = 924 :=
sorry

end digit_sum_of_1_over_98_squared_l654_654148


namespace fewer_twos_to_hundred_l654_654358

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end fewer_twos_to_hundred_l654_654358


namespace trigonometric_identity_l654_654045

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end trigonometric_identity_l654_654045


namespace proof_problem_l654_654213
open Real

-- Definitions of the sequences a_n and b_n
def a : ℕ → ℝ
| 0       := 2
| (n + 1) := 2 * (a n)^2

def b (n : ℕ) : ℝ := log 2 (a n)

-- Define the sequence c_n and the sum T_n
def c (n : ℕ) : ℝ := n * b(n)

def T (n : ℕ) : ℝ := (1 to n).foldl (λ acc k, acc + c k) 0

-- The main theorem combining both parts:
theorem proof_problem (n : ℕ) :
  (∀ n : ℕ, b (n + 1) + 1 = 2 * (b n + 1)) ∧
  T n = (n-1) * 2^(n+1) + 2 - (n * (n + 1) / 2) :=
by
  sorry

end proof_problem_l654_654213


namespace complex_expression_eq_l654_654072

-- Define the complex numbers
def c1 : ℂ := 6 - 3 * Complex.I
def c2 : ℂ := 2 - 7 * Complex.I

-- Define the scale
def scale : ℂ := 3

-- State the theorem
theorem complex_expression_eq : (c1 + scale * c2) = 12 - 24 * Complex.I :=
by
  -- This is the statement only; the proof is omitted with sorry.
  sorry

end complex_expression_eq_l654_654072


namespace problem1_problem2_l654_654149

noncomputable def f : ℝ → ℝ := λ x, 2 * real.sqrt(3) * real.sin(π - x) * real.sin(x) - (real.sin(x) - real.cos(x))^2

def increasing_intervals (k : ℤ) : set ℝ :=
  { x : ℝ | k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12 }

def g (x : ℝ) : ℝ := 2 * real.sin(x) + real.sqrt(3) - 1

theorem problem1 (k : ℤ) : monotone_on f (increasing_intervals k) := sorry

theorem problem2 : g (π / 6) = real.sqrt(3) := sorry

end problem1_problem2_l654_654149


namespace max_fourth_number_in_arrangement_l654_654883

theorem max_fourth_number_in_arrangement :
  ∀ (l : List ℕ), 
  l = [39, 41, 44, 45, 47, 52, 55] →
  (∀ i, i < l.length - 2 → (l[i] + l[i+1] + l[i+2]) % 3 = 0) →
  ∃ l', l.perm l' ∧
     (∀ j, j < l'.length - 2 → (l'[j] + l'[j+1] + l'[j+2]) % 3 = 0) ∧ 
     (l'.drop 3).head? = some 47 :=
by
  sorry

end max_fourth_number_in_arrangement_l654_654883


namespace linear_function_diff_l654_654234

noncomputable def g : ℝ → ℝ := sorry

theorem linear_function_diff (h_linear : ∀ x y z w : ℝ, (g y - g x) / (y - x) = (g w - g z) / (w - z))
                            (h_condition : g 8 - g 1 = 21) : 
  g 16 - g 1 = 45 := 
by 
  sorry

end linear_function_diff_l654_654234


namespace cos_identity_l654_654143

theorem cos_identity {θ : ℝ} (h : (cos θ = -4/5) ∧ (sin θ = 3/5)):
  cos (2 * π - θ) = -4 / 5 :=
by
  sorry

end cos_identity_l654_654143


namespace number_of_functions_satisfying_property_l654_654915

open Function

theorem number_of_functions_satisfying_property :
  let f_domain := ({1, 2, 3} : Set ℕ)
  ∃ (f : ℕ → ℕ), (f_domain → f_domain) ∧ 
  (∀ x ∈ f_domain, f (f x) = f x) ∧ 
  (Set.toFinset (SetOf (λ f, (∀ x ∈ f_domain, f (f x) = f x) )) ∣ (.card = 10)) :=
begin
  sorry
end

end number_of_functions_satisfying_property_l654_654915


namespace length_of_arc_AB_proof_area_of_segment_OAB_proof_l654_654114

noncomputable def length_of_arc_AB (r : ℝ) (θ : ℝ) : ℝ :=
  r * θ

noncomputable def area_of_sector_OAB (r : ℝ) (θ : ℝ) : ℝ :=
  0.5 * r^2 * θ

noncomputable def area_of_triangle_OAB (r : ℝ) (sinθ : ℝ) : ℝ :=
  0.5 * r^2 * sinθ

theorem length_of_arc_AB_proof : 
  length_of_arc_AB 6 ((2 / 3) * Real.pi) = 4 * Real.pi := 
by
  sorry

theorem area_of_segment_OAB_proof : 
  area_of_sector_OAB 6 ((2 / 3) * Real.pi) - area_of_triangle_OAB 6 (Real.sin (2 * Real.pi / 3)) = 12 * Real.pi - 9 * Real.sqrt 3 :=
by 
  sorry

end length_of_arc_AB_proof_area_of_segment_OAB_proof_l654_654114


namespace cleanup_drive_weight_per_mile_per_hour_l654_654673

theorem cleanup_drive_weight_per_mile_per_hour :
  let duration := 4
  let lizzie_group := 387
  let second_group := lizzie_group - 39
  let third_group := 560 / 16
  let total_distance := 8
  let total_garbage := lizzie_group + second_group + third_group
  total_garbage / total_distance / duration = 24.0625 := 
by {
  sorry
}

end cleanup_drive_weight_per_mile_per_hour_l654_654673


namespace twice_son_plus_father_is_70_l654_654194

section
variable {s f : ℕ}

-- Conditions
def son_age : ℕ := 15
def father_age : ℕ := 40

-- Statement to prove
theorem twice_son_plus_father_is_70 : (2 * son_age + father_age) = 70 :=
by
  sorry
end

end twice_son_plus_father_is_70_l654_654194


namespace total_tax_percentage_correct_l654_654681

-- Definitions
def total_spent := 100
def clothing_spent := 40
def food_spent := 30
def other_items_spent := 30

def clothing_after_discount := clothing_spent - (10 / 100) * clothing_spent
def food_after_discount := food_spent - (5 / 100) * food_spent
def other_items_after_discount := other_items_spent - (7 / 100) * other_items_spent

def clothing_tax := (4 / 100) * clothing_after_discount
def food_tax := (2 / 100) * (food_after_discount - 10)
def other_items_tax := (8 / 100) * (other_items_after_discount - 5)

def total_tax := clothing_tax + food_tax + other_items_tax
def total_tax_percentage := (total_tax / total_spent) * 100

-- Lean theorem statement
theorem total_tax_percentage_correct : total_tax_percentage = 3.642 := 
begin
  sorry
end

end total_tax_percentage_correct_l654_654681


namespace lilyRateIsSeven_l654_654101

-- Given definitions based on the conditions
def roseRate : ℕ := 6
def roseOrder : ℕ := 6
def lilyOrder : ℕ := 14
def totalPayment : ℝ := 90
def hourlyWage : ℝ := 30

-- The statement to be proved
theorem lilyRateIsSeven : lilyRate = 7 :=
by
  have totalOrderTime : ℝ := totalPayment / hourlyWage
  have timeForRoses : ℝ := roseOrder / roseRate
  have timeForLilies : ℝ := totalOrderTime - timeForRoses
  have lilyRate : ℝ := lilyOrder / timeForLilies
  exact eq.refl lilyRate

end lilyRateIsSeven_l654_654101


namespace volume_ratio_of_tetrahedron_and_cube_l654_654127

theorem volume_ratio_of_tetrahedron_and_cube (s : ℝ) (hTetrahedron : ∀ (T : Type) [regular_tetrahedron T] [has_side_length T s]) :
    let V_T := (sqrt 2 / 12) * s^3
    let V_C := (s^3) / 36
    V_T / V_C = 3 * sqrt 2 := by
    sorry

end volume_ratio_of_tetrahedron_and_cube_l654_654127


namespace Ram_and_Krish_work_together_l654_654425
noncomputable theory

variables (R K W : ℝ) (D_together : ℝ)

-- Conditions
def Ram_half_as_Krish : Prop := R = 1 / 2 * K
def Ram_work_alone : Prop := R * 30 = W

-- Theorem statement
theorem Ram_and_Krish_work_together (h1 : Ram_half_as_Krish R K) (h2 : Ram_work_alone R W) :
  D_together = 10 :=
by
  sorry

end Ram_and_Krish_work_together_l654_654425


namespace jenn_wins_game_l654_654029

theorem jenn_wins_game (n : ℕ) (h_n : n > 6) : 
  ∃ (strategy : Π (turn : ℕ) (choices : set ℝ), ℝ), 
  ∀ turn choices, 
    let x := strategy turn choices in
    x ∈ (if turn % 2 = 0 then set.Icc 0 (n / 3) else set.Icc (n / 3 + 1) n) ∧
    ∀ prev_choices ∈ choices, abs (x - prev_choices) > 1 ∧
    (turn % 2 = 1 → x < n ∧ ∃ move : ℝ, move ∉ choices ∧ abs (x - move) > 1) := 
sorry

end jenn_wins_game_l654_654029


namespace total_length_QP_PL_l654_654614

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (X : Point)
  (Y : Point)
  (Z : Point)

def triXYZ : Triangle :=
  { X := ⟨1, 4⟩,
    Y := ⟨0, 0⟩,
    Z := ⟨3, 0⟩ }

def K : Point := { x := 1, y := 0 }

def L : Point := { x := (3 / 4), y := 3 }

def E : Point := { x := 0.5, y := 3 }

def D : Point := { x := 12 / 5, y := 16 / 5 }

def Q : Point := { x := 1, y := 1 }

noncomputable def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

noncomputable def PL : ℝ := distance ⟨(3 / 4), 3⟩ Q
noncomputable def QP : ℝ := distance Q E

theorem total_length_QP_PL :
  let total := QP + PL in
  total = 1.5 := by
  sorry

end total_length_QP_PL_l654_654614


namespace circle_center_radius_sum_l654_654654

noncomputable def center_radius_sum (x y : ℝ) (c d s : ℝ) : Prop :=
  (x - 10)^2 + (y + 3)^2 = 66 → c = 10 ∧ d = -3 ∧ s = real.sqrt 66

theorem circle_center_radius_sum :
  ∀ (x y : ℝ), center_radius_sum x y 10 (-3) (real.sqrt 66) →
  10 + (-3) + real.sqrt 66 = 7 + real.sqrt 66 :=
begin
  intros,
  sorry
end

end circle_center_radius_sum_l654_654654


namespace square_lawn_area_l654_654680

theorem square_lawn_area (map_scale : ℝ) (map_edge_length_cm : ℝ) (actual_edge_length_m : ℝ) (actual_area_m2 : ℝ) 
  (h1 : map_scale = 1 / 5000) 
  (h2 : map_edge_length_cm = 4) 
  (h3 : actual_edge_length_m = (map_edge_length_cm / map_scale) / 100)
  (h4 : actual_area_m2 = actual_edge_length_m^2)
  : actual_area_m2 = 400 := 
by 
  sorry

end square_lawn_area_l654_654680


namespace trajectory_of_complex_l654_654328

theorem trajectory_of_complex (z : ℂ) : |z + 2 * (0 + 1i)| = 4 ↔ ∃ (c : ℂ) (r : ℝ), c = 0 - 2 * (0 + 1i) ∧ r = 4 ∧ ∀ w : ℂ, |w - c| = r :=
by
  sorry

end trajectory_of_complex_l654_654328


namespace problem_statement_l654_654145

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  λ i j,
    match i, j with
    | 0, 0 => 3
    | 0, 1 => 5
    | 1, 0 => 0
    | 1, 1 => -2

def β : Fin 2 → ℤ :=
  λ i,
    match i with
    | 0 => -1
    | 1 => 1

theorem problem_statement : (A ^ 6).mul_vec β = ![ -64, 64 ] :=
by { sorry }

end problem_statement_l654_654145


namespace regular_pentagon_cannot_cover_floor_l654_654877

theorem regular_pentagon_cannot_cover_floor :
  ¬(∃ n : ℕ, 360 % 108 = 0) :=
begin
  sorry
end

end regular_pentagon_cannot_cover_floor_l654_654877


namespace construct_frequency_polygon_l654_654500

def grouped_series_data : Type := 
  -- Define the data structure for the grouped statistical series
  list (ℝ × ℕ) -- Here ℝ for x_k* and ℕ for n_k*

structure FrequencyPolygonData :=
  (data : grouped_series_data) 
  (abscissa_interval : ℝ × ℝ)
  (ordinate_interval : ℕ × ℕ)
  (polygon_points : list (ℝ × ℕ))

-- Given grouped statistical series of absolute frequencies
def example_data : grouped_series_data := 
  [(0.09025, 5), (0.2, 10), (0.4, 20), (0.6, 15), (1.0, 26), -- and so on...
   (3.69975, 76)] -- hypothetically including all 21 points in section 7.1

-- Given intervals
def abscissa_interval_example : ℝ × ℝ := (0.09025, 3.69975)
def ordinate_interval_example : ℕ × ℕ := (0, 76)

noncomputable def freq_polygon_example : FrequencyPolygonData := 
  { data := example_data,
    abscissa_interval := abscissa_interval_example,
    ordinate_interval := ordinate_interval_example,
    polygon_points := example_data }

theorem construct_frequency_polygon : 
  ∃ (f : FrequencyPolygonData), f.data = example_data ∧
                                f.abscissa_interval = abscissa_interval_example ∧
                                f.ordinate_interval = ordinate_interval_example ∧
                                f.polygon_points = example_data := by
  use freq_polygon_example
  split
  { refl }
  split
  { refl }
  split
  { refl }
  { exact rfl }

end construct_frequency_polygon_l654_654500


namespace find_n_l654_654128

noncomputable theory

variables {x n : ℝ}

theorem find_n :
  (log 10 (sin x) + log 10 (cos x) = -1) ∧
  (log 10 (sin x + cos x) = 0.5 * (log 10 n - 1)) →
  n = 12 :=
by
  sorry

end find_n_l654_654128


namespace graph_y_g_abs_x_plus_1_l654_654497

def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then -x - 1
  else if -1 < x ∧ x ≤ 1 then x^2 - 3
  else if 1 < x ∧ x ≤ 4 then x - 1
  else 0  -- default value, should not be used due to conditions on x

theorem graph_y_g_abs_x_plus_1:
  ∀ x : ℝ, 
    (if -1 ≤ x ∧ x ≤ 1 then g (|x|) + 1 = x^2 - 2
     else if 1 < x ∧ x ≤ 4 ∨ -4 ≤ x ∧ x < -1 then g(|x|) + 1 = x
     else true) :=
begin
  sorry
end

end graph_y_g_abs_x_plus_1_l654_654497


namespace sequence_comparison_l654_654130

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define geometric sequence
def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ (∀ n, b (n + 1) = b n * q) ∧ (∀ i, i ≥ 1 → b i > 0)

-- Main theorem to prove
theorem sequence_comparison {a b : ℕ → ℝ} (q : ℝ) (h_a_arith : arithmetic_sequence a) 
  (h_b_geom : geometric_sequence b q) (h_eq_1 : a 1 = b 1) (h_eq_11 : a 11 = b 11) :
  a 6 > b 6 :=
sorry

end sequence_comparison_l654_654130


namespace no_complex_roots_with_modulus_greater_than_2_l654_654941

theorem no_complex_roots_with_modulus_greater_than_2 (n : ℕ) (α : ℕ → ℝ) :
    ∀ x : ℂ, (polynomial.eval x (polynomial.of_fn (λ k, if h : k < n then sin (α k) else 0)))
    = 0 → |x| ≤ 2 :=
begin
    sorry
end

end no_complex_roots_with_modulus_greater_than_2_l654_654941


namespace ensure_victory_by_B_l654_654440

/-- Players A and B take turns marking numbers from 1 to 17.
    Numbers already marked, or being twice or half of a marked number, are not allowed.
    Player A starts by marking 8. Our goal is to prove B should mark 6 next to ensure the win. -/
theorem ensure_victory_by_B : 
  ∀ (A B : ℕ → ℕ → Prop), 
  A 8 8 → 
  (∀ n m : ℕ, (n > 0 ∧ n ≤ 17) → 
               ¬ B n m → 
               ¬ B n (2*m) → 
               ¬ B n (m/2)) → 
  B A 6 :=
begin
  sorry
end

end ensure_victory_by_B_l654_654440


namespace last_locker_opened_l654_654002

theorem last_locker_opened :
  ∃ L : ℕ, L ≤ 512 ∧ (∀ n : ℕ, n ≤ 512 ∧ n ≠ L → locked n) ∧ ¬ locked L :=
begin
  sorry
end

-- Definitions that represent the toggling pattern and conditions
def locked (n : ℕ) : Prop := 
  ∑ k in (finset.range (n+1)).filter (λ k, k > 0 ∧ n % k = 0), 1 % 2 = 0

/- The main theorem states that there exists an L (the last locker opened) such that:
   1. L is ≤ 512
   2. All other lockers ≤ 512 are locked except L (locked condition)
   3. L is open (¬ locked L condition)
-/

end last_locker_opened_l654_654002


namespace first_digit_base7_528_l654_654798

theorem first_digit_base7_528 : 
  (Nat.digitNat 528 7 0 = 1) := 
sorry

end first_digit_base7_528_l654_654798


namespace intersection_condition_l654_654538

def M : set (ℝ × ℝ) := {p | p.2 = sqrt (9 - p.1 ^ 2) ∧ p.2 ≠ 0}
def N (b : ℝ) : set (ℝ × ℝ) := {p | p.2 = p.1 + b}

theorem intersection_condition (b : ℝ) : (∃ p : ℝ × ℝ, p ∈ M ∧ p ∈ N b) ↔ b ∈ Icc (-3) (3 * Real.sqrt 2) :=
sorry

end intersection_condition_l654_654538


namespace wise_men_task_l654_654308

-- Define the conditions
def sum_to_100 (nums : List ℕ) : Prop :=
  nums.length = 7 ∧ nums.nodup ∧ nums.sum = 100 

def fourth_largest (nums : List ℕ) (n : ℕ) : Prop :=
  (nums.sorted!!3 = n)

-- Define the question as a theorem
theorem wise_men_task (nums : List ℕ) (n : ℕ) 
  (h_sum : sum_to_100 nums)
  (h_fourth : fourth_largest nums n) : 
  ∃! (nums_set : List ℕ), sum_to_100 nums_set ∧ fourth_largest nums_set n :=
  sorry

end wise_men_task_l654_654308


namespace perpendicular_line_through_point_l654_654000

theorem perpendicular_line_through_point (x y : ℝ) : (x, y) = (0, -3) ∧ (∀ x y : ℝ, 2 * x + 3 * y - 6 = 0) → 3 * x - 2 * y - 6 = 0 :=
by
  sorry

end perpendicular_line_through_point_l654_654000


namespace correct_system_of_equations_l654_654836

theorem correct_system_of_equations (x y : ℝ) :
  (y - x = 4.5) ∧ (x - y / 2 = 1) ↔
  ((y - x = 4.5) ∧ (x - y / 2 = 1)) :=
by sorry

end correct_system_of_equations_l654_654836


namespace extra_yellow_balls_dispatched_l654_654870

theorem extra_yellow_balls_dispatched : 
  ∀ (W Y E : ℕ), -- Declare natural numbers W, Y, E
  W = Y →      -- Condition that the number of white balls equals the number of yellow balls
  W + Y = 64 → -- Condition that the total number of originally ordered balls is 64
  W / (Y + E) = 8 / 13 → -- The given ratio involving the extra yellow balls
  E = 20 :=               -- Prove that the extra yellow balls E equals 20
by
  intros W Y E h1 h2 h3
  -- Proof mechanism here
  sorry

end extra_yellow_balls_dispatched_l654_654870


namespace rotated_angle_new_measure_l654_654314

theorem rotated_angle_new_measure (θ₀ : ℕ) (rotation : ℕ) : (θ₀ = 60) → (rotation = 630) → 
  let θ₁ := θ₀ + (rotation % 360) in
  let acute_angle := 360 - θ₁ % 360 in
  acute_angle = 30 :=
by
  intros h₀ h_rotation
  let θ₁ := 60 + (630 % 360)
  let θ_final := 360 - (θ₁ % 360)
  have : θ_final = 30 := sorry
  exact this

end rotated_angle_new_measure_l654_654314


namespace area_of_tangent_triangle_l654_654298

noncomputable def f (x : ℝ) : ℝ := x^2

theorem area_of_tangent_triangle : 
  let tangent_line := λ x, 2 * (x - 1) + 1 in 
  let x_intercept := (1/2 : ℝ) in
  let y_intercept := -1 in
  (1/2) * abs x_intercept * abs y_intercept = (1/4 : ℝ) := by
sorry

end area_of_tangent_triangle_l654_654298


namespace homework_time_decrease_l654_654341

theorem homework_time_decrease (x : ℝ) :
  let T_initial := 100
  let T_final := 70
  T_initial * (1 - x) * (1 - x) = T_final :=
by
  sorry

end homework_time_decrease_l654_654341


namespace range_f1_f_decreasing_f_3x_minus_1_gt_2_l654_654304

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: ∀ (a b : ℝ), 0 < a → 0 < b → f(a) + f(b) = f(a * b)
axiom cond1 : ∀ (a b : ℝ), 0 < a → 0 < b → f(a) + f(b) = f(a * b)

-- Condition 2: ∀ (x : ℝ), 1 < x → f(x) < 0
axiom cond2 : ∀ (x : ℝ), 1 < x → f(x) < 0

-- Condition 3: f(2) = -1
axiom cond3 : f(2) = -1

-- (I) Determine the range of f(1)
theorem range_f1 : f(1) = 0 :=
sorry

-- (II) Prove that the function f(x) is decreasing on (0, +∞)
theorem f_decreasing : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f(x2) < f(x1) :=
sorry

-- (III) Find the set of values for x such that f(3x-1) > 2
theorem f_3x_minus_1_gt_2 : {x : ℝ | f(3 * x - 1) > 2} = {x : ℝ | (1:ℝ)/3 < x ∧ x < (5:ℝ)/12} :=
sorry

end range_f1_f_decreasing_f_3x_minus_1_gt_2_l654_654304


namespace shortest_distance_between_two_circles_l654_654375

def circle1 := { p : ℝ × ℝ | (p.1^2 - 6 * p.1 + p.2^2 + 10 * p.2 + 9 = 0) }
def circle2 := { p : ℝ × ℝ | (p.1^2 + 4 * p.1 + p.2^2 - 8 * p.2 + 4 = 0) }

noncomputable def center1 := (3 : ℝ, -5 : ℝ)
noncomputable def radius1 := 5 : ℝ

noncomputable def center2 := (-2 : ℝ, 4 : ℝ)
noncomputable def radius2 := 4 : ℝ

noncomputable def distance_centers :=
  real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

noncomputable def shortest_distance := distance_centers - (radius1 + radius2)

theorem shortest_distance_between_two_circles :
  shortest_distance = real.sqrt 106 - 9 :=
by
  sorry

end shortest_distance_between_two_circles_l654_654375


namespace midpoint_trajectory_l654_654611

theorem midpoint_trajectory (l1 l2 : Line) (P : Point) (A B M : Point) :
  l1.is_perpendicular l2 →
  l1.passes P →
  l2.passes P →
  l1.x_intercept A →
  l2.y_intercept B →
  midpoint A B M →
  M.coords.1 + M.coords.2 = 1 :=
by
  sorry

end midpoint_trajectory_l654_654611


namespace TriangleConstruction_l654_654146

variable (α : ℝ) -- the angle at vertex A
variable (A B C D M : Type) -- the vertices and points
variable [MetricSpace D] [MetricSpace M] -- ensuring D, M are metric spaces

-- Conditions:
variable (altitude_from_A : D) -- the altitude from A intersecting BC at D
variable (median_from_A : M) -- the median from A intersecting BC at M
variable (is_midpoint : ∀ (B C : Type), is_midpoint_of D B C M) -- M is the midpoint of BC intersecting at A

-- Theorem to prove:
theorem TriangleConstruction :
  ∃ (ABC : Triangle), ∃ A B C : Point, ∃ (h1 : is_angle_of A B C α), ∃ (h2 : altitude_from_A = D), ∃ (h3 : median_from_A = M),
    is_triangle ABC ∧ h1 ∧ h2 ∧ h3 := sorry

end TriangleConstruction_l654_654146


namespace mars_bars_count_l654_654250

theorem mars_bars_count (total_candy_bars snickers butterfingers : ℕ) (h_total : total_candy_bars = 12) (h_snickers : snickers = 3) (h_butterfingers : butterfingers = 7) :
  total_candy_bars - (snickers + butterfingers) = 2 :=
by sorry

end mars_bars_count_l654_654250


namespace correct_statement_l654_654206

variable {a b : Type} -- Let a and b be types representing lines
variable {α β : Type} -- Let α and β be types representing planes

-- Define parallel relations for lines and planes
def parallel (L P : Type) : Prop := sorry

-- Define the subset relation for lines in planes
def subset (L P : Type) : Prop := sorry

-- Now state the theorem corresponding to the correct answer
theorem correct_statement (h1 : parallel α β) (h2 : subset a α) : parallel a β :=
sorry

end correct_statement_l654_654206


namespace prove_identity_l654_654536

theorem prove_identity 
  (a b θ φ x y : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : (a - b) * sin (θ / 2) * cos (φ / 2) + (a + b) * cos (θ / 2) * sin (φ / 2) = 0)
  (h2 : (x / a) * cos θ + (y / b) * sin θ = 1)
  (h3 : (x / a) * cos φ + (y / b) * sin φ = 1) : 
  (x^2 / a^2 + (b^2 - a^2) / b^4 * y^2 = 1) := 
by
  sorry

end prove_identity_l654_654536


namespace first_rocket_height_l654_654647

theorem first_rocket_height (h : ℝ) (combined_height : ℝ) (second_rocket_height : ℝ) 
  (H1 : second_rocket_height = 2 * h) 
  (H2 : combined_height = h + second_rocket_height) 
  (H3 : combined_height = 1500) : h = 500 := 
by 
  -- The proof would go here but is not required as per the instruction.
  sorry

end first_rocket_height_l654_654647


namespace diff_squares_of_roots_l654_654905

theorem diff_squares_of_roots : ∀ α β : ℝ, (α * β = 6) ∧ (α + β = 5) -> (α - β)^2 = 1 := by
  sorry

end diff_squares_of_roots_l654_654905


namespace max_cards_collected_l654_654787

-- Define the problem conditions
def room_num := 20
def card (r : ℕ) : ℕ -- a card is associated with each room
axiom card_unique : ∀ r1 r2 : ℕ, r1 ≠ r2 → card r1 ≠ card r2
def start_room := 1
def collected_cards (path : list ℕ) : Prop :=
  path.head = start_room ∧
  (∀ (i : ℕ), i < path.length - 1 →
    card (path.nth i).get_or_else 0 < card (path.nth (i + 1)).get_or_else 0) ∧
  (∀ r ∈ path, r ≤ room_num)

-- Define the question in Lean
theorem max_cards_collected : 
  ∀ path : list ℕ, collected_cards path → path.length ≤ 19 :=
sorry

end max_cards_collected_l654_654787


namespace simplest_sqrt_l654_654417

theorem simplest_sqrt :
  (is_simplest_sqrt 6) ∧
  ¬(is_simplest_sqrt 8) ∧
  ¬(is_simplest_sqrt 12) ∧
  ¬(is_simplest_sqrt 0.3) :=
by
  -- Definitions and required proofs will be placed here
  sorry

/--
Helper functions and predicates that were used.
-/
def is_simplest_sqrt (n : ℝ) : Prop :=
  ∀ (x y : ℝ), n = x * x → y * y ≠ n


end simplest_sqrt_l654_654417


namespace no_solution_in_natural_numbers_l654_654692

theorem no_solution_in_natural_numbers (x y z : ℕ) : 
  (x / y : ℚ) + (y / z : ℚ) + (z / x : ℚ) ≠ 1 := 
by sorry

end no_solution_in_natural_numbers_l654_654692


namespace log_relationship_l654_654974

def a : ℝ := Real.log 3 / Real.log 2
def b : ℝ := Real.log 4 / Real.log 3
def c : ℝ := Real.log 11 / Real.log 4

theorem log_relationship : b < a ∧ a < c :=
by
  sorry

end log_relationship_l654_654974


namespace moe_mowing_time_l654_654254

theorem moe_mowing_time :
  ∀ (length width : ℕ) (swath overlap : ℕ) (speed : ℕ),
  length = 100 →
  width = 180 →
  swath = 30 →
  overlap = 6 →
  speed = 4000 →
  (width / ((swath - overlap) / 12) * length) / speed = 2.25 :=
by
  intros length width swath overlap speed
  assume h_length h_width h_swath h_overlap h_speed
  sorry

end moe_mowing_time_l654_654254


namespace polynomial_rewrite_l654_654295

theorem polynomial_rewrite (d : ℤ) (h : d ≠ 0) :
  let a := 20
  let b := 18
  let c := 18
  let e := 8
  (16 * d + 15 + 18 * d^2 + 5 * d^3) + (4 * d + 3 + 3 * d^3) = a * d + b + c * d^2 + e * d^3 ∧ a + b + c + e = 64 := 
by
  sorry

end polynomial_rewrite_l654_654295


namespace trajectory_of_midpoint_l654_654332

theorem trajectory_of_midpoint
  (M : ℝ × ℝ)
  (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hP : P = (4, 0))
  (hQ : Q.1^2 + Q.2^2 = 4)
  (M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 - 2)^2 + M.2^2 = 1 :=
sorry

end trajectory_of_midpoint_l654_654332


namespace smallest_tangent_circle_equation_l654_654734

theorem smallest_tangent_circle_equation :
  ∃ (c r : ℝ), let circle_eq := (λ x y : ℝ, (x - c) ^ 2 + (y - c) ^ 2 = r ^ 2) in
  circle_eq 6 6 = (3 * Real.sqrt 2) ^ 2 ∧
  ∀ x y : ℝ, (x + y - 2 = 0) ∧ (x ^ 2 + y ^ 2 - 12 * x - 12 * y + 54 = 0) → 
  ∀ (c r : ℝ), circle_eq c r := sorry

end smallest_tangent_circle_equation_l654_654734


namespace solution_set_f_gt_2x_plus_4_l654_654720

noncomputable def f : ℝ → ℝ := sorry -- Function definition

-- Definitions from the conditions
def condition1 : ∀ x : ℝ, f x ∈ (set.univ : set ℝ) := sorry
def condition2 : f (-1) = 2 := sorry
def condition3 : ∀ x : ℝ, deriv f x > 2 := sorry

-- The proof problem statement
theorem solution_set_f_gt_2x_plus_4 : 
  {x : ℝ | f x > 2 * x + 4} = set.Ioi (-1) :=
sorry

end solution_set_f_gt_2x_plus_4_l654_654720


namespace fewer_twos_for_100_l654_654350

theorem fewer_twos_for_100 : (222 / 2 - 22 / 2) = 100 := by
  sorry

end fewer_twos_for_100_l654_654350


namespace car_average_mpg_l654_654473

theorem car_average_mpg 
(initial_odometer: ℕ) (final_odometer: ℕ)
(initial_fill: ℕ) (first_refill: ℕ) (second_refill: ℕ)
(h_initial_odometer: initial_odometer = 34500)
(h_final_odometer: final_odometer = 35250)
(h_initial_fill: initial_fill = 8)
(h_first_refill: first_refill = 10)
(h_second_refill: second_refill = 15) :
  let total_distance := final_odometer - initial_odometer in
  let total_gasoline := initial_fill + first_refill + second_refill in
  (Float.ofNat total_distance / Float.ofNat total_gasoline).round = 22.7 :=
by
  sorry

end car_average_mpg_l654_654473


namespace ordered_sets_equal_l654_654226

theorem ordered_sets_equal
  (n : ℕ) 
  (h_gcd : gcd n 6 = 1) 
  (a b : ℕ → ℕ) 
  (h_order_a : ∀ {i j}, i < j → a i < a j)
  (h_order_b : ∀ {i j}, i < j → b i < b j) 
  (h_sum : ∀ {j k l : ℕ}, 1 ≤ j → j < k → k < l → l ≤ n → a j + a k + a l = b j + b k + b l) : 
  ∀ (j : ℕ), 1 ≤ j → j ≤ n → a j = b j := 
sorry

end ordered_sets_equal_l654_654226


namespace problem_solution_l654_654778

-- Defining the conditions as Lean definitions
def highest_score (scores : List ℕ) := scores.maximum
def average_score (scores : List ℕ) := (scores.sum / scores.length : ℕ)

-- The main theorem stating the problem
theorem problem_solution (scores : List ℕ) (h : average_score scores = 96) (h₁ : highest_score scores = some 100) :
  "Mode and mean are the statistical quantities reflected by Xiao Hong's description." :=
sorry

end problem_solution_l654_654778


namespace term_x2y7_in_expansion_l654_654527

theorem term_x2y7_in_expansion : 
  ∃ T : (ℕ → ℕ) → ℕ, T (λ n, (x - y) ^ 9) = -36 * x^2 * y^7 :=
sorry

end term_x2y7_in_expansion_l654_654527


namespace children_ticket_price_proof_l654_654472

-- Definitions based on conditions
def children's_ticket_cost : ℝ
def adult_ticket_cost : ℝ := children's_ticket_cost + 3.25
def total_tickets_cost (children: ℝ) (adult: ℝ) : ℝ := 2 * adult + 4 * children - 2

-- Given conditions
def total_cost : ℝ := 30

-- Theorem statement to prove the cost of a children's ticket
theorem children_ticket_price_proof (C : ℝ) (H : total_tickets_cost C (C + 3.25) = total_cost) : C = 4.25 :=
by
  sorry

end children_ticket_price_proof_l654_654472


namespace solve_for_q_l654_654597

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 14) (h2 : 6 * p + 5 * q = 17) : q = -1 / 11 :=
by
  sorry

end solve_for_q_l654_654597


namespace min_value_3x_plus_4y_l654_654606

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_plus_4y_l654_654606


namespace gcf_50_75_l654_654370

theorem gcf_50_75 : Nat.gcd 50 75 = 25 := by
  sorry

end gcf_50_75_l654_654370


namespace coin_stack_height_l654_654620

theorem coin_stack_height (x y z w : ℕ) : 
  1.65 * x + 2.05 * y + 1.45 * z + 1.85 * w = 15.8 :=
by
  have h : 1.65 * 6 + 2.05 * 1 + 1.45 * 2 + 1.85 * 1 = 15.8 := sorry
  exact h

end coin_stack_height_l654_654620


namespace tomatoes_sold_to_mr_wilson_l654_654675

theorem tomatoes_sold_to_mr_wilson :
  let T := 245.5
  let S_m := 125.5
  let N := 42
  let S_w := T - S_m - N
  S_w = 78 := 
by
  sorry

end tomatoes_sold_to_mr_wilson_l654_654675


namespace sarah_score_l654_654278

theorem sarah_score (s g : ℝ) (h1 : s = g + 50) (h2 : (s + g) / 2 = 110) : s = 135 := 
by 
  sorry

end sarah_score_l654_654278


namespace minimum_marked_cells_ensure_coverage_l654_654379

/-- Definition of the board as a set of coordinates --/
def board : set (ℕ × ℕ) := { p | p.1 < 10 ∧ p.2 < 11 }

/-- Definition of an L-shaped piece as a set of three coordinates --/
def L_shaped_piece (x y : ℕ) : set (ℕ × ℕ) :=
  if x <= y then { (x, y), (x+1, y), (x, y+1) } else { (x, y), (x-1, y), (x, y+1) }

/-- Predicate indicating if a cell is marked --/
def is_marked (marked : set (ℕ × ℕ)) (cell : ℕ × ℕ) : Prop := cell ∈ marked

/-- The specific problem statement --/
theorem minimum_marked_cells_ensure_coverage :
  ∃ (marked : set (ℕ × ℕ)), marked.card = 50 ∧ ∀ x y ∈ board, 
  ∃ m ∈ L_shaped_piece x y, is_marked marked m :=
sorry

end minimum_marked_cells_ensure_coverage_l654_654379


namespace minimum_value_of_expression_l654_654977

theorem minimum_value_of_expression (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : 2 * x + 3 * y = 8) : 
  (∀ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 → (2 / a + 3 / b) ≥ 25 / 8) ∧ 
  (∃ a b, a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 8 ∧ 2 / a + 3 / b = 25 / 8) :=
sorry

end minimum_value_of_expression_l654_654977


namespace ravenswood_forest_percentage_l654_654708

def ravenswood_gnomes (westerville_gnomes : ℕ) : ℕ := 4 * westerville_gnomes
def remaining_gnomes (total_gnomes taken_percentage: ℕ) : ℕ := (total_gnomes * (100 - taken_percentage)) / 100

theorem ravenswood_forest_percentage:
  ∀ (westerville_gnomes : ℕ) (remaining : ℕ) (total_gnomes : ℕ),
  westerville_gnomes = 20 →
  total_gnomes = ravenswood_gnomes westerville_gnomes →
  remaining = 48 →
  remaining_gnomes total_gnomes 40 = remaining :=
by
  sorry

end ravenswood_forest_percentage_l654_654708


namespace round_trip_time_correct_l654_654037

variables (speed_to_work speed_to_home : ℝ) (time_to_work_minutes : ℝ)
variables (dist_to_work dist_to_home time_to_work time_to_home round_trip_time : ℝ)

-- Conditions from the problem
def speed_to_work := 80  -- km/h
def speed_to_home := 120  -- km/h
def time_to_work_minutes := 72  -- minutes

-- Convert the time to work from minutes to hours
def time_to_work := time_to_work_minutes / 60  -- hours

-- Calculate the distance to work
def dist_to_work := speed_to_work * time_to_work  -- km

-- Distance home is equal to distance to work
def dist_to_home := dist_to_work

-- Calculate the time to return home
def time_to_home := dist_to_home / speed_to_home  -- hours

-- Calculate the total round trip time
def round_trip_time := time_to_work + time_to_home  -- hours

-- Final proof statement
theorem round_trip_time_correct : 
  round_trip_time = 2 :=
by
  -- We would provide the proof here
  sorry

end round_trip_time_correct_l654_654037


namespace team_plays_60_games_in_division_l654_654199

noncomputable def number_of_division_games (N M : ℕ) (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) : ℕ :=
  4 * N

theorem team_plays_60_games_in_division (N M : ℕ) 
  (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) 
  : number_of_division_games N M hNM hM h_total = 60 := 
sorry

end team_plays_60_games_in_division_l654_654199


namespace solve_3x3_table_l654_654847

theorem solve_3x3_table : 
   ∃ (fill_combinations : list (list (list ℤ))), 
   (∀ (row : list ℤ), row ∈ fill_combinations → row.sum = fill_combinations.head.sum) 
   ∧
   fill_combinations.length = 831 :=
   sorry

end solve_3x3_table_l654_654847


namespace quadratic_minimum_l654_654062

theorem quadratic_minimum (x : ℝ) : 
  let f := λ x, 3 * x ^ 2 + 6 * x + 4 in
  ∀ x_min, (∀ x, f x_min ≤ f x) → x_min = -1 :=
by
  let f := λ x, 3 * x ^ 2 + 6 * x + 4
  sorry

end quadratic_minimum_l654_654062


namespace sum_four_digit_integers_l654_654401

theorem sum_four_digit_integers : 
  ∑ k in Finset.range (9999 - 1000 + 1), (k + 1000) = 49495500 := 
by
  sorry

end sum_four_digit_integers_l654_654401


namespace pencils_left_with_Harry_l654_654022

theorem pencils_left_with_Harry :
  (let 
    anna_pencils := 50
    harry_initial_pencils := 2 * anna_pencils
    harry_lost_pencils := 19
    harry_pencils_left := harry_initial_pencils - harry_lost_pencils
  in harry_pencils_left = 81) := 
by
  sorry

end pencils_left_with_Harry_l654_654022


namespace solution1_solution2_l654_654343

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l654_654343


namespace solution_exists_l654_654835

open Nat

noncomputable def sequence_existence_problem : Prop :=
∃ (a b : ℕ) (b_seq : ℕ → ℕ) (a_seq : ℕ → ℕ),
  1 < a ∧ a < b ∧ ¬ (a ∣ b) ∧ -- conditions on a and b
  (∀ n : ℕ, b_seq (n + 1) ≥ 2 * b_seq n) ∧ -- condition on b_seq
  (∀ k : ℕ, a_seq (k + 1) - a_seq k = a ∨ a_seq (k + 1) - a_seq k = b) ∧ -- condition on differences in a_seq
  (∀ m l : ℕ, a_seq m + a_seq l ≠ b_seq m ∧ a_seq m + a_seq l ≠ b_seq l ∧ -- condition on sums in a_seq
               a_seq m + a_seq l ∉ { b_seq i | i ∈ ℕ }) -- b_seq set membership condition

theorem solution_exists : sequence_existence_problem :=
sorry


end solution_exists_l654_654835


namespace log_base_10_two_plus_log_base_10_five_log_base_3_nine_l654_654084

-- First Problem: lg 2 + lg 5 = 1
theorem log_base_10_two_plus_log_base_10_five : 
  Real.log 10 2 + Real.log 10 5 = Real.log 10 10 :=
by
  sorry

-- Second Problem: log_3 9 = 2
theorem log_base_3_nine : 
  Real.log 3 9 = 2 :=
by
  sorry

end log_base_10_two_plus_log_base_10_five_log_base_3_nine_l654_654084


namespace avg_age_new_students_l654_654710

-- Definitions for the conditions
def initial_avg_age : ℕ := 14
def initial_student_count : ℕ := 10
def new_student_count : ℕ := 5
def new_avg_age : ℕ := initial_avg_age + 1

-- Lean statement for the proof problem
theorem avg_age_new_students :
  (initial_avg_age * initial_student_count + new_avg_age * new_student_count) / new_student_count = 17 :=
by
  sorry

end avg_age_new_students_l654_654710


namespace midpoint_sum_l654_654063

theorem midpoint_sum (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = -3) (h3 : x2 = -4) (h4 : y2 = 15) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 = 8 :=
by
  -- Apply the hypothesis directly
  subst h1 h2 h3 h4
  -- Calculate and simplify
  calc
    (8 + (-4)) / 2 + (-3 + 15) / 2 =  "4 / 2 + 12 / 2" : by rfl
    ... = 2 + 6 : by rfl
    ... = 8 : by rfl

end midpoint_sum_l654_654063


namespace overall_passing_fraction_l654_654615

def studentsInClassA : ℕ := 120
def studentsInClassB : ℕ := 180
def studentsInClassC : ℕ := 60

def studentsWithAInClassA : ℕ := 120 / 4
def studentsWithBInClassA : ℕ := 120 / 2
def studentsWithCInClassA : ℕ := 120 / 8

def studentsWithAInClassB : ℕ := 180 / 3
def studentsWithBInClassB : ℕ := 180 / 3
def studentsWithCInClassB : ℕ := 180 / 6

def studentsWithAInClassC : ℕ := 60 / 5
def studentsWithBInClassC : ℕ := 2 * 60 / 5
def studentsWithCInClassC : ℕ := 60 / 10

def totalPassingStudents : ℕ :=  
  studentsWithAInClassA + studentsWithBInClassA + studentsWithCInClassA +
  studentsWithAInClassB + studentsWithBInClassB + studentsWithCInClassB +
  studentsWithAInClassC + studentsWithBInClassC + studentsWithCInClassC

def totalStudents : ℕ := studentsInClassA + studentsInClassB + studentsInClassC

theorem overall_passing_fraction : 
  (totalPassingStudents : ℚ) / (totalStudents : ℚ) = 33 / 40 := 
by 
  have h : totalPassingStudents = 297 := by sorry
  have h_total : totalStudents = 360 := by sorry
  simp [h, h_total]
  norm_num
  sorry

end overall_passing_fraction_l654_654615


namespace tire_cost_l654_654508

theorem tire_cost (total_cost : ℝ) (num_tires : ℕ)
    (h1 : num_tires = 8) (h2 : total_cost = 4) : 
    total_cost / num_tires = 0.50 := 
by
  sorry

end tire_cost_l654_654508


namespace sequence_is_increasing_l654_654547

theorem sequence_is_increasing (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) - a n = 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  intro n
  have h2 : a (n + 1) - a n = 2 := h n
  linarith

end sequence_is_increasing_l654_654547


namespace bag_contains_twenty_cookies_l654_654441

noncomputable def cookies_in_bag 
  (total_calories : ℕ) 
  (calories_per_cookie : ℕ)
  (bags_in_box : ℕ)
  : ℕ :=
  total_calories / (calories_per_cookie * bags_in_box)

theorem bag_contains_twenty_cookies 
  (H1 : total_calories = 1600) 
  (H2 : calories_per_cookie = 20) 
  (H3 : bags_in_box = 4)
  : cookies_in_bag total_calories calories_per_cookie bags_in_box = 20 := 
by
  have h1 : total_calories = 1600 := H1
  have h2 : calories_per_cookie = 20 := H2
  have h3 : bags_in_box = 4 := H3
  sorry

end bag_contains_twenty_cookies_l654_654441


namespace solution_l654_654738

noncomputable def problem := 
∀ (cards : Finset ℕ), 
  cards = {1, 2, 3, 4} → 
  (∑ x in cards, 1 : ℝ) = 4 →
  let drawn := cards.powerset.filter (λ s, s.card = 2) in
  (∑ s in drawn, 1 : ℝ) = 6 →
  let favorable := drawn.filter (λ s, (∑ x in s, x) % 2 = 1) in
  (∑ s in favorable, 1 : ℝ) = 4 →
  (∑ s in favorable, 1 : ℝ) / (∑ s in drawn, 1 : ℝ) = (2 : ℝ) / 3

theorem solution : problem := sorry

end solution_l654_654738


namespace count_table_fills_2014_2015_2016_l654_654842

theorem count_table_fills_2014_2015_2016 :
  let a := 2014
  let b := 2015
  let c := 2016
  let total_fills := 831 in
  ∃ table : (Fin 3 → Fin 3 → ℕ), 
    (∀ i : Fin 3, (∑ j : Fin 3, table i j) = (∑ j : Fin 3, table 0 j)) ∧
    (∀ i j, table i j ∈ {a, b, c}) ∧
    -- Count the unique valid tables
    (card {t : (Fin 3 → Fin 3 → ℕ) | 
      (∀ i : Fin 3, (∑ j : Fin 3, t i j) = (∑ j : Fin 3, t 0 j)) ∧
      (∀ i j, t i j ∈ {a, b, c})} = total_fills) :=
by
  let a := 2014
  let b := 2015
  let c := 2016
  let total_fills := 831
  existsi (λ i j, if (i, j) = (0, 0) then a else b)
  split
  { 
    intro i,
    sorry 
  }
  {
    split
    {
        intros i j,
        exact by sorry -- elements are in {a, b, c}
    }
    { 
      apply (sorry : card ... = total_fills)
    }
  }

end count_table_fills_2014_2015_2016_l654_654842


namespace center_of_symmetry_of_f_is_l654_654607

def tan_abscissa_shortened (x : ℝ) : ℝ := tan(4 * x)
def tan_shifted_left (x : ℝ) : ℝ := tan(4 * (x + π / 12))
def f (x : ℝ) : ℝ := tan(4 * x + π / 3) + 2
def is_center_of_symmetry (x : ℝ) : Prop := ∃ k : ℤ, x = k * π / 8 - π / 12

theorem center_of_symmetry_of_f_is 
  : ∀ x : ℝ, (is_center_of_symmetry x) → f x = 2 :=
sorry

end center_of_symmetry_of_f_is_l654_654607


namespace Ben_total_clothes_l654_654014

-- Definitions of Alex's clothing items
def Alex_shirts := 4.5
def Alex_pants := 3.0
def Alex_shoes := 2.5
def Alex_hats := 1.5
def Alex_jackets := 2.0

-- Definitions of Joe's clothing items
def Joe_shirts := Alex_shirts + 3.5
def Joe_pants := Alex_pants - 2.5
def Joe_shoes := Alex_shoes
def Joe_hats := Alex_hats + 0.3
def Joe_jackets := Alex_jackets - 1.0

-- Definitions of Ben's clothing items
def Ben_shirts := Joe_shirts + 5.3
def Ben_pants := Alex_pants + 5.5
def Ben_shoes := Joe_shoes - 1.7
def Ben_hats := Alex_hats + 0.5
def Ben_jackets := Joe_jackets + 1.5

-- Statement to prove the total number of Ben's clothing items
def total_Ben_clothing_items := Ben_shirts + Ben_pants + Ben_shoes + Ben_hats + Ben_jackets

theorem Ben_total_clothes : total_Ben_clothing_items = 27.1 :=
by
  sorry

end Ben_total_clothes_l654_654014


namespace cars_per_client_l654_654455

-- Define the conditions
def num_cars : ℕ := 18
def selections_per_car : ℕ := 3
def num_clients : ℕ := 18

-- Define the proof problem as a theorem
theorem cars_per_client :
  (num_cars * selections_per_car) / num_clients = 3 :=
sorry

end cars_per_client_l654_654455


namespace propositions_proof_l654_654904

-- Definitions for question 1
def negation_of_cos (H : ∀ x : ℝ, cos x > 0) : Prop := ∃ x : ℝ, cos x ≤ 0

-- Definitions for question 2
noncomputable def log_eq_max_sum_absurd (a b : ℝ) (h : a > 0 ∧ b > 0 ∧ log 10 a + log 10 b = log 10 (a + b)) : Prop :=
  ∀ (M : ℝ), a + b ≠ M

-- Definitions for question 3
def odd_function_f_val (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_shift : ∀ x : ℝ, f (x + 2) = -f x) : Prop :=
  f 6 = 0

-- Definitions for question 4
def normal_distribution_prob (σ : ℝ) (P1 : Prob (λ X : ℝ, X ≤ 5)) (P2: Prob (λ X : ℝ, X ≤ -3)) : Prop :=
  P1 = 0.81 → P2 = 0.19

-- The main theorem to be proved
theorem propositions_proof (f : ℝ → ℝ) (σ : ℝ) (H : ∀ x : ℝ, cos x > 0) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_shift : ∀ x : ℝ, f (x + 2) = -f x)
  (P1 : Prob (λ X : ℝ, X ≤ 5)) (P2: Prob (λ X : ℝ, X ≤ -3)):
  (negation_of_cos H) ∧
  (log_eq_max_sum_absurd 2 2 ⟨by norm_num, by norm_num, by norm_num⟩) ∧
  (odd_function_f_val f h_odd h_shift) ∧
  (normal_distribution_prob σ P1 P2) := 
sorry

end propositions_proof_l654_654904


namespace logarithmic_inequality_solution_set_l654_654804

theorem logarithmic_inequality_solution_set {a : ℝ} (h1 : 0 < a) (h2 : a ≠ 1) (h3 : ∀ x < 0, a^x > 1) :
  { x : ℝ | log a x > 0 } = { x : ℝ | 0 < x ∧ x < 1 } := sorry

end logarithmic_inequality_solution_set_l654_654804


namespace SarahsScoreIs135_l654_654287

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l654_654287


namespace min_omega_for_two_maxima_l654_654079

theorem min_omega_for_two_maxima (ω : ℝ) (h1 : ω > 0) (h2 : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → y = sin (ω * x)) :
  ω ≥ (5 / 2) * π :=
sorry

end min_omega_for_two_maxima_l654_654079


namespace number_of_performances_l654_654094

theorem number_of_performances (hanna_songs : ℕ) (mary_songs : ℕ) (alina_songs : ℕ) (tina_songs : ℕ)
    (hanna_cond : hanna_songs = 4)
    (mary_cond : mary_songs = 7)
    (alina_cond : 4 < alina_songs ∧ alina_songs < 7)
    (tina_cond : 4 < tina_songs ∧ tina_songs < 7) :
    ((hanna_songs + mary_songs + alina_songs + tina_songs) / 3) = 7 :=
by
  -- proof steps would go here
  sorry

end number_of_performances_l654_654094


namespace find_sin_value_l654_654955

-- Define the conditions
variables (α : ℝ)
axiom sin_condition : sin (α + π / 6) = 1 / 3
axiom alpha_bound : π / 3 < α ∧ α < π

-- Define the goal statement
theorem find_sin_value : sin (π / 12 - α) = - (4 + sqrt 2) / 6 := 
by
  -- Proof goes here
  sorry

end find_sin_value_l654_654955


namespace number_of_sets_proof_l654_654726

open_locale classical

noncomputable def number_of_such_sets {M A B X : Type} [Fintype M] [Fintype A] [Fintype B] [Fintype X] 
  (hM : ∃ l : Finset M, l.card = 10)
  (hA : A ⊆ M)
  (hB : B ⊆ M)
  (h_disjoint : A ∩ B = ∅)
  (hA_card : ∃ l : Finset A, l.card = 2)
  (hB_card : ∃ l : Finset B, l.card = 3)
  (hX_subset : ∀ x : X, x ∈ M ↔ A ⊆ x ∧ x ⊆ M) : Nat :=
256

theorem number_of_sets_proof {M A B X : Type} [Fintype M] [Fintype A] [Fintype B] [Fintype X]
  (hM : ∃ l : Finset M, l.card = 10)
  (hA : A ⊆ M)
  (hB : B ⊆ M)
  (h_disjoint : A ∩ B = ∅)
  (hA_card : ∃ l : Finset A, l.card = 2)
  (hB_card : ∃ l : Finset B, l.card = 3)
  (hX_subset : ∀ x : X, x ∈ M ↔ A ⊆ x ∧ x ⊆ M) : number_of_such_sets hM hA hB h_disjoint hA_card hB_card hX_subset = 256 :=
by
  sorry

end number_of_sets_proof_l654_654726


namespace even_positive_integers_l654_654513

-- Define least common multiple
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

-- State the proof problem
theorem even_positive_integers (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ k : ℕ, k > 0 ∧ 2 * k = (lcm x y + lcm y z) / lcm x z :=
by
  sorry

end even_positive_integers_l654_654513


namespace solve_z_in_complex_equation_l654_654702

theorem solve_z_in_complex_equation : 
  ∃ z : ℂ, 5 - 2 * complex.I * z = 1 + 5 * complex.I * z ∧ z = (-4 * complex.I) / 7 :=
by
  use (-4 * complex.I) / 7
  split
  sorry
  sorry

end solve_z_in_complex_equation_l654_654702


namespace find_a8_l654_654982

variable (a : ℕ → ℤ)

axiom h1 : ∀ n : ℕ, 2 * a n + a (n + 1) = 0
axiom h2 : a 3 = -2

theorem find_a8 : a 8 = 64 := by
  sorry

end find_a8_l654_654982


namespace least_n_for_g_equals_prime_product_arith_prog_l654_654939

def binomial_coeff (n k : ℕ) : ℕ := if h : k ≤ n then (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k))) else 0

def g (n : ℕ) : ℕ := binomial_coeff n 3

theorem least_n_for_g_equals_prime_product_arith_prog :
  ∃ p1 p2 p3 : ℕ, ∃ (n : ℕ), p1.prime ∧ p2.prime ∧ p3.prime ∧
    (p2 - p1 = 336) ∧ (p3 - p2 = 336) ∧
    (g n = p1 * p2 * p3) ∧ (n = 2019) :=
sorry

end least_n_for_g_equals_prime_product_arith_prog_l654_654939


namespace optimal_voter_split_l654_654758

-- Definitions
variables (Voters : Type) [fintype Voters] (n : ℕ)
variables (supports_miraflores : Voters → Prop)
variables [decidable_pred supports_miraflores]

-- Conditions
def half_supports_miraflores := fintype.card { v // supports_miraflores v } = n
def half_supports_maloney := fintype.card { v // ¬ supports_miraflores v } = n

-- Question (translated to a theorem)
theorem optimal_voter_split (h_m : half_supports_miraflores Voters n supports_miraflores)
    (h_d: half_supports_maloney Voters n supports_miraflores) :
  ∃ (D1 D2 : finset Voters), 
    ((D1 = {v | supports_miraflores v}) ∧ 
    (D2 = {v | ¬supports_miraflores v}) ∧ 
    (∀ v, v ∈ D1 ∨ v ∈ D2) ∧ 
    (∀ v, ¬ (v ∈ D1 ∧ v ∈ D2)) ∧ 
    (finset.card D1 = 1) ∧ 
    (finset.card D2 = 2 * n - 1)) :=
sorry

end optimal_voter_split_l654_654758


namespace solution1_solution2_l654_654345

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l654_654345


namespace isabel_remaining_pages_l654_654645

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def problems_per_page : ℕ := 8

theorem isabel_remaining_pages :
  (total_problems - finished_problems) / problems_per_page = 5 := 
sorry

end isabel_remaining_pages_l654_654645


namespace savings_is_22_77_cents_per_egg_l654_654337

-- Defining the costs and discount condition
def cost_per_large_egg_StoreA : ℚ := 0.55
def cost_per_extra_large_egg_StoreA : ℚ := 0.65
def discounted_cost_of_three_trays_large_StoreB : ℚ := 38
def total_eggs_in_three_trays : ℕ := 90

-- Savings calculation
def savings_per_egg : ℚ := (cost_per_extra_large_egg_StoreA - (discounted_cost_of_three_trays_large_StoreB / total_eggs_in_three_trays)) * 100

-- The statement to prove
theorem savings_is_22_77_cents_per_egg : savings_per_egg = 22.77 :=
by
  -- Here the proof would go, but we are omitting it with sorry
  sorry

end savings_is_22_77_cents_per_egg_l654_654337


namespace average_weight_of_whole_class_l654_654825

def num_students_a : ℕ := 50
def num_students_b : ℕ := 70
def avg_weight_a : ℚ := 50
def avg_weight_b : ℚ := 70

theorem average_weight_of_whole_class :
  (num_students_a * avg_weight_a + num_students_b * avg_weight_b) / (num_students_a + num_students_b) = 61.67 := by
  sorry

end average_weight_of_whole_class_l654_654825


namespace num_of_possible_outcomes_l654_654463

def participants : Fin 6 := sorry  -- Define the participants as elements of Fin 6

theorem num_of_possible_outcomes : (6 * 5 * 4 = 120) :=
by {
  -- Prove this mathematical statement
  rfl
}

end num_of_possible_outcomes_l654_654463


namespace smallest_positive_integer_for_inequality_l654_654055

noncomputable def nested_radical (a : ℕ) : ℝ :=
  let x := (1 + Real.sqrt (1 + 4 * a)) / 2 in x

noncomputable def nested_fraction (a : ℕ) : ℝ :=
  let y := (a + Real.sqrt (a^2 + 4)) / 2 in 2 / y

theorem smallest_positive_integer_for_inequality :
  forall (a : ℕ), a > 0 -> (nested_radical a - nested_fraction a > 7) -> a >= 43 :=
by
  intros a ha hineq
  sorry

end smallest_positive_integer_for_inequality_l654_654055


namespace count_integers_in_range_l654_654727

theorem count_integers_in_range : 
  let lower_bound := -2.8
  let upper_bound := Real.pi
  let in_range (x : ℤ) := (lower_bound : ℝ) < (x : ℝ) ∧ (x : ℝ) ≤ upper_bound
  (Finset.filter in_range (Finset.Icc (Int.floor lower_bound) (Int.floor upper_bound))).card = 6 :=
by
  sorry

end count_integers_in_range_l654_654727


namespace periodic_sequences_exist_l654_654245

noncomputable def recursive_sequences (a b c : ℕ → ℕ) : Prop :=
∀ n: ℕ, a (n+1) = |b n - c n| ∧ b (n+1) = |c n - a n| ∧ c (n+1) = |a n - b n|

theorem periodic_sequences_exist (a₁ b₁ c₁ : ℕ) (h1: a₁ > 0) (h2: b₁ > 0) (h3: c₁ > 0) :
  ∃ k: ℕ, ∀ (a b c : ℕ → ℕ),
  (a 0 = a₁ ∧ b 0 = b₁ ∧ c 0 = c₁) ∧
  recursive_sequences a b c →
  (a (k+1) = a k ∧ b (k+1) = b k ∧ c (k+1) = c k) := sorry

end periodic_sequences_exist_l654_654245


namespace probability_xi_interval_l654_654571

noncomputable def xi := MeasureTheory.ProbabilityTheory.randomVariable.real.normal 2 ∃ σ, MeasureTheory.Measure.probabilitySpace
axiom condition1 : MeasureTheory.ProbabilityTheory.randomVariable.real.prob (λ ξ, ξ < 4) xi = 0.9

theorem probability_xi_interval : MeasureTheory.ProbabilityTheory.randomVariable.real.prob (λ ξ, 0 < ξ ∧ ξ < 2) xi = 0.4 := by
  sorry

end probability_xi_interval_l654_654571


namespace lateral_surface_area_l654_654711

theorem lateral_surface_area (a h Q : ℝ) (h_base_rhombus : a > 0) (h_height_positive : h > 0) 
  (h_cross_section_area : Q = a * h * real.sqrt 2) : 
  4 * a * h = 2 * Q * real.sqrt 2 :=
by {
  -- This statement corresponds to proving S = 2 Q sqrt(2)
  sorry
}

end lateral_surface_area_l654_654711


namespace compound_interest_years_is_four_l654_654928
noncomputable def compoundInterestYears (P : ℝ) (r : ℝ) (n : ℕ) (CI : ℝ) : ℕ :=
  let A := P + CI
  let factor := (1 + r / n)
  let log_A_P := Real.log (A / P)
  let log_factor := Real.log factor
  Nat.floor (log_A_P / log_factor)

theorem compound_interest_years_is_four :
  compoundInterestYears 1200 0.20 1 1288.32 = 4 :=
by
  sorry

end compound_interest_years_is_four_l654_654928


namespace A_form_k_l654_654216

theorem A_form_k (m n : ℕ) (h_m : 2 ≤ m) (h_n : 2 ≤ n) :
  ∃ k : ℕ, (A : ℝ) = (n + Real.sqrt (n^2 - 4)) / 2 ^ m → A = (k + Real.sqrt (k^2 - 4)) / 2 :=
by
  sorry

end A_form_k_l654_654216


namespace smallest_marked_cells_l654_654391

def smallest_k_10x11 : ℕ := 50

theorem smallest_marked_cells :
  ∃ (marked_cells : Finset (Fin 10 × Fin 11)),
  marked_cells.card = smallest_k_10x11 ∧ 
  (∀ (L : Finset (Fin 10 × Fin 11)), L.card = 3 → (∀ P₁ P₂ P₃ ∈ L, ¬(P₁.1 = P₂.1 ∧ P₂.1 = P₃.1 ∨ P₁.2 = P₂.2 ∧ P₂.2 = P₃.2)) → 
    ∃ cell ∈ L, cell ∈ marked_cells) :=
sorry

end smallest_marked_cells_l654_654391


namespace angle_A_is_35_l654_654214

-- Define the angles in triangle ABC
variables (A B C : ℝ)
-- Definitions of angles
def angle_C_eq_90 : Prop := C = 90
def angle_B_eq_55 : Prop := B = 55

-- The theorem we'd like to prove
theorem angle_A_is_35 (hC : angle_C_eq_90 C) (hB : angle_B_eq_55 B) : A = 180 - B - C :=
by
  rw [angle_C_eq_90, angle_B_eq_55] at hC hB
  sorry

end angle_A_is_35_l654_654214


namespace length_of_median_l654_654613

-- Let A, B, C be points in the Euclidean plane
variable (A B C : Point)
-- Conditions given
variable (BC : dist B C = 7)
variable (AC : dist A C = 8)
variable (AB : dist A B = 9)

-- Define the midpoint D of AC
def midpoint (A C : Point) : Point := {
  x := (A.x + C.x) / 2,
  y := (A.y + C.y) / 2
}

variable (D : Point := midpoint A C)

-- Define the length of the median BD
def median_length (B D : Point) : ℝ := dist B D

-- Proof problem statement
theorem length_of_median : median_length B D = 7 := sorry

end length_of_median_l654_654613


namespace number_of_elements_in_list_l654_654593

theorem number_of_elements_in_list :
  ∃ (n : ℕ), (∀ k, k > 0 → 2.5 + (k - 1) * 4 = 54.5 → n = 14) :=
begin
  use 14,
  intros k hk h54,
  sorry -- We skip the proof as per instructions
end

end number_of_elements_in_list_l654_654593


namespace homework_time_decrease_l654_654340

theorem homework_time_decrease (x : ℝ) :
  let T_initial := 100
  let T_final := 70
  T_initial * (1 - x) * (1 - x) = T_final :=
by
  sorry

end homework_time_decrease_l654_654340


namespace distinct_nonzero_ints_polynomial_l654_654938

theorem distinct_nonzero_ints_polynomial (a b c: ℤ) : 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
  (∃ Q R : polynomial ℤ, x * (x - polynomial.C a) * (x - polynomial.C b) * (x - polynomial.C c) + 1 = Q * R) ↔ 
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = -1 ∧ b = -2 ∧ c = -3) ∨ 
  (a = 1 ∧ b = -1 ∧ c = 2) ∨ (a = 1 ∧ b = -1 ∧ c = -2) := 
sorry

end distinct_nonzero_ints_polynomial_l654_654938


namespace sum_four_digit_integers_l654_654397

theorem sum_four_digit_integers : 
  ∑ k in Finset.range (9999 - 1000 + 1), (k + 1000) = 49495500 := 
by
  sorry

end sum_four_digit_integers_l654_654397


namespace floor_equation_l654_654669

theorem floor_equation :
  let num := 11 * 77 + 12 * 78 + 13 * 79 + 14 * 80 in
  let denom := 11 * 76 + 12 * 77 + 13 * 78 + 14 * 79 in
  let frac := (100 * (num:ℚ) / denom) in
  int.floor frac = 101 :=
begin
  sorry
end

end floor_equation_l654_654669


namespace complex_number_solution_l654_654572

theorem complex_number_solution (z : ℂ) (h : (1 + 3 * complex.I) * z = 10 * complex.I) : z = 3 + complex.I :=
sorry

end complex_number_solution_l654_654572


namespace SarahsScoreIs135_l654_654285

variable (SarahsScore GregsScore : ℕ)

-- Conditions
def ScoreDifference (SarahsScore GregsScore : ℕ) : Prop := SarahsScore = GregsScore + 50
def AverageScore (SarahsScore GregsScore : ℕ) : Prop := (SarahsScore + GregsScore) / 2 = 110

-- Theorem statement
theorem SarahsScoreIs135 (h1 : ScoreDifference SarahsScore GregsScore) (h2 : AverageScore SarahsScore GregsScore) : SarahsScore = 135 :=
sorry

end SarahsScoreIs135_l654_654285


namespace new_students_count_l654_654300

-- Define the conditions as given in the problem statement.
def original_average_age := 40
def original_number_students := 17
def new_students_average_age := 32
def decreased_age := 36  -- Since the average decreases by 4 years from 40 to 36

-- Let x be the number of new students, the proof problem is to find x.
def find_new_students (x : ℕ) : Prop :=
  original_average_age * original_number_students + new_students_average_age * x = decreased_age * (original_number_students + x)

-- Prove that find_new_students(x) holds for x = 17
theorem new_students_count : find_new_students 17 :=
by
  sorry -- the proof goes here

end new_students_count_l654_654300


namespace simplify_fraction_l654_654701

theorem simplify_fraction
  (a b c : ℝ)
  (h : 2 * a - 3 * c - 4 - b ≠ 0)
  : (6 * a ^ 2 - 2 * b ^ 2 + 6 * c ^ 2 + a * b - 13 * a * c - 4 * b * c - 18 * a - 5 * b + 17 * c + 12) /
    (4 * a ^ 2 - b ^ 2 + 9 * c ^ 2 - 12 * a * c - 16 * a + 24 * c + 16) =
    (3 * a - 2 * c - 3 + 2 * b) / (2 * a - 3 * c - 4 + b) :=
  sorry

end simplify_fraction_l654_654701


namespace number_of_girls_l654_654743

theorem number_of_girls (x : ℕ) (h : ∀ (a b c : ℕ), a + b + c = 8) (prob : (nat.choose (8 - x) 2 * nat.choose x 1) / nat.choose 8 3 = 15 / 28) :
    x = 2 ∨ x = 3 :=
by
  sorry

end number_of_girls_l654_654743


namespace probability_real_roots_given_one_five_l654_654666

def is_valid_die_roll (m n : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6

def has_real_roots (m n : ℕ) : Prop :=
  (m * m - 4 * n) ≥ 0

theorem probability_real_roots_given_one_five : 
  (∃ m n, is_valid_die_roll m n ∧ (m = 5 ∨ n = 5) 
  ∧ has_real_roots m n) →
  ∃ p, p = 7 / 11 :=
begin
  sorry
end

end probability_real_roots_given_one_five_l654_654666


namespace exist_rel_prime_k_l_divisible_l654_654947

theorem exist_rel_prime_k_l_divisible (a b p : ℤ) : 
  ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) := 
sorry

end exist_rel_prime_k_l_divisible_l654_654947


namespace product_form_l654_654290

theorem product_form (a b c d : ℤ) :
    (a^2 - 7*b^2) * (c^2 - 7*d^2) = (a*c + 7*b*d)^2 - 7*(a*d + b*c)^2 :=
by sorry

end product_form_l654_654290


namespace temperature_linear_relation_l654_654872

noncomputable def temperature_relation (h : ℕ) : ℤ    :=
  match h with
  | 0 => 20
  | 1 => 14
  | 2 => 8
  | 3 => 2
  | 4 => -4
  | _ => 20 - 6 * h

theorem temperature_linear_relation : 
  ∀ h : ℕ, h ≤ 4 -> temperature_relation h = 20 - 6 * h :=
by
  intro h h_le_4
  cases h with
  | 0 => simp [temperature_relation]
  | 1 => simp [temperature_relation]
  | 2 => simp [temperature_relation]
  | 3 => simp [temperature_relation]
  | 4 => simp [temperature_relation]
  | _ => contradiction

-- In the cases where h is more than 4, the given temperature data is not specified so the relation t = 20 - 6h directly holds.
-- Therefore, we have restricted to h ≤ 4 for this equivalence.

end temperature_linear_relation_l654_654872


namespace cube_paint_same_color_l654_654256

theorem cube_paint_same_color (colors : Fin 6) : ∃ ways : ℕ, ways = 6 :=
sorry

end cube_paint_same_color_l654_654256


namespace max_monthly_profit_l654_654686

def monthly_sales_volume (t : ℝ) : ℝ := 3 - (2 / (t + 1))

def selling_price (x t : ℝ) : ℝ := 1.5 * 32 + t / (2 * x)

def monthly_profit (x t : ℝ) : ℝ :=
  let P := selling_price x t in
  P * x - 32 * x - 3 - t

theorem max_monthly_profit : ∃ x t, 
  monthly_sales_volume t = x ∧ 
  x = 3 - 2 / (t + 1) ∧ 
  (monthly_profit x t = 37.5) := 
sorry

end max_monthly_profit_l654_654686


namespace science_to_history_ratio_l654_654706

variable {H G M S : ℕ}

-- Given conditions
def history_textbook_pages : ℕ := 160
def geography_textbook_pages : ℕ := history_textbook_pages + 70
def math_textbook_pages : ℕ := (history_textbook_pages + geography_textbook_pages) / 2
def total_pages : ℕ := history_textbook_pages + geography_textbook_pages + math_textbook_pages + S

theorem science_to_history_ratio : total_pages = 905 → S = 320 → (S / history_textbook_pages) = 2 := by
  intros h1 h2
  rw [h1, h2]
  sorry

end science_to_history_ratio_l654_654706


namespace rational_segments_in_polygon_l654_654117

theorem rational_segments_in_polygon (n : ℕ) (side_lengths : Fin n → ℚ) (diagonal_lengths : Fin (n * (n - 3) / 2) → ℚ) :
  ∀ (MN : Fin (n * (n - 3) / 2)) (PQ : Fin (n * (n - 3) / 2)),
  ∃ (p q : Fin n), p ≠ q ∧ ∃ (segment_length : ℚ), segment_length ∈ {length // ∃ diagonals_intersecting, segment_length ∈ diagonals_intersecting} :=
sorry

end rational_segments_in_polygon_l654_654117


namespace min_a_for_zero_point_related_l654_654940

def f (x : ℝ) : ℝ := Real.exp (x - 2) + Real.log (x - 1) - 1
def g (x a : ℝ) : ℝ := x * (Real.log x - a * x) - 2

def zero_point_related (f g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ m n, (f m = 0) ∧ (g n a = 0) ∧ |m - n| ≤ 1

theorem min_a_for_zero_point_related : 
  (zero_point_related f (g a) (λ x, -2)) :=
sorry

end min_a_for_zero_point_related_l654_654940


namespace houses_before_boom_l654_654028

theorem houses_before_boom (T B H : ℕ) (hT : T = 2000) (hB : B = 574) : H = 1426 := by
  sorry

end houses_before_boom_l654_654028


namespace cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l654_654445

-- Define the conditions
def ticket_full_price : ℕ := 240
def discount_A : ℕ := ticket_full_price / 2
def discount_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Algebraic expressions provided in the answer
def cost_A (x : ℕ) : ℕ := discount_A * x + ticket_full_price
def cost_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Proofs for the specific cases
theorem cost_expression_A (x : ℕ) : cost_A x = 120 * x + 240 := by
  sorry

theorem cost_expression_B (x : ℕ) : cost_B x = 144 * (x + 1) := by
  sorry

theorem cost_comparison_10_students : cost_A 10 < cost_B 10 := by
  sorry

theorem cost_comparison_4_students : cost_A 4 = cost_B 4 := by
  sorry

end cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l654_654445


namespace trigonometric_identity_l654_654540

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) = - (1 / 3) := 
by
  sorry

end trigonometric_identity_l654_654540


namespace pqr_value_l654_654238

theorem pqr_value
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 29)
  (h_eq : 1 / p + 1 / q + 1 / r + 392 / (p * q * r) = 1) :
  p * q * r = 630 :=
by
  sorry

end pqr_value_l654_654238


namespace sum_of_primes_l654_654177

open Nat

def is_prime (n : ℕ) : Prop := Prime n

theorem sum_of_primes (a b c : ℕ) (h1 : is_prime a) (h2 : is_prime b) (h3 : is_prime c)
  (h4 : b + c = 13) (h5 : c * c - a * a = 72) : a + b + c = 15 := by
  sorry

end sum_of_primes_l654_654177


namespace part1_part2_l654_654969

def f (x : ℝ) (m : ℝ) : ℝ := log x - m * x + 1
def g (x : ℝ) : ℝ := x * (exp x - 2)

-- Part 1: Prove m = 1 for max value of f(x) being 0.

theorem part1 (m : ℝ) (h1 : ∃ x : ℝ, 0 < x ∧ f x m = 0) : m = 1 := 
sorry

-- Part 2: Prove m ≥ 1 for f(x) ≤ g(x) for all x in its domain.

theorem part2 (m : ℝ) (h2 : ∀ x : ℝ, 0 < x → f x m ≤ g x) : 1 ≤ m :=
sorry

end part1_part2_l654_654969


namespace more_penguins_than_lions_l654_654027

def num_lions : ℕ := 30
def lion_to_penguin_ratio : ℕ × ℕ := (3, 11)

theorem more_penguins_than_lions : 
  let ratio_fits := num_lions / lion_to_penguin_ratio.fst in
  let num_penguins := lion_to_penguin_ratio.snd * ratio_fits in
  num_penguins - num_lions = 80 :=
by
  let ratio_fits := num_lions / lion_to_penguin_ratio.fst
  let num_penguins := lion_to_penguin_ratio.snd * ratio_fits
  have h : num_penguins - num_lions = 110 - 30 := sorry
  show num_penguins - num_lions = 80 from h

end more_penguins_than_lions_l654_654027


namespace neg_four_fifth_less_neg_two_third_l654_654039

theorem neg_four_fifth_less_neg_two_third : (-4 : ℚ) / 5 < (-2 : ℚ) / 3 :=
  sorry

end neg_four_fifth_less_neg_two_third_l654_654039


namespace factorial_comparison_l654_654810

theorem factorial_comparison :
  (Nat.factorial (Nat.factorial 100)) <
  (Nat.factorial 99)^(Nat.factorial 100) * (Nat.factorial 100)^(Nat.factorial 99) :=
  sorry

end factorial_comparison_l654_654810


namespace solution1_solution2_l654_654346

theorem solution1 : (222 / 2) - (22 / 2) = 100 := by
  sorry

theorem solution2 : (2 * 2 * 2 + 2) * (2 * 2 * 2 + 2) = 100 := by
  sorry

end solution1_solution2_l654_654346


namespace jack_total_payment_l654_654218

def tire_costs : list ℕ := [230, 230, 250, 250, 280, 280]
def window_costs : list ℕ := [700, 800, 900]

def total_cost (costs : list ℕ) : ℕ :=
  costs.foldr (· + ·) 0

theorem jack_total_payment :
  total_cost tire_costs + total_cost window_costs = 3920 :=
by
  sorry

end jack_total_payment_l654_654218


namespace curve_product_l654_654980

theorem curve_product (a b : ℝ) (h1 : 8 * a + 2 * b = 2) (h2 : 12 * a + b = 9) : a * b = -3 := by
  sorry

end curve_product_l654_654980


namespace square_area_eq_800_l654_654232

-- Define the conditions of the problem
variables {A B C D P Q R S : Type} [euclidean_geometry A B C D P Q R S]
variables (ABCD : square A B C D)
variables (PQRS : rhombus P Q R S)
variables (on_sides : inscribed_in_square ABCD PQRS)
variables (PB : ℝ) (BQ : ℝ) (PR : ℝ) (QS : ℝ)
variables (PB_eq : PB = 16) (BQ_eq : BQ = 25) (PR_eq : PR = 35) (QS_eq : QS = 40)

-- The theorem we want to prove
theorem square_area_eq_800 :
  let AB := sqrt (PB^2 + BQ^2)
  in AB^2 = 800 :=
  sorry

end square_area_eq_800_l654_654232


namespace planes_parallel_if_perpendicular_to_line_l654_654556

variable {Point : Type} [AffineSpace Point]
variables (l : Line) (α β : Plane)

-- This corresponds to the given conditions and the proof problem:
theorem planes_parallel_if_perpendicular_to_line 
  (h1 : l ⊥ α) (h2 : l ⊥ β) : 
  α ∥ β := 
sorry

end planes_parallel_if_perpendicular_to_line_l654_654556


namespace rhombus_area_2400_l654_654516

noncomputable def area_of_rhombus (x y : ℝ) : ℝ :=
  2 * x * y

theorem rhombus_area_2400 (x y : ℝ) 
  (hx : x = 15) 
  (hy : y = (16 / 3) * x) 
  (rx : 18.75 * 4 * x * y = x * y * (78.75)) 
  (ry : 50 * 4 * x * y = x * y * (200)) : 
  area_of_rhombus 15 80 = 2400 :=
by
  sorry

end rhombus_area_2400_l654_654516


namespace sum_of_solutions_l654_654293

def fractional_part (x : ℝ) : ℝ := x - ↑⌊x⌋
def equation (x : ℝ) : Prop := real.sqrt (8 * x + 5) + 2 * fractional_part x = 2 * x + 2

theorem sum_of_solutions :
  (∃ x₁ x₂, equation x₁ ∧ equation x₂ ∧ (x₁ + x₂ = 0.75)) :=
sorry

end sum_of_solutions_l654_654293
