import MathLib.Data.Real.Basic
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Finprod
import Mathlib.Algebra.EquationSolving
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialGameTheory
import Mathlib.Combinatorics.Graph.Hamiltonian
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Defs
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Zmod.Basic
import Mathlib.FieldTheory.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Inversion
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.MeasurableSpace
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace sum_of_integer_solutions_to_absolute_inequalities_l59_59786

noncomputable def sum_of_solutions : ℤ :=
  if (∀ n : ℤ, |n| < |n-2| ∧ |n-2| < 10 → n ∈ (-8, 1] ∨ n ∈ [0, 1]) then -35 else 0

theorem sum_of_integer_solutions_to_absolute_inequalities :
  sum_of_solutions = -35 := by
  sorry

end sum_of_integer_solutions_to_absolute_inequalities_l59_59786


namespace square_area_from_diagonal_l59_59569

-- Define the vertices of the square
structure Point (α : Type) :=
(coord_x : α)
(coord_y : α)

def E := Point.mk 1 1
def F := Point.mk (-3) 4

-- Define the function to calculate the distance between two points
def distance (P Q : Point ℝ) :=
  real.sqrt ((Q.coord_x - P.coord_x)^2 + (Q.coord_y - P.coord_y)^2)

-- Definition to calculate the area of the square given its diagonal length
def square_area (d : ℝ) :=
  (d / real.sqrt 2)^2

-- Proof statement: given the points E and F, the area of the square is 12.5 square units
theorem square_area_from_diagonal : square_area (distance E F) = 12.5 :=
by 
  sorry

end square_area_from_diagonal_l59_59569


namespace common_ratio_geometric_sequence_l59_59689

-- Define a geometric sequence and sum of its first n terms
noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = (a n) * q

noncomputable def sum_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0       := a 0
| (n + 1) := (sum_first_n_terms n) + a (n + 1)

-- Define the condition S_3 = 3 * a_3
noncomputable def condition_S3 (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  S 3 = 3 * (a 3)

-- The proof statement
theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h1 : geometric_sequence a q)
  (h2 : ∀ n, S n = sum_first_n_terms a n)
  (h3 : condition_S3 a S) :
  q = 1 ∨ q = -1 / 2 := sorry

end common_ratio_geometric_sequence_l59_59689


namespace volume_not_determined_l59_59209

noncomputable def tetrahedron_volume_not_unique 
  (area1 area2 area3 : ℝ) (circumradius : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∃ a' b' c', 
      (a ≠ a' ∨ b ≠ b' ∨ c ≠ c') ∧ 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)))

theorem volume_not_determined 
  (area1 area2 area3 circumradius: ℝ) 
  (h: tetrahedron_volume_not_unique area1 area2 area3 circumradius) : 
  ¬ ∃ (a b c : ℝ), 
    (area1 = 1 / 2 * a * b) ∧ 
    (area2 = 1 / 2 * b * c) ∧ 
    (area3 = 1 / 2 * c * a) ∧ 
    (circumradius = Real.sqrt ((a^2 + b^2 + c^2) / 2)) ∧ 
    (∀ a' b' c', 
      (1 / 2 * a' * b' = area1) ∧ 
      (1 / 2 * b' * c' = area2) ∧ 
      (1 / 2 * c' * a' = area3) ∧ 
      (circumradius = Real.sqrt ((a'^2 + b'^2 + c'^2) / 2)) → 
      (a = a' ∧ b = b' ∧ c = c')) := 
by sorry

end volume_not_determined_l59_59209


namespace num_different_flavors_l59_59546

-- Define the number of blue candies and yellow candies
def blue_candies := 5
def yellow_candies := 4

-- Define the main proof statement
theorem num_different_flavors : 
  (∃ (x : ℕ) (y : ℕ), x ≤ blue_candies ∧ y ≤ yellow_candies ∧ (x ≠ 0 ∨ y ≠ 0)) →
  9 :=
sorry

end num_different_flavors_l59_59546


namespace inequality_neg_reciprocal_l59_59553

theorem inequality_neg_reciprocal (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  - (1 / a) < - (1 / b) :=
sorry

end inequality_neg_reciprocal_l59_59553


namespace domino_path_count_l59_59697

theorem domino_path_count : ∃ arrangements : Nat, 
  (∀ grid : Fin 6 × Fin 5 → Bool, 
     (∀ a b : Fin 6 × Fin 5, contiguous grid a b → grid a = true ∧ grid b = true ∨ grid a = false ∧ grid b = false) ∧
     (∀ a : Fin 6 × Fin 5, grid a = true ↔ ∃ d, d = (a.1 + 1, a.2) ∨ d = (a.1, a.2 + 1)) ∧
     (∀ a : Fin 6 × Fin 5, grid a = true → ∃ n, n < 10 ∧ ( Fin (n + 1).succ ≃ Fin 6 × Fin 5)) ∧
     (arrangements = 126)) :=
begin
  use 126,
  sorry
end

end domino_path_count_l59_59697


namespace number_of_solutions_to_subsets_l59_59853

theorem number_of_solutions_to_subsets :
  (nat.card { X : finset ℕ // {1, 2, 3} ⊆ X ∧ X ⊆ {1, 2, 3, 4, 5, 6, 7} } = 16) :=
by
  sorry

end number_of_solutions_to_subsets_l59_59853


namespace productive_employees_work_l59_59296

theorem productive_employees_work (total_work : ℝ) (P : ℝ) (Q : ℝ)
  (h1 : P = 0.2) (h2 : Q = 0.8) :
  (0.4 * total_work) = (0.85 * total_work) :=
by
  -- Given the conditions: 20% of employees do 80% of the work
  have h3 : 0.2 * total_work = 0.8 * total_work, from sorry
  -- 40% of the most productive employees perform 85% of the work
  exact sorry

end productive_employees_work_l59_59296


namespace four_digit_numbers_one_even_l59_59981

noncomputable def even_digits : Set ℕ := {0, 2, 4, 6, 8}
noncomputable def odd_digits : Set ℕ := {1, 3, 5, 7, 9}

theorem four_digit_numbers_one_even : 
  ∃ n : ℕ, 
  (∀ d1 d2 d3 d4, d1 ∈ odd_digits → d2 ∈ odd_digits → d3 ∈ odd_digits → d4 ∈ odd_digits → 
  d1 ≠ 0 → 1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10^4) → 
  (∀ i j k,
  ((j ∈ even_digits ∧ i ∈ odd_digits ∧ k ∈ odd_digits) →
  (1000 * j + 100 * i + 10 * k + d ≠ 0)) →
  n = 2375

   ∧
  (∀i j t,
  ((t ∈ even_digits ∧ i ∈ odd_digits ∧ j ∈ odd_digits) →
  (1000 * i + 100 * t + 10 * j + d ≠ 0)) →
  n = 2375))
, 
  n = 2375 :=
sorry

end four_digit_numbers_one_even_l59_59981


namespace exterior_angle_of_triangle_cond_40_degree_l59_59063

theorem exterior_angle_of_triangle_cond_40_degree (A B C : ℝ)
  (h1 : (A = 40 ∨ B = 40 ∨ C = 40))
  (h2 : A = B)
  (h3 : A + B + C = 180) :
  ((180 - C) = 80 ∨ (180 - C) = 140) :=
by
  sorry

end exterior_angle_of_triangle_cond_40_degree_l59_59063


namespace proof_problem_l59_59859

def diagonal_square_area (d : ℝ) : ℝ :=
  let s := d / Real.sqrt 2 in
  s * s

def rectangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  let width := (x2 - 2).abs in
  let height := (if y1 ≤  y2 then y2  - y1 else y1 - y2) * 2 in
  width * height

def triangle_area (m b : ℝ) : ℝ :=
  let x_int := 2 * b in
  let y_int := b in
  0.5 * x_int * y_int

theorem proof_problem :
  let A := diagonal_square_area (2 * Real.sqrt 2)
  let B := rectangle_area 2 2 4 3
  let C := triangle_area (-1/2) 2
  A = 4 ∧ B = 8 ∧ C = 4 ∧ A < B :=
by {
  have hA := calc
    A = 4             : by sorry,
  have hB := calc
    B = 8             : by sorry,
  have hC := calc
    C = 4             : by sorry,
  have hD := calc
    A < B             : by sorry,
  exact ⟨hA, hB, hC, hD⟩
}

end proof_problem_l59_59859


namespace number_of_integer_values_of_x_l59_59752

-- Define the new operation (star)
def star (a b : ℕ) : ℕ := a * a / b

-- Define the divisors function to count the number of divisors of a given number
def divisors_count (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

-- Main theorem to prove that there are 7 integer values of x such that 8 star x is a positive integer
theorem number_of_integer_values_of_x : divisors_count 64 = 7 :=
sorry

end number_of_integer_values_of_x_l59_59752


namespace quadratic_inequality_solution_l59_59310

theorem quadratic_inequality_solution (b c : ℝ) (f : ℝ → ℝ) 
  (h₀ : ∀ x, f x = x^2 + b*x + c)
  (h₁ : f (-1) = 0)
  (h₂ : f 2 = 0) :
  {x : ℝ | f x < 4} = set.Ioo (-2 : ℝ) 3 :=
sorry

end quadratic_inequality_solution_l59_59310


namespace find_data_set_l59_59832

-- Define the positive integers and conditions given
variables {x1 x2 x3 x4 : ℕ}

-- Define the conditions of the problem
def conditions (x1 x2 x3 x4 : ℕ) : Prop :=
  x1 + x2 + x3 + x4 = 8 ∧
  x2 = 2 ∧
  x3 = 2 ∧
  (1 ≤ x1 ∧ x1 ≤ 2) ∧
  (2 ≤ x3 ∧ x3 ≤ 3) ∧
  (2 ≤ x4 ∧ x4 ≤ 3) ∧
  x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧
  (let mean := 2 in let sd := 1 in
   let μ := (x1 + x2 + x3 + x4) / 4 in
   μ = mean ∧ 
   real.sqrt ((1 / 4) * ((x1 - mean)^2 + (x2 - mean)^2 + (x3 - mean)^2 + (x4 - mean)^2)) = sd)

-- Define the target data
def target_data : List ℕ := [1, 1, 3, 3]

-- Lean theorem statement for the proof problem
theorem find_data_set : conditions 1 1 3 3 :=
by {
  -- assumptions to be proven
  split,
  -- sum is 8
  simp,
  -- checks
  split; sorry, 
}

end find_data_set_l59_59832


namespace milkman_profit_l59_59461

theorem milkman_profit
    (total_pure_milk : ℝ)
    (water_mixed : ℝ)
    (milk_mixed : ℝ)
    (cost_per_liter : ℝ)
    (total_cost : total_pure_milk * cost_per_liter = 540)
    (unused_milk_cost : (total_pure_milk - milk_mixed) * cost_per_liter = 180)
    (mixture_volume : milk_mixed + water_mixed = 25)
    (selling_price : mixture_volume * cost_per_liter = 450)
    (used_milk_cost : milk_mixed * cost_per_liter = 360) :
    (selling_price - used_milk_cost = 90) :=
begin
    sorry
end

end milkman_profit_l59_59461


namespace logs_form_arithmetic_progression_l59_59620

variable (a b c s n : ℝ)
variable (h1 : a < b) (h2 : b < c) (h3 : 1 < s)
variable (h4 : b = a * s) (h5 : c = a * s ^ 2)

theorem logs_form_arithmetic_progression : 
  is_arithmetic_progression (log a n) (log b n) (log c n) := by sorry

-- Helper definition for arithmetic progression
def is_arithmetic_progression (u v w : ℝ) : Prop :=
  2 * v = u + w

end logs_form_arithmetic_progression_l59_59620


namespace parallel_lines_in_scalene_triangle_l59_59236

theorem parallel_lines_in_scalene_triangle :
  ∀ (A1 A2 A3 B12 B21 B13 B31 B23 B32 : Point),
  scalene_triangle A1 A2 A3 →
  point_symmetric_wrt_angle_bisector A1 A2 B12 B21 →
  point_symmetric_wrt_angle_bisector A1 A3 B13 B31 →
  point_symmetric_wrt_angle_bisector A2 A3 B23 B32 →
  parallel B12 B21 B13 B31 ∧ 
  parallel B12 B21 B23 B32 :=
begin
  sorry
end

end parallel_lines_in_scalene_triangle_l59_59236


namespace evaluate_fraction_l59_59879

theorem evaluate_fraction (n : ℕ) (h : n > 0) : 
  ((∑ k in Finset.range n, 8 * k^3)^(1/3 : ℝ)) / ((∑ k in Finset.range n, 27 * k^3)^(1/3 : ℝ)) = 2 / 3 := 
begin
  sorry
end

end evaluate_fraction_l59_59879


namespace part_i_part_ii_part_iii_l59_59957

-- Given equation condition
def equation_condition (a b c : ℕ) : Prop := 
  a^b * b^c = c^a

-- (i) Any prime divisor of a divides b
theorem part_i (a b c : ℕ) (h : equation_condition a b c) (p : ℕ) (hp : p.prime) (hpa : p ∣ a) : p ∣ b := 
sorry

-- (ii) Solve the equation under the assumption b ≥ a
theorem part_ii (a b c : ℕ) (h : equation_condition a b c) (hba : b ≥ a) : a = 1 ∧ b = 1 ∧ c = 1 := 
sorry

-- (iii) Prove that the equation has infinitely many solutions
theorem part_iii : ∃ (a b c : ℕ), (∃ t > 0, equation_condition (t ^ t) (t ^ (t - 1)) (t ^ t)) :=
sorry

end part_i_part_ii_part_iii_l59_59957


namespace find_vertex_C_coordinates_l59_59696

theorem find_vertex_C_coordinates :
  ∃ C : ℝ × ℝ, 
    C = (-4, 0) ∧ 
    (let A := (2, 0) in
     let B := (0, 4) in
     let euler_line (p : ℝ × ℝ) := p.1 - p.2 + 2 = 0 in
     let centroid (C : ℝ × ℝ) := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) in
     euler_line (centroid C)) :=
sorry

end find_vertex_C_coordinates_l59_59696


namespace sqrt_72_eq_6_sqrt_2_l59_59027

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := 
by
  sorry

end sqrt_72_eq_6_sqrt_2_l59_59027


namespace polynomial_balanced_sum_l59_59929

theorem polynomial_balanced_sum {P : Polynomial ℤ} (k : ℕ) (hdeg : P.degree = k) :
  ∃ (d : ℕ) (x : ℕ → ℤ), d ≤ k + 1 ∧ (∀ i j, i ≠ j → x i ≠ x j) ∧
    (∑ i in finset.range d, P.eval (x i)) = (∑ i in finset.range d + d, P.eval (x i)) :=
sorry

end polynomial_balanced_sum_l59_59929


namespace num_valid_permutations_l59_59276

theorem num_valid_permutations : 
  let digits := [2, 0, 2, 3]
  let num_2 := 2
  let total_permutations := Nat.factorial 4 / (Nat.factorial num_2 * Nat.factorial 1 * Nat.factorial 1)
  let valid_start_2 := Nat.factorial 3
  let valid_start_3 := Nat.factorial 3 / Nat.factorial 2
  total_permutations = 12 ∧ valid_start_2 = 6 ∧ valid_start_3 = 3 ∧ (valid_start_2 + valid_start_3 = 9) := 
by
  sorry

end num_valid_permutations_l59_59276


namespace cone_height_l59_59135

noncomputable def height_of_cone (V : ℝ) (θ : ℝ) : ℝ :=
  let r := (3 * V / Real.pi)^(1/3)
  in r

theorem cone_height
  (V : ℝ)
  (θ : ℝ)
  (h : ℝ)
  (hV : V = 8000 * Real.pi)
  (hθ : θ = 45) :
  h = 28.84 :=
by
  have r := (3 * V / Real.pi)^(1/3)
  have hr : h = r := by sorry
  rw [hV] at r
  norm_num at r
  sorry

end cone_height_l59_59135


namespace polynomial_constant_if_doubled_is_same_l59_59808

theorem polynomial_constant_if_doubled_is_same (Q : ℝ[X])
  (h : ∀ x : ℝ, Q.eval (2 * x) = Q.eval x) : ∃ c : ℝ, Q = polynomial.C c :=
sorry

end polynomial_constant_if_doubled_is_same_l59_59808


namespace amy_tiles_total_l59_59162

theorem amy_tiles_total :
  ∀ (length width border inner_tile_size : ℕ),
    length = 18 →
    width = 24 →
    border = 2 →
    inner_tile_size = 3 →
    let inner_length := length - 2 * border in
    let inner_width := width - 2 * border in
    let border_tiles := 2 * ((width - 2 * border) * border + (length - 2 * border) * border) in
    let inner_area := inner_length * inner_width in
    let inner_tiles := inner_area / (inner_tile_size * inner_tile_size) in
    let total_tiles := border_tiles + inner_tiles in
    total_tiles = 167 := 
by
  intros length width border inner_tile_size h_length h_width h_border h_inner_tile_size
  let inner_length := length - 2 * border
  let inner_width := width - 2 * border
  let border_tiles := 2 * ((width - 2 * border) * border + (length - 2 * border) * border)
  let inner_area := inner_length * inner_width
  let inner_tiles := inner_area / (inner_tile_size * inner_tile_size)
  let total_tiles := border_tiles + inner_tiles
  sorry

end amy_tiles_total_l59_59162


namespace min_value_and_arg_correct_l59_59951

def f (x : ℝ) : ℝ := 9 / (8 * (Real.cos (2 * x)) + 16) - (Real.sin x)^2

def min_value : ℝ := 0
def min_arg : ℝ := Real.pi / 3

theorem min_value_and_arg_correct :
  (∀ x, f x ≥ min_value) ∧ f min_arg = min_value ∧ min_arg > 0 ∧ (∀ y, y ≠ min_arg → f y ≠ min_value) →
  min_value + min_arg = Real.pi / 3 :=
by
  sorry

end min_value_and_arg_correct_l59_59951


namespace bhishma_speed_l59_59851

-- Given definitions based on conditions
def track_length : ℝ := 600
def bruce_speed : ℝ := 30
def time_meet : ℝ := 90

-- Main theorem we want to prove
theorem bhishma_speed : ∃ v : ℝ, v = 23.33 ∧ (bruce_speed * time_meet) = (v * time_meet + track_length) :=
  by
    sorry

end bhishma_speed_l59_59851


namespace coefficient_x2_l59_59523

def expression : ℕ → ℤ := λ n,
  if n = 1 then 3 else
  if n = 2 then -8 else
  if n = 3 then -5 else
  if n = 4 then -3 else
  0

theorem coefficient_x2 : expression 2 = -3 := by
  sorry

end coefficient_x2_l59_59523


namespace counterexample_to_T_exists_l59_59677

theorem counterexample_to_T_exists :
  ∃ n : ℕ, (sum_of_digits n % 9 = 0) ∧ (n % 9 ≠ 0) := sorry

end counterexample_to_T_exists_l59_59677


namespace twelve_sided_figure_area_is_13_cm2_l59_59762

def twelve_sided_figure_area_cm2 : ℝ :=
  let unit_square := 1
  let full_squares := 9
  let triangle_pairs := 4
  full_squares * unit_square + triangle_pairs * unit_square

theorem twelve_sided_figure_area_is_13_cm2 :
  twelve_sided_figure_area_cm2 = 13 := 
by
  sorry

end twelve_sided_figure_area_is_13_cm2_l59_59762


namespace jerry_rings_total_l59_59317

theorem jerry_rings_total (games : ℕ) (rings_per_game : ℕ) (total_rings : ℕ)
  (h1 : games = 8) (h2 : rings_per_game = 6) (h3 : total_rings = games * rings_per_game) :
  total_rings = 48 :=
by {
  rw [h1, h2] at h3,
  exact h3,
  sorry
}

end jerry_rings_total_l59_59317


namespace pipe_fill_time_l59_59359

def pipe_times (A B : ℕ) (T : ℕ) : Prop :=
  (1 / A - 1 / B) * B + 1 / A * (T - B) = 1

theorem pipe_fill_time :
  ∃ A : ℕ, pipe_times A 24 30 ∧ A = 15 :=
by
  use 15
  unfold pipe_times
  norm_num
  field_simp
  linarith

end pipe_fill_time_l59_59359


namespace find_function_l59_59890

theorem find_function (f : ℝ → ℝ)
  (h₁ : ∀ x : ℝ, x * (f (x + 1) - f x) = f x)
  (h₂ : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ |k| ≤ 1 :=
sorry

end find_function_l59_59890


namespace solve_inequality_l59_59215

theorem solve_inequality :
  {x : ℝ | 0 ≤ x ∧ x ≤ 1 } = {x : ℝ | x * (x - 1) ≤ 0} :=
by sorry

end solve_inequality_l59_59215


namespace arithmetic_sequence_sum_l59_59636

theorem arithmetic_sequence_sum :
  ∃ d : ℚ, let a_1 := 2 in let a_2 := a_1 + d in let a_5 := a_1 + 4 * d in
  a_2 + a_5 = 13 → a_5 + (a_1 + 5 * d) + (a_1 + 6 * d) = 33 :=
by
  sorry

end arithmetic_sequence_sum_l59_59636


namespace geom_seq_second_term_l59_59870

noncomputable def is_geometric_sequence (a1 a2 a3 : ℝ) := ∃ r : ℝ, a2 = a1 * r ∧ a3 = a2 * r

theorem geom_seq_second_term (b : ℝ) 
  (h1 : is_geometric_sequence 210 b (140 / 60)) 
  (h2 : b > 0) : 
  b = 7 * real.sqrt 10 :=
sorry

end geom_seq_second_term_l59_59870


namespace compute_value_l59_59279

def binom (x : ℝ) (k : ℕ) : ℝ :=
  (Finset.range k).prod (λ i => x - i) / (Nat.factorial k)

theorem compute_value (h : (-1: ℝ) / 2 ≠ 0) (k_pos : 1007 ≠ 0) :
  (finset.prod (finset.range 1007) (λ i, (-1 / 2 - i)) / (nat.factorial 1007) * 2 ^ 1007) / (finset.prod (finset.range 1007) (λ i, 2014 - i) / (nat.factorial 1007)) = - 1 / 2 ^ 1007 := sorry

end compute_value_l59_59279


namespace count_good_triangles_l59_59323

-- Define the context and conditions
def is_good_triangle (A B C : ℕ) : Prop :=
  A + B + C = 180 ∧  -- The sum of angles in a triangle is 180 degrees
  A > 0 ∧ B > 0 ∧ C > 0 ∧  -- Angles are positive integers
  ∃ P, P ≠ A ∧  -- There exists a point P on side AB
    ∃ Q, Q ≠ A ∧  -- There exists a point Q on side AC
      ∀ O, ( -- Circumcenter O exists
        (circumcircle_tangent_to_BO P O A B) ∧  -- POA circumcircle tangent to BO
        (circumcircle_tangent_to_CO Q O A C) ∧  -- QOA circumcircle tangent to CO
        (perimeter_APQ_geq_AB_plus_AC P Q A B C))  -- Perimeter condition

-- Proof that there are 59 good triangles
theorem count_good_triangles : { (A, B, C) : ℕ × ℕ × ℕ // is_good_triangle A B C }.card = 59 :=
sorry  -- Proof is not provided

end count_good_triangles_l59_59323


namespace greatest_positive_integer_difference_l59_59969

def A : Set ℤ := {-6, -5, -4, -3}
def B : Set ℚ := {2/3, 3/4, 7/9, 2.5}
def C : Set ℝ := {5, 5.5, 6, 6.5}

theorem greatest_positive_integer_difference :
  ∃ (a ∈ A) (b ∈ B) (c ∈ C), (c - real.sqrt (b.toReal)) - (a + real.sqrt (b.toReal)).toReal = 5 :=
sorry

end greatest_positive_integer_difference_l59_59969


namespace problem1_problem2_problem3_l59_59124

theorem problem1 : 9^(3/2) * 64^(1/6) / 3^0 = 54 :=
by sorry

theorem problem2 : (1/9)^(1/2) * 36^(-1/2) / 3^(-3) = 3/2 :=
by sorry

theorem problem3 (a : ℝ) (ha : 0 < a) : a^2 / (sqrt(a) * 3 * a^2) = a^(5/6) :=
by sorry

end problem1_problem2_problem3_l59_59124


namespace car_speed_proof_l59_59856

noncomputable def car_speed_problem (vB : ℕ) : Prop :=
  let dA := 3 * 58 in
  let dB := 3 * vB in
  (dA = dB + 24) → vB = 50

theorem car_speed_proof (vB : ℕ) : car_speed_problem vB :=
by
  sorry

end car_speed_proof_l59_59856


namespace decreasing_interval_f_x_plus_1_l59_59554

variable (f : ℝ → ℝ)
variable (x : ℝ)

-- Definition of the function f(x) given in the problem
def f_eq : (x : ℝ) → f x = (x - 2) ^ 2 := sorry

-- The interval for x defined in the problem
def x_in_interval : x ∈ Icc (-1 : ℝ) 3 := sorry

-- Prove that the interval where f(x + 1) is monotonically decreasing is x ∈ [-2, 2]
theorem decreasing_interval_f_x_plus_1 : 
  ∀ x, f_eq f x → x_in_interval x → (f (x + 1)).monotonically_decreasing_on (Icc (-2 : ℝ) 2) := sorry

end decreasing_interval_f_x_plus_1_l59_59554


namespace tan_of_alpha_l59_59623

theorem tan_of_alpha
  (y : ℝ)
  (h_cos : cos α = 3 / 5)
  (h_point : (3 : ℝ, y) ∈ { P : ℝ × ℝ | y < 0 }) :
  tan α = -4 / 3 :=
by sorry

end tan_of_alpha_l59_59623


namespace rectangle_area_l59_59448

noncomputable def circle_radius := 8
noncomputable def rect_ratio : ℕ × ℕ := (3, 1)
noncomputable def rect_area (width length : ℕ) : ℕ := width * length

theorem rectangle_area (width length : ℕ) 
  (h1 : 2 * circle_radius = width) 
  (h2 : rect_ratio.1 * width = length) : 
  rect_area width length = 768 := 
sorry

end rectangle_area_l59_59448


namespace total_toothpicks_needed_l59_59458

/-- The number of toothpicks needed to construct both a large and smaller equilateral triangle 
    side by side, given the large triangle has a base of 100 small triangles and the smaller triangle 
    has a base of 50 small triangles -/
theorem total_toothpicks_needed 
  (base_large : ℕ) (base_small : ℕ) (shared_boundary : ℕ) 
  (h1 : base_large = 100) (h2 : base_small = 50) (h3 : shared_boundary = base_small) :
  3 * (100 * 101 / 2) / 2 + 3 * (50 * 51 / 2) / 2 - shared_boundary = 9462 := 
sorry

end total_toothpicks_needed_l59_59458


namespace geometric_sequence_condition_l59_59758

-- Definition of a geometric sequence
def is_geometric_sequence (x y z : ℤ) : Prop :=
  y ^ 2 = x * z

-- Lean 4 statement based on the condition and correct answer tuple
theorem geometric_sequence_condition (a : ℤ) :
  is_geometric_sequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by 
  sorry

end geometric_sequence_condition_l59_59758


namespace solve_congruence_l59_59723

theorem solve_congruence (n : ℤ) : (14 * n ≡ 9 [MOD 53]) ↔ (n ≡ 36 [MOD 53]) := 
by 
  sorry

end solve_congruence_l59_59723


namespace median_name_length_l59_59474

theorem median_name_length :
  ∀ (names : List ℕ),
  (names = List.replicate 9 3 ++ List.replicate 5 4 ++ List.replicate 1 5 ++ List.replicate 2 6 ++ List.replicate 4 7) →
  median names = 4 :=
by 
  sorry

end median_name_length_l59_59474


namespace katy_books_l59_59654

theorem katy_books (x : ℕ) (h : x + 2 * x + (2 * x - 3) = 37) : x = 8 :=
by
  sorry

end katy_books_l59_59654


namespace solution_l59_59934

variable {n : ℕ}
variables {a : ℕ → ℕ} {S : ℕ → ℕ} {b : ℕ → ℝ}

-- Condition: Arithmetic sequence {a_n} with a specific a_2 and S_5 + a_3
def arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def conditions (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ) : Prop :=
  a 2 = 3 ∧ S 5 + a 3 = 30 ∧ arithmetic_seq a d

-- Sequence b_n and its sum T_n
def b (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) : ℝ :=
  (a (n + 1) : ℝ) / ((S n) * (S (n + 1)))

def T (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b a S i

theorem solution (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ) (H_arith : arithmetic_seq a d)
  (H_cond : conditions a S d)
  (H_b : ∀ n, b a S n = 1 / (n^2) - 1 / ((n+1)^2)) :
  (∀ n, a n = 2n - 1) ∧ (∀ n, S n = n^2) ∧ (∀ n, T S n = 1 - 1 / ((n+1)^2)) :=
sorry

end solution_l59_59934


namespace valid_k_values_l59_59660

-- Define the set of all lattice points in n-dimensional space
def lattice_points (n : ℕ) : Set (Fin n → ℤ) :=
  { p | True }

-- Define the distance between two n-dimensional points
def dist {n : ℕ} (A B : Fin n → ℤ) : ℤ :=
  Finset.univ.sum (λ i, Int.natAbs (A i - B i))

-- Define the set Q(A) as the set of all points within a distance of 1 from A
def Q_set {n : ℕ} (A : Fin n → ℤ) : Set (Fin n → ℤ) :=
  { P | dist A P ≤ 1 }

-- The Lean logo!
theorem valid_k_values {n : ℕ} (k : ℕ) :
  (∃ (red_points : Fin n → Bool), (∀ A, Finset.card ({ P ∈ Q_set A | red_points P }) = k) ∧ (∃ P, red_points P = true) ∧ (∃ P, red_points P = false)) →
  1 ≤ k ∧ k ≤ 2 * n :=
sorry

end valid_k_values_l59_59660


namespace return_trip_time_l59_59463

variables (d p w : ℝ)
-- Condition 1: The outbound trip against the wind took 120 minutes.
axiom h1 : d = 120 * (p - w)
-- Condition 2: The return trip with the wind took 15 minutes less than it would in still air.
axiom h2 : d / (p + w) = d / p - 15

-- Translate the conclusion that needs to be proven in Lean 4
theorem return_trip_time (h1 : d = 120 * (p - w)) (h2 : d / (p + w) = d / p - 15) : (d / (p + w) = 15) ∨ (d / (p + w) = 85) :=
sorry

end return_trip_time_l59_59463


namespace marble_pair_count_l59_59413

noncomputable def number_of_marble_pairs : ℕ :=
  let distinct_colors := 5
  let yellow_pairs := 1
  let distinct_pairs := Nat.choose distinct_colors 2
  yellow_pairs + distinct_pairs

theorem marble_pair_count :
  -- Conditions
  let red := 1
  let green := 1
  let blue := 1
  let purple := 1
  let yellow := 3
  -- Question and Answer
  number_of_marble_pairs red green blue purple yellow = 11 :=
by
  -- Proof is omitted
  sorry

end marble_pair_count_l59_59413


namespace log_ineq_condition_l59_59267

open Real

theorem log_ineq_condition (m : ℝ) (h_pos : 0 < m) (h_ne_one : m ≠ 1) :
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → x^2 < log m x) ↔ (1/16 ≤ m ∧ m < 1) :=
by sorry

end log_ineq_condition_l59_59267


namespace total_area_to_be_painted_l59_59812

def width := 12
def length := 15
def height := 6

def area_to_be_painted (width length height : ℕ) : ℕ :=
  2 * ((2 * width * height) + (2 * length * height) + (width * length) + (width * length))

theorem total_area_to_be_painted :
  area_to_be_painted width length height = 1368 :=
by
  sorry

end total_area_to_be_painted_l59_59812


namespace minimum_value_f_sum_of_roots_g_eq_4_l59_59274

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.cos x)
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + m
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 6) + 3

theorem minimum_value_f (m : ℝ) : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x m = 2 ↔ m = 2 :=
by sorry

theorem sum_of_roots_g_eq_4 : 
  (∑ x in {x | x ∈ Set.Icc 0 Real.pi ∧ g x = 4}, x) = 5 * Real.pi / 3 :=
by sorry

end minimum_value_f_sum_of_roots_g_eq_4_l59_59274


namespace no_solutions_l59_59725

theorem no_solutions
  (x y z : ℤ)
  (h : x^2 + y^2 = 4 * z - 1) : False :=
sorry

end no_solutions_l59_59725


namespace boat_speed_in_still_water_l59_59432

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 7) : b = 9 :=
by sorry

end boat_speed_in_still_water_l59_59432


namespace line_tangent_to_circle_l59_59564

theorem line_tangent_to_circle
  (θ : Real) :
  ∀ l : Real → Real → Prop,
  (l = λ x y, x * Real.cos θ + y * Real.sin θ + 2 = 0) →
  ∀ x y : Real, (x^2 + y^2 = 4) →  
  ∃ d : Real, (d = abs(2) / Real.sqrt(1)) ∧ d = 2 →
  (d = 2 → d = 2) :=
begin
  intros l hl x y hy d hd he h,
  sorry
end

end line_tangent_to_circle_l59_59564


namespace alice_paradox_l59_59420

theorem alice_paradox (answer : String) (judge : String → Prop) :
  (answer = "No" → ∀ outcome, ¬ (judge outcome ↔ ¬ judge outcome)) :=
begin
  sorry,
end

end alice_paradox_l59_59420


namespace subset_iff_a_values_l59_59602

theorem subset_iff_a_values (a : ℝ) :
  let P := { x : ℝ | x^2 = 1 }
  let Q := { x : ℝ | a * x = 1 }
  Q ⊆ P ↔ a = 0 ∨ a = 1 ∨ a = -1 :=
by sorry

end subset_iff_a_values_l59_59602


namespace exists_zero_in_interval_l59_59068

noncomputable def f (x : ℝ) : ℝ := logBase 2 x + x - 4

theorem exists_zero_in_interval : 
  monotone_on f (Ioi 0) 
  → f 2 < 0 
  → f 3 > 0 
  → ∃ c ∈ Ioo 2 3, f c = 0 :=
by {
  sorry -- Proof omitted
}

end exists_zero_in_interval_l59_59068


namespace find_difference_of_a_and_b_l59_59939

-- Define the conditions
variables (a b : ℝ)
axiom cond1 : 4 * a + 3 * b = 8
axiom cond2 : 3 * a + 4 * b = 6

-- Statement for the proof
theorem find_difference_of_a_and_b : a - b = 2 :=
by
  sorry

end find_difference_of_a_and_b_l59_59939


namespace same_price_missing_capital_l59_59409

-- Define the setup
variables {City : Type} [fintype City] {C : City} (d : City → City → ℝ)
variables (C1 C2 : City) [decidable_eq City]

-- Condition 1: Symmetric price function for direct flights
def symmetric_price : Prop :=
  ∀ (A B : City), d(A, B) = d(B, A)

-- Condition 2: All round trips visiting every city exactly once have the same cost
def equal_round_trip_cost (σ : ℝ) : Prop :=
  ∀ (p : finset City), p.card = fintype.card City → ∃ l : list City, 
  (∀ (i < (l.length - 1)), l.nth_le i (by auto) ∈ p) ∧ 
  (list.foldl (+) 0 (list.map (λ (i : ℕ), d (l.nth_le i sorry) (l.nth_le (i+1) sorry)) (list.range (l.length - 1))) = σ)

-- Theorem statement
theorem same_price_missing_capital [fintype {x // x ≠ C}] (σ : ℝ) :
  symmetric_price d →
  equal_round_trip_cost d σ →
  ∀ (s : finset {x // x ≠ C}), s.card = fintype.card {x // x ≠ C} → ∃ l : list {x // x ≠ C}, 
  (∀ (i < (l.length - 1)), l.nth_le i (by auto) ∈ s) ∧ 
  (list.foldl (+) 0 (list.map (λ (i : ℕ), d (subtype.val (l.nth_le i sorry)) (subtype.val (l.nth_le (i+1) sorry))) (list.range (l.length - 1))) = σ - d C (subtype.val (l.nth_le 0 sorry)) - d C (subtype.val (l.nth_le (l.length - 1) sorry))) :=
begin
  intros,
  sorry
end

end same_price_missing_capital_l59_59409


namespace skew_lines_common_perpendiculars_l59_59556

variables (a b c a' b' c' : Line)
variables [SkewLines a b c]
variables [Perpendicular a' b] [Perpendicular a' c]
variables [Perpendicular b' c] [Perpendicular b' a]
variables [Perpendicular c' a] [Perpendicular c' b]

theorem skew_lines_common_perpendiculars :
  (Perpendicular a b' ∧ Perpendicular a c') ∧
  (Perpendicular b c' ∧ Perpendicular b a') ∧
  (Perpendicular c a' ∧ Perpendicular c b') :=
sorry

end skew_lines_common_perpendiculars_l59_59556


namespace abc_def_intersection_l59_59684

theorem abc_def_intersection (A B C D : ℝ × ℝ)
  (hA : A = (0, 0)) (hB : B = (2, 3)) (hC : C = (5, 4)) (hD : D = (6, 1)) :
  ∃ (p q r s : ℤ), p + q + r + s = 20 ∧ 
  (p : ℚ) / q = ((5 : ℚ) + (6 : ℚ)) / 2 ∧ 
  (r : ℚ) / s = ((4 : ℚ) + (1 : ℚ)) / 2 :=
sorry

end abc_def_intersection_l59_59684


namespace odd_gcd_m_divides_sum_l59_59928

theorem odd_gcd_m_divides_sum (m n : ℕ) (hm : m % 2 = 1) (h_gcd : Nat.gcd m (2^n - 1) = 1) :
  m ∣ ∑ k in Finset.range(m), k^n := 
by 
  sorry

end odd_gcd_m_divides_sum_l59_59928


namespace coffee_table_price_l59_59658

theorem coffee_table_price :
  let sofa := 1250
  let armchairs := 2 * 425
  let rug := 350
  let bookshelf := 200
  let subtotal_without_coffee_table := sofa + armchairs + rug + bookshelf
  let C := 429.24
  let total_before_discount_and_tax := subtotal_without_coffee_table + C
  let discounted_total := total_before_discount_and_tax * 0.90
  let final_invoice_amount := discounted_total * 1.06
  final_invoice_amount = 2937.60 :=
by
  sorry

end coffee_table_price_l59_59658


namespace can_still_row_probability_l59_59457

/-- Define the probabilities for the left and right oars --/
def P_left1_work : ℚ := 3 / 5
def P_left2_work : ℚ := 2 / 5
def P_right1_work : ℚ := 4 / 5 
def P_right2_work : ℚ := 3 / 5

/-- Define the probabilities of the failures as complementary probabilities --/
def P_left1_fail : ℚ := 1 - P_left1_work
def P_left2_fail : ℚ := 1 - P_left2_work
def P_right1_fail : ℚ := 1 - P_right1_work
def P_right2_fail : ℚ := 1 - P_right2_work

/-- Define the probability of both left oars failing --/
def P_both_left_fail : ℚ := P_left1_fail * P_left2_fail

/-- Define the probability of both right oars failing --/
def P_both_right_fail : ℚ := P_right1_fail * P_right2_fail

/-- Define the probability of all four oars failing --/
def P_all_fail : ℚ := P_both_left_fail * P_both_right_fail

/-- Calculate the probability that at least one oar on each side works --/
def P_can_row : ℚ := 1 - (P_both_left_fail + P_both_right_fail - P_all_fail)

theorem can_still_row_probability :
  P_can_row = 437 / 625 :=
by {
  -- The proof is to be completed
  sorry
}

end can_still_row_probability_l59_59457


namespace final_score_l59_59875

def dart1 : ℕ := 50
def dart2 : ℕ := 0
def dart3 : ℕ := dart1 / 2

theorem final_score : dart1 + dart2 + dart3 = 75 := by
  sorry

end final_score_l59_59875


namespace commutative_l59_59776

variable (R : Type) [NonAssocRing R]
variable (star : R → R → R)

axiom assoc : ∀ x y z : R, star (star x y) z = star x (star y z)
axiom comm_left : ∀ x y z : R, star (star x y) z = star (star y z) x
axiom distinct : ∀ {x y : R}, x ≠ y → ∃ z : R, star z x ≠ star z y

theorem commutative (x y : R) : star x y = star y x := sorry

end commutative_l59_59776


namespace sec_neg_420_eq_2_l59_59887

theorem sec_neg_420_eq_2 : ∀ (x : ℝ), (x = -420) → 
  (∀ y : ℝ, sec y = 1 / cos y) →
  (∀ (z k : ℝ), cos (z + 360 * k) = cos z) →
  (cos 60 = 1 / 2) →
  sec x = 2 :=
by
  intros x hx hsec_cos hcos_period hcos_60
  sorry

end sec_neg_420_eq_2_l59_59887


namespace opposite_number_of_1_minus_sqrt2_l59_59540

theorem opposite_number_of_1_minus_sqrt2 : 
  ∃ x : ℝ, (1 - real.sqrt 2) + x = 0 ∧ x = -1 + real.sqrt 2 := 
by
  use (-1 + real.sqrt 2)
  split
  · sorry
  · rfl

end opposite_number_of_1_minus_sqrt2_l59_59540


namespace total_practice_hours_l59_59362

def weekly_practice_hours : ℕ := 4
def weeks_in_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours : (weekly_practice_hours * weeks_in_month) * months = 80 := by
  -- Calculation for weekly practice in hours
  let monthly_hours := weekly_practice_hours * weeks_in_month
  -- Calculation for total practice in hours
  have total_hours : ℕ := monthly_hours * months
  have calculation : total_hours = 80 := 
    by simp [weekly_practice_hours, weeks_in_month, months, monthly_hours, total_hours]
  exact calculation

end total_practice_hours_l59_59362


namespace largest_constant_for_inequality_l59_59902

variable (a b c d e : ℝ)

theorem largest_constant_for_inequality (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) :
  (sqrt (a / (b + c + d + e)) + sqrt (b / (a + c + d + e)) +
   sqrt (c / (a + b + d + e)) + sqrt (d / (a + b + c + e))) > 2 :=
sorry

end largest_constant_for_inequality_l59_59902


namespace train_length_l59_59157

/-- Definition of speed conversion from kmph to m/s. -/
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (5 / 18)

/-- The length of a train passing a man running in the opposite direction. -/
theorem train_length
  (train_speed_kmph : ℝ)
  (man_speed_kmph : ℝ)
  (time_sec : ℝ)
  (train_speed_kmph = 60)
  (man_speed_kmph = 6)
  (time_sec = 12) :
  let relative_speed_mps := kmph_to_mps (train_speed_kmph + man_speed_kmph)
  in relative_speed_mps * time_sec = 220 := by
  sorry

end train_length_l59_59157


namespace min_value_inequality_l59_59332

theorem min_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, (∀ (a b : ℝ), 0 < a → 0 < b → c ≤ (sqrt ((a^2 + b^2) * (4 * a^2 + b^2)) / (a * b))) ∧
           (∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ sqrt ((a^2 + b^2) * (4 * a^2 + b^2)) / (a * b) = c) :=
begin
  let c := 3,
  use c,
  split,
  { intros a b ha hb,
    calc
      (sqrt ((a^2 + b^2) * (4 * a^2 + b^2)) / (a * b))
           ≥ (3 : ℝ) : sorry
  },
  { use [a, a * real.sqrt 2],
    split,
    { exact ha },
    split,
    { exact mul_pos ha (real.sqrt_pos.mpr zero_lt_two) },
    { sorry }
  }
end

end min_value_inequality_l59_59332


namespace train_cross_bridge_time_l59_59275

theorem train_cross_bridge_time :
  ∀ (length_train : ℕ) (speed_train_kmph : ℕ) (length_bridge : ℕ),
    length_train = 110 →
    speed_train_kmph = 60 →
    length_bridge = 190 →
    let speed_train_mps := (speed_train_kmph * 1000) / 3600,
        total_distance := length_train + length_bridge,
        time := total_distance / speed_train_mps
    in time ≈ 18 :=
by
  intros length_train speed_train_kmph length_bridge h_train h_speed h_bridge
  let speed_train_mps := (speed_train_kmph * 1000) / 3600
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_train_mps
  have h1: length_train = 110 := by assumption
  have h2: speed_train_kmph = 60 := by assumption
  have h3: length_bridge = 190 := by assumption
  sorry

end train_cross_bridge_time_l59_59275


namespace minimal_m_exists_l59_59537

theorem minimal_m_exists (n k m : ℕ) (hnk : n > k) (hk1 : k > 1) 
    (h_eq : (10^n - 1) / 9 = ((10^k - 1) / 9) * m) : m = 101 := 
begin
  sorry
end

end minimal_m_exists_l59_59537


namespace abs_nested_expression_l59_59908

theorem abs_nested_expression : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry

end abs_nested_expression_l59_59908


namespace exists_plane_section_with_acute_angles_l59_59933

noncomputable def is_acute (angle : ℝ) : Prop := 
  0 < angle ∧ angle < π / 2

noncomputable def trihedral_angle (O : Point) : Prop := 
  -- This is a placeholder; actual definition requires additional geometry setup
  sorry

theorem exists_plane_section_with_acute_angles 
  (O : Point) 
  (h : trihedral_angle O) 
  : ∃ A B C : Point, 
    is_acute (∠ OAB) ∧ 
    is_acute (∠ OBA) ∧ 
    is_acute (∠ OBC) ∧
    is_acute (∠ OCB) ∧
    is_acute (∠ OAC) ∧
    is_acute (∠ OCA) :=
sorry

end exists_plane_section_with_acute_angles_l59_59933


namespace sum_of_number_and_reverse_l59_59042

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
  (h5 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 99 := 
sorry

end sum_of_number_and_reverse_l59_59042


namespace determine_pairs_l59_59867

theorem determine_pairs (p : ℕ) (hp: Nat.Prime p) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ p^x - y^3 = 1 ∧ ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2)) := 
sorry

end determine_pairs_l59_59867


namespace milk_production_days_l59_59989

theorem milk_production_days (x : ℝ) (hx : x ≠ -6 ∧ x ≠ -4 ∧ x ≠ -9) :
  let daily_production_per_cow := (x + 9) / ((x + 6) * (x + 4)),
      total_daily_production := (x + 4) * daily_production_per_cow,
      required_days := (x + 11) / total_daily_production
  in required_days = ((x + 11) * (x + 6)) / (x + 9) :=
by
  -- Definitions for clarity (although not strictly necessary)
  let daily_production_per_cow := (x + 9) / ((x + 6) * (x + 4)),
  let total_daily_production := (x + 4) * daily_production_per_cow,
  let required_days := (x + 11) / total_daily_production,
  -- Prove the required statement (proof omitted)
  sorry

end milk_production_days_l59_59989


namespace goods_train_pass_time_l59_59142

noncomputable def time_to_pass_man (speed_man_train speed_goods_train : ℤ) (length_goods_train : ℤ) : ℤ :=
let relative_speed_kmph := speed_man_train + speed_goods_train in
let relative_speed_mps := (relative_speed_kmph * 1000) / 3600 in
length_goods_train / relative_speed_mps

theorem goods_train_pass_time
  (speed_man_train : ℤ) (speed_goods_train : ℤ) (length_goods_train : ℤ)
  (h_man_train_speed : speed_man_train = 56)
  (h_goods_train_speed : speed_goods_train = 42.4)
  (h_length_goods_train : length_goods_train = 410) :
  time_to_pass_man speed_man_train speed_goods_train length_goods_train ≈ 15 :=
by
  sorry

end goods_train_pass_time_l59_59142


namespace polar_to_rect_l59_59542

-- Definition of the problem in Lean 4
theorem polar_to_rect (θ x y : ℝ) (h1 : ρ = sqrt (x^2 + y^2))
  (h2 : cos θ = x / ρ) (h3 : ρ = (2 + 2 * cos θ) / (sin θ) ^ 2) : y^2 = x + 1 :=
sorry

end polar_to_rect_l59_59542


namespace volume_of_solid_rotated_is_correct_l59_59855

noncomputable def volume_of_solid : ℝ := 
  let f₁ (y : ℝ) := 5 * Real.sin y
  let f₂ (y : ℝ) := Real.sin y
  let integrand (y : ℝ) := (f₁ y)^2 - (f₂ y)^2
  π * ∫ y in 0..(π / 2), integrand y

theorem volume_of_solid_rotated_is_correct :
  volume_of_solid = 6 * π^2 := 
sorry

end volume_of_solid_rotated_is_correct_l59_59855


namespace radius_of_ball_l59_59444

theorem radius_of_ball (diameter depth : ℝ) (h₁ : diameter = 30) (h₂ : depth = 10) : 
  ∃ r : ℝ, r = 25 :=
by
  sorry

end radius_of_ball_l59_59444


namespace b_profit_l59_59114

-- Defining the conditions
def ratio_a_c (a c : ℕ) : Prop := a = 2 * c
def ratio_a_b (a b : ℕ) : Prop := b = 3 * (a / 2)
def total_profit := 150000

-- Proving that B received Rs 75,000
theorem b_profit (a b c : ℕ) (h1 : ratio_a_c a c) (h2 : ratio_a_b a b) : 
  let inv_A := a,
      inv_B := b,
      inv_C := c
  let total := inv_A + inv_B + inv_C in
  (inv_B / total.toFloat) * total_profit = 75000 :=
by
  sorry

end b_profit_l59_59114


namespace solve_for_x_l59_59373

theorem solve_for_x : ∀ x : ℝ, -3 * x - 9 = 6 * x + 18 ↔ x = -3 := by
  intro x
  split
  · intro h
    sorry
  · intro h
    rw h
    sorry

end solve_for_x_l59_59373


namespace profit_function_profit_for_240_barrels_barrels_for_760_profit_l59_59387

-- Define fixed costs, cost price per barrel, and selling price per barrel as constants
def fixed_costs : ℝ := 200
def cost_price_per_barrel : ℝ := 5
def selling_price_per_barrel : ℝ := 8

-- Definitions for daily sales quantity (x) and daily profit (y)
def daily_sales_quantity (x : ℝ) : ℝ := x
def daily_profit (x : ℝ) : ℝ := (selling_price_per_barrel * x) - (cost_price_per_barrel * x) - fixed_costs

-- Prove the functional relationship y = 3x - 200
theorem profit_function (x : ℝ) : daily_profit x = 3 * x - fixed_costs :=
by sorry

-- Given sales quantity is 240 barrels, prove profit is 520 yuan
theorem profit_for_240_barrels : daily_profit 240 = 520 :=
by sorry

-- Given profit is 760 yuan, prove sales quantity is 320 barrels
theorem barrels_for_760_profit : ∃ (x : ℝ), daily_profit x = 760 ∧ x = 320 :=
by sorry

end profit_function_profit_for_240_barrels_barrels_for_760_profit_l59_59387


namespace largest_angle_of_pentagon_l59_59732

theorem largest_angle_of_pentagon (a d : ℝ) (h1 : a = 100) (h2 : d = 2) :
  let angle1 := a
  let angle2 := a + d
  let angle3 := a + 2 * d
  let angle4 := a + 3 * d
  let angle5 := a + 4 * d
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧ angle5 = 116 :=
by
  sorry

end largest_angle_of_pentagon_l59_59732


namespace MikeInvestment_l59_59347

theorem MikeInvestment (M : ℝ) : 
  let total_profit := 3000 
  let third_profit := total_profit / 3
  let equal_share := third_profit / 2
  let remaining_profit := (2 / 3) * total_profit
  let mary_investment := 650
  (equal_share + (mary_investment / (mary_investment + M)) * remaining_profit = 
   equal_share + (M / (mary_investment + M)) * remaining_profit + 600) →
  M = 350 :=
begin
  sorry
end

end MikeInvestment_l59_59347


namespace probability_flies_swept_by_minute_hand_l59_59001

theorem probability_flies_swept_by_minute_hand :
  let flies_positions := {12, 3, 6, 9}
  -- Define the favorable starting intervals for the 20-minute sweep.
  let favorable_intervals := [(55, 60), (20, 25), (35, 40), (50, 55)]
  -- Total possible minutes in an hour
  let total_minutes := 60
  -- Total favorable minutes
  let favorable_minutes := 20
  -- Calculate the probability
  (favorable_minutes / total_minutes : ℝ) = (1 / 3 : ℝ):=
by
  sorry

end probability_flies_swept_by_minute_hand_l59_59001


namespace total_area_of_removed_triangles_l59_59470

theorem total_area_of_removed_triangles (s : ℝ) (x : ℝ)
  (hsq_area : s * s = 256)
  (hsq_side : s = 16)
  (htri_area : ∀ x, 4 * (x^2 / 2) = 2 * x^2):
  s = 16 → ∀ x, 2 * x^2 = 768 - 512 * real.sqrt 2 := 
sorry

end total_area_of_removed_triangles_l59_59470


namespace determine_dracula_alive_l59_59804

-- Define general conditions
def dracula_statements_are_false (s : Prop) : Prop := ¬s

-- Define specific questions that can be asked to determine if Dracula is alive
def question1 (answer : Prop) : Prop := answer ↔ (("ball" → "you are a human") → ("Dracula is alive"))
def question2 (answer : Prop) : Prop := (("ball" → "you are reliable") → ("Dracula is alive"))

-- The proof problem
theorem determine_dracula_alive (q1 q2 : Prop) (H : ∀ s, dracula_statements_are_false s) :
  (question1 q1 ↔ "Dracula is alive") ∨ (question2 q2 ↔ "Dracula is alive") :=
  sorry

end determine_dracula_alive_l59_59804


namespace reciprocal_of_neg_1_point_5_l59_59399

theorem reciprocal_of_neg_1_point_5 : (1 / (-1.5) = -2 / 3) :=
by
  sorry

end reciprocal_of_neg_1_point_5_l59_59399


namespace intersection_points_of_pairs_of_points_l59_59926

theorem intersection_points_of_pairs_of_points 
  (P : Fin 22 → Point) 
  (h_no_collinear : ∀ (A B C : Fin 22), ¬ Collinear (P A) (P B) (P C)) :
  ∃ pairs : Fin 11 → (Point × Point), 
  (∃ intersections : Finset Point, 
  intersections.card ≥ 5 ∧
  ∀ i : Fin 11, let (a, b) := pairs i in a ≠ b ∧ a ∈ P ∧ b ∈ P ∧ 
  (∃ j : Fin 11, i ≠ j ∧ 
    (intersects (line_through_points a b) (line_through_points (pairs j).fst (pairs j).snd)))) :=
sorry

end intersection_points_of_pairs_of_points_l59_59926


namespace number_of_coplanar_point_sets_l59_59169

-- Define the problem conditions
variables 
  (A B C D : Type) -- Vertices of the tetrahedron
  (M_AB M_AC M_AD M_BC M_BD M_CD : Type) -- Midpoints of the edges

-- Formalize the points involved
def P : Type := {A, B, C, D, M_AB, M_AC, M_AD, M_BC, M_BD, M_CD}

-- Define the proposition to be proven: Number of sets of four coplanar points
theorem number_of_coplanar_point_sets
  (h₁ : A ∈ P)
  (h₂ : B ∈ P)
  (h₃ : C ∈ P)
  (h₄ : D ∈ P)
  (h₅ : M_AB ∈ P)
  (h₆ : M_AC ∈ P)
  (h₇ : M_AD ∈ P)
  (h₈ : M_BC ∈ P)
  (h₉ : M_BD ∈ P)
  (h₁₀ : M_CD ∈ P) :
  ∃ (χ : ℕ), χ = 33 := sorry

end number_of_coplanar_point_sets_l59_59169


namespace set_of_points_satisfying_inequality_l59_59713

theorem set_of_points_satisfying_inequality :
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ |x| ≤ 1 ∧ |y| ≤ 1 ∧ xy ≤ 0} ∪
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x^2 + y^2 ≤ 1 ∧ xy > 0} =
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ |x| ≤ 1 ∧ |y| ≤ 1 ∧ sqrt(1 - x^2) * sqrt(1 - y^2) ≥ xy} :=
sorry

end set_of_points_satisfying_inequality_l59_59713


namespace exponent_reciprocal_evaluation_l59_59435

theorem exponent_reciprocal_evaluation : (4⁻¹ - 3⁻¹)⁻¹ = -12 := by
  have h_4 := (show 4⁻¹ = 1 / 4 by norm_num)
  have h_3 := (show 3⁻¹ = 1 / 3 by norm_num)
  rw [h_4, h_3]
  have h := (show (1 / 4 - 1 / 3)⁻¹ = -12 by sorry)
  exact h

end exponent_reciprocal_evaluation_l59_59435


namespace raised_bed_length_l59_59177

theorem raised_bed_length (beds height width num_planks planks_per_bed: ℕ) (height_cond: height = 2) (width_cond: width = 2) (plank_width: ℕ) (plank_length: ℕ) (used_planks: ℕ):
  beds = 10 → 
  height_cond →
  width_cond →
  plank_width = 1 →
  plank_length = 8 →
  num_planks = 50 →
  planks_per_bed = used_planks / beds →
  (2 * height * plank_width + used_planks = 4) →
  4 * beds = height → 
  has_add.add (2 * width) plank_length = 8 := 
begin
  sorry
end

end raised_bed_length_l59_59177


namespace three_half_planes_cover_full_plane_l59_59225

theorem three_half_planes_cover_full_plane
  (H1 H2 H3 H4 : set (ℝ × ℝ))
  (cover_plane : ∀ p : ℝ × ℝ, p ∈ H1 ∨ p ∈ H2 ∨ p ∈ H3 ∨ p ∈ H4) :
  ∃ (A B C : set (ℝ × ℝ)), (A = H1 ∨ A = H2 ∨ A = H3 ∨ A = H4) ∧ 
                            (B = H1 ∨ B = H2 ∨ B = H3 ∨ B = H4) ∧ 
                            (C = H1 ∨ C = H2 ∨ C = H3 ∨ C = H4) ∧
                            A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
                            (∀ p : ℝ × ℝ, p ∈ A ∨ p ∈ B ∨ p ∈ C) :=
sorry

end three_half_planes_cover_full_plane_l59_59225


namespace mother_duck_multiple_of_first_two_groups_l59_59825

variables (num_ducklings : ℕ) (snails_first_batch : ℕ) (snails_second_batch : ℕ)
          (total_snails : ℕ) (mother_duck_snails : ℕ)

-- Given conditions
def conditions : Prop :=
  num_ducklings = 8 ∧ 
  snails_first_batch = 3 * 5 ∧ 
  snails_second_batch = 3 * 9 ∧ 
  total_snails = 294 ∧ 
  total_snails = snails_first_batch + snails_second_batch + 2 * mother_duck_snails ∧ 
  mother_duck_snails > 0

-- Our goal is to prove that the mother duck finds 3 times the snails the first two groups of ducklings find
theorem mother_duck_multiple_of_first_two_groups (h : conditions num_ducklings snails_first_batch snails_second_batch total_snails mother_duck_snails) : 
  mother_duck_snails / (snails_first_batch + snails_second_batch) = 3 :=
by 
  sorry

end mother_duck_multiple_of_first_two_groups_l59_59825


namespace sum_of_real_solutions_l59_59422

theorem sum_of_real_solutions : ∑ x in {x : ℝ | |x^2 - 10x + 29| = 3}.to_finset 0 = 0 := by
  sorry

end sum_of_real_solutions_l59_59422


namespace time_saved_by_increasing_speed_l59_59698

theorem time_saved_by_increasing_speed (d v1 v2 : ℕ) (h_v1 : v1 = 60) (h_v2 : v2 = 50) (h_d : d = 1200) : 
    d / v2 - d / v1 = 4 := 
by
  rw [h_v1, h_v2, h_d]
  have h1 : 1200 / 60 = 20 := by norm_num
  have h2 : 1200 / 50 = 24 := by norm_num
  rw [h1, h2]
  norm_num
  done

end time_saved_by_increasing_speed_l59_59698


namespace probability_of_bayonet_base_on_third_try_is_7_over_120_l59_59560

noncomputable def probability_picking_bayonet_base_bulb_on_third_try : ℚ :=
  (3 / 10) * (2 / 9) * (7 / 8)

/-- Given a box containing 3 screw base bulbs and 7 bayonet base bulbs, all with the
same shape and power and placed with their bases down. An electrician takes one bulb
at a time without returning it. The probability that he gets a bayonet base bulb on his
third try is 7/120. -/
theorem probability_of_bayonet_base_on_third_try_is_7_over_120 :
  probability_picking_bayonet_base_bulb_on_third_try = 7 / 120 :=
by 
  sorry

end probability_of_bayonet_base_on_third_try_is_7_over_120_l59_59560


namespace large_doll_cost_is_8_l59_59320

-- Define the cost of the large monkey doll
def cost_large_doll : ℝ := 8

-- Define the total amount spent
def total_spent : ℝ := 320

-- Define the price difference between large and small dolls
def price_difference : ℝ := 4

-- Define the count difference between buying small dolls and large dolls
def count_difference : ℝ := 40

theorem large_doll_cost_is_8 
    (h1 : total_spent = 320)
    (h2 : ∀ L, L - price_difference = 4)
    (h3 : ∀ L, (total_spent / (L - 4)) = (total_spent / L) + count_difference) :
    cost_large_doll = 8 := 
by 
  sorry

end large_doll_cost_is_8_l59_59320


namespace average_after_adding_ten_l59_59796

theorem average_after_adding_ten (avg initial_sum new_mean : ℕ) (n : ℕ) (h1 : n = 15) (h2 : avg = 40) (h3 : initial_sum = n * avg) (h4 : new_mean = (initial_sum + n * 10) / n) : new_mean = 50 := 
by
  sorry

end average_after_adding_ten_l59_59796


namespace quadratic_inequality_solution_l59_59401

variable (a x : ℝ)

-- Define the quadratic expression and the inequality condition
def quadratic_inequality (a x : ℝ) : Prop := 
  x^2 - (2 * a + 1) * x + a^2 + a < 0

-- Define the interval in which the inequality holds
def solution_set (a x : ℝ) : Prop :=
  a < x ∧ x < a + 1

-- The main statement to be proven
theorem quadratic_inequality_solution :
  ∀ a x, quadratic_inequality a x ↔ solution_set a x :=
sorry

end quadratic_inequality_solution_l59_59401


namespace correct_mean_l59_59434

-- Definitions of conditions
def n : ℕ := 30
def mean_incorrect : ℚ := 140
def value_correct : ℕ := 145
def value_incorrect : ℕ := 135

-- The statement to be proved
theorem correct_mean : 
  let S_incorrect := mean_incorrect * n
  let Difference := value_correct - value_incorrect
  let S_correct := S_incorrect + Difference
  let mean_correct := S_correct / n
  mean_correct = 140.33 := 
by
  sorry

end correct_mean_l59_59434


namespace train_length_l59_59799

theorem train_length
  (speed_km_hr : ℕ)
  (time_sec : ℕ)
  (length_train : ℕ)
  (length_platform : ℕ)
  (h_eq_len : length_train = length_platform)
  (h_speed : speed_km_hr = 108)
  (h_time : time_sec = 60) :
  length_train = 900 :=
by
  sorry

end train_length_l59_59799


namespace min_distance_pq_to_c3_l59_59252

def C1_polar_eq (ρ θ : ℝ) : Prop :=
  ρ^2 + 8*ρ*Real.cos θ - 6*ρ*Real.sin θ + 24 = 0

def C2_param_eq (θ : ℝ) : ℝ × ℝ :=
  (8 * Real.cos θ, 3 * Real.sin θ)

def P_coords: ℝ × ℝ :=
  (-4, 4)

def C3_line_eq (p : ℝ × ℝ) : Prop :=
  p.1 - 2 * p.2 - 7 = 0

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def distance_point_to_line (M : ℝ × ℝ) : ℝ :=
  (Real.sqrt 5 / 5) * Real.abs(4 * M.1 / 2 - 3 * M.2 / 2 - 13)

theorem min_distance_pq_to_c3 (θ φ : ℝ) (hcos : Real.cos φ = 4/5) (hsin : Real.sin φ = 3/5) :
  ∃ θ, distance_point_to_line (midpoint P_coords (C2_param_eq θ)) = 8 * Real.sqrt 5 / 5 := sorry

end min_distance_pq_to_c3_l59_59252


namespace sum_neg50_to_75_eq_1575_l59_59497

def sum_integers_from_neg50_to_75 : ℤ :=
  (∑ i in Finset.Icc (-50) 75, i)

theorem sum_neg50_to_75_eq_1575 : sum_integers_from_neg50_to_75 = 1575 := 
  sorry

end sum_neg50_to_75_eq_1575_l59_59497


namespace average_all_results_l59_59798

theorem average_all_results (s₁ s₂ : ℤ) (n₁ n₂ : ℤ) (h₁ : n₁ = 60) (h₂ : n₂ = 40) (avg₁ : s₁ / n₁ = 40) (avg₂ : s₂ / n₂ = 60) : 
  ((s₁ + s₂) / (n₁ + n₂) = 48) :=
sorry

end average_all_results_l59_59798


namespace increasing_intervals_min_b_for_10_zeros_l59_59961

open Real

def f (x : ℝ) := 2 * sin (2 * x - π / 3)

def g (x : ℝ) := 2 * sin (2 * x) + 1

theorem increasing_intervals (k : ℤ) : 
  ∀ x : ℝ, k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12 → 
  ∃ δ > (0 : ℝ), ∀ y : ℝ, ∥y - x∥ < δ → f y > f x := 
sorry

theorem min_b_for_10_zeros : 
  ∃ b : ℝ, b ≥ 59 * π / 12 ∧ 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ b → 
  ∃ n : ℤ, n ≥ 10 ∧ g x = 0 := 
sorry

end increasing_intervals_min_b_for_10_zeros_l59_59961


namespace retailer_profit_percentage_l59_59147

-- Conditions:
def wholesale_price : ℝ := 126
def retail_price : ℝ := 168
def discount_percentage : ℝ := 0.10

-- Calculate the selling price after the discount
def selling_price : ℝ := retail_price * (1 - discount_percentage)

-- Calculate the profit made by the retailer
def profit : ℝ := selling_price - wholesale_price

-- Calculate the percentage profit made by the retailer
def profit_percentage : ℝ := (profit / wholesale_price) * 100

-- The theorem to be proven
theorem retailer_profit_percentage : profit_percentage = 20 := by
  -- using the definitions provided
  have h1 : selling_price = retail_price * (1 - discount_percentage) := rfl
  have h2 : profit = selling_price - wholesale_price := rfl
  have h3 : profit_percentage = (profit / wholesale_price) * 100 := rfl
  sorry

end retailer_profit_percentage_l59_59147


namespace average_sleep_probability_l59_59619

theorem average_sleep_probability : 
  let (x y : ℝ) in 
  (x, y) ∈ set.Icc (6 : ℝ) (9 : ℝ) × set.Icc (6 : ℝ) (9 : ℝ) →
  (measure_theory.measure.prod 
    (measure_theory.measure.uniform 6 9) 
    (measure_theory.measure.uniform 6 9)) {p | (fst p + snd p) / 2 ≥ 7} = (7 : ℝ) / 9 :=
sorry

end average_sleep_probability_l59_59619


namespace exists_function_satisfying_conditions_l59_59426

theorem exists_function_satisfying_conditions :
  ∃ f : ℝ → ℝ, 
  (∀ x : ℝ, f(x+1) = f(-x+1)) ∧
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f(a) = 0 ∧ f(b) = 0 ∧ f(c) = 0) :=
by
  have f : ℝ → ℝ := λ x, (x - 1) ^ 2 - |x - 1|
  use f
  split
  . intro x
    simp
    sorry
  . use 1
    use 1 - sqrt 2
    use 1 + sqrt 2
    split; -- checks if 1 ≠ 1 - sqrt 2 ∧ 1 ≠ 1 + sqrt 2 ∧ 1 - sqrt 2 ≠ 1 + sqrt 2
      . linarith
      . split
        . linarith
        . split
          . rw sub_self
            rw add_eq_zero_iff_eq_neg
            rlinarith
  . split
    . exact rfl -- f(1) = 0
    . rw [(sub_add_cancel _ _).symm, sub_self]
      exact abs_of_nonneg rfl.ge
    . rw [sub_add_cancel _ _], abs_of_nonneg (add_nonneg rfl.le rfl.le)

end exists_function_satisfying_conditions_l59_59426


namespace projection_vector_l59_59541

open Real

noncomputable def proj (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
  let normSq := v.1^2 + v.2^2 + v.3^2
  (dot / normSq * v.1, dot / normSq * v.2, dot / normSq * v.3)

theorem projection_vector :
  proj (3, -4, 1) (1, 2, 2) = (-1/3, -2/3, -2/3) :=
by
  sorry

end projection_vector_l59_59541


namespace harriet_trip_time_l59_59106

theorem harriet_trip_time
  (speed_AB : ℕ := 100)
  (speed_BA : ℕ := 150)
  (total_trip_time : ℕ := 5)
  (time_threshold : ℕ := 180) :
  let D := (speed_AB * speed_BA * total_trip_time) / (speed_AB + speed_BA)
  let time_AB := D / speed_AB
  let time_AB_min := time_AB * 60
  time_AB_min = time_threshold :=
by
  sorry

end harriet_trip_time_l59_59106


namespace find_fraction_l59_59895

theorem find_fraction : ∃ x : ℚ, x = 7/15 ∧ (3/5) / (6/7) = x / (2/3) :=
by
  use 7/15
  split
  sorry

end find_fraction_l59_59895


namespace probability_two_flies_swept_l59_59006

/-- Defining the positions of flies on the clock -/
inductive positions : Type
| twelve   | three   | six   | nine

/-- Probability that the minute hand sweeps exactly two specific positions after 20 minutes -/
theorem probability_two_flies_swept (flies : list positions) (time : ℕ) :
  (flies = [positions.twelve, positions.three, positions.six, positions.nine]) →
  (time = 20) →
  (probability_sweeps_two_flies flies time = 1 / 3) := sorry

end probability_two_flies_swept_l59_59006


namespace transformed_line_polar_l59_59640

-- Define the original line equation
def original_line (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop := 
  x' = (1/2) * x ∧ y' = 4 * y

-- Define the new line equation after transformation
def new_line (x' y' : ℝ) : Prop := 4 * x' - y' = 4

-- Conversion to polar coordinates
def polar_coords (ρ θ x y : ℝ) : Prop := 
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Define the polar equation we aim to prove
def polar_equation (ρ θ : ℝ) : Prop :=
  4 * ρ * Real.cos θ - ρ * Real.sin θ = 4

-- The main theorem we want to prove
theorem transformed_line_polar (x y x' y' ρ θ : ℝ) :
  original_line x y → 
  transformation x y x' y' → 
  polar_coords ρ θ x' y' → 
  polar_equation ρ θ :=
begin
  intros h_original h_transformation h_polar_coords,
  sorry -- proof not required per instructions
end

end transformed_line_polar_l59_59640


namespace hitting_four_times_impossible_coin_toss_random_event_sum_of_interior_angles_of_triangle_is_180_l59_59123

-- Conditions
def shoots_three_times (hits: ℕ) : Prop := hits ≤ 3
def coin_toss : Prop := true
def triangle_sum_of_angles (angles: list ℕ) : Prop := angles.sum = 180

-- Proof problem statements
theorem hitting_four_times_impossible (hits: ℕ) (h : shoots_three_times hits) : hits ≠ 4 :=
by {
  -- Sorry placeholder for proof 
  sorry
}

theorem coin_toss_random_event : 
  random_event (outcome: bool) : Prop := (outcome = tt ∨ outcome = ff) :=
by {
  -- Sorry placeholder for proof
  sorry
}

theorem sum_of_interior_angles_of_triangle_is_180 (angles: list ℕ) (h : triangle_sum_of_angles angles) : angles.sum = 180 :=
by {
  -- Sorry placeholder for proof
  exact h
}

end hitting_four_times_impossible_coin_toss_random_event_sum_of_interior_angles_of_triangle_is_180_l59_59123


namespace ffour_times_l59_59547

def f (z : ℂ) : ℂ :=
  if z.im = 0 then z^3 else z^2 - 1

theorem ffour_times (z : ℂ) (h : z = 2 + I) : 
  f (f (f (f z))) = 114688 + 393216 * I :=
by
  sorry

end ffour_times_l59_59547


namespace simplify_fraction_l59_59720

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) : 
  1 / x - 1 / (x - 1) = -1 / (x * (x - 1)) :=
by
  sorry

end simplify_fraction_l59_59720


namespace minimum_rectangle_perimeter_l59_59467

-- Definitions of the conditions
def valid_rectangle (a b : ℕ) : Prop :=
  ∃ width height : ℕ, 
  height = 11 * a ∧ 
  width = 15 * a ∧ 
  a = 1 ∧ 
  b = 3

-- Definition of the perimeter calculation
def perimeter (width height : ℕ) : ℕ := 2 * (width + height)

-- Lean statement
theorem minimum_rectangle_perimeter : 
  ∀ a b, valid_rectangle a b → 
  perimeter (15 * a) (11 * a) = 52 := 
by 
  intros _ _ valid_rect
  cases valid_rect with width valid_height 
  cases valid_height with height a_eq_width 
  cases a_eq_width with width_eq_a height_eq_3a 
  rw [width_eq_a, height_eq_3a, height_eq_3a],
  sorry

end minimum_rectangle_perimeter_l59_59467


namespace simplify_3_375_to_fraction_l59_59107

def simplified_fraction_of_3_375 : ℚ := 3.375

theorem simplify_3_375_to_fraction : simplified_fraction_of_3_375 = 27 / 8 := 
by
  sorry

end simplify_3_375_to_fraction_l59_59107


namespace solution_for_equation_l59_59207

theorem solution_for_equation (m n : ℕ) (h : 0 < m ∧ 0 < n ∧ 2 * m^2 = 3 * n^3) :
  ∃ k : ℕ, 0 < k ∧ m = 18 * k^3 ∧ n = 6 * k^2 :=
by sorry

end solution_for_equation_l59_59207


namespace sum_of_solutions_of_fx_eq_neg3_l59_59682

def f (x : ℝ) : ℝ :=
  if x < -2 then 3 * x + 4 else -x^2 - 2 * x + 2

theorem sum_of_solutions_of_fx_eq_neg3 :
  (∃ x : ℝ, f x = -3) ∧ (f (-7/3) = -3) ∧ (f (-1 + 2 * Real.sqrt 6) = -3) →
  (-7/3) + (-1 + 2 * Real.sqrt 6) = (-10/3) + 2 * Real.sqrt 6 := by
  sorry

end sum_of_solutions_of_fx_eq_neg3_l59_59682


namespace sum_of_number_and_reverse_l59_59040

theorem sum_of_number_and_reverse :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → ((10 * a + b) - (10 * b + a) = 7 * (a + b)) → a = 8 * b → 
  (10 * a + b) + (10 * b + a) = 99 :=
by
  intros a b conditions eq diff
  sorry

end sum_of_number_and_reverse_l59_59040


namespace area_of_field_l59_59394

theorem area_of_field (w l A : ℝ) 
    (h1 : l = 2 * w + 35) 
    (h2 : 2 * (w + l) = 700) : 
    A = 25725 :=
by sorry

end area_of_field_l59_59394


namespace white_to_brown_eggs_ratio_l59_59692

-- Define variables W and B (the initial numbers of white and brown eggs respectively)
variable (W B : ℕ)

-- Conditions: 
-- 1. All 5 brown eggs survived.
-- 2. Total number of eggs after dropping is 12.
def egg_conditions : Prop :=
  B = 5 ∧ (W + B) = 12

-- Prove the ratio of white eggs to brown eggs is 7/5 given these conditions.
theorem white_to_brown_eggs_ratio (h : egg_conditions W B) : W / B = 7 / 5 :=
by 
  sorry

end white_to_brown_eggs_ratio_l59_59692


namespace max_b_value_l59_59821

noncomputable theory

open BigOperators

def is_integer (a : ℚ) : Prop := ∃ n : ℤ, a = n
def avoids_lattice_points (m b : ℚ) : Prop :=
  ∀ (x : ℤ), (0 < x ∧ x ≤ 50) → ¬ is_integer (m * x + 3)

theorem max_b_value :
  ∃ b : ℚ, (b = 17 / 51) ∧ ∀ m : ℚ, (1 / 3) < m ∧ m < b → avoids_lattice_points m b :=
begin
  sorry
end

end max_b_value_l59_59821


namespace bg_eq_ch_eq_ia_l59_59932

-- Defining the mathematical entities and conditions.
variables {A B C D E F G H I: Type*} [has_midpoint G H I] [equilateral_triangle A B D] [equilateral_triangle B C E] [equilateral_triangle C A F]

-- Stating the theorem
theorem bg_eq_ch_eq_ia (h1 : midpoint D E G) (h2 : midpoint E F H) (h3 : midpoint F D I)
  (h4 : equilateral_triangle A B D) (h5 : equilateral_triangle B C E) (h6 : equilateral_triangle C A F)
  : dist B G = dist C H ∧ dist C H = dist I A :=
  sorry -- Omit the proof.

end bg_eq_ch_eq_ia_l59_59932


namespace prime_sum_product_l59_59066

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∈ (finset.range n).filter (λ x, x > 1), n % m ≠ 0

theorem prime_sum_product :
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 91 ∧ p * q = 178 :=
begin
  sorry
end

end prime_sum_product_l59_59066


namespace general_term_formula_max_sum_n_l59_59954

-- Let's denote the conditions as variables and hypotheses.

variable (A : ℝ) (a b c : ℝ)
hypothesis (h1 : cos A = (b^2 + c^2 - a^2) / (2 * b * c) = 1 / 2)
hypothesis (h2 : A = π / 3)
hypothesis (h3 : a = 2 * sqrt 5)
hypothesis (h4 : bc <= 20)
hypothesis (h5 : a > 0)
hypothesis (h6 : (a_1 + 2 * d = -5))
hypothesis (h7 : (a_1 + 9 * d = -9))
hypothesis (deltapos : a^2 - 4 * a < 0)

-- Part (1) Prove the general term formula
theorem general_term_formula : ∀ (n : ℕ), a_n = 11 - 2 * n :=
begin
  sorry
end

-- Part (2) Prove the maximum value of the sum of the first n terms
theorem max_sum_n : max (∑ i in range (n + 1), 11 - 2 * i) = 25 :=
begin
  sorry
end

end general_term_formula_max_sum_n_l59_59954


namespace height_less_than_two_after_six_bounces_l59_59443

noncomputable def geometric_height (h₀ : ℝ) (r : ℝ) (k : ℕ) : ℝ :=
  h₀ * (r ^ k)

theorem height_less_than_two_after_six_bounces :
  ∀ (h₀ : ℝ) (r : ℝ), h₀ = 16 → r = 2 / 3 → 
  ∃ k : ℕ, k = 6 ∧ geometric_height h₀ r k < 2 :=
by
  intros h₀ r h₀_eq r_eq
  use 6
  rw [geometric_height, h₀_eq, r_eq]
  have : (2 / 3) ^ 6 < 2 / 16 := sorry
  linarith

end height_less_than_two_after_six_bounces_l59_59443


namespace matrix_condition_l59_59504

noncomputable theory

open Matrix

variables {x y z : ℝ}

def N : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2*x, -y],
    ![-z, x, y],
    ![z, x, -y]]

theorem matrix_condition (h : (N.transpose ⬝ N) = (1 : Matrix (Fin 3) (Fin 3) ℝ)) :
  x^2 + y^2 + z^2 = 1 :=
sorry

end matrix_condition_l59_59504


namespace workByFortyPercent_l59_59293

-- Definitions based on the conditions
def totalWork : ℝ := 100.0
def productiveEmployees1 : ℝ := 20.0 / 100.0 * totalWork
def workByProductiveEmployees1 : ℝ := 80.0 / 100.0 * totalWork

-- Helper definitions to split the next 20% of most productive employees
def remainingEmployees : ℝ := 100.0 - productiveEmployees1
def nextProductiveEmployees2 : ℝ := 20.0 / 100.0 * remainingEmployees
def workByNextProductiveEmployees2 : ℝ := nextProductiveEmployees2 / remainingEmployees * (totalWork - workByProductiveEmployees1)

-- Theorem we need to prove
theorem workByFortyPercent : (workByProductiveEmployees1 + workByNextProductiveEmployees2) = 85 := by
  sorry

end workByFortyPercent_l59_59293


namespace proof_area_AMB_CMD_l59_59790

-- Definitions based on conditions
def chord_len (θ : ℝ) (R : ℝ) : ℝ := 2 * R * Real.sin (θ / 2)

-- Given lengths of chords AB and CD
def AB_len (R : ℝ) : ℝ := chord_len 120 R
def CD_len (R : ℝ) : ℝ := chord_len 90 R

-- Similarity ratio of the triangles using the lengths derived
def sim_ratio (R : ℝ) : ℝ := AB_len R / CD_len R 

-- Areas are represented as variables to be proven
variables (area_AMB area_CMD : ℝ)

-- Total area condition
def total_area : Prop := area_AMB + area_CMD = 100

-- Similarity area ratio condition
def area_ratio : Prop := area_AMB / area_CMD = (sim_ratio 1) ^ 2

-- Prove that the areas are correct given the conditions
theorem proof_area_AMB_CMD (R : ℝ) (h1 : total_area) (h2 : area_ratio) : 
  area_AMB = 60 ∧ area_CMD = 40 :=
  sorry

end proof_area_AMB_CMD_l59_59790


namespace angle_C_value_range_a2_b2_l59_59645

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {triangle_ABC : ∃ (A B C : ℝ) (a b c : ℝ), (A + B + C = Real.pi) ∧ (a > 0) ∧ (b > 0) ∧ (c > 0)}

-- Part 1: Prove that angle C is π/3 given the tangent equation
theorem angle_C_value (h : ∃ (A B : ℝ), (sqrt 3) * (Real.tan A) * (Real.tan B) - (Real.tan A) - (Real.tan B) = sqrt 3) : 
    C = Real.pi / 3 :=
  sorry

-- Part 2: Prove the range of values for a^2 + b^2 given c = 2, angle C = π/3, and the triangle is acute
theorem range_a2_b2 (h1 : C = Real.pi / 3) (hc : c = 2) (acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
    (20 / 3) < a^2 + b^2 ∧ a^2 + b^2 <= 8 :=
  sorry

end angle_C_value_range_a2_b2_l59_59645


namespace four_digit_even_numbers_formed_without_repeating_digits_l59_59751

theorem four_digit_even_numbers_formed_without_repeating_digits : 
  let digits := {1, 2, 3, 4, 5}
  let even_digits := {2, 4}
  let other_digits := {1, 3, 5}
  number_of_valid_numbers = 48 :=
sorry

end four_digit_even_numbers_formed_without_repeating_digits_l59_59751


namespace fruit_seller_price_l59_59454

theorem fruit_seller_price 
  (CP SP SP_profit : ℝ)
  (h1 : SP = CP * 0.88)
  (h2 : SP_profit = CP * 1.20)
  (h3 : SP_profit = 21.818181818181817) :
  SP = 16 := 
by 
  sorry

end fruit_seller_price_l59_59454


namespace ratio_and_radii_l59_59430

-- Define the geometric entities
variables (ℓ1 ℓ2 : Line) (ω1 ω2 : Circle)
variables (O1 O2 A B C D E : Point)
variables (R1 R2 : ℝ)
variables (α : ℝ)

-- Define the conditions
-- Two parallel lines and their tangency to the circle ω1
axiom l1_parallel_l2 : ℓ1 ∥ ℓ2
axiom l1_tangent_ω1_A : tangent ℓ1 ω1 A
axiom l2_tangent_ω1_B : tangent ℓ2 ω1 B

-- Circle ω2's tangency and intersections
axiom ω2_tangent_l1_D : tangent ω2 ℓ1 D
axiom ω2_intersects_l2_B_E : intersects ω2 ℓ2 B E
axiom ω2_intersects_ω1_C : intersects ω2 ω1 [C]

-- Given area ratio
axiom area_ratio_quadrilateral_triangle : 
  area_quadrilateral B O1 C O2 / area_triangle O2 B E = 3/2

-- Given BD = 1
axiom BD_eq_1 : dist B D = 1

-- Prove the ratio of radii and their specific values
theorem ratio_and_radii : R2 / R1 = 4 / 3 ∧ R1 = sqrt 3 / 4 ∧ R2 = 1 / sqrt 3 :=
by sorry

end ratio_and_radii_l59_59430


namespace part1_part2_part3_l59_59571

-- Definition for Part 1
def sequence_a (n : ℕ) : ℕ := 2^(3 - n)
def S (n : ℕ) : ℝ := 8 * (1 - (1 / 2^n))

-- Proof statement for Part 1
theorem part1 (n : ℕ) (hn : n > 0) : 
  S n = ∑ k in Finset.range n, sequence_a (k + 1) := 
sorry

-- Definition for Part 2
def S_part2 (n : ℕ) : ℝ :=
  if even n then (n^2 + 8n) / 4 else (n^2 + 8n + 7) / 4

-- Proof statement for Part 2
theorem part2 (n : ℕ) : 
  S_part2 n = ∑ k in Finset.range n, (sequence_a k * sequence_a (k + 1)) :=
sorry

-- Definition for Part 3
def S_part3 (n : ℕ) : ℝ := 3 * ∑ k in Finset.range n, sequence_a (k + 1)

-- Proof statement for Part 3
theorem part3 (n : ℕ) (wn : ∃ m, 3*n - 1 = 8*m) : 
  ∃ m : ℕ, (3*n - 1) % 8 = 0 :=
sorry

end part1_part2_part3_l59_59571


namespace mary_total_spent_l59_59695

def store1_shirt : ℝ := 13.04
def store1_jacket : ℝ := 12.27
def store2_shoes : ℝ := 44.15
def store2_dress : ℝ := 25.50
def hat_price : ℝ := 9.99
def discount : ℝ := 0.10
def store4_handbag : ℝ := 30.93
def store4_scarf : ℝ := 7.42
def sunglasses_price : ℝ := 20.75
def sales_tax : ℝ := 0.05

def store1_total : ℝ := store1_shirt + store1_jacket
def store2_total : ℝ := store2_shoes + store2_dress
def store3_total : ℝ := 
  let hat_cost := hat_price * 2
  let discount_amt := hat_cost * discount
  hat_cost - discount_amt
def store4_total : ℝ := store4_handbag + store4_scarf
def store5_total : ℝ := 
  let tax := sunglasses_price * sales_tax
  sunglasses_price + tax

def total_spent : ℝ := store1_total + store2_total + store3_total + store4_total + store5_total

theorem mary_total_spent : total_spent = 173.08 := sorry

end mary_total_spent_l59_59695


namespace maximum_triangle_area_l59_59091

theorem maximum_triangle_area (a b c : ℝ) (h_a : 0 ≤ a ∧ a ≤ 1) (h_b : 1 ≤ b ∧ b ≤ 2) (h_c : 2 ≤ c ∧ c ≤ 3) :
  ∃ S : ℝ, S = 1 ∧ (∀ α : ℝ, sin α ≤ 1) ∧ S = (1 / 2) * a * b * sin (π / 2) ∧ c = sqrt (a ^ 2 + b ^ 2) := 
by
  sorry

end maximum_triangle_area_l59_59091


namespace expand_product_l59_59517

theorem expand_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * (7 / x^3 - 14 * x^4) = 3 / x^3 - 6 * x^4 :=
by
  sorry

end expand_product_l59_59517


namespace cannot_form_triangle_iff_sum_two_less_than_third_l59_59485

theorem cannot_form_triangle_iff_sum_two_less_than_third (a b c : ℕ) : 
  ¬ (a + b > c ∧ a + c > b ∧ b + c > a) ↔ a + b ≤ c := 
by 
  sorry

example : ¬(3 + 4 > 8 ∧ 3 + 8 > 4 ∧ 4 + 8 > 3) :=
begin
  apply cannot_form_triangle_iff_sum_two_less_than_third,
  exact nat.add_le_iff.2 (lt_of_lt_of_le (nat.add_lt_add_left (by norm_num) _) (by norm_num)),
end

end cannot_form_triangle_iff_sum_two_less_than_third_l59_59485


namespace num_roots_x_eq_fx_in_interval_l59_59259

noncomputable def f : ℝ → ℝ :=
λ x, if h : -2 ≤ x ∧ x ≤ 0 then x^2 + 2*x else
     if h : 0 < x ∧ x <= 2 then f (x - 1) + 1 else 0

theorem num_roots_x_eq_fx_in_interval : 
  (∃ n : ℕ, n = 4) ∧
  ∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), x - f x = 0 ↔ ∃ y : ℝ, y = x :=
begin
  sorry
end

end num_roots_x_eq_fx_in_interval_l59_59259


namespace paint_cost_per_liter_l59_59346

def cost_brush : ℕ := 20
def cost_canvas : ℕ := 3 * cost_brush
def min_liters : ℕ := 5
def total_earning : ℕ := 200
def total_profit : ℕ := 80
def total_cost : ℕ := total_earning - total_profit

theorem paint_cost_per_liter :
  (total_cost = cost_brush + cost_canvas + (5 * 8)) :=
by
  sorry

end paint_cost_per_liter_l59_59346


namespace area_of_circle_l59_59635

noncomputable def circle_area_eq_5_pi : Prop :=
  let equation := ∀ x y : ℝ, 4 * x^2 + 4 * y^2 - 8 * x + 24 * y + 60 = 0
  in ∃ r : ℝ, (r = real.sqrt 5) ∧ (∀ A : ℝ, A = real.pi * r^2 → A = 5 * real.pi)

theorem area_of_circle : circle_area_eq_5_pi :=
by
  sorry

end area_of_circle_l59_59635


namespace incorrect_statement_C_l59_59105

theorem incorrect_statement_C :
  (-(-3) = 3) ∧ (|2| = | -2 |) ∧ (0 > | -1 |) ∧ (-2 > -3) → ¬ (0 > | -1 |) :=
by
  sorry

end incorrect_statement_C_l59_59105


namespace probability_two_flies_swept_away_l59_59012

-- Defining the initial conditions: flies at 12, 3, 6, and 9 o'clock positions
def flies_positions : List ℕ := [12, 3, 6, 9]

-- The problem statement
theorem probability_two_flies_swept_away : 
  (let favorable_intervals := 20 in
   let total_intervals := 60 in
   favorable_intervals / total_intervals = 1 / 3) :=
by
  sorry

end probability_two_flies_swept_away_l59_59012


namespace no_solution_abs_eq_l59_59983

theorem no_solution_abs_eq : ∀ x : ℝ, |x-1| = |2x-4| + |x-5| → false :=
by
  sorry

end no_solution_abs_eq_l59_59983


namespace angle_DEF_eq_90_l59_59936

variables (A B C D E F : Type) [Square A B C D] (h1 : E ∈ Line A C) (h2 : AE > EC) (h3 : F ∈ Line A B) (h4 : F ≠ B) (h5 : EF = DE)

theorem angle_DEF_eq_90 [Square A B C D] (E : Type) (h1 : E ∈ Line A C) (h2 : AE > EC) (F : Type) (h3 : F ∈ Line A B) (h4 : F ≠ B) (h5 : EF = DE) : angle DEF = 90 :=
by
  sorry

end angle_DEF_eq_90_l59_59936


namespace sum_arithmetic_seq_l59_59238

-- Definition of the arithmetic sequence
def arithmetic_sequence (a d n : ℕ) : ℤ := a + (n-1) * d

-- Given conditions
def a1 : ℤ := -1
def a2 : ℤ := 2
def d : ℤ := a2 - a1
def a4 : ℤ := arithmetic_sequence a1 d 4
def a5 : ℤ := arithmetic_sequence a1 d 5

-- Problem statement
theorem sum_arithmetic_seq : a4 + a5 = 19 :=
by
  rw [arithmetic_sequence, arithmetic_sequence]
  sorry

end sum_arithmetic_seq_l59_59238


namespace number_of_planes_through_two_l59_59693

-- Define the properties of the lines and planes
variable {Line : Type} [nonempty Line]
variable {Plane : Type} [nonempty Plane]
variable (a b c : Line)

-- Define the conditions
def pairwise_parallel : Prop := parallel a b ∧ parallel b c ∧ parallel a c
def not_coplanar : Prop := ¬ (∃ (P : Plane), (a ∈ P ∧ b ∈ P ∧ c ∈ P))

-- Define the question translated into Lean
def number_of_planes (a b c : Line) (hparallel : pairwise_parallel) (hnotcoplanar : not_coplanar) : ℕ := 3

-- Create a theorem statement to encapsulate the problem
theorem number_of_planes_through_two (a b c : Line) (hparallel : pairwise_parallel) (hnotcoplanar : not_coplanar) : number_of_planes a b c hparallel hnotcoplanar = 3 :=
sorry

end number_of_planes_through_two_l59_59693


namespace simplify_expression_l59_59372

theorem simplify_expression:
  (a = 2) ∧ (b = 1) →
  - (1 / 3 : ℚ) * (a^3 * b - a * b) 
  + a * b^3 
  - (a * b - b) / 2 
  - b / 2 
  + (1 / 3 : ℚ) * (a^3 * b) 
  = (5 / 3 : ℚ) := by 
  intros h
  simp [h.1, h.2]
  sorry

end simplify_expression_l59_59372


namespace next_term_sequence_l59_59567

def sequence (n : ℕ) : ℤ := if n % 2 = 1 then (4 * (n / 2) + 1) else -(4 * (n / 2 + 1) - 1)

theorem next_term_sequence : sequence 6 = -21 :=
by sorry

end next_term_sequence_l59_59567


namespace sum_of_good_indices_nonneg_l59_59325

def is_good_index (a : ℕ → ℝ) (n m k : ℕ) : Prop :=
∃ l, 1 ≤ l ∧ l ≤ m ∧ (finset.sum (finset.range l) (λ i, a (k + i) % n) ≥ 0)

def good_indices (a : ℕ → ℝ) (n m : ℕ) : finset ℕ :=
(finset.range n).filter (λ k, is_good_index a n m k)

theorem sum_of_good_indices_nonneg (a : ℕ → ℝ) (n m : ℕ) (h_m_lt_n : m < n):
  0 ≤ finset.sum (good_indices a n m) (λ k, a k) :=
sorry

end sum_of_good_indices_nonneg_l59_59325


namespace find_ordered_pair_l59_59377

-- This definition encapsulates the conditions given in the problem
def problem_conditions (a b : ℝ) [Nonzero a] [Nonzero b] : Prop :=
  let p := a + b = -b in -- first condition
  let q := ab = a in     -- second condition
  p ∧ q

-- This is our main statement, asserting the correct answer given the conditions
theorem find_ordered_pair (a b : ℝ) [Nonzero a] [Nonzero b] :
  problem_conditions a b → (a, b) = (-2, 1) :=
sorry

end find_ordered_pair_l59_59377


namespace final_image_of_transformations_l59_59747

def initial_position := (base: ℝ × ℝ, stem: ℝ × ℝ)
def rotate_180 (pos: initial_position) : initial_position :=
  ((-pos.base.1, -pos.base.2), (-pos.stem.1, -pos.stem.2))

def reflect_y (pos: initial_position) : initial_position :=
  ((-pos.base.1, pos.base.2), (-pos.stem.1, pos.stem.2))

def rotate_90 (pos: initial_position) : initial_position :=
  ((pos.base.2, -pos.base.1), (pos.stem.2, -pos.stem.1))

theorem final_image_of_transformations :
  let initial := ((-1, 0), (0, -1)) in
  let after_180 := rotate_180 initial in
  let after_reflection := reflect_y after_180 in
  let final := rotate_90 after_reflection in
  final = ((0, -1), (-1, 0)) :=
by
  sorry

end final_image_of_transformations_l59_59747


namespace tangent_line_range_l59_59959

noncomputable def f (x : ℝ) := log x
noncomputable def f' (x : ℝ) := 1 / x
noncomputable def k (x0 : ℝ) := 1 / x0
noncomputable def b (x0 : ℝ) := log x0 - 1
noncomputable def g (x : ℝ) := log x - 1 + 1 / x

theorem tangent_line_range (x0 : ℝ) (hx0 : x0 > 0) :
  k x0 + b x0 ≥ 0 :=
begin
  unfold k b,
  unfold f' at hx0,
  have h_g_x0 : g x0 = log x0 - 1 + 1 / x0,
    by unfold g,
  have : ∀ x > 0, diff_cont g,
    from sorry,
  have : argmin x > 0, g x = 1,
    from sorry,
  have h_g_min : g 1 = 0,
    from sorry,
  have h_g_monotone : monotone_on g (Ioi 1) ∧ monotone_on g (Iio 1),
    from sorry,
  show log x0 - 1 + 1 / x0 ≥ 0,
    from sorry,
end

end tangent_line_range_l59_59959


namespace age_difference_l59_59822

theorem age_difference (M S : ℕ) (h1 : S = 16) (h2 : M + 2 = 2 * (S + 2)) : M - S = 18 :=
by
  sorry

end age_difference_l59_59822


namespace farm_transaction_difference_l59_59351

theorem farm_transaction_difference
  (x : ℕ)
  (h_initial : 6 * x - 15 > 0) -- Ensure initial horses are enough to sell 15
  (h_ratio_initial : 6 * x = x * 6)
  (h_ratio_final : (6 * x - 15) = 3 * (x + 15)) :
  (6 * x - 15) - (x + 15) = 70 :=
by
  sorry

end farm_transaction_difference_l59_59351


namespace paper_strip_total_covered_area_l59_59218

theorem paper_strip_total_covered_area :
  let length := 12
  let width := 2
  let strip_count := 5
  let overlap_per_intersection := 4
  let intersection_count := 10
  let area_per_strip := length * width
  let total_area_without_overlap := strip_count * area_per_strip
  let total_overlap_area := intersection_count * overlap_per_intersection
  total_area_without_overlap - total_overlap_area = 80 := 
by
  sorry

end paper_strip_total_covered_area_l59_59218


namespace find_x_y_l59_59735

theorem find_x_y 
  (x y : ℝ) 
  (h1 : (15 + 30 + x + y) / 4 = 25) 
  (h2 : x = y + 10) :
  x = 32.5 ∧ y = 22.5 := 
by 
  sorry

end find_x_y_l59_59735


namespace nina_jerome_age_ratio_l59_59659

variable (N J L : ℕ)

theorem nina_jerome_age_ratio (h1 : L = N - 4) (h2 : L + N + J = 36) (h3 : L = 6) : N / J = 1 / 2 := by
  sorry

end nina_jerome_age_ratio_l59_59659


namespace probability_circle_or_square_l59_59708

theorem probability_circle_or_square (total_figures : ℕ)
    (num_circles : ℕ) (num_squares : ℕ) (num_triangles : ℕ)
    (total_figures_eq : total_figures = 10)
    (num_circles_eq : num_circles = 3)
    (num_squares_eq : num_squares = 4)
    (num_triangles_eq : num_triangles = 3) :
    (num_circles + num_squares) / total_figures = 7 / 10 :=
by sorry

end probability_circle_or_square_l59_59708


namespace power_function_through_point_l59_59966

theorem power_function_through_point :
  ∃ α : ℝ, (∀ (x : ℝ), (y : ℝ),  y = x^α → (y = 4 ∧ x = 8) → (y = x^(2 / 3))) :=
begin
  sorry
end

end power_function_through_point_l59_59966


namespace order_of_expressions_l59_59671

theorem order_of_expressions (k : ℕ) (hk : k > 4) : (k + 2) < (2 * k) ∧ (2 * k) < (k^2) ∧ (k^2) < (2^k) := by
  sorry

end order_of_expressions_l59_59671


namespace angle_C_in_triangle_l59_59626

theorem angle_C_in_triangle (A B C : ℝ) (hA : tan A = 1/3) (hB : tan B = -2) (hTriangle : A + B + C = π) : 
  C = π / 4 :=
by
  sorry

end angle_C_in_triangle_l59_59626


namespace find_w_value_l59_59398

theorem find_w_value :
  let u := ⟨2, 4, w⟩ 
  let v := ⟨4, -3, 2⟩ 
  let k := (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) / (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)
  k * v = (10 / 29) * v → w = 7 :=
by
  intro u v k h
  sorry

end find_w_value_l59_59398


namespace sufficiency_not_necessity_l59_59603

theorem sufficiency_not_necessity (x y : ℝ) :
  (x > 3 ∧ y > 3) → (x + y > 6 ∧ x * y > 9) ∧ (¬ (x + y > 6 ∧ x * y > 9 → x > 3 ∧ y > 3)) :=
by
  sorry

end sufficiency_not_necessity_l59_59603


namespace time_saved_1200_miles_l59_59701

theorem time_saved_1200_miles
  (distance : ℕ)
  (speed1 speed2 : ℕ)
  (h_distance : distance = 1200)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 50) :
  (distance / speed2) - (distance / speed1) = 4 :=
by
  sorry

end time_saved_1200_miles_l59_59701


namespace find_h_l59_59333

-- Definition of n bowtie h
def bowtie (n h : ℝ) : ℝ := n + Real.sqrt (h + Real.sqrt (h + Real.sqrt (h + ...)))

theorem find_h (h : ℝ) : bowtie 5 h = 8 → h = 6 := 
by sorry

end find_h_l59_59333


namespace area_ratio_correct_l59_59466

noncomputable def area_ratio (b : ℝ) : ℝ :=
  let w := 3 * π / 2 * b in
  let l := 9 * π / 2 * b in
  let A_rectangle := l * w in
  let A_ellipse := π * 2 * b * b in
  A_rectangle / A_ellipse

theorem area_ratio_correct (b : ℝ) : area_ratio b = 27 * π / 8 :=
by {
    sorry
}

end area_ratio_correct_l59_59466


namespace solution_set_f_x_minus_1_lt_1_l59_59940

noncomputable def f : ℝ → ℝ := λ x, 
  if 0 ≤ x ∧ x ≤ 1 then 
    Real.sin (π / 2 * x) 
  else if 1 < x then 
    x^2 + Real.log x 
  else 
    0  -- In practice, this case will not be used because of domain conditions

theorem solution_set_f_x_minus_1_lt_1 : 
  ∀ x, 0 < x ∧ x < 2 ↔ f (x - 1) < 1 := 
begin
  sorry
end

end solution_set_f_x_minus_1_lt_1_l59_59940


namespace david_average_speed_l59_59506

-- Definition for driving time in hours
def driving_time : ℝ := 4.5

-- Definition for distance covered
def distance_covered : ℝ := 210

-- Definition for average speed calculation
def average_speed := distance_covered / driving_time

-- Statement to prove that David's average speed is 46.\overline{6} miles per hour.
theorem david_average_speed : average_speed = 46.666666666666664 := 
sorry

end david_average_speed_l59_59506


namespace triangle_inequality_set_D_l59_59104

theorem triangle_inequality_set_D : 
  (∀ a b c : ℕ, a + b > c ∧ a + c > b ∧ b + c > a → a = 5 ∧ b = 7 ∧ c = 9) := 
by {
  sorry,
}

end triangle_inequality_set_D_l59_59104


namespace domain_of_f_i_l59_59949

variable (f : ℝ → ℝ)

theorem domain_of_f_i (h : ∀ x, -1 ≤ x + 1 ∧ x + 1 ≤ 1) : ∀ x, -2 ≤ x ∧ x ≤ 0 :=
by
  intro x
  specialize h x
  sorry

end domain_of_f_i_l59_59949


namespace enclosed_fig_area_correct_l59_59166

noncomputable def enclosed_fig_area (US UT : ℝ) (angle_TUS : ℝ) (arc_ratio : ℝ) (radius : ℝ) : ℝ :=
  if h : US = UT ∧ US = 2 ∧ UT = 2 ∧ angle_TUS = 60 ∧ arc_ratio = 1/6 ∧ radius = 2 then
    4 * Real.sqrt 3 - 4 * Real.pi / 3
  else
    0

theorem enclosed_fig_area_correct :
  enclosed_fig_area 2 2 60 (1/6) 2 = 4 * Real.sqrt 3 - 4 * Real.pi / 3 :=
by
  simp [enclosed_fig_area]
  split_ifs
  . sorry
  . contradict h; tauto!

end enclosed_fig_area_correct_l59_59166


namespace beef_weight_after_processing_l59_59149

noncomputable def initial_weight : ℝ := 840
noncomputable def lost_percentage : ℝ := 35
noncomputable def retained_percentage : ℝ := 100 - lost_percentage
noncomputable def final_weight : ℝ := retained_percentage / 100 * initial_weight

theorem beef_weight_after_processing : final_weight = 546 := by
  sorry

end beef_weight_after_processing_l59_59149


namespace train_speed_l59_59741

theorem train_speed (distance_AB : ℕ) (start_time_A : ℕ) (start_time_B : ℕ) (meet_time : ℕ) (speed_B : ℕ) (time_travel_A : ℕ) (time_travel_B : ℕ)
  (total_distance : ℕ) (distance_B_covered : ℕ) (speed_A : ℕ)
  (h1 : distance_AB = 330)
  (h2 : start_time_A = 8)
  (h3 : start_time_B = 9)
  (h4 : meet_time = 11)
  (h5 : speed_B = 75)
  (h6 : time_travel_A = meet_time - start_time_A)
  (h7 : time_travel_B = meet_time - start_time_B)
  (h8 : distance_B_covered = time_travel_B * speed_B)
  (h9 : total_distance = distance_AB)
  (h10 : total_distance = time_travel_A * speed_A + distance_B_covered):
  speed_A = 60 := 
by
  sorry

end train_speed_l59_59741


namespace log_eq_solution_l59_59721

noncomputable def solve_log_eq (x : ℝ) : Prop :=
  log10 (x - 3) + log10 x = 1 ∧ x > 3

theorem log_eq_solution :
  ∃ (x : ℝ), solve_log_eq x ∧ x = 5 :=
by {
  sorry
}

end log_eq_solution_l59_59721


namespace parallelepiped_eq_l59_59657

-- Definitions of the variables and conditions
variables (a b c u v w : ℝ)

-- Prove the identity given the conditions:
theorem parallelepiped_eq :
  u * v * w = a * v * w + b * u * w + c * u * v :=
sorry

end parallelepiped_eq_l59_59657


namespace angle_is_pi_over_6_l59_59942

noncomputable def angle_between_vectors
  (a b : ℝ^3) (ha : ‖a‖ = sqrt 3) (hb : ‖b‖ = 2) (h : (a - b) ⬝ a = 0) : ℝ :=
classical.some (exists_angle (a) (b) ha hb h)

theorem angle_is_pi_over_6
  (a b : ℝ^3) (ha : ‖a‖ = sqrt 3) (hb : ‖b‖ = 2) (h : (a - b) ⬝ a = 0) : angle_between_vectors a b ha hb h = π / 6 :=
sorry

end angle_is_pi_over_6_l59_59942


namespace expand_product_l59_59519

-- We need to state the problem as a theorem
theorem expand_product (y : ℝ) (hy : y ≠ 0) : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry -- Skipping the proof

end expand_product_l59_59519


namespace sin_alpha_plus_beta_eq_l59_59229

theorem sin_alpha_plus_beta_eq :
  ∀ (α β : ℝ), 
    (0 < α ∧ α < π) → (0 < β ∧ β < π) →
    cos (π - α) = 1 / 3 →
    sin (π / 2 + β) = 2 / 3 →
    sin (α + β) = (4 * real.sqrt 2 - real.sqrt 5) / 9 :=
by
  intros α β hα hβ hcos hsin
  sorry

end sin_alpha_plus_beta_eq_l59_59229


namespace smallest_product_l59_59064

theorem smallest_product : 
  ∃ a b ∈ ({-10, -4, 0, 2, 6} : Set ℤ), 
  ∀ x y ∈ ({-10, -4, 0, 2, 6} : Set ℤ), a * b ≤ x * y ∧ a * b = -60 := 
by 
  sorry

end smallest_product_l59_59064


namespace algebra_statements_correct_l59_59916

theorem algebra_statements_correct (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃ m n : ℝ, m ≠ n ∧ a * m^2 + b * m + c = a * n^2 + b * n + c) ∧
  (ac < 0 → ∃ m n : ℝ, m > n ∧ a * m^2 + b * m + c < 0 ∧ 0 < a * n^2 + b * n + c) ∧
  (ab > 0 → ∃ p q : ℝ, p ≠ q ∧ a * p^2 + b * p + c = a * q^2 + b * q + c ∧ p + q < 0) :=
sorry

end algebra_statements_correct_l59_59916


namespace total_area_painted_correct_l59_59834

-- Defining the properties of the shed
def shed_w := 12  -- width in yards
def shed_l := 15  -- length in yards
def shed_h := 7   -- height in yards

-- Calculating area to be painted
def wall_area_1 := 2 * (shed_w * shed_h)
def wall_area_2 := 2 * (shed_l * shed_h)
def floor_ceiling_area := 2 * (shed_w * shed_l)
def total_painted_area := wall_area_1 + wall_area_2 + floor_ceiling_area

-- The theorem to be proved
theorem total_area_painted_correct :
  total_painted_area = 738 := by
  sorry

end total_area_painted_correct_l59_59834


namespace equivalent_proof_problem_l59_59599

noncomputable def point := ℝ × ℝ

def line (l: ℝ) (x: ℝ) (y: ℝ) : Prop := 2 * x - y = 0

def circle (C: ℝ) (x: ℝ) (y: ℝ) : Prop := x^2 + (y-4)^2 = 1

def point_on_line (P: point) : Prop := ∃ a : ℝ, P = (a, 2 * a)

def maximum_distance_to_line (M: point) (dist: ℝ) : Prop :=
  ∃ (x y: ℝ), circle C x y → M = (x, y) ∧ dist = (4 * Real.sqrt 5 + 5) / 5

def tangent_to_circle (P: point) (A: point) : Prop :=
  line l P.1 P.2 → ∃ (a: ℝ), P = (a, 2 * a) ∧ circle C A.1 A.2

def common_chord_fixed_point (A P: point) (fixed_pt: point) : Prop :=
  tangent_to_circle P A → fixed_pt = (1 / 2, 15 / 4)

theorem equivalent_proof_problem :
  (∀ (M: point), circle C M.1 M.2 → maximum_distance_to_line M ((4 * Real.sqrt 5 + 5) / 5)) ∧
  (∀ (P A: point), point_on_line P → tangent_to_circle P A →
  ∃ fixed_pt: point, common_chord_fixed_point A P fixed_pt) :=
sorry

end equivalent_proof_problem_l59_59599


namespace greatest_n_perfect_square_sum_l59_59900

theorem greatest_n_perfect_square_sum :
  ∃ n : ℕ, n ≤ 2023 ∧ 
  (∑ i in finset.range (n + 1), i ^ 2) * 
  (∑ i in finset.range (2 * n + 1) \ finset.range (n + 1), i ^ 2) = 1921 ^ 2 :=
sorry

end greatest_n_perfect_square_sum_l59_59900


namespace sequence_a_general_term_sum_of_sequence_b_l59_59566

-- Definitions for the sequences and their conditions
def sequence_a (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, 3 * S n = 2 * a n + 1

def sequence_b (a b : ℕ → ℤ) : Prop :=
  ∀ n, b n = (n + 1) * a n

-- Specify the general term formula for sequence a_n
def general_term_a (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = (-2)^(n - 1)

-- Specify the sum T_n for sequence b_n
def sum_of_b (b : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ k in Finset.range n, b (k + 1)

-- Main theorem statements
theorem sequence_a_general_term {a S : ℕ → ℤ} (h : sequence_a a S) : 
  general_term_a a :=
by
  sorry

theorem sum_of_sequence_b {a b : ℕ → ℤ} (T : ℕ → ℤ) 
  (h_a : general_term_a a) (h_b : sequence_b a b) : 
  sum_of_b b T :=
begin
  sorry
end

end sequence_a_general_term_sum_of_sequence_b_l59_59566


namespace MN_parallel_AB_l59_59647

-- Definitions of points and properties
variables {A B C D M N : Type}
variable [affine_space ℝ] -- Assuming ℝ is the base field for affine space

-- Assume ABCD is a parallelogram
axiom parallelogram_ABCD : parallelogram A B C D

-- M is a point inside ABCD
axiom M_in_ABCD : inside A B C D M

-- N is a point inside the triangle AMD
axiom N_in_AMD : inside_triangle A M D N

-- Angle condition 1
axiom angle_condition1 : ∠M N A + ∠M C B = 180

-- Angle condition 2
axiom angle_condition2 : ∠M N D + ∠M B C = 180

-- Proof goal
theorem MN_parallel_AB : parallel (line_through M N) (line_through A B) :=
by
    sorry

end MN_parallel_AB_l59_59647


namespace coins_distributable_l59_59750

theorem coins_distributable (n : ℕ) (h_even : ∀ k : ℕ, 2 ≤ k ∧ k < 77 → ∀ s : Fin k → ℕ → list ℕ, (∀ i : Fin k, s i = 0 → Finset.sum (Finset.univ.image (λ j, s j)) = n) → (∃ m : ℕ, Finset.Sum (Finset.range 77) (λ _, m) = n)) :
  ∃ m : ℕ, n = 77 * m := 
sorry

end coins_distributable_l59_59750


namespace min_value_Px_Py_l59_59507

def P (τ : ℝ) : ℝ := (τ + 1)^3

theorem min_value_Px_Py (x y : ℝ) (h : x + y = 0) : P x + P y = 2 :=
sorry

end min_value_Px_Py_l59_59507


namespace storage_cube_edge_length_l59_59030

namespace AndersonRemodel

-- Definitions for each room's wall dimensions
def hall_longer_wall_area : ℝ := 16
def hall_shorter_wall_area : ℝ := 14

def bedroom_longer_wall_area : ℝ := 12
def bedroom_shorter_wall_area : ℝ := 8

def kitchen_longer_wall_area : ℝ := 10
def kitchen_shorter_wall_area : ℝ := 5

def bathroom_longer_wall_area : ℝ := 6
def bathroom_shorter_wall_area : ℝ := 3

-- Calculate the total wall surface area for each room
def hall_total_wall_area : ℝ := 2 * (hall_longer_wall_area + hall_shorter_wall_area)
def bedroom_total_wall_area : ℝ := 2 * (bedroom_longer_wall_area + bedroom_shorter_wall_area)
def kitchen_total_wall_area : ℝ := 2 * (kitchen_longer_wall_area + kitchen_shorter_wall_area)
def bathroom_total_wall_area : ℝ := 2 * (bathroom_longer_wall_area + bathroom_shorter_wall_area)

-- Calculate the total surface area of all walls in the house
def total_wall_surface_area : ℝ :=
  hall_total_wall_area + bedroom_total_wall_area + kitchen_total_wall_area + bathroom_total_wall_area

-- Calculate the edge length of the cube from the total wall surface area
def cube_edge_length (total_surface_area : ℝ) : ℝ :=
  Real.sqrt (total_surface_area / 6)

-- Prove that given the conditions, the edge length of the storage cube is approximately 4.97 meters
theorem storage_cube_edge_length :
  hall_total_wall_area = 60 ∧
  bedroom_total_wall_area = 40 ∧
  kitchen_total_wall_area = 30 ∧
  bathroom_total_wall_area = 18 ∧
  total_wall_surface_area = 148 →
  abs (cube_edge_length 148 - 4.97) < 0.01 :=
by
  sorry

end AndersonRemodel

end storage_cube_edge_length_l59_59030


namespace ellipse_hyperbola_system_fixed_point_line_MN_dot_product_range_l59_59935

variables {a b c : ℝ} (e : ℝ) (k t : ℝ)
def equation_hyperbola : Prop := ∀ x y : ℝ, y^2 - x^2 = 1
def foci_common_ellipse_hyperbola : Prop := true -- This is known.
def ellipse_eccentricity : Prop := e = √6 / 3 
def ellipse_equation : Prop := ∀ x y : ℝ, (y^2 / 3) + x^2 = 1
def slope_condition (x1 y1 x2 y2 : ℝ) (AM_slope AN_slope : ℝ) : Prop := AM_slope * AN_slope = 1
def fixed_point_condition (M N : ℝ × ℝ) : Prop := (0, -2*√3) ∈ line_through_points M N
def range_dot_product (OM ON : ℝ × ℝ) : Prop := ∀ k : ℝ, -3 < (45 - 3*k^2) / (3 + k^2) < (3/2)

theorem ellipse_hyperbola_system :
  (equation_hyperbola ∧ foci_common_ellipse_hyperbola ∧ ellipse_eccentricity) →
  ellipse_equation :=
sorry

theorem fixed_point_line_MN (x1 y1 x2 y2 AM_slope AN_slope : ℝ) :
  (slope_condition x1 y1 x2 y2 AM_slope AN_slope) →
  fixed_point_condition (x1, y1) (x2, y2) :=
sorry

theorem dot_product_range (OM ON : ℝ × ℝ) :
  range_dot_product OM ON :=
sorry

end ellipse_hyperbola_system_fixed_point_line_MN_dot_product_range_l59_59935


namespace largest_integer_n_l59_59533

theorem largest_integer_n (n : ℤ) :
  (n^2 - 11 * n + 24 < 0) → n ≤ 7 :=
by
  sorry

end largest_integer_n_l59_59533


namespace fraction_of_number_l59_59281

theorem fraction_of_number (F : ℚ) (h : 0.5 * F * 120 = 36) : F = 3 / 5 :=
by
  sorry

end fraction_of_number_l59_59281


namespace calc_expression_l59_59181

theorem calc_expression : (2019 / 2018) - (2018 / 2019) = 4037 / 4036 := 
by sorry

end calc_expression_l59_59181


namespace abs_nested_expression_l59_59907

theorem abs_nested_expression : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry

end abs_nested_expression_l59_59907


namespace polyhedron_distance_greater_21_l59_59165

-- Define a sphere with a given radius
structure Sphere (α : Type*) [MetricSpace α] :=
(center : α)
(radius : ℝ)

-- Define a polyhedron structure
structure Polyhedron (α : Type*) :=
(faces : ℕ)
(inscribed_sphere : Sphere α)

-- Define the theorem with the given conditions
theorem polyhedron_distance_greater_21 {α : Type*} [MetricSpace α] (s : Sphere α) (p : Polyhedron α) 
  (hr : s.radius = 10) (hp : p.faces = 19) (h_inscribed : p.inscribed_sphere = s) :
  ∃ (x y : α), x ≠ y ∧ dist x y > 21 :=
by
  sorry

end polyhedron_distance_greater_21_l59_59165


namespace Suma_can_do_work_alone_in_eight_days_l59_59797

-- Definitions based on problem conditions
def Renu_time : ℕ := 8
def Combined_time : ℕ := 4

-- Theorem statement based on question and correct answer
theorem Suma_can_do_work_alone_in_eight_days : 
  (W : ℕ) → (Renu_work_rate := W / Renu_time) → (Combined_work_rate := W / Combined_time) 
  → (Suma_work_rate := Combined_work_rate - Renu_work_rate) 
  → (Suma_time := W / Suma_work_rate) 
  → Suma_time = 8 := 
by 
  intros W Renu_work_rate Combined_work_rate Suma_work_rate Suma_time
  sorry

end Suma_can_do_work_alone_in_eight_days_l59_59797


namespace sqrt_72_eq_6_sqrt_2_l59_59024

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_72_eq_6_sqrt_2_l59_59024


namespace problem1_problem2_l59_59437

-- Proof Problem 1
theorem problem1 (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x < -1 ∨ x > 5 :=
by sorry

-- Proof Problem 2
theorem problem2 (x a : ℝ) :
  if a = -1 then (x^2 + (1 - a) * x - a < 0 ↔ false) else
  if a > -1 then (x^2 + (1 - a) * x - a < 0 ↔ -1 < x ∧ x < a) else
  (x^2 + (1 - a) * x - a < 0 ↔ a < x ∧ x < -1) :=
by sorry

end problem1_problem2_l59_59437


namespace find_h_l59_59193

def infinite_sqrt_series (b : ℝ) : ℝ := sorry -- Placeholder for infinite series sqrt(b + sqrt(b + ...))

def diamond (a b : ℝ) : ℝ :=
  a^2 + infinite_sqrt_series b

theorem find_h (h : ℝ) : diamond 3 h = 12 → h = 6 :=
by
  intro h_condition
  -- Further steps will be used during proof
  sorry

end find_h_l59_59193


namespace inheritance_distributed_correctly_l59_59878

noncomputable def total_estate : ℝ := 8100
noncomputable def each_child_inheritance : ℝ := 900

theorem inheritance_distributed_correctly :
  ∀ (n : ℕ), (n + 1) * (n.succ * 100 + (total_estate - (∑ i in finset.range n.succ, i.succ * 100)) / 10) = each_child_inheritance := sorry

end inheritance_distributed_correctly_l59_59878


namespace infinite_squares_in_sequence_l59_59971

def a (n : ℕ) : ℕ := Int.floor (n * Real.sqrt 2)

theorem infinite_squares_in_sequence : ∀ N : ℕ, ∃ n ≥ N, ∃ k : ℕ, a n = k * k := by
  sorry

end infinite_squares_in_sequence_l59_59971


namespace quadratic_inequality_solution_l59_59995

theorem quadratic_inequality_solution {a : ℝ} :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ a < -1 ∨ a > 3 :=
by sorry

end quadratic_inequality_solution_l59_59995


namespace total_hits_and_misses_l59_59766

theorem total_hits_and_misses (h : ℕ) (m : ℕ) (hc : m = 3 * h) (hm : m = 50) : h + m = 200 :=
by
  sorry

end total_hits_and_misses_l59_59766


namespace chooseOneFromEachCategory_chooseTwoDifferentTypes_l59_59807

-- Define the number of different paintings in each category
def traditionalChinesePaintings : ℕ := 5
def oilPaintings : ℕ := 2
def watercolorPaintings : ℕ := 7

-- Part (1): Prove that the number of ways to choose one painting from each category is 70
theorem chooseOneFromEachCategory : traditionalChinesePaintings * oilPaintings * watercolorPaintings = 70 := by
  sorry

-- Part (2): Prove that the number of ways to choose two paintings of different types is 59
theorem chooseTwoDifferentTypes :
  (traditionalChinesePaintings * oilPaintings) + 
  (traditionalChinesePaintings * watercolorPaintings) + 
  (oilPaintings * watercolorPaintings) = 59 := by
  sorry

end chooseOneFromEachCategory_chooseTwoDifferentTypes_l59_59807


namespace common_ratio_sin_arithmetic_seq_l59_59062

theorem common_ratio_sin_arithmetic_seq (α₁ β q : ℝ) (h : ∀ n : ℕ, sin (α₁ + n * β) = q ^ n * sin α₁) : q = 1 ∨ q = -1 :=
sorry

end common_ratio_sin_arithmetic_seq_l59_59062


namespace solve_for_x_l59_59545

variable (x A : ℝ)
variable hA : A = 3
variable hEq : A * x^2 - 6 * x + 3 = 0

theorem solve_for_x : x = 1 :=
by
  -- Apply the given condition
  rw [hA] at hEq
  sorry

end solve_for_x_l59_59545


namespace book_arrangement_l59_59278

theorem book_arrangement : 
  let math_books := 4
  let english_books := 6
  let split_groups := 2
  let group_size := 3
  let ways_to_arrange_groups := factorial 3
  let ways_to_arrange_math_books := factorial 4
  let ways_to_arrange_first_english_group := factorial group_size
  let ways_to_arrange_second_english_group := factorial group_size
  in ways_to_arrange_groups * ways_to_arrange_math_books * ways_to_arrange_first_english_group * ways_to_arrange_second_english_group = 5184 := 
by
  sorry

end book_arrangement_l59_59278


namespace reflection_on_circumcircle_l59_59464

-- Defining the points and triangle geometry
variables {A B C P Q R D : Type}
variables [point A] [point B] [point C] [point P] [point Q] [point R] [point D]

-- Given conditions
axiom isosceles_triangle (AB AC : A = B = C) : B = C   -- Triangle ABC is isosceles
axiom point_on_base (P_base : P ∈ segment B C)         -- Point P on base BC
axiom parallel_lines (PQ_parallel : parallel (line P Q) (line A C)) (PR_parallel : parallel (line P R) (line A B)) -- Parallel conditions
axiom intersection_points (Q_inter : Q ∈ intersection (line P Q) (line A C)) (R_inter : R ∈ intersection (line P R) (line A B)) -- Q and R as intersection points
axiom reflection_D (D_ref : reflection P line QR = D)  -- D is the reflection of P over QR

-- To prove: Points A, B, C, and D are concyclic
theorem reflection_on_circumcircle :
  concyclic A B C D :=
sorry

end reflection_on_circumcircle_l59_59464


namespace optimal_height_l59_59116

variables {b a : ℝ}
variables (AB CD : ℝ → Prop)
variables (M : ℝ × ℝ)
variables (N : ℝ × ℝ)
variables (E : ℝ × ℝ)

-- Definitions
def parallel (L1 L2 : ℝ → Prop) (distance : ℝ) : Prop :=
  ∀ x y, L1 x → L2 y → abs (x - y) = distance

def segment (P Q : ℝ × ℝ) (length : ℝ) : Prop :=
  abs (P.1 - Q.1) = length ∨ abs (P.2 - Q.2) = length

def area_triangle (base height : ℝ) : ℝ := (1/2) * base * height

-- Conditions
axiom parallel_lines : parallel AB CD b
axiom point_M_on_AB : AB M.1
axiom point_N_on_CD : CD N.1
axiom segment_ME : segment M E a

theorem optimal_height :
  let h := M.2 - E.2 - b in -- given that M and E are at the same horizontal line
  let minimized_area := λ h, a * (2 * h + b^2 / h - 2 * b) in
  minimized_area h = minimized_area (b * real.sqrt 2 / 2) :=
sorry

end optimal_height_l59_59116


namespace max_possible_sum_y_diff_l59_59719

-- Definitions and conditions
def x_sequence : ℕ → ℝ
def y (k : ℕ) : ℝ := (1 : ℝ) / (k : ℝ + 1) * ∑ i in (finRange k.succ), (x_sequence (i + 1))

-- The main theorem we need to prove
theorem max_possible_sum_y_diff : 
  (∑ k in (finRange 2000), |x_sequence k - x_sequence (k + 1)| = 2001) →
  (∑ k in (finRange 2000), |y k - y (k + 1)| ≤ 2000) :=
begin
  sorry
end

end max_possible_sum_y_diff_l59_59719


namespace range_of_omega_l59_59555

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := Real.cos (ω * x + ϕ)

theorem range_of_omega (α ω ϕ : ℝ) (h₀ : ω > 0) (h₁ : f α ω ϕ = 0) (h₂ : (derivative (λ x, f x ω ϕ)) α > 0)
  (h₃ : ∀ x ∈ Ico α (α + π), ∃ y ∈ Ico α (α + π), f y ω ϕ < f x ω ϕ) :
  1 < ω ∧ ω ≤ 3 / 2 :=
sorry

end range_of_omega_l59_59555


namespace angle_at_5_40_is_70_degrees_l59_59092

noncomputable def calculate_minute_angle (minutes : ℕ) : ℝ :=
  minutes * 6

noncomputable def calculate_hour_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours * 30) + (minutes * 0.5)

noncomputable def calculate_angle (hour_angle : ℝ) (minute_angle : ℝ) : ℝ :=
  abs (hour_angle - minute_angle)

theorem angle_at_5_40_is_70_degrees :
  let minute_angle := calculate_minute_angle 40,
      hour_angle := calculate_hour_angle 5 40,
      angle := calculate_angle hour_angle minute_angle
  in angle = 70 :=
by
  sorry

end angle_at_5_40_is_70_degrees_l59_59092


namespace total_practice_hours_l59_59363

def weekly_practice_hours : ℕ := 4
def weeks_in_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours : (weekly_practice_hours * weeks_in_month) * months = 80 := by
  -- Calculation for weekly practice in hours
  let monthly_hours := weekly_practice_hours * weeks_in_month
  -- Calculation for total practice in hours
  have total_hours : ℕ := monthly_hours * months
  have calculation : total_hours = 80 := 
    by simp [weekly_practice_hours, weeks_in_month, months, monthly_hours, total_hours]
  exact calculation

end total_practice_hours_l59_59363


namespace perimeter_of_triangle_l59_59904

def distance (p1 p2: (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def perimeter_triangle (a b c: (ℝ × ℝ)) : ℝ :=
  distance a b + distance b c + distance c a

theorem perimeter_of_triangle :
  let A := (1, 2)
  let B := (1, 5)
  let C := (4, 5)
  perimeter_triangle A B C = 6 + 3 * real.sqrt 2 :=
by
  sorry

end perimeter_of_triangle_l59_59904


namespace cos_double_angle_identity_l59_59588

open Real

theorem cos_double_angle_identity (α : ℝ) 
  (h : tan (α + π / 4) = 1 / 3) : cos (2 * α) = 3 / 5 :=
sorry

end cos_double_angle_identity_l59_59588


namespace min_ab_l59_59996

theorem min_ab (a b : ℝ) (h : (1 / a) + (1 / b) = Real.sqrt (a * b)) : a * b ≥ 2 := by
  sorry

end min_ab_l59_59996


namespace kara_total_water_intake_l59_59321

-- Define dosages and water intake per tablet
def medicationA_doses_per_day := 3
def medicationB_doses_per_day := 4
def medicationC_doses_per_day := 2
def medicationD_doses_per_day := 1

def water_per_tablet_A := 4
def water_per_tablet_B := 5
def water_per_tablet_C := 6
def water_per_tablet_D := 8

-- Compute weekly water intake
def weekly_water_intake_medication (doses_per_day water_per_tablet : ℕ) (days : ℕ) : ℕ :=
  doses_per_day * water_per_tablet * days

-- Total water intake for two weeks if instructions are followed perfectly
def total_water_no_errors :=
  2 * (weekly_water_intake_medication medicationA_doses_per_day water_per_tablet_A 7 +
       weekly_water_intake_medication medicationB_doses_per_day water_per_tablet_B 7 +
       weekly_water_intake_medication medicationC_doses_per_day water_per_tablet_C 7 +
       weekly_water_intake_medication medicationD_doses_per_day water_per_tablet_D 7)

-- Missed doses in second week
def missed_water_second_week :=
  3 * water_per_tablet_A +
  2 * water_per_tablet_B +
  2 * water_per_tablet_C +
  1 * water_per_tablet_D

-- Total water actually drunk over two weeks
def total_water_real :=
  total_water_no_errors - missed_water_second_week

-- Proof statement
theorem kara_total_water_intake :
  total_water_real = 686 :=
by
  sorry

end kara_total_water_intake_l59_59321


namespace probability_point_on_line_l59_59153

theorem probability_point_on_line : 
  let total_outcomes := 36 in
  let valid_outcomes := { (x, y) | (x ∈ {1, 2, 3, 4, 5, 6}) ∧ (y ∈ {1, 2, 3, 4, 5, 6}) ∧ (2 * x + y = 8) } in
  let number_of_valid_outcomes := finset.card valid_outcomes in
    (number_of_valid_outcomes : ℝ) / total_outcomes = 1 / 12 :=
by
  sorry

end probability_point_on_line_l59_59153


namespace petri_dish_radius_l59_59136

variable (x y : ℝ)

-- Define the given condition as a hypothesis
def given_equation : Prop := x^2 + y^2 + 10 = 6 * x + 12 * y

-- Define the center of the circle after completing the square
def circle_center : ℝ × ℝ := (3, 6)

-- Define the radius based on the simplified equation of the circle
def circle_radius : ℝ := Real.sqrt 35

-- The theorem to prove that the radius is sqrt(35) given the condition
theorem petri_dish_radius (h : given_equation x y) : 
  ∃ r, r = circle_radius :=
by
  use circle_radius
  sorry

end petri_dish_radius_l59_59136


namespace tobias_total_distance_l59_59073

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * 60 + minutes

def swimming_distance_freestyle (total_time_minutes : ℕ) : ℕ :=
  let cycle_time := 26  -- 20 minutes swimming + 6-minute pause
  let effective_swim_time := 20
  let meters_per_100m := 100
  let cycles_completed := total_time_minutes / cycle_time
  cycles_completed * (effective_swim_time / 4 * meters_per_100m)

def swimming_distance_butterfly (total_time_minutes : ℕ) : ℕ :=
  let cycle_time := 35  -- 30 minutes swimming + 5-minute pause
  let effective_swim_time := 30
  let meters_per_100m := 100
  let cycles_completed := total_time_minutes / cycle_time
  cycles_completed * (effective_swim_time / 7 * meters_per_100m)

theorem tobias_total_distance :
  let total_time_freestyle := time_in_minutes 1 30
  let total_time_butterfly := time_in_minutes 1 30
  swimming_distance_freestyle(total_time_freestyle) +
  swimming_distance_butterfly(total_time_butterfly) =
    1500 + 858 := 
  by 
    sorry

end tobias_total_distance_l59_59073


namespace boat_downstream_distance_l59_59128

variable (speed_still_water : ℤ) (speed_stream : ℤ) (time_downstream : ℤ)

theorem boat_downstream_distance
    (h₁ : speed_still_water = 24)
    (h₂ : speed_stream = 4)
    (h₃ : time_downstream = 4) :
    (speed_still_water + speed_stream) * time_downstream = 112 := by
  sorry

end boat_downstream_distance_l59_59128


namespace sqrt_72_eq_6_sqrt_2_l59_59026

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := 
by
  sorry

end sqrt_72_eq_6_sqrt_2_l59_59026


namespace repeating_decimal_product_l59_59179

theorem repeating_decimal_product : ∀ (q : ℝ), q = 0.3333... → 9 * q = 3 :=
by
  intros q hq
  sorry

end repeating_decimal_product_l59_59179


namespace liquid_levels_proof_l59_59770

noncomputable def liquid_levels (H : ℝ) : ℝ × ℝ :=
  let ρ_water := 1000
  let ρ_gasoline := 600
  -- x = level drop in the left vessel
  let x := (3 / 14) * H
  let h_left := 0.9 * H - x
  let h_right := H
  (h_left, h_right)

theorem liquid_levels_proof (H : ℝ) (h : ℝ) :
  H > 0 →
  h = 0.9 * H →
  liquid_levels H = (0.69 * H, H) :=
by
  intros
  sorry

end liquid_levels_proof_l59_59770


namespace alice_paid_percentage_l59_59749

theorem alice_paid_percentage (SRP P : ℝ) (h1 : P = 0.60 * SRP) (h2 : P_alice = 0.60 * P) :
  (P_alice / SRP) * 100 = 36 := by
sorry

end alice_paid_percentage_l59_59749


namespace probability_both_heads_l59_59078

-- Define the sample space and the probability of each outcome
def sample_space : List (Bool × Bool) := [(true, true), (true, false), (false, true), (false, false)]

-- Define the function to check for both heads
def both_heads (outcome : Bool × Bool) : Bool :=
  outcome = (true, true)

-- Calculate the probability of both heads
theorem probability_both_heads :
  (sample_space.filter both_heads).length / sample_space.length = 1 / 4 := sorry

end probability_both_heads_l59_59078


namespace geometric_seq_a4_l59_59221

variable {a : ℕ → ℝ}

-- Definition: a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Condition
axiom h : a 2 * a 6 = 4

-- Theorem that needs to be proved
theorem geometric_seq_a4 (h_seq: is_geometric_sequence a) (h: a 2 * a 6 = 4) : a 4 = 2 ∨ a 4 = -2 := by
  sorry

end geometric_seq_a4_l59_59221


namespace rectangle_not_necessarily_similar_l59_59487

theorem rectangle_not_necessarily_similar
  (eq_triangle_similar : ∀ (a b : Type), (a = b) → (∠a = 60) → (∠b = 60) → (similar_figures a b))
  (iso_right_triangle_similar : ∀ (a b : Type), (a = b) → (∠a = 45) → (∠b = 45) → (similar_figures a b))
  (rectangle_not_similar : ∀ (a b : Type), (a = b) → (∠a = 90) → (∠b = 90) → (¬ similar_figures a b))
  (square_similar : ∀ (a b : Type), (a = b) → (∠a = 90) → (∠b = 90) → (similar_figures a b)) :
  ∀ (r1 r2 : Type), (r1 = r2) → (∠r1 = 90) → (∠r2 = 90) → (¬ similar_figures r1 r2) :=
sorry

end rectangle_not_necessarily_similar_l59_59487


namespace correct_propositions_l59_59341

noncomputable def seqSum (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = ∑ i in Finset.range (n + 1), a_n i

-- Propositions corresponding to ①, ②, ③, and ④
def prop1 (a_n : ℕ → ℝ) : Prop :=
  (∀ n, a_n n = a_n (n + 1)) ↔ (∀ n, (a_n (n + 1) - a_n n = 0) ∧ 
  (a_n (n + 1) / a_n n = 1))

def prop2 (S_n : ℕ → ℝ) (a b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (S_n = λ n, a * n^2 + b * n) → (∀ n, a_n n = 2 * a * n + b - a)

def prop3 (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  (S_n = λ n, 1 - (-1)^n) → (∀ n, (n % 2 = 1 → a_n n = 2) ∧ 
  (n % 2 = 0 → a_n n = -2))

def prop4 (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) : Prop :=
  (∀ m, (S_n m ≠ 0 ∧ S_n (2 * m) - S_n m ≠ 0 ∧ 
  S_n (3 * m) - S_n (2 * m) ≠ 0) → ((S_n m) * 
  (S_n (3 * m) - S_n (2 * m)) = ((S_n (2 * m) - S_n m)^2)))

-- Main theorem proving the correct propositions are ①, ②, and ③
theorem correct_propositions (a_n S_n : ℕ → ℝ) (a b : ℝ) :
  seqSum S_n a_n →
  (prop1 a_n ∧ prop2 S_n a b a_n ∧ prop3 S_n a_n ∧ ¬ prop4 a_n S_n) :=
by
  assume h_seqSum
  sorry

end correct_propositions_l59_59341


namespace matrix_problem_l59_59575

theorem matrix_problem :
  ∃ (X : Matrix (Fin 2) (Fin 2) ℝ),
    let M := ![![1, 0], ![0, -1]]
    let N := ![![1, 2], ![3, 4]]
    M * X = N ∧
    X = ![![1, 2], ![-3, -4]] ∧
    ∃ (lambda1 lambda2 : ℝ) (v1 v2 : Fin 2 → ℝ),
      (Eigenvalue X lambda1 ∧ Eigenvector X lambda1 v1 ∧ lambda1 = -1 ∧ v1 = ![1, -1]) ∧
      (Eigenvalue X lambda2 ∧ Eigenvector X lambda2 v2 ∧ lambda2 = -2 ∧ v2 = ![2, -3])
  := by
  sorry

end matrix_problem_l59_59575


namespace distance_between_stripes_l59_59472

noncomputable def street_width : ℝ := 60
noncomputable def curb_length_between_stripes : ℝ := 25
noncomputable def stripe_length : ℝ := 50

theorem distance_between_stripes :
  let area_of_parallelogram := curb_length_between_stripes * street_width,
      distance_between_stripes := area_of_parallelogram / stripe_length in
  distance_between_stripes = 30 :=
by
  let area_of_parallelogram := curb_length_between_stripes * street_width
  let distance_between_stripes := area_of_parallelogram / stripe_length
  have h1 : area_of_parallelogram = 1500 := by sorry
  have h2 : distance_between_stripes = 30 := by sorry
  exact h2

end distance_between_stripes_l59_59472


namespace sum_to_product_cos_l59_59881

theorem sum_to_product_cos (a b : ℝ) : 
  Real.cos (a + b) + Real.cos (a - b) = 2 * Real.cos a * Real.cos b := 
  sorry

end sum_to_product_cos_l59_59881


namespace students_remaining_l59_59356

theorem students_remaining (students_showed_up : ℕ) (students_checked_out : ℕ) (students_left : ℕ) :
  students_showed_up = 16 → students_checked_out = 7 → students_left = students_showed_up - students_checked_out → students_left = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end students_remaining_l59_59356


namespace last_digit_of_sum_l59_59792

theorem last_digit_of_sum :
  let f (n : ℕ) := n % 10 in
  f (1023 ^ 3923 + 3081 ^ 3921) = 8 := 
by
  -- Extract last digits of the base numbers
  let last_digit_1023 := 1023 % 10,
  let last_digit_3081 := 3081 % 10,
  
  -- Prove that each abides by their cyclic patterns
  let cycle_3 := [3, 9, 7, 1],
  let cycle_1 := [1, 1, 1, 1],
  
  -- Calculate position in cycles
  have pos_1023 := 3923 % 4,
  have pos_3081 := 3921 % 4,
  
  -- Determine last digits of powers
  let last_digit_1023_pow := cycle_3.nth_le (pos_1023) sorry,
  let last_digit_3081_pow := cycle_1.nth_le (pos_3081) sorry,
  
  -- Sum the last digits
  have last_digit_sum := (last_digit_1023_pow + last_digit_3081_pow) % 10,
  
  -- Assert and finish
  exact last_digit_sum = 8 sorry

end last_digit_of_sum_l59_59792


namespace two_points_less_than_half_unit_apart_l59_59360

noncomputable def equilateral_triangle_points_condition (triangle_side_length : ℝ) (number_of_points : ℕ) := 
  triangle_side_length = 1 ∧ number_of_points = 5

theorem two_points_less_than_half_unit_apart 
  (triangle_side_length : ℝ) (number_of_points : ℕ) 
  (points : set (ℝ × ℝ)) 
  (h_condition : equilateral_triangle_points_condition triangle_side_length number_of_points) 
  (h_points : set.card points = number_of_points) 
  (h_in_triangle : ∀ p ∈ points, (0 ≤ p.1) ∧ (0 ≤ p.2) ∧ (p.1 + p.2 * (sqrt 3 / 2) ≤ triangle_side_length)): 
  ∃ (p₁ p₂ : ℝ × ℝ), p₁ ∈ points ∧ p₂ ∈ points ∧ p₁ ≠ p₂ ∧ dist p₁ p₂ < 0.5 :=
sorry

end two_points_less_than_half_unit_apart_l59_59360


namespace scientific_notation_of_213_million_l59_59159

theorem scientific_notation_of_213_million : ∃ (n : ℝ), (213000000 : ℝ) = 2.13 * 10^8 :=
by
  sorry

end scientific_notation_of_213_million_l59_59159


namespace parallelogram_area_l59_59667

open Matrix

noncomputable def u : Fin 2 → ℝ := ![7, -4]
noncomputable def z : Fin 2 → ℝ := ![8, -1]

theorem parallelogram_area :
  let matrix := ![u, z]
  |det (of fun (i j : Fin 2) => (matrix i) j)| = 25 :=
by
  sorry

end parallelogram_area_l59_59667


namespace solution_set_l59_59244

-- Define f(x) as a differentiable function on (0, +\infty)
variable {f : ℝ → ℝ}

-- Declare the hypotheses
variables (h_diff : Differentiable ℝ f)
variables (h_ineq : ∀ x, 0 < x → f(x) > x * (Deriv f x))

-- State the goal
theorem solution_set (x : ℝ) (hx : 0 < x) : (x^2 * f (1/x) - f x > 0) ↔ (1 < x) :=
sorry

end solution_set_l59_59244


namespace circle_area_in_square_ABCD_is_169pi_l59_59167

-- Define the given conditions
def BM : ℝ := 8
def MC : ℝ := 17
def side_length : ℝ := BM + MC -- Side length of the square ABCD

-- Define the radius of the circle (r = 13)
def r : ℝ := 13

-- Define the formula for the area of a circle
def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

-- Theorem statement
theorem circle_area_in_square_ABCD_is_169pi 
  (BM : ℝ) (MC : ℝ) (h1 : BM = 8) (h2 : MC = 17) 
  (r : ℝ) (h3 : r = 13) :
  area_of_circle r = 169 * Real.pi :=
by
  simp [area_of_circle, h3]
  sorry

end circle_area_in_square_ABCD_is_169pi_l59_59167


namespace diagonals_intersect_at_l59_59715

theorem diagonals_intersect_at :
  let A := (3, -4)
  let B := (13, 8)
  let midpoint := (A.1 + B.1) / 2, (A.2 + B.2) / 2
  midpoint = (8, 2) := 
by
  sorry

end diagonals_intersect_at_l59_59715


namespace rooks_arrangement_count_l59_59352

theorem rooks_arrangement_count :
  let board := (λ (i j : ℕ), if i < 4 then i else 7 - i)
  (∃ (S : finset (ℕ × ℕ)), 
    S.card = 8 ∧ 
    (∀ (i j k l : ℕ), (i, j) ∈ S → (k, l) ∈ S → (i ≠ k ∧ j ≠ l)) ∧ 
    (∀ n, n ∈ finset.range 8 → ∃ (i j : ℕ), (i, j) ∈ S ∧ board i j = n))
  ↔ S.card = 3456 :=
by
  sorry

end rooks_arrangement_count_l59_59352


namespace total_practice_hours_l59_59369

def weekly_practice_hours : ℕ := 4
def weeks_per_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours :
  weekly_practice_hours * weeks_per_month * months = 80 := by
  sorry

end total_practice_hours_l59_59369


namespace minimum_type_A_tickets_value_of_m_l59_59705

theorem minimum_type_A_tickets (x : ℕ) (h1 : x + (500 - x) = 500) (h2 : x ≥ 3 * (500 - x)) : x = 375 := by
  sorry

theorem value_of_m (m : ℕ) (h : 500 * (1 + (m + 10) / 100) * (m + 20) = 56000) : m = 50 := by
  sorry

end minimum_type_A_tickets_value_of_m_l59_59705


namespace train_length_proof_l59_59839

def speed_kmh_to_mps (v_kmh : ℝ) : ℝ :=
  v_kmh * 1000 / 3600

def train_length (bridge_length time_sec speed_kmh : ℝ) : ℝ :=
  (speed_kmh_to_mps speed_kmh) * time_sec - bridge_length

theorem train_length_proof :
  train_length 140 24 75 = 360 :=
by
  -- By the conditions provided:
  -- Speed in m/s
  let speed_mps := speed_kmh_to_mps 75
  -- Distance covered in 24 seconds
  let distance := speed_mps * 24
  -- Distance is the sum of train length and bridge length
  -- train_length = distance - bridge_length
  have h1 : train_length 140 24 75 = distance - 140 := rfl
  -- Given distance and calculations
  have h2 : distance = 500 := by sorry
  have h3 : 500 - 140 = 360 := by norm_num
  exact eq.trans h1 (eq.trans h2 h3)

end train_length_proof_l59_59839


namespace percentage_decrease_in_price_l59_59755

theorem percentage_decrease_in_price (original_price new_price decrease percentage : ℝ) :
  original_price = 1300 → new_price = 988 →
  decrease = original_price - new_price →
  percentage = (decrease / original_price) * 100 →
  percentage = 24 := by
  sorry

end percentage_decrease_in_price_l59_59755


namespace num_parabolas_l59_59051

def A : Set ℤ := {n | n ∈ Set.Icc (-5) 5}

theorem num_parabolas : 
  (∃ (a b c : ℤ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a < 0 ∧ c = -1 ∧ 
    (∀ (x : ℤ), x = 0 → a * x ^ 2 + b * x + c = -1)) → 
    24 :=
by
  sorry

end num_parabolas_l59_59051


namespace remaining_integers_after_removal_l59_59831

open Finset

def T : Finset ℕ := range 81 \ {0}

def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, k = m * n

def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ :=
  s.filter (λ x, is_multiple_of n x)

theorem remaining_integers_after_removal : 
  let T := range 81 \ {0} in
  let multiples_of_4 := multiples_of 4 T in
  let multiples_of_5 := multiples_of 5 T in
  let multiples_of_20 := multiples_of 20 T in
  let removed := multiples_of_4 ∪ multiples_of_5 in
  (T.card - removed.card + multiples_of_20.card = 48) := 
by
  sorry

end remaining_integers_after_removal_l59_59831


namespace max_value_of_f_l59_59054

theorem max_value_of_f (φ : ℝ) :
    ∃ x : ℝ, 
    f(x) = cos(x + 2 * φ) + 2 * sin(φ) * sin(x + φ) → 
    ∀ y : ℝ, f(y) ≤ 1 :=
by
  sorry

end max_value_of_f_l59_59054


namespace journey_duration_correct_l59_59505

noncomputable def time_in_minutes_since_midnight (hours : ℕ) (minutes : ℕ) (seconds : ℕ) : ℕ := 
  (hours * 60) + minutes + (seconds / 60)

def initial_time := time_in_minutes_since_midnight 10 54 33
def final_time := time_in_minutes_since_midnight 17 27 16
def journey_duration := final_time - initial_time

theorem journey_duration_correct : journey_duration = (6 * 60 + 33) := 
  by 
    -- Placeholder for proof
    sorry

end journey_duration_correct_l59_59505


namespace prove_ram_map_distance_l59_59707

/-- Define the scale of the map as the ratio of the actual distance to the distance on the map -/
def map_scale (actual_distance_map : ℝ) (actual_distance_km : ℝ) : ℝ :=
  actual_distance_map / actual_distance_km

/-- Define the actual distance Ram is from the base of the mountain in km -/
def actual_distance_ram_km : ℝ := 10.897435897435898

/-- Define the conversion function to calculate the map distance given Ram's actual distance and the scale -/
def map_distance_ram (actual_distance_ram_km : ℝ) (scale_in_inches_per_km : ℝ) : ℝ :=
  actual_distance_ram_km * scale_in_inches_per_km

/-- The given conditions -/
def given_conditions : Prop :=
  let map_distance_mountains : ℝ := 312
  let actual_distance_mountains_km : ℝ := 136
  let scale : ℝ := map_scale map_distance_mountains actual_distance_mountains_km
  let scale_in_inches_per_km : ℝ := scale
  map_distance_ram actual_distance_ram_km scale_in_inches_per_km = 25

theorem prove_ram_map_distance : given_conditions :=
sorry

end prove_ram_map_distance_l59_59707


namespace relationship_between_abc_l59_59962

def f (x : ℝ) : ℝ := log 2 (4^x + 4) - x - 1

def a : ℝ := f (log 3 4)
def b : ℝ := f (log 2 3)
def c : ℝ := f (log 64 3)

theorem relationship_between_abc : a < b ∧ b < c :=
by
  sorry

end relationship_between_abc_l59_59962


namespace car_reaches_zillis_iff_l59_59656

def gcd (a b : ℕ) : ℕ :=
if b = 0 then a else gcd b (a % b)

def relatively_prime (a b : ℕ) : Prop :=
gcd a b = 1

def car_reaches_zillis (ell r : ℕ) : Prop :=
(ell % 4 = 1 ∧ r % 4 = 1) ∨ (ell % 4 = 3 ∧ r % 4 = 3)

theorem car_reaches_zillis_iff (ell r : ℕ) (h1 : relatively_prime ell r) :
  car_reaches_zillis ell r ↔ (ell % 4 = 1 ∧ r % 4 = 1) ∨ (ell % 4 = 3 ∧ r % 4 = 3) :=
sorry

end car_reaches_zillis_iff_l59_59656


namespace probability_two_flies_swept_l59_59007

/-- Defining the positions of flies on the clock -/
inductive positions : Type
| twelve   | three   | six   | nine

/-- Probability that the minute hand sweeps exactly two specific positions after 20 minutes -/
theorem probability_two_flies_swept (flies : list positions) (time : ℕ) :
  (flies = [positions.twelve, positions.three, positions.six, positions.nine]) →
  (time = 20) →
  (probability_sweeps_two_flies flies time = 1 / 3) := sorry

end probability_two_flies_swept_l59_59007


namespace distance_from_y_axis_l59_59385

theorem distance_from_y_axis (P : ℝ × ℝ) (x : ℝ) (hx : P = (x, -9)) 
  (h : (abs (P.2) = 1/2 * abs (P.1))) :
  abs x = 18 :=
by
  sorry

end distance_from_y_axis_l59_59385


namespace triangle_ABC_is_equilateral_l59_59576

noncomputable def vector3D := ℝ × ℝ × ℝ

-- Define the norm of a vector
def norm (v : vector3D) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Define the dot product of two vectors
def dot_product (v1 v2 : vector3D) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the normalized (unit) vector
def normalize (v : vector3D) : vector3D :=
  let n := norm v in (v.1 / n, v.2 / n, v.3 / n)

-- Define the vectors AB, AC, and BC
variables (A B C : vector3D)
def AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3)
def BC := (C.1 - B.1, C.2 - B.2, C.3 - B.3)

-- Theorem statement
theorem triangle_ABC_is_equilateral
  (h₁ : dot_product (normalize AB + normalize AC) BC = 0)
  (h₂ : dot_product (normalize AB) (normalize AC) = 1 / 2) :
  is_equilateral A B C :=
sorry

end triangle_ABC_is_equilateral_l59_59576


namespace asymptotes_and_parabola_slope_through_focus_of_parabola_l59_59228

variable (a : ℝ) (C L : set (ℝ × ℝ))

def hyperbola : set (ℝ × ℝ) := {P | P.1^2 / a^2 - P.2^2 / 9 = 1}
def parabola : set (ℝ × ℝ) := {P | P.2^2 = 4 * P.1}

axiom a_positive : a > 0
axiom foci_property : ∀ P : ℝ × ℝ, P ∈ hyperbola → |P.1 - 1| - |P.1 + 1| = 2

theorem asymptotes_and_parabola (hC : C = hyperbola) (hL : L = parabola) :
  (C = {P | P.1^2 - P.2^2 / 9 = 1}) ∧
  (L = {P | P.2^2 = 4 * P.1}) :=
by
  sorry

theorem slope_through_focus_of_parabola :
  ∀ (k : ℝ), 
    (-1 < k ∧ k< 1) → 
    (∃ M N : ℝ × ℝ, 
    M ∈ parabola ∧ N ∈ parabola ∧
    (k = - P.2 / (P.1 + 1)) → 
    (M.1 + N.1 = -2 * (k^2 - 2) / k^2) ∧ 
    (M.1 * N.1 = 1)) → 
    (k = - P.2 / (P.1 - 1)) → 
    (M.2 * N.2 + M.1 * N.1 - (M.1 + N.1) =  1) → 
    k = sqrt(2) / 2 ∨ k = - sqrt(2) / 2 :=
by
  sorry

end asymptotes_and_parabola_slope_through_focus_of_parabola_l59_59228


namespace trigonometric_problems_l59_59944

-- Lean 4 statement definitions for conditions.
variable (θ : ℝ)
variable (hθ1 : θ ∈ Ioo (π / 2) π)
variable (hθ2 : Real.sin θ = 3 / 5)

-- Theorem: Given conditions on θ, prove the values of tan θ and cos(θ + π / 3).
theorem trigonometric_problems
  (hθ1 : θ ∈ Ioo (π / 2) π)
  (hθ2 : Real.sin θ = 3 / 5):
  Real.tan θ = -3 / 4 ∧ 
  Real.cos (θ + π / 3) = -(4 + 3 * Real.sqrt 3) / 10 :=
  by
  sorry

end trigonometric_problems_l59_59944


namespace magnitude_a_sub_2b_l59_59608

open Real -- to work with Real numbers

def vector : Type := ℝ × ℝ

-- given vectors a and b
def a (x : ℝ) : vector := (1, x)
def b (x : ℝ) : vector := (1, x - 1)

-- vector subtraction and scalar multiplication
def vector_sub (v1 v2 : vector) : vector := (v1.1 - v2.1, v1.2 - v2.2)
def scalar_mul (k : ℝ) (v : vector) : vector := (k * v.1, k * v.2)

-- dot product of two vectors
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- check perpendicularity
def is_perpendicular (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

-- magnitude of a vector
def magnitude (v : vector) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- main theorem stating the magnitude of (a - 2b) is √2
theorem magnitude_a_sub_2b (x : ℝ) (h : is_perpendicular (vector_sub (a x) (scalar_mul 2 (b x))) (a x)) :
  magnitude (vector_sub (a x) (scalar_mul 2 (b x))) = Real.sqrt 2 :=
sorry

end magnitude_a_sub_2b_l59_59608


namespace solution_set_f_greater_x_l59_59234

noncomputable def f : ℝ → ℝ :=
  sorry

axiom functional_eq (x y : ℝ) : f (2 * x * y + 3) = f x * f y - 3 * f y - 6 * x + 9
axiom f_zero : f 0 = 3

theorem solution_set_f_greater_x : {x : ℝ | f x > x} = Ioi (-3) :=
by
  sorry

end solution_set_f_greater_x_l59_59234


namespace find_cosine_of_angle_l59_59525

variables (A B C : ℝ × ℝ × ℝ)

def vector_between (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def cosine_of_angle (v w : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v w / (magnitude v * magnitude w)

theorem find_cosine_of_angle (hA : A = (3, 3, -1))
  (hB : B = (5, 1, -2))
  (hC : C = (4, 1, -3)) :
  cosine_of_angle (vector_between A B) (vector_between A C) = 8 / 9 :=
by sorry

end find_cosine_of_angle_l59_59525


namespace second_student_weight_l59_59737

theorem second_student_weight (
  original_average_weight : ℕ := 28,
  number_of_students : ℕ := 29,
  new_average_weight : ℕ := 27.5,
  weight_one_new_student : ℕ := 25
): ℝ :=
  let original_total_weight := number_of_students * original_average_weight
  let new_number_of_students := number_of_students + 2
  let new_total_weight := new_number_of_students * new_average_weight
  let weight_second_student := (new_total_weight - (original_total_weight + weight_one_new_student)) / 1
  weight_second_student = 20.5
by
  sorry

end second_student_weight_l59_59737


namespace integral_roots_equations_l59_59641

theorem integral_roots_equations (x y z : ℕ) (hx : x = 4) (hy : y = 3) (hz : z = 9) :
  (z^x = y^(2*x)) ∧ (2^z = 2 * 4^x) ∧ (x + y + z = 16) :=
by
  rw [hx, hy, hz]
  -- First condition: 9^4 = 3^(2*4)
  have h1 : (9^4 = 3^(2*4)), by sorry
  -- Second condition: 2^9 = 2 * 4^4
  have h2 : (2^9 = 2 * 4^4), by sorry
  -- Third condition: 4 + 3 + 9 = 16
  have h3 : (4 + 3 + 9 = 16), by sorry
  exact ⟨h1, h2, h3⟩

end integral_roots_equations_l59_59641


namespace concyclic_points_l59_59304

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def perpendicular_bisector (A O : Point) : Line := sorry
noncomputable def reflection (A : Point) (l : Line) : Point := sorry
noncomputable def is_on_circle (p : Point) (c : Circle) : Prop := sorry

theorem concyclic_points
  (A B C O D E K L M : Point)
  (h_acute : acute_triangle A B C)
  (h_O : circumcenter A B C = O)
  (h_perpendicular : line_through_perpendicular A O (line_through_points A D) (line_through_points D E))
  (h_D_on_AB : on_segment D A B)
  (h_E_on_AC : on_segment E A C)
  (h_K_on_BC : on_segment K B C ∧ K ≠ intersection_point (line_through_points A O) (line_through_points B C))
  (h_AK_L_circle : is_on_circle L (circumcircle A D E) ∧ L ≠ A)
  (h_M_reflection : M = reflection A (line_through_points D E)) :
  ∃ (c : Circle), is_on_circle K c ∧ is_on_circle L c ∧ is_on_circle M c ∧ is_on_circle O c := sorry

end concyclic_points_l59_59304


namespace distance_between_foci_is_15_l59_59033

noncomputable def distance_between_foci_of_hyperbola (asymptotes : ℝ → ℝ → Prop) (point : ℝ × ℝ) : ℝ :=
  let intersection := (1 / 2, 2) in
  let a_squared := 45 in
  let b_squared := 11.25 in
  let c_squared := a_squared + b_squared in
  let c := Real.sqrt c_squared in
  2 * c

theorem distance_between_foci_is_15 :
  ∀ asymptotes point,
  (asymptotes 2 (-2) ∧ asymptotes 1 3 ∧ point = (4, 4)) →
  distance_between_foci_of_hyperbola asymptotes point = 15 :=
by
  intros,
  sorry

end distance_between_foci_is_15_l59_59033


namespace temperature_on_Friday_l59_59382

variable (M T W Th F : ℝ)

def avg_M_T_W_Th := (M + T + W + Th) / 4 = 48
def avg_T_W_Th_F := (T + W + Th + F) / 4 = 46
def temp_Monday := M = 42

theorem temperature_on_Friday
  (h1 : avg_M_T_W_Th M T W Th)
  (h2 : avg_T_W_Th_F T W Th F) 
  (h3 : temp_Monday M) : F = 34 := by
  sorry

end temperature_on_Friday_l59_59382


namespace number_of_valid_subsets_l59_59550

def is_odd (n: ℕ) : Prop := n % 2 = 1

def valid_subsets (S : Set ℕ) : Set (Set ℕ) :=
{ T | T ⊆ S ∧ T.card = 3 ∧ is_odd (T.sum (λ x, x)) }

theorem number_of_valid_subsets :
  valid_subsets {101, 106, 113, 129, 134, 145}.to_finset.card = 8 := by
  sorry

end number_of_valid_subsets_l59_59550


namespace sum_median_mode_list_l59_59343

theorem sum_median_mode_list (y : ℝ) : 
  let list_I := [y, 2, 4, 7, 10, 11]
  let list_II := [3, 3, 4, 6, 7, 10]
  y = 9 → 
  let median := (list_II.nth 2).get_or_else 0 + (list_II.nth 3).get_or_else 0 / 2
  let mode := 3  -- Directly assigning mode as it is directly given in the problem
  let r := median + mode in
  r = 8 :=
by
  intros
  sorry

end sum_median_mode_list_l59_59343


namespace train_B_length_l59_59083

variable (vA vB : ℝ)
variable (lengthA : ℝ)
variable (t_cross : ℝ)
variable (L : ℝ)

-- Assume given conditions
axiom h1 : vA = 54 -- Train A speed in km/hr
axiom h2 : vB = 36 -- Train B speed in km/hr
axiom h3 : lengthA = 225 -- Length of train A in meters
axiom h4 : t_cross = 15 -- Time to cross train B in seconds
axiom h5 : L = 150 -- Length of train B in meters

-- Conversion factor from km/hr to m/s
noncomputable def convert_speed (v : ℝ) : ℝ := v * (5 / 18)

-- Relative speed in m/s
noncomputable def relative_speed : ℝ := convert_speed (vA + vB)

-- Total distance covered in 15 seconds at relative speed
axiom distance_covered : lengthA + L = relative_speed * t_cross

-- The theorem to be proved: Length of train B is 150 meters
theorem train_B_length : L = 150 := by
  -- Proof will be added here
  sorry

end train_B_length_l59_59083


namespace sequence_length_16_l59_59188

noncomputable def a : ℕ → ℕ
| 0       := 1
| 1       := 0
| 2       := 1
| (n + 3) := a (n + 1) + b (n + 1)
with b : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 0
| (n + 3) := a (n + 2) + b (n + 1)

theorem sequence_length_16 : a 16 + b 16 = 225 :=
by sorry

end sequence_length_16_l59_59188


namespace parallelogram_area_correct_l59_59112

-- Define the base and height of the parallelogram
def base: ℕ := 21
def height: ℕ := 11

-- Define the formula for the area of a parallelogram
def parallelogram_area (b h : ℕ) : ℕ := b * h

-- The statement we want to prove
theorem parallelogram_area_correct :
  parallelogram_area base height = 231 := 
by
  -- Proof can be filled in later
  sorry

end parallelogram_area_correct_l59_59112


namespace smallest_n_div_75_has_75_divisors_l59_59393

theorem smallest_n_div_75_has_75_divisors :
  ∃ n : ℕ, (n % 75 = 0) ∧ (n.factors.length = 75) ∧ (n / 75 = 432) :=
by
  sorry

end smallest_n_div_75_has_75_divisors_l59_59393


namespace expand_product_l59_59518

-- We need to state the problem as a theorem
theorem expand_product (y : ℝ) (hy : y ≠ 0) : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry -- Skipping the proof

end expand_product_l59_59518


namespace function_machine_output_l59_59639

-- Define the function machine
def function_machine (input : ℕ) : ℕ :=
  let multiplied := input * 3
  in if multiplied > 20 then multiplied - 8 else multiplied + 10

-- Define the conditions and the expected output
theorem function_machine_output :
  function_machine 7 = 13 :=
by
  -- Placeholder for the proof
  sorry

end function_machine_output_l59_59639


namespace count_pairs_satisfying_eq_l59_59615

theorem count_pairs_satisfying_eq :
  {n : Nat // n = (Nat.card {p : (ℕ × ℕ) // (p.1 + p.2 ≤ 150)
    ∧ (p.2 ≠ 0)
    ∧ (p.2^(2) * p.1 + 1 = 17 * (p.1 * p.2^(2) + p.2)) })} = 8 :=
  sorry

end count_pairs_satisfying_eq_l59_59615


namespace granger_paid_correct_amount_l59_59611

-- Prices before any discounts or taxes
def price_spam : ℝ := 3
def price_peanut_butter : ℝ := 5
def price_bread : ℝ := 2
def price_milk : ℝ := 4
def price_eggs : ℝ := 3

-- Discounts and taxes
def discount_spam : ℝ := 0.10
def tax_peanut_butter : ℝ := 0.05
def discount_milk : ℝ := 0.20
def tax_milk : ℝ := 0.08
def discount_eggs : ℝ := 0.05

-- Quantities purchased
def quantity_spam : ℕ := 12
def quantity_peanut_butter : ℕ := 3
def quantity_bread : ℕ := 4
def quantity_milk : ℕ := 2
def quantity_eggs : ℕ := 1

-- Correct answer
def total_paid : ℝ := 65.92

-- Functions to compute the prices after discounts and taxes
def compute_price_after_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def compute_price_after_tax (price : ℝ) (tax : ℝ) : ℝ :=
  price * (1 + tax)

def total_spam : ℝ := quantity_spam * compute_price_after_discount price_spam discount_spam
def total_peanut_butter : ℝ := quantity_peanut_butter * compute_price_after_tax price_peanut_butter tax_peanut_butter
def total_bread : ℝ := quantity_bread * price_bread
def total_milk : ℝ := 
  let price_after_discount := compute_price_after_discount price_milk discount_milk in
  quantity_milk * compute_price_after_tax price_after_discount tax_milk
def total_eggs : ℝ := quantity_eggs * compute_price_after_discount price_eggs discount_eggs

def total_amount_paid : ℝ :=
  total_spam + total_peanut_butter + total_bread + total_milk + total_eggs

theorem granger_paid_correct_amount : total_amount_paid = total_paid := by
  sorry

end granger_paid_correct_amount_l59_59611


namespace max_handshakes_l59_59298

theorem max_handshakes (M : ℕ) (hM : M > 5) :
  (∃ n, n = M - 2) :=
by
  use M - 2
  rw [sub_eq_add_neg]
  exact rfl

end max_handshakes_l59_59298


namespace cost_relationships_cost_effectiveness_l59_59376

variable (x : ℝ)
variable (y1 y2 : ℝ)
variable (hx : x > 300)

def costA (x : ℝ) : ℝ := 300 + 0.8 * (x - 300)
def costB (x : ℝ) : ℝ := 200 + 0.9 * (x - 200)

theorem cost_relationships :
  (y1 = costA x) ∧ (y2 = costB x) :=
by {
  -- Functional relationships
  split,
  -- Proof for y1
  { unfold costA,
    simp,
    exact rfl },
  -- Proof for y2
  { unfold costB,
    simp,
    exact rfl }
}

theorem cost_effectiveness (h₁ : y1 = costA x) (h₂ : y2 = costB x) :
  (300 < x ∧ x < 400 → y1 > y2) ∧ (x > 400 → y1 < y2) ∧ (x = 400 → y1 = y2) :=
by {
  split,
  -- Proof for 300 < x < 400
  { intro h,
    have h300 : 300 < x := h.left,
    have h400 : x < 400 := h.right,
    simp [h₁, h₂, costA, costB],
    linarith },
  split,
  -- Proof for x > 400
  { intro h,
    have h400 : x > 400 := h,
    simp [h₁, h₂, costA, costB],
    linarith },
  -- Proof for x = 400
  { intro h,
    have h_eq : x = 400 := h,
    simp [h₁, h₂, costA, costB],
    rw h_eq,
    norm_num }
}

end cost_relationships_cost_effectiveness_l59_59376


namespace mul_digits_example_l59_59314

theorem mul_digits_example (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : C = 2) (h8 : D = 5) : A + B = 2 := by
  sorry

end mul_digits_example_l59_59314


namespace infinitely_many_solutions_l59_59508

def circ (x y : ℝ) : ℝ := 4 * x - 3 * y + x * y

theorem infinitely_many_solutions : ∀ y : ℝ, circ 3 y = 12 := by
  sorry

end infinitely_many_solutions_l59_59508


namespace log_diff_l59_59200

theorem log_diff (x y : ℝ) (hx : log 4 256 = x) (hx_def : 4 ^ x = 256)
                (hy : log 4 (1/16) = y) (hy_def : 4 ^ y = 1 / 16) :
  log 4 256 - log 4 (1/16) = 6 :=
by
  -- We are ignoring the proof steps and focusing on constructing a valid, buildable statement
  sorry

end log_diff_l59_59200


namespace solution_set_x_l59_59806

theorem solution_set_x (x : ℝ) (h₁ : 33 * 32 ≤ x)
  (h₂ : ⌊x⌋ + ⌈x⌉ = 5) : 2 < x ∧ x < 3 :=
by
  sorry

end solution_set_x_l59_59806


namespace solve_for_m_l59_59722

theorem solve_for_m : ∃ (m : ℤ), (m - 6)^4 = (1 / 16 : ℚ) ^ (-2) ∧ m = 10 := 
by
  use 10
  rw [pow_neg, one_div, (1 / (16 : ℚ))^2]
  norm_num
  sorry

end solve_for_m_l59_59722


namespace distinct_solutions_abs_eq_five_l59_59724

theorem distinct_solutions_abs_eq_five :
  (∀ x : ℝ, |x - |3 * x - 2|| = 5 → (x = 7 / 2 ∨ x = -3 / 4)) ↔ (# solutions {x : ℝ | |x - |3 * x - 2|| = 5} = 2) :=
sorry

end distinct_solutions_abs_eq_five_l59_59724


namespace number_of_participants_l59_59113

theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 :=
by
  sorry

end number_of_participants_l59_59113


namespace count_pairs_77_l59_59982

theorem count_pairs_77 : 
  let pairs := (List.range 50).product (List.range 50)
                |>.filter (λ (i, j) => 0 ≤ i ∧ i < j ∧ j ≤ 49 ∧ (j - i) % 6 = 0) in
  pairs.length = 182 :=
by
  sorry

end count_pairs_77_l59_59982


namespace relationship_between_a_b_c_l59_59924

def a : ℝ := Real.log 7 / Real.log 2
def b : ℝ := Real.log 8 / Real.log 3
def c : ℝ := 0.3 ^ 0.2

theorem relationship_between_a_b_c : c < b ∧ b < a := by
  sorry

end relationship_between_a_b_c_l59_59924


namespace sandy_total_puppies_l59_59021

-- Definitions based on conditions:
def original_puppies : ℝ := 8.0
def additional_puppies : ℝ := 4.0

-- Theorem statement: total_puppies should be 12.0
theorem sandy_total_puppies : original_puppies + additional_puppies = 12.0 := 
by
  sorry

end sandy_total_puppies_l59_59021


namespace palindromic_numbers_with_two_even_digits_count_l59_59805

theorem palindromic_numbers_with_two_even_digits_count :
  let is_palindromic (n : Nat) := n.div 10000 = n % 10 ∧ (n % 10000).div 1000 = (n.div 10) % 10
  let num_palindromic := (List.range (90000) ++ List.range 10000).filter (λ n, is_palindromic (10000 + n))
    .filter (λ n, (List.range 5).count (λ i, (n / 10^i) % 10 % 2 = 0) = 2)
  num_palindromic.length = 225 := sorry

end palindromic_numbers_with_two_even_digits_count_l59_59805


namespace pieces_per_small_load_l59_59119

theorem pieces_per_small_load (total_clothing : ℕ) (load_1 : ℕ) (small_loads : ℕ) (num_small_loads : ℕ) : total_clothing = 47 → load_1 = 17 → num_small_loads = 5 → small_loads = (total_clothing - load_1) / num_small_loads → small_loads = 6 :=
by
  intros htc hl1 hnl hsl
  rw [htc, hl1, hnl] at hsl
  norm_num at hsl
  exact hsl
  sorry

end pieces_per_small_load_l59_59119


namespace college_enrollment_change_l59_59815

theorem college_enrollment_change (E : ℝ) :
  let E_1992 := 1.2 * E in
  let E_1993 := 1.15 * E_1992 in
  let E_1994 := 0.9 * E_1993 in
  let E_1995 := 1.25 * E_1994 in
  (E_1995 - E) / E * 100 = 55.25 :=
by
  let E_1992 := 1.2 * E
  let E_1993 := 1.15 * E_1992
  let E_1994 := 0.9 * E_1993
  let E_1995 := 1.25 * E_1994
  sorry

end college_enrollment_change_l59_59815


namespace fraction_sum_geq_zero_l59_59923

theorem fraction_sum_geq_zero (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a)) ≥ 0 := 
by 
  sorry

end fraction_sum_geq_zero_l59_59923


namespace fewer_cucumbers_than_potatoes_l59_59818

theorem fewer_cucumbers_than_potatoes :
  ∃ C : ℕ, 237 + C + 2 * C = 768 ∧ 237 - C = 60 :=
by
  sorry

end fewer_cucumbers_than_potatoes_l59_59818


namespace find_x_l59_59338

def x_y_conditions (x y : ℝ) : Prop :=
  x > y ∧
  x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40 ∧
  x * y + x + y = 8

theorem find_x (x y : ℝ) (h : x_y_conditions x y) : x = 3 + Real.sqrt 7 :=
by
  sorry

end find_x_l59_59338


namespace count_numbers_without_3_6_9_l59_59543

theorem count_numbers_without_3_6_9 : 
  let allowed_digits := {0, 1, 2, 4, 5, 7, 8}
  ∃ total : ℕ,
  (∃ n5 : ℕ, (6 * 7^4 = n5) ∧ n5 > 0 ∧ n5.val ∈ allowed_digits) ∧
  (∃ n6 : ℕ, (6 * 7^5 = n6) ∧ n6 > 0 ∧ n6.val ∈ allowed_digits) ∧
  total = 6 * 7^4 + 6 * 7^5 :=
by sorry

end count_numbers_without_3_6_9_l59_59543


namespace probability_of_letter_in_mathematics_l59_59618

theorem probability_of_letter_in_mathematics : 
  let alphabet_size := 26;
  let mathematics_letters := 8;
  let favorable_outcomes := mathematics_letters;
  let total_outcomes := alphabet_size;
  (favorable_outcomes / total_outcomes : ℚ) = (4 / 13 : ℚ) :=
by
  let alphabet_size := 26;
  let mathematics_letters := 8;
  let favorable_outcomes := mathematics_letters;
  let total_outcomes := alphabet_size;
  calc
    (favorable_outcomes / total_outcomes : ℚ) = (8 / 26 : ℚ) : by norm_num
                                     ... = (4 / 13 : ℚ) : by norm_num

end probability_of_letter_in_mathematics_l59_59618


namespace no_solution_fraction_eq_l59_59220

theorem no_solution_fraction_eq {x m : ℝ} : 
  (∀ x, ¬ (1 - x = 0) → (2 - x) / (1 - x) = (m + x) / (1 - x) + 1) ↔ m = 0 := 
by
  sorry

end no_solution_fraction_eq_l59_59220


namespace contrapositive_of_implication_l59_59269

theorem contrapositive_of_implication (p q : Prop) (h : p → q) : ¬q → ¬p :=
by {
  sorry
}

end contrapositive_of_implication_l59_59269


namespace height_radius_ratio_l59_59468

noncomputable def cone_height_radius_ratio (r h : ℝ) :=
  h / r

theorem height_radius_ratio
  (r h : ℝ)
  (h_cone : r^2 + h^2 = 400 * r^2)
  : cone_height_radius_ratio r h = 3 * real.sqrt 133 :=
by
  sorry

end height_radius_ratio_l59_59468


namespace result_of_operation_l59_59914

def operation (v : ℝ) := v - v / 3

def iterated_operation (v : ℝ) := operation (operation v)

theorem result_of_operation : iterated_operation 26.999999999999993 = 12.0 := by
  sorry

end result_of_operation_l59_59914


namespace find_odd_function_l59_59843

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def passes_through_points (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  f p1.1 = p1.2 ∧ f p2.1 = p2.2

def candidate_functions (x : ℝ) : List (ℝ → ℝ) :=
  [λ x, x^(1/2), λ x, x^5, λ x, x^(-3), λ x, x^(-1/3)]

theorem find_odd_function :
  ∃ f ∈ candidate_functions,
    is_odd_function f ∧
    passes_through_points f (0, 0) (1, 1) ∧
    f = λ x, x^5 :=
sorry

end find_odd_function_l59_59843


namespace variance_data_set_l59_59219

def data_set : List ℕ := [3, 2, 1, 0, 0, 0, 1]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let m = mean l
  (l.map (λ x => (x - m.toNat)^2)).sum / l.length

theorem variance_data_set :
  variance data_set = 8 / 7 :=
by
  sorry

end variance_data_set_l59_59219


namespace line_parallel_to_plane_l59_59622

-- Definition for a line as a set of points
def Line := set (ℝ × ℝ × ℝ)

-- Definition for a plane as a set of points
def Plane := set (ℝ × ℝ × ℝ)

-- Definition for parallelism between a line and a plane
def is_parallel_line_plane (l : Line) (p : Plane) : Prop := 
∀ (p1 : ℝ × ℝ × ℝ), p1 ∈ l → ∃ (p2 : ℝ × ℝ × ℝ), p2 ∈ p ∧ p1 = p2 -- This is just for example purposes; needs mathematical sound definition

-- Here we define countless lines within a plane, for simplicity, we use a set of lines
def countless_lines_in_plane (α : Plane): set Line := {l : Line | ∀ (p : ℝ × ℝ × ℝ), p ∈ α → p ∈ l}

-- Proof Problem
theorem line_parallel_to_plane (a : Line) (α : Plane) (h1 : ∀ l ∈ countless_lines_in_plane α, a ∥ l) : a ∥ α :=
sorry

end line_parallel_to_plane_l59_59622


namespace card_arrangements_l59_59361

-- Definitions
def cards : List ℕ := [1, 2, 3, 4, 5, 6]
def envelopes : ℕ := 3
def cards_per_envelope : ℕ := 2
def card_unit : List ℕ := [1, 2]

-- Theorem to prove the total number of different arrangements
theorem card_arrangements
  (h_all_cards : length cards = 6)
  (h_envelopes : envelopes = 3)
  (h_cards_per_envelope : cards_per_envelope = 2)
  (h_card_unit_in_envelope : ∃ e : ℕ, e < envelopes ∧ ∀ x ∈ card_unit, x = e):
  ∃ n : ℕ, n = 18 := by
  sorry

end card_arrangements_l59_59361


namespace systematic_sampling_correct_l59_59438

theorem systematic_sampling_correct :
  ∀ (n : ℕ), n = 50 →
  ∃ (selected_students : list ℕ),
  selected_students = [5, 15, 25, 35, 45] ∧
  (∀ i j, i < j ∧ i < selected_students.length ∧ j < selected_students.length → selected_students.nth i ≤ selected_students.nth j) ∧
  (∀ k : ℕ, k < 5 → selected_students.nth k = some (5 + 10 * k)) :=
by
  intro n hn
  rw hn
  use [5, 15, 25, 35, 45]
  constructor
  . refl
  constructor
  . intros i j hij
    cases i
    . cases j
      . intro
        simp
      . intro
        simp
    trivial
  . intro k hk
    cases k
    . simp
    . cases k
      . simp
      . cases k
        . simp
        . cases k
          . simp
          . cases k
            . simp
            . simp
  sorry

end systematic_sampling_correct_l59_59438


namespace intersection_points_distance_l59_59773

noncomputable def distance_of_intersection_points {r1 r2 : ℝ} (l : ℝ) : ℝ :=
  2 * r1 * r2 / (r1 + r2)

theorem intersection_points_distance (r1 r2 : ℝ) (l : ℝ) : 
  ∃ M : ℝ, ∀ O1 O2 : ℝ, distance_of_intersection_points l = 2 * r1 * r2 / (r1 + r2) :=
sorry

end intersection_points_distance_l59_59773


namespace pen_slides_forms_line_l59_59788

def point_movement_forms_line (start_point end_point : point) : Prop :=
  ∃ path : ℝ → point, continuous path ∧ path 0 = start_point ∧ path 1 = end_point

-- Define the concept of pen writing the letter C
def writes_C (pen_tip : point) : Prop :=
  ∃ (path : ℝ → point), continuous path ∧ 
  (∀ t ∈ Icc (0:ℝ) (1:ℝ), path t ∈ letter_C) -- assuming letter_C is predefined

theorem pen_slides_forms_line (pen_tip : point) :
  writes_C pen_tip → point_movement_forms_line (start_point_of pen_tip) (end_point_of pen_tip) :=
sorry

end pen_slides_forms_line_l59_59788


namespace final_score_proof_l59_59872

def final_score (bullseye_points : ℕ) (miss_points : ℕ) (half_bullseye_points : ℕ) : ℕ :=
  bullseye_points + miss_points + half_bullseye_points

theorem final_score_proof : final_score 50 0 25 = 75 :=
by
  -- Considering the given conditions
  -- bullseye_points = 50
  -- miss_points = 0
  -- half_bullseye_points = half of bullseye_points = 25
  -- Summing them up: 50 + 0 + 25 = 75
  sorry

end final_score_proof_l59_59872


namespace largest_integer_n_neg_l59_59528

theorem largest_integer_n_neg (n : ℤ) : (n < 8 ∧ 3 < n) ∧ (n^2 - 11 * n + 24 < 0) → n ≤ 7 := by
  sorry

end largest_integer_n_neg_l59_59528


namespace intersection_of_A_and_B_l59_59577

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l59_59577


namespace computer_production_per_day_l59_59134

theorem computer_production_per_day (p r d x : ℕ) (h1 : p = 150) (h2 : r = 1_575_000) (h3 : d = 7) :
  150 * (7 * x) = 1_575_000 → x = 1_500 :=
by
  intros h
  have h4 : 1050 * x = 1_575_000 := by rw [←h1, ←h3, mul_assoc, mul_comm 7, mul_assoc] at h; exact h
  have h5 : x = 1_500 := by rw h2 at h4; exact (Nat.div_eq_of_eq_mul_left (by norm_num) h4.symm)
  exact h5

end computer_production_per_day_l59_59134


namespace total_pairs_of_jeans_purchased_l59_59548

-- Definitions based on the problem conditions
def price_fox : ℝ := 15
def price_pony : ℝ := 18
def discount_save : ℝ := 8.64
def pairs_fox : ℕ := 3
def pairs_pony : ℕ := 2
def sum_discount_rate : ℝ := 0.22
def discount_rate_pony : ℝ := 0.13999999999999993

-- Lean 4 statement to prove the total number of pairs of jeans purchased
theorem total_pairs_of_jeans_purchased :
  pairs_fox + pairs_pony = 5 :=
by
  sorry

end total_pairs_of_jeans_purchased_l59_59548


namespace debby_bottles_l59_59865

noncomputable def number_of_bottles_initial : ℕ := 301
noncomputable def number_of_bottles_drank : ℕ := 144
noncomputable def number_of_bottles_left : ℕ := 157

theorem debby_bottles:
  (number_of_bottles_initial - number_of_bottles_drank) = number_of_bottles_left :=
sorry

end debby_bottles_l59_59865


namespace convex_quad_side_relative_length_l59_59691

noncomputable def relativelength (AB CD : ℝ) (AB_parallel_chord : ℝ) : ℝ :=
AB / AB_parallel_chord

theorem convex_quad_side_relative_length 
  (ABCD : Type) [quadrilateral ABCD]
  (h_convex: convex ABCD):
  ∃ s, (∃ AB_parallel_chord, s = relativelength s AB_parallel_chord ∧ s ≥ 1) 
  ∧ ∃ s, (∃ CD_parallel_chord, s = relativelength s CD_parallel_chord ∧ s ≤ 1) := 
sorry

end convex_quad_side_relative_length_l59_59691


namespace parametric_equation_C1_ordinary_equation_C3_intersection_PA_PB_l59_59952

-- Definitions based on the given problem conditions
def C1_polar := ∀ (ρ θ : ℝ), ρ * cos θ - ρ * sin θ + 2 = 0
def C1_parametric := ∀ (t : ℝ), (x = (real.sqrt 2) / 2 * t ∧ y = 2 + (real.sqrt 2) / 2 * t)

def C2_parametric := ∀ (α : ℝ), (x = cos α ∧ y = 2 * sin α)
def C3_transformed := ∀ (α : ℝ), (x = 3 * cos α ∧ y = 3 * sin α)
def C3_ordinary := ∀ (x y : ℝ), x^2 + y^2 = 9

-- Problem statements as Lean propositions
theorem parametric_equation_C1 : C1_parametric :=
sorry

theorem ordinary_equation_C3 : C3_ordinary :=
sorry

theorem intersection_PA_PB (P A B : ℝ × ℝ) (hP : P = (0, 2))
    (hIntersection : C1_parametric ∧ C3_ordinary ∧ (intersect_at_points P A B))
    : |PA| + |PB| = 2 * real.sqrt 7 :=
sorry

end parametric_equation_C1_ordinary_equation_C3_intersection_PA_PB_l59_59952


namespace mike_changed_64_tires_l59_59702

def total_tires_mike_changed (motorcycles : ℕ) (cars : ℕ) (tires_per_motorcycle : ℕ) (tires_per_car : ℕ) : ℕ :=
  motorcycles * tires_per_motorcycle + cars * tires_per_car

theorem mike_changed_64_tires :
  total_tires_mike_changed 12 10 2 4 = 64 :=
by
  sorry

end mike_changed_64_tires_l59_59702


namespace no_triangle_with_heights_1_2_3_l59_59649

open Real

theorem no_triangle_with_heights_1_2_3 :
  ¬(∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
     ∃ (k : ℝ), k > 0 ∧ 
       a * k = 1 ∧ b * (k / 2) = 2 ∧ c * (k / 3) = 3 ∧
       (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by 
  sorry

end no_triangle_with_heights_1_2_3_l59_59649


namespace ordered_triples_count_l59_59666

theorem ordered_triples_count : 
  ∃ (triples : set (ℕ × ℕ × ℕ)), 
  (∀ t ∈ triples, let (a, b, c) := t in Nat.lcm a b = 3000 ∧ Nat.lcm b c = 6000 ∧ Nat.lcm c a = 6000) ∧ 
  triples.finite ∧ 
  triples.card = 3 :=
by
sorry

end ordered_triples_count_l59_59666


namespace greatest_n_perfect_square_sum_l59_59899

theorem greatest_n_perfect_square_sum :
  ∃ n : ℕ, n ≤ 2023 ∧ 
  (∑ i in finset.range (n + 1), i ^ 2) * 
  (∑ i in finset.range (2 * n + 1) \ finset.range (n + 1), i ^ 2) = 1921 ^ 2 :=
sorry

end greatest_n_perfect_square_sum_l59_59899


namespace egor_encryption_l59_59513

-- Define the problem statement
def unique_digit_encoding (l: List Char) : Prop :=
  l.all Char.isDigit ∧ l.nodup

-- Define the function checking divisibility by 8 for the last three digits
def divisible_by_8 (l : List ℕ) : Prop :=
  (l.length ≥ 3) → ((List.last l 0 + 10 * (List.get l (l.length - 2) 0) + 100 * (List.get l (l.length - 3) 0)) % 8 = 0)

-- Define the encoded version of the number
def encoded_number (letters : List Char) (digits : List ℕ) :=
  letters.map (fun c => digits.nthLe (letters.indexOf c) sorry)

-- Define the valid letters
def valid_letters : List Char := ['G', 'V', 'A', 'T', 'E', 'M', 'L', 'A']

theorem egor_encryption :
  ∃ digits : List ℕ,
    unique_digit_encoding valid_letters ∧
    divisible_by_8 (encoded_number valid_letters digits) ∧
    (List.permutations digits).length = 100800
:= by
  sorry

end egor_encryption_l59_59513


namespace max_mod_u_is_5_and_z_neg1_l59_59283

noncomputable def max_value_mod_u (z : ℂ) (hz : |z| = 1) : ℝ :=
  complex.abs (z^4 - z^3 - 3 * z^2 * complex.I - z + 1)

theorem max_mod_u_is_5_and_z_neg1 (z : ℂ) (hz : |z| = 1) :
  ∃ z_max : ℂ, z_max = -1 ∧ max_value_mod_u z hz ≤ max_value_mod_u z_max hz :=
begin
  sorry

end max_mod_u_is_5_and_z_neg1_l59_59283


namespace sec_neg_420_eq_2_l59_59888

theorem sec_neg_420_eq_2 : ∀ (x : ℝ), (x = -420) → 
  (∀ y : ℝ, sec y = 1 / cos y) →
  (∀ (z k : ℝ), cos (z + 360 * k) = cos z) →
  (cos 60 = 1 / 2) →
  sec x = 2 :=
by
  intros x hx hsec_cos hcos_period hcos_60
  sorry

end sec_neg_420_eq_2_l59_59888


namespace sequence_term_306_l59_59313

theorem sequence_term_306 (a1 a2 : ℤ) (r : ℤ) (n : ℕ) (h1 : a1 = 7) (h2 : a2 = -7) (h3 : r = -1) (h4 : a2 = r * a1) : 
  ∃ a306 : ℤ, a306 = -7 ∧ a306 = a1 * r^305 :=
by
  use -7
  sorry

end sequence_term_306_l59_59313


namespace total_meal_cost_l59_59118

variables (cost_per_adult : ℕ) (cost_per_kid : ℕ) (total_number_of_people : ℕ) (number_of_kids : ℕ)

theorem total_meal_cost : cost_per_adult = 3 → cost_per_kid = 0 → total_number_of_people = 12 → number_of_kids = 7 → 
                          let number_of_adults := total_number_of_people - number_of_kids in
                          let total_cost := number_of_adults * cost_per_adult + number_of_kids * cost_per_kid in
                          total_cost = 15 :=
by
  intros h_cost_per_adult h_cost_per_kid h_total_number h_number_of_kids
  let number_of_adults := total_number_of_people - number_of_kids
  let total_cost := number_of_adults * cost_per_adult + number_of_kids * cost_per_kid
  change total_cost = 15
  rw [h_cost_per_adult, h_cost_per_kid, h_total_number, h_number_of_kids]
  change (12 - 7) * 3 + 7 * 0 = 15
  norm_num
  sorry

end total_meal_cost_l59_59118


namespace maximum_area_of_rectangle_l59_59144

theorem maximum_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : ∃ A, A = 100 ∧ ∀ x' y', 2 * x' + 2 * y' = 40 → x' * y' ≤ A := by
  sorry

end maximum_area_of_rectangle_l59_59144


namespace circle_iff_m_gt_neg_1_over_2_l59_59285

noncomputable def represents_circle (m: ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + x + y - m = 0) → m > -1/2

theorem circle_iff_m_gt_neg_1_over_2 (m : ℝ) : represents_circle m ↔ m > -1/2 := by
  sorry

end circle_iff_m_gt_neg_1_over_2_l59_59285


namespace max_surface_area_of_triangular_prisms_l59_59930

def length : ℕ := 5
def width : ℕ := 4
def height : ℕ := 3

def surface_area_case1 : ℕ :=
  2 * (length * width + 2 * (length * height + width * height))

def surface_area_case2 : ℕ :=
  2 * (length * height + 2 * (length * width + height * width))

theorem max_surface_area_of_triangular_prisms :
  max surface_area_case1 surface_area_case2 = 158 := by
  sorry

end max_surface_area_of_triangular_prisms_l59_59930


namespace length_MN_l59_59490

theorem length_MN (a : ℝ) (E F G H M N : ℝ × ℝ)
(hE : E = (a / 2, 0)) (hF : F = (a, a / 2)) (hG : G = (a / 2, a))
(hM : M = ((a/2 + a) / 2, (0 + a/2) / 2)) (hN : N = ((a + a/2) / 2, (a/2 + a) / 2))
(h_area : a ^ 2 / 4 = 11) :
    dist M N = sqrt 11 :=
begin
  sorry
end

end length_MN_l59_59490


namespace polar_to_rectangular_l59_59953

-- Define the radius and angle for the polar coordinates
def r : ℝ := 5
def theta : ℝ := π / 3

-- Define the expected rectangular coordinates
def x : ℝ := 5 / 2
def y : ℝ := 5 * real.sqrt(3) / 2

theorem polar_to_rectangular :
  (r * real.cos theta, r * real.sin theta) = (x, y) :=
sorry

end polar_to_rectangular_l59_59953


namespace percentile_75th_correct_l59_59549

def pearl_weights : List ℝ := [7.9, 9.0, 8.9, 8.6, 8.4, 8.5, 8.5, 8.5, 9.9, 7.8, 8.3, 8.0]

noncomputable def sorted_pearl_weights : List ℝ := List.sort pearl_weights

noncomputable def percentile_75th (weights : List ℝ) : ℝ :=
  let sorted_weights := List.sort weights
  sorted_weights.getD 8 0 -- 9th element considering 0-based indexing
  
theorem percentile_75th_correct : percentile_75th pearl_weights = 8.6 :=
by
  sorry

end percentile_75th_correct_l59_59549


namespace parallelogram_midpoints_XY_square_l59_59309

theorem parallelogram_midpoints_XY_square (A B C D X Y : ℝ)
  (AB CD : ℝ) (BC DA : ℝ) (angle_D : ℝ)
  (mid_X : X = (B + C) / 2) (mid_Y : Y = (D + A) / 2)
  (h1: AB = 10) (h2: BC = 17) (h3: CD = 10) (h4 : angle_D = 60) :
  (XY ^ 2 = 219 / 4) :=
by
  sorry

end parallelogram_midpoints_XY_square_l59_59309


namespace four_gt_sqrt_fourteen_l59_59861

theorem four_gt_sqrt_fourteen : 4 > Real.sqrt 14 := 
  sorry

end four_gt_sqrt_fourteen_l59_59861


namespace existence_of_fixed_point_l59_59334

variable {S : Type}
variable {P : set (set S)}
variable {f : set S → set S}

axiom f_inclusion_preserve : ∀ {X Y : set S}, X ⊆ Y → f(X) ⊆ f(Y)

def is_superset {S : Type} (A : set S) (f : set S → set S) := A ⊆ f(A)

noncomputable def K (S : Type) (f : set S → set S) :=
  ⋃ A ∈ { A : set S | is_superset A f }

theorem existence_of_fixed_point :
  ∃ K : set S, K = f(K) :=
by
  let K := K S f
  use K
  sorry

end existence_of_fixed_point_l59_59334


namespace tan_alpha_neg_4_over_3_l59_59230

variable (α : ℝ)

def f (α : ℝ) : ℝ := 
  (Math.tan (Real.pi - α) * Math.cos (2 * Real.pi - α) * Math.sin (Real.pi / 2 + α)) / Math.cos (Real.pi + α)

theorem tan_alpha_neg_4_over_3 (h1 : f(α) = (Math.tan (Real.pi - α) * Math.cos (2 * Real.pi - α) * Math.sin (Real.pi / 2 + α)) / Math.cos (Real.pi + α))
  (h2 : f(Real.pi / 2 - α) = -3 / 5) (h3 : Real.pi / 2 < α ∧ α < Real.pi) : Math.tan α = -4 / 3 := 
  by
  sorry

end tan_alpha_neg_4_over_3_l59_59230


namespace sin_double_angle_tangent_identity_l59_59921

theorem sin_double_angle_tangent_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.sin (2 * x) = 3 / 5 :=
by
  -- proof is omitted
  sorry

end sin_double_angle_tangent_identity_l59_59921


namespace sum_series_x_k_l59_59757

noncomputable def x : ℕ → ℕ
| 0     := 115
| (n+1) := x n ^ 2 + x n

theorem sum_series_x_k :
  (∑' k, 1 / (x k + 1) : ℝ) = 1 / 115 :=
sorry

end sum_series_x_k_l59_59757


namespace abs_neg_sqrt_two_eq_sqrt_two_l59_59730

theorem abs_neg_sqrt_two_eq_sqrt_two : abs (-real.sqrt 2) = real.sqrt 2 := 
by sorry

end abs_neg_sqrt_two_eq_sqrt_two_l59_59730


namespace least_integer_solution_l59_59090

theorem least_integer_solution (x : ℤ) : (∀ y : ℤ, |2 * y + 9| <= 20 → x ≤ y) ↔ x = -14 := by
  sorry

end least_integer_solution_l59_59090


namespace part1_part2_part3_l59_59509

-- Definition of opposite equations
def opposite_equations (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) : Prop :=
  ∀ x, (a * x - b = 0) ↔ (b * x - a = 0)

-- Part (1)
theorem part1 (c : ℝ) (h : opposite_equations 4 3 (by norm_num) (by norm_num)) : c = 4 :=
sorry

-- Part (2)
theorem part2 (m n : ℝ)
  (h1 : opposite_equations 4 (-3 * m - 1) (by norm_num) (by norm_num))
  (h2 : opposite_equations 5 (n - 2) (by norm_num) (by norm_num)) : m / n = -1 / 3 :=
sorry

-- Part (3)
theorem part3 (c : ℤ)
  (h1 : (∃ x : ℤ, 3 * x - c = 0))
  (h2 : (∃ x : ℤ, c * x - 3 = 0)) : c = 3 ∨ c = -3 :=
sorry

end part1_part2_part3_l59_59509


namespace bouquet_daisies_percentage_l59_59462

theorem bouquet_daisies_percentage :
  (∀ (total white yellow white_tulips white_daisies yellow_tulips yellow_daisies : ℕ),
    total = white + yellow →
    white = 7 * total / 10 →
    yellow = total - white →
    white_tulips = white / 2 →
    white_daisies = white / 2 →
    yellow_daisies = 2 * yellow / 3 →
    yellow_tulips = yellow - yellow_daisies →
    (white_daisies + yellow_daisies) * 100 / total = 55) :=
by
  intros total white yellow white_tulips white_daisies yellow_tulips yellow_daisies h_total h_white h_yellow ht_wd hd_wd hd_yd ht_yt
  sorry

end bouquet_daisies_percentage_l59_59462


namespace ron_increase_spending_l59_59625

variable (P Q : ℝ) -- Original price and quantity
variable (x : ℝ) -- The percentage increase in spending we're proving is correct

theorem ron_increase_spending :
  let new_price := 1.25 * P
  let new_quantity := 0.91999999999999993 * Q
  let original_spending := P * Q
  let new_spending := original_spending * (1 + x / 100)
  new_spending = new_price * new_quantity → x = 15 :=
begin
  sorry
end

end ron_increase_spending_l59_59625


namespace unique_function_inverse_l59_59927

noncomputable def function_is_inverse (f : ℝ → ℝ) :=
  ∀ x y : ℝ, 0 < x → 0 < y → f(x * f(y)) = y * f(x)

noncomputable def limit_at_infinity (f : ℝ → ℝ) :=
  filter.tendsto f filter.at_top (𝓝 0)

theorem unique_function_inverse (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, 0 < x → 0 < f(x)) 
  (h2 : function_is_inverse f) 
  (h3 : limit_at_infinity f) : 
  ∀ x : ℝ, 0 < x → f(x) = 1 / x := 
sorry

end unique_function_inverse_l59_59927


namespace E_xi_to_the_p_le_E_eta_to_the_p_over_p_minus_q_l59_59330

variables {Ω : Type*} [MeasureSpace Ω]
variables {p q : ℝ} (hpq : p > q) (hq1 : q > 1)
variables {ξ η : Ω → ℝ} (hnn : ∀ ω, 0 ≤ ξ ω ∧ 0 ≤ η ω)

-- The given condition
axiom E_xi_to_the_p_le_E_xi_to_the_q_eta : 
  E (λ ω, ξ ω ^ p) ≤ E (λ ω, ξ ω ^ q * η ω)

-- The goal to prove
theorem E_xi_to_the_p_le_E_eta_to_the_p_over_p_minus_q : 
  E (λ ω, ξ ω ^ p) ≤ E (λ ω, η ω ^ (p / (p - q))) :=
sorry

end E_xi_to_the_p_le_E_eta_to_the_p_over_p_minus_q_l59_59330


namespace max_simple_subset_l59_59833

def is_simple_subset (S : Set ℝ) : Prop :=
  ∀ (x y z : ℝ), x ∈ S → y ∈ S → (x + y = z) → z ∈ S

theorem max_simple_subset {n : ℕ} : ∃ (S : Set ℝ), is_simple_subset S ∧ S ⊆ (set.range (λ k, (k : ℕ) + 1)) ∧ S.card = n + 1 :=
sorry

end max_simple_subset_l59_59833


namespace sum_of_reciprocals_of_roots_l59_59906

theorem sum_of_reciprocals_of_roots :
  (∀ r1 r2 : Rat, r1 + r2 = 26 ∧ r1 * r2 = 12 → 1 / r1 + 1 / r2 = 13 / 6) :=
by
  intros r1 r2 h
  cases h with h_sum h_product
  rw [← add_div, h_sum, h_product]
  norm_num

end sum_of_reciprocals_of_roots_l59_59906


namespace main_proof_l59_59246

noncomputable def polar_curve (θ a : ℝ) :=
  ρ = a * Real.sin (θ + π / 3)

def parametric_line_eq (t m : ℝ) :=
  (x, y) = (-3 * t, m + √3 * t)

def rectangular_curve_eq (x y : ℝ) :=
  (x - √3) ^ 2 + (y - 1) ^ 2 = 4

def polar_to_rectangular (ρ θ : ℝ) :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def symmetric_about_line (M N : ℝ × ℝ) (l : ℝ → ℝ × ℝ) :=
  ∃ t m, M.y = m + √3 * t ∧ N.y = m - √3 * t

theorem main_proof :
  let a := 4
  ∀ (m : ℝ) (t : ℝ),
    (parametric_line_eq t m) →
    (polar_curve (π / 6) a) →
    (polar_to_rectangular 4 (π / 6) = (2 * √3, 2)) →
    (symmetric_about_line (2 * √3, 2) N parametric_line_eq) →
    (N ∈ rectangular_curve_eq)
    ∧ (∃ d : ℝ, dist M (line_eq m N) = d ∨ dist N M = 2 * √3)
:= sorry

end main_proof_l59_59246


namespace number_of_members_l59_59299

theorem number_of_members 
  (num_committees : ℕ)
  (belongs_to_two_committees : ∀ (member : Type) [Fintype member], ∀ (m : member), Fintype (Subtype (λ c : Finset ℕ, c.card = 2)))
  (exactly_one_member_in_common : ∀ (committee1 committee2 : Finset ℕ), committee1 ≠ committee2 → Fintype (Subtype (λ m, committee1 ∈ m ∧ committee2 ∈ m))) :
  (num_committees = 5) → Fintype.card (Subtype (λ m : (Finset ℕ × Finset ℕ), m.1.card = 2 ∧ m.2.card = 2)) = 10 := 
begin
  sorry
end

end number_of_members_l59_59299


namespace mt_eq_4_l59_59328

-- Define T as the set of all nonzero real numbers.
def T := { x : ℝ // x ≠ 0 }

-- Define the function g : T → T
def g (x : T) : T := sorry -- Placeholder for the actual function

-- Conditions given in the problem:
axiom cond1 (x : T) : g ⟨1 / x, sorry⟩ = x.val * g x
axiom cond2 (x y : T) (h : (x.val + y.val) ≠ 0) : g ⟨1 / x.val, sorry⟩ + g ⟨1 / y.val, sorry⟩ = 2 + g ⟨1 / (x.val + y.val), sorry⟩

-- The number of possible values of g(1)
def m : ℕ := sorry

-- The sum of all possible values of g(1)
def t : ℕ := sorry

-- The final requirement of the problem:
theorem mt_eq_4 : m * t = 4 := sorry

end mt_eq_4_l59_59328


namespace inequality_proof_l59_59661

theorem inequality_proof 
  (a b c d : ℝ) 
  (h₀ : a ≥ 0) 
  (h₁ : b ≥ 0) 
  (h₂ : c ≥ 0) 
  (h₃ : d ≥ 0) 
  (h₄ : a + b + c + d = 4) : 
  a * real.sqrt (3 * a + b + c) + b * real.sqrt (3 * b + c + d) + c * real.sqrt (3 * c + d + a) + d * real.sqrt (3 * d + a + b) ≥ 4 * real.sqrt 5 :=
  sorry

end inequality_proof_l59_59661


namespace sum_distances_tomas_tyler_l59_59918

/-- Defining the distances run by the runners --/
def distance_katarina : ℕ := 51
def distance_harriet : ℕ := 48
def total_distance : ℕ := 195
def distance_tomas : ℕ
def distance_tyler : ℕ

/-- The sum of distances run by Tomas and Tyler --/
theorem sum_distances_tomas_tyler :
  distance_katarina + distance_tomas + distance_tyler + distance_harriet = total_distance → 
  distance_tomas + distance_tyler = 96 :=
by
  -- Insert proof steps here
  sorry

end sum_distances_tomas_tyler_l59_59918


namespace triangle_intersection_inequality_l59_59761

theorem triangle_intersection_inequality 
  (A B C M O : Point)
  (AB BC AM MC OB OM : ℝ)
  (h1 : AB = BC)
  (h2 : AM + MC = AB + BC)
  (h3 : intersects MC AB O) :
  OB > OM :=
sorry

end triangle_intersection_inequality_l59_59761


namespace initial_principal_amount_l59_59380

theorem initial_principal_amount (A : ℝ) (P : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) 
  (hA : A = 8820) (hr : r = 0.05) (ht : t = 2) (hn : n = 1) : 
  A = P * (1 + r/n)^(n*t) → P = 8000 :=
by
  sorry

end initial_principal_amount_l59_59380


namespace highest_percentage_difference_in_June_l59_59512

def sales := (drummers : ℕ) (bugle_players : ℕ) (flute_players : ℕ)

noncomputable def percentage_difference (y : ℕ) (x z : ℕ) : ℝ :=
  (abs (y - (x + z) / 2) / ((x + z) / 2)) * 100

theorem highest_percentage_difference_in_June :
  let months := [
      sales 5 4 3, -- January
      sales 7 5 5, -- February
      sales 6 6 8, -- March
      sales 5 8 7, -- April
      sales 8 6 9, -- May
      sales 9 7 5  -- June
    ],
    percentages := months.map (λ ⟨d, b, f⟩, max (percentage_difference d b f) (max (percentage_difference b d f) (percentage_difference f d b)))
  in max_list percentages = 50 := sorry

end highest_percentage_difference_in_June_l59_59512


namespace funnel_paper_area_l59_59450

theorem funnel_paper_area
  (slant_height : ℝ)
  (base_circumference : ℝ)
  (h1 : slant_height = 6)
  (h2 : base_circumference = 6 * Real.pi):
  (1 / 2) * base_circumference * slant_height = 18 * Real.pi :=
by
  sorry

end funnel_paper_area_l59_59450


namespace asymptotes_of_hyperbola_at_focus_l59_59248

theorem asymptotes_of_hyperbola_at_focus (m : ℝ) (hf : m = 16) :
  one_focus_of_hyperbola (x y : ℝ) : ℝ = 9 ∧ ℝ = m ∧ (F = (-5, 0))  :=
by sorry

end asymptotes_of_hyperbola_at_focus_l59_59248


namespace insurance_deduction_percentage_l59_59419

-- Define the conditions
variables (hours_worked : ℕ) (hourly_pay : ℕ) (tax_rate : ℕ) (union_dues : ℕ) (take_home_pay : ℕ)
variables (gross_earnings : ℕ)

-- Given conditions
def conditions : Prop := 
  hours_worked = 42 ∧ 
  hourly_pay = 10 ∧ 
  tax_rate = 20 ∧ 
  union_dues = 5 ∧ 
  take_home_pay = 310

-- Calculate gross earnings
def calculate_gross_earnings : ℕ := hours_worked * hourly_pay

-- Calculate tax deduction
def calculate_tax_deduction (gross_earnings : ℕ) : ℕ := (gross_earnings * tax_rate) / 100

-- Define the problem statement
theorem insurance_deduction_percentage (h : conditions): 
  let gross_earnings := calculate_gross_earnings in
  let tax_deduction := calculate_tax_deduction gross_earnings in
  let earnings_after_tax_and_union_dues := gross_earnings - tax_deduction - union_dues in
  let insurance_deduction := earnings_after_tax_and_union_dues - take_home_pay in
  let insurance_deduction_percentage := (insurance_deduction * 100) / gross_earnings in
  insurance_deduction_percentage = 5 :=
by 
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h_rest,
  cases h_rest with h4 h5,
  rw [h1, h2, h3, h4, h5], 
  sorry

end insurance_deduction_percentage_l59_59419


namespace linear_function_no_third_quadrant_l59_59311

theorem linear_function_no_third_quadrant :
  ∀ x y : ℝ, (y = -5 * x + 2023) → ¬ (x < 0 ∧ y < 0) := 
by
  intros x y h
  sorry

end linear_function_no_third_quadrant_l59_59311


namespace area_AEDF_proof_l59_59168

variables (S_CDF S_BCD S_BDE: ℝ)

-- Given areas of the triangles
def given_areas : Prop :=
  S_CDF = 3 ∧ S_BCD = 7 ∧ S_BDE = 7

-- The area of quadrilateral AEDF is 18
def area_quadrilateral_AEDF (S_CDF S_BCD S_BDE: ℝ) : Prop :=
  S_CDF = 3 → S_BCD = 7 → S_BDE = 7 → 18

-- Proof outline: given areas imply the area of AEDF is 18
theorem area_AEDF_proof (h : given_areas S_CDF S_BCD S_BDE) : 
  area_quadrilateral_AEDF S_CDF S_BCD S_BDE :=
  by {
    -- Sorry to skip the actual proof
    sorry,
  }

end area_AEDF_proof_l59_59168


namespace difference_greater_value_l59_59794

-- Define the conditions for the problem
def eighty_percent_of_sixty : ℝ := (80/100) * 60
def four_fifths_of_twentyfive : ℝ := (4/5) * 25

-- Define the proof goal
theorem difference_greater_value : eighty_percent_of_sixty - four_fifths_of_twentyfive = 28 := 
by
  -- skip the proof
  sorry

end difference_greater_value_l59_59794


namespace general_formula_a_comparison_inequality1_comparison_inequality2_l59_59965

open BigOperators

variable {R : Type*} [LinearOrderedField R]

def S (n : ℕ) (k : R) : R := 2 * 3^n + k
def a (n : ℕ) (k : R) : R := 4 * 3^(n-1)
def b (n : ℕ) (k : R) : R := (n - 1) / (4 * 3^(n-1))
def T (n : ℕ) (k : R) : R := ∑ i in finset.range n, b (i+1) k

theorem general_formula_a (n : ℕ) (k : R) (h : 1 ≤ n) : 
  a n k = S n k - S (n-1) k := by
  sorry

theorem comparison_inequality1 (n : ℕ) (k : R) (h : n > 5) :
  3 - 16 * T n k < 4 * (n + 1) * b (n+1) k := by
  sorry

theorem comparison_inequality2 (n : ℕ) (k : R) (h1 : 1 ≤ n) (h2 : n ≤ 5) :
  3 - 16 * T n k > 4 * (n + 1) * b (n+1) k := by
  sorry

end general_formula_a_comparison_inequality1_comparison_inequality2_l59_59965


namespace shorter_piece_is_20_l59_59442

def shorter_piece_length (total_length : ℕ) (ratio : ℚ) (shorter_piece : ℕ) : Prop :=
    shorter_piece * 7 = 2 * (total_length - shorter_piece)

theorem shorter_piece_is_20 : ∀ (total_length : ℕ) (shorter_piece : ℕ), 
    total_length = 90 ∧
    shorter_piece_length total_length (2/7 : ℚ) shorter_piece ->
    shorter_piece = 20 :=
by
  intro total_length shorter_piece
  intro h
  have h_total_length : total_length = 90 := h.1
  have h_equation : shorter_piece_length total_length (2/7 : ℚ) shorter_piece := h.2
  sorry

end shorter_piece_is_20_l59_59442


namespace find_Q_l59_59988

theorem find_Q (Q P : Real) (h : Q > 0) 
  (eq : (∛(13 * Q + 6 * P + 1) - ∛(13 * Q - 6 * P - 1)) = ∛2) : 
  Q = 7 := 
sorry

end find_Q_l59_59988


namespace maximize_x2y5_l59_59672

theorem maximize_x2y5 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 50) : 
  x = 100 / 7 ∧ y = 250 / 7 :=
sorry

end maximize_x2y5_l59_59672


namespace office_person_count_l59_59034

theorem office_person_count
    (N : ℕ)
    (avg_age_all : ℕ)
    (num_5 : ℕ)
    (avg_age_5 : ℕ)
    (num_9 : ℕ)
    (avg_age_9 : ℕ)
    (age_15th : ℕ)
    (h1 : avg_age_all = 15)
    (h2 : num_5 = 5)
    (h3 : avg_age_5 = 14)
    (h4 : num_9 = 9)
    (h5 : avg_age_9 = 16)
    (h6 : age_15th = 86)
    (h7 : 15 * N = (num_5 * avg_age_5) + (num_9 * avg_age_9) + age_15th) :
    N = 20 :=
by
    -- Proof will be provided here
    sorry

end office_person_count_l59_59034


namespace pilot_fish_speed_is_30_l59_59655

-- Define the initial conditions
def keanu_speed : ℝ := 20
def shark_initial_speed : ℝ := keanu_speed
def shark_speed_increase_factor : ℝ := 2
def pilot_fish_speed_increase_factor : ℝ := 0.5

-- Calculating final speeds
def shark_final_speed : ℝ := shark_initial_speed * shark_speed_increase_factor
def shark_speed_increase : ℝ := shark_final_speed - shark_initial_speed
def pilot_fish_speed_increase : ℝ := shark_speed_increase * pilot_fish_speed_increase_factor
def pilot_fish_final_speed : ℝ := keanu_speed + pilot_fish_speed_increase

-- The statement to prove
theorem pilot_fish_speed_is_30 : pilot_fish_final_speed = 30 := by
  sorry

end pilot_fish_speed_is_30_l59_59655


namespace purse_multiple_of_wallet_l59_59182

theorem purse_multiple_of_wallet (W P : ℤ) (hW : W = 22) (hc : W + P = 107) : ∃ n : ℤ, n * W > P ∧ n = 4 :=
by
  sorry

end purse_multiple_of_wallet_l59_59182


namespace trains_clear_time_l59_59082

noncomputable def length_first_train : ℝ := 111
noncomputable def length_second_train : ℝ := 165
noncomputable def speed_first_train_kmh : ℝ := 100
noncomputable def speed_second_train_kmh : ℝ := 120

def speed_conversion (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

def total_distance : ℝ :=
  length_first_train + length_second_train

def relative_speed_mps : ℝ :=
  speed_conversion speed_first_train_kmh + speed_conversion speed_second_train_kmh

def time_to_clear_trains : ℝ :=
  total_distance / relative_speed_mps

theorem trains_clear_time :
  time_to_clear_trains ≈ 4.51 := sorry

end trains_clear_time_l59_59082


namespace part1_tangent_line_part2_a_range_l59_59265

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := real.log (1 + x) + a * x * real.exp (-x)

-- Part (1)
theorem part1_tangent_line (x : ℝ) (h : x = 0) :
  (∀ y : ℝ, y = f 1 x → ∃ m : ℝ, m = 2 ∧ y = m * x) :=
by sorry

-- Part (2)
theorem part2_a_range (a : ℝ) :
  (∀ x : ℝ, (x ∈ set.Ioo (-1 : ℝ) 0 ∨ x ∈ set.Ioo 0 (⊤ : ℝ)) → (f a x = 0 → - ∞ < a ∧ a < - 1)) :=
by sorry

end part1_tangent_line_part2_a_range_l59_59265


namespace equivalent_conditions_l59_59712

def condition1 (x y : ℝ) : Prop :=
  abs x ≤ 1 ∧ abs y ≤ 1 ∧ x * y ≤ 0

def condition2 (x y : ℝ) : Prop :=
  abs x ≤ 1 ∧ abs y ≤ 1 ∧ x^2 + y^2 ≤ 1

def inequality (x y : ℝ) : Prop :=
  (√(1 - x^2) * √(1 - y^2)) ≥ (x * y)

theorem equivalent_conditions (x y : ℝ) : inequality x y ↔ condition1 x y ∨ condition2 x y := 
sorry

end equivalent_conditions_l59_59712


namespace total_books_l59_59175

theorem total_books :
  let benny_books := 24
  let sandy_share := benny_books / 3
  let benny_after := benny_books - sandy_share
  let tim_books := 33
  let katy_share := tim_books * 20 / 100
  let tim_after := tim_books - nat.floor katy_share
  let nathan_books := 15
  let sarah_books := 2 * benny_after
  let total := benny_after + tim_after + nathan_books + sarah_books
  total = 90 :=
by
  sorry

end total_books_l59_59175


namespace airplane_seats_total_l59_59486

theorem airplane_seats_total (s : ℝ) 
  (c1 : 24)
  (c2 : 0.25 * s)
  (c3 : (2/3) * s) :
  24 + 0.25 * s + (2/3) * s = s → s = 288 :=
by
  sorry

end airplane_seats_total_l59_59486


namespace sine_addition_formula_equation_delta_zero_conditions_l59_59232

noncomputable def z_poly (x y z : ℝ) : ℝ :=
  z^4 - 2 * (x^2 + y^2 - 2 * x^2 * y^2) * z^2 + (x^2 - y^2)^2

theorem sine_addition_formula_equation (x y z : ℝ) (hx : x = sin α) (hy : y = sin β) (hz : z = sin(α + β)) :
  z_poly x y z = 0 := 
sorry

theorem delta_zero_conditions (x y : ℝ): 
  (4 * x^2 * y^2 * (1 - x^2) * (1 - y^2) = 0) ↔ (x = 0 ∨ y = 0 ∨ |x| = 1 ∨ |y| = 1) :=
sorry

end sine_addition_formula_equation_delta_zero_conditions_l59_59232


namespace largest_n_multiple_3_l59_59089

theorem largest_n_multiple_3 (n : ℕ) (h1 : n < 100000) (h2 : (8 * (n + 2)^5 - n^2 + 14 * n - 30) % 3 = 0) : n = 99999 := 
sorry

end largest_n_multiple_3_l59_59089


namespace total_cost_l59_59471

def cost_of_items (x y : ℝ) : Prop :=
  (6 * x + 5 * y = 6.10) ∧ (3 * x + 4 * y = 4.60)

theorem total_cost (x y : ℝ) (h : cost_of_items x y) : 12 * x + 8 * y = 10.16 :=
by
  sorry

end total_cost_l59_59471


namespace equidistant_line_existence_l59_59210

def line_passing_through_intersection_and_equidistant (A B : Point) : Prop :=
  ∃ l : Line, 
  (A = ⟨-3, 1⟩ ∧ B = ⟨5, 7⟩) ∧
  (∃ x y : ℝ, 
    (2 * x + 7 * y - 4 = 0) ∧
    (7 * x - 21 * y - 1 = 0) ∧
    ((l = ⟨21 * x - 28 * y - 13, 0⟩ ∨ l = ⟨x, 1⟩) ∧ l.passes_through ⟨x, y⟩) ∧
    l.is_equidistant_from ⟨-3, 1⟩ ⟨5, 7⟩)

theorem equidistant_line_existence : ∃ l : line,
  line_passing_through_intersection_and_equidistant ⟨-3, 1⟩ ⟨5, 7⟩ := sorry

end equidistant_line_existence_l59_59210


namespace sum_S40_l59_59390

noncomputable def a (n : ℕ) : ℤ :=
  n * (Int.cos (n * Int.pi / 4))^2 - n * (Int.sin (n * Int.pi / 4))^2

noncomputable def S (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), a i

theorem sum_S40 : S 40 = 20 :=
by
  sorry

end sum_S40_l59_59390


namespace triangle_position_after_rolling_l59_59835

-- Define basic geometrical properties
def inner_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

-- Define the rotation per movement for a square around a regular polygon
def rotation_per_movement (n : ℕ) : ℝ := 360 - (inner_angle n + 90)

-- Total rotation after rolling over k sides
def total_rotation (n k : ℕ) : ℝ := k * rotation_per_movement n

-- Main theorem statement
theorem triangle_position_after_rolling (n k : ℕ) : total_rotation 8 2 = 270 :=
by
  sorry

end triangle_position_after_rolling_l59_59835


namespace correct_average_marks_l59_59824

theorem correct_average_marks (n : ℕ) (incorrect_avg correct_marks incorrect_marks : ℝ) 
  (hn : n = 35) (hincorrect_avg : incorrect_avg = 72) (hincorrect_marks : incorrect_marks = 46) 
  (hcorrect_marks : correct_marks = 56) : 
  let incorrect_total := incorrect_avg * n in
  let difference := incorrect_marks - correct_marks in
  let correct_total := incorrect_total + difference in
  let correct_avg := correct_total / n in
  abs (correct_avg - 71.71) < 0.01 :=
by 
  sorry

end correct_average_marks_l59_59824


namespace A_cannot_win_at_least_k_6_l59_59337

noncomputable def min_value_k_no_win_for_A : ℕ := 6

theorem A_cannot_win_at_least_k_6 (n : ℕ) (hpos : 0 < n) :
  (∃ (k : ℕ), k = 6 ∧ ∀ moves : ℕ, moves > 0 → (∃ counter_placements : set (ℕ × ℕ), (∀ i < k, (counter_placements ∘ (λ n, (n % k, n / k)) i)) ∧ has_strategy_to_prevent_A_win n moves)) :=
begin
  -- conditions for the game with counters on the grid and strategies for both players
  sorry
end

end A_cannot_win_at_least_k_6_l59_59337


namespace sarah_grocery_spending_l59_59203

noncomputable def total_grocery_cost (carrots milk pineapples flour cookies : ℕ) (discount_flour coupon_threshold coupon_discount : ℝ) : ℝ :=
  let carrot_price := 2.0 * carrots
  let milk_price := 3.0 * milk
  let pineapple_price := 4.0 * pineapples
  let flour_price := (6.0 * (1.0 - discount_flour)) * flour
  let cookie_price := cookies
  let total_cost := carrot_price + milk_price + pineapple_price + flour_price + cookie_price
  if total_cost ≥ coupon_threshold then total_cost - coupon_discount else total_cost

theorem sarah_grocery_spending : 
  total_grocery_cost 8 3 2 3 10 0.25 40 10 = 46.5 :=
by
  sorry

end sarah_grocery_spending_l59_59203


namespace problem_statement_l59_59593

noncomputable def f (x : ℝ) : ℝ := x / (1 - |x|)

def g (x : ℝ) := f x + x

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def range_is_real (f : ℝ → ℝ) : Prop := (Set.range f) = Set.univ
def one_to_one (f : ℝ → ℝ) : Prop := ∀ {x₁ x₂ : ℝ}, x₁ ≠ x₂ → f x₁ ≠ f x₂
def has_three_zeros (g : ℝ → ℝ) : Prop := ∃ (x₁ x₂ x₃ : ℝ), g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃

theorem problem_statement : 
  (is_odd f ∧ range_is_real f ∧ ¬one_to_one f ∧ has_three_zeros g) :=
by
  sorry

end problem_statement_l59_59593


namespace range_of_g_on_1_2_l59_59686

noncomputable def f (x : ℝ) : ℝ := real.log x / real.log (1/2)

noncomputable def g (x : ℝ) : ℝ := (1/2) ^ x

theorem range_of_g_on_1_2 : set.Icc (1/4 : ℝ) (1/2 : ℝ) = set.range (λ x, g x) ∩ set.Icc 1 2 := by
  sorry

end range_of_g_on_1_2_l59_59686


namespace tan_x_y_l59_59017

theorem tan_x_y (x y : ℝ) (h : Real.sin (2 * x + y) = 5 * Real.sin y) :
  Real.tan (x + y) = (3 / 2) * Real.tan x :=
sorry

end tan_x_y_l59_59017


namespace ordering_9_8_4_12_3_16_l59_59087

theorem ordering_9_8_4_12_3_16 : (4 ^ 12 < 9 ^ 8) ∧ (9 ^ 8 = 3 ^ 16) :=
by {
  sorry
}

end ordering_9_8_4_12_3_16_l59_59087


namespace expanded_product_term_count_l59_59756

theorem expanded_product_term_count : 
  let a := 6
  ∧ let b := 7
  ∧ let c := 5
  in (a * b * c) = 210 := 
by
  -- Define the given terms a, b, and c
  let a := 6
  let b := 7
  let c := 5

  -- Calculate the product
  have h := a * b * c
  
  -- Final step, compare the result with 210
  show h = 210,
  calc
    h = 6 * 7 * 5 : by rfl
    ... = 210 : by norm_num

end expanded_product_term_count_l59_59756


namespace total_coins_is_16_l59_59431

theorem total_coins_is_16 (x y : ℕ) (h₁ : x ≠ y) (h₂ : x^2 - y^2 = 16 * (x - y)) : x + y = 16 := 
sorry

end total_coins_is_16_l59_59431


namespace correct_conclusions_l59_59161

theorem correct_conclusions :
  (∀ (k : ℤ) (x : ℝ), (sin (k * π - x)) = -(sin (x - k * π))) ∧
  (∀ (x : ℝ), ¬ (tan (2 * x + π / 6) = 0) → ¬ (x = π / 12)) ∧
  (∀ (x : ℝ), -1 = cos (2 * (-2 * π / 3) + π / 3)) ∧
  (∀ (x : ℝ), tan (π - x) = 2 → cos x * cos x = 1 / 5) :=
by
  sorry

end correct_conclusions_l59_59161


namespace neither_sufficient_nor_necessary_l59_59607

variables {a b : Type} [linear_space a] [linear_space b]
variables {α : set a} [plane α]

-- Definitions from conditions
def line_in_plane (b : line) (α : plane) : Prop := b ⊆ α

-- Definitions of questions
def parallel_lines (a b : line) : Prop := -- definition for a parallel b
sorry

def parallel_line_plane (a : line) (α : plane) : Prop := -- definition for a parallel α
sorry

-- Translate proof problem
theorem neither_sufficient_nor_necessary
  (h1 : line_in_plane b α)
  (p : parallel_lines a b)
  (q : parallel_line_plane a α) : ¬((p → q) ∧ (q → p)) :=
sorry

end neither_sufficient_nor_necessary_l59_59607


namespace curvature_at_2_radius_of_curvature_at_2_center_of_curvature_at_2_l59_59526

noncomputable def y := λ x : ℝ, 4 / x
noncomputable def curvature (x : ℝ) :=
  abs (8 / x^3) / (1 + (-4 / x^2)^2)^(3/2)
noncomputable def radius_of_curvature (x : ℝ) :=
  ((1 + (-4 / x^2)^2)^(3/2)) / (8 / x^3)
noncomputable def center_of_curvature_x (x : ℝ) :=
  x - (-4 / x^2 * (1 + (-4 / x^2)^2)) / (8 / x^3)
noncomputable def center_of_curvature_y (x : ℝ) :=
  y x + (1 + (-4 / x^2)^2) / (8 / x^3)

theorem curvature_at_2 : curvature 2 = Real.sqrt 2 / 4 :=
by
  sorry

theorem radius_of_curvature_at_2 : radius_of_curvature 2 = 2 * Real.sqrt 2 :=
by
  sorry

theorem center_of_curvature_at_2 : center_of_curvature_x 2 = 4 ∧ center_of_curvature_y 2 = 4 :=
by
  sorry

end curvature_at_2_radius_of_curvature_at_2_center_of_curvature_at_2_l59_59526


namespace proof_solution_arithmetic_progression_l59_59646

noncomputable def system_has_solution (a b c m : ℝ) : Prop :=
  (m = 1 → a = b ∧ b = c) ∧
  (m = -2 → a + b + c = 0) ∧ 
  (m ≠ -2 ∧ m ≠ 1 → ∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c)

def abc_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem proof_solution_arithmetic_progression (a b c m : ℝ) : 
  system_has_solution a b c m → 
  (∃ x y z : ℝ, x + y + m * z = a ∧ x + m * y + z = b ∧ m * x + y + z = c ∧ 2 * y = x + z) ↔
  abc_arithmetic_progression a b c := 
by 
  sorry

end proof_solution_arithmetic_progression_l59_59646


namespace largest_n_for_quadratic_neg_l59_59536

theorem largest_n_for_quadratic_neg (n : ℤ) : n^2 - 11 * n + 24 < 0 → n ≤ 7 :=
begin
  sorry
end

end largest_n_for_quadratic_neg_l59_59536


namespace gcd_50420_35313_l59_59896

theorem gcd_50420_35313 : Int.gcd 50420 35313 = 19 := 
sorry

end gcd_50420_35313_l59_59896


namespace power_equivalence_l59_59423

theorem power_equivalence (L : ℕ) : 32^4 * 4^5 = 2^L → L = 30 :=
by
  sorry

end power_equivalence_l59_59423


namespace cats_not_eating_either_l59_59302

theorem cats_not_eating_either (total_cats : ℕ) (cats_liking_apples : ℕ) (cats_liking_fish : ℕ) (cats_liking_both : ℕ)
  (h1 : total_cats = 75) (h2 : cats_liking_apples = 15) (h3 : cats_liking_fish = 55) (h4 : cats_liking_both = 8) :
  ∃ cats_not_eating_either : ℕ, cats_not_eating_either = total_cats - (cats_liking_apples - cats_liking_both + cats_liking_fish - cats_liking_both + cats_liking_both) ∧ cats_not_eating_either = 13 :=
by
  sorry

end cats_not_eating_either_l59_59302


namespace problem_A_l59_59263

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - real.pi * x

variables (α β γ : ℝ)
variables (hα : 0 < α ∧ α < real.pi)
variables (hβ : 0 < β ∧ β < real.pi)
variables (hγ : 0 < γ ∧ γ < real.pi)
variables (sin_α : real.sin α = 1/3)
variables (tan_β : real.tan β = 5/4)
variables (cos_γ : real.cos γ = -1/3)

theorem problem_A : f α > f β ∧ f β > f γ :=
  sorry

end problem_A_l59_59263


namespace problem1_problem2_l59_59178

-- Problem 1: Calculation
theorem problem1 :
  sqrt 6 * (sqrt 2 / 2) - 6 * tan (real.pi / 6) + (- (1 / 2)) ^ (- 2 : ℤ) - abs (1 - sqrt 3) = 5 - 2 * sqrt 3 :=
by
  sorry

-- Problem 2: Simplification and Evaluation
theorem problem2 (x : ℝ) (hx : x = sqrt 2 - 3) :
  (x - 3) / (x - 2) / (x + 2 - 5 / (x - 2)) = sqrt 2 / 2 :=
by
  sorry

end problem1_problem2_l59_59178


namespace point_exists_l59_59676

theorem point_exists (P : Fin 1993 → ℤ × ℤ)
  (H0 : ∀ i : Fin 1993, (P i).1 ∈ Int ∧ (P i).2 ∈ Int)
  (H1 : ∀ i : Fin 1992, ∀ x y : ℚ, 
    (P i).1 ≤ x ∧ x ≤ (P (i+1)).1 ∧ 
    (P i).2 ≤ y ∧ y ≤ (P (i+1)).2 → 
    ((x ∉ Int) ∨ (y ∉ Int))): 
  ∃ i : Fin 1992, ∃ Q : ℚ × ℚ, 
  (Q.1 = (↑((P i).1) + ↑((P (i+1)).1)) / 2) ∧ 
  (Q.2 = (↑((P i).2) + ↑((P (i+1)).2)) / 2) ∧ 
  (Odd (2 * Q.1.num)) ∧ 
  (Odd (2 * Q.2.num)) :=
sorry

end point_exists_l59_59676


namespace market_price_level_6_l59_59501

variable (a b : ℝ)

-- Conditions
def condition1 := exp(4 * a) = 3
def condition2 := exp(3 * a + b) = 60

-- Statement
theorem market_price_level_6 {a b : ℝ} (h1 : exp(4 * a) = 3) (h2 : exp(3 * a + b) = 60) :
    abs (exp(6 * a + b) - 104) < 0.5 := 
sorry

end market_price_level_6_l59_59501


namespace find_coordinates_l59_59643

theorem find_coordinates 
  (A B C G H Q : Type) [has_vadd A (B × C)] [has_vadd A (H × Q)]
  (coords_G : G = (2/5) • A + (3/5) • C)
  (coords_H : H = (3/5) • A + (2/5) • B)
  (intersection_Q : Q = (u • A + v • B + w • C))
  (sum_coords : u + v + w = 1) :
  u = 5/13 ∧ v = 11/26 ∧ w = 3/13 := 
sorry

end find_coordinates_l59_59643


namespace num_possible_values_correct_l59_59108

noncomputable def num_possible_values : Nat :=
  let symbols := [Σ, #, △, ℤ]
  let U : Fin 6 → symbols := sorry  -- U sequence
  let J : Fin 6 → symbols := sorry  -- J sequence
  let cond1 : ∀ i : Fin 6, U i ∈ symbols := sorry
  let cond2 : ∀ (s : List (Fin 6)), (s = [0, 1, 3, 4] ∨ s = [0, 1, 2] ∨ s = [3, 4, 5]) → s.Nodup := sorry
  let cond3 : ∀ d : Fin 2, ∀ i j : Fin 3, U (i + 3 * d) = J (j + 3 * d) → i < j := sorry
  24

theorem num_possible_values_correct : num_possible_values = 24 := by
  sorry

end num_possible_values_correct_l59_59108


namespace solution_set_of_inequality_l59_59251

variables {ℝ : Type*} [linear_ordered_field ℝ]

-- Define f as a real-valued function on real numbers
variable {f : ℝ → ℝ}

-- Conditions: f is an odd function and decreasing function
def odd_function (f : ℝ → ℝ) := ∀ x, f(-x) = -f(x)
def decreasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f(x) > f(y)

-- Given/Assumptions
variables (h1 : odd_function f)
variables (h2 : decreasing_function f)

-- The goal
theorem solution_set_of_inequality (a : ℝ) : f(a^2) + f(2a) > 0 ↔ a ∈ Ioo (-2 : ℝ) 0 :=
sorry

end solution_set_of_inequality_l59_59251


namespace negative_integer_example_l59_59483

def is_negative_integer (n : ℤ) := n < 0

theorem negative_integer_example : is_negative_integer (-2) :=
by
  -- Proof will go here
  sorry

end negative_integer_example_l59_59483


namespace three_digit_numbers_with_sum_60_l59_59407

-- Definitions based on given conditions
def first_two_digits_sum {n : ℕ} (h : 100 ≤ n ∧ n < 500) : ℕ :=
  let A := n / 100 in
  let B := (n / 10) % 10 in
  A + B

-- Generate the main theorem statement
theorem three_digit_numbers_with_sum_60 :
  ∃ (P : ℕ → Prop), (∀ n, P n ↔ (100 ≤ n ∧ n < 500 ∧ first_two_digits_sum (by auto) n = 9)) ∧
                     (∃! n, P n) :=
begin
  sorry,
end

end three_digit_numbers_with_sum_60_l59_59407


namespace vertical_and_supplementary_sum_l59_59239

variable {α : Type} [Add α] [One α] [Zero α] [HasSub α] [PartialOrder α] [HasLe α]

theorem vertical_and_supplementary_sum 
  (angle1 angle2 angle3 : α) 
  (h1 : angle1 = angle2) 
  (h2 : angle2 + angle3 = 180) 
  : angle1 + angle3 = 180 := by
  sorry

end vertical_and_supplementary_sum_l59_59239


namespace volume_taken_up_by_cubes_l59_59827

noncomputable def volume_of_box := 4 * 7 * 8

noncomputable def volume_of_cube := 2 * 2 * 2

noncomputable def cubes_along_length := 4 / 2
noncomputable def cubes_along_width := 7 / 2
noncomputable def cubes_along_height := 8 / 2

noncomputable def total_cubes := nat.floor (cubes_along_length) * nat.floor (cubes_along_width) * nat.floor (cubes_along_height)

noncomputable def total_volume_of_cubes := total_cubes * volume_of_cube

noncomputable def percent_volume_taken := (total_volume_of_cubes.toFloat / volume_of_box.toFloat) * 100

theorem volume_taken_up_by_cubes : percent_volume_taken ≈ 85.714 := 
by
  sorry

end volume_taken_up_by_cubes_l59_59827


namespace triangle_area_correct_l59_59421

def triangle_area (base : ℝ) (height : ℝ) : ℝ :=
  (base * height) / 2

theorem triangle_area_correct :
  triangle_area 2 3 = 3 :=
by
  sorry

end triangle_area_correct_l59_59421


namespace gcd_459_357_l59_59418

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l59_59418


namespace investment_total_l59_59163

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem investment_total
  (P : ℝ := 8000)
  (r : ℝ := 0.08)
  (t : ℕ := 3)
  (A_rounded : ℕ := 10078)
  (A : ℝ := compoundInterest P r t) :
  A_rounded = Float.round A := by
  sorry

end investment_total_l59_59163


namespace find_linear_in_two_variables_l59_59100

def is_linear_in_two_variables (eq : String) : Bool :=
  eq = "x=y+1"

theorem find_linear_in_two_variables :
  (is_linear_in_two_variables "4xy=2" = false) ∧
  (is_linear_in_two_variables "1-x=7" = false) ∧
  (is_linear_in_two_variables "x^2+2y=-2" = false) ∧
  (is_linear_in_two_variables "x=y+1" = true) :=
by
  sorry

end find_linear_in_two_variables_l59_59100


namespace sigma_eq_iff_exists_borel_functions_l59_59331

variables {Ω : Type*} [measurable_space Ω]
variables {α : Type*} [measurable_space α] {β : Type*} [measurable_space β]
variables (ξ : Ω → α) (ζ : Ω → β)

/-- σ(ξ) = σ(ζ) if and only if there exist Borel functions φ and ψ such that ξ = φ(ζ) and ζ = ψ(ξ) -/
theorem sigma_eq_iff_exists_borel_functions :
  measurable_space.generate_from (ξ '' set.univ) = measurable_space.generate_from (ζ '' set.univ) ↔
  (∃ (φ : β → α) (ψ : α → β), measurable φ ∧ measurable ψ ∧ ∀ ω, ξ ω = φ (ζ ω) ∧ ζ ω = ψ (ξ ω)) :=
sorry

end sigma_eq_iff_exists_borel_functions_l59_59331


namespace pedestrians_speed_l59_59079

/-- Suppose two pedestrians start from points that are 28 km apart.
    The first pedestrian delays for 1 hour after walking 9 km and then increases their speed by 1 km/h.
    They meet 4 km from the first pedestrian's stopping point. 
    If there had been no delay, they would have met halfway (14 km from each starting point). 
    Prove the initial speed of both pedestrians is 3 km/h. -/
theorem pedestrians_speed (V : ℝ) (H : V > 0) :
  let first_pedestrian_time := 9 / V + 1 + 15 / (V + 1),
      second_pedestrian_time := 28 / V in
  second_pedestrian_time = first_pedestrian_time ↔ V = 3 := 
begin
  sorry
end

end pedestrians_speed_l59_59079


namespace harmonica_value_l59_59416

theorem harmonica_value (x : ℕ) (h1 : ∃ k : ℕ, ∃ r : ℕ, x = 12 * k + r ∧ r ≠ 0 
                                                   ∧ r ≠ 6 ∧ r ≠ 9 
                                                   ∧ r ≠ 10 ∧ r ≠ 11)
                         (h2 : ¬ (x * x % 12 = 0)) : 
                         4 = 4 :=
by 
  sorry

end harmonica_value_l59_59416


namespace evaluate_expression_equals_768_l59_59201

theorem evaluate_expression_equals_768 (x y : ℝ) (hx : x = 2 ^ 4) (hy : y = 2 ^ (-4)) : 
  (3 * x + 3 * x) / (y + y) = 768 :=
by
  sorry

end evaluate_expression_equals_768_l59_59201


namespace largest_n_for_quadratic_neg_l59_59534

theorem largest_n_for_quadratic_neg (n : ℤ) : n^2 - 11 * n + 24 < 0 → n ≤ 7 :=
begin
  sorry
end

end largest_n_for_quadratic_neg_l59_59534


namespace percentage_william_land_correct_l59_59303

-- Declare the necessary variables for clarity
variables (tax_william tax_thomas tax_mary tax_kimberly : ℝ)
variables (rate_william rate_thomas rate_mary rate_kimberly : ℝ)

-- Define the given conditions
def conditions : Prop :=
  tax_william = 480 ∧ rate_william = 0.05 ∧ 
  tax_thomas = 720 ∧ rate_thomas = 0.06 ∧ 
  tax_mary = 1440 ∧ rate_mary = 0.075 ∧ 
  tax_kimberly = 1200 ∧ rate_kimberly = 0.065

-- Define the taxable lands
def taxable_land (tax : ℝ) (rate : ℝ) : ℝ := tax / rate

def william_land : ℝ := taxable_land tax_william rate_william
def thomas_land : ℝ := taxable_land tax_thomas rate_thomas
def mary_land : ℝ := taxable_land tax_mary rate_mary
def kimberly_land : ℝ := taxable_land tax_kimberly rate_kimberly

-- Define the total taxable land
def total_taxable_land : ℝ := william_land + thomas_land + mary_land + kimberly_land

-- Define the percentage of Mr. William's land
def percentage_william_land : ℝ := (william_land / total_taxable_land) * 100

-- Prove the statement
theorem percentage_william_land_correct : 
  conditions →
  percentage_william_land = 16.19 :=
by
  intro h
  rw [conditions] at h
  cases h with hw h_rest
  cases h_rest with rt h_rest_2
  cases h_rest_2 with hr ht hm 
  cases hm with hk ha (tax_kimberly)

  sorry

end percentage_william_land_correct_l59_59303


namespace ratio_x_y_l59_59836

noncomputable def side_length_x (x : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧ 
    (12 - x) / x = 5 / 12 ∧
    12 * x = 5 * x + 60 ∧
    7 * x = 60

noncomputable def side_length_y (y : ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    a = 5 ∧ b = 12 ∧ c = 13 ∧
    y = 60 / 17

theorem ratio_x_y (x y : ℝ) (hx : side_length_x x) (hy : side_length_y y) : x / y = 17 / 7 :=
by
  sorry

end ratio_x_y_l59_59836


namespace relationship_y1_y2_y3_l59_59919

noncomputable def y1 : ℝ := 4 ^ 0.2
noncomputable def y2 : ℝ := (1 / 2) ^ -0.3
noncomputable def y3 : ℝ := Real.logBase (1 / 2) 8

theorem relationship_y1_y2_y3 : y1 > y2 ∧ y2 > y3 := by
  have y1_simplified : y1 = 2 ^ 0.4 := by
    calc
      y1 = 4 ^ 0.2 : by rfl
      ... = (2^2) ^ 0.2 : by rw [pow_two]
      ... = 2 ^ (2 * 0.2) : by rw [pow_mul]
      ... = 2 ^ 0.4 : by norm_num

  have y2_simplified : y2 = 2 ^ 0.3 := by
    calc
      y2 = (1 / 2) ^ -0.3 : by rfl
      ... = 2 ^ 0.3 : by rw [div_pow, one_pow, pow_neg]

  have y3_simplified : y3 = -3 := by
    calc
      y3 = Real.logBase (1 / 2) 8 : by rfl
      ... = Real.logBase (1 / 2) (2^3) : by norm_num
      ... = 3 * Real.logBase (1 / 2) 2 : by rw [log_base_pow]
      ... = -3 : by rw [Real.logBase_self, mul_neg]

  -- Note: Now we use Lean to show the inequalities
  have h1 : 2 ^ 0.4 > 2 ^ 0.3 := by
    apply Real.rpow_lt_rpow_of_exponent_lt
    norm_num
  
  have h2 : 2 ^ 0.3 > -3 := by
    linarith

  exact And.intro h1 h2

end relationship_y1_y2_y3_l59_59919


namespace inclusion_exclusion_example_correct_answer_l59_59493

def S := {n | 1 ≤ n ∧ n ≤ 8000}
def A := {n | n ∈ S ∧ 4 ∣ n}
def B := {n | n ∈ S ∧ 6 ∣ n}
def C := {n | n ∈ S ∧ 14 ∣ n}
def D := {n | n ∈ S ∧ 21 ∣ n}

theorem inclusion_exclusion_example :
  (A ∪ B) ∩ (C ∪ D)ᶜ = {n ∈ S | (4 ∣ n ∨ 6 ∣ n) ∧ ¬ (14 ∣ n ∨ 21 ∣ n)} :=
by sorry

theorem correct_answer : 
  ∃ n, n = (A ∪ B).card - ((A ∪ B) ∩ (C ∪ D)).card ∧ n = 2002 :=
by sorry

end inclusion_exclusion_example_correct_answer_l59_59493


namespace productive_employees_work_l59_59297

theorem productive_employees_work (total_work : ℝ) (P : ℝ) (Q : ℝ)
  (h1 : P = 0.2) (h2 : Q = 0.8) :
  (0.4 * total_work) = (0.85 * total_work) :=
by
  -- Given the conditions: 20% of employees do 80% of the work
  have h3 : 0.2 * total_work = 0.8 * total_work, from sorry
  -- 40% of the most productive employees perform 85% of the work
  exact sorry

end productive_employees_work_l59_59297


namespace sum_of_number_and_reverse_l59_59041

theorem sum_of_number_and_reverse :
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → ((10 * a + b) - (10 * b + a) = 7 * (a + b)) → a = 8 * b → 
  (10 * a + b) + (10 * b + a) = 99 :=
by
  intros a b conditions eq diff
  sorry

end sum_of_number_and_reverse_l59_59041


namespace square_area_calculation_l59_59629

noncomputable def square_area (a : ℝ) : ℝ := a*a

theorem square_area_calculation
  (A B C D P : (ℝ × ℝ))
  (dAP dBP dCP : ℝ)
  (h_square : let (ax, ay) := A, (bx, by) := B, (cx, cy) := C, (dx, dy) := D in 
                (bx - ax)^2 + (by - ay)^2 = (cx - bx)^2 + (cy - by)^2 ∧
                (cx - bx)^2 + (cy - by)^2 = (dx - cx)^2 + (dy - cy)^2 ∧
                (dx - cx)^2 + (dy - cy)^2 = (ax - dx)^2 + (ay - dy)^2 ∧
                (ax - dx)^2 + (ay - dy)^2 = (bx - ax)^2 + (by - ay)^2)
  (h_dAP : dist P A = dAP)
  (h_dBP : dist P B = dBP)
  (h_dCP : dist P C = dCP)
  (h_dAP_eq : dAP = 2)
  (h_dBP_eq : dBP = 3)
  (h_dCP_eq : dCP = 4)
  : ∃ (a : ℝ), square_area a = 10 + Real.sqrt 63 := by
  sorry

end square_area_calculation_l59_59629


namespace apples_in_box_ratio_mixed_fruits_to_total_l59_59764

variable (total_fruits : Nat) (oranges : Nat) (peaches : Nat) (apples : Nat) (mixed_fruits : Nat)
variable (one_fourth_of_box_contains_oranges : oranges = total_fruits / 4)
variable (half_as_many_peaches_as_oranges : peaches = oranges / 2)
variable (five_times_as_many_apples_as_peaches : apples = 5 * peaches)
variable (mixed_fruits_double_peaches : mixed_fruits = 2 * peaches)
variable (total_fruits_56 : total_fruits = 56)

theorem apples_in_box : apples = 35 := by
  sorry

theorem ratio_mixed_fruits_to_total : mixed_fruits / total_fruits = 1 / 4 := by
  sorry

end apples_in_box_ratio_mixed_fruits_to_total_l59_59764


namespace water_percentage_in_dried_grapes_l59_59226

noncomputable def fresh_grape_weight : ℝ := 40  -- weight of fresh grapes in kg
noncomputable def dried_grape_weight : ℝ := 5  -- weight of dried grapes in kg
noncomputable def water_percentage_fresh : ℝ := 0.90  -- percentage of water in fresh grapes

noncomputable def water_weight_fresh : ℝ := fresh_grape_weight * water_percentage_fresh
noncomputable def solid_weight_fresh : ℝ := fresh_grape_weight * (1 - water_percentage_fresh)
noncomputable def water_weight_dried : ℝ := dried_grape_weight - solid_weight_fresh
noncomputable def water_percentage_dried : ℝ := (water_weight_dried / dried_grape_weight) * 100

theorem water_percentage_in_dried_grapes : water_percentage_dried = 20 := by
  sorry

end water_percentage_in_dried_grapes_l59_59226


namespace range_of_a_l59_59067

theorem range_of_a (a : ℝ) : (3 + 5 > 1 - 2 * a) ∧ (3 + (1 - 2 * a) > 5) ∧ (5 + (1 - 2 * a) > 3) → -7 / 2 < a ∧ a < -1 / 2 :=
by
  sorry

end range_of_a_l59_59067


namespace probability_two_flies_swept_l59_59009

/-- Defining the positions of flies on the clock -/
inductive positions : Type
| twelve   | three   | six   | nine

/-- Probability that the minute hand sweeps exactly two specific positions after 20 minutes -/
theorem probability_two_flies_swept (flies : list positions) (time : ℕ) :
  (flies = [positions.twelve, positions.three, positions.six, positions.nine]) →
  (time = 20) →
  (probability_sweeps_two_flies flies time = 1 / 3) := sorry

end probability_two_flies_swept_l59_59009


namespace time_saved_by_increasing_speed_l59_59699

theorem time_saved_by_increasing_speed (d v1 v2 : ℕ) (h_v1 : v1 = 60) (h_v2 : v2 = 50) (h_d : d = 1200) : 
    d / v2 - d / v1 = 4 := 
by
  rw [h_v1, h_v2, h_d]
  have h1 : 1200 / 60 = 20 := by norm_num
  have h2 : 1200 / 50 = 24 := by norm_num
  rw [h1, h2]
  norm_num
  done

end time_saved_by_increasing_speed_l59_59699


namespace concyclic_B_C_N_L_l59_59340

noncomputable def incircle_touches_BC_at_D (A B C D : Point) : Prop := sorry
noncomputable def line_AD_intersects_k_at_L (A D L : Point) (k : Circle) : Prop := sorry
noncomputable def excentre_opposite_A (A B C K : Point) : Prop := sorry
noncomputable def midpoint (P Q R : Point) : Prop := sorry

theorem concyclic_B_C_N_L (A B C D L K M N : Point) (k : Circle)
  (h1 : incircle_touches_BC_at_D A B C D)
  (h2 : line_AD_intersects_k_at_L A D L k)
  (h3 : excentre_opposite_A A B C K)
  (h4 : midpoint B C M)
  (h5 : midpoint K M N) :
  concyclic B C N L :=
sorry

end concyclic_B_C_N_L_l59_59340


namespace count_integers_l59_59613

theorem count_integers (n : ℤ) (h : -11 ≤ n ∧ n ≤ 11) : ∃ (s : Finset ℤ), s.card = 7 ∧ ∀ x ∈ s, (x - 1) * (x + 3) * (x + 7) < 0 :=
by
  sorry

end count_integers_l59_59613


namespace probability_sum_odd_l59_59813

-- Conditions
def balls : Finset (Fin 15) := 
  { Fin.mk 0 (by norm_num) , Fin.mk 1 (by norm_num), Fin.mk 2 (by norm_num), Fin.mk 3 (by norm_num),
    Fin.mk 4 (by norm_num), Fin.mk 5 (by norm_num), Fin.mk 6 (by norm_num), Fin.mk 7 (by norm_num),
    Fin.mk 8 (by norm_num), Fin.mk 9 (by norm_num), Fin.mk 10 (by norm_num), Fin.mk 11 (by norm_num),
    Fin.mk 12 (by norm_num), Fin.mk 13 (by norm_num), Fin.mk 14 (by norm_num) }

def draw_count : Nat := 7

-- Proof statement
theorem probability_sum_odd : 
  (∃ favorable : ℚ, favorable = 3192 / 6435) →
  (∃ total : ℚ, total = 1) →
  ( (3192 / 6435 : ℚ) = (1064 / 2145 : ℚ) ) :=
by
  sorry

end probability_sum_odd_l59_59813


namespace largest_n_for_quadratic_neg_l59_59535

theorem largest_n_for_quadratic_neg (n : ℤ) : n^2 - 11 * n + 24 < 0 → n ≤ 7 :=
begin
  sorry
end

end largest_n_for_quadratic_neg_l59_59535


namespace find_third_number_in_ratio_l59_59617

theorem find_third_number_in_ratio (x : ℝ) (third_number : ℝ) (hx : x = 0.8571428571428571)
  (h : 0.75 / x = third_number / 8) : third_number ≈ 7 :=
by
  sorry

end find_third_number_in_ratio_l59_59617


namespace triangle_perimeter_ratio_l59_59152

theorem triangle_perimeter_ratio : 
  let side := 10
  let hypotenuse := Real.sqrt (side^2 + (side / 2) ^ 2)
  let triangle_perimeter := side + (side / 2) + hypotenuse
  let square_perimeter := 4 * side
  (triangle_perimeter / square_perimeter) = (15 + Real.sqrt 125) / 40 := 
by
  sorry

end triangle_perimeter_ratio_l59_59152


namespace integral_evaluation_limit_convergence_l59_59121

noncomputable def evaluate_integral : ℝ :=
  ∫ x in (1 : ℝ)..(3 * real.sqrt 3), (1 / real.cbrt (x^2) - 1 / (1 + real.cbrt (x^2)))

theorem integral_evaluation :
  evaluate_integral = (real.pi + 3 * real.sqrt 3 - 6) / 8 :=
sorry

noncomputable def integral_expression (t : ℝ) : ℝ :=
  ∫ x in (1 : ℝ)..t, 1 / (1 + real.cbrt (x^2))

theorem limit_convergence :
  ∃ a b : ℝ, a = 3 ∧ b = 1 / 3 ∧ ∀ t > 1,
  tendsto (λ t, (integral_expression t - a * t ^ b)) (𝓝 ⊤) (𝓝 0) :=
sorry

end integral_evaluation_limit_convergence_l59_59121


namespace largest_integer_n_neg_l59_59530

theorem largest_integer_n_neg (n : ℤ) : (n < 8 ∧ 3 < n) ∧ (n^2 - 11 * n + 24 < 0) → n ≤ 7 := by
  sorry

end largest_integer_n_neg_l59_59530


namespace part1_part2_l59_59264

noncomputable def f (x a b : ℝ) := |x - a| + |x + b|

theorem part1 (x : ℝ) : (f x 1 2 ≤ 5) → (-3 ≤ x ∧ x ≤ 2) :=
begin
  sorry
end

theorem part2 (a b : ℝ) (h₁ : f (a - a) a b = 3) (h₂ : a + b = 3) : (a > 0) → (b > 0) → 
  (∃ c : ℝ, (f c a b = 3) ∧ (a = b := 3 / 2) →
  (a^2 / b + b^2 / a = 3)) :=
begin
  sorry
end

end part1_part2_l59_59264


namespace exit_time_correct_l59_59035

def time_to_exit_wide : ℝ := 6
def time_to_exit_narrow : ℝ := 10

theorem exit_time_correct :
  ∃ x y : ℝ, x = 6 ∧ y = 10 ∧ 
  (1 / x + 1 / y = 4 / 15) ∧ 
  (y = x + 4) ∧ 
  (3.75 * (1 / x + 1 / y) = 1) :=
by
  use time_to_exit_wide
  use time_to_exit_narrow
  sorry

end exit_time_correct_l59_59035


namespace number_of_valid_subsets_l59_59568

-- Definitions based on given conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_subset (A : set ℕ) : Prop := 
  A ⊆ {2, 3, 7} ∧ (A ∩ {3, 7}).to_finset.card ≤ 1

-- The proof problem statement
theorem number_of_valid_subsets : 
  {A : set ℕ | valid_subset A}.to_finset.card = 6 :=
sorry

end number_of_valid_subsets_l59_59568


namespace common_perpendiculars_of_skew_lines_l59_59558

-- Definitions using the conditions from the problem
def is_skew (a b : Line) : Prop :=
  ¬ (∃ p, p ∈ a ∧ p ∈ b)

def is_perpendicular (a b : Line) : Prop :=
  ∀ (p : Point), p ∈ a → ∀ (q : Point), q ∈ b → orthogonal p q

-- Lean statement formalizing the problem
theorem common_perpendiculars_of_skew_lines
  (a b c a' b' c' : Line)
  (h1 : is_skew a b)
  (h2 : is_skew b c)
  (h3 : is_skew c a)
  (h4 : is_perpendicular a' b)
  (h5 : is_perpendicular a' c)
  (h6 : is_perpendicular b' c)
  (h7 : is_perpendicular b' a)
  (h8 : is_perpendicular c' a)
  (h9 : is_perpendicular c' b) :
  ∃ d : Line, is_perpendicular d a ∧ is_perpendicular d b ∧ is_perpendicular d c :=
sorry

end common_perpendiculars_of_skew_lines_l59_59558


namespace values_of_x_and_y_l59_59987

theorem values_of_x_and_y (x y : ℝ) (h1 : x - y > x + 1) (h2 : x + y < y - 2) : x < -2 ∧ y < -1 :=
by
  -- Proof goes here
  sorry

end values_of_x_and_y_l59_59987


namespace sum_c_d_eq_30_l59_59494

noncomputable def c_d_sum : ℕ :=
  let c : ℕ := 28
  let d : ℕ := 2
  c + d

theorem sum_c_d_eq_30 : c_d_sum = 30 :=
by {
  sorry
}

end sum_c_d_eq_30_l59_59494


namespace find_y_when_x_is_4_l59_59763

def inverse_proportional (x y : ℝ) : Prop :=
  ∃ C : ℝ, x * y = C

theorem find_y_when_x_is_4 :
  ∀ x y : ℝ,
  inverse_proportional x y →
  (x + y = 20) →
  (x - y = 4) →
  (∃ y, y = 24 ∧ x = 4) :=
by
  sorry

end find_y_when_x_is_4_l59_59763


namespace find_angle_bisector_length_l59_59633

-- Definition of the problem: An isosceles triangle with base AB and equal sides AC, BC.
variables (A B C D : Type) [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C] [linear_ordered_field D]
variables (AB AC BC : A) (BD CD : B)
variables (AD : C)

-- Assumptions given in the problem.
def isosceles_triangle (AB AC BC : A) : Prop :=
AB = 5 ∧ AC = 20 ∧ BC = 20

-- The length of the angle bisector from A to BC.
noncomputable def angle_bisector (AB AC BC AD : A) : Prop :=
  (isosceles_triangle AB AC BC) → (AD = 6)

-- Main theorem to prove.
theorem find_angle_bisector_length (AB AC BC : A) (AD : A) (h : isosceles_triangle AB AC BC) :
  angle_bisector AB AC BC AD :=
begin
  sorry
end

end find_angle_bisector_length_l59_59633


namespace ellipse_equation_l59_59386

theorem ellipse_equation (a b : ℝ) (h₁ : 2 * a * 2 = 4) 
  (h₂ : (b * b * 4) = 16)
  (h₃ : ∃ c : ℝ, c = sqrt 3 ∧ c * c = 3)
  (h₄ : a = 2 ∨ b = 2)
  (h₅ : e = sqrt 3 / 2)
  (h₆ : (a^2 - b^2 = 1 ∨ b^2 - a^2 = 1)) :
  x^2 + 4 * y^2 = 4 ∨ 4 * x^2 + y^2 = 16 := sorry

end ellipse_equation_l59_59386


namespace workshop_workers_transfer_l59_59480

theorem workshop_workers_transfer (w d t : ℕ) (h_w : 63 ≤ w) (h_d : d ≤ 31) 
(h_prod : 1994 = 31 * w + t * (t + 1) / 2) : 
(d = 28 ∧ t = 10) ∨ (d = 30 ∧ t = 21) := sorry

end workshop_workers_transfer_l59_59480


namespace sum_converges_to_3_l59_59187

-- Define the summand function
def summand (k : ℕ) : ℝ :=
  7^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

-- Define the infinite sum
def infinite_sum : ℝ :=
  ∑' k : ℕ, summand (k + 1)

-- Claim that the sum converges to 3
theorem sum_converges_to_3 : infinite_sum = 3 :=
  sorry

end sum_converges_to_3_l59_59187


namespace max_value_of_y_in_interval_l59_59920

theorem max_value_of_y_in_interval (x : ℝ) (h : 0 < x ∧ x < 1 / 3) : 
  ∃ y_max, ∀ x, 0 < x ∧ x < 1 / 3 → x * (1 - 3 * x) ≤ y_max ∧ y_max = 1 / 12 :=
by sorry

end max_value_of_y_in_interval_l59_59920


namespace factorial_division_l59_59183

theorem factorial_division :
  (7.factorial + 8.factorial) / 6.factorial = 63 := by
  sorry

end factorial_division_l59_59183


namespace C_completion_days_l59_59428

lemma work_rate_sum (A_work B_work C_work : ℝ) (hA : A_work = 1/24) (hB : B_work = 1/6)
  (hC_rate : A_work + B_work + C_work = 7/24) : C_work = 1/12 := 
by
  sorry

def days_to_complete_work (C_work : ℝ) : ℝ := 1 / C_work

theorem C_completion_days : days_to_complete_work (1/12) = 12 :=
by
  sorry

end C_completion_days_l59_59428


namespace math_dance_residents_l59_59345

theorem math_dance_residents (p a b : ℕ) (hp : Nat.Prime p) 
    (h1 : b ≥ 1) 
    (h2 : (a + b)^2 = (p + 1) * a + b) :
    b = 1 := by
  sorry

end math_dance_residents_l59_59345


namespace area_enclosed_by_region_l59_59194

theorem area_enclosed_by_region :
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 6*y - 3 = 0) → 
  (∃ r : ℝ, r = 4 ∧ area = (π * r^2)) :=
by
  -- Starting proof setup
  sorry

end area_enclosed_by_region_l59_59194


namespace shaded_area_correct_l59_59133

noncomputable def radius_large_circle : ℝ := 3
noncomputable def radius_small_circle : ℝ := 1.5
noncomputable def side_length_square : ℝ := sqrt 3

-- Definitions of points O, A, B, C, D, and E with given conditions
structure Point := (x : ℝ) (y : ℝ)

def O : Point := ⟨0, 0⟩
def A : Point := ⟨-(sqrt 3) / 2, (sqrt 3) / 2⟩
def B : Point := ⟨(sqrt 3) / 2, (sqrt 3) / 2⟩
def C : Point := ⟨(sqrt 3) / 2, -(sqrt 3) / 2⟩
def D : Point := -- to be defined along the line through AB
def E : Point := -- to be defined along the line through CB

-- Placeholder definition for small circle tangent point (tangent point supported by additional metric)
def F : Point := -- Tangent point on DE

-- Calculate the area of the sector, triangle, and the smaller circle
noncomputable def area_sector_DOE : ℝ :=
  (1 / 3) * π * (radius_large_circle ^ 2)

noncomputable def area_triangle_DOE : ℝ :=
  (1 / 2) * radius_large_circle * radius_large_circle * (sqrt 3 / 2) -- sin(120°) = sqrt(3)/2

noncomputable def area_smaller_circle : ℝ :=
  π * (radius_small_circle ^ 2)

noncomputable def shaded_area : ℝ :=
  area_sector_DOE - area_triangle_DOE - area_smaller_circle

-- The theorem to be proved
theorem shaded_area_correct :
  shaded_area = 0.75 * π - (9 * sqrt 3) / 4 :=
by
  sorry

end shaded_area_correct_l59_59133


namespace largest_integer_n_l59_59531

theorem largest_integer_n (n : ℤ) :
  (n^2 - 11 * n + 24 < 0) → n ≤ 7 :=
by
  sorry

end largest_integer_n_l59_59531


namespace inscribed_square_area_l59_59469

theorem inscribed_square_area :
    ∀ (x y : ℝ), (x^2 / 4 + y^2 / 8 = 1) →
    ∃ s : ℝ, s > 0 ∧ (2 * s)^2 = 32 / 3 :=
by
  -- problem conditions
  assume x y h,
  sorry

end inscribed_square_area_l59_59469


namespace rhombus_area_l59_59828

variable {R : Type} [LinearOrderedField R]

theorem rhombus_area (a : R) (θ : R) (ha : a = 3) (hθ : θ = π / 4) :
  (∃ b : R, b = a * a * sin θ / 2 ∧ b = 9 * (Real.sqrt 2) / 2) :=
by
  use a * a * Real.sin θ / 2
  split
  · rfl
  · sorry

end rhombus_area_l59_59828


namespace find_a1_for_geometric_sequence_l59_59222

noncomputable def geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : geometric_sequence) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

theorem find_a1_for_geometric_sequence (a : geometric_sequence)
  (h_geom : is_geometric_sequence a)
  (h1 : a 2 * a 5 = 2 * a 3)
  (h2 : (a 4 + a 6) / 2 = 5 / 4) :
  a 1 = 16 ∨ a 1 = -16 :=
sorry

end find_a1_for_geometric_sequence_l59_59222


namespace library_books_l59_59748

theorem library_books (x : ℕ) (h1 : 0.36 * x ∈ ℕ) (h2 : 0.75 * (0.36 * x) ∈ ℕ) (h3 : x = 0.36 * x + 0.27 * x + 185) : 
  x = 500 := 
by 
  sorry

end library_books_l59_59748


namespace minimize_sum_of_squares_l59_59511

noncomputable def sum_of_squares (x : ℝ) : ℝ := x^2 + (18 - x)^2

theorem minimize_sum_of_squares : ∃ x : ℝ, x = 9 ∧ (18 - x) = 9 ∧ ∀ y : ℝ, sum_of_squares y ≥ sum_of_squares 9 :=
by
  sorry

end minimize_sum_of_squares_l59_59511


namespace find_sequence_l59_59688

noncomputable def sequence_satisfies (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (1 / 2) * (a n + 1 / (a n))

theorem find_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h_pos : ∀ n, 0 < a n)
    (h_S : sequence_satisfies a S) :
    ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
sorry

end find_sequence_l59_59688


namespace average_rate_of_reduction_l59_59446

theorem average_rate_of_reduction
  (original_price final_price : ℝ)
  (h1 : original_price = 200)
  (h2 : final_price = 128)
  : ∃ (x : ℝ), 0 ≤ x ∧ x < 1 ∧ 200 * (1 - x) * (1 - x) = 128 :=
by
  sorry

end average_rate_of_reduction_l59_59446


namespace henry_initial_income_l59_59980

theorem henry_initial_income : 
  ∀ (new_income : ℝ) (percent_increase : ℝ), 
  new_income = 180 → 
  percent_increase = 0.50 → 
  ∃ (I : ℝ), I + percent_increase * I = new_income ∧ I = 120 :=
by {
  intros new_income percent_increase h_new_income h_percent_increase,
  use 120,
  split,
  { rw [←mul_add, add_comm, mul_assoc, Mul.mul_eq_one_iff.mp h_percent_increase, mul_one, h_new_income] },
  { rfl }
}

end henry_initial_income_l59_59980


namespace geom_seq_common_ratio_l59_59596

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geom_seq_common_ratio (h1 : a_n 0 + a_n 2 = 10)
                              (h2 : a_n 3 + a_n 5 = 5 / 4)
                              (h_geom : is_geom_seq a_n q) :
  q = 1 / 2 :=
by
  sorry

end geom_seq_common_ratio_l59_59596


namespace find_angle_D_l59_59255

noncomputable def calculate_angle (A B C D : ℝ) : ℝ :=
  if (A + B = 180) ∧ (C = D) ∧ (A = 2 * D - 10) then D else 0

theorem find_angle_D (A B C D : ℝ) (h1: A + B = 180) (h2: C = D) (h3: A = 2 * D - 10) : D = 70 :=
by
  sorry

end find_angle_D_l59_59255


namespace lucy_deposit_l59_59344

theorem lucy_deposit :
  ∃ D : ℝ, 
    let initial_balance := 65 
    let withdrawal := 4 
    let final_balance := 76 
    initial_balance + D - withdrawal = final_balance ∧ D = 15 :=
by
  -- sorry skips the proof
  sorry

end lucy_deposit_l59_59344


namespace final_price_of_jacket_l59_59031

noncomputable def originalPrice : ℝ := 250
noncomputable def firstDiscount : ℝ := 0.60
noncomputable def secondDiscount : ℝ := 0.25

theorem final_price_of_jacket :
  let P := originalPrice
  let D1 := firstDiscount
  let D2 := secondDiscount
  let priceAfterFirstDiscount := P * (1 - D1)
  let finalPrice := priceAfterFirstDiscount * (1 - D2)
  finalPrice = 75 :=
by
  sorry

end final_price_of_jacket_l59_59031


namespace log_diff_l59_59199

theorem log_diff (x y : ℝ) (hx : log 4 256 = x) (hx_def : 4 ^ x = 256)
                (hy : log 4 (1/16) = y) (hy_def : 4 ^ y = 1 / 16) :
  log 4 256 - log 4 (1/16) = 6 :=
by
  -- We are ignoring the proof steps and focusing on constructing a valid, buildable statement
  sorry

end log_diff_l59_59199


namespace find_point_in_second_quadrant_l59_59250

theorem find_point_in_second_quadrant :
  ∃ (x y : ℤ), (x < 0) ∧ (y > 0) ∧ (y ≤ x + 4) ∧ x = -1 ∧ y = 3 :=
by
  use -1, 3
  repeat { split };
  simp;
  sorry

end find_point_in_second_quadrant_l59_59250


namespace sixteen_right_angled_triangles_l59_59605

-- Define the conditions with necessary data structures and assumptions
structure Circle (α : Type) [MetricSpace α] :=
(center : α) (radius : ℝ) (radius_pos : 0 < radius)

variable {α : Type} [MetricSpace α]

-- Define two non-intersecting circles
def nonIntersectingCircles (c1 c2 : Circle α) : Prop :=
dist c1.center c2.center > c1.radius + c2.radius

-- Define a right-angled triangle with the specific properties
structure RightAngledTriangle (α : Type) :=
(hypotenuse : Line α)
(leg1 : Line α)
(leg2 : Line α)
(right_angle_vertex : α)

-- Define a common external tangent
def commonExternalTangent (c1 c2 : Circle α) (l : Line α) : Prop :=
tangentLineToCircle c1 l ∧ tangentLineToCircle c2 l ∧ l ∉ LineSegment c1.center c2.center

-- Define condition for right-angled triangle's legs being tangent to circles
def legsTangentToCircles (c1 c2 : Circle α) (t : RightAngledTriangle α) : Prop :=
tangentLineToCircle c1 t.leg1 ∧ tangentLineToCircle c2 t.leg2

-- Define the internal tangent passing through the right-angle vertex
def internalTangentThroughVertex (c1 c2 : Circle α) (t : RightAngledTriangle α) (l : Line α) : Prop :=
tangentLineToCircle c1 l ∧ tangentLineToCircle c2 l ∧ t.right_angle_vertex ∈ l

-- The main theorem statement
theorem sixteen_right_angled_triangles
  (c1 c2 : Circle α)
  (h_non_intersecting : nonIntersectingCircles c1 c2) :
  ∃ (triangles : Finset (RightAngledTriangle α)), triangles.card = 16 ∧ 
  (∀ t ∈ triangles, commonExternalTangent c1 c2 t.hypotenuse ∧ 
                    legsTangentToCircles c1 c2 t ∧ 
                    ∃ l, internalTangentThroughVertex c1 c2 t l) :=
sorry -- Proof to be provided.

end sixteen_right_angled_triangles_l59_59605


namespace tangent_line_at_neg1_3_l59_59744

noncomputable def f (x : ℝ) := log (x + 2) - 3 * x

theorem tangent_line_at_neg1_3 :
  let point := (-1, 3)
  ∃ (m b : ℝ), m = -2 ∧ b = 3 + 2 
  ∧ (∀ (x y : ℝ), y = m * x + b ↔ 2 * x + y - 1 = 0) ∧ 
  tangent_at point f (2 * x + y - 1 = 0) := 
by
  sorry

end tangent_line_at_neg1_3_l59_59744


namespace cereal_calories_l59_59318

-- Conditions
def pancakes_count := 6
def calories_per_pancake := 120
def bacon_count := 2
def calories_per_bacon := 100
def total_breakfast_calories := 1120

-- Statement
theorem cereal_calories : 
  let pancakes_calories := pancakes_count * calories_per_pancake,
      bacon_calories := bacon_count * calories_per_bacon,
      breakfast_calories := total_breakfast_calories in
  breakfast_calories - (pancakes_calories + bacon_calories) = 200 := 
by
  -- Proof will go here
  sorry

end cereal_calories_l59_59318


namespace five_inv_mod_33_l59_59884

theorem five_inv_mod_33 : ∃ x : ℤ, 0 ≤ x ∧ x < 33 ∧ 5 * x ≡ 1 [MOD 33] :=
by
  use 20
  split
  · norm_num
  split
  · norm_num
  · norm_num
  · rfl
sorry

end five_inv_mod_33_l59_59884


namespace number_of_odd_digits_in_base_4_l59_59538

theorem number_of_odd_digits_in_base_4 (n : ℕ) (h : n = 233) : 
  ∃ m : ℕ, (m = 2) ∧ (count_odd_digits (to_base_4 n) = m) :=
by 
  sorry

-- Auxiliary definitions
def to_base_4 (n : ℕ) : list ℕ :=
  sorry

def count_odd_digits (digits : list ℕ) : ℕ :=
  sorry

end number_of_odd_digits_in_base_4_l59_59538


namespace ratio_equality_l59_59459

variables {A B C M A1 B1 : Type} [IsoscelesTriangle A B C] [PointOnLine M A B]
          [LineThrough M A1] [LineThrough M B1] [LineIntersect A1 AC] [LineIntersect B1 BC]

theorem ratio_equality :
  M ∈ Segment A B →
  A1 ∈ Line AC →
  B1 ∈ Line BC →
  ∃ (AA1 A1M BB1 B1M : ℝ),
  (AA1 / A1M = BB1 / B1M) :=
by
  sorry

end ratio_equality_l59_59459


namespace fraction_of_white_roses_l59_59492

open Nat

def rows : ℕ := 10
def roses_per_row : ℕ := 20
def total_roses : ℕ := rows * roses_per_row
def red_roses : ℕ := total_roses / 2
def pink_roses : ℕ := 40
def white_roses : ℕ := total_roses - red_roses - pink_roses
def remaining_roses : ℕ := white_roses + pink_roses
def fraction_white_roses : ℚ := white_roses / remaining_roses

theorem fraction_of_white_roses :
  fraction_white_roses = 3 / 5 :=
by
  sorry

end fraction_of_white_roses_l59_59492


namespace relationship_between_a_b_c_l59_59585

noncomputable def a : ℝ := Real.pi^(1/3)
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi  -- log base π
noncomputable def c : ℝ := Real.ln (Real.sqrt 3 - 1)

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by
  -- Proof omitted, only statement provided.
  sorry

end relationship_between_a_b_c_l59_59585


namespace skew_lines_common_perpendiculars_l59_59557

variables (a b c a' b' c' : Line)
variables [SkewLines a b c]
variables [Perpendicular a' b] [Perpendicular a' c]
variables [Perpendicular b' c] [Perpendicular b' a]
variables [Perpendicular c' a] [Perpendicular c' b]

theorem skew_lines_common_perpendiculars :
  (Perpendicular a b' ∧ Perpendicular a c') ∧
  (Perpendicular b c' ∧ Perpendicular b a') ∧
  (Perpendicular c a' ∧ Perpendicular c b') :=
sorry

end skew_lines_common_perpendiculars_l59_59557


namespace volume_of_cylinder_with_sq_area_121_l59_59097

noncomputable def volume_of_cylinder (h : ℝ) : ℝ :=
  484 * h / Real.pi

theorem volume_of_cylinder_with_sq_area_121 (h : ℝ) :
  ∃ (perimeter : ℝ) (side_length : ℝ) (radius : ℝ), 
  let square_area := 121,
      side_length := Real.sqrt square_area,
      perimeter := 4 * side_length,
      radius := perimeter / (2 * Real.pi) in
  (volume_of_cylinder h = 484 * h / Real.pi) :=
by
  sorry

end volume_of_cylinder_with_sq_area_121_l59_59097


namespace each_member_score_l59_59478

def total_members : ℝ := 5.0
def members_didnt_show_up : ℝ := 2.0
def total_points_by_showed_up_members : ℝ := 6.0

theorem each_member_score
  (h1 : total_members - members_didnt_show_up = 3.0)
  (h2 : total_points_by_showed_up_members = 6.0) :
  total_points_by_showed_up_members / (total_members - members_didnt_show_up) = 2.0 :=
sorry

end each_member_score_l59_59478


namespace second_half_time_is_16_l59_59829

-- Definitions for the problem conditions
def total_distance : ℝ := 40
def half_distance : ℝ := total_distance / 2
def initial_speed (v : ℝ) : ℝ := v
def injured_speed (v : ℝ) : ℝ := v / 2
def time_first_half (v : ℝ) : ℝ := half_distance / initial_speed v
def time_second_half (v : ℝ) : ℝ := half_distance / injured_speed v
def time_relation (v : ℝ) : Prop := time_second_half v = time_first_half v + 8

-- Lean theorem statement
theorem second_half_time_is_16 (v : ℝ) (h : time_relation v) : time_second_half v = 16 := by
  sorry

end second_half_time_is_16_l59_59829


namespace sealed_envelope_problem_l59_59148

theorem sealed_envelope_problem :
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) →
  ((n = 12 ∧ (n % 10 ≠ 2) ∧ n ≠ 35 ∧ (n % 10 ≠ 5)) ∨
   (n ≠ 12 ∧ (n % 10 ≠ 2) ∧ n = 35 ∧ (n % 10 = 5))) →
  ¬(n % 10 ≠ 5) :=
by
  sorry

end sealed_envelope_problem_l59_59148


namespace degree_of_polynomial_l59_59088

-- Define the inner polynomial
def inner_poly : Polynomial ℝ := 2 * X^3 - 5 * X + 7

-- Define the polynomial raised to the power 8
def poly_exp := inner_poly ^ 8

-- Statement of the problem: Prove that the degree of poly_exp = 24
theorem degree_of_polynomial :
  Polynomial.degree poly_exp = 24 :=
by 
  sorry

end degree_of_polynomial_l59_59088


namespace integer_mod_105_l59_59729

theorem integer_mod_105 (x : ℤ) :
  (4 + x ≡ 2 * 2 [ZMOD 3^3]) →
  (6 + x ≡ 3 * 3 [ZMOD 5^3]) →
  (8 + x ≡ 5 * 5 [ZMOD 7^3]) →
  x % 105 = 3 :=
by
  sorry

end integer_mod_105_l59_59729


namespace total_practice_hours_l59_59364

def weekly_practice_hours : ℕ := 4
def weeks_in_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours : (weekly_practice_hours * weeks_in_month) * months = 80 := by
  -- Calculation for weekly practice in hours
  let monthly_hours := weekly_practice_hours * weeks_in_month
  -- Calculation for total practice in hours
  have total_hours : ℕ := monthly_hours * months
  have calculation : total_hours = 80 := 
    by simp [weekly_practice_hours, weeks_in_month, months, monthly_hours, total_hours]
  exact calculation

end total_practice_hours_l59_59364


namespace water_level_decrease_3m_l59_59999

-- Definitions from conditions
def increase (amount : ℝ) : ℝ := amount
def decrease (amount : ℝ) : ℝ := -amount

-- The claim to be proven
theorem water_level_decrease_3m : decrease 3 = -3 :=
by
  sorry

end water_level_decrease_3m_l59_59999


namespace part1_part2_l59_59268

variables (k m : ℝ)

-- Conditions
def line_eqn : ℝ → ℝ := λ x, k * (x + 1)
def ellipse_eqn : ℝ × ℝ → Prop := λ p, p.1 ^ 2 + 4 * p.2 ^ 2 = m ^ 2

-- Statement 1
theorem part1 (h1: k ≠ 0) (h2: m > 0) (A B : ℝ × ℝ) (h3: ellipse_eqn A) (h4: ellipse_eqn B) (h5: A ≠ B) : 
  m^2 > (4 * k ^ 2) / (1 + 4 * k ^ 2) :=
sorry

-- Statement 2
theorem part2 (h1: k ≠ 0) (h2: m > 0) (A B C : ℝ × ℝ) (h3: A ≠ B) (h4: B ≠ C) (h5: C.2 = 0)
  (h6: line_eqn C.1 = C.2) (h7: ellipse_eqn A) (h8: ellipse_eqn B) (h9: A ≠ B)
  (h10: vector.eq_scalar_mul_of_eq AC CB 3 : vector A - vector C = 3 * (vector C - vector B)) : 
  ellipse_eqn = (λ p, p.1 ^ 2 + 4 * p.2 ^ 2 = 4) :=
sorry

end part1_part2_l59_59268


namespace parabola_focus_coordinates_l59_59739

theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  ∃ x y : ℝ, y = 4 * a * x^2 → (x, y) = (0, 1 / (16 * a)) :=
by
  sorry

end parabola_focus_coordinates_l59_59739


namespace sum_absolute_difference_inequality_l59_59441

theorem sum_absolute_difference_inequality
  (n : ℕ) (h1 : 2 ≤ n)
  (a b : ℕ → ℤ)
  (ha : ∀ i j, i ≠ j → a i ≠ a j)
  (hb : ∀ i j, i ≠ j → b i ≠ b j) :
  (∑ i in Finset.range n, ∑ j in Finset.range n, |a i - b j|) 
  - (∑ i in Finset.range n, ∑ j in Finset.Ico i n, (|a j - a i| + |b j - b i|)) 
  ≥ n :=
sorry

end sum_absolute_difference_inequality_l59_59441


namespace factorial_division_l59_59184

theorem factorial_division :
  (7.factorial + 8.factorial) / 6.factorial = 63 := by
  sorry

end factorial_division_l59_59184


namespace find_all_n_l59_59208

open Nat

def euler_totient (k : ℕ) : ℕ := φ k  -- Euler's Totient function

theorem find_all_n (n : ℕ) (h_pos : 0 < n) :
  (∑ k in range n, euler_totient (k + 1)) = (3 * n^2 + 5) / 8 -> n = 1 ∨ n = 3 ∨ n = 5 :=
  sorry

end find_all_n_l59_59208


namespace time_for_train_to_pass_platform_is_190_seconds_l59_59127

def trainLength : ℕ := 1200
def timeToCrossTree : ℕ := 120
def platformLength : ℕ := 700
def speed (distance time : ℕ) := distance / time
def distanceToCrossPlatform (trainLength platformLength : ℕ) := trainLength + platformLength
def timeToCrossPlatform (distance speed : ℕ) := distance / speed

theorem time_for_train_to_pass_platform_is_190_seconds
  (trainLength timeToCrossTree platformLength : ℕ) (h1 : trainLength = 1200) (h2 : timeToCrossTree = 120) (h3 : platformLength = 700) :
  timeToCrossPlatform (distanceToCrossPlatform trainLength platformLength) (speed trainLength timeToCrossTree) = 190 := by
  sorry

end time_for_train_to_pass_platform_is_190_seconds_l59_59127


namespace dot_product_is_5_over_2_l59_59977

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define the condition of parallelism between the vectors
def parallel_condition (m : ℝ) : Prop :=
  let v1 := (2 * m - 1, 4)
  let v2 := (-2 - m, 3)
  (2 * m - 1) * 3 = 4 * (-2 - m)

-- Prove that the dot product of a and b is 5/2 given the conditions
theorem dot_product_is_5_over_2 (m : ℝ) (h : parallel_condition m) : ((a.1 * b m.1) + (a.2 * b m.2)) = 5 / 2 :=
sorry

end dot_product_is_5_over_2_l59_59977


namespace smallest_coefficients_in_binomial_expansion_l59_59272

def binomial_term (a b : ℝ) (n r : ℕ) : ℝ :=
  (-1)^r * (Nat.choose n r) * (a^(n-r)) * (b^r)

theorem smallest_coefficients_in_binomial_expansion :
  let a b : ℝ := 1
  let n : ℕ := 8
  let T : ℕ → ℝ := binomial_term a b n
  (∀ r, T r ≥ 0) → (min (T 3) (T 5) = T 3 ∧ min (T 3) (T 5) = T 5) :=
by
  sorry

end smallest_coefficients_in_binomial_expansion_l59_59272


namespace smallest_difference_l59_59433

def is_3_digit (x : ℕ) : Prop := 100 ≤ x ∧ x < 1000

def digits_of_number_are_valid (x : ℕ) : Prop :=
  let digits := (x / 100, (x / 10) % 10, x % 10)
  digits.1 ∈ {2, 3, 4, 6, 7, 8} ∧ digits.2 ∈ {2, 3, 4, 6, 7, 8} ∧ digits.3 ∈ {2, 3, 4, 6, 7, 8}

theorem smallest_difference (n m : ℕ) 
  (n_3_digit : is_3_digit n) 
  (m_3_digit : is_3_digit m)
  (n_digits_valid : digits_of_number_are_valid n)
  (m_digits_valid : digits_of_number_are_valid m)
  (distinct_digits : {2, 3, 4, 6, 7, 8} ⊆ (↑$ List.toFinset [((n / 100) : ℤ), (n % 100 / 10 : ℤ), (n % 10 : ℤ), (m / 100 : ℤ), (m % 100 / 10 : ℤ), (m % 10 : ℤ)])) :
  n ≠ m → abs (n - m) = 59 :=
by sorry

end smallest_difference_l59_59433


namespace problem_solution_l59_59579

theorem problem_solution (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 972) : (x + 2) * (x - 2) = 5 :=
by
  sorry

end problem_solution_l59_59579


namespace length_of_each_train_l59_59417

theorem length_of_each_train
  (speed_km_hr : ℕ) 
  (time_cross_sec : ℕ)
  (relative_speed_factor : ℕ)
  (L : ℕ) 
  (h1 : speed_km_hr = 54)
  (h2 : time_cross_sec = 8)
  (h3 : relative_speed_factor = 2 * 15) **  -- relative speed is 2 trains, each at 15 m/s
  (h4 : 2 * L = relative_speed_factor * time_cross_sec) : 
  L = 120 := 
by
  sorry

end length_of_each_train_l59_59417


namespace chord_length_cannot_be_8_l59_59580

theorem chord_length_cannot_be_8 (r : ℝ) (h : r = 3) (MN : ℝ) 
  (hMN : MN > 0 ∧ MN ≤ 2 * r) : MN ≠ 8 :=
by
  unfold has_lt.lt at hMN
  rw h at hMN
  have d : 2 * 3 = 6 := rfl
  rw d at hMN
  -- Here, MN ≤ 6 must be true, so MN cannot be 8
  have hMN_le : MN ≤ 6 := hMN.right
  sorry

end chord_length_cannot_be_8_l59_59580


namespace problem_statement_l59_59240

variable {α : Type*}
variable [ordered_ring α]

def arithmetic_sequence (a : ℕ → α) (d : α) := ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
(1 + n) * (a 0 + a n) / 2

theorem problem_statement (a : ℕ → α) (d : α) (S : ℕ → α) (h_seq : arithmetic_sequence a d)
(h_d_neg : d < 0) (h_S2_S7 : S 2 = S 7) :
a 4 ≠ 0 := sorry

end problem_statement_l59_59240


namespace market_price_level_6_l59_59500

variable (a b : ℝ)

-- Conditions
def condition1 := exp(4 * a) = 3
def condition2 := exp(3 * a + b) = 60

-- Statement
theorem market_price_level_6 {a b : ℝ} (h1 : exp(4 * a) = 3) (h2 : exp(3 * a + b) = 60) :
    abs (exp(6 * a + b) - 104) < 0.5 := 
sorry

end market_price_level_6_l59_59500


namespace eccentricity_of_hyperbola_is_sqrt_2_l59_59967

noncomputable def eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) : ℝ :=
  let c := real.sqrt (a^2 + b^2)
  c / a

theorem eccentricity_of_hyperbola_is_sqrt_2 {a b : ℝ} (h : a > 0 ∧ b > 0)
  (tangent_line : (x y : ℝ) → x^2 + y^2 = a^2) -- represents tangent line FM
  (focus_F : (c : ℝ) × 0 = (real.sqrt (a^2 + b^2)) × 0) -- F is at (c,0)
  (P : ℝ × ℝ → P.1 = 0) -- P is on the y-axis
  (M_midpoint : (x y : ℝ) → x = focus_F.1 / 2 ∧ y = focus_F.1 / 2)
  : eccentricity a b h = real.sqrt 2 :=
by
  sorry

end eccentricity_of_hyperbola_is_sqrt_2_l59_59967


namespace total_distance_covered_l59_59791

variable (h : ℝ) (initial_height : ℝ := h) (bounce_ratio : ℝ := 0.8)

theorem total_distance_covered :
  initial_height + 2 * initial_height * bounce_ratio / (1 - bounce_ratio) = 13 * h :=
by 
  -- Proof omitted for now
  sorry

end total_distance_covered_l59_59791


namespace cos_A_3_over_4_min_area_l59_59931

variable (A : Real) (m : Real) (a b c : Real) (S : Real) (ABC : Triangle)
  (hABC : (a + c = 2 * b))
  (hC : (ABC.C = 2 * A))
  (ha : (a = (4 * m^2 + 4 * m + 9) / (m + 1)))
  (hm_pos : m > 0)
  (hcosA : ABC.sides opposite angles A ABC.C ABC.B ABC.sides in arithmetic sequence)

theorem cos_A_3_over_4 (hABC : (a + c = 2 * b)) (hC: (ABC.C = 2 * A)) (ha: (a = (4 * m^2 + 4 * m + 9) / (m + 1))) (hm_pos: m > 0):
  cos A = 3 / 4 :=
by
  sorry

theorem min_area (hABC : (a + c = 2 * b)) (hC: (ABC.C = 2 * A)) (ha: (a = (4 * m^2 + 4 * m + 9) / (m + 1))) (hm_pos: m > 0):
  ∃ S, S = 15 * sqrt 7 :=
by
  sorry

end cos_A_3_over_4_min_area_l59_59931


namespace tomatoes_picked_yesterday_l59_59139

-- Definitions corresponding to the conditions in the problem.
def initial_tomatoes : Nat := 160
def tomatoes_left_after_yesterday : Nat := 104

-- Statement of the problem proving the number of tomatoes picked yesterday.
theorem tomatoes_picked_yesterday : initial_tomatoes - tomatoes_left_after_yesterday = 56 :=
by
  sorry

end tomatoes_picked_yesterday_l59_59139


namespace part1_part2_l59_59600

-- Statement for part (1)
theorem part1 (m : ℝ) : 
  (∀ x1 x2 : ℝ, (m - 1) * x1^2 + 3 * x1 - 2 = 0 ∧ 
               (m - 1) * x2^2 + 3 * x2 - 2 = 0 ∧ x1 ≠ x2) ↔ m > -1/8 :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 2 = 0 ∧ ∀ y : ℝ, (m - 1) * y^2 + 3 * y - 2 = 0 → y = x) ↔ 
  (m = 1 ∨ m = -1/8) :=
sorry

end part1_part2_l59_59600


namespace market_price_level6_l59_59498

noncomputable def market_price (x : ℕ) (a b : ℝ) : ℝ := real.exp (a * x + b)

theorem market_price_level6 :
  ∀ (a b : ℝ),
  (real.exp (4 * a) = 3) →
  (market_price 3 a b = 60) →
  market_price 6 a b ≈ 170 :=
by
  intros a b h1 h2,
  have h3 : real.exp a ≈ 1.41 := sorry,
  have h4 : market_price 6 a b = 120 * real.sqrt 2 := sorry,
  show 120 * real.sqrt 2 ≈ 170, from sorry

end market_price_level6_l59_59498


namespace locus_of_Y_l59_59237

variable (A B C D X Y : Point)
variable (AD BC l : Line)
variable [IsTrapezoid ABCD AD BC]

/-- The Point Y created as described lies on the line l' which is
    perpendicular to the bases AD and BC, dividing AD in the same
    ratio as l divides BC. -/
theorem locus_of_Y (h₁ : IsPerpendicular AD l) (h₂ : IsPerpendicular BC l)
  (h₃ : MovesAlongLine X l) (h₄ : Intersection Y (PerpendicularTo BX A) (PerpendicularTo CX D)) :
  ∃ l', IsLine l' ∧ IsPerpendicular AD l' ∧ IsPerpendicular BC l' ∧ DividesInSameRatio l' AD l BC := sorry

end locus_of_Y_l59_59237


namespace rational_numbers_count_l59_59539

theorem rational_numbers_count : 
  let S := { r : ℚ | 0 < r ∧ r < 1 ∧ ∃ n m: ℕ, r = n / m ∧ Nat.gcd n m = 1 ∧ n + m = 1000 }
  in card S = 200 :=
by sorry

end rational_numbers_count_l59_59539


namespace flux_calc_l59_59495

noncomputable def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (y, z, x)

theorem flux_calc (R : ℝ) :
  let S := { p : ℝ × ℝ × ℝ | p.2^2 + p.1^2 <= R^2 ∧ 0 <= p.3 ∧ p.3 <= p.1 } in
  let flux := ∫ p in S, (vector_field p.1 p.2 p.3).1 * p.2 + (vector_field p.1 p.2 p.3).2 * p.3 + (vector_field p.1 p.2 p.3).3 * p.1 d(p.1) in
  flux = 0 :=
by
  sorry

end flux_calc_l59_59495


namespace sum_of_positive_k_l59_59045

theorem sum_of_positive_k (k : ℤ) :
  (∃ x y : ℤ, x * y = -20 ∧ k = x + y) →
  (∑ k in {k | ∃ x y : ℤ, x * y = -20 ∧ k = x + y ∧ k > 0}.toFinset) = 56 :=
by
  sorry

end sum_of_positive_k_l59_59045


namespace concurrency_of_lines_l59_59675

variable {A B C A_1 B_1 C_1 I O : Type}

-- Definitions for points in triangle and the incenter and circumcenter
variable [incircle_point_tangent_BC : A_1]
variable [incircle_point_tangent_CA : B_1]
variable [incircle_point_tangent_AB : C_1]

-- Definitions for the incircle and circumcircle
noncomputable def incenter (A B C : Type) : Type := sorry
noncomputable def circumcenter (A B C : Type) : Type := sorry

-- Statements that these points are reflections over angle bisectors
axiom reflection_A1 : A_1 = sorry
axiom reflection_B1 : B_1 = sorry
axiom reflection_C1 : C_1 = sorry

-- Define the collinear lines going through these points
axiom line_AA1 : ∃ P, collinear ({A, A_1, P} : set Type)
axiom line_BB1 : ∃ P, collinear ({B, B_1, P} : set Type)
axiom line_CC1 : ∃ P, collinear ({C, C_1, P} : set Type)
axiom line_IO : ∃ P, collinear ({I, O, P} : set Type)

theorem concurrency_of_lines
  (A B C A_1 B_1 C_1 I O : Type)
  [incircle_point_tangent_BC : A_1]
  [incircle_point_tangent_CA : B_1]
  [incircle_point_tangent_AB : C_1]
  [incenter A B C : I]
  [circumcenter A B C : O]
  [reflection_A1 : A_1 = sorry]
  [reflection_B1 : B_1 = sorry]
  [reflection_C1 : C_1 = sorry]
  [line_AA1 : ∃ P, collinear ({A, A_1, P} : set Type)]
  [line_BB1 : ∃ P, collinear ({B, B_1, P} : set Type)]
  [line_CC1 : ∃ P, collinear ({C, C_1, P} : set Type)]
  [line_IO : ∃ P, collinear ({I, O, P} : set Type)] :
  ∃ P, collinear ({A, A_1, P} : set Type) ∧ collinear ({B, B_1, P} : set Type) ∧ collinear ({C, C_1, P} : set Type) ∧ collinear ({I, O, P} : set Type) :=
sorry

end concurrency_of_lines_l59_59675


namespace part1_part2_l59_59565

-- Given conditions:
-- M: Number of colors (where M ≥ 3 and M ∈ ℤ⁺)
variable (M : ℕ) (M_ge_3 : M ≥ 3)

-- Question 1: Prove that the largest positive integer N 
-- such that there exists a wonderful coloring of a regular N-gon satisfies N ≤ (M - 1)².
theorem part1 (N : ℕ) (wonderfully_colored : ∃ (N' : ℕ), N' = N ∧ is_wonderful_coloring N' M) :
  N ≤ (M - 1) * (M - 1) := sorry

-- Question 2: If M-1 is a prime number, prove there exists a wonderful regular (M-1)²-gon.
theorem part2 (p : ℕ) (h_prime : nat.prime p) (h : M = p + 1) :
  ∃ N, N = p * p ∧ is_wonderful_coloring N M := sorry

-- Definition of a wonderful coloring according to the problem statement.
def is_wonderful_coloring (N M : ℕ) : Prop := sorry
-- Note: The actual definition should capture the concept that no triangle in the
-- coloring has exactly two sides of the same color as described in the problem.

end part1_part2_l59_59565


namespace Sn_sum_of_consecutive_squares_l59_59716

noncomputable def S (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), nat.choose (2 * n + 1) (2 * k) * 2^(2 * n - 2 * k) * 3^k

theorem Sn_sum_of_consecutive_squares (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℕ, S n = a^2 + b^2 ∧ b = a + 1 :=
sorry

end Sn_sum_of_consecutive_squares_l59_59716


namespace probability_circle_or_square_l59_59709

theorem probability_circle_or_square (total_figures : ℕ)
    (num_circles : ℕ) (num_squares : ℕ) (num_triangles : ℕ)
    (total_figures_eq : total_figures = 10)
    (num_circles_eq : num_circles = 3)
    (num_squares_eq : num_squares = 4)
    (num_triangles_eq : num_triangles = 3) :
    (num_circles + num_squares) / total_figures = 7 / 10 :=
by sorry

end probability_circle_or_square_l59_59709


namespace circle_radius_and_circumference_l59_59085

theorem circle_radius_and_circumference (r C : ℝ) (A : ℝ := 81 * real.pi) :
  (A = real.pi * r ^ 2) → (C = 2 * real.pi * r) → 
  r = 9 ∧ C = 18 * real.pi :=
by
  intro h_area h_circumference
  sorry

end circle_radius_and_circumference_l59_59085


namespace factorial_division_l59_59186

theorem factorial_division (n : ℕ) (h : 6 ≤ n):
  ((nat.factorial 7 + nat.factorial 8) / nat.factorial 6) = 63 :=
sorry

end factorial_division_l59_59186


namespace complex_real_imag_sum_eq_zero_l59_59289

theorem complex_real_imag_sum_eq_zero (b : ℝ) : (realPart : ℂ → ℝ) (2 - b * complex.I) + (imaginaryPart : ℂ → ℝ) (2 - b * complex.I) = 0 → b = 2 :=
by
  sorry

end complex_real_imag_sum_eq_zero_l59_59289


namespace largest_integer_n_l59_59532

theorem largest_integer_n (n : ℤ) :
  (n^2 - 11 * n + 24 < 0) → n ≤ 7 :=
by
  sorry

end largest_integer_n_l59_59532


namespace problem_statement_l59_59101

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c 

def f_A (x : ℝ) : ℝ := -2 / x^2 - 3 * x
def f_B (x : ℝ) : ℝ := -(x - 1)^2 + x^2
def f_C (x : ℝ) : ℝ := 11 * x^2 + 29 * x

theorem problem_statement : is_quadratic f_C ∧ ¬ is_quadratic f_A ∧ ¬ is_quadratic f_B := by
  sorry

end problem_statement_l59_59101


namespace origin_on_circle_l59_59286

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem origin_on_circle (r : ℝ) (P : ℝ × ℝ) (h1 : r = 13) (h2 : P = (5, 12)) :
  distance 0 0 P.1 P.2 = r :=
by
  have h3 : distance 0 0 5 12 = sqrt ((5 - 0) ^ 2 + (12 - 0) ^ 2),
  {
    sorry
  },
  rw [h1, h2] at h3,
  have h4 : sqrt ((5 - 0) ^ 2 + (12 - 0) ^ 2) = sqrt (25 + 144),
  {
    sorry
  },
  rw h4 at h3,
  have h5 : sqrt (25 + 144) = sqrt 169,
  {
    sorry
  },
  rw h5 at h3,
  have h6 : sqrt 169 = 13,
  {
    sorry
  },
  rw h6 at h3,
  exact h3

end origin_on_circle_l59_59286


namespace total_cakes_served_l59_59627

def Cakes_Monday_Lunch : ℕ := 5
def Cakes_Monday_Dinner : ℕ := 6
def Cakes_Sunday : ℕ := 3
def cakes_served_twice (n : ℕ) : ℕ := 2 * n
def cakes_thrown_away : ℕ := 4

theorem total_cakes_served : 
  Cakes_Sunday + Cakes_Monday_Lunch + Cakes_Monday_Dinner + 
  (cakes_served_twice (Cakes_Monday_Lunch + Cakes_Monday_Dinner) - cakes_thrown_away) = 32 := 
by 
  sorry

end total_cakes_served_l59_59627


namespace pure_gala_trees_60_l59_59451

noncomputable def num_pure_gala_trees (T F G : ℕ) : Prop := 
  (T > 0) ∧
  (F = (3 / 4) * T) ∧
  (0.10 * T + F = 204) ∧
  (G = T - F) ∧
  (G = 60)

theorem pure_gala_trees_60 (T F G : ℕ) (hT : num_pure_gala_trees T F G) : G = 60 :=
sorry

end pure_gala_trees_60_l59_59451


namespace car_speed_is_90_mph_l59_59131

-- Define the given conditions
def distance_yards : ℚ := 22
def time_seconds : ℚ := 0.5
def yards_per_mile : ℚ := 1760

-- Define the car's speed in miles per hour
noncomputable def car_speed_mph : ℚ := (distance_yards / yards_per_mile) * (3600 / time_seconds)

-- The theorem to be proven
theorem car_speed_is_90_mph : car_speed_mph = 90 := by
  sorry

end car_speed_is_90_mph_l59_59131


namespace arithmetic_sequence_sum_l59_59241

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)
variable (d : α)

-- Condition definitions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = d

def sum_condition (a : ℕ → α) : Prop :=
  a 2 + a 5 + a 8 = 39

-- The goal statement to prove
theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_sum : sum_condition a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 117 :=
  sorry

end arithmetic_sequence_sum_l59_59241


namespace five_digit_number_from_grid_l59_59205

-- Define the grid of 6x4 elements with constraints
def grid_condition (grid : list (list ℕ)) : Prop :=
  (∀ row, row ∈ grid → list.nodup row) ∧  -- no duplicates in any row
  (∀ col, col < 4 → list.nodup ((grid.map (λ row, row.nth col)).filter (λ x, x.is_some)).map option.get) ∧  -- no duplicates in any column
  (grid.length = 3) ∧ (∀ row, row.length = 4) ∧ -- grid is 3x4
  (grid.nth 1).nth 1 = some 5 ∧  -- Second row, second column is 5
  (grid.nth 2).nth 3 = some 6 -- Third row, fourth column is 6

-- Example grid that satisfies the above condition
def example_grid := [
  [4, 6, 1, 2],
  [1, 5, 2, 3],
  [3, 2, 5, 6]
]

-- Prove that the five-digit number formed by the first five numbers in the last row, from left to right, is 46123
theorem five_digit_number_from_grid : grid_condition example_grid → 
  list.take 5 [4, 6, 1, 2] = [4, 6, 1, 2] := 
by
  intros h_grid
  simp [grid_condition, example_grid] -- Ensure the condition holds
  have row_len : example_grid.head = [4, 6, 1, 2] by simp [example_grid]
  show list.take 5 [4, 6, 1, 2] = [4, 6, 1, 2]
  by simp [row_len]

end five_digit_number_from_grid_l59_59205


namespace ava_average_speed_l59_59876

noncomputable def initial_odometer : ℕ := 14941
noncomputable def final_odometer : ℕ := 15051
noncomputable def elapsed_time : ℝ := 4 -- hours

theorem ava_average_speed :
  (final_odometer - initial_odometer) / elapsed_time = 27.5 :=
by
  sorry

end ava_average_speed_l59_59876


namespace there_exists_diagonally_black_cells_l59_59632

def knight_path_condition (n : ℕ) (L R : ℕ × ℕ) (black_cells : set (ℕ × ℕ)) : Prop :=
  (L = (1, 1)) ∧ (R = (n, n)) ∧ 
  ((L ∉ black_cells) ∧ (R ∉ black_cells)) ∧
  ∀ p : list (ℕ × ℕ), -- assumed path of knight
    (p.head = some L) ∧ (p.last = some R) →
    (∀ i < p.length - 1, 
      let (x, y) := p.nth_le i sorry in
      let (x', y') := p.nth_le (i+1) sorry in
      ((x', y') ∈ {(x+1, y+2), (x+1, y-2), (x-1, y+2), (x-1, y-2), (x+2, y+1), (x+2, y-1), (x-2, y+1), (x-2, y-1)})) →
    ∃ i, (p.nth_le i sorry) ∈ black_cells

theorem there_exists_diagonally_black_cells (n : ℕ) (k : ℕ) (black_cells : set (ℕ × ℕ)) :
  knight_path_condition n (1, 1) (n, n) black_cells →
  n = 3 * k + 1 →
  ∃ a b c : ℕ × ℕ, ({a, b, c} ⊆ diagonally_connected_cells n) ∧ (at_least_two_black {a, b, c} black_cells) :=
sorry

end there_exists_diagonally_black_cells_l59_59632


namespace average_sale_correct_l59_59819

-- Noncomputable theory because we are dealing with real numbers.
noncomputable theory

-- Define constants for the sales figures.
def sale_month_1 : ℝ := 6535
def sale_month_2 : ℝ := 6927
def sale_month_3 : ℝ := 6855
def sale_month_4 : ℝ := 7230
def sale_month_5 : ℝ := 6562
def required_sale_month_6 : ℝ := 4891

-- Define the total sales of the first 5 months.
def total_sales_first_5_months : ℝ :=
  sale_month_1 + sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5

-- Define the total sales of all 6 months.
def total_sales_all_6_months : ℝ :=
  total_sales_first_5_months + required_sale_month_6

-- Define the expected average sale.
def expected_average_sale : ℝ := 6500

-- Define the number of months.
def number_of_months : ℝ := 6

-- Define the average sale calculation.
def average_sale : ℝ :=
  total_sales_all_6_months / number_of_months

-- The proof statement: average_sale == expected_average_sale.
theorem average_sale_correct :
  average_sale = expected_average_sale :=
by
  sorry

end average_sale_correct_l59_59819


namespace num_valid_grids_l59_59277

def is_valid_grid (g : array (Fin 2 × Fin 2) (Fin 2)) : Prop :=
  (∀ i : Fin 2, ∃ j : Fin 2, g (i, j) = 0 ∧ g (i, 1 - j.val) = 1) ∧
  (∀ j : Fin 2, ∃ i : Fin 2, g (i, j) = 0 ∧ g (1 - i.val, j) = 1)

theorem num_valid_grids : 
  ∃ n : ℕ, n = 2 ∧ (∀ g : array (Fin 2 × Fin 2) (Fin 2), is_valid_grid g → true) := sorry

end num_valid_grids_l59_59277


namespace solve_system_l59_59028

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1, 2, 1],
  ![3, -5, 3],
  ![2, 7, -1]
]

noncomputable def b : Fin 3 → ℝ := ![4, 1, 8]

theorem solve_system :
  LinearAlgebra.det A ≠ 0 →
  LinearAlgebra.cramer A b = ![1, 1, 1] :=
by
  intros hA
  sorry

end solve_system_l59_59028


namespace cookies_donated_l59_59811

theorem cookies_donated (n d : ℕ) (h1 : n = 120) (h2 : d = 7) : (n % d) = 1 := by
  rw [h1, h2]
  exact Nat.mod_eq_of_lt (Nat.div_lt_self (by decide) (by decide)) (by decide)

end cookies_donated_l59_59811


namespace increasing_function_on_interval_l59_59424

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f x < f y

def func_A (x : ℝ) := log 2 (1 - x)
def func_B (x : ℝ) := 1 - x^2
def func_C (x : ℝ) := 2^x
def func_D (x : ℝ) := -(x + 1)^2

theorem increasing_function_on_interval : 
  is_increasing_on_interval func_C (-infty) 1 :=
sorry

end increasing_function_on_interval_l59_59424


namespace distance_product_l59_59247

noncomputable theory

-- Define the parametric equations of the curve C
def curveC (θ : ℝ) : ℝ × ℝ :=
(1 + 4 * Real.cos θ, 2 + 4 * Real.sin θ)

-- Define the parametric equations of the line l
def lineL (t : ℝ) : ℝ × ℝ :=
(2 + (Real.sqrt 3 / 2) * t, 1 + (1/2) * t)

-- Definition of the point P
def P : ℝ × ℝ := (2, 1)

-- Now state the theorem
theorem distance_product : ∀ t1 t2 : ℝ,
  (t1^2 + (Real.sqrt 3 - 1) * t1 - 14 = 0) →
  (t2^2 + (Real.sqrt 3 - 1) * t2 - 14 = 0) →
  |t1 * t2| = 14 :=
by
  intros t1 t2 h1 h2
  sorry

end distance_product_l59_59247


namespace find_tan_half_angle_l59_59943

variable {α : Real} (h₁ : Real.sin α = -24 / 25) (h₂ : α ∈ Set.Ioo (π:ℝ) (3 * π / 2))

theorem find_tan_half_angle : Real.tan (α / 2) = -4 / 3 :=
sorry

end find_tan_half_angle_l59_59943


namespace number_of_correct_statements_l59_59484

open Real

-- Setting the conditions as definitions or lemmas
lemma condition1 (x : ℝ) : x ∈ ℚ ∨ x ∈ (ℝ \ ℚ) :=
begin
  sorry
end

lemma condition2 (a : ℝ) : ¬ (a < a + a ↔ a ≤ 0) :=
begin
  sorry
end

lemma condition3 : sqrt 121 = 11 ∨ sqrt 121 = -11 :=
begin
  sorry
end

lemma condition4 (x : ℝ) : ¬ (x ≥ 0 → x > 0) :=
begin
  sorry
end

lemma condition5 (x y : ℝ) : ¬ (irrational x → irrational y → irrational (x + y)) :=
begin
  sorry
end

-- The main statement to prove
theorem number_of_correct_statements : 2 :=
by {
  let correct_statements := [
    condition1,
    not (condition2 0),
    condition3,
    not (condition4 0),
    not (condition5 1 2)
  ],
  sorry
}

end number_of_correct_statements_l59_59484


namespace sum_of_angles_l59_59979

theorem sum_of_angles (α β : ℝ) (h1 : 0 < α ∧ α < π / 2)
                      (h2 : 0 < β ∧ β < π / 2)
                      (h3 : Real.cot α = 3 / 4)
                      (h4 : Real.cot β = 1 / 7) :
                      α + β = 3 * π / 4 :=
by
  sorry

end sum_of_angles_l59_59979


namespace consecutive_integers_inequality_l59_59802

theorem consecutive_integers_inequality (n : ℕ) (t : ℕ) (h : t ≥ 0) :
  let a := λ i, t + i
  (∑ i in finset.range n, a i) * (∑ i in finset.range n, (1 : ℚ) / a i) < (n * (n + 1) * Real.log (Real.exp 1 * n) / 2) := by
  sorry

end consecutive_integers_inequality_l59_59802


namespace find_pq_sum_l59_59665

def is_two_one_binary (n : ℕ) : Prop :=
  ∃ j k : ℕ, (0 ≤ j) ∧ (j < k) ∧ (k ≤ 49) ∧ (n = 2^j + 2^k)

def count_binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def T : finset ℕ :=
  (finset.range (2^50)).filter is_two_one_binary

noncomputable def probability_div_by_3 : ℚ :=
  let count_T := count_binom 50 2
  let count_div_3 := 600 -- calculated from condition
  count_div_3 / count_T

theorem find_pq_sum : (p q : ℕ) (hpq : p / q = probability_div_by_3) (coprime p q) : p + q = 21 :=
by
  sorry

end find_pq_sum_l59_59665


namespace sum_ak_div_k2_ge_sum_inv_k_l59_59329

open BigOperators

theorem sum_ak_div_k2_ge_sum_inv_k
  (n : ℕ)
  (a : Fin n → ℕ)
  (hpos : ∀ k, 0 < a k)
  (hdist : Function.Injective a) :
  ∑ k : Fin n, (a k : ℝ) / (k + 1 : ℝ)^2 ≥ ∑ k : Fin n, 1 / (k + 1 : ℝ) := sorry

end sum_ak_div_k2_ge_sum_inv_k_l59_59329


namespace range_of_m_l59_59993

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 2^|x| + m = 0) → m ≤ -1 :=
by
  sorry

end range_of_m_l59_59993


namespace tiffany_cans_at_end_of_week_l59_59072

theorem tiffany_cans_at_end_of_week:
  (4 + 2.5 - 1.25 + 0 + 3.75 - 1.5 + 0 = 7.5) :=
by
  sorry

end tiffany_cans_at_end_of_week_l59_59072


namespace tangent_parallel_max_min_diff_eq_4_l59_59258

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 - 3 * x + b

theorem tangent_parallel_max_min_diff_eq_4 {a b : ℝ} 
  (h : deriv (f a b) (-1) = 0) : 
  ∃ max min : ℝ, (max - min = 4) :=
by 
  let g : ℝ → ℝ := λ x, x^3 + a * x^2 - 3 * x + b
  have g_deriv : ∀ x : ℝ, deriv g x = 3 * x^2 + 2 * a * x - 3, 
    {
      intro x,
      simp [g],
      simp [deriv],
      sorry
    },
  have a_zero : a = 0, 
    {
      sorry
    },
  have f_reduction : ∀ x, f 0 b x = x^3 - 3 * x + b,
    {
      intro x,
      simp [f],
     [rwa [a_zero],
      sorry
    },
  have critical_points : ∀ (x : ℝ), deriv (f 0 b) x = 0 → (x = 1 ∨ x = -1),
    {
      sorry
    },
  have values_at_critical : ∃ (b : ℝ), f 0 b 1 - f 0 b (-1) = 4,
    {
      sorry
    },
  exact values_at_critical

end tangent_parallel_max_min_diff_eq_4_l59_59258


namespace no_x_satisfies_f_f_x_eq_5_l59_59392

noncomputable def f : ℝ → ℝ := sorry

theorem no_x_satisfies_f_f_x_eq_5 : ¬ ∃ x : ℝ, f (f x) = 5 :=
by {
  have h1 : ∀ x, f x = 5 → x = 3, from sorry,
  have h2 : f 3 = 5, from sorry,
  have h3 : f (-3) = 3 ∧ f (1) = 3 ∧ f (5) = 3, from sorry,
  intro hx,
  cases hx with x hx,
  have hc : f x = 3, from sorry,
  have hx3x : x = -3 ∨ x = 1 ∨ x = 5, from sorry,
  cases hx3x;
  { rw hx3x at hc,
    exact hx.not (h1 _ hc) },
}

end no_x_satisfies_f_f_x_eq_5_l59_59392


namespace line_transformation_l59_59668

theorem line_transformation (a b : ℝ)
  (h1 : ∀ x y : ℝ, a * x + y - 7 = 0)
  (A : Matrix (Fin 2) (Fin 2) ℝ) (hA : A = ![![3, 0], ![-1, b]])
  (h2 : ∀ x' y' : ℝ, 9 * x' + y' - 91 = 0) :
  (a = 2) ∧ (b = 13) :=
by
  sorry

end line_transformation_l59_59668


namespace distance_from_point_to_y_axis_l59_59742

theorem distance_from_point_to_y_axis (A : ℝ × ℝ) (hA : A = (2, -3)) : abs (A.1) = 2 :=
by {
  have h : A.1 = 2, by rw [hA],
  rw [h, abs_of_nonneg],
  norm_num 
} sorry

end distance_from_point_to_y_axis_l59_59742


namespace perimeter_of_rectangular_field_l59_59746

theorem perimeter_of_rectangular_field (width length : ℝ) (h1 : length = (7/5) * width) (h2 : width = 80) :
    let perimeter := 2 * (length + width)
    perimeter = 384 :=
by
    have h3 : length = 112 := by
        calc
        length = (7/5) * width : h1
              ... = (7/5) * 80   : by rw [h2]
              ... = 112          : by norm_num
    let perimeter := 2 * (length + width)
    have h4 : perimeter = 2 * (112 + 80) := by rw [h3, h2]
    have h5 : perimeter = 2 * 192 := by rw [h4]
    have h6 : perimeter = 384 := by norm_num
    exact h6

end perimeter_of_rectangular_field_l59_59746


namespace total_telephone_bill_second_month_l59_59877

theorem total_telephone_bill_second_month
  (F C1 : ℝ) 
  (h1 : F + C1 = 46)
  (h2 : F + 2 * C1 = 76) :
  F + 2 * C1 = 76 :=
by
  sorry

end total_telephone_bill_second_month_l59_59877


namespace unstuck_rectangle_min_perimeter_l59_59145

open Real

/--
A rectangle that is inscribed in a larger rectangle (with one vertex on each side) is called unstuck if it is possible to rotate (however slightly) the smaller rectangle about its center within the confines of the larger. 
Of all the rectangles that can be inscribed unstuck in a 6 by 8 rectangle, the smallest perimeter has the form sqrt(N), for a positive integer N.
Prove that the smallest such N is 448.
-/
theorem unstuck_rectangle_min_perimeter :
  ∃ N : ℕ, (∃ P : ℝ, P = sqrt (N : ℝ)) ∧ (N = 448) :=
sorry

end unstuck_rectangle_min_perimeter_l59_59145


namespace log_bounds_l59_59189

theorem log_bounds (c d : ℤ) 
                   (h1 : 2048 < 3456) 
                   (h2 : 3456 < 4096)
                   (hlog1 : log 2 2048 = 11)
                   (hlog2 : log 2 4096 = 12) : 
                   (c = 11 ∧ d = 12 ∧ c + d = 23) :=
by 
  sorry

end log_bounds_l59_59189


namespace prod_sum_reciprocal_bounds_l59_59718

-- Define the product of the sum of three positive numbers and the sum of their reciprocals.
theorem prod_sum_reciprocal_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  9 ≤ (a + b + c) * (1 / a + 1 / b + 1 / c) :=
by
  sorry

end prod_sum_reciprocal_bounds_l59_59718


namespace intersection_point_of_lines_l59_59038

theorem intersection_point_of_lines :
  ∃ (x y : ℝ), x + 2 * y - 4 = 0 ∧ 2 * x - y + 2 = 0 ∧ (x, y) = (0, 2) :=
by
  sorry

end intersection_point_of_lines_l59_59038


namespace triangle_angle_area_l59_59946

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x
variables {A B C : ℝ}
variables {BC : ℝ}
variables {S : ℝ}

theorem triangle_angle_area (hABC : A + B + C = π) (hBC : BC = 2) (h_fA : f A = 0) 
  (hA : A = π / 3) : S = Real.sqrt 3 :=
by
  -- Sorry, proof skipped
  sorry

end triangle_angle_area_l59_59946


namespace problem_statement_l59_59858

def square_area_diagonal (d : ℝ) : ℝ :=
  let s := d / Real.sqrt 2
  s^2

def rectangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  2 * (Real.abs (x1 - x2) + Real.abs (y2 - y1))

def triangle_area_with_line (a b : ℝ) : ℝ :=
  let b := b
  let h := b
  0.5 * b * h

theorem problem_statement :
  let d := 2 * Real.sqrt 2
  let A := square_area_diagonal d
  let B := rectangle_area 2 2  4 1  4 3
  let C := triangle_area_with_line 4 2 in
  (A < B ∨ B < C) ∧ (A = C ∨ C < B) :=
by
  let d := 2 * Real.sqrt 2
  let A := square_area_diagonal d
  let B := rectangle_area 2 2 4 1 4 3
  let C := triangle_area_with_line 4 2
  sorry

end problem_statement_l59_59858


namespace sum_165_terms_is_neg3064_l59_59998

-- Definitions of conditions
def sum_first_n_terms_of_AP (n : ℕ) (a d : ℝ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

variables (a d : ℝ)

-- Given conditions
axiom sum_15_terms : sum_first_n_terms_of_AP 15 a d = 200
axiom sum_150_terms : sum_first_n_terms_of_AP 150 a d = 150

-- Theorem to prove
theorem sum_165_terms_is_neg3064 :
  sum_first_n_terms_of_AP 165 a d = -3064 :=
by {
  sorry
}

end sum_165_terms_is_neg3064_l59_59998


namespace parrot_consumption_l59_59710

theorem parrot_consumption :
  ∀ (parakeet_daily : ℕ) (finch_daily : ℕ) (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ) (weekly_birdseed : ℕ),
    parakeet_daily = 2 →
    finch_daily = parakeet_daily / 2 →
    num_parakeets = 3 →
    num_parrots = 2 →
    num_finches = 4 →
    weekly_birdseed = 266 →
    14 = (weekly_birdseed - ((num_parakeets * parakeet_daily + num_finches * finch_daily) * 7)) / num_parrots / 7 :=
by
  intros parakeet_daily finch_daily num_parakeets num_parrots num_finches weekly_birdseed
  intros hp1 hp2 hp3 hp4 hp5 hp6
  sorry

end parrot_consumption_l59_59710


namespace triangle_properties_l59_59624

theorem triangle_properties (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ b = 5 ∧ c = 4 * Real.sqrt 2 ∧ a^2 + b^2 = c^2 := by
{
  sorry
}

end triangle_properties_l59_59624


namespace range_of_a_product_of_extreme_points_l59_59260

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 - a * x * (Real.log x) + a * x + 2

def extreme_points (x1 x2 : ℝ) (a : ℝ) : Prop :=
  f'.x x1 = 0 ∧ f'.x x2 = 0 ∧ x1 < x2 ∧ x1 ≠ x2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h_extreme : extreme_points x1 x2 a) :
  a ∈ set.Ioo Real.exp 0 := sorry

theorem product_of_extreme_points (a : ℝ) (x1 x2 : ℝ) (h_extreme : extreme_points x1 x2 a) :
  x1 * x2 < a * a := sorry

end range_of_a_product_of_extreme_points_l59_59260


namespace probability_divisible_by_3_variant_l59_59410

theorem probability_divisible_by_3_variant : 
  let S := {n ∈ Finset.range 12 | n + 1} in
  let subset := S.filter (fun n => (n+1) % 3 = 0) in
  let favorable_outcomes : ℕ :=
  let comb_count := 
    (Finset.card (subset.combinations 3)).card + 
    (Finset.card (subset.combinations 2)).card * 
    (Finset.card ((S \ subset).combinations 1)).card in
  let total_outcomes := Finset.card (S.combinations 3) in
  comb_count = 52 / 220 :=
  sorry

end probability_divisible_by_3_variant_l59_59410


namespace overall_gain_percent_l59_59071

theorem overall_gain_percent (cp1 cp2 cp3: ℝ) (sp1 sp2 sp3: ℝ) (h1: cp1 = 840) (h2: cp2 = 1350) (h3: cp3 = 2250) (h4: sp1 = 1220) (h5: sp2 = 1550) (h6: sp3 = 2150) : 
  (sp1 + sp2 + sp3 - (cp1 + cp2 + cp3)) / (cp1 + cp2 + cp3) * 100 = 10.81 := 
by 
  sorry

end overall_gain_percent_l59_59071


namespace win_conditions_l59_59080

-- define the game setup and the sets W, L, T
structure Game :=
  (n0 : ℕ)
  (A_wins B_wins : ℕ → Prop)
  (valid_moves_for_A : ℕ → ℕ → Prop)
  (valid_moves_for_B : ℕ → ℕ → Prop)

def W : set ℕ := {n | ∀ g : Game, g.A_wins n}
def L : set ℕ := {n | ∀ g : Game, g.B_wins n}
def T : set ℕ := {n | ∀ g : Game, ¬ g.A_wins n ∧ ¬ g.B_wins n}

-- define valid moves for both players
def valid_moves_for_A (n_current n_next : ℕ) : Prop :=
  n_current ≤ n_next ∧ n_next ≤ n_current^2

def valid_moves_for_B (n_current n_next : ℕ) (p : ℕ) (r : ℕ) : Prop :=
  (p.prime ∧ r ∈ ℕ) ∧ (n_current = n_next * p^r)

-- define the winning conditions
def A_wins (n : ℕ) : Prop := n = 1990
def B_wins (n : ℕ) : Prop := n = 1

-- prove the required conditions about W, L, and T
theorem win_conditions (n0 : ℕ) :
  (n0 ∈ L ↔ n0 ∈ {2, 3, 4, 5}) ∧
  (n0 ∈ T ↔ n0 ∈ {6, 7}) ∧
  (n0 ∈ W ↔ n0 ≥ 8) := sorry

end win_conditions_l59_59080


namespace inversion_properties_l59_59674

open EuclideanGeometry

variables {A B C H H_A H_B H_C : Point}

-- Conditions
axiom triangle_ABC : is_triangle A B C
axiom altitudes_feet : feet_of_altitudes A B C H_A H_B H_C
axiom orthocenter : orthocenter A B C H
axiom cyclic_quadrilateral_A : cyclic_quad C H_B H_C B
axiom cyclic_quadrilateral_B : cyclic_quad C H_B H H_A

-- Theorem we want to prove
theorem inversion_properties :
  (inversion_center_A : by inversion_center A B H_C)
  (inversion_center_H : by inversion_center H A H_A) : 
  inversion_center_A  ↔ B ↔ H_C ∧ 
  inversion_center_H ↔ A ↔ H_A :=
sorry

end inversion_properties_l59_59674


namespace perpendicular_vector_a_value_l59_59227

theorem perpendicular_vector_a_value :
  ∀ (a : ℝ), let m := (3, a - 1)
  let n := (a, -2) in
  m.fst * n.fst + m.snd * n.snd = 0 → a = -2 :=
by intros a m n h
   sorry

end perpendicular_vector_a_value_l59_59227


namespace twenty_th_four_digit_number_l59_59753

theorem twenty_th_four_digit_number : 
  let digits := [5, 6, 7, 8]
  let permutations := List.permutations digits
  let sorted_numbers := List.sort (List.map (λ l, l.foldl (λ acc d, acc * 10 + d) 0) permutations)
  (sorted_numbers.nth (20 - 1)).get_or_else 0 = 5687 :=
by
  let digits := [5, 6, 7, 8]
  let permutations := List.permutations digits
  let sorted_numbers := List.sort (List.map (λ l, l.foldl (λ acc d, acc * 10 + d) 0) permutations)
  have total_permutations : sorted_numbers.length = 24 := -- since 4! = 24
    by rw [←List.length_permutations digits], rfl
  have nth_exists : (sorted_numbers.nth (20 - 1)).is_some := 
    by rw [←total_permutations], apply List.nth_le_exists_of_le, norm_num, exact nat.lt.base 19
  show (sorted_numbers.nth (20 - 1)).get_or_else 0 = 5687, from sorry

end twenty_th_four_digit_number_l59_59753


namespace hyperbola_asymptotes_l59_59950

theorem hyperbola_asymptotes (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (e : ℝ) (h3 : e = 2)
    (h4 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) :
    ∀ x : ℝ, ∀ y : ℝ, (y = sqrt 3 * x ∨ y = - (sqrt 3) * x) :=
by
  sorry

end hyperbola_asymptotes_l59_59950


namespace inequality_proof_l59_59022

theorem inequality_proof (a b : ℝ) (h1 : a > 1) (h2 : b > 1) :
    (a^2 / (b - 1)) + (b^2 / (a - 1)) ≥ 8 := 
by
  sorry

end inequality_proof_l59_59022


namespace calculate_expression_value_l59_59854

theorem calculate_expression_value :
  (∏ k in (range (1009)).filter (λ k, odd (2*k+1)), (2*k+1)^4 + 4) /
  (∏ k in (range (1009)).filter (λ k, even (2*k-1)), (2*k-1)^4 + 4) =
  2020^2 + 1 :=
by
  sorry

end calculate_expression_value_l59_59854


namespace am_gm_inequality_example_l59_59941

theorem am_gm_inequality_example (x1 x2 x3 : ℝ)
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h_sum1 : x1 + x2 + x3 = 1) :
  (x2^2 / x1) + (x3^2 / x2) + (x1^2 / x3) ≥ 1 :=
by
  sorry

end am_gm_inequality_example_l59_59941


namespace correct_student_mark_l59_59734

theorem correct_student_mark :
  ∀ (total_marks total_correct_marks incorrect_mark correct_average students : ℝ)
  (h1 : total_marks = students * 100)
  (h2 : incorrect_mark = 60)
  (h3 : correct_average = 95)
  (h4 : total_correct_marks = students * correct_average),
  total_marks - incorrect_mark + (total_correct_marks - (total_marks - incorrect_mark)) = 10 :=
by
  intros total_marks total_correct_marks incorrect_mark correct_average students h1 h2 h3 h4
  sorry

end correct_student_mark_l59_59734


namespace find_x_l59_59096

theorem find_x : ∃ x, (2015 + x)^2 = x^2 ∧ x = -2015 / 2 := 
by
  sorry

end find_x_l59_59096


namespace possible_values_ab_possible_values_a_sub_b_l59_59609

noncomputable def abs_eq (x y : ℤ) := abs x = y

theorem possible_values_ab (a b : ℤ) (ha : abs a = 5) (hb : abs b = 3) :
  a + b = 8 ∨ a + b = 2 ∨ a + b = -2 ∨ a + b = -8 := by
  sorry

theorem possible_values_a_sub_b (a b : ℤ) (ha : abs a = 5) (hb : abs b = 3) (h : abs (a + b) = a + b) :
  a - b = 2 ∨ a - b = 8 := by
  sorry

end possible_values_ab_possible_values_a_sub_b_l59_59609


namespace necessary_and_sufficient_condition_l59_59586

variables {α β : Type} [plane α] [plane β] (l : α) (h_intersect : intersecting α β) (h_subset : l.is_subset_of α)

theorem necessary_and_sufficient_condition :
  (l ⊥ β) ↔ (α ⊥ β) :=
sorry

end necessary_and_sufficient_condition_l59_59586


namespace largest_possible_distance_between_spheres_l59_59779

noncomputable def largest_distance_between_spheres : ℝ :=
  110 + Real.sqrt 1818

theorem largest_possible_distance_between_spheres :
  let center1 := (3, -5, 7)
  let radius1 := 15
  let center2 := (-10, 20, -25)
  let radius2 := 95
  ∀ A B : ℝ × ℝ × ℝ,
    (dist A center1 = radius1) →
    (dist B center2 = radius2) →
    dist A B ≤ largest_distance_between_spheres :=
  sorry

end largest_possible_distance_between_spheres_l59_59779


namespace point_P_bisects_AB_l59_59436

theorem point_P_bisects_AB (O : Type*) [metric_space O] [normed_space ℝ O]
  (A B C D P : O) (radius : ℝ) (h_AB : dist A B = 2 * radius) 
  (h_C_on_circle : dist O C = radius) (h_CD_perpendicular_AB : ∠ O C D = 90) :
  midpoint A B = P :=
begin
  -- Skipping the proof
  sorry
end

end point_P_bisects_AB_l59_59436


namespace solution_set_inequality_l59_59058

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry
def odd_function_on_domain (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x, f(-x) = -f(x) ∧ (x ∈ (a,0) ∨ x ∈ (0,b))

theorem solution_set_inequality :
  ∀ (f : ℝ → ℝ), odd_function_on_domain f (-π) π → 
    (∀ x, 0 < x ∧ x < π → f'' x * sin x - f x * cos x < 0) → 
    (∀ x, f x < sqrt 2 * (f (π / 4)) * sin x ↔ (x ∈ (-π / 4, 0) ∨ x ∈ (π / 4, π))) :=
sorry

end solution_set_inequality_l59_59058


namespace no_unique_solution_l59_59224

theorem no_unique_solution (d : ℝ) (x y : ℝ) :
  (3 * (3 * x + 4 * y) = 36) ∧ (9 * x + 12 * y = d) ↔ d ≠ 36 := sorry

end no_unique_solution_l59_59224


namespace trace_shopping_bags_count_l59_59075

theorem trace_shopping_bags_count (gordon_bag1_weight gordon_bag2_weight trace_bag_weight : ℕ)
  (h1 : gordon_bag1_weight = 3)
  (h2 : gordon_bag2_weight = 7)
  (h3 : trace_bag_weight = 2) :
  (gordon_bag1_weight + gordon_bag2_weight) / trace_bag_weight = 5 := by
  rw [h1, h2, h3]
  sorry

end trace_shopping_bags_count_l59_59075


namespace right_triangle_area_inscribed_3_4_l59_59522

theorem right_triangle_area_inscribed_3_4 (r1 r2: ℝ) (h1 : r1 = 3) (h2 : r2 = 4) : 
  ∃ (S: ℝ), S = 150 :=
by
  sorry

end right_triangle_area_inscribed_3_4_l59_59522


namespace injective_iff_surjective_l59_59378

noncomputable def f (a b : ℝ) : ℝ → ℝ :=
λ x, if x ∈ (SetOf (λ x => x ∈ ℚ)) then a * x else b * x

theorem injective_iff_surjective (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) : 
  (Function.Injective (f a b) ↔ Function.Surjective (f a b)) ↔ (a / b ∈ ℚ) :=
by
  sorry

end injective_iff_surjective_l59_59378


namespace a_4_eq_15_l59_59270

noncomputable def a : ℕ → ℕ
| 0 => 1
| (n + 1) => 2 * a n + 1

theorem a_4_eq_15 : a 3 = 15 :=
by
  sorry

end a_4_eq_15_l59_59270


namespace sound_pressure_level_conversation_effective_sound_pressure_classroom_l59_59397

-- Define variables and constants
def p_ref : ℝ := 2 * 10 ^ (-5)
def p_e_conversation : ℝ := 0.002
def SPL_conversation : ℝ := 40
def SPL_classroom : ℝ := 90
def p_e_classroom := Real.sqrt(10) / 5

-- Define proof problems as Lean theorems
theorem sound_pressure_level_conversation : 20 * Real.log (p_e_conversation / p_ref) = SPL_conversation := by
  sorry

theorem effective_sound_pressure_classroom : (20 * Real.log (p_e_classroom / p_ref) = SPL_classroom) → 
                                              (p_e_classroom = Real.sqrt(10) / 5) := by
  sorry

end sound_pressure_level_conversation_effective_sound_pressure_classroom_l59_59397


namespace transformed_curve_eq_l59_59894

-- Define the original ellipse curve
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Define the transformation
def transform (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y

-- Prove the transformed curve satisfies x'^2 + y'^2 = 4
theorem transformed_curve_eq :
  ∀ (x y x' y' : ℝ), ellipse x y → transform x y x' y' → (x'^2 + y'^2 = 4) :=
by
  intros x y x' y' h_ellipse h_transform
  simp [ellipse, transform] at *
  sorry

end transformed_curve_eq_l59_59894


namespace new_pizza_dough_flour_l59_59650

theorem new_pizza_dough_flour (total_flour : ℚ)
  (initial_doughs : ℕ)
  (initial_flour_per_dough : ℚ)
  (new_doughs : ℕ) :
  initial_doughs * initial_flour_per_dough = total_flour →
  initial_doughs = 45 →
  initial_flour_per_dough = 1 / 9 →
  new_doughs = 15 →
  total_flour / new_doughs = 1 / 3 :=
by
  intros h_total h_initial_doughs h_initial_flour_per_dough h_new_doughs
  rw [h_initial_doughs, h_initial_flour_per_dough] at h_total
  rw h_new_doughs
  sorry

end new_pizza_dough_flour_l59_59650


namespace probability_six_highest_two_selected_l59_59814

noncomputable def calcProbability : ℚ :=
  let total_ways := Nat.choose 7 4 in
  let favorable_ways := Nat.choose 5 3 in
  (3 / 7) * (favorable_ways / total_ways)

theorem probability_six_highest_two_selected :
  calcProbability = 6 / 49 :=
by
  -- This is just a statement of the problem; the proof is omitted.
  sorry

end probability_six_highest_two_selected_l59_59814


namespace reese_practice_hours_l59_59366

-- Define the average number of weeks in a month
def avg_weeks_per_month : ℝ := 4.345

-- Define the number of hours Reese practices per week
def hours_per_week : ℝ := 4 

-- Define the number of months under consideration
def num_months : ℝ := 5

-- Calculate the total hours Reese will practice after five months
theorem reese_practice_hours :
  (num_months * avg_weeks_per_month * hours_per_week) = 86.9 :=
by
  -- We'll skip the proof part by adding sorry here
  sorry

end reese_practice_hours_l59_59366


namespace eduardo_wins_l59_59680

theorem eduardo_wins {p : ℕ} (hp : Nat.Prime p) (hge_two : 2 ≤ p) :
  ∃ a : Fin p → Fin 10, (∑ i in Finset.range p, a i * 10 ^ i) % p = 0 := by
  sorry

end eduardo_wins_l59_59680


namespace angle_is_37_5_l59_59551

-- Definitions of angles and their relationships
variables {angle1 angle2 angle3 angle4 x y : ℝ}

-- Conditions
def conditions :=
  (angle1 + angle2 = 180) ∧
  (angle3 = angle4) ∧
  (angle1 = 45 + x) ∧
  (angle3 = 30 + y) ∧
  (x = 2 * y)

-- Theorem statement to prove that angle4 = 37.5 given the conditions
theorem angle_is_37_5 (h : conditions) : angle4 = 37.5 :=
by { sorry }

end angle_is_37_5_l59_59551


namespace group_contains_2007_l59_59350

theorem group_contains_2007 : 
  ∃ k, 2007 ∈ {a | (k * (k + 1)) / 2 < a ∧ a ≤ ((k + 1) * (k + 2)) / 2} ∧ k = 45 :=
by sorry

end group_contains_2007_l59_59350


namespace total_boys_selected_is_11_l59_59173

-- Define the number of boys in Class A and Class B
def boys_class_a : Nat := 30
def boys_class_b : Nat := 25

-- Define the percentage to be selected for the survey
def percentage_selected : Rat := 20 / 100

-- Calculate the number of boys selected from Class A and Class B
def boys_selected_class_a : Nat := boys_class_a * (percentage_selected.numer / percentage_selected.denom)
def boys_selected_class_b : Nat := boys_class_b * (percentage_selected.numer / percentage_selected.denom)

-- Total number of boys selected from both classes
def total_boys_selected : Nat := boys_selected_class_a + boys_selected_class_b

-- The proof statement
theorem total_boys_selected_is_11 :
  total_boys_selected = 11 :=
by
  sorry

end total_boys_selected_is_11_l59_59173


namespace slips_prob_undefined_l59_59917

theorem slips_prob_undefined :
  let num_slips := 42
  let num_numbers := 14
  let slips_per_number := 3
  let slips_drawn := 4
  let total_methods := (Finset.Icc 1 num_slips).card.choose slips_drawn
  let p := 0
  let q := (Finset.Icc 1 num_numbers).card.choose 2 * slips_per_number.choose 2 * slips_per_number.choose 2 in
  q / p = ⊥ :=
by
  sorry

end slips_prob_undefined_l59_59917


namespace PatriciaHighlightFilmTheorem_l59_59357

def PatriciaHighlightFilmProblem : Prop :=
  let point_guard_seconds := 130
  let shooting_guard_seconds := 145
  let small_forward_seconds := 85
  let power_forward_seconds := 60
  let center_seconds := 180
  let total_seconds := point_guard_seconds + shooting_guard_seconds + small_forward_seconds + power_forward_seconds + center_seconds
  let num_players := 5
  let average_seconds := total_seconds / num_players
  let average_minutes := average_seconds / 60
  average_minutes = 2

theorem PatriciaHighlightFilmTheorem : PatriciaHighlightFilmProblem :=
  by
    -- Proof goes here
    sorry

end PatriciaHighlightFilmTheorem_l59_59357


namespace triangles_area_equal_l59_59322

noncomputable def Point := ℝ × ℝ

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

variable (a b : ℝ)
-- Given conditions as variables
variable (h1 : a > b)
variable (A B C : Point)
variable (D E F : Point)

-- Coordinates in terms of a and b
def A := (a, 0)
def B := (2 * a, 0)
def C := (3 * a, 0)
def D := (0, b)
def E := (0, 2 * b)
def F := (0, 3 * b)

theorem triangles_area_equal (a b : ℝ) (h1 : a > b) :
  area_of_triangle A E C = area_of_triangle D B F := by
  sorry

end triangles_area_equal_l59_59322


namespace coeff_x3_in_expansion_l59_59997

theorem coeff_x3_in_expansion (n : ℕ) 
  (h_sum_coeffs : ∑ i in range (n + 1), (2 ^ i) * (1 ^ (n - i)) = 729) 
  : 
  let expansion := (2 * x - 3) * (x - 1) ^ n 
  in coefficient expansion x^3 = 90 := 
by 
sorry

end coeff_x3_in_expansion_l59_59997


namespace marquita_gardens_l59_59694

open Nat

theorem marquita_gardens (num_mancino_gardens : ℕ) 
  (length_mancino_garden width_mancino_garden : ℕ) 
  (num_marquita_gardens : ℕ) 
  (length_marquita_garden width_marquita_garden : ℕ)
  (total_area : ℕ) 
  (h1 : num_mancino_gardens = 3)
  (h2 : length_mancino_garden = 16)
  (h3 : width_mancino_garden = 5)
  (h4 : length_marquita_garden = 8)
  (h5 : width_marquita_garden = 4)
  (h6 : total_area = 304)
  (hmancino_area : num_mancino_gardens * (length_mancino_garden * width_mancino_garden) = 3 * (16 * 5))
  (hcombined_area : total_area = num_mancino_gardens * (length_mancino_garden * width_mancino_garden) + num_marquita_gardens * (length_marquita_garden * width_marquita_garden)) :
  num_marquita_gardens = 2 :=
sorry

end marquita_gardens_l59_59694


namespace trapezoid_reconstruction_possible_l59_59018

variables {Point : Type} [affine_space Point]

-- Assume we have known midpoints M, N, P and a vertex Q of a trapezoid
variables (M N P Q : Point)

-- The main theorem we want to prove
theorem trapezoid_reconstruction_possible
  (midpoint_MN_leg_1 : M = midpoint Q R)
  (midpoint_MN_leg_2 : N = midpoint Q S)
  (midpoint_base_lower : P = midpoint R S)
  (parallel_RQ_PS : parallel R Q S P)
  (parallel_PQ : parallel P Q)
  : ∃ (trapezoid : Trapezoid), is_reconstructed_trapezoid trapezoid M N P Q :=
sorry

end trapezoid_reconstruction_possible_l59_59018


namespace tony_can_send_presents_l59_59408
-- Import necessary mathematical library

-- Declare the main theorem for the problem
theorem tony_can_send_presents (n : ℕ) 
  (segments : finset (fin (2 * n) × fin (2 * n)))
  (no_three_intersect : ∀ a b c : (fin (2 * n) × fin (2 * n)), 
    a ∈ segments ∧ b ∈ segments ∧ c ∈ segments → 
    ¬ (a.fst = b.fst ∧ b.fst = c.fst ∨ a.snd = b.snd ∧ b.snd = c.snd))
  (each_pair_intersects_once : ∀ a b : (fin (2 * n) × fin (2 * n)), 
    a ∈ segments ∧ b ∈ segments → 
    a ≠ b → ∃ p, p ∈ segments ∧ (a.fst = p ∨ a.snd = p) ∧ (b.fst = p ∨ b.snd = p)):
    ∃ sinks : finset (fin (2 * n)), sinks.card = n := 
begin
  -- To be completed
  sorry
end

end tony_can_send_presents_l59_59408


namespace five_inv_mod_33_l59_59882

theorem five_inv_mod_33 : ∃ x : ℤ, 0 ≤ x ∧ x < 33 ∧ 5 * x ≡ 1 [MOD 33] :=
by
  use 20
  split
  · norm_num
  split
  · norm_num
  · norm_num
  · rfl
sorry

end five_inv_mod_33_l59_59882


namespace paper_stack_weight_l59_59129

theorem paper_stack_weight
  (sheets_total : ℕ)
  (height_total : ℝ)
  (height_stack : ℝ)
  (weight_stack : ℝ)
  (sheets_in_stack : ℕ)
  (thickness_per_sheet : ℝ)
  (weight_per_sheet : ℝ)
  : sheets_total = 300 →
    height_total = 2.4 →
    height_stack = 6.0 →
    weight_stack = 900 →
    thickness_per_sheet = height_total / sheets_total →
    sheets_in_stack = height_stack / thickness_per_sheet →
    weight_per_sheet = weight_stack / sheets_total →
    sheets_in_stack = 750 →
    weight_per_sheet = 3 →
    weight_stack = sheets_in_stack * weight_per_sheet →
    weight_stack = 2250 :=
begin
  sorry
end

end paper_stack_weight_l59_59129


namespace hyperbola_equation_l59_59140

def hyperbola_asymptotes (x y : ℝ) (a b : ℝ) : Prop :=
  y = (b / a) * x ∨ y = (-b / a) * x

theorem hyperbola_equation :
  ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ (∃ (c : ℝ), c = 4 ∧ (b / a = sqrt 7 / 3) ∧ (a^2 + b^2 = c^2))
  ∧ (a^2 = 9) ∧ (b^2 = 7) ∧
  (∀ x y : ℝ, (y = (b / a) * x ∨ y = (-b / a) * x) →
              (x^2 / 9 - y^2 / 7 = 1)) :=
by
  sorry

end hyperbola_equation_l59_59140


namespace sequence_bounded_by_sqrt_l59_59324

noncomputable def sequence (N : ℕ) : ℕ → ℝ
| 0       := 0
| 1       := 1
| (n + 1) := sequence n (2 - 1 / N) - sequence (n - 1)

theorem sequence_bounded_by_sqrt (N : ℕ) (hN : 0 < N) :
  ∀ n, sequence N n < Real.sqrt (N + 1) :=
begin
  sorry
end

end sequence_bounded_by_sqrt_l59_59324


namespace volume_comparison_l59_59604

-- Define points and their reflections
variables (O P Q R P' Q' R' A B C D : Type)
variables [geometry_space O P Q R] [geometry_space O P' Q' R']
variables [geometry_space A B C D]

-- Assumptions about the structure
axiom reflection_over_O : ∀ (P Q R : geometry_space) (O : Type), reflection(P, O) = P' ∧ reflection(Q, O) = Q' ∧ reflection(R, O) = R'
axiom edges_bisected : ∀ (A B C D : geometry_space) (P Q R P' Q' R' : geometry_space), bisected(A, P) ∧ bisected(B, Q) ∧ bisected(C, R) ∧ bisected(D, P') ∧ bisected(B, Q') ∧ bisected(C, R')
axiom volume_ratio : ∀ (ABCD OPQR : geometry_space), volume(ABCD) = 16 * volume(OPQR)

-- Problem statement to prove
theorem volume_comparison
  (O P Q R P' Q' R' A B C D : Type)
  [geometry_space O P Q R]
  [geometry_space O P' Q' R']
  [geometry_space A B C D]
  (reflection_over_O : ∀ (P Q R : geometry_space) (O : Type), reflection(P, O) = P' ∧ reflection(Q, O) = Q' ∧ reflection(R, O) = R')
  (edges_bisected : ∀ (A B C D : geometry_space) (P Q R P' Q' R' : geometry_space), bisected(A, P) ∧ bisected(B, Q) ∧ bisected(C, R) ∧ bisected(D, P') ∧ bisected(B, Q') ∧ bisected(C, R'))
  (volume_ratio : ∀ (ABCD OPQR : geometry_space), volume(ABCD) = 16 * volume(OPQR)) :
  volume(ABCD) = 16 * volume(OPQR) :=
by
  sorry

end volume_comparison_l59_59604


namespace phi_odd_function_l59_59544

theorem phi_odd_function (k : ℤ) :
  ∃ φ : ℝ, (∀ x : ℝ, (sqrt 3) * cos (3 * x - φ) - sin (3 * x - φ) = -((sqrt 3) * cos (3 * -x - φ) - sin (3 * -x - φ))) → 
  φ = k * π - π / 3 := 
sorry

end phi_odd_function_l59_59544


namespace greater_integer_is_80_l59_59065

theorem greater_integer_is_80 (m n : ℕ) (hm : 9 < m ∧ m < 100) (hn : 9 < n ∧ n < 100) (h_distinct : m ≠ n)
  (h_sum : m + n = 120)
  (h_div : (m + (n / 100.0)) ∣ 120) : max m n = 80 := 
sorry

end greater_integer_is_80_l59_59065


namespace describes_cylinder_l59_59913

noncomputable def is_cylinder (cylindrical_coords : ℝ × ℝ × ℝ) (d : ℝ) : Prop :=
  let (r, θ, z) := cylindrical_coords
  r = d

theorem describes_cylinder (d : ℝ) (h: d > 0) :
  ∀ (coords : ℝ × ℝ × ℝ), is_cylinder coords d :=
by
  intro coords
  cases coords with r θ z
  dsimp [is_cylinder]
  sorry

end describes_cylinder_l59_59913


namespace equivalent_conditions_l59_59711

def condition1 (x y : ℝ) : Prop :=
  abs x ≤ 1 ∧ abs y ≤ 1 ∧ x * y ≤ 0

def condition2 (x y : ℝ) : Prop :=
  abs x ≤ 1 ∧ abs y ≤ 1 ∧ x^2 + y^2 ≤ 1

def inequality (x y : ℝ) : Prop :=
  (√(1 - x^2) * √(1 - y^2)) ≥ (x * y)

theorem equivalent_conditions (x y : ℝ) : inequality x y ↔ condition1 x y ∨ condition2 x y := 
sorry

end equivalent_conditions_l59_59711


namespace f_2008th_derivative_at_0_l59_59678

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4))^6 + (Real.cos (x / 4))^6

theorem f_2008th_derivative_at_0 : (deriv^[2008] f) 0 = 3 / 8 :=
sorry

end f_2008th_derivative_at_0_l59_59678


namespace total_practice_hours_l59_59368

def weekly_practice_hours : ℕ := 4
def weeks_per_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours :
  weekly_practice_hours * weeks_per_month * months = 80 := by
  sorry

end total_practice_hours_l59_59368


namespace solution_set_of_inequality_eqn_l59_59891

theorem solution_set_of_inequality_eqn :
  {x : ℝ | ⌊x * ⌊x⌋⌋ = 42 } = set.Ico 7 (43 / 6) :=
by
  sorry

end solution_set_of_inequality_eqn_l59_59891


namespace number_of_valid_ab_l59_59614

def is_digit (d : ℕ) : Prop := d >= 0 ∧ d < 10

def distinct_digits (a b : ℕ) : Prop := is_digit a ∧ is_digit b ∧ a ≠ b

def divisible_by (n divisor : ℕ) : Prop := ∃ k : ℕ, n = k * divisor

theorem number_of_valid_ab : 
  ∃ (n : ℕ), n = 3 ∧ 
  ∀ (a b : ℕ), distinct_digits a b ∧ a ≠ 0 → 
  let ab := 10 * a + b in
  let ababab := 10101 * ab in
  divisible_by ababab 217 → 
  ab ∈ {31, 62, 93} :=
sorry

end number_of_valid_ab_l59_59614


namespace equation_solution_is_ten_l59_59195

theorem equation_solution_is_ten (x : ℝ) (hx : x > 0) : 
  x ^ (Real.log10 x) = x^2 / 10 ↔ x = 10 :=
sorry

end equation_solution_is_ten_l59_59195


namespace greatest_natural_number_perf_square_l59_59897

theorem greatest_natural_number_perf_square :
  ∃ n : ℕ, n ≤ 2023 ∧ 
  (∑ i in finset.range(n+1), i^2) * 
  (∑ i in finset.range(n+1, 2*n+1), i^2) ∈ ℕ ∧
  (∑ i in finset.range(n+1), i^2) * 
  (∑ i in finset.range(n+1, 2*n+1), i^2) = (1921 * (1921 + 1) * (2*1921 + 1) / 6) * 
  (1921 * (2*1921 + 1) * (7*1921 + 1) / 6) :=
sorry

end greatest_natural_number_perf_square_l59_59897


namespace sequence_contains_integer_term_l59_59339

theorem sequence_contains_integer_term (M : ℕ) (hM : 2 ≤ M) :
  ∃ k : ℕ, ∃ a_k : ℝ, a_k = (λ a_0 => 
    Nat.recOn k a_0 
      (λ (n : ℕ) (a_n : ℝ), a_n * ⌊a_n⌋)
  ) (↑((2 * M + 1) / 2)) 
  ∧ a_k ∈ ℤ :=
by
  sorry

end sequence_contains_integer_term_l59_59339


namespace ratio_Ruffy_Orlie_l59_59019

-- Definitions of variables R and O
variable (R O : ℕ)

-- Setting the known conditions
axiom Ruffy_age : R = 9
axiom Age_relation : R - 4 = 1 + (1 / 2) * (O - 4)

-- Statement to prove
theorem ratio_Ruffy_Orlie : (R / O) = (3 / 4) :=
by 
  -- Introduction of variables and known axioms
  intro R Ruffy_age Age_relation
  -- skipping the proof as required
  sorry

end ratio_Ruffy_Orlie_l59_59019


namespace sqrt_72_eq_6_sqrt_2_l59_59025

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end sqrt_72_eq_6_sqrt_2_l59_59025


namespace find_real_solutions_l59_59892

noncomputable def polynomial_expression (x : ℝ) : ℝ := (x - 2)^2 * (x - 4) * (x - 1)

theorem find_real_solutions :
  ∀ (x : ℝ), (x ≠ 3) ∧ (x ≠ 5) ∧ (polynomial_expression x = 1) ↔ (x = 1 ∨ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2) := sorry

end find_real_solutions_l59_59892


namespace cistern_fill_time_l59_59449

variable (A_rate : ℚ) (B_rate : ℚ) (C_rate : ℚ)
variable (total_rate : ℚ := A_rate + C_rate - B_rate)

theorem cistern_fill_time (hA : A_rate = 1/7) (hB : B_rate = 1/9) (hC : C_rate = 1/12) :
  (1/total_rate) = 252/29 :=
by
  rw [hA, hB, hC]
  sorry

end cistern_fill_time_l59_59449


namespace part1_intersections_two_points_part2_minimum_area_chord_l59_59598

variable (a : ℝ) (h : a > 0)

def C1 (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (2 * a^2) = 1
def F1 : ℝ × ℝ := (-Real.sqrt 3 * a, 0)
def C2 (x y : ℝ) : Prop := y^2 = -4 * Real.sqrt 3 * a * x

theorem part1_intersections_two_points :
  ∃ x1 x2 : ℝ, x1 < x2 ∧
  (∃ y1 y2 : ℝ, C1 a x1 y1 ∧ C2 a x1 y1 ∧ C1 a x2 y2 ∧ C2 a x2 y2) :=
sorry

theorem part2_minimum_area_chord :
  (∃ k : ℝ, ∃ A B : (ℝ × ℝ), (A.1+x=0 || B.1 ≠ x)  ∧
  A ≠ B ∧ F1 = (A+B.toℝ)/2) ∧
  (∀ x y : ℝ, (C2 x y → 
  (y for A) (x=x coordinate) ⊛ 
  (( y*y_ob : ℝ) y*y_ob=y/y*g -> ::(A to B)+coordina) → 
  (Area_ADT(xA +minimum (area6a^2))) :=
sorry

end part1_intersections_two_points_part2_minimum_area_chord_l59_59598


namespace area_of_triangle_ACD_l59_59634

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
1 / 2 * a * b

theorem area_of_triangle_ACD :
  ∀ (AC AD : ℝ),
    AC = 8 * Real.sqrt 2 →
    AD = 3 →
    ∃ DC : ℝ, DC = Real.sqrt (AC^2 - AD^2) ∧
    area_of_triangle AD DC = 3 * Real.sqrt 119 / 2 :=
by
  intros AC AD hAC hAD
  use Real.sqrt (AC^2 - AD^2)
  split
  sorry
  sorry

end area_of_triangle_ACD_l59_59634


namespace initial_birds_179_l59_59728

theorem initial_birds_179 (B : ℕ) (h1 : B + 38 = 217) : B = 179 :=
sorry

end initial_birds_179_l59_59728


namespace part1_part2_l59_59266

-- Define the function f
def f (x : ℝ) (a : ℝ) (k : ℝ) : ℝ := k * a^x - a^(-x)

-- Conditions for part 1
axiom a_pos : ∀ a : ℝ, a > 0 
axiom a_ne_one : ∀ a : ℝ, a ≠ 1 
axiom f_odd : ∀ x a k, f (x) (a) (k) = -f (-x) (a) (k)
axiom f_one_pos : ∀ a k, f (1) (a) (k) > 0
def k := 1
noncomputable def f_part_1 (x : ℝ) (a : ℝ) : ℝ := f (x) (a) (1)

-- Part 1 proof statement
theorem part1 (x : ℝ) (a : ℝ) : 
  a > 1 → f_part_1 (x^2 + 2*x) a + f_part_1 (x - 4) a > 0 ↔ (x > 1 ∨ x < -4) := sorry
  
-- Conditions for part 2
axiom f_one_eq_three_half : ∀ a k, f (1) (a) (k) = 3 / 2

-- Define the function g
def g (x : ℝ) (a : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 4 * f (x) (a) (1)

-- Part 2 proof statement
theorem part2 (a : ℝ) :
  f (1) (a) 1 = 3 / 2 →
  ∃ x_min : ℝ, (x_min ∈ Set.Ici 1) ∧ g x_min a = -2 ∧ x_min = Real.logb a (1 + Real.sqrt 2) := sorry

end part1_part2_l59_59266


namespace count_positive_sums_l59_59681

theorem count_positive_sums (n : ℕ) (a : ℕ → ℝ) : 
  (finset.filter (λ s : finset ℕ, (s.sum a.to_fun) > 0) (finset.powerset (finset.range n))).card = 2^(n-1) := 
sorry

end count_positive_sums_l59_59681


namespace egg_count_l59_59767

theorem egg_count :
  ∃ x : ℕ, 
    (∀ e1 e10 e100 : ℤ, 
      (e1 = 1 ∨ e1 = -1) →
      (e10 = 10 ∨ e10 = -10) →
      (e100 = 100 ∨ e100 = -100) →
      7 * x + e1 + e10 + e100 = 3162) → 
    x = 439 :=
by 
  sorry

end egg_count_l59_59767


namespace fraction_of_phone_numbers_l59_59491

theorem fraction_of_phone_numbers :
  let valid_phone_numbers := 7 * 10^5 * 9
  let valid_phone_numbers_3_and_5 := 10^5
  valid_phone_numbers_3_and_5 / valid_phone_numbers = 1 / 63 :=
by
  let valid_phone_numbers := 7 * 10^5 * 9
  let valid_phone_numbers_3_and_5 := 10^5
  have h : valid_phone_numbers_3_and_5 / valid_phone_numbers = (10^5) / (7 * 10^5 * 9) := by rfl
  rw h
  simp
  norm_num
  sorry

end fraction_of_phone_numbers_l59_59491


namespace students_behind_Yoongi_l59_59809

theorem students_behind_Yoongi :
  ∀ (n : ℕ), n = 20 → ∀ (j y : ℕ), j = 1 → y = 2 → n - y = 18 :=
by
  intros n h1 j h2 y h3
  sorry

end students_behind_Yoongi_l59_59809


namespace sec_of_negative_420_eq_2_l59_59886

theorem sec_of_negative_420_eq_2 :
  sec (-420 : ℝ) = 2 :=
by
  -- Definitions and conditions
  let sec (θ : ℝ) := 1 / cos θ
  have h_cos_period : ∀ θ k, cos (θ + 360 * k) = cos θ,
  from sorry
  have h_cos_60 : cos 60 = 1 / 2,
  from sorry

  -- Proof starts (skipping the actual steps)
  sorry

end sec_of_negative_420_eq_2_l59_59886


namespace find_paycheck_l59_59706

variable (P : ℝ) -- P represents the paycheck amount

def initial_balance : ℝ := 800
def rent_payment : ℝ := 450
def electricity_bill : ℝ := 117
def internet_bill : ℝ := 100
def phone_bill : ℝ := 70
def final_balance : ℝ := 1563

theorem find_paycheck :
  initial_balance - rent_payment + P - (electricity_bill + internet_bill) - phone_bill = final_balance → 
    P = 1563 :=
by
  sorry

end find_paycheck_l59_59706


namespace meaningful_fraction_l59_59769

theorem meaningful_fraction (x : ℝ) : (∃ (f : ℝ), f = 2 / x) ↔ x ≠ 0 :=
by
  sorry

end meaningful_fraction_l59_59769


namespace room_perimeter_l59_59994

theorem room_perimeter (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 12) : 2 * (l + b) = 16 :=
by sorry

end room_perimeter_l59_59994


namespace ice_cream_initial_amount_l59_59176

noncomputable def initial_ice_cream (milkshake_count : ℕ) : ℕ :=
  12 * milkshake_count

theorem ice_cream_initial_amount (m_i m_f : ℕ) (milkshake_count : ℕ) (I_f : ℕ) :
  m_i = 72 →
  m_f = 8 →
  milkshake_count = (m_i - m_f) / 4 →
  I_f = initial_ice_cream milkshake_count →
  I_f = 192 :=
by
  intros hmi hmf hcount hIc
  sorry

end ice_cream_initial_amount_l59_59176


namespace sum_of_integer_solutions_l59_59784

/-- 
  The sum of all integer solutions to the inequality |n| < |n - 2| < 10 is -36 
  We need to define n as an integer and specify a condition using absolute values
  which translates mathematically to the required sum.
-/

theorem sum_of_integer_solutions : 
  let solutions := { n : ℤ | abs n < abs (n - 2) ∧ abs (n - 2) < 10 } in
  ∑ n in solutions, n = -36 :=
by 
  sorry

end sum_of_integer_solutions_l59_59784


namespace geometric_seq_general_term_sum_new_seq_first_n_terms_l59_59597

-- Defining the geometric sequence with the given conditions
def geometric_seq (n : ℕ) : ℕ := 2^(n-1)

-- Defining the sequence 2n * a_n
def new_seq (n : ℕ) : ℕ := n * 2^n

-- Defining the sum of the first n terms of the new sequence
def sum_new_seq (n : ℕ) : ℕ := 2 + (n-1) * 2^(n+1)

-- Proof problem statement
theorem geometric_seq_general_term (a_n : ℕ → ℕ) (h1 : a_n 2 = 2) (h2 : a_n 3 = 4) (q > 1) :
  ∀ n, a_n n = 2^(n-1) := sorry

theorem sum_new_seq_first_n_terms :
  ∀ n, (∑ i in Finset.range n, new_seq (i + 1)) = sum_new_seq n := sorry

end geometric_seq_general_term_sum_new_seq_first_n_terms_l59_59597


namespace spider_crawl_distance_l59_59151

theorem spider_crawl_distance :
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  abs (b - a) + abs (c - b) + abs (d - c) = 20 :=
by
  let a := -3
  let b := -8
  let c := 4
  let d := 7
  sorry

end spider_crawl_distance_l59_59151


namespace smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l59_59905

-- Problem (a): Smallest n such that n! is divisible by 2016
theorem smallest_n_divisible_by_2016 : ∃ (n : ℕ), n = 8 ∧ 2016 ∣ n.factorial :=
by
  sorry

-- Problem (b): Smallest n such that n! is divisible by 2016^10
theorem smallest_n_divisible_by_2016_pow_10 : ∃ (n : ℕ), n = 63 ∧ 2016^10 ∣ n.factorial :=
by
  sorry

end smallest_n_divisible_by_2016_smallest_n_divisible_by_2016_pow_10_l59_59905


namespace monotone_decreasing_value_of_m_l59_59871

theorem monotone_decreasing_value_of_m :
  ∃ m : ℝ, (∀ x : ℝ, x > 0 → differentiable_at ℝ (λ x, (m^2 - 2*m - 2) * x^(2-m)) x) ∧ 
    (∀ x : ℝ, x > 0 → (∂ (λ x, (m^2 - 2*m - 2) * x^(2-m)) / ∂ x) x < 0) ↔
  m = 1 + real.sqrt 3 :=
begin
  sorry
end

end monotone_decreasing_value_of_m_l59_59871


namespace domain_of_f_l59_59044

def f (x : ℝ) : ℝ := sqrt (sin x - 1 / 2)

noncomputable def domain (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ k : ℤ, π / 6 + 2 * k * π ≤ x ∧ x ≤ 5 * π / 6 + 2 * k * π}

theorem domain_of_f :
  ∀ x, x ∈ domain f ↔ ∃ k : ℤ, (π / 6 + 2 * k * π ≤ x ∧ x ≤ 5 * π / 6 + 2 * k * π) :=
by
  sorry

end domain_of_f_l59_59044


namespace tan_alpha_value_l59_59938

open Real

-- Define the angle alpha in the third quadrant
variable {α : ℝ}

-- Given conditions
def third_quadrant (α : ℝ) : Prop :=  π < α ∧ α < 3 * π / 2
def sin_alpha (α : ℝ) : Prop := sin α = -4 / 5

-- Statement to prove
theorem tan_alpha_value (h1 : third_quadrant α) (h2 : sin_alpha α) : tan α = 4 / 3 :=
sorry

end tan_alpha_value_l59_59938


namespace part1_part2_l59_59242

-- Define the conditions
variable {a b c A B C : ℝ}
variable (h1 : a = c)
variable (h2 : A = 90)
variable (h3 : a = Real.sqrt 2)
variable (triangle_condition : sin^2 (B + C) = cos (B - C) - cos (B + C))

-- Part 1: Prove cos A = 1/4 given a = c
theorem part1 (h1 : a = c) (triangle_condition : sin^2 (B + C) = cos (B - C) - cos (B + C)) :
  cos A = 1 / 4 :=
sorry

-- Part 2: Prove the area of ΔABC is 1/2 given A = 90°, a = √2
theorem part2 (h2 : A = 90) (h3 : a = Real.sqrt 2) (triangle_condition : sin^2 (B + C) = cos (B - C) - cos (B + C)) :
  (1 / 2) * b * c = 1 / 2 :=
sorry

end part1_part2_l59_59242


namespace theta_value_decreasing_function_l59_59216

theorem theta_value_decreasing_function (θ : ℝ) (k : ℤ) :
  (θ ∈ Ioo 0 real.pi) ∧
  (∀ x, f (-x) = - f x) ∧
  (∀ x, f x = 2 * real.sin (2 * x + θ + real.pi / 3)) ∧
  (∀ x ∈ Icc (-(real.pi / 4)) 0, deriv f x < 0) →
  θ = 2 * real.pi / 3 := 
sorry

noncomputable def f (x : ℝ) (θ : ℝ) : ℝ :=
  real.sin (2 * x + θ) + real.sqrt 3 * real.cos (2 * x + θ)

end theta_value_decreasing_function_l59_59216


namespace isosceles_triangle_perimeter_l59_59479

-- Defining the basic characteristics of the right triangles and the unit square.
def right_triangle (a b : ℝ) : Prop :=
  a = 1 / 2 ∧ b = 1 / 2

-- Given four such right triangles cut from a unit square.
def pieces (triangles : ℕ) : Prop :=
  triangles = 4

-- These right triangles can be arranged to form an isosceles triangle.
def isosceles_triangle (base leg : ℝ) : Prop :=
  base = sqrt 2 ∧ leg = sqrt 2 / 2 ∧ 2 * leg + base = 2 * sqrt 2

-- Prove the perimeter of the formed isosceles triangle.
theorem isosceles_triangle_perimeter
  (a b base leg : ℝ)
  (h1 : pieces 4)
  (h2 : right_triangle a b)
  (h3 : isosceles_triangle base leg) :
  base + 2 * leg = 2 * sqrt 2 :=
by sorry

end isosceles_triangle_perimeter_l59_59479


namespace right_triangles_in_rectangle_l59_59300

/--
In a rectangle \(ABCD\), a line segment \(PR\) is drawn dividing the rectangle such that triangle \(APR\) 
and triangle \(PBC\) are congruent right triangles with vertex \(P\) on side \(AB\) and vertex \(R\) on 
side \(CD\). Prove that the number of right triangles that can be formed using the vertices 
\(\{A, P, R, B, C, D\}\) is 12.
-/
theorem right_triangles_in_rectangle 
  (A P R B C D : Type) 
  (rectangle : Rectangle ABCD) 
  (triangle_APR : is_congruent_right_triangle APR PBC) 
  (vertex_P : OnSide P AB) 
  (vertex_R : OnSide R CD) : 
  right_triangles_count {A, P, R, B, C, D} = 12 :=
sorry

end right_triangles_in_rectangle_l59_59300


namespace passengers_count_l59_59845

def num_people (total_bags : ℕ) (bags_per_person : ℕ) : ℕ := total_bags / bags_per_person

theorem passengers_count (total_bags : ℕ) (bags_per_person : ℕ) 
  (h1 : total_bags = 32) (h2 : bags_per_person = 8) :
  num_people total_bags bags_per_person = 4 :=
by
  rw [h1, h2]
  simp
  done

end passengers_count_l59_59845


namespace sum_slope_y_intercept_median_line_through_E_l59_59414

-- Define the points D, E, and F
structure Point where
  x : ℝ
  y : ℝ

def D := Point.mk 2 10
def E := Point.mk 1 0
def F := Point.mk 11 0

-- Define the midpoint function
def midpoint (A B : Point) : Point :=
  Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)

-- Define the line that cuts the area in half and passes through E. This is the median from E.
def median_line_through_E (N : Point) : (ℝ × ℝ) :=
  let slope := (N.y - E.y) / (N.x - E.x)
  let y_intercept := E.y - slope * E.x
  (slope, y_intercept)

-- Define the sum of the slope and y-intercept
def sum_slope_y_intercept (slope_y_intercept : ℝ × ℝ) : ℝ :=
  slope_y_intercept.fst + slope_y_intercept.snd

-- Prove that the sum of the slope and y-intercept of the median line through E is 0
theorem sum_slope_y_intercept_median_line_through_E : 
  let N := midpoint D F in
  sum_slope_y_intercept (median_line_through_E N) = 0 :=
by
  simp [D, E, F, midpoint, median_line_through_E, sum_slope_y_intercept]
  sorry

end sum_slope_y_intercept_median_line_through_E_l59_59414


namespace tan_beta_value_l59_59578

theorem tan_beta_value (α β : ℝ)
  (h1 : (sin α * cos α) / (1 - cos (2 * α)) = 1 / 2)
  (h2 : tan (α - β) = 1 / 2) : tan β = 1 / 3 :=
sorry

end tan_beta_value_l59_59578


namespace find_FCA_angle_l59_59968

noncomputable def hyperbola_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def right_focus (F : ℝ × ℝ) (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
  F.1 = a * sqrt (1 + (b / a)^2) ∧ F.2 = 0

def angle_sum_triangle (A B F : ℝ × ℝ) (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def intersect_line_hyperbola (F C A B : ℝ × ℝ) : Prop :=
  -- This would include details about intersection logic which is represented schematically
  True

theorem find_FCA_angle (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (F C A B : ℝ × ℝ) (hF : right_focus F a b ha hb)
  (hIntersections : intersect_line_hyperbola F C A B)
  (angle_FAB : ∠ F A B = 50)
  (angle_FBA : ∠ F B A = 20) :
  ∠ F C A = 35 :=
sorry

end find_FCA_angle_l59_59968


namespace solution_set_f_l59_59192

noncomputable
def function_f (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, (x1 - x2) * (f x1 - f x2) < 0

noncomputable
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (x + 1) = -f (-(x + 1))

theorem solution_set_f (
  f : ℝ → ℝ
) (hf : function_f f) (hf_odd : is_odd_function f) :
  {x : ℝ | f (1 - x) < 0} = set.Iio 0 :=
sorry

end solution_set_f_l59_59192


namespace boy_can_choose_last_two_from_same_box_l59_59406

/-- Given 2n candies distributed in n boxes,
    where a girl and a boy alternately take one candy at a time with the girl going first,
    prove that the boy can always choose the candies in such a way that the last two candies are from the same box. -/
theorem boy_can_choose_last_two_from_same_box (n : ℕ) (boxes : Finset (Finset ℕ)):
  (∀ i ∈ boxes, ∃ k, i.card = k ∧ k ≥ 1) ∧ (2 * n = boxes.card.sum (λ i, i.card)) →
  (∃ strategy : ℕ → Finset ℕ → ℕ,
    (∀ turn (boxes : Finset ℕ), 
       let player := if turn % 2 = 0 then "girl" else "boy" in
       ∃ box ∈ boxes, 
         (player = "girl" → ∃! k, k ∈ box ∧ k = strategy turn boxes) ∧
         (player = "boy" → ∃! k, k ∈ box ∧ k = strategy turn boxes))
    ∧ (∀ boxes, boxes.card = 2 → ∃ box ∈ boxes, box.card = 2)) :=
sorry

end boy_can_choose_last_two_from_same_box_l59_59406


namespace cosine_product_identity_l59_59717

open Real

theorem cosine_product_identity (α : ℝ) (n : ℕ) :
  (List.foldr (· * ·) 1 (List.map (λ k => cos (2^k * α)) (List.range (n + 1)))) =
  sin (2^(n + 1) * α) / (2^(n + 1) * sin α) :=
sorry

end cosine_product_identity_l59_59717


namespace probability_two_flies_swept_away_l59_59011

-- Defining the initial conditions: flies at 12, 3, 6, and 9 o'clock positions
def flies_positions : List ℕ := [12, 3, 6, 9]

-- The problem statement
theorem probability_two_flies_swept_away : 
  (let favorable_intervals := 20 in
   let total_intervals := 60 in
   favorable_intervals / total_intervals = 1 / 3) :=
by
  sorry

end probability_two_flies_swept_away_l59_59011


namespace function_properties_l59_59740

def f(x : ℝ) : ℝ := x^3

theorem function_properties : 
  (∀ x : ℝ, f(-x) = -f(x)) ∧ (∀ x y : ℝ, x < y → f(x) < f(y)) :=
by
  sorry

end function_properties_l59_59740


namespace probability_two_flies_swept_away_l59_59010

-- Defining the initial conditions: flies at 12, 3, 6, and 9 o'clock positions
def flies_positions : List ℕ := [12, 3, 6, 9]

-- The problem statement
theorem probability_two_flies_swept_away : 
  (let favorable_intervals := 20 in
   let total_intervals := 60 in
   favorable_intervals / total_intervals = 1 / 3) :=
by
  sorry

end probability_two_flies_swept_away_l59_59010


namespace determine_n_l59_59196

theorem determine_n (n : ℕ) (h : 17^(4 * n) = (1 / 17)^(n - 30)) : n = 6 :=
by {
  sorry
}

end determine_n_l59_59196


namespace finitely_many_good_numbers_l59_59849

-- Definitions of conditions
def odd_nat (k : ℕ) : Prop := k % 2 = 1
def composite (n : ℕ) : Prop := 1 < n ∧ ∃ d, 1 < d ∧ d < n ∧ d ∣ n
def proper_divisors (n : ℕ) : set ℕ := { d | d ∣ n ∧ 1 < d ∧ d < n }
def good_number (k n : ℕ) : Prop := ∃ m, proper_divisors m = proper_divisors n ∪ {k}

-- Theorem statement
theorem finitely_many_good_numbers (k : ℕ) (h_k : odd_nat k) :
  ∃ N, ∀ n, good_number k n → n < N :=
sorry

end finitely_many_good_numbers_l59_59849


namespace compound_interest_l59_59848

-- Definitions according to conditions
def principal : ℝ := 10000
def rate : ℝ := 0.06
def years : ℕ := 5

-- The calculated answer
def expected_amount : ℝ := 13382

-- Proof statement
theorem compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) :
  P = principal → r = rate → n = years → A = P * (1 + r)^n → A ≈ expected_amount :=
by
  intros hP hr hn hA
  simp [hP, hr, hn] at hA
  linarith only [hA]
  sorry

end compound_interest_l59_59848


namespace tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l59_59594
noncomputable def f (x : ℝ) : ℝ := x^2 / Real.exp x

theorem tangent_line_through_origin (x y : ℝ) :
  (∃ a : ℝ, (x, y) = (a, f a) ∧ (0, 0) = (0, 0) ∧ y - f a = ((2 * a - a^2) / Real.exp a) * (x - a)) →
  y = x / Real.exp 1 :=
sorry

theorem max_value_on_interval : ∃ (x : ℝ), x = 9 / Real.exp 3 :=
  sorry

theorem min_value_on_interval : ∃ (x : ℝ), x = 0 :=
  sorry

end tangent_line_through_origin_max_value_on_interval_min_value_on_interval_l59_59594


namespace infinite_squares_in_sequence_l59_59057

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := a n + nat.floor (real.sqrt (a n))

theorem infinite_squares_in_sequence :
  ∃ (M : ℕ → ℕ), (∀ n, M n = n) ∧ 
  (∀ m : ℕ, ∃ n : ℕ, a n = (M m) ^ 2) :=
sorry

end infinite_squares_in_sequence_l59_59057


namespace solve_fractional_eq_l59_59727

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) : (x / (x + 1) - 1 = 3 / (x - 1)) → x = -1 / 2 :=
by
  sorry

end solve_fractional_eq_l59_59727


namespace sum_digits_of_n_l59_59482

theorem sum_digits_of_n (n : ℕ) (h1 : ∀ k, k ≤ 1500 → 
  ∃ rep, (rep.toNat = k) ∧ (∀ c ∈ rep.toList, '0' ≤ c ∧ c ≤ '9')) :
  (∃ n = 599, (5 + 9 + 9 = 23)) :=
by sorry

end sum_digits_of_n_l59_59482


namespace log_diff_is_six_l59_59197

theorem log_diff_is_six (h1 : 256 = 4^4) (h2 : 1 / 16 = 4^(-2)) :
  Real.log 256 / Real.log 4 - Real.log (1 / 16) / Real.log 4 = 6 :=
by
  sorry

end log_diff_is_six_l59_59197


namespace toby_photos_l59_59074

variable (p0 d c e x : ℕ)
def photos_remaining : ℕ := p0 - d + c + x - e

theorem toby_photos (h1 : p0 = 63) (h2 : d = 7) (h3 : c = 15) (h4 : e = 3) : photos_remaining p0 d c e x = 68 + x :=
by
  rw [h1, h2, h3, h4]
  sorry

end toby_photos_l59_59074


namespace relationship_among_a_b_c_l59_59243

noncomputable def a := Real.log 2 / 2
noncomputable def b := Real.log 3 / 3
noncomputable def c := Real.log 5 / 5

theorem relationship_among_a_b_c : c < a ∧ a < b := by
  sorry

end relationship_among_a_b_c_l59_59243


namespace mean_median_difference_l59_59353

theorem mean_median_difference :
  let total_students := 40
  let scores := [15, 20, 25, 10, 30]
  let points := [60, 75, 88, 92, 98]
  let students := list.zip scores points
  let mean := 
    (students.foldl (λ acc (percentage, point), acc + (percentage * total_students / 100) * point) 0) / total_students
  let sorted_points := list.repeat 60 (15 * total_students / 100) ++
                       list.repeat 75 (20 * total_students / 100) ++
                       list.repeat 88 (25 * total_students / 100) ++
                       list.repeat 92 (10 * total_students / 100) ++
                       list.repeat 98 (30 * total_students / 100)
  let median := (sorted_points.nth (total_students / 2 - 1) + sorted_points.nth (total_students / 2)) / 2
  abs (mean - median) = 3.4 :=
by
  sorry

end mean_median_difference_l59_59353


namespace total_burning_time_l59_59852

theorem total_burning_time (structure : ℕ) (arrangement : ℕ × ℕ) 
(burn_time : ℕ) (fire_start : ℕ) (spread_rate : ℕ) : 
structure = 38 ∧ arrangement = (3, 5) ∧ burn_time = 10 ∧ 
fire_start = 2 ∧ spread_rate = burn_time →
∃ t : ℕ, t = 65 :=
by
  intros h
  sorry

end total_burning_time_l59_59852


namespace logarithm_inequality_solution_l59_59901

theorem logarithm_inequality_solution (x : ℕ) :
  (log 4 x + log 2 (sqrt x - 1) < log 2 (log (sqrt 5) 5)) ↔ (x = 2 ∨ x = 3) :=
sorry

end logarithm_inequality_solution_l59_59901


namespace proposition_2_proposition_3_only_propositions_2_and_3_are_correct_l59_59256

open Real

def f (x : ℝ) := 4 * sin (2 * x + π / 3)

theorem proposition_2 : ∀ x : ℝ, f x = 4 * cos (2 * x - π / 6) := sorry

theorem proposition_3 : ∃ x : ℝ, (f x = 0) ∧ ∀ x' : ℝ, f x' = f (2 * (-π / 6) - x') := sorry

theorem only_propositions_2_and_3_are_correct : 
  (proposition_2 ∧ proposition_3) ∧ 
  ¬ (∀ x1 x2 : ℝ, (f x1 = 0 ∧ f x2 = 0) → (x1 - x2) % π = 0) ∧ 
  ¬ (∃ x : ℝ, ∀ x' : ℝ, f x' = f (2 * x - π / 6)) := sorry

end proposition_2_proposition_3_only_propositions_2_and_3_are_correct_l59_59256


namespace total_seeds_l59_59703

-- Define the conditions given in the problem
def morningMikeTomato := 50
def morningMikePepper := 30

def morningTedTomato := 2 * morningMikeTomato
def morningTedPepper := morningMikePepper / 2

def morningSarahTomato := morningMikeTomato + 30
def morningSarahPepper := morningMikePepper + 30

def afternoonMikeTomato := 60
def afternoonMikePepper := 40

def afternoonTedTomato := afternoonMikeTomato - 20
def afternoonTedPepper := afternoonMikePepper

def afternoonSarahTomato := morningSarahTomato + 20
def afternoonSarahPepper := morningSarahPepper + 10

-- Prove that the total number of seeds planted is 685
theorem total_seeds (total: Nat) : 
    total = (
        (morningMikeTomato + afternoonMikeTomato) + 
        (morningTedTomato + afternoonTedTomato) + 
        (morningSarahTomato + afternoonSarahTomato) +
        (morningMikePepper + afternoonMikePepper) + 
        (morningTedPepper + afternoonTedPepper) + 
        (morningSarahPepper + afternoonSarahPepper)
    ) := 
    by 
        have tomato_seeds := (
            morningMikeTomato + afternoonMikeTomato +
            morningTedTomato + afternoonTedTomato + 
            morningSarahTomato + afternoonSarahTomato
        )
        have pepper_seeds := (
            morningMikePepper + afternoonMikePepper +
            morningTedPepper + afternoonTedPepper + 
            morningSarahPepper + afternoonSarahPepper
        )
        have total_seeds := tomato_seeds + pepper_seeds
        sorry

end total_seeds_l59_59703


namespace find_sales_tax_percentage_l59_59816

theorem find_sales_tax_percentage (x : ℝ) (h_food_paid : 30.0) (h_tip_percentage : 0.10) (h_total_paid : 35.75) :
    30 + (x / 100) * 30 + 30 * h_tip_percentage = h_total_paid := 
sorry

end find_sales_tax_percentage_l59_59816


namespace correct_diagram_for_patient_treatment_l59_59384

def process_diagram : Type := 
  Σ' (diagram : Type), 
    (represent_process : diagram → Prop)

-- Define the condition for the process
def process_condition {diagram : process_diagram} (d : diagram.1) : Prop := 
  diagram.2 d

axiom diagram_used_for_patient_treatment : process_diagram :=
⟨_, λ d, process_condition d⟩

theorem correct_diagram_for_patient_treatment :
  Π (d : diagram_used_for_patient_treatment.1),
  diagram_used_for_patient_treatment.2 d ↔ d = "Process flowchart" := 
sorry

end correct_diagram_for_patient_treatment_l59_59384


namespace not_hexagonal_pyramid_l59_59465

-- Definition of the pyramid with slant height, base radius, and height
structure Pyramid where
  r : ℝ  -- Side length of the base equilateral triangle
  h : ℝ  -- Height of the pyramid
  l : ℝ  -- Slant height (lateral edge)
  hypo : h^2 + (r / 2)^2 = l^2

-- The theorem to prove a pyramid with all edges equal cannot be hexagonal
theorem not_hexagonal_pyramid (p : Pyramid) : p.l ≠ p.r :=
sorry

end not_hexagonal_pyramid_l59_59465


namespace incenter_of_triangle_is_intersection_of_angle_bisectors_l59_59477

theorem incenter_of_triangle_is_intersection_of_angle_bisectors 
  (A B C : Type) [plane : Euclidean_space ℝ 2] (triangle : triangle A B C) :
  ∃ P, is_incenter P (triangle) ∧ ∀ (angle_bisector : set (line (Euclidean_space ℝ 2))),
    angle_bisector ∈ set_of_angle_bisectors triangle → P ∈ angle_bisector :=
sorry

end incenter_of_triangle_is_intersection_of_angle_bisectors_l59_59477


namespace percentage_of_literate_inhabitants_l59_59630

-- Define the conditions
def total_inhabitants : ℕ := 1000
def male_percentage : ℝ := 0.60
def male_literate_percentage : ℝ := 0.20
def female_literate_percentage : ℝ := 0.325

-- Calculate the result to prove
def literate_percentage : ℝ :=
  let males := male_percentage * total_inhabitants
  let females := total_inhabitants - males
  let literate_males := male_literate_percentage * males
  let literate_females := female_literate_percentage * females
  let literate_total := literate_males + literate_females
  (literate_total / total_inhabitants) * 100

-- State the theorem
theorem percentage_of_literate_inhabitants : literate_percentage = 25 := by
  sorry

end percentage_of_literate_inhabitants_l59_59630


namespace goose_eggs_l59_59704

theorem goose_eggs (E : ℕ) 
  (h1 : 1 / 3 * E)
  (h2 : 4 / 5 * (1 / 3 * E)) 
  (h3 : 2 / 5 * (4 / 5 * (1 / 3 * E)) = 120) : 
  E = 1125 := 
  sorry

end goose_eggs_l59_59704


namespace solid_fits_all_hollows_l59_59481

-- Definitions of the given solids and their parameters
variable (a : ℝ)

def square_prism_base (a : ℝ) := 
  { edge_length := a }

def triangular_prism_base (a : ℝ) := 
  { base_length := a, height := a }

def cylinder_base (a : ℝ) := 
  { diameter := a }

-- Statement of the proof problem
theorem solid_fits_all_hollows (a : ℝ) :
  ∃ (solid : ℝ), solid = a ∧ 
    (∀ s ∈ [square_prism_base a, triangular_prism_base a, cylinder_base a], 
    solid_fits_in_hollow s solid) := 
begin
  sorry
end

end solid_fits_all_hollows_l59_59481


namespace probability_two_flies_swept_away_l59_59013

-- Defining the initial conditions: flies at 12, 3, 6, and 9 o'clock positions
def flies_positions : List ℕ := [12, 3, 6, 9]

-- The problem statement
theorem probability_two_flies_swept_away : 
  (let favorable_intervals := 20 in
   let total_intervals := 60 in
   favorable_intervals / total_intervals = 1 / 3) :=
by
  sorry

end probability_two_flies_swept_away_l59_59013


namespace find_angle_BCD_l59_59731

noncomputable def ABC_isosceles (A B C D : Type) [euclidean_space A B C D] (AB AC : ℝ) (angle_A : ℝ) (AD : ℝ) : Prop :=
  is_isosceles_triangle A B C AB AC ∧ angle_A = 20 ∧ AD = distance B C

theorem find_angle_BCD (A B C D : Type) [euclidean_space A B C D] (AB AC : ℝ) (angle_A : ℝ) (AD : ℝ)
  (h : ABC_isosceles A B C D AB AC angle_A AD) : angle B C D = 70 := 
  sorry

end find_angle_BCD_l59_59731


namespace largest_of_five_numbers_l59_59103

theorem largest_of_five_numbers : ∀ (a b c d e : ℝ), 
  a = 0.938 → b = 0.9389 → c = 0.93809 → d = 0.839 → e = 0.893 → b = max a (max b (max c (max d e))) :=
by
  intros a b c d e ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end largest_of_five_numbers_l59_59103


namespace perpendicular_CE_AD_l59_59306

variables {α : Type*} [ordered_ring α] {A B C D E : euclidean_geometry.point α}

noncomputable def is_right_triangle (A B C : euclidean_geometry.point α) :=
euclidean_geometry.angle A B C = 90

namespace euclidean_geometry

variable (ABC_is_right : is_right_triangle A B C)

theorem perpendicular_CE_AD (D_midpoint_BC : midpoint α B C D)
  (E_on_AB : ∃ t : α, 0 < t ∧ t < 1 ∧ E = A +ᵥ (t • (B - A))) 
  (ratio_AE_EB : abs (dist A E) / abs (dist E B) = 2) :
  ⊥ (line_through α C E) (line_through α A D) := sorry

end euclidean_geometry

end perpendicular_CE_AD_l59_59306


namespace x_is_one_if_pure_imaginary_l59_59621

theorem x_is_one_if_pure_imaginary
  (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x^2 + 3 * x + 2 ≠ 0) :
  x = 1 :=
sorry

end x_is_one_if_pure_imaginary_l59_59621


namespace total_practice_hours_l59_59370

def weekly_practice_hours : ℕ := 4
def weeks_per_month : ℕ := 4
def months : ℕ := 5

theorem total_practice_hours :
  weekly_practice_hours * weeks_per_month * months = 80 := by
  sorry

end total_practice_hours_l59_59370


namespace factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l59_59520

-- Math Proof Problem 1
theorem factorize_a_squared_minus_25 (a : ℝ) : a^2 - 25 = (a + 5) * (a - 5) :=
by
  sorry

-- Math Proof Problem 2
theorem factorize_2x_squared_y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 :=
by
  sorry

end factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l59_59520


namespace count_of_incorrect_propositions_l59_59869

open Real

theorem count_of_incorrect_propositions :
  let p1 := ¬(x^2 - 3 * x + 2 = 0) → ¬(x = 1),
      p2 := ∀ x > 2, x^2 - 3*x + 2 > 0 ∧ ∃ x < 1, x^2 - 3*x + 2 > 0,
      p3 := ∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q),
      p4 := ∃ x : ℝ, x^2 + x + 1 < 0 → ∀ x : ℝ, x^2 + x + 1 ≥ 0
  in (¬p1 ∧ ¬p2 ∧ p3 ∧ ¬p4)
  
:= true :=
by
  sorry

end count_of_incorrect_propositions_l59_59869


namespace total_cost_expr_min_cost_l59_59817

def total_cost (x : ℝ) (t : ℝ) :=  t * (2 * (2 + x^2 / 360) + 14)
def trip_time (distance : ℝ) (speed : ℝ) := distance / speed
def truck_fuel_cost (x : ℝ) := 2 + x^2 / 360

theorem total_cost_expr (x : ℝ) (hx : 50 ≤ x ∧ x ≤ 100) :
  let t := 130 / x in
  total_cost x t = 2340 / x + 13 * x / 18 :=
by
  let t := 130 / x
  have ht : t = 130 / x := rfl
  rw [total_cost, ht]
  have hfuel : 2 * (2 + x^2 / 360) = 4 + x^2 / 180 := by ring
  rw [hfuel, mul_add, ←div_eq_mul_one_div, ←div_eq_mul_one_div]
  simp only [mul_add, div_add_div_same, div_self, mul_one]
  sorry

theorem min_cost (x : ℝ) (hx : 50 ≤ x ∧ x ≤ 100) :
  x = 18 * real.sqrt 10 →
  let y := total_cost x (trip_time 130 x) in
  y = 26 * real.sqrt 10 :=
by
  intro hx'
  have hx_sqrt : x = 18 * real.sqrt 10 := hx'
  rw [hx_sqrt, total_cost, trip_time, real.sqrt_mul_self, real.sqrt, smul_eq_mul, mul_add, div_self, mul_one]
  simp only [mul_div_cancel', div_self, ne_of_gt, lt_add_iff_pos_right, zero_lt_mul_left]
  sorry

end total_cost_expr_min_cost_l59_59817


namespace henry_friend_fireworks_l59_59612

-- Definitions of variables and conditions
variable 
  (F : ℕ) -- Number of fireworks Henry's friend bought

-- Main theorem statement
theorem henry_friend_fireworks (h1 : 6 + 2 + F = 11) : F = 3 :=
by
  sorry

end henry_friend_fireworks_l59_59612


namespace locus_of_Y_l59_59235

-- Definitions and conditions
variables {A B C X Y : Point}
variables (x y : Line)

-- Given conditions
axiom right_triangle_ABC : ∠A = 90
axiom x_perpendicular_y : ∀ p : Point, p ∈ x → p ∈ y → false
axiom A_on_x_and_y : A ∈ x ∧ A ∈ y

-- Statements about reflections
axiom yb_reflection : ∀ X : Point, X ∈ x → Reflection(y, Line_through(X, B)) = yb
axiom yc_reflection : ∀ X : Point, X ∈ x → Reflection(y, Line_through(X, C)) = yc

-- Intersection point condition
axiom yb_yc_intersect : ∀ X : Point, X ∈ x → ∃ Y : Point, Y ∈ yb ∧ Y ∈ yc

-- Theorem statement
theorem locus_of_Y :
  ∃ locus : Line, (∀ X : Point, X ∈ x →
    Y ∈ Reflection_of(BC, isotomic_image(x, ABC)) ∧
    Y ∉ {intersection(x, BC), symmetric(A, BC)}) :=
sorry

end locus_of_Y_l59_59235


namespace sum_of_first_220_terms_l59_59305

-- Definitions and conditions
noncomputable def a : ℚ := 319 / 12
noncomputable def d : ℚ := -1 / 6

def S (n : ℕ) : ℚ :=
  (n / 2) * (2 * a + (n - 1) * d)

-- Given conditions
def S_20 : ℚ := 500
def S_200 : ℚ := 2000

-- Proof goal: We want to show that S 220 = 5500 / 3
theorem sum_of_first_220_terms :
  S 20 = S_20 →
  S 200 = S_200 →
  S 220 = 5500 / 3 :=
by
  intros h20 h200
  sorry

end sum_of_first_220_terms_l59_59305


namespace total_distance_yards_remaining_yards_l59_59823

structure Distance where
  miles : Nat
  yards : Nat

def marathon_distance : Distance :=
  { miles := 26, yards := 385 }

def miles_to_yards (miles : Nat) : Nat :=
  miles * 1760

def total_yards_in_marathon (d : Distance) : Nat :=
  miles_to_yards d.miles + d.yards

def total_distance_in_yards (d : Distance) (n : Nat) : Nat :=
  n * total_yards_in_marathon d

def remaining_yards (total_yards : Nat) (yards_in_mile : Nat) : Nat :=
  total_yards % yards_in_mile

theorem total_distance_yards_remaining_yards :
    let total_yards := total_distance_in_yards marathon_distance 15
    remaining_yards total_yards 1760 = 495 :=
by
  sorry

end total_distance_yards_remaining_yards_l59_59823


namespace train_speed_l59_59156

def train_length : ℕ := 110
def bridge_length : ℕ := 265
def crossing_time : ℕ := 30

def speed_in_m_per_s (d t : ℕ) : ℕ := d / t
def speed_in_km_per_hr (s : ℕ) : ℕ := s * 36 / 10

theorem train_speed :
  speed_in_km_per_hr (speed_in_m_per_s (train_length + bridge_length) crossing_time) = 45 :=
by
  sorry

end train_speed_l59_59156


namespace range_of_dot_products_l59_59570

-- Define the variables and conditions
variables {a b k : ℝ}
variables (x y : ℝ)
variables ellipse_P ellipse_Q : Point

-- Define the ellipse equation and vertices conditions
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)

-- Given conditions
axiom major_axis_length : 2 * a = 8
axiom a_gt_b : a > b
axiom b_gt_0 : b > 0
axiom slopes_condition (T : Point) : (slope T (-a, 0) * slope T (a, 0)) = -3 / 4

-- Ellipse C defined by the determined a and b
def ellipse_C := ellipse_eq 4 (sqrt (16 / 3))

-- Define the points and required dot product range
def OP (P : Point) := sqrt (P.x^2 + P.y^2)
def OQ (Q : Point) := sqrt (Q.x^2 + Q.y^2)
def MP (P : Point) := sqrt (P.x^2 + (P.y - 2)^2)
def MQ (Q : Point) := sqrt (Q.x^2 + (Q.y - 2)^2)
def product_range := OP ellipse_P * OQ ellipse_Q + MP ellipse_P * MQ ellipse_Q

-- The final statement to prove
theorem range_of_dot_products :
  -20 ≤ product_range ∧ product_range ≤ -52 / 3 :=
sorry

end range_of_dot_products_l59_59570


namespace age_of_replaced_person_l59_59733

theorem age_of_replaced_person
    (T : ℕ) -- total age of the original group of 10 persons
    (age_person_replaced : ℕ) -- age of the person who was replaced
    (age_new_person : ℕ) -- age of the new person
    (h1 : age_new_person = 15)
    (h2 : (T / 10) - 3 = (T - age_person_replaced + age_new_person) / 10) :
    age_person_replaced = 45 :=
by
  sorry

end age_of_replaced_person_l59_59733


namespace find_f_2001_l59_59335

theorem find_f_2001 :
  ∃ f : ℕ → ℕ, (∀ n : ℕ, 0 < n → 0 < f n) ∧
               (∀ n : ℕ, 0 < n → f(n + 1) > f n) ∧
               (∀ n : ℕ, 0 < n → f(f(n)) = 3 * n) ∧
               f 2001 = 3816 :=
sorry

end find_f_2001_l59_59335


namespace yellow_balls_count_l59_59651

-- Definition of problem conditions
def initial_red_balls : ℕ := 16
def initial_blue_balls : ℕ := 2 * initial_red_balls
def red_balls_lost : ℕ := 6
def green_balls_given_away : ℕ := 7  -- This is not used in the calculations
def yellow_balls_bought : ℕ := 3 * red_balls_lost
def final_total_balls : ℕ := 74

-- Defining the total balls after all transactions
def remaining_red_balls : ℕ := initial_red_balls - red_balls_lost
def total_accounted_balls : ℕ := remaining_red_balls + initial_blue_balls + yellow_balls_bought

-- Lean statement to prove
theorem yellow_balls_count : yellow_balls_bought = 18 :=
by
  sorry

end yellow_balls_count_l59_59651


namespace solve_equation_l59_59400

theorem solve_equation : ∀ x : ℝ, (x (x - 1) = x) → (x = 0 ∨ x = 2) :=
by
  intro x
  intro h
  have h1 : x^2 - 2*x = 0 := by {
    rw [←h]
    ring
  }
  have h2 : x * (x - 2) = 0 := by {
    rw [h1]
  }
  exact sorry

end solve_equation_l59_59400


namespace log_base5_b2023_eq_11_l59_59866

noncomputable def clubsuit (a b : ℝ) : ℝ := a ^ real.logb 5 b
noncomputable def spadesuit (a b : ℝ) : ℝ := a ^ (1 / real.logb 5 b)

def seq_b : ℕ → ℝ
| 4 := spadesuit 4 3
| (n + 5) := clubsuit (spadesuit (n + 5) (n + 4)) (seq_b (n + 4))
| _ := 0 -- base cases (n < 4) which are irrelevant to this problem

theorem log_base5_b2023_eq_11 : real.logb 5 (seq_b 2023) = 11 := 
sorry

end log_base5_b2023_eq_11_l59_59866


namespace car_bus_initial_speed_l59_59445

theorem car_bus_initial_speed {d : ℝ} {t : ℝ} {s_c : ℝ} {s_b : ℝ}
    (h1 : t = 4) 
    (h2 : s_c = s_b + 8) 
    (h3 : d = 384)
    (h4 : ∀ t, 0 ≤ t → t ≤ 2 → d = s_c * t + s_b * t) 
    (h5 : ∀ t, 2 < t → t ≤ 4 → d = (s_c - 10) * (t - 2) + s_b * (t - 2)) 
    : s_b = 46.5 ∧ s_c = 54.5 := 
by 
    sorry

end car_bus_initial_speed_l59_59445


namespace definite_integral_l59_59180

theorem definite_integral (f : ℝ → ℝ) (a b : ℝ) : 
  (∫ x in a..b, f x) = (2/3) :=
begin
  assume a = 0,
  assume b = 1,
  assume f = λ x, 2 * x - x ^ 2,
  sorry
end

end definite_integral_l59_59180


namespace polynomial_degree_is_15_l59_59778

noncomputable def polynomial_degree (a b c d e f g : ℝ) : ℕ :=
nat_degree ((X^4 + C a * X^9 + C b * X + C c) * (X^3 + C d * X^5 + C e * X + C f) * (X + C g))

theorem polynomial_degree_is_15 (a b c d e f g : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) (hg : g ≠ 0) :
  polynomial_degree a b c d e f g = 15 :=
by 
  sorry

end polynomial_degree_is_15_l59_59778


namespace lattice_points_count_l59_59355

theorem lattice_points_count :
  let region (x y : ℤ) := (y ≤ 3 * x) ∧ (y ≥ x / 3) ∧ (x + y ≤ 100)
  in (finset.univ.product finset.univ).filter (λ (p : ℤ × ℤ), region p.fst p.snd).card = 2551 :=
by sorry

end lattice_points_count_l59_59355


namespace arc_length_150_deg_max_area_sector_l59_59948

noncomputable def alpha := 150 * (Real.pi / 180)
noncomputable def r := 6
noncomputable def perimeter := 24

-- 1. Proving the arc length when α = 150° and r = 6
theorem arc_length_150_deg : alpha * r = 5 * Real.pi := by
  sorry

-- 2. Proving the maximum area and corresponding alpha given the perimeter of 24
theorem max_area_sector : ∃ (α : ℝ), α = 2 ∧ (1 / 2) * ((perimeter - 2 * r) * r) = 36 := by
  sorry

end arc_length_150_deg_max_area_sector_l59_59948


namespace coefficient_of_term_in_binomial_expansion_l59_59638

theorem coefficient_of_term_in_binomial_expansion :
  let a : ℝ := sorry
  let term := (a - 1 / sqrt a)^8
  let target_term_exp := -1/2
  in ∀ k : ℕ, (8.choose k) * (-1)^k * a^(8-k-(k/2)) = a ^ target_term_exp → false :=
by
  sorry

end coefficient_of_term_in_binomial_expansion_l59_59638


namespace not_right_triangle_l59_59590

variable {α : Type*}
variables (a b c : ℝ)
variables (A B C : α)

-- Conditions
axiom condition2 : a^2 = 5 ∧ b^2 = 12 ∧ c^2 = 13

-- Function to check if a triangle given a, b, and c is a right triangle
noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

-- Theorem to prove that under given conditions, the triangle is not necessarily a right triangle
theorem not_right_triangle : ¬ is_right_triangle a b c := by
  rw [is_right_triangle]
  intro h
  cases condition2 with h₁ h₂
  cases h₂ with h₃ h₄
  rw [h₁, h₃, h₄] at h
  sorry

end not_right_triangle_l59_59590


namespace number_of_solutions_l59_59056

def system_of_equations (x y : ℝ) : Prop :=
  (x + y - 1) * (Real.sqrt (x - 1)) = 0 ∧
  x^2 + y^2 + 2 * x - 4 = 0

theorem number_of_solutions : 
  ↑(Finset.card {p : ℝ × ℝ | system_of_equations p.1 p.2}.to_finset) = 4 := 
sorry

end number_of_solutions_l59_59056


namespace percentage_of_tip_is_25_l59_59850

-- Definitions of the costs
def cost_samosas : ℕ := 3 * 2
def cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2

-- Definition of total food cost
def total_food_cost : ℕ := cost_samosas + cost_pakoras + cost_mango_lassi

-- Definition of the total meal cost including tax
def total_meal_cost_with_tax : ℕ := 25

-- Definition of the tip
def tip : ℕ := total_meal_cost_with_tax - total_food_cost

-- Definition of the percentage of the tip
def percentage_tip : ℕ := (tip * 100) / total_food_cost

-- The theorem to be proved
theorem percentage_of_tip_is_25 :
  percentage_tip = 25 :=
by
  sorry

end percentage_of_tip_is_25_l59_59850


namespace positive_difference_of_volumes_is_136_5_l59_59844

def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

def cylinder_a_radius := 6 / π
def cylinder_a_height := 9
def cylinder_a_volume := volume_of_cylinder cylinder_a_radius cylinder_a_height 

def cylinder_b_radius := 5 / π
def cylinder_b_height := 7.5
def cylinder_b_volume := volume_of_cylinder cylinder_b_radius cylinder_b_height

theorem positive_difference_of_volumes_is_136_5 :
  π * (abs (cylinder_b_volume - cylinder_a_volume)) = 136.5 :=
by
  sorry

end positive_difference_of_volumes_is_136_5_l59_59844


namespace isosceles_triangle_height_l59_59146

theorem isosceles_triangle_height (s : ℝ) (h : ℝ) 
  (pentagon_area : ℝ := (5 / 4) * s^2 * ((1 + real.sqrt 5) / 4))
  (triangle_base : ℝ := 5 * s)
  (triangle_area : ℝ := (1 / 2) * triangle_base * h) :
  pentagon_area = triangle_area → h = (s * (1 + real.sqrt 5)) / 8 :=
by sorry

end isosceles_triangle_height_l59_59146


namespace seq_general_term_l59_59288

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}

def sum_terms (n : ℕ) : ℕ := (3 / 2) * a_n n - 3

theorem seq_general_term (h : ∀ n, S_n n = sum_terms n) : 
  ∀ n, a_n n = 2 * 3^n := sorry

end seq_general_term_l59_59288


namespace sum_of_number_and_reverse_l59_59043

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
  (h5 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 99 := 
sorry

end sum_of_number_and_reverse_l59_59043


namespace find_lambdas_l59_59552

variable (e1 e2 a : ℝ × ℝ)
variable (λ1 λ2 : ℝ)

def vec1 : ℝ × ℝ := (2, 1)
def vec2 : ℝ × ℝ := (1, 3)
def veca : ℝ × ℝ := (-1, 2)

theorem find_lambdas (h1 : a = λ1 * e1 + λ2 * e2) :
  (e1, e2, a) = ((2, 1), (1, 3), (-1, 2)) →
  λ1 = -1 ∧ λ2 = 1 := by
  intro heq
  cases heq
  sorry

end find_lambdas_l59_59552


namespace point_after_transformations_l59_59076

-- Define the initial coordinates of point F
def F : ℝ × ℝ := (-1, -1)

-- Function to reflect a point over the x-axis
def reflect_over_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Function to reflect a point over the line y = x
def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Prove that F, when reflected over x-axis and then y=x, results in (1, -1)
theorem point_after_transformations : 
  reflect_over_y_eq_x (reflect_over_x F) = (1, -1) := by
  sorry

end point_after_transformations_l59_59076


namespace min_value_fx_l59_59595

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem min_value_fx : (x ∈ set.Icc (0 : ℝ) 3) → ∃ y ∈ {f x | x ∈ set.Icc (0 : ℝ) 3}, y = 2 :=
by
  sorry

end min_value_fx_l59_59595


namespace f_limit_at_neg_2_f_discontinuity_at_2_l59_59648

noncomputable def f : ℝ → ℝ
| x := if x < 2 then 
          if x = -2 then 11 / 4
          else (x ^ 2 + 7 * x + 10) / (x ^ 2 - 4)
       else 4 * x - 3

theorem f_limit_at_neg_2 : (filter.tendsto f (nhds (-2)) (nhds (-3 / 4))) :=
sorry

theorem f_discontinuity_at_2 : 
  ¬ (filter.tendsto f (nhds 2) (nhds 5)) :=
sorry

end f_limit_at_neg_2_f_discontinuity_at_2_l59_59648


namespace least_number_to_subtract_l59_59098

theorem least_number_to_subtract (x : ℕ) (h1 : 997 - x ≡ 3 [MOD 17]) (h2 : 997 - x ≡ 3 [MOD 19]) (h3 : 997 - x ≡ 3 [MOD 23]) : x = 3 :=
by
  sorry

end least_number_to_subtract_l59_59098


namespace probability_flies_swept_by_minute_hand_l59_59004

theorem probability_flies_swept_by_minute_hand :
  let flies_positions := {12, 3, 6, 9}
  -- Define the favorable starting intervals for the 20-minute sweep.
  let favorable_intervals := [(55, 60), (20, 25), (35, 40), (50, 55)]
  -- Total possible minutes in an hour
  let total_minutes := 60
  -- Total favorable minutes
  let favorable_minutes := 20
  -- Calculate the probability
  (favorable_minutes / total_minutes : ℝ) = (1 / 3 : ℝ):=
by
  sorry

end probability_flies_swept_by_minute_hand_l59_59004


namespace necessary_and_sufficient_condition_l59_59273

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

def proposition_p (a b : V) : Prop :=
  abs (inner_product a b) = norm a * norm b

def collinear (a b : V) : Prop :=
  ∃ (t : ℝ), a = t • b

theorem necessary_and_sufficient_condition (a b : V) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  (proposition_p a b ↔ collinear a b) :=
begin
  sorry,
end

end necessary_and_sufficient_condition_l59_59273


namespace num_sequences_15_l59_59616

def valid_sequences (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else valid_sequences (n - 3) + 2 * valid_sequences (n - 4) + valid_sequences (n - 5)

theorem num_sequences_15 : valid_sequences 15 = 50 :=
  sorry

end num_sequences_15_l59_59616


namespace rectangle_dimensions_l59_59893

theorem rectangle_dimensions (x : ℝ) (h : 3 * x * x = 8 * x) : (x = 8 / 3 ∧ 3 * x = 8) :=
by {
  sorry
}

end rectangle_dimensions_l59_59893


namespace triangle_angles_sum_l59_59476

theorem triangle_angles_sum (x : ℝ) (h : 40 + 3 * x + (x + 10) = 180) : x = 32.5 := by
  sorry

end triangle_angles_sum_l59_59476


namespace find_numbers_l59_59760

theorem find_numbers (A B: ℕ) (h1: A + B = 581) (h2: (Nat.lcm A B) / (Nat.gcd A B) = 240) : 
  (A = 560 ∧ B = 21) ∨ (A = 21 ∧ B = 560) :=
by
  sorry

end find_numbers_l59_59760


namespace worksheets_graded_l59_59840

theorem worksheets_graded (w : ℕ) (h1 : ∀ (n : ℕ), n = 3) (h2 : ∀ (n : ℕ), n = 15) (h3 : ∀ (p : ℕ), p = 24)  :
  w = 7 :=
sorry

end worksheets_graded_l59_59840


namespace total_prime_factors_l59_59793

theorem total_prime_factors (a b c : ℕ) (h₁ : a = 4) (h₂ : b = 7) (h₃ : c = 11) : 
  let exp := (a ^ 11) * (b ^ 5) * (c ^ 2)
  in (prime_factors a 11) + (prime_factors b 5) + (prime_factors c 2) = 29 :=
by 
  sorry

noncomputable def prime_factors (n : ℕ) (e : ℕ) : ℕ :=
  match n with
  | 4 => 2 * e
  | 7 => e
  | 11 => e
  | _ => 0

end total_prime_factors_l59_59793


namespace sec_of_negative_420_eq_2_l59_59885

theorem sec_of_negative_420_eq_2 :
  sec (-420 : ℝ) = 2 :=
by
  -- Definitions and conditions
  let sec (θ : ℝ) := 1 / cos θ
  have h_cos_period : ∀ θ k, cos (θ + 360 * k) = cos θ,
  from sorry
  have h_cos_60 : cos 60 = 1 / 2,
  from sorry

  -- Proof starts (skipping the actual steps)
  sorry

end sec_of_negative_420_eq_2_l59_59885


namespace smallest_j_2016_l59_59214

def smallest_j_divisible_by_2016 : ℕ :=
  ∃ j : ℕ,
    (∀ p k, is_poly_with_integer_coeffs p → is_integer k →
      divisible_by_2016 (derivative_j_k p j k)) ∧
    minimal_j j 2016

noncomputable def is_poly_with_integer_coeffs (p : Polynomial ℤ) : Prop := sorry
noncomputable def is_integer (k : ℤ) : Prop := true
noncomputable def derivative_j_k (p : Polynomial ℤ) (j : ℕ) (k : ℤ) : ℤ := sorry
noncomputable def divisible_by_2016 (n : ℤ) : Prop := 2016 ∣ n
noncomputable def minimal_j (j : ℕ) (n : ℤ) : Prop := sorry

theorem smallest_j_2016 : smallest_j_divisible_by_2016 = 8 :=
by sorry

end smallest_j_2016_l59_59214


namespace expression_equals_five_l59_59909

-- Define the innermost expression
def inner_expr : ℤ := -|( -2 + 3 )|

-- Define the next layer of the expression
def middle_expr : ℤ := |(inner_expr) - 2|

-- Define the outer expression
def outer_expr : ℤ := |middle_expr + 2|

-- The proof problem statement (in this case, without the proof)
theorem expression_equals_five : outer_expr = 5 :=
by
  -- Insert precise conditions directly from the problem statement
  have h_inner : inner_expr = -|1| := by sorry
  have h_middle : middle_expr = |-1 - 2| := by sorry
  have h_outer : outer_expr = |(-3) + 2| := by sorry
  -- The final goal that needs to be proven
  sorry

end expression_equals_five_l59_59909


namespace find_f_of_1_div_8_l59_59745

noncomputable def f (x : ℝ) (a : ℝ) := (a^2 + a - 5) * Real.logb a x

theorem find_f_of_1_div_8 (a : ℝ) (hx1 : x = 1 / 8) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^2 + a - 5 = 1) :
  f x a = -3 :=
by
  sorry

end find_f_of_1_div_8_l59_59745


namespace find_lambda_l59_59589

-- Defining the normal vectors
def normal_vector_alpha : (ℝ × ℝ × ℝ) := (2, 3, -1)
def normal_vector_beta (λ : ℝ) : (ℝ × ℝ × ℝ) := (4, λ, -2)

-- Defining the perpendicularity condition as the dot product being zero
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_lambda (λ : ℝ) : dot_product normal_vector_alpha (normal_vector_beta λ) = 0 → λ = -10/3 :=
by
  intro h
  sorry

end find_lambda_l59_59589


namespace largest_angle_isosceles_l59_59307

theorem largest_angle_isosceles (α : ℝ) (h1 : 0 < α ∧ α < 90) :
  let β := 2 * α in
  let γ := 180 - 3 * α in
  γ <= 90 ∧ β > γ := 
begin
  sorry
end

end largest_angle_isosceles_l59_59307


namespace geom_mean_of_all_non_empty_subsets_eq_geom_mean_l59_59391

open Finset

-- Define the geometric mean of a set of positive reals
def geom_mean (s : Finset ℝ) (h : ∀ x ∈ s, 0 < x) : ℝ :=
  (s.prod id) ^ (1 / s.card.to_real)

theorem geom_mean_of_all_non_empty_subsets_eq_geom_mean 
  (S : Finset ℝ) (hS : ∀ x ∈ S, 0 < x) (n : ℕ) (hn : S.card = n) (h_pos : 0 < n) :
  geom_mean S hS = geom_mean (S.powerset.erase ∅) sorry :=
sorry

end geom_mean_of_all_non_empty_subsets_eq_geom_mean_l59_59391


namespace eccentricity_of_conic_section_l59_59016

-- Definitions and conditions
def is_conic_section (P F1 F2 : ℝ) : Prop := sorry -- Placeholder for the conic section definition
def are_foci (F1 F2 : ℝ) : Prop := sorry -- Placeholder for the foci definition
def distance_ratio (P F1 F2 : ℝ) : Prop :=
  dist P F1 / dist F1 F2 = 4 / 3 ∧ dist P F2 / dist F1 F2 = 2 / 3

-- Theorem statement
theorem eccentricity_of_conic_section (P F1 F2 : ℝ) 
  (h_conic : is_conic_section P F1 F2)
  (h_foci : are_foci F1 F2)
  (h_ratio : distance_ratio P F1 F2) :
  eccentricity P F1 F2 = 1 / 2 ∨ eccentricity P F1 F2 = 3 / 2 := sorry

end eccentricity_of_conic_section_l59_59016


namespace relationship_points_l59_59991

noncomputable def is_on_inverse_proportion (m x y : ℝ) : Prop :=
  y = (-m^2 - 2) / x

theorem relationship_points (a b c m : ℝ) :
  is_on_inverse_proportion m a (-1) ∧
  is_on_inverse_proportion m b 2 ∧
  is_on_inverse_proportion m c 3 →
  a > c ∧ c > b :=
by
  sorry

end relationship_points_l59_59991


namespace no_valid_partition_exists_l59_59662

namespace MathProof

-- Define the set of positive integers
def N := {n : ℕ // n > 0}

-- Define non-empty sets A, B, C which are disjoint and partition N
def valid_partition (A B C : N → Prop) : Prop :=
  (∃ a, A a) ∧ (∃ b, B b) ∧ (∃ c, C c) ∧
  (∀ n, A n → ¬ B n ∧ ¬ C n) ∧
  (∀ n, B n → ¬ A n ∧ ¬ C n) ∧
  (∀ n, C n → ¬ A n ∧ ¬ B n) ∧
  (∀ n, A n ∨ B n ∨ C n)

-- Define the conditions in the problem
def condition_1 (A B C : N → Prop) : Prop :=
  ∀ a b, A a → B b → C ⟨a.val + b.val + 1, by linarith [a.prop, b.prop]⟩

def condition_2 (A B C : N → Prop) : Prop :=
  ∀ b c, B b → C c → A ⟨b.val + c.val + 1, by linarith [b.prop, c.prop]⟩

def condition_3 (A B C : N → Prop) : Prop :=
  ∀ c a, C c → A a → B ⟨c.val + a.val + 1, by linarith [c.prop, a.prop]⟩

-- State the problem that no valid partition exists
theorem no_valid_partition_exists :
  ¬ ∃ (A B C : N → Prop), valid_partition A B C ∧
    condition_1 A B C ∧
    condition_2 A B C ∧
    condition_3 A B C :=
by
  sorry

end MathProof

end no_valid_partition_exists_l59_59662


namespace find_XW_in_triangle_l59_59644

-- Define the conditions
variables (XY XZ XW YW ZW : ℝ) (h : ℝ) 

-- Define the problem setup
theorem find_XW_in_triangle 
    (H1 : XY = 13) 
    (H2 : XZ = 20) 
    (H3 : YW/Zw = 3/4) 
    (H4 : h = XW) 
    : XW = 8 * Real.sqrt 2 := 
by 
    -- Use the given conditions
    have H5 : YW^2 = 169 - h^2 := by sorry 
    have H6 : ZW^2 = 400 - h^2 := by sorry 
    have H7 : (169 - h^2) / (400 - h^2) = 9 / 16 := by sorry 
    -- Solve for h^2
    have H8 : 16*(169 - h^2) = 9*(400 - h)^2 := by sorry 
    have H9 : 7 * h^2 = 896 := by sorry 
    have H10 : h^2 = 128 := by sorry 
    have H11 : h = Real.sqrt 128 := by sorry 
    -- Find the final answer
    exact XW = 8 * Real.sqrt(2)

end find_XW_in_triangle_l59_59644


namespace function_value_l59_59960

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then Real.sin (π * x / 2) else 1 / 6 - Real.log x / Real.log 3

theorem function_value :
  f (f (3 * Real.sqrt 3)) = - Real.sqrt 3 / 2 :=
by sorry

end function_value_l59_59960


namespace liam_can_see_maya_l59_59342

noncomputable def liam_visibility_time 
  (initial_distance : ℕ) 
  (distance_behind : ℕ) 
  (liam_speed : ℕ) 
  (maya_speed : ℕ)
  (relative_speed := liam_speed + maya_speed) : ℕ := 
  (initial_distance / relative_speed) + (distance_behind / relative_speed)

theorem liam_can_see_maya 
  : liam_visibility_time 600 600 9 3 = 100 :=
by
  simp [liam_visibility_time, Nat.div]
  sorry

end liam_can_see_maya_l59_59342


namespace find_x_l59_59312

variables {P : Type} [Affine P]
variables (A D G E F B C : P)
variables (area : P → P → P → ℝ)

def midpoint (E : P) (B C : P) : Prop := (E = (B + C) / 2)

def area_triang (F E C : P) (a : ℝ) : Prop := area F E C = a

def area_quad (D B E G : P) (a : ℝ) : Prop := area D B E + area B E G  + area E G D + area G D B = a

def same_area (A D G E F : P) (x : ℝ) : Prop := (area A D G = x) ∧ (area G E F = x)

theorem find_x
    (h_midpoint : midpoint E B C)
    (h_area_FEC : area_triang F E C 7)
    (h_area_DBEG : area_quad D B E G 27)
    (h_same_area : same_area A D G E F x) :
    x = 8 := 
sorry

end find_x_l59_59312


namespace find_a5_l59_59973

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n > 1, a n = (a (n - 1))^2 - 1

theorem find_a5 (a : ℕ → ℤ) (h : sequence a) : a 5 = -1 :=
by
  sorry

end find_a5_l59_59973


namespace largest_integer_n_neg_l59_59529

theorem largest_integer_n_neg (n : ℤ) : (n < 8 ∧ 3 < n) ∧ (n^2 - 11 * n + 24 < 0) → n ≤ 7 := by
  sorry

end largest_integer_n_neg_l59_59529


namespace fraction_day_crew_loaded_l59_59110

variable (D W : ℕ)  -- D: Number of boxes loaded by each worker on the day crew, W: Number of workers on the day crew

-- Condition 1: Each worker on the night crew loaded 3/4 as many boxes as each worker on the day crew
def boxes_loaded_night_worker : ℕ := 3 * D / 4
-- Condition 2: The night crew has 5/6 as many workers as the day crew
def workers_night : ℕ := 5 * W / 6

-- Question: Fraction of all the boxes loaded by the day crew
theorem fraction_day_crew_loaded :
  (D * W : ℚ) / ((D * W) + (3 * D / 4) * (5 * W / 6)) = (8 / 13) := by
  sorry

end fraction_day_crew_loaded_l59_59110


namespace Somin_solved_most_problems_l59_59375

theorem Somin_solved_most_problems:
  (∀ (total : ℚ), 
      (Suhyeon_remaining : ℚ := total * 1 / 4) → 
      (Somin_remaining : ℚ := total * 1 / 8) → 
      (Jisoo_remaining : ℚ := total * 1 / 5),
    (Somin_remaining <= Suhyeon_remaining) ∧ (Somin_remaining <= Jisoo_remaining)) := 
by 
  sorry

end Somin_solved_most_problems_l59_59375


namespace problem_statement_l59_59102

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c 

def f_A (x : ℝ) : ℝ := -2 / x^2 - 3 * x
def f_B (x : ℝ) : ℝ := -(x - 1)^2 + x^2
def f_C (x : ℝ) : ℝ := 11 * x^2 + 29 * x

theorem problem_statement : is_quadratic f_C ∧ ¬ is_quadratic f_A ∧ ¬ is_quadratic f_B := by
  sorry

end problem_statement_l59_59102


namespace total_lunch_cost_l59_59841

theorem total_lunch_cost
  (jose_lunch : ℚ)
  (rick_lunch : ℚ)
  (sophia_lunch : ℚ)
  (adam_lunch : ℚ)
  (emma_lunch : ℚ)
  (before_tax : ℚ)
  (taxes : ℚ)
  (total_cost : ℚ)
  (jose_lunch_eq : jose_lunch = 60)
  (rick_lunch_eq : rick_lunch = jose_lunch / 1.5)
  (sophia_lunch_eq : sophia_lunch = rick_lunch)
  (adam_lunch_eq : adam_lunch = (2 / 3) * rick_lunch)
  (emma_lunch_eq : emma_lunch = jose_lunch * (1 - 0.2))
  (before_tax_eq : before_tax = adam_lunch + rick_lunch + jose_lunch + sophia_lunch + emma_lunch)
  (tax_eq : taxes = 0.08 * before_tax)
  (total_cost_eq : total_cost = before_tax + taxes) :
  total_cost = 231.84 := sorry

end total_lunch_cost_l59_59841


namespace bonnets_per_orphanage_correct_l59_59349

-- Definitions for each day's bonnet count
def monday_bonnets := 10
def tuesday_and_wednesday_bonnets := 2 * monday_bonnets
def thursday_bonnets := monday_bonnets + 5
def friday_bonnets := thursday_bonnets - 5
def saturday_bonnets := friday_bonnets - 8
def sunday_bonnets := 3 * saturday_bonnets

-- Total bonnets made in the week
def total_bonnets := 
  monday_bonnets +
  tuesday_and_wednesday_bonnets +
  thursday_bonnets +
  friday_bonnets +
  saturday_bonnets +
  sunday_bonnets

-- The number of orphanages
def orphanages := 10

-- Bonnets sent to each orphanage
def bonnets_per_orphanage := total_bonnets / orphanages

theorem bonnets_per_orphanage_correct :
  bonnets_per_orphanage = 6 :=
by
  sorry

end bonnets_per_orphanage_correct_l59_59349


namespace alissa_presents_l59_59515

def ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0

theorem alissa_presents : ethan_presents - difference = 9.0 := by sorry

end alissa_presents_l59_59515


namespace man_work_days_l59_59460

variable (W : ℝ) -- Denoting the amount of work by W

-- Defining the work rate variables
variables (M Wm B : ℝ)

-- Conditions from the problem:
-- Combined work rate of man, woman, and boy together completes the work in 3 days
axiom combined_work_rate : M + Wm + B = W / 3
-- Woman completes the work alone in 18 days
axiom woman_work_rate : Wm = W / 18
-- Boy completes the work alone in 9 days
axiom boy_work_rate : B = W / 9

-- The goal is to prove the man takes 6 days to complete the work alone
theorem man_work_days : (W / M) = 6 :=
by
  sorry

end man_work_days_l59_59460


namespace perp_of_condition_l59_59606

open_locale classical
noncomputable theory

-- Define the relevant points and conditions
variables {P Q M N : Type}
variables (dist : P → P → ℝ)
variables [metric_space P] [normed_group P] [normed_space ℝ P]

-- Define the equality condition as per the problem
def condition (P Q M N : P) : Prop :=
dist P M ^ 2 - dist P N ^ 2 = dist Q M ^ 2 - dist Q N ^ 2

-- The main theorem to prove the perpendicularity
theorem perp_of_condition (P Q M N : P) (h : condition P Q M N) : 
  ∃ E F, midpoint E F = ⟨M, N⟩ ∧ ∃ R S, midpoint R S = ⟨P, Q⟩ ∧ ⟨M, N⟩ ⊥ ⟨P, Q⟩ :=
sorry

end perp_of_condition_l59_59606


namespace normal_prob_ineq_l59_59687

noncomputable def X : ℝ → ℝ := sorry -- Define the random variable X following the normal distribution
noncomputable def normal_dist (μ σ : ℝ) : Measure (ℝ) := sorry -- Define the normal distribution measure

theorem normal_prob_ineq {X : ℝ → ℝ} {μ σ : ℝ} 
  (hX : X ~ normal_dist μ σ) 
  (hμ : μ = 100) 
  (hσ : σ > 0)
  (h_prob : ProbabilityTheory.probability (set.Ioc 80 120) = 3 / 4) :
  ProbabilityTheory.probability (set.Ioi 120) = 1 / 8 :=
sorry -- proof omitted

end normal_prob_ineq_l59_59687


namespace tangent_line_area_l59_59591

theorem tangent_line_area (a : ℝ) (h₁ : a > 0) (h₂ : (1/4) = (1/2) * (√a / 2 * -a)) : a = 1 := by
  sorry

end tangent_line_area_l59_59591


namespace sum_of_first_ten_excellent_numbers_l59_59143

def is_excellent (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p^4

theorem sum_of_first_ten_excellent_numbers :
  (∑ i in [16, 81, 625, 2401, 14641, 28561, 83521, 130321, 279841, 707281], i) = 1231378 :=
by {
  sorry
}

end sum_of_first_ten_excellent_numbers_l59_59143


namespace minimal_erasure_l59_59117

noncomputable def min_factors_to_erase : ℕ :=
  2016

theorem minimal_erasure:
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = g x) → 
    (∃ f' g' : ℝ → ℝ, (∀ x, f x ≠ g x) ∧ 
      ((∃ s : Finset ℕ, s.card = min_factors_to_erase ∧ (∀ i ∈ s, f' x = (x - i) * f x)) ∧ 
      (∃ t : Finset ℕ, t.card = min_factors_to_erase ∧ (∀ i ∈ t, g' x = (x - i) * g x)))) :=
by
  sorry

end minimal_erasure_l59_59117


namespace dana_total_earnings_weekend_l59_59864

def hourly_rate : ℝ := 13
def commission_rate : ℝ := 0.05

def friday_hours : ℝ := 9
def friday_sales : ℝ := 800

def saturday_hours : ℝ := 10
def saturday_sales : ℝ := 1000

def sunday_hours : ℝ := 3
def sunday_sales : ℝ := 300

def total_earnings (hourly_rate : ℝ) (commission_rate : ℝ)
  (hours_and_sales : List (ℝ × ℝ)) : ℝ :=
hours_and_sales.sum (λ ⟨hours, sales⟩, hourly_rate * hours + commission_rate * sales)

theorem dana_total_earnings_weekend :
total_earnings hourly_rate commission_rate
  [(friday_hours, friday_sales), (saturday_hours, saturday_sales), (sunday_hours, sunday_sales)] = 391 := 
sorry

end dana_total_earnings_weekend_l59_59864


namespace expression_equals_five_l59_59910

-- Define the innermost expression
def inner_expr : ℤ := -|( -2 + 3 )|

-- Define the next layer of the expression
def middle_expr : ℤ := |(inner_expr) - 2|

-- Define the outer expression
def outer_expr : ℤ := |middle_expr + 2|

-- The proof problem statement (in this case, without the proof)
theorem expression_equals_five : outer_expr = 5 :=
by
  -- Insert precise conditions directly from the problem statement
  have h_inner : inner_expr = -|1| := by sorry
  have h_middle : middle_expr = |-1 - 2| := by sorry
  have h_outer : outer_expr = |(-3) + 2| := by sorry
  -- The final goal that needs to be proven
  sorry

end expression_equals_five_l59_59910


namespace find_k_l59_59052

theorem find_k (k : ℚ) (x : ℚ) : (y = k * x + 3) ∧ (y = 0) ∧ (|x| = 6) → k = 1/2 ∨ k = -1/2 :=
by
  assume h
  sorry

end find_k_l59_59052


namespace tamtam_blue_shells_l59_59379

theorem tamtam_blue_shells 
  (total_shells : ℕ)
  (purple_shells : ℕ)
  (pink_shells : ℕ)
  (yellow_shells : ℕ)
  (orange_shells : ℕ)
  (H_total : total_shells = 65)
  (H_purple : purple_shells = 13)
  (H_pink : pink_shells = 8)
  (H_yellow : yellow_shells = 18)
  (H_orange : orange_shells = 14) :
  ∃ blue_shells : ℕ, blue_shells = 12 :=
by
  sorry

end tamtam_blue_shells_l59_59379


namespace variance_transformation_l59_59955

-- Definitions
def ξ_distribution (ξ : ℕ → ℝ) (a : ℝ) : Prop :=
ξ 0 = a ∧ ξ 1 = 1 - 2 * a ∧ ξ 2 = 1 / 4

def a_equation (a : ℝ) : Prop :=
a + (1 - 2 * a) + 1 / 4 = 1

def expected_value (ξ : ℕ → ℝ) (a : ℝ) : ℝ :=
0 * ξ 0 + 1 * ξ 1 + 2 * ξ 2

def variance (ξ : ℕ → ℝ) (a : ℝ) : ℝ :=
ξ 0 * (0 - expected_value ξ a) ^ 2 + ξ 1 * (1 - expected_value ξ a) ^ 2 + ξ 2 * (2 - expected_value ξ a) ^ 2

-- Proof statement
theorem variance_transformation (ξ : ℕ → ℝ) (a : ℝ) (h_dist : ξ_distribution ξ a) (h_eq : a_equation(a)) :
  4 * variance ξ a = 2 := 
by
  sorry

end variance_transformation_l59_59955


namespace find_n_l59_59984

theorem find_n (n : ℕ) (h : 4 ^ 6 = 8 ^ n) : n = 4 :=
by
  sorry

end find_n_l59_59984


namespace probability_two_flies_swept_away_l59_59014

-- Defining the initial conditions: flies at 12, 3, 6, and 9 o'clock positions
def flies_positions : List ℕ := [12, 3, 6, 9]

-- The problem statement
theorem probability_two_flies_swept_away : 
  (let favorable_intervals := 20 in
   let total_intervals := 60 in
   favorable_intervals / total_intervals = 1 / 3) :=
by
  sorry

end probability_two_flies_swept_away_l59_59014


namespace angle_between_hour_and_minute_hand_at_3_40_l59_59170

def angle_between_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (360 / 60) * minute
  let hour_angle := (360 / 12) + (30 / 60) * minute
  abs (minute_angle - hour_angle)

theorem angle_between_hour_and_minute_hand_at_3_40 : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_between_hour_and_minute_hand_at_3_40_l59_59170


namespace not_divisible_by_p2_divisible_by_qr_l59_59336

namespace MyProofs

variables {b c p q n : ℤ} {r : ℕ}

def f (x : ℤ) : ℤ := (x + b) ^ 2 + c

theorem not_divisible_by_p2 (hp1 : p.prime) (hp2 : p ∣ c) (hp3 : ¬ p ^ 2 ∣ c) (n : ℤ) : ¬ p ^ 2 ∣ f n := 
sorry

theorem divisible_by_qr (hq1 : q.prime) (hq2 : odd q) (hq3 : ¬ q ∣ c) (hq4 : ∃ n, q ∣ f n) (r : ℕ) : 
  ∃ N : ℤ, q^r ∣ f N :=
sorry

end MyProofs

end not_divisible_by_p2_divisible_by_qr_l59_59336


namespace greatest_natural_number_perf_square_l59_59898

theorem greatest_natural_number_perf_square :
  ∃ n : ℕ, n ≤ 2023 ∧ 
  (∑ i in finset.range(n+1), i^2) * 
  (∑ i in finset.range(n+1, 2*n+1), i^2) ∈ ℕ ∧
  (∑ i in finset.range(n+1), i^2) * 
  (∑ i in finset.range(n+1, 2*n+1), i^2) = (1921 * (1921 + 1) * (2*1921 + 1) / 6) * 
  (1921 * (2*1921 + 1) * (7*1921 + 1) / 6) :=
sorry

end greatest_natural_number_perf_square_l59_59898


namespace proof_correct_judgments_l59_59958

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

end proof_correct_judgments_l59_59958


namespace inverse_of_B_equation_of_line_l59_59937

open Matrix

variable (x y : ℝ)

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]

def B : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![0, 1]]

def B_inv : Matrix (Fin 2) (Fin 2) ℝ := ![![1, -2], ![0, 1]]

def l' (x y : ℝ) := x + y - 2 = 0

def AB_inv := A ⬝ B_inv

def transform (p : Fin 2 → ℝ) := AB_inv ⬝ p

theorem inverse_of_B :
  B ⬝ B_inv = 1 :=
by {
  -- Proof should be provided here
  sorry
}

theorem equation_of_line :
  ∀ (p : Fin 2 → ℝ), l' (transform p 0) (transform p 1) ↔ p 0 = 2 :=
by {
  -- Proof should be provided here
  sorry
}

end inverse_of_B_equation_of_line_l59_59937


namespace quarters_to_half_dollars_diff_l59_59160

theorem quarters_to_half_dollars_diff (p : ℚ) :
  let alice_quarters := 8 * p + 2
  let bob_quarters := 3 * p + 6
  let quarter_to_half_dollar := 0.5
  (alice_quarters - bob_quarters) * quarter_to_half_dollar = 2.5 * p - 2 :=
by
  let alice_quarters := 8 * p + 2
  let bob_quarters := 3 * p + 6
  let quarter_to_half_dollar := 0.5
  calc
    (alice_quarters - bob_quarters) * quarter_to_half_dollar
        = ((8 * p + 2) - (3 * p + 6)) * quarter_to_half_dollar : by sorry
    ... = (5 * p - 4) * quarter_to_half_dollar : by sorry
    ... = 2.5 * p - 2 : by sorry

end quarters_to_half_dollars_diff_l59_59160


namespace terminal_side_of_610_deg_is_250_deg_l59_59032

theorem terminal_side_of_610_deg_is_250_deg:
  ∃ k : ℤ, 610 % 360 = 250 := by
  sorry

end terminal_side_of_610_deg_is_250_deg_l59_59032


namespace max_square_area_of_fencing_l59_59231

theorem max_square_area_of_fencing (f : ℝ) (h : f = 36) : 
  let s := f / 4 in 
  s * s = 81 :=
by
  have h1 : s = 36 / 4 := by sorry
  have h2 : s * s = 81 := by sorry
  exact h2

end max_square_area_of_fencing_l59_59231


namespace probability_all_evns_before_odd_l59_59138

/-- 
A fair 8-sided die (each side numbered from 1 to 8) is rolled repeatedly until an odd number appears.
This theorem states that the probability that every even number (2, 4, 6, 8) appears at least once 
before the first occurrence of any odd number is 1/384. 
-/
theorem probability_all_evns_before_odd : 
  let evens := {2, 4, 6, 8},
      die_sides := {1, 2, 3, 4, 5, 6, 7, 8} in
  probability (evens_before_first_odd die_sides evens) = 1 / 384 :=
sorry

end probability_all_evns_before_odd_l59_59138


namespace distance_from_point_D_to_plane_l59_59527

-- Define points A, B, C, D as given in the problem.
def A : ℝ × ℝ × ℝ := (-3, 0, 1)
def B : ℝ × ℝ × ℝ := (2, 1, -1)
def C : ℝ × ℝ × ℝ := (-2, 2, 0)
def D : ℝ × ℝ × ℝ := (1, 3, 2)

-- Define a function to calculate the distance from a point to a plane.
def point_to_plane_distance (p : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (d : ℝ) : ℝ :=
  let (px, py, pz) := p in
  let (nx, ny, nz) := n in
  abs (nx * px + ny * py + nz * pz + d) / real.sqrt (nx^2 + ny^2 + nz^2)

-- Define the vectors AB, AC.
def AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def AC : ℝ × ℝ × ℝ := (C.1 - A.1, C.2 - A.2, C.3 - A.3)

-- Define the normal vector n which is perpendicular to both AB and AC (known solution).
def n : ℝ × ℝ × ℝ := (1, 1, 3)

-- Define the equation of the plane formed by point A and normal vector n, the constant term is zero in this case.
def d : ℝ := 0

-- State the final theorem to prove
theorem distance_from_point_D_to_plane :
  point_to_plane_distance D n d = 10 / real.sqrt 11 :=
sorry

end distance_from_point_D_to_plane_l59_59527


namespace greatest_consecutive_irreducible_five_digit_numbers_l59_59453

def is_irreducible_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ ∀ a b : ℕ, 100 ≤ a ∧ a ≤ 999 → 100 ≤ b ∧ b ≤ 999 → n ≠ a * b

theorem greatest_consecutive_irreducible_five_digit_numbers : 
  ∀ s : List ℕ, (∀ n ∈ s, is_irreducible_five_digit n) → 
  (∀ m, m ∈ s → (m + 1) ∈ s ∨ (m = s.getLast s.head)) →
  s.length ≤ 99 :=
by
  sorry

end greatest_consecutive_irreducible_five_digit_numbers_l59_59453


namespace fiona_pages_eq_587_l59_59403

theorem fiona_pages_eq_587 :
  (∃ p : ℝ, FionaReadsPages(p) ∧ p ≈ 587) :=
by
  let FionaReadSpeed := 40 -- Fiona reads a page in 40 seconds
  let DavidReadSpeed := 75 -- David reads a page in 75 seconds
  let TotalPages := 900 -- Total number of pages
  let p := TotalPages * DavidReadSpeed / (FionaReadSpeed + DavidReadSpeed) -- Solve for p
  exact ⟨p, sorry⟩

end fiona_pages_eq_587_l59_59403


namespace triangle_base_is_8_l59_59628

/- Problem Statement:
We have a square with a perimeter of 48 and a triangle with a height of 36.
We need to prove that if both the square and the triangle have the same area, then the base of the triangle (x) is 8.
-/

theorem triangle_base_is_8
  (square_perimeter : ℝ)
  (triangle_height : ℝ)
  (same_area : ℝ) :
  square_perimeter = 48 →
  triangle_height = 36 →
  same_area = (square_perimeter / 4) ^ 2 →
  same_area = (1 / 2) * x * triangle_height →
  x = 8 :=
by
  sorry

end triangle_base_is_8_l59_59628


namespace complex_identity_l59_59683

theorem complex_identity (z : ℂ) (h : z = 1 - I) : (1 / z + conj z = 3 / 2 + 3 / 2 * I) :=
by
  sorry

end complex_identity_l59_59683


namespace no_fixed_point_implies_no_double_fixed_point_l59_59429

theorem no_fixed_point_implies_no_double_fixed_point (f : ℝ → ℝ) 
  (hf : Continuous f)
  (h : ∀ x : ℝ, f x ≠ x) :
  ∀ x : ℝ, f (f x) ≠ x :=
sorry

end no_fixed_point_implies_no_double_fixed_point_l59_59429


namespace sin_750_eq_one_half_l59_59510

theorem sin_750_eq_one_half :
  sin (750 * real.pi / 180) = 1 / 2 := 
sorry

end sin_750_eq_one_half_l59_59510


namespace probability_left_oar_works_l59_59130

structure Oars where
  P_L : ℝ -- Probability that the left oar works
  P_R : ℝ -- Probability that the right oar works
  
def independent_prob (o : Oars) : Prop :=
  o.P_L = o.P_R ∧ (1 - o.P_L) * (1 - o.P_R) = 0.16

theorem probability_left_oar_works (o : Oars) (h1 : independent_prob o) (h2 : 1 - (1 - o.P_L) * (1 - o.P_R) = 0.84) : o.P_L = 0.6 :=
by
  sorry

end probability_left_oar_works_l59_59130


namespace hamiltonian_cycle_exists_l59_59327

theorem hamiltonian_cycle_exists {G : Graph V} (n : ℕ) (hG : G.is_connected) 
  (hdeg : ∀ v : G.V, G.degree v ≥ n / 2) (hn : G.verts.card = n) : 
  G.is_hamiltonian :=
begin
  sorry
end

end hamiltonian_cycle_exists_l59_59327


namespace two_digit_integer_eq_55_l59_59282

theorem two_digit_integer_eq_55
  (c : ℕ)
  (h1 : c / 10 + c % 10 = 10)
  (h2 : (c / 10) * (c % 10) = 25) :
  c = 55 :=
  sorry

end two_digit_integer_eq_55_l59_59282


namespace enrollment_difference_l59_59775

theorem enrollment_difference (varsity northwest central greenbriar : ℕ) 
  (h_varsity : varsity = 1250) 
  (h_northwest : northwest = 1430) 
  (h_central : central = 1900) 
  (h_greenbriar : greenbriar = 1720) : 
  (greenbriar - northwest = 290) :=
by 
  have h_second_highest := h_greenbriar,
  have h_second_lowest := h_northwest,
  calc
    greenbriar - northwest = 1720 - 1430 : by rw [h_second_highest, h_second_lowest]
                        ... = 290       : by norm_num
  sorry

end enrollment_difference_l59_59775


namespace proof_number_of_solutions_l59_59212

open Complex

noncomputable def num_solutions : Nat :=
  Set.card { p : ℂ × ℂ | let a := p.1; let b := p.2 in a^4 * b^7 = 1 ∧ a^8 * b^3 = 1 }

theorem proof_number_of_solutions : num_solutions = 16 :=
  sorry

end proof_number_of_solutions_l59_59212


namespace angle_QNR_l59_59077

theorem angle_QNR {P Q R N : Type} 
  (hPQR_isosceles : ∃ (PR QR : Real), PR = QR)
  (hAngle_PRQ : ∠ P R Q = 108)
  (hN_inter : ∃ (interior : Set (Triangle P Q R)), N ∈ interior)
  (hAngle_PNR : ∠ P N R = 9)
  (hAngle_PRN : ∠ P R N = 21) :
  ∠ Q N R = 165 := by
  sorry

end angle_QNR_l59_59077


namespace marion_paperclips_correct_l59_59690

def yun_initial_paperclips := 30
def yun_remaining_paperclips (x : ℕ) : ℕ := (2 * x) / 5
def marion_paperclips (x y : ℕ) : ℕ := (4 * (yun_remaining_paperclips x)) / 3 + y
def y := 7

theorem marion_paperclips_correct : marion_paperclips yun_initial_paperclips y = 23 := by
  sorry

end marion_paperclips_correct_l59_59690


namespace difference_smallest_integers_l59_59048

theorem difference_smallest_integers (m := Nat.lcm 2 3 4 5 6 7 8 9 10 11) 
  (k : Nat) (hk : 2 ≤ k ∧ k ≤ 11) :
  (29 * 13) % k = 1 → m = 27720 :=
by 
  sorry

end difference_smallest_integers_l59_59048


namespace number_of_lines_through_point_with_equal_intercepts_l59_59211

/-- Proposition:
There are exactly 2 lines that pass through the point (1, 2) and have equal intercepts on the coordinate axes.
-/
theorem number_of_lines_through_point_with_equal_intercepts : 
  ∃ (l1 l2 : ℝ → ℝ), 
  (l1 1 = 2 ∧ l2 1 = 2) ∧ (exists b1 b2 : ℝ, (b1 ≠ b2) ∧ 
  (l1 = λ x, x + 1) ∧ (l2 = λ x, -x + 3)) := 
by
  sorry

end number_of_lines_through_point_with_equal_intercepts_l59_59211


namespace final_score_l59_59874

def dart1 : ℕ := 50
def dart2 : ℕ := 0
def dart3 : ℕ := dart1 / 2

theorem final_score : dart1 + dart2 + dart3 = 75 := by
  sorry

end final_score_l59_59874


namespace triathlete_average_speed_l59_59158

-- Definitions of the conditions
def swimming_distance := 5 -- distance in km
def biking_distance := 30 -- distance in km
def running_distance := 15 -- distance in km

def swimming_speed := 3 -- speed in km/h
def biking_speed := 25 -- speed in km/h
def running_speed := 8 -- speed in km/h

-- Calculate the times for each segment
def swimming_time := swimming_distance / swimming_speed
def biking_time := biking_distance / biking_speed
def running_time := running_distance / running_speed

-- Calculate the total time
def total_time := swimming_time + biking_time + running_time

-- Calculate the total distance
def total_distance := swimming_distance + biking_distance + running_distance

-- Average speed calculation
def average_speed := total_distance / total_time

-- Proof statement
theorem triathlete_average_speed : average_speed = 10.5 := by
  sorry

end triathlete_average_speed_l59_59158


namespace find_f_l59_59925

theorem find_f (f : ℝ → ℝ) : (∀ x : ℝ, f(2 * x + 1) = x^2 + x) → (∀ x : ℝ, f(x) = ¼ * x^2 - ¼) :=
by
  intro h
  sorry

end find_f_l59_59925


namespace probability_flies_swept_by_minute_hand_l59_59000

theorem probability_flies_swept_by_minute_hand :
  let flies_positions := {12, 3, 6, 9}
  -- Define the favorable starting intervals for the 20-minute sweep.
  let favorable_intervals := [(55, 60), (20, 25), (35, 40), (50, 55)]
  -- Total possible minutes in an hour
  let total_minutes := 60
  -- Total favorable minutes
  let favorable_minutes := 20
  -- Calculate the probability
  (favorable_minutes / total_minutes : ℝ) = (1 / 3 : ℝ):=
by
  sorry

end probability_flies_swept_by_minute_hand_l59_59000


namespace hyperbola_through_point_foci_l59_59046

def foci_of_ellipse (a b : ℝ) : set ℝ :=
{c | c^2 = a^2 - b^2}

def hyperbola_eq (a b x y : ℝ) : Prop :=
x^2 / a^2 - y^2 / b^2 = 1

def ellipse_eq (x y : ℝ) : Prop :=
x^2 / 9 + y^2 / 5 = 1

theorem hyperbola_through_point_foci (x y : ℝ) (hx : x = 3) (hy : y = real.sqrt 2)
  (h₁ : ellipse_eq (x, y) → ∀ f ∈ foci_of_ellipse 3 5, abs f = 2)
  (h₂ : ∃ a b, hyperbola_eq a b 3 (real.sqrt 2)) :
  hyperbola_eq 3 1 :=
begin
  sorry
end

end hyperbola_through_point_foci_l59_59046


namespace magician_trick_l59_59141

def Hamiltonian_cycle : Type :=
  { f : Fin 64 → (Fin 8 × Fin 8) // ∀ i, ∃ j, f j = (i / 8, i % 8) }

def board_filled (f : Hamiltonian_cycle) (a : Fin 64 → Nat) :=
  ∀ i, a i < 64 ∧ a i ≠ a j → i = j

def sum_S (a : Fin 64 → Nat) : Fin 64 :=
  Fin.ofNat ((∑ i, i * a i) % 64)

def magician_knows (a : Fin 64 → Nat) (S : Fin 64) : Nat :=
  let T := (S * a S + ((S + 1) % 64) * a ((S + 1) % 64)) % 64
  (T - S * a S) % 64

theorem magician_trick (f : Hamiltonian_cycle) (a : Fin 64 → Nat)
  (H1 : board_filled f a) :
  let S := sum_S a
  a ((S + 1) % 64) = magician_knows a S :=
by
  sorry

end magician_trick_l59_59141


namespace lattice_points_count_l59_59190

def abs (x : ℝ) : ℝ := if x >= 0 then x else -x

def region_bounded_by (x y : ℤ) : Prop := 
  (y = abs x) ∨ (y = -x^3 + 9 * x)

def is_lattice_point (x y : ℤ) : Prop := 
  region_bounded_by x y

theorem lattice_points_count : 
  (finset.univ.filter (λ (xy : ℤ × ℤ), is_lattice_point xy.1 xy.2)).card = 37 :=
sorry

end lattice_points_count_l59_59190


namespace probability_two_flies_swept_l59_59005

/-- Defining the positions of flies on the clock -/
inductive positions : Type
| twelve   | three   | six   | nine

/-- Probability that the minute hand sweeps exactly two specific positions after 20 minutes -/
theorem probability_two_flies_swept (flies : list positions) (time : ℕ) :
  (flies = [positions.twelve, positions.three, positions.six, positions.nine]) →
  (time = 20) →
  (probability_sweeps_two_flies flies time = 1 / 3) := sorry

end probability_two_flies_swept_l59_59005


namespace largest_fully_communicating_sets_eq_l59_59150

noncomputable def largest_fully_communicating_sets :=
  let total_sets := Nat.choose 99 4
  let non_communicating_sets_per_pod := Nat.choose 48 3
  let total_non_communicating_sets := 99 * non_communicating_sets_per_pod
  total_sets - total_non_communicating_sets

theorem largest_fully_communicating_sets_eq : largest_fully_communicating_sets = 2051652 := by
  sorry

end largest_fully_communicating_sets_eq_l59_59150


namespace positive_integers_a_2014_b_l59_59868

theorem positive_integers_a_2014_b (a : ℕ) :
  (∃! b : ℕ, 2 ≤ a / b ∧ a / b ≤ 5) → a = 6710 ∨ a = 6712 ∨ a = 6713 :=
by
  sorry

end positive_integers_a_2014_b_l59_59868


namespace five_inv_mod_33_l59_59883

theorem five_inv_mod_33 : ∃ x : ℤ, 0 ≤ x ∧ x < 33 ∧ 5 * x ≡ 1 [MOD 33] :=
by
  use 20
  split
  · norm_num
  split
  · norm_num
  · norm_num
  · rfl
sorry

end five_inv_mod_33_l59_59883


namespace sqrt_four_eq_two_or_neg_two_l59_59402

theorem sqrt_four_eq_two_or_neg_two (x : ℝ) : x^2 = 4 → (x = 2 ∨ x = -2) :=
sorry

end sqrt_four_eq_two_or_neg_two_l59_59402


namespace condition_of_inequality_l59_59233

theorem condition_of_inequality (x y : ℝ) (h : x^2 + y^2 ≤ 2 * (x + y - 1)) : x = 1 ∧ y = 1 :=
by
  sorry

end condition_of_inequality_l59_59233


namespace square_side_length_approx_l59_59396

theorem square_side_length_approx (y : ℝ) : (4 * y = 6 * Real.pi) → y ≈ 4.71 :=
by
  intro h
  sorry

end square_side_length_approx_l59_59396


namespace employee_salary_l59_59801

theorem employee_salary (x y : ℝ) (h1 : x + y = 770) (h2 : x = 1.2 * y) : y = 350 :=
by
  sorry

end employee_salary_l59_59801


namespace cos_2alpha_eq_seven_ninths_l59_59978

variable (α : ℝ)
def vector_a := (1 / 3, Real.tan α)
def vector_b := (Real.cos α, 1 : ℝ)

theorem cos_2alpha_eq_seven_ninths
  (hp : (∀ (u v : ℝ × ℝ), u = (1/3, Real.tan α) → v = (Real.cos α, 1) → u.1 * v.2 = v.1 * u.2)) :
  Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cos_2alpha_eq_seven_ninths_l59_59978


namespace alice_skates_in_l59_59820

noncomputable def time_skated (length_ft : ℕ) (radius_ft : ℕ) (spacing_ft : ℕ) (speed_mph : ℝ) : ℝ :=
  let quarter_circle_length := (real.pi / 2) * radius_ft
  let num_quarter_circles := length_ft / spacing_ft
  let total_quarter_circles_distance := num_quarter_circles * quarter_circle_length
  let straight_distance := length_ft - (num_quarter_circles * 2 * radius_ft)
  let total_distance := straight_distance + total_quarter_circles_distance
  let distance_in_miles := total_distance / 5280
  distance_in_miles / speed_mph

theorem alice_skates_in (h_length : length_ft = 3000) 
                        (h_radius : radius_ft = 10) 
                        (h_spacing : spacing_ft = 100) 
                        (h_speed : speed_mph = 4) :
  time_skated length_ft radius_ft spacing_ft speed_mph = 0.1362 :=
  sorry

end alice_skates_in_l59_59820


namespace evaluate_expression_l59_59202

-- Define the problem conditions
def a : ℕ := 2 ^ 1501
def b : ℕ := 5 ^ 1502

-- State the theorem to prove 
theorem evaluate_expression : (a + b)^2 - (a - b)^2 = 20 * 10^1501 := by
  calc
    (a + b)^2 - (a - b)^2 = 4 * a * b   : by rw [pow_add_pow_sub_eq_mul_four]; simp [a, b]
                       ... = 20 * 10^1501 : by rw [calculate_product_simplify]; sorry

-- auxiliary lemma to simplify the calculations
lemma pow_add_pow_sub_eq_mul_four (a b : ℕ) : (a + b)^2 - (a - b)^2 = 4 * a * b :=
  by ring

-- auxiliary lemma to simplify the product at the final step
lemma calculate_product_simplify : 4 * (2 ^ 1501) * (5 ^ 1502) = 20 * 10^1501 :=
  by sorry

end evaluate_expression_l59_59202


namespace percentage_increase_in_lines_l59_59354

theorem percentage_increase_in_lines (original increase increased : ℝ) 
  (h1 : increase = 110) (h2 : increased = 240) (h3 : increased = original + increase) :
  (increase / original) * 100 ≈ 84.62 :=
by {
  have h4 : original = increased - increase,
  { rw [h3, h1], exact sub_eq_add_neg increased increase },
  have h5 : original = 130,
  { rw [h4, h2, h1], linarith },
  have h6 : (increase / 130) * 100 = 84.61538461538,
  { field_simp, norm_num },
  have h7 : 84.61538461538 ≈ 84.62,
  { exact real.abs_sub_lt original 84.61538461538 84.62 (by norm_num : |84.61538461538 - 84.62| < 0.01) },
  rw [←h5] at h6,
  exact h7
}
sorry

end percentage_increase_in_lines_l59_59354


namespace problem_statement_l59_59857

def square_area_diagonal (d : ℝ) : ℝ :=
  let s := d / Real.sqrt 2
  s^2

def rectangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  2 * (Real.abs (x1 - x2) + Real.abs (y2 - y1))

def triangle_area_with_line (a b : ℝ) : ℝ :=
  let b := b
  let h := b
  0.5 * b * h

theorem problem_statement :
  let d := 2 * Real.sqrt 2
  let A := square_area_diagonal d
  let B := rectangle_area 2 2  4 1  4 3
  let C := triangle_area_with_line 4 2 in
  (A < B ∨ B < C) ∧ (A = C ∨ C < B) :=
by
  let d := 2 * Real.sqrt 2
  let A := square_area_diagonal d
  let B := rectangle_area 2 2 4 1 4 3
  let C := triangle_area_with_line 4 2
  sorry

end problem_statement_l59_59857


namespace justin_tim_games_count_l59_59171

/-- 
At Barwell Middle School, a larger six-square league has 12 players, 
including Justin and Tim. Daily at recess, the twelve players form two six-square games,
each involving six players in no relevant order. Over a term, each possible 
configuration of six players happens exactly once. How many times did Justin and Tim 
end up in the same six-square game?
--/
theorem justin_tim_games_count :
  let total_players := 12 in
  let game_size := 6 in
  let justin_tim_teams_total := (@Nat.choose 10 4) in
  justin_tim_teams_total = 210 :=
by
  sorry

end justin_tim_games_count_l59_59171


namespace find_winning_numbers_l59_59771

-- The game step function following the rules
def game_step (n : ℕ) : ℕ :=
  if n % 2 = 1 then n + 7 else n / 2

-- The game sequence function applying game_step repeatedly
def game_sequence (start : ℕ) : Stream ℕ :=
  Stream.iterate game_step start

-- The condition that the sequence returns to the start number
def wins (start : ℕ) : Prop :=
  ∃ k > 0, game_sequence start k = start

-- The main statement
theorem find_winning_numbers (n : ℕ) : n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 7 ∨ n = 8 ∨ n = 14 ↔ wins n :=
sorry

end find_winning_numbers_l59_59771


namespace find_x_value_l59_59572

def f (x : ℝ) (b : ℝ) := 2 * x - b

theorem find_x_value : 
  ∀ (b : ℝ), (b = 3) → (∀ x, (2 * f x b - 11 = f (x - 2) b) ↔ x = 5) :=
by
  intros b hb
  rw hb
  intro x
  constructor
  { sorry },
  { solving_by_lean_arithmetic }

end find_x_value_l59_59572


namespace ellipse_equation_range_k_l59_59956

-- Define the given conditions
variables (a b : ℝ) (e : ℝ := 1 / 2)
variables (k : ℝ) (x y : ℝ)
variables (P : ℝ × ℝ := (2 * real.sqrt 6 / 3, 1))
variables (M : ℝ × ℝ := (1 / 6, 0))

-- a and b 
def conditions (a b : ℝ) := 
  a > b ∧ b > 0 ∧ a = 2 ∧ b = real.sqrt 3

-- Define the ellipse C
def ellipse_eq (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

-- The equation of the ellipse
theorem ellipse_equation :
  conditions a b → ellipse_eq x y a b → ellipse_eq x y 2 (real.sqrt 3) :=
  sorry

-- Define the line equation and intersection points
def line_eq (k b x : ℝ) := k * x + b
def ellipse_line_intersection (k b : ℝ) := 
  let D := (8 * k * b)^2 - 4 * (3 + 4 * k^2) * (4 * b^2 - 12) in 
  D > 0 ∧ b^2 < 3 + 4 * k^2

-- Define the midpoint and perpendicular bisector conditions
def midpoint_condition (k : ℝ) (M : ℝ × ℝ) :=
  let midpoint := (-4 * k * b / (3 + 4 * k^2), 3 * b / (3 + 4 * k^2)) in
  let perp_bisector := (-1 / k) * (x - 1 / 6) in
  midpoint.2 = - (1 / k) * (- 4 * k * b / (3 + 4 * k^2) - 1 / 6) ∧ M = midpoint

-- The range theorem for k
theorem range_k :
  ∀ k: ℝ, k ≠ 0 ∧ ellipse_line_intersection k b → 
           midpoint_condition k M → 
           k < -real.sqrt 6 / 8 ∨ k > real.sqrt 6 / 8 :=
  sorry

end ellipse_equation_range_k_l59_59956


namespace parabola_focus_coordinates_l59_59037

theorem parabola_focus_coordinates :
  let y := 4 * x^2 in
  ∃ (p : ℝ), p = 1 / 16 ∧ focus y = (0, p) :=
by
  sorry

end parabola_focus_coordinates_l59_59037


namespace train_speed_l59_59155

def train_length : ℕ := 110
def bridge_length : ℕ := 265
def crossing_time : ℕ := 30

def speed_in_m_per_s (d t : ℕ) : ℕ := d / t
def speed_in_km_per_hr (s : ℕ) : ℕ := s * 36 / 10

theorem train_speed :
  speed_in_km_per_hr (speed_in_m_per_s (train_length + bridge_length) crossing_time) = 45 :=
by
  sorry

end train_speed_l59_59155


namespace correct_statement_is_b_l59_59425

-- Definitions for geometrical objects
variables {A B C D : Type}
variables [is_rhombus : rhombus A] [is_rectangle : rectangle B] [is_parallelogram : parallelogram C] [is_square : square D]
variables [quad : quadrilateral A] (eq_diagonals : diag A = diag B) (perp_diag : perpendicular (diag A) (diag B))

-- Condition definitions
def rhombus_diagonals_not_equal_unless_square (A : Type) [rhombus A] : Prop :=
  ¬(diag A = diag B) ∨ (is_square A)
  
def rectangle_diagonals_equal_and_bisect (B : Type) [rectangle B] : Prop :=
  diag B = diag C ∧ bisect (diag B) (diag A)
  
def parallelogram_not_symmetrical_unless_rectangle_or_rhombus (C : Type) [parallelogram C] : Prop :=
  ¬(symmetrical C) ∨ (is_rectangle C ∨ is_rhombus C)
  
def quadrilateral_perp_equal_not_always_square (A : Type) [quadrilateral A] : Prop :=
  ¬ (quad A ∧ eq_diagonals ∧ perp_diag = is_square A)

-- Proof problem statement
theorem correct_statement_is_b :
  rhombus_diagonals_not_equal_unless_square A →
  rectangle_diagonals_equal_and_bisect B →
  parallelogram_not_symmetrical_unless_rectangle_or_rhombus C →
  quadrilateral_perp_equal_not_always_square A →
  (rectangle_diagonals_equal_and_bisect B) := 
sorry

end correct_statement_is_b_l59_59425


namespace num_not_M_functions_l59_59455

noncomputable def is_M_function (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Icc 0 1, f x ≥ 0) ∧ 
  (∀ x1 x2 ∈ Icc 0 1, x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2)

def f1 : ℝ → ℝ := λ x, x^2
def f2 : ℝ → ℝ := λ x, x^2 + 1
def f3 : ℝ → ℝ := λ x, 2^x - 1

theorem num_not_M_functions : 
  (∃ n ∈ ({f1, f2, f3} : finset (ℝ → ℝ)), ¬ is_M_function n) →
   1 :=
sorry

end num_not_M_functions_l59_59455


namespace relationship_between_lines_l59_59990

variables {a b : ℝ} {α : set ℝ}

-- Define parallelism between a line and a plane
def line_parallel_plane (a : ℝ) (α : set ℝ) : Prop := 
  ∀ b (h : b ∈ α), a ≠ b

-- Define a line being in a plane
def line_in_plane (b : ℝ) (α : set ℝ) : Prop := 
  b ∈ α

-- Define parallel or skew relationship between two lines
def line_parallel_or_skew (a b : ℝ) (α : set ℝ) : Prop := 
  (a ∈ α ∧ ∃ b (h : b ∈ α), a ≠ b) ∨ (∀ x (hx : x ∈ α), a ≠ x)

theorem relationship_between_lines {a b : ℝ} {α : set ℝ} :
  line_parallel_plane a α → line_in_plane b α → line_parallel_or_skew a b α :=
by
  intro h1 h2
  sorry

end relationship_between_lines_l59_59990


namespace complement_of_M_l59_59253

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}

theorem complement_of_M :
  (U \ M) = {x | 0 ≤ x ∧ x ≤ 1} :=
sorry

end complement_of_M_l59_59253


namespace train_length_l59_59427

theorem train_length
    (length_first_train : ℝ := 270)
    (speed_first_train_kmph : ℝ := 120)
    (speed_second_train_kmph : ℝ := 80)
    (crossing_time_sec : ℝ := 9) :
    let speed_first_train_mps := speed_first_train_kmph * 1000 / 3600 in
    let speed_second_train_mps := speed_second_train_kmph * 1000 / 3600 in
    let relative_speed_mps := speed_first_train_mps + speed_second_train_mps in
    let total_distance := relative_speed_mps * crossing_time_sec in
    total_distance - length_first_train = 229.95 :=
by
  -- sorry to skip the proof
  sorry

end train_length_l59_59427


namespace market_price_level6_l59_59499

noncomputable def market_price (x : ℕ) (a b : ℝ) : ℝ := real.exp (a * x + b)

theorem market_price_level6 :
  ∀ (a b : ℝ),
  (real.exp (4 * a) = 3) →
  (market_price 3 a b = 60) →
  market_price 6 a b ≈ 170 :=
by
  intros a b h1 h2,
  have h3 : real.exp a ≈ 1.41 := sorry,
  have h4 : market_price 6 a b = 120 * real.sqrt 2 := sorry,
  show 120 * real.sqrt 2 ≈ 170, from sorry

end market_price_level6_l59_59499


namespace find_m_value_l59_59050

def f (m x : ℝ) : ℝ := (m^2 - m - 1) * x^m

theorem find_m_value
  (h_increasing : ∀ x : ℝ, 0 < x → 0 < x → x > 0 → (m^2 - m - 1) * x^m > (m^2 - m - 1) * x^m)
  : ∃ m : ℝ, m = 2 :=
begin
  sorry
end

end find_m_value_l59_59050


namespace area_of_intersecting_fig_l59_59800

open Real

/-- The figure formed by the intersection of two lines y = x, x = -8, and the x-axis has an area of 32 square units. -/
theorem area_of_intersecting_fig : 
  let 
    line1 (x : ℝ) := x,
    line2 (x : ℝ) := -8
  in
  let x_inter := -8
  in
  let y_inter := line1 x_inter
  in
  let base := 0 + 8
  in
  let height := 0 + 8
  in
  1 / 2 * base * height = 32 := sorry

end area_of_intersecting_fig_l59_59800


namespace find_lambda_l59_59985

noncomputable def a : ℝ × ℝ × ℝ := (-1, 0, -2)
noncomputable def b : ℝ × ℝ × ℝ := (2, -1, 1)
noncomputable def cos_120 : ℝ := -1 / 2

theorem find_lambda (λ : ℝ) :
  let a := (-1, λ, -2)
  let dot_prod := (-1) * b.1 + λ * b.2 + (-2) * b.3
  let mag_a := Real.sqrt((-1)^2 + λ^2 + (-2)^2)
  let mag_b := Real.sqrt(2^2 + (-1)^2 + 1^2)
  cos_120 = dot_prod / (mag_a * mag_b) →
  λ = -1 ∨ λ = 17 :=
sorry

end find_lambda_l59_59985


namespace find_number_l59_59111

theorem find_number : ∃ x : ℝ, 0 < x ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end find_number_l59_59111


namespace percent_fair_haired_is_twenty_five_l59_59125

noncomputable def percent_fair_haired_employees (employee_count : ℕ) 
  (percent_women_fair_haired : ℝ) (percent_fair_haired_women : ℝ)
  (women_fair_haired_count : ℝ) : ℝ :=
  let fair_haired_employees := women_fair_haired_count / (percent_fair_haired_women / 100)
  in (fair_haired_employees / employee_count) * 100

theorem percent_fair_haired_is_twenty_five 
  (employee_count : ℕ) (percent_women_fair_haired : ℝ) (percent_fair_haired_women : ℝ) : 
  employee_count = 100 ∧ percent_women_fair_haired = 10 ∧ percent_fair_haired_women = 40 → 
  percent_fair_haired_employees employee_count percent_women_fair_haired percent_fair_haired_women 10 = 25 := 
by
  -- proof would go here
  sorry

end percent_fair_haired_is_twenty_five_l59_59125


namespace avg_height_of_remaining_students_l59_59381

-- Define the given conditions
def avg_height_11_members : ℝ := 145.7
def number_of_members : ℝ := 11
def height_of_two_students : ℝ := 142.1

-- Define what we need to prove
theorem avg_height_of_remaining_students :
  (avg_height_11_members * number_of_members - 2 * height_of_two_students) / (number_of_members - 2) = 146.5 :=
by
  sorry

end avg_height_of_remaining_students_l59_59381


namespace son_father_age_sum_l59_59810

theorem son_father_age_sum
    (S F : ℕ)
    (h1 : F - 6 = 3 * (S - 6))
    (h2 : F = 2 * S) :
    S + F = 36 :=
sorry

end son_father_age_sum_l59_59810


namespace sum_of_integer_solutions_to_absolute_inequalities_l59_59785

noncomputable def sum_of_solutions : ℤ :=
  if (∀ n : ℤ, |n| < |n-2| ∧ |n-2| < 10 → n ∈ (-8, 1] ∨ n ∈ [0, 1]) then -35 else 0

theorem sum_of_integer_solutions_to_absolute_inequalities :
  sum_of_solutions = -35 := by
  sorry

end sum_of_integer_solutions_to_absolute_inequalities_l59_59785


namespace ratio_DE_BC_is_1_2857_l59_59631

-- Definitions as per the conditions
variables {A B C D E : Type*}
variables [plane_geometry : Geometry A B C D E]

-- Given conditions
variables {triangle_ABC : is_triangle A B C}
variables {point_D_on_AB : on_line A D B}
variables {point_E_on_AC : on_line A E C}
variables {is_trapezium_BCED : is_trapezium B C E D}

-- Given ratio of areas
variables {area_ratio_ADE_to_BCED : area (triangle A D E) / area (trapezium B C E D) = 0.5625}

-- Prove the ratio of DE to BC
theorem ratio_DE_BC_is_1_2857 : (DE / BC = 1.2857) :=
sorry -- Proof steps go here

end ratio_DE_BC_is_1_2857_l59_59631


namespace ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l59_59846

theorem ellipse_equation_x_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 4 ∧ b = 3 ∧ a = 5 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 := by
  sorry

theorem ellipse_equation_y_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 3 ∧ b = 4 ∧ a = 5 ∧ (x^2 / b^2) + (y^2 / a^2) = 1 := by
  sorry

end ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l59_59846


namespace number_of_Qs_l59_59663

noncomputable def P (x : ℝ) := (x - 1) * (x - 2) * (x - 3) * (x - 4)

theorem number_of_Qs : 
  ∃ Q : polynomial ℝ, Q.degree = 3 ∧ 
  (∃ R : polynomial ℝ, R.degree = 4 ∧ P (Q x) = P x * R x) ∧ 
  ∑ x ∈ {1, 2, 3, 4}, (Q x) ∈ Finset.univ.attach = 228 :=
sorry

end number_of_Qs_l59_59663


namespace problem_equiv_l59_59582

noncomputable def a_n (n : ℕ) : ℕ := 3 * n
noncomputable def b_n (n : ℕ) : ℕ := 3 * n + 2 ^ (n - 1)

theorem problem_equiv 
  (b_n_has : ∀ n, b_n n = 3 * n + 2 ^ (n - 1))
  (a_sum : ∑ i in finset.range n, b_n i = (3 * n * (n + 1) / 2) + 2 ^ n - 1) : 
  true :=
by { trivial }

end problem_equiv_l59_59582


namespace find_two_primes_l59_59911

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m ≠ n → n % m ≠ 0

-- Prove the existence of two specific prime numbers with the desired properties
theorem find_two_primes :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p = 2 ∧ q = 5 ∧ is_prime (p + q) ∧ is_prime (q - p) :=
by
  exists 2
  exists 5
  repeat {split}
  sorry

end find_two_primes_l59_59911


namespace proof_problem_l59_59860

def diagonal_square_area (d : ℝ) : ℝ :=
  let s := d / Real.sqrt 2 in
  s * s

def rectangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  let width := (x2 - 2).abs in
  let height := (if y1 ≤  y2 then y2  - y1 else y1 - y2) * 2 in
  width * height

def triangle_area (m b : ℝ) : ℝ :=
  let x_int := 2 * b in
  let y_int := b in
  0.5 * x_int * y_int

theorem proof_problem :
  let A := diagonal_square_area (2 * Real.sqrt 2)
  let B := rectangle_area 2 2 4 3
  let C := triangle_area (-1/2) 2
  A = 4 ∧ B = 8 ∧ C = 4 ∧ A < B :=
by {
  have hA := calc
    A = 4             : by sorry,
  have hB := calc
    B = 8             : by sorry,
  have hC := calc
    C = 4             : by sorry,
  have hD := calc
    A < B             : by sorry,
  exact ⟨hA, hB, hC, hD⟩
}

end proof_problem_l59_59860


namespace derivative_y_l59_59039

noncomputable def y (x : ℝ) : ℝ := Real.sin x - Real.exp (x * Real.log 2)

theorem derivative_y (x : ℝ) : 
  deriv y x = Real.cos x - Real.exp (x * Real.log 2) * Real.log 2 := 
by 
  sorry

end derivative_y_l59_59039


namespace complex_quadrant_problem_l59_59254

open Complex

theorem complex_quadrant_problem
  (z1 z2 : ℂ)
  (h1 : z1 = -3 + Complex.i)
  (h2 : z2 = 1 - Complex.i) :
  let z := z1 - z2 in
  (z.re < 0 ∧ z.im > 0) :=
by
  sorry

end complex_quadrant_problem_l59_59254


namespace person_B_catches_up_after_meeting_point_on_return_l59_59120
noncomputable def distance_A := 46
noncomputable def speed_A := 15
noncomputable def speed_B := 40
noncomputable def initial_gap_time := 1

-- Prove that Person B catches up to Person A after 3/5 hours.
theorem person_B_catches_up_after : 
  ∃ x : ℚ, 40 * x = 15 * (x + 1) ∧ x = 3 / 5 := 
by
  sorry

-- Prove that they meet 10 kilometers away from point B on the return journey.
theorem meeting_point_on_return : 
  ∃ y : ℚ, (46 - y) / 15 - (46 + y) / 40 = 1 ∧ y = 10 := 
by 
  sorry

end person_B_catches_up_after_meeting_point_on_return_l59_59120


namespace trail_length_l59_59174

variables (a b c d e : ℕ)

theorem trail_length : 
  a + b + c = 45 ∧
  b + d = 36 ∧
  c + d + e = 60 ∧
  a + d = 32 → 
  a + b + c + d + e = 69 :=
by
  intro h
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end trail_length_l59_59174


namespace ratio_roots_l59_59404

theorem ratio_roots (p q r s : ℤ)
    (h1 : p ≠ 0)
    (h_roots : ∀ x : ℤ, (x = -1 ∨ x = 3 ∨ x = 4) → (p*x^3 + q*x^2 + r*x + s = 0)) : 
    (r : ℚ) / s = -5 / 12 :=
by sorry

end ratio_roots_l59_59404


namespace shortest_chord_length_l59_59592

theorem shortest_chord_length (x y : ℝ) (h : x^2 + y^2 - 6*x - 8*y = 0) : ∃ l, l = 4 * real.sqrt 6 :=
sorry

end shortest_chord_length_l59_59592


namespace exists_segment_seen_at_angle_exists_segment_seen_from_two_points_exists_n_gon_on_lines_inscribe_n_gon_passing_points_inscribe_polygon_with_properties_l59_59109

-- Problem 30.56 a)
theorem exists_segment_seen_at_angle (l : Line) (P : Point) (α : Angle) (XY : Length) :
  ∃ (X Y : Point), on_line X l ∧ on_line Y l ∧ sees_from P X Y α :=
sorry

-- Problem 30.56 b)
theorem exists_segment_seen_from_two_points (l1 l2 : Line) (P Q : Point) (α β : Angle) :
  ∃ (X : Point) (Y : Point), on_line X l1 ∧ on_line Y l2 ∧ sees_from P X Y α ∧ sees_from Q X Y β :=
sorry

-- Problem 30.57 a)
theorem exists_n_gon_on_lines (circle : Circle) (points : Fin n → Point) (lines : Fin n → Line) :
  ∃ (polygon : Polygon n), vertices_on_lines polygon circle points lines :=
sorry

-- Problem 30.57 b)
theorem inscribe_n_gon_passing_points (circle : Circle) (points : Fin n → Point) :
  ∃ (polygon : Polygon n), inscribed_polygon_passing_points polygon circle points :=
sorry

-- Problem 30.57 c)
theorem inscribe_polygon_with_properties (circle : Circle) 
  (passing_points : List Point) (parallel_lines : List Line) 
  (lengths : List Length) :
  ∃ (polygon : Polygon), inscribed_polygon_with_properties polygon circle passing_points parallel_lines lengths :=
sorry

end exists_segment_seen_at_angle_exists_segment_seen_from_two_points_exists_n_gon_on_lines_inscribe_n_gon_passing_points_inscribe_polygon_with_properties_l59_59109


namespace find_interest_rate_l59_59154

noncomputable def interest_rate (first_part second_part total : ℝ) (first_years second_years : ℕ) (first_rate second_rate : ℝ) : ℝ :=
  (total - second_part) * (first_rate / 100) * (first_years : ℝ) = second_part * (second_rate / 100) * (second_years : ℝ)

theorem find_interest_rate :
  let total := 2730
  let first_years := 8
  let second_years := 3
  let first_rate := 3
  let second_sum := 1680
  interest_rate second_sum first_years first_rate second_years 5 := sorry

end find_interest_rate_l59_59154


namespace sum_reciprocals_nonempty_subsets_prod_l59_59974

theorem sum_reciprocals_nonempty_subsets_prod (A : Set ℕ) (hA : A = {k : ℕ | 1 ≤ k ∧ k ≤ 2019}) :
  (∑ S in (Finset.powerset (Finset.filter (λ x, x ≠ 0) (Finset.range 2020))) \ {∅}, 
   (1 : ℚ) / S.prod id) = 2019 :=
by { sorry }

end sum_reciprocals_nonempty_subsets_prod_l59_59974


namespace value_of_f_at_2_plus_log2_3_l59_59261

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 4 then 2^x else f (x + 1)

theorem value_of_f_at_2_plus_log2_3 : f (2 + log 3 / log 2) = 24 :=
by
  sorry

end value_of_f_at_2_plus_log2_3_l59_59261


namespace blue_marbles_l59_59061

theorem blue_marbles (r b : ℕ) (h_ratio : 3 * b = 5 * r) (h_red : r = 18) : b = 30 := by
  -- proof
  sorry

end blue_marbles_l59_59061


namespace average_percentage_decrease_is_10_l59_59412

noncomputable def average_percentage_decrease (original_cost final_cost : ℝ) (n : ℕ) : ℝ :=
  1 - (final_cost / original_cost)^(1 / n)

theorem average_percentage_decrease_is_10
  (original_cost current_cost : ℝ)
  (n : ℕ)
  (h_original_cost : original_cost = 100)
  (h_current_cost : current_cost = 81)
  (h_n : n = 2) :
  average_percentage_decrease original_cost current_cost n = 0.1 :=
by
  -- The proof would go here if it were needed.
  sorry

end average_percentage_decrease_is_10_l59_59412


namespace length_of_PJ_in_triangle_PQR_l59_59316

noncomputable def triangle_PQR_PJ_length : ℝ :=
  let PQ := 17
  let PR := 19
  let QR := 20
  let incenter := (a b c : ℝ) → true -- Formal definition of incenter comes here
  let incircle_touches := (a b c d e f : ℝ) → true -- Formal definition of incircle touching sides
  let in_radius := 6 -- Inradius calculated
  let G := true -- Placeholder for more formal definition
  let H := true -- Placeholder for more formal definition
  let I := true -- Placeholder for more formal definition
  let J := incenter PQ PR QR -- J is the incenter
  let PJ := sqrt (11^2 + 6^2) -- Final computation for PJ
  PJ

theorem length_of_PJ_in_triangle_PQR :
  triangle_PQR_PJ_length = sqrt 157 :=
sorry

end length_of_PJ_in_triangle_PQR_l59_59316


namespace triangle_cosine_inequality_l59_59291

theorem triangle_cosine_inequality (A B C : ℝ) (h1 : A + B + C = π) :
  (cos A / cos B) ^ 2 + (cos B / cos C) ^ 2 + (cos C / cos A) ^ 2 ≥
  4 * (cos A ^ 2 + cos B ^ 2 + cos C ^ 2) :=
by {
  sorry
}

end triangle_cosine_inequality_l59_59291


namespace annual_rent_per_square_foot_is_172_l59_59055

def monthly_rent : ℕ := 3600
def local_taxes : ℕ := 500
def maintenance_fees : ℕ := 200
def length_of_shop : ℕ := 20
def width_of_shop : ℕ := 15

def total_monthly_cost : ℕ := monthly_rent + local_taxes + maintenance_fees
def annual_cost : ℕ := total_monthly_cost * 12
def area_of_shop : ℕ := length_of_shop * width_of_shop
def annual_rent_per_square_foot : ℕ := annual_cost / area_of_shop

theorem annual_rent_per_square_foot_is_172 :
  annual_rent_per_square_foot = 172 := by
    sorry

end annual_rent_per_square_foot_is_172_l59_59055


namespace find_omega_l59_59947

noncomputable def omega : ℝ := (Real.sqrt 3) / 3

theorem find_omega 
  (ω : ℝ) 
  (h_sym : ∀ (x : ℝ), 
    y = sqrt 2 * sin (ω * x) - cos (ω * x) → 
    P1 = (x, (sqrt 2 * sin (ω * x) - cos (ω * x))) → 
    P2 = (kπ + x, (sqrt 2 * sin (ω * (kπ + x)) - cos (ω * (kπ + x))))) 
  (h_perpendicular : ∀ (x1 x2 : ℝ), 
    tangent_at P1 = dy/dx ∣ P1 → 
    tangent_at P2 = dy/dx ∣ P2 → 
    (dy/dx ∣ P1) * (dy/dx ∣ P2) = -1) 
  (h_pos : ω > 0) : 
  ω = (Real.sqrt 3) / 3 := 
sorry

end find_omega_l59_59947


namespace fish_speed_in_still_water_l59_59452

theorem fish_speed_in_still_water (u d : ℕ) (v : ℕ) : 
  u = 35 → d = 55 → 2 * v = u + d → v = 45 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end fish_speed_in_still_water_l59_59452


namespace expand_polynomial_l59_59516

theorem expand_polynomial :
  (3 * x ^ 2 - 4 * x + 3) * (-2 * x ^ 2 + 3 * x - 4) = -6 * x ^ 4 + 17 * x ^ 3 - 30 * x ^ 2 + 25 * x - 12 :=
by
  sorry

end expand_polynomial_l59_59516


namespace angle_B_not_right_angle_sin_C_over_sin_A_range_l59_59922

variables {α : Type*} [linear_ordered_field α] [real_like α]

-- Part (1): Angle B cannot be a right angle
theorem angle_B_not_right_angle (A B C : α) (h : 2 * sin C * sin (B - A) = 2 * sin A * sin C - sin^2 B) :
  B ≠ π / 2 :=
begin
  sorry
end

-- Part (2): Range of sin(C)/sin(A) for an acute triangle ABC
theorem sin_C_over_sin_A_range (A B C : α) 
  (h : 2 * sin C * sin (B - A) = 2 * sin A * sin C - sin^2 B)
  (h_acute : A < π/2 ∧ B < π/2 ∧ C < π/2) :
  1 / 3 < sin C / sin A ∧ sin C / sin A < 5 / 3 :=
begin
  sorry
end

end angle_B_not_right_angle_sin_C_over_sin_A_range_l59_59922


namespace _l59_59780

-- Define a right triangle
structure RightTriangle (a b c : ℝ) : Prop :=
  (legs_nonnegative : a ≥ 0 ∧ b ≥ 0)
  (pythagorean_theorem : a^2 + b^2 = c^2)

example (a b : ℝ) (h : RightTriangle 90 120 a) : a = 150 :=
by
  -- Use the given conditions to state the Pythagorean theorem
  have h1 : 90^2 + 120^2 = a^2 := h.pythagorean_theorem
  -- Simplify to find a
  have h2 : a^2 = 22500 := by
    rw [pow_two, pow_two] at h1
    norm_num at h1
    exact h1
  -- Solve for a
  have h3 : abs a = 150 := by
    rw [eq_of_square_eq_square_iff_nonneg_of_nonneg] at h2
    norm_num at h2
    assumption
  -- Conclude that a = 150 since a is non-negative
  have : a = 150 := by
    rw abs_eq_self at h3
    exact h3
  -- Conclude the proof by stating the final value
  exact this

end _l59_59780


namespace quadratic_equation_general_form_l59_59389

theorem quadratic_equation_general_form (x : ℝ) (h : 4 * x = x^2 - 8) : x^2 - 4 * x - 8 = 0 :=
sorry

end quadratic_equation_general_form_l59_59389


namespace determine_abcd_l59_59584

theorem determine_abcd (a b c d e f p q : ℕ) 
  (H1 : b > c) (H2 : c > d) (H3 : d > a)
  (H4 : (1000 * c + 100 * d + 10 * a + b) - (1000 * a + 100 * b + 10 * c + d) = 1000 * p + 100 * q + 10 * e + f)
  (H5 : ∃ (k : ℕ), 10 * k + (H5 = 100 * k + l + 10 * m + ;)]
  (H6 : 100 * p + q % 5 ≠ 0) :
  1000 * a + 100 * b + 10 * c + d = 1983 :=
sorry

end determine_abcd_l59_59584


namespace part_a_part_b_l59_59122

theorem part_a (k : ℕ) : ∃ (a : ℕ → ℕ), (∀ i, i ≤ k → a i > 0) ∧ (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → a i < a j) ∧ (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) :=
sorry

theorem part_b : ∃ C > 0, ∀ a : ℕ → ℕ, (∀ k : ℕ, (∀ i j, 1 ≤ i ∧ i ≠ j ∧ i ≤ k ∧ j ≤ k → (a i - a j) ∣ a i) → a 1 > (k : ℕ) ^ (C * k : ℕ)) :=
sorry

end part_a_part_b_l59_59122


namespace find_x_l59_59889

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem find_x (x : ℕ) (h : x * factorial x + 2 * factorial x = 40320) : x = 6 :=
sorry

end find_x_l59_59889


namespace samantha_coins_value_l59_59020

theorem samantha_coins_value (n d : ℕ) (h1 : n + d = 25) 
    (original_value : ℕ := 250 - 5 * n) 
    (swapped_value : ℕ := 125 + 5 * n)
    (h2 : swapped_value = original_value + 100) : original_value = 140 := 
by
  sorry

end samantha_coins_value_l59_59020


namespace suzy_total_jumps_in_two_days_l59_59029

-- Definitions based on the conditions in the problem
def yesterdays_jumps : ℕ := 247
def additional_jumps_today : ℕ := 131
def todays_jumps : ℕ := yesterdays_jumps + additional_jumps_today

-- Lean statement of the proof problem
theorem suzy_total_jumps_in_two_days : yesterdays_jumps + todays_jumps = 625 := by
  sorry

end suzy_total_jumps_in_two_days_l59_59029


namespace distance_O_to_DEF_l59_59642

-- Define the coordinates of point D and the vector m
def point_D : ℝ × ℝ × ℝ := (2, 1, 0)
def vector_m : ℝ × ℝ × ℝ := (4, 1, 2)

-- Define the dot product of two vectors in ℝ^3
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the magnitude of a vector in ℝ^3
def magnitude (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Define the distance from point O to the plane DEF
def distance_from_O_to_DEF (D m : ℝ × ℝ × ℝ) : ℝ :=
  (dot_product (2, 1, 0) vector_m) / magnitude vector_m

theorem distance_O_to_DEF : distance_from_O_to_DEF point_D vector_m = 3 * Real.sqrt 21 / 7 :=
by sorry

end distance_O_to_DEF_l59_59642


namespace sum_of_integer_solutions_l59_59783

/-- 
  The sum of all integer solutions to the inequality |n| < |n - 2| < 10 is -36 
  We need to define n as an integer and specify a condition using absolute values
  which translates mathematically to the required sum.
-/

theorem sum_of_integer_solutions : 
  let solutions := { n : ℤ | abs n < abs (n - 2) ∧ abs (n - 2) < 10 } in
  ∑ n in solutions, n = -36 :=
by 
  sorry

end sum_of_integer_solutions_l59_59783


namespace general_formula_a_sum_first_n_terms_b_l59_59315

-- Definitions and conditions
def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sequence_b (n : ℕ) : ℕ → ℝ 
| n := (-1)^(n-1) * 4 * n / (sequence_a n * (sequence_a (n+1)))

-- First part: Prove general formula for sequence {a_n}
theorem general_formula_a (n : ℕ) (h1 : sequence_a 1 = 1) 
  (h2 : ∀ n, (sequence_a (n+1)) / (n+1) - (sequence_a n) / n = 1 / (n * (n+1))) 
  : ∀ n, sequence_a n = 2 * n - 1 := 
sorry

-- Second part: Prove the sum of the first n terms of the sequence {b_n}
theorem sum_first_n_terms_b (n : ℕ) 
  (h1 : ∀ n, sequence_a n = 2 * n - 1) 
  (h2 : ∀ n, b n = (-1)^(n-1) * 4 * n / (sequence_a n * sequence_a (n+1))) 
  : ∀ n, ∑ i in range n, sequence_b i = (2 * n + 1 + (-1)^(n-1)) / (2 * n + 1) := 
sorry

end general_formula_a_sum_first_n_terms_b_l59_59315


namespace factorial_division_l59_59185

theorem factorial_division (n : ℕ) (h : 6 ≤ n):
  ((nat.factorial 7 + nat.factorial 8) / nat.factorial 6) = 63 :=
sorry

end factorial_division_l59_59185


namespace find_n_l59_59191

noncomputable def t_seq : ℕ → ℚ
| 1           := 2
| (n + 1) :=
  if (n + 1) % 3 = 0 then
    2 + t_seq ((n + 1) / 3)
  else
    1 / t_seq n

theorem find_n (n : ℕ) (h : t_seq n = 3 / 29) : n = 40 :=
sorry

end find_n_l59_59191


namespace janessa_kept_correct_number_of_cards_l59_59652

-- Definitions based on the conditions
def initial_cards : ℕ := 4
def father_cards : ℕ := 13
def ordered_cards : ℕ := 36
def bad_cards : ℕ := 4
def percentage_keep : ℚ := 0.40
def cards_to_dexter : ℕ := 29

-- The total number of good cards
def good_condition_cards := ordered_cards - bad_cards
def total_cards := initial_cards + father_cards + good_condition_cards

-- The number of cards kept by Janessa
def cards_kept := (total_cards * percentage_keep).to_nat -- Using to_nat to convert from ℚ to ℕ, rounding down

-- The statement to prove
theorem janessa_kept_correct_number_of_cards : cards_kept = 19 := by
  sorry

end janessa_kept_correct_number_of_cards_l59_59652


namespace no_valid_placement_of_digits_l59_59803

theorem no_valid_placement_of_digits:
  ∀ (A B C : ℕ), 
  (∑ i in finset.range 10, i = 45) →
  (A + B = 45) → 
  (6 * C = 3 * A + B) → 
  false :=
by sorry

end no_valid_placement_of_digits_l59_59803


namespace initial_mixtureA_amount_l59_59348

-- Condition 1: Mixture A is 20% oil and 80% material B by weight.
def oil_content (x : ℝ) : ℝ := 0.20 * x
def materialB_content (x : ℝ) : ℝ := 0.80 * x

-- Condition 2: 2 more kilograms of oil are added to a certain amount of mixture A
def oil_added := 2

-- Condition 3: 6 kilograms of mixture A must be added to make a 70% material B in the new mixture.
def mixture_added := 6

-- The total weight of the new mixture
def total_weight (x : ℝ) : ℝ := x + mixture_added + oil_added

-- The total amount of material B in the new mixture
def total_materialB (x : ℝ) : ℝ := 0.80 * x + 0.80 * mixture_added

-- The new mixture is supposed to be 70% material B.
def is_70_percent_materialB (x : ℝ) : Prop := total_materialB x = 0.70 * total_weight x

-- Proving x == 8 given the conditions
theorem initial_mixtureA_amount : ∃ x : ℝ, is_70_percent_materialB x ∧ x = 8 :=
by
  sorry

end initial_mixtureA_amount_l59_59348


namespace red_balls_in_total_color_of_158th_ball_l59_59069

def totalBalls : Nat := 200
def redBallsPerCycle : Nat := 5
def whiteBallsPerCycle : Nat := 4
def blackBallsPerCycle : Nat := 3
def cycleLength : Nat := redBallsPerCycle + whiteBallsPerCycle + blackBallsPerCycle

theorem red_balls_in_total :
  (totalBalls / cycleLength) * redBallsPerCycle + min redBallsPerCycle (totalBalls % cycleLength) = 85 :=
by sorry

theorem color_of_158th_ball :
  let positionInCycle := (158 - 1) % cycleLength + 1
  positionInCycle ≤ redBallsPerCycle := by sorry

end red_balls_in_total_color_of_158th_ball_l59_59069


namespace three_liters_to_pints_l59_59245

def conversion_rate (liters pints : ℝ) : ℝ := pints / liters

theorem three_liters_to_pints : 
  (conversion_rate 0.75 1.58) * 3 ≈ 6.3 :=
begin
  sorry
end

end three_liters_to_pints_l59_59245


namespace count_nonempty_valid_subsets_eq_l59_59685

open Finset

namespace ProofProblem

def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def S (A : Finset ℕ) : ℕ := A.sum id

def valid_subset (A : Finset ℕ) : Prop :=
  A ≠ ∅ ∧ 3 ∣ S A ∧ ¬ 5 ∣ S A

noncomputable def count_valid_subsets (T : Finset ℕ) : ℕ :=
  (powerset T).filter valid_subset).card

theorem count_nonempty_valid_subsets_eq : count_valid_subsets T = 70 :=
  sorry

end ProofProblem

end count_nonempty_valid_subsets_eq_l59_59685


namespace minimum_value_of_sum_l59_59673

theorem minimum_value_of_sum (x : Fin 50 → ℝ) 
  (hx_pos : ∀ i, 0 < x i) 
  (hx_sum : (∑ i, (x i)^2) = 1 / 2) : 
  (∑ i, x i / (1 - (x i) ^ 2)) ≥ (3 * sqrt 3) / 4 :=
sorry

end minimum_value_of_sum_l59_59673


namespace sum_r_s_l59_59053

noncomputable def line := set_of (λ p : ℝ × ℝ, p.snd = -0.5 * p.fst + 8)

def point_P := (16, 0 : ℝ × ℝ)
def point_Q := (0, 8 : ℝ × ℝ)

def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem sum_r_s (r s : ℝ) (T : ℝ × ℝ := (r, s))
  (hT_on_line : T ∈ line)
  (h_area : area_triangle point_P point_Q (0, 0) = 4 * area_triangle point_P (r, s) (0, 0))
  : r + s = 14 := by
  sorry

end sum_r_s_l59_59053


namespace triangle_side_range_l59_59405

theorem triangle_side_range (x : ℝ) (h1 : x > 0) (h2 : x + (x + 1) + (x + 2) ≤ 12) :
  1 < x ∧ x ≤ 3 :=
by
  sorry

end triangle_side_range_l59_59405


namespace exists_splittable_set_of_6_points_not_all_sets_of_7_points_are_splittable_sets_of_6_l59_59084

open Set Function

/-- Part (a): Prove that there exists a set of 6 points in the plane that is splittable. -/
theorem exists_splittable_set_of_6_points : 
  ∃ (A B C D E F : ℝ × ℝ), 
    let centroid (X Y Z : ℝ × ℝ) := (X + Y + Z) / 3
    in centroid A B C = centroid D E F :=
sorry

/-- Part (b): Prove that any set of 7 points in the plane has a subset of 6 points that is not splittable. -/
theorem not_all_sets_of_7_points_are_splittable_sets_of_6 : 
  ∀ (P : Fin 7 → ℝ × ℝ), 
    ∃ (Q : Fin 6 → ℝ × ℝ), 
      let centroid (X Y Z : ℝ × ℝ) := (X + Y + Z) / 3
      in ¬ (∃ (a b c d e f : Fin 6), 
            (centroid (P a) (P b) (P c) = centroid (P d) (P e) (P f))) :=
sorry

end exists_splittable_set_of_6_points_not_all_sets_of_7_points_are_splittable_sets_of_6_l59_59084


namespace basketball_team_starters_l59_59015

theorem basketball_team_starters :
  let total_players := 12;
  let twins := 2;
  let other_players := total_players - twins;
  let select_from_others := Nat.choose other_players 5;
  let select_one_twin := 2 * Nat.choose other_players 4;
  select_from_others + select_one_twin = 672 :=
by
  let total_players := 12;
  let twins := 2;
  let other_players := total_players - twins;
  let select_from_others := Nat.choose other_players 5;
  let select_one_twin := 2 * Nat.choose other_players 4;
  calc
    select_from_others + select_one_twin
      = Nat.choose 10 5 + 2 * Nat.choose 10 4 : by 
        rw [select_from_others, select_one_twin, Nat.choose, Nat.choose]
      ... = 252 + 2 * 210 : by 
        norm_num
      ... = 252 + 420 : by 
        norm_num
      ... = 672 : by 
        norm_num

end basketball_team_starters_l59_59015


namespace least_time_correct_sum_of_digits_sum_of_least_time_digits_l59_59765

def horses : List ℕ := [3, 4, 5, 6, 7, 8, 9, 10]

noncomputable def least_time : ℕ :=
if h : ∃ T > 0, ∃ S ⊆ horses, S.length ≥ 4 ∧ ∀ s ∈ S, T % s = 0 then
  classical.some h
else
  0

theorem least_time_correct : least_time = 24 :=
sorry

theorem sum_of_digits (n : ℕ) : ℕ :=
(n.digits 10).sum

theorem sum_of_least_time_digits : sum_of_digits least_time = 6 :=
sorry

end least_time_correct_sum_of_digits_sum_of_least_time_digits_l59_59765


namespace reese_practice_hours_l59_59367

-- Define the average number of weeks in a month
def avg_weeks_per_month : ℝ := 4.345

-- Define the number of hours Reese practices per week
def hours_per_week : ℝ := 4 

-- Define the number of months under consideration
def num_months : ℝ := 5

-- Calculate the total hours Reese will practice after five months
theorem reese_practice_hours :
  (num_months * avg_weeks_per_month * hours_per_week) = 86.9 :=
by
  -- We'll skip the proof part by adding sorry here
  sorry

end reese_practice_hours_l59_59367


namespace find_f_9_l59_59670

def f : ℤ → ℤ
| x := if x ≥ 10 then x - 2 else f (x + 6)

theorem find_f_9 : f 9 = 13 := by
  sorry

end find_f_9_l59_59670


namespace first_caller_to_win_all_prizes_is_900_l59_59789

-- Define the conditions: frequencies of win types
def every_25th_caller_wins_music_player (n : ℕ) : Prop := n % 25 = 0
def every_36th_caller_wins_concert_tickets (n : ℕ) : Prop := n % 36 = 0
def every_45th_caller_wins_backstage_passes (n : ℕ) : Prop := n % 45 = 0

-- Formalize the problem to prove
theorem first_caller_to_win_all_prizes_is_900 :
  ∃ n : ℕ, every_25th_caller_wins_music_player n ∧
           every_36th_caller_wins_concert_tickets n ∧
           every_45th_caller_wins_backstage_passes n ∧
           n = 900 :=
by {
  sorry
}

end first_caller_to_win_all_prizes_is_900_l59_59789


namespace set_has_one_element_iff_double_root_l59_59284

theorem set_has_one_element_iff_double_root (k : ℝ) :
  (∃ x, ∀ y, y^2 - k*y + 1 = 0 ↔ y = x) ↔ k = 2 ∨ k = -2 :=
by
  sorry

end set_has_one_element_iff_double_root_l59_59284


namespace find_other_root_l59_59249

theorem find_other_root (a b : ℝ) (h₁ : 3^2 + 3 * a - 2 * a = 0) (h₂ : ∀ x, x^2 + a * x - 2 * a = 0 → (x = 3 ∨ x = b)) :
  b = 6 := 
sorry

end find_other_root_l59_59249


namespace workByFortyPercent_l59_59294

-- Definitions based on the conditions
def totalWork : ℝ := 100.0
def productiveEmployees1 : ℝ := 20.0 / 100.0 * totalWork
def workByProductiveEmployees1 : ℝ := 80.0 / 100.0 * totalWork

-- Helper definitions to split the next 20% of most productive employees
def remainingEmployees : ℝ := 100.0 - productiveEmployees1
def nextProductiveEmployees2 : ℝ := 20.0 / 100.0 * remainingEmployees
def workByNextProductiveEmployees2 : ℝ := nextProductiveEmployees2 / remainingEmployees * (totalWork - workByProductiveEmployees1)

-- Theorem we need to prove
theorem workByFortyPercent : (workByProductiveEmployees1 + workByNextProductiveEmployees2) = 85 := by
  sorry

end workByFortyPercent_l59_59294


namespace A_left_time_correct_l59_59768

-- Definitions of conditions
def time_to_complete_A : ℕ := 6
def time_to_complete_B : ℕ := 8
def time_to_complete_C : ℕ := 10
def start_time : ℕ := 8
def end_time : ℕ := 12

-- Definition of the problem in terms of proving the time A left
noncomputable def time_A_left : ℕ × ℕ :=
  let working_hours := end_time - start_time
  let work_done_by_B_C := working_hours * (1 / time_to_complete_B + 1 / time_to_complete_C)
  let remaining_work := 1 - work_done_by_B_C
  let time_A_worked := remaining_work * time_to_complete_A
  let time_A_left := (start_time : ℕ) + (time_A_worked.toRat).num / (time_A_worked.toRat).denom
  let hour := (time_A_left.floor : ℕ)
  let minute := ((time_A_left - hour) * 60).floor : ℕ
  (hour, minute)

-- The proof problem statement
theorem A_left_time_correct : time_A_left = (8, 36) := sorry

end A_left_time_correct_l59_59768


namespace salaries_of_a_and_b_l59_59115

theorem salaries_of_a_and_b {x y : ℝ}
  (h1 : x + y = 5000)
  (h2 : 0.05 * x = 0.15 * y) :
  x = 3750 :=
by sorry

end salaries_of_a_and_b_l59_59115


namespace joe_tests_number_l59_59653

theorem joe_tests_number (n : ℕ) (S : ℕ) 
  (h1 : S = 50 * n)
  (h2 : (S - 35) = 55 * (n - 1)) : 
  n = 4 := 
begin 
  sorry 
end

end joe_tests_number_l59_59653


namespace angle_between_a_and_a_plus_b_l59_59970

noncomputable def angle_between (a b : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥))

theorem angle_between_a_and_a_plus_b
  (a b : EuclideanSpace ℝ (Fin 2))
  (h₁ : ∥a∥ = ∥b∥)
  (h₂ : ∥a∥ = ∥a - b∥) :
  angle_between a (a + b) = real.pi / 6 := 
sorry

end angle_between_a_and_a_plus_b_l59_59970


namespace possible_values_of_K_l59_59287

theorem possible_values_of_K (K N : ℕ) (h1 : K * (K + 1) = 2 * N^2) (h2 : N < 100) :
  K = 1 ∨ K = 8 ∨ K = 49 :=
sorry

end possible_values_of_K_l59_59287


namespace distance_between_cars_l59_59415

open Real

noncomputable def speed_of_first_car := 100 -- km/h
noncomputable def speed_factor := 1.25
noncomputable def time_to_meet := 4 -- hours

theorem distance_between_cars :
  ∃ (v : ℝ), (speed_factor * v = speed_of_first_car) ∧
                 (let speed_of_second_car := v in
                  let relative_speed := speed_of_first_car + speed_of_second_car in
                  let distance := relative_speed * time_to_meet in
                  distance = 720) :=
sorry

end distance_between_cars_l59_59415


namespace period_of_f_l59_59945

constant a : ℝ

def f : ℝ → ℝ := sorry

axiom f_def : ∀ x, f x = (f (x - a) - 1) / (f (x - a) + 1)

theorem period_of_f : ∃ T, ∀ x, f x = f (x + T) ∧ T = 4 * a :=
by
  use 4 * a
  sorry

end period_of_f_l59_59945


namespace gift_combinations_l59_59456

theorem gift_combinations (wrapping_paper_count ribbon_count card_count : ℕ)
  (restricted_wrapping : ℕ)
  (restricted_ribbon : ℕ)
  (total_combinations := wrapping_paper_count * ribbon_count * card_count)
  (invalid_combinations := card_count)
  (valid_combinations := total_combinations - invalid_combinations) :
  wrapping_paper_count = 10 →
  ribbon_count = 4 →
  card_count = 5 →
  restricted_wrapping = 10 →
  restricted_ribbon = 1 →
  valid_combinations = 195 :=
by
  intros
  sorry

end gift_combinations_l59_59456


namespace first_player_score_at_least_55_l59_59081

theorem first_player_score_at_least_55 : 
  (∀ (f s : Finset ℕ), (f ∪ s = Finset.range 1 102) → (f.card = 99) → 
   (∀ x ∈ f ∪ s, ∃ y, (y ∈ f ∪ s ∧ (| x - y | = 55))) →
   ∃ x y, x ∈ f ∪ s ∧ y ∈ f ∪ s ∧ |x - y| = 55) :=
sorry

end first_player_score_at_least_55_l59_59081


namespace probability_flies_swept_by_minute_hand_l59_59002

theorem probability_flies_swept_by_minute_hand :
  let flies_positions := {12, 3, 6, 9}
  -- Define the favorable starting intervals for the 20-minute sweep.
  let favorable_intervals := [(55, 60), (20, 25), (35, 40), (50, 55)]
  -- Total possible minutes in an hour
  let total_minutes := 60
  -- Total favorable minutes
  let favorable_minutes := 20
  -- Calculate the probability
  (favorable_minutes / total_minutes : ℝ) = (1 / 3 : ℝ):=
by
  sorry

end probability_flies_swept_by_minute_hand_l59_59002


namespace subtraction_of_fractions_l59_59086

theorem subtraction_of_fractions :
  1 + 1 / 2 - 3 / 5 = 9 / 10 := by
  sorry

end subtraction_of_fractions_l59_59086


namespace sum_infinite_series_l59_59880

theorem sum_infinite_series : ∑' k : ℕ, (k^2 : ℝ) / (3^k) = 7 / 8 :=
sorry

end sum_infinite_series_l59_59880


namespace arithmetic_seq_general_formula_sum_of_abs_sequence_l59_59583

noncomputable def a_n (n : ℕ) : ℤ := -7 + 3 * n

noncomputable def S_n (n : ℕ) : ℚ :=
  if n = 1 then 4
  else if n = 2 then 5
  else (3/2) * n^2 - (11/2) * n + 10

theorem arithmetic_seq_general_formula :
  ∀ {a : ℤ} {d : ℤ}, (d > 0) → 
    (a + (a + d) + (a + 2 * d) = -3) → 
    (a * (a + d) * (a + 2 * d) = 8) → 
    ∃ a_n : ℕ → ℤ, (∀ n : ℕ, a_n n = -7 + 3 * n) :=
by
  sorry

theorem sum_of_abs_sequence :
  ∀ {a : ℤ} {d : ℤ}, (d > 0) → 
    (a + (a + d) + (a + 2 * d) = -3) → 
    (a * (a + d) * (a + 2 * d) = 8) → 
    ∃ S_n : ℕ → ℚ, (∀ n : ℕ, S_n n = 
      if n = 1 then 4 
      else if n = 2 then 5 
      else (3/2) * n^2 - (11/2) * n + 10) :=
by
  sorry

end arithmetic_seq_general_formula_sum_of_abs_sequence_l59_59583


namespace sin_cos_difference_l59_59986

theorem sin_cos_difference (θ : ℝ) (a : ℝ) (h1 : θ ∈ (0, π)) (h2 : sin θ * cos θ = a) 
    (h3 : sin θ + cos θ = 1) : sin θ - cos θ = 1 :=
by
  sorry

end sin_cos_difference_l59_59986


namespace count_solid_circles_l59_59837

theorem count_solid_circles (total_circles : Nat) (circle_seq_pattern : Nat → Nat) : 
  (∑ n in Finset.range total_circles, if circle_seq_pattern n then 1 else 0) = 60 :=
by
  -- Define circle sequence pattern properties
  let circle_seq_pattern := λ n, n % (n + 2) = (n + 1) // 2

  -- Sum of the groups up to the given total circle count (from problem condition)
  let total_circles := 2010

  -- Solid circles determined by end of groups, ensuring groups up to 60 add up within 2010 circle count
  sorry

end count_solid_circles_l59_59837


namespace min_value_of_n_l59_59308

theorem min_value_of_n (n : ℕ) (k : ℚ) (h1 : k > 0.9999) 
    (h2 : 4 * n * (n - 1) * (1 - k) = 1) : 
    n = 51 :=
sorry

end min_value_of_n_l59_59308


namespace a9_proof_l59_59972

variable {a : ℕ → ℝ}

-- Conditions
axiom a1 : a 1 = 1
axiom an_recurrence : ∀ n > 1, a n = (a (n - 1)) * 2^(n - 1)

-- Goal
theorem a9_proof : a 9 = 2^36 := 
by 
  sorry

end a9_proof_l59_59972


namespace train_speed_is_40_kmh_l59_59838

noncomputable def speed_of_train (train_length_m : ℝ) 
                                   (man_speed_kmh : ℝ) 
                                   (pass_time_s : ℝ) : ℝ :=
  let train_length_km := train_length_m / 1000
  let pass_time_h := pass_time_s / 3600
  let relative_speed_kmh := train_length_km / pass_time_h
  relative_speed_kmh - man_speed_kmh
  
theorem train_speed_is_40_kmh :
  speed_of_train 110 4 9 = 40 := 
by
  sorry

end train_speed_is_40_kmh_l59_59838


namespace sum_of_nine_numbers_l59_59023

theorem sum_of_nine_numbers (x : ℝ) :
  ∃ (a : fin 9 → ℝ), (∀ i, ∀ d, d ≠ 0 ∧ d ≠ 7 → ¬(d ∈ digits 10 (a i))) ∧ (finset.univ.sum a = x) :=
by
  sorry

end sum_of_nine_numbers_l59_59023


namespace total_length_of_arcs_l59_59059

-- Define the problem and given conditions
def Triangle (X Y Z : Type) := sorry
def Circumcircle (X Y Z : Type) (r : ℝ) := sorry
def PerpendicularBisectorsIntersect (X' Y' Z': Type) (circumcircle : Circumcircle) := sorry

-- Given conditions
def triangle_XYZ (XYZ : Triangle) : Prop := sorry
noncomputable def perimeter (XYZ : Triangle) : ℝ := 24
noncomputable def radius (circumcircle : Circumcircle) : ℝ := 5
def intersect_points (X' Y' Z' : Type) : Prop := sorry

-- Proof problem
theorem total_length_of_arcs (XYZ : Triangle) (circumcircle : Circumcircle) (X' Y' Z' : Type)
  (h_triang: triangle_XYZ XYZ)
  (h_radius: radius circumcircle = 5)
  (h_perimeter: perimeter XYZ = 24)
  (h_intersect: PerpendicularBisectorsIntersect X' Y' Z' circumcircle) :
  (total_length_of_arcs XYZ circumcircle X' Y' Z') = 30 * π :=
sorry

end total_length_of_arcs_l59_59059


namespace cyclic_quadrilateral_fourth_side_length_l59_59826

theorem cyclic_quadrilateral_fourth_side_length :
  ∃ (x : ℝ), x = 189.21 ∧ 
  ∃ (r : ℝ), 
  (r = 100 * Real.sqrt 3) ∧ 
  ∃ (AB BC CD : ℝ), 
  (AB = 100) ∧ 
  (BC = 150) ∧ 
  (CD = 200) ∧ 
  (AB + BC + CD > r) := 
by
  -- Conditions given in the problem
  let r := 100 * Real.sqrt 3
  let AB := 100
  let BC := 150
  let CD := 200
  have h1 : r = 100 * Real.sqrt 3 := rfl
  have h2 : AB = 100 := rfl
  have h3 : BC = 150 := rfl
  have h4 : CD = 200 := rfl
  -- Correct Answer
  use 189.21,
  split,
  { exact rfl },
  existsi r,
  split,
  { exact h1 },
  existsi AB,
  existsi BC,
  existsi CD,
  exact ⟨h2, h3, h4, sorry⟩ -- We need some non-trivial geometry to solve the last part

end cyclic_quadrilateral_fourth_side_length_l59_59826


namespace apple_cost_l59_59164

theorem apple_cost (rate_cost : ℕ) (rate_weight total_weight : ℕ) (h_rate : rate_cost = 5) (h_weight : rate_weight = 7) (h_total : total_weight = 21) :
  ∃ total_cost : ℕ, total_cost = 15 :=
by
  -- The proof will go here
  sorry

end apple_cost_l59_59164


namespace arnaldo_winning_strategy_l59_59326

-- Define the size of the table
def table_size : ℕ := 2020

-- Define what it means for a player to win
def arnaldo_wins (k : ℕ) (board : array (array ℕ)) : Prop :=
  ∃ row, (∃ i, i + k ≤ table_size ∧ ∀ j, j < k → board[row][i + j] = 1) ∨
  ∃ col, (∃ i, i + k ≤ table_size ∧ ∀ j, j < k → board[i + j][col] = 1)

-- Define the game condition
structure game_state : Type :=
(table : array (array ℕ))
(current_player : ℕ)
(empty_cells : list (ℕ × ℕ))
(arnaldo_wins : Prop)
(bernaldo_wins : Prop)

-- Define the problem statement
theorem arnaldo_winning_strategy (k : ℕ) : (k ≤ 1011) ↔ ∃ strat, ∀ gs : game_state, gs.current_player = 0 → strat gs :=
sorry

end arnaldo_winning_strategy_l59_59326


namespace conjugate_of_z_l59_59563

-- Definitions
def z : ℂ := (5 - Complex.i) / (1 + Complex.i)
def conj_z : ℂ := Complex.conj z

-- Theorem statement
theorem conjugate_of_z : conj_z = 2 + 3 * Complex.i := 
by
  sorry

end conjugate_of_z_l59_59563


namespace train_length_proof_l59_59475

/-
Problem: Prove that the length of the train is 145 meters 
given that the train is traveling at 45 km/hr,
crosses a bridge in 30 seconds, and the length of the bridge is 230 meters.
-/

def train_speed : ℕ := 45 -- speed in km/hr
def cross_time : ℕ := 30 -- time in seconds
def bridge_length : ℕ := 230 -- length in meters

theorem train_length_proof :
  ∃ length_of_train : ℕ, length_of_train = 145 :=
by
  let train_speed_m_s := 45 * 1000 / 3600
  let total_distance := train_speed_m_s * cross_time
  let train_length := total_distance - 230
  have h_train_speed : train_speed_m_s = 12.5 := sorry
  have h_total_distance : total_distance = 375 := sorry
  have h_train_length : train_length = 145 := sorry
  exact ⟨145, h_train_length⟩

end train_length_proof_l59_59475


namespace certain_number_div_5000_l59_59094

theorem certain_number_div_5000 (num : ℝ) (h : num / 5000 = 0.0114) : num = 57 :=
sorry

end certain_number_div_5000_l59_59094


namespace foci_of_ellipse_l59_59271

-- Given the standard equation of the ellipse x²/10 + y² = 1, prove the coordinates of the foci are (3,0) and (-3,0)
theorem foci_of_ellipse :
  let a² := 10 in
  let b² := 1 in
  ∃ (c : ℝ), c = 3 ∧ (c = real.sqrt (a² - b²)) →
    ( (3, 0) ∨ (-3, 0) ) := by
  let a_sq := 10
  let b_sq := 1
  have h1 : a_sq = 10 := rfl
  have h2 : b_sq = 1 := rfl
  have h3 : c = real.sqrt (a_sq - b_sq) := sorry
  have h4 : c = 3 := by
    calc
      c = real.sqrt (10 - 1) : sorry
    ... = 3 : sorry
  use 3
  constructor
  · exact h4
  · sorry

end foci_of_ellipse_l59_59271


namespace min_value_of_f_perimeter_of_ABC_l59_59262

-- Condition: the function f(x)
def f (x : Real) := Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

-- The first part: Proving the minimum value of f(x)
theorem min_value_of_f : ∃ x : Real, f x = -1 / 2 := 
sorry

-- Condition: information about triangle ABC
variables (a b c : Real) -- sides of the triangle
variable (C : Real) -- the angle opposite side c

-- Condition: given properties
axiom f_C : f C = 1
axiom area_ABC : (Real.sqrt 3) * (a * b * Real.sin C) / 4 = (3 * Real.sqrt 3) / 4
axiom c_eq_sqrt_7 : c = Real.sqrt 7

-- Auxiliary definitions
def perimeter (a b c : Real) := a + b + c

-- The second part: Proving the perimeter of the triangle
theorem perimeter_of_ABC : perimeter a b c = 4 + Real.sqrt 7 := 
sorry

end min_value_of_f_perimeter_of_ABC_l59_59262


namespace three_pipes_time_l59_59411

variable (R : ℝ) (T : ℝ)

-- Condition: Two pipes fill the tank in 18 hours
def two_pipes_fill : Prop := 2 * R * 18 = 1

-- Question: How long does it take for three pipes to fill the tank?
def three_pipes_fill : Prop := 3 * R * T = 1

theorem three_pipes_time (h : two_pipes_fill R) : three_pipes_fill R 12 :=
by
  sorry

end three_pipes_time_l59_59411


namespace probability_flies_swept_by_minute_hand_l59_59003

theorem probability_flies_swept_by_minute_hand :
  let flies_positions := {12, 3, 6, 9}
  -- Define the favorable starting intervals for the 20-minute sweep.
  let favorable_intervals := [(55, 60), (20, 25), (35, 40), (50, 55)]
  -- Total possible minutes in an hour
  let total_minutes := 60
  -- Total favorable minutes
  let favorable_minutes := 20
  -- Calculate the probability
  (favorable_minutes / total_minutes : ℝ) = (1 / 3 : ℝ):=
by
  sorry

end probability_flies_swept_by_minute_hand_l59_59003


namespace class_representation_l59_59992

def represent_class (grade class : ℕ) : ℕ × ℕ := (grade, class)

theorem class_representation :
  ∀ grade7_class8 grade8_class7: (ℕ × ℕ),
  grade7_class8 = (7, 8) →
  grade8_class7 = represent_class 8 7 →
  grade8_class7 = (8, 7) :=
by 
  intros grade7_class8 grade8_class7 h1 h2
  sorry

end class_representation_l59_59992


namespace remainder_of_sum_of_cubes_mod_13_l59_59782

theorem remainder_of_sum_of_cubes_mod_13 :
  (∑ i in Finset.range 16, i^3) % 13 = 5 :=
sorry

end remainder_of_sum_of_cubes_mod_13_l59_59782


namespace five_coins_total_cannot_be_30_cents_l59_59912

theorem five_coins_total_cannot_be_30_cents :
  ¬ ∃ (a b c d e : ℕ), 
  a + b + c + d + e = 5 ∧ 
  (a * 1 + b * 5 + c * 10 + d * 25 + e * 50) = 30 := 
sorry

end five_coins_total_cannot_be_30_cents_l59_59912


namespace tank_capacity_l59_59137

variable (c w : ℕ)

-- Conditions
def initial_fraction (w c : ℕ) : Prop := w = c / 7
def final_fraction (w c : ℕ) : Prop := (w + 2) = c / 5

-- The theorem statement
theorem tank_capacity : 
  initial_fraction w c → 
  final_fraction w c → 
  c = 35 := 
by
  sorry  -- indicates that the proof is not provided

end tank_capacity_l59_59137


namespace num_divisors_64n5_l59_59915

theorem num_divisors_64n5 (n : ℕ) (h₁ : n > 0) (h₂ : (∏ d in (finset.divisors (210 * n^3)), 1) = 210) :
  ∏ d in (finset.divisors (64 * n^5)), 1 = 22627 := sorry

end num_divisors_64n5_l59_59915


namespace number_of_Qs_l59_59664

noncomputable def P (x : ℝ) := (x - 1) * (x - 2) * (x - 3) * (x - 4)

theorem number_of_Qs : 
  ∃ Q : polynomial ℝ, Q.degree = 3 ∧ 
  (∃ R : polynomial ℝ, R.degree = 4 ∧ P (Q x) = P x * R x) ∧ 
  ∑ x ∈ {1, 2, 3, 4}, (Q x) ∈ Finset.univ.attach = 228 :=
sorry

end number_of_Qs_l59_59664


namespace reese_practice_hours_l59_59365

-- Define the average number of weeks in a month
def avg_weeks_per_month : ℝ := 4.345

-- Define the number of hours Reese practices per week
def hours_per_week : ℝ := 4 

-- Define the number of months under consideration
def num_months : ℝ := 5

-- Calculate the total hours Reese will practice after five months
theorem reese_practice_hours :
  (num_months * avg_weeks_per_month * hours_per_week) = 86.9 :=
by
  -- We'll skip the proof part by adding sorry here
  sorry

end reese_practice_hours_l59_59365


namespace fraction_walk_is_three_twentieths_l59_59172

-- Define the various fractions given in the conditions
def fraction_bus : ℚ := 1 / 2
def fraction_auto : ℚ := 1 / 4
def fraction_bicycle : ℚ := 1 / 10

-- Defining the total fraction for students that do not walk
def total_not_walk : ℚ := fraction_bus + fraction_auto + fraction_bicycle

-- The remaining fraction after subtracting from 1
def fraction_walk : ℚ := 1 - total_not_walk

-- The theorem we want to prove that fraction_walk is 3/20
theorem fraction_walk_is_three_twentieths : fraction_walk = 3 / 20 := by
  sorry

end fraction_walk_is_three_twentieths_l59_59172


namespace scaled_variance_l59_59290

open_locale classical

variables {α : Type*} [add_comm_group α] [vector_space ℝ α] [finite_dimensional ℝ α]

noncomputable def variance (s : finset ℕ) (f : ℕ → ℝ) : ℝ :=
  (s.sum (λ x, (f x - (s.sum f / s.card))^2)) / s.card

theorem scaled_variance (s : finset ℕ) (f : ℕ → ℝ)
  (hvar : variance s f = 1) :
  variance s (λ x, 2 * f x + 4) = 4 :=
begin
  sorry
end

end scaled_variance_l59_59290


namespace only_root_is_4_l59_59743

noncomputable def equation_one (x : ℝ) : ℝ := (2 * x^2) / (x - 1) - (2 * x + 7) / 3 + (4 - 6 * x) / (x - 1) + 1

noncomputable def equation_two (x : ℝ) : ℝ := x^2 - 5 * x + 4

theorem only_root_is_4 (x : ℝ) (h: equation_one x = 0) (h_transformation: equation_two x = 0) : x = 4 := sorry

end only_root_is_4_l59_59743


namespace shortest_path_length_l59_59301

-- Define the given conditions
variables (a : ℝ)

-- Define a pseudonym for the pyramid structure, regular and with given conditions
structure RegularHexagonalPyramid :=
(vertex : Type)
(base_vertices : Π (i : Fin 6), vertex)
(slanted_edges_length : vertex → vertex → ℝ)
(dihedral_angle : ℝ)
(pyramid_condition : ∀ (i : Fin 6), slanted_edges_length (base_vertices i) vertex = a ∧ dihedral_angle = 10)

-- Problem statement: Prove the shortest path length
theorem shortest_path_length 
  (S : RegularHexagonalPyramid) 
  (A : S.vertex) 
  : 
  ∃ path_length : ℝ, path_length = a 
  :=
sorry

end shortest_path_length_l59_59301


namespace solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l59_59726

-- Equation (1)
theorem solve_quadratic_eq1 (x : ℝ) : x^2 + 16 = 8*x ↔ x = 4 := by
  sorry

-- Equation (2)
theorem solve_quadratic_eq2 (x : ℝ) : 2*x^2 + 4*x - 3 = 0 ↔ 
  x = -1 + (Real.sqrt 10) / 2 ∨ x = -1 - (Real.sqrt 10) / 2 := by
  sorry

-- Equation (3)
theorem solve_quadratic_eq3 (x : ℝ) : x*(x - 1) = x ↔ x = 0 ∨ x = 2 := by
  sorry

-- Equation (4)
theorem solve_quadratic_eq4 (x : ℝ) : x*(x + 4) = 8*x - 3 ↔ x = 3 ∨ x = 1 := by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l59_59726


namespace negation_of_all_fractions_are_rational_l59_59395

-- Define the universal proposition: All fractions are rational numbers.
def all_fractions_are_rational := ∀ f : ℚ, true

-- The negation of the universal proposition.
def some_fractions_are_not_rational :=
  ∃ f : ℚ, false

theorem negation_of_all_fractions_are_rational :
  ¬ all_fractions_are_rational ↔ some_fractions_are_not_rational :=
by
  sorry

end negation_of_all_fractions_are_rational_l59_59395


namespace probability_of_event_l59_59774

def is_uniform (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1

theorem probability_of_event : 
  ∀ (a : ℝ), is_uniform a → ∀ (p : ℚ), (3 * a - 1 > 0) → p = 2 / 3 → 
  (∃ b, 0 ≤ b ∧ b ≤ 1 ∧ 3 * b - 1 > 0) := 
by
  intro a h_uniform p h_event h_prob
  sorry

end probability_of_event_l59_59774


namespace symmetric_line_equation_l59_59047

theorem symmetric_line_equation 
  (l1 : ∀ x y : ℝ, x - 2 * y - 2 = 0) 
  (l2 : ∀ x y : ℝ, x + y = 0) : 
  ∀ x y : ℝ, 2 * x - y - 2 = 0 :=
sorry

end symmetric_line_equation_l59_59047


namespace find_x_value_l59_59217

theorem find_x_value (x : ℝ) : 9^(-3) = (3^(72/x)) / (3^(42/x) * (9^(25/x))) → x = 10/3 :=
by
  intros h
  sorry

end find_x_value_l59_59217


namespace expected_games_to_win_two_consecutive_l59_59772

-- Conditions
def probability_win_single_game : ℚ := 1 / 3  -- Probability of A winning a single game
def probability_not_win_single_game : ℚ := 2 / 3  -- Probability of A not winning a single game

-- Expected number of games A will play until winning 2 consecutive games
noncomputable def expected_games_until_two_consecutive_wins : ℚ := 12

-- Theorem statement
theorem expected_games_to_win_two_consecutive :
  let E : ℚ := 12
  in E = (2 / 3) * (E + 1) + (2 / 9) * (E + 2) + (1 / 9) * 2 :=
by
  sorry

end expected_games_to_win_two_consecutive_l59_59772


namespace time_saved_1200_miles_l59_59700

theorem time_saved_1200_miles
  (distance : ℕ)
  (speed1 speed2 : ℕ)
  (h_distance : distance = 1200)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 50) :
  (distance / speed2) - (distance / speed1) = 4 :=
by
  sorry

end time_saved_1200_miles_l59_59700


namespace infinite_solutions_l59_59903

theorem infinite_solutions (x y : ℝ) 
    (h : 32^(x^2 + y) + 32^(x + y^2) ≥ 2) : 
    ∃∞ (x y : ℝ), 32^(x^2 + y) + 32^(x + y^2) ≥ 2 := sorry

end infinite_solutions_l59_59903


namespace smallest_yellow_marbles_l59_59842

-- Define the conditions given in the problem
variables {n : ℕ}
def blue_marbles := n / 5
def red_marbles := n / 6
def green_marbles := n / 10
def yellow_marbles := n - (blue_marbles + red_marbles + green_marbles)

-- State the theorem to be proved
theorem smallest_yellow_marbles (h : ∃ k : ℕ, n = 15 * k) : yellow_marbles n = 16 :=
  sorry

end smallest_yellow_marbles_l59_59842


namespace investment_time_q_l59_59060

/-- The ratio of investments of two partners P and Q is 7:5, 
  the ratio of their profits is 7:14, and P invested the money for 5 months.
  Prove that Q invested the money for 14 months. --/
theorem investment_time_q
    (x : ℝ)  -- common multiple of their investments
    (t : ℝ)  -- time in months Q invested the money
    (h1 : 7 * x / (5 * x * t) = 1 / 2) -- Condition derived from given profit ratio and investment-time relationship
    : t = 14 := 
begin
  -- Proof goes here
  sorry
end

end investment_time_q_l59_59060


namespace twenty_fourth_digit_sum_l59_59777

theorem twenty_fourth_digit_sum (a b : ℚ) (h₁ : a = 1/7) (h₂ : b = 1/9) : 
  (Nat.digits 10 (Rat.mkPnat (a + b - (a + b).floor)).numerator.digits.full 24) = 8 :=
by
  sorry

end twenty_fourth_digit_sum_l59_59777


namespace optimal_metro_station_placement_l59_59036

-- Definitions of the cells
def cell (x y : Nat) := (x, y)

-- Conditions
def is_grid_9x9 := 9 = 9
def travel_time_between_metro_stations := 10
def max_travel_time := 120

-- Positions of the metro stations
def metro_station_1 := cell 1 5
def metro_station_2 := cell 9 5

-- Main statement
theorem optimal_metro_station_placement (x1 y1 x2 y2 : Nat) (h1 : cell x1 y1 ∈ (finset.range 9).product (finset.range 9)) (h2 : cell x2 y2 ∈ (finset.range 9).product (finset.range 9))
: travel_time_between_metro_stations ≤ max_travel_time / 12 :=
by
  intros
  sorry

end optimal_metro_station_placement_l59_59036


namespace set_of_points_satisfying_inequality_l59_59714

theorem set_of_points_satisfying_inequality :
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ |x| ≤ 1 ∧ |y| ≤ 1 ∧ xy ≤ 0} ∪
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x^2 + y^2 ≤ 1 ∧ xy > 0} =
  {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ |x| ≤ 1 ∧ |y| ≤ 1 ∧ sqrt(1 - x^2) * sqrt(1 - y^2) ≥ xy} :=
sorry

end set_of_points_satisfying_inequality_l59_59714


namespace third_discount_l59_59132

noncomputable def find_discount (P S firstDiscount secondDiscount D3 : ℝ) : Prop :=
  S = P * (1 - firstDiscount / 100) * (1 - secondDiscount / 100) * (1 - D3 / 100)

theorem third_discount (P : ℝ) (S : ℝ) (firstDiscount : ℝ) (secondDiscount : ℝ) (D3 : ℝ) 
  (HP : P = 9649.12) (HS : S = 6600)
  (HfirstDiscount : firstDiscount = 20) (HsecondDiscount : secondDiscount = 10) : 
  find_discount P S firstDiscount secondDiscount 5.01 :=
  by
  rw [HP, HS, HfirstDiscount, HsecondDiscount]
  sorry

end third_discount_l59_59132


namespace problem_1_problem_2_l59_59975

noncomputable def a_n (n : ℕ) : ℝ := n + (-1 : ℝ)^n * n^2
def b_n (n : ℕ) : ℝ := n

theorem problem_1 (h1 : ∀ n, b_n n = a_n n - (-1 : ℝ)^n * (n : ℝ)^2)
(h2 : a_n 1 + b_n 1 = 1)
(h3 : a_n 2 + b_n 2 = 8)
(h4 : arithmetic_seq (b_n : ℕ → ℝ)) :
∀ n, a_n n = n + (-1 : ℝ)^n * (n : ℝ)^2 := by
  sorry

theorem problem_2 (h1 : ∀ n, b_n n = a_n n - (-1 : ℝ)^n * (n : ℝ)^2)
(h2 : a_n 1 + b_n 1 = 1)
(h3 : a_n 2 + b_n 2 = 8)
(h4 : arithmetic_seq (b_n : ℕ → ℝ)) :
∑ n in finset.range 100, a_n (n + 1) = 10100 := by
  sorry

end problem_1_problem_2_l59_59975


namespace value_of_a_minus_b_l59_59280

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = - (a + b)) :
  a - b = -2 ∨ a - b = -6 := sorry

end value_of_a_minus_b_l59_59280


namespace subset_relation_l59_59601

def a := Real.sqrt 5
def M := {x : ℝ | x > 2}

theorem subset_relation : {a} ⊆ M := sorry

end subset_relation_l59_59601


namespace math_problem_l59_59561

-- Definitions of points M, N, and P
structure Point where
  x : ℝ
  y : ℝ

def M : Point := { x := -3, y := 3 }
def N : Point := { x := 1, y := -5 }
def P : Point := { x := 3, y := -1 }

-- Definition of the line the center of the circle lies on
def line_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Definition for the circle equation
def circle_eq (x y cx cy : ℝ) (r : ℝ) : Prop :=
  (x - cx)^2 + (y - cy)^2 = r^2

-- Problem statement in Lean
theorem math_problem : 
  ∃ cx cy r, line_eq cx cy ∧ circle_eq (-3) 3 cx cy r ∧ circle_eq 1 (-5) cx cy r ∧
  (∀ k > 0, (∃ x : ℝ, cx - x * k ∈ Ioo (-∞ : ℝ) (5 : ℝ)) ∧ k > 15/8) ∧
  (∃ k > 15/8, ∃ x, ∃ y, 
    (∃ line_eq cx cy → circle_eq x y cx cy r)
    ∧ ((x + k * y + k = 3)))
    := sorry

end math_problem_l59_59561


namespace number_of_distinct_linear_recurrences_l59_59679

open BigOperators

/-
  Let p be a prime positive integer.
  Define a mod-p recurrence of degree n to be a sequence {a_k}_{k >= 0} of numbers modulo p 
  satisfying a relation of the form:

  ai+n = c_n-1 ai+n-1 + ... + c_1 ai+1 + c_0 ai
  for all i >= 0, where c_0, c_1, ..., c_n-1 are integers and c_0 not equivalent to 0 mod p.
  Compute the number of distinct linear recurrences of degree at most n in terms of p and n.
-/
theorem number_of_distinct_linear_recurrences (p n : ℕ) (hp : Nat.Prime p) : 
  ∃ d : ℕ, 
    (∀ {a : ℕ → ℕ} {c : ℕ → ℕ} (h : ∀ i, a (i + n) = ∑ j in Finset.range n, c j * a (i + j))
     (hc0 : c 0 ≠ 0), 
      d = (1 - n * (p - 1) / (p + 1) + p^2 * (p^(2 * n) - 1) / (p + 1)^2 : ℚ)) :=
  sorry

end number_of_distinct_linear_recurrences_l59_59679


namespace find_fraction_l59_59787

theorem find_fraction (f : ℝ) (n : ℝ) (h : n = 180) (eqn : f * ((1 / 3) * (1 / 5) * n) + 6 = (1 / 15) * n) : f = 1 / 2 :=
by
  -- Definitions and assumptions provided above will be used here.
  sorry

end find_fraction_l59_59787


namespace line_c_intersects_l59_59976

open Plane

-- Definitions based on the given conditions
variables {α β : Plane} -- α and β are planes
variables {a b c : Line} -- a, b, and c are lines

-- Given conditions
axiom skew_lines (a b : Line) : ¬(a ∥ b) ∧ ¬(a ⊥ b) ∧ ¬(∃ p : Point, p ∈ a ∧ p ∈ b)
axiom line_in_plane (a : Line) (α : Plane) : ∃ p : Point, p ∈ a ∧ p ∈ α
axiom planes_intersect (α β : Plane) (c : Line) : ∃ p : Point, p ∈ α ∧ p ∈ β ∧ p ∈ c
axiom plane_contains_line (a : Line) (α : Plane) : a ⊆ α

-- The mathematical statement to prove
theorem line_c_intersects : ∀ (α β : Plane) (a b c : Line),
  skew_lines a b →
  line_in_plane a α →
  line_in_plane b β →
  planes_intersect α β c →
  (∃ p : Point, p ∈ c ∧ (p ∈ a ∨ p ∈ b)) :=
by
  sorry

end line_c_intersects_l59_59976


namespace final_score_proof_l59_59873

def final_score (bullseye_points : ℕ) (miss_points : ℕ) (half_bullseye_points : ℕ) : ℕ :=
  bullseye_points + miss_points + half_bullseye_points

theorem final_score_proof : final_score 50 0 25 = 75 :=
by
  -- Considering the given conditions
  -- bullseye_points = 50
  -- miss_points = 0
  -- half_bullseye_points = half of bullseye_points = 25
  -- Summing them up: 50 + 0 + 25 = 75
  sorry

end final_score_proof_l59_59873


namespace range_of_f_l59_59257

noncomputable def f (x : ℤ) : ℤ := x ^ 2 + 1

def domain : Set ℤ := {-1, 0, 1, 2}

def range_f : Set ℤ := {1, 2, 5}

theorem range_of_f : Set.image f domain = range_f :=
by
  sorry

end range_of_f_l59_59257


namespace sum_XY_XZ_constant_l59_59447

theorem sum_XY_XZ_constant (A B C A' X : Point) (ω : Circle) 
  (h1 : inscribed_in_triangle ω A B C) 
  (h2 : exrayed_on_BC A' A B C)
  (h3 : is_segment A' A' X) 
  (h4 : does_not_intersect A' X ω)
  (Y Z : Point) 
  (h5 : tangents_from X ω Y Z)
  (h6 : on_segment BC Y)
  (h7 : on_segment BC Z)
  : XY + XZ = constant := 
sorry

end sum_XY_XZ_constant_l59_59447


namespace right_triangle_area_l59_59502

theorem right_triangle_area (a_square_area b_square_area hypotenuse_square_area : ℝ)
  (ha : a_square_area = 36) (hb : b_square_area = 64) (hc : hypotenuse_square_area = 100)
  (leg1 leg2 hypotenuse : ℝ)
  (hleg1 : leg1 * leg1 = a_square_area)
  (hleg2 : leg2 * leg2 = b_square_area)
  (hhyp : hypotenuse * hypotenuse = hypotenuse_square_area) :
  (1/2) * leg1 * leg2 = 24 :=
by
  sorry

end right_triangle_area_l59_59502


namespace fraction_to_terminating_decimal_l59_59204

theorem fraction_to_terminating_decimal :
  let frac := (63 : ℚ) / (2^3 * 5^4) in frac = 0.0126 := sorry

end fraction_to_terminating_decimal_l59_59204


namespace fair_stake_correct_l59_59070

theorem fair_stake_correct :
  let total_ways := nat.choose 18 9 in
  let favorable_ways := (nat.choose 5 2) * (nat.choose 6 3) * (nat.choose 7 4) in
  let favorable_probability := favorable_ways / total_ways in
  let unfavorable_probability := 1 - favorable_probability in
  let fair_stake := (2081 / 2431 : ℚ) * (1 / 2) - (350 / 2431 : ℚ) * 3 in
  fair_stake ≈ 2.97 :=
by
  sorry

end fair_stake_correct_l59_59070


namespace compute_expression_l59_59862

theorem compute_expression :
  (5 + 7)^2 + (5^2 + 7^2) * 2 = 292 := by
  sorry

end compute_expression_l59_59862


namespace positive_root_condition_negative_root_condition_zero_root_condition_l59_59524

variable (a b c : ℝ)

-- Condition for a positive root
theorem positive_root_condition : 
  ((a > 0 ∧ b > c) ∨ (a < 0 ∧ b < c)) ↔ (∃ x : ℝ, x > 0 ∧ a * x = b - c) :=
sorry

-- Condition for a negative root
theorem negative_root_condition : 
  ((a > 0 ∧ b < c) ∨ (a < 0 ∧ b > c)) ↔ (∃ x : ℝ, x < 0 ∧ a * x = b - c) :=
sorry

-- Condition for a root equal to zero
theorem zero_root_condition : 
  (a ≠ 0 ∧ b = c) ↔ (∃ x : ℝ, x = 0 ∧ a * x = b - c) :=
sorry

end positive_root_condition_negative_root_condition_zero_root_condition_l59_59524


namespace workByFortyPercent_l59_59292

-- Definitions based on the conditions
def totalWork : ℝ := 100.0
def productiveEmployees1 : ℝ := 20.0 / 100.0 * totalWork
def workByProductiveEmployees1 : ℝ := 80.0 / 100.0 * totalWork

-- Helper definitions to split the next 20% of most productive employees
def remainingEmployees : ℝ := 100.0 - productiveEmployees1
def nextProductiveEmployees2 : ℝ := 20.0 / 100.0 * remainingEmployees
def workByNextProductiveEmployees2 : ℝ := nextProductiveEmployees2 / remainingEmployees * (totalWork - workByProductiveEmployees1)

-- Theorem we need to prove
theorem workByFortyPercent : (workByProductiveEmployees1 + workByNextProductiveEmployees2) = 85 := by
  sorry

end workByFortyPercent_l59_59292


namespace possible_values_of_k_l59_59669

theorem possible_values_of_k (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ (k ∈ {1, 2, 3, 4, 5, 6, 8, 9}), k = ((a + b + c)^2) / (a * b * c) :=
by sorry

end possible_values_of_k_l59_59669


namespace quadratic_roots_opposite_l59_59223

theorem quadratic_roots_opposite (a : ℝ) (h : ∀ x1 x2 : ℝ, 
  (x1 + x2 = 0 ∧ x1 * x2 = a - 1) ∧
  (x1 - (-(x1)) = 0 ∧ x2 - x1 = 0)) :
  a = 0 :=
sorry

end quadratic_roots_opposite_l59_59223


namespace solve_for_m_l59_59795

theorem solve_for_m (m : ℤ) (h : (-2)^(2*m) = 2^(21 - m)) : m = 7 :=
sorry

end solve_for_m_l59_59795


namespace solve_for_x_l59_59374

-- Step d: Lean 4 statement
theorem solve_for_x : 
  (∃ x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 2)) → (∃ x : ℚ, x = 1 / 3) :=
sorry

end solve_for_x_l59_59374


namespace solve_equation_l59_59521

theorem solve_equation :
  ∀ x : ℝ, 
    (8 / (Real.sqrt (x - 10) - 10) + 
     2 / (Real.sqrt (x - 10) - 5) + 
     9 / (Real.sqrt (x - 10) + 5) + 
     16 / (Real.sqrt (x - 10) + 10) = 0)
    ↔ 
    x = 1841 / 121 ∨ x = 190 / 9 :=
by
  sorry

end solve_equation_l59_59521


namespace exists_positive_integer_N_with_leading_digits_l59_59371

noncomputable def alpha : ℝ := Real.logb 10 2

theorem exists_positive_integer_N_with_leading_digits :
  ∃ N : ℕ, N > 0 ∧ (2000^N).digits.take 12 = 200120012001.digits.take 12 := sorry

end exists_positive_integer_N_with_leading_digits_l59_59371


namespace number_of_sets_l59_59581

open Set

/-- Given that {4, 5} ⊆ M ⊆ {1, 2, 3, 4, 5}, prove that the number of such sets M is 7. -/
theorem number_of_sets (M : Set ℕ) (h1 : {4, 5} ⊆ M) (h2 : M ⊆ {1, 2, 3, 4, 5}) :
  Finset.card ({M | {4, 5} ⊆ M ∧ M ⊆ {1, 2, 3, 4, 5} }.to_finset) = 7 :=
sorry

end number_of_sets_l59_59581


namespace triangle_has_at_most_one_obtuse_angle_l59_59099

theorem triangle_has_at_most_one_obtuse_angle 
  (T : Type) [triangle T]
  (obtuse_angle : T → Prop)
  (at_most_one_obtuse_angle : ∀ t : T, ∃! θ. obtuse_angle θ) :
  ¬ (∃ t : T, ∃ θ1 θ2 : T, obtuse_angle θ1 ∧ obtuse_angle θ2) := 
by
  sorry

end triangle_has_at_most_one_obtuse_angle_l59_59099


namespace sum_lengths_AMC_l59_59489

theorem sum_lengths_AMC : 
  let length_A := 2 * (Real.sqrt 2) + 2
  let length_M := 3 + 3 + 2 * (Real.sqrt 2)
  let length_C := 3 + 3 + 2
  length_A + length_M + length_C = 13 + 4 * (Real.sqrt 2)
  := by
  sorry

end sum_lengths_AMC_l59_59489


namespace aphrodite_is_most_beautiful_l59_59610

-- Definitions for the goddesses
inductive Goddess
| aphrodite
| hera
| athena

open Goddess

-- Conditions given in the problem
def aphrodite_statement : Goddess → Prop
| aphrodite := True
| hera      := False
| athena    := False

def hera_statement : Goddess → Prop
| hera      := True
| aphrodite := False
| athena    := False

def athena_statement1 : Goddess → Prop
| aphrodite := False
| hera      := True
| athena    := False

def athena_statement2 : Goddess → Prop
| athena    := True
| hera      := False
| aphrodite := False

-- The problem's main question transformed into a proof goal
theorem aphrodite_is_most_beautiful : 
  (∀ g, (aphrodite_statement g ↔ (g = aphrodite)) ∧ 
        (hera_statement g ↔ (g = hera)) ∧ 
        (athena_statement1 g ↔ g ≠ aphrodite) ∧ 
        (athena_statement2 g ↔ (g = athena))) →
  ∀ g, g ≠ aphrodite → (aphrodite_statement g = False) :=
sorry

end aphrodite_is_most_beautiful_l59_59610


namespace inscribed_circle_radius_is_three_l59_59093

variable {A B C : Type}
variable [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variables (AB AC BC : ℝ)

def triangle_inscribed_circle_radius (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2 in
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) in
  K / s

theorem inscribed_circle_radius_is_three :
  ∀ (AB AC BC : ℝ), AB = 8 → AC = 15 → BC = 17 → triangle_inscribed_circle_radius AB AC BC = 3 :=
by
  intros AB AC BC hAB hAC hBC
  simp [hAB, hAC, hBC, triangle_inscribed_circle_radius]
  -- skipped proof
  sorry

end inscribed_circle_radius_is_three_l59_59093


namespace wire_length_correct_l59_59440

noncomputable def wire_length (V : ℝ) (d : ℝ) : ℝ :=
  let r := d / 2
  let r_cm := r / 10
  let h_cm := V / (real.pi * (r_cm ^ 2))
  h_cm / 100

theorem wire_length_correct : wire_length 11 1 = 14.01 :=
by
  sorry

end wire_length_correct_l59_59440


namespace locus_of_intersection_is_hyperbola_l59_59562

theorem locus_of_intersection_is_hyperbola 
(x y a : ℝ) (θ : ℝ) 
(hA : ∀ x, (A : ℝ × ℝ) = (a, 0))
(hA' : ∀ x, (A' : ℝ × ℝ) = (-a, 0))
(hM : ∀ x, ∃ θ, (M : ℝ × ℝ) = (a * Real.cos θ, (1/2) * a * Real.sin θ))
(hM' : ∀ x, ∃ θ, (M' : ℝ × ℝ) = (a * Real.cos θ, -(1/2) * a * Real.sin θ))
: 
(x^2 / a^2) - (y^2 / (a / 2)^2) = 1 :=
by
  sorry

end locus_of_intersection_is_hyperbola_l59_59562


namespace area_square_ABCD_is_64_l59_59488

-- Given initial definitions and conditions
def side_length_original_square : ℝ := 4
def radius_semicircle : ℝ := side_length_original_square / 2
def side_length_new_square : ℝ := side_length_original_square + 2 * radius_semicircle

-- Theorem statement proving the area of square ABCD is 64
theorem area_square_ABCD_is_64 : (side_length_new_square ^ 2 = 64) :=
by
  -- Proof goes here
  sorry

end area_square_ABCD_is_64_l59_59488


namespace no_function_exists_l59_59206

theorem no_function_exists (f : ℕ → ℕ) : ¬ (∀ x : ℕ, f(f(x)) = x + 1) := 
sorry

end no_function_exists_l59_59206


namespace log_diff_is_six_l59_59198

theorem log_diff_is_six (h1 : 256 = 4^4) (h2 : 1 / 16 = 4^(-2)) :
  Real.log 256 / Real.log 4 - Real.log (1 / 16) / Real.log 4 = 6 :=
by
  sorry

end log_diff_is_six_l59_59198


namespace extreme_values_of_f_l59_59574

noncomputable def f (a x : ℝ) : ℝ := (Real.log x) + (1 - x) / (a * x)

theorem extreme_values_of_f (a : ℝ) (h : a ≠ 0) :
  (a < 0 → ∀ x : ℝ, 0 < x → f a x ≠ real_min_or_max f a x) ∧
  (a > 0 → ∃ p : ℝ, p = 1 / a ∧ f a p = -Real.log a - 1 / a + 1) :=
  sorry

end extreme_values_of_f_l59_59574


namespace spherical_to_rectangular_coordinates_l59_59863

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 6 → θ = 7 * Real.pi / 4 → φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (3, -3, 3 * Real.sqrt 2) := by
  sorry

end spherical_to_rectangular_coordinates_l59_59863


namespace min_value_of_E_l59_59781

theorem min_value_of_E (E : ℝ → ℝ) : (∀ x : ℝ, |E x| + |x + 5| + |x - 5| = 10) → (∀ x, |E x| ≥ 0) :=
by {
  assume h,
  assume x,
  obtain hx1 | hx2 := le_or_gt x (-5),
  { have h1 : |x + 5| = 0, linarith [hx2], 
    sorry },
  { obtain hx3 | hx4 := le_or_gt x 5,
    { have h2 : |x - 5| = 0, linarith [hx3],
      sorry },
    { sorry } }
}

end min_value_of_E_l59_59781


namespace find_ordered_pair_l59_59213

theorem find_ordered_pair (x y : ℝ) 
  (h1 : x + y = (7 - x) + (7 - y)) 
  (h2 : x - y = (x - 2) + (y - 2)) : 
  (x = 5 ∧ y = 2) :=
by
  sorry

end find_ordered_pair_l59_59213


namespace relationship_alpha_beta_l59_59573

-- Definitions for the fixed points and moving point
variables (A B P : ℝ × ℝ)

-- Definition of the propositions
def proposition_alpha (A B P : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, ∀ P : ℝ × ℝ, k = dist P A + dist P B

def proposition_beta (A B P : ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ b < a ∧ ellipse A B a b

-- The logical relationship between the propositions
theorem relationship_alpha_beta :
  (∀ A B P, proposition_beta A B P → proposition_alpha A B P) ∧
  ¬(∀ A B P, proposition_alpha A B P → proposition_beta A B P) :=
sorry

end relationship_alpha_beta_l59_59573


namespace final_sum_l59_59963

def f (x : Real) : Real :=
  (2 * (x - 1/2)^4 + (x - 1/2)^2 * Real.sin (x - 1/2) + 4) / ((x - 1/2)^4 + 2)

theorem final_sum (n : ℕ) (h : n = 2017) :
  (Finset.range (n - 1)).sum (λ k, f ((k + 1 : ℝ) / n)) = 4032 :=
sorry

end final_sum_l59_59963


namespace number_of_odd_digits_in_base_6_of_523_l59_59496

theorem number_of_odd_digits_in_base_6_of_523 : let base_6_rep := nat_to_base 6 523 in count_odd_digits base_6_rep = 2 :=
by
  sorry

end number_of_odd_digits_in_base_6_of_523_l59_59496


namespace meal_cost_l59_59514

theorem meal_cost (M : ℝ) (h1 : 3 * M + 15 = 45) : M = 10 :=
by
  sorry

end meal_cost_l59_59514


namespace probability_two_flies_swept_l59_59008

/-- Defining the positions of flies on the clock -/
inductive positions : Type
| twelve   | three   | six   | nine

/-- Probability that the minute hand sweeps exactly two specific positions after 20 minutes -/
theorem probability_two_flies_swept (flies : list positions) (time : ℕ) :
  (flies = [positions.twelve, positions.three, positions.six, positions.nine]) →
  (time = 20) →
  (probability_sweeps_two_flies flies time = 1 / 3) := sorry

end probability_two_flies_swept_l59_59008


namespace equal_midlines_implies_perpendicular_diagonals_and_vice_versa_perpendicular_midlines_implies_equal_diagonals_and_vice_versa_l59_59439

variables {A B C D : Type} [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D]

/-- First theorem: if the midlines of a quadrilateral are equal, then its diagonals are perpendicular, and conversely. -/
theorem equal_midlines_implies_perpendicular_diagonals_and_vice_versa (ABCD : Quadrilateral A B C D) :
  (equal_midlines ABCD ↔ perpendicular_diagonals ABCD) :=
sorry

/-- Second theorem: if the midlines of a quadrilateral are perpendicular, then its diagonals are equal, and conversely. -/
theorem perpendicular_midlines_implies_equal_diagonals_and_vice_versa (ABCD : Quadrilateral A B C D) :
  (perpendicular_midlines ABCD ↔ equal_diagonals ABCD) :=
sorry

end equal_midlines_implies_perpendicular_diagonals_and_vice_versa_perpendicular_midlines_implies_equal_diagonals_and_vice_versa_l59_59439


namespace chosen_number_is_30_l59_59473

theorem chosen_number_is_30 (x : ℤ) 
  (h1 : 8 * x - 138 = 102) : x = 30 := 
sorry

end chosen_number_is_30_l59_59473


namespace productive_employees_work_l59_59295

theorem productive_employees_work (total_work : ℝ) (P : ℝ) (Q : ℝ)
  (h1 : P = 0.2) (h2 : Q = 0.8) :
  (0.4 * total_work) = (0.85 * total_work) :=
by
  -- Given the conditions: 20% of employees do 80% of the work
  have h3 : 0.2 * total_work = 0.8 * total_work, from sorry
  -- 40% of the most productive employees perform 85% of the work
  exact sorry

end productive_employees_work_l59_59295


namespace circle_symmetric_line_l59_59738

theorem circle_symmetric_line (a b : ℝ) (h : a < 2) (hb : b = -2) : a + b < 0 := by
  sorry

end circle_symmetric_line_l59_59738


namespace remainder_n_l59_59754

-- Definitions for the conditions
/-- m is a positive integer leaving a remainder of 2 when divided by 6 -/
def m (m : ℕ) : Prop := m % 6 = 2

/-- The remainder when m - n is divided by 6 is 5 -/
def mn_remainder (m n : ℕ) : Prop := (m - n) % 6 = 5

-- Theorem statement
theorem remainder_n (m n : ℕ) (h1 : m % 6 = 2) (h2 : (m - n) % 6 = 5) (h3 : m > n) :
  n % 6 = 4 :=
by
  sorry

end remainder_n_l59_59754


namespace shaded_to_unshaded_ratio_l59_59637

-- Define the lengths of the diameters
def XZ : ℝ := 12
def ZY : ℝ := 8
def XY : ℝ := XZ + ZY

-- Radii of the circles
def r1 := XZ / 2
def r2 := ZY / 2
def r3 := XY / 2

-- Areas of the circles
def Area1 := π * r1^2
def Area2 := π * r2^2
def Area3 := π * r3^2

-- Unshaded area
def Area_unshaded := Area1 + Area2

-- Shaded area
def Area_shaded := Area3 - Area_unshaded

-- Ratio of shaded to unshaded area
def ratio : ℝ := Area_shaded / Area_unshaded

theorem shaded_to_unshaded_ratio : ratio = 12 / 13 := 
by { sorry }

end shaded_to_unshaded_ratio_l59_59637


namespace january_first_2022_day_of_week_l59_59587

theorem january_first_2022_day_of_week : 
  ∀ (d : ℕ), (d % 7 = 0) ∧ 365 % 7 = 1 → d + 1 % 7 = 6
:=
by
  intros d day_0 days_cycle
  sorry

end january_first_2022_day_of_week_l59_59587


namespace solve_abs_eq_l59_59095

theorem solve_abs_eq (x : ℝ) (h : |x + 3| = |x - 5|) : x = 1 := by
  -- Case analysis of |x + 3| = |x - 5|
  cases abs_eq_iff_eq_or_eq_neg.mp h
  case inl => 
    -- x + 3 = x - 5 case
    exfalso
    -- This should be a contradiction since it simplifies to 3 = -5
    sorry
  case inr => 
    -- x + 3 = -(x - 5) case
    have : 2 * x = 2 := by
      -- Simplify this case to x = 1
      sorry
    -- Conclude x = 1
    exact eq_of_mul_eq_mul_left (by norm_num) this

end solve_abs_eq_l59_59095


namespace iterated_function_value_l59_59964

def f (x : ℝ) : ℝ := x / (Real.sqrt (1 + x^2))

def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | (n + 1) => f (f_iter n x)

theorem iterated_function_value :
  f_iter 99 1 = 1 / 10 :=
by
  sorry

end iterated_function_value_l59_59964


namespace common_perpendiculars_of_skew_lines_l59_59559

-- Definitions using the conditions from the problem
def is_skew (a b : Line) : Prop :=
  ¬ (∃ p, p ∈ a ∧ p ∈ b)

def is_perpendicular (a b : Line) : Prop :=
  ∀ (p : Point), p ∈ a → ∀ (q : Point), q ∈ b → orthogonal p q

-- Lean statement formalizing the problem
theorem common_perpendiculars_of_skew_lines
  (a b c a' b' c' : Line)
  (h1 : is_skew a b)
  (h2 : is_skew b c)
  (h3 : is_skew c a)
  (h4 : is_perpendicular a' b)
  (h5 : is_perpendicular a' c)
  (h6 : is_perpendicular b' c)
  (h7 : is_perpendicular b' a)
  (h8 : is_perpendicular c' a)
  (h9 : is_perpendicular c' b) :
  ∃ d : Line, is_perpendicular d a ∧ is_perpendicular d b ∧ is_perpendicular d c :=
sorry

end common_perpendiculars_of_skew_lines_l59_59559


namespace number_of_matches_in_second_set_l59_59736

theorem number_of_matches_in_second_set (x : ℕ) : (22 * 28) + (x * 15) = 35 * 23.17142857142857 → x = 13 := 
by
  sorry

end number_of_matches_in_second_set_l59_59736


namespace hexagon_shaded_fraction_l59_59383

/-
Given a regular hexagon that is divided into smaller equilateral triangles,
prove that the fraction of the area of the hexagon that is shaded is 1/2, given
the following conditions:
1. Inside the hexagon, there is a shaded large equilateral triangle.
2. There are twelve small equilateral triangles outside this large equilateral triangle.
-/
theorem hexagon_shaded_fraction (A : Type) [regular_hexagon A] [divisible_into_triangles A] :
  shaded_fraction A = 1/2 := 
sorry

end hexagon_shaded_fraction_l59_59383


namespace incorrect_statement_C_l59_59503

def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem incorrect_statement_C : g 2 ≠ 5 :=
by
  dsimp [g]
  have h : (2 * 2 + 3) / (2 - 2) = 7 / 0 := rfl
  have h_undef : ¬∃ (y : ℝ), 7 / 0 = y := by
    intro y
    contradiction
  contradiction

end incorrect_statement_C_l59_59503


namespace newton_method_approximation_bisection_method_approximation_l59_59358

noncomputable def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 3
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 + 4*x + 3

theorem newton_method_approximation :
  let x0 := -1
  let x1 := x0 - f x0 / f' x0
  let x2 := x1 - f x1 / f' x1
  x2 = -7 / 5 := sorry

theorem bisection_method_approximation :
  let a := -2
  let b := -1
  let midpoint1 := (a + b) / 2
  let new_a := if f midpoint1 < 0 then midpoint1 else a
  let new_b := if f midpoint1 < 0 then b else midpoint1
  let midpoint2 := (new_a + new_b) / 2
  midpoint2 = -11 / 8 := sorry

end newton_method_approximation_bisection_method_approximation_l59_59358


namespace divide_oranges_into_pieces_l59_59319

-- Definitions for conditions
def oranges : Nat := 80
def friends : Nat := 200
def pieces_per_friend : Nat := 4

-- Theorem stating the problem and the answer
theorem divide_oranges_into_pieces :
    (oranges > 0) → (friends > 0) → (pieces_per_friend > 0) →
    ((friends * pieces_per_friend) / oranges = 10) :=
by
  intros
  sorry

end divide_oranges_into_pieces_l59_59319


namespace square_area_is_360_l59_59049

variable (x : ℝ) -- Length of each segment

-- Given conditions
def side_segments := 6
def shaded_squares := 15
def shaded_area := 75.0
def segment_area (x : ℝ) := 15 * (x^2 / 2)

-- The goal is to prove that the area of the original square is 360
theorem square_area_is_360 (h : segment_area x = shaded_area) : (side_segments * x)^2 = 360 := by
  sorry

end square_area_is_360_l59_59049


namespace monotonically_increasing_iff_a_in_range_l59_59388

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 0 then x^3 - a * x^2 + a else 2^((2 - a) * x) + 1 / 2

theorem monotonically_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (0 ≤ a ∧ a ≤ 3 / 2) := 
sorry

end monotonically_increasing_iff_a_in_range_l59_59388


namespace trapezoid_dimension_l59_59126

theorem trapezoid_dimension:
  ∀ (a b: ℕ), a = 9 → b = 16 → y = sqrt (a * b) / 2 → y = 6 :=
by
  intros a b ha hb hy
  rw [ha, hb] at hy
  sorry

end trapezoid_dimension_l59_59126


namespace product_of_numbers_l59_59759

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : x * y = 200 :=
sorry

end product_of_numbers_l59_59759


namespace cars_to_sell_l59_59830

theorem cars_to_sell (n : ℕ) 
  (h1 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → ∃ m, m = 3)
  (h2 : ∀ c, c ∈ {c' : ℕ | c' ≤ n} → c ∈ {c' : ℕ | c' < 3})
  (h3 : 15 * 3 = 45)
  (h4 : ∀ n, n * 3 = 45 → n = 15):
  n = 15 := 
  by
    have n_eq: n * 3 = 45 := sorry
    exact h4 n n_eq

end cars_to_sell_l59_59830


namespace find_n_18_l59_59847

theorem find_n_18 (n : ℕ) (a : ℕ → ℕ) (m : ℕ) (h : n = (∑ i in finset.range(m), 10^i * a (m-i-1))) 
  (h_eq : n = ∏ i in finset.range(m), (a i + 1)) : n = 18 :=
by 
  sorry -- proof required here

end find_n_18_l59_59847
